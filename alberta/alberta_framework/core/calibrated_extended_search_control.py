# mypy: disable-error-code="attr-defined,call-arg"
"""Calibrated fixed-budget search over primitive and option models.

This standalone module is an opt-in L0 mechanism.  It owns a fixed real-anchor
bank and a small tabular extended-action value surface; it does not mutate
STOMP, OaK, Prototype, a behavior action, or any external learner/model.  Four
static modes share the same state and work budget:

``model_free_extended_q``
    Replays the most recent causally resolved real target for primitive and
    option extended actions.
``primitive_model``
    Searches only one-step primitive-model targets.
``option_model``
    Searches only naturally completed option-model targets.
``combined``
    Ranks the primitive/option union under one total backup budget.  It never
    allocates one budget per family.

Candidate order is kind, semantic index, then anchor index.  Consequently an
exact score tie is stable across eager/JIT execution and prefers primitive
before option, lower semantic index before higher, and finally lower anchor.

The transaction is predict-before-update.  ``arm`` is called after behavior
selection and freezes the decision identity, real anchor, option semantics,
representation/source identities, learner/model revisions, Q values, model
predictions, calibrated factors, and the complete ``B``-attempt schedule.
``observe`` can commit that schedule only for the exact next primitive
transition or a natural option completion.  Truncations/censoring close the
pending arm without calibration, support, reachability, or value updates.

Priority is deliberately noncompensating::

    value-change LCB
      * real-anchor reachability LCB
      * (1 - normalized model-error UCB)
      * support shrinkage

Every factor has its own evidence and validity gate.  Unavailable evidence is
ineligible rather than zero-filled and compensated by another large factor.
Reachability is not current-anchor availability: it is a Bernoulli estimate
from exact future observations revisiting a fixed, source-bound real anchor.

All state is fixed-shape, all state records are frozen chex dataclasses, and
the planner consumes no RNG.  The mechanism makes no efficacy, promotion,
WP7-completion, Alberta-Plan-completion, or L3 claim.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from typing import Any, ClassVar, cast

import chex
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import Bool, Float, Int, UInt

CALIBRATED_EXTENDED_SEARCH_CONFIG_SCHEMA = (
    "alberta.calibrated-extended-search-control.config.v1"
)
CALIBRATED_EXTENDED_SEARCH_CHECKPOINT_SCHEMA = (
    "alberta.calibrated-extended-search-control.state.v1"
)
CALIBRATED_EXTENDED_SEARCH_RANKING_SEMANTICS = (
    "v1-priority-desc-kind-primitive-before-option-semantic-index-anchor-index"
)
CALIBRATED_EXTENDED_SEARCH_MECHANISM_STATUS = "l0_development_mechanism_only"
CALIBRATED_EXTENDED_SEARCH_SCIENTIFIC_PROMOTION_ALLOWED = False
CALIBRATED_EXTENDED_SEARCH_POLICY_AUTHORITY = False

SEARCH_MODE_MODEL_FREE_EXTENDED_Q = "model_free_extended_q"
SEARCH_MODE_PRIMITIVE_MODEL = "primitive_model"
SEARCH_MODE_OPTION_MODEL = "option_model"
SEARCH_MODE_COMBINED = "combined"
SEARCH_MODES = (
    SEARCH_MODE_MODEL_FREE_EXTENDED_Q,
    SEARCH_MODE_PRIMITIVE_MODEL,
    SEARCH_MODE_OPTION_MODEL,
    SEARCH_MODE_COMBINED,
)

CANDIDATE_KIND_PRIMITIVE = 0
CANDIDATE_KIND_OPTION = 1

_CONFIG_TYPE = "CalibratedExtendedSearchControlConfig"
_OPTION_DESCRIPTOR_WIDTH = 4
_INT32_MAX = 2**31 - 1
_MAX_ANCHORS = 1_024
_MAX_EXTENDED_ACTIONS = 4_096
_MAX_CANDIDATES = 262_144
_MAX_BACKUP_BUDGET = 4_096


def _positive_int(value: object, *, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive exact Python int")
    return value


def _finite_float(
    value: object,
    *,
    name: str,
    positive: bool = False,
    lower: float | None = None,
    upper: float | None = None,
) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite exact Python float")
    resolved = value
    represented = float(jnp.asarray(resolved, dtype=jnp.float32))
    if not math.isfinite(represented):
        raise ValueError(f"{name} must remain finite in float32")
    if positive and represented <= 0.0:
        raise ValueError(f"{name} must remain positive in float32")
    if lower is not None and represented < lower:
        raise ValueError(f"{name} must be at least {lower}")
    if upper is not None and represented > upper:
        raise ValueError(f"{name} must be at most {upper}")
    return resolved


def _require_array(
    value: Array,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: Any,
) -> Array:
    array = jnp.asarray(value)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if array.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {array.dtype}")
    return array


def _int32_scalar(value: int | Array, *, name: str) -> Array:
    if type(value) is int:
        if not -(2**31) <= value <= _INT32_MAX:
            raise ValueError(f"{name} must be signed-int32 compatible")
        array = jnp.asarray(value, dtype=jnp.int32)
    else:
        array = jnp.asarray(value)
    if array.shape != ():
        raise ValueError(f"{name} must be scalar, got {array.shape}")
    if array.dtype != jnp.int32:
        raise TypeError(f"{name} must have dtype int32, got {array.dtype}")
    return array


def _float32_scalar(value: float | Array, *, name: str) -> Array:
    if type(value) is float:
        array = jnp.asarray(value, dtype=jnp.float32)
    else:
        array = jnp.asarray(value)
    if array.shape != ():
        raise ValueError(f"{name} must be scalar, got {array.shape}")
    if array.dtype != jnp.float32:
        raise TypeError(f"{name} must have dtype float32, got {array.dtype}")
    return array


def _bool_scalar(value: bool | Array, *, name: str) -> Array:
    if type(value) is bool:
        array = jnp.asarray(value, dtype=jnp.bool_)
    else:
        array = jnp.asarray(value)
    if array.shape != ():
        raise ValueError(f"{name} must be scalar, got {array.shape}")
    if array.dtype != jnp.bool_:
        raise TypeError(f"{name} must have dtype bool, got {array.dtype}")
    return array


def _checksum_arrays(arrays: tuple[Array, ...], *, seed: Array) -> Array:
    """Return a deterministic two-word checksum usable under JIT."""

    acc0 = seed[0] ^ jnp.uint32(0x9E3779B9)
    acc1 = seed[1] ^ jnp.uint32(0x85EBCA6B)
    offset = 1
    for value in arrays:
        array = jnp.asarray(value)
        if array.dtype == jnp.float32:
            words = jax.lax.bitcast_convert_type(array, jnp.uint32).reshape((-1,))
        elif array.dtype in (jnp.int32, jnp.uint32):
            words = array.astype(jnp.uint32).reshape((-1,))
        else:
            words = array.astype(jnp.uint32).reshape((-1,))
        if words.shape[0] == 0:
            continue
        indices = jnp.arange(offset, offset + words.shape[0], dtype=jnp.uint32)
        acc0 = acc0 + jnp.sum(words * (indices * jnp.uint32(0x27D4EB2D) + 1))
        acc1 = acc1 ^ jnp.bitwise_xor.reduce(
            words ^ (indices * jnp.uint32(0x165667B1))
        )
        offset += words.shape[0]
    return jnp.stack((acc0, acc1), axis=0).astype(jnp.uint32)


@dataclasses.dataclass(frozen=True, slots=True)
class CalibratedExtendedSearchControlConfig:
    """Static search mode, capacities, calibration, and update contract."""

    mode: str
    observation_dim: int
    anchor_capacity: int
    n_primitive_actions: int
    n_options: int
    backup_budget: int
    calibration_evidence_floor: int = 4
    model_support_floor: int = 4
    confidence_scale: float = 1.0
    support_prior: float = 4.0
    model_error_scale: float = 1.0
    backup_step_size: float = 0.1
    min_value_change_lcb: float = 0.0
    min_reachability_lcb: float = 0.0
    min_model_reliability: float = 0.0
    max_observations: int = 1_000_000

    SCHEMA_VERSION: ClassVar[str] = CALIBRATED_EXTENDED_SEARCH_CONFIG_SCHEMA

    def __post_init__(self) -> None:
        if type(self.mode) is not str or self.mode not in SEARCH_MODES:
            raise ValueError(f"mode must be exactly one of {SEARCH_MODES}")
        for name in (
            "observation_dim",
            "anchor_capacity",
            "n_primitive_actions",
            "n_options",
            "backup_budget",
            "calibration_evidence_floor",
            "model_support_floor",
            "max_observations",
        ):
            _positive_int(getattr(self, name), name=name)
        if self.anchor_capacity > _MAX_ANCHORS:
            raise ValueError("anchor_capacity exceeds the fixed 1024-anchor ceiling")
        if self.n_extended_actions > _MAX_EXTENDED_ACTIONS:
            raise ValueError("extended-action capacity exceeds 4096")
        if self.candidate_capacity > _MAX_CANDIDATES:
            raise ValueError("candidate capacity exceeds the fixed 262144-cell ceiling")
        if self.backup_budget > _MAX_BACKUP_BUDGET:
            raise ValueError("backup_budget exceeds the fixed 4096-attempt ceiling")
        if self.max_observations > _INT32_MAX - self.backup_budget - 1:
            raise ValueError("max_observations leaves no signed-int32 backup headroom")
        if self.calibration_evidence_floor < 2:
            raise ValueError("calibration_evidence_floor must be at least two")
        if max(self.calibration_evidence_floor, self.model_support_floor) > (
            self.max_observations
        ):
            raise ValueError("evidence floors must not exceed max_observations")
        for name in (
            "confidence_scale",
            "support_prior",
            "model_error_scale",
        ):
            _finite_float(getattr(self, name), name=name, positive=True)
        _finite_float(
            self.backup_step_size,
            name="backup_step_size",
            positive=True,
            upper=1.0,
        )
        _finite_float(
            self.min_value_change_lcb,
            name="min_value_change_lcb",
            lower=0.0,
        )
        for name in ("min_reachability_lcb", "min_model_reliability"):
            _finite_float(getattr(self, name), name=name, lower=0.0, upper=1.0)

    @property
    def n_extended_actions(self) -> int:
        return self.n_primitive_actions + self.n_options

    @property
    def candidate_capacity(self) -> int:
        return self.anchor_capacity * self.n_extended_actions

    def to_config(self) -> dict[str, object]:
        return {
            "schema": CALIBRATED_EXTENDED_SEARCH_CONFIG_SCHEMA,
            "type": _CONFIG_TYPE,
            "mechanism_status": CALIBRATED_EXTENDED_SEARCH_MECHANISM_STATUS,
            "scientific_promotion_allowed": False,
            "policy_authority": False,
            **dataclasses.asdict(self),
        }

    @classmethod
    def from_config(cls, value: object) -> CalibratedExtendedSearchControlConfig:
        if type(value) is not dict:
            raise ValueError("calibrated extended-search config must be an exact dict")
        payload = cast(dict[object, object], value)
        field_names = {field.name for field in dataclasses.fields(cls)}
        expected = {
            "schema",
            "type",
            "mechanism_status",
            "scientific_promotion_allowed",
            "policy_authority",
            *field_names,
        }
        if set(payload) != expected:
            raise ValueError("calibrated extended-search config fields differ from v1")
        raw = dict(payload)
        if raw.pop("schema") != CALIBRATED_EXTENDED_SEARCH_CONFIG_SCHEMA:
            raise ValueError("calibrated extended-search config schema differs")
        if raw.pop("type") != _CONFIG_TYPE:
            raise ValueError("calibrated extended-search config type differs")
        if raw.pop("mechanism_status") != CALIBRATED_EXTENDED_SEARCH_MECHANISM_STATUS:
            raise ValueError("calibrated extended-search mechanism status differs")
        if raw.pop("scientific_promotion_allowed") is not False:
            raise ValueError("calibrated extended search cannot claim promotion")
        if raw.pop("policy_authority") is not False:
            raise ValueError("calibrated extended search cannot claim policy authority")
        for name in (
            "observation_dim",
            "anchor_capacity",
            "n_primitive_actions",
            "n_options",
            "backup_budget",
            "calibration_evidence_floor",
            "model_support_floor",
            "max_observations",
        ):
            if type(raw[name]) is not int:
                raise ValueError(f"serialized {name} must be an exact JSON integer")
        for name in (
            "confidence_scale",
            "support_prior",
            "model_error_scale",
            "backup_step_size",
            "min_value_change_lcb",
            "min_reachability_lcb",
            "min_model_reliability",
        ):
            if type(raw[name]) is not float:
                raise ValueError(f"serialized {name} must be an exact JSON float")
        return cls(**cast(dict[str, Any], raw))


@chex.dataclass(frozen=True)
class CalibratedExtendedSearchControlState:
    """Fixed-shape calibrated planner state, including one pending arm."""

    representation_generation: Int[Array, ""]
    source_digest: UInt[Array, " 2"]
    canonical_digest: UInt[Array, " 2"]
    option_descriptors: Int[Array, "n_options 4"]
    option_generations: Int[Array, " n_options"]
    option_universe_digest: UInt[Array, " 2"]
    anchor_bank: Float[Array, "anchor_capacity observation_dim"]
    anchor_active: Bool[Array, " anchor_capacity"]
    q_values: Float[Array, "anchor_capacity n_extended_actions"]
    state_revision: Int[Array, ""]
    learner_revision: Int[Array, ""]
    primitive_model_revision: Int[Array, ""]
    option_model_revision: Int[Array, ""]
    has_last_decision: Bool[Array, ""]
    last_decision_id: UInt[Array, " 4"]
    last_realized_targets: Float[Array, " candidate_capacity"]
    last_target_available: Bool[Array, " candidate_capacity"]
    value_change_counts: Int[Array, " candidate_capacity"]
    value_change_means: Float[Array, " candidate_capacity"]
    value_change_m2: Float[Array, " candidate_capacity"]
    model_error_counts: Int[Array, " candidate_capacity"]
    model_error_means: Float[Array, " candidate_capacity"]
    model_error_m2: Float[Array, " candidate_capacity"]
    support_counts: Int[Array, " candidate_capacity"]
    anchor_revisit_trials: Int[Array, " anchor_capacity"]
    anchor_revisit_successes: Int[Array, " anchor_capacity"]
    pending: Bool[Array, ""]
    pending_decision_id: UInt[Array, " 4"]
    pending_anchor_observation: Float[Array, " observation_dim"]
    pending_executed_kind: Int[Array, ""]
    pending_executed_index: Int[Array, ""]
    pending_anchor_index: Int[Array, ""]
    pending_option_generation: Int[Array, ""]
    pending_representation_generation: Int[Array, ""]
    pending_source_digest: UInt[Array, " 2"]
    pending_option_universe_digest: UInt[Array, " 2"]
    pending_state_revision: Int[Array, ""]
    pending_learner_revision: Int[Array, ""]
    pending_primitive_model_revision: Int[Array, ""]
    pending_option_model_revision: Int[Array, ""]
    pending_average_reward: Float[Array, ""]
    pending_frozen_q_values: Float[Array, "anchor_capacity n_extended_actions"]
    pending_frozen_candidate_q: Float[Array, " candidate_capacity"]
    pending_candidate_targets: Float[Array, " candidate_capacity"]
    pending_target_available: Bool[Array, " candidate_capacity"]
    pending_value_change_lcb: Float[Array, " candidate_capacity"]
    pending_reachability_lcb: Float[Array, " candidate_capacity"]
    pending_model_error_ucb: Float[Array, " candidate_capacity"]
    pending_support_shrinkage: Float[Array, " candidate_capacity"]
    pending_priorities: Float[Array, " candidate_capacity"]
    pending_candidate_eligible: Bool[Array, " candidate_capacity"]
    pending_external_support: Int[Array, " candidate_capacity"]
    pending_selected_candidate_indices: Int[Array, " backup_budget"]
    pending_selected_kinds: Int[Array, " backup_budget"]
    pending_selected_semantic_indices: Int[Array, " backup_budget"]
    pending_selected_anchor_indices: Int[Array, " backup_budget"]
    pending_selected_targets: Float[Array, " backup_budget"]
    pending_selected_priorities: Float[Array, " backup_budget"]
    pending_selected_valid: Bool[Array, " backup_budget"]
    pending_cache_digest: UInt[Array, " 2"]


@chex.dataclass(frozen=True)
class CalibratedExtendedSearchArmDiagnostics:
    """Frozen factors and exact one-budget schedule from ``arm``."""

    state_valid: Bool[Array, ""]
    live_binding_matches: Bool[Array, ""]
    decision_anchor_matches: Bool[Array, ""]
    inputs_finite: Bool[Array, ""]
    derived_values_valid: Bool[Array, ""]
    model_contract_valid: Bool[Array, ""]
    capacity_available: Bool[Array, ""]
    transaction_valid: Bool[Array, ""]
    candidate_targets: Float[Array, " candidate_capacity"]
    target_available: Bool[Array, " candidate_capacity"]
    value_change_lcb: Float[Array, " candidate_capacity"]
    reachability_lcb: Float[Array, " candidate_capacity"]
    model_error_ucb: Float[Array, " candidate_capacity"]
    support_shrinkage: Float[Array, " candidate_capacity"]
    priorities: Float[Array, " candidate_capacity"]
    candidate_eligible: Bool[Array, " candidate_capacity"]
    selected_candidate_indices: Int[Array, " backup_budget"]
    selected_kinds: Int[Array, " backup_budget"]
    selected_semantic_indices: Int[Array, " backup_budget"]
    selected_anchor_indices: Int[Array, " backup_budget"]
    selected_targets: Float[Array, " backup_budget"]
    selected_priorities: Float[Array, " backup_budget"]
    selected_valid: Bool[Array, " backup_budget"]
    backup_attempt_count: Int[Array, ""]


@chex.dataclass(frozen=True)
class CalibratedExtendedSearchArmResult:
    state: CalibratedExtendedSearchControlState
    diagnostics: CalibratedExtendedSearchArmDiagnostics


@chex.dataclass(frozen=True)
class CalibratedExtendedSearchObserveDiagnostics:
    """Resolution, calibration, and learner-isolation facts."""

    state_valid: Bool[Array, ""]
    pending_cache_valid: Bool[Array, ""]
    binding_matches: Bool[Array, ""]
    resolution_structure_valid: Bool[Array, ""]
    future_anchor_evidence_valid: Bool[Array, ""]
    inputs_finite: Bool[Array, ""]
    derived_values_valid: Bool[Array, ""]
    capacity_available: Bool[Array, ""]
    transaction_valid: Bool[Array, ""]
    natural_resolution: Bool[Array, ""]
    censored_resolution: Bool[Array, ""]
    transaction_applied: Bool[Array, ""]
    resolved_candidate_index: Int[Array, ""]
    realized_differential_target: Float[Array, ""]
    frozen_model_target: Float[Array, ""]
    realized_value_change: Float[Array, ""]
    normalized_model_error: Float[Array, ""]
    calibration_updated: Bool[Array, ""]
    reachability_updated: Bool[Array, ""]
    backup_attempt_count: Int[Array, ""]
    selected_candidate_indices: Int[Array, " backup_budget"]
    selected_kinds: Int[Array, " backup_budget"]
    selected_semantic_indices: Int[Array, " backup_budget"]
    selected_anchor_indices: Int[Array, " backup_budget"]
    td_errors: Float[Array, " backup_budget"]
    learner_updates_applied: Bool[Array, " backup_budget"]
    learner_update_count: Int[Array, ""]
    rng_draw_count: Int[Array, ""]


@chex.dataclass(frozen=True)
class CalibratedExtendedSearchObserveResult:
    state: CalibratedExtendedSearchControlState
    diagnostics: CalibratedExtendedSearchObserveDiagnostics


@dataclasses.dataclass(frozen=True, slots=True)
class CalibratedExtendedSearchControlResourceBudget:
    """Exact fixed allocations and logical work maxima for version 1."""

    anchor_capacity: int
    n_primitive_actions: int
    n_options: int
    candidate_capacity: int
    backup_budget: int
    pending_arm_slots: int
    persistent_logical_scalars: int
    persistent_state_nbytes: int
    candidate_predictions_per_ranking: int
    candidate_evaluations_per_ranking: int
    model_transition_probability_cells_per_ranking: int
    max_candidate_comparisons_per_ranking: int
    anchor_identity_comparisons_per_state_validation: int
    backup_attempts_per_committed_observation: int
    max_learner_updates_per_committed_observation: int
    arm_diagnostic_payload_bytes: int
    observe_diagnostic_payload_bytes: int
    max_diagnostic_payload_bytes_per_call: int
    random_generator_calls_at_init: int
    random_generator_calls_per_arm: int
    random_generator_calls_per_observe: int
    random_draws_total: int
    persistent_state_growth_per_observation_bytes: int
    policy_dispatches_per_observation: int
    external_model_updates_per_observation: int
    scientific_promotion_allowed: bool
    policy_authority: bool
    checkpoint_schema: str
    ranking_semantics: str

    def to_config(self) -> dict[str, object]:
        return dataclasses.asdict(self)


def _state_resource_shape(config: CalibratedExtendedSearchControlConfig) -> tuple[int, int]:
    """Return exact logical scalar and byte counts for the state dataclass."""

    m = config.anchor_capacity
    d = config.observation_dim
    n = config.n_options
    a = config.n_extended_actions
    c = config.candidate_capacity
    b = config.backup_budget
    # (cell count, bytes per cell), in exact state-field order/grouping.
    groups = (
        (1, 4),  # representation generation
        (2 + 2, 4),  # source and canonical digests
        (n * _OPTION_DESCRIPTOR_WIDTH + n + 2, 4),
        (m * d, 4),
        (m, 1),
        (m * a, 4),
        (4, 4),  # state/learner/two model revisions
        (1, 1),
        (4, 4),
        (c, 4),
        (c, 1),
        (7 * c, 4),  # two Welford triples plus support count
        (2 * m, 4),
        (1, 1),
        (4, 4),
        (d, 4),
        (4, 4),  # kind/index/anchor/option generation
        (1 + 2 + 2 + 4 + 1, 4),  # generation, digests, revisions, average reward
        (m * a, 4),
        (2 * c, 4),  # candidate Q and targets
        (c, 1),
        (5 * c, 4),  # four factors and priority
        (c, 1),
        (c, 4),
        (4 * b, 4),
        (2 * b, 4),
        (b, 1),
        (2, 4),
    )
    logical = sum(cells for cells, _ in groups)
    nbytes = sum(cells * width for cells, width in groups)
    return logical, nbytes


class CalibratedExtendedSearchControl:
    """Fixed-capacity calibrated extended-action search controller."""

    def __init__(self, config: CalibratedExtendedSearchControlConfig) -> None:
        if type(config) is not CalibratedExtendedSearchControlConfig:
            raise TypeError("config must be an exact CalibratedExtendedSearchControlConfig")
        self._config = config
        canonical = json.dumps(config.to_config(), sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(canonical.encode("utf-8")).digest()
        self._config_seed = jnp.asarray(
            (
                int.from_bytes(digest[:4], "little"),
                int.from_bytes(digest[4:8], "little"),
            ),
            dtype=jnp.uint32,
        )
        m = config.anchor_capacity
        k = config.n_primitive_actions
        n = config.n_options
        self._candidate_anchor_indices = jnp.tile(
            jnp.arange(m, dtype=jnp.int32), k + n
        )
        self._candidate_extended_indices = jnp.repeat(
            jnp.arange(k + n, dtype=jnp.int32), m
        )
        self._candidate_kinds = jnp.where(
            self._candidate_extended_indices < k,
            CANDIDATE_KIND_PRIMITIVE,
            CANDIDATE_KIND_OPTION,
        ).astype(jnp.int32)
        self._candidate_semantic_indices = jnp.where(
            self._candidate_kinds == CANDIDATE_KIND_PRIMITIVE,
            self._candidate_extended_indices,
            self._candidate_extended_indices - k,
        ).astype(jnp.int32)

    @property
    def config(self) -> CalibratedExtendedSearchControlConfig:
        return self._config

    def to_config(self) -> dict[str, object]:
        return self._config.to_config()

    @classmethod
    def from_config(cls, value: object) -> CalibratedExtendedSearchControl:
        return cls(CalibratedExtendedSearchControlConfig.from_config(value))

    @property
    def resource_budget(self) -> CalibratedExtendedSearchControlResourceBudget:
        logical, nbytes = _state_resource_shape(self._config)
        c = self._config.candidate_capacity
        b = self._config.backup_budget
        arm_diagnostic_bytes = 12 + 26 * c + 25 * b
        observe_diagnostic_bytes = 46 + 21 * b
        return CalibratedExtendedSearchControlResourceBudget(
            anchor_capacity=self._config.anchor_capacity,
            n_primitive_actions=self._config.n_primitive_actions,
            n_options=self._config.n_options,
            candidate_capacity=c,
            backup_budget=b,
            pending_arm_slots=1,
            persistent_logical_scalars=logical,
            persistent_state_nbytes=nbytes,
            candidate_predictions_per_ranking=c,
            candidate_evaluations_per_ranking=c,
            model_transition_probability_cells_per_ranking=(
                c * self._config.anchor_capacity
            ),
            max_candidate_comparisons_per_ranking=b * c,
            anchor_identity_comparisons_per_state_validation=(
                self._config.anchor_capacity**2
            ),
            backup_attempts_per_committed_observation=b,
            max_learner_updates_per_committed_observation=b,
            arm_diagnostic_payload_bytes=arm_diagnostic_bytes,
            observe_diagnostic_payload_bytes=observe_diagnostic_bytes,
            max_diagnostic_payload_bytes_per_call=max(
                arm_diagnostic_bytes, observe_diagnostic_bytes
            ),
            random_generator_calls_at_init=0,
            random_generator_calls_per_arm=0,
            random_generator_calls_per_observe=0,
            random_draws_total=0,
            persistent_state_growth_per_observation_bytes=0,
            policy_dispatches_per_observation=0,
            external_model_updates_per_observation=0,
            scientific_promotion_allowed=False,
            policy_authority=False,
            checkpoint_schema=CALIBRATED_EXTENDED_SEARCH_CHECKPOINT_SCHEMA,
            ranking_semantics=CALIBRATED_EXTENDED_SEARCH_RANKING_SEMANTICS,
        )

    def _option_universe_digest(self, descriptors: Array, generations: Array) -> Array:
        return _checksum_arrays((descriptors, generations), seed=self._config_seed)

    def _canonical_digest(
        self,
        representation_generation: Array,
        source_digest: Array,
        anchors: Array,
        anchor_active: Array,
        option_universe_digest: Array,
    ) -> Array:
        return _checksum_arrays(
            (
                representation_generation,
                source_digest,
                anchors,
                anchor_active,
                option_universe_digest,
            ),
            seed=self._config_seed,
        )

    def _pending_checksum(self, state: CalibratedExtendedSearchControlState) -> Array:
        return _checksum_arrays(
            (
                state.pending,
                state.pending_decision_id,
                state.pending_anchor_observation,
                state.pending_executed_kind,
                state.pending_executed_index,
                state.pending_anchor_index,
                state.pending_option_generation,
                state.pending_representation_generation,
                state.pending_source_digest,
                state.pending_option_universe_digest,
                state.pending_state_revision,
                state.pending_learner_revision,
                state.pending_primitive_model_revision,
                state.pending_option_model_revision,
                state.pending_average_reward,
                state.pending_frozen_q_values,
                state.pending_frozen_candidate_q,
                state.pending_candidate_targets,
                state.pending_target_available,
                state.pending_value_change_lcb,
                state.pending_reachability_lcb,
                state.pending_model_error_ucb,
                state.pending_support_shrinkage,
                state.pending_priorities,
                state.pending_candidate_eligible,
                state.pending_external_support,
                state.pending_selected_candidate_indices,
                state.pending_selected_kinds,
                state.pending_selected_semantic_indices,
                state.pending_selected_anchor_indices,
                state.pending_selected_targets,
                state.pending_selected_priorities,
                state.pending_selected_valid,
            ),
            seed=state.canonical_digest,
        )

    def _check_state_contract(self, state: CalibratedExtendedSearchControlState) -> None:
        if type(state) is not CalibratedExtendedSearchControlState:
            raise TypeError("state must be an exact CalibratedExtendedSearchControlState")
        cfg = self._config
        m = cfg.anchor_capacity
        d = cfg.observation_dim
        n = cfg.n_options
        a = cfg.n_extended_actions
        c = cfg.candidate_capacity
        b = cfg.backup_budget
        contracts = (
            (state.representation_generation, "representation_generation", (), jnp.int32),
            (state.source_digest, "source_digest", (2,), jnp.uint32),
            (state.canonical_digest, "canonical_digest", (2,), jnp.uint32),
            (state.option_descriptors, "option_descriptors", (n, 4), jnp.int32),
            (state.option_generations, "option_generations", (n,), jnp.int32),
            (state.option_universe_digest, "option_universe_digest", (2,), jnp.uint32),
            (state.anchor_bank, "anchor_bank", (m, d), jnp.float32),
            (state.anchor_active, "anchor_active", (m,), jnp.bool_),
            (state.q_values, "q_values", (m, a), jnp.float32),
            (state.state_revision, "state_revision", (), jnp.int32),
            (state.learner_revision, "learner_revision", (), jnp.int32),
            (state.primitive_model_revision, "primitive_model_revision", (), jnp.int32),
            (state.option_model_revision, "option_model_revision", (), jnp.int32),
            (state.has_last_decision, "has_last_decision", (), jnp.bool_),
            (state.last_decision_id, "last_decision_id", (4,), jnp.uint32),
            (state.last_realized_targets, "last_realized_targets", (c,), jnp.float32),
            (state.last_target_available, "last_target_available", (c,), jnp.bool_),
            (state.value_change_counts, "value_change_counts", (c,), jnp.int32),
            (state.value_change_means, "value_change_means", (c,), jnp.float32),
            (state.value_change_m2, "value_change_m2", (c,), jnp.float32),
            (state.model_error_counts, "model_error_counts", (c,), jnp.int32),
            (state.model_error_means, "model_error_means", (c,), jnp.float32),
            (state.model_error_m2, "model_error_m2", (c,), jnp.float32),
            (state.support_counts, "support_counts", (c,), jnp.int32),
            (state.anchor_revisit_trials, "anchor_revisit_trials", (m,), jnp.int32),
            (
                state.anchor_revisit_successes,
                "anchor_revisit_successes",
                (m,),
                jnp.int32,
            ),
            (state.pending, "pending", (), jnp.bool_),
            (state.pending_decision_id, "pending_decision_id", (4,), jnp.uint32),
            (
                state.pending_anchor_observation,
                "pending_anchor_observation",
                (d,),
                jnp.float32,
            ),
            (state.pending_executed_kind, "pending_executed_kind", (), jnp.int32),
            (state.pending_executed_index, "pending_executed_index", (), jnp.int32),
            (state.pending_anchor_index, "pending_anchor_index", (), jnp.int32),
            (
                state.pending_option_generation,
                "pending_option_generation",
                (),
                jnp.int32,
            ),
            (
                state.pending_representation_generation,
                "pending_representation_generation",
                (),
                jnp.int32,
            ),
            (state.pending_source_digest, "pending_source_digest", (2,), jnp.uint32),
            (
                state.pending_option_universe_digest,
                "pending_option_universe_digest",
                (2,),
                jnp.uint32,
            ),
            (state.pending_state_revision, "pending_state_revision", (), jnp.int32),
            (state.pending_learner_revision, "pending_learner_revision", (), jnp.int32),
            (
                state.pending_primitive_model_revision,
                "pending_primitive_model_revision",
                (),
                jnp.int32,
            ),
            (
                state.pending_option_model_revision,
                "pending_option_model_revision",
                (),
                jnp.int32,
            ),
            (state.pending_average_reward, "pending_average_reward", (), jnp.float32),
            (state.pending_frozen_q_values, "pending_frozen_q_values", (m, a), jnp.float32),
            (state.pending_frozen_candidate_q, "pending_frozen_candidate_q", (c,), jnp.float32),
            (state.pending_candidate_targets, "pending_candidate_targets", (c,), jnp.float32),
            (state.pending_target_available, "pending_target_available", (c,), jnp.bool_),
            (state.pending_value_change_lcb, "pending_value_change_lcb", (c,), jnp.float32),
            (state.pending_reachability_lcb, "pending_reachability_lcb", (c,), jnp.float32),
            (state.pending_model_error_ucb, "pending_model_error_ucb", (c,), jnp.float32),
            (state.pending_support_shrinkage, "pending_support_shrinkage", (c,), jnp.float32),
            (state.pending_priorities, "pending_priorities", (c,), jnp.float32),
            (state.pending_candidate_eligible, "pending_candidate_eligible", (c,), jnp.bool_),
            (state.pending_external_support, "pending_external_support", (c,), jnp.int32),
            (
                state.pending_selected_candidate_indices,
                "pending_selected_candidate_indices",
                (b,),
                jnp.int32,
            ),
            (state.pending_selected_kinds, "pending_selected_kinds", (b,), jnp.int32),
            (
                state.pending_selected_semantic_indices,
                "pending_selected_semantic_indices",
                (b,),
                jnp.int32,
            ),
            (
                state.pending_selected_anchor_indices,
                "pending_selected_anchor_indices",
                (b,),
                jnp.int32,
            ),
            (state.pending_selected_targets, "pending_selected_targets", (b,), jnp.float32),
            (
                state.pending_selected_priorities,
                "pending_selected_priorities",
                (b,),
                jnp.float32,
            ),
            (state.pending_selected_valid, "pending_selected_valid", (b,), jnp.bool_),
            (state.pending_cache_digest, "pending_cache_digest", (2,), jnp.uint32),
        )
        for value, name, shape, dtype in contracts:
            _require_array(value, name=f"state.{name}", shape=shape, dtype=dtype)

    def _blank_pending(
        self, state: CalibratedExtendedSearchControlState
    ) -> CalibratedExtendedSearchControlState:
        cfg = self._config
        blank = state.replace(
            pending=jnp.asarray(False, dtype=jnp.bool_),
            pending_decision_id=jnp.zeros((4,), dtype=jnp.uint32),
            pending_anchor_observation=jnp.zeros(
                (cfg.observation_dim,), dtype=jnp.float32
            ),
            pending_executed_kind=jnp.asarray(-1, dtype=jnp.int32),
            pending_executed_index=jnp.asarray(-1, dtype=jnp.int32),
            pending_anchor_index=jnp.asarray(-1, dtype=jnp.int32),
            pending_option_generation=jnp.asarray(-1, dtype=jnp.int32),
            pending_representation_generation=jnp.asarray(-1, dtype=jnp.int32),
            pending_source_digest=jnp.zeros((2,), dtype=jnp.uint32),
            pending_option_universe_digest=jnp.zeros((2,), dtype=jnp.uint32),
            pending_state_revision=jnp.asarray(-1, dtype=jnp.int32),
            pending_learner_revision=jnp.asarray(-1, dtype=jnp.int32),
            pending_primitive_model_revision=jnp.asarray(-1, dtype=jnp.int32),
            pending_option_model_revision=jnp.asarray(-1, dtype=jnp.int32),
            pending_average_reward=jnp.asarray(0.0, dtype=jnp.float32),
            pending_frozen_q_values=jnp.zeros_like(state.q_values),
            pending_frozen_candidate_q=jnp.zeros(
                (cfg.candidate_capacity,), dtype=jnp.float32
            ),
            pending_candidate_targets=jnp.zeros(
                (cfg.candidate_capacity,), dtype=jnp.float32
            ),
            pending_target_available=jnp.zeros(
                (cfg.candidate_capacity,), dtype=jnp.bool_
            ),
            pending_value_change_lcb=jnp.zeros(
                (cfg.candidate_capacity,), dtype=jnp.float32
            ),
            pending_reachability_lcb=jnp.zeros(
                (cfg.candidate_capacity,), dtype=jnp.float32
            ),
            pending_model_error_ucb=jnp.ones(
                (cfg.candidate_capacity,), dtype=jnp.float32
            ),
            pending_support_shrinkage=jnp.zeros(
                (cfg.candidate_capacity,), dtype=jnp.float32
            ),
            pending_priorities=jnp.zeros(
                (cfg.candidate_capacity,), dtype=jnp.float32
            ),
            pending_candidate_eligible=jnp.zeros(
                (cfg.candidate_capacity,), dtype=jnp.bool_
            ),
            pending_external_support=jnp.zeros(
                (cfg.candidate_capacity,), dtype=jnp.int32
            ),
            pending_selected_candidate_indices=jnp.full(
                (cfg.backup_budget,), -1, dtype=jnp.int32
            ),
            pending_selected_kinds=jnp.full(
                (cfg.backup_budget,), -1, dtype=jnp.int32
            ),
            pending_selected_semantic_indices=jnp.full(
                (cfg.backup_budget,), -1, dtype=jnp.int32
            ),
            pending_selected_anchor_indices=jnp.full(
                (cfg.backup_budget,), -1, dtype=jnp.int32
            ),
            pending_selected_targets=jnp.zeros(
                (cfg.backup_budget,), dtype=jnp.float32
            ),
            pending_selected_priorities=jnp.zeros(
                (cfg.backup_budget,), dtype=jnp.float32
            ),
            pending_selected_valid=jnp.zeros(
                (cfg.backup_budget,), dtype=jnp.bool_
            ),
            pending_cache_digest=jnp.zeros((2,), dtype=jnp.uint32),
        )
        return cast(
            CalibratedExtendedSearchControlState,
            blank.replace(pending_cache_digest=self._pending_checksum(blank)),
        )

    def init(
        self,
        *,
        anchor_bank: Array,
        anchor_active: Array,
        q_values: Array,
        option_descriptors: Array,
        option_generations: Array,
        representation_generation: int | Array,
        source_digest: Array,
        learner_revision: int | Array = 0,
        primitive_model_revision: int | Array = 0,
        option_model_revision: int | Array = 0,
    ) -> CalibratedExtendedSearchControlState:
        """Initialize one exact fixed-capacity, RNG-free search state."""

        cfg = self._config
        anchors = _require_array(
            anchor_bank,
            name="anchor_bank",
            shape=(cfg.anchor_capacity, cfg.observation_dim),
            dtype=jnp.float32,
        )
        active = _require_array(
            anchor_active,
            name="anchor_active",
            shape=(cfg.anchor_capacity,),
            dtype=jnp.bool_,
        )
        q = _require_array(
            q_values,
            name="q_values",
            shape=(cfg.anchor_capacity, cfg.n_extended_actions),
            dtype=jnp.float32,
        )
        descriptors = _require_array(
            option_descriptors,
            name="option_descriptors",
            shape=(cfg.n_options, _OPTION_DESCRIPTOR_WIDTH),
            dtype=jnp.int32,
        )
        generations = _require_array(
            option_generations,
            name="option_generations",
            shape=(cfg.n_options,),
            dtype=jnp.int32,
        )
        generation = _int32_scalar(
            representation_generation, name="representation_generation"
        )
        source = _require_array(
            source_digest, name="source_digest", shape=(2,), dtype=jnp.uint32
        )
        learner = _int32_scalar(learner_revision, name="learner_revision")
        primitive_revision = _int32_scalar(
            primitive_model_revision, name="primitive_model_revision"
        )
        option_revision = _int32_scalar(
            option_model_revision, name="option_model_revision"
        )
        universe = self._option_universe_digest(descriptors, generations)
        canonical = self._canonical_digest(generation, source, anchors, active, universe)
        c = cfg.candidate_capacity
        m = cfg.anchor_capacity
        state = CalibratedExtendedSearchControlState(
            representation_generation=generation,
            source_digest=source,
            canonical_digest=canonical,
            option_descriptors=descriptors,
            option_generations=generations,
            option_universe_digest=universe,
            anchor_bank=anchors,
            anchor_active=active,
            q_values=q,
            state_revision=jnp.asarray(0, dtype=jnp.int32),
            learner_revision=learner,
            primitive_model_revision=primitive_revision,
            option_model_revision=option_revision,
            has_last_decision=jnp.asarray(False, dtype=jnp.bool_),
            last_decision_id=jnp.zeros((4,), dtype=jnp.uint32),
            last_realized_targets=jnp.zeros((c,), dtype=jnp.float32),
            last_target_available=jnp.zeros((c,), dtype=jnp.bool_),
            value_change_counts=jnp.zeros((c,), dtype=jnp.int32),
            value_change_means=jnp.zeros((c,), dtype=jnp.float32),
            value_change_m2=jnp.zeros((c,), dtype=jnp.float32),
            model_error_counts=jnp.zeros((c,), dtype=jnp.int32),
            model_error_means=jnp.zeros((c,), dtype=jnp.float32),
            model_error_m2=jnp.zeros((c,), dtype=jnp.float32),
            support_counts=jnp.zeros((c,), dtype=jnp.int32),
            anchor_revisit_trials=jnp.zeros((m,), dtype=jnp.int32),
            anchor_revisit_successes=jnp.zeros((m,), dtype=jnp.int32),
            pending=jnp.asarray(False, dtype=jnp.bool_),
            pending_decision_id=jnp.zeros((4,), dtype=jnp.uint32),
            pending_anchor_observation=jnp.zeros(
                (cfg.observation_dim,), dtype=jnp.float32
            ),
            pending_executed_kind=jnp.asarray(-1, dtype=jnp.int32),
            pending_executed_index=jnp.asarray(-1, dtype=jnp.int32),
            pending_anchor_index=jnp.asarray(-1, dtype=jnp.int32),
            pending_option_generation=jnp.asarray(-1, dtype=jnp.int32),
            pending_representation_generation=jnp.asarray(-1, dtype=jnp.int32),
            pending_source_digest=jnp.zeros((2,), dtype=jnp.uint32),
            pending_option_universe_digest=jnp.zeros((2,), dtype=jnp.uint32),
            pending_state_revision=jnp.asarray(-1, dtype=jnp.int32),
            pending_learner_revision=jnp.asarray(-1, dtype=jnp.int32),
            pending_primitive_model_revision=jnp.asarray(-1, dtype=jnp.int32),
            pending_option_model_revision=jnp.asarray(-1, dtype=jnp.int32),
            pending_average_reward=jnp.asarray(0.0, dtype=jnp.float32),
            pending_frozen_q_values=jnp.zeros_like(q),
            pending_frozen_candidate_q=jnp.zeros((c,), dtype=jnp.float32),
            pending_candidate_targets=jnp.zeros((c,), dtype=jnp.float32),
            pending_target_available=jnp.zeros((c,), dtype=jnp.bool_),
            pending_value_change_lcb=jnp.zeros((c,), dtype=jnp.float32),
            pending_reachability_lcb=jnp.zeros((c,), dtype=jnp.float32),
            pending_model_error_ucb=jnp.ones((c,), dtype=jnp.float32),
            pending_support_shrinkage=jnp.zeros((c,), dtype=jnp.float32),
            pending_priorities=jnp.zeros((c,), dtype=jnp.float32),
            pending_candidate_eligible=jnp.zeros((c,), dtype=jnp.bool_),
            pending_external_support=jnp.zeros((c,), dtype=jnp.int32),
            pending_selected_candidate_indices=jnp.full(
                (cfg.backup_budget,), -1, dtype=jnp.int32
            ),
            pending_selected_kinds=jnp.full(
                (cfg.backup_budget,), -1, dtype=jnp.int32
            ),
            pending_selected_semantic_indices=jnp.full(
                (cfg.backup_budget,), -1, dtype=jnp.int32
            ),
            pending_selected_anchor_indices=jnp.full(
                (cfg.backup_budget,), -1, dtype=jnp.int32
            ),
            pending_selected_targets=jnp.zeros(
                (cfg.backup_budget,), dtype=jnp.float32
            ),
            pending_selected_priorities=jnp.zeros(
                (cfg.backup_budget,), dtype=jnp.float32
            ),
            pending_selected_valid=jnp.zeros(
                (cfg.backup_budget,), dtype=jnp.bool_
            ),
            pending_cache_digest=jnp.zeros((2,), dtype=jnp.uint32),
        )
        state = state.replace(pending_cache_digest=self._pending_checksum(state))
        return cast(CalibratedExtendedSearchControlState, state)

    def validate_state(
        self,
        state: CalibratedExtendedSearchControlState,
        *,
        representation_generation: int | Array,
        source_digest: Array,
        option_descriptors: Array,
        option_generations: Array,
    ) -> Bool[Array, ""]:
        """Validate shape, values, canonical identities, and a pending cache."""

        self._check_state_contract(state)
        cfg = self._config
        generation = _int32_scalar(
            representation_generation, name="representation_generation"
        )
        source = _require_array(
            source_digest, name="source_digest", shape=(2,), dtype=jnp.uint32
        )
        descriptors = _require_array(
            option_descriptors,
            name="option_descriptors",
            shape=(cfg.n_options, 4),
            dtype=jnp.int32,
        )
        generations = _require_array(
            option_generations,
            name="option_generations",
            shape=(cfg.n_options,),
            dtype=jnp.int32,
        )
        universe = self._option_universe_digest(state.option_descriptors, state.option_generations)
        canonical = self._canonical_digest(
            state.representation_generation,
            state.source_digest,
            state.anchor_bank,
            state.anchor_active,
            state.option_universe_digest,
        )
        finite = (
            jnp.all(jnp.isfinite(state.anchor_bank))
            & jnp.all(jnp.isfinite(state.q_values))
            & jnp.all(jnp.isfinite(state.last_realized_targets))
            & jnp.all(jnp.isfinite(state.value_change_means))
            & jnp.all(jnp.isfinite(state.value_change_m2))
            & jnp.all(jnp.isfinite(state.model_error_means))
            & jnp.all(jnp.isfinite(state.model_error_m2))
            & jnp.all(jnp.isfinite(state.pending_anchor_observation))
            & jnp.isfinite(state.pending_average_reward)
            & jnp.all(jnp.isfinite(state.pending_frozen_q_values))
            & jnp.all(jnp.isfinite(state.pending_frozen_candidate_q))
            & jnp.all(jnp.isfinite(state.pending_candidate_targets))
            & jnp.all(jnp.isfinite(state.pending_value_change_lcb))
            & jnp.all(jnp.isfinite(state.pending_reachability_lcb))
            & jnp.all(jnp.isfinite(state.pending_model_error_ucb))
            & jnp.all(jnp.isfinite(state.pending_support_shrinkage))
            & jnp.all(jnp.isfinite(state.pending_priorities))
            & jnp.all(jnp.isfinite(state.pending_selected_targets))
            & jnp.all(jnp.isfinite(state.pending_selected_priorities))
        )
        counts = (
            jnp.all(
                (state.value_change_counts >= 0)
                & (state.value_change_counts <= cfg.max_observations)
            )
            & jnp.all(
                (state.model_error_counts >= 0)
                & (state.model_error_counts <= cfg.max_observations)
            )
            & jnp.all(
                (state.support_counts >= 0)
                & (state.support_counts <= cfg.max_observations)
            )
            & jnp.all(
                (state.anchor_revisit_trials >= 0)
                & (state.anchor_revisit_trials <= cfg.max_observations)
            )
            & jnp.all(state.anchor_revisit_successes >= 0)
            & jnp.all(state.anchor_revisit_successes <= state.anchor_revisit_trials)
            & (state.state_revision >= 0)
            & (state.state_revision <= cfg.max_observations)
            & (state.learner_revision >= 0)
            & (state.learner_revision <= _INT32_MAX)
            & (state.primitive_model_revision >= 0)
            & (state.option_model_revision >= 0)
        )
        moments = (
            jnp.all(state.value_change_m2 >= 0.0)
            & jnp.all(state.model_error_m2 >= 0.0)
            & jnp.all(state.value_change_means >= 0.0)
            & jnp.all((state.model_error_means >= 0.0) & (state.model_error_means <= 1.0))
        )
        identity = (
            (state.representation_generation == generation)
            & jnp.array_equal(state.source_digest, source)
            & jnp.array_equal(state.option_descriptors, descriptors)
            & jnp.array_equal(state.option_generations, generations)
            & jnp.array_equal(state.option_universe_digest, universe)
            & jnp.array_equal(state.canonical_digest, canonical)
            & jnp.all(state.option_generations >= 0)
            & jnp.any(state.anchor_active)
        )
        anchor_equal = jnp.all(
            state.anchor_bank[:, None, :] == state.anchor_bank[None, :, :], axis=2
        )
        active_pairs = state.anchor_active[:, None] & state.anchor_active[None, :]
        distinct_active_anchors = ~jnp.any(
            anchor_equal & active_pairs & ~jnp.eye(cfg.anchor_capacity, dtype=jnp.bool_)
        )
        pending_cache_valid = jnp.array_equal(
            state.pending_cache_digest, self._pending_checksum(state)
        )
        pending_semantics = (~state.pending) | (
            (state.pending_state_revision == state.state_revision)
            & (state.pending_learner_revision == state.learner_revision)
            & (state.pending_primitive_model_revision == state.primitive_model_revision)
            & (state.pending_option_model_revision == state.option_model_revision)
            & (state.pending_representation_generation == state.representation_generation)
            & jnp.array_equal(state.pending_source_digest, state.source_digest)
            & jnp.array_equal(
                state.pending_option_universe_digest, state.option_universe_digest
            )
            & (state.pending_anchor_index >= 0)
            & (state.pending_anchor_index < cfg.anchor_capacity)
            & state.anchor_active[jnp.clip(state.pending_anchor_index, 0, cfg.anchor_capacity - 1)]
            & jnp.array_equal(
                state.pending_anchor_observation,
                state.anchor_bank[jnp.clip(state.pending_anchor_index, 0, cfg.anchor_capacity - 1)],
            )
            & jnp.array_equal(state.pending_frozen_q_values, state.q_values)
        )
        return (
            finite
            & counts
            & moments
            & identity
            & distinct_active_anchors
            & pending_cache_valid
            & pending_semantics
        )

    def _factor_estimates(
        self, state: CalibratedExtendedSearchControlState
    ) -> tuple[Array, Array, Array]:
        cfg = self._config
        value_n = jnp.maximum(state.value_change_counts, 1).astype(jnp.float32)
        value_var = state.value_change_m2 / jnp.maximum(value_n - 1.0, 1.0)
        value_se = jnp.sqrt(jnp.maximum(value_var, 0.0) / value_n)
        value_lcb = jnp.maximum(
            state.value_change_means
            - jnp.asarray(cfg.confidence_scale, dtype=jnp.float32) * value_se,
            0.0,
        )
        error_n = jnp.maximum(state.model_error_counts, 1).astype(jnp.float32)
        error_var = state.model_error_m2 / jnp.maximum(error_n - 1.0, 1.0)
        error_se = jnp.sqrt(jnp.maximum(error_var, 0.0) / error_n)
        error_ucb = jnp.clip(
            state.model_error_means
            + jnp.asarray(cfg.confidence_scale, dtype=jnp.float32) * error_se,
            0.0,
            1.0,
        )
        trials = jnp.maximum(state.anchor_revisit_trials, 1).astype(jnp.float32)
        reach_mean = state.anchor_revisit_successes.astype(jnp.float32) / trials
        reach_se = jnp.sqrt(jnp.maximum(reach_mean * (1.0 - reach_mean), 0.0) / trials)
        anchor_lcb = jnp.clip(
            reach_mean
            - jnp.asarray(cfg.confidence_scale, dtype=jnp.float32) * reach_se,
            0.0,
            1.0,
        )
        reach_lcb = anchor_lcb[self._candidate_anchor_indices]
        return value_lcb, reach_lcb, error_ucb

    def _schedule(
        self, priorities: Array, eligible: Array, targets: Array
    ) -> tuple[Array, Array, Array, Array, Array, Array, Array]:
        cfg = self._config
        work = jnp.where(eligible, priorities, -jnp.ones_like(priorities))
        indices: list[Array] = []
        valid_rows: list[Array] = []
        for _ in range(cfg.backup_budget):
            index = jnp.argmax(work).astype(jnp.int32)
            valid = work[index] >= 0.0
            indices.append(jnp.where(valid, index, -1).astype(jnp.int32))
            valid_rows.append(valid)
            work = work.at[index].set(-1.0)
        selected = jnp.stack(indices, axis=0)
        valid = jnp.stack(valid_rows, axis=0)
        safe = jnp.clip(selected, 0, cfg.candidate_capacity - 1)
        return (
            selected,
            self._candidate_kinds[safe],
            self._candidate_semantic_indices[safe],
            self._candidate_anchor_indices[safe],
            jnp.where(valid, targets[safe], 0.0),
            jnp.where(valid, priorities[safe], 0.0),
            valid,
        )

    def arm(
        self,
        state: CalibratedExtendedSearchControlState,
        *,
        decision_id: Array,
        decision_observation: Array,
        decision_anchor_index: int | Array,
        executed_kind: int | Array,
        executed_index: int | Array,
        average_reward: float | Array,
        primitive_reward_predictions: Array,
        primitive_discount_predictions: Array,
        primitive_next_anchor_probabilities: Array,
        primitive_model_available: Array,
        primitive_model_support: Array,
        option_return_predictions: Array,
        option_baseline_mass_predictions: Array,
        option_discount_predictions: Array,
        option_next_anchor_probabilities: Array,
        option_model_available: Array,
        option_model_support: Array,
        option_initiation_available: Array,
        representation_generation: int | Array,
        source_digest: Array,
        option_descriptors: Array,
        option_generations: Array,
        learner_revision: int | Array,
        primitive_model_revision: int | Array,
        option_model_revision: int | Array,
    ) -> CalibratedExtendedSearchArmResult:
        """Freeze one decision and its complete pre-outcome search schedule."""

        cfg = self._config
        m, k, n = cfg.anchor_capacity, cfg.n_primitive_actions, cfg.n_options
        identity = _require_array(
            decision_id, name="decision_id", shape=(4,), dtype=jnp.uint32
        )
        observation = _require_array(
            decision_observation,
            name="decision_observation",
            shape=(cfg.observation_dim,),
            dtype=jnp.float32,
        )
        anchor = _int32_scalar(decision_anchor_index, name="decision_anchor_index")
        kind = _int32_scalar(executed_kind, name="executed_kind")
        semantic_index = _int32_scalar(executed_index, name="executed_index")
        avg_reward = _float32_scalar(average_reward, name="average_reward")
        primitive_rewards = _require_array(
            primitive_reward_predictions,
            name="primitive_reward_predictions",
            shape=(m, k),
            dtype=jnp.float32,
        )
        primitive_discounts = _require_array(
            primitive_discount_predictions,
            name="primitive_discount_predictions",
            shape=(m, k),
            dtype=jnp.float32,
        )
        primitive_next = _require_array(
            primitive_next_anchor_probabilities,
            name="primitive_next_anchor_probabilities",
            shape=(m, k, m),
            dtype=jnp.float32,
        )
        primitive_available = _require_array(
            primitive_model_available,
            name="primitive_model_available",
            shape=(m, k),
            dtype=jnp.bool_,
        )
        primitive_support = _require_array(
            primitive_model_support,
            name="primitive_model_support",
            shape=(m, k),
            dtype=jnp.int32,
        )
        option_returns = _require_array(
            option_return_predictions,
            name="option_return_predictions",
            shape=(m, n),
            dtype=jnp.float32,
        )
        option_mass = _require_array(
            option_baseline_mass_predictions,
            name="option_baseline_mass_predictions",
            shape=(m, n),
            dtype=jnp.float32,
        )
        option_discounts = _require_array(
            option_discount_predictions,
            name="option_discount_predictions",
            shape=(m, n),
            dtype=jnp.float32,
        )
        option_next = _require_array(
            option_next_anchor_probabilities,
            name="option_next_anchor_probabilities",
            shape=(m, n, m),
            dtype=jnp.float32,
        )
        option_available = _require_array(
            option_model_available,
            name="option_model_available",
            shape=(m, n),
            dtype=jnp.bool_,
        )
        option_support = _require_array(
            option_model_support,
            name="option_model_support",
            shape=(m, n),
            dtype=jnp.int32,
        )
        option_initiation = _require_array(
            option_initiation_available,
            name="option_initiation_available",
            shape=(m, n),
            dtype=jnp.bool_,
        )
        generation = _int32_scalar(
            representation_generation, name="representation_generation"
        )
        source = _require_array(
            source_digest, name="source_digest", shape=(2,), dtype=jnp.uint32
        )
        descriptors = _require_array(
            option_descriptors,
            name="option_descriptors",
            shape=(n, 4),
            dtype=jnp.int32,
        )
        generations = _require_array(
            option_generations,
            name="option_generations",
            shape=(n,),
            dtype=jnp.int32,
        )
        learner = _int32_scalar(learner_revision, name="learner_revision")
        primitive_revision = _int32_scalar(
            primitive_model_revision, name="primitive_model_revision"
        )
        option_revision = _int32_scalar(
            option_model_revision, name="option_model_revision"
        )

        state_valid = self.validate_state(
            state,
            representation_generation=generation,
            source_digest=source,
            option_descriptors=descriptors,
            option_generations=generations,
        )
        anchor_safe = jnp.clip(anchor, 0, m - 1)
        kind_valid = (kind == CANDIDATE_KIND_PRIMITIVE) | (
            kind == CANDIDATE_KIND_OPTION
        )
        executed_index_valid = jnp.where(
            kind == CANDIDATE_KIND_PRIMITIVE,
            (semantic_index >= 0) & (semantic_index < k),
            (semantic_index >= 0) & (semantic_index < n),
        )
        safe_option_index = jnp.clip(semantic_index, 0, n - 1)
        decision_anchor_matches = (
            (anchor >= 0)
            & (anchor < m)
            & state.anchor_active[anchor_safe]
            & jnp.array_equal(observation, state.anchor_bank[anchor_safe])
            & kind_valid
            & executed_index_valid
            & jnp.where(
                kind == CANDIDATE_KIND_OPTION,
                option_initiation[anchor_safe, safe_option_index],
                True,
            )
        )
        live_binding_matches = (
            ~state.pending
            & (generation == state.representation_generation)
            & jnp.array_equal(source, state.source_digest)
            & jnp.array_equal(descriptors, state.option_descriptors)
            & jnp.array_equal(generations, state.option_generations)
            & (learner == state.learner_revision)
            & (primitive_revision >= state.primitive_model_revision)
            & (option_revision >= state.option_model_revision)
            & (~state.has_last_decision | ~jnp.array_equal(identity, state.last_decision_id))
        )
        all_float_inputs = (
            observation,
            avg_reward,
            primitive_rewards,
            primitive_discounts,
            primitive_next,
            option_returns,
            option_mass,
            option_discounts,
            option_next,
        )
        inputs_finite = jnp.asarray(True, dtype=jnp.bool_)
        for value in all_float_inputs:
            inputs_finite = inputs_finite & jnp.all(jnp.isfinite(value))
        probability_tolerance = jnp.asarray(1.0e-5, dtype=jnp.float32)
        primitive_probability_valid = (
            jnp.all(primitive_next >= 0.0, axis=2)
            & (jnp.max(primitive_next, axis=2) <= 1.0)
            & (jnp.abs(jnp.sum(primitive_next, axis=2) - 1.0) <= probability_tolerance)
        )
        option_probability_valid = (
            jnp.all(option_next >= 0.0, axis=2)
            & (jnp.max(option_next, axis=2) <= 1.0)
            & (jnp.abs(jnp.sum(option_next, axis=2) - 1.0) <= probability_tolerance)
        )
        primitive_semantics_valid = (
            (primitive_discounts >= 0.0)
            & (primitive_discounts <= 1.0)
            & primitive_probability_valid
            & (primitive_support >= 0)
            & (primitive_support <= cfg.max_observations)
        )
        option_semantics_valid = (
            (option_mass >= 0.0)
            & (option_discounts >= 0.0)
            & (option_discounts <= 1.0)
            & option_probability_valid
            & (option_support >= 0)
            & (option_support <= cfg.max_observations)
        )
        model_contract_valid = jnp.all(primitive_semantics_valid) & jnp.all(
            option_semantics_valid
        )
        capacity_available = (
            (state.state_revision < cfg.max_observations)
            & (state.learner_revision <= _INT32_MAX - cfg.backup_budget)
        )

        next_values = jnp.max(state.q_values, axis=1)
        primitive_bootstrap = jnp.einsum("ikm,m->ik", primitive_next, next_values)
        option_bootstrap = jnp.einsum("ikm,m->ik", option_next, next_values)
        primitive_targets_matrix = (
            primitive_rewards - avg_reward + primitive_discounts * primitive_bootstrap
        )
        option_targets_matrix = (
            option_returns
            - avg_reward * option_mass
            + option_discounts * option_bootstrap
        )
        primitive_targets = jnp.transpose(primitive_targets_matrix).reshape((-1,))
        option_targets = jnp.transpose(option_targets_matrix).reshape((-1,))
        model_targets = jnp.concatenate((primitive_targets, option_targets), axis=0)
        primitive_target_available = jnp.transpose(
            primitive_available & primitive_semantics_valid
        ).reshape((-1,))
        option_target_available = jnp.transpose(
            option_available & option_initiation & option_semantics_valid
        ).reshape((-1,))
        model_available = jnp.concatenate(
            (primitive_target_available, option_target_available), axis=0
        )
        model_support = jnp.concatenate(
            (
                jnp.transpose(primitive_support).reshape((-1,)),
                jnp.transpose(option_support).reshape((-1,)),
            ),
            axis=0,
        )
        if cfg.mode == SEARCH_MODE_MODEL_FREE_EXTENDED_Q:
            targets = state.last_realized_targets
            target_available = state.last_target_available
            external_support = state.support_counts
            mode_mask = jnp.ones((cfg.candidate_capacity,), dtype=jnp.bool_)
        else:
            targets = model_targets
            target_available = model_available
            external_support = model_support
            if cfg.mode == SEARCH_MODE_PRIMITIVE_MODEL:
                mode_mask = self._candidate_kinds == CANDIDATE_KIND_PRIMITIVE
            elif cfg.mode == SEARCH_MODE_OPTION_MODEL:
                mode_mask = self._candidate_kinds == CANDIDATE_KIND_OPTION
            else:
                mode_mask = jnp.ones((cfg.candidate_capacity,), dtype=jnp.bool_)
        target_available = target_available & mode_mask & state.anchor_active[
            self._candidate_anchor_indices
        ]
        value_lcb, reach_lcb, error_ucb = self._factor_estimates(state)
        effective_support = jnp.minimum(state.support_counts, external_support)
        if cfg.mode == SEARCH_MODE_MODEL_FREE_EXTENDED_Q:
            effective_support = state.support_counts
        support_shrinkage = effective_support.astype(jnp.float32) / (
            effective_support.astype(jnp.float32)
            + jnp.asarray(cfg.support_prior, dtype=jnp.float32)
        )
        evidence_ready = (
            (state.value_change_counts >= cfg.calibration_evidence_floor)
            & (state.model_error_counts >= cfg.calibration_evidence_floor)
            & (
                state.anchor_revisit_trials[self._candidate_anchor_indices]
                >= cfg.calibration_evidence_floor
            )
            & (effective_support >= cfg.model_support_floor)
        )
        reliability = 1.0 - error_ucb
        candidate_eligible = (
            target_available
            & evidence_ready
            & (value_lcb > cfg.min_value_change_lcb)
            & (reach_lcb > cfg.min_reachability_lcb)
            & (reliability > cfg.min_model_reliability)
            & (support_shrinkage > 0.0)
        )
        priorities = jnp.where(
            candidate_eligible,
            value_lcb * reach_lcb * reliability * support_shrinkage,
            0.0,
        )
        schedule = self._schedule(priorities, candidate_eligible, targets)
        (
            selected,
            selected_kinds,
            selected_semantics,
            selected_anchors,
            selected_targets,
            selected_priorities,
            selected_valid,
        ) = schedule
        derived_values_valid = (
            jnp.all(jnp.isfinite(next_values))
            & jnp.all(jnp.isfinite(primitive_bootstrap))
            & jnp.all(jnp.isfinite(option_bootstrap))
            & jnp.all(jnp.isfinite(primitive_targets_matrix))
            & jnp.all(jnp.isfinite(option_targets_matrix))
            & jnp.all(jnp.isfinite(model_targets))
            & jnp.all(jnp.isfinite(targets))
            & jnp.all(jnp.isfinite(value_lcb))
            & jnp.all(jnp.isfinite(reach_lcb))
            & jnp.all(jnp.isfinite(error_ucb))
            & jnp.all(jnp.isfinite(support_shrinkage))
            & jnp.all(jnp.isfinite(reliability))
            & jnp.all(jnp.isfinite(priorities))
            & jnp.all(jnp.isfinite(selected_targets))
            & jnp.all(jnp.isfinite(selected_priorities))
            & jnp.all(value_lcb >= 0.0)
            & jnp.all((reach_lcb >= 0.0) & (reach_lcb <= 1.0))
            & jnp.all((error_ucb >= 0.0) & (error_ucb <= 1.0))
            & jnp.all((support_shrinkage >= 0.0) & (support_shrinkage <= 1.0))
            & jnp.all((reliability >= 0.0) & (reliability <= 1.0))
            & jnp.all(priorities >= 0.0)
            & jnp.all(selected_priorities >= 0.0)
        )
        pre_transaction_valid = (
            state_valid
            & live_binding_matches
            & decision_anchor_matches
            & inputs_finite
            & derived_values_valid
            & model_contract_valid
            & capacity_available
        )
        option_generation = jnp.where(
            kind == CANDIDATE_KIND_OPTION,
            generations[safe_option_index],
            -1,
        ).astype(jnp.int32)
        candidate_q = jnp.transpose(state.q_values).reshape((-1,))
        proposed = state.replace(
            primitive_model_revision=primitive_revision,
            option_model_revision=option_revision,
            pending=jnp.asarray(True, dtype=jnp.bool_),
            pending_decision_id=identity,
            pending_anchor_observation=observation,
            pending_executed_kind=kind,
            pending_executed_index=semantic_index,
            pending_anchor_index=anchor,
            pending_option_generation=option_generation,
            pending_representation_generation=generation,
            pending_source_digest=source,
            pending_option_universe_digest=state.option_universe_digest,
            pending_state_revision=state.state_revision,
            pending_learner_revision=state.learner_revision,
            pending_primitive_model_revision=primitive_revision,
            pending_option_model_revision=option_revision,
            pending_average_reward=avg_reward,
            pending_frozen_q_values=state.q_values,
            pending_frozen_candidate_q=candidate_q,
            pending_candidate_targets=targets,
            pending_target_available=target_available,
            pending_value_change_lcb=value_lcb,
            pending_reachability_lcb=reach_lcb,
            pending_model_error_ucb=error_ucb,
            pending_support_shrinkage=support_shrinkage,
            pending_priorities=priorities,
            pending_candidate_eligible=candidate_eligible,
            pending_external_support=external_support,
            pending_selected_candidate_indices=selected,
            pending_selected_kinds=selected_kinds,
            pending_selected_semantic_indices=selected_semantics,
            pending_selected_anchor_indices=selected_anchors,
            pending_selected_targets=selected_targets,
            pending_selected_priorities=selected_priorities,
            pending_selected_valid=selected_valid,
            pending_cache_digest=jnp.zeros((2,), dtype=jnp.uint32),
        )
        proposed = proposed.replace(pending_cache_digest=self._pending_checksum(proposed))
        proposed_state_valid = self.validate_state(
            proposed,
            representation_generation=generation,
            source_digest=source,
            option_descriptors=descriptors,
            option_generations=generations,
        )
        transaction_valid = pre_transaction_valid & proposed_state_valid
        committed = jax.lax.cond(transaction_valid, lambda _: proposed, lambda _: state, None)
        mask = transaction_valid
        diagnostics = CalibratedExtendedSearchArmDiagnostics(
            state_valid=state_valid,
            live_binding_matches=live_binding_matches,
            decision_anchor_matches=decision_anchor_matches,
            inputs_finite=inputs_finite,
            derived_values_valid=derived_values_valid,
            model_contract_valid=model_contract_valid,
            capacity_available=capacity_available,
            transaction_valid=transaction_valid,
            candidate_targets=jnp.where(mask, targets, 0.0),
            target_available=target_available & mask,
            value_change_lcb=jnp.where(mask, value_lcb, 0.0),
            reachability_lcb=jnp.where(mask, reach_lcb, 0.0),
            model_error_ucb=jnp.where(mask, error_ucb, 1.0),
            support_shrinkage=jnp.where(mask, support_shrinkage, 0.0),
            priorities=jnp.where(mask, priorities, 0.0),
            candidate_eligible=candidate_eligible & mask,
            selected_candidate_indices=jnp.where(mask, selected, -1),
            selected_kinds=jnp.where(mask, selected_kinds, -1),
            selected_semantic_indices=jnp.where(mask, selected_semantics, -1),
            selected_anchor_indices=jnp.where(mask, selected_anchors, -1),
            selected_targets=jnp.where(mask, selected_targets, 0.0),
            selected_priorities=jnp.where(mask, selected_priorities, 0.0),
            selected_valid=selected_valid & mask,
            backup_attempt_count=jnp.asarray(cfg.backup_budget, dtype=jnp.int32),
        )
        return CalibratedExtendedSearchArmResult(state=committed, diagnostics=diagnostics)

    @staticmethod
    def _welford_update(
        counts: Array, means: Array, m2: Array, index: Array, value: Array, apply: Array
    ) -> tuple[Array, Array, Array]:
        safe_index = jnp.clip(index, 0, counts.shape[0] - 1)
        old_count = counts[safe_index]
        new_count = old_count + jnp.asarray(1, dtype=jnp.int32)
        delta = value - means[safe_index]
        new_mean = means[safe_index] + delta / new_count.astype(jnp.float32)
        delta2 = value - new_mean
        new_m2 = m2[safe_index] + delta * delta2
        return (
            counts.at[safe_index].set(jnp.where(apply, new_count, old_count)),
            means.at[safe_index].set(jnp.where(apply, new_mean, means[safe_index])),
            m2.at[safe_index].set(jnp.where(apply, new_m2, m2[safe_index])),
        )

    def observe(
        self,
        state: CalibratedExtendedSearchControlState,
        *,
        decision_id: Array,
        future_observation: Array,
        observed_future_anchor_mask: Array,
        external_return: float | Array,
        baseline_mass: float | Array,
        terminal_discount: float | Array,
        elapsed_primitive_steps: int | Array,
        natural_completion: bool | Array,
        censored: bool | Array,
        representation_generation: int | Array,
        source_digest: Array,
        option_descriptors: Array,
        option_generations: Array,
        learner_revision: int | Array,
        primitive_model_revision: int | Array,
        option_model_revision: int | Array,
    ) -> CalibratedExtendedSearchObserveResult:
        """Resolve one exact pending arm and conditionally commit its schedule."""

        cfg = self._config
        m, n = cfg.anchor_capacity, cfg.n_options
        identity = _require_array(
            decision_id, name="decision_id", shape=(4,), dtype=jnp.uint32
        )
        future = _require_array(
            future_observation,
            name="future_observation",
            shape=(cfg.observation_dim,),
            dtype=jnp.float32,
        )
        observed_mask = _require_array(
            observed_future_anchor_mask,
            name="observed_future_anchor_mask",
            shape=(m,),
            dtype=jnp.bool_,
        )
        realized_return = _float32_scalar(external_return, name="external_return")
        realized_mass = _float32_scalar(baseline_mass, name="baseline_mass")
        realized_discount = _float32_scalar(terminal_discount, name="terminal_discount")
        elapsed = _int32_scalar(elapsed_primitive_steps, name="elapsed_primitive_steps")
        natural = _bool_scalar(natural_completion, name="natural_completion")
        censor = _bool_scalar(censored, name="censored")
        generation = _int32_scalar(
            representation_generation, name="representation_generation"
        )
        source = _require_array(
            source_digest, name="source_digest", shape=(2,), dtype=jnp.uint32
        )
        descriptors = _require_array(
            option_descriptors,
            name="option_descriptors",
            shape=(n, 4),
            dtype=jnp.int32,
        )
        generations = _require_array(
            option_generations,
            name="option_generations",
            shape=(n,),
            dtype=jnp.int32,
        )
        learner = _int32_scalar(learner_revision, name="learner_revision")
        primitive_revision = _int32_scalar(
            primitive_model_revision, name="primitive_model_revision"
        )
        option_revision = _int32_scalar(
            option_model_revision, name="option_model_revision"
        )
        pending_cache_valid = jnp.array_equal(
            state.pending_cache_digest, self._pending_checksum(state)
        )
        state_valid = self.validate_state(
            state,
            representation_generation=generation,
            source_digest=source,
            option_descriptors=descriptors,
            option_generations=generations,
        )
        safe_option = jnp.clip(state.pending_executed_index, 0, n - 1)
        option_generation_matches = jnp.where(
            state.pending_executed_kind == CANDIDATE_KIND_OPTION,
            (state.pending_option_generation == generations[safe_option])
            & (state.pending_executed_index >= 0)
            & (state.pending_executed_index < n),
            state.pending_option_generation == -1,
        )
        binding_matches = (
            state.pending
            & jnp.array_equal(identity, state.pending_decision_id)
            & (generation == state.pending_representation_generation)
            & jnp.array_equal(source, state.pending_source_digest)
            & jnp.array_equal(
                state.option_universe_digest, state.pending_option_universe_digest
            )
            & (learner == state.pending_learner_revision)
            & (primitive_revision == state.pending_primitive_model_revision)
            & (option_revision == state.pending_option_model_revision)
            & option_generation_matches
        )
        is_primitive = state.pending_executed_kind == CANDIDATE_KIND_PRIMITIVE
        natural_structure = natural & ~censor & jnp.where(
            is_primitive, elapsed == 1, elapsed >= 1
        )
        censor_structure = censor & ~natural & (elapsed >= 1)
        resolution_structure_valid = natural_structure | censor_structure
        exact_matches = (
            jnp.all(state.anchor_bank == future[None, :], axis=1) & state.anchor_active
        )
        future_anchor_evidence_valid = (
            jnp.array_equal(observed_mask, exact_matches)
            & (jnp.sum(exact_matches.astype(jnp.int32)) == 1)
        )
        inputs_finite = (
            jnp.all(jnp.isfinite(future))
            & jnp.isfinite(realized_return)
            & jnp.isfinite(realized_mass)
            & jnp.isfinite(realized_discount)
            & (realized_mass >= 0.0)
            & (realized_discount >= 0.0)
            & (realized_discount <= 1.0)
            & jnp.where(is_primitive, realized_mass == 1.0, True)
        )
        future_index = jnp.argmax(exact_matches).astype(jnp.int32)
        executed_extended = jnp.where(
            is_primitive,
            state.pending_executed_index,
            cfg.n_primitive_actions + state.pending_executed_index,
        ).astype(jnp.int32)
        candidate_index = executed_extended * cfg.anchor_capacity + state.pending_anchor_index
        safe_candidate = jnp.clip(candidate_index, 0, cfg.candidate_capacity - 1)
        future_value = jnp.max(state.pending_frozen_q_values[future_index])
        primitive_target = (
            realized_return - state.pending_average_reward + realized_discount * future_value
        )
        option_target = (
            realized_return
            - state.pending_average_reward * realized_mass
            + realized_discount * future_value
        )
        realized_target = jnp.where(is_primitive, primitive_target, option_target)
        frozen_model_target = state.pending_candidate_targets[safe_candidate]
        value_change = jnp.abs(
            realized_target - state.pending_frozen_candidate_q[safe_candidate]
        )
        model_prediction_available = state.pending_target_available[safe_candidate]
        model_error_magnitude = jnp.abs(frozen_model_target - realized_target)
        scaled_model_error = model_error_magnitude / jnp.asarray(
            cfg.model_error_scale, dtype=jnp.float32
        )
        if cfg.mode == SEARCH_MODE_MODEL_FREE_EXTENDED_Q:
            normalized_error = jnp.asarray(0.0, dtype=jnp.float32)
            error_observation_available = jnp.asarray(True, dtype=jnp.bool_)
        else:
            normalized_error = jnp.clip(scaled_model_error, 0.0, 1.0)
            error_observation_available = model_prediction_available
        natural_derived_values_valid = (
            jnp.isfinite(future_value)
            & jnp.isfinite(realized_target)
            & jnp.isfinite(frozen_model_target)
            & jnp.isfinite(value_change)
            & jnp.isfinite(normalized_error)
            & (value_change >= 0.0)
            & (normalized_error >= 0.0)
            & (normalized_error <= 1.0)
            & jnp.where(
                error_observation_available,
                jnp.isfinite(model_error_magnitude) & jnp.isfinite(scaled_model_error),
                True,
            )
        )
        derived_values_valid = jnp.where(
            natural_structure, natural_derived_values_valid, True
        )
        selected_count = jnp.sum(state.pending_selected_valid.astype(jnp.int32))
        resolution_capacity = (
            (state.state_revision < cfg.max_observations)
            & (state.value_change_counts[safe_candidate] < cfg.max_observations)
            & (state.support_counts[safe_candidate] < cfg.max_observations)
            & jnp.all(
                (~state.anchor_active)
                | (state.anchor_revisit_trials < cfg.max_observations)
            )
            & (state.learner_revision <= _INT32_MAX - selected_count)
            & jnp.where(
                error_observation_available,
                state.model_error_counts[safe_candidate] < cfg.max_observations,
                True,
            )
        )
        censor_capacity = state.state_revision < cfg.max_observations
        capacity_available = jnp.where(
            natural_structure, resolution_capacity, censor_capacity
        )
        outcome_semantics_valid = jnp.where(
            natural_structure, future_anchor_evidence_valid, ~jnp.any(observed_mask)
        )
        pre_transaction_valid = (
            state_valid
            & pending_cache_valid
            & binding_matches
            & resolution_structure_valid
            & outcome_semantics_valid
            & inputs_finite
            & derived_values_valid
            & capacity_available
        )
        natural_write = pre_transaction_valid & natural_structure

        next_value_counts, next_value_means, next_value_m2 = self._welford_update(
            state.value_change_counts,
            state.value_change_means,
            state.value_change_m2,
            safe_candidate,
            value_change,
            natural_write,
        )
        next_error_counts, next_error_means, next_error_m2 = self._welford_update(
            state.model_error_counts,
            state.model_error_means,
            state.model_error_m2,
            safe_candidate,
            normalized_error,
            natural_write & error_observation_available,
        )
        next_support = state.support_counts.at[safe_candidate].set(
            jnp.where(
                natural_write,
                state.support_counts[safe_candidate] + 1,
                state.support_counts[safe_candidate],
            )
        )
        next_targets = state.last_realized_targets.at[safe_candidate].set(
            jnp.where(
                natural_write,
                realized_target,
                state.last_realized_targets[safe_candidate],
            )
        )
        next_target_available = state.last_target_available.at[safe_candidate].set(
            state.last_target_available[safe_candidate] | natural_write
        )
        next_trials = jnp.where(
            state.anchor_active & natural_write,
            state.anchor_revisit_trials + 1,
            state.anchor_revisit_trials,
        )
        next_successes = jnp.where(
            state.anchor_active & observed_mask & natural_write,
            state.anchor_revisit_successes + 1,
            state.anchor_revisit_successes,
        )

        q_values = state.q_values
        td_rows: list[Array] = []
        applied_rows: list[Array] = []
        for attempt in range(cfg.backup_budget):
            flat_index = state.pending_selected_candidate_indices[attempt]
            safe_flat = jnp.clip(flat_index, 0, cfg.candidate_capacity - 1)
            extended_index = self._candidate_extended_indices[safe_flat]
            anchor_index = self._candidate_anchor_indices[safe_flat]
            target = state.pending_selected_targets[attempt]
            td_error = target - q_values[anchor_index, extended_index]
            apply = natural_write & state.pending_selected_valid[attempt]
            q_values = q_values.at[anchor_index, extended_index].set(
                jnp.where(
                    apply,
                    q_values[anchor_index, extended_index]
                    + jnp.asarray(cfg.backup_step_size, dtype=jnp.float32) * td_error,
                    q_values[anchor_index, extended_index],
                )
            )
            td_rows.append(jnp.where(apply, td_error, 0.0))
            applied_rows.append(apply)
        write_td_errors = jnp.stack(td_rows, axis=0)
        write_applied = jnp.stack(applied_rows, axis=0)
        write_applied_count = jnp.sum(write_applied.astype(jnp.int32))
        proposed = state.replace(
            q_values=q_values,
            state_revision=state.state_revision + 1,
            learner_revision=state.learner_revision + write_applied_count,
            has_last_decision=jnp.asarray(True, dtype=jnp.bool_),
            last_decision_id=identity,
            last_realized_targets=next_targets,
            last_target_available=next_target_available,
            value_change_counts=next_value_counts,
            value_change_means=next_value_means,
            value_change_m2=next_value_m2,
            model_error_counts=next_error_counts,
            model_error_means=next_error_means,
            model_error_m2=next_error_m2,
            support_counts=next_support,
            anchor_revisit_trials=next_trials,
            anchor_revisit_successes=next_successes,
        )
        proposed = self._blank_pending(proposed)
        proposed_state_valid = self.validate_state(
            proposed,
            representation_generation=generation,
            source_digest=source,
            option_descriptors=descriptors,
            option_generations=generations,
        )
        transaction_valid = pre_transaction_valid & proposed_state_valid
        natural_transaction = transaction_valid & natural_structure
        censor_transaction = transaction_valid & censor_structure
        applied = write_applied & transaction_valid
        td_errors = jnp.where(applied, write_td_errors, 0.0)
        applied_count = jnp.sum(applied.astype(jnp.int32))
        committed = jax.lax.cond(transaction_valid, lambda _: proposed, lambda _: state, None)
        diagnostics = CalibratedExtendedSearchObserveDiagnostics(
            state_valid=state_valid,
            pending_cache_valid=pending_cache_valid,
            binding_matches=binding_matches,
            resolution_structure_valid=resolution_structure_valid,
            future_anchor_evidence_valid=future_anchor_evidence_valid,
            inputs_finite=inputs_finite,
            derived_values_valid=derived_values_valid & proposed_state_valid,
            capacity_available=capacity_available,
            transaction_valid=transaction_valid,
            natural_resolution=natural_transaction,
            censored_resolution=censor_transaction,
            transaction_applied=transaction_valid,
            resolved_candidate_index=jnp.where(
                natural_transaction, candidate_index, -1
            ).astype(jnp.int32),
            realized_differential_target=jnp.where(
                natural_transaction, realized_target, 0.0
            ),
            frozen_model_target=jnp.where(
                natural_transaction & model_prediction_available,
                frozen_model_target,
                0.0,
            ),
            realized_value_change=jnp.where(natural_transaction, value_change, 0.0),
            normalized_model_error=jnp.where(
                natural_transaction & error_observation_available,
                normalized_error,
                0.0,
            ),
            calibration_updated=natural_transaction,
            reachability_updated=natural_transaction,
            backup_attempt_count=jnp.asarray(cfg.backup_budget, dtype=jnp.int32),
            selected_candidate_indices=state.pending_selected_candidate_indices,
            selected_kinds=state.pending_selected_kinds,
            selected_semantic_indices=state.pending_selected_semantic_indices,
            selected_anchor_indices=state.pending_selected_anchor_indices,
            td_errors=td_errors,
            learner_updates_applied=applied,
            learner_update_count=applied_count,
            rng_draw_count=jnp.asarray(0, dtype=jnp.int32),
        )
        return CalibratedExtendedSearchObserveResult(
            state=committed, diagnostics=diagnostics
        )

    def replace_option_universe(
        self,
        state: CalibratedExtendedSearchControlState,
        *,
        option_descriptors: Array,
        option_generations: Array,
    ) -> CalibratedExtendedSearchControlState:
        """Bounded maintenance replacement with complete changed-slot invalidation.

        A changed descriptor must advance its generation.  Every changed
        option loses Q values, cached real targets, calibration moments, and
        support at all anchors.  Any pending arm is cleared because its frozen
        option-universe digest is stale.  Unchanged option slots and real
        anchor reachability evidence are retained exactly.
        """

        self._check_state_contract(state)
        cfg = self._config
        state_valid = self.validate_state(
            state,
            representation_generation=state.representation_generation,
            source_digest=state.source_digest,
            option_descriptors=state.option_descriptors,
            option_generations=state.option_generations,
        )
        if not bool(jax.device_get(state_valid)):
            raise ValueError("cannot replace options in an invalid calibrated search state")
        if (
            int(jax.device_get(state.state_revision)) >= cfg.max_observations
            or int(jax.device_get(state.learner_revision)) >= _INT32_MAX
        ):
            raise ValueError("option replacement would overflow a bound state counter")
        descriptors = _require_array(
            option_descriptors,
            name="option_descriptors",
            shape=(cfg.n_options, 4),
            dtype=jnp.int32,
        )
        generations = _require_array(
            option_generations,
            name="option_generations",
            shape=(cfg.n_options,),
            dtype=jnp.int32,
        )
        old_descriptors = np.asarray(jax.device_get(state.option_descriptors))
        new_descriptors = np.asarray(jax.device_get(descriptors))
        old_generations = np.asarray(jax.device_get(state.option_generations))
        new_generations = np.asarray(jax.device_get(generations))
        changed_host = np.any(old_descriptors != new_descriptors, axis=1) | (
            old_generations != new_generations
        )
        if np.any(new_generations < 0):
            raise ValueError("option generations must remain non-negative")
        if np.any(new_generations[changed_host] <= old_generations[changed_host]):
            raise ValueError("every replaced option must strictly advance its generation")
        changed = jnp.asarray(changed_host, dtype=jnp.bool_)
        changed_candidates = changed[self._candidate_semantic_indices] & (
            self._candidate_kinds == CANDIDATE_KIND_OPTION
        )
        q_mask = jnp.concatenate(
            (
                jnp.zeros((cfg.n_primitive_actions,), dtype=jnp.bool_),
                changed,
            )
        )[None, :]
        q_values = jnp.where(q_mask, 0.0, state.q_values)

        def zero_changed(array: Array) -> Array:
            return jnp.where(changed_candidates, jnp.zeros_like(array), array)

        def false_changed(array: Array) -> Array:
            return jnp.where(changed_candidates, False, array)

        universe = self._option_universe_digest(descriptors, generations)
        canonical = self._canonical_digest(
            state.representation_generation,
            state.source_digest,
            state.anchor_bank,
            state.anchor_active,
            universe,
        )
        any_changed = bool(np.any(changed_host))
        if not any_changed:
            return state
        replaced = state.replace(
            canonical_digest=canonical,
            option_descriptors=descriptors,
            option_generations=generations,
            option_universe_digest=universe,
            q_values=q_values,
            state_revision=state.state_revision + 1,
            learner_revision=state.learner_revision + 1,
            last_realized_targets=zero_changed(state.last_realized_targets),
            last_target_available=false_changed(state.last_target_available),
            value_change_counts=zero_changed(state.value_change_counts),
            value_change_means=zero_changed(state.value_change_means),
            value_change_m2=zero_changed(state.value_change_m2),
            model_error_counts=zero_changed(state.model_error_counts),
            model_error_means=zero_changed(state.model_error_means),
            model_error_m2=zero_changed(state.model_error_m2),
            support_counts=zero_changed(state.support_counts),
        )
        return self._blank_pending(replaced)

    def _state_checksum(self, state: CalibratedExtendedSearchControlState) -> Array:
        digest = hashlib.sha256()
        for leaf in jax.tree_util.tree_leaves(state):
            host = np.asarray(jax.device_get(leaf))
            digest.update(host.dtype.str.encode("ascii"))
            digest.update(np.asarray(host.shape, dtype=np.int64).tobytes())
            digest.update(host.tobytes(order="C"))
        return jnp.asarray(tuple(digest.digest()), dtype=jnp.uint8)

    def checkpoint_payload(
        self, state: CalibratedExtendedSearchControlState
    ) -> dict[str, object]:
        """Return a strict generic-serializer payload, including pending arms."""

        valid = self.validate_state(
            state,
            representation_generation=state.representation_generation,
            source_digest=state.source_digest,
            option_descriptors=state.option_descriptors,
            option_generations=state.option_generations,
        )
        if not bool(jax.device_get(valid)):
            raise ValueError("cannot checkpoint an invalid calibrated search state")
        return {
            "schema_version": CALIBRATED_EXTENDED_SEARCH_CHECKPOINT_SCHEMA,
            "config": self.to_config(),
            "resource_budget": self.resource_budget.to_config(),
            "representation_generation": state.representation_generation,
            "source_digest": state.source_digest,
            "option_descriptors": state.option_descriptors,
            "option_generations": state.option_generations,
            "learner_revision": state.learner_revision,
            "primitive_model_revision": state.primitive_model_revision,
            "option_model_revision": state.option_model_revision,
            "state": state,
            "state_digest": self._state_checksum(state),
        }

    def restore_checkpoint(
        self,
        payload: object,
        *,
        representation_generation: int | Array,
        source_digest: Array,
        option_descriptors: Array,
        option_generations: Array,
        learner_revision: int | Array,
        primitive_model_revision: int | Array,
        option_model_revision: int | Array,
    ) -> CalibratedExtendedSearchControlState:
        """Restore only the exact v1 configuration and every live identity."""

        if type(payload) is not dict:
            raise ValueError("calibrated search checkpoint must be an exact dict")
        raw = cast(dict[object, object], payload)
        expected = {
            "schema_version",
            "config",
            "resource_budget",
            "representation_generation",
            "source_digest",
            "option_descriptors",
            "option_generations",
            "learner_revision",
            "primitive_model_revision",
            "option_model_revision",
            "state",
            "state_digest",
        }
        if set(raw) != expected:
            raise ValueError("calibrated search checkpoint fields differ from v1")
        if raw["schema_version"] != CALIBRATED_EXTENDED_SEARCH_CHECKPOINT_SCHEMA:
            raise ValueError("calibrated search checkpoint schema differs")
        restored_config = CalibratedExtendedSearchControlConfig.from_config(raw["config"])
        if restored_config != self._config:
            raise ValueError("calibrated search checkpoint config differs")
        if raw["resource_budget"] != self.resource_budget.to_config():
            raise ValueError("calibrated search checkpoint resource declaration differs")
        state = raw["state"]
        if type(state) is not CalibratedExtendedSearchControlState:
            raise ValueError("calibrated search checkpoint state type differs")
        restored = state
        digest = _require_array(
            cast(Array, raw["state_digest"]),
            name="checkpoint.state_digest",
            shape=(32,),
            dtype=jnp.uint8,
        )
        if not bool(jax.device_get(jnp.array_equal(digest, self._state_checksum(restored)))):
            raise ValueError("calibrated search checkpoint state digest differs")
        generation = _int32_scalar(
            representation_generation, name="representation_generation"
        )
        source = _require_array(
            source_digest, name="source_digest", shape=(2,), dtype=jnp.uint32
        )
        descriptors = _require_array(
            option_descriptors,
            name="option_descriptors",
            shape=(self._config.n_options, 4),
            dtype=jnp.int32,
        )
        generations = _require_array(
            option_generations,
            name="option_generations",
            shape=(self._config.n_options,),
            dtype=jnp.int32,
        )
        learner = _int32_scalar(learner_revision, name="learner_revision")
        primitive_revision = _int32_scalar(
            primitive_model_revision, name="primitive_model_revision"
        )
        option_revision = _int32_scalar(
            option_model_revision, name="option_model_revision"
        )
        top_level_binding = (
            jnp.array_equal(cast(Array, raw["representation_generation"]), generation)
            & jnp.array_equal(cast(Array, raw["source_digest"]), source)
            & jnp.array_equal(cast(Array, raw["option_descriptors"]), descriptors)
            & jnp.array_equal(cast(Array, raw["option_generations"]), generations)
            & jnp.array_equal(cast(Array, raw["learner_revision"]), learner)
            & jnp.array_equal(
                cast(Array, raw["primitive_model_revision"]), primitive_revision
            )
            & jnp.array_equal(cast(Array, raw["option_model_revision"]), option_revision)
        )
        if not bool(jax.device_get(top_level_binding)):
            raise ValueError("calibrated search checkpoint live identity differs")
        valid = self.validate_state(
            restored,
            representation_generation=generation,
            source_digest=source,
            option_descriptors=descriptors,
            option_generations=generations,
        )
        if not bool(jax.device_get(valid)):
            raise ValueError("calibrated search checkpoint state is invalid or stale")
        return restored


__all__ = [
    "CALIBRATED_EXTENDED_SEARCH_CHECKPOINT_SCHEMA",
    "CALIBRATED_EXTENDED_SEARCH_CONFIG_SCHEMA",
    "CALIBRATED_EXTENDED_SEARCH_MECHANISM_STATUS",
    "CALIBRATED_EXTENDED_SEARCH_POLICY_AUTHORITY",
    "CALIBRATED_EXTENDED_SEARCH_RANKING_SEMANTICS",
    "CALIBRATED_EXTENDED_SEARCH_SCIENTIFIC_PROMOTION_ALLOWED",
    "CANDIDATE_KIND_OPTION",
    "CANDIDATE_KIND_PRIMITIVE",
    "SEARCH_MODE_COMBINED",
    "SEARCH_MODE_MODEL_FREE_EXTENDED_Q",
    "SEARCH_MODE_OPTION_MODEL",
    "SEARCH_MODE_PRIMITIVE_MODEL",
    "SEARCH_MODES",
    "CalibratedExtendedSearchArmDiagnostics",
    "CalibratedExtendedSearchArmResult",
    "CalibratedExtendedSearchControl",
    "CalibratedExtendedSearchControlConfig",
    "CalibratedExtendedSearchControlResourceBudget",
    "CalibratedExtendedSearchControlState",
    "CalibratedExtendedSearchObserveDiagnostics",
    "CalibratedExtendedSearchObserveResult",
]
