# mypy: disable-error-code="attr-defined,call-arg,arg-type,type-var"
"""Bounded, opt-in L0 auditing for an option lifecycle.

The auditor records lifecycle facts; it never installs, selects, trains, or
retires an option.  Every outcome is accepted through an exact two-phase
``arm``/``observe`` transaction.  The arm binds the transition identity,
source and representation, the complete option-semantic set and generations,
the state revision, randomized-comparator declaration, and predictions made
before the outcome/model update.  Invalid transactions are atomic no-ops.

Fixed-horizon primitive comparisons are admitted only when the caller declares
random assignment and supplies a bounded treatment propensity.  Every
configured context must independently meet both the treatment and primitive
evidence floors.  Contexts retain equal fixed mass, so an observed subset can
never be renormalized into a complete comparison.  Completion reasons are
recorded as independent flags (and may co-occur); a censor-only ending is kept
distinct and is excluded from completed-return, signature, and model-error
moments.

``maintenance_report`` is a bounded proposal report, not curation authority.
``rebind`` preserves a slot only for a bit-identical semantic digest under the
same source and representation.  A genuinely changed semantic resets every
slot-local statistic and increments its generation.  Rebinding is deferred
while either an option execution or comparator trial is in flight.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any, ClassVar, cast

import chex
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import Bool, Float, Int, UInt

OPTION_LIFECYCLE_AUDIT_CONFIG_SCHEMA = "alberta.option-lifecycle-audit.config.v1"
OPTION_LIFECYCLE_AUDIT_CHECKPOINT_SCHEMA = "alberta.option-lifecycle-audit.state.v1"
OPTION_LIFECYCLE_AUDIT_MECHANISM_STATUS = "development_mechanism_only"
OPTION_LIFECYCLE_AUDIT_CURATION_AUTHORITY = False
OPTION_LIFECYCLE_AUDIT_PROMOTION_AUTHORITY = False
OPTION_LIFECYCLE_AUDIT_GO_NO_GO_AUTHORITY = False
OPTION_LIFECYCLE_AUDIT_SCIENTIFIC_PROMOTION_ALLOWED = False

_DIGEST_WORDS = 8
_TRANSITION_WORDS = 2
_INT32_MAX = 2_147_483_647
_MAX_OPTIONS = 1_024
_MAX_CONTEXTS = 4_096
_MAX_OUTCOME_DIM = 4_096
_MAX_PAIR_CELLS = 1_048_576
_MAX_STATE_CELLS = 8_388_608


def _positive_int(value: object, *, name: str, ceiling: int = _INT32_MAX) -> int:
    if type(value) is not int or not 1 <= value <= ceiling:
        raise ValueError(f"{name} must be an exact Python int in [1, {ceiling}]")
    return value


def _nonnegative_int(value: object, *, name: str, ceiling: int = _INT32_MAX) -> int:
    if type(value) is not int or not 0 <= value <= ceiling:
        raise ValueError(f"{name} must be an exact Python int in [0, {ceiling}]")
    return value


def _finite_float(
    value: object,
    *,
    name: str,
    lower: float | None = None,
    upper: float | None = None,
) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite exact Python float")
    represented = float(np.float32(value))
    if not math.isfinite(represented):
        raise ValueError(f"{name} must remain finite in float32")
    if lower is not None and value < lower:
        raise ValueError(f"{name} must be at least {lower}")
    if upper is not None and value > upper:
        raise ValueError(f"{name} must be at most {upper}")
    return value


def _float_tuple(value: object, *, name: str, length: int) -> tuple[float, ...]:
    if type(value) not in (tuple, list) or len(cast(Any, value)) != length:
        raise ValueError(f"{name} must contain exactly {length} floats")
    raw = cast(tuple[object, ...] | list[object], value)
    if any(type(cell) is not float for cell in raw):
        raise ValueError(f"{name} must contain exact Python floats")
    return tuple(cast(float, cell) for cell in raw)


def _require_array(value: Any, *, name: str, shape: tuple[int, ...], dtype: Any) -> Array:
    if not hasattr(value, "shape") or not hasattr(value, "dtype"):
        raise TypeError(f"{name} must be an array with exact shape and dtype")
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
        return jnp.asarray(value, dtype=jnp.int32)
    return _require_array(value, name=name, shape=(), dtype=jnp.int32)


def _float32_scalar(value: float | Array, *, name: str) -> Array:
    if type(value) is float:
        if not math.isfinite(value) or not math.isfinite(float(np.float32(value))):
            raise ValueError(f"{name} must be finite and float32 representable")
        return jnp.asarray(value, dtype=jnp.float32)
    return _require_array(value, name=name, shape=(), dtype=jnp.float32)


def _bool_scalar(value: bool | Array, *, name: str) -> Array:
    if type(value) is bool:
        return jnp.asarray(value, dtype=jnp.bool_)
    return _require_array(value, name=name, shape=(), dtype=jnp.bool_)


def _canonical_digest(value: object) -> tuple[int, ...]:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    raw = hashlib.sha256(canonical.encode("utf-8")).digest()
    return tuple(int.from_bytes(raw[offset : offset + 4], "little") for offset in range(0, 32, 4))


def option_semantic_digest(descriptor: object) -> UInt[Array, " 8"]:
    """Return a canonical SHA-256 semantic digest as eight uint32 words."""

    return jnp.asarray(_canonical_digest(descriptor), dtype=jnp.uint32)


def _checksum_arrays(arrays: tuple[Array, ...]) -> Array:
    """Return a deterministic two-word in-JAX integrity checksum."""

    acc0 = jnp.uint32(0x9E3779B9)
    acc1 = jnp.uint32(0x85EBCA6B)
    offset = 1
    for value in arrays:
        array = jnp.asarray(value)
        if array.dtype == jnp.float32:
            words = jax.lax.bitcast_convert_type(array, jnp.uint32).reshape((-1,))
        elif array.dtype == jnp.uint32:
            words = array.reshape((-1,))
        else:
            words = array.astype(jnp.uint32).reshape((-1,))
        if words.shape[0] == 0:
            continue
        indices = jnp.arange(offset, offset + words.shape[0], dtype=jnp.uint32)
        acc0 = acc0 + jnp.sum(words * (indices * jnp.uint32(0x27D4EB2D) + 1))
        acc1 = acc1 ^ jnp.bitwise_xor.reduce(words ^ (indices * jnp.uint32(0x165667B1)))
        offset += words.shape[0]
    return jnp.stack((acc0, acc1), dtype=jnp.uint32)


@dataclasses.dataclass(frozen=True, slots=True)
class OptionLifecycleAuditConfig:
    """Frozen evidence, work, and maintenance semantics for one auditor."""

    n_options: int
    n_contexts: int
    outcome_dim: int
    fixed_horizon: int
    maintenance_budget: int
    signature_scales: tuple[float, ...]
    initiation_opportunity_floor: int = 2
    completion_evidence_floor: int = 2
    model_error_evidence_floor: int = 2
    comparison_treatment_evidence_floor: int = 2
    comparison_primitive_evidence_floor: int = 2
    signature_evidence_floor_per_context: int = 2
    redundancy_shared_context_floor: int = 1
    min_initiation_coverage: float = 0.0
    min_completion_reliability: float = 0.0
    min_marginal_improvement: float = 0.0
    max_normalized_model_rmse: float = 1_000_000.0
    min_planning_uses: int = 0
    max_mean_compute_cost: float = 1_000_000.0
    max_resident_memory_bytes: int = _INT32_MAX
    redundancy_distance_threshold: float = 0.05
    propensity_floor: float = 0.05
    propensity_ceiling: float = 0.95
    max_planning_uses_per_observation: int = 4_096
    max_compute_cost_per_observation: float = 1_000_000.0
    max_observations: int = 100_000

    SCHEMA_VERSION: ClassVar[str] = OPTION_LIFECYCLE_AUDIT_CONFIG_SCHEMA

    def __post_init__(self) -> None:
        _positive_int(self.n_options, name="n_options", ceiling=_MAX_OPTIONS)
        _positive_int(self.n_contexts, name="n_contexts", ceiling=_MAX_CONTEXTS)
        _positive_int(self.outcome_dim, name="outcome_dim", ceiling=_MAX_OUTCOME_DIM)
        _positive_int(self.fixed_horizon, name="fixed_horizon")
        _positive_int(self.maintenance_budget, name="maintenance_budget", ceiling=self.n_options)
        if type(self.signature_scales) is not tuple:
            raise ValueError("signature_scales must be an exact tuple")
        scales = _float_tuple(
            self.signature_scales,
            name="signature_scales",
            length=self.signature_dim,
        )
        for index, scale in enumerate(scales):
            _finite_float(
                scale,
                name=f"signature_scales[{index}]",
                lower=float(np.finfo(np.float32).tiny),
            )

        floors = (
            "initiation_opportunity_floor",
            "completion_evidence_floor",
            "model_error_evidence_floor",
            "comparison_treatment_evidence_floor",
            "comparison_primitive_evidence_floor",
            "signature_evidence_floor_per_context",
            "redundancy_shared_context_floor",
        )
        for name in floors:
            _positive_int(getattr(self, name), name=name)
        _nonnegative_int(self.min_planning_uses, name="min_planning_uses")
        _positive_int(
            self.max_planning_uses_per_observation,
            name="max_planning_uses_per_observation",
            ceiling=_MAX_CONTEXTS,
        )
        _positive_int(self.max_observations, name="max_observations")
        if max(getattr(self, name) for name in floors[:-1]) > self.max_observations:
            raise ValueError("per-option evidence floors must not exceed max_observations")
        if self.fixed_horizon > self.max_observations:
            raise ValueError("fixed_horizon must not exceed max_observations")
        if self.max_observations * self.max_planning_uses_per_observation > _INT32_MAX:
            raise ValueError("planning-use counter ceiling would exceed signed int32")
        _nonnegative_int(
            self.max_resident_memory_bytes,
            name="max_resident_memory_bytes",
        )

        for name in (
            "min_initiation_coverage",
            "min_completion_reliability",
        ):
            _finite_float(getattr(self, name), name=name, lower=0.0, upper=1.0)
        for name in (
            "min_marginal_improvement",
            "max_normalized_model_rmse",
            "max_mean_compute_cost",
            "redundancy_distance_threshold",
            "max_compute_cost_per_observation",
        ):
            _finite_float(getattr(self, name), name=name, lower=0.0)
        _finite_float(self.propensity_floor, name="propensity_floor", lower=0.0, upper=1.0)
        _finite_float(
            self.propensity_ceiling,
            name="propensity_ceiling",
            lower=0.0,
            upper=1.0,
        )
        if not 0.0 < self.propensity_floor <= self.propensity_ceiling < 1.0:
            raise ValueError("propensity bounds must satisfy 0 < floor <= ceiling < 1")

        pair_cells = self.n_options * self.n_options
        if pair_cells > _MAX_PAIR_CELLS:
            raise ValueError("n_options squared exceeds the fixed pair-cell ceiling")
        if self._persistent_logical_cells() > _MAX_STATE_CELLS:
            raise ValueError("configured lifecycle state exceeds the fixed cell ceiling")

    @property
    def signature_dim(self) -> int:
        """External, pseudo, duration, baseline, discount, then outcome delta."""

        return 5 + self.outcome_dim

    def _persistent_logical_cells(self) -> int:
        n = self.n_options
        c = self.n_contexts
        d = self.outcome_dim
        s = self.signature_dim
        # Bindings, active/trial caches, counters, moments, context signatures,
        # costs, and the checksum.  This is exact for OptionLifecycleAuditState.
        return 46 + d + s + 24 * n + 13 * n * c + 4 * n * s + n * c * s

    def to_config(self) -> dict[str, object]:
        payload = dataclasses.asdict(self)
        payload["schema_version"] = self.SCHEMA_VERSION
        payload["mechanism_status"] = OPTION_LIFECYCLE_AUDIT_MECHANISM_STATUS
        payload["scientific_promotion_allowed"] = False
        return payload

    @classmethod
    def from_config(cls, value: Mapping[str, object]) -> OptionLifecycleAuditConfig:
        if type(value) is not dict:
            raise ValueError("option lifecycle config must be an exact dict")
        raw = dict(value)
        expected = {field.name for field in dataclasses.fields(cls)} | {
            "schema_version",
            "mechanism_status",
            "scientific_promotion_allowed",
        }
        if set(raw) != expected:
            raise ValueError("option lifecycle config keys differ from schema v1")
        if raw.pop("schema_version") != cls.SCHEMA_VERSION:
            raise ValueError("option lifecycle config schema_version differs")
        if raw.pop("mechanism_status") != OPTION_LIFECYCLE_AUDIT_MECHANISM_STATUS:
            raise ValueError("option lifecycle config must remain mechanism-only")
        if raw.pop("scientific_promotion_allowed") is not False:
            raise ValueError("option lifecycle config cannot claim scientific promotion")
        outcome_dim = raw.get("outcome_dim")
        if type(outcome_dim) is not int:
            raise ValueError("serialized outcome_dim must be an exact integer")
        raw["signature_scales"] = _float_tuple(
            raw["signature_scales"],
            name="signature_scales",
            length=5 + outcome_dim,
        )
        return cls(**cast(dict[str, Any], raw))


@chex.dataclass(frozen=True)
class OptionLifecycleAuditState:
    """Fixed-shape persistent facts for one semantic/source binding."""

    source_digest: UInt[Array, " 8"]
    representation_digest: UInt[Array, " 8"]
    config_digest: UInt[Array, " 8"]
    semantic_digests: UInt[Array, "n_options 8"]
    semantic_generations: Int[Array, " n_options"]
    revision: Int[Array, ""]
    observation_count: Int[Array, ""]
    has_last_transition: Bool[Array, ""]
    last_transition_id: UInt[Array, " 2"]
    active_option: Int[Array, ""]
    active_context: Int[Array, ""]
    active_generation: Int[Array, ""]
    active_steps: Int[Array, ""]
    active_external_return: Float[Array, ""]
    active_pseudo_return: Float[Array, ""]
    active_baseline_mass: Float[Array, ""]
    active_discount: Float[Array, ""]
    active_outcome_delta: Float[Array, " outcome_dim"]
    active_model_prediction: Float[Array, " signature_dim"]
    trial_active: Bool[Array, ""]
    trial_option: Int[Array, ""]
    trial_context: Int[Array, ""]
    trial_treatment: Bool[Array, ""]
    trial_propensity: Float[Array, ""]
    trial_steps: Int[Array, ""]
    trial_return: Float[Array, ""]
    initiation_opportunities: Int[Array, "n_options n_contexts"]
    initiation_starts: Int[Array, "n_options n_contexts"]
    execution_starts: Int[Array, " n_options"]
    natural_completions: Int[Array, " n_options"]
    goal_terminations: Int[Array, " n_options"]
    timeout_terminations: Int[Array, " n_options"]
    environment_terminations: Int[Array, " n_options"]
    censored_endings: Int[Array, " n_options"]
    censor_only_endings: Int[Array, " n_options"]
    completion_moment_counts: Int[Array, " n_options"]
    completion_signature_sums: Float[Array, "n_options signature_dim"]
    completion_signature_squared_sums: Float[Array, "n_options signature_dim"]
    model_error_counts: Int[Array, " n_options"]
    model_absolute_error_sums: Float[Array, "n_options signature_dim"]
    model_squared_error_sums: Float[Array, "n_options signature_dim"]
    context_signature_counts: Int[Array, "n_options n_contexts"]
    context_signature_sums: Float[Array, "n_options n_contexts signature_dim"]
    comparison_treatment_counts: Int[Array, "n_options n_contexts"]
    comparison_primitive_counts: Int[Array, "n_options n_contexts"]
    comparison_treatment_return_sums: Float[Array, "n_options n_contexts"]
    comparison_treatment_return_squared_sums: Float[Array, "n_options n_contexts"]
    comparison_primitive_return_sums: Float[Array, "n_options n_contexts"]
    comparison_primitive_return_squared_sums: Float[Array, "n_options n_contexts"]
    comparison_treatment_ipw_reward_sums: Float[Array, "n_options n_contexts"]
    comparison_treatment_ipw_masses: Float[Array, "n_options n_contexts"]
    comparison_primitive_ipw_reward_sums: Float[Array, "n_options n_contexts"]
    comparison_primitive_ipw_masses: Float[Array, "n_options n_contexts"]
    planning_use_counts: Int[Array, " n_options"]
    planning_decision_counts: Int[Array, " n_options"]
    compute_observation_counts: Int[Array, " n_options"]
    compute_cost_sums: Float[Array, " n_options"]
    compute_cost_squared_sums: Float[Array, " n_options"]
    resident_memory_max_bytes: Int[Array, " n_options"]
    state_checksum: UInt[Array, " 2"]


@chex.dataclass(frozen=True)
class OptionLifecycleAuditArm:
    """Frozen pre-outcome cache for one exact transition."""

    available: Bool[Array, ""]
    transition_id: UInt[Array, " 2"]
    source_digest: UInt[Array, " 8"]
    representation_digest: UInt[Array, " 8"]
    config_digest: UInt[Array, " 8"]
    semantic_digests: UInt[Array, "n_options 8"]
    semantic_generations: Int[Array, " n_options"]
    state_revision: Int[Array, ""]
    state_checksum: UInt[Array, " 2"]
    candidate_option: Int[Array, ""]
    initiation_context: Int[Array, ""]
    initiation_eligible: Bool[Array, ""]
    owner_option: Int[Array, ""]
    starts_execution: Bool[Array, ""]
    comparator_randomized: Bool[Array, ""]
    treatment_propensity: Float[Array, ""]
    frozen_model_prediction: Float[Array, " signature_dim"]
    cache_checksum: UInt[Array, " 2"]


@chex.dataclass(frozen=True)
class OptionLifecycleAuditResult:
    """Atomic next state and transaction facts."""

    state: OptionLifecycleAuditState
    transaction_valid: Bool[Array, ""]
    applied: Bool[Array, ""]
    execution_started: Bool[Array, ""]
    execution_completed: Bool[Array, ""]
    censor_only_ending: Bool[Array, ""]
    comparator_trial_completed: Bool[Array, ""]
    model_error_scored: Bool[Array, ""]


@chex.dataclass(frozen=True)
class OptionLifecycleMaintenanceReport:
    """Bounded, proposal-only lifecycle scorecard."""

    state_valid: Bool[Array, ""]
    state_revision: Int[Array, ""]
    initiation_coverage: Float[Array, " n_options"]
    initiation_evidence_ready: Bool[Array, " n_options"]
    completion_reliability: Float[Array, " n_options"]
    completion_evidence_ready: Bool[Array, " n_options"]
    external_return_means: Float[Array, " n_options"]
    pseudo_return_means: Float[Array, " n_options"]
    treatment_return_means: Float[Array, " n_options"]
    primitive_return_means: Float[Array, " n_options"]
    marginal_improvement: Float[Array, " n_options"]
    inverse_propensity_marginal_improvement: Float[Array, " n_options"]
    treatment_return_means_by_context: Float[Array, "n_options n_contexts"]
    primitive_return_means_by_context: Float[Array, "n_options n_contexts"]
    marginal_improvement_by_context: Float[Array, "n_options n_contexts"]
    inverse_propensity_marginal_improvement_by_context: Float[
        Array, "n_options n_contexts"
    ]
    comparison_ready: Bool[Array, " n_options"]
    normalized_model_rmse: Float[Array, "n_options signature_dim"]
    model_evidence_ready: Bool[Array, " n_options"]
    planning_use_counts: Int[Array, " n_options"]
    mean_compute_cost: Float[Array, " n_options"]
    resident_memory_max_bytes: Int[Array, " n_options"]
    shared_context_counts: Int[Array, "n_options n_options"]
    redundancy_distances: Float[Array, "n_options n_options"]
    redundancy_ready: Bool[Array, "n_options n_options"]
    redundant_pairs: Bool[Array, "n_options n_options"]
    redundancy_loser: Bool[Array, " n_options"]
    all_required_evidence_ready: Bool[Array, " n_options"]
    concern_counts: Int[Array, " n_options"]
    replacement_eligible: Bool[Array, " n_options"]
    proposed_replacement_slots: Int[Array, " maintenance_budget"]
    proposed_replacement_mask: Bool[Array, " maintenance_budget"]
    curation_authority: Bool[Array, ""]
    promotion_authority: Bool[Array, ""]
    go_no_go_authority: Bool[Array, ""]


@chex.dataclass(frozen=True)
class OptionLifecycleRebindResult:
    """Atomic semantic rebind/reset result."""

    state: OptionLifecycleAuditState
    transaction_valid: Bool[Array, ""]
    applied: Bool[Array, ""]
    deferred: Bool[Array, ""]
    preserved_slots: Bool[Array, " n_options"]
    reset_slots: Bool[Array, " n_options"]


@dataclasses.dataclass(frozen=True, slots=True)
class OptionLifecycleAuditResourceBudget:
    """Exact static allocation/work declarations for version 1."""

    persistent_logical_scalars: int
    persistent_state_nbytes: int
    option_slots: int
    context_slots: int
    signature_channels: int
    pair_cells_per_maintenance_report: int
    maintenance_proposal_slots: int
    max_observations: int
    max_planning_uses_per_observation: int
    state_checksum_cells_per_validation: int
    state_validation_calls_per_arm: int
    state_validation_calls_per_observe: int
    state_validation_calls_per_maintenance_report: int
    state_validation_calls_per_rebind: int
    max_option_counter_updates_per_observe: int
    max_signature_channel_updates_per_observe: int
    max_pair_distance_evaluations_per_maintenance_report: int
    max_slots_examined_per_rebind: int
    rng_draws_at_init: int
    rng_draws_per_arm: int
    rng_draws_per_observe: int
    backward_passes_per_observe: int
    consumer_calls_per_observe: int
    option_updates_per_observe: int
    policy_updates_per_observe: int
    model_updates_per_observe: int
    curation_decisions_per_report: int
    curation_authority: bool
    promotion_authority: bool
    go_no_go_authority: bool
    scientific_promotion_allowed: bool
    checkpoint_schema: str


class OptionLifecycleAudit:
    """Fixed-resource option lifecycle auditor with no control authority."""

    def __init__(self, config: OptionLifecycleAuditConfig) -> None:
        if type(config) is not OptionLifecycleAuditConfig:
            raise TypeError("config must be an exact OptionLifecycleAuditConfig")
        self._config = config
        self._config_digest = jnp.asarray(
            _canonical_digest(config.to_config()),
            dtype=jnp.uint32,
        )
        self._signature_scales = jnp.asarray(config.signature_scales, dtype=jnp.float32)

    @property
    def config(self) -> OptionLifecycleAuditConfig:
        return self._config

    def to_config(self) -> dict[str, object]:
        return self._config.to_config()

    @classmethod
    def from_config(cls, value: Mapping[str, object]) -> OptionLifecycleAudit:
        return cls(OptionLifecycleAuditConfig.from_config(value))

    @property
    def resource_budget(self) -> OptionLifecycleAuditResourceBudget:
        """Return exact allocation counts and zero-authority declarations."""

        dummy_digest = jnp.arange(_DIGEST_WORDS, dtype=jnp.uint32) + jnp.uint32(1)
        semantics = jnp.arange(
            1,
            self._config.n_options * _DIGEST_WORDS + 1,
            dtype=jnp.uint32,
        ).reshape((self._config.n_options, _DIGEST_WORDS))
        state = self.init(
            source_digest=dummy_digest,
            representation_digest=dummy_digest + jnp.uint32(17),
            semantic_digests=semantics,
        )
        leaves = jax.tree_util.tree_leaves(state)
        logical = sum(int(np.asarray(leaf).size) for leaf in leaves)
        nbytes = sum(int(np.asarray(leaf).nbytes) for leaf in leaves)
        return OptionLifecycleAuditResourceBudget(
            persistent_logical_scalars=logical,
            persistent_state_nbytes=nbytes,
            option_slots=self._config.n_options,
            context_slots=self._config.n_contexts,
            signature_channels=self._config.signature_dim,
            pair_cells_per_maintenance_report=self._config.n_options**2,
            maintenance_proposal_slots=self._config.maintenance_budget,
            max_observations=self._config.max_observations,
            max_planning_uses_per_observation=(
                self._config.max_planning_uses_per_observation
            ),
            state_checksum_cells_per_validation=logical - _TRANSITION_WORDS,
            state_validation_calls_per_arm=1,
            state_validation_calls_per_observe=2,
            state_validation_calls_per_maintenance_report=1,
            state_validation_calls_per_rebind=2,
            max_option_counter_updates_per_observe=self._config.n_options,
            max_signature_channel_updates_per_observe=self._config.signature_dim,
            max_pair_distance_evaluations_per_maintenance_report=(
                self._config.n_options**2 * self._config.n_contexts
            ),
            max_slots_examined_per_rebind=self._config.n_options,
            rng_draws_at_init=0,
            rng_draws_per_arm=0,
            rng_draws_per_observe=0,
            backward_passes_per_observe=0,
            consumer_calls_per_observe=0,
            option_updates_per_observe=0,
            policy_updates_per_observe=0,
            model_updates_per_observe=0,
            curation_decisions_per_report=0,
            curation_authority=False,
            promotion_authority=False,
            go_no_go_authority=False,
            scientific_promotion_allowed=False,
            checkpoint_schema=OPTION_LIFECYCLE_AUDIT_CHECKPOINT_SCHEMA,
        )

    def _check_digest_set(
        self,
        source_digest: Any,
        representation_digest: Any,
        semantic_digests: Any,
        semantic_generations: Any,
    ) -> tuple[Array, Array, Array, Array]:
        cfg = self._config
        source = _require_array(
            source_digest,
            name="source_digest",
            shape=(_DIGEST_WORDS,),
            dtype=jnp.uint32,
        )
        representation = _require_array(
            representation_digest,
            name="representation_digest",
            shape=(_DIGEST_WORDS,),
            dtype=jnp.uint32,
        )
        semantics = _require_array(
            semantic_digests,
            name="semantic_digests",
            shape=(cfg.n_options, _DIGEST_WORDS),
            dtype=jnp.uint32,
        )
        generations = _require_array(
            semantic_generations,
            name="semantic_generations",
            shape=(cfg.n_options,),
            dtype=jnp.int32,
        )
        return source, representation, semantics, generations

    def _state_payload_arrays(self, state: OptionLifecycleAuditState) -> tuple[Array, ...]:
        return tuple(
            cast(Array, getattr(state, field.name))
            for field in dataclasses.fields(OptionLifecycleAuditState)
            if field.name != "state_checksum"
        )

    def _with_checksum(self, state: OptionLifecycleAuditState) -> OptionLifecycleAuditState:
        return dataclasses.replace(
            state,
            state_checksum=_checksum_arrays(self._state_payload_arrays(state)),
        )

    def _arm_payload_arrays(self, arm: OptionLifecycleAuditArm) -> tuple[Array, ...]:
        return tuple(
            cast(Array, getattr(arm, field.name))
            for field in dataclasses.fields(OptionLifecycleAuditArm)
            if field.name != "cache_checksum"
        )

    def _state_specifications(self) -> dict[str, tuple[tuple[int, ...], Any]]:
        cfg = self._config
        n, c, d, s = cfg.n_options, cfg.n_contexts, cfg.outcome_dim, cfg.signature_dim
        return {
            "source_digest": ((_DIGEST_WORDS,), jnp.uint32),
            "representation_digest": ((_DIGEST_WORDS,), jnp.uint32),
            "config_digest": ((_DIGEST_WORDS,), jnp.uint32),
            "semantic_digests": ((n, _DIGEST_WORDS), jnp.uint32),
            "semantic_generations": ((n,), jnp.int32),
            "revision": ((), jnp.int32),
            "observation_count": ((), jnp.int32),
            "has_last_transition": ((), jnp.bool_),
            "last_transition_id": ((_TRANSITION_WORDS,), jnp.uint32),
            "active_option": ((), jnp.int32),
            "active_context": ((), jnp.int32),
            "active_generation": ((), jnp.int32),
            "active_steps": ((), jnp.int32),
            "active_external_return": ((), jnp.float32),
            "active_pseudo_return": ((), jnp.float32),
            "active_baseline_mass": ((), jnp.float32),
            "active_discount": ((), jnp.float32),
            "active_outcome_delta": ((d,), jnp.float32),
            "active_model_prediction": ((s,), jnp.float32),
            "trial_active": ((), jnp.bool_),
            "trial_option": ((), jnp.int32),
            "trial_context": ((), jnp.int32),
            "trial_treatment": ((), jnp.bool_),
            "trial_propensity": ((), jnp.float32),
            "trial_steps": ((), jnp.int32),
            "trial_return": ((), jnp.float32),
            "initiation_opportunities": ((n, c), jnp.int32),
            "initiation_starts": ((n, c), jnp.int32),
            "execution_starts": ((n,), jnp.int32),
            "natural_completions": ((n,), jnp.int32),
            "goal_terminations": ((n,), jnp.int32),
            "timeout_terminations": ((n,), jnp.int32),
            "environment_terminations": ((n,), jnp.int32),
            "censored_endings": ((n,), jnp.int32),
            "censor_only_endings": ((n,), jnp.int32),
            "completion_moment_counts": ((n,), jnp.int32),
            "completion_signature_sums": ((n, s), jnp.float32),
            "completion_signature_squared_sums": ((n, s), jnp.float32),
            "model_error_counts": ((n,), jnp.int32),
            "model_absolute_error_sums": ((n, s), jnp.float32),
            "model_squared_error_sums": ((n, s), jnp.float32),
            "context_signature_counts": ((n, c), jnp.int32),
            "context_signature_sums": ((n, c, s), jnp.float32),
            "comparison_treatment_counts": ((n, c), jnp.int32),
            "comparison_primitive_counts": ((n, c), jnp.int32),
            "comparison_treatment_return_sums": ((n, c), jnp.float32),
            "comparison_treatment_return_squared_sums": ((n, c), jnp.float32),
            "comparison_primitive_return_sums": ((n, c), jnp.float32),
            "comparison_primitive_return_squared_sums": ((n, c), jnp.float32),
            "comparison_treatment_ipw_reward_sums": ((n, c), jnp.float32),
            "comparison_treatment_ipw_masses": ((n, c), jnp.float32),
            "comparison_primitive_ipw_reward_sums": ((n, c), jnp.float32),
            "comparison_primitive_ipw_masses": ((n, c), jnp.float32),
            "planning_use_counts": ((n,), jnp.int32),
            "planning_decision_counts": ((n,), jnp.int32),
            "compute_observation_counts": ((n,), jnp.int32),
            "compute_cost_sums": ((n,), jnp.float32),
            "compute_cost_squared_sums": ((n,), jnp.float32),
            "resident_memory_max_bytes": ((n,), jnp.int32),
            "state_checksum": ((_TRANSITION_WORDS,), jnp.uint32),
        }

    def _check_state_contract(self, state: OptionLifecycleAuditState) -> None:
        if type(state) is not OptionLifecycleAuditState:
            raise TypeError("state must be an exact OptionLifecycleAuditState")
        for name, (shape, dtype) in self._state_specifications().items():
            _require_array(getattr(state, name), name=f"state.{name}", shape=shape, dtype=dtype)

    def _check_arm_contract(self, arm: OptionLifecycleAuditArm) -> None:
        if type(arm) is not OptionLifecycleAuditArm:
            raise TypeError("arm must be an exact OptionLifecycleAuditArm")
        cfg = self._config
        specs: dict[str, tuple[tuple[int, ...], Any]] = {
            "available": ((), jnp.bool_),
            "transition_id": ((_TRANSITION_WORDS,), jnp.uint32),
            "source_digest": ((_DIGEST_WORDS,), jnp.uint32),
            "representation_digest": ((_DIGEST_WORDS,), jnp.uint32),
            "config_digest": ((_DIGEST_WORDS,), jnp.uint32),
            "semantic_digests": ((cfg.n_options, _DIGEST_WORDS), jnp.uint32),
            "semantic_generations": ((cfg.n_options,), jnp.int32),
            "state_revision": ((), jnp.int32),
            "state_checksum": ((_TRANSITION_WORDS,), jnp.uint32),
            "candidate_option": ((), jnp.int32),
            "initiation_context": ((), jnp.int32),
            "initiation_eligible": ((), jnp.bool_),
            "owner_option": ((), jnp.int32),
            "starts_execution": ((), jnp.bool_),
            "comparator_randomized": ((), jnp.bool_),
            "treatment_propensity": ((), jnp.float32),
            "frozen_model_prediction": ((cfg.signature_dim,), jnp.float32),
            "cache_checksum": ((_TRANSITION_WORDS,), jnp.uint32),
        }
        for name, (shape, dtype) in specs.items():
            _require_array(getattr(arm, name), name=f"arm.{name}", shape=shape, dtype=dtype)

    def init(
        self,
        *,
        source_digest: Array,
        representation_digest: Array,
        semantic_digests: Array,
        semantic_generations: Array | None = None,
    ) -> OptionLifecycleAuditState:
        """Initialize one exact semantic/source binding without RNG."""

        cfg = self._config
        generations_value = (
            jnp.zeros((cfg.n_options,), dtype=jnp.int32)
            if semantic_generations is None
            else semantic_generations
        )
        source, representation, semantics, generations = self._check_digest_set(
            source_digest,
            representation_digest,
            semantic_digests,
            generations_value,
        )
        host_semantics = np.asarray(jax.device_get(semantics), dtype=np.uint32)
        host_generations = np.asarray(jax.device_get(generations), dtype=np.int32)
        if np.any(np.all(host_semantics == 0, axis=1)):
            raise ValueError("semantic digests must be nonzero")
        if len({row.tobytes() for row in host_semantics}) != cfg.n_options:
            raise ValueError("semantic digests must be unique across option slots")
        if np.any(host_generations < 0):
            raise ValueError("semantic generations must be non-negative")
        if not np.any(np.asarray(jax.device_get(source))):
            raise ValueError("source_digest must be nonzero")
        if not np.any(np.asarray(jax.device_get(representation))):
            raise ValueError("representation_digest must be nonzero")

        n, c, d, s = cfg.n_options, cfg.n_contexts, cfg.outcome_dim, cfg.signature_dim
        def zi(shape: tuple[int, ...]) -> Array:
            return jnp.zeros(shape, dtype=jnp.int32)

        def zf(shape: tuple[int, ...]) -> Array:
            return jnp.zeros(shape, dtype=jnp.float32)
        state = OptionLifecycleAuditState(
            source_digest=source,
            representation_digest=representation,
            config_digest=self._config_digest,
            semantic_digests=semantics,
            semantic_generations=generations,
            revision=zi(()),
            observation_count=zi(()),
            has_last_transition=jnp.asarray(False, dtype=jnp.bool_),
            last_transition_id=jnp.zeros((_TRANSITION_WORDS,), dtype=jnp.uint32),
            active_option=jnp.asarray(-1, dtype=jnp.int32),
            active_context=jnp.asarray(-1, dtype=jnp.int32),
            active_generation=jnp.asarray(-1, dtype=jnp.int32),
            active_steps=zi(()),
            active_external_return=zf(()),
            active_pseudo_return=zf(()),
            active_baseline_mass=zf(()),
            active_discount=jnp.asarray(1.0, dtype=jnp.float32),
            active_outcome_delta=zf((d,)),
            active_model_prediction=zf((s,)),
            trial_active=jnp.asarray(False, dtype=jnp.bool_),
            trial_option=jnp.asarray(-1, dtype=jnp.int32),
            trial_context=jnp.asarray(-1, dtype=jnp.int32),
            trial_treatment=jnp.asarray(False, dtype=jnp.bool_),
            trial_propensity=zf(()),
            trial_steps=zi(()),
            trial_return=zf(()),
            initiation_opportunities=zi((n, c)),
            initiation_starts=zi((n, c)),
            execution_starts=zi((n,)),
            natural_completions=zi((n,)),
            goal_terminations=zi((n,)),
            timeout_terminations=zi((n,)),
            environment_terminations=zi((n,)),
            censored_endings=zi((n,)),
            censor_only_endings=zi((n,)),
            completion_moment_counts=zi((n,)),
            completion_signature_sums=zf((n, s)),
            completion_signature_squared_sums=zf((n, s)),
            model_error_counts=zi((n,)),
            model_absolute_error_sums=zf((n, s)),
            model_squared_error_sums=zf((n, s)),
            context_signature_counts=zi((n, c)),
            context_signature_sums=zf((n, c, s)),
            comparison_treatment_counts=zi((n, c)),
            comparison_primitive_counts=zi((n, c)),
            comparison_treatment_return_sums=zf((n, c)),
            comparison_treatment_return_squared_sums=zf((n, c)),
            comparison_primitive_return_sums=zf((n, c)),
            comparison_primitive_return_squared_sums=zf((n, c)),
            comparison_treatment_ipw_reward_sums=zf((n, c)),
            comparison_treatment_ipw_masses=zf((n, c)),
            comparison_primitive_ipw_reward_sums=zf((n, c)),
            comparison_primitive_ipw_masses=zf((n, c)),
            planning_use_counts=zi((n,)),
            planning_decision_counts=zi((n,)),
            compute_observation_counts=zi((n,)),
            compute_cost_sums=zf((n,)),
            compute_cost_squared_sums=zf((n,)),
            resident_memory_max_bytes=zi((n,)),
            state_checksum=jnp.zeros((_TRANSITION_WORDS,), dtype=jnp.uint32),
        )
        return self._with_checksum(state)

    def state_valid(self, state: OptionLifecycleAuditState) -> Bool[Array, ""]:
        """Validate an exact persistent audit state through the public boundary."""

        self._check_state_contract(state)
        return self._state_valid(state)

    def _state_valid(self, state: OptionLifecycleAuditState) -> Bool[Array, ""]:
        cfg = self._config
        valid = jnp.array_equal(state.config_digest, self._config_digest)
        valid = valid & jnp.array_equal(
            state.state_checksum,
            _checksum_arrays(self._state_payload_arrays(state)),
        )
        valid = valid & (state.revision >= state.observation_count)
        valid = valid & (state.revision >= 0)
        valid = valid & (state.observation_count >= 0)
        valid = valid & (state.observation_count <= cfg.max_observations)
        valid = valid & jnp.all(state.semantic_generations >= 0)
        valid = valid & jnp.all(jnp.any(state.semantic_digests != 0, axis=1))
        valid = valid & jnp.any(state.source_digest != 0)
        valid = valid & jnp.any(state.representation_digest != 0)
        semantic_equality = jnp.all(
            state.semantic_digests[:, None, :] == state.semantic_digests[None, :, :],
            axis=2,
        )
        valid = valid & jnp.all(
            semantic_equality == jnp.eye(cfg.n_options, dtype=jnp.bool_)
        )

        counter_names = (
            "initiation_opportunities",
            "initiation_starts",
            "execution_starts",
            "natural_completions",
            "goal_terminations",
            "timeout_terminations",
            "environment_terminations",
            "censored_endings",
            "censor_only_endings",
            "completion_moment_counts",
            "model_error_counts",
            "context_signature_counts",
            "comparison_treatment_counts",
            "comparison_primitive_counts",
            "planning_use_counts",
            "planning_decision_counts",
            "compute_observation_counts",
            "resident_memory_max_bytes",
        )
        for name in counter_names:
            value = cast(Array, getattr(state, name))
            valid = valid & jnp.all(value >= 0)
            if name not in ("planning_use_counts", "resident_memory_max_bytes"):
                valid = valid & jnp.all(value <= cfg.max_observations)
        valid = valid & jnp.all(
            state.planning_use_counts
            <= cfg.max_observations * cfg.max_planning_uses_per_observation
        )
        valid = valid & jnp.array_equal(
            jnp.sum(state.initiation_starts, axis=1),
            state.execution_starts,
        )
        valid = valid & jnp.all(
            state.initiation_starts <= state.initiation_opportunities
        )
        valid = valid & jnp.array_equal(
            state.completion_moment_counts,
            state.natural_completions,
        )
        valid = valid & jnp.array_equal(
            state.model_error_counts,
            state.natural_completions,
        )
        valid = valid & jnp.array_equal(
            jnp.sum(state.context_signature_counts, axis=1),
            state.natural_completions,
        )
        valid = valid & jnp.all(
            state.natural_completions + state.censor_only_endings
            <= state.execution_starts
        )
        valid = valid & jnp.all(state.goal_terminations <= state.natural_completions)
        valid = valid & jnp.all(state.timeout_terminations <= state.natural_completions)
        valid = valid & jnp.all(
            state.environment_terminations <= state.natural_completions
        )
        valid = valid & jnp.all(
            state.censored_endings
            <= state.natural_completions + state.censor_only_endings
        )
        float_names = (
            "active_external_return",
            "active_pseudo_return",
            "active_baseline_mass",
            "active_discount",
            "active_outcome_delta",
            "active_model_prediction",
            "trial_propensity",
            "trial_return",
            "completion_signature_sums",
            "completion_signature_squared_sums",
            "model_absolute_error_sums",
            "model_squared_error_sums",
            "context_signature_sums",
            "comparison_treatment_return_sums",
            "comparison_treatment_return_squared_sums",
            "comparison_primitive_return_sums",
            "comparison_primitive_return_squared_sums",
            "comparison_treatment_ipw_reward_sums",
            "comparison_treatment_ipw_masses",
            "comparison_primitive_ipw_reward_sums",
            "comparison_primitive_ipw_masses",
            "compute_cost_sums",
            "compute_cost_squared_sums",
        )
        for name in float_names:
            valid = valid & jnp.all(jnp.isfinite(cast(Array, getattr(state, name))))
        nonnegative_float_names = (
            "completion_signature_squared_sums",
            "model_absolute_error_sums",
            "model_squared_error_sums",
            "comparison_treatment_return_squared_sums",
            "comparison_primitive_return_squared_sums",
            "comparison_treatment_ipw_masses",
            "comparison_primitive_ipw_masses",
            "compute_cost_sums",
            "compute_cost_squared_sums",
        )
        for name in nonnegative_float_names:
            valid = valid & jnp.all(cast(Array, getattr(state, name)) >= 0.0)

        active = state.active_option >= 0
        active_index = jnp.clip(state.active_option, 0, cfg.n_options - 1)
        active_valid = (
            (state.active_option < cfg.n_options)
            & (state.active_context >= 0)
            & (state.active_context < cfg.n_contexts)
            & (state.active_steps >= 1)
            & (state.active_steps < _INT32_MAX)
            & (state.active_generation == state.semantic_generations[active_index])
            & (state.active_discount >= 0.0)
            & (state.active_discount <= 1.0)
        )
        inactive_canonical = (
            (state.active_option == -1)
            & (state.active_context == -1)
            & (state.active_generation == -1)
            & (state.active_steps == 0)
            & (state.active_external_return == 0.0)
            & (state.active_pseudo_return == 0.0)
            & (state.active_baseline_mass == 0.0)
            & (state.active_discount == 1.0)
            & jnp.all(state.active_outcome_delta == 0.0)
            & jnp.all(state.active_model_prediction == 0.0)
        )
        valid = valid & jnp.where(active, active_valid, inactive_canonical)

        trial_valid = (
            (state.trial_option >= 0)
            & (state.trial_option < cfg.n_options)
            & (state.trial_context >= 0)
            & (state.trial_context < cfg.n_contexts)
            & (state.trial_propensity >= cfg.propensity_floor)
            & (state.trial_propensity <= cfg.propensity_ceiling)
            & (state.trial_steps >= 1)
            & (state.trial_steps < cfg.fixed_horizon)
        )
        trial_inactive_canonical = (
            (state.trial_option == -1)
            & (state.trial_context == -1)
            & (~state.trial_treatment)
            & (state.trial_propensity == 0.0)
            & (state.trial_steps == 0)
            & (state.trial_return == 0.0)
        )
        valid = valid & jnp.where(state.trial_active, trial_valid, trial_inactive_canonical)
        valid = valid & (state.has_last_transition == (state.observation_count > 0))
        return valid

    @staticmethod
    def _transition_is_new(state: OptionLifecycleAuditState, transition_id: Array) -> Array:
        greater = (transition_id[0] > state.last_transition_id[0]) | (
            (transition_id[0] == state.last_transition_id[0])
            & (transition_id[1] > state.last_transition_id[1])
        )
        return (~state.has_last_transition) | greater

    def arm(
        self,
        state: OptionLifecycleAuditState,
        *,
        transition_id: Array,
        source_digest: Array,
        representation_digest: Array,
        semantic_digests: Array,
        semantic_generations: Array,
        candidate_option: int | Array,
        initiation_context: int | Array,
        initiation_eligible: bool | Array,
        owner_option: int | Array,
        comparator_randomized: bool | Array,
        treatment_propensity: float | Array,
        frozen_model_prediction: Array,
    ) -> OptionLifecycleAuditArm:
        """Freeze one post-selection, pre-outcome lifecycle transaction."""

        self._check_state_contract(state)
        cfg = self._config
        identity = _require_array(
            transition_id,
            name="transition_id",
            shape=(_TRANSITION_WORDS,),
            dtype=jnp.uint32,
        )
        source, representation, semantics, generations = self._check_digest_set(
            source_digest,
            representation_digest,
            semantic_digests,
            semantic_generations,
        )
        candidate = _int32_scalar(candidate_option, name="candidate_option")
        context = _int32_scalar(initiation_context, name="initiation_context")
        eligible = _bool_scalar(initiation_eligible, name="initiation_eligible")
        owner = _int32_scalar(owner_option, name="owner_option")
        randomized = _bool_scalar(comparator_randomized, name="comparator_randomized")
        propensity = _float32_scalar(treatment_propensity, name="treatment_propensity")
        prediction = _require_array(
            frozen_model_prediction,
            name="frozen_model_prediction",
            shape=(cfg.signature_dim,),
            dtype=jnp.float32,
        )

        candidate_in_range = (candidate >= 0) & (candidate < cfg.n_options)
        context_in_range = (context >= 0) & (context < cfg.n_contexts)
        owner_valid = (owner == -1) | (owner == candidate)
        active = state.active_option >= 0
        starts = (~active) & (owner == candidate)
        active_path_valid = (
            (candidate == state.active_option)
            & (owner == state.active_option)
            & (context == state.active_context)
            & (~eligible)
            & (~randomized)
            & (propensity == 0.0)
        )
        trial_without_execution = (
            state.trial_active
            & (~active)
            & (candidate == state.trial_option)
            & (context == state.trial_context)
            & (owner == -1)
            & (~eligible)
            & (~randomized)
            & (propensity == 0.0)
        )
        idle_path_valid = (
            (~state.trial_active)
            & ((owner == -1) | eligible)
            & ((~randomized) | eligible)
            & (
                (~randomized)
                | (
                    (propensity >= cfg.propensity_floor)
                    & (propensity <= cfg.propensity_ceiling)
                )
            )
            & (randomized | (propensity == 0.0))
        )
        path_valid = jnp.where(
            active,
            active_path_valid,
            jnp.where(state.trial_active, trial_without_execution, idle_path_valid),
        )
        prediction_valid = jnp.all(jnp.isfinite(prediction)) & (
            starts | jnp.all(prediction == 0.0)
        )
        binding_valid = (
            jnp.array_equal(source, state.source_digest)
            & jnp.array_equal(representation, state.representation_digest)
            & jnp.array_equal(semantics, state.semantic_digests)
            & jnp.array_equal(generations, state.semantic_generations)
        )
        available = (
            self._state_valid(state)
            & (state.observation_count < cfg.max_observations)
            & (state.revision < _INT32_MAX)
            & self._transition_is_new(state, identity)
            & candidate_in_range
            & context_in_range
            & owner_valid
            & path_valid
            & prediction_valid
            & binding_valid
        )
        arm = OptionLifecycleAuditArm(
            available=available,
            transition_id=identity,
            source_digest=source,
            representation_digest=representation,
            config_digest=self._config_digest,
            semantic_digests=semantics,
            semantic_generations=generations,
            state_revision=state.revision,
            state_checksum=state.state_checksum,
            candidate_option=candidate,
            initiation_context=context,
            initiation_eligible=eligible,
            owner_option=owner,
            starts_execution=starts,
            comparator_randomized=randomized,
            treatment_propensity=propensity,
            frozen_model_prediction=prediction,
            cache_checksum=jnp.zeros((_TRANSITION_WORDS,), dtype=jnp.uint32),
        )
        return dataclasses.replace(
            arm,
            cache_checksum=_checksum_arrays(self._arm_payload_arrays(arm)),
        )

    def observe(
        self,
        state: OptionLifecycleAuditState,
        arm: OptionLifecycleAuditArm,
        *,
        transition_id: Array,
        external_reward: float | Array,
        pseudo_reward: float | Array,
        baseline_mass: float | Array,
        discount: float | Array,
        outcome_delta: Array,
        goal_terminated: bool | Array,
        timeout_terminated: bool | Array,
        environment_terminated: bool | Array,
        censored: bool | Array,
        planning_usage_delta: Array,
        compute_cost: float | Array,
        resident_memory_bytes: int | Array,
    ) -> OptionLifecycleAuditResult:
        """Atomically accept one outcome or return the exact input state."""

        self._check_state_contract(state)
        self._check_arm_contract(arm)
        cfg = self._config
        identity = _require_array(
            transition_id,
            name="transition_id",
            shape=(_TRANSITION_WORDS,),
            dtype=jnp.uint32,
        )
        reward = _float32_scalar(external_reward, name="external_reward")
        pseudo = _float32_scalar(pseudo_reward, name="pseudo_reward")
        baseline = _float32_scalar(baseline_mass, name="baseline_mass")
        gamma = _float32_scalar(discount, name="discount")
        outcome = _require_array(
            outcome_delta,
            name="outcome_delta",
            shape=(cfg.outcome_dim,),
            dtype=jnp.float32,
        )
        goal = _bool_scalar(goal_terminated, name="goal_terminated")
        timeout = _bool_scalar(timeout_terminated, name="timeout_terminated")
        environment = _bool_scalar(environment_terminated, name="environment_terminated")
        is_censored = _bool_scalar(censored, name="censored")
        planning = _require_array(
            planning_usage_delta,
            name="planning_usage_delta",
            shape=(cfg.n_options,),
            dtype=jnp.int32,
        )
        cost = _float32_scalar(compute_cost, name="compute_cost")
        memory = _int32_scalar(resident_memory_bytes, name="resident_memory_bytes")

        candidate = jnp.clip(arm.candidate_option, 0, cfg.n_options - 1)
        context = jnp.clip(arm.initiation_context, 0, cfg.n_contexts - 1)
        execution_existed = state.active_option >= 0
        has_execution = execution_existed | arm.starts_execution
        execution_option = jnp.where(
            execution_existed,
            state.active_option,
            arm.candidate_option,
        )
        execution_index = jnp.clip(execution_option, 0, cfg.n_options - 1)
        execution_context = jnp.where(
            execution_existed,
            state.active_context,
            arm.initiation_context,
        )
        execution_generation = jnp.where(
            execution_existed,
            state.active_generation,
            arm.semantic_generations[candidate],
        )
        prior_steps = jnp.where(execution_existed, state.active_steps, 0)
        prior_external = jnp.where(
            execution_existed,
            state.active_external_return,
            jnp.float32(0.0),
        )
        prior_pseudo = jnp.where(
            execution_existed,
            state.active_pseudo_return,
            jnp.float32(0.0),
        )
        prior_baseline = jnp.where(
            execution_existed,
            state.active_baseline_mass,
            jnp.float32(0.0),
        )
        prior_discount = jnp.where(
            execution_existed,
            state.active_discount,
            jnp.float32(1.0),
        )
        prior_outcome = jnp.where(
            execution_existed,
            state.active_outcome_delta,
            jnp.zeros((cfg.outcome_dim,), dtype=jnp.float32),
        )
        frozen_prediction = jnp.where(
            execution_existed,
            state.active_model_prediction,
            arm.frozen_model_prediction,
        )
        new_steps = prior_steps + jnp.int32(1)
        # STOMP's option environment-return model uses the discounted return
        # accumulated under the option's pre-step discount mass.
        new_external = prior_external + prior_discount * reward
        new_pseudo = prior_pseudo + pseudo
        new_baseline = prior_baseline + baseline
        new_discount = prior_discount * gamma
        new_outcome = prior_outcome + outcome
        natural = goal | timeout | environment
        execution_ends = has_execution & (natural | is_censored)
        natural_completion = has_execution & natural
        censor_only = has_execution & is_censored & (~natural)
        completion_signature = jnp.concatenate(
            (
                jnp.stack(
                    (
                        new_external,
                        new_pseudo,
                        new_steps.astype(jnp.float32),
                        new_baseline,
                        new_discount,
                    )
                ),
                new_outcome,
            )
        )
        model_error = completion_signature - frozen_prediction

        one_option = jax.nn.one_hot(candidate, cfg.n_options, dtype=jnp.int32)
        one_execution = jax.nn.one_hot(execution_index, cfg.n_options, dtype=jnp.int32)
        opportunity_cell = jax.nn.one_hot(
            candidate * cfg.n_contexts + context,
            cfg.n_options * cfg.n_contexts,
            dtype=jnp.int32,
        ).reshape((cfg.n_options, cfg.n_contexts))
        completion_context_cell = jax.nn.one_hot(
            execution_index * cfg.n_contexts + execution_context,
            cfg.n_options * cfg.n_contexts,
            dtype=jnp.int32,
        ).reshape((cfg.n_options, cfg.n_contexts))

        trial_existed = state.trial_active
        trial_exists = trial_existed | arm.comparator_randomized
        trial_option = jnp.where(trial_existed, state.trial_option, arm.candidate_option)
        trial_index = jnp.clip(trial_option, 0, cfg.n_options - 1)
        trial_context = jnp.where(
            trial_existed,
            state.trial_context,
            arm.initiation_context,
        )
        trial_context_index = jnp.clip(trial_context, 0, cfg.n_contexts - 1)
        trial_treatment = jnp.where(
            trial_existed,
            state.trial_treatment,
            arm.owner_option == arm.candidate_option,
        )
        trial_propensity = jnp.where(
            trial_existed,
            state.trial_propensity,
            arm.treatment_propensity,
        )
        trial_steps = jnp.where(trial_existed, state.trial_steps, 0) + jnp.int32(1)
        trial_return = jnp.where(
            trial_existed,
            state.trial_return,
            jnp.float32(0.0),
        ) + reward
        trial_completed = trial_exists & (trial_steps == cfg.fixed_horizon)
        one_trial = jax.nn.one_hot(
            trial_index * cfg.n_contexts + trial_context_index,
            cfg.n_options * cfg.n_contexts,
            dtype=jnp.int32,
        ).reshape((cfg.n_options, cfg.n_contexts))

        inputs_finite = (
            jnp.isfinite(reward)
            & jnp.isfinite(pseudo)
            & jnp.isfinite(baseline)
            & jnp.isfinite(gamma)
            & jnp.all(jnp.isfinite(outcome))
            & jnp.isfinite(cost)
        )
        planning_capacity = jnp.all(
            state.planning_use_counts <= (_INT32_MAX - planning)
        )
        runtime_valid = (
            inputs_finite
            & (baseline >= 0.0)
            & (gamma >= 0.0)
            & (gamma <= 1.0)
            & jnp.all(planning >= 0)
            & jnp.all(planning <= cfg.max_planning_uses_per_observation)
            & planning_capacity
            & (cost >= 0.0)
            & (cost <= cfg.max_compute_cost_per_observation)
            & (memory >= 0)
            & (has_execution | ((cost == 0.0) & (memory == 0)))
            & (has_execution | (~is_censored))
            & ((~has_execution) | (new_steps < _INT32_MAX))
        )
        binding_valid = (
            arm.available
            & jnp.array_equal(identity, arm.transition_id)
            & jnp.array_equal(arm.config_digest, self._config_digest)
            & jnp.array_equal(arm.source_digest, state.source_digest)
            & jnp.array_equal(arm.representation_digest, state.representation_digest)
            & jnp.array_equal(arm.semantic_digests, state.semantic_digests)
            & jnp.array_equal(arm.semantic_generations, state.semantic_generations)
            & (arm.state_revision == state.revision)
            & jnp.array_equal(arm.state_checksum, state.state_checksum)
            & jnp.array_equal(
                arm.cache_checksum,
                _checksum_arrays(self._arm_payload_arrays(arm)),
            )
        )
        base_valid = (
            self._state_valid(state)
            & binding_valid
            & runtime_valid
            & (state.observation_count < cfg.max_observations)
            & (state.revision < _INT32_MAX)
        )

        opportunity_increment = opportunity_cell * (
            arm.initiation_eligible.astype(jnp.int32)
        )
        start_increment = opportunity_cell * arm.starts_execution.astype(jnp.int32)
        execution_start_increment = one_option * arm.starts_execution.astype(jnp.int32)
        natural_increment = one_execution * natural_completion.astype(jnp.int32)
        goal_increment = one_execution * (has_execution & goal).astype(jnp.int32)
        timeout_increment = one_execution * (has_execution & timeout).astype(jnp.int32)
        environment_increment = one_execution * (
            has_execution & environment
        ).astype(jnp.int32)
        censored_increment = one_execution * (has_execution & is_censored).astype(jnp.int32)
        censor_only_increment = one_execution * censor_only.astype(jnp.int32)
        signature_increment = one_execution.astype(jnp.float32)[:, None] * (
            natural_completion.astype(jnp.float32) * completion_signature[None, :]
        )
        signature_squared_increment = one_execution.astype(jnp.float32)[:, None] * (
            natural_completion.astype(jnp.float32) * (completion_signature**2)[None, :]
        )
        error_increment = one_execution.astype(jnp.float32)[:, None] * (
            natural_completion.astype(jnp.float32) * model_error[None, :]
        )
        context_signature_increment = (
            completion_context_cell.astype(jnp.float32)[:, :, None]
            * natural_completion.astype(jnp.float32)
            * completion_signature[None, None, :]
        )
        planning_decisions = (planning > 0).astype(jnp.int32)
        compute_increment = one_execution.astype(jnp.float32) * (
            has_execution.astype(jnp.float32) * cost
        )
        memory_candidate = one_execution * (
            has_execution.astype(jnp.int32) * memory
        )

        treatment_finished = trial_completed & trial_treatment
        primitive_finished = trial_completed & (~trial_treatment)
        treatment_count_increment = one_trial * treatment_finished.astype(jnp.int32)
        primitive_count_increment = one_trial * primitive_finished.astype(jnp.int32)
        treatment_return_increment = one_trial.astype(jnp.float32) * (
            treatment_finished.astype(jnp.float32) * trial_return
        )
        primitive_return_increment = one_trial.astype(jnp.float32) * (
            primitive_finished.astype(jnp.float32) * trial_return
        )
        safe_propensity = jnp.clip(
            trial_propensity,
            cfg.propensity_floor,
            cfg.propensity_ceiling,
        )
        treatment_ipw = treatment_return_increment / safe_propensity
        treatment_mass = one_trial.astype(jnp.float32) * (
            treatment_finished.astype(jnp.float32) / safe_propensity
        )
        primitive_ipw = primitive_return_increment / (1.0 - safe_propensity)
        primitive_mass = one_trial.astype(jnp.float32) * (
            primitive_finished.astype(jnp.float32) / (1.0 - safe_propensity)
        )

        next_active = has_execution & (~execution_ends)
        next_trial_active = trial_exists & (~trial_completed)
        proposed = dataclasses.replace(
            state,
            revision=state.revision + jnp.int32(1),
            observation_count=state.observation_count + jnp.int32(1),
            has_last_transition=jnp.asarray(True, dtype=jnp.bool_),
            last_transition_id=identity,
            active_option=jnp.where(next_active, execution_option, -1).astype(jnp.int32),
            active_context=jnp.where(next_active, execution_context, -1).astype(jnp.int32),
            active_generation=jnp.where(next_active, execution_generation, -1).astype(jnp.int32),
            active_steps=jnp.where(next_active, new_steps, 0).astype(jnp.int32),
            active_external_return=jnp.where(next_active, new_external, 0.0).astype(jnp.float32),
            active_pseudo_return=jnp.where(next_active, new_pseudo, 0.0).astype(jnp.float32),
            active_baseline_mass=jnp.where(next_active, new_baseline, 0.0).astype(jnp.float32),
            active_discount=jnp.where(next_active, new_discount, 1.0).astype(jnp.float32),
            active_outcome_delta=jnp.where(next_active, new_outcome, jnp.zeros_like(new_outcome)),
            active_model_prediction=jnp.where(
                next_active,
                frozen_prediction,
                jnp.zeros_like(frozen_prediction),
            ),
            trial_active=next_trial_active,
            trial_option=jnp.where(next_trial_active, trial_option, -1).astype(jnp.int32),
            trial_context=jnp.where(
                next_trial_active,
                trial_context,
                -1,
            ).astype(jnp.int32),
            trial_treatment=jnp.where(next_trial_active, trial_treatment, False),
            trial_propensity=jnp.where(
                next_trial_active,
                trial_propensity,
                0.0,
            ).astype(jnp.float32),
            trial_steps=jnp.where(next_trial_active, trial_steps, 0).astype(jnp.int32),
            trial_return=jnp.where(next_trial_active, trial_return, 0.0).astype(jnp.float32),
            initiation_opportunities=state.initiation_opportunities + opportunity_increment,
            initiation_starts=state.initiation_starts + start_increment,
            execution_starts=state.execution_starts + execution_start_increment,
            natural_completions=state.natural_completions + natural_increment,
            goal_terminations=state.goal_terminations + goal_increment,
            timeout_terminations=state.timeout_terminations + timeout_increment,
            environment_terminations=(
                state.environment_terminations + environment_increment
            ),
            censored_endings=state.censored_endings + censored_increment,
            censor_only_endings=state.censor_only_endings + censor_only_increment,
            completion_moment_counts=(
                state.completion_moment_counts + natural_increment
            ),
            completion_signature_sums=(
                state.completion_signature_sums + signature_increment
            ),
            completion_signature_squared_sums=(
                state.completion_signature_squared_sums + signature_squared_increment
            ),
            model_error_counts=state.model_error_counts + natural_increment,
            model_absolute_error_sums=(
                state.model_absolute_error_sums + jnp.abs(error_increment)
            ),
            model_squared_error_sums=(
                state.model_squared_error_sums + error_increment**2
            ),
            context_signature_counts=(
                state.context_signature_counts
                + completion_context_cell * natural_completion.astype(jnp.int32)
            ),
            context_signature_sums=(
                state.context_signature_sums + context_signature_increment
            ),
            comparison_treatment_counts=(
                state.comparison_treatment_counts + treatment_count_increment
            ),
            comparison_primitive_counts=(
                state.comparison_primitive_counts + primitive_count_increment
            ),
            comparison_treatment_return_sums=(
                state.comparison_treatment_return_sums + treatment_return_increment
            ),
            comparison_treatment_return_squared_sums=(
                state.comparison_treatment_return_squared_sums
                + treatment_return_increment**2
            ),
            comparison_primitive_return_sums=(
                state.comparison_primitive_return_sums + primitive_return_increment
            ),
            comparison_primitive_return_squared_sums=(
                state.comparison_primitive_return_squared_sums
                + primitive_return_increment**2
            ),
            comparison_treatment_ipw_reward_sums=(
                state.comparison_treatment_ipw_reward_sums + treatment_ipw
            ),
            comparison_treatment_ipw_masses=(
                state.comparison_treatment_ipw_masses + treatment_mass
            ),
            comparison_primitive_ipw_reward_sums=(
                state.comparison_primitive_ipw_reward_sums + primitive_ipw
            ),
            comparison_primitive_ipw_masses=(
                state.comparison_primitive_ipw_masses + primitive_mass
            ),
            planning_use_counts=state.planning_use_counts + planning,
            planning_decision_counts=(
                state.planning_decision_counts + planning_decisions
            ),
            compute_observation_counts=(
                state.compute_observation_counts
                + one_execution * has_execution.astype(jnp.int32)
            ),
            compute_cost_sums=state.compute_cost_sums + compute_increment,
            compute_cost_squared_sums=(
                state.compute_cost_squared_sums + compute_increment**2
            ),
            resident_memory_max_bytes=jnp.maximum(
                state.resident_memory_max_bytes,
                memory_candidate,
            ),
            state_checksum=jnp.zeros((_TRANSITION_WORDS,), dtype=jnp.uint32),
        )
        proposed = self._with_checksum(proposed)
        transaction_valid = base_valid & self._state_valid(proposed)
        next_state = jax.lax.cond(
            transaction_valid,
            lambda _: proposed,
            lambda _: state,
            operand=None,
        )
        return OptionLifecycleAuditResult(
            state=next_state,
            transaction_valid=transaction_valid,
            applied=transaction_valid,
            execution_started=transaction_valid & arm.starts_execution,
            execution_completed=transaction_valid & natural_completion,
            censor_only_ending=transaction_valid & censor_only,
            comparator_trial_completed=transaction_valid & trial_completed,
            model_error_scored=transaction_valid & natural_completion,
        )

    def maintenance_report(
        self,
        state: OptionLifecycleAuditState,
    ) -> OptionLifecycleMaintenanceReport:
        """Return a bounded scorecard and replacement proposal without mutation."""

        self._check_state_contract(state)
        cfg = self._config
        n, c, s = cfg.n_options, cfg.n_contexts, cfg.signature_dim
        state_valid = self._state_valid(state)
        opportunity_totals = jnp.sum(state.initiation_opportunities, axis=1)
        start_totals = jnp.sum(state.initiation_starts, axis=1)
        initiation_ready = opportunity_totals >= cfg.initiation_opportunity_floor
        initiation_coverage = start_totals.astype(jnp.float32) / jnp.maximum(
            opportunity_totals,
            1,
        ).astype(jnp.float32)

        ending_totals = state.natural_completions + state.censor_only_endings
        completion_ready = ending_totals >= cfg.completion_evidence_floor
        completion_reliability = state.natural_completions.astype(jnp.float32) / jnp.maximum(
            ending_totals,
            1,
        ).astype(jnp.float32)
        completion_denominator = jnp.maximum(
            state.completion_moment_counts,
            1,
        ).astype(jnp.float32)
        completion_means = state.completion_signature_sums / completion_denominator[:, None]

        treatment_denominator = jnp.maximum(
            state.comparison_treatment_counts,
            1,
        ).astype(jnp.float32)
        primitive_denominator = jnp.maximum(
            state.comparison_primitive_counts,
            1,
        ).astype(jnp.float32)
        treatment_means_by_context = (
            state.comparison_treatment_return_sums / treatment_denominator
        )
        primitive_means_by_context = (
            state.comparison_primitive_return_sums / primitive_denominator
        )
        marginal_by_context = treatment_means_by_context - primitive_means_by_context
        # Every configured context has fixed, equal mass.  Missing contexts
        # remain zero cells and can never be renormalized away.
        treatment_means = jnp.mean(treatment_means_by_context, axis=1)
        primitive_means = jnp.mean(primitive_means_by_context, axis=1)
        marginal = jnp.mean(marginal_by_context, axis=1)
        comparison_ready_by_context = (
            (state.comparison_treatment_counts >= cfg.comparison_treatment_evidence_floor)
            & (
                state.comparison_primitive_counts
                >= cfg.comparison_primitive_evidence_floor
            )
        )
        comparison_ready = jnp.all(comparison_ready_by_context, axis=1)
        treatment_ipw_mean_by_context = (
            state.comparison_treatment_ipw_reward_sums
            / jnp.maximum(
            state.comparison_treatment_ipw_masses,
            jnp.asarray(np.finfo(np.float32).tiny, dtype=jnp.float32),
        )
        )
        primitive_ipw_mean_by_context = (
            state.comparison_primitive_ipw_reward_sums
            / jnp.maximum(
            state.comparison_primitive_ipw_masses,
            jnp.asarray(np.finfo(np.float32).tiny, dtype=jnp.float32),
        )
        )
        ipw_marginal_by_context = (
            treatment_ipw_mean_by_context - primitive_ipw_mean_by_context
        )
        ipw_marginal = jnp.mean(ipw_marginal_by_context, axis=1)

        model_ready = state.model_error_counts >= cfg.model_error_evidence_floor
        model_mse = state.model_squared_error_sums / jnp.maximum(
            state.model_error_counts,
            1,
        ).astype(jnp.float32)[:, None]
        normalized_model_rmse = jnp.sqrt(jnp.maximum(model_mse, 0.0)) / (
            self._signature_scales[None, :]
        )
        compute_means = state.compute_cost_sums / jnp.maximum(
            state.compute_observation_counts,
            1,
        ).astype(jnp.float32)

        context_supported = (
            state.context_signature_counts >= cfg.signature_evidence_floor_per_context
        )
        context_denominator = jnp.maximum(
            state.context_signature_counts,
            1,
        ).astype(jnp.float32)
        normalized_context_means = (
            state.context_signature_sums
            / context_denominator[:, :, None]
            / self._signature_scales[None, None, :]
        )

        def accumulate_context(
            context_index: int,
            carry: tuple[Array, Array],
        ) -> tuple[Array, Array]:
            distance_sum, shared_count = carry
            values = normalized_context_means[:, context_index, :]
            squared_norms = jnp.sum(values**2, axis=1)
            squared_distances = jnp.maximum(
                squared_norms[:, None]
                + squared_norms[None, :]
                - 2.0 * (values @ values.T),
                0.0,
            )
            distances = jnp.sqrt(squared_distances / jnp.float32(s))
            supported = context_supported[:, context_index]
            shared = supported[:, None] & supported[None, :]
            return (
                distance_sum + jnp.where(shared, distances, 0.0),
                shared_count + shared.astype(jnp.int32),
            )

        redundancy_distance_sums, shared_context_counts = jax.lax.fori_loop(
            0,
            c,
            accumulate_context,
            (
                jnp.zeros((n, n), dtype=jnp.float32),
                jnp.zeros((n, n), dtype=jnp.int32),
            ),
        )
        redundancy_distances = redundancy_distance_sums / jnp.maximum(
            shared_context_counts,
            1,
        ).astype(jnp.float32)
        redundancy_ready = shared_context_counts >= cfg.redundancy_shared_context_floor
        not_diagonal = ~jnp.eye(n, dtype=jnp.bool_)
        redundant_pairs = (
            redundancy_ready
            & not_diagonal
            & (redundancy_distances <= cfg.redundancy_distance_threshold)
        )
        # The later slot is the deterministic loser for an otherwise symmetric
        # redundant pair.  The report does not mutate either slot.
        redundancy_loser = jnp.any(jnp.triu(redundant_pairs, k=1), axis=0)

        all_ready = initiation_ready & completion_ready & comparison_ready & model_ready
        low_coverage = initiation_ready & (
            initiation_coverage < cfg.min_initiation_coverage
        )
        low_reliability = completion_ready & (
            completion_reliability < cfg.min_completion_reliability
        )
        low_improvement = comparison_ready & (
            marginal < cfg.min_marginal_improvement
        )
        high_model_error = model_ready & (
            jnp.sqrt(jnp.mean(normalized_model_rmse**2, axis=1))
            > cfg.max_normalized_model_rmse
        )
        low_planning_usage = all_ready & (
            state.planning_use_counts < cfg.min_planning_uses
        )
        high_compute_cost = (state.compute_observation_counts > 0) & (
            compute_means > cfg.max_mean_compute_cost
        )
        high_memory_cost = (
            state.resident_memory_max_bytes > cfg.max_resident_memory_bytes
        )
        concern_counts = (
            low_coverage.astype(jnp.int32)
            + low_reliability.astype(jnp.int32)
            + low_improvement.astype(jnp.int32)
            + high_model_error.astype(jnp.int32)
            + low_planning_usage.astype(jnp.int32)
            + high_compute_cost.astype(jnp.int32)
            + high_memory_cost.astype(jnp.int32)
            + redundancy_loser.astype(jnp.int32)
        )
        replacement_eligible = all_ready & (concern_counts > 0)
        ranking_score = jnp.where(replacement_eligible, 100 + concern_counts, 0)
        ranked = jnp.argsort(-ranking_score, stable=True)[: cfg.maintenance_budget]
        proposal_mask = replacement_eligible[ranked]
        proposed_slots = jnp.where(proposal_mask, ranked, -1).astype(jnp.int32)

        def gate_float(value: Array) -> Array:
            return jnp.where(state_valid, value, jnp.zeros_like(value))

        def gate_int(value: Array) -> Array:
            return jnp.where(state_valid, value, jnp.zeros_like(value))

        def gate_bool(value: Array) -> Array:
            return state_valid & value

        return OptionLifecycleMaintenanceReport(
            state_valid=state_valid,
            state_revision=jnp.where(state_valid, state.revision, -1).astype(jnp.int32),
            initiation_coverage=gate_float(initiation_coverage),
            initiation_evidence_ready=gate_bool(initiation_ready),
            completion_reliability=gate_float(completion_reliability),
            completion_evidence_ready=gate_bool(completion_ready),
            external_return_means=gate_float(completion_means[:, 0]),
            pseudo_return_means=gate_float(completion_means[:, 1]),
            treatment_return_means=gate_float(treatment_means),
            primitive_return_means=gate_float(primitive_means),
            marginal_improvement=gate_float(marginal),
            inverse_propensity_marginal_improvement=gate_float(ipw_marginal),
            treatment_return_means_by_context=gate_float(
                treatment_means_by_context
            ),
            primitive_return_means_by_context=gate_float(
                primitive_means_by_context
            ),
            marginal_improvement_by_context=gate_float(marginal_by_context),
            inverse_propensity_marginal_improvement_by_context=gate_float(
                ipw_marginal_by_context
            ),
            comparison_ready=gate_bool(comparison_ready),
            normalized_model_rmse=gate_float(normalized_model_rmse),
            model_evidence_ready=gate_bool(model_ready),
            planning_use_counts=gate_int(state.planning_use_counts),
            mean_compute_cost=gate_float(compute_means),
            resident_memory_max_bytes=gate_int(state.resident_memory_max_bytes),
            shared_context_counts=gate_int(shared_context_counts),
            redundancy_distances=gate_float(redundancy_distances),
            redundancy_ready=gate_bool(redundancy_ready),
            redundant_pairs=gate_bool(redundant_pairs),
            redundancy_loser=gate_bool(redundancy_loser),
            all_required_evidence_ready=gate_bool(all_ready),
            concern_counts=gate_int(concern_counts),
            replacement_eligible=gate_bool(replacement_eligible),
            proposed_replacement_slots=jnp.where(
                state_valid,
                proposed_slots,
                -jnp.ones_like(proposed_slots),
            ),
            proposed_replacement_mask=gate_bool(proposal_mask),
            curation_authority=jnp.asarray(False, dtype=jnp.bool_),
            promotion_authority=jnp.asarray(False, dtype=jnp.bool_),
            go_no_go_authority=jnp.asarray(False, dtype=jnp.bool_),
        )

    def rebind(
        self,
        state: OptionLifecycleAuditState,
        *,
        source_digest: Array,
        representation_digest: Array,
        semantic_digests: Array,
    ) -> OptionLifecycleRebindResult:
        """Preserve only compatible slots; reset changed semantics atomically."""

        self._check_state_contract(state)
        cfg = self._config
        source, representation, semantics, _ = self._check_digest_set(
            source_digest,
            representation_digest,
            semantic_digests,
            state.semantic_generations,
        )
        source_same = jnp.array_equal(source, state.source_digest)
        representation_same = jnp.array_equal(
            representation,
            state.representation_digest,
        )
        semantic_same = jnp.all(semantics == state.semantic_digests, axis=1)
        preserved = source_same & representation_same & semantic_same
        reset = ~preserved
        changed = jnp.any(reset)
        in_flight = (state.active_option >= 0) | state.trial_active
        semantic_nonzero = jnp.all(jnp.any(semantics != 0, axis=1))
        equality = jnp.all(semantics[:, None, :] == semantics[None, :, :], axis=2)
        unique = jnp.all(equality == jnp.eye(cfg.n_options, dtype=jnp.bool_))
        generations_available = jnp.all(
            (~reset) | (state.semantic_generations < _INT32_MAX)
        )
        binding_valid = (
            jnp.any(source != 0)
            & jnp.any(representation != 0)
            & semantic_nonzero
            & unique
            & generations_available
        )
        transaction_valid = (
            self._state_valid(state)
            & binding_valid
            & (state.revision < _INT32_MAX)
        )
        deferred = transaction_valid & changed & in_flight
        applied = transaction_valid & changed & (~in_flight)
        reset_i = reset.astype(jnp.int32)
        keep_i = (~reset).astype(jnp.int32)
        keep_f = keep_i.astype(jnp.float32)

        def keep_vector(value: Array) -> Array:
            return value * keep_i

        def keep_float_vector(value: Array) -> Array:
            return value * keep_f

        def keep_matrix(value: Array) -> Array:
            return value * keep_i[:, None]

        def keep_float_matrix(value: Array) -> Array:
            return value * keep_f[:, None]

        def keep_float_tensor(value: Array) -> Array:
            return value * keep_f[:, None, None]

        proposed = dataclasses.replace(
            state,
            source_digest=source,
            representation_digest=representation,
            semantic_digests=semantics,
            semantic_generations=state.semantic_generations + reset_i,
            revision=state.revision + jnp.int32(1),
            initiation_opportunities=keep_matrix(state.initiation_opportunities),
            initiation_starts=keep_matrix(state.initiation_starts),
            execution_starts=keep_vector(state.execution_starts),
            natural_completions=keep_vector(state.natural_completions),
            goal_terminations=keep_vector(state.goal_terminations),
            timeout_terminations=keep_vector(state.timeout_terminations),
            environment_terminations=keep_vector(state.environment_terminations),
            censored_endings=keep_vector(state.censored_endings),
            censor_only_endings=keep_vector(state.censor_only_endings),
            completion_moment_counts=keep_vector(state.completion_moment_counts),
            completion_signature_sums=keep_float_matrix(
                state.completion_signature_sums
            ),
            completion_signature_squared_sums=keep_float_matrix(
                state.completion_signature_squared_sums
            ),
            model_error_counts=keep_vector(state.model_error_counts),
            model_absolute_error_sums=keep_float_matrix(
                state.model_absolute_error_sums
            ),
            model_squared_error_sums=keep_float_matrix(
                state.model_squared_error_sums
            ),
            context_signature_counts=keep_matrix(state.context_signature_counts),
            context_signature_sums=keep_float_tensor(state.context_signature_sums),
            comparison_treatment_counts=keep_matrix(
                state.comparison_treatment_counts
            ),
            comparison_primitive_counts=keep_matrix(
                state.comparison_primitive_counts
            ),
            comparison_treatment_return_sums=keep_float_matrix(
                state.comparison_treatment_return_sums
            ),
            comparison_treatment_return_squared_sums=keep_float_matrix(
                state.comparison_treatment_return_squared_sums
            ),
            comparison_primitive_return_sums=keep_float_matrix(
                state.comparison_primitive_return_sums
            ),
            comparison_primitive_return_squared_sums=keep_float_matrix(
                state.comparison_primitive_return_squared_sums
            ),
            comparison_treatment_ipw_reward_sums=keep_float_matrix(
                state.comparison_treatment_ipw_reward_sums
            ),
            comparison_treatment_ipw_masses=keep_float_matrix(
                state.comparison_treatment_ipw_masses
            ),
            comparison_primitive_ipw_reward_sums=keep_float_matrix(
                state.comparison_primitive_ipw_reward_sums
            ),
            comparison_primitive_ipw_masses=keep_float_matrix(
                state.comparison_primitive_ipw_masses
            ),
            planning_use_counts=keep_vector(state.planning_use_counts),
            planning_decision_counts=keep_vector(state.planning_decision_counts),
            compute_observation_counts=keep_vector(state.compute_observation_counts),
            compute_cost_sums=keep_float_vector(state.compute_cost_sums),
            compute_cost_squared_sums=keep_float_vector(
                state.compute_cost_squared_sums
            ),
            resident_memory_max_bytes=keep_vector(
                state.resident_memory_max_bytes
            ),
            state_checksum=jnp.zeros((_TRANSITION_WORDS,), dtype=jnp.uint32),
        )
        proposed = self._with_checksum(proposed)
        applied = applied & self._state_valid(proposed)
        next_state = jax.lax.cond(applied, lambda _: proposed, lambda _: state, operand=None)
        return OptionLifecycleRebindResult(
            state=next_state,
            transaction_valid=transaction_valid,
            applied=applied,
            deferred=deferred,
            preserved_slots=transaction_valid & preserved,
            reset_slots=applied & reset,
        )

    @staticmethod
    def _encode_checkpoint_array(value: Array) -> dict[str, object]:
        host = np.asarray(jax.device_get(value))
        return {
            "dtype": host.dtype.str,
            "shape": list(host.shape),
            "bytes_hex": host.tobytes(order="C").hex(),
        }

    def checkpoint_payload(self, state: OptionLifecycleAuditState) -> dict[str, object]:
        """Serialize one valid state into an exact, bit-preserving v1 payload."""

        self._check_state_contract(state)
        if not bool(jax.device_get(self._state_valid(state))):
            raise ValueError("cannot checkpoint an invalid option lifecycle state")
        encoded_state = {
            field.name: self._encode_checkpoint_array(cast(Array, getattr(state, field.name)))
            for field in dataclasses.fields(OptionLifecycleAuditState)
        }
        core: dict[str, object] = {
            "schema_version": OPTION_LIFECYCLE_AUDIT_CHECKPOINT_SCHEMA,
            "state_type": "OptionLifecycleAuditState",
            "config": self.to_config(),
            "config_digest_hex": np.asarray(
                jax.device_get(self._config_digest),
                dtype=np.uint32,
            ).tobytes(order="C").hex(),
            "state": encoded_state,
        }
        canonical = json.dumps(core, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return {
            **core,
            "payload_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        }

    def _decode_checkpoint_state(self, value: object) -> OptionLifecycleAuditState:
        if type(value) is not dict:
            raise ValueError("checkpoint state must be an exact dict")
        raw = cast(dict[object, object], value)
        specs = self._state_specifications()
        if set(raw) != set(specs):
            raise ValueError("checkpoint state fields differ from schema v1")
        decoded: dict[str, Array] = {}
        for name, (expected_shape, expected_jax_dtype) in specs.items():
            record = raw[name]
            if type(record) is not dict:
                raise ValueError(f"checkpoint state.{name} must be an exact dict")
            encoded = cast(dict[object, object], record)
            if set(encoded) != {"dtype", "shape", "bytes_hex"}:
                raise ValueError(f"checkpoint state.{name} encoding keys differ")
            expected_dtype = np.dtype(expected_jax_dtype)
            if encoded["dtype"] != expected_dtype.str:
                raise ValueError(f"checkpoint state.{name} dtype differs")
            shape_value = encoded["shape"]
            if type(shape_value) is not list or any(
                type(cell) is not int for cell in cast(list[object], shape_value)
            ):
                raise ValueError(f"checkpoint state.{name} shape encoding differs")
            if tuple(cast(list[int], shape_value)) != expected_shape:
                raise ValueError(f"checkpoint state.{name} shape differs")
            bytes_hex = encoded["bytes_hex"]
            if type(bytes_hex) is not str:
                raise ValueError(f"checkpoint state.{name} bytes_hex must be a string")
            expected_nbytes = int(np.prod(expected_shape, dtype=np.int64)) * expected_dtype.itemsize
            if expected_shape == ():
                expected_nbytes = expected_dtype.itemsize
            if len(bytes_hex) != expected_nbytes * 2:
                raise ValueError(f"checkpoint state.{name} byte length differs")
            try:
                payload_bytes = bytes.fromhex(bytes_hex)
            except ValueError as error:
                raise ValueError(f"checkpoint state.{name} bytes_hex is malformed") from error
            host = np.frombuffer(payload_bytes, dtype=expected_dtype).copy().reshape(expected_shape)
            decoded[name] = jnp.asarray(host)
        state = OptionLifecycleAuditState(**cast(Any, decoded))
        self._check_state_contract(state)
        return state

    def restore_checkpoint(
        self,
        payload: object,
        *,
        expected_source_digest: Array,
        expected_representation_digest: Array,
        expected_semantic_digests: Array,
    ) -> OptionLifecycleAuditState:
        """Restore only exact v1 bytes under the exact live semantic binding."""

        if type(payload) is not dict:
            raise ValueError("option lifecycle checkpoint must be an exact dict")
        raw = cast(dict[object, object], payload)
        expected_keys = {
            "schema_version",
            "state_type",
            "config",
            "config_digest_hex",
            "state",
            "payload_sha256",
        }
        if set(raw) != expected_keys:
            raise ValueError("option lifecycle checkpoint keys differ from schema v1")
        if raw["schema_version"] != OPTION_LIFECYCLE_AUDIT_CHECKPOINT_SCHEMA:
            raise ValueError("option lifecycle checkpoint schema_version differs")
        if raw["state_type"] != "OptionLifecycleAuditState":
            raise ValueError("option lifecycle checkpoint state_type differs")
        restored_config = OptionLifecycleAuditConfig.from_config(
            cast(Mapping[str, object], raw["config"])
        )
        if restored_config != self._config:
            raise ValueError("option lifecycle checkpoint config differs")
        expected_config_digest_hex = np.asarray(
            jax.device_get(self._config_digest),
            dtype=np.uint32,
        ).tobytes(order="C").hex()
        if raw["config_digest_hex"] != expected_config_digest_hex:
            raise ValueError("option lifecycle checkpoint config digest differs")
        persisted_sha = raw["payload_sha256"]
        if type(persisted_sha) is not str or len(persisted_sha) != 64:
            raise ValueError("option lifecycle checkpoint payload_sha256 differs")
        core = {key: raw[key] for key in expected_keys if key != "payload_sha256"}
        canonical = json.dumps(core, sort_keys=True, separators=(",", ":"), allow_nan=False)
        actual_sha = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        if persisted_sha != actual_sha:
            raise ValueError("option lifecycle checkpoint payload digest differs")

        source = _require_array(
            expected_source_digest,
            name="expected_source_digest",
            shape=(_DIGEST_WORDS,),
            dtype=jnp.uint32,
        )
        representation = _require_array(
            expected_representation_digest,
            name="expected_representation_digest",
            shape=(_DIGEST_WORDS,),
            dtype=jnp.uint32,
        )
        semantics = _require_array(
            expected_semantic_digests,
            name="expected_semantic_digests",
            shape=(self._config.n_options, _DIGEST_WORDS),
            dtype=jnp.uint32,
        )
        state = self._decode_checkpoint_state(raw["state"])
        binding_valid = (
            jnp.array_equal(state.source_digest, source)
            & jnp.array_equal(state.representation_digest, representation)
            & jnp.array_equal(state.semantic_digests, semantics)
            & self._state_valid(state)
        )
        if not bool(jax.device_get(binding_valid)):
            raise ValueError("option lifecycle checkpoint is invalid, stale, or rebound")
        return state


# Long-form aliases keep the type family mechanically discoverable while the
# shorter report/result names remain readable at call sites.
OptionLifecycleAuditMaintenanceReport = OptionLifecycleMaintenanceReport
OptionLifecycleAuditRebindResult = OptionLifecycleRebindResult


__all__ = [
    "OPTION_LIFECYCLE_AUDIT_CHECKPOINT_SCHEMA",
    "OPTION_LIFECYCLE_AUDIT_CONFIG_SCHEMA",
    "OPTION_LIFECYCLE_AUDIT_CURATION_AUTHORITY",
    "OPTION_LIFECYCLE_AUDIT_GO_NO_GO_AUTHORITY",
    "OPTION_LIFECYCLE_AUDIT_MECHANISM_STATUS",
    "OPTION_LIFECYCLE_AUDIT_PROMOTION_AUTHORITY",
    "OPTION_LIFECYCLE_AUDIT_SCIENTIFIC_PROMOTION_ALLOWED",
    "OptionLifecycleAudit",
    "OptionLifecycleAuditArm",
    "OptionLifecycleAuditConfig",
    "OptionLifecycleAuditMaintenanceReport",
    "OptionLifecycleAuditRebindResult",
    "OptionLifecycleAuditResourceBudget",
    "OptionLifecycleAuditResult",
    "OptionLifecycleAuditState",
    "OptionLifecycleMaintenanceReport",
    "OptionLifecycleRebindResult",
    "option_semantic_digest",
]
