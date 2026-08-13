# mypy: disable-error-code="arg-type,attr-defined,call-arg,no-any-return,type-var"
"""Live, authority-free Prototype/STOMP adapter for calibrated search.

The adapter owns a normal :class:`PrototypeAgent` and a strictly secondary
``CalibratedExtendedSearchControl`` state.  It snapshots the learned legacy
one-step world model and the real STOMP option models only after Prototype has
selected and cached the action that will actually run.  Search therefore has
no route back into dispatch: its Q table is a sidecar, policy RNG is untouched,
and the one configured backup budget is shared by primitive and option
candidates.

Primitive arms settle on the next accepted real transition.  Option arms
accumulate the same discounted environment return, baseline mass, and terminal
discount as STOMP and settle only at a natural goal or timeout.  Truncation,
environment termination without that independent natural completion, an
unrepresented future anchor, and explicit rebind censor the arm without
calibration or Q updates.

An exhausted or rejected optional search transaction permanently quarantines
only the sidecar.  Valid Prototype transitions continue.  By contrast, a bad
persistent composition checksum fails closed and rolls back the whole wrapper
transaction.  This is an L0 integration mechanism, not automatic keyboard
dispatch, policy authority, evidence promotion, or a control-benefit result.
"""

from __future__ import annotations

import dataclasses
import hashlib
from typing import Any, ClassVar, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array
from jaxtyping import Bool, Float, Int, UInt

from alberta_framework.core.calibrated_extended_search_control import (
    CANDIDATE_KIND_OPTION,
    CANDIDATE_KIND_PRIMITIVE,
    CalibratedExtendedSearchArmResult,
    CalibratedExtendedSearchControl,
    CalibratedExtendedSearchControlConfig,
    CalibratedExtendedSearchControlState,
    CalibratedExtendedSearchObserveResult,
)
from alberta_framework.core.oak import OaKState
from alberta_framework.core.options import check_option_terminated
from alberta_framework.core.prototype_agent import (
    PrototypeAgent,
    PrototypeAgentConfig,
    PrototypeAgentState,
    PrototypeCandidateUpdateAuditEvidence,
    PrototypeDecision,
    PrototypeExperientialMemoryInput,
    PrototypeGradientJoyEvidence,
    PrototypePartnerPolicyFusionFeedback,
    PrototypePartnerPolicyFusionInput,
    PrototypeTransition,
    PrototypeUpdateResult,
)
from alberta_framework.core.world_model import (
    ActionConditionedWorldModel,
    ActionConditionedWorldModelState,
)

PROTOTYPE_STOMP_CALIBRATED_SEARCH_CONFIG_SCHEMA = (
    "alberta.prototype-stomp-calibrated-search.config.v1"
)
PROTOTYPE_STOMP_CALIBRATED_SEARCH_CHECKPOINT_SCHEMA = (
    "alberta.prototype-stomp-calibrated-search.state.v1"
)
PROTOTYPE_STOMP_CALIBRATED_SEARCH_MECHANISM_STATUS = "l0_live_sidecar_only"
PROTOTYPE_STOMP_CALIBRATED_SEARCH_POLICY_AUTHORITY = False
PROTOTYPE_STOMP_CALIBRATED_SEARCH_DISPATCH_AUTHORITY = False
PROTOTYPE_STOMP_CALIBRATED_SEARCH_KEYBOARD_AUTOMATION = False
PROTOTYPE_STOMP_CALIBRATED_SEARCH_CONTROL_BENEFIT_ESTABLISHED = False
PROTOTYPE_STOMP_CALIBRATED_SEARCH_SCIENTIFIC_PROMOTION_ALLOWED = False
PROTOTYPE_STOMP_CALIBRATED_SEARCH_CHECKPOINT_HOST_ONLY = True
PROTOTYPE_STOMP_CALIBRATED_SEARCH_REBIND_HOST_ONLY = True

PROTOTYPE_STOMP_CALIBRATED_SEARCH_ERROR_NONE = 0
PROTOTYPE_STOMP_CALIBRATED_SEARCH_ERROR_CAPACITY = 1
PROTOTYPE_STOMP_CALIBRATED_SEARCH_ERROR_ARM_REJECTED = 2
PROTOTYPE_STOMP_CALIBRATED_SEARCH_ERROR_OBSERVE_REJECTED = 3
PROTOTYPE_STOMP_CALIBRATED_SEARCH_ERROR_OWNERSHIP = 4

_INT32_MAX = 2_147_483_647
_OPTION_DESCRIPTOR_WIDTH = 4
_DIGEST_BYTES = 32


def _require_array(
    value: Any,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: Any,
) -> Array:
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
        if not 0 <= value <= _INT32_MAX:
            raise ValueError(f"{name} must be a non-negative signed-int32 value")
        return jnp.asarray(value, dtype=jnp.int32)
    return _require_array(value, name=name, shape=(), dtype=jnp.int32)


def _tree_select(predicate: Array, selected: Any, fallback: Any) -> Any:
    return jax.lax.cond(predicate, lambda _: selected, lambda _: fallback, None)


def _checksum_arrays(arrays: tuple[Array, ...]) -> Array:
    """Return a deterministic two-word checksum usable under JIT."""

    acc0 = jnp.uint32(0x9E3779B9)
    acc1 = jnp.uint32(0x85EBCA6B)
    offset = 1
    for value in arrays:
        array = jnp.asarray(value)
        if jax.dtypes.issubdtype(array.dtype, jax.dtypes.prng_key):
            array = jr.key_data(array)
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
        acc1 = acc1 ^ jnp.bitwise_xor.reduce(
            words ^ (indices * jnp.uint32(0x165667B1))
        )
        offset += words.shape[0]
    return jnp.stack((acc0, acc1), dtype=jnp.uint32)


def _tree_sha256(tree: object) -> Array:
    digest = hashlib.sha256()
    for leaf in jax.tree_util.tree_leaves(tree):
        array = jnp.asarray(leaf)
        if jax.dtypes.issubdtype(array.dtype, jax.dtypes.prng_key):
            array = jr.key_data(array)
        host = np.asarray(jax.device_get(array))
        digest.update(host.dtype.str.encode("ascii"))
        digest.update(np.asarray(host.shape, dtype=np.int64).tobytes())
        digest.update(host.tobytes(order="C"))
    return jnp.asarray(tuple(digest.digest()), dtype=jnp.uint8)


def _tree_nbytes(tree: object) -> int:
    total = 0
    for leaf in jax.tree_util.tree_leaves(tree):
        array = jnp.asarray(leaf)
        if jax.dtypes.issubdtype(array.dtype, jax.dtypes.prng_key):
            array = jr.key_data(array)
        total += int(array.size) * int(array.dtype.itemsize)
    return total


def _increment_words(words: Array) -> tuple[Array, Array]:
    low = words[1] + jnp.uint32(1)
    carry = (low == 0).astype(jnp.uint32)
    high = words[0] + carry
    available = ~((carry != 0) & (high == 0))
    candidate = jnp.stack((high, low), dtype=jnp.uint32)
    return jnp.where(available, candidate, words), available


def _saturating_increment(value: Array) -> Array:
    return jnp.where(value < jnp.int32(_INT32_MAX), value + jnp.int32(1), value)


def _option_descriptors(config: PrototypeAgentConfig) -> Array:
    """Encode complete STOMP subtask semantics into four exact int32 words."""

    rows: list[tuple[int, int, int, int]] = []
    for spec in config.oak.stomp.subtask_specs:
        threshold = int(np.asarray(np.float32(spec.threshold)).view(np.int32))
        scale = int(np.asarray(np.float32(spec.pseudo_reward_scale)).view(np.int32))
        rows.append((spec.feature_index, threshold, scale, spec.max_option_steps))
    return jnp.asarray(rows, dtype=jnp.int32)


@dataclasses.dataclass(frozen=True, slots=True)
class PrototypeSTOMPCalibratedSearchConfig:
    """Exact Prototype and calibrated-search configuration."""

    prototype: PrototypeAgentConfig
    search: CalibratedExtendedSearchControlConfig
    enabled: bool = True

    SCHEMA_VERSION: ClassVar[str] = PROTOTYPE_STOMP_CALIBRATED_SEARCH_CONFIG_SCHEMA

    def __post_init__(self) -> None:
        if type(self.prototype) is not PrototypeAgentConfig:
            raise TypeError("prototype must be an exact PrototypeAgentConfig")
        if type(self.search) is not CalibratedExtendedSearchControlConfig:
            raise TypeError(
                "search must be an exact CalibratedExtendedSearchControlConfig"
            )
        if type(self.enabled) is not bool:
            raise TypeError("enabled must be an exact Python bool")
        dimensions = (
            self.prototype.oak.observation_dim == self.search.observation_dim
            and self.prototype.oak.n_primitive_actions
            == self.search.n_primitive_actions
            and self.prototype.oak.n_options == self.search.n_options
        )
        if not dimensions:
            raise ValueError("Prototype/STOMP and search dimensions must match exactly")
        if not self.enabled:
            return
        if self.prototype.world_model is None:
            raise ValueError("enabled search requires the legacy Prototype world_model")
        if self.prototype.state_builder is not None or self.prototype.gru_perception is not None:
            raise ValueError(
                "enabled search currently requires the exact raw legacy representation path"
            )
        if self.prototype.prototype_feature_lifecycle is not None:
            raise ValueError("enabled search does not accept a routed feature lifecycle")
        if self.prototype.option_search_control is not None:
            raise ValueError("built-in Prototype option search would duplicate budget B")
        if self.prototype.oak.stomp.option_planning_backups_per_step != 0:
            raise ValueError("legacy STOMP planning must be zero under one total budget B")
        if self.prototype.n_dreams_per_step != 0:
            raise ValueError("Prototype dreaming would add an undeclared secondary budget")
        if self.prototype.auto_curate_every != 0:
            raise ValueError("automatic option curation requires an explicit adapter rebind")

    def to_config(self) -> dict[str, object]:
        return {
            "schema": self.SCHEMA_VERSION,
            "mechanism_status": PROTOTYPE_STOMP_CALIBRATED_SEARCH_MECHANISM_STATUS,
            "prototype": self.prototype.to_config(),
            "search": self.search.to_config(),
            "enabled": self.enabled,
            "one_total_backup_budget": self.search.backup_budget,
            "planner_rng_draws": 0,
            "q_surface": "sidecar_only",
            "policy_authority": False,
            "dispatch_authority": False,
            "automatic_keyboard_dispatch": False,
            "control_benefit_established": False,
            "scientific_promotion_allowed": False,
        }

    @classmethod
    def from_config(cls, payload: object) -> PrototypeSTOMPCalibratedSearchConfig:
        if type(payload) is not dict:
            raise ValueError("adapter config must be an exact dict")
        raw = cast(dict[object, object], payload)
        expected = {
            "schema",
            "mechanism_status",
            "prototype",
            "search",
            "enabled",
            "one_total_backup_budget",
            "planner_rng_draws",
            "q_surface",
            "policy_authority",
            "dispatch_authority",
            "automatic_keyboard_dispatch",
            "control_benefit_established",
            "scientific_promotion_allowed",
        }
        if set(raw) != expected:
            raise ValueError("adapter config fields differ from schema v1")
        if (
            raw["schema"] != cls.SCHEMA_VERSION
            or raw["mechanism_status"]
            != PROTOTYPE_STOMP_CALIBRATED_SEARCH_MECHANISM_STATUS
            or raw["planner_rng_draws"] != 0
            or raw["q_surface"] != "sidecar_only"
            or raw["policy_authority"] is not False
            or raw["dispatch_authority"] is not False
            or raw["automatic_keyboard_dispatch"] is not False
            or raw["control_benefit_established"] is not False
            or raw["scientific_promotion_allowed"] is not False
        ):
            raise ValueError("adapter config fixed fields differ")
        prototype_raw = raw["prototype"]
        if type(prototype_raw) is not dict:
            raise ValueError("prototype config must be an exact dict")
        search = CalibratedExtendedSearchControlConfig.from_config(raw["search"])
        if raw["one_total_backup_budget"] != search.backup_budget:
            raise ValueError("serialized total backup budget differs from search B")
        if type(raw["enabled"]) is not bool:
            raise ValueError("serialized enabled must be an exact bool")
        return cls(
            prototype=PrototypeAgentConfig.from_config(
                cast(dict[str, Any], prototype_raw)
            ),
            search=search,
            enabled=raw["enabled"],
        )


@chex.dataclass(frozen=True)
class PrototypeSTOMPModelSnapshot:
    """Actual learned model predictions frozen at one Prototype decision."""

    primitive_reward_predictions: Float[Array, "anchors primitive_actions"]
    primitive_discount_predictions: Float[Array, "anchors primitive_actions"]
    primitive_next_observation_predictions: Float[
        Array, "anchors primitive_actions observation_dim"
    ]
    primitive_next_anchor_probabilities: Float[
        Array, "anchors primitive_actions next_anchors"
    ]
    primitive_model_available: Bool[Array, "anchors primitive_actions"]
    primitive_model_support: Int[Array, "anchors primitive_actions"]
    option_return_predictions: Float[Array, "anchors options"]
    option_baseline_mass_predictions: Float[Array, "anchors options"]
    option_discount_predictions: Float[Array, "anchors options"]
    option_duration_predictions: Float[Array, "anchors options"]
    option_next_observation_predictions: Float[
        Array, "anchors options observation_dim"
    ]
    option_next_anchor_probabilities: Float[Array, "anchors options next_anchors"]
    option_completion_counts: Int[Array, " options"]
    option_model_available: Bool[Array, "anchors options"]
    option_model_support: Int[Array, "anchors options"]
    option_initiation_available: Bool[Array, "anchors options"]
    primitive_model_revision: Int[Array, ""]
    option_model_revision: Int[Array, ""]
    primitive_model_checksum: UInt[Array, " 2"]
    option_model_checksum: UInt[Array, " 2"]
    valid: Bool[Array, ""]


@chex.dataclass(frozen=True)
class PrototypeSTOMPCalibratedSearchState:
    """Prototype state and its identity-bound calibrated-search sidecar.

    The pending model words and unkeyed checksums bind the model snapshot that
    produced an arm for corruption detection; they do not authenticate it.
    They are frozen provenance, not live-equality locks: the
    primitive world model legitimately advances on every real primitive step
    while a multi-step option arm remains pending.  The exact current decision,
    STOMP step, base learner, option owner, option accumulators, and completion
    counts provide the live ownership checks.
    """

    prototype: PrototypeAgentState
    search: CalibratedExtendedSearchControlState
    enabled: Bool[Array, ""]
    search_unavailable: Bool[Array, ""]
    search_error: Int[Array, ""]
    revision: Int[Array, ""]
    adapter_pending: Bool[Array, ""]
    pending_current_decision_id: UInt[Array, " 4"]
    pending_expected_step_words: UInt[Array, " 2"]
    pending_base_learner_words: UInt[Array, " 2"]
    pending_primitive_model_words: UInt[Array, " 2"]
    pending_option_completion_counts: Int[Array, " options"]
    pending_primitive_model_checksum: UInt[Array, " 2"]
    pending_option_model_checksum: UInt[Array, " 2"]
    pending_elapsed_primitive_steps: Int[Array, ""]
    pending_external_return: Float[Array, ""]
    pending_baseline_mass: Float[Array, ""]
    pending_terminal_discount: Float[Array, ""]
    last_prototype_step_words: UInt[Array, " 2"]
    binding_checksum: UInt[Array, " 2"]


@chex.dataclass(frozen=True)
class PrototypeSTOMPCalibratedSearchDecisionDiagnostics:
    composed_state_valid: Bool[Array, ""]
    prototype_decision_armed: Bool[Array, ""]
    search_enabled: Bool[Array, ""]
    search_available: Bool[Array, ""]
    exact_real_anchor: Bool[Array, ""]
    arm_attempted: Bool[Array, ""]
    arm_applied: Bool[Array, ""]
    sidecar_failed: Bool[Array, ""]
    policy_authority: Bool[Array, ""]
    keyboard_dispatch_applied: Bool[Array, ""]
    rng_draw_count: Int[Array, ""]


@chex.dataclass(frozen=True)
class PrototypeSTOMPCalibratedSearchUpdateDiagnostics:
    composed_state_valid_before: Bool[Array, ""]
    prototype_transition_applied: Bool[Array, ""]
    pending_before: Bool[Array, ""]
    ownership_binding_matches: Bool[Array, ""]
    resolution_attempted: Bool[Array, ""]
    natural_resolution: Bool[Array, ""]
    censored_resolution: Bool[Array, ""]
    observe_applied: Bool[Array, ""]
    arm_attempted: Bool[Array, ""]
    arm_applied: Bool[Array, ""]
    search_unavailable: Bool[Array, ""]
    sidecar_failed_this_step: Bool[Array, ""]
    prototype_retained_after_sidecar_failure: Bool[Array, ""]
    composed_state_valid_after: Bool[Array, ""]
    transaction_committed: Bool[Array, ""]
    backup_attempt_budget: Int[Array, ""]
    planner_rng_draw_count: Int[Array, ""]
    policy_authority: Bool[Array, ""]
    keyboard_dispatch_applied: Bool[Array, ""]


@chex.dataclass(frozen=True)
class PrototypeSTOMPCalibratedSearchStartResult:
    state: PrototypeSTOMPCalibratedSearchState
    decision: PrototypeDecision
    model_snapshot: PrototypeSTOMPModelSnapshot
    search_arm: Any
    diagnostics: PrototypeSTOMPCalibratedSearchDecisionDiagnostics


@chex.dataclass(frozen=True)
class PrototypeSTOMPCalibratedSearchUpdateResult:
    state: PrototypeSTOMPCalibratedSearchState
    prototype: PrototypeUpdateResult
    decision: PrototypeDecision
    model_snapshot: PrototypeSTOMPModelSnapshot
    search_observe: Any
    search_arm: Any
    diagnostics: PrototypeSTOMPCalibratedSearchUpdateDiagnostics


@chex.dataclass(frozen=True)
class PrototypeSTOMPCalibratedSearchRebindResult:
    state: PrototypeSTOMPCalibratedSearchState
    pending_censored: Bool[Array, ""]
    full_search_reset: Bool[Array, ""]
    preserved_option_slots: Bool[Array, " options"]
    reset_option_slots: Bool[Array, " options"]
    transaction_applied: Bool[Array, ""]


@chex.dataclass(frozen=True)
class PrototypeSTOMPCalibratedSearchArrayResult:
    state: PrototypeSTOMPCalibratedSearchState
    actions: Int[Array, " steps"]
    prototype_transition_applied: Bool[Array, " steps"]
    natural_resolutions: Bool[Array, " steps"]
    censored_resolutions: Bool[Array, " steps"]
    search_unavailable: Bool[Array, " steps"]
    transaction_committed: Bool[Array, " steps"]


@dataclasses.dataclass(frozen=True, slots=True)
class PrototypeSTOMPCalibratedSearchResourceBudget:
    prototype_persistent_state_nbytes: int
    search_persistent_state_nbytes: int
    adapter_binding_nbytes: int
    total_persistent_state_nbytes: int
    anchor_capacity: int
    primitive_model_predictions_per_decision: int
    option_model_predictions_per_decision: int
    primitive_model_forward_calls_per_decision: int
    option_outcome_matrix_vector_products_per_decision: int
    total_secondary_backup_attempts_per_resolution: int
    primitive_and_option_share_one_budget: bool
    prototype_updates_per_transition: int
    external_model_updates_per_transition: int
    additional_rng_draws_per_init: int
    additional_rng_draws_per_start: int
    additional_rng_draws_per_transition: int
    planner_rng_draws_total: int
    max_observations: int
    search_exhaustion_can_block_prototype: bool
    q_surface_sidecar_only: bool
    policy_authority: bool
    dispatch_authority: bool
    automatic_keyboard_dispatch: bool
    control_benefit_established: bool
    scientific_promotion_allowed: bool
    checkpoint_schema: str
    checkpoint_host_only: bool
    rebind_host_only: bool

    def to_config(self) -> dict[str, object]:
        return dataclasses.asdict(self)


class PrototypeSTOMPCalibratedSearchAgent:
    """Compose live Prototype learning with non-authoritative calibrated search."""

    def __init__(self, config: PrototypeSTOMPCalibratedSearchConfig) -> None:
        if type(config) is not PrototypeSTOMPCalibratedSearchConfig:
            raise TypeError(
                "config must be an exact PrototypeSTOMPCalibratedSearchConfig"
            )
        self._config = config
        self._prototype = PrototypeAgent(config.prototype)
        self._controller = CalibratedExtendedSearchControl(config.search)
        self._world_model = (
            ActionConditionedWorldModel(config.prototype.world_model)
            if config.enabled and config.prototype.world_model is not None
            else None
        )
        self._descriptors = _option_descriptors(config.prototype)

    @property
    def config(self) -> PrototypeSTOMPCalibratedSearchConfig:
        return self._config

    @property
    def prototype(self) -> PrototypeAgent:
        return self._prototype

    @property
    def controller(self) -> CalibratedExtendedSearchControl:
        return self._controller

    @property
    def option_descriptors(self) -> Array:
        return self._descriptors

    def to_config(self) -> dict[str, object]:
        return self._config.to_config()

    @classmethod
    def from_config(cls, payload: object) -> PrototypeSTOMPCalibratedSearchAgent:
        return cls(PrototypeSTOMPCalibratedSearchConfig.from_config(payload))

    def _oak(self, state: PrototypeAgentState) -> OaKState:
        if type(state.oak_state) is not OaKState:
            raise TypeError("adapter requires Prototype's exact public OaKState slot")
        return state.oak_state

    def _world(self, state: PrototypeAgentState) -> ActionConditionedWorldModelState:
        if type(state.world_model_state) is not ActionConditionedWorldModelState:
            raise TypeError("adapter requires the exact legacy world-model state")
        return state.world_model_state

    def _payload_arrays(
        self, state: PrototypeSTOMPCalibratedSearchState
    ) -> tuple[Array, ...]:
        prototype = tuple(cast(Array, leaf) for leaf in jax.tree_util.tree_leaves(state.prototype))
        search = tuple(cast(Array, leaf) for leaf in jax.tree_util.tree_leaves(state.search))
        return (
            *prototype,
            *search,
            state.enabled,
            state.search_unavailable,
            state.search_error,
            state.revision,
            state.adapter_pending,
            state.pending_current_decision_id,
            state.pending_expected_step_words,
            state.pending_base_learner_words,
            state.pending_primitive_model_words,
            state.pending_option_completion_counts,
            state.pending_primitive_model_checksum,
            state.pending_option_model_checksum,
            state.pending_elapsed_primitive_steps,
            state.pending_external_return,
            state.pending_baseline_mass,
            state.pending_terminal_discount,
            state.last_prototype_step_words,
        )

    def _with_checksum(
        self, state: PrototypeSTOMPCalibratedSearchState
    ) -> PrototypeSTOMPCalibratedSearchState:
        return cast(
            PrototypeSTOMPCalibratedSearchState,
            state.replace(binding_checksum=_checksum_arrays(self._payload_arrays(state))),
        )

    def _check_state_contract(self, state: PrototypeSTOMPCalibratedSearchState) -> None:
        if type(state) is not PrototypeSTOMPCalibratedSearchState:
            raise TypeError("state must be an exact PrototypeSTOMPCalibratedSearchState")
        n = self._config.search.n_options
        contracts = (
            (state.enabled, "enabled", (), jnp.bool_),
            (state.search_unavailable, "search_unavailable", (), jnp.bool_),
            (state.search_error, "search_error", (), jnp.int32),
            (state.revision, "revision", (), jnp.int32),
            (state.adapter_pending, "adapter_pending", (), jnp.bool_),
            (state.pending_current_decision_id, "pending_current_decision_id", (4,), jnp.uint32),
            (state.pending_expected_step_words, "pending_expected_step_words", (2,), jnp.uint32),
            (state.pending_base_learner_words, "pending_base_learner_words", (2,), jnp.uint32),
            (
                state.pending_primitive_model_words,
                "pending_primitive_model_words",
                (2,),
                jnp.uint32,
            ),
            (
                state.pending_option_completion_counts,
                "pending_option_completion_counts",
                (n,),
                jnp.int32,
            ),
            (
                state.pending_primitive_model_checksum,
                "pending_primitive_model_checksum",
                (2,),
                jnp.uint32,
            ),
            (
                state.pending_option_model_checksum,
                "pending_option_model_checksum",
                (2,),
                jnp.uint32,
            ),
            (
                state.pending_elapsed_primitive_steps,
                "pending_elapsed_primitive_steps",
                (),
                jnp.int32,
            ),
            (state.pending_external_return, "pending_external_return", (), jnp.float32),
            (state.pending_baseline_mass, "pending_baseline_mass", (), jnp.float32),
            (state.pending_terminal_discount, "pending_terminal_discount", (), jnp.float32),
            (state.last_prototype_step_words, "last_prototype_step_words", (2,), jnp.uint32),
            (state.binding_checksum, "binding_checksum", (2,), jnp.uint32),
        )
        for value, name, shape, dtype in contracts:
            _require_array(value, name=f"state.{name}", shape=shape, dtype=dtype)

    def _pending_values_valid(self, state: PrototypeSTOMPCalibratedSearchState) -> Array:
        pending_values = (
            jnp.isfinite(state.pending_external_return)
            & jnp.isfinite(state.pending_baseline_mass)
            & jnp.isfinite(state.pending_terminal_discount)
            & (state.pending_elapsed_primitive_steps >= 0)
            & (state.pending_elapsed_primitive_steps <= self._config.search.max_observations)
            & (state.pending_baseline_mass >= 0.0)
            & (state.pending_terminal_discount >= 0.0)
            & (state.pending_terminal_discount <= 1.0)
            & jnp.all(state.pending_option_completion_counts >= 0)
        )
        blank = (
            (state.pending_elapsed_primitive_steps == 0)
            & (state.pending_external_return == 0.0)
            & (state.pending_baseline_mass == 0.0)
            & (state.pending_terminal_discount == 1.0)
        )
        return jnp.where(state.adapter_pending, pending_values, blank)

    def validate_state(
        self, state: PrototypeSTOMPCalibratedSearchState
    ) -> Bool[Array, ""]:
        """Validate the complete persistent composition and live bindings."""

        self._check_state_contract(state)
        search_valid = self._controller.validate_state(
            state.search,
            representation_generation=state.search.representation_generation,
            source_digest=state.search.source_digest,
            option_descriptors=state.search.option_descriptors,
            option_generations=state.search.option_generations,
        )
        prototype_valid = self._prototype.validate_state(state.prototype)
        oak = self._oak(state.prototype)
        available_pending_binding = (~state.search_unavailable) & state.adapter_pending
        live_pending = (
            jnp.array_equal(
                state.pending_current_decision_id,
                state.prototype.current_decision_id,
            )
            & jnp.array_equal(
                state.pending_expected_step_words,
                state.prototype.step_words,
            )
            & jnp.where(
                state.search.pending_executed_kind == CANDIDATE_KIND_OPTION,
                oak.stomp_state.executing_option
                == state.search.pending_executed_index,
                oak.stomp_state.executing_option < 0,
            )
        )
        error_valid = (
            (state.search_error >= PROTOTYPE_STOMP_CALIBRATED_SEARCH_ERROR_NONE)
            & (state.search_error <= PROTOTYPE_STOMP_CALIBRATED_SEARCH_ERROR_OWNERSHIP)
            & (
                state.search_unavailable
                == (state.search_error != PROTOTYPE_STOMP_CALIBRATED_SEARCH_ERROR_NONE)
            )
        )
        return (
            prototype_valid
            & search_valid
            & (state.enabled == self._config.enabled)
            & jnp.array_equal(state.search.option_descriptors, self._descriptors)
            & (state.adapter_pending == state.search.pending)
            & self._pending_values_valid(state)
            & jnp.where(available_pending_binding, live_pending, True)
            & jnp.array_equal(
                state.last_prototype_step_words,
                state.prototype.step_words,
            )
            & (state.revision >= 0)
            & error_valid
            & jnp.array_equal(
                state.binding_checksum,
                _checksum_arrays(self._payload_arrays(state)),
            )
        )

    def _q_values(self, prototype: PrototypeAgentState, anchors: Array, active: Array) -> Array:
        oak = self._oak(prototype)
        q_values = jax.vmap(
            lambda observation: self._prototype.oak_agent.base_q_values(
                oak, observation
            )
        )(anchors)
        return jnp.where(active[:, None], q_values, jnp.zeros_like(q_values))

    def init(
        self,
        key: Array,
        *,
        anchor_bank: Array,
        anchor_active: Array,
        source_digest: Array,
        representation_generation: int | Array,
        option_generations: Array | None = None,
        lifecycle_id: Array | None = None,
    ) -> PrototypeSTOMPCalibratedSearchState:
        """Initialize Prototype with the caller's exact key and no sidecar RNG."""

        cfg = self._config.search
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
        source = _require_array(
            source_digest, name="source_digest", shape=(2,), dtype=jnp.uint32
        )
        generation = _int32_scalar(
            representation_generation, name="representation_generation"
        )
        generations = (
            jnp.zeros((cfg.n_options,), dtype=jnp.int32)
            if option_generations is None
            else _require_array(
                option_generations,
                name="option_generations",
                shape=(cfg.n_options,),
                dtype=jnp.int32,
            )
        )
        if bool(jax.device_get(jnp.any(generations < 0))):
            raise ValueError("option_generations must be non-negative")
        prototype = self._prototype.init(key, lifecycle_id=lifecycle_id)
        oak = self._oak(prototype)
        if self._config.enabled:
            world = self._world(prototype)
            q_values = self._q_values(prototype, anchors, active)
            primitive_revision = world.step_count
        else:
            q_values = jnp.zeros(
                (cfg.anchor_capacity, cfg.n_extended_actions), dtype=jnp.float32
            )
            primitive_revision = jnp.asarray(0, dtype=jnp.int32)
        search = self._controller.init(
            anchor_bank=anchors,
            anchor_active=active,
            q_values=q_values,
            option_descriptors=self._descriptors,
            option_generations=generations,
            representation_generation=generation,
            source_digest=source,
            primitive_model_revision=primitive_revision,
            option_model_revision=oak.stomp_state.step_count,
        )
        state = PrototypeSTOMPCalibratedSearchState(
            prototype=prototype,
            search=search,
            enabled=jnp.asarray(self._config.enabled, dtype=jnp.bool_),
            search_unavailable=jnp.asarray(False, dtype=jnp.bool_),
            search_error=jnp.asarray(
                PROTOTYPE_STOMP_CALIBRATED_SEARCH_ERROR_NONE, dtype=jnp.int32
            ),
            revision=jnp.asarray(0, dtype=jnp.int32),
            adapter_pending=jnp.asarray(False, dtype=jnp.bool_),
            pending_current_decision_id=jnp.zeros((4,), dtype=jnp.uint32),
            pending_expected_step_words=jnp.zeros((2,), dtype=jnp.uint32),
            pending_base_learner_words=jnp.zeros((2,), dtype=jnp.uint32),
            pending_primitive_model_words=jnp.zeros((2,), dtype=jnp.uint32),
            pending_option_completion_counts=jnp.zeros(
                (cfg.n_options,), dtype=jnp.int32
            ),
            pending_primitive_model_checksum=jnp.zeros((2,), dtype=jnp.uint32),
            pending_option_model_checksum=jnp.zeros((2,), dtype=jnp.uint32),
            pending_elapsed_primitive_steps=jnp.asarray(0, dtype=jnp.int32),
            pending_external_return=jnp.asarray(0.0, dtype=jnp.float32),
            pending_baseline_mass=jnp.asarray(0.0, dtype=jnp.float32),
            pending_terminal_discount=jnp.asarray(1.0, dtype=jnp.float32),
            last_prototype_step_words=prototype.step_words,
            binding_checksum=jnp.zeros((2,), dtype=jnp.uint32),
        )
        return self._with_checksum(state)

    def _nearest_anchor_probabilities(
        self,
        predictions: Array,
        anchors: Array,
        active: Array,
    ) -> tuple[Array, Array]:
        flat = predictions.reshape((-1, self._config.search.observation_dim))

        def one(prediction: Array) -> tuple[Array, Array]:
            finite = jnp.all(jnp.isfinite(prediction))
            safe = jnp.where(finite, prediction, jnp.zeros_like(prediction))
            distances = jnp.sum(jnp.square(anchors - safe[None, :]), axis=1)
            masked = jnp.where(active, distances, jnp.asarray(jnp.inf, jnp.float32))
            index = jnp.argmin(masked).astype(jnp.int32)
            probability = jax.nn.one_hot(
                index, self._config.search.anchor_capacity, dtype=jnp.float32
            )
            return probability, finite

        probabilities, finite = jax.vmap(one)(flat)
        return (
            probabilities.reshape((*predictions.shape[:-1], self._config.search.anchor_capacity)),
            finite.reshape(predictions.shape[:-1]),
        )

    def _blank_snapshot(
        self, state: PrototypeSTOMPCalibratedSearchState
    ) -> PrototypeSTOMPModelSnapshot:
        cfg = self._config.search
        m, k, n, d = (
            cfg.anchor_capacity,
            cfg.n_primitive_actions,
            cfg.n_options,
            cfg.observation_dim,
        )
        primitive_next = jnp.broadcast_to(
            jax.nn.one_hot(0, m, dtype=jnp.float32), (m, k, m)
        )
        option_next = jnp.broadcast_to(
            jax.nn.one_hot(0, m, dtype=jnp.float32), (m, n, m)
        )
        return PrototypeSTOMPModelSnapshot(
            primitive_reward_predictions=jnp.zeros((m, k), dtype=jnp.float32),
            primitive_discount_predictions=jnp.zeros((m, k), dtype=jnp.float32),
            primitive_next_observation_predictions=jnp.zeros((m, k, d), dtype=jnp.float32),
            primitive_next_anchor_probabilities=primitive_next,
            primitive_model_available=jnp.zeros((m, k), dtype=jnp.bool_),
            primitive_model_support=jnp.zeros((m, k), dtype=jnp.int32),
            option_return_predictions=jnp.zeros((m, n), dtype=jnp.float32),
            option_baseline_mass_predictions=jnp.zeros((m, n), dtype=jnp.float32),
            option_discount_predictions=jnp.ones((m, n), dtype=jnp.float32),
            option_duration_predictions=jnp.zeros((m, n), dtype=jnp.float32),
            option_next_observation_predictions=jnp.zeros((m, n, d), dtype=jnp.float32),
            option_next_anchor_probabilities=option_next,
            option_completion_counts=jnp.zeros((n,), dtype=jnp.int32),
            option_model_available=jnp.zeros((m, n), dtype=jnp.bool_),
            option_model_support=jnp.zeros((m, n), dtype=jnp.int32),
            option_initiation_available=jnp.zeros((m, n), dtype=jnp.bool_),
            primitive_model_revision=jnp.asarray(0, dtype=jnp.int32),
            option_model_revision=jnp.asarray(0, dtype=jnp.int32),
            primitive_model_checksum=jnp.zeros((2,), dtype=jnp.uint32),
            option_model_checksum=jnp.zeros((2,), dtype=jnp.uint32),
            valid=jnp.asarray(True, dtype=jnp.bool_),
        )

    def model_snapshot(
        self, state: PrototypeSTOMPCalibratedSearchState
    ) -> PrototypeSTOMPModelSnapshot:
        """Read actual learned primitive and option models without mutation or RNG."""

        if not self._config.enabled:
            return self._blank_snapshot(state)
        cfg = self._config.search
        prototype = state.prototype
        oak = self._oak(prototype)
        stomp = oak.stomp_state
        world = self._world(prototype)
        model = self._world_model
        if model is None:
            raise RuntimeError("enabled adapter lost its legacy world model")
        anchors = state.search.anchor_bank
        active = state.search.anchor_active
        actions = jnp.arange(cfg.n_primitive_actions, dtype=jnp.int32)

        def anchor_predictions(anchor: Array) -> Any:
            return jax.vmap(lambda action: model.predict(world, anchor, action))(actions)

        primitive = jax.vmap(anchor_predictions)(anchors)
        primitive_next_probabilities, primitive_next_finite = (
            self._nearest_anchor_probabilities(
                primitive.next_observation, anchors, active
            )
        )
        primitive_finite = (
            primitive_next_finite
            & jnp.isfinite(primitive.reward)
            & jnp.isfinite(primitive.discount)
        )
        primitive_support_scalar = jnp.clip(
            world.step_count, 0, cfg.max_observations
        ).astype(jnp.int32)
        primitive_support = jnp.full(
            (cfg.anchor_capacity, cfg.n_primitive_actions),
            primitive_support_scalar,
            dtype=jnp.int32,
        )
        primitive_available = (
            active[:, None]
            & (world.step_count > 0)
            & primitive_finite
        )

        models = stomp.option_models
        predicted_delta = jnp.einsum(
            "nij,mj->mni", models.next_state_weights, anchors
        )
        option_next_observations = anchors[:, None, :] + predicted_delta
        option_next_probabilities, option_next_finite = (
            self._nearest_anchor_probabilities(
                option_next_observations, anchors, active
            )
        )
        option_returns = jnp.broadcast_to(
            models.env_return_ema[None, :],
            (cfg.anchor_capacity, cfg.n_options),
        )
        option_mass = jnp.broadcast_to(
            models.baseline_mass_ema[None, :],
            (cfg.anchor_capacity, cfg.n_options),
        )
        option_discounts = jnp.broadcast_to(
            models.discount_ema[None, :],
            (cfg.anchor_capacity, cfg.n_options),
        )
        option_durations = jnp.broadcast_to(
            models.duration_ema[None, :],
            (cfg.anchor_capacity, cfg.n_options),
        )
        completion_counts = models.n_completions
        option_support = jnp.broadcast_to(
            jnp.clip(completion_counts, 0, cfg.max_observations)[None, :],
            (cfg.anchor_capacity, cfg.n_options),
        ).astype(jnp.int32)
        option_finite = (
            option_next_finite
            & jnp.isfinite(option_returns)
            & jnp.isfinite(option_mass)
            & jnp.isfinite(option_discounts)
            & jnp.isfinite(option_durations)
        )
        option_available = (
            active[:, None]
            & (completion_counts[None, :] > 0)
            & option_finite
        )
        initiation = jnp.broadcast_to(
            active[:, None], (cfg.anchor_capacity, cfg.n_options)
        )
        primitive_checksum = _checksum_arrays(
            tuple(cast(Array, leaf) for leaf in jax.tree_util.tree_leaves(world))
        )
        option_checksum = _checksum_arrays(
            (
                models.env_return_ema,
                models.duration_ema,
                models.baseline_mass_ema,
                models.discount_ema,
                models.next_state_weights,
                models.n_completions,
            )
        )
        valid = (
            jnp.all(primitive_finite | ~active[:, None])
            & jnp.all(option_finite | ~active[:, None])
            & jnp.all((primitive.discount >= 0.0) & (primitive.discount <= 1.0))
            & jnp.all(option_mass >= 0.0)
            & jnp.all((option_discounts >= 0.0) & (option_discounts <= 1.0))
            & jnp.all(completion_counts >= 0)
        )
        return PrototypeSTOMPModelSnapshot(
            primitive_reward_predictions=jnp.where(
                primitive_finite, primitive.reward, 0.0
            ),
            primitive_discount_predictions=jnp.where(
                primitive_finite, primitive.discount, 0.0
            ),
            primitive_next_observation_predictions=jnp.where(
                primitive_finite[..., None], primitive.next_observation, 0.0
            ),
            primitive_next_anchor_probabilities=primitive_next_probabilities,
            primitive_model_available=primitive_available,
            primitive_model_support=primitive_support,
            option_return_predictions=jnp.where(option_finite, option_returns, 0.0),
            option_baseline_mass_predictions=jnp.where(option_finite, option_mass, 0.0),
            option_discount_predictions=jnp.where(option_finite, option_discounts, 0.0),
            option_duration_predictions=jnp.where(option_finite, option_durations, 0.0),
            option_next_observation_predictions=jnp.where(
                option_finite[..., None], option_next_observations, 0.0
            ),
            option_next_anchor_probabilities=option_next_probabilities,
            option_completion_counts=completion_counts,
            option_model_available=option_available,
            option_model_support=option_support,
            option_initiation_available=initiation,
            primitive_model_revision=world.step_count,
            option_model_revision=stomp.step_count,
            primitive_model_checksum=primitive_checksum,
            option_model_checksum=option_checksum,
            valid=valid,
        )

    def _arm_next(
        self,
        state: PrototypeSTOMPCalibratedSearchState,
    ) -> tuple[
        PrototypeSTOMPCalibratedSearchState,
        PrototypeSTOMPModelSnapshot,
        CalibratedExtendedSearchArmResult | None,
        PrototypeSTOMPCalibratedSearchDecisionDiagnostics,
    ]:
        decision = self._prototype.decision(state.prototype)
        if not self._config.enabled:
            snapshot = self._blank_snapshot(state)
            diagnostics = PrototypeSTOMPCalibratedSearchDecisionDiagnostics(
                composed_state_valid=self.validate_state(state),
                prototype_decision_armed=decision.armed,
                search_enabled=jnp.asarray(False, dtype=jnp.bool_),
                search_available=jnp.asarray(False, dtype=jnp.bool_),
                exact_real_anchor=jnp.asarray(False, dtype=jnp.bool_),
                arm_attempted=jnp.asarray(False, dtype=jnp.bool_),
                arm_applied=jnp.asarray(False, dtype=jnp.bool_),
                sidecar_failed=jnp.asarray(False, dtype=jnp.bool_),
                policy_authority=jnp.asarray(False, dtype=jnp.bool_),
                keyboard_dispatch_applied=jnp.asarray(False, dtype=jnp.bool_),
                rng_draw_count=jnp.asarray(0, dtype=jnp.int32),
            )
            return state, snapshot, None, diagnostics

        snapshot = self.model_snapshot(state)
        cfg = self._config.search
        oak = self._oak(state.prototype)
        stomp = oak.stomp_state
        matches = (
            jnp.all(
                state.search.anchor_bank
                == state.prototype.current_representation[None, :],
                axis=1,
            )
            & state.search.anchor_active
        )
        exact_anchor = jnp.sum(matches.astype(jnp.int32)) == 1
        anchor_index = jnp.argmax(matches).astype(jnp.int32)
        executing = stomp.executing_option >= 0
        option_start = executing & (stomp.option_steps == 0)
        primitive_start = (~executing) & (
            stomp.base_last_action < cfg.n_primitive_actions
        )
        startable = option_start | primitive_start
        kind = jnp.where(
            executing,
            jnp.int32(CANDIDATE_KIND_OPTION),
            jnp.int32(CANDIDATE_KIND_PRIMITIVE),
        )
        index = jnp.where(
            executing, stomp.executing_option, state.prototype.current_action
        ).astype(jnp.int32)
        attempted = (
            state.enabled
            & (~state.search_unavailable)
            & (~state.search.pending)
            & decision.armed
            & exact_anchor
            & startable
        )
        arm = self._controller.arm(
            state.search,
            decision_id=decision.decision_id,
            decision_observation=state.prototype.current_representation,
            decision_anchor_index=anchor_index,
            executed_kind=kind,
            executed_index=index,
            average_reward=stomp.base_average_reward,
            primitive_reward_predictions=snapshot.primitive_reward_predictions,
            primitive_discount_predictions=snapshot.primitive_discount_predictions,
            primitive_next_anchor_probabilities=(
                snapshot.primitive_next_anchor_probabilities
            ),
            primitive_model_available=snapshot.primitive_model_available,
            primitive_model_support=snapshot.primitive_model_support,
            option_return_predictions=snapshot.option_return_predictions,
            option_baseline_mass_predictions=(
                snapshot.option_baseline_mass_predictions
            ),
            option_discount_predictions=snapshot.option_discount_predictions,
            option_next_anchor_probabilities=snapshot.option_next_anchor_probabilities,
            option_model_available=snapshot.option_model_available,
            option_model_support=snapshot.option_model_support,
            option_initiation_available=snapshot.option_initiation_available,
            representation_generation=state.search.representation_generation,
            source_digest=state.search.source_digest,
            option_descriptors=state.search.option_descriptors,
            option_generations=state.search.option_generations,
            learner_revision=state.search.learner_revision,
            primitive_model_revision=snapshot.primitive_model_revision,
            option_model_revision=snapshot.option_model_revision,
        )
        applied = attempted & snapshot.valid & arm.diagnostics.transaction_valid
        failed = attempted & ~applied
        capacity_error = attempted & ~arm.diagnostics.capacity_available
        error = jnp.where(
            capacity_error,
            jnp.int32(PROTOTYPE_STOMP_CALIBRATED_SEARCH_ERROR_CAPACITY),
            jnp.int32(PROTOTYPE_STOMP_CALIBRATED_SEARCH_ERROR_ARM_REJECTED),
        )
        proposed = cast(
            PrototypeSTOMPCalibratedSearchState,
            state.replace(
                search=cast(
                    CalibratedExtendedSearchControlState,
                    _tree_select(applied, arm.state, state.search),
                ),
                search_unavailable=state.search_unavailable | failed,
                search_error=jnp.where(
                    failed,
                    error,
                    state.search_error,
                ).astype(jnp.int32),
                adapter_pending=state.adapter_pending | applied,
                pending_current_decision_id=jnp.where(
                    applied,
                    decision.decision_id,
                    state.pending_current_decision_id,
                ),
                pending_expected_step_words=jnp.where(
                    applied,
                    state.prototype.step_words,
                    state.pending_expected_step_words,
                ),
                pending_base_learner_words=jnp.where(
                    applied,
                    stomp.base_learner_state.step_words,
                    state.pending_base_learner_words,
                ),
                pending_primitive_model_words=jnp.where(
                    applied,
                    self._world(state.prototype).step_words,
                    state.pending_primitive_model_words,
                ),
                pending_option_completion_counts=jnp.where(
                    applied,
                    stomp.option_models.n_completions,
                    state.pending_option_completion_counts,
                ),
                pending_primitive_model_checksum=jnp.where(
                    applied,
                    snapshot.primitive_model_checksum,
                    state.pending_primitive_model_checksum,
                ),
                pending_option_model_checksum=jnp.where(
                    applied,
                    snapshot.option_model_checksum,
                    state.pending_option_model_checksum,
                ),
                pending_elapsed_primitive_steps=jnp.where(
                    applied,
                    jnp.int32(0),
                    state.pending_elapsed_primitive_steps,
                ),
                pending_external_return=jnp.where(
                    applied, jnp.float32(0.0), state.pending_external_return
                ),
                pending_baseline_mass=jnp.where(
                    applied, jnp.float32(0.0), state.pending_baseline_mass
                ),
                pending_terminal_discount=jnp.where(
                    applied, jnp.float32(1.0), state.pending_terminal_discount
                ),
            ),
        )
        proposed = self._with_checksum(proposed)
        diagnostics = PrototypeSTOMPCalibratedSearchDecisionDiagnostics(
            composed_state_valid=self.validate_state(proposed),
            prototype_decision_armed=decision.armed,
            search_enabled=state.enabled,
            search_available=~proposed.search_unavailable,
            exact_real_anchor=exact_anchor,
            arm_attempted=attempted,
            arm_applied=applied,
            sidecar_failed=failed,
            policy_authority=jnp.asarray(False, dtype=jnp.bool_),
            keyboard_dispatch_applied=jnp.asarray(False, dtype=jnp.bool_),
            rng_draw_count=jnp.asarray(0, dtype=jnp.int32),
        )
        return proposed, snapshot, arm, diagnostics

    def start(
        self,
        state: PrototypeSTOMPCalibratedSearchState,
        initial_observation: Array,
    ) -> PrototypeSTOMPCalibratedSearchStartResult:
        """Start raw Prototype, then freeze the selected pre-outcome search arm."""

        valid_before = self.validate_state(state)
        prototype = self._prototype.start(state.prototype, initial_observation)
        base_applied = prototype.started & self._prototype.validate_state(prototype)
        base_state = cast(
            PrototypeSTOMPCalibratedSearchState,
            state.replace(
                prototype=prototype,
                revision=_saturating_increment(state.revision),
                last_prototype_step_words=prototype.step_words,
            ),
        )
        base_state = self._with_checksum(base_state)
        armed_state, snapshot, arm, diagnostics = self._arm_next(base_state)
        candidate_valid = self.validate_state(armed_state)
        committed = valid_before & base_applied & candidate_valid
        final = cast(
            PrototypeSTOMPCalibratedSearchState,
            _tree_select(committed, armed_state, state),
        )
        decision = self._prototype.decision(final.prototype)
        return PrototypeSTOMPCalibratedSearchStartResult(
            state=final,
            decision=decision,
            model_snapshot=snapshot,
            search_arm=arm,
            diagnostics=diagnostics.replace(
                composed_state_valid=committed & diagnostics.composed_state_valid,
                prototype_decision_armed=committed & decision.armed,
                arm_applied=committed & diagnostics.arm_applied,
            ),
        )

    def decision(
        self, state: PrototypeSTOMPCalibratedSearchState
    ) -> PrototypeDecision:
        """Return Prototype's cached decision; search never rewrites it."""

        valid = self.validate_state(state)
        decision = self._prototype.decision(state.prototype)
        return PrototypeDecision(
            observation=decision.observation,
            action=jnp.where(valid, decision.action, jnp.int32(-1)),
            decision_id=decision.decision_id,
            armed=valid & decision.armed,
        )

    def _clear_pending_fields(
        self, state: PrototypeSTOMPCalibratedSearchState
    ) -> PrototypeSTOMPCalibratedSearchState:
        return cast(
            PrototypeSTOMPCalibratedSearchState,
            state.replace(
                adapter_pending=jnp.asarray(False, dtype=jnp.bool_),
                pending_current_decision_id=jnp.zeros((4,), dtype=jnp.uint32),
                pending_expected_step_words=jnp.zeros((2,), dtype=jnp.uint32),
                pending_base_learner_words=jnp.zeros((2,), dtype=jnp.uint32),
                pending_primitive_model_words=jnp.zeros((2,), dtype=jnp.uint32),
                pending_option_completion_counts=jnp.zeros(
                    (self._config.search.n_options,), dtype=jnp.int32
                ),
                pending_primitive_model_checksum=jnp.zeros((2,), dtype=jnp.uint32),
                pending_option_model_checksum=jnp.zeros((2,), dtype=jnp.uint32),
                pending_elapsed_primitive_steps=jnp.asarray(0, dtype=jnp.int32),
                pending_external_return=jnp.asarray(0.0, dtype=jnp.float32),
                pending_baseline_mass=jnp.asarray(0.0, dtype=jnp.float32),
                pending_terminal_discount=jnp.asarray(1.0, dtype=jnp.float32),
            ),
        )

    def _resolve_pending(
        self,
        state: PrototypeSTOMPCalibratedSearchState,
        transition: PrototypeTransition,
        next_prototype: PrototypeAgentState,
        prototype_applied: Array,
    ) -> tuple[
        PrototypeSTOMPCalibratedSearchState,
        CalibratedExtendedSearchObserveResult | None,
        Array,
        Array,
        Array,
        Array,
        Array,
    ]:
        if not self._config.enabled:
            updated = cast(
                PrototypeSTOMPCalibratedSearchState,
                state.replace(
                    prototype=next_prototype,
                    revision=_saturating_increment(state.revision),
                    last_prototype_step_words=next_prototype.step_words,
                ),
            )
            return (
                self._with_checksum(updated),
                None,
                jnp.asarray(False),
                jnp.asarray(False),
                jnp.asarray(False),
                jnp.asarray(True),
                jnp.asarray(False),
            )

        cfg = self._config.search
        pre_oak = self._oak(state.prototype)
        pre_stomp = pre_oak.stomp_state
        pending = state.adapter_pending & state.search.pending
        is_option = state.search.pending_executed_kind == CANDIDATE_KIND_OPTION
        option_index = jnp.clip(
            state.search.pending_executed_index, 0, cfg.n_options - 1
        )
        decision_matches = jnp.array_equal(
            transition.decision_id, state.pending_current_decision_id
        )
        step_matches = jnp.array_equal(
            state.prototype.step_words, state.pending_expected_step_words
        )
        learner_matches = jnp.array_equal(
            pre_stomp.base_learner_state.step_words,
            state.pending_base_learner_words,
        )
        option_counts_match = jnp.array_equal(
            pre_stomp.option_models.n_completions,
            state.pending_option_completion_counts,
        )
        primitive_owner = (
            (pre_stomp.executing_option < 0)
            & (state.prototype.current_action == state.search.pending_executed_index)
            & (state.pending_elapsed_primitive_steps == 0)
        )
        option_accumulators_match = (
            (pre_stomp.executing_option == state.search.pending_executed_index)
            & (pre_stomp.option_steps == state.pending_elapsed_primitive_steps)
            & (pre_stomp.option_env_cumreward == state.pending_external_return)
            & (pre_stomp.option_baseline_mass == state.pending_baseline_mass)
            & (pre_stomp.option_discount == state.pending_terminal_discount)
        )
        ownership = (
            decision_matches
            & step_matches
            & learner_matches
            & option_counts_match
            & jnp.where(is_option, option_accumulators_match, primitive_owner)
        )
        next_return = (
            state.pending_external_return
            + state.pending_terminal_discount * transition.reward
        )
        next_mass = state.pending_baseline_mass + state.pending_terminal_discount
        next_discount = state.pending_terminal_discount * transition.discount
        next_elapsed = state.pending_elapsed_primitive_steps + jnp.int32(1)
        future = jnp.asarray(transition.next_observation, dtype=jnp.float32)
        future_matches = (
            jnp.all(state.search.anchor_bank == future[None, :], axis=1)
            & state.search.anchor_active
        )
        exact_future = jnp.sum(future_matches.astype(jnp.int32)) == 1
        natural_option = check_option_terminated(
            self._prototype.oak_agent.stomp_agent.spec_arrays,
            option_index,
            future,
            next_elapsed,
        )
        raw_natural = jnp.where(
            is_option,
            natural_option,
            ~transition.truncated,
        )
        raw_censor = jnp.where(
            is_option,
            (transition.truncated | transition.terminated) & ~natural_option,
            transition.truncated,
        )
        ending = raw_natural | raw_censor
        natural = raw_natural & exact_future
        censored = raw_censor | (raw_natural & ~exact_future)
        resolution_attempted = (
            prototype_applied
            & pending
            & (~state.search_unavailable)
            & ownership
            & ending
        )
        observe = self._controller.observe(
            state.search,
            decision_id=state.search.pending_decision_id,
            future_observation=future,
            observed_future_anchor_mask=jnp.where(
                natural, future_matches, jnp.zeros_like(future_matches)
            ),
            external_return=next_return,
            baseline_mass=next_mass,
            terminal_discount=next_discount,
            elapsed_primitive_steps=next_elapsed,
            natural_completion=natural,
            censored=censored,
            representation_generation=state.search.representation_generation,
            source_digest=state.search.source_digest,
            option_descriptors=state.search.option_descriptors,
            option_generations=state.search.option_generations,
            learner_revision=state.search.learner_revision,
            primitive_model_revision=state.search.pending_primitive_model_revision,
            option_model_revision=state.search.pending_option_model_revision,
        )
        observe_applied = resolution_attempted & observe.diagnostics.transaction_valid
        ownership_failed = (
            prototype_applied
            & pending
            & (~state.search_unavailable)
            & (~ownership)
        )
        observe_failed = resolution_attempted & ~observe_applied
        failed = ownership_failed | observe_failed
        error = jnp.where(
            ownership_failed,
            jnp.int32(PROTOTYPE_STOMP_CALIBRATED_SEARCH_ERROR_OWNERSHIP),
            jnp.where(
                ~observe.diagnostics.capacity_available,
                jnp.int32(PROTOTYPE_STOMP_CALIBRATED_SEARCH_ERROR_CAPACITY),
                jnp.int32(PROTOTYPE_STOMP_CALIBRATED_SEARCH_ERROR_OBSERVE_REJECTED),
            ),
        )
        continuing = (
            prototype_applied
            & pending
            & ownership
            & (~ending)
            & (~state.search_unavailable)
        )
        proposed = cast(
            PrototypeSTOMPCalibratedSearchState,
            state.replace(
                prototype=next_prototype,
                search=cast(
                    CalibratedExtendedSearchControlState,
                    _tree_select(observe_applied, observe.state, state.search),
                ),
                search_unavailable=state.search_unavailable | failed,
                search_error=jnp.where(failed, error, state.search_error).astype(
                    jnp.int32
                ),
                revision=_saturating_increment(state.revision),
                pending_current_decision_id=jnp.where(
                    continuing,
                    next_prototype.current_decision_id,
                    state.pending_current_decision_id,
                ),
                pending_expected_step_words=jnp.where(
                    continuing,
                    next_prototype.step_words,
                    state.pending_expected_step_words,
                ),
                pending_elapsed_primitive_steps=jnp.where(
                    continuing,
                    next_elapsed,
                    state.pending_elapsed_primitive_steps,
                ),
                pending_external_return=jnp.where(
                    continuing, next_return, state.pending_external_return
                ),
                pending_baseline_mass=jnp.where(
                    continuing, next_mass, state.pending_baseline_mass
                ),
                pending_terminal_discount=jnp.where(
                    continuing, next_discount, state.pending_terminal_discount
                ),
                last_prototype_step_words=next_prototype.step_words,
            ),
        )
        cleared = self._clear_pending_fields(proposed)
        proposed = cast(
            PrototypeSTOMPCalibratedSearchState,
            _tree_select(observe_applied, cleared, proposed),
        )
        return (
            self._with_checksum(proposed),
            observe,
            resolution_attempted,
            observe_applied & natural,
            observe_applied & censored,
            ownership,
            failed,
        )

    def update_transition(
        self,
        state: PrototypeSTOMPCalibratedSearchState,
        transition: PrototypeTransition,
        candidate_update_audit_evidence: (
            PrototypeCandidateUpdateAuditEvidence | None
        ) = None,
        *,
        gradient_joy_evidence: PrototypeGradientJoyEvidence | None = None,
        experiential_memory_input: PrototypeExperientialMemoryInput | None = None,
        partner_policy_fusion_input: PrototypePartnerPolicyFusionInput | None = None,
        partner_policy_fusion_feedback: PrototypePartnerPolicyFusionFeedback | None = None,
    ) -> PrototypeSTOMPCalibratedSearchUpdateResult:
        """Apply one real Prototype transition, resolve search, then arm the next."""

        if (
            candidate_update_audit_evidence is not None
            and gradient_joy_evidence is not None
        ):
            raise ValueError(
                "candidate_update_audit_evidence and gradient_joy_evidence "
                "cannot both be supplied"
            )
        selected_audit_evidence = (
            candidate_update_audit_evidence
            if candidate_update_audit_evidence is not None
            else gradient_joy_evidence
        )

        valid_before = self.validate_state(state)
        prototype_result = self._prototype.update_transition(
            state.prototype,
            transition,
            selected_audit_evidence,
            experiential_memory_input=experiential_memory_input,
            partner_policy_fusion_input=partner_policy_fusion_input,
            partner_policy_fusion_feedback=partner_policy_fusion_feedback,
        )
        expected_words, counter_available = _increment_words(state.prototype.step_words)
        prototype_applied = (
            prototype_result.transition_diagnostics.valid
            & counter_available
            & jnp.array_equal(prototype_result.state.step_words, expected_words)
        )
        (
            resolved_state,
            observe,
            resolution_attempted,
            natural,
            censored,
            ownership,
            resolution_failed,
        ) = self._resolve_pending(
            state,
            transition,
            prototype_result.state,
            prototype_applied,
        )
        armed_state, snapshot, arm, arm_diagnostics = self._arm_next(resolved_state)
        candidate_valid = self.validate_state(armed_state)
        committed = valid_before & prototype_applied & candidate_valid
        final = cast(
            PrototypeSTOMPCalibratedSearchState,
            _tree_select(committed, armed_state, state),
        )
        final_action = jnp.where(
            committed,
            prototype_result.action,
            jnp.asarray(-1, dtype=jnp.int32),
        )
        final_prototype_result = cast(
            PrototypeUpdateResult,
            prototype_result.replace(
                state=final.prototype,
                action=final_action,
            ),
        )
        cached_decision = self.decision(final)
        decision = PrototypeDecision(
            observation=cached_decision.observation,
            action=jnp.where(committed, cached_decision.action, jnp.int32(-1)),
            decision_id=cached_decision.decision_id,
            armed=committed & cached_decision.armed,
        )
        sidecar_failed = resolution_failed | arm_diagnostics.sidecar_failed
        diagnostics = PrototypeSTOMPCalibratedSearchUpdateDiagnostics(
            composed_state_valid_before=valid_before,
            prototype_transition_applied=prototype_applied,
            pending_before=state.adapter_pending,
            ownership_binding_matches=ownership,
            resolution_attempted=resolution_attempted,
            natural_resolution=committed & natural,
            censored_resolution=committed & censored,
            observe_applied=(
                jnp.asarray(False, dtype=jnp.bool_)
                if observe is None
                else committed
                & resolution_attempted
                & observe.diagnostics.transaction_valid
            ),
            arm_attempted=committed & arm_diagnostics.arm_attempted,
            arm_applied=committed & arm_diagnostics.arm_applied,
            search_unavailable=final.search_unavailable,
            sidecar_failed_this_step=committed & sidecar_failed,
            prototype_retained_after_sidecar_failure=committed & sidecar_failed,
            composed_state_valid_after=candidate_valid,
            transaction_committed=committed,
            backup_attempt_budget=jnp.asarray(
                self._config.search.backup_budget, dtype=jnp.int32
            ),
            planner_rng_draw_count=jnp.asarray(0, dtype=jnp.int32),
            policy_authority=jnp.asarray(False, dtype=jnp.bool_),
            keyboard_dispatch_applied=jnp.asarray(False, dtype=jnp.bool_),
        )
        return PrototypeSTOMPCalibratedSearchUpdateResult(
            state=final,
            prototype=final_prototype_result,
            decision=decision,
            model_snapshot=snapshot,
            search_observe=observe,
            search_arm=arm,
            diagnostics=diagnostics,
        )

    def scan_transitions(
        self,
        state: PrototypeSTOMPCalibratedSearchState,
        transitions: PrototypeTransition,
    ) -> PrototypeSTOMPCalibratedSearchArrayResult:
        """Run the fixed adapter transaction through ``jax.lax.scan``."""

        def step(
            carry: PrototypeSTOMPCalibratedSearchState,
            transition: PrototypeTransition,
        ) -> tuple[PrototypeSTOMPCalibratedSearchState, tuple[Array, ...]]:
            result = self.update_transition(carry, transition)
            return result.state, (
                result.prototype.action,
                result.diagnostics.prototype_transition_applied,
                result.diagnostics.natural_resolution,
                result.diagnostics.censored_resolution,
                result.diagnostics.search_unavailable,
                result.diagnostics.transaction_committed,
            )

        final, outputs = jax.lax.scan(step, state, transitions)
        return PrototypeSTOMPCalibratedSearchArrayResult(
            state=final,
            actions=outputs[0],
            prototype_transition_applied=outputs[1],
            natural_resolutions=outputs[2],
            censored_resolutions=outputs[3],
            search_unavailable=outputs[4],
            transaction_committed=outputs[5],
        )

    def checkpoint_payload(
        self, state: PrototypeSTOMPCalibratedSearchState
    ) -> dict[str, object]:
        """Return a strict host-only checkpoint, including a pending option.

        Every leaf is materialized to compute an unkeyed SHA-256 corruption
        digest.  This Python boundary is unavailable under JIT/scan; the digest
        is not a MAC, signature, or authenticity claim.
        """

        if not bool(jax.device_get(self.validate_state(state))):
            raise ValueError("cannot checkpoint an invalid composed state")
        return {
            "schema": PROTOTYPE_STOMP_CALIBRATED_SEARCH_CHECKPOINT_SCHEMA,
            "config": self.to_config(),
            "state": state,
            "state_sha256": _tree_sha256(state),
        }

    def restore_checkpoint(
        self,
        payload: object,
        *,
        source_digest: Array,
        representation_generation: int | Array,
    ) -> PrototypeSTOMPCalibratedSearchState:
        """Host-restore only the exact config and source/representation binding."""

        if type(payload) is not dict:
            raise ValueError("adapter checkpoint must be an exact dict")
        raw = cast(dict[object, object], payload)
        if set(raw) != {"schema", "config", "state", "state_sha256"}:
            raise ValueError("adapter checkpoint fields differ from schema v1")
        if raw["schema"] != PROTOTYPE_STOMP_CALIBRATED_SEARCH_CHECKPOINT_SCHEMA:
            raise ValueError("adapter checkpoint schema differs")
        if PrototypeSTOMPCalibratedSearchConfig.from_config(raw["config"]) != self._config:
            raise ValueError("adapter checkpoint config differs")
        restored = raw["state"]
        if type(restored) is not PrototypeSTOMPCalibratedSearchState:
            raise ValueError("adapter checkpoint state type differs")
        digest = _require_array(
            raw["state_sha256"],
            name="checkpoint.state_sha256",
            shape=(_DIGEST_BYTES,),
            dtype=jnp.uint8,
        )
        if not bool(jax.device_get(jnp.array_equal(digest, _tree_sha256(restored)))):
            raise ValueError("adapter checkpoint SHA differs")
        source = _require_array(
            source_digest, name="source_digest", shape=(2,), dtype=jnp.uint32
        )
        generation = _int32_scalar(
            representation_generation, name="representation_generation"
        )
        binding = jnp.array_equal(restored.search.source_digest, source) & (
            restored.search.representation_generation == generation
        )
        if not bool(jax.device_get(binding & self.validate_state(restored))):
            raise ValueError("adapter checkpoint is invalid, stale, or rebound")
        return restored

    def rebind(
        self,
        state: PrototypeSTOMPCalibratedSearchState,
        *,
        prototype_state: PrototypeAgentState,
        source_digest: Array,
        representation_generation: int | Array,
        anchor_bank: Array | None = None,
        anchor_active: Array | None = None,
    ) -> PrototypeSTOMPCalibratedSearchRebindResult:
        """Host-rebind an explicitly supplied live Prototype state.

        Invoke this method on the adapter carrying the new Prototype/STOMP
        configuration.  Option-only semantic changes preserve unchanged slots;
        source, representation, or real-anchor changes conservatively reset the
        complete search sidecar.  The supplied Prototype state is never mutated.
        """

        self._check_state_contract(state)
        checksum_valid = jnp.array_equal(
            state.binding_checksum, _checksum_arrays(self._payload_arrays(state))
        )
        if not bool(jax.device_get(checksum_valid)):
            raise ValueError("cannot rebind a corrupted composed state")
        if not bool(jax.device_get(self._prototype.validate_state(prototype_state))):
            raise ValueError("replacement Prototype state is invalid")
        cfg = self._config.search
        source = _require_array(
            source_digest, name="source_digest", shape=(2,), dtype=jnp.uint32
        )
        generation = _int32_scalar(
            representation_generation, name="representation_generation"
        )
        anchors = (
            state.search.anchor_bank
            if anchor_bank is None
            else _require_array(
                anchor_bank,
                name="anchor_bank",
                shape=(cfg.anchor_capacity, cfg.observation_dim),
                dtype=jnp.float32,
            )
        )
        active = (
            state.search.anchor_active
            if anchor_active is None
            else _require_array(
                anchor_active,
                name="anchor_active",
                shape=(cfg.anchor_capacity,),
                dtype=jnp.bool_,
            )
        )
        old_descriptors = np.asarray(jax.device_get(state.search.option_descriptors))
        new_descriptors = np.asarray(jax.device_get(self._descriptors))
        changed_host = np.any(old_descriptors != new_descriptors, axis=1)
        old_generations = np.asarray(jax.device_get(state.search.option_generations))
        if np.any(changed_host & (old_generations >= _INT32_MAX)):
            raise ValueError("changed option generation would overflow int32")
        new_generations = jnp.asarray(
            old_generations + changed_host.astype(np.int32), dtype=jnp.int32
        )
        global_changed = (
            not bool(jax.device_get(jnp.array_equal(source, state.search.source_digest)))
            or int(jax.device_get(generation))
            != int(jax.device_get(state.search.representation_generation))
            or not bool(jax.device_get(jnp.array_equal(anchors, state.search.anchor_bank)))
            or not bool(jax.device_get(jnp.array_equal(active, state.search.anchor_active)))
        )
        changed = jnp.asarray(changed_host, dtype=jnp.bool_)
        if not global_changed and bool(np.any(changed_host)):
            search = self._controller.replace_option_universe(
                state.search,
                option_descriptors=self._descriptors,
                option_generations=new_generations,
            )
            full_reset = False
            preserved = ~changed
            reset = changed
        else:
            oak = self._oak(prototype_state)
            primitive_revision = (
                self._world(prototype_state).step_count
                if self._config.enabled
                else jnp.int32(0)
            )
            q_values = (
                self._q_values(prototype_state, anchors, active)
                if self._config.enabled
                else jnp.zeros(
                    (cfg.anchor_capacity, cfg.n_extended_actions), dtype=jnp.float32
                )
            )
            search = self._controller.init(
                anchor_bank=anchors,
                anchor_active=active,
                q_values=q_values,
                option_descriptors=self._descriptors,
                option_generations=new_generations,
                representation_generation=generation,
                source_digest=source,
                primitive_model_revision=primitive_revision,
                option_model_revision=oak.stomp_state.step_count,
            )
            full_reset = True
            preserved = jnp.zeros((cfg.n_options,), dtype=jnp.bool_)
            reset = jnp.ones((cfg.n_options,), dtype=jnp.bool_)
        rebound = PrototypeSTOMPCalibratedSearchState(
            prototype=prototype_state,
            search=search,
            enabled=jnp.asarray(self._config.enabled, dtype=jnp.bool_),
            search_unavailable=jnp.asarray(False, dtype=jnp.bool_),
            search_error=jnp.asarray(
                PROTOTYPE_STOMP_CALIBRATED_SEARCH_ERROR_NONE, dtype=jnp.int32
            ),
            revision=_saturating_increment(state.revision),
            adapter_pending=jnp.asarray(False, dtype=jnp.bool_),
            pending_current_decision_id=jnp.zeros((4,), dtype=jnp.uint32),
            pending_expected_step_words=jnp.zeros((2,), dtype=jnp.uint32),
            pending_base_learner_words=jnp.zeros((2,), dtype=jnp.uint32),
            pending_primitive_model_words=jnp.zeros((2,), dtype=jnp.uint32),
            pending_option_completion_counts=jnp.zeros(
                (cfg.n_options,), dtype=jnp.int32
            ),
            pending_primitive_model_checksum=jnp.zeros((2,), dtype=jnp.uint32),
            pending_option_model_checksum=jnp.zeros((2,), dtype=jnp.uint32),
            pending_elapsed_primitive_steps=jnp.asarray(0, dtype=jnp.int32),
            pending_external_return=jnp.asarray(0.0, dtype=jnp.float32),
            pending_baseline_mass=jnp.asarray(0.0, dtype=jnp.float32),
            pending_terminal_discount=jnp.asarray(1.0, dtype=jnp.float32),
            last_prototype_step_words=prototype_state.step_words,
            binding_checksum=jnp.zeros((2,), dtype=jnp.uint32),
        )
        rebound = self._with_checksum(rebound)
        applied = self.validate_state(rebound)
        return PrototypeSTOMPCalibratedSearchRebindResult(
            state=cast(
                PrototypeSTOMPCalibratedSearchState,
                _tree_select(applied, rebound, state),
            ),
            pending_censored=state.adapter_pending & applied,
            full_search_reset=jnp.asarray(full_reset, dtype=jnp.bool_) & applied,
            preserved_option_slots=preserved & applied,
            reset_option_slots=reset & applied,
            transaction_applied=applied,
        )

    def resource_budget(
        self, state: PrototypeSTOMPCalibratedSearchState
    ) -> PrototypeSTOMPCalibratedSearchResourceBudget:
        """Return exact persistent bytes and fixed logical sidecar work bounds."""

        self._check_state_contract(state)
        prototype_bytes = _tree_nbytes(state.prototype)
        search_bytes = self._controller.resource_budget.persistent_state_nbytes
        total_bytes = _tree_nbytes(state)
        binding_bytes = total_bytes - prototype_bytes - search_bytes
        cfg = self._config.search
        return PrototypeSTOMPCalibratedSearchResourceBudget(
            prototype_persistent_state_nbytes=prototype_bytes,
            search_persistent_state_nbytes=search_bytes,
            adapter_binding_nbytes=binding_bytes,
            total_persistent_state_nbytes=total_bytes,
            anchor_capacity=cfg.anchor_capacity,
            primitive_model_predictions_per_decision=(
                cfg.anchor_capacity * cfg.n_primitive_actions
            ),
            option_model_predictions_per_decision=(
                cfg.anchor_capacity * cfg.n_options
            ),
            primitive_model_forward_calls_per_decision=(
                cfg.anchor_capacity * cfg.n_primitive_actions
            ),
            option_outcome_matrix_vector_products_per_decision=(
                cfg.anchor_capacity * cfg.n_options
            ),
            total_secondary_backup_attempts_per_resolution=cfg.backup_budget,
            primitive_and_option_share_one_budget=True,
            prototype_updates_per_transition=1,
            external_model_updates_per_transition=0,
            additional_rng_draws_per_init=0,
            additional_rng_draws_per_start=0,
            additional_rng_draws_per_transition=0,
            planner_rng_draws_total=0,
            max_observations=cfg.max_observations,
            search_exhaustion_can_block_prototype=False,
            q_surface_sidecar_only=True,
            policy_authority=False,
            dispatch_authority=False,
            automatic_keyboard_dispatch=False,
            control_benefit_established=False,
            scientific_promotion_allowed=False,
            checkpoint_schema=PROTOTYPE_STOMP_CALIBRATED_SEARCH_CHECKPOINT_SCHEMA,
            checkpoint_host_only=True,
            rebind_host_only=True,
        )


# Concise aliases for callers that describe the composition as an adapter.
PrototypeSTOMPCalibratedSearchAdapter = PrototypeSTOMPCalibratedSearchAgent
PrototypeSTOMPCalibratedSearchAdapterConfig = PrototypeSTOMPCalibratedSearchConfig


__all__ = [
    "PROTOTYPE_STOMP_CALIBRATED_SEARCH_CHECKPOINT_SCHEMA",
    "PROTOTYPE_STOMP_CALIBRATED_SEARCH_CHECKPOINT_HOST_ONLY",
    "PROTOTYPE_STOMP_CALIBRATED_SEARCH_CONFIG_SCHEMA",
    "PROTOTYPE_STOMP_CALIBRATED_SEARCH_CONTROL_BENEFIT_ESTABLISHED",
    "PROTOTYPE_STOMP_CALIBRATED_SEARCH_DISPATCH_AUTHORITY",
    "PROTOTYPE_STOMP_CALIBRATED_SEARCH_ERROR_ARM_REJECTED",
    "PROTOTYPE_STOMP_CALIBRATED_SEARCH_ERROR_CAPACITY",
    "PROTOTYPE_STOMP_CALIBRATED_SEARCH_ERROR_NONE",
    "PROTOTYPE_STOMP_CALIBRATED_SEARCH_ERROR_OBSERVE_REJECTED",
    "PROTOTYPE_STOMP_CALIBRATED_SEARCH_ERROR_OWNERSHIP",
    "PROTOTYPE_STOMP_CALIBRATED_SEARCH_KEYBOARD_AUTOMATION",
    "PROTOTYPE_STOMP_CALIBRATED_SEARCH_MECHANISM_STATUS",
    "PROTOTYPE_STOMP_CALIBRATED_SEARCH_POLICY_AUTHORITY",
    "PROTOTYPE_STOMP_CALIBRATED_SEARCH_REBIND_HOST_ONLY",
    "PROTOTYPE_STOMP_CALIBRATED_SEARCH_SCIENTIFIC_PROMOTION_ALLOWED",
    "PrototypeSTOMPCalibratedSearchAdapter",
    "PrototypeSTOMPCalibratedSearchAdapterConfig",
    "PrototypeSTOMPCalibratedSearchAgent",
    "PrototypeSTOMPCalibratedSearchArrayResult",
    "PrototypeSTOMPCalibratedSearchConfig",
    "PrototypeSTOMPCalibratedSearchDecisionDiagnostics",
    "PrototypeSTOMPCalibratedSearchRebindResult",
    "PrototypeSTOMPCalibratedSearchResourceBudget",
    "PrototypeSTOMPCalibratedSearchStartResult",
    "PrototypeSTOMPCalibratedSearchState",
    "PrototypeSTOMPCalibratedSearchUpdateDiagnostics",
    "PrototypeSTOMPCalibratedSearchUpdateResult",
    "PrototypeSTOMPModelSnapshot",
]
