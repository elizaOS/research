# mypy: disable-error-code="arg-type,attr-defined,call-arg,no-any-return,type-var"
"""Default-off v2 calibrated-search policy and primitive-dispatch composition.

Version 1 of :mod:`prototype_stomp_calibrated_search` is intentionally an
authority-free sidecar: it settles real primitive/option outcomes and updates
one fixed-budget search Q surface only after Prototype has cached an action.
This module leaves that class and its schemas unchanged.  It adds a separate
v2 composition whose enabled path closes one narrower mechanical edge:

``settle prior arm -> form exact-anchor proposal -> audit hard mask ->
replace cached primitive owner -> arm the effective next action``.

The policy is deterministic and consumes no RNG.  At a unique exact real
anchor it stably selects the lowest-index maximum from the sidecar's current
extended-action Q row.  A primitive selection proposes that primitive
directly.  An option selection forms a one-hot option-keyboard chord and uses
OaK's public deterministic keyboard proposal to obtain a primitive command.
The option is *planned*, not started by this composition.  Prototype's public
cached-action replacement preserves the already-established base or active-
option owner, and that actual owner receives the next real transition's
credit.  The next calibrated arm is formed only after this replacement and is
therefore bound to the primitive/option owner that will really execute.

The hard primitive-action mask is caller-owned.  This module validates and
records it but neither authenticates the caller nor gains safety authority.
An unsafe planned primitive may use OaK's independently safe cached fallback;
an unsafe cached action is withheld as ``action=-1`` and no next arm is
adopted.  Optional policy unavailability never vetoes an otherwise valid
Prototype learning transition.  Persistent composition corruption remains a
whole-wrapper failure requiring checkpoint recovery.

This is a default-off L0 ``not_assessed`` mechanism.  It has no physical
dispatch, autonomous curation/replacement, empirical-benefit, evidence,
promotion, or Alberta Plan completion authority.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping
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
)
from alberta_framework.core.options import (
    DISPATCH_OWNER_BASE_PRIMITIVE,
    DISPATCH_OWNER_INVALID,
    DISPATCH_OWNER_OPTION,
)
from alberta_framework.core.prototype_agent import (
    PrototypeAgentState,
    PrototypeCachedPrimitiveActionReplacement,
    PrototypeCandidateUpdateAuditEvidence,
    PrototypeDecision,
    PrototypeExperientialMemoryInput,
    PrototypeGradientJoyEvidence,
    PrototypePartnerPolicyFusionFeedback,
    PrototypePartnerPolicyFusionInput,
    PrototypeTransition,
    PrototypeUpdateResult,
)
from alberta_framework.core.prototype_stomp_calibrated_search import (
    PrototypeSTOMPCalibratedSearchAgent,
    PrototypeSTOMPCalibratedSearchConfig,
    PrototypeSTOMPCalibratedSearchDecisionDiagnostics,
    PrototypeSTOMPCalibratedSearchStartResult,
    PrototypeSTOMPCalibratedSearchState,
    PrototypeSTOMPCalibratedSearchUpdateDiagnostics,
    PrototypeSTOMPCalibratedSearchUpdateResult,
    PrototypeSTOMPModelSnapshot,
)

PROTOTYPE_STOMP_CALIBRATED_DISPATCH_CONFIG_SCHEMA = (
    "alberta.prototype-stomp-calibrated-dispatch.config.v2"
)
PROTOTYPE_STOMP_CALIBRATED_DISPATCH_STATE_SCHEMA = (
    "alberta.prototype-stomp-calibrated-dispatch.state.v2"
)
PROTOTYPE_STOMP_CALIBRATED_DISPATCH_CHECKPOINT_SCHEMA = (
    "alberta.prototype-stomp-calibrated-dispatch.checkpoint.v2"
)
PROTOTYPE_STOMP_CALIBRATED_DISPATCH_MECHANISM_STATUS = (
    "l0-live-calibrated-policy-dispatch-not-assessed"
)
PROTOTYPE_STOMP_CALIBRATED_DISPATCH_EVIDENCE_LEVEL = "L0"
PROTOTYPE_STOMP_CALIBRATED_DISPATCH_ASSESSMENT = "not_assessed"
PROTOTYPE_STOMP_CALIBRATED_DISPATCH_SCIENTIFIC_PROMOTION_ALLOWED = False
PROTOTYPE_STOMP_CALIBRATED_DISPATCH_SAFETY_AUTHORITY = False
PROTOTYPE_STOMP_CALIBRATED_DISPATCH_PHYSICAL_AUTHORITY = False
PROTOTYPE_STOMP_CALIBRATED_DISPATCH_PLANNED_OPTION_STARTS_OPTION = False
PROTOTYPE_STOMP_CALIBRATED_DISPATCH_CHECKPOINT_HOST_ONLY = True

PROTOTYPE_STOMP_CALIBRATED_DISPATCH_ERROR_NONE = 0
PROTOTYPE_STOMP_CALIBRATED_DISPATCH_ERROR_CLOCK_EXHAUSTED = 1

_INT32_MAX = 2_147_483_647
_DIGEST_BYTES = 32
_CONFIG_FINGERPRINT_WORDS = 8


def _array_contract(value: object, shape: tuple[int, ...], dtype: Any) -> bool:
    return (
        hasattr(value, "shape")
        and hasattr(value, "dtype")
        and tuple(cast(Any, value).shape) == shape
        and cast(Any, value).dtype == jnp.dtype(dtype)
    )


def _require_array(
    value: object,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: Any,
) -> Array:
    if not _array_contract(value, shape, dtype):
        raise TypeError(f"{name} must have exact shape {shape} and dtype {jnp.dtype(dtype)}")
    return jnp.asarray(value)


def _tree_select(predicate: Array, selected: Any, fallback: Any) -> Any:
    return jax.lax.cond(predicate, lambda _: selected, lambda _: fallback, operand=None)


def _increment_words(words: Array) -> tuple[Array, Array]:
    low = words[1] + jnp.uint32(1)
    carry = (low == 0).astype(jnp.uint32)
    high = words[0] + carry
    available = ~((carry != 0) & (high == 0))
    candidate = jnp.stack((high, low), dtype=jnp.uint32)
    return jnp.where(available, candidate, words), available


def _increment_int32(value: Array) -> tuple[Array, Array]:
    available = value < jnp.int32(_INT32_MAX)
    candidate = jnp.where(available, value + jnp.int32(1), value)
    return candidate.astype(jnp.int32), available


def _words_less_equal(left: Array, right: Array) -> Array:
    return (left[0] < right[0]) | ((left[0] == right[0]) & (left[1] <= right[1]))


def _words_nonzero(words: Array) -> Array:
    return jnp.any(words != jnp.uint32(0))


def _checksum_arrays(arrays: tuple[Array, ...]) -> Array:
    """Return a deterministic two-word JIT-compatible integrity checksum."""

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
        acc1 = acc1 ^ jnp.bitwise_xor.reduce(words ^ (indices * jnp.uint32(0x165667B1)))
        offset += words.shape[0]
    return jnp.stack((acc0, acc1), dtype=jnp.uint32)


def _tree_nbytes(tree: object) -> int:
    total = 0
    for leaf in jax.tree_util.tree_leaves(tree):
        array = jnp.asarray(leaf)
        if jax.dtypes.issubdtype(array.dtype, jax.dtypes.prng_key):
            array = jr.key_data(array)
        total += int(array.size) * int(array.dtype.itemsize)
    return total


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


def _canonical_json(value: object) -> object:
    if dataclasses.is_dataclass(value):
        return _canonical_json(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _canonical_json(item) for key, item in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [_canonical_json(item) for item in value]
    return value


def _config_fingerprint(config: Mapping[str, object]) -> Array:
    payload = json.dumps(
        _canonical_json(config),
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    words = tuple(
        int.from_bytes(digest[index : index + 4], byteorder="big", signed=False)
        for index in range(0, len(digest), 4)
    )
    return jnp.asarray(words, dtype=jnp.uint32)


@dataclasses.dataclass(frozen=True, slots=True)
class PrototypeSTOMPCalibratedDispatchConfig:
    """Exact v1 sidecar plus a separately default-off v2 policy switch."""

    sidecar: PrototypeSTOMPCalibratedSearchConfig
    enabled: bool = False

    SCHEMA_VERSION: ClassVar[str] = PROTOTYPE_STOMP_CALIBRATED_DISPATCH_CONFIG_SCHEMA

    def __post_init__(self) -> None:
        if type(self.sidecar) is not PrototypeSTOMPCalibratedSearchConfig:
            raise TypeError("sidecar must be an exact PrototypeSTOMPCalibratedSearchConfig")
        if type(self.enabled) is not bool:
            raise TypeError("enabled must be an exact Python bool")
        if self.enabled and not self.sidecar.enabled:
            raise ValueError("enabled dispatch requires an enabled calibrated-search sidecar")

    def to_config(self) -> dict[str, object]:
        return {
            "schema": self.SCHEMA_VERSION,
            "state_schema": PROTOTYPE_STOMP_CALIBRATED_DISPATCH_STATE_SCHEMA,
            "checkpoint_schema": PROTOTYPE_STOMP_CALIBRATED_DISPATCH_CHECKPOINT_SCHEMA,
            "mechanism_status": PROTOTYPE_STOMP_CALIBRATED_DISPATCH_MECHANISM_STATUS,
            "evidence_level": PROTOTYPE_STOMP_CALIBRATED_DISPATCH_EVIDENCE_LEVEL,
            "assessment": PROTOTYPE_STOMP_CALIBRATED_DISPATCH_ASSESSMENT,
            "enabled": self.enabled,
            "default_enabled": False,
            "policy_rule": (
                "stable_candidate_eligible_extended_q_argmax_then_one_hot_option_keyboard"
            ),
            "policy_history_gate": ("exact_anchor_and_candidate_specific_calibrated_evidence"),
            "proposal_unavailable_fallback": (
                "independently_safe_current_owner_counterfactual_only"
            ),
            "proposal_available_distinct_from_dispatch_authorized": True,
            "composition_order": [
                "settle_prior_arm",
                "form_exact_anchor_proposal",
                "audit_caller_hard_mask",
                "replace_cached_primitive_owner",
                "arm_effective_post_dispatch_owner",
            ],
            "planned_option_starts_option": False,
            "safety_mask_owner": "caller",
            "safety_authority": False,
            "physical_dispatch_authority": False,
            "control_benefit_assessed": False,
            "scientific_promotion_allowed": False,
            "sidecar": self.sidecar.to_config(),
        }

    @classmethod
    def from_config(cls, payload: object) -> PrototypeSTOMPCalibratedDispatchConfig:
        if type(payload) is not dict:
            raise ValueError("dispatch config must be an exact dict")
        raw = cast(dict[object, object], payload)
        fixed = {
            "schema": cls.SCHEMA_VERSION,
            "state_schema": PROTOTYPE_STOMP_CALIBRATED_DISPATCH_STATE_SCHEMA,
            "checkpoint_schema": PROTOTYPE_STOMP_CALIBRATED_DISPATCH_CHECKPOINT_SCHEMA,
            "mechanism_status": PROTOTYPE_STOMP_CALIBRATED_DISPATCH_MECHANISM_STATUS,
            "evidence_level": PROTOTYPE_STOMP_CALIBRATED_DISPATCH_EVIDENCE_LEVEL,
            "assessment": PROTOTYPE_STOMP_CALIBRATED_DISPATCH_ASSESSMENT,
            "default_enabled": False,
            "policy_rule": (
                "stable_candidate_eligible_extended_q_argmax_then_one_hot_option_keyboard"
            ),
            "policy_history_gate": ("exact_anchor_and_candidate_specific_calibrated_evidence"),
            "proposal_unavailable_fallback": (
                "independently_safe_current_owner_counterfactual_only"
            ),
            "proposal_available_distinct_from_dispatch_authorized": True,
            "composition_order": [
                "settle_prior_arm",
                "form_exact_anchor_proposal",
                "audit_caller_hard_mask",
                "replace_cached_primitive_owner",
                "arm_effective_post_dispatch_owner",
            ],
            "planned_option_starts_option": False,
            "safety_mask_owner": "caller",
            "safety_authority": False,
            "physical_dispatch_authority": False,
            "control_benefit_assessed": False,
            "scientific_promotion_allowed": False,
        }
        expected = set(fixed) | {"enabled", "sidecar"}
        if set(raw) != expected:
            raise ValueError("dispatch config fields differ from schema v2")
        for name, value in fixed.items():
            if raw[name] != value:
                raise ValueError(f"dispatch config fixed field {name} differs")
        if type(raw["enabled"]) is not bool:
            raise ValueError("serialized enabled must be an exact bool")
        return cls(
            sidecar=PrototypeSTOMPCalibratedSearchConfig.from_config(raw["sidecar"]),
            enabled=raw["enabled"],
        )


@chex.dataclass(frozen=True)
class PrototypeSTOMPCalibratedDispatchProposal:
    """Content-bound search proposal; a planned option is never execution authority."""

    available: Bool[Array, ""]
    sidecar_state_valid: Bool[Array, ""]
    decision_armed: Bool[Array, ""]
    prior_arm_settled: Bool[Array, ""]
    history_available: Bool[Array, ""]
    policy_boundary: Bool[Array, ""]
    exact_anchor: Bool[Array, ""]
    q_values_valid: Bool[Array, ""]
    safety_mask_static_contract_valid: Bool[Array, ""]
    safety_mask_caller_owned: Bool[Array, ""]
    sidecar_revision: Int[Array, ""]
    search_revision: Int[Array, ""]
    source_digest: UInt[Array, " 2"]
    representation_generation: Int[Array, ""]
    option_universe_digest: UInt[Array, " 2"]
    decision_id: UInt[Array, " 4"]
    decision_observation: Float[Array, " observation_dim"]
    anchor_index: Int[Array, ""]
    q_row: Float[Array, " n_extended_actions"]
    candidate_flat_indices: Int[Array, " n_extended_actions"]
    candidate_target_available: Bool[Array, " n_extended_actions"]
    candidate_value_change_counts: Int[Array, " n_extended_actions"]
    candidate_model_error_counts: Int[Array, " n_extended_actions"]
    candidate_support_counts: Int[Array, " n_extended_actions"]
    candidate_value_change_lcb: Float[Array, " n_extended_actions"]
    candidate_reachability_lcb: Float[Array, " n_extended_actions"]
    candidate_model_reliability: Float[Array, " n_extended_actions"]
    candidate_support_shrinkage: Float[Array, " n_extended_actions"]
    candidate_eligible: Bool[Array, " n_extended_actions"]
    any_candidate_eligible: Bool[Array, ""]
    planned_extended_action: Int[Array, ""]
    planned_kind: Int[Array, ""]
    planned_semantic_index: Int[Array, ""]
    planned_option_index: Int[Array, ""]
    keyboard_used: Bool[Array, ""]
    keyboard_vector: Float[Array, " n_options"]
    keyboard_proposal_available: Bool[Array, ""]
    keyboard_q_values: Float[Array, " n_primitive_actions"]
    proposed_primitive_action: Int[Array, ""]
    safety_action_mask: Bool[Array, " n_primitive_actions"]
    content_tag: UInt[Array, " 2"]


@chex.dataclass(frozen=True)
class PrototypeSTOMPCalibratedDispatchState:
    """One exact v1 sidecar plus v2 policy provenance and word-pair clocks."""

    sidecar: PrototypeSTOMPCalibratedSearchState
    policy_enabled: Bool[Array, ""]
    config_fingerprint: UInt[Array, " 8"]
    policy_call_count_words: UInt[Array, " 2"]
    dispatch_commit_count_words: UInt[Array, " 2"]
    last_record_valid: Bool[Array, ""]
    last_decision_id: UInt[Array, " 4"]
    last_proposal_content_tag: UInt[Array, " 2"]
    last_safety_mask_digest: UInt[Array, " 2"]
    last_planned_extended_action: Int[Array, ""]
    last_planned_option_index: Int[Array, ""]
    last_effective_primitive_action: Int[Array, ""]
    last_actual_credit_owner: Int[Array, ""]
    last_actual_executing_option: Int[Array, ""]
    last_policy_available: Bool[Array, ""]
    last_keyboard_used: Bool[Array, ""]
    last_dispatch_committed: Bool[Array, ""]
    policy_unavailable: Bool[Array, ""]
    policy_error: Int[Array, ""]
    binding_checksum: UInt[Array, " 2"]


@chex.dataclass(frozen=True)
class PrototypeSTOMPCalibratedDispatchDiagnostics:
    """Exact timing, ownership, safety, and authority facts for one decision."""

    state_valid_before: Bool[Array, ""]
    state_valid_after: Bool[Array, ""]
    prototype_learning_applied: Bool[Array, ""]
    prior_arm_pending_before: Bool[Array, ""]
    prior_arm_resolution_attempted: Bool[Array, ""]
    prior_arm_settled_before_policy: Bool[Array, ""]
    policy_enabled: Bool[Array, ""]
    policy_call_attempted: Bool[Array, ""]
    proposal_available: Bool[Array, ""]
    proposal_content_valid: Bool[Array, ""]
    exact_anchor: Bool[Array, ""]
    candidate_eligible: Bool[Array, " n_extended_actions"]
    selected_candidate_flat_index: Int[Array, ""]
    selected_candidate_target_available: Bool[Array, ""]
    selected_candidate_value_change_count: Int[Array, ""]
    selected_candidate_model_error_count: Int[Array, ""]
    selected_candidate_support_count: Int[Array, ""]
    selected_candidate_value_change_lcb: Float[Array, ""]
    selected_candidate_reachability_lcb: Float[Array, ""]
    selected_candidate_model_reliability: Float[Array, ""]
    selected_candidate_support_shrinkage: Float[Array, ""]
    planned_extended_action: Int[Array, ""]
    planned_kind: Int[Array, ""]
    planned_semantic_index: Int[Array, ""]
    planned_option_index: Int[Array, ""]
    keyboard_used: Bool[Array, ""]
    keyboard_proposal_available: Bool[Array, ""]
    hard_mask_static_contract_valid: Bool[Array, ""]
    hard_mask_caller_owned: Bool[Array, ""]
    replacement_attempted: Bool[Array, ""]
    replacement_committed: Bool[Array, ""]
    used_safe_current_owner_fallback: Bool[Array, ""]
    dispatch_authorized: Bool[Array, ""]
    action_changed: Bool[Array, ""]
    effective_primitive_action: Int[Array, ""]
    actual_credit_owner: Int[Array, ""]
    actual_executing_option: Int[Array, ""]
    planned_option_started_by_dispatch: Bool[Array, ""]
    planned_option_matches_existing_owner: Bool[Array, ""]
    next_arm_attempted: Bool[Array, ""]
    next_arm_applied: Bool[Array, ""]
    next_arm_bound_after_dispatch: Bool[Array, ""]
    one_total_backup_budget_preserved: Bool[Array, ""]
    additional_rng_draw_count: Int[Array, ""]
    additional_backward_pass_count: Int[Array, ""]
    additional_model_update_count: Int[Array, ""]
    safety_authority: Bool[Array, ""]
    physical_dispatch_authority: Bool[Array, ""]
    scientific_promotion_allowed: Bool[Array, ""]
    transaction_committed: Bool[Array, ""]
    pre_policy_call_count_words: UInt[Array, " 2"]
    post_policy_call_count_words: UInt[Array, " 2"]


@chex.dataclass(frozen=True)
class PrototypeSTOMPCalibratedDispatchStartResult:
    state: PrototypeSTOMPCalibratedDispatchState
    decision: PrototypeDecision
    sidecar_result: PrototypeSTOMPCalibratedSearchStartResult
    proposal: PrototypeSTOMPCalibratedDispatchProposal
    replacement: PrototypeCachedPrimitiveActionReplacement | None
    diagnostics: PrototypeSTOMPCalibratedDispatchDiagnostics


@chex.dataclass(frozen=True)
class PrototypeSTOMPCalibratedDispatchUpdateResult:
    state: PrototypeSTOMPCalibratedDispatchState
    prototype: PrototypeUpdateResult
    decision: PrototypeDecision
    sidecar_result: PrototypeSTOMPCalibratedSearchUpdateResult
    proposal: PrototypeSTOMPCalibratedDispatchProposal
    replacement: PrototypeCachedPrimitiveActionReplacement | None
    diagnostics: PrototypeSTOMPCalibratedDispatchDiagnostics


@chex.dataclass(frozen=True)
class PrototypeSTOMPCalibratedDispatchRetryResult:
    """No-learning retry of one current, previously unauthorized decision."""

    state: PrototypeSTOMPCalibratedDispatchState
    decision: PrototypeDecision
    proposal: PrototypeSTOMPCalibratedDispatchProposal
    replacement: PrototypeCachedPrimitiveActionReplacement
    model_snapshot: PrototypeSTOMPModelSnapshot
    search_arm: CalibratedExtendedSearchArmResult
    diagnostics: PrototypeSTOMPCalibratedDispatchDiagnostics


@chex.dataclass(frozen=True)
class PrototypeSTOMPCalibratedDispatchArrayResult:
    state: PrototypeSTOMPCalibratedDispatchState
    actions: Int[Array, " steps"]
    prototype_learning_applied: Bool[Array, " steps"]
    proposal_available: Bool[Array, " steps"]
    dispatch_authorized: Bool[Array, " steps"]
    action_changed: Bool[Array, " steps"]
    next_arm_applied: Bool[Array, " steps"]
    transaction_committed: Bool[Array, " steps"]


@dataclasses.dataclass(frozen=True, slots=True)
class PrototypeSTOMPCalibratedDispatchResourceBudget:
    """Exact persistent bytes and bounded incremental v2 decision work."""

    sidecar_persistent_state_nbytes: int
    dispatch_binding_nbytes: int
    total_persistent_state_nbytes: int
    anchor_capacity: int
    n_extended_actions: int
    n_primitive_actions: int
    n_options: int
    total_secondary_backup_attempts_per_resolution: int
    primitive_and_option_share_one_budget: bool
    max_q_values_interpreted_per_policy_call: int
    max_argmax_comparisons_per_policy_call: int
    max_keyboard_proposals_per_policy_call: int
    max_cached_action_replacements_per_policy_call: int
    max_next_arm_calls_per_decision: int
    additional_rng_draws_per_policy_call: int
    additional_backward_passes_per_policy_call: int
    additional_model_updates_per_policy_call: int
    persistent_state_growth_per_transition_bytes: int
    planned_option_starts_option: bool
    proposal_unavailability_can_dispatch_safe_current_owner: bool
    proposal_available_distinct_from_dispatch_authorized: bool
    hard_safety_masks_per_policy_call: int
    safety_authority: bool
    physical_dispatch_authority: bool
    control_benefit_assessed: bool
    scientific_promotion_allowed: bool
    enabled_by_default: bool
    checkpoint_schema: str

    def to_config(self) -> dict[str, object]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True, slots=True)
class _PolicyApplication:
    state: PrototypeSTOMPCalibratedSearchState
    proposal: PrototypeSTOMPCalibratedDispatchProposal
    replacement: PrototypeCachedPrimitiveActionReplacement
    mask_static_valid: Array
    call_attempted: Array
    call_clock_available: Array
    commit_clock_available: Array
    dispatch_would_authorize: Array
    replacement_committed: Array
    dispatch_authorized: Array
    action_changed: Array
    effective_action: Array
    actual_credit_owner: Array
    actual_executing_option: Array
    planned_option_matches_existing_owner: Array


class PrototypeSTOMPCalibratedDispatchAgent:
    """Separate v2 policy consumer around an unchanged v1 search sidecar."""

    def __init__(self, config: PrototypeSTOMPCalibratedDispatchConfig) -> None:
        if type(config) is not PrototypeSTOMPCalibratedDispatchConfig:
            raise TypeError("config must be an exact PrototypeSTOMPCalibratedDispatchConfig")
        self._config = config
        self._sidecar = PrototypeSTOMPCalibratedSearchAgent(config.sidecar)
        self._config_fingerprint = _config_fingerprint(config.to_config())

    @property
    def config(self) -> PrototypeSTOMPCalibratedDispatchConfig:
        return self._config

    @property
    def sidecar(self) -> PrototypeSTOMPCalibratedSearchAgent:
        return self._sidecar

    @property
    def prototype(self) -> Any:
        return self._sidecar.prototype

    def to_config(self) -> dict[str, object]:
        return self._config.to_config()

    @classmethod
    def from_config(cls, payload: object) -> PrototypeSTOMPCalibratedDispatchAgent:
        return cls(PrototypeSTOMPCalibratedDispatchConfig.from_config(payload))

    def _payload_arrays(self, state: PrototypeSTOMPCalibratedDispatchState) -> tuple[Array, ...]:
        sidecar = tuple(cast(Array, leaf) for leaf in jax.tree_util.tree_leaves(state.sidecar))
        return (
            *sidecar,
            state.policy_enabled,
            state.config_fingerprint,
            state.policy_call_count_words,
            state.dispatch_commit_count_words,
            state.last_record_valid,
            state.last_decision_id,
            state.last_proposal_content_tag,
            state.last_safety_mask_digest,
            state.last_planned_extended_action,
            state.last_planned_option_index,
            state.last_effective_primitive_action,
            state.last_actual_credit_owner,
            state.last_actual_executing_option,
            state.last_policy_available,
            state.last_keyboard_used,
            state.last_dispatch_committed,
            state.policy_unavailable,
            state.policy_error,
        )

    def _with_checksum(
        self, state: PrototypeSTOMPCalibratedDispatchState
    ) -> PrototypeSTOMPCalibratedDispatchState:
        return cast(
            PrototypeSTOMPCalibratedDispatchState,
            state.replace(binding_checksum=_checksum_arrays(self._payload_arrays(state))),
        )

    def _blank_state(
        self, sidecar: PrototypeSTOMPCalibratedSearchState
    ) -> PrototypeSTOMPCalibratedDispatchState:
        state = PrototypeSTOMPCalibratedDispatchState(
            sidecar=sidecar,
            policy_enabled=jnp.asarray(self._config.enabled, dtype=jnp.bool_),
            config_fingerprint=self._config_fingerprint,
            policy_call_count_words=jnp.zeros((2,), dtype=jnp.uint32),
            dispatch_commit_count_words=jnp.zeros((2,), dtype=jnp.uint32),
            last_record_valid=jnp.asarray(False, dtype=jnp.bool_),
            last_decision_id=jnp.zeros((4,), dtype=jnp.uint32),
            last_proposal_content_tag=jnp.zeros((2,), dtype=jnp.uint32),
            last_safety_mask_digest=jnp.zeros((2,), dtype=jnp.uint32),
            last_planned_extended_action=jnp.asarray(-1, dtype=jnp.int32),
            last_planned_option_index=jnp.asarray(-1, dtype=jnp.int32),
            last_effective_primitive_action=jnp.asarray(-1, dtype=jnp.int32),
            last_actual_credit_owner=jnp.asarray(DISPATCH_OWNER_INVALID, dtype=jnp.int32),
            last_actual_executing_option=jnp.asarray(-1, dtype=jnp.int32),
            last_policy_available=jnp.asarray(False, dtype=jnp.bool_),
            last_keyboard_used=jnp.asarray(False, dtype=jnp.bool_),
            last_dispatch_committed=jnp.asarray(False, dtype=jnp.bool_),
            policy_unavailable=jnp.asarray(False, dtype=jnp.bool_),
            policy_error=jnp.asarray(
                PROTOTYPE_STOMP_CALIBRATED_DISPATCH_ERROR_NONE, dtype=jnp.int32
            ),
            binding_checksum=jnp.zeros((2,), dtype=jnp.uint32),
        )
        return self._with_checksum(state)

    def _check_state_contract(self, state: object) -> None:
        if type(state) is not PrototypeSTOMPCalibratedDispatchState:
            raise TypeError("state must be an exact PrototypeSTOMPCalibratedDispatchState")
        value = state
        contracts = (
            (value.policy_enabled, "policy_enabled", (), jnp.bool_),
            (value.config_fingerprint, "config_fingerprint", (8,), jnp.uint32),
            (value.policy_call_count_words, "policy_call_count_words", (2,), jnp.uint32),
            (
                value.dispatch_commit_count_words,
                "dispatch_commit_count_words",
                (2,),
                jnp.uint32,
            ),
            (value.last_record_valid, "last_record_valid", (), jnp.bool_),
            (value.last_decision_id, "last_decision_id", (4,), jnp.uint32),
            (
                value.last_proposal_content_tag,
                "last_proposal_content_tag",
                (2,),
                jnp.uint32,
            ),
            (
                value.last_safety_mask_digest,
                "last_safety_mask_digest",
                (2,),
                jnp.uint32,
            ),
            (
                value.last_planned_extended_action,
                "last_planned_extended_action",
                (),
                jnp.int32,
            ),
            (value.last_planned_option_index, "last_planned_option_index", (), jnp.int32),
            (
                value.last_effective_primitive_action,
                "last_effective_primitive_action",
                (),
                jnp.int32,
            ),
            (value.last_actual_credit_owner, "last_actual_credit_owner", (), jnp.int32),
            (
                value.last_actual_executing_option,
                "last_actual_executing_option",
                (),
                jnp.int32,
            ),
            (value.last_policy_available, "last_policy_available", (), jnp.bool_),
            (value.last_keyboard_used, "last_keyboard_used", (), jnp.bool_),
            (value.last_dispatch_committed, "last_dispatch_committed", (), jnp.bool_),
            (value.policy_unavailable, "policy_unavailable", (), jnp.bool_),
            (value.policy_error, "policy_error", (), jnp.int32),
            (value.binding_checksum, "binding_checksum", (2,), jnp.uint32),
        )
        for item, name, shape, dtype in contracts:
            _require_array(item, name=f"state.{name}", shape=shape, dtype=dtype)

    def validate_state(self, state: PrototypeSTOMPCalibratedDispatchState) -> Bool[Array, ""]:
        self._check_state_contract(state)
        cfg = self._config.sidecar.search
        any_call = _words_nonzero(state.policy_call_count_words)
        blank_record = (
            (~state.last_record_valid)
            & jnp.all(state.last_decision_id == 0)
            & jnp.all(state.last_proposal_content_tag == 0)
            & jnp.all(state.last_safety_mask_digest == 0)
            & (state.last_planned_extended_action == -1)
            & (state.last_planned_option_index == -1)
            & (state.last_effective_primitive_action == -1)
            & (state.last_actual_credit_owner == DISPATCH_OWNER_INVALID)
            & (state.last_actual_executing_option == -1)
            & (~state.last_policy_available)
            & (~state.last_keyboard_used)
            & (~state.last_dispatch_committed)
        )
        planned_valid = (state.last_planned_extended_action >= -1) & (
            state.last_planned_extended_action < cfg.n_extended_actions
        )
        planned_option_valid = (state.last_planned_option_index >= -1) & (
            state.last_planned_option_index < cfg.n_options
        )
        effective_valid = (state.last_effective_primitive_action >= -1) & (
            state.last_effective_primitive_action < cfg.n_primitive_actions
        )
        owner_valid = (
            (state.last_actual_credit_owner == DISPATCH_OWNER_INVALID)
            | (state.last_actual_credit_owner == DISPATCH_OWNER_BASE_PRIMITIVE)
            | (state.last_actual_credit_owner == DISPATCH_OWNER_OPTION)
        )
        executing_valid = (state.last_actual_executing_option >= -1) & (
            state.last_actual_executing_option < cfg.n_options
        )
        committed_record_valid = jnp.where(
            state.last_dispatch_committed,
            (state.last_effective_primitive_action >= 0)
            & (state.last_actual_credit_owner != DISPATCH_OWNER_INVALID),
            True,
        )
        option_record_valid = jnp.where(
            state.last_planned_option_index >= 0,
            state.last_keyboard_used
            & (state.last_planned_extended_action >= cfg.n_primitive_actions)
            & (
                state.last_planned_option_index
                == state.last_planned_extended_action - cfg.n_primitive_actions
            ),
            ~state.last_keyboard_used,
        )
        stomp = self._sidecar._oak(state.sidecar.prototype).stomp_state
        current_record = state.last_record_valid & jnp.array_equal(
            state.last_decision_id, state.sidecar.prototype.current_decision_id
        )
        current_owner_binding = jnp.where(
            stomp.executing_option >= 0,
            (state.last_actual_credit_owner == DISPATCH_OWNER_OPTION)
            & (state.last_actual_executing_option == stomp.executing_option),
            (state.last_actual_credit_owner == DISPATCH_OWNER_BASE_PRIMITIVE)
            & (state.last_actual_executing_option == -1),
        )
        current_record_valid = jnp.where(
            current_record,
            jnp.where(
                state.last_dispatch_committed,
                state.sidecar.prototype.started
                & (state.last_effective_primitive_action == state.sidecar.prototype.current_action)
                & current_owner_binding,
                (state.last_effective_primitive_action == -1)
                & (state.last_actual_credit_owner == DISPATCH_OWNER_INVALID)
                & (state.last_actual_executing_option == -1),
            ),
            True,
        )
        error_valid = (
            (state.policy_error >= PROTOTYPE_STOMP_CALIBRATED_DISPATCH_ERROR_NONE)
            & (state.policy_error <= PROTOTYPE_STOMP_CALIBRATED_DISPATCH_ERROR_CLOCK_EXHAUSTED)
            & (
                state.policy_unavailable
                == (state.policy_error != PROTOTYPE_STOMP_CALIBRATED_DISPATCH_ERROR_NONE)
            )
        )
        return (
            self._sidecar.validate_state(state.sidecar)
            & (state.policy_enabled == self._config.enabled)
            & jnp.array_equal(state.config_fingerprint, self._config_fingerprint)
            & _words_less_equal(state.dispatch_commit_count_words, state.policy_call_count_words)
            & jnp.where(any_call, state.last_record_valid, blank_record)
            & planned_valid
            & planned_option_valid
            & effective_valid
            & owner_valid
            & executing_valid
            & committed_record_valid
            & option_record_valid
            & current_record_valid
            & error_valid
            & jnp.array_equal(state.binding_checksum, _checksum_arrays(self._payload_arrays(state)))
        )

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
    ) -> PrototypeSTOMPCalibratedDispatchState:
        sidecar = self._sidecar.init(
            key,
            anchor_bank=anchor_bank,
            anchor_active=anchor_active,
            source_digest=source_digest,
            representation_generation=representation_generation,
            option_generations=option_generations,
            lifecycle_id=lifecycle_id,
        )
        return self._blank_state(sidecar)

    def _proposal_arrays(
        self, proposal: PrototypeSTOMPCalibratedDispatchProposal
    ) -> tuple[Array, ...]:
        return (
            proposal.available,
            proposal.sidecar_state_valid,
            proposal.decision_armed,
            proposal.prior_arm_settled,
            proposal.history_available,
            proposal.policy_boundary,
            proposal.exact_anchor,
            proposal.q_values_valid,
            proposal.safety_mask_static_contract_valid,
            proposal.safety_mask_caller_owned,
            proposal.sidecar_revision,
            proposal.search_revision,
            proposal.source_digest,
            proposal.representation_generation,
            proposal.option_universe_digest,
            proposal.decision_id,
            proposal.decision_observation,
            proposal.anchor_index,
            proposal.q_row,
            proposal.candidate_flat_indices,
            proposal.candidate_target_available,
            proposal.candidate_value_change_counts,
            proposal.candidate_model_error_counts,
            proposal.candidate_support_counts,
            proposal.candidate_value_change_lcb,
            proposal.candidate_reachability_lcb,
            proposal.candidate_model_reliability,
            proposal.candidate_support_shrinkage,
            proposal.candidate_eligible,
            proposal.any_candidate_eligible,
            proposal.planned_extended_action,
            proposal.planned_kind,
            proposal.planned_semantic_index,
            proposal.planned_option_index,
            proposal.keyboard_used,
            proposal.keyboard_vector,
            proposal.keyboard_proposal_available,
            proposal.keyboard_q_values,
            proposal.proposed_primitive_action,
            proposal.safety_action_mask,
        )

    def proposal_content_tag(
        self, proposal: PrototypeSTOMPCalibratedDispatchProposal
    ) -> UInt[Array, " 2"]:
        return _checksum_arrays(self._proposal_arrays(proposal))

    def _safe_mask(self, value: object) -> tuple[Array, Array]:
        cfg = self._config.sidecar.search
        valid = _array_contract(value, (cfg.n_primitive_actions,), jnp.bool_)
        mask = (
            jnp.asarray(value) if valid else jnp.zeros((cfg.n_primitive_actions,), dtype=jnp.bool_)
        )
        return mask, jnp.asarray(valid, dtype=jnp.bool_)

    def _form_proposal(
        self,
        wrapper_state: PrototypeSTOMPCalibratedDispatchState,
        sidecar_state: PrototypeSTOMPCalibratedSearchState,
        safety_action_mask: object,
    ) -> PrototypeSTOMPCalibratedDispatchProposal:
        cfg = self._config.sidecar.search
        mask, mask_static_valid = self._safe_mask(safety_action_mask)
        decision = self._sidecar.prototype.decision(sidecar_state.prototype)
        oak = self._sidecar._oak(sidecar_state.prototype)
        stomp = oak.stomp_state
        observation = sidecar_state.prototype.current_representation
        bank_bits = jax.lax.bitcast_convert_type(sidecar_state.search.anchor_bank, jnp.uint32)
        observation_bits = jax.lax.bitcast_convert_type(observation, jnp.uint32)
        matches = (
            jnp.all(bank_bits == observation_bits[None, :], axis=1)
            & sidecar_state.search.anchor_active
        )
        exact_anchor = jnp.sum(matches.astype(jnp.int32)) == 1
        anchor_index = jnp.argmax(matches).astype(jnp.int32)
        raw_q = sidecar_state.search.q_values[anchor_index]
        q_values_valid = jnp.all(jnp.isfinite(raw_q))
        q_row = jnp.where(q_values_valid, raw_q, jnp.zeros_like(raw_q))
        extended_indices = jnp.arange(cfg.n_extended_actions, dtype=jnp.int32)
        candidate_flat_indices = extended_indices * jnp.int32(cfg.anchor_capacity) + anchor_index
        candidate_target_available = sidecar_state.search.last_target_available[
            candidate_flat_indices
        ]
        candidate_value_counts = sidecar_state.search.value_change_counts[candidate_flat_indices]
        candidate_error_counts = sidecar_state.search.model_error_counts[candidate_flat_indices]
        candidate_support_counts = sidecar_state.search.support_counts[candidate_flat_indices]
        value_lcb, reach_lcb, error_ucb = self._sidecar.controller._factor_estimates(
            sidecar_state.search
        )
        candidate_value_lcb = value_lcb[candidate_flat_indices]
        candidate_reach_lcb = reach_lcb[candidate_flat_indices]
        candidate_reliability = 1.0 - error_ucb[candidate_flat_indices]
        support = candidate_support_counts.astype(jnp.float32)
        candidate_support_shrinkage = support / (
            support + jnp.asarray(cfg.support_prior, dtype=jnp.float32)
        )
        candidate_evidence_ready = (
            candidate_target_available
            & (candidate_value_counts >= cfg.calibration_evidence_floor)
            & (candidate_error_counts >= cfg.calibration_evidence_floor)
            & (candidate_support_counts >= cfg.model_support_floor)
            & (
                sidecar_state.search.anchor_revisit_trials[anchor_index]
                >= cfg.calibration_evidence_floor
            )
        )
        candidate_eligible = (
            candidate_evidence_ready
            & (candidate_value_lcb > cfg.min_value_change_lcb)
            & (candidate_reach_lcb > cfg.min_reachability_lcb)
            & (candidate_reliability > cfg.min_model_reliability)
            & (candidate_support_shrinkage > 0.0)
            & exact_anchor
        )
        any_candidate_eligible = jnp.any(candidate_eligible)
        eligible_q = jnp.where(candidate_eligible, q_row, -jnp.inf)
        raw_planned = jnp.argmax(eligible_q).astype(jnp.int32)
        planned_is_option = raw_planned >= cfg.n_primitive_actions
        raw_option_index = raw_planned - jnp.int32(cfg.n_primitive_actions)
        primitive_boundary = (stomp.executing_option < 0) & (
            stomp.base_last_action < cfg.n_primitive_actions
        )
        option_boundary = (stomp.executing_option >= 0) & (stomp.option_steps == 0)
        prior_arm_settled = ~sidecar_state.search.pending
        policy_boundary = prior_arm_settled & (primitive_boundary | option_boundary)
        history_available = sidecar_state.search.has_last_decision
        sidecar_valid = self._sidecar.validate_state(sidecar_state)
        core_available = (
            jnp.asarray(self._config.enabled, dtype=jnp.bool_)
            & (~wrapper_state.policy_unavailable)
            & sidecar_valid
            & (~sidecar_state.search_unavailable)
            & decision.armed
            & prior_arm_settled
            & history_available
            & policy_boundary
            & exact_anchor
            & q_values_valid
            & any_candidate_eligible
            & mask_static_valid
        )
        chord = jax.nn.one_hot(
            jnp.clip(raw_option_index, 0, cfg.n_options - 1),
            cfg.n_options,
            dtype=jnp.float32,
        )
        chord = jnp.where(
            core_available & planned_is_option,
            chord,
            jnp.zeros_like(chord),
        )
        keyboard = self._sidecar.prototype.oak_agent.propose_keyboard_policy(
            oak,
            observation,
            chord,
        )
        available = core_available & jnp.where(planned_is_option, keyboard.available, True)
        reported_option_index = jnp.where(
            available & planned_is_option,
            raw_option_index,
            jnp.int32(-1),
        )
        proposed_primitive = jnp.where(
            planned_is_option,
            keyboard.action,
            raw_planned,
        ).astype(jnp.int32)
        planned_extended = jnp.where(available, raw_planned, jnp.int32(-1))
        planned_kind = jnp.where(
            available,
            jnp.where(
                planned_is_option,
                jnp.int32(CANDIDATE_KIND_OPTION),
                jnp.int32(CANDIDATE_KIND_PRIMITIVE),
            ),
            jnp.int32(-1),
        )
        planned_semantic = jnp.where(
            available,
            jnp.where(planned_is_option, raw_option_index, raw_planned),
            jnp.int32(-1),
        )
        provisional = PrototypeSTOMPCalibratedDispatchProposal(
            available=available,
            sidecar_state_valid=sidecar_valid,
            decision_armed=decision.armed,
            prior_arm_settled=prior_arm_settled,
            history_available=history_available,
            policy_boundary=policy_boundary,
            exact_anchor=exact_anchor,
            q_values_valid=q_values_valid,
            safety_mask_static_contract_valid=mask_static_valid,
            safety_mask_caller_owned=jnp.asarray(True, dtype=jnp.bool_),
            sidecar_revision=sidecar_state.revision,
            search_revision=sidecar_state.search.state_revision,
            source_digest=sidecar_state.search.source_digest,
            representation_generation=sidecar_state.search.representation_generation,
            option_universe_digest=sidecar_state.search.option_universe_digest,
            decision_id=decision.decision_id,
            decision_observation=observation,
            anchor_index=jnp.where(exact_anchor, anchor_index, jnp.int32(-1)),
            q_row=q_row,
            candidate_flat_indices=candidate_flat_indices,
            candidate_target_available=candidate_target_available,
            candidate_value_change_counts=candidate_value_counts,
            candidate_model_error_counts=candidate_error_counts,
            candidate_support_counts=candidate_support_counts,
            candidate_value_change_lcb=candidate_value_lcb,
            candidate_reachability_lcb=candidate_reach_lcb,
            candidate_model_reliability=candidate_reliability,
            candidate_support_shrinkage=candidate_support_shrinkage,
            candidate_eligible=candidate_eligible,
            any_candidate_eligible=any_candidate_eligible,
            planned_extended_action=planned_extended,
            planned_kind=planned_kind,
            planned_semantic_index=planned_semantic,
            planned_option_index=reported_option_index,
            keyboard_used=available & planned_is_option,
            keyboard_vector=chord,
            keyboard_proposal_available=keyboard.available,
            keyboard_q_values=keyboard.q_values,
            proposed_primitive_action=jnp.where(available, proposed_primitive, jnp.int32(-1)),
            safety_action_mask=mask,
            content_tag=jnp.zeros((2,), dtype=jnp.uint32),
        )
        return cast(
            PrototypeSTOMPCalibratedDispatchProposal,
            provisional.replace(content_tag=self.proposal_content_tag(provisional)),
        )

    def _apply_policy(
        self,
        wrapper_state: PrototypeSTOMPCalibratedDispatchState,
        sidecar_state: PrototypeSTOMPCalibratedSearchState,
        safety_action_mask: object,
    ) -> _PolicyApplication:
        proposal = self._form_proposal(wrapper_state, sidecar_state, safety_action_mask)
        decision = self._sidecar.prototype.decision(sidecar_state.prototype)
        candidate_action = jnp.where(
            proposal.available,
            proposal.proposed_primitive_action,
            decision.action,
        ).astype(jnp.int32)
        replacement = self._sidecar.prototype.replace_cached_primitive_action(
            sidecar_state.prototype,
            decision_id=decision.decision_id,
            decision_observation=sidecar_state.prototype.current_representation,
            proposed_action=candidate_action,
            safety_action_mask=proposal.safety_action_mask,
        )
        call_attempted = jnp.asarray(self._config.enabled, dtype=jnp.bool_) & decision.armed
        _, call_clock_available = _increment_words(wrapper_state.policy_call_count_words)
        _, commit_clock_available = _increment_words(wrapper_state.dispatch_commit_count_words)
        dispatch_would_authorize = (
            call_attempted
            & replacement.committed
            & proposal.safety_mask_static_contract_valid
            & replacement.dispatch_replacement.counterfactual_action_safe
        )
        dispatch_authorized = (
            dispatch_would_authorize & call_clock_available & commit_clock_available
        )
        replacement_committed = dispatch_authorized
        prototype = cast(
            PrototypeAgentState,
            _tree_select(
                dispatch_authorized,
                replacement.state,
                sidecar_state.prototype,
            ),
        )
        candidate = cast(
            PrototypeSTOMPCalibratedSearchState,
            sidecar_state.replace(prototype=prototype),
        )
        candidate = self._sidecar._with_checksum(candidate)
        oak = self._sidecar._oak(candidate.prototype)
        actual_executing = oak.stomp_state.executing_option
        planned_matches = (proposal.planned_option_index >= 0) & (
            actual_executing == proposal.planned_option_index
        )
        return _PolicyApplication(
            state=candidate,
            proposal=proposal,
            replacement=replacement,
            mask_static_valid=proposal.safety_mask_static_contract_valid,
            call_attempted=call_attempted,
            call_clock_available=call_clock_available,
            commit_clock_available=commit_clock_available,
            dispatch_would_authorize=dispatch_would_authorize,
            replacement_committed=replacement_committed,
            dispatch_authorized=dispatch_authorized,
            action_changed=(dispatch_authorized & (replacement.action != decision.action)),
            effective_action=jnp.where(dispatch_authorized, replacement.action, jnp.int32(-1)),
            actual_credit_owner=jnp.where(
                dispatch_authorized,
                replacement.dispatch_replacement.owner,
                jnp.int32(DISPATCH_OWNER_INVALID),
            ),
            actual_executing_option=jnp.where(dispatch_authorized, actual_executing, jnp.int32(-1)),
            planned_option_matches_existing_owner=planned_matches,
        )

    def _record_policy(
        self,
        state: PrototypeSTOMPCalibratedDispatchState,
        sidecar_state: PrototypeSTOMPCalibratedSearchState,
        application: _PolicyApplication,
        transaction_committed: Array,
    ) -> PrototypeSTOMPCalibratedDispatchState:
        next_calls, call_clock_available = _increment_words(state.policy_call_count_words)
        next_commits, commit_clock_available = _increment_words(state.dispatch_commit_count_words)
        clock_preflight_consistent = (call_clock_available == application.call_clock_available) & (
            commit_clock_available == application.commit_clock_available
        )
        record = (
            transaction_committed
            & application.call_attempted
            & application.call_clock_available
            & clock_preflight_consistent
        )
        record_commit = record & application.dispatch_authorized & commit_clock_available
        exhausted = (
            transaction_committed
            & application.call_attempted
            & (
                (~application.call_clock_available)
                | (application.dispatch_would_authorize & ~application.commit_clock_available)
                | (~clock_preflight_consistent)
            )
        )
        proposed = cast(
            PrototypeSTOMPCalibratedDispatchState,
            state.replace(
                sidecar=sidecar_state,
                policy_call_count_words=jnp.where(
                    record, next_calls, state.policy_call_count_words
                ),
                dispatch_commit_count_words=jnp.where(
                    record_commit,
                    next_commits,
                    state.dispatch_commit_count_words,
                ),
                last_record_valid=jnp.where(record, True, state.last_record_valid),
                last_decision_id=jnp.where(
                    record,
                    application.proposal.decision_id,
                    state.last_decision_id,
                ),
                last_proposal_content_tag=jnp.where(
                    record,
                    application.proposal.content_tag,
                    state.last_proposal_content_tag,
                ),
                last_safety_mask_digest=jnp.where(
                    record,
                    _checksum_arrays((application.proposal.safety_action_mask,)),
                    state.last_safety_mask_digest,
                ),
                last_planned_extended_action=jnp.where(
                    record,
                    application.proposal.planned_extended_action,
                    state.last_planned_extended_action,
                ),
                last_planned_option_index=jnp.where(
                    record,
                    application.proposal.planned_option_index,
                    state.last_planned_option_index,
                ),
                last_effective_primitive_action=jnp.where(
                    record,
                    application.effective_action,
                    state.last_effective_primitive_action,
                ),
                last_actual_credit_owner=jnp.where(
                    record,
                    application.actual_credit_owner,
                    state.last_actual_credit_owner,
                ),
                last_actual_executing_option=jnp.where(
                    record,
                    application.actual_executing_option,
                    state.last_actual_executing_option,
                ),
                last_policy_available=jnp.where(
                    record,
                    application.proposal.available,
                    state.last_policy_available,
                ),
                last_keyboard_used=jnp.where(
                    record,
                    application.proposal.keyboard_used,
                    state.last_keyboard_used,
                ),
                last_dispatch_committed=jnp.where(
                    record,
                    application.dispatch_authorized,
                    state.last_dispatch_committed,
                ),
                policy_unavailable=state.policy_unavailable | exhausted,
                policy_error=jnp.where(
                    exhausted,
                    jnp.int32(PROTOTYPE_STOMP_CALIBRATED_DISPATCH_ERROR_CLOCK_EXHAUSTED),
                    state.policy_error,
                ),
            ),
        )
        return self._with_checksum(proposed)

    def _arm_binding_valid(
        self,
        state: PrototypeSTOMPCalibratedSearchState,
        application: _PolicyApplication,
        arm_applied: Array,
    ) -> Array:
        stomp = self._sidecar._oak(state.prototype).stomp_state
        expected_kind = jnp.where(
            stomp.executing_option >= 0,
            jnp.int32(CANDIDATE_KIND_OPTION),
            jnp.int32(CANDIDATE_KIND_PRIMITIVE),
        )
        expected_index = jnp.where(
            stomp.executing_option >= 0,
            stomp.executing_option,
            application.effective_action,
        )
        return jnp.where(
            arm_applied,
            state.search.pending
            & (state.search.pending_executed_kind == expected_kind)
            & (state.search.pending_executed_index == expected_index)
            & jnp.array_equal(
                state.search.pending_decision_id,
                state.prototype.current_decision_id,
            ),
            True,
        )

    def _zero_proposal(
        self,
        sidecar_state: PrototypeSTOMPCalibratedSearchState,
        safety_action_mask: object,
    ) -> PrototypeSTOMPCalibratedDispatchProposal:
        # The ordinary proposal builder already yields the exact unavailable
        # sentinel when v2 is disabled or no resolved history exists.
        return self._form_proposal(
            self._blank_state(sidecar_state), sidecar_state, safety_action_mask
        )

    def _candidate_diagnostics(
        self, proposal: PrototypeSTOMPCalibratedDispatchProposal
    ) -> dict[str, Array]:
        cfg = self._config.sidecar.search
        index = jnp.clip(
            proposal.planned_extended_action,
            0,
            cfg.n_extended_actions - 1,
        )
        selected = proposal.available
        return {
            "candidate_eligible": proposal.candidate_eligible,
            "selected_candidate_flat_index": jnp.where(
                selected, proposal.candidate_flat_indices[index], jnp.int32(-1)
            ),
            "selected_candidate_target_available": selected
            & proposal.candidate_target_available[index],
            "selected_candidate_value_change_count": jnp.where(
                selected, proposal.candidate_value_change_counts[index], jnp.int32(0)
            ),
            "selected_candidate_model_error_count": jnp.where(
                selected, proposal.candidate_model_error_counts[index], jnp.int32(0)
            ),
            "selected_candidate_support_count": jnp.where(
                selected, proposal.candidate_support_counts[index], jnp.int32(0)
            ),
            "selected_candidate_value_change_lcb": jnp.where(
                selected, proposal.candidate_value_change_lcb[index], jnp.float32(0.0)
            ),
            "selected_candidate_reachability_lcb": jnp.where(
                selected, proposal.candidate_reachability_lcb[index], jnp.float32(0.0)
            ),
            "selected_candidate_model_reliability": jnp.where(
                selected, proposal.candidate_model_reliability[index], jnp.float32(0.0)
            ),
            "selected_candidate_support_shrinkage": jnp.where(
                selected, proposal.candidate_support_shrinkage[index], jnp.float32(0.0)
            ),
        }

    def _diagnostics(
        self,
        *,
        source: PrototypeSTOMPCalibratedDispatchState,
        final: PrototypeSTOMPCalibratedDispatchState,
        application: _PolicyApplication,
        prototype_learning_applied: Array,
        pending_before: Array,
        resolution_attempted: Array,
        arm_attempted: Array,
        arm_applied: Array,
        arm_bound: Array,
        transaction_committed: Array,
    ) -> PrototypeSTOMPCalibratedDispatchDiagnostics:
        dispatch_committed = transaction_committed & application.dispatch_authorized
        return PrototypeSTOMPCalibratedDispatchDiagnostics(
            state_valid_before=self.validate_state(source),
            state_valid_after=self.validate_state(final),
            prototype_learning_applied=prototype_learning_applied,
            prior_arm_pending_before=pending_before,
            prior_arm_resolution_attempted=resolution_attempted,
            prior_arm_settled_before_policy=application.proposal.prior_arm_settled,
            policy_enabled=jnp.asarray(self._config.enabled, dtype=jnp.bool_),
            policy_call_attempted=application.call_attempted,
            proposal_available=application.proposal.available,
            proposal_content_valid=jnp.array_equal(
                application.proposal.content_tag,
                self.proposal_content_tag(application.proposal),
            ),
            exact_anchor=application.proposal.exact_anchor,
            **self._candidate_diagnostics(application.proposal),
            planned_extended_action=application.proposal.planned_extended_action,
            planned_kind=application.proposal.planned_kind,
            planned_semantic_index=application.proposal.planned_semantic_index,
            planned_option_index=application.proposal.planned_option_index,
            keyboard_used=application.proposal.keyboard_used,
            keyboard_proposal_available=(application.proposal.keyboard_proposal_available),
            hard_mask_static_contract_valid=application.mask_static_valid,
            hard_mask_caller_owned=jnp.asarray(True, dtype=jnp.bool_),
            replacement_attempted=application.call_attempted,
            replacement_committed=(transaction_committed & application.replacement_committed),
            used_safe_current_owner_fallback=(
                dispatch_committed
                & application.replacement.dispatch_replacement.used_safe_base_fallback
            ),
            dispatch_authorized=dispatch_committed,
            action_changed=transaction_committed & application.action_changed,
            effective_primitive_action=jnp.where(
                dispatch_committed, application.effective_action, jnp.int32(-1)
            ),
            actual_credit_owner=jnp.where(
                dispatch_committed,
                application.actual_credit_owner,
                jnp.int32(DISPATCH_OWNER_INVALID),
            ),
            actual_executing_option=jnp.where(
                dispatch_committed,
                application.actual_executing_option,
                jnp.int32(-1),
            ),
            planned_option_started_by_dispatch=jnp.asarray(False, dtype=jnp.bool_),
            planned_option_matches_existing_owner=(
                application.planned_option_matches_existing_owner
            ),
            next_arm_attempted=arm_attempted,
            next_arm_applied=arm_applied,
            next_arm_bound_after_dispatch=arm_bound,
            one_total_backup_budget_preserved=jnp.asarray(True, dtype=jnp.bool_),
            additional_rng_draw_count=jnp.asarray(0, dtype=jnp.int32),
            additional_backward_pass_count=jnp.asarray(0, dtype=jnp.int32),
            additional_model_update_count=jnp.asarray(0, dtype=jnp.int32),
            safety_authority=jnp.asarray(False, dtype=jnp.bool_),
            physical_dispatch_authority=jnp.asarray(False, dtype=jnp.bool_),
            scientific_promotion_allowed=jnp.asarray(False, dtype=jnp.bool_),
            transaction_committed=transaction_committed,
            pre_policy_call_count_words=source.policy_call_count_words,
            post_policy_call_count_words=final.policy_call_count_words,
        )

    def start(
        self,
        state: PrototypeSTOMPCalibratedDispatchState,
        initial_observation: Array,
        *,
        safety_action_mask: Array,
    ) -> PrototypeSTOMPCalibratedDispatchStartResult:
        """Start Prototype, audit/replace the primitive, then arm from that owner."""

        self._check_state_contract(state)
        if not self._config.enabled:
            sidecar_result = self._sidecar.start(state.sidecar, initial_observation)
            candidate = self._with_checksum(
                cast(
                    PrototypeSTOMPCalibratedDispatchState,
                    state.replace(sidecar=sidecar_result.state),
                )
            )
            committed = self.validate_state(state) & self.validate_state(candidate)
            final = cast(
                PrototypeSTOMPCalibratedDispatchState,
                _tree_select(committed, candidate, state),
            )
            proposal = self._zero_proposal(final.sidecar, safety_action_mask)
            diagnostics = PrototypeSTOMPCalibratedDispatchDiagnostics(
                state_valid_before=self.validate_state(state),
                state_valid_after=self.validate_state(final),
                prototype_learning_applied=committed & sidecar_result.decision.armed,
                prior_arm_pending_before=jnp.asarray(False),
                prior_arm_resolution_attempted=jnp.asarray(False),
                prior_arm_settled_before_policy=jnp.asarray(True),
                policy_enabled=jnp.asarray(False),
                policy_call_attempted=jnp.asarray(False),
                proposal_available=jnp.asarray(False),
                proposal_content_valid=jnp.array_equal(
                    proposal.content_tag, self.proposal_content_tag(proposal)
                ),
                exact_anchor=proposal.exact_anchor,
                **self._candidate_diagnostics(proposal),
                planned_extended_action=jnp.int32(-1),
                planned_kind=jnp.int32(-1),
                planned_semantic_index=jnp.int32(-1),
                planned_option_index=jnp.int32(-1),
                keyboard_used=jnp.asarray(False),
                keyboard_proposal_available=proposal.keyboard_proposal_available,
                hard_mask_static_contract_valid=proposal.safety_mask_static_contract_valid,
                hard_mask_caller_owned=jnp.asarray(True),
                replacement_attempted=jnp.asarray(False),
                replacement_committed=jnp.asarray(False),
                used_safe_current_owner_fallback=jnp.asarray(False),
                dispatch_authorized=jnp.asarray(False),
                action_changed=jnp.asarray(False),
                effective_primitive_action=jnp.int32(-1),
                actual_credit_owner=jnp.int32(DISPATCH_OWNER_INVALID),
                actual_executing_option=jnp.int32(-1),
                planned_option_started_by_dispatch=jnp.asarray(False),
                planned_option_matches_existing_owner=jnp.asarray(False),
                next_arm_attempted=sidecar_result.diagnostics.arm_attempted,
                next_arm_applied=sidecar_result.diagnostics.arm_applied,
                next_arm_bound_after_dispatch=jnp.asarray(True),
                one_total_backup_budget_preserved=jnp.asarray(True),
                additional_rng_draw_count=jnp.int32(0),
                additional_backward_pass_count=jnp.int32(0),
                additional_model_update_count=jnp.int32(0),
                safety_authority=jnp.asarray(False),
                physical_dispatch_authority=jnp.asarray(False),
                scientific_promotion_allowed=jnp.asarray(False),
                transaction_committed=committed,
                pre_policy_call_count_words=state.policy_call_count_words,
                post_policy_call_count_words=final.policy_call_count_words,
            )
            return PrototypeSTOMPCalibratedDispatchStartResult(
                state=final,
                decision=sidecar_result.decision,
                sidecar_result=sidecar_result,
                proposal=proposal,
                replacement=None,
                diagnostics=diagnostics,
            )

        valid_before = self.validate_state(state)
        prototype = self._sidecar.prototype.start(state.sidecar.prototype, initial_observation)
        base_applied = prototype.started & self._sidecar.prototype.validate_state(prototype)
        next_revision, revision_available = _increment_int32(state.sidecar.revision)
        base_sidecar = cast(
            PrototypeSTOMPCalibratedSearchState,
            state.sidecar.replace(
                prototype=prototype,
                revision=next_revision,
                last_prototype_step_words=prototype.step_words,
            ),
        )
        base_sidecar = self._sidecar._with_checksum(base_sidecar)
        application = self._apply_policy(state, base_sidecar, safety_action_mask)
        armed, snapshot, arm, arm_diagnostics = self._sidecar._arm_next(application.state)
        arm_adopted = application.dispatch_authorized
        candidate_sidecar = cast(
            PrototypeSTOMPCalibratedSearchState,
            _tree_select(arm_adopted, armed, application.state),
        )
        candidate_valid = self._sidecar.validate_state(candidate_sidecar)
        candidate_arm_applied = arm_adopted & arm_diagnostics.arm_applied
        candidate_arm_bound = self._arm_binding_valid(
            candidate_sidecar,
            application,
            candidate_arm_applied,
        )
        preliminary = valid_before & base_applied & revision_available & candidate_valid
        preliminary = preliminary & candidate_arm_bound
        recorded = self._record_policy(
            state,
            candidate_sidecar,
            application,
            preliminary,
        )
        candidate_wrapper_valid = self.validate_state(recorded)
        committed = preliminary & candidate_wrapper_valid
        final = cast(
            PrototypeSTOMPCalibratedDispatchState,
            _tree_select(committed, recorded, state),
        )
        dispatch_ok = committed & application.dispatch_authorized
        decision = PrototypeDecision(
            observation=final.sidecar.prototype.current_raw_observation,
            action=jnp.where(
                dispatch_ok,
                final.sidecar.prototype.current_action,
                jnp.int32(-1),
            ),
            decision_id=final.sidecar.prototype.current_decision_id,
            armed=final.sidecar.prototype.started & dispatch_ok,
        )
        arm_applied = committed & candidate_arm_applied
        arm_bound = candidate_arm_bound
        sidecar_diagnostics = cast(
            PrototypeSTOMPCalibratedSearchDecisionDiagnostics,
            arm_diagnostics.replace(
                composed_state_valid=self._sidecar.validate_state(final.sidecar),
                prototype_decision_armed=decision.armed,
                arm_applied=arm_applied,
            ),
        )
        sidecar_result = PrototypeSTOMPCalibratedSearchStartResult(
            state=final.sidecar,
            decision=decision,
            model_snapshot=snapshot,
            search_arm=arm,
            diagnostics=sidecar_diagnostics,
        )
        diagnostics = self._diagnostics(
            source=state,
            final=final,
            application=application,
            prototype_learning_applied=committed & base_applied,
            pending_before=jnp.asarray(False),
            resolution_attempted=jnp.asarray(False),
            arm_attempted=arm_diagnostics.arm_attempted & dispatch_ok,
            arm_applied=arm_applied,
            arm_bound=arm_bound,
            transaction_committed=committed,
        )
        return PrototypeSTOMPCalibratedDispatchStartResult(
            state=final,
            decision=decision,
            sidecar_result=sidecar_result,
            proposal=application.proposal,
            replacement=application.replacement,
            diagnostics=diagnostics,
        )

    def decision(self, state: PrototypeSTOMPCalibratedDispatchState) -> PrototypeDecision:
        """Return only the exact currently authorized cached dispatch record."""

        valid = self.validate_state(state)
        if not self._config.enabled:
            decision = self._sidecar.prototype.decision(state.sidecar.prototype)
            return PrototypeDecision(
                observation=decision.observation,
                action=jnp.where(valid, decision.action, jnp.int32(-1)),
                decision_id=decision.decision_id,
                armed=valid & decision.armed,
            )
        decision = PrototypeDecision(
            observation=state.sidecar.prototype.current_raw_observation,
            action=state.sidecar.prototype.current_action,
            decision_id=state.sidecar.prototype.current_decision_id,
            armed=state.sidecar.prototype.started,
        )
        stomp = self._sidecar._oak(state.sidecar.prototype).stomp_state
        owner_matches = jnp.where(
            stomp.executing_option >= 0,
            (state.last_actual_credit_owner == DISPATCH_OWNER_OPTION)
            & (state.last_actual_executing_option == stomp.executing_option),
            (state.last_actual_credit_owner == DISPATCH_OWNER_BASE_PRIMITIVE)
            & (state.last_actual_executing_option == -1),
        )
        record_authorized = (
            valid
            & state.last_record_valid
            & state.last_dispatch_committed
            & decision.armed
            & jnp.array_equal(state.last_decision_id, decision.decision_id)
            & (state.last_effective_primitive_action == decision.action)
            & owner_matches
        )
        return PrototypeDecision(
            observation=decision.observation,
            action=jnp.where(
                record_authorized,
                state.last_effective_primitive_action,
                jnp.int32(-1),
            ),
            decision_id=decision.decision_id,
            armed=record_authorized,
        )

    def _transition_matches_authorized_dispatch(
        self,
        state: PrototypeSTOMPCalibratedDispatchState,
        transition: PrototypeTransition,
    ) -> Array:
        """Bind inbound learning to the exact v2 dispatch receipt."""

        authorized = self.decision(state)
        raw_observation = jnp.asarray(transition.observation)
        observation_contract = (
            raw_observation.shape == authorized.observation.shape
            and raw_observation.dtype == jnp.float32
        )
        observation = (
            raw_observation if observation_contract else jnp.zeros_like(authorized.observation)
        )
        raw_action = jnp.asarray(transition.action)
        action_contract = raw_action.shape == () and raw_action.dtype == jnp.int32
        action = raw_action if action_contract else jnp.int32(-1)
        raw_id = jnp.asarray(transition.decision_id)
        id_contract = raw_id.shape == (4,) and raw_id.dtype == jnp.uint32
        decision_id = raw_id if id_contract else jnp.zeros((4,), dtype=jnp.uint32)
        return (
            authorized.armed
            & jnp.asarray(observation_contract & action_contract & id_contract)
            & jnp.array_equal(
                jax.lax.bitcast_convert_type(observation, jnp.uint32),
                jax.lax.bitcast_convert_type(authorized.observation, jnp.uint32),
            )
            & (action == authorized.action)
            & jnp.array_equal(decision_id, authorized.decision_id)
        )

    def update_transition(
        self,
        state: PrototypeSTOMPCalibratedDispatchState,
        transition: PrototypeTransition,
        candidate_update_audit_evidence: (
            PrototypeCandidateUpdateAuditEvidence | None
        ) = None,
        *,
        gradient_joy_evidence: PrototypeGradientJoyEvidence | None = None,
        safety_action_mask: Array,
        experiential_memory_input: PrototypeExperientialMemoryInput | None = None,
        partner_policy_fusion_input: PrototypePartnerPolicyFusionInput | None = None,
        partner_policy_fusion_feedback: PrototypePartnerPolicyFusionFeedback | None = None,
    ) -> PrototypeSTOMPCalibratedDispatchUpdateResult:
        """Learn the real transition, settle, dispatch, and only then arm."""

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

        self._check_state_contract(state)
        if not self._config.enabled:
            sidecar_result = self._sidecar.update_transition(
                state.sidecar,
                transition,
                selected_audit_evidence,
                experiential_memory_input=experiential_memory_input,
                partner_policy_fusion_input=partner_policy_fusion_input,
                partner_policy_fusion_feedback=partner_policy_fusion_feedback,
            )
            candidate = self._with_checksum(
                cast(
                    PrototypeSTOMPCalibratedDispatchState,
                    state.replace(sidecar=sidecar_result.state),
                )
            )
            committed = self.validate_state(state) & self.validate_state(candidate)
            final = cast(
                PrototypeSTOMPCalibratedDispatchState,
                _tree_select(committed, candidate, state),
            )
            proposal = self._zero_proposal(final.sidecar, safety_action_mask)
            diagnostics = PrototypeSTOMPCalibratedDispatchDiagnostics(
                state_valid_before=self.validate_state(state),
                state_valid_after=self.validate_state(final),
                prototype_learning_applied=(
                    committed & sidecar_result.diagnostics.prototype_transition_applied
                ),
                prior_arm_pending_before=state.sidecar.adapter_pending,
                prior_arm_resolution_attempted=(sidecar_result.diagnostics.resolution_attempted),
                prior_arm_settled_before_policy=~final.sidecar.search.pending,
                policy_enabled=jnp.asarray(False),
                policy_call_attempted=jnp.asarray(False),
                proposal_available=jnp.asarray(False),
                proposal_content_valid=jnp.array_equal(
                    proposal.content_tag, self.proposal_content_tag(proposal)
                ),
                exact_anchor=proposal.exact_anchor,
                **self._candidate_diagnostics(proposal),
                planned_extended_action=jnp.int32(-1),
                planned_kind=jnp.int32(-1),
                planned_semantic_index=jnp.int32(-1),
                planned_option_index=jnp.int32(-1),
                keyboard_used=jnp.asarray(False),
                keyboard_proposal_available=proposal.keyboard_proposal_available,
                hard_mask_static_contract_valid=proposal.safety_mask_static_contract_valid,
                hard_mask_caller_owned=jnp.asarray(True),
                replacement_attempted=jnp.asarray(False),
                replacement_committed=jnp.asarray(False),
                used_safe_current_owner_fallback=jnp.asarray(False),
                dispatch_authorized=jnp.asarray(False),
                action_changed=jnp.asarray(False),
                effective_primitive_action=jnp.int32(-1),
                actual_credit_owner=jnp.int32(DISPATCH_OWNER_INVALID),
                actual_executing_option=jnp.int32(-1),
                planned_option_started_by_dispatch=jnp.asarray(False),
                planned_option_matches_existing_owner=jnp.asarray(False),
                next_arm_attempted=sidecar_result.diagnostics.arm_attempted,
                next_arm_applied=sidecar_result.diagnostics.arm_applied,
                next_arm_bound_after_dispatch=jnp.asarray(True),
                one_total_backup_budget_preserved=jnp.asarray(True),
                additional_rng_draw_count=jnp.int32(0),
                additional_backward_pass_count=jnp.int32(0),
                additional_model_update_count=jnp.int32(0),
                safety_authority=jnp.asarray(False),
                physical_dispatch_authority=jnp.asarray(False),
                scientific_promotion_allowed=jnp.asarray(False),
                transaction_committed=committed,
                pre_policy_call_count_words=state.policy_call_count_words,
                post_policy_call_count_words=final.policy_call_count_words,
            )
            return PrototypeSTOMPCalibratedDispatchUpdateResult(
                state=final,
                prototype=sidecar_result.prototype,
                decision=sidecar_result.decision,
                sidecar_result=sidecar_result,
                proposal=proposal,
                replacement=None,
                diagnostics=diagnostics,
            )

        valid_before = self.validate_state(state)
        inbound_dispatch_authorized = self._transition_matches_authorized_dispatch(
            state, transition
        )
        prototype_result = self._sidecar.prototype.update_transition(
            state.sidecar.prototype,
            transition,
            selected_audit_evidence,
            experiential_memory_input=experiential_memory_input,
            partner_policy_fusion_input=partner_policy_fusion_input,
            partner_policy_fusion_feedback=partner_policy_fusion_feedback,
        )
        expected_words, counter_available = _increment_words(state.sidecar.prototype.step_words)
        prototype_applied = (
            prototype_result.transition_diagnostics.valid
            & counter_available
            & jnp.array_equal(prototype_result.state.step_words, expected_words)
        )
        (
            resolved,
            observe,
            resolution_attempted,
            natural,
            censored,
            ownership,
            resolution_failed,
        ) = self._sidecar._resolve_pending(
            state.sidecar,
            transition,
            prototype_result.state,
            prototype_applied,
        )
        application = self._apply_policy(state, resolved, safety_action_mask)
        armed, snapshot, arm, arm_diagnostics = self._sidecar._arm_next(application.state)
        arm_adopted = application.dispatch_authorized
        candidate_sidecar = cast(
            PrototypeSTOMPCalibratedSearchState,
            _tree_select(arm_adopted, armed, application.state),
        )
        candidate_valid = self._sidecar.validate_state(candidate_sidecar)
        candidate_arm_applied = arm_adopted & arm_diagnostics.arm_applied
        candidate_arm_bound = self._arm_binding_valid(
            candidate_sidecar,
            application,
            candidate_arm_applied,
        )
        preliminary = (
            valid_before & inbound_dispatch_authorized & prototype_applied & candidate_valid
        )
        preliminary = preliminary & candidate_arm_bound
        recorded = self._record_policy(
            state,
            candidate_sidecar,
            application,
            preliminary,
        )
        candidate_wrapper_valid = self.validate_state(recorded)
        committed = preliminary & candidate_wrapper_valid
        final = cast(
            PrototypeSTOMPCalibratedDispatchState,
            _tree_select(committed, recorded, state),
        )
        dispatch_ok = committed & application.dispatch_authorized
        final_action = jnp.where(
            dispatch_ok,
            application.effective_action,
            jnp.asarray(-1, dtype=jnp.int32),
        )
        final_prototype = cast(
            PrototypeUpdateResult,
            prototype_result.replace(
                state=final.sidecar.prototype,
                action=final_action,
            ),
        )
        decision_raw = self._sidecar.prototype.decision(final.sidecar.prototype)
        decision = PrototypeDecision(
            observation=decision_raw.observation,
            action=jnp.where(dispatch_ok, decision_raw.action, jnp.int32(-1)),
            decision_id=decision_raw.decision_id,
            armed=decision_raw.armed & dispatch_ok,
        )
        arm_applied = committed & candidate_arm_applied
        arm_bound = candidate_arm_bound
        sidecar_failed = resolution_failed | arm_diagnostics.sidecar_failed
        sidecar_diagnostics = PrototypeSTOMPCalibratedSearchUpdateDiagnostics(
            composed_state_valid_before=self._sidecar.validate_state(state.sidecar),
            prototype_transition_applied=committed & prototype_applied,
            pending_before=state.sidecar.adapter_pending,
            ownership_binding_matches=ownership,
            resolution_attempted=committed & resolution_attempted,
            natural_resolution=committed & natural,
            censored_resolution=committed & censored,
            observe_applied=(
                jnp.asarray(False, dtype=jnp.bool_)
                if observe is None
                else committed & resolution_attempted & observe.diagnostics.transaction_valid
            ),
            arm_attempted=committed & dispatch_ok & arm_diagnostics.arm_attempted,
            arm_applied=arm_applied,
            search_unavailable=final.sidecar.search_unavailable,
            sidecar_failed_this_step=committed & sidecar_failed,
            prototype_retained_after_sidecar_failure=committed & sidecar_failed,
            composed_state_valid_after=self._sidecar.validate_state(final.sidecar),
            transaction_committed=committed,
            backup_attempt_budget=jnp.asarray(
                self._config.sidecar.search.backup_budget, dtype=jnp.int32
            ),
            planner_rng_draw_count=jnp.asarray(0, dtype=jnp.int32),
            policy_authority=jnp.asarray(False, dtype=jnp.bool_),
            keyboard_dispatch_applied=jnp.asarray(False, dtype=jnp.bool_),
        )
        sidecar_result = PrototypeSTOMPCalibratedSearchUpdateResult(
            state=final.sidecar,
            prototype=final_prototype,
            decision=decision,
            model_snapshot=snapshot,
            search_observe=observe,
            search_arm=arm,
            diagnostics=sidecar_diagnostics,
        )
        diagnostics = self._diagnostics(
            source=state,
            final=final,
            application=application,
            prototype_learning_applied=committed & prototype_applied,
            pending_before=state.sidecar.adapter_pending,
            resolution_attempted=resolution_attempted,
            arm_attempted=committed & dispatch_ok & arm_diagnostics.arm_attempted,
            arm_applied=arm_applied,
            arm_bound=arm_bound,
            transaction_committed=committed,
        )
        return PrototypeSTOMPCalibratedDispatchUpdateResult(
            state=final,
            prototype=final_prototype,
            decision=decision,
            sidecar_result=sidecar_result,
            proposal=application.proposal,
            replacement=application.replacement,
            diagnostics=diagnostics,
        )

    def retry_dispatch(
        self,
        state: PrototypeSTOMPCalibratedDispatchState,
        *,
        safety_action_mask: Array,
    ) -> PrototypeSTOMPCalibratedDispatchRetryResult:
        """Retry only the current unauthorized decision without learning.

        The retry is admissible only when the current Prototype decision has
        an exact v2 record whose previous dispatch was withheld, no calibrated
        arm is pending, and the v2 policy clocks are not quarantined.  A retry
        consumes a new policy-call receipt.  A successful replacement also
        consumes a dispatch-commit receipt and adopts exactly one new arm.
        """

        if not self._config.enabled:
            raise ValueError("retry_dispatch requires enabled v2 dispatch")
        self._check_state_contract(state)
        valid_before = self.validate_state(state)
        raw_decision = self._sidecar.prototype.decision(state.sidecar.prototype)
        current_unauthorized_record = (
            state.last_record_valid
            & jnp.array_equal(state.last_decision_id, raw_decision.decision_id)
            & (~state.last_dispatch_committed)
            & (state.last_effective_primitive_action == -1)
            & (state.last_actual_credit_owner == DISPATCH_OWNER_INVALID)
            & (state.last_actual_executing_option == -1)
        )
        retry_admissible = (
            valid_before
            & raw_decision.armed
            & current_unauthorized_record
            & (~state.sidecar.adapter_pending)
            & (~state.sidecar.search.pending)
            & (~state.policy_unavailable)
        )
        application = self._apply_policy(
            state,
            state.sidecar,
            safety_action_mask,
        )
        armed, snapshot, arm, arm_diagnostics = self._sidecar._arm_next(application.state)
        arm_adopted = retry_admissible & application.dispatch_authorized
        candidate_sidecar = cast(
            PrototypeSTOMPCalibratedSearchState,
            _tree_select(arm_adopted, armed, application.state),
        )
        candidate_arm_applied = arm_adopted & arm_diagnostics.arm_applied
        candidate_arm_bound = self._arm_binding_valid(
            candidate_sidecar,
            application,
            candidate_arm_applied,
        )
        preliminary = (
            retry_admissible & self._sidecar.validate_state(candidate_sidecar) & candidate_arm_bound
        )
        recorded = self._record_policy(
            state,
            candidate_sidecar,
            application,
            preliminary,
        )
        committed = preliminary & self.validate_state(recorded)
        final = cast(
            PrototypeSTOMPCalibratedDispatchState,
            _tree_select(committed, recorded, state),
        )
        dispatch_ok = committed & application.dispatch_authorized
        arm_applied = committed & candidate_arm_applied
        arm_bound = candidate_arm_bound
        decision = self.decision(final)
        diagnostics = self._diagnostics(
            source=state,
            final=final,
            application=application,
            prototype_learning_applied=jnp.asarray(False, dtype=jnp.bool_),
            pending_before=jnp.asarray(False, dtype=jnp.bool_),
            resolution_attempted=jnp.asarray(False, dtype=jnp.bool_),
            arm_attempted=dispatch_ok & arm_diagnostics.arm_attempted,
            arm_applied=arm_applied,
            arm_bound=arm_bound,
            transaction_committed=committed,
        )
        return PrototypeSTOMPCalibratedDispatchRetryResult(
            state=final,
            decision=decision,
            proposal=application.proposal,
            replacement=application.replacement,
            model_snapshot=snapshot,
            search_arm=arm,
            diagnostics=diagnostics,
        )

    def dispatch_current(
        self,
        state: PrototypeSTOMPCalibratedDispatchState,
        *,
        safety_action_mask: Array,
    ) -> PrototypeSTOMPCalibratedDispatchRetryResult:
        """Alias for :meth:`retry_dispatch` for dispatch-oriented callers."""

        return self.retry_dispatch(state, safety_action_mask=safety_action_mask)

    def scan_transitions(
        self,
        state: PrototypeSTOMPCalibratedDispatchState,
        transitions: PrototypeTransition,
        safety_action_masks: Array,
    ) -> PrototypeSTOMPCalibratedDispatchArrayResult:
        """Run the v2 transaction through one fixed-shape ``jax.lax.scan``."""

        def step(
            carry: PrototypeSTOMPCalibratedDispatchState,
            inputs: tuple[PrototypeTransition, Array],
        ) -> tuple[PrototypeSTOMPCalibratedDispatchState, tuple[Array, ...]]:
            transition, mask = inputs
            result = self.update_transition(
                carry,
                transition,
                safety_action_mask=mask,
            )
            return result.state, (
                result.decision.action,
                result.diagnostics.prototype_learning_applied,
                result.diagnostics.proposal_available,
                result.diagnostics.dispatch_authorized,
                result.diagnostics.action_changed,
                result.diagnostics.next_arm_applied,
                result.diagnostics.transaction_committed,
            )

        final, rows = jax.lax.scan(step, state, (transitions, safety_action_masks))
        return PrototypeSTOMPCalibratedDispatchArrayResult(
            state=final,
            actions=rows[0],
            prototype_learning_applied=rows[1],
            proposal_available=rows[2],
            dispatch_authorized=rows[3],
            action_changed=rows[4],
            next_arm_applied=rows[5],
            transaction_committed=rows[6],
        )

    def checkpoint_payload(self, state: PrototypeSTOMPCalibratedDispatchState) -> dict[str, object]:
        """Return a host-only v2 checkpoint with complete state SHA binding."""

        if not bool(jax.device_get(self.validate_state(state))):
            raise ValueError("cannot checkpoint an invalid calibrated-dispatch state")
        return {
            "schema": PROTOTYPE_STOMP_CALIBRATED_DISPATCH_CHECKPOINT_SCHEMA,
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
    ) -> PrototypeSTOMPCalibratedDispatchState:
        """Restore only exact config, SHA, source, and representation bindings."""

        if type(payload) is not dict:
            raise ValueError("calibrated-dispatch checkpoint must be an exact dict")
        raw = cast(dict[object, object], payload)
        if set(raw) != {"schema", "config", "state", "state_sha256"}:
            raise ValueError("calibrated-dispatch checkpoint fields differ from v2")
        if raw["schema"] != PROTOTYPE_STOMP_CALIBRATED_DISPATCH_CHECKPOINT_SCHEMA:
            raise ValueError("calibrated-dispatch checkpoint schema differs")
        if PrototypeSTOMPCalibratedDispatchConfig.from_config(raw["config"]) != self._config:
            raise ValueError("calibrated-dispatch checkpoint config differs")
        restored = raw["state"]
        if type(restored) is not PrototypeSTOMPCalibratedDispatchState:
            raise ValueError("calibrated-dispatch checkpoint state type differs")
        digest = _require_array(
            raw["state_sha256"],
            name="checkpoint.state_sha256",
            shape=(_DIGEST_BYTES,),
            dtype=jnp.uint8,
        )
        if not bool(jax.device_get(jnp.array_equal(digest, _tree_sha256(restored)))):
            raise ValueError("calibrated-dispatch checkpoint SHA differs")
        source = _require_array(
            source_digest,
            name="source_digest",
            shape=(2,),
            dtype=jnp.uint32,
        )
        generation = (
            jnp.asarray(representation_generation, dtype=jnp.int32)
            if type(representation_generation) is int
            else _require_array(
                representation_generation,
                name="representation_generation",
                shape=(),
                dtype=jnp.int32,
            )
        )
        binding = jnp.array_equal(restored.sidecar.search.source_digest, source) & (
            restored.sidecar.search.representation_generation == generation
        )
        if not bool(jax.device_get(binding & self.validate_state(restored))):
            raise ValueError("calibrated-dispatch checkpoint is invalid, stale, or rebound")
        return restored

    def resource_budget(
        self, state: PrototypeSTOMPCalibratedDispatchState
    ) -> PrototypeSTOMPCalibratedDispatchResourceBudget:
        """Return exact storage plus fixed incremental policy work bounds."""

        self._check_state_contract(state)
        sidecar_budget = self._sidecar.resource_budget(state.sidecar)
        sidecar_bytes = _tree_nbytes(state.sidecar)
        total_bytes = _tree_nbytes(state)
        cfg = self._config.sidecar.search
        return PrototypeSTOMPCalibratedDispatchResourceBudget(
            sidecar_persistent_state_nbytes=sidecar_bytes,
            dispatch_binding_nbytes=total_bytes - sidecar_bytes,
            total_persistent_state_nbytes=total_bytes,
            anchor_capacity=cfg.anchor_capacity,
            n_extended_actions=cfg.n_extended_actions,
            n_primitive_actions=cfg.n_primitive_actions,
            n_options=cfg.n_options,
            total_secondary_backup_attempts_per_resolution=(
                sidecar_budget.total_secondary_backup_attempts_per_resolution
            ),
            primitive_and_option_share_one_budget=True,
            max_q_values_interpreted_per_policy_call=cfg.n_extended_actions,
            max_argmax_comparisons_per_policy_call=max(0, cfg.n_extended_actions - 1),
            max_keyboard_proposals_per_policy_call=1,
            max_cached_action_replacements_per_policy_call=1,
            max_next_arm_calls_per_decision=1,
            additional_rng_draws_per_policy_call=0,
            additional_backward_passes_per_policy_call=0,
            additional_model_updates_per_policy_call=0,
            persistent_state_growth_per_transition_bytes=0,
            planned_option_starts_option=False,
            proposal_unavailability_can_dispatch_safe_current_owner=True,
            proposal_available_distinct_from_dispatch_authorized=True,
            hard_safety_masks_per_policy_call=1,
            safety_authority=False,
            physical_dispatch_authority=False,
            control_benefit_assessed=False,
            scientific_promotion_allowed=False,
            enabled_by_default=False,
            checkpoint_schema=PROTOTYPE_STOMP_CALIBRATED_DISPATCH_CHECKPOINT_SCHEMA,
        )


# The longer name makes the v2 relationship explicit for discovery tools.
PrototypeSTOMPCalibratedSearchDispatchAgent = PrototypeSTOMPCalibratedDispatchAgent
PrototypeSTOMPCalibratedSearchDispatchConfig = PrototypeSTOMPCalibratedDispatchConfig
PrototypeSTOMPCalibratedSearchDispatchState = PrototypeSTOMPCalibratedDispatchState


__all__ = [
    "PROTOTYPE_STOMP_CALIBRATED_DISPATCH_ASSESSMENT",
    "PROTOTYPE_STOMP_CALIBRATED_DISPATCH_CHECKPOINT_HOST_ONLY",
    "PROTOTYPE_STOMP_CALIBRATED_DISPATCH_CHECKPOINT_SCHEMA",
    "PROTOTYPE_STOMP_CALIBRATED_DISPATCH_CONFIG_SCHEMA",
    "PROTOTYPE_STOMP_CALIBRATED_DISPATCH_ERROR_CLOCK_EXHAUSTED",
    "PROTOTYPE_STOMP_CALIBRATED_DISPATCH_ERROR_NONE",
    "PROTOTYPE_STOMP_CALIBRATED_DISPATCH_EVIDENCE_LEVEL",
    "PROTOTYPE_STOMP_CALIBRATED_DISPATCH_MECHANISM_STATUS",
    "PROTOTYPE_STOMP_CALIBRATED_DISPATCH_PHYSICAL_AUTHORITY",
    "PROTOTYPE_STOMP_CALIBRATED_DISPATCH_PLANNED_OPTION_STARTS_OPTION",
    "PROTOTYPE_STOMP_CALIBRATED_DISPATCH_SAFETY_AUTHORITY",
    "PROTOTYPE_STOMP_CALIBRATED_DISPATCH_SCIENTIFIC_PROMOTION_ALLOWED",
    "PROTOTYPE_STOMP_CALIBRATED_DISPATCH_STATE_SCHEMA",
    "PrototypeSTOMPCalibratedDispatchAgent",
    "PrototypeSTOMPCalibratedDispatchArrayResult",
    "PrototypeSTOMPCalibratedDispatchConfig",
    "PrototypeSTOMPCalibratedDispatchDiagnostics",
    "PrototypeSTOMPCalibratedDispatchProposal",
    "PrototypeSTOMPCalibratedDispatchResourceBudget",
    "PrototypeSTOMPCalibratedDispatchRetryResult",
    "PrototypeSTOMPCalibratedDispatchStartResult",
    "PrototypeSTOMPCalibratedDispatchState",
    "PrototypeSTOMPCalibratedDispatchUpdateResult",
    "PrototypeSTOMPCalibratedSearchDispatchAgent",
    "PrototypeSTOMPCalibratedSearchDispatchConfig",
    "PrototypeSTOMPCalibratedSearchDispatchState",
]
