# mypy: disable-error-code="attr-defined,call-arg,arg-type,type-var"
"""Persistent STOMP-to-option-lifecycle audit composition.

This opt-in L0 wrapper executes the real :class:`STOMPAgent` and records each
dispatched primitive transition in :class:`OptionLifecycleAudit`.  It derives
option ownership, starts, natural goal/timeout/environment terminations,
censored boundaries, frozen pre-update option-model predictions, discounted
return inputs, outcome deltas, and deterministic planning attribution from the
actual STOMP transaction.  The caller supplies only context and, for an idle
primitive opportunity, the comparator candidate/randomization declaration.

The auditor has zero control authority over STOMP.  From a valid persistent
composition, a valid real STOMP update always commits.  If the audit is
exhausted or rejects malformed/stale external attribution metadata, its state
is frozen, the composition records a terminal audit error, and subsequent real
STOMP updates continue as audit-unavailable no-ops.  Corruption of the
persistent composed state itself still fails closed and requires checkpoint
recovery.  Disabling auditing adds no RNG calls and returns the bit-identical
STOMP transition produced by the wrapped agent while leaving audit state
unchanged.

Semantic rebinding is explicit and bounded.  A new, shape-compatible wrapper
may preserve bit-identical slots and reset changed option policies, models,
traces, optimizer state, and base option heads from an explicitly keyed fresh
template.  Rebinding is deferred while STOMP, the audit, or a comparator trial
is in flight.

This module has no dispatch, option-selection, curation, promotion, go/no-go,
or scientific-evidence authority.  In particular, it observes randomized
comparator declarations but cannot force STOMP to honor a multi-step treatment
or primitive assignment; incompatible attribution stops the observer, never
the real controller.
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
import jax.random as jr
import numpy as np
from jax import Array
from jaxtyping import Bool, Float, Int, UInt

from alberta_framework.core.option_lifecycle_audit import (
    OptionLifecycleAudit,
    OptionLifecycleAuditState,
    option_semantic_digest,
)
from alberta_framework.core.options import (
    STOMPAgent,
    STOMPState,
    STOMPUpdateResult,
    _checked_lifetime_words_advance,
    measure_stomp_state_nbytes,
)
from alberta_framework.core.stomp_owner_finalization import (
    STOMPOwnerFinalizationTrace,
    stomp_owner_finalization_trace_valid,
)

STOMP_OPTION_LIFECYCLE_CONFIG_SCHEMA = "alberta.stomp-option-lifecycle.config.v1"
STOMP_OPTION_LIFECYCLE_CHECKPOINT_SCHEMA = "alberta.stomp-option-lifecycle.state.v1"
STOMP_OPTION_LIFECYCLE_BORROWED_BINDING_SCHEMA = (
    "alberta.stomp-option-lifecycle.borrowed-binding.v1"
)
STOMP_OPTION_LIFECYCLE_CURATION_AUTHORITY = False
STOMP_OPTION_LIFECYCLE_PROMOTION_AUTHORITY = False
STOMP_OPTION_LIFECYCLE_DISPATCH_AUTHORITY = False
STOMP_OPTION_LIFECYCLE_GO_NO_GO_AUTHORITY = False
STOMP_OPTION_LIFECYCLE_SCIENTIFIC_PROMOTION_ALLOWED = False
STOMP_OPTION_LIFECYCLE_AUDIT_ERROR_NONE = 0
STOMP_OPTION_LIFECYCLE_AUDIT_ERROR_CAPACITY = 1
STOMP_OPTION_LIFECYCLE_AUDIT_ERROR_PERSISTENT_STATE_INVALID = 2
STOMP_OPTION_LIFECYCLE_AUDIT_ERROR_ATTRIBUTION_REJECTED = 3

_DIGEST_WORDS = 8
_LIFECYCLE_WORDS = 2
_INT32_MAX = 2_147_483_647
_BORROWED_BINDING_TAG = jnp.asarray((0x42535431, 0x00000001), dtype=jnp.uint32)


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


def _checksum_arrays(arrays: tuple[Array, ...]) -> Array:
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


def _static_uint32_tag(text: str) -> int:
    """Return a stable non-cryptographic tag for static dtype/schema text."""

    value = 0x811C9DC5
    for byte in text.encode("ascii"):
        value ^= byte
        value = (value * 0x01000193) & 0xFFFFFFFF
    return value


def _typed_tree_checksum(value: object) -> Array:
    """Return an array-only, type-and-shape-tagged checksum of a JAX tree.

    This checksum is an exact-source integrity binding for the supported STOMP
    leaf dtypes.  It is deliberately unkeyed and is not authentication.  The
    static dtype/shape tags prevent values with different array
    representations from being silently treated as the same source.
    """

    payload: list[Array] = []
    for index, leaf in enumerate(jax.tree_util.tree_leaves(value)):
        array = jnp.asarray(leaf)
        dtype_text = str(array.dtype)
        if jax.dtypes.issubdtype(array.dtype, jax.dtypes.prng_key):
            array = jr.key_data(array)
        elif array.dtype not in (jnp.float32, jnp.int32, jnp.uint32, jnp.bool_):
            raise TypeError(
                "borrowed STOMP binding supports only float32, int32, uint32, "
                "bool, and typed PRNG-key leaves"
            )
        header = jnp.asarray(
            (
                index,
                _static_uint32_tag(dtype_text),
                array.ndim,
                array.size,
                *array.shape,
            ),
            dtype=jnp.uint32,
        )
        payload.extend((header, array))
    lanes = tuple(
        _checksum_arrays(
            (
                jnp.asarray((salt, lane), dtype=jnp.uint32),
                *payload,
            )
        )
        for lane, salt in enumerate(
            (0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344)
        )
    )
    return jnp.concatenate(lanes).astype(jnp.uint32)


def _trees_exactly_equal(left: object, right: object) -> Bool[Array, ""]:
    """Return bit-exact equality for two matching array pytrees."""

    left_leaves, left_tree = jax.tree_util.tree_flatten(left)
    right_leaves, right_tree = jax.tree_util.tree_flatten(right)
    if cast(object, left_tree) != cast(object, right_tree):
        return jnp.asarray(False, dtype=jnp.bool_)
    if len(left_leaves) != len(right_leaves):
        return jnp.asarray(False, dtype=jnp.bool_)
    valid = jnp.asarray(True, dtype=jnp.bool_)
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        left_array = jnp.asarray(left_leaf)
        right_array = jnp.asarray(right_leaf)
        if left_array.shape != right_array.shape or left_array.dtype != right_array.dtype:
            return jnp.asarray(False, dtype=jnp.bool_)
        if jax.dtypes.issubdtype(left_array.dtype, jax.dtypes.prng_key):
            valid = valid & jnp.array_equal(
                jr.key_data(left_array),
                jr.key_data(right_array),
            )
        elif left_array.dtype == jnp.float32:
            valid = valid & jnp.array_equal(
                jax.lax.bitcast_convert_type(left_array, jnp.uint32),
                jax.lax.bitcast_convert_type(right_array, jnp.uint32),
            )
        else:
            valid = valid & jnp.array_equal(left_array, right_array)
    return valid


def _float32_bits_equal(left: Array, right: Array) -> Array:
    return jnp.array_equal(
        jax.lax.bitcast_convert_type(jnp.asarray(left, dtype=jnp.float32), jnp.uint32),
        jax.lax.bitcast_convert_type(jnp.asarray(right, dtype=jnp.float32), jnp.uint32),
    )


def _validate_stomp_update_result_contract(result: STOMPUpdateResult) -> None:
    if type(result) is not STOMPUpdateResult:
        raise TypeError("stomp_result must be an exact STOMPUpdateResult")
    if type(result.state) is not STOMPState:
        raise TypeError("stomp_result.state must be an exact STOMPState")
    for name in (
        "td_error",
        "average_reward",
        "pseudo_reward",
        "option_importance_ratio",
        "planning_td_error",
    ):
        _require_array(
            getattr(result, name),
            name=f"stomp_result.{name}",
            shape=(),
            dtype=jnp.float32,
        )
    for name in (
        "primitive_action",
        "executing_option",
        "planning_backups",
        "nested_updates_required",
        "nested_updates_applied",
    ):
        _require_array(
            getattr(result, name),
            name=f"stomp_result.{name}",
            shape=(),
            dtype=jnp.int32,
        )
    for name in (
        "option_terminated",
        "inputs_valid",
        "lifetime_counter_valid",
        "lifetime_capacity_available",
        "nested_lifetime_counter_valid",
        "nested_lifetime_capacity_available",
        "proposed_state_valid",
        "update_applied",
    ):
        _require_array(
            getattr(result, name),
            name=f"stomp_result.{name}",
            shape=(),
            dtype=jnp.bool_,
        )
    for name in ("pre_step_words", "post_step_words"):
        _require_array(
            getattr(result, name),
            name=f"stomp_result.{name}",
            shape=(_LIFECYCLE_WORDS,),
            dtype=jnp.uint32,
        )


def _increment_words(words: Array) -> tuple[Array, Array]:
    low = words[1] + jnp.uint32(1)
    carry = (low == 0).astype(jnp.uint32)
    high = words[0] + carry
    available = ~((carry != 0) & (high == 0))
    candidate = jnp.stack((high, low), dtype=jnp.uint32)
    return jnp.where(available, candidate, words), available


def _canonical_digest(value: object) -> Array:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    raw = hashlib.sha256(canonical.encode("utf-8")).digest()
    words = tuple(
        int.from_bytes(raw[offset : offset + 4], "little")
        for offset in range(0, 32, 4)
    )
    return jnp.asarray(words, dtype=jnp.uint32)


def _saturating_revision_increment(revision: Array) -> Array:
    return jnp.where(
        revision < jnp.int32(_INT32_MAX),
        revision + jnp.int32(1),
        jnp.int32(_INT32_MAX),
    )


@dataclasses.dataclass(frozen=True, slots=True)
class STOMPOptionLifecycleConfig:
    """Static enablement and logical option-step compute declaration."""

    audit_enabled: bool = True
    option_compute_units_per_step: float = 1.0

    SCHEMA_VERSION: ClassVar[str] = STOMP_OPTION_LIFECYCLE_CONFIG_SCHEMA

    def __post_init__(self) -> None:
        if type(self.audit_enabled) is not bool:
            raise ValueError("audit_enabled must be an exact Python bool")
        if (
            type(self.option_compute_units_per_step) is not float
            or not math.isfinite(self.option_compute_units_per_step)
            or self.option_compute_units_per_step < 0.0
            or not math.isfinite(float(np.float32(self.option_compute_units_per_step)))
        ):
            raise ValueError(
                "option_compute_units_per_step must be a non-negative finite float32 value"
            )

    def to_config(self) -> dict[str, object]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "audit_enabled": self.audit_enabled,
            "option_compute_units_per_step": self.option_compute_units_per_step,
            "curation_authority": False,
            "promotion_authority": False,
            "dispatch_authority": False,
            "go_no_go_authority": False,
            "scientific_promotion_allowed": False,
        }

    @classmethod
    def from_config(cls, value: Mapping[str, object]) -> STOMPOptionLifecycleConfig:
        if type(value) is not dict:
            raise ValueError("STOMP lifecycle config must be an exact dict")
        raw = dict(value)
        expected = {
            "schema_version",
            "audit_enabled",
            "option_compute_units_per_step",
            "curation_authority",
            "promotion_authority",
            "dispatch_authority",
            "go_no_go_authority",
            "scientific_promotion_allowed",
        }
        if set(raw) != expected:
            raise ValueError("STOMP lifecycle config keys differ from schema v1")
        if raw.pop("schema_version") != cls.SCHEMA_VERSION:
            raise ValueError("STOMP lifecycle config schema_version differs")
        for authority in (
            "curation_authority",
            "promotion_authority",
            "dispatch_authority",
            "go_no_go_authority",
            "scientific_promotion_allowed",
        ):
            if raw.pop(authority) is not False:
                raise ValueError(f"STOMP lifecycle config cannot claim {authority}")
        return cls(**cast(dict[str, Any], raw))


@chex.dataclass(frozen=True)
class STOMPOptionLifecycleState:
    """Real STOMP state and its persistent, identity-bound audit sidecar."""

    stomp_state: STOMPState
    audit_state: OptionLifecycleAuditState
    lifecycle_id: UInt[Array, " 2"]
    stomp_structure_digest: UInt[Array, " 8"]
    started: Bool[Array, ""]
    revision: Int[Array, ""]
    audit_unavailable: Bool[Array, ""]
    audit_error: Int[Array, ""]
    binding_checksum: UInt[Array, " 2"]


@chex.dataclass(frozen=True)
class STOMPOptionLifecycleMetadataState:
    """Detached lifecycle sidecar borrowing one externally owned STOMP state.

    No :class:`STOMPState` is reachable from this tree.  ``stomp_binding_checksum``
    is a typed, shape-tagged, unkeyed integrity checksum of the sole external
    owner.  It is not a credential and does not authenticate the caller.
    """

    audit_state: OptionLifecycleAuditState
    lifecycle_id: UInt[Array, " 2"]
    stomp_structure_digest: UInt[Array, " 8"]
    started: Bool[Array, ""]
    revision: Int[Array, ""]
    audit_unavailable: Bool[Array, ""]
    audit_error: Int[Array, ""]
    stomp_step_count: Int[Array, ""]
    stomp_step_words: UInt[Array, " 2"]
    stomp_executing_option: Int[Array, ""]
    stomp_binding_tag: UInt[Array, " 2"]
    stomp_binding_checksum: UInt[Array, " 8"]
    source_binding_checksum: UInt[Array, " 2"]
    metadata_checksum: UInt[Array, " 2"]


@chex.dataclass(frozen=True)
class STOMPOptionLifecycleBorrowResult:
    """Fail-closed result of transiently attaching a borrowed STOMP owner."""

    state: STOMPOptionLifecycleState
    metadata_valid: Bool[Array, ""]
    stomp_state_valid: Bool[Array, ""]
    binding_matches: Bool[Array, ""]
    transaction_applied: Bool[Array, ""]
    caller_authenticated: Bool[Array, ""]


@chex.dataclass(frozen=True)
class STOMPOptionLifecycleExternalTransitionDeclaration:
    """Trusted-caller declaration binding one already-evaluated STOMP result.

    Every field is independently checked against the borrowed source, supplied
    result, and transition inputs.  The declaration is unkeyed: setting
    ``caller_derivation_declared`` asserts lifecycle authority but supplies no
    credential and therefore does not authenticate the caller.
    """

    source_stomp_checksum: UInt[Array, " 8"]
    destination_stomp_checksum: UInt[Array, " 8"]
    pre_step_words: UInt[Array, " 2"]
    post_step_words: UInt[Array, " 2"]
    source_executing_option: Int[Array, ""]
    destination_executing_option: Int[Array, ""]
    primitive_action: Int[Array, ""]
    option_terminated: Bool[Array, ""]
    natural_completion: Bool[Array, ""]
    censor_only_ending: Bool[Array, ""]
    external_reward: Float[Array, ""]
    pseudo_reward: Float[Array, ""]
    average_reward: Float[Array, ""]
    td_error: Float[Array, ""]
    planning_backups: Int[Array, ""]
    extended_action_mask: Bool[Array, " n_total_actions"]
    frozen_model_signature: Float[Array, " signature_dim"]
    caller_derivation_declared: Bool[Array, ""]


@chex.dataclass(frozen=True)
class STOMPOptionLifecycleExternalStartDeclaration:
    """Trusted-caller binding for one already-evaluated STOMP start."""

    source_stomp_checksum: UInt[Array, " 8"]
    destination_stomp_checksum: UInt[Array, " 8"]
    primitive_action: Int[Array, ""]
    executing_option: Int[Array, ""]
    caller_derivation_declared: Bool[Array, ""]


@chex.dataclass(frozen=True)
class STOMPOptionLifecycleExternalStartAdoptionResult:
    """Detached metadata advance for an authoritative external STOMP start."""

    state: STOMPOptionLifecycleMetadataState
    source_binding_matches: Bool[Array, ""]
    destination_state_valid: Bool[Array, ""]
    clocks_preserved: Bool[Array, ""]
    endpoint_binding_valid: Bool[Array, ""]
    declaration_binding_valid: Bool[Array, ""]
    metadata_advanced: Bool[Array, ""]
    derivation_recomputed: Bool[Array, ""]
    caller_authority_required: Bool[Array, ""]
    caller_authenticated: Bool[Array, ""]


@chex.dataclass(frozen=True)
class STOMPOptionLifecycleExternalAdoptionResult:
    """Detached audit advance for one authoritative external control result."""

    state: STOMPOptionLifecycleMetadataState
    source_metadata_valid: Bool[Array, ""]
    source_stomp_valid: Bool[Array, ""]
    source_binding_matches: Bool[Array, ""]
    result_static_contract_valid: Bool[Array, ""]
    result_clock_binding_valid: Bool[Array, ""]
    result_endpoint_binding_valid: Bool[Array, ""]
    termination_binding_valid: Bool[Array, ""]
    reward_binding_valid: Bool[Array, ""]
    model_signature_binding_valid: Bool[Array, ""]
    declaration_binding_valid: Bool[Array, ""]
    audit_applied: Bool[Array, ""]
    metadata_advanced: Bool[Array, ""]
    control_transition_rolled_back: Bool[Array, ""]
    derivation_recomputed: Bool[Array, ""]
    caller_authority_required: Bool[Array, ""]
    caller_authenticated: Bool[Array, ""]
    transaction_applied: Bool[Array, ""]


@chex.dataclass(frozen=True)
class STOMPOptionLifecycleExternalOwnerFinalizationResult:
    """Metadata-only adoption of one classified raw-to-final owner trace."""

    state: STOMPOptionLifecycleMetadataState
    raw_metadata_valid: Bool[Array, ""]
    raw_owner_binding_matches: Bool[Array, ""]
    final_owner_state_valid: Bool[Array, ""]
    stage_trace_valid: Bool[Array, ""]
    audit_state_preserved: Bool[Array, ""]
    lifecycle_identity_preserved: Bool[Array, ""]
    metadata_finalized: Bool[Array, ""]
    derivation_recomputed: Bool[Array, ""]
    caller_authenticated: Bool[Array, ""]


@chex.dataclass(frozen=True)
class STOMPOptionLifecycleStartResult:
    """Potentially primed composition and first primitive dispatch."""

    state: STOMPOptionLifecycleState
    primitive_action: Int[Array, ""]
    applied: Bool[Array, ""]
    audit_enabled: Bool[Array, ""]


@chex.dataclass(frozen=True)
class STOMPOptionLifecycleUpdateResult:
    """Real-STOMP update plus non-authoritative audit diagnostics."""

    state: STOMPOptionLifecycleState
    td_error: Float[Array, ""]
    average_reward: Float[Array, ""]
    primitive_action: Int[Array, ""]
    executing_option: Int[Array, ""]
    option_terminated: Bool[Array, ""]
    natural_completion: Bool[Array, ""]
    censor_only_ending: Bool[Array, ""]
    pseudo_reward: Float[Array, ""]
    planning_usage: Int[Array, " n_options"]
    pre_step_words: UInt[Array, " 2"]
    post_step_words: UInt[Array, " 2"]
    transaction_identity: UInt[Array, " 4"]
    stomp_update_applied: Bool[Array, ""]
    audit_applied: Bool[Array, ""]
    audit_sidecar_accepted: Bool[Array, ""]
    audit_capacity_available: Bool[Array, ""]
    audit_unavailable_noop: Bool[Array, ""]
    audit_error: Int[Array, ""]
    transaction_applied: Bool[Array, ""]
    rolled_back: Bool[Array, ""]
    audit_enabled: Bool[Array, ""]


@chex.dataclass(frozen=True)
class STOMPOptionLifecycleRebindResult:
    """Explicit semantic transfer/reset result; never an autonomous decision."""

    state: STOMPOptionLifecycleState
    transaction_valid: Bool[Array, ""]
    applied: Bool[Array, ""]
    deferred: Bool[Array, ""]
    preserved_slots: Bool[Array, " n_options"]
    reset_slots: Bool[Array, " n_options"]


@dataclasses.dataclass(frozen=True, slots=True)
class STOMPOptionLifecycleResourceBudget:
    """Exact wrapped allocation and additional work/authority declaration."""

    wrapped_persistent_state_nbytes: int
    stomp_persistent_state_nbytes: int
    audit_persistent_state_nbytes: int
    composition_binding_nbytes: int
    option_slot_nbytes: int
    stomp_updates_per_update: int
    audit_arms_per_enabled_update: int
    audit_observations_per_enabled_update: int
    additional_rng_draws_per_start: int
    additional_rng_draws_per_update: int
    additional_backward_passes_per_update: int
    additional_consumer_calls_per_update: int
    max_planning_attribution_slots_per_update: int
    semantic_slots_examined_per_rebind: int
    max_audited_observations: int
    stomp_lifetime_identity_bits: int
    wrapper_revision_saturation: int
    audit_capacity_can_block_stomp: bool
    curation_authority: bool
    promotion_authority: bool
    dispatch_authority: bool
    go_no_go_authority: bool
    scientific_promotion_allowed: bool
    checkpoint_schema: str


class STOMPOptionLifecycle:
    """Live persistent composition over one STOMP agent and one L0 auditor."""

    def __init__(
        self,
        stomp_agent: STOMPAgent,
        audit: OptionLifecycleAudit,
        config: STOMPOptionLifecycleConfig | None = None,
        *,
        external_semantic_digests: Array | None = None,
    ) -> None:
        if type(stomp_agent) is not STOMPAgent:
            raise TypeError("stomp_agent must be an exact STOMPAgent")
        if type(audit) is not OptionLifecycleAudit:
            raise TypeError("audit must be an exact OptionLifecycleAudit")
        self._agent = stomp_agent
        self._audit = audit
        self._config = config or STOMPOptionLifecycleConfig()
        if audit.config.n_options != stomp_agent.config.n_options:
            raise ValueError("audit and STOMP option counts must match exactly")
        if audit.config.outcome_dim != stomp_agent.config.observation_dim:
            raise ValueError("audit outcome_dim must equal STOMP observation_dim")
        if (
            self._config.option_compute_units_per_step
            > audit.config.max_compute_cost_per_observation
        ):
            raise ValueError("declared option compute cost exceeds the audit input ceiling")
        if (
            stomp_agent.config.option_planning_backups_per_step
            > audit.config.max_planning_uses_per_observation
        ):
            raise ValueError("STOMP planning budget exceeds the audit input ceiling")
        derived_semantic_digests = jnp.stack(
            tuple(
                option_semantic_digest(
                    {
                        "schema": "alberta.stomp-option-semantic.v1",
                        "slot": index,
                        "subtask": dataclasses.asdict(spec),
                    }
                )
                for index, spec in enumerate(stomp_agent.config.subtask_specs)
            )
        )
        if external_semantic_digests is None:
            self._semantic_digests = derived_semantic_digests
            self._external_semantic_digests = False
        else:
            semantics = _require_array(
                external_semantic_digests,
                name="external_semantic_digests",
                shape=(stomp_agent.config.n_options, _DIGEST_WORDS),
                dtype=jnp.uint32,
            )
            self._semantic_digests = semantics
            self._external_semantic_digests = True
        structure = dict(stomp_agent.to_config())
        structure.pop("subtask_specs")
        structure["n_options"] = stomp_agent.config.n_options
        self._structure_digest = _canonical_digest(structure)

    @property
    def config(self) -> STOMPOptionLifecycleConfig:
        return self._config

    @property
    def semantic_digests(self) -> Array:
        return self._semantic_digests

    def with_external_semantic_digests(
        self,
        semantic_digests: Array,
    ) -> STOMPOptionLifecycle:
        """Return a shape-identical wrapper bound to caller semantic identities."""

        return STOMPOptionLifecycle(
            self._agent,
            self._audit,
            self._config,
            external_semantic_digests=semantic_digests,
        )

    @property
    def stomp_agent(self) -> STOMPAgent:
        return self._agent

    @property
    def audit(self) -> OptionLifecycleAudit:
        return self._audit

    def to_config(self) -> dict[str, object]:
        config: dict[str, object] = {
            "schema_version": STOMP_OPTION_LIFECYCLE_CONFIG_SCHEMA,
            "composition": self._config.to_config(),
            "stomp": self._agent.to_config(),
            "audit": self._audit.to_config(),
        }
        if self._external_semantic_digests:
            config["external_semantic_digests"] = np.asarray(
                jax.device_get(self._semantic_digests)
            ).tolist()
        return config

    def _payload_arrays(self, state: STOMPOptionLifecycleState) -> tuple[Array, ...]:
        stomp_leaves = tuple(
            cast(Array, leaf)
            for leaf in jax.tree_util.tree_leaves(state.stomp_state)
        )
        audit_leaves = tuple(
            cast(Array, leaf)
            for leaf in jax.tree_util.tree_leaves(state.audit_state)
        )
        return (
            *stomp_leaves,
            *audit_leaves,
            state.lifecycle_id,
            state.stomp_structure_digest,
            state.started,
            state.revision,
            state.audit_unavailable,
            state.audit_error,
        )

    def _with_checksum(
        self,
        state: STOMPOptionLifecycleState,
    ) -> STOMPOptionLifecycleState:
        return dataclasses.replace(
            state,
            binding_checksum=_checksum_arrays(self._payload_arrays(state)),
        )

    def _metadata_payload_arrays(
        self,
        state: STOMPOptionLifecycleMetadataState,
    ) -> tuple[Array, ...]:
        audit_leaves = tuple(
            cast(Array, leaf) for leaf in jax.tree_util.tree_leaves(state.audit_state)
        )
        return (
            *audit_leaves,
            state.lifecycle_id,
            state.stomp_structure_digest,
            state.started,
            state.revision,
            state.audit_unavailable,
            state.audit_error,
            state.stomp_step_count,
            state.stomp_step_words,
            state.stomp_executing_option,
            state.stomp_binding_tag,
            state.stomp_binding_checksum,
            state.source_binding_checksum,
        )

    def _with_metadata_checksum(
        self,
        state: STOMPOptionLifecycleMetadataState,
    ) -> STOMPOptionLifecycleMetadataState:
        return dataclasses.replace(
            state,
            metadata_checksum=_checksum_arrays(self._metadata_payload_arrays(state)),
        )

    def _check_metadata_contract(
        self,
        state: STOMPOptionLifecycleMetadataState,
    ) -> None:
        if type(state) is not STOMPOptionLifecycleMetadataState:
            raise TypeError(
                "state must be an exact STOMPOptionLifecycleMetadataState"
            )
        _require_array(
            state.lifecycle_id,
            name="state.lifecycle_id",
            shape=(_LIFECYCLE_WORDS,),
            dtype=jnp.uint32,
        )
        _require_array(
            state.stomp_structure_digest,
            name="state.stomp_structure_digest",
            shape=(_DIGEST_WORDS,),
            dtype=jnp.uint32,
        )
        _require_array(state.started, name="state.started", shape=(), dtype=jnp.bool_)
        _require_array(state.revision, name="state.revision", shape=(), dtype=jnp.int32)
        _require_array(
            state.audit_unavailable,
            name="state.audit_unavailable",
            shape=(),
            dtype=jnp.bool_,
        )
        _require_array(
            state.audit_error,
            name="state.audit_error",
            shape=(),
            dtype=jnp.int32,
        )
        _require_array(
            state.stomp_step_count,
            name="state.stomp_step_count",
            shape=(),
            dtype=jnp.int32,
        )
        _require_array(
            state.stomp_step_words,
            name="state.stomp_step_words",
            shape=(_LIFECYCLE_WORDS,),
            dtype=jnp.uint32,
        )
        _require_array(
            state.stomp_executing_option,
            name="state.stomp_executing_option",
            shape=(),
            dtype=jnp.int32,
        )
        _require_array(
            state.stomp_binding_tag,
            name="state.stomp_binding_tag",
            shape=(_LIFECYCLE_WORDS,),
            dtype=jnp.uint32,
        )
        _require_array(
            state.stomp_binding_checksum,
            name="state.stomp_binding_checksum",
            shape=(_DIGEST_WORDS,),
            dtype=jnp.uint32,
        )
        _require_array(
            state.source_binding_checksum,
            name="state.source_binding_checksum",
            shape=(_LIFECYCLE_WORDS,),
            dtype=jnp.uint32,
        )
        _require_array(
            state.metadata_checksum,
            name="state.metadata_checksum",
            shape=(_LIFECYCLE_WORDS,),
            dtype=jnp.uint32,
        )

    def _metadata_audit_binding_valid(
        self,
        state: STOMPOptionLifecycleMetadataState,
    ) -> Array:
        audit_state = state.audit_state
        lifetime_aligned = audit_state.observation_count == state.stomp_step_count
        identity_aligned = (~audit_state.has_last_transition) | jnp.array_equal(
            audit_state.last_transition_id,
            state.stomp_step_words,
        )
        active_aligned = (audit_state.active_option < 0) | (
            audit_state.active_option == state.stomp_executing_option
        )
        pending_trial_aligned = (~audit_state.trial_active) | (
            (audit_state.active_option >= 0) | (state.stomp_executing_option < 0)
        )
        return (
            self._audit.state_valid(audit_state)
            & jnp.array_equal(audit_state.semantic_digests, self._semantic_digests)
            & lifetime_aligned
            & identity_aligned
            & active_aligned
            & pending_trial_aligned
        )

    def metadata_state_valid(
        self,
        state: STOMPOptionLifecycleMetadataState,
    ) -> Bool[Array, ""]:
        """Validate a detached sidecar without evaluating or owning STOMP."""

        self._check_metadata_contract(state)
        words_saturated = (state.stomp_step_words[0] != 0) | (
            state.stomp_step_words[1] > jnp.uint32(_INT32_MAX)
        )
        expected_step_count = jnp.where(
            words_saturated,
            jnp.int32(_INT32_MAX),
            state.stomp_step_words[1].astype(jnp.int32),
        )
        base = (
            jnp.any(state.lifecycle_id != 0)
            & jnp.array_equal(state.stomp_structure_digest, self._structure_digest)
            & (state.revision >= 0)
            & (state.revision <= _INT32_MAX)
            & (state.audit_error >= STOMP_OPTION_LIFECYCLE_AUDIT_ERROR_NONE)
            & (
                state.audit_error
                <= STOMP_OPTION_LIFECYCLE_AUDIT_ERROR_ATTRIBUTION_REJECTED
            )
            & (
                state.audit_unavailable
                == (state.audit_error != STOMP_OPTION_LIFECYCLE_AUDIT_ERROR_NONE)
            )
            & (state.stomp_step_count == expected_step_count)
            & (state.stomp_executing_option >= -1)
            & (state.stomp_executing_option < self._agent.config.n_options)
            & jnp.array_equal(state.stomp_binding_tag, _BORROWED_BINDING_TAG)
            & jnp.array_equal(
                state.metadata_checksum,
                _checksum_arrays(self._metadata_payload_arrays(state)),
            )
        )
        if self._config.audit_enabled:
            return base & (
                state.audit_unavailable | self._metadata_audit_binding_valid(state)
            )
        return base

    def detach_borrowed_stomp(
        self,
        state: STOMPOptionLifecycleState,
    ) -> STOMPOptionLifecycleMetadataState:
        """Detach all lifecycle metadata while retaining no STOMP owner.

        The returned checksum binds to every typed leaf of ``state.stomp_state``.
        Because the checksum is unkeyed, consumers must still require a trusted
        caller to identify the authoritative owner at a transition boundary.
        """

        self._check_state_contract(state)
        metadata = STOMPOptionLifecycleMetadataState(
            audit_state=state.audit_state,
            lifecycle_id=state.lifecycle_id,
            stomp_structure_digest=state.stomp_structure_digest,
            started=state.started,
            revision=state.revision,
            audit_unavailable=state.audit_unavailable,
            audit_error=state.audit_error,
            stomp_step_count=state.stomp_state.step_count,
            stomp_step_words=state.stomp_state.step_words,
            stomp_executing_option=state.stomp_state.executing_option,
            stomp_binding_tag=_BORROWED_BINDING_TAG,
            stomp_binding_checksum=_typed_tree_checksum(state.stomp_state),
            source_binding_checksum=state.binding_checksum,
            metadata_checksum=jnp.zeros((_LIFECYCLE_WORDS,), dtype=jnp.uint32),
        )
        return self._with_metadata_checksum(metadata)

    def attach_borrowed_stomp(
        self,
        metadata: STOMPOptionLifecycleMetadataState,
        stomp_state: STOMPState,
    ) -> STOMPOptionLifecycleBorrowResult:
        """Build a transient full lifecycle only for the exact borrowed owner."""

        self._check_metadata_contract(metadata)
        if type(stomp_state) is not STOMPState:
            raise TypeError("stomp_state must be an exact STOMPState")
        metadata_valid = self.metadata_state_valid(metadata)
        stomp_state_valid = self._agent.state_valid(stomp_state)
        binding_matches = (
            jnp.array_equal(metadata.stomp_binding_tag, _BORROWED_BINDING_TAG)
            & jnp.array_equal(
                metadata.stomp_binding_checksum,
                _typed_tree_checksum(stomp_state),
            )
            & (metadata.stomp_step_count == stomp_state.step_count)
            & jnp.array_equal(metadata.stomp_step_words, stomp_state.step_words)
            & (metadata.stomp_executing_option == stomp_state.executing_option)
        )
        candidate = STOMPOptionLifecycleState(
            stomp_state=stomp_state,
            audit_state=metadata.audit_state,
            lifecycle_id=metadata.lifecycle_id,
            stomp_structure_digest=metadata.stomp_structure_digest,
            started=metadata.started,
            revision=metadata.revision,
            audit_unavailable=metadata.audit_unavailable,
            audit_error=metadata.audit_error,
            binding_checksum=jnp.zeros((_LIFECYCLE_WORDS,), dtype=jnp.uint32),
        )
        candidate = self._with_checksum(candidate)
        binding_matches = binding_matches & jnp.array_equal(
            metadata.source_binding_checksum,
            candidate.binding_checksum,
        )
        transaction_applied = (
            metadata_valid
            & stomp_state_valid
            & binding_matches
            & self.state_valid(candidate)
        )
        return STOMPOptionLifecycleBorrowResult(
            state=candidate,
            metadata_valid=metadata_valid,
            stomp_state_valid=stomp_state_valid,
            binding_matches=binding_matches,
            transaction_applied=transaction_applied,
            caller_authenticated=jnp.asarray(False, dtype=jnp.bool_),
        )

    def _check_external_declaration_contract(
        self,
        declaration: STOMPOptionLifecycleExternalTransitionDeclaration,
    ) -> None:
        if type(declaration) is not STOMPOptionLifecycleExternalTransitionDeclaration:
            raise TypeError(
                "declaration must be an exact "
                "STOMPOptionLifecycleExternalTransitionDeclaration"
            )
        for name in ("source_stomp_checksum", "destination_stomp_checksum"):
            _require_array(
                getattr(declaration, name),
                name=f"declaration.{name}",
                shape=(_DIGEST_WORDS,),
                dtype=jnp.uint32,
            )
        for name in ("pre_step_words", "post_step_words"):
            _require_array(
                getattr(declaration, name),
                name=f"declaration.{name}",
                shape=(_LIFECYCLE_WORDS,),
                dtype=jnp.uint32,
            )
        for name in (
            "source_executing_option",
            "destination_executing_option",
            "primitive_action",
            "planning_backups",
        ):
            _require_array(
                getattr(declaration, name),
                name=f"declaration.{name}",
                shape=(),
                dtype=jnp.int32,
            )
        for name in (
            "option_terminated",
            "natural_completion",
            "censor_only_ending",
            "caller_derivation_declared",
        ):
            _require_array(
                getattr(declaration, name),
                name=f"declaration.{name}",
                shape=(),
                dtype=jnp.bool_,
            )
        for name in (
            "external_reward",
            "pseudo_reward",
            "average_reward",
            "td_error",
        ):
            _require_array(
                getattr(declaration, name),
                name=f"declaration.{name}",
                shape=(),
                dtype=jnp.float32,
            )
        _require_array(
            declaration.frozen_model_signature,
            name="declaration.frozen_model_signature",
            shape=(5 + self._agent.config.observation_dim,),
            dtype=jnp.float32,
        )
        _require_array(
            declaration.extended_action_mask,
            name="declaration.extended_action_mask",
            shape=(self._agent.config.n_total_actions,),
            dtype=jnp.bool_,
        )

    def declare_external_stomp_start(
        self,
        metadata: STOMPOptionLifecycleMetadataState,
        source_stomp_state: STOMPState,
        destination_stomp_state: STOMPState,
        *,
        caller_derivation_declared: bool | Array,
    ) -> STOMPOptionLifecycleExternalStartDeclaration:
        """Materialize an unkeyed trusted-caller declaration for STOMP start."""

        self._check_metadata_contract(metadata)
        if type(source_stomp_state) is not STOMPState:
            raise TypeError("source_stomp_state must be an exact STOMPState")
        if type(destination_stomp_state) is not STOMPState:
            raise TypeError("destination_stomp_state must be an exact STOMPState")
        declared = _bool_scalar(
            caller_derivation_declared,
            name="caller_derivation_declared",
        )
        return STOMPOptionLifecycleExternalStartDeclaration(
            source_stomp_checksum=_typed_tree_checksum(source_stomp_state),
            destination_stomp_checksum=_typed_tree_checksum(destination_stomp_state),
            primitive_action=destination_stomp_state.last_primitive_action,
            executing_option=destination_stomp_state.executing_option,
            caller_derivation_declared=declared,
        )

    def adopt_external_stomp_start(
        self,
        metadata: STOMPOptionLifecycleMetadataState,
        source_stomp_state: STOMPState,
        destination_stomp_state: STOMPState,
        initial_observation: Array,
        declaration: STOMPOptionLifecycleExternalStartDeclaration,
    ) -> STOMPOptionLifecycleExternalStartAdoptionResult:
        """Advance detached start metadata without evaluating STOMP again."""

        self._check_metadata_contract(metadata)
        if type(declaration) is not STOMPOptionLifecycleExternalStartDeclaration:
            raise TypeError(
                "declaration must be an exact "
                "STOMPOptionLifecycleExternalStartDeclaration"
            )
        if type(source_stomp_state) is not STOMPState:
            raise TypeError("source_stomp_state must be an exact STOMPState")
        if type(destination_stomp_state) is not STOMPState:
            raise TypeError("destination_stomp_state must be an exact STOMPState")
        for name in ("source_stomp_checksum", "destination_stomp_checksum"):
            _require_array(
                getattr(declaration, name),
                name=f"declaration.{name}",
                shape=(_DIGEST_WORDS,),
                dtype=jnp.uint32,
            )
        for name in ("primitive_action", "executing_option"):
            _require_array(
                getattr(declaration, name),
                name=f"declaration.{name}",
                shape=(),
                dtype=jnp.int32,
            )
        _require_array(
            declaration.caller_derivation_declared,
            name="declaration.caller_derivation_declared",
            shape=(),
            dtype=jnp.bool_,
        )
        observation = _require_array(
            initial_observation,
            name="initial_observation",
            shape=(self._agent.config.observation_dim,),
            dtype=jnp.float32,
        )
        borrowed = self.attach_borrowed_stomp(metadata, source_stomp_state)
        destination_state_valid = self._agent.state_valid(destination_stomp_state)
        clocks_preserved = (
            destination_stomp_state.step_count == source_stomp_state.step_count
        ) & jnp.array_equal(
            destination_stomp_state.step_words,
            source_stomp_state.step_words,
        ) & jnp.array_equal(
            destination_stomp_state.base_learner_state.step_words,
            source_stomp_state.base_learner_state.step_words,
        )
        endpoint_binding_valid = (
            jnp.array_equal(
                jax.lax.bitcast_convert_type(
                    destination_stomp_state.base_last_obs,
                    jnp.uint32,
                ),
                jax.lax.bitcast_convert_type(observation, jnp.uint32),
            )
            & (destination_stomp_state.last_primitive_action >= 0)
            & (
                destination_stomp_state.last_primitive_action
                < self._agent.config.n_primitive_actions
            )
            & (destination_stomp_state.executing_option >= -1)
            & (
                destination_stomp_state.executing_option
                < self._agent.config.n_options
            )
        )
        declaration_binding_valid = (
            declaration.caller_derivation_declared
            & jnp.array_equal(
                declaration.source_stomp_checksum,
                metadata.stomp_binding_checksum,
            )
            & jnp.array_equal(
                declaration.source_stomp_checksum,
                _typed_tree_checksum(source_stomp_state),
            )
            & jnp.array_equal(
                declaration.destination_stomp_checksum,
                _typed_tree_checksum(destination_stomp_state),
            )
            & (
                declaration.primitive_action
                == destination_stomp_state.last_primitive_action
            )
            & (declaration.executing_option == destination_stomp_state.executing_option)
        )
        full_candidate = STOMPOptionLifecycleState(
            stomp_state=destination_stomp_state,
            audit_state=metadata.audit_state,
            lifecycle_id=metadata.lifecycle_id,
            stomp_structure_digest=metadata.stomp_structure_digest,
            started=jnp.asarray(True, dtype=jnp.bool_),
            revision=_saturating_revision_increment(metadata.revision),
            audit_unavailable=metadata.audit_unavailable,
            audit_error=metadata.audit_error,
            binding_checksum=jnp.zeros((_LIFECYCLE_WORDS,), dtype=jnp.uint32),
        )
        full_candidate = self._with_checksum(full_candidate)
        candidate = self.detach_borrowed_stomp(full_candidate)
        metadata_advanced = (
            borrowed.transaction_applied
            & (~metadata.started)
            & destination_state_valid
            & clocks_preserved
            & endpoint_binding_valid
            & declaration_binding_valid
            & self.metadata_state_valid(candidate)
        )
        next_metadata = jax.lax.cond(
            metadata_advanced,
            lambda _: candidate,
            lambda _: metadata,
            None,
        )
        return STOMPOptionLifecycleExternalStartAdoptionResult(
            state=next_metadata,
            source_binding_matches=borrowed.binding_matches,
            destination_state_valid=destination_state_valid,
            clocks_preserved=clocks_preserved,
            endpoint_binding_valid=endpoint_binding_valid,
            declaration_binding_valid=declaration_binding_valid,
            metadata_advanced=metadata_advanced,
            derivation_recomputed=jnp.asarray(False, dtype=jnp.bool_),
            caller_authority_required=jnp.asarray(True, dtype=jnp.bool_),
            caller_authenticated=jnp.asarray(False, dtype=jnp.bool_),
        )

    def _external_transition_facts(
        self,
        metadata: STOMPOptionLifecycleMetadataState,
        source: STOMPState,
        next_observation: Array,
        discount: Array | None,
        execution_boundary: Array,
    ) -> tuple[Array, Array, Array, Array, Array, Array, Array, Array, Array]:
        cfg = self._agent.config
        executing = source.executing_option >= 0
        option_index = jnp.clip(source.executing_option, 0, cfg.n_options - 1)
        spec = self._agent.spec_arrays
        notional_pseudo = (
            spec.pseudo_reward_scales[option_index]
            * next_observation[spec.feature_indices[option_index]]
        )
        pseudo_reward = jnp.where(executing, notional_pseudo, jnp.float32(0.0))
        goal = executing & (notional_pseudo >= spec.thresholds[option_index])
        next_option_steps = _saturating_revision_increment(source.option_steps)
        timeout = executing & (
            next_option_steps >= spec.max_option_steps[option_index]
        )
        environment = (
            jnp.asarray(False, dtype=jnp.bool_)
            if discount is None
            else executing & (discount <= 0.0)
        )
        natural_completion = executing & (goal | timeout | environment)
        option_terminated = executing & (natural_completion | execution_boundary)
        censor_only_ending = executing & execution_boundary & (~natural_completion)
        model = source.option_models
        predicted_delta = model.next_state_weights[option_index] @ source.option_start_obs
        frozen_signature = jnp.concatenate(
            (
                jnp.stack(
                    (
                        model.env_return_ema[option_index],
                        model.cumreward_ema[option_index],
                        model.duration_ema[option_index],
                        model.baseline_mass_ema[option_index],
                        model.discount_ema[option_index],
                    )
                ),
                predicted_delta,
            )
        )
        starts_execution = executing & (metadata.audit_state.active_option < 0)
        frozen_signature = jnp.where(
            starts_execution,
            frozen_signature,
            jnp.zeros_like(frozen_signature),
        )
        return (
            option_index,
            pseudo_reward,
            goal,
            timeout,
            environment,
            natural_completion,
            option_terminated,
            censor_only_ending,
            frozen_signature,
        )

    def declare_external_stomp_transition(
        self,
        metadata: STOMPOptionLifecycleMetadataState,
        source_stomp_state: STOMPState,
        stomp_result: STOMPUpdateResult,
        *,
        env_reward: float | Array,
        next_observation: Array,
        discount: float | Array | None = None,
        execution_boundary: bool | Array = False,
        extended_action_mask: Array | None = None,
        caller_derivation_declared: bool | Array,
    ) -> STOMPOptionLifecycleExternalTransitionDeclaration:
        """Materialize the trusted caller's unkeyed transition declaration.

        This helper derives audit-facing facts from an already-evaluated result.
        It never invokes :meth:`STOMPAgent.update`; the caller remains
        responsible for asserting that the supplied result is authoritative.
        """

        self._check_metadata_contract(metadata)
        if type(source_stomp_state) is not STOMPState:
            raise TypeError("source_stomp_state must be an exact STOMPState")
        _validate_stomp_update_result_contract(stomp_result)
        reward = _float32_scalar(env_reward, name="env_reward")
        next_obs = _require_array(
            next_observation,
            name="next_observation",
            shape=(self._agent.config.observation_dim,),
            dtype=jnp.float32,
        )
        supplied_discount = (
            None if discount is None else _float32_scalar(discount, name="discount")
        )
        boundary = _bool_scalar(execution_boundary, name="execution_boundary")
        declared = _bool_scalar(
            caller_derivation_declared,
            name="caller_derivation_declared",
        )
        action_mask = self._effective_extended_action_mask(
            extended_action_mask
        )
        (
            _option_index,
            _pseudo_reward,
            _goal,
            _timeout,
            _environment,
            natural_completion,
            option_terminated,
            censor_only_ending,
            frozen_signature,
        ) = self._external_transition_facts(
            metadata,
            source_stomp_state,
            next_obs,
            supplied_discount,
            boundary,
        )
        del _option_index, _pseudo_reward, _goal, _timeout, _environment
        return STOMPOptionLifecycleExternalTransitionDeclaration(
            source_stomp_checksum=_typed_tree_checksum(source_stomp_state),
            destination_stomp_checksum=_typed_tree_checksum(stomp_result.state),
            pre_step_words=stomp_result.pre_step_words,
            post_step_words=stomp_result.post_step_words,
            source_executing_option=source_stomp_state.executing_option,
            destination_executing_option=stomp_result.executing_option,
            primitive_action=stomp_result.primitive_action,
            option_terminated=option_terminated,
            natural_completion=natural_completion,
            censor_only_ending=censor_only_ending,
            external_reward=reward,
            pseudo_reward=stomp_result.pseudo_reward,
            average_reward=stomp_result.average_reward,
            td_error=stomp_result.td_error,
            planning_backups=stomp_result.planning_backups,
            extended_action_mask=action_mask,
            frozen_model_signature=frozen_signature,
            caller_derivation_declared=declared,
        )

    def adopt_external_stomp_update(
        self,
        metadata: STOMPOptionLifecycleMetadataState,
        source_stomp_state: STOMPState,
        stomp_result: STOMPUpdateResult,
        declaration: STOMPOptionLifecycleExternalTransitionDeclaration,
        *,
        env_reward: float | Array,
        next_observation: Array,
        discount: float | Array | None = None,
        decision_observation: Array | None = None,
        execution_boundary: bool | Array = False,
        context: int | Array = 0,
        idle_candidate_option: int | Array = 0,
        idle_initiation_eligible: bool | Array = False,
        comparator_randomized: bool | Array = False,
        treatment_propensity: float | Array = 0.0,
        extended_action_mask: Array | None = None,
    ) -> STOMPOptionLifecycleExternalAdoptionResult:
        """Advance detached audit metadata from one authoritative STOMP result.

        STOMP is not evaluated here.  Exact clocks, endpoint diagnostics,
        termination/censor facts, rewards, and the frozen option-model signature
        are checked against the borrowed source and caller declaration.  Any
        mismatch leaves the complete metadata tree unchanged and never rolls
        back or otherwise mutates the caller-owned control destination.
        """

        self._check_metadata_contract(metadata)
        if type(source_stomp_state) is not STOMPState:
            raise TypeError("source_stomp_state must be an exact STOMPState")
        _validate_stomp_update_result_contract(stomp_result)
        self._check_external_declaration_contract(declaration)
        cfg = self._agent.config
        reward = _float32_scalar(env_reward, name="env_reward")
        next_obs = _require_array(
            next_observation,
            name="next_observation",
            shape=(cfg.observation_dim,),
            dtype=jnp.float32,
        )
        decision_obs = (
            next_obs
            if decision_observation is None
            else _require_array(
                decision_observation,
                name="decision_observation",
                shape=(cfg.observation_dim,),
                dtype=jnp.float32,
            )
        )
        supplied_discount = (
            None if discount is None else _float32_scalar(discount, name="discount")
        )
        boundary = _bool_scalar(execution_boundary, name="execution_boundary")
        audit_context = _int32_scalar(context, name="context")
        idle_candidate = _int32_scalar(
            idle_candidate_option,
            name="idle_candidate_option",
        )
        idle_eligible = _bool_scalar(
            idle_initiation_eligible,
            name="idle_initiation_eligible",
        )
        randomized = _bool_scalar(
            comparator_randomized,
            name="comparator_randomized",
        )
        propensity = _float32_scalar(
            treatment_propensity,
            name="treatment_propensity",
        )
        action_mask = self._effective_extended_action_mask(
            extended_action_mask
        )
        action_mask_valid = jnp.all(
            action_mask[: cfg.n_primitive_actions]
        ) & jnp.any(action_mask)

        borrowed = self.attach_borrowed_stomp(metadata, source_stomp_state)
        source_metadata_valid = borrowed.metadata_valid
        source_stomp_valid = borrowed.stomp_state_valid
        source_binding_matches = borrowed.binding_matches
        expected_words, outer_capacity = _increment_words(source_stomp_state.step_words)
        expected_nested_words, nested_capacity = _checked_lifetime_words_advance(
            source_stomp_state.base_learner_state.step_words,
            stomp_result.nested_updates_required,
        )
        expected_step_count = _saturating_revision_increment(
            source_stomp_state.step_count
        )
        result_clock_binding_valid = (
            outer_capacity
            & nested_capacity
            & jnp.array_equal(
                stomp_result.pre_step_words,
                source_stomp_state.step_words,
            )
            & jnp.array_equal(stomp_result.post_step_words, expected_words)
            & jnp.array_equal(stomp_result.state.step_words, expected_words)
            & (stomp_result.state.step_count == expected_step_count)
            & jnp.array_equal(
                stomp_result.state.base_learner_state.step_words,
                expected_nested_words,
            )
        )
        result_values_finite = (
            jnp.isfinite(stomp_result.td_error)
            & jnp.isfinite(stomp_result.average_reward)
            & jnp.isfinite(stomp_result.pseudo_reward)
            & jnp.isfinite(stomp_result.option_importance_ratio)
            & jnp.isfinite(stomp_result.planning_td_error)
        )
        source_executing = source_stomp_state.executing_option >= 0
        result_endpoint_binding_valid = (
            self._agent.state_valid(stomp_result.state)
            & result_values_finite
            & _float32_bits_equal(
                stomp_result.average_reward,
                stomp_result.state.base_average_reward,
            )
            & (stomp_result.primitive_action == stomp_result.state.last_primitive_action)
            & (stomp_result.executing_option == stomp_result.state.executing_option)
            & (stomp_result.primitive_action >= 0)
            & (stomp_result.primitive_action < cfg.n_primitive_actions)
            & (stomp_result.executing_option >= -1)
            & (stomp_result.executing_option < cfg.n_options)
            & jnp.array_equal(
                jax.lax.bitcast_convert_type(
                    stomp_result.state.base_last_obs,
                    jnp.uint32,
                ),
                jax.lax.bitcast_convert_type(decision_obs, jnp.uint32),
            )
        )
        inferred_real_updates = (
            stomp_result.nested_updates_applied - stomp_result.planning_backups
        )
        result_diagnostics_valid = (
            stomp_result.inputs_valid
            & stomp_result.lifetime_counter_valid
            & stomp_result.lifetime_capacity_available
            & stomp_result.nested_lifetime_counter_valid
            & stomp_result.nested_lifetime_capacity_available
            & stomp_result.proposed_state_valid
            & stomp_result.update_applied
            & (stomp_result.nested_updates_required >= 0)
            & (
                stomp_result.nested_updates_applied
                == stomp_result.nested_updates_required
            )
            & (stomp_result.planning_backups >= 0)
            & (
                stomp_result.planning_backups
                <= cfg.option_planning_backups_per_step
            )
            & (inferred_real_updates >= 0)
            & (inferred_real_updates <= 1)
            & (stomp_result.option_importance_ratio >= 0.0)
            & (stomp_result.option_importance_ratio <= cfg.option_importance_clip)
        )
        (
            option_index,
            expected_pseudo_reward,
            goal_terminated,
            timeout_terminated,
            environment_terminated,
            natural_completion,
            option_terminated,
            censor_only_ending,
            frozen_signature,
        ) = self._external_transition_facts(
            metadata,
            source_stomp_state,
            next_obs,
            supplied_discount,
            boundary,
        )
        continuing_owner_valid = (
            (~source_executing)
            | option_terminated
            | (stomp_result.executing_option == source_stomp_state.executing_option)
        )
        termination_binding_valid = (
            (stomp_result.option_terminated == option_terminated)
            & continuing_owner_valid
            & (declaration.option_terminated == option_terminated)
            & (declaration.natural_completion == natural_completion)
            & (declaration.censor_only_ending == censor_only_ending)
        )
        reward_binding_valid = (
            _float32_bits_equal(stomp_result.pseudo_reward, expected_pseudo_reward)
            & _float32_bits_equal(declaration.pseudo_reward, expected_pseudo_reward)
            & _float32_bits_equal(declaration.external_reward, reward)
            & jnp.isfinite(reward)
            & jnp.all(jnp.isfinite(next_obs))
            & jnp.all(jnp.isfinite(decision_obs))
        )
        model_signature_binding_valid = jnp.array_equal(
            jax.lax.bitcast_convert_type(
                declaration.frozen_model_signature,
                jnp.uint32,
            ),
            jax.lax.bitcast_convert_type(frozen_signature, jnp.uint32),
        )
        declaration_binding_valid = (
            declaration.caller_derivation_declared
            & jnp.array_equal(
                declaration.source_stomp_checksum,
                metadata.stomp_binding_checksum,
            )
            & jnp.array_equal(
                declaration.source_stomp_checksum,
                _typed_tree_checksum(source_stomp_state),
            )
            & jnp.array_equal(
                declaration.destination_stomp_checksum,
                _typed_tree_checksum(stomp_result.state),
            )
            & jnp.array_equal(
                declaration.pre_step_words,
                source_stomp_state.step_words,
            )
            & jnp.array_equal(declaration.post_step_words, expected_words)
            & (
                declaration.source_executing_option
                == source_stomp_state.executing_option
            )
            & (declaration.destination_executing_option == stomp_result.executing_option)
            & (declaration.primitive_action == stomp_result.primitive_action)
            & _float32_bits_equal(
                declaration.average_reward,
                stomp_result.average_reward,
            )
            & _float32_bits_equal(declaration.td_error, stomp_result.td_error)
            & (declaration.planning_backups == stomp_result.planning_backups)
            & jnp.array_equal(
                declaration.extended_action_mask,
                action_mask,
            )
        )
        control_binding_valid = (
            borrowed.transaction_applied
            & result_clock_binding_valid
            & result_endpoint_binding_valid
            & result_diagnostics_valid
            & termination_binding_valid
            & reward_binding_valid
            & model_signature_binding_valid
            & declaration_binding_valid
            & action_mask_valid
        )

        pre_audit = metadata.audit_state
        executing = source_stomp_state.executing_option >= 0
        audit_active = pre_audit.active_option >= 0
        candidate = jnp.where(executing, option_index, idle_candidate).astype(jnp.int32)
        owner = jnp.where(executing, option_index, -1).astype(jnp.int32)
        actual_context = jnp.where(
            executing & audit_active,
            pre_audit.active_context,
            audit_context,
        ).astype(jnp.int32)
        eligible = jnp.where(executing, ~audit_active, idle_eligible)
        arm = self._audit.arm(
            pre_audit,
            transition_id=expected_words,
            source_digest=pre_audit.source_digest,
            representation_digest=pre_audit.representation_digest,
            semantic_digests=pre_audit.semantic_digests,
            semantic_generations=pre_audit.semantic_generations,
            candidate_option=candidate,
            initiation_context=actual_context,
            initiation_eligible=eligible,
            owner_option=owner,
            comparator_randomized=randomized,
            treatment_propensity=propensity,
            frozen_model_prediction=frozen_signature,
        )
        planning_usage = self._planning_usage(
            stomp_result.state.option_models,
            source_stomp_state.step_words,
            stomp_result.planning_backups,
            action_mask,
        )
        step_discount = (
            jnp.asarray(cfg.option_gamma, dtype=jnp.float32)
            if supplied_discount is None
            else supplied_discount
        )
        memory_nbytes = self._option_slot_nbytes(source_stomp_state)
        if memory_nbytes > _INT32_MAX:
            raise ValueError("one STOMP option slot exceeds signed-int32 byte accounting")
        audit_result = self._audit.observe(
            pre_audit,
            arm,
            transition_id=expected_words,
            external_reward=reward,
            pseudo_reward=expected_pseudo_reward,
            baseline_mass=jnp.where(
                executing,
                source_stomp_state.option_discount,
                0.0,
            ).astype(jnp.float32),
            discount=step_discount,
            outcome_delta=next_obs - source_stomp_state.base_last_obs,
            goal_terminated=goal_terminated,
            timeout_terminated=timeout_terminated,
            environment_terminated=environment_terminated,
            censored=executing & boundary,
            planning_usage_delta=planning_usage,
            compute_cost=jnp.where(
                executing,
                jnp.asarray(
                    self._config.option_compute_units_per_step,
                    dtype=jnp.float32,
                ),
                jnp.float32(0.0),
            ),
            resident_memory_bytes=jnp.where(
                executing,
                jnp.asarray(memory_nbytes, dtype=jnp.int32),
                jnp.int32(0),
            ),
        )
        audit_capacity_available = (
            (pre_audit.observation_count < self._audit.config.max_observations)
            & (pre_audit.revision < _INT32_MAX)
        )
        audit_applied = (
            control_binding_valid
            & (~metadata.audit_unavailable)
            & self._metadata_audit_binding_valid(metadata)
            & audit_capacity_available
            & audit_result.applied
            & jnp.array_equal(audit_result.state.last_transition_id, expected_words)
        )
        selected_audit = jax.lax.cond(
            audit_applied,
            lambda _: audit_result.state,
            lambda _: pre_audit,
            None,
        )
        full_candidate = STOMPOptionLifecycleState(
            stomp_state=stomp_result.state,
            audit_state=selected_audit,
            lifecycle_id=metadata.lifecycle_id,
            stomp_structure_digest=metadata.stomp_structure_digest,
            started=metadata.started,
            revision=_saturating_revision_increment(metadata.revision),
            audit_unavailable=metadata.audit_unavailable,
            audit_error=metadata.audit_error,
            binding_checksum=jnp.zeros((_LIFECYCLE_WORDS,), dtype=jnp.uint32),
        )
        full_candidate = self._with_checksum(full_candidate)
        candidate_metadata = self.detach_borrowed_stomp(full_candidate)
        if self._config.audit_enabled:
            metadata_advanced = audit_applied & self.metadata_state_valid(
                candidate_metadata
            )
        else:
            metadata_advanced = control_binding_valid & self.metadata_state_valid(
                candidate_metadata
            )
        next_metadata = jax.lax.cond(
            metadata_advanced,
            lambda _: candidate_metadata,
            lambda _: metadata,
            None,
        )
        return STOMPOptionLifecycleExternalAdoptionResult(
            state=next_metadata,
            source_metadata_valid=source_metadata_valid,
            source_stomp_valid=source_stomp_valid,
            source_binding_matches=source_binding_matches,
            result_static_contract_valid=jnp.asarray(True, dtype=jnp.bool_),
            result_clock_binding_valid=result_clock_binding_valid,
            result_endpoint_binding_valid=(
                result_endpoint_binding_valid & result_diagnostics_valid
            ),
            termination_binding_valid=termination_binding_valid,
            reward_binding_valid=reward_binding_valid,
            model_signature_binding_valid=model_signature_binding_valid,
            declaration_binding_valid=declaration_binding_valid,
            audit_applied=audit_applied,
            metadata_advanced=metadata_advanced,
            control_transition_rolled_back=jnp.asarray(False, dtype=jnp.bool_),
            derivation_recomputed=jnp.asarray(False, dtype=jnp.bool_),
            caller_authority_required=jnp.asarray(True, dtype=jnp.bool_),
            caller_authenticated=jnp.asarray(False, dtype=jnp.bool_),
            transaction_applied=metadata_advanced,
        )

    def finalize_external_stomp_owner(
        self,
        metadata_after_raw_audit: STOMPOptionLifecycleMetadataState,
        trace: STOMPOwnerFinalizationTrace,
    ) -> STOMPOptionLifecycleExternalOwnerFinalizationResult:
        """Bind the sole final owner after auditing the raw real-STOMP result.

        The raw external adoption remains the only lifecycle/audit transition.
        This method evaluates no learner and advances no lifecycle counter.  It
        accepts only the five fixed, class-specific post-control stages in the
        supplied transient trace, preserves the already-committed audit tree,
        and changes only the borrowed-owner binding to the trace's final state.
        Digests are integrity checks, not caller authentication.
        """

        self._check_metadata_contract(metadata_after_raw_audit)
        if type(trace) is not STOMPOwnerFinalizationTrace:
            raise TypeError("trace must be an exact STOMPOwnerFinalizationTrace")
        raw_borrowed = self.attach_borrowed_stomp(
            metadata_after_raw_audit,
            trace.raw_state,
        )
        trace_valid = stomp_owner_finalization_trace_valid(trace)
        final_owner_state_valid = self._agent.state_valid(trace.final_state)
        lifecycle_identity_preserved = (
            (trace.final_state.step_count == trace.raw_state.step_count)
            & jnp.array_equal(
                trace.final_state.step_words,
                trace.raw_state.step_words,
            )
            & (
                trace.final_state.executing_option
                == trace.raw_state.executing_option
            )
        )
        full_candidate = raw_borrowed.state.replace(
            stomp_state=trace.final_state,
            binding_checksum=jnp.zeros(
                (_LIFECYCLE_WORDS,),
                dtype=jnp.uint32,
            ),
        )
        full_candidate = self._with_checksum(full_candidate)
        candidate_metadata = self.detach_borrowed_stomp(full_candidate)
        audit_state_preserved = _trees_exactly_equal(
            candidate_metadata.audit_state,
            metadata_after_raw_audit.audit_state,
        )
        metadata_identity_preserved = (
            jnp.array_equal(
                candidate_metadata.lifecycle_id,
                metadata_after_raw_audit.lifecycle_id,
            )
            & jnp.array_equal(
                candidate_metadata.stomp_structure_digest,
                metadata_after_raw_audit.stomp_structure_digest,
            )
            & (candidate_metadata.started == metadata_after_raw_audit.started)
            & (candidate_metadata.revision == metadata_after_raw_audit.revision)
            & (
                candidate_metadata.audit_unavailable
                == metadata_after_raw_audit.audit_unavailable
            )
            & (candidate_metadata.audit_error == metadata_after_raw_audit.audit_error)
        )
        metadata_finalized = (
            raw_borrowed.transaction_applied
            & trace_valid
            & final_owner_state_valid
            & lifecycle_identity_preserved
            & audit_state_preserved
            & metadata_identity_preserved
            & self.metadata_state_valid(candidate_metadata)
        )
        next_metadata = jax.lax.cond(
            metadata_finalized,
            lambda _: candidate_metadata,
            lambda _: metadata_after_raw_audit,
            None,
        )
        return STOMPOptionLifecycleExternalOwnerFinalizationResult(
            state=next_metadata,
            raw_metadata_valid=raw_borrowed.metadata_valid,
            raw_owner_binding_matches=raw_borrowed.binding_matches,
            final_owner_state_valid=final_owner_state_valid,
            stage_trace_valid=trace_valid,
            audit_state_preserved=audit_state_preserved,
            lifecycle_identity_preserved=(
                lifecycle_identity_preserved & metadata_identity_preserved
            ),
            metadata_finalized=metadata_finalized,
            derivation_recomputed=jnp.asarray(False, dtype=jnp.bool_),
            caller_authenticated=jnp.asarray(False, dtype=jnp.bool_),
        )

    def _check_state_contract(self, state: STOMPOptionLifecycleState) -> None:
        if type(state) is not STOMPOptionLifecycleState:
            raise TypeError("state must be an exact STOMPOptionLifecycleState")
        _require_array(
            state.lifecycle_id,
            name="state.lifecycle_id",
            shape=(_LIFECYCLE_WORDS,),
            dtype=jnp.uint32,
        )
        _require_array(
            state.stomp_structure_digest,
            name="state.stomp_structure_digest",
            shape=(_DIGEST_WORDS,),
            dtype=jnp.uint32,
        )
        _require_array(state.started, name="state.started", shape=(), dtype=jnp.bool_)
        _require_array(state.revision, name="state.revision", shape=(), dtype=jnp.int32)
        _require_array(
            state.audit_unavailable,
            name="state.audit_unavailable",
            shape=(),
            dtype=jnp.bool_,
        )
        _require_array(
            state.audit_error,
            name="state.audit_error",
            shape=(),
            dtype=jnp.int32,
        )
        _require_array(
            state.binding_checksum,
            name="state.binding_checksum",
            shape=(_LIFECYCLE_WORDS,),
            dtype=jnp.uint32,
        )

    def _base_binding_valid(self, state: STOMPOptionLifecycleState) -> Array:
        return (
            jnp.any(state.lifecycle_id != 0)
            & jnp.array_equal(state.stomp_structure_digest, self._structure_digest)
            & (state.revision >= 0)
            & (state.revision <= _INT32_MAX)
            & (state.audit_error >= STOMP_OPTION_LIFECYCLE_AUDIT_ERROR_NONE)
            & (
                state.audit_error
                <= STOMP_OPTION_LIFECYCLE_AUDIT_ERROR_ATTRIBUTION_REJECTED
            )
            & (
                state.audit_unavailable
                == (state.audit_error != STOMP_OPTION_LIFECYCLE_AUDIT_ERROR_NONE)
            )
            & self._agent.state_valid(state.stomp_state)
            & jnp.array_equal(
                state.binding_checksum,
                _checksum_arrays(self._payload_arrays(state)),
            )
        )

    def _audit_binding_valid(self, state: STOMPOptionLifecycleState) -> Array:
        audit_state = state.audit_state
        lifetime_aligned = audit_state.observation_count == state.stomp_state.step_count
        identity_aligned = (~audit_state.has_last_transition) | jnp.array_equal(
            audit_state.last_transition_id,
            state.stomp_state.step_words,
        )
        active_aligned = (audit_state.active_option < 0) | (
            audit_state.active_option == state.stomp_state.executing_option
        )
        pending_trial_aligned = (~audit_state.trial_active) | (
            (audit_state.active_option >= 0)
            | (state.stomp_state.executing_option < 0)
        )
        return (
            self._audit.state_valid(audit_state)
            & jnp.array_equal(audit_state.semantic_digests, self._semantic_digests)
            & lifetime_aligned
            & identity_aligned
            & active_aligned
            & pending_trial_aligned
        )

    def state_valid(self, state: STOMPOptionLifecycleState) -> Bool[Array, ""]:
        """Return exact structural, STOMP, audit, and lifetime validity."""

        self._check_state_contract(state)
        base = self._base_binding_valid(state)
        if self._config.audit_enabled:
            return base & (state.audit_unavailable | self._audit_binding_valid(state))
        return base

    def init(
        self,
        key: Array,
        *,
        source_digest: Array,
        representation_digest: Array,
        lifecycle_id: Array,
        semantic_generations: Array | None = None,
    ) -> STOMPOptionLifecycleState:
        """Initialize real STOMP and its automatically derived semantic sidecar."""

        key_data = _require_array(
            jr.key_data(key),
            name="key data",
            shape=(2,),
            dtype=jnp.uint32,
        )
        del key_data
        lifecycle = _require_array(
            lifecycle_id,
            name="lifecycle_id",
            shape=(_LIFECYCLE_WORDS,),
            dtype=jnp.uint32,
        )
        if not bool(jax.device_get(jnp.any(lifecycle != 0))):
            raise ValueError("lifecycle_id must be nonzero")
        stomp_state = self._agent.init(key)
        audit_state = self._audit.init(
            source_digest=source_digest,
            representation_digest=representation_digest,
            semantic_digests=self._semantic_digests,
            semantic_generations=semantic_generations,
        )
        state = STOMPOptionLifecycleState(
            stomp_state=stomp_state,
            audit_state=audit_state,
            lifecycle_id=lifecycle,
            stomp_structure_digest=self._structure_digest,
            started=jnp.asarray(False, dtype=jnp.bool_),
            revision=jnp.asarray(0, dtype=jnp.int32),
            audit_unavailable=jnp.asarray(False, dtype=jnp.bool_),
            audit_error=jnp.asarray(
                STOMP_OPTION_LIFECYCLE_AUDIT_ERROR_NONE,
                dtype=jnp.int32,
            ),
            binding_checksum=jnp.zeros((_LIFECYCLE_WORDS,), dtype=jnp.uint32),
        )
        return self._with_checksum(state)

    def start(
        self,
        state: STOMPOptionLifecycleState,
        initial_observation: Array,
    ) -> STOMPOptionLifecycleStartResult:
        """Prime the real STOMP policy without yet inventing an audit outcome."""

        self._check_state_contract(state)
        observation = _require_array(
            initial_observation,
            name="initial_observation",
            shape=(self._agent.config.observation_dim,),
            dtype=jnp.float32,
        )
        primed_stomp = self._agent.start(state.stomp_state, observation)
        proposed = STOMPOptionLifecycleState(
            stomp_state=primed_stomp,
            audit_state=state.audit_state,
            lifecycle_id=state.lifecycle_id,
            stomp_structure_digest=state.stomp_structure_digest,
            started=jnp.asarray(True, dtype=jnp.bool_),
            revision=_saturating_revision_increment(state.revision),
            audit_unavailable=state.audit_unavailable,
            audit_error=state.audit_error,
            binding_checksum=jnp.zeros((_LIFECYCLE_WORDS,), dtype=jnp.uint32),
        )
        proposed = self._with_checksum(proposed)
        applied = (
            self.state_valid(state)
            & (~state.started)
            & self._agent.state_valid(primed_stomp)
            & self.state_valid(proposed)
        )
        next_state = jax.lax.cond(applied, lambda _: proposed, lambda _: state, None)
        primitive = jnp.where(
            applied,
            primed_stomp.last_primitive_action,
            state.stomp_state.last_primitive_action,
        )
        return STOMPOptionLifecycleStartResult(
            state=next_state,
            primitive_action=primitive,
            applied=applied,
            audit_enabled=jnp.asarray(self._config.audit_enabled, dtype=jnp.bool_),
        )

    def start_with_extended_action_mask(
        self,
        state: STOMPOptionLifecycleState,
        initial_observation: Array,
        extended_action_mask: Array,
    ) -> STOMPOptionLifecycleStartResult:
        """Prime STOMP while keeping masked option slots behavior-ineligible."""

        self._check_state_contract(state)
        observation = _require_array(
            initial_observation,
            name="initial_observation",
            shape=(self._agent.config.observation_dim,),
            dtype=jnp.float32,
        )
        mask = _require_array(
            extended_action_mask,
            name="extended_action_mask",
            shape=(self._agent.config.n_total_actions,),
            dtype=jnp.bool_,
        )
        mask_valid = jnp.all(mask[: self._agent.config.n_primitive_actions]) & jnp.any(mask)
        raw = self._agent.start_with_extended_action_mask(
            state.stomp_state,
            observation,
            mask,
        )
        proposed = STOMPOptionLifecycleState(
            stomp_state=raw.state,
            audit_state=state.audit_state,
            lifecycle_id=state.lifecycle_id,
            stomp_structure_digest=state.stomp_structure_digest,
            started=jnp.asarray(True, dtype=jnp.bool_),
            revision=_saturating_revision_increment(state.revision),
            audit_unavailable=state.audit_unavailable,
            audit_error=state.audit_error,
            binding_checksum=jnp.zeros((_LIFECYCLE_WORDS,), dtype=jnp.uint32),
        )
        proposed = self._with_checksum(proposed)
        applied = (
            self.state_valid(state)
            & (~state.started)
            & mask_valid
            & self._agent.state_valid(raw.state)
            & self.state_valid(proposed)
        )
        next_state = jax.lax.cond(applied, lambda _: proposed, lambda _: state, None)
        return STOMPOptionLifecycleStartResult(
            state=next_state,
            primitive_action=jnp.where(
                applied,
                raw.primitive_action,
                state.stomp_state.last_primitive_action,
            ),
            applied=applied,
            audit_enabled=jnp.asarray(self._config.audit_enabled, dtype=jnp.bool_),
        )

    @staticmethod
    def _tree_nbytes(value: Any) -> int:
        return sum(
            int(np.prod(np.shape(leaf), dtype=np.int64)) * int(np.dtype(leaf.dtype).itemsize)
            for leaf in jax.tree_util.tree_leaves(value)
            if hasattr(leaf, "dtype")
        )

    def _option_slot_nbytes(self, state: STOMPState) -> int:
        option_index = 0
        head_index = self._agent.config.n_primitive_actions
        policy = state.option_policies
        models = state.option_models
        learner = state.base_learner_state
        values: tuple[Any, ...] = (
            policy.q_weights[option_index],
            policy.traces[option_index],
            policy.average_rewards[option_index],
            models.cumreward_ema[option_index],
            models.env_return_ema[option_index],
            models.duration_ema[option_index],
            models.baseline_mass_ema[option_index],
            models.discount_ema[option_index],
            models.next_state_weights[option_index],
            models.n_completions[option_index],
            learner.head_params.weights[head_index],
            learner.head_params.biases[head_index],
            learner.head_optimizer_states[head_index],
            learner.head_traces[head_index],
        )
        return self._tree_nbytes(values)

    def _effective_extended_action_mask(
        self,
        extended_action_mask: Array | None,
    ) -> Array:
        cfg = self._agent.config
        if extended_action_mask is None:
            return jnp.ones((cfg.n_total_actions,), dtype=jnp.bool_)
        mask = _require_array(
            extended_action_mask,
            name="extended_action_mask",
            shape=(cfg.n_total_actions,),
            dtype=jnp.bool_,
        )
        return mask

    def _planning_usage(
        self,
        models: Any,
        selection_words: Array,
        applied_backups: Array,
        extended_action_mask: Array | None = None,
    ) -> Array:
        cfg = self._agent.config
        action_mask = self._effective_extended_action_mask(
            extended_action_mask
        )
        completed = (models.n_completions > 0) & action_mask[
            cfg.n_primitive_actions :
        ]
        n_completed = jnp.sum(completed.astype(jnp.int32))
        safe_count = jnp.maximum(n_completed, 1).astype(jnp.uint32)
        completed_indices = jnp.nonzero(
            completed,
            size=cfg.n_options,
            fill_value=0,
        )[0]
        # Compute 2**32 mod the dynamic completion count without enabling x64.
        two_to_32_mod_completed = jax.lax.fori_loop(
            0,
            32,
            lambda _index, remainder: jnp.mod(remainder * jnp.uint32(2), safe_count),
            jnp.mod(jnp.uint32(1), safe_count),
        )
        phase = jnp.mod(
            jnp.mod(selection_words[0], safe_count) * two_to_32_mod_completed
            + jnp.mod(selection_words[1], safe_count),
            safe_count,
        ).astype(jnp.int32)

        def body(index: int, counts: Array) -> Array:
            rank = jnp.mod(phase + index, jnp.maximum(n_completed, 1))
            option = completed_indices[rank]
            # A malformed external trace must never attribute backups to the
            # ``nonzero(..., fill_value=0)`` sentinel when no live completed
            # option exists.  Normal STOMP results already satisfy this, but
            # the borrowed-owner adoption seam remains independently safe.
            use = (index < applied_backups) & (n_completed > 0)
            return counts.at[option].add(use.astype(jnp.int32))

        return cast(
            Array,
            jax.lax.fori_loop(
                0,
                cfg.option_planning_backups_per_step,
                body,
                jnp.zeros((cfg.n_options,), dtype=jnp.int32),
            ),
        )

    def update(
        self,
        state: STOMPOptionLifecycleState,
        env_reward: float | Array,
        next_observation: Array,
        discount: float | Array | None = None,
        *,
        decision_observation: Array | None = None,
        execution_boundary: bool | Array = False,
        context: int | Array = 0,
        idle_candidate_option: int | Array = 0,
        idle_initiation_eligible: bool | Array = False,
        comparator_randomized: bool | Array = False,
        treatment_propensity: float | Array = 0.0,
        extended_action_mask: Array | None = None,
        enable_planning: bool = True,
    ) -> STOMPOptionLifecycleUpdateResult:
        """Apply one STOMP transition and observe it without control authority."""

        self._check_state_contract(state)
        cfg = self._agent.config
        reward = _float32_scalar(env_reward, name="env_reward")
        next_obs = _require_array(
            next_observation,
            name="next_observation",
            shape=(cfg.observation_dim,),
            dtype=jnp.float32,
        )
        decision_obs = (
            next_obs
            if decision_observation is None
            else _require_array(
                decision_observation,
                name="decision_observation",
                shape=(cfg.observation_dim,),
                dtype=jnp.float32,
            )
        )
        boundary = _bool_scalar(execution_boundary, name="execution_boundary")
        supplied_discount = (
            None if discount is None else _float32_scalar(discount, name="discount")
        )
        expected_words, word_capacity = _increment_words(state.stomp_state.step_words)
        raw = self._agent.update(
            state.stomp_state,
            reward,
            next_obs,
            supplied_discount,
            decision_observation=decision_obs,
            execution_boundary=boundary,
            extended_action_mask=extended_action_mask,
            enable_planning=enable_planning,
        )
        transaction_identity = jnp.concatenate((state.lifecycle_id, expected_words))
        persistent_state_valid = self.state_valid(state)
        wrapper_valid = persistent_state_valid & state.started & word_capacity
        stomp_identity_matches = (
            jnp.array_equal(raw.pre_step_words, state.stomp_state.step_words)
            & jnp.array_equal(raw.post_step_words, expected_words)
        )
        stomp_applied = (
            wrapper_valid
            & raw.update_applied
            & stomp_identity_matches
            & self._agent.state_valid(raw.state)
        )

        if not self._config.audit_enabled:
            proposed = STOMPOptionLifecycleState(
                stomp_state=raw.state,
                audit_state=state.audit_state,
                lifecycle_id=state.lifecycle_id,
                stomp_structure_digest=state.stomp_structure_digest,
                started=state.started,
                revision=_saturating_revision_increment(state.revision),
                audit_unavailable=state.audit_unavailable,
                audit_error=state.audit_error,
                binding_checksum=jnp.zeros((_LIFECYCLE_WORDS,), dtype=jnp.uint32),
            )
            proposed = self._with_checksum(proposed)
            applied = stomp_applied & self.state_valid(proposed)
            next_state = jax.lax.cond(applied, lambda _: proposed, lambda _: state, None)
            planning_usage = self._planning_usage(
                raw.state.option_models,
                state.stomp_state.step_words,
                raw.planning_backups,
                extended_action_mask,
            )
            return STOMPOptionLifecycleUpdateResult(
                state=next_state,
                td_error=jnp.where(applied, raw.td_error, 0.0),
                average_reward=jnp.where(
                    applied,
                    raw.average_reward,
                    state.stomp_state.base_average_reward,
                ),
                primitive_action=jnp.where(
                    applied,
                    raw.primitive_action,
                    state.stomp_state.last_primitive_action,
                ),
                executing_option=jnp.where(
                    applied,
                    raw.executing_option,
                    state.stomp_state.executing_option,
                ),
                option_terminated=applied & raw.option_terminated,
                natural_completion=jnp.asarray(False, dtype=jnp.bool_),
                censor_only_ending=jnp.asarray(False, dtype=jnp.bool_),
                pseudo_reward=jnp.where(applied, raw.pseudo_reward, 0.0),
                planning_usage=jnp.where(
                    applied,
                    planning_usage,
                    jnp.zeros_like(planning_usage),
                ),
                pre_step_words=state.stomp_state.step_words,
                post_step_words=jnp.where(
                    applied,
                    raw.post_step_words,
                    state.stomp_state.step_words,
                ),
                transaction_identity=transaction_identity,
                stomp_update_applied=raw.update_applied,
                audit_applied=jnp.asarray(False, dtype=jnp.bool_),
                audit_sidecar_accepted=jnp.asarray(False, dtype=jnp.bool_),
                audit_capacity_available=jnp.asarray(False, dtype=jnp.bool_),
                audit_unavailable_noop=jnp.asarray(False, dtype=jnp.bool_),
                audit_error=jnp.asarray(
                    STOMP_OPTION_LIFECYCLE_AUDIT_ERROR_NONE,
                    dtype=jnp.int32,
                ),
                transaction_applied=applied,
                rolled_back=raw.update_applied & (~applied),
                audit_enabled=jnp.asarray(False, dtype=jnp.bool_),
            )

        audit_context = _int32_scalar(context, name="context")
        idle_candidate = _int32_scalar(
            idle_candidate_option,
            name="idle_candidate_option",
        )
        idle_eligible = _bool_scalar(
            idle_initiation_eligible,
            name="idle_initiation_eligible",
        )
        randomized = _bool_scalar(
            comparator_randomized,
            name="comparator_randomized",
        )
        propensity = _float32_scalar(
            treatment_propensity,
            name="treatment_propensity",
        )
        pre_stomp = state.stomp_state
        pre_audit = state.audit_state
        audit_source_valid = self._audit_binding_valid(state)
        audit_capacity_available = (
            (pre_audit.observation_count < self._audit.config.max_observations)
            & (pre_audit.revision < _INT32_MAX)
        )
        executing = pre_stomp.executing_option >= 0
        option_index = jnp.clip(pre_stomp.executing_option, 0, cfg.n_options - 1)
        audit_active = pre_audit.active_option >= 0
        candidate = jnp.where(executing, option_index, idle_candidate).astype(jnp.int32)
        owner = jnp.where(executing, option_index, -1).astype(jnp.int32)
        actual_context = jnp.where(
            executing & audit_active,
            pre_audit.active_context,
            audit_context,
        ).astype(jnp.int32)
        eligible = jnp.where(
            executing,
            ~audit_active,
            idle_eligible,
        )
        starts_execution = executing & (~audit_active)
        model = pre_stomp.option_models
        predicted_delta = model.next_state_weights[option_index] @ pre_stomp.option_start_obs
        frozen_signature = jnp.concatenate(
            (
                jnp.stack(
                    (
                        model.env_return_ema[option_index],
                        model.cumreward_ema[option_index],
                        model.duration_ema[option_index],
                        model.baseline_mass_ema[option_index],
                        model.discount_ema[option_index],
                    )
                ),
                predicted_delta,
            )
        )
        frozen_signature = jnp.where(
            starts_execution,
            frozen_signature,
            jnp.zeros_like(frozen_signature),
        )
        arm = self._audit.arm(
            pre_audit,
            transition_id=expected_words,
            source_digest=pre_audit.source_digest,
            representation_digest=pre_audit.representation_digest,
            semantic_digests=pre_audit.semantic_digests,
            semantic_generations=pre_audit.semantic_generations,
            candidate_option=candidate,
            initiation_context=actual_context,
            initiation_eligible=eligible,
            owner_option=owner,
            comparator_randomized=randomized,
            treatment_propensity=propensity,
            frozen_model_prediction=frozen_signature,
        )

        step_discount = (
            jnp.asarray(cfg.option_gamma, dtype=jnp.float32)
            if supplied_discount is None
            else supplied_discount
        )
        pseudo = raw.pseudo_reward
        spec = self._agent.spec_arrays
        goal = executing & (pseudo >= spec.thresholds[option_index])
        next_option_steps = pre_stomp.option_steps + jnp.int32(1)
        timeout = executing & (
            next_option_steps >= spec.max_option_steps[option_index]
        )
        environment = (
            jnp.asarray(False, dtype=jnp.bool_)
            if supplied_discount is None
            else executing & (supplied_discount <= 0.0)
        )
        is_censored = executing & boundary
        planning_usage = self._planning_usage(
            raw.state.option_models,
            pre_stomp.step_words,
            raw.planning_backups,
            extended_action_mask,
        )
        memory_nbytes = self._option_slot_nbytes(pre_stomp)
        if memory_nbytes > _INT32_MAX:
            raise ValueError("one STOMP option slot exceeds signed-int32 byte accounting")
        audit_result = self._audit.observe(
            pre_audit,
            arm,
            transition_id=expected_words,
            external_reward=reward,
            pseudo_reward=pseudo,
            baseline_mass=jnp.where(executing, pre_stomp.option_discount, 0.0).astype(
                jnp.float32
            ),
            discount=step_discount,
            outcome_delta=next_obs - pre_stomp.base_last_obs,
            goal_terminated=goal,
            timeout_terminated=timeout,
            environment_terminated=environment,
            censored=is_censored,
            planning_usage_delta=planning_usage,
            compute_cost=jnp.where(
                executing,
                jnp.asarray(
                    self._config.option_compute_units_per_step,
                    dtype=jnp.float32,
                ),
                jnp.float32(0.0),
            ),
            resident_memory_bytes=jnp.where(
                executing,
                jnp.asarray(memory_nbytes, dtype=jnp.int32),
                jnp.int32(0),
            ),
        )
        audit_candidate = STOMPOptionLifecycleState(
            stomp_state=raw.state,
            audit_state=audit_result.state,
            lifecycle_id=state.lifecycle_id,
            stomp_structure_digest=state.stomp_structure_digest,
            started=state.started,
            revision=_saturating_revision_increment(state.revision),
            audit_unavailable=jnp.asarray(False, dtype=jnp.bool_),
            audit_error=jnp.asarray(
                STOMP_OPTION_LIFECYCLE_AUDIT_ERROR_NONE,
                dtype=jnp.int32,
            ),
            binding_checksum=jnp.zeros((_LIFECYCLE_WORDS,), dtype=jnp.uint32),
        )
        audit_identity_matches = jnp.array_equal(
            audit_result.state.last_transition_id,
            expected_words,
        )
        audit_applied = (
            stomp_applied
            & (~state.audit_unavailable)
            & audit_source_valid
            & audit_capacity_available
            & audit_result.applied
            & audit_identity_matches
            & self._audit_binding_valid(audit_candidate)
        )
        audit_reaches_capacity = (
            audit_result.state.observation_count
            >= self._audit.config.max_observations
        ) | (audit_result.state.revision >= _INT32_MAX)
        audit_failed = stomp_applied & (~audit_applied)
        failure_error = jnp.where(
            state.audit_unavailable,
            state.audit_error,
            jnp.where(
                ~audit_capacity_available,
                jnp.int32(STOMP_OPTION_LIFECYCLE_AUDIT_ERROR_CAPACITY),
                jnp.where(
                    ~audit_source_valid,
                    jnp.int32(
                        STOMP_OPTION_LIFECYCLE_AUDIT_ERROR_PERSISTENT_STATE_INVALID
                    ),
                    jnp.int32(
                        STOMP_OPTION_LIFECYCLE_AUDIT_ERROR_ATTRIBUTION_REJECTED
                    ),
                ),
            ),
        )
        next_audit_unavailable = (
            state.audit_unavailable
            | (audit_applied & audit_reaches_capacity)
            | audit_failed
        )
        next_audit_error = jnp.where(
            state.audit_unavailable,
            state.audit_error,
            jnp.where(
                audit_applied & audit_reaches_capacity,
                jnp.int32(STOMP_OPTION_LIFECYCLE_AUDIT_ERROR_CAPACITY),
                jnp.where(
                    audit_failed,
                    failure_error,
                    jnp.int32(STOMP_OPTION_LIFECYCLE_AUDIT_ERROR_NONE),
                ),
            ),
        )
        next_audit = jax.lax.cond(
            audit_applied,
            lambda _: audit_result.state,
            lambda _: pre_audit,
            None,
        )
        proposed = STOMPOptionLifecycleState(
            stomp_state=raw.state,
            audit_state=next_audit,
            lifecycle_id=state.lifecycle_id,
            stomp_structure_digest=state.stomp_structure_digest,
            started=state.started,
            revision=_saturating_revision_increment(state.revision),
            audit_unavailable=next_audit_unavailable,
            audit_error=next_audit_error,
            binding_checksum=jnp.zeros((_LIFECYCLE_WORDS,), dtype=jnp.uint32),
        )
        proposed = self._with_checksum(proposed)
        next_state = jax.lax.cond(
            stomp_applied,
            lambda _: proposed,
            lambda _: state,
            None,
        )
        natural = executing & (goal | timeout | environment)
        censor_only = is_censored & (~natural)
        reported_audit_error = jnp.where(
            stomp_applied,
            next_audit_error,
            jnp.where(
                ~persistent_state_valid,
                jnp.int32(
                    STOMP_OPTION_LIFECYCLE_AUDIT_ERROR_PERSISTENT_STATE_INVALID
                ),
                jnp.int32(STOMP_OPTION_LIFECYCLE_AUDIT_ERROR_NONE),
            ),
        )
        return STOMPOptionLifecycleUpdateResult(
            state=next_state,
            td_error=jnp.where(stomp_applied, raw.td_error, 0.0),
            average_reward=jnp.where(
                stomp_applied,
                raw.average_reward,
                pre_stomp.base_average_reward,
            ),
            primitive_action=jnp.where(
                stomp_applied,
                raw.primitive_action,
                pre_stomp.last_primitive_action,
            ),
            executing_option=jnp.where(
                stomp_applied,
                raw.executing_option,
                pre_stomp.executing_option,
            ),
            option_terminated=stomp_applied & raw.option_terminated,
            natural_completion=stomp_applied & natural,
            censor_only_ending=stomp_applied & censor_only,
            pseudo_reward=jnp.where(stomp_applied, pseudo, 0.0),
            planning_usage=jnp.where(
                stomp_applied,
                planning_usage,
                jnp.zeros_like(planning_usage),
            ),
            pre_step_words=pre_stomp.step_words,
            post_step_words=jnp.where(
                stomp_applied,
                raw.post_step_words,
                pre_stomp.step_words,
            ),
            transaction_identity=transaction_identity,
            stomp_update_applied=raw.update_applied,
            audit_applied=audit_applied,
            audit_sidecar_accepted=audit_applied,
            audit_capacity_available=audit_capacity_available,
            audit_unavailable_noop=stomp_applied & (~audit_applied),
            audit_error=reported_audit_error,
            transaction_applied=stomp_applied,
            rolled_back=raw.update_applied & (~stomp_applied),
            audit_enabled=jnp.asarray(True, dtype=jnp.bool_),
        )

    @staticmethod
    def _merge_tree(reset: Array, fresh: Any, old: Any) -> Any:
        return jax.tree_util.tree_map(
            lambda fresh_leaf, old_leaf: jnp.where(reset, fresh_leaf, old_leaf),
            fresh,
            old,
        )

    def _merge_rebound_stomp(
        self,
        old: STOMPState,
        fresh: STOMPState,
        reset_slots: Array,
    ) -> STOMPState:
        mask1 = reset_slots
        mask3 = reset_slots[:, None, None]
        policies = old.option_policies.replace(
            q_weights=jnp.where(
                mask3,
                fresh.option_policies.q_weights,
                old.option_policies.q_weights,
            ),
            traces=jnp.where(mask3, fresh.option_policies.traces, old.option_policies.traces),
            average_rewards=jnp.where(
                mask1,
                fresh.option_policies.average_rewards,
                old.option_policies.average_rewards,
            ),
        )
        mask_models = reset_slots[:, None, None]
        models = old.option_models.replace(
            cumreward_ema=jnp.where(
                mask1,
                fresh.option_models.cumreward_ema,
                old.option_models.cumreward_ema,
            ),
            env_return_ema=jnp.where(
                mask1,
                fresh.option_models.env_return_ema,
                old.option_models.env_return_ema,
            ),
            duration_ema=jnp.where(
                mask1,
                fresh.option_models.duration_ema,
                old.option_models.duration_ema,
            ),
            baseline_mass_ema=jnp.where(
                mask1,
                fresh.option_models.baseline_mass_ema,
                old.option_models.baseline_mass_ema,
            ),
            discount_ema=jnp.where(
                mask1,
                fresh.option_models.discount_ema,
                old.option_models.discount_ema,
            ),
            next_state_weights=jnp.where(
                mask_models,
                fresh.option_models.next_state_weights,
                old.option_models.next_state_weights,
            ),
            n_completions=jnp.where(
                mask1,
                fresh.option_models.n_completions,
                old.option_models.n_completions,
            ),
        )
        learner = old.base_learner_state
        fresh_learner = fresh.base_learner_state
        n_primitive = self._agent.config.n_primitive_actions
        merged_weights: list[Array] = []
        merged_biases: list[Array] = []
        merged_optimizer_states: list[Any] = []
        merged_traces: list[Any] = []
        for head in range(self._agent.config.n_total_actions):
            if head < n_primitive:
                reset = jnp.asarray(False, dtype=jnp.bool_)
            else:
                reset = reset_slots[head - n_primitive]
            merged_weights.append(
                jnp.where(
                    reset,
                    fresh_learner.head_params.weights[head],
                    learner.head_params.weights[head],
                )
            )
            merged_biases.append(
                jnp.where(
                    reset,
                    fresh_learner.head_params.biases[head],
                    learner.head_params.biases[head],
                )
            )
            merged_optimizer_states.append(
                self._merge_tree(
                    reset,
                    fresh_learner.head_optimizer_states[head],
                    learner.head_optimizer_states[head],
                )
            )
            merged_traces.append(
                self._merge_tree(
                    reset,
                    fresh_learner.head_traces[head],
                    learner.head_traces[head],
                )
            )
        learner = learner.replace(
            head_params=learner.head_params.replace(
                weights=tuple(merged_weights),
                biases=tuple(merged_biases),
            ),
            head_optimizer_states=tuple(merged_optimizer_states),
            head_traces=tuple(merged_traces),
        )
        return cast(
            STOMPState,
            old.replace(
                base_learner_state=learner,
                option_policies=policies,
                option_models=models,
            ),
        )

    def rebind(
        self,
        state: STOMPOptionLifecycleState,
        fresh_key: Array,
        *,
        source_digest: Array,
        representation_digest: Array,
    ) -> STOMPOptionLifecycleRebindResult:
        """Transfer identical slots and fully reset explicitly changed semantics.

        Call this method on the wrapper carrying the new STOMP subtask specs.
        All non-subtask STOMP configuration and all audit dimensions/config
        must remain shape-compatible.  ``fresh_key`` initializes only reset
        option-local state; the live STOMP policy RNG and global learning state
        are preserved.
        """

        self._check_state_contract(state)
        _require_array(
            jr.key_data(fresh_key),
            name="fresh key data",
            shape=(2,),
            dtype=jnp.uint32,
        )
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
        old_audit_valid = self._audit.state_valid(state.audit_state)
        source_state_valid = (
            self._base_binding_valid(state)
            & old_audit_valid
            & (state.stomp_state.executing_option < 0)
        )
        semantic_changed = jnp.any(
            state.audit_state.semantic_digests != self._semantic_digests,
            axis=1,
        )
        global_changed = (
            ~jnp.array_equal(source, state.audit_state.source_digest)
            | ~jnp.array_equal(
                representation,
                state.audit_state.representation_digest,
            )
        )
        change_requested = global_changed | jnp.any(semantic_changed)
        in_flight = (
            (state.stomp_state.executing_option >= 0)
            | (state.audit_state.active_option >= 0)
            | state.audit_state.trial_active
        )
        audit_rebind = self._audit.rebind(
            state.audit_state,
            source_digest=source,
            representation_digest=representation,
            semantic_digests=self._semantic_digests,
        )
        fresh = self._agent.init(fresh_key)
        reset_slots = audit_rebind.reset_slots
        rebound_stomp = self._merge_rebound_stomp(
            state.stomp_state,
            fresh,
            reset_slots,
        )
        proposed = STOMPOptionLifecycleState(
            stomp_state=rebound_stomp,
            audit_state=audit_rebind.state,
            lifecycle_id=state.lifecycle_id,
            stomp_structure_digest=self._structure_digest,
            started=state.started,
            revision=_saturating_revision_increment(state.revision),
            audit_unavailable=state.audit_unavailable,
            audit_error=state.audit_error,
            binding_checksum=jnp.zeros((_LIFECYCLE_WORDS,), dtype=jnp.uint32),
        )
        proposed = self._with_checksum(proposed)
        transaction_valid = (
            self._base_binding_valid(state)
            & old_audit_valid
            & jnp.array_equal(state.stomp_structure_digest, self._structure_digest)
            & self._agent.state_valid(fresh)
        )
        applied = (
            transaction_valid
            & change_requested
            & (~in_flight)
            & source_state_valid
            & audit_rebind.applied
            & self.state_valid(proposed)
        )
        deferred = transaction_valid & change_requested & in_flight
        next_state = jax.lax.cond(applied, lambda _: proposed, lambda _: state, None)
        return STOMPOptionLifecycleRebindResult(
            state=next_state,
            transaction_valid=transaction_valid,
            applied=applied,
            deferred=deferred,
            preserved_slots=transaction_valid & audit_rebind.preserved_slots,
            reset_slots=applied & audit_rebind.reset_slots,
        )

    @staticmethod
    def _cryptographic_state_digest(state: STOMPOptionLifecycleState) -> Array:
        digest = hashlib.sha256()
        for leaf in jax.tree_util.tree_leaves(state):
            array = jnp.asarray(leaf)
            if jax.dtypes.issubdtype(array.dtype, jax.dtypes.prng_key):
                array = jr.key_data(array)
            host = np.asarray(jax.device_get(array))
            digest.update(host.dtype.str.encode("ascii"))
            digest.update(np.asarray(host.shape, dtype=np.int64).tobytes())
            digest.update(host.tobytes(order="C"))
        return jnp.asarray(tuple(digest.digest()), dtype=jnp.uint8)

    def checkpoint_payload(
        self,
        state: STOMPOptionLifecycleState,
    ) -> dict[str, object]:
        """Return a strict exact-state checkpoint, including mid-option state."""

        if not bool(jax.device_get(self.state_valid(state))):
            raise ValueError("cannot checkpoint an invalid STOMP lifecycle composition")
        return {
            "schema_version": STOMP_OPTION_LIFECYCLE_CHECKPOINT_SCHEMA,
            "config": self.to_config(),
            "state": state,
            "state_digest": self._cryptographic_state_digest(state),
        }

    def restore_checkpoint(
        self,
        payload: object,
        *,
        expected_source_digest: Array,
        expected_representation_digest: Array,
        expected_lifecycle_id: Array,
    ) -> STOMPOptionLifecycleState:
        """Restore only an exact v1 checkpoint under the exact live binding."""

        if type(payload) is not dict:
            raise ValueError("STOMP lifecycle checkpoint must be an exact dict")
        raw = cast(dict[object, object], payload)
        if set(raw) != {"schema_version", "config", "state", "state_digest"}:
            raise ValueError("STOMP lifecycle checkpoint keys differ from schema v1")
        if raw["schema_version"] != STOMP_OPTION_LIFECYCLE_CHECKPOINT_SCHEMA:
            raise ValueError("STOMP lifecycle checkpoint schema_version differs")
        if raw["config"] != self.to_config():
            raise ValueError("STOMP lifecycle checkpoint config differs")
        restored = raw["state"]
        if type(restored) is not STOMPOptionLifecycleState:
            raise ValueError("STOMP lifecycle checkpoint state type differs")
        state = restored
        persisted = _require_array(
            raw["state_digest"],
            name="checkpoint.state_digest",
            shape=(32,),
            dtype=jnp.uint8,
        )
        if not bool(
            jax.device_get(
                jnp.array_equal(persisted, self._cryptographic_state_digest(state))
            )
        ):
            raise ValueError("STOMP lifecycle checkpoint state digest differs")
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
        lifecycle = _require_array(
            expected_lifecycle_id,
            name="expected_lifecycle_id",
            shape=(_LIFECYCLE_WORDS,),
            dtype=jnp.uint32,
        )
        valid = (
            jnp.array_equal(state.audit_state.source_digest, source)
            & jnp.array_equal(state.audit_state.representation_digest, representation)
            & jnp.array_equal(state.lifecycle_id, lifecycle)
            & jnp.array_equal(state.audit_state.semantic_digests, self._semantic_digests)
            & self.state_valid(state)
        )
        if not bool(jax.device_get(valid)):
            raise ValueError("STOMP lifecycle checkpoint is invalid, stale, or rebound")
        return state

    def resource_budget(
        self,
        state: STOMPOptionLifecycleState,
    ) -> STOMPOptionLifecycleResourceBudget:
        """Return exact wrapped persistent bytes and additional wrapper work."""

        self._check_state_contract(state)
        stomp_nbytes = measure_stomp_state_nbytes(state.stomp_state)
        audit_nbytes = self._audit.resource_budget.persistent_state_nbytes
        binding_nbytes = (
            state.lifecycle_id.size * state.lifecycle_id.dtype.itemsize
            + state.stomp_structure_digest.size
            * state.stomp_structure_digest.dtype.itemsize
            + state.started.size * state.started.dtype.itemsize
            + state.revision.size * state.revision.dtype.itemsize
            + state.audit_unavailable.size * state.audit_unavailable.dtype.itemsize
            + state.audit_error.size * state.audit_error.dtype.itemsize
            + state.binding_checksum.size * state.binding_checksum.dtype.itemsize
        )
        binding_nbytes = int(binding_nbytes)
        return STOMPOptionLifecycleResourceBudget(
            wrapped_persistent_state_nbytes=stomp_nbytes + audit_nbytes + binding_nbytes,
            stomp_persistent_state_nbytes=stomp_nbytes,
            audit_persistent_state_nbytes=audit_nbytes,
            composition_binding_nbytes=binding_nbytes,
            option_slot_nbytes=self._option_slot_nbytes(state.stomp_state),
            stomp_updates_per_update=1,
            audit_arms_per_enabled_update=int(self._config.audit_enabled),
            audit_observations_per_enabled_update=int(self._config.audit_enabled),
            additional_rng_draws_per_start=0,
            additional_rng_draws_per_update=0,
            additional_backward_passes_per_update=0,
            additional_consumer_calls_per_update=0,
            max_planning_attribution_slots_per_update=(
                self._agent.config.option_planning_backups_per_step
            ),
            semantic_slots_examined_per_rebind=self._agent.config.n_options,
            max_audited_observations=self._audit.config.max_observations,
            stomp_lifetime_identity_bits=64,
            wrapper_revision_saturation=_INT32_MAX,
            audit_capacity_can_block_stomp=False,
            curation_authority=False,
            promotion_authority=False,
            dispatch_authority=False,
            go_no_go_authority=False,
            scientific_promotion_allowed=False,
            checkpoint_schema=STOMP_OPTION_LIFECYCLE_CHECKPOINT_SCHEMA,
        )


__all__ = [
    "STOMP_OPTION_LIFECYCLE_AUDIT_ERROR_ATTRIBUTION_REJECTED",
    "STOMP_OPTION_LIFECYCLE_AUDIT_ERROR_CAPACITY",
    "STOMP_OPTION_LIFECYCLE_AUDIT_ERROR_NONE",
    "STOMP_OPTION_LIFECYCLE_AUDIT_ERROR_PERSISTENT_STATE_INVALID",
    "STOMP_OPTION_LIFECYCLE_BORROWED_BINDING_SCHEMA",
    "STOMP_OPTION_LIFECYCLE_CHECKPOINT_SCHEMA",
    "STOMP_OPTION_LIFECYCLE_CONFIG_SCHEMA",
    "STOMP_OPTION_LIFECYCLE_CURATION_AUTHORITY",
    "STOMP_OPTION_LIFECYCLE_DISPATCH_AUTHORITY",
    "STOMP_OPTION_LIFECYCLE_GO_NO_GO_AUTHORITY",
    "STOMP_OPTION_LIFECYCLE_PROMOTION_AUTHORITY",
    "STOMP_OPTION_LIFECYCLE_SCIENTIFIC_PROMOTION_ALLOWED",
    "STOMPOptionLifecycle",
    "STOMPOptionLifecycleBorrowResult",
    "STOMPOptionLifecycleConfig",
    "STOMPOptionLifecycleExternalAdoptionResult",
    "STOMPOptionLifecycleExternalOwnerFinalizationResult",
    "STOMPOptionLifecycleExternalStartAdoptionResult",
    "STOMPOptionLifecycleExternalStartDeclaration",
    "STOMPOptionLifecycleExternalTransitionDeclaration",
    "STOMPOptionLifecycleMetadataState",
    "STOMPOptionLifecycleRebindResult",
    "STOMPOptionLifecycleResourceBudget",
    "STOMPOptionLifecycleStartResult",
    "STOMPOptionLifecycleState",
    "STOMPOptionLifecycleUpdateResult",
]
