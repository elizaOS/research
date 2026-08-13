# mypy: disable-error-code="arg-type,attr-defined,call-arg,type-var"
"""Single-owner bridge from Prototype control to external option authority.

The persistent bridge owns one :class:`PrototypeAgentState`.  Its nested
Prototype→OaK→STOMP path is the sole live controller.  Scheduling, installation,
lifecycle audit, retirement, and replacement persist only detached metadata
whose typed checksums borrow that STOMP owner; none stores a second
``STOMPState``.

Ordinary control is evaluated exactly once by Prototype.  The exact transient
``STOMPUpdateResult`` returned by that OaK evaluation is consumed by the
lifecycle audit without recomputing STOMP.  Audit refusal never rolls back a
valid Prototype transition: the bridge records a terminal desynchronization
and blocks later authority transactions.  Checksums and host checkpoint hashes
are integrity mechanisms, not caller authentication.

This L0 mechanism makes no benefit, evidence, promotion, safety, go/no-go,
retirement, discovery, dispatch, or autonomous-curation claim.
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
from typing import Any, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array
from jaxtyping import Bool, Int, UInt

from alberta_framework.core.authorized_option_replacement import (
    AuthorizedOptionReplacementArm,
    AuthorizedOptionReplacementController,
    AuthorizedOptionReplacementMetadataState,
    AuthorizedOptionReplacementPrepared,
    AuthorizedOptionReplacementResult,
    AuthorizedOptionReplacementRetirementResult,
    AuthorizedOptionReplacementState,
    OptionReplacementAuthorityReceipt,
)
from alberta_framework.core.authorized_option_retirement import (
    OptionRetirementAuthorityReceipt,
)
from alberta_framework.core.cumulant_option_installation import (
    CumulantOptionInstallationState,
    CumulantOptionLiveInputs,
)
from alberta_framework.core.cumulant_option_scheduler import (
    CumulantOptionRetirementHandoff,
    CumulantOptionSchedulerObservation,
)
from alberta_framework.core.oak import OaKOptionSlotRebindResult, OaKState
from alberta_framework.core.options import STOMPState
from alberta_framework.core.prototype_agent import (
    PrototypeAgent,
    PrototypeAgentState,
    PrototypeCandidateUpdateAuditEvidence,
    PrototypeExperientialMemoryInput,
    PrototypeFeatureOaKHordeState,
    PrototypeFeatureOaKHordeUtilityCurationState,
    PrototypeFeatureOaKHordeUtilityState,
    PrototypeFeatureOaKState,
    PrototypeGradientJoyEvidence,
    PrototypePartnerPolicyFusionFeedback,
    PrototypePartnerPolicyFusionInput,
    PrototypeTransition,
    PrototypeUpdateResult,
)
from alberta_framework.core.stomp_option_lifecycle import (
    STOMPOptionLifecycleExternalAdoptionResult,
    STOMPOptionLifecycleExternalOwnerFinalizationResult,
    STOMPOptionLifecycleExternalStartAdoptionResult,
)
from alberta_framework.core.stomp_owner_finalization import (
    stomp_typed_tree_digest,
)

PROTOTYPE_OPTION_AUTHORITY_BRIDGE_STATE_SCHEMA = (
    "alberta.prototype-option-authority-bridge.state.v1"
)
PROTOTYPE_OPTION_AUTHORITY_BRIDGE_CHECKPOINT_SCHEMA = (
    "alberta.prototype-option-authority-bridge.checkpoint.v1"
)
PROTOTYPE_OPTION_AUTHORITY_BRIDGE_INITIAL_OWNER_BINDING_SCHEMA = (
    "alberta.prototype-option-authority-bridge.initial-owner-binding.v1"
)
PROTOTYPE_OPTION_AUTHORITY_BRIDGE_ASSESSMENT = "not_assessed"
PROTOTYPE_OPTION_AUTHORITY_BRIDGE_ERROR_NONE = 0
PROTOTYPE_OPTION_AUTHORITY_BRIDGE_ERROR_DESYNCHRONIZED = 1
PROTOTYPE_OPTION_AUTHORITY_BRIDGE_BENEFIT_CLAIM = False
PROTOTYPE_OPTION_AUTHORITY_BRIDGE_EVIDENCE_AUTHORITY = False
PROTOTYPE_OPTION_AUTHORITY_BRIDGE_PROMOTION_AUTHORITY = False
PROTOTYPE_OPTION_AUTHORITY_BRIDGE_SAFETY_AUTHORITY = False
PROTOTYPE_OPTION_AUTHORITY_BRIDGE_GO_NO_GO_AUTHORITY = False
PROTOTYPE_OPTION_AUTHORITY_BRIDGE_RETIREMENT_AUTHORITY = False
PROTOTYPE_OPTION_AUTHORITY_BRIDGE_DISCOVERY_AUTHORITY = False
PROTOTYPE_OPTION_AUTHORITY_BRIDGE_DISPATCH_AUTHORITY = False
PROTOTYPE_OPTION_AUTHORITY_BRIDGE_AUTONOMOUS_CURATION_AUTHORITY = False
PROTOTYPE_OPTION_AUTHORITY_BRIDGE_SCIENTIFIC_PROMOTION_ALLOWED = False

_INT32_MAX = 2**31 - 1
_INITIAL_OWNER_BINDING_TAG = jnp.asarray(
    (0x504F4131, 0x00000001),
    dtype=jnp.uint32,
)


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
        elif array.dtype == jnp.int32:
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


def _tree_exact_equal(left: object, right: object) -> Array:
    left_leaves, left_tree = jax.tree_util.tree_flatten(left)
    right_leaves, right_tree = jax.tree_util.tree_flatten(right)
    if cast(object, left_tree) != cast(object, right_tree):
        return jnp.asarray(False, dtype=jnp.bool_)
    if len(left_leaves) != len(right_leaves):
        return jnp.asarray(False, dtype=jnp.bool_)
    equal = jnp.asarray(True, dtype=jnp.bool_)
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        left_array = jnp.asarray(left_leaf)
        right_array = jnp.asarray(right_leaf)
        if left_array.shape != right_array.shape or left_array.dtype != right_array.dtype:
            return jnp.asarray(False, dtype=jnp.bool_)
        if jax.dtypes.issubdtype(left_array.dtype, jax.dtypes.prng_key):
            leaf_equal = jnp.array_equal(
                jr.key_data(left_array),
                jr.key_data(right_array),
            )
        elif left_array.dtype == jnp.float32:
            leaf_equal = jnp.array_equal(
                jax.lax.bitcast_convert_type(left_array, jnp.uint32),
                jax.lax.bitcast_convert_type(right_array, jnp.uint32),
            )
        else:
            leaf_equal = jnp.array_equal(left_array, right_array)
        equal = equal & leaf_equal
    return equal


def _increment_words(words: Array) -> tuple[Array, Array]:
    low = words[1] + jnp.uint32(1)
    carry = (low == 0).astype(jnp.uint32)
    high = words[0] + carry
    available = ~((carry != 0) & (high == 0))
    candidate = jnp.stack((high, low), dtype=jnp.uint32)
    return jnp.where(available, candidate, words), available


def _saturating_increment(value: Array) -> Array:
    return jnp.where(value < _INT32_MAX, value + jnp.int32(1), value)


def _tree_nbytes(value: object) -> int:
    total = 0
    for leaf in jax.tree_util.tree_leaves(value):
        array = jnp.asarray(leaf)
        if jax.dtypes.issubdtype(array.dtype, jax.dtypes.prng_key):
            array = jr.key_data(array)
        total += int(array.size * array.dtype.itemsize)
    return total


def _prototype_oak_state(slot: object) -> OaKState:
    if type(slot) is OaKState:
        return slot
    if type(slot) is PrototypeFeatureOaKState:
        return slot.oak_state
    if type(slot) is PrototypeFeatureOaKHordeState:
        return slot.oak_state
    if type(slot) is PrototypeFeatureOaKHordeUtilityState:
        return slot.oak_state
    if type(slot) is PrototypeFeatureOaKHordeUtilityCurationState:
        return slot.oak_state
    raise TypeError("Prototype oak_state has an unsupported exact wrapper type")


def _replace_prototype_oak_slot(slot: object, oak_state: OaKState) -> object:
    if type(slot) is OaKState:
        return oak_state
    if type(slot) is PrototypeFeatureOaKState:
        return slot.replace(oak_state=oak_state)
    if type(slot) is PrototypeFeatureOaKHordeState:
        return slot.replace(oak_state=oak_state)
    if type(slot) is PrototypeFeatureOaKHordeUtilityState:
        utility = slot
        return utility.replace(
            consumer_state=utility.consumer_state.replace(oak_state=oak_state)
        )
    if type(slot) is PrototypeFeatureOaKHordeUtilityCurationState:
        curation = slot
        utility = curation.utility_state
        return curation.replace(
            utility_state=utility.replace(
                consumer_state=utility.consumer_state.replace(oak_state=oak_state)
            )
        )
    raise TypeError("Prototype oak_state has an unsupported exact wrapper type")


def _replace_prototype_oak_state(
    state: PrototypeAgentState,
    oak_state: OaKState,
) -> PrototypeAgentState:
    return cast(
        PrototypeAgentState,
        state.replace(
            oak_state=_replace_prototype_oak_slot(state.oak_state, oak_state)
        ),
    )


def _normalize_audit_int32(
    value: int | Array,
    *,
    name: str,
    lower: int,
    upper: int,
) -> tuple[Array, Array]:
    """Normalize a scalar audit identity while preserving dynamic refusal."""

    if type(value) is int:
        representable = -(2**31) <= value <= _INT32_MAX
        candidate = value if representable else 0
        array = jnp.asarray(candidate, dtype=jnp.int32)
        static_valid = representable and lower <= value < upper
        return array, jnp.asarray(static_valid, dtype=jnp.bool_)
    if not hasattr(value, "shape") or not hasattr(value, "dtype"):
        raise TypeError(f"{name} must be a Python int or int32 scalar array")
    array = jnp.asarray(value)
    if array.shape != ():
        raise ValueError(f"{name} must have scalar shape")
    if array.dtype != jnp.int32:
        raise TypeError(f"{name} must have dtype int32")
    dynamic_valid = (array >= lower) & (array < upper)
    return (
        jnp.where(dynamic_valid, array, jnp.asarray(0, dtype=jnp.int32)),
        dynamic_valid,
    )


def _normalize_audit_bool(
    value: bool | Array,
    *,
    name: str,
) -> tuple[Array, Array]:
    """Normalize one exact audit boolean; static malformation still raises."""

    if type(value) is bool:
        return (
            jnp.asarray(value, dtype=jnp.bool_),
            jnp.asarray(True, dtype=jnp.bool_),
        )
    if not hasattr(value, "shape") or not hasattr(value, "dtype"):
        raise TypeError(f"{name} must be a Python bool or bool scalar array")
    array = jnp.asarray(value)
    if array.shape != ():
        raise ValueError(f"{name} must have scalar shape")
    if array.dtype != jnp.bool_:
        raise TypeError(f"{name} must have dtype bool")
    return array, jnp.asarray(True, dtype=jnp.bool_)


def _normalize_audit_propensity(
    value: float | Array,
) -> tuple[Array, Array]:
    """Normalize propensity; non-finite/out-of-range values refuse only audit."""

    if type(value) is float:
        representable = math.isfinite(value) and abs(value) <= np.finfo(np.float32).max
        candidate = value if representable else 0.0
        array = jnp.asarray(candidate, dtype=jnp.float32)
        static_valid = representable and 0.0 <= value <= 1.0
        return array, jnp.asarray(static_valid, dtype=jnp.bool_)
    if not hasattr(value, "shape") or not hasattr(value, "dtype"):
        raise TypeError(
            "treatment_propensity must be a Python float or float32 scalar array"
        )
    array = jnp.asarray(value)
    if array.shape != ():
        raise ValueError("treatment_propensity must have scalar shape")
    if array.dtype != jnp.float32:
        raise TypeError("treatment_propensity must have dtype float32")
    dynamic_valid = jnp.isfinite(array) & (array >= 0.0) & (array <= 1.0)
    return (
        jnp.where(
            dynamic_valid,
            array,
            jnp.asarray(0.0, dtype=jnp.float32),
        ),
        dynamic_valid,
    )


@chex.dataclass(frozen=True)
class PrototypeOptionAuthorityBridgeInitialOwnerBindingReceipt:
    """Source-bound caller authorization for canonical authority adoption.

    The full source states make replay/staleness checks exact.  The checksum is
    unkeyed integrity only; ``caller_authenticated`` is permanently false.
    """

    source_prototype_state: PrototypeAgentState
    source_authority_state: AuthorizedOptionReplacementState
    prototype_owner_digest: UInt[Array, " 8"]
    authority_owner_digest: UInt[Array, " 8"]
    binding_authorized: Bool[Array, ""]
    receipt_checksum: UInt[Array, " 2"]
    caller_authenticated: Bool[Array, ""]


@chex.dataclass(frozen=True)
class PrototypeOptionAuthorityBridgeInitialOwnerBindingResult:
    """Fail-closed authority-to-pristine-Prototype initial binding result."""

    prototype_state: PrototypeAgentState
    source_prototype_matches: Bool[Array, ""]
    source_authority_matches: Bool[Array, ""]
    receipt_integrity_valid: Bool[Array, ""]
    prototype_pristine: Bool[Array, ""]
    authority_quiescent: Bool[Array, ""]
    canonical_owner_valid: Bool[Array, ""]
    canonical_owner_adopted: Bool[Array, ""]
    caller_authority_required: Bool[Array, ""]
    caller_authenticated: Bool[Array, ""]
    transaction_applied: Bool[Array, ""]


@chex.dataclass(frozen=True)
class PrototypeOptionAuthorityBridgeState:
    """One Prototype owner plus detached option-authority metadata."""

    prototype_state: PrototypeAgentState
    authority_metadata: AuthorizedOptionReplacementMetadataState
    extended_action_mask: Bool[Array, " n_total_actions"]
    authority_synchronized: Bool[Array, ""]
    authority_error: Int[Array, ""]
    revision: Int[Array, ""]
    binding_checksum: UInt[Array, " 2"]


@chex.dataclass(frozen=True)
class PrototypeOptionAuthorityBridgeStartResult:
    """Prototype-owned start plus detached lifecycle adoption diagnostics."""

    state: PrototypeOptionAuthorityBridgeState
    lifecycle: STOMPOptionLifecycleExternalStartAdoptionResult
    prototype_started: Bool[Array, ""]
    authority_metadata_advanced: Bool[Array, ""]
    authority_desynchronized: Bool[Array, ""]
    stomp_start_evaluations: Int[Array, ""]
    transaction_applied: Bool[Array, ""]


@chex.dataclass(frozen=True)
class PrototypeOptionAuthorityBridgeUpdateResult:
    """One ordinary Prototype transition and its best-effort audit adoption."""

    state: PrototypeOptionAuthorityBridgeState
    prototype: PrototypeUpdateResult
    lifecycle: STOMPOptionLifecycleExternalAdoptionResult
    lifecycle_owner_finalization: (
        STOMPOptionLifecycleExternalOwnerFinalizationResult
    )
    audit_inputs_valid: Bool[Array, ""]
    prototype_control_applied: Bool[Array, ""]
    authority_metadata_advanced: Bool[Array, ""]
    authority_desynchronized: Bool[Array, ""]
    stomp_update_evaluations: Int[Array, ""]
    real_control_stomp_update_evaluations: Int[Array, ""]
    imagined_stomp_update_evaluations: Int[Array, ""]
    total_stomp_update_evaluations: Int[Array, ""]
    option_search_learner_updates: Int[Array, ""]
    stomp_internal_planning_backups: Int[Array, ""]
    control_transition_rolled_back_by_bridge: Bool[Array, ""]
    transaction_applied: Bool[Array, ""]


@dataclasses.dataclass(frozen=True, slots=True)
class PrototypeOptionAuthorityBridgeResourceBudget:
    """Measured persistence and exact ownership/work declarations."""

    persistent_state_nbytes: int
    prototype_state_nbytes: int
    detached_authority_metadata_nbytes: int
    bridge_binding_nbytes: int
    persistent_stomp_state_owners: int
    detached_metadata_stomp_state_owners: int
    borrowed_stomp_bindings: int
    persistent_prepared_transactions: int
    real_control_stomp_updates_per_ordinary_transition: int
    configured_imagined_stomp_updates_per_ordinary_transition: int
    max_total_stomp_updates_per_ordinary_transition: int
    option_search_learner_updates_per_ordinary_transition: int
    stomp_internal_planning_backups_per_ordinary_transition: int
    stomp_updates_per_audit_adoption: int
    stomp_updates_per_retirement_transaction: int
    stomp_updates_per_replacement_transaction: int
    retirement_preparation_recomputations_per_commit: int
    replacement_preparation_recomputations_per_commit: int
    retirement_prepare_host_only: bool
    retirement_commit_host_only: bool
    replacement_prepare_host_only: bool
    replacement_commit_host_only: bool
    derivation_recomputed_on_audit_adoption: bool
    caller_authority_required: bool
    caller_authenticated: bool
    checksum_authenticated: bool
    assessment: str
    benefit_claim: bool
    evidence_authority: bool
    promotion_authority: bool
    safety_authority: bool
    go_no_go_authority: bool
    retirement_authority: bool
    discovery_authority: bool
    dispatch_authority: bool
    autonomous_curation_authority: bool
    scientific_promotion_allowed: bool
    checkpoint_schema: str


@chex.dataclass(frozen=True)
class PrototypeOptionAuthorityBridgeRetirementPrepared:
    """Read-only, source-bound proposal for one authorized retirement.

    The nested authority result is transient. Its ``STOMPState`` is never
    copied into persistent bridge state unless :meth:`commit_retirement`
    re-derives this complete preparation and atomically rebinds OaK.
    """

    source_state: PrototypeOptionAuthorityBridgeState
    handoff: CumulantOptionRetirementHandoff
    authority_receipt: OptionRetirementAuthorityReceipt
    phase_one_key: Array
    phase_two_key: Array
    authority_result: AuthorizedOptionReplacementRetirementResult
    oak_rebind: OaKOptionSlotRebindResult
    proposed_state: PrototypeOptionAuthorityBridgeState
    reset_slots: Bool[Array, " option_budget"]
    source_binding_valid: Bool[Array, ""]
    candidate_valid: Bool[Array, ""]
    preparation_valid: Bool[Array, ""]
    prepared_checksum: UInt[Array, " 2"]


@chex.dataclass(frozen=True)
class PrototypeOptionAuthorityBridgeRetirementCommitResult:
    """Atomic adoption diagnostics for one re-derived retirement proposal."""

    state: PrototypeOptionAuthorityBridgeState
    retirement: AuthorizedOptionReplacementRetirementResult
    oak_rebind: OaKOptionSlotRebindResult
    destination_matches_source: Bool[Array, ""]
    prepared_integrity_valid: Bool[Array, ""]
    preparation_derivation_valid: Bool[Array, ""]
    exact_owner_rebind: Bool[Array, ""]
    cold_mask_applied: Bool[Array, ""]
    caller_authority_required: Bool[Array, ""]
    caller_authenticated: Bool[Array, ""]
    transaction_applied: Bool[Array, ""]


@chex.dataclass(frozen=True)
class PrototypeOptionAuthorityBridgeReplacementPrepared:
    """Read-only lower-level replacement preparation bound to one bridge."""

    source_state: PrototypeOptionAuthorityBridgeState
    authority_prepared: AuthorizedOptionReplacementPrepared
    source_binding_valid: Bool[Array, ""]
    preparation_valid: Bool[Array, ""]
    prepared_checksum: UInt[Array, " 2"]


@chex.dataclass(frozen=True)
class PrototypeOptionAuthorityBridgeReplacementCommitResult:
    """Atomic ordinary-authority advance plus optional live OaK replacement."""

    state: PrototypeOptionAuthorityBridgeState
    replacement: AuthorizedOptionReplacementResult
    oak_rebind: OaKOptionSlotRebindResult
    destination_matches_source: Bool[Array, ""]
    prepared_integrity_valid: Bool[Array, ""]
    lower_preparation_derivation_valid: Bool[Array, ""]
    exact_owner_rebind: Bool[Array, ""]
    cold_mask_applied: Bool[Array, ""]
    ordinary_advance_applied: Bool[Array, ""]
    replacement_applied: Bool[Array, ""]
    caller_authority_required: Bool[Array, ""]
    caller_authenticated: Bool[Array, ""]
    transaction_applied: Bool[Array, ""]


class PrototypeOptionAuthorityBridge:
    """Compose one Prototype controller with detached option authority."""

    def __init__(
        self,
        prototype: PrototypeAgent,
        authority: AuthorizedOptionReplacementController,
    ) -> None:
        if type(prototype) is not PrototypeAgent:
            raise TypeError("prototype must be an exact PrototypeAgent")
        if type(authority) is not AuthorizedOptionReplacementController:
            raise TypeError(
                "authority must be an exact AuthorizedOptionReplacementController"
            )
        if (
            prototype.config.oak.stomp.to_config()
            != authority.scheduler.installation.stomp_agent.config.to_config()
        ):
            raise ValueError(
                "Prototype and option authority must use the exact same STOMP config"
            )
        self._prototype = prototype
        self._authority = authority
        self._scheduler = authority.scheduler
        self._installation = self._scheduler.installation
        self._oak = prototype._oak

    @property
    def prototype(self) -> PrototypeAgent:
        return self._prototype

    @property
    def authority(self) -> AuthorizedOptionReplacementController:
        return self._authority

    def _payload_arrays(
        self,
        state: PrototypeOptionAuthorityBridgeState,
    ) -> tuple[Array, ...]:
        return tuple(
            cast(Array, leaf)
            for leaf in jax.tree_util.tree_leaves(
                (
                    state.prototype_state,
                    state.authority_metadata,
                    state.extended_action_mask,
                    state.authority_synchronized,
                    state.authority_error,
                    state.revision,
                )
            )
        )

    def _with_checksum(
        self,
        state: PrototypeOptionAuthorityBridgeState,
    ) -> PrototypeOptionAuthorityBridgeState:
        return cast(
            PrototypeOptionAuthorityBridgeState,
            state.replace(
                binding_checksum=_checksum_arrays(self._payload_arrays(state))
            ),
        )

    def _check_state_contract(
        self,
        state: PrototypeOptionAuthorityBridgeState,
    ) -> None:
        if type(state) is not PrototypeOptionAuthorityBridgeState:
            raise TypeError(
                "state must be an exact PrototypeOptionAuthorityBridgeState"
            )
        if type(state.prototype_state) is not PrototypeAgentState:
            raise TypeError("state.prototype_state has the wrong exact type")
        self._authority._check_metadata_contract(state.authority_metadata)
        arrays = (
            (
                state.extended_action_mask,
                "extended_action_mask",
                (self._prototype.config.oak.stomp.n_total_actions,),
                jnp.bool_,
            ),
            (state.authority_synchronized, "authority_synchronized", (), jnp.bool_),
            (state.authority_error, "authority_error", (), jnp.int32),
            (state.revision, "revision", (), jnp.int32),
            (state.binding_checksum, "binding_checksum", (2,), jnp.uint32),
        )
        for value, name, shape, dtype in arrays:
            array = jnp.asarray(value)
            if array.shape != shape:
                raise ValueError(f"state.{name} must have shape {shape}")
            if array.dtype != dtype:
                raise TypeError(f"state.{name} must have dtype {dtype}")

    def _expected_action_mask(
        self,
        metadata: AuthorizedOptionReplacementMetadataState,
    ) -> Array:
        primitive = jnp.ones(
            (self._prototype.config.oak.n_primitive_actions,),
            dtype=jnp.bool_,
        )
        return jnp.concatenate((primitive, metadata.installed_slot_mask))

    @staticmethod
    def _installation_from_state(
        state: AuthorizedOptionReplacementState,
    ) -> CumulantOptionInstallationState:
        return state.scheduler_state.installation_state

    def _initial_owner_receipt_payload_arrays(
        self,
        receipt: PrototypeOptionAuthorityBridgeInitialOwnerBindingReceipt,
    ) -> tuple[Array, ...]:
        return tuple(
            cast(Array, leaf)
            for leaf in jax.tree_util.tree_leaves(
                (
                    _INITIAL_OWNER_BINDING_TAG,
                    receipt.source_prototype_state,
                    receipt.source_authority_state,
                    receipt.prototype_owner_digest,
                    receipt.authority_owner_digest,
                    receipt.binding_authorized,
                    receipt.caller_authenticated,
                )
            )
        )

    def declare_initial_owner_binding(
        self,
        prototype_state: PrototypeAgentState,
        authority_state: AuthorizedOptionReplacementState,
        *,
        binding_authorized: bool | Array,
    ) -> PrototypeOptionAuthorityBridgeInitialOwnerBindingReceipt:
        """Source-bind one explicit, unauthenticated initial adoption request."""

        if type(prototype_state) is not PrototypeAgentState:
            raise TypeError("prototype_state must be an exact PrototypeAgentState")
        if type(authority_state) is not AuthorizedOptionReplacementState:
            raise TypeError(
                "authority_state must be an exact AuthorizedOptionReplacementState"
            )
        authorized, _ = _normalize_audit_bool(
            binding_authorized,
            name="binding_authorized",
        )
        prototype_owner = _prototype_oak_state(prototype_state.oak_state).stomp_state
        authority_owner = self._installation_from_state(
            authority_state
        ).lifecycle_state.stomp_state
        receipt = PrototypeOptionAuthorityBridgeInitialOwnerBindingReceipt(
            source_prototype_state=prototype_state,
            source_authority_state=authority_state,
            prototype_owner_digest=stomp_typed_tree_digest(prototype_owner),
            authority_owner_digest=stomp_typed_tree_digest(authority_owner),
            binding_authorized=authorized,
            receipt_checksum=jnp.zeros((2,), dtype=jnp.uint32),
            caller_authenticated=jnp.asarray(False, dtype=jnp.bool_),
        )
        return cast(
            PrototypeOptionAuthorityBridgeInitialOwnerBindingReceipt,
            receipt.replace(
                receipt_checksum=_checksum_arrays(
                    self._initial_owner_receipt_payload_arrays(receipt)
                )
            ),
        )

    def bind_initial_prototype_owner(
        self,
        prototype_state: PrototypeAgentState,
        authority_state: AuthorizedOptionReplacementState,
        receipt: PrototypeOptionAuthorityBridgeInitialOwnerBindingReceipt,
    ) -> PrototypeOptionAuthorityBridgeInitialOwnerBindingResult:
        """Adopt the canonical authority owner into an untouched Prototype.

        This is the only non-equality initialization seam.  It is directional:
        authority history is never rewritten from a Prototype owner.  Reuse
        against an already-adopted destination, stale sources, tampering,
        non-pristine Prototype state, or non-quiescent authority state returns
        an exact Prototype no-op.  Re-evaluation against the receipt's exact
        unchanged sources is intentionally idempotent; the unkeyed receipt is
        source-bound authorization, not a single-use authenticated capability.
        """

        if type(prototype_state) is not PrototypeAgentState:
            raise TypeError("prototype_state must be an exact PrototypeAgentState")
        if type(authority_state) is not AuthorizedOptionReplacementState:
            raise TypeError(
                "authority_state must be an exact AuthorizedOptionReplacementState"
            )
        if type(receipt) is not PrototypeOptionAuthorityBridgeInitialOwnerBindingReceipt:
            raise TypeError(
                "receipt must be an exact initial-owner binding receipt"
            )
        for value, name, shape, dtype in (
            (
                receipt.prototype_owner_digest,
                "prototype_owner_digest",
                (8,),
                jnp.uint32,
            ),
            (
                receipt.authority_owner_digest,
                "authority_owner_digest",
                (8,),
                jnp.uint32,
            ),
            (receipt.binding_authorized, "binding_authorized", (), jnp.bool_),
            (receipt.receipt_checksum, "receipt_checksum", (2,), jnp.uint32),
            (receipt.caller_authenticated, "caller_authenticated", (), jnp.bool_),
        ):
            array = jnp.asarray(value)
            if array.shape != shape:
                raise ValueError(f"receipt.{name} must have shape {shape}")
            if array.dtype != dtype:
                raise TypeError(f"receipt.{name} must have dtype {dtype}")
        source_prototype_matches = _tree_exact_equal(
            prototype_state,
            receipt.source_prototype_state,
        )
        source_authority_matches = _tree_exact_equal(
            authority_state,
            receipt.source_authority_state,
        )
        receipt_integrity_valid = jnp.array_equal(
            receipt.receipt_checksum,
            _checksum_arrays(self._initial_owner_receipt_payload_arrays(receipt)),
        ) & (~receipt.caller_authenticated)
        source_oak = _prototype_oak_state(prototype_state.oak_state)
        authority_installation = self._installation_from_state(authority_state)
        authority_lifecycle = authority_installation.lifecycle_state
        canonical_owner = authority_lifecycle.stomp_state
        owner_digests_match = jnp.array_equal(
            receipt.prototype_owner_digest,
            stomp_typed_tree_digest(source_oak.stomp_state),
        ) & jnp.array_equal(
            receipt.authority_owner_digest,
            stomp_typed_tree_digest(canonical_owner),
        )
        prototype_pristine = (
            self._prototype.validate_state(prototype_state)
            & self._prototype._pristine_state_consistent(prototype_state)
        )
        authority_quiescent = (
            self._authority.state_valid(authority_state)
            & (~authority_lifecycle.started)
            & (canonical_owner.executing_option < 0)
            & (canonical_owner.step_count == 0)
            & jnp.all(canonical_owner.step_words == 0)
            & (authority_lifecycle.audit_state.active_option < 0)
            & (~authority_lifecycle.audit_state.trial_active)
        )
        canonical_owner_valid = self._oak.stomp_agent.state_valid(canonical_owner)
        candidate_oak = source_oak.replace(stomp_state=canonical_owner)
        candidate = _replace_prototype_oak_state(prototype_state, candidate_oak)
        candidate_valid = (
            self._prototype.validate_state(candidate)
            & self._prototype._pristine_state_consistent(candidate)
        )
        transaction_applied = (
            source_prototype_matches
            & source_authority_matches
            & receipt_integrity_valid
            & receipt.binding_authorized
            & owner_digests_match
            & prototype_pristine
            & authority_quiescent
            & canonical_owner_valid
            & candidate_valid
        )
        destination = cast(
            PrototypeAgentState,
            jax.lax.cond(
                transaction_applied,
                lambda _: candidate,
                lambda _: prototype_state,
                None,
            ),
        )
        return PrototypeOptionAuthorityBridgeInitialOwnerBindingResult(
            prototype_state=destination,
            source_prototype_matches=source_prototype_matches,
            source_authority_matches=source_authority_matches,
            receipt_integrity_valid=receipt_integrity_valid,
            prototype_pristine=prototype_pristine,
            authority_quiescent=authority_quiescent,
            canonical_owner_valid=canonical_owner_valid,
            canonical_owner_adopted=(
                transaction_applied
                & _tree_exact_equal(
                    _prototype_oak_state(destination.oak_state).stomp_state,
                    canonical_owner,
                )
            ),
            caller_authority_required=jnp.asarray(True, dtype=jnp.bool_),
            caller_authenticated=jnp.asarray(False, dtype=jnp.bool_),
            transaction_applied=transaction_applied,
        )

    def init(
        self,
        prototype_state: PrototypeAgentState,
        authority_state: AuthorizedOptionReplacementState,
    ) -> PrototypeOptionAuthorityBridgeState:
        """Bind valid initial states while preserving Prototype as sole owner."""

        if type(prototype_state) is not PrototypeAgentState:
            raise TypeError("prototype_state must be an exact PrototypeAgentState")
        if type(authority_state) is not AuthorizedOptionReplacementState:
            raise TypeError(
                "authority_state must be an exact AuthorizedOptionReplacementState"
            )
        if not bool(jax.device_get(self._prototype.validate_state(prototype_state))):
            raise ValueError("prototype_state must satisfy its complete contract")
        if not bool(jax.device_get(self._authority.state_valid(authority_state))):
            raise ValueError("authority_state must satisfy its complete contract")
        oak = _prototype_oak_state(prototype_state.oak_state)
        authority_stomp = self._installation_from_state(
            authority_state
        ).lifecycle_state.stomp_state
        if not bool(jax.device_get(_tree_exact_equal(oak.stomp_state, authority_stomp))):
            raise ValueError(
                "Prototype and authority owners differ; use the explicit "
                "authority-to-pristine-Prototype initial binding receipt"
            )
        lifecycle = self._installation_from_state(authority_state).lifecycle_state
        if bool(jax.device_get(prototype_state.started != lifecycle.started)):
            raise ValueError("Prototype and lifecycle start phases must match")
        metadata = self._authority.detach_borrowed_stomp(authority_state)
        state = PrototypeOptionAuthorityBridgeState(
            prototype_state=prototype_state,
            authority_metadata=metadata,
            extended_action_mask=self._expected_action_mask(metadata),
            authority_synchronized=jnp.asarray(True, dtype=jnp.bool_),
            authority_error=jnp.asarray(
                PROTOTYPE_OPTION_AUTHORITY_BRIDGE_ERROR_NONE,
                dtype=jnp.int32,
            ),
            revision=jnp.asarray(0, dtype=jnp.int32),
            binding_checksum=jnp.zeros((2,), dtype=jnp.uint32),
        )
        state = self._with_checksum(state)
        if not bool(jax.device_get(self.state_valid(state))):
            raise ValueError("initialized bridge state failed its exact contract")
        return state

    def state_valid(
        self,
        state: PrototypeOptionAuthorityBridgeState,
    ) -> Bool[Array, ""]:
        """Validate sole ownership, mask, detached metadata, and synchronization."""

        self._check_state_contract(state)
        prototype_valid = self._prototype.validate_state(state.prototype_state)
        metadata_valid = self._authority.metadata_state_valid(
            state.authority_metadata
        )
        expected_mask = self._expected_action_mask(state.authority_metadata)
        error_contract = (
            state.authority_synchronized
            == (
                state.authority_error
                == PROTOTYPE_OPTION_AUTHORITY_BRIDGE_ERROR_NONE
            )
        ) & (
            (state.authority_error == PROTOTYPE_OPTION_AUTHORITY_BRIDGE_ERROR_NONE)
            | (
                state.authority_error
                == PROTOTYPE_OPTION_AUTHORITY_BRIDGE_ERROR_DESYNCHRONIZED
            )
        )
        oak = _prototype_oak_state(state.prototype_state.oak_state)
        attached = self._authority.attach_borrowed_stomp(
            state.authority_metadata,
            oak.stomp_state,
        )
        lifecycle = (
            state.authority_metadata.scheduler_metadata.installation_metadata
            .lifecycle_metadata
        )
        synchronized_contract = (
            attached.transaction_applied
            & (state.prototype_state.started == lifecycle.started)
            & jnp.array_equal(oak.step_words, lifecycle.stomp_step_words)
        )
        return (
            prototype_valid
            & metadata_valid
            & jnp.array_equal(state.extended_action_mask, expected_mask)
            & jnp.all(
                state.extended_action_mask[
                    : self._prototype.config.oak.n_primitive_actions
                ]
            )
            & error_contract
            & (state.revision >= 0)
            & jnp.where(
                state.authority_synchronized,
                synchronized_contract,
                jnp.asarray(True, dtype=jnp.bool_),
            )
            & jnp.array_equal(
                state.binding_checksum,
                _checksum_arrays(self._payload_arrays(state)),
            )
        )

    def _lift_lifecycle_metadata(
        self,
        source: AuthorizedOptionReplacementState,
        lifecycle_metadata: Any,
        destination_stomp: STOMPState,
    ) -> tuple[AuthorizedOptionReplacementMetadataState, Array]:
        """Lift one lifecycle advance through installer/scheduler/replacement."""

        installation = source.scheduler_state.installation_state
        lifecycle_api = self._installation.lifecycle.with_external_semantic_digests(
            installation.installed_semantic_digests
        )
        attached = lifecycle_api.attach_borrowed_stomp(
            lifecycle_metadata,
            destination_stomp,
        )
        next_installation = self._installation._with_checksum(
            installation.replace(
                lifecycle_state=attached.state,
                revision=_saturating_increment(installation.revision),
                binding_checksum=jnp.zeros((2,), dtype=jnp.uint32),
            )
        )
        next_control_words, control_capacity = _increment_words(
            source.scheduler_state.control_update_words
        )
        next_scheduler = self._scheduler._with_checksum(
            source.scheduler_state.replace(
                installation_state=next_installation,
                control_update_words=next_control_words,
                binding_checksum=jnp.zeros((2,), dtype=jnp.uint32),
            )
        )
        next_authority = self._authority._with_checksum(
            source.replace(
                scheduler_state=next_scheduler,
                canonical_scheduler_checksum=next_scheduler.binding_checksum,
                binding_checksum=jnp.zeros((2,), dtype=jnp.uint32),
            )
        )
        valid = (
            attached.transaction_applied
            & control_capacity
            & self._authority.state_valid(next_authority)
        )
        return self._authority.detach_borrowed_stomp(next_authority), valid

    def _next_state(
        self,
        source: PrototypeOptionAuthorityBridgeState,
        prototype_state: PrototypeAgentState,
        authority_metadata: AuthorizedOptionReplacementMetadataState,
        *,
        synchronized: Array,
    ) -> PrototypeOptionAuthorityBridgeState:
        error = jnp.where(
            synchronized,
            jnp.asarray(PROTOTYPE_OPTION_AUTHORITY_BRIDGE_ERROR_NONE, jnp.int32),
            jnp.asarray(
                PROTOTYPE_OPTION_AUTHORITY_BRIDGE_ERROR_DESYNCHRONIZED,
                jnp.int32,
            ),
        )
        candidate = PrototypeOptionAuthorityBridgeState(
            prototype_state=prototype_state,
            authority_metadata=authority_metadata,
            extended_action_mask=self._expected_action_mask(authority_metadata),
            authority_synchronized=synchronized,
            authority_error=error,
            revision=_saturating_increment(source.revision),
            binding_checksum=jnp.zeros((2,), dtype=jnp.uint32),
        )
        return self._with_checksum(candidate)

    def _retirement_prepared_payload_arrays(
        self,
        prepared: PrototypeOptionAuthorityBridgeRetirementPrepared,
    ) -> tuple[Array, ...]:
        return tuple(
            cast(Array, leaf)
            for leaf in jax.tree_util.tree_leaves(
                (
                    prepared.source_state,
                    prepared.handoff,
                    prepared.authority_receipt,
                    prepared.phase_one_key,
                    prepared.phase_two_key,
                    prepared.authority_result,
                    prepared.oak_rebind,
                    prepared.proposed_state,
                    prepared.reset_slots,
                    prepared.source_binding_valid,
                    prepared.candidate_valid,
                    prepared.preparation_valid,
                )
            )
        )

    def _with_retirement_prepared_checksum(
        self,
        prepared: PrototypeOptionAuthorityBridgeRetirementPrepared,
    ) -> PrototypeOptionAuthorityBridgeRetirementPrepared:
        return cast(
            PrototypeOptionAuthorityBridgeRetirementPrepared,
            prepared.replace(
                prepared_checksum=_checksum_arrays(
                    self._retirement_prepared_payload_arrays(prepared)
                )
            ),
        )

    def prepare_retirement(
        self,
        state: PrototypeOptionAuthorityBridgeState,
        handoff: CumulantOptionRetirementHandoff,
        authority_receipt: OptionRetirementAuthorityReceipt,
        phase_one_key: Array,
        phase_two_key: Array,
    ) -> PrototypeOptionAuthorityBridgeRetirementPrepared:
        """Prepare, but do not persist, one exact authorized retirement.

        The lower retirement transaction is pure.  Its changed STOMP option
        slot is projected into OaK through the strict quiescent rebind seam,
        then the complete candidate is source-bound for a later commit.
        """

        self._check_state_contract(state)
        source_valid = self.state_valid(state) & state.authority_synchronized
        source_oak = _prototype_oak_state(state.prototype_state.oak_state)
        source_authority = self._authority.attach_borrowed_stomp(
            state.authority_metadata,
            source_oak.stomp_state,
        )
        retirement = self._authority.retire(
            source_authority.state,
            handoff,
            authority_receipt,
            phase_one_key,
            phase_two_key,
        )
        destination_installation = self._installation_from_state(retirement.state)
        destination_stomp = destination_installation.lifecycle_state.stomp_state
        reset_slots = state.authority_metadata.installed_slot_mask & (
            ~retirement.state.installed_slot_mask
        )
        oak_rebind = self._oak.rebind_option_slots(
            source_oak,
            destination_stomp,
            reset_slots,
        )
        prototype_candidate = _replace_prototype_oak_state(
            state.prototype_state,
            oak_rebind.state,
        )
        authority_metadata = self._authority.detach_borrowed_stomp(retirement.state)
        candidate = self._next_state(
            state,
            prototype_candidate,
            authority_metadata,
            synchronized=jnp.asarray(True, dtype=jnp.bool_),
        )
        source_binding_valid = source_valid & source_authority.transaction_applied
        exact_one_reset = jnp.sum(reset_slots, dtype=jnp.int32) == 1
        exact_owner = _tree_exact_equal(
            _prototype_oak_state(candidate.prototype_state.oak_state).stomp_state,
            destination_stomp,
        )
        cold_mask_applied = jnp.array_equal(
            candidate.extended_action_mask,
            self._expected_action_mask(authority_metadata),
        )
        candidate_valid = (
            retirement.transaction_applied
            & exact_one_reset
            & oak_rebind.transaction_applied
            & exact_owner
            & cold_mask_applied
            & self.state_valid(candidate)
        )
        preparation_valid = source_binding_valid & candidate_valid
        proposed = cast(
            PrototypeOptionAuthorityBridgeState,
            jax.lax.cond(
                preparation_valid,
                lambda _: candidate,
                lambda _: state,
                None,
            ),
        )
        prepared = PrototypeOptionAuthorityBridgeRetirementPrepared(
            source_state=state,
            handoff=handoff,
            authority_receipt=authority_receipt,
            phase_one_key=phase_one_key,
            phase_two_key=phase_two_key,
            authority_result=retirement,
            oak_rebind=oak_rebind,
            proposed_state=proposed,
            reset_slots=reset_slots,
            source_binding_valid=source_binding_valid,
            candidate_valid=candidate_valid,
            preparation_valid=preparation_valid,
            prepared_checksum=jnp.zeros((2,), dtype=jnp.uint32),
        )
        return self._with_retirement_prepared_checksum(prepared)

    def commit_retirement(
        self,
        state: PrototypeOptionAuthorityBridgeState,
        prepared: PrototypeOptionAuthorityBridgeRetirementPrepared,
    ) -> PrototypeOptionAuthorityBridgeRetirementCommitResult:
        """Re-derive and atomically commit one exact retirement proposal."""

        self._check_state_contract(state)
        if type(prepared) is not PrototypeOptionAuthorityBridgeRetirementPrepared:
            raise TypeError(
                "prepared must be an exact "
                "PrototypeOptionAuthorityBridgeRetirementPrepared"
            )
        supplied_integrity = jnp.array_equal(
            prepared.prepared_checksum,
            _checksum_arrays(self._retirement_prepared_payload_arrays(prepared)),
        )
        recomputed = self.prepare_retirement(
            prepared.source_state,
            prepared.handoff,
            prepared.authority_receipt,
            prepared.phase_one_key,
            prepared.phase_two_key,
        )
        derivation_valid = _tree_exact_equal(prepared, recomputed)
        destination_matches = _tree_exact_equal(state, prepared.source_state)
        exact_owner_rebind = (
            recomputed.oak_rebind.transaction_applied
            & _tree_exact_equal(
                _prototype_oak_state(
                    recomputed.proposed_state.prototype_state.oak_state
                ).stomp_state,
                self._installation_from_state(
                    recomputed.authority_result.state
                ).lifecycle_state.stomp_state,
            )
        )
        cold_mask_applied = jnp.array_equal(
            recomputed.proposed_state.extended_action_mask,
            self._expected_action_mask(
                recomputed.proposed_state.authority_metadata
            ),
        )
        applied = (
            destination_matches
            & supplied_integrity
            & derivation_valid
            & recomputed.preparation_valid
            & exact_owner_rebind
            & cold_mask_applied
            & self.state_valid(state)
        )
        next_state = cast(
            PrototypeOptionAuthorityBridgeState,
            jax.lax.cond(
                applied,
                lambda _: recomputed.proposed_state,
                lambda _: state,
                None,
            ),
        )
        return PrototypeOptionAuthorityBridgeRetirementCommitResult(
            state=next_state,
            retirement=recomputed.authority_result,
            oak_rebind=recomputed.oak_rebind,
            destination_matches_source=destination_matches,
            prepared_integrity_valid=supplied_integrity,
            preparation_derivation_valid=derivation_valid,
            exact_owner_rebind=exact_owner_rebind,
            cold_mask_applied=cold_mask_applied,
            caller_authority_required=jnp.asarray(True, dtype=jnp.bool_),
            caller_authenticated=jnp.asarray(False, dtype=jnp.bool_),
            transaction_applied=applied,
        )

    def _replacement_prepared_payload_arrays(
        self,
        prepared: PrototypeOptionAuthorityBridgeReplacementPrepared,
    ) -> tuple[Array, ...]:
        return tuple(
            cast(Array, leaf)
            for leaf in jax.tree_util.tree_leaves(
                (
                    prepared.source_state,
                    prepared.authority_prepared,
                    prepared.source_binding_valid,
                    prepared.preparation_valid,
                )
            )
        )

    def prepare_replacement(
        self,
        state: PrototypeOptionAuthorityBridgeState,
        arm: AuthorizedOptionReplacementArm,
        observation: CumulantOptionSchedulerObservation,
        live_inputs: CumulantOptionLiveInputs,
    ) -> PrototypeOptionAuthorityBridgeReplacementPrepared:
        """Stage one source-bound replacement without changing bridge state."""

        self._check_state_contract(state)
        source_valid = self.state_valid(state) & state.authority_synchronized
        source_oak = _prototype_oak_state(state.prototype_state.oak_state)
        source_authority = self._authority.attach_borrowed_stomp(
            state.authority_metadata,
            source_oak.stomp_state,
        )
        authority_prepared = self._authority.prepare(
            source_authority.state,
            arm,
            observation,
            live_inputs,
        )
        source_binding_valid = source_valid & source_authority.transaction_applied
        preparation_valid = (
            source_binding_valid
            & authority_prepared.diagnostics.transaction_valid
            & _tree_exact_equal(
                authority_prepared.source_state,
                source_authority.state,
            )
        )
        prepared = PrototypeOptionAuthorityBridgeReplacementPrepared(
            source_state=state,
            authority_prepared=authority_prepared,
            source_binding_valid=source_binding_valid,
            preparation_valid=preparation_valid,
            prepared_checksum=jnp.zeros((2,), dtype=jnp.uint32),
        )
        return cast(
            PrototypeOptionAuthorityBridgeReplacementPrepared,
            prepared.replace(
                prepared_checksum=_checksum_arrays(
                    self._replacement_prepared_payload_arrays(prepared)
                )
            ),
        )

    def commit_replacement(
        self,
        state: PrototypeOptionAuthorityBridgeState,
        prepared: PrototypeOptionAuthorityBridgeReplacementPrepared,
        authority_receipt: OptionReplacementAuthorityReceipt,
    ) -> PrototypeOptionAuthorityBridgeReplacementCommitResult:
        """Commit an ordinary authority advance and optional slot replacement.

        The lower controller independently replays its complete preparation.
        A live replacement is then accepted only if the same changed STOMP
        slot can be rebound into OaK.  Failure rolls back the complete bridge
        authority transaction; it never rolls back an ordinary real Prototype
        transition, which uses :meth:`update_transition` instead.
        """

        self._check_state_contract(state)
        if type(prepared) is not PrototypeOptionAuthorityBridgeReplacementPrepared:
            raise TypeError(
                "prepared must be an exact "
                "PrototypeOptionAuthorityBridgeReplacementPrepared"
            )
        supplied_integrity = jnp.array_equal(
            prepared.prepared_checksum,
            _checksum_arrays(self._replacement_prepared_payload_arrays(prepared)),
        )
        destination_matches = _tree_exact_equal(state, prepared.source_state)
        source_oak = _prototype_oak_state(state.prototype_state.oak_state)
        source_authority = self._authority.attach_borrowed_stomp(
            state.authority_metadata,
            source_oak.stomp_state,
        )
        source_binding_valid = (
            self.state_valid(state)
            & state.authority_synchronized
            & source_authority.transaction_applied
        )
        preparation_facts_valid = (
            prepared.source_binding_valid == source_binding_valid
        ) & (
            prepared.preparation_valid
            == (
                source_binding_valid
                & prepared.authority_prepared.diagnostics.transaction_valid
                & _tree_exact_equal(
                    prepared.authority_prepared.source_state,
                    source_authority.state,
                )
            )
        )
        replacement = self._authority.commit(
            source_authority.state,
            prepared.authority_prepared,
            authority_receipt,
        )
        destination_stomp = self._installation_from_state(
            replacement.state
        ).lifecycle_state.stomp_state
        oak_rebind = self._oak.rebind_option_slots(
            source_oak,
            destination_stomp,
            replacement.reset_slots,
        )
        replacement_applied = replacement.diagnostics.replacement_applied
        owner_unchanged = _tree_exact_equal(source_oak.stomp_state, destination_stomp)
        exact_owner_rebind = jnp.where(
            replacement_applied,
            oak_rebind.transaction_applied
            & _tree_exact_equal(oak_rebind.state.stomp_state, destination_stomp),
            owner_unchanged,
        )
        destination_oak = cast(
            OaKState,
            jax.lax.cond(
                replacement_applied,
                lambda _: oak_rebind.state,
                lambda _: source_oak,
                None,
            ),
        )
        prototype_candidate = _replace_prototype_oak_state(
            state.prototype_state,
            destination_oak,
        )
        authority_metadata = self._authority.detach_borrowed_stomp(
            replacement.state
        )
        candidate = self._next_state(
            state,
            prototype_candidate,
            authority_metadata,
            synchronized=jnp.asarray(True, dtype=jnp.bool_),
        )
        cold_mask_applied = (
            jnp.array_equal(
                candidate.extended_action_mask,
                replacement.extended_action_mask,
            )
            & jnp.array_equal(
                candidate.extended_action_mask,
                self._expected_action_mask(authority_metadata),
            )
        )
        lower_derivation_valid = (
            replacement.diagnostics.preparation_derivation_valid
        )
        ordinary_advance = replacement.diagnostics.ordinary_advance_applied
        applied = (
            self.state_valid(state)
            & state.authority_synchronized
            & source_authority.transaction_applied
            & destination_matches
            & supplied_integrity
            & preparation_facts_valid
            & lower_derivation_valid
            & ordinary_advance
            & exact_owner_rebind
            & cold_mask_applied
            & self.state_valid(candidate)
        )
        next_state = cast(
            PrototypeOptionAuthorityBridgeState,
            jax.lax.cond(
                applied,
                lambda _: candidate,
                lambda _: state,
                None,
            ),
        )
        return PrototypeOptionAuthorityBridgeReplacementCommitResult(
            state=next_state,
            replacement=replacement,
            oak_rebind=oak_rebind,
            destination_matches_source=destination_matches,
            prepared_integrity_valid=supplied_integrity,
            lower_preparation_derivation_valid=lower_derivation_valid,
            exact_owner_rebind=exact_owner_rebind,
            cold_mask_applied=cold_mask_applied,
            ordinary_advance_applied=applied,
            replacement_applied=applied & replacement_applied,
            caller_authority_required=jnp.asarray(True, dtype=jnp.bool_),
            caller_authenticated=jnp.asarray(False, dtype=jnp.bool_),
            transaction_applied=applied,
        )

    def start(
        self,
        state: PrototypeOptionAuthorityBridgeState,
        initial_observation: Array,
    ) -> PrototypeOptionAuthorityBridgeStartResult:
        """Start Prototype once under the persistent cold/live option mask."""

        self._check_state_contract(state)
        source_valid = self.state_valid(state) & state.authority_synchronized
        source_oak = _prototype_oak_state(state.prototype_state.oak_state)
        source_authority = self._authority.attach_borrowed_stomp(
            state.authority_metadata,
            source_oak.stomp_state,
        )
        prototype_destination = self._prototype.start(
            state.prototype_state,
            initial_observation,
            extended_action_mask=state.extended_action_mask,
        )
        destination_oak = _prototype_oak_state(prototype_destination.oak_state)
        lifecycle_api = self._installation.lifecycle.with_external_semantic_digests(
            source_authority.state.scheduler_state.installation_state
            .installed_semantic_digests
        )
        lifecycle_metadata = (
            state.authority_metadata.scheduler_metadata.installation_metadata
            .lifecycle_metadata
        )
        declaration = lifecycle_api.declare_external_stomp_start(
            lifecycle_metadata,
            source_oak.stomp_state,
            destination_oak.stomp_state,
            caller_derivation_declared=jnp.asarray(True, dtype=jnp.bool_),
        )
        lifecycle = lifecycle_api.adopt_external_stomp_start(
            lifecycle_metadata,
            source_oak.stomp_state,
            destination_oak.stomp_state,
            destination_oak.stomp_state.base_last_obs,
            declaration,
        )
        lifted_metadata, lifted_valid = self._lift_lifecycle_metadata(
            source_authority.state,
            lifecycle.state,
            destination_oak.stomp_state,
        )
        prototype_started = (
            (~state.prototype_state.started)
            & prototype_destination.started
            & self._prototype.validate_state(prototype_destination)
        )
        synchronized = (
            source_valid
            & source_authority.transaction_applied
            & prototype_started
            & lifecycle.metadata_advanced
            & lifted_valid
        )
        selected_metadata = jax.lax.cond(
            synchronized,
            lambda _: lifted_metadata,
            lambda _: state.authority_metadata,
            None,
        )
        control_applied = source_valid & prototype_started
        candidate = self._next_state(
            state,
            prototype_destination,
            selected_metadata,
            synchronized=synchronized,
        )
        next_state = jax.lax.cond(
            control_applied,
            lambda _: candidate,
            lambda _: state,
            None,
        )
        return PrototypeOptionAuthorityBridgeStartResult(
            state=next_state,
            lifecycle=lifecycle,
            prototype_started=prototype_started,
            authority_metadata_advanced=synchronized,
            authority_desynchronized=control_applied & (~synchronized),
            stomp_start_evaluations=control_applied.astype(jnp.int32),
            transaction_applied=control_applied,
        )

    def update_transition(
        self,
        state: PrototypeOptionAuthorityBridgeState,
        transition: PrototypeTransition,
        candidate_update_audit_evidence: (
            PrototypeCandidateUpdateAuditEvidence | None
        ) = None,
        *,
        gradient_joy_evidence: PrototypeGradientJoyEvidence | None = None,
        experiential_memory_input: PrototypeExperientialMemoryInput | None = None,
        partner_policy_fusion_input: PrototypePartnerPolicyFusionInput | None = None,
        partner_policy_fusion_feedback: (
            PrototypePartnerPolicyFusionFeedback | None
        ) = None,
        context: int | Array = 0,
        idle_candidate_option: int | Array = 0,
        idle_initiation_eligible: bool | Array = False,
        comparator_randomized: bool | Array = False,
        treatment_propensity: float | Array = 0.0,
    ) -> PrototypeOptionAuthorityBridgeUpdateResult:
        """Run one Prototype transition and adopt its sole transient STOMP trace."""

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
        safe_context, context_valid = _normalize_audit_int32(
            context,
            name="context",
            lower=0,
            upper=self._installation.lifecycle.audit.config.n_contexts,
        )
        safe_idle_candidate, idle_candidate_valid = _normalize_audit_int32(
            idle_candidate_option,
            name="idle_candidate_option",
            lower=0,
            upper=self._prototype.config.oak.stomp.n_options,
        )
        safe_idle_eligible, idle_eligible_valid = _normalize_audit_bool(
            idle_initiation_eligible,
            name="idle_initiation_eligible",
        )
        safe_randomized, randomized_valid = _normalize_audit_bool(
            comparator_randomized,
            name="comparator_randomized",
        )
        safe_propensity, propensity_valid = _normalize_audit_propensity(
            treatment_propensity
        )
        audit_inputs_valid = (
            context_valid
            & idle_candidate_valid
            & idle_eligible_valid
            & randomized_valid
            & propensity_valid
        )
        source_valid = self.state_valid(state)
        # Even though invalid-source candidates are never committed below,
        # evaluate the pure Prototype candidate under a primitives-only mask.
        # This prevents a forged/resealed live-option mask from influencing
        # any returned transient control trace at the fail-closed boundary.
        fail_closed_action_mask = jnp.concatenate(
            (
                jnp.ones(
                    (self._prototype.config.oak.n_primitive_actions,),
                    dtype=jnp.bool_,
                ),
                jnp.zeros(
                    (self._prototype.config.oak.stomp.n_options,),
                    dtype=jnp.bool_,
                ),
            )
        )
        control_action_mask = jnp.where(
            source_valid,
            state.extended_action_mask,
            fail_closed_action_mask,
        )
        source_oak = _prototype_oak_state(state.prototype_state.oak_state)
        source_authority = self._authority.attach_borrowed_stomp(
            state.authority_metadata,
            source_oak.stomp_state,
        )
        prototype_result = self._prototype.update_transition(
            state.prototype_state,
            transition,
            candidate_update_audit_evidence=selected_audit_evidence,
            experiential_memory_input=experiential_memory_input,
            partner_policy_fusion_input=partner_policy_fusion_input,
            partner_policy_fusion_feedback=partner_policy_fusion_feedback,
            extended_action_mask=control_action_mask,
        )
        destination_oak = _prototype_oak_state(
            prototype_result.state.oak_state
        )
        installation = source_authority.state.scheduler_state.installation_state
        lifecycle_api = self._installation.lifecycle.with_external_semantic_digests(
            installation.installed_semantic_digests
        )
        lifecycle_metadata = (
            state.authority_metadata.scheduler_metadata.installation_metadata
            .lifecycle_metadata
        )
        declaration = lifecycle_api.declare_external_stomp_transition(
            lifecycle_metadata,
            source_oak.stomp_state,
            prototype_result.oak_stomp_update_result,
            env_reward=transition.reward,
            next_observation=prototype_result.oak_bootstrap_observation,
            discount=transition.discount,
            execution_boundary=prototype_result.oak_execution_boundary,
            extended_action_mask=control_action_mask,
            caller_derivation_declared=prototype_result.oak_stomp_update_available,
        )
        raw_lifecycle = lifecycle_api.adopt_external_stomp_update(
            lifecycle_metadata,
            source_oak.stomp_state,
            prototype_result.oak_stomp_update_result,
            declaration,
            env_reward=transition.reward,
            next_observation=prototype_result.oak_bootstrap_observation,
            discount=transition.discount,
            decision_observation=prototype_result.oak_decision_observation,
            execution_boundary=prototype_result.oak_execution_boundary,
            context=safe_context,
            idle_candidate_option=safe_idle_candidate,
            idle_initiation_eligible=safe_idle_eligible,
            comparator_randomized=safe_randomized,
            treatment_propensity=safe_propensity,
            extended_action_mask=control_action_mask,
        )
        audit_attempt_valid = (
            audit_inputs_valid
            & source_valid
            & state.authority_synchronized
            & source_authority.transaction_applied
        )
        unavailable = jnp.asarray(False, dtype=jnp.bool_)
        unavailable_lifecycle = raw_lifecycle.replace(
            state=lifecycle_metadata,
            source_metadata_valid=unavailable,
            source_stomp_valid=unavailable,
            source_binding_matches=unavailable,
            result_static_contract_valid=unavailable,
            result_clock_binding_valid=unavailable,
            result_endpoint_binding_valid=unavailable,
            termination_binding_valid=unavailable,
            reward_binding_valid=unavailable,
            model_signature_binding_valid=unavailable,
            declaration_binding_valid=unavailable,
            audit_applied=unavailable,
            metadata_advanced=unavailable,
            control_transition_rolled_back=unavailable,
            derivation_recomputed=unavailable,
            caller_authority_required=jnp.asarray(True, dtype=jnp.bool_),
            caller_authenticated=unavailable,
            transaction_applied=unavailable,
        )
        lifecycle = cast(
            STOMPOptionLifecycleExternalAdoptionResult,
            jax.tree.map(
                lambda available, missing: jnp.where(
                    audit_attempt_valid,
                    available,
                    missing,
                ),
                raw_lifecycle,
                unavailable_lifecycle,
            ),
        )
        lifecycle_owner_finalization = (
            lifecycle_api.finalize_external_stomp_owner(
                lifecycle.state,
                prototype_result.oak_owner_finalization_trace,
            )
        )
        final_trace_matches_prototype = _tree_exact_equal(
            prototype_result.oak_owner_finalization_trace.final_state,
            destination_oak.stomp_state,
        )
        lifted_metadata, lifted_valid = self._lift_lifecycle_metadata(
            source_authority.state,
            lifecycle_owner_finalization.state,
            destination_oak.stomp_state,
        )
        prototype_control_applied = (
            source_valid
            & prototype_result.transition_diagnostics.valid
            & prototype_result.oak_stomp_update_available
            & (prototype_result.oak_real_stomp_update_evaluations == 1)
            & self._prototype.validate_state(prototype_result.state)
        )
        synchronized = (
            state.authority_synchronized
            & source_authority.transaction_applied
            & prototype_control_applied
            & audit_inputs_valid
            & lifecycle.metadata_advanced
            & lifecycle_owner_finalization.metadata_finalized
            & final_trace_matches_prototype
            & lifted_valid
        )
        selected_metadata = jax.lax.cond(
            synchronized,
            lambda _: lifted_metadata,
            lambda _: state.authority_metadata,
            None,
        )
        candidate = self._next_state(
            state,
            prototype_result.state,
            selected_metadata,
            synchronized=synchronized,
        )
        next_state = jax.lax.cond(
            prototype_control_applied,
            lambda _: candidate,
            lambda _: state,
            None,
        )
        return PrototypeOptionAuthorityBridgeUpdateResult(
            state=next_state,
            prototype=prototype_result,
            lifecycle=lifecycle,
            lifecycle_owner_finalization=lifecycle_owner_finalization,
            audit_inputs_valid=audit_inputs_valid,
            prototype_control_applied=prototype_control_applied,
            authority_metadata_advanced=synchronized,
            authority_desynchronized=prototype_control_applied & (~synchronized),
            stomp_update_evaluations=prototype_result.oak_stomp_update_evaluations,
            real_control_stomp_update_evaluations=(
                prototype_result.oak_real_stomp_update_evaluations
            ),
            imagined_stomp_update_evaluations=(
                prototype_result.oak_imagined_stomp_update_evaluations
            ),
            total_stomp_update_evaluations=(
                prototype_result.oak_total_stomp_update_evaluations
            ),
            option_search_learner_updates=(
                prototype_result.oak_option_search_learner_updates
            ),
            stomp_internal_planning_backups=(
                prototype_result.oak_stomp_update_result.planning_backups
            ),
            control_transition_rolled_back_by_bridge=jnp.asarray(
                False,
                dtype=jnp.bool_,
            ),
            transaction_applied=prototype_control_applied,
        )

    def resource_budget(
        self,
        state: PrototypeOptionAuthorityBridgeState,
    ) -> PrototypeOptionAuthorityBridgeResourceBudget:
        """Measure persistence and report the bridge's exact trust boundary."""

        if not bool(jax.device_get(self.state_valid(state))):
            raise ValueError("resource measurement requires a valid bridge state")
        total = _tree_nbytes(state)
        prototype = _tree_nbytes(state.prototype_state)
        metadata = _tree_nbytes(state.authority_metadata)
        configured_imagined = (
            self._prototype.config.n_dreams_per_step
            if self._prototype._world_model is not None
            and self._prototype._buffer is not None
            and self._prototype._dreamer is not None
            else 0
        )
        option_search_updates = (
            self._prototype.config.option_search_control.backup_budget
            if self._prototype.config.option_search_control is not None
            else 0
        )
        return PrototypeOptionAuthorityBridgeResourceBudget(
            persistent_state_nbytes=total,
            prototype_state_nbytes=prototype,
            detached_authority_metadata_nbytes=metadata,
            bridge_binding_nbytes=total - prototype - metadata,
            persistent_stomp_state_owners=1,
            detached_metadata_stomp_state_owners=0,
            borrowed_stomp_bindings=1,
            persistent_prepared_transactions=0,
            real_control_stomp_updates_per_ordinary_transition=1,
            configured_imagined_stomp_updates_per_ordinary_transition=(
                configured_imagined
            ),
            max_total_stomp_updates_per_ordinary_transition=(
                1 + configured_imagined
            ),
            option_search_learner_updates_per_ordinary_transition=(
                option_search_updates
            ),
            stomp_internal_planning_backups_per_ordinary_transition=(
                self._prototype.config.oak.stomp.option_planning_backups_per_step
            ),
            stomp_updates_per_audit_adoption=0,
            stomp_updates_per_retirement_transaction=0,
            stomp_updates_per_replacement_transaction=0,
            retirement_preparation_recomputations_per_commit=1,
            replacement_preparation_recomputations_per_commit=1,
            retirement_prepare_host_only=False,
            retirement_commit_host_only=False,
            replacement_prepare_host_only=True,
            replacement_commit_host_only=True,
            derivation_recomputed_on_audit_adoption=False,
            caller_authority_required=True,
            caller_authenticated=False,
            checksum_authenticated=False,
            assessment=PROTOTYPE_OPTION_AUTHORITY_BRIDGE_ASSESSMENT,
            benefit_claim=False,
            evidence_authority=False,
            promotion_authority=False,
            safety_authority=False,
            go_no_go_authority=False,
            retirement_authority=False,
            discovery_authority=False,
            dispatch_authority=False,
            autonomous_curation_authority=False,
            scientific_promotion_allowed=False,
            checkpoint_schema=PROTOTYPE_OPTION_AUTHORITY_BRIDGE_CHECKPOINT_SCHEMA,
        )

    @staticmethod
    def _state_sha256(state: PrototypeOptionAuthorityBridgeState) -> str:
        digest = hashlib.sha256()
        for leaf in jax.tree_util.tree_leaves(state):
            array = jnp.asarray(leaf)
            if jax.dtypes.issubdtype(array.dtype, jax.dtypes.prng_key):
                array = jr.key_data(array)
            host = np.asarray(jax.device_get(array))
            digest.update(host.dtype.str.encode("ascii"))
            digest.update(np.asarray(host.shape, dtype=np.int64).tobytes())
            digest.update(host.tobytes(order="C"))
        return digest.hexdigest()

    def checkpoint_payload(
        self,
        state: PrototypeOptionAuthorityBridgeState,
    ) -> dict[str, object]:
        """Return a host-only exact checkpoint; its hash is not authentication."""

        if not bool(jax.device_get(self.state_valid(state))):
            raise ValueError("cannot checkpoint an invalid bridge state")
        return {
            "schema_version": PROTOTYPE_OPTION_AUTHORITY_BRIDGE_CHECKPOINT_SCHEMA,
            "prototype_config": self._prototype.to_config(),
            "authority_config": self._authority.to_config(),
            "state": state,
            "state_sha256": self._state_sha256(state),
            "checksum_authenticated": False,
            "caller_authenticated": False,
        }

    def restore_checkpoint(
        self,
        payload: object,
    ) -> PrototypeOptionAuthorityBridgeState:
        """Restore one exact host checkpoint under the current composition."""

        if type(payload) is not dict:
            raise TypeError("bridge checkpoint payload must be an exact dict")
        raw = cast(dict[str, object], payload)
        expected = {
            "schema_version",
            "prototype_config",
            "authority_config",
            "state",
            "state_sha256",
            "checksum_authenticated",
            "caller_authenticated",
        }
        if set(raw) != expected:
            raise ValueError("bridge checkpoint fields differ from v1")
        if raw["schema_version"] != PROTOTYPE_OPTION_AUTHORITY_BRIDGE_CHECKPOINT_SCHEMA:
            raise ValueError("bridge checkpoint schema differs")
        if raw["prototype_config"] != self._prototype.to_config():
            raise ValueError("bridge checkpoint Prototype config differs")
        if raw["authority_config"] != self._authority.to_config():
            raise ValueError("bridge checkpoint authority config differs")
        if raw["checksum_authenticated"] is not False:
            raise ValueError("bridge checkpoint cannot claim checksum authentication")
        if raw["caller_authenticated"] is not False:
            raise ValueError("bridge checkpoint cannot claim caller authentication")
        state = raw["state"]
        if type(state) is not PrototypeOptionAuthorityBridgeState:
            raise TypeError("bridge checkpoint state has the wrong exact type")
        restored = state
        state_sha256 = raw["state_sha256"]
        if type(state_sha256) is not str or len(state_sha256) != 64:
            raise TypeError("bridge checkpoint state_sha256 must be a 64-character str")
        if self._state_sha256(restored) != state_sha256:
            raise ValueError("bridge checkpoint state hash differs")
        if not bool(jax.device_get(self.state_valid(restored))):
            raise ValueError("bridge checkpoint state failed its exact contract")
        return restored


__all__ = [
    "PROTOTYPE_OPTION_AUTHORITY_BRIDGE_ASSESSMENT",
    "PROTOTYPE_OPTION_AUTHORITY_BRIDGE_AUTONOMOUS_CURATION_AUTHORITY",
    "PROTOTYPE_OPTION_AUTHORITY_BRIDGE_BENEFIT_CLAIM",
    "PROTOTYPE_OPTION_AUTHORITY_BRIDGE_CHECKPOINT_SCHEMA",
    "PROTOTYPE_OPTION_AUTHORITY_BRIDGE_DISCOVERY_AUTHORITY",
    "PROTOTYPE_OPTION_AUTHORITY_BRIDGE_DISPATCH_AUTHORITY",
    "PROTOTYPE_OPTION_AUTHORITY_BRIDGE_ERROR_DESYNCHRONIZED",
    "PROTOTYPE_OPTION_AUTHORITY_BRIDGE_ERROR_NONE",
    "PROTOTYPE_OPTION_AUTHORITY_BRIDGE_EVIDENCE_AUTHORITY",
    "PROTOTYPE_OPTION_AUTHORITY_BRIDGE_GO_NO_GO_AUTHORITY",
    "PROTOTYPE_OPTION_AUTHORITY_BRIDGE_INITIAL_OWNER_BINDING_SCHEMA",
    "PROTOTYPE_OPTION_AUTHORITY_BRIDGE_PROMOTION_AUTHORITY",
    "PROTOTYPE_OPTION_AUTHORITY_BRIDGE_RETIREMENT_AUTHORITY",
    "PROTOTYPE_OPTION_AUTHORITY_BRIDGE_SAFETY_AUTHORITY",
    "PROTOTYPE_OPTION_AUTHORITY_BRIDGE_SCIENTIFIC_PROMOTION_ALLOWED",
    "PROTOTYPE_OPTION_AUTHORITY_BRIDGE_STATE_SCHEMA",
    "PrototypeOptionAuthorityBridge",
    "PrototypeOptionAuthorityBridgeInitialOwnerBindingReceipt",
    "PrototypeOptionAuthorityBridgeInitialOwnerBindingResult",
    "PrototypeOptionAuthorityBridgeReplacementCommitResult",
    "PrototypeOptionAuthorityBridgeReplacementPrepared",
    "PrototypeOptionAuthorityBridgeResourceBudget",
    "PrototypeOptionAuthorityBridgeRetirementCommitResult",
    "PrototypeOptionAuthorityBridgeRetirementPrepared",
    "PrototypeOptionAuthorityBridgeStartResult",
    "PrototypeOptionAuthorityBridgeState",
    "PrototypeOptionAuthorityBridgeUpdateResult",
]
