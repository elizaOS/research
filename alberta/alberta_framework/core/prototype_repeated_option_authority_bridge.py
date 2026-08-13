# mypy: disable-error-code="arg-type,attr-defined,call-arg,type-var"
"""Versioned single-owner Prototype bridge for repeated option cycles.

The v1 :mod:`prototype_option_authority_bridge` remains unchanged.  This v2
adapter composes it with :class:`RepeatedOptionLifecycle` while preserving one
and only one persistent Prototype→OaK→STOMP owner.  The v1 bridge retains the
canonical detached one-shot authority metadata.  This module persists only a
small repeated-cycle overlay bound to that borrowed child; the overlay contains
no ``STOMPState`` and no duplicate scheduler/installation subtree.

Retirement and replacement first reconstruct a transient repeated state around
Prototype's exact STOMP owner.  Every accepted reset is source-bound and must
rebind the resulting STOMP option slots into OaK before the detached metadata
and cycle overlay are atomically adopted.  Declined replacement authority
retains the cold mask and active cycle key while adopting only the ordinary
discovery/incumbent advance.  Stale and cross-cycle wrapper receipts are exact
no-ops.

Ordinary control delegates once to the v1 bridge.  Consequently Prototype's
raw ``STOMPUpdateResult`` is consumed by the lifecycle audit exactly once with
zero STOMP reevaluation, and every optional Prototype sidecar passes through
unchanged.  If the repeated overlay cannot follow an otherwise valid Prototype
transition, control is not rolled back: the adapter latches desynchronization,
retains the last valid overlay, and refuses later lifecycle authority work.

Validation and ordinary control are array-only.  Retirement and replacement
prepare/commit, checkpointing, and resource measurement are explicit host
orchestration; the authority boundary intentionally prioritizes complete
derivation replay over a very large compiled graph.  Checksums and receipts are
integrity bindings, not authentication.  This L0 mechanism is ``not_assessed``
and owns no safety, go/no-go, retirement, replacement, discovery, dispatch,
evidence, promotion, or autonomous-curation authority.
"""

from __future__ import annotations

import dataclasses
import hashlib
from typing import Any, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array
from jaxtyping import Bool, Int, UInt

from alberta_framework.core.authorized_option_replacement import (
    AuthorizedOptionReplacementState,
)
from alberta_framework.core.authorized_option_retirement import (
    OptionRetirementAuthorityReceipt,
)
from alberta_framework.core.cumulant_option_installation import (
    CumulantOptionLiveInputs,
)
from alberta_framework.core.cumulant_option_scheduler import (
    CumulantOptionInstallationAuthorityReceipt,
    CumulantOptionRetirementHandoff,
    CumulantOptionSchedulerArmInputs,
    CumulantOptionSchedulerObservation,
)
from alberta_framework.core.oak import OaKOptionSlotRebindResult, OaKState
from alberta_framework.core.prototype_agent import (
    PrototypeCandidateUpdateAuditEvidence,
    PrototypeExperientialMemoryInput,
    PrototypeGradientJoyEvidence,
    PrototypePartnerPolicyFusionFeedback,
    PrototypePartnerPolicyFusionInput,
    PrototypeTransition,
)
from alberta_framework.core.prototype_option_authority_bridge import (
    PrototypeOptionAuthorityBridge,
    PrototypeOptionAuthorityBridgeStartResult,
    PrototypeOptionAuthorityBridgeState,
    PrototypeOptionAuthorityBridgeUpdateResult,
    _checksum_arrays,
    _prototype_oak_state,
    _replace_prototype_oak_state,
    _tree_exact_equal,
    _tree_nbytes,
)
from alberta_framework.core.repeated_option_lifecycle import (
    REPEATED_OPTION_LIFECYCLE_ERROR_CAPACITY,
    REPEATED_OPTION_LIFECYCLE_ERROR_NONE,
    RepeatedOptionLifecycle,
    RepeatedOptionLifecycleArm,
    RepeatedOptionLifecycleCommitResult,
    RepeatedOptionLifecyclePrepared,
    RepeatedOptionLifecycleRetirementResult,
    RepeatedOptionLifecycleState,
    RepeatedOptionReplacementAuthorityReceipt,
    RepeatedOptionRetirementAuthorityReceipt,
)

PROTOTYPE_REPEATED_OPTION_AUTHORITY_BRIDGE_STATE_SCHEMA = (
    "alberta.prototype-option-authority-bridge.state.v2"
)
PROTOTYPE_REPEATED_OPTION_AUTHORITY_BRIDGE_CHECKPOINT_SCHEMA = (
    "alberta.prototype-option-authority-bridge.checkpoint.v2"
)
PROTOTYPE_REPEATED_OPTION_AUTHORITY_BRIDGE_ASSESSMENT = "not_assessed"
PROTOTYPE_REPEATED_OPTION_AUTHORITY_BRIDGE_ERROR_NONE = 0
PROTOTYPE_REPEATED_OPTION_AUTHORITY_BRIDGE_ERROR_DESYNCHRONIZED = 1
PROTOTYPE_REPEATED_OPTION_AUTHORITY_BRIDGE_BENEFIT_CLAIM = False
PROTOTYPE_REPEATED_OPTION_AUTHORITY_BRIDGE_EVIDENCE_AUTHORITY = False
PROTOTYPE_REPEATED_OPTION_AUTHORITY_BRIDGE_PROMOTION_AUTHORITY = False
PROTOTYPE_REPEATED_OPTION_AUTHORITY_BRIDGE_SAFETY_AUTHORITY = False
PROTOTYPE_REPEATED_OPTION_AUTHORITY_BRIDGE_GO_NO_GO_AUTHORITY = False
PROTOTYPE_REPEATED_OPTION_AUTHORITY_BRIDGE_RETIREMENT_AUTHORITY = False
PROTOTYPE_REPEATED_OPTION_AUTHORITY_BRIDGE_REPLACEMENT_AUTHORITY = False
PROTOTYPE_REPEATED_OPTION_AUTHORITY_BRIDGE_DISCOVERY_AUTHORITY = False
PROTOTYPE_REPEATED_OPTION_AUTHORITY_BRIDGE_DISPATCH_AUTHORITY = False
PROTOTYPE_REPEATED_OPTION_AUTHORITY_BRIDGE_AUTONOMOUS_CURATION_AUTHORITY = False
PROTOTYPE_REPEATED_OPTION_AUTHORITY_BRIDGE_SCIENTIFIC_PROMOTION_ALLOWED = False

_INT32_MAX = 2**31 - 1


def _saturating_increment(value: Array) -> Array:
    return jnp.where(value < _INT32_MAX, value + jnp.int32(1), value)


@chex.dataclass(frozen=True)
class PrototypeRepeatedOptionLifecycleMetadata:
    """Repeated-cycle overlay borrowing the v1 bridge's exact child owner."""

    completed_cycles: Int[Array, ""]
    total_retirements: Int[Array, ""]
    total_replacements: Int[Array, ""]
    cycle_key_active: Bool[Array, ""]
    active_cycle_key_data: UInt[Array, " 2"]
    has_completed_cycle: Bool[Array, ""]
    cycle_key_history: UInt[Array, "max_cycles 2"]
    last_completed_cycle_key_data: UInt[Array, " 2"]
    last_retirement_authority_revision_words: UInt[Array, " 2"]
    last_replacement_authority_revision_words: UInt[Array, " 2"]
    lifecycle_revision: Int[Array, ""]
    unavailable: Bool[Array, ""]
    error: Int[Array, ""]
    source_repeated_checksum: UInt[Array, " 2"]
    source_child_checksum: UInt[Array, " 2"]
    metadata_checksum: UInt[Array, " 2"]


@chex.dataclass(frozen=True)
class PrototypeRepeatedOptionAuthorityBridgeState:
    """One v1 bridge owner plus detached repeated-cycle overlay."""

    bridge_state: PrototypeOptionAuthorityBridgeState
    lifecycle_metadata: PrototypeRepeatedOptionLifecycleMetadata
    repeated_synchronized: Bool[Array, ""]
    repeated_error: Int[Array, ""]
    revision: Int[Array, ""]
    binding_checksum: UInt[Array, " 2"]


@chex.dataclass(frozen=True)
class PrototypeRepeatedOptionRetirementPrepared:
    """Transient repeated retirement projected through exact OaK rebind."""

    source_state: PrototypeRepeatedOptionAuthorityBridgeState
    handoff: CumulantOptionRetirementHandoff
    authority_receipt: RepeatedOptionRetirementAuthorityReceipt
    cycle_key: Array
    phase_one_key: Array
    phase_two_key: Array
    lifecycle_result: RepeatedOptionLifecycleRetirementResult
    oak_rebind: OaKOptionSlotRebindResult
    proposed_state: PrototypeRepeatedOptionAuthorityBridgeState
    reset_slots: Bool[Array, " option_budget"]
    source_binding_valid: Bool[Array, ""]
    exact_owner_rebind: Bool[Array, ""]
    cold_mask_applied: Bool[Array, ""]
    preparation_valid: Bool[Array, ""]
    prepared_checksum: UInt[Array, " 2"]


@chex.dataclass(frozen=True)
class PrototypeRepeatedOptionRetirementCommitResult:
    """Atomic source/derivation-bound repeated retirement adoption."""

    state: PrototypeRepeatedOptionAuthorityBridgeState
    lifecycle_result: RepeatedOptionLifecycleRetirementResult
    oak_rebind: OaKOptionSlotRebindResult
    destination_matches_source: Bool[Array, ""]
    prepared_integrity_valid: Bool[Array, ""]
    preparation_derivation_valid: Bool[Array, ""]
    exact_owner_rebind: Bool[Array, ""]
    cold_mask_applied: Bool[Array, ""]
    transaction_applied: Bool[Array, ""]
    caller_authenticated: Bool[Array, ""]


@chex.dataclass(frozen=True)
class PrototypeRepeatedOptionReplacementPrepared:
    """Transient host preparation bound to bridge, cycle, arm, and inputs."""

    source_state: PrototypeRepeatedOptionAuthorityBridgeState
    arm: RepeatedOptionLifecycleArm
    observation: CumulantOptionSchedulerObservation
    live_inputs: CumulantOptionLiveInputs
    lifecycle_prepared: RepeatedOptionLifecyclePrepared
    source_binding_valid: Bool[Array, ""]
    preparation_valid: Bool[Array, ""]
    prepared_checksum: UInt[Array, " 2"]


@dataclasses.dataclass(frozen=True, slots=True)
class PrototypeRepeatedOptionReplacementCommitResult:
    """Host replacement/decline result; lower result is absent on refusal."""

    state: PrototypeRepeatedOptionAuthorityBridgeState
    lifecycle_result: RepeatedOptionLifecycleCommitResult | None
    oak_rebind: OaKOptionSlotRebindResult | None
    destination_matches_source: bool
    prepared_integrity_valid: bool
    preparation_derivation_valid: bool
    exact_owner_rebind: bool
    cold_mask_applied: bool
    ordinary_advance_applied: bool
    replacement_applied: bool
    cycle_completed: bool
    transaction_applied: bool
    caller_authenticated: bool


@chex.dataclass(frozen=True)
class PrototypeRepeatedOptionAuthorityBridgeStartResult:
    """One v1 start plus repeated-overlay adoption/desynchronization."""

    state: PrototypeRepeatedOptionAuthorityBridgeState
    bridge: PrototypeOptionAuthorityBridgeStartResult
    repeated_metadata_advanced: Bool[Array, ""]
    repeated_desynchronized: Bool[Array, ""]
    transaction_applied: Bool[Array, ""]


@chex.dataclass(frozen=True)
class PrototypeRepeatedOptionAuthorityBridgeUpdateResult:
    """One unchanged v1 Prototype update plus repeated-overlay adoption."""

    state: PrototypeRepeatedOptionAuthorityBridgeState
    bridge: PrototypeOptionAuthorityBridgeUpdateResult
    repeated_metadata_advanced: Bool[Array, ""]
    repeated_desynchronized: Bool[Array, ""]
    control_transition_rolled_back_by_adapter: Bool[Array, ""]
    transaction_applied: Bool[Array, ""]


@dataclasses.dataclass(frozen=True, slots=True)
class PrototypeRepeatedOptionAuthorityBridgeResourceBudget:
    """Exact persistence, ownership, work, and host-boundary declaration."""

    persistent_state_nbytes: int
    v1_bridge_state_nbytes: int
    prototype_state_nbytes: int
    detached_authority_metadata_nbytes: int
    repeated_overlay_nbytes: int
    adapter_binding_nbytes: int
    persistent_stomp_state_owners: int
    detached_authority_metadata_stomp_state_owners: int
    repeated_overlay_stomp_state_owners: int
    borrowed_stomp_bindings: int
    persistent_prepared_transactions: int
    max_cycles: int
    completed_cycles: int
    remaining_cycles: int
    active_cycle: bool
    real_control_stomp_updates_per_ordinary_transition: int
    configured_imagined_stomp_updates_per_ordinary_transition: int
    max_total_stomp_updates_per_ordinary_transition: int
    stomp_updates_per_audit_adoption: int
    stomp_updates_per_retirement_transaction: int
    stomp_updates_per_replacement_transaction: int
    retirement_prepare_host_only: bool
    retirement_commit_host_only: bool
    replacement_prepare_host_only: bool
    replacement_commit_host_only: bool
    checkpoint_host_only: bool
    caller_authenticated: bool
    checksum_authenticated: bool
    assessment: str
    benefit_claim: bool
    evidence_authority: bool
    promotion_authority: bool
    safety_authority: bool
    go_no_go_authority: bool
    retirement_authority: bool
    replacement_authority: bool
    discovery_authority: bool
    dispatch_authority: bool
    autonomous_curation_authority: bool
    scientific_promotion_allowed: bool
    checkpoint_schema: str


class PrototypeRepeatedOptionAuthorityBridge:
    """Compose the v1 single-owner bridge with a detached repeated overlay."""

    def __init__(
        self,
        bridge: PrototypeOptionAuthorityBridge,
        lifecycle: RepeatedOptionLifecycle,
    ) -> None:
        if type(bridge) is not PrototypeOptionAuthorityBridge:
            raise TypeError("bridge must be an exact PrototypeOptionAuthorityBridge")
        if type(lifecycle) is not RepeatedOptionLifecycle:
            raise TypeError("lifecycle must be an exact RepeatedOptionLifecycle")
        if bridge.authority is not lifecycle.replacement:
            raise ValueError("v1 bridge and repeated lifecycle must share one authority")
        self._bridge = bridge
        self._lifecycle = lifecycle
        self._authority = bridge.authority
        self._prototype = bridge.prototype
        self._oak = bridge._oak

    @property
    def bridge(self) -> PrototypeOptionAuthorityBridge:
        return self._bridge

    @property
    def lifecycle(self) -> RepeatedOptionLifecycle:
        return self._lifecycle

    def to_config(self) -> dict[str, object]:
        return {
            "schema_version": PROTOTYPE_REPEATED_OPTION_AUTHORITY_BRIDGE_STATE_SCHEMA,
            "prototype": self._prototype.to_config(),
            "authority": self._authority.to_config(),
            "repeated_lifecycle": self._lifecycle.to_config(),
            "persistent_stomp_state_owners": 1,
            "repeated_overlay_stomp_state_owners": 0,
            "assessment": PROTOTYPE_REPEATED_OPTION_AUTHORITY_BRIDGE_ASSESSMENT,
            "benefit_claim": False,
            "evidence_authority": False,
            "promotion_authority": False,
            "safety_authority": False,
            "go_no_go_authority": False,
            "retirement_authority": False,
            "replacement_authority": False,
            "discovery_authority": False,
            "dispatch_authority": False,
            "autonomous_curation_authority": False,
            "scientific_promotion_allowed": False,
        }

    def _overlay_payload_arrays(
        self,
        metadata: PrototypeRepeatedOptionLifecycleMetadata,
    ) -> tuple[Array, ...]:
        return tuple(
            cast(Array, leaf)
            for leaf in jax.tree_util.tree_leaves(
                tuple(
                    getattr(metadata, field.name)
                    for field in dataclasses.fields(PrototypeRepeatedOptionLifecycleMetadata)
                    if field.name != "metadata_checksum"
                )
            )
        )

    def _with_overlay_checksum(
        self,
        metadata: PrototypeRepeatedOptionLifecycleMetadata,
    ) -> PrototypeRepeatedOptionLifecycleMetadata:
        return dataclasses.replace(
            metadata,
            metadata_checksum=_checksum_arrays(self._overlay_payload_arrays(metadata)),
        )

    def _detach_lifecycle(
        self,
        state: RepeatedOptionLifecycleState,
    ) -> PrototypeRepeatedOptionLifecycleMetadata:
        return self._with_overlay_checksum(
            PrototypeRepeatedOptionLifecycleMetadata(
                completed_cycles=state.completed_cycles,
                total_retirements=state.total_retirements,
                total_replacements=state.total_replacements,
                cycle_key_active=state.cycle_key_active,
                active_cycle_key_data=state.active_cycle_key_data,
                has_completed_cycle=state.has_completed_cycle,
                cycle_key_history=state.cycle_key_history,
                last_completed_cycle_key_data=(state.last_completed_cycle_key_data),
                last_retirement_authority_revision_words=(
                    state.last_retirement_authority_revision_words
                ),
                last_replacement_authority_revision_words=(
                    state.last_replacement_authority_revision_words
                ),
                lifecycle_revision=state.revision,
                unavailable=state.unavailable,
                error=state.error,
                source_repeated_checksum=state.binding_checksum,
                source_child_checksum=state.cycle_state.binding_checksum,
                metadata_checksum=jnp.zeros((2,), dtype=jnp.uint32),
            )
        )

    def _check_overlay_contract(
        self,
        metadata: PrototypeRepeatedOptionLifecycleMetadata,
    ) -> None:
        if type(metadata) is not PrototypeRepeatedOptionLifecycleMetadata:
            raise TypeError("lifecycle_metadata has the wrong exact type")
        max_cycles = self._lifecycle.config.max_cycles
        contracts = (
            (metadata.completed_cycles, "completed_cycles", (), jnp.int32),
            (metadata.total_retirements, "total_retirements", (), jnp.int32),
            (metadata.total_replacements, "total_replacements", (), jnp.int32),
            (metadata.cycle_key_active, "cycle_key_active", (), jnp.bool_),
            (metadata.active_cycle_key_data, "active_cycle_key_data", (2,), jnp.uint32),
            (metadata.has_completed_cycle, "has_completed_cycle", (), jnp.bool_),
            (metadata.cycle_key_history, "cycle_key_history", (max_cycles, 2), jnp.uint32),
            (
                metadata.last_completed_cycle_key_data,
                "last_completed_cycle_key_data",
                (2,),
                jnp.uint32,
            ),
            (
                metadata.last_retirement_authority_revision_words,
                "last_retirement_authority_revision_words",
                (2,),
                jnp.uint32,
            ),
            (
                metadata.last_replacement_authority_revision_words,
                "last_replacement_authority_revision_words",
                (2,),
                jnp.uint32,
            ),
            (metadata.lifecycle_revision, "lifecycle_revision", (), jnp.int32),
            (metadata.unavailable, "unavailable", (), jnp.bool_),
            (metadata.error, "error", (), jnp.int32),
            (metadata.source_repeated_checksum, "source_repeated_checksum", (2,), jnp.uint32),
            (metadata.source_child_checksum, "source_child_checksum", (2,), jnp.uint32),
            (metadata.metadata_checksum, "metadata_checksum", (2,), jnp.uint32),
        )
        for value, name, shape, dtype in contracts:
            array = jnp.asarray(value)
            if array.shape != shape:
                raise ValueError(f"lifecycle_metadata.{name} must have shape {shape}")
            if array.dtype != dtype:
                raise TypeError(f"lifecycle_metadata.{name} must have dtype {dtype}")

    def _overlay_integrity_valid(
        self,
        metadata: PrototypeRepeatedOptionLifecycleMetadata,
    ) -> Array:
        self._check_overlay_contract(metadata)
        max_cycles = self._lifecycle.config.max_cycles
        completed = metadata.completed_cycles
        active = metadata.cycle_key_active.astype(jnp.int32)
        indices = jnp.arange(max_cycles, dtype=jnp.int32)
        used = indices < completed
        pairwise = jnp.all(
            metadata.cycle_key_history[:, None, :] == metadata.cycle_key_history[None, :, :],
            axis=2,
        )
        duplicate = jnp.any(
            pairwise & used[:, None] & used[None, :] & (~jnp.eye(max_cycles, dtype=jnp.bool_))
        )
        tail_zero = jnp.all(
            jnp.where(
                used[:, None],
                jnp.zeros_like(metadata.cycle_key_history),
                metadata.cycle_key_history,
            )
            == 0
        )
        last_key_valid = jnp.where(
            completed > 0,
            jnp.array_equal(
                metadata.last_completed_cycle_key_data,
                metadata.cycle_key_history[jnp.clip(completed - 1, 0, max_cycles - 1)],
            ),
            jnp.all(metadata.last_completed_cycle_key_data == 0),
        )
        retired_revision_bound = jnp.where(
            metadata.total_retirements == 0,
            jnp.all(metadata.last_retirement_authority_revision_words == 0),
            jnp.any(metadata.last_retirement_authority_revision_words != 0),
        )
        replacement_revision_bound = jnp.where(
            metadata.total_replacements == 0,
            jnp.all(metadata.last_replacement_authority_revision_words == 0),
            jnp.any(metadata.last_replacement_authority_revision_words != 0),
        )
        active_key_bound = jnp.where(
            metadata.cycle_key_active,
            jnp.any(metadata.active_cycle_key_data != 0),
            jnp.all(metadata.active_cycle_key_data == 0),
        )
        active_key_is_fresh = (~metadata.cycle_key_active) | (
            ~jnp.any(
                used
                & jnp.all(
                    metadata.cycle_key_history == metadata.active_cycle_key_data[None, :],
                    axis=1,
                )
            )
        )
        expected_error = jnp.where(
            completed == max_cycles,
            jnp.asarray(REPEATED_OPTION_LIFECYCLE_ERROR_CAPACITY, dtype=jnp.int32),
            jnp.asarray(REPEATED_OPTION_LIFECYCLE_ERROR_NONE, dtype=jnp.int32),
        )
        return (
            (completed >= 0)
            & (completed <= max_cycles)
            & (metadata.total_replacements == completed)
            & (metadata.total_retirements == completed + active)
            & (metadata.total_retirements <= max_cycles)
            & (metadata.has_completed_cycle == (completed > 0))
            & (~duplicate)
            & tail_zero
            & last_key_valid
            & retired_revision_bound
            & replacement_revision_bound
            & active_key_bound
            & active_key_is_fresh
            & (
                metadata.lifecycle_revision
                >= metadata.total_retirements + metadata.total_replacements
            )
            & (metadata.unavailable == (completed == max_cycles))
            & (metadata.error == expected_error)
            & jnp.array_equal(
                metadata.metadata_checksum,
                _checksum_arrays(self._overlay_payload_arrays(metadata)),
            )
        )

    def _attach_lifecycle(
        self,
        metadata: PrototypeRepeatedOptionLifecycleMetadata,
        child: AuthorizedOptionReplacementState,
    ) -> tuple[RepeatedOptionLifecycleState, Array]:
        self._check_overlay_contract(metadata)
        state = RepeatedOptionLifecycleState(
            cycle_state=child,
            completed_cycles=metadata.completed_cycles,
            total_retirements=metadata.total_retirements,
            total_replacements=metadata.total_replacements,
            cycle_key_active=metadata.cycle_key_active,
            active_cycle_key_data=metadata.active_cycle_key_data,
            has_completed_cycle=metadata.has_completed_cycle,
            cycle_key_history=metadata.cycle_key_history,
            last_completed_cycle_key_data=metadata.last_completed_cycle_key_data,
            last_retirement_authority_revision_words=(
                metadata.last_retirement_authority_revision_words
            ),
            last_replacement_authority_revision_words=(
                metadata.last_replacement_authority_revision_words
            ),
            revision=metadata.lifecycle_revision,
            unavailable=metadata.unavailable,
            error=metadata.error,
            binding_checksum=metadata.source_repeated_checksum,
        )
        valid = (
            self._overlay_integrity_valid(metadata)
            & jnp.array_equal(child.binding_checksum, metadata.source_child_checksum)
            & self._lifecycle.state_valid(state)
        )
        return state, valid

    def _payload_arrays(
        self,
        state: PrototypeRepeatedOptionAuthorityBridgeState,
    ) -> tuple[Array, ...]:
        return tuple(
            cast(Array, leaf)
            for leaf in jax.tree_util.tree_leaves(
                (
                    state.bridge_state,
                    state.lifecycle_metadata,
                    state.repeated_synchronized,
                    state.repeated_error,
                    state.revision,
                )
            )
        )

    def _with_checksum(
        self,
        state: PrototypeRepeatedOptionAuthorityBridgeState,
    ) -> PrototypeRepeatedOptionAuthorityBridgeState:
        return dataclasses.replace(
            state,
            binding_checksum=_checksum_arrays(self._payload_arrays(state)),
        )

    def _check_state_contract(
        self,
        state: PrototypeRepeatedOptionAuthorityBridgeState,
    ) -> None:
        if type(state) is not PrototypeRepeatedOptionAuthorityBridgeState:
            raise TypeError("state has the wrong exact v2 bridge type")
        self._bridge._check_state_contract(state.bridge_state)
        self._check_overlay_contract(state.lifecycle_metadata)
        for value, name, dtype in (
            (state.repeated_synchronized, "repeated_synchronized", jnp.bool_),
            (state.repeated_error, "repeated_error", jnp.int32),
            (state.revision, "revision", jnp.int32),
        ):
            array = jnp.asarray(value)
            if array.shape != ():
                raise ValueError(f"state.{name} must be scalar")
            if array.dtype != dtype:
                raise TypeError(f"state.{name} has the wrong dtype")
        checksum = jnp.asarray(state.binding_checksum)
        if checksum.shape != (2,) or checksum.dtype != jnp.uint32:
            raise TypeError("state.binding_checksum must be uint32[2]")

    def _attach_source(
        self,
        state: PrototypeRepeatedOptionAuthorityBridgeState,
    ) -> tuple[RepeatedOptionLifecycleState, Array]:
        oak = _prototype_oak_state(state.bridge_state.prototype_state.oak_state)
        child = self._authority.attach_borrowed_stomp(
            state.bridge_state.authority_metadata,
            oak.stomp_state,
        )
        repeated, overlay_valid = self._attach_lifecycle(
            state.lifecycle_metadata,
            child.state,
        )
        return repeated, child.transaction_applied & overlay_valid

    def init(
        self,
        bridge_state: PrototypeOptionAuthorityBridgeState,
        lifecycle_state: RepeatedOptionLifecycleState,
    ) -> PrototypeRepeatedOptionAuthorityBridgeState:
        """Bind an already-initialized v1 bridge to its exact repeated child."""

        if type(bridge_state) is not PrototypeOptionAuthorityBridgeState:
            raise TypeError("bridge_state must be an exact v1 bridge state")
        if type(lifecycle_state) is not RepeatedOptionLifecycleState:
            raise TypeError("lifecycle_state must be exact")
        if not bool(jax.device_get(self._bridge.state_valid(bridge_state))):
            raise ValueError("bridge_state must satisfy the v1 contract")
        if not bool(jax.device_get(self._lifecycle.state_valid(lifecycle_state))):
            raise ValueError("lifecycle_state must satisfy the repeated contract")
        oak = _prototype_oak_state(bridge_state.prototype_state.oak_state)
        borrowed = self._authority.attach_borrowed_stomp(
            bridge_state.authority_metadata,
            oak.stomp_state,
        )
        sources_match = borrowed.transaction_applied & _tree_exact_equal(
            borrowed.state,
            lifecycle_state.cycle_state,
        )
        if not bool(jax.device_get(sources_match)):
            raise ValueError("v1 bridge and repeated lifecycle child differ")
        state = PrototypeRepeatedOptionAuthorityBridgeState(
            bridge_state=bridge_state,
            lifecycle_metadata=self._detach_lifecycle(lifecycle_state),
            repeated_synchronized=jnp.asarray(True, dtype=jnp.bool_),
            repeated_error=jnp.asarray(
                PROTOTYPE_REPEATED_OPTION_AUTHORITY_BRIDGE_ERROR_NONE,
                dtype=jnp.int32,
            ),
            revision=jnp.asarray(0, dtype=jnp.int32),
            binding_checksum=jnp.zeros((2,), dtype=jnp.uint32),
        )
        state = self._with_checksum(state)
        if not bool(jax.device_get(self.state_valid(state))):
            raise ValueError("initialized v2 bridge state failed its contract")
        return state

    def state_valid(
        self,
        state: PrototypeRepeatedOptionAuthorityBridgeState,
    ) -> Bool[Array, ""]:
        """Validate one owner, overlay integrity, synchronization, and checksum."""

        self._check_state_contract(state)
        _, attached = self._attach_source(state)
        error_contract = (
            state.repeated_synchronized
            == (state.repeated_error == PROTOTYPE_REPEATED_OPTION_AUTHORITY_BRIDGE_ERROR_NONE)
        ) & (
            (state.repeated_error == PROTOTYPE_REPEATED_OPTION_AUTHORITY_BRIDGE_ERROR_NONE)
            | (
                state.repeated_error
                == PROTOTYPE_REPEATED_OPTION_AUTHORITY_BRIDGE_ERROR_DESYNCHRONIZED
            )
        )
        return (
            self._bridge.state_valid(state.bridge_state)
            & self._overlay_integrity_valid(state.lifecycle_metadata)
            & error_contract
            & (state.revision >= 0)
            & jnp.where(
                state.repeated_synchronized,
                attached & state.bridge_state.authority_synchronized,
                jnp.asarray(True, dtype=jnp.bool_),
            )
            & jnp.array_equal(
                state.binding_checksum,
                _checksum_arrays(self._payload_arrays(state)),
            )
        )

    def _next_state(
        self,
        source: PrototypeRepeatedOptionAuthorityBridgeState,
        bridge_state: PrototypeOptionAuthorityBridgeState,
        metadata: PrototypeRepeatedOptionLifecycleMetadata,
        *,
        synchronized: Array,
    ) -> PrototypeRepeatedOptionAuthorityBridgeState:
        error = jnp.where(
            synchronized,
            jnp.asarray(PROTOTYPE_REPEATED_OPTION_AUTHORITY_BRIDGE_ERROR_NONE, jnp.int32),
            jnp.asarray(
                PROTOTYPE_REPEATED_OPTION_AUTHORITY_BRIDGE_ERROR_DESYNCHRONIZED,
                jnp.int32,
            ),
        )
        return self._with_checksum(
            PrototypeRepeatedOptionAuthorityBridgeState(
                bridge_state=bridge_state,
                lifecycle_metadata=metadata,
                repeated_synchronized=synchronized,
                repeated_error=error,
                revision=_saturating_increment(source.revision),
                binding_checksum=jnp.zeros((2,), dtype=jnp.uint32),
            )
        )

    def _compose_repeated_destination(
        self,
        source: PrototypeRepeatedOptionAuthorityBridgeState,
        repeated: RepeatedOptionLifecycleState,
        prototype_state: Any,
    ) -> PrototypeRepeatedOptionAuthorityBridgeState:
        authority_metadata = self._authority.detach_borrowed_stomp(repeated.cycle_state)
        bridge_state = self._bridge._next_state(
            source.bridge_state,
            prototype_state,
            authority_metadata,
            synchronized=jnp.asarray(True, dtype=jnp.bool_),
        )
        return self._next_state(
            source,
            bridge_state,
            self._detach_lifecycle(repeated),
            synchronized=jnp.asarray(True, dtype=jnp.bool_),
        )

    def retirement_authority_receipt(
        self,
        state: PrototypeRepeatedOptionAuthorityBridgeState,
        retirement_authority: OptionRetirementAuthorityReceipt,
        cycle_key: Array,
    ) -> RepeatedOptionRetirementAuthorityReceipt:
        """Bind a lower retirement receipt to the exact synchronized v2 source."""

        self._check_state_contract(state)
        repeated, attached = self._attach_source(state)
        if not bool(
            jax.device_get(self.state_valid(state) & state.repeated_synchronized & attached)
        ):
            raise ValueError("v2 bridge is not synchronized for retirement authority")
        return self._lifecycle.retirement_authority_receipt(
            repeated,
            retirement_authority,
            cycle_key,
        )

    def _retirement_payload_arrays(
        self,
        prepared: PrototypeRepeatedOptionRetirementPrepared,
    ) -> tuple[Array, ...]:
        return tuple(
            cast(Array, leaf)
            for leaf in jax.tree_util.tree_leaves(
                tuple(
                    getattr(prepared, field.name)
                    for field in dataclasses.fields(PrototypeRepeatedOptionRetirementPrepared)
                    if field.name != "prepared_checksum"
                )
            )
        )

    def _with_retirement_checksum(
        self,
        prepared: PrototypeRepeatedOptionRetirementPrepared,
    ) -> PrototypeRepeatedOptionRetirementPrepared:
        return dataclasses.replace(
            prepared,
            prepared_checksum=_checksum_arrays(self._retirement_payload_arrays(prepared)),
        )

    def prepare_retirement(
        self,
        state: PrototypeRepeatedOptionAuthorityBridgeState,
        handoff: CumulantOptionRetirementHandoff,
        authority_receipt: RepeatedOptionRetirementAuthorityReceipt,
        cycle_key: Array,
        phase_one_key: Array,
        phase_two_key: Array,
    ) -> PrototypeRepeatedOptionRetirementPrepared:
        """Prepare one exact repeated retirement and its OaK owner reset."""

        self._check_state_contract(state)
        repeated, attached = self._attach_source(state)
        source_valid = self.state_valid(state) & state.repeated_synchronized & attached
        lifecycle_result = self._lifecycle.retire(
            repeated,
            handoff,
            authority_receipt,
            cycle_key,
            phase_one_key,
            phase_two_key,
        )
        source_oak = _prototype_oak_state(state.bridge_state.prototype_state.oak_state)
        destination_installation = (
            lifecycle_result.state.cycle_state.scheduler_state.installation_state
        )
        destination_stomp = destination_installation.lifecycle_state.stomp_state
        reset_slots = repeated.cycle_state.installed_slot_mask & (
            ~lifecycle_result.state.cycle_state.installed_slot_mask
        )
        oak_rebind = self._oak.rebind_option_slots(
            source_oak,
            destination_stomp,
            reset_slots,
        )
        prototype_candidate = _replace_prototype_oak_state(
            state.bridge_state.prototype_state,
            oak_rebind.state,
        )
        candidate = self._compose_repeated_destination(
            state,
            lifecycle_result.state,
            prototype_candidate,
        )
        exact_owner = oak_rebind.transaction_applied & _tree_exact_equal(
            _prototype_oak_state(candidate.bridge_state.prototype_state.oak_state).stomp_state,
            destination_stomp,
        )
        cold_mask = jnp.array_equal(
            candidate.bridge_state.extended_action_mask,
            self._lifecycle.extended_action_mask(lifecycle_result.state),
        ) & jnp.array_equal(
            candidate.bridge_state.authority_metadata.installed_slot_mask,
            lifecycle_result.state.cycle_state.installed_slot_mask,
        )
        valid = (
            source_valid
            & lifecycle_result.diagnostics.wrapper_transaction_applied
            & (jnp.sum(reset_slots, dtype=jnp.int32) == 1)
            & exact_owner
            & cold_mask
            & self.state_valid(candidate)
        )
        proposed = cast(
            PrototypeRepeatedOptionAuthorityBridgeState,
            jax.lax.cond(valid, lambda _: candidate, lambda _: state, None),
        )
        prepared = PrototypeRepeatedOptionRetirementPrepared(
            source_state=state,
            handoff=handoff,
            authority_receipt=authority_receipt,
            cycle_key=cycle_key,
            phase_one_key=phase_one_key,
            phase_two_key=phase_two_key,
            lifecycle_result=lifecycle_result,
            oak_rebind=oak_rebind,
            proposed_state=proposed,
            reset_slots=reset_slots,
            source_binding_valid=source_valid,
            exact_owner_rebind=exact_owner,
            cold_mask_applied=cold_mask,
            preparation_valid=valid,
            prepared_checksum=jnp.zeros((2,), dtype=jnp.uint32),
        )
        return self._with_retirement_checksum(prepared)

    def commit_retirement(
        self,
        state: PrototypeRepeatedOptionAuthorityBridgeState,
        prepared: PrototypeRepeatedOptionRetirementPrepared,
    ) -> PrototypeRepeatedOptionRetirementCommitResult:
        """Re-derive and atomically adopt one repeated retirement."""

        self._check_state_contract(state)
        if type(prepared) is not PrototypeRepeatedOptionRetirementPrepared:
            raise TypeError("prepared has the wrong exact retirement type")
        integrity = jnp.array_equal(
            prepared.prepared_checksum,
            _checksum_arrays(self._retirement_payload_arrays(prepared)),
        )
        recomputed = self.prepare_retirement(
            prepared.source_state,
            prepared.handoff,
            prepared.authority_receipt,
            prepared.cycle_key,
            prepared.phase_one_key,
            prepared.phase_two_key,
        )
        derivation = _tree_exact_equal(prepared, recomputed)
        destination_matches = _tree_exact_equal(state, prepared.source_state)
        applied = (
            self.state_valid(state)
            & destination_matches
            & integrity
            & derivation
            & recomputed.preparation_valid
            & recomputed.exact_owner_rebind
            & recomputed.cold_mask_applied
        )
        next_state = cast(
            PrototypeRepeatedOptionAuthorityBridgeState,
            jax.lax.cond(
                applied,
                lambda _: recomputed.proposed_state,
                lambda _: state,
                None,
            ),
        )
        return PrototypeRepeatedOptionRetirementCommitResult(
            state=next_state,
            lifecycle_result=recomputed.lifecycle_result,
            oak_rebind=recomputed.oak_rebind,
            destination_matches_source=destination_matches,
            prepared_integrity_valid=integrity,
            preparation_derivation_valid=derivation,
            exact_owner_rebind=recomputed.exact_owner_rebind,
            cold_mask_applied=recomputed.cold_mask_applied,
            transaction_applied=applied,
            caller_authenticated=jnp.asarray(False, dtype=jnp.bool_),
        )

    def arm(
        self,
        state: PrototypeRepeatedOptionAuthorityBridgeState,
        inputs: CumulantOptionSchedulerArmInputs,
    ) -> RepeatedOptionLifecycleArm:
        """Arm one replacement attempt through the borrowed repeated child."""

        self._check_state_contract(state)
        repeated, attached = self._attach_source(state)
        arm = self._lifecycle.arm(repeated, inputs)
        available = arm.available & self.state_valid(state) & state.repeated_synchronized & attached
        return dataclasses.replace(
            arm,
            replacement_arm=dataclasses.replace(
                arm.replacement_arm,
                available=available,
            ),
            available=available,
        )

    def _replacement_payload_arrays(
        self,
        prepared: PrototypeRepeatedOptionReplacementPrepared,
    ) -> tuple[Array, ...]:
        return tuple(
            cast(Array, leaf)
            for leaf in jax.tree_util.tree_leaves(
                tuple(
                    getattr(prepared, field.name)
                    for field in dataclasses.fields(PrototypeRepeatedOptionReplacementPrepared)
                    if field.name != "prepared_checksum"
                )
            )
        )

    def _with_replacement_checksum(
        self,
        prepared: PrototypeRepeatedOptionReplacementPrepared,
    ) -> PrototypeRepeatedOptionReplacementPrepared:
        return dataclasses.replace(
            prepared,
            prepared_checksum=_checksum_arrays(self._replacement_payload_arrays(prepared)),
        )

    def prepare_replacement(
        self,
        state: PrototypeRepeatedOptionAuthorityBridgeState,
        arm: RepeatedOptionLifecycleArm,
        observation: CumulantOptionSchedulerObservation,
        live_inputs: CumulantOptionLiveInputs,
    ) -> PrototypeRepeatedOptionReplacementPrepared:
        """Stage one source-bound repeated replacement without persistence."""

        self._check_state_contract(state)
        repeated, attached = self._attach_source(state)
        source_valid = self.state_valid(state) & state.repeated_synchronized & attached
        lifecycle_prepared = self._lifecycle.prepare(
            repeated,
            arm,
            observation,
            live_inputs,
        )
        valid = (
            source_valid
            & lifecycle_prepared.replacement_prepared.diagnostics.transaction_valid
            & _tree_exact_equal(
                lifecycle_prepared.replacement_prepared.source_state,
                repeated.cycle_state,
            )
        )
        prepared = PrototypeRepeatedOptionReplacementPrepared(
            source_state=state,
            arm=arm,
            observation=observation,
            live_inputs=live_inputs,
            lifecycle_prepared=lifecycle_prepared,
            source_binding_valid=source_valid,
            preparation_valid=valid,
            prepared_checksum=jnp.zeros((2,), dtype=jnp.uint32),
        )
        return self._with_replacement_checksum(prepared)

    def replacement_authority_receipt(
        self,
        state: PrototypeRepeatedOptionAuthorityBridgeState,
        prepared: PrototypeRepeatedOptionReplacementPrepared,
        installation_authority: CumulantOptionInstallationAuthorityReceipt,
        cycle_key: Array,
        *,
        replacement_authorized: bool | Array,
    ) -> RepeatedOptionReplacementAuthorityReceipt:
        """Create one exact v2-source-bound repeated replacement receipt."""

        self._check_state_contract(state)
        if type(prepared) is not PrototypeRepeatedOptionReplacementPrepared:
            raise TypeError("prepared has the wrong exact replacement type")
        repeated, attached = self._attach_source(state)
        source_match = _tree_exact_equal(state, prepared.source_state)
        integrity = jnp.array_equal(
            prepared.prepared_checksum,
            _checksum_arrays(self._replacement_payload_arrays(prepared)),
        )
        if not bool(
            jax.device_get(
                self.state_valid(state)
                & state.repeated_synchronized
                & attached
                & source_match
                & integrity
                & prepared.preparation_valid
            )
        ):
            raise ValueError("v2 replacement preparation is stale or desynchronized")
        return self._lifecycle.replacement_authority_receipt(
            repeated,
            prepared.lifecycle_prepared,
            installation_authority,
            cycle_key,
            replacement_authorized=replacement_authorized,
        )

    @staticmethod
    def _refused_replacement(
        state: PrototypeRepeatedOptionAuthorityBridgeState,
        *,
        destination_matches: bool,
        integrity: bool,
        derivation: bool,
    ) -> PrototypeRepeatedOptionReplacementCommitResult:
        return PrototypeRepeatedOptionReplacementCommitResult(
            state=state,
            lifecycle_result=None,
            oak_rebind=None,
            destination_matches_source=destination_matches,
            prepared_integrity_valid=integrity,
            preparation_derivation_valid=derivation,
            exact_owner_rebind=False,
            cold_mask_applied=False,
            ordinary_advance_applied=False,
            replacement_applied=False,
            cycle_completed=False,
            transaction_applied=False,
            caller_authenticated=False,
        )

    def commit_replacement(
        self,
        state: PrototypeRepeatedOptionAuthorityBridgeState,
        prepared: PrototypeRepeatedOptionReplacementPrepared,
        authority_receipt: RepeatedOptionReplacementAuthorityReceipt,
        cycle_key: Array,
    ) -> PrototypeRepeatedOptionReplacementCommitResult:
        """Host-rederive, rebind the sole owner, and atomically adopt."""

        self._check_state_contract(state)
        if type(prepared) is not PrototypeRepeatedOptionReplacementPrepared:
            raise TypeError("prepared has the wrong exact replacement type")
        integrity_array = jnp.array_equal(
            prepared.prepared_checksum,
            _checksum_arrays(self._replacement_payload_arrays(prepared)),
        )
        recomputed = self.prepare_replacement(
            prepared.source_state,
            prepared.arm,
            prepared.observation,
            prepared.live_inputs,
        )
        derivation_array = _tree_exact_equal(prepared, recomputed)
        destination_array = _tree_exact_equal(state, prepared.source_state)
        valid = bool(
            jax.device_get(
                self.state_valid(state)
                & state.repeated_synchronized
                & integrity_array
                & derivation_array
                & destination_array
                & recomputed.preparation_valid
            )
        )
        if not valid:
            return self._refused_replacement(
                state,
                destination_matches=bool(jax.device_get(destination_array)),
                integrity=bool(jax.device_get(integrity_array)),
                derivation=bool(jax.device_get(derivation_array)),
            )
        repeated, attached = self._attach_source(state)
        if not bool(jax.device_get(attached)):
            return self._refused_replacement(
                state,
                destination_matches=True,
                integrity=True,
                derivation=True,
            )
        lifecycle_result = self._lifecycle.commit(
            repeated,
            recomputed.lifecycle_prepared,
            authority_receipt,
            cycle_key,
        )
        ordinary = bool(
            jax.device_get(
                lifecycle_result.diagnostics.ordinary_advance_adopted
                | lifecycle_result.diagnostics.cycle_completed
            )
        )
        if not ordinary or lifecycle_result.replacement is None:
            return PrototypeRepeatedOptionReplacementCommitResult(
                state=state,
                lifecycle_result=lifecycle_result,
                oak_rebind=None,
                destination_matches_source=True,
                prepared_integrity_valid=True,
                preparation_derivation_valid=True,
                exact_owner_rebind=False,
                cold_mask_applied=False,
                ordinary_advance_applied=False,
                replacement_applied=False,
                cycle_completed=False,
                transaction_applied=False,
                caller_authenticated=False,
            )
        source_oak = _prototype_oak_state(state.bridge_state.prototype_state.oak_state)
        destination_installation = (
            lifecycle_result.state.cycle_state.scheduler_state.installation_state
        )
        destination_stomp = destination_installation.lifecycle_state.stomp_state
        replacement_applied = bool(jax.device_get(lifecycle_result.diagnostics.cycle_completed))
        reset_slots = lifecycle_result.replacement.reset_slots
        oak_rebind = self._oak.rebind_option_slots(
            source_oak,
            destination_stomp,
            reset_slots,
        )
        if replacement_applied:
            exact_owner = bool(
                jax.device_get(
                    oak_rebind.transaction_applied
                    & _tree_exact_equal(
                        oak_rebind.state.stomp_state,
                        destination_stomp,
                    )
                )
            )
            destination_oak: OaKState = oak_rebind.state
        else:
            exact_owner = bool(
                jax.device_get(
                    _tree_exact_equal(source_oak.stomp_state, destination_stomp)
                    & (~jnp.any(reset_slots))
                )
            )
            destination_oak = source_oak
        prototype_candidate = _replace_prototype_oak_state(
            state.bridge_state.prototype_state,
            destination_oak,
        )
        candidate = self._compose_repeated_destination(
            state,
            lifecycle_result.state,
            prototype_candidate,
        )
        cold_mask = bool(
            jax.device_get(
                jnp.array_equal(
                    candidate.bridge_state.extended_action_mask,
                    self._lifecycle.extended_action_mask(lifecycle_result.state),
                )
                & self.state_valid(candidate)
            )
        )
        applied = exact_owner and cold_mask
        return PrototypeRepeatedOptionReplacementCommitResult(
            state=candidate if applied else state,
            lifecycle_result=lifecycle_result,
            oak_rebind=oak_rebind,
            destination_matches_source=True,
            prepared_integrity_valid=True,
            preparation_derivation_valid=True,
            exact_owner_rebind=exact_owner,
            cold_mask_applied=cold_mask,
            ordinary_advance_applied=applied,
            replacement_applied=applied and replacement_applied,
            cycle_completed=applied and replacement_applied,
            transaction_applied=applied,
            caller_authenticated=False,
        )

    def _adopt_control_bridge_state(
        self,
        source: PrototypeRepeatedOptionAuthorityBridgeState,
        bridge_state: PrototypeOptionAuthorityBridgeState,
    ) -> tuple[PrototypeRepeatedOptionAuthorityBridgeState, Array]:
        source_repeated, source_attached = self._attach_source(source)
        destination_oak = _prototype_oak_state(bridge_state.prototype_state.oak_state)
        destination_child = self._authority.attach_borrowed_stomp(
            bridge_state.authority_metadata,
            destination_oak.stomp_state,
        )
        rebased = self._lifecycle._with_checksum(
            dataclasses.replace(
                source_repeated,
                cycle_state=destination_child.state,
                revision=_saturating_increment(source_repeated.revision),
                binding_checksum=jnp.zeros((2,), dtype=jnp.uint32),
            )
        )
        synchronized = (
            source.repeated_synchronized
            & source_attached
            & bridge_state.authority_synchronized
            & destination_child.transaction_applied
            & self._lifecycle.state_valid(rebased)
        )
        fresh_metadata = self._detach_lifecycle(rebased)
        selected_metadata = cast(
            PrototypeRepeatedOptionLifecycleMetadata,
            jax.tree.map(
                lambda fresh, old: jnp.where(synchronized, fresh, old),
                fresh_metadata,
                source.lifecycle_metadata,
            ),
        )
        candidate = self._next_state(
            source,
            bridge_state,
            selected_metadata,
            synchronized=synchronized,
        )
        return candidate, synchronized

    def start(
        self,
        state: PrototypeRepeatedOptionAuthorityBridgeState,
        initial_observation: Array,
    ) -> PrototypeRepeatedOptionAuthorityBridgeStartResult:
        """Run the v1 start once and rebind only detached repeated metadata."""

        self._check_state_contract(state)
        bridge_result = self._bridge.start(state.bridge_state, initial_observation)
        candidate, synchronized = self._adopt_control_bridge_state(
            state,
            bridge_result.state,
        )
        applied = bridge_result.transaction_applied & self.state_valid(candidate)
        next_state = cast(
            PrototypeRepeatedOptionAuthorityBridgeState,
            jax.lax.cond(applied, lambda _: candidate, lambda _: state, None),
        )
        return PrototypeRepeatedOptionAuthorityBridgeStartResult(
            state=next_state,
            bridge=bridge_result,
            repeated_metadata_advanced=applied & synchronized,
            repeated_desynchronized=applied & (~synchronized),
            transaction_applied=applied,
        )

    def update_transition(
        self,
        state: PrototypeRepeatedOptionAuthorityBridgeState,
        transition: PrototypeTransition,
        candidate_update_audit_evidence: PrototypeCandidateUpdateAuditEvidence | None = None,
        *,
        gradient_joy_evidence: PrototypeGradientJoyEvidence | None = None,
        experiential_memory_input: PrototypeExperientialMemoryInput | None = None,
        partner_policy_fusion_input: PrototypePartnerPolicyFusionInput | None = None,
        partner_policy_fusion_feedback: PrototypePartnerPolicyFusionFeedback | None = None,
        context: int | Array = 0,
        idle_candidate_option: int | Array = 0,
        idle_initiation_eligible: bool | Array = False,
        comparator_randomized: bool | Array = False,
        treatment_propensity: float | Array = 0.0,
    ) -> PrototypeRepeatedOptionAuthorityBridgeUpdateResult:
        """Forward every sidecar and consume Prototype's raw STOMP result once."""

        self._check_state_contract(state)
        bridge_result = self._bridge.update_transition(
            state.bridge_state,
            transition,
            candidate_update_audit_evidence=candidate_update_audit_evidence,
            gradient_joy_evidence=gradient_joy_evidence,
            experiential_memory_input=experiential_memory_input,
            partner_policy_fusion_input=partner_policy_fusion_input,
            partner_policy_fusion_feedback=partner_policy_fusion_feedback,
            context=context,
            idle_candidate_option=idle_candidate_option,
            idle_initiation_eligible=idle_initiation_eligible,
            comparator_randomized=comparator_randomized,
            treatment_propensity=treatment_propensity,
        )
        candidate, synchronized = self._adopt_control_bridge_state(
            state,
            bridge_result.state,
        )
        applied = bridge_result.transaction_applied & self.state_valid(candidate)
        next_state = cast(
            PrototypeRepeatedOptionAuthorityBridgeState,
            jax.lax.cond(applied, lambda _: candidate, lambda _: state, None),
        )
        return PrototypeRepeatedOptionAuthorityBridgeUpdateResult(
            state=next_state,
            bridge=bridge_result,
            repeated_metadata_advanced=applied & synchronized,
            repeated_desynchronized=applied & (~synchronized),
            control_transition_rolled_back_by_adapter=jnp.asarray(False, dtype=jnp.bool_),
            transaction_applied=applied,
        )

    @staticmethod
    def _state_sha256(state: PrototypeRepeatedOptionAuthorityBridgeState) -> str:
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
        state: PrototypeRepeatedOptionAuthorityBridgeState,
    ) -> dict[str, object]:
        """Return one strict host-only v2 checkpoint with no preparations."""

        if not bool(jax.device_get(self.state_valid(state))):
            raise ValueError("cannot checkpoint an invalid v2 bridge state")
        return {
            "schema_version": PROTOTYPE_REPEATED_OPTION_AUTHORITY_BRIDGE_CHECKPOINT_SCHEMA,
            "config": self.to_config(),
            "state": state,
            "state_sha256": self._state_sha256(state),
            "persistent_stomp_state_owners": 1,
            "persistent_prepared_transactions": 0,
            "checksum_authenticated": False,
            "caller_authenticated": False,
        }

    def restore_checkpoint(
        self,
        payload: object,
        *,
        expected_completed_cycles: int | Array,
        expected_revision: int | Array,
    ) -> PrototypeRepeatedOptionAuthorityBridgeState:
        """Restore exact v2 state under caller-supplied anti-rollback clocks."""

        if type(payload) is not dict:
            raise TypeError("v2 bridge checkpoint payload must be an exact dict")
        raw = cast(dict[str, object], payload)
        expected = {
            "schema_version",
            "config",
            "state",
            "state_sha256",
            "persistent_stomp_state_owners",
            "persistent_prepared_transactions",
            "checksum_authenticated",
            "caller_authenticated",
        }
        if set(raw) != expected:
            raise ValueError("v2 bridge checkpoint fields differ")
        fixed = {
            "schema_version": PROTOTYPE_REPEATED_OPTION_AUTHORITY_BRIDGE_CHECKPOINT_SCHEMA,
            "config": self.to_config(),
            "persistent_stomp_state_owners": 1,
            "persistent_prepared_transactions": 0,
            "checksum_authenticated": False,
            "caller_authenticated": False,
        }
        for name, expected_value in fixed.items():
            if raw[name] != expected_value:
                raise ValueError(f"v2 bridge checkpoint {name} differs")
        state = raw["state"]
        if type(state) is not PrototypeRepeatedOptionAuthorityBridgeState:
            raise TypeError("v2 bridge checkpoint state has the wrong exact type")
        restored = state
        digest = raw["state_sha256"]
        if type(digest) is not str or digest != self._state_sha256(restored):
            raise ValueError("v2 bridge checkpoint state hash differs")
        completed = jnp.asarray(expected_completed_cycles, dtype=jnp.int32)
        revision = jnp.asarray(expected_revision, dtype=jnp.int32)
        clocks = (restored.lifecycle_metadata.completed_cycles == completed) & (
            restored.revision == revision
        )
        if not bool(jax.device_get(clocks & self.state_valid(restored))):
            raise ValueError("v2 bridge checkpoint state is invalid, stale, or rebound")
        return restored

    def resource_budget(
        self,
        state: PrototypeRepeatedOptionAuthorityBridgeState,
    ) -> PrototypeRepeatedOptionAuthorityBridgeResourceBudget:
        """Measure one owner and disclose all inherited host boundaries."""

        if not bool(jax.device_get(self.state_valid(state))):
            raise ValueError("resource measurement requires a valid v2 bridge state")
        base = self._bridge.resource_budget(state.bridge_state)
        total = _tree_nbytes(state)
        bridge_nbytes = _tree_nbytes(state.bridge_state)
        overlay_nbytes = _tree_nbytes(state.lifecycle_metadata)
        completed = int(jax.device_get(state.lifecycle_metadata.completed_cycles))
        return PrototypeRepeatedOptionAuthorityBridgeResourceBudget(
            persistent_state_nbytes=total,
            v1_bridge_state_nbytes=bridge_nbytes,
            prototype_state_nbytes=base.prototype_state_nbytes,
            detached_authority_metadata_nbytes=(base.detached_authority_metadata_nbytes),
            repeated_overlay_nbytes=overlay_nbytes,
            adapter_binding_nbytes=total - bridge_nbytes - overlay_nbytes,
            persistent_stomp_state_owners=1,
            detached_authority_metadata_stomp_state_owners=0,
            repeated_overlay_stomp_state_owners=0,
            borrowed_stomp_bindings=1,
            persistent_prepared_transactions=0,
            max_cycles=self._lifecycle.config.max_cycles,
            completed_cycles=completed,
            remaining_cycles=self._lifecycle.config.max_cycles - completed,
            active_cycle=bool(jax.device_get(state.lifecycle_metadata.cycle_key_active)),
            real_control_stomp_updates_per_ordinary_transition=(
                base.real_control_stomp_updates_per_ordinary_transition
            ),
            configured_imagined_stomp_updates_per_ordinary_transition=(
                base.configured_imagined_stomp_updates_per_ordinary_transition
            ),
            max_total_stomp_updates_per_ordinary_transition=(
                base.max_total_stomp_updates_per_ordinary_transition
            ),
            stomp_updates_per_audit_adoption=0,
            stomp_updates_per_retirement_transaction=0,
            stomp_updates_per_replacement_transaction=0,
            retirement_prepare_host_only=True,
            retirement_commit_host_only=True,
            replacement_prepare_host_only=True,
            replacement_commit_host_only=True,
            checkpoint_host_only=True,
            caller_authenticated=False,
            checksum_authenticated=False,
            assessment=PROTOTYPE_REPEATED_OPTION_AUTHORITY_BRIDGE_ASSESSMENT,
            benefit_claim=False,
            evidence_authority=False,
            promotion_authority=False,
            safety_authority=False,
            go_no_go_authority=False,
            retirement_authority=False,
            replacement_authority=False,
            discovery_authority=False,
            dispatch_authority=False,
            autonomous_curation_authority=False,
            scientific_promotion_allowed=False,
            checkpoint_schema=PROTOTYPE_REPEATED_OPTION_AUTHORITY_BRIDGE_CHECKPOINT_SCHEMA,
        )


__all__ = [
    "PROTOTYPE_REPEATED_OPTION_AUTHORITY_BRIDGE_ASSESSMENT",
    "PROTOTYPE_REPEATED_OPTION_AUTHORITY_BRIDGE_AUTONOMOUS_CURATION_AUTHORITY",
    "PROTOTYPE_REPEATED_OPTION_AUTHORITY_BRIDGE_BENEFIT_CLAIM",
    "PROTOTYPE_REPEATED_OPTION_AUTHORITY_BRIDGE_CHECKPOINT_SCHEMA",
    "PROTOTYPE_REPEATED_OPTION_AUTHORITY_BRIDGE_DISCOVERY_AUTHORITY",
    "PROTOTYPE_REPEATED_OPTION_AUTHORITY_BRIDGE_DISPATCH_AUTHORITY",
    "PROTOTYPE_REPEATED_OPTION_AUTHORITY_BRIDGE_ERROR_DESYNCHRONIZED",
    "PROTOTYPE_REPEATED_OPTION_AUTHORITY_BRIDGE_ERROR_NONE",
    "PROTOTYPE_REPEATED_OPTION_AUTHORITY_BRIDGE_EVIDENCE_AUTHORITY",
    "PROTOTYPE_REPEATED_OPTION_AUTHORITY_BRIDGE_GO_NO_GO_AUTHORITY",
    "PROTOTYPE_REPEATED_OPTION_AUTHORITY_BRIDGE_PROMOTION_AUTHORITY",
    "PROTOTYPE_REPEATED_OPTION_AUTHORITY_BRIDGE_REPLACEMENT_AUTHORITY",
    "PROTOTYPE_REPEATED_OPTION_AUTHORITY_BRIDGE_RETIREMENT_AUTHORITY",
    "PROTOTYPE_REPEATED_OPTION_AUTHORITY_BRIDGE_SAFETY_AUTHORITY",
    "PROTOTYPE_REPEATED_OPTION_AUTHORITY_BRIDGE_SCIENTIFIC_PROMOTION_ALLOWED",
    "PROTOTYPE_REPEATED_OPTION_AUTHORITY_BRIDGE_STATE_SCHEMA",
    "PrototypeRepeatedOptionAuthorityBridge",
    "PrototypeRepeatedOptionAuthorityBridgeResourceBudget",
    "PrototypeRepeatedOptionAuthorityBridgeStartResult",
    "PrototypeRepeatedOptionAuthorityBridgeState",
    "PrototypeRepeatedOptionAuthorityBridgeUpdateResult",
    "PrototypeRepeatedOptionLifecycleMetadata",
    "PrototypeRepeatedOptionReplacementCommitResult",
    "PrototypeRepeatedOptionReplacementPrepared",
    "PrototypeRepeatedOptionRetirementCommitResult",
    "PrototypeRepeatedOptionRetirementPrepared",
]
