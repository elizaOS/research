# mypy: disable-error-code="arg-type,attr-defined,call-arg,no-any-return,type-var"
"""Borrowed repeated-option metadata for one external coordinator owner.

The persistent state contains exactly one
``ExternalLearnedStateRouterAuditCoordinatorState``.  Its nested
Prototype→OaK→STOMP state is the sole control owner.  The option-authority and
bounded repeated-cycle trees are detached metadata whose checksums borrow that
owner; neither can reach another ``STOMPState``.

Ordinary control is evaluated by the existing coordinator before this module
is called.  The sidecar consumes Prototype's already-evaluated raw
``STOMPUpdateResult`` through the lifecycle observer exactly once, performs no
STOMP reevaluation, binds the fixed owner-finalization trace, and adopts the
coordinator plus both metadata overlays atomically.  Attempted lower results
remain visible on rejection, while every public applied fact is outer-gated.

Lifecycle curation uses only an all-installed-to-all-installed host transaction
through ``AuthorizedOptionAtomicSwapController``.  A transient cold state is
never persistent.  Accepted swaps rebind the exact final STOMP option slots
into the coordinator's existing OaK state before rolling the same one-shot
child into the next bounded cycle.  Decline, tamper, replay, destination drift,
or downstream refusal returns the complete source bit-exactly.

This is L0 lifecycle/memory metadata.  Delight is unavailable here: the
sidecar does not ask whether a gradient sparks joy, and it adds no actor
backward pass, control owner, evidence, benefit, or promotion authority.
"""

from __future__ import annotations

import dataclasses
from typing import Any, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import Array
from jaxtyping import Bool, Int, UInt

from alberta_framework.core.authorized_option_atomic_swap import (
    AuthorizedOptionAtomicSwapAuthorityReceipt,
    AuthorizedOptionAtomicSwapController,
    AuthorizedOptionAtomicSwapPrepared,
    AuthorizedOptionAtomicSwapResult,
)
from alberta_framework.core.authorized_option_replacement import (
    AuthorizedOptionReplacementMetadataState,
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
from alberta_framework.core.external_learned_state_router_audit_coordinator import (
    ExternalLearnedStateRouterAuditCoordinator,
    ExternalLearnedStateRouterAuditCoordinatorEvaluatedTransition,
    ExternalLearnedStateRouterAuditCoordinatorIntegrityReceipt,
    ExternalLearnedStateRouterAuditCoordinatorResult,
    ExternalLearnedStateRouterAuditCoordinatorState,
)
from alberta_framework.core.oak import OaKOptionSlotRebindResult, OaKState
from alberta_framework.core.options import (
    DISPATCH_OWNER_BASE_PRIMITIVE,
    DISPATCH_OWNER_OPTION,
    STOMPState,
)
from alberta_framework.core.prototype_agent import (
    PrototypeCachedPrimitiveActionReplacement,
)
from alberta_framework.core.prototype_option_authority_bridge import (
    _checksum_arrays,
    _increment_words,
    _prototype_oak_state,
    _replace_prototype_oak_state,
    _saturating_increment,
    _tree_exact_equal,
)
from alberta_framework.core.repeated_option_lifecycle import (
    REPEATED_OPTION_LIFECYCLE_ERROR_CAPACITY,
    REPEATED_OPTION_LIFECYCLE_ERROR_NONE,
    RepeatedOptionLifecycle,
    RepeatedOptionLifecycleState,
)
from alberta_framework.core.stomp_option_lifecycle import (
    STOMPOptionLifecycleExternalAdoptionResult,
    STOMPOptionLifecycleExternalOwnerFinalizationResult,
    STOMPOptionLifecycleExternalStartAdoptionResult,
)
from alberta_framework.core.stomp_owner_finalization import (
    stomp_typed_tree_digest,
)

EXTERNAL_COORDINATOR_REPEATED_OPTION_SIDECAR_STATE_SCHEMA = (
    "alberta.external-coordinator-repeated-option-sidecar.state.v1"
)
EXTERNAL_COORDINATOR_REPEATED_OPTION_SIDECAR_STATUS = (
    "l0-development-external-coordinator-repeated-option-sidecar"
)
EXTERNAL_COORDINATOR_REPEATED_OPTION_SIDECAR_EVIDENCE_LEVEL = "L0"
EXTERNAL_COORDINATOR_REPEATED_OPTION_SIDECAR_ASSESSMENT = "not_assessed"
EXTERNAL_COORDINATOR_REPEATED_OPTION_SIDECAR_LIMITATIONS = (
    "exact-match-only-initial-owner-binding",
    "one-borrowed-coordinator-prototype-oak-stomp-owner",
    "raw-stomp-result-consumed-with-zero-reevaluation",
    "all-installed-to-all-installed-atomic-swap-only",
    "host-only-composite-orchestration",
    "delight-unavailable-lifecycle-memory-metadata-only",
    "integrity-bindings-are-not-caller-authentication",
    "no-output-artifact-evidence-benefit-or-promotion-authority",
)

_INT32_MAX = 2**31 - 1


def _tree_select(condition: Array, yes: Any, no: Any) -> Any:
    return jax.tree.map(lambda left, right: jnp.where(condition, left, right), yes, no)


def _tree_nbytes(value: object) -> int:
    total = 0
    for leaf in jax.tree.leaves(value):
        array = jnp.asarray(leaf)
        if jax.dtypes.issubdtype(array.dtype, jax.dtypes.prng_key):
            array = jr.key_data(array)
        total += int(array.nbytes)
    return total


def _contains_tracer(value: object) -> bool:
    return any(isinstance(leaf, jax.core.Tracer) for leaf in jax.tree.leaves(value))


def _require_array(
    value: object,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: Any,
) -> Array:
    if not hasattr(value, "shape") or not hasattr(value, "dtype"):
        raise TypeError(f"{name} must expose array metadata")
    array = cast(Array, value)
    if tuple(array.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if jnp.dtype(array.dtype) != jnp.dtype(dtype):
        raise TypeError(f"{name} must have dtype {jnp.dtype(dtype)}")
    return array


def _require_threefry_key(value: object, *, name: str) -> Array:
    try:
        implementation = str(jr.key_impl(value))
        data = jr.key_data(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a scalar typed Threefry key") from exc
    array = cast(Array, value)
    if (
        array.shape != ()
        or not jax.dtypes.issubdtype(array.dtype, jax.dtypes.prng_key)
        or implementation != "threefry2x32"
        or data.shape != (2,)
        or data.dtype != jnp.uint32
    ):
        raise TypeError(f"{name} must be a scalar typed Threefry key")
    return array


def _normalize_bool(value: bool | Array, *, name: str) -> Array:
    if type(value) is bool:
        return jnp.asarray(value, dtype=jnp.bool_)
    return _require_array(value, name=name, shape=(), dtype=jnp.bool_)


def _normalize_int(
    value: int | Array,
    *,
    name: str,
    lower: int,
    upper: int,
) -> tuple[Array, Array]:
    if type(value) is int:
        host_valid = lower <= value < upper
        return (
            jnp.asarray(value if host_valid else lower, dtype=jnp.int32),
            jnp.asarray(host_valid, dtype=jnp.bool_),
        )
    array = _require_array(value, name=name, shape=(), dtype=jnp.int32)
    valid = (array >= lower) & (array < upper)
    return jnp.where(valid, array, jnp.asarray(lower, dtype=jnp.int32)), valid


def _normalize_propensity(value: float | Array) -> tuple[Array, Array]:
    if type(value) is float:
        candidate = jnp.asarray(value, dtype=jnp.float32)
    else:
        candidate = _require_array(
            value,
            name="treatment_propensity",
            shape=(),
            dtype=jnp.float32,
        )
    valid = jnp.isfinite(candidate) & (candidate >= 0.0) & (candidate <= 1.0)
    return jnp.where(valid, candidate, jnp.asarray(0.0, jnp.float32)), valid


def _words_less(left: Array, right: Array) -> Array:
    return (left[0] < right[0]) | ((left[0] == right[0]) & (left[1] < right[1]))


@chex.dataclass(frozen=True)
class ExternalCoordinatorRepeatedOptionLifecycleMetadata:
    """Bounded repeated-cycle overlay containing no STOMP owner."""

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
class ExternalCoordinatorRepeatedOptionSidecarState:
    """One coordinator owner plus two detached metadata overlays."""

    coordinator_state: ExternalLearnedStateRouterAuditCoordinatorState
    authority_metadata: AuthorizedOptionReplacementMetadataState
    lifecycle_metadata: ExternalCoordinatorRepeatedOptionLifecycleMetadata
    extended_action_mask: Bool[Array, " n_total_actions"]
    revision: Int[Array, ""]
    binding_checksum: UInt[Array, " 2"]


@chex.dataclass(frozen=True)
class ExternalCoordinatorRepeatedOptionBorrowedMetadata:
    """Coordinator-free projection of one exact sidecar binding.

    ``binding_checksum`` remains the checksum of the complete sidecar state,
    including its borrowed coordinator.  Consequently the bundle attaches
    only to that exact coordinator endpoint (or to a candidate produced by a
    sidecar transaction that emitted a newly detached bundle).  The bundle
    contains no coordinator, Prototype, OaK, or STOMP state.
    """

    authority_metadata: AuthorizedOptionReplacementMetadataState
    lifecycle_metadata: ExternalCoordinatorRepeatedOptionLifecycleMetadata
    extended_action_mask: Bool[Array, " n_total_actions"]
    revision: Int[Array, ""]
    binding_checksum: UInt[Array, " 2"]


@chex.dataclass(frozen=True)
class ExternalCoordinatorRepeatedOptionBorrowedMetadataAttachResult:
    """Transient exact-owner attachment; never part of persistent state."""

    state: ExternalCoordinatorRepeatedOptionSidecarState
    metadata: ExternalCoordinatorRepeatedOptionBorrowedMetadata
    stomp_owner_digest: UInt[Array, " 8"]
    coordinator_binding_valid: Bool[Array, ""]
    transaction_applied: Bool[Array, ""]


@chex.dataclass(frozen=True)
class ExternalCoordinatorRepeatedOptionLiveActionProjectionResult:
    """Zero-evaluation rebind onto an exact live-memory action replacement."""

    state: ExternalCoordinatorRepeatedOptionSidecarState
    metadata: ExternalCoordinatorRepeatedOptionBorrowedMetadata
    replacement_supplied: Bool[Array, ""]
    replacement_committed: Bool[Array, ""]
    source_binding_valid: Bool[Array, ""]
    replacement_stomp_delta_exact: Bool[Array, ""]
    replacement_prototype_exact: Bool[Array, ""]
    coordinator_wrapper_delta_exact: Bool[Array, ""]
    stomp_owner_matches_final_live: Bool[Array, ""]
    stomp_clocks_preserved: Bool[Array, ""]
    evaluated_stomp_digest: UInt[Array, " 8"]
    replacement_stomp_digest: UInt[Array, " 8"]
    final_live_stomp_digest: UInt[Array, " 8"]
    metadata_rebased: Bool[Array, ""]
    additional_coordinator_evaluations: Int[Array, ""]
    additional_prototype_evaluations: Int[Array, ""]
    additional_stomp_update_evaluations: Int[Array, ""]
    additional_lifecycle_observations: Int[Array, ""]
    transaction_applied: Bool[Array, ""]


@chex.dataclass(frozen=True)
class ExternalCoordinatorRepeatedOptionSidecarStartResult:
    state: ExternalCoordinatorRepeatedOptionSidecarState
    lifecycle_attempt: STOMPOptionLifecycleExternalStartAdoptionResult
    source_binding_valid: Bool[Array, ""]
    coordinator_started_attempted: Bool[Array, ""]
    coordinator_started: Bool[Array, ""]
    lifecycle_metadata_applied: Bool[Array, ""]
    delight_available: Bool[Array, ""]
    transaction_applied: Bool[Array, ""]


@chex.dataclass(frozen=True)
class ExternalCoordinatorRepeatedOptionSidecarResult:
    state: ExternalCoordinatorRepeatedOptionSidecarState
    coordinator_result: ExternalLearnedStateRouterAuditCoordinatorResult
    lifecycle_attempt: STOMPOptionLifecycleExternalAdoptionResult
    lifecycle_owner_finalization: STOMPOptionLifecycleExternalOwnerFinalizationResult
    source_binding_valid: Bool[Array, ""]
    audit_inputs_valid: Bool[Array, ""]
    raw_stomp_result_bound: Bool[Array, ""]
    raw_stomp_result_digest_bound: Bool[Array, ""]
    finalization_trace_bound: Bool[Array, ""]
    final_stomp_owner_digest_bound: Bool[Array, ""]
    raw_stomp_update_evaluations: Int[Array, ""]
    additional_stomp_update_evaluations: Int[Array, ""]
    lifecycle_observation_evaluations: Int[Array, ""]
    raw_stomp_result_consumed: Bool[Array, ""]
    coordinator_update_applied: Bool[Array, ""]
    lifecycle_metadata_applied: Bool[Array, ""]
    delight_available: Bool[Array, ""]
    additional_delight_evaluations: Int[Array, ""]
    additional_actor_backward_calls: Int[Array, ""]
    downstream_candidate_valid: Bool[Array, ""]
    candidate_state_valid: Bool[Array, ""]
    transaction_applied: Bool[Array, ""]


@chex.dataclass(frozen=True)
class ExternalCoordinatorRepeatedOptionAtomicSwapPrepared:
    """Source/cycle-bound host preparation; never persistent."""

    source_state: ExternalCoordinatorRepeatedOptionSidecarState
    cycle_key: Array
    atomic_swap_prepared: AuthorizedOptionAtomicSwapPrepared
    source_binding_valid: Bool[Array, ""]
    cycle_capacity_available: Bool[Array, ""]
    cycle_key_fresh: Bool[Array, ""]
    retirement_authority_revision_fresh: Bool[Array, ""]
    preparation_valid: Bool[Array, ""]
    prepared_checksum: UInt[Array, " 2"]


@chex.dataclass(frozen=True)
class ExternalCoordinatorRepeatedOptionAtomicSwapAuthorityReceipt:
    """Unkeyed wrapper receipt binding one complete authorized swap."""

    atomic_swap_authority: AuthorizedOptionAtomicSwapAuthorityReceipt
    cycle_index: Int[Array, ""]
    cycle_key_data: UInt[Array, " 2"]
    source_binding_checksum: UInt[Array, " 2"]
    source_revision: Int[Array, ""]
    prepared_checksum: UInt[Array, " 2"]
    retirement_authority_revision_fresh: Bool[Array, ""]
    replacement_authority_revision_fresh: Bool[Array, ""]
    caller_authenticated: Bool[Array, ""]


@chex.dataclass(frozen=True)
class ExternalCoordinatorRepeatedOptionAtomicSwapResult:
    state: ExternalCoordinatorRepeatedOptionSidecarState
    atomic_swap_attempt: AuthorizedOptionAtomicSwapResult
    oak_rebind: OaKOptionSlotRebindResult
    destination_matches_source: Bool[Array, ""]
    prepared_integrity_valid: Bool[Array, ""]
    preparation_derivation_valid: Bool[Array, ""]
    authority_binding_valid: Bool[Array, ""]
    cycle_key_fresh: Bool[Array, ""]
    exact_owner_rebind: Bool[Array, ""]
    all_slots_installed_before: Bool[Array, ""]
    all_slots_installed_after: Bool[Array, ""]
    cold_state_persisted: Bool[Array, ""]
    retirement_applied: Bool[Array, ""]
    replacement_applied: Bool[Array, ""]
    downstream_candidate_valid: Bool[Array, ""]
    candidate_state_valid: Bool[Array, ""]
    caller_authenticated: Bool[Array, ""]
    transaction_applied: Bool[Array, ""]


@dataclasses.dataclass(frozen=True)
class ExternalCoordinatorRepeatedOptionSidecarResourceBudget:
    persistent_state_nbytes: int
    coordinator_state_nbytes: int
    detached_authority_metadata_nbytes: int
    repeated_overlay_nbytes: int
    mask_revision_binding_nbytes: int
    coordinator_state_owners: int
    prototype_state_owners: int
    oak_state_owners: int
    stomp_state_owners: int
    detached_authority_metadata_stomp_state_owners: int
    repeated_overlay_stomp_state_owners: int
    borrowed_stomp_bindings: int
    additional_stomp_update_evaluations_per_adoption: int
    maximum_lifecycle_observations_per_adoption: int
    atomic_swap_prepare_host_only: bool
    atomic_swap_adopt_host_only: bool
    delight_available: bool
    additional_delight_evaluations: int
    additional_actor_backward_calls: int
    output_write_calls: int
    artifact_bytes_written: int


class ExternalCoordinatorRepeatedOptionSidecar:
    """Atomic metadata sidecar borrowing one external coordinator owner."""

    def __init__(
        self,
        coordinator: ExternalLearnedStateRouterAuditCoordinator,
        lifecycle: RepeatedOptionLifecycle,
    ) -> None:
        if type(coordinator) is not ExternalLearnedStateRouterAuditCoordinator:
            raise TypeError("coordinator must be an exact external coordinator")
        if type(lifecycle) is not RepeatedOptionLifecycle:
            raise TypeError("lifecycle must be an exact RepeatedOptionLifecycle")
        self._coordinator = coordinator
        self._lifecycle = lifecycle
        self._replacement = lifecycle.replacement
        self._atomic_swap = AuthorizedOptionAtomicSwapController(self._replacement)
        self._scheduler = self._replacement.scheduler
        self._installation = self._replacement.scheduler.installation
        self._prototype = coordinator.inner.prototype
        self._oak = self._prototype._oak
        if self._prototype.config.oak.stomp != self._installation.stomp_agent.config:
            raise ValueError("coordinator and lifecycle must share the exact STOMP config")
        if self._prototype.config.gradient_joy is not None:
            raise ValueError("sidecar requires Prototype delight to remain unavailable")

    @property
    def coordinator(self) -> ExternalLearnedStateRouterAuditCoordinator:
        return self._coordinator

    @property
    def lifecycle(self) -> RepeatedOptionLifecycle:
        return self._lifecycle

    @property
    def atomic_swap(self) -> AuthorizedOptionAtomicSwapController:
        return self._atomic_swap

    def to_config(self) -> dict[str, object]:
        return {
            "type": type(self).__name__,
            "state_schema": EXTERNAL_COORDINATOR_REPEATED_OPTION_SIDECAR_STATE_SCHEMA,
            "mechanism_status": EXTERNAL_COORDINATOR_REPEATED_OPTION_SIDECAR_STATUS,
            "evidence_level": EXTERNAL_COORDINATOR_REPEATED_OPTION_SIDECAR_EVIDENCE_LEVEL,
            "assessment": EXTERNAL_COORDINATOR_REPEATED_OPTION_SIDECAR_ASSESSMENT,
            "coordinator": self._coordinator.to_config(),
            "repeated_lifecycle": self._lifecycle.to_config(),
            "coordinator_state_owners": 1,
            "prototype_state_owners": 1,
            "oak_state_owners": 1,
            "stomp_state_owners": 1,
            "additional_coordinator_state_owners": 0,
            "additional_prototype_state_owners": 0,
            "additional_bridge_state_owners": 0,
            "detached_authority_metadata_stomp_state_owners": 0,
            "repeated_overlay_stomp_state_owners": 0,
            "borrowed_stomp_bindings": 1,
            "raw_stomp_result_relation": "already-evaluated-consumed-once",
            "additional_stomp_update_evaluations": 0,
            "atomic_swap_semantics": "all-installed-to-all-installed-only",
            "separate_retirement_commit_exposed": False,
            "delight_available": False,
            "delight_interpretation": "lifecycle-memory-metadata-only",
            "additional_actor_backward_calls": 0,
            "host_only": True,
            "scan_supported": False,
            "caller_authenticated": False,
            "retirement_authority": False,
            "replacement_authority": False,
            "dispatch_authority": False,
            "safety_authority": False,
            "evidence_authority": False,
            "promotion_authority": False,
            "output_writes_authorized": False,
            "artifact_authorized": False,
            "limitations": list(
                EXTERNAL_COORDINATOR_REPEATED_OPTION_SIDECAR_LIMITATIONS
            ),
        }

    def _overlay_payload_arrays(
        self,
        metadata: ExternalCoordinatorRepeatedOptionLifecycleMetadata,
    ) -> tuple[Array, ...]:
        return tuple(
            cast(Array, leaf)
            for leaf in jax.tree.leaves(
                tuple(
                    getattr(metadata, field.name)
                    for field in dataclasses.fields(
                        ExternalCoordinatorRepeatedOptionLifecycleMetadata
                    )
                    if field.name != "metadata_checksum"
                )
            )
        )

    def _with_overlay_checksum(
        self,
        metadata: ExternalCoordinatorRepeatedOptionLifecycleMetadata,
    ) -> ExternalCoordinatorRepeatedOptionLifecycleMetadata:
        return dataclasses.replace(
            metadata,
            metadata_checksum=_checksum_arrays(
                self._overlay_payload_arrays(metadata)
            ),
        )

    def _detach_lifecycle(
        self,
        state: RepeatedOptionLifecycleState,
    ) -> ExternalCoordinatorRepeatedOptionLifecycleMetadata:
        return self._with_overlay_checksum(
            ExternalCoordinatorRepeatedOptionLifecycleMetadata(
                completed_cycles=state.completed_cycles,
                total_retirements=state.total_retirements,
                total_replacements=state.total_replacements,
                cycle_key_active=state.cycle_key_active,
                active_cycle_key_data=state.active_cycle_key_data,
                has_completed_cycle=state.has_completed_cycle,
                cycle_key_history=state.cycle_key_history,
                last_completed_cycle_key_data=state.last_completed_cycle_key_data,
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
        metadata: ExternalCoordinatorRepeatedOptionLifecycleMetadata,
    ) -> None:
        if type(metadata) is not ExternalCoordinatorRepeatedOptionLifecycleMetadata:
            raise TypeError("lifecycle_metadata has the wrong exact type")
        max_cycles = self._lifecycle.config.max_cycles
        contracts = (
            (metadata.completed_cycles, "completed_cycles", (), jnp.int32),
            (metadata.total_retirements, "total_retirements", (), jnp.int32),
            (metadata.total_replacements, "total_replacements", (), jnp.int32),
            (metadata.cycle_key_active, "cycle_key_active", (), jnp.bool_),
            (metadata.active_cycle_key_data, "active_cycle_key_data", (2,), jnp.uint32),
            (metadata.has_completed_cycle, "has_completed_cycle", (), jnp.bool_),
            (
                metadata.cycle_key_history,
                "cycle_key_history",
                (max_cycles, 2),
                jnp.uint32,
            ),
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
            (
                metadata.source_repeated_checksum,
                "source_repeated_checksum",
                (2,),
                jnp.uint32,
            ),
            (
                metadata.source_child_checksum,
                "source_child_checksum",
                (2,),
                jnp.uint32,
            ),
            (metadata.metadata_checksum, "metadata_checksum", (2,), jnp.uint32),
        )
        for value, name, shape, dtype in contracts:
            _require_array(
                value,
                name=f"lifecycle_metadata.{name}",
                shape=shape,
                dtype=dtype,
            )

    def _overlay_integrity_valid(
        self,
        metadata: ExternalCoordinatorRepeatedOptionLifecycleMetadata,
    ) -> Array:
        self._check_overlay_contract(metadata)
        max_cycles = self._lifecycle.config.max_cycles
        completed = metadata.completed_cycles
        active = metadata.cycle_key_active.astype(jnp.int32)
        indices = jnp.arange(max_cycles, dtype=jnp.int32)
        used = indices < completed
        pairwise = jnp.all(
            metadata.cycle_key_history[:, None, :]
            == metadata.cycle_key_history[None, :, :],
            axis=2,
        )
        duplicate = jnp.any(
            pairwise
            & used[:, None]
            & used[None, :]
            & (~jnp.eye(max_cycles, dtype=jnp.bool_))
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
                metadata.cycle_key_history[
                    jnp.clip(completed - 1, 0, max_cycles - 1)
                ],
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
                    metadata.cycle_key_history
                    == metadata.active_cycle_key_data[None, :],
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
        metadata: ExternalCoordinatorRepeatedOptionLifecycleMetadata,
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
        state: ExternalCoordinatorRepeatedOptionSidecarState,
    ) -> tuple[Array, ...]:
        return tuple(
            cast(Array, leaf)
            for leaf in jax.tree.leaves(
                (
                    state.coordinator_state,
                    state.authority_metadata,
                    state.lifecycle_metadata,
                    state.extended_action_mask,
                    state.revision,
                )
            )
        )

    def _with_checksum(
        self,
        state: ExternalCoordinatorRepeatedOptionSidecarState,
    ) -> ExternalCoordinatorRepeatedOptionSidecarState:
        return dataclasses.replace(
            state,
            binding_checksum=_checksum_arrays(self._payload_arrays(state)),
        )

    def _check_state_contract(
        self,
        state: ExternalCoordinatorRepeatedOptionSidecarState,
    ) -> None:
        if type(state) is not ExternalCoordinatorRepeatedOptionSidecarState:
            raise TypeError("state has the wrong exact sidecar type")
        if type(state.coordinator_state) is not (
            ExternalLearnedStateRouterAuditCoordinatorState
        ):
            raise TypeError("state.coordinator_state has the wrong exact type")
        self._replacement._check_metadata_contract(state.authority_metadata)
        self._check_overlay_contract(state.lifecycle_metadata)
        _require_array(
            state.extended_action_mask,
            name="state.extended_action_mask",
            shape=(self._prototype.config.oak.stomp.n_total_actions,),
            dtype=jnp.bool_,
        )
        _require_array(
            state.revision,
            name="state.revision",
            shape=(),
            dtype=jnp.int32,
        )
        _require_array(
            state.binding_checksum,
            name="state.binding_checksum",
            shape=(2,),
            dtype=jnp.uint32,
        )

    @staticmethod
    def _prototype_state(
        state: ExternalLearnedStateRouterAuditCoordinatorState,
    ) -> Any:
        return state.inner_state.prototype_state

    def _oak_state(
        self,
        state: ExternalLearnedStateRouterAuditCoordinatorState,
    ) -> OaKState:
        return _prototype_oak_state(self._prototype_state(state).oak_state)

    def _replace_coordinator_oak_state(
        self,
        state: ExternalLearnedStateRouterAuditCoordinatorState,
        oak_state: OaKState,
    ) -> ExternalLearnedStateRouterAuditCoordinatorState:
        prototype = _replace_prototype_oak_state(
            self._prototype_state(state),
            oak_state,
        )
        inner = state.inner_state.replace(prototype_state=prototype)
        return state.replace(
            inner_state=inner,
            current_action=prototype.current_action,
            current_decision_id=prototype.current_decision_id,
            cached_prototype_step_words=prototype.step_words,
            cached_feature_generation_words=(
                self._coordinator._feature_generation_words(inner)
            ),
            started=prototype.started,
        )

    def _attach_source(
        self,
        state: ExternalCoordinatorRepeatedOptionSidecarState,
    ) -> tuple[RepeatedOptionLifecycleState, Array]:
        self._check_state_contract(state)
        oak = self._oak_state(state.coordinator_state)
        child = self._replacement.attach_borrowed_stomp(
            state.authority_metadata,
            oak.stomp_state,
        )
        repeated, overlay_valid = self._attach_lifecycle(
            state.lifecycle_metadata,
            child.state,
        )
        return repeated, child.transaction_applied & overlay_valid

    def state_valid(
        self,
        state: ExternalCoordinatorRepeatedOptionSidecarState,
    ) -> Bool[Array, ""]:
        """Validate the sole nested owner and both exact metadata bindings."""

        self._check_state_contract(state)
        repeated, attached = self._attach_source(state)
        oak = self._oak_state(state.coordinator_state)
        lifecycle_metadata = (
            state.authority_metadata.scheduler_metadata.installation_metadata
            .lifecycle_metadata
        )
        return (
            self._coordinator.state_valid(state.coordinator_state)
            & self._replacement.metadata_state_valid(state.authority_metadata)
            & self._overlay_integrity_valid(state.lifecycle_metadata)
            & attached
            & (
                state.coordinator_state.started
                == lifecycle_metadata.started
            )
            & jnp.array_equal(oak.step_words, lifecycle_metadata.stomp_step_words)
            & jnp.array_equal(
                state.extended_action_mask,
                self._lifecycle.extended_action_mask(repeated),
            )
            & jnp.all(
                state.extended_action_mask[
                    : self._prototype.config.oak.n_primitive_actions
                ]
            )
            & (state.revision >= 0)
            & jnp.array_equal(
                state.binding_checksum,
                _checksum_arrays(self._payload_arrays(state)),
            )
        )

    def init(
        self,
        coordinator_state: ExternalLearnedStateRouterAuditCoordinatorState,
        repeated_state: RepeatedOptionLifecycleState,
    ) -> ExternalCoordinatorRepeatedOptionSidecarState:
        """Exact-bind an existing coordinator owner to one repeated child."""

        if type(coordinator_state) is not (
            ExternalLearnedStateRouterAuditCoordinatorState
        ):
            raise TypeError("coordinator_state must be an exact coordinator state")
        if type(repeated_state) is not RepeatedOptionLifecycleState:
            raise TypeError("repeated_state must be exact")
        if not bool(jax.device_get(self._coordinator.state_valid(coordinator_state))):
            raise ValueError("coordinator_state must satisfy its complete contract")
        if not bool(jax.device_get(self._lifecycle.state_valid(repeated_state))):
            raise ValueError("repeated_state must satisfy its complete contract")
        oak = self._oak_state(coordinator_state)
        lifecycle = (
            repeated_state.cycle_state.scheduler_state.installation_state
            .lifecycle_state
        )
        exact_owner = _tree_exact_equal(oak.stomp_state, lifecycle.stomp_state)
        phase_matches = coordinator_state.started == lifecycle.started
        if not bool(jax.device_get(exact_owner & phase_matches)):
            raise ValueError("coordinator and repeated lifecycle exact owner must match")
        state = self._with_checksum(
            ExternalCoordinatorRepeatedOptionSidecarState(
                coordinator_state=coordinator_state,
                authority_metadata=self._replacement.detach_borrowed_stomp(
                    repeated_state.cycle_state
                ),
                lifecycle_metadata=self._detach_lifecycle(repeated_state),
                extended_action_mask=self._lifecycle.extended_action_mask(
                    repeated_state
                ),
                revision=jnp.asarray(0, dtype=jnp.int32),
                binding_checksum=jnp.zeros((2,), dtype=jnp.uint32),
            )
        )
        if not bool(jax.device_get(self.state_valid(state))):
            raise ValueError("initialized sidecar state failed its exact contract")
        return state

    def detach_borrowed_metadata(
        self,
        state: ExternalCoordinatorRepeatedOptionSidecarState,
    ) -> ExternalCoordinatorRepeatedOptionBorrowedMetadata:
        """Remove the sole coordinator owner without weakening its checksum.

        This public boundary is intentionally a projection only.  It does not
        advance lifecycle state, evaluate a learner, or mint a new binding.
        """

        self._check_state_contract(state)
        return ExternalCoordinatorRepeatedOptionBorrowedMetadata(
            authority_metadata=state.authority_metadata,
            lifecycle_metadata=state.lifecycle_metadata,
            extended_action_mask=state.extended_action_mask,
            revision=state.revision,
            binding_checksum=state.binding_checksum,
        )

    def attach_borrowed_metadata(
        self,
        coordinator_state: ExternalLearnedStateRouterAuditCoordinatorState,
        metadata: ExternalCoordinatorRepeatedOptionBorrowedMetadata,
    ) -> ExternalCoordinatorRepeatedOptionBorrowedMetadataAttachResult:
        """Transiently restore one exact sidecar around a borrowed owner.

        Foreign, stale, or tampered metadata remains inspectable but produces
        ``transaction_applied=False``.  No alternate owner is persisted.
        """

        if type(coordinator_state) is not (
            ExternalLearnedStateRouterAuditCoordinatorState
        ):
            raise TypeError("coordinator_state must be an exact coordinator state")
        if type(metadata) is not ExternalCoordinatorRepeatedOptionBorrowedMetadata:
            raise TypeError("metadata must be exact borrowed sidecar metadata")
        state = ExternalCoordinatorRepeatedOptionSidecarState(
            coordinator_state=coordinator_state,
            authority_metadata=metadata.authority_metadata,
            lifecycle_metadata=metadata.lifecycle_metadata,
            extended_action_mask=metadata.extended_action_mask,
            revision=metadata.revision,
            binding_checksum=metadata.binding_checksum,
        )
        self._check_state_contract(state)
        bound = self.state_valid(state)
        return ExternalCoordinatorRepeatedOptionBorrowedMetadataAttachResult(
            state=state,
            metadata=metadata,
            stomp_owner_digest=stomp_typed_tree_digest(
                self._oak_state(coordinator_state).stomp_state
            ),
            coordinator_binding_valid=bound,
            transaction_applied=bound,
        )

    def _project_repeated_stomp_owner(
        self,
        source: RepeatedOptionLifecycleState,
        destination_stomp: STOMPState,
    ) -> tuple[RepeatedOptionLifecycleState, Array]:
        """Rebind checksum layers without advancing lifecycle observations."""

        authority = source.cycle_state
        scheduler = authority.scheduler_state
        installation = scheduler.installation_state
        lifecycle_api = self._installation.lifecycle.with_external_semantic_digests(
            installation.installed_semantic_digests
        )
        lifecycle = lifecycle_api._with_checksum(
            installation.lifecycle_state.replace(
                stomp_state=destination_stomp,
                binding_checksum=jnp.zeros((2,), dtype=jnp.uint32),
            )
        )
        next_installation = self._installation._with_checksum(
            installation.replace(
                lifecycle_state=lifecycle,
                binding_checksum=jnp.zeros((2,), dtype=jnp.uint32),
            )
        )
        next_scheduler = self._scheduler._with_checksum(
            scheduler.replace(
                installation_state=next_installation,
                binding_checksum=jnp.zeros((2,), dtype=jnp.uint32),
            )
        )
        next_authority = self._replacement._with_checksum(
            authority.replace(
                scheduler_state=next_scheduler,
                canonical_scheduler_checksum=next_scheduler.binding_checksum,
                binding_checksum=jnp.zeros((2,), dtype=jnp.uint32),
            )
        )
        projected = self._lifecycle._with_checksum(
            source.replace(
                cycle_state=next_authority,
                binding_checksum=jnp.zeros((2,), dtype=jnp.uint32),
            )
        )
        valid = (
            lifecycle_api.state_valid(lifecycle)
            & self._installation.state_valid(next_installation)
            & self._scheduler.state_valid(next_scheduler)
            & self._replacement.state_valid(next_authority)
            & self._lifecycle.state_valid(projected)
        )
        return projected, valid

    def project_live_cached_action_replacement(
        self,
        state: ExternalCoordinatorRepeatedOptionSidecarState,
        replacement: PrototypeCachedPrimitiveActionReplacement | None,
        final_live_coordinator_state: (
            ExternalLearnedStateRouterAuditCoordinatorState
        ),
    ) -> ExternalCoordinatorRepeatedOptionLiveActionProjectionResult:
        """Project metadata onto the live adapter's exact cached-action result.

        The lifecycle observation has already occurred.  This method permits
        only the three STOMP dispatch-credit leaves changed by
        ``replace_dispatched_primitive_action`` and the enclosing Prototype /
        coordinator cache changes emitted by the supplied committed result.
        It advances no clock, learner, lifecycle audit, or actor backward.
        """

        self._check_state_contract(state)
        if type(final_live_coordinator_state) is not (
            ExternalLearnedStateRouterAuditCoordinatorState
        ):
            raise TypeError("final_live_coordinator_state must be exact")
        if replacement is not None and type(replacement) is not (
            PrototypeCachedPrimitiveActionReplacement
        ):
            raise TypeError("replacement must be exact or None")
        if _contains_tracer((state, replacement, final_live_coordinator_state)):
            raise TypeError("live action projection is host-only")

        source_repeated, source_attached = self._attach_source(state)
        source_valid = self.state_valid(state) & source_attached
        source_coordinator = state.coordinator_state
        source_prototype = self._prototype_state(source_coordinator)
        source_oak = self._oak_state(source_coordinator)
        source_stomp = source_oak.stomp_state
        supplied = jnp.asarray(replacement is not None, dtype=jnp.bool_)

        if replacement is None:
            coordinator_exact = _tree_exact_equal(
                final_live_coordinator_state,
                source_coordinator,
            )
            selected = cast(
                ExternalCoordinatorRepeatedOptionSidecarState,
                _tree_select(source_valid & coordinator_exact, state, state),
            )
            metadata = self.detach_borrowed_metadata(selected)
            valid = source_valid & coordinator_exact
            return ExternalCoordinatorRepeatedOptionLiveActionProjectionResult(
                state=selected,
                metadata=metadata,
                replacement_supplied=supplied,
                replacement_committed=jnp.asarray(False, dtype=jnp.bool_),
                source_binding_valid=source_valid,
                replacement_stomp_delta_exact=coordinator_exact,
                replacement_prototype_exact=coordinator_exact,
                coordinator_wrapper_delta_exact=coordinator_exact,
                stomp_owner_matches_final_live=coordinator_exact,
                stomp_clocks_preserved=coordinator_exact,
                evaluated_stomp_digest=stomp_typed_tree_digest(source_stomp),
                replacement_stomp_digest=stomp_typed_tree_digest(source_stomp),
                final_live_stomp_digest=stomp_typed_tree_digest(source_stomp),
                metadata_rebased=coordinator_exact,
                additional_coordinator_evaluations=jnp.asarray(0, dtype=jnp.int32),
                additional_prototype_evaluations=jnp.asarray(0, dtype=jnp.int32),
                additional_stomp_update_evaluations=jnp.asarray(0, dtype=jnp.int32),
                additional_lifecycle_observations=jnp.asarray(0, dtype=jnp.int32),
                transaction_applied=valid,
            )

        decision = replacement.dispatch_replacement
        owner = decision.owner
        changed = decision.applied
        expected_stomp = source_stomp.replace(
            last_primitive_action=jnp.where(
                changed,
                replacement.action,
                source_stomp.last_primitive_action,
            ),
            base_last_action=jnp.where(
                changed & (owner == DISPATCH_OWNER_BASE_PRIMITIVE),
                replacement.action,
                source_stomp.base_last_action,
            ),
            option_last_intra_action=jnp.where(
                changed & (owner == DISPATCH_OWNER_OPTION),
                replacement.action,
                source_stomp.option_last_intra_action,
            ),
        )
        replacement_oak = _prototype_oak_state(replacement.state.oak_state)
        replacement_stomp = replacement_oak.stomp_state
        stomp_delta_exact = (
            replacement.committed
            & decision.state_valid
            & decision.observation_matches
            & decision.proposed_action_valid
            & decision.counterfactual_action_safe
            & (~decision.failed_closed)
            & (replacement.action == decision.effective_action)
            & _tree_exact_equal(replacement_stomp, expected_stomp)
        )
        expected_oak = source_oak.replace(stomp_state=expected_stomp)
        expected_prototype = _replace_prototype_oak_state(
            source_prototype,
            expected_oak,
        ).replace(current_action=replacement.action)
        prototype_exact = (
            replacement.decision_id_matches
            & replacement.observation_matches
            & replacement.state_valid_before
            & replacement.state_valid_after
            & _tree_exact_equal(replacement.state, expected_prototype)
        )
        expected_inner = source_coordinator.inner_state.replace(
            prototype_state=replacement.state
        )
        expected_coordinator = source_coordinator.replace(
            inner_state=expected_inner,
            current_action=replacement.action,
            current_decision_id=replacement.state.current_decision_id,
            cached_prototype_step_words=replacement.state.step_words,
            cached_feature_generation_words=(
                self._coordinator._feature_generation_words(expected_inner)
            ),
        )
        coordinator_exact = _tree_exact_equal(
            final_live_coordinator_state,
            expected_coordinator,
        )
        final_stomp = self._oak_state(final_live_coordinator_state).stomp_state
        final_owner_exact = (
            _tree_exact_equal(final_stomp, replacement_stomp)
            & _tree_exact_equal(
                self._prototype_state(final_live_coordinator_state),
                replacement.state,
            )
        )
        clocks_preserved = (
            (source_stomp.step_count == final_stomp.step_count)
            & jnp.array_equal(source_stomp.step_words, final_stomp.step_words)
            & (
                source_stomp.base_learner_state.step_count
                == final_stomp.base_learner_state.step_count
            )
            & jnp.array_equal(
                source_stomp.base_learner_state.step_words,
                final_stomp.base_learner_state.step_words,
            )
            & (source_stomp.executing_option == final_stomp.executing_option)
        )
        projected_repeated, projected_valid = self._project_repeated_stomp_owner(
            source_repeated,
            final_stomp,
        )
        candidate = self._compose_destination(
            state,
            final_live_coordinator_state,
            projected_repeated,
        )
        candidate_valid = self.state_valid(candidate)
        applied = (
            source_valid
            & stomp_delta_exact
            & prototype_exact
            & coordinator_exact
            & final_owner_exact
            & clocks_preserved
            & projected_valid
            & candidate_valid
        )
        selected = cast(
            ExternalCoordinatorRepeatedOptionSidecarState,
            _tree_select(applied, candidate, state),
        )
        return ExternalCoordinatorRepeatedOptionLiveActionProjectionResult(
            state=selected,
            metadata=self.detach_borrowed_metadata(selected),
            replacement_supplied=supplied,
            replacement_committed=replacement.committed,
            source_binding_valid=source_valid,
            replacement_stomp_delta_exact=stomp_delta_exact,
            replacement_prototype_exact=prototype_exact,
            coordinator_wrapper_delta_exact=coordinator_exact,
            stomp_owner_matches_final_live=final_owner_exact,
            stomp_clocks_preserved=clocks_preserved,
            evaluated_stomp_digest=stomp_typed_tree_digest(source_stomp),
            replacement_stomp_digest=stomp_typed_tree_digest(replacement_stomp),
            final_live_stomp_digest=stomp_typed_tree_digest(final_stomp),
            metadata_rebased=applied,
            additional_coordinator_evaluations=jnp.asarray(0, dtype=jnp.int32),
            additional_prototype_evaluations=jnp.asarray(0, dtype=jnp.int32),
            additional_stomp_update_evaluations=jnp.asarray(0, dtype=jnp.int32),
            additional_lifecycle_observations=jnp.asarray(0, dtype=jnp.int32),
            transaction_applied=applied,
        )

    def _lift_lifecycle_metadata(
        self,
        source: AuthorizedOptionReplacementState,
        lifecycle_metadata: Any,
        destination_stomp: Any,
    ) -> tuple[AuthorizedOptionReplacementMetadataState, Array]:
        """Lift one external lifecycle observation through the owner metadata."""

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
        next_authority = self._replacement._with_checksum(
            source.replace(
                scheduler_state=next_scheduler,
                canonical_scheduler_checksum=next_scheduler.binding_checksum,
                binding_checksum=jnp.zeros((2,), dtype=jnp.uint32),
            )
        )
        valid = (
            attached.transaction_applied
            & control_capacity
            & self._replacement.state_valid(next_authority)
        )
        return self._replacement.detach_borrowed_stomp(next_authority), valid

    def _rebase_repeated(
        self,
        source: RepeatedOptionLifecycleState,
        authority_metadata: AuthorizedOptionReplacementMetadataState,
        destination_stomp: Any,
    ) -> tuple[RepeatedOptionLifecycleState, Array]:
        destination = self._replacement.attach_borrowed_stomp(
            authority_metadata,
            destination_stomp,
        )
        rebased = self._lifecycle._with_checksum(
            dataclasses.replace(
                source,
                cycle_state=destination.state,
                revision=_saturating_increment(source.revision),
                binding_checksum=jnp.zeros((2,), dtype=jnp.uint32),
            )
        )
        return rebased, destination.transaction_applied & self._lifecycle.state_valid(rebased)

    def _compose_destination(
        self,
        source: ExternalCoordinatorRepeatedOptionSidecarState,
        coordinator_state: ExternalLearnedStateRouterAuditCoordinatorState,
        repeated_state: RepeatedOptionLifecycleState,
    ) -> ExternalCoordinatorRepeatedOptionSidecarState:
        return self._with_checksum(
            ExternalCoordinatorRepeatedOptionSidecarState(
                coordinator_state=coordinator_state,
                authority_metadata=self._replacement.detach_borrowed_stomp(
                    repeated_state.cycle_state
                ),
                lifecycle_metadata=self._detach_lifecycle(repeated_state),
                extended_action_mask=self._lifecycle.extended_action_mask(
                    repeated_state
                ),
                revision=_saturating_increment(source.revision),
                binding_checksum=jnp.zeros((2,), dtype=jnp.uint32),
            )
        )

    def start(
        self,
        state: ExternalCoordinatorRepeatedOptionSidecarState,
        initial_observation: Array,
    ) -> ExternalCoordinatorRepeatedOptionSidecarStartResult:
        """Start the coordinator and advance only its borrowed lifecycle metadata."""

        self._check_state_contract(state)
        source_repeated, attached = self._attach_source(state)
        source_valid = self.state_valid(state) & attached
        source_oak = self._oak_state(state.coordinator_state)
        destination_coordinator = self._coordinator.start(
            state.coordinator_state,
            initial_observation,
            extended_action_mask=state.extended_action_mask,
        )
        destination_oak = self._oak_state(destination_coordinator)
        installation = source_repeated.cycle_state.scheduler_state.installation_state
        lifecycle_api = self._installation.lifecycle.with_external_semantic_digests(
            installation.installed_semantic_digests
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
        lifecycle_attempt = lifecycle_api.adopt_external_stomp_start(
            lifecycle_metadata,
            source_oak.stomp_state,
            destination_oak.stomp_state,
            destination_oak.stomp_state.base_last_obs,
            declaration,
        )
        lifted, lifted_valid = self._lift_lifecycle_metadata(
            source_repeated.cycle_state,
            lifecycle_attempt.state,
            destination_oak.stomp_state,
        )
        rebased, rebased_valid = self._rebase_repeated(
            source_repeated,
            lifted,
            destination_oak.stomp_state,
        )
        coordinator_started_attempted = (
            (~state.coordinator_state.started)
            & destination_coordinator.started
            & self._coordinator.state_valid(destination_coordinator)
        )
        candidate = self._compose_destination(
            state,
            destination_coordinator,
            rebased,
        )
        candidate_valid = self.state_valid(candidate)
        applied = (
            source_valid
            & coordinator_started_attempted
            & lifecycle_attempt.metadata_advanced
            & lifted_valid
            & rebased_valid
            & candidate_valid
        )
        selected = cast(
            ExternalCoordinatorRepeatedOptionSidecarState,
            _tree_select(applied, candidate, state),
        )
        return ExternalCoordinatorRepeatedOptionSidecarStartResult(
            state=selected,
            lifecycle_attempt=lifecycle_attempt,
            source_binding_valid=source_valid,
            coordinator_started_attempted=coordinator_started_attempted,
            coordinator_started=applied,
            lifecycle_metadata_applied=applied,
            delight_available=jnp.asarray(False, dtype=jnp.bool_),
            transaction_applied=applied,
        )

    def adopt_evaluated_transition(
        self,
        state: ExternalCoordinatorRepeatedOptionSidecarState,
        evaluated: ExternalLearnedStateRouterAuditCoordinatorEvaluatedTransition,
        receipt: ExternalLearnedStateRouterAuditCoordinatorIntegrityReceipt,
        *,
        context: int | Array = 0,
        idle_candidate_option: int | Array = 0,
        idle_initiation_eligible: bool | Array = False,
        comparator_randomized: bool | Array = False,
        treatment_propensity: float | Array = 0.0,
        downstream_candidate_valid: bool | Array = True,
    ) -> ExternalCoordinatorRepeatedOptionSidecarResult:
        """Adopt one coordinator candidate and its lifecycle observation together."""

        self._check_state_contract(state)
        if type(evaluated) is not (
            ExternalLearnedStateRouterAuditCoordinatorEvaluatedTransition
        ):
            raise TypeError("evaluated must be an exact coordinator evaluation")
        if type(receipt) is not (
            ExternalLearnedStateRouterAuditCoordinatorIntegrityReceipt
        ):
            raise TypeError("receipt must be an exact coordinator receipt")
        safe_context, context_valid = _normalize_int(
            context,
            name="context",
            lower=0,
            upper=self._installation.lifecycle.audit.config.n_contexts,
        )
        safe_idle, idle_valid = _normalize_int(
            idle_candidate_option,
            name="idle_candidate_option",
            lower=0,
            upper=self._prototype.config.oak.stomp.n_options,
        )
        safe_eligible = _normalize_bool(
            idle_initiation_eligible,
            name="idle_initiation_eligible",
        )
        safe_randomized = _normalize_bool(
            comparator_randomized,
            name="comparator_randomized",
        )
        safe_propensity, propensity_valid = _normalize_propensity(
            treatment_propensity
        )
        downstream = _normalize_bool(
            downstream_candidate_valid,
            name="downstream_candidate_valid",
        )
        audit_inputs_valid = context_valid & idle_valid & propensity_valid

        source_repeated, attached = self._attach_source(state)
        source_valid = self.state_valid(state) & attached
        coordinator_result = self._coordinator.adopt_evaluated_transition(
            state.coordinator_state,
            evaluated,
            receipt,
        )
        prototype_result = evaluated.prepared.inner_result.prototype_result
        trace = prototype_result.oak_owner_finalization_trace
        raw_result = prototype_result.oak_stomp_update_result
        source_oak = self._oak_state(state.coordinator_state)
        destination_oak = self._oak_state(evaluated.candidate_state)
        installation = source_repeated.cycle_state.scheduler_state.installation_state
        lifecycle_api = self._installation.lifecycle.with_external_semantic_digests(
            installation.installed_semantic_digests
        )
        lifecycle_metadata = (
            state.authority_metadata.scheduler_metadata.installation_metadata
            .lifecycle_metadata
        )
        transition = evaluated.prepared.transition
        declaration = lifecycle_api.declare_external_stomp_transition(
            lifecycle_metadata,
            source_oak.stomp_state,
            raw_result,
            env_reward=transition.reward,
            next_observation=prototype_result.oak_bootstrap_observation,
            discount=transition.discount,
            execution_boundary=prototype_result.oak_execution_boundary,
            extended_action_mask=state.extended_action_mask,
            caller_derivation_declared=prototype_result.oak_stomp_update_available,
        )
        lifecycle_attempt = lifecycle_api.adopt_external_stomp_update(
            lifecycle_metadata,
            source_oak.stomp_state,
            raw_result,
            declaration,
            env_reward=transition.reward,
            next_observation=prototype_result.oak_bootstrap_observation,
            discount=transition.discount,
            decision_observation=prototype_result.oak_decision_observation,
            execution_boundary=prototype_result.oak_execution_boundary,
            context=safe_context,
            idle_candidate_option=safe_idle,
            idle_initiation_eligible=safe_eligible,
            comparator_randomized=safe_randomized,
            treatment_propensity=safe_propensity,
            extended_action_mask=state.extended_action_mask,
        )
        lifecycle_finalization = lifecycle_api.finalize_external_stomp_owner(
            lifecycle_attempt.state,
            trace,
        )

        raw_bound = (
            prototype_result.oak_stomp_update_available
            & _tree_exact_equal(raw_result.state, trace.raw_state)
        )
        raw_digest_bound = (
            jnp.array_equal(stomp_typed_tree_digest(raw_result.state), trace.raw_digest)
            & jnp.array_equal(stomp_typed_tree_digest(trace.raw_state), trace.raw_digest)
        )
        final_bound = _tree_exact_equal(
            trace.final_state,
            destination_oak.stomp_state,
        )
        final_digest_bound = (
            jnp.array_equal(
                stomp_typed_tree_digest(destination_oak.stomp_state),
                trace.final_digest,
            )
            & jnp.array_equal(
                stomp_typed_tree_digest(trace.final_state),
                trace.final_digest,
            )
        )
        lifted, lifted_valid = self._lift_lifecycle_metadata(
            source_repeated.cycle_state,
            lifecycle_finalization.state,
            destination_oak.stomp_state,
        )
        rebased, rebased_valid = self._rebase_repeated(
            source_repeated,
            lifted,
            destination_oak.stomp_state,
        )
        candidate = self._compose_destination(
            state,
            evaluated.candidate_state,
            rebased,
        )
        candidate_valid = self.state_valid(candidate)
        coordinator_attempted = coordinator_result.diagnostics.transaction_applied
        applied = (
            source_valid
            & coordinator_attempted
            & audit_inputs_valid
            & lifecycle_attempt.metadata_advanced
            & lifecycle_attempt.transaction_applied
            & lifecycle_finalization.metadata_finalized
            & raw_bound
            & raw_digest_bound
            & final_bound
            & final_digest_bound
            & (prototype_result.oak_real_stomp_update_evaluations == 1)
            & lifted_valid
            & rebased_valid
            & downstream
            & candidate_valid
        )
        selected = cast(
            ExternalCoordinatorRepeatedOptionSidecarState,
            _tree_select(applied, candidate, state),
        )
        return ExternalCoordinatorRepeatedOptionSidecarResult(
            state=selected,
            coordinator_result=coordinator_result,
            lifecycle_attempt=lifecycle_attempt,
            lifecycle_owner_finalization=lifecycle_finalization,
            source_binding_valid=source_valid,
            audit_inputs_valid=audit_inputs_valid,
            raw_stomp_result_bound=raw_bound,
            raw_stomp_result_digest_bound=raw_digest_bound,
            finalization_trace_bound=final_bound,
            final_stomp_owner_digest_bound=final_digest_bound,
            raw_stomp_update_evaluations=(
                prototype_result.oak_real_stomp_update_evaluations
            ),
            additional_stomp_update_evaluations=jnp.asarray(0, dtype=jnp.int32),
            lifecycle_observation_evaluations=jnp.asarray(1, dtype=jnp.int32),
            raw_stomp_result_consumed=applied,
            coordinator_update_applied=applied,
            lifecycle_metadata_applied=applied,
            delight_available=jnp.asarray(False, dtype=jnp.bool_),
            additional_delight_evaluations=jnp.asarray(0, dtype=jnp.int32),
            additional_actor_backward_calls=jnp.asarray(0, dtype=jnp.int32),
            downstream_candidate_valid=downstream,
            candidate_state_valid=candidate_valid,
            transaction_applied=applied,
        )

    def _atomic_prepared_payload_arrays(
        self,
        prepared: ExternalCoordinatorRepeatedOptionAtomicSwapPrepared,
    ) -> tuple[Array, ...]:
        return tuple(
            cast(Array, leaf)
            for leaf in jax.tree.leaves(
                tuple(
                    getattr(prepared, field.name)
                    for field in dataclasses.fields(
                        ExternalCoordinatorRepeatedOptionAtomicSwapPrepared
                    )
                    if field.name != "prepared_checksum"
                )
            )
        )

    def _with_atomic_prepared_checksum(
        self,
        prepared: ExternalCoordinatorRepeatedOptionAtomicSwapPrepared,
    ) -> ExternalCoordinatorRepeatedOptionAtomicSwapPrepared:
        return dataclasses.replace(
            prepared,
            prepared_checksum=_checksum_arrays(
                self._atomic_prepared_payload_arrays(prepared)
            ),
        )

    def _check_atomic_prepared_contract(
        self,
        prepared: ExternalCoordinatorRepeatedOptionAtomicSwapPrepared,
    ) -> None:
        if type(prepared) is not ExternalCoordinatorRepeatedOptionAtomicSwapPrepared:
            raise TypeError("prepared has the wrong exact atomic-swap type")
        self._check_state_contract(prepared.source_state)
        _require_threefry_key(prepared.cycle_key, name="prepared.cycle_key")
        if type(prepared.atomic_swap_prepared) is not AuthorizedOptionAtomicSwapPrepared:
            raise TypeError("prepared.atomic_swap_prepared has the wrong exact type")
        for value, name in (
            (prepared.source_binding_valid, "source_binding_valid"),
            (prepared.cycle_capacity_available, "cycle_capacity_available"),
            (prepared.cycle_key_fresh, "cycle_key_fresh"),
            (
                prepared.retirement_authority_revision_fresh,
                "retirement_authority_revision_fresh",
            ),
            (prepared.preparation_valid, "preparation_valid"),
        ):
            _require_array(
                value,
                name=f"prepared.{name}",
                shape=(),
                dtype=jnp.bool_,
            )
        _require_array(
            prepared.prepared_checksum,
            name="prepared.prepared_checksum",
            shape=(2,),
            dtype=jnp.uint32,
        )

    def prepare_atomic_swap(
        self,
        state: ExternalCoordinatorRepeatedOptionSidecarState,
        cycle_key: Array,
        retirement_handoff: CumulantOptionRetirementHandoff,
        retirement_authority: OptionRetirementAuthorityReceipt,
        phase_one_key: Array,
        phase_two_key: Array,
        arm_inputs: CumulantOptionSchedulerArmInputs,
        observation: CumulantOptionSchedulerObservation,
        live_inputs: CumulantOptionLiveInputs,
    ) -> ExternalCoordinatorRepeatedOptionAtomicSwapPrepared:
        """Host-prepare one coupled all-installed-to-all-installed swap."""

        if _contains_tracer(
            (
                state,
                cycle_key,
                retirement_handoff,
                retirement_authority,
                phase_one_key,
                phase_two_key,
                arm_inputs,
                observation,
                live_inputs,
            )
        ):
            raise TypeError("atomic swap preparation is host-only")
        self._check_state_contract(state)
        key = _require_threefry_key(cycle_key, name="cycle_key")
        repeated, attached = self._attach_source(state)
        source_valid = self.state_valid(state) & attached
        atomic = self._atomic_swap.prepare(
            repeated.cycle_state,
            retirement_handoff,
            retirement_authority,
            phase_one_key,
            phase_two_key,
            arm_inputs,
            observation,
            live_inputs,
        )
        key_data = jr.key_data(key)
        nonzero_key = jnp.any(key_data != 0)
        capacity = (
            (~repeated.cycle_key_active)
            & (~repeated.unavailable)
            & (repeated.completed_cycles < self._lifecycle.config.max_cycles)
            & jnp.all(repeated.cycle_state.installed_slot_mask)
        )
        key_fresh = nonzero_key & self._lifecycle._cycle_key_fresh(
            repeated,
            key_data,
        )
        retirement_fresh = _words_less(
            repeated.last_retirement_authority_revision_words,
            retirement_authority.authority_revision_words,
        )
        preparation_valid = (
            source_valid
            & capacity
            & key_fresh
            & retirement_fresh
            & atomic.diagnostics.atomic_swap_ready
        )
        return self._with_atomic_prepared_checksum(
            ExternalCoordinatorRepeatedOptionAtomicSwapPrepared(
                source_state=state,
                cycle_key=key,
                atomic_swap_prepared=atomic,
                source_binding_valid=source_valid,
                cycle_capacity_available=capacity,
                cycle_key_fresh=key_fresh,
                retirement_authority_revision_fresh=retirement_fresh,
                preparation_valid=preparation_valid,
                prepared_checksum=jnp.zeros((2,), dtype=jnp.uint32),
            )
        )

    def _check_atomic_authority_contract(
        self,
        receipt: ExternalCoordinatorRepeatedOptionAtomicSwapAuthorityReceipt,
    ) -> None:
        if type(receipt) is not (
            ExternalCoordinatorRepeatedOptionAtomicSwapAuthorityReceipt
        ):
            raise TypeError("authority receipt has the wrong exact atomic-swap type")
        self._atomic_swap._check_authority_contract(receipt.atomic_swap_authority)
        contracts = (
            (receipt.cycle_index, "cycle_index", (), jnp.int32),
            (receipt.cycle_key_data, "cycle_key_data", (2,), jnp.uint32),
            (
                receipt.source_binding_checksum,
                "source_binding_checksum",
                (2,),
                jnp.uint32,
            ),
            (receipt.source_revision, "source_revision", (), jnp.int32),
            (receipt.prepared_checksum, "prepared_checksum", (2,), jnp.uint32),
            (
                receipt.retirement_authority_revision_fresh,
                "retirement_authority_revision_fresh",
                (),
                jnp.bool_,
            ),
            (
                receipt.replacement_authority_revision_fresh,
                "replacement_authority_revision_fresh",
                (),
                jnp.bool_,
            ),
            (receipt.caller_authenticated, "caller_authenticated", (), jnp.bool_),
        )
        for value, name, shape, dtype in contracts:
            _require_array(
                value,
                name=f"authority_receipt.{name}",
                shape=shape,
                dtype=dtype,
            )

    def authorize_atomic_swap(
        self,
        state: ExternalCoordinatorRepeatedOptionSidecarState,
        prepared: ExternalCoordinatorRepeatedOptionAtomicSwapPrepared,
        installation_authority: CumulantOptionInstallationAuthorityReceipt,
        cycle_key: Array,
        *,
        swap_authorized: bool | Array,
    ) -> ExternalCoordinatorRepeatedOptionAtomicSwapAuthorityReceipt:
        """Bind caller declarations to one complete, source-exact host swap."""

        if _contains_tracer(
            (state, prepared, installation_authority, cycle_key, swap_authorized)
        ):
            raise TypeError("atomic swap authorization is host-only")
        self._check_state_contract(state)
        self._check_atomic_prepared_contract(prepared)
        key = _require_threefry_key(cycle_key, name="cycle_key")
        authorized = _normalize_bool(swap_authorized, name="swap_authorized")
        repeated, attached = self._attach_source(state)
        source_matches = _tree_exact_equal(state, prepared.source_state)
        integrity = jnp.array_equal(
            prepared.prepared_checksum,
            _checksum_arrays(self._atomic_prepared_payload_arrays(prepared)),
        )
        key_matches = jnp.array_equal(jr.key_data(key), jr.key_data(prepared.cycle_key))
        replacement_fresh = _words_less(
            repeated.last_replacement_authority_revision_words,
            installation_authority.authority_revision_words,
        )
        wrapper_valid = (
            self.state_valid(state)
            & attached
            & source_matches
            & integrity
            & key_matches
            & prepared.preparation_valid
            & prepared.retirement_authority_revision_fresh
            & replacement_fresh
        )
        nested = self._atomic_swap.authority_receipt(
            prepared.atomic_swap_prepared,
            installation_authority,
            swap_authorized=authorized & wrapper_valid,
        )
        return ExternalCoordinatorRepeatedOptionAtomicSwapAuthorityReceipt(
            atomic_swap_authority=nested,
            cycle_index=repeated.completed_cycles,
            cycle_key_data=jr.key_data(key),
            source_binding_checksum=state.binding_checksum,
            source_revision=state.revision,
            prepared_checksum=prepared.prepared_checksum,
            retirement_authority_revision_fresh=(
                prepared.retirement_authority_revision_fresh
            ),
            replacement_authority_revision_fresh=replacement_fresh,
            caller_authenticated=jnp.asarray(False, dtype=jnp.bool_),
        )

    def _atomic_authority_binding_valid(
        self,
        source: ExternalCoordinatorRepeatedOptionSidecarState,
        repeated: RepeatedOptionLifecycleState,
        prepared: ExternalCoordinatorRepeatedOptionAtomicSwapPrepared,
        receipt: ExternalCoordinatorRepeatedOptionAtomicSwapAuthorityReceipt,
        key_data: Array,
    ) -> Array:
        nested = receipt.atomic_swap_authority
        expected_nested = self._atomic_swap.authority_receipt(
            prepared.atomic_swap_prepared,
            nested.replacement_authority.installation_authority,
            swap_authorized=nested.swap_authorized,
        )
        retirement_fresh = _words_less(
            repeated.last_retirement_authority_revision_words,
            prepared.atomic_swap_prepared.retirement_authority.authority_revision_words,
        )
        replacement_fresh = _words_less(
            repeated.last_replacement_authority_revision_words,
            nested.replacement_authority.installation_authority.authority_revision_words,
        )
        return (
            _tree_exact_equal(nested, expected_nested)
            & (receipt.cycle_index == repeated.completed_cycles)
            & jnp.array_equal(receipt.cycle_key_data, key_data)
            & jnp.array_equal(
                receipt.source_binding_checksum,
                source.binding_checksum,
            )
            & (receipt.source_revision == source.revision)
            & jnp.array_equal(receipt.prepared_checksum, prepared.prepared_checksum)
            & receipt.retirement_authority_revision_fresh
            & receipt.replacement_authority_revision_fresh
            & retirement_fresh
            & replacement_fresh
            & (~receipt.caller_authenticated)
        )

    def _rolled_atomic_repeated(
        self,
        source: RepeatedOptionLifecycleState,
        atomic: AuthorizedOptionAtomicSwapResult,
        receipt: ExternalCoordinatorRepeatedOptionAtomicSwapAuthorityReceipt,
        key_data: Array,
    ) -> RepeatedOptionLifecycleState:
        rolled_child = self._replacement.init(
            atomic.state.scheduler_state,
            retirement_authority_issuer_digest=(
                source.cycle_state.expected_retirement_authority_issuer_digest
            ),
            controller_owner_digest=source.cycle_state.controller_owner_digest,
        )
        next_completed = source.completed_cycles + jnp.int32(1)
        exhausted = next_completed == self._lifecycle.config.max_cycles
        next_revision = _saturating_increment(
            _saturating_increment(source.revision)
        )
        return self._lifecycle._with_checksum(
            dataclasses.replace(
                source,
                cycle_state=rolled_child,
                completed_cycles=next_completed,
                total_retirements=source.total_retirements + jnp.int32(1),
                total_replacements=source.total_replacements + jnp.int32(1),
                cycle_key_active=jnp.asarray(False, dtype=jnp.bool_),
                active_cycle_key_data=jnp.zeros((2,), dtype=jnp.uint32),
                has_completed_cycle=jnp.asarray(True, dtype=jnp.bool_),
                cycle_key_history=source.cycle_key_history.at[
                    source.completed_cycles
                ].set(key_data),
                last_completed_cycle_key_data=key_data,
                last_retirement_authority_revision_words=(
                    atomic.retirement_result.state
                    .last_retirement_authority_revision_words
                ),
                last_replacement_authority_revision_words=(
                    receipt.atomic_swap_authority.replacement_authority
                    .installation_authority.authority_revision_words
                ),
                revision=next_revision,
                unavailable=exhausted,
                error=jnp.where(
                    exhausted,
                    jnp.asarray(
                        REPEATED_OPTION_LIFECYCLE_ERROR_CAPACITY,
                        dtype=jnp.int32,
                    ),
                    jnp.asarray(
                        REPEATED_OPTION_LIFECYCLE_ERROR_NONE,
                        dtype=jnp.int32,
                    ),
                ),
                binding_checksum=jnp.zeros((2,), dtype=jnp.uint32),
            )
        )

    def adopt_atomic_swap(
        self,
        state: ExternalCoordinatorRepeatedOptionSidecarState,
        prepared: ExternalCoordinatorRepeatedOptionAtomicSwapPrepared,
        authority_receipt: ExternalCoordinatorRepeatedOptionAtomicSwapAuthorityReceipt,
        cycle_key: Array,
        *,
        downstream_candidate_valid: bool | Array = True,
    ) -> ExternalCoordinatorRepeatedOptionAtomicSwapResult:
        """Host-rederive, exact-rebind, and atomically adopt one full swap."""

        if _contains_tracer(
            (
                state,
                prepared,
                authority_receipt,
                cycle_key,
                downstream_candidate_valid,
            )
        ):
            raise TypeError("atomic swap adoption is host-only")
        self._check_state_contract(state)
        self._check_atomic_prepared_contract(prepared)
        self._check_atomic_authority_contract(authority_receipt)
        key = _require_threefry_key(cycle_key, name="cycle_key")
        downstream = _normalize_bool(
            downstream_candidate_valid,
            name="downstream_candidate_valid",
        )
        repeated, attached = self._attach_source(state)
        source_valid = self.state_valid(state) & attached
        integrity = jnp.array_equal(
            prepared.prepared_checksum,
            _checksum_arrays(self._atomic_prepared_payload_arrays(prepared)),
        )
        lower = prepared.atomic_swap_prepared
        recomputed = self.prepare_atomic_swap(
            prepared.source_state,
            prepared.cycle_key,
            lower.retirement_handoff,
            lower.retirement_authority,
            lower.phase_one_key,
            lower.phase_two_key,
            lower.arm_inputs,
            lower.observation,
            lower.live_inputs,
        )
        derivation = _tree_exact_equal(prepared, recomputed)
        destination_matches = _tree_exact_equal(state, prepared.source_state)
        key_data = jr.key_data(key)
        key_matches = (
            jnp.array_equal(key_data, jr.key_data(prepared.cycle_key))
            & jnp.array_equal(key_data, authority_receipt.cycle_key_data)
        )
        key_fresh = (
            recomputed.cycle_key_fresh
            & self._lifecycle._cycle_key_fresh(repeated, key_data)
            & key_matches
        )
        authority_valid = self._atomic_authority_binding_valid(
            state,
            repeated,
            recomputed,
            authority_receipt,
            key_data,
        )
        atomic_attempt = self._atomic_swap.commit(
            repeated.cycle_state,
            recomputed.atomic_swap_prepared,
            authority_receipt.atomic_swap_authority,
        )
        source_oak = self._oak_state(state.coordinator_state)
        destination_stomp = (
            atomic_attempt.state.scheduler_state.installation_state.lifecycle_state
            .stomp_state
        )
        oak_rebind = self._oak.rebind_option_slots(
            source_oak,
            destination_stomp,
            atomic_attempt.reset_slots,
        )
        exact_owner = oak_rebind.transaction_applied & _tree_exact_equal(
            oak_rebind.state.stomp_state,
            destination_stomp,
        )
        all_before = jnp.all(repeated.cycle_state.installed_slot_mask)
        all_after = jnp.all(atomic_attempt.state.installed_slot_mask)
        rolled = self._rolled_atomic_repeated(
            repeated,
            atomic_attempt,
            authority_receipt,
            key_data,
        )
        rolled_valid = self._lifecycle.state_valid(rolled)
        coordinator_candidate = self._replace_coordinator_oak_state(
            state.coordinator_state,
            oak_rebind.state,
        )
        candidate = self._compose_destination(
            state,
            coordinator_candidate,
            rolled,
        )
        candidate_valid = self.state_valid(candidate)
        applied = (
            source_valid
            & destination_matches
            & integrity
            & derivation
            & recomputed.preparation_valid
            & key_fresh
            & authority_valid
            & atomic_attempt.transaction_applied
            & atomic_attempt.retirement_applied
            & atomic_attempt.replacement_applied
            & (~atomic_attempt.cold_state_persisted)
            & all_before
            & all_after
            & exact_owner
            & rolled_valid
            & downstream
            & candidate_valid
        )
        selected = cast(
            ExternalCoordinatorRepeatedOptionSidecarState,
            _tree_select(applied, candidate, state),
        )
        return ExternalCoordinatorRepeatedOptionAtomicSwapResult(
            state=selected,
            atomic_swap_attempt=atomic_attempt,
            oak_rebind=oak_rebind,
            destination_matches_source=destination_matches,
            prepared_integrity_valid=integrity,
            preparation_derivation_valid=derivation,
            authority_binding_valid=authority_valid,
            cycle_key_fresh=key_fresh,
            exact_owner_rebind=exact_owner,
            all_slots_installed_before=all_before,
            all_slots_installed_after=all_after,
            cold_state_persisted=atomic_attempt.cold_state_persisted,
            retirement_applied=applied & atomic_attempt.retirement_applied,
            replacement_applied=applied & atomic_attempt.replacement_applied,
            downstream_candidate_valid=downstream,
            candidate_state_valid=candidate_valid,
            caller_authenticated=jnp.asarray(False, dtype=jnp.bool_),
            transaction_applied=applied,
        )

    def resource_budget(
        self,
        state: ExternalCoordinatorRepeatedOptionSidecarState,
    ) -> ExternalCoordinatorRepeatedOptionSidecarResourceBudget:
        """Measure the one-owner persistence and fixed sidecar work boundary."""

        if not bool(jax.device_get(self.state_valid(state))):
            raise ValueError("resource measurement requires a valid sidecar state")
        total = _tree_nbytes(state)
        coordinator = _tree_nbytes(state.coordinator_state)
        authority = _tree_nbytes(state.authority_metadata)
        overlay = _tree_nbytes(state.lifecycle_metadata)
        return ExternalCoordinatorRepeatedOptionSidecarResourceBudget(
            persistent_state_nbytes=total,
            coordinator_state_nbytes=coordinator,
            detached_authority_metadata_nbytes=authority,
            repeated_overlay_nbytes=overlay,
            mask_revision_binding_nbytes=(
                total - coordinator - authority - overlay
            ),
            coordinator_state_owners=1,
            prototype_state_owners=1,
            oak_state_owners=1,
            stomp_state_owners=1,
            detached_authority_metadata_stomp_state_owners=0,
            repeated_overlay_stomp_state_owners=0,
            borrowed_stomp_bindings=1,
            additional_stomp_update_evaluations_per_adoption=0,
            maximum_lifecycle_observations_per_adoption=1,
            atomic_swap_prepare_host_only=True,
            atomic_swap_adopt_host_only=True,
            delight_available=False,
            additional_delight_evaluations=0,
            additional_actor_backward_calls=0,
            output_write_calls=0,
            artifact_bytes_written=0,
        )


__all__ = [
    "EXTERNAL_COORDINATOR_REPEATED_OPTION_SIDECAR_ASSESSMENT",
    "EXTERNAL_COORDINATOR_REPEATED_OPTION_SIDECAR_EVIDENCE_LEVEL",
    "EXTERNAL_COORDINATOR_REPEATED_OPTION_SIDECAR_LIMITATIONS",
    "EXTERNAL_COORDINATOR_REPEATED_OPTION_SIDECAR_STATE_SCHEMA",
    "EXTERNAL_COORDINATOR_REPEATED_OPTION_SIDECAR_STATUS",
    "ExternalCoordinatorRepeatedOptionAtomicSwapAuthorityReceipt",
    "ExternalCoordinatorRepeatedOptionAtomicSwapPrepared",
    "ExternalCoordinatorRepeatedOptionAtomicSwapResult",
    "ExternalCoordinatorRepeatedOptionBorrowedMetadata",
    "ExternalCoordinatorRepeatedOptionBorrowedMetadataAttachResult",
    "ExternalCoordinatorRepeatedOptionLifecycleMetadata",
    "ExternalCoordinatorRepeatedOptionLiveActionProjectionResult",
    "ExternalCoordinatorRepeatedOptionSidecar",
    "ExternalCoordinatorRepeatedOptionSidecarResourceBudget",
    "ExternalCoordinatorRepeatedOptionSidecarResult",
    "ExternalCoordinatorRepeatedOptionSidecarStartResult",
    "ExternalCoordinatorRepeatedOptionSidecarState",
]
