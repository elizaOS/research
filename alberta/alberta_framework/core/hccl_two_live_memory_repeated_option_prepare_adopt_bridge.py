# mypy: disable-error-code="arg-type,attr-defined,call-arg,no-any-return,type-var"
"""No-planner HCCL prepare/adopt with two borrowed repeated-option sidecars.

This v0 composition retains exactly the persistent owners of
:class:`HCCLTwoLiveMemoryPrepareAdoptBridge`: one HCCL state and two live
memory adapter states.  Each live state contains its sole coordinator,
Prototype, OaK, and STOMP owner.  Two coordinator-free sidecar metadata
bundles add option-authority and repeated-cycle bookkeeping without copying
any of those owners.

Preparation evaluates the reviewed HCCL transaction once and each live
adapter once.  It then transiently attaches each metadata bundle and consumes
the already-evaluated raw STOMP result through that sidecar exactly once.  The
complete candidate binds the raw/final STOMP digests and owner-finalization
trace through the reviewed downstream receipt.  Adoption performs integrity
and child adoption only; it reevaluates no world, coordinator, Prototype,
STOMP learner, or learned-memory donor.

The same transient attachment exposes one host-only per-agent
all-installed-to-all-installed atomic option swap.  The other live agent,
HCCL world/attribution, learned memory, pending feedback binding, and action
masks are preserved bit-exactly.  Any refusal, tamper, replay, foreign
metadata, or outer veto returns the complete source.

P=M is the explicit no-planner rung.  Delight is unavailable: this module
does not ask whether a gradient sparks joy, never exposes ``sparks_joy``, and
runs no Kondo call or actor backward.
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
from alberta_framework.core.external_coordinator_repeated_option_sidecar import (
    ExternalCoordinatorRepeatedOptionAtomicSwapAuthorityReceipt,
    ExternalCoordinatorRepeatedOptionAtomicSwapPrepared,
    ExternalCoordinatorRepeatedOptionAtomicSwapResult,
    ExternalCoordinatorRepeatedOptionBorrowedMetadata,
    ExternalCoordinatorRepeatedOptionBorrowedMetadataAttachResult,
    ExternalCoordinatorRepeatedOptionLiveActionProjectionResult,
    ExternalCoordinatorRepeatedOptionSidecar,
    ExternalCoordinatorRepeatedOptionSidecarResult,
)
from alberta_framework.core.external_learned_state_live_memory_adapter import (
    ExternalLearnedStateLiveMemoryAdapterState,
    ExternalLearnedStateLiveMemoryEventInput,
)
from alberta_framework.core.external_learned_state_router_audit_coordinator import (
    ExternalBuilderCandidateAuditEvidence,
)
from alberta_framework.core.hccl_two_live_memory_bridge import (
    HCCLTwoLiveMemoryBridgeState,
    _contains_tracer,
    _require_array,
    _tree_exact_equal,
    _tree_select,
)
from alberta_framework.core.hccl_two_live_memory_prepare_adopt_bridge import (
    HCCLTwoLiveMemoryDownstreamAdoptionReceipt,
    HCCLTwoLiveMemoryPrepareAdoptBridge,
    HCCLTwoLiveMemoryPrepareAdoptResult,
    HCCLTwoLiveMemoryPreparedAgentFacts,
    HCCLTwoLiveMemoryPreparedTransaction,
    _tree_digest,
)
from alberta_framework.core.prototype_agent import (
    PrototypePartnerPolicyFusionFeedback,
    PrototypePartnerPolicyFusionInput,
)
from alberta_framework.core.prototype_option_authority_bridge import (
    _checksum_arrays,
    _saturating_increment,
)
from alberta_framework.core.repeated_option_lifecycle import (
    RepeatedOptionLifecycleState,
)

HCCL_TWO_LIVE_MEMORY_REPEATED_OPTION_PREPARE_ADOPT_STATE_SCHEMA = (
    "alberta.hccl-two-live-memory-repeated-option-prepare-adopt.state.v0"
)
HCCL_TWO_LIVE_MEMORY_REPEATED_OPTION_PREPARED_SCHEMA = (
    "alberta.hccl-two-live-memory-repeated-option-prepared.v0"
)
HCCL_TWO_LIVE_MEMORY_REPEATED_OPTION_PREPARATION_RECEIPT_SCHEMA = (
    "alberta.hccl-two-live-memory-repeated-option-preparation-receipt.v0"
)
HCCL_TWO_LIVE_MEMORY_REPEATED_OPTION_ATOMIC_PREPARED_SCHEMA = (
    "alberta.hccl-two-live-memory-repeated-option-atomic-prepared.v0"
)
HCCL_TWO_LIVE_MEMORY_REPEATED_OPTION_ATOMIC_AUTHORITY_SCHEMA = (
    "alberta.hccl-two-live-memory-repeated-option-atomic-authority.v0"
)
HCCL_TWO_LIVE_MEMORY_REPEATED_OPTION_PREPARE_ADOPT_STATUS = (
    "l0-development-hccl-two-live-memory-repeated-option-prepare-adopt-v0"
)
HCCL_TWO_LIVE_MEMORY_REPEATED_OPTION_PREPARE_ADOPT_EVIDENCE_LEVEL = "L0"
HCCL_TWO_LIVE_MEMORY_REPEATED_OPTION_PREPARE_ADOPT_LIMITATIONS = (
    "P-equals-M-no-planner-rung",
    "two-live-memory-and-two-borrowed-repeated-metadata-bundles-only",
    "preparations-and-atomic-swap-records-are-transient",
    "integrity-receipts-are-not-caller-authentication",
    "all-installed-to-all-installed-per-agent-atomic-swap-only",
    "host-eager-only",
    "no-cold-persistence-planner-physical-dispatch-or-Kondo-call",
    "delight-unavailable-and-zero-actor-backward",
    "no-safety-evidence-benefit-or-promotion-authority",
)

_N_AGENTS = 2
_DIGEST_WORDS = 8


def _tree_nbytes(value: object) -> int:
    total = 0
    for leaf in jax.tree.leaves(value):
        array = jnp.asarray(leaf)
        if jax.dtypes.issubdtype(array.dtype, jax.dtypes.prng_key):
            array = jr.key_data(array)
        total += int(array.nbytes)
    return total


def _exact_bool(value: bool | Array, *, label: str) -> Array:
    if type(value) is bool:
        return jnp.asarray(value, dtype=jnp.bool_)
    return _require_array(
        value,
        shape=(),
        dtype=jnp.dtype(jnp.bool_),
        label=label,
    )


def _exact_digest(value: Any, *, label: str) -> Array:
    return _require_array(
        value,
        shape=(_DIGEST_WORDS,),
        dtype=jnp.dtype(jnp.uint32),
        label=label,
    )


@chex.dataclass(frozen=True)
class HCCLTwoLiveMemoryRepeatedOptionPrepareAdoptState:
    """One HCCL/two-live owner plus two coordinator-free metadata bundles."""

    inner_state: HCCLTwoLiveMemoryBridgeState
    agent_0_metadata: ExternalCoordinatorRepeatedOptionBorrowedMetadata
    agent_1_metadata: ExternalCoordinatorRepeatedOptionBorrowedMetadata
    revision: Int[Array, ""]
    binding_checksum: UInt[Array, " 2"]


@chex.dataclass(frozen=True)
class HCCLTwoLiveMemoryRepeatedOptionPreparedAgent:
    """One already-evaluated live candidate and its lifecycle consumption."""

    agent_index: Int[Array, ""]
    inner_facts: HCCLTwoLiveMemoryPreparedAgentFacts
    source_metadata: ExternalCoordinatorRepeatedOptionBorrowedMetadata
    sidecar_attempt: ExternalCoordinatorRepeatedOptionSidecarResult
    live_action_projection: ExternalCoordinatorRepeatedOptionLiveActionProjectionResult
    candidate_metadata: ExternalCoordinatorRepeatedOptionBorrowedMetadata
    downstream_receipt: HCCLTwoLiveMemoryDownstreamAdoptionReceipt
    candidate_coordinator_matches_live: Bool[Array, ""]
    candidate_metadata_owner_bound: Bool[Array, ""]
    raw_stomp_result_consumed_once: Bool[Array, ""]
    finalization_digests_bound: Bool[Array, ""]
    final_live_stomp_digest: UInt[Array, " 8"]
    final_live_owner_digest_bound: Bool[Array, ""]
    preparation_valid: Bool[Array, ""]


@chex.dataclass(frozen=True)
class HCCLTwoLiveMemoryRepeatedOptionPrepareWork:
    """Exact donor and lifecycle work performed only during preparation."""

    hccl_stage_calls: Int[Array, ""]
    live_prepare_calls: Int[Array, " 2"]
    coordinator_update_calls: Int[Array, " 2"]
    prototype_update_calls: Int[Array, " 2"]
    raw_stomp_update_evaluations: Int[Array, " 2"]
    lifecycle_observation_evaluations: Int[Array, " 2"]
    additional_stomp_update_evaluations: Int[Array, " 2"]
    memory_query_calls: Int[Array, " 2"]
    memory_write_calls: Int[Array, " 2"]
    kondo_calls: Int[Array, " 2"]
    actor_backward_calls: Int[Array, " 2"]
    outer_metadata_attachment_evaluations: Int[Array, ""]


@chex.dataclass(frozen=True)
class HCCLTwoLiveMemoryRepeatedOptionPreparedTransaction:
    """Transient complete HCCL/live/metadata candidate."""

    source_state: HCCLTwoLiveMemoryRepeatedOptionPrepareAdoptState
    inner_prepared: HCCLTwoLiveMemoryPreparedTransaction
    agent_0: HCCLTwoLiveMemoryRepeatedOptionPreparedAgent
    agent_1: HCCLTwoLiveMemoryRepeatedOptionPreparedAgent
    candidate_state: HCCLTwoLiveMemoryRepeatedOptionPrepareAdoptState
    work: HCCLTwoLiveMemoryRepeatedOptionPrepareWork
    source_state_receipt_words: UInt[Array, " 8"]
    config_receipt_words: UInt[Array, " 8"]
    source_metadata_bindings_valid: Bool[Array, " 2"]
    candidate_metadata_bindings_valid: Bool[Array, " 2"]
    finalization_digests_bound: Bool[Array, " 2"]
    final_live_owner_digests_bound: Bool[Array, " 2"]
    raw_stomp_results_consumed_once: Bool[Array, " 2"]
    no_planner_rung_valid: Bool[Array, ""]
    candidate_state_valid: Bool[Array, ""]
    preparation_valid: Bool[Array, ""]
    delight_available: Bool[Array, ""]
    actor_backward_calls: Int[Array, ""]
    content_tag_words: UInt[Array, " 8"]


@chex.dataclass(frozen=True)
class HCCLTwoLiveMemoryRepeatedOptionPreparationReceipt:
    """Unkeyed source/config/content integrity receipt."""

    source_state_receipt_words: UInt[Array, " 8"]
    config_receipt_words: UInt[Array, " 8"]
    prepared_content_tag_words: UInt[Array, " 8"]
    integrity_bound: Bool[Array, ""]
    content_tag_words: UInt[Array, " 8"]


@chex.dataclass(frozen=True)
class HCCLTwoLiveMemoryRepeatedOptionAdoptionWork:
    """Adoption-only work; every donor reevaluation count is exact zero."""

    composite_integrity_checks: Int[Array, ""]
    metadata_owner_binding_checks: Int[Array, " 2"]
    inner_integrity_adoption_calls: Int[Array, ""]
    world_proposal_calls: Int[Array, ""]
    attribution_proposal_calls: Int[Array, ""]
    coordinator_update_calls: Int[Array, " 2"]
    prototype_update_calls: Int[Array, " 2"]
    stomp_update_evaluations: Int[Array, " 2"]
    lifecycle_observation_evaluations: Int[Array, " 2"]
    memory_query_calls: Int[Array, " 2"]
    memory_write_calls: Int[Array, " 2"]
    memory_donor_reevaluations: Int[Array, " 2"]
    kondo_calls: Int[Array, " 2"]
    actor_backward_calls: Int[Array, " 2"]
    outer_metadata_attachment_evaluations: Int[Array, ""]


@chex.dataclass(frozen=True)
class HCCLTwoLiveMemoryRepeatedOptionPrepareAdoptResult:
    """Outer-gated all-owner adoption with retained prepare attempts."""

    state: HCCLTwoLiveMemoryRepeatedOptionPrepareAdoptState
    prepared: HCCLTwoLiveMemoryRepeatedOptionPreparedTransaction
    preparation_receipt: HCCLTwoLiveMemoryRepeatedOptionPreparationReceipt
    inner_result: HCCLTwoLiveMemoryPrepareAdoptResult
    adoption_work: HCCLTwoLiveMemoryRepeatedOptionAdoptionWork
    source_state_receipt_valid: Bool[Array, ""]
    config_receipt_valid: Bool[Array, ""]
    preparation_receipt_valid: Bool[Array, ""]
    downstream_receipts_valid: Bool[Array, " 2"]
    source_metadata_bindings_valid: Bool[Array, " 2"]
    candidate_metadata_bindings_valid: Bool[Array, " 2"]
    final_live_owner_bindings_valid: Bool[Array, " 2"]
    downstream_candidates_valid: Bool[Array, " 2"]
    hccl_update_applied: Bool[Array, ""]
    live_adapter_updates_applied: Bool[Array, " 2"]
    coordinator_updates_applied: Bool[Array, " 2"]
    prototype_updates_applied: Bool[Array, " 2"]
    stomp_updates_applied: Bool[Array, " 2"]
    learned_memory_updates_applied: Bool[Array, " 2"]
    lifecycle_metadata_updates_applied: Bool[Array, " 2"]
    delight_available: Bool[Array, ""]
    additional_delight_evaluations: Int[Array, ""]
    additional_actor_backward_calls: Int[Array, ""]
    outer_veto: Bool[Array, ""]
    candidate_state_valid: Bool[Array, ""]
    update_applied: Bool[Array, ""]


@chex.dataclass(frozen=True)
class HCCLTwoLiveMemoryRepeatedOptionAtomicSwapPrepared:
    """Transient per-agent sidecar swap bound to a complete composite source."""

    source_state: HCCLTwoLiveMemoryRepeatedOptionPrepareAdoptState
    agent_index: Int[Array, ""]
    sidecar_prepared: ExternalCoordinatorRepeatedOptionAtomicSwapPrepared
    source_state_receipt_words: UInt[Array, " 8"]
    config_receipt_words: UInt[Array, " 8"]
    source_metadata_binding_valid: Bool[Array, ""]
    preparation_valid: Bool[Array, ""]
    work: HCCLTwoLiveMemoryRepeatedOptionAtomicSwapPrepareWork
    content_tag_words: UInt[Array, " 8"]


@chex.dataclass(frozen=True)
class HCCLTwoLiveMemoryRepeatedOptionAtomicSwapAuthorityReceipt:
    """Unkeyed outer binding for one exact per-agent sidecar authority."""

    agent_index: Int[Array, ""]
    sidecar_authority: ExternalCoordinatorRepeatedOptionAtomicSwapAuthorityReceipt
    source_state_receipt_words: UInt[Array, " 8"]
    config_receipt_words: UInt[Array, " 8"]
    prepared_content_tag_words: UInt[Array, " 8"]
    integrity_bound: Bool[Array, ""]
    caller_authenticated: Bool[Array, ""]
    content_tag_words: UInt[Array, " 8"]


@chex.dataclass(frozen=True)
class HCCLTwoLiveMemoryRepeatedOptionAtomicSwapPrepareWork:
    """Actual lower work performed by the public atomic prepare boundary."""

    selected_sidecar_overlays: Int[Array, ""]
    outer_metadata_attachment_evaluations: Int[Array, ""]
    sidecar_atomic_prepare_calls: Int[Array, ""]
    authorized_atomic_prepare_derivations: Int[Array, ""]
    retirement_filter_derivations: Int[Array, ""]
    scheduler_observations: Int[Array, ""]
    replacement_candidate_preparations: Int[Array, ""]
    candidate_installation_evaluations: Int[Array, ""]
    oak_option_slot_rebind_calls: Int[Array, ""]


@chex.dataclass(frozen=True)
class HCCLTwoLiveMemoryRepeatedOptionAtomicSwapWork:
    """Exact host-only swap work and forbidden donor counts."""

    selected_sidecar_overlays: Int[Array, ""]
    outer_metadata_attachment_evaluations: Int[Array, ""]
    sidecar_atomic_prepare_calls: Int[Array, ""]
    sidecar_atomic_adopt_calls: Int[Array, ""]
    authorized_atomic_prepare_rederivations: Int[Array, ""]
    retirement_filter_rederivations: Int[Array, ""]
    scheduler_observations: Int[Array, ""]
    replacement_candidate_preparations: Int[Array, ""]
    candidate_installation_evaluations: Int[Array, ""]
    oak_option_slot_rebind_calls: Int[Array, ""]
    world_or_attribution_calls: Int[Array, ""]
    coordinator_update_calls: Int[Array, ""]
    prototype_update_calls: Int[Array, ""]
    stomp_update_evaluations: Int[Array, ""]
    memory_query_or_write_calls: Int[Array, ""]
    memory_donor_reevaluations: Int[Array, ""]
    kondo_calls: Int[Array, ""]
    actor_backward_calls: Int[Array, ""]


@chex.dataclass(frozen=True)
class HCCLTwoLiveMemoryRepeatedOptionAtomicSwapResult:
    """All-or-none per-agent option swap over the same persistent owners."""

    state: HCCLTwoLiveMemoryRepeatedOptionPrepareAdoptState
    sidecar_attempt: ExternalCoordinatorRepeatedOptionAtomicSwapResult
    work: HCCLTwoLiveMemoryRepeatedOptionAtomicSwapWork
    source_state_receipt_valid: Bool[Array, ""]
    config_receipt_valid: Bool[Array, ""]
    prepared_integrity_valid: Bool[Array, ""]
    authority_binding_valid: Bool[Array, ""]
    source_metadata_binding_valid: Bool[Array, ""]
    selected_agent_preserved_memory: Bool[Array, ""]
    selected_agent_preserved_pending_binding: Bool[Array, ""]
    other_agent_preserved: Bool[Array, ""]
    hccl_world_attribution_preserved: Bool[Array, ""]
    primitive_action_masks_preserved: Bool[Array, ""]
    exact_final_owner_binding: Bool[Array, ""]
    cold_state_persisted: Bool[Array, ""]
    delight_available: Bool[Array, ""]
    additional_actor_backward_calls: Int[Array, ""]
    outer_veto: Bool[Array, ""]
    candidate_state_valid: Bool[Array, ""]
    transaction_applied: Bool[Array, ""]


@dataclasses.dataclass(frozen=True, slots=True)
class HCCLTwoLiveMemoryRepeatedOptionPrepareAdoptResourceBudget:
    """Measured persistent topology plus exact logical work bounds."""

    persistent_state_nbytes: int
    inner_hccl_two_live_memory_state_nbytes: int
    detached_sidecar_metadata_nbytes: int
    hccl_state_owners: int
    live_memory_adapter_state_owners: int
    external_coordinator_state_owners: int
    prototype_state_owners: int
    oak_state_owners: int
    stomp_state_owners: int
    detached_metadata_stomp_state_owners: int
    persistent_cold_states: int
    persisted_preparations: int
    planner_state_owners: int
    prepare_hccl_stage_calls: int
    prepare_live_adapter_calls: int
    prepare_lifecycle_observation_calls: int
    prepare_additional_stomp_evaluations: int
    prepare_outer_metadata_attachment_evaluations: int
    adopt_world_or_learner_reevaluations: int
    adopt_outer_metadata_attachment_evaluations: int
    atomic_swap_selected_sidecar_overlays: int
    atomic_prepare_outer_metadata_attachment_evaluations: int
    atomic_authorize_outer_metadata_attachment_evaluations: int
    atomic_adopt_outer_metadata_attachment_evaluations: int
    atomic_total_outer_metadata_attachment_evaluations: int
    atomic_prepare_retirement_filter_derivations: int
    atomic_adopt_retirement_filter_rederivations: int
    atomic_total_retirement_filter_derivations: int
    atomic_prepare_scheduler_observations: int
    atomic_adopt_scheduler_observations: int
    atomic_total_scheduler_observations: int
    atomic_prepare_replacement_candidate_preparations: int
    atomic_adopt_replacement_candidate_preparations: int
    atomic_total_replacement_candidate_preparations: int
    atomic_total_candidate_installation_evaluations: int
    atomic_total_oak_option_slot_rebind_calls: int
    atomic_swap_world_or_learner_reevaluations: int
    delight_available: bool
    additional_delight_evaluations: int
    additional_actor_backward_calls: int
    output_write_calls: int
    artifact_bytes_written: int

    def to_config(self) -> dict[str, int | bool]:
        return dataclasses.asdict(self)


class HCCLTwoLiveMemoryRepeatedOptionPrepareAdoptBridge:
    """Stateless no-planner composition over one reviewed bridge and two sidecars."""

    def __init__(
        self,
        inner: HCCLTwoLiveMemoryPrepareAdoptBridge,
        agent_0_sidecar: ExternalCoordinatorRepeatedOptionSidecar,
        agent_1_sidecar: ExternalCoordinatorRepeatedOptionSidecar,
    ) -> None:
        if type(inner) is not HCCLTwoLiveMemoryPrepareAdoptBridge:
            raise TypeError("inner must be the exact reviewed prepare/adopt bridge")
        if type(agent_0_sidecar) is not ExternalCoordinatorRepeatedOptionSidecar:
            raise TypeError("agent_0_sidecar must be exact")
        if type(agent_1_sidecar) is not ExternalCoordinatorRepeatedOptionSidecar:
            raise TypeError("agent_1_sidecar must be exact")
        if agent_0_sidecar is agent_1_sidecar:
            raise ValueError("the two agents require distinct sidecar instances")
        if agent_0_sidecar.coordinator is not inner.inner.agent_0.coordinator:
            raise ValueError("agent 0 sidecar must borrow the inner agent 0 coordinator")
        if agent_1_sidecar.coordinator is not inner.inner.agent_1.coordinator:
            raise ValueError("agent 1 sidecar must borrow the inner agent 1 coordinator")
        self._inner = inner
        self._sidecars = (agent_0_sidecar, agent_1_sidecar)

    @property
    def inner(self) -> HCCLTwoLiveMemoryPrepareAdoptBridge:
        return self._inner

    @property
    def agent_0_sidecar(self) -> ExternalCoordinatorRepeatedOptionSidecar:
        return self._sidecars[0]

    @property
    def agent_1_sidecar(self) -> ExternalCoordinatorRepeatedOptionSidecar:
        return self._sidecars[1]

    def to_config(self) -> dict[str, object]:
        return {
            "type": type(self).__name__,
            "state_schema": HCCL_TWO_LIVE_MEMORY_REPEATED_OPTION_PREPARE_ADOPT_STATE_SCHEMA,
            "prepared_schema": HCCL_TWO_LIVE_MEMORY_REPEATED_OPTION_PREPARED_SCHEMA,
            "preparation_receipt_schema": (
                HCCL_TWO_LIVE_MEMORY_REPEATED_OPTION_PREPARATION_RECEIPT_SCHEMA
            ),
            "atomic_prepared_schema": (
                HCCL_TWO_LIVE_MEMORY_REPEATED_OPTION_ATOMIC_PREPARED_SCHEMA
            ),
            "atomic_authority_schema": (
                HCCL_TWO_LIVE_MEMORY_REPEATED_OPTION_ATOMIC_AUTHORITY_SCHEMA
            ),
            "mechanism_status": (
                HCCL_TWO_LIVE_MEMORY_REPEATED_OPTION_PREPARE_ADOPT_STATUS
            ),
            "evidence_level": (
                HCCL_TWO_LIVE_MEMORY_REPEATED_OPTION_PREPARE_ADOPT_EVIDENCE_LEVEL
            ),
            "inner": self._inner.to_config(),
            "agent_0_sidecar": self._sidecars[0].to_config(),
            "agent_1_sidecar": self._sidecars[1].to_config(),
            "hccl_state_owners": 1,
            "live_memory_adapter_state_owners": 2,
            "external_coordinator_state_owners": 2,
            "prototype_state_owners": 2,
            "oak_state_owners": 2,
            "stomp_state_owners": 2,
            "detached_metadata_stomp_state_owners": 0,
            "persistent_cold_states": 0,
            "preparation_persisted": False,
            "planner_state_owners": 0,
            "planner_action_relation": "P=M-no-planner-rung",
            "prepare_outer_metadata_attachment_evaluations": 8,
            "adopt_outer_metadata_attachment_evaluations": 10,
            "atomic_selected_sidecar_overlays": 1,
            "atomic_outer_metadata_attachment_evaluations": {
                "prepare": 3,
                "authorize": 3,
                "adopt": 6,
                "total": 12,
            },
            "atomic_retirement_filter_derivations": {
                "prepare": 1,
                "adopt": 2,
                "total": 3,
            },
            "atomic_scheduler_observations": {
                "prepare": 1,
                "adopt": 3,
                "total": 4,
            },
            "atomic_replacement_candidate_preparations": {
                "prepare": 1,
                "adopt": 3,
                "total": 4,
            },
            "atomic_candidate_installation_evaluations": 1,
            "atomic_oak_option_slot_rebind_calls": 1,
            "atomic_world_or_learner_reevaluations": 0,
            "physical_dispatch_authority": False,
            "safety_authority": False,
            "evidence_authority": False,
            "promotion_authority": False,
            "kondo_calls": 0,
            "delight_available": False,
            "delight_interpretation": "unavailable-no-Kondo-actor-backward",
            "additional_delight_evaluations": 0,
            "additional_actor_backward_calls": 0,
            "host_eager_only": True,
            "output_writes_authorized": False,
            "artifact_authorized": False,
            "limitations": list(
                HCCL_TWO_LIVE_MEMORY_REPEATED_OPTION_PREPARE_ADOPT_LIMITATIONS
            ),
        }

    def _config_receipt_words(self) -> Array:
        return _tree_digest("composite-config-v0", self.to_config())

    @staticmethod
    def _children(
        state: HCCLTwoLiveMemoryBridgeState,
    ) -> tuple[
        ExternalLearnedStateLiveMemoryAdapterState,
        ExternalLearnedStateLiveMemoryAdapterState,
    ]:
        return state.agent_0_state, state.agent_1_state

    @staticmethod
    def _metadata(
        state: HCCLTwoLiveMemoryRepeatedOptionPrepareAdoptState,
    ) -> tuple[
        ExternalCoordinatorRepeatedOptionBorrowedMetadata,
        ExternalCoordinatorRepeatedOptionBorrowedMetadata,
    ]:
        return state.agent_0_metadata, state.agent_1_metadata

    def _attachments(
        self,
        state: HCCLTwoLiveMemoryRepeatedOptionPrepareAdoptState,
    ) -> tuple[
        ExternalCoordinatorRepeatedOptionBorrowedMetadataAttachResult,
        ExternalCoordinatorRepeatedOptionBorrowedMetadataAttachResult,
    ]:
        children = self._children(state.inner_state)
        metadata = self._metadata(state)
        return (
            self._sidecars[0].attach_borrowed_metadata(
                children[0].coordinator_state,
                metadata[0],
            ),
            self._sidecars[1].attach_borrowed_metadata(
                children[1].coordinator_state,
                metadata[1],
            ),
        )

    def _attachment(
        self,
        state: HCCLTwoLiveMemoryRepeatedOptionPrepareAdoptState,
        agent_index: int,
    ) -> ExternalCoordinatorRepeatedOptionBorrowedMetadataAttachResult:
        """Evaluate only the selected outer borrowed-metadata attachment."""

        index = self._agent_index(agent_index)
        child = self._children(state.inner_state)[index]
        metadata = self._metadata(state)[index]
        return self._sidecars[index].attach_borrowed_metadata(
            child.coordinator_state,
            metadata,
        )

    def _payload_arrays(
        self,
        state: HCCLTwoLiveMemoryRepeatedOptionPrepareAdoptState,
    ) -> tuple[Array, ...]:
        return tuple(
            cast(Array, leaf)
            for leaf in jax.tree.leaves(
                (
                    state.inner_state,
                    state.agent_0_metadata,
                    state.agent_1_metadata,
                    state.revision,
                )
            )
        )

    def _with_checksum(
        self,
        state: HCCLTwoLiveMemoryRepeatedOptionPrepareAdoptState,
    ) -> HCCLTwoLiveMemoryRepeatedOptionPrepareAdoptState:
        return cast(
            HCCLTwoLiveMemoryRepeatedOptionPrepareAdoptState,
            state.replace(binding_checksum=_checksum_arrays(self._payload_arrays(state))),
        )

    def _check_state_contract(
        self,
        state: HCCLTwoLiveMemoryRepeatedOptionPrepareAdoptState,
    ) -> None:
        if type(state) is not HCCLTwoLiveMemoryRepeatedOptionPrepareAdoptState:
            raise TypeError("state must be the exact composite state")
        self._inner.inner._require_state_contract(state.inner_state)
        for index, metadata in enumerate(self._metadata(state)):
            if type(metadata) is not ExternalCoordinatorRepeatedOptionBorrowedMetadata:
                raise TypeError(f"agent {index} metadata must be exact")
        _require_array(
            state.revision,
            shape=(),
            dtype=jnp.dtype(jnp.int32),
            label="state.revision",
        )
        _require_array(
            state.binding_checksum,
            shape=(2,),
            dtype=jnp.dtype(jnp.uint32),
            label="state.binding_checksum",
        )

    def state_valid(
        self,
        state: HCCLTwoLiveMemoryRepeatedOptionPrepareAdoptState,
    ) -> Bool[Array, ""]:
        self._check_state_contract(state)
        attached = self._attachments(state)
        return (
            self._inner.state_valid(state.inner_state)
            & attached[0].transaction_applied
            & attached[1].transaction_applied
            & (state.revision >= 0)
            & jnp.array_equal(
                state.binding_checksum,
                _checksum_arrays(self._payload_arrays(state)),
            )
        )

    def init(
        self,
        inner_state: HCCLTwoLiveMemoryBridgeState,
        agent_0_repeated_state: RepeatedOptionLifecycleState,
        agent_1_repeated_state: RepeatedOptionLifecycleState,
    ) -> HCCLTwoLiveMemoryRepeatedOptionPrepareAdoptState:
        """Exact-bind two repeated overlays to the two nested coordinator owners."""

        if not bool(jax.device_get(self._inner.state_valid(inner_state))):
            raise ValueError("inner_state must satisfy the reviewed bridge contract")
        children = self._children(inner_state)
        sidecar_0 = self._sidecars[0].init(
            children[0].coordinator_state,
            agent_0_repeated_state,
        )
        sidecar_1 = self._sidecars[1].init(
            children[1].coordinator_state,
            agent_1_repeated_state,
        )
        state = self._with_checksum(
            HCCLTwoLiveMemoryRepeatedOptionPrepareAdoptState(
                inner_state=inner_state,
                agent_0_metadata=self._sidecars[0].detach_borrowed_metadata(sidecar_0),
                agent_1_metadata=self._sidecars[1].detach_borrowed_metadata(sidecar_1),
                revision=jnp.asarray(0, dtype=jnp.int32),
                binding_checksum=jnp.zeros((2,), dtype=jnp.uint32),
            )
        )
        if not bool(jax.device_get(self.state_valid(state))):
            raise ValueError("initialized composite failed its exact owner contract")
        return state

    @staticmethod
    def _prepared_tag(
        prepared: HCCLTwoLiveMemoryRepeatedOptionPreparedTransaction,
    ) -> Array:
        bare = cast(
            HCCLTwoLiveMemoryRepeatedOptionPreparedTransaction,
            prepared.replace(
                content_tag_words=jnp.zeros((_DIGEST_WORDS,), dtype=jnp.uint32)
            ),
        )
        return _tree_digest(
            HCCL_TWO_LIVE_MEMORY_REPEATED_OPTION_PREPARED_SCHEMA,
            bare,
        )

    def _agent_prepared(
        self,
        index: int,
        source_metadata: ExternalCoordinatorRepeatedOptionBorrowedMetadata,
        source_attachment: ExternalCoordinatorRepeatedOptionBorrowedMetadataAttachResult,
        inner_prepared: HCCLTwoLiveMemoryPreparedTransaction,
        *,
        context: int | Array,
        idle_candidate_option: int | Array,
        idle_initiation_eligible: bool | Array,
        comparator_randomized: bool | Array,
        treatment_propensity: float | Array,
        downstream_candidate_valid: bool | Array,
    ) -> HCCLTwoLiveMemoryRepeatedOptionPreparedAgent:
        sidecar = self._sidecars[index]
        facts = inner_prepared.agent_0 if index == 0 else inner_prepared.agent_1
        coordinator = facts.live_prepared.coordinator_result
        if coordinator is None:
            raise ValueError(f"agent {index} preparation did not reach its coordinator")
        receipt = sidecar.coordinator.integrity_receipt(coordinator.evaluated)
        attempt = sidecar.adopt_evaluated_transition(
            source_attachment.state,
            coordinator.evaluated,
            receipt,
            context=context,
            idle_candidate_option=idle_candidate_option,
            idle_initiation_eligible=idle_initiation_eligible,
            comparator_randomized=comparator_randomized,
            treatment_propensity=treatment_propensity,
            downstream_candidate_valid=downstream_candidate_valid,
        )
        candidate_live = facts.live_prepared.candidate_state
        projection = sidecar.project_live_cached_action_replacement(
            attempt.state,
            facts.live_prepared.cached_action_replacement,
            candidate_live.coordinator_state,
        )
        candidate_metadata = projection.metadata
        matches = projection.coordinator_wrapper_delta_exact
        candidate_attachment = sidecar.attach_borrowed_metadata(
            candidate_live.coordinator_state,
            candidate_metadata,
        )
        raw_once = (
            (attempt.raw_stomp_update_evaluations == 1)
            & (attempt.additional_stomp_update_evaluations == 0)
            & (attempt.lifecycle_observation_evaluations == 1)
            & attempt.raw_stomp_result_consumed
        )
        final_bound = (
            attempt.raw_stomp_result_bound
            & attempt.raw_stomp_result_digest_bound
            & attempt.finalization_trace_bound
            & attempt.final_stomp_owner_digest_bound
        )
        final_live_digest_bound = (
            projection.stomp_owner_matches_final_live
            & jnp.array_equal(
                projection.replacement_stomp_digest,
                projection.final_live_stomp_digest,
            )
            & jnp.array_equal(
                candidate_attachment.stomp_owner_digest,
                projection.final_live_stomp_digest,
            )
        )
        candidate_valid = (
            attempt.transaction_applied
            & projection.transaction_applied
            & matches
            & candidate_attachment.transaction_applied
            & raw_once
            & final_bound
            & final_live_digest_bound
        )
        revision_words = jnp.stack(
            (
                jnp.asarray(0, dtype=jnp.uint32),
                candidate_metadata.revision.astype(jnp.uint32),
            )
        )
        downstream = self._inner.bind_downstream_adoption_receipt(
            inner_prepared,
            agent_index=index,
            downstream_revision_words=revision_words,
            downstream_content_digest_words=_tree_digest(
                "repeated-option-candidate-metadata",
                index,
                candidate_metadata,
                projection.final_live_stomp_digest,
            ),
            downstream_candidate_valid=candidate_valid,
        )
        return HCCLTwoLiveMemoryRepeatedOptionPreparedAgent(
            agent_index=jnp.asarray(index, dtype=jnp.int32),
            inner_facts=facts,
            source_metadata=source_metadata,
            sidecar_attempt=attempt,
            live_action_projection=projection,
            candidate_metadata=candidate_metadata,
            downstream_receipt=downstream,
            candidate_coordinator_matches_live=matches,
            candidate_metadata_owner_bound=(candidate_attachment.transaction_applied),
            raw_stomp_result_consumed_once=raw_once,
            finalization_digests_bound=final_bound,
            final_live_stomp_digest=projection.final_live_stomp_digest,
            final_live_owner_digest_bound=final_live_digest_bound,
            preparation_valid=candidate_valid,
        )

    def prepare_transaction(
        self,
        state: HCCLTwoLiveMemoryRepeatedOptionPrepareAdoptState,
        event: Any,
        binding: Any,
        agent_0_event_input: ExternalLearnedStateLiveMemoryEventInput,
        agent_1_event_input: ExternalLearnedStateLiveMemoryEventInput,
        *,
        next_decision_hard_action_masks: Array,
        agent_0_candidate_evidence: ExternalBuilderCandidateAuditEvidence | None = None,
        agent_1_candidate_evidence: ExternalBuilderCandidateAuditEvidence | None = None,
        agent_0_partner_policy_fusion_input: (
            PrototypePartnerPolicyFusionInput | None
        ) = None,
        agent_1_partner_policy_fusion_input: (
            PrototypePartnerPolicyFusionInput | None
        ) = None,
        agent_0_partner_policy_fusion_feedback: (
            PrototypePartnerPolicyFusionFeedback | None
        ) = None,
        agent_1_partner_policy_fusion_feedback: (
            PrototypePartnerPolicyFusionFeedback | None
        ) = None,
        agent_0_context: int | Array = 0,
        agent_1_context: int | Array = 0,
        agent_0_idle_candidate_option: int | Array = 0,
        agent_1_idle_candidate_option: int | Array = 0,
        agent_0_idle_initiation_eligible: bool | Array = False,
        agent_1_idle_initiation_eligible: bool | Array = False,
        agent_0_comparator_randomized: bool | Array = False,
        agent_1_comparator_randomized: bool | Array = False,
        agent_0_treatment_propensity: float | Array = 0.0,
        agent_1_treatment_propensity: float | Array = 0.0,
        agent_0_downstream_candidate_valid: bool | Array = True,
        agent_1_downstream_candidate_valid: bool | Array = True,
    ) -> HCCLTwoLiveMemoryRepeatedOptionPreparedTransaction:
        """Evaluate all donors once, then consume each raw STOMP result once."""

        self._check_state_contract(state)
        if _contains_tracer(
            (
                state,
                event,
                binding,
                agent_0_event_input,
                agent_1_event_input,
                next_decision_hard_action_masks,
            )
        ):
            raise TypeError("repeated-option HCCL composition is host/eager-only")
        attachments = self._attachments(state)
        source_bindings = jnp.stack(
            tuple(item.transaction_applied for item in attachments)
        ).astype(jnp.bool_)
        inner_prepared = self._inner.prepare_transaction(
            state.inner_state,
            event,
            binding,
            agent_0_event_input,
            agent_1_event_input,
            next_decision_hard_action_masks=next_decision_hard_action_masks,
            agent_0_candidate_evidence=agent_0_candidate_evidence,
            agent_1_candidate_evidence=agent_1_candidate_evidence,
            agent_0_partner_policy_fusion_input=agent_0_partner_policy_fusion_input,
            agent_1_partner_policy_fusion_input=agent_1_partner_policy_fusion_input,
            agent_0_partner_policy_fusion_feedback=(
                agent_0_partner_policy_fusion_feedback
            ),
            agent_1_partner_policy_fusion_feedback=(
                agent_1_partner_policy_fusion_feedback
            ),
            agent_0_extended_action_mask=attachments[0].state.extended_action_mask,
            agent_1_extended_action_mask=attachments[1].state.extended_action_mask,
        )
        metadata = self._metadata(state)
        agent_0 = self._agent_prepared(
            0,
            metadata[0],
            attachments[0],
            inner_prepared,
            context=agent_0_context,
            idle_candidate_option=agent_0_idle_candidate_option,
            idle_initiation_eligible=agent_0_idle_initiation_eligible,
            comparator_randomized=agent_0_comparator_randomized,
            treatment_propensity=agent_0_treatment_propensity,
            downstream_candidate_valid=agent_0_downstream_candidate_valid,
        )
        agent_1 = self._agent_prepared(
            1,
            metadata[1],
            attachments[1],
            inner_prepared,
            context=agent_1_context,
            idle_candidate_option=agent_1_idle_candidate_option,
            idle_initiation_eligible=agent_1_idle_initiation_eligible,
            comparator_randomized=agent_1_comparator_randomized,
            treatment_propensity=agent_1_treatment_propensity,
            downstream_candidate_valid=agent_1_downstream_candidate_valid,
        )
        candidate = self._with_checksum(
            HCCLTwoLiveMemoryRepeatedOptionPrepareAdoptState(
                inner_state=inner_prepared.candidate_state,
                agent_0_metadata=agent_0.candidate_metadata,
                agent_1_metadata=agent_1.candidate_metadata,
                revision=_saturating_increment(state.revision),
                binding_checksum=jnp.zeros((2,), dtype=jnp.uint32),
            )
        )
        candidate_bindings = jnp.stack(
            (
                agent_0.candidate_metadata_owner_bound,
                agent_1.candidate_metadata_owner_bound,
            )
        ).astype(jnp.bool_)
        finalization = jnp.stack(
            (
                agent_0.finalization_digests_bound,
                agent_1.finalization_digests_bound,
            )
        ).astype(jnp.bool_)
        raw_once = jnp.stack(
            (
                agent_0.raw_stomp_result_consumed_once,
                agent_1.raw_stomp_result_consumed_once,
            )
        ).astype(jnp.bool_)
        final_live_digests = jnp.stack(
            (
                agent_0.final_live_owner_digest_bound,
                agent_1.final_live_owner_digest_bound,
            )
        ).astype(jnp.bool_)
        candidate_valid = self.state_valid(candidate)
        valid = (
            self.state_valid(state)
            & jnp.all(source_bindings)
            & inner_prepared.preparation_valid
            & inner_prepared.no_planner_rung_valid
            & agent_0.preparation_valid
            & agent_1.preparation_valid
            & jnp.all(candidate_bindings)
            & jnp.all(finalization)
            & jnp.all(final_live_digests)
            & jnp.all(raw_once)
            & candidate_valid
        )
        inner_work = inner_prepared.work
        zero_pair = jnp.zeros((_N_AGENTS,), dtype=jnp.int32)
        work = HCCLTwoLiveMemoryRepeatedOptionPrepareWork(
            hccl_stage_calls=inner_work.hccl_stage_calls,
            live_prepare_calls=inner_work.live_prepare_calls,
            coordinator_update_calls=inner_work.coordinator_update_calls,
            prototype_update_calls=inner_work.prototype_update_calls,
            raw_stomp_update_evaluations=(
                inner_work.real_stomp_update_evaluations
            ),
            lifecycle_observation_evaluations=jnp.ones(
                (_N_AGENTS,), dtype=jnp.int32
            ),
            additional_stomp_update_evaluations=zero_pair,
            memory_query_calls=inner_work.memory_query_calls,
            memory_write_calls=inner_work.memory_write_calls,
            kondo_calls=zero_pair,
            actor_backward_calls=zero_pair,
            outer_metadata_attachment_evaluations=jnp.asarray(
                8, dtype=jnp.int32
            ),
        )
        bare = HCCLTwoLiveMemoryRepeatedOptionPreparedTransaction(
            source_state=state,
            inner_prepared=inner_prepared,
            agent_0=agent_0,
            agent_1=agent_1,
            candidate_state=candidate,
            work=work,
            source_state_receipt_words=_tree_digest("composite-source", state),
            config_receipt_words=self._config_receipt_words(),
            source_metadata_bindings_valid=source_bindings,
            candidate_metadata_bindings_valid=candidate_bindings,
            finalization_digests_bound=finalization,
            final_live_owner_digests_bound=final_live_digests,
            raw_stomp_results_consumed_once=raw_once,
            no_planner_rung_valid=inner_prepared.no_planner_rung_valid,
            candidate_state_valid=candidate_valid,
            preparation_valid=valid,
            delight_available=jnp.asarray(False, dtype=jnp.bool_),
            actor_backward_calls=jnp.asarray(0, dtype=jnp.int32),
            content_tag_words=jnp.zeros((_DIGEST_WORDS,), dtype=jnp.uint32),
        )
        return cast(
            HCCLTwoLiveMemoryRepeatedOptionPreparedTransaction,
            bare.replace(content_tag_words=self._prepared_tag(bare)),
        )

    def prepare_event(
        self,
        state: HCCLTwoLiveMemoryRepeatedOptionPrepareAdoptState,
    ) -> Any:
        self._check_state_contract(state)
        return self._inner.prepare_event(state.inner_state)

    def bind_live_memory_actions(
        self,
        state: HCCLTwoLiveMemoryRepeatedOptionPrepareAdoptState,
        event: Any,
    ) -> Any:
        self._check_state_contract(state)
        return self._inner.bind_live_memory_actions(state.inner_state, event)

    def _check_prepared_contract(
        self,
        prepared: HCCLTwoLiveMemoryRepeatedOptionPreparedTransaction,
    ) -> None:
        if type(prepared) is not HCCLTwoLiveMemoryRepeatedOptionPreparedTransaction:
            raise TypeError("prepared must be the exact repeated-option transaction")
        self._check_state_contract(prepared.source_state)
        self._check_state_contract(prepared.candidate_state)
        self._inner._require_prepared_contract(prepared.inner_prepared)
        for index, agent in enumerate((prepared.agent_0, prepared.agent_1)):
            if type(agent) is not HCCLTwoLiveMemoryRepeatedOptionPreparedAgent:
                raise TypeError(f"prepared agent {index} has the wrong exact type")
            if type(agent.sidecar_attempt) is not ExternalCoordinatorRepeatedOptionSidecarResult:
                raise TypeError(f"prepared agent {index} sidecar attempt differs")
            if type(agent.live_action_projection) is not (
                ExternalCoordinatorRepeatedOptionLiveActionProjectionResult
            ):
                raise TypeError(f"prepared agent {index} projection differs")
        _exact_digest(
            prepared.source_state_receipt_words,
            label="prepared.source_state_receipt_words",
        )
        _exact_digest(
            prepared.config_receipt_words,
            label="prepared.config_receipt_words",
        )
        _exact_digest(prepared.content_tag_words, label="prepared.content_tag_words")
        for name in (
            "source_metadata_bindings_valid",
            "candidate_metadata_bindings_valid",
            "finalization_digests_bound",
            "final_live_owner_digests_bound",
            "raw_stomp_results_consumed_once",
        ):
            _require_array(
                getattr(prepared, name),
                shape=(_N_AGENTS,),
                dtype=jnp.dtype(jnp.bool_),
                label=f"prepared.{name}",
            )

    @staticmethod
    def _receipt_tag(
        receipt: HCCLTwoLiveMemoryRepeatedOptionPreparationReceipt,
    ) -> Array:
        bare = cast(
            HCCLTwoLiveMemoryRepeatedOptionPreparationReceipt,
            receipt.replace(
                content_tag_words=jnp.zeros((_DIGEST_WORDS,), dtype=jnp.uint32)
            ),
        )
        return _tree_digest(
            HCCL_TWO_LIVE_MEMORY_REPEATED_OPTION_PREPARATION_RECEIPT_SCHEMA,
            bare,
        )

    def integrity_receipt(
        self,
        prepared: HCCLTwoLiveMemoryRepeatedOptionPreparedTransaction,
    ) -> HCCLTwoLiveMemoryRepeatedOptionPreparationReceipt:
        self._check_prepared_contract(prepared)
        bare = HCCLTwoLiveMemoryRepeatedOptionPreparationReceipt(
            source_state_receipt_words=prepared.source_state_receipt_words,
            config_receipt_words=prepared.config_receipt_words,
            prepared_content_tag_words=prepared.content_tag_words,
            integrity_bound=jnp.asarray(True, dtype=jnp.bool_),
            content_tag_words=jnp.zeros((_DIGEST_WORDS,), dtype=jnp.uint32),
        )
        return cast(
            HCCLTwoLiveMemoryRepeatedOptionPreparationReceipt,
            bare.replace(content_tag_words=self._receipt_tag(bare)),
        )

    def _check_receipt_contract(
        self,
        receipt: HCCLTwoLiveMemoryRepeatedOptionPreparationReceipt,
    ) -> None:
        if type(receipt) is not HCCLTwoLiveMemoryRepeatedOptionPreparationReceipt:
            raise TypeError("preparation_receipt has the wrong exact type")
        for name in (
            "source_state_receipt_words",
            "config_receipt_words",
            "prepared_content_tag_words",
            "content_tag_words",
        ):
            _exact_digest(getattr(receipt, name), label=f"preparation_receipt.{name}")
        _exact_bool(receipt.integrity_bound, label="preparation_receipt.integrity_bound")

    def _gate_inner_result(
        self,
        result: HCCLTwoLiveMemoryPrepareAdoptResult,
        source: HCCLTwoLiveMemoryBridgeState,
        applied: Array,
    ) -> HCCLTwoLiveMemoryPrepareAdoptResult:
        public_0 = self._inner._outer_live_result(
            result.agent_0_result,
            source.agent_0_state,
            applied,
        )
        public_1 = self._inner._outer_live_result(
            result.agent_1_result,
            source.agent_1_state,
            applied,
        )
        public_hccl = self._inner._outer_hccl_result(
            result.hccl_result,
            source.hccl_state,
            applied,
        )
        pair = jnp.full((_N_AGENTS,), applied, dtype=jnp.bool_)
        return cast(
            HCCLTwoLiveMemoryPrepareAdoptResult,
            result.replace(
                state=_tree_select(applied, result.state, source),
                hccl_result=public_hccl,
                agent_0_result=public_0,
                agent_1_result=public_1,
                hccl_update_applied=result.hccl_update_applied & applied,
                live_adapter_updates_applied=(
                    result.live_adapter_updates_applied & pair
                ),
                coordinator_updates_applied=(
                    result.coordinator_updates_applied & pair
                ),
                prototype_updates_applied=result.prototype_updates_applied & pair,
                stomp_updates_applied=result.stomp_updates_applied & pair,
                learned_memory_updates_applied=(
                    result.learned_memory_updates_applied & pair
                ),
                builder_learning_applied=result.builder_learning_applied & pair,
                next_decision_masks_installed=(
                    result.next_decision_masks_installed & applied
                ),
                update_applied=result.update_applied & applied,
            ),
        )

    def adopt_prepared_transaction(
        self,
        state: HCCLTwoLiveMemoryRepeatedOptionPrepareAdoptState,
        prepared: HCCLTwoLiveMemoryRepeatedOptionPreparedTransaction,
        preparation_receipt: HCCLTwoLiveMemoryRepeatedOptionPreparationReceipt,
        *,
        downstream_candidate_valid: bool | Array = True,
    ) -> HCCLTwoLiveMemoryRepeatedOptionPrepareAdoptResult:
        """Adopt HCCL, both live states, and both overlays all-or-none."""

        self._check_state_contract(state)
        self._check_prepared_contract(prepared)
        self._check_receipt_contract(preparation_receipt)
        if _contains_tracer((state, prepared, preparation_receipt)):
            raise TypeError("repeated-option HCCL adoption is host/eager-only")
        downstream = _exact_bool(
            downstream_candidate_valid,
            label="downstream_candidate_valid",
        )
        source_words = _tree_digest("composite-source", state)
        source_valid = (
            self.state_valid(state)
            & _tree_exact_equal(state, prepared.source_state)
            & jnp.array_equal(source_words, prepared.source_state_receipt_words)
            & jnp.array_equal(
                _tree_digest("composite-source", prepared.source_state),
                prepared.source_state_receipt_words,
            )
            & jnp.array_equal(
                preparation_receipt.source_state_receipt_words,
                prepared.source_state_receipt_words,
            )
        )
        config_valid = (
            jnp.array_equal(self._config_receipt_words(), prepared.config_receipt_words)
            & jnp.array_equal(
                preparation_receipt.config_receipt_words,
                prepared.config_receipt_words,
            )
        )
        expected_receipt = self.integrity_receipt(prepared)
        receipt_valid = (
            _tree_exact_equal(preparation_receipt, expected_receipt)
            & preparation_receipt.integrity_bound
            & jnp.array_equal(
                prepared.content_tag_words,
                self._prepared_tag(prepared),
            )
        )
        source_attachments = self._attachments(state)
        source_bindings = jnp.stack(
            tuple(item.transaction_applied for item in source_attachments)
        ).astype(jnp.bool_)
        candidate_attachments = self._attachments(prepared.candidate_state)
        candidate_bindings = jnp.stack(
            tuple(item.transaction_applied for item in candidate_attachments)
        ).astype(jnp.bool_)
        final_owner_bindings = jnp.stack(
            (
                prepared.agent_0.live_action_projection.transaction_applied
                & prepared.agent_0.final_live_owner_digest_bound
                & jnp.array_equal(
                    prepared.agent_0.final_live_stomp_digest,
                    prepared.agent_0.live_action_projection.final_live_stomp_digest,
                )
                & jnp.array_equal(
                    prepared.agent_0.final_live_stomp_digest,
                    candidate_attachments[0].stomp_owner_digest,
                )
                & candidate_bindings[0],
                prepared.agent_1.live_action_projection.transaction_applied
                & prepared.agent_1.final_live_owner_digest_bound
                & jnp.array_equal(
                    prepared.agent_1.final_live_stomp_digest,
                    prepared.agent_1.live_action_projection.final_live_stomp_digest,
                )
                & jnp.array_equal(
                    prepared.agent_1.final_live_stomp_digest,
                    candidate_attachments[1].stomp_owner_digest,
                )
                & candidate_bindings[1],
            )
        ).astype(jnp.bool_)
        downstream_receipt_validity: list[Array] = []
        for index, agent in enumerate((prepared.agent_0, prepared.agent_1)):
            revision_words = jnp.stack(
                (
                    jnp.asarray(0, dtype=jnp.uint32),
                    agent.candidate_metadata.revision.astype(jnp.uint32),
                )
            )
            expected = self._inner.bind_downstream_adoption_receipt(
                prepared.inner_prepared,
                agent_index=index,
                downstream_revision_words=revision_words,
                downstream_content_digest_words=_tree_digest(
                    "repeated-option-candidate-metadata",
                    index,
                    agent.candidate_metadata,
                    agent.final_live_stomp_digest,
                ),
                downstream_candidate_valid=agent.preparation_valid,
            )
            downstream_receipt_validity.append(
                _tree_exact_equal(agent.downstream_receipt, expected)
            )
        downstream_receipts_valid = jnp.stack(
            tuple(downstream_receipt_validity)
        ).astype(jnp.bool_)
        downstream_candidates = jnp.stack(
            (
                prepared.agent_0.downstream_receipt.downstream_candidate_valid,
                prepared.agent_1.downstream_receipt.downstream_candidate_valid,
            )
        ).astype(jnp.bool_)
        candidate_valid = self.state_valid(prepared.candidate_state)
        preliminary = (
            source_valid
            & config_valid
            & receipt_valid
            & jnp.all(source_bindings)
            & jnp.all(candidate_bindings)
            & jnp.all(final_owner_bindings)
            & jnp.all(downstream_receipts_valid)
            & jnp.all(downstream_candidates)
            & prepared.preparation_valid
            & prepared.no_planner_rung_valid
            & (~prepared.delight_available)
            & (prepared.actor_backward_calls == 0)
            & candidate_valid
            & downstream
        )

        gated_receipts: list[HCCLTwoLiveMemoryDownstreamAdoptionReceipt] = []
        for index, agent in enumerate((prepared.agent_0, prepared.agent_1)):
            receipt = agent.downstream_receipt
            gated_receipts.append(
                self._inner.bind_downstream_adoption_receipt(
                    prepared.inner_prepared,
                    agent_index=index,
                    downstream_revision_words=receipt.downstream_revision_words,
                    downstream_content_digest_words=(
                        receipt.downstream_content_digest_words
                    ),
                    downstream_candidate_valid=(
                        receipt.downstream_candidate_valid & preliminary
                    ),
                )
            )
        inner_result = self._inner.adopt_prepared_transaction(
            state.inner_state,
            prepared.inner_prepared,
            self._inner.integrity_receipt(prepared.inner_prepared),
            gated_receipts[0],
            gated_receipts[1],
        )
        adopted_candidate = self._with_checksum(
            HCCLTwoLiveMemoryRepeatedOptionPrepareAdoptState(
                inner_state=inner_result.state,
                agent_0_metadata=prepared.agent_0.candidate_metadata,
                agent_1_metadata=prepared.agent_1.candidate_metadata,
                revision=prepared.candidate_state.revision,
                binding_checksum=jnp.zeros((2,), dtype=jnp.uint32),
            )
        )
        candidate_matches = _tree_exact_equal(
            adopted_candidate,
            prepared.candidate_state,
        )
        final_candidate_valid = self.state_valid(adopted_candidate)
        applied = (
            preliminary
            & inner_result.update_applied
            & candidate_matches
            & final_candidate_valid
        )
        final_state = cast(
            HCCLTwoLiveMemoryRepeatedOptionPrepareAdoptState,
            _tree_select(applied, adopted_candidate, state),
        )
        public_inner = self._gate_inner_result(inner_result, state.inner_state, applied)
        pair = jnp.full((_N_AGENTS,), applied, dtype=jnp.bool_)
        zero_pair = jnp.zeros((_N_AGENTS,), dtype=jnp.int32)
        return HCCLTwoLiveMemoryRepeatedOptionPrepareAdoptResult(
            state=final_state,
            prepared=prepared,
            preparation_receipt=preparation_receipt,
            inner_result=public_inner,
            adoption_work=HCCLTwoLiveMemoryRepeatedOptionAdoptionWork(
                composite_integrity_checks=jnp.asarray(1, dtype=jnp.int32),
                metadata_owner_binding_checks=jnp.full(
                    (_N_AGENTS,), 5, dtype=jnp.int32
                ),
                inner_integrity_adoption_calls=jnp.asarray(1, dtype=jnp.int32),
                world_proposal_calls=jnp.asarray(0, dtype=jnp.int32),
                attribution_proposal_calls=jnp.asarray(0, dtype=jnp.int32),
                coordinator_update_calls=zero_pair,
                prototype_update_calls=zero_pair,
                stomp_update_evaluations=zero_pair,
                lifecycle_observation_evaluations=zero_pair,
                memory_query_calls=zero_pair,
                memory_write_calls=zero_pair,
                memory_donor_reevaluations=zero_pair,
                kondo_calls=zero_pair,
                actor_backward_calls=zero_pair,
                outer_metadata_attachment_evaluations=jnp.asarray(
                    10, dtype=jnp.int32
                ),
            ),
            source_state_receipt_valid=source_valid,
            config_receipt_valid=config_valid,
            preparation_receipt_valid=receipt_valid,
            downstream_receipts_valid=downstream_receipts_valid,
            source_metadata_bindings_valid=source_bindings,
            candidate_metadata_bindings_valid=candidate_bindings,
            final_live_owner_bindings_valid=final_owner_bindings,
            downstream_candidates_valid=downstream_candidates,
            hccl_update_applied=public_inner.hccl_update_applied,
            live_adapter_updates_applied=(
                public_inner.live_adapter_updates_applied & pair
            ),
            coordinator_updates_applied=(
                public_inner.coordinator_updates_applied & pair
            ),
            prototype_updates_applied=public_inner.prototype_updates_applied & pair,
            stomp_updates_applied=public_inner.stomp_updates_applied & pair,
            learned_memory_updates_applied=(
                public_inner.learned_memory_updates_applied & pair
            ),
            lifecycle_metadata_updates_applied=pair,
            delight_available=jnp.asarray(False, dtype=jnp.bool_),
            additional_delight_evaluations=jnp.asarray(0, dtype=jnp.int32),
            additional_actor_backward_calls=jnp.asarray(0, dtype=jnp.int32),
            outer_veto=~downstream,
            candidate_state_valid=final_candidate_valid,
            update_applied=applied,
        )

    @staticmethod
    def _atomic_prepared_tag(
        prepared: HCCLTwoLiveMemoryRepeatedOptionAtomicSwapPrepared,
    ) -> Array:
        bare = cast(
            HCCLTwoLiveMemoryRepeatedOptionAtomicSwapPrepared,
            prepared.replace(
                content_tag_words=jnp.zeros((_DIGEST_WORDS,), dtype=jnp.uint32)
            ),
        )
        return _tree_digest(
            HCCL_TWO_LIVE_MEMORY_REPEATED_OPTION_ATOMIC_PREPARED_SCHEMA,
            bare,
        )

    @staticmethod
    def _agent_index(value: int) -> int:
        if type(value) is not int or value not in {0, 1}:
            raise ValueError("agent_index must be the exact integer 0 or 1")
        return value

    def prepare_agent_atomic_swap(
        self,
        state: HCCLTwoLiveMemoryRepeatedOptionPrepareAdoptState,
        *,
        agent_index: int,
        cycle_key: Array,
        retirement_handoff: CumulantOptionRetirementHandoff,
        retirement_authority: OptionRetirementAuthorityReceipt,
        phase_one_key: Array,
        phase_two_key: Array,
        arm_inputs: CumulantOptionSchedulerArmInputs,
        observation: CumulantOptionSchedulerObservation,
        live_inputs: CumulantOptionLiveInputs,
    ) -> HCCLTwoLiveMemoryRepeatedOptionAtomicSwapPrepared:
        """Transiently attach one sidecar and prepare one atomic option swap."""

        self._check_state_contract(state)
        index = self._agent_index(agent_index)
        attachment = self._attachment(state, index)
        lower = self._sidecars[index].prepare_atomic_swap(
            attachment.state,
            cycle_key,
            retirement_handoff,
            retirement_authority,
            phase_one_key,
            phase_two_key,
            arm_inputs,
            observation,
            live_inputs,
        )
        valid = (
            self.state_valid(state)
            & attachment.transaction_applied
            & lower.preparation_valid
        )
        bare = HCCLTwoLiveMemoryRepeatedOptionAtomicSwapPrepared(
            source_state=state,
            agent_index=jnp.asarray(index, dtype=jnp.int32),
            sidecar_prepared=lower,
            source_state_receipt_words=_tree_digest("composite-source", state),
            config_receipt_words=self._config_receipt_words(),
            source_metadata_binding_valid=attachment.transaction_applied,
            preparation_valid=valid,
            work=HCCLTwoLiveMemoryRepeatedOptionAtomicSwapPrepareWork(
                selected_sidecar_overlays=jnp.asarray(1, dtype=jnp.int32),
                outer_metadata_attachment_evaluations=jnp.asarray(
                    3, dtype=jnp.int32
                ),
                sidecar_atomic_prepare_calls=jnp.asarray(1, dtype=jnp.int32),
                authorized_atomic_prepare_derivations=jnp.asarray(
                    1, dtype=jnp.int32
                ),
                retirement_filter_derivations=jnp.asarray(1, dtype=jnp.int32),
                scheduler_observations=jnp.asarray(1, dtype=jnp.int32),
                replacement_candidate_preparations=jnp.asarray(
                    1, dtype=jnp.int32
                ),
                candidate_installation_evaluations=jnp.asarray(
                    0, dtype=jnp.int32
                ),
                oak_option_slot_rebind_calls=jnp.asarray(0, dtype=jnp.int32),
            ),
            content_tag_words=jnp.zeros((_DIGEST_WORDS,), dtype=jnp.uint32),
        )
        return cast(
            HCCLTwoLiveMemoryRepeatedOptionAtomicSwapPrepared,
            bare.replace(content_tag_words=self._atomic_prepared_tag(bare)),
        )

    def _check_atomic_prepared_contract(
        self,
        prepared: HCCLTwoLiveMemoryRepeatedOptionAtomicSwapPrepared,
    ) -> None:
        if type(prepared) is not HCCLTwoLiveMemoryRepeatedOptionAtomicSwapPrepared:
            raise TypeError("atomic prepared has the wrong exact type")
        self._check_state_contract(prepared.source_state)
        _require_array(
            prepared.agent_index,
            shape=(),
            dtype=jnp.dtype(jnp.int32),
            label="atomic_prepared.agent_index",
        )
        _exact_digest(
            prepared.source_state_receipt_words,
            label="atomic_prepared.source_state_receipt_words",
        )
        _exact_digest(
            prepared.config_receipt_words,
            label="atomic_prepared.config_receipt_words",
        )
        _exact_digest(
            prepared.content_tag_words,
            label="atomic_prepared.content_tag_words",
        )

    @staticmethod
    def _atomic_authority_tag(
        receipt: HCCLTwoLiveMemoryRepeatedOptionAtomicSwapAuthorityReceipt,
    ) -> Array:
        bare = cast(
            HCCLTwoLiveMemoryRepeatedOptionAtomicSwapAuthorityReceipt,
            receipt.replace(
                content_tag_words=jnp.zeros((_DIGEST_WORDS,), dtype=jnp.uint32)
            ),
        )
        return _tree_digest(
            HCCL_TWO_LIVE_MEMORY_REPEATED_OPTION_ATOMIC_AUTHORITY_SCHEMA,
            bare,
        )

    def authorize_agent_atomic_swap(
        self,
        state: HCCLTwoLiveMemoryRepeatedOptionPrepareAdoptState,
        prepared: HCCLTwoLiveMemoryRepeatedOptionAtomicSwapPrepared,
        installation_authority: CumulantOptionInstallationAuthorityReceipt,
        cycle_key: Array,
        *,
        swap_authorized: bool | Array,
    ) -> HCCLTwoLiveMemoryRepeatedOptionAtomicSwapAuthorityReceipt:
        """Bind one caller declaration to the exact source/agent preparation."""

        self._check_state_contract(state)
        self._check_atomic_prepared_contract(prepared)
        index_host = int(jax.device_get(prepared.agent_index))
        index = self._agent_index(index_host)
        authorized = _exact_bool(swap_authorized, label="swap_authorized")
        attachment = self._attachment(state, index)
        wrapper_valid = (
            self.state_valid(state)
            & _tree_exact_equal(state, prepared.source_state)
            & attachment.transaction_applied
            & prepared.source_metadata_binding_valid
            & prepared.preparation_valid
            & jnp.array_equal(
                prepared.content_tag_words,
                self._atomic_prepared_tag(prepared),
            )
            & jnp.array_equal(
                prepared.source_state_receipt_words,
                _tree_digest("composite-source", state),
            )
            & jnp.array_equal(prepared.config_receipt_words, self._config_receipt_words())
        )
        lower = self._sidecars[index].authorize_atomic_swap(
            attachment.state,
            prepared.sidecar_prepared,
            installation_authority,
            cycle_key,
            swap_authorized=authorized & wrapper_valid,
        )
        bare = HCCLTwoLiveMemoryRepeatedOptionAtomicSwapAuthorityReceipt(
            agent_index=prepared.agent_index,
            sidecar_authority=lower,
            source_state_receipt_words=prepared.source_state_receipt_words,
            config_receipt_words=prepared.config_receipt_words,
            prepared_content_tag_words=prepared.content_tag_words,
            integrity_bound=wrapper_valid,
            caller_authenticated=jnp.asarray(False, dtype=jnp.bool_),
            content_tag_words=jnp.zeros((_DIGEST_WORDS,), dtype=jnp.uint32),
        )
        return cast(
            HCCLTwoLiveMemoryRepeatedOptionAtomicSwapAuthorityReceipt,
            bare.replace(content_tag_words=self._atomic_authority_tag(bare)),
        )

    def _check_atomic_authority_contract(
        self,
        receipt: HCCLTwoLiveMemoryRepeatedOptionAtomicSwapAuthorityReceipt,
    ) -> None:
        if type(receipt) is not (
            HCCLTwoLiveMemoryRepeatedOptionAtomicSwapAuthorityReceipt
        ):
            raise TypeError("atomic authority has the wrong exact type")
        _require_array(
            receipt.agent_index,
            shape=(),
            dtype=jnp.dtype(jnp.int32),
            label="atomic_authority.agent_index",
        )
        for name in (
            "source_state_receipt_words",
            "config_receipt_words",
            "prepared_content_tag_words",
            "content_tag_words",
        ):
            _exact_digest(getattr(receipt, name), label=f"atomic_authority.{name}")
        _exact_bool(receipt.integrity_bound, label="atomic_authority.integrity_bound")
        _exact_bool(
            receipt.caller_authenticated,
            label="atomic_authority.caller_authenticated",
        )

    def adopt_agent_atomic_swap(
        self,
        state: HCCLTwoLiveMemoryRepeatedOptionPrepareAdoptState,
        prepared: HCCLTwoLiveMemoryRepeatedOptionAtomicSwapPrepared,
        authority_receipt: HCCLTwoLiveMemoryRepeatedOptionAtomicSwapAuthorityReceipt,
        cycle_key: Array,
        *,
        downstream_candidate_valid: bool | Array = True,
    ) -> HCCLTwoLiveMemoryRepeatedOptionAtomicSwapResult:
        """Adopt one agent's exact sidecar swap or return the complete source."""

        self._check_state_contract(state)
        self._check_atomic_prepared_contract(prepared)
        self._check_atomic_authority_contract(authority_receipt)
        index = self._agent_index(int(jax.device_get(prepared.agent_index)))
        downstream = _exact_bool(
            downstream_candidate_valid,
            label="downstream_candidate_valid",
        )
        source_words = _tree_digest("composite-source", state)
        source_receipt_valid = (
            _tree_exact_equal(state, prepared.source_state)
            & jnp.array_equal(source_words, prepared.source_state_receipt_words)
            & jnp.array_equal(
                authority_receipt.source_state_receipt_words,
                prepared.source_state_receipt_words,
            )
        )
        config_valid = (
            jnp.array_equal(self._config_receipt_words(), prepared.config_receipt_words)
            & jnp.array_equal(
                authority_receipt.config_receipt_words,
                prepared.config_receipt_words,
            )
        )
        prepared_integrity = jnp.array_equal(
            prepared.content_tag_words,
            self._atomic_prepared_tag(prepared),
        )
        authority_valid = (
            (authority_receipt.agent_index == prepared.agent_index)
            & authority_receipt.integrity_bound
            & (~authority_receipt.caller_authenticated)
            & jnp.array_equal(
                authority_receipt.prepared_content_tag_words,
                prepared.content_tag_words,
            )
            & jnp.array_equal(
                authority_receipt.content_tag_words,
                self._atomic_authority_tag(authority_receipt),
            )
        )
        attachment = self._attachment(state, index)
        preliminary = (
            self.state_valid(state)
            & source_receipt_valid
            & config_valid
            & prepared_integrity
            & authority_valid
            & prepared.preparation_valid
            & attachment.transaction_applied
            & downstream
        )
        lower = self._sidecars[index].adopt_atomic_swap(
            attachment.state,
            prepared.sidecar_prepared,
            authority_receipt.sidecar_authority,
            cycle_key,
            downstream_candidate_valid=preliminary,
        )
        children = self._children(state.inner_state)
        selected_child = children[index]
        candidate_child = cast(
            ExternalLearnedStateLiveMemoryAdapterState,
            selected_child.replace(coordinator_state=lower.state.coordinator_state),
        )
        candidate_inner = cast(
            HCCLTwoLiveMemoryBridgeState,
            state.inner_state.replace(
                agent_0_state=(candidate_child if index == 0 else children[0]),
                agent_1_state=(candidate_child if index == 1 else children[1]),
            ),
        )
        metadata = list(self._metadata(state))
        metadata[index] = self._sidecars[index].detach_borrowed_metadata(lower.state)
        candidate = self._with_checksum(
            HCCLTwoLiveMemoryRepeatedOptionPrepareAdoptState(
                inner_state=candidate_inner,
                agent_0_metadata=metadata[0],
                agent_1_metadata=metadata[1],
                revision=_saturating_increment(state.revision),
                binding_checksum=jnp.zeros((2,), dtype=jnp.uint32),
            )
        )
        memory_preserved = _tree_exact_equal(
            candidate_child.learned_memory_state,
            selected_child.learned_memory_state,
        )
        pending_preserved = _tree_exact_equal(
            candidate_child.pending_binding,
            selected_child.pending_binding,
        )
        other_preserved = _tree_exact_equal(
            self._children(candidate_inner)[1 - index],
            children[1 - index],
        )
        hccl_preserved = _tree_exact_equal(
            candidate_inner.hccl_state,
            state.inner_state.hccl_state,
        )
        masks_preserved = jnp.array_equal(
            candidate_inner.current_hard_action_masks,
            state.inner_state.current_hard_action_masks,
        )
        final_attachment = self._attachment(candidate, index)
        exact_owner = final_attachment.transaction_applied
        candidate_valid = self.state_valid(candidate)
        applied = (
            preliminary
            & lower.transaction_applied
            & memory_preserved
            & pending_preserved
            & other_preserved
            & hccl_preserved
            & masks_preserved
            & exact_owner
            & (~lower.cold_state_persisted)
            & candidate_valid
        )
        final_state = cast(
            HCCLTwoLiveMemoryRepeatedOptionPrepareAdoptState,
            _tree_select(applied, candidate, state),
        )
        return HCCLTwoLiveMemoryRepeatedOptionAtomicSwapResult(
            state=final_state,
            sidecar_attempt=lower,
            work=HCCLTwoLiveMemoryRepeatedOptionAtomicSwapWork(
                selected_sidecar_overlays=jnp.asarray(1, dtype=jnp.int32),
                outer_metadata_attachment_evaluations=jnp.asarray(
                    6, dtype=jnp.int32
                ),
                sidecar_atomic_prepare_calls=jnp.asarray(1, dtype=jnp.int32),
                sidecar_atomic_adopt_calls=jnp.asarray(1, dtype=jnp.int32),
                authorized_atomic_prepare_rederivations=jnp.asarray(
                    2, dtype=jnp.int32
                ),
                retirement_filter_rederivations=jnp.asarray(2, dtype=jnp.int32),
                scheduler_observations=jnp.asarray(3, dtype=jnp.int32),
                replacement_candidate_preparations=jnp.asarray(
                    3, dtype=jnp.int32
                ),
                candidate_installation_evaluations=jnp.asarray(
                    1, dtype=jnp.int32
                ),
                oak_option_slot_rebind_calls=jnp.asarray(1, dtype=jnp.int32),
                world_or_attribution_calls=jnp.asarray(0, dtype=jnp.int32),
                coordinator_update_calls=jnp.asarray(0, dtype=jnp.int32),
                prototype_update_calls=jnp.asarray(0, dtype=jnp.int32),
                stomp_update_evaluations=jnp.asarray(0, dtype=jnp.int32),
                memory_query_or_write_calls=jnp.asarray(0, dtype=jnp.int32),
                memory_donor_reevaluations=jnp.asarray(0, dtype=jnp.int32),
                kondo_calls=jnp.asarray(0, dtype=jnp.int32),
                actor_backward_calls=jnp.asarray(0, dtype=jnp.int32),
            ),
            source_state_receipt_valid=source_receipt_valid,
            config_receipt_valid=config_valid,
            prepared_integrity_valid=prepared_integrity,
            authority_binding_valid=authority_valid,
            source_metadata_binding_valid=attachment.transaction_applied,
            selected_agent_preserved_memory=memory_preserved,
            selected_agent_preserved_pending_binding=pending_preserved,
            other_agent_preserved=other_preserved,
            hccl_world_attribution_preserved=hccl_preserved,
            primitive_action_masks_preserved=masks_preserved,
            exact_final_owner_binding=exact_owner,
            cold_state_persisted=lower.cold_state_persisted,
            delight_available=jnp.asarray(False, dtype=jnp.bool_),
            additional_actor_backward_calls=jnp.asarray(0, dtype=jnp.int32),
            outer_veto=~downstream,
            candidate_state_valid=candidate_valid,
            transaction_applied=applied,
        )

    def resource_budget(
        self,
        state: HCCLTwoLiveMemoryRepeatedOptionPrepareAdoptState,
    ) -> HCCLTwoLiveMemoryRepeatedOptionPrepareAdoptResourceBudget:
        """Measure the exact two-owner topology and fixed logical call bounds."""

        if not bool(jax.device_get(self.state_valid(state))):
            raise ValueError("resource measurement requires a valid composite state")
        total = _tree_nbytes(state)
        inner = _tree_nbytes(state.inner_state)
        return HCCLTwoLiveMemoryRepeatedOptionPrepareAdoptResourceBudget(
            persistent_state_nbytes=total,
            inner_hccl_two_live_memory_state_nbytes=inner,
            detached_sidecar_metadata_nbytes=total - inner,
            hccl_state_owners=1,
            live_memory_adapter_state_owners=2,
            external_coordinator_state_owners=2,
            prototype_state_owners=2,
            oak_state_owners=2,
            stomp_state_owners=2,
            detached_metadata_stomp_state_owners=0,
            persistent_cold_states=0,
            persisted_preparations=0,
            planner_state_owners=0,
            prepare_hccl_stage_calls=1,
            prepare_live_adapter_calls=2,
            prepare_lifecycle_observation_calls=2,
            prepare_additional_stomp_evaluations=0,
            prepare_outer_metadata_attachment_evaluations=8,
            adopt_world_or_learner_reevaluations=0,
            adopt_outer_metadata_attachment_evaluations=10,
            atomic_swap_selected_sidecar_overlays=1,
            atomic_prepare_outer_metadata_attachment_evaluations=3,
            atomic_authorize_outer_metadata_attachment_evaluations=3,
            atomic_adopt_outer_metadata_attachment_evaluations=6,
            atomic_total_outer_metadata_attachment_evaluations=12,
            atomic_prepare_retirement_filter_derivations=1,
            atomic_adopt_retirement_filter_rederivations=2,
            atomic_total_retirement_filter_derivations=3,
            atomic_prepare_scheduler_observations=1,
            atomic_adopt_scheduler_observations=3,
            atomic_total_scheduler_observations=4,
            atomic_prepare_replacement_candidate_preparations=1,
            atomic_adopt_replacement_candidate_preparations=3,
            atomic_total_replacement_candidate_preparations=4,
            atomic_total_candidate_installation_evaluations=1,
            atomic_total_oak_option_slot_rebind_calls=1,
            atomic_swap_world_or_learner_reevaluations=0,
            delight_available=False,
            additional_delight_evaluations=0,
            additional_actor_backward_calls=0,
            output_write_calls=0,
            artifact_bytes_written=0,
        )


__all__ = [
    "HCCL_TWO_LIVE_MEMORY_REPEATED_OPTION_ATOMIC_AUTHORITY_SCHEMA",
    "HCCL_TWO_LIVE_MEMORY_REPEATED_OPTION_ATOMIC_PREPARED_SCHEMA",
    "HCCL_TWO_LIVE_MEMORY_REPEATED_OPTION_PREPARATION_RECEIPT_SCHEMA",
    "HCCL_TWO_LIVE_MEMORY_REPEATED_OPTION_PREPARED_SCHEMA",
    "HCCL_TWO_LIVE_MEMORY_REPEATED_OPTION_PREPARE_ADOPT_EVIDENCE_LEVEL",
    "HCCL_TWO_LIVE_MEMORY_REPEATED_OPTION_PREPARE_ADOPT_LIMITATIONS",
    "HCCL_TWO_LIVE_MEMORY_REPEATED_OPTION_PREPARE_ADOPT_STATE_SCHEMA",
    "HCCL_TWO_LIVE_MEMORY_REPEATED_OPTION_PREPARE_ADOPT_STATUS",
    "HCCLTwoLiveMemoryRepeatedOptionAdoptionWork",
    "HCCLTwoLiveMemoryRepeatedOptionAtomicSwapAuthorityReceipt",
    "HCCLTwoLiveMemoryRepeatedOptionAtomicSwapPrepareWork",
    "HCCLTwoLiveMemoryRepeatedOptionAtomicSwapPrepared",
    "HCCLTwoLiveMemoryRepeatedOptionAtomicSwapResult",
    "HCCLTwoLiveMemoryRepeatedOptionAtomicSwapWork",
    "HCCLTwoLiveMemoryRepeatedOptionPreparationReceipt",
    "HCCLTwoLiveMemoryRepeatedOptionPrepareAdoptBridge",
    "HCCLTwoLiveMemoryRepeatedOptionPrepareAdoptResourceBudget",
    "HCCLTwoLiveMemoryRepeatedOptionPrepareAdoptResult",
    "HCCLTwoLiveMemoryRepeatedOptionPrepareAdoptState",
    "HCCLTwoLiveMemoryRepeatedOptionPreparedAgent",
    "HCCLTwoLiveMemoryRepeatedOptionPreparedTransaction",
    "HCCLTwoLiveMemoryRepeatedOptionPrepareWork",
]
