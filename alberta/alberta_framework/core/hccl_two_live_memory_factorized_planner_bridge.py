# mypy: disable-error-code="arg-type,attr-defined,call-arg,no-any-return,type-var"
"""Snapshot-free factorized P over one HCCL owner and two live M owners.

This L0 sibling leaves :mod:`hccl_two_live_memory_bridge` unchanged.  Its
persistent state contains that exact v1 state (one HCCL owner and two live
memory/Prototype M owners) plus the paired factorized behavior/world-model
state and cache.  It never persists the derived P Prototype snapshots.

For each event, the cached planner proposal is reconstructed through
``PrototypeAgent.replace_cached_primitive_action`` from the exact live M
Prototype.  The reconstruction is authenticated against both factorized
models, the cache, the current hard mask, and the M base action.  Prior memory
feedback is then settled once against M, the pending receipt is blanked, and a
valid transient P-dispatch live source is formed.  Each live adapter prepares
one real transition from that source, while the factorized planner completes
one paired update and prepares the next cache from the two post-memory M
candidates.  Only those M candidates and the next factorized cache/model state
enter the persistent candidate.

Preparation is transient.  Adoption performs content checks and child
integrity selection only; it never reevaluates the world or a learner.  P is a
modeled HCCL layer, not physical dispatch, safety, evidence, or promotion
authority.  Delight is unavailable because no Kondo actor backward executes.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array
from jaxtyping import Bool, Float, Int, UInt

from alberta_framework.core.external_learned_state_live_memory_adapter import (
    ExternalLearnedStateLiveMemoryAdapter,
    ExternalLearnedStateLiveMemoryAdapterState,
    ExternalLearnedStateLiveMemoryEventInput,
    ExternalLearnedStateLiveMemoryFeedback,
    ExternalLearnedStateLiveMemoryIntegrityReceipt,
    ExternalLearnedStateLiveMemoryPreparedTransition,
    ExternalLearnedStateLiveMemoryResult,
)
from alberta_framework.core.external_learned_state_router_audit_coordinator import (
    ExternalBuilderCandidateAuditEvidence,
    ExternalLearnedStateTransition,
)
from alberta_framework.core.hccl_causal_attribution import (
    HCCLActionLayer,
    HCCLActionReceipt,
)
from alberta_framework.core.hccl_two_live_memory_bridge import (
    _B0M1_SLOT,
    _BB_SLOT,
    _M0B1_SLOT,
    _PP_SLOT,
    HCCLTwoLiveMemoryBridge,
    HCCLTwoLiveMemoryBridgeConfig,
    HCCLTwoLiveMemoryBridgeState,
    _contains_tracer,
    _content_tag,
    _require_array,
    _tree_exact_equal,
    _tree_select,
)
from alberta_framework.core.hccl_world_attribution_adapter import (
    HCCLWorldAttributionAdapterResult,
)
from alberta_framework.core.learned_experiential_memory_controller import (
    LearnedExperientialMemoryFeedback,
    LearnedExperientialMemoryFeedbackResult,
)
from alberta_framework.core.options import STOMPUpdateResult
from alberta_framework.core.prototype_agent import (
    PrototypeAgentState,
    PrototypeCachedPrimitiveActionReplacement,
    PrototypePartnerPolicyFusionFeedback,
    PrototypePartnerPolicyFusionInput,
    PrototypeUpdateResult,
)
from alberta_framework.core.prototype_factorized_partner_planner import (
    FactorizedPartnerDecisionCache,
    PrototypeFactorizedPartnerPlanner,
    PrototypeFactorizedPartnerPlannerConfig,
    PrototypeFactorizedPartnerPlannerState,
    PrototypeFactorizedPartnerTransitionResult,
)
from alberta_framework.core.stomp_owner_finalization import (
    STOMPOwnerFinalizationTrace,
)
from alberta_framework.streams.hccl_causal_core import (
    HCCLCausalCoreEventReceipt,
    HCCLCausalCoreProposal,
)

HCCL_TWO_LIVE_MEMORY_FACTORIZED_PLANNER_SCHEMA = (
    "alberta.hccl-two-live-memory-factorized-planner.v1"
)
HCCL_TWO_LIVE_MEMORY_FACTORIZED_PLANNER_STATE_SCHEMA = (
    "alberta.hccl-two-live-memory-factorized-planner-state.v1"
)
HCCL_TWO_LIVE_MEMORY_FACTORIZED_PLANNER_BINDING_SCHEMA = (
    "alberta.hccl-two-live-memory-factorized-planner-binding.v1"
)
HCCL_TWO_LIVE_MEMORY_FACTORIZED_PLANNER_PREPARED_SCHEMA = (
    "alberta.hccl-two-live-memory-factorized-planner-prepared.v1"
)
HCCL_TWO_LIVE_MEMORY_FACTORIZED_PLANNER_PREPARATION_RECEIPT_SCHEMA = (
    "alberta.hccl-two-live-memory-factorized-planner-preparation-receipt.v1"
)
HCCL_TWO_LIVE_MEMORY_FACTORIZED_PLANNER_DOWNSTREAM_RECEIPT_SCHEMA = (
    "alberta.hccl-two-live-memory-factorized-planner-downstream-receipt.v1"
)
HCCL_TWO_LIVE_MEMORY_FACTORIZED_PLANNER_STATUS = (
    "l0-development-hccl-two-live-memory-factorized-planner"
)
HCCL_TWO_LIVE_MEMORY_FACTORIZED_PLANNER_EVIDENCE_LEVEL = "L0"
HCCL_TWO_LIVE_MEMORY_FACTORIZED_PLANNER_LIMITATIONS = (
    "persistent-state-owns-M-and-factorized-cache-models-not-P-snapshots",
    "P-is-reconstructed-transiently-from-M-cache-model-and-mask",
    "agent-0-memory-feedback-is-M0B1-minus-BB-only",
    "agent-1-memory-feedback-is-B0M1-minus-BB-only",
    "factorized-readiness-is-not-assessed",
    "planner-layer-has-no-external-physical-dispatch-or-safety-authority",
    "integrity-receipts-are-not-caller-authentication",
    "delight-unavailable-no-Kondo-actor-backward",
    "host-eager-only",
    "no-schedule-output-artifact-threshold-evidence-or-promotion-authority",
)

_N_AGENTS = 2
_N_ACTIONS = 2
_RAW_OBSERVATION_DIM = 16
_DIGEST_WORDS = 8


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _digest_bytes(payload: bytes) -> UInt[Array, " 8"]:
    digest = hashlib.sha256(payload).digest()
    return jnp.asarray(
        tuple(
            int.from_bytes(digest[offset : offset + 4], "big")
            for offset in range(0, len(digest), 4)
        ),
        dtype=jnp.uint32,
    )


def _tree_digest(*values: object) -> UInt[Array, " 8"]:
    """Hash exact host material at this deliberately host-only boundary."""

    digest = hashlib.sha256()
    for value in values:
        digest.update(type(value).__module__.encode("utf-8"))
        digest.update(type(value).__qualname__.encode("utf-8"))
        leaves, tree = jax.tree.flatten(value)
        digest.update(repr(tree).encode("utf-8"))
        digest.update(len(leaves).to_bytes(8, "big"))
        for leaf in leaves:
            if hasattr(leaf, "dtype") and hasattr(leaf, "shape"):
                array = jnp.asarray(leaf)
                material = (
                    jr.key_data(array)
                    if jax.dtypes.issubdtype(array.dtype, jax.dtypes.prng_key)
                    else array
                )
                host = np.asarray(jax.device_get(material))
                digest.update(str(host.dtype).encode("ascii"))
                digest.update(np.asarray(host.shape, dtype=np.int64).tobytes())
                digest.update(np.ascontiguousarray(host).tobytes())
            else:
                digest.update(type(leaf).__module__.encode("utf-8"))
                digest.update(type(leaf).__qualname__.encode("utf-8"))
                digest.update(repr(leaf).encode("utf-8"))
    return _digest_bytes(digest.digest())


def _exact_digest(value: Any, *, label: str) -> Array:
    return _require_array(
        value,
        shape=(_DIGEST_WORDS,),
        dtype=jnp.dtype(jnp.uint32),
        label=label,
    )


def _tree_nbytes(value: object) -> int:
    total = 0
    for leaf in jax.tree.leaves(value):
        if not hasattr(leaf, "dtype"):
            continue
        material = (
            jr.key_data(leaf)
            if jax.dtypes.issubdtype(leaf.dtype, jax.dtypes.prng_key)
            else leaf
        )
        total += int(material.size) * int(material.dtype.itemsize)
    return total


@dataclasses.dataclass(frozen=True, slots=True)
class HCCLTwoLiveMemoryFactorizedPlannerConfig:
    """Exact v1 owner configuration plus one paired factorized sidecar."""

    inner: HCCLTwoLiveMemoryBridgeConfig
    planner: PrototypeFactorizedPartnerPlannerConfig

    def __post_init__(self) -> None:
        if type(self.inner) is not HCCLTwoLiveMemoryBridgeConfig:
            raise TypeError("inner must be an exact HCCLTwoLiveMemoryBridgeConfig")
        if type(self.planner) is not PrototypeFactorizedPartnerPlannerConfig:
            raise TypeError("planner must be an exact factorized planner config")
        if not self.planner.planning_enabled:
            raise ValueError("the factorized P sibling requires planning_enabled=True")
        prototype_0 = self.inner.agent_0.coordinator.inner.prototype
        prototype_1 = self.inner.agent_1.coordinator.inner.prototype
        builder_0 = prototype_0.state_builder
        builder_1 = prototype_1.state_builder
        constructed_dim_0 = (
            prototype_0.oak.observation_dim
            if builder_0 is None
            else builder_0.observation_dim
        )
        constructed_dim_1 = (
            prototype_1.oak.observation_dim
            if builder_1 is None
            else builder_1.observation_dim
        )
        if constructed_dim_0 != constructed_dim_1:
            raise ValueError("the two Prototype constructed-state widths must match")
        if self.planner.observation_dim != constructed_dim_0:
            raise ValueError(
                "factorized planner observation width must equal the live "
                "Prototype builder-base/current_raw width"
            )
        if self.planner.n_actions != _N_ACTIONS:
            raise ValueError("factorized planner must expose exactly two actions")
        representation = self.inner.agent_0.coordinator.inner.prototype.oak.observation_dim
        if self.planner.prototype_representation_dim != representation:
            raise ValueError("factorized planner and Prototype representation widths differ")
        representation_1 = self.inner.agent_1.coordinator.inner.prototype.oak.observation_dim
        if representation_1 != representation:
            raise ValueError("the two Prototype representation widths must match")
        gamma_0 = self.inner.agent_0.coordinator.inner.ensemble.world_model.gamma
        gamma_1 = self.inner.agent_1.coordinator.inner.ensemble.world_model.gamma
        if gamma_0 != gamma_1:
            raise ValueError("paired factorized completion requires one shared discount")


@chex.dataclass(frozen=True)
class HCCLTwoLiveMemoryFactorizedPlannerState:
    """Only the two live M owners and factorized cache/models persist."""

    inner_state: HCCLTwoLiveMemoryBridgeState
    planner_state: PrototypeFactorizedPartnerPlannerState


@chex.dataclass(frozen=True)
class HCCLTwoLiveMemoryFactorizedPlannerDispatchView:
    """One reconstructed transient P view; never a persistent state field."""

    memory_prototype: PrototypeAgentState
    planner_prototype: PrototypeAgentState
    replacement: PrototypeCachedPrimitiveActionReplacement
    cache: FactorizedPartnerDecisionCache
    applied_partner_probabilities: Float[Array, " actions"]
    expected_rewards: Float[Array, " actions"]
    proposed_action: Int[Array, ""]
    effective_action: Int[Array, ""]
    cache_base_matches_memory: Bool[Array, ""]
    cache_effective_matches_replacement: Bool[Array, ""]
    mask_relation_valid: Bool[Array, ""]
    planner_cache_authenticated: Bool[Array, ""]
    valid: Bool[Array, ""]


@chex.dataclass(frozen=True)
class HCCLTwoLiveMemoryFactorizedPlannerActionBinding:
    """Exact B, M, and reconstructed P identities for one HCCL event."""

    source_world_words: UInt[Array, " 2"]
    source_world_tag_words: UInt[Array, " 4"]
    event_content_tag_words: UInt[Array, " 4"]
    feedback_binding_available: Bool[Array, " 2"]
    live_memory_transaction_words: UInt[Array, "2 2"]
    prototype_decision_words: UInt[Array, "2 4"]
    base_actions: Int[Array, " 2"]
    memory_actions_before_mask: Int[Array, " 2"]
    memory_actions: Int[Array, " 2"]
    planner_proposed_actions: Int[Array, " 2"]
    planner_actions: Int[Array, " 2"]
    current_hard_action_masks: Bool[Array, "2 2"]
    planner_config_token: UInt[Array, " 32"]
    behavior_step_words: UInt[Array, "2 2"]
    grounded_update_words: UInt[Array, "2 2"]
    base: HCCLActionReceipt
    memory: HCCLActionReceipt
    planner: HCCLActionReceipt
    content_tag_words: UInt[Array, " 4"]


@chex.dataclass(frozen=True)
class HCCLTwoLiveMemorySettledPlannerDispatchSource:
    """M feedback settled once and a valid transient P live source installed."""

    memory_source: ExternalLearnedStateLiveMemoryAdapterState
    dispatch_state: ExternalLearnedStateLiveMemoryAdapterState
    feedback: ExternalLearnedStateLiveMemoryFeedback
    settlement_result: LearnedExperientialMemoryFeedbackResult | None
    feedback_required: Bool[Array, ""]
    feedback_supplied: Bool[Array, ""]
    feedback_identity_valid: Bool[Array, ""]
    settlement_evaluations: Int[Array, ""]
    settlement_valid: Bool[Array, ""]
    replacement_committed: Bool[Array, ""]
    dispatch_source_valid: Bool[Array, ""]
    valid: Bool[Array, ""]


@chex.dataclass(frozen=True)
class HCCLTwoLiveMemoryFactorizedPlannerPreparedAgentFacts:
    """One transient P source and one live M-successor preparation."""

    agent_index: Int[Array, ""]
    dispatch: HCCLTwoLiveMemoryFactorizedPlannerDispatchView
    settled_dispatch: HCCLTwoLiveMemorySettledPlannerDispatchSource
    transition: ExternalLearnedStateTransition
    live_prepared: ExternalLearnedStateLiveMemoryPreparedTransition
    prototype_result: PrototypeUpdateResult
    raw_stomp_result: STOMPUpdateResult
    owner_finalization_trace: STOMPOwnerFinalizationTrace
    extended_action_mask: Bool[Array, " extended_actions"]
    candidate_live_state_receipt_words: UInt[Array, " 8"]


@chex.dataclass(frozen=True)
class HCCLTwoLiveMemoryFactorizedPlannerPrepareWork:
    hccl_stage_calls: Int[Array, ""]
    world_proposal_calls: Int[Array, ""]
    attribution_proposal_calls: Int[Array, ""]
    planner_reconstruction_replacements: Int[Array, " 2"]
    feedback_settlement_calls: Int[Array, " 2"]
    live_prepare_calls: Int[Array, " 2"]
    live_internal_feedback_settlement_calls: Int[Array, " 2"]
    coordinator_update_calls: Int[Array, " 2"]
    prototype_update_calls: Int[Array, " 2"]
    real_stomp_update_evaluations: Int[Array, " 2"]
    total_stomp_update_evaluations: Int[Array, " 2"]
    memory_query_calls: Int[Array, " 2"]
    memory_write_calls: Int[Array, " 2"]
    factorized_completed_transition_calls: Int[Array, ""]
    factorized_behavior_update_attempts: Int[Array, " 2"]
    factorized_grounded_update_attempts: Int[Array, " 2"]
    actor_backward_calls: Int[Array, ""]


@chex.dataclass(frozen=True)
class HCCLTwoLiveMemoryFactorizedPlannerPreparedTransaction:
    """Complete transient donor proposal; never checkpointed or persisted."""

    source_state: HCCLTwoLiveMemoryFactorizedPlannerState
    event: HCCLCausalCoreEventReceipt
    binding: HCCLTwoLiveMemoryFactorizedPlannerActionBinding
    agent_0_event_input: ExternalLearnedStateLiveMemoryEventInput
    agent_1_event_input: ExternalLearnedStateLiveMemoryEventInput
    next_decision_hard_action_masks: Bool[Array, "2 2"]
    attempted_hccl_result: HCCLWorldAttributionAdapterResult
    agent_0: HCCLTwoLiveMemoryFactorizedPlannerPreparedAgentFacts
    agent_1: HCCLTwoLiveMemoryFactorizedPlannerPreparedAgentFacts
    planner_result: PrototypeFactorizedPartnerTransitionResult
    candidate_state: HCCLTwoLiveMemoryFactorizedPlannerState
    agent_unilateral_counterfactual_delta: Float[Array, " 2"]
    effective_planner_actions: Int[Array, " 2"]
    physical_hccl_next_observations: Float[Array, "2 16"]
    grounded_next_constructed_states: Float[Array, "2 constructed"]
    prior_feedback_required: Bool[Array, " 2"]
    prior_feedback_supplied: Bool[Array, " 2"]
    work: HCCLTwoLiveMemoryFactorizedPlannerPrepareWork
    source_state_receipt_words: UInt[Array, " 8"]
    event_receipt_words: UInt[Array, " 8"]
    config_receipt_words: UInt[Array, " 8"]
    source_state_valid: Bool[Array, ""]
    event_receipt_valid: Bool[Array, ""]
    planner_reconstruction_valid: Bool[Array, ""]
    binding_integrity_valid: Bool[Array, ""]
    binding_matches_source: Bool[Array, ""]
    feedback_bindings_complete: Bool[Array, ""]
    feedback_bindings_match_children: Bool[Array, ""]
    current_event_masks_bound: Bool[Array, ""]
    pp_executes_planner_actions: Bool[Array, ""]
    planner_may_differ_from_memory: Bool[Array, ""]
    live_preparations_valid: Bool[Array, " 2"]
    factorized_completion_valid: Bool[Array, ""]
    candidate_state_valid: Bool[Array, ""]
    preparation_valid: Bool[Array, ""]
    delight_or_actor_backward: Bool[Array, ""]
    content_tag_words: UInt[Array, " 8"]


@chex.dataclass(frozen=True)
class HCCLTwoLiveMemoryFactorizedPlannerPreparationReceipt:
    source_state_receipt_words: UInt[Array, " 8"]
    event_receipt_words: UInt[Array, " 8"]
    config_receipt_words: UInt[Array, " 8"]
    prepared_content_tag_words: UInt[Array, " 8"]
    integrity_bound: Bool[Array, ""]
    content_tag_words: UInt[Array, " 8"]


@chex.dataclass(frozen=True)
class HCCLTwoLiveMemoryFactorizedPlannerDownstreamReceipt:
    agent_index: Int[Array, ""]
    source_state_receipt_words: UInt[Array, " 8"]
    event_receipt_words: UInt[Array, " 8"]
    config_receipt_words: UInt[Array, " 8"]
    prepared_content_tag_words: UInt[Array, " 8"]
    candidate_live_state_receipt_words: UInt[Array, " 8"]
    raw_stomp_digest: UInt[Array, " 8"]
    final_stomp_digest: UInt[Array, " 8"]
    owner_finalization_trace_checksum: UInt[Array, " 8"]
    extended_action_mask: Bool[Array, " extended_actions"]
    downstream_revision_words: UInt[Array, " 2"]
    downstream_content_digest_words: UInt[Array, " 8"]
    downstream_candidate_valid: Bool[Array, ""]
    content_tag_words: UInt[Array, " 8"]


@chex.dataclass(frozen=True)
class HCCLTwoLiveMemoryFactorizedPlannerAdoptionWork:
    preparation_integrity_checks: Int[Array, ""]
    downstream_receipt_integrity_checks: Int[Array, ""]
    live_integrity_calls: Int[Array, " 2"]
    world_proposal_calls: Int[Array, ""]
    attribution_proposal_calls: Int[Array, ""]
    coordinator_update_calls: Int[Array, " 2"]
    prototype_update_calls: Int[Array, " 2"]
    stomp_update_evaluations: Int[Array, " 2"]
    memory_query_calls: Int[Array, " 2"]
    memory_write_calls: Int[Array, " 2"]
    factorized_model_update_calls: Int[Array, ""]
    planner_reconstruction_replacements: Int[Array, " 2"]
    planner_cache_authentication_evaluations: Int[Array, " 2"]


@chex.dataclass(frozen=True)
class HCCLTwoLiveMemoryFactorizedPlannerResult:
    state: HCCLTwoLiveMemoryFactorizedPlannerState
    prepared: HCCLTwoLiveMemoryFactorizedPlannerPreparedTransaction
    preparation_receipt: HCCLTwoLiveMemoryFactorizedPlannerPreparationReceipt
    agent_0_downstream_receipt: HCCLTwoLiveMemoryFactorizedPlannerDownstreamReceipt
    agent_1_downstream_receipt: HCCLTwoLiveMemoryFactorizedPlannerDownstreamReceipt
    hccl_result: HCCLWorldAttributionAdapterResult
    agent_0_result: ExternalLearnedStateLiveMemoryResult
    agent_1_result: ExternalLearnedStateLiveMemoryResult
    planner_result: PrototypeFactorizedPartnerTransitionResult
    adoption_work: HCCLTwoLiveMemoryFactorizedPlannerAdoptionWork
    source_state_receipt_valid: Bool[Array, ""]
    event_receipt_valid: Bool[Array, ""]
    config_receipt_valid: Bool[Array, ""]
    preparation_receipt_valid: Bool[Array, ""]
    downstream_receipts_valid: Bool[Array, " 2"]
    downstream_candidates_valid: Bool[Array, " 2"]
    candidate_state_valid: Bool[Array, ""]
    source_state_authenticated: Bool[Array, ""]
    final_candidate_authenticated: Bool[Array, ""]
    hccl_update_applied: Bool[Array, ""]
    live_adapter_updates_applied: Bool[Array, " 2"]
    factorized_planner_update_applied: Bool[Array, ""]
    next_decision_masks_installed: Bool[Array, ""]
    modeled_planner_proposal_available: Bool[Array, ""]
    bounded_hccl_world_planner_action_consumed: Bool[Array, ""]
    pp_candidate_committed: Bool[Array, ""]
    external_environment_dispatch_authority: Bool[Array, ""]
    physical_dispatch_authority: Bool[Array, ""]
    safety_authority: Bool[Array, ""]
    evidence_authority: Bool[Array, ""]
    promotion_authority: Bool[Array, ""]
    actor_backward_calls: Int[Array, ""]
    delight_or_actor_backward: Bool[Array, ""]
    update_applied: Bool[Array, ""]


@dataclasses.dataclass(frozen=True, slots=True)
class HCCLTwoLiveMemoryFactorizedPlannerResourceBudget:
    hccl_state_owners: int
    live_memory_adapter_state_owners: int
    prototype_state_owners: int
    additional_prototype_state_owners: int
    factorized_planner_state_owners: int
    persisted_planner_prototype_snapshots: int
    inner_persistent_state_nbytes: int
    factorized_planner_state_nbytes: int
    total_persistent_state_nbytes: int
    persisted_preparation_records: int
    persisted_preparation_bytes: int
    prepare_hccl_stage_calls_per_transaction: int
    prepare_live_calls_per_transaction: int
    maximum_feedback_settlements_per_transaction: int
    planner_reconstruction_replacements_per_transaction: int
    factorized_completed_transition_calls_per_transaction: int
    adopt_planner_reconstruction_replacements_per_transaction: int
    adopt_planner_cache_authentication_evaluations_per_transaction: int
    adopt_world_or_learner_reevaluations: int
    output_write_calls: int
    artifact_bytes_written: int

    def to_config(self) -> dict[str, int]:
        return dataclasses.asdict(self)


class HCCLTwoLiveMemoryFactorizedPlannerBridge:
    """Host-only prepare/adopt owner for distinct persistent B/M and transient P."""

    def __init__(self, config: HCCLTwoLiveMemoryFactorizedPlannerConfig):
        if type(config) is not HCCLTwoLiveMemoryFactorizedPlannerConfig:
            raise TypeError("config must be an exact factorized HCCL config")
        self._config = config
        self._inner = HCCLTwoLiveMemoryBridge(config.inner)
        prototype_0 = self._inner.agent_0.coordinator.inner.prototype
        prototype_1 = self._inner.agent_1.coordinator.inner.prototype
        if _canonical_json_bytes(prototype_0.to_config()) != _canonical_json_bytes(
            prototype_1.to_config()
        ):
            raise ValueError("both live adapters must use the same Prototype config")
        self._planner = PrototypeFactorizedPartnerPlanner(prototype_0, config.planner)
        self._owner = jnp.asarray(
            config.inner.binding_owner_digest,
            dtype=jnp.uint32,
        )
        self._gamma = config.inner.agent_0.coordinator.inner.ensemble.world_model.gamma

    @property
    def config(self) -> HCCLTwoLiveMemoryFactorizedPlannerConfig:
        return self._config

    @property
    def inner(self) -> HCCLTwoLiveMemoryBridge:
        return self._inner

    @property
    def planner(self) -> PrototypeFactorizedPartnerPlanner:
        return self._planner

    def to_config(self) -> dict[str, object]:
        return {
            "type": type(self).__name__,
            "schema": HCCL_TWO_LIVE_MEMORY_FACTORIZED_PLANNER_SCHEMA,
            "state_schema": HCCL_TWO_LIVE_MEMORY_FACTORIZED_PLANNER_STATE_SCHEMA,
            "binding_schema": HCCL_TWO_LIVE_MEMORY_FACTORIZED_PLANNER_BINDING_SCHEMA,
            "prepared_schema": HCCL_TWO_LIVE_MEMORY_FACTORIZED_PLANNER_PREPARED_SCHEMA,
            "preparation_receipt_schema": (
                HCCL_TWO_LIVE_MEMORY_FACTORIZED_PLANNER_PREPARATION_RECEIPT_SCHEMA
            ),
            "downstream_receipt_schema": (
                HCCL_TWO_LIVE_MEMORY_FACTORIZED_PLANNER_DOWNSTREAM_RECEIPT_SCHEMA
            ),
            "mechanism_status": HCCL_TWO_LIVE_MEMORY_FACTORIZED_PLANNER_STATUS,
            "evidence_level": HCCL_TWO_LIVE_MEMORY_FACTORIZED_PLANNER_EVIDENCE_LEVEL,
            "inner": self._inner.to_config(),
            "planner": self._config.planner.to_config(),
            "hccl_state_owners": 1,
            "live_memory_adapter_state_owners": 2,
            "external_coordinator_state_owners": 2,
            "learned_memory_controller_state_owners": 2,
            "prototype_state_owners": 2,
            "additional_prototype_state_owners": 0,
            "factorized_planner_state_owners": 1,
            "persisted_planner_prototype_snapshots": 0,
            "planner_action_relation": "P-is-transient-and-may-differ-from-M",
            "post_memory_transition_binding_owned_here": True,
            "grounded_state_semantics": (
                "external-GRU-builder-base-constructed-state"
            ),
            "physical_raw_grounding": False,
            "physical_hccl_plant_observation_dim": _RAW_OBSERVATION_DIM,
            "grounded_constructed_state_dim": self._config.planner.observation_dim,
            "physical_plant_observation_bound_to_live_transition": True,
            "physical_plant_observation_bound_to_pp_world_commit": True,
            "factorized_planner_readiness_assessed": False,
            "preparation_persisted": False,
            "preparation_checkpoint_supported": False,
            "modeled_planner_proposal_available": True,
            "bounded_hccl_world_planner_action_consumed": True,
            "pp_candidate_is_committed_world_successor": True,
            "external_environment_dispatch_authority": False,
            "physical_dispatch_authority": False,
            "safety_authority": False,
            "evidence_authority": False,
            "promotion_authority": False,
            "delight_or_actor_backward": False,
            "actor_backward_calls_per_transaction": 0,
            "delight_interpretation": "unavailable-no-Kondo-actor-backward",
            "caller_identity_authenticated": False,
            "host_eager_only": True,
            "schedule_execution_authorized": False,
            "output_writes_authorized": False,
            "artifact_authorized": False,
            "threshold_authorized": False,
            "limitations": list(HCCL_TWO_LIVE_MEMORY_FACTORIZED_PLANNER_LIMITATIONS),
        }

    def _config_receipt_words(self) -> Array:
        return _digest_bytes(_canonical_json_bytes(self.to_config()))

    @staticmethod
    def _prototype_state(
        state: ExternalLearnedStateLiveMemoryAdapterState,
    ) -> PrototypeAgentState:
        return state.coordinator_state.inner_state.prototype_state

    def _child_states(
        self,
        state: HCCLTwoLiveMemoryFactorizedPlannerState,
    ) -> tuple[
        ExternalLearnedStateLiveMemoryAdapterState,
        ExternalLearnedStateLiveMemoryAdapterState,
    ]:
        return state.inner_state.agent_0_state, state.inner_state.agent_1_state

    def _require_state_contract(
        self,
        state: HCCLTwoLiveMemoryFactorizedPlannerState,
    ) -> None:
        if type(state) is not HCCLTwoLiveMemoryFactorizedPlannerState:
            raise TypeError("state must be an exact factorized HCCL state")
        self._inner._require_state_contract(state.inner_state)
        if type(state.planner_state) is not PrototypeFactorizedPartnerPlannerState:
            raise TypeError("planner_state must be an exact factorized planner state")

    def _structure_valid(
        self,
        state: HCCLTwoLiveMemoryFactorizedPlannerState,
    ) -> Bool[Array, ""]:
        """Cheap owner/cache links; model projections are checked in reconstruction."""

        self._require_state_contract(state)
        valid = self._inner.state_valid(state.inner_state) & jnp.array_equal(
            state.planner_state.config_token,
            cast(Any, self._planner)._config_token,
        )
        planner_agents = (state.planner_state.agent_0, state.planner_state.agent_1)
        for index, (child, planner_agent) in enumerate(
            zip(self._child_states(state), planner_agents, strict=True)
        ):
            prototype = self._prototype_state(child)
            cache = planner_agent.cache
            mask = state.inner_state.current_hard_action_masks[index]
            effective = jnp.clip(cache.effective_action, 0, _N_ACTIONS - 1)
            valid = (
                valid
                & (cache.base_action == prototype.current_action)
                & (cache.base_action_guard == jnp.bitwise_not(cache.base_action))
                & jnp.array_equal(cache.prototype_decision_id, prototype.current_decision_id)
                & jnp.array_equal(
                    cache.prototype_representation,
                    prototype.current_representation,
                )
                & jnp.array_equal(cache.world_input, prototype.current_raw_observation)
                & jnp.array_equal(cache.behavior_step_words, planner_agent.behavior.step_words)
                & jnp.array_equal(cache.grounded_update_words, planner_agent.grounded.update_words)
                & jnp.array_equal(prototype.step_words, planner_agent.behavior.step_words)
                & jnp.array_equal(
                    planner_agent.behavior.step_words,
                    planner_agent.grounded.update_words,
                )
                & (cache.effective_action >= 0)
                & (cache.effective_action < _N_ACTIONS)
                & mask[effective]
                & cache.belief_valid
                & cache.replacement_candidate_committed
                & cache.planner_consumed
            )
        return valid

    def _reconstruct_dispatch_impl(
        self,
        state: HCCLTwoLiveMemoryFactorizedPlannerState,
    ) -> tuple[
        HCCLTwoLiveMemoryFactorizedPlannerDispatchView,
        HCCLTwoLiveMemoryFactorizedPlannerDispatchView,
    ]:
        children = self._child_states(state)
        planner_agents = (state.planner_state.agent_0, state.planner_state.agent_1)
        prototypes = tuple(self._prototype_state(child) for child in children)
        masks = state.inner_state.current_hard_action_masks
        views: list[HCCLTwoLiveMemoryFactorizedPlannerDispatchView] = []
        for index, (prototype, planner_agent) in enumerate(
            zip(prototypes, planner_agents, strict=True)
        ):
            cache = planner_agent.cache
            applied = (
                jnp.full(
                    (_N_ACTIONS,),
                    1.0 / _N_ACTIONS,
                    dtype=jnp.float32,
                )
                if self._config.planner.uniform_partner_belief
                else cache.learned_partner_probabilities
            )
            reward_cells = cache.world_raw_predictions[
                :, :, self._config.planner.observation_dim
            ]
            expected = reward_cells @ applied
            proposed = jnp.argmax(expected).astype(jnp.int32)
            replacement = (
                self._inner.agent_0.coordinator.inner.prototype.replace_cached_primitive_action(
                    prototype,
                    decision_id=prototype.current_decision_id,
                    decision_observation=prototype.current_representation,
                    proposed_action=proposed,
                    safety_action_mask=masks[index],
                )
            )
            proposed_safe = masks[index, jnp.clip(proposed, 0, _N_ACTIONS - 1)]
            relation = (
                jnp.any(masks[index])
                & masks[index, jnp.clip(prototype.current_action, 0, _N_ACTIONS - 1)]
                & jnp.where(
                    proposed_safe,
                    replacement.action == proposed,
                    (replacement.action == prototype.current_action)
                    & replacement.dispatch_replacement.used_safe_base_fallback,
                )
                & (
                    replacement.dispatch_replacement.proposed_action_safe
                    == proposed_safe
                )
            )
            base_matches = cache.base_action == prototype.current_action
            effective_matches = (
                cache.effective_action == replacement.action
            ) & (replacement.state.current_action == replacement.action)
            views.append(
                HCCLTwoLiveMemoryFactorizedPlannerDispatchView(
                    memory_prototype=prototype,
                    planner_prototype=replacement.state,
                    replacement=replacement,
                    cache=cache,
                    applied_partner_probabilities=applied,
                    expected_rewards=expected,
                    proposed_action=proposed,
                    effective_action=replacement.action,
                    cache_base_matches_memory=base_matches,
                    cache_effective_matches_replacement=effective_matches,
                    mask_relation_valid=relation,
                    planner_cache_authenticated=jnp.asarray(False, dtype=jnp.bool_),
                    valid=jnp.asarray(False, dtype=jnp.bool_),
                )
            )
        authenticated = self._planner.authenticate_pair(
            state.planner_state,
            views[0].planner_prototype,
            views[1].planner_prototype,
        )
        final: list[HCCLTwoLiveMemoryFactorizedPlannerDispatchView] = []
        structure = self._structure_valid(state)
        for index, view in enumerate(views):
            valid = (
                structure
                & view.replacement.committed
                & view.cache_base_matches_memory
                & view.cache_effective_matches_replacement
                & view.mask_relation_valid
                & authenticated[index]
                & jnp.all(jnp.isfinite(view.expected_rewards))
            )
            final.append(
                cast(
                    HCCLTwoLiveMemoryFactorizedPlannerDispatchView,
                    view.replace(
                        planner_cache_authenticated=authenticated[index],
                        valid=valid,
                    ),
                )
            )
        return final[0], final[1]

    def reconstruct_planner_dispatch(
        self,
        state: HCCLTwoLiveMemoryFactorizedPlannerState,
    ) -> tuple[
        HCCLTwoLiveMemoryFactorizedPlannerDispatchView,
        HCCLTwoLiveMemoryFactorizedPlannerDispatchView,
    ]:
        """Reconstruct exact transient P views and persist neither snapshot."""

        self._require_state_contract(state)
        if _contains_tracer(state):
            raise TypeError("factorized HCCL reconstruction is host/eager-only")
        return self._reconstruct_dispatch_impl(state)

    def state_valid(
        self,
        state: HCCLTwoLiveMemoryFactorizedPlannerState,
    ) -> Bool[Array, ""]:
        views = self.reconstruct_planner_dispatch(state)
        return self._structure_valid(state) & views[0].valid & views[1].valid

    def init(
        self,
        key: Array,
        *,
        initial_hard_action_masks: Array | None = None,
    ) -> HCCLTwoLiveMemoryFactorizedPlannerState:
        if not (
            hasattr(key, "shape")
            and hasattr(key, "dtype")
            and key.shape == ()
            and jax.dtypes.issubdtype(key.dtype, jax.dtypes.prng_key)
        ):
            raise TypeError("key must be a scalar typed JAX PRNG key")
        inner_key, planner_key = jr.split(key)
        inner_state = self._inner.init(
            inner_key,
            initial_hard_action_masks=initial_hard_action_masks,
        )
        planner_state = self._planner.init(planner_key)
        children = (inner_state.agent_0_state, inner_state.agent_1_state)
        prepared = self._planner.prepare_pair(
            planner_state,
            self._prototype_state(children[0]),
            self._prototype_state(children[1]),
            inner_state.current_hard_action_masks,
        )
        state = HCCLTwoLiveMemoryFactorizedPlannerState(
            inner_state=inner_state,
            planner_state=prepared.state,
        )
        if not bool(prepared.diagnostics.pair_committed) or not bool(self.state_valid(state)):
            raise ValueError("initial masks must admit both reconstructed P actions")
        return state

    def prepare_event(
        self,
        state: HCCLTwoLiveMemoryFactorizedPlannerState,
    ) -> HCCLCausalCoreEventReceipt:
        self._require_state_contract(state)
        if not bool(self._structure_valid(state)):
            raise ValueError("cannot prepare an event from an invalid persistent structure")
        return self._inner.hccl.world.prepare_event(
            state.inner_state.hccl_state.world_state
        )

    def _planner_identity_rows(
        self,
        state: HCCLTwoLiveMemoryFactorizedPlannerState,
        event: HCCLCausalCoreEventReceipt,
        binding_parts: tuple[Array, Array, Array, Array, Array, Array],
        views: tuple[
            HCCLTwoLiveMemoryFactorizedPlannerDispatchView,
            HCCLTwoLiveMemoryFactorizedPlannerDispatchView,
        ],
    ) -> Array:
        available, transactions, decisions, base_actions, _, memory_actions = (
            binding_parts
        )
        rows: list[Array] = []
        children = self._child_states(state)
        planner_agents = (state.planner_state.agent_0, state.planner_state.agent_1)
        for index, (child, planner_agent, view) in enumerate(
            zip(children, planner_agents, views, strict=True)
        ):
            coordinator = child.coordinator_state
            rows.append(
                _content_tag(
                    self._owner,
                    jnp.asarray(int(HCCLActionLayer.PLANNER), dtype=jnp.int32),
                    jnp.asarray(index, dtype=jnp.int32),
                    state.inner_state.hccl_state.world_state.step_words,
                    event.content_tag_words,
                    coordinator.event_words,
                    coordinator.cached_builder_step_words,
                    coordinator.cached_prototype_step_words,
                    coordinator.cached_feature_generation_words,
                    available[index],
                    transactions[index],
                    decisions[index],
                    base_actions[index],
                    memory_actions[index],
                    view.proposed_action,
                    view.effective_action,
                    state.inner_state.current_hard_action_masks[index],
                    state.planner_state.config_token.astype(jnp.uint32),
                    planner_agent.behavior.step_words,
                    planner_agent.grounded.update_words,
                    view.cache.learned_partner_probabilities,
                    view.cache.world_raw_predictions,
                )
            )
        return jnp.stack(tuple(rows)).astype(jnp.uint32)

    @staticmethod
    def _identities_distinct(
        binding: HCCLTwoLiveMemoryFactorizedPlannerActionBinding,
    ) -> Array:
        identities = jnp.concatenate(
            (
                binding.base.action_receipt_identity_words,
                binding.memory.action_receipt_identity_words,
                binding.planner.action_receipt_identity_words,
            ),
            axis=0,
        )
        distinct = jnp.asarray(True, dtype=jnp.bool_)
        for left in range(6):
            for right in range(left):
                distinct = distinct & (~jnp.all(identities[left] == identities[right]))
        return distinct

    def _binding_tag(
        self,
        binding: HCCLTwoLiveMemoryFactorizedPlannerActionBinding,
    ) -> Array:
        return _content_tag(
            self._owner,
            binding.source_world_words,
            binding.source_world_tag_words,
            binding.event_content_tag_words,
            binding.feedback_binding_available,
            binding.live_memory_transaction_words,
            binding.prototype_decision_words,
            binding.base_actions,
            binding.memory_actions_before_mask,
            binding.memory_actions,
            binding.planner_proposed_actions,
            binding.planner_actions,
            binding.current_hard_action_masks,
            binding.planner_config_token.astype(jnp.uint32),
            binding.behavior_step_words,
            binding.grounded_update_words,
            binding.base.action_receipt_identity_words,
            binding.memory.action_receipt_identity_words,
            binding.planner.action_receipt_identity_words,
            binding.base.content_tag_words,
            binding.memory.content_tag_words,
            binding.planner.content_tag_words,
        )

    def _make_binding(
        self,
        state: HCCLTwoLiveMemoryFactorizedPlannerState,
        event: HCCLCausalCoreEventReceipt,
        views: tuple[
            HCCLTwoLiveMemoryFactorizedPlannerDispatchView,
            HCCLTwoLiveMemoryFactorizedPlannerDispatchView,
        ],
    ) -> HCCLTwoLiveMemoryFactorizedPlannerActionBinding:
        inner_state = state.inner_state
        parts = self._inner._binding_components(inner_state)
        available, transactions, decisions, base_actions, memory_before, memory = parts
        proposed = jnp.stack(tuple(view.proposed_action for view in views)).astype(
            jnp.int32
        )
        planner_actions = jnp.stack(
            tuple(view.effective_action for view in views)
        ).astype(jnp.int32)
        base = self._inner.hccl.bind_action_receipt(
            inner_state.hccl_state,
            event,
            layer=HCCLActionLayer.BASE,
            actions_before_mask=base_actions,
            actions_after_mask=base_actions,
            hard_action_masks=inner_state.current_hard_action_masks,
            action_receipt_identity_words=self._inner._receipt_identity_rows(
                inner_state,
                event,
                HCCLActionLayer.BASE,
                transactions,
                decisions,
                base_actions,
                memory,
            ),
        )
        memory_receipt = self._inner.hccl.bind_action_receipt(
            inner_state.hccl_state,
            event,
            layer=HCCLActionLayer.MEMORY,
            actions_before_mask=memory_before,
            actions_after_mask=memory,
            hard_action_masks=inner_state.current_hard_action_masks,
            action_receipt_identity_words=self._inner._receipt_identity_rows(
                inner_state,
                event,
                HCCLActionLayer.MEMORY,
                transactions,
                decisions,
                base_actions,
                memory,
            ),
        )
        planner_receipt = self._inner.hccl.bind_action_receipt(
            inner_state.hccl_state,
            event,
            layer=HCCLActionLayer.PLANNER,
            actions_before_mask=proposed,
            actions_after_mask=planner_actions,
            hard_action_masks=inner_state.current_hard_action_masks,
            action_receipt_identity_words=self._planner_identity_rows(
                state,
                event,
                parts,
                views,
            ),
        )
        behavior_words = jnp.stack(
            (
                state.planner_state.agent_0.behavior.step_words,
                state.planner_state.agent_1.behavior.step_words,
            )
        ).astype(jnp.uint32)
        grounded_words = jnp.stack(
            (
                state.planner_state.agent_0.grounded.update_words,
                state.planner_state.agent_1.grounded.update_words,
            )
        ).astype(jnp.uint32)
        bare = HCCLTwoLiveMemoryFactorizedPlannerActionBinding(
            source_world_words=inner_state.hccl_state.world_state.step_words,
            source_world_tag_words=inner_state.hccl_state.world_state.content_tag_words,
            event_content_tag_words=event.content_tag_words,
            feedback_binding_available=available,
            live_memory_transaction_words=transactions,
            prototype_decision_words=decisions,
            base_actions=base_actions,
            memory_actions_before_mask=memory_before,
            memory_actions=memory,
            planner_proposed_actions=proposed,
            planner_actions=planner_actions,
            current_hard_action_masks=inner_state.current_hard_action_masks,
            planner_config_token=state.planner_state.config_token,
            behavior_step_words=behavior_words,
            grounded_update_words=grounded_words,
            base=base,
            memory=memory_receipt,
            planner=planner_receipt,
            content_tag_words=jnp.zeros((4,), dtype=jnp.uint32),
        )
        return cast(
            HCCLTwoLiveMemoryFactorizedPlannerActionBinding,
            bare.replace(content_tag_words=self._binding_tag(bare)),
        )

    def _require_binding_contract(
        self,
        binding: HCCLTwoLiveMemoryFactorizedPlannerActionBinding,
    ) -> None:
        if type(binding) is not HCCLTwoLiveMemoryFactorizedPlannerActionBinding:
            raise TypeError("binding must be an exact factorized planner binding")
        for name, shape, dtype in (
            ("source_world_words", (2,), jnp.uint32),
            ("source_world_tag_words", (4,), jnp.uint32),
            ("event_content_tag_words", (4,), jnp.uint32),
            ("feedback_binding_available", (2,), jnp.bool_),
            ("live_memory_transaction_words", (2, 2), jnp.uint32),
            ("prototype_decision_words", (2, 4), jnp.uint32),
            ("base_actions", (2,), jnp.int32),
            ("memory_actions_before_mask", (2,), jnp.int32),
            ("memory_actions", (2,), jnp.int32),
            ("planner_proposed_actions", (2,), jnp.int32),
            ("planner_actions", (2,), jnp.int32),
            ("current_hard_action_masks", (2, 2), jnp.bool_),
            ("planner_config_token", (32,), jnp.uint8),
            ("behavior_step_words", (2, 2), jnp.uint32),
            ("grounded_update_words", (2, 2), jnp.uint32),
            ("content_tag_words", (4,), jnp.uint32),
        ):
            _require_array(
                getattr(binding, name),
                shape=shape,
                dtype=jnp.dtype(dtype),
                label=f"binding.{name}",
            )
        for receipt in (binding.base, binding.memory, binding.planner):
            self._inner.hccl.attribution._require_action_contract(receipt)

    def _binding_integrity_valid(
        self,
        binding: HCCLTwoLiveMemoryFactorizedPlannerActionBinding,
    ) -> Array:
        return (
            jnp.array_equal(binding.base.actions_before_mask, binding.base_actions)
            & jnp.array_equal(binding.base.actions_after_mask, binding.base_actions)
            & jnp.array_equal(
                binding.memory.actions_before_mask,
                binding.memory_actions_before_mask,
            )
            & jnp.array_equal(binding.memory.actions_after_mask, binding.memory_actions)
            & jnp.array_equal(
                binding.planner.actions_before_mask,
                binding.planner_proposed_actions,
            )
            & jnp.array_equal(binding.planner.actions_after_mask, binding.planner_actions)
            & jnp.array_equal(binding.base.hard_action_masks, binding.current_hard_action_masks)
            & jnp.array_equal(
                binding.memory.hard_action_masks,
                binding.current_hard_action_masks,
            )
            & jnp.array_equal(
                binding.planner.hard_action_masks,
                binding.current_hard_action_masks,
            )
            & self._identities_distinct(binding)
            & jnp.array_equal(binding.content_tag_words, self._binding_tag(binding))
        )

    def _feedback_binding_relations(
        self,
        state: HCCLTwoLiveMemoryFactorizedPlannerState,
        binding: HCCLTwoLiveMemoryFactorizedPlannerActionBinding,
    ) -> tuple[Array, Array, Array]:
        children = self._child_states(state)
        required = jnp.stack(tuple(child.pending_binding.available for child in children))
        complete = jnp.array_equal(binding.feedback_binding_available, required)
        matches = jnp.asarray(True, dtype=jnp.bool_)
        for index, child in enumerate(children):
            pending = child.pending_binding
            coordinator = child.coordinator_state
            active = (
                binding.feedback_binding_available[index]
                & jnp.array_equal(
                    binding.live_memory_transaction_words[index],
                    pending.memory_transaction_words,
                )
                & jnp.array_equal(
                    binding.prototype_decision_words[index],
                    pending.prototype_decision_id,
                )
                & (binding.base_actions[index] == pending.base_action_before_retrieval)
                & (binding.memory_actions[index] == pending.effective_action)
                & jnp.array_equal(
                    binding.current_hard_action_masks[index],
                    pending.hard_action_mask,
                )
            )
            inactive = (
                ~binding.feedback_binding_available[index]
                & jnp.all(binding.live_memory_transaction_words[index] == 0)
                & jnp.array_equal(
                    binding.prototype_decision_words[index],
                    coordinator.current_decision_id,
                )
                & (binding.base_actions[index] == coordinator.current_action)
                & (binding.memory_actions[index] == coordinator.current_action)
            )
            matches = matches & jnp.where(pending.available, active, inactive)
        return required.astype(jnp.bool_), complete, matches

    def _settled_dispatch_source(
        self,
        adapter: ExternalLearnedStateLiveMemoryAdapter,
        memory_source: ExternalLearnedStateLiveMemoryAdapterState,
        dispatch: HCCLTwoLiveMemoryFactorizedPlannerDispatchView,
        feedback: ExternalLearnedStateLiveMemoryFeedback | None,
    ) -> HCCLTwoLiveMemorySettledPlannerDispatchSource:
        """Settle prior M feedback once, blank pending, and synchronize P."""

        required = memory_source.pending_binding.available
        supplied = feedback is not None
        supplied_array = jnp.asarray(supplied, dtype=jnp.bool_)
        feedback_value = adapter._blank_feedback() if feedback is None else feedback
        adapter._validate_feedback_static(feedback_value)
        identity = adapter._feedback_identity_valid(
            memory_source,
            feedback_value,
            supplied,
        )
        settlement: LearnedExperientialMemoryFeedbackResult | None = None
        evaluations = jnp.asarray(0, dtype=jnp.int32)
        settled_memory = memory_source.learned_memory_state
        settlement_valid = ~required & ~supplied_array & identity
        if bool(required & identity):
            evaluations = jnp.asarray(1, dtype=jnp.int32)
            settlement = adapter.learned_memory.settle(
                memory_source.learned_memory_state,
                LearnedExperientialMemoryFeedback(
                    transaction_words=feedback_value.memory_transaction_words,
                    retrieval_used=feedback_value.retrieval_used,
                    counterfactual_available=feedback_value.counterfactual_available,
                    counterfactual_delta=feedback_value.counterfactual_delta,
                ),
            )
            settled_memory = settlement.state
            settlement_valid = (
                settlement.diagnostics.transaction_applied
                & ~settlement.state.pending.available
                & adapter.learned_memory.state_valid(settlement.state)
            )
        coordinator = adapter._replace_coordinator_action(
            memory_source.coordinator_state,
            dispatch.replacement,
        )
        transient = ExternalLearnedStateLiveMemoryAdapterState(
            coordinator_state=coordinator,
            learned_memory_state=settled_memory,
            pending_binding=adapter._blank_pending(),
            schema_digest=memory_source.schema_digest,
        )
        dispatch_valid = adapter.state_valid(transient)
        valid = (
            adapter.state_valid(memory_source)
            & identity
            & (required == supplied_array)
            & settlement_valid
            & dispatch.replacement.committed
            & dispatch.valid
            & (coordinator.current_action == dispatch.effective_action)
            & dispatch_valid
        )
        return HCCLTwoLiveMemorySettledPlannerDispatchSource(
            memory_source=memory_source,
            dispatch_state=transient,
            feedback=feedback_value,
            settlement_result=settlement,
            feedback_required=required,
            feedback_supplied=supplied_array,
            feedback_identity_valid=identity,
            settlement_evaluations=evaluations,
            settlement_valid=settlement_valid,
            replacement_committed=dispatch.replacement.committed,
            dispatch_source_valid=dispatch_valid,
            valid=valid,
        )

    @staticmethod
    def _extended_width(adapter: ExternalLearnedStateLiveMemoryAdapter) -> int:
        return adapter.config.coordinator.inner.prototype.oak.stomp.n_total_actions

    def _extended_mask(
        self,
        adapter: ExternalLearnedStateLiveMemoryAdapter,
        value: Array | None,
        *,
        label: str,
    ) -> Array:
        width = self._extended_width(adapter)
        if value is None:
            return jnp.ones((width,), dtype=jnp.bool_)
        return _require_array(
            value,
            shape=(width,),
            dtype=jnp.dtype(jnp.bool_),
            label=label,
        )

    @staticmethod
    def _agent_facts(
        index: int,
        dispatch: HCCLTwoLiveMemoryFactorizedPlannerDispatchView,
        settled: HCCLTwoLiveMemorySettledPlannerDispatchSource,
        transition: ExternalLearnedStateTransition,
        prepared: ExternalLearnedStateLiveMemoryPreparedTransition,
        extended_action_mask: Array,
    ) -> HCCLTwoLiveMemoryFactorizedPlannerPreparedAgentFacts:
        coordinator = prepared.coordinator_result
        if coordinator is None:
            raise ValueError(f"agent {index} preparation did not reach its coordinator")
        prototype = coordinator.evaluated.prepared.inner_result.prototype_result
        finalization = prototype.oak_owner_finalization_trace
        return HCCLTwoLiveMemoryFactorizedPlannerPreparedAgentFacts(
            agent_index=jnp.asarray(index, dtype=jnp.int32),
            dispatch=dispatch,
            settled_dispatch=settled,
            transition=transition,
            live_prepared=prepared,
            prototype_result=prototype,
            raw_stomp_result=prototype.oak_stomp_update_result,
            owner_finalization_trace=finalization,
            extended_action_mask=extended_action_mask,
            candidate_live_state_receipt_words=_tree_digest(
                "candidate-live-state",
                index,
                prepared.candidate_state,
            ),
        )

    @staticmethod
    def _prepared_content_tag(
        prepared: HCCLTwoLiveMemoryFactorizedPlannerPreparedTransaction,
    ) -> Array:
        bare = cast(
            HCCLTwoLiveMemoryFactorizedPlannerPreparedTransaction,
            prepared.replace(
                content_tag_words=jnp.zeros((_DIGEST_WORDS,), dtype=jnp.uint32)
            ),
        )
        return _tree_digest(
            HCCL_TWO_LIVE_MEMORY_FACTORIZED_PLANNER_PREPARED_SCHEMA,
            bare,
        )

    def _require_prepared_contract(
        self,
        prepared: HCCLTwoLiveMemoryFactorizedPlannerPreparedTransaction,
    ) -> None:
        if type(prepared) is not HCCLTwoLiveMemoryFactorizedPlannerPreparedTransaction:
            raise TypeError("prepared must be an exact factorized prepared transaction")
        self._require_state_contract(prepared.source_state)
        self._inner.hccl.world._require_event_contract(prepared.event)
        self._require_binding_contract(prepared.binding)
        self._require_state_contract(prepared.candidate_state)
        _require_array(
            prepared.next_decision_hard_action_masks,
            shape=(_N_AGENTS, _N_ACTIONS),
            dtype=jnp.dtype(jnp.bool_),
            label="prepared.next_decision_hard_action_masks",
        )
        _require_array(
            prepared.effective_planner_actions,
            shape=(_N_AGENTS,),
            dtype=jnp.dtype(jnp.int32),
            label="prepared.effective_planner_actions",
        )
        _require_array(
            prepared.physical_hccl_next_observations,
            shape=(_N_AGENTS, _RAW_OBSERVATION_DIM),
            dtype=jnp.dtype(jnp.float32),
            label="prepared.physical_hccl_next_observations",
        )
        _require_array(
            prepared.grounded_next_constructed_states,
            shape=(_N_AGENTS, self._config.planner.observation_dim),
            dtype=jnp.dtype(jnp.float32),
            label="prepared.grounded_next_constructed_states",
        )
        for name in (
            "source_state_receipt_words",
            "event_receipt_words",
            "config_receipt_words",
            "content_tag_words",
        ):
            _exact_digest(getattr(prepared, name), label=f"prepared.{name}")
        for index, facts in enumerate((prepared.agent_0, prepared.agent_1)):
            if type(facts) is not HCCLTwoLiveMemoryFactorizedPlannerPreparedAgentFacts:
                raise TypeError(f"prepared agent {index} facts have the wrong type")
            if type(facts.live_prepared) is not ExternalLearnedStateLiveMemoryPreparedTransition:
                raise TypeError(f"prepared agent {index} live preparation differs")
            _exact_digest(
                facts.candidate_live_state_receipt_words,
                label=f"prepared.agent_{index}.candidate_live_state_receipt_words",
            )

    def prepare_transaction(
        self,
        state: HCCLTwoLiveMemoryFactorizedPlannerState,
        event: HCCLCausalCoreEventReceipt,
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
        agent_0_extended_action_mask: Array | None = None,
        agent_1_extended_action_mask: Array | None = None,
    ) -> HCCLTwoLiveMemoryFactorizedPlannerPreparedTransaction:
        """Evaluate HCCL, two live transitions, and one paired planner update once."""

        self._require_state_contract(state)
        self._inner.hccl.world._require_event_contract(event)
        next_masks = _require_array(
            next_decision_hard_action_masks,
            shape=(_N_AGENTS, _N_ACTIONS),
            dtype=jnp.dtype(jnp.bool_),
            label="next_decision_hard_action_masks",
        )
        extended_0 = self._extended_mask(
            self._inner.agent_0,
            agent_0_extended_action_mask,
            label="agent_0_extended_action_mask",
        )
        extended_1 = self._extended_mask(
            self._inner.agent_1,
            agent_1_extended_action_mask,
            label="agent_1_extended_action_mask",
        )
        all_inputs = (
            state,
            event,
            agent_0_event_input,
            agent_1_event_input,
            next_masks,
            agent_0_candidate_evidence,
            agent_1_candidate_evidence,
            agent_0_partner_policy_fusion_input,
            agent_1_partner_policy_fusion_input,
            agent_0_partner_policy_fusion_feedback,
            agent_1_partner_policy_fusion_feedback,
            extended_0,
            extended_1,
        )
        if _contains_tracer(all_inputs):
            raise TypeError("factorized HCCL preparation is host/eager-only")
        self._inner.agent_0._validate_event_input_static(agent_0_event_input)
        self._inner.agent_1._validate_event_input_static(agent_1_event_input)

        views = self.reconstruct_planner_dispatch(state)
        reconstruction_valid = views[0].valid & views[1].valid
        source_valid = self._structure_valid(state) & reconstruction_valid
        event_valid = self._inner.hccl.world.event_receipt_valid(
            state.inner_state.hccl_state.world_state,
            event,
        )
        binding = self._make_binding(state, event, views)
        self._require_binding_contract(binding)
        binding_integrity = self._binding_integrity_valid(binding)
        binding_matches = _tree_exact_equal(
            binding,
            self._make_binding(state, event, views),
        )
        required, bindings_complete, bindings_match = self._feedback_binding_relations(
            state,
            binding,
        )
        current_masks_bound = (
            jnp.array_equal(
                binding.current_hard_action_masks,
                state.inner_state.current_hard_action_masks,
            )
            & jnp.array_equal(
                binding.base.hard_action_masks,
                state.inner_state.current_hard_action_masks,
            )
            & jnp.array_equal(
                binding.memory.hard_action_masks,
                state.inner_state.current_hard_action_masks,
            )
            & jnp.array_equal(
                binding.planner.hard_action_masks,
                state.inner_state.current_hard_action_masks,
            )
        )

        hccl_result = self._inner.hccl.stage(
            state.inner_state.hccl_state,
            event,
            binding.base,
            binding.memory,
            binding.planner,
            downstream_candidate_valid=jnp.asarray(True, dtype=jnp.bool_),
        )
        proposals = hccl_result.world_proposals
        unilateral = jnp.stack(
            (
                proposals.signals.net_reward[_M0B1_SLOT, 0]
                - proposals.signals.net_reward[_BB_SLOT, 0],
                proposals.signals.net_reward[_B0M1_SLOT, 1]
                - proposals.signals.net_reward[_BB_SLOT, 1],
            )
        ).astype(jnp.float32)
        children = self._child_states(state)
        feedback_0 = self._inner._feedback(children[0].pending_binding, unilateral[0])
        feedback_1 = self._inner._feedback(children[1].pending_binding, unilateral[1])
        settled_0 = self._settled_dispatch_source(
            self._inner.agent_0,
            children[0],
            views[0],
            feedback_0 if bool(required[0]) else None,
        )
        settled_1 = self._settled_dispatch_source(
            self._inner.agent_1,
            children[1],
            views[1],
            feedback_1 if bool(required[1]) else None,
        )
        pp = cast(
            HCCLCausalCoreProposal,
            jax.tree.map(lambda leaf: leaf[_PP_SLOT], proposals),
        )
        transition_0 = self._inner._transition(
            settled_0.dispatch_state,
            pp,
            agent=0,
            discount=self._gamma,
        )
        transition_1 = self._inner._transition(
            settled_1.dispatch_state,
            pp,
            agent=1,
            discount=self._gamma,
        )
        live_0 = self._inner.agent_0.prepare_transition(
            settled_0.dispatch_state,
            transition_0,
            agent_0_event_input,
            next_masks[0],
            None,
            agent_0_candidate_evidence,
            partner_policy_fusion_input=agent_0_partner_policy_fusion_input,
            partner_policy_fusion_feedback=agent_0_partner_policy_fusion_feedback,
            extended_action_mask=extended_0,
        )
        live_1 = self._inner.agent_1.prepare_transition(
            settled_1.dispatch_state,
            transition_1,
            agent_1_event_input,
            next_masks[1],
            None,
            agent_1_candidate_evidence,
            partner_policy_fusion_input=agent_1_partner_policy_fusion_input,
            partner_policy_fusion_feedback=agent_1_partner_policy_fusion_feedback,
            extended_action_mask=extended_1,
        )
        facts_0 = self._agent_facts(
            0,
            views[0],
            settled_0,
            transition_0,
            live_0,
            extended_0,
        )
        facts_1 = self._agent_facts(
            1,
            views[1],
            settled_1,
            transition_1,
            live_1,
            extended_1,
        )
        post_memory_0 = self._prototype_state(live_0.candidate_state)
        post_memory_1 = self._prototype_state(live_1.candidate_state)
        grounded_next = jnp.stack(
            (
                post_memory_0.current_raw_observation,
                post_memory_1.current_raw_observation,
            )
        ).astype(jnp.float32)
        planner_result = self._planner.completed_transition(
            state.planner_state,
            views[0].planner_prototype,
            views[1].planner_prototype,
            post_memory_0,
            post_memory_1,
            binding.planner_actions,
            pp.signals.net_reward.astype(jnp.float32),
            grounded_next,
            jnp.asarray(self._gamma, dtype=jnp.float32),
            next_masks,
        )
        candidate_inner = HCCLTwoLiveMemoryBridgeState(
            hccl_state=hccl_result.state,
            agent_0_state=live_0.candidate_state,
            agent_1_state=live_1.candidate_state,
            current_hard_action_masks=next_masks,
        )
        candidate = HCCLTwoLiveMemoryFactorizedPlannerState(
            inner_state=candidate_inner,
            planner_state=planner_result.state,
        )
        next_memory_actions = jnp.stack(
            (post_memory_0.current_action, post_memory_1.current_action)
        ).astype(jnp.int32)
        next_base_bound = jnp.array_equal(
            planner_result.diagnostics.next_prepare.base_actions,
            next_memory_actions,
        )
        candidate_valid = (
            self._structure_valid(candidate)
            & planner_result.diagnostics.transaction_committed
            & next_base_bound
            & jnp.array_equal(
                planner_result.state.config_token,
                state.planner_state.config_token,
            )
        )
        supplied = jnp.stack(
            (settled_0.feedback_supplied, settled_1.feedback_supplied)
        ).astype(jnp.bool_)
        live_valid = jnp.stack((live_0.preparation_valid, live_1.preparation_valid))
        pp_executes_p = jnp.array_equal(pp.joint_action_ids, binding.planner_actions)
        preparation_valid = (
            source_valid
            & event_valid
            & binding_integrity
            & binding_matches
            & bindings_complete
            & bindings_match
            & current_masks_bound
            & hccl_result.update_applied
            & settled_0.valid
            & settled_1.valid
            & jnp.all(supplied == required)
            & jnp.all(live_valid)
            & (live_0.settlement_evaluations == 0)
            & (live_1.settlement_evaluations == 0)
            & pp_executes_p
            & planner_result.diagnostics.transaction_committed
            & candidate_valid
        )
        coordinator_0 = cast(Any, live_0.coordinator_result)
        coordinator_1 = cast(Any, live_1.coordinator_result)
        work = HCCLTwoLiveMemoryFactorizedPlannerPrepareWork(
            hccl_stage_calls=jnp.asarray(1, dtype=jnp.int32),
            world_proposal_calls=hccl_result.work.world_proposal_calls,
            attribution_proposal_calls=hccl_result.work.attribution_proposal_calls,
            planner_reconstruction_replacements=jnp.ones(
                (_N_AGENTS,), dtype=jnp.int32
            ),
            feedback_settlement_calls=jnp.stack(
                (settled_0.settlement_evaluations, settled_1.settlement_evaluations)
            ).astype(jnp.int32),
            live_prepare_calls=jnp.ones((_N_AGENTS,), dtype=jnp.int32),
            live_internal_feedback_settlement_calls=jnp.stack(
                (live_0.settlement_evaluations, live_1.settlement_evaluations)
            ).astype(jnp.int32),
            coordinator_update_calls=jnp.stack(
                (live_0.coordinator_evaluations, live_1.coordinator_evaluations)
            ).astype(jnp.int32),
            prototype_update_calls=jnp.stack(
                (
                    coordinator_0.diagnostics.inner_prototype_update_evaluations,
                    coordinator_1.diagnostics.inner_prototype_update_evaluations,
                )
            ).astype(jnp.int32),
            real_stomp_update_evaluations=jnp.stack(
                (
                    facts_0.prototype_result.oak_real_stomp_update_evaluations,
                    facts_1.prototype_result.oak_real_stomp_update_evaluations,
                )
            ).astype(jnp.int32),
            total_stomp_update_evaluations=jnp.stack(
                (
                    facts_0.prototype_result.oak_total_stomp_update_evaluations,
                    facts_1.prototype_result.oak_total_stomp_update_evaluations,
                )
            ).astype(jnp.int32),
            memory_query_calls=jnp.stack(
                (
                    live_0.learned_memory_query_evaluations,
                    live_1.learned_memory_query_evaluations,
                )
            ).astype(jnp.int32),
            memory_write_calls=jnp.stack(
                (
                    live_0.learned_memory_write_evaluations,
                    live_1.learned_memory_write_evaluations,
                )
            ).astype(jnp.int32),
            factorized_completed_transition_calls=jnp.asarray(1, dtype=jnp.int32),
            factorized_behavior_update_attempts=jnp.ones(
                (_N_AGENTS,), dtype=jnp.int32
            ),
            factorized_grounded_update_attempts=jnp.ones(
                (_N_AGENTS,), dtype=jnp.int32
            ),
            actor_backward_calls=jnp.asarray(0, dtype=jnp.int32),
        )
        source_words = _tree_digest("source-state", state)
        event_words = _tree_digest("event", event)
        config_words = self._config_receipt_words()
        bare = HCCLTwoLiveMemoryFactorizedPlannerPreparedTransaction(
            source_state=state,
            event=event,
            binding=binding,
            agent_0_event_input=agent_0_event_input,
            agent_1_event_input=agent_1_event_input,
            next_decision_hard_action_masks=next_masks,
            attempted_hccl_result=hccl_result,
            agent_0=facts_0,
            agent_1=facts_1,
            planner_result=planner_result,
            candidate_state=candidate,
            agent_unilateral_counterfactual_delta=unilateral,
            effective_planner_actions=binding.planner_actions,
            physical_hccl_next_observations=pp.next_observation.astype(jnp.float32),
            grounded_next_constructed_states=grounded_next,
            prior_feedback_required=required,
            prior_feedback_supplied=supplied,
            work=work,
            source_state_receipt_words=source_words,
            event_receipt_words=event_words,
            config_receipt_words=config_words,
            source_state_valid=source_valid,
            event_receipt_valid=event_valid,
            planner_reconstruction_valid=reconstruction_valid,
            binding_integrity_valid=binding_integrity,
            binding_matches_source=binding_matches,
            feedback_bindings_complete=bindings_complete,
            feedback_bindings_match_children=bindings_match,
            current_event_masks_bound=current_masks_bound,
            pp_executes_planner_actions=pp_executes_p,
            planner_may_differ_from_memory=jnp.any(
                binding.planner_actions != binding.memory_actions
            ),
            live_preparations_valid=live_valid,
            factorized_completion_valid=(
                planner_result.diagnostics.transaction_committed
            ),
            candidate_state_valid=candidate_valid,
            preparation_valid=preparation_valid,
            delight_or_actor_backward=jnp.asarray(False, dtype=jnp.bool_),
            content_tag_words=jnp.zeros((_DIGEST_WORDS,), dtype=jnp.uint32),
        )
        prepared = cast(
            HCCLTwoLiveMemoryFactorizedPlannerPreparedTransaction,
            bare.replace(content_tag_words=self._prepared_content_tag(bare)),
        )
        self._require_prepared_contract(prepared)
        return prepared

    @staticmethod
    def _preparation_receipt_tag(
        receipt: HCCLTwoLiveMemoryFactorizedPlannerPreparationReceipt,
    ) -> Array:
        bare = cast(
            HCCLTwoLiveMemoryFactorizedPlannerPreparationReceipt,
            receipt.replace(
                content_tag_words=jnp.zeros((_DIGEST_WORDS,), dtype=jnp.uint32)
            ),
        )
        return _tree_digest(
            HCCL_TWO_LIVE_MEMORY_FACTORIZED_PLANNER_PREPARATION_RECEIPT_SCHEMA,
            bare,
        )

    def integrity_receipt(
        self,
        prepared: HCCLTwoLiveMemoryFactorizedPlannerPreparedTransaction,
    ) -> HCCLTwoLiveMemoryFactorizedPlannerPreparationReceipt:
        self._require_prepared_contract(prepared)
        bare = HCCLTwoLiveMemoryFactorizedPlannerPreparationReceipt(
            source_state_receipt_words=prepared.source_state_receipt_words,
            event_receipt_words=prepared.event_receipt_words,
            config_receipt_words=prepared.config_receipt_words,
            prepared_content_tag_words=prepared.content_tag_words,
            integrity_bound=jnp.asarray(True, dtype=jnp.bool_),
            content_tag_words=jnp.zeros((_DIGEST_WORDS,), dtype=jnp.uint32),
        )
        return cast(
            HCCLTwoLiveMemoryFactorizedPlannerPreparationReceipt,
            bare.replace(content_tag_words=self._preparation_receipt_tag(bare)),
        )

    @staticmethod
    def _downstream_receipt_tag(
        receipt: HCCLTwoLiveMemoryFactorizedPlannerDownstreamReceipt,
    ) -> Array:
        bare = cast(
            HCCLTwoLiveMemoryFactorizedPlannerDownstreamReceipt,
            receipt.replace(
                content_tag_words=jnp.zeros((_DIGEST_WORDS,), dtype=jnp.uint32)
            ),
        )
        return _tree_digest(
            HCCL_TWO_LIVE_MEMORY_FACTORIZED_PLANNER_DOWNSTREAM_RECEIPT_SCHEMA,
            bare,
        )

    def bind_downstream_adoption_receipt(
        self,
        prepared: HCCLTwoLiveMemoryFactorizedPlannerPreparedTransaction,
        *,
        agent_index: int,
        downstream_revision_words: Array,
        downstream_content_digest_words: Array,
        downstream_candidate_valid: Array,
    ) -> HCCLTwoLiveMemoryFactorizedPlannerDownstreamReceipt:
        self._require_prepared_contract(prepared)
        if type(agent_index) is not int or agent_index not in {0, 1}:
            raise ValueError("agent_index must be the exact integer 0 or 1")
        revision = _require_array(
            downstream_revision_words,
            shape=(2,),
            dtype=jnp.dtype(jnp.uint32),
            label="downstream_revision_words",
        )
        digest = _exact_digest(
            downstream_content_digest_words,
            label="downstream_content_digest_words",
        )
        candidate_valid = _require_array(
            downstream_candidate_valid,
            shape=(),
            dtype=jnp.dtype(jnp.bool_),
            label="downstream_candidate_valid",
        )
        facts = prepared.agent_0 if agent_index == 0 else prepared.agent_1
        finalization = facts.owner_finalization_trace
        bare = HCCLTwoLiveMemoryFactorizedPlannerDownstreamReceipt(
            agent_index=jnp.asarray(agent_index, dtype=jnp.int32),
            source_state_receipt_words=prepared.source_state_receipt_words,
            event_receipt_words=prepared.event_receipt_words,
            config_receipt_words=prepared.config_receipt_words,
            prepared_content_tag_words=prepared.content_tag_words,
            candidate_live_state_receipt_words=(
                facts.candidate_live_state_receipt_words
            ),
            raw_stomp_digest=finalization.raw_digest,
            final_stomp_digest=finalization.final_digest,
            owner_finalization_trace_checksum=finalization.trace_checksum,
            extended_action_mask=facts.extended_action_mask,
            downstream_revision_words=revision,
            downstream_content_digest_words=digest,
            downstream_candidate_valid=candidate_valid,
            content_tag_words=jnp.zeros((_DIGEST_WORDS,), dtype=jnp.uint32),
        )
        return cast(
            HCCLTwoLiveMemoryFactorizedPlannerDownstreamReceipt,
            bare.replace(content_tag_words=self._downstream_receipt_tag(bare)),
        )

    @staticmethod
    def _require_preparation_receipt_contract(
        receipt: HCCLTwoLiveMemoryFactorizedPlannerPreparationReceipt,
    ) -> None:
        if type(receipt) is not HCCLTwoLiveMemoryFactorizedPlannerPreparationReceipt:
            raise TypeError("preparation receipt has the wrong exact type")
        for name in (
            "source_state_receipt_words",
            "event_receipt_words",
            "config_receipt_words",
            "prepared_content_tag_words",
            "content_tag_words",
        ):
            _exact_digest(getattr(receipt, name), label=f"preparation_receipt.{name}")
        _require_array(
            receipt.integrity_bound,
            shape=(),
            dtype=jnp.dtype(jnp.bool_),
            label="preparation_receipt.integrity_bound",
        )

    @staticmethod
    def _require_downstream_receipt_contract(
        receipt: HCCLTwoLiveMemoryFactorizedPlannerDownstreamReceipt,
    ) -> None:
        if type(receipt) is not HCCLTwoLiveMemoryFactorizedPlannerDownstreamReceipt:
            raise TypeError("downstream receipt has the wrong exact type")
        _require_array(
            receipt.agent_index,
            shape=(),
            dtype=jnp.dtype(jnp.int32),
            label="downstream_receipt.agent_index",
        )
        for name in (
            "source_state_receipt_words",
            "event_receipt_words",
            "config_receipt_words",
            "prepared_content_tag_words",
            "candidate_live_state_receipt_words",
            "raw_stomp_digest",
            "final_stomp_digest",
            "owner_finalization_trace_checksum",
            "downstream_content_digest_words",
            "content_tag_words",
        ):
            _exact_digest(getattr(receipt, name), label=f"downstream_receipt.{name}")
        _require_array(
            receipt.downstream_revision_words,
            shape=(2,),
            dtype=jnp.dtype(jnp.uint32),
            label="downstream_receipt.downstream_revision_words",
        )
        if (
            getattr(receipt.extended_action_mask, "ndim", None) != 1
            or getattr(receipt.extended_action_mask, "dtype", None)
            != jnp.dtype(jnp.bool_)
        ):
            raise TypeError("downstream receipt extended mask must be rank-one bool")
        _require_array(
            receipt.downstream_candidate_valid,
            shape=(),
            dtype=jnp.dtype(jnp.bool_),
            label="downstream_receipt.downstream_candidate_valid",
        )

    def _downstream_receipt_valid(
        self,
        prepared: HCCLTwoLiveMemoryFactorizedPlannerPreparedTransaction,
        receipt: HCCLTwoLiveMemoryFactorizedPlannerDownstreamReceipt,
        *,
        expected_agent: int,
    ) -> Array:
        self._require_downstream_receipt_contract(receipt)
        expected = self.bind_downstream_adoption_receipt(
            prepared,
            agent_index=expected_agent,
            downstream_revision_words=receipt.downstream_revision_words,
            downstream_content_digest_words=receipt.downstream_content_digest_words,
            downstream_candidate_valid=receipt.downstream_candidate_valid,
        )
        return _tree_exact_equal(receipt, expected)

    @staticmethod
    def _outer_live_result(
        result: ExternalLearnedStateLiveMemoryResult,
        memory_source: ExternalLearnedStateLiveMemoryAdapterState,
        applied: Array,
    ) -> ExternalLearnedStateLiveMemoryResult:
        selected = cast(
            ExternalLearnedStateLiveMemoryAdapterState,
            _tree_select(applied, result.state, memory_source),
        )
        diagnostics = result.diagnostics.replace(
            prior_feedback_settled=(
                result.diagnostics.prior_feedback_settled & applied
            ),
            prior_feedback_learning_applied=(
                result.diagnostics.prior_feedback_learning_applied & applied
            ),
            coordinator_transaction_applied=(
                result.diagnostics.coordinator_transaction_applied & applied
            ),
            learned_memory_transaction_applied=(
                result.diagnostics.learned_memory_transaction_applied & applied
            ),
            cached_action_replacement_committed=(
                result.diagnostics.cached_action_replacement_committed & applied
            ),
            pending_feedback_created=(
                result.diagnostics.pending_feedback_created & applied
            ),
            transaction_applied=result.diagnostics.transaction_applied & applied,
            complete_source_returned=~applied,
            rejected=~applied,
        )
        return cast(
            ExternalLearnedStateLiveMemoryResult,
            result.replace(
                state=selected,
                receipt=result.receipt.replace(
                    integrity_bound=result.receipt.integrity_bound & applied
                ),
                diagnostics=diagnostics,
            ),
        )

    @staticmethod
    def _outer_hccl_result(
        attempted: HCCLWorldAttributionAdapterResult,
        source_state: Any,
        applied: Array,
    ) -> HCCLWorldAttributionAdapterResult:
        state = cast(Any, _tree_select(applied, attempted.state, source_state))
        attribution = attempted.attribution.replace(
            state=state.attribution_state,
            post_transaction_words=state.attribution_state.transaction_words,
            work=attempted.attribution.work.replace(
                discarded_proposal_calls=(
                    attempted.attribution.work.proposal_calls
                    - applied.astype(jnp.int32)
                ),
                committed_pp_calls=applied.astype(jnp.int32),
            ),
            downstream_candidate_valid=(
                attempted.attribution.downstream_candidate_valid & applied
            ),
            update_applied=attempted.attribution.update_applied & applied,
        )
        work = attempted.work.replace(
            discarded_world_proposal_calls=(
                attempted.work.world_proposal_calls - applied.astype(jnp.int32)
            ),
            committed_pp_world_successors=applied.astype(jnp.int32),
        )
        return cast(
            HCCLWorldAttributionAdapterResult,
            attempted.replace(
                state=state,
                attribution=attribution,
                work=work,
                post_transaction_words=state.world_state.step_words,
                downstream_candidate_valid=(
                    attempted.downstream_candidate_valid & applied
                ),
                update_applied=attempted.update_applied & applied,
            ),
        )

    def adopt_prepared_transaction(
        self,
        state: HCCLTwoLiveMemoryFactorizedPlannerState,
        prepared: HCCLTwoLiveMemoryFactorizedPlannerPreparedTransaction,
        preparation_receipt: HCCLTwoLiveMemoryFactorizedPlannerPreparationReceipt,
        agent_0_downstream_receipt: HCCLTwoLiveMemoryFactorizedPlannerDownstreamReceipt,
        agent_1_downstream_receipt: HCCLTwoLiveMemoryFactorizedPlannerDownstreamReceipt,
    ) -> HCCLTwoLiveMemoryFactorizedPlannerResult:
        """Adopt HCCL, both next-M owners, and the planner state or none."""

        self._require_state_contract(state)
        self._require_prepared_contract(prepared)
        self._require_preparation_receipt_contract(preparation_receipt)
        self._require_downstream_receipt_contract(agent_0_downstream_receipt)
        self._require_downstream_receipt_contract(agent_1_downstream_receipt)
        if _contains_tracer(
            (
                state,
                prepared,
                preparation_receipt,
                agent_0_downstream_receipt,
                agent_1_downstream_receipt,
            )
        ):
            raise TypeError("factorized HCCL adoption is host/eager-only")

        current_source_words = _tree_digest("source-state", state)
        prepared_source_words = _tree_digest("source-state", prepared.source_state)
        source_receipt_valid = (
            _tree_exact_equal(state, prepared.source_state)
            & jnp.array_equal(
                current_source_words,
                prepared.source_state_receipt_words,
            )
            & jnp.array_equal(
                prepared_source_words,
                prepared.source_state_receipt_words,
            )
            & jnp.array_equal(
                preparation_receipt.source_state_receipt_words,
                prepared.source_state_receipt_words,
            )
        )
        event_receipt_valid = (
            jnp.array_equal(
                _tree_digest("event", prepared.event),
                prepared.event_receipt_words,
            )
            & jnp.array_equal(
                preparation_receipt.event_receipt_words,
                prepared.event_receipt_words,
            )
            & self._inner.hccl.world.event_receipt_valid(
                state.inner_state.hccl_state.world_state,
                prepared.event,
            )
        )
        config_words = self._config_receipt_words()
        config_receipt_valid = (
            jnp.array_equal(config_words, prepared.config_receipt_words)
            & jnp.array_equal(
                preparation_receipt.config_receipt_words,
                prepared.config_receipt_words,
            )
        )
        expected_preparation = self.integrity_receipt(prepared)
        prepared_tag_valid = jnp.array_equal(
            prepared.content_tag_words,
            self._prepared_content_tag(prepared),
        )
        preparation_receipt_valid = (
            _tree_exact_equal(preparation_receipt, expected_preparation)
            & preparation_receipt.integrity_bound
            & prepared_tag_valid
        )
        downstream_receipts_valid = jnp.stack(
            (
                self._downstream_receipt_valid(
                    prepared,
                    agent_0_downstream_receipt,
                    expected_agent=0,
                ),
                self._downstream_receipt_valid(
                    prepared,
                    agent_1_downstream_receipt,
                    expected_agent=1,
                ),
            )
        ).astype(jnp.bool_)
        downstream_candidates = jnp.stack(
            (
                agent_0_downstream_receipt.downstream_candidate_valid,
                agent_1_downstream_receipt.downstream_candidate_valid,
            )
        ).astype(jnp.bool_)
        candidate_valid = (
            self._structure_valid(prepared.candidate_state)
            & prepared.candidate_state_valid
            & prepared.planner_result.diagnostics.transaction_committed
        )
        current_state_valid = self.state_valid(state)
        preliminary = (
            current_state_valid
            & source_receipt_valid
            & event_receipt_valid
            & config_receipt_valid
            & preparation_receipt_valid
            & jnp.all(downstream_receipts_valid)
            & jnp.all(downstream_candidates)
            & prepared.preparation_valid
            & candidate_valid
        )

        child_receipt_0 = self._inner.agent_0.integrity_receipt(
            prepared.agent_0.live_prepared
        ).replace(integrity_bound=preliminary)
        child_receipt_1 = self._inner.agent_1.integrity_receipt(
            prepared.agent_1.live_prepared
        ).replace(integrity_bound=preliminary)
        raw_0 = self._inner.agent_0.adopt_prepared_transition(
            prepared.agent_0.settled_dispatch.dispatch_state,
            prepared.agent_0.live_prepared,
            cast(ExternalLearnedStateLiveMemoryIntegrityReceipt, child_receipt_0),
        )
        raw_1 = self._inner.agent_1.adopt_prepared_transition(
            prepared.agent_1.settled_dispatch.dispatch_state,
            prepared.agent_1.live_prepared,
            cast(ExternalLearnedStateLiveMemoryIntegrityReceipt, child_receipt_1),
        )
        adopted_candidate = HCCLTwoLiveMemoryFactorizedPlannerState(
            inner_state=HCCLTwoLiveMemoryBridgeState(
                hccl_state=prepared.attempted_hccl_result.state,
                agent_0_state=raw_0.state,
                agent_1_state=raw_1.state,
                current_hard_action_masks=(
                    prepared.next_decision_hard_action_masks
                ),
            ),
            planner_state=prepared.planner_result.state,
        )
        candidate_matches = _tree_exact_equal(
            adopted_candidate,
            prepared.candidate_state,
        )
        final_candidate_authenticated = self.state_valid(adopted_candidate)
        applied = (
            preliminary
            & raw_0.diagnostics.transaction_applied
            & raw_1.diagnostics.transaction_applied
            & candidate_matches
            & final_candidate_authenticated
        )
        final_state = cast(
            HCCLTwoLiveMemoryFactorizedPlannerState,
            _tree_select(applied, adopted_candidate, state),
        )
        public_0 = self._outer_live_result(
            raw_0,
            state.inner_state.agent_0_state,
            applied,
        )
        public_1 = self._outer_live_result(
            raw_1,
            state.inner_state.agent_1_state,
            applied,
        )
        public_hccl = self._outer_hccl_result(
            prepared.attempted_hccl_result,
            state.inner_state.hccl_state,
            applied,
        )
        live_applied = jnp.stack(
            (
                public_0.diagnostics.transaction_applied,
                public_1.diagnostics.transaction_applied,
            )
        ).astype(jnp.bool_)
        zero_pair = jnp.zeros((_N_AGENTS,), dtype=jnp.int32)
        false = jnp.asarray(False, dtype=jnp.bool_)
        return HCCLTwoLiveMemoryFactorizedPlannerResult(
            state=final_state,
            prepared=prepared,
            preparation_receipt=preparation_receipt,
            agent_0_downstream_receipt=agent_0_downstream_receipt,
            agent_1_downstream_receipt=agent_1_downstream_receipt,
            hccl_result=public_hccl,
            agent_0_result=public_0,
            agent_1_result=public_1,
            planner_result=prepared.planner_result,
            adoption_work=HCCLTwoLiveMemoryFactorizedPlannerAdoptionWork(
                preparation_integrity_checks=jnp.asarray(1, dtype=jnp.int32),
                downstream_receipt_integrity_checks=jnp.asarray(
                    _N_AGENTS, dtype=jnp.int32
                ),
                live_integrity_calls=jnp.ones((_N_AGENTS,), dtype=jnp.int32),
                world_proposal_calls=jnp.asarray(0, dtype=jnp.int32),
                attribution_proposal_calls=jnp.asarray(0, dtype=jnp.int32),
                coordinator_update_calls=zero_pair,
                prototype_update_calls=zero_pair,
                stomp_update_evaluations=zero_pair,
                memory_query_calls=zero_pair,
                memory_write_calls=zero_pair,
                factorized_model_update_calls=jnp.asarray(0, dtype=jnp.int32),
                planner_reconstruction_replacements=jnp.full(
                    (_N_AGENTS,), 2, dtype=jnp.int32
                ),
                planner_cache_authentication_evaluations=jnp.full(
                    (_N_AGENTS,), 2, dtype=jnp.int32
                ),
            ),
            source_state_receipt_valid=source_receipt_valid,
            event_receipt_valid=event_receipt_valid,
            config_receipt_valid=config_receipt_valid,
            preparation_receipt_valid=preparation_receipt_valid,
            downstream_receipts_valid=downstream_receipts_valid,
            downstream_candidates_valid=downstream_candidates,
            candidate_state_valid=candidate_valid,
            source_state_authenticated=current_state_valid,
            final_candidate_authenticated=final_candidate_authenticated,
            hccl_update_applied=public_hccl.update_applied,
            live_adapter_updates_applied=live_applied,
            factorized_planner_update_applied=(
                prepared.planner_result.diagnostics.transaction_committed & applied
            ),
            next_decision_masks_installed=applied,
            modeled_planner_proposal_available=jnp.asarray(True, dtype=jnp.bool_),
            bounded_hccl_world_planner_action_consumed=(
                prepared.pp_executes_planner_actions & public_hccl.update_applied
            ),
            pp_candidate_committed=public_hccl.update_applied,
            external_environment_dispatch_authority=false,
            physical_dispatch_authority=false,
            safety_authority=false,
            evidence_authority=false,
            promotion_authority=false,
            actor_backward_calls=jnp.asarray(0, dtype=jnp.int32),
            delight_or_actor_backward=false,
            update_applied=applied,
        )

    def resource_budget(
        self,
        state: HCCLTwoLiveMemoryFactorizedPlannerState | None = None,
    ) -> HCCLTwoLiveMemoryFactorizedPlannerResourceBudget:
        reference = self.init(jr.key(0)) if state is None else state
        self._require_state_contract(reference)
        if not bool(self.state_valid(reference)):
            raise ValueError("resource measurement requires a valid composite state")
        inner = self._inner.resource_budget(reference.inner_state)
        planner = self._planner.resource_budget(reference.planner_state)
        return HCCLTwoLiveMemoryFactorizedPlannerResourceBudget(
            hccl_state_owners=1,
            live_memory_adapter_state_owners=2,
            prototype_state_owners=2,
            additional_prototype_state_owners=0,
            factorized_planner_state_owners=1,
            persisted_planner_prototype_snapshots=0,
            inner_persistent_state_nbytes=inner.total_persistent_state_nbytes,
            factorized_planner_state_nbytes=planner.measured_pair_nbytes,
            total_persistent_state_nbytes=(
                inner.total_persistent_state_nbytes + planner.measured_pair_nbytes
            ),
            persisted_preparation_records=0,
            persisted_preparation_bytes=0,
            prepare_hccl_stage_calls_per_transaction=1,
            prepare_live_calls_per_transaction=2,
            maximum_feedback_settlements_per_transaction=2,
            planner_reconstruction_replacements_per_transaction=2,
            factorized_completed_transition_calls_per_transaction=1,
            adopt_planner_reconstruction_replacements_per_transaction=4,
            adopt_planner_cache_authentication_evaluations_per_transaction=4,
            adopt_world_or_learner_reevaluations=0,
            output_write_calls=0,
            artifact_bytes_written=0,
        )


def measure_hccl_two_live_memory_factorized_planner_state_nbytes(
    state: HCCLTwoLiveMemoryFactorizedPlannerState,
) -> int:
    if type(state) is not HCCLTwoLiveMemoryFactorizedPlannerState:
        raise TypeError("state must be an exact factorized HCCL state")
    return _tree_nbytes(state)


__all__ = [
    "HCCL_TWO_LIVE_MEMORY_FACTORIZED_PLANNER_BINDING_SCHEMA",
    "HCCL_TWO_LIVE_MEMORY_FACTORIZED_PLANNER_DOWNSTREAM_RECEIPT_SCHEMA",
    "HCCL_TWO_LIVE_MEMORY_FACTORIZED_PLANNER_EVIDENCE_LEVEL",
    "HCCL_TWO_LIVE_MEMORY_FACTORIZED_PLANNER_LIMITATIONS",
    "HCCL_TWO_LIVE_MEMORY_FACTORIZED_PLANNER_PREPARATION_RECEIPT_SCHEMA",
    "HCCL_TWO_LIVE_MEMORY_FACTORIZED_PLANNER_PREPARED_SCHEMA",
    "HCCL_TWO_LIVE_MEMORY_FACTORIZED_PLANNER_SCHEMA",
    "HCCL_TWO_LIVE_MEMORY_FACTORIZED_PLANNER_STATE_SCHEMA",
    "HCCL_TWO_LIVE_MEMORY_FACTORIZED_PLANNER_STATUS",
    "HCCLTwoLiveMemoryFactorizedPlannerActionBinding",
    "HCCLTwoLiveMemoryFactorizedPlannerAdoptionWork",
    "HCCLTwoLiveMemoryFactorizedPlannerBridge",
    "HCCLTwoLiveMemoryFactorizedPlannerConfig",
    "HCCLTwoLiveMemoryFactorizedPlannerDispatchView",
    "HCCLTwoLiveMemoryFactorizedPlannerDownstreamReceipt",
    "HCCLTwoLiveMemoryFactorizedPlannerPreparationReceipt",
    "HCCLTwoLiveMemoryFactorizedPlannerPreparedAgentFacts",
    "HCCLTwoLiveMemoryFactorizedPlannerPreparedTransaction",
    "HCCLTwoLiveMemoryFactorizedPlannerPrepareWork",
    "HCCLTwoLiveMemoryFactorizedPlannerResourceBudget",
    "HCCLTwoLiveMemoryFactorizedPlannerResult",
    "HCCLTwoLiveMemoryFactorizedPlannerState",
    "HCCLTwoLiveMemorySettledPlannerDispatchSource",
    "measure_hccl_two_live_memory_factorized_planner_state_nbytes",
]
