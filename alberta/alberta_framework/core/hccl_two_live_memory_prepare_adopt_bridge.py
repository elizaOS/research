# mypy: disable-error-code="arg-type,attr-defined,call-arg,no-any-return,type-var"
"""Transient two-phase adoption for one HCCL and two live-memory agents.

This L0 wrapper deliberately reuses the exact configuration and persistent
state of :class:`HCCLTwoLiveMemoryBridge`.  ``prepare_transaction`` evaluates
one HCCL transaction and exactly one pure live-memory preparation per agent.
It retains those attempted donor facts only in a transient prepared value.
``adopt_prepared_transaction`` performs content checks, two child integrity
adoptions, and one all-or-none state selection; it never reevaluates a world,
coordinator, Prototype, STOMP learner, state builder, or learned memory.

Per-agent downstream receipts bind the prepared live candidate, the raw and
final STOMP owner digests, the complete owner-finalization trace checksum, and
the independently sized extended-action mask.  They are unkeyed integrity
receipts, not caller authentication.  P=M remains the explicit no-planner
rung.  Delight is unavailable because this wrapper executes no Kondo actor
backward.
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
from jaxtyping import Bool, Int, UInt

from alberta_framework.core.external_learned_state_live_memory_adapter import (
    ExternalLearnedStateLiveMemoryAdapter,
    ExternalLearnedStateLiveMemoryEventInput,
    ExternalLearnedStateLiveMemoryIntegrityReceipt,
    ExternalLearnedStateLiveMemoryPreparedTransition,
    ExternalLearnedStateLiveMemoryResult,
)
from alberta_framework.core.external_learned_state_router_audit_coordinator import (
    ExternalBuilderCandidateAuditEvidence,
)
from alberta_framework.core.hccl_two_live_memory_bridge import (
    _B0M1_SLOT,
    _BB_SLOT,
    _M0B1_SLOT,
    _PP_SLOT,
    HCCLTwoLiveMemoryActionBinding,
    HCCLTwoLiveMemoryBridge,
    HCCLTwoLiveMemoryBridgeConfig,
    HCCLTwoLiveMemoryBridgeState,
    _contains_tracer,
    _contrast_exact_zero,
    _require_array,
    _tree_exact_equal,
    _tree_select,
)
from alberta_framework.core.hccl_world_attribution_adapter import (
    HCCLWorldAttributionAdapterResult,
)
from alberta_framework.core.options import STOMPUpdateResult
from alberta_framework.core.prototype_agent import (
    PrototypePartnerPolicyFusionFeedback,
    PrototypePartnerPolicyFusionInput,
    PrototypeUpdateResult,
)
from alberta_framework.core.stomp_owner_finalization import (
    STOMPOwnerFinalizationTrace,
)
from alberta_framework.streams.hccl_causal_core import (
    HCCLCausalCoreEventReceipt,
    HCCLCausalCoreProposal,
)

HCCL_TWO_LIVE_MEMORY_PREPARE_ADOPT_SCHEMA = (
    "alberta.hccl-two-live-memory-prepare-adopt.v1"
)
HCCL_TWO_LIVE_MEMORY_PREPARED_SCHEMA = (
    "alberta.hccl-two-live-memory-prepared-transaction.v1"
)
HCCL_TWO_LIVE_MEMORY_PREPARATION_RECEIPT_SCHEMA = (
    "alberta.hccl-two-live-memory-preparation-receipt.v1"
)
HCCL_TWO_LIVE_MEMORY_DOWNSTREAM_RECEIPT_SCHEMA = (
    "alberta.hccl-two-live-memory-downstream-adoption-receipt.v1"
)
HCCL_TWO_LIVE_MEMORY_PREPARE_ADOPT_STATUS = (
    "l0-development-hccl-two-live-memory-prepare-adopt"
)
HCCL_TWO_LIVE_MEMORY_PREPARE_ADOPT_EVIDENCE_LEVEL = "L0"
HCCL_TWO_LIVE_MEMORY_PREPARE_ADOPT_LIMITATIONS = (
    "preparation-is-transient-and-not-checkpointed",
    "integrity-receipts-are-not-caller-authentication",
    "agent-0-feedback-is-M0B1-minus-BB-only",
    "agent-1-feedback-is-B0M1-minus-BB-only",
    "P-equals-M-no-planner-rung",
    "delight-and-actor-backward-are-unavailable",
    "host-eager-only",
    "no-life-schedule-output-artifact-evidence-or-promotion-authority",
)

_N_AGENTS = 2
_N_ACTIONS = 2
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
    """Hash exact host material for this deliberately host-only boundary."""

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


def _exact_bool_scalar(value: Any, *, label: str) -> Array:
    return _require_array(
        value,
        shape=(),
        dtype=jnp.dtype(jnp.bool_),
        label=label,
    )


@chex.dataclass(frozen=True)
class HCCLTwoLiveMemoryPreparedAgentFacts:
    """One live preparation and its exact nested Prototype/STOMP witnesses."""

    agent_index: Int[Array, ""]
    live_prepared: ExternalLearnedStateLiveMemoryPreparedTransition
    prototype_result: PrototypeUpdateResult
    raw_stomp_result: STOMPUpdateResult
    owner_finalization_trace: STOMPOwnerFinalizationTrace
    extended_action_mask: Bool[Array, " extended_actions"]
    candidate_live_state_receipt_words: UInt[Array, " 8"]


@chex.dataclass(frozen=True)
class HCCLTwoLiveMemoryPrepareWork:
    """Exact attempted donor evaluations performed only during preparation."""

    hccl_stage_calls: Int[Array, ""]
    world_proposal_calls: Int[Array, ""]
    attribution_proposal_calls: Int[Array, ""]
    live_prepare_calls: Int[Array, " 2"]
    feedback_settlement_calls: Int[Array, " 2"]
    coordinator_update_calls: Int[Array, " 2"]
    prototype_update_calls: Int[Array, " 2"]
    real_stomp_update_evaluations: Int[Array, " 2"]
    total_stomp_update_evaluations: Int[Array, " 2"]
    memory_query_calls: Int[Array, " 2"]
    memory_write_calls: Int[Array, " 2"]


@chex.dataclass(frozen=True)
class HCCLTwoLiveMemoryPreparedTransaction:
    """Transient complete donor proposal; never part of persistent state."""

    source_state: HCCLTwoLiveMemoryBridgeState
    event: HCCLCausalCoreEventReceipt
    binding: HCCLTwoLiveMemoryActionBinding
    agent_0_event_input: ExternalLearnedStateLiveMemoryEventInput
    agent_1_event_input: ExternalLearnedStateLiveMemoryEventInput
    next_decision_hard_action_masks: Bool[Array, "2 2"]
    attempted_hccl_result: HCCLWorldAttributionAdapterResult
    agent_0: HCCLTwoLiveMemoryPreparedAgentFacts
    agent_1: HCCLTwoLiveMemoryPreparedAgentFacts
    candidate_state: HCCLTwoLiveMemoryBridgeState
    agent_unilateral_counterfactual_delta: Array
    prior_feedback_required: Bool[Array, " 2"]
    prior_feedback_supplied: Bool[Array, " 2"]
    work: HCCLTwoLiveMemoryPrepareWork
    source_state_receipt_words: UInt[Array, " 8"]
    event_receipt_words: UInt[Array, " 8"]
    config_receipt_words: UInt[Array, " 8"]
    binding_integrity_valid: Bool[Array, ""]
    binding_matches_source: Bool[Array, ""]
    feedback_bindings_complete: Bool[Array, ""]
    feedback_bindings_match_children: Bool[Array, ""]
    current_event_masks_bound: Bool[Array, ""]
    planner_equals_memory: Bool[Array, ""]
    pp_executes_memory_actions: Bool[Array, ""]
    no_planner_rung_valid: Bool[Array, ""]
    candidate_state_valid: Bool[Array, ""]
    preparation_valid: Bool[Array, ""]
    delight_or_actor_backward: Bool[Array, ""]
    content_tag_words: UInt[Array, " 8"]


@chex.dataclass(frozen=True)
class HCCLTwoLiveMemoryPreparationReceipt:
    """Exact source/event/config/preparation integrity binding."""

    source_state_receipt_words: UInt[Array, " 8"]
    event_receipt_words: UInt[Array, " 8"]
    config_receipt_words: UInt[Array, " 8"]
    prepared_content_tag_words: UInt[Array, " 8"]
    integrity_bound: Bool[Array, ""]
    content_tag_words: UInt[Array, " 8"]


@chex.dataclass(frozen=True)
class HCCLTwoLiveMemoryDownstreamAdoptionReceipt:
    """Agent-specific repeated-option/downstream integrity verdict."""

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
class HCCLTwoLiveMemoryAdoptionWork:
    """Integrity/adoption work; all donor reevaluation counts are exact zero."""

    preparation_integrity_checks: Int[Array, ""]
    downstream_receipt_integrity_checks: Int[Array, ""]
    live_integrity_adoption_calls: Int[Array, " 2"]
    world_proposal_calls: Int[Array, ""]
    attribution_proposal_calls: Int[Array, ""]
    coordinator_update_calls: Int[Array, " 2"]
    prototype_update_calls: Int[Array, " 2"]
    stomp_update_evaluations: Int[Array, " 2"]
    memory_query_calls: Int[Array, " 2"]
    memory_write_calls: Int[Array, " 2"]


@chex.dataclass(frozen=True)
class HCCLTwoLiveMemoryPrepareAdoptResult:
    """Outer-gated public result plus inspectable transient donor attempts."""

    state: HCCLTwoLiveMemoryBridgeState
    prepared: HCCLTwoLiveMemoryPreparedTransaction
    preparation_receipt: HCCLTwoLiveMemoryPreparationReceipt
    agent_0_downstream_receipt: HCCLTwoLiveMemoryDownstreamAdoptionReceipt
    agent_1_downstream_receipt: HCCLTwoLiveMemoryDownstreamAdoptionReceipt
    hccl_result: HCCLWorldAttributionAdapterResult
    agent_0_result: ExternalLearnedStateLiveMemoryResult
    agent_1_result: ExternalLearnedStateLiveMemoryResult
    adoption_work: HCCLTwoLiveMemoryAdoptionWork
    source_state_receipt_valid: Bool[Array, ""]
    event_receipt_valid: Bool[Array, ""]
    config_receipt_valid: Bool[Array, ""]
    preparation_receipt_valid: Bool[Array, ""]
    downstream_receipts_valid: Bool[Array, " 2"]
    downstream_candidates_valid: Bool[Array, " 2"]
    candidate_state_valid: Bool[Array, ""]
    hccl_update_applied: Bool[Array, ""]
    live_adapter_updates_applied: Bool[Array, " 2"]
    coordinator_updates_applied: Bool[Array, " 2"]
    prototype_updates_applied: Bool[Array, " 2"]
    stomp_updates_applied: Bool[Array, " 2"]
    learned_memory_updates_applied: Bool[Array, " 2"]
    builder_learning_applied: Bool[Array, " 2"]
    next_decision_masks_installed: Bool[Array, ""]
    delight_or_actor_backward: Bool[Array, ""]
    update_applied: Bool[Array, ""]


@dataclasses.dataclass(frozen=True, slots=True)
class HCCLTwoLiveMemoryPrepareAdoptResourceBudget:
    """Persistent ownership plus exact prepare/adopt logical call bounds."""

    hccl_state_owners: int
    live_memory_adapter_state_owners: int
    total_persistent_state_nbytes: int
    persisted_preparation_records: int
    persisted_preparation_bytes: int
    prepared_checkpoint_supported: bool
    prepare_hccl_stage_calls_per_transaction: int
    prepare_live_adapter_calls_per_transaction: int
    prepare_world_proposal_calls_per_transaction: int
    prepare_attribution_proposal_calls_per_transaction: int
    adopt_live_integrity_calls_per_transaction: int
    adopt_world_or_learner_reevaluations: int
    output_write_calls: int
    artifact_bytes_written: int

    def to_config(self) -> dict[str, int | bool]:
        return dataclasses.asdict(self)


class HCCLTwoLiveMemoryPrepareAdoptBridge:
    """Stateless two-phase view over one exact v1 composite owner."""

    def __init__(self, config: HCCLTwoLiveMemoryBridgeConfig):
        if type(config) is not HCCLTwoLiveMemoryBridgeConfig:
            raise TypeError("config must be exact HCCLTwoLiveMemoryBridgeConfig")
        self._inner = HCCLTwoLiveMemoryBridge(config)

    @property
    def config(self) -> HCCLTwoLiveMemoryBridgeConfig:
        return self._inner.config

    @property
    def inner(self) -> HCCLTwoLiveMemoryBridge:
        return self._inner

    def to_config(self) -> dict[str, object]:
        return {
            "type": type(self).__name__,
            "schema": HCCL_TWO_LIVE_MEMORY_PREPARE_ADOPT_SCHEMA,
            "prepared_schema": HCCL_TWO_LIVE_MEMORY_PREPARED_SCHEMA,
            "preparation_receipt_schema": (
                HCCL_TWO_LIVE_MEMORY_PREPARATION_RECEIPT_SCHEMA
            ),
            "downstream_receipt_schema": (
                HCCL_TWO_LIVE_MEMORY_DOWNSTREAM_RECEIPT_SCHEMA
            ),
            "mechanism_status": HCCL_TWO_LIVE_MEMORY_PREPARE_ADOPT_STATUS,
            "evidence_level": HCCL_TWO_LIVE_MEMORY_PREPARE_ADOPT_EVIDENCE_LEVEL,
            "inner": self._inner.to_config(),
            "hccl_state_owners": 1,
            "live_memory_adapter_state_owners": 2,
            "external_coordinator_state_owners": 2,
            "learned_memory_controller_state_owners": 2,
            "prototype_state_owners": 2,
            "preparation_persisted": False,
            "preparation_checkpoint_supported": False,
            "prepare_live_calls": 2,
            "adopt_world_or_learner_reevaluations": 0,
            "planner_action_relation": "P=M-no-planner-rung",
            "planner_layer_authority": False,
            "delight_or_actor_backward": False,
            "delight_interpretation": "unavailable-no-Kondo-actor-backward",
            "caller_identity_authenticated": False,
            "host_eager_only": True,
            "output_writes_authorized": False,
            "artifact_authorized": False,
            "evidence_authorized": False,
            "promotion_authorized": False,
            "limitations": list(HCCL_TWO_LIVE_MEMORY_PREPARE_ADOPT_LIMITATIONS),
        }

    def _config_receipt_words(self) -> Array:
        return _digest_bytes(_canonical_json_bytes(self.to_config()))

    def init(
        self,
        key: Array,
        *,
        initial_hard_action_masks: Array | None = None,
    ) -> HCCLTwoLiveMemoryBridgeState:
        return self._inner.init(
            key,
            initial_hard_action_masks=initial_hard_action_masks,
        )

    def state_valid(self, state: HCCLTwoLiveMemoryBridgeState) -> Array:
        return self._inner.state_valid(state)

    def prepare_event(
        self,
        state: HCCLTwoLiveMemoryBridgeState,
    ) -> HCCLCausalCoreEventReceipt:
        return self._inner.prepare_event(state)

    def bind_live_memory_actions(
        self,
        state: HCCLTwoLiveMemoryBridgeState,
        event: HCCLCausalCoreEventReceipt,
    ) -> HCCLTwoLiveMemoryActionBinding:
        return self._inner.bind_live_memory_actions(state, event)

    @staticmethod
    def _extended_width(adapter: ExternalLearnedStateLiveMemoryAdapter) -> int:
        prototype = adapter.config.coordinator.inner.prototype
        return prototype.oak.stomp.n_total_actions

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
        prepared: ExternalLearnedStateLiveMemoryPreparedTransition,
        extended_action_mask: Array,
    ) -> HCCLTwoLiveMemoryPreparedAgentFacts:
        coordinator = prepared.coordinator_result
        if coordinator is None:
            raise ValueError(
                f"agent {index} preparation did not reach its coordinator"
            )
        prototype = coordinator.evaluated.prepared.inner_result.prototype_result
        finalization = prototype.oak_owner_finalization_trace
        raw_stomp = prototype.oak_stomp_update_result
        return HCCLTwoLiveMemoryPreparedAgentFacts(
            agent_index=jnp.asarray(index, dtype=jnp.int32),
            live_prepared=prepared,
            prototype_result=prototype,
            raw_stomp_result=raw_stomp,
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
        prepared: HCCLTwoLiveMemoryPreparedTransaction,
    ) -> Array:
        bare = cast(
            HCCLTwoLiveMemoryPreparedTransaction,
            prepared.replace(
                content_tag_words=jnp.zeros((_DIGEST_WORDS,), dtype=jnp.uint32)
            ),
        )
        return _tree_digest(HCCL_TWO_LIVE_MEMORY_PREPARED_SCHEMA, bare)

    def _require_prepared_contract(
        self,
        prepared: HCCLTwoLiveMemoryPreparedTransaction,
    ) -> None:
        if type(prepared) is not HCCLTwoLiveMemoryPreparedTransaction:
            raise TypeError("prepared must be an exact prepared transaction")
        self._inner._require_state_contract(prepared.source_state)
        self._inner.hccl.world._require_event_contract(prepared.event)
        self._inner._require_binding_contract(prepared.binding)
        self._inner._require_state_contract(prepared.candidate_state)
        _require_array(
            prepared.next_decision_hard_action_masks,
            shape=(_N_AGENTS, _N_ACTIONS),
            dtype=jnp.dtype(jnp.bool_),
            label="prepared.next_decision_hard_action_masks",
        )
        for name in (
            "source_state_receipt_words",
            "event_receipt_words",
            "config_receipt_words",
            "content_tag_words",
        ):
            _exact_digest(getattr(prepared, name), label=f"prepared.{name}")
        for index, agent in enumerate((prepared.agent_0, prepared.agent_1)):
            if type(agent) is not HCCLTwoLiveMemoryPreparedAgentFacts:
                raise TypeError(f"prepared agent {index} facts have the wrong type")
            if type(agent.live_prepared) is not ExternalLearnedStateLiveMemoryPreparedTransition:
                raise TypeError(f"prepared agent {index} live preparation differs")
            _require_array(
                agent.agent_index,
                shape=(),
                dtype=jnp.dtype(jnp.int32),
                label=f"prepared.agent_{index}.agent_index",
            )
            _exact_digest(
                agent.candidate_live_state_receipt_words,
                label=f"prepared.agent_{index}.candidate_live_state_receipt_words",
            )

    def prepare_transaction(
        self,
        state: HCCLTwoLiveMemoryBridgeState,
        event: HCCLCausalCoreEventReceipt,
        binding: HCCLTwoLiveMemoryActionBinding,
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
    ) -> HCCLTwoLiveMemoryPreparedTransaction:
        """Evaluate every donor once without persisting or adopting the result."""

        self._inner._require_state_contract(state)
        self._inner.hccl.world._require_event_contract(event)
        self._inner._require_binding_contract(binding)
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
            binding,
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
            raise TypeError("prepare/adopt HCCL wrapper is host/eager-only")
        self._inner.agent_0._validate_event_input_static(agent_0_event_input)
        self._inner.agent_1._validate_event_input_static(agent_1_event_input)

        source_valid = self._inner.state_valid(state)
        event_valid = self._inner.hccl.world.event_receipt_valid(
            state.hccl_state.world_state,
            event,
        )
        binding_integrity = self._inner._binding_integrity_valid(binding)
        binding_matches = _tree_exact_equal(
            binding,
            self._inner._make_binding(state, event),
        )
        required, bindings_complete, bindings_match = (
            self._inner._feedback_binding_relations(state, binding)
        )
        current_masks_bound = (
            jnp.all(binding.current_hard_action_masks == state.current_hard_action_masks)
            & jnp.all(binding.base.hard_action_masks == state.current_hard_action_masks)
            & jnp.all(binding.memory.hard_action_masks == state.current_hard_action_masks)
            & jnp.all(binding.planner.hard_action_masks == state.current_hard_action_masks)
        )

        hccl_result = self._inner.hccl.stage(
            state.hccl_state,
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
        feedback_0 = self._inner._feedback(
            state.agent_0_state.pending_binding,
            unilateral[0],
        )
        feedback_1 = self._inner._feedback(
            state.agent_1_state.pending_binding,
            unilateral[1],
        )
        pp = cast(
            HCCLCausalCoreProposal,
            jax.tree.map(lambda leaf: leaf[_PP_SLOT], proposals),
        )
        transition_0 = self._inner._transition(
            state.agent_0_state,
            pp,
            agent=0,
            discount=(
                self.config.agent_0.coordinator.inner.ensemble.world_model.gamma
            ),
        )
        transition_1 = self._inner._transition(
            state.agent_1_state,
            pp,
            agent=1,
            discount=(
                self.config.agent_1.coordinator.inner.ensemble.world_model.gamma
            ),
        )
        prepared_0 = self._inner.agent_0.prepare_transition(
            state.agent_0_state,
            transition_0,
            agent_0_event_input,
            next_masks[0],
            feedback_0 if bool(required[0]) else None,
            agent_0_candidate_evidence,
            partner_policy_fusion_input=agent_0_partner_policy_fusion_input,
            partner_policy_fusion_feedback=agent_0_partner_policy_fusion_feedback,
            extended_action_mask=extended_0,
        )
        prepared_1 = self._inner.agent_1.prepare_transition(
            state.agent_1_state,
            transition_1,
            agent_1_event_input,
            next_masks[1],
            feedback_1 if bool(required[1]) else None,
            agent_1_candidate_evidence,
            partner_policy_fusion_input=agent_1_partner_policy_fusion_input,
            partner_policy_fusion_feedback=agent_1_partner_policy_fusion_feedback,
            extended_action_mask=extended_1,
        )
        facts_0 = self._agent_facts(0, prepared_0, extended_0)
        facts_1 = self._agent_facts(1, prepared_1, extended_1)

        planner_equals_memory = (
            jnp.all(binding.planner.actions_before_mask == binding.memory_actions)
            & jnp.all(binding.planner.actions_after_mask == binding.memory_actions)
        )
        pp_executes_memory = jnp.all(pp.joint_action_ids == binding.memory_actions)
        planner_zero = _contrast_exact_zero(
            hccl_result.attribution.contrasts.planner_total
        ) & _contrast_exact_zero(hccl_result.attribution.contrasts.planner_interaction)
        no_planner = planner_equals_memory & pp_executes_memory & planner_zero
        candidate = HCCLTwoLiveMemoryBridgeState(
            hccl_state=hccl_result.state,
            agent_0_state=prepared_0.candidate_state,
            agent_1_state=prepared_1.candidate_state,
            current_hard_action_masks=next_masks,
        )
        candidate_valid = self._inner.state_valid(candidate)
        supplied = jnp.stack(
            (prepared_0.feedback_supplied, prepared_1.feedback_supplied)
        ).astype(jnp.bool_)
        preparation_valid = (
            source_valid
            & event_valid
            & binding_integrity
            & binding_matches
            & bindings_complete
            & bindings_match
            & current_masks_bound
            & hccl_result.update_applied
            & prepared_0.preparation_valid
            & prepared_1.preparation_valid
            & jnp.all(supplied == required)
            & no_planner
            & candidate_valid
        )
        coordinator_0 = cast(Any, prepared_0.coordinator_result)
        coordinator_1 = cast(Any, prepared_1.coordinator_result)
        work = HCCLTwoLiveMemoryPrepareWork(
            hccl_stage_calls=jnp.asarray(1, dtype=jnp.int32),
            world_proposal_calls=hccl_result.work.world_proposal_calls,
            attribution_proposal_calls=hccl_result.work.attribution_proposal_calls,
            live_prepare_calls=jnp.ones((_N_AGENTS,), dtype=jnp.int32),
            feedback_settlement_calls=jnp.stack(
                (prepared_0.settlement_evaluations, prepared_1.settlement_evaluations)
            ).astype(jnp.int32),
            coordinator_update_calls=jnp.stack(
                (prepared_0.coordinator_evaluations, prepared_1.coordinator_evaluations)
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
                    prepared_0.learned_memory_query_evaluations,
                    prepared_1.learned_memory_query_evaluations,
                )
            ).astype(jnp.int32),
            memory_write_calls=jnp.stack(
                (
                    prepared_0.learned_memory_write_evaluations,
                    prepared_1.learned_memory_write_evaluations,
                )
            ).astype(jnp.int32),
        )
        source_words = _tree_digest("source-state", state)
        event_words = _tree_digest("event", event)
        config_words = self._config_receipt_words()
        bare = HCCLTwoLiveMemoryPreparedTransaction(
            source_state=state,
            event=event,
            binding=binding,
            agent_0_event_input=agent_0_event_input,
            agent_1_event_input=agent_1_event_input,
            next_decision_hard_action_masks=next_masks,
            attempted_hccl_result=hccl_result,
            agent_0=facts_0,
            agent_1=facts_1,
            candidate_state=candidate,
            agent_unilateral_counterfactual_delta=unilateral,
            prior_feedback_required=required,
            prior_feedback_supplied=supplied,
            work=work,
            source_state_receipt_words=source_words,
            event_receipt_words=event_words,
            config_receipt_words=config_words,
            binding_integrity_valid=binding_integrity,
            binding_matches_source=binding_matches,
            feedback_bindings_complete=bindings_complete,
            feedback_bindings_match_children=bindings_match,
            current_event_masks_bound=current_masks_bound,
            planner_equals_memory=planner_equals_memory,
            pp_executes_memory_actions=pp_executes_memory,
            no_planner_rung_valid=no_planner,
            candidate_state_valid=candidate_valid,
            preparation_valid=preparation_valid,
            delight_or_actor_backward=jnp.asarray(False, dtype=jnp.bool_),
            content_tag_words=jnp.zeros((_DIGEST_WORDS,), dtype=jnp.uint32),
        )
        prepared = cast(
            HCCLTwoLiveMemoryPreparedTransaction,
            bare.replace(content_tag_words=self._prepared_content_tag(bare)),
        )
        self._require_prepared_contract(prepared)
        return prepared

    @staticmethod
    def _preparation_receipt_tag(
        receipt: HCCLTwoLiveMemoryPreparationReceipt,
    ) -> Array:
        bare = cast(
            HCCLTwoLiveMemoryPreparationReceipt,
            receipt.replace(
                content_tag_words=jnp.zeros((_DIGEST_WORDS,), dtype=jnp.uint32)
            ),
        )
        return _tree_digest(HCCL_TWO_LIVE_MEMORY_PREPARATION_RECEIPT_SCHEMA, bare)

    def integrity_receipt(
        self,
        prepared: HCCLTwoLiveMemoryPreparedTransaction,
    ) -> HCCLTwoLiveMemoryPreparationReceipt:
        self._require_prepared_contract(prepared)
        bare = HCCLTwoLiveMemoryPreparationReceipt(
            source_state_receipt_words=prepared.source_state_receipt_words,
            event_receipt_words=prepared.event_receipt_words,
            config_receipt_words=prepared.config_receipt_words,
            prepared_content_tag_words=prepared.content_tag_words,
            integrity_bound=jnp.asarray(True, dtype=jnp.bool_),
            content_tag_words=jnp.zeros((_DIGEST_WORDS,), dtype=jnp.uint32),
        )
        return cast(
            HCCLTwoLiveMemoryPreparationReceipt,
            bare.replace(content_tag_words=self._preparation_receipt_tag(bare)),
        )

    @staticmethod
    def _downstream_receipt_tag(
        receipt: HCCLTwoLiveMemoryDownstreamAdoptionReceipt,
    ) -> Array:
        bare = cast(
            HCCLTwoLiveMemoryDownstreamAdoptionReceipt,
            receipt.replace(
                content_tag_words=jnp.zeros((_DIGEST_WORDS,), dtype=jnp.uint32)
            ),
        )
        return _tree_digest(HCCL_TWO_LIVE_MEMORY_DOWNSTREAM_RECEIPT_SCHEMA, bare)

    def bind_downstream_adoption_receipt(
        self,
        prepared: HCCLTwoLiveMemoryPreparedTransaction,
        *,
        agent_index: int,
        downstream_revision_words: Array,
        downstream_content_digest_words: Array,
        downstream_candidate_valid: Array,
    ) -> HCCLTwoLiveMemoryDownstreamAdoptionReceipt:
        """Bind one downstream verdict to one exact nested owner candidate."""

        self._require_prepared_contract(prepared)
        if type(agent_index) is not int or agent_index not in {0, 1}:
            raise ValueError("agent_index must be the exact integer 0 or 1")
        revision = _require_array(
            downstream_revision_words,
            shape=(2,),
            dtype=jnp.dtype(jnp.uint32),
            label="downstream_revision_words",
        )
        downstream_digest = _exact_digest(
            downstream_content_digest_words,
            label="downstream_content_digest_words",
        )
        candidate_valid = _exact_bool_scalar(
            downstream_candidate_valid,
            label="downstream_candidate_valid",
        )
        facts = prepared.agent_0 if agent_index == 0 else prepared.agent_1
        finalization = facts.owner_finalization_trace
        bare = HCCLTwoLiveMemoryDownstreamAdoptionReceipt(
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
            downstream_content_digest_words=downstream_digest,
            downstream_candidate_valid=candidate_valid,
            content_tag_words=jnp.zeros((_DIGEST_WORDS,), dtype=jnp.uint32),
        )
        return cast(
            HCCLTwoLiveMemoryDownstreamAdoptionReceipt,
            bare.replace(content_tag_words=self._downstream_receipt_tag(bare)),
        )

    def _require_preparation_receipt_contract(
        self,
        receipt: HCCLTwoLiveMemoryPreparationReceipt,
    ) -> None:
        if type(receipt) is not HCCLTwoLiveMemoryPreparationReceipt:
            raise TypeError("preparation receipt has the wrong exact type")
        for name in (
            "source_state_receipt_words",
            "event_receipt_words",
            "config_receipt_words",
            "prepared_content_tag_words",
            "content_tag_words",
        ):
            _exact_digest(getattr(receipt, name), label=f"preparation_receipt.{name}")
        _exact_bool_scalar(
            receipt.integrity_bound,
            label="preparation_receipt.integrity_bound",
        )

    def _require_downstream_receipt_contract(
        self,
        receipt: HCCLTwoLiveMemoryDownstreamAdoptionReceipt,
    ) -> None:
        if type(receipt) is not HCCLTwoLiveMemoryDownstreamAdoptionReceipt:
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
        _exact_bool_scalar(
            receipt.downstream_candidate_valid,
            label="downstream_receipt.downstream_candidate_valid",
        )

    def _downstream_receipt_valid(
        self,
        prepared: HCCLTwoLiveMemoryPreparedTransaction,
        receipt: HCCLTwoLiveMemoryDownstreamAdoptionReceipt,
        *,
        expected_agent: int,
    ) -> Array:
        self._require_downstream_receipt_contract(receipt)
        expected = self.bind_downstream_adoption_receipt(
            prepared,
            agent_index=expected_agent,
            downstream_revision_words=receipt.downstream_revision_words,
            downstream_content_digest_words=(
                receipt.downstream_content_digest_words
            ),
            downstream_candidate_valid=receipt.downstream_candidate_valid,
        )
        return _tree_exact_equal(receipt, expected)

    @staticmethod
    def _outer_live_result(
        result: ExternalLearnedStateLiveMemoryResult,
        source_state: Any,
        applied: Array,
    ) -> ExternalLearnedStateLiveMemoryResult:
        selected = cast(
            Any,
            _tree_select(applied, result.state, source_state),
        )
        diagnostics = result.diagnostics.replace(
            prior_feedback_settled=result.diagnostics.prior_feedback_settled
            & applied,
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
        receipt = result.receipt.replace(
            integrity_bound=result.receipt.integrity_bound & applied
        )
        return result.replace(
            state=selected,
            receipt=receipt,
            diagnostics=diagnostics,
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
        return attempted.replace(
            state=state,
            attribution=attribution,
            work=work,
            post_transaction_words=state.world_state.step_words,
            downstream_candidate_valid=(
                attempted.downstream_candidate_valid & applied
            ),
            update_applied=attempted.update_applied & applied,
        )

    @staticmethod
    def _raw_memory_applied(
        prepared: ExternalLearnedStateLiveMemoryPreparedTransition,
    ) -> Array:
        memory = prepared.learned_memory_result
        return (
            jnp.asarray(False, dtype=jnp.bool_)
            if memory is None
            else memory.diagnostics.transaction_applied
        )

    def adopt_prepared_transaction(
        self,
        state: HCCLTwoLiveMemoryBridgeState,
        prepared: HCCLTwoLiveMemoryPreparedTransaction,
        preparation_receipt: HCCLTwoLiveMemoryPreparationReceipt,
        agent_0_downstream_receipt: HCCLTwoLiveMemoryDownstreamAdoptionReceipt,
        agent_1_downstream_receipt: HCCLTwoLiveMemoryDownstreamAdoptionReceipt,
    ) -> HCCLTwoLiveMemoryPrepareAdoptResult:
        """Validate receipts and select all owner candidates or the source."""

        self._inner._require_state_contract(state)
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
            raise TypeError("prepare/adopt HCCL wrapper is host/eager-only")

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
                state.hccl_state.world_state,
                prepared.event,
            )
        )
        current_config_words = self._config_receipt_words()
        config_receipt_valid = (
            jnp.array_equal(current_config_words, prepared.config_receipt_words)
            & jnp.array_equal(
                preparation_receipt.config_receipt_words,
                prepared.config_receipt_words,
            )
        )
        expected_preparation_receipt = self.integrity_receipt(prepared)
        prepared_tag_valid = jnp.array_equal(
            prepared.content_tag_words,
            self._prepared_content_tag(prepared),
        )
        preparation_receipt_valid = (
            _tree_exact_equal(
                preparation_receipt,
                expected_preparation_receipt,
            )
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
        candidate_valid = self._inner.state_valid(prepared.candidate_state)
        preliminary = (
            self._inner.state_valid(state)
            & source_receipt_valid
            & event_receipt_valid
            & config_receipt_valid
            & preparation_receipt_valid
            & jnp.all(downstream_receipts_valid)
            & jnp.all(downstream_candidates)
            & prepared.preparation_valid
            & prepared.no_planner_rung_valid
            & candidate_valid
        )

        child_receipt_0 = self._inner.agent_0.integrity_receipt(
            prepared.agent_0.live_prepared
        ).replace(integrity_bound=preliminary)
        child_receipt_1 = self._inner.agent_1.integrity_receipt(
            prepared.agent_1.live_prepared
        ).replace(integrity_bound=preliminary)
        result_0 = self._inner.agent_0.adopt_prepared_transition(
            state.agent_0_state,
            prepared.agent_0.live_prepared,
            cast(ExternalLearnedStateLiveMemoryIntegrityReceipt, child_receipt_0),
        )
        result_1 = self._inner.agent_1.adopt_prepared_transition(
            state.agent_1_state,
            prepared.agent_1.live_prepared,
            cast(ExternalLearnedStateLiveMemoryIntegrityReceipt, child_receipt_1),
        )
        adopted_candidate = HCCLTwoLiveMemoryBridgeState(
            hccl_state=prepared.attempted_hccl_result.state,
            agent_0_state=result_0.state,
            agent_1_state=result_1.state,
            current_hard_action_masks=prepared.next_decision_hard_action_masks,
        )
        child_candidate_matches = _tree_exact_equal(
            adopted_candidate,
            prepared.candidate_state,
        )
        applied = (
            preliminary
            & result_0.diagnostics.transaction_applied
            & result_1.diagnostics.transaction_applied
            & child_candidate_matches
            & self._inner.state_valid(adopted_candidate)
        )
        final_state = cast(
            HCCLTwoLiveMemoryBridgeState,
            _tree_select(applied, adopted_candidate, state),
        )
        public_0 = self._outer_live_result(result_0, state.agent_0_state, applied)
        public_1 = self._outer_live_result(result_1, state.agent_1_state, applied)
        public_hccl = self._outer_hccl_result(
            prepared.attempted_hccl_result,
            state.hccl_state,
            applied,
        )

        coordinator_raw = jnp.stack(
            (
                cast(Any, prepared.agent_0.live_prepared.coordinator_result)
                .diagnostics.transaction_applied,
                cast(Any, prepared.agent_1.live_prepared.coordinator_result)
                .diagnostics.transaction_applied,
            )
        ).astype(jnp.bool_)
        prototype_raw = jnp.stack(
            (
                ~prepared.agent_0.prototype_result.transition_diagnostics.rejected,
                ~prepared.agent_1.prototype_result.transition_diagnostics.rejected,
            )
        ).astype(jnp.bool_)
        stomp_raw = jnp.stack(
            (
                prepared.agent_0.raw_stomp_result.update_applied,
                prepared.agent_1.raw_stomp_result.update_applied,
            )
        ).astype(jnp.bool_)
        memory_raw = jnp.stack(
            (
                self._raw_memory_applied(prepared.agent_0.live_prepared),
                self._raw_memory_applied(prepared.agent_1.live_prepared),
            )
        ).astype(jnp.bool_)
        builder_raw = jnp.stack(
            (
                cast(Any, prepared.agent_0.live_prepared.coordinator_result)
                .diagnostics.builder_learning_applied,
                cast(Any, prepared.agent_1.live_prepared.coordinator_result)
                .diagnostics.builder_learning_applied,
            )
        ).astype(jnp.bool_)
        live_public = jnp.stack(
            (
                public_0.diagnostics.transaction_applied,
                public_1.diagnostics.transaction_applied,
            )
        ).astype(jnp.bool_)
        zero_pair = jnp.zeros((_N_AGENTS,), dtype=jnp.int32)
        return HCCLTwoLiveMemoryPrepareAdoptResult(
            state=final_state,
            prepared=prepared,
            preparation_receipt=preparation_receipt,
            agent_0_downstream_receipt=agent_0_downstream_receipt,
            agent_1_downstream_receipt=agent_1_downstream_receipt,
            hccl_result=public_hccl,
            agent_0_result=public_0,
            agent_1_result=public_1,
            adoption_work=HCCLTwoLiveMemoryAdoptionWork(
                preparation_integrity_checks=jnp.asarray(1, dtype=jnp.int32),
                downstream_receipt_integrity_checks=jnp.asarray(
                    _N_AGENTS, dtype=jnp.int32
                ),
                live_integrity_adoption_calls=jnp.ones(
                    (_N_AGENTS,), dtype=jnp.int32
                ),
                world_proposal_calls=jnp.asarray(0, dtype=jnp.int32),
                attribution_proposal_calls=jnp.asarray(0, dtype=jnp.int32),
                coordinator_update_calls=zero_pair,
                prototype_update_calls=zero_pair,
                stomp_update_evaluations=zero_pair,
                memory_query_calls=zero_pair,
                memory_write_calls=zero_pair,
            ),
            source_state_receipt_valid=source_receipt_valid,
            event_receipt_valid=event_receipt_valid,
            config_receipt_valid=config_receipt_valid,
            preparation_receipt_valid=preparation_receipt_valid,
            downstream_receipts_valid=downstream_receipts_valid,
            downstream_candidates_valid=downstream_candidates,
            candidate_state_valid=candidate_valid,
            hccl_update_applied=public_hccl.update_applied,
            live_adapter_updates_applied=live_public,
            coordinator_updates_applied=coordinator_raw & applied,
            prototype_updates_applied=prototype_raw & applied,
            stomp_updates_applied=stomp_raw & applied,
            learned_memory_updates_applied=memory_raw & applied,
            builder_learning_applied=builder_raw & applied,
            next_decision_masks_installed=applied,
            delight_or_actor_backward=jnp.asarray(False, dtype=jnp.bool_),
            update_applied=applied,
        )

    def resource_budget(
        self,
        state: HCCLTwoLiveMemoryBridgeState | None = None,
    ) -> HCCLTwoLiveMemoryPrepareAdoptResourceBudget:
        inner = self._inner.resource_budget(state)
        return HCCLTwoLiveMemoryPrepareAdoptResourceBudget(
            hccl_state_owners=1,
            live_memory_adapter_state_owners=2,
            total_persistent_state_nbytes=inner.total_persistent_state_nbytes,
            persisted_preparation_records=0,
            persisted_preparation_bytes=0,
            prepared_checkpoint_supported=False,
            prepare_hccl_stage_calls_per_transaction=1,
            prepare_live_adapter_calls_per_transaction=2,
            prepare_world_proposal_calls_per_transaction=(
                inner.max_world_proposal_calls_per_transaction
            ),
            prepare_attribution_proposal_calls_per_transaction=(
                inner.max_attribution_proposal_calls_per_transaction
            ),
            adopt_live_integrity_calls_per_transaction=2,
            adopt_world_or_learner_reevaluations=0,
            output_write_calls=0,
            artifact_bytes_written=0,
        )


__all__ = [
    "HCCL_TWO_LIVE_MEMORY_DOWNSTREAM_RECEIPT_SCHEMA",
    "HCCL_TWO_LIVE_MEMORY_PREPARATION_RECEIPT_SCHEMA",
    "HCCL_TWO_LIVE_MEMORY_PREPARED_SCHEMA",
    "HCCL_TWO_LIVE_MEMORY_PREPARE_ADOPT_EVIDENCE_LEVEL",
    "HCCL_TWO_LIVE_MEMORY_PREPARE_ADOPT_LIMITATIONS",
    "HCCL_TWO_LIVE_MEMORY_PREPARE_ADOPT_SCHEMA",
    "HCCL_TWO_LIVE_MEMORY_PREPARE_ADOPT_STATUS",
    "HCCLTwoLiveMemoryAdoptionWork",
    "HCCLTwoLiveMemoryDownstreamAdoptionReceipt",
    "HCCLTwoLiveMemoryPreparationReceipt",
    "HCCLTwoLiveMemoryPrepareAdoptBridge",
    "HCCLTwoLiveMemoryPrepareAdoptResourceBudget",
    "HCCLTwoLiveMemoryPrepareAdoptResult",
    "HCCLTwoLiveMemoryPrepareWork",
    "HCCLTwoLiveMemoryPreparedAgentFacts",
    "HCCLTwoLiveMemoryPreparedTransaction",
]
