# mypy: disable-error-code="arg-type,attr-defined,call-arg,no-any-return,type-var"
"""Red-first contract for transient two-phase HCCL/live-memory adoption.

The successor in this file wraps :class:`HCCLTwoLiveMemoryBridge`; it does not
add methods or state to that v1 owner.  Preparation performs the expensive
donor work exactly once and exposes the raw nested STOMP owner/finalization
facts required by binding-only repeated-option sidecars.  Adoption validates
content-bound receipts and either installs all three owner candidates or
returns the complete source.

The HCCL causal world fixes raw observations at width 16.  All other dimensions
are the smallest compatible values.  This is an L0 development mechanism test,
not evidence or a life runner.  Delight is unavailable: none of these paths
executes a Kondo actor backward.
"""

from __future__ import annotations

import dataclasses
from typing import Any, cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from test_hccl_two_live_memory_bridge import (
    N_ACTIONS,
    _event_input,
    _tree_exact,
)
from test_hccl_two_live_memory_bridge import (
    _config as _v1_config,
)

import alberta_framework.core as core_api
from alberta_framework.core.delight import CandidateUpdateAuditConfig
from alberta_framework.core.external_learned_state_router_audit_coordinator import (
    ExternalBuilderCandidateAuditEvidence,
)
from alberta_framework.core.hccl_two_live_memory_bridge import (
    HCCLTwoLiveMemoryBridge,
    HCCLTwoLiveMemoryBridgeConfig,
    HCCLTwoLiveMemoryBridgeState,
)
from alberta_framework.core.hccl_two_live_memory_prepare_adopt_bridge import (
    HCCL_TWO_LIVE_MEMORY_PREPARE_ADOPT_STATUS,
    HCCLTwoLiveMemoryDownstreamAdoptionReceipt,
    HCCLTwoLiveMemoryPreparationReceipt,
    HCCLTwoLiveMemoryPrepareAdoptBridge,
    HCCLTwoLiveMemoryPrepareAdoptResult,
    HCCLTwoLiveMemoryPreparedAgentFacts,
    HCCLTwoLiveMemoryPreparedTransaction,
)
from alberta_framework.core.partner_policy_fusion import (
    PartnerPolicyFusionConfig,
    PartnerPolicyFusionFeedback,
)
from alberta_framework.core.prototype_agent import (
    PrototypePartnerPolicyFusionFeedback,
    PrototypePartnerPolicyFusionInput,
)

pytestmark = [pytest.mark.integration, pytest.mark.slow]

N_AGENTS = 2
N_OPTIONS = 1
EXTENDED_ACTIONS = N_ACTIONS + N_OPTIONS
TOTAL_DIM = 18


@pytest.fixture(autouse=True)
def _bounded_jax_execution() -> object:
    with jax.disable_jit():
        yield


def _config(*, replacement_interval: int = 0) -> HCCLTwoLiveMemoryBridgeConfig:
    """Enable the smallest partner sidecar and a permissive mechanism audit."""

    source = _v1_config()
    candidate_audit = CandidateUpdateAuditConfig(
        candidate_semantics="update",
        max_update_norm=100.0,
        max_retention_loss_increase=100.0,
        max_safety_cost_increase=100.0,
        min_objective_descent_alignment=-1.0,
        min_retention_descent_alignment=-1.0,
        min_safety_descent_alignment=-1.0,
    )
    fusion = PartnerPolicyFusionConfig(
        max_partners=1,
        context_dim=TOTAL_DIM,
        n_actions=N_ACTIONS,
        min_feedback_for_learned_routing=1,
        counter_cap=8,
    )

    def with_sidecars(live: Any) -> Any:
        coordinator = live.coordinator
        feature = coordinator.inner.prototype.prototype_feature_lifecycle
        assert feature is not None
        prototype = dataclasses.replace(
            coordinator.inner.prototype,
            prototype_feature_lifecycle=dataclasses.replace(
                feature,
                replacement_interval=replacement_interval,
            ),
            partner_policy_fusion=fusion,
        )
        inner = dataclasses.replace(coordinator.inner, prototype=prototype)
        return dataclasses.replace(
            live,
            coordinator=dataclasses.replace(
                coordinator,
                inner=inner,
                candidate_audit=candidate_audit,
            ),
        )

    return dataclasses.replace(
        source,
        agent_0=with_sidecars(source.agent_0),
        agent_1=with_sidecars(source.agent_1),
    )


def _bridge() -> tuple[
    HCCLTwoLiveMemoryPrepareAdoptBridge,
    HCCLTwoLiveMemoryBridgeState,
]:
    bridge = HCCLTwoLiveMemoryPrepareAdoptBridge(_config())
    return bridge, bridge.init(jr.key(101))


def _next_decision_id(state: HCCLTwoLiveMemoryBridgeState, agent: int) -> jax.Array:
    child = state.agent_0_state if agent == 0 else state.agent_1_state
    return child.coordinator_state.inner_state.prototype_state.current_decision_id.at[3].add(
        jnp.asarray(1, dtype=jnp.uint32)
    )


def _partner_input(
    bridge: HCCLTwoLiveMemoryPrepareAdoptBridge,
    state: HCCLTwoLiveMemoryBridgeState,
    agent: int,
) -> PrototypePartnerPolicyFusionInput:
    live = bridge.inner.agent_0 if agent == 0 else bridge.inner.agent_1
    fusion = live.coordinator.inner.prototype.partner_policy_fusion
    assert fusion is not None
    return PrototypePartnerPolicyFusionInput(
        available=jnp.asarray(False, dtype=jnp.bool_),
        prototype_decision_id=_next_decision_id(state, agent),
        observation_id=jnp.asarray(100 + agent, dtype=jnp.int32),
        context_id=jnp.asarray(200 + agent, dtype=jnp.int32),
        context_features=jnp.full(
            (TOTAL_DIM,),
            jnp.asarray(0.25 + 0.25 * agent, dtype=jnp.float32),
            dtype=jnp.float32,
        ),
        safety_action_mask=jnp.ones((N_ACTIONS,), dtype=jnp.bool_),
        keyboard_available=jnp.asarray(False, dtype=jnp.bool_),
        keyboard_vector=jnp.zeros((N_OPTIONS,), dtype=jnp.float32),
        messages=fusion.empty_messages(),
    )


def _partner_feedback() -> PrototypePartnerPolicyFusionFeedback:
    unavailable = jnp.asarray(False, dtype=jnp.bool_)
    missing = jnp.asarray(-1, dtype=jnp.int32)
    return PrototypePartnerPolicyFusionFeedback(
        prototype_decision_id=jnp.zeros((4,), dtype=jnp.uint32),
        feedback=PartnerPolicyFusionFeedback(
            available=unavailable,
            decision_id=missing,
            executed_event_id=missing,
            decision_words=jnp.zeros((2,), dtype=jnp.uint32),
            executed_event_words=jnp.zeros((2,), dtype=jnp.uint32),
            executed_action=missing,
            partner_id=missing,
            assistance_value_available=unavailable,
            realized_assistance_value=jnp.asarray(0.0, dtype=jnp.float32),
            safety_outcome_available=unavailable,
            safety_outcome_ok=unavailable,
        ),
    )


def _prepare(
    bridge: HCCLTwoLiveMemoryPrepareAdoptBridge,
    state: HCCLTwoLiveMemoryBridgeState,
    *,
    evidence: tuple[
        ExternalBuilderCandidateAuditEvidence | None,
        ExternalBuilderCandidateAuditEvidence | None,
    ] = (None, None),
    sidecars: bool = True,
) -> HCCLTwoLiveMemoryPreparedTransaction:
    event = bridge.prepare_event(state)
    binding = bridge.bind_live_memory_actions(state, event)
    partner_inputs = (
        (_partner_input(bridge, state, 0), _partner_input(bridge, state, 1))
        if sidecars
        else (None, None)
    )
    partner_feedback = (
        (_partner_feedback(), _partner_feedback()) if sidecars else (None, None)
    )
    return bridge.prepare_transaction(
        state,
        event,
        binding,
        _event_input(30),
        _event_input(31),
        next_decision_hard_action_masks=jnp.ones(
            (N_AGENTS, N_ACTIONS), dtype=jnp.bool_
        ),
        agent_0_candidate_evidence=evidence[0],
        agent_1_candidate_evidence=evidence[1],
        agent_0_partner_policy_fusion_input=partner_inputs[0],
        agent_1_partner_policy_fusion_input=partner_inputs[1],
        agent_0_partner_policy_fusion_feedback=partner_feedback[0],
        agent_1_partner_policy_fusion_feedback=partner_feedback[1],
        agent_0_extended_action_mask=jnp.asarray(
            (True, True, True), dtype=jnp.bool_
        ),
        agent_1_extended_action_mask=jnp.asarray(
            (True, True, False), dtype=jnp.bool_
        ),
    )


def _audit_evidence(
    agent: HCCLTwoLiveMemoryPreparedAgentFacts,
) -> ExternalBuilderCandidateAuditEvidence:
    coordinator = agent.live_prepared.coordinator_result
    assert coordinator is not None
    prepared = coordinator.evaluated.prepared
    source = prepared.source_state
    update = prepared.learning_proposal.candidate_parameter_update
    true = jnp.asarray(True, dtype=jnp.bool_)
    return ExternalBuilderCandidateAuditEvidence(
        source_event_words=source.event_words,
        source_builder_step_words=source.cached_builder_step_words,
        source_prototype_step_words=source.cached_prototype_step_words,
        source_feature_generation_words=source.cached_feature_generation_words,
        decision_id=source.current_decision_id,
        objective_probe_gradient=-update,
        retention_probe_gradient=-update,
        safety_cost_gradient=-update,
        objective_probe_available=true,
        retention_probe_available=true,
        safety_probe_available=true,
        probe_independence_attested=true,
        advantage=jnp.asarray(0.5, dtype=jnp.float32),
        action_surprisal=jnp.asarray(0.5, dtype=jnp.float32),
        safety_cost=jnp.asarray(0.0, dtype=jnp.float32),
        advantage_available=true,
        action_surprisal_available=true,
        safety_cost_available=true,
    )


def _receipts(
    bridge: HCCLTwoLiveMemoryPrepareAdoptBridge,
    prepared: HCCLTwoLiveMemoryPreparedTransaction,
    *,
    valid: tuple[bool, bool] = (True, True),
) -> tuple[
    HCCLTwoLiveMemoryPreparationReceipt,
    HCCLTwoLiveMemoryDownstreamAdoptionReceipt,
    HCCLTwoLiveMemoryDownstreamAdoptionReceipt,
]:
    preparation = bridge.integrity_receipt(prepared)
    downstream = tuple(
        bridge.bind_downstream_adoption_receipt(
            prepared,
            agent_index=agent,
            downstream_revision_words=jnp.asarray((0, 41 + agent), dtype=jnp.uint32),
            downstream_content_digest_words=jnp.asarray(
                tuple(range(11 + 8 * agent, 19 + 8 * agent)),
                dtype=jnp.uint32,
            ),
            downstream_candidate_valid=jnp.asarray(valid[agent], dtype=jnp.bool_),
        )
        for agent in range(N_AGENTS)
    )
    return preparation, downstream[0], downstream[1]


def _adopt(
    bridge: HCCLTwoLiveMemoryPrepareAdoptBridge,
    state: HCCLTwoLiveMemoryBridgeState,
    prepared: HCCLTwoLiveMemoryPreparedTransaction,
    receipts: tuple[
        HCCLTwoLiveMemoryPreparationReceipt,
        HCCLTwoLiveMemoryDownstreamAdoptionReceipt,
        HCCLTwoLiveMemoryDownstreamAdoptionReceipt,
    ],
) -> HCCLTwoLiveMemoryPrepareAdoptResult:
    return bridge.adopt_prepared_transaction(
        state,
        prepared,
        receipts[0],
        receipts[1],
        receipts[2],
    )


def _warm_two_events(
    bridge: HCCLTwoLiveMemoryPrepareAdoptBridge,
    state: HCCLTwoLiveMemoryBridgeState,
) -> HCCLTwoLiveMemoryBridgeState:
    """Reach the first event where every causal audit channel is available."""

    current = state
    for _ in range(2):
        prepared = _prepare(bridge, current, sidecars=False)
        adopted = _adopt(bridge, current, prepared, _receipts(bridge, prepared))
        assert bool(adopted.update_applied)
        current = adopted.state
    return current


def _assert_all_false(value: object) -> None:
    assert not bool(jnp.any(jnp.asarray(value, dtype=jnp.bool_)))


def test_prepare_is_transient_forwards_every_sidecar_and_evaluates_each_owner_once() -> None:
    with pytest.raises(ValueError, match="replaceable feature axis"):
        _config(replacement_interval=1)
    bridge, state = _bridge()
    assert type(bridge.inner) is HCCLTwoLiveMemoryBridge
    assert type(state) is HCCLTwoLiveMemoryBridgeState
    assert "prepare_transaction" not in HCCLTwoLiveMemoryBridge.__dict__
    assert core_api.HCCLTwoLiveMemoryPrepareAdoptBridge is HCCLTwoLiveMemoryPrepareAdoptBridge
    assert not any(
        "prepar" in field.name
        for field in dataclasses.fields(HCCLTwoLiveMemoryBridgeState)
    )

    payload = bridge.to_config()
    assert payload["mechanism_status"] == HCCL_TWO_LIVE_MEMORY_PREPARE_ADOPT_STATUS
    assert payload["mechanism_status"] == "l0-development-hccl-two-live-memory-prepare-adopt"
    assert payload["hccl_state_owners"] == 1
    assert payload["live_memory_adapter_state_owners"] == 2
    assert payload["preparation_persisted"] is False
    assert payload["preparation_checkpoint_supported"] is False
    assert payload["planner_layer_authority"] is False
    assert payload["delight_or_actor_backward"] is False

    source_snapshot = jax.tree.map(lambda leaf: leaf.copy(), state)
    prepared = _prepare(bridge, state)
    assert isinstance(prepared, HCCLTwoLiveMemoryPreparedTransaction)
    assert bool(prepared.preparation_valid)
    _tree_exact(prepared.source_state, state)
    # Birth timestamps intentionally use wall time.  A fresh same-key init can
    # cross a float32 timestamp quantum during this slow preparation, so the
    # transience oracle must be the pre-call source itself rather than re-init.
    _tree_exact(state, source_snapshot)
    np.testing.assert_array_equal(
        prepared.next_decision_hard_action_masks,
        jnp.ones((N_AGENTS, N_ACTIONS), dtype=jnp.bool_),
    )
    np.testing.assert_array_equal(
        prepared.agent_0.extended_action_mask,
        jnp.asarray((True, True, True), dtype=jnp.bool_),
    )
    np.testing.assert_array_equal(
        prepared.agent_1.extended_action_mask,
        jnp.asarray((True, True, False), dtype=jnp.bool_),
    )
    for receipt_words in (
        prepared.source_state_receipt_words,
        prepared.event_receipt_words,
        prepared.config_receipt_words,
        prepared.content_tag_words,
    ):
        assert receipt_words.shape == (8,)
        assert receipt_words.dtype == jnp.uint32
        assert bool(jnp.any(receipt_words != 0))

    for agent, expected_context in (
        (prepared.agent_0, 0.25),
        (prepared.agent_1, 0.50),
    ):
        assert isinstance(agent, HCCLTwoLiveMemoryPreparedAgentFacts)
        assert agent.live_prepared.candidate_evidence is None
        supplied_input = cast(
            PrototypePartnerPolicyFusionInput,
            agent.live_prepared.partner_policy_fusion_input,
        )
        supplied_feedback = cast(
            PrototypePartnerPolicyFusionFeedback,
            agent.live_prepared.partner_policy_fusion_feedback,
        )
        np.testing.assert_array_equal(
            supplied_input.context_features,
            jnp.full((TOTAL_DIM,), expected_context, dtype=jnp.float32),
        )
        assert not bool(supplied_feedback.feedback.available)
        coordinator = agent.live_prepared.coordinator_result
        assert coordinator is not None
        nested = coordinator.evaluated.prepared.inner_result.prototype_result
        _tree_exact(agent.prototype_result, nested)
        _tree_exact(agent.raw_stomp_result, nested.oak_stomp_update_result)
        _tree_exact(agent.owner_finalization_trace, nested.oak_owner_finalization_trace)
        _tree_exact(
            agent.raw_stomp_result.state,
            agent.owner_finalization_trace.raw_state,
        )
        _tree_exact(
            agent.owner_finalization_trace.final_state,
            nested.state.oak_state.oak_state.stomp_state,
        )
        np.testing.assert_array_equal(
            agent.live_prepared.hard_action_mask,
            jnp.ones((N_ACTIONS,), dtype=jnp.bool_),
        )
        np.testing.assert_array_equal(
            agent.live_prepared.extended_action_mask,
            agent.extended_action_mask,
        )
        assert not bool(agent.live_prepared.feedback_supplied)

    np.testing.assert_array_equal(prepared.work.live_prepare_calls, (1, 1))
    np.testing.assert_array_equal(prepared.work.coordinator_update_calls, (1, 1))
    np.testing.assert_array_equal(prepared.work.prototype_update_calls, (1, 1))
    np.testing.assert_array_equal(prepared.work.real_stomp_update_evaluations, (1, 1))
    np.testing.assert_array_equal(
        prepared.work.total_stomp_update_evaluations,
        jnp.stack(
            (
                prepared.agent_0.prototype_result.oak_total_stomp_update_evaluations,
                prepared.agent_1.prototype_result.oak_total_stomp_update_evaluations,
            )
        ),
    )
    np.testing.assert_array_equal(prepared.work.feedback_settlement_calls, (0, 0))
    np.testing.assert_array_equal(prepared.work.memory_query_calls, (1, 1))
    np.testing.assert_array_equal(prepared.work.memory_write_calls, (1, 1))
    assert int(prepared.work.hccl_stage_calls) == 1
    assert int(prepared.work.world_proposal_calls) == 8
    assert int(prepared.work.attribution_proposal_calls) == 8
    assert bool(prepared.planner_equals_memory)
    assert bool(prepared.pp_executes_memory_actions)
    assert bool(prepared.no_planner_rung_valid)
    assert not bool(prepared.delight_or_actor_backward)

    budget = bridge.resource_budget(state)
    assert budget.persisted_preparation_records == 0
    assert budget.persisted_preparation_bytes == 0
    assert budget.prepared_checkpoint_supported is False
    assert budget.prepare_hccl_stage_calls_per_transaction == 1
    assert budget.prepare_live_adapter_calls_per_transaction == 2
    assert budget.adopt_world_or_learner_reevaluations == 0


def test_valid_candidate_evidence_can_enable_builder_and_adopt_does_no_donor_work() -> None:
    bridge, state = _bridge()
    state = _warm_two_events(bridge, state)
    assert bool(state.agent_0_state.pending_binding.available)
    assert bool(state.agent_1_state.pending_binding.available)
    missing = _prepare(bridge, state, sidecars=False)
    np.testing.assert_array_equal(missing.prior_feedback_required, (True, True))
    np.testing.assert_array_equal(missing.prior_feedback_supplied, (True, True))
    for index, agent in enumerate((missing.agent_0, missing.agent_1)):
        assert bool(agent.live_prepared.feedback_supplied)
        np.testing.assert_array_equal(
            agent.live_prepared.feedback.counterfactual_delta,
            missing.agent_unilateral_counterfactual_delta[index],
        )
        coordinator = agent.live_prepared.coordinator_result
        assert coordinator is not None
        assert bool(coordinator.diagnostics.builder_learning_vetoed)
        assert not bool(coordinator.diagnostics.builder_learning_applied)

    evidence = (_audit_evidence(missing.agent_0), _audit_evidence(missing.agent_1))
    prepared = _prepare(bridge, state, evidence=evidence, sidecars=False)
    for index, agent in enumerate((prepared.agent_0, prepared.agent_1)):
        _tree_exact(agent.live_prepared.candidate_evidence, evidence[index])
        coordinator = agent.live_prepared.coordinator_result
        assert coordinator is not None
        assert bool(coordinator.diagnostics.candidate_evidence_supplied)
        assert bool(coordinator.diagnostics.candidate_evidence_identity_valid)
        assert bool(coordinator.diagnostics.candidate_audit_accepted)
        assert bool(coordinator.diagnostics.builder_learning_applied)
        assert not bool(coordinator.diagnostics.builder_learning_vetoed)

    receipts = _receipts(bridge, prepared)
    accepted = _adopt(bridge, state, prepared, receipts)
    assert bool(accepted.update_applied)
    _tree_exact(accepted.state, prepared.candidate_state)
    assert bool(accepted.source_state_receipt_valid)
    assert bool(accepted.event_receipt_valid)
    assert bool(accepted.config_receipt_valid)
    assert bool(accepted.preparation_receipt_valid)
    np.testing.assert_array_equal(accepted.downstream_receipts_valid, (True, True))
    assert bool(accepted.hccl_result.update_applied)
    assert bool(accepted.agent_0_result.diagnostics.transaction_applied)
    assert bool(accepted.agent_1_result.diagnostics.transaction_applied)
    np.testing.assert_array_equal(accepted.live_adapter_updates_applied, (True, True))
    np.testing.assert_array_equal(accepted.coordinator_updates_applied, (True, True))
    np.testing.assert_array_equal(accepted.prototype_updates_applied, (True, True))
    np.testing.assert_array_equal(accepted.stomp_updates_applied, (True, True))
    np.testing.assert_array_equal(accepted.learned_memory_updates_applied, (True, True))
    np.testing.assert_array_equal(accepted.builder_learning_applied, (True, True))
    assert bool(accepted.next_decision_masks_installed)
    assert not bool(accepted.delight_or_actor_backward)

    assert int(accepted.adoption_work.preparation_integrity_checks) == 1
    assert int(accepted.adoption_work.downstream_receipt_integrity_checks) == 2
    np.testing.assert_array_equal(
        accepted.adoption_work.live_integrity_adoption_calls,
        (1, 1),
    )
    assert int(accepted.adoption_work.world_proposal_calls) == 0
    assert int(accepted.adoption_work.attribution_proposal_calls) == 0
    _assert_all_false(accepted.adoption_work.coordinator_update_calls)
    _assert_all_false(accepted.adoption_work.prototype_update_calls)
    _assert_all_false(accepted.adoption_work.stomp_update_evaluations)
    _assert_all_false(accepted.adoption_work.memory_query_calls)
    _assert_all_false(accepted.adoption_work.memory_write_calls)
    _tree_exact(accepted.prepared.work, prepared.work)


def test_one_downstream_veto_rolls_every_owner_back_and_outer_gates_child_flags() -> None:
    bridge, state = _bridge()
    prepared = _prepare(bridge, state, sidecars=False)
    assert bool(prepared.attempted_hccl_result.update_applied)
    for agent in (prepared.agent_0, prepared.agent_1):
        coordinator = agent.live_prepared.coordinator_result
        assert coordinator is not None
        assert bool(coordinator.diagnostics.transaction_applied)
        assert bool(agent.raw_stomp_result.update_applied)

    rejected = _adopt(
        bridge,
        state,
        prepared,
        _receipts(bridge, prepared, valid=(True, False)),
    )
    assert not bool(rejected.update_applied)
    _tree_exact(rejected.state, state)
    np.testing.assert_array_equal(rejected.downstream_candidates_valid, (True, False))
    assert not bool(rejected.hccl_result.update_applied)
    assert not bool(rejected.agent_0_result.diagnostics.transaction_applied)
    assert not bool(rejected.agent_1_result.diagnostics.transaction_applied)
    for public_flags in (
        rejected.live_adapter_updates_applied,
        rejected.coordinator_updates_applied,
        rejected.prototype_updates_applied,
        rejected.stomp_updates_applied,
        rejected.learned_memory_updates_applied,
        rejected.builder_learning_applied,
    ):
        _assert_all_false(public_flags)
    assert not bool(rejected.next_decision_masks_installed)
    assert int(rejected.prepared.work.hccl_stage_calls) == 1
    np.testing.assert_array_equal(rejected.prepared.work.live_prepare_calls, (1, 1))
    _tree_exact(rejected.prepared.agent_0.raw_stomp_result, prepared.agent_0.raw_stomp_result)
    _tree_exact(rejected.prepared.agent_1.raw_stomp_result, prepared.agent_1.raw_stomp_result)
    _assert_all_false(rejected.adoption_work.coordinator_update_calls)
    _assert_all_false(rejected.adoption_work.prototype_update_calls)
    _assert_all_false(rejected.adoption_work.stomp_update_evaluations)
    _assert_all_false(rejected.adoption_work.memory_query_calls)
    _assert_all_false(rejected.adoption_work.memory_write_calls)


def test_receipts_reject_tamper_replay_cross_agent_and_foreign_configuration() -> None:
    bridge, state = _bridge()
    prepared = _prepare(bridge, state, sidecars=False)
    preparation, receipt_0, receipt_1 = _receipts(bridge, prepared)
    for receipt, agent in (
        (receipt_0, prepared.agent_0),
        (receipt_1, prepared.agent_1),
    ):
        np.testing.assert_array_equal(
            receipt.raw_stomp_digest,
            agent.owner_finalization_trace.raw_digest,
        )
        np.testing.assert_array_equal(
            receipt.final_stomp_digest,
            agent.owner_finalization_trace.final_digest,
        )
        np.testing.assert_array_equal(
            receipt.owner_finalization_trace_checksum,
            agent.owner_finalization_trace.trace_checksum,
        )
        np.testing.assert_array_equal(
            receipt.extended_action_mask,
            agent.extended_action_mask,
        )

    tampered_preparation = cast(
        HCCLTwoLiveMemoryPreparationReceipt,
        preparation.replace(
            config_receipt_words=preparation.config_receipt_words.at[0].add(
                jnp.asarray(1, dtype=jnp.uint32)
            )
        ),
    )
    rejected_preparation = _adopt(
        bridge,
        state,
        prepared,
        (tampered_preparation, receipt_0, receipt_1),
    )
    assert not bool(rejected_preparation.config_receipt_valid)
    assert not bool(rejected_preparation.update_applied)
    _tree_exact(rejected_preparation.state, state)

    cross_agent = _adopt(
        bridge,
        state,
        prepared,
        (preparation, receipt_1, receipt_0),
    )
    np.testing.assert_array_equal(cross_agent.downstream_receipts_valid, (False, False))
    assert not bool(cross_agent.update_applied)
    _tree_exact(cross_agent.state, state)

    tampered_finalization = cast(
        HCCLTwoLiveMemoryDownstreamAdoptionReceipt,
        receipt_0.replace(
            owner_finalization_trace_checksum=(
                receipt_0.owner_finalization_trace_checksum.at[0].add(
                    jnp.asarray(1, dtype=jnp.uint32)
                )
            )
        ),
    )
    tampered_mask = cast(
        HCCLTwoLiveMemoryDownstreamAdoptionReceipt,
        receipt_1.replace(
            extended_action_mask=receipt_1.extended_action_mask.at[2].set(True)
        ),
    )
    rejected_nested = _adopt(
        bridge,
        state,
        prepared,
        (preparation, tampered_finalization, tampered_mask),
    )
    np.testing.assert_array_equal(rejected_nested.downstream_receipts_valid, (False, False))
    assert not bool(rejected_nested.update_applied)
    _tree_exact(rejected_nested.state, state)

    accepted = _adopt(
        bridge,
        state,
        prepared,
        (preparation, receipt_0, receipt_1),
    )
    assert bool(accepted.update_applied)
    replay = _adopt(
        bridge,
        accepted.state,
        prepared,
        (preparation, receipt_0, receipt_1),
    )
    assert not bool(replay.source_state_receipt_valid)
    assert not bool(replay.update_applied)
    _tree_exact(replay.state, accepted.state)

    foreign_config = dataclasses.replace(
        _config(),
        binding_owner_digest=(
            0xDEADBEEF,
            0xA5A5A5A5,
            0x13572468,
            0x24681357,
            0x10293847,
            0x56473829,
            0xABCDEF01,
            0x10FEDCBA,
        ),
    )
    foreign = HCCLTwoLiveMemoryPrepareAdoptBridge(foreign_config)
    rejected_foreign = _adopt(
        foreign,
        state,
        prepared,
        (preparation, receipt_0, receipt_1),
    )
    assert not bool(rejected_foreign.config_receipt_valid)
    assert not bool(rejected_foreign.update_applied)
    _tree_exact(rejected_foreign.state, state)
