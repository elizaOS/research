# mypy: disable-error-code="arg-type,attr-defined,call-arg,no-any-return,type-var"
"""Red-first contracts for transient factorized P over two live M owners.

The persistent composite owns one v1 HCCL/two-live-memory state and one paired
factorized planner state.  P Prototype views are reconstructed for the current
event and discarded; they are never additional persistent Prototype owners.
Delight is unavailable because this path executes no Kondo actor backward.
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
from alberta_framework.core.hccl_two_live_memory_factorized_planner_bridge import (
    HCCL_TWO_LIVE_MEMORY_FACTORIZED_PLANNER_STATUS,
    HCCLTwoLiveMemoryFactorizedPlannerBridge,
    HCCLTwoLiveMemoryFactorizedPlannerConfig,
    HCCLTwoLiveMemoryFactorizedPlannerState,
)
from alberta_framework.core.prototype_factorized_partner_planner import (
    FactorizedPartnerPlannerAgentState,
    PrototypeFactorizedPartnerPlannerConfig,
)

pytestmark = [pytest.mark.integration, pytest.mark.slow]

N_AGENTS = 2
PHYSICAL_RAW_DIM = 16
PP_SLOT = 4


@pytest.fixture(autouse=True)
def _bounded_jax_execution() -> object:
    with jax.disable_jit():
        yield


def _config() -> HCCLTwoLiveMemoryFactorizedPlannerConfig:
    inner = _v1_config()
    prototype = inner.agent_0.coordinator.inner.prototype
    assert prototype.state_builder is not None
    constructed_state_dim = prototype.state_builder.observation_dim
    representation_dim = inner.agent_0.coordinator.inner.prototype.oak.observation_dim
    return HCCLTwoLiveMemoryFactorizedPlannerConfig(
        inner=inner,
        planner=PrototypeFactorizedPartnerPlannerConfig(
            observation_dim=constructed_state_dim,
            prototype_representation_dim=representation_dim,
            n_actions=N_ACTIONS,
            planning_enabled=True,
        ),
    )


def _bridge() -> tuple[
    HCCLTwoLiveMemoryFactorizedPlannerBridge,
    HCCLTwoLiveMemoryFactorizedPlannerState,
]:
    bridge = HCCLTwoLiveMemoryFactorizedPlannerBridge(_config())
    return bridge, bridge.init(jr.key(811))


def _prototype_states(state: HCCLTwoLiveMemoryFactorizedPlannerState) -> tuple[Any, Any]:
    inner = state.inner_state
    return (
        inner.agent_0_state.coordinator_state.inner_state.prototype_state,
        inner.agent_1_state.coordinator_state.inner_state.prototype_state,
    )


def _force_planner_away_from_memory(
    bridge: HCCLTwoLiveMemoryFactorizedPlannerBridge,
    state: HCCLTwoLiveMemoryFactorizedPlannerState,
) -> HCCLTwoLiveMemoryFactorizedPlannerState:
    """Install finite grounded values, then rebuild only the paired P cache."""

    prototypes = _prototype_states(state)
    desired = tuple(1 - int(item.current_action) for item in prototypes)
    reward_index = bridge.config.planner.observation_dim

    def force(
        agent: FactorizedPartnerPlannerAgentState,
        action: int,
    ) -> FactorizedPartnerPlannerAgentState:
        rewards = jnp.zeros((N_ACTIONS, N_ACTIONS), dtype=jnp.float32)
        rewards = rewards.at[action, :].set(jnp.asarray(5.0, dtype=jnp.float32))
        grounded = agent.grounded.replace(
            weights=jnp.zeros_like(agent.grounded.weights),
            bias=agent.grounded.bias.at[:, reward_index].set(rewards.reshape((-1,))),
        )
        return agent.replace(grounded=grounded)

    stale = state.planner_state.replace(
        agent_0=force(state.planner_state.agent_0, desired[0]),
        agent_1=force(state.planner_state.agent_1, desired[1]),
    )
    prepared = bridge.planner.prepare_pair(
        stale,
        prototypes[0],
        prototypes[1],
        state.inner_state.current_hard_action_masks,
    )
    assert bool(prepared.diagnostics.pair_committed)
    candidate = cast(
        HCCLTwoLiveMemoryFactorizedPlannerState,
        state.replace(planner_state=prepared.state),
    )
    assert bool(bridge.state_valid(candidate))
    return candidate


def _prepare(
    bridge: HCCLTwoLiveMemoryFactorizedPlannerBridge,
    state: HCCLTwoLiveMemoryFactorizedPlannerState,
) -> Any:
    event = bridge.prepare_event(state)
    return bridge.prepare_transaction(
        state,
        event,
        _event_input(410),
        _event_input(411),
        next_decision_hard_action_masks=jnp.ones(
            (N_AGENTS, N_ACTIONS), dtype=jnp.bool_
        ),
    )


def _receipts(bridge: Any, prepared: Any, *, valid: tuple[bool, bool] = (True, True)) -> Any:
    integrity = bridge.integrity_receipt(prepared)
    downstream = tuple(
        bridge.bind_downstream_adoption_receipt(
            prepared,
            agent_index=index,
            downstream_revision_words=jnp.asarray((0, index + 1), dtype=jnp.uint32),
            downstream_content_digest_words=jnp.arange(
                8, dtype=jnp.uint32
            )
            + jnp.asarray(100 + 10 * index, dtype=jnp.uint32),
            downstream_candidate_valid=jnp.asarray(valid[index], dtype=jnp.bool_),
        )
        for index in range(N_AGENTS)
    )
    return integrity, downstream[0], downstream[1]


def test_topology_is_snapshot_free_and_reconstruction_authenticates_distinct_p() -> None:
    bridge, state = _bridge()
    state = _force_planner_away_from_memory(bridge, state)
    payload = bridge.to_config()
    budget = bridge.resource_budget(state)

    assert payload["mechanism_status"] == HCCL_TWO_LIVE_MEMORY_FACTORIZED_PLANNER_STATUS
    assert payload["prototype_state_owners"] == 2
    assert payload["additional_prototype_state_owners"] == 0
    assert payload["persisted_planner_prototype_snapshots"] == 0
    assert payload["grounded_state_semantics"] == (
        "external-GRU-builder-base-constructed-state"
    )
    assert payload["physical_raw_grounding"] is False
    assert payload["physical_hccl_plant_observation_dim"] == PHYSICAL_RAW_DIM
    assert payload["grounded_constructed_state_dim"] == 17
    assert bridge.config.planner.observation_dim == 17
    assert (
        bridge.config.inner.agent_0.coordinator.builder.observation_dim
        == PHYSICAL_RAW_DIM
    )
    assert payload["modeled_planner_proposal_available"] is True
    assert payload["bounded_hccl_world_planner_action_consumed"] is True
    assert payload["pp_candidate_is_committed_world_successor"] is True
    assert payload["external_environment_dispatch_authority"] is False
    assert payload["physical_dispatch_authority"] is False
    assert payload["safety_authority"] is False
    assert payload["evidence_authority"] is False
    assert payload["promotion_authority"] is False
    assert payload["delight_or_actor_backward"] is False
    assert payload["actor_backward_calls_per_transaction"] == 0
    assert payload["delight_interpretation"] == "unavailable-no-Kondo-actor-backward"
    assert {field.name for field in dataclasses.fields(cast(Any, state))} == {
        "inner_state",
        "planner_state",
    }
    assert budget.prototype_state_owners == 2
    assert budget.persisted_planner_prototype_snapshots == 0

    views = bridge.reconstruct_planner_dispatch(state)
    memory_actions = jnp.stack(tuple(item.memory_prototype.current_action for item in views))
    planner_actions = jnp.stack(tuple(item.planner_prototype.current_action for item in views))
    np.testing.assert_array_equal(np.asarray(planner_actions), 1 - np.asarray(memory_actions))
    for view in views:
        assert bool(view.cache_base_matches_memory)
        assert bool(view.replacement.committed)
        assert bool(view.cache_effective_matches_replacement)
        assert bool(view.mask_relation_valid)
        assert bool(view.planner_cache_authenticated)
        assert bool(view.valid)

    assert core_api.HCCLTwoLiveMemoryFactorizedPlannerBridge is (
        HCCLTwoLiveMemoryFactorizedPlannerBridge
    )


def test_prepare_uses_one_hccl_two_live_updates_and_one_paired_completion() -> None:
    bridge, state = _bridge()
    state = _force_planner_away_from_memory(bridge, state)
    prepared = _prepare(bridge, state)
    binding = prepared.binding
    pp = jax.tree.map(
        lambda leaf: leaf[PP_SLOT],
        prepared.attempted_hccl_result.world_proposals,
    )

    np.testing.assert_array_equal(
        np.asarray(binding.planner.actions_before_mask),
        np.asarray(binding.planner_proposed_actions),
    )
    np.testing.assert_array_equal(
        np.asarray(binding.planner.actions_after_mask),
        np.asarray(binding.planner_actions),
    )
    assert bool(jnp.any(binding.planner_actions != binding.memory_actions))
    np.testing.assert_array_equal(
        np.asarray(pp.joint_action_ids), np.asarray(binding.planner_actions)
    )
    np.testing.assert_array_equal(
        np.asarray(prepared.effective_planner_actions),
        np.asarray(binding.planner_actions),
    )
    np.testing.assert_array_equal(np.asarray(prepared.work.hccl_stage_calls), 1)
    np.testing.assert_array_equal(np.asarray(prepared.work.world_proposal_calls), 8)
    np.testing.assert_array_equal(np.asarray(prepared.work.attribution_proposal_calls), 8)
    np.testing.assert_array_equal(np.asarray(prepared.work.live_prepare_calls), (1, 1))
    np.testing.assert_array_equal(
        np.asarray(prepared.work.live_internal_feedback_settlement_calls), (0, 0)
    )
    np.testing.assert_array_equal(
        np.asarray(prepared.work.planner_reconstruction_replacements), (1, 1)
    )
    np.testing.assert_array_equal(
        np.asarray(prepared.work.factorized_completed_transition_calls), 1
    )
    np.testing.assert_array_equal(
        np.asarray(prepared.work.real_stomp_update_evaluations), (1, 1)
    )
    np.testing.assert_array_equal(np.asarray(prepared.work.memory_query_calls), (1, 1))
    np.testing.assert_array_equal(np.asarray(prepared.work.memory_write_calls), (1, 1))
    assert bool(prepared.planner_result.diagnostics.transaction_committed)
    assert bool(prepared.preparation_valid)
    assert not bool(prepared.delight_or_actor_backward)
    np.testing.assert_array_equal(np.asarray(prepared.work.actor_backward_calls), 0)

    next_memory = _prototype_states(prepared.candidate_state)
    np.testing.assert_array_equal(
        np.asarray(prepared.planner_result.diagnostics.next_prepare.base_actions),
        np.asarray(tuple(item.current_action for item in next_memory)),
    )
    assert prepared.physical_hccl_next_observations.shape == (
        N_AGENTS,
        PHYSICAL_RAW_DIM,
    )
    assert prepared.grounded_next_constructed_states.shape == (N_AGENTS, 17)
    np.testing.assert_array_equal(
        np.asarray(prepared.physical_hccl_next_observations),
        np.asarray(pp.next_observation),
    )
    np.testing.assert_array_equal(
        np.asarray(prepared.grounded_next_constructed_states),
        np.asarray(tuple(item.current_raw_observation for item in next_memory)),
    )
    np.testing.assert_array_equal(
        np.asarray(
            prepared.planner_result.diagnostics.grounded_targets[:, :17]
        ),
        np.asarray(prepared.grounded_next_constructed_states),
    )
    assert prepared.candidate_state.inner_state.agent_0_state is (
        prepared.agent_0.live_prepared.candidate_state
    )
    assert prepared.candidate_state.inner_state.agent_1_state is (
        prepared.agent_1.live_prepared.candidate_state
    )


def test_hard_mask_replaces_raw_planner_proposal_and_pp_uses_effective_p() -> None:
    bridge, state = _bridge()
    state = _force_planner_away_from_memory(bridge, state)
    prototypes = _prototype_states(state)
    memory_actions = jnp.stack(tuple(item.current_action for item in prototypes)).astype(
        jnp.int32
    )
    masks = jax.nn.one_hot(memory_actions, N_ACTIONS, dtype=jnp.bool_)
    masked_inner = cast(
        Any,
        state.inner_state.replace(current_hard_action_masks=masks),
    )
    refreshed = bridge.planner.prepare_pair(
        state.planner_state,
        prototypes[0],
        prototypes[1],
        masks,
    )
    assert bool(refreshed.diagnostics.pair_committed)
    masked = cast(
        HCCLTwoLiveMemoryFactorizedPlannerState,
        state.replace(inner_state=masked_inner, planner_state=refreshed.state),
    )
    assert bool(bridge.state_valid(masked))

    prepared = _prepare(bridge, masked)
    binding = prepared.binding
    pp = jax.tree.map(
        lambda leaf: leaf[PP_SLOT],
        prepared.attempted_hccl_result.world_proposals,
    )
    assert bool(jnp.all(binding.planner_proposed_actions != memory_actions))
    np.testing.assert_array_equal(
        np.asarray(binding.planner_actions), np.asarray(memory_actions)
    )
    assert bool(
        jnp.all(
            ~binding.current_hard_action_masks[
                jnp.arange(N_AGENTS), binding.planner_proposed_actions
            ]
        )
    )
    np.testing.assert_array_equal(
        np.asarray(pp.joint_action_ids), np.asarray(binding.planner_actions)
    )
    assert bool(prepared.pp_executes_planner_actions)
    assert bool(prepared.preparation_valid)


def test_adoption_is_atomic_content_bound_and_does_no_donor_reevaluation() -> None:
    bridge, state = _bridge()
    state = _force_planner_away_from_memory(bridge, state)
    prepared = _prepare(bridge, state)
    integrity, downstream_0, downstream_1 = _receipts(bridge, prepared)
    result = bridge.adopt_prepared_transaction(
        state,
        prepared,
        integrity,
        downstream_0,
        downstream_1,
    )

    assert bool(result.update_applied)
    _tree_exact(result.state, prepared.candidate_state)
    np.testing.assert_array_equal(np.asarray(result.live_adapter_updates_applied), (True, True))
    assert bool(result.hccl_update_applied)
    assert bool(result.factorized_planner_update_applied)
    assert bool(result.next_decision_masks_installed)
    np.testing.assert_array_equal(np.asarray(result.adoption_work.live_integrity_calls), (1, 1))
    np.testing.assert_array_equal(np.asarray(result.adoption_work.world_proposal_calls), 0)
    np.testing.assert_array_equal(np.asarray(result.adoption_work.attribution_proposal_calls), 0)
    np.testing.assert_array_equal(np.asarray(result.adoption_work.prototype_update_calls), (0, 0))
    np.testing.assert_array_equal(np.asarray(result.adoption_work.stomp_update_evaluations), (0, 0))
    np.testing.assert_array_equal(np.asarray(result.adoption_work.memory_query_calls), (0, 0))
    np.testing.assert_array_equal(np.asarray(result.adoption_work.memory_write_calls), (0, 0))
    np.testing.assert_array_equal(
        np.asarray(result.adoption_work.planner_reconstruction_replacements),
        (2, 2),
    )
    np.testing.assert_array_equal(
        np.asarray(result.adoption_work.planner_cache_authentication_evaluations),
        (2, 2),
    )
    assert bool(result.modeled_planner_proposal_available)
    assert bool(result.bounded_hccl_world_planner_action_consumed)
    assert bool(result.pp_candidate_committed)
    assert not bool(result.external_environment_dispatch_authority)
    assert not bool(result.physical_dispatch_authority)
    assert not bool(result.safety_authority)
    assert not bool(result.evidence_authority)
    assert not bool(result.promotion_authority)
    assert not bool(result.delight_or_actor_backward)
    np.testing.assert_array_equal(np.asarray(result.actor_backward_calls), 0)
    assert bool(bridge.state_valid(result.state))


def test_two_consecutive_events_settle_prior_m_once_and_reconstruct_next_p() -> None:
    bridge, state = _bridge()
    state = _force_planner_away_from_memory(bridge, state)

    # Warm the query-before-write memory so observed event 1 can create an
    # admitted retrieval and therefore a real prior-M feedback obligation.
    warm = _prepare(bridge, state)
    warm_receipts = _receipts(bridge, warm)
    warmed = bridge.adopt_prepared_transaction(
        state,
        warm,
        warm_receipts[0],
        warm_receipts[1],
        warm_receipts[2],
    )
    assert bool(warmed.update_applied)

    event_1 = _prepare(bridge, warmed.state)
    event_1_receipts = _receipts(bridge, event_1)
    adopted_1 = bridge.adopt_prepared_transaction(
        warmed.state,
        event_1,
        event_1_receipts[0],
        event_1_receipts[1],
        event_1_receipts[2],
    )
    assert bool(adopted_1.update_applied)
    pending = (
        adopted_1.state.inner_state.agent_0_state.pending_binding,
        adopted_1.state.inner_state.agent_1_state.pending_binding,
    )
    assert all(bool(item.available) for item in pending)

    next_p = bridge.reconstruct_planner_dispatch(adopted_1.state)
    next_p_actions = jnp.stack(tuple(item.effective_action for item in next_p)).astype(
        jnp.int32
    )
    event_2 = _prepare(bridge, adopted_1.state)
    np.testing.assert_array_equal(
        np.asarray(event_2.effective_planner_actions), np.asarray(next_p_actions)
    )
    np.testing.assert_array_equal(
        np.asarray(event_2.work.feedback_settlement_calls), (1, 1)
    )
    np.testing.assert_array_equal(
        np.asarray(event_2.work.live_internal_feedback_settlement_calls), (0, 0)
    )
    np.testing.assert_array_equal(
        np.asarray(event_2.prior_feedback_required), (True, True)
    )
    np.testing.assert_array_equal(
        np.asarray(event_2.prior_feedback_supplied), (True, True)
    )

    for index, facts in enumerate((event_2.agent_0, event_2.agent_1)):
        memory_child = (
            adopted_1.state.inner_state.agent_0_state
            if index == 0
            else adopted_1.state.inner_state.agent_1_state
        )
        np.testing.assert_array_equal(
            np.asarray(facts.transition.action), np.asarray(next_p_actions[index])
        )
        expected_one_hot = jax.nn.one_hot(
            next_p_actions[index], N_ACTIONS, dtype=jnp.float32
        )
        np.testing.assert_array_equal(
            np.asarray(facts.live_prepared.completed_entry.action),
            np.asarray(expected_one_hot),
        )
        feedback = facts.settled_dispatch.feedback
        np.testing.assert_array_equal(
            np.asarray(feedback.memory_transaction_words),
            np.asarray(memory_child.pending_binding.memory_transaction_words),
        )
        np.testing.assert_array_equal(
            np.asarray(feedback.prototype_decision_id),
            np.asarray(memory_child.pending_binding.prototype_decision_id),
        )
        np.testing.assert_array_equal(
            np.asarray(feedback.effective_action),
            np.asarray(memory_child.coordinator_state.current_action),
        )
        assert bool(facts.settled_dispatch.feedback_identity_valid)
        assert bool(facts.settled_dispatch.settlement_valid)

    next_memory = _prototype_states(event_2.candidate_state)
    np.testing.assert_array_equal(
        np.asarray(event_2.planner_result.diagnostics.next_prepare.base_actions),
        np.asarray(tuple(item.current_action for item in next_memory)),
    )
    assert {field.name for field in dataclasses.fields(cast(Any, event_2.candidate_state))} == {
        "inner_state",
        "planner_state",
    }
    event_2_receipts = _receipts(bridge, event_2)
    adopted_2 = bridge.adopt_prepared_transaction(
        adopted_1.state,
        event_2,
        event_2_receipts[0],
        event_2_receipts[1],
        event_2_receipts[2],
    )
    assert bool(adopted_2.update_applied)
    assert bool(bridge.state_valid(adopted_2.state))


def test_tamper_replay_or_one_downstream_refusal_returns_both_m_owners() -> None:
    bridge, state = _bridge()
    state = _force_planner_away_from_memory(bridge, state)
    prepared = _prepare(bridge, state)
    integrity, downstream_0, downstream_1 = _receipts(
        bridge, prepared, valid=(True, False)
    )
    rejected = bridge.adopt_prepared_transaction(
        state,
        prepared,
        integrity,
        downstream_0,
        downstream_1,
    )
    assert not bool(rejected.update_applied)
    _tree_exact(rejected.state, state)
    _tree_exact(rejected.agent_0_result.state, state.inner_state.agent_0_state)
    _tree_exact(rejected.agent_1_result.state, state.inner_state.agent_1_state)
    np.testing.assert_array_equal(
        np.asarray(rejected.live_adapter_updates_applied), (False, False)
    )

    tampered = prepared.replace(
        effective_planner_actions=prepared.effective_planner_actions.at[0].set(
            1 - prepared.effective_planner_actions[0]
        )
    )
    tamper_rejected = bridge.adopt_prepared_transaction(
        state,
        tampered,
        integrity,
        downstream_0.replace(downstream_candidate_valid=jnp.asarray(True)),
        downstream_1.replace(downstream_candidate_valid=jnp.asarray(True)),
    )
    assert not bool(tamper_rejected.update_applied)
    _tree_exact(tamper_rejected.state, state)

    accepted_integrity, accepted_0, accepted_1 = _receipts(bridge, prepared)
    accepted = bridge.adopt_prepared_transaction(
        state, prepared, accepted_integrity, accepted_0, accepted_1
    )
    assert bool(accepted.update_applied)
    replay = bridge.adopt_prepared_transaction(
        accepted.state, prepared, accepted_integrity, accepted_0, accepted_1
    )
    assert not bool(replay.update_applied)
    _tree_exact(replay.state, accepted.state)


def test_stale_candidate_planner_cache_defeats_resealed_structural_receipts() -> None:
    bridge, state = _bridge()
    state = _force_planner_away_from_memory(bridge, state)
    prepared = _prepare(bridge, state)

    planner_state = prepared.candidate_state.planner_state
    changed_grounded = planner_state.agent_0.grounded.replace(
        bias=planner_state.agent_0.grounded.bias.at[0, 0].add(
            jnp.asarray(0.25, dtype=jnp.float32)
        )
    )
    stale_cache_planner = planner_state.replace(
        agent_0=planner_state.agent_0.replace(grounded=changed_grounded)
    )
    tampered_candidate = prepared.candidate_state.replace(
        planner_state=stale_cache_planner
    )
    tampered_planner_result = prepared.planner_result.replace(
        state=stale_cache_planner
    )
    assert bool(bridge._structure_valid(tampered_candidate))
    assert not bool(bridge.state_valid(tampered_candidate))

    bare = prepared.replace(
        candidate_state=tampered_candidate,
        planner_result=tampered_planner_result,
        content_tag_words=jnp.zeros_like(prepared.content_tag_words),
    )
    forged = bare.replace(content_tag_words=bridge._prepared_content_tag(bare))
    integrity, downstream_0, downstream_1 = _receipts(bridge, forged)
    rejected = bridge.adopt_prepared_transaction(
        state,
        forged,
        integrity,
        downstream_0,
        downstream_1,
    )

    assert bool(rejected.preparation_receipt_valid)
    np.testing.assert_array_equal(
        np.asarray(rejected.downstream_receipts_valid), (True, True)
    )
    assert bool(rejected.candidate_state_valid)
    assert bool(rejected.source_state_authenticated)
    assert not bool(rejected.final_candidate_authenticated)
    assert not bool(rejected.update_applied)
    _tree_exact(rejected.state, state)
    _tree_exact(rejected.agent_0_result.state, state.inner_state.agent_0_state)
    _tree_exact(rejected.agent_1_result.state, state.inner_state.agent_1_state)
    np.testing.assert_array_equal(
        np.asarray(rejected.live_adapter_updates_applied), (False, False)
    )
