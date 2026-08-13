# mypy: disable-error-code="attr-defined,call-arg,no-any-return,arg-type,type-var,union-attr"
"""First-class recurring contract for actor-owned P over the HCCL dyad."""

from __future__ import annotations

import dataclasses
import inspect
from collections.abc import Iterator

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from test_external_learned_state_live_memory_adapter import _tree_exact
from test_hccl_continual_dyad_transaction import BASE_DIM, MASKS, _dyad

from alberta_framework.core.hccl_kondo_continual_dyad_route import (
    HCCL_KONDO_CONTINUAL_DYAD_ROUTE_SCHEMA,
    HCCL_KONDO_CONTINUAL_DYAD_ROUTE_STATUS,
    HCCLKondoContinualDyadEventResult,
    HCCLKondoContinualDyadRoute,
    HCCLKondoContinualDyadRouteConfig,
    HCCLKondoContinualDyadState,
)
from alberta_framework.core.kondo_executed_action_lineage_bridge import (
    KondoExecutedActionLineageBridgeConfig,
)
from alberta_framework.core.kondo_gate import KondoGateConfig
from alberta_framework.core.kondo_protected_td import KondoProtectedTDConfig
from alberta_framework.core.kondo_sparse_actor import (
    KondoActorParameters,
    KondoSparseActorConfig,
)

pytestmark = [pytest.mark.integration, pytest.mark.slow]

N_AGENTS = 2
N_ACTIONS = 2


@dataclasses.dataclass(frozen=True, slots=True)
class _RouteRig:
    route: HCCLKondoContinualDyadRoute
    genesis: HCCLKondoContinualDyadState


@dataclasses.dataclass(frozen=True, slots=True)
class _LifeRig:
    route_rig: _RouteRig
    event_0: HCCLKondoContinualDyadEventResult
    event_1: HCCLKondoContinualDyadEventResult


@pytest.fixture(autouse=True)
def _bounded_jax_execution() -> Iterator[None]:
    with jax.disable_jit():
        yield


def _actor_config(*, max_screenings: int) -> KondoSparseActorConfig:
    return KondoSparseActorConfig(
        feature_dim=BASE_DIM,
        hidden_dim=4,
        action_count=N_ACTIONS,
        critic_dim=BASE_DIM,
        safety_dim=BASE_DIM,
        learning_rate=0.01,
        gate=KondoGateConfig(
            batch_size=N_AGENTS,
            mode="top_k_rate",
            target_rate=1.0,
            price=0.0,
            temperature=0.1,
            max_screenings=max_screenings,
        ),
    )


def _parameters() -> KondoActorParameters:
    return KondoActorParameters(
        hidden_weight=jnp.linspace(
            -0.18,
            0.22,
            BASE_DIM * 4,
            dtype=jnp.float32,
        ).reshape((BASE_DIM, 4)),
        hidden_bias=jnp.asarray((0.03, -0.02, 0.01, 0.04), dtype=jnp.float32),
        output_weight=jnp.asarray(
            (
                (0.35, -0.20),
                (-0.15, 0.30),
                (0.25, 0.10),
                (-0.10, 0.20),
            ),
            dtype=jnp.float32,
        ),
        output_bias=jnp.asarray((0.04, -0.03), dtype=jnp.float32),
    )


def _keys(seed: int) -> jax.Array:
    return jr.split(jr.key(seed, impl="threefry2x32"), N_AGENTS)


@pytest.fixture(scope="module")
def route_rig() -> _RouteRig:
    with jax.disable_jit():
        dyad = _dyad(planning_enabled=False)
        maximum_recurring_events = (
            dyad.config.hccl.world_config.maximum_committed_transitions - 1
        )
        lineage = KondoExecutedActionLineageBridgeConfig(
            actor=_actor_config(max_screenings=maximum_recurring_events),
            action_stack=dyad.config.agent_0,
            action_stack_rows=(dyad.config.agent_0, dyad.config.agent_1),
        )
        route = HCCLKondoContinualDyadRoute(
            HCCLKondoContinualDyadRouteConfig(
                dyad=dyad.config,
                lineage=lineage,
                protected_td=KondoProtectedTDConfig(
                    batch_size=N_AGENTS,
                    feature_dim=BASE_DIM,
                    action_count=N_ACTIONS,
                    learning_rate=0.01,
                    max_updates=maximum_recurring_events,
                ),
            )
        )
        genesis = route.init(
            jr.key(901, impl="threefry2x32"),
            _parameters(),
            jr.key(902, impl="threefry2x32"),
        )
    return _RouteRig(route=route, genesis=genesis)


@pytest.fixture(scope="module")
def event_0(route_rig: _RouteRig) -> HCCLKondoContinualDyadEventResult:
    route = route_rig.route
    with jax.disable_jit():
        result = route.event0(
            route_rig.genesis,
            MASKS,
            _keys(923),
        )
    return result


@pytest.fixture(scope="module")
def life_rig(
    route_rig: _RouteRig,
    event_0: HCCLKondoContinualDyadEventResult,
) -> _LifeRig:
    route = route_rig.route
    with jax.disable_jit():
        event_1 = route.event(
            event_0.state,
            MASKS,
            _keys(926),
        )
    return _LifeRig(route_rig=route_rig, event_0=event_0, event_1=event_1)


@pytest.fixture(scope="module")
def event_2(life_rig: _LifeRig) -> HCCLKondoContinualDyadEventResult:
    route = life_rig.route_rig.route
    with jax.disable_jit():
        result = route.event(
            life_rig.event_1.state,
            MASKS,
            _keys(929),
        )
    return result


def test_config_and_genesis_are_l0_shadow_only_without_duplicate_agent_owners(
    route_rig: _RouteRig,
) -> None:
    route = route_rig.route
    state = route_rig.genesis
    payload = route.to_config()

    assert HCCL_KONDO_CONTINUAL_DYAD_ROUTE_SCHEMA == (
        "alberta.hccl-kondo-continual-dyad-route.v3"
    )
    assert HCCL_KONDO_CONTINUAL_DYAD_ROUTE_STATUS == (
        "l0-development-hccl-kondo-actor-owned-p"
    )
    assert payload["mechanism_status"] == HCCL_KONDO_CONTINUAL_DYAD_ROUTE_STATUS
    assert payload["evidence_level"] == "L0"
    assert payload["shadow_planner_enabled"] is True
    assert payload["shadow_planner_planning_enabled"] is False
    assert payload["actor_batch_rows"] == N_AGENTS
    assert payload["caller_memory_event_inputs"] is False
    assert payload["memory_event_input_authority"] == (
        "exact-causal-core-event-and-agent"
    )
    assert payload["memory_event_input_derivations_per_event"] == N_AGENTS
    assert payload["caller_protected_actor_inputs"] is False
    assert payload["protected_actor_input_authority"] == (
        "exact-pending-P-and-current-PP-transition"
    )
    assert payload["protected_full_batch_rows"] == N_AGENTS
    assert payload["protected_reward_heads"] == 1
    assert payload["protected_cost_heads"] == 1
    assert payload["actor_and_protected_clocks_bound_to_route"] is True
    assert payload["route_checkpoint_supported"] is False
    assert payload["protected_td_checkpoint_is_route_checkpoint"] is False
    assert payload["hard_action_mask_support"] == "exact-all-true-only"
    assert payload["pair_atomic_compact_preflight"] is True
    assert payload["recurring_events_supported"] is True
    assert payload["outer_joy_alias"] is False
    assert payload["dispatch_authority"] is False
    assert payload["promotion_authority"] is False
    assert route.config.dyad.planner.planning_enabled is False
    assert tuple(field.name for field in dataclasses.fields(state)) == (
        "config_token",
        "content_token",
        "components",
        "actor_state",
        "protected_td_state",
        "pending_proposal",
        "pending_compact_adoptions",
        "event_count",
    )
    assert state.pending_proposal is None
    assert state.pending_compact_adoptions is None
    assert int(state.event_count) == 0
    assert int(state.protected_td_state.update_count) == 0
    assert bool(route.state_valid(state))
    assert HCCLKondoContinualDyadRouteConfig.from_config(payload).to_config() == payload
    assert HCCLKondoContinualDyadRoute.from_config(payload).to_config() == payload
    maximum_recurring_events = (
        route.config.dyad.hccl.world_config.maximum_committed_transitions - 1
    )
    with pytest.raises(ValueError, match="max_updates"):
        dataclasses.replace(
            route.config,
            protected_td=dataclasses.replace(
                route.config.protected_td,
                max_updates=maximum_recurring_events - 1,
            ),
        )
    short_gate = dataclasses.replace(
        route.config.lineage.actor.gate,
        max_screenings=maximum_recurring_events - 1,
    )
    with pytest.raises(ValueError, match="max_screenings"):
        dataclasses.replace(
            route.config,
            lineage=dataclasses.replace(
                route.config.lineage,
                actor=dataclasses.replace(
                    route.config.lineage.actor,
                    gate=short_gate,
                ),
            ),
        )
    desynchronized_protected = route.protected_td.reseal_state(
        state.protected_td_state.replace(
            update_count=jnp.asarray(1, dtype=jnp.int32),
        )
    )
    desynchronized = route._seal_state(
        state.replace(
            protected_td_state=desynchronized_protected,
            content_token=jnp.zeros_like(state.content_token),
        )
    )
    assert bool(route.protected_td.state_valid(desynchronized_protected))
    assert not bool(route.state_valid(desynchronized))
    assert not hasattr(state, "sparks_joy")
    for method in (route.event0, route.event1, route.event):
        parameters = inspect.signature(method).parameters
        assert "agent_0_event_input" not in parameters
        assert "agent_1_event_input" not in parameters
        assert "protected" not in parameters


def test_event0_installs_exact_actor_owned_p_and_pending_compact_certificate(
    route_rig: _RouteRig,
    event_0: HCCLKondoContinualDyadEventResult,
) -> None:
    route = route_rig.route
    source = route_rig.genesis
    result = event_0
    state = result.state
    proposal = result.proposal
    compact = result.compact_adoptions

    assert bool(result.update_applied)
    assert not bool(result.complete_source_returned)
    assert result.lineage_result is None
    assert result.protected_td_result is None
    assert proposal is not None
    assert compact is not None
    assert int(state.event_count) == 1
    assert state.pending_proposal is proposal
    assert state.pending_compact_adoptions is compact
    _tree_exact(state.actor_state, source.actor_state)
    _tree_exact(state.protected_td_state, source.protected_td_state)
    assert bool(route.state_valid(state))
    retagged_proposal = proposal.replace(
        behavior_log_probability=proposal.behavior_log_probability.at[0].add(
            jnp.asarray(0.125, dtype=jnp.float32)
        ),
        proposal_digest_words=jnp.zeros_like(proposal.proposal_digest_words),
    )
    retagged_proposal = retagged_proposal.replace(
        proposal_digest_words=route.lineage.rederive_proposal_digest_words(
            retagged_proposal
        )
    )
    retagged_state = route._seal_state(
        state.replace(
            pending_proposal=retagged_proposal,
            content_token=jnp.zeros_like(state.content_token),
        )
    )
    assert not bool(route.state_valid(retagged_state))
    assert int(result.work.pending_compact_lineage_steps) == 0
    assert int(result.work.causal_core_memory_event_input_derivations) == N_AGENTS
    assert int(result.work.actor_proposal_batches) == 1
    assert int(result.work.protected_td_full_batch_backward_calls) == 0
    assert int(result.work.protected_td_rows) == 0
    assert int(result.work.shadow_planner_completed_transition_calls) == 1
    np.testing.assert_array_equal(result.work.action_stack_final_bindings, (1, 1))
    np.testing.assert_array_equal(result.work.public_child_adoptions, (1, 1))
    assert int(result.work.compact_certificate_issuances) == 1
    assert bool(result.work.pending_consumed_before_new_sampling)
    assert not hasattr(result, "sparks_joy")

    agents = (state.components.agent_0_state, state.components.agent_1_state)
    planner_agents = (
        state.components.planner_state.agent_0,
        state.components.planner_state.agent_1,
    )
    shadow_agents = (
        result.shadow_planner_result.state.agent_0,
        result.shadow_planner_result.state.agent_1,
    )
    adopted = (result.agent_0_adoption, result.agent_1_adoption)
    assert bool(jnp.all(proposal.hard_action_masks))
    assert bool(jnp.all(compact.planner_consumed))
    for row, (agent, planner_agent, shadow_agent, adoption) in enumerate(
        zip(agents, planner_agents, shadow_agents, adopted, strict=True)
    ):
        assert adoption is not None
        assert bool(adoption.diagnostics.transaction_applied)
        event_input = adoption.finalized.memory_preparation.event_input
        assert not bool(event_input.query_uncertainty_available)
        assert not bool(event_input.entry_uncertainty_available)
        assert bool(event_input.entry_safety_cost_available)
        assert float(event_input.entry_reliability) == 1.0
        assert int(event_input.provenance_id) == row
        assert int(event_input.source_id) == row
        assert int(
            jax.lax.bitcast_convert_type(
                event_input.entry_safety_cost,
                jnp.uint32,
            )
        ) == 0
        binding = agent.action_binding
        selected = proposal.selected_actions[row]
        assert bool(binding.planner_bound)
        assert bool(binding.planner_consumed)
        assert int(binding.planner_action_before_mask) == int(selected)
        assert int(binding.final_action) == int(selected)
        np.testing.assert_array_equal(
            binding.planner_candidate_words,
            proposal.proposal_digest_words[row],
        )
        cache = planner_agent.cache
        assert int(cache.base_action) == int(selected)
        assert int(cache.base_action_guard) == int(jnp.bitwise_not(selected))
        assert int(cache.effective_action) == int(selected)
        assert not bool(cache.planner_consumed)
        for field in dataclasses.fields(type(cache)):
            if field.name in {"base_action", "base_action_guard", "effective_action"}:
                continue
            _tree_exact(getattr(cache, field.name), getattr(shadow_agent.cache, field.name))
        np.testing.assert_array_equal(
            compact.planner_candidate_words[row],
            proposal.proposal_digest_words[row],
        )
        assert bool(compact.planner_consumed[row])
        assert bool(compact.adoption_applied[row])


def test_event1_consumes_pending_compact_lineage_before_sampling_the_next_pair(
    life_rig: _LifeRig,
) -> None:
    route = life_rig.route_rig.route
    prior = life_rig.event_0
    result = life_rig.event_1
    lineage = result.lineage_result
    protected = result.protected_td_result
    proposal = result.proposal

    assert bool(result.update_applied)
    assert not bool(result.complete_source_returned)
    assert lineage is not None
    assert protected is not None
    assert proposal is not None
    assert int(result.state.event_count) == 2
    assert int(result.work.pending_compact_lineage_steps) == 1
    assert int(result.work.causal_core_memory_event_input_derivations) == N_AGENTS
    assert int(result.work.actor_proposal_batches) == 1
    assert int(result.work.protected_td_full_batch_backward_calls) == 1
    assert int(result.work.protected_td_rows) == N_AGENTS
    assert bool(result.work.pending_consumed_before_new_sampling)
    assert int(result.work.shadow_planner_completed_transition_calls) == 1
    np.testing.assert_array_equal(lineage.diagnostics.lineage_mode, (1, 1))
    np.testing.assert_array_equal(lineage.diagnostics.actor_eligible, (True, True))
    assert protected.batch is not None
    assert bool(protected.full_batch_backward_executed)
    assert int(protected.full_batch_rows) == N_AGENTS
    assert bool(protected.transaction_applied)
    assert lineage.protected is protected.actor_inputs
    assert lineage.actor_result.protected is protected.actor_inputs
    np.testing.assert_array_equal(
        protected.batch.current_features,
        prior.proposal.actor_features,
    )
    np.testing.assert_array_equal(
        protected.batch.next_features,
        proposal.actor_features,
    )
    np.testing.assert_array_equal(
        protected.batch.actions,
        prior.proposal.selected_actions,
    )
    np.testing.assert_array_equal(
        protected.batch.decision_identities,
        prior.proposal.action_stack_decision_identities,
    )
    assert result.agent_0_adoption is not None
    assert result.agent_1_adoption is not None
    transitions = (
        result.agent_0_adoption.finalized.memory_preparation.transition,
        result.agent_1_adoption.finalized.memory_preparation.transition,
    )
    np.testing.assert_array_equal(
        protected.batch.current_features,
        jnp.stack(tuple(item.representation for item in transitions)),
    )
    np.testing.assert_array_equal(
        protected.batch.actions,
        jnp.stack(tuple(item.action for item in transitions)),
    )
    np.testing.assert_array_equal(
        protected.batch.decision_identities,
        jnp.stack(tuple(item.decision_id for item in transitions)),
    )
    np.testing.assert_array_equal(
        protected.batch.rewards,
        jnp.stack(tuple(item.reward for item in transitions)),
    )
    np.testing.assert_array_equal(
        protected.batch.discounts,
        jnp.stack(tuple(item.discount for item in transitions)),
    )
    np.testing.assert_array_equal(
        jax.lax.bitcast_convert_type(protected.batch.costs, jnp.uint32),
        jnp.zeros((N_AGENTS,), dtype=jnp.uint32),
    )
    pre = prior.state.protected_td_state.parameters
    expected_reward_baseline = (
        protected.batch.current_features @ pre.reward_weight + pre.reward_bias
    )
    expected_reward_bootstrap = (
        protected.batch.next_features @ pre.reward_weight + pre.reward_bias
    )
    expected_return_targets = (
        protected.batch.rewards
        + protected.batch.discounts * expected_reward_bootstrap
    )
    expected_cost_baseline = (
        protected.batch.current_features @ pre.cost_weight + pre.cost_bias
    )
    expected_cost_bootstrap = (
        protected.batch.next_features @ pre.cost_weight + pre.cost_bias
    )
    expected_cost_targets = (
        protected.batch.costs
        + protected.batch.discounts * expected_cost_bootstrap
    )
    np.testing.assert_array_equal(protected.reward_baseline, expected_reward_baseline)
    np.testing.assert_array_equal(protected.reward_bootstrap, expected_reward_bootstrap)
    np.testing.assert_array_equal(protected.return_targets, expected_return_targets)
    np.testing.assert_array_equal(protected.cost_baseline, expected_cost_baseline)
    np.testing.assert_array_equal(protected.cost_bootstrap, expected_cost_bootstrap)
    np.testing.assert_array_equal(protected.cost_targets, expected_cost_targets)
    np.testing.assert_array_equal(
        protected.actor_inputs.critic_features,
        protected.batch.current_features,
    )
    np.testing.assert_array_equal(
        protected.actor_inputs.safety_features,
        protected.batch.current_features,
    )
    np.testing.assert_array_equal(
        protected.actor_inputs.baseline_predictions,
        protected.reward_baseline,
    )
    np.testing.assert_array_equal(
        protected.actor_inputs.return_targets,
        protected.return_targets,
    )
    assert not hasattr(protected, "sparks_joy")
    assert bool(lineage.actor_result.transaction_applied)
    np.testing.assert_array_equal(lineage.actor_result.sparks_joy, (True, True))
    np.testing.assert_array_equal(
        lineage.actor_result.executed_actor_backward_mask,
        (True, True),
    )
    assert int(lineage.actor_result.backward_selected_count) == N_AGENTS
    assert bool(lineage.actor_result.backward_delight_exact)
    np.testing.assert_array_equal(
        jax.lax.bitcast_convert_type(
            lineage.actor_result.executed_delight,
            jnp.uint32,
        ),
        jax.lax.bitcast_convert_type(
            lineage.actor_result.screen.delight,
            jnp.uint32,
        ),
    )
    assert not hasattr(lineage, "sparks_joy")
    assert not hasattr(result, "sparks_joy")
    assert int(result.state.actor_state.policy_revision) == (
        int(prior.state.actor_state.policy_revision) + 1
    )
    assert int(result.state.protected_td_state.update_count) == 1
    np.testing.assert_array_equal(
        proposal.actor_state_words,
        route.lineage.actor_state_digest_words(result.state.actor_state),
    )
    assert result.state.pending_proposal is proposal
    assert result.state.pending_proposal is not prior.state.pending_proposal
    assert result.state.pending_compact_adoptions is result.compact_adoptions
    assert result.state.pending_compact_adoptions is not (
        prior.state.pending_compact_adoptions
    )
    for row, adoption in enumerate(
        (result.agent_0_adoption, result.agent_1_adoption)
    ):
        assert adoption is not None
        event_input = adoption.finalized.memory_preparation.event_input
        assert int(event_input.provenance_id) == N_AGENTS + row
        assert int(event_input.source_id) == row
    assert bool(route.state_valid(result.state))


def test_generic_event_repeats_compact_lineage_before_each_new_actor_pair(
    life_rig: _LifeRig,
    event_2: HCCLKondoContinualDyadEventResult,
) -> None:
    route = life_rig.route_rig.route
    prior = life_rig.event_1
    result = event_2
    lineage = result.lineage_result
    protected = result.protected_td_result

    assert lineage is not None
    assert protected is not None
    assert bool(result.update_applied)
    assert not bool(result.complete_source_returned)
    assert int(result.state.event_count) == 3
    assert int(result.work.pending_compact_lineage_steps) == 1
    assert int(result.work.actor_proposal_batches) == 1
    assert int(result.work.protected_td_full_batch_backward_calls) == 1
    assert int(result.work.protected_td_rows) == N_AGENTS
    assert bool(result.work.pending_consumed_before_new_sampling)
    pre = prior.state.protected_td_state.parameters
    expected_reward_baseline = (
        protected.batch.current_features @ pre.reward_weight + pre.reward_bias
    )
    expected_reward_bootstrap = (
        protected.batch.next_features @ pre.reward_weight + pre.reward_bias
    )
    expected_return_targets = (
        protected.batch.rewards
        + protected.batch.discounts * expected_reward_bootstrap
    )
    expected_cost_baseline = (
        protected.batch.current_features @ pre.cost_weight + pre.cost_bias
    )
    expected_cost_bootstrap = (
        protected.batch.next_features @ pre.cost_weight + pre.cost_bias
    )
    expected_cost_targets = (
        protected.batch.costs
        + protected.batch.discounts * expected_cost_bootstrap
    )
    np.testing.assert_array_equal(protected.reward_baseline, expected_reward_baseline)
    np.testing.assert_array_equal(protected.reward_bootstrap, expected_reward_bootstrap)
    np.testing.assert_array_equal(protected.return_targets, expected_return_targets)
    np.testing.assert_array_equal(protected.cost_baseline, expected_cost_baseline)
    np.testing.assert_array_equal(protected.cost_bootstrap, expected_cost_bootstrap)
    np.testing.assert_array_equal(protected.cost_targets, expected_cost_targets)
    assert bool(lineage.actor_result.transaction_applied)
    np.testing.assert_array_equal(lineage.actor_result.sparks_joy, (True, True))
    assert int(result.state.actor_state.policy_revision) == (
        int(prior.state.actor_state.policy_revision) + 1
    )
    assert int(result.state.protected_td_state.update_count) == (
        int(prior.state.protected_td_state.update_count) + 1
    )
    assert result.state.pending_proposal is result.proposal
    assert result.state.pending_proposal is not prior.state.pending_proposal
    assert result.state.pending_compact_adoptions is result.compact_adoptions
    assert result.state.pending_compact_adoptions is not (
        prior.state.pending_compact_adoptions
    )
    assert bool(route.state_valid(result.state))


def test_one_child_veto_returns_every_route_owner_bit_exactly(
    route_rig: _RouteRig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route = route_rig.route
    adapter_type = type(route.dyad.agent_0)
    original = adapter_type.adopt_finalized_transition
    calls = [0, 0]

    def _veto_agent_1(self: object, *args: object, **kwargs: object) -> object:
        row = 0 if self is route.dyad.agent_0 else 1
        calls[row] += 1
        adopted = original(self, *args, **kwargs)
        if row == 0:
            return adopted
        return adopted.replace(
            diagnostics=adopted.diagnostics.replace(
                transaction_applied=jnp.asarray(False, dtype=jnp.bool_)
            )
        )

    monkeypatch.setattr(adapter_type, "adopt_finalized_transition", _veto_agent_1)
    result = route.event0(
        route_rig.genesis,
        MASKS,
        _keys(933),
    )

    assert calls == [1, 1]
    assert not bool(result.update_applied)
    assert bool(result.complete_source_returned)
    assert result.compact_adoptions is None
    _tree_exact(result.state, route_rig.genesis)


def test_event1_veto_rolls_back_actor_state_but_preserves_nested_backward_fact(
    route_rig: _RouteRig,
    event_0: HCCLKondoContinualDyadEventResult,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route = route_rig.route
    source = event_0.state
    adapter_type = type(route.dyad.agent_0)
    original = adapter_type.adopt_finalized_transition
    calls = [0, 0]

    def _veto_agent_1(self: object, *args: object, **kwargs: object) -> object:
        row = 0 if self is route.dyad.agent_0 else 1
        calls[row] += 1
        adopted = original(self, *args, **kwargs)
        if row == 0:
            return adopted
        return adopted.replace(
            diagnostics=adopted.diagnostics.replace(
                transaction_applied=jnp.asarray(False, dtype=jnp.bool_)
            )
        )

    monkeypatch.setattr(adapter_type, "adopt_finalized_transition", _veto_agent_1)
    result = route.event1(
        source,
        MASKS,
        _keys(943),
    )

    assert calls == [1, 1]
    assert result.lineage_result is not None
    assert result.protected_td_result is not None
    assert bool(result.protected_td_result.full_batch_backward_executed)
    assert bool(result.protected_td_result.transaction_applied)
    np.testing.assert_array_equal(
        result.lineage_result.actor_result.sparks_joy,
        (True, True),
    )
    assert not hasattr(result, "sparks_joy")
    assert not bool(result.update_applied)
    assert bool(result.complete_source_returned)
    assert result.compact_adoptions is None
    _tree_exact(result.state.protected_td_state, source.protected_td_state)
    _tree_exact(result.state, source)
