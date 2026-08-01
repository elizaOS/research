"""Causal online bridge contracts for the noisy world and integrated agent."""

from __future__ import annotations

import copy
import dataclasses
import functools

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.evaluation.hidden_partner_world_filter import (
    HiddenPartnerWorldBayesFilter,
    HiddenPartnerWorldFilterConfig,
)
from alberta_framework.evaluation.hidden_partner_world_online_bridge import (
    HIDDEN_PARTNER_WORLD_ONLINE_BRIDGE_CONFIG_SCHEMA,
    HiddenPartnerWorldOnlineBridge,
    HiddenPartnerWorldOnlineResourceBudget,
    HiddenPartnerWorldOnlineState,
    HiddenPartnerWorldOnlineStep,
    LearnerHiddenPartnerWorldTransition,
    strip_hidden_partner_world_oracle,
)
from alberta_framework.streams.hidden_partner_world_feedback import (
    CUE_1_INDEX,
    CUE_2_INDEX,
    HiddenPartnerWorldFeedbackConfig,
    HiddenPartnerWorldFeedbackWorld,
)

pytestmark = pytest.mark.unit

_INT32_MAX = 2**31 - 1


@functools.lru_cache(maxsize=1)
def _shared_bridge() -> HiddenPartnerWorldOnlineBridge:
    return HiddenPartnerWorldOnlineBridge()


def _unwrap_prng_keys(tree: object) -> object:
    def unwrap(value: object) -> object:
        dtype = getattr(value, "dtype", None)
        if dtype is not None and jax.dtypes.issubdtype(dtype, jax.dtypes.prng_key):
            return jr.key_data(value)  # type: ignore[arg-type]
        return value

    return jax.tree_util.tree_map(unwrap, tree)


def _assert_atomic_rejection(
    before: HiddenPartnerWorldOnlineState,
    rejected: HiddenPartnerWorldOnlineStep,
) -> None:
    assert not bool(rejected.trace.accepted)
    assert not bool(rejected.state.valid)
    expected = before.replace(valid=jnp.asarray(False, dtype=jnp.bool_))
    chex.assert_trees_all_equal(
        _unwrap_prng_keys(rejected.state),
        _unwrap_prng_keys(expected),
    )
    assert int(rejected.trace.committed_world_step_delta) == 0
    assert int(rejected.trace.committed_agent_step_delta) == 0
    assert int(rejected.trace.committed_filter_step_delta) == 0
    assert int(rejected.trace.committed_bridge_step_delta) == 0


def test_oracle_strip_is_exact_and_oracle_mutation_cannot_change_learner_transition() -> None:
    world = HiddenPartnerWorldFeedbackWorld()
    state = world.init(jr.key(1))
    transition, _ = world.step(state, jnp.asarray(1, dtype=jnp.int32))
    stripped = strip_hidden_partner_world_oracle(transition)
    adversarial = transition.replace(
        oracle=transition.oracle.replace(
            world_sign=-transition.oracle.world_sign,
            regime_id=jnp.asarray(99, dtype=jnp.int32),
            counterfactual_rewards=1.0 - transition.oracle.counterfactual_rewards,
        )
    )
    adversarial_stripped = strip_hidden_partner_world_oracle(adversarial)

    assert isinstance(stripped, LearnerHiddenPartnerWorldTransition)
    assert not hasattr(stripped, "oracle")
    assert set(field.name for field in dataclasses.fields(stripped)) == {
        "observation",
        "focal_action",
        "partner_action",
        "reward",
        "outcome",
        "next_observation",
        "terminated",
        "discount",
    }
    chex.assert_trees_all_equal(stripped, adversarial_stripped)


def test_online_bridge_advances_world_agent_and_filter_once_in_causal_order() -> None:
    bridge = _shared_bridge()
    state = bridge.initialize(jr.key(2), jr.key(3))
    pre_cells = bridge.world_filter.expected_reward_cells(
        state.world_filter.posterior_mean
    )
    pre_decision = bridge.world_filter.marginalize_partner(
        pre_cells,
        state.agent.current_evaluation.partner_probabilities,
    )
    step = bridge.step(state)

    assert bool(state.valid)
    assert bool(step.trace.active)
    assert bool(step.trace.accepted)
    assert int(step.trace.step) == 0
    assert int(step.trace.focal_action) == int(state.action)
    assert int(step.trace.next_action) == int(step.state.action)
    assert int(step.trace.committed_world_step_delta) == 1
    assert int(step.trace.committed_agent_step_delta) == 1
    assert int(step.trace.committed_filter_step_delta) == 1
    assert int(step.trace.committed_bridge_step_delta) == 1
    assert int(step.trace.proposed_world_step_delta) == 1
    assert int(step.trace.proposed_agent_step_delta) == 1
    assert int(step.trace.proposed_filter_step_delta) == 1
    assert int(step.trace.proposed_bridge_step_delta) == 1
    assert int(step.state.step_count) == 1
    assert int(step.state.world.step_count) == 1
    assert int(step.state.agent.step_count) == 1
    assert int(step.state.world_filter.step_count) == 1
    assert bool(step.trace.entry_state_contract_valid)
    assert bool(step.trace.config_token_valid)
    assert bool(step.trace.counters_synchronized)
    assert bool(step.trace.action_valid)
    assert bool(step.trace.filter_entry_valid)
    assert bool(step.trace.proposed_agent_update_valid)
    assert bool(step.trace.proposed_filter_update_valid)
    assert bool(step.trace.proposed_filter_decision_valid)
    assert bool(step.trace.oracle_trace_valid)
    assert bool(step.trace.all_finite)
    chex.assert_trees_all_equal(step.trace.observation_pre, state.agent.raw_observation)
    chex.assert_trees_all_equal(step.trace.next_observation, step.state.agent.raw_observation)
    corrected = (
        step.trace.outcome
        * (2.0 * step.trace.focal_action.astype(jnp.float32) - 1.0)
        * (2.0 * step.trace.partner_action.astype(jnp.float32) - 1.0)
    )
    chex.assert_trees_all_equal(step.trace.corrected_outcome, corrected)
    expected_filter = bridge.world_filter.advance(
        state.world_filter,
        corrected,
        step.trace.next_observation[jnp.asarray((CUE_1_INDEX, CUE_2_INDEX))],
    )
    chex.assert_trees_all_close(
        step.state.world_filter,
        expected_filter.state,
        atol=1e-6,
        rtol=0.0,
    )
    chex.assert_trees_all_equal(
        step.trace.agent_partner_belief_conditioned_reward_cells,
        pre_cells.rewards,
    )
    chex.assert_trees_all_equal(
        step.trace.agent_partner_belief_conditioned_expected_rewards,
        pre_decision.expected_rewards,
    )
    assert (
        int(step.trace.agent_partner_belief_conditioned_greedy_action)
        == int(pre_decision.greedy_action)
    )
    assert 0.0 <= float(
        step.trace.agent_partner_belief_conditioned_selected_regret
    ) <= 1.0
    assert step.trace.agent_partner_belief_conditioned_reward_cells.shape == (2, 2)
    assert step.trace.agent_partner_belief_conditioned_expected_rewards.shape == (2,)
    np.testing.assert_allclose(
        np.sum(np.asarray(step.trace.agent_applied_partner_probabilities)),
        1.0,
        atol=1e-6,
        rtol=0.0,
    )


def test_bridge_constructor_rejects_types_before_reading_nested_properties() -> None:
    with pytest.raises(TypeError, match="world"):
        HiddenPartnerWorldOnlineBridge(world=object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="agent"):
        HiddenPartnerWorldOnlineBridge(agent=object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="world_filter"):
        HiddenPartnerWorldOnlineBridge(world_filter=object())  # type: ignore[arg-type]


def test_bridge_constructor_rejects_world_filter_probability_mismatch() -> None:
    world = HiddenPartnerWorldFeedbackWorld(
        HiddenPartnerWorldFeedbackConfig(world_flip_probability=0.02)
    )
    mismatched = HiddenPartnerWorldBayesFilter(HiddenPartnerWorldFilterConfig())
    with pytest.raises(ValueError, match="probability contracts"):
        HiddenPartnerWorldOnlineBridge(world=world, world_filter=mismatched)


def test_agent_cache_rejection_atomically_latches_bridge_without_advancing_world() -> None:
    bridge = _shared_bridge()
    state = bridge.initialize(jr.key(4), jr.key(5))
    corrupt_agent = state.agent.replace(
        behavior=state.agent.behavior.replace(
            weights=state.agent.behavior.weights.at[0, 0].add(0.25)
        )
    )
    corrupt = state.replace(agent=corrupt_agent)
    rejected = bridge.step(corrupt)

    assert not bool(rejected.trace.accepted)
    assert not bool(rejected.trace.proposed_agent_update_valid)
    _assert_atomic_rejection(corrupt, rejected)
    assert int(rejected.trace.proposed_world_step_delta) == 1
    assert int(rejected.trace.proposed_filter_step_delta) == 1
    assert int(rejected.trace.proposed_agent_step_delta) == 0
    np.testing.assert_array_equal(
        rejected.trace.agent_partner_belief_conditioned_reward_cells,
        np.full((2, 2), 0.5, dtype=np.float32),
    )
    assert int(rejected.trace.oracle_regime_id) == -1

    blocked = jax.jit(bridge.step)(rejected.state)
    assert not bool(blocked.trace.active)
    assert not bool(blocked.trace.accepted)
    chex.assert_trees_all_equal(
        _unwrap_prng_keys(blocked.state),
        _unwrap_prng_keys(rejected.state),
    )


def test_bridge_config_roundtrip_token_authority_and_tamper_rejection() -> None:
    bridge = _shared_bridge()
    payload = bridge.to_config()
    restored = HiddenPartnerWorldOnlineBridge.from_config(payload)

    assert payload["schema"] == HIDDEN_PARTNER_WORLD_ONLINE_BRIDGE_CONFIG_SCHEMA
    assert payload["development_only"] is True
    assert payload["execution_authorized"] is False
    assert payload["evidence_authorized"] is False
    assert payload["scientific_promotion_allowed"] is False
    assert restored.to_config() == payload
    assert restored.config_token_hex == bridge.config_token_hex
    assert len(bridge.config_token_hex) == 64

    for field in (
        "execution_authorized",
        "evidence_authorized",
        "scientific_promotion_allowed",
    ):
        hostile = copy.deepcopy(payload)
        hostile[field] = True
        with pytest.raises(ValueError):
            HiddenPartnerWorldOnlineBridge.from_config(hostile)
    extra = copy.deepcopy(payload)
    extra["extra"] = False
    with pytest.raises(ValueError, match="fields"):
        HiddenPartnerWorldOnlineBridge.from_config(extra)


def test_resource_budget_is_exact_includes_bridge_metadata_and_zero_replay() -> None:
    bridge = _shared_bridge()
    state = bridge.initialize(jr.key(10), jr.key(11))
    budget = bridge.resource_budget(state)

    assert isinstance(budget, HiddenPartnerWorldOnlineResourceBudget)
    assert budget.world_state_nbytes == bridge.world.resource_budget.state_nbytes
    assert budget.agent_state_nbytes == bridge.agent.resource_budget(state.agent).total_state_nbytes
    assert budget.filter_state_nbytes == 9
    assert budget.config_token_nbytes == 32
    assert budget.action_nbytes == 4
    assert budget.valid_nbytes == 1
    assert budget.step_count_nbytes == 4
    assert budget.bridge_metadata_nbytes == 41
    assert budget.component_state_nbytes == (
        budget.world_state_nbytes + budget.agent_state_nbytes + budget.filter_state_nbytes
    )
    assert budget.total_state_nbytes == budget.component_state_nbytes + 41
    assert budget.world_replay_capacity == 0
    assert budget.agent_replay_capacity == 0
    assert budget.replay_capacity == 0
    assert budget.to_dict()["total_state_nbytes"] == budget.total_state_nbytes


@pytest.mark.parametrize(
    "mutation",
    (
        "bridge_negative",
        "bridge_saturated",
        "world_desynchronized",
        "agent_desynchronized",
    ),
)
def test_counter_contract_rejects_atomically_before_component_work(mutation: str) -> None:
    bridge = _shared_bridge()
    state = bridge.initialize(jr.key(12), jr.key(13))
    if mutation == "bridge_negative":
        corrupt = state.replace(step_count=jnp.asarray(-1, dtype=jnp.int32))
    elif mutation == "bridge_saturated":
        corrupt = state.replace(step_count=jnp.asarray(_INT32_MAX, dtype=jnp.int32))
    elif mutation == "world_desynchronized":
        corrupt = state.replace(
            world=state.world.replace(step_count=jnp.asarray(1, dtype=jnp.int32))
        )
    else:
        corrupt = state.replace(
            agent=state.agent.replace(step_count=jnp.asarray(1, dtype=jnp.int32))
        )

    with jax.disable_jit():
        rejected = bridge.step(corrupt)
    assert bool(rejected.trace.active)
    assert not bool(rejected.trace.entry_state_contract_valid)
    assert not bool(rejected.trace.counters_synchronized)
    assert int(rejected.trace.proposed_world_step_delta) == 0
    assert int(rejected.trace.proposed_agent_step_delta) == 0
    assert int(rejected.trace.proposed_filter_step_delta) == 0
    _assert_atomic_rejection(corrupt, rejected)


def test_invalid_action_rejects_before_component_work() -> None:
    bridge = _shared_bridge()
    state = bridge.initialize(jr.key(14), jr.key(15))
    corrupt = state.replace(action=jnp.asarray(2, dtype=jnp.int32))

    with jax.disable_jit():
        rejected = bridge.step(corrupt)
    assert bool(rejected.trace.active)
    assert not bool(rejected.trace.entry_state_contract_valid)
    assert not bool(rejected.trace.action_valid)
    _assert_atomic_rejection(corrupt, rejected)


@pytest.mark.parametrize("mutation", ("posterior_nan", "invalid_flag", "counter_desync"))
def test_filter_corruption_is_diagnostic_only_and_cannot_change_learning_path(
    mutation: str,
) -> None:
    bridge = _shared_bridge()
    state = bridge.initialize(jr.key(14), jr.key(15))
    if mutation == "posterior_nan":
        corrupt_filter = state.world_filter.replace(
            posterior_mean=jnp.asarray(jnp.nan, dtype=jnp.float32)
        )
    elif mutation == "invalid_flag":
        corrupt_filter = state.world_filter.replace(
            valid=jnp.asarray(False, dtype=jnp.bool_)
        )
    else:
        corrupt_filter = state.world_filter.replace(
            step_count=jnp.asarray(1, dtype=jnp.int32)
        )
    corrupt = state.replace(world_filter=corrupt_filter)

    with jax.disable_jit():
        ordinary_first = bridge.step(state)
        hostile_first = bridge.step(corrupt)
        ordinary_second = bridge.step(ordinary_first.state)
        hostile_second = bridge.step(hostile_first.state)

    for hostile in (hostile_first, hostile_second):
        assert bool(hostile.trace.active)
        assert bool(hostile.trace.accepted)
        assert bool(hostile.trace.entry_state_contract_valid)
        assert bool(hostile.trace.counters_synchronized)
        assert not bool(hostile.trace.filter_entry_valid)
        assert not bool(hostile.trace.all_finite)
        assert bool(hostile.state.valid)
        assert int(hostile.trace.committed_world_step_delta) == 1
        assert int(hostile.trace.committed_agent_step_delta) == 1
        assert int(hostile.trace.committed_bridge_step_delta) == 1
        np.testing.assert_array_equal(
            hostile.trace.agent_partner_belief_conditioned_reward_cells,
            np.full((2, 2), 0.5, dtype=np.float32),
        )
        np.testing.assert_array_equal(
            hostile.trace.agent_partner_belief_conditioned_expected_rewards,
            np.full((2,), 0.5, dtype=np.float32),
        )

    for ordinary, hostile in (
        (ordinary_first, hostile_first),
        (ordinary_second, hostile_second),
    ):
        chex.assert_trees_all_equal(
            _unwrap_prng_keys(hostile.state.world),
            _unwrap_prng_keys(ordinary.state.world),
        )
        chex.assert_trees_all_equal(
            _unwrap_prng_keys(hostile.state.agent),
            _unwrap_prng_keys(ordinary.state.agent),
        )
        assert int(hostile.state.action) == int(ordinary.state.action)
        assert int(hostile.state.step_count) == int(ordinary.state.step_count)


def test_filter_failure_cannot_change_jitted_scan_actions_or_updates() -> None:
    bridge = _shared_bridge()
    initial = bridge.initialize(jr.key(140), jr.key(150))
    hostile_initial = initial.replace(
        world_filter=initial.world_filter.replace(
            valid=jnp.asarray(False, dtype=jnp.bool_)
        )
    )

    @jax.jit
    def run(state: HiddenPartnerWorldOnlineState):
        def body(carry, _):
            result = bridge.step(carry)
            return result.state, result.trace

        return jax.lax.scan(body, state, xs=None, length=3)

    ordinary_final, ordinary_traces = run(initial)
    hostile_final, hostile_traces = run(hostile_initial)

    np.testing.assert_array_equal(ordinary_traces.accepted, np.ones((3,), dtype=bool))
    np.testing.assert_array_equal(hostile_traces.accepted, np.ones((3,), dtype=bool))
    np.testing.assert_array_equal(hostile_traces.filter_entry_valid, np.zeros((3,), dtype=bool))
    np.testing.assert_array_equal(hostile_traces.all_finite, np.zeros((3,), dtype=bool))
    np.testing.assert_array_equal(
        hostile_traces.focal_action,
        ordinary_traces.focal_action,
    )
    np.testing.assert_array_equal(
        hostile_traces.next_action,
        ordinary_traces.next_action,
    )
    chex.assert_trees_all_equal(
        _unwrap_prng_keys(hostile_final.world),
        _unwrap_prng_keys(ordinary_final.world),
    )
    chex.assert_trees_all_equal(
        _unwrap_prng_keys(hostile_final.agent),
        _unwrap_prng_keys(ordinary_final.agent),
    )
    assert int(hostile_final.action) == int(ordinary_final.action)
    assert int(hostile_final.step_count) == int(ordinary_final.step_count) == 3


def test_valid_filter_posterior_perturbation_changes_only_evaluator_trace() -> None:
    bridge = _shared_bridge()
    initial = bridge.initialize(jr.key(141), jr.key(151))
    perturbed_initial = initial.replace(
        world_filter=initial.world_filter.replace(
            posterior_mean=jnp.asarray(0.25, dtype=jnp.float32)
        )
    )

    with jax.disable_jit():
        ordinary_first = bridge.step(initial)
        perturbed_first = bridge.step(perturbed_initial)
        ordinary_second = bridge.step(ordinary_first.state)
        perturbed_second = bridge.step(perturbed_first.state)

    for ordinary, perturbed in (
        (ordinary_first, perturbed_first),
        (ordinary_second, perturbed_second),
    ):
        assert bool(ordinary.trace.accepted)
        assert bool(perturbed.trace.accepted)
        assert bool(perturbed.trace.filter_entry_valid)
        assert bool(perturbed.trace.proposed_filter_update_valid)
        assert bool(perturbed.trace.proposed_filter_decision_valid)
        chex.assert_trees_all_equal(
            _unwrap_prng_keys(perturbed.state.world),
            _unwrap_prng_keys(ordinary.state.world),
        )
        chex.assert_trees_all_equal(
            _unwrap_prng_keys(perturbed.state.agent),
            _unwrap_prng_keys(ordinary.state.agent),
        )
        assert int(perturbed.state.action) == int(ordinary.state.action)
        assert int(perturbed.state.step_count) == int(ordinary.state.step_count)

    assert not np.array_equal(
        np.asarray(
            perturbed_first.trace.agent_partner_belief_conditioned_reward_cells
        ),
        np.asarray(ordinary_first.trace.agent_partner_belief_conditioned_reward_cells),
    )


def test_cross_bridge_config_token_rejects_and_latches_invalid() -> None:
    source = _shared_bridge()
    source_state = source.initialize(jr.key(16), jr.key(17))
    other = HiddenPartnerWorldOnlineBridge(
        world=HiddenPartnerWorldFeedbackWorld(
            HiddenPartnerWorldFeedbackConfig(world_flip_probability=0.02)
        )
    )
    other.initialize(jr.key(18), jr.key(19))

    with jax.disable_jit():
        rejected = other.step(source_state)
    assert not bool(rejected.trace.config_token_valid)
    assert not bool(rejected.trace.entry_state_contract_valid)
    _assert_atomic_rejection(source_state, rejected)


def test_static_component_tree_shape_and_dtype_contracts_fail_before_computation() -> None:
    bridge = _shared_bridge()
    state = bridge.initialize(jr.key(20), jr.key(21))
    wrong_dtype = state.replace(action=state.action.astype(jnp.float32))
    with pytest.raises((TypeError, ValueError), match="static state contract"):
        bridge.step(wrong_dtype)

    wrong_component_shape = state.replace(
        world=state.world.replace(current_cues=jnp.ones((3,), dtype=jnp.float32))
    )
    with pytest.raises((TypeError, ValueError), match="static state contract"):
        bridge.step(wrong_component_shape)


def test_filter_decision_is_pretransition_and_two_step_filter_state_is_continuous() -> None:
    bridge = _shared_bridge()
    initial = bridge.initialize(jr.key(22), jr.key(23))
    initial = initial.replace(
        world_filter=initial.world_filter.replace(
            posterior_mean=jnp.asarray(0.0, dtype=jnp.float32)
        )
    )
    pre_cells = bridge.world_filter.expected_reward_cells(
        initial.world_filter.posterior_mean
    )
    first = bridge.step(initial)
    post_cells = bridge.world_filter.expected_reward_cells(
        first.state.world_filter.posterior_mean
    )
    second = bridge.step(first.state)

    assert bool(first.trace.accepted)
    assert bool(second.trace.accepted)
    chex.assert_trees_all_equal(
        first.trace.agent_partner_belief_conditioned_reward_cells,
        pre_cells.rewards,
    )
    assert not np.allclose(
        np.asarray(first.trace.agent_partner_belief_conditioned_reward_cells),
        np.asarray(post_cells.rewards),
    )
    chex.assert_trees_all_equal(
        first.trace.filter_mean_post,
        second.trace.filter_mean_pre,
    )


def test_lax_scan_midstream_rejection_latches_remaining_steps() -> None:
    bridge = _shared_bridge()
    initial = bridge.initialize(jr.key(24), jr.key(25))

    def body(state, corrupt_action):
        attempted = jax.lax.cond(
            corrupt_action,
            lambda current: current.replace(action=jnp.asarray(2, dtype=jnp.int32)),
            lambda current: current,
            state,
        )
        result = bridge.step(attempted)
        return result.state, result.trace

    final, traces = jax.lax.scan(
        body,
        initial,
        jnp.asarray((False, True, False), dtype=jnp.bool_),
    )
    np.testing.assert_array_equal(traces.active, np.asarray((True, True, False)))
    np.testing.assert_array_equal(traces.accepted, np.asarray((True, False, False)))
    assert int(final.step_count) == 1
    assert not bool(final.valid)


class _OracleCorruptingWorld(HiddenPartnerWorldFeedbackWorld):
    def step(self, state, focal_action):
        transition, next_state = super().step(state, focal_action)
        corrupted = transition.replace(
            oracle=transition.oracle.replace(
                world_sign=jnp.asarray(jnp.nan, dtype=jnp.float32),
                counterfactual_rewards=jnp.full((2,), jnp.nan, dtype=jnp.float32),
            )
        )
        return corrupted, next_state


def test_whole_bridge_oracle_corruption_cannot_change_committed_learning_path() -> None:
    world_key = jr.key(26)
    agent_key = jr.key(27)
    ordinary = _shared_bridge()
    hostile = HiddenPartnerWorldOnlineBridge(world=_OracleCorruptingWorld())
    ordinary_state = ordinary.initialize(world_key, agent_key)
    hostile_state = hostile.initialize(world_key, agent_key)

    ordinary_step = ordinary.step(ordinary_state)
    hostile_step = hostile.step(hostile_state)

    assert bool(ordinary_step.trace.accepted)
    assert bool(hostile_step.trace.accepted)
    assert bool(ordinary_step.trace.oracle_trace_valid)
    assert not bool(hostile_step.trace.oracle_trace_valid)
    assert not bool(hostile_step.trace.all_finite)
    assert float(hostile_step.trace.oracle_world_sign) == 0.0
    assert int(hostile_step.trace.oracle_regime_id) == -1
    chex.assert_trees_all_equal(
        _unwrap_prng_keys(hostile_step.state.world),
        _unwrap_prng_keys(ordinary_step.state.world),
    )
    chex.assert_trees_all_equal(
        _unwrap_prng_keys(hostile_step.state.agent),
        _unwrap_prng_keys(ordinary_step.state.agent),
    )
    chex.assert_trees_all_close(
        hostile_step.state.world_filter,
        ordinary_step.state.world_filter,
        atol=1e-6,
        rtol=0.0,
    )
    assert int(hostile_step.state.action) == int(ordinary_step.state.action)
    assert int(hostile_step.state.step_count) == int(ordinary_step.state.step_count)
