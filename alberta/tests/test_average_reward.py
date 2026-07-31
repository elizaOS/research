"""Tests for average-reward Step 5/6 primitives."""

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from alberta_framework import (
    DifferentialSARSAAgent as TopLevelDifferentialSARSAAgent,
)
from alberta_framework.core import DifferentialTDLearner as CoreDifferentialTDLearner
from alberta_framework.core.average_reward import (
    AverageRewardHordeActorCriticAgent,
    AverageRewardHordeActorCriticConfig,
    AverageRewardHordeLearner,
    DifferentialGTDConfig,
    DifferentialGTDLearner,
    DifferentialSARSAAgent,
    DifferentialSARSAConfig,
    DifferentialTDConfig,
    DifferentialTDLearner,
    run_average_reward_horde_actor_critic_from_arrays,
    run_average_reward_horde_from_arrays,
    run_differential_gtd_from_arrays,
    run_differential_sarsa_from_arrays,
    run_differential_td_from_arrays,
)


def test_differential_td_config_and_top_level_exports() -> None:
    config = DifferentialTDConfig(
        step_size=0.1,
        average_reward_step_size=0.02,
        trace_decay=0.5,
    )
    learner = DifferentialTDLearner.from_config(
        DifferentialTDLearner(config).to_config()
    )

    assert learner.config == config
    assert CoreDifferentialTDLearner is DifferentialTDLearner
    assert TopLevelDifferentialSARSAAgent is DifferentialSARSAAgent


def test_differential_td_error_matches_average_reward_target() -> None:
    learner = DifferentialTDLearner(DifferentialTDConfig(step_size=0.0))
    state = learner.init(2).replace(  # type: ignore[attr-defined]
        weights=jnp.array([1.0, -1.0], dtype=jnp.float32),
        bias=jnp.array(0.5, dtype=jnp.float32),
        average_reward=jnp.array(0.25, dtype=jnp.float32),
    )
    obs = jnp.array([2.0, 1.0], dtype=jnp.float32)
    next_obs = jnp.array([0.0, 3.0], dtype=jnp.float32)

    td_error = learner.td_error(
        state,
        obs,
        jnp.array(1.25, dtype=jnp.float32),
        next_obs,
    )

    chex.assert_trees_all_close(td_error, jnp.array(-3.0, dtype=jnp.float32))


def test_differential_td_update_moves_average_reward_and_is_jittable() -> None:
    learner = DifferentialTDLearner(
        DifferentialTDConfig(
            step_size=0.1,
            average_reward_step_size=0.2,
            trace_decay=0.0,
        )
    )
    state = learner.init(1)
    update = jax.jit(learner.update)
    result = update(
        state,
        jnp.array([1.0], dtype=jnp.float32),
        jnp.array(1.0, dtype=jnp.float32),
        jnp.array([1.0], dtype=jnp.float32),
    )

    chex.assert_trees_all_close(
        result.average_reward,
        jnp.array(0.2, dtype=jnp.float32),
    )
    assert int(result.state.step_count) == 1
    chex.assert_tree_all_finite(result)


def test_differential_td_scan_shapes_and_finite_metrics() -> None:
    learner = DifferentialTDLearner(DifferentialTDConfig(trace_decay=0.2))
    state = learner.init(2)
    observations = jnp.array(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        dtype=jnp.float32,
    )
    next_observations = jnp.array(
        [[0.0, 1.0], [1.0, 1.0], [1.0, -1.0]],
        dtype=jnp.float32,
    )
    rewards = jnp.array([0.0, 1.0, 0.5], dtype=jnp.float32)

    result = run_differential_td_from_arrays(
        learner,
        state,
        observations,
        rewards,
        next_observations,
    )

    chex.assert_shape(result.predictions, (3,))
    chex.assert_shape(result.td_errors, (3,))
    chex.assert_shape(result.average_rewards, (3,))
    chex.assert_shape(result.metrics, (3, 4))
    assert int(result.state.step_count) == 3
    chex.assert_tree_all_finite(
        (result.predictions, result.td_errors, result.average_rewards, result.metrics)
    )


def test_differential_gtd_config_roundtrip_and_ratio_clipping() -> None:
    config = DifferentialGTDConfig(
        value_step_size=0.1,
        secondary_step_size=0.05,
        average_reward_step_size=0.02,
        trace_decay=0.3,
        ratio_clip=1.5,
    )
    learner = DifferentialGTDLearner.from_config(
        DifferentialGTDLearner(config).to_config()
    )
    state = learner.init(1)

    result = learner.update(
        state,
        jnp.array([1.0], dtype=jnp.float32),
        jnp.array(1.0, dtype=jnp.float32),
        jnp.array([1.0], dtype=jnp.float32),
        jnp.array(3.0, dtype=jnp.float32),
    )

    assert learner.config == config
    chex.assert_trees_all_close(result.rho_clipped, jnp.array(1.5, dtype=jnp.float32))
    assert int(result.state.step_count) == 1
    chex.assert_tree_all_finite(result)


def test_differential_gtd_scan_learns_average_reward_cycle() -> None:
    learner = DifferentialGTDLearner(
        DifferentialGTDConfig(
            value_step_size=0.05,
            secondary_step_size=0.01,
            average_reward_step_size=0.01,
            trace_decay=0.0,
            ratio_clip=2.0,
        )
    )
    rewards_by_state = jnp.array([0.0, 1.0, 2.0], dtype=jnp.float32)
    steps = 20_000
    states = jnp.arange(steps, dtype=jnp.int32) % 3
    next_states = (states + 1) % 3
    observations = jnp.eye(3, dtype=jnp.float32)[states]
    next_observations = jnp.eye(3, dtype=jnp.float32)[next_states]
    rewards = rewards_by_state[states]
    rhos = jnp.ones((steps,), dtype=jnp.float32)
    state = learner.init(3)

    result = run_differential_gtd_from_arrays(
        learner,
        state,
        observations,
        rewards,
        next_observations,
        rhos,
    )

    predictions = learner.predict(result.state, jnp.eye(3, dtype=jnp.float32))
    centered_predictions = predictions - jnp.mean(predictions)
    true_values = jnp.array([-2.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=jnp.float32)
    chex.assert_trees_all_close(
        result.state.average_reward,
        jnp.array(1.0, dtype=jnp.float32),
        atol=2e-2,
    )
    chex.assert_trees_all_close(centered_predictions, true_values, atol=5e-2)
    assert float(jnp.mean(result.td_errors[-1000:] ** 2)) <= 2e-3
    chex.assert_tree_all_finite(result)


def test_average_reward_horde_shared_trunk_scan_learns_reward_rates() -> None:
    learner = AverageRewardHordeLearner(
        n_demons=2,
        hidden_sizes=(8,),
        step_size=0.02,
        average_reward_step_size=0.01,
        sparsity=0.0,
        use_layer_norm=False,
    )
    restored = AverageRewardHordeLearner.from_config(learner.to_config())
    assert restored.n_demons == 2

    steps = 20_000
    states = jnp.arange(steps, dtype=jnp.int32) % 3
    next_states = (states + 1) % 3
    observations = jnp.eye(3, dtype=jnp.float32)[states]
    next_observations = jnp.eye(3, dtype=jnp.float32)[next_states]
    cumulants = jnp.stack(
        [
            jnp.array([0.0, 1.0, 2.0], dtype=jnp.float32)[states],
            jnp.array([2.0, 1.0, 0.0], dtype=jnp.float32)[states],
        ],
        axis=1,
    )
    state = learner.init(3, jr.key(0))

    result = run_average_reward_horde_from_arrays(
        learner,
        state,
        observations,
        cumulants,
        next_observations,
    )

    chex.assert_trees_all_close(
        result.state.average_rewards,
        jnp.array([1.0, 1.0], dtype=jnp.float32),
        atol=3e-2,
    )
    assert float(jnp.mean(result.td_errors[-1000:] ** 2)) <= 5e-3
    chex.assert_tree_all_finite(result)


def test_average_reward_horde_actor_critic_single_update_is_finite() -> None:
    agent = AverageRewardHordeActorCriticAgent(
        AverageRewardHordeActorCriticConfig(
            n_actions=2,
            hidden_sizes=(4,),
            critic_step_size=0.02,
            average_reward_step_size=0.01,
        )
    )
    restored = AverageRewardHordeActorCriticAgent.from_config(agent.to_config())
    assert restored.config == agent.config
    assert type(restored.actor_optimizer) is type(agent.actor_optimizer)
    state = agent.init(2, jr.key(0))
    state, action = agent.start(state, jnp.array([1.0, 0.0], dtype=jnp.float32))

    result = agent.update(
        state,
        jnp.array(1.0, dtype=jnp.float32),
        jnp.array([0.0, 1.0], dtype=jnp.float32),
    )

    assert int(action) in (0, 1)
    assert int(result.action) in (0, 1)
    assert int(result.state.step_count) == 1
    chex.assert_tree_all_finite(
        (
            result.policy,
            result.target_policy,
            result.behavior_action_probability,
            result.target_action_probability,
            result.actor_score_scale,
            result.td_error,
            result.average_reward,
            result.critic_prediction,
            result.state.actor_weights,
            result.state.actor_bias,
            result.state.critic_state.average_rewards,
        )
    )


def test_average_reward_actor_critic_behavior_policy_is_exact_epsilon_mixture() -> None:
    config = AverageRewardHordeActorCriticConfig(
        n_actions=3,
        hidden_sizes=(4,),
        epsilon=0.25,
    )
    agent = AverageRewardHordeActorCriticAgent(config)
    state = agent.init(2, jr.key(0)).replace(
        actor_bias=jnp.array([1.0, -0.5, 0.25], dtype=jnp.float32)
    )
    observation = jnp.array([0.5, -0.25], dtype=jnp.float32)

    target = agent.policy(state, observation)
    behavior = agent.behavior_policy(state, observation)
    expected = 0.75 * target + 0.25 / 3.0
    chex.assert_trees_all_close(behavior, expected)
    assert float(jnp.sum(behavior)) == pytest.approx(1.0)

    sample, _ = agent.sample_policy(state, observation)
    chex.assert_trees_all_close(sample.target_policy, target)
    chex.assert_trees_all_close(sample.behavior_policy, behavior)
    chex.assert_trees_all_close(
        sample.target_probability,
        target[sample.action],
    )
    chex.assert_trees_all_close(
        sample.behavior_probability,
        behavior[sample.action],
    )
    chex.assert_trees_all_close(
        sample.target_log_probability,
        jnp.log(sample.target_probability),
    )
    chex.assert_trees_all_close(
        sample.behavior_log_probability,
        jnp.log(sample.behavior_probability),
    )


@pytest.mark.parametrize("epsilon", [0.0, 0.3, 1.0])
def test_average_reward_actor_critic_score_matches_mixture_derivative(
    epsilon: float,
) -> None:
    config = AverageRewardHordeActorCriticConfig(
        n_actions=2,
        hidden_sizes=(4,),
        critic_step_size=0.0,
        average_reward_step_size=0.0,
        epsilon=epsilon,
        temperature=0.7,
    )
    agent = AverageRewardHordeActorCriticAgent(config)
    initial = agent.init(2, jr.key(4)).replace(
        actor_bias=jnp.array([1.0, -1.0], dtype=jnp.float32)
    )
    observation = jnp.array([1.0, -0.5], dtype=jnp.float32)
    state, _ = agent.start(initial, observation)
    stored = state.last_policy_sample

    result = agent.update(
        state,
        jnp.array(1.0, dtype=jnp.float32),
        jnp.array([-0.25, 0.75], dtype=jnp.float32),
    )
    expected_scale = (
        (1.0 - epsilon)
        * stored.target_probability
        / stored.behavior_probability
    )
    chex.assert_trees_all_close(result.actor_score_scale, expected_scale)

    # Finite-difference d log(mu_a) / d target-logit_a agrees with the
    # analytical mixture score's selected-action component.
    action = int(stored.action)
    bias = initial.actor_bias

    def selected_log_behavior(selected_bias):
        logits = bias.at[action].set(selected_bias)
        target = jax.nn.softmax(logits / config.temperature)
        behavior = (1.0 - epsilon) * target + epsilon / config.n_actions
        return jnp.log(behavior[action])

    finite_difference = jax.grad(selected_log_behavior)(bias[action])
    expected_component = expected_scale * (
        1.0 - stored.target_policy[action]
    ) / config.temperature
    chex.assert_trees_all_close(
        finite_difference,
        expected_component,
        atol=1e-6,
    )


def test_epsilon_one_freezes_actor_but_not_critic() -> None:
    config = AverageRewardHordeActorCriticConfig(
        n_actions=2,
        hidden_sizes=(4,),
        critic_step_size=0.02,
        average_reward_step_size=0.01,
        epsilon=1.0,
    )
    agent = AverageRewardHordeActorCriticAgent(config)
    initial = agent.init(2, jr.key(9)).replace(
        actor_weights=jnp.array(
            [
                [0.2, -0.1, 0.3, 0.4],
                [-0.2, 0.5, -0.4, 0.1],
            ],
            dtype=jnp.float32,
        ),
        actor_bias=jnp.array([0.7, -0.2], dtype=jnp.float32),
    )
    state, _ = agent.start(initial, jnp.array([1.0, 0.0], dtype=jnp.float32))
    result = agent.update(
        state,
        jnp.array(2.0, dtype=jnp.float32),
        jnp.array([0.0, 1.0], dtype=jnp.float32),
    )

    chex.assert_trees_all_equal(result.state.actor_weights, state.actor_weights)
    chex.assert_trees_all_equal(result.state.actor_bias, state.actor_bias)
    chex.assert_trees_all_equal(result.state.actor_opt_w, state.actor_opt_w)
    chex.assert_trees_all_equal(result.state.actor_opt_b, state.actor_opt_b)
    chex.assert_trees_all_close(
        result.policy,
        jnp.array([0.5, 0.5], dtype=jnp.float32),
    )
    assert float(result.actor_score_scale) == pytest.approx(0.0)
    assert not jnp.allclose(
        result.state.critic_state.average_rewards,
        state.critic_state.average_rewards,
    )


def test_update_logs_policy_that_sampled_next_action_before_parameter_update() -> None:
    config = AverageRewardHordeActorCriticConfig(
        n_actions=2,
        hidden_sizes=(4,),
        critic_step_size=0.0,
        average_reward_step_size=0.0,
        epsilon=0.2,
        actor_update_clip=1.0,
    )
    agent = AverageRewardHordeActorCriticAgent(config)
    state, _ = agent.start(
        agent.init(2, jr.key(12)),
        jnp.array([1.0, 0.0], dtype=jnp.float32),
    )
    next_observation = jnp.array([0.0, 1.0], dtype=jnp.float32)
    expected_sample, _ = agent.sample_policy(state, next_observation)
    result = agent.update(state, 10.0, next_observation)

    chex.assert_trees_all_equal(result.action, expected_sample.action)
    chex.assert_trees_all_close(result.policy, expected_sample.behavior_policy)
    chex.assert_trees_all_close(
        result.target_policy,
        expected_sample.target_policy,
    )
    chex.assert_trees_all_close(
        result.behavior_action_probability,
        expected_sample.behavior_probability,
    )
    chex.assert_trees_all_close(
        result.target_action_probability,
        expected_sample.target_probability,
    )
    chex.assert_trees_all_close(
        result.state.last_policy_sample,
        expected_sample,
    )


def test_average_reward_actor_critic_scan_logs_action_probabilities() -> None:
    agent = AverageRewardHordeActorCriticAgent(
        AverageRewardHordeActorCriticConfig(
            n_actions=3,
            hidden_sizes=(4,),
            epsilon=0.2,
        )
    )
    state, _ = agent.start(
        agent.init(2, jr.key(17)),
        jnp.array([1.0, 0.0], dtype=jnp.float32),
    )
    result = run_average_reward_horde_actor_critic_from_arrays(
        agent,
        state,
        jnp.array([1.0, -0.5, 0.25], dtype=jnp.float32),
        jnp.array(
            [
                [0.0, 1.0],
                [-1.0, 0.0],
                [0.5, 0.5],
            ],
            dtype=jnp.float32,
        ),
    )

    row = jnp.arange(result.actions.shape[0])
    chex.assert_trees_all_close(
        result.behavior_action_probabilities,
        result.policies[row, result.actions],
    )
    chex.assert_trees_all_close(
        result.target_action_probabilities,
        result.target_policies[row, result.actions],
    )
    chex.assert_trees_all_close(
        jnp.sum(result.policies, axis=1),
        jnp.ones(3),
    )
    chex.assert_tree_all_finite(
        (
            result.actions,
            result.policies,
            result.target_policies,
            result.behavior_action_probabilities,
            result.target_action_probabilities,
            result.actor_score_scales,
            result.td_errors,
            result.average_rewards,
        )
    )


def test_differential_sarsa_config_roundtrip_and_exact_td_error() -> None:
    config = DifferentialSARSAConfig(
        n_actions=2,
        q_step_size=0.0,
        average_reward_step_size=0.0,
        epsilon_start=0.0,
    )
    agent = DifferentialSARSAAgent.from_config(DifferentialSARSAAgent(config).to_config())
    state = agent.init(2, jr.key(0)).replace(  # type: ignore[attr-defined]
        q_weights=jnp.array([[1.0, 0.0], [0.0, 2.0]], dtype=jnp.float32),
        q_bias=jnp.array([0.5, -0.5], dtype=jnp.float32),
        average_reward=jnp.array(0.25, dtype=jnp.float32),
        last_observation=jnp.array([2.0, 1.0], dtype=jnp.float32),
        last_action=jnp.array(0, dtype=jnp.int32),
    )
    next_obs = jnp.array([1.0, 3.0], dtype=jnp.float32)

    result = agent.update(
        state,
        jnp.array(2.0, dtype=jnp.float32),
        next_obs,
        next_action=jnp.array(1, dtype=jnp.int32),
    )

    assert agent.config == config
    chex.assert_trees_all_close(result.td_error, jnp.array(4.75, dtype=jnp.float32))
    chex.assert_trees_all_close(result.average_reward, state.average_reward)


def test_differential_sarsa_update_and_scan_are_finite() -> None:
    agent = DifferentialSARSAAgent(
        DifferentialSARSAConfig(
            n_actions=3,
            q_step_size=0.05,
            average_reward_step_size=0.01,
            trace_decay=0.2,
            epsilon_start=0.2,
        )
    )
    state = agent.init(2, jr.key(1))
    state, _ = agent.start(state, jnp.array([1.0, 0.0], dtype=jnp.float32))
    rewards = jnp.array([1.0, 0.0, 0.5, -0.25], dtype=jnp.float32)
    next_observations = jnp.array(
        [[0.0, 1.0], [1.0, 1.0], [0.5, -0.5], [1.0, 0.0]],
        dtype=jnp.float32,
    )

    result = run_differential_sarsa_from_arrays(
        agent,
        state,
        rewards,
        next_observations,
    )

    chex.assert_shape(result.q_values, (4, 3))
    chex.assert_shape(result.td_errors, (4,))
    chex.assert_shape(result.average_rewards, (4,))
    chex.assert_shape(result.actions, (4,))
    assert int(result.state.step_count) == 4
    chex.assert_tree_all_finite(
        (result.q_values, result.td_errors, result.average_rewards)
    )
    assert bool(jnp.all(result.actions >= 0))
    assert bool(jnp.all(result.actions < 3))


def test_differential_sarsa_learns_better_action_on_continuing_bandit() -> None:
    agent = DifferentialSARSAAgent(
        DifferentialSARSAConfig(
            n_actions=2,
            q_step_size=0.04,
            average_reward_step_size=0.01,
            trace_decay=0.0,
            epsilon_start=0.1,
            epsilon_end=0.02,
            epsilon_decay_steps=200,
        )
    )
    obs = jnp.array([1.0], dtype=jnp.float32)
    state = agent.init(1, jr.key(42))
    state, _ = agent.start(state, obs)

    for _ in range(800):
        reward = jnp.asarray(state.last_action == 1, dtype=jnp.float32)
        result = agent.update(state, reward, obs)
        state = result.state

    q_values = agent.q_values(state, obs)
    assert float(q_values[1]) > float(q_values[0]) + 0.25
    assert float(state.average_reward) > 0.75
