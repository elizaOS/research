"""Reward-channel and semi-MDP discount regressions for STOMP base control."""

import jax.numpy as jnp
import jax.random as jr
import numpy as np

from alberta_framework.core.options import STOMPAgent, STOMPConfig, SubtaskSpec

OBS_DIM = 3
N_PRIMITIVE = 2
OPTION_ACTION = N_PRIMITIVE


def test_option_base_target_uses_discounted_environment_return() -> None:
    """An option's base target uses task reward; its model keeps pseudo-reward."""
    gamma = 0.5
    model_decay = 0.8
    config = STOMPConfig(
        subtask_specs=(
            SubtaskSpec(
                feature_index=0,
                threshold=1.0e6,
                pseudo_reward_scale=20.0,
                max_option_steps=3,
            ),
        ),
        observation_dim=OBS_DIM,
        n_primitive_actions=N_PRIMITIVE,
        option_gamma=gamma,
        option_model_decay=model_decay,
        epsilon_base=0.0,
    )
    agent = STOMPAgent(config)
    learner = agent.base_learner
    observations = jnp.array(
        [
            [1.0, 0.1, -0.1],
            [10.0, 0.2, -0.2],
            [20.0, 0.3, -0.3],
            [30.0, 0.4, -0.4],
        ],
        dtype=jnp.float32,
    )
    env_rewards = jnp.array([2.0, 3.0, 5.0], dtype=jnp.float32)
    average_reward = jnp.array(0.4, dtype=jnp.float32)

    state = agent.start(agent.init(jr.key(0)), observations[0]).replace(
        executing_option=jnp.array(0, dtype=jnp.int32),
        base_last_obs=observations[0],
        base_last_action=jnp.array(OPTION_ACTION, dtype=jnp.int32),
        base_average_reward=average_reward,
        option_start_obs=observations[0],
        option_cumreward=jnp.array(0.0, dtype=jnp.float32),
        option_env_cumreward=jnp.array(0.0, dtype=jnp.float32),
        option_baseline_mass=jnp.array(0.0, dtype=jnp.float32),
        option_discount=jnp.array(1.0, dtype=jnp.float32),
        option_steps=jnp.array(0, dtype=jnp.int32),
    )

    for step in range(2):
        result = agent.update(state, env_rewards[step], observations[step + 1])
        state = result.state
        assert not bool(result.option_terminated)
        expected_partial_return = sum(gamma**k * float(env_rewards[k]) for k in range(step + 1))
        np.testing.assert_allclose(
            float(state.option_env_cumreward),
            expected_partial_return,
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            float(state.option_discount),
            gamma ** (step + 1),
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            float(state.option_baseline_mass),
            sum(gamma**k for k in range(step + 1)),
            rtol=1e-6,
            atol=1e-6,
        )

    pre_update_state = state
    q_last = learner.predict(
        pre_update_state.base_learner_state, pre_update_state.option_start_obs
    )[OPTION_ACTION]
    max_next_q = jnp.max(learner.predict(pre_update_state.base_learner_state, observations[3]))
    discounted_env_return = sum(gamma**k * env_rewards[k] for k in range(len(env_rewards)))
    discounted_baseline_mass = sum(gamma**k for k in range(len(env_rewards)))
    expected_td = (
        discounted_env_return
        - average_reward * discounted_baseline_mass
        + gamma ** len(env_rewards) * max_next_q
        - q_last
    )

    result = agent.update(state, env_rewards[2], observations[3])

    assert bool(result.option_terminated)
    np.testing.assert_allclose(float(result.td_error), float(expected_td), rtol=1e-5, atol=1e-5)

    pseudo_return = jnp.sum(config.subtask_specs[0].pseudo_reward_scale * observations[1:, 0])
    expected_model_cumulant = (1.0 - model_decay) * pseudo_return
    np.testing.assert_allclose(
        float(result.state.option_models.cumreward_ema[0]),
        float(expected_model_cumulant),
        rtol=1e-5,
        atol=1e-5,
    )
    wrong_pseudo_td = (
        pseudo_return
        - average_reward * discounted_baseline_mass
        + gamma ** len(env_rewards) * max_next_q
        - q_last
    )
    assert not np.isclose(float(result.td_error), float(wrong_pseudo_td))


def test_option_head_update_is_credited_to_option_start_state() -> None:
    """A multi-step option changes only features active where it was selected."""
    config = STOMPConfig(
        subtask_specs=(
            SubtaskSpec(
                feature_index=0,
                threshold=1.0e6,
                max_option_steps=2,
            ),
        ),
        observation_dim=OBS_DIM,
        n_primitive_actions=N_PRIMITIVE,
        base_step_size=0.1,
        base_trace_decay=0.0,
        option_gamma=1.0,
        epsilon_base=0.0,
    )
    agent = STOMPAgent(config)
    start_obs = jnp.array([1.0, 0.0, 0.0], dtype=jnp.float32)
    penultimate_obs = jnp.array([0.0, 1.0, 0.0], dtype=jnp.float32)
    terminal_obs = jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32)

    state = agent.start(agent.init(jr.key(2)), start_obs).replace(
        executing_option=jnp.array(0, dtype=jnp.int32),
        base_last_obs=start_obs,
        base_last_action=jnp.array(OPTION_ACTION, dtype=jnp.int32),
        option_start_obs=start_obs,
        option_cumreward=jnp.array(0.0, dtype=jnp.float32),
        option_env_cumreward=jnp.array(0.0, dtype=jnp.float32),
        option_baseline_mass=jnp.array(0.0, dtype=jnp.float32),
        option_discount=jnp.array(1.0, dtype=jnp.float32),
        option_steps=jnp.array(0, dtype=jnp.int32),
    )

    first = agent.update(state, jnp.array(0.0), penultimate_obs)
    assert not bool(first.option_terminated)
    np.testing.assert_array_equal(
        np.asarray(first.state.base_last_obs), np.asarray(penultimate_obs)
    )
    before_weights = tuple(
        np.asarray(weight).copy() for weight in first.state.base_learner_state.head_params.weights
    )

    result = agent.update(first.state, jnp.array(10.0), terminal_obs)

    assert bool(result.option_terminated)
    after_weights = result.state.base_learner_state.head_params.weights
    option_delta = np.asarray(after_weights[OPTION_ACTION]) - before_weights[OPTION_ACTION]
    assert abs(float(option_delta[0, 0])) > 1.0e-4
    np.testing.assert_allclose(option_delta[0, 1:], 0.0, rtol=0.0, atol=1.0e-8)
    for head in range(N_PRIMITIVE):
        np.testing.assert_array_equal(np.asarray(after_weights[head]), before_weights[head])


def test_primitive_base_target_ignores_option_reward_state() -> None:
    """A primitive transition remains a one-step environment-reward update."""
    config = STOMPConfig(
        subtask_specs=(
            SubtaskSpec(
                feature_index=0,
                threshold=1.0e6,
                pseudo_reward_scale=100.0,
                max_option_steps=8,
            ),
        ),
        observation_dim=OBS_DIM,
        n_primitive_actions=N_PRIMITIVE,
        option_gamma=0.25,
        epsilon_base=0.0,
    )
    agent = STOMPAgent(config)
    learner = agent.base_learner
    observations = jnp.array([[1.0, 0.2, 0.3], [2.0, 0.4, 0.8]], dtype=jnp.float32)
    env_reward = jnp.array(7.0, dtype=jnp.float32)
    average_reward = jnp.array(0.75, dtype=jnp.float32)
    primitive_action = jnp.array(1, dtype=jnp.int32)

    state = agent.start(agent.init(jr.key(1)), observations[0]).replace(
        executing_option=jnp.array(-1, dtype=jnp.int32),
        base_last_obs=observations[0],
        base_last_action=primitive_action,
        base_average_reward=average_reward,
        # Poison all option statistics: none may enter the primitive target.
        option_cumreward=jnp.array(-456.0, dtype=jnp.float32),
        option_env_cumreward=jnp.array(123.0, dtype=jnp.float32),
        option_baseline_mass=jnp.array(321.0, dtype=jnp.float32),
        option_discount=jnp.array(0.01, dtype=jnp.float32),
        option_steps=jnp.array(9, dtype=jnp.int32),
    )
    q_last = learner.predict(state.base_learner_state, state.base_last_obs)[primitive_action]
    max_next_q = jnp.max(learner.predict(state.base_learner_state, observations[1]))
    expected_td = env_reward - average_reward + max_next_q - q_last

    result = agent.update(state, env_reward, observations[1])

    assert not bool(result.option_terminated)
    np.testing.assert_allclose(float(result.td_error), float(expected_td), rtol=1e-5, atol=1e-5)
