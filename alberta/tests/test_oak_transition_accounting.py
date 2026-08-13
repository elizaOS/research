"""Transition-ownership regressions for OaK option utility accounting."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.oak import OaKAgent, OaKConfig
from alberta_framework.core.options import STOMPConfig, SubtaskSpec

OBS = jnp.array([1.0, 0.0], dtype=jnp.float32)
N_PRIMITIVE = 2
OPTION_0_ACTION = N_PRIMITIVE
OPTION_1_ACTION = N_PRIMITIVE + 1
PSEUDO_REWARD = 4.0


def test_initial_option_selected_at_start_is_counted() -> None:
    config = OaKConfig(
        stomp=STOMPConfig(
            subtask_specs=(
                SubtaskSpec(feature_index=0),
                SubtaskSpec(feature_index=1),
            ),
            observation_dim=2,
            n_primitive_actions=N_PRIMITIVE,
            epsilon_base=0.0,
            epsilon_option=0.0,
        )
    )
    agent = OaKAgent(config)
    state = agent.init(jr.key(4))
    option_1_action = N_PRIMITIVE + 1
    weights = tuple(
        jnp.array(
            [[10.0 if action == option_1_action else -10.0, 0.0]],
            dtype=jnp.float32,
        )
        for action in range(config.stomp.n_total_actions)
    )
    state = state.replace(
        stomp_state=state.stomp_state.replace(
            base_learner_state=state.stomp_state.base_learner_state.replace(
                head_params=state.stomp_state.base_learner_state.head_params.replace(
                    weights=weights
                )
            )
        )
    )

    started = agent.start(state, OBS)

    assert int(started.stomp_state.executing_option) == 1
    np.testing.assert_array_equal(
        np.asarray(started.execution_counts),
        np.array([0, 1], dtype=np.int32),
    )


def _prepared_state(desired_next_extended_action: int):
    config = OaKConfig(
        stomp=STOMPConfig(
            subtask_specs=(
                SubtaskSpec(
                    feature_index=0,
                    threshold=1.0e6,
                    pseudo_reward_scale=PSEUDO_REWARD,
                    max_option_steps=1,
                ),
                SubtaskSpec(
                    feature_index=1,
                    threshold=1.0e6,
                    max_option_steps=1,
                ),
            ),
            observation_dim=2,
            n_primitive_actions=N_PRIMITIVE,
            base_step_size=0.05,
            base_trace_decay=0.0,
            epsilon_base=0.0,
            epsilon_option=0.0,
            option_gamma=0.5,
        ),
        utility_ema_decay=0.5,
    )
    agent = OaKAgent(config)
    state = agent.start(agent.init(jr.key(0)), OBS)

    # Make the desired post-termination extended action unambiguous. The
    # terminating real backup can move option 0's head slightly but cannot
    # overcome this 200-point action gap.
    head_weights = tuple(
        jnp.array(
            [[100.0 if action == desired_next_extended_action else -100.0, 0.0]],
            dtype=jnp.float32,
        )
        for action in range(config.stomp.n_total_actions)
    )
    learner_state = state.stomp_state.base_learner_state.replace(
        head_params=state.stomp_state.base_learner_state.head_params.replace(
            weights=head_weights,
            biases=tuple(
                jnp.zeros_like(b)
                for b in state.stomp_state.base_learner_state.head_params.biases
            ),
        )
    )
    stomp_state = state.stomp_state.replace(
        base_learner_state=learner_state,
        base_average_reward=jnp.array(0.0, dtype=jnp.float32),
        base_last_obs=OBS,
        base_last_action=jnp.array(OPTION_0_ACTION, dtype=jnp.int32),
        last_primitive_action=jnp.array(0, dtype=jnp.int32),
        executing_option=jnp.array(0, dtype=jnp.int32),
        option_start_obs=OBS,
        option_last_intra_action=jnp.array(0, dtype=jnp.int32),
        option_cumreward=jnp.array(0.0, dtype=jnp.float32),
        option_env_cumreward=jnp.array(0.0, dtype=jnp.float32),
        option_baseline_mass=jnp.array(0.0, dtype=jnp.float32),
        option_discount=jnp.array(1.0, dtype=jnp.float32),
        option_steps=jnp.array(0, dtype=jnp.int32),
        step_count=jnp.array(11, dtype=jnp.int32),
        step_words=jnp.array([0, 11], dtype=jnp.uint32),
    )
    state = state.replace(
        stomp_state=stomp_state,
        execution_counts=jnp.array([7, 11], dtype=jnp.int32),
        cumulative_pseudo_rewards=jnp.array([2.0, 3.0], dtype=jnp.float32),
        utility_ema=jnp.array([0.2, 0.6], dtype=jnp.float32),
        step_count=jnp.array(11, dtype=jnp.int32),
        step_words=jnp.array([0, 11], dtype=jnp.uint32),
    )
    return agent, state


@pytest.mark.parametrize(
    ("desired_action", "expected_post_option", "expected_counts"),
    [
        (1, -1, np.array([7, 11], dtype=np.int32)),
        (OPTION_1_ACTION, 1, np.array([7, 12], dtype=np.int32)),
        (OPTION_0_ACTION, 0, np.array([8, 11], dtype=np.int32)),
    ],
    ids=("terminate-to-primitive", "terminate-to-different", "terminate-to-same"),
)
def test_terminal_pseudo_reward_credits_prior_option_and_counts_restart(
    desired_action: int,
    expected_post_option: int,
    expected_counts: np.ndarray,
) -> None:
    """Terminal credit stays with option 0 regardless of the next selection."""
    agent, state = _prepared_state(desired_action)

    result = agent.update(state, jnp.array(0.0, dtype=jnp.float32), OBS)
    after = result.state

    assert bool(result.option_terminated)
    assert int(after.stomp_state.executing_option) == expected_post_option
    np.testing.assert_allclose(float(result.pseudo_reward), PSEUDO_REWARD)

    # Utility and cumulative pseudo-reward belong to the option that generated
    # the transition, not the post-transition option.
    np.testing.assert_allclose(
        np.asarray(after.utility_ema),
        np.array([2.1, 0.6], dtype=np.float32),
        rtol=0.0,
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        np.asarray(after.cumulative_pseudo_rewards),
        np.array([6.0, 3.0], dtype=np.float32),
        rtol=0.0,
        atol=1.0e-6,
    )
    np.testing.assert_array_equal(
        np.asarray(after.execution_counts),
        expected_counts,
    )
