"""External scored-action start contract for differential SARSA."""

from __future__ import annotations

from typing import Any

import chex
import jax
import jax.numpy as jnp
import pytest

from alberta_framework.core.average_reward import (
    DifferentialSARSAAgent,
    DifferentialSARSAConfig,
)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("n_actions", True),
        ("q_step_size", float("nan")),
        ("q_step_size", float("inf")),
        ("average_reward_step_size", float("nan")),
        ("trace_decay", float("nan")),
        ("epsilon_start", float("inf")),
        ("epsilon_end", float("nan")),
        ("epsilon_decay_steps", True),
        ("use_bias", 1),
    ],
)
def test_config_rejects_nonfinite_or_ambiguous_values(
    field: str,
    value: Any,
) -> None:
    kwargs: dict[str, Any] = {"n_actions": 2, field: value}
    with pytest.raises(ValueError):
        DifferentialSARSAConfig(**kwargs)


def test_start_with_action_stores_external_decision_without_other_state_change() -> None:
    agent = DifferentialSARSAAgent(
        DifferentialSARSAConfig(
            n_actions=2,
            epsilon_start=0.3,
            epsilon_end=0.3,
        )
    )
    state = agent.init(3, jax.random.key(9), average_reward=0.2)
    observation = jnp.array([1.0, -0.5, 0.25], dtype=jnp.float32)
    before_key = jax.random.key_data(state.rng_key)
    started, action = jax.jit(agent.start_with_action)(
        state,
        observation,
        jnp.asarray(1, dtype=jnp.int32),
    )

    assert int(action) == 1
    assert int(started.last_action) == 1
    chex.assert_trees_all_close(started.last_observation, observation)
    chex.assert_trees_all_equal(jax.random.key_data(started.rng_key), before_key)
    chex.assert_trees_all_equal(started.q_weights, state.q_weights)
    chex.assert_trees_all_equal(started.q_trace_weights, state.q_trace_weights)
    chex.assert_trees_all_equal(started.average_reward, state.average_reward)
    chex.assert_trees_all_equal(started.epsilon, state.epsilon)
    assert int(started.step_count) == 0


def test_explicit_next_action_path_never_consumes_internal_rng() -> None:
    agent = DifferentialSARSAAgent(
        DifferentialSARSAConfig(
            n_actions=2,
            q_step_size=0.1,
            average_reward_step_size=0.01,
            epsilon_start=0.5,
            epsilon_end=0.5,
        )
    )
    state = agent.init(2, jax.random.key(10))
    state, _ = agent.start_with_action(
        state,
        jnp.array([1.0, 0.0], dtype=jnp.float32),
        jnp.asarray(0, dtype=jnp.int32),
    )
    before_key = jax.random.key_data(state.rng_key)
    result = agent.update(
        state,
        jnp.asarray(1.0),
        jnp.array([0.0, 1.0], dtype=jnp.float32),
        next_action=jnp.asarray(1, dtype=jnp.int32),
    )

    assert int(result.action) == 1
    chex.assert_trees_all_equal(
        jax.random.key_data(result.state.rng_key),
        before_key,
    )
    assert int(result.state.step_count) == 1


def test_external_action_start_composes_with_vmap() -> None:
    agent = DifferentialSARSAAgent(DifferentialSARSAConfig(n_actions=2))
    state = agent.init(2, jax.random.key(11))
    observations = jnp.eye(2, dtype=jnp.float32)
    actions = jnp.array([1, 0], dtype=jnp.int32)

    states, selected = jax.jit(jax.vmap(agent.start_with_action, in_axes=(None, 0, 0)))(
        state,
        observations,
        actions,
    )

    chex.assert_trees_all_equal(selected, actions)
    chex.assert_trees_all_equal(states.last_observation, observations)
    chex.assert_trees_all_equal(states.last_action, actions)
    chex.assert_tree_all_finite(
        (
            states.q_weights,
            states.q_trace_weights,
            states.average_reward,
            states.last_observation,
            states.epsilon,
        )
    )
