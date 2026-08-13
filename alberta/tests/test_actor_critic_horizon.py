"""Exact-horizon and atomicity contracts for the base actor-critic agents."""

from __future__ import annotations

import dataclasses
from typing import Any, Literal

import chex
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.actor_critic import (
    ACTOR_CRITIC_CONFIG_SCHEMA,
    ACTOR_CRITIC_STATE_SCHEMA,
    CONTINUOUS_ACTOR_CRITIC_CONFIG_SCHEMA,
    CONTINUOUS_ACTOR_CRITIC_STATE_SCHEMA,
    ActorCriticAgent,
    ActorCriticConfig,
    ContinuousActorCriticAgent,
    ContinuousActorCriticConfig,
    measure_actor_critic_state_nbytes,
    measure_continuous_actor_critic_state_nbytes,
    migrate_legacy_actor_critic_config,
    migrate_legacy_actor_critic_state,
    migrate_legacy_continuous_actor_critic_config,
    migrate_legacy_continuous_actor_critic_state,
    run_actor_critic_from_arrays,
    run_continuous_actor_critic_from_arrays,
)

pytestmark = pytest.mark.unit

_INT32_MAX = 2**31 - 1
_UINT32_MAX = 2**32 - 1
Kind = Literal["discrete", "continuous"]


def _case(kind: Kind) -> tuple[Any, Any]:
    if kind == "discrete":
        agent: Any = ActorCriticAgent(ActorCriticConfig(n_actions=2))
        state = agent.init(2, jr.key(1)).replace(
            last_observation=jnp.asarray([1.0, -0.5], dtype=jnp.float32),
            last_action=jnp.asarray(0, dtype=jnp.int32),
        )
        return agent, state
    agent = ContinuousActorCriticAgent(ContinuousActorCriticConfig(action_dim=1))
    state = agent.init(2, jr.key(2)).replace(
        last_observation=jnp.asarray([1.0, -0.5], dtype=jnp.float32),
        last_action=jnp.asarray([0.25], dtype=jnp.float32),
    )
    return agent, state


def _update(agent: Any, state: Any, *, terminal: bool = False) -> Any:
    return agent.update(
        state,
        jnp.asarray(1.0, dtype=jnp.float32),
        jnp.asarray([0.5, 0.25], dtype=jnp.float32),
        terminated=jnp.asarray(terminal),
    )


@pytest.mark.parametrize("kind", ["discrete", "continuous"])
def test_exact_clock_crosses_uint32_low_word_and_terminal_resets_traces(
    kind: Kind,
) -> None:
    agent, state = _case(kind)
    state = state.replace(
        step_count=jnp.asarray(_INT32_MAX, dtype=jnp.int32),
        step_words=jnp.asarray((0, _UINT32_MAX), dtype=jnp.uint32),
    )
    if kind == "discrete":
        state = state.replace(
            actor_trace_weights=jnp.ones_like(state.actor_trace_weights),
            actor_trace_bias=jnp.ones_like(state.actor_trace_bias),
            critic_trace_weights=jnp.ones_like(state.critic_trace_weights),
            critic_trace_bias=jnp.asarray(1.0, dtype=jnp.float32),
        )
    else:
        state = state.replace(
            mean_trace_weights=jnp.ones_like(state.mean_trace_weights),
            mean_trace_bias=jnp.ones_like(state.mean_trace_bias),
            log_sigma_trace=jnp.ones_like(state.log_sigma_trace),
            critic_trace_weights=jnp.ones_like(state.critic_trace_weights),
            critic_trace_bias=jnp.asarray(1.0, dtype=jnp.float32),
        )

    result = _update(agent, state, terminal=True)

    assert bool(result.update_applied)
    np.testing.assert_array_equal(result.pre_step_words, (0, _UINT32_MAX))
    np.testing.assert_array_equal(result.proposed_step_words, (1, 0))
    np.testing.assert_array_equal(result.post_step_words, (1, 0))
    assert int(result.state.step_count) == _INT32_MAX
    if kind == "discrete":
        assert bool(jnp.all(result.state.actor_trace_weights == 0.0))
        assert bool(jnp.all(result.state.critic_trace_weights == 0.0))
    else:
        assert bool(jnp.all(result.state.mean_trace_weights == 0.0))
        assert bool(jnp.all(result.state.log_sigma_trace == 0.0))
        assert bool(jnp.all(result.state.critic_trace_weights == 0.0))


@pytest.mark.parametrize("kind", ["discrete", "continuous"])
def test_all_ones_exhaustion_rolls_back_learning_clock_and_rng(kind: Kind) -> None:
    agent, state = _case(kind)
    exhausted = state.replace(
        step_count=jnp.asarray(_INT32_MAX, dtype=jnp.int32),
        step_words=jnp.asarray((_UINT32_MAX, _UINT32_MAX), dtype=jnp.uint32),
    )

    result = _update(agent, exhausted)

    assert bool(result.lifetime_counter_valid)
    assert not bool(result.lifetime_capacity_available)
    assert not bool(result.update_applied)
    chex.assert_trees_all_equal(result.state, exhausted)
    np.testing.assert_array_equal(result.post_step_words, exhausted.step_words)


@pytest.mark.parametrize("kind", ["discrete", "continuous"])
def test_corrupt_or_nonfinite_source_is_rejected_byte_exactly(kind: Kind) -> None:
    agent, state = _case(kind)
    corrupt_clock = state.replace(step_count=jnp.asarray(1, dtype=jnp.int32))
    clock_result = _update(agent, corrupt_clock)
    assert not bool(clock_result.lifetime_counter_valid)
    assert not bool(clock_result.update_applied)
    chex.assert_trees_all_equal(clock_result.state, corrupt_clock)

    corrupt_float = state.replace(critic_bias=jnp.asarray(jnp.nan, dtype=jnp.float32))
    float_result = _update(agent, corrupt_float)
    assert not bool(float_result.source_state_finite)
    assert not bool(float_result.state_valid)
    assert not bool(float_result.update_applied)
    chex.assert_trees_all_equal(float_result.state, corrupt_float)
    assert float(float_result.td_error) == 0.0


@pytest.mark.parametrize("kind", ["discrete", "continuous"])
def test_nonfinite_candidate_rolls_back_even_when_sources_are_finite(kind: Kind) -> None:
    agent, state = _case(kind)
    maximum = jnp.asarray(np.finfo(np.float32).max, dtype=jnp.float32)
    state = state.replace(last_observation=jnp.asarray([maximum, maximum], dtype=jnp.float32))
    if kind == "continuous":
        state = state.replace(last_action=jnp.asarray([maximum], dtype=jnp.float32))

    result = agent.update(
        state,
        maximum,
        jnp.zeros((2,), dtype=jnp.float32),
        discount=jnp.asarray(0.0, dtype=jnp.float32),
    )

    assert bool(result.state_valid)
    assert bool(result.input_valid)
    assert not bool(result.candidate_state_finite)
    assert not bool(result.update_applied)
    chex.assert_trees_all_equal(result.state, state)


def test_discrete_scan_surfaces_rejection_and_rolls_back_start_rng() -> None:
    agent = ActorCriticAgent(ActorCriticConfig(n_actions=2))
    state = agent.init(2, jr.key(11))
    observations = jnp.asarray(
        [[1.0, 0.0], [jnp.inf, 1.0], [0.0, 1.0]],
        dtype=jnp.float32,
    )
    next_observations = jnp.asarray(
        [[0.0, 1.0], [0.5, 0.5], [1.0, 1.0]],
        dtype=jnp.float32,
    )
    rewards = jnp.asarray([1.0, 1.0, 1.0], dtype=jnp.float32)
    terminals = jnp.asarray([False, False, True])

    result = run_actor_critic_from_arrays(
        agent,
        state,
        observations,
        rewards,
        terminals,
        next_observations,
    )
    reference = run_actor_critic_from_arrays(
        agent,
        state,
        observations[jnp.asarray((0, 2))],
        rewards[jnp.asarray((0, 2))],
        terminals[jnp.asarray((0, 2))],
        next_observations[jnp.asarray((0, 2))],
    )

    np.testing.assert_array_equal(result.update_applied, (True, False, True))
    np.testing.assert_array_equal(result.input_valid, (True, False, True))
    np.testing.assert_array_equal(result.actions[1], -1)
    chex.assert_trees_all_equal(result.state, reference.state)
    np.testing.assert_array_equal(result.actions[jnp.asarray((0, 2))], reference.actions)


def test_continuous_scan_surfaces_rejection_and_rolls_back_start_rng() -> None:
    agent = ContinuousActorCriticAgent(ContinuousActorCriticConfig(action_dim=1))
    state = agent.init(2, jr.key(12))
    observations = jnp.asarray(
        [[1.0, 0.0], [jnp.nan, 1.0], [0.0, 1.0]],
        dtype=jnp.float32,
    )
    next_observations = jnp.asarray(
        [[0.0, 1.0], [0.5, 0.5], [1.0, 1.0]],
        dtype=jnp.float32,
    )
    rewards = jnp.asarray([1.0, 1.0, 1.0], dtype=jnp.float32)
    terminals = jnp.asarray([False, False, True])

    result = run_continuous_actor_critic_from_arrays(
        agent,
        state,
        observations,
        rewards,
        terminals,
        next_observations,
    )
    reference = run_continuous_actor_critic_from_arrays(
        agent,
        state,
        observations[jnp.asarray((0, 2))],
        rewards[jnp.asarray((0, 2))],
        terminals[jnp.asarray((0, 2))],
        next_observations[jnp.asarray((0, 2))],
    )

    np.testing.assert_array_equal(result.update_applied, (True, False, True))
    np.testing.assert_array_equal(result.input_valid, (True, False, True))
    np.testing.assert_array_equal(result.actions[1], (0.0,))
    chex.assert_trees_all_equal(result.state, reference.state)
    np.testing.assert_array_equal(result.actions[jnp.asarray((0, 2))], reference.actions)


def test_v2_configs_resources_and_explicit_legacy_migrations() -> None:
    discrete = ActorCriticAgent(ActorCriticConfig(n_actions=3))
    continuous = ContinuousActorCriticAgent(ContinuousActorCriticConfig(action_dim=2))
    discrete_state = discrete.init(4, jr.key(21))
    continuous_state = continuous.init(4, jr.key(22))

    assert discrete.config.to_config()["schema"] == ACTOR_CRITIC_CONFIG_SCHEMA
    assert discrete.to_config()["state_schema"] == ACTOR_CRITIC_STATE_SCHEMA
    assert continuous.config.to_config()["schema"] == CONTINUOUS_ACTOR_CRITIC_CONFIG_SCHEMA
    assert continuous.to_config()["state_schema"] == CONTINUOUS_ACTOR_CRITIC_STATE_SCHEMA
    assert discrete.resource_budget(4).state_nbytes == measure_actor_critic_state_nbytes(
        discrete_state
    )
    assert continuous.resource_budget(
        4
    ).state_nbytes == measure_continuous_actor_critic_state_nbytes(continuous_state)

    discrete_legacy_config = discrete.config.to_config()
    discrete_legacy_config.pop("schema")
    discrete_legacy_config.pop("type")
    continuous_legacy_config = continuous.config.to_config()
    continuous_legacy_config.pop("schema")
    continuous_legacy_config.pop("type")
    with pytest.raises(ValueError, match="explicit migration"):
        ActorCriticConfig.from_config(discrete_legacy_config)
    with pytest.raises(ValueError, match="explicit migration"):
        ContinuousActorCriticConfig.from_config(continuous_legacy_config)
    assert migrate_legacy_actor_critic_config(discrete_legacy_config) == discrete.config
    assert (
        migrate_legacy_continuous_actor_critic_config(continuous_legacy_config) == continuous.config
    )

    discrete_fields = {
        field.name: getattr(discrete_state, field.name)
        for field in dataclasses.fields(discrete_state)  # type: ignore[arg-type]
        if field.name != "step_words"
    }
    continuous_fields = {
        field.name: getattr(continuous_state, field.name)
        for field in dataclasses.fields(continuous_state)  # type: ignore[arg-type]
        if field.name != "step_words"
    }
    chex.assert_trees_all_equal(
        migrate_legacy_actor_critic_state(discrete, discrete_fields),
        discrete_state,
    )
    chex.assert_trees_all_equal(
        migrate_legacy_continuous_actor_critic_state(
            continuous,
            continuous_fields,
        ),
        continuous_state,
    )
    discrete_fields["step_count"] = jnp.asarray(_INT32_MAX, dtype=jnp.int32)
    with pytest.raises(ValueError, match="ambiguous"):
        migrate_legacy_actor_critic_state(discrete, discrete_fields)


def test_static_contracts_reject_wrong_counter_and_scan_action_dtypes() -> None:
    discrete, state = _case("discrete")
    malformed = state.replace(step_words=jnp.zeros((2,), dtype=jnp.int32))
    with pytest.raises(TypeError, match="step_words"):
        _update(discrete, malformed)
    with pytest.raises(TypeError, match="actions"):
        run_actor_critic_from_arrays(
            discrete,
            state,
            jnp.zeros((1, 2), dtype=jnp.float32),
            jnp.zeros((1,), dtype=jnp.float32),
            None,
            jnp.zeros((1, 2), dtype=jnp.float32),
            actions=jnp.zeros((1,), dtype=jnp.float32),
            discounts=jnp.zeros((1,), dtype=jnp.float32),
        )
    with pytest.raises(ValueError, match="gamma"):
        ActorCriticConfig(n_actions=2, gamma=1.1)
    with pytest.raises(ValueError, match="log_sigma"):
        ContinuousActorCriticConfig(action_dim=1, log_sigma_min=-200.0)
