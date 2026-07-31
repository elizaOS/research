"""Production-facing Step 4 SARSA facade tests (mirrors test_step3_production.py)."""

import chex
import jax.numpy as jnp
import jax.random as jr
import pytest

from alberta_framework.core.optimizers import IDBD, LMS, Autostep, ObGDBounding
from alberta_framework.steps import (
    Step4SARSAConfig,
    init_step4_state,
    make_step4_bounder,
    make_step4_optimizer,
    make_step4_sarsa_agent,
    run_step4_scan,
    run_step4_smoke,
    step4_update,
)


def test_step4_config_roundtrip() -> None:
    config = Step4SARSAConfig(
        n_actions=3,
        hidden_sizes=(8, 4),
        gamma=0.95,
        optimizer="idbd",
        bounder="none",
        lamda=0.5,
        trace_mode="replacing",
    )
    payload = config.to_dict()

    assert payload["hidden_sizes"] == [8, 4]
    assert Step4SARSAConfig.from_dict(payload) == config

    sarsa_config = config.to_sarsa_config()
    assert sarsa_config.n_actions == 3
    assert sarsa_config.gamma == 0.95


def test_step4_factories_and_validation() -> None:
    assert isinstance(make_step4_optimizer(Step4SARSAConfig(optimizer="lms")), LMS)
    assert isinstance(make_step4_optimizer(Step4SARSAConfig(optimizer="idbd")), IDBD)
    assert isinstance(make_step4_optimizer(Step4SARSAConfig(optimizer="autostep")), Autostep)
    assert make_step4_bounder(Step4SARSAConfig(bounder="none")) is None
    assert isinstance(make_step4_bounder(Step4SARSAConfig(bounder="obgd")), ObGDBounding)

    with pytest.raises(ValueError, match="n_actions"):
        Step4SARSAConfig(n_actions=0)
    with pytest.raises(ValueError, match="optimizer"):
        make_step4_optimizer(Step4SARSAConfig(optimizer="bogus"))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="bounder"):
        make_step4_bounder(Step4SARSAConfig(bounder="bogus"))  # type: ignore[arg-type]


def test_step4_prime_and_one_transition() -> None:
    config = Step4SARSAConfig(n_actions=2, hidden_sizes=(8,))
    agent = make_step4_sarsa_agent(config)
    feature_dim = 5
    data_key, state_key = jr.split(jr.key(3))
    features = jr.normal(data_key, (2, feature_dim), dtype=jnp.float32)

    state = init_step4_state(
        agent,
        feature_dim=feature_dim,
        key=state_key,
        initial_features=features[0],
    )
    chex.assert_trees_all_close(state.last_observation, features[0])
    assert 0 <= int(state.last_action) < config.n_actions

    result = step4_update(
        agent,
        state,
        jnp.asarray(0.5, dtype=jnp.float32),
        features[1],
        jnp.asarray(0.0, dtype=jnp.float32),
    )
    chex.assert_shape(result.q_values, (config.n_actions,))
    chex.assert_tree_all_finite(result.q_values)
    chex.assert_tree_all_finite(result.td_error)
    assert 0 <= int(result.action) < config.n_actions
    assert float(result.reward) == 0.5
    chex.assert_trees_all_close(result.state.last_observation, features[1])


def test_step4_scan_shapes_and_finiteness() -> None:
    config = Step4SARSAConfig(n_actions=3, hidden_sizes=(), epsilon_decay_steps=8)
    agent = make_step4_sarsa_agent(config)
    steps, feature_dim = 12, 4
    data_key, state_key = jr.split(jr.key(7))
    observations = jr.normal(data_key, (steps + 1, feature_dim), dtype=jnp.float32)
    rewards = jnp.tanh(observations[1:, 0])
    terminated = jnp.zeros(steps, dtype=jnp.float32)

    state = init_step4_state(
        agent,
        feature_dim=feature_dim,
        key=state_key,
        initial_features=observations[0],
    )
    result = run_step4_scan(agent, state, observations[1:], rewards, terminated)

    chex.assert_shape(result.q_values, (steps, config.n_actions))
    chex.assert_shape(result.td_errors, (steps,))
    chex.assert_shape(result.actions, (steps,))
    chex.assert_tree_all_finite(result.q_values)
    chex.assert_tree_all_finite(result.td_errors)
    assert bool(jnp.all(result.actions >= 0))
    assert bool(jnp.all(result.actions < config.n_actions))


def test_step4_smoke_is_finite_and_serializable() -> None:
    config = Step4SARSAConfig(n_actions=2, hidden_sizes=(8,), optimizer="autostep")
    result = run_step4_smoke(config, steps=16, feature_dim=6, seed=1)
    payload = result.to_dict()

    assert result.finite
    assert result.q_values_shape == (16, 2)
    assert result.td_errors_shape == (16,)
    assert result.actions_shape == (16,)
    assert payload["config"] == config.to_dict()
    assert payload["q_values_shape"] == [16, 2]
    agent_config = payload["agent_config"]
    assert isinstance(agent_config, dict)
    assert agent_config["type"] == "SARSAAgent"


def test_step4_smoke_validation() -> None:
    with pytest.raises(ValueError, match="steps"):
        run_step4_smoke(steps=0)
    with pytest.raises(ValueError, match="feature_dim"):
        run_step4_smoke(feature_dim=0)
