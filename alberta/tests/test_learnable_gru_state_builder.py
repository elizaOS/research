"""Contracts for the conventional full-GRU online state-builder baseline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import alberta_framework as alberta
import alberta_framework.core as core
from alberta_framework.core.state_builder import (
    LEARNABLE_GRU_STATE_BUILDER_STATE_SCHEMA,
    LearnableGRUStateBuilder,
    LearnableGRUStateBuilderConfig,
    StateBuilder,
    load_state_builder_checkpoint,
    save_state_builder_checkpoint,
    state_builder_config_from_config,
    state_builder_from_config,
)

pytestmark = pytest.mark.unit


def _builder(*, include_raw_observation: bool = True) -> LearnableGRUStateBuilder:
    return LearnableGRUStateBuilder(
        LearnableGRUStateBuilderConfig(
            observation_dim=2,
            n_actions=3,
            hidden_dim=2,
            step_size=0.02,
            gradient_clip=3.0,
            initial_update_bias=0.5,
            initial_reset_bias=-0.25,
            initialization_scale=0.08,
            include_raw_observation=include_raw_observation,
        )
    )


def _scalar_count(tree: object) -> int:
    return sum(int(np.asarray(leaf).size) for leaf in jax.tree_util.tree_leaves(tree))


def test_full_gru_is_public_protocol_builder_with_exact_budget_and_config() -> None:
    builder = _builder()
    state = builder.init(jr.key(1, impl="threefry2x32"))
    config = builder.to_config()

    assert isinstance(builder, StateBuilder)
    assert config["type"] == "LearnableGRUStateBuilder"
    assert config["state_schema"] == LEARNABLE_GRU_STATE_BUILDER_STATE_SCHEMA
    assert isinstance(state_builder_config_from_config(config), LearnableGRUStateBuilderConfig)
    restored = state_builder_from_config(config)
    assert isinstance(restored, LearnableGRUStateBuilder)
    assert restored.to_config() == config
    assert builder.resource_budget().state_scalars == _scalar_count(state)
    assert builder.resource_budget().state_bytes == 4 * _scalar_count(state)
    assert builder.resource_budget().trainable_scalars == builder.config.parameter_count()
    assert alberta.LearnableGRUStateBuilder is core.LearnableGRUStateBuilder
    assert alberta.LearnableGRUStateBuilderConfig is core.LearnableGRUStateBuilderConfig


def test_full_gru_rtrl_sensitivity_matches_fixed_parameter_unroll_jacobian() -> None:
    builder = _builder(include_raw_observation=False)
    initial = builder.init(jr.key(2, impl="threefry2x32"))
    observations = jnp.asarray([[0.2, -0.4], [0.8, 0.1], [-0.3, 0.6]], dtype=jnp.float32)
    actions = jnp.asarray([-1, 1, 2], dtype=jnp.int32)
    rewards = jnp.asarray([0.0, 0.25, -0.1], dtype=jnp.float32)
    discounts = jnp.asarray([1.0, 0.9, 0.8], dtype=jnp.float32)

    state = initial
    for observation, action, reward, discount in zip(
        observations, actions, rewards, discounts, strict=True
    ):
        state, _ = builder.update(state, observation, action, reward, discount)

    def unroll(parameters: jax.Array) -> jax.Array:
        hidden = jnp.zeros((builder.config.hidden_dim,), dtype=jnp.float32)
        for observation, action, reward, discount in zip(
            observations, actions, rewards, discounts, strict=True
        ):
            event = builder._event(observation, action, reward, discount)
            hidden = builder._transition(parameters, hidden, event)
        return hidden

    expected = jax.jacfwd(unroll)(initial.parameters)
    np.testing.assert_allclose(state.parameter_sensitivity, expected, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(state.hidden, unroll(initial.parameters), rtol=1e-6, atol=1e-6)


def test_full_gru_uses_dense_recurrent_coupling_and_all_event_fields() -> None:
    builder = _builder(include_raw_observation=False)
    state = builder.init(jr.key(3, impl="threefry2x32"))
    state = cast(
        Any,
        state,
    ).replace(
        hidden=jnp.asarray((0.35, -0.2), dtype=jnp.float32),
    )
    base = builder._event(
        jnp.asarray((0.1, -0.3), dtype=jnp.float32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.2, dtype=jnp.float32),
        jnp.asarray(0.9, dtype=jnp.float32),
    )
    reference = builder._transition(state.parameters, state.hidden, base)

    variants = (
        builder._event(jnp.asarray((0.2, -0.3), dtype=jnp.float32), 0, 0.2, 0.9),
        builder._event(jnp.asarray((0.1, -0.3), dtype=jnp.float32), 2, 0.2, 0.9),
        builder._event(jnp.asarray((0.1, -0.3), dtype=jnp.float32), 0, -0.4, 0.9),
        builder._event(jnp.asarray((0.1, -0.3), dtype=jnp.float32), 0, 0.2, 0.1),
    )
    assert all(
        not np.array_equal(
            reference,
            builder._transition(state.parameters, state.hidden, event),
        )
        for event in variants
    )

    hidden_jacobian = jax.jacfwd(builder._transition, argnums=1)(
        state.parameters,
        state.hidden,
        base,
    )
    assert abs(float(hidden_jacobian[0, 1])) > 1e-7
    assert abs(float(hidden_jacobian[1, 0])) > 1e-7


def test_full_gru_learning_proposal_is_source_bound_and_atomic() -> None:
    builder = _builder()
    initial = builder.init(jr.key(4, impl="threefry2x32"))
    source, representation = builder.start(
        initial,
        jnp.asarray((0.4, -0.1), dtype=jnp.float32),
        last_action=-1,
    )
    gradient = jnp.linspace(-0.4, 0.6, builder.feature_dim(), dtype=jnp.float32)
    proposal = builder.propose_learning_update(source, gradient)
    expected = source.parameter_sensitivity.T @ gradient[-builder.config.hidden_dim :]
    np.testing.assert_allclose(proposal.raw_parameter_gradient, expected, rtol=1e-6, atol=1e-6)
    assert bool(proposal.valid)

    destination, _ = builder.update(
        source,
        jnp.asarray((-0.2, 0.7), dtype=jnp.float32),
        jnp.asarray(1, dtype=jnp.int32),
        jnp.asarray(0.3, dtype=jnp.float32),
        jnp.asarray(0.95, dtype=jnp.float32),
    )
    committed, diagnostics = builder.commit_learning_update(destination, proposal)
    assert bool(diagnostics.applied)
    assert not np.array_equal(committed.parameters, destination.parameters)
    np.testing.assert_array_equal(committed.hidden, destination.hidden)
    np.testing.assert_array_equal(
        committed.parameter_sensitivity,
        destination.parameter_sensitivity,
    )
    assert committed.update_words.tolist() == [0, 1]

    stale, rejected = builder.commit_learning_update(committed, proposal)
    assert not bool(rejected.applied)
    chex.assert_trees_all_equal(stale, committed)
    assert representation.shape == (builder.feature_dim(),)


def test_full_gru_invalid_transition_and_candidate_learning_fail_closed() -> None:
    builder = _builder()
    state = builder.init(jr.key(5, impl="threefry2x32"))
    invalid = builder.update_with_status(
        state,
        jnp.asarray((jnp.nan, 0.0), dtype=jnp.float32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray(1.0, dtype=jnp.float32),
    )
    assert not bool(invalid.transition_applied)
    chex.assert_trees_all_equal(invalid.state, state)

    valid_state, _ = builder.start(
        state,
        jnp.asarray((0.1, 0.2), dtype=jnp.float32),
    )
    nonfinite = builder.propose_learning_update(
        valid_state,
        jnp.full((builder.feature_dim(),), jnp.inf, dtype=jnp.float32),
    )
    rejected_state, diagnostics = builder.commit_learning_update(valid_state, nonfinite)
    assert not bool(diagnostics.applied)
    chex.assert_trees_all_equal(rejected_state, valid_state)

    corrupt = cast(
        Any,
        valid_state,
    ).replace(
        step_words=jnp.asarray((0, 7), dtype=jnp.uint32),
    )
    assert not bool(builder.state_valid(corrupt))


def test_full_gru_eager_loop_matches_jitted_scan_and_reset_preserves_lifetime() -> None:
    builder = _builder()
    initial = builder.init(jr.key(6, impl="threefry2x32"))
    observations = jnp.asarray([[0.1, 0.2], [0.3, -0.4], [0.8, 0.6]], dtype=jnp.float32)
    actions = jnp.asarray([-1, 0, 2], dtype=jnp.int32)
    rewards = jnp.asarray([0.0, 0.5, -0.2], dtype=jnp.float32)
    discounts = jnp.asarray([1.0, 0.95, 0.0], dtype=jnp.float32)

    loop_state = initial
    loop_features = []
    for inputs in zip(observations, actions, rewards, discounts, strict=True):
        loop_state, features = builder.update(loop_state, *inputs)
        loop_features.append(features)

    def scan_step(state: object, inputs: tuple[jax.Array, ...]) -> tuple[object, jax.Array]:
        next_state, features = builder.update(state, *inputs)
        return next_state, features

    scan_state, scan_features = jax.jit(
        lambda state: jax.lax.scan(
            scan_step,
            state,
            (observations, actions, rewards, discounts),
        )
    )(initial)
    chex.assert_trees_all_close(scan_state, loop_state, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(scan_features, jnp.stack(loop_features), rtol=1e-6, atol=1e-6)

    reset = builder.reset_episode(loop_state)
    np.testing.assert_array_equal(reset.hidden, jnp.zeros_like(reset.hidden))
    np.testing.assert_array_equal(
        reset.parameter_sensitivity,
        jnp.zeros_like(reset.parameter_sensitivity),
    )
    np.testing.assert_array_equal(reset.parameters, loop_state.parameters)
    np.testing.assert_array_equal(reset.step_words, loop_state.step_words)
    np.testing.assert_array_equal(reset.update_words, loop_state.update_words)


def test_full_gru_checkpoint_round_trip_binds_exact_config(tmp_path: Path) -> None:
    builder = _builder()
    state, _ = builder.start(
        builder.init(jr.key(7, impl="threefry2x32")),
        jnp.asarray((0.25, -0.5), dtype=jnp.float32),
    )
    state, diagnostics = builder.learn(
        state,
        jnp.ones((builder.feature_dim(),), dtype=jnp.float32),
    )
    assert bool(diagnostics.applied)
    path = tmp_path / "learnable-gru"
    save_state_builder_checkpoint(builder, state, path)
    restored_builder, restored_state = load_state_builder_checkpoint(path)
    assert isinstance(restored_builder, LearnableGRUStateBuilder)
    assert restored_builder.to_config() == builder.to_config()
    chex.assert_trees_all_equal(restored_state, state)
