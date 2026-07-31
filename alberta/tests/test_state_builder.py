"""Contract, learning, resource, and checkpoint tests for state builders."""

from __future__ import annotations

from pathlib import Path

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import alberta_framework as alberta
import alberta_framework.core as core
from alberta_framework.core.state_builder import (
    FixedTraceStateBuilder,
    FixedTraceStateBuilderConfig,
    IdentityStateBuilder,
    IdentityStateBuilderConfig,
    OnlineGatedStateBuilder,
    OnlineGatedStateBuilderConfig,
    StateBuilder,
    load_state_builder_checkpoint,
    save_state_builder_checkpoint,
    state_builder_from_config,
)


def test_state_builder_public_exports_resolve_to_core_implementation() -> None:
    assert alberta.StateBuilder is core.StateBuilder
    assert alberta.OnlineGatedStateBuilder is core.OnlineGatedStateBuilder
    assert alberta.state_builder_from_config is core.state_builder_from_config
    assert alberta.save_state_builder_checkpoint is core.save_state_builder_checkpoint


def _state_scalar_count(state: object) -> int:
    return sum(
        int(np.prod(np.asarray(leaf).shape, dtype=np.int64))
        for leaf in jax.tree_util.tree_leaves(state)
    )


@pytest.mark.parametrize(
    "builder",
    [
        IdentityStateBuilder(IdentityStateBuilderConfig(observation_dim=3)),
        FixedTraceStateBuilder(FixedTraceStateBuilderConfig(observation_dim=3, n_actions=2)),
        OnlineGatedStateBuilder(
            OnlineGatedStateBuilderConfig(
                observation_dim=3,
                n_actions=2,
                hidden_dim=4,
            )
        ),
    ],
)
def test_builders_satisfy_runtime_contract_and_exact_state_budget(
    builder: StateBuilder[object],
) -> None:
    assert isinstance(builder, StateBuilder)
    state = builder.init(jr.key(0))
    budget = builder.resource_budget()

    assert budget.output_scalars == builder.feature_dim()
    assert budget.state_scalars == _state_scalar_count(state)
    assert budget.state_bytes == 4 * budget.state_scalars
    assert budget.trainable_scalars <= budget.state_scalars


def test_identity_is_observation_only_and_encode_is_pure() -> None:
    builder = IdentityStateBuilder(IdentityStateBuilderConfig(observation_dim=2))
    state = builder.init(jr.key(0))
    observation = jnp.asarray([1.0, -2.0])

    state, features = builder.start(state, observation)
    encoded = builder.encode(state, observation)
    learned_state, diagnostics = builder.learn(state, jnp.ones(2))

    chex.assert_trees_all_close(features, observation)
    chex.assert_trees_all_close(encoded, observation)
    chex.assert_trees_all_equal(learned_state, state)
    assert int(state.step_count) == 1
    assert float(diagnostics.parameter_update_norm) == 0.0


def test_fixed_trace_state_is_post_update_and_encode_does_not_advance() -> None:
    builder = FixedTraceStateBuilder(
        FixedTraceStateBuilderConfig(
            observation_dim=1,
            observation_decay_rates=(0.5,),
            action_decay_rates=(),
            outcome_decay_rates=(),
        )
    )
    state = builder.init(jr.key(0))
    state, first = builder.start(state, jnp.asarray([2.0]))
    encoded = builder.encode(state, jnp.asarray([2.0]))
    next_state, second = builder.update(
        state,
        jnp.asarray([0.0]),
        -1,
        0.0,
        1.0,
    )

    # Representation is [raw observation, post-update trace].
    chex.assert_trees_all_close(first, jnp.asarray([2.0, 1.0]))
    chex.assert_trees_all_close(encoded, first)
    chex.assert_trees_all_close(second, jnp.asarray([0.0, 0.5]))
    assert int(state.step_count) == 1
    assert int(next_state.step_count) == 2


def test_online_gated_builder_updates_recurrent_parameters_from_delayed_gradient() -> None:
    builder = OnlineGatedStateBuilder(
        OnlineGatedStateBuilderConfig(
            observation_dim=2,
            hidden_dim=3,
            step_size=0.05,
        )
    )
    state = builder.init(jr.key(7))
    initial_parameters = state.parameters

    state, _ = builder.start(state, jnp.asarray([1.0, 1.0]))
    for _ in range(4):
        state, features = builder.update(
            state,
            jnp.asarray([0.0, 0.0]),
            -1,
            0.0,
            1.0,
        )
    gradient = jnp.concatenate([jnp.zeros(2), jnp.ones(3)])
    learned_state, diagnostics = builder.learn(state, gradient)

    assert features.shape == (5,)
    assert float(jnp.linalg.norm(learned_state.parameters - initial_parameters)) > 0.0
    assert float(diagnostics.gradient_norm) > 0.0
    assert float(diagnostics.parameter_update_norm) > 0.0
    assert int(learned_state.update_count) == 1
    chex.assert_tree_all_finite(learned_state)


def test_event_uses_current_observation_and_preceding_transition_values() -> None:
    builder = OnlineGatedStateBuilder(
        OnlineGatedStateBuilderConfig(
            observation_dim=2,
            n_actions=3,
            hidden_dim=2,
        )
    )
    event = builder._event(  # noqa: SLF001
        jnp.asarray([0.25, -0.75], dtype=jnp.float32),
        1,
        -0.5,
        0.0,
    )
    chex.assert_trees_all_close(
        event,
        jnp.asarray(
            [0.25, -0.75, 0.0, 1.0, 0.0, -0.5, 0.0],
            dtype=jnp.float32,
        ),
    )

    state = builder.init(jr.key(1))
    started, _ = builder.start(
        state,
        jnp.asarray([0.25, -0.75], dtype=jnp.float32),
        last_action=1,
        last_reward=-0.5,
        last_discount=0.0,
    )
    expected_hidden = builder._transition(  # noqa: SLF001
        state.parameters,
        state.hidden,
        event,
    )
    chex.assert_trees_all_close(started.hidden, expected_hidden)


def test_online_recurrent_sensitivity_and_learn_match_central_finite_difference() -> None:
    config = OnlineGatedStateBuilderConfig(
        observation_dim=2,
        n_actions=2,
        hidden_dim=2,
        step_size=0.03,
        gradient_clip=1.0e6,
        initialization_scale=0.15,
    )
    builder = OnlineGatedStateBuilder(config)
    initial_state = builder.init(jr.key(13))
    observations = jnp.asarray(
        [[0.4, -0.7], [1.1, 0.2], [-0.3, 0.8]],
        dtype=jnp.float32,
    )
    actions = jnp.asarray([0, 1, 0], dtype=jnp.int32)
    rewards = jnp.asarray([0.2, -0.5, 0.7], dtype=jnp.float32)
    discounts = jnp.asarray([0.9, 0.4, 1.0], dtype=jnp.float32)

    state = initial_state
    for index in range(observations.shape[0]):
        state, _ = builder.update(
            state,
            observations[index],
            actions[index],
            rewards[index],
            discounts[index],
        )

    def unrolled_hidden(parameters: jax.Array) -> jax.Array:
        hidden = initial_state.hidden
        for index in range(observations.shape[0]):
            event = builder._event(  # noqa: SLF001
                observations[index],
                actions[index],
                rewards[index],
                discounts[index],
            )
            hidden = builder._transition(  # noqa: SLF001
                parameters,
                hidden,
                event,
            )
        return hidden

    epsilon = jnp.asarray(1.0e-3, dtype=jnp.float32)
    basis = jnp.eye(config.parameter_count(), dtype=jnp.float32)
    finite_difference_sensitivity = jax.vmap(
        lambda direction: (
            (
                unrolled_hidden(initial_state.parameters + epsilon * direction)
                - unrolled_hidden(initial_state.parameters - epsilon * direction)
            )
            / (2.0 * epsilon)
        )
    )(basis).T

    chex.assert_trees_all_close(
        state.parameter_sensitivity,
        finite_difference_sensitivity,
        atol=3.0e-5,
        rtol=3.0e-4,
    )

    representation_gradient = jnp.asarray(
        [0.6, -0.2, 0.7, -1.1],
        dtype=jnp.float32,
    )

    def scalar_loss(parameters: jax.Array) -> jax.Array:
        representation = jnp.concatenate([observations[-1], unrolled_hidden(parameters)])
        return representation_gradient @ representation

    finite_difference_gradient = jax.vmap(
        lambda direction: (
            (
                scalar_loss(initial_state.parameters + epsilon * direction)
                - scalar_loss(initial_state.parameters - epsilon * direction)
            )
            / (2.0 * epsilon)
        )
    )(basis)
    learned_state, diagnostics = builder.learn(state, representation_gradient)
    implemented_gradient = (state.parameters - learned_state.parameters) / config.step_size

    chex.assert_trees_all_close(
        implemented_gradient,
        finite_difference_gradient,
        atol=3.0e-5,
        rtol=3.0e-4,
    )
    chex.assert_trees_all_close(
        diagnostics.gradient_norm,
        jnp.linalg.norm(finite_difference_gradient),
        atol=3.0e-5,
        rtol=3.0e-4,
    )
    chex.assert_trees_all_close(
        diagnostics.clipped_gradient_norm,
        diagnostics.gradient_norm,
        atol=1.0e-6,
    )


def test_online_gated_python_loop_matches_jitted_scan() -> None:
    builder = OnlineGatedStateBuilder(
        OnlineGatedStateBuilderConfig(
            observation_dim=2,
            n_actions=2,
            hidden_dim=3,
            step_size=0.02,
            gradient_clip=2.0,
        )
    )
    observations = jnp.asarray(
        [[0.2, -0.4], [0.8, 0.1], [-0.5, 0.7], [0.3, 0.9]],
        dtype=jnp.float32,
    )
    actions = jnp.asarray([0, 1, 1, 0], dtype=jnp.int32)
    rewards = jnp.asarray([0.1, -0.2, 0.4, 0.0], dtype=jnp.float32)
    discounts = jnp.asarray([1.0, 0.9, 0.5, 1.0], dtype=jnp.float32)
    gradients = jnp.asarray(
        [
            [0.1, -0.2, 0.3, 0.1, -0.4],
            [-0.3, 0.2, 0.1, -0.2, 0.5],
            [0.0, 0.1, -0.4, 0.3, 0.2],
            [0.2, 0.0, 0.5, -0.1, -0.3],
        ],
        dtype=jnp.float32,
    )
    initial_state = builder.init(jr.key(21))

    loop_state = initial_state
    loop_features = []
    loop_update_norms = []
    for index in range(observations.shape[0]):
        loop_state, features = builder.update(
            loop_state,
            observations[index],
            actions[index],
            rewards[index],
            discounts[index],
        )
        loop_state, diagnostics = builder.learn(loop_state, gradients[index])
        loop_features.append(features)
        loop_update_norms.append(diagnostics.parameter_update_norm)

    def run_scan(initial: object) -> tuple[object, tuple[jax.Array, jax.Array]]:
        def step(
            state: object,
            inputs: tuple[jax.Array, ...],
        ) -> tuple[object, tuple[jax.Array, jax.Array]]:
            observation, action, reward, discount, gradient = inputs
            next_state, features = builder.update(
                state,
                observation,
                action,
                reward,
                discount,
            )
            next_state, diagnostics = builder.learn(next_state, gradient)
            return next_state, (features, diagnostics.parameter_update_norm)

        return jax.lax.scan(
            step,
            initial,
            (observations, actions, rewards, discounts, gradients),
        )

    scan_state, (scan_features, scan_update_norms) = jax.jit(run_scan)(initial_state)

    chex.assert_trees_all_close(scan_state, loop_state, atol=1.0e-6)
    chex.assert_trees_all_close(
        scan_features,
        jnp.stack(loop_features),
        atol=1.0e-6,
    )
    chex.assert_trees_all_close(
        scan_update_norms,
        jnp.stack(loop_update_norms),
        atol=1.0e-6,
    )


def test_online_gated_encode_is_pure_and_does_not_apply_recurrence_twice() -> None:
    builder = OnlineGatedStateBuilder(
        OnlineGatedStateBuilderConfig(observation_dim=2, hidden_dim=3)
    )
    state, features = builder.start(
        builder.init(jr.key(3)),
        jnp.asarray([0.25, -0.75]),
    )
    state_before = jax.tree_util.tree_map(lambda value: value.copy(), state)

    encoded_once = builder.encode(state, jnp.asarray([0.25, -0.75]))
    encoded_twice = builder.encode(state, jnp.asarray([0.25, -0.75]))

    chex.assert_trees_all_close(encoded_once, features)
    chex.assert_trees_all_close(encoded_twice, features)
    chex.assert_trees_all_equal(state, state_before)


@pytest.mark.parametrize(
    "builder",
    [
        IdentityStateBuilder(IdentityStateBuilderConfig(observation_dim=2)),
        FixedTraceStateBuilder(
            FixedTraceStateBuilderConfig(
                observation_dim=2,
                n_actions=3,
                observation_decay_rates=(0.25, 0.9),
            )
        ),
        OnlineGatedStateBuilder(
            OnlineGatedStateBuilderConfig(
                observation_dim=2,
                n_actions=3,
                hidden_dim=5,
            )
        ),
    ],
)
def test_config_factory_round_trip(builder: StateBuilder[object]) -> None:
    restored = state_builder_from_config(builder.to_config())
    assert restored.to_config() == builder.to_config()
    assert restored.feature_dim() == builder.feature_dim()
    assert restored.resource_budget() == builder.resource_budget()


@pytest.mark.parametrize(
    "builder",
    [
        IdentityStateBuilder(IdentityStateBuilderConfig(observation_dim=2)),
        FixedTraceStateBuilder(
            FixedTraceStateBuilderConfig(
                observation_dim=2,
                n_actions=2,
                observation_decay_rates=(0.5, 0.9),
            )
        ),
    ],
    ids=("identity", "fixed-trace"),
)
def test_fixed_builder_checkpoints_restore_config_and_state(
    tmp_path: Path,
    builder: StateBuilder[object],
) -> None:
    state, _ = builder.start(
        builder.init(jr.key(4)),
        jnp.asarray([0.5, -0.25], dtype=jnp.float32),
        last_action=1,
        last_reward=0.75,
        last_discount=0.0,
    )
    checkpoint_path = tmp_path / type(builder).__name__
    save_state_builder_checkpoint(builder, state, checkpoint_path)
    restored_builder, restored_state = load_state_builder_checkpoint(checkpoint_path)

    assert restored_builder.to_config() == builder.to_config()
    assert restored_builder.resource_budget() == builder.resource_budget()
    chex.assert_trees_all_close(restored_state, state)


def test_online_gated_state_checkpoint_restores_config_parameters_and_sensitivity(
    tmp_path: Path,
) -> None:
    builder = OnlineGatedStateBuilder(
        OnlineGatedStateBuilderConfig(
            observation_dim=2,
            n_actions=2,
            hidden_dim=3,
            step_size=0.02,
        )
    )
    state, _ = builder.start(builder.init(jr.key(9)), jnp.asarray([1.0, 0.0]))
    state, _ = builder.update(state, jnp.asarray([0.0, 1.0]), 1, 0.5, 0.9)
    state, _ = builder.learn(state, jnp.ones(builder.feature_dim()))
    checkpoint_path = tmp_path / "state_builder"

    save_state_builder_checkpoint(builder, state, checkpoint_path)
    restored_builder, restored_state = load_state_builder_checkpoint(checkpoint_path)

    assert restored_builder.to_config() == builder.to_config()
    assert restored_builder.resource_budget() == builder.resource_budget()
    chex.assert_trees_all_close(restored_state, state)


def test_online_gated_checkpoint_resume_matches_uninterrupted_learning(
    tmp_path: Path,
) -> None:
    builder = OnlineGatedStateBuilder(
        OnlineGatedStateBuilderConfig(
            observation_dim=2,
            n_actions=2,
            hidden_dim=3,
            step_size=0.015,
            gradient_clip=1.5,
        )
    )
    observations = jnp.asarray(
        [
            [0.3, -0.1],
            [0.7, 0.4],
            [-0.2, 0.9],
            [0.5, -0.8],
            [0.1, 0.6],
            [-0.4, -0.3],
        ],
        dtype=jnp.float32,
    )
    actions = jnp.asarray([0, 1, 0, 1, 1, 0], dtype=jnp.int32)
    rewards = jnp.asarray([0.2, 0.0, -0.3, 0.6, -0.1, 0.4], dtype=jnp.float32)
    discounts = jnp.asarray([1.0, 0.8, 0.4, 1.0, 0.9, 0.0], dtype=jnp.float32)
    gradients = jnp.reshape(
        jnp.linspace(-0.5, 0.7, observations.shape[0] * builder.feature_dim()),
        (observations.shape[0], builder.feature_dim()),
    )

    def advance(
        active_builder: StateBuilder[object],
        state: object,
        start: int,
        stop: int,
    ) -> tuple[object, jax.Array]:
        emitted = []
        for index in range(start, stop):
            state, features = active_builder.update(
                state,
                observations[index],
                actions[index],
                rewards[index],
                discounts[index],
            )
            state, _ = active_builder.learn(state, gradients[index])
            emitted.append(features)
        return state, jnp.stack(emitted)

    initial_state = builder.init(jr.key(31))
    uninterrupted_state, uninterrupted_features = advance(
        builder,
        initial_state,
        0,
        observations.shape[0],
    )
    prefix_state, prefix_features = advance(builder, initial_state, 0, 3)
    checkpoint_path = tmp_path / "resume_state_builder"
    save_state_builder_checkpoint(builder, prefix_state, checkpoint_path)
    restored_builder, restored_state = load_state_builder_checkpoint(checkpoint_path)
    resumed_state, suffix_features = advance(
        restored_builder,
        restored_state,
        3,
        observations.shape[0],
    )

    chex.assert_trees_all_close(resumed_state, uninterrupted_state, atol=1.0e-7)
    chex.assert_trees_all_close(
        jnp.concatenate([prefix_features, suffix_features], axis=0),
        uninterrupted_features,
        atol=1.0e-7,
    )


def test_invalid_builder_configs_are_rejected() -> None:
    with pytest.raises(ValueError, match="observation_dim"):
        IdentityStateBuilderConfig(observation_dim=0)
    with pytest.raises(ValueError, match="n_actions"):
        FixedTraceStateBuilderConfig(observation_dim=1, n_actions=-1)
    with pytest.raises(ValueError, match="step_size"):
        OnlineGatedStateBuilderConfig(observation_dim=1, step_size=0.0)
    with pytest.raises(ValueError, match="unknown state builder"):
        state_builder_from_config({"type": "not-a-builder"})


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_builder_hyperparameters_are_rejected(invalid: float) -> None:
    with pytest.raises(ValueError, match="observation_decay_rates"):
        FixedTraceStateBuilderConfig(
            observation_dim=1,
            observation_decay_rates=(invalid,),
        )
    with pytest.raises(ValueError, match="step_size"):
        OnlineGatedStateBuilderConfig(observation_dim=1, step_size=invalid)
    with pytest.raises(ValueError, match="gradient_clip"):
        OnlineGatedStateBuilderConfig(observation_dim=1, gradient_clip=invalid)
    with pytest.raises(ValueError, match="initial_gate_bias"):
        OnlineGatedStateBuilderConfig(observation_dim=1, initial_gate_bias=invalid)
    with pytest.raises(ValueError, match="initialization_scale"):
        OnlineGatedStateBuilderConfig(observation_dim=1, initialization_scale=invalid)
