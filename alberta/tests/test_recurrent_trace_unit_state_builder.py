"""Contracts for the compressed recurrent-trace-unit state builder."""

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
from alberta_framework.core.recurrent_trace_actor_critic import RTUSensitivities, RTUState
from alberta_framework.core.state_builder import (
    RECURRENT_TRACE_UNIT_STATE_BUILDER_STATE_SCHEMA,
    RecurrentTraceUnitStateBuilder,
    RecurrentTraceUnitStateBuilderConfig,
    RecurrentTraceUnitStateBuilderState,
    load_state_builder_checkpoint,
    save_state_builder_checkpoint,
    state_builder_config_from_config,
    state_builder_from_config,
)

pytestmark = pytest.mark.unit


def _builder(
    *,
    include_raw_observation: bool = True,
    rtrl_taylor_correction: bool = False,
) -> RecurrentTraceUnitStateBuilder:
    return RecurrentTraceUnitStateBuilder(
        RecurrentTraceUnitStateBuilderConfig(
            observation_dim=2,
            n_actions=3,
            hidden_dim=2,
            step_size=0.02,
            gradient_clip=3.0,
            r_min=0.2,
            r_max=0.95,
            max_phase=3.0,
            rtu_epsilon=1.0e-8,
            include_raw_observation=include_raw_observation,
            rtrl_taylor_correction=rtrl_taylor_correction,
        )
    )


def _scalar_count(tree: object) -> int:
    return sum(int(np.asarray(leaf).size) for leaf in jax.tree_util.tree_leaves(tree))


def _dense_sensitivity(
    builder: RecurrentTraceUnitStateBuilder,
    sensitivities: RTUSensitivities,
) -> jax.Array:
    """Expand compressed unit-diagonal sensitivities only for the test oracle."""

    hidden_dim = builder.config.hidden_dim
    event_dim = builder.config.event_dim()
    parameter_count = builder.config.parameter_count()
    dense = jnp.zeros((2 * hidden_dim, parameter_count), dtype=jnp.float32)
    recurrent_rows = jnp.arange(hidden_dim, dtype=jnp.int32)
    component_rows = jnp.stack((recurrent_rows, hidden_dim + recurrent_rows))
    for hidden_index in range(hidden_dim):
        rows = component_rows[:, hidden_index]
        dense = dense.at[rows, hidden_index].set(
            sensitivities.nu_log[:, hidden_index]
        )
        dense = dense.at[rows, hidden_dim + hidden_index].set(
            sensitivities.theta_log[:, hidden_index]
        )
        for event_index in range(event_dim):
            matrix_index = hidden_index * event_dim + event_index
            real_column = 2 * hidden_dim + matrix_index
            imaginary_column = 2 * hidden_dim + hidden_dim * event_dim + matrix_index
            dense = dense.at[rows, real_column].set(
                sensitivities.b_real[:, hidden_index, event_index]
            )
            dense = dense.at[rows, imaginary_column].set(
                sensitivities.b_imag[:, hidden_index, event_index]
            )
    return dense


def test_rtu_builder_has_strict_factory_config_and_exact_compressed_budget() -> None:
    builder = _builder()
    state = builder.init(jr.key(1, impl="threefry2x32"))
    config = builder.to_config()

    assert alberta.RecurrentTraceUnitStateBuilder is core.RecurrentTraceUnitStateBuilder
    assert (
        alberta.RECURRENT_TRACE_UNIT_STATE_BUILDER_STATE_SCHEMA
        == core.RECURRENT_TRACE_UNIT_STATE_BUILDER_STATE_SCHEMA
    )

    assert config["type"] == "RecurrentTraceUnitStateBuilder"
    assert config["state_schema"] == RECURRENT_TRACE_UNIT_STATE_BUILDER_STATE_SCHEMA
    parsed = state_builder_config_from_config(config)
    assert isinstance(parsed, RecurrentTraceUnitStateBuilderConfig)
    restored = state_builder_from_config(config)
    assert isinstance(restored, RecurrentTraceUnitStateBuilder)
    assert restored.to_config() == config
    assert builder.resource_budget().state_scalars == _scalar_count(state)
    assert builder.resource_budget().state_bytes == 4 * _scalar_count(state)
    assert builder.resource_budget().trainable_scalars == builder.config.parameter_count()

    sensitivity_scalars = sum(
        int(np.asarray(leaf).size)
        for leaf in jax.tree_util.tree_leaves(state.sensitivities)
    )
    expected_compressed = (
        4 * builder.config.hidden_dim
        + 4 * builder.config.hidden_dim * builder.config.event_dim()
    )
    assert sensitivity_scalars == expected_compressed
    assert sensitivity_scalars < 2 * builder.config.hidden_dim * builder.config.parameter_count()
    assert state.taylor_trace is None
    assert state.sensitivity_source_parameters is None
    assert state.sensitivity_source_update_words is None

    with pytest.raises(ValueError, match="manifest is not exact"):
        RecurrentTraceUnitStateBuilderConfig.from_config({**config, "extra": 1})


def test_rtu_compressed_rtrl_matches_fixed_parameter_jacfwd_unroll() -> None:
    builder = _builder(include_raw_observation=False)
    initial = builder.init(jr.key(2, impl="threefry2x32"))
    observations = jnp.asarray(
        [[0.2, -0.4], [0.8, 0.1], [-0.3, 0.6]],
        dtype=jnp.float32,
    )
    actions = jnp.asarray([-1, 1, 2], dtype=jnp.int32)
    rewards = jnp.asarray([0.0, 0.25, -0.1], dtype=jnp.float32)
    discounts = jnp.asarray([1.0, 0.9, 0.8], dtype=jnp.float32)

    state = initial
    events = []
    for observation, action, reward, discount in zip(
        observations,
        actions,
        rewards,
        discounts,
        strict=True,
    ):
        events.append(builder._event(observation, action, reward, discount))
        state, _ = builder.update(state, observation, action, reward, discount)

    def unroll(parameters: jax.Array) -> jax.Array:
        rtu_state = RTUState(
            real=jnp.zeros((builder.config.hidden_dim,), dtype=jnp.float32),
            imaginary=jnp.zeros((builder.config.hidden_dim,), dtype=jnp.float32),
        )
        for event in events:
            rtu_state = builder._transition(parameters, rtu_state, event)
        return jnp.concatenate((rtu_state.real, rtu_state.imaginary))

    expected = jax.jacfwd(unroll)(initial.parameters)
    actual = _dense_sensitivity(builder, state.sensitivities)
    np.testing.assert_allclose(actual, expected, rtol=3.0e-5, atol=3.0e-5)
    np.testing.assert_allclose(
        jnp.concatenate((state.rtu_state.real, state.rtu_state.imaginary)),
        unroll(initial.parameters),
        rtol=1.0e-6,
        atol=1.0e-6,
    )


def test_rtu_learning_is_source_bound_and_commits_into_advanced_destination() -> None:
    builder = _builder()
    source, representation = builder.start(
        builder.init(jr.key(3, impl="threefry2x32")),
        jnp.asarray((0.4, -0.1), dtype=jnp.float32),
        last_action=-1,
    )
    gradient = jnp.linspace(-0.4, 0.6, builder.feature_dim(), dtype=jnp.float32)
    proposal = builder.propose_learning_update(source, gradient)
    hidden_gradient = gradient[-2 * builder.config.hidden_dim :].reshape((2, -1))
    sensitivities = source.sensitivities
    expected = jnp.concatenate(
        (
            jnp.sum(sensitivities.nu_log * hidden_gradient, axis=0),
            jnp.sum(sensitivities.theta_log * hidden_gradient, axis=0),
            jnp.sum(sensitivities.b_real * hidden_gradient[:, :, None], axis=0).reshape(-1),
            jnp.sum(sensitivities.b_imag * hidden_gradient[:, :, None], axis=0).reshape(-1),
        )
    )
    np.testing.assert_allclose(
        proposal.raw_parameter_gradient,
        expected,
        rtol=1.0e-6,
        atol=1.0e-6,
    )

    destination, _ = builder.update(
        source,
        jnp.asarray((-0.2, 0.7), dtype=jnp.float32),
        jnp.asarray(1, dtype=jnp.int32),
        jnp.asarray(0.3, dtype=jnp.float32),
        jnp.asarray(0.95, dtype=jnp.float32),
    )
    committed, diagnostics = jax.jit(builder.commit_learning_update)(
        destination,
        proposal,
    )
    assert bool(diagnostics.applied)
    assert not np.array_equal(committed.parameters, destination.parameters)
    chex.assert_trees_all_equal(committed.rtu_state, destination.rtu_state)
    chex.assert_trees_all_equal(committed.sensitivities, destination.sensitivities)
    assert committed.update_words.tolist() == [0, 1]
    assert representation.shape == (builder.feature_dim(),)

    stale, rejected = builder.commit_learning_update(committed, proposal)
    assert not bool(rejected.applied)
    chex.assert_trees_all_equal(stale, committed)
    tampered = cast(
        RecurrentTraceUnitStateBuilderState,
        destination.replace(parameters=destination.parameters.at[0].add(1.0e-4)),
    )
    rolled_back, rejected = builder.commit_learning_update(tampered, proposal)
    assert not bool(rejected.applied)
    chex.assert_trees_all_equal(rolled_back, tampered)


def test_rtu_taylor_path_owns_exact_source_and_delta_across_commit_and_reset() -> None:
    builder = _builder(rtrl_taylor_correction=True)
    state, _ = builder.start(
        builder.init(jr.key(4, impl="threefry2x32")),
        jnp.asarray((0.2, -0.5), dtype=jnp.float32),
    )
    assert state.taylor_trace is not None
    assert state.sensitivity_source_parameters is not None
    assert state.sensitivity_source_update_words is not None
    chex.assert_trees_all_equal(state.sensitivity_source_parameters, state.parameters)
    chex.assert_trees_all_equal(
        state.sensitivity_source_update_words,
        state.update_words,
    )

    learned, diagnostics = builder.learn(
        state,
        jnp.ones((builder.feature_dim(),), dtype=jnp.float32),
    )
    assert bool(diagnostics.applied)
    assert learned.sensitivity_source_parameters is not None
    assert learned.sensitivity_source_update_words is not None
    chex.assert_trees_all_equal(
        learned.sensitivity_source_parameters,
        state.parameters,
    )
    chex.assert_trees_all_equal(
        learned.sensitivity_source_update_words,
        state.update_words,
    )

    advanced, _ = builder.update(
        learned,
        jnp.asarray((-0.1, 0.7), dtype=jnp.float32),
        jnp.asarray(2, dtype=jnp.int32),
        jnp.asarray(0.4, dtype=jnp.float32),
        jnp.asarray(0.8, dtype=jnp.float32),
    )
    assert advanced.sensitivity_source_parameters is not None
    assert advanced.sensitivity_source_update_words is not None
    chex.assert_trees_all_equal(advanced.sensitivity_source_parameters, learned.parameters)
    chex.assert_trees_all_equal(
        advanced.sensitivity_source_update_words,
        learned.update_words,
    )

    reset = builder.reset_episode(advanced)
    chex.assert_trees_all_equal(
        reset.rtu_state,
        RTUState(
            real=jnp.zeros_like(advanced.rtu_state.real),
            imaginary=jnp.zeros_like(advanced.rtu_state.imaginary),
        ),
    )
    assert reset.sensitivity_source_parameters is not None
    chex.assert_trees_all_equal(reset.sensitivity_source_parameters, reset.parameters)
    chex.assert_trees_all_equal(
        reset.sensitivity_source_update_words,
        reset.update_words,
    )


def test_rtu_nonfinite_corrupt_and_clock_exhaustion_fail_stop_atomically() -> None:
    builder = _builder(rtrl_taylor_correction=True)
    state, _ = builder.start(
        builder.init(jr.key(5, impl="threefry2x32")),
        jnp.asarray((0.1, 0.2), dtype=jnp.float32),
    )
    invalid_input = builder.update_with_status(
        state,
        jnp.asarray((jnp.nan, 0.0), dtype=jnp.float32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray(1.0, dtype=jnp.float32),
    )
    assert not bool(invalid_input.transition_applied)
    chex.assert_trees_all_equal(invalid_input.state, state)

    assert state.sensitivity_source_parameters is not None
    corrupt = cast(
        RecurrentTraceUnitStateBuilderState,
        cast(Any, state).replace(
            sensitivity_source_parameters=state.sensitivity_source_parameters.at[0].set(
                jnp.nan
            )
        ),
    )
    assert not bool(builder.state_valid(corrupt))
    rejected = builder.update_with_status(
        corrupt,
        jnp.zeros((2,), dtype=jnp.float32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray(1.0, dtype=jnp.float32),
    )
    chex.assert_trees_all_equal(rejected.state, corrupt)

    exhausted = cast(
        RecurrentTraceUnitStateBuilderState,
        cast(Any, state).replace(
            step_count=jnp.asarray(2**31 - 1, dtype=jnp.int32),
            step_words=jnp.full((2,), 2**32 - 1, dtype=jnp.uint32),
        ),
    )
    result = jax.jit(builder.update_with_status)(
        exhausted,
        jnp.zeros((2,), dtype=jnp.float32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray(1.0, dtype=jnp.float32),
    )
    assert not bool(result.step_capacity_available)
    assert not bool(result.transition_applied)
    chex.assert_trees_all_equal(result.state, exhausted)

    update_exhausted = cast(
        RecurrentTraceUnitStateBuilderState,
        cast(Any, state).replace(
            update_count=jnp.asarray(2**31 - 1, dtype=jnp.int32),
            update_words=jnp.full((2,), 2**32 - 1, dtype=jnp.uint32),
        ),
    )
    proposal = builder.propose_learning_update(
        update_exhausted,
        jnp.ones((builder.feature_dim(),), dtype=jnp.float32),
    )
    unchanged, diagnostics = builder.commit_learning_update(update_exhausted, proposal)
    assert not bool(diagnostics.applied)
    chex.assert_trees_all_equal(unchanged, update_exhausted)


def test_rtu_proposal_tamper_arithmetic_overflow_and_uint64_rollover_rollback() -> None:
    builder = _builder()
    state, _ = builder.start(
        builder.init(jr.key(51, impl="threefry2x32")),
        jnp.asarray((0.2, -0.3), dtype=jnp.float32),
    )
    maximum_i32 = jnp.asarray(2**31 - 1, dtype=jnp.int32)
    maximum_u32 = jnp.asarray(2**32 - 1, dtype=jnp.uint32)

    step_rollover = cast(
        RecurrentTraceUnitStateBuilderState,
        cast(Any, state).replace(
            step_count=maximum_i32,
            step_words=jnp.asarray((3, maximum_u32), dtype=jnp.uint32),
        ),
    )
    advanced = jax.jit(builder.update_with_status)(
        step_rollover,
        jnp.asarray((0.1, 0.4), dtype=jnp.float32),
        jnp.asarray(1, dtype=jnp.int32),
        jnp.asarray(0.2, dtype=jnp.float32),
        jnp.asarray(0.9, dtype=jnp.float32),
    )
    assert bool(advanced.transition_applied)
    chex.assert_trees_all_equal(
        advanced.state.step_words,
        jnp.asarray((4, 0), dtype=jnp.uint32),
    )
    assert int(advanced.state.step_count) == 2**31 - 1

    update_rollover = cast(
        RecurrentTraceUnitStateBuilderState,
        cast(Any, state).replace(
            update_count=maximum_i32,
            update_words=jnp.asarray((7, maximum_u32), dtype=jnp.uint32),
        ),
    )
    proposal = builder.propose_learning_update(
        update_rollover,
        jnp.ones((builder.feature_dim(),), dtype=jnp.float32),
    )
    tampered = cast(
        Any,
        proposal,
    ).replace(gradient_norm=proposal.gradient_norm + jnp.float32(1.0))
    rejected_state, rejected = builder.commit_learning_update(
        update_rollover,
        tampered,
    )
    assert not bool(rejected.applied)
    chex.assert_trees_all_equal(rejected_state, update_rollover)
    committed, diagnostics = builder.commit_learning_update(update_rollover, proposal)
    assert bool(diagnostics.applied)
    chex.assert_trees_all_equal(
        committed.update_words,
        jnp.asarray((8, 0), dtype=jnp.uint32),
    )

    maximum = jnp.asarray(np.finfo(np.float32).max, dtype=jnp.float32)
    extreme = cast(
        RecurrentTraceUnitStateBuilderState,
        cast(Any, state).replace(parameters=jnp.full_like(state.parameters, maximum)),
    )
    overflow = jax.jit(builder.update_with_status)(
        extreme,
        jnp.full((2,), maximum, dtype=jnp.float32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray(1.0, dtype=jnp.float32),
    )
    assert not bool(overflow.candidate_state_valid)
    assert not bool(overflow.transition_applied)
    chex.assert_trees_all_equal(overflow.state, extreme)


def test_rtu_eager_loop_matches_jitted_scan_and_checkpoint_round_trip(
    tmp_path: Path,
) -> None:
    builder = _builder(rtrl_taylor_correction=True)
    initial = builder.init(jr.key(6, impl="threefry2x32"))
    observations = jnp.asarray(
        [[0.1, 0.2], [0.3, -0.4], [0.8, 0.6]],
        dtype=jnp.float32,
    )
    actions = jnp.asarray([-1, 0, 2], dtype=jnp.int32)
    rewards = jnp.asarray([0.0, 0.5, -0.2], dtype=jnp.float32)
    discounts = jnp.asarray([1.0, 0.95, 0.0], dtype=jnp.float32)

    loop_state = initial
    loop_features = []
    for inputs in zip(observations, actions, rewards, discounts, strict=True):
        loop_state, features = builder.update(loop_state, *inputs)
        loop_features.append(features)

    def scan_step(
        state: RecurrentTraceUnitStateBuilderState,
        inputs: tuple[jax.Array, ...],
    ) -> tuple[RecurrentTraceUnitStateBuilderState, jax.Array]:
        next_state, features = builder.update(state, *inputs)
        return next_state, features

    scan_state, scan_features = jax.jit(
        lambda state: jax.lax.scan(
            scan_step,
            state,
            (observations, actions, rewards, discounts),
        )
    )(initial)
    chex.assert_trees_all_close(scan_state, loop_state, rtol=1.0e-6, atol=1.0e-6)
    np.testing.assert_allclose(
        scan_features,
        jnp.stack(loop_features),
        rtol=1.0e-6,
        atol=1.0e-6,
    )

    checkpoint = tmp_path / "rtu-builder"
    save_state_builder_checkpoint(builder, loop_state, checkpoint)
    restored_builder, restored_state = load_state_builder_checkpoint(checkpoint)
    assert isinstance(restored_builder, RecurrentTraceUnitStateBuilder)
    assert restored_builder.to_config() == builder.to_config()
    chex.assert_trees_all_equal(restored_state, loop_state)
