# mypy: disable-error-code="attr-defined,call-arg,type-var"
"""Integration checks for JAX execution and long-stream SNR behavior."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import Any, cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.self_normalized_resets import (
    DenseReLUFreshParameters,
    DenseReLUParameters,
    DenseReLUResetTarget,
    SelfNormalizedResetConfig,
    SelfNormalizedResets,
    SelfNormalizedResetState,
    empty_dense_relu_optimizer_state,
)

pytestmark = pytest.mark.integration


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _controller(
    *,
    window_size: int,
    min_intervals: int,
    eta: float,
    maximum_updates: int = 10_000,
    max_total_resets: int = 20,
) -> tuple[SelfNormalizedResets, DenseReLUResetTarget, DenseReLUFreshParameters]:
    config = SelfNormalizedResetConfig(
        input_dim=3,
        unit_count=1,
        output_dim=2,
        source_sha256=_digest("integration-source"),
        representation_sha256=_digest("integration-representation"),
        window_size=window_size,
        min_intervals=min_intervals,
        warmup_observations=min_intervals,
        rejection_percentile=eta,
        initialization_mode="caller_provided",
        maximum_updates=maximum_updates,
        max_total_resets=max_total_resets,
    )
    parameters = DenseReLUParameters(
        incoming_weight=jnp.asarray([[1.0], [2.0], [3.0]], dtype=jnp.float32),
        bias=jnp.asarray([0.5], dtype=jnp.float32),
        outgoing_weight=jnp.asarray([[4.0, 5.0]], dtype=jnp.float32),
    )
    target = DenseReLUResetTarget(
        parameters=parameters,
        optimizer=empty_dense_relu_optimizer_state(),
    )
    fresh = DenseReLUFreshParameters(
        incoming_weight=jnp.asarray([[7.0], [8.0], [9.0]], dtype=jnp.float32),
        bias=jnp.asarray([0.0], dtype=jnp.float32),
    )
    return SelfNormalizedResets(config), target, fresh


def _broadcast_tree(tree: object, steps: int) -> Any:
    return jax.tree_util.tree_map(
        lambda value: jnp.broadcast_to(value, (steps, *value.shape)), tree
    )


def _assert_tree_equal(left: object, right: object) -> None:
    left_leaves = jax.tree_util.tree_leaves(left)
    right_leaves = jax.tree_util.tree_leaves(right)
    assert len(left_leaves) == len(right_leaves)
    for lhs, rhs in zip(left_leaves, right_leaves, strict=True):
        if jax.dtypes.issubdtype(lhs.dtype, jax.dtypes.prng_key):
            lhs = jr.key_data(lhs)
            rhs = jr.key_data(rhs)
        if lhs.dtype == jnp.float32:
            lhs = jax.lax.bitcast_convert_type(lhs, jnp.uint32)
            rhs = jax.lax.bitcast_convert_type(rhs, jnp.uint32)
        np.testing.assert_array_equal(lhs, rhs)


def test_eager_jit_python_loop_and_scan_have_exact_parity() -> None:
    controller, target, fresh = _controller(window_size=3, min_intervals=1, eta=0.2)
    initial = controller.init(jr.key(31), target)
    activation = jnp.asarray([1.0], dtype=jnp.float32)
    eager = controller._step_kernel(initial, activation, target, fresh)
    compiled = jax.jit(
        lambda state, values, post_target, fresh_values: controller._step_kernel(
            state, values, post_target, fresh_values
        )
    )(initial, activation, target, fresh)
    _assert_tree_equal(eager, compiled)

    activations = jnp.asarray(
        [[1.0], [0.0], [1.0], [0.0], [0.0], [0.0], [1.0]],
        dtype=jnp.float32,
    )
    targets = _broadcast_tree(target, activations.shape[0])
    fresh_values = _broadcast_tree(fresh, activations.shape[0])
    scanned = controller.run_scan(initial, activations, targets, fresh_values)

    loop_state = initial
    loop_masks = []
    loop_log_survival = []
    for values in activations:
        result = controller.step(loop_state, values, target, fresh)
        loop_state = result.state
        loop_masks.append(result.reset_mask)
        loop_log_survival.append(result.log_survival)
    _assert_tree_equal(scanned.state, loop_state)
    np.testing.assert_array_equal(scanned.reset_masks, jnp.stack(loop_masks))
    np.testing.assert_array_equal(
        jax.lax.bitcast_convert_type(scanned.log_survivals, jnp.uint32),
        jax.lax.bitcast_convert_type(jnp.stack(loop_log_survival), jnp.uint32),
    )

    compiled_scan = jax.jit(
        lambda state, values, post_targets, reset_values: controller.run_scan(
            state, values, post_targets, reset_values
        )
    )(initial, activations, targets, fresh_values)
    _assert_tree_equal(scanned, compiled_scan)


def test_long_bernoulli_stream_resets_only_on_statistically_surprising_inactivity() -> None:
    eta = 1.0e-8
    controller, target, fresh = _controller(
        window_size=64,
        min_intervals=64,
        eta=eta,
        maximum_updates=2_000,
        max_total_resets=1,
    )
    state = controller.init(jr.key(32), target)
    training_steps = 1_200
    nominal = jr.bernoulli(
        jr.key(33), p=jnp.asarray(0.25, jnp.float32), shape=(training_steps, 1)
    ).astype(jnp.float32)
    forced_inactivity = jnp.zeros((120, 1), dtype=jnp.float32)
    activations = jnp.concatenate((nominal, forced_inactivity), axis=0)
    targets = _broadcast_tree(target, activations.shape[0])
    fresh_values = _broadcast_tree(fresh, activations.shape[0])

    result = controller.run_scan(state, activations, targets, fresh_values)
    reset_indices = np.flatnonzero(np.asarray(result.reset_masks[:, 0]))

    assert reset_indices.size == 1
    reset_index = int(reset_indices[0])
    assert training_steps <= reset_index < activations.shape[0]
    assert float(result.log_survivals[reset_index, 0]) <= np.log(eta)
    assert int(result.state.total_reset_count) == 1
    assert int(result.state.interval_count[0]) == 64
    # With nominal p=0.25, a 1e-8 tail requires a long run. This guards
    # against silently replacing SNR with a short fixed dead-unit threshold.
    assert reset_index - training_steps >= 35


def test_dense_forward_sgd_transition_and_next_step_reuse_reset_slice() -> None:
    base, target, fresh = _controller(window_size=2, min_intervals=1, eta=0.25)
    controller = SelfNormalizedResets(replace(base.config, optimizer_kind="sgd"))
    state = controller.init(jr.key(34), target)

    def activation(current: SelfNormalizedResetState, features: jax.Array) -> jax.Array:
        parameters = current.target.parameters
        return jnp.maximum(
            features @ parameters.incoming_weight + parameters.bias,
            jnp.asarray(0.0, dtype=jnp.float32),
        )

    def sgd_transition(current: SelfNormalizedResetState, step_size: float) -> DenseReLUResetTarget:
        parameters = current.target.parameters
        post_parameters = parameters.replace(
            outgoing_weight=parameters.outgoing_weight - jnp.asarray(step_size, dtype=jnp.float32)
        )
        return cast(
            DenseReLUResetTarget,
            current.target.replace(parameters=post_parameters),
        )

    positive_features = jnp.zeros((3,), dtype=jnp.float32)
    for _ in range(2):
        values = activation(state, positive_features)
        assert float(values[0]) > 0.0
        state = controller.step(state, values, sgd_transition(state, 0.1), fresh).state

    silent_features = -jnp.ones((3,), dtype=jnp.float32) * 10.0
    silent_values = activation(state, silent_features)
    np.testing.assert_array_equal(silent_values, [0.0])
    reset_result = controller.step(state, silent_values, sgd_transition(state, 0.1), fresh)
    assert bool(reset_result.reset_mask[0])
    np.testing.assert_array_equal(
        reset_result.state.target.parameters.incoming_weight,
        fresh.incoming_weight,
    )
    np.testing.assert_array_equal(reset_result.state.target.parameters.outgoing_weight, 0.0)

    # The next real forward consumes the reset slice. Its stateless SGD target
    # is derived from that returned state, rather than replaying the old target.
    next_values = activation(reset_result.state, jnp.ones((3,), dtype=jnp.float32))
    np.testing.assert_array_equal(next_values, [24.0])
    next_result = controller.step(
        reset_result.state,
        next_values,
        sgd_transition(reset_result.state, 0.25),
        fresh,
    )
    np.testing.assert_array_equal(
        next_result.state.target.parameters.incoming_weight,
        fresh.incoming_weight,
    )
    np.testing.assert_array_equal(next_result.state.target.parameters.outgoing_weight, -0.25)
