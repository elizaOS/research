# mypy: disable-error-code="attr-defined,call-arg,no-untyped-def,no-untyped-call,assignment"
"""JIT, scan, checkpoint, and public-surface tests for the WP2 mechanism arms."""

from pathlib import Path

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from alberta_framework.core.adam_o import AdamO, AdamOConfig
from alberta_framework.core.calibrated_partial_resets import (
    CalibratedPartialResets,
    CalibratedPartialResetsConfig,
    CalibratedPartialResetsParameters,
)
from alberta_framework.core.checkpoints import load_checkpoint, save_checkpoint
from alberta_framework.core.spectral_regularization import (
    SpectralRegularizationConfig,
    SpectralRegularizer,
)


def _with_raw_key_data(tree):
    """Make typed PRNG leaves NumPy-comparable for Chex assertions."""

    def convert(value):
        if hasattr(value, "dtype") and jax.dtypes.issubdtype(
            value.dtype, jax.dtypes.prng_key
        ):
            return jr.key_data(value)
        return value

    return jax.tree.map(convert, tree)


@pytest.mark.integration
def test_spectral_regularizer_eager_jit_and_scan_are_deterministic() -> None:
    controller = SpectralRegularizer(
        SpectralRegularizationConfig(output_dim=2, input_dim=2)
    )
    state = controller.init(jr.key(4))
    weight = jnp.asarray([[1.3, 0.1], [0.2, 0.7]], dtype=jnp.float32)
    bias = jnp.asarray([0.1, -0.2], dtype=jnp.float32)

    eager = controller.evaluate(state, weight, bias)
    compiled = jax.jit(controller.evaluate)(state, weight, bias)
    chex.assert_trees_all_close(
        _with_raw_key_data(eager), _with_raw_key_data(compiled), atol=1e-7
    )

    def body(carry, _: jax.Array):
        result = controller.evaluate(carry, weight, bias)
        return result.state, result.scaled_loss

    scan_state, losses = jax.jit(
        lambda initial: jax.lax.scan(body, initial, jnp.arange(3))[0:2]
    )(state)
    manual_state = state
    manual_losses = []
    for _ in range(3):
        result = controller.evaluate(manual_state, weight, bias)
        manual_state = result.state
        manual_losses.append(result.scaled_loss)
    chex.assert_trees_all_close(
        _with_raw_key_data(scan_state), _with_raw_key_data(manual_state), atol=1e-7
    )
    chex.assert_trees_all_close(losses, jnp.stack(manual_losses), atol=1e-7)


@pytest.mark.integration
def test_adamo_eager_jit_and_scan_are_deterministic() -> None:
    optimizer = AdamO(AdamOConfig(rows=2, columns=2))
    state = optimizer.init()
    weight = jnp.asarray([[1.2, 0.1], [0.0, 0.8]], dtype=jnp.float32)
    gradient = jnp.asarray([[0.2, -0.1], [0.3, -0.4]], dtype=jnp.float32)

    eager = optimizer.update(state, weight, gradient)
    compiled = jax.jit(optimizer.update)(state, weight, gradient)
    chex.assert_trees_all_close(
        _with_raw_key_data(eager), _with_raw_key_data(compiled), atol=1e-7
    )

    def body(carry, _: jax.Array):
        current_state, current_weight = carry
        result = optimizer.update(current_state, current_weight, gradient)
        return (result.state, current_weight - result.parameter_delta), result.parameter_delta

    (scan_state, scan_weight), deltas = jax.jit(
        lambda initial: jax.lax.scan(body, initial, jnp.arange(3))
    )((state, weight))
    manual_state = state
    manual_weight = weight
    manual_deltas = []
    for _ in range(3):
        result = optimizer.update(manual_state, manual_weight, gradient)
        manual_state = result.state
        manual_weight = manual_weight - result.parameter_delta
        manual_deltas.append(result.parameter_delta)
    chex.assert_trees_all_close(
        _with_raw_key_data(scan_state), _with_raw_key_data(manual_state), atol=1e-7
    )
    chex.assert_trees_all_close(scan_weight, manual_weight, atol=1e-7)
    chex.assert_trees_all_close(deltas, jnp.stack(manual_deltas), atol=1e-7)


@pytest.mark.integration
def test_cpr_eager_jit_and_scan_have_identical_rng_ownership() -> None:
    controller = CalibratedPartialResets(
        CalibratedPartialResetsConfig(
            input_dim=1,
            unit_count=2,
            output_dim=1,
            update_frequency=2,
            utility_decay=0.0,
        )
    )
    state = controller.init(jr.key(8))
    parameters = CalibratedPartialResetsParameters(
        incoming_weight=jnp.ones((1, 2), dtype=jnp.float32),
        outgoing_weight=jnp.ones((2, 1), dtype=jnp.float32),
    )
    gradients = jnp.asarray([[[1.0, 2.0]], [[1.0, 2.0]]], dtype=jnp.float32)
    eager = controller.update_after_optimizer(state, parameters, gradients)
    compiled = jax.jit(controller.update_after_optimizer)(
        state, parameters, gradients
    )
    chex.assert_trees_all_close(
        _with_raw_key_data(eager), _with_raw_key_data(compiled), atol=1e-7
    )

    def body(carry, _: jax.Array):
        current_state, current_parameters = carry
        result = controller.update_after_optimizer(
            current_state, current_parameters, gradients
        )
        return (result.state, result.parameters), result.reset_applied

    (scan_state, scan_parameters), reset_trace = jax.jit(
        lambda initial: jax.lax.scan(body, initial, jnp.arange(5))
    )((state, parameters))
    manual_state = state
    manual_parameters = parameters
    manual_trace = []
    for _ in range(5):
        result = controller.update_after_optimizer(
            manual_state, manual_parameters, gradients
        )
        manual_state = result.state
        manual_parameters = result.parameters
        manual_trace.append(result.reset_applied)
    chex.assert_trees_all_close(
        _with_raw_key_data(scan_state), _with_raw_key_data(manual_state), atol=1e-7
    )
    chex.assert_trees_all_close(scan_parameters, manual_parameters, atol=1e-7)
    chex.assert_trees_all_equal(reset_trace, jnp.stack(manual_trace))


@pytest.mark.integration
@pytest.mark.parametrize("arm", ["spectral", "adamo", "cpr"])
def test_states_round_trip_through_repository_checkpoint_helper(
    tmp_path: Path, arm: str
) -> None:
    if arm == "spectral":
        controller = SpectralRegularizer(
            SpectralRegularizationConfig(output_dim=1, input_dim=1)
        )
        template = controller.init(jr.key(1))
        state = controller.evaluate(
            template, jnp.ones((1, 1), dtype=jnp.float32), jnp.zeros((1,))
        ).state
    elif arm == "adamo":
        controller = AdamO(AdamOConfig(rows=1, columns=1))
        template = controller.init()
        state = controller.update(
            template, jnp.ones((1, 1)), jnp.ones((1, 1))
        ).state
    else:
        controller = CalibratedPartialResets(
            CalibratedPartialResetsConfig(
                input_dim=1, unit_count=1, output_dim=1
            )
        )
        template = controller.init(jr.key(2))
        state = controller.update_after_optimizer(
            template,
            CalibratedPartialResetsParameters(
                incoming_weight=jnp.ones((1, 1)),
                outgoing_weight=jnp.ones((1, 1)),
            ),
            jnp.ones((1, 1, 1)),
        ).state

    checkpoint = tmp_path / arm
    save_checkpoint(state, checkpoint, metadata=controller.checkpoint_metadata())
    restored_state, metadata = load_checkpoint(template, checkpoint)
    chex.assert_trees_all_equal(restored_state, state)
    assert metadata == controller.checkpoint_metadata()


@pytest.mark.integration
def test_wp2_arms_are_exported_from_core_and_package_root() -> None:
    import alberta_framework
    import alberta_framework.core as core

    for name in (
        "AdamO",
        "AdamOConfig",
        "CalibratedPartialResets",
        "CalibratedPartialResetsConfig",
        "SpectralRegularizer",
        "SpectralRegularizationConfig",
    ):
        assert hasattr(core, name)
        assert hasattr(alberta_framework, name)
