# mypy: disable-error-code="attr-defined"
"""Mechanism tests for the bounded ICLR 2025 spectral regularizer."""

import chex
import jax.numpy as jnp
import jax.random as jr
import pytest

from alberta_framework.core.spectral_regularization import (
    SPECTRAL_REGULARIZATION_SCIENTIFIC_PROMOTION_ALLOWED,
    SpectralRegularizationConfig,
    SpectralRegularizer,
)


@pytest.mark.unit
def test_scalar_paper_objective_and_gradient_are_exact() -> None:
    regularizer = SpectralRegularizer(
        SpectralRegularizationConfig(
            output_dim=1,
            input_dim=1,
            coefficient=0.1,
            exponent=2,
        )
    )
    result = regularizer.evaluate(
        regularizer.init(jr.key(7)),
        jnp.asarray([[2.0]], dtype=jnp.float32),
        jnp.asarray([3.0], dtype=jnp.float32),
    )

    assert bool(result.accepted)
    # (2**2 - 1)**2 + ||3||**4 = 9 + 81.
    assert float(result.regularizer) == pytest.approx(90.0)
    assert float(result.scaled_loss) == pytest.approx(9.0)
    # 0.1 * d/dw (w**2 - 1)**2 = 0.1 * 4*w*(w**2 - 1).
    chex.assert_trees_all_close(result.weight_gradient, jnp.asarray([[2.4]]))
    # 0.1 * d/db b**4.
    chex.assert_trees_all_close(result.bias_gradient, jnp.asarray([10.8]))
    assert float(result.spectral_norm_estimate) == pytest.approx(2.0)


@pytest.mark.unit
def test_nonfinite_input_rejects_atomically() -> None:
    regularizer = SpectralRegularizer(
        SpectralRegularizationConfig(output_dim=2, input_dim=2)
    )
    state = regularizer.init(jr.key(1))
    result = regularizer.evaluate(
        state,
        jnp.asarray([[1.0, jnp.nan], [0.0, 1.0]], dtype=jnp.float32),
        jnp.zeros((2,), dtype=jnp.float32),
    )

    assert not bool(result.accepted)
    chex.assert_trees_all_equal(result.state, state)
    chex.assert_trees_all_equal(result.weight_gradient, jnp.zeros((2, 2)))
    chex.assert_trees_all_equal(result.bias_gradient, jnp.zeros((2,)))


@pytest.mark.unit
def test_corrupted_nonnormalized_power_probe_rejects_atomically() -> None:
    regularizer = SpectralRegularizer(
        SpectralRegularizationConfig(output_dim=2, input_dim=2)
    )
    state = regularizer.init(jr.key(3))
    corrupted = state.replace(right_probe=jnp.zeros((2,), dtype=jnp.float32))
    result = regularizer.evaluate(
        corrupted,
        jnp.eye(2, dtype=jnp.float32),
        jnp.zeros((2,), dtype=jnp.float32),
    )

    assert not bool(result.accepted)
    chex.assert_trees_all_equal(result.state, corrupted)


@pytest.mark.unit
def test_lifetime_limit_fails_closed_without_counter_wrap() -> None:
    regularizer = SpectralRegularizer(
        SpectralRegularizationConfig(
            output_dim=1, input_dim=1, maximum_updates=1
        )
    )
    weight = jnp.ones((1, 1), dtype=jnp.float32)
    bias = jnp.zeros((1,), dtype=jnp.float32)
    first = regularizer.evaluate(regularizer.init(jr.key(2)), weight, bias)
    second = regularizer.evaluate(first.state, weight, bias)

    assert bool(first.accepted)
    assert not bool(second.accepted)
    assert bool(second.exhausted)
    chex.assert_trees_all_equal(second.state, first.state)


@pytest.mark.unit
def test_config_checkpoint_metadata_and_resources_are_strict() -> None:
    config = SpectralRegularizationConfig(
        output_dim=3,
        input_dim=4,
        coefficient=0.002,
        exponent=4,
        power_iterations=2,
        maximum_updates=99,
    )
    regularizer = SpectralRegularizer(config)
    restored = SpectralRegularizer.from_checkpoint_metadata(
        regularizer.checkpoint_metadata()
    )
    assert restored.to_config() == regularizer.to_config()
    resource = regularizer.resource_declaration()
    assert resource.parameter_count == 15
    assert resource.power_matvecs_per_evaluation == 4
    assert resource.backward_evaluations_per_update == 1
    assert SPECTRAL_REGULARIZATION_SCIENTIFIC_PROMOTION_ALLOWED is False

    malformed = dict(config.to_config())
    malformed["unexpected"] = 1
    with pytest.raises(ValueError, match="noncanonical"):
        SpectralRegularizationConfig.from_config(malformed)
