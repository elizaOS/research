# mypy: disable-error-code="attr-defined"
"""Equation-level tests for the clean-room AdamO matrix transform."""

import chex
import jax.numpy as jnp
import pytest

from alberta_framework.core.adam_o import (
    ADAMO_SCIENTIFIC_PROMOTION_ALLOWED,
    AdamO,
    AdamOConfig,
    orthogonality_gradient,
    orthogonality_regularizer,
)


@pytest.mark.unit
def test_equation_16_tall_and_wide_closed_forms() -> None:
    tall = jnp.asarray([[2.0], [0.0]], dtype=jnp.float32)
    wide = tall.T

    assert float(orthogonality_regularizer(tall)) == pytest.approx(9.0)
    assert float(orthogonality_regularizer(wide)) == pytest.approx(9.0)
    chex.assert_trees_all_close(
        orthogonality_gradient(tall),
        jnp.asarray([[24.0], [0.0]], dtype=jnp.float32),
    )
    chex.assert_trees_all_close(
        orthogonality_gradient(wide),
        jnp.asarray([[24.0, 0.0]], dtype=jnp.float32),
    )


@pytest.mark.unit
def test_equations_19_and_20_keep_task_moments_separate() -> None:
    optimizer = AdamO(
        AdamOConfig(
            rows=1,
            columns=1,
            learning_rate=0.1,
            orthogonality_strength=0.5,
            isometry_step_size=0.2,
        )
    )
    result = optimizer.update(
        optimizer.init(),
        jnp.asarray([[2.0]], dtype=jnp.float32),
        jnp.asarray([[3.0]], dtype=jnp.float32),
    )

    assert bool(result.accepted)
    # At t=1 Adam's bias-corrected task direction is 3 / sqrt(9) = 1.
    chex.assert_trees_all_close(result.task_delta, jnp.asarray([[0.1]]), atol=1e-6)
    # Equation 16 gradient is 24; eta_iso * lambda * r = .2 * .5 * 24.
    chex.assert_trees_all_close(
        result.isometry_delta, jnp.asarray([[2.4]]), atol=1e-6
    )
    chex.assert_trees_all_close(
        result.parameter_delta, jnp.asarray([[2.5]]), atol=1e-6
    )
    # Moments contain only the task gradient, never the +24 regularizer gradient.
    chex.assert_trees_all_close(result.state.first_moment, jnp.asarray([[0.3]]))
    expected_second = (1.0 - optimizer.config.beta2) * jnp.asarray([[9.0]])
    chex.assert_trees_all_close(result.state.second_moment, expected_second)


@pytest.mark.unit
def test_zero_task_gradient_can_take_only_the_isometry_step() -> None:
    optimizer = AdamO(
        AdamOConfig(rows=1, columns=1, orthogonality_strength=1.0)
    )
    result = optimizer.update(
        optimizer.init(),
        jnp.asarray([[2.0]], dtype=jnp.float32),
        jnp.zeros((1, 1), dtype=jnp.float32),
    )

    chex.assert_trees_all_equal(result.task_delta, jnp.zeros((1, 1)))
    assert float(result.isometry_delta[0, 0]) > 0.0
    chex.assert_trees_all_equal(result.state.first_moment, jnp.zeros((1, 1)))
    chex.assert_trees_all_equal(result.state.second_moment, jnp.zeros((1, 1)))


@pytest.mark.unit
def test_nonfinite_proposal_and_exhaustion_are_atomic() -> None:
    optimizer = AdamO(AdamOConfig(rows=1, columns=1, maximum_updates=1))
    state = optimizer.init()
    invalid = optimizer.update(
        state,
        jnp.asarray([[jnp.inf]], dtype=jnp.float32),
        jnp.ones((1, 1), dtype=jnp.float32),
    )
    assert not bool(invalid.accepted)
    chex.assert_trees_all_equal(invalid.state, state)

    first = optimizer.update(
        state,
        jnp.ones((1, 1), dtype=jnp.float32),
        jnp.ones((1, 1), dtype=jnp.float32),
    )
    exhausted = optimizer.update(
        first.state,
        jnp.ones((1, 1), dtype=jnp.float32),
        jnp.ones((1, 1), dtype=jnp.float32),
    )
    assert bool(first.accepted)
    assert not bool(exhausted.accepted)
    assert bool(exhausted.exhausted)
    chex.assert_trees_all_equal(exhausted.state, first.state)


@pytest.mark.unit
def test_corrupted_negative_second_moment_rejects_atomically() -> None:
    optimizer = AdamO(AdamOConfig(rows=1, columns=1))
    corrupted = optimizer.init().replace(
        second_moment=jnp.asarray([[-1.0]], dtype=jnp.float32)
    )
    result = optimizer.update(
        corrupted,
        jnp.ones((1, 1), dtype=jnp.float32),
        jnp.ones((1, 1), dtype=jnp.float32),
    )

    assert not bool(result.accepted)
    chex.assert_trees_all_equal(result.state, corrupted)


@pytest.mark.unit
def test_config_checkpoint_metadata_and_resources_are_strict() -> None:
    optimizer = AdamO(
        AdamOConfig(
            rows=3,
            columns=2,
            learning_rate=0.02,
            orthogonality_strength=0.4,
            maximum_updates=50,
        )
    )
    restored = AdamO.from_checkpoint_metadata(optimizer.checkpoint_metadata())
    assert restored.to_config() == optimizer.to_config()
    resource = optimizer.resource_declaration()
    assert resource.parameter_count == 6
    assert resource.persistent_bytes == 56
    assert resource.gram_dimension == 2
    assert resource.gram_matrix_elements == 4
    assert ADAMO_SCIENTIFIC_PROMOTION_ALLOWED is False
