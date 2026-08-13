# mypy: disable-error-code="attr-defined,call-arg"
"""Mechanism tests for bounded Calibrated Partial Resets."""

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest
from jax import Array

from alberta_framework.core.calibrated_partial_resets import (
    CPR_BASE_OPTIMIZER_STATE_MUTATED,
    CPR_BIAS_POLICY,
    CPR_SCIENTIFIC_PROMOTION_ALLOWED,
    CalibratedPartialResets,
    CalibratedPartialResetsConfig,
    CalibratedPartialResetsParameters,
)


def _parameters() -> CalibratedPartialResetsParameters:
    return CalibratedPartialResetsParameters(
        incoming_weight=jnp.ones((1, 2), dtype=jnp.float32),
        outgoing_weight=jnp.ones((2, 1), dtype=jnp.float32),
    )


def _gradient_samples() -> Array:
    return jnp.asarray([[[1.0, 3.0]], [[1.0, 3.0]]], dtype=jnp.float32)


@pytest.mark.unit
def test_equations_3_and_5_use_per_example_layer_normalized_gradient_utility() -> None:
    controller = CalibratedPartialResets(
        CalibratedPartialResetsConfig(
            input_dim=1,
            unit_count=2,
            output_dim=1,
            utility_decay=0.5,
            update_frequency=10,
        )
    )
    state = controller.init(jr.key(11))
    result = controller.update_after_optimizer(
        state, _parameters(), _gradient_samples()
    )

    assert bool(result.accepted)
    assert not bool(result.reset_applied)
    chex.assert_trees_all_close(result.raw_utility, jnp.asarray([1.0, 3.0]))
    chex.assert_trees_all_close(
        result.normalized_utility, jnp.asarray([0.5, 1.5]), atol=1e-7
    )
    chex.assert_trees_all_close(result.state.utility, jnp.asarray([0.75, 1.25]))
    chex.assert_trees_all_equal(result.parameters, _parameters())
    chex.assert_trees_all_equal(jr.key_data(result.state.rng_key), jr.key_data(state.rng_key))


@pytest.mark.unit
def test_equations_6_and_7_apply_calibrated_partial_reset_on_source_clock() -> None:
    config = CalibratedPartialResetsConfig(
        input_dim=1,
        unit_count=2,
        output_dim=1,
        replacement_rate=0.2,
        sharpness=4.0,
        utility_decay=0.0,
        update_frequency=2,
    )
    controller = CalibratedPartialResets(config)
    state = controller.init(jr.key(19))
    params = _parameters()

    first = controller.update_after_optimizer(state, params, _gradient_samples())
    second = controller.update_after_optimizer(
        first.state, first.parameters, _gradient_samples()
    )
    third = controller.update_after_optimizer(
        second.state, second.parameters, _gradient_samples()
    )

    # The official implementation tests source time_step before increment:
    # initialized t=0 => reset on the third call when source t=2.
    assert not bool(first.reset_applied)
    assert not bool(second.reset_applied)
    assert bool(third.reset_applied)
    expected_transform = jnp.minimum(
        2.0 * jax.nn.sigmoid(-4.0 * (jnp.asarray([0.5, 1.5]) - 1.0)),
        1.0,
    )
    expected_fraction = 0.2 * expected_transform
    chex.assert_trees_all_close(third.reset_fraction, expected_fraction, atol=1e-7)
    assert float(third.reset_fraction[0]) == pytest.approx(0.2)
    assert float(third.reset_fraction[1]) < float(third.reset_fraction[0])

    next_key, initialization_key = jr.split(state.rng_key)
    bound = jnp.sqrt(jnp.asarray(6.0, dtype=jnp.float32))
    fresh = jr.uniform(
        initialization_key,
        (1, 2),
        dtype=jnp.float32,
        minval=-bound,
        maxval=bound,
    )
    expected_incoming = (1.0 - expected_fraction[None, :]) + (
        expected_fraction[None, :] * fresh
    )
    expected_outgoing = (1.0 - expected_fraction[:, None])
    chex.assert_trees_all_close(
        third.parameters.incoming_weight, expected_incoming, atol=1e-6
    )
    chex.assert_trees_all_close(
        third.parameters.outgoing_weight, expected_outgoing, atol=1e-6
    )
    chex.assert_trees_all_equal(third.state.utility, jnp.ones((2,)))
    chex.assert_trees_all_equal(
        jr.key_data(third.state.rng_key), jr.key_data(next_key)
    )


@pytest.mark.unit
def test_nonfinite_gradient_rejects_without_rng_or_clock_advance() -> None:
    controller = CalibratedPartialResets(
        CalibratedPartialResetsConfig(input_dim=1, unit_count=2, output_dim=1)
    )
    state = controller.init(jr.key(5))
    gradients = _gradient_samples().at[0, 0, 0].set(jnp.nan)
    result = controller.update_after_optimizer(state, _parameters(), gradients)

    assert not bool(result.accepted)
    chex.assert_trees_all_equal(result.state, state)
    chex.assert_trees_all_equal(result.parameters, _parameters())


@pytest.mark.unit
def test_exhausted_reset_event_clock_rolls_back_the_whole_transaction() -> None:
    controller = CalibratedPartialResets(
        CalibratedPartialResetsConfig(
            input_dim=1,
            unit_count=2,
            output_dim=1,
            update_frequency=1,
        )
    )
    state = controller.init(jr.key(6)).replace(
        update_count_words=jnp.asarray([0, 1], dtype=jnp.uint32),
        reset_event_count_words=jnp.asarray(
            [4_294_967_295, 4_294_967_295], dtype=jnp.uint32
        ),
    )
    result = controller.update_after_optimizer(
        state, _parameters(), _gradient_samples()
    )

    assert not bool(result.accepted)
    assert bool(result.exhausted)
    assert not bool(result.reset_applied)
    chex.assert_trees_all_equal(result.state, state)
    chex.assert_trees_all_equal(result.parameters, _parameters())


@pytest.mark.unit
def test_reset_clock_ahead_of_update_clock_is_rejected_as_corruption() -> None:
    controller = CalibratedPartialResets(
        CalibratedPartialResetsConfig(input_dim=1, unit_count=2, output_dim=1)
    )
    state = controller.init(jr.key(9)).replace(
        reset_event_count_words=jnp.asarray([0, 1], dtype=jnp.uint32)
    )
    result = controller.update_after_optimizer(
        state, _parameters(), _gradient_samples()
    )

    assert not bool(result.accepted)
    chex.assert_trees_all_equal(result.state, state)


@pytest.mark.unit
def test_config_checkpoint_resources_and_scope_disclosures_are_strict() -> None:
    controller = CalibratedPartialResets(
        CalibratedPartialResetsConfig(
            input_dim=3,
            unit_count=4,
            output_dim=2,
            replacement_rate=0.05,
            maximum_updates=100,
        )
    )
    restored = CalibratedPartialResets.from_checkpoint_metadata(
        controller.checkpoint_metadata()
    )
    assert restored.to_config() == controller.to_config()
    resource = controller.resource_declaration()
    assert resource.caller_owned_parameter_count == 20
    assert resource.persistent_bytes == 40
    assert resource.initialization_draws_per_reset_event == 12
    assert resource.base_optimizer_state_bytes_owned == 0
    assert CPR_BASE_OPTIMIZER_STATE_MUTATED is False
    assert CPR_SCIENTIFIC_PROMOTION_ALLOWED is False
    assert "excluded" in CPR_BIAS_POLICY

    with pytest.raises(ValueError, match="sample, input, unit"):
        controller.update_after_optimizer(
            controller.init(jr.key(0)),
            CalibratedPartialResetsParameters(
                incoming_weight=jnp.ones((3, 4), dtype=jnp.float32),
                outgoing_weight=jnp.ones((4, 2), dtype=jnp.float32),
            ),
            jnp.ones((3, 4), dtype=jnp.float32),
        )
