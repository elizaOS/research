"""Mechanism tests for conventional option value plus expected duration.

These are analytic and deterministic development tests, not held-out evidence
that Alberta Plan Step 5 is complete.
"""

import chex
import jax
import jax.numpy as jnp
import pytest

from alberta_framework.core.option_value_duration import (
    OptionValueDurationConfig,
    OptionValueDurationLearner,
)
from alberta_framework.evaluation.option_value_duration_diagnostic import (
    FAST_OPTION,
    SLOW_OPTION,
    OptionValueDurationDiagnosticConfig,
    run_option_value_duration_diagnostic,
)

pytestmark = pytest.mark.development


def test_config_roundtrip_validation_and_fixed_parameter_count() -> None:
    config = OptionValueDurationConfig(
        reward_step_size=0.2,
        duration_step_size=0.3,
        duration_floor=1e-4,
    )
    learner = OptionValueDurationLearner.from_config(
        OptionValueDurationLearner(3, config).to_config()
    )

    assert learner.n_options == 3
    assert learner.config == config
    assert learner.trainable_parameter_count(feature_dim=5) == 3 * 2 * 5
    chex.assert_shape(learner.init(5).weights, (3, 2, 5))

    with pytest.raises(ValueError, match="n_options"):
        OptionValueDurationLearner(0)
    with pytest.raises(ValueError, match="reward_step_size"):
        OptionValueDurationConfig(reward_step_size=-0.1)
    with pytest.raises(ValueError, match="duration_step_size"):
        OptionValueDurationConfig(duration_step_size=-0.1)
    with pytest.raises(ValueError, match="duration_floor"):
        OptionValueDurationConfig(duration_floor=0.0)
    for invalid in (float("nan"), float("inf"), -float("inf")):
        with pytest.raises(ValueError, match="reward_step_size"):
            OptionValueDurationConfig(reward_step_size=invalid)
        with pytest.raises(ValueError, match="duration_step_size"):
            OptionValueDurationConfig(duration_step_size=invalid)
        with pytest.raises(ValueError, match="duration_floor"):
            OptionValueDurationConfig(duration_floor=invalid)


def test_two_head_td_targets_and_updates_match_exact_analytic_values() -> None:
    learner = OptionValueDurationLearner(
        2,
        OptionValueDurationConfig(
            reward_step_size=0.1,
            duration_step_size=0.2,
        ),
    )
    initial_weights = jnp.array(
        [
            [[2.0, -1.0], [0.5, 1.5]],
            [[7.0, 8.0], [9.0, 10.0]],
        ],
        dtype=jnp.float32,
    )
    state = learner.init(2).replace(weights=initial_weights)  # type: ignore[attr-defined]

    result = jax.jit(learner.update)(
        state,
        jnp.array([1.0, 2.0], dtype=jnp.float32),
        jnp.array(0, dtype=jnp.int32),
        jnp.array(3.0, dtype=jnp.float32),
        jnp.array([2.0, -1.0], dtype=jnp.float32),
        jnp.array(0.75, dtype=jnp.float32),
    )

    # predictions = [0, 3.5], next_predictions = [5, -0.5].
    # targets = [3, 1] + 0.75 * next_predictions = [6.75, 0.625].
    chex.assert_trees_all_close(
        result.predictions,
        jnp.array([0.0, 3.5], dtype=jnp.float32),
    )
    chex.assert_trees_all_close(
        result.next_predictions,
        jnp.array([5.0, -0.5], dtype=jnp.float32),
    )
    chex.assert_trees_all_close(
        result.td_targets,
        jnp.array([6.75, 0.625], dtype=jnp.float32),
    )
    chex.assert_trees_all_close(
        result.td_errors,
        jnp.array([6.75, -2.875], dtype=jnp.float32),
    )
    chex.assert_trees_all_close(
        result.state.weights[0],
        jnp.array([[2.675, 0.35], [-0.075, 0.35]], dtype=jnp.float32),
        atol=1e-6,
    )
    chex.assert_trees_all_close(result.state.weights[1], initial_weights[1])
    chex.assert_trees_all_equal(
        result.state.option_update_counts,
        jnp.array([1, 0], dtype=jnp.int32),
    )
    assert int(result.state.step_count) == 1


def test_termination_discount_zeros_bootstrap_and_no_average_reward_is_subtracted() -> None:
    learner = OptionValueDurationLearner(
        1,
        OptionValueDurationConfig(
            reward_step_size=0.0,
            duration_step_size=0.0,
        ),
    )
    state = learner.init(1).replace(  # type: ignore[attr-defined]
        weights=jnp.array([[[2.0], [7.0]]], dtype=jnp.float32)
    )

    result = learner.update(
        state,
        jnp.array([1.0], dtype=jnp.float32),
        jnp.array(0, dtype=jnp.int32),
        jnp.array(5.0, dtype=jnp.float32),
        jnp.array([100.0], dtype=jnp.float32),
        jnp.array(0.0, dtype=jnp.float32),
    )

    # An arbitrarily large next prediction cannot leak through termination.
    # The conventional reward target is the raw reward, not reward minus rbar.
    chex.assert_trees_all_close(
        result.td_targets,
        jnp.array([5.0, 1.0], dtype=jnp.float32),
    )
    assert not hasattr(result.state, "average_reward")


def test_reward_rate_prediction_preserves_raw_duration_and_floors_only_score() -> None:
    learner = OptionValueDurationLearner(
        2,
        OptionValueDurationConfig(duration_floor=0.5),
    )
    state = learner.init(1).replace(  # type: ignore[attr-defined]
        weights=jnp.array(
            [
                [[6.0], [10.0]],
                [[4.0], [0.0]],
            ],
            dtype=jnp.float32,
        )
    )

    prediction = learner.predict(state, jnp.array([1.0], dtype=jnp.float32))

    chex.assert_trees_all_close(
        prediction.reward_values,
        jnp.array([6.0, 4.0], dtype=jnp.float32),
    )
    chex.assert_trees_all_close(
        prediction.durations,
        jnp.array([10.0, 0.0], dtype=jnp.float32),
    )
    chex.assert_trees_all_close(
        prediction.reward_rates,
        jnp.array([0.6, 8.0], dtype=jnp.float32),
    )


def test_continuing_semi_markov_diagnostic_exposes_value_only_misranking() -> None:
    result = run_option_value_duration_diagnostic()

    assert result.evidence_level == "L1 deterministic development diagnostic"
    assert result.transition_count == 12 * (10 + 2)
    assert result.option_update_counts == (12 * 10, 12 * 2)
    assert result.learned_reward_values == pytest.approx((6.0, 4.0))
    assert result.learned_durations == pytest.approx((10.0, 2.0))
    assert result.learned_reward_rates == pytest.approx((0.6, 2.0))
    assert result.true_reward_rates == pytest.approx((0.6, 2.0))
    assert result.value_only_choice == SLOW_OPTION
    assert result.reward_rate_choice == FAST_OPTION
    assert result.optimal_reward_rate_choice == FAST_OPTION
    assert result.mechanism_passed


def test_diagnostic_rejects_non_diagnostic_or_undertrained_configuration() -> None:
    with pytest.raises(ValueError, match="returns must be finite"):
        OptionValueDurationDiagnosticConfig(slow_return=float("nan"))
    with pytest.raises(ValueError, match="slow_return"):
        OptionValueDurationDiagnosticConfig(slow_return=3.0, fast_return=4.0)
    with pytest.raises(ValueError, match="larger true reward rate"):
        OptionValueDurationDiagnosticConfig(
            slow_return=6.0,
            slow_duration=2,
            fast_return=4.0,
            fast_duration=2,
        )
    with pytest.raises(ValueError, match="longest TD"):
        OptionValueDurationDiagnosticConfig(executions_per_option=9)
