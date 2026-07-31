"""Mechanism tests for the publication-shaped slowly-changing task."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import pytest

from alberta_framework.benchmarks.slowly_changing_regression import (
    SCR_LEARNER_KINDS,
    SCRLearnerParams,
    SlowlyChangingRegressionConfig,
    describe_scr_curve_windows,
    make_scr_env,
    run_scr_binned_errors,
    scr_example,
    summarize_scr_curve,
)

TINY = SlowlyChangingRegressionConfig(
    num_bits=8,
    num_flipping_bits=4,
    flip_period=50,
    target_hidden_units=20,
    num_examples=600,
)


class TestConfig:
    def test_paper_defaults(self):
        config = SlowlyChangingRegressionConfig()
        assert config.num_bits == 20
        assert config.num_flipping_bits == 15
        assert config.flip_period == 10_000
        assert config.target_hidden_units == 100
        assert config.ltu_beta == 0.7
        assert config.num_examples == 3_000_000
        assert config.feature_dim == 21
        assert config.num_segments == 300

    def test_validate_rejects_bad_flipping_bits(self):
        config = SlowlyChangingRegressionConfig(num_bits=8, num_flipping_bits=9)
        with pytest.raises(ValueError, match="num_flipping_bits"):
            config.validate()

    def test_validate_rejects_zero_flip_period(self):
        config = SlowlyChangingRegressionConfig(flip_period=0)
        with pytest.raises(ValueError, match="flip_period"):
            config.validate()


class TestEnv:
    def test_target_net_deterministic_per_seed(self):
        env_a = make_scr_env(TINY, jr.key(7))
        env_b = make_scr_env(TINY, jr.key(7))
        assert jnp.array_equal(env_a.input_weights, env_b.input_weights)
        assert jnp.array_equal(env_a.thresholds, env_b.thresholds)
        assert jnp.array_equal(env_a.output_weights, env_b.output_weights)
        assert jnp.array_equal(env_a.slow_bits, env_b.slow_bits)
        for t in (0, 123, 599):
            x_a, y_a = scr_example(env_a, TINY, t)
            x_b, y_b = scr_example(env_b, TINY, t)
            assert jnp.array_equal(x_a, x_b)
            assert jnp.array_equal(y_a, y_b)

    def test_target_net_differs_across_seeds(self):
        env_a = make_scr_env(TINY, jr.key(0))
        env_b = make_scr_env(TINY, jr.key(1))
        assert not jnp.array_equal(env_a.input_weights, env_b.input_weights)

    def test_target_net_weights_are_signs(self):
        env = make_scr_env(TINY, jr.key(3))
        assert set(jnp.unique(env.input_weights).tolist()) <= {-1.0, 1.0}
        assert set(jnp.unique(env.output_weights).tolist()) <= {-1.0, 1.0}

    def test_ltu_threshold_formula(self):
        env = make_scr_env(TINY, jr.key(3))
        s = jnp.sum(env.input_weights < 0, axis=0).astype(jnp.float32)
        expected = TINY.feature_dim * TINY.ltu_beta - s
        assert jnp.allclose(env.thresholds, expected)

    def test_target_output_matches_manual_ltu(self):
        env = make_scr_env(TINY, jr.key(5))
        x, y = scr_example(env, TINY, 42)
        ltu = (x @ env.input_weights > env.thresholds).astype(jnp.float32)
        assert jnp.allclose(y, jnp.dot(ltu, env.output_weights))
        assert set(jnp.unique(ltu).tolist()) <= {0.0, 1.0}

    def test_flip_schedule_changes_exactly_one_bit_per_segment(self):
        env = make_scr_env(TINY, jr.key(11))
        assert env.slow_bits.shape == (TINY.num_segments, TINY.num_flipping_bits)
        diffs = jnp.sum(jnp.abs(env.slow_bits[1:] - env.slow_bits[:-1]), axis=1)
        assert jnp.array_equal(diffs, jnp.ones_like(diffs))

    def test_slow_bits_constant_within_segment_and_bias_always_one(self):
        env = make_scr_env(TINY, jr.key(13))
        f = TINY.num_flipping_bits
        period = TINY.flip_period
        for t in (0, 1, period - 1, period, 2 * period + 3):
            x, _ = scr_example(env, TINY, t)
            assert x.shape == (TINY.feature_dim,)
            assert jnp.array_equal(x[:f], env.slow_bits[t // period])
            assert float(x[-1]) == 1.0
            assert set(jnp.unique(x).tolist()) <= {0.0, 1.0}

    def test_fast_bits_vary_within_segment(self):
        env = make_scr_env(TINY, jr.key(17))
        f = TINY.num_flipping_bits
        fasts = jnp.stack([scr_example(env, TINY, t)[0][f:-1] for t in range(20)])
        # 5 i.i.d. fast bits over 20 examples: all-identical is (2^-4)^19.
        assert float(jnp.std(fasts)) > 0.0


class TestRunner:
    @pytest.mark.parametrize("kind", SCR_LEARNER_KINDS)
    def test_binned_error_shapes_and_finiteness(self, kind):
        binned = run_scr_binned_errors(
            kind, TINY, SCRLearnerParams(), num_runs=2, seed=0, bin_size=100
        )
        assert binned.shape == (2, 6)
        assert bool(jnp.all(jnp.isfinite(binned)))
        assert bool(jnp.all(binned >= 0.0))

    def test_runner_deterministic_per_seed(self):
        a = run_scr_binned_errors("sgd", TINY, SCRLearnerParams(), num_runs=2, seed=3, bin_size=100)
        b = run_scr_binned_errors("sgd", TINY, SCRLearnerParams(), num_runs=2, seed=3, bin_size=100)
        assert jnp.array_equal(a, b)

    def test_runner_rejects_indivisible_bin_size(self):
        with pytest.raises(ValueError, match="multiple"):
            run_scr_binned_errors("sgd", TINY, SCRLearnerParams(), num_runs=1, seed=0, bin_size=7)

    def test_runner_rejects_unknown_kind(self):
        with pytest.raises(ValueError, match="unknown learner kind"):
            run_scr_binned_errors(
                "adamw", TINY, SCRLearnerParams(), num_runs=1, seed=0, bin_size=100
            )

    @pytest.mark.parametrize("kind", SCR_LEARNER_KINDS)
    def test_small_scale_learning_ordering_smoke(self, kind):
        """First bin (untrained net) error exceeds the best later bin."""
        config = SlowlyChangingRegressionConfig(
            num_bits=8,
            num_flipping_bits=4,
            flip_period=200,
            target_hidden_units=20,
            num_examples=2_000,
        )
        binned = run_scr_binned_errors(
            kind, config, SCRLearnerParams(), num_runs=3, seed=0, bin_size=200
        )
        mean_curve = jnp.mean(binned, axis=0)
        assert float(mean_curve[0]) > float(jnp.min(mean_curve[1:]))


class TestSummaries:
    def test_summarize_scr_curve_shapes(self):
        binned = jnp.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
        summary = summarize_scr_curve(binned)
        assert summary["bin_mean"] == [2.0, 2.0, 2.0]
        assert len(summary["bin_std"]) == 3
        assert len(summary["bin_stderr"]) == 3

    def test_windows_describe_rise_and_flat_without_gate(self):
        rising = [5.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0]
        flat = [5.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        result = describe_scr_curve_windows(
            {"sgd": rising, "cbp": flat, "upgd": flat},
            early_window=(2, 7),
            late_bins=5,
        )
        assert result["measures"]["sgd"]["late_over_early"] > 1.0
        assert result["measures"]["cbp"]["late_over_early"] == 1.0
        assert result["measures"]["upgd"]["late_over_early"] == 1.0
        assert result["interpretation"] == "descriptive_only_no_threshold_or_claim"
        assert "checks" not in result
        assert "all_pass" not in result

    def test_windows_do_not_turn_a_flat_curve_into_a_pass_or_failure(self):
        flat = [5.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        result = describe_scr_curve_windows({"sgd": flat}, early_window=(2, 7))
        assert result["measures"]["sgd"]["late_over_early"] == 1.0
        assert set(result) == {
            "early_window_bins",
            "late_bins",
            "measures",
            "interpretation",
        }

    def test_windows_reject_short_curves(self):
        with pytest.raises(ValueError, match="bins"):
            describe_scr_curve_windows({"sgd": [1.0, 2.0]}, early_window=(2, 7))
