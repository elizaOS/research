"""Validation of the publication-statistics machinery in ``utils/statistics.py``.

Every certification claim in the framework leans on these functions (confidence
intervals, paired significance tests, multiple-comparison corrections, effect
sizes), so this file validates them empirically against known distributions and
hand-computed fixtures rather than trusting the implementation.

Calibration (measured on this machine, scripts in the session scratchpad):

- t-based 95% CI empirical coverage over 2000 replications of n=10 draws from
  N(3, 2): 0.9445 / 0.9570 / 0.9480 / 0.9500 across four data seeds
  (binomial std at 2000 reps is ~0.0049). Assertion band [0.93, 0.97] is
  ~3x the observed spread on each side.
- Percentile-bootstrap 95% CI coverage (n=25 samples, 250 replications,
  400 bootstrap resamples): 0.920 / 0.896 / 0.924 across three data seeds —
  the percentile method under-covers at small n, which is expected.
  Assertion band [0.85, 0.99] leaves >2 binomial sigma below the worst
  observed value.
- Paired t-test under the null (400 replications, 20 pairs, alpha=0.05):
  rejection rate 0.0575; Wilcoxon: 0.0500. Assertion <= 0.10 (= 2*alpha,
  ~4.5 binomial sigma above the mean under exact calibration).
- Power under a strong paired shift (+1.0 with sd-0.5 noise, 20 pairs):
  1.000 for both tests over 400 replications. Assertion >= 0.95.
- Holm-vs-Bonferroni superset property held on 2000/2000 random p-value
  draws, with Holm strictly larger on 295 of them. Assertion requires the
  superset always and strictness on >= 50 draws.
"""

import numpy as np
import pytest

from alberta_framework.utils.experiments import AggregatedResults
from alberta_framework.utils.statistics import (
    SignificanceResult,
    bonferroni_correction,
    bootstrap_ci,
    cohens_d,
    compute_statistics,
    compute_timeseries_statistics,
    holm_correction,
    mann_whitney_comparison,
    pairwise_comparisons,
    ttest_comparison,
    wilcoxon_comparison,
)

# ---------------------------------------------------------------------------
# 1. Confidence intervals: hand-computed fixtures + empirical coverage
# ---------------------------------------------------------------------------


class TestComputeStatistics:
    def test_hand_computed_fixture(self) -> None:
        """All summary fields match hand-derived values for [1, 2, 3, 4, 5]."""
        s = compute_statistics([1.0, 2.0, 3.0, 4.0, 5.0], confidence_level=0.95)
        assert s.mean == pytest.approx(3.0)
        assert s.std == pytest.approx(np.sqrt(2.5))  # ddof=1
        assert s.sem == pytest.approx(np.sqrt(2.5) / np.sqrt(5))
        assert s.median == pytest.approx(3.0)
        assert s.iqr == pytest.approx(2.0)  # 75th=4, 25th=2
        assert s.n_seeds == 5
        # t_{0.975, df=4} = 2.7764; margin = 2.7764 * 0.70711 = 1.9633
        assert s.ci_lower == pytest.approx(3.0 - 2.7764 * np.sqrt(0.5), abs=1e-3)
        assert s.ci_upper == pytest.approx(3.0 + 2.7764 * np.sqrt(0.5), abs=1e-3)
        # CI must bracket the mean symmetrically
        assert s.ci_lower < s.mean < s.ci_upper

    def test_single_value_degenerate(self) -> None:
        s = compute_statistics([4.2])
        assert s.mean == pytest.approx(4.2)
        assert s.std == 0.0
        assert s.sem == 0.0
        assert s.ci_lower == pytest.approx(4.2)
        assert s.ci_upper == pytest.approx(4.2)
        assert s.n_seeds == 1

    def test_empirical_ci_coverage(self) -> None:
        """95% t-CI covers the true mean ~95% of the time.

        2000 replications of n=10 Gaussian draws. Measured coverage across
        seeds: 0.9445-0.9570 (see module docstring); band [0.93, 0.97] is a
        multi-sigma margin around nominal 0.95.
        """
        rng = np.random.default_rng(0)
        true_mean, true_sd = 3.0, 2.0
        n_reps, n = 2000, 10
        data = rng.normal(true_mean, true_sd, size=(n_reps, n))
        covered = 0
        for i in range(n_reps):
            s = compute_statistics(data[i], confidence_level=0.95)
            covered += int(s.ci_lower <= true_mean <= s.ci_upper)
        coverage = covered / n_reps
        assert 0.93 <= coverage <= 0.97, f"coverage {coverage} outside [0.93, 0.97]"

    def test_wider_confidence_level_gives_wider_interval(self) -> None:
        values = np.random.default_rng(1).normal(0.0, 1.0, size=20)
        s95 = compute_statistics(values, confidence_level=0.95)
        s99 = compute_statistics(values, confidence_level=0.99)
        assert (s99.ci_upper - s99.ci_lower) > (s95.ci_upper - s95.ci_lower)


class TestTimeseriesStatistics:
    def test_matches_per_column_compute_statistics(self) -> None:
        """Vectorised timeseries CI agrees with per-step scalar CI."""
        rng = np.random.default_rng(2)
        arr = rng.normal(1.0, 0.5, size=(8, 6))  # (n_seeds, n_steps)
        mean, lo, hi = compute_timeseries_statistics(arr, confidence_level=0.95)
        assert mean.shape == lo.shape == hi.shape == (6,)
        for step in range(6):
            s = compute_statistics(arr[:, step], confidence_level=0.95)
            assert mean[step] == pytest.approx(s.mean)
            assert lo[step] == pytest.approx(s.ci_lower)
            assert hi[step] == pytest.approx(s.ci_upper)


class TestBootstrapCI:
    def test_deterministic_and_brackets_estimate(self) -> None:
        values = np.random.default_rng(3).normal(5.0, 1.0, size=30)
        r1 = bootstrap_ci(values, statistic="mean", n_bootstrap=500, seed=42)
        r2 = bootstrap_ci(values, statistic="mean", n_bootstrap=500, seed=42)
        assert r1 == r2  # same seed, same result
        point, lo, hi = r1
        assert point == pytest.approx(float(np.mean(values)))
        assert lo < point < hi

    def test_median_statistic(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0, 100.0]
        point, lo, hi = bootstrap_ci(values, statistic="median", n_bootstrap=300, seed=0)
        assert point == pytest.approx(3.0)
        assert lo <= point <= hi

    def test_empirical_coverage(self) -> None:
        """Percentile-bootstrap 95% CI coverage stays near nominal.

        250 replications, n=25 samples, 400 resamples. Measured coverage
        0.896-0.924 across seeds (the percentile method under-covers at
        small n); assert the wide calibrated band [0.85, 0.99].
        """
        rng = np.random.default_rng(0)
        true_mean = 3.0
        n_reps, n, n_boot = 250, 25, 400
        covered = 0
        for i in range(n_reps):
            sample = rng.normal(true_mean, 2.0, size=n)
            _, lo, hi = bootstrap_ci(sample, statistic="mean", n_bootstrap=n_boot, seed=i)
            covered += int(lo <= true_mean <= hi)
        coverage = covered / n_reps
        assert 0.85 <= coverage <= 0.99, f"coverage {coverage} outside [0.85, 0.99]"


# ---------------------------------------------------------------------------
# 2. Paired significance tests: null calibration + power
# ---------------------------------------------------------------------------


def _paired_replication_rates(
    test_fn,
    n_reps: int = 400,
    n_pairs: int = 20,
) -> tuple[float, float]:
    """Return (null rejection rate, power under a +1.0 paired shift)."""
    null_rej = alt_rej = 0
    for i in range(n_reps):
        r = np.random.default_rng(10_000 + i)
        base = r.normal(0.0, 1.0, size=n_pairs)
        a_null = base + r.normal(0.0, 0.5, size=n_pairs)
        b = base + r.normal(0.0, 0.5, size=n_pairs)
        null_rej += int(test_fn(a_null, b).significant)
        a_alt = b + 1.0 + r.normal(0.0, 0.5, size=n_pairs)
        alt_rej += int(test_fn(a_alt, b).significant)
    return null_rej / n_reps, alt_rej / n_reps


class TestPairedTests:
    def test_paired_ttest_null_and_power(self) -> None:
        """Null rejection ~alpha (measured 0.0575), power ~1.0 (measured 1.000)."""
        null_rate, power = _paired_replication_rates(
            lambda a, b: ttest_comparison(a, b, paired=True, alpha=0.05)
        )
        assert null_rate <= 0.10, f"null rejection rate {null_rate} > 2*alpha"
        assert power >= 0.95, f"power {power} < 0.95 under a strong true shift"

    def test_wilcoxon_null_and_power(self) -> None:
        """Null rejection ~alpha (measured 0.0500), power ~1.0 (measured 1.000)."""
        null_rate, power = _paired_replication_rates(
            lambda a, b: wilcoxon_comparison(a, b, alpha=0.05)
        )
        assert null_rate <= 0.10, f"null rejection rate {null_rate} > 2*alpha"
        assert power >= 0.95, f"power {power} < 0.95 under a strong true shift"

    def test_ttest_result_fields(self) -> None:
        res = ttest_comparison(
            [1.0, 2.0, 3.2], [1.3, 2.4, 3.5], paired=True, method_a="x", method_b="y"
        )
        assert isinstance(res, SignificanceResult)
        assert res.test_name == "paired t-test"
        assert res.method_a == "x" and res.method_b == "y"
        assert 0.0 <= res.p_value <= 1.0
        # every a_i < b_i: effect size sign must reflect a < b
        assert res.effect_size < 0.0

    def test_unpaired_ttest_separated_groups(self) -> None:
        rng = np.random.default_rng(5)
        a = rng.normal(10.0, 0.5, size=15)
        b = rng.normal(0.0, 0.5, size=15)
        res = ttest_comparison(a, b, paired=False, alpha=0.01)
        assert res.test_name == "independent t-test"
        assert res.significant
        assert res.effect_size > 5.0  # enormous separation

    def test_mann_whitney_separated_groups(self) -> None:
        rng = np.random.default_rng(6)
        a = rng.normal(10.0, 0.5, size=15)
        b = rng.normal(0.0, 0.5, size=15)
        res = mann_whitney_comparison(a, b, alpha=0.01)
        assert res.test_name == "Mann-Whitney U"
        assert res.significant


# ---------------------------------------------------------------------------
# 3. Multiple-comparison corrections
# ---------------------------------------------------------------------------


class TestCorrections:
    def test_bonferroni_hand_computed(self) -> None:
        significant, corrected_alpha = bonferroni_correction([0.01, 0.02, 0.04], alpha=0.05)
        assert corrected_alpha == pytest.approx(0.05 / 3)
        assert significant == [True, False, False]

    def test_holm_hand_computed_step_down(self) -> None:
        # sorted p: [0.01, 0.02, 0.03, 0.04]; thresholds [1/80, 1/60, 1/40, 1/20]
        # 0.01 < 0.0125 -> reject; 0.02 > 0.0167 -> stop: only p=0.01 rejected.
        assert holm_correction([0.03, 0.01, 0.04, 0.02], alpha=0.05) == [
            False,
            True,
            False,
            False,
        ]

    def test_holm_strictly_more_powerful_fixture(self) -> None:
        # Bonferroni (alpha/3 = 0.0167) rejects only the first two;
        # Holm thresholds [0.0167, 0.025, 0.05] reject all three.
        p = [0.01, 0.015, 0.04]
        bonf, _ = bonferroni_correction(p, alpha=0.05)
        holm = holm_correction(p, alpha=0.05)
        assert bonf == [True, True, False]
        assert holm == [True, True, True]

    def test_holm_rejects_superset_of_bonferroni_property(self) -> None:
        """Property: Holm rejections are always a superset of Bonferroni's.

        2000 random p-vectors (uniform, small-skewed beta, and mixed).
        Calibration: superset held on 2000/2000 draws and Holm was strictly
        larger on 295 draws; assert strictness on >= 50 (>10 sigma margin).
        """
        n_draws = 2000
        strictly_more = 0
        for i in range(n_draws):
            r = np.random.default_rng(50_000 + i)
            m = int(r.integers(2, 12))
            kind = i % 3
            if kind == 0:
                p = r.uniform(0, 1, size=m)
            elif kind == 1:
                p = r.beta(0.3, 4.0, size=m)
            else:
                p = np.concatenate(
                    [r.beta(0.2, 8.0, size=m // 2 + 1), r.uniform(0, 1, size=m // 2)]
                )[:m]
            p_list = [float(v) for v in p]
            bonf, _ = bonferroni_correction(p_list, alpha=0.05)
            holm = holm_correction(p_list, alpha=0.05)
            assert len(holm) == len(bonf) == m
            for b_sig, h_sig in zip(bonf, holm, strict=True):
                assert (not b_sig) or h_sig, (
                    f"Bonferroni rejected but Holm did not on p={p_list}"
                )
            strictly_more += int(sum(holm) > sum(bonf))
        assert strictly_more >= 50, f"Holm strictly larger on only {strictly_more}/2000 draws"

    def test_corrections_all_significant_and_none_significant(self) -> None:
        tiny = [1e-6, 1e-7, 1e-8]
        assert holm_correction(tiny, alpha=0.05) == [True, True, True]
        assert bonferroni_correction(tiny, alpha=0.05)[0] == [True, True, True]
        huge = [0.5, 0.9, 0.7]
        assert holm_correction(huge, alpha=0.05) == [False, False, False]
        assert bonferroni_correction(huge, alpha=0.05)[0] == [False, False, False]


# ---------------------------------------------------------------------------
# 4. Effect sizes: hand-computed fixtures
# ---------------------------------------------------------------------------


class TestEffectSizes:
    def test_cohens_d_hand_computed(self) -> None:
        # a=[2,4,6]: mean 4, var 4; b=[1,3,5]: mean 3, var 4.
        # pooled sd = sqrt((2*4 + 2*4)/4) = 2 -> d = (4-3)/2 = 0.5
        assert cohens_d([2.0, 4.0, 6.0], [1.0, 3.0, 5.0]) == pytest.approx(0.5)

    def test_cohens_d_antisymmetric(self) -> None:
        a, b = [2.0, 4.0, 6.0], [1.0, 3.0, 5.0]
        assert cohens_d(a, b) == pytest.approx(-cohens_d(b, a))

    def test_cohens_d_zero_variance_returns_zero(self) -> None:
        assert cohens_d([3.0, 3.0, 3.0], [3.0, 3.0, 3.0]) == 0.0

    def test_cohens_d_positive_means_a_greater(self) -> None:
        rng = np.random.default_rng(7)
        a = rng.normal(2.0, 1.0, size=50)
        b = rng.normal(0.0, 1.0, size=50)
        assert cohens_d(a, b) > 1.0

    def test_mann_whitney_rank_biserial_direction(self) -> None:
        """Rank-biserial sign must match the module convention (positive => a > b).

        With a completely dominating b, every one of the n_a*n_b pairs favors
        a, so the rank-biserial correlation is exactly +1; fully reversed
        groups give exactly -1 (Kerby 2014: r = 2*U1/(n_a*n_b) - 1).
        """
        a_dom = [10.0, 11.0, 12.0, 13.0]
        b_low = [1.0, 2.0, 3.0, 4.0]
        res_a_wins = mann_whitney_comparison(a_dom, b_low)
        assert res_a_wins.effect_size == pytest.approx(1.0)
        res_b_wins = mann_whitney_comparison(b_low, a_dom)
        assert res_b_wins.effect_size == pytest.approx(-1.0)

    def test_mann_whitney_rank_biserial_partial_overlap(self) -> None:
        # a=[3,5], b=[1,4]: favorable pairs (3>1, 5>1, 5>4) = 3 of 4
        # => U1 = 3, r = 2*3/4 - 1 = 0.5
        res = mann_whitney_comparison([3.0, 5.0], [1.0, 4.0])
        assert res.statistic == pytest.approx(3.0)
        assert res.effect_size == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# 5. pairwise_comparisons end-to-end on synthetic AggregatedResults
# ---------------------------------------------------------------------------


def _make_aggregated(name: str, level: float, seed: int, n_seeds: int = 12) -> AggregatedResults:
    """AggregatedResults whose metric hovers at `level` with small seed noise."""
    rng = np.random.default_rng(seed)
    n_steps = 30
    per_seed_offset = rng.normal(0.0, 0.02 * max(level, 0.1), size=(n_seeds, 1))
    arr = level + per_seed_offset + rng.normal(0.0, 0.01, size=(n_seeds, n_steps))
    return AggregatedResults(
        config_name=name,
        seeds=list(range(n_seeds)),
        metric_arrays={"squared_error": arr},
        summary={},
    )


class TestPairwiseComparisons:
    def _results(self) -> dict[str, AggregatedResults]:
        return {
            "good": _make_aggregated("good", 0.1, seed=1),
            "mid": _make_aggregated("mid", 0.5, seed=2),
            "bad": _make_aggregated("bad", 2.0, seed=3),
        }

    def test_all_pairs_present_and_significant(self) -> None:
        comps = pairwise_comparisons(
            self._results(), metric="squared_error", test="ttest", correction="holm", window=10
        )
        assert set(comps) == {("good", "mid"), ("good", "bad"), ("mid", "bad")}
        for (name_a, name_b), res in comps.items():
            assert res.significant, f"{name_a} vs {name_b} should separate cleanly"
            assert res.method_a == name_a and res.method_b == name_b
            assert "(holm)" in res.test_name
            # lower squared error listed first in every pair => negative d
            assert res.effect_size < 0.0

    def test_correction_matches_manual_holm(self) -> None:
        comps = pairwise_comparisons(
            self._results(), test="ttest", correction="holm", window=10
        )
        p_values = [r.p_value for r in comps.values()]
        expected = holm_correction(p_values, alpha=0.05)
        assert [r.significant for r in comps.values()] == expected

    def test_bonferroni_correction_path(self) -> None:
        comps = pairwise_comparisons(
            self._results(), test="wilcoxon", correction="bonferroni", window=10
        )
        p_values = [r.p_value for r in comps.values()]
        expected, _ = bonferroni_correction(p_values, alpha=0.05)
        assert [r.significant for r in comps.values()] == expected
        assert all("(bonferroni)" in r.test_name for r in comps.values())

    def test_fewer_than_two_methods_returns_empty(self) -> None:
        assert pairwise_comparisons({"only": _make_aggregated("only", 0.1, seed=4)}) == {}

    def test_unknown_test_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown test"):
            pairwise_comparisons(self._results(), test="anova")

    def test_unknown_correction_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown correction"):
            pairwise_comparisons(self._results(), correction="fdr")

    def test_non_aggregated_results_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            pairwise_comparisons({"a": object(), "b": object()})  # type: ignore[dict-item]
