"""One-million-step longevity extension of the lifetime diagnostic.

``test_lifetime_demonstration.py`` certifies the oracle-gated retention
mechanism over 8 cycles (64k steps).  This module extends the identical
protocol to **125 cycles — 1,000,000 uninterrupted steps — per seed** (8
seeds, one ``lax.scan`` per seed, a few seconds of CPU wall-clock), making the
long-horizon claim reproducible in-tree rather than a scratchpad anecdote.

Same scope caveat as the 64k version: this is a *mechanism* diagnostic
(supervised targets, hand-designed oracle context gating), not an integrated
agent demonstration.  What it adds is the horizon: fifteen times the CI
protocol, ~375 fresh-task interference events, ~41 input-scale shocks, and
124 recurrences of each persistent task, all inside one learner life.

Calibration (medians over 8 seeds, jr.key(11), measured 2026-07-30):

- savings_c at cycles [1, 10, 30, 60, 90, 124]:
  [10.2, 17.3, 10.2, 10.3, 11.5, 18.4] — no erosion trend over 124
  recurrences; the final cycle's savings are among the highest of the life.
- re-entry error (recur_c_early) at cycles [0, 10, 30, 60, 90, 124]:
  [2.00, 0.12, 0.17, 0.20, 0.18, 0.10]; mean over the last 20 cycles 0.19
  vs first exposure 2.00.
- fresh-task adaptation (unstressed cycles): linear-fit slope
  +0.0002/cycle — flat; plasticity does not decay with age.
- zero non-finite steps across all 8 million learner updates.
"""

import jax.numpy as jnp
import jax.random as jr
import pytest

from alberta_framework.core.learners import LinearLearner
from alberta_framework.core.optimizers import Autostep
from alberta_framework.streams.gauntlet import (
    ContextGatedFeatures,
    GauntletConfig,
    LifetimeGauntletStream,
    lifetime_scorecard,
    run_gauntlet_batched,
)

pytestmark = pytest.mark.slow

N_SEEDS = 8
N_CYCLES = 125


@pytest.fixture(scope="module")
def config() -> GauntletConfig:
    return GauntletConfig(segment_length=2000)


@pytest.fixture(scope="module")
def scorecard(config: GauntletConfig):
    stream = LifetimeGauntletStream(config, scale_cycle_period=3)
    keys = jr.split(jr.key(11), N_SEEDS)
    sq = run_gauntlet_batched(
        LinearLearner(optimizer=Autostep()),
        ContextGatedFeatures(stream),
        N_CYCLES * stream.cycle_length,
        keys,
    )
    return lifetime_scorecard(sq, config, N_CYCLES)


def _med(a):
    return jnp.median(a, axis=0)


class TestMillionStepLongevity:
    def test_no_divergence_over_a_million_steps(self, scorecard):
        """Zero non-finite squared errors across 8 seeds x 1M steps."""
        assert int(jnp.sum(scorecard["nan_steps"])) == 0

    def test_memory_does_not_erode_across_124_recurrences(self, scorecard):
        """Late-life savings are as high as early-life savings.

        Calibration: savings_c median over cycles 100-124 is ~11-18x; over
        cycles 1-25 ~10-17x.  Assert the late-life median stays above 5x and
        above half the early-life median (no erosion trend).
        """
        savings_c = _med(scorecard["savings_c"])  # (124,)
        early = float(jnp.median(savings_c[:25]))
        late = float(jnp.median(savings_c[-25:]))
        assert late > 5.0
        assert late > 0.5 * early

    def test_late_life_reentry_stays_near_solution(self, scorecard):
        """Mean re-entry error over the last 20 cycles is a small fraction
        of the first exposure (calibration: 0.19 vs 2.00)."""
        recur_c = _med(scorecard["recur_c_early"])  # (125,)
        late_mean = float(jnp.mean(recur_c[-20:]))
        assert late_mean < 0.25 * float(recur_c[0])

    def test_plasticity_flat_over_the_life(self, scorecard, config):
        """Unstressed fresh-task entry error does not trend upward: the mean
        over the last 30 unstressed cycles stays within 2x of the mean over
        unstressed cycles 3-33 (calibration: 11.2 vs early-window means in
        the same range once warmup geometry passes)."""
        fresh = _med(scorecard["fresh_early"])  # (125,)
        # Cycles with index % 3 == 2 carry the 10x scale stressor.
        unstressed = jnp.array(
            [i for i in range(3, N_CYCLES) if i % 3 != 2], dtype=jnp.int32
        )
        vals = fresh[unstressed]
        early_mean = float(jnp.mean(vals[:20]))
        late_mean = float(jnp.mean(vals[-20:]))
        assert late_mean < 2.0 * early_mean
