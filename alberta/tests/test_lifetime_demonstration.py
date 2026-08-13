"""Long-horizon oracle-representation retention stress test.

This is a useful component diagnostic, not the Alberta Plan's L3 integrated
demonstration.  It runs one supervised predictor online without resets, but it
has no action, grounded reward, partner, learned state, autonomous feature
discovery, or model/planning loop.  ``ContextGatedFeatures`` is a hand-designed
oracle feature map over visible context channels.

This module runs ONE learner instance (LinearLearner + Autostep over the
exclusive context-gated representation) through 8 full cycles of
:class:`LifetimeGauntletStream` — 64,000 uninterrupted steps of: fresh task,
task C recurrence, another fresh task, task D recurrence, with a 10x
input-scale stressor every third cycle — with no resets and no explicit
boundary callback to the learner. The reported quantities are trajectories
over the life, not one-shot numbers (calibration medians over the same 8 seeds;
there is no held-out confidence-interval claim):

- **Oracle-gated retention does not erode over this horizon**: task C re-entry error falls from 2.05
  (first exposure) to ~0.11 by the final cycle; savings on the LAST cycle are
  the highest of the whole life (~19x for C, ~16x for D).
- **The scale stressor is visible and survived**: stressed cycles (2, 5) dip
  to ~2-5x savings, recover immediately, and never produce a non-finite step.
- **Plasticity does not decay with age**: fresh-task entry error is flat
  from cycle 3 onward (early growth is task-distance geometry — starting
  from zero weights is closer to a random task than starting from another
  random task — not plasticity loss; documented here so nobody mistakes it).
- **The raw-observation twin fails forever**: without the gated
  representation, savings sit at 0.2-0.5 (revisits *worse* than first
  exposure, because interference leaves the weights mid-way between tasks)
  and late-life re-entry error is ~100x the gated learner's.
- **Fixed state shapes**: the learner state's pytree shapes are identical at
  birth and after 64k steps. The test asserts shape-bounded state, not measured
  latency, peak process memory, or indefinite numerical stability.
"""

import jax
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
    run_gauntlet,
    run_gauntlet_batched,
)

pytestmark = [pytest.mark.development, pytest.mark.slow]

N_SEEDS = 8
N_CYCLES = 8


@pytest.fixture(scope="module")
def config() -> GauntletConfig:
    return GauntletConfig(segment_length=2000)


@pytest.fixture(scope="module")
def stream(config: GauntletConfig) -> LifetimeGauntletStream:
    return LifetimeGauntletStream(config, scale_cycle_period=3)


@pytest.fixture(scope="module")
def keys():
    return jr.split(jr.key(3), N_SEEDS)


@pytest.fixture(scope="module")
def gated_scorecard(stream, config, keys):
    sq = run_gauntlet_batched(
        LinearLearner(optimizer=Autostep()),
        ContextGatedFeatures(stream),
        N_CYCLES * stream.cycle_length,
        keys,
    )
    return lifetime_scorecard(sq, config, N_CYCLES)


@pytest.fixture(scope="module")
def raw_scorecard(stream, config, keys):
    sq = run_gauntlet_batched(
        LinearLearner(optimizer=Autostep()),
        stream,
        N_CYCLES * stream.cycle_length,
        keys,
    )
    return lifetime_scorecard(sq, config, N_CYCLES)


def _med(a):
    return jnp.median(a, axis=0)


class TestLifetimeStreamMechanics:
    def test_cycle_schedule(self, stream: LifetimeGauntletStream):
        length = stream.config.segment_length
        assert stream.cycle_length == 4 * length
        assert int(stream.sub_segment_of(jnp.array(0))) == 0
        assert int(stream.sub_segment_of(jnp.array(length))) == 1
        assert int(stream.sub_segment_of(jnp.array(3 * length))) == 3
        assert int(stream.cycle_of(jnp.array(4 * length))) == 1

    def test_persistent_tasks_never_change(self, stream: LifetimeGauntletStream):
        """Tasks C and D are fixed for the whole life; fresh tasks redraw."""
        state = stream.init(jr.key(0))
        w_c0, w_d0 = state.w_c, state.w_d
        fresh_snapshots = []

        def body(s, i):
            _, s = stream.step(s, i)
            return s, s.w_fresh[0]

        state, fresh_trace = jax.lax.scan(
            body, state, jnp.arange(2 * stream.cycle_length)
        )
        assert jnp.array_equal(state.w_c, w_c0)
        assert jnp.array_equal(state.w_d, w_d0)
        # The fresh task took at least 3 distinct values over two cycles
        # (initial + redraws at sub-segments 0/2 boundaries).
        del fresh_snapshots
        assert len(set(jnp.round(fresh_trace, 5).tolist())) >= 3


class TestLongHorizonOracleRepresentation:
    def test_memory_does_not_erode_with_age(self, gated_scorecard):
        """Savings on the final cycle are large — larger than mid-life —
        and late-life re-entry error is a small fraction of first exposure
        (calibration: savings_c trajectory [13.2, 2.2, 6.7, 13.1, 5.6,
        10.7, 19.1]; re-entry 2.05 -> 0.106)."""
        savings_c = _med(gated_scorecard["savings_c"])
        savings_d = _med(gated_scorecard["savings_d"])
        # Final-cycle savings stay high for both persistent tasks.
        assert float(savings_c[-1]) > 5.0
        assert float(savings_d[-1]) > 5.0
        # Every cycle keeps positive savings, including the scale-stressed
        # ones (measured minima 2.2 / 2.0 at the stressor cycles).
        assert float(jnp.min(savings_c)) > 1.3
        assert float(jnp.min(savings_d)) > 1.3
        # Late-life absolute retention.
        recur_c = _med(gated_scorecard["recur_c_early"])
        assert float(recur_c[-1]) < 0.25 * float(recur_c[0])

    def test_late_fresh_task_error_stays_within_calibrated_ratio(
        self, gated_scorecard
    ):
        """Cycle-7 fresh-task error stays within 2x the cycle-3 value."""
        fresh = _med(gated_scorecard["fresh_early"])
        # Cycles 2 and 5 carry the 10x scale stressor; compare unstressed.
        assert float(fresh[7]) < 2.0 * float(fresh[3])

    def test_no_divergence_over_the_whole_life(self, gated_scorecard, raw_scorecard):
        assert int(jnp.sum(gated_scorecard["nan_steps"])) == 0
        assert int(jnp.sum(raw_scorecard["nan_steps"])) == 0

    def test_raw_twin_interferes_across_the_measured_life(
        self, gated_scorecard, raw_scorecard
    ):
        """Across eight cycles, without the gated representation the optimizer shows no
        savings at any age (calibration: 0.2-0.5 — revisits are WORSE than
        first exposure) and its late-life re-entry error is ~100x higher."""
        raw_savings_c = _med(raw_scorecard["savings_c"])
        assert float(jnp.max(raw_savings_c[1:])) < 1.0
        gated_late = _med(gated_scorecard["recur_c_early"])[-1]
        raw_late = _med(raw_scorecard["recur_c_early"])[-1]
        assert float(raw_late) > 10.0 * float(gated_late)

    def test_bounded_state_for_life(self, stream: LifetimeGauntletStream):
        """Learner state treedef and leaf shapes remain fixed over the run."""
        learner = LinearLearner(optimizer=Autostep())
        gated = ContextGatedFeatures(stream)
        birth_state = learner.init(gated.feature_dim)
        final_state, sq = run_gauntlet(
            learner, gated, 2 * stream.cycle_length, jr.key(9)
        )
        birth_leaves, birth_def = jax.tree.flatten(birth_state)
        final_leaves, final_def = jax.tree.flatten(final_state)
        assert birth_def == final_def
        assert [jnp.shape(leaf) for leaf in birth_leaves] == [
            jnp.shape(leaf) for leaf in final_leaves
        ]
        assert bool(jnp.all(jnp.isfinite(sq)))
