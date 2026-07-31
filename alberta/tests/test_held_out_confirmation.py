"""Held-out confirmation sweeps: wave-3 results re-run on never-inspected seeds.

Every protocol below was frozen and calibrated on development seed batches that
were inspected repeatedly while thresholds were chosen.  This module promotes
those results from calibration-seed evidence to held-out evidence by the strict
rule: run the EXACT frozen protocol (same machinery imported from the source
test module wherever importable, same hyperparameters, same metrics) on seed
batches that were never inspected during any calibration, asserting floors that
were committed BEFORE the held-out batches were ever run.  Nothing here was
tuned on the held-out seeds; the measured values below are the single
confirmation run (2026-07-30, CPU), recorded after the floors were fixed.

Held-out batches (pre-committed)
================================

1. **Historical scale-robust discovery closure**
   (``test_gauntlet_discovery.py``, ``TestScaleRobustClosure``): this file
   used then-uninspected seeds 30..45 (16 seeds, ``jr.key(seed)``, learner
   init ``jr.key(123)``) on the full 9-segment gauntlet program.  That
   primary-only exposure now excludes the entire numeric 30..59 namespace
   from the separate v2 three-arm evidence protocol.
2. **Discovery-driven control** (``test_discovery_control_life.py``): direct
   keys ``jr.key(1000+i)``, i in 0..7.  Its calibration batches were
   ``jr.split(jr.key(b), 8)`` for b in {0, 100, 7, 42}.
3. **L3 integrated life, gated rung + paired ablation**
   (``test_integrated_life.py``): seeds 5000..5007 (its calibration used
   seeds 0..7).
4. **Lifetime longevity** (``test_lifetime_longevity.py``): base key
   ``jr.key(77)`` split 8 ways (its calibration used ``jr.key(11)``).

Floors and measured held-out values
===================================

Protocol 1 — all TestScaleRobustClosure frozen floors, with the two
pre-declared WEAKER acceptance floors from that module's documentation:
final unique ctx pairs >= 7 per seed (dev-seed measurement was median 16) and
``savings_c_final`` median >= 3 (the dev-era frozen floor was 5; both are
documented, the weaker one is asserted).  Measured on seeds 30..45:

    nan_steps 0; max scaled-tail MSE 25.70 (ceil 50); max final-tail 0.0520
    (ceil 0.1); medians: first-C early/tail 8.59/0.651 (<=15/<=2), recurrent-C
    early/tail 0.661/0.358 (<=2), scaled tail 4.50 (<=10), final-C early/tail
    1.452/0.0399 (<=3/<=0.1), nonlinear 0.0347 (<=0.1), savings_c 11.41
    (>=8), savings_d 11.54 (>=5), savings_c_final 5.80 (>=3 asserted; also
    clears the stricter dev floor 5); final ctx pairs per seed min 15
    (>=7 asserted).

Protocol 2 — frozen floors of ``TestRecurrenceRetention`` /
``TestDiscoveredRepresentation``.  Measured on keys 1000..1007:

    paired recurrence diff (coupled - raw) mean +0.1648 (>=0.06), min +0.1250
    (>=0.03); coupled recurrence mean 0.9319 (>=0.85); distinct oracle
    products in the final bank min 3 (>=2), median 4.0 (>=3); all finite.

Protocol 3 — frozen floors of ``TestCertifiedMemoryLife`` (its exact
constants are imported).  Measured on seeds 5000..5007:

    life min 0.9674 (>=0.85), life mean 0.9694 (>= random+0.35 = 0.85), late
    min 0.9709 (>= optimum-0.08 = 0.92); settled early-window per-seed min
    0.8800 (>=0.60); paired gated-ablation settled gap min +0.5657 / mean
    +0.5906 (>=0.30); zero non-finite rewards or weights.

Protocol 4 — frozen floors of ``TestMillionStepLongevity``.  Measured on
``jr.key(77)``:

    nan_steps 0; late savings 12.46 (>5) vs early 9.55 (ratio 1.31 > 0.5);
    late re-entry 0.196 vs first exposure 2.108 (ratio 0.093 < 0.25); fresh
    late/early 11.92/11.44 = 1.04 (< 2.0).

Every protocol PASSED its pre-committed floors on the first held-out run — the
wave-3 results survive on seeds that never informed those pre-committed
wave-3 floors.  Protocol 1 is historical primary-only evidence and is not an
untouched v2 result.  Scope notes inherited from the source modules still
apply (supplied context cues, closed pair spaces, hand-built gated features
for protocols 3-4).

Runtime: ~80-100s on CPU (dominated by the 8 x 1M-step longevity scan and the
16 x 27k-step discovery program; module-scoped fixtures share all rollouts).
"""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

# The frozen protocol machinery, imported from the source test modules so the
# held-out runs cannot drift from what was calibrated (pytest puts tests/ on
# sys.path; these are the exact functions the source suites execute).
from test_discovery_control_life import (
    recurrence_early,
    run_coupled_batch,
    run_fixed_map_batch,
)
from test_gauntlet_discovery import (
    LEARNER_INIT_KEY,
    make_discovery_learner,
    run_discovery_traced,
)
from test_integrated_life import (
    LIFE_FLOOR,
    LIFE_VS_RANDOM,
    MEM_ABLATION_GAP,
    MEM_SETTLED_MIN,
    NEAR_OPT_SLACK,
    OPTIMAL,
    RANDOM,
    _gated_features,
    _metric,
    _plain_features,
    _run_sarsa_life,
)

from alberta_framework.core.learners import LinearLearner
from alberta_framework.core.optimizers import Autostep
from alberta_framework.streams.gauntlet import (
    ContextGatedFeatures,
    GauntletConfig,
    GauntletStream,
    LifetimeGauntletStream,
    gauntlet_scorecard,
    lifetime_scorecard,
    run_gauntlet_batched,
)

pytestmark = [pytest.mark.slow, pytest.mark.scientific]

# ---------------------------------------------------------------------------
# Pre-committed held-out batches (see module docstring; do not retune)
# ---------------------------------------------------------------------------

# Historical primary-only confirmation; these keys are excluded from v2.
P1_HELD_OUT_SEEDS = tuple(range(30, 46))
P1_CTX_PAIR_FLOOR = 7  # documented acceptance floor (dev measurement: 15-16)
P1_SAVINGS_C_FINAL_FLOOR = 3.0  # weaker acceptance floor (dev-era floor: 5.0)

P2_HELD_OUT_KEYS = tuple(1000 + i for i in range(8))

P3_HELD_OUT_SEEDS = tuple(range(5000, 5008))

P4_HELD_OUT_KEY = 77
P4_N_SEEDS = 8
P4_N_CYCLES = 125


# ---------------------------------------------------------------------------
# Module-scoped held-out rollouts
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def p1_config() -> GauntletConfig:
    return GauntletConfig()


@pytest.fixture(scope="module")
def p1_run(p1_config: GauntletConfig):
    """Historical primary-only 9-segment program on seeds 30..45."""
    stream = GauntletStream(p1_config)
    keys = jnp.stack([jr.key(seed) for seed in P1_HELD_OUT_SEEDS])
    return run_discovery_traced(
        make_discovery_learner(),
        stream,
        p1_config.num_steps,
        keys,
        jr.key(LEARNER_INIT_KEY),
    )


@pytest.fixture(scope="module")
def p2_runs():
    """Coupled discovery-control arm and raw twin on keys 1000..1007."""
    keys = jnp.stack([jr.key(k) for k in P2_HELD_OUT_KEYS])
    coupled = run_coupled_batch(keys)
    raw = run_fixed_map_batch(keys, "raw")
    return coupled, raw


@pytest.fixture(scope="module")
def p3_lives():
    """Paired gated / no-context 48k-step lives on seeds 5000..5007."""
    gated = [_run_sarsa_life(seed, _gated_features, 4) for seed in P3_HELD_OUT_SEEDS]
    ablation = [_run_sarsa_life(seed, _plain_features, 2) for seed in P3_HELD_OUT_SEEDS]
    return gated, ablation


@pytest.fixture(scope="module")
def p4_scorecard():
    """1M-step longevity protocol on jr.split(jr.key(77), 8)."""
    config = GauntletConfig(segment_length=2000)
    stream = LifetimeGauntletStream(config, scale_cycle_period=3)
    keys = jr.split(jr.key(P4_HELD_OUT_KEY), P4_N_SEEDS)
    sq = run_gauntlet_batched(
        LinearLearner(optimizer=Autostep()),
        ContextGatedFeatures(stream),
        P4_N_CYCLES * stream.cycle_length,
        keys,
    )
    return lifetime_scorecard(sq, config, P4_N_CYCLES)


# ---------------------------------------------------------------------------
# Protocol 1: historical primary-only scale-robust confirmation
# ---------------------------------------------------------------------------


class TestHeldOutScaleRobustDiscovery:
    def test_closure_floors_hold_on_held_out_seeds(self, p1_run, p1_config) -> None:
        """The TestScaleRobustClosure gate passes verbatim on seeds 30..45.

        Measured (medians unless stated): nan 0; per-seed max scaled-tail
        25.70 / final-tail 0.0520; first-C 8.59/0.651; recurrent-C
        0.661/0.358; scaled tail 4.50; final-C 1.452/0.0399; nonlinear
        0.0347; savings_c/d 11.41/11.54.
        """
        _, disc_sq, n_ctx, n_ctx_c, n_ctx_d = p1_run
        score = gauntlet_scorecard(disc_sq, p1_config)
        length = p1_config.segment_length

        def window(start: int, stop: int):
            return jnp.mean(disc_sq[:, start:stop], axis=1)

        first_early = window(2 * length, 2 * length + 200)
        first_tail = window(3 * length - 500, 3 * length)
        recur_early = window(4 * length, 4 * length + 200)
        recur_tail = window(5 * length - 500, 5 * length)
        scaled_tail = window(7 * length - 500, 7 * length)
        final_early = window(8 * length, 8 * length + 200)
        final_tail = window(9 * length - 500, 9 * length)

        # Per-seed stability and representation safeguards (frozen values).
        assert int(jnp.sum(score["nan_steps"])) == 0
        assert int(jnp.max(n_ctx_c)) <= p1_config.relevant_dim
        assert int(jnp.max(n_ctx_d)) <= p1_config.relevant_dim
        assert float(jnp.max(scaled_tail)) <= 50.0
        assert float(jnp.max(final_tail)) <= 0.1

        # Aggregate median acquisition, recurrence, and scale safeguards.
        assert float(jnp.median(first_early)) <= 15.0
        assert float(jnp.median(first_tail)) <= 2.0
        assert float(jnp.median(recur_early)) <= 2.0
        assert float(jnp.median(recur_tail)) <= 2.0
        assert float(jnp.median(scaled_tail)) <= 10.0
        assert float(jnp.median(final_early)) <= 3.0
        assert float(jnp.median(final_tail)) <= 0.1
        assert float(jnp.median(score["nonlinear_mse"])) <= 0.1
        assert float(jnp.median(score["savings_c"])) >= 8.0
        assert float(jnp.median(score["savings_d"])) >= 5.0

    def test_weaker_acceptance_floors_hold_and_stricter_dev_values_recorded(
        self, p1_run, p1_config
    ) -> None:
        """The pre-declared acceptance floors hold; dev-level values too.

        Asserted (weaker documented acceptance floors): savings_c_final
        median >= 3 and >= 7 unique final ctx pairs per seed.  Measured:
        savings_c_final median 5.80 — which also clears the stricter dev-era
        floor of 5 — and per-seed final ctx pairs 15-16 (min 15), matching the
        dev-seed measurement rather than merely the floor.
        """
        _, disc_sq, n_ctx, _, _ = p1_run
        score = gauntlet_scorecard(disc_sq, p1_config)
        final_ctx = n_ctx[:, -1]
        assert final_ctx.shape == (len(P1_HELD_OUT_SEEDS),)
        assert int(jnp.min(final_ctx)) >= P1_CTX_PAIR_FLOOR
        assert float(jnp.median(score["savings_c_final"])) >= P1_SAVINGS_C_FINAL_FLOOR


# ---------------------------------------------------------------------------
# Protocol 2: discovery-driven control, held out
# ---------------------------------------------------------------------------


class TestHeldOutDiscoveryControl:
    def test_recurrence_retention_floors_hold(self, p2_runs) -> None:
        """Paired coupled-vs-raw recurrence advantage on keys 1000..1007.

        Frozen floors: paired mean >= 0.06, per-seed min >= 0.03, coupled
        absolute mean >= 0.85.  Measured held-out: +0.1648 mean / +0.1250 min,
        coupled mean 0.9319 (per-seed range 0.9017-0.9583).
        """
        coupled, raw = p2_runs
        assert bool(jnp.all(jnp.isfinite(coupled[0])))
        assert bool(jnp.all(jnp.isfinite(raw)))
        coupled_rec = recurrence_early(coupled[0])
        raw_rec = recurrence_early(raw)
        paired = coupled_rec - raw_rec
        assert float(paired.mean()) >= 0.06
        assert float(paired.min()) >= 0.03
        assert float(coupled_rec.mean()) >= 0.85

    def test_discovered_bank_floors_hold(self, p2_runs) -> None:
        """Distinct oracle products in the final control bank, held out.

        Frozen floors: per-seed min >= 2, median >= 3.  Measured held-out:
        per-seed counts [3 3 4 4 4 4 4 3] — min 3, median 4.
        """
        coupled, _ = p2_runs
        distinct_final = np.asarray(coupled[1])[:, -1]
        assert int(distinct_final.min()) >= 2
        assert float(np.median(distinct_final)) >= 3.0


# ---------------------------------------------------------------------------
# Protocol 3: L3 integrated life, gated rung + paired ablation, held out
# ---------------------------------------------------------------------------


class TestHeldOutIntegratedLife:
    def test_stability_zero_nan(self, p3_lives) -> None:
        """No non-finite reward or Q-weight anywhere in 16 held-out lives."""
        gated, ablation = p3_lives
        for run in gated + ablation:
            assert np.isfinite(run.rewards).all()
            assert np.isfinite(np.asarray(run.final_state.q_weights)).all()

    def test_memory_floors_hold(self, p3_lives) -> None:
        """Settled early-window and paired ablation gap on seeds 5000..5007.

        Frozen floors (imported): settled early-window >= 0.60 per seed and
        paired gated-ablation settled gap >= 0.30 (min and mean).  Measured
        held-out: settled min 0.8800 per-seed worst; gap min +0.5657 / mean
        +0.5906 — matching the calibration-seed values (0.860 / +0.586).
        """
        gated, ablation = p3_lives
        settled_min = _metric(gated, "early_settled_min")
        assert float(settled_min.min()) >= MEM_SETTLED_MIN
        gap = _metric(gated, "early_settled_mean") - _metric(ablation, "early_settled_mean")
        assert float(gap.min()) >= MEM_ABLATION_GAP
        assert float(gap.mean()) >= MEM_ABLATION_GAP

    def test_control_floors_hold(self, p3_lives) -> None:
        """Lifetime and late-life control floors on seeds 5000..5007.

        Frozen floors (imported): life per-seed min >= 0.85, life mean >=
        random + 0.35, late-8000 per-seed min >= optimum - 0.08.  Measured
        held-out: life min 0.9674 / mean 0.9694, late min 0.9709.
        """
        gated, _ = p3_lives
        life = _metric(gated, "life")
        late = _metric(gated, "late")
        assert float(life.min()) >= LIFE_FLOOR
        assert float(life.mean()) >= RANDOM + LIFE_VS_RANDOM
        assert float(late.min()) >= OPTIMAL - NEAR_OPT_SLACK


# ---------------------------------------------------------------------------
# Protocol 4: 1M-step lifetime longevity, held out
# ---------------------------------------------------------------------------


class TestHeldOutLifetimeLongevity:
    def test_no_divergence(self, p4_scorecard) -> None:
        """Zero non-finite squared errors across 8 held-out 1M-step lives."""
        assert int(jnp.sum(p4_scorecard["nan_steps"])) == 0

    def test_memory_does_not_erode(self, p4_scorecard) -> None:
        """Late savings floors on jr.key(77): measured 12.46 late vs 9.55
        early (frozen floors: late > 5 and late > 0.5 x early)."""
        savings_c = jnp.median(p4_scorecard["savings_c"], axis=0)
        early = float(jnp.median(savings_c[:25]))
        late = float(jnp.median(savings_c[-25:]))
        assert late > 5.0
        assert late > 0.5 * early

    def test_late_reentry_near_solution(self, p4_scorecard) -> None:
        """Measured held-out: late re-entry 0.196 vs first exposure 2.108
        (ratio 0.093; frozen ceiling 0.25)."""
        recur_c = jnp.median(p4_scorecard["recur_c_early"], axis=0)
        late_mean = float(jnp.mean(recur_c[-20:]))
        assert late_mean < 0.25 * float(recur_c[0])

    def test_plasticity_flat(self, p4_scorecard) -> None:
        """Measured held-out: unstressed fresh-entry late/early 11.92/11.44 =
        1.04 (frozen ceiling 2.0)."""
        fresh = jnp.median(p4_scorecard["fresh_early"], axis=0)
        unstressed = jnp.array([i for i in range(3, P4_N_CYCLES) if i % 3 != 2], dtype=jnp.int32)
        vals = fresh[unstressed]
        early_mean = float(jnp.mean(vals[:20]))
        late_mean = float(jnp.mean(vals[-20:]))
        assert late_mean < 2.0 * early_mean
