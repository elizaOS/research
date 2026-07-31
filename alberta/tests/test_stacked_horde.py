"""Tests for the stacked linear Horde (core/stacked_horde.py).

Covers exact TD(lambda) semantics (hand-computed two-step scenario),
convergence to analytic GVF fixed points on a 3-state cycle, NaN-cumulant
masking, per-decision IS composition, nexting-style multi-timescale
prediction, and — the module's reason to exist — demon-count scaling: the
demon axis is a stacked array axis, so program size is constant in
``n_demons`` and 1024 demons run a 2000-step scan in well under a second
after compile (the loop-unrolled hordes measured ~14 steps/s = ~140 s for
the same workload, with a ~144 s compile; see the scaling notes in
CONTINUAL_LEARNING_EVIDENCE.md).
"""

import time

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.stacked_horde import (
    StackedHordeConfig,
    StackedLinearHorde,
    nexting_spec,
    run_stacked_horde_scan,
)


def _simple_config(n_demons=2, feature_dim=3, **kw) -> StackedHordeConfig:
    defaults = dict(
        n_demons=n_demons,
        feature_dim=feature_dim,
        gammas=(0.9,) * n_demons,
        lamdas=(0.5,) * n_demons,
        cumulant_indices=tuple(range(n_demons)),
        step_size=0.1,
    )
    defaults.update(kw)
    return StackedHordeConfig(**defaults)


class TestConfig:
    def test_validation(self):
        with pytest.raises(ValueError, match="n_demons"):
            _simple_config(n_demons=0, gammas=(), lamdas=(), cumulant_indices=())
        with pytest.raises(ValueError, match="length"):
            _simple_config(gammas=(0.9,))
        with pytest.raises(ValueError, match="gamma"):
            _simple_config(gammas=(1.5, 0.9))
        with pytest.raises(ValueError, match="step_size"):
            _simple_config(step_size=0.0)

    def test_roundtrip(self):
        cfg = _simple_config()
        horde = StackedLinearHorde(cfg)
        restored = StackedLinearHorde.from_config(horde.to_config())
        assert restored.config == cfg

    def test_nexting_spec_shape(self):
        cfg = nexting_spec(feature_dim=6, cumulant_indices=(0, 2), gammas=(0.0, 0.9))
        assert cfg.n_demons == 4
        assert cfg.cumulant_indices == (0, 0, 2, 2)
        assert cfg.gammas == (0.0, 0.9, 0.0, 0.9)


class TestExactSemantics:
    def test_hand_computed_two_step(self):
        """Exact TD(lambda) values for one demon over two transitions."""
        cfg = StackedHordeConfig(
            n_demons=1,
            feature_dim=2,
            gammas=(0.9,),
            lamdas=(0.5,),
            cumulant_indices=(0,),
            step_size=0.1,
        )
        horde = StackedLinearHorde(cfg)
        state = horde.init()

        x0 = jnp.array([1.0, 0.0])
        x1 = jnp.array([0.0, 1.0])
        c = jnp.array([2.0])

        # Step 1: v=0, v'=0, delta = 2.0; z = 0.45*0 + x0 = [1,0];
        # w = 0.1*2.0*[1,0] = [0.2, 0].
        r1 = horde.update(state, x0, x1, c)
        np.testing.assert_allclose(np.asarray(r1.td_errors), [2.0], rtol=1e-6)
        np.testing.assert_allclose(np.asarray(r1.state.weights), [[0.2, 0.0]], rtol=1e-6)
        np.testing.assert_allclose(np.asarray(r1.state.traces), [[1.0, 0.0]], rtol=1e-6)

        # Step 2 (x1 -> x0): v = w@x1 = 0, v' = w@x0 = 0.2,
        # delta = 2.0 + 0.9*0.2 - 0 = 2.18; z = 0.45*[1,0] + [0,1] = [0.45,1];
        # w += 0.1*2.18*[0.45,1] = [0.2981, 0.218].
        r2 = horde.update(r1.state, x1, x0, c)
        np.testing.assert_allclose(np.asarray(r2.td_errors), [2.18], rtol=1e-6)
        np.testing.assert_allclose(
            np.asarray(r2.state.weights), [[0.2 + 0.0981, 0.218]], rtol=1e-5
        )

    def test_nan_cumulant_freezes_weights_decays_trace(self):
        cfg = _simple_config()
        horde = StackedLinearHorde(cfg)
        state = horde.init()
        x = jnp.array([1.0, 2.0, 3.0])

        # Warm up demon traces/weights.
        r = horde.update(state, x, x, jnp.array([1.0, 1.0, 0.0]))
        # Demon 0 goes inactive; demon 1 stays active.
        r2 = horde.update(r.state, x, x, jnp.array([jnp.nan, 1.0, 0.0]))
        assert bool(jnp.isnan(r2.td_errors[0]))
        assert bool(jnp.isfinite(r2.td_errors[1]))
        # Demon 0 weights frozen.
        np.testing.assert_array_equal(
            np.asarray(r2.state.weights[0]), np.asarray(r.state.weights[0])
        )
        # Demon 0 trace decayed (no gradient added): z2 = gamma*lamda*z1.
        np.testing.assert_allclose(
            np.asarray(r2.state.traces[0]),
            0.9 * 0.5 * np.asarray(r.state.traces[0]),
            rtol=1e-6,
        )
        # Demon 1 weights moved.
        assert not np.array_equal(
            np.asarray(r2.state.weights[1]), np.asarray(r.state.weights[1])
        )

    def test_rho_composes_into_trace(self):
        """z = rho * (decay * z + x): rho=0 zeroes the trace and the update."""
        cfg = _simple_config()
        horde = StackedLinearHorde(cfg)
        state = horde.init()
        x = jnp.array([1.0, 0.0, 0.0])

        r = horde.update(state, x, x, jnp.array([1.0, 1.0, 0.0]), rho=0.0)
        np.testing.assert_array_equal(np.asarray(r.state.weights), 0.0)
        np.testing.assert_array_equal(np.asarray(r.state.traces), 0.0)

        # rho=2 doubles the trace relative to rho=1.
        r1 = horde.update(state, x, x, jnp.array([1.0, 1.0, 0.0]), rho=1.0)
        r2 = horde.update(state, x, x, jnp.array([1.0, 1.0, 0.0]), rho=2.0)
        np.testing.assert_allclose(
            np.asarray(r2.state.traces), 2.0 * np.asarray(r1.state.traces), rtol=1e-6
        )


class TestConvergence:
    def test_three_state_cycle_analytic_fixed_point(self):
        """On a deterministic 3-state cycle with one-hot features, every
        demon's values converge to the analytic discounted fixed point
        v(s) = sum_k gamma^k c(s_{t+1+k}) — checked for gamma 0 and 0.8 at
        two different cumulant channels simultaneously."""
        gammas = (0.0, 0.8, 0.0, 0.8)
        cumulant_indices = (0, 0, 1, 1)
        cfg = StackedHordeConfig(
            n_demons=4,
            feature_dim=3,
            gammas=gammas,
            lamdas=(0.9,) * 4,
            cumulant_indices=cumulant_indices,
            step_size=0.05,
        )
        horde = StackedLinearHorde(cfg)
        state = horde.init()

        # Cycle s0 -> s1 -> s2 -> s0; cumulant channel 0 = [1, 0, 0] by next
        # state, channel 1 = [0, 2, 0].
        eye = jnp.eye(3, dtype=jnp.float32)
        c_by_state = jnp.array([[1.0, 0.0], [0.0, 2.0], [0.0, 0.0]])
        num_steps = 4000
        order = jnp.arange(num_steps) % 3
        features = eye[order]
        # Cumulant for the t -> t+1 transition is the channel value at s_{t+1}.
        next_order = (order + 1) % 3
        sources = c_by_state[next_order]

        state, _ = run_stacked_horde_scan(horde, state, features, sources)

        # Analytic: for gamma, v(s_i) = sum_{k>=0} gamma^k c(s_{i+1+k}).
        def analytic(gamma, channel):
            c = np.asarray(c_by_state[:, channel])
            v = np.zeros(3)
            for i in range(3):
                # Geometric sum over the period-3 cycle.
                per = np.array([c[(i + 1 + k) % 3] * gamma**k for k in range(3)])
                v[i] = per.sum() / (1.0 - gamma**3)
            return v

        w = np.asarray(state.weights)
        for d, (g, ch) in enumerate(zip(gammas, cumulant_indices)):
            np.testing.assert_allclose(w[d], analytic(g, ch), atol=0.02)

    def test_nexting_multi_timescale_orderings(self):
        """Nexting demons at gammas (0, 0.5, 0.9) over a recurring pulse:
        longer-timescale predictions are larger ahead of the pulse (they
        accumulate more future signal) — the qualitative nexting signature."""
        cfg = nexting_spec(
            feature_dim=8,
            cumulant_indices=(0,),
            gammas=(0.0, 0.5, 0.9),
            step_size=0.1,
        )
        horde = StackedLinearHorde(cfg)
        state = horde.init()

        # Period-8 one-hot cycle; pulse fires at phase 0 (cumulant = 1 when
        # the next state is phase 0).
        num_steps = 6000
        order = jnp.arange(num_steps) % 8
        features = jnp.eye(8, dtype=jnp.float32)[order]
        pulse = (jnp.roll(order, -1) % 8 == 0).astype(jnp.float32)
        sources = pulse[:, None]

        state, _ = run_stacked_horde_scan(horde, state, features, sources)
        # At phase 7 (one step before the pulse) all timescales see it;
        # at phase 4 only the long-timescale demon still sees much of it.
        v7 = np.asarray(horde.predict(state, jnp.eye(8)[7]))
        v4 = np.asarray(horde.predict(state, jnp.eye(8)[4]))
        assert v7[0] > 0.9  # gamma=0: next-step pulse predicted ~1
        assert v4[0] < 0.1  # gamma=0: nothing next step
        assert v4[2] > v4[1] > v4[0]  # longer horizons see the coming pulse


class TestDemonAxisScaling:
    def test_1024_demons_run_fast_with_constant_program_size(self):
        """1024 demons x 2000 steps completes in seconds, not minutes.

        The loop-unrolled hordes measured ~14 steps/s at 1024 demons with a
        ~144 s compile (16.4 GB working set).  The stacked horde runs the
        same demon count as one batched update; this test bounds the whole
        thing — compile included — at 30 s and the post-compile scan at 5 s,
        both enormous (>25x) margins over measured values (~1.5 s / ~0.02 s
        on the 24-core dev box).
        """
        n_demons, feature_dim, num_steps = 1024, 32, 2000
        key = jr.key(0)
        rng = jr.split(key, 2)
        cfg = StackedHordeConfig(
            n_demons=n_demons,
            feature_dim=feature_dim,
            gammas=tuple(float(g) for g in np.linspace(0.0, 0.99, n_demons)),
            lamdas=(0.7,) * n_demons,
            cumulant_indices=tuple(int(i) for i in np.arange(n_demons) % feature_dim),
            step_size=0.01,
        )
        horde = StackedLinearHorde(cfg)
        state = horde.init()
        features = jr.normal(rng[0], (num_steps, feature_dim), dtype=jnp.float32)
        sources = jr.normal(rng[1], (num_steps, feature_dim), dtype=jnp.float32)

        t0 = time.time()
        final_state, td_errors = run_stacked_horde_scan(
            horde, state, features, sources
        )
        td_errors.block_until_ready()
        first_call = time.time() - t0
        assert first_call < 30.0, f"compile+run took {first_call:.1f}s"

        t1 = time.time()
        final_state, td_errors = run_stacked_horde_scan(
            horde, state, features, sources
        )
        td_errors.block_until_ready()
        steady = time.time() - t1
        assert steady < 5.0, f"steady-state run took {steady:.1f}s"

        assert bool(jnp.all(jnp.isfinite(final_state.weights)))
        assert td_errors.shape == (num_steps - 1, n_demons)

    def test_learning_quality_survives_at_scale(self):
        """All 1024 demons actually learn: on a 3-state cycle every demon's
        prediction error shrinks between the first and last 200 steps."""
        n_demons = 1024
        cfg = StackedHordeConfig(
            n_demons=n_demons,
            feature_dim=3,
            gammas=tuple(float(g) for g in np.linspace(0.0, 0.95, n_demons)),
            lamdas=(0.8,) * n_demons,
            cumulant_indices=(0,) * n_demons,
            step_size=0.05,
        )
        horde = StackedLinearHorde(cfg)
        state = horde.init()
        num_steps = 3000
        order = jnp.arange(num_steps) % 3
        features = jnp.eye(3, dtype=jnp.float32)[order]
        pulse = ((order + 1) % 3 == 0).astype(jnp.float32)
        sources = pulse[:, None]

        _, td_errors = run_stacked_horde_scan(horde, state, features, sources)
        early = jnp.mean(td_errors[:200] ** 2, axis=0)  # (n_demons,)
        late = jnp.mean(td_errors[-200:] ** 2, axis=0)
        # Every single demon improved.
        assert bool(jnp.all(late < early))
        # And the late TD error is near zero for all timescales.
        assert float(jnp.max(late)) < 0.01
