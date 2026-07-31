"""Baird counterexample suite for the off-policy Horde backends.

Horde-level companion to ``tests/test_baird.py`` (which covers the linear
learners in ``core/off_policy_td.py``): the same canonical 7-state Baird star
(Baird 1995; Sutton & Barto 2018, section 11.2) is fed to the two Horde
backends in ``core/off_policy_horde.py``, now that the per-decision IS fix
landed there (the ratio moved into the eligibility trace, and the GTD
backend's correction term carries rho).

Star construction (identical to ``tests/test_baird.py``):

- Six upper states (0..5) and one lower state (6).  Behavior takes *dashed*
  (uniform over upper states) with probability 6/7 and *solid* (to the lower
  state) with 1/7; the target policy always takes solid.  Importance ratios
  are rho = 0 on dashed and rho = 7 on solid transitions.
- All cumulants are zero and gamma = 0.99, so the true value function is zero.
- Classic 8-feature representation phi(s) = 2 e_s + e_7 for upper states,
  phi(6) = e_6 + 2 e_7; classic initial head weights (1, 1, 1, 1, 1, 1, 10, 1).

Because the 7x8 feature matrix has full row rank, the MSPBE projection is the
identity for linear heads and ``MSPBE(V) = mean_s (gamma V(6) - V(s))**2``
under the uniform distribution.  The same formula is used for the tanh-trunk
GTD backend, where it is the uniform-weighted mean squared *expected* TD error
under the target policy — zero iff V is identically zero — a representation-
independent convergence criterion.

Everything runs with 2 demons; demon 1 receives NaN cumulants on a fixed
schedule to exercise the active-mask paths alongside the star dynamics.

All thresholds calibrated empirically on CPU (jax float32):

- Semi-gradient divergence (LMS alpha=0.01, clips inf, trajectory seeds
  0/1/2): per-1000-step weight-norm growth ratios 5.0x-44x (assert > 2x);
  final norms 7.6e4-2.1e5 from initial 10.34 (assert > 1e3).
- GTD convergence (alpha=2e-3, beta=5e-2, ratio_clip=10, 60k steps,
  trajectory seeds 1/2/3/7): initial surrogate MSPBE 14.446; final MSPBE
  1.2e-4-4.1e-3, i.e. reduction 3.5e3x-1.2e5x (assert >= 100x reduction and
  final < 0.05); max parameter norm 15.0-15.5 from initial 14.7
  (assert < 30).
"""

from __future__ import annotations

import chex
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from alberta_framework.core.multi_head_learner import MultiHeadMLPState
from alberta_framework.core.off_policy_horde import (
    NonlinearSharedGTDHordeLearner,
    NonlinearSharedGTDHordeState,
    OffPolicyHordeLearner,
)
from alberta_framework.core.optimizers import LMS
from alberta_framework.core.types import (
    DemonType,
    GVFSpec,
    HordeSpec,
    MLPParams,
    create_horde_spec,
)

# =============================================================================
# Baird star: hand-computed dynamics and representation
# =============================================================================

GAMMA = 0.99
N_STATES = 7
N_FEATURES = 8
LOWER = 6  # index of the lower ("solid") state
RHO_SOLID = 7.0  # pi(solid|s) / b(solid|s) = 1 / (1/7)
RHO_DASHED = 0.0  # pi(dashed|s) / b(dashed|s) = 0 / (6/7)
N_DEMONS = 2


def _baird_features() -> np.ndarray:
    """The classic 8-feature Baird star representation, one row per state."""
    phi = np.zeros((N_STATES, N_FEATURES), dtype=np.float32)
    for s in range(6):
        phi[s, s] = 2.0
        phi[s, N_FEATURES - 1] = 1.0
    phi[LOWER, LOWER] = 1.0
    phi[LOWER, N_FEATURES - 1] = 2.0
    return phi


PHI = _baird_features()
W_INIT = np.array([1, 1, 1, 1, 1, 1, 10, 1], dtype=np.float32)


def _sample_trajectory(n_steps: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample a behavior-policy trajectory; returns (obs, next_obs, rhos)."""
    rng = np.random.default_rng(seed)
    states = np.zeros(n_steps + 1, dtype=np.int64)
    rhos = np.zeros(n_steps, dtype=np.float32)
    states[0] = LOWER
    for t in range(n_steps):
        if rng.random() < 6.0 / 7.0:
            states[t + 1] = rng.integers(0, 6)  # dashed: uniform over upper states
            rhos[t] = RHO_DASHED
        else:
            states[t + 1] = LOWER  # solid: always to the lower state
            rhos[t] = RHO_SOLID
    return PHI[states[:-1]], PHI[states[1:]], rhos


def _make_spec(lamdas: tuple[float, ...] = (0.0, 0.0)) -> HordeSpec:
    """Two prediction demons on the star, gamma = 0.99, per-demon lambdas."""
    demons = tuple(
        GVFSpec(  # type: ignore[call-arg]
            name=f"demon_{i}",
            demon_type=DemonType.PREDICTION,
            gamma=GAMMA,
            lamda=lamdas[i],
            cumulant_index=i,
        )
        for i in range(N_DEMONS)
    )
    return create_horde_spec(demons)


def _mspbe(values: np.ndarray) -> float:
    """MSPBE under the uniform distribution (projection is the identity for
    the full-row-rank linear heads; the same expression is the mean squared
    expected TD error for the tanh backend, zero iff values are all zero)."""
    return float(np.mean((GAMMA * values[LOWER] - values) ** 2))


def _horde_stream(
    n_steps: int, seed: int, nan_every: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Baird stream broadcast to 2 demons; demon 1 gets NaN cumulants on
    every ``nan_every``-th step to exercise the active-mask paths."""
    obs, next_obs, rhos = _sample_trajectory(n_steps, seed=seed)
    cums = np.zeros((n_steps, N_DEMONS), dtype=np.float32)
    cums[nan_every - 1 :: nan_every, 1] = np.nan
    rhos2 = np.tile(rhos[:, None], (1, N_DEMONS)).astype(np.float32)
    return obs, cums, next_obs, rhos2


# =============================================================================
# (1) Semi-gradient off-policy Horde diverges on the star
# =============================================================================


class TestSemiGradientHordeDivergence:
    def test_linear_semi_gradient_horde_diverges(self) -> None:
        """OffPolicyHordeLearner with linear heads (``hidden_sizes=()``), no
        gradient correction, and effectively infinite ratio clips is exactly
        semi-gradient per-decision IS TD(0) — Baird's counterexample.

        Measured (LMS alpha=0.01, trajectory seed 0): demon-0 norms at 1000-step
        checkpoints [359, 3.1e3, 2.6e4, 2.1e5]; demon-1 (with 1-in-8 NaN-masked
        steps) [212, 1.6e3, 1.1e4, 7.6e4]; worst per-window growth across seeds
        0/1/2 is 5.0x.  Asserted: > 2x per window, final > 1e3, all finite.
        """
        learner = OffPolicyHordeLearner(
            _make_spec(),
            hidden_sizes=(),
            optimizer=LMS(step_size=0.01),
            ratio_clip=float("inf"),
            trace_ratio_clip=float("inf"),
        )
        state = learner.init(N_FEATURES, jax.random.key(0))
        w0 = jnp.asarray(W_INIT).reshape(1, -1)
        state = state.replace(  # type: ignore[attr-defined]
            head_params=MLPParams(  # type: ignore[call-arg]
                weights=tuple(w0 for _ in range(N_DEMONS)),
                biases=state.head_params.biases,
            )
        )
        n_steps, nan_every = 4000, 8
        obs, cums, next_obs, rhos = _horde_stream(n_steps, seed=0, nan_every=nan_every)

        def step(
            carry: MultiHeadMLPState, inputs: tuple[Array, Array, Array, Array]
        ) -> tuple[MultiHeadMLPState, tuple[Array, Array]]:
            o, c, no, r = inputs
            result = learner.update_with_ratios(carry, o, c, no, r)
            norms = jnp.stack(
                [
                    jnp.linalg.norm(result.state.head_params.weights[i])
                    for i in range(N_DEMONS)
                ]
            )
            return result.state, (norms, result.td_errors)

        final_state, (norms, td_errors) = jax.lax.scan(
            step,
            state,
            (jnp.asarray(obs), jnp.asarray(cums), jnp.asarray(next_obs), jnp.asarray(rhos)),
        )
        norms = np.asarray(norms)
        td_errors = np.asarray(td_errors)

        # Monotone explosive growth across 1000-step windows for both demons...
        initial_norm = float(np.linalg.norm(W_INIT))
        for demon in range(N_DEMONS):
            previous = initial_norm
            for norm in norms[999::1000, demon]:
                assert norm > 2.0 * previous, (
                    f"demon {demon} weight norm not exploding: {norms[999::1000, demon]}"
                )
                previous = float(norm)
            # ...ending far above any bounded iterate (initial norm is ~10.3).
            assert float(norms[-1, demon]) > 1e3
        # Divergence stays finite (no NaN poisoning through the masked demon).
        chex.assert_tree_all_finite(final_state.head_params)
        # The mask schedule is reported faithfully: demon 1's TD error is NaN on
        # exactly the masked steps, demon 0's never.
        assert int(np.isnan(td_errors[:, 1]).sum()) == n_steps // nan_every
        assert int(np.isnan(td_errors[:, 0]).sum()) == 0


# =============================================================================
# (2) GTD-corrected Horde converges on the identical stream
# =============================================================================


class TestGTDHordeConvergence:
    def test_gtd_horde_reduces_mspbe_100x_with_bounded_norms(self) -> None:
        """NonlinearSharedGTDHordeLearner in its (near-)linear configuration
        converges on the same star stream that explodes semi-gradient TD.

        The backend has no strictly linear mode, so the linear configuration is
        an identity trunk in the tanh near-linear regime: hidden_size = 8,
        trunk_w = 0.5 I, zero biases, and the classic (1,...,1,10,1) head
        weights on both demons.  The induced initial values (upper ~1.22,
        lower ~5.38) preserve the star's divergence pressure: surrogate MSPBE
        starts at 14.446.  The rho-carrying correction term keeps the iterates
        bounded and drives the MSPBE down.

        Measured (alpha=2e-3, beta=5e-2, ratio_clip=10, 60k steps): final MSPBE
        [1.2e-4, 1.6e-4] at trajectory seed 1; worst over seeds 1/2/3/7 is
        4.1e-3 (a 3.5e3x reduction); max sqrt(|trunk_w|^2+|head_w|^2) is 15.5
        from initial 14.7.  Asserted: >= 100x reduction per demon, final
        < 0.05, max norm < 30, all finite.
        """
        learner = NonlinearSharedGTDHordeLearner(
            _make_spec(),
            hidden_size=N_FEATURES,
            primary_step_size=2e-3,
            secondary_step_size=5e-2,
            ratio_clip=10.0,
        )
        state = learner.init(N_FEATURES, jax.random.key(1)).replace(  # type: ignore[attr-defined]
            trunk_w=0.5 * jnp.eye(N_FEATURES, dtype=jnp.float32),
            trunk_b=jnp.zeros(N_FEATURES, dtype=jnp.float32),
            head_w=jnp.tile(jnp.asarray(W_INIT).reshape(1, -1), (N_DEMONS, 1)),
            head_b=jnp.zeros(N_DEMONS, dtype=jnp.float32),
        )

        def values(s: NonlinearSharedGTDHordeState) -> np.ndarray:
            """Per-state predictions, shape (n_states, n_demons)."""
            return np.stack(
                [np.asarray(learner.predict(s, jnp.asarray(PHI[i]))) for i in range(N_STATES)]
            )

        initial_mspbe = [_mspbe(values(state)[:, d]) for d in range(N_DEMONS)]
        for m0 in initial_mspbe:
            assert m0 > 5.0  # sanity: the adversarial init starts far from the fix-point

        n_steps, nan_every = 60_000, 16
        obs, cums, next_obs, rhos = _horde_stream(n_steps, seed=1, nan_every=nan_every)
        discounts = np.full((n_steps, N_DEMONS), GAMMA, dtype=np.float32)

        def step(
            carry: NonlinearSharedGTDHordeState,
            inputs: tuple[Array, Array, Array, Array, Array],
        ) -> tuple[NonlinearSharedGTDHordeState, tuple[Array, Array]]:
            o, c, no, r, g = inputs
            result = learner.update_with_ratios_and_discounts(carry, o, c, no, r, g)
            norm = jnp.sqrt(
                jnp.vdot(result.state.trunk_w, result.state.trunk_w)
                + jnp.vdot(result.state.head_w, result.state.head_w)
            )
            return result.state, (norm, result.td_errors)

        final_state, (norms, td_errors) = jax.lax.scan(
            step,
            state,
            (
                jnp.asarray(obs),
                jnp.asarray(cums),
                jnp.asarray(next_obs),
                jnp.asarray(rhos),
                jnp.asarray(discounts),
            ),
        )
        norms = np.asarray(norms)
        td_errors = np.asarray(td_errors)

        chex.assert_tree_all_finite(final_state)
        # Bounded iterates on the stream that explodes the uncorrected backend
        # (semi-gradient norms pass 2e5 in a fifteenth of these steps).
        assert float(np.max(norms)) < 30.0, "GTD horde iterates must stay bounded on Baird"

        final_values = values(final_state)
        for demon in range(N_DEMONS):
            final_mspbe = _mspbe(final_values[:, demon])
            assert final_mspbe < 0.05, f"demon {demon} MSPBE not near zero: {final_mspbe}"
            assert final_mspbe < 1e-2 * initial_mspbe[demon], (
                f"demon {demon} MSPBE reduction < 100x: "
                f"{initial_mspbe[demon]} -> {final_mspbe}"
            )
        # NaN masking: demon 1 was inactive on exactly the scheduled steps and
        # still converged; demon 0 never went inactive.
        assert int(np.isnan(td_errors[:, 1]).sum()) == n_steps // nan_every
        assert int(np.isnan(td_errors[:, 0]).sum()) == 0


# =============================================================================
# (3) rho = 0 transitions leave primary weights invariant at the horde level
# =============================================================================


class TestRhoZeroInvariance:
    """A dashed transition (rho = 0) must not move any primary weights: the
    ratio multiplies the head gradient, the trace decay, and (for the GTD
    backend) both the delta term and the correction term."""

    @staticmethod
    def _warmed_semi_gradient_horde() -> tuple[OffPolicyHordeLearner, MultiHeadMLPState]:
        """Nonlinear semi-gradient horde warmed with two rho=1 transitions so
        heads, trunk, and traces are all away from their init."""
        learner = OffPolicyHordeLearner(
            _make_spec(lamdas=(0.7, 0.0)),
            hidden_sizes=(16,),
            optimizer=LMS(step_size=0.05),
            ratio_clip=float("inf"),
            trace_ratio_clip=float("inf"),
            sparsity=0.5,
        )
        state = learner.init(N_FEATURES, jax.random.key(4))
        for t in range(2):
            state = learner.update_with_ratios(
                state,
                jnp.asarray(PHI[t]),
                jnp.array([1.0, -0.5], dtype=jnp.float32),
                jnp.asarray(PHI[t + 1]),
                jnp.ones(N_DEMONS, dtype=jnp.float32),
            ).state
        return learner, state

    def test_all_demons_rho_zero_freezes_heads_and_trunk(self) -> None:
        learner, state = self._warmed_semi_gradient_horde()
        result = learner.update_with_ratios(
            state,
            jnp.asarray(PHI[2]),
            jnp.array([1.3, 0.7], dtype=jnp.float32),
            jnp.asarray(PHI[LOWER]),
            jnp.zeros(N_DEMONS, dtype=jnp.float32),
        )
        # Head weights/biases and the shared trunk are bit-exactly unchanged.
        chex.assert_trees_all_equal(result.state.head_params, state.head_params)
        chex.assert_trees_all_equal(result.state.trunk_params, state.trunk_params)
        # rho = 0 also cuts the eligibility traces to zero (z = rho (...) = 0),
        # so nothing can leak into the next update.
        for i in range(N_DEMONS):
            chex.assert_trees_all_close(
                result.state.head_traces[i],
                jax.tree.map(jnp.zeros_like, state.head_traces[i]),
            )

    def test_mixed_rho_zero_is_per_demon(self) -> None:
        """Demon 0 dashed (rho=0), demon 1 solid (rho=7): demon 0's head is
        frozen while demon 1's head and the shared trunk both move (measured
        deltas 0.41 and 0.36; asserted > 0)."""
        learner, state = self._warmed_semi_gradient_horde()
        result = learner.update_with_ratios(
            state,
            jnp.asarray(PHI[2]),
            jnp.array([1.3, 0.7], dtype=jnp.float32),
            jnp.asarray(PHI[LOWER]),
            jnp.array([RHO_DASHED, RHO_SOLID], dtype=jnp.float32),
        )
        chex.assert_trees_all_equal(
            result.state.head_params.weights[0], state.head_params.weights[0]
        )
        chex.assert_trees_all_equal(
            result.state.head_params.biases[0], state.head_params.biases[0]
        )
        head1_delta = float(
            jnp.linalg.norm(result.state.head_params.weights[1] - state.head_params.weights[1])
        )
        trunk_delta = float(
            jnp.linalg.norm(
                result.state.trunk_params.weights[0] - state.trunk_params.weights[0]
            )
        )
        assert head1_delta > 0.0
        assert trunk_delta > 0.0

    def test_gtd_rho_zero_freezes_primary_weights(self) -> None:
        """rho = 0 zeroes both the rho*delta term and the rho-carrying
        correction, so all primary GTD parameters are exactly unchanged; the
        secondary weights still take their prescribed -(v . grad) grad decay
        step (measured delta 0.109 from a warm norm of 0.50; asserted > 0)."""
        learner = NonlinearSharedGTDHordeLearner(
            _make_spec(),
            hidden_size=N_FEATURES,
            primary_step_size=0.01,
            secondary_step_size=0.05,
            ratio_clip=10.0,
        )
        state = learner.init(N_FEATURES, jax.random.key(5))
        obs = jnp.asarray(PHI[1])
        next_obs = jnp.asarray(PHI[LOWER])
        discounts = jnp.full(N_DEMONS, GAMMA, dtype=jnp.float32)
        # Warm up with one solid transition so the secondary weights are nonzero.
        warm = learner.update_with_ratios_and_discounts(
            state,
            obs,
            jnp.array([1.0, 1.0], dtype=jnp.float32),
            next_obs,
            jnp.full(N_DEMONS, RHO_SOLID, dtype=jnp.float32),
            discounts,
        ).state
        assert float(jnp.linalg.norm(warm.secondary_head_w)) > 0.0

        result = learner.update_with_ratios_and_discounts(
            warm,
            obs,
            jnp.array([1.3, -0.4], dtype=jnp.float32),
            next_obs,
            jnp.zeros(N_DEMONS, dtype=jnp.float32),
            discounts,
        )
        chex.assert_trees_all_equal(result.state.trunk_w, warm.trunk_w)
        chex.assert_trees_all_equal(result.state.trunk_b, warm.trunk_b)
        chex.assert_trees_all_equal(result.state.head_w, warm.head_w)
        chex.assert_trees_all_equal(result.state.head_b, warm.head_b)
        secondary_delta = float(
            jnp.linalg.norm(result.state.secondary_head_w - warm.secondary_head_w)
        )
        assert secondary_delta > 0.0


# =============================================================================
# (4) Fixed per-decision trace composition (regression guard for the rho fix)
# =============================================================================


class TestPerDecisionTraceComposition:
    def test_three_step_trace_carries_one_rho_per_decision(self) -> None:
        """Hand-computed 3-step check of ``z_t = rho_t (gamma lambda z_{t-1}
        + phi_t)`` on star features, with a NaN-masked middle step for demon 1.

        After step 2 the coefficient on phi_1 in demon 0's trace must be
        ``gamma * lambda * rho_2 * rho_1`` — one ratio per decision.  With
        rho_1=7, rho_2=2 that is 9.702; the pre-fix composition that kept the
        trace decay ratio-free gives ``gamma * lambda * rho_2`` = 1.386, and a
        double-application gives ``gamma * lambda * rho_2**2`` = 2.772, so the
        exact-value assertions distinguish all three.  Demon 1's masked step
        must freeze both its trace and its head parameters.
        """
        lam0, lam1 = 0.7, 0.4
        learner = OffPolicyHordeLearner(
            _make_spec(lamdas=(lam0, lam1)),
            hidden_sizes=(),
            optimizer=LMS(step_size=0.1),
            ratio_clip=float("inf"),
            trace_ratio_clip=float("inf"),
        )
        state = learner.init(N_FEATURES, jax.random.key(3))

        # (obs, cumulants, next_obs, rhos) per step; star features 0 -> 3 -> 6.
        # Demon 1's step-2 cumulant is NaN (inactive).
        steps = [
            (PHI[0], [0.0, 0.0], PHI[3], [7.0, 0.5]),
            (PHI[3], [1.0, np.nan], PHI[LOWER], [2.0, 3.0]),
            (PHI[LOWER], [0.5, -0.5], PHI[0], [5.0, 2.0]),
        ]
        states = [state]
        for o, c, no, r in steps:
            state = learner.update_with_ratios(
                state,
                jnp.asarray(o),
                jnp.asarray(c, dtype=jnp.float32),
                jnp.asarray(no),
                jnp.asarray(r, dtype=jnp.float32),
            ).state
            states.append(state)

        def assert_traces(s: MultiHeadMLPState, demon: int, z: np.ndarray, zb: float) -> None:
            np.testing.assert_allclose(
                np.asarray(s.head_traces[demon][0]).ravel(), z, rtol=1e-5, atol=1e-5
            )
            np.testing.assert_allclose(
                np.asarray(s.head_traces[demon][1]).ravel(), [zb], rtol=1e-5, atol=1e-5
            )

        phi = PHI.astype(np.float64)

        # Demon 0 (never masked): one ratio per decision, composed step by step.
        z0_1 = 7.0 * phi[0]
        zb0_1 = 7.0
        z0_2 = 2.0 * (GAMMA * lam0 * z0_1) + 2.0 * phi[3]
        zb0_2 = 2.0 * (GAMMA * lam0 * zb0_1) + 2.0
        z0_3 = 5.0 * (GAMMA * lam0 * z0_2) + 5.0 * phi[LOWER]
        zb0_3 = 5.0 * (GAMMA * lam0 * zb0_2) + 5.0
        assert_traces(states[1], 0, z0_1, zb0_1)
        assert_traces(states[2], 0, z0_2, zb0_2)
        assert_traces(states[3], 0, z0_3, zb0_3)
        # The phi_1 coefficient after step 2 is gamma*lambda*rho_2*rho_1 exactly
        # (phi[0] and phi[3] only overlap on feature 7, so element 0 isolates it).
        expected_coeff = GAMMA * lam0 * 2.0 * 7.0
        np.testing.assert_allclose(
            float(states[2].head_traces[0][0].ravel()[0]) / phi[0][0],
            expected_coeff,
            rtol=1e-6,
        )

        # Demon 1: step 2 is masked, so the trace (and head) freeze; step 3
        # then composes off the *frozen* step-1 trace.
        z1_1 = 0.5 * phi[0]
        zb1_1 = 0.5
        z1_3 = 2.0 * (GAMMA * lam1 * z1_1) + 2.0 * phi[LOWER]
        zb1_3 = 2.0 * (GAMMA * lam1 * zb1_1) + 2.0
        assert_traces(states[1], 1, z1_1, zb1_1)
        assert_traces(states[2], 1, z1_1, zb1_1)  # frozen through the NaN step
        assert_traces(states[3], 1, z1_3, zb1_3)
        chex.assert_trees_all_equal(
            states[2].head_params.weights[1], states[1].head_params.weights[1]
        )
        chex.assert_trees_all_equal(
            states[2].head_params.biases[1], states[1].head_params.biases[1]
        )
        # The active demon did learn on that step.
        active_delta = float(
            jnp.linalg.norm(states[2].head_params.weights[0] - states[1].head_params.weights[0])
        )
        assert active_delta > 0.0
