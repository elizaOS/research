"""Forward-view / backward-view TD(lambda) equivalence harness.

Validates the repository's central architectural claim (see
``core/independent_demon_horde.py``): with one independent network per
demon, full per-parameter accumulating eligibility traces with
``gamma * lamda > 0`` are forward-view-correct for EVERY layer, whereas
a shared trunk that folds per-head error into the trunk cotangent before
trace accumulation is not.

Method (Sutton & Barto Ch. 12, offline equivalence): on a fixed short
trajectory with FROZEN weights (``LMS(step_size=0.0)`` so the horde's
own update path runs but applies zero deltas), the accumulated
backward-view update ``sum_t delta_t * e_t`` must equal the offline
forward-view lambda-return update ``sum_t (G_t^lambda - V(s_t)) *
grad V(s_t)``, where the interim lambda-return ``G_t^lambda`` bootstraps
with the frozen ``V`` at the trajectory horizon. The identity is exact
(up to float32 rounding) for any architecture at frozen weights, and
approximately holds in direction for small applied step-sizes.

The pathological case: a hand-rolled 2-layer shared-trunk replica that
accumulates error-folded trunk gradients into a trace (exactly what
``MultiHeadMLPLearner``'s VJP path would do if its guard were removed)
breaks the equivalence badly for ``gamma * lamda > 0``. Measured
mismatch on the fixed trajectory below (gamma=0.9, lamda=0.9, T=20):
relative Frobenius error 2.99 (~300% of the forward-view update's norm),
versus ~1e-7 for the per-head backward view and exactly 0 for the same
replica at ``gamma * lamda = 0``.
"""

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from alberta_framework import DemonType, GVFSpec, create_horde_spec
from alberta_framework.core.independent_demon_horde import (
    IndependentDemonHorde,
    _forward_mlp,
)
from alberta_framework.core.multi_head_learner import MultiHeadMLPLearner
from alberta_framework.core.optimizers import LMS

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_demons(gammas_lamdas: list[tuple[float, float]]) -> list[GVFSpec]:
    """Create prediction demons with the given (gamma, lamda) pairs."""
    return [
        GVFSpec(
            name=f"d{i}",
            demon_type=DemonType.PREDICTION,
            gamma=g,
            lamda=lam,
            cumulant_index=i,
        )
        for i, (g, lam) in enumerate(gammas_lamdas)
    ]


def _lambda_returns(
    cumulants: jnp.ndarray, values: jnp.ndarray, gamma: float, lamda: float
) -> jnp.ndarray:
    """Interim forward-view lambda-returns for a truncated trajectory.

    Computed by the standard recursion (independent of TD errors and
    traces, so the comparison against the backward view is meaningful):

        G_T = V(s_T)                                (bootstrap at horizon)
        G_t = c_t + gamma * ((1 - lamda) * V(s_{t+1}) + lamda * G_{t+1})

    Args:
        cumulants: Per-step cumulants, shape ``(T,)``.
        values: Frozen-weight values ``V(s_0..s_T)``, shape ``(T + 1,)``.
        gamma: Discount.
        lamda: Trace decay.

    Returns:
        Lambda-returns ``G_0..G_{T-1}``, shape ``(T,)``.
    """
    num_steps = cumulants.shape[0]
    g = values[num_steps]
    out: list[jnp.ndarray] = [jnp.zeros(())] * num_steps
    for t in reversed(range(num_steps)):
        g = cumulants[t] + gamma * ((1.0 - lamda) * values[t + 1] + lamda * g)
        out[t] = g
    return jnp.stack(out)


def _run_horde_and_accumulate_backward(
    horde: IndependentDemonHorde,
    state,
    obs_seq: jnp.ndarray,
    cumulants: jnp.ndarray,
) -> tuple[list[list[jnp.ndarray]], jnp.ndarray, object]:
    """Run the horde over the trajectory, accumulating ``sum_t delta_t * e_t``.

    Returns ``(per_demon_accumulators, td_errors, final_state)`` where
    ``per_demon_accumulators[i]`` is a list matching the demon's
    interleaved trace layout ``(w0, b0, w1, b1, ...)``.
    """
    num_steps = cumulants.shape[0]
    n_demons = horde.n_demons
    acc: list[list[jnp.ndarray]] = [
        [jnp.zeros_like(t) for t in state.demon_states[i].traces]
        for i in range(n_demons)
    ]
    deltas: list[jnp.ndarray] = []
    for t in range(num_steps):
        result = horde.update(state, obs_seq[t], cumulants[t], obs_seq[t + 1])
        state = result.state
        deltas.append(result.td_errors)
        for i in range(n_demons):
            traces = state.demon_states[i].traces
            acc[i] = [
                a + result.td_errors[i] * tr
                for a, tr in zip(acc[i], traces, strict=True)
            ]
    return acc, jnp.stack(deltas), state


# ---------------------------------------------------------------------------
# (a) IndependentDemonHorde: backward view == forward view (frozen weights)
# ---------------------------------------------------------------------------


class TestIndependentHordeForwardViewEquivalence:
    """Accumulated TD(lambda) updates match offline lambda-return updates."""

    def test_linear_demons_frozen_weights_exact(self) -> None:
        """Linear demons, frozen weights: the equivalence is exact.

        Uses ``LMS(step_size=0.0)`` so the horde's real update path runs
        (traces accumulate, TD errors are computed) but parameters stay
        frozen — the offline setting where the identity holds exactly.
        """
        num_steps, feature_dim = 20, 6
        gl_pairs = [(0.9, 0.95), (0.8, 0.5), (0.9, 0.0), (0.0, 0.0)]
        demons = _make_demons(gl_pairs)
        spec = create_horde_spec(demons)
        horde = IndependentDemonHorde(
            horde_spec=spec,
            hidden_sizes=(),
            optimizer=LMS(step_size=0.0),
            sparsity=0.0,
        )
        init_state = horde.init(feature_dim, jr.key(0))

        k_obs, k_cum = jr.split(jr.key(1))
        obs_seq = jr.normal(k_obs, (num_steps + 1, feature_dim))
        cumulants = jr.normal(k_cum, (num_steps, len(demons)))

        acc, deltas, final_state = _run_horde_and_accumulate_backward(
            horde, init_state, obs_seq, cumulants
        )

        # Sanity: step_size=0.0 really froze the parameters.
        for i in range(len(demons)):
            assert jnp.array_equal(
                final_state.demon_states[i].params.weights[0],
                init_state.demon_states[i].params.weights[0],
            )

        # Frozen-weight values V(s_0..s_T) per demon, shape (T+1, n_demons).
        values = jnp.stack(
            [horde.predict(init_state, obs_seq[t]) for t in range(num_steps + 1)]
        )

        # Internal consistency: the horde's TD errors are the frozen-weight
        # one-step errors c_t + gamma * V(s_{t+1}) - V(s_t).
        expected_deltas = cumulants + spec.gammas[None, :] * values[1:] - values[:-1]
        chex.assert_trees_all_close(deltas, expected_deltas, rtol=1e-4, atol=1e-5)

        # Forward-view lambda-return update, computed independently.
        for i, (gamma, lamda) in enumerate(gl_pairs):
            lam_returns = _lambda_returns(cumulants[:, i], values[:, i], gamma, lamda)
            coeff = lam_returns - values[:-1, i]
            fwd_w = jnp.einsum("t,td->d", coeff, obs_seq[:-1])[None, :]
            fwd_b = jnp.sum(coeff)[None]

            # Non-trivial comparison.
            assert float(jnp.linalg.norm(fwd_w)) > 1e-2

            chex.assert_trees_all_close(acc[i][0], fwd_w, rtol=1e-3, atol=1e-3)
            chex.assert_trees_all_close(acc[i][1], fwd_b, rtol=1e-3, atol=1e-3)

    def test_mlp_demons_frozen_weights_every_layer(self) -> None:
        """Independent MLP demons: equivalence holds for trunk AND head layers.

        This is the architectural point of ``IndependentDemonHorde``:
        with no parameter sharing, full per-parameter traces are
        forward-view-correct for every layer, not just the output layer.
        """
        num_steps, feature_dim = 15, 5
        slope, use_ln = 0.01, True
        gl_pairs = [(0.9, 0.9), (0.7, 0.4)]
        demons = _make_demons(gl_pairs)
        spec = create_horde_spec(demons)
        horde = IndependentDemonHorde(
            horde_spec=spec,
            hidden_sizes=(8,),
            optimizer=LMS(step_size=0.0),
            sparsity=0.0,
            leaky_relu_slope=slope,
            use_layer_norm=use_ln,
        )
        init_state = horde.init(feature_dim, jr.key(2))

        k_obs, k_cum = jr.split(jr.key(3))
        obs_seq = jr.normal(k_obs, (num_steps + 1, feature_dim))
        cumulants = jr.normal(k_cum, (num_steps, len(demons)))

        acc, _, _ = _run_horde_and_accumulate_backward(
            horde, init_state, obs_seq, cumulants
        )

        values = jnp.stack(
            [horde.predict(init_state, obs_seq[t]) for t in range(num_steps + 1)]
        )

        for i, (gamma, lamda) in enumerate(gl_pairs):
            params = init_state.demon_states[i].params

            def pred_fn(weights, biases, obs):
                return _forward_mlp(weights, biases, obs, slope, use_ln)

            grad_fn = jax.grad(pred_fn, argnums=(0, 1))

            lam_returns = _lambda_returns(cumulants[:, i], values[:, i], gamma, lamda)
            coeff = lam_returns - values[:-1, i]

            # Forward-view update per parameter leaf (interleaved layout).
            fwd = [jnp.zeros_like(t) for t in acc[i]]
            n_layers = len(params.weights)
            for t in range(num_steps):
                w_grads, b_grads = grad_fn(params.weights, params.biases, obs_seq[t])
                for layer in range(n_layers):
                    fwd[2 * layer] = fwd[2 * layer] + coeff[t] * w_grads[layer]
                    fwd[2 * layer + 1] = fwd[2 * layer + 1] + coeff[t] * b_grads[layer]

            # Every layer (trunk weight/bias, head weight/bias) matches.
            for leaf_backward, leaf_forward in zip(acc[i], fwd, strict=True):
                chex.assert_trees_all_close(
                    leaf_backward, leaf_forward, rtol=2e-3, atol=1e-3
                )

    def test_tiny_step_size_applied_updates_match_direction(self) -> None:
        """With a small applied step-size, total weight change ~ alpha * forward view."""
        num_steps, feature_dim = 20, 6
        alpha = 1e-3
        gl_pairs = [(0.9, 0.95), (0.8, 0.5)]
        demons = _make_demons(gl_pairs)
        spec = create_horde_spec(demons)
        horde = IndependentDemonHorde(
            horde_spec=spec,
            hidden_sizes=(),
            optimizer=LMS(step_size=alpha),
            sparsity=0.0,
        )
        init_state = horde.init(feature_dim, jr.key(4))

        k_obs, k_cum = jr.split(jr.key(5))
        obs_seq = jr.normal(k_obs, (num_steps + 1, feature_dim))
        cumulants = jr.normal(k_cum, (num_steps, len(demons)))

        state = init_state
        for t in range(num_steps):
            state = horde.update(state, obs_seq[t], cumulants[t], obs_seq[t + 1]).state

        values = jnp.stack(
            [horde.predict(init_state, obs_seq[t]) for t in range(num_steps + 1)]
        )

        for i, (gamma, lamda) in enumerate(gl_pairs):
            lam_returns = _lambda_returns(cumulants[:, i], values[:, i], gamma, lamda)
            coeff = lam_returns - values[:-1, i]
            fwd_w = jnp.einsum("t,td->d", coeff, obs_seq[:-1])
            fwd_b = jnp.sum(coeff)

            applied = jnp.concatenate(
                [
                    (
                        state.demon_states[i].params.weights[0]
                        - init_state.demon_states[i].params.weights[0]
                    ).ravel(),
                    (
                        state.demon_states[i].params.biases[0]
                        - init_state.demon_states[i].params.biases[0]
                    ).ravel(),
                ]
            )
            expected = alpha * jnp.concatenate([fwd_w.ravel(), fwd_b[None]])

            cos = float(
                jnp.dot(applied, expected)
                / (jnp.linalg.norm(applied) * jnp.linalg.norm(expected))
            )
            ratio = float(jnp.linalg.norm(applied) / jnp.linalg.norm(expected))
            assert cos > 0.99, f"demon {i}: cosine {cos:.5f} <= 0.99"
            assert 0.9 < ratio < 1.1, f"demon {i}: norm ratio {ratio:.4f}"


# ---------------------------------------------------------------------------
# (b) Pathological shared trunk: error-folded trunk traces break equivalence
# ---------------------------------------------------------------------------


def _shared_trunk_setup(
    num_steps: int, feature_dim: int, hidden_dim: int, n_heads: int
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Fixed frozen 2-layer shared-trunk toy: trunk W1, per-head rows w2."""
    k_w1, k_w2, k_obs, k_cum = jr.split(jr.key(6), 4)
    w1 = jr.normal(k_w1, (hidden_dim, feature_dim)) / jnp.sqrt(feature_dim)
    w2 = jr.normal(k_w2, (n_heads, hidden_dim)) / jnp.sqrt(hidden_dim)
    obs_seq = jr.normal(k_obs, (num_steps + 1, feature_dim))
    cumulants = jr.normal(k_cum, (num_steps, n_heads))
    return w1, w2, obs_seq, cumulants


def _trunk_updates(
    w1: jnp.ndarray,
    w2: jnp.ndarray,
    obs_seq: jnp.ndarray,
    cumulants: jnp.ndarray,
    gamma: float,
    lamda: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute three accumulated trunk updates on the frozen toy network.

    Returns ``(pathological, correct_backward, forward_view)`` where:

    - ``pathological`` replicates ``MultiHeadMLPLearner``'s trunk-trace
      computation with the ``gamma * lamda > 0`` guard removed: the VJP
      cotangent folds per-head TD errors into the trunk gradient BEFORE
      trace accumulation (``g_t = sum_h delta_ht * grad V_h``), the trace
      decays as ``e_t = gl * e_{t-1} + g_t``, and the trace is applied
      directly (no error multiply — error is already in the gradient).
    - ``correct_backward`` keeps per-head traces of pure prediction
      gradients and applies ``sum_h sum_t delta_ht * e_ht``.
    - ``forward_view`` is the offline lambda-return update
      ``sum_h sum_t (G_ht^lambda - V_h(s_t)) * grad V_h(s_t)``.
    """
    num_steps = cumulants.shape[0]
    n_heads = w2.shape[0]
    gl = gamma * lamda

    hidden = obs_seq @ w1.T  # (T+1, hidden)  linear trunk
    values = hidden @ w2.T  # (T+1, n_heads)
    deltas = cumulants + gamma * values[1:] - values[:-1]  # (T, n_heads)

    # grad_{W1} V_h(x_t) = outer(w2[h], x_t) for the linear trunk.
    pathological = jnp.zeros_like(w1)
    folded_trace = jnp.zeros_like(w1)
    head_traces = [jnp.zeros_like(w1) for _ in range(n_heads)]
    correct_backward = jnp.zeros_like(w1)
    for t in range(num_steps):
        folded_grad = jnp.outer(deltas[t] @ w2, obs_seq[t])
        folded_trace = gl * folded_trace + folded_grad
        pathological = pathological + folded_trace
        for h in range(n_heads):
            head_traces[h] = gl * head_traces[h] + jnp.outer(w2[h], obs_seq[t])
            correct_backward = correct_backward + deltas[t, h] * head_traces[h]

    forward_view = jnp.zeros_like(w1)
    for h in range(n_heads):
        lam_returns = _lambda_returns(cumulants[:, h], values[:, h], gamma, lamda)
        coeff = lam_returns - values[:-1, h]
        forward_view = forward_view + jnp.einsum(
            "t,td->d", coeff, obs_seq[:-1]
        ) * w2[h][:, None]

    return pathological, correct_backward, forward_view


class TestSharedTrunkTracesBreakEquivalence:
    """The forbidden shared-trunk trace scheme violates forward-view equivalence."""

    def test_error_folded_trunk_traces_mismatch_forward_view(self) -> None:
        """gamma*lamda > 0: the error-folded trunk trace diverges badly.

        Measured on this fixed trajectory: the pathological update's
        relative Frobenius error vs the forward view is 2.99 (~300%),
        while the per-head backward view matches to float32 precision
        (relative error ~1.4e-7).
        The folded trace inflates each self-term ``delta_t * grad_t`` by
        the future decay sum ``(1 - gl^{T-t}) / (1 - gl)`` and drops all
        cross-terms ``delta_k * grad_j`` (k > j) that carry temporal
        credit backwards — exactly the bias the guard exists to prevent.
        """
        gamma, lamda = 0.9, 0.9
        w1, w2, obs_seq, cumulants = _shared_trunk_setup(20, 5, 4, 2)

        pathological, correct_backward, forward_view = _trunk_updates(
            w1, w2, obs_seq, cumulants, gamma, lamda
        )

        fwd_norm = float(jnp.linalg.norm(forward_view))
        assert fwd_norm > 1e-2

        # The CORRECT per-head backward view matches the forward view.
        chex.assert_trees_all_close(
            correct_backward, forward_view, rtol=1e-3, atol=1e-3
        )

        # The pathological folded-trace update does NOT.
        rel_err = float(jnp.linalg.norm(pathological - forward_view)) / fwd_norm
        assert rel_err > 0.25, (
            f"expected the error-folded trunk trace to break forward-view "
            f"equivalence, but relative error was only {rel_err:.4f}"
        )

    def test_folded_traces_are_correct_at_gamma_lamda_zero(self) -> None:
        """gamma*lamda = 0: the same folded scheme IS forward-view-correct.

        This is precisely why ``MultiHeadMLPLearner`` forces trunk
        ``gamma * lamda = 0`` instead of forbidding shared trunks
        entirely: with per-step trace resets the folded error-gradient
        product is the correct (lambda = 0) update.
        """
        gamma = 0.9
        w1, w2, obs_seq, cumulants = _shared_trunk_setup(20, 5, 4, 2)

        pathological, correct_backward, forward_view = _trunk_updates(
            w1, w2, obs_seq, cumulants, gamma, 0.0
        )

        chex.assert_trees_all_close(pathological, forward_view, rtol=1e-3, atol=1e-3)
        chex.assert_trees_all_close(
            correct_backward, forward_view, rtol=1e-3, atol=1e-3
        )

    def test_multi_head_learner_guards_against_trunk_traces(self) -> None:
        """The library refuses the pathological configuration outright."""
        with pytest.raises(ValueError, match=r"Trunk gamma\*lamda must be 0"):
            MultiHeadMLPLearner(n_heads=2, hidden_sizes=(8,), gamma=0.9, lamda=0.9)
