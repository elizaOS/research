"""Loss-of-plasticity stress gate: permuted-target regression over 60 blocks.

Protocol (Slowly-Changing-Regression analog, Dohare et al., Nature 2024):
a fixed nonlinear teacher net ``y = v @ tanh(W (2 x_perm - 1))`` receives a
fresh random permutation of its 16 binary input bits at the start of each of
60 blocks (400 steps per block), forcing the learner to re-adapt every block.
Binary non-negative inputs plus a plain ReLU trunk (``leaky_relu_slope=0``,
no layer norm) and an aggressive fixed step-size (0.15) reproduce the classic
plasticity-loss mechanism: hidden units die over blocks and plain SGD cannot
revive them, so its within-block error drop (adaptation) decays.

Learners (identical trunk ``(30,)``, dense init, step-size 0.15):

- plain SGD: ``MLPLearner`` + fixed-step ``LMS``, no bounder
  (``track_neuron_utility=True`` for dormant-neuron logging)
- UPGD variant: ``UPGDLearner`` with utility-gated perturbation
  (``sigma=3e-3``) plus utility-based hidden-unit recycling (rate ``3e-3``)
- CBP variant: ``CBPMLPLearner`` with utility-based unit replacement
  (rate ``1e-3``, maturity 100)

Per-block adaptation = ``mean MSE(first 25 steps) - mean MSE(last 25 steps)``
(also its start-normalized version), averaged over 8 vmapped seeds. The gate
compares mean adaptation over the last 10 blocks against the first 10 blocks.

Measured ratios (last-10 / first-10 adaptation, seed-mean, ``jr.key(0)``,
8 seeds; abs = raw drop, rel = drop / start MSE):

    plain SGD:  abs ~ 0.30, rel ~ 0.29   (loses plasticity)
    UPGD:       abs ~ 1.16, rel ~ 1.11   (maintains adaptation)
    CBP:        abs ~ 1.05, rel ~ 1.01   (maintains adaptation)

Across three other seed sets the SGD ratios stayed <= 0.30 while UPGD/CBP
stayed >= 0.93, so the gate asserts UPGD/CBP >= 0.8 and a robust
between-learner inequality (SGD at least 0.3 below each) rather than a tight
absolute threshold on SGD.

Logged but not asserted: dormant-neuron fraction of the plain-SGD baseline
(rises from ~0.28 to ~0.95 of hidden units) and trunk feature effective rank
(collapses ~9.7 -> ~1.8 for SGD; stays ~13-14 for UPGD/CBP), both per block.
"""

from __future__ import annotations

import functools
import logging

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest
from jaxtyping import Array, Float

from alberta_framework.core.continual_backprop import (
    CBPMLPLearner,
    ContinualBackpropConfig,
)
from alberta_framework.core.learners import MLPLearner
from alberta_framework.core.optimizers import LMS
from alberta_framework.core.upgd import UPGDLearner

logger = logging.getLogger(__name__)

pytestmark = [pytest.mark.development, pytest.mark.slow]

# --- protocol constants ------------------------------------------------------
FEATURE_DIM = 16
TEACHER_HIDDEN = 32
HIDDEN_SIZES = (30,)
NUM_BLOCKS = 60
STEPS_PER_BLOCK = 400
WINDOW = 25  # steps averaged for block-start / block-end MSE
NUM_SEEDS = 8
STEP_SIZE = 0.15
RELU_SLOPE = 0.0  # plain ReLU: dead units cannot recover through the leak
PROBE_SIZE = 64  # probe batch for feature effective rank
DORMANT_THRESHOLD = 0.01

# Gate thresholds (see measured ratios in the module docstring).
MAINTAIN_RATIO = 0.8
SGD_GAP = 0.3


def _make_block_data(
    key: Array,
) -> tuple[
    Float[Array, "num_blocks steps_per_block feature_dim"],
    Float[Array, "num_blocks steps_per_block"],
]:
    """Per-seed permuted-target regression data.

    A fixed tanh teacher net is sampled once per seed; each block draws a
    fresh random permutation of the input bits and fresh Bernoulli inputs.
    """
    k_teacher, k_blocks = jr.split(key)
    k1, k2 = jr.split(k_teacher)
    w_teacher = jr.normal(k1, (TEACHER_HIDDEN, FEATURE_DIM)) / jnp.sqrt(FEATURE_DIM)
    v_teacher = 2.0 * jr.normal(k2, (TEACHER_HIDDEN,)) / jnp.sqrt(TEACHER_HIDDEN)

    def block(block_key: Array) -> tuple[Array, Array]:
        k_perm, k_x = jr.split(block_key)
        perm = jr.permutation(k_perm, FEATURE_DIM)
        x = jr.bernoulli(k_x, 0.5, (STEPS_PER_BLOCK, FEATURE_DIM)).astype(jnp.float32)
        # The teacher sees centered bits of the permuted input.
        y = jax.vmap(lambda xi: v_teacher @ jnp.tanh(w_teacher @ (2.0 * xi[perm] - 1.0)))(x)
        return x, y

    return jax.vmap(block)(jr.split(k_blocks, NUM_BLOCKS))


def _trunk_features(
    weights: tuple[Array, ...],
    biases: tuple[Array, ...],
    x_batch: Float[Array, "batch feature_dim"],
) -> Float[Array, "batch hidden"]:
    """Hidden-layer activations of the shared ReLU trunk for a probe batch."""
    h = x_batch
    for w, b in zip(weights, biases):
        h = h @ w.T + b
        h = jnp.where(h >= 0, h, RELU_SLOPE * h)
    return h


def _effective_rank(feats: Float[Array, "batch hidden"]) -> Float[Array, ""]:
    """Effective rank: exp of the entropy of the normalized singular values."""
    s = jnp.linalg.svd(feats, compute_uv=False)
    p = s / (jnp.sum(s) + 1e-12)
    return jnp.exp(-jnp.sum(p * jnp.log(p + 1e-12)))


def _run_protocol(init_fn, update_fn, trunk_fn, dormant_fn=None):
    """Build a jitted, seed-vmapped runner for one learner.

    ``update_fn(state, x, y) -> (state, error)`` adapts each learner's update
    signature; ``trunk_fn(state, probe)`` extracts hidden features for the
    effective-rank diagnostic; ``dormant_fn(state)`` optionally reports the
    dormant-neuron fraction. Returns per-block ``(start_mse, end_mse,
    effective_rank, dormant_fraction)`` arrays of shape (num_seeds, num_blocks).
    """

    def run(key: Array):
        k_data, k_init, k_probe = jr.split(key, 3)
        xs, ys = _make_block_data(k_data)
        probe = jr.bernoulli(k_probe, 0.5, (PROBE_SIZE, FEATURE_DIM)).astype(jnp.float32)
        state0 = init_fn(k_init)

        def block_step(state, block_xy):
            x, y = block_xy

            def step(s, xy):
                xi, yi = xy
                s2, err = update_fn(s, xi, yi)
                return s2, err**2

            state, sq_err = jax.lax.scan(step, state, (x, y))
            start_mse = jnp.mean(sq_err[:WINDOW])
            end_mse = jnp.mean(sq_err[-WINDOW:])
            eff_rank = _effective_rank(trunk_fn(state, probe))
            dormant = dormant_fn(state) if dormant_fn is not None else jnp.array(0.0)
            return state, (start_mse, end_mse, eff_rank, dormant)

        _, per_block = jax.lax.scan(block_step, state0, (xs, ys))
        return per_block

    return jax.jit(jax.vmap(run))


def _adaptation_ratios(
    start: Float[Array, "num_seeds num_blocks"],
    end: Float[Array, "num_seeds num_blocks"],
) -> tuple[float, float]:
    """(abs, rel) last-10/first-10 ratios of the seed-mean adaptation profile."""
    drop = jnp.mean(start - end, axis=0)
    rel_drop = jnp.mean((start - end) / (start + 1e-8), axis=0)
    ratio_abs = float(jnp.mean(drop[-10:]) / (jnp.mean(drop[:10]) + 1e-12))
    ratio_rel = float(jnp.mean(rel_drop[-10:]) / (jnp.mean(rel_drop[:10]) + 1e-12))
    return ratio_abs, ratio_rel


@functools.cache
def _experiment() -> dict[str, dict[str, Array]]:
    """Run all three learners once; shared by every test in this module."""
    sgd = MLPLearner(
        hidden_sizes=HIDDEN_SIZES,
        optimizer=LMS(step_size=STEP_SIZE),
        use_layer_norm=False,
        sparsity=0.0,
        leaky_relu_slope=RELU_SLOPE,
        track_neuron_utility=True,
    )
    upgd = UPGDLearner(
        n_heads=1,
        hidden_sizes=HIDDEN_SIZES,
        step_size=STEP_SIZE,
        sparsity=0.0,
        use_layer_norm=False,
        leaky_relu_slope=RELU_SLOPE,
        perturbation_sigma=3e-3,
        unit_replacement_rate=3e-3,
        unit_maturity_threshold=100,
    )
    cbp = CBPMLPLearner(
        hidden_sizes=HIDDEN_SIZES,
        cbp_config=ContinualBackpropConfig(
            decay_rate=0.99, replacement_rate=1e-3, maturity_threshold=100
        ),
        optimizer=LMS(step_size=STEP_SIZE),
        sparsity=0.0,
        use_layer_norm=False,
        leaky_relu_slope=RELU_SLOPE,
    )

    runners = {
        "sgd": _run_protocol(
            lambda k: sgd.init(FEATURE_DIM, k),
            lambda s, x, y: (lambda r: (r.state, r.error[0]))(sgd.update(s, x, y)),
            lambda s, p: _trunk_features(s.params.weights[:-1], s.params.biases[:-1], p),
            lambda s: jnp.mean((s.neuron_utility[0] < DORMANT_THRESHOLD).astype(jnp.float32)),
        ),
        "upgd": _run_protocol(
            lambda k: upgd.init(FEATURE_DIM, k),
            lambda s, x, y: (lambda r: (r.state, r.errors[0]))(
                upgd.update(s, x, jnp.reshape(y, (1,)))
            ),
            lambda s, p: _trunk_features(s.trunk_params.weights, s.trunk_params.biases, p),
        ),
        "cbp": _run_protocol(
            lambda k: cbp.init(FEATURE_DIM, k),
            lambda s, x, y: (lambda r: (r.state, jnp.squeeze(r.error)))(cbp.update(s, x, y)),
            lambda s, p: _trunk_features(
                s.multi_state.mlp_state.trunk_params.weights,
                s.multi_state.mlp_state.trunk_params.biases,
                p,
            ),
        ),
    }

    keys = jr.split(jr.key(0), NUM_SEEDS)
    results: dict[str, dict[str, Array]] = {}
    for name, runner in runners.items():
        start, end, eff_rank, dormant = jax.block_until_ready(runner(keys))
        results[name] = {
            "start_mse": start,
            "end_mse": end,
            "eff_rank": eff_rank,
            "dormant": dormant,
        }
        ratio_abs, ratio_rel = _adaptation_ratios(start, end)
        er = jnp.mean(eff_rank, axis=0)
        logger.info(
            "%s: adaptation ratio abs=%.3f rel=%.3f | eff-rank b0=%.1f b%d=%.1f b%d=%.1f",
            name,
            ratio_abs,
            ratio_rel,
            float(er[0]),
            NUM_BLOCKS // 2,
            float(er[NUM_BLOCKS // 2]),
            NUM_BLOCKS - 1,
            float(er[-1]),
        )
    dorm = jnp.mean(results["sgd"]["dormant"], axis=0)
    logger.info(
        "sgd dormant-neuron fraction (threshold %.2f): b0=%.2f b%d=%.2f b%d=%.2f",
        DORMANT_THRESHOLD,
        float(dorm[0]),
        NUM_BLOCKS // 2,
        float(dorm[NUM_BLOCKS // 2]),
        NUM_BLOCKS - 1,
        float(dorm[-1]),
    )
    return results


# --- gate assertions ---------------------------------------------------------


def test_all_metrics_finite() -> None:
    """NaN/inf anywhere would silently corrupt the ratio gate; fail loudly."""
    for name, res in _experiment().items():
        for metric, arr in res.items():
            assert bool(jnp.all(jnp.isfinite(arr))), f"{name}.{metric} has non-finite entries"
            assert arr.shape == (NUM_SEEDS, NUM_BLOCKS)


def test_every_learner_adapts_in_early_blocks() -> None:
    """All three learners must show a real error drop within the first blocks."""
    for name, res in _experiment().items():
        drop = jnp.mean(res["start_mse"] - res["end_mse"], axis=0)
        early = float(jnp.mean(drop[:10]))
        assert early > 0.5, f"{name}: early-block adaptation {early:.3f} too small"


def test_upgd_and_cbp_maintain_adaptation() -> None:
    """UPGD and CBP keep late-block adaptation >= 0.8x their early-block adaptation."""
    results = _experiment()
    for name in ("upgd", "cbp"):
        ratio_abs, ratio_rel = _adaptation_ratios(
            results[name]["start_mse"], results[name]["end_mse"]
        )
        assert ratio_abs >= MAINTAIN_RATIO, f"{name}: abs adaptation ratio {ratio_abs:.3f} < 0.8"
        assert ratio_rel >= MAINTAIN_RATIO, f"{name}: rel adaptation ratio {ratio_rel:.3f} < 0.8"


def test_plain_sgd_loses_plasticity_relative_to_upgd_and_cbp() -> None:
    """Plain SGD's adaptation ratio sits measurably below both plastic learners.

    Robust between-learner inequality (margin 0.3) rather than a tight
    absolute threshold on SGD: measured gaps are >= 0.6 across seed sets.
    """
    results = _experiment()
    sgd_abs, sgd_rel = _adaptation_ratios(
        results["sgd"]["start_mse"], results["sgd"]["end_mse"]
    )
    for name in ("upgd", "cbp"):
        other_abs, other_rel = _adaptation_ratios(
            results[name]["start_mse"], results[name]["end_mse"]
        )
        assert sgd_abs <= other_abs - SGD_GAP, (
            f"sgd abs ratio {sgd_abs:.3f} not measurably below {name} {other_abs:.3f}"
        )
        assert sgd_rel <= other_rel - SGD_GAP, (
            f"sgd rel ratio {sgd_rel:.3f} not measurably below {name} {other_rel:.3f}"
        )
