"""Slowly-changing-regression task mechanics and local learner adapters.

Reference: Dohare, S., Hernandez-Garcia, J. F., Lan, Q., Rahman, P.,
Mahmood, A. R. & Sutton, R. S. (2024). "Loss of plasticity in deep continual
learning". Nature 632, 768-774.

The task construction follows the shape described in the publication's
"Slowly changing regression" Methods section:

- The input is a binary vector of ``m + 1`` bits. The first ``f`` bits are the
  *flipping* (slowly changing) bits: after every ``T`` examples one of them is
  chosen uniformly at random and flipped. The next ``m - f`` bits are sampled
  i.i.d. uniform ``{0, 1}`` for every example. The last bit is a constant
  ``1`` input bias.
- The target is produced by a fixed random network with one hidden layer of
  linear threshold units (LTUs). Every target-network weight is sampled from
  ``{-1, +1}`` with equal probability. LTU ``i`` outputs 1 when
  ``w_i . x > theta_i`` with ``theta_i = (m + 1) * beta - S_i`` where ``S_i``
  is the number of ``-1`` input weights of unit ``i`` and ``beta = 0.7``.
- Paper scale: ``m = 20``, ``f = 15``, ``T = 10_000``, 100 LTUs, 3M examples,
  100 independent runs, squared error binned per 40,000 examples. The learner
  is a small MLP with a single hidden layer of 5 units.

This is not an exact replication implementation.  In particular, the JAX
stream/RNG and target-bias representation differ from the PyTorch reference;
Alberta's generic ``MLPLearner`` uses a different initializer and half-squared
error update convention; and the Alberta CBP and UPGD learners are local
extensions, not publication-source comparators.

The safe executable surface delegates to
``slowly_changing_regression_v2``.  That path adds a selected ordinary-BP arm
with ReLU Kaiming initialization and true-MSE gradients, requires a pre-run
source/runtime-bound plan, runs one method/seed per immutable shard, and
strictly merges descriptive, permanently nonpromoting artifacts.  Historical
v1-shaped outputs are neither generated nor upgraded by this module.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import Array, lax
from jaxtyping import Float

from alberta_framework.core.continual_backprop import (
    CBPMLPLearner,
    ContinualBackpropConfig,
)
from alberta_framework.core.learners import MLPLearner
from alberta_framework.core.optimizers import LMS
from alberta_framework.core.upgd import UPGDLearner

SCR_PAPER_REFERENCE = (
    "Dohare et al. 2024, 'Loss of plasticity in deep continual learning', "
    "Nature 632, 768-774 (slowly changing regression)"
)
SCR_PAPER_CLAIM = (
    "Ordinary backprop loses plasticity in the slowly-changing-regression study; "
    "the publication evaluates activation and step-size arms."
)

SCR_LEARNER_KINDS = ("sgd", "cbp", "upgd")

# =============================================================================
# Protocol configuration
# =============================================================================


@chex.dataclass(frozen=True)
class SlowlyChangingRegressionConfig:
    """Publication-shaped slowly-changing-regression task parameters.

    Attributes:
        num_bits: ``m`` -- number of non-bias input bits. The learner input
            has ``m + 1`` dimensions (constant 1 bias bit appended).
        num_flipping_bits: ``f`` -- number of slowly changing bits (the first
            ``f`` of the ``m`` input bits).
        flip_period: ``T`` -- one flipping bit is flipped every ``T`` examples.
        target_hidden_units: Number of LTUs in the fixed random target net.
        ltu_beta: ``beta`` in the LTU threshold ``(m + 1) * beta - S_i``.
        num_examples: Total number of examples in one run.
    """

    num_bits: int = 20
    num_flipping_bits: int = 15
    flip_period: int = 10_000
    target_hidden_units: int = 100
    ltu_beta: float = 0.7
    num_examples: int = 3_000_000

    @property
    def feature_dim(self) -> int:
        """Learner input dimension ``m + 1`` (with the constant bias bit)."""
        return self.num_bits + 1

    @property
    def num_segments(self) -> int:
        """Number of constant-slow-bit segments covering ``num_examples``."""
        return -(-self.num_examples // self.flip_period)

    def validate(self) -> None:
        """Raise ``ValueError`` for structurally invalid protocol settings."""
        if self.num_bits < 1:
            raise ValueError(f"num_bits must be >= 1, got {self.num_bits}")
        if not 1 <= self.num_flipping_bits <= self.num_bits:
            raise ValueError(
                "num_flipping_bits must be in [1, num_bits], got "
                f"{self.num_flipping_bits} with num_bits={self.num_bits}"
            )
        if self.flip_period < 1:
            raise ValueError(f"flip_period must be >= 1, got {self.flip_period}")
        if self.target_hidden_units < 1:
            raise ValueError(f"target_hidden_units must be >= 1, got {self.target_hidden_units}")
        if self.num_examples < 1:
            raise ValueError(f"num_examples must be >= 1, got {self.num_examples}")


@chex.dataclass(frozen=True)
class SlowlyChangingRegressionEnv:
    """One sampled slowly-changing-regression environment (target + schedule).

    Attributes:
        input_weights: Target-net input weights in ``{-1, +1}``, shape
            ``(feature_dim, target_hidden_units)``.
        thresholds: Per-LTU thresholds ``(m + 1) * beta - S_i``, shape
            ``(target_hidden_units,)``.
        output_weights: Target-net output weights in ``{-1, +1}``, shape
            ``(target_hidden_units,)``.
        slow_bits: Flipping-bit values per segment, shape
            ``(num_segments, num_flipping_bits)`` in ``{0.0, 1.0}``.
            Consecutive rows differ in exactly one bit.
        data_key: PRNG key for the i.i.d. fast bits (folded with the
            example index).
    """

    input_weights: Float[Array, "feature_dim target_hidden"]
    thresholds: Float[Array, " target_hidden"]
    output_weights: Float[Array, " target_hidden"]
    slow_bits: Float[Array, "num_segments num_flipping_bits"]
    data_key: Array


def make_scr_env(config: SlowlyChangingRegressionConfig, key: Array) -> SlowlyChangingRegressionEnv:
    """Sample a target network and full bit-flip schedule from *key*.

    Deterministic per key: the same key always produces the identical
    target network, flip schedule, and fast-bit stream.

    Args:
        config: Protocol parameters.
        key: JAX PRNG key (``jax.random.key`` style).

    Returns:
        A fully materialized :class:`SlowlyChangingRegressionEnv`.
    """
    config.validate()
    w_key, out_key, slow_key, flip_key, data_key = jr.split(key, 5)
    d = config.feature_dim
    n = config.target_hidden_units
    f = config.num_flipping_bits

    input_weights = jnp.where(jr.bernoulli(w_key, 0.5, (d, n)), 1.0, -1.0).astype(jnp.float32)
    negative_counts = jnp.sum(input_weights < 0.0, axis=0).astype(jnp.float32)
    thresholds = jnp.float32(d * config.ltu_beta) - negative_counts
    output_weights = jnp.where(jr.bernoulli(out_key, 0.5, (n,)), 1.0, -1.0).astype(jnp.float32)

    num_segments = config.num_segments
    initial_slow = jr.bernoulli(slow_key, 0.5, (f,)).astype(jnp.int32)
    flip_indices = jr.randint(flip_key, (num_segments - 1,), 0, f)
    flip_onehots = jax.nn.one_hot(flip_indices, f, dtype=jnp.int32)
    # Segment s (>0) equals the initial bits XOR the parity of flips so far.
    parity = jnp.cumsum(flip_onehots, axis=0) % 2
    later_segments = jnp.bitwise_xor(initial_slow[None, :], parity)
    slow_bits = jnp.concatenate([initial_slow[None, :], later_segments], axis=0)

    return SlowlyChangingRegressionEnv(  # type: ignore[call-arg]
        input_weights=input_weights,
        thresholds=thresholds,
        output_weights=output_weights,
        slow_bits=slow_bits.astype(jnp.float32),
        data_key=data_key,
    )


def scr_target_output(env: SlowlyChangingRegressionEnv, x: Array) -> Array:
    """Evaluate the fixed LTU target network on input *x* (shape ``(m+1,)``)."""
    ltu = (x @ env.input_weights > env.thresholds).astype(jnp.float32)
    return jnp.dot(ltu, env.output_weights)


def scr_example(
    env: SlowlyChangingRegressionEnv,
    config: SlowlyChangingRegressionConfig,
    t: Array | int,
) -> tuple[Array, Array]:
    """Generate example ``t`` of the stream: ``(x_t, y_t)``.

    ``x_t`` is ``[slow bits (f), fast bits (m - f), 1.0]`` and ``y_t`` is the
    target-network output. Pure function of ``(env, t)``.
    """
    t = jnp.asarray(t, dtype=jnp.int32)
    segment = t // config.flip_period
    slow = env.slow_bits[segment]
    n_fast = config.num_bits - config.num_flipping_bits
    fast = jr.bernoulli(jr.fold_in(env.data_key, t), 0.5, (n_fast,)).astype(jnp.float32)
    x = jnp.concatenate([slow, fast, jnp.ones((1,), dtype=jnp.float32)])
    return x, scr_target_output(env, x)


# =============================================================================
# Learners
# =============================================================================


@chex.dataclass(frozen=True)
class SCRLearnerParams:
    """Learner hyperparameters for the slowly-changing-regression benchmark.

    The width and selected step size follow one publication BP arm.  The CBP
    and UPGD fields configure Alberta-local extensions and must not be
    interpreted as publication hyperparameters.

    Attributes:
        hidden_units: Hidden-layer width of the learning network.
        step_size: SGD step-size (shared by all three learners).
        cbp_replacement_rate: CBP fraction of units replaced per step.
        cbp_maturity_threshold: CBP minimum unit age before replacement.
        cbp_decay_rate: CBP utility EMA decay.
        upgd_sigma: UPGD perturbation noise scale.
        upgd_utility_decay: UPGD utility EMA decay.
        upgd_beta: UPGD ``(1 - u)`` gating exponent.
    """

    hidden_units: int = 5
    step_size: float = 0.01
    cbp_replacement_rate: float = 1e-4
    cbp_maturity_threshold: int = 100
    cbp_decay_rate: float = 0.99
    upgd_sigma: float = 1e-3
    upgd_utility_decay: float = 0.995
    upgd_beta: float = 2.0


def build_scr_learner(
    kind: str, params: SCRLearnerParams
) -> MLPLearner | CBPMLPLearner | UPGDLearner:
    """Construct one of the three benchmark learners.

    These are Alberta learner adapters.  Dense initialization, no layer norm,
    and ReLU match broad architectural choices, but the ordinary ``sgd`` path
    does not use the publication's Kaiming initialization or true-MSE factor.
    The strict v2 runner uses its separate ``publication_bp_relu_sgd`` path
    for that selected comparator arm.

    Args:
        kind: One of ``"sgd"``, ``"cbp"``, ``"upgd"``.
        params: Learner hyperparameters.

    Returns:
        The configured learner instance.
    """
    hidden = (params.hidden_units,)
    common: dict[str, Any] = {
        "hidden_sizes": hidden,
        "sparsity": 0.0,
        "leaky_relu_slope": 0.0,
        "use_layer_norm": False,
    }
    if kind == "sgd":
        return MLPLearner(optimizer=LMS(step_size=params.step_size), **common)
    if kind == "cbp":
        return CBPMLPLearner(
            cbp_config=ContinualBackpropConfig(  # type: ignore[call-arg]
                decay_rate=params.cbp_decay_rate,
                replacement_rate=params.cbp_replacement_rate,
                maturity_threshold=params.cbp_maturity_threshold,
                enabled=True,
            ),
            optimizer=LMS(step_size=params.step_size),
            **common,
        )
    if kind == "upgd":
        return UPGDLearner(
            n_heads=1,
            step_size=params.step_size,
            utility_decay=params.upgd_utility_decay,
            perturbation_sigma=params.upgd_sigma,
            perturbation_beta=params.upgd_beta,
            track_unit_utilities=False,
            track_gradient_history=False,
            **common,
        )
    raise ValueError(f"unknown learner kind {kind!r}; expected one of {SCR_LEARNER_KINDS}")


def _make_update_fn(
    kind: str, learner: MLPLearner | CBPMLPLearner | UPGDLearner
) -> Callable[[Any, Array, Array], tuple[Any, Array]]:
    """Adapter: ``(state, x, y) -> (new_state, squared_error)`` per kind."""
    if kind == "sgd":

        def update_sgd(state: Any, x: Array, y: Array) -> tuple[Any, Array]:
            result = learner.update(state, x, y)
            err = jnp.squeeze(result.error)
            return result.state, err * err

        return update_sgd
    if kind == "cbp":

        def update_cbp(state: Any, x: Array, y: Array) -> tuple[Any, Array]:
            result = learner.update(state, x, y)
            return result.state, result.error * result.error

        return update_cbp
    if kind == "upgd":

        def update_upgd(state: Any, x: Array, y: Array) -> tuple[Any, Array]:
            result = learner.update(state, x, jnp.reshape(y, (1,)))
            err = result.errors[0]
            return result.state, err * err

        return update_upgd
    raise ValueError(f"unknown learner kind {kind!r}; expected one of {SCR_LEARNER_KINDS}")


# =============================================================================
# Runner
# =============================================================================


def run_scr_binned_errors(
    kind: str,
    config: SlowlyChangingRegressionConfig,
    params: SCRLearnerParams,
    num_runs: int,
    seed: int,
    bin_size: int,
) -> Float[Array, "num_runs num_bins"]:
    """Run *num_runs* independent runs of one learner, all inside one vmap.

    Each run draws its own target network, flip schedule, fast-bit stream,
    and learner initialization from a per-run key split off ``jr.key(seed)``.
    The per-example squared error is averaged per consecutive ``bin_size``
    examples.  This compatibility helper uses Alberta-local learner and RNG
    semantics and is not a publication-exact or artifact-producing path.

    Args:
        kind: One of ``"sgd"``, ``"cbp"``, ``"upgd"``.
        config: Protocol parameters (``num_examples`` must be a multiple of
            ``bin_size``).
        params: Learner hyperparameters.
        num_runs: Number of independent runs (paper: 100).
        seed: Base seed for the per-run key split.
        bin_size: Examples per error bin (paper: 40,000).

    Returns:
        Binned mean squared errors, shape ``(num_runs, num_bins)``.
    """
    config.validate()
    if num_runs < 1:
        raise ValueError(f"num_runs must be >= 1, got {num_runs}")
    if bin_size < 1:
        raise ValueError(f"bin_size must be >= 1, got {bin_size}")
    if config.num_examples % bin_size != 0:
        raise ValueError(
            f"num_examples ({config.num_examples}) must be a multiple of bin_size ({bin_size})"
        )
    num_bins = config.num_examples // bin_size
    learner = build_scr_learner(kind, params)
    update_fn = _make_update_fn(kind, learner)
    feature_dim = config.feature_dim

    def run_one(key: Array) -> Array:
        env_key, init_key = jr.split(key)
        env = make_scr_env(config, env_key)
        state0 = learner.init(feature_dim, init_key)

        def bin_step(state: Any, bin_idx: Array) -> tuple[Any, Array]:
            start = bin_idx * bin_size

            def body(i: Array, carry: tuple[Any, Array]) -> tuple[Any, Array]:
                st, acc = carry
                x, y = scr_example(env, config, start + i)
                st, sq = update_fn(st, x, y)
                return st, acc + sq

            state, total = lax.fori_loop(0, bin_size, body, (state, jnp.float32(0.0)))
            return state, total / jnp.float32(bin_size)

        _, binned = lax.scan(bin_step, state0, jnp.arange(num_bins, dtype=jnp.int32))
        return binned

    keys = jr.split(jr.key(seed), num_runs)
    return cast(Array, jax.jit(jax.vmap(run_one))(keys))


# =============================================================================
# Descriptive summaries
# =============================================================================


def summarize_scr_curve(binned: Array) -> dict[str, list[float]]:
    """Mean/std-across-runs summary of a ``(num_runs, num_bins)`` matrix."""
    arr = jnp.asarray(binned, dtype=jnp.float32)
    mean = jnp.mean(arr, axis=0)
    std = jnp.std(arr, axis=0)
    stderr = std / jnp.sqrt(jnp.float32(arr.shape[0]))
    return {
        "bin_mean": [round(float(v), 6) for v in mean],
        "bin_std": [round(float(v), 6) for v in std],
        "bin_stderr": [round(float(v), 6) for v in stderr],
    }


def describe_scr_curve_windows(
    bin_means: Mapping[str, Sequence[float]],
    early_window: tuple[int, int] = (2, 7),
    late_bins: int = 5,
) -> dict[str, Any]:
    """Describe early and late windows without applying a pass/fail gate.

    ``early`` is the mean binned error over ``early_window`` (default bins
    2..6, i.e. examples 80k-280k at the paper bin size, past the initial
    transient), ``late`` is the mean over the final ``late_bins`` bins.

    This helper intentionally reports no threshold and no boolean conclusion.
    The former v1 helper used thresholds calibrated after inspecting a reduced
    development run; those thresholds are not a valid promotion gate.

    Args:
        bin_means: Mapping from learner kind to mean-across-runs bin curve.
        early_window: ``(start, stop)`` bin slice for the early reference.
        late_bins: Number of final bins for the late measurement.
    Returns:
        Dict with only per-learner early/late/ratio descriptive values.
    """
    lo, hi = early_window
    measures: dict[str, dict[str, float]] = {}
    for kind, curve in bin_means.items():
        n = len(curve)
        if n < max(hi, late_bins) or lo >= hi:
            raise ValueError(
                f"curve for {kind!r} has {n} bins; needs >= {max(hi, late_bins)} "
                f"for early_window={early_window}, late_bins={late_bins}"
            )
        early = float(sum(curve[lo:hi]) / (hi - lo))
        late = float(sum(curve[n - late_bins :]) / late_bins)
        measures[kind] = {
            "early_error": round(early, 6),
            "late_error": round(late, 6),
            "late_over_early": round(late / early, 6),
        }

    return {
        "early_window_bins": [lo, hi],
        "late_bins": late_bins,
        "measures": measures,
        "interpretation": "descriptive_only_no_threshold_or_claim",
    }


# =============================================================================
# Strict v2 CLI delegation
# =============================================================================


def main(argv: Sequence[str] | None = None) -> Any:
    """Delegate to the strict, sharded, permanently nonpromoting v2 CLI."""

    from alberta_framework.benchmarks.slowly_changing_regression_v2 import (
        main as strict_v2_main,
    )

    return strict_v2_main(argv)


if __name__ == "__main__":
    main()
