"""Streams that exercise the Alberta Plan Step 1 supervised-learning spec.

The Alberta Plan Step 1 specifies a non-stationary supervised problem in which
the desired output is

    y*_t = w*_t . x_t + b*_t + eta_t

with ``eta_t`` an independent mean-zero noise signal. The problem is non-
stationary if ``w*_t`` or ``b*_t`` change over time, OR if the distribution of
``x_t`` changes over time. This module provides two streams that cover the
two non-stationarity cases:

* :class:`AlbertaPlanStep1Stream` — the canonical Step 1 task: ``w*_t`` and
  ``b*_t`` follow Gaussian random walks and ``eta_t`` is included.
* :class:`XDistShiftStream` — fixes the target function and shifts only the
  input distribution (per-feature scales redrawn at fixed intervals).

Both streams follow the :class:`~alberta_framework.streams.base.ScanStream`
protocol and are JIT-friendly (no Python control flow on traced values).

Reference: Sutton et al., "The Alberta Plan for AI Research", Step 1.
"""

import dataclasses
import functools
import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from typing import Any, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import Array
from jaxtyping import Bool, Float, Int, PRNGKeyArray, UInt

from alberta_framework.core.normalizers import (
    _checked_lifetime_words_increment,
    _lifetime_counter_valid,
    _saturating_int32_counter_increment,
)
from alberta_framework.core.types import TimeStep

ALBERTA_PLAN_STEP1_CONFIG_SCHEMA = "alberta.plan-step1-stream.config.v2"
ALBERTA_PLAN_STEP1_STATE_SCHEMA = "alberta.plan-step1-stream.state.v2"
XDIST_SHIFT_CONFIG_SCHEMA = "alberta.xdist-shift-stream.config.v2"
XDIST_SHIFT_STATE_SCHEMA = "alberta.xdist-shift-stream.state.v2"
STEP1_STREAM_RESOURCE_SCHEMA = "alberta.step1-stream.resource-budget.v2"
STEP1_STREAM_CLOCK_NBYTES = 12
STEP1_STREAM_CLOCK_DELTA_NBYTES = 8

_INT32_MAX = 2**31 - 1
_FLOAT32_MAX = 3.4028234663852886e38

__all__ = [
    "ALBERTA_PLAN_STEP1_CONFIG_SCHEMA",
    "ALBERTA_PLAN_STEP1_STATE_SCHEMA",
    "STEP1_STREAM_CLOCK_DELTA_NBYTES",
    "STEP1_STREAM_CLOCK_NBYTES",
    "STEP1_STREAM_RESOURCE_SCHEMA",
    "XDIST_SHIFT_CONFIG_SCHEMA",
    "XDIST_SHIFT_STATE_SCHEMA",
    "AlbertaPlanStep1State",
    "AlbertaPlanStep1StepResult",
    "AlbertaPlanStep1Stream",
    "Step1StreamResourceBudget",
    "XDistShiftState",
    "XDistShiftStepResult",
    "XDistShiftStream",
    "measure_step1_stream_state_nbytes",
    "migrate_legacy_alberta_plan_step1_state",
    "migrate_legacy_xdist_shift_state",
    "step1_stream_clock_nbytes",
]


def _require_array(
    value: Any,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: jnp.dtype,
) -> Array:
    array = jnp.asarray(value)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if array.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {array.dtype}")
    return array


def _require_prng_key(key: Array, *, name: str) -> None:
    key_data = jr.key_data(key)
    if key_data.shape != (2,) or key_data.dtype != jnp.dtype(jnp.uint32):
        raise TypeError(f"{name} must be a scalar JAX PRNG key")


def _tree_floating_arrays_finite(value: Any) -> Bool[Array, ""]:
    valid = jnp.asarray(True, dtype=jnp.bool_)
    for leaf in jax.tree.leaves(value):
        if hasattr(leaf, "dtype") and jnp.issubdtype(leaf.dtype, jnp.inexact):
            valid = valid & jnp.all(jnp.isfinite(leaf))
    return valid


def _require_positive_int32(value: Any, *, name: str) -> None:
    if type(value) is not int or not 0 < value <= _INT32_MAX:
        raise ValueError(f"{name} must be an exact integer in [1, {_INT32_MAX}]")


def _require_finite_real(
    value: Any,
    *,
    name: str,
    nonnegative: bool = False,
) -> None:
    valid = (
        not isinstance(value, bool)
        and isinstance(value, Real)
        and math.isfinite(float(value))
        and abs(float(value)) <= _FLOAT32_MAX
    )
    if nonnegative:
        valid = valid and float(value) >= 0.0
    if not valid:
        qualifier = " non-negative" if nonnegative else ""
        raise ValueError(f"{name} must be a finite{qualifier} real")


def _step_input_valid(idx: Array) -> Bool[Array, ""]:
    idx_array = jnp.asarray(idx)
    if idx_array.shape != ():
        raise ValueError(f"idx must be scalar, got shape {idx_array.shape}")
    if not (
        jnp.issubdtype(idx_array.dtype, jnp.integer)
        or jnp.issubdtype(idx_array.dtype, jnp.floating)
    ):
        raise TypeError("idx must have an integer or floating dtype")
    return jnp.isfinite(idx_array)


@functools.partial(jax.jit, static_argnums=(1,))
def _lifetime_words_remainder(words: Array, divisor: int) -> UInt[Array, ""]:
    """Compute an exact uint64-by-positive-int32 remainder without x64."""

    _require_array(
        words,
        name="Step 1 stream step_words",
        shape=(2,),
        dtype=jnp.dtype(jnp.uint32),
    )
    if type(divisor) is not int or not 0 < divisor <= _INT32_MAX:
        raise ValueError("schedule divisor must be a positive int32 integer")
    divisor_array = jnp.asarray(divisor, dtype=jnp.uint32)

    def body(index: Array, remainder: Array) -> Array:
        in_high = index < 32
        bit_index = jnp.asarray(31, dtype=jnp.int32) - jnp.mod(index, 32)
        source = jnp.where(in_high, words[0], words[1])
        bit = jnp.bitwise_and(
            jnp.right_shift(source, bit_index.astype(jnp.uint32)),
            jnp.asarray(1, dtype=jnp.uint32),
        )
        doubled = remainder + remainder + bit
        return jnp.where(doubled >= divisor_array, doubled - divisor_array, doubled)

    return cast(
        Array,
        jax.lax.fori_loop(
            0,
            64,
            body,
            jnp.asarray(0, dtype=jnp.uint32),
        ),
    ).astype(jnp.uint32)


def _require_exact_fields(
    payload: Mapping[str, Any],
    expected: set[str],
    *,
    label: str,
) -> dict[str, Any]:
    fields = dict(payload)
    if set(fields) != expected:
        missing = sorted(expected - set(fields))
        extra = sorted(set(fields) - expected)
        raise ValueError(f"{label} fields are invalid; missing={missing}, extra={extra}")
    return fields


def _host_state_fields(state: Any, *, label: str) -> dict[str, Any]:
    if isinstance(state, Mapping):
        return dict(state)
    if dataclasses.is_dataclass(state) and not isinstance(state, type):
        return {
            field.name: getattr(state, field.name)
            for field in dataclasses.fields(state)
        }
    raise TypeError(f"legacy {label} state must be a mapping or dataclass")


def _legacy_unsaturated_count(fields: Mapping[str, Any], *, label: str) -> int:
    count_array = jnp.asarray(fields["step_count"])
    if count_array.shape != () or count_array.dtype != jnp.dtype(jnp.int32):
        raise TypeError(f"legacy {label} step_count must be scalar int32")
    count = int(count_array)
    if count < 0:
        raise ValueError(f"negative legacy {label} step_count indicates wrap")
    if count >= _INT32_MAX:
        raise ValueError(f"saturated legacy {label} step_count is ambiguous")
    return count


@chex.dataclass(frozen=True)
class AlbertaPlanStep1State:
    """State for :class:`AlbertaPlanStep1Stream`.

    Attributes:
        key: JAX random key for generating randomness
        true_weights: Current target weight vector ``w*_t`` (only the first
            ``num_relevant`` entries are nonzero; the rest stay at zero)
        true_bias: Current scalar target bias ``b*_t``
        step_count: Saturating int32 compatibility telemetry.
        step_words: Exact big-endian uint32 pair identifying admitted events.
    """

    key: PRNGKeyArray
    true_weights: Float[Array, " feature_dim"]
    true_bias: Float[Array, ""]
    step_count: Int[Array, ""]
    step_words: UInt[Array, " 2"] = None  # type: ignore[assignment]


@chex.dataclass(frozen=True)
class AlbertaPlanStep1StepResult:
    """One staged canonical Step 1 event with commit diagnostics."""

    timestep: TimeStep
    state: AlbertaPlanStep1State
    pre_step_words: UInt[Array, " 2"]
    post_step_words: UInt[Array, " 2"]
    lifetime_counter_valid: Bool[Array, ""]
    lifetime_capacity_available: Bool[Array, ""]
    input_valid: Bool[Array, ""]
    state_valid: Bool[Array, ""]
    output_valid: Bool[Array, ""]
    candidate_state_valid: Bool[Array, ""]
    update_applied: Bool[Array, ""]
    update_rejected: Bool[Array, ""]


@dataclass(frozen=True)
class Step1StreamResourceBudget:
    """Exact persistent-state accounting for one Step 1 stream."""

    stream_type: str
    state_nbytes: int
    exact_clock_nbytes: int = STEP1_STREAM_CLOCK_NBYTES
    exact_clock_delta_nbytes: int = STEP1_STREAM_CLOCK_DELTA_NBYTES
    trainable_scalars: int = 0
    replay_capacity: int = 0

    def to_dict(self) -> dict[str, int | str]:
        return {
            "schema": STEP1_STREAM_RESOURCE_SCHEMA,
            "stream_type": self.stream_type,
            "state_nbytes": self.state_nbytes,
            "exact_clock_nbytes": self.exact_clock_nbytes,
            "exact_clock_delta_nbytes": self.exact_clock_delta_nbytes,
            "trainable_scalars": self.trainable_scalars,
            "replay_capacity": self.replay_capacity,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Step1StreamResourceBudget":
        expected = {
            "schema",
            "stream_type",
            "state_nbytes",
            "exact_clock_nbytes",
            "exact_clock_delta_nbytes",
            "trainable_scalars",
            "replay_capacity",
        }
        fields = _require_exact_fields(
            payload,
            expected,
            label="Step 1 stream resource budget",
        )
        if fields.pop("schema") != STEP1_STREAM_RESOURCE_SCHEMA:
            raise ValueError("Step 1 stream resource schema is unsupported")
        if not isinstance(fields["stream_type"], str) or fields["stream_type"] not in {
            "AlbertaPlanStep1Stream",
            "XDistShiftStream",
        }:
            raise ValueError("Step 1 resource stream type is unsupported")
        for name in expected - {"schema", "stream_type"}:
            if type(fields[name]) is not int or fields[name] < 0:
                raise ValueError(f"Step 1 resource {name} must be non-negative")
        if fields["exact_clock_nbytes"] != STEP1_STREAM_CLOCK_NBYTES:
            raise ValueError("Step 1 exact clock accounting is invalid")
        if fields["exact_clock_delta_nbytes"] != STEP1_STREAM_CLOCK_DELTA_NBYTES:
            raise ValueError("Step 1 exact clock delta is invalid")
        if fields["trainable_scalars"] != 0 or fields["replay_capacity"] != 0:
            raise ValueError("Step 1 streams must not own learner or replay state")
        return cls(**fields)


class AlbertaPlanStep1Stream:
    """Canonical Alberta Plan Step 1 supervised stream.

    Generates targets

        y*_t = w*_t . x_t + b*_t + eta_t,    eta_t ~ N(0, noise_std^2)

    where the first ``num_relevant`` entries of ``w*_t`` follow independent
    Gaussian random walks with std ``drift_rate_w`` per step (the remaining
    entries stay identically zero, mirroring the Sutton 1992 sparse-relevance
    setup), and ``b*_t`` follows a Gaussian random walk with std
    ``drift_rate_b`` per step. Inputs ``x_t`` are drawn iid from
    ``N(0, feature_std^2)``.

    With ``drift_rate_w = drift_rate_b = 0.0`` this becomes a stationary
    target with additive observation noise; the stream is non-stationary
    whenever either drift rate is positive.

    Attributes:
        feature_dim: Dimension of observation vectors (default 20)
        num_relevant: Number of relevant inputs whose weights are nonzero
            (default 5)
        drift_rate_w: Std dev of Gaussian random walk on the relevant
            entries of ``w*_t`` per step (default 0.001)
        drift_rate_b: Std dev of Gaussian random walk on ``b*_t`` per step
            (default 0.001)
        noise_std: Std dev of additive mean-zero target noise ``eta_t``
            (default 1.0)
        feature_std: Std dev of input features ``x_t`` (default 1.0)
    """

    def __init__(
        self,
        feature_dim: int = 20,
        num_relevant: int = 5,
        drift_rate_w: float = 0.001,
        drift_rate_b: float = 0.001,
        noise_std: float = 1.0,
        feature_std: float = 1.0,
    ):
        """Initialize the Alberta Plan Step 1 stream.

        Args:
            feature_dim: Dimension of feature vectors
            num_relevant: Number of relevant inputs (must be <= feature_dim)
            drift_rate_w: Std dev of weight drift per step
            drift_rate_b: Std dev of bias drift per step
            noise_std: Std dev of additive target noise
            feature_std: Std dev of input features

        Raises:
            ValueError: If ``num_relevant > feature_dim`` or either is
                non-positive.
        """
        _require_positive_int32(feature_dim, name="feature_dim")
        _require_positive_int32(num_relevant, name="num_relevant")
        if num_relevant > feature_dim:
            raise ValueError(
                f"num_relevant ({num_relevant}) must not exceed "
                f"feature_dim ({feature_dim})"
            )
        _require_finite_real(
            drift_rate_w,
            name="drift_rate_w",
            nonnegative=True,
        )
        _require_finite_real(
            drift_rate_b,
            name="drift_rate_b",
            nonnegative=True,
        )
        _require_finite_real(noise_std, name="noise_std", nonnegative=True)
        _require_finite_real(feature_std, name="feature_std", nonnegative=True)
        self._feature_dim = feature_dim
        self._num_relevant = num_relevant
        self._drift_rate_w = drift_rate_w
        self._drift_rate_b = drift_rate_b
        self._noise_std = noise_std
        self._feature_std = feature_std

    @property
    def feature_dim(self) -> int:
        """Return the dimension of observation vectors."""
        return self._feature_dim

    @property
    def num_relevant(self) -> int:
        """Return the number of relevant input dimensions."""
        return self._num_relevant

    @property
    def resource_budget(self) -> Step1StreamResourceBudget:
        """Return exact persistent-state accounting for this stream."""

        return Step1StreamResourceBudget(
            stream_type=type(self).__name__,
            state_nbytes=measure_step1_stream_state_nbytes(self.init(jr.key(0))),
        )

    def to_config(self) -> dict[str, Any]:
        """Serialize the canonical Step 1 stream under strict v2 schemas."""

        return {
            "type": type(self).__name__,
            "config_schema": ALBERTA_PLAN_STEP1_CONFIG_SCHEMA,
            "state_schema": ALBERTA_PLAN_STEP1_STATE_SCHEMA,
            "feature_dim": self._feature_dim,
            "num_relevant": self._num_relevant,
            "drift_rate_w": float(self._drift_rate_w),
            "drift_rate_b": float(self._drift_rate_b),
            "noise_std": float(self._noise_std),
            "feature_std": float(self._feature_std),
        }

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "AlbertaPlanStep1Stream":
        """Strictly reconstruct one versioned canonical Step 1 stream."""

        expected = {
            "type",
            "config_schema",
            "state_schema",
            "feature_dim",
            "num_relevant",
            "drift_rate_w",
            "drift_rate_b",
            "noise_std",
            "feature_std",
        }
        fields = _require_exact_fields(
            config,
            expected,
            label="Alberta Plan Step 1 config",
        )
        if fields.pop("type") != cls.__name__:
            raise ValueError("Alberta Plan Step 1 config type is unsupported")
        if fields.pop("config_schema") != ALBERTA_PLAN_STEP1_CONFIG_SCHEMA:
            raise ValueError("Alberta Plan Step 1 config schema is unsupported")
        if fields.pop("state_schema") != ALBERTA_PLAN_STEP1_STATE_SCHEMA:
            raise ValueError("Alberta Plan Step 1 state schema is unsupported")
        return cls(**fields)

    def _require_state_contract(self, state: AlbertaPlanStep1State) -> None:
        """Require every fixed-shape v2 canonical Step 1 state leaf."""

        _require_prng_key(state.key, name="Alberta Plan Step 1 key")
        _require_array(
            state.true_weights,
            name="Alberta Plan Step 1 true_weights",
            shape=(self._feature_dim,),
            dtype=jnp.dtype(jnp.float32),
        )
        _require_array(
            state.true_bias,
            name="Alberta Plan Step 1 true_bias",
            shape=(),
            dtype=jnp.dtype(jnp.float32),
        )
        _require_array(
            state.step_count,
            name="Alberta Plan Step 1 step_count",
            shape=(),
            dtype=jnp.dtype(jnp.int32),
        )
        _require_array(
            state.step_words,
            name="Alberta Plan Step 1 step_words",
            shape=(2,),
            dtype=jnp.dtype(jnp.uint32),
        )

    def state_is_valid(self, state: AlbertaPlanStep1State) -> Bool[Array, ""]:
        """Authenticate exact time, sparse relevance, and finite oracle state."""

        self._require_state_contract(state)
        irrelevant_zero = jnp.all(state.true_weights[self._num_relevant :] == 0.0)
        return (
            _lifetime_counter_valid(state.step_words, state.step_count)
            & irrelevant_zero
            & _tree_floating_arrays_finite(state)
        )

    def init(self, key: Array) -> AlbertaPlanStep1State:
        """Initialize stream state.

        Args:
            key: JAX random key

        Returns:
            Initial stream state with random relevant weights and zero bias.
        """
        _require_prng_key(key, name="Alberta Plan Step 1 init key")
        key, k_init = jr.split(key)
        relevant_init = jr.normal(k_init, (self._num_relevant,), dtype=jnp.float32)
        weights = jnp.zeros(self._feature_dim, dtype=jnp.float32)
        weights = weights.at[: self._num_relevant].set(relevant_init)
        return AlbertaPlanStep1State(
            key=key,
            true_weights=weights,
            true_bias=jnp.array(0.0, dtype=jnp.float32),
            step_count=jnp.array(0, dtype=jnp.int32),
            step_words=jnp.zeros((2,), dtype=jnp.uint32),
        )

    def step(
        self, state: AlbertaPlanStep1State, idx: Array
    ) -> tuple[TimeStep, AlbertaPlanStep1State]:
        """Generate one time step.

        Args:
            state: Current stream state
            idx: Current step index (unused)

        Returns:
            Tuple of (timestep, new_state)
        """
        result = self.step_result(state, idx)
        return result.timestep, result.state

    @functools.partial(jax.jit, static_argnums=(0,))
    def step_result(
        self,
        state: AlbertaPlanStep1State,
        idx: Array,
    ) -> AlbertaPlanStep1StepResult:
        """Stage and atomically commit one exact canonical Step 1 event."""

        self._require_state_contract(state)
        input_valid = _step_input_valid(idx)
        lifetime_counter_valid = _lifetime_counter_valid(
            state.step_words,
            state.step_count,
        )
        state_valid = (
            jnp.all(state.true_weights[self._num_relevant :] == 0.0)
            & _tree_floating_arrays_finite(state)
        )
        proposed_words, lifetime_capacity_available = (
            _checked_lifetime_words_increment(state.step_words)
        )
        key, k_w_drift, k_b_drift, k_x, k_eta = jr.split(state.key, 5)

        # Random walk on the relevant slice of w*_t. Use a full-length zero
        # vector with a relevant-only update so the irrelevant entries never
        # leave zero (preserves the sparse-relevance setting).
        relevant_drift = self._drift_rate_w * jr.normal(
            k_w_drift, (self._num_relevant,), dtype=jnp.float32
        )
        weight_drift = jnp.zeros(self._feature_dim, dtype=jnp.float32)
        weight_drift = weight_drift.at[: self._num_relevant].set(relevant_drift)
        new_weights = state.true_weights + weight_drift

        # Random walk on b*_t.
        bias_drift = self._drift_rate_b * jr.normal(k_b_drift, (), dtype=jnp.float32)
        new_bias = state.true_bias + bias_drift

        # Sample input features.
        x = self._feature_std * jr.normal(k_x, (self._feature_dim,), dtype=jnp.float32)

        # Compute target: y* = w* . x + b* + eta.
        eta = self._noise_std * jr.normal(k_eta, (), dtype=jnp.float32)
        target = jnp.dot(new_weights, x) + new_bias + eta

        candidate_state = AlbertaPlanStep1State(
            key=key,
            true_weights=new_weights,
            true_bias=new_bias,
            step_count=_saturating_int32_counter_increment(state.step_count),
            step_words=proposed_words,
        )
        candidate_state_valid = self.state_is_valid(candidate_state)
        output_valid = jnp.all(jnp.isfinite(x)) & jnp.isfinite(target)
        update_applied = (
            lifetime_counter_valid
            & lifetime_capacity_available
            & input_valid
            & state_valid
            & output_valid
            & candidate_state_valid
        )
        new_state = jax.lax.cond(
            update_applied,
            lambda _: candidate_state,
            lambda _: state,
            operand=None,
        )
        timestep = TimeStep(
            observation=jnp.where(
                update_applied,
                x,
                jnp.full_like(x, jnp.nan),
            ),
            target=jnp.where(
                update_applied,
                jnp.atleast_1d(target),
                jnp.full((1,), jnp.nan, dtype=jnp.float32),
            ),
        )
        return AlbertaPlanStep1StepResult(
            timestep=timestep,
            state=new_state,
            pre_step_words=state.step_words,
            post_step_words=cast(Array, new_state.step_words),
            lifetime_counter_valid=lifetime_counter_valid,
            lifetime_capacity_available=lifetime_capacity_available,
            input_valid=input_valid,
            state_valid=state_valid,
            output_valid=output_valid,
            candidate_state_valid=candidate_state_valid,
            update_applied=update_applied,
            update_rejected=~update_applied,
        )


@chex.dataclass(frozen=True)
class XDistShiftState:
    """State for :class:`XDistShiftStream`.

    Attributes:
        key: JAX random key for generating randomness
        true_weights: Fixed target weight vector (only the first
            ``num_relevant`` entries are nonzero; sampled once at init)
        current_scales: Current per-feature scale vector ``s_t``
        step_count: Saturating int32 compatibility telemetry.
        step_words: Exact big-endian uint32 pair identifying admitted events.
    """

    key: PRNGKeyArray
    true_weights: Float[Array, " feature_dim"]
    current_scales: Float[Array, " feature_dim"]
    step_count: Int[Array, ""]
    step_words: UInt[Array, " 2"] = None  # type: ignore[assignment]


@chex.dataclass(frozen=True)
class XDistShiftStepResult:
    """One staged x-distribution-shift event with commit diagnostics."""

    timestep: TimeStep
    state: XDistShiftState
    pre_step_words: UInt[Array, " 2"]
    post_step_words: UInt[Array, " 2"]
    interval_remainder: UInt[Array, ""]
    scale_change_due: Bool[Array, ""]
    scale_changed: Bool[Array, ""]
    lifetime_counter_valid: Bool[Array, ""]
    lifetime_capacity_available: Bool[Array, ""]
    input_valid: Bool[Array, ""]
    state_valid: Bool[Array, ""]
    output_valid: Bool[Array, ""]
    candidate_state_valid: Bool[Array, ""]
    update_applied: Bool[Array, ""]
    update_rejected: Bool[Array, ""]


class XDistShiftStream:
    """Step 1 stream that holds the target fixed and shifts the x distribution.

    Implements the third Step 1 case from the Alberta Plan: "The problem is
    non-stationary if w*_t or b*_t change over time OR if the distribution of
    x_t changes over time." This stream isolates input-distribution
    non-stationarity from target non-stationarity.

    The TARGET function is fixed: ``w*`` is sampled once at ``init`` and never
    changes. Only the INPUT distribution shifts: every
    ``scale_change_interval`` steps a new per-feature scale vector
    ``s ~ Uniform[scale_min, scale_max]`` is drawn, and observations are

        x_t = s * z_t,    z_t ~ N(0, 1)

    The target is computed from the SCALED observation ``x_t`` so that the
    learner sees the same (observation, target) relationship the underlying
    affine map describes; the scale changes induce non-stationarity through
    the distribution of features (their variances and norms) rather than
    through the target function itself.

    Attributes:
        feature_dim: Dimension of observation vectors
        num_relevant: Number of relevant inputs whose weights are nonzero
        noise_std: Std dev of additive mean-zero target noise (default 0.1).
            Only added to the target when ``noise_in_target=True``.
        scale_change_interval: Steps between scale resamplings (default 2000)
        scale_min: Minimum per-feature scale, inclusive (default 0.1)
        scale_max: Maximum per-feature scale, exclusive (default 10.0)
        noise_in_target: If True, add Gaussian noise to the target. If False,
            the target is exactly ``w* . x``.
    """

    def __init__(
        self,
        feature_dim: int,
        num_relevant: int,
        noise_std: float = 0.1,
        scale_change_interval: int = 2000,
        scale_min: float = 0.1,
        scale_max: float = 10.0,
        noise_in_target: bool = True,
    ):
        """Initialize the x-distribution-shift stream.

        Args:
            feature_dim: Dimension of feature vectors
            num_relevant: Number of relevant inputs (must be <= feature_dim)
            noise_std: Std dev of additive target noise (only used if
                ``noise_in_target=True``)
            scale_change_interval: Steps between abrupt scale resamplings
            scale_min: Lower bound of uniform scale distribution
            scale_max: Upper bound of uniform scale distribution
            noise_in_target: Whether to add Gaussian noise to the target

        Raises:
            ValueError: If ``num_relevant > feature_dim``,
                ``scale_min >= scale_max``, ``scale_change_interval <= 0``, or
                if ``feature_dim`` / ``num_relevant`` are non-positive.
        """
        _require_positive_int32(feature_dim, name="feature_dim")
        _require_positive_int32(num_relevant, name="num_relevant")
        if num_relevant > feature_dim:
            raise ValueError(
                f"num_relevant ({num_relevant}) must not exceed "
                f"feature_dim ({feature_dim})"
            )
        _require_finite_real(noise_std, name="noise_std", nonnegative=True)
        _require_positive_int32(
            scale_change_interval,
            name="scale_change_interval",
        )
        _require_finite_real(scale_min, name="scale_min")
        _require_finite_real(scale_max, name="scale_max")
        if scale_min >= scale_max:
            raise ValueError(
                f"scale_min ({scale_min}) must be less than scale_max ({scale_max})"
            )
        stored_scale_min = float(jnp.asarray(scale_min, dtype=jnp.float32))
        stored_scale_max = float(jnp.asarray(scale_max, dtype=jnp.float32))
        if stored_scale_min >= stored_scale_max:
            raise ValueError("scale bounds must remain ordered when stored as float32")
        if type(noise_in_target) is not bool:
            raise ValueError("noise_in_target must be an exact bool")
        self._feature_dim = feature_dim
        self._num_relevant = num_relevant
        self._noise_std = noise_std
        self._scale_change_interval = scale_change_interval
        self._scale_min = scale_min
        self._scale_max = scale_max
        self._noise_in_target = noise_in_target

    @property
    def feature_dim(self) -> int:
        """Return the dimension of observation vectors."""
        return self._feature_dim

    @property
    def num_relevant(self) -> int:
        """Return the number of relevant input dimensions."""
        return self._num_relevant

    @property
    def scale_change_interval(self) -> int:
        """Return the exact number of events between scale redraws."""

        return self._scale_change_interval

    @property
    def resource_budget(self) -> Step1StreamResourceBudget:
        """Return exact persistent-state accounting for this stream."""

        return Step1StreamResourceBudget(
            stream_type=type(self).__name__,
            state_nbytes=measure_step1_stream_state_nbytes(self.init(jr.key(0))),
        )

    def to_config(self) -> dict[str, Any]:
        """Serialize the x-distribution stream under strict v2 schemas."""

        return {
            "type": type(self).__name__,
            "config_schema": XDIST_SHIFT_CONFIG_SCHEMA,
            "state_schema": XDIST_SHIFT_STATE_SCHEMA,
            "feature_dim": self._feature_dim,
            "num_relevant": self._num_relevant,
            "noise_std": float(self._noise_std),
            "scale_change_interval": self._scale_change_interval,
            "scale_min": float(self._scale_min),
            "scale_max": float(self._scale_max),
            "noise_in_target": self._noise_in_target,
        }

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "XDistShiftStream":
        """Strictly reconstruct one versioned x-distribution stream."""

        expected = {
            "type",
            "config_schema",
            "state_schema",
            "feature_dim",
            "num_relevant",
            "noise_std",
            "scale_change_interval",
            "scale_min",
            "scale_max",
            "noise_in_target",
        }
        fields = _require_exact_fields(
            config,
            expected,
            label="x-distribution-shift config",
        )
        if fields.pop("type") != cls.__name__:
            raise ValueError("x-distribution-shift config type is unsupported")
        if fields.pop("config_schema") != XDIST_SHIFT_CONFIG_SCHEMA:
            raise ValueError("x-distribution-shift config schema is unsupported")
        if fields.pop("state_schema") != XDIST_SHIFT_STATE_SCHEMA:
            raise ValueError("x-distribution-shift state schema is unsupported")
        return cls(**fields)

    def _require_state_contract(self, state: XDistShiftState) -> None:
        """Require every fixed-shape v2 x-distribution state leaf."""

        _require_prng_key(state.key, name="x-distribution-shift key")
        _require_array(
            state.true_weights,
            name="x-distribution-shift true_weights",
            shape=(self._feature_dim,),
            dtype=jnp.dtype(jnp.float32),
        )
        _require_array(
            state.current_scales,
            name="x-distribution-shift current_scales",
            shape=(self._feature_dim,),
            dtype=jnp.dtype(jnp.float32),
        )
        _require_array(
            state.step_count,
            name="x-distribution-shift step_count",
            shape=(),
            dtype=jnp.dtype(jnp.int32),
        )
        _require_array(
            state.step_words,
            name="x-distribution-shift step_words",
            shape=(2,),
            dtype=jnp.dtype(jnp.uint32),
        )

    def state_is_valid(self, state: XDistShiftState) -> Bool[Array, ""]:
        """Authenticate exact time, fixed target, and current scales."""

        self._require_state_contract(state)
        irrelevant_zero = jnp.all(state.true_weights[self._num_relevant :] == 0.0)
        scales_in_range = jnp.all(
            (state.current_scales >= self._scale_min)
            & (state.current_scales <= self._scale_max)
        )
        return (
            _lifetime_counter_valid(state.step_words, state.step_count)
            & irrelevant_zero
            & scales_in_range
            & _tree_floating_arrays_finite(state)
        )

    def init(self, key: Array) -> XDistShiftState:
        """Initialize stream state.

        Args:
            key: JAX random key

        Returns:
            Initial stream state with a fixed target function and an initial
            per-feature scale vector.
        """
        _require_prng_key(key, name="x-distribution-shift init key")
        key, k_w, k_scales = jr.split(key, 3)
        relevant_w = jr.normal(k_w, (self._num_relevant,), dtype=jnp.float32)
        weights = jnp.zeros(self._feature_dim, dtype=jnp.float32)
        weights = weights.at[: self._num_relevant].set(relevant_w)
        initial_scales = jr.uniform(
            k_scales,
            (self._feature_dim,),
            minval=self._scale_min,
            maxval=self._scale_max,
            dtype=jnp.float32,
        )
        return XDistShiftState(
            key=key,
            true_weights=weights,
            current_scales=initial_scales,
            step_count=jnp.array(0, dtype=jnp.int32),
            step_words=jnp.zeros((2,), dtype=jnp.uint32),
        )

    def step(self, state: XDistShiftState, idx: Array) -> tuple[TimeStep, XDistShiftState]:
        """Generate one time step.

        Args:
            state: Current stream state
            idx: Current step index (unused)

        Returns:
            Tuple of (timestep, new_state)
        """
        result = self.step_result(state, idx)
        return result.timestep, result.state

    @functools.partial(jax.jit, static_argnums=(0,))
    def step_result(
        self,
        state: XDistShiftState,
        idx: Array,
    ) -> XDistShiftStepResult:
        """Stage and atomically commit one exact x-distribution event."""

        self._require_state_contract(state)
        input_valid = _step_input_valid(idx)
        lifetime_counter_valid = _lifetime_counter_valid(
            state.step_words,
            state.step_count,
        )
        scales_in_range = jnp.all(
            (state.current_scales >= self._scale_min)
            & (state.current_scales <= self._scale_max)
        )
        state_valid = (
            jnp.all(state.true_weights[self._num_relevant :] == 0.0)
            & scales_in_range
            & _tree_floating_arrays_finite(state)
        )
        proposed_words, lifetime_capacity_available = (
            _checked_lifetime_words_increment(state.step_words)
        )
        key, k_scales, k_z, k_eta = jr.split(state.key, 4)

        # Decide whether to redraw scales this step. Always sample candidate
        # scales (jit-friendly) and use jnp.where to commit conditionally.
        interval_remainder = _lifetime_words_remainder(
            state.step_words,
            self._scale_change_interval,
        )
        should_change = interval_remainder == jnp.asarray(0, dtype=jnp.uint32)
        candidate_scales = jr.uniform(
            k_scales,
            (self._feature_dim,),
            minval=self._scale_min,
            maxval=self._scale_max,
            dtype=jnp.float32,
        )
        new_scales = jnp.where(should_change, candidate_scales, state.current_scales)

        # Sample latent z ~ N(0, 1), then form x = s * z.
        z = jr.normal(k_z, (self._feature_dim,), dtype=jnp.float32)
        x = new_scales * z

        # Compute target from the SCALED observation.
        target = jnp.dot(state.true_weights, x)
        eta = self._noise_std * jr.normal(k_eta, (), dtype=jnp.float32)
        target = target + jnp.where(self._noise_in_target, eta, jnp.float32(0.0))

        candidate_state = XDistShiftState(
            key=key,
            true_weights=state.true_weights,
            current_scales=new_scales,
            step_count=_saturating_int32_counter_increment(state.step_count),
            step_words=proposed_words,
        )
        candidate_state_valid = self.state_is_valid(candidate_state)
        output_valid = jnp.all(jnp.isfinite(x)) & jnp.isfinite(target)
        update_applied = (
            lifetime_counter_valid
            & lifetime_capacity_available
            & input_valid
            & state_valid
            & output_valid
            & candidate_state_valid
        )
        new_state = jax.lax.cond(
            update_applied,
            lambda _: candidate_state,
            lambda _: state,
            operand=None,
        )
        timestep = TimeStep(
            observation=jnp.where(
                update_applied,
                x,
                jnp.full_like(x, jnp.nan),
            ),
            target=jnp.where(
                update_applied,
                jnp.atleast_1d(target),
                jnp.full((1,), jnp.nan, dtype=jnp.float32),
            ),
        )
        return XDistShiftStepResult(
            timestep=timestep,
            state=new_state,
            pre_step_words=state.step_words,
            post_step_words=cast(Array, new_state.step_words),
            interval_remainder=interval_remainder,
            scale_change_due=should_change,
            scale_changed=should_change & update_applied,
            lifetime_counter_valid=lifetime_counter_valid,
            lifetime_capacity_available=lifetime_capacity_available,
            input_valid=input_valid,
            state_valid=state_valid,
            output_valid=output_valid,
            candidate_state_valid=candidate_state_valid,
            update_applied=update_applied,
            update_rejected=~update_applied,
        )


def step1_stream_clock_nbytes() -> int:
    """Return bytes owned by saturating telemetry plus exact identity."""

    return STEP1_STREAM_CLOCK_NBYTES


def measure_step1_stream_state_nbytes(
    state: AlbertaPlanStep1State | XDistShiftState,
) -> int:
    """Measure every persistent JAX-array byte in one concrete stream state."""

    return sum(
        int(leaf.size) * int(leaf.dtype.itemsize)
        for leaf in jax.tree.leaves(state)
        if isinstance(leaf, Array)
    )


def migrate_legacy_alberta_plan_step1_state(
    legacy_state: Any,
    *,
    stream: AlbertaPlanStep1Stream,
) -> AlbertaPlanStep1State:
    """Migrate an exact unsaturated canonical Step 1 stream state."""

    fields = _host_state_fields(legacy_state, label="Alberta Plan Step 1")
    fields = _require_exact_fields(
        fields,
        {"key", "true_weights", "true_bias", "step_count"},
        label="legacy Alberta Plan Step 1 state",
    )
    count = _legacy_unsaturated_count(fields, label="Alberta Plan Step 1")
    fields["step_words"] = jnp.asarray((0, count), dtype=jnp.uint32)
    migrated = AlbertaPlanStep1State(**fields)
    stream._require_state_contract(migrated)
    if not bool(jax.device_get(stream.state_is_valid(migrated))):
        raise ValueError("legacy Alberta Plan Step 1 state is invalid")
    return migrated


def migrate_legacy_xdist_shift_state(
    legacy_state: Any,
    *,
    stream: XDistShiftStream,
) -> XDistShiftState:
    """Migrate an exact unsaturated x-distribution stream state."""

    fields = _host_state_fields(legacy_state, label="x-distribution-shift")
    fields = _require_exact_fields(
        fields,
        {"key", "true_weights", "current_scales", "step_count"},
        label="legacy x-distribution-shift state",
    )
    count = _legacy_unsaturated_count(fields, label="x-distribution-shift")
    fields["step_words"] = jnp.asarray((0, count), dtype=jnp.uint32)
    migrated = XDistShiftState(**fields)
    stream._require_state_contract(migrated)
    if not bool(jax.device_get(stream.state_is_valid(migrated))):
        raise ValueError("legacy x-distribution-shift state is invalid")
    return migrated
