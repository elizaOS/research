"""Synthetic non-stationary experience streams for testing continual learning.

These streams generate non-stationary supervised learning problems where
the target function changes over time, testing the learner's ability to
track and adapt.

All streams use JAX-compatible pure functions that work with jax.lax.scan.
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
from alberta_framework.streams.base import ScanStream

RANDOM_WALK_CONFIG_SCHEMA = "alberta.random-walk-stream.config.v2"
RANDOM_WALK_STATE_SCHEMA = "alberta.random-walk-stream.state.v2"
HIDDEN_STATE_AR2_CONFIG_SCHEMA = "alberta.hidden-state-ar2-stream.config.v2"
HIDDEN_STATE_AR2_STATE_SCHEMA = "alberta.hidden-state-ar2-stream.state.v2"
ABRUPT_CHANGE_CONFIG_SCHEMA = "alberta.abrupt-change-stream.config.v2"
ABRUPT_CHANGE_STATE_SCHEMA = "alberta.abrupt-change-stream.state.v2"
SUTTON_EXPERIMENT1_CONFIG_SCHEMA = "alberta.sutton-experiment1-stream.config.v2"
SUTTON_EXPERIMENT1_STATE_SCHEMA = "alberta.sutton-experiment1-stream.state.v2"
CYCLIC_CONFIG_SCHEMA = "alberta.cyclic-stream.config.v2"
CYCLIC_STATE_SCHEMA = "alberta.cyclic-stream.state.v2"
PERIODIC_CHANGE_CONFIG_SCHEMA = "alberta.periodic-change-stream.config.v2"
PERIODIC_CHANGE_STATE_SCHEMA = "alberta.periodic-change-stream.state.v2"
SCALED_STREAM_CONFIG_SCHEMA = "alberta.scaled-stream.config.v2"
SCALED_STREAM_STATE_SCHEMA = "alberta.scaled-stream.state.v2"
DYNAMIC_SCALE_SHIFT_CONFIG_SCHEMA = "alberta.dynamic-scale-shift-stream.config.v2"
DYNAMIC_SCALE_SHIFT_STATE_SCHEMA = "alberta.dynamic-scale-shift-stream.state.v2"
SCALE_DRIFT_CONFIG_SCHEMA = "alberta.scale-drift-stream.config.v2"
SCALE_DRIFT_STATE_SCHEMA = "alberta.scale-drift-stream.state.v2"
SYNTHETIC_STREAM_RESOURCE_SCHEMA = "alberta.synthetic-stream.resource-budget.v2"
SYNTHETIC_STREAM_CLOCK_NBYTES = 12
SYNTHETIC_STREAM_CLOCK_DELTA_NBYTES = 8
SYNTHETIC_STREAM_NEW_CLOCK_DELTA_NBYTES = 12

_INT32_MAX = 2**31 - 1
_FLOAT32_MAX = 3.4028234663852886e38


def _require_array(
    value: Any,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: jnp.dtype,
) -> Array:
    """Require an exact public array contract without silent narrowing."""

    array = jnp.asarray(value)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if array.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {array.dtype}")
    return array


def _require_prng_key(key: Array, *, name: str) -> None:
    """Require a scalar typed or legacy JAX PRNG key."""

    key_data = jr.key_data(key)
    if key_data.shape != (2,) or key_data.dtype != jnp.dtype(jnp.uint32):
        raise TypeError(f"{name} must be a scalar JAX PRNG key")


def _tree_floating_arrays_finite(value: Any) -> Bool[Array, ""]:
    """Return whether every persistent floating or complex leaf is finite."""

    valid = jnp.asarray(True, dtype=jnp.bool_)
    for leaf in jax.tree.leaves(value):
        if hasattr(leaf, "dtype") and jnp.issubdtype(leaf.dtype, jnp.inexact):
            valid = valid & jnp.all(jnp.isfinite(leaf))
    return valid


def _require_positive_int32(value: Any, *, name: str, minimum: int = 1) -> None:
    if type(value) is not int or not minimum <= value <= _INT32_MAX:
        raise ValueError(f"{name} must be an exact integer in [{minimum}, {_INT32_MAX}]")


def _require_finite_real(
    value: Any,
    *,
    name: str,
    nonnegative: bool = False,
    positive: bool = False,
) -> None:
    valid = (
        not isinstance(value, bool)
        and isinstance(value, Real)
        and math.isfinite(float(value))
        and abs(float(value)) <= _FLOAT32_MAX
    )
    if nonnegative:
        valid = valid and float(value) >= 0.0
    if positive:
        valid = valid and float(value) > 0.0
    if not valid:
        qualifier = " positive" if positive else " non-negative" if nonnegative else ""
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
def _divmod_lifetime_words(words: Array, divisor: int) -> tuple[Array, Array]:
    """Divide an exact uint64 word pair by a positive int32 without x64."""

    _require_array(
        words,
        name="synthetic stream step_words",
        shape=(2,),
        dtype=jnp.dtype(jnp.uint32),
    )
    _require_positive_int32(divisor, name="schedule divisor")
    divisor_array = jnp.asarray(divisor, dtype=jnp.uint32)

    def body(index: Array, carry: tuple[Array, Array, Array]) -> tuple[Array, Array, Array]:
        remainder, quotient_high, quotient_low = carry
        in_high = index < 32
        bit_index = jnp.asarray(31, dtype=jnp.int32) - jnp.mod(index, 32)
        source = jnp.where(in_high, words[0], words[1])
        bit = jnp.bitwise_and(
            jnp.right_shift(source, bit_index.astype(jnp.uint32)),
            jnp.asarray(1, dtype=jnp.uint32),
        )
        doubled = remainder + remainder + bit
        subtract = doubled >= divisor_array
        next_remainder = jnp.where(subtract, doubled - divisor_array, doubled)
        mask = jnp.left_shift(
            jnp.asarray(1, dtype=jnp.uint32),
            bit_index.astype(jnp.uint32),
        )
        next_high = jnp.where(
            in_high & subtract,
            jnp.bitwise_or(quotient_high, mask),
            quotient_high,
        )
        next_low = jnp.where(
            (~in_high) & subtract,
            jnp.bitwise_or(quotient_low, mask),
            quotient_low,
        )
        return next_remainder, next_high, next_low

    zero = jnp.asarray(0, dtype=jnp.uint32)
    remainder, high, low = jax.lax.fori_loop(0, 64, body, (zero, zero, zero))
    return jnp.stack((high, low)).astype(jnp.uint32), remainder


def _lifetime_words_remainder(words: Array, divisor: int) -> UInt[Array, ""]:
    return cast(Array, _divmod_lifetime_words(words, divisor)[1]).astype(jnp.uint32)


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
        return {field.name: getattr(state, field.name) for field in dataclasses.fields(state)}
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


def _require_external_legacy_count(value: Any, *, label: str) -> int:
    if type(value) is not int or not 0 <= value < _INT32_MAX:
        raise ValueError(
            f"legacy {label} requires an externally authenticated exact count "
            f"in [0, {_INT32_MAX - 1}]"
        )
    return value


@chex.dataclass(frozen=True)
class SyntheticStreamStepResult:
    """One staged synthetic event with exact-clock and commit diagnostics."""

    timestep: TimeStep
    state: Any
    pre_step_words: UInt[Array, " 2"]
    post_step_words: UInt[Array, " 2"]
    schedule_remainder: UInt[Array, ""]
    secondary_schedule_remainder: UInt[Array, ""]
    schedule_index: Int[Array, ""]
    schedule_due: Bool[Array, ""]
    secondary_schedule_due: Bool[Array, ""]
    oracle_changed: Bool[Array, ""]
    child_update_applied: Bool[Array, ""]
    child_counter_aligned: Bool[Array, ""]
    lifetime_counter_valid: Bool[Array, ""]
    lifetime_capacity_available: Bool[Array, ""]
    input_valid: Bool[Array, ""]
    state_valid: Bool[Array, ""]
    output_valid: Bool[Array, ""]
    candidate_state_valid: Bool[Array, ""]
    update_applied: Bool[Array, ""]
    update_rejected: Bool[Array, ""]


# Per-stream aliases make result annotations discoverable without multiplying
# identical pytrees.
RandomWalkStepResult = SyntheticStreamStepResult
HiddenStateAR2StepResult = SyntheticStreamStepResult
AbruptChangeStepResult = SyntheticStreamStepResult
SuttonExperiment1StepResult = SyntheticStreamStepResult
CyclicStepResult = SyntheticStreamStepResult
PeriodicChangeStepResult = SyntheticStreamStepResult
ScaledStreamStepResult = SyntheticStreamStepResult
DynamicScaleShiftStepResult = SyntheticStreamStepResult
ScaleDriftStepResult = SyntheticStreamStepResult


@dataclass(frozen=True)
class SyntheticStreamResourceBudget:
    """Exact persistent-state accounting for one synthetic stream."""

    stream_type: str
    state_nbytes: int
    exact_clock_delta_nbytes: int
    exact_clock_nbytes: int = SYNTHETIC_STREAM_CLOCK_NBYTES
    trainable_scalars: int = 0
    replay_capacity: int = 0

    def to_dict(self) -> dict[str, int | str]:
        return {
            "schema": SYNTHETIC_STREAM_RESOURCE_SCHEMA,
            "stream_type": self.stream_type,
            "state_nbytes": self.state_nbytes,
            "exact_clock_nbytes": self.exact_clock_nbytes,
            "exact_clock_delta_nbytes": self.exact_clock_delta_nbytes,
            "trainable_scalars": self.trainable_scalars,
            "replay_capacity": self.replay_capacity,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SyntheticStreamResourceBudget":
        expected = {
            "schema",
            "stream_type",
            "state_nbytes",
            "exact_clock_nbytes",
            "exact_clock_delta_nbytes",
            "trainable_scalars",
            "replay_capacity",
        }
        fields = _require_exact_fields(payload, expected, label="synthetic stream resource budget")
        if fields.pop("schema") != SYNTHETIC_STREAM_RESOURCE_SCHEMA:
            raise ValueError("synthetic stream resource schema is unsupported")
        supported = {
            "RandomWalkStream",
            "HiddenStateAR2Stream",
            "AbruptChangeStream",
            "SuttonExperiment1Stream",
            "CyclicStream",
            "PeriodicChangeStream",
            "ScaledStreamWrapper",
            "DynamicScaleShiftStream",
            "ScaleDriftStream",
        }
        if not isinstance(fields["stream_type"], str) or fields["stream_type"] not in supported:
            raise ValueError("synthetic resource stream type is unsupported")
        for name in expected - {"schema", "stream_type"}:
            if type(fields[name]) is not int or fields[name] < 0:
                raise ValueError(f"synthetic resource {name} must be non-negative")
        clock_nbytes = fields["exact_clock_nbytes"]
        clock_delta_nbytes = fields["exact_clock_delta_nbytes"]
        if fields["stream_type"] == "ScaledStreamWrapper":
            clock_valid = (
                clock_nbytes >= SYNTHETIC_STREAM_CLOCK_NBYTES
                and clock_nbytes % SYNTHETIC_STREAM_CLOCK_NBYTES == 0
            )
            clock_count = clock_nbytes // SYNTHETIC_STREAM_CLOCK_NBYTES
            delta_valid = (
                SYNTHETIC_STREAM_CLOCK_DELTA_NBYTES * clock_count
                <= clock_delta_nbytes
                <= SYNTHETIC_STREAM_NEW_CLOCK_DELTA_NBYTES * clock_count
                and clock_delta_nbytes % 4 == 0
            )
        else:
            clock_valid = clock_nbytes == SYNTHETIC_STREAM_CLOCK_NBYTES
            expected_delta = (
                SYNTHETIC_STREAM_NEW_CLOCK_DELTA_NBYTES
                if fields["stream_type"] in {"RandomWalkStream", "HiddenStateAR2Stream"}
                else SYNTHETIC_STREAM_CLOCK_DELTA_NBYTES
            )
            delta_valid = clock_delta_nbytes == expected_delta
        if not clock_valid:
            raise ValueError("synthetic exact clock accounting is invalid")
        if not delta_valid:
            raise ValueError("synthetic exact clock delta is invalid")
        if fields["trainable_scalars"] != 0 or fields["replay_capacity"] != 0:
            raise ValueError("synthetic streams must not own learner or replay state")
        return cls(**fields)


def _commit_step_result(
    *,
    current_state: Any,
    candidate_state: Any,
    observation: Array,
    target: Array,
    pre_step_words: Array,
    proposed_step_words: Array,
    schedule_remainder: Array,
    secondary_schedule_remainder: Array,
    schedule_index: Array,
    schedule_due: Array,
    secondary_schedule_due: Array,
    oracle_changed: Array,
    child_update_applied: Array,
    child_counter_aligned: Array,
    lifetime_counter_valid: Array,
    lifetime_capacity_available: Array,
    input_valid: Array,
    state_valid: Array,
    output_valid: Array,
    candidate_state_valid: Array,
) -> SyntheticStreamStepResult:
    """Atomically commit every persistent leaf or emit a refusal sentinel."""

    update_applied = (
        lifetime_counter_valid
        & lifetime_capacity_available
        & input_valid
        & state_valid
        & output_valid
        & candidate_state_valid
        & child_update_applied
        & child_counter_aligned
    )
    new_state = jax.lax.cond(
        update_applied,
        lambda _: candidate_state,
        lambda _: current_state,
        operand=None,
    )
    timestep = TimeStep(
        observation=jnp.where(update_applied, observation, jnp.full_like(observation, jnp.nan)),
        target=jnp.where(update_applied, target, jnp.full_like(target, jnp.nan)),
    )
    return SyntheticStreamStepResult(
        timestep=timestep,
        state=new_state,
        pre_step_words=pre_step_words,
        post_step_words=jnp.where(update_applied, proposed_step_words, pre_step_words),
        schedule_remainder=schedule_remainder,
        secondary_schedule_remainder=secondary_schedule_remainder,
        schedule_index=schedule_index,
        schedule_due=schedule_due,
        secondary_schedule_due=secondary_schedule_due,
        oracle_changed=oracle_changed & update_applied,
        child_update_applied=child_update_applied,
        child_counter_aligned=child_counter_aligned,
        lifetime_counter_valid=lifetime_counter_valid,
        lifetime_capacity_available=lifetime_capacity_available,
        input_valid=input_valid,
        state_valid=state_valid,
        output_valid=output_valid,
        candidate_state_valid=candidate_state_valid,
        update_applied=update_applied,
        update_rejected=~update_applied,
    )


def _no_schedule_diagnostics() -> tuple[Array, Array, Array, Array, Array]:
    zero_u32 = jnp.asarray(0, dtype=jnp.uint32)
    zero_i32 = jnp.asarray(0, dtype=jnp.int32)
    false = jnp.asarray(False, dtype=jnp.bool_)
    return zero_u32, zero_u32, zero_i32, false, false


@chex.dataclass(frozen=True)
class RandomWalkState:
    """State for RandomWalkStream.

    Attributes:
        key: JAX random key for generating randomness
        true_weights: Current true target weights
        step_count: Saturating int32 compatibility telemetry
        step_words: Exact big-endian uint32 event identity
    """

    key: PRNGKeyArray
    true_weights: Float[Array, " feature_dim"]
    step_count: Int[Array, ""] = None  # type: ignore[assignment]
    step_words: UInt[Array, " 2"] = None  # type: ignore[assignment]


class RandomWalkStream:
    """Non-stationary stream where target weights drift via random walk.

    The true target function is linear: `y* = w_true @ x + noise`
    where w_true evolves via random walk at each time step.

    This tests the learner's ability to continuously track a moving target.

    Attributes:
        feature_dim: Dimension of observation vectors
        drift_rate: Standard deviation of weight drift per step
        noise_std: Standard deviation of observation noise
        feature_std: Standard deviation of features
    """

    def __init__(
        self,
        feature_dim: int,
        drift_rate: float = 0.001,
        noise_std: float = 0.1,
        feature_std: float = 1.0,
    ):
        """Initialize the random walk target stream.

        Args:
            feature_dim: Dimension of the feature/observation vectors
            drift_rate: Std dev of weight changes per step (controls non-stationarity)
            noise_std: Std dev of target noise
            feature_std: Std dev of feature values
        """
        _require_positive_int32(feature_dim, name="feature_dim")
        _require_finite_real(drift_rate, name="drift_rate", nonnegative=True)
        _require_finite_real(noise_std, name="noise_std", nonnegative=True)
        _require_finite_real(feature_std, name="feature_std", nonnegative=True)
        self._feature_dim = feature_dim
        self._drift_rate = drift_rate
        self._noise_std = noise_std
        self._feature_std = feature_std

    @property
    def feature_dim(self) -> int:
        """Return the dimension of observation vectors."""
        return self._feature_dim

    @property
    def resource_budget(self) -> SyntheticStreamResourceBudget:
        """Return exact persistent-state accounting for this stream."""

        return SyntheticStreamResourceBudget(
            stream_type=type(self).__name__,
            state_nbytes=measure_synthetic_stream_state_nbytes(self.init(jr.key(0))),
            exact_clock_delta_nbytes=SYNTHETIC_STREAM_NEW_CLOCK_DELTA_NBYTES,
        )

    def to_config(self) -> dict[str, Any]:
        """Serialize this random-walk stream under strict v2 schemas."""

        return {
            "type": type(self).__name__,
            "config_schema": RANDOM_WALK_CONFIG_SCHEMA,
            "state_schema": RANDOM_WALK_STATE_SCHEMA,
            "feature_dim": self._feature_dim,
            "drift_rate": float(self._drift_rate),
            "noise_std": float(self._noise_std),
            "feature_std": float(self._feature_std),
        }

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "RandomWalkStream":
        expected = {
            "type",
            "config_schema",
            "state_schema",
            "feature_dim",
            "drift_rate",
            "noise_std",
            "feature_std",
        }
        fields = _require_exact_fields(config, expected, label="random-walk config")
        if fields.pop("type") != cls.__name__:
            raise ValueError("random-walk config type is unsupported")
        if fields.pop("config_schema") != RANDOM_WALK_CONFIG_SCHEMA:
            raise ValueError("random-walk config schema is unsupported")
        if fields.pop("state_schema") != RANDOM_WALK_STATE_SCHEMA:
            raise ValueError("random-walk state schema is unsupported")
        return cls(**fields)

    def _require_state_contract(self, state: RandomWalkState) -> None:
        _require_prng_key(state.key, name="random-walk key")
        _require_array(
            state.true_weights,
            name="random-walk true_weights",
            shape=(self._feature_dim,),
            dtype=jnp.dtype(jnp.float32),
        )
        _require_array(
            state.step_count,
            name="random-walk step_count",
            shape=(),
            dtype=jnp.dtype(jnp.int32),
        )
        _require_array(
            state.step_words,
            name="random-walk step_words",
            shape=(2,),
            dtype=jnp.dtype(jnp.uint32),
        )

    def state_is_valid(self, state: RandomWalkState) -> Bool[Array, ""]:
        self._require_state_contract(state)
        return _lifetime_counter_valid(state.step_words, state.step_count) & (
            _tree_floating_arrays_finite(state)
        )

    def init(self, key: Array) -> RandomWalkState:
        """Initialize stream state.

        Args:
            key: JAX random key

        Returns:
            Initial stream state with random weights
        """
        _require_prng_key(key, name="random-walk init key")
        key, subkey = jr.split(key)
        weights = jr.normal(subkey, (self._feature_dim,), dtype=jnp.float32)
        return RandomWalkState(
            key=key,
            true_weights=weights,
            step_count=jnp.asarray(0, dtype=jnp.int32),
            step_words=jnp.zeros((2,), dtype=jnp.uint32),
        )

    def step(self, state: RandomWalkState, idx: Array) -> tuple[TimeStep, RandomWalkState]:
        """Generate one time step.

        Args:
            state: Current stream state
            idx: Current step index (unused)

        Returns:
            Tuple of (timestep, new_state)
        """
        result = self.step_result(state, idx)
        return result.timestep, cast(RandomWalkState, result.state)

    @functools.partial(jax.jit, static_argnums=(0,))
    def step_result(self, state: RandomWalkState, idx: Array) -> RandomWalkStepResult:
        """Stage and atomically commit one exact random-walk event."""

        self._require_state_contract(state)
        input_valid = _step_input_valid(idx)
        lifetime_counter_valid = _lifetime_counter_valid(state.step_words, state.step_count)
        state_valid = _tree_floating_arrays_finite(state)
        proposed_words, lifetime_capacity_available = _checked_lifetime_words_increment(
            state.step_words
        )
        key, k_drift, k_x, k_noise = jr.split(state.key, 4)

        drift = jr.normal(k_drift, state.true_weights.shape, dtype=jnp.float32)
        new_weights = state.true_weights + self._drift_rate * drift

        x = self._feature_std * jr.normal(k_x, (self._feature_dim,), dtype=jnp.float32)
        noise = self._noise_std * jr.normal(k_noise, (), dtype=jnp.float32)
        target = jnp.dot(new_weights, x) + noise

        target_array = jnp.atleast_1d(target)
        candidate_state = RandomWalkState(
            key=key,
            true_weights=new_weights,
            step_count=_saturating_int32_counter_increment(state.step_count),
            step_words=proposed_words,
        )
        candidate_state_valid = self.state_is_valid(candidate_state)
        output_valid = jnp.all(jnp.isfinite(x)) & jnp.all(jnp.isfinite(target_array))
        rem, secondary_rem, index, due, secondary_due = _no_schedule_diagnostics()
        return _commit_step_result(
            current_state=state,
            candidate_state=candidate_state,
            observation=x,
            target=target_array,
            pre_step_words=state.step_words,
            proposed_step_words=proposed_words,
            schedule_remainder=rem,
            secondary_schedule_remainder=secondary_rem,
            schedule_index=index,
            schedule_due=due,
            secondary_schedule_due=secondary_due,
            oracle_changed=jnp.asarray(self._drift_rate != 0.0, dtype=jnp.bool_),
            child_update_applied=jnp.asarray(True, dtype=jnp.bool_),
            child_counter_aligned=jnp.asarray(True, dtype=jnp.bool_),
            lifetime_counter_valid=lifetime_counter_valid,
            lifetime_capacity_available=lifetime_capacity_available,
            input_valid=input_valid,
            state_valid=state_valid,
            output_valid=output_valid,
            candidate_state_valid=candidate_state_valid,
        )


@chex.dataclass(frozen=True)
class HiddenStateAR2State:
    """AR memory, RNG state, and exact admitted-event identity."""

    key: PRNGKeyArray
    x_prev: Float[Array, " feature_dim"]
    x_prev2: Float[Array, " feature_dim"]
    step_count: Int[Array, ""] = None  # type: ignore[assignment]
    step_words: UInt[Array, " 2"] = None  # type: ignore[assignment]


class HiddenStateAR2Stream:
    """Stationary AR(2) stream with partially observable hidden channels.

    The emitted observation contains the full AR state so downstream feature
    construction can decide what to mask. The target includes visible linear
    terms plus a hidden interaction, making the hidden channels behaviorally
    relevant for Step 3 feature-discovery probes.
    """

    def __init__(
        self,
        feature_dim: int,
        visible_dim: int = 2,
        phi1: float = 0.6,
        phi2: float = -0.2,
        innovation_std: float = 0.2,
        nonlinear_coeff: float = 0.5,
        target_noise_std: float = 0.01,
    ):
        """Initialize the AR(2) stream.

        Each channel evolves as ``x_t = phi1*x_{t-1} + phi2*x_{t-2} + eps``.
        The target is the sum of the first ``visible_dim`` channels plus
        ``nonlinear_coeff * h0 * h1``, where ``h0``/``h1`` are the first two
        hidden channels — the pairwise product is what makes the hidden block
        behaviorally relevant to a feature-discovery probe.

        Args:
            feature_dim: Total number of AR channels emitted per step
            visible_dim: Channels counted linearly in the target; the rest
                form the hidden block (must contain at least two channels)
            phi1: AR lag-1 coefficient
            phi2: AR lag-2 coefficient; (phi1, phi2) must lie inside the
                AR(2) stationarity triangle, except for the deterministic
                copy case (phi1=1, phi2=0, innovation_std=0)
            innovation_std: Std dev of the per-channel AR innovation
            nonlinear_coeff: Weight on the hidden h0*h1 interaction term
            target_noise_std: Std dev of additive target noise
        """
        _require_positive_int32(feature_dim, name="feature_dim")
        _require_positive_int32(visible_dim, name="visible_dim")
        _require_finite_real(phi1, name="phi1")
        _require_finite_real(phi2, name="phi2")
        _require_finite_real(innovation_std, name="innovation_std", nonnegative=True)
        _require_finite_real(nonlinear_coeff, name="nonlinear_coeff")
        _require_finite_real(target_noise_std, name="target_noise_std", nonnegative=True)
        if visible_dim >= feature_dim:
            raise ValueError("visible_dim must be in [1, feature_dim)")
        if feature_dim - visible_dim < 2:
            raise ValueError("hidden block must contain at least two channels")
        deterministic_copy = phi1 == 1.0 and phi2 == 0.0 and innovation_std == 0.0
        violates_stationarity = phi1 + phi2 >= 1.0 or phi2 - phi1 >= 1.0 or abs(phi2) >= 1.0
        if violates_stationarity and not deterministic_copy:
            raise ValueError("AR(2) coefficients violate the stationarity triangle")
        self._feature_dim = feature_dim
        self._visible_dim = visible_dim
        self._phi1 = phi1
        self._phi2 = phi2
        self._innovation_std = innovation_std
        self._nonlinear_coeff = nonlinear_coeff
        self._target_noise_std = target_noise_std

    @property
    def feature_dim(self) -> int:
        """Return the dimension of observation vectors."""
        return self._feature_dim

    @property
    def visible_dim(self) -> int:
        """Return the number of visible channels."""
        return self._visible_dim

    @property
    def resource_budget(self) -> SyntheticStreamResourceBudget:
        return SyntheticStreamResourceBudget(
            stream_type=type(self).__name__,
            state_nbytes=measure_synthetic_stream_state_nbytes(self.init(jr.key(0))),
            exact_clock_delta_nbytes=SYNTHETIC_STREAM_NEW_CLOCK_DELTA_NBYTES,
        )

    def to_config(self) -> dict[str, Any]:
        return {
            "type": type(self).__name__,
            "config_schema": HIDDEN_STATE_AR2_CONFIG_SCHEMA,
            "state_schema": HIDDEN_STATE_AR2_STATE_SCHEMA,
            "feature_dim": self._feature_dim,
            "visible_dim": self._visible_dim,
            "phi1": float(self._phi1),
            "phi2": float(self._phi2),
            "innovation_std": float(self._innovation_std),
            "nonlinear_coeff": float(self._nonlinear_coeff),
            "target_noise_std": float(self._target_noise_std),
        }

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "HiddenStateAR2Stream":
        expected = {
            "type",
            "config_schema",
            "state_schema",
            "feature_dim",
            "visible_dim",
            "phi1",
            "phi2",
            "innovation_std",
            "nonlinear_coeff",
            "target_noise_std",
        }
        fields = _require_exact_fields(config, expected, label="hidden-state AR2 config")
        if fields.pop("type") != cls.__name__:
            raise ValueError("hidden-state AR2 config type is unsupported")
        if fields.pop("config_schema") != HIDDEN_STATE_AR2_CONFIG_SCHEMA:
            raise ValueError("hidden-state AR2 config schema is unsupported")
        if fields.pop("state_schema") != HIDDEN_STATE_AR2_STATE_SCHEMA:
            raise ValueError("hidden-state AR2 state schema is unsupported")
        return cls(**fields)

    def _require_state_contract(self, state: HiddenStateAR2State) -> None:
        _require_prng_key(state.key, name="hidden-state AR2 key")
        for name, value in (("x_prev", state.x_prev), ("x_prev2", state.x_prev2)):
            _require_array(
                value,
                name=f"hidden-state AR2 {name}",
                shape=(self._feature_dim,),
                dtype=jnp.dtype(jnp.float32),
            )
        _require_array(
            state.step_count,
            name="hidden-state AR2 step_count",
            shape=(),
            dtype=jnp.dtype(jnp.int32),
        )
        _require_array(
            state.step_words,
            name="hidden-state AR2 step_words",
            shape=(2,),
            dtype=jnp.dtype(jnp.uint32),
        )

    def state_is_valid(self, state: HiddenStateAR2State) -> Bool[Array, ""]:
        self._require_state_contract(state)
        return _lifetime_counter_valid(state.step_words, state.step_count) & (
            _tree_floating_arrays_finite(state)
        )

    def init(self, key: Array) -> HiddenStateAR2State:
        """Initialize AR state."""
        _require_prng_key(key, name="hidden-state AR2 init key")
        key, k1, k2 = jr.split(key, 3)
        x_prev = jr.normal(k1, (self._feature_dim,), dtype=jnp.float32)
        x_prev2 = jr.normal(k2, (self._feature_dim,), dtype=jnp.float32)
        return HiddenStateAR2State(
            key=key,
            x_prev=x_prev,
            x_prev2=x_prev2,
            step_count=jnp.asarray(0, dtype=jnp.int32),
            step_words=jnp.zeros((2,), dtype=jnp.uint32),
        )

    def step(
        self,
        state: HiddenStateAR2State,
        idx: Array,
    ) -> tuple[TimeStep, HiddenStateAR2State]:
        """Generate one AR(2) sample."""
        result = self.step_result(state, idx)
        return result.timestep, cast(HiddenStateAR2State, result.state)

    @functools.partial(jax.jit, static_argnums=(0,))
    def step_result(
        self,
        state: HiddenStateAR2State,
        idx: Array,
    ) -> HiddenStateAR2StepResult:
        """Stage and atomically commit one exact AR(2) event."""

        self._require_state_contract(state)
        input_valid = _step_input_valid(idx)
        lifetime_counter_valid = _lifetime_counter_valid(state.step_words, state.step_count)
        state_valid = _tree_floating_arrays_finite(state)
        proposed_words, lifetime_capacity_available = _checked_lifetime_words_increment(
            state.step_words
        )
        key, k_innov, k_noise = jr.split(state.key, 3)
        innovation = self._innovation_std * jr.normal(
            k_innov,
            (self._feature_dim,),
            dtype=jnp.float32,
        )
        x_t = self._phi1 * state.x_prev + self._phi2 * state.x_prev2 + innovation
        visible_term = jnp.sum(x_t[: self._visible_dim])
        h0 = x_t[self._visible_dim]
        h1 = x_t[self._visible_dim + 1]
        hidden_term = self._nonlinear_coeff * h0 * h1
        noise = self._target_noise_std * jr.normal(k_noise, (), dtype=jnp.float32)
        target = visible_term + hidden_term + noise
        target_array = jnp.atleast_1d(target)
        candidate_state = HiddenStateAR2State(
            key=key,
            x_prev=x_t,
            x_prev2=state.x_prev,
            step_count=_saturating_int32_counter_increment(state.step_count),
            step_words=proposed_words,
        )
        candidate_state_valid = self.state_is_valid(candidate_state)
        output_valid = jnp.all(jnp.isfinite(x_t)) & jnp.all(jnp.isfinite(target_array))
        rem, secondary_rem, index, due, secondary_due = _no_schedule_diagnostics()
        return _commit_step_result(
            current_state=state,
            candidate_state=candidate_state,
            observation=x_t,
            target=target_array,
            pre_step_words=state.step_words,
            proposed_step_words=proposed_words,
            schedule_remainder=rem,
            secondary_schedule_remainder=secondary_rem,
            schedule_index=index,
            schedule_due=due,
            secondary_schedule_due=secondary_due,
            oracle_changed=jnp.asarray(False, dtype=jnp.bool_),
            child_update_applied=jnp.asarray(True, dtype=jnp.bool_),
            child_counter_aligned=jnp.asarray(True, dtype=jnp.bool_),
            lifetime_counter_valid=lifetime_counter_valid,
            lifetime_capacity_available=lifetime_capacity_available,
            input_valid=input_valid,
            state_valid=state_valid,
            output_valid=output_valid,
            candidate_state_valid=candidate_state_valid,
        )


@chex.dataclass(frozen=True)
class AbruptChangeState:
    """State for AbruptChangeStream.

    Attributes:
        key: JAX random key for generating randomness
        true_weights: Current true target weights
        step_count: Saturating int32 compatibility telemetry
        step_words: Exact big-endian uint32 event identity
    """

    key: PRNGKeyArray
    true_weights: Float[Array, " feature_dim"]
    step_count: Int[Array, ""]
    step_words: UInt[Array, " 2"] = None  # type: ignore[assignment]


class AbruptChangeStream:
    """Non-stationary stream with sudden target weight changes.

    Target weights remain constant for a period, then abruptly change
    to new random values. Tests the learner's ability to detect and
    rapidly adapt to distribution shifts.

    Attributes:
        feature_dim: Dimension of observation vectors
        change_interval: Number of steps between weight changes
        noise_std: Standard deviation of observation noise
        feature_std: Standard deviation of features
    """

    def __init__(
        self,
        feature_dim: int,
        change_interval: int = 1000,
        noise_std: float = 0.1,
        feature_std: float = 1.0,
    ):
        """Initialize the abrupt change stream.

        Args:
            feature_dim: Dimension of feature vectors
            change_interval: Steps between abrupt weight changes
            noise_std: Std dev of target noise
            feature_std: Std dev of feature values
        """
        _require_positive_int32(feature_dim, name="feature_dim")
        _require_positive_int32(change_interval, name="change_interval")
        _require_finite_real(noise_std, name="noise_std", nonnegative=True)
        _require_finite_real(feature_std, name="feature_std", nonnegative=True)
        self._feature_dim = feature_dim
        self._change_interval = change_interval
        self._noise_std = noise_std
        self._feature_std = feature_std

    @property
    def feature_dim(self) -> int:
        """Return the dimension of observation vectors."""
        return self._feature_dim

    @property
    def resource_budget(self) -> SyntheticStreamResourceBudget:
        return SyntheticStreamResourceBudget(
            stream_type=type(self).__name__,
            state_nbytes=measure_synthetic_stream_state_nbytes(self.init(jr.key(0))),
            exact_clock_delta_nbytes=SYNTHETIC_STREAM_CLOCK_DELTA_NBYTES,
        )

    def to_config(self) -> dict[str, Any]:
        return {
            "type": type(self).__name__,
            "config_schema": ABRUPT_CHANGE_CONFIG_SCHEMA,
            "state_schema": ABRUPT_CHANGE_STATE_SCHEMA,
            "feature_dim": self._feature_dim,
            "change_interval": self._change_interval,
            "noise_std": float(self._noise_std),
            "feature_std": float(self._feature_std),
        }

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "AbruptChangeStream":
        expected = {
            "type",
            "config_schema",
            "state_schema",
            "feature_dim",
            "change_interval",
            "noise_std",
            "feature_std",
        }
        fields = _require_exact_fields(config, expected, label="abrupt-change config")
        if fields.pop("type") != cls.__name__:
            raise ValueError("abrupt-change config type is unsupported")
        if fields.pop("config_schema") != ABRUPT_CHANGE_CONFIG_SCHEMA:
            raise ValueError("abrupt-change config schema is unsupported")
        if fields.pop("state_schema") != ABRUPT_CHANGE_STATE_SCHEMA:
            raise ValueError("abrupt-change state schema is unsupported")
        return cls(**fields)

    def _require_state_contract(self, state: AbruptChangeState) -> None:
        _require_prng_key(state.key, name="abrupt-change key")
        _require_array(
            state.true_weights,
            name="abrupt-change true_weights",
            shape=(self._feature_dim,),
            dtype=jnp.dtype(jnp.float32),
        )
        _require_array(
            state.step_count,
            name="abrupt-change step_count",
            shape=(),
            dtype=jnp.dtype(jnp.int32),
        )
        _require_array(
            state.step_words,
            name="abrupt-change step_words",
            shape=(2,),
            dtype=jnp.dtype(jnp.uint32),
        )

    def state_is_valid(self, state: AbruptChangeState) -> Bool[Array, ""]:
        self._require_state_contract(state)
        return _lifetime_counter_valid(state.step_words, state.step_count) & (
            _tree_floating_arrays_finite(state)
        )

    def init(self, key: Array) -> AbruptChangeState:
        """Initialize stream state.

        Args:
            key: JAX random key

        Returns:
            Initial stream state
        """
        _require_prng_key(key, name="abrupt-change init key")
        key, subkey = jr.split(key)
        weights = jr.normal(subkey, (self._feature_dim,), dtype=jnp.float32)
        return AbruptChangeState(
            key=key,
            true_weights=weights,
            step_count=jnp.array(0, dtype=jnp.int32),
            step_words=jnp.zeros((2,), dtype=jnp.uint32),
        )

    def step(self, state: AbruptChangeState, idx: Array) -> tuple[TimeStep, AbruptChangeState]:
        """Generate one time step.

        Args:
            state: Current stream state
            idx: Current step index (unused)

        Returns:
            Tuple of (timestep, new_state)
        """
        result = self.step_result(state, idx)
        return result.timestep, cast(AbruptChangeState, result.state)

    @functools.partial(jax.jit, static_argnums=(0,))
    def step_result(
        self,
        state: AbruptChangeState,
        idx: Array,
    ) -> AbruptChangeStepResult:
        """Stage and atomically commit one exact abrupt-change event."""

        self._require_state_contract(state)
        input_valid = _step_input_valid(idx)
        lifetime_counter_valid = _lifetime_counter_valid(state.step_words, state.step_count)
        state_valid = _tree_floating_arrays_finite(state)
        proposed_words, lifetime_capacity_available = _checked_lifetime_words_increment(
            state.step_words
        )
        key, key_weights, key_x, key_noise = jr.split(state.key, 4)

        schedule_remainder = _lifetime_words_remainder(state.step_words, self._change_interval)
        should_change = schedule_remainder == jnp.asarray(0, dtype=jnp.uint32)

        # Generate new weights (always generated but only used if should_change)
        new_random_weights = jr.normal(key_weights, (self._feature_dim,), dtype=jnp.float32)

        # Use jnp.where to conditionally update weights (JIT-compatible)
        new_weights = jnp.where(should_change, new_random_weights, state.true_weights)

        x = self._feature_std * jr.normal(key_x, (self._feature_dim,), dtype=jnp.float32)

        noise = self._noise_std * jr.normal(key_noise, (), dtype=jnp.float32)
        target = jnp.dot(new_weights, x) + noise

        target_array = jnp.atleast_1d(target)
        candidate_state = AbruptChangeState(
            key=key,
            true_weights=new_weights,
            step_count=_saturating_int32_counter_increment(state.step_count),
            step_words=proposed_words,
        )
        candidate_state_valid = self.state_is_valid(candidate_state)
        output_valid = jnp.all(jnp.isfinite(x)) & jnp.all(jnp.isfinite(target_array))
        return _commit_step_result(
            current_state=state,
            candidate_state=candidate_state,
            observation=x,
            target=target_array,
            pre_step_words=state.step_words,
            proposed_step_words=proposed_words,
            schedule_remainder=schedule_remainder,
            secondary_schedule_remainder=jnp.asarray(0, dtype=jnp.uint32),
            schedule_index=jnp.asarray(0, dtype=jnp.int32),
            schedule_due=should_change,
            secondary_schedule_due=jnp.asarray(False, dtype=jnp.bool_),
            oracle_changed=should_change,
            child_update_applied=jnp.asarray(True, dtype=jnp.bool_),
            child_counter_aligned=jnp.asarray(True, dtype=jnp.bool_),
            lifetime_counter_valid=lifetime_counter_valid,
            lifetime_capacity_available=lifetime_capacity_available,
            input_valid=input_valid,
            state_valid=state_valid,
            output_valid=output_valid,
            candidate_state_valid=candidate_state_valid,
        )


@chex.dataclass(frozen=True)
class SuttonExperiment1State:
    """State for SuttonExperiment1Stream.

    Attributes:
        key: JAX random key for generating randomness
        signs: Signs (+1/-1) for the relevant inputs
        step_count: Saturating int32 compatibility telemetry
        step_words: Exact big-endian uint32 event identity
    """

    key: PRNGKeyArray
    signs: Float[Array, " num_relevant"]
    wt_irr: Float[Array, " num_irrelevant"]
    step_count: Int[Array, ""]
    step_words: UInt[Array, " 2"] = None  # type: ignore[assignment]


class SuttonExperiment1Stream:
    """Non-stationary stream modeled on Experiment 1 from Sutton 1992.

    At the defaults (``noise_std=0.0``, ``bias_drift_rate=0.0``) this
    reproduces the task from Sutton's IDBD paper:
    - 20 real-valued inputs drawn from N(0, 1)
    - Only first 5 inputs are relevant (weights are ±1)
    - Last 15 inputs are irrelevant (weights are 0)
    - Every change_interval steps, one of the 5 relevant signs is flipped

    Nonzero ``noise_std`` or ``bias_drift_rate`` are extensions beyond the
    paper: they add target noise and let the nominally irrelevant weights
    drift away from zero, respectively.

    Reference: Sutton, R.S. (1992). "Adapting Bias by Gradient Descent:
    An Incremental Version of Delta-Bar-Delta"

    Attributes:
        num_relevant: Number of relevant inputs (default 5)
        num_irrelevant: Number of irrelevant inputs (default 15)
        change_interval: Steps between sign changes (default 20)
    """

    def __init__(
        self,
        num_relevant: int = 5,
        num_irrelevant: int = 15,
        change_interval: int = 20,
        noise_std: float = 0.0,
        bias_drift_rate: float = 0.0,
    ):
        """Initialize the Sutton Experiment 1 stream.

        Args:
            num_relevant: Number of relevant inputs with ±1 weights
            num_irrelevant: Number of irrelevant inputs with 0 weights
            change_interval: Number of steps between sign flips
            noise_std: Std dev of additive target noise (0.0 = the
                noise-free paper task)
            bias_drift_rate: Per-step random-walk std dev applied to the
                irrelevant-input weights (0.0 = they stay exactly zero,
                as in the paper)
        """
        _require_positive_int32(num_relevant, name="num_relevant")
        _require_positive_int32(num_irrelevant, name="num_irrelevant", minimum=0)
        if num_relevant + num_irrelevant > _INT32_MAX:
            raise ValueError("total feature dimension must fit a positive int32")
        _require_positive_int32(change_interval, name="change_interval")
        _require_finite_real(noise_std, name="noise_std", nonnegative=True)
        _require_finite_real(bias_drift_rate, name="bias_drift_rate", nonnegative=True)
        self._num_relevant = num_relevant
        self._num_irrelevant = num_irrelevant
        self._change_interval = change_interval
        self._noise_std = noise_std
        self._bias_drift_rate = bias_drift_rate

    @property
    def feature_dim(self) -> int:
        """Return the dimension of observation vectors."""
        return self._num_relevant + self._num_irrelevant

    @property
    def resource_budget(self) -> SyntheticStreamResourceBudget:
        return SyntheticStreamResourceBudget(
            stream_type=type(self).__name__,
            state_nbytes=measure_synthetic_stream_state_nbytes(self.init(jr.key(0))),
            exact_clock_delta_nbytes=SYNTHETIC_STREAM_CLOCK_DELTA_NBYTES,
        )

    def to_config(self) -> dict[str, Any]:
        return {
            "type": type(self).__name__,
            "config_schema": SUTTON_EXPERIMENT1_CONFIG_SCHEMA,
            "state_schema": SUTTON_EXPERIMENT1_STATE_SCHEMA,
            "num_relevant": self._num_relevant,
            "num_irrelevant": self._num_irrelevant,
            "change_interval": self._change_interval,
            "noise_std": float(self._noise_std),
            "bias_drift_rate": float(self._bias_drift_rate),
        }

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "SuttonExperiment1Stream":
        expected = {
            "type",
            "config_schema",
            "state_schema",
            "num_relevant",
            "num_irrelevant",
            "change_interval",
            "noise_std",
            "bias_drift_rate",
        }
        fields = _require_exact_fields(config, expected, label="Sutton Experiment 1 config")
        if fields.pop("type") != cls.__name__:
            raise ValueError("Sutton Experiment 1 config type is unsupported")
        if fields.pop("config_schema") != SUTTON_EXPERIMENT1_CONFIG_SCHEMA:
            raise ValueError("Sutton Experiment 1 config schema is unsupported")
        if fields.pop("state_schema") != SUTTON_EXPERIMENT1_STATE_SCHEMA:
            raise ValueError("Sutton Experiment 1 state schema is unsupported")
        return cls(**fields)

    def _require_state_contract(self, state: SuttonExperiment1State) -> None:
        _require_prng_key(state.key, name="Sutton Experiment 1 key")
        _require_array(
            state.signs,
            name="Sutton Experiment 1 signs",
            shape=(self._num_relevant,),
            dtype=jnp.dtype(jnp.float32),
        )
        _require_array(
            state.wt_irr,
            name="Sutton Experiment 1 wt_irr",
            shape=(self._num_irrelevant,),
            dtype=jnp.dtype(jnp.float32),
        )
        _require_array(
            state.step_count,
            name="Sutton Experiment 1 step_count",
            shape=(),
            dtype=jnp.dtype(jnp.int32),
        )
        _require_array(
            state.step_words,
            name="Sutton Experiment 1 step_words",
            shape=(2,),
            dtype=jnp.dtype(jnp.uint32),
        )

    def state_is_valid(self, state: SuttonExperiment1State) -> Bool[Array, ""]:
        self._require_state_contract(state)
        signs_valid = jnp.all((state.signs == 1.0) | (state.signs == -1.0))
        return (
            _lifetime_counter_valid(state.step_words, state.step_count)
            & signs_valid
            & _tree_floating_arrays_finite(state)
        )

    def init(self, key: Array) -> SuttonExperiment1State:
        """Initialize stream state.

        Args:
            key: JAX random key

        Returns:
            Initial stream state with all +1 signs
        """
        _require_prng_key(key, name="Sutton Experiment 1 init key")
        signs = jnp.ones(self._num_relevant, dtype=jnp.float32)
        return SuttonExperiment1State(
            key=key,
            signs=signs,
            wt_irr=jnp.zeros(self._num_irrelevant, dtype=jnp.float32),
            step_count=jnp.array(0, dtype=jnp.int32),
            step_words=jnp.zeros((2,), dtype=jnp.uint32),
        )

    def step(
        self, state: SuttonExperiment1State, idx: Array
    ) -> tuple[TimeStep, SuttonExperiment1State]:
        """Generate one time step.

        At each step:
        1. If at a change interval (and not step 0), flip one random sign
        2. Generate random inputs from N(0, 1)
        3. Compute target as sum of relevant inputs weighted by signs

        Args:
            state: Current stream state
            idx: Current step index (unused)

        Returns:
            Tuple of (timestep, new_state)
        """
        result = self.step_result(state, idx)
        return result.timestep, cast(SuttonExperiment1State, result.state)

    @functools.partial(jax.jit, static_argnums=(0,))
    def step_result(
        self,
        state: SuttonExperiment1State,
        idx: Array,
    ) -> SuttonExperiment1StepResult:
        """Stage and atomically commit one exact Sutton Experiment 1 event."""

        self._require_state_contract(state)
        input_valid = _step_input_valid(idx)
        lifetime_counter_valid = _lifetime_counter_valid(state.step_words, state.step_count)
        signs_valid = jnp.all((state.signs == 1.0) | (state.signs == -1.0))
        state_valid = signs_valid & _tree_floating_arrays_finite(state)
        proposed_words, lifetime_capacity_available = _checked_lifetime_words_increment(
            state.step_words
        )
        key, key_x, key_which, key_irr, key_noise = jr.split(state.key, 5)

        # Determine if we should flip a sign (not at step 0)
        schedule_remainder = _lifetime_words_remainder(state.step_words, self._change_interval)
        should_flip = (~jnp.all(state.step_words == 0)) & (
            schedule_remainder == jnp.asarray(0, dtype=jnp.uint32)
        )

        idx_to_flip = jr.randint(key_which, (), 0, self._num_relevant)

        flip_mask = jnp.where(
            jnp.arange(self._num_relevant) == idx_to_flip,
            jnp.array(-1.0, dtype=jnp.float32),
            jnp.array(1.0, dtype=jnp.float32),
        )

        # Apply flip mask conditionally
        new_signs = jnp.where(should_flip, state.signs * flip_mask, state.signs)
        wt_irr = state.wt_irr + self._bias_drift_rate * jr.normal(
            key_irr,
            (self._num_irrelevant,),
            dtype=jnp.float32,
        )

        # Generate observation from N(0, 1)
        x = jr.normal(key_x, (self.feature_dim,), dtype=jnp.float32)

        # Compute target: sum of first num_relevant inputs weighted by signs
        target = jnp.dot(new_signs, x[: self._num_relevant])
        target = target + jnp.dot(wt_irr, x[self._num_relevant :])
        target = target + self._noise_std * jr.normal(key_noise, (), dtype=jnp.float32)

        target_array = jnp.atleast_1d(target)
        candidate_state = SuttonExperiment1State(
            key=key,
            signs=new_signs,
            wt_irr=wt_irr,
            step_count=_saturating_int32_counter_increment(state.step_count),
            step_words=proposed_words,
        )
        candidate_state_valid = self.state_is_valid(candidate_state)
        output_valid = jnp.all(jnp.isfinite(x)) & jnp.all(jnp.isfinite(target_array))
        return _commit_step_result(
            current_state=state,
            candidate_state=candidate_state,
            observation=x,
            target=target_array,
            pre_step_words=state.step_words,
            proposed_step_words=proposed_words,
            schedule_remainder=schedule_remainder,
            secondary_schedule_remainder=jnp.asarray(0, dtype=jnp.uint32),
            schedule_index=idx_to_flip.astype(jnp.int32),
            schedule_due=should_flip,
            secondary_schedule_due=jnp.asarray(False, dtype=jnp.bool_),
            oracle_changed=should_flip,
            child_update_applied=jnp.asarray(True, dtype=jnp.bool_),
            child_counter_aligned=jnp.asarray(True, dtype=jnp.bool_),
            lifetime_counter_valid=lifetime_counter_valid,
            lifetime_capacity_available=lifetime_capacity_available,
            input_valid=input_valid,
            state_valid=state_valid,
            output_valid=output_valid,
            candidate_state_valid=candidate_state_valid,
        )


@chex.dataclass(frozen=True)
class CyclicState:
    """State for CyclicStream.

    Attributes:
        key: JAX random key for generating randomness
        configurations: Pre-generated weight configurations
        step_count: Saturating int32 compatibility telemetry
        step_words: Exact big-endian uint32 event identity
    """

    key: PRNGKeyArray
    configurations: Float[Array, "num_configs feature_dim"]
    step_count: Int[Array, ""]
    step_words: UInt[Array, " 2"] = None  # type: ignore[assignment]


class CyclicStream:
    """Non-stationary stream that cycles between known weight configurations.

    Weights cycle through a fixed set of configurations. Tests whether
    the learner can re-adapt quickly to previously seen targets.

    Attributes:
        feature_dim: Dimension of observation vectors
        cycle_length: Number of steps per configuration before switching
        num_configurations: Number of weight configurations to cycle through
        noise_std: Standard deviation of observation noise
        feature_std: Standard deviation of features
    """

    def __init__(
        self,
        feature_dim: int,
        cycle_length: int = 500,
        num_configurations: int = 4,
        noise_std: float = 0.1,
        feature_std: float = 1.0,
    ):
        """Initialize the cyclic target stream.

        Args:
            feature_dim: Dimension of feature vectors
            cycle_length: Steps spent in each configuration
            num_configurations: Number of configurations to cycle through
            noise_std: Std dev of target noise
            feature_std: Std dev of feature values
        """
        _require_positive_int32(feature_dim, name="feature_dim")
        _require_positive_int32(cycle_length, name="cycle_length")
        _require_positive_int32(num_configurations, name="num_configurations")
        _require_finite_real(noise_std, name="noise_std", nonnegative=True)
        _require_finite_real(feature_std, name="feature_std", nonnegative=True)
        self._feature_dim = feature_dim
        self._cycle_length = cycle_length
        self._num_configurations = num_configurations
        self._noise_std = noise_std
        self._feature_std = feature_std

    @property
    def feature_dim(self) -> int:
        """Return the dimension of observation vectors."""
        return self._feature_dim

    @property
    def resource_budget(self) -> SyntheticStreamResourceBudget:
        return SyntheticStreamResourceBudget(
            stream_type=type(self).__name__,
            state_nbytes=measure_synthetic_stream_state_nbytes(self.init(jr.key(0))),
            exact_clock_delta_nbytes=SYNTHETIC_STREAM_CLOCK_DELTA_NBYTES,
        )

    def to_config(self) -> dict[str, Any]:
        return {
            "type": type(self).__name__,
            "config_schema": CYCLIC_CONFIG_SCHEMA,
            "state_schema": CYCLIC_STATE_SCHEMA,
            "feature_dim": self._feature_dim,
            "cycle_length": self._cycle_length,
            "num_configurations": self._num_configurations,
            "noise_std": float(self._noise_std),
            "feature_std": float(self._feature_std),
        }

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "CyclicStream":
        expected = {
            "type",
            "config_schema",
            "state_schema",
            "feature_dim",
            "cycle_length",
            "num_configurations",
            "noise_std",
            "feature_std",
        }
        fields = _require_exact_fields(config, expected, label="cyclic config")
        if fields.pop("type") != cls.__name__:
            raise ValueError("cyclic config type is unsupported")
        if fields.pop("config_schema") != CYCLIC_CONFIG_SCHEMA:
            raise ValueError("cyclic config schema is unsupported")
        if fields.pop("state_schema") != CYCLIC_STATE_SCHEMA:
            raise ValueError("cyclic state schema is unsupported")
        return cls(**fields)

    def _require_state_contract(self, state: CyclicState) -> None:
        _require_prng_key(state.key, name="cyclic key")
        _require_array(
            state.configurations,
            name="cyclic configurations",
            shape=(self._num_configurations, self._feature_dim),
            dtype=jnp.dtype(jnp.float32),
        )
        _require_array(
            state.step_count,
            name="cyclic step_count",
            shape=(),
            dtype=jnp.dtype(jnp.int32),
        )
        _require_array(
            state.step_words,
            name="cyclic step_words",
            shape=(2,),
            dtype=jnp.dtype(jnp.uint32),
        )

    def state_is_valid(self, state: CyclicState) -> Bool[Array, ""]:
        self._require_state_contract(state)
        return _lifetime_counter_valid(state.step_words, state.step_count) & (
            _tree_floating_arrays_finite(state)
        )

    def init(self, key: Array) -> CyclicState:
        """Initialize stream state.

        Args:
            key: JAX random key

        Returns:
            Initial stream state with pre-generated configurations
        """
        _require_prng_key(key, name="cyclic init key")
        key, key_configs = jr.split(key)
        configurations = jr.normal(
            key_configs,
            (self._num_configurations, self._feature_dim),
            dtype=jnp.float32,
        )
        return CyclicState(
            key=key,
            configurations=configurations,
            step_count=jnp.array(0, dtype=jnp.int32),
            step_words=jnp.zeros((2,), dtype=jnp.uint32),
        )

    def step(self, state: CyclicState, idx: Array) -> tuple[TimeStep, CyclicState]:
        """Generate one time step.

        Args:
            state: Current stream state
            idx: Current step index (unused)

        Returns:
            Tuple of (timestep, new_state)
        """
        result = self.step_result(state, idx)
        return result.timestep, cast(CyclicState, result.state)

    @functools.partial(jax.jit, static_argnums=(0,))
    def step_result(self, state: CyclicState, idx: Array) -> CyclicStepResult:
        """Stage and atomically commit one exact cyclic event."""

        self._require_state_contract(state)
        input_valid = _step_input_valid(idx)
        lifetime_counter_valid = _lifetime_counter_valid(state.step_words, state.step_count)
        state_valid = _tree_floating_arrays_finite(state)
        proposed_words, lifetime_capacity_available = _checked_lifetime_words_increment(
            state.step_words
        )
        key, key_x, key_noise = jr.split(state.key, 3)

        cycle_quotient, cycle_remainder = _divmod_lifetime_words(
            state.step_words, self._cycle_length
        )
        _completed_cycles, config_remainder = _divmod_lifetime_words(
            cycle_quotient, self._num_configurations
        )
        config_idx = config_remainder.astype(jnp.int32)
        true_weights = state.configurations[config_idx]

        x = self._feature_std * jr.normal(key_x, (self._feature_dim,), dtype=jnp.float32)

        noise = self._noise_std * jr.normal(key_noise, (), dtype=jnp.float32)
        target = jnp.dot(true_weights, x) + noise

        target_array = jnp.atleast_1d(target)
        candidate_state = CyclicState(
            key=key,
            configurations=state.configurations,
            step_count=_saturating_int32_counter_increment(state.step_count),
            step_words=proposed_words,
        )
        candidate_state_valid = self.state_is_valid(candidate_state)
        output_valid = jnp.all(jnp.isfinite(x)) & jnp.all(jnp.isfinite(target_array))
        schedule_due = cycle_remainder == jnp.asarray(0, dtype=jnp.uint32)
        oracle_changed = schedule_due & (~jnp.all(state.step_words == 0))
        return _commit_step_result(
            current_state=state,
            candidate_state=candidate_state,
            observation=x,
            target=target_array,
            pre_step_words=state.step_words,
            proposed_step_words=proposed_words,
            schedule_remainder=cycle_remainder,
            secondary_schedule_remainder=jnp.asarray(0, dtype=jnp.uint32),
            schedule_index=config_idx,
            schedule_due=schedule_due,
            secondary_schedule_due=jnp.asarray(False, dtype=jnp.bool_),
            oracle_changed=oracle_changed,
            child_update_applied=jnp.asarray(True, dtype=jnp.bool_),
            child_counter_aligned=jnp.asarray(True, dtype=jnp.bool_),
            lifetime_counter_valid=lifetime_counter_valid,
            lifetime_capacity_available=lifetime_capacity_available,
            input_valid=input_valid,
            state_valid=state_valid,
            output_valid=output_valid,
            candidate_state_valid=candidate_state_valid,
        )


@chex.dataclass(frozen=True)
class PeriodicChangeState:
    """State for PeriodicChangeStream.

    Attributes:
        key: JAX random key for generating randomness
        base_weights: Base target weights (center of oscillation)
        phases: Per-weight phase offsets
        step_count: Saturating int32 compatibility telemetry
        step_words: Exact big-endian uint32 event identity
    """

    key: PRNGKeyArray
    base_weights: Float[Array, " feature_dim"]
    phases: Float[Array, " feature_dim"]
    step_count: Int[Array, ""]
    step_words: UInt[Array, " 2"] = None  # type: ignore[assignment]


class PeriodicChangeStream:
    """Non-stationary stream where target weights oscillate sinusoidally.

    Target weights follow: w(t) = base + amplitude * sin(2π * t / period + phase)
    where each weight has a random phase offset for diversity.

    This tests the learner's ability to track predictable periodic changes,
    which is qualitatively different from random drift or abrupt changes.

    Attributes:
        feature_dim: Dimension of observation vectors
        period: Number of steps for one complete oscillation
        amplitude: Magnitude of weight oscillation
        noise_std: Standard deviation of observation noise
        feature_std: Standard deviation of features
    """

    def __init__(
        self,
        feature_dim: int,
        period: int = 1000,
        amplitude: float = 1.0,
        noise_std: float = 0.1,
        feature_std: float = 1.0,
    ):
        """Initialize the periodic change stream.

        Args:
            feature_dim: Dimension of feature vectors
            period: Steps for one complete oscillation cycle
            amplitude: Magnitude of weight oscillations around base
            noise_std: Std dev of target noise
            feature_std: Std dev of feature values
        """
        _require_positive_int32(feature_dim, name="feature_dim")
        _require_positive_int32(period, name="period")
        _require_finite_real(amplitude, name="amplitude", nonnegative=True)
        _require_finite_real(noise_std, name="noise_std", nonnegative=True)
        _require_finite_real(feature_std, name="feature_std", nonnegative=True)
        self._feature_dim = feature_dim
        self._period = period
        self._amplitude = amplitude
        self._noise_std = noise_std
        self._feature_std = feature_std

    @property
    def feature_dim(self) -> int:
        """Return the dimension of observation vectors."""
        return self._feature_dim

    @property
    def resource_budget(self) -> SyntheticStreamResourceBudget:
        return SyntheticStreamResourceBudget(
            stream_type=type(self).__name__,
            state_nbytes=measure_synthetic_stream_state_nbytes(self.init(jr.key(0))),
            exact_clock_delta_nbytes=SYNTHETIC_STREAM_CLOCK_DELTA_NBYTES,
        )

    def to_config(self) -> dict[str, Any]:
        return {
            "type": type(self).__name__,
            "config_schema": PERIODIC_CHANGE_CONFIG_SCHEMA,
            "state_schema": PERIODIC_CHANGE_STATE_SCHEMA,
            "feature_dim": self._feature_dim,
            "period": self._period,
            "amplitude": float(self._amplitude),
            "noise_std": float(self._noise_std),
            "feature_std": float(self._feature_std),
        }

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "PeriodicChangeStream":
        expected = {
            "type",
            "config_schema",
            "state_schema",
            "feature_dim",
            "period",
            "amplitude",
            "noise_std",
            "feature_std",
        }
        fields = _require_exact_fields(config, expected, label="periodic-change config")
        if fields.pop("type") != cls.__name__:
            raise ValueError("periodic-change config type is unsupported")
        if fields.pop("config_schema") != PERIODIC_CHANGE_CONFIG_SCHEMA:
            raise ValueError("periodic-change config schema is unsupported")
        if fields.pop("state_schema") != PERIODIC_CHANGE_STATE_SCHEMA:
            raise ValueError("periodic-change state schema is unsupported")
        return cls(**fields)

    def _require_state_contract(self, state: PeriodicChangeState) -> None:
        _require_prng_key(state.key, name="periodic-change key")
        for name, value in (("base_weights", state.base_weights), ("phases", state.phases)):
            _require_array(
                value,
                name=f"periodic-change {name}",
                shape=(self._feature_dim,),
                dtype=jnp.dtype(jnp.float32),
            )
        _require_array(
            state.step_count,
            name="periodic-change step_count",
            shape=(),
            dtype=jnp.dtype(jnp.int32),
        )
        _require_array(
            state.step_words,
            name="periodic-change step_words",
            shape=(2,),
            dtype=jnp.dtype(jnp.uint32),
        )

    def state_is_valid(self, state: PeriodicChangeState) -> Bool[Array, ""]:
        self._require_state_contract(state)
        phases_valid = jnp.all((state.phases >= 0.0) & (state.phases <= 2.0 * jnp.pi))
        return (
            _lifetime_counter_valid(state.step_words, state.step_count)
            & phases_valid
            & _tree_floating_arrays_finite(state)
        )

    def init(self, key: Array) -> PeriodicChangeState:
        """Initialize stream state.

        Args:
            key: JAX random key

        Returns:
            Initial stream state with random base weights and phases
        """
        _require_prng_key(key, name="periodic-change init key")
        key, key_weights, key_phases = jr.split(key, 3)
        base_weights = jr.normal(key_weights, (self._feature_dim,), dtype=jnp.float32)
        # Random phases in [0, 2π) for each weight
        phases = jr.uniform(
            key_phases,
            (self._feature_dim,),
            minval=0.0,
            maxval=2.0 * jnp.pi,
            dtype=jnp.float32,
        )
        return PeriodicChangeState(
            key=key,
            base_weights=base_weights,
            phases=phases,
            step_count=jnp.array(0, dtype=jnp.int32),
            step_words=jnp.zeros((2,), dtype=jnp.uint32),
        )

    def step(self, state: PeriodicChangeState, idx: Array) -> tuple[TimeStep, PeriodicChangeState]:
        """Generate one time step.

        Args:
            state: Current stream state
            idx: Current step index (unused)

        Returns:
            Tuple of (timestep, new_state)
        """
        result = self.step_result(state, idx)
        return result.timestep, cast(PeriodicChangeState, result.state)

    @functools.partial(jax.jit, static_argnums=(0,))
    def step_result(
        self,
        state: PeriodicChangeState,
        idx: Array,
    ) -> PeriodicChangeStepResult:
        """Stage and atomically commit one exactly periodic event."""

        self._require_state_contract(state)
        input_valid = _step_input_valid(idx)
        lifetime_counter_valid = _lifetime_counter_valid(state.step_words, state.step_count)
        phases_valid = jnp.all((state.phases >= 0.0) & (state.phases <= 2.0 * jnp.pi))
        state_valid = phases_valid & _tree_floating_arrays_finite(state)
        proposed_words, lifetime_capacity_available = _checked_lifetime_words_increment(
            state.step_words
        )
        key, key_x, key_noise = jr.split(state.key, 3)

        # Compute oscillating weights: w(t) = base + amplitude * sin(2π * t / period + phase)
        schedule_remainder = _lifetime_words_remainder(state.step_words, self._period)
        t = schedule_remainder.astype(jnp.float32)
        oscillation = self._amplitude * jnp.sin(2.0 * jnp.pi * t / self._period + state.phases)
        true_weights = state.base_weights + oscillation

        x = self._feature_std * jr.normal(key_x, (self._feature_dim,), dtype=jnp.float32)

        noise = self._noise_std * jr.normal(key_noise, (), dtype=jnp.float32)
        target = jnp.dot(true_weights, x) + noise

        target_array = jnp.atleast_1d(target)
        candidate_state = PeriodicChangeState(
            key=key,
            base_weights=state.base_weights,
            phases=state.phases,
            step_count=_saturating_int32_counter_increment(state.step_count),
            step_words=proposed_words,
        )
        candidate_state_valid = self.state_is_valid(candidate_state)
        output_valid = jnp.all(jnp.isfinite(x)) & jnp.all(jnp.isfinite(target_array))
        schedule_due = schedule_remainder == jnp.asarray(0, dtype=jnp.uint32)
        return _commit_step_result(
            current_state=state,
            candidate_state=candidate_state,
            observation=x,
            target=target_array,
            pre_step_words=state.step_words,
            proposed_step_words=proposed_words,
            schedule_remainder=schedule_remainder,
            secondary_schedule_remainder=jnp.asarray(0, dtype=jnp.uint32),
            schedule_index=jnp.asarray(0, dtype=jnp.int32),
            schedule_due=schedule_due,
            secondary_schedule_due=jnp.asarray(False, dtype=jnp.bool_),
            oracle_changed=jnp.asarray(self._amplitude != 0.0, dtype=jnp.bool_),
            child_update_applied=jnp.asarray(True, dtype=jnp.bool_),
            child_counter_aligned=jnp.asarray(True, dtype=jnp.bool_),
            lifetime_counter_valid=lifetime_counter_valid,
            lifetime_capacity_available=lifetime_capacity_available,
            input_valid=input_valid,
            state_valid=state_valid,
            output_valid=output_valid,
            candidate_state_valid=candidate_state_valid,
        )


@chex.dataclass(frozen=True)
class ScaledStreamState:
    """State for ScaledStreamWrapper.

    Attributes:
        inner_state: State of the wrapped stream
        step_count: Saturating int32 wrapper telemetry
        step_words: Exact big-endian wrapper event identity
    """

    inner_state: Any
    step_count: Int[Array, ""] = None  # type: ignore[assignment]
    step_words: UInt[Array, " 2"] = None  # type: ignore[assignment]


class ScaledStreamWrapper:
    """Wrapper that applies per-feature scaling to any stream's observations.

    This wrapper multiplies each feature of the observation by a corresponding
    scale factor. Useful for testing how learners handle features at different
    scales, which is important for understanding normalization benefits.

    Examples
    --------
    ```python
    stream = ScaledStreamWrapper(
        AbruptChangeStream(feature_dim=10, change_interval=1000),
        feature_scales=jnp.array([0.001, 0.01, 0.1, 1.0, 10.0,
                                  100.0, 1000.0, 0.001, 0.01, 0.1])
    )
    ```

    Attributes:
        inner_stream: The wrapped stream instance
        feature_scales: Per-feature scale factors (must match feature_dim)
    """

    def __init__(self, inner_stream: ScanStream[Any], feature_scales: Array):
        """Initialize the scaled stream wrapper.

        Args:
            inner_stream: Stream to wrap (must implement ScanStream protocol)
            feature_scales: Array of scale factors, one per feature. Must have
                shape (feature_dim,) matching the inner stream's feature_dim.

        Raises:
            ValueError: If feature_scales length doesn't match inner stream's feature_dim
        """
        self._inner_stream: ScanStream[Any] = inner_stream
        _require_positive_int32(inner_stream.feature_dim, name="inner_stream.feature_dim")
        self._feature_scales = jnp.asarray(feature_scales, dtype=jnp.float32)

        if self._feature_scales.shape != (inner_stream.feature_dim,):
            raise ValueError(
                f"feature_scales shape ({self._feature_scales.shape}) must match "
                f"({inner_stream.feature_dim},)"
            )
        if not bool(jax.device_get(jnp.all(jnp.isfinite(self._feature_scales)))):
            raise ValueError("feature_scales must be finite")

    @property
    def feature_dim(self) -> int:
        """Return the dimension of observation vectors."""
        return int(self._inner_stream.feature_dim)

    @property
    def inner_stream(self) -> ScanStream[Any]:
        """Return the wrapped stream."""
        return self._inner_stream

    @property
    def feature_scales(self) -> Array:
        """Return the per-feature scale factors."""
        return self._feature_scales

    @property
    def resource_budget(self) -> SyntheticStreamResourceBudget:
        inner_budget = getattr(self._inner_stream, "resource_budget", None)
        inner_clock_nbytes = getattr(inner_budget, "exact_clock_nbytes", 0)
        inner_clock_delta_nbytes = getattr(inner_budget, "exact_clock_delta_nbytes", 0)
        if type(inner_clock_nbytes) is not int or inner_clock_nbytes < 0:
            raise TypeError("inner stream exact_clock_nbytes accounting is invalid")
        if type(inner_clock_delta_nbytes) is not int or inner_clock_delta_nbytes < 0:
            raise TypeError("inner stream exact_clock_delta_nbytes accounting is invalid")
        return SyntheticStreamResourceBudget(
            stream_type=type(self).__name__,
            state_nbytes=measure_synthetic_stream_state_nbytes(self.init(jr.key(0))),
            exact_clock_nbytes=SYNTHETIC_STREAM_CLOCK_NBYTES + inner_clock_nbytes,
            exact_clock_delta_nbytes=(
                SYNTHETIC_STREAM_NEW_CLOCK_DELTA_NBYTES + inner_clock_delta_nbytes
            ),
        )

    def to_config(self) -> dict[str, Any]:
        """Serialize this wrapper and a supported versioned inner stream."""

        serializer = getattr(self._inner_stream, "to_config", None)
        if not callable(serializer):
            raise TypeError("inner_stream must provide to_config for wrapper serialization")
        return {
            "type": type(self).__name__,
            "config_schema": SCALED_STREAM_CONFIG_SCHEMA,
            "state_schema": SCALED_STREAM_STATE_SCHEMA,
            "inner_stream": serializer(),
            "feature_scales": [float(value) for value in self._feature_scales.tolist()],
        }

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "ScaledStreamWrapper":
        expected = {
            "type",
            "config_schema",
            "state_schema",
            "inner_stream",
            "feature_scales",
        }
        fields = _require_exact_fields(config, expected, label="scaled-stream config")
        if fields.pop("type") != cls.__name__:
            raise ValueError("scaled-stream config type is unsupported")
        if fields.pop("config_schema") != SCALED_STREAM_CONFIG_SCHEMA:
            raise ValueError("scaled-stream config schema is unsupported")
        if fields.pop("state_schema") != SCALED_STREAM_STATE_SCHEMA:
            raise ValueError("scaled-stream state schema is unsupported")
        inner_payload = fields.pop("inner_stream")
        if not isinstance(inner_payload, Mapping):
            raise TypeError("scaled-stream inner_stream config must be a mapping")
        inner_stream = synthetic_stream_from_config(inner_payload)
        return cls(inner_stream=inner_stream, **fields)

    def _require_state_contract(self, state: ScaledStreamState) -> None:
        _require_array(
            state.step_count,
            name="scaled-stream step_count",
            shape=(),
            dtype=jnp.dtype(jnp.int32),
        )
        _require_array(
            state.step_words,
            name="scaled-stream step_words",
            shape=(2,),
            dtype=jnp.dtype(jnp.uint32),
        )
        inner_require = getattr(self._inner_stream, "_require_state_contract", None)
        if callable(inner_require):
            inner_require(state.inner_state)

    def _inner_counter_aligned(self, state: ScaledStreamState) -> Bool[Array, ""]:
        inner_words = getattr(state.inner_state, "step_words", None)
        if inner_words is None:
            return jnp.asarray(True, dtype=jnp.bool_)
        return jnp.all(inner_words == state.step_words)

    def _inner_state_valid(self, state: ScaledStreamState) -> Bool[Array, ""]:
        validator = getattr(self._inner_stream, "state_is_valid", None)
        if callable(validator):
            return cast(Array, validator(state.inner_state))
        return _tree_floating_arrays_finite(state.inner_state)

    def state_is_valid(self, state: ScaledStreamState) -> Bool[Array, ""]:
        self._require_state_contract(state)
        return (
            _lifetime_counter_valid(state.step_words, state.step_count)
            & self._inner_counter_aligned(state)
            & self._inner_state_valid(state)
            & _tree_floating_arrays_finite(state)
        )

    def init(self, key: Array) -> ScaledStreamState:
        """Initialize stream state.

        Args:
            key: JAX random key

        Returns:
            Initial stream state wrapping the inner stream's state
        """
        _require_prng_key(key, name="scaled-stream init key")
        inner_state = self._inner_stream.init(key)
        return ScaledStreamState(
            inner_state=inner_state,
            step_count=jnp.asarray(0, dtype=jnp.int32),
            step_words=jnp.zeros((2,), dtype=jnp.uint32),
        )

    def step(self, state: ScaledStreamState, idx: Array) -> tuple[TimeStep, ScaledStreamState]:
        """Generate one time step with scaled observations.

        Args:
            state: Current stream state
            idx: Current step index

        Returns:
            Tuple of (timestep with scaled observation, new_state)
        """
        result = self.step_result(state, idx)
        return result.timestep, cast(ScaledStreamState, result.state)

    @functools.partial(jax.jit, static_argnums=(0,))
    def step_result(
        self,
        state: ScaledStreamState,
        idx: Array,
    ) -> ScaledStreamStepResult:
        """Stage the child and atomically commit the entire wrapped event."""

        self._require_state_contract(state)
        input_valid = _step_input_valid(idx)
        lifetime_counter_valid = _lifetime_counter_valid(state.step_words, state.step_count)
        child_pre_aligned = self._inner_counter_aligned(state)
        state_valid = (
            child_pre_aligned & self._inner_state_valid(state) & _tree_floating_arrays_finite(state)
        )
        proposed_words, lifetime_capacity_available = _checked_lifetime_words_increment(
            state.step_words
        )

        child_step_result = getattr(self._inner_stream, "step_result", None)
        if callable(child_step_result):
            child_result = child_step_result(state.inner_state, idx)
            timestep = child_result.timestep
            new_inner_state = child_result.state
            child_update_applied = cast(Array, child_result.update_applied)
        else:
            timestep, new_inner_state = self._inner_stream.step(state.inner_state, idx)
            child_update_applied = jnp.asarray(True, dtype=jnp.bool_)

        inner_words = getattr(new_inner_state, "step_words", None)
        child_post_aligned = (
            jnp.asarray(True, dtype=jnp.bool_)
            if inner_words is None
            else jnp.all(inner_words == proposed_words)
        )
        child_counter_aligned = child_pre_aligned & child_post_aligned
        scaled_observation = timestep.observation * self._feature_scales
        target = jnp.asarray(timestep.target)
        candidate_state = ScaledStreamState(
            inner_state=new_inner_state,
            step_count=_saturating_int32_counter_increment(state.step_count),
            step_words=proposed_words,
        )
        candidate_state_valid = self.state_is_valid(candidate_state)
        output_valid = (
            jnp.all(jnp.isfinite(scaled_observation))
            & jnp.all(jnp.isfinite(target))
            & (scaled_observation.shape == (self.feature_dim,))
            & (target.shape == (1,))
        )
        rem, secondary_rem, index, due, secondary_due = _no_schedule_diagnostics()
        return _commit_step_result(
            current_state=state,
            candidate_state=candidate_state,
            observation=scaled_observation,
            target=target,
            pre_step_words=state.step_words,
            proposed_step_words=proposed_words,
            schedule_remainder=rem,
            secondary_schedule_remainder=secondary_rem,
            schedule_index=index,
            schedule_due=due,
            secondary_schedule_due=secondary_due,
            oracle_changed=jnp.asarray(False, dtype=jnp.bool_),
            child_update_applied=child_update_applied,
            child_counter_aligned=child_counter_aligned,
            lifetime_counter_valid=lifetime_counter_valid,
            lifetime_capacity_available=lifetime_capacity_available,
            input_valid=input_valid,
            state_valid=state_valid,
            output_valid=output_valid,
            candidate_state_valid=candidate_state_valid,
        )


def make_scale_range(
    feature_dim: int,
    min_scale: float = 0.001,
    max_scale: float = 1000.0,
    log_spaced: bool = True,
) -> Array:
    """Create a per-feature scale array spanning a range.

    Utility function to generate scale factors for ScaledStreamWrapper.

    Args:
        feature_dim: Number of features
        min_scale: Minimum scale factor
        max_scale: Maximum scale factor
        log_spaced: If True, scales are logarithmically spaced (default).
            If False, scales are linearly spaced.

    Returns:
        Array of shape (feature_dim,) with scale factors

    Examples
    --------
    ```python
    scales = make_scale_range(10, min_scale=0.01, max_scale=100.0)
    stream = ScaledStreamWrapper(RandomWalkStream(10), scales)
    ```
    """
    _require_positive_int32(feature_dim, name="feature_dim")
    _require_finite_real(min_scale, name="min_scale")
    _require_finite_real(max_scale, name="max_scale")
    if float(max_scale) < float(min_scale):
        raise ValueError("max_scale must be greater than or equal to min_scale")
    if type(log_spaced) is not bool:
        raise TypeError("log_spaced must be bool")
    if log_spaced:
        if float(min_scale) <= 0.0 or float(max_scale) <= 0.0:
            raise ValueError("log-spaced min_scale and max_scale must be positive")
        return jnp.logspace(
            jnp.log10(min_scale),
            jnp.log10(max_scale),
            feature_dim,
            dtype=jnp.float32,
        )
    else:
        return jnp.linspace(min_scale, max_scale, feature_dim, dtype=jnp.float32)


@chex.dataclass(frozen=True)
class DynamicScaleShiftState:
    """State for DynamicScaleShiftStream.

    Attributes:
        key: JAX random key for generating randomness
        true_weights: Current true target weights
        current_scales: Current per-feature scaling factors
        step_count: Saturating int32 compatibility telemetry
        step_words: Exact big-endian uint32 event identity
    """

    key: PRNGKeyArray
    true_weights: Float[Array, " feature_dim"]
    current_scales: Float[Array, " feature_dim"]
    step_count: Int[Array, ""]
    step_words: UInt[Array, " 2"] = None  # type: ignore[assignment]


class DynamicScaleShiftStream:
    """Non-stationary stream with abruptly changing feature scales.

    Both target weights AND feature scales change at specified intervals.
    This tests whether an online normalizer (e.g. :class:`EMANormalizer`)
    can track scale shifts faster than Autostep's internal v_i adaptation.

    The target is computed from unscaled features to maintain consistent
    difficulty across scale changes (only the feature representation changes,
    not the underlying prediction task).

    Attributes:
        feature_dim: Dimension of observation vectors
        scale_change_interval: Steps between scale changes
        weight_change_interval: Steps between weight changes
        min_scale: Minimum scale factor
        max_scale: Maximum scale factor
        noise_std: Standard deviation of observation noise
    """

    def __init__(
        self,
        feature_dim: int,
        scale_change_interval: int = 2000,
        weight_change_interval: int = 1000,
        min_scale: float = 0.01,
        max_scale: float = 100.0,
        noise_std: float = 0.1,
    ):
        """Initialize the dynamic scale shift stream.

        Args:
            feature_dim: Dimension of feature vectors
            scale_change_interval: Steps between abrupt scale changes
            weight_change_interval: Steps between abrupt weight changes
            min_scale: Minimum scale factor (log-uniform sampling)
            max_scale: Maximum scale factor (log-uniform sampling)
            noise_std: Std dev of target noise
        """
        _require_positive_int32(feature_dim, name="feature_dim")
        _require_positive_int32(scale_change_interval, name="scale_change_interval")
        _require_positive_int32(weight_change_interval, name="weight_change_interval")
        _require_finite_real(min_scale, name="min_scale", positive=True)
        _require_finite_real(max_scale, name="max_scale", positive=True)
        if float(max_scale) < float(min_scale):
            raise ValueError("max_scale must be greater than or equal to min_scale")
        _require_finite_real(noise_std, name="noise_std", nonnegative=True)
        self._feature_dim = feature_dim
        self._scale_change_interval = scale_change_interval
        self._weight_change_interval = weight_change_interval
        self._min_scale = min_scale
        self._max_scale = max_scale
        self._noise_std = noise_std

    @property
    def feature_dim(self) -> int:
        """Return the dimension of observation vectors."""
        return self._feature_dim

    @property
    def resource_budget(self) -> SyntheticStreamResourceBudget:
        return SyntheticStreamResourceBudget(
            stream_type=type(self).__name__,
            state_nbytes=measure_synthetic_stream_state_nbytes(self.init(jr.key(0))),
            exact_clock_delta_nbytes=SYNTHETIC_STREAM_CLOCK_DELTA_NBYTES,
        )

    def to_config(self) -> dict[str, Any]:
        return {
            "type": type(self).__name__,
            "config_schema": DYNAMIC_SCALE_SHIFT_CONFIG_SCHEMA,
            "state_schema": DYNAMIC_SCALE_SHIFT_STATE_SCHEMA,
            "feature_dim": self._feature_dim,
            "scale_change_interval": self._scale_change_interval,
            "weight_change_interval": self._weight_change_interval,
            "min_scale": float(self._min_scale),
            "max_scale": float(self._max_scale),
            "noise_std": float(self._noise_std),
        }

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "DynamicScaleShiftStream":
        expected = {
            "type",
            "config_schema",
            "state_schema",
            "feature_dim",
            "scale_change_interval",
            "weight_change_interval",
            "min_scale",
            "max_scale",
            "noise_std",
        }
        fields = _require_exact_fields(config, expected, label="dynamic-scale-shift config")
        if fields.pop("type") != cls.__name__:
            raise ValueError("dynamic-scale-shift config type is unsupported")
        if fields.pop("config_schema") != DYNAMIC_SCALE_SHIFT_CONFIG_SCHEMA:
            raise ValueError("dynamic-scale-shift config schema is unsupported")
        if fields.pop("state_schema") != DYNAMIC_SCALE_SHIFT_STATE_SCHEMA:
            raise ValueError("dynamic-scale-shift state schema is unsupported")
        return cls(**fields)

    def _require_state_contract(self, state: DynamicScaleShiftState) -> None:
        _require_prng_key(state.key, name="dynamic-scale-shift key")
        for name, value in (
            ("true_weights", state.true_weights),
            ("current_scales", state.current_scales),
        ):
            _require_array(
                value,
                name=f"dynamic-scale-shift {name}",
                shape=(self._feature_dim,),
                dtype=jnp.dtype(jnp.float32),
            )
        _require_array(
            state.step_count,
            name="dynamic-scale-shift step_count",
            shape=(),
            dtype=jnp.dtype(jnp.int32),
        )
        _require_array(
            state.step_words,
            name="dynamic-scale-shift step_words",
            shape=(2,),
            dtype=jnp.dtype(jnp.uint32),
        )

    def state_is_valid(self, state: DynamicScaleShiftState) -> Bool[Array, ""]:
        self._require_state_contract(state)
        scales_valid = jnp.all(
            (state.current_scales >= self._min_scale) & (state.current_scales <= self._max_scale)
        )
        return (
            _lifetime_counter_valid(state.step_words, state.step_count)
            & scales_valid
            & _tree_floating_arrays_finite(state)
        )

    def init(self, key: Array) -> DynamicScaleShiftState:
        """Initialize stream state.

        Args:
            key: JAX random key

        Returns:
            Initial stream state with random weights and scales
        """
        _require_prng_key(key, name="dynamic-scale-shift init key")
        key, k_weights, k_scales = jr.split(key, 3)
        weights = jr.normal(k_weights, (self._feature_dim,), dtype=jnp.float32)
        # Initial scales: log-uniform between min and max
        log_scales = jr.uniform(
            k_scales,
            (self._feature_dim,),
            minval=jnp.log(self._min_scale),
            maxval=jnp.log(self._max_scale),
        )
        scales = jnp.exp(log_scales).astype(jnp.float32)
        return DynamicScaleShiftState(
            key=key,
            true_weights=weights,
            current_scales=scales,
            step_count=jnp.array(0, dtype=jnp.int32),
            step_words=jnp.zeros((2,), dtype=jnp.uint32),
        )

    def step(
        self, state: DynamicScaleShiftState, idx: Array
    ) -> tuple[TimeStep, DynamicScaleShiftState]:
        """Generate one time step.

        Args:
            state: Current stream state
            idx: Current step index (unused)

        Returns:
            Tuple of (timestep, new_state)
        """
        result = self.step_result(state, idx)
        return result.timestep, cast(DynamicScaleShiftState, result.state)

    @functools.partial(jax.jit, static_argnums=(0,))
    def step_result(
        self,
        state: DynamicScaleShiftState,
        idx: Array,
    ) -> DynamicScaleShiftStepResult:
        """Stage and atomically commit one exact scale/weight-shift event."""

        self._require_state_contract(state)
        input_valid = _step_input_valid(idx)
        lifetime_counter_valid = _lifetime_counter_valid(state.step_words, state.step_count)
        scales_valid = jnp.all(
            (state.current_scales >= self._min_scale) & (state.current_scales <= self._max_scale)
        )
        state_valid = scales_valid & _tree_floating_arrays_finite(state)
        proposed_words, lifetime_capacity_available = _checked_lifetime_words_increment(
            state.step_words
        )
        key, k_weights, k_scales, k_x, k_noise = jr.split(state.key, 5)

        scale_remainder = _lifetime_words_remainder(state.step_words, self._scale_change_interval)
        should_change_scales = scale_remainder == jnp.asarray(0, dtype=jnp.uint32)
        new_log_scales = jr.uniform(
            k_scales,
            (self._feature_dim,),
            minval=jnp.log(self._min_scale),
            maxval=jnp.log(self._max_scale),
        )
        new_random_scales = jnp.exp(new_log_scales).astype(jnp.float32)
        new_scales = jnp.where(should_change_scales, new_random_scales, state.current_scales)

        weight_remainder = _lifetime_words_remainder(state.step_words, self._weight_change_interval)
        should_change_weights = weight_remainder == jnp.asarray(0, dtype=jnp.uint32)
        new_random_weights = jr.normal(k_weights, (self._feature_dim,), dtype=jnp.float32)
        new_weights = jnp.where(should_change_weights, new_random_weights, state.true_weights)

        raw_x = jr.normal(k_x, (self._feature_dim,), dtype=jnp.float32)

        x = raw_x * new_scales

        # Target from true weights using RAW features (for consistent difficulty)
        noise = self._noise_std * jr.normal(k_noise, (), dtype=jnp.float32)
        target = jnp.dot(new_weights, raw_x) + noise

        target_array = jnp.atleast_1d(target)
        candidate_state = DynamicScaleShiftState(
            key=key,
            true_weights=new_weights,
            current_scales=new_scales,
            step_count=_saturating_int32_counter_increment(state.step_count),
            step_words=proposed_words,
        )
        candidate_state_valid = self.state_is_valid(candidate_state)
        output_valid = jnp.all(jnp.isfinite(x)) & jnp.all(jnp.isfinite(target_array))
        return _commit_step_result(
            current_state=state,
            candidate_state=candidate_state,
            observation=x,
            target=target_array,
            pre_step_words=state.step_words,
            proposed_step_words=proposed_words,
            schedule_remainder=scale_remainder,
            secondary_schedule_remainder=weight_remainder,
            schedule_index=jnp.asarray(0, dtype=jnp.int32),
            schedule_due=should_change_scales,
            secondary_schedule_due=should_change_weights,
            oracle_changed=should_change_scales | should_change_weights,
            child_update_applied=jnp.asarray(True, dtype=jnp.bool_),
            child_counter_aligned=jnp.asarray(True, dtype=jnp.bool_),
            lifetime_counter_valid=lifetime_counter_valid,
            lifetime_capacity_available=lifetime_capacity_available,
            input_valid=input_valid,
            state_valid=state_valid,
            output_valid=output_valid,
            candidate_state_valid=candidate_state_valid,
        )


@chex.dataclass(frozen=True)
class ScaleDriftState:
    """State for ScaleDriftStream.

    Attributes:
        key: JAX random key for generating randomness
        true_weights: Current true target weights
        log_scales: Current log-scale factors (random walk on log-scale)
        step_count: Saturating int32 compatibility telemetry
        step_words: Exact big-endian uint32 event identity
    """

    key: PRNGKeyArray
    true_weights: Float[Array, " feature_dim"]
    log_scales: Float[Array, " feature_dim"]
    step_count: Int[Array, ""]
    step_words: UInt[Array, " 2"] = None  # type: ignore[assignment]


class ScaleDriftStream:
    """Non-stationary stream where feature scales drift via random walk.

    Both target weights and feature scales drift continuously. Weights drift
    in linear space while scales drift in log-space (bounded random walk).
    This tests continuous scale tracking where :class:`EMANormalizer`'s
    exponential moving statistics may adapt differently than Autostep's v_i.

    The target is computed from unscaled features to maintain consistent
    difficulty across scale changes.

    Attributes:
        feature_dim: Dimension of observation vectors
        weight_drift_rate: Std dev of weight drift per step
        scale_drift_rate: Std dev of log-scale drift per step
        min_log_scale: Minimum log-scale (clips random walk)
        max_log_scale: Maximum log-scale (clips random walk)
        noise_std: Standard deviation of observation noise
    """

    def __init__(
        self,
        feature_dim: int,
        weight_drift_rate: float = 0.001,
        scale_drift_rate: float = 0.01,
        min_log_scale: float = -4.0,  # exp(-4) ~ 0.018
        max_log_scale: float = 4.0,  # exp(4) ~ 54.6
        noise_std: float = 0.1,
    ):
        """Initialize the scale drift stream.

        Args:
            feature_dim: Dimension of feature vectors
            weight_drift_rate: Std dev of weight drift per step
            scale_drift_rate: Std dev of log-scale drift per step
            min_log_scale: Minimum log-scale (clips drift)
            max_log_scale: Maximum log-scale (clips drift)
            noise_std: Std dev of target noise
        """
        _require_positive_int32(feature_dim, name="feature_dim")
        _require_finite_real(weight_drift_rate, name="weight_drift_rate", nonnegative=True)
        _require_finite_real(scale_drift_rate, name="scale_drift_rate", nonnegative=True)
        _require_finite_real(min_log_scale, name="min_log_scale")
        _require_finite_real(max_log_scale, name="max_log_scale")
        if float(max_log_scale) < float(min_log_scale):
            raise ValueError("max_log_scale must be greater than or equal to min_log_scale")
        _require_finite_real(noise_std, name="noise_std", nonnegative=True)
        self._feature_dim = feature_dim
        self._weight_drift_rate = weight_drift_rate
        self._scale_drift_rate = scale_drift_rate
        self._min_log_scale = min_log_scale
        self._max_log_scale = max_log_scale
        self._noise_std = noise_std

    @property
    def feature_dim(self) -> int:
        """Return the dimension of observation vectors."""
        return self._feature_dim

    @property
    def resource_budget(self) -> SyntheticStreamResourceBudget:
        return SyntheticStreamResourceBudget(
            stream_type=type(self).__name__,
            state_nbytes=measure_synthetic_stream_state_nbytes(self.init(jr.key(0))),
            exact_clock_delta_nbytes=SYNTHETIC_STREAM_CLOCK_DELTA_NBYTES,
        )

    def to_config(self) -> dict[str, Any]:
        return {
            "type": type(self).__name__,
            "config_schema": SCALE_DRIFT_CONFIG_SCHEMA,
            "state_schema": SCALE_DRIFT_STATE_SCHEMA,
            "feature_dim": self._feature_dim,
            "weight_drift_rate": float(self._weight_drift_rate),
            "scale_drift_rate": float(self._scale_drift_rate),
            "min_log_scale": float(self._min_log_scale),
            "max_log_scale": float(self._max_log_scale),
            "noise_std": float(self._noise_std),
        }

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "ScaleDriftStream":
        expected = {
            "type",
            "config_schema",
            "state_schema",
            "feature_dim",
            "weight_drift_rate",
            "scale_drift_rate",
            "min_log_scale",
            "max_log_scale",
            "noise_std",
        }
        fields = _require_exact_fields(config, expected, label="scale-drift config")
        if fields.pop("type") != cls.__name__:
            raise ValueError("scale-drift config type is unsupported")
        if fields.pop("config_schema") != SCALE_DRIFT_CONFIG_SCHEMA:
            raise ValueError("scale-drift config schema is unsupported")
        if fields.pop("state_schema") != SCALE_DRIFT_STATE_SCHEMA:
            raise ValueError("scale-drift state schema is unsupported")
        return cls(**fields)

    def _require_state_contract(self, state: ScaleDriftState) -> None:
        _require_prng_key(state.key, name="scale-drift key")
        for name, value in (
            ("true_weights", state.true_weights),
            ("log_scales", state.log_scales),
        ):
            _require_array(
                value,
                name=f"scale-drift {name}",
                shape=(self._feature_dim,),
                dtype=jnp.dtype(jnp.float32),
            )
        _require_array(
            state.step_count,
            name="scale-drift step_count",
            shape=(),
            dtype=jnp.dtype(jnp.int32),
        )
        _require_array(
            state.step_words,
            name="scale-drift step_words",
            shape=(2,),
            dtype=jnp.dtype(jnp.uint32),
        )

    def state_is_valid(self, state: ScaleDriftState) -> Bool[Array, ""]:
        self._require_state_contract(state)
        scales_valid = jnp.all(
            (state.log_scales >= self._min_log_scale) & (state.log_scales <= self._max_log_scale)
        )
        return (
            _lifetime_counter_valid(state.step_words, state.step_count)
            & scales_valid
            & _tree_floating_arrays_finite(state)
        )

    def init(self, key: Array) -> ScaleDriftState:
        """Initialize stream state.

        Args:
            key: JAX random key

        Returns:
            Initial stream state with random weights and unit scales
        """
        _require_prng_key(key, name="scale-drift init key")
        key, k_weights = jr.split(key)
        weights = jr.normal(k_weights, (self._feature_dim,), dtype=jnp.float32)
        # Initial log-scales at 0 (scale = 1)
        log_scales = jnp.zeros(self._feature_dim, dtype=jnp.float32)
        return ScaleDriftState(
            key=key,
            true_weights=weights,
            log_scales=log_scales,
            step_count=jnp.array(0, dtype=jnp.int32),
            step_words=jnp.zeros((2,), dtype=jnp.uint32),
        )

    def step(self, state: ScaleDriftState, idx: Array) -> tuple[TimeStep, ScaleDriftState]:
        """Generate one time step.

        Args:
            state: Current stream state
            idx: Current step index (unused)

        Returns:
            Tuple of (timestep, new_state)
        """
        result = self.step_result(state, idx)
        return result.timestep, cast(ScaleDriftState, result.state)

    @functools.partial(jax.jit, static_argnums=(0,))
    def step_result(self, state: ScaleDriftState, idx: Array) -> ScaleDriftStepResult:
        """Stage and atomically commit one exact continuous scale-drift event."""

        self._require_state_contract(state)
        input_valid = _step_input_valid(idx)
        lifetime_counter_valid = _lifetime_counter_valid(state.step_words, state.step_count)
        scales_valid = jnp.all(
            (state.log_scales >= self._min_log_scale) & (state.log_scales <= self._max_log_scale)
        )
        state_valid = scales_valid & _tree_floating_arrays_finite(state)
        proposed_words, lifetime_capacity_available = _checked_lifetime_words_increment(
            state.step_words
        )
        key, k_w_drift, k_s_drift, k_x, k_noise = jr.split(state.key, 5)

        weight_drift = self._weight_drift_rate * jr.normal(
            k_w_drift, (self._feature_dim,), dtype=jnp.float32
        )
        new_weights = state.true_weights + weight_drift

        # Drift log-scales (bounded random walk)
        scale_drift = self._scale_drift_rate * jr.normal(
            k_s_drift, (self._feature_dim,), dtype=jnp.float32
        )
        new_log_scales = state.log_scales + scale_drift
        new_log_scales = jnp.clip(new_log_scales, self._min_log_scale, self._max_log_scale)

        raw_x = jr.normal(k_x, (self._feature_dim,), dtype=jnp.float32)

        scales = jnp.exp(new_log_scales)
        x = raw_x * scales

        # Target from true weights using RAW features
        noise = self._noise_std * jr.normal(k_noise, (), dtype=jnp.float32)
        target = jnp.dot(new_weights, raw_x) + noise

        target_array = jnp.atleast_1d(target)
        candidate_state = ScaleDriftState(
            key=key,
            true_weights=new_weights,
            log_scales=new_log_scales,
            step_count=_saturating_int32_counter_increment(state.step_count),
            step_words=proposed_words,
        )
        candidate_state_valid = self.state_is_valid(candidate_state)
        output_valid = jnp.all(jnp.isfinite(x)) & jnp.all(jnp.isfinite(target_array))
        rem, secondary_rem, index, due, secondary_due = _no_schedule_diagnostics()
        return _commit_step_result(
            current_state=state,
            candidate_state=candidate_state,
            observation=x,
            target=target_array,
            pre_step_words=state.step_words,
            proposed_step_words=proposed_words,
            schedule_remainder=rem,
            secondary_schedule_remainder=secondary_rem,
            schedule_index=index,
            schedule_due=due,
            secondary_schedule_due=secondary_due,
            oracle_changed=jnp.asarray(
                self._weight_drift_rate != 0.0 or self._scale_drift_rate != 0.0,
                dtype=jnp.bool_,
            ),
            child_update_applied=jnp.asarray(True, dtype=jnp.bool_),
            child_counter_aligned=jnp.asarray(True, dtype=jnp.bool_),
            lifetime_counter_valid=lifetime_counter_valid,
            lifetime_capacity_available=lifetime_capacity_available,
            input_valid=input_valid,
            state_valid=state_valid,
            output_valid=output_valid,
            candidate_state_valid=candidate_state_valid,
        )


def synthetic_stream_clock_nbytes() -> int:
    """Return bytes owned by saturating telemetry plus exact identity."""

    return SYNTHETIC_STREAM_CLOCK_NBYTES


def measure_synthetic_stream_state_nbytes(state: Any) -> int:
    """Measure every persistent JAX-array byte in one concrete stream state."""

    return sum(
        int(leaf.size) * int(leaf.dtype.itemsize)
        for leaf in jax.tree.leaves(state)
        if isinstance(leaf, Array)
    )


def synthetic_stream_from_config(config: Mapping[str, Any]) -> ScanStream[Any]:
    """Strictly reconstruct one supported versioned synthetic stream."""

    if set(config).isdisjoint({"type"}):
        raise ValueError("synthetic stream config requires a type field")
    stream_type = config["type"]
    stream_types: dict[str, type[Any]] = {
        "RandomWalkStream": RandomWalkStream,
        "HiddenStateAR2Stream": HiddenStateAR2Stream,
        "AbruptChangeStream": AbruptChangeStream,
        "SuttonExperiment1Stream": SuttonExperiment1Stream,
        "CyclicStream": CyclicStream,
        "PeriodicChangeStream": PeriodicChangeStream,
        "ScaledStreamWrapper": ScaledStreamWrapper,
        "DynamicScaleShiftStream": DynamicScaleShiftStream,
        "ScaleDriftStream": ScaleDriftStream,
    }
    if not isinstance(stream_type, str) or stream_type not in stream_types:
        raise ValueError("synthetic stream config type is unsupported")
    return cast(ScanStream[Any], stream_types[stream_type].from_config(config))


def _migrate_clocked_state(
    legacy_state: Any,
    *,
    stream: Any,
    state_type: type[Any],
    expected: set[str],
    label: str,
) -> Any:
    fields = _host_state_fields(legacy_state, label=label)
    fields = _require_exact_fields(fields, expected, label=f"legacy {label} state")
    count = _legacy_unsaturated_count(fields, label=label)
    fields["step_words"] = jnp.asarray((0, count), dtype=jnp.uint32)
    migrated = state_type(**fields)
    stream._require_state_contract(migrated)
    if not bool(jax.device_get(stream.state_is_valid(migrated))):
        raise ValueError(f"legacy {label} state is invalid")
    return migrated


def migrate_legacy_random_walk_state(
    legacy_state: Any,
    *,
    stream: RandomWalkStream,
    legacy_step_count: int,
) -> RandomWalkState:
    """Migrate an old clockless state using an authenticated external count."""

    fields = _host_state_fields(legacy_state, label="random-walk")
    fields = _require_exact_fields(
        fields,
        {"key", "true_weights"},
        label="legacy random-walk state",
    )
    count = _require_external_legacy_count(legacy_step_count, label="random-walk state")
    migrated = RandomWalkState(
        **fields,
        step_count=jnp.asarray(count, dtype=jnp.int32),
        step_words=jnp.asarray((0, count), dtype=jnp.uint32),
    )
    stream._require_state_contract(migrated)
    if not bool(jax.device_get(stream.state_is_valid(migrated))):
        raise ValueError("legacy random-walk state is invalid")
    return migrated


def migrate_legacy_hidden_state_ar2_state(
    legacy_state: Any,
    *,
    stream: HiddenStateAR2Stream,
    legacy_step_count: int,
) -> HiddenStateAR2State:
    """Migrate an old clockless AR(2) state using an external exact count."""

    fields = _host_state_fields(legacy_state, label="hidden-state AR2")
    fields = _require_exact_fields(
        fields,
        {"key", "x_prev", "x_prev2"},
        label="legacy hidden-state AR2 state",
    )
    count = _require_external_legacy_count(legacy_step_count, label="hidden-state AR2 state")
    migrated = HiddenStateAR2State(
        **fields,
        step_count=jnp.asarray(count, dtype=jnp.int32),
        step_words=jnp.asarray((0, count), dtype=jnp.uint32),
    )
    stream._require_state_contract(migrated)
    if not bool(jax.device_get(stream.state_is_valid(migrated))):
        raise ValueError("legacy hidden-state AR2 state is invalid")
    return migrated


def migrate_legacy_abrupt_change_state(
    legacy_state: Any,
    *,
    stream: AbruptChangeStream,
) -> AbruptChangeState:
    """Migrate an exact unsaturated legacy abrupt-change state."""

    return cast(
        AbruptChangeState,
        _migrate_clocked_state(
            legacy_state,
            stream=stream,
            state_type=AbruptChangeState,
            expected={"key", "true_weights", "step_count"},
            label="abrupt-change",
        ),
    )


def migrate_legacy_sutton_experiment1_state(
    legacy_state: Any,
    *,
    stream: SuttonExperiment1Stream,
) -> SuttonExperiment1State:
    """Migrate an exact unsaturated legacy Sutton Experiment 1 state."""

    return cast(
        SuttonExperiment1State,
        _migrate_clocked_state(
            legacy_state,
            stream=stream,
            state_type=SuttonExperiment1State,
            expected={"key", "signs", "wt_irr", "step_count"},
            label="Sutton Experiment 1",
        ),
    )


def migrate_legacy_cyclic_state(
    legacy_state: Any,
    *,
    stream: CyclicStream,
) -> CyclicState:
    """Migrate an exact unsaturated legacy cyclic state."""

    return cast(
        CyclicState,
        _migrate_clocked_state(
            legacy_state,
            stream=stream,
            state_type=CyclicState,
            expected={"key", "configurations", "step_count"},
            label="cyclic",
        ),
    )


def migrate_legacy_periodic_change_state(
    legacy_state: Any,
    *,
    stream: PeriodicChangeStream,
) -> PeriodicChangeState:
    """Migrate an exact unsaturated legacy periodic-change state."""

    return cast(
        PeriodicChangeState,
        _migrate_clocked_state(
            legacy_state,
            stream=stream,
            state_type=PeriodicChangeState,
            expected={"key", "base_weights", "phases", "step_count"},
            label="periodic-change",
        ),
    )


def migrate_legacy_dynamic_scale_shift_state(
    legacy_state: Any,
    *,
    stream: DynamicScaleShiftStream,
) -> DynamicScaleShiftState:
    """Migrate an exact unsaturated legacy dynamic-scale-shift state."""

    return cast(
        DynamicScaleShiftState,
        _migrate_clocked_state(
            legacy_state,
            stream=stream,
            state_type=DynamicScaleShiftState,
            expected={"key", "true_weights", "current_scales", "step_count"},
            label="dynamic-scale-shift",
        ),
    )


def migrate_legacy_scale_drift_state(
    legacy_state: Any,
    *,
    stream: ScaleDriftStream,
) -> ScaleDriftState:
    """Migrate an exact unsaturated legacy scale-drift state."""

    return cast(
        ScaleDriftState,
        _migrate_clocked_state(
            legacy_state,
            stream=stream,
            state_type=ScaleDriftState,
            expected={"key", "true_weights", "log_scales", "step_count"},
            label="scale-drift",
        ),
    )


def _migrate_legacy_inner_state(
    legacy_state: Any,
    *,
    stream: ScanStream[Any],
    legacy_step_count: int,
) -> Any:
    if isinstance(stream, ScaledStreamWrapper):
        return migrate_legacy_scaled_stream_state(
            legacy_state,
            stream=stream,
            legacy_step_count=legacy_step_count,
        )
    if isinstance(stream, RandomWalkStream):
        return migrate_legacy_random_walk_state(
            legacy_state,
            stream=stream,
            legacy_step_count=legacy_step_count,
        )
    if isinstance(stream, HiddenStateAR2Stream):
        return migrate_legacy_hidden_state_ar2_state(
            legacy_state,
            stream=stream,
            legacy_step_count=legacy_step_count,
        )
    migration_by_type: tuple[tuple[type[Any], Any], ...] = (
        (AbruptChangeStream, migrate_legacy_abrupt_change_state),
        (SuttonExperiment1Stream, migrate_legacy_sutton_experiment1_state),
        (CyclicStream, migrate_legacy_cyclic_state),
        (PeriodicChangeStream, migrate_legacy_periodic_change_state),
        (DynamicScaleShiftStream, migrate_legacy_dynamic_scale_shift_state),
        (ScaleDriftStream, migrate_legacy_scale_drift_state),
    )
    for stream_type, migration in migration_by_type:
        if isinstance(stream, stream_type):
            migrated = migration(legacy_state, stream=stream)
            if not bool(
                jax.device_get(
                    jnp.all(migrated.step_words == jnp.asarray((0, legacy_step_count), jnp.uint32))
                )
            ):
                raise ValueError("legacy child count does not match wrapper count")
            return migrated
    raise TypeError("legacy scaled-stream child type has no exact migration")


def migrate_legacy_scaled_stream_state(
    legacy_state: Any,
    *,
    stream: ScaledStreamWrapper,
    legacy_step_count: int,
) -> ScaledStreamState:
    """Migrate a clockless wrapper and its supported legacy child atomically."""

    count = _require_external_legacy_count(legacy_step_count, label="scaled-stream state")
    fields = _host_state_fields(legacy_state, label="scaled-stream")
    fields = _require_exact_fields(
        fields,
        {"inner_state"},
        label="legacy scaled-stream state",
    )
    inner_state = _migrate_legacy_inner_state(
        fields["inner_state"],
        stream=stream.inner_stream,
        legacy_step_count=count,
    )
    migrated = ScaledStreamState(
        inner_state=inner_state,
        step_count=jnp.asarray(count, dtype=jnp.int32),
        step_words=jnp.asarray((0, count), dtype=jnp.uint32),
    )
    stream._require_state_contract(migrated)
    if not bool(jax.device_get(stream.state_is_valid(migrated))):
        raise ValueError("legacy scaled-stream state is invalid")
    return migrated


# Original "*Target" class names, kept as aliases because they remain part of
# the exported API surface (re-exported from streams/__init__.py).
RandomWalkTarget = RandomWalkStream
AbruptChangeTarget = AbruptChangeStream
CyclicTarget = CyclicStream
PeriodicChangeTarget = PeriodicChangeStream
