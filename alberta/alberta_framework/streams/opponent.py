# mypy: disable-error-code="call-arg"
"""Adaptive-opponent streams with exact continuing-life transactions.

The streams in this module make non-stationarity endogenous.  A
``LearningOpponentStream`` exposes the output of an adapting LMS opponent,
while an ``AdversarialPursuitStream`` moves a target within a fixed per-event
budget after observing a learner prediction.

Both streams use a big-endian ``uint32[2]`` lifetime identity.  The scalar
``step_count`` is saturating compatibility telemetry only.  All-ones is a
terminal capacity sentinel: no random key, state leaf, or clock is changed by
an attempted transaction there.

Pursuit is deliberately two-phase.  ``emit_result`` arms exactly one pending
observation and returns its owner identity; ``resolve_result`` requires that
identity.  Duplicate emission, duplicate resolution, and stale resolution are
therefore rejected atomically.  ``emit`` and ``resolve`` remain as tuple
compatibility wrappers, but new code should carry the owner returned by
``emit_result``.  ``run_pursuit_loop`` uses the authenticated protocol.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from numbers import Real
from typing import Any, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import Array
from jaxtyping import Bool, Float, Int, PRNGKeyArray, UInt

from alberta_framework.core.types import TimeStep

LEARNING_OPPONENT_CONFIG_SCHEMA = "alberta.learning-opponent-stream.config.v2"
LEARNING_OPPONENT_STATE_SCHEMA = "alberta.learning-opponent-stream.state.v2"
LEARNING_OPPONENT_INPUT_SCHEMA = "alberta.learning-opponent-stream.input.v2"
LEARNING_OPPONENT_RESULT_SCHEMA = "alberta.learning-opponent-stream.result.v2"

ADVERSARIAL_PURSUIT_CONFIG_SCHEMA = "alberta.adversarial-pursuit-stream.config.v2"
ADVERSARIAL_PURSUIT_STATE_SCHEMA = "alberta.adversarial-pursuit-stream.state.v2"
ADVERSARIAL_PURSUIT_EMIT_INPUT_SCHEMA = (
    "alberta.adversarial-pursuit-stream.emit-input.v2"
)
ADVERSARIAL_PURSUIT_EMIT_RESULT_SCHEMA = (
    "alberta.adversarial-pursuit-stream.emit-result.v2"
)
ADVERSARIAL_PURSUIT_RESOLVE_INPUT_SCHEMA = (
    "alberta.adversarial-pursuit-stream.resolve-input.v2"
)
ADVERSARIAL_PURSUIT_RESOLVE_RESULT_SCHEMA = (
    "alberta.adversarial-pursuit-stream.resolve-result.v2"
)

OPPONENT_STREAM_RESOURCE_SCHEMA = "alberta.opponent-stream.resource-budget.v2"
OPPONENT_STREAM_CLOCK_NBYTES = 12
OPPONENT_STREAM_CLOCK_DELTA_NBYTES = 8

_INT32_MAX = 2**31 - 1
_UINT32_MAX = 2**32 - 1

__all__ = [
    "ADVERSARIAL_PURSUIT_CONFIG_SCHEMA",
    "ADVERSARIAL_PURSUIT_EMIT_INPUT_SCHEMA",
    "ADVERSARIAL_PURSUIT_EMIT_RESULT_SCHEMA",
    "ADVERSARIAL_PURSUIT_RESOLVE_INPUT_SCHEMA",
    "ADVERSARIAL_PURSUIT_RESOLVE_RESULT_SCHEMA",
    "ADVERSARIAL_PURSUIT_STATE_SCHEMA",
    "LEARNING_OPPONENT_CONFIG_SCHEMA",
    "LEARNING_OPPONENT_INPUT_SCHEMA",
    "LEARNING_OPPONENT_RESULT_SCHEMA",
    "LEARNING_OPPONENT_STATE_SCHEMA",
    "OPPONENT_STREAM_CLOCK_DELTA_NBYTES",
    "OPPONENT_STREAM_CLOCK_NBYTES",
    "OPPONENT_STREAM_RESOURCE_SCHEMA",
    "AdversarialPursuitEmitResult",
    "AdversarialPursuitResolveResult",
    "AdversarialPursuitState",
    "AdversarialPursuitStream",
    "LearningOpponentState",
    "LearningOpponentStepResult",
    "LearningOpponentStream",
    "OpponentStreamResourceBudget",
    "measure_opponent_stream_state_nbytes",
    "migrate_legacy_adversarial_pursuit_state",
    "migrate_legacy_learning_opponent_state",
    "run_pursuit_loop",
]


def _require_exact_fields(
    payload: Mapping[str, Any],
    expected: set[str],
    *,
    label: str,
) -> dict[str, Any]:
    """Return a copy only when a serialized record has exact fields."""

    if not isinstance(payload, Mapping):
        raise TypeError(f"{label} must be a mapping")
    fields = dict(payload)
    if set(fields) != expected:
        missing = sorted(expected - set(fields))
        extra = sorted(set(fields) - expected)
        raise ValueError(f"{label} fields are invalid; missing={missing}, extra={extra}")
    return fields


def _require_positive_int32(value: Any, *, name: str) -> None:
    if type(value) is not int or not 1 <= value <= _INT32_MAX:
        raise ValueError(f"{name} must be an exact integer in [1, {_INT32_MAX}]")


def _require_nonnegative_int32(value: Any, *, name: str) -> None:
    if type(value) is not int or not 0 <= value <= _INT32_MAX:
        raise ValueError(f"{name} must be an exact integer in [0, {_INT32_MAX}]")


def _require_finite_real(
    value: Any,
    *,
    name: str,
    positive: bool = False,
    nonnegative: bool = False,
) -> None:
    valid = (
        not isinstance(value, bool)
        and isinstance(value, Real)
        and math.isfinite(float(value))
    )
    if positive:
        valid = valid and float(value) > 0.0
    if nonnegative:
        valid = valid and float(value) >= 0.0
    if not valid:
        qualifier = " positive" if positive else " non-negative" if nonnegative else ""
        raise ValueError(f"{name} must be a finite{qualifier} real")


def _require_prng_key(key: Any, *, name: str) -> None:
    """Require one scalar typed or legacy JAX PRNG key."""

    if not isinstance(key, Array):
        raise TypeError(f"{name} must be a scalar JAX PRNG key")
    try:
        data = jnp.asarray(jr.key_data(key))
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a scalar JAX PRNG key") from error
    if data.shape != (2,) or data.dtype != jnp.dtype(jnp.uint32):
        raise TypeError(f"{name} must be a scalar JAX PRNG key")


def _require_array(
    value: Any,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: Any,
) -> Array:
    array = jnp.asarray(value)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if array.dtype != jnp.dtype(dtype):
        raise TypeError(f"{name} must have dtype {jnp.dtype(dtype)}, got {array.dtype}")
    return array


def _floating_tree_is_finite(value: Any) -> Bool[Array, ""]:
    valid = jnp.asarray(True, dtype=jnp.bool_)
    for leaf in jax.tree.leaves(value):
        if isinstance(leaf, Array) and jnp.issubdtype(leaf.dtype, jnp.inexact):
            valid = valid & jnp.all(jnp.isfinite(leaf))
    return valid


def _checked_words_increment(words: Array) -> tuple[UInt[Array, " 2"], Bool[Array, ""]]:
    """Increment one big-endian uint32 pair without wrapping all-ones."""

    array = _require_array(
        words,
        name="step_words",
        shape=(2,),
        dtype=jnp.uint32,
    )
    maximum = jnp.asarray(_UINT32_MAX, dtype=jnp.uint32)
    capacity_available = ~jnp.all(array == maximum)
    one = jnp.asarray(1, dtype=jnp.uint32)
    low = array[1] + one
    carry = (low == jnp.asarray(0, dtype=jnp.uint32)).astype(jnp.uint32)
    candidate = jnp.stack((array[0] + carry, low)).astype(jnp.uint32)
    return jnp.where(capacity_available, candidate, array), capacity_available


def _words_to_saturating_int32(words: Array) -> Int[Array, ""]:
    """Project exact words onto non-negative saturating int32 telemetry."""

    array = _require_array(
        words,
        name="step_words",
        shape=(2,),
        dtype=jnp.uint32,
    )
    maximum = jnp.asarray(_INT32_MAX, dtype=jnp.uint32)
    representable = (array[0] == 0) & (array[1] <= maximum)
    return jnp.where(
        representable,
        array[1].astype(jnp.int32),
        jnp.asarray(_INT32_MAX, dtype=jnp.int32),
    )


def _lifetime_counter_valid(words: Array, telemetry: Array) -> Bool[Array, ""]:
    count = _require_array(
        telemetry,
        name="step_count",
        shape=(),
        dtype=jnp.int32,
    )
    return (count >= 0) & (count == _words_to_saturating_int32(words))


def _divmod_words_by_positive_int32(words: Array, divisor: int) -> tuple[Array, Array]:
    """Divide exact uint64 words by a static positive int32 without x64."""

    array = _require_array(
        words,
        name="schedule step_words",
        shape=(2,),
        dtype=jnp.uint32,
    )
    _require_positive_int32(divisor, name="schedule divisor")
    divisor_u = jnp.asarray(divisor, dtype=jnp.uint32)
    zero = jnp.asarray(0, dtype=jnp.uint32)
    one = jnp.asarray(1, dtype=jnp.uint32)

    def divide_bit(
        index: int,
        carry: tuple[Array, Array, Array],
    ) -> tuple[Array, Array, Array]:
        quotient_high, quotient_low, remainder = carry
        bit_index = jnp.asarray(63 - index, dtype=jnp.int32)
        from_high = bit_index >= 32
        shift = jnp.where(from_high, bit_index - 32, bit_index)
        source = jnp.where(from_high, array[0], array[1])
        bit = (source >> shift.astype(jnp.uint32)) & one
        doubled = remainder + remainder + bit
        quotient_bit = doubled >= divisor_u
        next_remainder = jnp.where(quotient_bit, doubled - divisor_u, doubled)
        next_high = (quotient_high << one) | (quotient_low >> jnp.uint32(31))
        next_low = (quotient_low << one) | quotient_bit.astype(jnp.uint32)
        return next_high, next_low, next_remainder

    high, low, remainder = jax.lax.fori_loop(
        0,
        64,
        divide_bit,
        (zero, zero, zero),
    )
    return jnp.stack((high, low)).astype(jnp.uint32), remainder.astype(jnp.uint32)


def _step_input_valid(idx: Any) -> Bool[Array, ""]:
    """Require one scalar numeric scan index and authenticate finiteness."""

    array = jnp.asarray(idx)
    if array.shape != ():
        raise ValueError(f"idx must be scalar, got shape {array.shape}")
    if not (
        jnp.issubdtype(array.dtype, jnp.integer)
        or jnp.issubdtype(array.dtype, jnp.floating)
    ):
        raise TypeError("idx must have an integer or floating dtype")
    return jnp.isfinite(array)


def _prediction_input(prediction: Any) -> tuple[Array, Bool[Array, ""]]:
    """Require the legacy scalar/length-one floating prediction contract."""

    array = jnp.asarray(prediction)
    if array.shape not in ((), (1,)):
        raise ValueError(f"prediction must be scalar or shape (1,), got {array.shape}")
    if not jnp.issubdtype(array.dtype, jnp.floating):
        raise TypeError("prediction must have a floating dtype")
    scalar = jnp.reshape(array, ()).astype(jnp.float32)
    return scalar, jnp.isfinite(scalar)


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
    count = jnp.asarray(fields["step_count"])
    if count.shape != () or count.dtype != jnp.dtype(jnp.int32):
        raise TypeError(f"legacy {label} step_count must be scalar int32")
    host_count = int(jax.device_get(count))
    if host_count < 0:
        raise ValueError(f"negative legacy {label} step_count indicates wrap")
    if host_count >= _INT32_MAX:
        raise ValueError(f"saturated legacy {label} step_count is ambiguous")
    return host_count


@chex.dataclass(frozen=True)
class LearningOpponentState:
    """Persistent internal-opponent state with exact lifetime identity."""

    # The first four fields preserve the pre-v2 positional order.
    key: PRNGKeyArray
    step_count: Int[Array, ""]
    w_star: Float[Array, " feature_dim"]
    w_opp: Float[Array, " feature_dim"]
    step_words: UInt[Array, " 2"]


@chex.dataclass(frozen=True)
class LearningOpponentStepResult:
    """One staged internal-opponent event and atomic commit diagnostics."""

    timestep: TimeStep
    state: LearningOpponentState
    pre_step_words: UInt[Array, " 2"]
    post_step_words: UInt[Array, " 2"]
    lifetime_counter_valid: Bool[Array, ""]
    lifetime_capacity_available: Bool[Array, ""]
    input_valid: Bool[Array, ""]
    state_valid: Bool[Array, ""]
    output_valid: Bool[Array, ""]
    candidate_state_valid: Bool[Array, ""]
    reset_requested: Bool[Array, ""]
    opponent_reset: Bool[Array, ""]
    update_applied: Bool[Array, ""]
    update_rejected: Bool[Array, ""]


@chex.dataclass(frozen=True)
class AdversarialPursuitState:
    """Adversarial weights plus one authenticated pending observation."""

    # The first four fields preserve the pre-v2 positional order.
    key: PRNGKeyArray
    step_count: Int[Array, ""]
    w_adv: Float[Array, " feature_dim"]
    pending_x: Float[Array, " feature_dim"]
    step_words: UInt[Array, " 2"]
    pending_owner_words: UInt[Array, " 2"]
    pending_armed: Bool[Array, ""]


@chex.dataclass(frozen=True)
class AdversarialPursuitEmitResult:
    """One authenticated pursuit observation emission."""

    observation: Float[Array, " feature_dim"]
    state: AdversarialPursuitState
    owner_words: UInt[Array, " 2"]
    pre_step_words: UInt[Array, " 2"]
    post_step_words: UInt[Array, " 2"]
    lifetime_counter_valid: Bool[Array, ""]
    lifetime_capacity_available: Bool[Array, ""]
    state_valid: Bool[Array, ""]
    protocol_idle: Bool[Array, ""]
    output_valid: Bool[Array, ""]
    candidate_state_valid: Bool[Array, ""]
    update_applied: Bool[Array, ""]
    update_rejected: Bool[Array, ""]


@chex.dataclass(frozen=True)
class AdversarialPursuitResolveResult:
    """One owner-authenticated target resolution and clock commit."""

    target: Float[Array, " 1"]
    state: AdversarialPursuitState
    owner_words: UInt[Array, " 2"]
    pre_step_words: UInt[Array, " 2"]
    post_step_words: UInt[Array, " 2"]
    lifetime_counter_valid: Bool[Array, ""]
    lifetime_capacity_available: Bool[Array, ""]
    state_valid: Bool[Array, ""]
    protocol_armed: Bool[Array, ""]
    owner_matches: Bool[Array, ""]
    prediction_valid: Bool[Array, ""]
    output_valid: Bool[Array, ""]
    candidate_state_valid: Bool[Array, ""]
    update_applied: Bool[Array, ""]
    update_rejected: Bool[Array, ""]


@dataclasses.dataclass(frozen=True)
class OpponentStreamResourceBudget:
    """Exact logical and byte accounting for one opponent stream state."""

    schema: str
    stream_type: str
    state_schema: str
    input_schema: str
    result_schema: str
    state_nbytes: int
    exact_clock_nbytes: int
    exact_clock_delta_nbytes: int
    rng_nbytes: int
    persistent_float32_scalars: int
    protocol_uint32_scalars: int
    protocol_bool_scalars: int
    trainable_scalars: int
    replay_capacity: int

    def to_dict(self) -> dict[str, int | str]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> OpponentStreamResourceBudget:
        expected = {field.name for field in dataclasses.fields(cls)}
        fields = _require_exact_fields(payload, expected, label="opponent resource budget")
        if fields["schema"] != OPPONENT_STREAM_RESOURCE_SCHEMA:
            raise ValueError("opponent resource schema is unsupported")
        integer_fields = expected - {
            "schema",
            "stream_type",
            "state_schema",
            "input_schema",
            "result_schema",
        }
        for name in integer_fields:
            if type(fields[name]) is not int or fields[name] < 0:
                raise ValueError(f"opponent resource {name} must be a non-negative integer")
        if fields["exact_clock_nbytes"] != OPPONENT_STREAM_CLOCK_NBYTES:
            raise ValueError("opponent exact_clock_nbytes is invalid")
        if fields["exact_clock_delta_nbytes"] != OPPONENT_STREAM_CLOCK_DELTA_NBYTES:
            raise ValueError("opponent exact_clock_delta_nbytes is invalid")
        if fields["rng_nbytes"] != 8:
            raise ValueError("opponent rng_nbytes is invalid")
        if fields["replay_capacity"] != 0:
            raise ValueError("opponent streams do not own replay storage")

        stream_type = fields["stream_type"]
        if stream_type == "LearningOpponentStream":
            required = (
                LEARNING_OPPONENT_STATE_SCHEMA,
                LEARNING_OPPONENT_INPUT_SCHEMA,
                LEARNING_OPPONENT_RESULT_SCHEMA,
            )
            if (
                fields["protocol_uint32_scalars"] != 0
                or fields["protocol_bool_scalars"] != 0
                or fields["persistent_float32_scalars"] <= 0
                or fields["persistent_float32_scalars"] % 2 != 0
                or fields["trainable_scalars"]
                != fields["persistent_float32_scalars"] // 2
            ):
                raise ValueError("learning-opponent resource structure is invalid")
        elif stream_type == "AdversarialPursuitStream":
            required = (
                ADVERSARIAL_PURSUIT_STATE_SCHEMA,
                ADVERSARIAL_PURSUIT_RESOLVE_INPUT_SCHEMA,
                ADVERSARIAL_PURSUIT_RESOLVE_RESULT_SCHEMA,
            )
            if (
                fields["protocol_uint32_scalars"] != 2
                or fields["protocol_bool_scalars"] != 1
                or fields["persistent_float32_scalars"] <= 0
                or fields["persistent_float32_scalars"] % 2 != 0
                or fields["trainable_scalars"] != 0
            ):
                raise ValueError("adversarial-pursuit resource structure is invalid")
        else:
            raise ValueError("opponent resource stream_type is unsupported")
        if (
            fields["state_schema"],
            fields["input_schema"],
            fields["result_schema"],
        ) != required:
            raise ValueError("opponent resource schemas are inconsistent")

        expected_nbytes = (
            fields["rng_nbytes"]
            + fields["exact_clock_nbytes"]
            + 4 * fields["persistent_float32_scalars"]
            + 4 * fields["protocol_uint32_scalars"]
            + fields["protocol_bool_scalars"]
        )
        if fields["state_nbytes"] != expected_nbytes:
            raise ValueError("opponent resource state_nbytes is invalid")
        return cls(**fields)


class LearningOpponentStream:
    """Predict the output of an internal LMS learner over a continuing life."""

    def __init__(
        self,
        feature_dim: int,
        opponent_step_size: float = 0.02,
        reset_interval: int = 4000,
        opponent_noise_std: float = 0.1,
        target_noise_std: float = 0.05,
        feature_std: float = 1.0,
    ) -> None:
        _require_positive_int32(feature_dim, name="feature_dim")
        _require_finite_real(
            opponent_step_size,
            name="opponent_step_size",
            positive=True,
        )
        _require_nonnegative_int32(reset_interval, name="reset_interval")
        _require_finite_real(
            opponent_noise_std,
            name="opponent_noise_std",
            nonnegative=True,
        )
        _require_finite_real(
            target_noise_std,
            name="target_noise_std",
            nonnegative=True,
        )
        _require_finite_real(feature_std, name="feature_std", nonnegative=True)
        self._feature_dim = feature_dim
        self._opp_alpha = float(opponent_step_size)
        self._reset_interval = reset_interval
        self._opp_noise_std = float(opponent_noise_std)
        self._target_noise_std = float(target_noise_std)
        self._feature_std = float(feature_std)

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    @property
    def reset_interval(self) -> int:
        return self._reset_interval

    @property
    def resource_budget(self) -> OpponentStreamResourceBudget:
        state = self.init(jr.key(0))
        return OpponentStreamResourceBudget(
            schema=OPPONENT_STREAM_RESOURCE_SCHEMA,
            stream_type=type(self).__name__,
            state_schema=LEARNING_OPPONENT_STATE_SCHEMA,
            input_schema=LEARNING_OPPONENT_INPUT_SCHEMA,
            result_schema=LEARNING_OPPONENT_RESULT_SCHEMA,
            state_nbytes=measure_opponent_stream_state_nbytes(state),
            exact_clock_nbytes=OPPONENT_STREAM_CLOCK_NBYTES,
            exact_clock_delta_nbytes=OPPONENT_STREAM_CLOCK_DELTA_NBYTES,
            rng_nbytes=8,
            persistent_float32_scalars=2 * self._feature_dim,
            protocol_uint32_scalars=0,
            protocol_bool_scalars=0,
            trainable_scalars=self._feature_dim,
            replay_capacity=0,
        )

    def to_config(self) -> dict[str, Any]:
        return {
            "type": type(self).__name__,
            "config_schema": LEARNING_OPPONENT_CONFIG_SCHEMA,
            "state_schema": LEARNING_OPPONENT_STATE_SCHEMA,
            "input_schema": LEARNING_OPPONENT_INPUT_SCHEMA,
            "result_schema": LEARNING_OPPONENT_RESULT_SCHEMA,
            "feature_dim": self._feature_dim,
            "opponent_step_size": self._opp_alpha,
            "reset_interval": self._reset_interval,
            "opponent_noise_std": self._opp_noise_std,
            "target_noise_std": self._target_noise_std,
            "feature_std": self._feature_std,
        }

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> LearningOpponentStream:
        expected = {
            "type",
            "config_schema",
            "state_schema",
            "input_schema",
            "result_schema",
            "feature_dim",
            "opponent_step_size",
            "reset_interval",
            "opponent_noise_std",
            "target_noise_std",
            "feature_std",
        }
        fields = _require_exact_fields(config, expected, label="learning-opponent config")
        schema_values = {
            "type": cls.__name__,
            "config_schema": LEARNING_OPPONENT_CONFIG_SCHEMA,
            "state_schema": LEARNING_OPPONENT_STATE_SCHEMA,
            "input_schema": LEARNING_OPPONENT_INPUT_SCHEMA,
            "result_schema": LEARNING_OPPONENT_RESULT_SCHEMA,
        }
        for name, expected_value in schema_values.items():
            if fields.pop(name) != expected_value:
                raise ValueError(f"learning-opponent {name} schema value is unsupported")
        return cls(**fields)

    def _require_state_contract(self, state: LearningOpponentState) -> None:
        if not isinstance(state, LearningOpponentState):
            raise TypeError("state must be a LearningOpponentState")
        _require_prng_key(state.key, name="learning-opponent key")
        _require_array(
            state.step_count,
            name="learning-opponent step_count",
            shape=(),
            dtype=jnp.int32,
        )
        _require_array(
            state.w_star,
            name="learning-opponent w_star",
            shape=(self._feature_dim,),
            dtype=jnp.float32,
        )
        _require_array(
            state.w_opp,
            name="learning-opponent w_opp",
            shape=(self._feature_dim,),
            dtype=jnp.float32,
        )
        _require_array(
            state.step_words,
            name="learning-opponent step_words",
            shape=(2,),
            dtype=jnp.uint32,
        )

    def state_is_valid(self, state: LearningOpponentState) -> Bool[Array, ""]:
        self._require_state_contract(state)
        return _lifetime_counter_valid(
            state.step_words,
            state.step_count,
        ) & _floating_tree_is_finite(state)

    def init(self, key: Array) -> LearningOpponentState:
        _require_prng_key(key, name="key")
        key, k_star = jr.split(key)
        w_star = jr.normal(k_star, (self._feature_dim,), dtype=jnp.float32)
        return LearningOpponentState(
            key=key,
            step_count=jnp.asarray(0, dtype=jnp.int32),
            w_star=w_star,
            w_opp=jnp.zeros((self._feature_dim,), dtype=jnp.float32),
            step_words=jnp.zeros((2,), dtype=jnp.uint32),
        )

    def _reset_requested(self, words: Array) -> Bool[Array, ""]:
        if self._reset_interval == 0:
            return jnp.asarray(False, dtype=jnp.bool_)
        _quotient, remainder = _divmod_words_by_positive_int32(
            words,
            self._reset_interval,
        )
        return jnp.any(words != 0) & (remainder == 0)

    def step(
        self,
        state: LearningOpponentState,
        idx: Array,
    ) -> tuple[TimeStep, LearningOpponentState]:
        result = self.step_result(state, idx)
        return result.timestep, result.state

    def step_result(
        self,
        state: LearningOpponentState,
        idx: Array,
    ) -> LearningOpponentStepResult:
        """Stage one sample/opponent update and commit it atomically."""

        self._require_state_contract(state)
        input_valid = _step_input_valid(idx)
        lifetime_counter_valid = _lifetime_counter_valid(
            state.step_words,
            state.step_count,
        )
        state_valid = lifetime_counter_valid & _floating_tree_is_finite(state)
        proposed_words, lifetime_capacity_available = _checked_words_increment(
            state.step_words
        )
        reset_requested = self._reset_requested(state.step_words)
        working_w = jnp.where(reset_requested, jnp.zeros_like(state.w_opp), state.w_opp)
        key, k_x, k_tnoise, k_ox, k_onoise = jr.split(state.key, 5)

        x = self._feature_std * jr.normal(
            k_x,
            (self._feature_dim,),
            dtype=jnp.float32,
        )
        noise = self._target_noise_std * jr.normal(k_tnoise, (), dtype=jnp.float32)
        target = jnp.dot(working_w, x) + noise

        x_opp = self._feature_std * jr.normal(
            k_ox,
            (self._feature_dim,),
            dtype=jnp.float32,
        )
        y_opp = jnp.dot(state.w_star, x_opp) + self._opp_noise_std * jr.normal(
            k_onoise,
            (),
            dtype=jnp.float32,
        )
        opp_error = y_opp - jnp.dot(working_w, x_opp)
        new_w_opp = working_w + self._opp_alpha * opp_error * x_opp
        candidate_state = LearningOpponentState(
            key=key,
            step_count=_words_to_saturating_int32(proposed_words),
            w_star=state.w_star,
            w_opp=new_w_opp,
            step_words=proposed_words,
        )
        output_valid = jnp.all(jnp.isfinite(x)) & jnp.isfinite(target)
        candidate_state_valid = self.state_is_valid(candidate_state)
        update_applied = (
            state_valid
            & input_valid
            & lifetime_capacity_available
            & output_valid
            & candidate_state_valid
        )
        committed_state = jax.lax.cond(
            update_applied,
            lambda _: candidate_state,
            lambda _: state,
            operand=None,
        )
        timestep = TimeStep(
            observation=jnp.where(update_applied, x, jnp.zeros_like(x)),
            target=jnp.atleast_1d(
                jnp.where(update_applied, target, jnp.asarray(0.0, dtype=jnp.float32))
            ),
        )
        return LearningOpponentStepResult(
            timestep=timestep,
            state=committed_state,
            pre_step_words=state.step_words,
            post_step_words=cast(Array, committed_state.step_words),
            lifetime_counter_valid=lifetime_counter_valid,
            lifetime_capacity_available=lifetime_capacity_available,
            input_valid=input_valid,
            state_valid=state_valid,
            output_valid=output_valid,
            candidate_state_valid=candidate_state_valid,
            reset_requested=reset_requested,
            opponent_reset=update_applied & reset_requested,
            update_applied=update_applied,
            update_rejected=~update_applied,
        )


class AdversarialPursuitStream:
    """Budgeted worst-case drift with an owner-authenticated two-phase step."""

    def __init__(
        self,
        feature_dim: int,
        drift_budget: float = 0.02,
        noise_std: float = 0.05,
        feature_std: float = 1.0,
    ) -> None:
        _require_positive_int32(feature_dim, name="feature_dim")
        _require_finite_real(drift_budget, name="drift_budget", nonnegative=True)
        _require_finite_real(noise_std, name="noise_std", nonnegative=True)
        _require_finite_real(feature_std, name="feature_std", nonnegative=True)
        self._feature_dim = feature_dim
        self._budget = float(drift_budget)
        self._noise_std = float(noise_std)
        self._feature_std = float(feature_std)

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    @property
    def resource_budget(self) -> OpponentStreamResourceBudget:
        state = self.init(jr.key(0))
        return OpponentStreamResourceBudget(
            schema=OPPONENT_STREAM_RESOURCE_SCHEMA,
            stream_type=type(self).__name__,
            state_schema=ADVERSARIAL_PURSUIT_STATE_SCHEMA,
            input_schema=ADVERSARIAL_PURSUIT_RESOLVE_INPUT_SCHEMA,
            result_schema=ADVERSARIAL_PURSUIT_RESOLVE_RESULT_SCHEMA,
            state_nbytes=measure_opponent_stream_state_nbytes(state),
            exact_clock_nbytes=OPPONENT_STREAM_CLOCK_NBYTES,
            exact_clock_delta_nbytes=OPPONENT_STREAM_CLOCK_DELTA_NBYTES,
            rng_nbytes=8,
            persistent_float32_scalars=2 * self._feature_dim,
            protocol_uint32_scalars=2,
            protocol_bool_scalars=1,
            trainable_scalars=0,
            replay_capacity=0,
        )

    def to_config(self) -> dict[str, Any]:
        return {
            "type": type(self).__name__,
            "config_schema": ADVERSARIAL_PURSUIT_CONFIG_SCHEMA,
            "state_schema": ADVERSARIAL_PURSUIT_STATE_SCHEMA,
            "emit_input_schema": ADVERSARIAL_PURSUIT_EMIT_INPUT_SCHEMA,
            "emit_result_schema": ADVERSARIAL_PURSUIT_EMIT_RESULT_SCHEMA,
            "resolve_input_schema": ADVERSARIAL_PURSUIT_RESOLVE_INPUT_SCHEMA,
            "resolve_result_schema": ADVERSARIAL_PURSUIT_RESOLVE_RESULT_SCHEMA,
            "feature_dim": self._feature_dim,
            "drift_budget": self._budget,
            "noise_std": self._noise_std,
            "feature_std": self._feature_std,
        }

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> AdversarialPursuitStream:
        expected = {
            "type",
            "config_schema",
            "state_schema",
            "emit_input_schema",
            "emit_result_schema",
            "resolve_input_schema",
            "resolve_result_schema",
            "feature_dim",
            "drift_budget",
            "noise_std",
            "feature_std",
        }
        fields = _require_exact_fields(config, expected, label="adversarial-pursuit config")
        schema_values = {
            "type": cls.__name__,
            "config_schema": ADVERSARIAL_PURSUIT_CONFIG_SCHEMA,
            "state_schema": ADVERSARIAL_PURSUIT_STATE_SCHEMA,
            "emit_input_schema": ADVERSARIAL_PURSUIT_EMIT_INPUT_SCHEMA,
            "emit_result_schema": ADVERSARIAL_PURSUIT_EMIT_RESULT_SCHEMA,
            "resolve_input_schema": ADVERSARIAL_PURSUIT_RESOLVE_INPUT_SCHEMA,
            "resolve_result_schema": ADVERSARIAL_PURSUIT_RESOLVE_RESULT_SCHEMA,
        }
        for name, expected_value in schema_values.items():
            if fields.pop(name) != expected_value:
                raise ValueError(f"adversarial-pursuit {name} schema value is unsupported")
        return cls(**fields)

    def _require_state_contract(self, state: AdversarialPursuitState) -> None:
        if not isinstance(state, AdversarialPursuitState):
            raise TypeError("state must be an AdversarialPursuitState")
        _require_prng_key(state.key, name="adversarial-pursuit key")
        _require_array(
            state.step_count,
            name="adversarial-pursuit step_count",
            shape=(),
            dtype=jnp.int32,
        )
        _require_array(
            state.w_adv,
            name="adversarial-pursuit w_adv",
            shape=(self._feature_dim,),
            dtype=jnp.float32,
        )
        _require_array(
            state.pending_x,
            name="adversarial-pursuit pending_x",
            shape=(self._feature_dim,),
            dtype=jnp.float32,
        )
        _require_array(
            state.step_words,
            name="adversarial-pursuit step_words",
            shape=(2,),
            dtype=jnp.uint32,
        )
        _require_array(
            state.pending_owner_words,
            name="adversarial-pursuit pending_owner_words",
            shape=(2,),
            dtype=jnp.uint32,
        )
        _require_array(
            state.pending_armed,
            name="adversarial-pursuit pending_armed",
            shape=(),
            dtype=jnp.bool_,
        )

    def state_is_valid(self, state: AdversarialPursuitState) -> Bool[Array, ""]:
        self._require_state_contract(state)
        _proposed, capacity_available = _checked_words_increment(state.step_words)
        return (
            _lifetime_counter_valid(state.step_words, state.step_count)
            & _floating_tree_is_finite(state)
            & jnp.array_equal(state.pending_owner_words, state.step_words)
            & ((~state.pending_armed) | capacity_available)
        )

    def init(self, key: Array) -> AdversarialPursuitState:
        _require_prng_key(key, name="key")
        key, k_w = jr.split(key)
        words = jnp.zeros((2,), dtype=jnp.uint32)
        return AdversarialPursuitState(
            key=key,
            step_count=jnp.asarray(0, dtype=jnp.int32),
            w_adv=jr.normal(k_w, (self._feature_dim,), dtype=jnp.float32),
            pending_x=jnp.zeros((self._feature_dim,), dtype=jnp.float32),
            step_words=words,
            pending_owner_words=words,
            pending_armed=jnp.asarray(False, dtype=jnp.bool_),
        )

    def emit(
        self,
        state: AdversarialPursuitState,
    ) -> tuple[Array, AdversarialPursuitState]:
        """Compatibility wrapper returning observation and armed state."""

        result = self.emit_result(state)
        return result.observation, result.state

    def emit_result(self, state: AdversarialPursuitState) -> AdversarialPursuitEmitResult:
        """Arm exactly one observation without advancing the event clock."""

        self._require_state_contract(state)
        lifetime_counter_valid = _lifetime_counter_valid(
            state.step_words,
            state.step_count,
        )
        _proposed_words, lifetime_capacity_available = _checked_words_increment(
            state.step_words
        )
        state_valid = self.state_is_valid(state)
        protocol_idle = ~state.pending_armed
        key, k_x = jr.split(state.key)
        x = self._feature_std * jr.normal(
            k_x,
            (self._feature_dim,),
            dtype=jnp.float32,
        )
        candidate_state = AdversarialPursuitState(
            key=key,
            step_count=state.step_count,
            w_adv=state.w_adv,
            pending_x=x,
            step_words=state.step_words,
            pending_owner_words=state.step_words,
            pending_armed=jnp.asarray(True, dtype=jnp.bool_),
        )
        output_valid = jnp.all(jnp.isfinite(x))
        candidate_state_valid = self.state_is_valid(candidate_state)
        update_applied = (
            state_valid
            & protocol_idle
            & lifetime_capacity_available
            & output_valid
            & candidate_state_valid
        )
        committed_state = jax.lax.cond(
            update_applied,
            lambda _: candidate_state,
            lambda _: state,
            operand=None,
        )
        # A rejected emission receives an identity that cannot authenticate the
        # pending event.  This matters when a caller accidentally feeds the
        # result of a duplicate emit into resolve_result.
        rejected_owner = jnp.bitwise_not(state.pending_owner_words).astype(jnp.uint32)
        owner_words = jnp.where(update_applied, state.step_words, rejected_owner)
        observation = jnp.where(update_applied, x, jnp.zeros_like(x))
        return AdversarialPursuitEmitResult(
            observation=observation,
            state=committed_state,
            owner_words=owner_words,
            pre_step_words=state.step_words,
            post_step_words=cast(Array, committed_state.step_words),
            lifetime_counter_valid=lifetime_counter_valid,
            lifetime_capacity_available=lifetime_capacity_available,
            state_valid=state_valid,
            protocol_idle=protocol_idle,
            output_valid=output_valid,
            candidate_state_valid=candidate_state_valid,
            update_applied=update_applied,
            update_rejected=~update_applied,
        )

    def resolve(
        self,
        state: AdversarialPursuitState,
        prediction: Array,
    ) -> tuple[Array, AdversarialPursuitState]:
        """Compatibility wrapper resolving the state's current owner.

        This preserves the historical tuple API.  Code that can carry a token
        should call :meth:`resolve_result` with the owner from
        :meth:`emit_result` so stale caller state is detectable.
        """

        result = self.resolve_result(state, prediction, state.pending_owner_words)
        return result.target, result.state

    def resolve_result(
        self,
        state: AdversarialPursuitState,
        prediction: Array,
        owner_words: Array,
    ) -> AdversarialPursuitResolveResult:
        """Resolve one pending observation and atomically advance its owner."""

        self._require_state_contract(state)
        prediction_scalar, prediction_valid = _prediction_input(prediction)
        owner = _require_array(
            owner_words,
            name="owner_words",
            shape=(2,),
            dtype=jnp.uint32,
        )
        lifetime_counter_valid = _lifetime_counter_valid(
            state.step_words,
            state.step_count,
        )
        proposed_words, lifetime_capacity_available = _checked_words_increment(
            state.step_words
        )
        state_valid = self.state_is_valid(state)
        protocol_armed = state.pending_armed
        owner_matches = jnp.array_equal(owner, state.pending_owner_words) & jnp.array_equal(
            owner,
            state.step_words,
        )
        safe_prediction = jnp.where(
            prediction_valid,
            prediction_scalar,
            jnp.asarray(0.0, dtype=jnp.float32),
        )
        key, k_noise = jr.split(state.key)
        x = state.pending_x
        noise = self._noise_std * jr.normal(k_noise, (), dtype=jnp.float32)
        target_scalar = jnp.dot(state.w_adv, x) + noise
        err = jnp.dot(state.w_adv, x) - safe_prediction
        direction = err * x
        norm = jnp.linalg.norm(direction)
        unit = jnp.where(
            norm > jnp.asarray(1e-8, dtype=jnp.float32),
            direction / jnp.maximum(norm, jnp.asarray(1e-8, dtype=jnp.float32)),
            jnp.zeros_like(direction),
        )
        new_w = state.w_adv + self._budget * unit
        candidate_state = AdversarialPursuitState(
            key=key,
            step_count=_words_to_saturating_int32(proposed_words),
            w_adv=new_w,
            pending_x=jnp.zeros_like(state.pending_x),
            step_words=proposed_words,
            pending_owner_words=proposed_words,
            pending_armed=jnp.asarray(False, dtype=jnp.bool_),
        )
        output_valid = jnp.isfinite(target_scalar) & jnp.all(jnp.isfinite(new_w))
        candidate_state_valid = self.state_is_valid(candidate_state)
        update_applied = (
            state_valid
            & protocol_armed
            & owner_matches
            & prediction_valid
            & lifetime_capacity_available
            & output_valid
            & candidate_state_valid
        )
        committed_state = jax.lax.cond(
            update_applied,
            lambda _: candidate_state,
            lambda _: state,
            operand=None,
        )
        target = jnp.atleast_1d(
            jnp.where(
                update_applied,
                target_scalar,
                jnp.asarray(0.0, dtype=jnp.float32),
            )
        )
        return AdversarialPursuitResolveResult(
            target=target,
            state=committed_state,
            owner_words=owner,
            pre_step_words=state.step_words,
            post_step_words=cast(Array, committed_state.step_words),
            lifetime_counter_valid=lifetime_counter_valid,
            lifetime_capacity_available=lifetime_capacity_available,
            state_valid=state_valid,
            protocol_armed=protocol_armed,
            owner_matches=owner_matches,
            prediction_valid=prediction_valid,
            output_valid=output_valid,
            candidate_state_valid=candidate_state_valid,
            update_applied=update_applied,
            update_rejected=~update_applied,
        )


def measure_opponent_stream_state_nbytes(
    state: LearningOpponentState | AdversarialPursuitState,
) -> int:
    """Measure every persistent JAX-array byte in a concrete stream state."""

    if not isinstance(state, (LearningOpponentState, AdversarialPursuitState)):
        raise TypeError("state must be an opponent stream state")
    return sum(
        int(leaf.nbytes)
        for leaf in jax.tree.leaves(state)
        if isinstance(leaf, Array)
    )


def migrate_legacy_learning_opponent_state(
    legacy_state: Any,
    *,
    stream: LearningOpponentStream,
) -> LearningOpponentState:
    """Migrate only an exact, finite, unsaturated pre-v2 opponent state."""

    if not isinstance(stream, LearningOpponentStream):
        raise TypeError("stream must be a LearningOpponentStream")
    fields = _host_state_fields(legacy_state, label="learning-opponent")
    expected = {"key", "step_count", "w_star", "w_opp"}
    fields = _require_exact_fields(fields, expected, label="legacy learning-opponent state")
    count = _legacy_unsaturated_count(fields, label="learning-opponent")
    fields["step_words"] = jnp.asarray((0, count), dtype=jnp.uint32)
    migrated = LearningOpponentState(**fields)
    stream._require_state_contract(migrated)
    if not bool(jax.device_get(stream.state_is_valid(migrated))):
        raise ValueError("legacy learning-opponent state is invalid")
    return migrated


def migrate_legacy_adversarial_pursuit_state(
    legacy_state: Any,
    *,
    stream: AdversarialPursuitStream,
    pending_armed: bool | None = None,
) -> AdversarialPursuitState:
    """Migrate an unambiguous pre-v2 pursuit state.

    Pre-v2 state stored ``pending_x`` but no phase bit, so it cannot reveal
    whether it was captured before or after ``emit``.  The caller must provide
    that historical fact explicitly; guessing would permit a skipped or
    duplicated resolution.
    """

    if not isinstance(stream, AdversarialPursuitStream):
        raise TypeError("stream must be an AdversarialPursuitStream")
    if type(pending_armed) is not bool:
        raise TypeError("pending_armed must be provided as an exact bool")
    fields = _host_state_fields(legacy_state, label="adversarial-pursuit")
    expected = {"key", "step_count", "w_adv", "pending_x"}
    fields = _require_exact_fields(
        fields,
        expected,
        label="legacy adversarial-pursuit state",
    )
    count = _legacy_unsaturated_count(fields, label="adversarial-pursuit")
    words = jnp.asarray((0, count), dtype=jnp.uint32)
    fields["step_words"] = words
    fields["pending_owner_words"] = words
    fields["pending_armed"] = jnp.asarray(pending_armed, dtype=jnp.bool_)
    migrated = AdversarialPursuitState(**fields)
    stream._require_state_contract(migrated)
    if not bool(jax.device_get(stream.state_is_valid(migrated))):
        raise ValueError("legacy adversarial-pursuit state is invalid")
    return migrated


def run_pursuit_loop(
    learner: Any,
    stream: AdversarialPursuitStream,
    num_steps: int,
    key: Array,
    learner_state: Any = None,
    frozen: bool = False,
) -> tuple[Any, Array]:
    """Run a learner through the owner-authenticated pursuit protocol."""

    if not isinstance(stream, AdversarialPursuitStream):
        raise TypeError("stream must be an AdversarialPursuitStream")
    _require_nonnegative_int32(num_steps, name="num_steps")
    if type(frozen) is not bool:
        raise TypeError("frozen must be an exact bool")
    _require_prng_key(key, name="key")
    l_state = learner.init(stream.feature_dim) if learner_state is None else learner_state
    s_state = stream.init(key)

    def step_fn(
        carry: tuple[Any, AdversarialPursuitState],
        _: Array,
    ) -> tuple[tuple[Any, AdversarialPursuitState], Array]:
        l_st, s_st = carry
        emitted = stream.emit_result(s_st)
        prediction = learner.predict(l_st, emitted.observation)
        resolved = stream.resolve_result(
            emitted.state,
            prediction,
            emitted.owner_words,
        )
        sq_error = (
            jnp.squeeze(resolved.target) - jnp.squeeze(prediction)
        ) ** 2
        sq_error = jnp.where(
            emitted.update_applied & resolved.update_applied,
            sq_error,
            jnp.asarray(0.0, dtype=jnp.float32),
        )
        if not frozen:
            learner_result = learner.update(
                l_st,
                emitted.observation,
                resolved.target,
            )
            l_st = jax.lax.cond(
                emitted.update_applied & resolved.update_applied,
                lambda _: learner_result.state,
                lambda _: l_st,
                operand=None,
            )
        return (l_st, resolved.state), sq_error

    (final_state, _), sq_errors = jax.lax.scan(
        step_fn,
        (l_state, s_state),
        jnp.arange(num_steps, dtype=jnp.int32),
    )
    return final_state, sq_errors
