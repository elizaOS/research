# mypy: disable-error-code="attr-defined,call-arg,no-any-return"
"""Bounded Self-Normalized Resets for a fixed-width dense ReLU layer.

This module implements the practical Self-Normalized Resets (SNR) estimator
described by Farias and Jozefiak, *Self-Normalized Resets for Plasticity in
Continual Learning* (ICLR 2025, arXiv:2410.20098).  It is not a generic
dead-unit score: each unit has its own empirical inter-firing-time law.

The paper leaves several implementation choices implicit.  They are fixed here:

* the input is one post-ReLU activation vector; exactly ``activation > 0`` fires;
* age is zero on a firing and otherwise counts observations since that firing;
* completed inter-firing intervals are ``age + 1`` and therefore live on the
  positive-integer geometric support ``{1, 2, ...}``;
* the fixed trailing window estimates the geometric mean, so ``p = 1 / mean``;
* ``log P(A >= age + 1) = age * log(1 - p)`` because ``age`` silent
  observations imply that the next inclusive inter-firing distance exceeds age;
* a unit is eligible only after explicit per-reset-epoch warmup and a configured
  number of completed intervals;
* the caller supplies parameters *after* its optimizer update.  SNR then resets
  them, matching Algorithm 1's forward, age update, optimize, reset ordering.

Algorithm 1 writes ``P(A >= age)`` while setting age to zero on a firing. To
resolve that indexing against a positive-support geometric law, this module
treats ``A`` as the inclusive distance between positive firings: after ``age``
silent observations the surviving event is ``A >= age + 1``. This mapping agrees
with the hypothesis-test probability of the observed silent run. It is serialized
explicitly; the surrounding fixed-window implementation is not claimed to be
bit-equivalent to the authors' released histogram code, which bins silent ages.

On reset, the first post-reset positive activation starts a new firing epoch;
it does not fabricate a completed interval.  The prior fixed window is retained
as the unit's nominal firing-law estimate.  Warmup prevents an immediate repeat
reset.  These lifecycle rules, plus finite resource caps, are Alberta engineering
choices rather than additional claims made by the paper.

The concrete consumer supports one dense hidden ReLU layer with kernels shaped
``(input_dim, unit_count)`` and ``(unit_count, output_dim)``.  Selected incoming
columns and biases receive fresh initialization, selected outgoing rows become
zero, and the corresponding Adam moments become zero when Adam is configured.
The only accepted optimizer contracts are ``none``, stateless ``sgd``, and the
exact Adam moment structure defined below; arbitrary optimizer trees are rejected.
The slice directions, zero outgoing weights, and Adam clearing follow the authors'
released MLP implementation. Owned mode's default zero bias also matches it;
caller-provided mode deliberately permits a different explicit initializer.

Checkpoint and live-state checksums are unkeyed integrity checks.  They detect
ordinary corruption and noncanonical payloads; they do not authenticate a state
against a malicious writer.  This is bounded L0 mechanism code, not scientific
evidence of continual-learning benefit.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array
from jaxtyping import Bool, Float, Int, PRNGKeyArray, UInt

SELF_NORMALIZED_RESETS_SCHEMA = "alberta.self-normalized-resets.v1"
SelfNormalizedOptimizerKind = Literal["none", "sgd", "adam"]
SelfNormalizedInitializationMode = Literal["owned_lecun_uniform", "caller_provided"]

_INT32_MAX = 2_147_483_647
_UINT32_MAX = 2**32 - 1
_UINT64_MAX = 2**64 - 1
_FLOAT32_MAX = float(np.finfo(np.float32).max)
_FLOAT32_TINY = float(np.finfo(np.float32).tiny)
_MAX_DIMENSION = 4_096
_MAX_WINDOW_SIZE = 4_096
_MAX_PARAMETER_COUNT = 16_777_216
_MAX_INTERVAL_SLOTS = 16_777_216
_CHECKPOINT_FIELDS = {
    "schema",
    "type",
    "integrity_notice",
    "implementation_source_sha256",
    "config",
    "state",
    "checkpoint_sha256",
}
_STATE_FIELDS = {
    "rng_key_data",
    "target",
    "ages_words",
    "epoch_observations_words",
    "has_fired",
    "intervals_words",
    "interval_count",
    "interval_cursor",
    "unit_reset_count",
    "total_reset_count",
    "step_words",
    "step_count",
    "source_binding_words",
    "representation_binding_words",
    "integrity_tag",
}
_TARGET_FIELDS = {"parameters", "optimizer"}
_PARAMETER_FIELDS = {"incoming_weight", "bias", "outgoing_weight"}
_OPTIMIZER_FIELDS = {
    "count",
    "incoming_first_moment",
    "incoming_second_moment",
    "bias_first_moment",
    "bias_second_moment",
    "outgoing_first_moment",
    "outgoing_second_moment",
}


def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def self_normalized_resets_source_sha256() -> str:
    """Return the exact SHA-256 of this implementation source."""

    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _sha256_words_from_hex(value: str) -> UInt[Array, " 8"]:
    data = bytes.fromhex(value)
    return jnp.asarray(
        [int.from_bytes(data[offset : offset + 4], "big") for offset in range(0, 32, 4)],
        dtype=jnp.uint32,
    )


def _sha256_words(data: bytes) -> UInt[Array, " 8"]:
    return _sha256_words_from_hex(hashlib.sha256(data).hexdigest())


def _validate_sha256(name: str, value: object) -> str:
    if type(value) is not str or len(value) != 64:
        raise ValueError(f"{name} must be a lowercase 64-character SHA-256 hex digest")
    if value.lower() != value:
        raise ValueError(f"{name} must use lowercase hexadecimal")
    try:
        bytes.fromhex(value)
    except ValueError as error:
        raise ValueError(f"{name} must be hexadecimal") from error
    return value


def _finite_float32(
    name: str,
    value: object,
    *,
    positive: bool = False,
    allow_zero: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a real number")
    number = float(value)
    if not math.isfinite(number) or abs(number) > _FLOAT32_MAX:
        raise ValueError(f"{name} must be finite and representable as float32")
    if number != 0.0 and abs(number) < _FLOAT32_TINY:
        raise ValueError(f"{name} must be zero or a normal float32 value")
    if positive and number < _FLOAT32_TINY:
        raise ValueError(f"{name} must be a positive normal float32 value")
    if not allow_zero and number == 0.0 and positive:
        raise ValueError(f"{name} must be positive")
    return float(np.float32(number))


def _require_array(
    value: object,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: Any,
) -> None:
    if not hasattr(value, "shape") or not hasattr(value, "dtype"):
        raise TypeError(f"{name} must expose array shape and dtype")
    actual_shape = tuple(cast(Any, value).shape)
    if actual_shape != shape:
        raise ValueError(f"{name} must have shape {shape}; got {actual_shape}")
    expected_dtype = jnp.dtype(dtype)
    actual_dtype = jnp.dtype(cast(Any, value).dtype)
    if actual_dtype != expected_dtype:
        raise TypeError(f"{name} must have dtype {expected_dtype}; got {actual_dtype}")


def _typed_key_valid(key: object) -> bool:
    if not (
        hasattr(key, "shape")
        and tuple(cast(Any, key).shape) == ()
        and jax.dtypes.issubdtype(cast(Any, key).dtype, jax.dtypes.prng_key)
    ):
        return False
    try:
        return str(jr.key_impl(cast(Any, key))) == "threefry2x32" and tuple(
            jr.key_data(cast(Any, key)).shape
        ) == (2,)
    except (TypeError, ValueError):
        return False


def _words_from_int(value: int) -> UInt[Array, " 2"]:
    return jnp.asarray([(value >> 32) & _UINT32_MAX, value & _UINT32_MAX], dtype=jnp.uint32)


def _increment_words(words: Array) -> tuple[Array, Array]:
    """Increment uint64-as-two-uint32 words without silent wrap."""

    one = jnp.asarray(1, dtype=jnp.uint32)
    low = words[..., 1] + one
    carry = (low == jnp.asarray(0, dtype=jnp.uint32)).astype(jnp.uint32)
    high = words[..., 0] + carry
    available = ~jnp.all(words == jnp.asarray(_UINT32_MAX, dtype=jnp.uint32), axis=-1)
    candidate = jnp.stack((high, low), axis=-1).astype(jnp.uint32)
    return jnp.where(available[..., None], candidate, words), available


def _words_less(left: Array, right: Array) -> Array:
    return (left[..., 0] < right[..., 0]) | (
        (left[..., 0] == right[..., 0]) & (left[..., 1] < right[..., 1])
    )


def _words_at_least(words: Array, threshold: int) -> Array:
    threshold_words = _words_from_int(threshold)
    return ~_words_less(words, threshold_words)


def _words_to_float32(words: Array) -> Float[Array, ...]:
    return words[..., 0].astype(jnp.float32) * jnp.asarray(
        4_294_967_296.0, dtype=jnp.float32
    ) + words[..., 1].astype(jnp.float32)


def _saturating_int32_increment(value: Array) -> Array:
    maximum = jnp.asarray(_INT32_MAX, dtype=jnp.int32)
    return jnp.minimum(jnp.maximum(value, 0), maximum - 1) + 1


def _sum_nonnegative_int32_words(values: Array) -> Array:
    """Exactly sum bounded nonnegative int32 values into uint64 words."""

    def body(index: int, words: Array) -> Array:
        addend = values[index].astype(jnp.uint32)
        low = words[1] + addend
        carry = (low < words[1]).astype(jnp.uint32)
        return jnp.stack((words[0] + carry, low)).astype(jnp.uint32)

    return jax.lax.fori_loop(0, values.size, body, jnp.zeros((2,), dtype=jnp.uint32))


def _step_telemetry_valid(step_words: Array, step_count: Array) -> Array:
    below_saturation = (step_words[0] == jnp.asarray(0, dtype=jnp.uint32)) & (
        step_words[1] < jnp.asarray(_INT32_MAX, dtype=jnp.uint32)
    )
    return (step_count >= 0) & jnp.where(
        below_saturation,
        step_count == step_words[1].astype(jnp.int32),
        step_count == jnp.asarray(_INT32_MAX, dtype=jnp.int32),
    )


@dataclasses.dataclass(frozen=True)
class SelfNormalizedResetConfig:
    """Static SNR law, dense-layer shape, ownership, and resource bounds."""

    input_dim: int
    unit_count: int
    output_dim: int
    source_sha256: str
    representation_sha256: str
    window_size: int = 128
    min_intervals: int = 8
    warmup_observations: int = 32
    rejection_percentile: float = 0.001
    optimizer_kind: SelfNormalizedOptimizerKind = "none"
    initialization_mode: SelfNormalizedInitializationMode = "owned_lecun_uniform"
    initialization_scale: float = 1.0
    initial_bias: float = 0.0
    max_resets_per_step: int | None = None
    max_total_resets: int = _INT32_MAX
    maximum_updates: int = _UINT64_MAX
    enabled: bool = True

    def __post_init__(self) -> None:
        for name in ("input_dim", "unit_count", "output_dim"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
            if not 1 <= value <= _MAX_DIMENSION:
                raise ValueError(f"{name} must be in [1, {_MAX_DIMENSION}]")
        parameter_count = (
            self.input_dim * self.unit_count + self.unit_count + self.unit_count * self.output_dim
        )
        if parameter_count > _MAX_PARAMETER_COUNT:
            raise ValueError("dense-ReLU parameter count exceeds the finite cap")
        if isinstance(self.window_size, bool) or not isinstance(self.window_size, int):
            raise ValueError("window_size must be an integer")
        if not 1 <= self.window_size <= _MAX_WINDOW_SIZE:
            raise ValueError(f"window_size must be in [1, {_MAX_WINDOW_SIZE}]")
        if self.unit_count * self.window_size > _MAX_INTERVAL_SLOTS:
            raise ValueError("unit_count * window_size exceeds the interval-slot cap")
        if isinstance(self.min_intervals, bool) or not isinstance(self.min_intervals, int):
            raise ValueError("min_intervals must be an integer")
        if not 1 <= self.min_intervals <= self.window_size:
            raise ValueError("min_intervals must be in [1, window_size]")
        for name, upper in (
            ("warmup_observations", _UINT64_MAX),
            ("maximum_updates", _UINT64_MAX),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
            minimum = 0 if name == "warmup_observations" else 1
            if not minimum <= value <= upper:
                raise ValueError(f"{name} must be in [{minimum}, {upper}]")
        eta = _finite_float32("rejection_percentile", self.rejection_percentile, positive=True)
        if not eta < 1.0:
            raise ValueError("rejection_percentile must be in (0, 1)")
        object.__setattr__(self, "rejection_percentile", eta)
        object.__setattr__(
            self,
            "initialization_scale",
            _finite_float32("initialization_scale", self.initialization_scale, positive=True),
        )
        object.__setattr__(
            self,
            "initial_bias",
            _finite_float32("initial_bias", self.initial_bias, allow_zero=True),
        )
        if type(self.optimizer_kind) is not str or self.optimizer_kind not in (
            "none",
            "sgd",
            "adam",
        ):
            raise ValueError("optimizer_kind must be 'none', 'sgd', or 'adam'")
        if type(self.initialization_mode) is not str or self.initialization_mode not in (
            "owned_lecun_uniform",
            "caller_provided",
        ):
            raise ValueError(
                "initialization_mode must be 'owned_lecun_uniform' or 'caller_provided'"
            )
        if self.max_resets_per_step is None:
            object.__setattr__(self, "max_resets_per_step", self.unit_count)
        elif isinstance(self.max_resets_per_step, bool) or not isinstance(
            self.max_resets_per_step, int
        ):
            raise ValueError("max_resets_per_step must be an integer or None")
        if not 0 <= cast(int, self.max_resets_per_step) <= self.unit_count:
            raise ValueError("max_resets_per_step must be in [0, unit_count]")
        if isinstance(self.max_total_resets, bool) or not isinstance(self.max_total_resets, int):
            raise ValueError("max_total_resets must be an integer")
        if not 0 <= self.max_total_resets <= _INT32_MAX:
            raise ValueError(f"max_total_resets must be in [0, {_INT32_MAX}]")
        if type(self.enabled) is not bool:
            raise ValueError("enabled must be a bool")
        object.__setattr__(
            self, "source_sha256", _validate_sha256("source_sha256", self.source_sha256)
        )
        object.__setattr__(
            self,
            "representation_sha256",
            _validate_sha256("representation_sha256", self.representation_sha256),
        )

    @property
    def parameter_count(self) -> int:
        return (
            self.input_dim * self.unit_count + self.unit_count + self.unit_count * self.output_dim
        )

    def to_config(self) -> dict[str, object]:
        return {
            "schema": SELF_NORMALIZED_RESETS_SCHEMA,
            "type": "SelfNormalizedResetConfig",
            "input_dim": self.input_dim,
            "unit_count": self.unit_count,
            "output_dim": self.output_dim,
            "source_sha256": self.source_sha256,
            "representation_sha256": self.representation_sha256,
            "window_size": self.window_size,
            "min_intervals": self.min_intervals,
            "warmup_observations": self.warmup_observations,
            "rejection_percentile": self.rejection_percentile,
            "optimizer_kind": self.optimizer_kind,
            "initialization_mode": self.initialization_mode,
            "initialization_scale": self.initialization_scale,
            "initial_bias": self.initial_bias,
            "max_resets_per_step": self.max_resets_per_step,
            "max_total_resets": self.max_total_resets,
            "maximum_updates": self.maximum_updates,
            "enabled": self.enabled,
            "algorithm": "self-normalized-resets",
            "paper": "Farias-Jozefiak-ICLR-2025-arXiv-2410.20098",
            "activation_convention": "one-post-relu-vector-positive-means-firing",
            "geometric_support": "positive-integers",
            "completed_interval": "pre-firing-age-plus-one",
            "survival_convention": "P(A>=age+1)=(1-p)^age",
            "mean_estimator": "fixed-trailing-window-of-completed-intervals",
            "reset_ordering": "observe-then-caller-optimize-then-reset",
            "post_reset_history": "retain-window-start-new-firing-epoch",
            "warmup_ordering": "post-observation-per-reset-epoch",
            "owned_rng_consumption": "one-split-one-full-incoming-draw-per-reset-step",
            "probability_arithmetic": "float32-log-space-from-exact-uint64-word-counters",
            "support_choice_authority": "alberta-explicit-positive-support-resolution",
            "official_histogram_code_bit_equivalent": False,
            "integrity_authenticated": False,
            "scientific_evidence_claimed": False,
        }

    @classmethod
    def from_config(cls, payload: Mapping[str, object]) -> SelfNormalizedResetConfig:
        expected = cls(
            input_dim=1,
            unit_count=1,
            output_dim=1,
            source_sha256="0" * 64,
            representation_sha256="0" * 64,
        ).to_config()
        if set(payload) != set(expected):
            raise ValueError("SNR config fields do not match schema v1")
        for name in (
            "schema",
            "type",
            "algorithm",
            "paper",
            "activation_convention",
            "geometric_support",
            "completed_interval",
            "survival_convention",
            "mean_estimator",
            "reset_ordering",
            "post_reset_history",
            "warmup_ordering",
            "owned_rng_consumption",
            "probability_arithmetic",
            "support_choice_authority",
            "official_histogram_code_bit_equivalent",
            "integrity_authenticated",
            "scientific_evidence_claimed",
        ):
            if payload.get(name) != expected[name]:
                raise ValueError(f"SNR config {name} is invalid")
        integer_names = (
            "input_dim",
            "unit_count",
            "output_dim",
            "window_size",
            "min_intervals",
            "warmup_observations",
            "max_resets_per_step",
            "max_total_resets",
            "maximum_updates",
        )
        for name in integer_names:
            if type(payload.get(name)) is not int:
                raise ValueError(f"SNR config {name} must be an integer")
        for name in ("rejection_percentile", "initialization_scale", "initial_bias"):
            if type(payload.get(name)) is not float:
                raise ValueError(f"SNR config {name} must be a float")
        if type(payload.get("enabled")) is not bool:
            raise ValueError("SNR config enabled must be a bool")
        return cls(
            input_dim=cast(int, payload["input_dim"]),
            unit_count=cast(int, payload["unit_count"]),
            output_dim=cast(int, payload["output_dim"]),
            source_sha256=cast(str, payload["source_sha256"]),
            representation_sha256=cast(str, payload["representation_sha256"]),
            window_size=cast(int, payload["window_size"]),
            min_intervals=cast(int, payload["min_intervals"]),
            warmup_observations=cast(int, payload["warmup_observations"]),
            rejection_percentile=cast(float, payload["rejection_percentile"]),
            optimizer_kind=cast(SelfNormalizedOptimizerKind, payload["optimizer_kind"]),
            initialization_mode=cast(
                SelfNormalizedInitializationMode, payload["initialization_mode"]
            ),
            initialization_scale=cast(float, payload["initialization_scale"]),
            initial_bias=cast(float, payload["initial_bias"]),
            max_resets_per_step=cast(int, payload["max_resets_per_step"]),
            max_total_resets=cast(int, payload["max_total_resets"]),
            maximum_updates=cast(int, payload["maximum_updates"]),
            enabled=cast(bool, payload["enabled"]),
        )


@chex.dataclass(frozen=True)
class DenseReLUParameters:
    """The exact fixed-width dense-ReLU parameter segment consumed by SNR."""

    incoming_weight: Float[Array, "input_dim unit_count"]
    bias: Float[Array, " unit_count"]
    outgoing_weight: Float[Array, "unit_count output_dim"]


@chex.dataclass(frozen=True)
class DenseReLUOptimizerState:
    """Exact supported optimizer state; moment arrays are empty outside Adam."""

    count: Int[Array, ""]
    incoming_first_moment: Float[Array, ...]
    incoming_second_moment: Float[Array, ...]
    bias_first_moment: Float[Array, ...]
    bias_second_moment: Float[Array, ...]
    outgoing_first_moment: Float[Array, ...]
    outgoing_second_moment: Float[Array, ...]


@chex.dataclass(frozen=True)
class DenseReLUResetTarget:
    """Parameters and exact optimizer state after the caller's optimizer step."""

    parameters: DenseReLUParameters
    optimizer: DenseReLUOptimizerState


@chex.dataclass(frozen=True)
class DenseReLUFreshParameters:
    """Caller-owned fresh incoming values for ``caller_provided`` mode."""

    incoming_weight: Float[Array, "input_dim unit_count"]
    bias: Float[Array, " unit_count"]


@chex.dataclass(frozen=True)
class SelfNormalizedResetState:
    """Frozen detector, reset consumer, PRNG, binding, and integrity state."""

    rng_key: PRNGKeyArray
    target: DenseReLUResetTarget
    ages_words: UInt[Array, "unit_count 2"]
    epoch_observations_words: UInt[Array, "unit_count 2"]
    has_fired: Bool[Array, " unit_count"]
    intervals_words: UInt[Array, "unit_count window_size 2"]
    interval_count: Int[Array, " unit_count"]
    interval_cursor: Int[Array, " unit_count"]
    unit_reset_count: Int[Array, " unit_count"]
    total_reset_count: Int[Array, ""]
    step_words: UInt[Array, " 2"]
    step_count: Int[Array, ""]
    source_binding_words: UInt[Array, " 8"]
    representation_binding_words: UInt[Array, " 8"]
    integrity_tag: UInt[Array, " 4"]


@chex.dataclass(frozen=True)
class SelfNormalizedResetResult:
    """Result of one atomic post-optimizer SNR transaction."""

    state: SelfNormalizedResetState
    reset_mask: Bool[Array, " unit_count"]
    eligible_mask: Bool[Array, " unit_count"]
    estimated_mean_interval: Float[Array, " unit_count"]
    log_survival: Float[Array, " unit_count"]
    completed_interval_mask: Bool[Array, " unit_count"]
    activation_valid: Bool[Array, ""]
    target_valid: Bool[Array, ""]
    fresh_values_valid: Bool[Array, ""]
    state_valid: Bool[Array, ""]
    lifetime_capacity_available: Bool[Array, ""]
    tracker_capacity_available: Bool[Array, ""]
    update_applied: Bool[Array, ""]
    update_rejected: Bool[Array, ""]
    reset_count: Int[Array, ""]
    pre_step_words: UInt[Array, " 2"]
    post_step_words: UInt[Array, " 2"]


@chex.dataclass(frozen=True)
class SelfNormalizedResetScanResult:
    """Fixed-shape outputs of :meth:`SelfNormalizedResets.run_scan`."""

    state: SelfNormalizedResetState
    reset_masks: Bool[Array, "steps unit_count"]
    log_survivals: Float[Array, "steps unit_count"]
    update_applied: Bool[Array, " steps"]


@dataclasses.dataclass(frozen=True)
class SelfNormalizedResetResourceDeclaration:
    """Exact persistent size and bounded source-level scratch accounting."""

    unit_count: int
    window_size: int
    interval_slots: int
    dense_parameter_count: int
    persistent_state_bytes: int
    maximum_resets_per_step: int
    maximum_total_resets: int
    maximum_updates: int
    maximum_random_float32_draws_per_step: int
    maximum_temporary_bytes_per_step: int
    temporary_bytes_scope: str
    checkpoint_host_only: bool
    integrity_authenticated: bool
    external_side_effects: bool


def empty_dense_relu_optimizer_state() -> DenseReLUOptimizerState:
    """Return the canonical optimizer payload for ``none`` or stateless SGD."""

    empty = jnp.zeros((0,), dtype=jnp.float32)
    return DenseReLUOptimizerState(
        count=jnp.asarray(0, dtype=jnp.int32),
        incoming_first_moment=empty,
        incoming_second_moment=empty,
        bias_first_moment=empty,
        bias_second_moment=empty,
        outgoing_first_moment=empty,
        outgoing_second_moment=empty,
    )


def zero_adam_state(parameters: DenseReLUParameters) -> DenseReLUOptimizerState:
    """Return the exact supported Adam moments initialized to zero."""

    return DenseReLUOptimizerState(
        count=jnp.asarray(0, dtype=jnp.int32),
        incoming_first_moment=jnp.zeros_like(parameters.incoming_weight),
        incoming_second_moment=jnp.zeros_like(parameters.incoming_weight),
        bias_first_moment=jnp.zeros_like(parameters.bias),
        bias_second_moment=jnp.zeros_like(parameters.bias),
        outgoing_first_moment=jnp.zeros_like(parameters.outgoing_weight),
        outgoing_second_moment=jnp.zeros_like(parameters.outgoing_weight),
    )


def _array_u32_bits(array: Array) -> Array:
    dtype = jnp.dtype(array.dtype)
    if dtype == jnp.dtype(jnp.uint32):
        return array.reshape(-1)
    if dtype in (jnp.dtype(jnp.int32), jnp.dtype(jnp.float32)):
        return jax.lax.bitcast_convert_type(array, jnp.uint32).reshape(-1)
    if dtype == jnp.dtype(jnp.bool_):
        return array.astype(jnp.uint32).reshape(-1)
    raise TypeError(f"integrity field has unsupported dtype {dtype}")


def _mix_integrity_words(tag: Array, words: Array, field_index: int) -> Array:
    """Mix one canonical uint32 field into a small unkeyed live-state tag."""

    field_salt = jnp.asarray(((field_index + 1) * 0x9E3779B1) & _UINT32_MAX, dtype=jnp.uint32)
    primes = jnp.asarray([0x01000193, 0x85EBCA6B, 0xC2B2AE35, 0x27D4EB2F], dtype=jnp.uint32)
    lane_salts = jnp.asarray([0xA5A5A5A5, 0x3C6EF372, 0xBB67AE85, 0x1B873593], dtype=jnp.uint32)

    def body(index: int, accumulator: Array) -> Array:
        position = jnp.asarray(index + 1, dtype=jnp.uint32)
        value = words[index] + field_salt + position * jnp.asarray(0x7F4A7C15, dtype=jnp.uint32)
        rotated = jnp.stack(
            (
                value,
                (value << jnp.asarray(7, dtype=jnp.uint32))
                | (value >> jnp.asarray(25, dtype=jnp.uint32)),
                (value << jnp.asarray(13, dtype=jnp.uint32))
                | (value >> jnp.asarray(19, dtype=jnp.uint32)),
                (value << jnp.asarray(21, dtype=jnp.uint32))
                | (value >> jnp.asarray(11, dtype=jnp.uint32)),
            )
        )
        return ((accumulator ^ rotated ^ lane_salts) * primes).astype(jnp.uint32)

    mixed = tag if words.size == 0 else jax.lax.fori_loop(0, words.size, body, tag)
    return (mixed ^ jnp.asarray(words.size, dtype=jnp.uint32) ^ field_salt).astype(jnp.uint32)


class SelfNormalizedResets:
    """Paper-grounded SNR tracker and exact dense-ReLU reset consumer."""

    def __init__(self, config: SelfNormalizedResetConfig) -> None:
        if type(config) is not SelfNormalizedResetConfig:
            raise TypeError("config must be SelfNormalizedResetConfig")
        self._config = config
        self._config_id_words = _sha256_words(_canonical_json_bytes(config.to_config()))
        self._source_words = _sha256_words_from_hex(config.source_sha256)
        self._representation_words = _sha256_words_from_hex(config.representation_sha256)
        self._maximum_update_words = _words_from_int(config.maximum_updates)

    @property
    def config(self) -> SelfNormalizedResetConfig:
        return self._config

    def to_config(self) -> dict[str, object]:
        return self._config.to_config()

    @classmethod
    def from_config(cls, payload: Mapping[str, object]) -> SelfNormalizedResets:
        return cls(SelfNormalizedResetConfig.from_config(payload))

    def _validate_parameters_static(self, parameters: DenseReLUParameters) -> None:
        if type(parameters) is not DenseReLUParameters:
            raise TypeError("target.parameters must be DenseReLUParameters")
        cfg = self._config
        _require_array(
            parameters.incoming_weight,
            name="target.parameters.incoming_weight",
            shape=(cfg.input_dim, cfg.unit_count),
            dtype=jnp.float32,
        )
        _require_array(
            parameters.bias,
            name="target.parameters.bias",
            shape=(cfg.unit_count,),
            dtype=jnp.float32,
        )
        _require_array(
            parameters.outgoing_weight,
            name="target.parameters.outgoing_weight",
            shape=(cfg.unit_count, cfg.output_dim),
            dtype=jnp.float32,
        )

    def _validate_optimizer_static(self, optimizer: DenseReLUOptimizerState) -> None:
        if type(optimizer) is not DenseReLUOptimizerState:
            raise TypeError("target.optimizer must be DenseReLUOptimizerState")
        _require_array(optimizer.count, name="target.optimizer.count", shape=(), dtype=jnp.int32)
        cfg = self._config
        if cfg.optimizer_kind == "adam":
            shapes = {
                "incoming_first_moment": (cfg.input_dim, cfg.unit_count),
                "incoming_second_moment": (cfg.input_dim, cfg.unit_count),
                "bias_first_moment": (cfg.unit_count,),
                "bias_second_moment": (cfg.unit_count,),
                "outgoing_first_moment": (cfg.unit_count, cfg.output_dim),
                "outgoing_second_moment": (cfg.unit_count, cfg.output_dim),
            }
        else:
            shapes = {name: (0,) for name in _OPTIMIZER_FIELDS if name != "count"}
        for name, shape in shapes.items():
            _require_array(
                getattr(optimizer, name),
                name=f"target.optimizer.{name}",
                shape=shape,
                dtype=jnp.float32,
            )

    def _validate_target_static(self, target: DenseReLUResetTarget) -> None:
        if type(target) is not DenseReLUResetTarget:
            raise TypeError("post_optimizer_target must be DenseReLUResetTarget")
        self._validate_parameters_static(target.parameters)
        self._validate_optimizer_static(target.optimizer)

    def _target_valid(self, target: DenseReLUResetTarget) -> Array:
        leaves_finite = jnp.asarray(True, dtype=jnp.bool_)
        for leaf in jax.tree_util.tree_leaves(target):
            if jnp.issubdtype(leaf.dtype, jnp.inexact):
                leaves_finite = leaves_finite & jnp.all(jnp.isfinite(leaf))
        count_valid = target.optimizer.count >= 0
        if self._config.optimizer_kind != "adam":
            count_valid = count_valid & (target.optimizer.count == 0)
        return leaves_finite & count_valid

    def _validate_fresh_static(self, fresh: DenseReLUFreshParameters) -> None:
        if type(fresh) is not DenseReLUFreshParameters:
            raise TypeError("fresh_parameters must be DenseReLUFreshParameters")
        cfg = self._config
        _require_array(
            fresh.incoming_weight,
            name="fresh_parameters.incoming_weight",
            shape=(cfg.input_dim, cfg.unit_count),
            dtype=jnp.float32,
        )
        _require_array(
            fresh.bias,
            name="fresh_parameters.bias",
            shape=(cfg.unit_count,),
            dtype=jnp.float32,
        )

    @staticmethod
    def _fresh_valid(fresh: DenseReLUFreshParameters) -> Array:
        return jnp.all(jnp.isfinite(fresh.incoming_weight)) & jnp.all(jnp.isfinite(fresh.bias))

    def _validate_state_static(self, state: SelfNormalizedResetState) -> None:
        if type(state) is not SelfNormalizedResetState:
            raise TypeError("state must be SelfNormalizedResetState")
        if not _typed_key_valid(state.rng_key):
            raise TypeError("state.rng_key must be a typed scalar threefry2x32 JAX PRNG key")
        self._validate_target_static(state.target)
        cfg = self._config
        contracts = {
            "ages_words": ((cfg.unit_count, 2), jnp.uint32),
            "epoch_observations_words": ((cfg.unit_count, 2), jnp.uint32),
            "has_fired": ((cfg.unit_count,), jnp.bool_),
            "intervals_words": (
                (cfg.unit_count, cfg.window_size, 2),
                jnp.uint32,
            ),
            "interval_count": ((cfg.unit_count,), jnp.int32),
            "interval_cursor": ((cfg.unit_count,), jnp.int32),
            "unit_reset_count": ((cfg.unit_count,), jnp.int32),
            "total_reset_count": ((), jnp.int32),
            "step_words": ((2,), jnp.uint32),
            "step_count": ((), jnp.int32),
            "source_binding_words": ((8,), jnp.uint32),
            "representation_binding_words": ((8,), jnp.uint32),
            "integrity_tag": ((4,), jnp.uint32),
        }
        for name, (shape, dtype) in contracts.items():
            _require_array(getattr(state, name), name=f"state.{name}", shape=shape, dtype=dtype)

    def _integrity_tag(self, state: SelfNormalizedResetState) -> Array:
        cfg_seed = self._config_id_words
        tag = (cfg_seed[:4] ^ cfg_seed[4:]).astype(jnp.uint32)
        arrays = [
            jr.key_data(state.rng_key),
            state.target.parameters.incoming_weight,
            state.target.parameters.bias,
            state.target.parameters.outgoing_weight,
            state.target.optimizer.count,
            state.target.optimizer.incoming_first_moment,
            state.target.optimizer.incoming_second_moment,
            state.target.optimizer.bias_first_moment,
            state.target.optimizer.bias_second_moment,
            state.target.optimizer.outgoing_first_moment,
            state.target.optimizer.outgoing_second_moment,
            state.ages_words,
            state.epoch_observations_words,
            state.has_fired,
            state.intervals_words,
            state.interval_count,
            state.interval_cursor,
            state.unit_reset_count,
            state.total_reset_count,
            state.step_words,
            state.step_count,
            state.source_binding_words,
            state.representation_binding_words,
        ]
        for field_index, array in enumerate(arrays):
            tag = _mix_integrity_words(tag, _array_u32_bits(array), field_index)
        return tag.astype(jnp.uint32)

    def _state_valid(self, state: SelfNormalizedResetState) -> Array:
        cfg = self._config
        count_valid = (state.interval_count >= 0) & (state.interval_count <= cfg.window_size)
        cursor_valid = (state.interval_cursor >= 0) & (state.interval_cursor < cfg.window_size)
        cursor_valid = cursor_valid & jnp.where(
            state.interval_count < cfg.window_size,
            state.interval_cursor == state.interval_count,
            jnp.asarray(True, dtype=jnp.bool_),
        )
        slot_indices = jnp.arange(cfg.window_size, dtype=jnp.int32)[None, :]
        used_slots = (state.interval_count[:, None] == cfg.window_size) | (
            slot_indices < state.interval_count[:, None]
        )
        nonzero_intervals = jnp.any(state.intervals_words != 0, axis=-1)
        interval_layout_valid = jnp.all(used_slots == nonzero_intervals)
        interval_not_after_step = _words_less(state.intervals_words, state.step_words) | jnp.all(
            state.intervals_words == state.step_words[None, None, :], axis=-1
        )
        age_is_zero = jnp.all(state.ages_words == 0, axis=-1)
        age_not_after_epoch = _words_less(
            state.ages_words, state.epoch_observations_words
        ) | jnp.all(state.ages_words == state.epoch_observations_words, axis=-1)
        epoch_not_after_step = _words_less(
            state.epoch_observations_words, state.step_words
        ) | jnp.all(state.epoch_observations_words == state.step_words[None, :], axis=-1)
        reset_count_words = _sum_nonnegative_int32_words(state.unit_reset_count)
        expected_reset_count_words = jnp.stack(
            (
                jnp.asarray(0, dtype=jnp.uint32),
                state.total_reset_count.astype(jnp.uint32),
            )
        )
        reset_counts_valid = (
            jnp.all(state.unit_reset_count >= 0)
            & (state.total_reset_count >= 0)
            & (state.total_reset_count <= cfg.max_total_resets)
            & jnp.all(state.unit_reset_count <= state.total_reset_count)
            & jnp.all(reset_count_words == expected_reset_count_words)
        )
        under_update_cap = _words_less(state.step_words, self._maximum_update_words) | jnp.all(
            state.step_words == self._maximum_update_words
        )
        return (
            self._target_valid(state.target)
            & jnp.all(count_valid)
            & jnp.all(cursor_valid)
            & interval_layout_valid
            & jnp.all(interval_not_after_step)
            & jnp.all(state.has_fired | age_is_zero)
            & jnp.all(age_not_after_epoch)
            & jnp.all(epoch_not_after_step)
            & reset_counts_valid
            & _step_telemetry_valid(state.step_words, state.step_count)
            & under_update_cap
            & jnp.all(state.source_binding_words == self._source_words)
            & jnp.all(state.representation_binding_words == self._representation_words)
            & jnp.all(state.integrity_tag == self._integrity_tag(state))
        )

    def init(
        self,
        rng_key: PRNGKeyArray,
        target: DenseReLUResetTarget,
    ) -> SelfNormalizedResetState:
        """Initialize a detector around caller-owned dense parameters."""

        if not _typed_key_valid(rng_key):
            raise TypeError("rng_key must be a typed scalar threefry2x32 JAX PRNG key")
        self._validate_target_static(target)
        if not bool(np.asarray(self._target_valid(target))):
            raise ValueError("target must be finite and optimizer state must be valid")
        cfg = self._config
        state = SelfNormalizedResetState(
            rng_key=rng_key,
            target=target,
            ages_words=jnp.zeros((cfg.unit_count, 2), dtype=jnp.uint32),
            epoch_observations_words=jnp.zeros((cfg.unit_count, 2), dtype=jnp.uint32),
            has_fired=jnp.zeros((cfg.unit_count,), dtype=jnp.bool_),
            intervals_words=jnp.zeros((cfg.unit_count, cfg.window_size, 2), dtype=jnp.uint32),
            interval_count=jnp.zeros((cfg.unit_count,), dtype=jnp.int32),
            interval_cursor=jnp.zeros((cfg.unit_count,), dtype=jnp.int32),
            unit_reset_count=jnp.zeros((cfg.unit_count,), dtype=jnp.int32),
            total_reset_count=jnp.asarray(0, dtype=jnp.int32),
            step_words=jnp.zeros((2,), dtype=jnp.uint32),
            step_count=jnp.asarray(0, dtype=jnp.int32),
            source_binding_words=self._source_words,
            representation_binding_words=self._representation_words,
            integrity_tag=jnp.zeros((4,), dtype=jnp.uint32),
        )
        state = state.replace(integrity_tag=self._integrity_tag(state))
        if not bool(np.asarray(self._state_valid(state))):
            raise RuntimeError("constructed SNR state failed its integrity contract")
        return state

    def _fresh_placeholder(self) -> DenseReLUFreshParameters:
        cfg = self._config
        return DenseReLUFreshParameters(
            incoming_weight=jnp.zeros((cfg.input_dim, cfg.unit_count), dtype=jnp.float32),
            bias=jnp.zeros((cfg.unit_count,), dtype=jnp.float32),
        )

    def step(
        self,
        state: SelfNormalizedResetState,
        relu_activations: Float[Array, " unit_count"],
        post_optimizer_target: DenseReLUResetTarget,
        fresh_parameters: DenseReLUFreshParameters | None = None,
    ) -> SelfNormalizedResetResult:
        """Observe one activation and apply resets after caller optimization.

        ``post_optimizer_target`` is the dense segment after the caller has
        applied its ordinary gradient/optimizer update.  The returned state's
        target is exactly that value except at selected unit slices.
        """

        self._validate_state_static(state)
        self._validate_target_static(post_optimizer_target)
        _require_array(
            relu_activations,
            name="relu_activations",
            shape=(self._config.unit_count,),
            dtype=jnp.float32,
        )
        if not self._config.enabled and fresh_parameters is None:
            fresh = self._fresh_placeholder()
        elif self._config.initialization_mode == "owned_lecun_uniform":
            if fresh_parameters is not None:
                raise ValueError("owned initialization does not accept fresh_parameters")
            fresh = self._fresh_placeholder()
        else:
            if fresh_parameters is None:
                raise ValueError("caller_provided initialization requires fresh_parameters")
            self._validate_fresh_static(fresh_parameters)
            fresh = fresh_parameters
        return cast(
            SelfNormalizedResetResult,
            self._step_jit(state, relu_activations, post_optimizer_target, fresh),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def _step_jit(
        self,
        state: SelfNormalizedResetState,
        relu_activations: Array,
        post_optimizer_target: DenseReLUResetTarget,
        fresh: DenseReLUFreshParameters,
    ) -> SelfNormalizedResetResult:
        return self._step_kernel(state, relu_activations, post_optimizer_target, fresh)

    def _step_kernel(
        self,
        state: SelfNormalizedResetState,
        relu_activations: Array,
        post_optimizer_target: DenseReLUResetTarget,
        fresh: DenseReLUFreshParameters,
    ) -> SelfNormalizedResetResult:
        cfg = self._config
        state_valid = self._state_valid(state)
        target_valid = self._target_valid(post_optimizer_target)
        activation_valid = jnp.all(jnp.isfinite(relu_activations)) & jnp.all(
            relu_activations >= jnp.asarray(0.0, dtype=jnp.float32)
        )
        fresh_values_valid = (
            self._fresh_valid(fresh)
            if cfg.initialization_mode == "caller_provided"
            else jnp.asarray(True, dtype=jnp.bool_)
        )

        next_step_words, raw_lifetime_capacity = _increment_words(state.step_words)
        lifetime_capacity = raw_lifetime_capacity & _words_less(
            state.step_words, self._maximum_update_words
        )
        next_step_count = _saturating_int32_increment(state.step_count)

        firing = relu_activations > jnp.asarray(0.0, dtype=jnp.float32)
        next_epoch_words, epoch_capacity = _increment_words(state.epoch_observations_words)
        next_age_words, age_capacity = _increment_words(state.ages_words)
        completed_interval_words, interval_capacity = _increment_words(state.ages_words)
        completed_interval = firing & state.has_fired
        age_increment_needed = (~firing) & state.has_fired
        tracker_capacity = (
            jnp.all(epoch_capacity)
            & jnp.all((~age_increment_needed) | age_capacity)
            & jnp.all((~completed_interval) | interval_capacity)
        )

        zero_unit_words = jnp.zeros((cfg.unit_count, 2), dtype=jnp.uint32)
        observed_ages = jnp.where(
            firing[:, None],
            zero_unit_words,
            jnp.where(state.has_fired[:, None], next_age_words, state.ages_words),
        )
        observed_has_fired = state.has_fired | firing

        rows = jnp.arange(cfg.unit_count, dtype=jnp.int32)
        safe_cursor = jnp.clip(state.interval_cursor, 0, cfg.window_size - 1)
        prior_interval_at_cursor = state.intervals_words[rows, safe_cursor]
        interval_at_cursor = jnp.where(
            completed_interval[:, None],
            completed_interval_words,
            prior_interval_at_cursor,
        )
        observed_intervals = state.intervals_words.at[rows, safe_cursor].set(interval_at_cursor)
        count_can_grow = state.interval_count < cfg.window_size
        observed_interval_count = state.interval_count + (
            completed_interval & count_can_grow
        ).astype(jnp.int32)
        observed_interval_cursor = jnp.where(
            completed_interval,
            (safe_cursor + jnp.asarray(1, dtype=jnp.int32)) % cfg.window_size,
            state.interval_cursor,
        )

        interval_values = _words_to_float32(observed_intervals)
        slot_indices = jnp.arange(cfg.window_size, dtype=jnp.int32)[None, :]
        used_slots = (observed_interval_count[:, None] == cfg.window_size) | (
            slot_indices < observed_interval_count[:, None]
        )
        interval_total = jnp.sum(
            jnp.where(used_slots, interval_values, 0.0), axis=1, dtype=jnp.float32
        )
        safe_interval_count = jnp.maximum(observed_interval_count, 1).astype(jnp.float32)
        mean_interval = interval_total / safe_interval_count
        mean_interval = jnp.where(
            observed_interval_count > 0,
            mean_interval,
            jnp.asarray(1.0, dtype=jnp.float32),
        )
        firing_probability = jnp.reciprocal(
            jnp.maximum(mean_interval, jnp.asarray(1.0, dtype=jnp.float32))
        )
        # Direct log1p is essential here. Materializing ``1 - p`` first rounds
        # to exactly one for small-but-representable p (large learned means),
        # which would incorrectly give log survival zero at every age.
        log_failure = jnp.where(
            firing_probability == jnp.asarray(1.0, dtype=jnp.float32),
            jnp.asarray(-jnp.inf, dtype=jnp.float32),
            jnp.log1p(-firing_probability),
        )
        age_float = _words_to_float32(observed_ages)
        survival_exponent = age_float
        log_survival = jnp.where(
            survival_exponent == 0.0,
            jnp.asarray(0.0, dtype=jnp.float32),
            survival_exponent * log_failure,
        )
        history_ready = observed_interval_count >= cfg.min_intervals
        warmup_ready = _words_at_least(next_epoch_words, cfg.warmup_observations)
        eligible_mask = (~firing) & observed_has_fired & history_ready & warmup_ready
        raw_reset_mask = eligible_mask & (
            log_survival <= jnp.log(jnp.asarray(cfg.rejection_percentile, dtype=jnp.float32))
        )
        reset_rank = jnp.cumsum(raw_reset_mask.astype(jnp.int32), dtype=jnp.int32)
        remaining_total = jnp.maximum(
            jnp.asarray(cfg.max_total_resets, dtype=jnp.int32) - state.total_reset_count,
            jnp.asarray(0, dtype=jnp.int32),
        )
        reset_allowance = jnp.minimum(
            jnp.asarray(cast(int, cfg.max_resets_per_step), dtype=jnp.int32),
            remaining_total,
        )
        bounded_reset_mask = (
            raw_reset_mask & (reset_rank <= reset_allowance) & (reset_allowance > 0)
        )

        common_valid = state_valid & target_valid & activation_valid & fresh_values_valid
        transaction_applied = common_valid
        if cfg.enabled:
            transaction_applied = transaction_applied & lifetime_capacity & tracker_capacity
            effective_reset_mask = bounded_reset_mask & transaction_applied
        else:
            effective_reset_mask = jnp.zeros((cfg.unit_count,), dtype=jnp.bool_)
        effective_reset_count = jnp.sum(effective_reset_mask.astype(jnp.int32), dtype=jnp.int32)

        if cfg.initialization_mode == "owned_lecun_uniform":
            any_reset = jnp.any(effective_reset_mask)

            def draw_fresh(_: None) -> tuple[Array, Array, Array]:
                next_key, draw_key = jr.split(state.rng_key)
                limit = jnp.asarray(
                    math.sqrt(3.0 / cfg.input_dim) * cfg.initialization_scale,
                    dtype=jnp.float32,
                )
                incoming = jr.uniform(
                    draw_key,
                    shape=(cfg.input_dim, cfg.unit_count),
                    dtype=jnp.float32,
                    minval=-limit,
                    maxval=limit,
                )
                bias = jnp.full((cfg.unit_count,), cfg.initial_bias, dtype=jnp.float32)
                return next_key, incoming, bias

            def retain_key(_: None) -> tuple[Array, Array, Array]:
                placeholder = self._fresh_placeholder()
                return state.rng_key, placeholder.incoming_weight, placeholder.bias

            next_rng_key, fresh_incoming, fresh_bias = jax.lax.cond(
                any_reset, draw_fresh, retain_key, operand=None
            )
        else:
            next_rng_key = state.rng_key
            fresh_incoming = fresh.incoming_weight
            fresh_bias = fresh.bias

        parameters = post_optimizer_target.parameters
        reset_parameters = DenseReLUParameters(
            incoming_weight=jnp.where(
                effective_reset_mask[None, :],
                fresh_incoming,
                parameters.incoming_weight,
            ),
            bias=jnp.where(effective_reset_mask, fresh_bias, parameters.bias),
            outgoing_weight=jnp.where(
                effective_reset_mask[:, None],
                jnp.asarray(0.0, dtype=jnp.float32),
                parameters.outgoing_weight,
            ),
        )
        optimizer = post_optimizer_target.optimizer
        if cfg.optimizer_kind == "adam":
            reset_optimizer = DenseReLUOptimizerState(
                count=optimizer.count,
                incoming_first_moment=jnp.where(
                    effective_reset_mask[None, :],
                    jnp.asarray(0.0, dtype=jnp.float32),
                    optimizer.incoming_first_moment,
                ),
                incoming_second_moment=jnp.where(
                    effective_reset_mask[None, :],
                    jnp.asarray(0.0, dtype=jnp.float32),
                    optimizer.incoming_second_moment,
                ),
                bias_first_moment=jnp.where(
                    effective_reset_mask,
                    jnp.asarray(0.0, dtype=jnp.float32),
                    optimizer.bias_first_moment,
                ),
                bias_second_moment=jnp.where(
                    effective_reset_mask,
                    jnp.asarray(0.0, dtype=jnp.float32),
                    optimizer.bias_second_moment,
                ),
                outgoing_first_moment=jnp.where(
                    effective_reset_mask[:, None],
                    jnp.asarray(0.0, dtype=jnp.float32),
                    optimizer.outgoing_first_moment,
                ),
                outgoing_second_moment=jnp.where(
                    effective_reset_mask[:, None],
                    jnp.asarray(0.0, dtype=jnp.float32),
                    optimizer.outgoing_second_moment,
                ),
            )
        else:
            reset_optimizer = optimizer
        reset_target = DenseReLUResetTarget(
            parameters=reset_parameters,
            optimizer=reset_optimizer,
        )

        if cfg.enabled:
            candidate_ages = jnp.where(
                effective_reset_mask[:, None], zero_unit_words, observed_ages
            )
            candidate_epoch_words = jnp.where(
                effective_reset_mask[:, None], zero_unit_words, next_epoch_words
            )
            candidate_has_fired = observed_has_fired & (~effective_reset_mask)
            candidate_unit_reset_count = state.unit_reset_count + (
                effective_reset_mask.astype(jnp.int32)
            )
            candidate_state = state.replace(
                rng_key=next_rng_key,
                target=reset_target,
                ages_words=candidate_ages,
                epoch_observations_words=candidate_epoch_words,
                has_fired=candidate_has_fired,
                intervals_words=observed_intervals,
                interval_count=observed_interval_count,
                interval_cursor=observed_interval_cursor,
                unit_reset_count=candidate_unit_reset_count,
                total_reset_count=state.total_reset_count + effective_reset_count,
                step_words=next_step_words,
                step_count=next_step_count,
                integrity_tag=jnp.zeros((4,), dtype=jnp.uint32),
            )
        else:
            candidate_state = state.replace(
                target=post_optimizer_target,
                integrity_tag=jnp.zeros((4,), dtype=jnp.uint32),
            )
        candidate_state = candidate_state.replace(
            integrity_tag=self._integrity_tag(candidate_state)
        )
        output_state = jax.lax.cond(
            transaction_applied,
            lambda _: candidate_state,
            lambda _: state,
            operand=None,
        )
        diagnostic_mask = jnp.full(
            (cfg.unit_count,), transaction_applied & cfg.enabled, dtype=jnp.bool_
        )
        reported_reset_mask = effective_reset_mask & diagnostic_mask
        reported_eligible = eligible_mask & diagnostic_mask
        reported_completed = completed_interval & diagnostic_mask
        reported_mean = jnp.where(
            diagnostic_mask,
            mean_interval,
            jnp.asarray(0.0, dtype=jnp.float32),
        )
        reported_log_survival = jnp.where(
            diagnostic_mask,
            log_survival,
            jnp.asarray(0.0, dtype=jnp.float32),
        )
        return SelfNormalizedResetResult(
            state=output_state,
            reset_mask=reported_reset_mask,
            eligible_mask=reported_eligible,
            estimated_mean_interval=reported_mean,
            log_survival=reported_log_survival,
            completed_interval_mask=reported_completed,
            activation_valid=activation_valid,
            target_valid=target_valid,
            fresh_values_valid=fresh_values_valid,
            state_valid=state_valid,
            lifetime_capacity_available=(
                lifetime_capacity if cfg.enabled else jnp.asarray(True, dtype=jnp.bool_)
            ),
            tracker_capacity_available=(
                tracker_capacity if cfg.enabled else jnp.asarray(True, dtype=jnp.bool_)
            ),
            update_applied=transaction_applied,
            update_rejected=~transaction_applied,
            reset_count=jnp.sum(reported_reset_mask.astype(jnp.int32), dtype=jnp.int32),
            pre_step_words=state.step_words,
            post_step_words=output_state.step_words,
        )

    def run_scan(
        self,
        state: SelfNormalizedResetState,
        relu_activations: Float[Array, "steps unit_count"],
        post_optimizer_targets: DenseReLUResetTarget,
        fresh_parameters: DenseReLUFreshParameters | None = None,
    ) -> SelfNormalizedResetScanResult:
        """Run the same atomic kernel through ``jax.lax.scan``.

        Every target/fresh array carries a leading step dimension.  This method
        exists to make scan allocation and parity explicit; it does not infer or
        perform optimizer updates.
        """

        self._validate_state_static(state)
        if not hasattr(relu_activations, "shape") or relu_activations.ndim != 2:
            raise ValueError("relu_activations must have shape (steps, unit_count)")
        steps = relu_activations.shape[0]
        _require_array(
            relu_activations,
            name="relu_activations",
            shape=(steps, self._config.unit_count),
            dtype=jnp.float32,
        )
        self._validate_target_sequence_static(post_optimizer_targets, steps)
        if not self._config.enabled and fresh_parameters is None:
            placeholder = self._fresh_placeholder()
            fresh_sequence = jax.tree_util.tree_map(
                lambda value: jnp.broadcast_to(value, (steps, *value.shape)),
                placeholder,
            )
        elif self._config.initialization_mode == "owned_lecun_uniform":
            if fresh_parameters is not None:
                raise ValueError("owned initialization does not accept fresh_parameters")
            placeholder = self._fresh_placeholder()
            fresh_sequence = jax.tree_util.tree_map(
                lambda value: jnp.broadcast_to(value, (steps, *value.shape)),
                placeholder,
            )
        else:
            if fresh_parameters is None:
                raise ValueError("caller_provided initialization requires fresh_parameters")
            self._validate_fresh_sequence_static(fresh_parameters, steps)
            fresh_sequence = fresh_parameters

        def scan_body(
            carry: SelfNormalizedResetState,
            inputs: tuple[Array, DenseReLUResetTarget, DenseReLUFreshParameters],
        ) -> tuple[SelfNormalizedResetState, tuple[Array, Array, Array]]:
            activation, target, fresh_value = inputs
            result = self._step_kernel(carry, activation, target, fresh_value)
            return result.state, (
                result.reset_mask,
                result.log_survival,
                result.update_applied,
            )

        final_state, outputs = jax.lax.scan(
            scan_body,
            state,
            (relu_activations, post_optimizer_targets, fresh_sequence),
        )
        reset_masks, log_survivals, applied = outputs
        return SelfNormalizedResetScanResult(
            state=final_state,
            reset_masks=reset_masks,
            log_survivals=log_survivals,
            update_applied=applied,
        )

    def _validate_target_sequence_static(self, target: DenseReLUResetTarget, steps: int) -> None:
        if type(target) is not DenseReLUResetTarget:
            raise TypeError("post_optimizer_targets must be DenseReLUResetTarget")
        cfg = self._config
        if type(target.parameters) is not DenseReLUParameters:
            raise TypeError("post_optimizer_targets.parameters has invalid type")
        parameter_shapes = {
            "incoming_weight": (steps, cfg.input_dim, cfg.unit_count),
            "bias": (steps, cfg.unit_count),
            "outgoing_weight": (steps, cfg.unit_count, cfg.output_dim),
        }
        for name, shape in parameter_shapes.items():
            _require_array(
                getattr(target.parameters, name),
                name=f"post_optimizer_targets.parameters.{name}",
                shape=shape,
                dtype=jnp.float32,
            )
        if type(target.optimizer) is not DenseReLUOptimizerState:
            raise TypeError("post_optimizer_targets.optimizer has invalid type")
        _require_array(
            target.optimizer.count,
            name="post_optimizer_targets.optimizer.count",
            shape=(steps,),
            dtype=jnp.int32,
        )
        if cfg.optimizer_kind == "adam":
            optimizer_shapes = {
                "incoming_first_moment": (steps, cfg.input_dim, cfg.unit_count),
                "incoming_second_moment": (steps, cfg.input_dim, cfg.unit_count),
                "bias_first_moment": (steps, cfg.unit_count),
                "bias_second_moment": (steps, cfg.unit_count),
                "outgoing_first_moment": (steps, cfg.unit_count, cfg.output_dim),
                "outgoing_second_moment": (steps, cfg.unit_count, cfg.output_dim),
            }
        else:
            optimizer_shapes = {name: (steps, 0) for name in _OPTIMIZER_FIELDS if name != "count"}
        for name, shape in optimizer_shapes.items():
            _require_array(
                getattr(target.optimizer, name),
                name=f"post_optimizer_targets.optimizer.{name}",
                shape=shape,
                dtype=jnp.float32,
            )

    def _validate_fresh_sequence_static(self, fresh: DenseReLUFreshParameters, steps: int) -> None:
        if type(fresh) is not DenseReLUFreshParameters:
            raise TypeError("fresh_parameters must be DenseReLUFreshParameters")
        cfg = self._config
        _require_array(
            fresh.incoming_weight,
            name="fresh_parameters.incoming_weight",
            shape=(steps, cfg.input_dim, cfg.unit_count),
            dtype=jnp.float32,
        )
        _require_array(
            fresh.bias,
            name="fresh_parameters.bias",
            shape=(steps, cfg.unit_count),
            dtype=jnp.float32,
        )

    def resource_declaration(
        self, state: SelfNormalizedResetState
    ) -> SelfNormalizedResetResourceDeclaration:
        """Measure exact persistent bytes and report finite update bounds."""

        self._validate_state_static(state)
        if not bool(np.asarray(self._state_valid(state))):
            raise ValueError("cannot measure resources for an invalid SNR state")
        persistent_bytes = 0
        for leaf in jax.tree_util.tree_leaves(state):
            if jax.dtypes.issubdtype(leaf.dtype, jax.dtypes.prng_key):
                persistent_bytes += int(np.asarray(jr.key_data(leaf)).nbytes)
            else:
                persistent_bytes += int(np.asarray(leaf).nbytes)
        cfg = self._config
        random_draws = 0
        if (
            cfg.enabled
            and cfg.initialization_mode == "owned_lecun_uniform"
            and cast(int, cfg.max_resets_per_step) > 0
            and cfg.max_total_resets > 0
        ):
            random_draws = cfg.input_dim * cfg.unit_count
        # Source-level bound for the explicitly named fresh matrix,
        # diagnostics/masks, interval copy, and one candidate state. This is
        # not a measured device peak and excludes compiler/XLA workspaces.
        maximum_temporary_bytes = (
            persistent_bytes
            + 4 * cfg.input_dim * cfg.unit_count
            + 32 * cfg.unit_count
            + 8 * cfg.unit_count * cfg.window_size
        )
        return SelfNormalizedResetResourceDeclaration(
            unit_count=cfg.unit_count,
            window_size=cfg.window_size,
            interval_slots=cfg.unit_count * cfg.window_size,
            dense_parameter_count=cfg.parameter_count,
            persistent_state_bytes=persistent_bytes,
            maximum_resets_per_step=cast(int, cfg.max_resets_per_step),
            maximum_total_resets=cfg.max_total_resets,
            maximum_updates=cfg.maximum_updates,
            maximum_random_float32_draws_per_step=random_draws,
            maximum_temporary_bytes_per_step=maximum_temporary_bytes,
            temporary_bytes_scope=(
                "source-level-named-arrays; excludes-compiler-and-xla-workspaces; "
                "not-a-measured-device-peak"
            ),
            checkpoint_host_only=True,
            integrity_authenticated=False,
            external_side_effects=False,
        )

    def checkpoint_payload(self, state: SelfNormalizedResetState) -> dict[str, object]:
        """Return an exact host-only checkpoint with unkeyed integrity."""

        self._validate_state_static(state)
        if not bool(np.asarray(self._state_valid(state))):
            raise ValueError("cannot checkpoint an invalid SNR state")
        target_payload = {
            "parameters": {
                name: _encode_array(getattr(state.target.parameters, name))
                for name in sorted(_PARAMETER_FIELDS)
            },
            "optimizer": {
                name: _encode_array(getattr(state.target.optimizer, name))
                for name in sorted(_OPTIMIZER_FIELDS)
            },
        }
        state_payload: dict[str, object] = {
            "rng_key_data": _encode_array(jr.key_data(state.rng_key)),
            "target": target_payload,
        }
        for name in sorted(_STATE_FIELDS - {"rng_key_data", "target"}):
            state_payload[name] = _encode_array(getattr(state, name))
        body: dict[str, object] = {
            "schema": SELF_NORMALIZED_RESETS_SCHEMA,
            "type": "SelfNormalizedResetsCheckpoint",
            "integrity_notice": ("unkeyed-sha256-detects-corruption-not-authentication"),
            "implementation_source_sha256": self_normalized_resets_source_sha256(),
            "config": self.to_config(),
            "state": state_payload,
        }
        return {
            **body,
            "checkpoint_sha256": hashlib.sha256(_canonical_json_bytes(body)).hexdigest(),
        }

    @classmethod
    def from_checkpoint_payload(
        cls,
        payload: Mapping[str, object],
        *,
        expected_source_sha256: str,
        expected_representation_sha256: str,
    ) -> tuple[SelfNormalizedResets, SelfNormalizedResetState]:
        """Restore an exact source-bound checkpoint after integrity checks.

        The SHA-256 checksum is unkeyed. It detects accidental corruption and
        noncanonical payloads but is not a cryptographic authenticity proof.
        """

        expected_source = _validate_sha256("expected_source_sha256", expected_source_sha256)
        expected_representation = _validate_sha256(
            "expected_representation_sha256", expected_representation_sha256
        )
        if set(payload) != _CHECKPOINT_FIELDS:
            raise ValueError("SNR checkpoint fields do not match schema v1")
        if payload.get("schema") != SELF_NORMALIZED_RESETS_SCHEMA:
            raise ValueError("SNR checkpoint schema is invalid")
        if payload.get("type") != "SelfNormalizedResetsCheckpoint":
            raise ValueError("SNR checkpoint type is invalid")
        if payload.get("integrity_notice") != (
            "unkeyed-sha256-detects-corruption-not-authentication"
        ):
            raise ValueError("SNR checkpoint integrity notice is invalid")
        if payload.get("implementation_source_sha256") != self_normalized_resets_source_sha256():
            raise ValueError("SNR checkpoint implementation source hash is invalid")
        checkpoint_sha256 = payload.get("checkpoint_sha256")
        if type(checkpoint_sha256) is not str or len(checkpoint_sha256) != 64:
            raise ValueError("SNR checkpoint digest is invalid")
        body = {name: payload[name] for name in payload if name != "checkpoint_sha256"}
        if hashlib.sha256(_canonical_json_bytes(body)).hexdigest() != checkpoint_sha256:
            raise ValueError("SNR checkpoint integrity check failed")
        config_payload = payload.get("config")
        state_payload = payload.get("state")
        if not isinstance(config_payload, Mapping) or not isinstance(state_payload, Mapping):
            raise ValueError("SNR checkpoint config/state must be objects")
        controller = cls.from_config(cast(Mapping[str, object], config_payload))
        if controller.config.source_sha256 != expected_source:
            raise ValueError("SNR checkpoint source binding differs from expected")
        if controller.config.representation_sha256 != expected_representation:
            raise ValueError("SNR checkpoint representation binding differs from expected")
        if set(state_payload) != _STATE_FIELDS:
            raise ValueError("SNR checkpoint state fields do not match schema v1")
        target_payload = state_payload.get("target")
        if not isinstance(target_payload, Mapping) or set(target_payload) != _TARGET_FIELDS:
            raise ValueError("SNR checkpoint target fields are invalid")
        parameter_payload = target_payload.get("parameters")
        optimizer_payload = target_payload.get("optimizer")
        if (
            not isinstance(parameter_payload, Mapping)
            or set(parameter_payload) != _PARAMETER_FIELDS
        ):
            raise ValueError("SNR checkpoint parameter fields are invalid")
        if (
            not isinstance(optimizer_payload, Mapping)
            or set(optimizer_payload) != _OPTIMIZER_FIELDS
        ):
            raise ValueError("SNR checkpoint optimizer fields are invalid")
        cfg = controller.config
        parameters = DenseReLUParameters(
            incoming_weight=_decode_array(
                parameter_payload["incoming_weight"],
                name="incoming_weight",
                shape=(cfg.input_dim, cfg.unit_count),
                dtype=jnp.float32,
            ),
            bias=_decode_array(
                parameter_payload["bias"],
                name="bias",
                shape=(cfg.unit_count,),
                dtype=jnp.float32,
            ),
            outgoing_weight=_decode_array(
                parameter_payload["outgoing_weight"],
                name="outgoing_weight",
                shape=(cfg.unit_count, cfg.output_dim),
                dtype=jnp.float32,
            ),
        )
        if cfg.optimizer_kind == "adam":
            optimizer_shapes = {
                "incoming_first_moment": (cfg.input_dim, cfg.unit_count),
                "incoming_second_moment": (cfg.input_dim, cfg.unit_count),
                "bias_first_moment": (cfg.unit_count,),
                "bias_second_moment": (cfg.unit_count,),
                "outgoing_first_moment": (cfg.unit_count, cfg.output_dim),
                "outgoing_second_moment": (cfg.unit_count, cfg.output_dim),
            }
        else:
            optimizer_shapes = {name: (0,) for name in _OPTIMIZER_FIELDS if name != "count"}
        optimizer_arrays = {
            name: _decode_array(
                optimizer_payload[name],
                name=name,
                shape=shape,
                dtype=jnp.float32,
            )
            for name, shape in optimizer_shapes.items()
        }
        optimizer = DenseReLUOptimizerState(
            count=_decode_array(
                optimizer_payload["count"],
                name="optimizer.count",
                shape=(),
                dtype=jnp.int32,
            ),
            incoming_first_moment=optimizer_arrays["incoming_first_moment"],
            incoming_second_moment=optimizer_arrays["incoming_second_moment"],
            bias_first_moment=optimizer_arrays["bias_first_moment"],
            bias_second_moment=optimizer_arrays["bias_second_moment"],
            outgoing_first_moment=optimizer_arrays["outgoing_first_moment"],
            outgoing_second_moment=optimizer_arrays["outgoing_second_moment"],
        )

        def state_array(name: str, shape: tuple[int, ...], dtype: Any) -> Array:
            return _decode_array(
                state_payload[name], name=f"state.{name}", shape=shape, dtype=dtype
            )

        key_data = state_array("rng_key_data", (2,), jnp.uint32)
        state = SelfNormalizedResetState(
            rng_key=jr.wrap_key_data(key_data, impl="threefry2x32"),
            target=DenseReLUResetTarget(parameters=parameters, optimizer=optimizer),
            ages_words=state_array("ages_words", (cfg.unit_count, 2), jnp.uint32),
            epoch_observations_words=state_array(
                "epoch_observations_words", (cfg.unit_count, 2), jnp.uint32
            ),
            has_fired=state_array("has_fired", (cfg.unit_count,), jnp.bool_),
            intervals_words=state_array(
                "intervals_words",
                (cfg.unit_count, cfg.window_size, 2),
                jnp.uint32,
            ),
            interval_count=state_array("interval_count", (cfg.unit_count,), jnp.int32),
            interval_cursor=state_array("interval_cursor", (cfg.unit_count,), jnp.int32),
            unit_reset_count=state_array("unit_reset_count", (cfg.unit_count,), jnp.int32),
            total_reset_count=state_array("total_reset_count", (), jnp.int32),
            step_words=state_array("step_words", (2,), jnp.uint32),
            step_count=state_array("step_count", (), jnp.int32),
            source_binding_words=state_array("source_binding_words", (8,), jnp.uint32),
            representation_binding_words=state_array(
                "representation_binding_words", (8,), jnp.uint32
            ),
            integrity_tag=state_array("integrity_tag", (4,), jnp.uint32),
        )
        controller._validate_state_static(state)
        if not bool(np.asarray(controller._state_valid(state))):
            raise ValueError("SNR checkpoint state is invalid or corrupt")
        if controller.checkpoint_payload(state) != dict(payload):
            raise ValueError("SNR checkpoint is noncanonical")
        return controller, state

    def rebind_reset(
        self,
        state: SelfNormalizedResetState,
        *,
        rng_key: PRNGKeyArray,
        source_sha256: str,
        representation_sha256: str,
    ) -> tuple[SelfNormalizedResets, SelfNormalizedResetState]:
        """Bind a fresh detector lifetime while preserving every target bit."""

        self._validate_state_static(state)
        if not bool(np.asarray(self._state_valid(state))):
            raise ValueError("cannot rebind an invalid SNR state")
        if not _typed_key_valid(rng_key):
            raise TypeError("rng_key must be a typed scalar threefry2x32 JAX PRNG key")
        new_config = dataclasses.replace(
            self._config,
            source_sha256=_validate_sha256("source_sha256", source_sha256),
            representation_sha256=_validate_sha256("representation_sha256", representation_sha256),
        )
        rebound = SelfNormalizedResets(new_config)
        rebound_state = rebound.init(rng_key, state.target)
        return rebound, rebound_state


def _encode_array(value: Array) -> dict[str, object]:
    array = np.asarray(jax.device_get(value))
    if array.dtype not in (
        np.dtype(np.float32),
        np.dtype(np.int32),
        np.dtype(np.uint32),
        np.dtype(np.bool_),
    ):
        raise TypeError(f"checkpoint array dtype {array.dtype} is unsupported")
    return {
        "dtype": array.dtype.name,
        "shape": list(array.shape),
        "data_hex": array.tobytes(order="C").hex(),
    }


def _decode_array(
    payload: object,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: Any,
) -> Array:
    if not isinstance(payload, Mapping) or set(payload) != {
        "dtype",
        "shape",
        "data_hex",
    }:
        raise ValueError(f"checkpoint {name} must be a canonical array object")
    expected_dtype = np.dtype(dtype)
    if payload.get("dtype") != expected_dtype.name or payload.get("shape") != list(shape):
        raise ValueError(f"checkpoint {name} dtype/shape is invalid")
    data_hex = payload.get("data_hex")
    if type(data_hex) is not str or len(data_hex) != math.prod(shape) * expected_dtype.itemsize * 2:
        raise ValueError(f"checkpoint {name} data_hex length is invalid")
    try:
        raw = bytes.fromhex(data_hex)
    except ValueError as error:
        raise ValueError(f"checkpoint {name} data_hex is invalid") from error
    array = np.frombuffer(raw, dtype=expected_dtype).copy().reshape(shape)
    return jnp.asarray(array, dtype=jnp.dtype(dtype))


__all__ = [
    "SELF_NORMALIZED_RESETS_SCHEMA",
    "DenseReLUFreshParameters",
    "DenseReLUOptimizerState",
    "DenseReLUParameters",
    "DenseReLUResetTarget",
    "SelfNormalizedResetConfig",
    "SelfNormalizedResetResourceDeclaration",
    "SelfNormalizedResetResult",
    "SelfNormalizedResetScanResult",
    "SelfNormalizedResetState",
    "SelfNormalizedResets",
    "empty_dense_relu_optimizer_state",
    "self_normalized_resets_source_sha256",
    "zero_adam_state",
]
