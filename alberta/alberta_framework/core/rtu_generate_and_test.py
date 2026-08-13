# mypy: disable-error-code="call-arg"
"""Causal fixed-capacity generate-and-test for RTU state-builder units.

The lifecycle in this module is deliberately narrower than a controller.  It
observes a downstream loss gradient *before* that gradient changes the RTU,
maintains one contribution-based utility estimate per complex recurrent unit,
and, at fixed periodic boundaries, replaces a bounded number of mature units.
It never consumes labels, task identifiers, future observations, safety
decisions, or scientific-evidence verdicts.

For unit ``i`` the instantaneous observation is

``abs(real_i * dL/dreal_i) + abs(imag_i * dL/dimag_i)``.

This is downstream-loss sensitivity/effective contribution.  It is not the
paper's delight signal and is not described with the delight catchphrase.
The sensitivity EMA remains useful telemetry, but a live replacement owner may
also require frozen-consumer causal-deletion evidence.  In that mode the
positive bounded loss increase from deleting both channels of one complex unit
has its own EMA and evidence count, and only that causal utility ranks mature
replacement candidates.  Missing evidence defers replacement without deferring
observation, recurrence, or ordinary learning.

A proposal may optionally embed a content-bound bootstrap event plus reset /
restart event and one ordinary
:class:`~alberta_framework.core.state_builder.StateBuilderLearningProposal`.
The RTU builder recomputes the complete recurrence from the exact pre-update
state, applies the source-bound ordinary update to that exact destination, and
only then forms replacement.  Commit accepts only that bit-identical live
state.  Thus callers cannot smuggle in an unverified "newer state".

Replacement is whole-unit: polar parameters and both dense input rows are
redrawn; real/imaginary activation, compressed sensitivities, optional Taylor
trace, and selected Taylor source/delta slices are scrubbed.  The RTU update
clock advances once per nonempty replacement transaction.  No per-unit
optimizer state exists in the RTU builder; its global last-gradient norm is
retained because it describes the preceding ordinary update, not one unit.

This is an L0 mechanism with status ``not_assessed``.  It grants no control,
safety, promotion, or efficacy authority.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import Array
from jaxtyping import Bool, Float, Int, UInt

from alberta_framework.core.checkpoints import (
    load_checkpoint,
    load_checkpoint_metadata,
    save_checkpoint,
)
from alberta_framework.core.recurrent_trace_actor_critic import (
    RTUSensitivities,
    RTUState,
)
from alberta_framework.core.state_builder import (
    RecurrentTraceUnitStateBuilder,
    RecurrentTraceUnitStateBuilderConfig,
    RecurrentTraceUnitStateBuilderState,
    StateBuilderLearningDiagnostics,
    StateBuilderLearningProposal,
)

RTU_GENERATE_AND_TEST_CONFIG_SCHEMA = "alberta.rtu-generate-and-test.config.v2"
RTU_GENERATE_AND_TEST_STATE_SCHEMA = "alberta.rtu-generate-and-test.state.v3"
RTU_GENERATE_AND_TEST_CHECKPOINT_SCHEMA = (
    "alberta.rtu-generate-and-test.checkpoint.v1"
)
RTU_GENERATE_AND_TEST_MECHANISM_STATUS = "not_assessed"
RTU_GENERATE_AND_TEST_EVIDENCE_LEVEL = "L0"
RTU_GENERATE_AND_TEST_SCIENTIFIC_PROMOTION_ALLOWED = False
RTU_GENERATE_AND_TEST_ADVANCE_RECEIPT_SCHEMA = (
    "alberta.rtu-generate-and-test.advance-receipt.v1"
)

_INT32_MAX = 2**31 - 1
_UINT32_MAX = 2**32 - 1
_UINT64_MAX = 2**64 - 1
_FLOAT32_MAX = 3.4028234663852886e38
_FLOAT32_EPSILON = 1.1920928955078125e-07
_FLOAT32_TINY = 1.1754943508222875e-38
_FINGERPRINT_WORDS = 8
_ADVANCE_TAG_WORDS = 8


def _uint32_bits(value: Array) -> UInt[Array, " words"]:
    """Return exact uint32 words for one supported RTU-state/event array."""

    array = jnp.asarray(value)
    if array.dtype == jnp.dtype(jnp.uint32):
        return array.reshape((-1,))
    if array.dtype in {jnp.dtype(jnp.float32), jnp.dtype(jnp.int32)}:
        return jax.lax.bitcast_convert_type(array, jnp.uint32).reshape((-1,))
    if array.dtype == jnp.dtype(jnp.bool_):
        return array.astype(jnp.uint32).reshape((-1,))
    raise TypeError(f"content tag does not support dtype {array.dtype}")


def _mix_content_words(words: Array) -> UInt[Array, " 8"]:
    """Deterministically tag exact runtime content without host callbacks."""

    lanes = jnp.asarray(
        (
            0x243F6A88,
            0x85A308D3,
            0x13198A2E,
            0x03707344,
            0xA4093822,
            0x299F31D0,
            0x082EFA98,
            0xEC4E6C89,
        ),
        dtype=jnp.uint32,
    )
    multipliers = jnp.asarray(
        (
            0x9E3779B1,
            0x85EBCA77,
            0xC2B2AE3D,
            0x27D4EB2F,
            0x165667B1,
            0xD3A2646D,
            0xFD7046C5,
            0xB55A4F09,
        ),
        dtype=jnp.uint32,
    )
    shifts = jnp.arange(5, 13, dtype=jnp.uint32)

    def mix(tag: Array, indexed_word: tuple[Array, Array]) -> tuple[Array, None]:
        index, word = indexed_word
        salted = word + (index + jnp.asarray(1, dtype=jnp.uint32)) * multipliers
        candidate = (tag ^ salted) * multipliers
        rotated = (candidate << shifts) | (candidate >> (32 - shifts))
        return candidate ^ rotated ^ lanes, None

    indices = jnp.arange(words.shape[0], dtype=jnp.uint32)
    mixed, _ = jax.lax.scan(mix, lanes, (indices, words))
    return mixed


def _rtu_builder_content_tag(
    state: RecurrentTraceUnitStateBuilderState,
) -> UInt[Array, " 8"]:
    """Bind every persistent RTU builder leaf, including Taylor ownership."""

    arrays: list[Array] = [
        state.parameters,
        state.rtu_state.real,
        state.rtu_state.imaginary,
        *jax.tree.leaves(state.sensitivities),
    ]
    if state.taylor_trace is not None:
        arrays.extend(jax.tree.leaves(state.taylor_trace))
    if state.sensitivity_source_parameters is not None:
        arrays.append(state.sensitivity_source_parameters)
    if state.sensitivity_parameter_delta is not None:
        arrays.append(state.sensitivity_parameter_delta)
    if state.sensitivity_source_update_words is not None:
        arrays.append(state.sensitivity_source_update_words)
    arrays.extend(
        (
            state.step_count,
            state.step_words,
            state.update_count,
            state.update_words,
            state.last_gradient_norm,
        )
    )
    return _mix_content_words(jnp.concatenate(tuple(_uint32_bits(value) for value in arrays)))


def _strict_int(
    value: object,
    *,
    name: str,
    minimum: int,
    maximum: int,
) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an exact integer in [{minimum}, {maximum}]")
    return value


def _strict_float(
    value: object,
    *,
    name: str,
    minimum: float,
    maximum: float,
    maximum_inclusive: bool,
) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{name} must be an exact finite float")
    upper = value <= maximum if maximum_inclusive else value < maximum
    if value < minimum or not upper:
        right = "]" if maximum_inclusive else ")"
        raise ValueError(f"{name} must lie in [{minimum}, {maximum}{right}")
    converted = float(jnp.asarray(value, dtype=jnp.float32))
    if not math.isfinite(converted) or (value != 0.0 and converted == 0.0):
        raise ValueError(f"{name} must remain finite and nonzero as float32")
    return value


def _require_array(
    value: Any,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: jnp.dtype[Any],
) -> Array:
    if getattr(value, "shape", None) != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if getattr(value, "dtype", None) != dtype:
        raise TypeError(f"{name} must have dtype {dtype}")
    return jnp.asarray(value)


def _require_threefry_key(value: Any, *, name: str) -> None:
    try:
        data = jr.key_data(value)
        implementation = str(jr.key_impl(value))
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be one typed Threefry JAX key") from error
    if (
        getattr(value, "shape", None) != ()
        or data.shape != (2,)
        or data.dtype != jnp.dtype(jnp.uint32)
        or implementation != "threefry2x32"
    ):
        raise TypeError(f"{name} must be one typed Threefry JAX key")


def _fingerprint(config: dict[str, Any]) -> UInt[Array, " 8"]:
    canonical = json.dumps(
        config,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    digest = hashlib.sha256(canonical).digest()
    return jnp.asarray(
        [int.from_bytes(digest[index : index + 4], "big") for index in range(0, 32, 4)],
        dtype=jnp.uint32,
    )


def _increment_words(words: Array) -> tuple[UInt[Array, " 2"], Bool[Array, ""]]:
    maximum = jnp.asarray(_UINT32_MAX, dtype=jnp.uint32)
    carry = words[1] == maximum
    capacity = ~(carry & (words[0] == maximum))
    candidate = jnp.stack(
        (
            words[0] + carry.astype(jnp.uint32),
            words[1] + jnp.asarray(1, dtype=jnp.uint32),
        )
    ).astype(jnp.uint32)
    return jnp.where(capacity, candidate, words), capacity


def _saturating_count_increment(value: Array) -> Int[Array, ""]:
    maximum = jnp.asarray(_INT32_MAX, dtype=jnp.int32)
    return jnp.minimum(jnp.maximum(value, 0), maximum - 1) + 1


def _counter_mirror_valid(words: Array, count: Array) -> Bool[Array, ""]:
    maximum_u32 = jnp.asarray(_INT32_MAX, dtype=jnp.uint32)
    expected = jnp.where(
        (words[0] != 0) | (words[1] >= maximum_u32),
        jnp.asarray(_INT32_MAX, dtype=jnp.int32),
        jnp.minimum(words[1], maximum_u32).astype(jnp.int32),
    )
    return (count >= 0) & (count == expected)


def _add_to_counter(
    words: Array,
    count: Array,
    amount: Array,
    *,
    maximum_amount: int,
) -> tuple[UInt[Array, " 2"], Int[Array, ""], Bool[Array, ""]]:
    def body(
        index: int,
        carry: tuple[Array, Array, Array],
    ) -> tuple[Array, Array, Array]:
        current_words, current_count, capacity = carry
        active = jnp.asarray(index, dtype=jnp.int32) < amount
        proposed_words, available = _increment_words(current_words)
        next_words = jnp.where(active & available, proposed_words, current_words)
        next_count = jnp.where(
            active & available,
            _saturating_count_increment(current_count),
            current_count,
        )
        return next_words, next_count, capacity & (~active | available)

    result_words, result_count, capacity = jax.lax.fori_loop(
        0,
        maximum_amount,
        body,
        (words, count, jnp.asarray(True, dtype=jnp.bool_)),
    )
    return result_words, result_count, capacity


def _words_mod(words: Array, divisor: int) -> UInt[Array, ""]:
    """Compute one uint64-word-pair remainder without enabling JAX x64."""

    modulus = jnp.asarray(divisor, dtype=jnp.uint32)

    def bit_step(remainder: Array, bit_index: Array) -> tuple[Array, None]:
        use_high = bit_index >= jnp.asarray(32, dtype=jnp.int32)
        shift = jnp.where(use_high, bit_index - 32, bit_index)
        word = jnp.where(use_high, words[0], words[1])
        bit = (word >> shift.astype(jnp.uint32)) & jnp.asarray(1, dtype=jnp.uint32)
        doubled = remainder + remainder
        reduced = jnp.where(doubled >= modulus, doubled - modulus, doubled)
        with_bit = reduced + bit
        return jnp.where(with_bit >= modulus, with_bit - modulus, with_bit), None

    indices = jnp.arange(63, -1, -1, dtype=jnp.int32)
    remainder, _ = jax.lax.scan(
        bit_step,
        jnp.asarray(0, dtype=jnp.uint32),
        indices,
    )
    return remainder


def _words_at_least(words: Array, threshold: int) -> Bool[Array, ""]:
    high = jnp.asarray(threshold >> 32, dtype=jnp.uint32)
    low = jnp.asarray(threshold & _UINT32_MAX, dtype=jnp.uint32)
    return (words[0] > high) | ((words[0] == high) & (words[1] >= low))


def _words_not_earlier(left: Array, right: Array) -> Bool[Array, ""]:
    """Return whether one exact uint64 word pair is at least another."""

    return (left[0] > right[0]) | (
        (left[0] == right[0]) & (left[1] >= right[1])
    )


def _saturating_words_product(
    words: Array,
    multiplier: int,
) -> UInt[Array, " 2"]:
    """Multiply a uint64 word pair by one static uint32 and saturate."""

    maximum = jnp.full((2,), _UINT32_MAX, dtype=jnp.uint32)
    factor = jnp.asarray(multiplier, dtype=jnp.uint32)

    def add_saturating(left: Array, right: Array) -> tuple[Array, Array]:
        low = left[1] + right[1]
        carry = low < left[1]
        partial_high = left[0] + right[0]
        high_overflow = partial_high < left[0]
        high = partial_high + carry.astype(jnp.uint32)
        carry_overflow = carry & (high == 0)
        overflow = high_overflow | carry_overflow
        value = jnp.stack((high, low)).astype(jnp.uint32)
        return jnp.where(overflow, maximum, value), overflow

    def bit_step(index: int, carry: tuple[Array, Array, Array]) -> tuple[Array, ...]:
        total, addend, saturated = carry
        selected = (
            (factor >> jnp.asarray(index, dtype=jnp.uint32))
            & jnp.asarray(1, dtype=jnp.uint32)
        ) != 0
        candidate, addition_overflow = add_saturating(total, addend)
        next_saturated = saturated | (selected & addition_overflow)
        next_total = jnp.where(
            next_saturated,
            maximum,
            jnp.where(selected, candidate, total),
        )
        doubled, _ = add_saturating(addend, addend)
        return next_total, doubled, next_saturated

    product, _, _ = jax.lax.fori_loop(
        0,
        32,
        bit_step,
        (
            jnp.zeros((2,), dtype=jnp.uint32),
            words,
            jnp.asarray(False, dtype=jnp.bool_),
        ),
    )
    return cast(UInt[Array, " 2"], product)


def _float_bits_equal(left: Array, right: Array) -> Bool[Array, ""]:
    return jnp.array_equal(
        jax.lax.bitcast_convert_type(left, jnp.uint32),
        jax.lax.bitcast_convert_type(right, jnp.uint32),
    )


def _leaf_exact_equal(left: Any, right: Any) -> Bool[Array, ""]:
    left_dtype = getattr(left, "dtype", None)
    right_dtype = getattr(right, "dtype", None)
    if left_dtype != right_dtype or getattr(left, "shape", None) != getattr(
        right, "shape", None
    ):
        return jnp.asarray(False, dtype=jnp.bool_)
    if left_dtype is not None and jnp.issubdtype(left_dtype, jax.dtypes.prng_key):
        return jnp.array_equal(jr.key_data(left), jr.key_data(right))
    if left_dtype == jnp.dtype(jnp.float32):
        return _float_bits_equal(jnp.asarray(left), jnp.asarray(right))
    return jnp.array_equal(left, right)


def _tree_exact_equal(left: Any, right: Any) -> Bool[Array, ""]:
    left_structure = jax.tree.structure(left)
    if not cast(bool, cast(Any, left_structure) == jax.tree.structure(right)):
        return jnp.asarray(False, dtype=jnp.bool_)
    comparisons = [
        _leaf_exact_equal(a, b)
        for a, b in zip(
            jax.tree.leaves(left),
            jax.tree.leaves(right),
            strict=True,
        )
    ]
    if not comparisons:
        return jnp.asarray(True, dtype=jnp.bool_)
    return jnp.all(jnp.stack(comparisons))


def _tree_nbytes(tree: Any) -> int:
    total = 0
    for leaf in jax.tree.leaves(tree):
        if jnp.issubdtype(leaf.dtype, jax.dtypes.prng_key):
            leaf = jr.key_data(leaf)
        total += math.prod(leaf.shape) * leaf.dtype.itemsize
    return int(total)


def _exact_json_equal(left: Any, right: Any) -> bool:
    return json.dumps(left, sort_keys=True, separators=(",", ":")) == json.dumps(
        right,
        sort_keys=True,
        separators=(",", ":"),
    )


@dataclass(frozen=True)
class RTUGenerateAndTestConfig:
    """Static utility, evidence-floor, protection, and replacement contract."""

    builder: RecurrentTraceUnitStateBuilderConfig
    utility_decay: float = 0.99
    replacement_interval: int = 100
    replacement_quota: int = 1
    warmup_observations: int = 0
    minimum_age: int = 100
    minimum_support: int = 1
    minimum_causal_evidence: int = 1
    minimum_sensitivity_for_support: float = 0.0
    protected_units: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if type(self.builder) is not RecurrentTraceUnitStateBuilderConfig:
            raise TypeError("builder must be an exact RecurrentTraceUnitStateBuilderConfig")
        _strict_float(
            self.utility_decay,
            name="utility_decay",
            minimum=0.0,
            maximum=1.0,
            maximum_inclusive=False,
        )
        _strict_int(
            self.replacement_interval,
            name="replacement_interval",
            minimum=1,
            maximum=_INT32_MAX,
        )
        _strict_int(
            self.replacement_quota,
            name="replacement_quota",
            minimum=1,
            maximum=self.builder.hidden_dim,
        )
        _strict_int(
            self.warmup_observations,
            name="warmup_observations",
            minimum=0,
            maximum=_UINT64_MAX,
        )
        for name in ("minimum_age", "minimum_support"):
            _strict_int(
                getattr(self, name),
                name=name,
                minimum=0,
                maximum=_UINT32_MAX,
            )
        _strict_int(
            self.minimum_causal_evidence,
            name="minimum_causal_evidence",
            minimum=1,
            maximum=_UINT32_MAX,
        )
        _strict_float(
            self.minimum_sensitivity_for_support,
            name="minimum_sensitivity_for_support",
            minimum=0.0,
            maximum=_FLOAT32_MAX,
            maximum_inclusive=True,
        )
        if type(self.protected_units) is not tuple:
            raise TypeError("protected_units must be an exact tuple")
        if tuple(sorted(set(self.protected_units))) != self.protected_units:
            raise ValueError("protected_units must be sorted and unique")
        for index in self.protected_units:
            _strict_int(
                index,
                name="protected unit index",
                minimum=0,
                maximum=self.builder.hidden_dim - 1,
            )

    def to_config(self) -> dict[str, Any]:
        """Return the exact JSON-compatible lifecycle configuration."""

        return {
            "schema": RTU_GENERATE_AND_TEST_CONFIG_SCHEMA,
            "builder": self.builder.to_config(),
            "utility_decay": self.utility_decay,
            "replacement_interval": self.replacement_interval,
            "replacement_quota": self.replacement_quota,
            "warmup_observations": self.warmup_observations,
            "minimum_age": self.minimum_age,
            "minimum_support": self.minimum_support,
            "minimum_causal_evidence": self.minimum_causal_evidence,
            "minimum_sensitivity_for_support": self.minimum_sensitivity_for_support,
            "protected_units": list(self.protected_units),
        }

    @classmethod
    def from_config(cls, payload: dict[str, Any]) -> RTUGenerateAndTestConfig:
        """Strictly reconstruct current-schema configuration."""

        if type(payload) is not dict:
            raise TypeError("RTU generate-and-test config must be an exact dict")
        expected = {
            "schema",
            "builder",
            "utility_decay",
            "replacement_interval",
            "replacement_quota",
            "warmup_observations",
            "minimum_age",
            "minimum_support",
            "minimum_causal_evidence",
            "minimum_sensitivity_for_support",
            "protected_units",
        }
        if set(payload) != expected:
            raise ValueError("RTU generate-and-test config fields are not exact")
        if payload["schema"] != RTU_GENERATE_AND_TEST_CONFIG_SCHEMA:
            raise ValueError("RTU generate-and-test config schema is unsupported")
        raw_builder = payload["builder"]
        if type(raw_builder) is not dict:
            raise TypeError("builder config must be an exact dict")
        raw_protected = payload["protected_units"]
        if type(raw_protected) is not list:
            raise TypeError("protected_units config field must be an exact list")
        return cls(
            builder=RecurrentTraceUnitStateBuilderConfig.from_config(raw_builder),
            utility_decay=payload["utility_decay"],
            replacement_interval=payload["replacement_interval"],
            replacement_quota=payload["replacement_quota"],
            warmup_observations=payload["warmup_observations"],
            minimum_age=payload["minimum_age"],
            minimum_support=payload["minimum_support"],
            minimum_causal_evidence=payload["minimum_causal_evidence"],
            minimum_sensitivity_for_support=payload[
                "minimum_sensitivity_for_support"
            ],
            protected_units=tuple(raw_protected),
        )


@chex.dataclass(frozen=True)
class RTUGenerateAndTestState:
    """Fixed-capacity causal utility, age, support, RNG, and exact clocks.

    ``replacement_words`` counts individual units, while
    ``replacement_event_words`` counts nonempty atomic replacement events.
    The latter is the revision delta coupled to the live builder update clock.
    """

    lifecycle_fingerprint: UInt[Array, " 8"]
    utility: Float[Array, " hidden_dim"]
    causal_utility: Float[Array, " hidden_dim"]
    age: UInt[Array, " hidden_dim"]
    support: UInt[Array, " hidden_dim"]
    causal_evidence_count: UInt[Array, " hidden_dim"]
    last_effective_contribution: Float[Array, " hidden_dim"]
    last_causal_deletion_loss_change: Float[Array, " hidden_dim"]
    last_replaced_mask: Bool[Array, " hidden_dim"]
    rng_key: Array
    observation_count: Int[Array, ""]
    observation_words: UInt[Array, " 2"]
    replacement_count: Int[Array, ""]
    replacement_words: UInt[Array, " 2"]
    replacement_event_count: Int[Array, ""]
    replacement_event_words: UInt[Array, " 2"]


@chex.dataclass(frozen=True)
class RTUGenerateAndTestCompositionState:
    """Checkpoint/scan carry pairing lifecycle state with its live RTU state."""

    lifecycle: RTUGenerateAndTestState
    builder: RecurrentTraceUnitStateBuilderState


@chex.dataclass(frozen=True)
class RTUGenerateAndTestAdvanceReceipt:
    """Content-bound bootstrap plus optional reset/restart event sequence.

    The tag detects mutation after construction.  Authority for the real
    transition remains with the caller; the Prototype adapter constructs this
    receipt only from its already-authenticated transition.
    """

    builder_fingerprint: UInt[Array, " 8"]
    source_builder_content_tag_words: UInt[Array, " 8"]
    source_step_words: UInt[Array, " 2"]
    source_update_words: UInt[Array, " 2"]
    bootstrap_observation: Float[Array, " observation_dim"]
    previous_action: Int[Array, ""]
    previous_reward: Float[Array, ""]
    previous_discount: Float[Array, ""]
    episode_boundary: Bool[Array, ""]
    restart_observation: Float[Array, " observation_dim"]
    sequence_length: Int[Array, ""]
    content_tag_words: UInt[Array, " 8"]


@chex.dataclass(frozen=True)
class RTUGenerateAndTestProposal:
    """Pure source-bound observation and whole-unit replacement proposal."""

    lifecycle_fingerprint: UInt[Array, " 8"]
    source_state: RTUGenerateAndTestState
    pre_update_builder_state: RecurrentTraceUnitStateBuilderState
    live_builder_state: RecurrentTraceUnitStateBuilderState
    learning_proposal: StateBuilderLearningProposal | None
    ordinary_learning_diagnostics: StateBuilderLearningDiagnostics | None
    advance_receipt: RTUGenerateAndTestAdvanceReceipt | None
    downstream_loss_gradient: Float[Array, " feature_dim"]
    replacement_allowed: Bool[Array, ""]
    causal_deletion_loss_change: Float[Array, " hidden_dim"]
    causal_deletion_evidence_declared: Bool[Array, ""]
    causal_deletion_evidence_available: Bool[Array, ""]
    causal_deletion_evidence_valid: Bool[Array, ""]
    causal_evidence_required: Bool[Array, ""]
    effective_contribution: Float[Array, " hidden_dim"]
    observed_utility: Float[Array, " hidden_dim"]
    observed_causal_utility: Float[Array, " hidden_dim"]
    observed_causal_evidence_count: UInt[Array, " hidden_dim"]
    selection_utility: Float[Array, " hidden_dim"]
    observed_age: UInt[Array, " hidden_dim"]
    observed_support: UInt[Array, " hidden_dim"]
    selected_indices: Int[Array, " replacement_quota"]
    selected_slots: Bool[Array, " replacement_quota"]
    selected_mask: Bool[Array, " hidden_dim"]
    fresh_parameter_slices: Float[Array, "replacement_quota parameter_slice"]
    candidate_state: RTUGenerateAndTestState
    candidate_builder_state: RecurrentTraceUnitStateBuilderState
    ordinary_advance_valid: Bool[Array, ""]
    advance_receipt_valid: Bool[Array, ""]
    source_state_valid: Bool[Array, ""]
    pre_update_builder_valid: Bool[Array, ""]
    input_valid: Bool[Array, ""]
    observation_capacity_available: Bool[Array, ""]
    per_unit_capacity_available: Bool[Array, ""]
    replacement_capacity_available: Bool[Array, ""]
    builder_capacity_available: Bool[Array, ""]
    candidate_state_valid: Bool[Array, ""]
    candidate_builder_valid: Bool[Array, ""]
    valid: Bool[Array, ""]
    rejected: Bool[Array, ""]


@chex.dataclass(frozen=True)
class RTUGenerateAndTestDiagnostics:
    """Fail-closed transaction diagnostics; never an evidence verdict."""

    effective_contribution: Float[Array, " hidden_dim"]
    observed_utility: Float[Array, " hidden_dim"]
    causal_deletion_loss_change: Float[Array, " hidden_dim"]
    observed_causal_utility: Float[Array, " hidden_dim"]
    observed_causal_evidence_count: UInt[Array, " hidden_dim"]
    selection_utility: Float[Array, " hidden_dim"]
    causal_deletion_evidence_available: Bool[Array, ""]
    causal_deletion_evidence_valid: Bool[Array, ""]
    causal_evidence_required: Bool[Array, ""]
    selected_indices: Int[Array, " replacement_quota"]
    selected_slots: Bool[Array, " replacement_quota"]
    selected_mask: Bool[Array, " hidden_dim"]
    selected_count: Int[Array, ""]
    proposal_integrity: Bool[Array, ""]
    lifecycle_source_matches: Bool[Array, ""]
    live_builder_matches: Bool[Array, ""]
    proposal_valid: Bool[Array, ""]
    applied: Bool[Array, ""]
    rejected: Bool[Array, ""]
    pre_observation_words: UInt[Array, " 2"]
    post_observation_words: UInt[Array, " 2"]
    pre_replacement_words: UInt[Array, " 2"]
    post_replacement_words: UInt[Array, " 2"]
    pre_replacement_event_words: UInt[Array, " 2"]
    post_replacement_event_words: UInt[Array, " 2"]
    pre_builder_update_words: UInt[Array, " 2"]
    post_builder_update_words: UInt[Array, " 2"]
    pre_rng_key_data: UInt[Array, " 2"]
    post_rng_key_data: UInt[Array, " 2"]


@chex.dataclass(frozen=True)
class RTUGenerateAndTestCommitResult:
    """Atomic lifecycle/builder commit result convenient as a scan output."""

    state: RTUGenerateAndTestState
    builder_state: RecurrentTraceUnitStateBuilderState
    composition: RTUGenerateAndTestCompositionState
    diagnostics: RTUGenerateAndTestDiagnostics


@dataclass(frozen=True)
class RTUGenerateAndTestResourceBudget:
    """Exact persistent and maximum transient storage accounting."""

    hidden_units: int
    replacement_quota: int
    parameter_slice_scalars: int
    lifecycle_state_nbytes: int
    builder_state_nbytes: int
    composition_state_nbytes: int
    maximum_proposal_nbytes: int
    random_replacement_roots_per_observation: int
    random_subkeys_per_replacement_root: int

    def to_config(self) -> dict[str, int]:
        """Return a JSON-compatible exact resource disclosure."""

        return {
            "hidden_units": self.hidden_units,
            "replacement_quota": self.replacement_quota,
            "parameter_slice_scalars": self.parameter_slice_scalars,
            "lifecycle_state_nbytes": self.lifecycle_state_nbytes,
            "builder_state_nbytes": self.builder_state_nbytes,
            "composition_state_nbytes": self.composition_state_nbytes,
            "maximum_proposal_nbytes": self.maximum_proposal_nbytes,
            "random_replacement_roots_per_observation": (
                self.random_replacement_roots_per_observation
            ),
            "random_subkeys_per_replacement_root": (
                self.random_subkeys_per_replacement_root
            ),
        }


class RTUGenerateAndTest:
    """Causal utility tracker and bounded whole-RTU-unit recycler."""

    evidence_level = RTU_GENERATE_AND_TEST_EVIDENCE_LEVEL
    mechanism_status = RTU_GENERATE_AND_TEST_MECHANISM_STATUS
    scientific_promotion_allowed = RTU_GENERATE_AND_TEST_SCIENTIFIC_PROMOTION_ALLOWED

    def __init__(self, config: RTUGenerateAndTestConfig):
        if type(config) is not RTUGenerateAndTestConfig:
            raise TypeError("config must be an exact RTUGenerateAndTestConfig")
        self._config = config
        self._builder = RecurrentTraceUnitStateBuilder(config.builder)
        self._fingerprint = _fingerprint(config.to_config())
        self._advance_fingerprint = _fingerprint(
            {
                "schema": RTU_GENERATE_AND_TEST_ADVANCE_RECEIPT_SCHEMA,
                "builder": config.builder.to_config(),
            }
        )
        protected = jnp.zeros((config.builder.hidden_dim,), dtype=jnp.bool_)
        if config.protected_units:
            protected = protected.at[jnp.asarray(config.protected_units)].set(True)
        self._protected_mask = protected

    @property
    def config(self) -> RTUGenerateAndTestConfig:
        """Return the immutable mechanism configuration."""

        return self._config

    @property
    def builder(self) -> RecurrentTraceUnitStateBuilder:
        """Return the exact RTU builder whose state contract is managed."""

        return self._builder

    def to_config(self) -> dict[str, Any]:
        """Serialize the exact lifecycle configuration."""

        return self.config.to_config()

    @classmethod
    def from_config(cls, payload: dict[str, Any]) -> RTUGenerateAndTest:
        """Construct from a strict current-schema config manifest."""

        return cls(RTUGenerateAndTestConfig.from_config(payload))

    def init(self, key: Array) -> RTUGenerateAndTestState:
        """Initialize zero utility/evidence state under one owned Threefry key."""

        _require_threefry_key(key, name="key")
        hidden = self.config.builder.hidden_dim
        return RTUGenerateAndTestState(
            lifecycle_fingerprint=self._fingerprint,
            utility=jnp.zeros((hidden,), dtype=jnp.float32),
            causal_utility=jnp.zeros((hidden,), dtype=jnp.float32),
            age=jnp.zeros((hidden,), dtype=jnp.uint32),
            support=jnp.zeros((hidden,), dtype=jnp.uint32),
            causal_evidence_count=jnp.zeros((hidden,), dtype=jnp.uint32),
            last_effective_contribution=jnp.zeros((hidden,), dtype=jnp.float32),
            last_causal_deletion_loss_change=jnp.zeros(
                (hidden,), dtype=jnp.float32
            ),
            last_replaced_mask=jnp.zeros((hidden,), dtype=jnp.bool_),
            rng_key=key,
            observation_count=jnp.asarray(0, dtype=jnp.int32),
            observation_words=jnp.zeros((2,), dtype=jnp.uint32),
            replacement_count=jnp.asarray(0, dtype=jnp.int32),
            replacement_words=jnp.zeros((2,), dtype=jnp.uint32),
            replacement_event_count=jnp.asarray(0, dtype=jnp.int32),
            replacement_event_words=jnp.zeros((2,), dtype=jnp.uint32),
        )

    def init_composition(
        self,
        lifecycle_key: Array,
        builder_key: Array,
    ) -> RTUGenerateAndTestCompositionState:
        """Initialize independently owned lifecycle and builder RNG domains."""

        _require_threefry_key(lifecycle_key, name="lifecycle_key")
        _require_threefry_key(builder_key, name="builder_key")
        return RTUGenerateAndTestCompositionState(
            lifecycle=self.init(lifecycle_key),
            builder=self.builder.init(builder_key),
        )

    def _check_state_contract(self, state: RTUGenerateAndTestState) -> None:
        if type(state) is not RTUGenerateAndTestState:
            raise TypeError("state must be an exact RTUGenerateAndTestState")
        hidden = self.config.builder.hidden_dim
        _require_array(
            state.lifecycle_fingerprint,
            name="state.lifecycle_fingerprint",
            shape=(_FINGERPRINT_WORDS,),
            dtype=jnp.dtype(jnp.uint32),
        )
        for name in (
            "utility",
            "causal_utility",
            "last_effective_contribution",
            "last_causal_deletion_loss_change",
        ):
            _require_array(
                getattr(state, name),
                name=f"state.{name}",
                shape=(hidden,),
                dtype=jnp.dtype(jnp.float32),
            )
        for name in ("age", "support", "causal_evidence_count"):
            _require_array(
                getattr(state, name),
                name=f"state.{name}",
                shape=(hidden,),
                dtype=jnp.dtype(jnp.uint32),
            )
        _require_array(
            state.last_replaced_mask,
            name="state.last_replaced_mask",
            shape=(hidden,),
            dtype=jnp.dtype(jnp.bool_),
        )
        _require_threefry_key(state.rng_key, name="state.rng_key")
        for name in (
            "observation_count",
            "replacement_count",
            "replacement_event_count",
        ):
            _require_array(
                getattr(state, name),
                name=f"state.{name}",
                shape=(),
                dtype=jnp.dtype(jnp.int32),
            )
        for name in (
            "observation_words",
            "replacement_words",
            "replacement_event_words",
        ):
            _require_array(
                getattr(state, name),
                name=f"state.{name}",
                shape=(2,),
                dtype=jnp.dtype(jnp.uint32),
            )

    def _state_is_valid(self, state: RTUGenerateAndTestState) -> Bool[Array, ""]:
        zero_evidence = state.causal_evidence_count == 0
        observation_started = jnp.any(state.observation_words != 0)
        last_replaced_count = jnp.sum(
            state.last_replaced_mask.astype(jnp.uint32),
            dtype=jnp.uint32,
        )
        replacement_words_cover_last_mask = (
            (state.replacement_words[0] != 0)
            | (state.replacement_words[1] >= last_replaced_count)
        )
        last_replacement_history_reachable = (
            last_replaced_count
            <= jnp.asarray(self._config.replacement_quota, dtype=jnp.uint32)
        ) & jnp.where(
            observation_started,
            jnp.all((state.age == 0) == state.last_replaced_mask),
            jnp.all(~state.last_replaced_mask),
        )
        last_replacement_history_reachable = (
            last_replacement_history_reachable
            & jnp.where(
                last_replaced_count != 0,
                jnp.any(state.replacement_event_words != 0)
                & replacement_words_cover_last_mask,
                jnp.asarray(True, dtype=jnp.bool_),
            )
        )
        causal_linear_envelope = jnp.minimum(
            jnp.float32(1.0),
            state.causal_evidence_count.astype(jnp.float32)
            * jnp.asarray(
                1.0 - self._config.utility_decay + _FLOAT32_EPSILON,
                dtype=jnp.float32,
            ),
        )
        age_within_observation_clock = (
            state.observation_words[0] != 0
        ) | jnp.all(state.age <= state.observation_words[1])
        maximum_replacements = _saturating_words_product(
            state.replacement_event_words,
            self._config.replacement_quota,
        )
        last_replacement_is_scrubbed = jnp.all(
            (~state.last_replaced_mask)
            | (
                (state.utility == 0.0)
                & (state.causal_utility == 0.0)
                & (state.age == 0)
                & (state.support == 0)
                & (state.causal_evidence_count == 0)
                & (state.last_effective_contribution == 0.0)
                & (state.last_causal_deletion_loss_change == 0.0)
            )
        )
        return (
            jnp.array_equal(state.lifecycle_fingerprint, self._fingerprint)
            & jnp.all(jnp.isfinite(state.utility))
            & jnp.all(state.utility >= 0.0)
            & jnp.all(jnp.isfinite(state.causal_utility))
            & jnp.all(state.causal_utility >= 0.0)
            & jnp.all(state.causal_utility <= 1.0)
            & jnp.all(state.causal_utility <= causal_linear_envelope)
            & jnp.all(jnp.isfinite(state.last_effective_contribution))
            & jnp.all(state.last_effective_contribution >= 0.0)
            & jnp.all(jnp.isfinite(state.last_causal_deletion_loss_change))
            & jnp.all(state.support <= state.age)
            & jnp.all(state.causal_evidence_count <= state.age)
            & age_within_observation_clock
            & jnp.all((~zero_evidence) | (state.causal_utility == 0.0))
            & jnp.all(
                (~zero_evidence)
                | (state.last_causal_deletion_loss_change == 0.0)
            )
            & last_replacement_is_scrubbed
            & last_replacement_history_reachable
            & ~jnp.any(state.last_replaced_mask & self._protected_mask)
            & _counter_mirror_valid(state.observation_words, state.observation_count)
            & _counter_mirror_valid(state.replacement_words, state.replacement_count)
            & _counter_mirror_valid(
                state.replacement_event_words,
                state.replacement_event_count,
            )
            & _words_not_earlier(
                state.observation_words,
                state.replacement_event_words,
            )
            & _words_not_earlier(
                state.replacement_words,
                state.replacement_event_words,
            )
            & _words_not_earlier(
                maximum_replacements,
                state.replacement_words,
            )
        )

    def state_valid(self, state: RTUGenerateAndTestState) -> Bool[Array, ""]:
        """Validate fixed shapes, typed ownership, finite telemetry, and clocks."""

        self._check_state_contract(state)
        return self._state_is_valid(state)

    def composition_valid(
        self,
        state: RTUGenerateAndTestCompositionState,
    ) -> Bool[Array, ""]:
        """Validate both independently clocked halves of a composition."""

        if type(state) is not RTUGenerateAndTestCompositionState:
            raise TypeError("state must be an exact RTUGenerateAndTestCompositionState")
        return self.state_valid(state.lifecycle) & self.builder.state_valid(state.builder)

    def _check_gradient_contract(self, gradient: Array) -> None:
        _require_array(
            gradient,
            name="downstream_loss_gradient",
            shape=(self.config.builder.feature_dim(),),
            dtype=jnp.dtype(jnp.float32),
        )

    def _check_advance_receipt_contract(
        self,
        receipt: RTUGenerateAndTestAdvanceReceipt,
    ) -> None:
        if type(receipt) is not RTUGenerateAndTestAdvanceReceipt:
            raise TypeError(
                "advance_receipt must be an exact RTUGenerateAndTestAdvanceReceipt"
            )
        observation_dim = self.config.builder.observation_dim
        for name, shape, dtype in (
            ("builder_fingerprint", (_FINGERPRINT_WORDS,), jnp.uint32),
            (
                "source_builder_content_tag_words",
                (_ADVANCE_TAG_WORDS,),
                jnp.uint32,
            ),
            ("source_step_words", (2,), jnp.uint32),
            ("source_update_words", (2,), jnp.uint32),
            ("bootstrap_observation", (observation_dim,), jnp.float32),
            ("previous_action", (), jnp.int32),
            ("previous_reward", (), jnp.float32),
            ("previous_discount", (), jnp.float32),
            ("episode_boundary", (), jnp.bool_),
            ("restart_observation", (observation_dim,), jnp.float32),
            ("sequence_length", (), jnp.int32),
            ("content_tag_words", (_ADVANCE_TAG_WORDS,), jnp.uint32),
        ):
            _require_array(
                getattr(receipt, name),
                name=f"advance_receipt.{name}",
                shape=shape,
                dtype=jnp.dtype(dtype),
            )

    def _advance_content_tag(
        self,
        *,
        source_builder_content_tag_words: Array,
        source_step_words: Array,
        source_update_words: Array,
        bootstrap_observation: Array,
        previous_action: Array,
        previous_reward: Array,
        previous_discount: Array,
        episode_boundary: Array,
        restart_observation: Array,
        sequence_length: Array,
    ) -> UInt[Array, " 8"]:
        words = jnp.concatenate(
            (
                self._advance_fingerprint,
                _uint32_bits(source_builder_content_tag_words),
                _uint32_bits(source_step_words),
                _uint32_bits(source_update_words),
                _uint32_bits(bootstrap_observation),
                _uint32_bits(previous_action),
                _uint32_bits(previous_reward),
                _uint32_bits(previous_discount),
                _uint32_bits(episode_boundary),
                _uint32_bits(restart_observation),
                _uint32_bits(sequence_length),
            )
        )
        return _mix_content_words(words)

    def make_advance_receipt(
        self,
        source_builder_state: RecurrentTraceUnitStateBuilderState,
        *,
        bootstrap_observation: Array,
        previous_action: Array,
        previous_reward: Array,
        previous_discount: Array,
        episode_boundary: Array,
        restart_observation: Array,
    ) -> RTUGenerateAndTestAdvanceReceipt:
        """Bind an exact one-event or bootstrap/reset/restart RTU sequence."""

        self.builder.state_valid(source_builder_state)
        observation_dim = self.config.builder.observation_dim
        for name, value, shape, dtype in (
            (
                "bootstrap_observation",
                bootstrap_observation,
                (observation_dim,),
                jnp.float32,
            ),
            ("previous_action", previous_action, (), jnp.int32),
            ("previous_reward", previous_reward, (), jnp.float32),
            ("previous_discount", previous_discount, (), jnp.float32),
            ("episode_boundary", episode_boundary, (), jnp.bool_),
            (
                "restart_observation",
                restart_observation,
                (observation_dim,),
                jnp.float32,
            ),
        ):
            _require_array(
                value,
                name=name,
                shape=shape,
                dtype=jnp.dtype(dtype),
            )
        return cast(
            RTUGenerateAndTestAdvanceReceipt,
            self._make_advance_receipt_jit(
                source_builder_state,
                bootstrap_observation,
                previous_action,
                previous_reward,
                previous_discount,
                episode_boundary,
                restart_observation,
            ),
        )

    @staticmethod
    def _static_self_jit(function: Any) -> Any:
        return jax.jit(function, static_argnums=(0,))

    @_static_self_jit
    def _make_advance_receipt_jit(
        self,
        source_builder_state: RecurrentTraceUnitStateBuilderState,
        bootstrap_observation: Array,
        previous_action: Array,
        previous_reward: Array,
        previous_discount: Array,
        episode_boundary: Array,
        restart_observation: Array,
    ) -> RTUGenerateAndTestAdvanceReceipt:
        source_tag = _rtu_builder_content_tag(source_builder_state)
        sequence_length = jnp.where(
            episode_boundary,
            jnp.asarray(2, dtype=jnp.int32),
            jnp.asarray(1, dtype=jnp.int32),
        )
        content_tag = self._advance_content_tag(
            source_builder_content_tag_words=source_tag,
            source_step_words=source_builder_state.step_words,
            source_update_words=source_builder_state.update_words,
            bootstrap_observation=bootstrap_observation,
            previous_action=previous_action,
            previous_reward=previous_reward,
            previous_discount=previous_discount,
            episode_boundary=episode_boundary,
            restart_observation=restart_observation,
            sequence_length=sequence_length,
        )
        return RTUGenerateAndTestAdvanceReceipt(
            builder_fingerprint=self._advance_fingerprint,
            source_builder_content_tag_words=source_tag,
            source_step_words=source_builder_state.step_words,
            source_update_words=source_builder_state.update_words,
            bootstrap_observation=bootstrap_observation,
            previous_action=previous_action,
            previous_reward=previous_reward,
            previous_discount=previous_discount,
            episode_boundary=episode_boundary,
            restart_observation=restart_observation,
            sequence_length=sequence_length,
            content_tag_words=content_tag,
        )

    def _fresh_parameter_slices(self, key: Array) -> tuple[Array, Array]:
        cfg = self.config
        quota = cfg.replacement_quota
        split = jr.split(key, quota + 1)
        next_key = split[0]
        roots = split[1:]
        event_dim = cfg.builder.event_dim()
        input_scale = jnp.sqrt(jnp.asarray(event_dim, dtype=jnp.float32))
        min_radius = jnp.asarray(
            max(cfg.builder.r_min**2, _FLOAT32_TINY),
            dtype=jnp.float32,
        )
        max_radius = jnp.asarray(cfg.builder.r_max**2, dtype=jnp.float32)
        max_phase = jnp.asarray(cfg.builder.max_phase, dtype=jnp.float32)

        def draw(root: Array) -> Array:
            nu_key, theta_key, real_key, imaginary_key = jr.split(root, 4)
            radius_squared = jr.uniform(
                nu_key,
                (),
                minval=min_radius,
                maxval=max_radius,
                dtype=jnp.float32,
            )
            radius_squared = jnp.maximum(radius_squared, min_radius)
            nu_log = jnp.log(-0.5 * jnp.log(radius_squared))
            phase = max_phase * jr.uniform(theta_key, (), dtype=jnp.float32)
            phase = jnp.maximum(
                phase,
                jnp.asarray(_FLOAT32_TINY, dtype=jnp.float32),
            )
            theta_log = jnp.log(phase)
            real = jr.normal(real_key, (event_dim,), dtype=jnp.float32) / input_scale
            imaginary = (
                jr.normal(imaginary_key, (event_dim,), dtype=jnp.float32)
                / input_scale
            )
            return jnp.concatenate(
                (jnp.atleast_1d(nu_log), jnp.atleast_1d(theta_log), real, imaginary)
            )

        return next_key, jax.vmap(draw)(roots)

    def _replace_parameter_units(
        self,
        parameters: Array,
        selected_indices: Array,
        selected_slots: Array,
        selected_mask: Array,
        fresh_slices: Array,
    ) -> Array:
        cfg = self.config.builder
        hidden = cfg.hidden_dim
        event_dim = cfg.event_dim()
        nu = parameters[:hidden]
        theta = parameters[hidden : 2 * hidden]
        offset = 2 * hidden
        real = parameters[offset : offset + hidden * event_dim].reshape(
            (hidden, event_dim)
        )
        offset += hidden * event_dim
        imaginary = parameters[offset:].reshape((hidden, event_dim))

        unit_ids = jnp.arange(hidden, dtype=jnp.int32)
        selected_by_slot = selected_slots[:, None] & (
            selected_indices[:, None] == unit_ids[None, :]
        )
        selected_rank = jnp.argmax(selected_by_slot, axis=0)
        unit_slices = fresh_slices[selected_rank]
        nu = jnp.where(selected_mask, unit_slices[:, 0], nu)
        theta = jnp.where(selected_mask, unit_slices[:, 1], theta)
        real = jnp.where(selected_mask[:, None], unit_slices[:, 2 : 2 + event_dim], real)
        imaginary = jnp.where(
            selected_mask[:, None],
            unit_slices[:, 2 + event_dim :],
            imaginary,
        )
        return jnp.concatenate(
            (nu, theta, real.reshape((-1,)), imaginary.reshape((-1,)))
        ).astype(jnp.float32)

    @staticmethod
    def _scrub_sensitivities(
        sensitivities: RTUSensitivities,
        selected_mask: Array,
    ) -> RTUSensitivities:
        polar_mask = selected_mask[None, :]
        input_mask = selected_mask[None, :, None]
        return RTUSensitivities(
            nu_log=jnp.where(polar_mask, 0.0, sensitivities.nu_log),
            theta_log=jnp.where(polar_mask, 0.0, sensitivities.theta_log),
            b_real=jnp.where(input_mask, 0.0, sensitivities.b_real),
            b_imag=jnp.where(input_mask, 0.0, sensitivities.b_imag),
        )

    def _candidate_builder(
        self,
        live: RecurrentTraceUnitStateBuilderState,
        selected_indices: Array,
        selected_slots: Array,
        selected_mask: Array,
        fresh_slices: Array,
    ) -> tuple[RecurrentTraceUnitStateBuilderState, Bool[Array, ""]]:
        has_replacement = jnp.any(selected_mask)
        new_parameters = self._replace_parameter_units(
            live.parameters,
            selected_indices,
            selected_slots,
            selected_mask,
            fresh_slices,
        )
        next_update_words, update_capacity = _increment_words(live.update_words)
        rtu_state = RTUState(
            real=jnp.where(selected_mask, 0.0, live.rtu_state.real),
            imaginary=jnp.where(selected_mask, 0.0, live.rtu_state.imaginary),
        )
        sensitivities = self._scrub_sensitivities(
            live.sensitivities,
            selected_mask,
        )

        taylor_trace: RTUSensitivities | None = None
        source_parameters: Array | None = None
        parameter_delta: Array | None = None
        source_update_words: Array | None = None
        if self.config.builder.rtrl_taylor_correction:
            if (
                live.taylor_trace is None
                or live.sensitivity_source_parameters is None
                or live.sensitivity_parameter_delta is None
                or live.sensitivity_source_update_words is None
            ):
                raise ValueError("Taylor-enabled live RTU state lacks ownership fields")
            taylor_trace = self._scrub_sensitivities(
                live.taylor_trace,
                selected_mask,
            )
            source_parameters = self._replace_parameter_units(
                live.sensitivity_source_parameters,
                selected_indices,
                selected_slots,
                selected_mask,
                fresh_slices,
            )
            parameter_delta = jnp.where(
                self._parameter_mask(selected_mask),
                0.0,
                live.sensitivity_parameter_delta,
            )
            source_update_words = live.sensitivity_source_update_words

        candidate = RecurrentTraceUnitStateBuilderState(
            parameters=jnp.where(has_replacement, new_parameters, live.parameters),
            rtu_state=RTUState(
                real=jnp.where(has_replacement, rtu_state.real, live.rtu_state.real),
                imaginary=jnp.where(
                    has_replacement,
                    rtu_state.imaginary,
                    live.rtu_state.imaginary,
                ),
            ),
            sensitivities=RTUSensitivities(
                nu_log=jnp.where(
                    has_replacement,
                    sensitivities.nu_log,
                    live.sensitivities.nu_log,
                ),
                theta_log=jnp.where(
                    has_replacement,
                    sensitivities.theta_log,
                    live.sensitivities.theta_log,
                ),
                b_real=jnp.where(
                    has_replacement,
                    sensitivities.b_real,
                    live.sensitivities.b_real,
                ),
                b_imag=jnp.where(
                    has_replacement,
                    sensitivities.b_imag,
                    live.sensitivities.b_imag,
                ),
            ),
            taylor_trace=(
                RTUSensitivities(
                    nu_log=jnp.where(
                        has_replacement,
                        cast(RTUSensitivities, taylor_trace).nu_log,
                        cast(RTUSensitivities, live.taylor_trace).nu_log,
                    ),
                    theta_log=jnp.where(
                        has_replacement,
                        cast(RTUSensitivities, taylor_trace).theta_log,
                        cast(RTUSensitivities, live.taylor_trace).theta_log,
                    ),
                    b_real=jnp.where(
                        has_replacement,
                        cast(RTUSensitivities, taylor_trace).b_real,
                        cast(RTUSensitivities, live.taylor_trace).b_real,
                    ),
                    b_imag=jnp.where(
                        has_replacement,
                        cast(RTUSensitivities, taylor_trace).b_imag,
                        cast(RTUSensitivities, live.taylor_trace).b_imag,
                    ),
                )
                if self.config.builder.rtrl_taylor_correction
                else None
            ),
            sensitivity_source_parameters=(
                jnp.where(
                    has_replacement,
                    cast(Array, source_parameters),
                    cast(Array, live.sensitivity_source_parameters),
                )
                if self.config.builder.rtrl_taylor_correction
                else None
            ),
            sensitivity_parameter_delta=(
                jnp.where(
                    has_replacement,
                    cast(Array, parameter_delta),
                    cast(Array, live.sensitivity_parameter_delta),
                )
                if self.config.builder.rtrl_taylor_correction
                else None
            ),
            sensitivity_source_update_words=(
                cast(Array, source_update_words)
                if self.config.builder.rtrl_taylor_correction
                else None
            ),
            step_count=live.step_count,
            step_words=live.step_words,
            update_count=jnp.where(
                has_replacement,
                _saturating_count_increment(live.update_count),
                live.update_count,
            ),
            update_words=jnp.where(
                has_replacement & update_capacity,
                next_update_words,
                live.update_words,
            ),
            last_gradient_norm=live.last_gradient_norm,
        )
        return candidate, (~has_replacement) | update_capacity

    def _parameter_mask(self, selected_mask: Array) -> Bool[Array, " parameter_count"]:
        event_dim = self.config.builder.event_dim()
        return jnp.concatenate(
            (
                selected_mask,
                selected_mask,
                jnp.repeat(selected_mask, event_dim),
                jnp.repeat(selected_mask, event_dim),
            )
        )

    def _expected_live_builder(
        self,
        pre_update: RecurrentTraceUnitStateBuilderState,
        learning_proposal: StateBuilderLearningProposal | None,
        advance_receipt: RTUGenerateAndTestAdvanceReceipt | None,
    ) -> tuple[
        RecurrentTraceUnitStateBuilderState,
        Bool[Array, ""],
        Bool[Array, ""],
        StateBuilderLearningDiagnostics | None,
    ]:
        advanced = pre_update
        advance_receipt_valid = jnp.asarray(True, dtype=jnp.bool_)
        sequence_applied = jnp.asarray(True, dtype=jnp.bool_)
        if advance_receipt is not None:
            expected_source_tag = _rtu_builder_content_tag(pre_update)
            expected_length = jnp.where(
                advance_receipt.episode_boundary,
                jnp.asarray(2, dtype=jnp.int32),
                jnp.asarray(1, dtype=jnp.int32),
            )
            expected_content_tag = self._advance_content_tag(
                source_builder_content_tag_words=(
                    advance_receipt.source_builder_content_tag_words
                ),
                source_step_words=advance_receipt.source_step_words,
                source_update_words=advance_receipt.source_update_words,
                bootstrap_observation=advance_receipt.bootstrap_observation,
                previous_action=advance_receipt.previous_action,
                previous_reward=advance_receipt.previous_reward,
                previous_discount=advance_receipt.previous_discount,
                episode_boundary=advance_receipt.episode_boundary,
                restart_observation=advance_receipt.restart_observation,
                sequence_length=advance_receipt.sequence_length,
            )
            unused_restart_is_canonical = jnp.array_equal(
                jax.lax.bitcast_convert_type(
                    advance_receipt.restart_observation,
                    jnp.uint32,
                ),
                jax.lax.bitcast_convert_type(
                    advance_receipt.bootstrap_observation,
                    jnp.uint32,
                ),
            )
            advance_receipt_valid = (
                jnp.array_equal(
                    advance_receipt.builder_fingerprint,
                    self._advance_fingerprint,
                )
                & jnp.array_equal(
                    advance_receipt.source_builder_content_tag_words,
                    expected_source_tag,
                )
                & jnp.array_equal(
                    advance_receipt.source_step_words,
                    pre_update.step_words,
                )
                & jnp.array_equal(
                    advance_receipt.source_update_words,
                    pre_update.update_words,
                )
                & (advance_receipt.sequence_length == expected_length)
                & jnp.array_equal(
                    advance_receipt.content_tag_words,
                    expected_content_tag,
                )
                & (
                    advance_receipt.episode_boundary
                    | unused_restart_is_canonical
                )
            )
            bootstrap = self.builder.update_with_status(
                pre_update,
                advance_receipt.bootstrap_observation,
                advance_receipt.previous_action,
                advance_receipt.previous_reward,
                advance_receipt.previous_discount,
            )
            reset = self.builder.reset_episode(bootstrap.state)
            restart = self.builder.update_with_status(
                reset,
                advance_receipt.restart_observation,
                jnp.asarray(-1, dtype=jnp.int32),
                jnp.asarray(0.0, dtype=jnp.float32),
                jnp.asarray(1.0, dtype=jnp.float32),
            )
            sequence_applied = (
                advance_receipt_valid
                & bootstrap.transition_applied
                & jnp.where(
                    advance_receipt.episode_boundary,
                    restart.transition_applied,
                    jnp.asarray(True, dtype=jnp.bool_),
                )
            )
            expected_advanced = cast(
                RecurrentTraceUnitStateBuilderState,
                jax.lax.cond(
                    advance_receipt.episode_boundary,
                    lambda: restart.state,
                    lambda: bootstrap.state,
                ),
            )
            advanced = cast(
                RecurrentTraceUnitStateBuilderState,
                jax.lax.cond(
                    sequence_applied,
                    lambda: expected_advanced,
                    lambda: pre_update,
                ),
            )
        if learning_proposal is None:
            return advanced, sequence_applied, advance_receipt_valid, None
        live, diagnostics = self.builder.commit_learning_update(
            advanced,
            learning_proposal,
        )
        return (
            live,
            sequence_applied & diagnostics.applied,
            advance_receipt_valid,
            diagnostics,
        )

    def _make_proposal(
        self,
        source_state: RTUGenerateAndTestState,
        pre_update_builder_state: RecurrentTraceUnitStateBuilderState,
        downstream_loss_gradient: Array,
        learning_proposal: StateBuilderLearningProposal | None,
        advance_receipt: RTUGenerateAndTestAdvanceReceipt | None,
        replacement_allowed: Array,
        causal_deletion_loss_change: Array,
        causal_deletion_evidence_available: Array,
        require_causal_evidence: Array,
    ) -> RTUGenerateAndTestProposal:
        cfg = self.config
        hidden = cfg.builder.hidden_dim
        (
            live_builder_state,
            ordinary_advance_valid,
            advance_receipt_valid,
            ordinary_learning_diagnostics,
        ) = self._expected_live_builder(
            pre_update_builder_state,
            learning_proposal,
            advance_receipt,
        )
        source_state_valid = self._state_is_valid(source_state)
        pre_update_builder_valid = self.builder.state_valid(pre_update_builder_state)
        live_builder_valid = self.builder.state_valid(live_builder_state)
        gradient = jnp.asarray(downstream_loss_gradient, dtype=jnp.float32)
        input_valid = jnp.all(jnp.isfinite(gradient))
        hidden_gradient = gradient[-2 * hidden :].reshape((2, hidden))
        safe_gradient = jnp.where(jnp.isfinite(hidden_gradient), hidden_gradient, 0.0)
        safe_real = jnp.where(
            jnp.isfinite(pre_update_builder_state.rtu_state.real),
            pre_update_builder_state.rtu_state.real,
            0.0,
        )
        safe_imaginary = jnp.where(
            jnp.isfinite(pre_update_builder_state.rtu_state.imaginary),
            pre_update_builder_state.rtu_state.imaginary,
            0.0,
        )
        effective_contribution = (
            jnp.abs(safe_real * safe_gradient[0])
            + jnp.abs(safe_imaginary * safe_gradient[1])
        )
        contribution_valid = jnp.all(jnp.isfinite(effective_contribution))
        observed_utility = (
            jnp.asarray(cfg.utility_decay, dtype=jnp.float32) * source_state.utility
            + jnp.asarray(1.0 - cfg.utility_decay, dtype=jnp.float32)
            * effective_contribution
        )
        utility_valid = jnp.all(jnp.isfinite(observed_utility)) & jnp.all(
            observed_utility >= 0.0
        )

        causal_change_finite = jnp.all(jnp.isfinite(causal_deletion_loss_change))
        causal_evidence_available = (
            causal_deletion_evidence_available & causal_change_finite
        )
        safe_causal_change = jnp.where(
            jnp.isfinite(causal_deletion_loss_change),
            causal_deletion_loss_change,
            0.0,
        )
        positive_causal_change = jnp.maximum(safe_causal_change, 0.0)
        bounded_causal_gain = positive_causal_change / (
            1.0 + positive_causal_change
        )
        causal_candidate = (
            jnp.asarray(cfg.utility_decay, dtype=jnp.float32)
            * source_state.causal_utility
            + jnp.asarray(1.0 - cfg.utility_decay, dtype=jnp.float32)
            * bounded_causal_gain
        )
        observed_causal_utility = jnp.where(
            causal_evidence_available,
            causal_candidate,
            source_state.causal_utility,
        )
        causal_support_capacity = jnp.all(
            ~causal_evidence_available
            | (
                source_state.causal_evidence_count
                < jnp.asarray(_UINT32_MAX, dtype=jnp.uint32)
            )
        )
        observed_causal_evidence_count = source_state.causal_evidence_count + (
            causal_evidence_available.astype(jnp.uint32)
        )
        causal_ready = causal_evidence_available & (
            observed_causal_evidence_count
            >= jnp.asarray(cfg.minimum_causal_evidence, dtype=jnp.uint32)
        )
        selection_utility = jnp.where(
            require_causal_evidence,
            observed_causal_utility,
            observed_utility,
        )
        causal_utility_valid = (
            jnp.all(jnp.isfinite(observed_causal_utility))
            & jnp.all(observed_causal_utility >= 0.0)
            & jnp.all(jnp.isfinite(selection_utility))
            & jnp.all(selection_utility >= 0.0)
        )

        age_capacity = jnp.all(
            source_state.age < jnp.asarray(_UINT32_MAX, dtype=jnp.uint32)
        )
        observed_age = source_state.age + jnp.asarray(1, dtype=jnp.uint32)
        sensitivity = jnp.abs(safe_gradient[0]) + jnp.abs(safe_gradient[1])
        supported = sensitivity > jnp.asarray(
            cfg.minimum_sensitivity_for_support,
            dtype=jnp.float32,
        )
        support_capacity = jnp.all(
            ~supported
            | (source_state.support < jnp.asarray(_UINT32_MAX, dtype=jnp.uint32))
        )
        observed_support = source_state.support + supported.astype(jnp.uint32)
        per_unit_capacity = age_capacity & support_capacity
        per_unit_capacity = per_unit_capacity & causal_support_capacity

        observation_words, observation_capacity = _increment_words(
            source_state.observation_words
        )
        observation_count = _saturating_count_increment(source_state.observation_count)
        due = _words_mod(observation_words, cfg.replacement_interval) == 0
        warmed = _words_at_least(observation_words, cfg.warmup_observations)
        eligible = (
            due
            & warmed
            & replacement_allowed
            & ((~require_causal_evidence) | causal_ready)
            & (observed_age >= jnp.asarray(cfg.minimum_age, dtype=jnp.uint32))
            & (
                observed_support
                >= jnp.asarray(cfg.minimum_support, dtype=jnp.uint32)
            )
            & ~self._protected_mask
        )
        scores = jnp.where(eligible, selection_utility, jnp.asarray(jnp.inf))
        stable_order = jnp.argsort(scores, stable=True)
        selected_indices = stable_order[: cfg.replacement_quota].astype(jnp.int32)
        eligible_count = jnp.sum(eligible.astype(jnp.int32))
        selected_slots = jnp.arange(cfg.replacement_quota, dtype=jnp.int32) < jnp.minimum(
            eligible_count,
            jnp.asarray(cfg.replacement_quota, dtype=jnp.int32),
        )
        unit_ids = jnp.arange(hidden, dtype=jnp.int32)
        selected_mask = jnp.any(
            selected_slots[:, None]
            & (selected_indices[:, None] == unit_ids[None, :]),
            axis=0,
        )
        selected_count = jnp.sum(selected_slots.astype(jnp.int32))
        has_replacement = selected_count > 0

        next_key, fresh_parameter_slices = self._fresh_parameter_slices(
            source_state.rng_key
        )
        fresh_valid = jnp.all(jnp.isfinite(fresh_parameter_slices))
        candidate_builder_state, builder_capacity = self._candidate_builder(
            live_builder_state,
            selected_indices,
            selected_slots,
            selected_mask,
            fresh_parameter_slices,
        )
        replacement_words, replacement_count, replacement_capacity = _add_to_counter(
            source_state.replacement_words,
            source_state.replacement_count,
            selected_count,
            maximum_amount=cfg.replacement_quota,
        )
        replacement_event_words, replacement_event_count, event_capacity = (
            _add_to_counter(
                source_state.replacement_event_words,
                source_state.replacement_event_count,
                has_replacement.astype(jnp.int32),
                maximum_amount=1,
            )
        )
        candidate_state = RTUGenerateAndTestState(
            lifecycle_fingerprint=source_state.lifecycle_fingerprint,
            utility=jnp.where(selected_mask, 0.0, observed_utility),
            causal_utility=jnp.where(
                selected_mask,
                0.0,
                observed_causal_utility,
            ),
            age=jnp.where(selected_mask, 0, observed_age).astype(jnp.uint32),
            support=jnp.where(selected_mask, 0, observed_support).astype(jnp.uint32),
            causal_evidence_count=jnp.where(
                selected_mask,
                0,
                observed_causal_evidence_count,
            ).astype(jnp.uint32),
            last_effective_contribution=jnp.where(
                selected_mask,
                0.0,
                effective_contribution,
            ),
            last_causal_deletion_loss_change=jnp.where(
                selected_mask,
                0.0,
                jnp.where(
                    causal_evidence_available,
                    safe_causal_change,
                    source_state.last_causal_deletion_loss_change,
                ),
            ),
            last_replaced_mask=selected_mask,
            rng_key=jax.lax.cond(
                has_replacement,
                lambda: next_key,
                lambda: source_state.rng_key,
            ),
            observation_count=observation_count,
            observation_words=observation_words,
            replacement_count=replacement_count,
            replacement_words=replacement_words,
            replacement_event_count=replacement_event_count,
            replacement_event_words=replacement_event_words,
        )
        candidate_state_valid = self._state_is_valid(candidate_state)
        candidate_builder_valid = self.builder.state_valid(candidate_builder_state)
        valid = (
            source_state_valid
            & pre_update_builder_valid
            & live_builder_valid
            & ordinary_advance_valid
            & input_valid
            & contribution_valid
            & utility_valid
            & causal_change_finite
            & causal_utility_valid
            & observation_capacity
            & per_unit_capacity
            & replacement_capacity
            & event_capacity
            & builder_capacity
            & fresh_valid
            & candidate_state_valid
            & candidate_builder_valid
        )
        return RTUGenerateAndTestProposal(
            lifecycle_fingerprint=self._fingerprint,
            source_state=source_state,
            pre_update_builder_state=pre_update_builder_state,
            live_builder_state=live_builder_state,
            learning_proposal=learning_proposal,
            ordinary_learning_diagnostics=ordinary_learning_diagnostics,
            advance_receipt=advance_receipt,
            downstream_loss_gradient=gradient,
            replacement_allowed=replacement_allowed,
            causal_deletion_loss_change=causal_deletion_loss_change,
            causal_deletion_evidence_declared=(
                causal_deletion_evidence_available
            ),
            causal_deletion_evidence_available=causal_evidence_available,
            causal_deletion_evidence_valid=causal_change_finite,
            causal_evidence_required=require_causal_evidence,
            effective_contribution=effective_contribution,
            observed_utility=observed_utility,
            observed_causal_utility=observed_causal_utility,
            observed_causal_evidence_count=observed_causal_evidence_count,
            selection_utility=selection_utility,
            observed_age=observed_age,
            observed_support=observed_support,
            selected_indices=selected_indices,
            selected_slots=selected_slots,
            selected_mask=selected_mask,
            fresh_parameter_slices=fresh_parameter_slices,
            candidate_state=candidate_state,
            candidate_builder_state=candidate_builder_state,
            ordinary_advance_valid=ordinary_advance_valid,
            advance_receipt_valid=advance_receipt_valid,
            source_state_valid=source_state_valid,
            pre_update_builder_valid=pre_update_builder_valid,
            input_valid=input_valid,
            observation_capacity_available=observation_capacity,
            per_unit_capacity_available=per_unit_capacity,
            replacement_capacity_available=(replacement_capacity & event_capacity),
            builder_capacity_available=builder_capacity,
            candidate_state_valid=candidate_state_valid,
            candidate_builder_valid=candidate_builder_valid,
            valid=valid,
            rejected=~valid,
        )

    def propose(
        self,
        source_state: RTUGenerateAndTestState,
        pre_update_builder_state: RecurrentTraceUnitStateBuilderState,
        downstream_loss_gradient: Array,
        learning_proposal: StateBuilderLearningProposal | None = None,
        advance_receipt: RTUGenerateAndTestAdvanceReceipt | None = None,
        *,
        replacement_allowed: Array | bool = True,
        causal_deletion_loss_change: Array | None = None,
        causal_deletion_evidence_available: Array | bool | None = None,
        require_causal_evidence: Array | bool = False,
    ) -> RTUGenerateAndTestProposal:
        """Observe causal pre-update contribution and form a pure proposal.

        When ``advance_receipt`` is supplied, its bootstrap and optional
        reset/restart sequence is recomputed from ``pre_update_builder_state``.
        ``learning_proposal`` is then committed into that exact destination.
        The resulting state is the only live destination accepted by
        :meth:`commit`. ``replacement_allowed=False`` still observes utility,
        age, support, recurrence, and ordinary learning but defers selection;
        the Prototype integration uses this while an option owns the action.
        When ``require_causal_evidence=True``, a supplied per-unit frozen-head
        deletion-loss change is converted to a positive bounded utility EMA and
        becomes the sole replacement rank. Missing or immature evidence defers
        only replacement. A declared non-finite vector rejects the standalone
        transaction atomically; the live Prototype adapter exposes a failed
        counterfactual as typed invalid evidence and rejects its outer
        transaction. The live adapter constructs the vector internally; the
        standalone seam cannot authenticate a caller-supplied vector.
        """

        self._check_state_contract(source_state)
        self.builder.state_valid(pre_update_builder_state)
        self._check_gradient_contract(downstream_loss_gradient)
        if learning_proposal is not None:
            # Executes the builder's complete static proposal contract before
            # this mechanism enters a trace.
            self.builder.commit_learning_update(
                pre_update_builder_state,
                learning_proposal,
            )
        if advance_receipt is not None:
            self._check_advance_receipt_contract(advance_receipt)
        allowed = jnp.asarray(replacement_allowed)
        _require_array(
            allowed,
            name="replacement_allowed",
            shape=(),
            dtype=jnp.dtype(jnp.bool_),
        )
        if causal_deletion_loss_change is None:
            causal_available = jnp.asarray(
                False
                if causal_deletion_evidence_available is None
                else causal_deletion_evidence_available
            )
            _require_array(
                causal_available,
                name="causal_deletion_evidence_available",
                shape=(),
                dtype=jnp.dtype(jnp.bool_),
            )
            causal_change = jnp.where(
                causal_available,
                jnp.full(
                    (self.config.builder.hidden_dim,),
                    jnp.nan,
                    dtype=jnp.float32,
                ),
                jnp.zeros(
                    (self.config.builder.hidden_dim,),
                    dtype=jnp.float32,
                ),
            )
        else:
            causal_change = jnp.asarray(causal_deletion_loss_change)
            _require_array(
                causal_change,
                name="causal_deletion_loss_change",
                shape=(self.config.builder.hidden_dim,),
                dtype=jnp.dtype(jnp.float32),
            )
            causal_available = jnp.asarray(
                True
                if causal_deletion_evidence_available is None
                else causal_deletion_evidence_available
            )
            _require_array(
                causal_available,
                name="causal_deletion_evidence_available",
                shape=(),
                dtype=jnp.dtype(jnp.bool_),
            )
        causal_required = jnp.asarray(require_causal_evidence)
        _require_array(
            causal_required,
            name="require_causal_evidence",
            shape=(),
            dtype=jnp.dtype(jnp.bool_),
        )
        return cast(
            RTUGenerateAndTestProposal,
            self._propose_jit(
                source_state,
                pre_update_builder_state,
                downstream_loss_gradient,
                learning_proposal,
                advance_receipt,
                allowed,
                causal_change,
                causal_available,
                causal_required,
            ),
        )

    @_static_self_jit
    def _propose_jit(
        self,
        source_state: RTUGenerateAndTestState,
        pre_update_builder_state: RecurrentTraceUnitStateBuilderState,
        downstream_loss_gradient: Array,
        learning_proposal: StateBuilderLearningProposal | None,
        advance_receipt: RTUGenerateAndTestAdvanceReceipt | None,
        replacement_allowed: Array,
        causal_deletion_loss_change: Array,
        causal_deletion_evidence_available: Array,
        require_causal_evidence: Array,
    ) -> RTUGenerateAndTestProposal:
        return self._make_proposal(
            source_state,
            pre_update_builder_state,
            downstream_loss_gradient,
            learning_proposal,
            advance_receipt,
            replacement_allowed,
            causal_deletion_loss_change,
            causal_deletion_evidence_available,
            require_causal_evidence,
        )

    def _check_proposal_contract(self, proposal: RTUGenerateAndTestProposal) -> None:
        if type(proposal) is not RTUGenerateAndTestProposal:
            raise TypeError("proposal must be an exact RTUGenerateAndTestProposal")
        self._check_state_contract(proposal.source_state)
        self._check_state_contract(proposal.candidate_state)
        self.builder.state_valid(proposal.pre_update_builder_state)
        self.builder.state_valid(proposal.live_builder_state)
        self.builder.state_valid(proposal.candidate_builder_state)
        self._check_gradient_contract(proposal.downstream_loss_gradient)
        learning_diagnostics = proposal.ordinary_learning_diagnostics
        if proposal.learning_proposal is None:
            if learning_diagnostics is not None:
                raise TypeError(
                    "proposal without ordinary learning cannot contain diagnostics"
                )
        else:
            if type(learning_diagnostics) is not StateBuilderLearningDiagnostics:
                raise TypeError(
                    "proposal with ordinary learning requires exact diagnostics"
                )
            for name, dtype in (
                ("gradient_norm", jnp.float32),
                ("clipped_gradient_norm", jnp.float32),
                ("parameter_update_norm", jnp.float32),
                ("proposal_valid", jnp.bool_),
                ("source_matches", jnp.bool_),
                ("capacity_available", jnp.bool_),
                ("candidate_parameters_valid", jnp.bool_),
                ("applied", jnp.bool_),
                ("fixed_noop", jnp.bool_),
                ("valid", jnp.bool_),
                ("rejected", jnp.bool_),
                ("lifetime_counter_valid", jnp.bool_),
                ("lifetime_capacity_available", jnp.bool_),
                ("update_applied", jnp.bool_),
            ):
                _require_array(
                    getattr(learning_diagnostics, name),
                    name=f"proposal.ordinary_learning_diagnostics.{name}",
                    shape=(),
                    dtype=jnp.dtype(dtype),
                )
            for name in ("pre_update_words", "post_update_words"):
                _require_array(
                    getattr(learning_diagnostics, name),
                    name=f"proposal.ordinary_learning_diagnostics.{name}",
                    shape=(2,),
                    dtype=jnp.dtype(jnp.uint32),
                )
        if proposal.advance_receipt is not None:
            self._check_advance_receipt_contract(proposal.advance_receipt)
        hidden = self.config.builder.hidden_dim
        quota = self.config.replacement_quota
        slice_width = 2 + 2 * self.config.builder.event_dim()
        for name, shape, dtype in (
            ("lifecycle_fingerprint", (_FINGERPRINT_WORDS,), jnp.uint32),
            ("effective_contribution", (hidden,), jnp.float32),
            ("observed_utility", (hidden,), jnp.float32),
            ("causal_deletion_loss_change", (hidden,), jnp.float32),
            ("observed_causal_utility", (hidden,), jnp.float32),
            ("observed_causal_evidence_count", (hidden,), jnp.uint32),
            ("selection_utility", (hidden,), jnp.float32),
            ("observed_age", (hidden,), jnp.uint32),
            ("observed_support", (hidden,), jnp.uint32),
            ("selected_indices", (quota,), jnp.int32),
            ("selected_slots", (quota,), jnp.bool_),
            ("selected_mask", (hidden,), jnp.bool_),
            ("fresh_parameter_slices", (quota, slice_width), jnp.float32),
        ):
            _require_array(
                getattr(proposal, name),
                name=f"proposal.{name}",
                shape=shape,
                dtype=jnp.dtype(dtype),
            )
        for name in (
            "ordinary_advance_valid",
            "advance_receipt_valid",
            "source_state_valid",
            "pre_update_builder_valid",
            "input_valid",
            "observation_capacity_available",
            "per_unit_capacity_available",
            "replacement_capacity_available",
            "builder_capacity_available",
            "candidate_state_valid",
            "candidate_builder_valid",
            "valid",
            "rejected",
            "replacement_allowed",
            "causal_deletion_evidence_declared",
            "causal_deletion_evidence_available",
            "causal_deletion_evidence_valid",
            "causal_evidence_required",
        ):
            _require_array(
                getattr(proposal, name),
                name=f"proposal.{name}",
                shape=(),
                dtype=jnp.dtype(jnp.bool_),
            )

    def commit(
        self,
        destination_state: RTUGenerateAndTestState,
        live_builder_state: RecurrentTraceUnitStateBuilderState,
        proposal: RTUGenerateAndTestProposal,
    ) -> RTUGenerateAndTestCommitResult:
        """Recompute and atomically commit only an exact current proposal."""

        self._check_state_contract(destination_state)
        self.builder.state_valid(live_builder_state)
        self._check_proposal_contract(proposal)
        return cast(
            RTUGenerateAndTestCommitResult,
            self._commit_jit(destination_state, live_builder_state, proposal),
        )

    @_static_self_jit
    def _commit_jit(
        self,
        destination_state: RTUGenerateAndTestState,
        live_builder_state: RecurrentTraceUnitStateBuilderState,
        proposal: RTUGenerateAndTestProposal,
    ) -> RTUGenerateAndTestCommitResult:
        expected = self._make_proposal(
            proposal.source_state,
            proposal.pre_update_builder_state,
            proposal.downstream_loss_gradient,
            proposal.learning_proposal,
            proposal.advance_receipt,
            proposal.replacement_allowed,
            proposal.causal_deletion_loss_change,
            proposal.causal_deletion_evidence_declared,
            proposal.causal_evidence_required,
        )
        proposal_integrity = _tree_exact_equal(expected, proposal)
        lifecycle_source_matches = _tree_exact_equal(
            destination_state,
            proposal.source_state,
        )
        live_builder_matches = _tree_exact_equal(
            live_builder_state,
            proposal.live_builder_state,
        )
        proposal_valid = proposal_integrity & proposal.valid
        applied = (
            self._state_is_valid(destination_state)
            & self.builder.state_valid(live_builder_state)
            & lifecycle_source_matches
            & live_builder_matches
            & proposal_valid
        )
        next_state = cast(
            RTUGenerateAndTestState,
            jax.lax.cond(
                applied,
                lambda: expected.candidate_state,
                lambda: destination_state,
            ),
        )
        next_builder = cast(
            RecurrentTraceUnitStateBuilderState,
            jax.lax.cond(
                applied,
                lambda: expected.candidate_builder_state,
                lambda: live_builder_state,
            ),
        )
        diagnostics = RTUGenerateAndTestDiagnostics(
            effective_contribution=expected.effective_contribution,
            observed_utility=expected.observed_utility,
            causal_deletion_loss_change=(
                expected.causal_deletion_loss_change
            ),
            observed_causal_utility=expected.observed_causal_utility,
            observed_causal_evidence_count=(
                expected.observed_causal_evidence_count
            ),
            selection_utility=expected.selection_utility,
            causal_deletion_evidence_available=(
                expected.causal_deletion_evidence_available
            ),
            causal_deletion_evidence_valid=(
                expected.causal_deletion_evidence_valid
            ),
            causal_evidence_required=expected.causal_evidence_required,
            selected_indices=expected.selected_indices,
            selected_slots=expected.selected_slots,
            selected_mask=expected.selected_mask,
            selected_count=jnp.sum(expected.selected_slots.astype(jnp.int32)),
            proposal_integrity=proposal_integrity,
            lifecycle_source_matches=lifecycle_source_matches,
            live_builder_matches=live_builder_matches,
            proposal_valid=proposal_valid,
            applied=applied,
            rejected=~applied,
            pre_observation_words=destination_state.observation_words,
            post_observation_words=next_state.observation_words,
            pre_replacement_words=destination_state.replacement_words,
            post_replacement_words=next_state.replacement_words,
            pre_replacement_event_words=(
                destination_state.replacement_event_words
            ),
            post_replacement_event_words=next_state.replacement_event_words,
            pre_builder_update_words=live_builder_state.update_words,
            post_builder_update_words=next_builder.update_words,
            pre_rng_key_data=jr.key_data(destination_state.rng_key),
            post_rng_key_data=jr.key_data(next_state.rng_key),
        )
        composition = RTUGenerateAndTestCompositionState(
            lifecycle=next_state,
            builder=next_builder,
        )
        return RTUGenerateAndTestCommitResult(
            state=next_state,
            builder_state=next_builder,
            composition=composition,
            diagnostics=diagnostics,
        )

    def transact(
        self,
        source_state: RTUGenerateAndTestState,
        builder_state: RecurrentTraceUnitStateBuilderState,
        downstream_loss_gradient: Array,
    ) -> RTUGenerateAndTestCommitResult:
        """Convenience pure proposal/commit with no ordinary learning advance."""

        self._check_state_contract(source_state)
        self.builder.state_valid(builder_state)
        self._check_gradient_contract(downstream_loss_gradient)
        return cast(
            RTUGenerateAndTestCommitResult,
            self._transact_jit(source_state, builder_state, downstream_loss_gradient),
        )

    @_static_self_jit
    def _transact_jit(
        self,
        source_state: RTUGenerateAndTestState,
        builder_state: RecurrentTraceUnitStateBuilderState,
        downstream_loss_gradient: Array,
    ) -> RTUGenerateAndTestCommitResult:
        proposal = self._make_proposal(
            source_state,
            builder_state,
            downstream_loss_gradient,
            None,
            None,
            jnp.asarray(True, dtype=jnp.bool_),
            jnp.zeros(
                (self.config.builder.hidden_dim,),
                dtype=jnp.float32,
            ),
            jnp.asarray(False, dtype=jnp.bool_),
            jnp.asarray(False, dtype=jnp.bool_),
        )
        return cast(
            RTUGenerateAndTestCommitResult,
            self._commit_jit(source_state, builder_state, proposal),
        )

    def resource_budget(
        self,
        state: RTUGenerateAndTestState | None = None,
        builder_state: RecurrentTraceUnitStateBuilderState | None = None,
    ) -> RTUGenerateAndTestResourceBudget:
        """Measure exact persistent state and the larger advanced proposal tree."""

        lifecycle = (
            self.init(jr.key(0, impl="threefry2x32")) if state is None else state
        )
        builder = (
            self.builder.init(jr.key(1, impl="threefry2x32"))
            if builder_state is None
            else builder_state
        )
        if not bool(jax.device_get(self.state_valid(lifecycle))):
            raise ValueError("cannot account for an invalid lifecycle state")
        if not bool(jax.device_get(self.builder.state_valid(builder))):
            raise ValueError("cannot account for an invalid RTU builder state")
        gradient = jnp.zeros((self.config.builder.feature_dim(),), dtype=jnp.float32)
        learning_proposal = self.builder.propose_learning_update(builder, gradient)
        zero_observation = jnp.zeros(
            (self.config.builder.observation_dim,),
            dtype=jnp.float32,
        )
        advance_receipt = self.make_advance_receipt(
            builder,
            bootstrap_observation=zero_observation,
            previous_action=jnp.asarray(-1, dtype=jnp.int32),
            previous_reward=jnp.asarray(0.0, dtype=jnp.float32),
            previous_discount=jnp.asarray(1.0, dtype=jnp.float32),
            episode_boundary=jnp.asarray(False, dtype=jnp.bool_),
            restart_observation=zero_observation,
        )
        maximum_proposal = self.propose(
            lifecycle,
            builder,
            gradient,
            learning_proposal,
            advance_receipt,
        )
        lifecycle_nbytes = _tree_nbytes(lifecycle)
        builder_nbytes = _tree_nbytes(builder)
        composition_nbytes = _tree_nbytes(
            RTUGenerateAndTestCompositionState(
                lifecycle=lifecycle,
                builder=builder,
            )
        )
        return RTUGenerateAndTestResourceBudget(
            hidden_units=self.config.builder.hidden_dim,
            replacement_quota=self.config.replacement_quota,
            parameter_slice_scalars=2 + 2 * self.config.builder.event_dim(),
            lifecycle_state_nbytes=lifecycle_nbytes,
            builder_state_nbytes=builder_nbytes,
            composition_state_nbytes=composition_nbytes,
            maximum_proposal_nbytes=_tree_nbytes(maximum_proposal),
            random_replacement_roots_per_observation=self.config.replacement_quota,
            random_subkeys_per_replacement_root=4,
        )


def save_rtu_generate_and_test_checkpoint(
    lifecycle: RTUGenerateAndTest,
    state: RTUGenerateAndTestCompositionState,
    path: str | Path,
) -> None:
    """Persist an exact valid L0 composition and its resource disclosure."""

    if type(lifecycle) is not RTUGenerateAndTest:
        raise TypeError("lifecycle must be an exact RTUGenerateAndTest")
    if not bool(jax.device_get(lifecycle.composition_valid(state))):
        raise ValueError("refusing to checkpoint an invalid RTU lifecycle composition")
    save_checkpoint(
        state,
        path,
        metadata={
            "schema": RTU_GENERATE_AND_TEST_CHECKPOINT_SCHEMA,
            "state_schema": RTU_GENERATE_AND_TEST_STATE_SCHEMA,
            "evidence_level": RTU_GENERATE_AND_TEST_EVIDENCE_LEVEL,
            "mechanism_status": RTU_GENERATE_AND_TEST_MECHANISM_STATUS,
            "scientific_promotion_allowed": (
                RTU_GENERATE_AND_TEST_SCIENTIFIC_PROMOTION_ALLOWED
            ),
            "config": lifecycle.to_config(),
            "resource_budget": lifecycle.resource_budget(
                state.lifecycle,
                state.builder,
            ).to_config(),
        },
    )


def load_rtu_generate_and_test_checkpoint(
    path: str | Path,
) -> tuple[RTUGenerateAndTest, RTUGenerateAndTestCompositionState]:
    """Restore only an exact current-schema, nonpromoting composition."""

    metadata = load_checkpoint_metadata(path)
    expected_fields = {
        "schema",
        "state_schema",
        "evidence_level",
        "mechanism_status",
        "scientific_promotion_allowed",
        "config",
        "resource_budget",
    }
    if set(metadata) != expected_fields:
        raise ValueError("RTU generate-and-test checkpoint fields are not exact")
    if metadata["schema"] != RTU_GENERATE_AND_TEST_CHECKPOINT_SCHEMA:
        raise ValueError("RTU generate-and-test checkpoint schema is unsupported")
    if metadata["state_schema"] != RTU_GENERATE_AND_TEST_STATE_SCHEMA:
        raise ValueError("RTU generate-and-test state schema is unsupported")
    if metadata["evidence_level"] != RTU_GENERATE_AND_TEST_EVIDENCE_LEVEL:
        raise ValueError("RTU generate-and-test checkpoint evidence level differs")
    if metadata["mechanism_status"] != RTU_GENERATE_AND_TEST_MECHANISM_STATUS:
        raise ValueError("RTU generate-and-test checkpoint is not not_assessed")
    if metadata["scientific_promotion_allowed"] is not False:
        raise ValueError("RTU generate-and-test checkpoint cannot claim promotion")
    raw_config = metadata["config"]
    if type(raw_config) is not dict:
        raise TypeError("RTU generate-and-test checkpoint config must be an exact dict")
    lifecycle = RTUGenerateAndTest.from_config(raw_config)
    template = lifecycle.init_composition(
        jr.key(0, impl="threefry2x32"),
        jr.key(1, impl="threefry2x32"),
    )
    restored, second_metadata = load_checkpoint(template, path)
    if not _exact_json_equal(metadata, second_metadata):
        raise ValueError("RTU generate-and-test checkpoint metadata changed between reads")
    state = cast(RTUGenerateAndTestCompositionState, restored)
    if not bool(jax.device_get(lifecycle.composition_valid(state))):
        raise ValueError("RTU generate-and-test checkpoint restored invalid state")
    expected_budget = lifecycle.resource_budget(
        state.lifecycle,
        state.builder,
    ).to_config()
    if not _exact_json_equal(metadata["resource_budget"], expected_budget):
        raise ValueError("RTU generate-and-test checkpoint resource contract differs")
    return lifecycle, state


__all__ = [
    "RTU_GENERATE_AND_TEST_ADVANCE_RECEIPT_SCHEMA",
    "RTU_GENERATE_AND_TEST_CHECKPOINT_SCHEMA",
    "RTU_GENERATE_AND_TEST_CONFIG_SCHEMA",
    "RTU_GENERATE_AND_TEST_EVIDENCE_LEVEL",
    "RTU_GENERATE_AND_TEST_MECHANISM_STATUS",
    "RTU_GENERATE_AND_TEST_SCIENTIFIC_PROMOTION_ALLOWED",
    "RTU_GENERATE_AND_TEST_STATE_SCHEMA",
    "RTUGenerateAndTest",
    "RTUGenerateAndTestAdvanceReceipt",
    "RTUGenerateAndTestCommitResult",
    "RTUGenerateAndTestCompositionState",
    "RTUGenerateAndTestConfig",
    "RTUGenerateAndTestDiagnostics",
    "RTUGenerateAndTestProposal",
    "RTUGenerateAndTestResourceBudget",
    "RTUGenerateAndTestState",
    "load_rtu_generate_and_test_checkpoint",
    "save_rtu_generate_and_test_checkpoint",
]
