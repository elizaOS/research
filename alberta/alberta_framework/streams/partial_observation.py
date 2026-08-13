"""Partial-observation stream wrapper for POMDP testbeds.

The wrapper owns an exact event identity independently of the scan index and
of any clock a child stream may expose.  Its big-endian ``uint32[2]`` clock is
exact through event ``2**64 - 1``; the accompanying ``int32`` count is
saturating compatibility telemetry.  A wrapper event commits atomically only
when its input, current state, child output, candidate child state, and exact
clock all validate.

Three masking modes are supported:

* ``MaskMode.FIXED`` uses the same visibility mask at every event.
* ``MaskMode.RANDOM`` samples an independent visibility mask per event.
* ``MaskMode.PERIODIC`` indexes a fixed schedule from the exact event words.

``True`` means visible in every mask.  The observation dimension and target
are preserved.  A generic ``ScanStream`` child has no universal semantic state
validator: this wrapper can enforce PyTree/leaf preservation, finite floating
state, and an optional child ``state_is_valid`` contract, but cannot infer
unstated child invariants or account for a child's static Python metadata.
"""

from __future__ import annotations

import dataclasses
import enum
import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from typing import Any, TypeVar, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import Array
from jaxtyping import Bool, Int, PRNGKeyArray, UInt

from alberta_framework.core.normalizers import (
    _checked_lifetime_words_increment,
    _lifetime_counter_valid,
    _saturating_int32_counter_increment,
)
from alberta_framework.core.types import TimeStep
from alberta_framework.streams.base import ScanStream

InnerStateT = TypeVar("InnerStateT")

PARTIAL_OBSERVATION_CONFIG_SCHEMA = "alberta.partial-observation-wrapper.config.v2"
PARTIAL_OBSERVATION_STATE_SCHEMA = "alberta.partial-observation-wrapper.state.v2"
PARTIAL_OBSERVATION_RESOURCE_SCHEMA = (
    "alberta.partial-observation-wrapper.resource-budget.v2"
)
PARTIAL_OBSERVATION_EXACT_CLOCK_NBYTES = 12
PARTIAL_OBSERVATION_EXACT_CLOCK_DELTA_NBYTES = 8
PARTIAL_OBSERVATION_RNG_NBYTES = 8
PARTIAL_OBSERVATION_WRAPPER_STATE_NBYTES = (
    PARTIAL_OBSERVATION_EXACT_CLOCK_NBYTES + PARTIAL_OBSERVATION_RNG_NBYTES
)
PARTIAL_OBSERVATION_CHILD_ACCOUNTING = (
    "declared-separately-excluded-from-wrapper-owned"
)

_INT32_MAX = 2**31 - 1
_FLOAT32_MAX = 3.4028234663852886e38


class MaskMode(enum.Enum):
    """Channel-masking mode for ``PartialObservationWrapper``."""

    FIXED = "fixed"
    RANDOM = "random"
    PERIODIC = "periodic"


@chex.dataclass(frozen=True)
class PartialObservationState[InnerStateT]:
    """Persistent state of a partial-observation wrapper.

    ``inner_state`` remains owned by the child.  ``key``, ``step_count``, and
    ``step_words`` are wrapper-owned and commit or roll back as one PyTree.
    """

    inner_state: InnerStateT
    key: PRNGKeyArray
    step_count: Int[Array, ""]
    step_words: UInt[Array, " 2"]


@chex.dataclass(frozen=True)
class PartialObservationStepResult[InnerStateT]:
    """One staged mask event and its atomic-commit diagnostics."""

    timestep: TimeStep
    state: PartialObservationState[InnerStateT]
    visibility_mask: Bool[Array, " feature_dim"]
    pre_step_words: UInt[Array, " 2"]
    post_step_words: UInt[Array, " 2"]
    schedule_index: Int[Array, ""]
    lifetime_counter_valid: Bool[Array, ""]
    lifetime_capacity_available: Bool[Array, ""]
    input_valid: Bool[Array, ""]
    state_valid: Bool[Array, ""]
    output_valid: Bool[Array, ""]
    candidate_state_valid: Bool[Array, ""]
    update_applied: Bool[Array, ""]
    update_rejected: Bool[Array, ""]


@dataclass(frozen=True)
class PartialObservationResourceBudget:
    """Exact JAX-array accounting with child ownership kept separate.

    ``wrapper_owned_nbytes`` includes wrapper state and static mask arrays.
    ``child_state_nbytes`` is declared for composition but deliberately
    excluded from that ownership total so a parent does not double count it.
    Host Python objects and the child's static configuration are outside this
    portable array-byte contract.
    """

    stream_type: str
    mode: str
    feature_dim: int
    schedule_length: int
    wrapper_state_nbytes: int
    mask_metadata_nbytes: int
    child_state_nbytes: int
    wrapper_owned_nbytes: int
    composed_persistent_nbytes: int
    exact_clock_nbytes: int = PARTIAL_OBSERVATION_EXACT_CLOCK_NBYTES
    exact_clock_delta_nbytes: int = PARTIAL_OBSERVATION_EXACT_CLOCK_DELTA_NBYTES
    rng_nbytes: int = PARTIAL_OBSERVATION_RNG_NBYTES
    trainable_scalars: int = 0
    replay_capacity: int = 0
    child_state_accounting: str = PARTIAL_OBSERVATION_CHILD_ACCOUNTING
    schema: str = PARTIAL_OBSERVATION_RESOURCE_SCHEMA

    def to_dict(self) -> dict[str, int | str]:
        """Return the exact versioned resource payload."""

        return {
            "schema": self.schema,
            "stream_type": self.stream_type,
            "mode": self.mode,
            "feature_dim": self.feature_dim,
            "schedule_length": self.schedule_length,
            "wrapper_state_nbytes": self.wrapper_state_nbytes,
            "mask_metadata_nbytes": self.mask_metadata_nbytes,
            "child_state_nbytes": self.child_state_nbytes,
            "wrapper_owned_nbytes": self.wrapper_owned_nbytes,
            "composed_persistent_nbytes": self.composed_persistent_nbytes,
            "exact_clock_nbytes": self.exact_clock_nbytes,
            "exact_clock_delta_nbytes": self.exact_clock_delta_nbytes,
            "rng_nbytes": self.rng_nbytes,
            "trainable_scalars": self.trainable_scalars,
            "replay_capacity": self.replay_capacity,
            "child_state_accounting": self.child_state_accounting,
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> PartialObservationResourceBudget:
        """Strictly validate and reconstruct a resource declaration."""

        expected = {
            "schema",
            "stream_type",
            "mode",
            "feature_dim",
            "schedule_length",
            "wrapper_state_nbytes",
            "mask_metadata_nbytes",
            "child_state_nbytes",
            "wrapper_owned_nbytes",
            "composed_persistent_nbytes",
            "exact_clock_nbytes",
            "exact_clock_delta_nbytes",
            "rng_nbytes",
            "trainable_scalars",
            "replay_capacity",
            "child_state_accounting",
        }
        fields = _require_exact_fields(
            payload,
            expected,
            label="partial-observation resource budget",
        )
        if fields.pop("schema") != PARTIAL_OBSERVATION_RESOURCE_SCHEMA:
            raise ValueError("partial-observation resource schema is unsupported")
        if fields["stream_type"] != "PartialObservationWrapper":
            raise ValueError("partial-observation resource stream type is unsupported")
        if fields["mode"] not in {mode.value for mode in MaskMode}:
            raise ValueError("partial-observation resource mode is unsupported")
        if fields["child_state_accounting"] != PARTIAL_OBSERVATION_CHILD_ACCOUNTING:
            raise ValueError("partial-observation child accounting is invalid")
        integer_fields = expected - {
            "schema",
            "stream_type",
            "mode",
            "child_state_accounting",
        }
        for name in integer_fields:
            if type(fields[name]) is not int or fields[name] < 0:
                raise ValueError(f"partial-observation resource {name} must be non-negative")
        if not 0 < fields["feature_dim"] <= _INT32_MAX:
            raise ValueError("partial-observation resource feature_dim is invalid")
        mode = MaskMode(fields["mode"])
        schedule_length = fields["schedule_length"]
        if mode is MaskMode.PERIODIC:
            if not 0 < schedule_length <= _INT32_MAX:
                raise ValueError("periodic resource schedule_length is invalid")
            expected_mask_nbytes = schedule_length * fields["feature_dim"]
        else:
            if schedule_length != 0:
                raise ValueError("non-periodic resource schedule_length must be zero")
            expected_mask_nbytes = (
                fields["feature_dim"] if mode is MaskMode.FIXED else 0
            )
        if fields["wrapper_state_nbytes"] != PARTIAL_OBSERVATION_WRAPPER_STATE_NBYTES:
            raise ValueError("partial-observation wrapper-state accounting is invalid")
        if fields["mask_metadata_nbytes"] != expected_mask_nbytes:
            raise ValueError("partial-observation mask metadata accounting is invalid")
        expected_owned = fields["wrapper_state_nbytes"] + fields["mask_metadata_nbytes"]
        if fields["wrapper_owned_nbytes"] != expected_owned:
            raise ValueError("partial-observation wrapper-owned accounting is invalid")
        expected_composed = expected_owned + fields["child_state_nbytes"]
        if fields["composed_persistent_nbytes"] != expected_composed:
            raise ValueError("partial-observation composed accounting is invalid")
        if fields["exact_clock_nbytes"] != PARTIAL_OBSERVATION_EXACT_CLOCK_NBYTES:
            raise ValueError("partial-observation exact-clock accounting is invalid")
        if (
            fields["exact_clock_delta_nbytes"]
            != PARTIAL_OBSERVATION_EXACT_CLOCK_DELTA_NBYTES
        ):
            raise ValueError("partial-observation exact-clock delta is invalid")
        if fields["rng_nbytes"] != PARTIAL_OBSERVATION_RNG_NBYTES:
            raise ValueError("partial-observation RNG accounting is invalid")
        if fields["trainable_scalars"] != 0 or fields["replay_capacity"] != 0:
            raise ValueError("partial-observation wrapper owns no learner or replay state")
        return cls(**fields)


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


def _require_array_contract(
    value: Any,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: jnp.dtype,
) -> Array:
    if getattr(value, "shape", None) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {getattr(value, 'shape', None)}")
    if getattr(value, "dtype", None) != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {getattr(value, 'dtype', None)}")
    return cast(Array, value)


def _require_prng_key(key: Any, *, name: str) -> None:
    try:
        key_data = jr.key_data(key)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a scalar JAX PRNG key") from error
    if key_data.shape != (2,) or key_data.dtype != jnp.dtype(jnp.uint32):
        raise TypeError(f"{name} must be a scalar JAX PRNG key")


def _require_child_array_pytree(value: Any, *, name: str) -> None:
    """Require child leaves that have stable JAX shape and dtype metadata."""

    for index, leaf in enumerate(jax.tree.leaves(value)):
        if not hasattr(leaf, "shape") or not hasattr(leaf, "dtype"):
            raise TypeError(f"{name} leaf {index} must be a JAX-compatible array")


def _require_preserved_child_pytree(before: Any, after: Any) -> None:
    """Require a child's returned state to preserve its input PyTree contract."""

    before_structure = jax.tree.structure(before)
    after_structure = jax.tree.structure(after)
    if not bool(cast(Any, before_structure) == after_structure):
        raise TypeError("child state PyTree structure changed during step")
    before_leaves = jax.tree.leaves(before)
    after_leaves = jax.tree.leaves(after)
    for index, (old, new) in enumerate(zip(before_leaves, after_leaves, strict=True)):
        if getattr(old, "shape", None) != getattr(new, "shape", None):
            raise ValueError(f"child state leaf {index} shape changed during step")
        if getattr(old, "dtype", None) != getattr(new, "dtype", None):
            raise TypeError(f"child state leaf {index} dtype changed during step")


def _tree_floating_arrays_finite(value: Any) -> Bool[Array, ""]:
    valid = jnp.asarray(True, dtype=jnp.bool_)
    for leaf in jax.tree.leaves(value):
        dtype = getattr(leaf, "dtype", None)
        if dtype is not None and jnp.issubdtype(dtype, jnp.inexact):
            valid = valid & jnp.all(jnp.isfinite(leaf))
    return valid


def _array_tree_nbytes(value: Any, *, name: str) -> int:
    """Measure concrete JAX-array leaves without inventing host-object sizes."""

    total = 0
    for index, leaf in enumerate(jax.tree.leaves(value)):
        if not hasattr(leaf, "size") or not hasattr(leaf, "dtype"):
            raise TypeError(f"{name} leaf {index} has no portable array-byte accounting")
        itemsize = getattr(leaf.dtype, "itemsize", None)
        if type(itemsize) is not int or itemsize <= 0:
            raise TypeError(f"{name} leaf {index} dtype has no exact itemsize")
        total += int(leaf.size) * itemsize
    return total


def _require_finite_probability(value: Any, *, name: str) -> float:
    valid = (
        not isinstance(value, bool)
        and isinstance(value, Real)
        and math.isfinite(float(value))
        and 0.0 <= float(value) <= 1.0
    )
    if not valid:
        raise ValueError(f"{name} must be a finite real in [0, 1]")
    return float(value)


def _require_finite_sentinel(value: Any) -> float:
    valid = (
        not isinstance(value, bool)
        and isinstance(value, Real)
        and math.isfinite(float(value))
        and abs(float(value)) <= _FLOAT32_MAX
    )
    if not valid:
        raise ValueError("sentinel must be a finite float32-representable real")
    return float(value)


def _require_bool_mask(value: Any, *, name: str, feature_dim: int) -> Array:
    array = jnp.asarray(value)
    if array.shape != (feature_dim,):
        raise ValueError(f"{name} shape {array.shape} != (feature_dim={feature_dim},)")
    if array.dtype != jnp.dtype(jnp.bool_):
        raise TypeError(f"{name} must have dtype bool, got {array.dtype}")
    return array


def _step_input_valid(idx: Any) -> Bool[Array, ""]:
    array = jnp.asarray(idx)
    if array.shape != ():
        raise ValueError(f"idx must be scalar, got shape {array.shape}")
    if not (
        jnp.issubdtype(array.dtype, jnp.integer)
        or jnp.issubdtype(array.dtype, jnp.floating)
    ):
        raise TypeError("idx must have an integer or floating dtype")
    return jnp.isfinite(array)


def _lifetime_words_remainder(words: Array, divisor: int) -> UInt[Array, ""]:
    """Compute a uint64 word pair modulo a positive int32 exactly."""

    _require_array_contract(
        words,
        name="partial-observation step_words",
        shape=(2,),
        dtype=jnp.dtype(jnp.uint32),
    )
    if type(divisor) is not int or not 0 < divisor <= _INT32_MAX:
        raise ValueError("periodic schedule length must be a positive int32 integer")
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
    )


def _host_state_fields(state: Any) -> dict[str, Any]:
    if isinstance(state, Mapping):
        return dict(state)
    if dataclasses.is_dataclass(state) and not isinstance(state, type):
        return {
            field.name: getattr(state, field.name)
            for field in dataclasses.fields(state)
        }
    raise TypeError("legacy partial-observation state must be a mapping or dataclass")


class PartialObservationWrapper[InnerStateT]:
    """Mask observation channels while preserving a child's target."""

    def __init__(
        self,
        inner: ScanStream[InnerStateT],
        mode: MaskMode = MaskMode.FIXED,
        fixed_mask: Bool[Array, " feature_dim"] | None = None,
        mask_prob: float = 0.5,
        schedule: tuple[Bool[Array, " feature_dim"], ...] | None = None,
        sentinel: float = 0.0,
    ) -> None:
        if type(mode) is not MaskMode:
            raise TypeError("mode must be an exact MaskMode")
        feature_dim = getattr(inner, "feature_dim", None)
        if type(feature_dim) is not int or not 0 < feature_dim <= _INT32_MAX:
            raise ValueError("inner.feature_dim must be a positive int32 integer")
        probability = _require_finite_probability(mask_prob, name="mask_prob")
        sentinel_value = _require_finite_sentinel(sentinel)

        if mode is MaskMode.FIXED:
            if fixed_mask is None:
                raise ValueError("MaskMode.FIXED requires fixed_mask")
            fixed = _require_bool_mask(
                fixed_mask,
                name="fixed_mask",
                feature_dim=feature_dim,
            )
        else:
            if fixed_mask is not None:
                raise ValueError("fixed_mask is only valid for MaskMode.FIXED")
            fixed = None

        if mode is MaskMode.PERIODIC:
            if schedule is None:
                raise ValueError("MaskMode.PERIODIC requires a non-empty schedule")
            if type(schedule) is not tuple:
                raise TypeError("MaskMode.PERIODIC schedule must be a tuple")
            if not 0 < len(schedule) <= _INT32_MAX:
                raise ValueError("MaskMode.PERIODIC requires a non-empty int32 schedule")
            masks = tuple(
                _require_bool_mask(
                    mask,
                    name=f"schedule[{index}]",
                    feature_dim=feature_dim,
                )
                for index, mask in enumerate(schedule)
            )
            schedule_array = jnp.stack(masks, axis=0)
        else:
            if schedule is not None:
                raise ValueError("schedule is only valid for MaskMode.PERIODIC")
            schedule_array = None

        self._inner = inner
        self._mode = mode
        self._feature_dim = feature_dim
        self._fixed_mask = fixed
        self._mask_prob = probability
        self._schedule = schedule_array
        self._sentinel = sentinel_value

    @property
    def feature_dim(self) -> int:
        """Return the child observation dimension authenticated at construction."""

        return self._feature_dim

    @property
    def mode(self) -> MaskMode:
        """Return the masking mode."""

        return self._mode

    @property
    def resource_budget(self) -> PartialObservationResourceBudget:
        """Measure wrapper arrays and declare child arrays without owning them."""

        state = self.init(jr.key(0))
        wrapper_state_nbytes = (
            _array_tree_nbytes(state.key, name="partial-observation key")
            + _array_tree_nbytes(state.step_count, name="partial-observation telemetry")
            + _array_tree_nbytes(state.step_words, name="partial-observation words")
        )
        child_state_nbytes = _array_tree_nbytes(
            state.inner_state,
            name="partial-observation child state",
        )
        if self._fixed_mask is not None:
            mask_metadata_nbytes = _array_tree_nbytes(
                self._fixed_mask,
                name="partial-observation fixed mask",
            )
        elif self._schedule is not None:
            mask_metadata_nbytes = _array_tree_nbytes(
                self._schedule,
                name="partial-observation schedule",
            )
        else:
            mask_metadata_nbytes = 0
        schedule_length = 0 if self._schedule is None else int(self._schedule.shape[0])
        wrapper_owned_nbytes = wrapper_state_nbytes + mask_metadata_nbytes
        return PartialObservationResourceBudget(
            stream_type=type(self).__name__,
            mode=self._mode.value,
            feature_dim=self._feature_dim,
            schedule_length=schedule_length,
            wrapper_state_nbytes=wrapper_state_nbytes,
            mask_metadata_nbytes=mask_metadata_nbytes,
            child_state_nbytes=child_state_nbytes,
            wrapper_owned_nbytes=wrapper_owned_nbytes,
            composed_persistent_nbytes=wrapper_owned_nbytes + child_state_nbytes,
        )

    def to_config(self) -> dict[str, Any]:
        """Serialize wrapper configuration while declaring the external child."""

        inner_type = f"{type(self._inner).__module__}.{type(self._inner).__qualname__}"
        fixed_mask = (
            None
            if self._fixed_mask is None
            else [bool(value) for value in self._fixed_mask.tolist()]
        )
        schedule = (
            None
            if self._schedule is None
            else [
                [bool(value) for value in row]
                for row in self._schedule.tolist()
            ]
        )
        return {
            "type": type(self).__name__,
            "config_schema": PARTIAL_OBSERVATION_CONFIG_SCHEMA,
            "state_schema": PARTIAL_OBSERVATION_STATE_SCHEMA,
            "resource_schema": PARTIAL_OBSERVATION_RESOURCE_SCHEMA,
            "inner_stream_type": inner_type,
            "inner_stream_ownership": "external",
            "feature_dim": self._feature_dim,
            "mode": self._mode.value,
            "fixed_mask": fixed_mask,
            "mask_prob": self._mask_prob,
            "schedule": schedule,
            "sentinel": self._sentinel,
        }

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        inner: ScanStream[InnerStateT],
    ) -> PartialObservationWrapper[InnerStateT]:
        """Strictly reconstruct a wrapper around an explicitly supplied child."""

        expected = {
            "type",
            "config_schema",
            "state_schema",
            "resource_schema",
            "inner_stream_type",
            "inner_stream_ownership",
            "feature_dim",
            "mode",
            "fixed_mask",
            "mask_prob",
            "schedule",
            "sentinel",
        }
        fields = _require_exact_fields(
            config,
            expected,
            label="partial-observation config",
        )
        if fields.pop("type") != cls.__name__:
            raise ValueError("partial-observation config type is unsupported")
        if fields.pop("config_schema") != PARTIAL_OBSERVATION_CONFIG_SCHEMA:
            raise ValueError("partial-observation config schema is unsupported")
        if fields.pop("state_schema") != PARTIAL_OBSERVATION_STATE_SCHEMA:
            raise ValueError("partial-observation state schema is unsupported")
        if fields.pop("resource_schema") != PARTIAL_OBSERVATION_RESOURCE_SCHEMA:
            raise ValueError("partial-observation resource schema is unsupported")
        if fields.pop("inner_stream_ownership") != "external":
            raise ValueError("partial-observation inner stream ownership is unsupported")
        expected_inner_type = f"{type(inner).__module__}.{type(inner).__qualname__}"
        if fields.pop("inner_stream_type") != expected_inner_type:
            raise ValueError("partial-observation inner stream type does not match")
        feature_dim = fields.pop("feature_dim")
        if type(feature_dim) is not int or feature_dim != getattr(inner, "feature_dim", None):
            raise ValueError("partial-observation feature_dim does not match child")
        mode_value = fields.pop("mode")
        if type(mode_value) is not str:
            raise TypeError("partial-observation config mode must be a string")
        try:
            mode = MaskMode(mode_value)
        except ValueError as error:
            raise ValueError("partial-observation config mode is unsupported") from error

        fixed_payload = fields.pop("fixed_mask")
        if fixed_payload is None:
            fixed_mask = None
        else:
            if not isinstance(fixed_payload, list) or not all(
                type(value) is bool for value in fixed_payload
            ):
                raise TypeError("partial-observation fixed_mask config must be a bool list")
            fixed_mask = jnp.asarray(fixed_payload, dtype=jnp.bool_)

        schedule_payload = fields.pop("schedule")
        if schedule_payload is None:
            schedule = None
        else:
            if not isinstance(schedule_payload, list) or not all(
                isinstance(row, list) and all(type(value) is bool for value in row)
                for row in schedule_payload
            ):
                raise TypeError("partial-observation schedule config must be bool lists")
            schedule = tuple(jnp.asarray(row, dtype=jnp.bool_) for row in schedule_payload)
        return cls(
            inner=inner,
            mode=mode,
            fixed_mask=fixed_mask,
            schedule=schedule,
            **fields,
        )

    def _require_state_contract(
        self,
        state: PartialObservationState[InnerStateT],
    ) -> None:
        if not isinstance(state, PartialObservationState):
            raise TypeError("state must be a PartialObservationState")
        _require_prng_key(state.key, name="partial-observation key")
        _require_array_contract(
            state.step_count,
            name="partial-observation step_count",
            shape=(),
            dtype=jnp.dtype(jnp.int32),
        )
        _require_array_contract(
            state.step_words,
            name="partial-observation step_words",
            shape=(2,),
            dtype=jnp.dtype(jnp.uint32),
        )
        _require_child_array_pytree(
            state.inner_state,
            name="partial-observation child state",
        )

    def _child_state_valid(self, inner_state: InnerStateT) -> Bool[Array, ""]:
        valid = _tree_floating_arrays_finite(inner_state)
        validator = getattr(self._inner, "state_is_valid", None)
        if callable(validator):
            child_valid = jnp.asarray(validator(inner_state))
            if child_valid.shape != () or child_valid.dtype != jnp.dtype(jnp.bool_):
                raise TypeError("child state_is_valid must return a scalar bool array")
            valid = valid & child_valid
        return valid

    def state_is_valid(
        self,
        state: PartialObservationState[InnerStateT],
    ) -> Bool[Array, ""]:
        """Validate wrapper identity and only child invariants actually exposed."""

        self._require_state_contract(state)
        return _lifetime_counter_valid(
            state.step_words,
            state.step_count,
        ) & self._child_state_valid(state.inner_state)

    def init(self, key: Array) -> PartialObservationState[InnerStateT]:
        """Initialize independent child and wrapper RNG state."""

        _require_prng_key(key, name="partial-observation init key")
        key_inner, key_mask = jr.split(key)
        inner_state = self._inner.init(key_inner)
        _require_child_array_pytree(
            inner_state,
            name="partial-observation initialized child state",
        )
        return PartialObservationState(
            inner_state=inner_state,
            key=key_mask,
            step_count=jnp.asarray(0, dtype=jnp.int32),
            step_words=jnp.zeros((2,), dtype=jnp.uint32),
        )

    def step(
        self,
        state: PartialObservationState[InnerStateT],
        idx: Array,
    ) -> tuple[TimeStep, PartialObservationState[InnerStateT]]:
        """Preserve the historical tuple interface."""

        result = self.step_result(state, idx)
        return result.timestep, result.state

    def step_result(
        self,
        state: PartialObservationState[InnerStateT],
        idx: Array,
    ) -> PartialObservationStepResult[InnerStateT]:
        """Stage the child and atomically commit one wrapper-owned event."""

        self._require_state_contract(state)
        input_valid = _step_input_valid(idx)
        lifetime_counter_valid = _lifetime_counter_valid(
            state.step_words,
            state.step_count,
        )
        state_valid = self._child_state_valid(state.inner_state)
        proposed_words, lifetime_capacity_available = (
            _checked_lifetime_words_increment(state.step_words)
        )
        idx_array = jnp.asarray(idx)
        safe_idx = jnp.where(input_valid, idx_array, jnp.zeros_like(idx_array))
        child_output = self._inner.step(state.inner_state, safe_idx)
        if type(child_output) is not tuple or len(child_output) != 2:
            raise TypeError("child step must return exactly (TimeStep, state)")
        timestep, candidate_inner_state = child_output
        if not isinstance(timestep, TimeStep):
            raise TypeError("child step must return a TimeStep")
        _require_child_array_pytree(
            candidate_inner_state,
            name="partial-observation candidate child state",
        )
        _require_preserved_child_pytree(state.inner_state, candidate_inner_state)
        observation = timestep.observation
        target = timestep.target
        if getattr(observation, "shape", None) != (self._feature_dim,):
            raise ValueError(
                "child observation shape "
                f"{getattr(observation, 'shape', None)} != ({self._feature_dim},)"
            )
        observation_dtype = getattr(observation, "dtype", None)
        if observation_dtype is None or not jnp.issubdtype(observation_dtype, jnp.floating):
            raise TypeError("child observation must have a floating dtype")
        target_shape = getattr(target, "shape", None)
        if target_shape is None or len(target_shape) == 0 or math.prod(target_shape) == 0:
            raise ValueError("child target must be a non-empty array with rank at least one")
        target_dtype = getattr(target, "dtype", None)
        if target_dtype is None or not jnp.issubdtype(target_dtype, jnp.floating):
            raise TypeError("child target must have a floating dtype")

        new_key = state.key
        if self._mode is MaskMode.FIXED:
            assert self._fixed_mask is not None
            visibility_mask = self._fixed_mask
            schedule_index = jnp.asarray(-1, dtype=jnp.int32)
        elif self._mode is MaskMode.RANDOM:
            new_key, key_use = jr.split(state.key)
            visibility_mask = (
                jr.uniform(key_use, (self._feature_dim,), dtype=jnp.float32)
                < self._mask_prob
            )
            schedule_index = jnp.asarray(-1, dtype=jnp.int32)
        else:
            assert self._schedule is not None
            remainder = _lifetime_words_remainder(
                state.step_words,
                int(self._schedule.shape[0]),
            )
            schedule_index = remainder.astype(jnp.int32)
            visibility_mask = self._schedule[schedule_index]

        masked_observation = jnp.where(
            visibility_mask,
            observation,
            jnp.full_like(observation, self._sentinel),
        )
        candidate_state: PartialObservationState[InnerStateT] = PartialObservationState(
            inner_state=candidate_inner_state,
            key=new_key,
            step_count=_saturating_int32_counter_increment(state.step_count),
            step_words=proposed_words,
        )
        candidate_state_valid = self.state_is_valid(candidate_state)
        output_valid = (
            jnp.all(jnp.isfinite(observation))
            & jnp.all(jnp.isfinite(target))
            & jnp.all(jnp.isfinite(masked_observation))
        )
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
        committed_timestep = TimeStep(
            observation=jnp.where(
                update_applied,
                masked_observation,
                jnp.full_like(masked_observation, jnp.nan),
            ),
            target=jnp.where(
                update_applied,
                target,
                jnp.full_like(target, jnp.nan),
            ),
        )
        return PartialObservationStepResult(
            timestep=committed_timestep,
            state=new_state,
            visibility_mask=visibility_mask,
            pre_step_words=state.step_words,
            post_step_words=cast(Array, new_state.step_words),
            schedule_index=schedule_index,
            lifetime_counter_valid=lifetime_counter_valid,
            lifetime_capacity_available=lifetime_capacity_available,
            input_valid=input_valid,
            state_valid=state_valid,
            output_valid=output_valid,
            candidate_state_valid=candidate_state_valid,
            update_applied=update_applied,
            update_rejected=~update_applied,
        )


def migrate_legacy_partial_observation_state[InnerStateT](
    legacy_state: Any,
    *,
    wrapper: PartialObservationWrapper[InnerStateT],
) -> PartialObservationState[InnerStateT]:
    """Migrate only an unsaturated legacy periodic identity.

    The former FIXED and RANDOM states never advanced ``period_index``, so
    their event identity cannot be recovered from their wrapper-owned leaves.
    PERIODIC stored its event count until signed-int32 saturation/wrap; only a
    non-negative value below ``INT32_MAX`` is therefore representable.
    """

    if wrapper.mode is not MaskMode.PERIODIC:
        raise ValueError(
            "legacy FIXED/RANDOM wrapper event identity is not representable"
        )
    fields = _require_exact_fields(
        _host_state_fields(legacy_state),
        {"inner_state", "key", "period_index"},
        label="legacy partial-observation state",
    )
    period_index = fields["period_index"]
    _require_array_contract(
        period_index,
        name="legacy partial-observation period_index",
        shape=(),
        dtype=jnp.dtype(jnp.int32),
    )
    count = int(period_index)
    if count < 0:
        raise ValueError("negative legacy period_index indicates signed wrap")
    if count >= _INT32_MAX:
        raise ValueError("saturated legacy period_index is ambiguous")
    migrated: PartialObservationState[InnerStateT] = PartialObservationState(
        inner_state=fields["inner_state"],
        key=fields["key"],
        step_count=jnp.asarray(count, dtype=jnp.int32),
        step_words=jnp.asarray((0, count), dtype=jnp.uint32),
    )
    wrapper._require_state_contract(migrated)
    if not bool(jax.device_get(wrapper.state_is_valid(migrated))):
        raise ValueError("legacy partial-observation state is invalid")
    return migrated
