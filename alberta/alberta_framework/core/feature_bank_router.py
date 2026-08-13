"""Atomic routing for fixed-width dynamic pair-feature banks.

An integrated continual learner commonly deploys a fixed feature vector

``[stable base prefix | discovered pair-feature tail]``.

When the discovered bank is reordered or refreshed, every downstream linear
consumer must move the same columns by *descriptor identity*.  Updating only
one consumer, carrying by slot number, or accepting duplicate descriptors can
silently corrupt learned predictions.  This module provides one fixed-shape,
pure-JAX routing transaction over an arbitrary PyTree of downstream arrays.

Live descriptors are canonical integer pairs ``0 <= left < right < base_dim``.
Inactive slots are exactly ``(-1, -1)``.  Both the old and proposed banks must
contain unique live descriptors.  Invalid transactions fail closed: router
state, counters, and every consumer leaf are returned unchanged, while
machine-readable masks identify duplicate, noncanonical, and out-of-range
slots.

All consumer leaves must contain the full deployed feature width along their
declared feature axis.  The default is the last axis for every leaf.  For any
other layout, callers must provide a PyTree of integer axes with exactly the
same structure as the consumer PyTree; axes are never guessed from shapes.
"""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Bool, Int, UInt

CONFIG_SCHEMA_VERSION = "alberta.feature_bank_router.config.v2"
FEATURE_BANK_ROUTER_STATE_SCHEMA = "alberta.feature-bank-router-state.v2"
FEATURE_BANK_ROUTER_LIFETIME_COUNTER_NBYTES = 24
FEATURE_BANK_ROUTER_LIFETIME_COUNTER_DELTA_NBYTES = 16
INACTIVE_DESCRIPTOR = (-1, -1)
_INT32_MAX = 2**31 - 1
_UINT32_MAX = 2**32 - 1


@dataclasses.dataclass(frozen=True)
class FeatureBankRouterConfig:
    """Static shape and descriptor-domain contract for a router.

    ``base_dim`` is both the stable deployed prefix width and the exclusive
    upper bound for pair endpoints.  ``active_slots`` is the fixed capacity of
    the discovered tail; inactive capacity remains represented by ``(-1,-1)``.
    """

    base_dim: int
    active_slots: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.base_dim, bool)
            or not isinstance(self.base_dim, int)
            or self.base_dim < 2
        ):
            raise ValueError("base_dim must be an integer of at least 2")
        if (
            isinstance(self.active_slots, bool)
            or not isinstance(self.active_slots, int)
            or self.active_slots <= 0
        ):
            raise ValueError("active_slots must be a positive integer")

    @property
    def total_feature_dim(self) -> int:
        """Fixed deployed width of the stable prefix plus dynamic tail."""

        return self.base_dim + self.active_slots

    def to_config(self) -> dict[str, object]:
        """Return a strict JSON-compatible configuration record."""

        return {
            "type": "FeatureBankRouter",
            "schema_version": CONFIG_SCHEMA_VERSION,
            "state_schema": FEATURE_BANK_ROUTER_STATE_SCHEMA,
            "base_dim": self.base_dim,
            "active_slots": self.active_slots,
        }

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, object],
    ) -> FeatureBankRouterConfig:
        """Reconstruct only the exact versioned configuration schema."""

        expected_keys = {
            "type",
            "schema_version",
            "state_schema",
            "base_dim",
            "active_slots",
        }
        if set(config) != expected_keys:
            raise ValueError("feature-bank router config keys do not match the v2 schema")
        if config.get("type") != "FeatureBankRouter":
            raise ValueError("feature-bank router config type is invalid")
        if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
            raise ValueError("feature-bank router config schema version is unsupported")
        if config.get("state_schema") != FEATURE_BANK_ROUTER_STATE_SCHEMA:
            raise ValueError("feature-bank router state schema is unsupported")
        return cls(
            base_dim=config["base_dim"],  # type: ignore[arg-type]
            active_slots=config["active_slots"],  # type: ignore[arg-type]
        )


@functools.partial(
    jax.tree_util.register_dataclass,
    data_fields=(
        "descriptors",
        "route_count",
        "generation_count",
        "route_words",
        "generation_words",
    ),
    meta_fields=(),
)
@dataclasses.dataclass(frozen=True)
class FeatureBankRouterState:
    """Persistent descriptor identity and monotonic routing counters."""

    descriptors: Int[Array, "active_slots 2"]
    route_count: Int[Array, ""]
    generation_count: Int[Array, ""]
    route_words: UInt[Array, " 2"]
    generation_words: UInt[Array, " 2"]


@functools.partial(
    jax.tree_util.register_dataclass,
    data_fields=(
        "valid",
        "live_mask",
        "inactive_mask",
        "duplicate_mask",
        "noncanonical_mask",
        "out_of_range_mask",
    ),
    meta_fields=(),
)
@dataclasses.dataclass(frozen=True)
class PairDescriptorValidation:
    """Fixed-shape diagnostics for one proposed or deployed descriptor bank."""

    valid: Bool[Array, ""]
    live_mask: Bool[Array, " active_slots"]
    inactive_mask: Bool[Array, " active_slots"]
    duplicate_mask: Bool[Array, " active_slots"]
    noncanonical_mask: Bool[Array, " active_slots"]
    out_of_range_mask: Bool[Array, " active_slots"]


@functools.partial(
    jax.tree_util.register_dataclass,
    data_fields=(
        "valid",
        "route_applied",
        "carry_survivors",
        "counter_invalid",
        "descriptors_changed",
        "old_validation",
        "new_validation",
        "source_slots",
        "survivor_mask",
        "new_mask",
        "evicted_mask",
        "survivor_count",
        "new_count",
        "evicted_count",
        "old_live_count",
        "new_live_count",
        "route_count_before",
        "route_count_after",
        "generation_count_before",
        "generation_count_after",
        "route_words_before",
        "route_words_after",
        "generation_words_before",
        "generation_words_after",
        "route_counter_valid",
        "generation_counter_valid",
        "counter_order_valid",
        "lifetime_counter_valid",
        "route_capacity_available",
        "generation_capacity_available",
        "lifetime_capacity_available",
    ),
    meta_fields=(),
)
@dataclasses.dataclass(frozen=True)
class FeatureBankRouteDiagnostics:
    """Machine-readable outcome of one atomic route attempt."""

    valid: Bool[Array, ""]
    route_applied: Bool[Array, ""]
    carry_survivors: Bool[Array, ""]
    counter_invalid: Bool[Array, ""]
    descriptors_changed: Bool[Array, ""]
    old_validation: PairDescriptorValidation
    new_validation: PairDescriptorValidation
    source_slots: Int[Array, " active_slots"]
    survivor_mask: Bool[Array, " active_slots"]
    new_mask: Bool[Array, " active_slots"]
    evicted_mask: Bool[Array, " active_slots"]
    survivor_count: Int[Array, ""]
    new_count: Int[Array, ""]
    evicted_count: Int[Array, ""]
    old_live_count: Int[Array, ""]
    new_live_count: Int[Array, ""]
    route_count_before: Int[Array, ""]
    route_count_after: Int[Array, ""]
    generation_count_before: Int[Array, ""]
    generation_count_after: Int[Array, ""]
    route_words_before: UInt[Array, " 2"]
    route_words_after: UInt[Array, " 2"]
    generation_words_before: UInt[Array, " 2"]
    generation_words_after: UInt[Array, " 2"]
    route_counter_valid: Bool[Array, ""]
    generation_counter_valid: Bool[Array, ""]
    counter_order_valid: Bool[Array, ""]
    lifetime_counter_valid: Bool[Array, ""]
    route_capacity_available: Bool[Array, ""]
    generation_capacity_available: Bool[Array, ""]
    lifetime_capacity_available: Bool[Array, ""]


@functools.partial(
    jax.tree_util.register_dataclass,
    data_fields=("state", "consumers", "diagnostics"),
    meta_fields=(),
)
@dataclasses.dataclass(frozen=True)
class FeatureBankRouteResult:
    """New router state, atomically remapped consumer PyTree, and diagnostics."""

    state: FeatureBankRouterState
    consumers: Any
    diagnostics: FeatureBankRouteDiagnostics


@dataclasses.dataclass(frozen=True)
class FeatureBankRouterResourceBudget:
    """Exact persistent logical-scalar and byte accounting.

    Diagnostic and route-result buffers are transient and intentionally
    excluded.  Consumer bytes count the supplied managed PyTree exactly,
    including mixed dtypes and arbitrary leading dimensions.
    """

    base_feature_slots: int
    dynamic_feature_slots: int
    total_feature_slots: int
    descriptor_int32_scalars: int
    counter_int32_scalars: int
    counter_uint32_scalars: int
    router_state_scalars: int
    router_state_nbytes: int
    consumer_leaf_count: int
    consumer_feature_groups: int
    consumer_stable_prefix_scalars: int
    consumer_dynamic_tail_scalars: int
    consumer_total_scalars: int
    consumer_state_nbytes: int
    total_managed_nbytes: int

    def to_dict(self) -> dict[str, int]:
        """Return a JSON-compatible exact accounting record."""

        return dataclasses.asdict(self)


def _require_descriptor_contract(
    descriptors: Array,
    *,
    active_slots: int,
    location: str,
) -> Array:
    value = jnp.asarray(descriptors)
    if value.shape != (active_slots, 2):
        raise ValueError(f"{location} must have shape ({active_slots}, 2)")
    if value.dtype != jnp.dtype(jnp.int32):
        raise TypeError(f"{location} must have dtype int32")
    return value


def _descriptor_validation(
    descriptors: Array,
    *,
    base_dim: int,
) -> PairDescriptorValidation:
    left = descriptors[:, 0]
    right = descriptors[:, 1]
    inactive = (left == INACTIVE_DESCRIPTOR[0]) & (right == INACTIVE_DESCRIPTOR[1])
    in_range = (left >= 0) & (right >= 0) & (left < base_dim) & (right < base_dim)
    live = in_range & (left < right)
    noncanonical = in_range & ~(left < right)
    out_of_range = ~inactive & ~in_range

    equal_pairs = jnp.all(descriptors[:, None, :] == descriptors[None, :, :], axis=-1)
    distinct_slots = ~jnp.eye(descriptors.shape[0], dtype=jnp.bool_)
    duplicates = jnp.any(
        equal_pairs & distinct_slots & live[:, None] & live[None, :],
        axis=1,
    )
    valid = ~jnp.any(duplicates | noncanonical | out_of_range)
    return PairDescriptorValidation(
        valid=valid,
        live_mask=live,
        inactive_mask=inactive,
        duplicate_mask=duplicates,
        noncanonical_mask=noncanonical,
        out_of_range_mask=out_of_range,
    )


def _saturating_increment(value: Array) -> Array:
    return jnp.where(value < _INT32_MAX, value + jnp.int32(1), value)


def _checked_lifetime_words_increment(
    words: Array,
) -> tuple[UInt[Array, " 2"], Bool[Array, ""]]:
    """Propose one exact increment without wrapping the all-ones identity."""

    if getattr(words, "shape", None) != (2,):
        raise ValueError("feature-bank router lifetime words must have shape (2,)")
    if getattr(words, "dtype", None) != jnp.dtype(jnp.uint32):
        raise TypeError("feature-bank router lifetime words must have dtype uint32")
    maximum = jnp.asarray(_UINT32_MAX, dtype=jnp.uint32)
    capacity_available = ~jnp.all(words == maximum)
    one = jnp.asarray(1, dtype=jnp.uint32)
    low = words[1] + one
    carry = (low == jnp.asarray(0, dtype=jnp.uint32)).astype(jnp.uint32)
    proposed = jnp.stack((words[0] + carry, low))
    return (
        jnp.where(capacity_available, proposed, words).astype(jnp.uint32),
        capacity_available,
    )


def _lifetime_counter_valid(words: Array, telemetry: Array) -> Bool[Array, ""]:
    """Validate one exact identity against its saturating int32 telemetry."""

    if getattr(words, "shape", None) != (2,):
        raise ValueError("feature-bank router lifetime words must have shape (2,)")
    if getattr(words, "dtype", None) != jnp.dtype(jnp.uint32):
        raise TypeError("feature-bank router lifetime words must have dtype uint32")
    if getattr(telemetry, "shape", None) != ():
        raise ValueError("feature-bank router telemetry must be scalar")
    if getattr(telemetry, "dtype", None) != jnp.dtype(jnp.int32):
        raise TypeError("feature-bank router telemetry must have dtype int32")
    maximum_i32 = jnp.asarray(_INT32_MAX, dtype=jnp.int32)
    below_saturation = (words[0] == jnp.asarray(0, dtype=jnp.uint32)) & (
        words[1] < jnp.asarray(_INT32_MAX, dtype=jnp.uint32)
    )
    return (telemetry >= 0) & jnp.where(
        below_saturation,
        telemetry == words[1].astype(jnp.int32),
        telemetry == maximum_i32,
    )


def _lifetime_words_le(left: Array, right: Array) -> Bool[Array, ""]:
    """Compare two big-endian uint32 word identities without narrowing."""

    return (left[0] < right[0]) | ((left[0] == right[0]) & (left[1] <= right[1]))


class FeatureBankRouter:
    """Atomic fixed-shape router for every downstream feature consumer."""

    def __init__(self, config: FeatureBankRouterConfig):
        self._config = config

    @property
    def config(self) -> FeatureBankRouterConfig:
        """Static router contract."""

        return self._config

    def to_config(self) -> dict[str, object]:
        """Serialize the exact router configuration."""

        return self._config.to_config()

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, object],
    ) -> FeatureBankRouter:
        """Construct a router from the strict versioned configuration."""

        return cls(FeatureBankRouterConfig.from_config(config))

    def init(
        self,
        descriptors: Array | None = None,
    ) -> FeatureBankRouterState:
        """Initialize counters and an optional fixed-shape descriptor bank.

        Value-level descriptor validity is deliberately checked by
        :meth:`route`, where it can produce a JIT-compatible diagnostic.
        Shape and dtype violations are rejected immediately.
        """

        if descriptors is None:
            descriptor_array = jnp.tile(
                jnp.asarray(INACTIVE_DESCRIPTOR, dtype=jnp.int32),
                (self._config.active_slots, 1),
            )
        else:
            descriptor_array = _require_descriptor_contract(
                descriptors,
                active_slots=self._config.active_slots,
                location="descriptors",
            )
        return FeatureBankRouterState(
            descriptors=descriptor_array,
            route_count=jnp.array(0, dtype=jnp.int32),
            generation_count=jnp.array(0, dtype=jnp.int32),
            route_words=jnp.zeros((2,), dtype=jnp.uint32),
            generation_words=jnp.zeros((2,), dtype=jnp.uint32),
        )

    def validate_descriptors(
        self,
        descriptors: Array,
    ) -> PairDescriptorValidation:
        """Return explicit fixed-shape validity diagnostics for one bank."""

        descriptor_array = _require_descriptor_contract(
            descriptors,
            active_slots=self._config.active_slots,
            location="descriptors",
        )
        return _descriptor_validation(
            descriptor_array,
            base_dim=self._config.base_dim,
        )

    def _consumer_layout(
        self,
        consumers: Any,
        feature_axes: Any | None,
    ) -> tuple[list[Array], jax.tree_util.PyTreeDef, tuple[int, ...]]:
        leaves, tree_definition = jax.tree_util.tree_flatten(consumers)
        if not leaves:
            raise ValueError("consumers must contain at least one array leaf")
        if feature_axes is None:
            raw_axes = [-1] * len(leaves)
        else:
            raw_axes, axes_definition = jax.tree_util.tree_flatten(feature_axes)
            if axes_definition != tree_definition:  # type: ignore[operator]
                raise ValueError(
                    "feature_axes must have exactly the same PyTree structure as consumers"
                )
        arrays: list[Array] = []
        normalized_axes: list[int] = []
        for index, (leaf, raw_axis) in enumerate(zip(leaves, raw_axes, strict=True)):
            value = jnp.asarray(leaf)
            if isinstance(raw_axis, bool) or not isinstance(raw_axis, int):
                raise TypeError(f"feature axis for consumer leaf {index} must be an integer")
            axis = raw_axis if raw_axis >= 0 else value.ndim + raw_axis
            if axis < 0 or axis >= value.ndim:
                raise ValueError(f"feature axis for consumer leaf {index} is out of range")
            if value.shape[axis] != self._config.total_feature_dim:
                raise ValueError(
                    f"consumer leaf {index} feature axis must have width "
                    f"{self._config.total_feature_dim}"
                )
            arrays.append(value)
            normalized_axes.append(axis)
        return arrays, tree_definition, tuple(normalized_axes)

    def _require_state_contract(self, state: FeatureBankRouterState) -> None:
        _require_descriptor_contract(
            state.descriptors,
            active_slots=self._config.active_slots,
            location="state.descriptors",
        )
        for name, value in (
            ("state.route_count", state.route_count),
            ("state.generation_count", state.generation_count),
        ):
            array = jnp.asarray(value)
            if array.shape != ():
                raise ValueError(f"{name} must be scalar")
            if array.dtype != jnp.dtype(jnp.int32):
                raise TypeError(f"{name} must have dtype int32")
        for name, value in (
            ("state.route_words", state.route_words),
            ("state.generation_words", state.generation_words),
        ):
            array = jnp.asarray(value)
            if array.shape != (2,):
                raise ValueError(f"{name} must have shape (2,)")
            if array.dtype != jnp.dtype(jnp.uint32):
                raise TypeError(f"{name} must have dtype uint32")

    def route(
        self,
        state: FeatureBankRouterState,
        consumers: Any,
        new_descriptors: Array,
        *,
        feature_axes: Any | None = None,
        carry_survivors: bool = True,
    ) -> FeatureBankRouteResult:
        """Atomically route every consumer from the old bank to the new bank.

        Surviving live descriptors copy their old column by exact pair
        identity.  Newly born and inactive tail slots are zero.  Evicted
        columns disappear.  The stable prefix is copied without arithmetic.

        ``feature_axes`` is either ``None`` (last axis for every leaf) or a
        PyTree of Python integer axes matching ``consumers`` exactly.  Capture
        non-default axis trees as static closures when using :func:`jax.jit`.

        Setting ``carry_survivors=False`` is the matched no-carry ablation: the
        same valid descriptor transaction and counters are applied, but the
        entire discovered tail is zeroed for every consumer.
        """

        if not isinstance(carry_survivors, bool):
            raise TypeError("carry_survivors must be a Python boolean")
        self._require_state_contract(state)
        proposed = _require_descriptor_contract(
            new_descriptors,
            active_slots=self._config.active_slots,
            location="new_descriptors",
        )
        consumer_leaves, consumer_tree, axes = self._consumer_layout(
            consumers,
            feature_axes,
        )

        old_validation = _descriptor_validation(
            state.descriptors,
            base_dim=self._config.base_dim,
        )
        new_validation = _descriptor_validation(
            proposed,
            base_dim=self._config.base_dim,
        )
        proposed_route_words, route_capacity_available = (
            _checked_lifetime_words_increment(state.route_words)
        )
        proposed_generation_words, generation_capacity_available = (
            _checked_lifetime_words_increment(state.generation_words)
        )
        route_counter_valid = _lifetime_counter_valid(
            state.route_words,
            state.route_count,
        )
        generation_counter_valid = _lifetime_counter_valid(
            state.generation_words,
            state.generation_count,
        )
        counter_order_valid = _lifetime_words_le(
            state.generation_words,
            state.route_words,
        )
        lifetime_counter_valid = (
            route_counter_valid & generation_counter_valid & counter_order_valid
        )
        counter_invalid = ~lifetime_counter_valid
        raw_descriptors_changed = jnp.any(proposed != state.descriptors)
        generation_capacity_sufficient = ~raw_descriptors_changed | (
            generation_capacity_available
        )
        lifetime_capacity_available = (
            route_capacity_available & generation_capacity_sufficient
        )
        valid = (
            old_validation.valid
            & new_validation.valid
            & lifetime_counter_valid
            & lifetime_capacity_available
        )

        identity_match = jnp.all(
            proposed[:, None, :] == state.descriptors[None, :, :],
            axis=-1,
        )
        identity_match &= new_validation.live_mask[:, None] & old_validation.live_mask[None, :]
        raw_survivor_mask = new_validation.live_mask & jnp.any(identity_match, axis=1)
        raw_new_mask = new_validation.live_mask & ~raw_survivor_mask
        raw_evicted_mask = old_validation.live_mask & ~jnp.any(identity_match, axis=0)
        raw_source_slots = jnp.argmax(identity_match, axis=1).astype(jnp.int32)
        safe_source_slots = jnp.where(
            raw_survivor_mask,
            raw_source_slots,
            jnp.int32(0),
        )

        routed_leaves: list[Array] = []
        for leaf, axis in zip(consumer_leaves, axes, strict=True):
            prefix_index = [slice(None)] * leaf.ndim
            prefix_index[axis] = slice(0, self._config.base_dim)
            tail_index = [slice(None)] * leaf.ndim
            tail_index[axis] = slice(
                self._config.base_dim,
                self._config.total_feature_dim,
            )
            stable_prefix = leaf[tuple(prefix_index)]
            old_tail = leaf[tuple(tail_index)]
            gathered = jnp.take(old_tail, safe_source_slots, axis=axis)
            mask_shape = [1] * leaf.ndim
            mask_shape[axis] = self._config.active_slots
            survivor_mask = raw_survivor_mask.reshape(mask_shape)
            carried_tail = jnp.where(
                survivor_mask,
                gathered,
                jnp.zeros_like(gathered),
            )
            if not carry_survivors:
                carried_tail = jnp.zeros_like(carried_tail)
            candidate = jnp.concatenate((stable_prefix, carried_tail), axis=axis)
            routed_leaves.append(jnp.where(valid, candidate, leaf))
        routed_consumers = jax.tree_util.tree_unflatten(consumer_tree, routed_leaves)

        descriptors_changed = valid & raw_descriptors_changed
        candidate_route_count = _saturating_increment(state.route_count)
        candidate_generation_count = jnp.where(
            descriptors_changed,
            _saturating_increment(state.generation_count),
            state.generation_count,
        )
        next_state = FeatureBankRouterState(
            descriptors=jnp.where(valid, proposed, state.descriptors),
            route_count=jnp.where(valid, candidate_route_count, state.route_count),
            generation_count=jnp.where(
                valid,
                candidate_generation_count,
                state.generation_count,
            ),
            route_words=jnp.where(
                valid,
                proposed_route_words,
                state.route_words,
            ),
            generation_words=jnp.where(
                descriptors_changed,
                proposed_generation_words,
                state.generation_words,
            ),
        )

        survivor_mask = valid & raw_survivor_mask
        new_mask = valid & raw_new_mask
        evicted_mask = valid & raw_evicted_mask
        source_slots = jnp.where(
            survivor_mask,
            raw_source_slots,
            jnp.int32(-1),
        )
        diagnostics = FeatureBankRouteDiagnostics(
            valid=valid,
            route_applied=valid,
            carry_survivors=jnp.asarray(carry_survivors, dtype=jnp.bool_),
            counter_invalid=counter_invalid,
            descriptors_changed=descriptors_changed,
            old_validation=old_validation,
            new_validation=new_validation,
            source_slots=source_slots,
            survivor_mask=survivor_mask,
            new_mask=new_mask,
            evicted_mask=evicted_mask,
            survivor_count=jnp.sum(survivor_mask, dtype=jnp.int32),
            new_count=jnp.sum(new_mask, dtype=jnp.int32),
            evicted_count=jnp.sum(evicted_mask, dtype=jnp.int32),
            old_live_count=jnp.sum(old_validation.live_mask, dtype=jnp.int32),
            new_live_count=jnp.sum(new_validation.live_mask, dtype=jnp.int32),
            route_count_before=state.route_count,
            route_count_after=next_state.route_count,
            generation_count_before=state.generation_count,
            generation_count_after=next_state.generation_count,
            route_words_before=state.route_words,
            route_words_after=next_state.route_words,
            generation_words_before=state.generation_words,
            generation_words_after=next_state.generation_words,
            route_counter_valid=route_counter_valid,
            generation_counter_valid=generation_counter_valid,
            counter_order_valid=counter_order_valid,
            lifetime_counter_valid=lifetime_counter_valid,
            route_capacity_available=route_capacity_available,
            generation_capacity_available=generation_capacity_available,
            lifetime_capacity_available=lifetime_capacity_available,
        )
        return FeatureBankRouteResult(
            state=next_state,
            consumers=routed_consumers,
            diagnostics=diagnostics,
        )

    def resource_budget(
        self,
        state: FeatureBankRouterState,
        consumers: Any,
        *,
        feature_axes: Any | None = None,
    ) -> FeatureBankRouterResourceBudget:
        """Return exact persistent logical-scalar and byte accounting."""

        self._require_state_contract(state)
        leaves, _, axes = self._consumer_layout(consumers, feature_axes)
        total_scalars = 0
        total_bytes = 0
        feature_groups = 0
        for leaf, axis in zip(leaves, axes, strict=True):
            scalar_count = int(leaf.size)
            groups = scalar_count // int(leaf.shape[axis])
            total_scalars += scalar_count
            feature_groups += groups
            total_bytes += scalar_count * int(leaf.dtype.itemsize)

        descriptor_scalars = 2 * self._config.active_slots
        counter_int32_scalars = 2
        counter_uint32_scalars = 4
        router_state_scalars = (
            descriptor_scalars + counter_int32_scalars + counter_uint32_scalars
        )
        router_state_nbytes = 4 * router_state_scalars
        stable_prefix_scalars = feature_groups * self._config.base_dim
        dynamic_tail_scalars = feature_groups * self._config.active_slots
        return FeatureBankRouterResourceBudget(
            base_feature_slots=self._config.base_dim,
            dynamic_feature_slots=self._config.active_slots,
            total_feature_slots=self._config.total_feature_dim,
            descriptor_int32_scalars=descriptor_scalars,
            counter_int32_scalars=counter_int32_scalars,
            counter_uint32_scalars=counter_uint32_scalars,
            router_state_scalars=router_state_scalars,
            router_state_nbytes=router_state_nbytes,
            consumer_leaf_count=len(leaves),
            consumer_feature_groups=feature_groups,
            consumer_stable_prefix_scalars=stable_prefix_scalars,
            consumer_dynamic_tail_scalars=dynamic_tail_scalars,
            consumer_total_scalars=total_scalars,
            consumer_state_nbytes=total_bytes,
            total_managed_nbytes=router_state_nbytes + total_bytes,
        )


def feature_bank_router_lifetime_counter_nbytes() -> int:
    """Return bytes occupied by telemetry plus both exact router identities."""

    return FEATURE_BANK_ROUTER_LIFETIME_COUNTER_NBYTES


def measure_feature_bank_router_state_nbytes(state: FeatureBankRouterState) -> int:
    """Measure persistent JAX-array bytes in one router state."""

    return sum(
        int(leaf.size) * int(leaf.dtype.itemsize)
        for leaf in jax.tree.leaves(state)
        if isinstance(leaf, Array)
    )


def migrate_legacy_feature_bank_router_state(
    legacy_state: Any,
) -> FeatureBankRouterState:
    """Migrate a pre-v2 router state only when both int32 clocks are exact."""

    if isinstance(legacy_state, Mapping):
        fields = dict(legacy_state)
    elif dataclasses.is_dataclass(legacy_state) and not isinstance(legacy_state, type):
        fields = {
            field.name: getattr(legacy_state, field.name)
            for field in dataclasses.fields(legacy_state)
        }
    else:
        raise TypeError("legacy feature-bank router state must be a mapping or dataclass")
    current_names = {
        field.name
        for field in dataclasses.fields(FeatureBankRouterState)
    }
    legacy_names = current_names - {"route_words", "generation_words"}
    supplied_names = set(fields)
    if supplied_names != legacy_names:
        missing = sorted(legacy_names - supplied_names)
        extra = sorted(supplied_names - legacy_names)
        raise ValueError(
            "legacy feature-bank router field manifest is not exact; "
            f"missing={missing}, extra={extra}"
        )
    exact_counts: dict[str, int] = {}
    for name in ("route_count", "generation_count"):
        value = jnp.asarray(fields[name])
        if value.shape != () or value.dtype != jnp.dtype(jnp.int32):
            raise TypeError(f"legacy feature-bank router {name} must be scalar int32")
        count = int(value)
        if count < 0:
            raise ValueError(f"negative legacy feature-bank router {name} indicates wrap")
        if count >= _INT32_MAX:
            raise ValueError(
                f"saturated legacy feature-bank router {name} is ambiguous"
            )
        exact_counts[name] = count
    if exact_counts["generation_count"] > exact_counts["route_count"]:
        raise ValueError("legacy feature-bank router generation count exceeds route count")
    fields["route_words"] = jnp.asarray(
        (0, exact_counts["route_count"]),
        dtype=jnp.uint32,
    )
    fields["generation_words"] = jnp.asarray(
        (0, exact_counts["generation_count"]),
        dtype=jnp.uint32,
    )
    return FeatureBankRouterState(**fields)


__all__ = [
    "CONFIG_SCHEMA_VERSION",
    "FEATURE_BANK_ROUTER_LIFETIME_COUNTER_DELTA_NBYTES",
    "FEATURE_BANK_ROUTER_LIFETIME_COUNTER_NBYTES",
    "FEATURE_BANK_ROUTER_STATE_SCHEMA",
    "INACTIVE_DESCRIPTOR",
    "FeatureBankRouteDiagnostics",
    "FeatureBankRouteResult",
    "FeatureBankRouter",
    "FeatureBankRouterConfig",
    "FeatureBankRouterResourceBudget",
    "FeatureBankRouterState",
    "PairDescriptorValidation",
    "feature_bank_router_lifetime_counter_nbytes",
    "measure_feature_bank_router_state_nbytes",
    "migrate_legacy_feature_bank_router_state",
]
