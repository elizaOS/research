# mypy: disable-error-code="call-arg,name-defined"
"""Bounded, fail-closed experiential memory for continuing agents.

The memory stores fixed-width episodic exemplars and retrieves from the state
that existed before the current exemplar is written.  Retrieval keeps reward,
safety, uncertainty, reliability, and eviction utility as separate signals;
in particular, reward is never folded into either reliability or utility.

All persistent state is held in fixed-shape JAX arrays.  The implementation is
therefore compatible with ``jax.jit`` and ``jax.lax.scan`` and exposes exact
array-byte and entry-count accounting.

Chronology is authoritative in an exact two-word global transaction identity
and exact per-slot insertion/access timestamps.  Caller-supplied initial ages
are retained as int32-safe offsets.  The legacy ``ages`` and ``recency_ages``
arrays remain saturating diagnostics authenticated from those identities; no
freshness or eviction decision relies on saturated telemetry.

The design follows the episodic-memory lineage: stored transitions reused as
learning signal (Lin 1992) and similarity-weighted nearest-neighbor retrieval
over exemplar keys (Blundell et al. 2016), here with explicit reliability,
staleness, and safety gating instead of unconditional recall.

References:
    Lin (1992). "Self-Improving Reactive Agents Based on Reinforcement
        Learning, Planning and Teaching."
    Blundell et al. (2016). "Model-Free Episodic Control."
"""

from __future__ import annotations

import dataclasses
import functools
import math
import struct
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import chex
import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Bool, Float, Int, UInt

from alberta_framework.core.checkpoints import (
    load_checkpoint,
    load_checkpoint_metadata,
    save_checkpoint,
)

_INT32_MAX = 2_147_483_647
_UINT32_MAX = 4_294_967_295

EXPERIENTIAL_MEMORY_CONFIG_SCHEMA = "alberta.experiential-memory.config.v2"
EXPERIENTIAL_MEMORY_STATE_SCHEMA = "alberta.experiential-memory.state.v2"
EXPERIENTIAL_MEMORY_CHECKPOINT_SCHEMA = "alberta.experiential-memory.checkpoint.v2"
_LEGACY_EXPERIENTIAL_MEMORY_CHECKPOINT_SCHEMA = (
    "alberta.experiential-memory.checkpoint.v1"
)
EXPERIENTIAL_MEMORY_MECHANISM_STATUS = "development_mechanism_only"
EXPERIENTIAL_MEMORY_SCIENTIFIC_PROMOTION_ALLOWED = False
EXPERIENTIAL_MEMORY_EXACT_GLOBAL_STEP_IDENTITY_NBYTES = 8
EXPERIENTIAL_MEMORY_EXACT_SLOT_TEMPORAL_IDENTITY_NBYTES = 24


@dataclass(frozen=True)
class ExperientialMemoryConfig:
    """Static allocation and retrieval policy for experiential memory.

    Args:
        capacity: Maximum number of persistent exemplars.
        observation_dim: Width of the stored grounded observation.
        key_dim: Width of the retrieval key.
        action_dim: Width of the stored action.
        outcome_dim: Width of the stored outcome vector.
        top_k: Maximum neighbors considered by one query.
        min_neighbors: Minimum eligible neighbors required to retrieve.
        distance_scale: Positive scale for the RBF key similarity
            ``exp(-mean_squared_distance / distance_scale)``, where the
            distance is the per-dimension mean of squared key differences.
            It is therefore in squared key units and must be calibrated to
            the key representation's magnitude.
        min_similarity: Minimum RBF similarity for an eligible neighbor.
            With the defaults, a neighbor qualifies when its mean squared
            per-dimension key distance is at most
            ``distance_scale * ln(2) ~= 0.69``.
        min_effective_reliability: Minimum reliability after staleness decay.
        max_uncertainty: Maximum query and exemplar uncertainty.
        max_safety_cost: Maximum exemplar safety cost.
        max_age: Maximum chronological exemplar age.  This decision threshold
            is restricted to non-negative int32 so saturated age telemetry can
            never alias an eligible age; comparison itself uses exact clocks.
        staleness_scale: Exponential reliability-decay timescale.
        utility_decay: Per-step decay for explicit eviction utility.
        eviction_utility_weight: Utility contribution to retention priority.
        eviction_recency_weight: Recency contribution to retention priority.
        recency_scale: Scale of the reciprocal recency score.
    """

    capacity: int
    observation_dim: int
    key_dim: int
    action_dim: int
    outcome_dim: int
    top_k: int = 4
    min_neighbors: int = 1
    distance_scale: float = 1.0
    min_similarity: float = 0.5
    min_effective_reliability: float = 0.1
    max_uncertainty: float = 1.0
    max_safety_cost: float = 1.0
    max_age: int = 10_000
    staleness_scale: float = 1_000.0
    utility_decay: float = 0.999
    eviction_utility_weight: float = 1.0
    eviction_recency_weight: float = 1.0
    recency_scale: float = 100.0

    def to_config(self) -> dict[str, object]:
        """Return the strict v2 JSON-serializable configuration."""
        payload = asdict(self)
        payload["type"] = "ExperientialMemoryConfig"
        payload["schema"] = EXPERIENTIAL_MEMORY_CONFIG_SCHEMA
        return payload

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> ExperientialMemoryConfig:
        """Strictly reconstruct a v2 configuration dictionary."""
        if type(config) is not dict:
            raise TypeError("experiential-memory config must be an exact dict")
        payload = dict(config)
        expected = {
            field.name for field in dataclasses.fields(cls)
        } | {"type", "schema"}
        if set(payload) != expected:
            if "schema" not in payload:
                raise ValueError(
                    "legacy experiential-memory config requires explicit migration"
                )
            missing = sorted(expected - set(payload))
            extra = sorted(set(payload) - expected)
            raise ValueError(
                "experiential-memory config fields do not match v2; "
                f"missing={missing}, extra={extra}"
            )
        schema = payload.pop("schema")
        if schema != EXPERIENTIAL_MEMORY_CONFIG_SCHEMA:
            raise ValueError("experiential-memory config schema is unsupported")
        type_name = payload.pop("type")
        if type_name != "ExperientialMemoryConfig":
            raise ValueError(f"unexpected config type: {type_name!r}")
        result = cls(**payload)
        _validate_config(result)
        return result


@chex.dataclass(frozen=True)
class ExperientialMemoryEntry:
    """One typed experiential exemplar presented for bounded storage."""

    observation: Float[Array, " observation_dim"]
    key: Float[Array, " key_dim"]
    action: Float[Array, " action_dim"]
    outcome: Float[Array, " outcome_dim"]
    reward: Float[Array, ""]
    uncertainty: Float[Array, ""]
    uncertainty_available: Bool[Array, ""]
    safety_cost: Float[Array, ""]
    safety_cost_available: Bool[Array, ""]
    reliability: Float[Array, ""]
    utility: Float[Array, ""]
    utility_available: Bool[Array, ""]
    representation_version: Int[Array, ""]
    valid: Bool[Array, ""]
    age: Int[Array, ""]
    provenance_id: Int[Array, ""]
    source_id: Int[Array, ""]


@chex.dataclass(frozen=True)
class ExperientialMemoryEntries:
    """Fixed-capacity structure-of-arrays for persistent exemplars.

    ``ages`` and ``recency_ages`` are saturating int32 telemetry.  Exact
    decisions derive elapsed time from the timestamp/offset pairs.
    """

    observations: Float[Array, "capacity observation_dim"]
    keys: Float[Array, "capacity key_dim"]
    actions: Float[Array, "capacity action_dim"]
    outcomes: Float[Array, "capacity outcome_dim"]
    rewards: Float[Array, " capacity"]
    uncertainties: Float[Array, " capacity"]
    uncertainty_available: Bool[Array, " capacity"]
    safety_costs: Float[Array, " capacity"]
    safety_cost_available: Bool[Array, " capacity"]
    reliabilities: Float[Array, " capacity"]
    utilities: Float[Array, " capacity"]
    utility_available: Bool[Array, " capacity"]
    representation_versions: Int[Array, " capacity"]
    valid: Bool[Array, " capacity"]
    ages: Int[Array, " capacity"]
    recency_ages: Int[Array, " capacity"]
    insertion_step_words: UInt[Array, "capacity 2"]
    last_access_step_words: UInt[Array, "capacity 2"]
    insertion_age_offsets: Int[Array, " capacity"]
    last_access_age_offsets: Int[Array, " capacity"]
    provenance_ids: Int[Array, " capacity"]
    source_ids: Int[Array, " capacity"]
    retrieval_counts: Int[Array, " capacity"]


@chex.dataclass(frozen=True)
class ExperientialMemoryState:
    """Complete fixed-shape state with a finite 64-bit transaction identity."""

    entries: ExperientialMemoryEntries
    active_count: Int[Array, ""]
    step_count: Int[Array, ""]
    step_words: UInt[Array, " 2"]
    query_count: Int[Array, ""]
    accepted_query_count: Int[Array, ""]
    write_count: Int[Array, ""]
    rejected_write_count: Int[Array, ""]
    eviction_count: Int[Array, ""]
    persistent_bytes: Array


@chex.dataclass(frozen=True)
class ExperientialMemoryRetrieval:
    """A gated retrieval and its auditable neighbor diagnostics.

    Retrieved payloads are all-zero when ``accepted`` is false.  Neighbor
    diagnostics remain available so an evaluator can localize an abstention.
    """

    accepted: Bool[Array, ""]
    observation: Float[Array, " observation_dim"]
    action: Float[Array, " action_dim"]
    outcome: Float[Array, " outcome_dim"]
    reward: Float[Array, ""]
    uncertainty: Float[Array, ""]
    safety_cost: Float[Array, ""]
    effective_reliability: Float[Array, ""]
    neighbor_indices: Int[Array, " top_k"]
    neighbor_mask: Bool[Array, " top_k"]
    neighbor_weights: Float[Array, " top_k"]
    neighbor_similarities: Float[Array, " top_k"]
    neighbor_reliabilities: Float[Array, " top_k"]
    neighbor_ages: Int[Array, " top_k"]
    neighbor_provenance_ids: Int[Array, " top_k"]
    state_valid: Bool[Array, ""]
    query_valid: Bool[Array, ""]
    version_compatible: Bool[Array, ""]
    freshness_ok: Bool[Array, ""]
    uncertainty_available: Bool[Array, ""]
    safety_cost_available: Bool[Array, ""]
    uncertainty_ok: Bool[Array, ""]
    safety_ok: Bool[Array, ""]
    has_neighbors: Bool[Array, ""]


@chex.dataclass(frozen=True)
class ExperientialMemoryWriteResult:
    """State and accounting returned by one bounded write."""

    state: ExperientialMemoryState
    wrote: Bool[Array, ""]
    slot: Int[Array, ""]
    evicted: Bool[Array, ""]
    evicted_provenance_id: Int[Array, ""]


@chex.dataclass(frozen=True)
class ExperientialMemoryStepResult:
    """One causal query-before-write operation."""

    state: ExperientialMemoryState
    retrieval: ExperientialMemoryRetrieval
    wrote: Bool[Array, ""]
    slot: Int[Array, ""]
    evicted: Bool[Array, ""]
    evicted_provenance_id: Int[Array, ""]


@chex.dataclass(frozen=True)
class ExperientialMemoryAccounting:
    """Exact allocation with saturating int32 operation telemetry."""

    active_entries: Int[Array, ""]
    capacity_entries: Int[Array, ""]
    slot_bytes: Array
    persistent_bytes: Array
    step_words: UInt[Array, " 2"]
    queries: Int[Array, ""]
    accepted_queries: Int[Array, ""]
    writes: Int[Array, ""]
    rejected_writes: Int[Array, ""]
    evictions: Int[Array, ""]


@dataclass(frozen=True)
class ExperientialMemoryResourceBudget:
    """Exact fixed allocation and finite temporal-identity declaration."""

    capacity_entries: int
    slot_bytes: int
    persistent_state_bytes: int
    exact_global_step_identity_bytes: int
    exact_slot_temporal_identity_bytes: int
    lifetime_identity_bits: int
    age_telemetry_saturation: int
    operation_telemetry_saturation: int
    random_draws_per_query: int
    random_draws_per_write: int
    scientific_promotion_allowed: bool

    def to_config(self) -> dict[str, int | bool]:
        """Return the exact JSON-compatible resource declaration."""

        return dataclasses.asdict(self)


def _validate_finite(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a real number")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    try:
        float32_value = struct.unpack("!f", struct.pack("!f", value))[0]
    except OverflowError as error:
        raise ValueError(f"{name} must remain finite as float32") from error
    if not math.isfinite(float32_value) or (value != 0.0 and float32_value == 0.0):
        raise ValueError(f"{name} must remain finite and nonzero as float32")


def _validate_config(config: ExperientialMemoryConfig) -> None:
    for name in (
        "capacity",
        "observation_dim",
        "key_dim",
        "action_dim",
        "outcome_dim",
        "top_k",
        "min_neighbors",
    ):
        value = getattr(config, name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")
        if value < 1:
            raise ValueError(f"{name} must be positive")
    if config.top_k > config.capacity:
        raise ValueError("top_k must be <= capacity")
    if config.min_neighbors > config.top_k:
        raise ValueError("min_neighbors must be <= top_k")
    if isinstance(config.max_age, bool) or not isinstance(config.max_age, int):
        raise ValueError("max_age must be an integer")
    if not 0 <= config.max_age <= _INT32_MAX:
        raise ValueError("max_age must be in [0, 2147483647]")

    for name in (
        "distance_scale",
        "min_similarity",
        "min_effective_reliability",
        "max_uncertainty",
        "max_safety_cost",
        "staleness_scale",
        "utility_decay",
        "eviction_utility_weight",
        "eviction_recency_weight",
        "recency_scale",
    ):
        _validate_finite(name, cast(float, getattr(config, name)))

    if config.distance_scale <= 0.0:
        raise ValueError("distance_scale must be positive")
    if not 0.0 <= config.min_similarity <= 1.0:
        raise ValueError("min_similarity must be in [0, 1]")
    if not 0.0 < config.min_effective_reliability <= 1.0:
        raise ValueError("min_effective_reliability must be in (0, 1]")
    if config.max_uncertainty < 0.0:
        raise ValueError("max_uncertainty must be non-negative")
    if config.max_safety_cost < 0.0:
        raise ValueError("max_safety_cost must be non-negative")
    if config.staleness_scale <= 0.0:
        raise ValueError("staleness_scale must be positive")
    if not 0.0 <= config.utility_decay <= 1.0:
        raise ValueError("utility_decay must be in [0, 1]")
    if config.eviction_utility_weight < 0.0:
        raise ValueError("eviction_utility_weight must be non-negative")
    if config.eviction_recency_weight < 0.0:
        raise ValueError("eviction_recency_weight must be non-negative")
    if config.eviction_utility_weight + config.eviction_recency_weight <= 0.0:
        raise ValueError("at least one eviction retention weight must be positive")
    if config.recency_scale <= 0.0:
        raise ValueError("recency_scale must be positive")


def migrate_legacy_experiential_memory_config(
    legacy_config: Mapping[str, object],
) -> ExperientialMemoryConfig:
    """Explicitly stamp one exact pre-v2 construction configuration."""

    if not isinstance(legacy_config, Mapping):
        raise TypeError("legacy experiential-memory config must be a mapping")
    fields = dict(legacy_config)
    expected = {
        field.name for field in dataclasses.fields(ExperientialMemoryConfig)
    } | {"type"}
    if set(fields) != expected:
        missing = sorted(expected - set(fields))
        extra = sorted(set(fields) - expected)
        raise ValueError(
            "legacy experiential-memory config fields are not exact; "
            f"missing={missing}, extra={extra}"
        )
    fields["schema"] = EXPERIENTIAL_MEMORY_CONFIG_SCHEMA
    return ExperientialMemoryConfig.from_config(cast(dict[str, Any], fields))


def _saturating_increment(value: Array) -> Array:
    maximum_minus_one = jnp.asarray(_INT32_MAX - 1, dtype=jnp.int32)
    result: Array = jnp.minimum(value, maximum_minus_one) + jnp.asarray(
        1,
        dtype=jnp.int32,
    )
    return result


def _telemetry_from_words(words: Array) -> Array:
    """Project an exact two-word identity to saturating int32 telemetry."""

    below_saturation = (words[0] == jnp.asarray(0, dtype=jnp.uint32)) & (
        words[1] < jnp.asarray(_INT32_MAX, dtype=jnp.uint32)
    )
    result: Array = jnp.where(
        below_saturation,
        words[1].astype(jnp.int32),
        jnp.asarray(_INT32_MAX, dtype=jnp.int32),
    )
    return result


def _lifetime_counter_valid(words: Array, telemetry: Array) -> Array:
    """Authenticate saturating telemetry against an exact finite identity."""

    return (telemetry >= 0) & (telemetry == _telemetry_from_words(words))


def _checked_lifetime_words_increment(words: Array) -> tuple[Array, Array]:
    """Propose one exact increment without wrapping the all-ones terminal."""

    maximum = jnp.asarray(_UINT32_MAX, dtype=jnp.uint32)
    capacity_available = ~jnp.all(words == maximum)
    low = words[1] + jnp.asarray(1, dtype=jnp.uint32)
    carry = (low == jnp.asarray(0, dtype=jnp.uint32)).astype(jnp.uint32)
    proposed = jnp.stack((words[0] + carry, low)).astype(jnp.uint32)
    return jnp.where(capacity_available, proposed, words), capacity_available


def _timestamps_not_after(now_words: Array, timestamps: Array) -> Array:
    """Return a vectorized unsigned lexicographic ``timestamps <= now``."""

    return (timestamps[:, 0] < now_words[0]) | (
        (timestamps[:, 0] == now_words[0]) & (timestamps[:, 1] <= now_words[1])
    )


def _elapsed_words(now_words: Array, timestamps: Array) -> Array:
    """Return exact unsigned elapsed words for authenticated past timestamps."""

    low = now_words[1] - timestamps[:, 1]
    borrow = (now_words[1] < timestamps[:, 1]).astype(jnp.uint32)
    high = now_words[0] - timestamps[:, 0] - borrow
    result: Array = jnp.stack((high, low), axis=1).astype(jnp.uint32)
    return result


def _age_components(
    now_words: Array,
    timestamps: Array,
    age_offsets: Array,
) -> tuple[Array, Array, Array]:
    """Return an exact 65-bit age as overflow, high word, and low word."""

    elapsed = _elapsed_words(now_words, timestamps)
    offsets = age_offsets.astype(jnp.uint32)
    low = elapsed[:, 1] + offsets
    low_carry = (low < elapsed[:, 1]).astype(jnp.uint32)
    high = elapsed[:, 0] + low_carry
    overflow = high < elapsed[:, 0]
    return overflow, high, low


def _exact_age_not_greater(
    now_words: Array,
    left_timestamps: Array,
    left_offsets: Array,
    right_timestamps: Array,
    right_offsets: Array,
) -> Array:
    """Compare two exact elapsed-plus-offset ages without float conversion."""

    left_overflow, left_high, left_low = _age_components(
        now_words,
        left_timestamps,
        left_offsets,
    )
    right_overflow, right_high, right_low = _age_components(
        now_words,
        right_timestamps,
        right_offsets,
    )
    result: Array = (~left_overflow & right_overflow) | (
        (left_overflow == right_overflow)
        & (
            (left_high < right_high)
            | ((left_high == right_high) & (left_low <= right_low))
        )
    )
    return result


def _derived_age_telemetry(
    now_words: Array,
    timestamps: Array,
    age_offsets: Array,
) -> Array:
    """Derive capped int32 telemetry from exact elapsed time plus an offset."""

    elapsed = _elapsed_words(now_words, timestamps)
    offsets = age_offsets.astype(jnp.uint32)
    remaining = jnp.asarray(_INT32_MAX, dtype=jnp.uint32) - offsets
    representable = (elapsed[:, 0] == 0) & (elapsed[:, 1] <= remaining)
    exact_low = (elapsed[:, 1] + offsets).astype(jnp.int32)
    result: Array = jnp.where(
        representable,
        exact_low,
        jnp.asarray(_INT32_MAX, dtype=jnp.int32),
    )
    return result


def _exact_age_at_most(
    now_words: Array,
    timestamps: Array,
    age_offsets: Array,
    maximum_age: int,
) -> Array:
    """Compare exact ages to an int32-safe configured threshold."""

    elapsed = _elapsed_words(now_words, timestamps)
    maximum = jnp.asarray(maximum_age, dtype=jnp.uint32)
    offsets = age_offsets.astype(jnp.uint32)
    offset_fits = offsets <= maximum
    remaining = maximum - offsets
    result: Array = offset_fits & (elapsed[:, 0] == 0) & (
        elapsed[:, 1] <= remaining
    )
    return result


def _exact_age_float32(
    now_words: Array,
    timestamps: Array,
    age_offsets: Array,
) -> Array:
    """Round exact bounded ages to float32 only at score computation."""

    elapsed = _elapsed_words(now_words, timestamps)
    result: Array = (
        elapsed[:, 0].astype(jnp.float32)
        * jnp.asarray(4_294_967_296.0, dtype=jnp.float32)
        + elapsed[:, 1].astype(jnp.float32)
        + age_offsets.astype(jnp.float32)
    )
    return result


def _tree_nbytes(tree: Any) -> int:
    return sum(int(leaf.size) * int(leaf.dtype.itemsize) for leaf in jax.tree.leaves(tree))


def _configured_nbytes(config: ExperientialMemoryConfig) -> tuple[int, int]:
    vector_values = config.observation_dim + config.key_dim + config.action_dim + config.outcome_dim
    # Five float scalars, eight int32 scalars, four uint32 clock words,
    # and four bools per slot.
    slot_bytes = 4 * (vector_values + 5 + 8 + 4) + 4
    # Seven int32 counters, two uint32 step words, and one uint32 byte scalar.
    persistent_bytes = config.capacity * slot_bytes + 10 * 4
    return persistent_bytes, slot_bytes


def _legacy_configured_nbytes(config: ExperientialMemoryConfig) -> tuple[int, int]:
    """Return the exact pre-v2 allocation for migration authentication."""

    vector_values = config.observation_dim + config.key_dim + config.action_dim + config.outcome_dim
    slot_bytes = 4 * (vector_values + 5 + 6) + 4
    persistent_bytes = config.capacity * slot_bytes + 8 * 4
    return persistent_bytes, slot_bytes


def _require_array(
    value: Any,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: Any,
) -> None:
    """Reject structural drift before eager execution or JAX tracing."""
    if not hasattr(value, "shape") or not hasattr(value, "dtype"):
        raise TypeError(f"{name} must be an array with shape and dtype metadata")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}; got {tuple(value.shape)}")
    expected_dtype = jnp.dtype(dtype)
    actual_dtype = jnp.dtype(value.dtype)
    if actual_dtype != expected_dtype:
        raise TypeError(f"{name} must have dtype {expected_dtype}; got {actual_dtype}")


class ExperientialMemory:
    """Fixed-capacity episodic memory with conservative retrieval.

    ``query`` is read-only.  ``step`` first performs that query against the
    previous state, records any accepted access, advances ages once, and only
    then writes the supplied exemplar.  This ordering prevents target leakage
    from a current transition into its own prediction.  ``write`` and ``step``
    are exact no-ops once the two-word global identity reaches all ones.
    """

    def __init__(self, config: ExperientialMemoryConfig):
        _validate_config(config)
        self._config = config
        persistent_bytes, slot_bytes = _configured_nbytes(config)
        if persistent_bytes > _UINT32_MAX:
            raise ValueError("persistent memory allocation exceeds uint32 byte accounting")
        self._persistent_bytes = persistent_bytes
        self._slot_bytes = slot_bytes

    @property
    def config(self) -> ExperientialMemoryConfig:
        """Memory configuration."""
        return self._config

    @property
    def persistent_bytes(self) -> int:
        """Exact bytes occupied by all persistent JAX array leaves."""
        return self._persistent_bytes

    @property
    def slot_bytes(self) -> int:
        """Exact bytes occupied by one fixed-capacity entry slot."""
        return self._slot_bytes

    def to_config(self) -> dict[str, object]:
        """Serialize the exact v2 memory construction."""
        return {
            "schema": EXPERIENTIAL_MEMORY_CONFIG_SCHEMA,
            "type": "ExperientialMemory",
            "state_schema": EXPERIENTIAL_MEMORY_STATE_SCHEMA,
            "config": self._config.to_config(),
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> ExperientialMemory:
        """Strictly reconstruct an exact v2 memory construction."""
        if type(config) is not dict:
            raise TypeError("experiential-memory construction must be an exact dict")
        expected = {"schema", "type", "state_schema", "config"}
        if set(config) != expected:
            if "schema" not in config:
                raise ValueError(
                    "legacy experiential-memory construction requires explicit migration"
                )
            raise ValueError("experiential-memory construction fields do not match v2")
        if config["schema"] != EXPERIENTIAL_MEMORY_CONFIG_SCHEMA:
            raise ValueError("experiential-memory construction schema is unsupported")
        if config["type"] != "ExperientialMemory":
            raise ValueError(f"unexpected memory type: {config['type']!r}")
        if config["state_schema"] != EXPERIENTIAL_MEMORY_STATE_SCHEMA:
            raise ValueError("experiential-memory state schema is unsupported")
        inner = config["config"]
        if type(inner) is not dict:
            raise ValueError("experiential-memory inner config must be an exact dict")
        return cls(ExperientialMemoryConfig.from_config(inner))

    def state_valid(self, state: ExperientialMemoryState) -> Bool[Array, ""]:
        """Return the complete exact-clock invariant after structural checks."""

        self._validate_state_static_contract(state)
        return cast(Bool[Array, ""], self._state_valid_jit(state))

    @functools.partial(jax.jit, static_argnums=(0,))
    def _state_valid_jit(self, state: ExperientialMemoryState) -> Array:
        return self._state_is_valid(state)

    def _validate_state_static_contract(self, state: ExperientialMemoryState) -> None:
        """Validate every persistent leaf without reshaping or lossy casting."""
        if not isinstance(state, ExperientialMemoryState):
            raise TypeError("state must be an ExperientialMemoryState")
        if not isinstance(state.entries, ExperientialMemoryEntries):
            raise TypeError("state.entries must be ExperientialMemoryEntries")
        cfg = self._config
        entries = state.entries
        float_fields = {
            "observations": (cfg.capacity, cfg.observation_dim),
            "keys": (cfg.capacity, cfg.key_dim),
            "actions": (cfg.capacity, cfg.action_dim),
            "outcomes": (cfg.capacity, cfg.outcome_dim),
            "rewards": (cfg.capacity,),
            "uncertainties": (cfg.capacity,),
            "safety_costs": (cfg.capacity,),
            "reliabilities": (cfg.capacity,),
            "utilities": (cfg.capacity,),
        }
        for field_name, shape in float_fields.items():
            _require_array(
                getattr(entries, field_name),
                name=f"state.entries.{field_name}",
                shape=shape,
                dtype=jnp.float32,
            )
        for field_name in (
            "uncertainty_available",
            "safety_cost_available",
            "utility_available",
            "valid",
        ):
            _require_array(
                getattr(entries, field_name),
                name=f"state.entries.{field_name}",
                shape=(cfg.capacity,),
                dtype=jnp.bool_,
            )
        for field_name in (
            "representation_versions",
            "ages",
            "recency_ages",
            "insertion_age_offsets",
            "last_access_age_offsets",
            "provenance_ids",
            "source_ids",
            "retrieval_counts",
        ):
            _require_array(
                getattr(entries, field_name),
                name=f"state.entries.{field_name}",
                shape=(cfg.capacity,),
                dtype=jnp.int32,
            )
        for field_name in ("insertion_step_words", "last_access_step_words"):
            _require_array(
                getattr(entries, field_name),
                name=f"state.entries.{field_name}",
                shape=(cfg.capacity, 2),
                dtype=jnp.uint32,
            )
        for field_name in (
            "active_count",
            "step_count",
            "query_count",
            "accepted_query_count",
            "write_count",
            "rejected_write_count",
            "eviction_count",
        ):
            _require_array(
                getattr(state, field_name),
                name=f"state.{field_name}",
                shape=(),
                dtype=jnp.int32,
            )
        _require_array(
            state.step_words,
            name="state.step_words",
            shape=(2,),
            dtype=jnp.uint32,
        )
        _require_array(
            state.persistent_bytes,
            name="state.persistent_bytes",
            shape=(),
            dtype=jnp.uint32,
        )

    def _validate_entry_static_contract(self, entry: ExperientialMemoryEntry) -> None:
        """Reject same-size wrong shapes and dtype aliases at the write boundary."""
        if not isinstance(entry, ExperientialMemoryEntry):
            raise TypeError("entry must be an ExperientialMemoryEntry")
        cfg = self._config
        for field_name, shape in {
            "observation": (cfg.observation_dim,),
            "key": (cfg.key_dim,),
            "action": (cfg.action_dim,),
            "outcome": (cfg.outcome_dim,),
            "reward": (),
            "uncertainty": (),
            "safety_cost": (),
            "reliability": (),
            "utility": (),
        }.items():
            _require_array(
                getattr(entry, field_name),
                name=f"entry.{field_name}",
                shape=shape,
                dtype=jnp.float32,
            )
        for field_name in (
            "uncertainty_available",
            "safety_cost_available",
            "utility_available",
            "valid",
        ):
            _require_array(
                getattr(entry, field_name),
                name=f"entry.{field_name}",
                shape=(),
                dtype=jnp.bool_,
            )
        for field_name in (
            "representation_version",
            "age",
            "provenance_id",
            "source_id",
        ):
            _require_array(
                getattr(entry, field_name),
                name=f"entry.{field_name}",
                shape=(),
                dtype=jnp.int32,
            )

    def _state_is_valid(self, state: ExperientialMemoryState) -> Bool[Array, ""]:
        """Return the global dynamic invariant for one structurally valid state."""
        entries = state.entries
        valid = entries.valid
        all_float_payload_finite = (
            jnp.all(jnp.isfinite(entries.observations))
            & jnp.all(jnp.isfinite(entries.keys))
            & jnp.all(jnp.isfinite(entries.actions))
            & jnp.all(jnp.isfinite(entries.outcomes))
            & jnp.all(jnp.isfinite(entries.rewards))
            & jnp.all(jnp.isfinite(entries.uncertainties))
            & jnp.all(jnp.isfinite(entries.safety_costs))
            & jnp.all(jnp.isfinite(entries.reliabilities))
            & jnp.all(jnp.isfinite(entries.utilities))
        )
        scalar_ranges_valid = (
            jnp.all(entries.uncertainties >= 0.0)
            & jnp.all(entries.safety_costs >= 0.0)
            & jnp.all((entries.reliabilities >= 0.0) & (entries.reliabilities <= 1.0))
            & jnp.all(entries.utilities >= 0.0)
            & jnp.all(entries.ages >= 0)
            & jnp.all(entries.recency_ages >= 0)
            & jnp.all(entries.insertion_age_offsets >= 0)
            & jnp.all(entries.last_access_age_offsets >= 0)
            & jnp.all(entries.retrieval_counts >= 0)
        )
        insertion_times_valid = _timestamps_not_after(
            state.step_words,
            entries.insertion_step_words,
        )
        access_times_valid = _timestamps_not_after(
            state.step_words,
            entries.last_access_step_words,
        )
        access_not_before_insertion = (
            entries.last_access_step_words[:, 0]
            > entries.insertion_step_words[:, 0]
        ) | (
            (
                entries.last_access_step_words[:, 0]
                == entries.insertion_step_words[:, 0]
            )
            & (
                entries.last_access_step_words[:, 1]
                >= entries.insertion_step_words[:, 1]
            )
        )
        recency_not_greater_than_age = _exact_age_not_greater(
            state.step_words,
            entries.last_access_step_words,
            entries.last_access_age_offsets,
            entries.insertion_step_words,
            entries.insertion_age_offsets,
        )
        expected_ages = _derived_age_telemetry(
            state.step_words,
            entries.insertion_step_words,
            entries.insertion_age_offsets,
        )
        expected_recency_ages = _derived_age_telemetry(
            state.step_words,
            entries.last_access_step_words,
            entries.last_access_age_offsets,
        )
        exact_temporal_metadata_valid = jnp.all(
            (~valid)
            | (
                insertion_times_valid
                & access_times_valid
                & access_not_before_insertion
                & recency_not_greater_than_age
                & (entries.ages == expected_ages)
                & (entries.recency_ages == expected_recency_ages)
            )
        )
        availability_honest = (
            jnp.all(entries.uncertainty_available | (entries.uncertainties == 0.0))
            & jnp.all(entries.safety_cost_available | (entries.safety_costs == 0.0))
            & jnp.all(entries.utility_available | (entries.utilities == 0.0))
        )
        active_metadata_valid = jnp.all(
            (~valid)
            | (
                (entries.representation_versions >= 0)
                & (entries.provenance_ids >= 0)
                & (entries.source_ids >= 0)
            )
        )
        inactive_metadata_valid = jnp.all(
            valid
            | (
                (entries.representation_versions == -1)
                & (entries.provenance_ids == -1)
                & (entries.source_ids == -1)
                & (~entries.uncertainty_available)
                & (~entries.safety_cost_available)
                & (~entries.utility_available)
                & (entries.ages == 0)
                & (entries.recency_ages == 0)
                & jnp.all(entries.insertion_step_words == 0, axis=1)
                & jnp.all(entries.last_access_step_words == 0, axis=1)
                & (entries.insertion_age_offsets == 0)
                & (entries.last_access_age_offsets == 0)
                & (entries.retrieval_counts == 0)
            )
        )
        active_count = jnp.sum(valid.astype(jnp.int32))
        counters_nonnegative = (
            (state.active_count >= 0)
            & _lifetime_counter_valid(state.step_words, state.step_count)
            & (state.query_count >= 0)
            & (state.accepted_query_count >= 0)
            & (state.write_count >= 0)
            & (state.rejected_write_count >= 0)
            & (state.eviction_count >= 0)
        )
        counter_relations_valid = (
            (state.active_count == active_count)
            & (state.active_count <= self._config.capacity)
            & (state.accepted_query_count <= state.query_count)
            & (state.eviction_count <= state.write_count)
            & (state.write_count >= state.active_count)
        )
        result: Bool[Array, ""] = (
            all_float_payload_finite
            & scalar_ranges_valid
            & availability_honest
            & exact_temporal_metadata_valid
            & active_metadata_valid
            & inactive_metadata_valid
            & counters_nonnegative
            & counter_relations_valid
            & (
                state.persistent_bytes
                == jnp.asarray(self._persistent_bytes, dtype=jnp.uint32)
            )
        )
        return result

    def _make_initial_state(self, persistent_bytes: int) -> ExperientialMemoryState:
        cfg = self._config
        zeros = functools.partial(jnp.zeros, dtype=jnp.float32)
        entries = ExperientialMemoryEntries(
            observations=zeros((cfg.capacity, cfg.observation_dim)),
            keys=zeros((cfg.capacity, cfg.key_dim)),
            actions=zeros((cfg.capacity, cfg.action_dim)),
            outcomes=zeros((cfg.capacity, cfg.outcome_dim)),
            rewards=zeros((cfg.capacity,)),
            uncertainties=zeros((cfg.capacity,)),
            uncertainty_available=jnp.zeros((cfg.capacity,), dtype=jnp.bool_),
            safety_costs=zeros((cfg.capacity,)),
            safety_cost_available=jnp.zeros((cfg.capacity,), dtype=jnp.bool_),
            reliabilities=zeros((cfg.capacity,)),
            utilities=zeros((cfg.capacity,)),
            utility_available=jnp.zeros((cfg.capacity,), dtype=jnp.bool_),
            representation_versions=jnp.full((cfg.capacity,), -1, dtype=jnp.int32),
            valid=jnp.zeros((cfg.capacity,), dtype=jnp.bool_),
            ages=jnp.zeros((cfg.capacity,), dtype=jnp.int32),
            recency_ages=jnp.zeros((cfg.capacity,), dtype=jnp.int32),
            insertion_step_words=jnp.zeros((cfg.capacity, 2), dtype=jnp.uint32),
            last_access_step_words=jnp.zeros((cfg.capacity, 2), dtype=jnp.uint32),
            insertion_age_offsets=jnp.zeros((cfg.capacity,), dtype=jnp.int32),
            last_access_age_offsets=jnp.zeros((cfg.capacity,), dtype=jnp.int32),
            provenance_ids=jnp.full((cfg.capacity,), -1, dtype=jnp.int32),
            source_ids=jnp.full((cfg.capacity,), -1, dtype=jnp.int32),
            retrieval_counts=jnp.zeros((cfg.capacity,), dtype=jnp.int32),
        )
        zero = jnp.asarray(0, dtype=jnp.int32)
        return ExperientialMemoryState(
            entries=entries,
            active_count=zero,
            step_count=zero,
            step_words=jnp.zeros((2,), dtype=jnp.uint32),
            query_count=zero,
            accepted_query_count=zero,
            write_count=zero,
            rejected_write_count=zero,
            eviction_count=zero,
            persistent_bytes=jnp.asarray(persistent_bytes, dtype=jnp.uint32),
        )

    def init(self) -> ExperientialMemoryState:
        """Return an empty, fixed-shape memory state."""
        state = self._make_initial_state(persistent_bytes=self._persistent_bytes)
        if _tree_nbytes(state) != self._persistent_bytes:
            raise RuntimeError("persistent byte accounting disagrees with allocated state")
        return state

    def _canonical_entry(self, entry: ExperientialMemoryEntry) -> ExperientialMemoryEntry:
        cfg = self._config
        return ExperientialMemoryEntry(
            observation=jnp.asarray(entry.observation, dtype=jnp.float32).reshape(
                (cfg.observation_dim,)
            ),
            key=jnp.asarray(entry.key, dtype=jnp.float32).reshape((cfg.key_dim,)),
            action=jnp.asarray(entry.action, dtype=jnp.float32).reshape((cfg.action_dim,)),
            outcome=jnp.asarray(entry.outcome, dtype=jnp.float32).reshape((cfg.outcome_dim,)),
            reward=jnp.asarray(entry.reward, dtype=jnp.float32).reshape(()),
            uncertainty=jnp.asarray(entry.uncertainty, dtype=jnp.float32).reshape(()),
            uncertainty_available=jnp.asarray(
                entry.uncertainty_available, dtype=jnp.bool_
            ).reshape(()),
            safety_cost=jnp.asarray(entry.safety_cost, dtype=jnp.float32).reshape(()),
            safety_cost_available=jnp.asarray(
                entry.safety_cost_available, dtype=jnp.bool_
            ).reshape(()),
            reliability=jnp.asarray(entry.reliability, dtype=jnp.float32).reshape(()),
            utility=jnp.asarray(entry.utility, dtype=jnp.float32).reshape(()),
            utility_available=jnp.asarray(entry.utility_available, dtype=jnp.bool_).reshape(()),
            representation_version=jnp.asarray(
                entry.representation_version, dtype=jnp.int32
            ).reshape(()),
            valid=jnp.asarray(entry.valid, dtype=jnp.bool_).reshape(()),
            age=jnp.asarray(entry.age, dtype=jnp.int32).reshape(()),
            provenance_id=jnp.asarray(entry.provenance_id, dtype=jnp.int32).reshape(()),
            source_id=jnp.asarray(entry.source_id, dtype=jnp.int32).reshape(()),
        )

    @staticmethod
    def _entry_is_valid(entry: ExperientialMemoryEntry) -> Array:
        finite_payload = (
            jnp.all(jnp.isfinite(entry.observation))
            & jnp.all(jnp.isfinite(entry.key))
            & jnp.all(jnp.isfinite(entry.action))
            & jnp.all(jnp.isfinite(entry.outcome))
            & jnp.isfinite(entry.reward)
            & jnp.isfinite(entry.uncertainty)
            & jnp.isfinite(entry.safety_cost)
            & jnp.isfinite(entry.reliability)
            & jnp.isfinite(entry.utility)
        )
        valid_metadata = (
            (entry.uncertainty >= 0.0)
            & (entry.uncertainty_available | (entry.uncertainty == 0.0))
            & (entry.safety_cost >= 0.0)
            & (entry.safety_cost_available | (entry.safety_cost == 0.0))
            & (entry.reliability >= 0.0)
            & (entry.reliability <= 1.0)
            & (entry.utility >= 0.0)
            & (entry.utility_available | (entry.utility == 0.0))
            & (entry.representation_version >= 0)
            & (entry.age >= 0)
            & (entry.provenance_id >= 0)
            & (entry.source_id >= 0)
        )
        result: Array = entry.valid & finite_payload & valid_metadata
        return result

    def query(
        self,
        state: ExperientialMemoryState,
        key: Float[Array, " key_dim"],
        representation_version: Int[Array, ""],
        query_uncertainty: Float[Array, ""],
        query_uncertainty_available: Bool[Array, ""],
    ) -> ExperientialMemoryRetrieval:
        """Retrieve an eligible neighborhood without mutating persistent state.

        ``query_uncertainty_available`` is explicit: an unavailable estimate is
        not interchangeable with a measured zero.  Such a query abstains.
        """
        self._validate_state_static_contract(state)
        _require_array(key, name="key", shape=(self._config.key_dim,), dtype=jnp.float32)
        _require_array(
            representation_version,
            name="representation_version",
            shape=(),
            dtype=jnp.int32,
        )
        _require_array(
            query_uncertainty,
            name="query_uncertainty",
            shape=(),
            dtype=jnp.float32,
        )
        _require_array(
            query_uncertainty_available,
            name="query_uncertainty_available",
            shape=(),
            dtype=jnp.bool_,
        )
        return cast(
            ExperientialMemoryRetrieval,
            self._query_jit(
                state,
                key,
                representation_version,
                query_uncertainty,
                query_uncertainty_available,
            ),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def _query_jit(
        self,
        state: ExperientialMemoryState,
        key: Array,
        representation_version: Array,
        query_uncertainty: Array,
        query_uncertainty_available: Array,
    ) -> ExperientialMemoryRetrieval:
        """Compiled query after the public structural contract is established."""
        cfg = self._config
        entries = state.entries
        query_key = jnp.asarray(key, dtype=jnp.float32)
        query_version = jnp.asarray(representation_version, dtype=jnp.int32)
        query_uncertainty_value = jnp.asarray(query_uncertainty, dtype=jnp.float32)
        query_uncertainty_is_available = jnp.asarray(
            query_uncertainty_available, dtype=jnp.bool_
        )
        state_valid = self._state_is_valid(state)

        query_valid = (
            jnp.all(jnp.isfinite(query_key))
            & jnp.isfinite(query_uncertainty_value)
            & (query_uncertainty_value >= 0.0)
            & (query_version >= 0)
            & query_uncertainty_is_available
        )

        finite_rows = (
            jnp.all(jnp.isfinite(entries.observations), axis=1)
            & jnp.all(jnp.isfinite(entries.keys), axis=1)
            & jnp.all(jnp.isfinite(entries.actions), axis=1)
            & jnp.all(jnp.isfinite(entries.outcomes), axis=1)
            & jnp.isfinite(entries.rewards)
            & jnp.isfinite(entries.uncertainties)
            & jnp.isfinite(entries.safety_costs)
            & jnp.isfinite(entries.reliabilities)
            & jnp.isfinite(entries.utilities)
        )
        sane_rows = (
            entries.valid
            & finite_rows
            & (entries.uncertainties >= 0.0)
            & (entries.safety_costs >= 0.0)
            & (entries.reliabilities >= 0.0)
            & (entries.reliabilities <= 1.0)
            & (entries.utilities >= 0.0)
            & (entries.representation_versions >= 0)
            & (entries.ages >= 0)
            & (entries.recency_ages >= 0)
            & (entries.provenance_ids >= 0)
            & (entries.source_ids >= 0)
            & (entries.retrieval_counts >= 0)
        )
        same_version = sane_rows & (entries.representation_versions == query_version)
        exact_fresh = _exact_age_at_most(
            state.step_words,
            entries.insertion_step_words,
            entries.insertion_age_offsets,
            cfg.max_age,
        )
        fresh = same_version & exact_fresh
        uncertainty_available = fresh & entries.uncertainty_available
        safety_cost_available = fresh & entries.safety_cost_available
        uncertainty_eligible = uncertainty_available & (
            entries.uncertainties <= cfg.max_uncertainty
        )
        safety_eligible = safety_cost_available & (
            entries.safety_costs <= cfg.max_safety_cost
        )

        safe_query_key = jnp.where(jnp.isfinite(query_key), query_key, 0.0)
        safe_keys = jnp.where(finite_rows[:, None], entries.keys, 0.0)
        squared_distance = jnp.mean(
            (safe_keys - safe_query_key[None, :]) ** 2,
            axis=1,
        )
        similarities = jnp.exp(
            -squared_distance / jnp.asarray(cfg.distance_scale, dtype=jnp.float32)
        )
        safe_ages = jnp.where(
            entries.ages >= 0,
            _exact_age_float32(
                state.step_words,
                entries.insertion_step_words,
                entries.insertion_age_offsets,
            ),
            0.0,
        )
        staleness = jnp.exp(
            -safe_ages
            / jnp.asarray(cfg.staleness_scale, dtype=jnp.float32)
        )
        safe_reliabilities = jnp.where(
            jnp.isfinite(entries.reliabilities)
            & (entries.reliabilities >= 0.0)
            & (entries.reliabilities <= 1.0),
            entries.reliabilities,
            0.0,
        )
        effective_reliabilities = safe_reliabilities * staleness
        eligible = (
            uncertainty_eligible
            & safety_eligible
            & (similarities >= cfg.min_similarity)
            & (effective_reliabilities >= cfg.min_effective_reliability)
        )
        scores = jnp.where(eligible, similarities * effective_reliabilities, -jnp.inf)
        top_scores, indices = jax.lax.top_k(scores, cfg.top_k)
        neighbor_mask = jnp.isfinite(top_scores) & (top_scores > 0.0)
        positive_scores = jnp.where(neighbor_mask, top_scores, 0.0)
        score_sum = jnp.sum(positive_scores)
        neighbor_weights = positive_scores / jnp.maximum(score_sum, 1.0e-12)

        neighbor_similarities = similarities[indices]
        neighbor_reliabilities = effective_reliabilities[indices]
        neighbor_ages = entries.ages[indices]
        neighbor_provenance_ids = entries.provenance_ids[indices]

        safe_observations = jnp.where(
            jnp.isfinite(entries.observations), entries.observations, 0.0
        )
        safe_actions = jnp.where(jnp.isfinite(entries.actions), entries.actions, 0.0)
        safe_outcomes = jnp.where(jnp.isfinite(entries.outcomes), entries.outcomes, 0.0)
        safe_rewards = jnp.where(jnp.isfinite(entries.rewards), entries.rewards, 0.0)
        safe_uncertainties = jnp.where(
            jnp.isfinite(entries.uncertainties), entries.uncertainties, 0.0
        )
        safe_safety_costs = jnp.where(
            jnp.isfinite(entries.safety_costs), entries.safety_costs, 0.0
        )
        weighted_observation = jnp.sum(
            neighbor_weights[:, None] * safe_observations[indices], axis=0
        )
        weighted_action = jnp.sum(
            neighbor_weights[:, None] * safe_actions[indices], axis=0
        )
        weighted_outcome = jnp.sum(
            neighbor_weights[:, None] * safe_outcomes[indices], axis=0
        )
        weighted_reward = jnp.sum(neighbor_weights * safe_rewards[indices])
        weighted_uncertainty = jnp.sum(
            neighbor_weights * safe_uncertainties[indices]
        )
        weighted_safety_cost = jnp.sum(
            neighbor_weights * safe_safety_costs[indices]
        )
        weighted_reliability = jnp.sum(neighbor_weights * effective_reliabilities[indices])

        has_neighbors = jnp.sum(neighbor_mask.astype(jnp.int32)) >= cfg.min_neighbors
        version_compatible = jnp.any(same_version)
        freshness_ok = jnp.any(fresh)
        uncertainty_is_available = (
            state_valid
            & query_uncertainty_is_available
            & jnp.any(uncertainty_available)
        )
        safety_is_available = state_valid & jnp.any(safety_cost_available)
        uncertainty_ok = (
            uncertainty_is_available
            & (query_uncertainty_value <= cfg.max_uncertainty)
            & jnp.any(uncertainty_eligible)
        )
        safety_ok = jnp.any(safety_eligible)
        aggregate_ok = (
            (weighted_uncertainty <= cfg.max_uncertainty)
            & (weighted_safety_cost <= cfg.max_safety_cost)
            & jnp.isfinite(weighted_reliability)
        )
        accepted = (
            state_valid
            & query_valid
            & version_compatible
            & freshness_ok
            & uncertainty_ok
            & safety_ok
            & has_neighbors
            & aggregate_ok
        )

        def gated(value: Array) -> Array:
            result: Array = jnp.where(accepted, value, jnp.zeros_like(value))
            return result

        return ExperientialMemoryRetrieval(
            accepted=accepted,
            observation=gated(weighted_observation),
            action=gated(weighted_action),
            outcome=gated(weighted_outcome),
            reward=gated(weighted_reward),
            uncertainty=gated(weighted_uncertainty),
            safety_cost=gated(weighted_safety_cost),
            effective_reliability=gated(weighted_reliability),
            neighbor_indices=indices.astype(jnp.int32),
            neighbor_mask=neighbor_mask,
            neighbor_weights=neighbor_weights,
            neighbor_similarities=neighbor_similarities,
            neighbor_reliabilities=neighbor_reliabilities,
            neighbor_ages=neighbor_ages,
            neighbor_provenance_ids=neighbor_provenance_ids,
            state_valid=state_valid,
            query_valid=query_valid,
            version_compatible=version_compatible,
            freshness_ok=freshness_ok,
            uncertainty_available=uncertainty_is_available,
            safety_cost_available=safety_is_available,
            uncertainty_ok=uncertainty_ok,
            safety_ok=safety_ok,
            has_neighbors=has_neighbors,
        )

    def _advance(self, state: ExperientialMemoryState) -> ExperientialMemoryState:
        entries = state.entries
        valid = entries.valid
        next_step_words, _ = _checked_lifetime_words_increment(state.step_words)
        ages = jnp.where(
            valid,
            _derived_age_telemetry(
                next_step_words,
                entries.insertion_step_words,
                entries.insertion_age_offsets,
            ),
            entries.ages,
        )
        recency_ages = jnp.where(
            valid,
            _derived_age_telemetry(
                next_step_words,
                entries.last_access_step_words,
                entries.last_access_age_offsets,
            ),
            entries.recency_ages,
        )
        utilities = jnp.where(
            valid,
            entries.utilities * jnp.asarray(self._config.utility_decay, dtype=jnp.float32),
            entries.utilities,
        )
        return ExperientialMemoryState(
            entries=ExperientialMemoryEntries(
                observations=entries.observations,
                keys=entries.keys,
                actions=entries.actions,
                outcomes=entries.outcomes,
                rewards=entries.rewards,
                uncertainties=entries.uncertainties,
                uncertainty_available=entries.uncertainty_available,
                safety_costs=entries.safety_costs,
                safety_cost_available=entries.safety_cost_available,
                reliabilities=entries.reliabilities,
                utilities=utilities,
                utility_available=entries.utility_available,
                representation_versions=entries.representation_versions,
                valid=entries.valid,
                ages=ages,
                recency_ages=recency_ages,
                insertion_step_words=entries.insertion_step_words,
                last_access_step_words=entries.last_access_step_words,
                insertion_age_offsets=entries.insertion_age_offsets,
                last_access_age_offsets=entries.last_access_age_offsets,
                provenance_ids=entries.provenance_ids,
                source_ids=entries.source_ids,
                retrieval_counts=entries.retrieval_counts,
            ),
            active_count=state.active_count,
            step_count=_telemetry_from_words(next_step_words),
            step_words=next_step_words,
            query_count=state.query_count,
            accepted_query_count=state.accepted_query_count,
            write_count=state.write_count,
            rejected_write_count=state.rejected_write_count,
            eviction_count=state.eviction_count,
            persistent_bytes=state.persistent_bytes,
        )

    @staticmethod
    def _record_query(
        state: ExperientialMemoryState,
        retrieval: ExperientialMemoryRetrieval,
    ) -> ExperientialMemoryState:
        entries = state.entries
        access_increments = (
            jnp.zeros_like(entries.retrieval_counts)
            .at[retrieval.neighbor_indices]
            .add(retrieval.neighbor_mask.astype(jnp.int32))
        )
        access_mask = access_increments > 0
        accepted_access = retrieval.accepted & access_mask
        retrieval_counts = jnp.where(
            accepted_access,
            _saturating_increment(entries.retrieval_counts),
            entries.retrieval_counts,
        )
        recency_ages = jnp.where(accepted_access, 0, entries.recency_ages)
        last_access_step_words = jnp.where(
            accepted_access[:, None],
            state.step_words[None, :],
            entries.last_access_step_words,
        )
        last_access_age_offsets = jnp.where(
            accepted_access,
            jnp.asarray(0, dtype=jnp.int32),
            entries.last_access_age_offsets,
        )
        return ExperientialMemoryState(
            entries=ExperientialMemoryEntries(
                observations=entries.observations,
                keys=entries.keys,
                actions=entries.actions,
                outcomes=entries.outcomes,
                rewards=entries.rewards,
                uncertainties=entries.uncertainties,
                uncertainty_available=entries.uncertainty_available,
                safety_costs=entries.safety_costs,
                safety_cost_available=entries.safety_cost_available,
                reliabilities=entries.reliabilities,
                utilities=entries.utilities,
                utility_available=entries.utility_available,
                representation_versions=entries.representation_versions,
                valid=entries.valid,
                ages=entries.ages,
                recency_ages=recency_ages,
                insertion_step_words=entries.insertion_step_words,
                last_access_step_words=last_access_step_words,
                insertion_age_offsets=entries.insertion_age_offsets,
                last_access_age_offsets=last_access_age_offsets,
                provenance_ids=entries.provenance_ids,
                source_ids=entries.source_ids,
                retrieval_counts=retrieval_counts,
            ),
            active_count=state.active_count,
            step_count=state.step_count,
            step_words=state.step_words,
            query_count=_saturating_increment(state.query_count),
            accepted_query_count=jnp.where(
                retrieval.accepted,
                _saturating_increment(state.accepted_query_count),
                state.accepted_query_count,
            ),
            write_count=state.write_count,
            rejected_write_count=state.rejected_write_count,
            eviction_count=state.eviction_count,
            persistent_bytes=state.persistent_bytes,
        )

    def _write_advanced(
        self,
        state: ExperientialMemoryState,
        raw_entry: ExperientialMemoryEntry,
    ) -> ExperientialMemoryWriteResult:
        entry = self._canonical_entry(raw_entry)
        can_write = self._entry_is_valid(entry)
        cfg = self._config

        def do_write(
            current: ExperientialMemoryState,
        ) -> tuple[ExperientialMemoryState, Array, Array, Array]:
            current_entries = current.entries
            has_empty = jnp.any(~current_entries.valid)
            empty_slot = jnp.argmax((~current_entries.valid).astype(jnp.int32))
            exact_recency_ages = _exact_age_float32(
                current.step_words,
                current_entries.last_access_step_words,
                current_entries.last_access_age_offsets,
            )
            recency_score = 1.0 / (
                1.0
                + exact_recency_ages
                / jnp.asarray(cfg.recency_scale, dtype=jnp.float32)
            )
            eviction_utilities = jnp.where(
                current_entries.utility_available,
                current_entries.utilities,
                0.0,
            )
            retention_score = (
                jnp.asarray(cfg.eviction_utility_weight, dtype=jnp.float32)
                * eviction_utilities
                + jnp.asarray(cfg.eviction_recency_weight, dtype=jnp.float32) * recency_score
            )
            retention_score = jnp.where(current_entries.valid, retention_score, jnp.inf)
            eviction_slot = jnp.argmin(retention_score)
            slot = jnp.where(has_empty, empty_slot, eviction_slot).astype(jnp.int32)
            evicted = ~has_empty
            evicted_provenance_id = jnp.where(
                evicted, current_entries.provenance_ids[slot], -1
            ).astype(jnp.int32)

            next_entries = ExperientialMemoryEntries(
                observations=current_entries.observations.at[slot].set(entry.observation),
                keys=current_entries.keys.at[slot].set(entry.key),
                actions=current_entries.actions.at[slot].set(entry.action),
                outcomes=current_entries.outcomes.at[slot].set(entry.outcome),
                rewards=current_entries.rewards.at[slot].set(entry.reward),
                uncertainties=current_entries.uncertainties.at[slot].set(entry.uncertainty),
                uncertainty_available=current_entries.uncertainty_available.at[slot].set(
                    entry.uncertainty_available
                ),
                safety_costs=current_entries.safety_costs.at[slot].set(entry.safety_cost),
                safety_cost_available=current_entries.safety_cost_available.at[slot].set(
                    entry.safety_cost_available
                ),
                reliabilities=current_entries.reliabilities.at[slot].set(entry.reliability),
                utilities=current_entries.utilities.at[slot].set(entry.utility),
                utility_available=current_entries.utility_available.at[slot].set(
                    entry.utility_available
                ),
                representation_versions=current_entries.representation_versions.at[slot].set(
                    entry.representation_version
                ),
                valid=current_entries.valid.at[slot].set(True),
                ages=current_entries.ages.at[slot].set(entry.age),
                recency_ages=current_entries.recency_ages.at[slot].set(entry.age),
                insertion_step_words=current_entries.insertion_step_words.at[slot].set(
                    current.step_words
                ),
                last_access_step_words=current_entries.last_access_step_words.at[slot].set(
                    current.step_words
                ),
                insertion_age_offsets=current_entries.insertion_age_offsets.at[slot].set(
                    entry.age
                ),
                last_access_age_offsets=current_entries.last_access_age_offsets.at[slot].set(
                    entry.age
                ),
                provenance_ids=current_entries.provenance_ids.at[slot].set(entry.provenance_id),
                source_ids=current_entries.source_ids.at[slot].set(entry.source_id),
                retrieval_counts=current_entries.retrieval_counts.at[slot].set(0),
            )
            next_state = ExperientialMemoryState(
                entries=next_entries,
                active_count=jnp.where(
                    has_empty,
                    _saturating_increment(current.active_count),
                    current.active_count,
                ),
                step_count=current.step_count,
                step_words=current.step_words,
                query_count=current.query_count,
                accepted_query_count=current.accepted_query_count,
                write_count=_saturating_increment(current.write_count),
                rejected_write_count=current.rejected_write_count,
                eviction_count=jnp.where(
                    evicted,
                    _saturating_increment(current.eviction_count),
                    current.eviction_count,
                ),
                persistent_bytes=current.persistent_bytes,
            )
            return next_state, slot, evicted, evicted_provenance_id

        def reject_write(
            current: ExperientialMemoryState,
        ) -> tuple[ExperientialMemoryState, Array, Array, Array]:
            rejected = ExperientialMemoryState(
                entries=current.entries,
                active_count=current.active_count,
                step_count=current.step_count,
                step_words=current.step_words,
                query_count=current.query_count,
                accepted_query_count=current.accepted_query_count,
                write_count=current.write_count,
                rejected_write_count=_saturating_increment(current.rejected_write_count),
                eviction_count=current.eviction_count,
                persistent_bytes=current.persistent_bytes,
            )
            return (
                rejected,
                jnp.asarray(-1, dtype=jnp.int32),
                jnp.asarray(False),
                jnp.asarray(-1, dtype=jnp.int32),
            )

        next_state, slot, evicted, evicted_provenance_id = jax.lax.cond(
            can_write, do_write, reject_write, state
        )
        return ExperientialMemoryWriteResult(
            state=next_state,
            wrote=can_write,
            slot=slot,
            evicted=evicted,
            evicted_provenance_id=evicted_provenance_id,
        )

    def write(
        self,
        state: ExperientialMemoryState,
        entry: ExperientialMemoryEntry,
    ) -> ExperientialMemoryWriteResult:
        """Advance time once and attempt one bounded exemplar write.

        A dynamically corrupt state is an exact no-op rather than a substrate
        into which new data is partially written.
        """
        self._validate_state_static_contract(state)
        self._validate_entry_static_contract(entry)
        return cast(ExperientialMemoryWriteResult, self._write_jit(state, entry))

    @functools.partial(jax.jit, static_argnums=(0,))
    def _write_jit(
        self,
        state: ExperientialMemoryState,
        entry: ExperientialMemoryEntry,
    ) -> ExperientialMemoryWriteResult:
        def apply(_: None) -> ExperientialMemoryWriteResult:
            candidate = self._write_advanced(self._advance(state), entry)

            def commit(_: None) -> ExperientialMemoryWriteResult:
                return candidate

            def roll_back(_: None) -> ExperientialMemoryWriteResult:
                return ExperientialMemoryWriteResult(
                    state=state,
                    wrote=jnp.asarray(False),
                    slot=jnp.asarray(-1, dtype=jnp.int32),
                    evicted=jnp.asarray(False),
                    evicted_provenance_id=jnp.asarray(-1, dtype=jnp.int32),
                )

            return cast(
                ExperientialMemoryWriteResult,
                jax.lax.cond(
                    self._state_is_valid(candidate.state),
                    commit,
                    roll_back,
                    operand=None,
                ),
            )

        def reject(_: None) -> ExperientialMemoryWriteResult:
            return ExperientialMemoryWriteResult(
                state=state,
                wrote=jnp.asarray(False),
                slot=jnp.asarray(-1, dtype=jnp.int32),
                evicted=jnp.asarray(False),
                evicted_provenance_id=jnp.asarray(-1, dtype=jnp.int32),
            )

        _, lifetime_available = _checked_lifetime_words_increment(state.step_words)
        return cast(
            ExperientialMemoryWriteResult,
            jax.lax.cond(
                self._state_is_valid(state) & lifetime_available,
                apply,
                reject,
                operand=None,
            ),
        )

    def step(
        self,
        state: ExperientialMemoryState,
        query_key: Float[Array, " key_dim"],
        representation_version: Int[Array, ""],
        query_uncertainty: Float[Array, ""],
        query_uncertainty_available: Bool[Array, ""],
        entry: ExperientialMemoryEntry,
    ) -> ExperientialMemoryStepResult:
        """Query the pre-write state, then age/access/write exactly once."""
        self._validate_state_static_contract(state)
        self._validate_entry_static_contract(entry)
        _require_array(
            query_key,
            name="query_key",
            shape=(self._config.key_dim,),
            dtype=jnp.float32,
        )
        _require_array(
            representation_version,
            name="representation_version",
            shape=(),
            dtype=jnp.int32,
        )
        _require_array(
            query_uncertainty,
            name="query_uncertainty",
            shape=(),
            dtype=jnp.float32,
        )
        _require_array(
            query_uncertainty_available,
            name="query_uncertainty_available",
            shape=(),
            dtype=jnp.bool_,
        )
        return cast(
            ExperientialMemoryStepResult,
            self._step_jit(
                state,
                query_key,
                representation_version,
                query_uncertainty,
                query_uncertainty_available,
                entry,
            ),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def _step_jit(
        self,
        state: ExperientialMemoryState,
        query_key: Array,
        representation_version: Array,
        query_uncertainty: Array,
        query_uncertainty_available: Array,
        entry: ExperientialMemoryEntry,
    ) -> ExperientialMemoryStepResult:
        retrieval = self._query_jit(
            state,
            query_key,
            representation_version,
            query_uncertainty,
            query_uncertainty_available,
        )

        def apply(_: None) -> ExperientialMemoryStepResult:
            advanced = self._advance(state)
            accessed = self._record_query(advanced, retrieval)
            write_result = self._write_advanced(accessed, entry)
            candidate = ExperientialMemoryStepResult(
                state=write_result.state,
                retrieval=retrieval,
                wrote=write_result.wrote,
                slot=write_result.slot,
                evicted=write_result.evicted,
                evicted_provenance_id=write_result.evicted_provenance_id,
            )

            def commit(_: None) -> ExperientialMemoryStepResult:
                return candidate

            def roll_back(_: None) -> ExperientialMemoryStepResult:
                return ExperientialMemoryStepResult(
                    state=state,
                    retrieval=retrieval,
                    wrote=jnp.asarray(False),
                    slot=jnp.asarray(-1, dtype=jnp.int32),
                    evicted=jnp.asarray(False),
                    evicted_provenance_id=jnp.asarray(-1, dtype=jnp.int32),
                )

            return cast(
                ExperientialMemoryStepResult,
                jax.lax.cond(
                    self._state_is_valid(candidate.state),
                    commit,
                    roll_back,
                    operand=None,
                ),
            )

        def reject(_: None) -> ExperientialMemoryStepResult:
            return ExperientialMemoryStepResult(
                state=state,
                retrieval=retrieval,
                wrote=jnp.asarray(False),
                slot=jnp.asarray(-1, dtype=jnp.int32),
                evicted=jnp.asarray(False),
                evicted_provenance_id=jnp.asarray(-1, dtype=jnp.int32),
            )

        _, lifetime_available = _checked_lifetime_words_increment(state.step_words)
        return cast(
            ExperientialMemoryStepResult,
            jax.lax.cond(
                self._state_is_valid(state) & lifetime_available,
                apply,
                reject,
                operand=None,
            ),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def accounting(
        self,
        state: ExperientialMemoryState,
    ) -> ExperientialMemoryAccounting:
        """Return exact capacity, byte, and lifetime-operation accounting."""
        return ExperientialMemoryAccounting(
            active_entries=state.active_count,
            capacity_entries=jnp.asarray(self._config.capacity, dtype=jnp.int32),
            slot_bytes=jnp.asarray(self._slot_bytes, dtype=jnp.uint32),
            persistent_bytes=state.persistent_bytes,
            step_words=state.step_words,
            queries=state.query_count,
            accepted_queries=state.accepted_query_count,
            writes=state.write_count,
            rejected_writes=state.rejected_write_count,
            evictions=state.eviction_count,
        )

    def resource_budget(
        self,
        state: ExperientialMemoryState | None = None,
    ) -> ExperientialMemoryResourceBudget:
        """Return the exact fixed allocation and finite clock declaration."""

        if state is not None:
            self._validate_state_static_contract(state)
            if not bool(jax.device_get(self._state_is_valid(state))):
                raise ValueError("experiential-memory state is invalid")
        return ExperientialMemoryResourceBudget(
            capacity_entries=self._config.capacity,
            slot_bytes=self._slot_bytes,
            persistent_state_bytes=self._persistent_bytes,
            exact_global_step_identity_bytes=(
                EXPERIENTIAL_MEMORY_EXACT_GLOBAL_STEP_IDENTITY_NBYTES
            ),
            exact_slot_temporal_identity_bytes=(
                EXPERIENTIAL_MEMORY_EXACT_SLOT_TEMPORAL_IDENTITY_NBYTES
                * self._config.capacity
            ),
            lifetime_identity_bits=64,
            age_telemetry_saturation=_INT32_MAX,
            operation_telemetry_saturation=_INT32_MAX,
            random_draws_per_query=0,
            random_draws_per_write=0,
            scientific_promotion_allowed=(
                EXPERIENTIAL_MEMORY_SCIENTIFIC_PROMOTION_ALLOWED
            ),
        )


def _legacy_dataclass_or_mapping_fields(value: Any, *, name: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: getattr(value, field.name)
            for field in dataclasses.fields(value)
        }
    raise TypeError(f"legacy {name} must be a mapping or dataclass")


def migrate_legacy_experiential_memory_state(
    memory: ExperientialMemory,
    legacy_state: Any,
) -> ExperientialMemoryState:
    """Migrate only an unsaturated pre-v2 temporal state.

    Legacy ages cannot recover their historical insertion timestamps.  The
    migration therefore anchors each active timestamp at the legacy global
    step and preserves its exact current age as an offset.  Future age and
    recency evolution is then exact.  Any saturated legacy clock or telemetry
    count is rejected because its history is ambiguous.
    """

    if type(memory) is not ExperientialMemory:
        raise TypeError("memory must be an ExperientialMemory")
    fields = _legacy_dataclass_or_mapping_fields(
        legacy_state,
        name="experiential-memory state",
    )
    current_state_names = {
        field.name
        for field in dataclasses.fields(ExperientialMemoryState)  # type: ignore[arg-type]
    }
    legacy_state_names = current_state_names - {"step_words"}
    if set(fields) != legacy_state_names:
        missing = sorted(legacy_state_names - set(fields))
        extra = sorted(set(fields) - legacy_state_names)
        raise ValueError(
            "legacy experiential-memory state fields are not exact; "
            f"missing={missing}, extra={extra}"
        )
    raw_entries = _legacy_dataclass_or_mapping_fields(
        fields["entries"],
        name="experiential-memory entries",
    )
    temporal_fields = {
        "insertion_step_words",
        "last_access_step_words",
        "insertion_age_offsets",
        "last_access_age_offsets",
    }
    current_entry_names = {
        field.name
        for field in dataclasses.fields(ExperientialMemoryEntries)  # type: ignore[arg-type]
    }
    legacy_entry_names = current_entry_names - temporal_fields
    if set(raw_entries) != legacy_entry_names:
        missing = sorted(legacy_entry_names - set(raw_entries))
        extra = sorted(set(raw_entries) - legacy_entry_names)
        raise ValueError(
            "legacy experiential-memory entry fields are not exact; "
            f"missing={missing}, extra={extra}"
        )

    cfg = memory.config
    legacy_persistent_bytes, _ = _legacy_configured_nbytes(cfg)
    persisted_bytes = jnp.asarray(fields["persistent_bytes"])
    if persisted_bytes.shape != () or persisted_bytes.dtype != jnp.dtype(jnp.uint32):
        raise TypeError("legacy persistent_bytes must be scalar uint32")
    if int(persisted_bytes) != legacy_persistent_bytes:
        raise ValueError("legacy persistent byte accounting is inconsistent")
    for counter_name in (
        "step_count",
        "query_count",
        "accepted_query_count",
        "write_count",
        "rejected_write_count",
        "eviction_count",
    ):
        counter = jnp.asarray(fields[counter_name])
        if counter.shape != () or counter.dtype != jnp.dtype(jnp.int32):
            raise TypeError(f"legacy {counter_name} must be scalar int32")
        count = int(counter)
        if count < 0:
            raise ValueError(f"negative legacy {counter_name} indicates wrap")
        if count >= _INT32_MAX:
            raise ValueError(f"saturated legacy {counter_name} is ambiguous")
    for counter_name in ("ages", "recency_ages", "retrieval_counts"):
        counter = jnp.asarray(raw_entries[counter_name])
        if counter.shape != (cfg.capacity,) or counter.dtype != jnp.dtype(jnp.int32):
            raise TypeError(
                f"legacy entries.{counter_name} must be int32 with capacity shape"
            )
        if bool(jax.device_get(jnp.any(counter < 0))):
            raise ValueError(f"negative legacy entries.{counter_name} indicates wrap")
        if bool(jax.device_get(jnp.any(counter >= _INT32_MAX))):
            raise ValueError(
                f"saturated legacy entries.{counter_name} is ambiguous"
            )

    valid = jnp.asarray(raw_entries["valid"])
    if valid.shape != (cfg.capacity,) or valid.dtype != jnp.dtype(jnp.bool_):
        raise TypeError("legacy entries.valid must be bool with capacity shape")
    step = int(jnp.asarray(fields["step_count"]))
    step_words = jnp.asarray((0, step), dtype=jnp.uint32)
    timestamps = jnp.where(
        valid[:, None],
        jnp.broadcast_to(step_words, (cfg.capacity, 2)),
        jnp.zeros((cfg.capacity, 2), dtype=jnp.uint32),
    )
    zero_offsets = jnp.zeros((cfg.capacity,), dtype=jnp.int32)
    raw_entries["insertion_step_words"] = timestamps
    raw_entries["last_access_step_words"] = timestamps
    raw_entries["insertion_age_offsets"] = jnp.where(
        valid,
        raw_entries["ages"],
        zero_offsets,
    )
    raw_entries["last_access_age_offsets"] = jnp.where(
        valid,
        raw_entries["recency_ages"],
        zero_offsets,
    )
    fields["entries"] = ExperientialMemoryEntries(**raw_entries)
    fields["step_words"] = step_words
    fields["persistent_bytes"] = jnp.asarray(
        memory.persistent_bytes,
        dtype=jnp.uint32,
    )
    migrated = ExperientialMemoryState(**fields)
    memory._validate_state_static_contract(migrated)
    if not bool(jax.device_get(memory._state_is_valid(migrated))):
        raise ValueError("migrated experiential-memory state is invalid")
    return migrated


def save_experiential_memory_checkpoint(
    memory: ExperientialMemory,
    state: ExperientialMemoryState,
    path: str | Path,
) -> None:
    """Persist one valid exact-clock memory state with a strict v2 manifest."""

    if type(memory) is not ExperientialMemory:
        raise TypeError("memory must be an ExperientialMemory")
    memory._validate_state_static_contract(state)
    if not bool(jax.device_get(memory._state_is_valid(state))):
        raise ValueError("experiential-memory state is invalid")
    save_checkpoint(
        state,
        path,
        metadata={
            "schema": EXPERIENTIAL_MEMORY_CHECKPOINT_SCHEMA,
            "state_schema": EXPERIENTIAL_MEMORY_STATE_SCHEMA,
            "mechanism_status": EXPERIENTIAL_MEMORY_MECHANISM_STATUS,
            "scientific_promotion_allowed": (
                EXPERIENTIAL_MEMORY_SCIENTIFIC_PROMOTION_ALLOWED
            ),
            "memory": memory.to_config(),
            "resource_budget": memory.resource_budget(state).to_config(),
        },
    )


def load_experiential_memory_checkpoint(
    path: str | Path,
) -> tuple[ExperientialMemory, ExperientialMemoryState]:
    """Restore only a strict v2 state and revalidate its resource contract."""

    metadata = load_checkpoint_metadata(path)
    expected = {
        "schema",
        "state_schema",
        "mechanism_status",
        "scientific_promotion_allowed",
        "memory",
        "resource_budget",
    }
    schema = metadata.get("schema")
    if schema == _LEGACY_EXPERIENTIAL_MEMORY_CHECKPOINT_SCHEMA:
        raise ValueError(
            "legacy experiential-memory checkpoint lacks exact temporal identities; "
            "migrate its decoded state with migrate_legacy_experiential_memory_state "
            "and resave it"
        )
    if set(metadata) != expected:
        raise ValueError("experiential-memory checkpoint fields do not match v2")
    if schema != EXPERIENTIAL_MEMORY_CHECKPOINT_SCHEMA:
        raise ValueError("experiential-memory checkpoint schema is unsupported")
    if metadata["state_schema"] != EXPERIENTIAL_MEMORY_STATE_SCHEMA:
        raise ValueError("experiential-memory checkpoint state schema is unsupported")
    if metadata["mechanism_status"] != EXPERIENTIAL_MEMORY_MECHANISM_STATUS:
        raise ValueError("experiential-memory checkpoint mechanism status differs")
    if metadata["scientific_promotion_allowed"] is not False:
        raise ValueError("experiential-memory checkpoint cannot claim promotion")
    raw_memory = metadata["memory"]
    if type(raw_memory) is not dict:
        raise ValueError("experiential-memory checkpoint construction is invalid")
    memory = ExperientialMemory.from_config(raw_memory)
    restored, restored_metadata = load_checkpoint(memory.init(), path)
    if restored_metadata != metadata:
        raise ValueError("experiential-memory checkpoint metadata changed between reads")
    state = cast(ExperientialMemoryState, restored)
    memory._validate_state_static_contract(state)
    if not bool(jax.device_get(memory._state_is_valid(state))):
        raise ValueError("experiential-memory checkpoint state is invalid")
    resource_budget = metadata["resource_budget"]
    if type(resource_budget) is not dict:
        raise ValueError("experiential-memory checkpoint resource budget is invalid")
    if resource_budget != memory.resource_budget(state).to_config():
        raise ValueError("experiential-memory checkpoint resource contract changed")
    return memory, state


__all__ = [
    "EXPERIENTIAL_MEMORY_CHECKPOINT_SCHEMA",
    "EXPERIENTIAL_MEMORY_CONFIG_SCHEMA",
    "EXPERIENTIAL_MEMORY_EXACT_GLOBAL_STEP_IDENTITY_NBYTES",
    "EXPERIENTIAL_MEMORY_EXACT_SLOT_TEMPORAL_IDENTITY_NBYTES",
    "EXPERIENTIAL_MEMORY_MECHANISM_STATUS",
    "EXPERIENTIAL_MEMORY_SCIENTIFIC_PROMOTION_ALLOWED",
    "EXPERIENTIAL_MEMORY_STATE_SCHEMA",
    "ExperientialMemory",
    "ExperientialMemoryAccounting",
    "ExperientialMemoryConfig",
    "ExperientialMemoryEntries",
    "ExperientialMemoryEntry",
    "ExperientialMemoryRetrieval",
    "ExperientialMemoryResourceBudget",
    "ExperientialMemoryState",
    "ExperientialMemoryStepResult",
    "ExperientialMemoryWriteResult",
    "load_experiential_memory_checkpoint",
    "migrate_legacy_experiential_memory_config",
    "migrate_legacy_experiential_memory_state",
    "save_experiential_memory_checkpoint",
]
