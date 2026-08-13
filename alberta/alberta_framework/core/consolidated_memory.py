# mypy: disable-error-code="arg-type,attr-defined,call-arg,redundant-cast,type-var"
"""Bounded semantic and procedural consolidation for continuing agents.

This module is an L0 mechanism: it demonstrates fixed-capacity storage,
causal query-before-write, compatibility gates, deterministic retirement,
and strict checkpoint bindings.  It does not claim that consolidation
improves an agent, and it has no authority to select actions, mutate an
agent, promote evidence, or make a go/no-go decision.

Canonical identity is supplied as a SHA-256 byte vector.  The helper
:func:`canonical_memory_digest` creates such vectors from an explicit domain
and canonical text.  A live source digest and semantic-namespace digest bind
the whole state; per-record provenance digests bind individual observations.

Within one canonical identity, a write at the current generation performs a
Welford merge.  A write at exactly the next generation is an explicit
revision and resets all evidence.  A different identity inserted into a
retired or evicted slot also receives a fully reset record.  Consequently,
evidence never transfers across changed semantics or generations.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
import math
from typing import Any, ClassVar, cast

import chex
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import Bool, Float, Int, UInt

CONSOLIDATED_MEMORY_CONFIG_SCHEMA = "alberta.consolidated-memory.config.v1"
CONSOLIDATED_MEMORY_CHECKPOINT_SCHEMA = "alberta.consolidated-memory.state.v1"
CONSOLIDATED_MEMORY_AGENT_MUTATION_AUTHORITY = False
CONSOLIDATED_MEMORY_ACTION_SELECTION_AUTHORITY = False
CONSOLIDATED_MEMORY_PROMOTION_AUTHORITY = False
CONSOLIDATED_MEMORY_GO_NO_GO_AUTHORITY = False
CONSOLIDATED_MEMORY_SCIENTIFIC_PROMOTION_ALLOWED = False

SEMANTIC_KIND_GVF = 0
SEMANTIC_KIND_FACT = 1
SEMANTIC_KIND_AFFORDANCE = 2

_DIGEST_BYTES = 32
_INT32_MAX = 2**31 - 1
_MAX_CAPACITY = 65_536
_MAX_VECTOR_DIM = 16_384
_MAX_LOGICAL_CELLS = 16_777_216


def canonical_memory_digest(domain: str, canonical_text: str) -> Array:
    """Return the SHA-256 bytes for an explicitly domain-separated label.

    This host-side helper is deliberately not used inside JIT regions.
    Callers own canonicalization of ``canonical_text``; using a different
    spelling is therefore a different semantic identity rather than an alias.
    """

    if type(domain) is not str or not domain:
        raise ValueError("domain must be a nonempty exact string")
    if type(canonical_text) is not str or not canonical_text:
        raise ValueError("canonical_text must be a nonempty exact string")
    encoded = json.dumps(
        {"domain": domain, "text": canonical_text},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return jnp.asarray(tuple(hashlib.sha256(encoded).digest()), dtype=jnp.uint8)


def _positive_int(value: object, *, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive exact Python int")
    return value


def _nonnegative_int(value: object, *, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative exact Python int")
    return value


def _probability(value: object, *, name: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite exact Python float")
    represented = float(jnp.asarray(value, dtype=jnp.float32))
    if not math.isfinite(represented) or not 0.0 <= represented <= 1.0:
        raise ValueError(f"{name} must remain in [0, 1] in float32")
    return value


def _require_array(
    value: Any,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: Any,
) -> None:
    if not hasattr(value, "shape") or not hasattr(value, "dtype"):
        raise TypeError(f"{name} must expose array shape and dtype metadata")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")
    if jnp.dtype(value.dtype) != jnp.dtype(dtype):
        raise TypeError(f"{name} must have dtype {jnp.dtype(dtype)}, got {value.dtype}")


def _int32_scalar(value: int | Array, *, name: str) -> Array:
    if type(value) is int:
        if not -(2**31) <= value <= _INT32_MAX:
            raise ValueError(f"{name} must be signed-int32 compatible")
        result = jnp.asarray(value, dtype=jnp.int32)
    else:
        result = jnp.asarray(value)
    _require_array(result, name=name, shape=(), dtype=jnp.int32)
    return result


def _tree_nbytes(tree: Any) -> int:
    return sum(
        int(np.prod(leaf.shape, dtype=np.int64)) * int(leaf.dtype.itemsize)
        for leaf in jax.tree_util.tree_leaves(tree)
    )


def _digest_is_nonzero(value: Array) -> Array:
    return jnp.any(value != jnp.asarray(0, dtype=jnp.uint8), axis=-1)


def _first_true(mask: Array) -> Array:
    return jnp.argmax(mask.astype(jnp.int32)).astype(jnp.int32)


def _welford_scalar(mean: Array, m2: Array, count: Array, value: Array) -> tuple[Array, Array]:
    next_count = count + jnp.asarray(1, dtype=jnp.int32)
    delta = value - mean
    next_mean = mean + delta / next_count.astype(jnp.float32)
    next_m2 = m2 + delta * (value - next_mean)
    return next_mean, next_m2


def _welford_vector(mean: Array, m2: Array, count: Array, value: Array) -> tuple[Array, Array]:
    next_count = count + jnp.asarray(1, dtype=jnp.int32)
    delta = value - mean
    next_mean = mean + delta / next_count.astype(jnp.float32)
    next_m2 = m2 + delta * (value - next_mean)
    return next_mean, next_m2


@dataclasses.dataclass(frozen=True, slots=True)
class ConsolidatedMemoryConfig:
    """Static allocation and compatibility policy."""

    semantic_capacity: int
    procedural_capacity: int
    semantic_payload_dim: int
    procedural_payload_dim: int
    procedural_outcome_dim: int
    semantic_max_age: int
    procedural_max_age: int
    max_operations: int
    semantic_min_confidence: float = 0.5
    procedural_min_confidence: float = 0.5

    SCHEMA_VERSION: ClassVar[str] = CONSOLIDATED_MEMORY_CONFIG_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "semantic_capacity",
            "procedural_capacity",
            "semantic_payload_dim",
            "procedural_payload_dim",
            "procedural_outcome_dim",
            "max_operations",
        ):
            value = _positive_int(getattr(self, name), name=name)
            if "capacity" in name and value > _MAX_CAPACITY:
                raise ValueError(f"{name} exceeds the fixed capacity ceiling")
            if "dim" in name and value > _MAX_VECTOR_DIM:
                raise ValueError(f"{name} exceeds the fixed vector-width ceiling")
        for name in ("semantic_max_age", "procedural_max_age"):
            _nonnegative_int(getattr(self, name), name=name)
        if self.max_operations > _INT32_MAX - 1:
            raise ValueError("max_operations exceeds the exact signed-int32 counter ceiling")
        if max(self.semantic_max_age, self.procedural_max_age) > self.max_operations:
            raise ValueError("maximum ages must not exceed max_operations")
        _probability(self.semantic_min_confidence, name="semantic_min_confidence")
        _probability(self.procedural_min_confidence, name="procedural_min_confidence")
        cells = self.semantic_capacity * (
            2 * self.semantic_payload_dim + 90
        ) + self.procedural_capacity * (
            2 * self.procedural_payload_dim + 2 * self.procedural_outcome_dim + 127
        )
        if cells > _MAX_LOGICAL_CELLS:
            raise ValueError("configured consolidated memory exceeds the fixed cell ceiling")

    def to_config(self) -> dict[str, object]:
        payload = dataclasses.asdict(self)
        payload["schema_version"] = self.SCHEMA_VERSION
        return payload

    @classmethod
    def from_config(cls, value: object) -> ConsolidatedMemoryConfig:
        if type(value) is not dict:
            raise ValueError("consolidated memory config must be an exact dict")
        raw = cast(dict[object, object], value)
        expected = {field.name for field in dataclasses.fields(cls)} | {"schema_version"}
        if set(raw) != expected:
            raise ValueError("consolidated memory config keys differ from schema v1")
        if raw["schema_version"] != cls.SCHEMA_VERSION:
            raise ValueError("consolidated memory config schema differs")
        kwargs = {name: raw[name] for name in expected if name != "schema_version"}
        return cls(**cast(Any, kwargs))


@chex.dataclass(frozen=True)
class SemanticMemoryRecord:
    """One candidate semantic observation presented for consolidation."""

    semantic_digest: UInt[Array, " 32"]
    generation: Int[Array, ""]
    kind: Int[Array, ""]
    payload: Float[Array, " semantic_payload_dim"]
    confidence: Float[Array, ""]
    provenance_digest: UInt[Array, " 32"]
    representation_revision: Int[Array, ""]
    source_revision: Int[Array, ""]
    evidence: Float[Array, ""]


@chex.dataclass(frozen=True)
class SemanticMemoryRequest:
    """Exact compatibility request for one semantic identity."""

    semantic_digest: UInt[Array, " 32"]
    generation: Int[Array, ""]
    kind: Int[Array, ""]
    provenance_digest: UInt[Array, " 32"]
    representation_revision: Int[Array, ""]
    source_revision: Int[Array, ""]


@chex.dataclass(frozen=True)
class ProceduralMemoryRecord:
    """One procedural observation and its option-lifecycle binding."""

    semantic_digest: UInt[Array, " 32"]
    generation: Int[Array, ""]
    payload: Float[Array, " procedural_payload_dim"]
    confidence: Float[Array, ""]
    provenance_digest: UInt[Array, " 32"]
    representation_revision: Int[Array, ""]
    source_revision: Int[Array, ""]
    evidence: Float[Array, ""]
    succeeded: Bool[Array, ""]
    outcome: Float[Array, " procedural_outcome_dim"]
    lifecycle_link_available: Bool[Array, ""]
    lifecycle_digest: UInt[Array, " 32"]
    lifecycle_generation: Int[Array, ""]
    lifecycle_revision: Int[Array, ""]


@chex.dataclass(frozen=True)
class ProceduralMemoryRequest:
    """Exact compatibility request for one procedural identity."""

    semantic_digest: UInt[Array, " 32"]
    generation: Int[Array, ""]
    provenance_digest: UInt[Array, " 32"]
    representation_revision: Int[Array, ""]
    source_revision: Int[Array, ""]
    lifecycle_link_available: Bool[Array, ""]
    lifecycle_digest: UInt[Array, " 32"]
    lifecycle_generation: Int[Array, ""]
    lifecycle_revision: Int[Array, ""]


@chex.dataclass(frozen=True)
class SemanticMemoryRecords:
    semantic_digests: UInt[Array, "semantic_capacity 32"]
    generations: Int[Array, " semantic_capacity"]
    kinds: Int[Array, " semantic_capacity"]
    payload_means: Float[Array, "semantic_capacity semantic_payload_dim"]
    payload_m2: Float[Array, "semantic_capacity semantic_payload_dim"]
    confidences: Float[Array, " semantic_capacity"]
    provenance_digests: UInt[Array, "semantic_capacity 32"]
    representation_revisions: Int[Array, " semantic_capacity"]
    source_revisions: Int[Array, " semantic_capacity"]
    creation_steps: Int[Array, " semantic_capacity"]
    last_use_steps: Int[Array, " semantic_capacity"]
    access_counts: Int[Array, " semantic_capacity"]
    evidence_counts: Int[Array, " semantic_capacity"]
    evidence_means: Float[Array, " semantic_capacity"]
    evidence_m2: Float[Array, " semantic_capacity"]
    occupied: Bool[Array, " semantic_capacity"]
    valid: Bool[Array, " semantic_capacity"]
    stale: Bool[Array, " semantic_capacity"]
    invalidated: Bool[Array, " semantic_capacity"]


@chex.dataclass(frozen=True)
class ProceduralMemoryRecords:
    semantic_digests: UInt[Array, "procedural_capacity 32"]
    generations: Int[Array, " procedural_capacity"]
    payload_means: Float[Array, "procedural_capacity procedural_payload_dim"]
    payload_m2: Float[Array, "procedural_capacity procedural_payload_dim"]
    confidences: Float[Array, " procedural_capacity"]
    provenance_digests: UInt[Array, "procedural_capacity 32"]
    representation_revisions: Int[Array, " procedural_capacity"]
    source_revisions: Int[Array, " procedural_capacity"]
    creation_steps: Int[Array, " procedural_capacity"]
    last_use_steps: Int[Array, " procedural_capacity"]
    access_counts: Int[Array, " procedural_capacity"]
    evidence_counts: Int[Array, " procedural_capacity"]
    evidence_means: Float[Array, " procedural_capacity"]
    evidence_m2: Float[Array, " procedural_capacity"]
    success_counts: Int[Array, " procedural_capacity"]
    failure_counts: Int[Array, " procedural_capacity"]
    outcome_means: Float[Array, "procedural_capacity procedural_outcome_dim"]
    outcome_m2: Float[Array, "procedural_capacity procedural_outcome_dim"]
    lifecycle_link_available: Bool[Array, " procedural_capacity"]
    lifecycle_digests: UInt[Array, "procedural_capacity 32"]
    lifecycle_generations: Int[Array, " procedural_capacity"]
    lifecycle_revisions: Int[Array, " procedural_capacity"]
    occupied: Bool[Array, " procedural_capacity"]
    valid: Bool[Array, " procedural_capacity"]
    stale: Bool[Array, " procedural_capacity"]
    invalidated: Bool[Array, " procedural_capacity"]


@chex.dataclass(frozen=True)
class ConsolidatedMemoryState:
    """Complete fixed-shape memory and exact lifetime counters."""

    semantic: SemanticMemoryRecords
    procedural: ProceduralMemoryRecords
    source_digest: UInt[Array, " 32"]
    semantic_namespace_digest: UInt[Array, " 32"]
    representation_revision: Int[Array, ""]
    source_revision: Int[Array, ""]
    operation_count: Int[Array, ""]
    semantic_query_count: Int[Array, ""]
    semantic_accepted_query_count: Int[Array, ""]
    semantic_write_count: Int[Array, ""]
    semantic_merge_count: Int[Array, ""]
    semantic_revision_count: Int[Array, ""]
    semantic_replacement_count: Int[Array, ""]
    semantic_rejected_write_count: Int[Array, ""]
    semantic_invalidation_count: Int[Array, ""]
    semantic_retirement_count: Int[Array, ""]
    procedural_query_count: Int[Array, ""]
    procedural_accepted_query_count: Int[Array, ""]
    procedural_write_count: Int[Array, ""]
    procedural_merge_count: Int[Array, ""]
    procedural_revision_count: Int[Array, ""]
    procedural_replacement_count: Int[Array, ""]
    procedural_rejected_write_count: Int[Array, ""]
    procedural_invalidation_count: Int[Array, ""]
    procedural_retirement_count: Int[Array, ""]


@chex.dataclass(frozen=True)
class SemanticMemoryRetrieval:
    accepted: Bool[Array, ""]
    transaction_applied: Bool[Array, ""]
    slot: Int[Array, ""]
    payload: Float[Array, " semantic_payload_dim"]
    confidence: Float[Array, ""]
    evidence_count: Int[Array, ""]
    evidence_mean: Float[Array, ""]
    evidence_m2: Float[Array, ""]
    state_valid: Bool[Array, ""]
    request_valid: Bool[Array, ""]
    identity_found: Bool[Array, ""]
    compatible: Bool[Array, ""]
    fresh: Bool[Array, ""]
    confidence_ok: Bool[Array, ""]


@chex.dataclass(frozen=True)
class ProceduralMemoryRetrieval:
    accepted: Bool[Array, ""]
    transaction_applied: Bool[Array, ""]
    slot: Int[Array, ""]
    payload: Float[Array, " procedural_payload_dim"]
    confidence: Float[Array, ""]
    evidence_count: Int[Array, ""]
    evidence_mean: Float[Array, ""]
    evidence_m2: Float[Array, ""]
    success_count: Int[Array, ""]
    failure_count: Int[Array, ""]
    outcome_mean: Float[Array, " procedural_outcome_dim"]
    outcome_m2: Float[Array, " procedural_outcome_dim"]
    lifecycle_link_available: Bool[Array, ""]
    lifecycle_digest: UInt[Array, " 32"]
    lifecycle_generation: Int[Array, ""]
    lifecycle_revision: Int[Array, ""]
    state_valid: Bool[Array, ""]
    request_valid: Bool[Array, ""]
    identity_found: Bool[Array, ""]
    compatible: Bool[Array, ""]
    fresh: Bool[Array, ""]
    confidence_ok: Bool[Array, ""]


@chex.dataclass(frozen=True)
class MemoryWriteDiagnostics:
    transaction_applied: Bool[Array, ""]
    wrote: Bool[Array, ""]
    merged: Bool[Array, ""]
    revised: Bool[Array, ""]
    replaced: Bool[Array, ""]
    reset_evidence: Bool[Array, ""]
    slot: Int[Array, ""]
    state_valid: Bool[Array, ""]
    record_valid: Bool[Array, ""]
    identity_collision: Bool[Array, ""]
    generation_compatible: Bool[Array, ""]
    metadata_compatible: Bool[Array, ""]


@chex.dataclass(frozen=True)
class SemanticMemoryStepResult:
    state: ConsolidatedMemoryState
    retrieval: SemanticMemoryRetrieval
    write: MemoryWriteDiagnostics


@chex.dataclass(frozen=True)
class ProceduralMemoryStepResult:
    state: ConsolidatedMemoryState
    retrieval: ProceduralMemoryRetrieval
    write: MemoryWriteDiagnostics


@chex.dataclass(frozen=True)
class MemoryInvalidationResult:
    state: ConsolidatedMemoryState
    transaction_applied: Bool[Array, ""]
    invalidated: Bool[Array, ""]
    slot: Int[Array, ""]


@chex.dataclass(frozen=True)
class ConsolidatedMemoryAccounting:
    persistent_state_bytes: Int[Array, ""]
    semantic_capacity: Int[Array, ""]
    procedural_capacity: Int[Array, ""]
    active_semantic_records: Int[Array, ""]
    active_procedural_records: Int[Array, ""]
    occupied_semantic_records: Int[Array, ""]
    occupied_procedural_records: Int[Array, ""]
    operation_count: Int[Array, ""]
    semantic_queries: Int[Array, ""]
    semantic_writes: Int[Array, ""]
    procedural_queries: Int[Array, ""]
    procedural_writes: Int[Array, ""]


@dataclasses.dataclass(frozen=True, slots=True)
class ConsolidatedMemoryResourceBudget:
    persistent_logical_scalars: int
    persistent_state_bytes: int
    semantic_capacity: int
    procedural_capacity: int
    semantic_payload_dim: int
    procedural_payload_dim: int
    procedural_outcome_dim: int
    maximum_operations: int
    dynamic_persistent_allocations_per_operation: int
    random_generator_calls_at_init: int
    random_generator_calls_per_operation: int
    agent_parameter_mutations_per_operation: int
    action_selection_calls_per_operation: int
    promotion_decisions_per_operation: int
    agent_mutation_authority: bool
    action_selection_authority: bool
    promotion_authority: bool
    go_no_go_authority: bool
    scientific_promotion_allowed: bool
    checkpoint_schema: str


class ConsolidatedMemory:
    """Fixed-capacity semantic and procedural consolidation mechanism."""

    def __init__(self, config: ConsolidatedMemoryConfig) -> None:
        if type(config) is not ConsolidatedMemoryConfig:
            raise TypeError("config must be an exact ConsolidatedMemoryConfig")
        self._config = config
        config_bytes = json.dumps(config.to_config(), sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        self._config_sha256 = tuple(hashlib.sha256(config_bytes).digest())
        dummy = self._make_initial_state(
            source_digest=jnp.ones((_DIGEST_BYTES,), dtype=jnp.uint8),
            semantic_namespace_digest=jnp.full((_DIGEST_BYTES,), 2, dtype=jnp.uint8),
            representation_revision=jnp.asarray(0, dtype=jnp.int32),
            source_revision=jnp.asarray(0, dtype=jnp.int32),
        )
        self._persistent_state_bytes = _tree_nbytes(dummy)
        self._persistent_logical_scalars = sum(
            int(np.prod(leaf.shape, dtype=np.int64)) for leaf in jax.tree_util.tree_leaves(dummy)
        )

    @property
    def config(self) -> ConsolidatedMemoryConfig:
        return self._config

    def to_config(self) -> dict[str, object]:
        return self._config.to_config()

    @classmethod
    def from_config(cls, value: object) -> ConsolidatedMemory:
        return cls(ConsolidatedMemoryConfig.from_config(value))

    @property
    def resource_budget(self) -> ConsolidatedMemoryResourceBudget:
        cfg = self._config
        return ConsolidatedMemoryResourceBudget(
            persistent_logical_scalars=self._persistent_logical_scalars,
            persistent_state_bytes=self._persistent_state_bytes,
            semantic_capacity=cfg.semantic_capacity,
            procedural_capacity=cfg.procedural_capacity,
            semantic_payload_dim=cfg.semantic_payload_dim,
            procedural_payload_dim=cfg.procedural_payload_dim,
            procedural_outcome_dim=cfg.procedural_outcome_dim,
            maximum_operations=cfg.max_operations,
            dynamic_persistent_allocations_per_operation=0,
            random_generator_calls_at_init=0,
            random_generator_calls_per_operation=0,
            agent_parameter_mutations_per_operation=0,
            action_selection_calls_per_operation=0,
            promotion_decisions_per_operation=0,
            agent_mutation_authority=False,
            action_selection_authority=False,
            promotion_authority=False,
            go_no_go_authority=False,
            scientific_promotion_allowed=False,
            checkpoint_schema=CONSOLIDATED_MEMORY_CHECKPOINT_SCHEMA,
        )

    def _make_initial_state(
        self,
        *,
        source_digest: Array,
        semantic_namespace_digest: Array,
        representation_revision: Array,
        source_revision: Array,
    ) -> ConsolidatedMemoryState:
        cfg = self._config
        sc = cfg.semantic_capacity
        pc = cfg.procedural_capacity
        zeros_i_s = jnp.zeros((sc,), dtype=jnp.int32)
        zeros_i_p = jnp.zeros((pc,), dtype=jnp.int32)
        semantic = SemanticMemoryRecords(
            semantic_digests=jnp.zeros((sc, _DIGEST_BYTES), dtype=jnp.uint8),
            generations=jnp.full((sc,), -1, dtype=jnp.int32),
            kinds=jnp.full((sc,), -1, dtype=jnp.int32),
            payload_means=jnp.zeros((sc, cfg.semantic_payload_dim), dtype=jnp.float32),
            payload_m2=jnp.zeros((sc, cfg.semantic_payload_dim), dtype=jnp.float32),
            confidences=jnp.zeros((sc,), dtype=jnp.float32),
            provenance_digests=jnp.zeros((sc, _DIGEST_BYTES), dtype=jnp.uint8),
            representation_revisions=jnp.full((sc,), -1, dtype=jnp.int32),
            source_revisions=jnp.full((sc,), -1, dtype=jnp.int32),
            creation_steps=zeros_i_s,
            last_use_steps=zeros_i_s,
            access_counts=zeros_i_s,
            evidence_counts=zeros_i_s,
            evidence_means=jnp.zeros((sc,), dtype=jnp.float32),
            evidence_m2=jnp.zeros((sc,), dtype=jnp.float32),
            occupied=jnp.zeros((sc,), dtype=jnp.bool_),
            valid=jnp.zeros((sc,), dtype=jnp.bool_),
            stale=jnp.zeros((sc,), dtype=jnp.bool_),
            invalidated=jnp.zeros((sc,), dtype=jnp.bool_),
        )
        procedural = ProceduralMemoryRecords(
            semantic_digests=jnp.zeros((pc, _DIGEST_BYTES), dtype=jnp.uint8),
            generations=jnp.full((pc,), -1, dtype=jnp.int32),
            payload_means=jnp.zeros((pc, cfg.procedural_payload_dim), dtype=jnp.float32),
            payload_m2=jnp.zeros((pc, cfg.procedural_payload_dim), dtype=jnp.float32),
            confidences=jnp.zeros((pc,), dtype=jnp.float32),
            provenance_digests=jnp.zeros((pc, _DIGEST_BYTES), dtype=jnp.uint8),
            representation_revisions=jnp.full((pc,), -1, dtype=jnp.int32),
            source_revisions=jnp.full((pc,), -1, dtype=jnp.int32),
            creation_steps=zeros_i_p,
            last_use_steps=zeros_i_p,
            access_counts=zeros_i_p,
            evidence_counts=zeros_i_p,
            evidence_means=jnp.zeros((pc,), dtype=jnp.float32),
            evidence_m2=jnp.zeros((pc,), dtype=jnp.float32),
            success_counts=zeros_i_p,
            failure_counts=zeros_i_p,
            outcome_means=jnp.zeros((pc, cfg.procedural_outcome_dim), dtype=jnp.float32),
            outcome_m2=jnp.zeros((pc, cfg.procedural_outcome_dim), dtype=jnp.float32),
            lifecycle_link_available=jnp.zeros((pc,), dtype=jnp.bool_),
            lifecycle_digests=jnp.zeros((pc, _DIGEST_BYTES), dtype=jnp.uint8),
            lifecycle_generations=jnp.full((pc,), -1, dtype=jnp.int32),
            lifecycle_revisions=jnp.full((pc,), -1, dtype=jnp.int32),
            occupied=jnp.zeros((pc,), dtype=jnp.bool_),
            valid=jnp.zeros((pc,), dtype=jnp.bool_),
            stale=jnp.zeros((pc,), dtype=jnp.bool_),
            invalidated=jnp.zeros((pc,), dtype=jnp.bool_),
        )
        zero = jnp.asarray(0, dtype=jnp.int32)
        return ConsolidatedMemoryState(
            semantic=semantic,
            procedural=procedural,
            source_digest=source_digest,
            semantic_namespace_digest=semantic_namespace_digest,
            representation_revision=representation_revision,
            source_revision=source_revision,
            operation_count=zero,
            semantic_query_count=zero,
            semantic_accepted_query_count=zero,
            semantic_write_count=zero,
            semantic_merge_count=zero,
            semantic_revision_count=zero,
            semantic_replacement_count=zero,
            semantic_rejected_write_count=zero,
            semantic_invalidation_count=zero,
            semantic_retirement_count=zero,
            procedural_query_count=zero,
            procedural_accepted_query_count=zero,
            procedural_write_count=zero,
            procedural_merge_count=zero,
            procedural_revision_count=zero,
            procedural_replacement_count=zero,
            procedural_rejected_write_count=zero,
            procedural_invalidation_count=zero,
            procedural_retirement_count=zero,
        )

    def init(
        self,
        *,
        source_digest: Array,
        semantic_namespace_digest: Array,
        representation_revision: int | Array,
        source_revision: int | Array,
    ) -> ConsolidatedMemoryState:
        """Return an empty state bound to one source and semantic namespace."""

        _require_array(source_digest, name="source_digest", shape=(32,), dtype=jnp.uint8)
        _require_array(
            semantic_namespace_digest,
            name="semantic_namespace_digest",
            shape=(32,),
            dtype=jnp.uint8,
        )
        representation = _int32_scalar(representation_revision, name="representation_revision")
        source = _int32_scalar(source_revision, name="source_revision")
        if not bool(jax.device_get(_digest_is_nonzero(source_digest))):
            raise ValueError("source_digest must be nonzero")
        if not bool(jax.device_get(_digest_is_nonzero(semantic_namespace_digest))):
            raise ValueError("semantic_namespace_digest must be nonzero")
        if int(representation) < 0 or int(source) < 0:
            raise ValueError("representation and source revisions must be non-negative")
        state = self._make_initial_state(
            source_digest=source_digest,
            semantic_namespace_digest=semantic_namespace_digest,
            representation_revision=representation,
            source_revision=source,
        )
        if _tree_nbytes(state) != self._persistent_state_bytes:
            raise RuntimeError("persistent byte accounting differs from the allocated state")
        return state

    def _validate_state_static_contract(self, state: ConsolidatedMemoryState) -> None:
        if not isinstance(state, ConsolidatedMemoryState):
            raise TypeError("state must be a ConsolidatedMemoryState")
        cfg = self._config
        semantic = state.semantic
        procedural = state.procedural
        if not isinstance(semantic, SemanticMemoryRecords):
            raise TypeError("state.semantic must be SemanticMemoryRecords")
        if not isinstance(procedural, ProceduralMemoryRecords):
            raise TypeError("state.procedural must be ProceduralMemoryRecords")
        semantic_shapes: dict[str, tuple[tuple[int, ...], Any]] = {
            "semantic_digests": ((cfg.semantic_capacity, 32), jnp.uint8),
            "generations": ((cfg.semantic_capacity,), jnp.int32),
            "kinds": ((cfg.semantic_capacity,), jnp.int32),
            "payload_means": (
                (cfg.semantic_capacity, cfg.semantic_payload_dim),
                jnp.float32,
            ),
            "payload_m2": ((cfg.semantic_capacity, cfg.semantic_payload_dim), jnp.float32),
            "confidences": ((cfg.semantic_capacity,), jnp.float32),
            "provenance_digests": ((cfg.semantic_capacity, 32), jnp.uint8),
            "representation_revisions": ((cfg.semantic_capacity,), jnp.int32),
            "source_revisions": ((cfg.semantic_capacity,), jnp.int32),
            "creation_steps": ((cfg.semantic_capacity,), jnp.int32),
            "last_use_steps": ((cfg.semantic_capacity,), jnp.int32),
            "access_counts": ((cfg.semantic_capacity,), jnp.int32),
            "evidence_counts": ((cfg.semantic_capacity,), jnp.int32),
            "evidence_means": ((cfg.semantic_capacity,), jnp.float32),
            "evidence_m2": ((cfg.semantic_capacity,), jnp.float32),
            "occupied": ((cfg.semantic_capacity,), jnp.bool_),
            "valid": ((cfg.semantic_capacity,), jnp.bool_),
            "stale": ((cfg.semantic_capacity,), jnp.bool_),
            "invalidated": ((cfg.semantic_capacity,), jnp.bool_),
        }
        procedural_shapes: dict[str, tuple[tuple[int, ...], Any]] = {
            "semantic_digests": ((cfg.procedural_capacity, 32), jnp.uint8),
            "generations": ((cfg.procedural_capacity,), jnp.int32),
            "payload_means": (
                (cfg.procedural_capacity, cfg.procedural_payload_dim),
                jnp.float32,
            ),
            "payload_m2": (
                (cfg.procedural_capacity, cfg.procedural_payload_dim),
                jnp.float32,
            ),
            "confidences": ((cfg.procedural_capacity,), jnp.float32),
            "provenance_digests": ((cfg.procedural_capacity, 32), jnp.uint8),
            "representation_revisions": ((cfg.procedural_capacity,), jnp.int32),
            "source_revisions": ((cfg.procedural_capacity,), jnp.int32),
            "creation_steps": ((cfg.procedural_capacity,), jnp.int32),
            "last_use_steps": ((cfg.procedural_capacity,), jnp.int32),
            "access_counts": ((cfg.procedural_capacity,), jnp.int32),
            "evidence_counts": ((cfg.procedural_capacity,), jnp.int32),
            "evidence_means": ((cfg.procedural_capacity,), jnp.float32),
            "evidence_m2": ((cfg.procedural_capacity,), jnp.float32),
            "success_counts": ((cfg.procedural_capacity,), jnp.int32),
            "failure_counts": ((cfg.procedural_capacity,), jnp.int32),
            "outcome_means": (
                (cfg.procedural_capacity, cfg.procedural_outcome_dim),
                jnp.float32,
            ),
            "outcome_m2": (
                (cfg.procedural_capacity, cfg.procedural_outcome_dim),
                jnp.float32,
            ),
            "lifecycle_link_available": ((cfg.procedural_capacity,), jnp.bool_),
            "lifecycle_digests": ((cfg.procedural_capacity, 32), jnp.uint8),
            "lifecycle_generations": ((cfg.procedural_capacity,), jnp.int32),
            "lifecycle_revisions": ((cfg.procedural_capacity,), jnp.int32),
            "occupied": ((cfg.procedural_capacity,), jnp.bool_),
            "valid": ((cfg.procedural_capacity,), jnp.bool_),
            "stale": ((cfg.procedural_capacity,), jnp.bool_),
            "invalidated": ((cfg.procedural_capacity,), jnp.bool_),
        }
        for name, (shape, dtype) in semantic_shapes.items():
            _require_array(
                getattr(semantic, name), name=f"state.semantic.{name}", shape=shape, dtype=dtype
            )
        for name, (shape, dtype) in procedural_shapes.items():
            _require_array(
                getattr(procedural, name),
                name=f"state.procedural.{name}",
                shape=shape,
                dtype=dtype,
            )
        _require_array(
            state.source_digest, name="state.source_digest", shape=(32,), dtype=jnp.uint8
        )
        _require_array(
            state.semantic_namespace_digest,
            name="state.semantic_namespace_digest",
            shape=(32,),
            dtype=jnp.uint8,
        )
        for field in dataclasses.fields(ConsolidatedMemoryState):
            if field.name in {
                "semantic",
                "procedural",
                "source_digest",
                "semantic_namespace_digest",
            }:
                continue
            _require_array(
                getattr(state, field.name), name=f"state.{field.name}", shape=(), dtype=jnp.int32
            )

    def _validate_semantic_record_static(self, record: SemanticMemoryRecord) -> None:
        if not isinstance(record, SemanticMemoryRecord):
            raise TypeError("record must be a SemanticMemoryRecord")
        _require_array(
            record.semantic_digest, name="record.semantic_digest", shape=(32,), dtype=jnp.uint8
        )
        _require_array(
            record.provenance_digest,
            name="record.provenance_digest",
            shape=(32,),
            dtype=jnp.uint8,
        )
        _require_array(
            record.payload,
            name="record.payload",
            shape=(self._config.semantic_payload_dim,),
            dtype=jnp.float32,
        )
        for name in ("generation", "kind", "representation_revision", "source_revision"):
            _require_array(getattr(record, name), name=f"record.{name}", shape=(), dtype=jnp.int32)
        for name in ("confidence", "evidence"):
            _require_array(
                getattr(record, name), name=f"record.{name}", shape=(), dtype=jnp.float32
            )

    def _validate_semantic_request_static(self, request: SemanticMemoryRequest) -> None:
        if not isinstance(request, SemanticMemoryRequest):
            raise TypeError("request must be a SemanticMemoryRequest")
        _require_array(
            request.semantic_digest, name="request.semantic_digest", shape=(32,), dtype=jnp.uint8
        )
        _require_array(
            request.provenance_digest,
            name="request.provenance_digest",
            shape=(32,),
            dtype=jnp.uint8,
        )
        for name in ("generation", "kind", "representation_revision", "source_revision"):
            _require_array(
                getattr(request, name), name=f"request.{name}", shape=(), dtype=jnp.int32
            )

    def _validate_procedural_record_static(self, record: ProceduralMemoryRecord) -> None:
        if not isinstance(record, ProceduralMemoryRecord):
            raise TypeError("record must be a ProceduralMemoryRecord")
        for name in ("semantic_digest", "provenance_digest", "lifecycle_digest"):
            _require_array(
                getattr(record, name), name=f"record.{name}", shape=(32,), dtype=jnp.uint8
            )
        _require_array(
            record.payload,
            name="record.payload",
            shape=(self._config.procedural_payload_dim,),
            dtype=jnp.float32,
        )
        _require_array(
            record.outcome,
            name="record.outcome",
            shape=(self._config.procedural_outcome_dim,),
            dtype=jnp.float32,
        )
        for name in (
            "generation",
            "representation_revision",
            "source_revision",
            "lifecycle_generation",
            "lifecycle_revision",
        ):
            _require_array(getattr(record, name), name=f"record.{name}", shape=(), dtype=jnp.int32)
        for name in ("confidence", "evidence"):
            _require_array(
                getattr(record, name), name=f"record.{name}", shape=(), dtype=jnp.float32
            )
        for name in ("succeeded", "lifecycle_link_available"):
            _require_array(getattr(record, name), name=f"record.{name}", shape=(), dtype=jnp.bool_)

    def _validate_procedural_request_static(self, request: ProceduralMemoryRequest) -> None:
        if not isinstance(request, ProceduralMemoryRequest):
            raise TypeError("request must be a ProceduralMemoryRequest")
        for name in ("semantic_digest", "provenance_digest", "lifecycle_digest"):
            _require_array(
                getattr(request, name), name=f"request.{name}", shape=(32,), dtype=jnp.uint8
            )
        for name in (
            "generation",
            "representation_revision",
            "source_revision",
            "lifecycle_generation",
            "lifecycle_revision",
        ):
            _require_array(
                getattr(request, name), name=f"request.{name}", shape=(), dtype=jnp.int32
            )
        _require_array(
            request.lifecycle_link_available,
            name="request.lifecycle_link_available",
            shape=(),
            dtype=jnp.bool_,
        )

    def _semantic_records_valid(self, records: SemanticMemoryRecords, step: Array) -> Array:
        occupied = records.occupied
        empty = ~occupied
        active_metadata = (
            (records.generations >= 0)
            & (records.kinds >= SEMANTIC_KIND_GVF)
            & (records.kinds <= SEMANTIC_KIND_AFFORDANCE)
            & (records.representation_revisions >= 0)
            & (records.source_revisions >= 0)
            & (records.creation_steps >= 0)
            & (records.creation_steps <= records.last_use_steps)
            & (records.last_use_steps <= step)
            & (records.access_counts >= 0)
            & (records.evidence_counts >= 1)
            & _digest_is_nonzero(records.semantic_digests)
            & _digest_is_nonzero(records.provenance_digests)
        )
        empty_metadata = (
            (records.generations == -1)
            & (records.kinds == -1)
            & (records.representation_revisions == -1)
            & (records.source_revisions == -1)
            & (records.creation_steps == 0)
            & (records.last_use_steps == 0)
            & (records.access_counts == 0)
            & (records.evidence_counts == 0)
            & (~_digest_is_nonzero(records.semantic_digests))
            & (~_digest_is_nonzero(records.provenance_digests))
            & (~records.valid)
            & (~records.stale)
            & (~records.invalidated)
        )
        finite = (
            jnp.all(jnp.isfinite(records.payload_means))
            & jnp.all(jnp.isfinite(records.payload_m2))
            & jnp.all(jnp.isfinite(records.confidences))
            & jnp.all(jnp.isfinite(records.evidence_means))
            & jnp.all(jnp.isfinite(records.evidence_m2))
        )
        ranges = (
            jnp.all((records.confidences >= 0.0) & (records.confidences <= 1.0))
            & jnp.all(records.payload_m2 >= -1.0e-6)
            & jnp.all(records.evidence_m2 >= -1.0e-6)
            & jnp.all(~records.invalidated | ~records.valid)
            & jnp.all(records.valid | records.stale | records.invalidated | ~occupied)
        )
        same_digest = jnp.all(
            records.semantic_digests[:, None, :] == records.semantic_digests[None, :, :],
            axis=-1,
        )
        distinct = ~jnp.eye(self._config.semantic_capacity, dtype=jnp.bool_)
        duplicates = same_digest & occupied[:, None] & occupied[None, :] & distinct
        return (
            finite
            & ranges
            & jnp.all((~occupied) | active_metadata)
            & jnp.all((~empty) | empty_metadata)
            & (~jnp.any(duplicates))
        )

    def _procedural_records_valid(self, records: ProceduralMemoryRecords, step: Array) -> Array:
        occupied = records.occupied
        empty = ~occupied
        link_shape_valid = jnp.where(
            records.lifecycle_link_available,
            _digest_is_nonzero(records.lifecycle_digests)
            & (records.lifecycle_generations >= 0)
            & (records.lifecycle_revisions >= 0),
            (~_digest_is_nonzero(records.lifecycle_digests))
            & (records.lifecycle_generations == -1)
            & (records.lifecycle_revisions == -1),
        )
        active_metadata = (
            (records.generations >= 0)
            & (records.representation_revisions >= 0)
            & (records.source_revisions >= 0)
            & (records.creation_steps >= 0)
            & (records.creation_steps <= records.last_use_steps)
            & (records.last_use_steps <= step)
            & (records.access_counts >= 0)
            & (records.evidence_counts >= 1)
            & (records.success_counts >= 0)
            & (records.failure_counts >= 0)
            & (records.success_counts + records.failure_counts == records.evidence_counts)
            & _digest_is_nonzero(records.semantic_digests)
            & _digest_is_nonzero(records.provenance_digests)
            & link_shape_valid
        )
        empty_metadata = (
            (records.generations == -1)
            & (records.representation_revisions == -1)
            & (records.source_revisions == -1)
            & (records.creation_steps == 0)
            & (records.last_use_steps == 0)
            & (records.access_counts == 0)
            & (records.evidence_counts == 0)
            & (records.success_counts == 0)
            & (records.failure_counts == 0)
            & (~_digest_is_nonzero(records.semantic_digests))
            & (~_digest_is_nonzero(records.provenance_digests))
            & (~records.lifecycle_link_available)
            & (~_digest_is_nonzero(records.lifecycle_digests))
            & (records.lifecycle_generations == -1)
            & (records.lifecycle_revisions == -1)
            & (~records.valid)
            & (~records.stale)
            & (~records.invalidated)
        )
        finite = (
            jnp.all(jnp.isfinite(records.payload_means))
            & jnp.all(jnp.isfinite(records.payload_m2))
            & jnp.all(jnp.isfinite(records.confidences))
            & jnp.all(jnp.isfinite(records.evidence_means))
            & jnp.all(jnp.isfinite(records.evidence_m2))
            & jnp.all(jnp.isfinite(records.outcome_means))
            & jnp.all(jnp.isfinite(records.outcome_m2))
        )
        ranges = (
            jnp.all((records.confidences >= 0.0) & (records.confidences <= 1.0))
            & jnp.all(records.payload_m2 >= -1.0e-6)
            & jnp.all(records.evidence_m2 >= -1.0e-6)
            & jnp.all(records.outcome_m2 >= -1.0e-6)
            & jnp.all(~records.invalidated | ~records.valid)
            & jnp.all(records.valid | records.stale | records.invalidated | ~occupied)
        )
        same_digest = jnp.all(
            records.semantic_digests[:, None, :] == records.semantic_digests[None, :, :],
            axis=-1,
        )
        distinct = ~jnp.eye(self._config.procedural_capacity, dtype=jnp.bool_)
        duplicates = same_digest & occupied[:, None] & occupied[None, :] & distinct
        return (
            finite
            & ranges
            & jnp.all((~occupied) | active_metadata)
            & jnp.all((~empty) | empty_metadata)
            & (~jnp.any(duplicates))
        )

    def _state_is_valid(
        self,
        state: ConsolidatedMemoryState,
        *,
        source_digest: Array,
        semantic_namespace_digest: Array,
        representation_revision: Array,
        source_revision: Array,
    ) -> Array:
        counter_names = tuple(
            field.name
            for field in dataclasses.fields(ConsolidatedMemoryState)
            if field.name
            not in {
                "semantic",
                "procedural",
                "source_digest",
                "semantic_namespace_digest",
                "representation_revision",
                "source_revision",
            }
        )
        counters_valid = jnp.asarray(True, dtype=jnp.bool_)
        for name in counter_names:
            value = getattr(state, name)
            counters_valid = counters_valid & (value >= 0) & (value <= state.operation_count)
        return (
            self._semantic_records_valid(state.semantic, state.operation_count)
            & self._procedural_records_valid(state.procedural, state.operation_count)
            & counters_valid
            & (state.operation_count <= self._config.max_operations)
            & _digest_is_nonzero(state.source_digest)
            & _digest_is_nonzero(state.semantic_namespace_digest)
            & jnp.array_equal(state.source_digest, source_digest)
            & jnp.array_equal(state.semantic_namespace_digest, semantic_namespace_digest)
            & (state.representation_revision == representation_revision)
            & (state.source_revision == source_revision)
            & (state.representation_revision >= 0)
            & (state.source_revision >= 0)
        )

    def validate_state(
        self,
        state: ConsolidatedMemoryState,
        *,
        source_digest: Array,
        semantic_namespace_digest: Array,
        representation_revision: int | Array,
        source_revision: int | Array,
    ) -> Array:
        """Return whether all structure, bindings, and dynamic invariants hold."""

        self._validate_state_static_contract(state)
        _require_array(source_digest, name="source_digest", shape=(32,), dtype=jnp.uint8)
        _require_array(
            semantic_namespace_digest,
            name="semantic_namespace_digest",
            shape=(32,),
            dtype=jnp.uint8,
        )
        representation = _int32_scalar(representation_revision, name="representation_revision")
        source = _int32_scalar(source_revision, name="source_revision")
        return self._state_is_valid(
            state,
            source_digest=source_digest,
            semantic_namespace_digest=semantic_namespace_digest,
            representation_revision=representation,
            source_revision=source,
        )

    def _semantic_record_valid(
        self, state: ConsolidatedMemoryState, record: SemanticMemoryRecord
    ) -> Array:
        return (
            _digest_is_nonzero(record.semantic_digest)
            & _digest_is_nonzero(record.provenance_digest)
            & (record.generation >= 0)
            & (record.kind >= SEMANTIC_KIND_GVF)
            & (record.kind <= SEMANTIC_KIND_AFFORDANCE)
            & jnp.all(jnp.isfinite(record.payload))
            & jnp.isfinite(record.confidence)
            & (record.confidence >= 0.0)
            & (record.confidence <= 1.0)
            & jnp.isfinite(record.evidence)
            & (record.representation_revision == state.representation_revision)
            & (record.source_revision == state.source_revision)
        )

    def _semantic_request_valid(
        self, state: ConsolidatedMemoryState, request: SemanticMemoryRequest
    ) -> Array:
        return (
            _digest_is_nonzero(request.semantic_digest)
            & _digest_is_nonzero(request.provenance_digest)
            & (request.generation >= 0)
            & (request.kind >= SEMANTIC_KIND_GVF)
            & (request.kind <= SEMANTIC_KIND_AFFORDANCE)
            & (request.representation_revision == state.representation_revision)
            & (request.source_revision == state.source_revision)
        )

    @staticmethod
    def _link_valid(available: Array, digest: Array, generation: Array, revision: Array) -> Array:
        return jnp.where(
            available,
            _digest_is_nonzero(digest) & (generation >= 0) & (revision >= 0),
            (~_digest_is_nonzero(digest)) & (generation == -1) & (revision == -1),
        )

    def _procedural_record_valid(
        self, state: ConsolidatedMemoryState, record: ProceduralMemoryRecord
    ) -> Array:
        return (
            _digest_is_nonzero(record.semantic_digest)
            & _digest_is_nonzero(record.provenance_digest)
            & (record.generation >= 0)
            & jnp.all(jnp.isfinite(record.payload))
            & jnp.all(jnp.isfinite(record.outcome))
            & jnp.isfinite(record.confidence)
            & (record.confidence >= 0.0)
            & (record.confidence <= 1.0)
            & jnp.isfinite(record.evidence)
            & (record.representation_revision == state.representation_revision)
            & (record.source_revision == state.source_revision)
            & self._link_valid(
                record.lifecycle_link_available,
                record.lifecycle_digest,
                record.lifecycle_generation,
                record.lifecycle_revision,
            )
        )

    def _procedural_request_valid(
        self, state: ConsolidatedMemoryState, request: ProceduralMemoryRequest
    ) -> Array:
        return (
            _digest_is_nonzero(request.semantic_digest)
            & _digest_is_nonzero(request.provenance_digest)
            & (request.generation >= 0)
            & (request.representation_revision == state.representation_revision)
            & (request.source_revision == state.source_revision)
            & self._link_valid(
                request.lifecycle_link_available,
                request.lifecycle_digest,
                request.lifecycle_generation,
                request.lifecycle_revision,
            )
        )

    def _refresh_stale(
        self, state: ConsolidatedMemoryState, next_step: Array
    ) -> ConsolidatedMemoryState:
        semantic = state.semantic
        procedural = state.procedural
        new_semantic_stale = (
            semantic.occupied
            & semantic.valid
            & (~semantic.stale)
            & (~semantic.invalidated)
            & (
                next_step - semantic.last_use_steps
                > jnp.asarray(self._config.semantic_max_age, dtype=jnp.int32)
            )
        )
        new_procedural_stale = (
            procedural.occupied
            & procedural.valid
            & (~procedural.stale)
            & (~procedural.invalidated)
            & (
                next_step - procedural.last_use_steps
                > jnp.asarray(self._config.procedural_max_age, dtype=jnp.int32)
            )
        )
        return dataclasses.replace(
            state,
            semantic=dataclasses.replace(semantic, stale=semantic.stale | new_semantic_stale),
            procedural=dataclasses.replace(
                procedural, stale=procedural.stale | new_procedural_stale
            ),
            semantic_retirement_count=(
                state.semantic_retirement_count + jnp.sum(new_semantic_stale.astype(jnp.int32))
            ),
            procedural_retirement_count=(
                state.procedural_retirement_count + jnp.sum(new_procedural_stale.astype(jnp.int32))
            ),
        )

    def _semantic_retrieval(
        self,
        state: ConsolidatedMemoryState,
        request: SemanticMemoryRequest,
        *,
        state_valid: Array,
        request_valid: Array,
        transaction_applied: Array,
        access_step: Array,
    ) -> tuple[ConsolidatedMemoryState, SemanticMemoryRetrieval]:
        records = state.semantic
        identity = records.occupied & jnp.all(
            records.semantic_digests == request.semantic_digest[None, :], axis=1
        )
        found = jnp.any(identity)
        compatible_mask = (
            identity
            & (records.generations == request.generation)
            & (records.kinds == request.kind)
            & jnp.all(records.provenance_digests == request.provenance_digest[None, :], axis=1)
            & (records.representation_revisions == request.representation_revision)
            & (records.source_revisions == request.source_revision)
        )
        compatible = jnp.any(compatible_mask)
        chosen = _first_true(compatible_mask)
        fresh = (
            compatible
            & records.valid[chosen]
            & (~records.stale[chosen])
            & (~records.invalidated[chosen])
        )
        confidence_ok = compatible & (
            records.confidences[chosen]
            >= jnp.asarray(self._config.semantic_min_confidence, dtype=jnp.float32)
        )
        accepted = transaction_applied & state_valid & request_valid & fresh & confidence_ok
        safe_slot = jnp.where(compatible, chosen, jnp.asarray(0, dtype=jnp.int32))
        updated_records = dataclasses.replace(
            records,
            last_use_steps=records.last_use_steps.at[safe_slot].set(
                jnp.where(accepted, access_step, records.last_use_steps[safe_slot])
            ),
            access_counts=records.access_counts.at[safe_slot].set(
                records.access_counts[safe_slot] + accepted.astype(jnp.int32)
            ),
        )
        updated = dataclasses.replace(state, semantic=updated_records)
        zeros = jnp.zeros((self._config.semantic_payload_dim,), dtype=jnp.float32)
        retrieval = SemanticMemoryRetrieval(
            accepted=accepted,
            transaction_applied=transaction_applied,
            slot=jnp.where(accepted, chosen, jnp.asarray(-1, dtype=jnp.int32)),
            payload=jnp.where(accepted, records.payload_means[chosen], zeros),
            confidence=jnp.where(accepted, records.confidences[chosen], 0.0),
            evidence_count=jnp.where(accepted, records.evidence_counts[chosen], 0),
            evidence_mean=jnp.where(accepted, records.evidence_means[chosen], 0.0),
            evidence_m2=jnp.where(accepted, records.evidence_m2[chosen], 0.0),
            state_valid=state_valid,
            request_valid=request_valid,
            identity_found=found,
            compatible=compatible,
            fresh=fresh,
            confidence_ok=confidence_ok,
        )
        return updated, retrieval

    def _procedural_retrieval(
        self,
        state: ConsolidatedMemoryState,
        request: ProceduralMemoryRequest,
        *,
        state_valid: Array,
        request_valid: Array,
        transaction_applied: Array,
        access_step: Array,
    ) -> tuple[ConsolidatedMemoryState, ProceduralMemoryRetrieval]:
        records = state.procedural
        identity = records.occupied & jnp.all(
            records.semantic_digests == request.semantic_digest[None, :], axis=1
        )
        found = jnp.any(identity)
        compatible_mask = (
            identity
            & (records.generations == request.generation)
            & jnp.all(records.provenance_digests == request.provenance_digest[None, :], axis=1)
            & (records.representation_revisions == request.representation_revision)
            & (records.source_revisions == request.source_revision)
            & (records.lifecycle_link_available == request.lifecycle_link_available)
            & jnp.all(records.lifecycle_digests == request.lifecycle_digest[None, :], axis=1)
            & (records.lifecycle_generations == request.lifecycle_generation)
            & (records.lifecycle_revisions == request.lifecycle_revision)
        )
        compatible = jnp.any(compatible_mask)
        chosen = _first_true(compatible_mask)
        fresh = (
            compatible
            & records.valid[chosen]
            & (~records.stale[chosen])
            & (~records.invalidated[chosen])
        )
        confidence_ok = compatible & (
            records.confidences[chosen]
            >= jnp.asarray(self._config.procedural_min_confidence, dtype=jnp.float32)
        )
        accepted = transaction_applied & state_valid & request_valid & fresh & confidence_ok
        safe_slot = jnp.where(compatible, chosen, jnp.asarray(0, dtype=jnp.int32))
        updated_records = dataclasses.replace(
            records,
            last_use_steps=records.last_use_steps.at[safe_slot].set(
                jnp.where(accepted, access_step, records.last_use_steps[safe_slot])
            ),
            access_counts=records.access_counts.at[safe_slot].set(
                records.access_counts[safe_slot] + accepted.astype(jnp.int32)
            ),
        )
        updated = dataclasses.replace(state, procedural=updated_records)
        payload_zeros = jnp.zeros((self._config.procedural_payload_dim,), dtype=jnp.float32)
        outcome_zeros = jnp.zeros((self._config.procedural_outcome_dim,), dtype=jnp.float32)
        digest_zeros = jnp.zeros((_DIGEST_BYTES,), dtype=jnp.uint8)
        retrieval = ProceduralMemoryRetrieval(
            accepted=accepted,
            transaction_applied=transaction_applied,
            slot=jnp.where(accepted, chosen, jnp.asarray(-1, dtype=jnp.int32)),
            payload=jnp.where(accepted, records.payload_means[chosen], payload_zeros),
            confidence=jnp.where(accepted, records.confidences[chosen], 0.0),
            evidence_count=jnp.where(accepted, records.evidence_counts[chosen], 0),
            evidence_mean=jnp.where(accepted, records.evidence_means[chosen], 0.0),
            evidence_m2=jnp.where(accepted, records.evidence_m2[chosen], 0.0),
            success_count=jnp.where(accepted, records.success_counts[chosen], 0),
            failure_count=jnp.where(accepted, records.failure_counts[chosen], 0),
            outcome_mean=jnp.where(accepted, records.outcome_means[chosen], outcome_zeros),
            outcome_m2=jnp.where(accepted, records.outcome_m2[chosen], outcome_zeros),
            lifecycle_link_available=jnp.where(
                accepted, records.lifecycle_link_available[chosen], False
            ),
            lifecycle_digest=jnp.where(accepted, records.lifecycle_digests[chosen], digest_zeros),
            lifecycle_generation=jnp.where(accepted, records.lifecycle_generations[chosen], -1),
            lifecycle_revision=jnp.where(accepted, records.lifecycle_revisions[chosen], -1),
            state_valid=state_valid,
            request_valid=request_valid,
            identity_found=found,
            compatible=compatible,
            fresh=fresh,
            confidence_ok=confidence_ok,
        )
        return updated, retrieval

    @staticmethod
    def _choose_slot(
        occupied: Array,
        valid: Array,
        stale: Array,
        invalidated: Array,
        confidences: Array,
        last_use_steps: Array,
        evidence_counts: Array,
    ) -> Array:
        """Choose free, then retired, then least-retained; all ties use lowest slot."""

        free = ~occupied
        retired = occupied & ((~valid) | stale | invalidated)
        has_free = jnp.any(free)
        has_retired = jnp.any(retired)
        active = occupied & valid & (~stale) & (~invalidated)
        min_confidence = jnp.min(jnp.where(active, confidences, jnp.inf))
        priority = active & (confidences == min_confidence)
        oldest = jnp.min(
            jnp.where(priority, last_use_steps, jnp.asarray(_INT32_MAX, dtype=jnp.int32))
        )
        priority = priority & (last_use_steps == oldest)
        least_evidence = jnp.min(
            jnp.where(priority, evidence_counts, jnp.asarray(_INT32_MAX, dtype=jnp.int32))
        )
        priority = priority & (evidence_counts == least_evidence)
        return jnp.where(
            has_free,
            _first_true(free),
            jnp.where(has_retired, _first_true(retired), _first_true(priority)),
        ).astype(jnp.int32)

    def _semantic_write(
        self,
        state: ConsolidatedMemoryState,
        record: SemanticMemoryRecord,
        *,
        write_step: Array,
        state_valid: Array,
        record_valid: Array,
        operation_available: Array,
    ) -> tuple[ConsolidatedMemoryState, MemoryWriteDiagnostics]:
        records = state.semantic
        identity = records.occupied & jnp.all(
            records.semantic_digests == record.semantic_digest[None, :], axis=1
        )
        found = jnp.any(identity)
        identity_slot = _first_true(identity)
        selected_slot = self._choose_slot(
            records.occupied,
            records.valid,
            records.stale,
            records.invalidated,
            records.confidences,
            records.last_use_steps,
            records.evidence_counts,
        )
        slot = jnp.where(found, identity_slot, selected_slot)
        same_generation = found & (record.generation == records.generations[identity_slot])
        next_generation = found & (record.generation == records.generations[identity_slot] + 1)
        same_kind = (~found) | (record.kind == records.kinds[identity_slot])
        same_metadata = (
            same_kind
            & jnp.all(records.provenance_digests[identity_slot] == record.provenance_digest)
            & (records.representation_revisions[identity_slot] == record.representation_revision)
            & (records.source_revisions[identity_slot] == record.source_revision)
        )
        active_existing = (
            records.valid[identity_slot]
            & (~records.stale[identity_slot])
            & (~records.invalidated[identity_slot])
        )
        merged = found & same_generation & same_metadata & active_existing
        revised = found & next_generation & same_kind
        inserted = ~found
        writable = inserted | merged | revised
        wrote = state_valid & record_valid & operation_available & writable
        reset = inserted | revised
        old_count = records.evidence_counts[slot]
        merged_payload, merged_payload_m2 = _welford_vector(
            records.payload_means[slot], records.payload_m2[slot], old_count, record.payload
        )
        merged_evidence, merged_evidence_m2 = _welford_scalar(
            records.evidence_means[slot], records.evidence_m2[slot], old_count, record.evidence
        )
        merged_confidence = records.confidences[slot] + (
            record.confidence - records.confidences[slot]
        ) / (old_count + 1).astype(jnp.float32)
        next_payload = jnp.where(merged, merged_payload, record.payload)
        next_payload_m2 = jnp.where(merged, merged_payload_m2, jnp.zeros_like(record.payload))
        next_evidence = jnp.where(merged, merged_evidence, record.evidence)
        next_evidence_m2 = jnp.where(merged, merged_evidence_m2, 0.0)
        next_confidence = jnp.where(merged, merged_confidence, record.confidence)
        next_count = jnp.where(merged, old_count + 1, 1).astype(jnp.int32)
        next_creation = jnp.where(merged, records.creation_steps[slot], write_step)
        replacement = inserted & records.occupied[slot]

        def set_value(array: Array, value: Array) -> Array:
            return array.at[slot].set(jnp.where(wrote, value, array[slot]))

        updated_records = SemanticMemoryRecords(
            semantic_digests=set_value(records.semantic_digests, record.semantic_digest),
            generations=set_value(records.generations, record.generation),
            kinds=set_value(records.kinds, record.kind),
            payload_means=set_value(records.payload_means, next_payload),
            payload_m2=set_value(records.payload_m2, next_payload_m2),
            confidences=set_value(records.confidences, next_confidence),
            provenance_digests=set_value(records.provenance_digests, record.provenance_digest),
            representation_revisions=set_value(
                records.representation_revisions, record.representation_revision
            ),
            source_revisions=set_value(records.source_revisions, record.source_revision),
            creation_steps=set_value(records.creation_steps, next_creation),
            last_use_steps=set_value(records.last_use_steps, write_step),
            access_counts=set_value(
                records.access_counts,
                jnp.where(merged, records.access_counts[slot], 0),
            ),
            evidence_counts=set_value(records.evidence_counts, next_count),
            evidence_means=set_value(records.evidence_means, next_evidence),
            evidence_m2=set_value(records.evidence_m2, next_evidence_m2),
            occupied=set_value(records.occupied, jnp.asarray(True, dtype=jnp.bool_)),
            valid=set_value(records.valid, jnp.asarray(True, dtype=jnp.bool_)),
            stale=set_value(records.stale, jnp.asarray(False, dtype=jnp.bool_)),
            invalidated=set_value(records.invalidated, jnp.asarray(False, dtype=jnp.bool_)),
        )
        updated = dataclasses.replace(
            state,
            semantic=updated_records,
            semantic_write_count=state.semantic_write_count + wrote.astype(jnp.int32),
            semantic_merge_count=state.semantic_merge_count + (wrote & merged).astype(jnp.int32),
            semantic_revision_count=state.semantic_revision_count
            + (wrote & revised).astype(jnp.int32),
            semantic_replacement_count=state.semantic_replacement_count
            + (wrote & replacement).astype(jnp.int32),
        )
        diagnostics = MemoryWriteDiagnostics(
            transaction_applied=wrote,
            wrote=wrote,
            merged=wrote & merged,
            revised=wrote & revised,
            replaced=wrote & replacement,
            reset_evidence=wrote & reset,
            slot=jnp.where(wrote, slot, -1),
            state_valid=state_valid,
            record_valid=record_valid,
            identity_collision=found & (~writable),
            generation_compatible=(~found) | same_generation | next_generation,
            metadata_compatible=(~found) | revised | same_metadata,
        )
        return updated, diagnostics

    def _procedural_write(
        self,
        state: ConsolidatedMemoryState,
        record: ProceduralMemoryRecord,
        *,
        write_step: Array,
        state_valid: Array,
        record_valid: Array,
        operation_available: Array,
    ) -> tuple[ConsolidatedMemoryState, MemoryWriteDiagnostics]:
        records = state.procedural
        identity = records.occupied & jnp.all(
            records.semantic_digests == record.semantic_digest[None, :], axis=1
        )
        found = jnp.any(identity)
        identity_slot = _first_true(identity)
        selected_slot = self._choose_slot(
            records.occupied,
            records.valid,
            records.stale,
            records.invalidated,
            records.confidences,
            records.last_use_steps,
            records.evidence_counts,
        )
        slot = jnp.where(found, identity_slot, selected_slot)
        same_generation = found & (record.generation == records.generations[identity_slot])
        next_generation = found & (record.generation == records.generations[identity_slot] + 1)
        same_metadata = (
            jnp.all(records.provenance_digests[identity_slot] == record.provenance_digest)
            & (records.representation_revisions[identity_slot] == record.representation_revision)
            & (records.source_revisions[identity_slot] == record.source_revision)
            & (records.lifecycle_link_available[identity_slot] == record.lifecycle_link_available)
            & jnp.all(records.lifecycle_digests[identity_slot] == record.lifecycle_digest)
            & (records.lifecycle_generations[identity_slot] == record.lifecycle_generation)
            & (records.lifecycle_revisions[identity_slot] == record.lifecycle_revision)
        )
        active_existing = (
            records.valid[identity_slot]
            & (~records.stale[identity_slot])
            & (~records.invalidated[identity_slot])
        )
        merged = found & same_generation & same_metadata & active_existing
        revised = found & next_generation
        inserted = ~found
        writable = inserted | merged | revised
        wrote = state_valid & record_valid & operation_available & writable
        reset = inserted | revised
        old_count = records.evidence_counts[slot]
        merged_payload, merged_payload_m2 = _welford_vector(
            records.payload_means[slot], records.payload_m2[slot], old_count, record.payload
        )
        merged_evidence, merged_evidence_m2 = _welford_scalar(
            records.evidence_means[slot], records.evidence_m2[slot], old_count, record.evidence
        )
        merged_outcome, merged_outcome_m2 = _welford_vector(
            records.outcome_means[slot], records.outcome_m2[slot], old_count, record.outcome
        )
        merged_confidence = records.confidences[slot] + (
            record.confidence - records.confidences[slot]
        ) / (old_count + 1).astype(jnp.float32)
        next_payload = jnp.where(merged, merged_payload, record.payload)
        next_payload_m2 = jnp.where(merged, merged_payload_m2, jnp.zeros_like(record.payload))
        next_evidence = jnp.where(merged, merged_evidence, record.evidence)
        next_evidence_m2 = jnp.where(merged, merged_evidence_m2, 0.0)
        next_outcome = jnp.where(merged, merged_outcome, record.outcome)
        next_outcome_m2 = jnp.where(merged, merged_outcome_m2, jnp.zeros_like(record.outcome))
        next_confidence = jnp.where(merged, merged_confidence, record.confidence)
        next_count = jnp.where(merged, old_count + 1, 1).astype(jnp.int32)
        next_successes = jnp.where(
            merged, records.success_counts[slot], 0
        ) + record.succeeded.astype(jnp.int32)
        next_failures = jnp.where(merged, records.failure_counts[slot], 0) + (
            ~record.succeeded
        ).astype(jnp.int32)
        next_creation = jnp.where(merged, records.creation_steps[slot], write_step)
        replacement = inserted & records.occupied[slot]

        def set_value(array: Array, value: Array) -> Array:
            return array.at[slot].set(jnp.where(wrote, value, array[slot]))

        updated_records = ProceduralMemoryRecords(
            semantic_digests=set_value(records.semantic_digests, record.semantic_digest),
            generations=set_value(records.generations, record.generation),
            payload_means=set_value(records.payload_means, next_payload),
            payload_m2=set_value(records.payload_m2, next_payload_m2),
            confidences=set_value(records.confidences, next_confidence),
            provenance_digests=set_value(records.provenance_digests, record.provenance_digest),
            representation_revisions=set_value(
                records.representation_revisions, record.representation_revision
            ),
            source_revisions=set_value(records.source_revisions, record.source_revision),
            creation_steps=set_value(records.creation_steps, next_creation),
            last_use_steps=set_value(records.last_use_steps, write_step),
            access_counts=set_value(
                records.access_counts,
                jnp.where(merged, records.access_counts[slot], 0),
            ),
            evidence_counts=set_value(records.evidence_counts, next_count),
            evidence_means=set_value(records.evidence_means, next_evidence),
            evidence_m2=set_value(records.evidence_m2, next_evidence_m2),
            success_counts=set_value(records.success_counts, next_successes),
            failure_counts=set_value(records.failure_counts, next_failures),
            outcome_means=set_value(records.outcome_means, next_outcome),
            outcome_m2=set_value(records.outcome_m2, next_outcome_m2),
            lifecycle_link_available=set_value(
                records.lifecycle_link_available, record.lifecycle_link_available
            ),
            lifecycle_digests=set_value(records.lifecycle_digests, record.lifecycle_digest),
            lifecycle_generations=set_value(
                records.lifecycle_generations, record.lifecycle_generation
            ),
            lifecycle_revisions=set_value(records.lifecycle_revisions, record.lifecycle_revision),
            occupied=set_value(records.occupied, jnp.asarray(True, dtype=jnp.bool_)),
            valid=set_value(records.valid, jnp.asarray(True, dtype=jnp.bool_)),
            stale=set_value(records.stale, jnp.asarray(False, dtype=jnp.bool_)),
            invalidated=set_value(records.invalidated, jnp.asarray(False, dtype=jnp.bool_)),
        )
        updated = dataclasses.replace(
            state,
            procedural=updated_records,
            procedural_write_count=state.procedural_write_count + wrote.astype(jnp.int32),
            procedural_merge_count=state.procedural_merge_count
            + (wrote & merged).astype(jnp.int32),
            procedural_revision_count=state.procedural_revision_count
            + (wrote & revised).astype(jnp.int32),
            procedural_replacement_count=state.procedural_replacement_count
            + (wrote & replacement).astype(jnp.int32),
        )
        diagnostics = MemoryWriteDiagnostics(
            transaction_applied=wrote,
            wrote=wrote,
            merged=wrote & merged,
            revised=wrote & revised,
            replaced=wrote & replacement,
            reset_evidence=wrote & reset,
            slot=jnp.where(wrote, slot, -1),
            state_valid=state_valid,
            record_valid=record_valid,
            identity_collision=found & (~writable),
            generation_compatible=(~found) | same_generation | next_generation,
            metadata_compatible=(~found) | revised | same_metadata,
        )
        return updated, diagnostics

    def query_semantic(
        self, state: ConsolidatedMemoryState, request: SemanticMemoryRequest
    ) -> tuple[ConsolidatedMemoryState, SemanticMemoryRetrieval]:
        """Query one semantic identity and record an accepted use."""

        self._validate_state_static_contract(state)
        self._validate_semantic_request_static(request)
        return cast(
            tuple[ConsolidatedMemoryState, SemanticMemoryRetrieval],
            self._query_semantic_jit(state, request),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def _query_semantic_jit(
        self, state: ConsolidatedMemoryState, request: SemanticMemoryRequest
    ) -> tuple[ConsolidatedMemoryState, SemanticMemoryRetrieval]:
        state_valid = self._state_is_valid(
            state,
            source_digest=state.source_digest,
            semantic_namespace_digest=state.semantic_namespace_digest,
            representation_revision=state.representation_revision,
            source_revision=state.source_revision,
        )
        request_valid = self._semantic_request_valid(state, request)
        available = state.operation_count < self._config.max_operations
        transaction = state_valid & request_valid & available
        next_step = state.operation_count + jnp.asarray(1, dtype=jnp.int32)
        refreshed = self._refresh_stale(state, next_step)
        queried, retrieval = self._semantic_retrieval(
            refreshed,
            request,
            state_valid=state_valid,
            request_valid=request_valid,
            transaction_applied=transaction,
            access_step=next_step,
        )
        committed = dataclasses.replace(
            queried,
            operation_count=next_step,
            semantic_query_count=state.semantic_query_count + 1,
            semantic_accepted_query_count=(
                state.semantic_accepted_query_count + retrieval.accepted.astype(jnp.int32)
            ),
        )
        final_state = jax.lax.cond(transaction, lambda: committed, lambda: state)
        return final_state, retrieval

    def query_procedural(
        self, state: ConsolidatedMemoryState, request: ProceduralMemoryRequest
    ) -> tuple[ConsolidatedMemoryState, ProceduralMemoryRetrieval]:
        """Query one procedural identity and record an accepted use."""

        self._validate_state_static_contract(state)
        self._validate_procedural_request_static(request)
        return cast(
            tuple[ConsolidatedMemoryState, ProceduralMemoryRetrieval],
            self._query_procedural_jit(state, request),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def _query_procedural_jit(
        self, state: ConsolidatedMemoryState, request: ProceduralMemoryRequest
    ) -> tuple[ConsolidatedMemoryState, ProceduralMemoryRetrieval]:
        state_valid = self._state_is_valid(
            state,
            source_digest=state.source_digest,
            semantic_namespace_digest=state.semantic_namespace_digest,
            representation_revision=state.representation_revision,
            source_revision=state.source_revision,
        )
        request_valid = self._procedural_request_valid(state, request)
        available = state.operation_count < self._config.max_operations
        transaction = state_valid & request_valid & available
        next_step = state.operation_count + jnp.asarray(1, dtype=jnp.int32)
        refreshed = self._refresh_stale(state, next_step)
        queried, retrieval = self._procedural_retrieval(
            refreshed,
            request,
            state_valid=state_valid,
            request_valid=request_valid,
            transaction_applied=transaction,
            access_step=next_step,
        )
        committed = dataclasses.replace(
            queried,
            operation_count=next_step,
            procedural_query_count=state.procedural_query_count + 1,
            procedural_accepted_query_count=(
                state.procedural_accepted_query_count + retrieval.accepted.astype(jnp.int32)
            ),
        )
        final_state = jax.lax.cond(transaction, lambda: committed, lambda: state)
        return final_state, retrieval

    def write_semantic(
        self, state: ConsolidatedMemoryState, record: SemanticMemoryRecord
    ) -> tuple[ConsolidatedMemoryState, MemoryWriteDiagnostics]:
        """Perform one bounded semantic insert, merge, or explicit revision."""

        self._validate_state_static_contract(state)
        self._validate_semantic_record_static(record)
        return cast(
            tuple[ConsolidatedMemoryState, MemoryWriteDiagnostics],
            self._write_semantic_jit(state, record),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def _write_semantic_jit(
        self, state: ConsolidatedMemoryState, record: SemanticMemoryRecord
    ) -> tuple[ConsolidatedMemoryState, MemoryWriteDiagnostics]:
        state_valid = self._state_is_valid(
            state,
            source_digest=state.source_digest,
            semantic_namespace_digest=state.semantic_namespace_digest,
            representation_revision=state.representation_revision,
            source_revision=state.source_revision,
        )
        record_valid = self._semantic_record_valid(state, record)
        available = state.operation_count < self._config.max_operations
        next_step = state.operation_count + jnp.asarray(1, dtype=jnp.int32)
        refreshed = self._refresh_stale(state, next_step)
        written, diagnostics = self._semantic_write(
            refreshed,
            record,
            write_step=next_step,
            state_valid=state_valid,
            record_valid=record_valid,
            operation_available=available,
        )
        committed = dataclasses.replace(written, operation_count=next_step)
        final_state = jax.lax.cond(diagnostics.wrote, lambda: committed, lambda: state)
        return final_state, diagnostics

    def write_procedural(
        self, state: ConsolidatedMemoryState, record: ProceduralMemoryRecord
    ) -> tuple[ConsolidatedMemoryState, MemoryWriteDiagnostics]:
        """Perform one bounded procedural insert, merge, or explicit revision."""

        self._validate_state_static_contract(state)
        self._validate_procedural_record_static(record)
        return cast(
            tuple[ConsolidatedMemoryState, MemoryWriteDiagnostics],
            self._write_procedural_jit(state, record),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def _write_procedural_jit(
        self, state: ConsolidatedMemoryState, record: ProceduralMemoryRecord
    ) -> tuple[ConsolidatedMemoryState, MemoryWriteDiagnostics]:
        state_valid = self._state_is_valid(
            state,
            source_digest=state.source_digest,
            semantic_namespace_digest=state.semantic_namespace_digest,
            representation_revision=state.representation_revision,
            source_revision=state.source_revision,
        )
        record_valid = self._procedural_record_valid(state, record)
        available = state.operation_count < self._config.max_operations
        next_step = state.operation_count + jnp.asarray(1, dtype=jnp.int32)
        refreshed = self._refresh_stale(state, next_step)
        written, diagnostics = self._procedural_write(
            refreshed,
            record,
            write_step=next_step,
            state_valid=state_valid,
            record_valid=record_valid,
            operation_available=available,
        )
        committed = dataclasses.replace(written, operation_count=next_step)
        final_state = jax.lax.cond(diagnostics.wrote, lambda: committed, lambda: state)
        return final_state, diagnostics

    def semantic_step(
        self,
        state: ConsolidatedMemoryState,
        request: SemanticMemoryRequest,
        record: SemanticMemoryRecord,
    ) -> SemanticMemoryStepResult:
        """Causally query the pre-transition state, then consolidate the record."""

        self._validate_state_static_contract(state)
        self._validate_semantic_request_static(request)
        self._validate_semantic_record_static(record)
        return cast(SemanticMemoryStepResult, self._semantic_step_jit(state, request, record))

    @functools.partial(jax.jit, static_argnums=(0,))
    def _semantic_step_jit(
        self,
        state: ConsolidatedMemoryState,
        request: SemanticMemoryRequest,
        record: SemanticMemoryRecord,
    ) -> SemanticMemoryStepResult:
        state_valid = self._state_is_valid(
            state,
            source_digest=state.source_digest,
            semantic_namespace_digest=state.semantic_namespace_digest,
            representation_revision=state.representation_revision,
            source_revision=state.source_revision,
        )
        request_valid = self._semantic_request_valid(state, request)
        record_valid = self._semantic_record_valid(state, record)
        available = state.operation_count < self._config.max_operations
        preliminary = state_valid & request_valid & record_valid & available
        next_step = state.operation_count + jnp.asarray(1, dtype=jnp.int32)
        refreshed = self._refresh_stale(state, next_step)
        queried, retrieval = self._semantic_retrieval(
            refreshed,
            request,
            state_valid=state_valid,
            request_valid=request_valid,
            transaction_applied=preliminary,
            access_step=next_step,
        )
        written, diagnostics = self._semantic_write(
            queried,
            record,
            write_step=next_step,
            state_valid=state_valid,
            record_valid=record_valid,
            operation_available=preliminary,
        )
        committed = dataclasses.replace(
            written,
            operation_count=next_step,
            semantic_query_count=state.semantic_query_count + 1,
            semantic_accepted_query_count=(
                state.semantic_accepted_query_count + retrieval.accepted.astype(jnp.int32)
            ),
        )
        final_state = jax.lax.cond(diagnostics.wrote, lambda: committed, lambda: state)
        retrieval = dataclasses.replace(
            retrieval,
            transaction_applied=diagnostics.wrote,
            accepted=retrieval.accepted & diagnostics.wrote,
        )
        return SemanticMemoryStepResult(state=final_state, retrieval=retrieval, write=diagnostics)

    def procedural_step(
        self,
        state: ConsolidatedMemoryState,
        request: ProceduralMemoryRequest,
        record: ProceduralMemoryRecord,
    ) -> ProceduralMemoryStepResult:
        """Causally query the pre-transition state, then consolidate the skill."""

        self._validate_state_static_contract(state)
        self._validate_procedural_request_static(request)
        self._validate_procedural_record_static(record)
        return cast(
            ProceduralMemoryStepResult,
            self._procedural_step_jit(state, request, record),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def _procedural_step_jit(
        self,
        state: ConsolidatedMemoryState,
        request: ProceduralMemoryRequest,
        record: ProceduralMemoryRecord,
    ) -> ProceduralMemoryStepResult:
        state_valid = self._state_is_valid(
            state,
            source_digest=state.source_digest,
            semantic_namespace_digest=state.semantic_namespace_digest,
            representation_revision=state.representation_revision,
            source_revision=state.source_revision,
        )
        request_valid = self._procedural_request_valid(state, request)
        record_valid = self._procedural_record_valid(state, record)
        available = state.operation_count < self._config.max_operations
        preliminary = state_valid & request_valid & record_valid & available
        next_step = state.operation_count + jnp.asarray(1, dtype=jnp.int32)
        refreshed = self._refresh_stale(state, next_step)
        queried, retrieval = self._procedural_retrieval(
            refreshed,
            request,
            state_valid=state_valid,
            request_valid=request_valid,
            transaction_applied=preliminary,
            access_step=next_step,
        )
        written, diagnostics = self._procedural_write(
            queried,
            record,
            write_step=next_step,
            state_valid=state_valid,
            record_valid=record_valid,
            operation_available=preliminary,
        )
        committed = dataclasses.replace(
            written,
            operation_count=next_step,
            procedural_query_count=state.procedural_query_count + 1,
            procedural_accepted_query_count=(
                state.procedural_accepted_query_count + retrieval.accepted.astype(jnp.int32)
            ),
        )
        final_state = jax.lax.cond(diagnostics.wrote, lambda: committed, lambda: state)
        retrieval = dataclasses.replace(
            retrieval,
            transaction_applied=diagnostics.wrote,
            accepted=retrieval.accepted & diagnostics.wrote,
        )
        return ProceduralMemoryStepResult(state=final_state, retrieval=retrieval, write=diagnostics)

    def invalidate_semantic(
        self, state: ConsolidatedMemoryState, request: SemanticMemoryRequest
    ) -> MemoryInvalidationResult:
        """Explicitly invalidate one exactly compatible semantic generation."""

        self._validate_state_static_contract(state)
        self._validate_semantic_request_static(request)
        return cast(MemoryInvalidationResult, self._invalidate_semantic_jit(state, request))

    @functools.partial(jax.jit, static_argnums=(0,))
    def _invalidate_semantic_jit(
        self, state: ConsolidatedMemoryState, request: SemanticMemoryRequest
    ) -> MemoryInvalidationResult:
        state_valid = self._state_is_valid(
            state,
            source_digest=state.source_digest,
            semantic_namespace_digest=state.semantic_namespace_digest,
            representation_revision=state.representation_revision,
            source_revision=state.source_revision,
        )
        request_valid = self._semantic_request_valid(state, request)
        available = state.operation_count < self._config.max_operations
        next_step = state.operation_count + jnp.asarray(1, dtype=jnp.int32)
        refreshed = self._refresh_stale(state, next_step)
        records = refreshed.semantic
        match = (
            records.occupied
            & records.valid
            & (~records.invalidated)
            & jnp.all(records.semantic_digests == request.semantic_digest[None, :], axis=1)
            & (records.generations == request.generation)
            & (records.kinds == request.kind)
            & jnp.all(records.provenance_digests == request.provenance_digest[None, :], axis=1)
            & (records.representation_revisions == request.representation_revision)
            & (records.source_revisions == request.source_revision)
        )
        found = jnp.any(match)
        applied = state_valid & request_valid & available & found
        slot = _first_true(match)
        updated_records = dataclasses.replace(
            records,
            valid=records.valid.at[slot].set(jnp.where(applied, False, records.valid[slot])),
            invalidated=records.invalidated.at[slot].set(
                jnp.where(applied, True, records.invalidated[slot])
            ),
        )
        committed = dataclasses.replace(
            refreshed,
            semantic=updated_records,
            operation_count=next_step,
            semantic_invalidation_count=state.semantic_invalidation_count + 1,
        )
        final_state = jax.lax.cond(applied, lambda: committed, lambda: state)
        return MemoryInvalidationResult(
            state=final_state,
            transaction_applied=applied,
            invalidated=applied,
            slot=jnp.where(applied, slot, -1),
        )

    def invalidate_procedural(
        self, state: ConsolidatedMemoryState, request: ProceduralMemoryRequest
    ) -> MemoryInvalidationResult:
        """Explicitly invalidate one exactly compatible procedural generation."""

        self._validate_state_static_contract(state)
        self._validate_procedural_request_static(request)
        return cast(
            MemoryInvalidationResult,
            self._invalidate_procedural_jit(state, request),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def _invalidate_procedural_jit(
        self, state: ConsolidatedMemoryState, request: ProceduralMemoryRequest
    ) -> MemoryInvalidationResult:
        state_valid = self._state_is_valid(
            state,
            source_digest=state.source_digest,
            semantic_namespace_digest=state.semantic_namespace_digest,
            representation_revision=state.representation_revision,
            source_revision=state.source_revision,
        )
        request_valid = self._procedural_request_valid(state, request)
        available = state.operation_count < self._config.max_operations
        next_step = state.operation_count + jnp.asarray(1, dtype=jnp.int32)
        refreshed = self._refresh_stale(state, next_step)
        records = refreshed.procedural
        match = (
            records.occupied
            & records.valid
            & (~records.invalidated)
            & jnp.all(records.semantic_digests == request.semantic_digest[None, :], axis=1)
            & (records.generations == request.generation)
            & jnp.all(records.provenance_digests == request.provenance_digest[None, :], axis=1)
            & (records.representation_revisions == request.representation_revision)
            & (records.source_revisions == request.source_revision)
            & (records.lifecycle_link_available == request.lifecycle_link_available)
            & jnp.all(records.lifecycle_digests == request.lifecycle_digest[None, :], axis=1)
            & (records.lifecycle_generations == request.lifecycle_generation)
            & (records.lifecycle_revisions == request.lifecycle_revision)
        )
        found = jnp.any(match)
        applied = state_valid & request_valid & available & found
        slot = _first_true(match)
        updated_records = dataclasses.replace(
            records,
            valid=records.valid.at[slot].set(jnp.where(applied, False, records.valid[slot])),
            invalidated=records.invalidated.at[slot].set(
                jnp.where(applied, True, records.invalidated[slot])
            ),
        )
        committed = dataclasses.replace(
            refreshed,
            procedural=updated_records,
            operation_count=next_step,
            procedural_invalidation_count=state.procedural_invalidation_count + 1,
        )
        final_state = jax.lax.cond(applied, lambda: committed, lambda: state)
        return MemoryInvalidationResult(
            state=final_state,
            transaction_applied=applied,
            invalidated=applied,
            slot=jnp.where(applied, slot, -1),
        )

    def accounting(self, state: ConsolidatedMemoryState) -> ConsolidatedMemoryAccounting:
        """Return exact allocation and current occupancy without allocating slots."""

        self._validate_state_static_contract(state)
        semantic_active = (
            state.semantic.occupied
            & state.semantic.valid
            & (~state.semantic.stale)
            & (~state.semantic.invalidated)
        )
        procedural_active = (
            state.procedural.occupied
            & state.procedural.valid
            & (~state.procedural.stale)
            & (~state.procedural.invalidated)
        )
        return ConsolidatedMemoryAccounting(
            persistent_state_bytes=jnp.asarray(self._persistent_state_bytes, dtype=jnp.int32),
            semantic_capacity=jnp.asarray(self._config.semantic_capacity, dtype=jnp.int32),
            procedural_capacity=jnp.asarray(self._config.procedural_capacity, dtype=jnp.int32),
            active_semantic_records=jnp.sum(semantic_active.astype(jnp.int32)),
            active_procedural_records=jnp.sum(procedural_active.astype(jnp.int32)),
            occupied_semantic_records=jnp.sum(state.semantic.occupied.astype(jnp.int32)),
            occupied_procedural_records=jnp.sum(state.procedural.occupied.astype(jnp.int32)),
            operation_count=state.operation_count,
            semantic_queries=state.semantic_query_count,
            semantic_writes=state.semantic_write_count,
            procedural_queries=state.procedural_query_count,
            procedural_writes=state.procedural_write_count,
        )

    @staticmethod
    def _state_sha256(state: ConsolidatedMemoryState) -> Array:
        digest = hashlib.sha256()
        for leaf in jax.tree_util.tree_leaves(state):
            host = np.asarray(jax.device_get(leaf))
            digest.update(host.dtype.str.encode("ascii"))
            digest.update(np.asarray(host.shape, dtype=np.int64).tobytes())
            digest.update(host.tobytes(order="C"))
        return jnp.asarray(tuple(digest.digest()), dtype=jnp.uint8)

    def _binding_sha256(
        self,
        *,
        source_digest: Array,
        semantic_namespace_digest: Array,
        representation_revision: Array,
        source_revision: Array,
    ) -> Array:
        digest = hashlib.sha256()
        digest.update(bytes(self._config_sha256))
        digest.update(np.asarray(jax.device_get(source_digest), dtype=np.uint8).tobytes())
        digest.update(
            np.asarray(jax.device_get(semantic_namespace_digest), dtype=np.uint8).tobytes()
        )
        digest.update(
            np.asarray(
                [
                    int(jax.device_get(representation_revision)),
                    int(jax.device_get(source_revision)),
                ],
                dtype=np.int32,
            ).tobytes()
        )
        return jnp.asarray(tuple(digest.digest()), dtype=jnp.uint8)

    def checkpoint_payload(
        self,
        state: ConsolidatedMemoryState,
        *,
        source_digest: Array,
        semantic_namespace_digest: Array,
        representation_revision: int | Array,
        source_revision: int | Array,
    ) -> dict[str, object]:
        """Return a strict SHA-bound PyTree checkpoint payload."""

        representation = _int32_scalar(representation_revision, name="representation_revision")
        source = _int32_scalar(source_revision, name="source_revision")
        valid = self.validate_state(
            state,
            source_digest=source_digest,
            semantic_namespace_digest=semantic_namespace_digest,
            representation_revision=representation,
            source_revision=source,
        )
        if not bool(jax.device_get(valid)):
            raise ValueError("cannot checkpoint an invalid or stale consolidated memory")
        return {
            "schema_version": CONSOLIDATED_MEMORY_CHECKPOINT_SCHEMA,
            "config": self.to_config(),
            "state": state,
            "state_sha256": self._state_sha256(state),
            "binding_sha256": self._binding_sha256(
                source_digest=source_digest,
                semantic_namespace_digest=semantic_namespace_digest,
                representation_revision=representation,
                source_revision=source,
            ),
        }

    def restore_checkpoint(
        self,
        payload: object,
        *,
        source_digest: Array,
        semantic_namespace_digest: Array,
        representation_revision: int | Array,
        source_revision: int | Array,
    ) -> ConsolidatedMemoryState:
        """Restore only the exact schema, config, state SHA, and live bindings."""

        if type(payload) is not dict:
            raise ValueError("consolidated memory checkpoint must be an exact dict")
        raw = cast(dict[object, object], payload)
        expected = {
            "schema_version",
            "config",
            "state",
            "state_sha256",
            "binding_sha256",
        }
        if set(raw) != expected:
            raise ValueError("consolidated memory checkpoint keys differ from schema v1")
        if raw["schema_version"] != CONSOLIDATED_MEMORY_CHECKPOINT_SCHEMA:
            raise ValueError("consolidated memory checkpoint schema differs")
        if ConsolidatedMemoryConfig.from_config(raw["config"]) != self._config:
            raise ValueError("consolidated memory checkpoint config differs")
        state = raw["state"]
        if type(state) is not ConsolidatedMemoryState:
            raise ValueError("consolidated memory checkpoint state type differs")
        restored = cast(ConsolidatedMemoryState, state)
        self._validate_state_static_contract(restored)
        for name in ("state_sha256", "binding_sha256"):
            _require_array(
                cast(Array, raw[name]), name=f"checkpoint.{name}", shape=(32,), dtype=jnp.uint8
            )
        if not bool(
            jax.device_get(
                jnp.array_equal(cast(Array, raw["state_sha256"]), self._state_sha256(restored))
            )
        ):
            raise ValueError("consolidated memory checkpoint state SHA differs")
        _require_array(source_digest, name="source_digest", shape=(32,), dtype=jnp.uint8)
        _require_array(
            semantic_namespace_digest,
            name="semantic_namespace_digest",
            shape=(32,),
            dtype=jnp.uint8,
        )
        representation = _int32_scalar(representation_revision, name="representation_revision")
        source = _int32_scalar(source_revision, name="source_revision")
        expected_binding = self._binding_sha256(
            source_digest=source_digest,
            semantic_namespace_digest=semantic_namespace_digest,
            representation_revision=representation,
            source_revision=source,
        )
        if not bool(
            jax.device_get(jnp.array_equal(cast(Array, raw["binding_sha256"]), expected_binding))
        ):
            raise ValueError("consolidated memory checkpoint source or relabel binding differs")
        valid = self.validate_state(
            restored,
            source_digest=source_digest,
            semantic_namespace_digest=semantic_namespace_digest,
            representation_revision=representation,
            source_revision=source,
        )
        if not bool(jax.device_get(valid)):
            raise ValueError("consolidated memory checkpoint state is invalid or stale")
        return restored


__all__ = [
    "CONSOLIDATED_MEMORY_ACTION_SELECTION_AUTHORITY",
    "CONSOLIDATED_MEMORY_AGENT_MUTATION_AUTHORITY",
    "CONSOLIDATED_MEMORY_CHECKPOINT_SCHEMA",
    "CONSOLIDATED_MEMORY_CONFIG_SCHEMA",
    "CONSOLIDATED_MEMORY_GO_NO_GO_AUTHORITY",
    "CONSOLIDATED_MEMORY_PROMOTION_AUTHORITY",
    "CONSOLIDATED_MEMORY_SCIENTIFIC_PROMOTION_ALLOWED",
    "SEMANTIC_KIND_AFFORDANCE",
    "SEMANTIC_KIND_FACT",
    "SEMANTIC_KIND_GVF",
    "ConsolidatedMemory",
    "ConsolidatedMemoryAccounting",
    "ConsolidatedMemoryConfig",
    "ConsolidatedMemoryResourceBudget",
    "ConsolidatedMemoryState",
    "MemoryInvalidationResult",
    "MemoryWriteDiagnostics",
    "ProceduralMemoryRecord",
    "ProceduralMemoryRecords",
    "ProceduralMemoryRequest",
    "ProceduralMemoryRetrieval",
    "ProceduralMemoryStepResult",
    "SemanticMemoryRecord",
    "SemanticMemoryRecords",
    "SemanticMemoryRequest",
    "SemanticMemoryRetrieval",
    "SemanticMemoryStepResult",
    "canonical_memory_digest",
]
