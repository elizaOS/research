# mypy: disable-error-code="arg-type,call-arg"
"""Integration coverage for JAX execution and checkpointed continuation."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from alberta_framework.core.consolidated_memory import (
    SEMANTIC_KIND_AFFORDANCE,
    ConsolidatedMemory,
    ConsolidatedMemoryConfig,
    ConsolidatedMemoryState,
    ProceduralMemoryRecord,
    ProceduralMemoryRequest,
    SemanticMemoryRecord,
    SemanticMemoryRequest,
    canonical_memory_digest,
)

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def _digest(text: str) -> jax.Array:
    return canonical_memory_digest("test.consolidated-memory.integration", text)


def _setup() -> tuple[ConsolidatedMemory, ConsolidatedMemoryState]:
    memory = ConsolidatedMemory(
        ConsolidatedMemoryConfig(
            semantic_capacity=3,
            procedural_capacity=2,
            semantic_payload_dim=2,
            procedural_payload_dim=2,
            procedural_outcome_dim=1,
            semantic_max_age=8,
            procedural_max_age=8,
            max_operations=100,
            semantic_min_confidence=0.5,
            procedural_min_confidence=0.5,
        )
    )
    state = memory.init(
        source_digest=_digest("source"),
        semantic_namespace_digest=_digest("namespace"),
        representation_revision=2,
        source_revision=5,
    )
    return memory, state


def _semantic(identity: str, value: float) -> SemanticMemoryRecord:
    return SemanticMemoryRecord(
        semantic_digest=_digest(identity),
        generation=jnp.asarray(0, dtype=jnp.int32),
        kind=jnp.asarray(SEMANTIC_KIND_AFFORDANCE, dtype=jnp.int32),
        payload=jnp.asarray([value, -value], dtype=jnp.float32),
        confidence=jnp.asarray(0.8, dtype=jnp.float32),
        provenance_digest=_digest(f"provenance:{identity}"),
        representation_revision=jnp.asarray(2, dtype=jnp.int32),
        source_revision=jnp.asarray(5, dtype=jnp.int32),
        evidence=jnp.asarray(value, dtype=jnp.float32),
    )


def _semantic_request(record: SemanticMemoryRecord) -> SemanticMemoryRequest:
    return SemanticMemoryRequest(
        semantic_digest=record.semantic_digest,
        generation=record.generation,
        kind=record.kind,
        provenance_digest=record.provenance_digest,
        representation_revision=record.representation_revision,
        source_revision=record.source_revision,
    )


def _procedural(identity: str, success: bool, outcome: float) -> ProceduralMemoryRecord:
    return ProceduralMemoryRecord(
        semantic_digest=_digest(identity),
        generation=jnp.asarray(0, dtype=jnp.int32),
        payload=jnp.asarray([outcome, 1.0], dtype=jnp.float32),
        confidence=jnp.asarray(0.75, dtype=jnp.float32),
        provenance_digest=_digest(f"provenance:{identity}"),
        representation_revision=jnp.asarray(2, dtype=jnp.int32),
        source_revision=jnp.asarray(5, dtype=jnp.int32),
        evidence=jnp.asarray(abs(outcome), dtype=jnp.float32),
        succeeded=jnp.asarray(success, dtype=jnp.bool_),
        outcome=jnp.asarray([outcome], dtype=jnp.float32),
        lifecycle_link_available=jnp.asarray(True, dtype=jnp.bool_),
        lifecycle_digest=_digest(f"option:{identity}"),
        lifecycle_generation=jnp.asarray(1, dtype=jnp.int32),
        lifecycle_revision=jnp.asarray(3, dtype=jnp.int32),
    )


def _procedural_request(record: ProceduralMemoryRecord) -> ProceduralMemoryRequest:
    return ProceduralMemoryRequest(
        semantic_digest=record.semantic_digest,
        generation=record.generation,
        provenance_digest=record.provenance_digest,
        representation_revision=record.representation_revision,
        source_revision=record.source_revision,
        lifecycle_link_available=record.lifecycle_link_available,
        lifecycle_digest=record.lifecycle_digest,
        lifecycle_generation=record.lifecycle_generation,
        lifecycle_revision=record.lifecycle_revision,
    )


def _tree_equal(left: object, right: object) -> bool:
    return bool(jax.tree_util.tree_all(jax.tree_util.tree_map(jnp.array_equal, left, right)))


def test_semantic_eager_jit_and_scan_have_exact_parity() -> None:
    memory, initial = _setup()
    records = (_semantic("a", 1.0), _semantic("a", 3.0), _semantic("b", 2.0))
    requests = tuple(_semantic_request(record) for record in records)

    with jax.disable_jit():
        eager = initial
        eager_trace: list[tuple[jax.Array, jax.Array]] = []
        for request, record in zip(requests, records, strict=True):
            result = memory.semantic_step(eager, request, record)
            eager = result.state
            eager_trace.append((result.retrieval.accepted, result.write.wrote))

    compiled = initial
    compiled_trace: list[tuple[jax.Array, jax.Array]] = []
    for request, record in zip(requests, records, strict=True):
        result = memory.semantic_step(compiled, request, record)
        compiled = result.state
        compiled_trace.append((result.retrieval.accepted, result.write.wrote))
    assert _tree_equal(eager, compiled)
    assert _tree_equal(eager_trace, compiled_trace)

    stacked_requests = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *requests)
    stacked_records = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *records)

    def body(
        state: ConsolidatedMemoryState,
        inputs: tuple[SemanticMemoryRequest, SemanticMemoryRecord],
    ) -> tuple[ConsolidatedMemoryState, tuple[jax.Array, jax.Array]]:
        request, record = inputs
        result = memory.semantic_step(state, request, record)
        return result.state, (result.retrieval.accepted, result.write.wrote)

    scanned, scanned_trace = jax.jit(lambda state, inputs: jax.lax.scan(body, state, inputs))(
        initial, (stacked_requests, stacked_records)
    )
    assert _tree_equal(compiled, scanned)
    assert _tree_equal(compiled_trace, list(zip(*scanned_trace, strict=True)))


def test_procedural_scan_preserves_success_failure_and_fixed_shapes() -> None:
    memory, initial = _setup()
    records = (
        _procedural("walk", True, 2.0),
        _procedural("walk", False, -2.0),
        _procedural("turn", True, 1.0),
    )
    requests = tuple(_procedural_request(record) for record in records)
    stacked_requests = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *requests)
    stacked_records = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *records)

    def body(
        state: ConsolidatedMemoryState,
        inputs: tuple[ProceduralMemoryRequest, ProceduralMemoryRecord],
    ) -> tuple[ConsolidatedMemoryState, jax.Array]:
        request, record = inputs
        result = memory.procedural_step(state, request, record)
        return result.state, result.retrieval.accepted

    final, accepted = jax.jit(lambda state, inputs: jax.lax.scan(body, state, inputs))(
        initial, (stacked_requests, stacked_records)
    )
    assert accepted.tolist() == [False, True, False]
    assert final.procedural.payload_means.shape == (2, 2)
    assert final.procedural.outcome_means.shape == (2, 1)
    assert final.procedural.success_counts.tolist() == [1, 1]
    assert final.procedural.failure_counts.tolist() == [1, 0]
    assert memory.resource_budget.persistent_state_bytes == sum(
        int(leaf.size) * int(leaf.dtype.itemsize) for leaf in jax.tree_util.tree_leaves(final)
    )


def test_checkpoint_resume_is_exact_and_rejects_live_source_revision() -> None:
    memory, initial = _setup()
    a1 = _semantic("a", 1.0)
    a2 = _semantic("a", 3.0)
    b = _semantic("b", 2.0)
    partial = memory.semantic_step(initial, _semantic_request(a1), a1).state
    kwargs = {
        "source_digest": _digest("source"),
        "semantic_namespace_digest": _digest("namespace"),
        "representation_revision": 2,
        "source_revision": 5,
    }
    checkpoint = memory.checkpoint_payload(partial, **kwargs)
    restored = memory.restore_checkpoint(checkpoint, **kwargs)

    uninterrupted = memory.semantic_step(partial, _semantic_request(a2), a2).state
    uninterrupted = memory.semantic_step(uninterrupted, _semantic_request(b), b).state
    resumed = memory.semantic_step(restored, _semantic_request(a2), a2).state
    resumed = memory.semantic_step(resumed, _semantic_request(b), b).state
    assert _tree_equal(uninterrupted, resumed)
    assert bool(memory.validate_state(resumed, **kwargs))

    with pytest.raises(ValueError, match="source or relabel"):
        memory.restore_checkpoint(checkpoint, **{**kwargs, "source_revision": 6})
