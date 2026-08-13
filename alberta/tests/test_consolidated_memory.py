# mypy: disable-error-code="arg-type,call-arg,type-var"
"""Unit contracts for bounded semantic and procedural consolidation."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import pytest

from alberta_framework.core.consolidated_memory import (
    CONSOLIDATED_MEMORY_ACTION_SELECTION_AUTHORITY,
    CONSOLIDATED_MEMORY_AGENT_MUTATION_AUTHORITY,
    CONSOLIDATED_MEMORY_GO_NO_GO_AUTHORITY,
    CONSOLIDATED_MEMORY_PROMOTION_AUTHORITY,
    CONSOLIDATED_MEMORY_SCIENTIFIC_PROMOTION_ALLOWED,
    SEMANTIC_KIND_FACT,
    ConsolidatedMemory,
    ConsolidatedMemoryConfig,
    ConsolidatedMemoryState,
    ProceduralMemoryRecord,
    ProceduralMemoryRequest,
    SemanticMemoryRecord,
    SemanticMemoryRequest,
    canonical_memory_digest,
)

pytestmark = pytest.mark.unit


def _digest(text: str) -> jax.Array:
    return canonical_memory_digest("test.consolidated-memory", text)


def _memory(
    *,
    semantic_capacity: int = 2,
    procedural_capacity: int = 2,
    semantic_max_age: int = 4,
    procedural_max_age: int = 4,
    semantic_min_confidence: float = 0.5,
    procedural_min_confidence: float = 0.5,
) -> tuple[ConsolidatedMemory, ConsolidatedMemoryState]:
    memory = ConsolidatedMemory(
        ConsolidatedMemoryConfig(
            semantic_capacity=semantic_capacity,
            procedural_capacity=procedural_capacity,
            semantic_payload_dim=3,
            procedural_payload_dim=2,
            procedural_outcome_dim=2,
            semantic_max_age=semantic_max_age,
            procedural_max_age=procedural_max_age,
            max_operations=100,
            semantic_min_confidence=semantic_min_confidence,
            procedural_min_confidence=procedural_min_confidence,
        )
    )
    state = memory.init(
        source_digest=_digest("source"),
        semantic_namespace_digest=_digest("namespace"),
        representation_revision=3,
        source_revision=7,
    )
    return memory, state


def _semantic(
    identity: str,
    *,
    generation: int = 0,
    confidence: float = 0.8,
    provenance: str = "provenance-a",
    payload: tuple[float, float, float] = (1.0, 2.0, 3.0),
    evidence: float = 1.0,
) -> SemanticMemoryRecord:
    return SemanticMemoryRecord(
        semantic_digest=_digest(identity),
        generation=jnp.asarray(generation, dtype=jnp.int32),
        kind=jnp.asarray(SEMANTIC_KIND_FACT, dtype=jnp.int32),
        payload=jnp.asarray(payload, dtype=jnp.float32),
        confidence=jnp.asarray(confidence, dtype=jnp.float32),
        provenance_digest=_digest(provenance),
        representation_revision=jnp.asarray(3, dtype=jnp.int32),
        source_revision=jnp.asarray(7, dtype=jnp.int32),
        evidence=jnp.asarray(evidence, dtype=jnp.float32),
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


def _procedural(
    identity: str,
    *,
    generation: int = 0,
    confidence: float = 0.9,
    provenance: str = "skill-source",
    payload: tuple[float, float] = (2.0, -1.0),
    evidence: float = 2.0,
    succeeded: bool = True,
    outcome: tuple[float, float] = (1.0, 0.0),
    link: str | None = "option-a",
    lifecycle_generation: int = 4,
    lifecycle_revision: int = 9,
) -> ProceduralMemoryRecord:
    available = link is not None
    return ProceduralMemoryRecord(
        semantic_digest=_digest(identity),
        generation=jnp.asarray(generation, dtype=jnp.int32),
        payload=jnp.asarray(payload, dtype=jnp.float32),
        confidence=jnp.asarray(confidence, dtype=jnp.float32),
        provenance_digest=_digest(provenance),
        representation_revision=jnp.asarray(3, dtype=jnp.int32),
        source_revision=jnp.asarray(7, dtype=jnp.int32),
        evidence=jnp.asarray(evidence, dtype=jnp.float32),
        succeeded=jnp.asarray(succeeded, dtype=jnp.bool_),
        outcome=jnp.asarray(outcome, dtype=jnp.float32),
        lifecycle_link_available=jnp.asarray(available, dtype=jnp.bool_),
        lifecycle_digest=(_digest(link) if link is not None else jnp.zeros((32,), dtype=jnp.uint8)),
        lifecycle_generation=jnp.asarray(
            lifecycle_generation if available else -1, dtype=jnp.int32
        ),
        lifecycle_revision=jnp.asarray(lifecycle_revision if available else -1, dtype=jnp.int32),
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


def test_config_digest_and_empty_state_are_strict_and_frozen() -> None:
    assert jnp.array_equal(_digest("fact"), _digest("fact"))
    assert not jnp.array_equal(_digest("fact"), _digest("Fact"))
    with pytest.raises(ValueError, match="positive exact"):
        ConsolidatedMemoryConfig(
            semantic_capacity=True,
            procedural_capacity=1,
            semantic_payload_dim=1,
            procedural_payload_dim=1,
            procedural_outcome_dim=1,
            semantic_max_age=1,
            procedural_max_age=1,
            max_operations=2,
        )
    memory, state = _memory()
    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
        state.operation_count = jnp.asarray(1, dtype=jnp.int32)
    assert bool(
        memory.validate_state(
            state,
            source_digest=_digest("source"),
            semantic_namespace_digest=_digest("namespace"),
            representation_revision=3,
            source_revision=7,
        )
    )


def test_semantic_step_is_query_before_write_and_merges_welford_moments() -> None:
    memory, state = _memory()
    first = _semantic("fact-a", evidence=2.0, payload=(1.0, 2.0, 3.0))
    result = memory.semantic_step(state, _semantic_request(first), first)
    assert not bool(result.retrieval.accepted)
    assert bool(result.write.wrote)
    assert bool(result.write.reset_evidence)

    second = _semantic("fact-a", evidence=4.0, payload=(3.0, 4.0, 5.0), confidence=0.6)
    result = memory.semantic_step(result.state, _semantic_request(second), second)
    assert bool(result.retrieval.accepted)
    assert jnp.array_equal(result.retrieval.payload, jnp.asarray([1.0, 2.0, 3.0]))
    assert bool(result.write.merged)
    slot = int(result.write.slot)
    assert int(result.state.semantic.evidence_counts[slot]) == 2
    assert jnp.allclose(
        result.state.semantic.payload_means[slot],
        jnp.asarray([2.0, 3.0, 4.0], dtype=jnp.float32),
    )
    assert float(result.state.semantic.evidence_means[slot]) == pytest.approx(3.0)
    assert float(result.state.semantic.evidence_m2[slot]) == pytest.approx(2.0)
    assert int(result.state.semantic.creation_steps[slot]) == 1
    assert int(result.state.semantic.last_use_steps[slot]) == 2


def test_confidence_provenance_and_revision_gates_fail_closed() -> None:
    memory, state = _memory(semantic_min_confidence=0.7)
    low = _semantic("fact-a", confidence=0.6)
    state, write = memory.write_semantic(state, low)
    assert bool(write.wrote)
    state_after_query, retrieval = memory.query_semantic(state, _semantic_request(low))
    assert not bool(retrieval.accepted)
    assert not bool(retrieval.confidence_ok)

    wrong_provenance = dataclasses.replace(
        _semantic_request(low), provenance_digest=_digest("untrusted")
    )
    _, retrieval = memory.query_semantic(state_after_query, wrong_provenance)
    assert bool(retrieval.identity_found)
    assert not bool(retrieval.compatible)

    colliding = _semantic("fact-a", confidence=0.9, provenance="untrusted")
    unchanged, rejected = memory.write_semantic(state_after_query, colliding)
    assert not bool(rejected.wrote)
    assert bool(rejected.identity_collision)
    assert jax.tree_util.tree_all(
        jax.tree_util.tree_map(jnp.array_equal, unchanged, state_after_query)
    )

    revised = _semantic(
        "fact-a",
        generation=1,
        confidence=0.95,
        provenance="reviewed-v2",
        payload=(9.0, 8.0, 7.0),
        evidence=10.0,
    )
    revised_state, revision = memory.write_semantic(state_after_query, revised)
    assert bool(revision.revised)
    assert bool(revision.reset_evidence)
    slot = int(revision.slot)
    assert int(revised_state.semantic.evidence_counts[slot]) == 1
    assert float(revised_state.semantic.evidence_m2[slot]) == 0.0
    assert int(revised_state.semantic.creation_steps[slot]) == int(revised_state.operation_count)


def test_bounded_replacement_uses_deterministic_retired_ties_and_resets() -> None:
    memory, state = _memory(semantic_capacity=2)
    first = _semantic("a", confidence=0.9)
    second = _semantic("b", confidence=0.9)
    state, first_write = memory.write_semantic(state, first)
    state, second_write = memory.write_semantic(state, second)
    assert (int(first_write.slot), int(second_write.slot)) == (0, 1)
    state = memory.invalidate_semantic(state, _semantic_request(first)).state
    state = memory.invalidate_semantic(state, _semantic_request(second)).state

    replacement = _semantic("c", confidence=0.7, evidence=5.0)
    state, write = memory.write_semantic(state, replacement)
    assert bool(write.replaced)
    assert int(write.slot) == 0
    assert int(jnp.sum(state.semantic.occupied)) == 2
    assert int(state.semantic.evidence_counts[0]) == 1
    assert float(state.semantic.evidence_m2[0]) == 0.0
    assert jnp.array_equal(state.semantic.semantic_digests[0], replacement.semantic_digest)


def test_stale_and_invalidated_records_cannot_harm_retrieval() -> None:
    memory, state = _memory(semantic_max_age=1)
    record = _semantic("stale")
    state, _ = memory.write_semantic(state, record)
    unknown = _semantic("unknown")
    state, _ = memory.query_semantic(state, _semantic_request(unknown))
    state, retrieval = memory.query_semantic(state, _semantic_request(record))
    assert not bool(retrieval.accepted)
    assert not bool(retrieval.fresh)
    assert bool(state.semantic.stale[0])

    revision = _semantic("stale", generation=1, payload=(8.0, 8.0, 8.0))
    state, write = memory.write_semantic(state, revision)
    assert bool(write.revised)
    state, retrieval = memory.query_semantic(state, _semantic_request(revision))
    assert bool(retrieval.accepted)

    invalidated = memory.invalidate_semantic(state, _semantic_request(revision))
    assert bool(invalidated.invalidated)
    _, retrieval = memory.query_semantic(invalidated.state, _semantic_request(revision))
    assert not bool(retrieval.accepted)


def test_procedural_success_failure_outcomes_and_lifecycle_are_bound() -> None:
    memory, state = _memory()
    success = _procedural("walk", succeeded=True, outcome=(1.0, 3.0))
    first = memory.procedural_step(state, _procedural_request(success), success)
    assert not bool(first.retrieval.accepted)
    failure = _procedural("walk", succeeded=False, outcome=(-1.0, 1.0), evidence=4.0)
    second = memory.procedural_step(first.state, _procedural_request(failure), failure)
    assert bool(second.retrieval.accepted)
    assert bool(second.write.merged)
    slot = int(second.write.slot)
    assert int(second.state.procedural.success_counts[slot]) == 1
    assert int(second.state.procedural.failure_counts[slot]) == 1
    assert jnp.allclose(
        second.state.procedural.outcome_means[slot],
        jnp.asarray([0.0, 2.0], dtype=jnp.float32),
    )

    wrong_link = dataclasses.replace(
        _procedural_request(failure), lifecycle_digest=_digest("option-b")
    )
    _, retrieval = memory.query_procedural(second.state, wrong_link)
    assert bool(retrieval.identity_found)
    assert not bool(retrieval.compatible)

    revised = _procedural(
        "walk",
        generation=1,
        link="option-b",
        lifecycle_generation=0,
        lifecycle_revision=10,
        succeeded=False,
    )
    revised_state, write = memory.write_procedural(second.state, revised)
    assert bool(write.revised)
    slot = int(write.slot)
    assert int(revised_state.procedural.success_counts[slot]) == 0
    assert int(revised_state.procedural.failure_counts[slot]) == 1
    assert int(revised_state.procedural.evidence_counts[slot]) == 1


def test_resources_are_exact_fixed_and_have_no_agent_or_action_authority() -> None:
    memory, state = _memory(semantic_capacity=3, procedural_capacity=4)
    budget = memory.resource_budget
    accounting = memory.accounting(state)
    actual_bytes = sum(
        int(leaf.size) * int(leaf.dtype.itemsize) for leaf in jax.tree_util.tree_leaves(state)
    )
    assert budget.persistent_state_bytes == actual_bytes
    assert int(accounting.persistent_state_bytes) == actual_bytes
    assert budget.semantic_capacity == 3
    assert budget.procedural_capacity == 4
    assert budget.dynamic_persistent_allocations_per_operation == 0
    assert budget.random_generator_calls_at_init == 0
    assert budget.random_generator_calls_per_operation == 0
    assert budget.agent_parameter_mutations_per_operation == 0
    assert budget.action_selection_calls_per_operation == 0
    assert not budget.agent_mutation_authority
    assert not budget.action_selection_authority
    assert not budget.promotion_authority
    assert not budget.go_no_go_authority
    assert not budget.scientific_promotion_allowed
    assert not CONSOLIDATED_MEMORY_AGENT_MUTATION_AUTHORITY
    assert not CONSOLIDATED_MEMORY_ACTION_SELECTION_AUTHORITY
    assert not CONSOLIDATED_MEMORY_PROMOTION_AUTHORITY
    assert not CONSOLIDATED_MEMORY_GO_NO_GO_AUTHORITY
    assert not CONSOLIDATED_MEMORY_SCIENTIFIC_PROMOTION_ALLOWED


def test_checkpoint_roundtrip_rejects_tamper_source_and_relabel() -> None:
    memory, state = _memory()
    state, _ = memory.write_semantic(state, _semantic("fact-a"))
    kwargs = {
        "source_digest": _digest("source"),
        "semantic_namespace_digest": _digest("namespace"),
        "representation_revision": 3,
        "source_revision": 7,
    }
    checkpoint = memory.checkpoint_payload(state, **kwargs)
    restored = memory.restore_checkpoint(checkpoint, **kwargs)
    assert jax.tree_util.tree_all(jax.tree_util.tree_map(jnp.array_equal, state, restored))

    tampered = dict(checkpoint)
    tampered["state"] = dataclasses.replace(state, operation_count=state.operation_count + 1)
    with pytest.raises(ValueError, match="state SHA"):
        memory.restore_checkpoint(tampered, **kwargs)
    with pytest.raises(ValueError, match="source or relabel"):
        memory.restore_checkpoint(
            checkpoint, **{**kwargs, "source_digest": _digest("different-source")}
        )
    with pytest.raises(ValueError, match="source or relabel"):
        memory.restore_checkpoint(
            checkpoint,
            **{
                **kwargs,
                "semantic_namespace_digest": _digest("relabeled-namespace"),
            },
        )
    with pytest.raises(ValueError, match="source or relabel"):
        memory.restore_checkpoint(checkpoint, **{**kwargs, "representation_revision": 4})
