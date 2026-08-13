# mypy: disable-error-code="arg-type,call-arg,type-var"
"""JAX parity and real-memory composition for the procedural policy boundary."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from alberta_framework.core.consolidated_memory import (
    CONSOLIDATED_MEMORY_ACTION_SELECTION_AUTHORITY,
    CONSOLIDATED_MEMORY_AGENT_MUTATION_AUTHORITY,
    ConsolidatedMemory,
    ConsolidatedMemoryConfig,
    ProceduralMemoryRecord,
    ProceduralMemoryRequest,
    ProceduralMemoryRetrieval,
    canonical_memory_digest,
)
from alberta_framework.core.consolidated_memory_policy import (
    ConsolidatedProceduralMemoryPolicy,
    ConsolidatedProceduralMemoryPolicyConfig,
    ConsolidatedProceduralMemoryProposal,
)

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def _digest(text: str) -> jax.Array:
    return canonical_memory_digest("test.consolidated-memory-policy.integration", text)


def _policy() -> ConsolidatedProceduralMemoryPolicy:
    return ConsolidatedProceduralMemoryPolicy(
        ConsolidatedProceduralMemoryPolicyConfig(
            n_actions=3,
            outcome_dim=1,
            min_evidence_count=2,
            min_success_lower_bound=0.5,
            wilson_z=1.0,
            max_outcome_standard_error=0.5,
            max_abs_outcome_mean=10.0,
        )
    )


def _retrieval(payload: tuple[float, float, float]) -> ProceduralMemoryRetrieval:
    return ProceduralMemoryRetrieval(
        accepted=jnp.asarray(True, dtype=jnp.bool_),
        transaction_applied=jnp.asarray(True, dtype=jnp.bool_),
        slot=jnp.asarray(0, dtype=jnp.int32),
        payload=jnp.asarray(payload, dtype=jnp.float32),
        confidence=jnp.asarray(0.9, dtype=jnp.float32),
        evidence_count=jnp.asarray(4, dtype=jnp.int32),
        evidence_mean=jnp.asarray(1.0, dtype=jnp.float32),
        evidence_m2=jnp.asarray(0.0, dtype=jnp.float32),
        success_count=jnp.asarray(4, dtype=jnp.int32),
        failure_count=jnp.asarray(0, dtype=jnp.int32),
        outcome_mean=jnp.asarray([1.0], dtype=jnp.float32),
        outcome_m2=jnp.asarray([0.0], dtype=jnp.float32),
        lifecycle_link_available=jnp.asarray(True, dtype=jnp.bool_),
        lifecycle_digest=_digest("option-a"),
        lifecycle_generation=jnp.asarray(2, dtype=jnp.int32),
        lifecycle_revision=jnp.asarray(3, dtype=jnp.int32),
        state_valid=jnp.asarray(True, dtype=jnp.bool_),
        request_valid=jnp.asarray(True, dtype=jnp.bool_),
        identity_found=jnp.asarray(True, dtype=jnp.bool_),
        compatible=jnp.asarray(True, dtype=jnp.bool_),
        fresh=jnp.asarray(True, dtype=jnp.bool_),
        confidence_ok=jnp.asarray(True, dtype=jnp.bool_),
    )


def _tree_equal(left: object, right: object) -> bool:
    return bool(jax.tree_util.tree_all(jax.tree_util.tree_map(jnp.array_equal, left, right)))


def test_eager_jit_and_scan_proposals_have_exact_parity() -> None:
    policy = _policy()
    retrievals = (
        _retrieval((0.9, 0.2, 0.1)),
        _retrieval((0.1, 0.8, 0.8)),
        _retrieval((0.1, 0.2, 0.9)),
    )
    masks = jnp.asarray(
        ((True, True, True), (True, True, True), (True, True, False)),
        dtype=jnp.bool_,
    )
    expected_digest = _digest("option-a")

    def propose(
        retrieval: ProceduralMemoryRetrieval, mask: jax.Array
    ) -> ConsolidatedProceduralMemoryProposal:
        return policy.propose(
            retrieval,
            hard_safety_mask=mask,
            expected_lifecycle_digest=expected_digest,
            expected_lifecycle_generation=2,
            expected_lifecycle_revision=3,
        )

    with jax.disable_jit():
        eager = tuple(
            propose(retrieval, masks[index]) for index, retrieval in enumerate(retrievals)
        )
    compiled = tuple(
        jax.jit(propose)(retrieval, masks[index]) for index, retrieval in enumerate(retrievals)
    )
    assert _tree_equal(eager, compiled)
    assert [int(item.action) for item in compiled] == [0, 1, 1]

    stacked = jax.tree_util.tree_map(lambda *leaves: jnp.stack(leaves), *retrievals)

    def body(
        carry: jax.Array,
        inputs: tuple[ProceduralMemoryRetrieval, jax.Array],
    ) -> tuple[jax.Array, ConsolidatedProceduralMemoryProposal]:
        retrieval, mask = inputs
        proposal = propose(retrieval, mask)
        return carry, proposal

    _, scanned = jax.jit(
        lambda values, safety: jax.lax.scan(body, jnp.asarray(0, dtype=jnp.int32), (values, safety))
    )(stacked, masks)
    stacked_compiled = jax.tree_util.tree_map(lambda *leaves: jnp.stack(leaves), *compiled)
    assert _tree_equal(scanned, stacked_compiled)


def test_query_before_write_memory_composition_changes_only_a_later_safe_proposal() -> None:
    memory = ConsolidatedMemory(
        ConsolidatedMemoryConfig(
            semantic_capacity=1,
            procedural_capacity=1,
            semantic_payload_dim=1,
            procedural_payload_dim=3,
            procedural_outcome_dim=1,
            semantic_max_age=20,
            procedural_max_age=20,
            max_operations=20,
            semantic_min_confidence=0.5,
            procedural_min_confidence=0.5,
        )
    )
    source_digest = _digest("source")
    namespace_digest = _digest("namespace")
    state = memory.init(
        source_digest=source_digest,
        semantic_namespace_digest=namespace_digest,
        representation_revision=1,
        source_revision=1,
    )
    lifecycle_digest = _digest("live-option")
    record = ProceduralMemoryRecord(
        semantic_digest=_digest("skill"),
        generation=jnp.asarray(0, dtype=jnp.int32),
        payload=jnp.asarray([0.2, 0.9, 0.4], dtype=jnp.float32),
        confidence=jnp.asarray(0.9, dtype=jnp.float32),
        provenance_digest=_digest("provenance"),
        representation_revision=jnp.asarray(1, dtype=jnp.int32),
        source_revision=jnp.asarray(1, dtype=jnp.int32),
        evidence=jnp.asarray(1.0, dtype=jnp.float32),
        succeeded=jnp.asarray(True, dtype=jnp.bool_),
        outcome=jnp.asarray([1.0], dtype=jnp.float32),
        lifecycle_link_available=jnp.asarray(True, dtype=jnp.bool_),
        lifecycle_digest=lifecycle_digest,
        lifecycle_generation=jnp.asarray(0, dtype=jnp.int32),
        lifecycle_revision=jnp.asarray(0, dtype=jnp.int32),
    )
    request = ProceduralMemoryRequest(
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
    policy = _policy()
    mask = jnp.asarray([True, False, True], dtype=jnp.bool_)

    first = memory.procedural_step(state, request, record)
    first_proposal = policy.propose(
        first.retrieval,
        hard_safety_mask=mask,
        expected_lifecycle_digest=lifecycle_digest,
        expected_lifecycle_generation=0,
        expected_lifecycle_revision=0,
    )
    assert not bool(first.retrieval.accepted)
    assert not bool(first_proposal.available)

    second = memory.procedural_step(first.state, request, record)
    second_proposal = policy.propose(
        second.retrieval,
        hard_safety_mask=mask,
        expected_lifecycle_digest=lifecycle_digest,
        expected_lifecycle_generation=0,
        expected_lifecycle_revision=0,
    )
    assert bool(second.retrieval.accepted)
    assert int(second.retrieval.evidence_count) == 1
    assert not bool(second_proposal.available)

    third = memory.procedural_step(second.state, request, record)
    before_policy = third.state
    third_proposal = policy.propose(
        third.retrieval,
        hard_safety_mask=mask,
        expected_lifecycle_digest=lifecycle_digest,
        expected_lifecycle_generation=0,
        expected_lifecycle_revision=0,
    )
    assert bool(third.retrieval.accepted)
    assert int(third.retrieval.evidence_count) == 2
    assert bool(third_proposal.available)
    assert int(third_proposal.action) == 2
    assert not bool(third_proposal.hard_safety_mask[1])
    assert _tree_equal(before_policy, third.state)

    valid = memory.validate_state(
        third.state,
        source_digest=source_digest,
        semantic_namespace_digest=namespace_digest,
        representation_revision=1,
        source_revision=1,
    )
    assert bool(valid)
    assert not CONSOLIDATED_MEMORY_ACTION_SELECTION_AUTHORITY
    assert not CONSOLIDATED_MEMORY_AGENT_MUTATION_AUTHORITY
    assert policy.resource_budget.memory_queries_per_proposal == 0
    assert policy.resource_budget.memory_writes_per_proposal == 0
    assert policy.resource_budget.action_dispatches_per_proposal == 0


def test_policy_call_is_pure_and_has_no_rng_or_checkpoint_state() -> None:
    policy = _policy()
    retrieval = _retrieval((0.1, 0.6, 0.4))
    before = jax.tree_util.tree_map(lambda value: value.copy(), retrieval)
    first = policy.propose(
        retrieval,
        hard_safety_mask=jnp.asarray([True, True, True], dtype=jnp.bool_),
        expected_lifecycle_digest=_digest("option-a"),
        expected_lifecycle_generation=2,
        expected_lifecycle_revision=3,
    )
    second = policy.propose(
        retrieval,
        hard_safety_mask=jnp.asarray([True, True, True], dtype=jnp.bool_),
        expected_lifecycle_digest=_digest("option-a"),
        expected_lifecycle_generation=2,
        expected_lifecycle_revision=3,
    )
    assert _tree_equal(retrieval, before)
    assert _tree_equal(first, second)
    assert policy.resource_budget.persistent_state_bytes == 0
    assert policy.resource_budget.checkpoint_bytes == 0
    assert policy.resource_budget.random_generator_calls_per_proposal == 0
