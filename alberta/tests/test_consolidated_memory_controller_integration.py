# mypy: disable-error-code="arg-type,call-arg,type-var"
"""JAX, lifecycle, checkpoint, and recovery integration for the controller."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import pytest

from alberta_framework.core.consolidated_memory import (
    ConsolidatedMemoryConfig,
    ProceduralMemoryRequest,
    canonical_memory_digest,
)
from alberta_framework.core.consolidated_memory_controller import (
    MEMORY_ERROR_WRITE_REJECTED,
    ConsolidatedProceduralMemoryController,
    ConsolidatedProceduralMemoryControllerConfig,
    ConsolidatedProceduralMemoryControllerState,
)
from alberta_framework.core.consolidated_memory_policy import (
    ConsolidatedProceduralMemoryPolicyConfig,
)

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def _digest(text: str) -> jax.Array:
    return canonical_memory_digest("test.consolidated-memory-controller.integration", text)


def _controller(
    max_operations: int = 100,
) -> ConsolidatedProceduralMemoryController:
    return ConsolidatedProceduralMemoryController(
        ConsolidatedProceduralMemoryControllerConfig(
            memory=ConsolidatedMemoryConfig(
                semantic_capacity=1,
                procedural_capacity=2,
                semantic_payload_dim=1,
                procedural_payload_dim=3,
                procedural_outcome_dim=2,
                semantic_max_age=min(20, max_operations),
                procedural_max_age=min(20, max_operations),
                max_operations=max_operations,
                semantic_min_confidence=0.0,
                procedural_min_confidence=0.0,
            ),
            policy=ConsolidatedProceduralMemoryPolicyConfig(
                n_actions=3,
                outcome_dim=2,
                min_evidence_count=2,
                min_success_lower_bound=0.0,
                wilson_z=1.0,
                max_outcome_standard_error=10.0,
                max_abs_outcome_mean=100.0,
            ),
        )
    )


def _initial(
    controller: ConsolidatedProceduralMemoryController,
    *, source: str = "source-a",
    namespace: str = "namespace-a",
    representation_revision: int = 2,
    source_revision: int = 5,
) -> ConsolidatedProceduralMemoryControllerState:
    return controller.init(
        source_digest=_digest(source),
        semantic_namespace_digest=_digest(namespace),
        representation_revision=representation_revision,
        source_revision=source_revision,
    )


def _request() -> ProceduralMemoryRequest:
    return ProceduralMemoryRequest(
        semantic_digest=_digest("skill-a"),
        generation=jnp.asarray(0, dtype=jnp.int32),
        provenance_digest=_digest("provenance-a"),
        representation_revision=jnp.asarray(2, dtype=jnp.int32),
        source_revision=jnp.asarray(5, dtype=jnp.int32),
        lifecycle_link_available=jnp.asarray(True, dtype=jnp.bool_),
        lifecycle_digest=_digest("lifecycle-a"),
        lifecycle_generation=jnp.asarray(1, dtype=jnp.int32),
        lifecycle_revision=jnp.asarray(3, dtype=jnp.int32),
    )


def _identity(value: jax.Array | int) -> jax.Array:
    scalar = jnp.asarray(value, dtype=jnp.uint32)
    return scalar + jnp.arange(4, dtype=jnp.uint32)


def _tree_equal(left: object, right: object) -> bool:
    return bool(
        jax.tree_util.tree_all(
            jax.tree_util.tree_map(jnp.array_equal, left, right)
        )
    )


def _compiled_lifecycle_contract() -> None:
    controller = _controller()
    initial = _initial(controller)
    request = _request()
    base_actions = jnp.asarray((2, 2, 0), dtype=jnp.int32)
    decision_ids = jnp.asarray((1, 10, 20), dtype=jnp.uint32)
    event_ids = jnp.asarray((101, 110, 120), dtype=jnp.uint32)
    outcomes = jnp.asarray(((1.0, 0.5), (1.0, 0.5), (1.0, 0.5)), dtype=jnp.float32)

    def lifecycle_step(
        state: ConsolidatedProceduralMemoryControllerState,
        decision_word: jax.Array,
        event_word: jax.Array,
        base_action: jax.Array,
        outcome: jax.Array,
    ) -> tuple[
        ConsolidatedProceduralMemoryControllerState,
        tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    ]:
        decision = controller.decide(
            state,
            decision_id=_identity(decision_word),
            base_categorical_score_mass=jax.nn.one_hot(
                base_action, 3, dtype=jnp.float32
            ),
            base_action=base_action,
            base_action_available=jnp.asarray(True, dtype=jnp.bool_),
            hard_safety_mask=jnp.ones((3,), dtype=jnp.bool_),
            request=request,
        )
        pre_feedback_writes = decision.state.memory.procedural_write_count
        feedback = controller.feedback(
            decision.state,
            decision_id=_identity(decision_word),
            feedback_event_id=_identity(event_word),
            base_action=base_action,
            effective_action=decision.action,
            request=request,
            succeeded=jnp.asarray(True, dtype=jnp.bool_),
            outcome=outcome,
            confidence=jnp.asarray(1.0, dtype=jnp.float32),
            evidence=jnp.asarray(1.0, dtype=jnp.float32),
        )
        return feedback.state, (
            decision.action,
            decision.diagnostics.memory_selected,
            pre_feedback_writes,
            feedback.state.memory.procedural_write_count,
        )

    eager_state = initial
    eager_trace: list[tuple[jax.Array, jax.Array, jax.Array, jax.Array]] = []
    with jax.disable_jit():
        for values in zip(decision_ids, event_ids, base_actions, outcomes, strict=True):
            eager_state, trace = lifecycle_step(eager_state, *values)
            eager_trace.append(trace)
    compiled_state = initial
    compiled_trace_list: list[tuple[jax.Array, jax.Array, jax.Array, jax.Array]] = []
    for values in zip(decision_ids, event_ids, base_actions, outcomes, strict=True):
        compiled_state, trace = lifecycle_step(compiled_state, *values)
        compiled_trace_list.append(trace)
    assert _tree_equal(eager_state, compiled_state)
    assert _tree_equal(eager_trace, compiled_trace_list)
    compiled_trace = jax.tree_util.tree_map(
        lambda *values: jnp.stack(values), *compiled_trace_list
    )
    actions, selected, pre_feedback_writes, post_feedback_writes = compiled_trace
    assert actions.tolist() == [2, 2, 2]
    assert selected.tolist() == [False, False, True]
    assert pre_feedback_writes.tolist() == [0, 1, 2]
    assert post_feedback_writes.tolist() == [1, 2, 3]
    assert int(compiled_state.memory.procedural_query_count) == 3
    assert int(compiled_state.memory.procedural_write_count) == 3

    stale_decision = jax.jit(controller.decide)(
        compiled_state,
        decision_id=_identity(10),
        base_categorical_score_mass=jnp.asarray((0.0, 1.0, 0.0), dtype=jnp.float32),
        base_action=jnp.asarray(1, dtype=jnp.int32),
        base_action_available=jnp.asarray(True, dtype=jnp.bool_),
        hard_safety_mask=jnp.ones((3,), dtype=jnp.bool_),
        request=request,
    )
    assert bool(stale_decision.action_available)
    assert int(stale_decision.action) == 1
    assert bool(stale_decision.diagnostics.stale_decision)
    assert _tree_equal(stale_decision.state, compiled_state)

    pending = controller.decide(
        compiled_state,
        decision_id=_identity(200),
        base_categorical_score_mass=jnp.asarray((1.0, 0.0, 0.0), dtype=jnp.float32),
        base_action=0,
        base_action_available=True,
        hard_safety_mask=jnp.ones((3,), dtype=jnp.bool_),
        request=request,
    ).state
    stale_feedback = jax.jit(controller.feedback)(
        pending,
        decision_id=_identity(200),
        feedback_event_id=_identity(110),
        base_action=jnp.asarray(0, dtype=jnp.int32),
        effective_action=pending.pending_effective_action,
        request=request,
        succeeded=jnp.asarray(True, dtype=jnp.bool_),
        outcome=jnp.asarray((1.0, 0.5), dtype=jnp.float32),
        confidence=jnp.asarray(1.0, dtype=jnp.float32),
        evidence=jnp.asarray(1.0, dtype=jnp.float32),
    )
    assert bool(stale_feedback.diagnostics.stale_feedback_event)
    assert not bool(stale_feedback.diagnostics.write_applied)
    assert _tree_equal(stale_feedback.state, pending)

    def pending_scan_body(
        state: ConsolidatedProceduralMemoryControllerState,
        values: tuple[jax.Array, jax.Array],
    ) -> tuple[ConsolidatedProceduralMemoryControllerState, jax.Array]:
        identity, base_action = values
        decision = controller.decide(
            state,
            decision_id=_identity(identity),
            base_categorical_score_mass=jax.nn.one_hot(
                base_action, 3, dtype=jnp.float32
            ),
            base_action=base_action,
            base_action_available=jnp.asarray(True, dtype=jnp.bool_),
            hard_safety_mask=jnp.ones((3,), dtype=jnp.bool_),
            request=request,
        )
        return decision.state, decision.action

    scanned_state, scanned_actions = jax.jit(
        lambda state, values: jax.lax.scan(pending_scan_body, state, values)
    )(
        pending,
        (
            jnp.asarray((210, 220, 230), dtype=jnp.uint32),
            jnp.asarray((0, 1, 2), dtype=jnp.int32),
        ),
    )
    assert scanned_actions.tolist() == [0, 1, 2]
    assert _tree_equal(scanned_state, pending)


def test_checkpoint_resumes_exact_pending_identity_and_detects_tampering() -> None:
    controller = _controller()
    source = _digest("source-a")
    namespace = _digest("namespace-a")
    with jax.disable_jit():
        decision = controller.decide(
            _initial(controller),
            decision_id=_identity(1),
            base_categorical_score_mass=jnp.asarray(
                (0.0, 0.0, 1.0), dtype=jnp.float32
            ),
            base_action=2,
            base_action_available=True,
            hard_safety_mask=jnp.ones((3,), dtype=jnp.bool_),
            request=_request(),
        )
    checkpoint = controller.checkpoint_payload(
        decision.state,
        source_digest=source,
        semantic_namespace_digest=namespace,
        representation_revision=2,
        source_revision=5,
    )
    restored = controller.restore_checkpoint(
        checkpoint,
        source_digest=source,
        semantic_namespace_digest=namespace,
        representation_revision=2,
        source_revision=5,
    )
    assert _tree_equal(restored, decision.state)

    with jax.disable_jit():
        wrong_identity = controller.feedback(
            restored,
            decision_id=_identity(2),
            feedback_event_id=_identity(100),
            base_action=2,
            effective_action=2,
            request=_request(),
            succeeded=True,
            outcome=jnp.asarray((1.0, 0.0), dtype=jnp.float32),
            confidence=1.0,
            evidence=1.0,
        )
        assert not bool(wrong_identity.diagnostics.write_applied)
        assert _tree_equal(wrong_identity.state, restored)
        resumed = controller.feedback(
            restored,
            decision_id=_identity(1),
            feedback_event_id=_identity(100),
            base_action=2,
            effective_action=2,
            request=_request(),
            succeeded=True,
            outcome=jnp.asarray((1.0, 0.0), dtype=jnp.float32),
            confidence=1.0,
            evidence=1.0,
        )
        uninterrupted = controller.feedback(
            decision.state,
            decision_id=_identity(1),
            feedback_event_id=_identity(100),
            base_action=2,
            effective_action=2,
            request=_request(),
            succeeded=True,
            outcome=jnp.asarray((1.0, 0.0), dtype=jnp.float32),
            confidence=1.0,
            evidence=1.0,
        )
    assert _tree_equal(resumed, uninterrupted)

    sha_tamper = dict(checkpoint)
    sha_tamper["state"] = dataclasses.replace(
        restored,
        decision_count=restored.decision_count + jnp.asarray(1, dtype=jnp.int32),
    )
    with pytest.raises(ValueError, match="state SHA"):
        controller.restore_checkpoint(
            sha_tamper,
            source_digest=source,
            semantic_namespace_digest=namespace,
            representation_revision=2,
            source_revision=5,
        )

    checksum_tamper = dict(checkpoint)
    altered = dataclasses.replace(
        restored,
        checksum=restored.checksum.at[0].add(jnp.asarray(1, dtype=jnp.uint32)),
    )
    checksum_tamper["state"] = altered
    checksum_tamper["state_sha256"] = controller._state_sha256(altered)
    with pytest.raises(ValueError, match="invalid or stale"):
        controller.restore_checkpoint(
            checksum_tamper,
            source_digest=source,
            semantic_namespace_digest=namespace,
            representation_revision=2,
            source_revision=5,
        )

    with pytest.raises(ValueError, match="binding"):
        controller.restore_checkpoint(
            checkpoint,
            source_digest=_digest("wrong-source"),
            semantic_namespace_digest=namespace,
            representation_revision=2,
            source_revision=5,
        )


def test_rebind_reset_is_explicit_and_requires_pending_discard_authority() -> None:
    controller = _controller()
    with jax.disable_jit():
        pending = controller.decide(
            _initial(controller),
            decision_id=_identity(1),
            base_categorical_score_mass=jnp.asarray(
                (0.0, 0.0, 1.0), dtype=jnp.float32
            ),
            base_action=2,
            base_action_available=True,
            hard_safety_mask=jnp.ones((3,), dtype=jnp.bool_),
            request=_request(),
        ).state
    with pytest.raises(ValueError, match="discard_pending=True"):
        controller.rebind_reset(
            pending,
            source_digest=_digest("source-b"),
            semantic_namespace_digest=_digest("namespace-b"),
            representation_revision=7,
            source_revision=8,
        )
    rebound = controller.rebind_reset(
        pending,
        source_digest=_digest("source-b"),
        semantic_namespace_digest=_digest("namespace-b"),
        representation_revision=7,
        source_revision=8,
        discard_pending=True,
    )
    assert not bool(rebound.pending)
    assert int(rebound.memory.operation_count) == 0
    assert rebound.memory.procedural.payload_means.shape == (2, 3)
    assert bool(
        controller.validate_state(
            rebound,
            source_digest=_digest("source-b"),
            semantic_namespace_digest=_digest("namespace-b"),
            representation_revision=7,
            source_revision=8,
        )
    )
    assert not bool(
        controller.validate_state(
            rebound,
            source_digest=_digest("source-a"),
            semantic_namespace_digest=_digest("namespace-a"),
            representation_revision=2,
            source_revision=5,
        )
    )


def test_rejected_feedback_write_clears_pending_and_preserves_base_control() -> None:
    controller = _controller()
    generation_zero = _request()
    with jax.disable_jit():
        first = controller.decide(
            _initial(controller),
            decision_id=_identity(1),
            base_categorical_score_mass=jnp.asarray(
                (0.0, 0.0, 1.0), dtype=jnp.float32
            ),
            base_action=2,
            base_action_available=True,
            hard_safety_mask=jnp.ones((3,), dtype=jnp.bool_),
            request=generation_zero,
        )
        first_feedback = controller.feedback(
            first.state,
            decision_id=_identity(1),
            feedback_event_id=_identity(100),
            base_action=2,
            effective_action=2,
            request=generation_zero,
            succeeded=True,
            outcome=jnp.asarray((1.0, 0.0), dtype=jnp.float32),
            confidence=1.0,
            evidence=1.0,
        )
        assert bool(first_feedback.diagnostics.write_applied)
        skipped_generation = dataclasses.replace(
            generation_zero, generation=jnp.asarray(2, dtype=jnp.int32)
        )
        decision = controller.decide(
            first_feedback.state,
            decision_id=_identity(10),
            base_categorical_score_mass=jnp.asarray(
                (0.0, 0.0, 1.0), dtype=jnp.float32
            ),
            base_action=2,
            base_action_available=True,
            hard_safety_mask=jnp.ones((3,), dtype=jnp.bool_),
            request=skipped_generation,
        )
        assert bool(decision.diagnostics.query_transaction_applied)
        assert not bool(decision.diagnostics.query_accepted)
        feedback = controller.feedback(
            decision.state,
            decision_id=_identity(10),
            feedback_event_id=_identity(110),
            base_action=2,
            effective_action=2,
            request=skipped_generation,
            succeeded=True,
            outcome=jnp.asarray((1.0, 0.0), dtype=jnp.float32),
            confidence=1.0,
            evidence=1.0,
        )
        assert not bool(feedback.diagnostics.write_applied)
        assert bool(feedback.write.identity_collision)
        assert not bool(feedback.write.generation_compatible)
        assert bool(feedback.diagnostics.pending_cleared)
        assert bool(feedback.state.memory_unavailable)
        assert int(feedback.state.memory_error) == MEMORY_ERROR_WRITE_REJECTED
        assert int(feedback.state.memory.procedural_write_count) == 1
        later = controller.decide(
            feedback.state,
            decision_id=_identity(20),
            base_categorical_score_mass=jnp.asarray(
                (0.0, 1.0, 0.0), dtype=jnp.float32
            ),
            base_action=1,
            base_action_available=True,
            hard_safety_mask=jnp.ones((3,), dtype=jnp.bool_),
            request=dataclasses.replace(
                skipped_generation,
                lifecycle_link_available=jnp.asarray(False, dtype=jnp.bool_),
                lifecycle_digest=jnp.zeros((32,), dtype=jnp.uint8),
                lifecycle_generation=jnp.asarray(-1, dtype=jnp.int32),
                lifecycle_revision=jnp.asarray(-1, dtype=jnp.int32),
            ),
        )
    assert bool(later.action_available)
    assert int(later.action) == 1
    assert not bool(later.diagnostics.query_attempted)
    assert _tree_equal(later.state.memory, feedback.state.memory)


def test_complete_lifecycle_has_eager_jitted_scan_parity_and_causal_counts() -> None:
    _compiled_lifecycle_contract()
