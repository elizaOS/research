# mypy: disable-error-code="arg-type,call-arg,type-var"
"""Unit contracts for causal consolidated procedural-memory composition."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator

import jax
import jax.numpy as jnp
import pytest

from alberta_framework.core.consolidated_memory import (
    ConsolidatedMemoryConfig,
    ProceduralMemoryRequest,
    canonical_memory_digest,
)
from alberta_framework.core.consolidated_memory_controller import (
    CONSOLIDATED_MEMORY_CONTROLLER_ACTION_DISPATCH_AUTHORITY,
    CONSOLIDATED_MEMORY_CONTROLLER_AGENT_MUTATION_AUTHORITY,
    CONSOLIDATED_MEMORY_CONTROLLER_AUTONOMOUS_SKILL_CREATION_AUTHORITY,
    CONSOLIDATED_MEMORY_CONTROLLER_PROMOTION_AUTHORITY,
    CONSOLIDATED_MEMORY_CONTROLLER_SCIENTIFIC_PROMOTION_ALLOWED,
    DECISION_REASON_MEMORY_UNAVAILABLE_FALLBACK,
    FEEDBACK_REASON_NONFINITE,
    MEMORY_ERROR_CAP_EXHAUSTED,
    MEMORY_ERROR_COMPOSED_STATE_INVALID,
    ConsolidatedProceduralMemoryController,
    ConsolidatedProceduralMemoryControllerConfig,
    ConsolidatedProceduralMemoryControllerState,
    ConsolidatedProceduralMemoryDecisionResult,
)
from alberta_framework.core.consolidated_memory_policy import (
    ConsolidatedProceduralMemoryPolicyConfig,
)

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _run_unit_contracts_without_compilation() -> Iterator[None]:
    with jax.disable_jit():
        yield


def _digest(text: str) -> jax.Array:
    return canonical_memory_digest("test.consolidated-memory-controller", text)


def _controller(
    *, max_operations: int = 100
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
) -> ConsolidatedProceduralMemoryControllerState:
    return controller.init(
        source_digest=_digest(source),
        semantic_namespace_digest=_digest(namespace),
        representation_revision=2,
        source_revision=5,
    )


def _request(
    *,
    provenance: str = "provenance-a",
    lifecycle: str = "lifecycle-a",
    lifecycle_link_available: bool = True,
) -> ProceduralMemoryRequest:
    return ProceduralMemoryRequest(
        semantic_digest=_digest("skill-a"),
        generation=jnp.asarray(0, dtype=jnp.int32),
        provenance_digest=_digest(provenance),
        representation_revision=jnp.asarray(2, dtype=jnp.int32),
        source_revision=jnp.asarray(5, dtype=jnp.int32),
        lifecycle_link_available=jnp.asarray(
            lifecycle_link_available, dtype=jnp.bool_
        ),
        lifecycle_digest=(
            _digest(lifecycle)
            if lifecycle_link_available
            else jnp.zeros((32,), dtype=jnp.uint8)
        ),
        lifecycle_generation=jnp.asarray(
            1 if lifecycle_link_available else -1, dtype=jnp.int32
        ),
        lifecycle_revision=jnp.asarray(
            3 if lifecycle_link_available else -1, dtype=jnp.int32
        ),
    )


def _decision_id(value: int) -> jax.Array:
    return jnp.asarray((value, value + 1, value + 2, value + 3), dtype=jnp.uint32)


def _tree_equal(left: object, right: object) -> bool:
    return bool(
        jax.tree_util.tree_all(
            jax.tree_util.tree_map(jnp.array_equal, left, right)
        )
    )


def _decide(
    controller: ConsolidatedProceduralMemoryController,
    state: ConsolidatedProceduralMemoryControllerState,
    *,
    identity: int,
    base_action: int,
    request: ProceduralMemoryRequest | None = None,
    mask: tuple[bool, bool, bool] = (True, True, True),
) -> ConsolidatedProceduralMemoryDecisionResult:
    scores = jnp.zeros((3,), dtype=jnp.float32).at[base_action].set(1.0)
    return controller.decide(
        state,
        decision_id=_decision_id(identity),
        base_categorical_score_mass=scores,
        base_action=base_action,
        base_action_available=True,
        hard_safety_mask=jnp.asarray(mask, dtype=jnp.bool_),
        request=_request() if request is None else request,
    )


def test_config_resource_budget_and_state_are_explicit_and_non_authoritative() -> None:
    controller = _controller()
    state = _initial(controller)
    config = controller.to_config()
    assert ConsolidatedProceduralMemoryController.from_config(config).to_config() == config
    assert not CONSOLIDATED_MEMORY_CONTROLLER_ACTION_DISPATCH_AUTHORITY
    assert not CONSOLIDATED_MEMORY_CONTROLLER_AGENT_MUTATION_AUTHORITY
    assert not CONSOLIDATED_MEMORY_CONTROLLER_AUTONOMOUS_SKILL_CREATION_AUTHORITY
    assert not CONSOLIDATED_MEMORY_CONTROLLER_PROMOTION_AUTHORITY
    assert not CONSOLIDATED_MEMORY_CONTROLLER_SCIENTIFIC_PROMOTION_ALLOWED
    budget = controller.resource_budget
    assert budget.pending_slots == 1
    assert budget.pending_cancellation_identity_checks_per_call == 2
    assert budget.memory_writes_per_dispatch_cancellation == 0
    assert budget.counter_advances_per_dispatch_cancellation == 0
    assert budget.learning_updates_per_dispatch_cancellation == 0
    assert budget.memory_operations_per_complete_lifecycle == 2
    assert budget.random_generator_calls_per_event == 0
    assert budget.persistent_growth_per_event_bytes == 0
    assert budget.caller_base_fallback_guaranteed_when_memory_unavailable
    assert bool(
        controller.validate_state(
            state,
            source_digest=_digest("source-a"),
            semantic_namespace_digest=_digest("namespace-a"),
            representation_revision=2,
            source_revision=5,
        )
    )


def test_dispatch_cancellation_is_exact_owner_zero_work_and_fail_closed() -> None:
    controller = _controller()
    pending = _decide(
        controller,
        _initial(controller),
        identity=11,
        base_action=2,
    ).state
    assert bool(pending.pending)
    memory_before = pending.memory
    counters_before = (
        pending.last_decision_id,
        pending.last_feedback_event_id,
    )

    stale_identity = _decision_id(12)
    stale = controller.cancel_pending_dispatch(
        pending,
        cancellation_requested=jnp.asarray(True, dtype=jnp.bool_),
        decision_id=stale_identity,
        effective_action=pending.pending_effective_action,
    )
    assert not bool(stale.diagnostics.cancellation_applied)
    assert not bool(stale.diagnostics.transaction_satisfied)
    assert _tree_equal(stale.state, pending)

    canceled = controller.cancel_pending_dispatch(
        pending,
        cancellation_requested=jnp.asarray(True, dtype=jnp.bool_),
        decision_id=_decision_id(11),
        effective_action=pending.pending_effective_action,
    )
    assert bool(canceled.diagnostics.cancellation_applied)
    assert bool(canceled.diagnostics.transaction_satisfied)
    assert not bool(canceled.state.pending)
    assert _tree_equal(canceled.state.memory, memory_before)
    assert (canceled.state.last_decision_id, canceled.state.last_feedback_event_id) == (
        counters_before
    )
    assert bool(canceled.diagnostics.memory_unchanged)
    assert bool(canceled.diagnostics.counters_unchanged)
    assert not bool(canceled.diagnostics.learning_applied)
    assert not bool(canceled.diagnostics.evidence_written)


def test_query_precedes_feedback_write_and_recall_changes_later_safe_action() -> None:
    controller = _controller()
    state = _initial(controller)
    request = _request()
    for index in range(2):
        decision = _decide(
            controller,
            state,
            identity=10 * index + 1,
            base_action=2,
            request=request,
        )
        assert bool(decision.diagnostics.query_transaction_applied)
        assert bool(decision.diagnostics.query_pre_write_verified)
        assert int(decision.state.memory.procedural_write_count) == index
        feedback = controller.feedback(
            decision.state,
            decision_id=_decision_id(10 * index + 1),
            feedback_event_id=_decision_id(100 + 10 * index),
            base_action=2,
            effective_action=decision.action,
            request=request,
            succeeded=True,
            outcome=jnp.asarray((1.0, 0.5), dtype=jnp.float32),
            confidence=1.0,
            evidence=1.0,
        )
        assert bool(feedback.diagnostics.write_applied)
        assert int(feedback.state.memory.procedural_write_count) == index + 1
        state = feedback.state

    later = _decide(controller, state, identity=30, base_action=0, request=request)
    assert int(later.retrieval.evidence_count) == 2
    assert bool(later.diagnostics.memory_selected)
    assert int(later.counterfactual_base_action) == 0
    assert int(later.memory_proposed_action) == 2
    assert int(later.action) == 2
    assert int(later.state.memory.procedural_query_count) == 3
    assert int(later.state.memory.procedural_write_count) == 2


def test_base_control_must_be_finite_positive_indexed_and_hard_safe() -> None:
    controller = _controller()
    state = _initial(controller)
    common = {
        "decision_id": _decision_id(1),
        "base_action_available": True,
        "request": _request(),
    }
    nan_scores = controller.decide(
        state,
        base_categorical_score_mass=jnp.asarray(
            (float("nan"), 0.0, 0.0), dtype=jnp.float32
        ),
        base_action=0,
        hard_safety_mask=jnp.ones((3,), dtype=jnp.bool_),
        **common,
    )
    assert not bool(nan_scores.action_available)
    assert _tree_equal(nan_scores.state, state)

    unsafe = controller.decide(
        state,
        base_categorical_score_mass=jnp.asarray((1.0, 0.0, 0.0), dtype=jnp.float32),
        base_action=0,
        hard_safety_mask=jnp.asarray((False, True, True), dtype=jnp.bool_),
        **common,
    )
    assert not bool(unsafe.action_available)
    assert not bool(unsafe.diagnostics.base_action_hard_safe)
    assert _tree_equal(unsafe.state, state)

    no_lifecycle = _decide(
        controller,
        state,
        identity=2,
        base_action=0,
        request=_request(lifecycle_link_available=False),
    )
    assert not bool(no_lifecycle.action_available)
    assert _tree_equal(no_lifecycle.state, state)


def test_pending_decision_returns_exact_untracked_base_without_state_mutation() -> None:
    controller = _controller()
    initial = _initial(controller)
    first = _decide(controller, initial, identity=1, base_action=2)
    assert bool(first.diagnostics.feedback_trackable)
    conflict = _decide(controller, first.state, identity=20, base_action=0)
    assert bool(conflict.action_available)
    assert int(conflict.action) == 0
    assert int(conflict.counterfactual_base_action) == 0
    assert bool(conflict.diagnostics.pending_conflict)
    assert not bool(conflict.diagnostics.feedback_trackable)
    assert _tree_equal(conflict.state, first.state)

    invalid_optional_request = _decide(
        controller,
        first.state,
        identity=30,
        base_action=1,
        request=_request(lifecycle_link_available=False),
    )
    assert bool(invalid_optional_request.action_available)
    assert int(invalid_optional_request.action) == 1
    assert bool(invalid_optional_request.diagnostics.pending_conflict)
    assert _tree_equal(invalid_optional_request.state, first.state)

    duplicate = _decide(
        controller,
        first.state,
        identity=1,
        base_action=1,
        request=_request(lifecycle_link_available=False),
    )
    assert bool(duplicate.action_available)
    assert int(duplicate.action) == 1
    assert bool(duplicate.diagnostics.duplicate_decision)
    assert not bool(duplicate.diagnostics.feedback_trackable)
    assert _tree_equal(duplicate.state, first.state)

    stale = _decide(
        controller,
        first.state,
        identity=0,
        base_action=1,
        request=_request(lifecycle_link_available=False),
    )
    assert bool(stale.action_available)
    assert int(stale.action) == 1
    assert bool(stale.diagnostics.stale_decision)
    assert not bool(stale.diagnostics.decision_id_strictly_advancing)
    assert _tree_equal(stale.state, first.state)


def test_misattributed_duplicate_and_nonfinite_feedback_are_atomic_noops() -> None:
    controller = _controller()
    decision = _decide(controller, _initial(controller), identity=1, base_action=2)
    state = decision.state
    request = _request()
    cases = (
        {"decision_id": _decision_id(90)},
        {"base_action": 1},
        {"effective_action": 1},
        {"request": _request(provenance="wrong")},
        {"outcome": jnp.asarray((float("nan"), 0.0), dtype=jnp.float32)},
    )
    for overrides in cases:
        kwargs = {
            "decision_id": _decision_id(1),
            "feedback_event_id": _decision_id(100),
            "base_action": 2,
            "effective_action": 2,
            "request": request,
            "succeeded": True,
            "outcome": jnp.asarray((1.0, 0.0), dtype=jnp.float32),
            "confidence": 1.0,
            "evidence": 1.0,
        }
        kwargs.update(overrides)
        rejected = controller.feedback(state, **kwargs)
        assert not bool(rejected.diagnostics.write_applied)
        assert not bool(rejected.write.wrote)
        assert _tree_equal(rejected.state.memory, state.memory)
        assert _tree_equal(rejected.state, state)
    nonfinite = controller.feedback(
        state,
        decision_id=_decision_id(1),
        feedback_event_id=_decision_id(100),
        base_action=2,
        effective_action=2,
        request=request,
        succeeded=True,
        outcome=jnp.asarray((float("nan"), 0.0), dtype=jnp.float32),
        confidence=1.0,
        evidence=1.0,
    )
    assert int(nonfinite.diagnostics.reason) == FEEDBACK_REASON_NONFINITE

    applied = controller.feedback(
        state,
        decision_id=_decision_id(1),
        feedback_event_id=_decision_id(100),
        base_action=2,
        effective_action=2,
        request=request,
        succeeded=False,
        outcome=jnp.asarray((-1.0, 0.0), dtype=jnp.float32),
        confidence=1.0,
        evidence=1.0,
    )
    assert bool(applied.diagnostics.failure_recorded)
    duplicate = controller.feedback(
        applied.state,
        decision_id=_decision_id(1),
        feedback_event_id=_decision_id(100),
        base_action=2,
        effective_action=2,
        request=request,
        succeeded=True,
        outcome=jnp.asarray((1.0, 0.0), dtype=jnp.float32),
        confidence=1.0,
        evidence=1.0,
    )
    assert not bool(duplicate.diagnostics.write_applied)
    assert _tree_equal(duplicate.state, applied.state)

    next_decision = _decide(
        controller, applied.state, identity=2, base_action=2, request=request
    )
    stale_event = controller.feedback(
        next_decision.state,
        decision_id=_decision_id(2),
        feedback_event_id=_decision_id(50),
        base_action=2,
        effective_action=2,
        request=request,
        succeeded=True,
        outcome=jnp.asarray((1.0, 0.0), dtype=jnp.float32),
        confidence=1.0,
        evidence=1.0,
    )
    assert bool(stale_event.diagnostics.stale_feedback_event)
    assert not bool(stale_event.diagnostics.feedback_event_strictly_advancing)
    assert not bool(stale_event.diagnostics.write_applied)
    assert _tree_equal(stale_event.state, next_decision.state)


def test_operation_exhaustion_freezes_only_memory_and_base_decisions_continue() -> None:
    controller = _controller(max_operations=2)
    first = _decide(controller, _initial(controller), identity=1, base_action=2)
    feedback = controller.feedback(
        first.state,
        decision_id=_decision_id(1),
        feedback_event_id=_decision_id(100),
        base_action=2,
        effective_action=2,
        request=_request(),
        succeeded=True,
        outcome=jnp.asarray((1.0, 0.0), dtype=jnp.float32),
        confidence=1.0,
        evidence=1.0,
    )
    assert bool(feedback.diagnostics.write_applied)
    assert bool(feedback.state.memory_unavailable)
    assert int(feedback.state.memory_error) == MEMORY_ERROR_CAP_EXHAUSTED
    frozen_memory = feedback.state.memory

    later = _decide(
        controller,
        feedback.state,
        identity=20,
        base_action=0,
        request=_request(lifecycle_link_available=False),
    )
    assert bool(later.action_available)
    assert int(later.action) == 0
    assert int(later.diagnostics.reason) == DECISION_REASON_MEMORY_UNAVAILABLE_FALLBACK
    assert not bool(later.diagnostics.query_attempted)
    assert _tree_equal(later.state.memory, frozen_memory)
    assert int(later.state.memory_unavailable_noop_count) == 1

    no_complete_lifecycle = _controller(max_operations=1)
    exhausted_query = _decide(
        no_complete_lifecycle,
        _initial(no_complete_lifecycle),
        identity=1,
        base_action=1,
    )
    assert bool(exhausted_query.action_available)
    assert int(exhausted_query.action) == 1
    assert bool(exhausted_query.diagnostics.query_attempted)
    assert not bool(exhausted_query.diagnostics.query_transaction_applied)
    assert bool(exhausted_query.state.memory_unavailable)
    assert int(exhausted_query.state.memory_error) == MEMORY_ERROR_CAP_EXHAUSTED
    assert int(exhausted_query.state.memory.operation_count) == 0


def test_checksum_corruption_fails_closed_and_cannot_be_reset() -> None:
    controller = _controller()
    state = _initial(controller)
    corrupted = dataclasses.replace(
        state, decision_count=state.decision_count + jnp.asarray(1, dtype=jnp.int32)
    )
    result = _decide(controller, corrupted, identity=1, base_action=0)
    assert not bool(result.action_available)
    assert not bool(result.diagnostics.checksum_valid)
    assert bool(result.diagnostics.memory_became_unavailable)
    assert int(result.diagnostics.memory_error) == MEMORY_ERROR_COMPOSED_STATE_INVALID
    assert _tree_equal(result.state, corrupted)

    pending = _decide(controller, state, identity=1, base_action=2).state
    corrupted_pending = dataclasses.replace(
        pending,
        checksum=pending.checksum.at[0].add(jnp.asarray(1, dtype=jnp.uint32)),
    )
    feedback = controller.feedback(
        corrupted_pending,
        decision_id=_decision_id(1),
        feedback_event_id=_decision_id(100),
        base_action=2,
        effective_action=2,
        request=_request(),
        succeeded=True,
        outcome=jnp.asarray((1.0, 0.0), dtype=jnp.float32),
        confidence=1.0,
        evidence=1.0,
    )
    assert not bool(feedback.diagnostics.write_applied)
    assert bool(feedback.diagnostics.memory_became_unavailable)
    assert int(feedback.diagnostics.memory_error) == MEMORY_ERROR_COMPOSED_STATE_INVALID
    assert _tree_equal(feedback.state, corrupted_pending)
    with pytest.raises(ValueError, match="corrupted"):
        controller.rebind_reset(
            corrupted,
            source_digest=_digest("source-b"),
            semantic_namespace_digest=_digest("namespace-b"),
            representation_revision=3,
            source_revision=6,
        )
