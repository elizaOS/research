# mypy: disable-error-code="arg-type,attr-defined,call-arg,no-any-return,no-untyped-call,type-var"
"""Unit contracts for the live shared-store semantic Prototype consumer."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from test_prototype_consolidated_memory import (
    _agent as _procedural_agent,
)
from test_prototype_consolidated_memory import (
    _decision_input,
    _digest,
    _feedback,
    _settlement,
    _tree_equal,
)

import alberta_framework as alberta
import alberta_framework.core as core
from alberta_framework.core.consolidated_memory import (
    SEMANTIC_KIND_FACT,
    SemanticMemoryRecord,
    SemanticMemoryRequest,
)
from alberta_framework.core.consolidated_memory_controller import (
    SEMANTIC_REASON_MEMORY_UNAVAILABLE,
)
from alberta_framework.core.prototype_agent import (
    GRUPerceptionConfig,
    PrototypeTransition,
)
from alberta_framework.core.prototype_consolidated_semantic_memory import (
    PROTOTYPE_CONSOLIDATED_SEMANTIC_MEMORY_CHECKPOINT_HOST_ONLY,
    PROTOTYPE_CONSOLIDATED_SEMANTIC_MEMORY_COMPOSITION_ORDER,
    PROTOTYPE_CONSOLIDATED_SEMANTIC_MEMORY_CONTEXT_INFLUENCE_ENABLED,
    PROTOTYPE_CONSOLIDATED_SEMANTIC_MEMORY_DIRECT_DISPATCH_AUTHORITY,
    PROTOTYPE_CONSOLIDATED_SEMANTIC_MEMORY_EFFICACY_CLAIM,
    PROTOTYPE_CONSOLIDATED_SEMANTIC_MEMORY_EVIDENCE_AUTHORITY,
    PROTOTYPE_CONSOLIDATED_SEMANTIC_MEMORY_PROMOTION_AUTHORITY,
    PROTOTYPE_CONSOLIDATED_SEMANTIC_MEMORY_SAFETY_OVERRIDE_AUTHORITY,
    PrototypeConsolidatedSemanticMemoryAgent,
    PrototypeConsolidatedSemanticMemoryConfig,
    PrototypeConsolidatedSemanticMemoryInput,
    PrototypeConsolidatedSemanticMemoryState,
    PrototypeConsolidatedSemanticTransition,
)
from alberta_framework.core.state_builder import IdentityStateBuilderConfig

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _run_unit_contracts_without_compilation() -> Iterator[None]:
    with jax.disable_jit():
        yield
    jax.clear_caches()


def _agent(
    *,
    max_operations: int = 100,
    semantic_max_age: int | None = None,
    experiential: bool = False,
    partner: bool = False,
) -> PrototypeConsolidatedSemanticMemoryAgent:
    composition = _procedural_agent(
        max_operations=max_operations,
        experiential=experiential,
        partner=partner,
    ).config
    if semantic_max_age is not None:
        controller = dataclasses.replace(
            composition.controller,
            memory=dataclasses.replace(
                composition.controller.memory,
                semantic_max_age=semantic_max_age,
            ),
        )
        composition = dataclasses.replace(composition, controller=controller)
    return PrototypeConsolidatedSemanticMemoryAgent(
        PrototypeConsolidatedSemanticMemoryConfig(
            composition=composition,
            raw_observation_dim=1,
        )
    )


def _initial(
    agent: PrototypeConsolidatedSemanticMemoryAgent,
) -> PrototypeConsolidatedSemanticMemoryState:
    return agent.init(
        jr.key(7),
        source_digest=_digest("source"),
        semantic_namespace_digest=_digest("namespace"),
        representation_revision=0,
        source_revision=0,
        lifecycle_id=jnp.asarray((17, 19), dtype=jnp.uint32),
    )


def _started(
    agent: PrototypeConsolidatedSemanticMemoryAgent,
    *,
    decision: bool = False,
) -> PrototypeConsolidatedSemanticMemoryState:
    initial = _initial(agent)
    return agent.start(
        initial,
        jnp.zeros((1,), dtype=jnp.float32),
        decision_input=(
            _decision_input(initial.composition) if decision else None
        ),
    ).state


def _increment_decision_id(value: jax.Array) -> jax.Array:
    return value.at[3].add(jnp.asarray(1, dtype=jnp.uint32))


def _request(
    *,
    identity: str = "semantic-a",
    provenance: str = "semantic-provenance-a",
    kind: int = SEMANTIC_KIND_FACT,
    generation: int = 0,
    representation_revision: int = 0,
    source_revision: int = 0,
) -> SemanticMemoryRequest:
    return SemanticMemoryRequest(
        semantic_digest=_digest(identity),
        generation=jnp.asarray(generation, dtype=jnp.int32),
        kind=jnp.asarray(kind, dtype=jnp.int32),
        provenance_digest=_digest(provenance),
        representation_revision=jnp.asarray(
            representation_revision, dtype=jnp.int32
        ),
        source_revision=jnp.asarray(source_revision, dtype=jnp.int32),
    )


def _record(
    payload: float = 1.0,
    *,
    identity: str = "semantic-a",
    provenance: str = "semantic-provenance-a",
    kind: int = SEMANTIC_KIND_FACT,
    generation: int = 0,
    representation_revision: int = 0,
    source_revision: int = 0,
) -> SemanticMemoryRecord:
    return SemanticMemoryRecord(
        semantic_digest=_digest(identity),
        generation=jnp.asarray(generation, dtype=jnp.int32),
        kind=jnp.asarray(kind, dtype=jnp.int32),
        payload=jnp.asarray((payload,), dtype=jnp.float32),
        confidence=jnp.asarray(1.0, dtype=jnp.float32),
        provenance_digest=_digest(provenance),
        representation_revision=jnp.asarray(
            representation_revision, dtype=jnp.int32
        ),
        source_revision=jnp.asarray(source_revision, dtype=jnp.int32),
        evidence=jnp.asarray(1.0, dtype=jnp.float32),
    )


def _semantic_input(
    state: PrototypeConsolidatedSemanticMemoryState,
    *,
    request: SemanticMemoryRequest | None = None,
    record: SemanticMemoryRecord | None = None,
) -> PrototypeConsolidatedSemanticMemoryInput:
    current = state.composition.prototype.current_decision_id
    return PrototypeConsolidatedSemanticMemoryInput(
        available=jnp.asarray(True, dtype=jnp.bool_),
        current_prototype_decision_id=current,
        next_prototype_decision_id=_increment_decision_id(current),
        request=_request() if request is None else request,
        record=_record() if record is None else record,
    )


def _transition(
    state: PrototypeConsolidatedSemanticMemoryState,
    *,
    next_raw: float = 0.25,
    reward: float = 0.0,
) -> PrototypeConsolidatedSemanticTransition:
    prototype = state.composition.prototype
    return PrototypeConsolidatedSemanticTransition(
        observation=prototype.current_raw_observation[:1],
        action=prototype.current_action,
        decision_id=prototype.current_decision_id,
        reward=jnp.asarray(reward, dtype=jnp.float32),
        discount=jnp.asarray(1.0, dtype=jnp.float32),
        terminated=jnp.asarray(False, dtype=jnp.bool_),
        truncated=jnp.asarray(False, dtype=jnp.bool_),
        next_observation=jnp.asarray((next_raw,), dtype=jnp.float32),
        next_decision_observation=jnp.asarray((next_raw,), dtype=jnp.float32),
    )


def _procedural_transition(
    state: PrototypeConsolidatedSemanticMemoryState,
    transition: PrototypeConsolidatedSemanticTransition,
) -> PrototypeTransition:
    zero = jnp.zeros((1,), dtype=jnp.float32)
    return PrototypeTransition(
        observation=state.composition.prototype.current_raw_observation,
        action=transition.action,
        decision_id=transition.decision_id,
        reward=transition.reward,
        discount=transition.discount,
        terminated=transition.terminated,
        truncated=transition.truncated,
        next_observation=jnp.concatenate((transition.next_observation, zero)),
        next_decision_observation=jnp.concatenate(
            (transition.next_decision_observation, zero)
        ),
    )


def test_config_state_and_resource_contract_wrap_the_exact_shared_composition() -> None:
    agent = _agent()
    wrapper_state = _initial(agent)
    direct_state = agent.composition.init(
        jr.key(7),
        source_digest=_digest("source"),
        semantic_namespace_digest=_digest("namespace"),
        representation_revision=0,
        source_revision=0,
        lifecycle_id=jnp.asarray((17, 19), dtype=jnp.uint32),
    )
    assert _tree_equal(wrapper_state.composition, direct_state)
    assert len(jax.tree_util.tree_leaves(wrapper_state)) == len(
        jax.tree_util.tree_leaves(direct_state)
    )
    payload = agent.to_config()
    restored = PrototypeConsolidatedSemanticMemoryAgent.from_config(payload)
    assert restored.config == agent.config
    assert restored.to_config() == payload

    altered = dict(payload)
    altered["context_influence_enabled"] = 1
    with pytest.raises(ValueError, match="fixed fields differ"):
        PrototypeConsolidatedSemanticMemoryAgent.from_config(altered)
    with pytest.raises(ValueError, match="observation_dim must equal"):
        PrototypeConsolidatedSemanticMemoryConfig(
            composition=agent.config.composition,
            raw_observation_dim=2,
        )
    transformed_prototypes = (
        dataclasses.replace(
            agent.config.composition.prototype,
            state_builder=IdentityStateBuilderConfig(observation_dim=2),
        ),
        dataclasses.replace(
            agent.config.composition.prototype,
            gru_perception=GRUPerceptionConfig(
                observation_dim=1,
                hidden_dim=1,
            ),
        ),
    )
    for prototype in transformed_prototypes:
        with pytest.raises(ValueError, match="direct context requires"):
            PrototypeConsolidatedSemanticMemoryConfig(
                composition=dataclasses.replace(
                    agent.config.composition,
                    prototype=prototype,
                ),
                raw_observation_dim=1,
            )

    budget = agent.resource_budget
    assert budget.incremental_persistent_state_bytes == 0
    assert budget.incremental_persistent_logical_scalars == 0
    assert budget.shared_controller_memory_states == 1
    assert budget.semantic_memory_operations_per_valid_transition == 1
    assert budget.additional_random_generator_calls_per_transition == 0
    assert budget.context_vectors_built_per_transition == 7
    assert budget.context_cells_per_vector == 2
    assert budget.dispatch_settlement_delegations_per_external_action == 1
    assert budget.additional_dispatch_settlement_state_bytes == 0
    assert budget.direct_dispatches_per_transition == 0
    assert budget.safety_overrides_per_transition == 0
    assert budget.persistent_growth_per_transition_bytes == 0


def test_miss_queries_prewrite_writes_once_and_only_next_context_gets_zero() -> None:
    agent = _agent()
    state = _started(agent)
    producing_action = int(state.composition.prototype.current_action)
    result = agent.update_transition(
        state,
        _transition(state),
        semantic_input=_semantic_input(state),
    )
    assert bool(result.diagnostics.outer_transaction_committed)
    assert bool(result.diagnostics.semantic_query_before_write_verified)
    assert not bool(result.semantic_candidate.retrieval.accepted)
    assert bool(result.semantic_candidate.write.wrote)
    assert bool(result.diagnostics.semantic_zero_tail_used)
    np.testing.assert_array_equal(np.asarray(result.semantic_payload), np.zeros((1,)))
    np.testing.assert_array_equal(
        np.asarray(result.state.composition.prototype.current_raw_observation),
        np.asarray((0.25, 0.0), dtype=np.float32),
    )
    memory = result.state.composition.controller.memory
    assert int(memory.operation_count) == 1
    assert int(memory.semantic_query_count) == 1
    assert int(memory.semantic_write_count) == 1
    assert int(memory.procedural_query_count) == 0
    assert int(memory.procedural_write_count) == 0
    assert bool(result.diagnostics.current_action_unchanged_before_learning)
    assert producing_action == int(state.composition.prototype.current_action)


def test_prior_payload_is_consumed_and_current_record_cannot_retroact() -> None:
    agent = _agent()
    first_state = _started(agent)
    first = agent.update_transition(
        first_state,
        _transition(first_state, next_raw=0.1),
        semantic_input=_semantic_input(first_state, record=_record(2.0)),
    )
    second_state = first.state
    producing_action = int(second_state.composition.prototype.current_action)
    # The current record deliberately disagrees with the prior record. The
    # returned context must still be the pre-write value 2.0.
    second = agent.update_transition(
        second_state,
        _transition(second_state, next_raw=0.2),
        semantic_input=_semantic_input(second_state, record=_record(-8.0)),
    )
    assert bool(second.semantic_candidate.retrieval.accepted)
    assert bool(second.semantic_candidate.write.wrote)
    np.testing.assert_array_equal(
        np.asarray(second.semantic_payload), np.asarray((2.0,), dtype=np.float32)
    )
    np.testing.assert_array_equal(
        np.asarray(second.state.composition.prototype.current_raw_observation),
        np.asarray((0.2, 2.0), dtype=np.float32),
    )
    assert bool(
        second.diagnostics.semantic_context_consumed_by_next_prototype_decision
    )
    assert bool(second.diagnostics.current_action_unchanged_before_learning)
    assert producing_action == int(second_state.composition.prototype.current_action)


def test_semantic_lifecycle_binding_has_an_exact_static_contract() -> None:
    agent = _agent()
    state = _started(agent)
    transition = _transition(state)
    semantic = _semantic_input(state)
    with pytest.raises(TypeError, match="available must have dtype bool"):
        agent.update_transition(
            state,
            transition,
            semantic_input=semantic.replace(
                available=jnp.asarray(1, dtype=jnp.int32)
            ),
        )
    with pytest.raises(ValueError, match="current_prototype_decision_id must have shape"):
        agent.update_transition(
            state,
            transition,
            semantic_input=semantic.replace(
                current_prototype_decision_id=jnp.zeros(
                    (3,), dtype=jnp.uint32
                )
            ),
        )


@pytest.mark.parametrize(
    "case",
    [
        "missing",
        "stale-current-id",
        "stale-next-id",
        "binding",
        "representation-revision",
        "source-revision",
        "invalid",
    ],
)
def test_missing_or_rejected_semantic_input_is_exact_zero_and_control_continues(
    case: str,
) -> None:
    agent = _agent()
    state = _started(agent)
    semantic: PrototypeConsolidatedSemanticMemoryInput | None
    semantic = _semantic_input(state)
    if case == "missing":
        semantic = None
    elif case == "stale-current-id":
        semantic = semantic.replace(
            current_prototype_decision_id=_increment_decision_id(
                semantic.current_prototype_decision_id
            )
        )
    elif case == "stale-next-id":
        semantic = semantic.replace(
            next_prototype_decision_id=_increment_decision_id(
                semantic.next_prototype_decision_id
            )
        )
    elif case == "binding":
        semantic = semantic.replace(record=_record(identity="semantic-b"))
    elif case == "representation-revision":
        semantic = semantic.replace(
            request=_request(representation_revision=1),
            record=_record(representation_revision=1),
        )
    elif case == "source-revision":
        semantic = semantic.replace(
            request=_request(source_revision=1),
            record=_record(source_revision=1),
        )
    else:
        semantic = semantic.replace(
            request=_request(kind=99),
            record=_record(kind=99),
        )
    result = agent.update_transition(
        state,
        _transition(state),
        semantic_input=semantic,
    )
    assert bool(result.diagnostics.outer_transaction_committed)
    assert int(result.action) >= 0
    assert int(result.state.composition.prototype.step_count) == 1
    np.testing.assert_array_equal(np.asarray(result.semantic_payload), np.zeros((1,)))
    np.testing.assert_array_equal(
        np.asarray(result.state.composition.prototype.current_raw_observation[1:]),
        np.zeros((1,)),
    )
    assert int(result.state.composition.controller.memory.operation_count) == 0
    if case == "binding":
        assert not bool(
            result.diagnostics.semantic_request_record_binding_matches
        )
    if case == "stale-next-id":
        assert not bool(result.diagnostics.semantic_next_decision_matches)


def test_stale_compatible_retrieval_uses_zero_tail_but_keeps_learning() -> None:
    agent = _agent(semantic_max_age=0)
    state = _started(agent)
    first = agent.update_transition(
        state,
        _transition(state, next_raw=0.1),
        semantic_input=_semantic_input(state, record=_record(3.0)),
    )
    second = agent.update_transition(
        first.state,
        _transition(first.state, next_raw=0.2),
        semantic_input=_semantic_input(
            first.state,
            request=_request(generation=1),
            record=_record(4.0, generation=1),
        ),
    )
    assert bool(second.semantic_candidate.retrieval.identity_found)
    assert not bool(second.semantic_candidate.retrieval.compatible)
    assert not bool(second.semantic_candidate.retrieval.fresh)
    assert not bool(second.semantic_candidate.retrieval.accepted)
    assert bool(second.semantic_candidate.write.revised)
    assert bool(second.diagnostics.semantic_zero_tail_used)
    np.testing.assert_array_equal(np.asarray(second.semantic_payload), np.zeros((1,)))
    assert int(second.action) >= 0
    assert int(second.state.composition.prototype.step_count) == 2


def test_procedural_feedback_serializes_shared_semantic_operation_exactly() -> None:
    agent = _agent()
    pending = _started(agent, decision=True)
    assert bool(pending.composition.controller.pending)
    assert int(pending.composition.controller.memory.operation_count) == 1

    raw_transition = _transition(pending)
    semantic = _semantic_input(pending)
    blocked = agent.update_transition(
        pending,
        raw_transition,
        semantic_input=semantic,
    )
    assert bool(blocked.diagnostics.outer_transaction_committed)
    assert bool(blocked.diagnostics.semantic_serialized_behind_procedural_feedback)
    assert bool(blocked.state.composition.controller.pending)
    assert int(blocked.state.composition.controller.memory.operation_count) == 1
    np.testing.assert_array_equal(np.asarray(blocked.semantic_payload), np.zeros((1,)))
    assert int(blocked.state.composition.prototype.step_count) == 1

    full_transition = _procedural_transition(pending, raw_transition)
    feedback = _feedback(pending.composition, full_transition, 1)
    settled = agent.update_transition(
        pending,
        raw_transition,
        feedback_input=feedback,
        semantic_input=semantic,
    )
    assert bool(settled.diagnostics.procedural_feedback_candidate_applied)
    assert bool(settled.diagnostics.procedural_feedback_cleared_before_semantic)
    assert bool(settled.composition.diagnostics.feedback_input_supplied)
    assert bool(settled.composition.diagnostics.feedback_input_available)
    assert bool(
        settled.composition.diagnostics.feedback_settled_before_prototype_learning
    )
    assert bool(settled.semantic_candidate.write.wrote)
    assert not bool(settled.state.composition.controller.pending)
    memory = settled.state.composition.controller.memory
    assert int(memory.operation_count) == 3
    assert int(memory.procedural_query_count) == 1
    assert int(memory.procedural_write_count) == 1
    assert int(memory.semantic_query_count) == 1
    assert int(memory.semantic_write_count) == 1
    assert bool(settled.diagnostics.procedural_order_preserved)


def test_terminal_shared_capacity_never_freezes_prototype_base_control() -> None:
    agent = _agent(max_operations=1)
    state = _started(agent)
    terminal = agent.update_transition(
        state,
        _transition(state, next_raw=0.1),
        semantic_input=_semantic_input(state),
    )
    assert bool(terminal.semantic_candidate.write.wrote)
    assert bool(terminal.state.composition.controller.memory_unavailable)
    assert int(terminal.state.composition.prototype.step_count) == 1
    assert int(terminal.action) >= 0

    continued = agent.update_transition(
        terminal.state,
        _transition(terminal.state, next_raw=0.2),
        semantic_input=_semantic_input(terminal.state),
    )
    assert int(continued.state.composition.controller.memory.operation_count) == 1
    assert int(continued.state.composition.prototype.step_count) == 2
    assert int(continued.action) >= 0
    assert (
        int(continued.semantic_candidate.diagnostics.reason)
        == SEMANTIC_REASON_MEMORY_UNAVAILABLE
    )
    assert bool(agent.validate_state(continued.state))
    np.testing.assert_array_equal(np.asarray(continued.semantic_payload), np.zeros((1,)))


def test_corruption_checkpoint_binding_and_rebind_are_fail_closed() -> None:
    agent = _agent()
    state = _started(agent)
    payload = agent.checkpoint_payload(state)
    assert PROTOTYPE_CONSOLIDATED_SEMANTIC_MEMORY_CHECKPOINT_HOST_ONLY
    restored = agent.restore_checkpoint(
        payload,
        source_digest=_digest("source"),
        semantic_namespace_digest=_digest("namespace"),
        representation_revision=0,
        source_revision=0,
    )
    assert _tree_equal(restored, state)
    assert _tree_equal(
        restored.composition.dispatch_owner,
        state.composition.dispatch_owner,
    )
    tampered = dict(payload)
    tampered_composition = dict(tampered["composition"])  # type: ignore[arg-type]
    tampered_composition["dispatch_owner_state"] = (
        state.composition.dispatch_owner.replace(
            checksum=state.composition.dispatch_owner.checksum.at[0].add(
                jnp.asarray(1, dtype=jnp.uint32)
            )
        )
    )
    tampered["composition"] = tampered_composition
    with pytest.raises(ValueError, match="dispatch owner state SHA differs"):
        agent.restore_checkpoint(
            tampered,
            source_digest=_digest("source"),
            semantic_namespace_digest=_digest("namespace"),
            representation_revision=0,
            source_revision=0,
        )
    with pytest.raises(ValueError, match="binding"):
        agent.restore_checkpoint(
            payload,
            source_digest=_digest("wrong-source"),
            semantic_namespace_digest=_digest("namespace"),
            representation_revision=0,
            source_revision=0,
        )

    corrupt = state.replace(
        composition=state.composition.replace(
            controller=state.composition.controller.replace(
                checksum=state.composition.controller.checksum.at[0].add(
                    jnp.asarray(1, dtype=jnp.uint32)
                )
            )
        )
    )
    failed = agent.update_transition(corrupt, _transition(corrupt))
    assert int(failed.action) == -1
    assert not bool(failed.diagnostics.outer_transaction_committed)
    assert _tree_equal(failed.state, corrupt)
    with pytest.raises(ValueError, match="invalid semantic composition"):
        agent.checkpoint_payload(corrupt)

    reset = agent.rebind_reset(
        state,
        source_digest=_digest("new-source"),
        semantic_namespace_digest=_digest("new-namespace"),
        representation_revision=1,
        source_revision=1,
    )
    assert _tree_equal(reset.composition.prototype, state.composition.prototype)
    assert int(reset.composition.controller.memory.operation_count) == 0
    np.testing.assert_array_equal(
        np.asarray(reset.composition.controller.memory.source_digest),
        np.asarray(_digest("new-source")),
    )


def test_public_exports_status_and_authority_are_explicit() -> None:
    names = (
        "ConsolidatedSemanticMemoryControllerDiagnostics",
        "ConsolidatedSemanticMemoryControllerResult",
        "PrototypeConsolidatedMemoryFeedbackAttempt",
        "PrototypeConsolidatedSemanticMemoryAgent",
        "PrototypeConsolidatedSemanticMemoryConfig",
        "PrototypeConsolidatedSemanticMemoryDiagnostics",
        "PrototypeConsolidatedSemanticMemoryDispatchSettlementResult",
        "PrototypeConsolidatedSemanticMemoryInput",
        "PrototypeConsolidatedSemanticMemoryResourceBudget",
        "PrototypeConsolidatedSemanticMemoryStartResult",
        "PrototypeConsolidatedSemanticMemoryState",
        "PrototypeConsolidatedSemanticMemoryUpdateResult",
        "PrototypeConsolidatedSemanticTransition",
    )
    for name in names:
        assert getattr(alberta, name) is getattr(core, name)
        assert name in alberta.__all__
        assert name in core.__all__
    assert _agent().composition_order == (
        PROTOTYPE_CONSOLIDATED_SEMANTIC_MEMORY_COMPOSITION_ORDER
    )
    assert PROTOTYPE_CONSOLIDATED_SEMANTIC_MEMORY_CONTEXT_INFLUENCE_ENABLED
    assert not PROTOTYPE_CONSOLIDATED_SEMANTIC_MEMORY_DIRECT_DISPATCH_AUTHORITY
    assert not PROTOTYPE_CONSOLIDATED_SEMANTIC_MEMORY_SAFETY_OVERRIDE_AUTHORITY
    assert not PROTOTYPE_CONSOLIDATED_SEMANTIC_MEMORY_EFFICACY_CLAIM
    assert not PROTOTYPE_CONSOLIDATED_SEMANTIC_MEMORY_EVIDENCE_AUTHORITY
    assert not PROTOTYPE_CONSOLIDATED_SEMANTIC_MEMORY_PROMOTION_AUTHORITY


def test_semantic_wrapper_delegates_typed_dispatch_settlement_without_shadow_state() -> None:
    agent = _agent()
    state = _started(agent, decision=True)
    selected = int(state.composition.prototype.current_action)
    fallback = 1 - selected
    result = agent.settle_dispatch(
        state,
        _settlement(state.composition, executed_action=fallback),
    )
    assert bool(result.composition.diagnostics.transaction_committed)
    assert bool(result.composition.diagnostics.state_changed)
    assert int(result.action) == fallback
    assert _tree_equal(result.state.composition, result.composition.state)
    assert int(result.state.composition.dispatch_owner.selected_action) == fallback
    assert agent.resource_budget.additional_dispatch_settlement_state_bytes == 0


def test_full_semantic_memory_partner_wrapper_delegates_owner_cancellation() -> None:
    from tests.test_prototype_partner_policy_fusion import _sidecar

    agent = _agent(experiential=True, partner=True)
    state = _started(agent)
    base_action = int(state.composition.prototype.current_action)
    partner_action = 1 - base_action
    partner_dispatch = agent.update_transition(
        state,
        _transition(state),
        partner_policy_fusion_input=_sidecar(
            agent.composition.prototype,
            state.composition.prototype,
            suggested_action=partner_action,
        ),
    ).state
    settlement = agent.settle_dispatch(
        partner_dispatch,
        _settlement(
            partner_dispatch.composition,
            executed_action=base_action,
        ),
    )
    audit = settlement.composition.diagnostics
    assert bool(audit.transaction_committed)
    assert bool(audit.partner_cancellation_applied)
    memory_interaction = settlement.state.composition.prototype.ia_state
    interaction = memory_interaction.interaction_state
    assert not bool(interaction.feedback_prototype_decision_id_available)
    assert not bool(interaction.partner_policy_fusion_state.feedback_armed)
