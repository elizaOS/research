# mypy: disable-error-code="arg-type,attr-defined,call-arg,no-any-return,type-var"
"""Unit contracts for the opt-in live Prototype memory composition."""

from __future__ import annotations

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import alberta_framework as alberta
import alberta_framework.core as core
from alberta_framework.core.consolidated_memory import (
    ConsolidatedMemoryConfig,
    ProceduralMemoryRequest,
    canonical_memory_digest,
)
from alberta_framework.core.consolidated_memory_controller import (
    ConsolidatedProceduralMemoryControllerConfig,
)
from alberta_framework.core.consolidated_memory_policy import (
    ConsolidatedProceduralMemoryPolicyConfig,
)
from alberta_framework.core.experiential_memory import ExperientialMemoryConfig
from alberta_framework.core.oak import OaKConfig
from alberta_framework.core.options import STOMPConfig, SubtaskSpec
from alberta_framework.core.partner_policy_fusion import PartnerPolicyFusionConfig
from alberta_framework.core.prototype_agent import (
    PrototypeAgent,
    PrototypeAgentConfig,
    PrototypeExperientialMemoryInput,
    PrototypePartnerPolicyFusionInput,
    PrototypeTransition,
)
from alberta_framework.core.prototype_consolidated_memory import (
    PROTOTYPE_CONSOLIDATED_MEMORY_AUTONOMOUS_POLICY_AUTHORITY,
    PROTOTYPE_CONSOLIDATED_MEMORY_CACHED_ACTION_REPLACEMENT_ENABLED,
    PROTOTYPE_CONSOLIDATED_MEMORY_COMPOSITION_ORDER,
    PROTOTYPE_CONSOLIDATED_MEMORY_DISPATCH_SETTLEMENT_ENABLED,
    PROTOTYPE_CONSOLIDATED_MEMORY_PHYSICAL_DISPATCH_AUTHORITY,
    PrototypeConsolidatedMemoryAgent,
    PrototypeConsolidatedMemoryConfig,
    PrototypeConsolidatedMemoryDecisionInput,
    PrototypeConsolidatedMemoryDispatchSettlementInput,
    PrototypeConsolidatedMemoryFeedbackInput,
    PrototypeConsolidatedMemoryState,
)

pytestmark = pytest.mark.unit


def _digest(text: str) -> jax.Array:
    return canonical_memory_digest("test.prototype-consolidated-memory", text)


def _oak(n_actions: int = 2) -> OaKConfig:
    return OaKConfig(
        stomp=STOMPConfig(
            subtask_specs=(
                SubtaskSpec(
                    feature_index=0,
                    threshold=1.0e6,
                    max_option_steps=8,
                ),
            ),
            observation_dim=2,
            n_primitive_actions=n_actions,
            base_hidden_sizes=(),
            epsilon_base=0.0,
            epsilon_option=0.0,
        )
    )


def _controller_config(
    n_actions: int = 2,
    *,
    max_operations: int = 100,
) -> ConsolidatedProceduralMemoryControllerConfig:
    return ConsolidatedProceduralMemoryControllerConfig(
        memory=ConsolidatedMemoryConfig(
            semantic_capacity=1,
            procedural_capacity=2,
            semantic_payload_dim=1,
            procedural_payload_dim=n_actions,
            procedural_outcome_dim=1,
            semantic_max_age=min(20, max_operations),
            procedural_max_age=min(20, max_operations),
            max_operations=max_operations,
            semantic_min_confidence=0.0,
            procedural_min_confidence=0.0,
        ),
        policy=ConsolidatedProceduralMemoryPolicyConfig(
            n_actions=n_actions,
            outcome_dim=1,
            min_evidence_count=2,
            min_success_lower_bound=0.0,
            wilson_z=1.0,
            max_outcome_standard_error=10.0,
            max_abs_outcome_mean=100.0,
        ),
    )


def _agent(
    *,
    n_actions: int = 2,
    max_operations: int = 100,
    experiential: bool = False,
    partner: bool = False,
) -> PrototypeConsolidatedMemoryAgent:
    prototype = PrototypeAgentConfig(
        oak=_oak(n_actions),
        experiential_memory=(
            ExperientialMemoryConfig(
                capacity=2,
                observation_dim=2,
                key_dim=2,
                action_dim=n_actions,
                outcome_dim=3,
                top_k=1,
                min_neighbors=1,
                distance_scale=1.0,
                min_similarity=0.0,
                min_effective_reliability=0.01,
                max_uncertainty=1.0,
                max_safety_cost=1.0,
                max_age=100,
                staleness_scale=100.0,
                utility_decay=1.0,
            )
            if experiential
            else None
        ),
        partner_policy_fusion=(
            PartnerPolicyFusionConfig(
                max_partners=1,
                context_dim=2,
                n_actions=n_actions,
                max_abs_context=10.0,
                assistance_value_bound=10.0,
            )
            if partner
            else None
        ),
    )
    return PrototypeConsolidatedMemoryAgent(
        PrototypeConsolidatedMemoryConfig(
            prototype=prototype,
            controller=_controller_config(
                n_actions,
                max_operations=max_operations,
            ),
        )
    )


def _initial(agent: PrototypeConsolidatedMemoryAgent) -> PrototypeConsolidatedMemoryState:
    return agent.init(
        jr.key(7),
        source_digest=_digest("source"),
        semantic_namespace_digest=_digest("namespace"),
        representation_revision=0,
        source_revision=0,
        lifecycle_id=jnp.asarray((17, 19), dtype=jnp.uint32),
    )


def _request() -> ProceduralMemoryRequest:
    return ProceduralMemoryRequest(
        semantic_digest=_digest("skill"),
        generation=jnp.asarray(0, dtype=jnp.int32),
        provenance_digest=_digest("provenance"),
        representation_revision=jnp.asarray(0, dtype=jnp.int32),
        source_revision=jnp.asarray(0, dtype=jnp.int32),
        lifecycle_link_available=jnp.asarray(True, dtype=jnp.bool_),
        lifecycle_digest=_digest("option-lifecycle"),
        lifecycle_generation=jnp.asarray(3, dtype=jnp.int32),
        lifecycle_revision=jnp.asarray(5, dtype=jnp.int32),
    )


def _decision_input(
    state: PrototypeConsolidatedMemoryState,
    *,
    mask: tuple[bool, ...] | None = None,
) -> PrototypeConsolidatedMemoryDecisionInput:
    n_actions = state.controller.pending_hard_safety_mask.shape[0]
    return PrototypeConsolidatedMemoryDecisionInput(
        available=jnp.asarray(True, dtype=jnp.bool_),
        prototype_decision_id=state.prototype.current_decision_id,
        request=_request(),
        hard_safety_action_mask=jnp.asarray(
            (True,) * n_actions if mask is None else mask,
            dtype=jnp.bool_,
        ),
    )


def _transition(
    state: PrototypeConsolidatedMemoryState,
    *,
    reward: float = 1.0,
) -> PrototypeTransition:
    next_observation = jnp.asarray((0.1, -0.2), dtype=jnp.float32)
    return PrototypeTransition(
        observation=state.prototype.current_raw_observation,
        action=state.prototype.current_action,
        decision_id=state.prototype.current_decision_id,
        reward=jnp.asarray(reward, dtype=jnp.float32),
        discount=jnp.asarray(1.0, dtype=jnp.float32),
        terminated=jnp.asarray(False, dtype=jnp.bool_),
        truncated=jnp.asarray(False, dtype=jnp.bool_),
        next_observation=next_observation,
        next_decision_observation=next_observation,
    )


def _feedback(
    state: PrototypeConsolidatedMemoryState,
    transition: PrototypeTransition,
    event: int,
) -> PrototypeConsolidatedMemoryFeedbackInput:
    return PrototypeConsolidatedMemoryFeedbackInput(
        available=jnp.asarray(True, dtype=jnp.bool_),
        prototype_decision_id=transition.decision_id,
        feedback_event_id=jnp.asarray((0, 0, 0, event), dtype=jnp.uint32),
        base_action=state.controller.pending_base_action,
        effective_action=transition.action,
        request=_request(),
        succeeded=jnp.asarray(True, dtype=jnp.bool_),
        outcome=jnp.asarray((1.0,), dtype=jnp.float32),
        confidence=jnp.asarray(1.0, dtype=jnp.float32),
        evidence=jnp.asarray(1.0, dtype=jnp.float32),
    )


def _tree_equal(left: object, right: object) -> bool:
    def equal_leaf(a: Any, b: Any) -> Any:
        if jnp.issubdtype(a.dtype, jax.dtypes.prng_key):
            return jnp.array_equal(jr.key_data(a), jr.key_data(b))
        return jnp.array_equal(a, b)

    return bool(jax.tree_util.tree_all(jax.tree.map(equal_leaf, left, right)))


def _force_action(
    agent: PrototypeConsolidatedMemoryAgent,
    state: PrototypeConsolidatedMemoryState,
    action: int,
) -> PrototypeConsolidatedMemoryState:
    bound_mask = state.dispatch_owner.hard_safety_action_mask
    replacement = agent.prototype.replace_cached_primitive_action(
        state.prototype,
        decision_id=state.prototype.current_decision_id,
        decision_observation=state.prototype.current_representation,
        proposed_action=jnp.asarray(action, dtype=jnp.int32),
        safety_action_mask=bound_mask,
    )
    assert bool(replacement.committed)
    legacy_shape = state.replace(prototype=replacement.state)
    if action != int(state.prototype.current_action):
        assert not bool(agent.validate_state(legacy_shape))
    return legacy_shape.replace(
        dispatch_owner=agent._dispatch_owner_record(
            available=state.dispatch_owner.available,
            prototype_decision_id=state.prototype.current_decision_id,
            selected_action=jnp.asarray(action, dtype=jnp.int32),
            hard_safety_action_mask=bound_mask,
        )
    )


def test_public_prototype_replacement_rejects_stale_id_and_updates_real_owner() -> None:
    prototype = PrototypeAgent(PrototypeAgentConfig(oak=_oak()))
    state = prototype.start(
        prototype.init(
            jr.key(1), lifecycle_id=jnp.asarray((1, 2), dtype=jnp.uint32)
        ),
        jnp.zeros((2,), dtype=jnp.float32),
    )
    stale_id = state.current_decision_id.at[3].add(jnp.asarray(1, dtype=jnp.uint32))
    stale = prototype.replace_cached_primitive_action(
        state,
        decision_id=stale_id,
        decision_observation=state.current_representation,
        proposed_action=jnp.asarray(1, dtype=jnp.int32),
        safety_action_mask=jnp.ones((2,), dtype=jnp.bool_),
    )
    assert not bool(stale.committed)
    assert _tree_equal(stale.state, state)

    exact = prototype.replace_cached_primitive_action(
        state,
        decision_id=state.current_decision_id,
        decision_observation=state.current_representation,
        proposed_action=jnp.asarray(1, dtype=jnp.int32),
        safety_action_mask=jnp.ones((2,), dtype=jnp.bool_),
    )
    assert bool(exact.committed)
    assert int(exact.state.current_action) == 1
    assert int(exact.state.oak_state.stomp_state.last_primitive_action) == 1
    assert int(exact.state.oak_state.stomp_state.base_last_action) == 1
    np.testing.assert_array_equal(
        np.asarray(jr.key_data(exact.state.oak_state.stomp_state.rng_key)),
        np.asarray(jr.key_data(state.oak_state.stomp_state.rng_key)),
    )

    active_stomp = exact.state.oak_state.stomp_state.replace(
        executing_option=jnp.asarray(0, dtype=jnp.int32),
        base_last_action=jnp.asarray(2, dtype=jnp.int32),
        option_last_intra_action=jnp.asarray(1, dtype=jnp.int32),
    )
    active_state = exact.state.replace(
        oak_state=exact.state.oak_state.replace(stomp_state=active_stomp)
    )
    assert bool(prototype.validate_state(active_state))
    active = prototype.replace_cached_primitive_action(
        active_state,
        decision_id=active_state.current_decision_id,
        decision_observation=active_state.current_representation,
        proposed_action=jnp.asarray(0, dtype=jnp.int32),
        safety_action_mask=jnp.ones((2,), dtype=jnp.bool_),
    )
    assert bool(active.committed)
    assert int(active.state.current_action) == 0
    assert int(active.state.oak_state.stomp_state.base_last_action) == 2
    assert int(active.state.oak_state.stomp_state.option_last_intra_action) == 0


def test_missing_feedback_and_terminal_memory_cap_do_not_freeze_prototype() -> None:
    with jax.disable_jit():
        agent = _agent(max_operations=2)
        initial = _initial(agent)
        started = agent.start(
            initial,
            jnp.zeros((2,), dtype=jnp.float32),
            decision_input=_decision_input(initial),
        ).state
        assert bool(started.controller.pending)

        first = agent.update_transition(started, _transition(started))
        assert int(first.state.prototype.step_count) == 1
        assert bool(first.state.controller.pending)
        assert int(first.action) >= 0

        # Restore the exact prior outcome and consume the final memory
        # operation. The terminal sidecar becomes unavailable, while Prototype
        # learning and safe base dispatch continue.
        transition = _transition(started)
        settled = agent.update_transition(
            started,
            transition,
            feedback_input=_feedback(started, transition, 1),
        )
        assert bool(settled.memory_feedback.diagnostics.write_applied)
        assert bool(settled.state.controller.memory_unavailable)
        assert int(settled.state.prototype.step_count) == 1
        assert int(settled.action) >= 0


def test_persistent_controller_corruption_fails_closed_and_checkpoint_recovers() -> None:
    with jax.disable_jit():
        agent = _agent()
        initial = _initial(agent)
        started = agent.start(
            initial,
            jnp.zeros((2,), dtype=jnp.float32),
            decision_input=_decision_input(initial),
        ).state
        payload = agent.checkpoint_payload(started)
        restored = agent.restore_checkpoint(
            payload,
            source_digest=_digest("source"),
            semantic_namespace_digest=_digest("namespace"),
            representation_revision=0,
            source_revision=0,
        )
        assert _tree_equal(restored, started)
        corrupt = started.replace(
            controller=started.controller.replace(
                checksum=started.controller.checksum.at[3].add(
                    jnp.asarray(1, dtype=jnp.uint32)
                )
            )
        )
        result = agent.update_transition(corrupt, _transition(corrupt))
        assert int(result.action) == -1
        assert not bool(result.diagnostics.transaction_committed)
        assert _tree_equal(result.state, corrupt)
        with pytest.raises(ValueError, match="corrupted composed state"):
            agent.rebind_reset(
                corrupt,
                source_digest=_digest("new-source"),
                semantic_namespace_digest=_digest("new-namespace"),
                representation_revision=1,
                source_revision=1,
            )


def test_upstream_order_and_final_hard_mask_are_exact_intersection() -> None:
    with jax.disable_jit():
        agent = _agent(n_actions=4, experiential=True, partner=True)
        initial = _initial(agent)
        started = agent.start(
            initial,
            jnp.zeros((2,), dtype=jnp.float32),
        ).state
        transition = _transition(started, reward=0.0)
        upstream_base = agent.prototype.update_transition(
            started.prototype,
            transition,
        ).action
        base_index = int(upstream_base)
        other_indices = [index for index in range(4) if index != base_index]
        experiential_mask = jnp.ones((4,), dtype=jnp.bool_).at[
            other_indices[0]
        ].set(False)
        partner_mask = jnp.ones((4,), dtype=jnp.bool_).at[
            other_indices[1]
        ].set(False)
        caller_mask = jnp.ones((4,), dtype=jnp.bool_).at[
            other_indices[2]
        ].set(False)
        next_id = started.prototype.current_decision_id.at[3].add(
            jnp.asarray(1, dtype=jnp.uint32)
        )
        experiential = PrototypeExperientialMemoryInput(
            available=jnp.asarray(True, dtype=jnp.bool_),
            current_prototype_decision_id=transition.decision_id,
            next_prototype_decision_id=next_id,
            query_representation_version=jnp.asarray(0, dtype=jnp.int32),
            entry_representation_version=jnp.asarray(0, dtype=jnp.int32),
            query_uncertainty=jnp.asarray(0.0, dtype=jnp.float32),
            query_uncertainty_available=jnp.asarray(True, dtype=jnp.bool_),
            entry_uncertainty=jnp.asarray(0.0, dtype=jnp.float32),
            entry_uncertainty_available=jnp.asarray(True, dtype=jnp.bool_),
            safety_cost=jnp.asarray(0.0, dtype=jnp.float32),
            safety_cost_available=jnp.asarray(True, dtype=jnp.bool_),
            reliability=jnp.asarray(1.0, dtype=jnp.float32),
            utility=jnp.asarray(1.0, dtype=jnp.float32),
            utility_available=jnp.asarray(True, dtype=jnp.bool_),
            provenance_id=jnp.asarray(11, dtype=jnp.int32),
            source_id=jnp.asarray(13, dtype=jnp.int32),
            next_action_safety_mask=experiential_mask,
        )
        fusion = agent.prototype.partner_policy_fusion
        assert fusion is not None
        partner = PrototypePartnerPolicyFusionInput(
            available=jnp.asarray(True, dtype=jnp.bool_),
            prototype_decision_id=next_id,
            observation_id=jnp.asarray(17, dtype=jnp.int32),
            context_id=jnp.asarray(19, dtype=jnp.int32),
            context_features=jnp.zeros((2,), dtype=jnp.float32),
            safety_action_mask=partner_mask,
            keyboard_available=jnp.asarray(False, dtype=jnp.bool_),
            keyboard_vector=jnp.zeros((1,), dtype=jnp.float32),
            messages=fusion.empty_messages(),
        )
        decision = PrototypeConsolidatedMemoryDecisionInput(
            available=jnp.asarray(True, dtype=jnp.bool_),
            prototype_decision_id=next_id,
            request=_request(),
            hard_safety_action_mask=caller_mask,
        )
        result = agent.update_transition(
            started,
            transition,
            decision_input=decision,
            experiential_memory_input=experiential,
            partner_policy_fusion_input=partner,
        )
        assert agent.composition_order == PROTOTYPE_CONSOLIDATED_MEMORY_COMPOSITION_ORDER
        assert bool(result.diagnostics.experiential_memory_precedes_consolidated)
        assert bool(result.diagnostics.partner_fusion_precedes_consolidated)
        assert bool(result.diagnostics.experiential_mask_applicable)
        assert bool(result.diagnostics.partner_mask_applicable)
        np.testing.assert_array_equal(
            np.asarray(result.diagnostics.final_hard_safety_action_mask),
            np.asarray(jax.nn.one_hot(base_index, 4, dtype=jnp.bool_)),
        )
        assert bool(result.diagnostics.final_mask_is_exact_intersection)
        assert int(result.action) == base_index


def test_disabled_prototype_state_and_config_are_bit_identical() -> None:
    prototype_config = PrototypeAgentConfig(oak=_oak())
    direct = PrototypeAgent(prototype_config)
    adapter = PrototypeConsolidatedMemoryAgent(
        PrototypeConsolidatedMemoryConfig(
            prototype=prototype_config,
            controller=_controller_config(),
        )
    )
    direct_state = direct.init(
        jr.key(9), lifecycle_id=jnp.asarray((23, 29), dtype=jnp.uint32)
    )
    composed_state = adapter.init(
        jr.key(9),
        lifecycle_id=jnp.asarray((23, 29), dtype=jnp.uint32),
        source_digest=_digest("source"),
        semantic_namespace_digest=_digest("namespace"),
        representation_revision=0,
        source_revision=0,
    )
    assert direct.to_config() == prototype_config.to_config()
    assert adapter.prototype.to_config() == direct.to_config()
    assert _tree_equal(composed_state.prototype, direct_state)
    assert dataclasses.asdict(prototype_config) == dataclasses.asdict(
        adapter.config.prototype
    )
    payload = adapter.to_config()
    restored = PrototypeConsolidatedMemoryAgent.from_config(payload)
    assert restored.config == adapter.config
    assert restored.to_config() == payload
    noncanonical = dict(payload)
    noncanonical["cached_action_replacement_enabled"] = 1
    with pytest.raises(ValueError, match="fixed fields differ"):
        PrototypeConsolidatedMemoryAgent.from_config(noncanonical)


def test_public_exports_and_action_authority_are_explicit() -> None:
    names = (
        "ConsolidatedMemory",
        "ConsolidatedProceduralMemoryPolicy",
        "ConsolidatedProceduralMemoryController",
        "PrototypeCachedPrimitiveActionReplacement",
        "PrototypeConsolidatedMemoryAgent",
        "PrototypeConsolidatedMemoryDecisionInput",
        "PrototypeConsolidatedMemoryDispatchOwnerState",
        "PrototypeConsolidatedMemoryDispatchSettlementInput",
        "PrototypeConsolidatedMemoryDispatchSettlementResult",
        "PrototypeConsolidatedMemoryFeedbackInput",
        "PrototypeConsolidatedMemoryResourceBudget",
        "PrototypeConsolidatedMemoryUpstreamMaskState",
    )
    for name in names:
        assert getattr(alberta, name) is getattr(core, name)
        assert name in alberta.__all__
        assert name in core.__all__
    assert PROTOTYPE_CONSOLIDATED_MEMORY_CACHED_ACTION_REPLACEMENT_ENABLED
    assert PROTOTYPE_CONSOLIDATED_MEMORY_DISPATCH_SETTLEMENT_ENABLED
    assert not PROTOTYPE_CONSOLIDATED_MEMORY_AUTONOMOUS_POLICY_AUTHORITY
    assert not PROTOTYPE_CONSOLIDATED_MEMORY_PHYSICAL_DISPATCH_AUTHORITY
    budget = _agent().resource_budget
    assert budget.memory_feedback_attempts_per_transition == 1
    assert budget.controller_feedback_evaluations_per_transition == 2
    assert budget.memory_queries_per_decision_call == 1
    assert budget.cached_action_replacements_per_decision_call == 1
    assert budget.pending_upstream_mask_records == 1
    assert budget.upstream_mask_input_cells_composed_per_transition == 4
    assert (
        budget.final_mask_input_cells_composed_per_dispatch_composition == 4
    )
    assert budget.realized_action_mask_checks_per_transition == 1
    assert budget.cached_action_mask_checks_per_dispatch_composition == 1
    assert budget.upstream_mask_checksum_words == 4
    assert budget.dispatch_owner_checksum_words == 4
    assert budget.pending_dispatch_settlement_records == 1
    assert budget.dispatch_settlement_identity_checks_per_call == 3
    assert budget.memory_writes_per_dispatch_settlement == 0
    assert budget.learner_parameter_updates_per_dispatch_settlement == 0
    assert budget.partner_parameter_updates_per_dispatch_settlement == 0
    assert budget.random_generator_calls_per_dispatch_settlement == 0
    assert budget.physical_commands_per_dispatch_settlement == 0
    assert budget.dispatch_settlement_enabled
    assert budget.incremental_persistent_state_bytes == (
        budget.controller.persistent_state_bytes + 74
    )
    assert budget.incremental_persistent_logical_scalars == (
        budget.controller.persistent_logical_scalars + 23
    )
    assert budget.additional_random_generator_calls_per_event == 0
    assert budget.physical_commands_per_event == 0
    assert budget.persistent_growth_per_event_bytes == 0
    assert budget.cached_action_replacement_enabled


def _settlement(
    state: PrototypeConsolidatedMemoryState,
    *,
    executed_action: int,
    action_available: bool = True,
    decision_id: jax.Array | None = None,
    selected_action: int | None = None,
) -> PrototypeConsolidatedMemoryDispatchSettlementInput:
    return PrototypeConsolidatedMemoryDispatchSettlementInput(
        action_available=jnp.asarray(action_available, dtype=jnp.bool_),
        prototype_decision_id=(
            state.prototype.current_decision_id
            if decision_id is None
            else decision_id
        ),
        selected_action=jnp.asarray(
            int(state.prototype.current_action)
            if selected_action is None
            else selected_action,
            dtype=jnp.int32,
        ),
        executed_action=jnp.asarray(executed_action, dtype=jnp.int32),
    )


def test_changed_dispatch_settlement_is_atomic_zero_work_jittable_and_checkpointed() -> None:
    agent = _agent()
    initial = _initial(agent)
    with jax.disable_jit():
        started = agent.start(
            initial,
            jnp.zeros((2,), dtype=jnp.float32),
            decision_input=_decision_input(initial),
        ).state
    selected = int(started.prototype.current_action)
    fallback = 1 - selected
    receipt = _settlement(started, executed_action=fallback)

    eager = agent.settle_dispatch(started, receipt)
    compiled = jax.jit(agent.settle_dispatch)(started, receipt)
    assert _tree_equal(eager, compiled)
    assert bool(eager.diagnostics.transaction_committed)
    assert bool(eager.diagnostics.state_changed)
    assert int(eager.action) == fallback
    assert int(eager.state.prototype.current_action) == fallback
    assert int(eager.state.dispatch_owner.selected_action) == fallback
    np.testing.assert_array_equal(
        eager.state.dispatch_owner.hard_safety_action_mask,
        started.dispatch_owner.hard_safety_action_mask,
    )
    assert not bool(eager.state.controller.pending)
    assert bool(eager.diagnostics.procedural_cancellation_applied)
    assert _tree_equal(eager.state.controller.memory, started.controller.memory)
    assert int(eager.state.prototype.step_count) == int(
        started.prototype.step_count
    )
    np.testing.assert_array_equal(
        eager.state.prototype.step_words,
        started.prototype.step_words,
    )
    np.testing.assert_array_equal(
        jr.key_data(eager.state.prototype.oak_state.stomp_state.rng_key),
        jr.key_data(started.prototype.oak_state.stomp_state.rng_key),
    )
    assert not bool(eager.diagnostics.learner_update_applied)
    assert not bool(eager.diagnostics.memory_evidence_written)
    assert not bool(eager.diagnostics.partner_learning_applied)
    assert not bool(eager.diagnostics.random_generator_consumed)

    payload = agent.checkpoint_payload(eager.state)
    restored = agent.restore_checkpoint(
        payload,
        source_digest=_digest("source"),
        semantic_namespace_digest=_digest("namespace"),
        representation_revision=0,
        source_revision=0,
    )
    assert _tree_equal(restored, eager.state)
    assert bool(agent.validate_state(restored))
    assert int(restored.dispatch_owner.selected_action) == fallback

    replay = agent.settle_dispatch(eager.state, receipt)
    assert not bool(replay.diagnostics.transaction_committed)
    assert _tree_equal(replay.state, eager.state)


def test_unchanged_no_action_stale_disallowed_corrupt_and_partial_settlements_are_noops() -> None:
    agent = _agent()
    initial = _initial(agent)
    with jax.disable_jit():
        started = agent.start(
            initial,
            jnp.zeros((2,), dtype=jnp.float32),
            decision_input=_decision_input(initial),
        ).state
    selected = int(started.prototype.current_action)
    fallback = 1 - selected

    unchanged = agent.settle_dispatch(
        started,
        _settlement(started, executed_action=selected),
    )
    assert bool(unchanged.diagnostics.transaction_committed)
    assert not bool(unchanged.diagnostics.state_changed)
    assert _tree_equal(unchanged.state, started)

    no_action = agent.settle_dispatch(
        started,
        _settlement(started, executed_action=-1, action_available=False),
    )
    assert bool(no_action.diagnostics.transaction_committed)
    assert int(no_action.action) == -1
    assert _tree_equal(no_action.state, started)

    stale_id = started.prototype.current_decision_id.at[3].add(
        jnp.asarray(1, dtype=jnp.uint32)
    )
    stale = agent.settle_dispatch(
        started,
        _settlement(started, executed_action=fallback, decision_id=stale_id),
    )
    assert not bool(stale.diagnostics.transaction_committed)
    assert _tree_equal(stale.state, started)

    current_only_mask = tuple(index == selected for index in range(2))
    with jax.disable_jit():
        restricted = agent.decide(
            started,
            decision_input=_decision_input(started, mask=current_only_mask),
        ).state
    assert not bool(restricted.dispatch_owner.hard_safety_action_mask[fallback])
    disallowed = agent.settle_dispatch(
        restricted,
        _settlement(restricted, executed_action=fallback),
    )
    assert not bool(disallowed.diagnostics.transaction_committed)
    assert not bool(disallowed.diagnostics.executed_action_allowed_by_bound_mask)
    assert _tree_equal(disallowed.state, restricted)

    corrupt_owner = started.replace(
        dispatch_owner=started.dispatch_owner.replace(
            checksum=started.dispatch_owner.checksum.at[0].add(
                jnp.asarray(1, dtype=jnp.uint32)
            )
        )
    )
    corrupt = agent.settle_dispatch(
        corrupt_owner,
        _settlement(corrupt_owner, executed_action=fallback),
    )
    assert not bool(corrupt.diagnostics.transaction_committed)
    assert _tree_equal(corrupt.state, corrupt_owner)

    forced_prototype = agent.prototype.replace_cached_primitive_action(
        started.prototype,
        decision_id=started.prototype.current_decision_id,
        decision_observation=started.prototype.current_representation,
        proposed_action=jnp.asarray(fallback, dtype=jnp.int32),
        safety_action_mask=started.dispatch_owner.hard_safety_action_mask,
    )
    assert bool(forced_prototype.committed)
    partial_state = started.replace(
        prototype=forced_prototype.state,
        dispatch_owner=agent._dispatch_owner_record(
            available=jnp.asarray(True, dtype=jnp.bool_),
            prototype_decision_id=started.prototype.current_decision_id,
            selected_action=jnp.asarray(fallback, dtype=jnp.int32),
            hard_safety_action_mask=started.dispatch_owner.hard_safety_action_mask,
        ),
    )
    assert bool(agent.validate_state(partial_state))
    partial = agent.settle_dispatch(
        partial_state,
        _settlement(partial_state, executed_action=selected),
    )
    assert bool(partial.diagnostics.procedural_cancellation_required)
    assert not bool(partial.diagnostics.procedural_cancellation_applied)
    assert not bool(partial.diagnostics.transaction_committed)
    assert _tree_equal(partial.state, partial_state)


def test_dispatch_owner_checkpoint_tamper_is_rejected() -> None:
    agent = _agent()
    initial = _initial(agent)
    with jax.disable_jit():
        started = agent.start(
            initial,
            jnp.zeros((2,), dtype=jnp.float32),
        ).state
    payload = agent.checkpoint_payload(started)
    payload["dispatch_owner_state"] = started.dispatch_owner.replace(
        hard_safety_action_mask=(
            ~started.dispatch_owner.hard_safety_action_mask
        )
    )
    with pytest.raises(ValueError, match="dispatch owner state SHA differs"):
        agent.restore_checkpoint(
            payload,
            source_digest=_digest("source"),
            semantic_namespace_digest=_digest("namespace"),
            representation_revision=0,
            source_revision=0,
        )


def test_changed_partner_only_dispatch_cancels_exact_feedback_owner_without_learning() -> None:
    from tests.test_prototype_partner_policy_fusion import _sidecar

    agent = _agent(partner=True)
    initial = _initial(agent)
    with jax.disable_jit():
        started = agent.start(
            initial,
            jnp.zeros((2,), dtype=jnp.float32),
        ).state
        base_action = int(started.prototype.current_action)
        partner_action = 1 - base_action
        partner_dispatch = agent.update_transition(
            started,
            _transition(started),
            partner_policy_fusion_input=_sidecar(
                agent.prototype,
                started.prototype,
                suggested_action=partner_action,
            ),
        ).state
    assert not bool(partner_dispatch.controller.pending)
    interaction = partner_dispatch.prototype.ia_state
    assert bool(interaction.feedback_prototype_decision_id_available)
    partner_weights_before = (
        interaction.partner_policy_fusion_state.reliability_weights
    )
    partner_counts_before = (
        interaction.partner_policy_fusion_state.feedback_counts,
        interaction.partner_policy_fusion_state.safe_feedback_counts,
        interaction.partner_policy_fusion_state.feedback_applied_count,
    )

    settled = agent.settle_dispatch(
        partner_dispatch,
        _settlement(partner_dispatch, executed_action=base_action),
    )
    assert bool(settled.diagnostics.transaction_committed)
    assert bool(settled.diagnostics.partner_owner_current)
    assert bool(settled.diagnostics.partner_cancellation_applied)
    assert not bool(settled.diagnostics.procedural_cancellation_required)
    settled_interaction = settled.state.prototype.ia_state
    assert not bool(settled_interaction.feedback_prototype_decision_id_available)
    assert not bool(settled_interaction.partner_policy_fusion_state.feedback_armed)
    np.testing.assert_array_equal(
        settled_interaction.partner_policy_fusion_state.reliability_weights,
        partner_weights_before,
    )
    for after, before in zip(
        (
            settled_interaction.partner_policy_fusion_state.feedback_counts,
            settled_interaction.partner_policy_fusion_state.safe_feedback_counts,
            settled_interaction.partner_policy_fusion_state.feedback_applied_count,
        ),
        partner_counts_before,
        strict=True,
    ):
        np.testing.assert_array_equal(after, before)


def test_historical_procedural_owner_is_preserved_and_cannot_be_misattributed() -> None:
    agent = _agent()
    initial = _initial(agent)
    with jax.disable_jit():
        started = agent.start(
            initial,
            jnp.zeros((2,), dtype=jnp.float32),
            decision_input=_decision_input(initial),
        ).state
        advanced = agent.update_transition(started, _transition(started)).state
    assert bool(advanced.controller.pending)
    assert not bool(
        jnp.array_equal(
            advanced.controller.pending_decision_id,
            advanced.prototype.current_decision_id,
        )
    )
    controller_before = advanced.controller
    fallback = 1 - int(advanced.prototype.current_action)
    settled = agent.settle_dispatch(
        advanced,
        _settlement(advanced, executed_action=fallback),
    )
    assert bool(settled.diagnostics.transaction_committed)
    assert not bool(settled.diagnostics.procedural_owner_current)
    assert not bool(settled.diagnostics.procedural_cancellation_required)
    assert _tree_equal(settled.state.controller, controller_before)


def test_partner_owner_mismatch_and_source_identity_corruption_are_atomic_noops() -> None:
    from tests.test_prototype_partner_policy_fusion import _sidecar

    agent = _agent(partner=True)
    initial = _initial(agent)
    with jax.disable_jit():
        started = agent.start(
            initial,
            jnp.zeros((2,), dtype=jnp.float32),
        ).state
        partner_dispatch = agent.update_transition(
            started,
            _transition(started),
            partner_policy_fusion_input=_sidecar(
                agent.prototype,
                started.prototype,
                suggested_action=1 - int(started.prototype.current_action),
            ),
        ).state
    fallback = 1 - int(partner_dispatch.prototype.current_action)
    interaction = partner_dispatch.prototype.ia_state

    historical_interaction = interaction.replace(
        feedback_prototype_decision_id=started.prototype.current_decision_id
    )
    historical = partner_dispatch.replace(
        prototype=partner_dispatch.prototype.replace(
            ia_state=historical_interaction
        )
    )
    assert bool(agent.validate_state(historical))
    mismatched = agent.settle_dispatch(
        historical,
        _settlement(historical, executed_action=fallback),
    )
    assert not bool(mismatched.diagnostics.partner_owner_current)
    assert not bool(
        mismatched.diagnostics.partner_owner_consistent_for_change
    )
    assert not bool(mismatched.diagnostics.transaction_committed)
    assert _tree_equal(mismatched.state, historical)

    partner_state = interaction.partner_policy_fusion_state
    corrupt_partner_state = partner_state.replace(
        armed_decision_id=jnp.asarray(0, dtype=jnp.int32),
        armed_decision_words=jnp.zeros((2,), dtype=jnp.uint32),
    )
    corrupt_interaction = interaction.replace(
        partner_policy_fusion_state=corrupt_partner_state
    )
    corrupt = partner_dispatch.replace(
        prototype=partner_dispatch.prototype.replace(ia_state=corrupt_interaction)
    )
    assert bool(agent.validate_state(corrupt))
    rejected = agent.settle_dispatch(
        corrupt,
        _settlement(corrupt, executed_action=fallback),
    )
    assert not bool(rejected.diagnostics.partner_armed_identity_current)
    assert not bool(rejected.diagnostics.transaction_committed)
    assert _tree_equal(rejected.state, corrupt)
