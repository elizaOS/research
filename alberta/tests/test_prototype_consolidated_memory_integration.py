# mypy: disable-error-code="arg-type,attr-defined,call-arg,type-var"
"""Live lifecycle integration for Prototype consolidated procedural memory."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from test_prototype_consolidated_memory import (
    _agent,
    _decision_input,
    _digest,
    _feedback,
    _force_action,
    _initial,
    _oak,
    _request,
    _transition,
    _tree_equal,
)

from alberta_framework.core.prototype_agent import (
    PrototypeAgent,
    PrototypeAgentConfig,
    PrototypeExperientialMemoryInput,
    PrototypeRecurrentLatentWorldModelState,
    PrototypeTransition,
)
from alberta_framework.core.prototype_consolidated_memory import (
    PROTOTYPE_CONSOLIDATED_MEMORY_CHECKPOINT_HOST_ONLY,
    PrototypeConsolidatedMemoryFeedbackInput,
)
from alberta_framework.core.recurrent_latent_world_model_ensemble import (
    RecurrentLatentWorldModelEnsembleConfig,
)

pytestmark = pytest.mark.integration


def _experiential_for_next_decision(
    state: Any,
    transition: PrototypeTransition,
    mask: jax.Array,
) -> PrototypeExperientialMemoryInput:
    next_id = state.prototype.current_decision_id.at[3].add(
        jnp.asarray(1, dtype=jnp.uint32)
    )
    return PrototypeExperientialMemoryInput(
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
        provenance_id=jnp.asarray(31, dtype=jnp.int32),
        source_id=jnp.asarray(37, dtype=jnp.int32),
        next_action_safety_mask=mask,
    )


def test_miss_feedback_writes_then_later_decide_changes_real_cached_action() -> None:
    """A retrieval miss remains feedback-trackable and later changes dispatch."""

    with jax.disable_jit():
        agent = _agent()
        initial = _initial(agent)
        started = agent.start(
            initial,
            jnp.zeros((2,), dtype=jnp.float32),
        ).state
        started = _force_action(agent, started, 1)

        miss = agent.decide(started, decision_input=_decision_input(started))
        assert not bool(miss.memory_decision.retrieval.accepted)
        assert bool(miss.memory_decision.diagnostics.feedback_trackable)
        assert bool(miss.state.controller.pending)
        assert int(miss.state.prototype.current_action) == 1

        repeated = agent.decide(
            miss.state,
            decision_input=_decision_input(miss.state),
        )
        assert bool(repeated.memory_decision.diagnostics.duplicate_decision)
        assert _tree_equal(repeated.state.controller, miss.state.controller)
        assert _tree_equal(repeated.state.prototype, miss.state.prototype)

        state = miss.state
        for event in (1, 2):
            transition = _transition(state)
            feedback = _feedback(state, transition, event)
            # The runner need not predict the next Prototype decision ID.
            updated = agent.update_transition(
                state,
                transition,
                feedback_input=feedback,
                decision_input=None,
            )
            assert bool(updated.memory_feedback.diagnostics.write_applied)
            assert int(updated.state.prototype.step_count) == event
            assert not bool(updated.state.controller.pending)
            state = updated.state
            if event == 1:
                state = _force_action(agent, state, 1)
                next_decision = agent.decide(
                    state,
                    decision_input=_decision_input(state),
                )
                assert not bool(next_decision.memory_decision.proposal.available)
                assert bool(next_decision.state.controller.pending)
                state = next_decision.state

        state = _force_action(agent, state, 0)
        changed = agent.decide(state, decision_input=_decision_input(state))
        assert bool(changed.memory_decision.retrieval.accepted)
        assert bool(changed.memory_decision.proposal.available)
        assert int(changed.memory_decision.counterfactual_base_action) == 0
        assert int(changed.memory_decision.action) == 1
        assert bool(changed.dispatch_replacement.committed)
        assert int(changed.state.prototype.current_action) == 1
        stomp = changed.state.prototype.oak_state.stomp_state
        assert int(stomp.last_primitive_action) == 1
        owner_action = jnp.where(
            stomp.executing_option >= 0,
            stomp.option_last_intra_action,
            stomp.base_last_action,
        )
        assert int(owner_action) == 1


def test_deferred_decision_preserves_exact_upstream_mask_until_consumed() -> None:
    """A two-step runner cannot lose, stale-clear, or rewrite upstream safety."""

    with jax.disable_jit():
        agent = _agent(experiential=True)
        started = agent.start(
            _initial(agent),
            jnp.zeros((2,), dtype=jnp.float32),
        ).state
        transition = _transition(started, reward=0.0)
        counterfactual = agent.prototype.update_transition(
            started.prototype,
            transition,
        )
        allowed_action = int(counterfactual.action)
        upstream_mask = jax.nn.one_hot(
            allowed_action, 2, dtype=jnp.bool_
        )
        deferred = agent.update_transition(
            started,
            transition,
            decision_input=None,
            experiential_memory_input=_experiential_for_next_decision(
                started,
                transition,
                upstream_mask,
            ),
        )
        assert bool(deferred.state.upstream_mask.available)
        assert int(deferred.action) == allowed_action
        np.testing.assert_array_equal(
            np.asarray(deferred.state.upstream_mask.prototype_decision_id),
            np.asarray(deferred.state.prototype.current_decision_id),
        )
        np.testing.assert_array_equal(
            np.asarray(deferred.state.upstream_mask.hard_safety_action_mask),
            np.asarray(upstream_mask),
        )

        no_input = agent.decide(deferred.state)
        assert _tree_equal(no_input.state, deferred.state)
        assert bool(no_input.state.upstream_mask.available)

        stale_input = _decision_input(no_input.state).replace(
            prototype_decision_id=(
                no_input.state.prototype.current_decision_id.at[3].add(
                    jnp.asarray(1, dtype=jnp.uint32)
                )
            )
        )
        stale = agent.decide(no_input.state, decision_input=stale_input)
        assert not bool(stale.diagnostics.decision_prototype_id_matches)
        assert _tree_equal(stale.state, no_input.state)
        assert bool(stale.state.upstream_mask.available)

        payload = agent.checkpoint_payload(stale.state)
        restored = agent.restore_checkpoint(
            payload,
            source_digest=_digest("source"),
            semantic_namespace_digest=_digest("namespace"),
            representation_revision=0,
            source_revision=0,
        )
        assert _tree_equal(restored.upstream_mask, stale.state.upstream_mask)

        exact = agent.decide(
            restored,
            decision_input=_decision_input(restored),
        )
        assert bool(exact.dispatch_replacement.committed)
        assert bool(exact.diagnostics.upstream_mask_consumed)
        assert not bool(exact.state.upstream_mask.available)
        np.testing.assert_array_equal(
            np.asarray(exact.diagnostics.final_hard_safety_action_mask),
            np.asarray(upstream_mask),
        )

        corrupt_record = stale.state.upstream_mask.replace(
            hard_safety_action_mask=(
                stale.state.upstream_mask.hard_safety_action_mask.at[
                    allowed_action
                ].set(False)
            )
        )
        corrupt = stale.state.replace(upstream_mask=corrupt_record)
        assert not bool(agent.validate_state(corrupt))
        failed_closed = agent.decide(
            corrupt,
            decision_input=_decision_input(corrupt),
        )
        assert int(failed_closed.action) == -1
        assert not bool(failed_closed.diagnostics.transaction_committed)
        assert _tree_equal(failed_closed.state, corrupt)

        tampered_payload = dict(payload)
        tampered_payload["upstream_mask_state"] = corrupt_record
        with pytest.raises(ValueError, match="upstream mask state SHA differs"):
            agent.restore_checkpoint(
                tampered_payload,
                source_digest=_digest("source"),
                semantic_namespace_digest=_digest("namespace"),
                representation_revision=0,
                source_revision=0,
            )


def test_skipped_decision_checks_realized_action_then_rotates_upstream_mask() -> None:
    """A skipped consolidated query cannot bypass the stored upstream mask."""

    with jax.disable_jit():
        agent = _agent(experiential=True)
        started = agent.start(
            _initial(agent),
            jnp.zeros((2,), dtype=jnp.float32),
        ).state
        first_transition = _transition(started, reward=0.0)
        first_probe = agent.prototype.update_transition(
            started.prototype,
            first_transition,
        )
        first_mask = jax.nn.one_hot(
            int(first_probe.action), 2, dtype=jnp.bool_
        )
        deferred = agent.update_transition(
            started,
            first_transition,
            experiential_memory_input=_experiential_for_next_decision(
                started,
                first_transition,
                first_mask,
            ),
        )
        old_id = deferred.state.upstream_mask.prototype_decision_id

        safe_transition = _transition(deferred.state, reward=0.0)
        safe = agent.update_transition(
            deferred.state,
            safe_transition,
            experiential_memory_input=_experiential_for_next_decision(
                deferred.state,
                safe_transition,
                jnp.ones((2,), dtype=jnp.bool_),
            ),
        )
        assert bool(safe.diagnostics.prior_upstream_mask_available)
        assert bool(safe.diagnostics.prior_upstream_mask_decision_matches)
        assert bool(
            safe.diagnostics.realized_action_allowed_by_prior_upstream_mask
        )
        assert bool(safe.diagnostics.prototype_learning_retained)
        assert bool(safe.state.upstream_mask.available)
        assert not bool(
            jnp.array_equal(
                safe.state.upstream_mask.prototype_decision_id,
                old_id,
            )
        )

        unsafe_action = 1 - int(deferred.state.prototype.current_action)
        unsafe_replacement = agent.prototype.replace_cached_primitive_action(
            deferred.state.prototype,
            decision_id=deferred.state.prototype.current_decision_id,
            decision_observation=deferred.state.prototype.current_representation,
            proposed_action=jnp.asarray(unsafe_action, dtype=jnp.int32),
            safety_action_mask=jnp.ones((2,), dtype=jnp.bool_),
        )
        assert bool(unsafe_replacement.committed)
        # The historical helper shape changed only Prototype. The v2 bound
        # owner makes that incoherent fixture explicitly invalid.
        unsafe_state = deferred.state.replace(
            prototype=unsafe_replacement.state
        )
        assert not bool(agent.validate_state(unsafe_state))
        unsafe_transition = _transition(unsafe_state, reward=0.0)
        rejected = agent.update_transition(
            unsafe_state,
            unsafe_transition,
            experiential_memory_input=_experiential_for_next_decision(
                unsafe_state,
                unsafe_transition,
                jnp.ones((2,), dtype=jnp.bool_),
            ),
        )
        assert bool(rejected.diagnostics.prior_upstream_mask_available)
        assert not bool(
            rejected.diagnostics.realized_action_allowed_by_prior_upstream_mask
        )
        assert not bool(rejected.diagnostics.transition_valid_before_feedback)
        assert not bool(rejected.diagnostics.prototype_learning_retained)
        assert not bool(rejected.diagnostics.transaction_committed)
        assert int(rejected.action) == -1
        assert _tree_equal(rejected.state, unsafe_state)


def test_bad_feedback_is_memory_noop_while_valid_prototype_learning_advances() -> None:
    with jax.disable_jit():
        agent = _agent()
        initial = _initial(agent)
        armed = agent.start(
            initial,
            jnp.zeros((2,), dtype=jnp.float32),
            decision_input=_decision_input(initial),
        ).state
        transition = _transition(armed)
        exact = _feedback(armed, transition, 1)
        settled = agent.update_transition(
            armed,
            transition,
            feedback_input=exact,
            decision_input=None,
        )
        assert bool(settled.memory_feedback.diagnostics.write_applied)
        controller_after_write = settled.state.controller

        duplicate_transition = _transition(settled.state)
        duplicate = PrototypeConsolidatedMemoryFeedbackInput(
            available=jnp.asarray(True, dtype=jnp.bool_),
            prototype_decision_id=duplicate_transition.decision_id,
            feedback_event_id=jnp.asarray((0, 0, 0, 1), dtype=jnp.uint32),
            base_action=jnp.asarray(0, dtype=jnp.int32),
            effective_action=duplicate_transition.action,
            request=_request(),
            succeeded=jnp.asarray(True, dtype=jnp.bool_),
            outcome=jnp.asarray((1.0,), dtype=jnp.float32),
            confidence=jnp.asarray(1.0, dtype=jnp.float32),
            evidence=jnp.asarray(1.0, dtype=jnp.float32),
        )
        duplicate_result = agent.update_transition(
            settled.state,
            duplicate_transition,
            feedback_input=duplicate,
        )
        assert bool(
            duplicate_result.memory_feedback.diagnostics.duplicate_feedback_event
        )
        assert _tree_equal(
            duplicate_result.state.controller,
            controller_after_write,
        )
        assert int(duplicate_result.state.prototype.step_count) == 2

        misattributed_transition = _transition(duplicate_result.state)
        wrong_action = duplicate.replace(
            prototype_decision_id=misattributed_transition.decision_id,
            feedback_event_id=jnp.asarray((0, 0, 0, 2), dtype=jnp.uint32),
            effective_action=(
                misattributed_transition.action + jnp.asarray(1, dtype=jnp.int32)
            )
            % jnp.asarray(2, dtype=jnp.int32),
        )
        misattributed = agent.update_transition(
            duplicate_result.state,
            misattributed_transition,
            feedback_input=wrong_action,
        )
        assert not bool(
            misattributed.diagnostics.feedback_realized_action_matches
        )
        assert _tree_equal(
            misattributed.state.controller,
            controller_after_write,
        )
        assert int(misattributed.state.prototype.step_count) == 3


def test_prototype_optional_composition_rollback_also_rolls_back_memory_feedback() -> None:
    """A raw-valid transition is not a commit until Prototype's postcheck passes."""

    with jax.disable_jit():
        agent = _agent(experiential=True)
        initial = _initial(agent)
        pending = agent.start(
            initial,
            jnp.zeros((2,), dtype=jnp.float32),
            decision_input=_decision_input(initial),
        ).state
        transition = _transition(pending)
        next_id = pending.prototype.current_decision_id.at[3].add(
            jnp.asarray(1, dtype=jnp.uint32)
        )
        rejected_memory = PrototypeExperientialMemoryInput(
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
            provenance_id=jnp.asarray(7, dtype=jnp.int32),
            source_id=jnp.asarray(11, dtype=jnp.int32),
            next_action_safety_mask=jnp.zeros((2,), dtype=jnp.bool_),
        )
        result = agent.update_transition(
            pending,
            transition,
            feedback_input=_feedback(pending, transition, 1),
            experiential_memory_input=rejected_memory,
        )
        assert bool(result.diagnostics.transition_valid_before_feedback)
        assert bool(result.diagnostics.composed_state_valid_before)
        assert not bool(result.diagnostics.next_dispatch_allowed)
        assert not bool(result.prototype.transition_diagnostics.valid)
        assert not bool(result.diagnostics.prototype_learning_retained)
        assert not bool(result.memory_feedback.diagnostics.write_applied)
        assert _tree_equal(result.memory_feedback.state, pending.controller)
        assert _tree_equal(result.state, pending)
        assert int(result.action) == -1


def test_atomic_replacement_synchronizes_recurrent_decision_cache() -> None:
    prototype = PrototypeAgent(
        PrototypeAgentConfig(
            oak=_oak(),
            recurrent_latent_world_model_ensemble=(
                RecurrentLatentWorldModelEnsembleConfig(
                    observation_dim=2,
                    n_actions=2,
                    latent_dim=2,
                    ensemble_size=2,
                    learning_rate=0.01,
                    bootstrap_probability=0.7,
                    uncertainty_warmup_steps=1,
                    initialization_scale=0.1,
                    max_updates=20,
                )
            ),
        )
    )
    state = prototype.start(
        prototype.init(
            jr.key(3), lifecycle_id=jnp.asarray((31, 37), dtype=jnp.uint32)
        ),
        jnp.asarray((0.25, -0.5), dtype=jnp.float32),
    )
    replacement = prototype.replace_cached_primitive_action(
        state,
        decision_id=state.current_decision_id,
        decision_observation=state.current_representation,
        proposed_action=(state.current_action + 1) % jnp.asarray(2, dtype=jnp.int32),
        safety_action_mask=jnp.ones((2,), dtype=jnp.bool_),
    )
    assert bool(replacement.committed)
    wrapper = replacement.state.world_model_state
    assert isinstance(wrapper, PrototypeRecurrentLatentWorldModelState)
    assert int(wrapper.decision_cache.action) == int(replacement.state.current_action)
    assert bool(wrapper.decision_cache.valid)
    assert bool(prototype.validate_state(replacement.state))


def test_enabled_checkpoint_pending_recovery_tamper_and_rebind_reset() -> None:
    with jax.disable_jit():
        agent = _agent()
        initial = _initial(agent)
        pending = agent.start(
            initial,
            jnp.zeros((2,), dtype=jnp.float32),
            decision_input=_decision_input(initial),
        ).state
        payload = agent.checkpoint_payload(pending)
        assert PROTOTYPE_CONSOLIDATED_MEMORY_CHECKPOINT_HOST_ONLY
        restored = agent.restore_checkpoint(
            payload,
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
        transition = _transition(restored)
        recovered = agent.update_transition(
            restored,
            transition,
            feedback_input=_feedback(restored, transition, 1),
        )
        assert bool(recovered.memory_feedback.diagnostics.write_applied)

        tampered = dict(payload)
        tampered["prototype_state"] = pending.prototype.replace(
            current_action=(pending.prototype.current_action + 1)
            % jnp.asarray(2, dtype=jnp.int32)
        )
        with pytest.raises(ValueError, match="SHA differs"):
            agent.restore_checkpoint(
                tampered,
                source_digest=_digest("source"),
                semantic_namespace_digest=_digest("namespace"),
                representation_revision=0,
                source_revision=0,
            )
        with pytest.raises(ValueError, match="discard_pending=True"):
            agent.rebind_reset(
                pending,
                source_digest=_digest("source-new"),
                semantic_namespace_digest=_digest("namespace-new"),
                representation_revision=1,
                source_revision=1,
            )
        reset = agent.rebind_reset(
            pending,
            source_digest=_digest("source-new"),
            semantic_namespace_digest=_digest("namespace-new"),
            representation_revision=1,
            source_revision=1,
            discard_pending=True,
        )
        assert not bool(reset.controller.pending)
        assert _tree_equal(reset.prototype, pending.prototype)
        np.testing.assert_array_equal(
            np.asarray(reset.controller.memory.source_digest),
            np.asarray(_digest("source-new")),
        )


def test_eager_jit_and_scan_use_the_same_no_sidecar_transition_path() -> None:
    agent = _agent()
    initial = _initial(agent)
    state = agent.start(
        initial,
        jnp.zeros((2,), dtype=jnp.float32),
    ).state
    transition = _transition(state, reward=0.25)
    eager = agent.update_transition(state, transition)
    compiled = jax.jit(agent.update_transition)(state, transition)
    assert _tree_equal(eager.state, compiled.state)
    np.testing.assert_array_equal(np.asarray(eager.action), np.asarray(compiled.action))

    observations = jnp.asarray(
        ((0.1, -0.2), (0.2, -0.1), (0.3, 0.0)),
        dtype=jnp.float32,
    )

    def scan_step(
        carry: Any,
        next_observation: jax.Array,
    ) -> tuple[Any, jax.Array]:
        step_transition = PrototypeTransition(
            observation=carry.prototype.current_raw_observation,
            action=carry.prototype.current_action,
            decision_id=carry.prototype.current_decision_id,
            reward=jnp.asarray(0.25, dtype=jnp.float32),
            discount=jnp.asarray(1.0, dtype=jnp.float32),
            terminated=jnp.asarray(False, dtype=jnp.bool_),
            truncated=jnp.asarray(False, dtype=jnp.bool_),
            next_observation=next_observation,
            next_decision_observation=next_observation,
        )
        result = agent.update_transition(carry, step_transition)
        return result.state, result.action

    scan_state, scan_actions = jax.lax.scan(scan_step, state, observations)
    sequential_state = state
    sequential_actions: list[jax.Array] = []
    for observation in observations:
        sequential_state, action = scan_step(sequential_state, observation)
        sequential_actions.append(action)
    assert _tree_equal(scan_state, sequential_state)
    np.testing.assert_array_equal(
        np.asarray(scan_actions),
        np.asarray(jnp.stack(sequential_actions)),
    )
