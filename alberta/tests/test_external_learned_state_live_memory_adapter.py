# mypy: disable-error-code="attr-defined,call-arg,no-any-return,operator"
# mypy: disable-error-code="arg-type,type-var,union-attr"
"""One-owner live learned-memory adapter around the external-state coordinator."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from test_external_learned_state_router_audit_coordinator import _coordinator, _transition

import alberta_framework as alberta
import alberta_framework.core as core
from alberta_framework.core.experiential_memory import ExperientialMemoryConfig
from alberta_framework.core.external_learned_state_live_memory_adapter import (
    EXTERNAL_LEARNED_STATE_LIVE_MEMORY_ADAPTER_CHECKPOINT_SCHEMA,
    EXTERNAL_LEARNED_STATE_LIVE_MEMORY_ADAPTER_EVIDENCE_LEVEL,
    EXTERNAL_LEARNED_STATE_LIVE_MEMORY_ADAPTER_OUTCOME_STATUS,
    ExternalLearnedStateLiveMemoryAdapter,
    ExternalLearnedStateLiveMemoryAdapterConfig,
    ExternalLearnedStateLiveMemoryAdapterState,
    ExternalLearnedStateLiveMemoryEventInput,
    ExternalLearnedStateLiveMemoryFeedback,
    load_external_learned_state_live_memory_adapter_checkpoint,
    save_external_learned_state_live_memory_adapter_checkpoint,
)
from alberta_framework.core.learned_experiential_memory_controller import (
    LearnedExperientialMemoryControllerConfig,
)

pytestmark = [pytest.mark.integration, pytest.mark.slow]


@pytest.fixture(autouse=True)
def _bounded_jax_execution(request: pytest.FixtureRequest) -> object:
    if request.node.name == "test_monolithic_jit_rejects_before_donor_work":
        yield
    else:
        with jax.disable_jit():
            yield


def _adapter(*, top_k: int = 1) -> ExternalLearnedStateLiveMemoryAdapter:
    coordinator = _coordinator()
    memory = ExperientialMemoryConfig(
        capacity=4,
        observation_dim=2,
        key_dim=2,
        action_dim=2,
        outcome_dim=2,
        top_k=top_k,
        min_neighbors=1,
        distance_scale=1.0,
        min_similarity=0.0,
        min_effective_reliability=0.01,
        max_uncertainty=1.0,
        max_safety_cost=1.0,
        max_age=100,
        staleness_scale=100.0,
        utility_decay=1.0,
        eviction_utility_weight=1.0,
        eviction_recency_weight=1.0,
        recency_scale=10.0,
    )
    return ExternalLearnedStateLiveMemoryAdapter(
        ExternalLearnedStateLiveMemoryAdapterConfig(
            coordinator=coordinator.config,
            learned_memory=LearnedExperientialMemoryControllerConfig(
                memory=memory,
                admission_threshold=0.0,
                initial_admission_bias=0.0,
            ),
        )
    )


def _event_input(
    *,
    provenance: int = 1,
    query_uncertainty: float = 0.0,
) -> ExternalLearnedStateLiveMemoryEventInput:
    return ExternalLearnedStateLiveMemoryEventInput(
        query_uncertainty=jnp.asarray(query_uncertainty, dtype=jnp.float32),
        query_uncertainty_available=jnp.asarray(True, dtype=jnp.bool_),
        entry_uncertainty=jnp.asarray(0.0, dtype=jnp.float32),
        entry_uncertainty_available=jnp.asarray(True, dtype=jnp.bool_),
        entry_safety_cost=jnp.asarray(0.0, dtype=jnp.float32),
        entry_safety_cost_available=jnp.asarray(True, dtype=jnp.bool_),
        entry_reliability=jnp.asarray(1.0, dtype=jnp.float32),
        provenance_id=jnp.asarray(provenance, dtype=jnp.int32),
        source_id=jnp.asarray(9, dtype=jnp.int32),
    )


def _tree_exact(left: object, right: object) -> None:
    left_leaves, left_tree = jax.tree.flatten(left)
    right_leaves, right_tree = jax.tree.flatten(right)
    assert left_tree == right_tree
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        left_array = jnp.asarray(left_leaf)
        right_array = jnp.asarray(right_leaf)
        if jax.dtypes.issubdtype(left_array.dtype, jax.dtypes.prng_key):
            left_array = jr.key_data(left_array)
            right_array = jr.key_data(right_array)
        np.testing.assert_array_equal(np.asarray(left_array), np.asarray(right_array))


def _force_action(
    adapter: ExternalLearnedStateLiveMemoryAdapter,
    state: ExternalLearnedStateLiveMemoryAdapterState,
    action: int,
    mask: jax.Array | None = None,
) -> ExternalLearnedStateLiveMemoryAdapterState:
    coordinator = state.coordinator_state
    prototype_state = coordinator.inner_state.prototype_state
    replacement = adapter.coordinator.inner.prototype.replace_cached_primitive_action(
        prototype_state,
        decision_id=prototype_state.current_decision_id,
        decision_observation=prototype_state.current_representation,
        proposed_action=jnp.asarray(action, dtype=jnp.int32),
        safety_action_mask=(
            jnp.ones((2,), dtype=jnp.bool_) if mask is None else mask
        ),
    )
    assert bool(replacement.committed)
    inner = coordinator.inner_state.replace(prototype_state=replacement.state)
    rebound = coordinator.replace(
        inner_state=inner,
        current_action=replacement.action,
        current_decision_id=replacement.state.current_decision_id,
        cached_prototype_step_words=replacement.state.step_words,
        cached_feature_generation_words=(
            adapter.coordinator._feature_generation_words(inner)
        ),
    )
    result = state.replace(coordinator_state=rebound)
    assert bool(adapter.state_valid(result))
    return result


def _replace_stored_action(
    adapter: ExternalLearnedStateLiveMemoryAdapter,
    state: ExternalLearnedStateLiveMemoryAdapterState,
    slot: int,
    action: jax.Array,
) -> ExternalLearnedStateLiveMemoryAdapterState:
    learned = state.learned_memory_state
    memory = learned.memory
    entries = dataclasses.replace(
        memory.entries,
        actions=memory.entries.actions.at[slot].set(action),
    )
    candidate = state.replace(
        learned_memory_state=learned.replace(
            memory=dataclasses.replace(memory, entries=entries)
        )
    )
    assert bool(adapter.state_valid(candidate))
    return candidate


def _feedback(
    state: ExternalLearnedStateLiveMemoryAdapterState,
    *,
    learn: bool,
) -> ExternalLearnedStateLiveMemoryFeedback:
    pending = state.pending_binding
    assert bool(pending.available)
    expected = bool(pending.retrieval_used_expected)
    return ExternalLearnedStateLiveMemoryFeedback(
        memory_transaction_words=pending.memory_transaction_words,
        prototype_decision_id=pending.prototype_decision_id,
        base_action_before_retrieval=pending.base_action_before_retrieval,
        effective_action=pending.effective_action,
        hard_action_mask=pending.hard_action_mask,
        retrieval_used=pending.retrieval_used_expected,
        counterfactual_available=jnp.asarray(
            expected and learn,
            dtype=jnp.bool_,
        ),
        counterfactual_delta=jnp.asarray(
            0.5 if expected and learn else 0.0,
            dtype=jnp.float32,
        ),
    )


def _one_entry_state(
    adapter: ExternalLearnedStateLiveMemoryAdapter,
    *,
    seed: int,
) -> tuple[ExternalLearnedStateLiveMemoryAdapterState, int, jax.Array]:
    state = adapter.start(
        adapter.init(jr.key(seed)),
        jnp.asarray((0.2, -0.1), dtype=jnp.float32),
    )
    first = adapter.step(
        state,
        _transition(state.coordinator_state),
        _event_input(),
        jnp.ones((2,), dtype=jnp.bool_),
    )
    assert bool(first.diagnostics.transaction_applied)
    assert not bool(first.state.pending_binding.available)
    return (
        first.state,
        int(first.prepared.learned_memory_result.slot),
        state.coordinator_state.current_raw_observation,
    )


def test_live_memory_adapter_has_a_separate_versioned_owner() -> None:
    assert ExternalLearnedStateLiveMemoryAdapter.__module__.endswith(
        "external_learned_state_live_memory_adapter"
    )


def test_first_event_queries_empty_store_then_writes_executed_transition() -> None:
    adapter = _adapter()
    state = adapter.start(
        adapter.init(jr.key(3)),
        jnp.asarray((0.2, -0.1), dtype=jnp.float32),
    )
    transition = _transition(state.coordinator_state)
    result = adapter.step(
        state,
        transition,
        _event_input(),
        jnp.ones((2,), dtype=jnp.bool_),
    )

    assert bool(result.diagnostics.transaction_applied)
    assert bool(result.diagnostics.query_before_write)
    assert bool(result.diagnostics.completed_entry_executed_action_exact)
    assert not bool(result.prepared.learned_memory_result.retrieval.accepted)
    assert bool(result.prepared.learned_memory_result.wrote)
    assert not bool(result.state.pending_binding.available)
    assert int(result.state.learned_memory_state.memory.active_count) == 1
    assert int(result.state.learned_memory_state.transaction_words[1]) == 1
    assert int(result.state.coordinator_state.event_words[1]) == 1
    slot = int(result.prepared.learned_memory_result.slot)
    entries = result.state.learned_memory_state.memory.entries
    np.testing.assert_array_equal(entries.observations[slot], transition.observation)
    np.testing.assert_array_equal(entries.keys[slot], transition.observation)
    np.testing.assert_array_equal(entries.outcomes[slot], transition.next_observation)
    np.testing.assert_array_equal(
        entries.actions[slot],
        jax.nn.one_hot(transition.action, 2, dtype=jnp.float32),
    )
    np.testing.assert_array_equal(entries.rewards[slot], transition.reward)
    assert bool(adapter.state_valid(result.state))


def test_categorical_retrieval_changes_real_next_action_and_feedback_is_exact(
    tmp_path: Path,
) -> None:
    adapter = _adapter()
    state, slot, stored_key = _one_entry_state(adapter, seed=5)
    next_observation = tuple(float(value) for value in np.asarray(stored_key))
    transition = _transition(
        state.coordinator_state,
        next_observation=next_observation,
    )
    preview = adapter.coordinator.step(state.coordinator_state, transition)
    desired = 1 - int(preview.state.current_action)
    state = _replace_stored_action(
        adapter,
        state,
        slot,
        jax.nn.one_hot(desired, 2, dtype=jnp.float32),
    )

    changed = adapter.step(
        state,
        transition,
        _event_input(provenance=2),
        jnp.ones((2,), dtype=jnp.bool_),
    )
    assert bool(changed.diagnostics.transaction_applied)
    assert bool(changed.diagnostics.categorical_retrieval)
    assert bool(changed.diagnostics.cached_action_replacement_committed)
    assert bool(changed.diagnostics.next_action_changed)
    assert int(changed.state.coordinator_state.current_action) == desired
    prototype = changed.state.coordinator_state.inner_state.prototype_state
    assert int(prototype.current_action) == desired
    stomp = prototype.oak_state.oak_state.stomp_state
    assert int(stomp.last_primitive_action) == desired
    owner_action = jnp.where(
        stomp.executing_option >= 0,
        stomp.option_last_intra_action,
        stomp.base_last_action,
    )
    assert int(owner_action) == desired
    assert bool(changed.state.pending_binding.available)
    assert bool(changed.state.pending_binding.retrieval_used_expected)
    assert int(changed.state.pending_binding.base_action_before_retrieval) == int(
        preview.state.current_action
    )
    assert int(changed.state.pending_binding.effective_action) == desired
    assert int(changed.state.pending_binding.retrieval_action) == desired
    assert int(changed.state.pending_binding.base_action_before_retrieval) != int(
        changed.state.pending_binding.effective_action
    )
    np.testing.assert_array_equal(
        changed.state.pending_binding.hard_action_mask,
        jnp.ones((2,), dtype=jnp.bool_),
    )
    np.testing.assert_array_equal(
        changed.prepared.completed_entry.action,
        jax.nn.one_hot(transition.action, 2, dtype=jnp.float32),
    )
    assert int(transition.action) != desired or bool(
        changed.diagnostics.next_action_changed
    )

    checkpoint = tmp_path / "live-memory-pending"
    save_external_learned_state_live_memory_adapter_checkpoint(
        adapter,
        changed.state,
        checkpoint,
    )
    restored_adapter, restored_state = (
        load_external_learned_state_live_memory_adapter_checkpoint(checkpoint)
    )
    _tree_exact(restored_state, changed.state)
    assert int(restored_state.pending_binding.base_action_before_retrieval) == int(
        changed.state.pending_binding.base_action_before_retrieval
    )
    np.testing.assert_array_equal(
        restored_state.pending_binding.hard_action_mask,
        changed.state.pending_binding.hard_action_mask,
    )

    retry_transition = _transition(
        changed.state.coordinator_state,
        next_observation=(0.7, -0.3),
    )
    missing = adapter.step(
        changed.state,
        retry_transition,
        _event_input(provenance=3),
        jnp.ones((2,), dtype=jnp.bool_),
    )
    assert bool(missing.diagnostics.rejected)
    assert int(missing.diagnostics.settlement_evaluations) == 0
    assert int(missing.diagnostics.coordinator_evaluations) == 0
    _tree_exact(missing.state, changed.state)

    mismatched_base = dataclasses.replace(
        _feedback(changed.state, learn=True),
        base_action_before_retrieval=jnp.asarray(desired, dtype=jnp.int32),
    )
    refused_base = adapter.step(
        changed.state,
        retry_transition,
        _event_input(provenance=3),
        jnp.ones((2,), dtype=jnp.bool_),
        mismatched_base,
    )
    assert not bool(refused_base.diagnostics.prior_feedback_identity_valid)
    assert int(refused_base.diagnostics.settlement_evaluations) == 0
    assert int(refused_base.diagnostics.coordinator_evaluations) == 0
    _tree_exact(refused_base.state, changed.state)

    mismatched_mask = dataclasses.replace(
        _feedback(changed.state, learn=True),
        hard_action_mask=jnp.zeros((2,), dtype=jnp.bool_),
    )
    refused_mask = adapter.step(
        changed.state,
        retry_transition,
        _event_input(provenance=3),
        jnp.ones((2,), dtype=jnp.bool_),
        mismatched_mask,
    )
    assert not bool(refused_mask.diagnostics.prior_feedback_identity_valid)
    assert int(refused_mask.diagnostics.settlement_evaluations) == 0
    assert int(refused_mask.diagnostics.coordinator_evaluations) == 0
    _tree_exact(refused_mask.state, changed.state)

    invalid_feedback = dataclasses.replace(
        _feedback(changed.state, learn=True),
        counterfactual_delta=jnp.asarray(2.0, dtype=jnp.float32),
    )
    failed_settlement = adapter.step(
        changed.state,
        retry_transition,
        _event_input(provenance=3),
        jnp.ones((2,), dtype=jnp.bool_),
        invalid_feedback,
    )
    assert bool(failed_settlement.diagnostics.rejected)
    assert int(failed_settlement.diagnostics.settlement_evaluations) == 1
    assert int(failed_settlement.diagnostics.coordinator_evaluations) == 0
    _tree_exact(failed_settlement.state, changed.state)

    before_weights = changed.state.learned_memory_state.admission_weights
    settled = restored_adapter.step(
        restored_state,
        retry_transition,
        _event_input(provenance=3, query_uncertainty=2.0),
        jnp.ones((2,), dtype=jnp.bool_),
        _feedback(changed.state, learn=True),
    )
    assert bool(settled.diagnostics.transaction_applied)
    assert bool(settled.diagnostics.prior_feedback_settled)
    assert bool(settled.diagnostics.prior_feedback_learning_applied)
    assert int(settled.state.learned_memory_state.feedback_count) == 1
    assert int(settled.state.learned_memory_state.learned_feedback_count) == 1
    assert not np.array_equal(
        np.asarray(settled.state.learned_memory_state.admission_weights),
        np.asarray(before_weights),
    )
    assert not bool(settled.state.pending_binding.available)


def test_hard_mask_fallback_and_replacement_failure_are_atomic() -> None:
    adapter = _adapter()
    state, slot, stored_key = _one_entry_state(adapter, seed=7)
    transition = _transition(
        state.coordinator_state,
        next_observation=tuple(float(value) for value in np.asarray(stored_key)),
    )
    preview = adapter.coordinator.step(state.coordinator_state, transition)
    safe_current = int(preview.state.current_action)
    unsafe_retrieval = 1 - safe_current
    state = _replace_stored_action(
        adapter,
        state,
        slot,
        jax.nn.one_hot(unsafe_retrieval, 2, dtype=jnp.float32),
    )
    hard_mask = jax.nn.one_hot(safe_current, 2, dtype=jnp.bool_)
    fallback = adapter.step(
        state,
        transition,
        _event_input(provenance=4),
        hard_mask,
    )
    assert bool(fallback.diagnostics.transaction_applied)
    assert bool(fallback.diagnostics.cached_action_replacement_committed)
    assert bool(fallback.diagnostics.used_safe_current_action_fallback)
    assert not bool(fallback.diagnostics.next_action_changed)
    assert int(fallback.state.coordinator_state.current_action) == safe_current
    assert bool(fallback.state.pending_binding.available)
    assert not bool(fallback.state.pending_binding.retrieval_used_expected)
    assert int(fallback.state.pending_binding.base_action_before_retrieval) == int(
        fallback.state.pending_binding.effective_action
    )
    np.testing.assert_array_equal(
        fallback.state.pending_binding.hard_action_mask,
        hard_mask,
    )

    refused = adapter.step(
        state,
        transition,
        _event_input(provenance=4),
        jnp.zeros((2,), dtype=jnp.bool_),
    )
    assert bool(refused.diagnostics.rejected)
    assert int(refused.diagnostics.coordinator_evaluations) == 1
    assert int(refused.diagnostics.learned_memory_query_evaluations) == 1
    assert int(refused.diagnostics.learned_memory_write_evaluations) == 1
    assert int(refused.diagnostics.cached_action_replacement_evaluations) == 1
    assert not bool(refused.diagnostics.cached_action_replacement_committed)
    _tree_exact(refused.state, state)


def test_soft_retrieval_writes_but_has_no_action_or_learning_authority() -> None:
    adapter = _adapter()
    state, slot, stored_key = _one_entry_state(adapter, seed=11)
    state = _replace_stored_action(
        adapter,
        state,
        slot,
        jnp.asarray((0.25, 0.75), dtype=jnp.float32),
    )
    transition = _transition(
        state.coordinator_state,
        next_observation=tuple(float(value) for value in np.asarray(stored_key)),
    )
    soft = adapter.step(
        state,
        transition,
        _event_input(provenance=5),
        jnp.ones((2,), dtype=jnp.bool_),
    )
    assert bool(soft.diagnostics.transaction_applied)
    assert bool(soft.prepared.learned_memory_result.retrieval.accepted)
    assert not bool(soft.diagnostics.categorical_retrieval)
    assert bool(soft.diagnostics.soft_retrieval_denied_action_authority)
    assert int(soft.diagnostics.cached_action_replacement_evaluations) == 0
    assert not bool(soft.diagnostics.next_action_changed)
    assert int(soft.state.coordinator_state.current_action) == int(
        soft.prepared.coordinator_result.state.current_action
    )
    assert bool(soft.state.pending_binding.available)
    assert not bool(soft.state.pending_binding.retrieval_used_expected)
    assert int(soft.state.pending_binding.base_action_before_retrieval) == int(
        soft.state.pending_binding.effective_action
    )
    np.testing.assert_array_equal(
        soft.state.pending_binding.hard_action_mask,
        jnp.ones((2,), dtype=jnp.bool_),
    )

    weights = soft.state.learned_memory_state.admission_weights
    learned_count = soft.state.learned_memory_state.learned_feedback_count
    settled = adapter.step(
        soft.state,
        _transition(
            soft.state.coordinator_state,
            next_observation=(0.8, -0.4),
        ),
        _event_input(provenance=6, query_uncertainty=2.0),
        jnp.ones((2,), dtype=jnp.bool_),
        _feedback(soft.state, learn=False),
    )
    assert bool(settled.diagnostics.transaction_applied)
    assert bool(settled.diagnostics.prior_feedback_settled)
    assert not bool(settled.diagnostics.prior_feedback_learning_applied)
    np.testing.assert_array_equal(
        settled.state.learned_memory_state.admission_weights,
        weights,
    )
    np.testing.assert_array_equal(
        settled.state.learned_memory_state.learned_feedback_count,
        learned_count,
    )
    assert not bool(settled.state.pending_binding.available)


def test_config_resources_exports_and_single_memory_owner() -> None:
    adapter = _adapter()
    state = adapter.init(jr.key(13))
    assert core.ExternalLearnedStateLiveMemoryAdapter is (
        ExternalLearnedStateLiveMemoryAdapter
    )
    assert alberta.EXTERNAL_LEARNED_STATE_LIVE_MEMORY_ADAPTER_CHECKPOINT_SCHEMA == (
        EXTERNAL_LEARNED_STATE_LIVE_MEMORY_ADAPTER_CHECKPOINT_SCHEMA
    )
    payload = adapter.to_config()
    assert ExternalLearnedStateLiveMemoryAdapter.from_config(
        payload
    ).to_config() == payload
    assert payload["evidence_level"] == (
        EXTERNAL_LEARNED_STATE_LIVE_MEMORY_ADAPTER_EVIDENCE_LEVEL
    )
    assert payload["outcome_status"] == (
        EXTERNAL_LEARNED_STATE_LIVE_MEMORY_ADAPTER_OUTCOME_STATUS
    )
    assert payload["raw_schema_version"] == 0
    assert payload["learned_memory_owner_count"] == 1
    assert payload["prototype_historical_memory_enabled"] is False
    assert payload["pending_binding_base_action_recorded"] is True
    assert payload["pending_binding_hard_action_mask_recorded"] is True
    assert payload["learned_embedding_enabled"] is False
    assert payload["reencoding_enabled"] is False
    assert payload["monolithic_jit_supported"] is False
    assert payload["scan_supported"] is False
    assert payload["feedback_authenticated"] is False
    assert payload["dispatch_authority"] is False
    assert payload["safety_authority"] is False
    assert payload["evidence_authority"] is False
    assert payload["promotion_authority"] is False

    budget = adapter.resource_budget
    assert budget.persistent_state_bytes > 0
    assert budget.persistent_capacity_growth == 0
    assert budget.learned_memory_owner_count == 1
    assert budget.prototype_historical_memory_owner_count == 0
    assert budget.coordinator_owner_count == 1
    assert budget.pending_binding_base_action_fields == 1
    assert budget.pending_binding_hard_action_mask_elements == 2
    assert budget.maximum_settlements_per_event == 1
    assert budget.coordinator_evaluations_per_event == 1
    assert budget.learned_memory_queries_per_event == 1
    assert budget.learned_memory_writes_per_event == 1
    assert budget.maximum_cached_action_replacements_per_event == 1
    assert budget.learned_embedding_evaluations_per_event == 0
    assert budget.reencoding_evaluations_per_event == 0
    assert budget.monolithic_jit_supported is False
    assert budget.scan_supported is False
    assert budget.dispatch_authority is False
    assert budget.safety_authority is False
    assert budget.evidence_authority is False
    assert bool(adapter.state_valid(state))

    with pytest.raises(ValueError, match="sole memory owner"):
        ExternalLearnedStateLiveMemoryAdapterConfig(
            coordinator=_coordinator(memory=True).config,
            learned_memory=adapter.config.learned_memory,
        )


@pytest.mark.parametrize(
    ("field", "alias"),
    (
        ("scientific_promotion_allowed", 0),
        ("raw_schema_version", False),
        ("learned_memory_owner_count", True),
    ),
)
def test_config_rejects_bool_integer_canonical_type_aliases(
    field: str,
    alias: object,
) -> None:
    payload = _adapter().to_config()
    payload[field] = alias

    with pytest.raises(ValueError, match="fixed semantics|canonical"):
        ExternalLearnedStateLiveMemoryAdapter.from_config(payload)


def test_checkpoint_metadata_rejects_boolean_integer_alias(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _adapter()
    state = adapter.init(jr.key(131))
    checkpoint = tmp_path / "canonical-type-alias"
    save_external_learned_state_live_memory_adapter_checkpoint(
        adapter,
        state,
        checkpoint,
    )
    from alberta_framework.core import external_learned_state_live_memory_adapter as module

    metadata = module.load_checkpoint_metadata(checkpoint)
    metadata["learned_memory_owner_count"] = True
    monkeypatch.setattr(module, "load_checkpoint_metadata", lambda _: metadata)

    with pytest.raises(ValueError, match="fixed semantics"):
        load_external_learned_state_live_memory_adapter_checkpoint(checkpoint)


def test_receipt_staleness_tamper_and_corrupt_source_roll_back() -> None:
    adapter = _adapter()
    state = adapter.start(
        adapter.init(jr.key(17)),
        jnp.asarray((0.2, -0.1), dtype=jnp.float32),
    )
    transition = _transition(state.coordinator_state)
    prepared = adapter.prepare_transition(
        state,
        transition,
        _event_input(provenance=7),
        jnp.ones((2,), dtype=jnp.bool_),
    )
    receipt = adapter.integrity_receipt(prepared)

    tampered_entry = dataclasses.replace(
        prepared.completed_entry,
        reward=prepared.completed_entry.reward + jnp.asarray(1.0, dtype=jnp.float32),
    )
    tampered_receipt = dataclasses.replace(
        receipt,
        prepared=dataclasses.replace(prepared, completed_entry=tampered_entry),
    )
    refused_tamper = adapter.adopt_prepared_transition(
        state,
        prepared,
        tampered_receipt,
    )
    assert not bool(refused_tamper.diagnostics.receipt_matches_preparation)
    assert bool(refused_tamper.diagnostics.rejected)
    _tree_exact(refused_tamper.state, state)

    accepted = adapter.adopt_prepared_transition(state, prepared, receipt)
    assert bool(accepted.diagnostics.transaction_applied)
    stale = adapter.adopt_prepared_transition(accepted.state, prepared, receipt)
    assert not bool(stale.diagnostics.source_state_matches)
    assert bool(stale.diagnostics.rejected)
    _tree_exact(stale.state, accepted.state)

    corrupt = state.replace(
        learned_memory_state=state.learned_memory_state.replace(
            admission_weights=state.learned_memory_state.admission_weights.at[0].set(
                jnp.asarray(jnp.nan, dtype=jnp.float32)
            )
        )
    )
    refused_corrupt = adapter.step(
        corrupt,
        transition,
        _event_input(provenance=7),
        jnp.ones((2,), dtype=jnp.bool_),
    )
    assert not bool(refused_corrupt.diagnostics.source_state_valid)
    assert int(refused_corrupt.diagnostics.settlement_evaluations) == 0
    assert int(refused_corrupt.diagnostics.coordinator_evaluations) == 0
    assert int(refused_corrupt.diagnostics.learned_memory_query_evaluations) == 0
    _tree_exact(refused_corrupt.state, corrupt)


def test_monolithic_jit_rejects_before_donor_work() -> None:
    adapter = _adapter()
    state = adapter.start(
        adapter.init(jr.key(19)),
        jnp.asarray((0.2, -0.1), dtype=jnp.float32),
    )
    transition = _transition(state.coordinator_state)
    with pytest.raises(RuntimeError, match="monolithic JIT"):
        jax.jit(adapter.step)(
            state,
            transition,
            _event_input(provenance=8),
            jnp.ones((2,), dtype=jnp.bool_),
        )
    assert not hasattr(adapter, "scan")
