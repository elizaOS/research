"""Contracts for the standalone compositional-representation adapter."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.compositional_feature_adapter import (
    COMPOSITIONAL_FEATURE_ADAPTER_CONFIG_SCHEMA,
    CompositionalFeatureAdapter,
)
from alberta_framework.core.compositional_features import (
    OP_PRODUCT,
    OP_RAW,
    CompositionalFeatureLearner,
)


def _learner(*, replacement_interval: int = 0) -> CompositionalFeatureLearner:
    return CompositionalFeatureLearner(
        n_features=4,
        n_tasks=1,
        candidate_count=1,
        step_size_output=0.0,
        step_size_theta=0.0,
        utility_decay=0.99,
        replacement_interval=replacement_interval,
        min_feature_age=0,
        candidate_min_age=100,
        use_obgd=False,
        train_candidate_theta=False,
    )


def _duplicate_product_state(adapter: CompositionalFeatureAdapter):
    state = adapter.init(jr.key(3))
    learner_state = state.learner_state.replace(
        ops=jnp.asarray((OP_RAW, OP_RAW, OP_PRODUCT, OP_PRODUCT), dtype=jnp.int32),
        parent_a=jnp.asarray((0, 1, 0, 0), dtype=jnp.int32),
        parent_b=jnp.asarray((-1, -1, 1, 1), dtype=jnp.int32),
        theta=jnp.zeros((4, 2), dtype=jnp.float32),
        depth=jnp.asarray((0, 0, 1, 1), dtype=jnp.int32),
    )
    return adapter.rebind_pristine_state(learner_state)


def test_config_requires_frozen_theta_and_a_nonempty_composed_tail() -> None:
    with pytest.raises(ValueError, match="step_size_theta"):
        CompositionalFeatureAdapter(
            CompositionalFeatureLearner(
                n_features=3,
                n_tasks=1,
                candidate_count=0,
                step_size_theta=0.01,
            ),
            base_feature_dim=2,
        )
    with pytest.raises(ValueError, match="train_candidate_theta"):
        CompositionalFeatureAdapter(
            CompositionalFeatureLearner(
                n_features=3,
                n_tasks=1,
                candidate_count=1,
                step_size_theta=0.0,
                train_candidate_theta=True,
            ),
            base_feature_dim=2,
        )
    with pytest.raises(ValueError, match="composed slot"):
        CompositionalFeatureAdapter(
            CompositionalFeatureLearner(
                n_features=2,
                n_tasks=1,
                candidate_count=0,
                step_size_theta=0.0,
            ),
            base_feature_dim=2,
        )


def test_deployed_representation_preserves_exact_base_and_appends_only_tail() -> None:
    adapter = CompositionalFeatureAdapter(_learner(), base_feature_dim=2)
    state = _duplicate_product_state(adapter)
    observation = jnp.asarray((20.0, 2.0), dtype=jnp.float32)

    representation = adapter.representation(state, observation)

    # Raw slots inside the compositional DAG clip to 10, while the deployed
    # stable prefix must remain the exact caller observation. Both products
    # therefore clip to 10 and raw coordinates are not duplicated.
    np.testing.assert_array_equal(
        np.asarray(representation),
        np.asarray((20.0, 2.0, 10.0, 10.0), dtype=np.float32),
    )
    assert bool(adapter.state_valid(state))


def test_duplicate_ast_rows_are_valid_but_keep_distinct_slot_birth_identity() -> None:
    adapter = CompositionalFeatureAdapter(_learner(), base_feature_dim=2)
    state = _duplicate_product_state(adapter)

    np.testing.assert_array_equal(state.binding.ops[2:], (OP_PRODUCT, OP_PRODUCT))
    np.testing.assert_array_equal(state.binding.parent_a[2:], (0, 0))
    np.testing.assert_array_equal(state.binding.parent_b[2:], (1, 1))
    np.testing.assert_array_equal(
        state.binding.slot_birth_words,
        np.zeros((4, 2), dtype=np.uint32),
    )
    # Identity is (slot, birth words, full bank), never descriptor equality.
    assert adapter.dynamic_slot_identity(state, 2) != adapter.dynamic_slot_identity(
        state,
        3,
    )


def test_no_curation_update_keeps_generation_and_advances_learner() -> None:
    adapter = CompositionalFeatureAdapter(_learner(), base_feature_dim=2)
    state = adapter.init(jr.key(4))

    result = adapter.update(
        state,
        jnp.asarray((0.25, -0.5), dtype=jnp.float32),
        jnp.asarray((1.0,), dtype=jnp.float32),
    )

    assert bool(result.diagnostics.transaction_applied)
    assert not bool(result.diagnostics.active_bank_changed)
    np.testing.assert_array_equal(
        result.state.binding.semantic_generation_words,
        state.binding.semantic_generation_words,
    )
    np.testing.assert_array_equal(
        result.state.binding.slot_birth_words,
        state.binding.slot_birth_words,
    )
    np.testing.assert_array_equal(result.state.learner_state.step_words, (0, 1))


def test_structural_update_advances_bank_and_only_changed_slot_births() -> None:
    learner = CompositionalFeatureLearner(
        n_features=4,
        n_tasks=1,
        candidate_count=0,
        step_size_output=0.0,
        step_size_theta=0.0,
        utility_decay=0.999,
        replacement_interval=1,
        min_feature_age=0,
        use_obgd=False,
        train_candidate_theta=False,
    )
    adapter = CompositionalFeatureAdapter(learner, base_feature_dim=2)
    state = _duplicate_product_state(adapter)
    state = adapter.rebind_pristine_state(
        state.learner_state.replace(
            utilities=jnp.asarray((10.0, 10.0, 0.0, 10.0), dtype=jnp.float32),
            ages=jnp.full((4,), 10, dtype=jnp.int32),
        )
    )

    result = adapter.update(
        state,
        jnp.asarray((0.5, 0.25), dtype=jnp.float32),
        jnp.asarray((0.0,), dtype=jnp.float32),
    )

    assert bool(result.diagnostics.transaction_applied)
    assert bool(result.diagnostics.active_bank_changed)
    np.testing.assert_array_equal(result.state.binding.semantic_generation_words, (0, 1))
    change = np.asarray(result.diagnostics.active_change_mask)
    assert change.shape == (4,)
    assert not np.any(change[:2])
    assert np.any(change[2:])
    births = np.asarray(result.state.binding.slot_birth_words)
    np.testing.assert_array_equal(births[change], np.tile((0, 1), (int(change.sum()), 1)))
    np.testing.assert_array_equal(births[~change], np.zeros((int((~change).sum()), 2)))


def test_binding_tamper_fails_closed_before_update() -> None:
    adapter = CompositionalFeatureAdapter(_learner(), base_feature_dim=2)
    state = adapter.init(jr.key(5))
    forged = state.replace(
        binding=state.binding.replace(
            parent_a=state.binding.parent_a.at[2].set(3),
        )
    )

    assert not bool(adapter.state_valid(forged))
    result = adapter.update(
        forged,
        jnp.asarray((0.1, 0.2), dtype=jnp.float32),
        jnp.asarray((0.0,), dtype=jnp.float32),
    )
    assert not bool(result.diagnostics.transaction_applied)
    assert jax.tree.all(
        jax.tree.map(
            lambda left, right: jnp.array_equal(jnp.asarray(left), jnp.asarray(right)),
            result.state,
            forged,
        )
    )


def test_atomic_reencode_authenticates_source_rows_and_one_generation_successor() -> None:
    learner = CompositionalFeatureLearner(
        n_features=4,
        n_tasks=1,
        candidate_count=0,
        step_size_output=0.0,
        step_size_theta=0.0,
        utility_decay=0.999,
        replacement_interval=1,
        min_feature_age=0,
        use_obgd=False,
        train_candidate_theta=False,
    )
    adapter = CompositionalFeatureAdapter(learner, base_feature_dim=2)
    source = _duplicate_product_state(adapter)
    source = adapter.rebind_pristine_state(
        source.learner_state.replace(
            utilities=jnp.asarray((10.0, 10.0, 0.0, 10.0), dtype=jnp.float32),
            ages=jnp.full((4,), 10, dtype=jnp.int32),
        )
    )
    advanced = adapter.update(
        source,
        jnp.asarray((0.5, 0.25), dtype=jnp.float32),
        jnp.asarray((0.0,), dtype=jnp.float32),
    )
    assert bool(advanced.diagnostics.active_bank_changed)
    destination = advanced.state
    base_rows = jnp.asarray(((20.0, 2.0), (-0.0, 3.0), (9.0, 9.0)), dtype=jnp.float32)
    source_rows = jax.vmap(lambda row: adapter.representation(source, row))(base_rows)
    valid_rows = jnp.asarray((True, True, False), dtype=jnp.bool_)
    source_rows = source_rows.at[2].set(
        jnp.asarray((123.0, 456.0, 789.0, 999.0), dtype=jnp.float32)
    )

    rebound = adapter.reencode_rows(source, destination, source_rows, valid_rows)

    assert bool(rebound.diagnostics.transaction_applied)
    assert int(rebound.diagnostics.valid_rows_reencoded) == 2
    expected = jax.vmap(lambda row: adapter.representation(destination, row))(base_rows)
    np.testing.assert_array_equal(np.asarray(rebound.values[:2]), np.asarray(expected[:2]))
    np.testing.assert_array_equal(np.asarray(rebound.values[2]), np.asarray(source_rows[2]))

    corrupted = source_rows.at[0, 2].add(jnp.float32(1.0))
    rejected = adapter.reencode_rows(source, destination, corrupted, valid_rows)
    assert not bool(rejected.diagnostics.transaction_applied)
    np.testing.assert_array_equal(np.asarray(rejected.values), np.asarray(corrupted))

    skipped_step = destination.replace(
        learner_state=destination.learner_state.replace(
            step_count=jnp.asarray(2, dtype=jnp.int32),
            step_words=jnp.asarray((0, 2), dtype=jnp.uint32),
        )
    )
    assert bool(adapter.state_valid(skipped_step))
    rejected_step = adapter.reencode_rows(
        source,
        skipped_step,
        source_rows,
        valid_rows,
    )
    assert not bool(rejected_step.diagnostics.learner_step_is_successor)
    assert not bool(rejected_step.diagnostics.transaction_applied)
    np.testing.assert_array_equal(
        np.asarray(rejected_step.values),
        np.asarray(source_rows),
    )


def test_config_roundtrip_is_exact_and_resource_budget_matches_tree() -> None:
    adapter = CompositionalFeatureAdapter(_learner(), base_feature_dim=2)
    payload = adapter.to_config()
    assert payload["schema"] == COMPOSITIONAL_FEATURE_ADAPTER_CONFIG_SCHEMA
    restored = CompositionalFeatureAdapter.from_config(payload)
    assert restored.to_config() == payload

    missing = dict(payload)
    missing_learner = dict(missing["learner"])
    missing_learner.pop("ancestor_utility_backup_decay")
    missing["learner"] = missing_learner
    with pytest.raises(ValueError, match="learner configuration"):
        CompositionalFeatureAdapter.from_config(missing)

    state = adapter.init(jr.key(9))
    budget = adapter.resource_budget(state, reencode_capacity=7)
    measured = adapter.measure_state_nbytes(state)
    assert budget.total_persistent_state_nbytes == measured
    assert budget.binding_persistent_nbytes == 32 * adapter.n_features + 44
    assert budget.max_reencode_feature_slot_evaluations == 2 * 7 * adapter.n_features
    assert budget.max_reencode_rows == 7


def test_schema_digest_or_birth_history_tamper_is_rejected() -> None:
    adapter = CompositionalFeatureAdapter(_learner(), base_feature_dim=2)
    state = adapter.init(jr.key(10))
    bad_digest = state.replace(
        binding=state.binding.replace(
            schema_digest=state.binding.schema_digest.at[0].set(jnp.uint8(255)),
        )
    )
    assert not bool(adapter.state_valid(bad_digest))

    future_birth = state.replace(
        binding=state.binding.replace(
            slot_birth_words=state.binding.slot_birth_words.at[2].set(
                jnp.asarray((0, 1), dtype=jnp.uint32)
            )
        )
    )
    assert not bool(adapter.state_valid(future_birth))

    skipped_generation = state.replace(
        binding=state.binding.replace(
            semantic_generation=jnp.asarray(1, dtype=jnp.int32),
            semantic_generation_words=jnp.asarray((0, 1), dtype=jnp.uint32),
        )
    )
    assert not bool(adapter.state_valid(skipped_generation))


def test_fixed_curation_phase_is_bound_to_exact_learner_lifetime() -> None:
    adapter = CompositionalFeatureAdapter(_learner(replacement_interval=3), base_feature_dim=2)
    state = adapter.init(jr.key(10))
    forged = state.replace(
        learner_state=state.learner_state.replace(
            replacement_phase=jnp.asarray(1, dtype=jnp.int32),
        )
    )
    assert not bool(adapter.state_valid(forged))


def test_resource_budget_rejects_boolean_capacity() -> None:
    adapter = CompositionalFeatureAdapter(_learner(), base_feature_dim=2)
    with pytest.raises(ValueError, match="reencode_capacity"):
        adapter.resource_budget(adapter.init(jr.key(11)), reencode_capacity=True)


def test_terminal_counters_are_not_reported_as_successors_and_noop_is_explicit() -> None:
    adapter = CompositionalFeatureAdapter(_learner(), base_feature_dim=2)
    state = adapter.init(jr.key(12))
    terminal_words = jnp.full((2,), np.iinfo(np.uint32).max, dtype=jnp.uint32)
    terminal_births = state.binding.slot_birth_words.at[2:].set(terminal_words)
    terminal = state.replace(
        learner_state=state.learner_state.replace(
            step_count=jnp.asarray(np.iinfo(np.int32).max, dtype=jnp.int32),
            step_words=terminal_words,
        ),
        binding=state.binding.replace(
            semantic_generation=jnp.asarray(
                np.iinfo(np.int32).max,
                dtype=jnp.int32,
            ),
            semantic_generation_words=terminal_words,
            slot_birth_words=terminal_births,
        ),
    )
    assert bool(adapter.state_valid(terminal))
    base_rows = jnp.asarray(((0.25, -0.5),), dtype=jnp.float32)
    rows = jax.vmap(lambda row: adapter.representation(terminal, row))(base_rows)

    result = adapter.reencode_rows(
        terminal,
        terminal,
        rows,
        jnp.asarray((True,), dtype=jnp.bool_),
    )

    assert not bool(result.diagnostics.generation_is_successor)
    assert not bool(result.diagnostics.learner_step_is_successor)
    assert not bool(result.diagnostics.transaction_applied)
    assert bool(result.diagnostics.transaction_noop)
    np.testing.assert_array_equal(np.asarray(result.values), np.asarray(rows))


def test_binding_dataclasses_are_frozen() -> None:
    adapter = CompositionalFeatureAdapter(_learner(), base_feature_dim=2)
    state = adapter.init(jr.key(12))
    with pytest.raises(dataclasses.FrozenInstanceError):
        state.binding.semantic_generation = jnp.int32(3)  # type: ignore[misc]


def test_prepared_update_defers_adoption_until_consumer_authorization() -> None:
    adapter = CompositionalFeatureAdapter(_learner(), base_feature_dim=2)
    source = adapter.init(jr.key(13))
    observation = jnp.asarray((0.25, -0.5), dtype=jnp.float32)
    targets = jnp.asarray((1.0,), dtype=jnp.float32)

    proposal = adapter.prepare_update(source, observation, targets)

    np.testing.assert_array_equal(source.learner_state.step_words, (0, 0))
    np.testing.assert_array_equal(proposal.source_state.learner_state.step_words, (0, 0))
    np.testing.assert_array_equal(proposal.candidate_state.learner_state.step_words, (0, 1))
    assert bool(proposal.diagnostics.transaction_applied)

    deferred = adapter.commit_prepared_update(
        source,
        proposal,
        consumers_ready=jnp.asarray(False, dtype=jnp.bool_),
    )
    assert bool(deferred.diagnostics.proposal_integrity)
    assert bool(deferred.diagnostics.source_matches)
    assert bool(deferred.diagnostics.proposal_valid)
    assert not bool(deferred.diagnostics.consumers_ready)
    assert not bool(deferred.diagnostics.applied)
    assert bool(deferred.diagnostics.rejected)
    assert jax.tree.all(
        jax.tree.map(
            lambda left, right: jnp.array_equal(jnp.asarray(left), jnp.asarray(right)),
            deferred.state,
            source,
        )
    )

    committed = adapter.commit_prepared_update(
        source,
        proposal,
        consumers_ready=jnp.asarray(True, dtype=jnp.bool_),
    )
    assert bool(committed.diagnostics.applied)
    assert not bool(committed.diagnostics.rejected)
    assert int(committed.diagnostics.preparation_learner_update_evaluations) == 1
    assert int(committed.diagnostics.commit_recomputed_learner_update_evaluations) == 1
    assert int(committed.diagnostics.total_learner_update_evaluations) == 2
    assert jax.tree.all(
        jax.tree.map(
            lambda left, right: jnp.array_equal(jnp.asarray(left), jnp.asarray(right)),
            committed.state,
            proposal.candidate_state,
        )
    )

    stale_retry = adapter.commit_prepared_update(
        committed.state,
        proposal,
        consumers_ready=jnp.asarray(True, dtype=jnp.bool_),
    )
    assert bool(stale_retry.diagnostics.proposal_integrity)
    assert not bool(stale_retry.diagnostics.source_matches)
    assert not bool(stale_retry.diagnostics.applied)
    assert jax.tree.all(
        jax.tree.map(
            lambda left, right: jnp.array_equal(jnp.asarray(left), jnp.asarray(right)),
            stale_retry.state,
            committed.state,
        )
    )


def test_prepared_update_recomputes_identity_and_rejects_tampering_atomically() -> None:
    adapter = CompositionalFeatureAdapter(_learner(), base_feature_dim=2)
    source = adapter.init(jr.key(14))
    proposal = adapter.prepare_update(
        source,
        jnp.asarray((0.25, -0.5), dtype=jnp.float32),
        jnp.asarray((1.0,), dtype=jnp.float32),
    )
    tampered_prediction = proposal.replace(
        predictions=proposal.predictions.at[0].add(jnp.float32(1.0)),
    )
    rejected_prediction = adapter.commit_prepared_update(
        source,
        tampered_prediction,
        consumers_ready=jnp.asarray(True, dtype=jnp.bool_),
    )
    assert not bool(rejected_prediction.diagnostics.proposal_integrity)
    assert not bool(rejected_prediction.diagnostics.applied)

    tampered_state = proposal.replace(
        candidate_state=proposal.candidate_state.replace(
            learner_state=proposal.candidate_state.learner_state.replace(
                output_weights=(
                    proposal.candidate_state.learner_state.output_weights.at[0, 0].add(
                        jnp.float32(1.0)
                    )
                ),
            )
        )
    )
    rejected_state = adapter.commit_prepared_update(
        source,
        tampered_state,
        consumers_ready=jnp.asarray(True, dtype=jnp.bool_),
    )
    assert not bool(rejected_state.diagnostics.proposal_integrity)
    assert not bool(rejected_state.diagnostics.applied)
    assert jax.tree.all(
        jax.tree.map(
            lambda left, right: jnp.array_equal(jnp.asarray(left), jnp.asarray(right)),
            rejected_state.state,
            source,
        )
    )


def test_prepared_update_commit_is_outer_jit_and_scan_safe() -> None:
    adapter = CompositionalFeatureAdapter(_learner(), base_feature_dim=2)
    source = adapter.init(jr.key(15))
    observation = jnp.asarray((0.25, -0.5), dtype=jnp.float32)
    targets = jnp.asarray((1.0,), dtype=jnp.float32)
    proposal = jax.jit(adapter.prepare_update)(source, observation, targets)
    eager = adapter.commit_prepared_update(
        source,
        proposal,
        consumers_ready=jnp.asarray(True, dtype=jnp.bool_),
    )
    compiled = jax.jit(adapter.commit_prepared_update)(
        source,
        proposal,
        consumers_ready=jnp.asarray(True, dtype=jnp.bool_),
    )
    assert jax.tree.all(
        jax.tree.map(
            lambda left, right: jnp.array_equal(jnp.asarray(left), jnp.asarray(right)),
            eager,
            compiled,
        )
    )

    def step(state, ready):
        prepared = adapter.prepare_update(state, observation, targets)
        result = adapter.commit_prepared_update(
            state,
            prepared,
            consumers_ready=ready,
        )
        return result.state, result.diagnostics.applied

    final, applied = jax.jit(lambda state: jax.lax.scan(
        step,
        state,
        jnp.asarray((True, False, True), dtype=jnp.bool_),
    ))(source)
    np.testing.assert_array_equal(applied, (True, False, True))
    np.testing.assert_array_equal(final.learner_state.step_words, (0, 2))


def test_prepared_structural_update_waits_for_atomic_row_reencoding() -> None:
    learner = CompositionalFeatureLearner(
        n_features=4,
        n_tasks=1,
        candidate_count=0,
        step_size_output=0.0,
        step_size_theta=0.0,
        utility_decay=0.999,
        replacement_interval=1,
        min_feature_age=0,
        use_obgd=False,
        train_candidate_theta=False,
    )
    adapter = CompositionalFeatureAdapter(learner, base_feature_dim=2)
    source = _duplicate_product_state(adapter)
    source = adapter.rebind_pristine_state(
        source.learner_state.replace(
            utilities=jnp.asarray((10.0, 10.0, 0.0, 10.0), dtype=jnp.float32),
            ages=jnp.full((4,), 10, dtype=jnp.int32),
        )
    )
    proposal = adapter.prepare_update(
        source,
        jnp.asarray((0.5, 0.25), dtype=jnp.float32),
        jnp.asarray((0.0,), dtype=jnp.float32),
    )
    assert bool(proposal.diagnostics.active_bank_changed)

    base_rows = jnp.asarray(((0.5, 0.25), (-0.5, 0.75)), dtype=jnp.float32)
    rows = jax.vmap(lambda row: adapter.representation(source, row))(base_rows)
    rebound = adapter.reencode_rows(
        source,
        proposal.candidate_state,
        rows,
        jnp.asarray((True, True), dtype=jnp.bool_),
    )
    assert bool(rebound.diagnostics.transaction_applied)
    committed = adapter.commit_prepared_update(
        source,
        proposal,
        consumers_ready=rebound.diagnostics.transaction_applied,
    )
    assert bool(committed.diagnostics.applied)
    expected_rows = jax.vmap(
        lambda row: adapter.representation(committed.state, row)
    )(base_rows)
    np.testing.assert_array_equal(rebound.values, expected_rows)

    corrupt_rows = rows.at[0, 2].add(jnp.float32(1.0))
    refused = adapter.reencode_rows(
        source,
        proposal.candidate_state,
        corrupt_rows,
        jnp.asarray((True, True), dtype=jnp.bool_),
    )
    assert not bool(refused.diagnostics.transaction_applied)
    deferred = adapter.commit_prepared_update(
        source,
        proposal,
        consumers_ready=refused.diagnostics.transaction_applied,
    )
    assert not bool(deferred.diagnostics.applied)
    assert jax.tree.all(
        jax.tree.map(
            lambda left, right: jnp.array_equal(jnp.asarray(left), jnp.asarray(right)),
            deferred.state,
            source,
        )
    )


def test_prepared_update_contract_and_transient_bytes_are_strict() -> None:
    adapter = CompositionalFeatureAdapter(_learner(), base_feature_dim=2)
    source = adapter.init(jr.key(16))
    observation = jnp.asarray((0.25, -0.5), dtype=jnp.float32)
    targets = jnp.asarray((1.0,), dtype=jnp.float32)
    proposal = adapter.prepare_update(source, observation, targets)

    proposal_nbytes = adapter.measure_prepared_update_nbytes(proposal)
    assert proposal_nbytes > 2 * adapter.measure_state_nbytes(source)
    compiled_proposal = jax.jit(adapter.prepare_update)(source, observation, targets)
    assert adapter.measure_prepared_update_nbytes(compiled_proposal) == proposal_nbytes
    compiled_source = compiled_proposal.source_state
    assert adapter.measure_state_nbytes(compiled_source) == adapter.measure_state_nbytes(
        source
    )
    with pytest.raises(ValueError, match="context_id"):
        adapter.prepare_update(
            source,
            observation,
            targets,
            context_id=jnp.asarray((0,), dtype=jnp.int32),
        )
    with pytest.raises(ValueError, match="consumers_ready"):
        adapter.commit_prepared_update(
            source,
            proposal,
            consumers_ready=True,
        )
