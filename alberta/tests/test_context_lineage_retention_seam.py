# mypy: disable-error-code="attr-defined,call-arg"
"""Contracts for the one-agent context/lineage retention seam."""

from __future__ import annotations

import dataclasses
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from alberta_framework.core.context_inference import ContextInferenceConfig
from alberta_framework.core.context_lineage_retention_seam import (
    CONTEXT_LINEAGE_RETENTION_CONFIRMATION_HORIZON,
    ContextLineageRetentionSeam,
    ContextLineageRetentionSeamConfig,
    ContextLineageRetentionSeamState,
    ContextLineageRetentionStepResult,
)

pytestmark = pytest.mark.integration


CONTEXT_CONFIG = ContextInferenceConfig(
    n_actions=2,
    observation_dim=2,
    max_contexts=3,
    model_step_size=1.0,
    error_decay=0.0,
    switch_threshold=0.75,
    novelty_prior_error=0.5,
    update_error_gate=0.75,
    min_dwell=0,
    initial_reward_estimate=0.5,
)
CONFIG = ContextLineageRetentionSeamConfig(context=CONTEXT_CONFIG)

# Four observable state-action cells.  Every newly introduced rule has a
# diagnostic cell on which all preceding rules disagree, while R0's recurrence
# has one cell on which every live replacement disagrees.
CELLS = ((0, 0), (0, 1), (1, 0), (1, 1))
R0 = (0.0, 0.0, 0.0, 0.0)
R1 = (1.0, 1.0, 0.0, 0.0)
R2 = (1.0, 0.0, 1.0, 0.0)
R3 = (1.0, 0.0, 0.0, 1.0)
R4 = (1.0, 1.0, 1.0, 1.0)


def _obs(index: int) -> jax.Array:
    return jax.nn.one_hot(CELLS[index][0], 2, dtype=jnp.float32)


def _tree_exact(left: object, right: object) -> bool:
    left_leaves, left_tree = jax.tree.flatten(left)
    right_leaves, right_tree = jax.tree.flatten(right)
    if cast(object, left_tree) != cast(object, right_tree) or len(left_leaves) != len(
        right_leaves
    ):
        return False
    return all(
        np.asarray(a).dtype == np.asarray(b).dtype
        and np.asarray(a).shape == np.asarray(b).shape
        and np.asarray(a).tobytes() == np.asarray(b).tobytes()
        for a, b in zip(left_leaves, right_leaves, strict=True)
    )


def _step(
    seam: ContextLineageRetentionSeam,
    state: ContextLineageRetentionSeamState,
    cell: int,
    reward: float,
) -> ContextLineageRetentionStepResult:
    preparation = seam.prepare(
        state,
        _obs(cell),
        jnp.asarray(CELLS[cell][1], dtype=jnp.int32),
    )
    result = seam.step(state, preparation, jnp.asarray(reward, dtype=jnp.float32))
    assert bool(result.update_applied)
    assert bool(result.context_owner_committed)
    assert bool(result.lineage_owner_committed)
    return result


def _train_rule(
    seam: ContextLineageRetentionSeam,
    state: ContextLineageRetentionSeamState,
    rule: tuple[float, float, float, float],
    order: tuple[int, int, int, int],
) -> tuple[ContextLineageRetentionSeamState, list[object]]:
    results: list[object] = []
    for cell in order:
        result = _step(seam, state, cell, rule[cell])
        state = result.state
        results.append(result)
    return state, results


def _through_first_full_birth(
    seam: ContextLineageRetentionSeam,
) -> tuple[ContextLineageRetentionSeamState, object]:
    state = seam.init()
    state, _ = _train_rule(seam, state, R0, (0, 1, 2, 3))
    state, _ = _train_rule(seam, state, R1, (0, 1, 2, 3))
    state, _ = _train_rule(seam, state, R2, (2, 0, 1, 3))
    state, r3_results = _train_rule(seam, state, R3, (3, 0, 1, 2))
    return state, r3_results[0]


def _open_r0_recurrence(
    seam: ContextLineageRetentionSeam,
) -> tuple[ContextLineageRetentionSeamState, object]:
    state, _ = _through_first_full_birth(seam)
    opening = _step(seam, state, 0, R0[0])
    return opening.state, opening


def _confirmed_r0_recurrence(
    seam: ContextLineageRetentionSeam,
) -> tuple[ContextLineageRetentionSeamState, object]:
    state, opening = _open_r0_recurrence(seam)
    assert bool(opening.prospective_quarantine_opened)
    confirmation = _step(seam, state, 0, R0[0])
    assert bool(confirmation.prospective_quarantine_confirmed)
    return confirmation.state, confirmation


def test_genesis_coordinates_config_and_exact_receipts() -> None:
    seam = ContextLineageRetentionSeam(CONFIG)
    state = seam.init()

    assert bool(seam.state_is_valid(state))
    assert int(state.context.active_context) == 0
    np.testing.assert_array_equal(state.context.in_use, [True, False, False])
    np.testing.assert_array_equal(state.slot_birth_words, np.zeros((3, 2), dtype=np.uint32))
    np.testing.assert_array_equal(state.lineage.bound_birth_words, state.slot_birth_words)
    np.testing.assert_array_equal(seam.context_coordinates(state), [1.0, 0.0, 0.0])
    assert CONTEXT_LINEAGE_RETENTION_CONFIRMATION_HORIZON == 2

    restored = ContextLineageRetentionSeam.from_config(seam.to_config())
    assert restored.to_config() == seam.to_config()
    assert restored.config == CONFIG
    alias = seam.to_config()
    alias["context_state_owners"] = True
    with pytest.raises(ValueError, match="not canonical"):
        ContextLineageRetentionSeam.from_config(alias)

    resources = seam.resource_record(state)
    assert resources.context_state_owners == 1
    assert resources.sequential_lineage_state_owners == 1
    assert resources.context_coordinate_dim == 3
    assert resources.context_state_nbytes == 127
    assert resources.birth_ledger_nbytes == 24
    assert resources.lineage_state_nbytes == 323
    assert resources.composite_integrity_nbytes == 64
    assert resources.total_persistent_state_nbytes == 538
    assert resources.measured_total_persistent_state_nbytes == 538
    assert resources.preparation_binding_nbytes == 201
    assert resources.logical_atomic_candidate_nbytes == 538
    assert resources.replay_capacity == 0
    assert resources.persistent_capacity_growth == 0
    assert resources.composite_jit_supported is False
    assert resources.preoutcome_call_order_authenticated is False
    assert resources.outcome_provenance_claimed is False

    work = seam.work_record(total_steps=9)
    assert work.pre_outcome_protection_snapshots == 9
    assert work.protection_binding_recomputations == 9
    assert work.context_update_proposals == 9
    assert work.birth_ledger_proposals == 9
    assert work.sequential_lineage_proposals == 9
    assert work.outer_commit_decisions == 9
    assert work.ordinal_rescue_word_comparisons == 162
    assert work.composite_state_audits == 36
    assert work.context_state_audits == 54
    assert work.birth_ledger_audits == 36
    assert work.outer_lineage_binding_audits == 36
    assert work.outer_lineage_content_digest_evaluations == 36
    assert work.composite_state_integrity_evaluations == 45
    assert work.preparation_integrity_evaluations == 27
    assert work.context_scalar_reward_predictions == 27
    assert work.context_reward_prediction_coefficient_products == 54
    assert work.lineage.transaction_proposals == 9
    assert work.replay_updates == work.random_draws == work.reset_callbacks == 0
    assert work.exact_named_logical_counts is True
    assert work.exhaustive_primitive_operation_count_claimed is False


@pytest.mark.parametrize(
    ("field", "alias"),
    tuple(
        (field, alias)
        for field, canonical in (
            ("model_step_size", 1.0),
            ("error_decay", 0.0),
            ("switch_threshold", 1.0),
            ("novelty_prior_error", 1.0),
            ("update_error_gate", 1.0),
            ("initial_reward_estimate", 0.0),
        )
        for alias in (bool(canonical), int(canonical), np.float32(canonical))
    ),
)
def test_nested_context_float_fields_reject_python_and_numpy_aliases(
    field: str,
    alias: object,
) -> None:
    values: dict[str, Any] = {
        "n_actions": 2,
        "observation_dim": 2,
        "max_contexts": 3,
        "model_step_size": 1.0,
        "error_decay": 0.0,
        "switch_threshold": 0.75,
        "novelty_prior_error": 0.5,
        "update_error_gate": 0.75,
        "min_dwell": 0,
        "initial_reward_estimate": 0.5,
    }
    values[field] = alias
    if field == "novelty_prior_error" and float(cast(Any, alias)) >= 0.75:
        values["update_error_gate"] = 2.0
    nested = ContextInferenceConfig(**values)

    with pytest.raises(TypeError, match=rf"context\.{field} must be an exact float"):
        ContextLineageRetentionSeamConfig(context=nested)


def test_online_births_fill_the_bank_then_archive_the_ordinary_victim() -> None:
    seam = ContextLineageRetentionSeam(CONFIG)
    state, first_full_birth = _through_first_full_birth(seam)

    assert bool(seam.state_is_valid(state))
    assert int(state.context.in_use.sum()) == 3
    assert len({tuple(row) for row in np.asarray(state.slot_birth_words).tolist()}) == 3
    assert bool(first_full_birth.context_allocation_requested)
    assert bool(first_full_birth.context_full_bank_eviction_requested)
    assert int(first_full_birth.context_ordinary_lru_slot) == 0
    assert int(first_full_birth.context_selected_eviction_slot) == 0
    assert bool(first_full_birth.lineage_full_bank_birth)
    assert bool(first_full_birth.lineage_archive_current_victim_selected)
    assert not bool(first_full_birth.prospective_cache_tested)
    assert bool(state.lineage.archive.valid)


def test_h2_recurrence_is_confirmed_only_after_second_post_outcome_event() -> None:
    seam = ContextLineageRetentionSeam(CONFIG)
    opening_state, opening = _open_r0_recurrence(seam)

    target = int(opening.state.context.active_context)
    target_birth = np.asarray(opening.state.slot_birth_words[target]).copy()
    assert bool(opening.context_full_bank_eviction_requested)
    assert bool(opening.prospective_cache_tested)
    assert bool(opening.prospective_quarantine_opened)
    assert not bool(opening.prospective_quarantine_confirmed)
    assert not bool(opening.prospective_quarantine_rejected)
    assert not bool(opening.lineage_transferred)
    assert bool(opening.state.lineage.pending.valid)
    assert bool(opening.protection_snapshotted_before_outcome)
    assert not bool(opening.current_outcome_changed_current_eviction_protection)

    confirmation = _step(seam, opening_state, 0, R0[0])
    assert bool(confirmation.prospective_second_evidence)
    assert bool(confirmation.prospective_quarantine_confirmed)
    assert not bool(confirmation.prospective_quarantine_rejected)
    assert bool(confirmation.lineage_transferred)
    assert bool(confirmation.rescue_incremented)
    assert not bool(confirmation.state.lineage.pending.valid)
    np.testing.assert_array_equal(
        confirmation.state.lineage.live_lineage_words[target],
        np.zeros((2,), dtype=np.uint32),
    )
    np.testing.assert_array_equal(
        confirmation.state.lineage.live_rescue_words[target],
        np.asarray((0, 1), dtype=np.uint32),
    )
    assert not np.array_equal(
        confirmation.state.lineage.live_lineage_words[target], target_birth
    )


def test_contradictory_second_event_rejects_without_lineage_transfer() -> None:
    seam = ContextLineageRetentionSeam(CONFIG)
    opening_state, opening = _open_r0_recurrence(seam)
    target = int(opening.state.context.active_context)
    fresh_lineage = np.asarray(opening.state.lineage.live_lineage_words[target]).copy()

    rejection = _step(seam, opening_state, 1, 1.0)

    assert bool(rejection.prospective_second_evidence)
    assert not bool(rejection.prospective_quarantine_confirmed)
    assert bool(rejection.prospective_quarantine_rejected)
    assert not bool(rejection.lineage_transferred)
    assert not bool(rejection.rescue_incremented)
    assert not bool(rejection.state.lineage.pending.valid)
    np.testing.assert_array_equal(
        rejection.state.lineage.live_lineage_words[target], fresh_lineage
    )
    np.testing.assert_array_equal(
        rejection.state.lineage.live_rescue_words[target],
        np.zeros((2,), dtype=np.uint32),
    )


def test_confirmed_rescue_protects_an_inactive_ordinary_lru_victim() -> None:
    seam = ContextLineageRetentionSeam(CONFIG)
    state, _ = _confirmed_r0_recurrence(seam)

    # Finish the recurrent R0 model, then revisit R2 and R3.  This makes the
    # confirmed R0 lineage the ordinary LRU while it remains inactive.
    for cell in (1, 2, 3):
        state = _step(seam, state, cell, R0[cell]).state
    state = _step(seam, state, 2, R2[2]).state
    state = _step(seam, state, 3, R3[3]).state

    preparation = seam.prepare(
        state,
        _obs(1),
        jnp.asarray(CELLS[1][1], dtype=jnp.int32),
    )
    protected_slot = int(jnp.argmax(preparation.eviction_protection))
    assert protected_slot == 1
    assert float(preparation.eviction_protection[protected_slot]) > 0.0
    result = seam.step(state, preparation, jnp.asarray(R4[1], dtype=jnp.float32))

    assert bool(result.update_applied)
    assert bool(result.context_full_bank_eviction_requested)
    assert bool(result.context_eviction_protection_used)
    assert bool(result.context_eviction_target_adjusted)
    assert int(result.context_ordinary_lru_slot) == protected_slot
    assert int(result.context_selected_eviction_slot) == 2
    assert int(result.state.context.active_context) == 2
    np.testing.assert_array_equal(
        result.state.lineage.live_lineage_words[protected_slot],
        state.lineage.live_lineage_words[protected_slot],
    )
    np.testing.assert_array_equal(
        result.state.lineage.live_rescue_words[protected_slot],
        state.lineage.live_rescue_words[protected_slot],
    )


def test_stale_or_tampered_preparation_rolls_back_both_owners_exactly() -> None:
    seam = ContextLineageRetentionSeam(CONFIG)
    source = seam.init()
    stale = seam.prepare(source, _obs(0), jnp.asarray(0, dtype=jnp.int32))
    advanced = _step(seam, source, 0, R0[0]).state

    stale_result = seam.step(
        advanced,
        stale,
        jnp.asarray(R0[0], dtype=jnp.float32),
    )
    assert not bool(stale_result.preparation_matches_source)
    assert not bool(stale_result.update_applied)
    assert not bool(stale_result.context_owner_committed)
    assert not bool(stale_result.lineage_owner_committed)
    assert _tree_exact(stale_result.state, advanced)
    assert _tree_exact(stale_result.state.context, advanced.context)
    assert _tree_exact(stale_result.state.lineage, advanced.lineage)

    current = seam.prepare(advanced, _obs(1), jnp.asarray(1, dtype=jnp.int32))
    tampered = current.replace(
        eviction_protection=current.eviction_protection.at[0].set(2.0)
    )
    tampered_result = seam.step(
        advanced,
        tampered,
        jnp.asarray(0.0, dtype=jnp.float32),
    )
    assert not bool(tampered_result.preparation_integrity_valid)
    assert not bool(tampered_result.update_applied)
    assert _tree_exact(tampered_result.state, advanced)


def test_tampered_state_and_invalid_outcome_fail_closed_without_partial_progress() -> None:
    seam = ContextLineageRetentionSeam(CONFIG)
    source = _step(seam, seam.init(), 0, R0[0]).state
    tampered_context = source.context.replace(
        reward_weights=source.context.reward_weights.at[0, 0, 0].set(0.25)
    )
    tampered_state = source.replace(context=tampered_context)
    preparation = seam.prepare(
        tampered_state,
        _obs(1),
        jnp.asarray(1, dtype=jnp.int32),
    )
    result = seam.step(
        tampered_state,
        preparation,
        jnp.asarray(0.0, dtype=jnp.float32),
    )
    assert not bool(preparation.source_state_valid)
    assert not bool(result.source_state_valid)
    assert not bool(result.update_applied)
    assert _tree_exact(result.state, tampered_state)
    assert _tree_exact(result.state.context, tampered_state.context)
    assert _tree_exact(result.state.lineage, tampered_state.lineage)

    clean_preparation = seam.prepare(source, _obs(1), jnp.asarray(1, dtype=jnp.int32))
    invalid = seam.step(source, clean_preparation, jnp.asarray(jnp.nan, dtype=jnp.float32))
    assert not bool(invalid.context_update_applied)
    assert not bool(invalid.update_applied)
    assert _tree_exact(invalid.state, source)


def test_surface_contains_no_phase_reset_replay_or_rng_authority() -> None:
    seam = ContextLineageRetentionSeam(CONFIG)
    field_names = {
        field.name
        for value in (seam.init(), seam.prepare(seam.init(), _obs(0), jnp.asarray(0, jnp.int32)))
        for field in dataclasses.fields(cast(Any, value))
    }
    assert all("phase" not in name for name in field_names)
    assert all("schedule" not in name for name in field_names)
    assert all("reset" not in name for name in field_names)
    assert all("replay" not in name for name in field_names)
    assert all("rng" not in name and "key" not in name for name in field_names)

    work = seam.work_record(total_steps=1_000)
    resources = seam.resource_record()
    assert work.replay_updates == 0
    assert work.random_draws == 0
    assert work.reset_callbacks == 0
    assert work.lineage.replay_updates == 0
    assert work.lineage.random_draws == 0
    assert work.lineage.reset_callbacks == 0
    assert resources.replay_capacity == 0
