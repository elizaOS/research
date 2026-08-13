"""Fast contracts for bounded prior-conditioned prospective retention."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import chex
import jax
import jax.numpy as jnp
import pytest

from alberta_framework.core.checkpoints import save_checkpoint
from alberta_framework.core.prospective_lineage_retention import (
    PROSPECTIVE_LINEAGE_RETENTION_ARCHIVE_CAPACITY,
    PROSPECTIVE_LINEAGE_RETENTION_CHECKPOINT_SCHEMA,
    PROSPECTIVE_LINEAGE_RETENTION_EVIDENCE_LEVEL,
    PROSPECTIVE_LINEAGE_RETENTION_EXTERNAL_PROVENANCE_CLAIMED,
    PROSPECTIVE_LINEAGE_RETENTION_HOST_EVENT_BINDING_CLAIMED,
    PROSPECTIVE_LINEAGE_RETENTION_REPLAY_CAPACITY,
    PROSPECTIVE_LINEAGE_RETENTION_SCIENTIFIC_PROMOTION_ALLOWED,
    PROSPECTIVE_LINEAGE_RETENTION_STATUS,
    ProspectiveLineageRetention,
    ProspectiveLineageRetentionConfig,
    ProspectiveLineageRetentionEvent,
    ProspectiveLineageRetentionPreparation,
    ProspectiveLineageRetentionState,
    load_prospective_lineage_retention_checkpoint,
    measure_prospective_lineage_retention_state_nbytes,
    save_prospective_lineage_retention_checkpoint,
)

pytestmark = pytest.mark.unit

CONFIG = ProspectiveLineageRetentionConfig(
    max_contexts=3,
    initial_reacquisition_cost=1.0,
    cost_ema_decay=0.5,
    max_abs_cost=10.0,
    minimax_guardrail_slack=0.0,
)


def _words(value: int) -> jax.Array:
    return jnp.asarray((0, value), dtype=jnp.uint32)


def _births(*values: int) -> jax.Array:
    return jnp.stack(tuple(_words(value) for value in values)).astype(jnp.uint32)


def _next_words(words: jax.Array) -> jax.Array:
    value = (int(words[0]) << 32) | int(words[1])
    advanced = value + 1
    return jnp.asarray(
        ((advanced >> 32) & (2**32 - 1), advanced & (2**32 - 1)),
        dtype=jnp.uint32,
    )


def _mechanism_and_state(
    priors: tuple[float, float, float] = (0.99, 0.75, 0.25),
) -> tuple[ProspectiveLineageRetention, ProspectiveLineageRetentionState]:
    mechanism = ProspectiveLineageRetention(CONFIG)
    state = mechanism.init(
        step_words=_words(10),
        birth_words=_births(1, 2, 3),
        in_use=jnp.ones((3,), dtype=jnp.bool_),
        return_priors=jnp.asarray(priors, dtype=jnp.float32),
    )
    return mechanism, state


def _prepare(
    mechanism: ProspectiveLineageRetention,
    state: ProspectiveLineageRetentionState,
    *,
    routed: bool,
    active_slot: int = 0,
    recency: tuple[int, int, int] = (9, 5, 7),
) -> ProspectiveLineageRetentionPreparation:
    return mechanism.prepare(
        state,
        source_birth_words=state.bound_birth_words[:3],
        source_in_use=state.live_in_use,
        active_slot=jnp.asarray(active_slot, dtype=jnp.int32),
        last_active_words=_births(*recency),
        route_protection=jnp.asarray(routed, dtype=jnp.bool_),
    )


def _allocation_event(
    state: ProspectiveLineageRetentionState,
    *,
    target_slot: int = 2,
    new_birth: int = 11,
    newborn_prior: float = 0.4,
) -> ProspectiveLineageRetentionEvent:
    post_births = state.bound_birth_words[:3].at[target_slot].set(_words(new_birth))
    return ProspectiveLineageRetentionEvent(  # type: ignore[call-arg]
        source_step_words=state.step_words,
        post_step_words=_next_words(state.step_words),
        source_birth_words=state.bound_birth_words[:3],
        post_birth_words=post_births,
        source_in_use=state.live_in_use,
        post_in_use=state.live_in_use,
        allocated=jnp.asarray(True, dtype=jnp.bool_),
        evicted=jnp.asarray(True, dtype=jnp.bool_),
        target_slot=jnp.asarray(target_slot, dtype=jnp.int32),
        newborn_return_prior=jnp.asarray(newborn_prior, dtype=jnp.float32),
        lineage_transfer_confirmed=jnp.asarray(False, dtype=jnp.bool_),
        transfer_slot=jnp.asarray(-1, dtype=jnp.int32),
        transferred_lineage_words=jnp.zeros((2,), dtype=jnp.uint32),
        archived_loss=jnp.asarray(0.0, dtype=jnp.float32),
        fresh_loss=jnp.asarray(0.0, dtype=jnp.float32),
        context_update_applied=jnp.asarray(True, dtype=jnp.bool_),
    )


def _free_allocation_event(
    state: ProspectiveLineageRetentionState,
    *,
    target_slot: int,
    new_birth: int,
    newborn_prior: float,
) -> ProspectiveLineageRetentionEvent:
    post_births = state.bound_birth_words[:3].at[target_slot].set(_words(new_birth))
    post_in_use = state.live_in_use.at[target_slot].set(True)
    return ProspectiveLineageRetentionEvent(  # type: ignore[call-arg]
        source_step_words=state.step_words,
        post_step_words=_next_words(state.step_words),
        source_birth_words=state.bound_birth_words[:3],
        post_birth_words=post_births,
        source_in_use=state.live_in_use,
        post_in_use=post_in_use,
        allocated=jnp.asarray(True, dtype=jnp.bool_),
        evicted=jnp.asarray(False, dtype=jnp.bool_),
        target_slot=jnp.asarray(target_slot, dtype=jnp.int32),
        newborn_return_prior=jnp.asarray(newborn_prior, dtype=jnp.float32),
        lineage_transfer_confirmed=jnp.asarray(False, dtype=jnp.bool_),
        transfer_slot=jnp.asarray(-1, dtype=jnp.int32),
        transferred_lineage_words=jnp.zeros((2,), dtype=jnp.uint32),
        archived_loss=jnp.asarray(0.0, dtype=jnp.float32),
        fresh_loss=jnp.asarray(0.0, dtype=jnp.float32),
        context_update_applied=jnp.asarray(True, dtype=jnp.bool_),
    )


def _neutral_event(
    state: ProspectiveLineageRetentionState,
    *,
    transfer_slot: int | None = None,
    lineage: jax.Array | None = None,
    archived_loss: float = 0.0,
    fresh_loss: float = 0.0,
) -> ProspectiveLineageRetentionEvent:
    confirmed = transfer_slot is not None
    return ProspectiveLineageRetentionEvent(  # type: ignore[call-arg]
        source_step_words=state.step_words,
        post_step_words=_next_words(state.step_words),
        source_birth_words=state.bound_birth_words[:3],
        post_birth_words=state.bound_birth_words[:3],
        source_in_use=state.live_in_use,
        post_in_use=state.live_in_use,
        allocated=jnp.asarray(False, dtype=jnp.bool_),
        evicted=jnp.asarray(False, dtype=jnp.bool_),
        target_slot=jnp.asarray(-1, dtype=jnp.int32),
        newborn_return_prior=jnp.asarray(0.0, dtype=jnp.float32),
        lineage_transfer_confirmed=jnp.asarray(confirmed, dtype=jnp.bool_),
        transfer_slot=jnp.asarray(-1 if transfer_slot is None else transfer_slot, dtype=jnp.int32),
        transferred_lineage_words=(
            jnp.zeros((2,), dtype=jnp.uint32) if lineage is None else lineage
        ),
        archived_loss=jnp.asarray(archived_loss, dtype=jnp.float32),
        fresh_loss=jnp.asarray(fresh_loss, dtype=jnp.float32),
        context_update_applied=jnp.asarray(True, dtype=jnp.bool_),
    )


def test_config_is_strict_bounded_and_has_no_claim_authority() -> None:
    mechanism = ProspectiveLineageRetention(CONFIG)
    payload = mechanism.to_config()
    assert payload["evidence_level"] == PROSPECTIVE_LINEAGE_RETENTION_EVIDENCE_LEVEL == "L0"
    assert payload["status"] == PROSPECTIVE_LINEAGE_RETENTION_STATUS
    assert payload["metadata_capacity"] == 4
    assert payload["archive_capacity"] == PROSPECTIVE_LINEAGE_RETENTION_ARCHIVE_CAPACITY == 1
    assert payload["scientific_promotion_allowed"] is False
    assert payload["external_provenance_claimed"] is False
    assert payload["host_event_binding_claimed"] is False
    assert ProspectiveLineageRetention.from_config(payload).to_config() == payload
    assert PROSPECTIVE_LINEAGE_RETENTION_SCIENTIFIC_PROMOTION_ALLOWED is False
    assert PROSPECTIVE_LINEAGE_RETENTION_EXTERNAL_PROVENANCE_CLAIMED is False
    assert PROSPECTIVE_LINEAGE_RETENTION_HOST_EVENT_BINDING_CLAIMED is False

    malformed = dict(payload)
    malformed["scientific_promotion_allowed"] = True
    with pytest.raises(ValueError):
        ProspectiveLineageRetention.from_config(malformed)


@pytest.mark.parametrize(
    ("field", "wrong_type_equal_value"),
    (
        ("metadata_capacity", 4.0),
        ("archive_capacity", 1.0),
        ("initial_reacquisition_cost", 1),
        ("scientific_promotion_allowed", 0),
    ),
)
def test_config_rejects_python_equal_values_with_noncanonical_types(
    field: str,
    wrong_type_equal_value: object,
) -> None:
    payload = ProspectiveLineageRetention(CONFIG).to_config()
    payload[field] = wrong_type_equal_value

    with pytest.raises(ValueError, match="types are not canonical"):
        ProspectiveLineageRetention.from_config(payload)


def test_expected_prior_score_selects_d_and_routed_unrouted_work_is_identical() -> None:
    mechanism, state = _mechanism_and_state()
    routed = _prepare(mechanism, state, routed=True)
    unrouted = _prepare(mechanism, state, routed=False)

    assert bool(routed.preparation_valid)
    assert bool(unrouted.preparation_valid)
    assert int(routed.expected_victim) == 2
    assert int(routed.minimax_victim) == 1
    assert int(routed.selected_victim) == 2
    assert bool(routed.guardrail_passed)
    assert not bool(routed.guardrail_fallback_used)
    chex.assert_trees_all_close(
        routed.expected_scores,
        jnp.asarray((0.99, 0.75, 0.25), dtype=jnp.float32),
    )
    chex.assert_trees_all_equal(routed.expected_scores, unrouted.expected_scores)
    chex.assert_trees_all_equal(routed.minimax_scores, unrouted.minimax_scores)
    chex.assert_trees_all_equal(routed.raw_protection, unrouted.raw_protection)
    assert int(routed.selected_victim) == int(unrouted.selected_victim)
    assert bool(jnp.any(routed.protection_to_route))
    assert not bool(jnp.any(unrouted.protection_to_route))
    chex.assert_trees_all_equal(state, _mechanism_and_state()[1])


@pytest.mark.parametrize("live_count", (0, 1))
def test_sparse_genesis_can_fill_a_free_slot_without_eviction(live_count: int) -> None:
    mechanism = ProspectiveLineageRetention(CONFIG)
    in_use = jnp.asarray(
        tuple(index < live_count for index in range(3)),
        dtype=jnp.bool_,
    )
    state = mechanism.init(
        step_words=_words(10),
        birth_words=_births(
            *(index + 1 if index < live_count else 0 for index in range(3))
        ),
        in_use=in_use,
        return_priors=jnp.asarray(
            tuple(0.75 if index < live_count else 0.0 for index in range(3)),
            dtype=jnp.float32,
        ),
    )
    target_slot = live_count
    preparation = mechanism.prepare(
        state,
        source_birth_words=state.bound_birth_words[:3],
        source_in_use=state.live_in_use,
        active_slot=jnp.asarray(-1 if live_count == 0 else 0, dtype=jnp.int32),
        last_active_words=_births(*(9 if index < live_count else 0 for index in range(3))),
        route_protection=jnp.asarray(True, dtype=jnp.bool_),
    )
    result = mechanism.settle(
        state,
        preparation,
        _free_allocation_event(
            state,
            target_slot=target_slot,
            new_birth=11,
            newborn_prior=0.4,
        ),
    )

    assert bool(preparation.preparation_valid)
    assert int(preparation.expected_victim) == -1
    assert int(preparation.minimax_victim) == -1
    assert int(preparation.selected_victim) == -1
    assert not bool(preparation.guardrail_passed)
    assert not bool(preparation.guardrail_fallback_used)
    assert bool(result.update_applied)
    assert not bool(result.archive_created)
    assert not bool(result.archive_replaced)
    assert bool(result.newborn_bound)
    assert bool(result.state.live_in_use[target_slot])
    chex.assert_trees_all_equal(result.state.bound_birth_words[target_slot], _words(11))
    assert float(result.state.return_prior[target_slot]) == pytest.approx(0.4)
    assert bool(mechanism.state_valid(result.state))


def test_sparse_bank_rejects_eviction_while_a_free_slot_exists() -> None:
    mechanism = ProspectiveLineageRetention(CONFIG)
    state = mechanism.init(
        step_words=_words(10),
        birth_words=_births(1, 2, 0),
        in_use=jnp.asarray((True, True, False), dtype=jnp.bool_),
        return_priors=jnp.asarray((0.75, 0.25, 0.0), dtype=jnp.float32),
    )
    preparation = mechanism.prepare(
        state,
        source_birth_words=state.bound_birth_words[:3],
        source_in_use=state.live_in_use,
        active_slot=jnp.asarray(0, dtype=jnp.int32),
        last_active_words=_births(9, 5, 0),
        route_protection=jnp.asarray(True, dtype=jnp.bool_),
    )
    assert int(preparation.expected_victim) == -1
    assert int(preparation.minimax_victim) == -1
    assert int(preparation.selected_victim) == -1
    assert not bool(jnp.any(preparation.protection_to_route))
    result = mechanism.settle(
        state,
        preparation,
        _allocation_event(state, target_slot=1),
    )

    assert not bool(result.event_valid)
    assert not bool(result.update_applied)
    chex.assert_trees_all_equal(result.state, state)


def test_minimax_guardrail_blocks_low_prior_high_cost_sacrifice() -> None:
    mechanism, state = _mechanism_and_state(priors=(0.99, 0.9, 0.01))
    high_cost = state.reacquisition_cost.at[2].set(jnp.float32(10.0))
    # Synthetic integrity fixture: resealing establishes no external provenance.
    state = mechanism._with_content_token(  # noqa: SLF001
        state.replace(reacquisition_cost=high_cost)  # type: ignore[attr-defined]
    )
    preparation = _prepare(mechanism, state, routed=True)

    assert int(preparation.expected_victim) == 2
    assert int(preparation.minimax_victim) == 1
    assert int(preparation.selected_victim) == 1
    assert not bool(preparation.guardrail_passed)
    assert bool(preparation.guardrail_fallback_used)
    assert float(preparation.expected_victim_worst_cost) == 10.0
    assert float(preparation.minimax_victim_worst_cost) == 1.0
    chex.assert_trees_all_equal(
        preparation.raw_protection,
        jnp.asarray((1.0, 1.0, 10.0), dtype=jnp.float32),
    )


def test_allocation_archives_exact_victim_and_binds_newborn_prior() -> None:
    mechanism, state = _mechanism_and_state()
    preparation = _prepare(mechanism, state, routed=True)
    event = _allocation_event(state)
    result = mechanism.settle(state, preparation, event)
    archive = CONFIG.max_contexts

    assert bool(result.update_applied)
    assert bool(result.routed_eviction_binding_valid)
    assert bool(result.archive_created)
    assert not bool(result.archive_replaced)
    assert bool(result.newborn_bound)
    assert not bool(result.lineage_restored)
    assert not bool(result.parameter_transplanted)
    assert bool(mechanism.state_valid(result.state))
    chex.assert_trees_all_equal(result.state.bound_birth_words[archive], _words(3))
    chex.assert_trees_all_equal(result.state.lineage_words[archive], _words(3))
    assert float(result.state.return_prior[archive]) == pytest.approx(0.25)
    chex.assert_trees_all_equal(result.state.bound_birth_words[2], _words(11))
    chex.assert_trees_all_equal(result.state.lineage_words[2], _words(11))
    assert float(result.state.return_prior[2]) == pytest.approx(0.4)
    assert int(result.state.cost_support_words[2, 1]) == 0


def test_later_eviction_replaces_the_single_archive_exactly() -> None:
    mechanism, state = _mechanism_and_state()
    first = mechanism.settle(
        state,
        _prepare(mechanism, state, routed=True),
        _allocation_event(state),
    )
    assert bool(first.update_applied)
    second_preparation = _prepare(mechanism, first.state, routed=True)
    assert int(second_preparation.selected_victim) == 2
    second = mechanism.settle(
        first.state,
        second_preparation,
        _allocation_event(first.state, new_birth=12, newborn_prior=0.6),
    )
    archive = CONFIG.max_contexts

    assert bool(second.update_applied)
    assert not bool(second.archive_created)
    assert bool(second.archive_replaced)
    chex.assert_trees_all_equal(second.state.bound_birth_words[archive], _words(11))
    chex.assert_trees_all_equal(second.state.lineage_words[archive], _words(11))
    chex.assert_trees_all_equal(second.state.bound_birth_words[2], _words(12))
    assert bool(mechanism.state_valid(second.state))


def test_confirmed_rebirth_restores_lineage_prior_and_causal_cost_only_later() -> None:
    mechanism, source = _mechanism_and_state()
    opening_preparation = _prepare(mechanism, source, routed=True)
    opening = mechanism.settle(source, opening_preparation, _allocation_event(source))
    assert bool(opening.update_applied)
    newborn = opening.state

    before_confirmation = _prepare(mechanism, newborn, routed=True)
    assert float(before_confirmation.expected_scores[2]) == pytest.approx(0.4)
    event = _neutral_event(
        newborn,
        transfer_slot=2,
        lineage=_words(3),
        archived_loss=0.1,
        fresh_loss=0.9,
    )
    restored = mechanism.settle(newborn, before_confirmation, event)

    assert bool(restored.update_applied)
    assert bool(restored.lineage_restored)
    assert bool(restored.prior_restored)
    assert bool(restored.cost_updated)
    assert float(restored.cost_observation) == pytest.approx(0.8)
    chex.assert_trees_all_equal(restored.state.bound_birth_words[2], _words(11))
    chex.assert_trees_all_equal(restored.state.lineage_words[2], _words(3))
    chex.assert_trees_all_equal(restored.state.prior_source_birth_words[2], _words(3))
    assert float(restored.state.return_prior[2]) == pytest.approx(0.25)
    assert float(restored.state.reacquisition_cost[2]) == pytest.approx(0.9)
    assert int(restored.state.cost_support_words[2, 1]) == 1
    assert not bool(restored.state.valid[3])
    assert bool(mechanism.state_valid(restored.state))

    # The completed loss evidence did not and could not rewrite the preparation
    # that preceded it.
    assert float(before_confirmation.expected_scores[2]) == pytest.approx(0.4)
    after_confirmation = _prepare(mechanism, restored.state, routed=True)
    assert float(after_confirmation.expected_scores[2]) == pytest.approx(0.225)


def test_same_descriptor_new_birth_is_not_restored_without_confirmation() -> None:
    mechanism, state = _mechanism_and_state()
    preparation = _prepare(mechanism, state, routed=True)
    result = mechanism.settle(state, preparation, _allocation_event(state))

    assert bool(result.update_applied)
    assert not bool(result.lineage_restored)
    chex.assert_trees_all_equal(result.state.lineage_words[2], _words(11))
    assert float(result.state.return_prior[2]) == pytest.approx(0.4)
    assert float(result.state.return_prior[3]) == pytest.approx(0.25)


@pytest.mark.parametrize("failure", ("wrong_target", "nan_prior"))
def test_invalid_host_binding_or_payload_rolls_back_exactly(failure: str) -> None:
    mechanism, state = _mechanism_and_state()
    preparation = _prepare(mechanism, state, routed=True)
    event = _allocation_event(state)
    if failure == "wrong_target":
        event = _allocation_event(state, target_slot=1)
    else:
        event = _allocation_event(state, newborn_prior=float("nan"))

    result = mechanism.settle(state, preparation, event)
    assert not bool(result.update_applied)
    chex.assert_trees_all_equal(result.state, state)


def test_wrong_confirmed_lineage_isolated_against_a_valid_archive() -> None:
    mechanism, state = _mechanism_and_state()
    opened = mechanism.settle(
        state,
        _prepare(mechanism, state, routed=True),
        _allocation_event(state),
    )
    assert bool(opened.update_applied)
    prepared = _prepare(mechanism, opened.state, routed=True)
    result = mechanism.settle(
        opened.state,
        prepared,
        _neutral_event(
            opened.state,
            transfer_slot=2,
            lineage=_words(999),
            archived_loss=0.1,
            fresh_loss=0.9,
        ),
    )

    assert not bool(result.event_valid)
    assert not bool(result.update_applied)
    chex.assert_trees_all_equal(result.state, opened.state)


def test_resealed_lineage_prior_source_mismatch_is_invalid() -> None:
    mechanism, state = _mechanism_and_state()
    donor = mechanism.init(
        step_words=_words(10),
        birth_words=_births(99, 0, 0),
        in_use=jnp.asarray((True, False, False), dtype=jnp.bool_),
        return_priors=jnp.asarray((0.75, 0.0, 0.0), dtype=jnp.float32),
    )
    impossible = mechanism._with_content_token(  # noqa: SLF001
        state.replace(  # type: ignore[attr-defined]
            prior_source_birth_words=state.prior_source_birth_words.at[1].set(_words(99)),
            prior_receipt_words=state.prior_receipt_words.at[1].set(
                donor.prior_receipt_words[0]
            ),
        )
    )

    assert not bool(mechanism.state_valid(impossible))


def test_stale_preparation_and_state_tamper_fail_closed() -> None:
    mechanism, state = _mechanism_and_state()
    preparation = _prepare(mechanism, state, routed=True)
    applied = mechanism.settle(state, preparation, _allocation_event(state))
    assert bool(applied.update_applied)

    stale = mechanism.settle(applied.state, preparation, _allocation_event(state))
    assert not bool(stale.preparation_authenticated)
    assert not bool(stale.update_applied)
    chex.assert_trees_all_equal(stale.state, applied.state)

    tampered = state.replace(  # type: ignore[attr-defined]
        reacquisition_cost=state.reacquisition_cost.at[1].set(jnp.float32(7.0))
    )
    rejected = _prepare(mechanism, tampered, routed=True)
    assert not bool(rejected.source_state_valid)
    assert not bool(rejected.preparation_valid)
    assert not bool(jnp.any(rejected.protection_to_route))


@pytest.mark.parametrize(
    "tamper",
    ("selected_victim", "raw_and_route", "protection_only", "route_bundle"),
)
def test_preparation_value_tamper_rolls_back_exactly(tamper: str) -> None:
    mechanism, state = _mechanism_and_state()
    preparation = _prepare(mechanism, state, routed=True)
    if tamper == "selected_victim":
        forged = preparation.replace(  # type: ignore[attr-defined]
            selected_victim=jnp.asarray(1, dtype=jnp.int32)
        )
    elif tamper == "raw_and_route":
        forged_raw = preparation.raw_protection.at[2].set(jnp.float32(9.0))
        forged = preparation.replace(  # type: ignore[attr-defined]
            raw_protection=forged_raw,
            protection_to_route=forged_raw,
        )
    elif tamper == "protection_only":
        forged = preparation.replace(  # type: ignore[attr-defined]
            protection_to_route=jnp.zeros((3,), dtype=jnp.float32)
        )
    else:
        # These two fields are internally coherent for an unrouted preparation,
        # but the unchanged route receipt binds the actual pre-outcome choice.
        forged = preparation.replace(  # type: ignore[attr-defined]
            route_protection=jnp.asarray(False, dtype=jnp.bool_),
            protection_to_route=jnp.zeros((3,), dtype=jnp.float32),
        )

    result = mechanism.settle(state, forged, _allocation_event(state))
    assert not bool(result.preparation_authenticated)
    assert not bool(result.update_applied)
    chex.assert_trees_all_equal(result.state, state)


def test_eager_jit_prepare_and_settlement_are_exact() -> None:
    mechanism, state = _mechanism_and_state()
    births = state.bound_birth_words[:3]
    in_use = state.live_in_use
    active = jnp.asarray(0, dtype=jnp.int32)
    recency = _births(9, 5, 7)
    route = jnp.asarray(True, dtype=jnp.bool_)
    eager_preparation = mechanism.prepare(
        state,
        source_birth_words=births,
        source_in_use=in_use,
        active_slot=active,
        last_active_words=recency,
        route_protection=route,
    )
    compiled_preparation = jax.jit(
        lambda source, source_births, source_use, source_active, source_recency, routed: (
            mechanism.prepare(
                source,
                source_birth_words=source_births,
                source_in_use=source_use,
                active_slot=source_active,
                last_active_words=source_recency,
                route_protection=routed,
            )
        )
    )(state, births, in_use, active, recency, route)
    chex.assert_trees_all_equal(
        eager_preparation,
        compiled_preparation,
    )

    event = _allocation_event(state)
    eager = mechanism.settle(state, eager_preparation, event)
    compiled = jax.jit(mechanism.settle)(state, compiled_preparation, event)
    chex.assert_trees_all_equal(eager, compiled)


def test_compiled_sparse_restoration_invalid_indices_and_exhaustion_are_exact() -> None:
    mechanism = ProspectiveLineageRetention(CONFIG)
    compiled_prepare = jax.jit(
        lambda source, source_births, source_use, source_active, source_recency, routed: (
            mechanism.prepare(
                source,
                source_birth_words=source_births,
                source_in_use=source_use,
                active_slot=source_active,
                last_active_words=source_recency,
                route_protection=routed,
            )
        )
    )
    compiled_settle = jax.jit(mechanism.settle)

    sparse = mechanism.init(
        step_words=_words(10),
        birth_words=_births(1, 0, 0),
        in_use=jnp.asarray((True, False, False), dtype=jnp.bool_),
        return_priors=jnp.asarray((0.75, 0.0, 0.0), dtype=jnp.float32),
    )
    sparse_inputs = (
        sparse,
        sparse.bound_birth_words[:3],
        sparse.live_in_use,
        jnp.asarray(0, dtype=jnp.int32),
        _births(9, 0, 0),
        jnp.asarray(True, dtype=jnp.bool_),
    )
    sparse_eager_preparation = mechanism.prepare(
        sparse,
        source_birth_words=sparse_inputs[1],
        source_in_use=sparse_inputs[2],
        active_slot=sparse_inputs[3],
        last_active_words=sparse_inputs[4],
        route_protection=sparse_inputs[5],
    )
    sparse_compiled_preparation = compiled_prepare(*sparse_inputs)
    chex.assert_trees_all_equal(sparse_eager_preparation, sparse_compiled_preparation)
    assert int(sparse_compiled_preparation.selected_victim) == -1
    assert not bool(jnp.any(sparse_compiled_preparation.protection_to_route))

    free_event = _free_allocation_event(
        sparse,
        target_slot=1,
        new_birth=11,
        newborn_prior=0.4,
    )
    sparse_eager = mechanism.settle(sparse, sparse_eager_preparation, free_event)
    sparse_compiled = compiled_settle(sparse, sparse_compiled_preparation, free_event)
    chex.assert_trees_all_equal(sparse_eager, sparse_compiled)
    assert bool(sparse_compiled.update_applied)
    assert not bool(sparse_compiled.archive_created)

    invalid_active_inputs = (
        sparse,
        sparse.bound_birth_words[:3],
        sparse.live_in_use,
        jnp.asarray(99, dtype=jnp.int32),
        _births(9, 0, 0),
        jnp.asarray(True, dtype=jnp.bool_),
    )
    invalid_active_eager = mechanism.prepare(
        sparse,
        source_birth_words=invalid_active_inputs[1],
        source_in_use=invalid_active_inputs[2],
        active_slot=invalid_active_inputs[3],
        last_active_words=invalid_active_inputs[4],
        route_protection=invalid_active_inputs[5],
    )
    invalid_active_compiled = compiled_prepare(*invalid_active_inputs)
    chex.assert_trees_all_equal(invalid_active_eager, invalid_active_compiled)
    assert not bool(invalid_active_compiled.preparation_valid)

    _, full = _mechanism_and_state()
    full_preparation = _prepare(mechanism, full, routed=True)
    invalid_target_event = _allocation_event(full).replace(  # type: ignore[attr-defined]
        target_slot=jnp.asarray(99, dtype=jnp.int32)
    )
    invalid_target_eager = mechanism.settle(full, full_preparation, invalid_target_event)
    invalid_target_compiled = compiled_settle(
        full,
        full_preparation,
        invalid_target_event,
    )
    chex.assert_trees_all_equal(invalid_target_eager, invalid_target_compiled)
    assert not bool(invalid_target_compiled.update_applied)
    chex.assert_trees_all_equal(invalid_target_compiled.state, full)

    opened = mechanism.settle(full, full_preparation, _allocation_event(full))
    assert bool(opened.update_applied)
    restoration_preparation = _prepare(mechanism, opened.state, routed=True)
    restoration_event = _neutral_event(
        opened.state,
        transfer_slot=2,
        lineage=_words(3),
        archived_loss=0.1,
        fresh_loss=0.9,
    )
    restoration_eager = mechanism.settle(
        opened.state,
        restoration_preparation,
        restoration_event,
    )
    restoration_compiled = compiled_settle(
        opened.state,
        restoration_preparation,
        restoration_event,
    )
    chex.assert_trees_all_equal(restoration_eager, restoration_compiled)
    assert bool(restoration_compiled.lineage_restored)
    assert int(restoration_compiled.state.cost_support_words[2, 1]) == 1

    invalid_transfer_event = restoration_event.replace(  # type: ignore[attr-defined]
        transfer_slot=jnp.asarray(99, dtype=jnp.int32)
    )
    invalid_transfer_eager = mechanism.settle(
        opened.state,
        restoration_preparation,
        invalid_transfer_event,
    )
    invalid_transfer_compiled = compiled_settle(
        opened.state,
        restoration_preparation,
        invalid_transfer_event,
    )
    chex.assert_trees_all_equal(invalid_transfer_eager, invalid_transfer_compiled)
    assert not bool(invalid_transfer_compiled.update_applied)
    chex.assert_trees_all_equal(invalid_transfer_compiled.state, opened.state)

    exhausted = mechanism._with_content_token(  # noqa: SLF001
        full.replace(  # type: ignore[attr-defined]
            revision_words=jnp.asarray((2**32 - 1, 2**32 - 1), dtype=jnp.uint32)
        )
    )
    exhausted_preparation = _prepare(mechanism, exhausted, routed=True)
    exhausted_event = _allocation_event(exhausted)
    exhausted_eager = mechanism.settle(
        exhausted,
        exhausted_preparation,
        exhausted_event,
    )
    exhausted_compiled = compiled_settle(
        exhausted,
        exhausted_preparation,
        exhausted_event,
    )
    chex.assert_trees_all_equal(exhausted_eager, exhausted_compiled)
    assert not bool(exhausted_compiled.revision_capacity_available)
    assert not bool(exhausted_compiled.update_applied)
    chex.assert_trees_all_equal(exhausted_compiled.state, exhausted)


def test_two_eviction_scan_matches_compiled_scan_exactly() -> None:
    mechanism, initial = _mechanism_and_state()

    def run(
        source: ProspectiveLineageRetentionState,
    ) -> tuple[ProspectiveLineageRetentionState, tuple[jax.Array, jax.Array]]:
        def body(
            state: ProspectiveLineageRetentionState,
            newborn: jax.Array,
        ) -> tuple[ProspectiveLineageRetentionState, tuple[jax.Array, jax.Array]]:
            preparation = mechanism.prepare(
                state,
                source_birth_words=state.bound_birth_words[:3],
                source_in_use=state.live_in_use,
                active_slot=jnp.asarray(0, dtype=jnp.int32),
                last_active_words=_births(9, 5, 7),
                route_protection=jnp.asarray(True, dtype=jnp.bool_),
            )
            target = preparation.selected_victim
            new_birth = jnp.stack((jnp.uint32(0), newborn)).astype(jnp.uint32)
            post_births = state.bound_birth_words[:3].at[target].set(new_birth)
            low = state.step_words[1] + jnp.uint32(1)
            post_step = jnp.stack(
                (
                    state.step_words[0] + (low == 0).astype(jnp.uint32),
                    low,
                )
            ).astype(jnp.uint32)
            event = ProspectiveLineageRetentionEvent(  # type: ignore[call-arg]
                source_step_words=state.step_words,
                post_step_words=post_step,
                source_birth_words=state.bound_birth_words[:3],
                post_birth_words=post_births,
                source_in_use=state.live_in_use,
                post_in_use=state.live_in_use,
                allocated=jnp.asarray(True, dtype=jnp.bool_),
                evicted=jnp.asarray(True, dtype=jnp.bool_),
                target_slot=target,
                newborn_return_prior=jnp.asarray(0.4, dtype=jnp.float32),
                lineage_transfer_confirmed=jnp.asarray(False, dtype=jnp.bool_),
                transfer_slot=jnp.asarray(-1, dtype=jnp.int32),
                transferred_lineage_words=jnp.zeros((2,), dtype=jnp.uint32),
                archived_loss=jnp.asarray(0.0, dtype=jnp.float32),
                fresh_loss=jnp.asarray(0.0, dtype=jnp.float32),
                context_update_applied=jnp.asarray(True, dtype=jnp.bool_),
            )
            proposal = mechanism.settle(state, preparation, event)
            return proposal.state, (proposal.update_applied, target)

        return jax.lax.scan(
            body,
            source,
            jnp.asarray((11, 12), dtype=jnp.uint32),
        )

    eager = run(initial)
    compiled = jax.jit(run)(initial)
    chex.assert_trees_all_equal(eager, compiled)
    chex.assert_trees_all_equal(
        eager[1][0],
        jnp.asarray((True, True), dtype=jnp.bool_),
    )


def test_checkpoint_config_and_state_round_trip(tmp_path: Path) -> None:
    mechanism, state = _mechanism_and_state()
    preparation = _prepare(mechanism, state, routed=True)
    advanced = mechanism.settle(state, preparation, _allocation_event(state)).state
    checkpoint = tmp_path / "prospective-retention"

    save_prospective_lineage_retention_checkpoint(mechanism, advanced, checkpoint)
    restored_mechanism, restored_state = load_prospective_lineage_retention_checkpoint(
        checkpoint
    )
    assert restored_mechanism.to_config() == mechanism.to_config()
    chex.assert_trees_all_equal(restored_state, advanced)
    assert bool(restored_mechanism.state_valid(restored_state))


def test_checkpoint_rejects_noncanonical_config_types_and_state_tamper(
    tmp_path: Path,
) -> None:
    mechanism, state = _mechanism_and_state()
    preparation = _prepare(mechanism, state, routed=True)
    advanced = mechanism.settle(state, preparation, _allocation_event(state)).state

    config = mechanism.to_config()
    config["archive_capacity"] = 1.0
    config_tamper = tmp_path / "prospective-retention-config-tamper"
    save_checkpoint(
        advanced,
        config_tamper,
        metadata={
            "schema": PROSPECTIVE_LINEAGE_RETENTION_CHECKPOINT_SCHEMA,
            "config": config,
        },
    )
    with pytest.raises(ValueError, match="types are not canonical"):
        load_prospective_lineage_retention_checkpoint(config_tamper)

    state_tamper = dataclasses.replace(  # type: ignore[type-var]
        advanced,
        return_prior=advanced.return_prior.at[0].add(jnp.float32(0.125))
    )
    state_path = tmp_path / "prospective-retention-state-tamper"
    save_checkpoint(
        state_tamper,
        state_path,
        metadata={
            "schema": PROSPECTIVE_LINEAGE_RETENTION_CHECKPOINT_SCHEMA,
            "config": mechanism.to_config(),
        },
    )
    with pytest.raises(ValueError, match="state is invalid"):
        load_prospective_lineage_retention_checkpoint(state_path)


def test_resource_and_work_records_are_fixed_and_branch_neutral() -> None:
    mechanism, state = _mechanism_and_state()
    resource = mechanism.resource_record()
    work = mechanism.work_record(preparations=100, settlements=80)

    assert measure_prospective_lineage_retention_state_nbytes(state) == 295
    assert resource.state_nbytes == resource.measured_state_nbytes == 295
    assert resource.metadata_capacity == 4
    assert resource.archive_capacity == 1
    assert resource.replay_capacity == PROSPECTIVE_LINEAGE_RETENTION_REPLAY_CAPACITY == 0
    assert resource.persistent_capacity_growth == 0
    assert resource.parameter_transplant_allowed is False
    assert resource.external_provenance_claimed is False
    assert resource.host_event_binding_claimed is False

    assert work.authentication_repreparations == 80
    assert work.total_score_preparations == 180
    assert work.score_products == 540
    assert work.expected_selection_cells == 540
    assert work.minimax_selection_cells == 540
    assert work.guardrail_comparisons == 180
    assert work.metadata_route_cells == 320
    assert work.settlement_proposals == 80
    assert work.random_draws == work.replay_updates == work.reset_callbacks == 0
    assert work.routed_unrouted_same_preparation_work is True
    assert work.exhaustive_primitive_operation_count_claimed is False
    assert work.compiled_flop_count_claimed is False


def test_revision_exhaustion_rolls_back_without_partial_metadata() -> None:
    mechanism, state = _mechanism_and_state()
    exhausted = mechanism._with_content_token(  # noqa: SLF001
        state.replace(  # type: ignore[attr-defined]
            revision_words=jnp.asarray((2**32 - 1, 2**32 - 1), dtype=jnp.uint32)
        )
    )
    preparation = _prepare(mechanism, exhausted, routed=True)
    result = mechanism.settle(exhausted, preparation, _allocation_event(exhausted))

    assert not bool(result.revision_capacity_available)
    assert not bool(result.update_applied)
    chex.assert_trees_all_equal(result.state, exhausted)


def test_step_exhaustion_rolls_back_without_partial_metadata() -> None:
    mechanism, state = _mechanism_and_state()
    exhausted = mechanism._with_content_token(  # noqa: SLF001
        state.replace(  # type: ignore[attr-defined]
            step_words=jnp.asarray((2**32 - 1, 2**32 - 1), dtype=jnp.uint32)
        )
    )
    preparation = _prepare(
        mechanism,
        exhausted,
        routed=True,
        recency=(2**32 - 1, 2**32 - 1, 2**32 - 1),
    )
    event = _allocation_event(exhausted).replace(  # type: ignore[attr-defined]
        post_step_words=exhausted.step_words
    )
    result = mechanism.settle(exhausted, preparation, event)

    assert not bool(result.step_capacity_available)
    assert not bool(result.update_applied)
    chex.assert_trees_all_equal(result.state, exhausted)


def test_low_word_carry_advances_step_and_revision_exactly() -> None:
    mechanism, state = _mechanism_and_state()
    near_carry = mechanism._with_content_token(  # noqa: SLF001
        state.replace(  # type: ignore[attr-defined]
            step_words=jnp.asarray((7, 2**32 - 1), dtype=jnp.uint32),
            revision_words=jnp.asarray((5, 2**32 - 1), dtype=jnp.uint32),
        )
    )
    preparation = _prepare(mechanism, near_carry, routed=True)
    result = mechanism.settle(
        near_carry,
        preparation,
        _allocation_event(near_carry),
    )

    assert bool(result.update_applied)
    chex.assert_trees_all_equal(
        result.state.step_words,
        jnp.asarray((8, 0), dtype=jnp.uint32),
    )
    chex.assert_trees_all_equal(
        result.state.revision_words,
        jnp.asarray((6, 0), dtype=jnp.uint32),
    )


def test_duplicate_newborn_birth_rolls_back_exactly() -> None:
    mechanism, state = _mechanism_and_state()
    preparation = _prepare(mechanism, state, routed=True)
    result = mechanism.settle(
        state,
        preparation,
        _allocation_event(state, new_birth=1),
    )

    assert not bool(result.event_valid)
    assert not bool(result.update_applied)
    chex.assert_trees_all_equal(result.state, state)


def test_cost_support_exhaustion_rolls_back_confirmed_restoration_exactly() -> None:
    mechanism, state = _mechanism_and_state()
    preparation = _prepare(mechanism, state, routed=True)
    opened = mechanism.settle(state, preparation, _allocation_event(state))
    assert bool(opened.update_applied)

    archive = CONFIG.max_contexts
    exhausted = mechanism._with_content_token(  # noqa: SLF001
        opened.state.replace(  # type: ignore[attr-defined]
            cost_support_words=opened.state.cost_support_words.at[archive].set(
                jnp.asarray((2**32 - 1, 2**32 - 1), dtype=jnp.uint32)
            )
        )
    )
    prepared = _prepare(mechanism, exhausted, routed=True)
    event = _neutral_event(
        exhausted,
        transfer_slot=2,
        lineage=_words(3),
        archived_loss=0.1,
        fresh_loss=0.9,
    )
    result = mechanism.settle(exhausted, prepared, event)

    assert bool(result.preparation_authenticated)
    assert bool(result.event_valid)
    assert bool(result.step_capacity_available)
    assert bool(result.revision_capacity_available)
    assert not bool(result.cost_support_capacity_available)
    assert not bool(result.update_applied)
    assert not bool(result.lineage_restored)
    assert not bool(result.cost_updated)
    chex.assert_trees_all_equal(result.state, exhausted)
