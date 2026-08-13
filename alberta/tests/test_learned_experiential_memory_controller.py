# mypy: disable-error-code="arg-type,call-arg,type-var"
"""Learned retrieval-admission and eviction-retention memory contracts."""

from __future__ import annotations

import copy
import dataclasses
import json
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from alberta_framework.core.checkpoints import save_checkpoint
from alberta_framework.core.experiential_memory import (
    ExperientialMemoryConfig,
    ExperientialMemoryEntry,
)
from alberta_framework.core.learned_experiential_memory_controller import (
    LEARNED_EXPERIENTIAL_MEMORY_CHECKPOINT_SCHEMA,
    LEARNED_EXPERIENTIAL_MEMORY_CONFIG_SCHEMA,
    LEARNED_EXPERIENTIAL_MEMORY_FEATURE_COUNT,
    LearnedExperientialMemoryController,
    LearnedExperientialMemoryControllerConfig,
    LearnedExperientialMemoryFeedback,
    load_learned_experiential_memory_checkpoint,
    save_learned_experiential_memory_checkpoint,
)

pytestmark = pytest.mark.integration


def test_learned_memory_controller_has_a_separate_owner_type() -> None:
    """The learned policy must not be hidden inside the fixed v2 store."""

    assert LearnedExperientialMemoryController.__module__.endswith(
        "learned_experiential_memory_controller"
    )

    import alberta_framework
    import alberta_framework.core as core

    assert alberta_framework.LearnedExperientialMemoryController is (
        LearnedExperientialMemoryController
    )
    assert core.LearnedExperientialMemoryController is (
        LearnedExperientialMemoryController
    )


def _config(
    *,
    capacity: int = 3,
    top_k: int = 1,
    admission_step_size: float = 0.5,
    retention_step_size: float = 1.0,
) -> LearnedExperientialMemoryControllerConfig:
    return LearnedExperientialMemoryControllerConfig(
        memory=ExperientialMemoryConfig(
            capacity=capacity,
            observation_dim=2,
            key_dim=2,
            action_dim=2,
            outcome_dim=1,
            top_k=top_k,
            min_neighbors=1,
            distance_scale=1.0,
            min_similarity=0.0,
            min_effective_reliability=1.0e-6,
            max_uncertainty=1.0,
            max_safety_cost=1.0,
            max_age=100,
            staleness_scale=100.0,
            utility_decay=1.0,
            eviction_utility_weight=1.0,
            eviction_recency_weight=0.0,
            recency_scale=10.0,
        ),
        admission_step_size=admission_step_size,
        retention_step_size=retention_step_size,
        admission_threshold=0.0,
        initial_admission_bias=0.0,
        max_abs_admission_weight=8.0,
        max_abs_counterfactual_delta=1.0,
        retention_prior=0.5,
    )


def _entry(
    provenance_id: int,
    *,
    key: tuple[float, float] | None = None,
    reward: float = 1.0,
) -> ExperientialMemoryEntry:
    chosen_key = key or (float(provenance_id), 0.0)
    return ExperientialMemoryEntry(
        observation=jnp.asarray(chosen_key, dtype=jnp.float32),
        key=jnp.asarray(chosen_key, dtype=jnp.float32),
        action=jnp.asarray((1.0, 0.0), dtype=jnp.float32),
        outcome=jnp.asarray((reward,), dtype=jnp.float32),
        reward=jnp.asarray(reward, dtype=jnp.float32),
        uncertainty=jnp.asarray(0.1, dtype=jnp.float32),
        uncertainty_available=jnp.asarray(True, dtype=jnp.bool_),
        safety_cost=jnp.asarray(0.0, dtype=jnp.float32),
        safety_cost_available=jnp.asarray(True, dtype=jnp.bool_),
        reliability=jnp.asarray(1.0, dtype=jnp.float32),
        # This is deliberately ignored by the learned owner.
        utility=jnp.asarray(99.0, dtype=jnp.float32),
        utility_available=jnp.asarray(True, dtype=jnp.bool_),
        representation_version=jnp.asarray(0, dtype=jnp.int32),
        valid=jnp.asarray(True, dtype=jnp.bool_),
        age=jnp.asarray(0, dtype=jnp.int32),
        provenance_id=jnp.asarray(provenance_id, dtype=jnp.int32),
        source_id=jnp.asarray(7, dtype=jnp.int32),
    )


def _step(
    controller: LearnedExperientialMemoryController,
    state: Any,
    *,
    query_key: tuple[float, float],
    entry: ExperientialMemoryEntry,
) -> Any:
    return controller.step(
        state,
        jnp.asarray(query_key, dtype=jnp.float32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.1, dtype=jnp.float32),
        jnp.asarray(True, dtype=jnp.bool_),
        entry,
    )


def _feedback(
    state: Any,
    delta: float,
    *,
    used: bool = True,
    available: bool = True,
) -> LearnedExperientialMemoryFeedback:
    return LearnedExperientialMemoryFeedback(
        transaction_words=state.pending.transaction_words,
        retrieval_used=jnp.asarray(used, dtype=jnp.bool_),
        counterfactual_available=jnp.asarray(available, dtype=jnp.bool_),
        counterfactual_delta=jnp.asarray(delta, dtype=jnp.float32),
    )


def _assert_tree_equal(left: object, right: object) -> None:
    left_leaves, left_tree = jax.tree_util.tree_flatten(left)
    right_leaves, right_tree = jax.tree_util.tree_flatten(right)
    assert str(left_tree) == str(right_tree)
    assert len(left_leaves) == len(right_leaves)
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        np.testing.assert_array_equal(np.asarray(left_leaf), np.asarray(right_leaf))


def test_config_roundtrip_is_strict_and_declares_narrow_authority() -> None:
    config = _config(capacity=4, top_k=2)
    payload = config.to_config()
    assert payload["schema"] == LEARNED_EXPERIENTIAL_MEMORY_CONFIG_SCHEMA
    assert payload["mechanism_status"] == "l0-mechanism-only-not-assessed"
    assert payload["fixed_store_gate_can_only_reject"] is True
    assert payload["controller_owns_entry_utility_channel"] is True
    assert payload["counterfactual_feedback_authenticated"] is False
    for field in (
        "action_dispatch_authority",
        "safety_authority",
        "evidence_authority",
        "promotion_authority",
        "scientific_promotion_allowed",
    ):
        assert payload[field] is False
    decoded = cast(dict[str, object], json.loads(json.dumps(payload)))
    assert LearnedExperientialMemoryControllerConfig.from_config(decoded) == config
    assert LearnedExperientialMemoryController.from_config(decoded).config == config

    for malformed in (
        {**decoded, "extra": 1},
        {key: value for key, value in decoded.items() if key != "retention_prior"},
        {**decoded, "schema": "wrong"},
        {**decoded, "mechanism_status": "accepted"},
        {**decoded, "scientific_promotion_allowed": True},
        {**decoded, "retention_prior": 1},
    ):
        with pytest.raises(ValueError):
            LearnedExperientialMemoryControllerConfig.from_config(malformed)


@pytest.mark.parametrize(
    "changes",
    [
        {"admission_step_size": -0.1},
        {"retention_step_size": 1.1},
        {"initial_admission_bias": float("nan")},
        {"max_abs_admission_weight": 0.0},
        {"max_abs_counterfactual_delta": float("inf")},
        {"retention_prior": -0.1},
    ],
)
def test_config_rejects_invalid_learning_bounds(changes: dict[str, object]) -> None:
    base = _config()
    with pytest.raises(ValueError):
        dataclasses.replace(base, **changes)

    no_utility = dataclasses.replace(base.memory, eviction_utility_weight=0.0)
    with pytest.raises(ValueError):
        dataclasses.replace(base, memory=no_utility)


def test_initial_state_and_resource_budget_are_exact_and_bounded() -> None:
    controller = LearnedExperientialMemoryController(_config(capacity=4, top_k=2))
    state = controller.init()
    assert bool(controller.state_valid(state))
    np.testing.assert_array_equal(state.admission_weights, np.zeros((7,), np.float32))
    assert not bool(state.pending.available)
    budget = controller.resource_budget(state)
    assert budget.memory_capacity == 4
    assert budget.top_k == 2
    assert budget.admission_feature_count == LEARNED_EXPERIENTIAL_MEMORY_FEATURE_COUNT
    assert budget.admission_trainable_float32_scalars == 7
    assert budget.owned_persistent_state_bytes > budget.nested_memory_persistent_state_bytes
    assert budget.maximum_memory_queries_per_step == 1
    assert budget.maximum_memory_writes_per_step == 1
    assert budget.maximum_retention_updates_per_feedback == 2
    assert budget.random_draws_per_step == 0
    assert budget.caller_counterfactual_feedback_authenticated is False
    assert budget.action_dispatch_authority is False
    assert budget.scientific_promotion_allowed is False


def test_query_is_pre_write_and_admitted_access_creates_one_pending_owner() -> None:
    controller = LearnedExperientialMemoryController(_config())
    state = controller.init()
    first = _step(controller, state, query_key=(0.0, 0.0), entry=_entry(10, key=(0.0, 0.0)))
    assert bool(first.diagnostics.transaction_applied)
    assert bool(first.wrote)
    assert not bool(first.fixed_store_retrieval.accepted)
    assert not bool(first.state.pending.available)
    assert int(first.state.memory.query_count) == 1
    np.testing.assert_allclose(
        first.state.memory.entries.utilities[first.slot], 0.5, rtol=0, atol=0
    )

    second = _step(
        controller,
        first.state,
        query_key=(0.0, 0.0),
        entry=_entry(11, key=(0.0, 0.0)),
    )
    assert bool(second.fixed_store_retrieval.accepted)
    assert bool(second.retrieval.accepted)
    assert bool(second.diagnostics.learned_retrieval_admitted)
    assert bool(second.state.pending.available)
    assert int(second.fixed_store_retrieval.neighbor_provenance_ids[0]) == 10
    assert int(second.state.memory.query_count) == 2
    assert int(second.state.memory.accepted_query_count) == 1
    assert int(second.state.memory.entries.retrieval_counts[0]) == 1
    np.testing.assert_array_equal(
        second.state.pending.transaction_words, second.state.transaction_words
    )


def test_fixed_store_rejection_cannot_be_promoted_and_does_not_veto_write() -> None:
    config = dataclasses.replace(_config(), initial_admission_bias=8.0)
    controller = LearnedExperientialMemoryController(config)
    result = controller.step(
        controller.init(),
        jnp.asarray((jnp.nan, 0.0), dtype=jnp.float32),
        jnp.asarray(-1, dtype=jnp.int32),
        jnp.asarray(jnp.nan, dtype=jnp.float32),
        jnp.asarray(False, dtype=jnp.bool_),
        _entry(1, key=(0.0, 0.0)),
    )
    assert bool(result.diagnostics.transaction_applied)
    assert bool(result.wrote)
    assert not bool(result.fixed_store_retrieval.accepted)
    assert not bool(result.retrieval.accepted)
    assert not bool(result.state.pending.available)
    assert int(result.state.memory.write_count) == 1
    assert int(result.state.memory.query_count) == 1


def test_pending_blocks_next_step_and_stale_feedback_is_exact_noop() -> None:
    controller = LearnedExperientialMemoryController(_config())
    first = _step(
        controller,
        controller.init(),
        query_key=(0.0, 0.0),
        entry=_entry(1, key=(0.0, 0.0)),
    )
    pending = _step(
        controller,
        first.state,
        query_key=(0.0, 0.0),
        entry=_entry(2, key=(0.0, 0.0)),
    )
    blocked = _step(
        controller,
        pending.state,
        query_key=(0.0, 0.0),
        entry=_entry(3, key=(0.0, 0.0)),
    )
    assert bool(blocked.diagnostics.pending_blocked)
    assert not bool(blocked.diagnostics.transaction_applied)
    _assert_tree_equal(blocked.state, pending.state)

    stale = dataclasses.replace(
        _feedback(pending.state, 1.0),
        transaction_words=jnp.asarray((99, 99), dtype=jnp.uint32),
    )
    rejected = controller.settle(pending.state, stale)
    assert not bool(rejected.diagnostics.transaction_applied)
    _assert_tree_equal(rejected.state, pending.state)


@pytest.mark.parametrize(
    ("used", "available"),
    ((False, False), (False, True), (True, False)),
)
def test_matching_feedback_without_complete_causal_use_clears_without_learning(
    used: bool,
    available: bool,
) -> None:
    controller = LearnedExperientialMemoryController(_config())
    first = _step(
        controller,
        controller.init(),
        query_key=(0.0, 0.0),
        entry=_entry(1, key=(0.0, 0.0)),
    )
    pending = _step(
        controller,
        first.state,
        query_key=(0.0, 0.0),
        entry=_entry(2, key=(0.0, 0.0)),
    )
    result = controller.settle(
        pending.state,
        _feedback(pending.state, 1.0, used=used, available=available),
    )
    assert bool(result.diagnostics.transaction_applied)
    assert not bool(result.diagnostics.learning_eligible)
    assert not bool(result.state.pending.available)
    assert int(result.state.feedback_count) == 1
    assert int(result.state.learned_feedback_count) == 0
    np.testing.assert_array_equal(
        result.state.admission_weights, pending.state.admission_weights
    )
    np.testing.assert_array_equal(
        result.state.memory.entries.utilities,
        pending.state.memory.entries.utilities,
    )


def test_negative_feedback_learns_admission_rejection_and_eviction_retention() -> None:
    controller = LearnedExperientialMemoryController(_config(capacity=2))
    first = _step(
        controller,
        controller.init(),
        query_key=(0.0, 0.0),
        entry=_entry(10, key=(0.0, 0.0)),
    )
    pending = _step(
        controller,
        first.state,
        query_key=(0.0, 0.0),
        entry=_entry(11, key=(2.0, 0.0)),
    )
    learned = controller.settle(pending.state, _feedback(pending.state, -1.0))
    assert bool(learned.diagnostics.learning_eligible)
    assert bool(learned.diagnostics.admission_updated)
    assert int(learned.diagnostics.retention_rows_updated) == 1
    assert int(learned.state.nonpositive_feedback_count) == 1
    assert float(learned.state.memory.entries.utilities[0]) == pytest.approx(0.0)
    assert float(learned.state.memory.entries.utilities[1]) == pytest.approx(0.5)
    assert float(learned.state.admission_weights[0]) < 0.0

    third = _step(
        controller,
        learned.state,
        query_key=(0.0, 0.0),
        entry=_entry(12, key=(3.0, 0.0)),
    )
    assert bool(third.fixed_store_retrieval.accepted)
    assert not bool(third.retrieval.accepted)
    assert not bool(third.state.pending.available)
    assert bool(third.evicted)
    assert int(third.evicted_provenance_id) == 10
    assert int(third.state.memory.accepted_query_count) == 1


def test_positive_feedback_protects_a_live_neighbor_from_next_eviction() -> None:
    controller = LearnedExperientialMemoryController(_config(capacity=2))
    first = _step(
        controller,
        controller.init(),
        query_key=(0.0, 0.0),
        entry=_entry(10, key=(0.0, 0.0)),
    )
    pending = _step(
        controller,
        first.state,
        query_key=(0.0, 0.0),
        entry=_entry(11, key=(2.0, 0.0)),
    )
    learned = controller.settle(pending.state, _feedback(pending.state, 1.0))
    assert float(learned.state.memory.entries.utilities[0]) == pytest.approx(1.0)
    assert float(learned.state.memory.entries.utilities[1]) == pytest.approx(0.5)
    third = _step(
        controller,
        learned.state,
        query_key=(0.0, 0.0),
        entry=_entry(12, key=(3.0, 0.0)),
    )
    assert bool(third.evicted)
    assert int(third.evicted_provenance_id) == 11
    assert 10 in set(np.asarray(third.state.memory.entries.provenance_ids).tolist())


def test_slot_reuse_prevents_feedback_from_updating_a_different_exemplar() -> None:
    controller = LearnedExperientialMemoryController(_config(capacity=1))
    first = _step(
        controller,
        controller.init(),
        query_key=(0.0, 0.0),
        entry=_entry(10, key=(0.0, 0.0)),
    )
    # Query row 10, then overwrite that only slot with row 11 before feedback.
    pending = _step(
        controller,
        first.state,
        query_key=(0.0, 0.0),
        entry=_entry(11, key=(1.0, 0.0)),
    )
    assert int(pending.evicted_provenance_id) == 10
    settled = controller.settle(pending.state, _feedback(pending.state, 1.0))
    assert bool(settled.diagnostics.admission_updated)
    assert int(settled.diagnostics.retention_rows_updated) == 0
    assert int(settled.state.memory.entries.provenance_ids[0]) == 11
    assert float(settled.state.memory.entries.utilities[0]) == pytest.approx(0.5)


def test_corruption_and_nonfinite_feedback_fail_closed() -> None:
    controller = LearnedExperientialMemoryController(_config())
    state = controller.init()
    corrupt = dataclasses.replace(
        state,
        admission_weights=state.admission_weights.at[0].set(jnp.nan),
    )
    result = _step(
        controller,
        corrupt,
        query_key=(0.0, 0.0),
        entry=_entry(1, key=(0.0, 0.0)),
    )
    assert not bool(result.diagnostics.source_state_valid)
    assert not bool(result.diagnostics.transaction_applied)
    _assert_tree_equal(result.state, corrupt)

    clock_tamper = dataclasses.replace(
        state,
        transaction_words=jnp.asarray((0, 1), dtype=jnp.uint32),
    )
    assert not bool(controller.state_valid(clock_tamper))

    first = _step(
        controller,
        state,
        query_key=(0.0, 0.0),
        entry=_entry(1, key=(0.0, 0.0)),
    )
    pending = _step(
        controller,
        first.state,
        query_key=(0.0, 0.0),
        entry=_entry(2, key=(0.0, 0.0)),
    )
    invalid_feedback = dataclasses.replace(
        _feedback(pending.state, 1.0), counterfactual_delta=jnp.asarray(jnp.nan)
    )
    rejected = controller.settle(pending.state, invalid_feedback)
    assert not bool(rejected.diagnostics.feedback_valid)
    _assert_tree_equal(rejected.state, pending.state)


def test_step_and_feedback_are_eager_jit_and_scan_compatible() -> None:
    controller = LearnedExperientialMemoryController(_config())
    state = controller.init()
    entry = _entry(1, key=(0.0, 0.0))
    args = (
        state,
        jnp.asarray((0.0, 0.0), dtype=jnp.float32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.1, dtype=jnp.float32),
        jnp.asarray(True, dtype=jnp.bool_),
        entry,
    )
    eager = controller._step_jit(*args)
    compiled = jax.jit(controller._step_jit)(*args)
    _assert_tree_equal(eager, compiled)

    def body(carry: Any, provenance: Any) -> tuple[Any, Any]:
        dynamic_entry = dataclasses.replace(
            entry, provenance_id=provenance.astype(jnp.int32)
        )
        result = controller._step_jit(
            carry,
            args[1],
            args[2],
            args[3],
            args[4],
            dynamic_entry,
        )
        return result.state, result.diagnostics.transaction_applied

    scan_state, applied = jax.lax.scan(
        body, state, jnp.asarray((1,), dtype=jnp.int32)
    )
    assert bool(applied[0])
    _assert_tree_equal(scan_state, eager.state)

    pending = controller._step_jit(
        eager.state,
        args[1],
        args[2],
        args[3],
        args[4],
        dataclasses.replace(entry, provenance_id=jnp.asarray(2, dtype=jnp.int32)),
    )
    feedback = _feedback(pending.state, 1.0)
    eager_feedback = controller._settle_jit(pending.state, feedback)
    compiled_feedback = jax.jit(controller._settle_jit)(pending.state, feedback)
    _assert_tree_equal(eager_feedback, compiled_feedback)


def test_checkpoint_roundtrip_and_metadata_tamper_rejection(tmp_path: Path) -> None:
    controller = LearnedExperientialMemoryController(_config())
    state = _step(
        controller,
        controller.init(),
        query_key=(0.0, 0.0),
        entry=_entry(1, key=(0.0, 0.0)),
    ).state
    path = tmp_path / "learned-memory"
    save_learned_experiential_memory_checkpoint(controller, state, path)
    restored_controller, restored = load_learned_experiential_memory_checkpoint(path)
    assert restored_controller.config == controller.config
    _assert_tree_equal(restored, state)

    from alberta_framework.core.checkpoints import load_checkpoint_metadata

    metadata = load_checkpoint_metadata(path)
    assert metadata["schema"] == LEARNED_EXPERIENTIAL_MEMORY_CHECKPOINT_SCHEMA
    tampered = copy.deepcopy(metadata)
    tampered["scientific_promotion_allowed"] = True
    tampered_path = tmp_path / "tampered-learned-memory"
    save_checkpoint(state, tampered_path, metadata=tampered)
    with pytest.raises(ValueError):
        load_learned_experiential_memory_checkpoint(tampered_path)
