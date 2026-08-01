# mypy: disable-error-code="arg-type,attr-defined,call-arg"
"""Exact mechanism tests for fixed-budget dual replay memory."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import fields, replace
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.dual_replay import (
    DUAL_REPLAY_CHECKPOINT_SCHEMA,
    DUAL_REPLAY_CONFIG_SCHEMA,
    LONG_TERM_STRATUM,
    MECHANISM_STATUS,
    SHORT_TERM_STRATUM,
    DualReplayConfig,
    DualReplayMemory,
    ReplayOutcome,
    ReplayPrediction,
    reservoir_selection,
)

pytestmark = pytest.mark.unit

_INT32_MAX = 2_147_483_647


def _config(**overrides: Any) -> DualReplayConfig:
    values: dict[str, Any] = {
        "total_capacity": 6,
        "short_term_capacity": 3,
        "observation_dim": 2,
        "action_dim": 2,
        "short_term_sample_size": 2,
        "long_term_sample_size": 2,
        "long_term_policy": "reservoir",
    }
    values.update(overrides)
    return DualReplayConfig(**values)


def _prediction(
    provenance: int,
    *,
    observation: tuple[float, float] | None = None,
    version: int = 0,
    epistemic: float = 0.5,
    epistemic_available: bool = True,
    aleatoric: float = 0.1,
    aleatoric_available: bool = True,
    value_target: float | None = None,
    valid: bool = True,
) -> ReplayPrediction:
    obs = observation if observation is not None else (float(provenance), float(provenance + 1))
    return ReplayPrediction(
        observation=jnp.asarray(obs, dtype=jnp.float32),
        action=jnp.asarray([float(provenance % 2), 1.0], dtype=jnp.float32),
        old_behavior_probability=jnp.asarray(0.25, dtype=jnp.float32),
        old_behavior_probability_available=jnp.asarray(True),
        old_behavior_logit=jnp.asarray(-1.25, dtype=jnp.float32),
        old_behavior_logit_available=jnp.asarray(True),
        old_value_target=jnp.asarray(
            float(provenance) if value_target is None else value_target,
            dtype=jnp.float32,
        ),
        old_value_target_available=jnp.asarray(True),
        epistemic_surprise=jnp.asarray(
            epistemic if epistemic_available else 0.0,
            dtype=jnp.float32,
        ),
        epistemic_surprise_available=jnp.asarray(epistemic_available),
        aleatoric_uncertainty=jnp.asarray(
            aleatoric if aleatoric_available else 0.0,
            dtype=jnp.float32,
        ),
        aleatoric_uncertainty_available=jnp.asarray(aleatoric_available),
        representation_version=jnp.asarray(version, dtype=jnp.int32),
        provenance_id=jnp.asarray(provenance, dtype=jnp.int32),
        source_id=jnp.asarray(7, dtype=jnp.int32),
        valid=jnp.asarray(valid),
    )


def _outcome(
    provenance: int,
    *,
    terminated: bool = False,
    truncated: bool = False,
    discount: float | None = None,
    progress: float = 0.5,
    progress_available: bool = True,
    safety_cost: float = 0.0,
    safety_available: bool = True,
    valid: bool = True,
) -> ReplayOutcome:
    return ReplayOutcome(
        next_observation=jnp.asarray(
            [float(provenance + 1), float(provenance + 2)], dtype=jnp.float32
        ),
        reward=jnp.asarray(float(provenance) / 10.0, dtype=jnp.float32),
        discount=jnp.asarray(
            (0.0 if terminated else 0.9) if discount is None else discount,
            dtype=jnp.float32,
        ),
        terminated=jnp.asarray(terminated),
        truncated=jnp.asarray(truncated),
        learning_progress=jnp.asarray(
            progress if progress_available else 0.0,
            dtype=jnp.float32,
        ),
        learning_progress_available=jnp.asarray(progress_available),
        safety_cost=jnp.asarray(
            safety_cost if safety_available else 0.0,
            dtype=jnp.float32,
        ),
        safety_cost_available=jnp.asarray(safety_available),
        valid=jnp.asarray(valid),
    )


def _record_range(
    memory: DualReplayMemory,
    state: Any,
    start: int,
    stop: int,
    *,
    versions: list[int] | None = None,
) -> Any:
    for position, provenance in enumerate(range(start, stop)):
        version = versions[position] if versions is not None else 0
        state = memory.record(
            state,
            _prediction(provenance, version=version),
            _outcome(provenance),
        ).state
    return state


def _assert_trees_equal(left: object, right: object) -> None:
    left_leaves = jax.tree.leaves(left)
    right_leaves = jax.tree.leaves(right)
    assert len(left_leaves) == len(right_leaves)
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        np.testing.assert_array_equal(np.asarray(left_leaf), np.asarray(right_leaf))


def _digest(payload: object) -> str:
    encoded = json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def test_config_roundtrip_is_versioned_mechanism_only_and_strict() -> None:
    config = _config(long_term_policy="calibrated", aleatoric_control="downweight")
    payload = json.loads(json.dumps(config.to_config()))
    restored = DualReplayConfig.from_config(payload)

    assert restored == config
    assert payload["schema"] == DUAL_REPLAY_CONFIG_SCHEMA
    assert payload["mechanism_status"] == MECHANISM_STATUS
    memory_payload = DualReplayMemory(config).to_config()
    assert DualReplayMemory.from_config(memory_payload).config == config

    with pytest.raises(ValueError, match="fields"):
        DualReplayConfig.from_config({**payload, "unknown": 1})
    with pytest.raises(ValueError, match="mechanism-only"):
        DualReplayConfig.from_config({**payload, "mechanism_status": "scientific"})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("total_capacity", 1),
        ("short_term_capacity", 6),
        ("short_term_capacity", 0),
        ("observation_dim", 0),
        ("action_dim", True),
        ("short_term_sample_size", 4),
        ("long_term_sample_size", 4),
        ("long_term_policy", "priority-error"),
        ("aleatoric_control", "ignore"),
        ("max_representation_lag", -1),
        ("surprise_scale", 0.0),
        ("coverage_weight", float("nan")),
        ("calibrated_priority_threshold", 1.1),
        ("calibrated_replacement_margin", -0.1),
        ("max_aleatoric_uncertainty", -1.0),
    ],
)
def test_config_rejects_invalid_allocation_or_calibration(field: str, value: Any) -> None:
    with pytest.raises(ValueError):
        DualReplayMemory(replace(_config(), **{field: value}))


def test_fixed_total_capacity_and_resource_accounting_are_exact() -> None:
    memory = DualReplayMemory(_config(total_capacity=7, short_term_capacity=3))
    state = memory.init(jr.key(3))
    exact_bytes = sum(
        int(leaf.size) * int(leaf.dtype.itemsize) for leaf in jax.tree.leaves(state)
    )
    short_slot_bytes = sum(
        int(leaf.size) * int(leaf.dtype.itemsize)
        for leaf in jax.tree.leaves(state.short_term)
    ) // 3
    long_slot_bytes = sum(
        int(leaf.size) * int(leaf.dtype.itemsize)
        for leaf in jax.tree.leaves(state.long_term)
    ) // 4
    accounting = memory.accounting(state)

    assert memory.config.long_term_capacity == 4
    assert int(accounting.total_capacity) == 7
    assert int(accounting.short_term_capacity) == 3
    assert int(accounting.long_term_capacity) == 4
    assert memory.slot_bytes == short_slot_bytes == long_slot_bytes
    assert memory.persistent_bytes == exact_bytes == int(accounting.persistent_bytes)
    assert state.short_term.observations.shape == (3, 2)
    assert state.long_term.observations.shape == (4, 2)

    with pytest.raises(ValueError, match="max_persistent_bytes"):
        DualReplayMemory(replace(memory.config, max_persistent_bytes=exact_bytes - 1))


def test_record_is_prediction_before_outcome_and_fifo_eviction_is_auditable() -> None:
    memory = DualReplayMemory(_config(total_capacity=5, short_term_capacity=2))
    prediction_fields = {field.name for field in fields(ReplayPrediction)}
    outcome_fields = {field.name for field in fields(ReplayOutcome)}
    assert {
        "reward",
        "discount",
        "terminated",
        "truncated",
        "next_observation",
    }.isdisjoint(prediction_fields)
    assert {
        "old_behavior_probability",
        "old_behavior_logit",
        "old_value_target",
    }.isdisjoint(outcome_fields)
    state = memory.init(jr.key(0))
    first = memory.record(state, _prediction(10), _outcome(10))

    assert bool(first.wrote_short_term)
    assert int(first.short_term_slot) == 0
    assert not bool(first.short_term_evicted)
    assert int(first.state.short_term.provenance_ids[0]) == 10
    assert float(first.state.short_term.old_behavior_probabilities[0]) == pytest.approx(0.25)
    assert float(first.state.short_term.rewards[0]) == pytest.approx(1.0)
    assert int(state.short_term_size) == 0

    state = memory.record(first.state, _prediction(11), _outcome(11)).state
    third = memory.record(state, _prediction(12), _outcome(12))
    assert bool(third.short_term_evicted)
    assert int(third.short_term_slot) == 0
    assert int(third.short_term_evicted_provenance_id) == 10
    assert int(third.state.short_term.provenance_ids[0]) == 12
    assert int(third.state.short_term.eviction_provenance_ids[0]) == 10
    assert int(third.state.short_term.provenance_ids[1]) == 11
    assert int(third.state.short_term_head) == 1


def test_all_transition_fields_and_availability_survive_storage_and_sampling() -> None:
    memory = DualReplayMemory(_config(max_representation_lag=0))
    prediction = _prediction(20, version=3, epistemic=0.7, aleatoric=0.2).replace(
        old_behavior_probability=jnp.asarray(0.0, dtype=jnp.float32),
        old_behavior_probability_available=jnp.asarray(False),
        old_behavior_logit=jnp.asarray(2.5, dtype=jnp.float32),
        old_value_target=jnp.asarray(-4.0, dtype=jnp.float32),
    )
    outcome = _outcome(20, progress_available=False, safety_cost=2.0).replace(
        discount=jnp.asarray(0.0, dtype=jnp.float32),
        terminated=jnp.asarray(True),
    )
    state = memory.record(memory.init(jr.key(30)), prediction, outcome).state

    for entries in (state.short_term, state.long_term):
        assert bool(entries.valid[0])
        np.testing.assert_array_equal(entries.observations[0], prediction.observation)
        np.testing.assert_array_equal(entries.next_observations[0], outcome.next_observation)
        np.testing.assert_array_equal(entries.actions[0], prediction.action)
        assert float(entries.rewards[0]) == 2.0
        assert float(entries.discounts[0]) == 0.0
        assert bool(entries.terminated[0])
        assert not bool(entries.truncated[0])
        assert not bool(entries.old_behavior_probability_available[0])
        assert float(entries.old_behavior_probabilities[0]) == 0.0
        assert float(entries.old_behavior_logits[0]) == 2.5
        assert float(entries.old_value_targets[0]) == -4.0
        assert float(entries.epistemic_surprises[0]) == pytest.approx(0.7)
        assert float(entries.aleatoric_uncertainties[0]) == pytest.approx(0.2)
        assert not bool(entries.learning_progress_available[0])
        assert float(entries.learning_progress[0]) == 0.0
        assert float(entries.safety_costs[0]) == 2.0
        assert int(entries.representation_versions[0]) == 3
        assert int(entries.provenance_ids[0]) == 20
        assert int(entries.source_ids[0]) == 7
        assert int(entries.eviction_provenance_ids[0]) == -1

    sample = memory.sample(state, jnp.asarray(3, dtype=jnp.int32))
    assert int(jnp.sum(sample.batch.valid)) == 2
    assert np.all(np.asarray(sample.batch.entries.provenance_ids)[sample.batch.valid] == 20)
    assert np.all(np.asarray(sample.batch.entries.safety_costs)[sample.batch.valid] == 2.0)


@pytest.mark.parametrize(
    ("terminated", "truncated", "discount"),
    [
        (False, False, 0.9),
        (False, True, 0.9),
        (True, False, 0.0),
        (True, True, 0.0),
    ],
)
def test_termination_truncation_truth_table_is_stored_without_conflation(
    terminated: bool,
    truncated: bool,
    discount: float,
) -> None:
    memory = DualReplayMemory(_config())
    result = memory.record(
        memory.init(jr.key(31)),
        _prediction(1),
        _outcome(
            1,
            terminated=terminated,
            truncated=truncated,
            discount=discount,
        ),
    )

    assert bool(result.input_valid)
    assert bool(result.wrote_short_term)
    assert bool(result.state.short_term.terminated[0]) is terminated
    assert bool(result.state.short_term.truncated[0]) is truncated
    assert float(result.state.short_term.discounts[0]) == pytest.approx(discount)


@pytest.mark.parametrize(
    ("terminated", "truncated", "discount"),
    [
        (False, False, 0.0),
        (False, True, 0.0),
        (True, False, 0.9),
        (True, True, 0.9),
    ],
)
def test_discount_terminal_mismatch_is_rejected_atomically(
    terminated: bool,
    truncated: bool,
    discount: float,
) -> None:
    memory = DualReplayMemory(_config())
    state = memory.init(jr.key(32))
    result = memory.record(
        state,
        _prediction(1),
        _outcome(
            1,
            terminated=terminated,
            truncated=truncated,
            discount=discount,
        ),
    )

    assert not bool(result.input_valid)
    assert not bool(result.wrote_short_term)
    _assert_trees_equal(result.state.short_term, state.short_term)
    _assert_trees_equal(result.state.long_term, state.long_term)


def test_available_dispatched_action_probability_must_be_strictly_positive() -> None:
    memory = DualReplayMemory(_config())
    state = memory.init(jr.key(33))
    zero_probability = _prediction(1).replace(
        old_behavior_probability=jnp.asarray(0.0, dtype=jnp.float32)
    )
    result = memory.record(state, zero_probability, _outcome(1))

    assert not bool(result.input_valid)
    assert not bool(result.wrote_short_term)
    assert int(result.state.rejected_transition_count) == 1


def test_reservoir_selection_contract_and_lifetime_calculations_are_exact() -> None:
    capacity = 3
    invalid = reservoir_selection(
        jr.key(4), jnp.asarray(0, dtype=jnp.int32), capacity
    )
    assert not bool(invalid.selected)
    assert not bool(invalid.draw_available)
    assert int(invalid.slot) == -1

    fill = reservoir_selection(
        jr.key(4), jnp.asarray(3, dtype=jnp.int32), capacity
    )
    assert bool(fill.selected)
    assert int(fill.slot) == 2
    assert not bool(fill.draw_available)
    assert int(fill.draw) == -1

    key = jr.key(8)
    expected_draw = int(jr.randint(key, shape=(), minval=0, maxval=4, dtype=jnp.int32))
    replacement = reservoir_selection(key, jnp.asarray(4, dtype=jnp.int32), capacity)
    assert int(replacement.population_size) == 4
    assert bool(replacement.draw_available)
    assert int(replacement.draw) == expected_draw
    assert bool(replacement.selected) is (expected_draw < capacity)
    assert int(replacement.slot) == (expected_draw if expected_draw < capacity else -1)

    memory = DualReplayMemory(_config(total_capacity=5, short_term_capacity=2))
    state = memory.init(jr.key(17))
    for candidate in range(1, 12):
        result = memory.record(state, _prediction(candidate), _outcome(candidate))
        assert int(result.long_term_candidate_number) == candidate
        assert int(result.reservoir_population_size) == candidate
        if candidate <= 3:
            assert bool(result.wrote_long_term)
            assert int(result.long_term_slot) == candidate - 1
            assert not bool(result.reservoir_draw_available)
        else:
            draw = int(result.reservoir_draw)
            assert bool(result.wrote_long_term) is (draw < 3)
            assert int(result.long_term_slot) == (draw if draw < 3 else -1)
        state = result.state
    memory.validate_state(state)
    assert int(state.long_term_candidate_count) == 11
    assert int(state.long_term_write_count + state.long_term_rejection_count) == 11


def test_calibrated_priority_uses_surprise_coverage_and_progress_exactly() -> None:
    memory = DualReplayMemory(
        _config(
            long_term_policy="calibrated",
            surprise_scale=2.0,
            coverage_scale=10.0,
            progress_scale=4.0,
            calibrated_priority_threshold=0.6,
            max_aleatoric_uncertainty=0.5,
        )
    )
    result = memory.record(
        memory.init(jr.key(1)),
        _prediction(1, observation=(0.0, 0.0), epistemic=1.0, aleatoric=0.1),
        _outcome(1, progress=2.0),
    )

    assert bool(result.long_term_priority_available)
    assert float(result.surprise_component) == pytest.approx(0.5)
    assert float(result.coverage_component) == pytest.approx(1.0)
    assert float(result.progress_component) == pytest.approx(0.5)
    assert float(result.long_term_priority) == pytest.approx(2.0 / 3.0)
    assert bool(result.aleatoric_control_available)
    assert bool(result.aleatoric_control_passed)
    assert bool(result.wrote_long_term)


def test_calibrated_coverage_and_long_term_eviction_provenance_are_explicit() -> None:
    memory = DualReplayMemory(
        _config(
            total_capacity=4,
            short_term_capacity=2,
            long_term_policy="calibrated",
            calibrated_priority_threshold=0.0,
            surprise_scale=1.0,
            coverage_scale=1.0,
            progress_scale=1.0,
        )
    )
    state = memory.init(jr.key(4))
    first = memory.record(
        state,
        _prediction(1, observation=(0.0, 0.0), epistemic=0.5),
        _outcome(1, progress=0.5),
    )
    duplicate = memory.record(
        first.state,
        _prediction(2, observation=(0.0, 0.0), epistemic=0.5),
        _outcome(2, progress=0.5),
    )
    replacement = memory.record(
        duplicate.state,
        _prediction(3, observation=(10.0, 10.0), epistemic=1.0),
        _outcome(3, progress=1.0),
    )

    assert float(first.coverage_component) == pytest.approx(1.0)
    assert float(duplicate.coverage_component) == pytest.approx(0.0)
    assert bool(replacement.long_term_evicted)
    assert int(replacement.long_term_evicted_provenance_id) == 2
    slot = int(replacement.long_term_slot)
    assert int(replacement.state.long_term.provenance_ids[slot]) == 3
    assert int(replacement.state.long_term.eviction_provenance_ids[slot]) == 2
    assert int(replacement.state.long_term_eviction_count) == 1


def test_raw_value_error_proxy_alone_cannot_create_long_term_priority() -> None:
    memory = DualReplayMemory(
        _config(long_term_policy="calibrated", calibrated_priority_threshold=0.0)
    )
    result = memory.record(
        memory.init(jr.key(2)),
        _prediction(
            1,
            value_target=1.0e20,
            epistemic_available=False,
            aleatoric=0.0,
        ),
        _outcome(1, progress_available=False),
    )

    assert bool(result.wrote_short_term)
    assert not bool(result.long_term_priority_available)
    assert float(result.long_term_priority) == 0.0
    assert not bool(result.wrote_long_term)
    assert int(result.state.long_term_rejection_count) == 1


def test_noisy_tv_aleatoric_veto_blocks_high_surprise_candidate() -> None:
    memory = DualReplayMemory(
        _config(
            long_term_policy="calibrated",
            calibrated_priority_threshold=0.1,
            max_aleatoric_uncertainty=0.2,
            aleatoric_control="veto",
        )
    )
    result = memory.record(
        memory.init(jr.key(2)),
        _prediction(1, epistemic=100.0, aleatoric=100.0),
        _outcome(1, progress=100.0),
    )

    assert float(result.long_term_priority) == pytest.approx(1.0)
    assert bool(result.aleatoric_control_available)
    assert not bool(result.aleatoric_control_passed)
    assert not bool(result.wrote_long_term)
    assert bool(result.wrote_short_term)


def test_aleatoric_downweight_is_an_explicit_alternative_control() -> None:
    memory = DualReplayMemory(
        _config(
            long_term_policy="calibrated",
            calibrated_priority_threshold=0.5,
            aleatoric_control="downweight",
            aleatoric_downweight_scale=1.0,
        )
    )
    result = memory.record(
        memory.init(jr.key(2)),
        _prediction(1, epistemic=1.0, aleatoric=9.0),
        _outcome(1, progress=1.0),
    )

    assert bool(result.aleatoric_control_passed)
    assert float(result.long_term_priority) == pytest.approx(0.1)
    assert not bool(result.wrote_long_term)


def test_invalid_transition_is_atomic_and_explicitly_rejected() -> None:
    memory = DualReplayMemory(_config())
    state = memory.init(jr.key(5))
    dishonest = _prediction(1).replace(
        epistemic_surprise=jnp.asarray(1.0, dtype=jnp.float32),
        epistemic_surprise_available=jnp.asarray(False),
    )
    result = memory.record(state, dishonest, _outcome(1))

    assert bool(result.state_valid)
    assert not bool(result.input_valid)
    assert bool(result.counter_available)
    assert not bool(result.wrote_short_term)
    assert not bool(result.wrote_long_term)
    _assert_trees_equal(result.state.short_term, state.short_term)
    _assert_trees_equal(result.state.long_term, state.long_term)
    np.testing.assert_array_equal(result.state.rng_key_data, state.rng_key_data)
    assert int(result.state.write_attempt_count) == 1
    assert int(result.state.rejected_transition_count) == 1


def test_corrupt_state_and_exhausted_counters_fail_closed_exactly() -> None:
    memory = DualReplayMemory(_config())
    state = memory.init(jr.key(6))
    corrupt = state.replace(
        short_term=state.short_term.replace(
            rewards=state.short_term.rewards.at[0].set(jnp.nan)
        )
    )
    corrupt_result = memory.record(corrupt, _prediction(1), _outcome(1))
    assert not bool(corrupt_result.state_valid)
    _assert_trees_equal(corrupt_result.state, corrupt)

    exhausted = state.replace(
        write_attempt_count=jnp.asarray(_INT32_MAX, dtype=jnp.int32),
        rejected_transition_count=jnp.asarray(_INT32_MAX, dtype=jnp.int32),
    )
    memory.validate_state(exhausted)
    exhausted_result = memory.record(exhausted, _prediction(1), _outcome(1))
    assert not bool(exhausted_result.counter_available)
    _assert_trees_equal(exhausted_result.state, exhausted)

    sample_exhausted = state.replace(sample_count=jnp.asarray(_INT32_MAX, dtype=jnp.int32))
    memory.validate_state(sample_exhausted)
    sample = memory.sample(sample_exhausted, jnp.asarray(0, dtype=jnp.int32))
    assert not bool(sample.counter_available)
    assert not bool(jnp.any(sample.batch.valid))
    _assert_trees_equal(sample.state, sample_exhausted)

    invalid_version = memory.sample(state, jnp.asarray(-1, dtype=jnp.int32))
    assert not bool(invalid_version.representation_version_valid)
    assert not bool(jnp.any(invalid_version.batch.valid))
    _assert_trees_equal(invalid_version.state, state)


def test_stratified_sampling_filters_stale_and_future_representations() -> None:
    memory = DualReplayMemory(
        _config(
            total_capacity=6,
            short_term_capacity=3,
            short_term_sample_size=3,
            long_term_sample_size=3,
            max_representation_lag=0,
        )
    )
    state = _record_range(
        memory,
        memory.init(jr.key(7)),
        0,
        3,
        versions=[0, 1, 2],
    )
    current = memory.sample(state, jnp.asarray(2, dtype=jnp.int32))

    assert int(current.stale_short_term_count) == 2
    assert int(current.stale_long_term_count) == 2
    assert int(current.future_short_term_count) == 0
    assert int(current.future_long_term_count) == 0
    assert int(current.eligible_short_term_count) == 1
    assert int(current.eligible_long_term_count) == 1
    assert bool(current.stale_detected)
    assert int(jnp.sum(current.batch.valid)) == 2
    np.testing.assert_array_equal(
        current.batch.entries.representation_versions[current.batch.valid],
        np.asarray([2, 2], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        current.batch.stratum,
        np.asarray([SHORT_TERM_STRATUM] * 3 + [LONG_TERM_STRATUM] * 3),
    )

    earlier = memory.sample(state, jnp.asarray(1, dtype=jnp.int32))
    assert int(earlier.stale_short_term_count) == 1
    assert int(earlier.stale_long_term_count) == 1
    assert int(earlier.future_short_term_count) == 1
    assert int(earlier.future_long_term_count) == 1
    sampled_versions = np.asarray(earlier.batch.entries.representation_versions)[
        earlier.batch.valid
    ]
    assert np.all(sampled_versions == 1)


def test_stratified_sampling_is_deterministic_bounded_and_without_replacement() -> None:
    memory = DualReplayMemory(_config())
    state = _record_range(memory, memory.init(jr.key(9)), 0, 6)
    first = memory.sample(state, jnp.asarray(0, dtype=jnp.int32))
    second = memory.sample(state, jnp.asarray(0, dtype=jnp.int32))

    _assert_trees_equal(first, second)
    assert first.batch.valid.shape == (4,)
    assert int(first.state.sample_count) == 1
    assert len(set(np.asarray(first.batch.slot[:2]).tolist())) == 2
    assert len(set(np.asarray(first.batch.slot[2:]).tolist())) == 2
    assert not np.array_equal(first.state.rng_key_data, state.rng_key_data)


def _stack(items: list[Any]) -> Any:
    return jax.tree.map(lambda *leaves: jnp.stack(leaves), *items)


def test_eager_jit_and_scan_record_and_sample_paths_are_bitwise_equal() -> None:
    memory = DualReplayMemory(_config(total_capacity=5, short_term_capacity=2))
    predictions = [_prediction(index) for index in range(8)]
    predictions[3] = predictions[3].replace(valid=jnp.asarray(False))
    outcomes = [_outcome(index) for index in range(8)]
    initial = memory.init(jr.key(11))

    eager = initial
    with jax.disable_jit():
        for prediction, outcome in zip(predictions, outcomes, strict=True):
            eager = memory.record(eager, prediction, outcome).state

    step_jit = jax.jit(memory.record)
    compiled = initial
    for prediction, outcome in zip(predictions, outcomes, strict=True):
        compiled = step_jit(compiled, prediction, outcome).state

    def scan_step(state: Any, inputs: tuple[Any, Any]) -> tuple[Any, jax.Array]:
        prediction, outcome = inputs
        result = memory.record(state, prediction, outcome)
        return result.state, result.wrote_long_term

    scanned, _ = jax.lax.scan(scan_step, initial, (_stack(predictions), _stack(outcomes)))
    _assert_trees_equal(eager, compiled)
    _assert_trees_equal(eager, scanned)
    assert int(eager.accepted_transition_count) == 7
    assert int(eager.rejected_transition_count) == 1

    with jax.disable_jit():
        eager_sample = memory.sample(eager, jnp.asarray(0, dtype=jnp.int32))
    compiled_sample = jax.jit(memory.sample)(compiled, jnp.asarray(0, dtype=jnp.int32))
    _assert_trees_equal(eager_sample, compiled_sample)

    eager_sample_state = eager
    eager_batches = []
    with jax.disable_jit():
        for _ in range(3):
            sample = memory.sample(
                eager_sample_state,
                jnp.asarray(0, dtype=jnp.int32),
            )
            eager_sample_state = sample.state
            eager_batches.append(sample.batch)

    def sample_scan_step(state: Any, version: jax.Array) -> tuple[Any, Any]:
        sample = memory.sample(state, version)
        return sample.state, sample.batch

    scan_sample_state, scan_batches = jax.lax.scan(
        sample_scan_step,
        eager,
        jnp.zeros((3,), dtype=jnp.int32),
    )
    _assert_trees_equal(eager_sample_state, scan_sample_state)
    _assert_trees_equal(_stack(eager_batches), scan_batches)


def test_checkpoint_resume_is_bitwise_equal_and_digest_bound() -> None:
    memory = DualReplayMemory(_config(total_capacity=5, short_term_capacity=2))
    initial = memory.init(jr.key(13))
    uninterrupted = _record_range(memory, initial, 0, 10)
    uninterrupted = memory.sample(uninterrupted, jnp.asarray(0, dtype=jnp.int32)).state

    interrupted = _record_range(memory, initial, 0, 5)
    payload = json.loads(json.dumps(memory.checkpoint_payload(interrupted)))
    assert payload["schema"] == DUAL_REPLAY_CHECKPOINT_SCHEMA
    restored_memory, restored = DualReplayMemory.from_checkpoint_payload(payload)
    resumed = _record_range(restored_memory, restored, 5, 10)
    resumed = restored_memory.sample(resumed, jnp.asarray(0, dtype=jnp.int32)).state

    _assert_trees_equal(uninterrupted, resumed)
    assert memory.checkpoint_payload(uninterrupted) == restored_memory.checkpoint_payload(resumed)

    tampered = copy.deepcopy(payload)
    tampered["state"]["short_term"]["rewards"][0] = 123.0
    with pytest.raises(ValueError, match="state digest"):
        DualReplayMemory.from_checkpoint_payload(tampered)

    config_tampered = copy.deepcopy(payload)
    config_tampered["memory"]["config"]["max_representation_lag"] = 5
    with pytest.raises(ValueError, match="config digest"):
        DualReplayMemory.from_checkpoint_payload(config_tampered)

    malformed = copy.deepcopy(payload)
    malformed["state"]["short_term"]["rewards"][0] = "not-a-number"
    malformed["state_digest"] = _digest(malformed["state"])
    with pytest.raises(ValueError, match="JSON real"):
        DualReplayMemory.from_checkpoint_payload(malformed)

    dishonest = copy.deepcopy(payload)
    dishonest["state"]["short_term"]["old_value_targets"][0] = 1.0
    dishonest["state"]["short_term"]["old_value_target_available"][0] = False
    dishonest["state_digest"] = _digest(dishonest["state"])
    with pytest.raises(ValueError, match="dynamic invariants"):
        DualReplayMemory.from_checkpoint_payload(dishonest)


def test_checkpoint_rejects_nonfinite_and_boolean_numeric_corruption() -> None:
    memory = DualReplayMemory(_config())
    state = _record_range(memory, memory.init(jr.key(21)), 0, 1)
    payload = json.loads(json.dumps(memory.checkpoint_payload(state)))

    nonfinite = copy.deepcopy(payload)
    nonfinite["state"]["short_term"]["rewards"][0] = float("nan")
    nonfinite["state_digest"] = hashlib.sha256(
        json.dumps(
            nonfinite["state"],
            allow_nan=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    with pytest.raises((ValueError, TypeError)):
        DualReplayMemory.from_checkpoint_payload(nonfinite)

    boolean = copy.deepcopy(payload)
    boolean["state"]["short_term"]["rewards"][0] = True
    boolean["state_digest"] = _digest(boolean["state"])
    with pytest.raises(ValueError, match="JSON real"):
        DualReplayMemory.from_checkpoint_payload(boolean)
