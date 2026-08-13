"""Exact-lifetime and atomic-transaction tests for the extended UPGD learner."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from alberta_framework.core.checkpoints import save_checkpoint
from alberta_framework.core.upgd import (
    UPGD_CHECKPOINT_SCHEMA,
    UPGD_LIFETIME_COUNTER_DELTA_NBYTES,
    UPGD_LIFETIME_COUNTER_NBYTES,
    UPGD_STATE_SCHEMA,
    UPGD_TRANSACTION_CLOCK_DELTA_NBYTES,
    UPGD_TRANSACTION_CLOCK_NBYTES,
    UPGDLearner,
    UPGDState,
    load_upgd_checkpoint,
    measure_upgd_state_nbytes,
    migrate_legacy_upgd_state,
    save_upgd_checkpoint,
    upgd_lifetime_counter_nbytes,
    upgd_transaction_clock_nbytes,
)

_INT32_MAX = 2**31 - 1
_UINT32_MAX = 2**32 - 1


def _learner(**kwargs: Any) -> UPGDLearner:
    return UPGDLearner(
        n_heads=1,
        hidden_sizes=(2,),
        sparsity=0.0,
        step_size=0.0,
        perturbation_sigma=0.0,
        **kwargs,
    )


def _at_step(state: UPGDState, step: int, *, interval: int) -> UPGDState:
    high, low = divmod(step, 2**32)
    telemetry = min(step, _INT32_MAX)
    return cast(
        UPGDState,
        state.replace(  # type: ignore[attr-defined]
            step_count=jnp.asarray(telemetry, dtype=jnp.int32),
            step_words=jnp.asarray((high, low), dtype=jnp.uint32),
            perturbation_phase=jnp.asarray(step % interval, dtype=jnp.int32),
        ),
    )


def _legacy_mapping(state: UPGDState) -> dict[str, object]:
    return {
        field.name: getattr(state, field.name)
        for field in dataclasses.fields(cast(Any, state))
        if field.name not in {"step_words", "perturbation_phase"}
    }


def test_init_has_authenticated_exact_clock_and_schema() -> None:
    learner = _learner(perturbation_interval=7)
    state = learner.init(3, jr.key(0))

    assert learner.to_config()["state_schema"] == UPGD_STATE_SCHEMA
    assert state.step_count.dtype == jnp.dtype(jnp.int32)
    assert state.step_words.dtype == jnp.dtype(jnp.uint32)
    assert state.step_words.shape == (2,)
    assert int(state.perturbation_phase) == 0

    result = learner.update(
        state,
        jnp.zeros(3, dtype=jnp.float32),
        jnp.zeros(1, dtype=jnp.float32),
    )
    assert bool(result.update_applied)
    assert not bool(result.update_rejected)
    assert bool(result.lifetime_counter_valid)
    assert bool(result.state_valid)
    assert bool(result.candidate_state_valid)
    chex.assert_trees_all_equal(result.pre_step_words, jnp.asarray((0, 0), jnp.uint32))
    chex.assert_trees_all_equal(result.post_step_words, jnp.asarray((0, 1), jnp.uint32))
    assert int(result.state.perturbation_phase) == 1


def test_rollover_saturates_telemetry_and_unit_ages_without_changing_phase() -> None:
    learner = _learner(perturbation_interval=5)
    state = learner.init(3, jr.key(1))
    state = _at_step(state, _UINT32_MAX, interval=5).replace(  # type: ignore[attr-defined]
        unit_ages=(jnp.full((2,), _INT32_MAX, dtype=jnp.int32),)
    )

    result = learner.update(
        state,
        jnp.zeros(3, dtype=jnp.float32),
        jnp.zeros(1, dtype=jnp.float32),
    )

    assert bool(result.update_applied)
    chex.assert_trees_all_equal(
        result.state.step_words,
        jnp.asarray((1, 0), dtype=jnp.uint32),
    )
    assert int(result.state.step_count) == _INT32_MAX
    assert int(result.state.perturbation_phase) == 1
    chex.assert_trees_all_equal(
        result.state.unit_ages[0],
        jnp.full((2,), _INT32_MAX, dtype=jnp.int32),
    )


def test_perturbation_cadence_uses_exact_phase_after_uint32_rollover() -> None:
    learner = UPGDLearner(
        n_heads=1,
        hidden_sizes=(2,),
        sparsity=0.0,
        step_size=0.0,
        utility_decay=0.999,
        perturbation_sigma=1.0,
        perturbation_interval=5,
    )
    initial = learner.init(3, jr.key(2))
    due = _at_step(initial, 2**32 + 4, interval=5)
    not_due = _at_step(initial, 2**32 + 5, interval=5)
    observation = jnp.zeros(3, dtype=jnp.float32)
    target = jnp.zeros(1, dtype=jnp.float32)

    due_result = learner.update(due, observation, target)
    not_due_result = learner.update(not_due, observation, target)

    assert bool(due_result.perturbation_due)
    assert bool(due_result.perturbation_applied)
    assert float(due_result.metrics[3]) > 0.0
    assert not bool(not_due_result.perturbation_due)
    assert not bool(not_due_result.perturbation_applied)
    assert float(not_due_result.metrics[3]) == 0.0


def test_zero_length_ramp_honors_high_word_warmup_boundary() -> None:
    warmup = 2**32 + 3
    learner = UPGDLearner(
        n_heads=1,
        hidden_sizes=(2,),
        sparsity=0.0,
        step_size=0.0,
        utility_decay=0.999,
        perturbation_sigma=1.0,
        perturbation_interval=1,
        perturbation_warmup_steps=warmup,
        perturbation_ramp_steps=0,
    )
    initial = learner.init(3, jr.key(20))
    before = _at_step(initial, warmup - 1, interval=1)
    at_boundary = _at_step(initial, warmup, interval=1)
    observation = jnp.zeros(3, dtype=jnp.float32)
    target = jnp.zeros(1, dtype=jnp.float32)

    before_result = learner.update(before, observation, target)
    boundary_result = learner.update(at_boundary, observation, target)

    assert bool(before_result.perturbation_due)
    assert not bool(before_result.perturbation_applied)
    assert float(before_result.metrics[3]) == 0.0
    assert bool(boundary_result.perturbation_due)
    assert bool(boundary_result.perturbation_applied)
    assert float(boundary_result.metrics[3]) > 0.0


def test_all_ones_capacity_exhaustion_is_atomic_under_eager_jit_and_scan() -> None:
    learner = _learner(perturbation_interval=3)
    initial = learner.init(3, jr.key(3))
    exhausted = initial.replace(  # type: ignore[attr-defined]
        step_count=jnp.asarray(_INT32_MAX, dtype=jnp.int32),
        step_words=jnp.asarray((_UINT32_MAX, _UINT32_MAX), dtype=jnp.uint32),
        perturbation_phase=jnp.asarray(((2**64) - 1) % 3, dtype=jnp.int32),
    )
    observation = jnp.ones(3, dtype=jnp.float32)
    target = jnp.ones(1, dtype=jnp.float32)

    with jax.disable_jit():
        eager = learner.update(exhausted, observation, target)
    compiled = jax.jit(learner.update)(exhausted, observation, target)

    def scan_step(carry: object, _: jax.Array) -> tuple[object, jax.Array]:
        result = learner.update(carry, observation, target)
        return result.state, result.update_rejected

    scanned_state, rejected = jax.lax.scan(
        scan_step,
        exhausted,
        jnp.arange(3, dtype=jnp.int32),
    )

    for result in (eager, compiled):
        assert bool(result.update_rejected)
        assert not bool(result.update_applied)
        assert not bool(result.lifetime_capacity_available)
        chex.assert_trees_all_equal(result.state, exhausted)
        chex.assert_trees_all_equal(result.metrics, jnp.zeros(4, dtype=jnp.float32))
        assert bool(jnp.all(jnp.isnan(result.predictions)))
    chex.assert_trees_all_equal(scanned_state, exhausted)
    assert bool(jnp.all(rejected))


def test_invalid_source_and_nonfinite_candidate_are_atomic_rejections() -> None:
    learner = _learner(perturbation_interval=4)
    state = learner.init(3, jr.key(4))
    bad_source = state.replace(  # type: ignore[attr-defined]
        utilities=(state.utilities[0].at[0, 0].set(jnp.nan),)
    )

    source_result = learner.update(
        bad_source,
        jnp.ones(3, dtype=jnp.float32),
        jnp.ones(1, dtype=jnp.float32),
    )
    candidate_result = learner.update(
        state,
        jnp.full((3,), 3.4e38, dtype=jnp.float32),
        jnp.full((1,), 3.4e38, dtype=jnp.float32),
    )

    assert not bool(source_result.state_valid)
    assert bool(source_result.update_rejected)
    chex.assert_trees_all_equal(source_result.state, bad_source)
    assert bool(candidate_result.state_valid)
    assert not bool(candidate_result.candidate_state_valid)
    assert bool(candidate_result.update_rejected)
    chex.assert_trees_all_equal(candidate_result.state, state)


def test_corrupt_clock_or_phase_is_rejected_and_contract_errors_are_static() -> None:
    learner = _learner(perturbation_interval=7)
    state = learner.init(3, jr.key(5))
    bad_telemetry = state.replace(  # type: ignore[attr-defined]
        step_count=jnp.asarray(1, dtype=jnp.int32)
    )
    bad_phase = state.replace(  # type: ignore[attr-defined]
        perturbation_phase=jnp.asarray(1, dtype=jnp.int32)
    )

    assert bool(
        learner.update(
            bad_telemetry,
            jnp.zeros(3, dtype=jnp.float32),
            jnp.zeros(1, dtype=jnp.float32),
        ).update_rejected
    )
    assert bool(
        learner.update(
            bad_phase,
            jnp.zeros(3, dtype=jnp.float32),
            jnp.zeros(1, dtype=jnp.float32),
        ).update_rejected
    )
    with pytest.raises(TypeError, match="step_words"):
        learner.update(
            state.replace(  # type: ignore[attr-defined]
                step_words=jnp.zeros((2,), dtype=jnp.int32)
            ),
            jnp.zeros(3, dtype=jnp.float32),
            jnp.zeros(1, dtype=jnp.float32),
        )


def test_config_schema_is_exact_and_schedule_ranges_are_bounded() -> None:
    learner = _learner(perturbation_interval=11)
    config = learner.to_config()
    assert UPGDLearner.from_config(config).to_config() == config

    for mutation in ("missing", "extra", "schema"):
        bad = dict(config)
        if mutation == "missing":
            bad.pop("utility_decay")
        elif mutation == "extra":
            bad["unknown"] = 1
        else:
            bad["state_schema"] = "alberta.upgd-state.v1"
        with pytest.raises(ValueError):
            UPGDLearner.from_config(bad)

    with pytest.raises(ValueError, match="int32-safe"):
        _learner(perturbation_interval=2**31)
    with pytest.raises(ValueError, match="uint64-safe"):
        _learner(head_repetition_warmup_steps=2**64)


def test_legacy_state_migration_is_exact_and_rejects_ambiguous_counters() -> None:
    learner = _learner(perturbation_interval=13)
    current = learner.init(3, jr.key(6))
    legacy = _legacy_mapping(current)
    legacy["step_count"] = jnp.asarray(17, dtype=jnp.int32)
    legacy["unit_ages"] = (jnp.asarray((3, 4), dtype=jnp.int32),)
    legacy["unit_replacement_counts"] = jnp.asarray((2.0,), dtype=jnp.float32)

    migrated = migrate_legacy_upgd_state(legacy, perturbation_interval=13)
    chex.assert_trees_all_equal(
        migrated.step_words,
        jnp.asarray((0, 17), dtype=jnp.uint32),
    )
    assert int(migrated.perturbation_phase) == 4
    result = learner.update(
        migrated,
        jnp.zeros(3, dtype=jnp.float32),
        jnp.zeros(1, dtype=jnp.float32),
    )
    assert bool(result.update_applied)

    saturated_step = dict(legacy)
    saturated_step["step_count"] = jnp.asarray(_INT32_MAX, dtype=jnp.int32)
    with pytest.raises(ValueError, match="ambiguous"):
        migrate_legacy_upgd_state(saturated_step, perturbation_interval=13)
    saturated_age = dict(legacy)
    saturated_age["unit_ages"] = (
        jnp.asarray((_INT32_MAX, 0), dtype=jnp.int32),
    )
    with pytest.raises(ValueError, match="ambiguous"):
        migrate_legacy_upgd_state(saturated_age, perturbation_interval=13)
    ambiguous_replacements = dict(legacy)
    ambiguous_replacements["unit_replacement_counts"] = jnp.asarray(
        (float(2**24),),
        dtype=jnp.float32,
    )
    with pytest.raises(ValueError, match="ambiguous"):
        migrate_legacy_upgd_state(
            ambiguous_replacements,
            perturbation_interval=13,
        )


def test_recycling_pending_credit_is_bounded_and_early_schedule_is_preserved() -> None:
    learner = UPGDLearner(
        n_heads=1,
        hidden_sizes=(4,),
        sparsity=0.0,
        step_size=0.0,
        perturbation_sigma=0.0,
        unit_replacement_rate=0.1,
        unit_maturity_threshold=0,
    )
    state = learner.init(3, jr.key(21))
    observation = jnp.zeros(3, dtype=jnp.float32)
    target = jnp.zeros(1, dtype=jnp.float32)

    first = learner.update(state, observation, target).state
    second = learner.update(first, observation, target).state
    third = learner.update(second, observation, target).state

    assert float(first.unit_replacement_accumulators[0]) == pytest.approx(0.4)
    assert float(second.unit_replacement_accumulators[0]) == pytest.approx(0.8)
    assert float(third.unit_replacement_accumulators[0]) == pytest.approx(0.2)
    assert int(first.unit_replacement_counts[0]) == 0
    assert int(second.unit_replacement_counts[0]) == 0
    assert int(third.unit_replacement_counts[0]) == 1

    blocked = UPGDLearner(
        n_heads=1,
        hidden_sizes=(4,),
        sparsity=0.0,
        step_size=0.0,
        perturbation_sigma=0.0,
        unit_replacement_rate=1.0,
        unit_maturity_threshold=_INT32_MAX,
    )
    blocked_state = blocked.init(3, jr.key(22))

    def scan_step(carry: UPGDState, _: jax.Array) -> tuple[UPGDState, jax.Array]:
        result = blocked.update(carry, observation, target)
        return result.state, result.update_applied

    final, applied = jax.lax.scan(
        scan_step,
        blocked_state,
        jnp.arange(64, dtype=jnp.int32),
    )
    assert bool(jnp.all(applied))
    assert float(final.unit_replacement_accumulators[0]) == 1.0
    assert int(final.unit_replacement_counts[0]) == 0


def test_replacement_count_is_saturating_non_authoritative_telemetry() -> None:
    learner = UPGDLearner(
        n_heads=1,
        hidden_sizes=(2,),
        sparsity=0.0,
        step_size=0.0,
        perturbation_sigma=0.0,
        unit_replacement_rate=0.5,
        unit_maturity_threshold=0,
    )
    state = learner.init(3, jr.key(23)).replace(  # type: ignore[attr-defined]
        unit_replacement_counts=jnp.asarray((_INT32_MAX,), dtype=jnp.int32)
    )
    result = learner.update(
        state,
        jnp.zeros(3, dtype=jnp.float32),
        jnp.zeros(1, dtype=jnp.float32),
    )

    assert bool(result.update_applied)
    assert int(result.state.unit_replacement_counts[0]) == _INT32_MAX


def test_resource_helpers_measure_exact_clock_delta() -> None:
    state = _learner(perturbation_interval=3).init(3, jr.key(7))
    without_clock = state.replace(  # type: ignore[attr-defined]
        step_words=jnp.zeros((0,), dtype=jnp.uint32),
        perturbation_phase=jnp.zeros((0,), dtype=jnp.int32),
    )
    assert UPGD_LIFETIME_COUNTER_NBYTES == 12
    assert UPGD_LIFETIME_COUNTER_DELTA_NBYTES == 8
    assert UPGD_TRANSACTION_CLOCK_NBYTES == 16
    assert UPGD_TRANSACTION_CLOCK_DELTA_NBYTES == 12
    assert upgd_lifetime_counter_nbytes() == UPGD_LIFETIME_COUNTER_NBYTES
    assert upgd_transaction_clock_nbytes() == UPGD_TRANSACTION_CLOCK_NBYTES
    assert (
        measure_upgd_state_nbytes(state)
        == measure_upgd_state_nbytes(without_clock)
        + UPGD_TRANSACTION_CLOCK_DELTA_NBYTES
    )


def test_strict_checkpoint_roundtrip_and_legacy_rejection(tmp_path: Path) -> None:
    learner = _learner(perturbation_interval=5)
    state = learner.init(3, jr.key(8))
    path = tmp_path / "upgd"
    save_upgd_checkpoint(learner, state, path, feature_dim=3)
    restored_learner, restored = load_upgd_checkpoint(path)
    assert restored_learner.to_config() == learner.to_config()
    chex.assert_trees_all_equal(restored, state)

    legacy_path = tmp_path / "legacy"
    save_checkpoint(
        state,
        legacy_path,
        metadata={
            "schema": "alberta.upgd-checkpoint.v1",
            "learner_config": learner.to_config(),
            "feature_dim": 3,
            "memory_accounting": learner.memory_accounting(state),
        },
    )
    with pytest.raises(ValueError, match="migrate_legacy_upgd_state"):
        load_upgd_checkpoint(legacy_path)
    assert UPGD_CHECKPOINT_SCHEMA == "alberta.upgd-checkpoint.v2"
