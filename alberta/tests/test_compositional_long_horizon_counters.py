"""Near-boundary contracts for compositional lifetime counters."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.compositional_features import (
    CompositionalFeatureLearner,
    migrate_legacy_compositional_feature_state,
)

pytestmark = pytest.mark.unit

_INT32_MAX = 2**31 - 1
_UINT32_MAX = 2**32 - 1
_OBSERVATION = jnp.asarray((0.25, -0.5), dtype=jnp.float32)
_TARGET = jnp.asarray((0.0,), dtype=jnp.float32)


def _learner(*, replacement_interval: int) -> CompositionalFeatureLearner:
    return CompositionalFeatureLearner(
        n_features=4,
        n_tasks=1,
        candidate_count=1,
        step_size_output=0.0,
        step_size_theta=0.0,
        replacement_interval=replacement_interval,
        min_feature_age=_INT32_MAX,
        candidate_min_age=_INT32_MAX,
        parent_novelty_weight=1.0,
        use_obgd=False,
    )


def _canonical_state(learner: CompositionalFeatureLearner):
    return learner.init(2, jr.key(91)).replace(  # type: ignore[attr-defined]
        birth_timestamp=jnp.asarray(0.0, dtype=jnp.float32),
        uptime_s=jnp.asarray(0.0, dtype=jnp.float32),
    )


def _key_data(key: jax.Array) -> np.ndarray:
    return np.asarray(jr.key_data(key))


def _assert_state_arrays_equal(first: object, second: object) -> None:
    first_leaves, first_tree = jax.tree_util.tree_flatten(first)
    second_leaves, second_tree = jax.tree_util.tree_flatten(second)
    assert first_tree == second_tree
    for first_leaf, second_leaf in zip(first_leaves, second_leaves, strict=True):
        if isinstance(first_leaf, jax.Array) and jax.dtypes.issubdtype(
            first_leaf.dtype, jax.dtypes.prng_key
        ):
            np.testing.assert_array_equal(_key_data(first_leaf), _key_data(second_leaf))
        else:
            np.testing.assert_array_equal(np.asarray(first_leaf), np.asarray(second_leaf))


def test_fixed_cadence_preserves_increment_then_modulo_under_eager_and_scan() -> None:
    learner = _learner(replacement_interval=3)
    initial = _canonical_state(learner)

    eager_phases: list[int] = []
    eager_due: list[bool] = []
    eager_words: list[tuple[int, int]] = []
    eager_state = initial
    with jax.disable_jit():
        for _ in range(7):
            result = learner.update(eager_state, _OBSERVATION, _TARGET)
            eager_state = result.state
            eager_phases.append(int(result.curation_trace.post_replacement_phase))
            eager_due.append(bool(result.curation_trace.should_try_replace))
            eager_words.append(tuple(int(word) for word in result.state.step_words))

    def scan_step(state, _: jax.Array):
        result = learner.update(state, _OBSERVATION, _TARGET)
        trace = result.curation_trace
        return result.state, (
            trace.post_replacement_phase,
            trace.should_try_replace,
            result.state.step_words,
        )

    scan_state, (scan_phases, scan_due, scan_words) = jax.lax.scan(
        scan_step,
        initial,
        jnp.arange(7, dtype=jnp.int32),
    )

    assert eager_phases == [1, 2, 0, 1, 2, 0, 1]
    assert eager_due == [False, False, True, False, False, True, False]
    assert eager_words == [(0, step) for step in range(1, 8)]
    np.testing.assert_array_equal(scan_phases, eager_phases)
    np.testing.assert_array_equal(scan_due, eager_due)
    np.testing.assert_array_equal(scan_words, eager_words)
    assert int(scan_state.step_count) == int(eager_state.step_count) == 7
    assert int(scan_state.replacement_phase) == int(eager_state.replacement_phase) == 1
    np.testing.assert_array_equal(scan_state.step_words, eager_state.step_words)
    np.testing.assert_array_equal(scan_state.ages, eager_state.ages)
    np.testing.assert_array_equal(
        scan_state.candidate_ages,
        eager_state.candidate_ages,
    )
    np.testing.assert_array_equal(_key_data(scan_state.key), _key_data(eager_state.key))


def test_int32_saturates_uint32_low_carries_and_prng_schedule_is_unchanged() -> None:
    learner = _learner(replacement_interval=0)
    initial = _canonical_state(learner)
    near_carry = initial.replace(  # type: ignore[attr-defined]
        step_count=jnp.asarray(_INT32_MAX, dtype=jnp.int32),
        step_words=jnp.asarray((0, _UINT32_MAX), dtype=jnp.uint32),
        ages=jnp.full_like(initial.ages, _INT32_MAX),
        candidate_ages=jnp.full_like(initial.candidate_ages, _INT32_MAX),
    )

    with jax.disable_jit():
        eager = learner.update(near_carry, _OBSERVATION, _TARGET)
    compiled = learner.update(near_carry, _OBSERVATION, _TARGET)

    for result in (eager, compiled):
        assert int(result.state.step_count) == _INT32_MAX
        np.testing.assert_array_equal(result.state.step_words, (1, 0))
        np.testing.assert_array_equal(result.state.ages, _INT32_MAX)
        np.testing.assert_array_equal(result.state.candidate_ages, _INT32_MAX)
        assert bool(result.curation_trace.lifetime_counter_valid)
        assert bool(result.curation_trace.lifetime_capacity_available)
        assert np.isfinite(np.asarray(result.metrics)).all()
        expected_key = jr.split(near_carry.key, 3)[0]
        np.testing.assert_array_equal(_key_data(result.state.key), _key_data(expected_key))
    _assert_state_arrays_equal(eager.state, compiled.state)

    scan_start = near_carry.replace(  # type: ignore[attr-defined]
        step_words=jnp.asarray((0, _UINT32_MAX - 1), dtype=jnp.uint32)
    )

    def scan_step(state, _: jax.Array):
        result = learner.update(state, _OBSERVATION, _TARGET)
        return result.state, result.state.step_words

    scan_state, words = jax.lax.scan(
        scan_step,
        scan_start,
        jnp.arange(2, dtype=jnp.int32),
    )
    np.testing.assert_array_equal(words, ((0, _UINT32_MAX), (1, 0)))
    np.testing.assert_array_equal(scan_state.step_words, (1, 0))
    assert int(scan_state.step_count) == _INT32_MAX


def test_all_ones_lifetime_is_an_explicit_bit_exact_noop() -> None:
    learner = _learner(replacement_interval=1)
    initial = _canonical_state(learner)
    exhausted = initial.replace(  # type: ignore[attr-defined]
        step_count=jnp.asarray(_INT32_MAX, dtype=jnp.int32),
        step_words=jnp.full((2,), _UINT32_MAX, dtype=jnp.uint32),
        ages=jnp.full_like(initial.ages, _INT32_MAX),
        candidate_ages=jnp.full_like(initial.candidate_ages, _INT32_MAX),
    )

    with jax.disable_jit():
        eager = learner.update(exhausted, _OBSERVATION, _TARGET)
    compiled = learner.update(exhausted, _OBSERVATION, _TARGET)

    for result in (eager, compiled):
        _assert_state_arrays_equal(result.state, exhausted)
        trace = result.curation_trace
        assert bool(trace.lifetime_counter_valid)
        assert not bool(trace.lifetime_capacity_available)
        assert not bool(trace.should_try_replace)
        assert not bool(trace.has_event)
        assert int(trace.logical_event_count) == 0
        np.testing.assert_array_equal(trace.pre_step_words, (_UINT32_MAX, _UINT32_MAX))
        np.testing.assert_array_equal(trace.post_step_words, trace.pre_step_words)
        assert int(trace.pre_step) == int(trace.post_step) == _INT32_MAX
        assert int(trace.pre_replacement_phase) == int(trace.post_replacement_phase) == 0


def test_legacy_migration_is_exact_and_rejects_ambiguous_or_wrapped_states() -> None:
    learner = _learner(replacement_interval=3)
    state = _canonical_state(learner)
    omitted = {"step_words", "replacement_phase"}
    legacy = {
        field.name: getattr(state, field.name)
        for field in dataclasses.fields(type(state))
        if field.name not in omitted
    }
    legacy["step_count"] = jnp.asarray(5, dtype=jnp.int32)

    migrated = migrate_legacy_compositional_feature_state(
        legacy,
        replacement_interval=3,
    )
    assert int(migrated.step_count) == 5
    np.testing.assert_array_equal(migrated.step_words, (0, 5))
    assert int(migrated.replacement_phase) == 2

    negative = dict(legacy)
    negative["step_count"] = jnp.asarray(-1, dtype=jnp.int32)
    with pytest.raises(ValueError, match="signed wrap"):
        migrate_legacy_compositional_feature_state(negative, replacement_interval=3)

    ambiguous = dict(legacy)
    ambiguous["step_count"] = jnp.asarray(_INT32_MAX, dtype=jnp.int32)
    with pytest.raises(ValueError, match="ambiguous"):
        migrate_legacy_compositional_feature_state(ambiguous, replacement_interval=3)

    negative_ages = dict(legacy)
    negative_ages["ages"] = state.ages.at[0].set(-1)
    with pytest.raises(ValueError, match="signed wrap"):
        migrate_legacy_compositional_feature_state(negative_ages, replacement_interval=3)


def test_intrinsically_unbounded_learned_replacement_clock_is_rejected() -> None:
    with pytest.raises(ValueError, match="unbounded replacement-credit backlog"):
        CompositionalFeatureLearner(
            n_features=4,
            n_tasks=1,
            replacement_interval=1,
            learn_generator_resources=True,
        )
    CompositionalFeatureLearner(
        n_features=4,
        n_tasks=1,
        replacement_interval=2,
        learn_generator_resources=True,
    )
