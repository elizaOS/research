"""Exact finite-horizon contracts for independent and mixed Horde wrappers."""

from __future__ import annotations

import dataclasses

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import alberta_framework as alberta
import alberta_framework.core as core
from alberta_framework.core.horde import (
    MIXED_HORDE_LIFETIME_COUNTER_DELTA_NBYTES,
    MIXED_HORDE_LIFETIME_COUNTER_NBYTES,
    MIXED_HORDE_STATE_SCHEMA,
    MixedHorde,
    measure_mixed_horde_state_nbytes,
    migrate_legacy_mixed_horde_state,
)
from alberta_framework.core.independent_demon_horde import (
    INDEPENDENT_DEMON_HORDE_LIFETIME_COUNTER_DELTA_NBYTES,
    INDEPENDENT_DEMON_HORDE_LIFETIME_COUNTER_NBYTES,
    INDEPENDENT_DEMON_HORDE_STATE_SCHEMA,
    IndependentDemonHorde,
    measure_independent_demon_horde_state_nbytes,
    migrate_legacy_independent_demon_horde_state,
)
from alberta_framework.core.normalizers import EMANormalizer
from alberta_framework.core.optimizers import LMS
from alberta_framework.core.types import DemonType, GVFSpec, create_horde_spec

pytestmark = pytest.mark.unit

_INT32_MAX = 2**31 - 1
_UINT32_MAX = 2**32 - 1
_OBSERVATION = jnp.asarray((0.25, -0.5), dtype=jnp.float32)
_NEXT_OBSERVATION = jnp.asarray((-0.75, 0.125), dtype=jnp.float32)
_CUMULANTS = jnp.asarray((0.75, -0.25), dtype=jnp.float32)


def _mixed(*, normalizer=None) -> MixedHorde:  # type: ignore[no-untyped-def]
    spec = create_horde_spec(
        (
            GVFSpec(
                name="shared",
                demon_type=DemonType.PREDICTION,
                gamma=0.0,
                lamda=0.0,
                cumulant_index=0,
            ),  # type: ignore[call-arg]
            GVFSpec(
                name="independent",
                demon_type=DemonType.PREDICTION,
                gamma=0.8,
                lamda=0.5,
                cumulant_index=1,
            ),  # type: ignore[call-arg]
        )
    )
    return MixedHorde(
        spec,
        hidden_sizes=(),
        optimizer=LMS(step_size=0.05),
        normalizer=normalizer,
        sparsity=0.0,
        use_layer_norm=False,
    )


def _independent(*, normalizer=None) -> IndependentDemonHorde:  # type: ignore[no-untyped-def]
    mixed = _mixed(normalizer=normalizer)
    return IndependentDemonHorde(
        mixed.horde_spec,
        hidden_sizes=(),
        optimizer=LMS(step_size=0.05),
        normalizer=normalizer,
        sparsity=0.0,
        use_layer_norm=False,
    )


def _array_tree_equal(first: object, second: object) -> None:
    first_leaves, first_tree = jax.tree_util.tree_flatten(first)
    second_leaves, second_tree = jax.tree_util.tree_flatten(second)
    assert first_tree == second_tree
    for first_leaf, second_leaf in zip(first_leaves, second_leaves, strict=True):
        if isinstance(first_leaf, jax.Array) and isinstance(second_leaf, jax.Array):
            np.testing.assert_array_equal(first_leaf, second_leaf)


def _array_tree_close(first: object, second: object) -> None:
    first_leaves, first_tree = jax.tree_util.tree_flatten(first)
    second_leaves, second_tree = jax.tree_util.tree_flatten(second)
    assert first_tree == second_tree
    for first_leaf, second_leaf in zip(first_leaves, second_leaves, strict=True):
        if isinstance(first_leaf, jax.Array) and isinstance(second_leaf, jax.Array):
            np.testing.assert_allclose(first_leaf, second_leaf, rtol=1e-6, atol=1e-7)


def _set_independent_clock(state, words: tuple[int, int]):  # type: ignore[no-untyped-def]
    telemetry = min((words[0] << 32) | words[1], _INT32_MAX)
    return state.replace(
        step_count=jnp.asarray(telemetry, dtype=jnp.int32),
        step_words=jnp.asarray(words, dtype=jnp.uint32),
    )


def _set_mixed_clock(state, words: tuple[int, int]):  # type: ignore[no-untyped-def]
    telemetry = min((words[0] << 32) | words[1], _INT32_MAX)
    shared = state.shared_state.replace(
        step_count=jnp.asarray(telemetry, dtype=jnp.int32),
        step_words=jnp.asarray(words, dtype=jnp.uint32),
    )
    independent = _set_independent_clock(state.independent_state, words)
    return state.replace(
        shared_state=shared,
        independent_state=independent,
        step_count=jnp.asarray(telemetry, dtype=jnp.int32),
        step_words=jnp.asarray(words, dtype=jnp.uint32),
    )


def test_independent_crosses_uint32_carry_and_saturates_telemetry_in_scan() -> None:
    horde = _independent()
    state = _set_independent_clock(
        horde.init(2, jax.random.key(0)),
        (0, _UINT32_MAX),
    )

    def step(carry, unused):  # type: ignore[no-untyped-def]
        del unused
        result = horde.update(carry, _OBSERVATION, _CUMULANTS, _NEXT_OBSERVATION)
        return result.state, (
            result.pre_step_words,
            result.post_step_words,
            result.update_applied,
        )

    final_state, (pre_words, post_words, applied) = jax.lax.scan(
        step,
        state,
        jnp.arange(2, dtype=jnp.int32),
    )

    np.testing.assert_array_equal(applied, (True, True))
    np.testing.assert_array_equal(
        pre_words,
        np.asarray(((0, _UINT32_MAX), (1, 0)), dtype=np.uint32),
    )
    np.testing.assert_array_equal(
        post_words,
        np.asarray(((1, 0), (1, 1)), dtype=np.uint32),
    )
    np.testing.assert_array_equal(final_state.step_words, (1, 1))
    assert int(final_state.step_count) == _INT32_MAX


def test_mixed_eager_jit_scan_keeps_all_three_exact_clocks_aligned() -> None:
    horde = _mixed()
    state = _set_mixed_clock(horde.init(2, jax.random.key(1)), (0, _UINT32_MAX))

    with jax.disable_jit():
        eager = horde.update(state, _OBSERVATION, _CUMULANTS, _NEXT_OBSERVATION)
    compiled = jax.jit(horde.update)(
        state,
        _OBSERVATION,
        _CUMULANTS,
        _NEXT_OBSERVATION,
    )
    for result in (eager, compiled):
        assert bool(result.update_applied)
        assert bool(result.child_counters_aligned)
        np.testing.assert_array_equal(result.pre_step_words, (0, _UINT32_MAX))
        np.testing.assert_array_equal(result.post_step_words, (1, 0))
        np.testing.assert_array_equal(result.state.step_words, (1, 0))
        np.testing.assert_array_equal(result.state.shared_state.step_words, (1, 0))
        np.testing.assert_array_equal(
            result.state.independent_state.step_words,
            (1, 0),
        )
        assert int(result.state.step_count) == _INT32_MAX
    _array_tree_close(eager.state, compiled.state)

    final_state, applied = jax.lax.scan(
        lambda carry, _: (
            (result := horde.update(
                carry,
                _OBSERVATION,
                _CUMULANTS,
                _NEXT_OBSERVATION,
            )).state,
            result.update_applied,
        ),
        compiled.state,
        jnp.arange(2, dtype=jnp.int32),
    )
    np.testing.assert_array_equal(applied, (True, True))
    np.testing.assert_array_equal(final_state.step_words, (1, 2))
    np.testing.assert_array_equal(final_state.shared_state.step_words, (1, 2))
    np.testing.assert_array_equal(final_state.independent_state.step_words, (1, 2))


@pytest.mark.parametrize("kind", ("outer-corrupt", "child-misaligned", "terminal"))
def test_mixed_corruption_or_terminal_capacity_is_an_atomic_noop(kind: str) -> None:
    horde = _mixed()
    state = horde.init(2, jax.random.key(2))
    if kind == "outer-corrupt":
        state = state.replace(step_count=jnp.asarray(1, dtype=jnp.int32))
    elif kind == "child-misaligned":
        state = state.replace(
            shared_state=state.shared_state.replace(
                step_count=jnp.asarray(1, dtype=jnp.int32),
                step_words=jnp.asarray((0, 1), dtype=jnp.uint32),
            )
        )
    else:
        state = _set_mixed_clock(state, (_UINT32_MAX, _UINT32_MAX))

    result = horde.update(state, _OBSERVATION, _CUMULANTS, _NEXT_OBSERVATION)

    assert not bool(result.update_applied)
    np.testing.assert_array_equal(result.pre_step_words, state.step_words)
    np.testing.assert_array_equal(result.post_step_words, state.step_words)
    _array_tree_equal(result.state, state)
    if kind == "child-misaligned":
        assert not bool(result.child_counters_aligned)
    if kind == "terminal":
        assert not bool(result.lifetime_capacity_available)


def test_mixed_discards_independent_candidate_when_shared_child_refuses() -> None:
    horde = _mixed(normalizer=EMANormalizer(decay=0.9))
    state = horde.init(2, jax.random.key(3))
    shared = state.shared_state
    assert shared.normalizer_state is not None
    # Both route wrappers remain aligned at event zero, but the shared route's
    # nested normalizer is corrupted.  The independent route can construct a
    # valid candidate, which the mixed wrapper must nevertheless discard.
    shared = shared.replace(
        normalizer_state=shared.normalizer_state.replace(
            sample_count=jnp.asarray(1, dtype=jnp.int32),
            sample_count_words=jnp.asarray((0, 1), dtype=jnp.uint32),
        ),
    )
    state = state.replace(shared_state=shared)
    independent_before = state.independent_state

    result = horde.update(state, _OBSERVATION, _CUMULANTS, _NEXT_OBSERVATION)

    assert not bool(result.update_applied)
    assert bool(result.child_counters_aligned)
    assert not bool(result.normalizer_counter_aligned)
    _array_tree_equal(result.state, state)
    _array_tree_equal(result.state.independent_state, independent_before)


def test_independent_rejects_nonfinite_source_and_nonfinite_candidate() -> None:
    horde = _independent()
    state = horde.init(2, jax.random.key(4))
    invalid_source = horde.update(
        state,
        jnp.asarray((jnp.inf, 0.0), dtype=jnp.float32),
        _CUMULANTS,
        _NEXT_OBSERVATION,
    )
    assert not bool(invalid_source.source_valid)
    assert not bool(invalid_source.update_applied)
    _array_tree_equal(invalid_source.state, state)

    huge = jnp.asarray(jnp.finfo(jnp.float32).max, dtype=jnp.float32)
    candidate_failure = horde.update(
        state,
        jnp.asarray((huge, huge), dtype=jnp.float32),
        _CUMULANTS,
        _NEXT_OBSERVATION,
    )
    assert bool(candidate_failure.source_valid)
    assert not bool(candidate_failure.candidate_valid)
    assert not bool(candidate_failure.update_applied)
    _array_tree_equal(candidate_failure.state, state)


def test_independent_all_ones_clock_and_nested_normalizer_refuse_atomically() -> None:
    horde = _independent(normalizer=EMANormalizer(decay=0.9))
    state = horde.init(2, jax.random.key(5))
    terminal = jnp.full((2,), _UINT32_MAX, dtype=jnp.uint32)
    state = state.replace(
        step_count=jnp.asarray(_INT32_MAX, dtype=jnp.int32),
        step_words=terminal,
    )
    result = horde.update(state, _OBSERVATION, _CUMULANTS, _NEXT_OBSERVATION)
    assert bool(result.lifetime_counter_valid)
    assert not bool(result.lifetime_capacity_available)
    assert not bool(result.update_applied)
    _array_tree_equal(result.state, state)

    ordinary = horde.init(2, jax.random.key(6))
    first = ordinary.demon_states[0]
    assert first.normalizer_state is not None
    corrupted_first = first.replace(
        normalizer_state=first.normalizer_state.replace(
            sample_count=jnp.asarray(1, dtype=jnp.int32),
            sample_count_words=jnp.asarray((0, 1), dtype=jnp.uint32),
        )
    )
    corrupted = ordinary.replace(
        demon_states=(corrupted_first, *ordinary.demon_states[1:])
    )
    rejected = horde.update(
        corrupted,
        _OBSERVATION,
        _CUMULANTS,
        _NEXT_OBSERVATION,
    )
    assert not bool(rejected.normalizer_counter_aligned)
    assert not bool(rejected.update_applied)
    _array_tree_equal(rejected.state, corrupted)


def test_v2_schema_migration_resource_delta_and_public_exports() -> None:
    independent = _independent()
    independent_state = independent.init(2, jax.random.key(7))
    mixed = _mixed()
    mixed_state = mixed.init(2, jax.random.key(8))

    assert independent.to_config()["state_schema"] == INDEPENDENT_DEMON_HORDE_STATE_SCHEMA
    assert mixed.to_config()["state_schema"] == MIXED_HORDE_STATE_SCHEMA
    assert INDEPENDENT_DEMON_HORDE_LIFETIME_COUNTER_NBYTES == 12
    assert INDEPENDENT_DEMON_HORDE_LIFETIME_COUNTER_DELTA_NBYTES == 8
    assert MIXED_HORDE_LIFETIME_COUNTER_NBYTES == 12
    assert MIXED_HORDE_LIFETIME_COUNTER_DELTA_NBYTES == 8
    assert measure_independent_demon_horde_state_nbytes(independent_state) >= 12
    assert measure_mixed_horde_state_nbytes(mixed_state) >= 36
    independent_bytes = measure_independent_demon_horde_state_nbytes(
        independent_state
    )
    independent_updated = independent.update(
        independent_state,
        _OBSERVATION,
        _CUMULANTS,
        _NEXT_OBSERVATION,
    )
    assert (
        measure_independent_demon_horde_state_nbytes(independent_updated.state)
        == independent_bytes
    )
    mixed_bytes = measure_mixed_horde_state_nbytes(mixed_state)
    mixed_updated = mixed.update(
        mixed_state,
        _OBSERVATION,
        _CUMULANTS,
        _NEXT_OBSERVATION,
    )
    assert measure_mixed_horde_state_nbytes(mixed_updated.state) == mixed_bytes

    independent_legacy = {
        field.name: getattr(independent_state, field.name)
        for field in dataclasses.fields(independent_state)
        if field.name != "step_words"
    }
    independent_legacy["step_count"] = jnp.asarray(3, dtype=jnp.int32)
    migrated_independent = migrate_legacy_independent_demon_horde_state(
        independent_legacy
    )
    np.testing.assert_array_equal(migrated_independent.step_words, (0, 3))

    normalized_horde = _independent(normalizer=EMANormalizer(decay=0.9))
    normalized_state = normalized_horde.update(
        normalized_horde.init(2, jax.random.key(70)),
        _OBSERVATION,
        _CUMULANTS,
        _NEXT_OBSERVATION,
    ).state
    normalized_legacy = {
        field.name: getattr(normalized_state, field.name)
        for field in dataclasses.fields(normalized_state)
        if field.name != "step_words"
    }
    legacy_demons = []
    for demon_state in normalized_state.demon_states:
        assert demon_state.normalizer_state is not None
        legacy_normalizer = {
            field.name: getattr(demon_state.normalizer_state, field.name)
            for field in dataclasses.fields(demon_state.normalizer_state)
            if field.name != "sample_count_words"
        }
        legacy_normalizer["sample_count"] = jnp.asarray(1.0, dtype=jnp.float32)
        legacy_demon = {
            field.name: getattr(demon_state, field.name)
            for field in dataclasses.fields(demon_state)
        }
        legacy_demon["normalizer_state"] = legacy_normalizer
        legacy_demons.append(legacy_demon)
    normalized_legacy["demon_states"] = tuple(legacy_demons)
    migrated_normalized = migrate_legacy_independent_demon_horde_state(
        normalized_legacy
    )
    np.testing.assert_array_equal(migrated_normalized.step_words, (0, 1))
    for demon_state in migrated_normalized.demon_states:
        assert demon_state.normalizer_state is not None
        np.testing.assert_array_equal(
            demon_state.normalizer_state.sample_count_words,
            (0, 1),
        )

    aligned = mixed_state.replace(
        step_count=jnp.asarray(3, dtype=jnp.int32),
        shared_state=mixed_state.shared_state.replace(
            step_count=jnp.asarray(3, dtype=jnp.int32),
            step_words=jnp.asarray((0, 3), dtype=jnp.uint32),
        ),
        independent_state=mixed_state.independent_state.replace(
            step_count=jnp.asarray(3, dtype=jnp.int32),
            step_words=jnp.asarray((0, 3), dtype=jnp.uint32),
        ),
    )
    mixed_legacy = {
        field.name: getattr(aligned, field.name)
        for field in dataclasses.fields(aligned)
        if field.name != "step_words"
    }
    migrated_mixed = migrate_legacy_mixed_horde_state(mixed_legacy)
    np.testing.assert_array_equal(migrated_mixed.step_words, (0, 3))

    for migrate, legacy in (
        (migrate_legacy_independent_demon_horde_state, independent_legacy),
        (migrate_legacy_mixed_horde_state, mixed_legacy),
    ):
        saturated = dict(legacy)
        saturated["step_count"] = jnp.asarray(_INT32_MAX, dtype=jnp.int32)
        with pytest.raises(ValueError, match="saturated"):
            migrate(saturated)
        extra = dict(legacy)
        extra["surprise"] = 1
        with pytest.raises(ValueError, match="manifest"):
            migrate(extra)

    expected_exports = (
        "INDEPENDENT_DEMON_HORDE_STATE_SCHEMA",
        "MIXED_HORDE_STATE_SCHEMA",
        "IndependentDemonHorde",
        "MixedHorde",
        "measure_independent_demon_horde_state_nbytes",
        "measure_mixed_horde_state_nbytes",
        "migrate_legacy_independent_demon_horde_state",
        "migrate_legacy_mixed_horde_state",
    )
    for name in expected_exports:
        assert hasattr(core, name)
        assert hasattr(alberta, name)


def test_v2_schema_rejects_unknown_schema_and_strict_legacy_manifest() -> None:
    independent = _independent()
    independent_config = independent.to_config()
    independent_config["state_schema"] = "alberta.independent-demon-horde-state.v1"
    with pytest.raises(ValueError, match="state schema"):
        IndependentDemonHorde.from_config(independent_config)

    mixed = _mixed()
    mixed_config = mixed.to_config()
    mixed_config["state_schema"] = "alberta.mixed-horde-state.v1"
    with pytest.raises(ValueError, match="state schema"):
        MixedHorde.from_config(mixed_config)

    state = _independent().init(2, jax.random.key(9))
    legacy = {
        field.name: getattr(state, field.name)
        for field in dataclasses.fields(state)
        if field.name != "step_words"
    }
    legacy.pop("uptime_s")
    with pytest.raises(ValueError, match="manifest"):
        migrate_legacy_independent_demon_horde_state(legacy)


def test_normal_update_preserves_existing_learning_semantics() -> None:
    horde = _mixed()
    state = horde.init(2, jax.random.key(10))
    result = horde.update(state, _OBSERVATION, _CUMULANTS, _NEXT_OBSERVATION)

    assert bool(result.update_applied)
    chex.assert_shape(result.predictions, (2,))
    chex.assert_shape(result.td_errors, (2,))
    chex.assert_shape(result.per_demon_metrics, (2, 3))
    np.testing.assert_array_equal(result.state.step_words, (0, 1))
    np.testing.assert_array_equal(result.state.shared_state.step_words, (0, 1))
    np.testing.assert_array_equal(result.state.independent_state.step_words, (0, 1))
