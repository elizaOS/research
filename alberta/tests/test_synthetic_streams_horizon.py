"""Exact-horizon and atomicity contracts for the general synthetic streams."""

from __future__ import annotations

import copy
import dataclasses
import random
from typing import Any, cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.streams.synthetic import (
    SYNTHETIC_STREAM_CLOCK_DELTA_NBYTES,
    SYNTHETIC_STREAM_CLOCK_NBYTES,
    SYNTHETIC_STREAM_NEW_CLOCK_DELTA_NBYTES,
    SYNTHETIC_STREAM_RESOURCE_SCHEMA,
    AbruptChangeStream,
    CyclicStream,
    DynamicScaleShiftStream,
    HiddenStateAR2Stream,
    PeriodicChangeStream,
    RandomWalkStream,
    ScaleDriftStream,
    ScaledStreamState,
    ScaledStreamWrapper,
    SuttonExperiment1Stream,
    SyntheticStreamResourceBudget,
    _divmod_lifetime_words,
    measure_synthetic_stream_state_nbytes,
    migrate_legacy_abrupt_change_state,
    migrate_legacy_cyclic_state,
    migrate_legacy_dynamic_scale_shift_state,
    migrate_legacy_hidden_state_ar2_state,
    migrate_legacy_periodic_change_state,
    migrate_legacy_random_walk_state,
    migrate_legacy_scale_drift_state,
    migrate_legacy_scaled_stream_state,
    migrate_legacy_sutton_experiment1_state,
    synthetic_stream_clock_nbytes,
    synthetic_stream_from_config,
)

_INT32_MAX = 2**31 - 1
_UINT32_MAX = 2**32 - 1
_UINT64_MAX = 2**64 - 1


def _stream(kind: str) -> Any:
    streams: dict[str, Any] = {
        "random": RandomWalkStream(3, drift_rate=0.01),
        "ar2": HiddenStateAR2Stream(4, visible_dim=1),
        "abrupt": AbruptChangeStream(3, change_interval=7),
        "sutton": SuttonExperiment1Stream(2, 1, change_interval=7),
        "cyclic": CyclicStream(3, cycle_length=5, num_configurations=3),
        "periodic": PeriodicChangeStream(3, period=7),
        "dynamic": DynamicScaleShiftStream(
            3,
            scale_change_interval=7,
            weight_change_interval=5,
        ),
        "drift": ScaleDriftStream(3),
        "scaled": ScaledStreamWrapper(RandomWalkStream(3), jnp.asarray((0.5, 1.0, 2.0))),
    }
    return streams[kind]


_KINDS = [
    "random",
    "ar2",
    "abrupt",
    "sutton",
    "cyclic",
    "periodic",
    "dynamic",
    "drift",
    "scaled",
]


def _event_words(event: int) -> jax.Array:
    return jnp.asarray((event >> 32, event & _UINT32_MAX), dtype=jnp.uint32)


def _at_event(stream: Any, state: Any, event: int) -> Any:
    telemetry = min(event, _INT32_MAX)
    if isinstance(state, ScaledStreamState):
        return dataclasses.replace(
            cast(Any, state),
            inner_state=_at_event(stream.inner_stream, state.inner_state, event),
            step_count=jnp.asarray(telemetry, dtype=jnp.int32),
            step_words=_event_words(event),
        )
    return dataclasses.replace(
        state,
        step_count=jnp.asarray(telemetry, dtype=jnp.int32),
        step_words=_event_words(event),
    )


def _assert_bit_exact(left: object, right: object) -> None:
    left_leaves = jax.tree.leaves(left)
    right_leaves = jax.tree.leaves(right)
    assert len(left_leaves) == len(right_leaves)
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        try:
            left_array = np.asarray(left_leaf)
            right_array = np.asarray(right_leaf)
        except TypeError:
            left_array = np.asarray(jr.key_data(left_leaf))
            right_array = np.asarray(jr.key_data(right_leaf))
        assert left_array.dtype == right_array.dtype
        assert left_array.shape == right_array.shape
        assert left_array.tobytes() == right_array.tobytes()


@pytest.mark.parametrize("kind", _KINDS)
def test_exact_low_word_carry_and_saturating_telemetry(kind: str) -> None:
    stream = _stream(kind)
    state = _at_event(stream, stream.init(jr.key(0)), _UINT32_MAX)

    result = stream.step_result(state, jnp.asarray(0, dtype=jnp.int32))

    assert bool(result.update_applied)
    np.testing.assert_array_equal(result.pre_step_words, (0, _UINT32_MAX))
    np.testing.assert_array_equal(result.post_step_words, (1, 0))
    assert int(result.state.step_count) == _INT32_MAX
    if kind == "scaled":
        np.testing.assert_array_equal(result.state.inner_state.step_words, (1, 0))


@pytest.mark.parametrize("kind", _KINDS)
def test_terminal_all_ones_refuses_and_rolls_back_every_leaf(kind: str) -> None:
    stream = _stream(kind)
    state = _at_event(stream, stream.init(jr.key(1)), _UINT64_MAX)

    result = stream.step_result(state, jnp.asarray(0, dtype=jnp.int32))

    assert bool(result.lifetime_counter_valid)
    assert not bool(result.lifetime_capacity_available)
    assert not bool(result.update_applied)
    _assert_bit_exact(result.state, state)
    assert bool(jnp.all(jnp.isnan(result.timestep.observation)))
    assert bool(jnp.all(jnp.isnan(result.timestep.target)))


@pytest.mark.parametrize("kind", _KINDS)
def test_invalid_input_is_an_atomic_rng_oracle_and_clock_noop(kind: str) -> None:
    stream = _stream(kind)
    state = stream.init(jr.key(2))

    result = stream.step_result(state, jnp.asarray(jnp.nan, dtype=jnp.float32))

    assert not bool(result.input_valid)
    assert not bool(result.update_applied)
    _assert_bit_exact(result.state, state)


@pytest.mark.parametrize("kind", _KINDS)
def test_tuple_api_matches_diagnostic_result(kind: str) -> None:
    stream = _stream(kind)
    state = stream.init(jr.key(3))
    idx = jnp.asarray(0, dtype=jnp.int32)

    result = stream.step_result(state, idx)
    timestep, next_state = stream.step(state, idx)

    _assert_bit_exact(timestep, result.timestep)
    _assert_bit_exact(next_state, result.state)


@pytest.mark.parametrize("kind", _KINDS)
def test_scan_crosses_low_word_carry_without_losing_events(kind: str) -> None:
    stream = _stream(kind)
    state = _at_event(stream, stream.init(jr.key(30)), _UINT32_MAX - 1)

    def scan_step(carry: Any, idx: jax.Array) -> tuple[Any, jax.Array]:
        result = stream.step_result(carry, idx)
        return result.state, result.update_applied

    final_state, admitted = jax.lax.scan(
        scan_step,
        state,
        jnp.arange(4, dtype=jnp.int32),
    )

    np.testing.assert_array_equal(final_state.step_words, (1, 2))
    np.testing.assert_array_equal(admitted, np.ones((4,), dtype=np.bool_))


def test_schedules_are_exact_beyond_2p32() -> None:
    base = 1 << 32
    divisible_by_7 = base + ((-base) % 7)

    for stream in (
        AbruptChangeStream(2, change_interval=7),
        SuttonExperiment1Stream(2, 1, change_interval=7),
    ):
        state = _at_event(stream, stream.init(jr.key(4)), divisible_by_7)
        result = stream.step_result(state, jnp.asarray(0))
        assert int(result.schedule_remainder) == 0
        assert bool(result.schedule_due)
        assert bool(result.oracle_changed)

    cyclic = CyclicStream(2, cycle_length=5, num_configurations=3)
    cyclic_event = base + 123
    cyclic_result = cyclic.step_result(
        _at_event(cyclic, cyclic.init(jr.key(5)), cyclic_event),
        jnp.asarray(0),
    )
    assert int(cyclic_result.schedule_remainder) == cyclic_event % 5
    assert int(cyclic_result.schedule_index) == (cyclic_event // 5) % 3

    dynamic = DynamicScaleShiftStream(
        2,
        scale_change_interval=7,
        weight_change_interval=5,
    )
    dynamic_result = dynamic.step_result(
        _at_event(dynamic, dynamic.init(jr.key(6)), divisible_by_7),
        jnp.asarray(0),
    )
    assert int(dynamic_result.schedule_remainder) == divisible_by_7 % 7
    assert int(dynamic_result.secondary_schedule_remainder) == divisible_by_7 % 5
    assert bool(dynamic_result.schedule_due)
    assert bool(dynamic_result.secondary_schedule_due) == (divisible_by_7 % 5 == 0)


def test_periodic_phase_uses_exact_remainder_beyond_2p32() -> None:
    stream = PeriodicChangeStream(3, period=7, noise_std=0.0)
    initial = stream.init(jr.key(7))
    event = (1 << 32) + 123
    first = stream.step_result(_at_event(stream, initial, event), jnp.asarray(0))
    repeated = stream.step_result(_at_event(stream, initial, event + 7), jnp.asarray(0))

    assert int(first.schedule_remainder) == event % 7
    assert int(repeated.schedule_remainder) == event % 7
    _assert_bit_exact(first.timestep, repeated.timestep)


def test_uint64_divmod_matches_python_for_randomized_schedules() -> None:
    generator = random.Random(20260802)
    divisors = (1, 3, 7, 65537, _INT32_MAX)
    for divisor in divisors:
        for _ in range(20):
            event = generator.randrange(0, 1 << 64)
            quotient_words, remainder = _divmod_lifetime_words(_event_words(event), divisor)
            quotient = (int(quotient_words[0]) << 32) | int(quotient_words[1])
            assert quotient == event // divisor
            assert int(remainder) == event % divisor


@pytest.mark.parametrize("kind", _KINDS)
def test_strict_config_roundtrip_and_exact_resource_accounting(kind: str) -> None:
    stream = _stream(kind)
    config = stream.to_config()
    restored: Any = synthetic_stream_from_config(config)
    assert restored.to_config() == config

    extra = copy.deepcopy(config)
    extra["unknown"] = 1
    with pytest.raises(ValueError, match="fields"):
        synthetic_stream_from_config(extra)

    budget = stream.resource_budget
    assert budget.state_nbytes == measure_synthetic_stream_state_nbytes(stream.init(jr.key(8)))
    expected_clock_nbytes = (
        2 * SYNTHETIC_STREAM_CLOCK_NBYTES if kind == "scaled" else SYNTHETIC_STREAM_CLOCK_NBYTES
    )
    assert budget.exact_clock_nbytes == expected_clock_nbytes
    expected_delta = (
        2 * SYNTHETIC_STREAM_NEW_CLOCK_DELTA_NBYTES
        if kind == "scaled"
        else SYNTHETIC_STREAM_NEW_CLOCK_DELTA_NBYTES
        if kind in {"random", "ar2"}
        else SYNTHETIC_STREAM_CLOCK_DELTA_NBYTES
    )
    assert budget.exact_clock_delta_nbytes == expected_delta
    assert synthetic_stream_clock_nbytes() == SYNTHETIC_STREAM_CLOCK_NBYTES
    serialized = budget.to_dict()
    assert serialized["schema"] == SYNTHETIC_STREAM_RESOURCE_SCHEMA
    assert SyntheticStreamResourceBudget.from_dict(serialized) == budget


def _legacy_fields(state: Any, *, omit: set[str]) -> dict[str, Any]:
    return {
        field.name: getattr(state, field.name)
        for field in dataclasses.fields(state)
        if field.name not in omit
    }


def test_clockless_legacy_migration_requires_external_authenticated_time() -> None:
    random_stream = RandomWalkStream(3)
    random_state = random_stream.init(jr.key(9))
    random_legacy = _legacy_fields(random_state, omit={"step_count", "step_words"})
    migrated_random = migrate_legacy_random_walk_state(
        random_legacy,
        stream=random_stream,
        legacy_step_count=31,
    )
    np.testing.assert_array_equal(migrated_random.step_words, (0, 31))

    ar2_stream = HiddenStateAR2Stream(4, visible_dim=1)
    ar2_state = ar2_stream.init(jr.key(10))
    ar2_legacy = _legacy_fields(ar2_state, omit={"step_count", "step_words"})
    migrated_ar2 = migrate_legacy_hidden_state_ar2_state(
        ar2_legacy,
        stream=ar2_stream,
        legacy_step_count=31,
    )
    np.testing.assert_array_equal(migrated_ar2.step_words, (0, 31))

    with pytest.raises(ValueError, match="authenticated"):
        migrate_legacy_random_walk_state(
            random_legacy,
            stream=random_stream,
            legacy_step_count=_INT32_MAX,
        )


@pytest.mark.parametrize(
    ("stream", "migration"),
    [
        (AbruptChangeStream(3), migrate_legacy_abrupt_change_state),
        (SuttonExperiment1Stream(2, 1), migrate_legacy_sutton_experiment1_state),
        (CyclicStream(3), migrate_legacy_cyclic_state),
        (PeriodicChangeStream(3), migrate_legacy_periodic_change_state),
        (DynamicScaleShiftStream(3), migrate_legacy_dynamic_scale_shift_state),
        (ScaleDriftStream(3), migrate_legacy_scale_drift_state),
    ],
)
def test_clocked_legacy_migration_rejects_saturated_ambiguity(
    stream: Any,
    migration: Any,
) -> None:
    state = stream.init(jr.key(11))
    legacy = _legacy_fields(state, omit={"step_words"})
    legacy["step_count"] = jnp.asarray(31, dtype=jnp.int32)
    migrated = migration(legacy, stream=stream)
    np.testing.assert_array_equal(migrated.step_words, (0, 31))

    ambiguous = dict(legacy)
    ambiguous["step_count"] = jnp.asarray(_INT32_MAX, dtype=jnp.int32)
    with pytest.raises(ValueError, match="ambiguous"):
        migration(ambiguous, stream=stream)


def test_scaled_wrapper_migrates_child_and_rejects_clock_misalignment() -> None:
    inner = RandomWalkStream(3)
    stream = ScaledStreamWrapper(inner, jnp.ones(3))
    state = stream.init(jr.key(12))
    legacy_inner = _legacy_fields(state.inner_state, omit={"step_count", "step_words"})
    migrated = migrate_legacy_scaled_stream_state(
        {"inner_state": legacy_inner},
        stream=stream,
        legacy_step_count=17,
    )
    np.testing.assert_array_equal(migrated.step_words, (0, 17))
    np.testing.assert_array_equal(migrated.inner_state.step_words, (0, 17))

    corrupt = dataclasses.replace(cast(Any, migrated), step_words=_event_words(18))
    result = stream.step_result(corrupt, jnp.asarray(0))
    assert not bool(result.child_counter_aligned)
    assert not bool(result.update_applied)
    _assert_bit_exact(result.state, corrupt)


def test_strict_state_shape_and_nonfinite_oracle_fail_closed() -> None:
    stream = RandomWalkStream(3)
    state = stream.init(jr.key(13))
    malformed = dataclasses.replace(
        cast(Any, state),
        step_words=jnp.zeros((3,), dtype=jnp.uint32),
    )
    with pytest.raises(ValueError, match="shape"):
        stream.step_result(malformed, jnp.asarray(0))

    corrupt = dataclasses.replace(
        cast(Any, state),
        true_weights=state.true_weights.at[0].set(jnp.nan),
    )
    result = stream.step_result(corrupt, jnp.asarray(0))
    assert not bool(result.state_valid)
    assert not bool(result.update_applied)
    _assert_bit_exact(result.state, corrupt)
