"""Exact-horizon contracts for the two Alberta Plan Step 1 streams."""

from __future__ import annotations

import copy
import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.streams.alberta_plan_step1 import (
    ALBERTA_PLAN_STEP1_CONFIG_SCHEMA,
    ALBERTA_PLAN_STEP1_STATE_SCHEMA,
    STEP1_STREAM_CLOCK_DELTA_NBYTES,
    STEP1_STREAM_CLOCK_NBYTES,
    STEP1_STREAM_RESOURCE_SCHEMA,
    XDIST_SHIFT_CONFIG_SCHEMA,
    XDIST_SHIFT_STATE_SCHEMA,
    AlbertaPlanStep1Stream,
    Step1StreamResourceBudget,
    XDistShiftStream,
    measure_step1_stream_state_nbytes,
    migrate_legacy_alberta_plan_step1_state,
    migrate_legacy_xdist_shift_state,
    step1_stream_clock_nbytes,
)

_INT32_MAX = 2**31 - 1
_UINT32_MAX = 2**32 - 1
_FLOAT32_MAX = 3.4028234663852886e38


def _stream(kind: str) -> Any:
    if kind == "alberta":
        return AlbertaPlanStep1Stream(
            feature_dim=3,
            num_relevant=2,
            drift_rate_w=0.01,
            drift_rate_b=0.01,
            noise_std=0.1,
            feature_std=1.0,
        )
    return XDistShiftStream(
        feature_dim=3,
        num_relevant=2,
        noise_std=0.1,
        scale_change_interval=7,
        scale_min=0.25,
        scale_max=2.0,
        noise_in_target=True,
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


@pytest.mark.parametrize("kind", ["alberta", "xdist"])
def test_exact_low_word_carry_and_saturating_telemetry(kind: str) -> None:
    stream = _stream(kind)
    state = stream.init(jr.key(0)).replace(
        step_count=jnp.asarray(_INT32_MAX, dtype=jnp.int32),
        step_words=jnp.asarray((0, _UINT32_MAX), dtype=jnp.uint32),
    )

    result = stream.step_result(state, jnp.asarray(0, dtype=jnp.int32))

    assert bool(result.update_applied)
    np.testing.assert_array_equal(result.pre_step_words, (0, _UINT32_MAX))
    np.testing.assert_array_equal(result.post_step_words, (1, 0))
    assert int(result.state.step_count) == _INT32_MAX


def test_xdist_interval_schedule_is_exact_beyond_2p32() -> None:
    stream = _stream("xdist")
    offset = (-(1 << 32)) % 7
    event = (1 << 32) + offset
    state = stream.init(jr.key(1)).replace(
        step_count=jnp.asarray(_INT32_MAX, dtype=jnp.int32),
        step_words=jnp.asarray((1, offset), dtype=jnp.uint32),
    )

    due = stream.step_result(state, jnp.asarray(0, dtype=jnp.int32))

    assert event % 7 == 0
    assert int(due.interval_remainder) == 0
    assert bool(due.scale_change_due)
    assert bool(due.scale_changed)
    assert not bool(jnp.array_equal(due.state.current_scales, state.current_scales))

    non_due_state = state.replace(
        step_words=jnp.asarray((1, offset + 1), dtype=jnp.uint32)
    )
    non_due = stream.step_result(
        non_due_state,
        jnp.asarray(0, dtype=jnp.int32),
    )
    assert int(non_due.interval_remainder) == 1
    assert not bool(non_due.scale_change_due)
    assert not bool(non_due.scale_changed)
    np.testing.assert_array_equal(
        non_due.state.current_scales,
        non_due_state.current_scales,
    )


@pytest.mark.parametrize("kind", ["alberta", "xdist"])
def test_terminal_all_ones_refuses_with_bit_exact_rng_rollback(kind: str) -> None:
    stream = _stream(kind)
    state = stream.init(jr.key(2)).replace(
        step_count=jnp.asarray(_INT32_MAX, dtype=jnp.int32),
        step_words=jnp.asarray((_UINT32_MAX, _UINT32_MAX), dtype=jnp.uint32),
    )

    result = stream.step_result(state, jnp.asarray(0, dtype=jnp.int32))

    assert bool(result.lifetime_counter_valid)
    assert not bool(result.lifetime_capacity_available)
    assert not bool(result.update_applied)
    _assert_bit_exact(result.state, state)
    assert bool(jnp.all(jnp.isnan(result.timestep.observation)))
    assert bool(jnp.all(jnp.isnan(result.timestep.target)))


@pytest.mark.parametrize("kind", ["alberta", "xdist"])
def test_invalid_input_and_corrupt_clock_are_atomic_noops(kind: str) -> None:
    stream = _stream(kind)
    state = stream.init(jr.key(3))
    invalid_input = stream.step_result(
        state,
        jnp.asarray(jnp.nan, dtype=jnp.float32),
    )
    assert not bool(invalid_input.input_valid)
    assert not bool(invalid_input.update_applied)
    _assert_bit_exact(invalid_input.state, state)

    corrupt = state.replace(step_words=jnp.asarray((0, 5), dtype=jnp.uint32))
    invalid_clock = stream.step_result(corrupt, jnp.asarray(0, dtype=jnp.int32))
    assert not bool(invalid_clock.lifetime_counter_valid)
    assert not bool(invalid_clock.update_applied)
    _assert_bit_exact(invalid_clock.state, corrupt)


@pytest.mark.parametrize("kind", ["alberta", "xdist"])
def test_nonfinite_oracle_state_is_an_atomic_noop(kind: str) -> None:
    stream = _stream(kind)
    state = stream.init(jr.key(4))
    if kind == "alberta":
        corrupt = state.replace(true_bias=jnp.asarray(jnp.nan, dtype=jnp.float32))
    else:
        corrupt = state.replace(current_scales=state.current_scales.at[0].set(jnp.inf))

    result = stream.step_result(corrupt, jnp.asarray(0, dtype=jnp.int32))

    assert not bool(result.state_valid)
    assert not bool(result.update_applied)
    _assert_bit_exact(result.state, corrupt)


@pytest.mark.parametrize("kind", ["alberta", "xdist"])
def test_finite_state_with_nonfinite_output_rolls_back(kind: str) -> None:
    maximum = jnp.asarray(_FLOAT32_MAX, dtype=jnp.float32)
    stream: Any
    state: Any
    if kind == "alberta":
        stream = AlbertaPlanStep1Stream(
            feature_dim=2,
            num_relevant=2,
            drift_rate_w=0.0,
            drift_rate_b=0.0,
            noise_std=0.0,
        )
        state = stream.init(jr.key(0)).replace(
            true_weights=jnp.asarray((maximum, maximum), dtype=jnp.float32),
            true_bias=maximum,
        )
    else:
        stream = XDistShiftStream(
            feature_dim=2,
            num_relevant=2,
            noise_std=0.0,
            scale_change_interval=2,
            scale_min=0.1,
            scale_max=_FLOAT32_MAX,
            noise_in_target=False,
        )
        state = stream.init(jr.key(0)).replace(
            true_weights=jnp.asarray((maximum, maximum), dtype=jnp.float32),
            current_scales=jnp.ones((2,), dtype=jnp.float32),
        )

    result = stream.step_result(state, jnp.asarray(0, dtype=jnp.int32))

    assert bool(result.state_valid)
    assert not bool(result.output_valid)
    assert not bool(result.update_applied)
    _assert_bit_exact(result.state, state)


@pytest.mark.parametrize("kind", ["alberta", "xdist"])
def test_tuple_api_and_eager_jit_results_match(kind: str) -> None:
    stream = _stream(kind)
    state = stream.init(jr.key(5))
    idx = jnp.asarray(0, dtype=jnp.int32)
    with jax.disable_jit():
        eager = stream.step_result(state, idx)
    compiled = stream.step_result(state, idx)
    _assert_bit_exact(eager, compiled)

    timestep, tuple_state = stream.step(state, idx)
    _assert_bit_exact(timestep, compiled.timestep)
    _assert_bit_exact(tuple_state, compiled.state)


@pytest.mark.parametrize("kind", ["alberta", "xdist"])
def test_scan_crosses_low_word_carry_with_exact_diagnostics(kind: str) -> None:
    stream = _stream(kind)
    initial_event = _UINT32_MAX - 1
    state = stream.init(jr.key(6)).replace(
        step_count=jnp.asarray(_INT32_MAX, dtype=jnp.int32),
        step_words=jnp.asarray((0, initial_event), dtype=jnp.uint32),
    )

    def scan_step(carry: Any, idx: jax.Array) -> tuple[Any, jax.Array]:
        result = stream.step_result(carry, idx)
        diagnostic = (
            result.scale_change_due
            if kind == "xdist"
            else result.update_applied
        )
        return result.state, diagnostic

    final_state, diagnostics = jax.lax.scan(
        scan_step,
        state,
        jnp.arange(4, dtype=jnp.int32),
    )
    np.testing.assert_array_equal(final_state.step_words, (1, 2))
    assert int(final_state.step_count) == _INT32_MAX
    if kind == "xdist":
        expected = [((initial_event + offset) % 7) == 0 for offset in range(4)]
        np.testing.assert_array_equal(diagnostics, expected)
    else:
        np.testing.assert_array_equal(diagnostics, np.ones((4,), dtype=np.bool_))


@pytest.mark.parametrize("kind", ["alberta", "xdist"])
def test_strict_config_and_resource_contracts(kind: str) -> None:
    stream = _stream(kind)
    config = stream.to_config()
    if kind == "alberta":
        assert config["config_schema"] == ALBERTA_PLAN_STEP1_CONFIG_SCHEMA
        assert config["state_schema"] == ALBERTA_PLAN_STEP1_STATE_SCHEMA
        restored: Any = AlbertaPlanStep1Stream.from_config(config)
    else:
        assert config["config_schema"] == XDIST_SHIFT_CONFIG_SCHEMA
        assert config["state_schema"] == XDIST_SHIFT_STATE_SCHEMA
        restored = XDistShiftStream.from_config(config)
    assert restored.to_config() == config

    extra = copy.deepcopy(config)
    extra["unknown"] = 1
    with pytest.raises(ValueError, match="fields"):
        type(stream).from_config(extra)

    state = stream.init(jr.key(7))
    budget = stream.resource_budget
    assert budget.state_nbytes == measure_step1_stream_state_nbytes(state)
    assert budget.exact_clock_nbytes == STEP1_STREAM_CLOCK_NBYTES
    assert budget.exact_clock_delta_nbytes == STEP1_STREAM_CLOCK_DELTA_NBYTES
    assert step1_stream_clock_nbytes() == STEP1_STREAM_CLOCK_NBYTES
    assert state.step_words.nbytes == STEP1_STREAM_CLOCK_DELTA_NBYTES
    serialized = budget.to_dict()
    assert serialized["schema"] == STEP1_STREAM_RESOURCE_SCHEMA
    assert Step1StreamResourceBudget.from_dict(serialized) == budget


@pytest.mark.parametrize("kind", ["alberta", "xdist"])
def test_explicit_legacy_migration_rejects_ambiguous_history(kind: str) -> None:
    stream = _stream(kind)
    state = stream.init(jr.key(8))
    legacy = {
        field.name: getattr(state, field.name)
        for field in dataclasses.fields(state)
        if field.name != "step_words"
    }
    legacy["step_count"] = jnp.asarray(31, dtype=jnp.int32)
    migration: Any
    if kind == "alberta":
        migrated: Any = migrate_legacy_alberta_plan_step1_state(
            legacy,
            stream=stream,
        )
        migration = migrate_legacy_alberta_plan_step1_state
    else:
        migrated = migrate_legacy_xdist_shift_state(legacy, stream=stream)
        migration = migrate_legacy_xdist_shift_state
    np.testing.assert_array_equal(migrated.step_words, (0, 31))
    assert bool(stream.state_is_valid(migrated))

    ambiguous = dict(legacy)
    ambiguous["step_count"] = jnp.asarray(_INT32_MAX, dtype=jnp.int32)
    with pytest.raises(ValueError, match="ambiguous"):
        migration(ambiguous, stream=stream)


@pytest.mark.parametrize("kind", ["alberta", "xdist"])
def test_structural_contracts_fail_fast(kind: str) -> None:
    stream = _stream(kind)
    state = stream.init(jr.key(9))
    with pytest.raises(ValueError, match="idx must be scalar"):
        stream.step_result(state, jnp.zeros((1,), dtype=jnp.int32))
    malformed = state.replace(step_words=jnp.zeros((3,), dtype=jnp.uint32))
    with pytest.raises(ValueError, match="step_words"):
        stream.step_result(malformed, jnp.asarray(0, dtype=jnp.int32))


def test_constructor_rejects_nonfinite_and_unrepresentable_configs() -> None:
    with pytest.raises(ValueError, match="drift_rate_w"):
        AlbertaPlanStep1Stream(drift_rate_w=float("nan"))
    with pytest.raises(ValueError, match="scale_change_interval"):
        XDistShiftStream(
            feature_dim=2,
            num_relevant=1,
            scale_change_interval=_INT32_MAX + 1,
        )
    with pytest.raises(ValueError, match="float32"):
        XDistShiftStream(
            feature_dim=2,
            num_relevant=1,
            scale_min=1.0,
            scale_max=1.0 + 1e-12,
        )
