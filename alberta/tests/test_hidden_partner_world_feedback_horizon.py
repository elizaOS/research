"""Exact-horizon contracts for the hidden-partner world-feedback stream."""

from __future__ import annotations

import dataclasses
from typing import Any, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.streams import hidden_partner_world_feedback as feedback_module
from alberta_framework.streams.hidden_partner_mapping import DEFAULT_REGIME_SCHEDULE
from alberta_framework.streams.hidden_partner_world_feedback import (
    HIDDEN_PARTNER_WORLD_FEEDBACK_CONFIG_SCHEMA,
    HIDDEN_PARTNER_WORLD_FEEDBACK_EXACT_IDENTITY_NBYTES,
    HIDDEN_PARTNER_WORLD_FEEDBACK_STATE_SCHEMA,
    HiddenPartnerWorldFeedbackConfig,
    HiddenPartnerWorldFeedbackState,
    HiddenPartnerWorldFeedbackWorld,
    measure_hidden_partner_world_feedback_state_nbytes,
    migrate_legacy_hidden_partner_world_feedback_config,
    migrate_legacy_hidden_partner_world_feedback_state,
)
from alberta_framework.streams.hidden_partner_world_feedback import (
    _divmod_words_by_positive_int32 as _exact_divmod,
)

pytestmark = pytest.mark.unit

_INT32_MAX = 2**31 - 1
_UINT32_MAX = 2**32 - 1


def _world(*, jitter_radius: int = 0) -> HiddenPartnerWorldFeedbackWorld:
    return HiddenPartnerWorldFeedbackWorld(
        HiddenPartnerWorldFeedbackConfig(
            base_segment_lengths=(5,) * 9,
            jitter_radius=jitter_radius,
            partner_flip_probability=0.0,
            world_flip_probability=0.0,
            cue_flip_probabilities=(0.0, 0.0),
            outcome_flip_probability=0.0,
        )
    )


def _words(value: int) -> jax.Array:
    return jnp.asarray(
        ((value >> 32) & _UINT32_MAX, value & _UINT32_MAX),
        dtype=jnp.uint32,
    )


def _replace_state(
    state: HiddenPartnerWorldFeedbackState,
    **changes: Any,
) -> HiddenPartnerWorldFeedbackState:
    return cast(HiddenPartnerWorldFeedbackState, cast(Any, state).replace(**changes))


def _valid_state_at(
    world: HiddenPartnerWorldFeedbackWorld,
    exact_step: int,
    *,
    key: int = 101,
) -> HiddenPartnerWorldFeedbackState:
    _, state = world.step(
        world.init(jr.key(key)),
        jnp.asarray(0, dtype=jnp.int32),
    )
    return _replace_state(
        state,
        step_count=jnp.asarray(min(exact_step, _INT32_MAX), dtype=jnp.int32),
        step_words=_words(exact_step),
    )


def _assert_state_equal_including_nan(
    actual: HiddenPartnerWorldFeedbackState,
    expected: HiddenPartnerWorldFeedbackState,
) -> None:
    for field in dataclasses.fields(cast(Any, HiddenPartnerWorldFeedbackState)):
        actual_value = getattr(actual, field.name)
        expected_value = getattr(expected, field.name)
        if field.name.endswith("_key"):
            actual_value = jr.key_data(actual_value)
            expected_value = jr.key_data(expected_value)
        np.testing.assert_equal(np.asarray(actual_value), np.asarray(expected_value))


def _assert_transition_finite(transition: object) -> None:
    for leaf in jax.tree.leaves(transition):
        if hasattr(leaf, "dtype") and jnp.issubdtype(leaf.dtype, jnp.inexact):
            assert np.all(np.isfinite(np.asarray(leaf)))


def _expected_schedule(
    exact_step: int,
    segment_ends: tuple[int, ...],
) -> tuple[int, int, int, int]:
    cycle_length = segment_ends[-1]
    cycle_index, cycle_step = divmod(exact_step, cycle_length)
    segment_index = sum(cycle_step >= end for end in segment_ends)
    return cycle_index, cycle_step, segment_index, DEFAULT_REGIME_SCHEDULE[segment_index]


def test_exact_word_division_matches_python_across_full_uint64_domain() -> None:
    values = (
        0,
        1,
        _INT32_MAX,
        _UINT32_MAX - 1,
        _UINT32_MAX,
        1 << 32,
        (1 << 32) + 1,
        1 << 63,
        (1 << 64) - 2,
        (1 << 64) - 1,
    )
    divisors = (1, 2, 3, 17, 45, _INT32_MAX)
    pairs = tuple((value, divisor) for value in values for divisor in divisors)
    word_batch = jnp.stack([_words(value) for value, _ in pairs])
    divisor_batch = jnp.asarray([divisor for _, divisor in pairs], dtype=jnp.int32)

    quotient_words, remainders = jax.jit(jax.vmap(_exact_divmod))(
        word_batch,
        divisor_batch,
    )
    for index, (value, divisor) in enumerate(pairs):
        quotient, remainder = divmod(value, divisor)
        chex.assert_trees_all_equal(quotient_words[index], _words(quotient))
        assert int(remainders[index]) == remainder


def test_jittered_schedule_survives_int32_saturation_carry_eager_jit_and_scan() -> None:
    world = _world(jitter_radius=2)
    start = _UINT32_MAX - 1
    initial = _valid_state_at(world, start)
    segment_ends = tuple(int(value) for value in initial.segment_ends.tolist())
    actions = jnp.asarray((0, 1, 0), dtype=jnp.int32)

    eager_results = []
    eager_state = initial
    for action in actions:
        result = world.step_result(eager_state, action)
        eager_results.append(result)
        eager_state = result.state

    compiled_step = jax.jit(world.step_result)
    compiled_state = initial
    for eager, action in zip(eager_results, actions, strict=True):
        compiled = compiled_step(compiled_state, action)
        chex.assert_trees_all_equal(eager, compiled)
        compiled_state = compiled.state

    for offset, result in enumerate(eager_results):
        exact_step = start + offset
        cycle, cycle_step, segment, regime = _expected_schedule(exact_step, segment_ends)
        next_cycle, next_cycle_step, next_segment, _ = _expected_schedule(
            exact_step + 1,
            segment_ends,
        )
        chex.assert_trees_all_equal(result.pre_step_words, _words(exact_step))
        chex.assert_trees_all_equal(result.post_step_words, _words(exact_step + 1))
        chex.assert_trees_all_equal(result.transition.oracle.step_words, _words(exact_step))
        chex.assert_trees_all_equal(
            result.transition.oracle.next_step_words,
            _words(exact_step + 1),
        )
        chex.assert_trees_all_equal(result.transition.oracle.cycle_words, _words(cycle))
        chex.assert_trees_all_equal(
            result.transition.oracle.next_cycle_words,
            _words(next_cycle),
        )
        assert int(result.state.step_count) == _INT32_MAX
        assert int(result.transition.oracle.cycle_index) == min(cycle, _INT32_MAX)
        assert int(result.transition.oracle.cycle_step) == cycle_step
        assert int(result.transition.oracle.segment_index) == segment
        assert int(result.transition.oracle.regime_id) == regime
        assert bool(result.transition.oracle.schedule_switched) == (
            cycle != next_cycle or segment != next_segment
        )
        assert bool(result.lifetime_counter_valid)
        assert bool(result.lifetime_capacity_available)
        assert bool(result.state_valid)
        assert bool(result.input_valid)
        assert bool(result.candidate_state_finite)
        assert bool(result.candidate_state_valid)
        assert bool(result.update_applied)

    def scan_body(
        state: HiddenPartnerWorldFeedbackState,
        action: jax.Array,
    ) -> tuple[HiddenPartnerWorldFeedbackState, tuple[jax.Array, ...]]:
        result = world.step_result(state, action)
        trace = (
            result.pre_step_words,
            result.post_step_words,
            result.transition.oracle.cycle_words,
            result.transition.oracle.cycle_step,
            result.transition.oracle.segment_index,
            result.transition.oracle.regime_id,
            result.update_applied,
        )
        return result.state, trace

    scan = jax.jit(lambda state, xs: jax.lax.scan(scan_body, state, xs))
    scanned_state, scanned_trace = scan(initial, actions)
    chex.assert_trees_all_equal(scanned_state, eager_state)
    expected_traces = tuple(
        jnp.stack(values)
        for values in zip(
            *(
                (
                    result.pre_step_words,
                    result.post_step_words,
                    result.transition.oracle.cycle_words,
                    result.transition.oracle.cycle_step,
                    result.transition.oracle.segment_index,
                    result.transition.oracle.regime_id,
                    result.update_applied,
                )
                for result in eager_results
            ),
            strict=True,
        )
    )
    chex.assert_trees_all_equal(scanned_trace, expected_traces)


def test_exact_jittered_boundary_and_saturated_cycle_telemetry_are_authoritative() -> None:
    world = _world(jitter_radius=2)
    seeded = world.init(jr.key(102))
    segment_ends = tuple(int(value) for value in seeded.segment_ends.tolist())
    cycle_length = segment_ends[-1]
    cycle = (1 << 32) // cycle_length + 7
    boundary_step = cycle * cycle_length + segment_ends[0] - 1
    state = _valid_state_at(world, boundary_step, key=102)

    result = jax.jit(world.step_result)(state, jnp.asarray(1, dtype=jnp.int32))
    assert int(result.transition.oracle.segment_index) == 0
    assert int(result.transition.oracle.next_segment_index) == 1
    assert bool(result.transition.oracle.schedule_switched)
    chex.assert_trees_all_equal(result.transition.oracle.cycle_words, _words(cycle))
    legacy_projection = world._schedule_position(  # noqa: SLF001
        state,
        jnp.asarray(0, dtype=jnp.int32),
    )
    chex.assert_trees_all_equal(legacy_projection.cycle_words, _words(cycle))
    assert int(legacy_projection.cycle_step) == segment_ends[0] - 1

    wide_step = (64 << 32) + 7
    wide = world.step_result(
        _valid_state_at(world, wide_step, key=102),
        jnp.asarray(0, dtype=jnp.int32),
    )
    wide_cycle, wide_phase = divmod(wide_step, cycle_length)
    chex.assert_trees_all_equal(wide.transition.oracle.cycle_words, _words(wide_cycle))
    assert int(wide.transition.oracle.cycle_index) == _INT32_MAX
    assert int(wide.transition.oracle.cycle_step) == wide_phase
    assert bool(wide.update_applied)


@pytest.mark.parametrize(
    "failure",
    ("counter", "signals", "cues", "world", "schedule", "action"),
)
def test_invalid_transactions_roll_back_all_state_and_rng_leaves(failure: str) -> None:
    world = _world()
    valid = _valid_state_at(world, 1)
    state = valid
    action = jnp.asarray(0, dtype=jnp.int32)
    counter_valid = True
    state_valid = True
    input_valid = True
    candidate_valid = True

    if failure == "counter":
        state = _replace_state(valid, step_count=jnp.asarray(0, dtype=jnp.int32))
        counter_valid = False
        state_valid = False
    elif failure == "signals":
        state = _replace_state(
            valid,
            current_signals=valid.current_signals.at[0].set(jnp.nan),
        )
        state_valid = False
    elif failure == "cues":
        state = _replace_state(valid, current_cues=valid.current_cues.at[1].set(jnp.inf))
        state_valid = False
    elif failure == "world":
        state = _replace_state(valid, world_sign=jnp.asarray(jnp.nan, dtype=jnp.float32))
        state_valid = False
    elif failure == "schedule":
        state = _replace_state(valid, segment_ends=jnp.zeros((9,), dtype=jnp.int32))
        state_valid = False
        candidate_valid = False
    else:
        action = jnp.asarray(_UINT32_MAX, dtype=jnp.uint32)
        input_valid = False

    eager = world.step_result(state, action)
    compiled = jax.jit(world.step_result)(state, action)
    for result in (eager, compiled):
        _assert_state_equal_including_nan(result.state, state)
        chex.assert_trees_all_equal(result.pre_step_words, state.step_words)
        chex.assert_trees_all_equal(result.post_step_words, state.step_words)
        assert bool(result.lifetime_counter_valid) is counter_valid
        assert bool(result.lifetime_capacity_available)
        assert bool(result.state_valid) is state_valid
        assert bool(result.input_valid) is input_valid
        assert bool(result.candidate_state_finite)
        assert bool(result.candidate_state_valid) is candidate_valid
        assert not bool(result.update_applied)
        assert not bool(result.transition.terminated)
        assert float(result.transition.discount) == 0.0
        assert int(result.transition.focal_action) == -1
        assert int(result.transition.partner_action) == -1
        _assert_transition_finite(result.transition)

    chex.assert_trees_all_equal(eager.transition, compiled.transition)


def test_terminal_identity_returns_finite_terminal_noop_and_tuple_api_matches() -> None:
    world = _world()
    terminal = _valid_state_at(world, (1 << 64) - 1)
    result = world.step_result(terminal, jnp.asarray(1, dtype=jnp.int32))
    compiled = jax.jit(world.step_result)(terminal, jnp.asarray(1, dtype=jnp.int32))
    chex.assert_trees_all_equal(result, compiled)

    _assert_state_equal_including_nan(result.state, terminal)
    assert bool(result.lifetime_counter_valid)
    assert not bool(result.lifetime_capacity_available)
    assert bool(result.state_valid)
    assert bool(result.input_valid)
    assert bool(result.candidate_state_finite)
    assert not bool(result.update_applied)
    assert bool(result.transition.terminated)
    assert float(result.transition.discount) == 0.0
    _assert_transition_finite(result.transition)

    ordinary = _valid_state_at(world, 9)
    transition, next_state = world.step(ordinary, jnp.asarray(0, dtype=jnp.int32))
    audited = world.step_result(ordinary, jnp.asarray(0, dtype=jnp.int32))
    chex.assert_trees_all_equal((transition, next_state), (audited.transition, audited.state))


def test_static_contracts_and_wide_input_alias_fail_closed() -> None:
    world = _world()
    state = world.init(jr.key(103))
    with pytest.raises(TypeError, match="step_words"):
        world.step_result(
            _replace_state(state, step_words=jnp.zeros((2,), dtype=jnp.int32)),
            jnp.asarray(0, dtype=jnp.int32),
        )
    with pytest.raises(TypeError, match="step_words"):
        world.step_result(
            _replace_state(state, step_words=jnp.zeros((1,), dtype=jnp.uint32)),
            jnp.asarray(0, dtype=jnp.int32),
        )
    with pytest.raises(TypeError, match="integer"):
        world.step_result(state, jnp.asarray(jnp.nan, dtype=jnp.float32))

    with jax.enable_x64():
        wide_action = jnp.asarray(2**32, dtype=jnp.int64)
        rejected = jax.jit(world.step_result)(state, wide_action)
        assert not bool(rejected.input_valid)
        assert not bool(rejected.update_applied)
        _assert_state_equal_including_nan(rejected.state, state)


def test_v2_migrations_are_explicit_strict_and_resource_exact() -> None:
    world = _world(jitter_radius=2)
    config = world.config
    payload = config.to_config()
    assert payload["schema"] == HIDDEN_PARTNER_WORLD_FEEDBACK_CONFIG_SCHEMA
    assert payload["state_schema"] == HIDDEN_PARTNER_WORLD_FEEDBACK_STATE_SCHEMA

    legacy_config = dict(payload)
    legacy_config.pop("schema")
    legacy_config.pop("state_schema")
    with pytest.raises(ValueError, match="explicit migration"):
        HiddenPartnerWorldFeedbackConfig.from_config(legacy_config)
    assert migrate_legacy_hidden_partner_world_feedback_config(legacy_config) == config
    with pytest.raises(ValueError, match="not exact"):
        migrate_legacy_hidden_partner_world_feedback_config({**legacy_config, "extra": 1})

    state = world.init(jr.key(104))
    for action in (0, 1, 1, 0):
        _, state = world.step(state, jnp.asarray(action, dtype=jnp.int32))
    legacy_state = {
        field.name: getattr(state, field.name)
        for field in dataclasses.fields(cast(Any, HiddenPartnerWorldFeedbackState))
        if field.name != "step_words"
    }
    migrated = migrate_legacy_hidden_partner_world_feedback_state(legacy_state)
    chex.assert_trees_all_equal(migrated, state)
    with pytest.raises(ValueError, match="ambiguous"):
        migrate_legacy_hidden_partner_world_feedback_state(
            {
                **legacy_state,
                "step_count": jnp.asarray(_INT32_MAX, dtype=jnp.int32),
            }
        )
    with pytest.raises(ValueError, match="not exact"):
        migrate_legacy_hidden_partner_world_feedback_state({**legacy_state, "extra": 1})
    with pytest.raises(ValueError, match="inconsistent"):
        migrate_legacy_hidden_partner_world_feedback_state(
            {
                **legacy_state,
                "world_sign": jnp.asarray(0.0, dtype=jnp.float32),
            }
        )

    budget = world.resource_budget
    assert budget.state_schema == HIDDEN_PARTNER_WORLD_FEEDBACK_STATE_SCHEMA
    assert budget.exact_identity_uint32_scalars == 2
    assert budget.exact_identity_nbytes == HIDDEN_PARTNER_WORLD_FEEDBACK_EXACT_IDENTITY_NBYTES
    assert budget.lifetime_identity_bits == 64
    assert budget.telemetry_saturation == _INT32_MAX
    assert budget.state_nbytes == 157
    assert budget.to_dict() == {
        "state_schema": HIDDEN_PARTNER_WORLD_FEEDBACK_STATE_SCHEMA,
        "observation_float32_scalars": 8,
        "persistent_float32_scalars": 7,
        "persistent_int32_scalars": 20,
        "persistent_bool_scalars": 1,
        "exact_identity_uint32_scalars": 2,
        "exact_identity_nbytes": 8,
        "lifetime_identity_bits": 64,
        "telemetry_saturation": _INT32_MAX,
        "rng_uint32_scalars": 10,
        "persistent_state_scalars": 40,
        "state_nbytes": 157,
        "trainable_scalars": 0,
        "replay_capacity": 0,
    }
    assert measure_hidden_partner_world_feedback_state_nbytes(state) == budget.state_nbytes


def test_new_horizon_surface_is_module_local_and_complete() -> None:
    names = {
        "HIDDEN_PARTNER_WORLD_FEEDBACK_CONFIG_SCHEMA",
        "HIDDEN_PARTNER_WORLD_FEEDBACK_EXACT_IDENTITY_NBYTES",
        "HIDDEN_PARTNER_WORLD_FEEDBACK_STATE_SCHEMA",
        "HiddenPartnerWorldFeedbackStepResult",
        "measure_hidden_partner_world_feedback_state_nbytes",
        "migrate_legacy_hidden_partner_world_feedback_config",
        "migrate_legacy_hidden_partner_world_feedback_state",
    }
    for name in names:
        assert name in feedback_module.__all__
        assert getattr(feedback_module, name) is not None
