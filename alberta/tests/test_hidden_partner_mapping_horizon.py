"""Exact-horizon and transactional tests for the hidden-partner stream."""

from __future__ import annotations

import dataclasses
from typing import Any, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import alberta_framework as alberta
import alberta_framework.streams as streams
from alberta_framework.streams import hidden_partner_mapping as mapping_module
from alberta_framework.streams.hidden_partner_mapping import (
    DEFAULT_REGIME_SCHEDULE,
    HIDDEN_PARTNER_MAPPING_CONFIG_SCHEMA,
    HIDDEN_PARTNER_MAPPING_EXACT_IDENTITY_NBYTES,
    HIDDEN_PARTNER_MAPPING_STATE_SCHEMA,
    HiddenPartnerMappingConfig,
    HiddenPartnerMappingState,
    HiddenPartnerMappingWorld,
    measure_hidden_partner_mapping_state_nbytes,
    migrate_legacy_hidden_partner_mapping_config,
    migrate_legacy_hidden_partner_mapping_state,
)
from alberta_framework.streams.hidden_partner_mapping import (
    _divmod_words_by_positive_int32 as _exact_divmod,
)

_INT32_MAX = 2**31 - 1
_UINT32_MAX = 2**32 - 1


def _world() -> HiddenPartnerMappingWorld:
    return HiddenPartnerMappingWorld(
        HiddenPartnerMappingConfig(
            base_segment_lengths=(2,) * 9,
            jitter_radius=0,
            partner_flip_probability=0.0,
        )
    )


def _words(value: int) -> jax.Array:
    return jnp.asarray(
        ((value >> 32) & _UINT32_MAX, value & _UINT32_MAX),
        dtype=jnp.uint32,
    )


def _replace_state(
    state: HiddenPartnerMappingState,
    **changes: Any,
) -> HiddenPartnerMappingState:
    return cast(HiddenPartnerMappingState, cast(Any, state).replace(**changes))


def _valid_state_at(
    world: HiddenPartnerMappingWorld,
    exact_step: int,
) -> HiddenPartnerMappingState:
    _, state = world.step(
        world.init(jr.key(41)),
        jnp.asarray(0, dtype=jnp.int32),
    )
    return _replace_state(
        state,
        step_count=jnp.asarray(min(exact_step, _INT32_MAX), dtype=jnp.int32),
        step_words=_words(exact_step),
    )


def _assert_state_equal_including_nan(
    actual: HiddenPartnerMappingState,
    expected: HiddenPartnerMappingState,
) -> None:
    for field in dataclasses.fields(cast(Any, HiddenPartnerMappingState)):
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


def test_exact_word_division_matches_python_across_full_uint64_domain() -> None:
    values = (
        0,
        1,
        _INT32_MAX - 1,
        _INT32_MAX,
        _UINT32_MAX - 1,
        _UINT32_MAX,
        1 << 32,
        (1 << 32) + 1,
        1 << 63,
        (1 << 64) - 2,
        (1 << 64) - 1,
    )
    divisors = (1, 2, 3, 17, 18, _INT32_MAX)
    pairs = tuple((value, divisor) for value in values for divisor in divisors)
    word_batch = jnp.stack([_words(value) for value, _ in pairs])
    divisor_batch = jnp.asarray([divisor for _, divisor in pairs], dtype=jnp.int32)

    batched_divmod = jax.jit(jax.vmap(_exact_divmod))
    quotient_words, remainders = batched_divmod(word_batch, divisor_batch)
    for index, (value, divisor) in enumerate(pairs):
        quotient, remainder = divmod(value, divisor)
        chex.assert_trees_all_equal(quotient_words[index], _words(quotient))
        assert int(remainders[index]) == remainder


def test_exact_schedule_survives_int32_saturation_uint32_carry_and_scan() -> None:
    world = _world()
    start = _UINT32_MAX - 1
    initial = _valid_state_at(world, start)
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
        quotient, cycle_step = divmod(exact_step, 18)
        next_quotient, next_cycle_step = divmod(exact_step + 1, 18)
        segment_index = cycle_step // 2
        next_segment_index = next_cycle_step // 2

        chex.assert_trees_all_equal(result.pre_step_words, _words(exact_step))
        chex.assert_trees_all_equal(result.post_step_words, _words(exact_step + 1))
        chex.assert_trees_all_equal(result.transition.oracle.step_words, _words(exact_step))
        chex.assert_trees_all_equal(
            result.transition.oracle.next_step_words,
            _words(exact_step + 1),
        )
        chex.assert_trees_all_equal(result.transition.oracle.cycle_words, _words(quotient))
        chex.assert_trees_all_equal(
            result.transition.oracle.next_cycle_words,
            _words(next_quotient),
        )
        assert int(result.state.step_count) == _INT32_MAX
        assert int(result.transition.oracle.cycle_index) == min(quotient, _INT32_MAX)
        assert int(result.transition.oracle.cycle_step) == cycle_step
        assert int(result.transition.oracle.segment_index) == segment_index
        assert int(result.transition.oracle.regime_id) == DEFAULT_REGIME_SCHEDULE[segment_index]
        assert bool(result.transition.oracle.schedule_switched) == (
            quotient != next_quotient or segment_index != next_segment_index
        )
        assert bool(result.lifetime_counter_valid)
        assert bool(result.lifetime_capacity_available)
        assert bool(result.state_valid)
        assert bool(result.input_valid)
        assert bool(result.candidate_state_finite)
        assert bool(result.update_applied)

    def scan_body(
        state: HiddenPartnerMappingState,
        action: jax.Array,
    ) -> tuple[HiddenPartnerMappingState, tuple[jax.Array, ...]]:
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
    expected_trace = jax.tree.map(
        lambda *values: jnp.stack(values),
        *(result.pre_step_words for result in eager_results),
    )
    chex.assert_trees_all_equal(scanned_trace[0], expected_trace)
    chex.assert_trees_all_equal(
        scanned_trace[1],
        jnp.stack([result.post_step_words for result in eager_results]),
    )
    chex.assert_trees_all_equal(
        scanned_trace[2],
        jnp.stack([result.transition.oracle.cycle_words for result in eager_results]),
    )
    chex.assert_trees_all_equal(
        scanned_trace[3],
        jnp.stack([result.transition.oracle.cycle_step for result in eager_results]),
    )
    chex.assert_trees_all_equal(
        scanned_trace[4],
        jnp.stack([result.transition.oracle.segment_index for result in eager_results]),
    )
    chex.assert_trees_all_equal(
        scanned_trace[5],
        jnp.stack([result.transition.oracle.regime_id for result in eager_results]),
    )
    assert bool(jnp.all(scanned_trace[6]))

    wide_step = (10 << 32) + 7
    wide = world.step_result(
        _valid_state_at(world, wide_step),
        jnp.asarray(0, dtype=jnp.int32),
    )
    wide_quotient, wide_cycle_step = divmod(wide_step, 18)
    chex.assert_trees_all_equal(wide.transition.oracle.cycle_words, _words(wide_quotient))
    assert int(wide.transition.oracle.cycle_index) == _INT32_MAX
    assert int(wide.transition.oracle.cycle_step) == wide_cycle_step
    assert bool(wide.update_applied)


@pytest.mark.parametrize("failure", ("counter", "nonfinite", "action"))
def test_invalid_transactions_roll_back_every_state_leaf_under_jit(failure: str) -> None:
    world = _world()
    valid = _valid_state_at(world, 1)
    action = jnp.asarray(0, dtype=jnp.int32)
    counter_valid = True
    state_valid = True
    input_valid = True

    if failure == "counter":
        state = _replace_state(
            valid,
            step_count=jnp.asarray(0, dtype=jnp.int32),
        )
        counter_valid = False
        state_valid = False
    elif failure == "nonfinite":
        state = _replace_state(valid, current_signals=valid.current_signals.at[2].set(jnp.nan))
        state_valid = False
    else:
        state = valid
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
        assert not bool(result.update_applied)
        assert not bool(result.transition.terminated)
        assert float(result.transition.discount) == 0.0
        assert int(result.transition.focal_action) == -1
        assert int(result.transition.partner_action) == -1
        _assert_transition_finite(result.transition)

    chex.assert_trees_all_equal(eager.transition, compiled.transition)


def test_terminal_exact_identity_returns_finite_terminal_noop() -> None:
    world = _world()
    state = _valid_state_at(world, (1 << 64) - 1)

    eager = world.step_result(state, jnp.asarray(1, dtype=jnp.int32))
    compiled = jax.jit(world.step_result)(state, jnp.asarray(1, dtype=jnp.int32))
    chex.assert_trees_all_equal(eager, compiled)

    _assert_state_equal_including_nan(eager.state, state)
    chex.assert_trees_all_equal(eager.pre_step_words, _words((1 << 64) - 1))
    chex.assert_trees_all_equal(eager.post_step_words, eager.pre_step_words)
    assert bool(eager.lifetime_counter_valid)
    assert not bool(eager.lifetime_capacity_available)
    assert bool(eager.state_valid)
    assert bool(eager.input_valid)
    assert bool(eager.candidate_state_finite)
    assert not bool(eager.update_applied)
    assert bool(eager.transition.terminated)
    assert float(eager.transition.discount) == 0.0
    _assert_transition_finite(eager.transition)


def test_static_contracts_reject_malformed_clock_and_action_inputs() -> None:
    world = _world()
    state = world.init(jr.key(9))

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

    # Authenticating before the int32 execution cast prevents wide integers
    # such as 2**32 from aliasing to action zero when x64 is available.
    with jax.enable_x64():
        wide_action = jnp.asarray(2**32, dtype=jnp.int64)
        rejected = jax.jit(world.step_result)(state, wide_action)
        assert not bool(rejected.input_valid)
        assert not bool(rejected.update_applied)
        _assert_state_equal_including_nan(rejected.state, state)


def test_schema_migrations_are_explicit_strict_and_unambiguous() -> None:
    config = _world().config
    payload = config.to_config()
    assert payload["schema"] == HIDDEN_PARTNER_MAPPING_CONFIG_SCHEMA
    assert payload["state_schema"] == HIDDEN_PARTNER_MAPPING_STATE_SCHEMA

    legacy_config = dict(payload)
    legacy_config.pop("schema")
    legacy_config.pop("state_schema")
    with pytest.raises(ValueError, match="explicit migration"):
        HiddenPartnerMappingConfig.from_config(legacy_config)
    assert migrate_legacy_hidden_partner_mapping_config(legacy_config) == config
    with pytest.raises(ValueError, match="not exact"):
        migrate_legacy_hidden_partner_mapping_config({**legacy_config, "extra": 1})

    world = _world()
    state = world.init(jr.key(12))
    for action in (0, 1, 1, 0):
        _, state = world.step(state, jnp.asarray(action, dtype=jnp.int32))
    legacy_state = {
        field.name: getattr(state, field.name)
        for field in dataclasses.fields(cast(Any, HiddenPartnerMappingState))
        if field.name != "step_words"
    }
    migrated = migrate_legacy_hidden_partner_mapping_state(legacy_state)
    chex.assert_trees_all_equal(migrated, state)

    with pytest.raises(ValueError, match="ambiguous"):
        migrate_legacy_hidden_partner_mapping_state(
            {
                **legacy_state,
                "step_count": jnp.asarray(_INT32_MAX, dtype=jnp.int32),
            }
        )
    with pytest.raises(ValueError, match="not exact"):
        migrate_legacy_hidden_partner_mapping_state({**legacy_state, "extra": 1})
    with pytest.raises(ValueError, match="inconsistent"):
        migrate_legacy_hidden_partner_mapping_state(
            {
                **legacy_state,
                "segment_ends": jnp.zeros((9,), dtype=jnp.int32),
            }
        )


def test_resource_accounting_matches_measured_state_and_exact_identity() -> None:
    world = _world()
    state = world.init(jr.key(13))
    budget = world.resource_budget

    assert budget.state_schema == HIDDEN_PARTNER_MAPPING_STATE_SCHEMA
    assert budget.exact_identity_uint32_scalars == 2
    assert budget.exact_identity_nbytes == HIDDEN_PARTNER_MAPPING_EXACT_IDENTITY_NBYTES
    assert budget.lifetime_identity_bits == 64
    assert budget.telemetry_saturation == _INT32_MAX
    assert measure_hidden_partner_mapping_state_nbytes(state) == budget.state_nbytes

    non_jax_state = _replace_state(
        state,
        current_signals=np.asarray(state.current_signals),
    )
    with pytest.raises(TypeError, match="current_signals"):
        measure_hidden_partner_mapping_state_nbytes(non_jax_state)


def test_horizon_surface_is_exported_from_module_streams_and_package_root() -> None:
    names = {
        "HIDDEN_PARTNER_MAPPING_CONFIG_SCHEMA",
        "HIDDEN_PARTNER_MAPPING_EXACT_IDENTITY_NBYTES",
        "HIDDEN_PARTNER_MAPPING_STATE_SCHEMA",
        "HiddenPartnerMappingConfig",
        "HiddenPartnerMappingOracle",
        "HiddenPartnerMappingResourceBudget",
        "HiddenPartnerMappingState",
        "HiddenPartnerMappingStepResult",
        "HiddenPartnerMappingTransition",
        "HiddenPartnerMappingWorld",
        "measure_hidden_partner_mapping_state_nbytes",
        "migrate_legacy_hidden_partner_mapping_config",
        "migrate_legacy_hidden_partner_mapping_state",
    }
    for name in names:
        assert name in mapping_module.__all__
        assert name in streams.__all__
        assert name in alberta.__all__
        assert getattr(streams, name) is getattr(mapping_module, name)
        assert getattr(alberta, name) is getattr(mapping_module, name)
