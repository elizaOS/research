"""Exact-horizon and transactional contracts for the learning-partner world."""

from __future__ import annotations

import dataclasses
import json
from typing import Any, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.streams import learning_partner as partner_module
from alberta_framework.streams.learning_partner import (
    LEARNING_PARTNER_WORLD_CONFIG_SCHEMA,
    LEARNING_PARTNER_WORLD_EXACT_IDENTITY_NBYTES,
    LEARNING_PARTNER_WORLD_INPUT_SCHEMA,
    LEARNING_PARTNER_WORLD_OUTPUT_SCHEMA,
    LEARNING_PARTNER_WORLD_STATE_SCHEMA,
    LearningPartnerWorld,
    LearningPartnerWorldConfig,
    LearningPartnerWorldState,
    learning_partner_world_keys,
    measure_learning_partner_world_state_nbytes,
    migrate_legacy_learning_partner_world_config,
    migrate_legacy_learning_partner_world_state,
)

pytestmark = pytest.mark.unit

_INT32_MAX = 2**31 - 1
_UINT32_MAX = 2**32 - 1


def _words(value: int) -> jax.Array:
    return jnp.asarray(
        ((value >> 32) & _UINT32_MAX, value & _UINT32_MAX),
        dtype=jnp.uint32,
    )


def _replace_state(
    state: LearningPartnerWorldState,
    **changes: Any,
) -> LearningPartnerWorldState:
    return cast(LearningPartnerWorldState, cast(Any, state).replace(**changes))


def _state_at(world: LearningPartnerWorld, exact_step: int) -> LearningPartnerWorldState:
    state = world.init(learning_partner_world_keys(jr.key(17)))
    return _replace_state(
        state,
        step_count=jnp.asarray(min(exact_step, _INT32_MAX), dtype=jnp.int32),
        step_words=_words(exact_step),
    )


def _assert_state_equal(
    actual: LearningPartnerWorldState,
    expected: LearningPartnerWorldState,
) -> None:
    for field in dataclasses.fields(cast(Any, LearningPartnerWorldState)):
        left = getattr(actual, field.name)
        right = getattr(expected, field.name)
        if field.name.endswith("_key"):
            left = jr.key_data(left)
            right = jr.key_data(right)
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))


def _assert_floating_leaves_finite(value: object) -> None:
    for leaf in jax.tree.leaves(value):
        if hasattr(leaf, "dtype") and jnp.issubdtype(leaf.dtype, jnp.inexact):
            assert bool(jnp.all(jnp.isfinite(leaf)))


def test_exact_division_and_phase_cycle_oracle_match_python_uint64() -> None:
    values = (
        0,
        1,
        _INT32_MAX,
        _UINT32_MAX,
        1 << 32,
        (1 << 32) + 7,
        1 << 63,
        (1 << 64) - 2,
        (1 << 64) - 1,
    )
    divisors = (1, 2, 3, 17, _INT32_MAX)
    pairs = tuple((value, divisor) for value in values for divisor in divisors)
    word_batch = jnp.stack([_words(value) for value, _ in pairs])
    divisor_batch = jnp.asarray([divisor for _, divisor in pairs], dtype=jnp.int32)
    exact_divmod = jax.jit(jax.vmap(partner_module._divmod_words_by_positive_int32))
    quotients, remainders = exact_divmod(word_batch, divisor_batch)
    for index, (value, divisor) in enumerate(pairs):
        quotient, remainder = divmod(value, divisor)
        chex.assert_trees_all_equal(quotients[index], _words(quotient))
        assert int(remainders[index]) == remainder

    world = LearningPartnerWorld(LearningPartnerWorldConfig(phase_length=17))
    for exact_step in (_INT32_MAX, 1 << 32, (7 << 32) + 29):
        result = world.step_result(
            _state_at(world, exact_step),
            jnp.asarray(1, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
        )
        phase, phase_step = divmod(exact_step, 17)
        next_phase, _ = divmod(exact_step + 1, 17)
        oracle = result.transition.oracle
        chex.assert_trees_all_equal(oracle.step_words, _words(exact_step))
        chex.assert_trees_all_equal(oracle.phase_words, _words(phase))
        chex.assert_trees_all_equal(oracle.cycle_words, _words(phase // 2))
        chex.assert_trees_all_equal(oracle.next_phase_words, _words(next_phase))
        assert int(oracle.phase_index) == min(phase, _INT32_MAX)
        assert int(oracle.phase_step) == phase_step
        assert int(oracle.context) == phase % 2
        assert int(oracle.cycle_index) == min(phase // 2, _INT32_MAX)
        assert int(oracle.next_context) == next_phase % 2
        assert bool(oracle.phase_switched) is (phase != next_phase)
        assert int(result.state.step_count) == _INT32_MAX
        assert bool(result.update_applied)


def test_uint32_carry_is_eager_jit_scan_equivalent_and_rng_exact() -> None:
    world = LearningPartnerWorld(LearningPartnerWorldConfig(phase_length=3))
    start = _UINT32_MAX - 1
    initial = _state_at(world, start)
    inputs = jnp.asarray(((0, 0), (1, 1), (1, 0), (0, 1)), dtype=jnp.int32)

    eager_results = []
    eager_state = initial
    for pair in inputs:
        result = world.step_result(eager_state, pair[0], pair[1])
        eager_results.append(result)
        eager_state = result.state

    compiled_step = jax.jit(world.step_result)
    compiled_state = initial
    for pair, eager in zip(inputs, eager_results, strict=True):
        compiled = compiled_step(compiled_state, pair[0], pair[1])
        chex.assert_trees_all_equal(compiled, eager)
        compiled_state = compiled.state

    def body(
        state: LearningPartnerWorldState,
        pair: jax.Array,
    ) -> tuple[LearningPartnerWorldState, tuple[jax.Array, ...]]:
        result = world.step_result(state, pair[0], pair[1])
        trace = (
            result.pre_step_words,
            result.post_step_words,
            result.transition.oracle.phase_words,
            result.transition.oracle.cycle_words,
            result.transition.oracle.context,
            result.update_applied,
        )
        return result.state, trace

    scanned_state, trace = jax.jit(lambda state, xs: jax.lax.scan(body, state, xs))(
        initial,
        inputs,
    )
    chex.assert_trees_all_equal(scanned_state, eager_state)
    chex.assert_trees_all_equal(compiled_state, eager_state)
    chex.assert_trees_all_equal(
        trace[0],
        jnp.stack([result.pre_step_words for result in eager_results]),
    )
    chex.assert_trees_all_equal(
        trace[1],
        jnp.stack([result.post_step_words for result in eager_results]),
    )
    assert bool(jnp.all(trace[-1]))
    chex.assert_trees_all_equal(eager_results[1].post_step_words, _words(1 << 32))


def test_invalid_transactions_and_all_ones_roll_back_every_rng_and_state_leaf() -> None:
    world = LearningPartnerWorld(LearningPartnerWorldConfig(phase_length=5))
    valid = _state_at(world, 9)
    compiled = jax.jit(world.step_with_delivery_result)
    cases = (
        (
            _replace_state(valid, step_count=jnp.asarray(8, dtype=jnp.int32)),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            False,
            True,
        ),
        (
            _replace_state(valid, cue=jnp.asarray(7, dtype=jnp.int32)),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            False,
            True,
        ),
        (
            valid,
            jnp.asarray(2, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            False,
            True,
        ),
        (
            _state_at(world, (1 << 64) - 1),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            True,
            False,
        ),
    )
    for state, helper, delivered, action, terminal, capacity in cases:
        eager = world.step_with_delivery_result(state, helper, delivered, action)
        jitted = compiled(state, helper, delivered, action)
        chex.assert_trees_all_equal(eager, jitted)
        _assert_state_equal(eager.state, state)
        chex.assert_trees_all_equal(eager.pre_step_words, state.step_words)
        chex.assert_trees_all_equal(eager.post_step_words, state.step_words)
        assert bool(eager.lifetime_capacity_available) is capacity
        assert not bool(eager.update_applied)
        assert bool(eager.transition.terminated) is terminal
        assert float(eager.transition.discount) == 0.0
        assert int(eager.transition.helper_message) == -1
        _assert_floating_leaves_finite(eager.transition)

    invalid = world.step_result(
        valid,
        jnp.asarray(9),
        jnp.asarray(0),
        partner_module.SHUFFLED_CHANNEL,
    )
    direct = world.step_result(valid, jnp.asarray(1), jnp.asarray(1))
    after_rejection = world.step_result(invalid.state, jnp.asarray(1), jnp.asarray(1))
    chex.assert_trees_all_equal(after_rejection, direct)


def test_static_input_state_and_channel_contracts_reject_aliasing() -> None:
    world = LearningPartnerWorld()
    state = world.init(learning_partner_world_keys(jr.key(9)))
    with pytest.raises(TypeError, match="step_words"):
        world.step_result(
            _replace_state(state, step_words=jnp.zeros((2,), dtype=jnp.int32)),
            jnp.asarray(0),
            jnp.asarray(0),
        )
    with pytest.raises(TypeError, match="step_words"):
        world.step_result(
            _replace_state(state, step_words=jnp.zeros((1,), dtype=jnp.uint32)),
            jnp.asarray(0),
            jnp.asarray(0),
        )
    with pytest.raises(TypeError, match="integer dtype"):
        world.step_result(state, jnp.asarray(0.0), jnp.asarray(0))
    with pytest.raises(ValueError, match="scalar"):
        world.step_result(state, jnp.asarray((0,)), jnp.asarray(0))
    with pytest.raises(TypeError, match="channel must be a string"):
        world.step_result(state, jnp.asarray(0), jnp.asarray(0), 0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unknown"):
        world.step_result(state, jnp.asarray(0), jnp.asarray(0), "future")  # type: ignore[arg-type]

    with jax.enable_x64():
        wide = jnp.asarray(2**32, dtype=jnp.int64)
        rejected = jax.jit(world.step_result)(state, wide, jnp.asarray(0, dtype=jnp.int64))
        assert not bool(rejected.input_valid)
        assert not bool(rejected.update_applied)
        _assert_state_equal(rejected.state, state)


def test_schemas_migrations_and_resources_are_strict_and_exact() -> None:
    config = LearningPartnerWorldConfig(phase_length=7)
    payload = json.loads(config.canonical_json())
    assert payload == config.to_dict() == config.to_config()
    assert payload["schema"] == LEARNING_PARTNER_WORLD_CONFIG_SCHEMA
    assert payload["state_schema"] == LEARNING_PARTNER_WORLD_STATE_SCHEMA
    assert payload["input_schema"] == LEARNING_PARTNER_WORLD_INPUT_SCHEMA
    assert payload["output_schema"] == LEARNING_PARTNER_WORLD_OUTPUT_SCHEMA
    assert LearningPartnerWorldConfig.from_config(payload) == config
    world = LearningPartnerWorld.from_config(LearningPartnerWorld(config).to_config())
    assert world.config == config

    legacy_config = {"phase_length": 7}
    with pytest.raises(ValueError, match="explicit migration"):
        LearningPartnerWorldConfig.from_config(legacy_config)
    assert migrate_legacy_learning_partner_world_config(legacy_config) == config
    with pytest.raises(ValueError, match="not exact"):
        migrate_legacy_learning_partner_world_config({**legacy_config, "extra": 1})
    for invalid in (0, True, _INT32_MAX + 1):
        with pytest.raises(ValueError, match="phase_length"):
            LearningPartnerWorldConfig(phase_length=invalid)

    state = world.init(learning_partner_world_keys(jr.key(5)))
    for _ in range(4):
        _, state = world.step(state, jnp.asarray(0), jnp.asarray(0))
    legacy_state = {
        field.name: getattr(state, field.name)
        for field in dataclasses.fields(cast(Any, LearningPartnerWorldState))
        if field.name != "step_words"
    }
    migrated = migrate_legacy_learning_partner_world_state(legacy_state)
    chex.assert_trees_all_equal(migrated, state)
    with pytest.raises(ValueError, match="ambiguous"):
        migrate_legacy_learning_partner_world_state(
            {**legacy_state, "step_count": jnp.asarray(_INT32_MAX, dtype=jnp.int32)}
        )
    with pytest.raises(ValueError, match="inconsistent"):
        migrate_legacy_learning_partner_world_state(
            {**legacy_state, "cue": jnp.asarray(3, dtype=jnp.int32)}
        )
    with pytest.raises(ValueError, match="not exact"):
        migrate_legacy_learning_partner_world_state({**legacy_state, "extra": 1})

    budget = world.resource_budget
    assert budget.state_schema == LEARNING_PARTNER_WORLD_STATE_SCHEMA
    assert budget.exact_identity_nbytes == LEARNING_PARTNER_WORLD_EXACT_IDENTITY_NBYTES == 8
    assert budget.lifetime_identity_bits == 64
    assert budget.telemetry_saturation == _INT32_MAX
    assert budget.state_nbytes == 32
    assert measure_learning_partner_world_state_nbytes(state) == budget.state_nbytes
    with pytest.raises(TypeError, match="cue"):
        measure_learning_partner_world_state_nbytes(
            _replace_state(state, cue=np.asarray(state.cue))
        )


def test_module_surface_and_output_record_bind_all_contract_versions() -> None:
    names = {
        "LEARNING_PARTNER_WORLD_CONFIG_SCHEMA",
        "LEARNING_PARTNER_WORLD_EXACT_IDENTITY_NBYTES",
        "LEARNING_PARTNER_WORLD_INPUT_SCHEMA",
        "LEARNING_PARTNER_WORLD_OUTPUT_SCHEMA",
        "LEARNING_PARTNER_WORLD_STATE_SCHEMA",
        "LearningPartnerWorldResourceBudget",
        "LearningPartnerWorldStepResult",
        "measure_learning_partner_world_state_nbytes",
        "migrate_legacy_learning_partner_world_config",
        "migrate_legacy_learning_partner_world_state",
    }
    assert names <= set(partner_module.__all__)
    world = LearningPartnerWorld(LearningPartnerWorldConfig(phase_length=2))
    result = world.step_result(
        world.init(learning_partner_world_keys(jr.key(1))),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
    )
    assert {field.name for field in dataclasses.fields(cast(Any, result))} == {
        "transition",
        "state",
        "pre_step_words",
        "post_step_words",
        "lifetime_counter_valid",
        "lifetime_capacity_available",
        "state_valid",
        "input_valid",
        "candidate_state_finite",
        "update_applied",
    }
    assert bool(result.lifetime_counter_valid)
    assert bool(result.state_valid)
    assert bool(result.input_valid)
    assert bool(result.candidate_state_finite)
    assert bool(result.update_applied)
    assert result.pre_step_words.shape == result.post_step_words.shape == (2,)
    assert result.pre_step_words.dtype == result.post_step_words.dtype == jnp.uint32
