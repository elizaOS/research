"""Exact-lifetime and atomic-transaction contracts for dream rollouts."""

from __future__ import annotations

import dataclasses
from typing import Any

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from alberta_framework.core.dreaming import (
    DREAM_ROLLOUT_CLOCK_DELTA_NBYTES,
    DREAM_ROLLOUT_CLOCK_NBYTES,
    DREAM_ROLLOUT_CONFIG_SCHEMA,
    DREAM_ROLLOUT_RESOURCE_SCHEMA,
    DREAM_ROLLOUT_RESULT_SCHEMA,
    DREAM_ROLLOUT_STATE_SCHEMA,
    DreamBehaviorModelPrediction,
    DreamRolloutConfig,
    DreamWorldModelPrediction,
    dream_one_step_result,
    dream_rollout,
    dream_rollout_resource_budget,
    dream_rollout_state_is_valid,
    init_dream_rollout_state,
    measure_dream_rollout_state_nbytes,
    migrate_legacy_dream_rollout_config,
    migrate_legacy_dream_rollout_state,
)

pytestmark = pytest.mark.unit

_INT32_MAX = 2**31 - 1
_UINT32_MAX = 2**32 - 1
_UINT64_MAX = 2**64 - 1


def _words(value: int) -> jax.Array:
    return jnp.asarray(
        ((value >> 32) & _UINT32_MAX, value & _UINT32_MAX),
        dtype=jnp.uint32,
    )


def _telemetry(value: int) -> jax.Array:
    return jnp.asarray(min(value, _INT32_MAX), dtype=jnp.int32)


@chex.dataclass(frozen=True)
class _BehaviorState:
    action: jax.Array


class _Behavior:
    def sample_action(
        self,
        state: _BehaviorState,
        observation: jax.Array,
        key: Any,
    ) -> DreamBehaviorModelPrediction:
        del observation, key
        return DreamBehaviorModelPrediction(
            action=state.action,
            action_probability=jnp.asarray(1.0, dtype=jnp.float32),
            log_probability=jnp.asarray(0.0, dtype=jnp.float32),
        )


@chex.dataclass(frozen=True)
class _WorldState:
    finite: jax.Array


class _World:
    def predict(
        self,
        state: _WorldState,
        observation: jax.Array,
        action: jax.Array,
        key: Any,
    ) -> DreamWorldModelPrediction:
        del action, key
        next_observation = observation + jnp.asarray((0.25, -0.5), dtype=jnp.float32)
        next_observation = jnp.where(state.finite, next_observation, jnp.nan)
        return DreamWorldModelPrediction(
            next_observation=next_observation,
            reward=jnp.where(state.finite, 0.75, jnp.nan).astype(jnp.float32),
            discount=jnp.asarray(0.9, dtype=jnp.float32),
            terminated=jnp.asarray(False, dtype=jnp.bool_),
            confidence=jnp.asarray(1.0, dtype=jnp.float32),
            model_error=jnp.asarray(0.0, dtype=jnp.float32),
        )


def _initial(*, value: int = 0):  # type: ignore[no-untyped-def]
    state = init_dream_rollout_state(
        jnp.asarray((1.0, 2.0), dtype=jnp.float32),
        jr.key(7),
    )
    return state.replace(step_count=_telemetry(value), step_words=_words(value))


def _step(state, *, finite: bool = True):  # type: ignore[no-untyped-def]
    return dream_one_step_result(
        _World(),
        _WorldState(finite=jnp.asarray(finite, dtype=jnp.bool_)),
        _Behavior(),
        _BehaviorState(action=jnp.asarray(1, dtype=jnp.int32)),
        state,
        DreamRolloutConfig(rollout_horizon=3),
    )


def test_config_identity_is_strict_complete_and_finite() -> None:
    config = DreamRolloutConfig(
        rollout_horizon=3,
        confidence_threshold=0.2,
        max_model_error=4.0,
        discount_floor=0.1,
        stop_on_terminal=False,
    )
    payload = config.to_config()
    assert payload["config_schema"] == DREAM_ROLLOUT_CONFIG_SCHEMA
    assert payload["state_schema"] == DREAM_ROLLOUT_STATE_SCHEMA
    assert payload["result_schema"] == DREAM_ROLLOUT_RESULT_SCHEMA
    assert DreamRolloutConfig.from_config(payload) == config

    with pytest.raises(ValueError, match="fields"):
        DreamRolloutConfig.from_config({**payload, "extra": 1})
    with pytest.raises(ValueError, match="schema"):
        DreamRolloutConfig.from_config({**payload, "state_schema": "legacy"})
    with pytest.raises(ValueError, match="finite"):
        DreamRolloutConfig(max_model_error=float("inf"))
    with pytest.raises(ValueError, match="integer"):
        DreamRolloutConfig(rollout_horizon=True)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("start", "expected"),
    [
        (_UINT32_MAX, 1 << 32),
        ((1 << 32) + 11, (1 << 32) + 12),
        ((1 << 63) + 9, (1 << 63) + 10),
    ],
)
def test_exact_clock_crosses_uint32_and_saturates_telemetry(
    start: int,
    expected: int,
) -> None:
    state = _initial(value=start)
    eager = _step(state)
    compiled = jax.jit(_step)(state)

    assert bool(eager.update_applied)
    chex.assert_trees_all_equal(eager.pre_step_words, _words(start))
    chex.assert_trees_all_equal(eager.post_step_words, _words(expected))
    chex.assert_trees_all_equal(eager.state.step_words, _words(expected))
    chex.assert_trees_all_equal(eager.state.step_count, _telemetry(expected))
    chex.assert_trees_all_equal(eager.transition.step_index_words, _words(start))
    chex.assert_trees_all_equal(eager, compiled)


def test_terminal_and_nonfinite_prediction_reject_atomically() -> None:
    terminal = _initial(value=_UINT64_MAX)
    terminal_result = jax.jit(_step)(terminal)
    assert not bool(terminal_result.lifetime_capacity_available)
    assert not bool(terminal_result.update_applied)
    chex.assert_trees_all_equal(terminal_result.state, terminal)
    assert not bool(terminal_result.transition.valid)
    assert bool(
        jax.tree.all(
            jax.tree.map(
                lambda x: jnp.all(jnp.isfinite(x)),
                terminal_result.transition,
            )
        )
    )

    initial = _initial()
    rejected = _step(initial, finite=False)
    assert not bool(rejected.prediction_valid)
    assert not bool(rejected.update_applied)
    chex.assert_trees_all_equal(rejected.state, initial)
    assert not bool(rejected.transition.valid)
    assert bool(jax.tree.all(jax.tree.map(lambda x: jnp.all(jnp.isfinite(x)), rejected.transition)))


def test_corrupt_counter_refuses_without_advancing_key_or_state() -> None:
    corrupt = _initial(value=5).replace(step_count=jnp.asarray(4, dtype=jnp.int32))
    assert not bool(dream_rollout_state_is_valid(corrupt))
    result = _step(corrupt)
    assert not bool(result.state_valid)
    assert not bool(result.update_applied)
    chex.assert_trees_all_equal(result.state, corrupt)
    chex.assert_trees_all_equal(result.pre_step_words, result.post_step_words)


def test_scan_fail_stops_at_exact_terminal_without_wrap() -> None:
    config = DreamRolloutConfig(rollout_horizon=3)
    initial = _initial(value=_UINT64_MAX - 1)
    rollout = dream_rollout(
        _World(),
        _WorldState(finite=jnp.asarray(True, dtype=jnp.bool_)),
        _Behavior(),
        _BehaviorState(action=jnp.asarray(1, dtype=jnp.int32)),
        initial,
        config,
    )

    chex.assert_trees_all_equal(rollout.state.step_words, _words(_UINT64_MAX))
    chex.assert_trees_all_equal(
        rollout.transitions.step_index_words,
        jnp.stack((_words(_UINT64_MAX - 1), _words(_UINT64_MAX), _words(_UINT64_MAX))),
    )
    chex.assert_trees_all_equal(
        rollout.transitions.valid,
        jnp.asarray((True, False, False), dtype=jnp.bool_),
    )


def test_state_contract_rejects_wrong_exact_clock_shape_and_dtype() -> None:
    state = _initial()
    with pytest.raises(TypeError, match="step_words"):
        _step(state.replace(step_words=jnp.zeros((2,), dtype=jnp.int32)))
    with pytest.raises(ValueError, match="step_words"):
        _step(state.replace(step_words=jnp.zeros((3,), dtype=jnp.uint32)))


def test_legacy_migrations_are_explicit_and_representable_only() -> None:
    config = DreamRolloutConfig(rollout_horizon=5, max_model_error=7.0)
    legacy_config = {"type": "DreamRolloutConfig", **dataclasses.asdict(config)}
    assert migrate_legacy_dream_rollout_config(legacy_config) == config
    with pytest.raises(ValueError, match="fields"):
        migrate_legacy_dream_rollout_config({**legacy_config, "extra": 1})

    state = _initial(value=27)
    legacy_state = {
        "observation": state.observation,
        "rng_key": state.rng_key,
        "active": state.active,
        "cumulative_confidence": state.cumulative_confidence,
        "step_count": state.step_count,
    }
    chex.assert_trees_all_equal(migrate_legacy_dream_rollout_state(legacy_state), state)
    with pytest.raises(ValueError, match="ambiguous"):
        migrate_legacy_dream_rollout_state(
            {**legacy_state, "step_count": jnp.asarray(_INT32_MAX, dtype=jnp.int32)}
        )
    with pytest.raises(ValueError, match="fields"):
        migrate_legacy_dream_rollout_state({**legacy_state, "extra": 1})


def test_resource_budget_matches_concrete_state_and_declared_work() -> None:
    state = _initial()
    config = DreamRolloutConfig(rollout_horizon=7)
    budget = dream_rollout_resource_budget(state, config)

    assert budget.schema == DREAM_ROLLOUT_RESOURCE_SCHEMA
    assert budget.state_schema == DREAM_ROLLOUT_STATE_SCHEMA
    assert budget.state_nbytes == measure_dream_rollout_state_nbytes(state)
    assert budget.exact_clock_nbytes == DREAM_ROLLOUT_CLOCK_NBYTES
    assert budget.exact_clock_delta_nbytes == DREAM_ROLLOUT_CLOCK_DELTA_NBYTES
    assert budget.maximum_steps == 7
    assert budget.maximum_behavior_calls == 7
    assert budget.maximum_world_calls == 7
    assert budget.maximum_key_splits == 7
    assert budget.persistent_capacity_growth == 0
