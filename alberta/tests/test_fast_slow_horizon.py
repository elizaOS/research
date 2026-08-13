# mypy: disable-error-code="attr-defined,call-arg,no-any-return"
"""Exact-lifetime and fail-closed contracts for the fast/slow learner."""

from __future__ import annotations

import dataclasses
from typing import Any

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.fast_slow import (
    FAST_SLOW_CONFIG_SCHEMA,
    FAST_SLOW_EXACT_LIFETIME_IDENTITY_NBYTES,
    FAST_SLOW_LIFETIME_COUNTER_NBYTES,
    FAST_SLOW_RESOURCE_SCHEMA,
    FAST_SLOW_RESULT_SCHEMA,
    FAST_SLOW_STATE_SCHEMA,
    FastSlowConfig,
    FastSlowLearner,
    FastSlowParams,
    FastSlowState,
    measure_fast_slow_state_nbytes,
    migrate_legacy_fast_slow_state,
    run_fast_slow_arrays,
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


def _learner(**overrides: Any) -> FastSlowLearner:
    values: dict[str, Any] = {
        "input_dim": 2,
        "output_dim": 1,
        "hidden_dim": 3,
    }
    values.update(overrides)
    return FastSlowLearner(FastSlowConfig(**values))


def _state_at(learner: FastSlowLearner, value: int) -> FastSlowState:
    return learner.init(jr.key(17)).replace(
        step_count=_telemetry(value),
        step_words=_words(value),
    )


def _update(
    learner: FastSlowLearner,
    state: FastSlowState,
    observation: jax.Array | None = None,
    target: jax.Array | None = None,
) -> Any:
    return learner.update(
        state,
        (
            jnp.asarray([0.25, -0.75], dtype=jnp.float32)
            if observation is None
            else observation
        ),
        jnp.asarray([0.5], dtype=jnp.float32) if target is None else target,
    )


def _assert_neutral_rejection(result: Any, source: FastSlowState) -> None:
    assert not bool(result.update_applied)
    chex.assert_trees_all_equal(result.state, source)
    chex.assert_trees_all_equal(result.pre_step_words, result.post_step_words)
    chex.assert_trees_all_equal(result.prediction, jnp.zeros((1,), dtype=jnp.float32))
    chex.assert_trees_all_equal(result.error, jnp.zeros((1,), dtype=jnp.float32))
    chex.assert_trees_all_equal(result.metrics, jnp.zeros((6,), dtype=jnp.float32))


def test_valid_short_run_is_bit_exact_and_scan_preserves_metrics_contract() -> None:
    """Horizon guards must not perturb the historical finite update graph."""

    learner = _learner()
    observations = jnp.asarray(
        [[0.25, -0.75], [1.0, 0.5], [-0.5, 0.125]],
        dtype=jnp.float32,
    )
    targets = jnp.asarray([[0.5], [-1.25], [0.75]], dtype=jnp.float32)
    expected_predictions = jnp.asarray(
        [[0.0], [0.005234825424849987], [-0.012006797827780247]],
        dtype=jnp.float32,
    )
    expected_errors = jnp.asarray(
        [[0.5], [-1.2552348375320435], [0.7620068192481995]],
        dtype=jnp.float32,
    )
    expected_metrics = jnp.asarray(
        [
            [
                0.125,
                0.25,
                0.5009610056877136,
                0.6799513697624207,
                0.008661797270178795,
                0.003458072431385517,
            ],
            [
                0.7878072261810303,
                1.5756144523620605,
                0.49880802631378174,
                1.8743857145309448,
                0.03539344295859337,
                0.01424158550798893,
            ],
            [
                0.29032719135284424,
                0.5806543827056885,
                0.5000540018081665,
                0.8759019374847412,
                0.03785577788949013,
                0.015506349503993988,
            ],
        ],
        dtype=jnp.float32,
    )

    state = learner.init(jr.key(17))
    predictions = []
    errors = []
    metrics = []
    for observation, target in zip(observations, targets, strict=True):
        with jax.disable_jit():
            eager = learner.update(state, observation, target)
        compiled = learner.update(state, observation, target)
        chex.assert_trees_all_close(eager, compiled, rtol=1.0e-6, atol=1.0e-7)
        assert bool(eager.update_applied)
        predictions.append(compiled.prediction)
        errors.append(compiled.error)
        metrics.append(compiled.metrics)
        state = compiled.state

    chex.assert_trees_all_equal(jnp.stack(predictions), expected_predictions)
    chex.assert_trees_all_equal(jnp.stack(errors), expected_errors)
    chex.assert_trees_all_equal(jnp.stack(metrics), expected_metrics)

    scanned = run_fast_slow_arrays(
        learner,
        observations,
        targets,
        key=jr.key(17),
    )
    chex.assert_shape(scanned.metrics, (3, 6))
    chex.assert_trees_all_equal(scanned.metrics, expected_metrics)
    chex.assert_trees_all_equal(
        scanned.update_applied,
        jnp.ones((3,), dtype=jnp.bool_),
    )


def test_low_word_carry_and_int32_telemetry_saturation_are_independent() -> None:
    learner = _learner()
    carried = _update(learner, _state_at(learner, _UINT32_MAX))

    assert bool(carried.lifetime_counter_valid)
    assert bool(carried.lifetime_capacity_available)
    assert bool(carried.update_applied)
    chex.assert_trees_all_equal(carried.pre_step_words, _words(_UINT32_MAX))
    chex.assert_trees_all_equal(carried.post_step_words, _words(1 << 32))
    chex.assert_trees_all_equal(carried.state.step_count, _telemetry(1 << 32))

    before_saturation = _state_at(learner, _INT32_MAX - 1)
    saturated = _update(learner, before_saturation)
    beyond = _update(learner, saturated.state)
    chex.assert_trees_all_equal(saturated.state.step_count, _telemetry(_INT32_MAX))
    chex.assert_trees_all_equal(beyond.state.step_count, _telemetry(_INT32_MAX + 1))
    chex.assert_trees_all_equal(beyond.state.step_words, _words(_INT32_MAX + 1))
    assert bool(beyond.update_applied)


def test_all_ones_lifetime_is_a_permanent_whole_state_fail_stop() -> None:
    learner = _learner()
    terminal = _state_at(learner, _UINT64_MAX)

    with jax.disable_jit():
        eager = _update(learner, terminal)
    compiled = learner.update(
        terminal,
        jnp.asarray([0.25, -0.75], dtype=jnp.float32),
        jnp.asarray([0.5], dtype=jnp.float32),
    )
    chex.assert_trees_all_equal(eager, compiled)
    assert bool(eager.state_valid)
    assert not bool(eager.lifetime_capacity_available)
    _assert_neutral_rejection(eager, terminal)
    _assert_neutral_rejection(_update(learner, eager.state), terminal)


def test_scan_exposes_applied_facts_and_fail_stops_at_terminal() -> None:
    learner = _learner()
    observations = jnp.asarray(
        [[0.25, -0.75], [1.0, 0.5], [-0.5, 0.125]],
        dtype=jnp.float32,
    )
    targets = jnp.asarray([[0.5], [-1.25], [0.75]], dtype=jnp.float32)
    initial = _state_at(learner, _UINT64_MAX - 1)

    result = run_fast_slow_arrays(
        learner,
        observations,
        targets,
        state=initial,
    )

    chex.assert_trees_all_equal(result.state.step_words, _words(_UINT64_MAX))
    chex.assert_trees_all_equal(
        result.update_applied,
        jnp.asarray([True, False, False], dtype=jnp.bool_),
    )
    chex.assert_trees_all_equal(
        result.pre_step_words,
        jnp.stack((_words(_UINT64_MAX - 1), _words(_UINT64_MAX), _words(_UINT64_MAX))),
    )
    chex.assert_trees_all_equal(result.post_step_words[1:], result.pre_step_words[1:])
    chex.assert_trees_all_equal(result.metrics[1:], jnp.zeros((2, 6), dtype=jnp.float32))


@pytest.mark.parametrize("bad_value", [jnp.nan, jnp.inf, -jnp.inf])
def test_nonfinite_inputs_reject_atomically_and_return_neutral_diagnostics(
    bad_value: float,
) -> None:
    learner = _learner()
    state = learner.init(jr.key(1))
    bad_observation = jnp.asarray([bad_value, 0.0], dtype=jnp.float32)
    bad_target = jnp.asarray([bad_value], dtype=jnp.float32)

    observation_result = _update(learner, state, observation=bad_observation)
    assert not bool(observation_result.observation_valid)
    assert bool(observation_result.target_valid)
    assert not bool(observation_result.input_valid)
    _assert_neutral_rejection(observation_result, state)

    target_result = _update(learner, state, target=bad_target)
    assert bool(target_result.observation_valid)
    assert not bool(target_result.target_valid)
    assert not bool(target_result.input_valid)
    _assert_neutral_rejection(target_result, state)


def test_corrupt_state_and_counter_reject_without_partial_fast_decay() -> None:
    learner = _learner(fast_decay=0.5)
    state = learner.init(jr.key(3))
    params = state.params.replace(
        fast_kernel=jnp.ones_like(state.params.fast_kernel),
    )
    corrupt_counter = state.replace(
        params=params,
        step_words=_words(9),
        step_count=jnp.asarray(8, dtype=jnp.int32),
    )
    counter_result = _update(learner, corrupt_counter)
    assert not bool(counter_result.lifetime_counter_valid)
    assert not bool(counter_result.state_valid)
    _assert_neutral_rejection(counter_result, corrupt_counter)

    corrupt_params = state.replace(
        params=state.params.replace(
            fast_kernel=state.params.fast_kernel.at[0, 0].set(jnp.nan),
        )
    )
    state_result = _update(learner, corrupt_params)
    assert bool(state_result.lifetime_counter_valid)
    assert not bool(state_result.state_valid)
    _assert_neutral_rejection(state_result, corrupt_params)


def test_nonfinite_candidate_rolls_back_and_a_retry_matches_direct_update() -> None:
    explosive = _learner()
    state = explosive.init(jr.key(5))
    maximum = np.finfo(np.float32).max
    rejected = _update(
        explosive,
        state,
        observation=jnp.full((2,), maximum, dtype=jnp.float32),
        target=jnp.asarray([maximum], dtype=jnp.float32),
    )
    assert bool(rejected.input_valid)
    assert not bool(rejected.candidate_state_valid)
    _assert_neutral_rejection(rejected, state)

    learner = _learner()
    retry_source = learner.init(jr.key(11))
    first = _update(
        learner,
        retry_source,
        observation=jnp.asarray([jnp.nan, 0.0], dtype=jnp.float32),
    )
    retried = _update(learner, first.state)
    direct = _update(learner, retry_source)
    chex.assert_trees_all_equal(retried, direct)


def test_exact_shapes_and_dtypes_are_enforced_before_update_or_scan() -> None:
    learner = _learner()
    state = learner.init(jr.key(0))
    with pytest.raises(ValueError, match="observation"):
        _update(learner, state, observation=jnp.zeros((3,), dtype=jnp.float32))
    with pytest.raises(TypeError, match="observation"):
        _update(learner, state, observation=jnp.zeros((2,), dtype=jnp.float16))
    with pytest.raises(ValueError, match="target"):
        _update(learner, state, target=jnp.zeros((), dtype=jnp.float32))
    with pytest.raises(TypeError, match="target"):
        _update(learner, state, target=jnp.zeros((1,), dtype=jnp.int32))
    with pytest.raises(ValueError, match="step_words"):
        _update(learner, state.replace(step_words=jnp.zeros((3,), dtype=jnp.uint32)))
    with pytest.raises(TypeError, match="step_words"):
        _update(learner, state.replace(step_words=jnp.zeros((2,), dtype=jnp.int32)))
    with pytest.raises(ValueError, match="observations"):
        run_fast_slow_arrays(
            learner,
            jnp.zeros((2, 3), dtype=jnp.float32),
            jnp.zeros((2, 1), dtype=jnp.float32),
            state=state,
        )


def test_strict_versioned_config_preserves_exact_legacy_roundtrip() -> None:
    config = FastSlowConfig(input_dim=5, output_dim=2, hidden_dim=7, fast_decay=0.9)
    current = config.to_config()
    assert current["schema"] == FAST_SLOW_CONFIG_SCHEMA
    assert current["state_schema"] == FAST_SLOW_STATE_SCHEMA
    assert current["result_schema"] == FAST_SLOW_RESULT_SCHEMA
    assert current["resource_schema"] == FAST_SLOW_RESOURCE_SCHEMA
    assert FastSlowConfig.from_config(current) == config

    legacy = {
        key: value
        for key, value in current.items()
        if key not in {"schema", "state_schema", "result_schema", "resource_schema"}
    }
    assert FastSlowConfig.from_config(legacy) == config
    with pytest.raises(ValueError, match="fields"):
        FastSlowConfig.from_config({**current, "extra": 1})
    with pytest.raises(ValueError, match="schema"):
        FastSlowConfig.from_config({**current, "state_schema": "wrong"})

    learner = FastSlowLearner(config)
    learner_current = learner.to_config()
    assert FastSlowLearner.from_config(learner_current).config == config
    learner_legacy = {"type": "FastSlowLearner", "config": legacy}
    assert FastSlowLearner.from_config(learner_legacy).config == config


def test_legacy_state_migration_is_host_only_and_unambiguous() -> None:
    learner = _learner()
    state = learner.init(jr.key(8)).replace(step_count=jnp.asarray(19, dtype=jnp.int32))
    legacy = {"params": state.params, "step_count": state.step_count}
    migrated = migrate_legacy_fast_slow_state(learner, legacy)
    chex.assert_trees_all_equal(migrated.step_words, _words(19))
    assert bool(learner.state_is_valid(migrated))

    @dataclasses.dataclass(frozen=True)
    class LegacyState:
        params: FastSlowParams
        step_count: jax.Array

    migrated_dataclass = migrate_legacy_fast_slow_state(
        learner,
        LegacyState(params=state.params, step_count=state.step_count),
    )
    chex.assert_trees_all_equal(migrated_dataclass, migrated)

    with pytest.raises(ValueError, match="fields"):
        migrate_legacy_fast_slow_state(learner, {**legacy, "extra": 1})
    with pytest.raises(ValueError, match="ambiguous"):
        migrate_legacy_fast_slow_state(
            learner,
            {**legacy, "step_count": jnp.asarray(_INT32_MAX, dtype=jnp.int32)},
        )
    with pytest.raises(ValueError, match="ambiguous"):
        migrate_legacy_fast_slow_state(
            learner,
            {**legacy, "step_count": jnp.asarray(-1, dtype=jnp.int32)},
        )


def test_state_and_resource_records_measure_every_persistent_byte() -> None:
    learner = _learner()
    state = learner.init(jr.key(0))
    measured = measure_fast_slow_state_nbytes(state)
    record = learner.state_record(state)
    resources = learner.resource_record(state)

    assert record.schema == FAST_SLOW_STATE_SCHEMA
    assert record.config_schema == FAST_SLOW_CONFIG_SCHEMA
    assert record.state_nbytes == measured
    assert record.parameter_nbytes + FAST_SLOW_LIFETIME_COUNTER_NBYTES == measured
    assert record.step_words == (0, 0)
    assert resources.schema == FAST_SLOW_RESOURCE_SCHEMA
    assert resources.state_schema == FAST_SLOW_STATE_SCHEMA
    assert resources.state_nbytes == measured
    assert resources.parameter_nbytes == record.parameter_nbytes
    assert resources.exact_lifetime_identity_nbytes == (
        FAST_SLOW_EXACT_LIFETIME_IDENTITY_NBYTES
    )
    assert resources.lifetime_counter_nbytes == FAST_SLOW_LIFETIME_COUNTER_NBYTES
    assert resources.legacy_state_nbytes + resources.versioned_state_delta_nbytes == measured
    assert resources.versioned_state_delta_nbytes == 8
    assert resources.persistent_capacity_growth == 0
