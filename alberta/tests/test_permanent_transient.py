# mypy: disable-error-code="attr-defined,call-arg,no-any-return"
"""Contracts for the Alberta-derived permanent/transient regression baseline."""

from __future__ import annotations

from typing import Any

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.permanent_transient import (
    PERMANENT_TRANSIENT_CONFIG_SCHEMA,
    PERMANENT_TRANSIENT_DESIGN_SCHEMA,
    PERMANENT_TRANSIENT_EXACT_LIFETIME_NBYTES,
    PERMANENT_TRANSIENT_LIFETIME_COUNTER_NBYTES,
    PERMANENT_TRANSIENT_RESOURCE_SCHEMA,
    PERMANENT_TRANSIENT_RESULT_SCHEMA,
    PERMANENT_TRANSIENT_STATE_SCHEMA,
    AlbertaPermanentTransientConfig,
    AlbertaPermanentTransientLearner,
    AlbertaPermanentTransientState,
    measure_permanent_transient_state_nbytes,
    permanent_transient_design_record,
    run_permanent_transient_arrays,
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


def _learner(**overrides: Any) -> AlbertaPermanentTransientLearner:
    values: dict[str, Any] = {
        "input_dim": 2,
        "output_dim": 1,
        "permanent_hidden_dim": 3,
        "transient_hidden_dim": 4,
    }
    values.update(overrides)
    return AlbertaPermanentTransientLearner(AlbertaPermanentTransientConfig(**values))


def _state_at(
    learner: AlbertaPermanentTransientLearner,
    value: int,
) -> AlbertaPermanentTransientState:
    return learner.init(jr.key(7)).replace(
        step_count=_telemetry(value),
        step_words=_words(value),
    )


def _update(
    learner: AlbertaPermanentTransientLearner,
    state: AlbertaPermanentTransientState,
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


def _assert_rejected(result: Any, source: AlbertaPermanentTransientState) -> None:
    assert not bool(result.update_applied)
    chex.assert_trees_all_equal(result.state, source)
    chex.assert_trees_all_equal(result.pre_step_words, result.post_step_words)
    chex.assert_trees_all_equal(result.prediction, jnp.zeros((1,), dtype=jnp.float32))
    chex.assert_trees_all_equal(result.error, jnp.zeros((1,), dtype=jnp.float32))
    chex.assert_trees_all_equal(result.metrics, jnp.zeros((8,), dtype=jnp.float32))


def test_design_record_refuses_source_faithfulness_and_enumerates_departures() -> None:
    record = permanent_transient_design_record()

    assert record.schema == PERMANENT_TRANSIENT_DESIGN_SCHEMA
    assert record.method_name == "Alberta-derived online permanent/transient regression"
    assert record.source_faithful is False
    assert record.primary_paper_license == "CC BY 4.0"
    assert record.reference_code_license == "MIT"
    assert "openreview.net/forum?id=5XfxEQ2SCt" in record.primary_paper_url
    assert len(record.departures) >= 6
    assert any("task" in departure for departure in record.departures)
    assert any("buffer" in departure for departure in record.departures)
    assert any("supervised" in departure for departure in record.departures)


def test_representations_heads_and_updates_are_structurally_separate() -> None:
    learner = _learner()
    initial = learner.init(jr.key(0))
    params = initial.params

    chex.assert_shape(params.permanent_encoder_kernel, (2, 3))
    chex.assert_shape(params.permanent_head_kernel, (3, 1))
    chex.assert_shape(params.transient_encoder_kernel, (2, 4))
    chex.assert_shape(params.transient_head_kernel, (4, 1))
    assert not np.array_equal(
        np.asarray(params.permanent_encoder_kernel).reshape(-1)[:6],
        np.asarray(params.transient_encoder_kernel).reshape(-1)[:6],
    )

    first = _update(learner, initial)
    fifth = first
    for _ in range(4):
        fifth = _update(learner, fifth.state)
    assert bool(first.update_applied)
    assert bool(fifth.update_applied)
    assert not np.array_equal(
        np.asarray(fifth.state.params.permanent_encoder_kernel),
        np.asarray(initial.params.permanent_encoder_kernel),
    )
    assert not np.array_equal(
        np.asarray(fifth.state.params.transient_encoder_kernel),
        np.asarray(initial.params.transient_encoder_kernel),
    )


def test_no_consolidation_is_equal_state_and_work_but_freezes_permanent_subtree() -> None:
    ordinary = _learner()
    ablation = _learner(
        permanent_encoder_step_size=0.0,
        permanent_head_step_size=0.0,
    )
    key = jr.key(3)
    ordinary_initial = ordinary.init(key)
    ablation_initial = ablation.init(key)
    chex.assert_trees_all_equal(ordinary_initial, ablation_initial)
    assert ordinary.resource_record().state_nbytes == ablation.resource_record().state_nbytes
    assert ordinary.resource_record().maximum_gradient_evaluations_per_update == 2
    assert ablation.resource_record().maximum_gradient_evaluations_per_update == 2

    observations = jnp.asarray(
        [[0.25, -0.75], [1.0, 0.5], [-0.5, 0.125]],
        dtype=jnp.float32,
    )
    targets = jnp.asarray([[0.5], [-1.25], [0.75]], dtype=jnp.float32)
    result = run_permanent_transient_arrays(
        ablation,
        observations,
        targets,
        state=ablation_initial,
    )
    chex.assert_trees_all_equal(
        result.state.params.permanent_encoder_kernel,
        ablation_initial.params.permanent_encoder_kernel,
    )
    chex.assert_trees_all_equal(
        result.state.params.permanent_head_kernel,
        ablation_initial.params.permanent_head_kernel,
    )
    assert not np.array_equal(
        np.asarray(result.state.params.transient_head_kernel),
        np.asarray(ablation_initial.params.transient_head_kernel),
    )


def test_eager_jit_and_scan_are_deterministic_with_fixed_metrics_shape() -> None:
    learner = _learner()
    state = learner.init(jr.key(11))
    observation = jnp.asarray([0.25, -0.75], dtype=jnp.float32)
    target = jnp.asarray([0.5], dtype=jnp.float32)
    with jax.disable_jit():
        eager = learner.update(state, observation, target)
    compiled = learner.update(state, observation, target)
    chex.assert_trees_all_close(eager, compiled, rtol=1.0e-6, atol=1.0e-7)
    assert bool(eager.update_applied)

    observations = jnp.stack((observation, -observation, observation))
    targets = jnp.asarray([[0.5], [-0.5], [0.5]], dtype=jnp.float32)
    scan = run_permanent_transient_arrays(
        learner,
        observations,
        targets,
        state=state,
    )
    chex.assert_shape(scan.metrics, (3, 8))
    chex.assert_shape(scan.update_applied, (3,))
    chex.assert_shape(scan.pre_step_words, (3, 2))
    assert bool(jnp.all(scan.update_applied))
    assert int(scan.state.step_count) == 3


def test_exact_clock_carry_telemetry_saturation_and_terminal_fail_stop() -> None:
    learner = _learner()
    carry = _update(learner, _state_at(learner, _UINT32_MAX))
    chex.assert_trees_all_equal(carry.pre_step_words, _words(_UINT32_MAX))
    chex.assert_trees_all_equal(carry.post_step_words, _words(1 << 32))
    chex.assert_trees_all_equal(carry.state.step_count, _telemetry(1 << 32))
    assert bool(carry.update_applied)

    boundary = _update(learner, _state_at(learner, _INT32_MAX - 1))
    beyond = _update(learner, boundary.state)
    assert int(boundary.state.step_count) == _INT32_MAX
    assert int(beyond.state.step_count) == _INT32_MAX
    chex.assert_trees_all_equal(beyond.state.step_words, _words(_INT32_MAX + 1))

    terminal = _state_at(learner, _UINT64_MAX)
    exhausted = _update(learner, terminal)
    assert bool(exhausted.state_valid)
    assert not bool(exhausted.lifetime_capacity_available)
    _assert_rejected(exhausted, terminal)
    _assert_rejected(_update(learner, exhausted.state), terminal)


@pytest.mark.parametrize("bad", [jnp.nan, jnp.inf, -jnp.inf])
def test_nonfinite_inputs_and_corrupt_state_reject_atomically(bad: float) -> None:
    learner = _learner(transient_decay=0.5)
    source = learner.init(jr.key(5))
    bad_observation = _update(
        learner,
        source,
        observation=jnp.asarray([bad, 0.0], dtype=jnp.float32),
    )
    assert not bool(bad_observation.observation_valid)
    assert not bool(bad_observation.input_valid)
    _assert_rejected(bad_observation, source)

    bad_target = _update(
        learner,
        source,
        target=jnp.asarray([bad], dtype=jnp.float32),
    )
    assert not bool(bad_target.target_valid)
    _assert_rejected(bad_target, source)

    corrupt_counter = source.replace(
        step_count=jnp.asarray(3, dtype=jnp.int32),
        step_words=_words(4),
    )
    counter_result = _update(learner, corrupt_counter)
    assert not bool(counter_result.lifetime_counter_valid)
    assert not bool(counter_result.state_valid)
    _assert_rejected(counter_result, corrupt_counter)

    corrupt_params = source.replace(
        params=source.params.replace(
            transient_head_kernel=source.params.transient_head_kernel.at[0, 0].set(bad)
        )
    )
    state_result = _update(learner, corrupt_params)
    assert not bool(state_result.state_valid)
    _assert_rejected(state_result, corrupt_params)


def test_nonfinite_candidate_and_retry_are_fail_closed() -> None:
    learner = _learner()
    source = learner.init(jr.key(13))
    maximum = np.finfo(np.float32).max
    candidate = _update(
        learner,
        source,
        observation=jnp.full((2,), maximum, dtype=jnp.float32),
        target=jnp.asarray([maximum], dtype=jnp.float32),
    )
    assert bool(candidate.input_valid)
    assert not bool(candidate.candidate_state_valid)
    _assert_rejected(candidate, source)

    rejected = _update(
        learner,
        source,
        observation=jnp.asarray([jnp.nan, 0.0], dtype=jnp.float32),
    )
    retried = _update(learner, rejected.state)
    direct = _update(learner, source)
    chex.assert_trees_all_equal(retried, direct)


def test_strict_static_contracts_and_versioned_config() -> None:
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

    payload = learner.config.to_config()
    assert payload["schema"] == PERMANENT_TRANSIENT_CONFIG_SCHEMA
    assert payload["state_schema"] == PERMANENT_TRANSIENT_STATE_SCHEMA
    assert payload["result_schema"] == PERMANENT_TRANSIENT_RESULT_SCHEMA
    assert payload["resource_schema"] == PERMANENT_TRANSIENT_RESOURCE_SCHEMA
    assert AlbertaPermanentTransientConfig.from_config(payload) == learner.config
    with pytest.raises(ValueError, match="fields"):
        AlbertaPermanentTransientConfig.from_config({**payload, "extra": 1})
    with pytest.raises(ValueError, match="schema"):
        AlbertaPermanentTransientConfig.from_config({**payload, "state_schema": "wrong"})

    with pytest.raises(ValueError, match="grad_clip"):
        _learner(grad_clip=-1.0)
    assert _learner(grad_clip=0.0).config.grad_clip == 0.0


def test_state_and_resource_records_measure_exact_fixed_capacity() -> None:
    learner = _learner()
    state = learner.init(jr.key(0))
    measured = measure_permanent_transient_state_nbytes(state)
    record = learner.state_record(state)
    resources = learner.resource_record(state)

    assert record.schema == PERMANENT_TRANSIENT_STATE_SCHEMA
    assert record.state_nbytes == measured
    assert record.parameter_nbytes + PERMANENT_TRANSIENT_LIFETIME_COUNTER_NBYTES == measured
    assert resources.schema == PERMANENT_TRANSIENT_RESOURCE_SCHEMA
    assert resources.state_nbytes == measured
    assert resources.parameter_nbytes == record.parameter_nbytes
    assert resources.exact_lifetime_identity_nbytes == (
        PERMANENT_TRANSIENT_EXACT_LIFETIME_NBYTES
    )
    assert resources.lifetime_counter_nbytes == PERMANENT_TRANSIENT_LIFETIME_COUNTER_NBYTES
    assert resources.replay_capacity == 0
    assert resources.maximum_stored_examples == 0
    assert resources.persistent_capacity_growth == 0
