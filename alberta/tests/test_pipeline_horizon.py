# mypy: disable-error-code="attr-defined,call-arg,no-untyped-def"
"""Exact finite-horizon and atomicity contracts for the Step 1-4 pipeline."""

from __future__ import annotations

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import alberta_framework as alberta
from alberta_framework.core.horde import HordeUpdateResult
from alberta_framework.pipeline import (
    ALBERTA_PIPELINE_CHECKPOINT_SCHEMA,
    ALBERTA_PIPELINE_CONFIG_SCHEMA,
    ALBERTA_PIPELINE_LIFETIME_COUNTER_DELTA_NBYTES,
    ALBERTA_PIPELINE_LIFETIME_COUNTER_NBYTES,
    ALBERTA_PIPELINE_STATE_SCHEMA,
    PIPELINE_REJECTION_LIFETIME_EXHAUSTED,
    PIPELINE_REJECTION_SOURCE_INVALID,
    PIPELINE_REJECTION_STATE_INVALID,
    PIPELINE_REJECTION_STEP2_UNAVAILABLE,
    PIPELINE_REJECTION_STEP3_REFUSED,
    AlbertaPipeline,
    AlbertaPipelineConfig,
    HordeActorCriticPipelineConfig,
    Step2FeatureConfig,
    Step2UPGDConfig,
    load_alberta_pipeline_checkpoint,
    measure_alberta_pipeline_state_nbytes,
    migrate_legacy_alberta_pipeline_config,
    migrate_legacy_alberta_pipeline_state,
    save_alberta_pipeline_checkpoint,
)
from alberta_framework.steps.step3 import Step3HordeConfig
from alberta_framework.steps.step4 import Step4SARSAConfig

_INT32_MAX = 2**31 - 1
_UINT32_MAX = 2**32 - 1


def _config(
    *,
    step2: str = "identity",
    control_mode: str = "sarsa",
) -> AlbertaPipelineConfig:
    return AlbertaPipelineConfig(
        features=Step2FeatureConfig.identity(observation_dim=2),
        horde=Step3HordeConfig(
            gammas=(0.0,),
            lamdas=(0.0,),
            hidden_sizes=(),
            step_size=0.05,
        ),
        control=Step4SARSAConfig(
            n_actions=2,
            hidden_sizes=(),
            epsilon_start=0.0,
            epsilon_end=0.0,
            step_size=0.05,
        ),
        horde_ac=(
            HordeActorCriticPipelineConfig(
                n_actions=2,
                actor_step_size=0.02,
                actor_lamda=0.0,
                value_head_index=0,
            )
            if control_mode == "horde_ac"
            else None
        ),
        step2=step2,  # type: ignore[arg-type]
        control_mode=control_mode,  # type: ignore[arg-type]
    )


def _transition(pipeline: AlbertaPipeline, state: Any):
    return pipeline.update(
        state,
        jnp.asarray((0.25, -0.5), dtype=jnp.float32),
        jnp.asarray(0.2, dtype=jnp.float32),
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray((0.4,), dtype=jnp.float32),
    )


def _assert_trees_equal(left: Any, right: Any) -> None:
    left_leaves = jax.tree.leaves(left)
    right_leaves = jax.tree.leaves(right)
    assert len(left_leaves) == len(right_leaves)
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        left_dtype = getattr(left_leaf, "dtype", None)
        right_dtype = getattr(right_leaf, "dtype", None)
        # Host-only birth/uptime floats are explicitly outside the persistent
        # JAX transaction contract and can be traced to float32 by lax.cond.
        if left_dtype is None or right_dtype is None:
            continue
        if jax.dtypes.issubdtype(left_dtype, jax.dtypes.prng_key):
            left_leaf = jr.key_data(left_leaf)
            right_leaf = jr.key_data(right_leaf)
        np.testing.assert_array_equal(np.asarray(left_leaf), np.asarray(right_leaf))


def _seed_identity_words(pipeline: AlbertaPipeline, state: Any, words: tuple[int, int]):
    step_words = jnp.asarray(words, dtype=jnp.uint32)
    saturated = words[0] != 0 or words[1] >= _INT32_MAX
    count = jnp.asarray(
        _INT32_MAX if saturated else words[1],
        dtype=jnp.int32,
    )
    horde_state = state.horde_state.replace(
        step_words=step_words,
        step_count=count,
    )
    if pipeline.config.control_mode == "horde_ac":
        control_state = state.control_state.replace(
            critic_state=horde_state,
            step_count=count,
        )
    else:
        control_state = state.control_state.replace(
            learner_state=state.control_state.learner_state.replace(
                step_words=step_words,
                step_count=count,
            ),
            step_count=count,
        )
    return state.replace(
        horde_state=horde_state,
        control_state=control_state,
        step_count=count,
        step_words=step_words,
    )


@pytest.mark.parametrize("control_mode", ["sarsa", "horde_ac"])
def test_exact_low_word_carry_and_saturated_control_telemetry(control_mode: str) -> None:
    pipeline = AlbertaPipeline(_config(control_mode=control_mode))
    initial = pipeline.init(
        jr.key(0),
        jnp.asarray((0.1, -0.2), dtype=jnp.float32),
    )
    state = _seed_identity_words(pipeline, initial, (0, _UINT32_MAX))
    assert bool(pipeline.state_valid(state))

    result = _transition(pipeline, state)

    assert bool(result.update_applied)
    np.testing.assert_array_equal(
        result.state.step_words,
        jnp.asarray((1, 0), dtype=jnp.uint32),
    )
    assert int(result.state.step_count) == _INT32_MAX
    assert int(result.state.control_state.step_count) == _INT32_MAX
    np.testing.assert_array_equal(
        result.state.horde_state.step_words,
        result.state.step_words,
    )
    control_horde = (
        result.state.control_state.critic_state
        if control_mode == "horde_ac"
        else result.state.control_state.learner_state
    )
    np.testing.assert_array_equal(control_horde.step_words, result.state.step_words)


def test_all_ones_terminal_refuses_bit_exactly() -> None:
    pipeline = AlbertaPipeline(_config())
    initial = pipeline.init(
        jr.key(1),
        jnp.asarray((0.1, -0.2), dtype=jnp.float32),
    )
    terminal = _seed_identity_words(
        pipeline,
        initial,
        (_UINT32_MAX, _UINT32_MAX),
    )

    result = _transition(pipeline, terminal)

    assert not bool(result.update_applied)
    assert int(result.rejection_reason) == PIPELINE_REJECTION_LIFETIME_EXHAUSTED
    _assert_trees_equal(result.state, terminal)


def test_corrupt_clock_and_nonfinite_source_roll_back_every_child() -> None:
    pipeline = AlbertaPipeline(_config())
    state = pipeline.init(
        jr.key(2),
        jnp.asarray((0.1, -0.2), dtype=jnp.float32),
    )
    corrupt = state.replace(
        step_words=jnp.asarray((0, 1), dtype=jnp.uint32),
        step_count=jnp.asarray(1, dtype=jnp.int32),
    )

    corrupt_result = _transition(pipeline, corrupt)
    assert not bool(corrupt_result.update_applied)
    assert int(corrupt_result.rejection_reason) == PIPELINE_REJECTION_STATE_INVALID
    _assert_trees_equal(corrupt_result.state, corrupt)

    nonfinite_result = pipeline.update(
        state,
        jnp.asarray((jnp.inf, -0.5), dtype=jnp.float32),
        jnp.asarray(0.2, dtype=jnp.float32),
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray((0.4,), dtype=jnp.float32),
    )
    assert not bool(nonfinite_result.update_applied)
    assert int(nonfinite_result.rejection_reason) == PIPELINE_REJECTION_SOURCE_INVALID
    _assert_trees_equal(nonfinite_result.state, state)
    assert bool(jnp.all(jnp.isfinite(nonfinite_result.features)))

    with pytest.raises(TypeError, match="observation must have dtype float32"):
        pipeline.update(
            state,
            jnp.asarray((1, 2), dtype=jnp.int32),
            jnp.asarray(0.2, dtype=jnp.float32),
            jnp.asarray(0.0, dtype=jnp.float32),
            jnp.asarray((0.4,), dtype=jnp.float32),
        )
    with pytest.raises(ValueError, match="horde_cumulants must have shape"):
        pipeline.update(
            state,
            jnp.asarray((0.25, -0.5), dtype=jnp.float32),
            jnp.asarray(0.2, dtype=jnp.float32),
            jnp.asarray(0.0, dtype=jnp.float32),
            jnp.asarray((0.4, 0.5), dtype=jnp.float32),
        )


class _RefusingHorde:
    """Delegate predictions but make the staged Step 3 transaction refuse."""

    def __init__(self, delegate: Any):
        self._delegate = delegate

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)

    def update(self, state: Any, *args: Any) -> HordeUpdateResult:
        candidate = self._delegate.update(state, *args)
        return candidate.replace(
            state=state,
            post_step_words=state.step_words,
            update_applied=jnp.asarray(False, dtype=jnp.bool_),
        )


def test_step3_child_refusal_rolls_back_step2_control_rng_and_optimizers() -> None:
    pipeline = AlbertaPipeline(_config())
    state = pipeline.init(
        jr.key(3),
        jnp.asarray((0.1, -0.2), dtype=jnp.float32),
    )
    pipeline._horde = _RefusingHorde(pipeline.horde)  # noqa: SLF001

    result = _transition(pipeline, state)

    assert not bool(result.step3_update_applied)
    assert bool(result.control_update_applied)
    assert int(result.rejection_reason) == PIPELINE_REJECTION_STEP3_REFUSED
    _assert_trees_equal(result.state, state)


def test_temporal_context_exact_priming_offset_carries_with_pipeline() -> None:
    pipeline = AlbertaPipeline(_config(step2="temporal_context"))
    initial = pipeline.init(
        jr.key(4),
        jnp.asarray((0.1, -0.2), dtype=jnp.float32),
    )
    state = _seed_identity_words(pipeline, initial, (0, _UINT32_MAX - 1))
    state = state.replace(
        feature_state=state.feature_state.replace(
            step_count=jnp.asarray(_INT32_MAX, dtype=jnp.int32),
            step_words=jnp.asarray((0, _UINT32_MAX), dtype=jnp.uint32),
        )
    )
    assert bool(pipeline.state_valid(state))

    result = _transition(pipeline, state)

    assert bool(result.update_applied)
    np.testing.assert_array_equal(
        result.state.step_words,
        jnp.asarray((0, _UINT32_MAX), dtype=jnp.uint32),
    )
    np.testing.assert_array_equal(
        result.state.feature_state.step_words,
        jnp.asarray((1, 0), dtype=jnp.uint32),
    )


def test_temporal_context_priming_offset_terminal_is_fail_closed() -> None:
    pipeline = AlbertaPipeline(_config(step2="temporal_context"))
    initial = pipeline.init(
        jr.key(41),
        jnp.asarray((0.1, -0.2), dtype=jnp.float32),
    )
    state = _seed_identity_words(
        pipeline,
        initial,
        (_UINT32_MAX, _UINT32_MAX - 1),
    )
    state = state.replace(
        feature_state=state.feature_state.replace(
            step_count=jnp.asarray(_INT32_MAX, dtype=jnp.int32),
            step_words=jnp.asarray(
                (_UINT32_MAX, _UINT32_MAX),
                dtype=jnp.uint32,
            ),
        )
    )
    assert bool(pipeline.state_valid(state))

    result = _transition(pipeline, state)

    assert not bool(result.step2_contract_available)
    assert int(result.rejection_reason) == PIPELINE_REJECTION_STEP2_UNAVAILABLE
    _assert_trees_equal(result.state, state)


def test_upgd_requires_an_explicit_exact_learning_event() -> None:
    config = _config()
    config = dataclasses.replace(
        config,
        upgd=Step2UPGDConfig(
            observation_dim=2,
            n_heads=1,
            hidden_sizes=(4,),
            step_size=0.01,
        ),
        step2="upgd",
    )
    pipeline = AlbertaPipeline(config)
    state = pipeline.init(
        jr.key(40),
        jnp.asarray((0.1, -0.2), dtype=jnp.float32),
    )

    unavailable = _transition(pipeline, state)
    assert not bool(unavailable.update_applied)
    assert int(unavailable.rejection_reason) == PIPELINE_REJECTION_STEP2_UNAVAILABLE
    _assert_trees_equal(unavailable.state, state)

    applied = pipeline.update(
        state,
        jnp.asarray((0.25, -0.5), dtype=jnp.float32),
        jnp.asarray(0.2, dtype=jnp.float32),
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray((0.4,), dtype=jnp.float32),
        upgd_targets=jnp.asarray((0.1,), dtype=jnp.float32),
    )
    assert bool(applied.update_applied)
    np.testing.assert_array_equal(
        applied.state.upgd_state.step_words,
        applied.state.step_words,
    )


def test_eager_jit_and_scan_share_the_atomic_contract() -> None:
    pipeline = AlbertaPipeline(_config())
    state = pipeline.init(
        jr.key(5),
        jnp.asarray((0.1, -0.2), dtype=jnp.float32),
    )
    args = (
        jnp.asarray((0.25, -0.5), dtype=jnp.float32),
        jnp.asarray(0.2, dtype=jnp.float32),
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray((0.4,), dtype=jnp.float32),
    )
    eager = pipeline.update(state, *args)
    compiled = jax.jit(pipeline.update)(state, *args)
    _assert_trees_equal(eager, compiled)

    observations = jnp.asarray(((0.25, -0.5), (0.4, 0.1)), dtype=jnp.float32)
    rewards = jnp.asarray((0.2, -0.1), dtype=jnp.float32)
    terminated = jnp.zeros((2,), dtype=jnp.float32)
    cumulants = jnp.asarray(((0.4,), (-0.2,)), dtype=jnp.float32)
    scanned = pipeline.run_arrays(
        state,
        observations,
        rewards,
        terminated,
        cumulants,
    )
    assert bool(jnp.all(scanned.update_applied))
    np.testing.assert_array_equal(
        scanned.state.step_words,
        jnp.asarray((0, 2), dtype=jnp.uint32),
    )


def test_v2_schemas_migration_checkpoint_and_exact_resource_delta(tmp_path: Any) -> None:
    config = _config()
    payload = config.to_dict()
    assert payload["schema"] == ALBERTA_PIPELINE_CONFIG_SCHEMA
    assert payload["state_schema"] == ALBERTA_PIPELINE_STATE_SCHEMA
    assert AlbertaPipelineConfig.from_dict(payload) == config

    legacy_config = dict(payload)
    legacy_config.pop("type")
    legacy_config.pop("schema")
    legacy_config.pop("state_schema")
    assert migrate_legacy_alberta_pipeline_config(legacy_config) == config
    with pytest.raises(ValueError, match="explicit migration"):
        AlbertaPipelineConfig.from_dict(legacy_config)

    pipeline = AlbertaPipeline(config)
    state = _transition(
        pipeline,
        pipeline.init(
            jr.key(6),
            jnp.asarray((0.1, -0.2), dtype=jnp.float32),
        ),
    ).state
    budget = pipeline.resource_budget(state)
    assert ALBERTA_PIPELINE_LIFETIME_COUNTER_NBYTES == 12
    assert ALBERTA_PIPELINE_LIFETIME_COUNTER_DELTA_NBYTES == 8
    assert budget.exact_pipeline_identity_nbytes == 8
    assert budget.compatibility_telemetry_nbytes == 4
    assert budget.persistent_state_nbytes == measure_alberta_pipeline_state_nbytes(state)

    legacy_state = {
        item.name: getattr(state, item.name)
        for item in dataclasses.fields(state)
        if item.name != "step_words"
    }
    migrated = migrate_legacy_alberta_pipeline_state(pipeline, legacy_state)
    _assert_trees_equal(migrated, state)
    saturated = dict(legacy_state)
    saturated["step_count"] = jnp.asarray(_INT32_MAX, dtype=jnp.int32)
    with pytest.raises(ValueError, match="saturated"):
        migrate_legacy_alberta_pipeline_state(pipeline, saturated)

    path = tmp_path / "pipeline-v2"
    save_alberta_pipeline_checkpoint(pipeline, state, path)
    restored_pipeline, restored_state = load_alberta_pipeline_checkpoint(path)
    assert restored_pipeline.config == config
    _assert_trees_equal(restored_state, state)

    assert alberta.ALBERTA_PIPELINE_CHECKPOINT_SCHEMA == (
        ALBERTA_PIPELINE_CHECKPOINT_SCHEMA
    )
    assert "measure_alberta_pipeline_state_nbytes" in alberta.__all__
