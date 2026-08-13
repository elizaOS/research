"""Exact lifetime and eviction-clock contracts for latent context inference."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, cast

import chex
import jax
import jax.numpy as jnp
import pytest

import alberta_framework.core as public_core
from alberta_framework.core.checkpoints import save_checkpoint
from alberta_framework.core.context_inference import (
    CONTEXT_INFERENCE_CHECKPOINT_SCHEMA,
    CONTEXT_INFERENCE_STATE_SCHEMA,
    ContextInference,
    ContextInferenceConfig,
    ContextInferenceState,
    ContextInferenceUpdateResult,
    context_inference_clock_nbytes,
    context_inference_exact_clock_delta_nbytes,
    load_context_inference_checkpoint,
    measure_context_inference_state_nbytes,
    migrate_legacy_context_inference_state,
    save_context_inference_checkpoint,
)

pytestmark = pytest.mark.unit

_INT32_MAX = 2**31 - 1
_UINT32_MAX = 2**32 - 1


def _module(**overrides: object) -> ContextInference:
    values: dict[str, object] = {
        "n_actions": 2,
        "observation_dim": 2,
        "max_contexts": 3,
    }
    values.update(overrides)
    return ContextInference(ContextInferenceConfig(**values))


def _replace(state: ContextInferenceState, **changes: object) -> ContextInferenceState:
    return cast(ContextInferenceState, state.replace(**changes))  # type: ignore[attr-defined]


def _telemetry(words: tuple[int, int]) -> int:
    high, low = words
    return low if high == 0 and low <= _INT32_MAX else _INT32_MAX


def _exact_state(
    module: ContextInference,
    *,
    step_words: tuple[int, int],
    dwell_words: tuple[int, int],
    last_active_words: tuple[tuple[int, int], ...],
    in_use: tuple[bool, ...],
    active_context: int = 0,
    reward_weights: jax.Array | None = None,
) -> ContextInferenceState:
    """Construct a structurally aligned high-word state for focused tests."""
    initial = module.init()
    last_telemetry = tuple(
        _telemetry(words) if used else -1
        for words, used in zip(last_active_words, in_use, strict=True)
    )
    return _replace(
        initial,
        reward_weights=(initial.reward_weights if reward_weights is None else reward_weights),
        in_use=jnp.asarray(in_use, dtype=jnp.bool_),
        active_context=jnp.asarray(active_context, dtype=jnp.int32),
        last_active_step=jnp.asarray(last_telemetry, dtype=jnp.int32),
        dwell=jnp.asarray(_telemetry(dwell_words), dtype=jnp.int32),
        step_count=jnp.asarray(_telemetry(step_words), dtype=jnp.int32),
        last_active_words=jnp.asarray(last_active_words, dtype=jnp.uint32),
        dwell_words=jnp.asarray(dwell_words, dtype=jnp.uint32),
        step_words=jnp.asarray(step_words, dtype=jnp.uint32),
    )


def _neutral_update(
    module: ContextInference,
    state: ContextInferenceState,
) -> ContextInferenceUpdateResult:
    return cast(
        ContextInferenceUpdateResult,
        module.update_result(
            state,
            jnp.asarray((1.0, 0.0), dtype=jnp.float32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0.5, dtype=jnp.float32),
        ),
    )


def test_init_declares_v2_schema_and_exact_clock_identity() -> None:
    module = _module()
    state = module.init()

    assert CONTEXT_INFERENCE_STATE_SCHEMA.endswith(".v2")
    assert CONTEXT_INFERENCE_CHECKPOINT_SCHEMA.endswith(".v2")
    chex.assert_trees_all_equal(state.step_words, jnp.zeros((2,), dtype=jnp.uint32))
    chex.assert_trees_all_equal(state.dwell_words, jnp.zeros((2,), dtype=jnp.uint32))
    chex.assert_trees_all_equal(
        state.last_active_words,
        jnp.zeros((module.config.max_contexts, 2), dtype=jnp.uint32),
    )
    assert bool(module.state_is_valid(state))
    with pytest.raises(ValueError, match="terminal"):
        _module(min_dwell=2**64 - 1)


def test_stable_exact_clock_surface_is_exported_from_core() -> None:
    expected = {
        "CONTEXT_INFERENCE_CHECKPOINT_SCHEMA",
        "CONTEXT_INFERENCE_STATE_SCHEMA",
        "ContextInferenceResourceBudget",
        "ContextInferenceUpdateResult",
        "context_inference_clock_nbytes",
        "context_inference_exact_clock_delta_nbytes",
        "load_context_inference_checkpoint",
        "measure_context_inference_state_nbytes",
        "migrate_legacy_context_inference_state",
        "save_context_inference_checkpoint",
    }
    assert expected <= set(public_core.__all__)
    assert all(hasattr(public_core, name) for name in expected)


def test_step_dwell_and_recency_carry_without_signed_wrap() -> None:
    module = _module()
    state = _exact_state(
        module,
        step_words=(0, _UINT32_MAX),
        dwell_words=(0, _UINT32_MAX),
        last_active_words=((0, _UINT32_MAX - 1), (0, 0), (0, 0)),
        in_use=(True, False, False),
    )

    result = _neutral_update(module, state)

    assert bool(result.source_state_valid)
    assert bool(result.candidate_state_valid)
    assert bool(result.lifetime_capacity_available)
    assert bool(result.update_applied)
    chex.assert_trees_all_equal(result.state.step_words, jnp.asarray((1, 0), dtype=jnp.uint32))
    chex.assert_trees_all_equal(result.state.dwell_words, jnp.asarray((1, 0), dtype=jnp.uint32))
    chex.assert_trees_all_equal(
        result.state.last_active_words[0],
        jnp.asarray((0, _UINT32_MAX), dtype=jnp.uint32),
    )
    assert int(result.state.step_count) == _INT32_MAX
    assert int(result.state.dwell) == _INT32_MAX
    assert int(result.state.last_active_step[0]) == _INT32_MAX


def test_high_word_dwell_controls_change_point_gate() -> None:
    module = _module(
        max_contexts=2,
        min_dwell=2**32 + 1,
        error_decay=0.0,
    )
    weights = module.init().reward_weights.at[0, 0, 0].set(0.0).at[1, 0, 0].set(1.0)
    state = _exact_state(
        module,
        step_words=(1, 1),
        dwell_words=(1, 0),
        last_active_words=((1, 0), (0, 7)),
        in_use=(True, True),
        reward_weights=weights,
    )

    before_threshold = module.update_result(
        state,
        jnp.asarray((1.0, 0.0), dtype=jnp.float32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(1.0, dtype=jnp.float32),
    )
    assert int(before_threshold.state.active_context) == 0
    chex.assert_trees_all_equal(
        before_threshold.state.dwell_words,
        jnp.asarray((1, 1), dtype=jnp.uint32),
    )

    at_threshold = module.update_result(
        before_threshold.state,
        jnp.asarray((1.0, 0.0), dtype=jnp.float32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(1.0, dtype=jnp.float32),
    )
    assert bool(at_threshold.update_applied)
    assert int(at_threshold.state.active_context) == 1
    chex.assert_trees_all_equal(
        at_threshold.state.dwell_words,
        jnp.zeros((2,), dtype=jnp.uint32),
    )


@pytest.mark.parametrize(
    ("slot_one_words", "slot_two_words", "expected"),
    [
        ((0, _UINT32_MAX), (1, 0), 1),
        ((1, 0), (0, _UINT32_MAX), 2),
        ((1, 0), (1, 0), 1),
    ],
)
def test_lru_uses_exact_lexicographic_recency_then_slot_id(
    slot_one_words: tuple[int, int],
    slot_two_words: tuple[int, int],
    expected: int,
) -> None:
    module = _module(min_dwell=0, error_decay=0.0)
    weights = jnp.zeros_like(module.init().reward_weights)
    state = _exact_state(
        module,
        step_words=(2, 9),
        dwell_words=(2, 9),
        last_active_words=((2, 8), slot_one_words, slot_two_words),
        in_use=(True, True, True),
        reward_weights=weights,
    )

    result = module.update_result(
        state,
        jnp.asarray((1.0, 0.0), dtype=jnp.float32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(1.0, dtype=jnp.float32),
    )

    assert bool(result.update_applied)
    assert int(result.state.active_context) == expected
    chex.assert_trees_all_equal(
        result.state.last_active_words[expected],
        jnp.asarray((2, 9), dtype=jnp.uint32),
    )


def test_corrupt_source_and_nonfinite_candidate_both_roll_back_atomically() -> None:
    module = _module()
    initial = module.init()
    corrupt = _replace(initial, step_count=jnp.asarray(1, dtype=jnp.int32))
    rejected = _neutral_update(module, corrupt)
    assert not bool(rejected.source_state_valid)
    assert not bool(rejected.update_applied)
    chex.assert_trees_all_equal(rejected.state, corrupt)

    invalid_active = _replace(
        initial,
        active_context=jnp.asarray(-1, dtype=jnp.int32),
    )
    invalid_active_result = _neutral_update(module, invalid_active)
    assert not bool(invalid_active_result.source_state_valid)
    chex.assert_trees_all_equal(invalid_active_result.state, invalid_active)
    chex.assert_trees_all_equal(
        invalid_active_result.context_onehot,
        jnp.zeros((module.config.max_contexts,), dtype=jnp.float32),
    )

    explosive = _module(model_step_size=3.0e38, update_error_gate=10.0)
    candidate_rejected = explosive.update_result(
        explosive.init(),
        jnp.asarray((1.0, 0.0), dtype=jnp.float32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(2.0, dtype=jnp.float32),
    )
    assert bool(candidate_rejected.source_state_valid)
    assert bool(candidate_rejected.input_valid)
    assert not bool(candidate_rejected.candidate_state_valid)
    assert not bool(candidate_rejected.update_applied)
    chex.assert_trees_all_equal(candidate_rejected.state, explosive.init())


def test_future_recency_and_exact_lifetime_exhaustion_fail_closed() -> None:
    module = _module()
    future = _replace(
        module.init(),
        last_active_words=module.init().last_active_words.at[0].set(
            jnp.asarray((0, 1), dtype=jnp.uint32)
        ),
        last_active_step=module.init().last_active_step.at[0].set(1),
    )
    future_result = _neutral_update(module, future)
    assert not bool(future_result.source_state_valid)
    chex.assert_trees_all_equal(future_result.state, future)

    after_one, _ = module.update(
        module.init(),
        jnp.asarray((1.0, 0.0), dtype=jnp.float32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.5, dtype=jnp.float32),
    )
    impossible_nonactive = _replace(
        after_one,
        in_use=jnp.asarray((True, True, False), dtype=jnp.bool_),
        error_ema=after_one.error_ema.at[1].set(0.0),
        last_active_step=after_one.last_active_step.at[1].set(1),
        last_active_words=after_one.last_active_words.at[1].set(
            jnp.asarray((0, 1), dtype=jnp.uint32)
        ),
    )
    impossible_result = _neutral_update(module, impossible_nonactive)
    assert not bool(impossible_result.source_state_valid)
    chex.assert_trees_all_equal(impossible_result.state, impossible_nonactive)

    exhausted = _exact_state(
        module,
        step_words=(_UINT32_MAX, _UINT32_MAX),
        dwell_words=(_UINT32_MAX, _UINT32_MAX),
        last_active_words=((_UINT32_MAX, _UINT32_MAX - 1), (0, 0), (0, 0)),
        in_use=(True, False, False),
    )
    stopped = _neutral_update(module, exhausted)
    assert bool(stopped.source_state_valid)
    assert not bool(stopped.lifetime_capacity_available)
    assert not bool(stopped.update_applied)
    chex.assert_trees_all_equal(stopped.state, exhausted)


def test_eager_jit_and_scan_have_identical_high_word_trajectories() -> None:
    module = _module()
    initial = _exact_state(
        module,
        step_words=(0, _UINT32_MAX - 1),
        dwell_words=(0, _UINT32_MAX - 1),
        last_active_words=((0, _UINT32_MAX - 2), (0, 0), (0, 0)),
        in_use=(True, False, False),
    )
    observations = jnp.tile(jnp.asarray(((1.0, 0.0),), dtype=jnp.float32), (3, 1))
    actions = jnp.zeros((3,), dtype=jnp.int32)
    rewards = jnp.full((3,), 0.5, dtype=jnp.float32)

    eager = initial
    eager_words = []
    for observation, action, reward in zip(observations, actions, rewards, strict=True):
        eager, _ = module.update(eager, observation, action, reward)
        eager_words.append(eager.step_words)

    def scan_step(
        state: ContextInferenceState,
        inputs: tuple[jax.Array, jax.Array, jax.Array],
    ) -> tuple[ContextInferenceState, tuple[jax.Array, jax.Array]]:
        next_state, onehot = module.update(state, *inputs)
        return next_state, (next_state.step_words, onehot)

    scanned, (scan_words, onehots) = jax.jit(
        lambda state: jax.lax.scan(
            scan_step,
            state,
            (observations, actions, rewards),
        )
    )(initial)
    chex.assert_trees_all_equal(scanned, eager)
    chex.assert_trees_all_equal(scan_words, jnp.stack(eager_words))
    chex.assert_trees_all_equal(
        scan_words,
        jnp.asarray(((0, _UINT32_MAX), (1, 0), (1, 1)), dtype=jnp.uint32),
    )
    chex.assert_trees_all_equal(onehots, jnp.asarray(((1.0, 0.0, 0.0),) * 3))


def test_resource_accounting_includes_all_exact_authorities() -> None:
    module = _module(max_contexts=5)
    state = module.init()
    budget = module.resource_budget

    assert context_inference_exact_clock_delta_nbytes(5) == 8 * (5 + 2)
    assert context_inference_clock_nbytes(5) == 12 * (5 + 2)
    assert budget.exact_clock_delta_nbytes == context_inference_exact_clock_delta_nbytes(5)
    assert budget.clock_nbytes == context_inference_clock_nbytes(5)
    assert budget.state_nbytes == measure_context_inference_state_nbytes(state)
    assert budget.allocated_uint32_scalars == 2 * 5 + 4


def test_legacy_migration_requires_unambiguous_exact_counter_history() -> None:
    module = _module()
    current = module.init()
    for _ in range(3):
        current, _ = module.update(
            current,
            jnp.asarray((1.0, 0.0), dtype=jnp.float32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0.5, dtype=jnp.float32),
        )
    new_fields = {"step_words", "dwell_words", "last_active_words"}
    legacy = {
        field.name: getattr(current, field.name)
        for field in dataclasses.fields(cast(Any, type(current)))
        if field.name not in new_fields
    }
    migrated = migrate_legacy_context_inference_state(legacy, config=module.config)
    chex.assert_trees_all_equal(migrated, current)

    saturated = dict(legacy)
    saturated["step_count"] = jnp.asarray(_INT32_MAX, dtype=jnp.int32)
    with pytest.raises(ValueError, match="ambiguous"):
        migrate_legacy_context_inference_state(saturated, config=module.config)

    with pytest.raises(ValueError, match="manifest"):
        migrate_legacy_context_inference_state(
            {**legacy, "invented": jnp.asarray(0)},
            config=module.config,
        )


def test_checkpoint_round_trip_authenticates_schema_state_and_resources(
    tmp_path: Path,
) -> None:
    module = _module()
    with pytest.raises(ValueError, match="state is invalid"):
        save_context_inference_checkpoint(
            module,
            _replace(module.init(), step_count=jnp.asarray(1, dtype=jnp.int32)),
            tmp_path / "invalid-context",
        )
    state, _ = module.update(
        module.init(),
        jnp.asarray((1.0, 0.0), dtype=jnp.float32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.5, dtype=jnp.float32),
    )
    path = tmp_path / "context"
    save_context_inference_checkpoint(module, state, path)
    restored_module, restored = load_context_inference_checkpoint(path)
    assert restored_module.config == module.config
    chex.assert_trees_all_equal(restored, state)

    legacy_path = tmp_path / "legacy-context"
    save_checkpoint(
        state,
        legacy_path,
        metadata={
            "schema": "alberta.context-inference-checkpoint.v1",
            "module_config": module.to_config(),
            "memory_accounting": dataclasses.asdict(module.resource_budget),
        },
    )
    with pytest.raises(ValueError, match="migrate_legacy_context_inference_state"):
        load_context_inference_checkpoint(legacy_path)
