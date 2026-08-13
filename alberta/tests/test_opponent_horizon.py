"""Exact-lifetime and transaction tests for adaptive-opponent streams."""

from __future__ import annotations

import dataclasses
from typing import Any

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from alberta_framework.core.learners import LinearLearner
from alberta_framework.core.optimizers import LMS
from alberta_framework.streams.opponent import (
    ADVERSARIAL_PURSUIT_CONFIG_SCHEMA,
    ADVERSARIAL_PURSUIT_EMIT_INPUT_SCHEMA,
    ADVERSARIAL_PURSUIT_EMIT_RESULT_SCHEMA,
    ADVERSARIAL_PURSUIT_RESOLVE_INPUT_SCHEMA,
    ADVERSARIAL_PURSUIT_RESOLVE_RESULT_SCHEMA,
    ADVERSARIAL_PURSUIT_STATE_SCHEMA,
    LEARNING_OPPONENT_CONFIG_SCHEMA,
    LEARNING_OPPONENT_INPUT_SCHEMA,
    LEARNING_OPPONENT_RESULT_SCHEMA,
    LEARNING_OPPONENT_STATE_SCHEMA,
    OPPONENT_STREAM_CLOCK_DELTA_NBYTES,
    OPPONENT_STREAM_CLOCK_NBYTES,
    OPPONENT_STREAM_RESOURCE_SCHEMA,
    AdversarialPursuitState,
    AdversarialPursuitStream,
    LearningOpponentState,
    LearningOpponentStream,
    OpponentStreamResourceBudget,
    measure_opponent_stream_state_nbytes,
    migrate_legacy_adversarial_pursuit_state,
    migrate_legacy_learning_opponent_state,
    run_pursuit_loop,
)

_INT32_MAX = 2**31 - 1
_UINT32_MAX = 2**32 - 1


def _words(value: int) -> jax.Array:
    return jnp.asarray(((value >> 32) & _UINT32_MAX, value & _UINT32_MAX), dtype=jnp.uint32)


def _telemetry(value: int) -> jax.Array:
    return jnp.asarray(min(value, _INT32_MAX), dtype=jnp.int32)


def _learning_at(
    stream: LearningOpponentStream,
    value: int,
    *,
    w_opp: jax.Array | None = None,
) -> LearningOpponentState:
    state = stream.init(jr.key(7))
    return state.replace(
        step_count=_telemetry(value),
        step_words=_words(value),
        w_opp=state.w_opp if w_opp is None else w_opp,
    )


def _pursuit_at(
    stream: AdversarialPursuitStream,
    value: int,
    *,
    armed: bool = False,
) -> AdversarialPursuitState:
    state = stream.init(jr.key(9))
    words = _words(value)
    return state.replace(
        step_count=_telemetry(value),
        step_words=words,
        pending_owner_words=words,
        pending_armed=jnp.asarray(armed, dtype=jnp.bool_),
    )


def _legacy_fields(state: Any, omitted: set[str]) -> dict[str, Any]:
    return {
        field.name: getattr(state, field.name)
        for field in dataclasses.fields(state)
        if field.name not in omitted
    }


@pytest.mark.unit
def test_learning_config_is_complete_strict_and_finite() -> None:
    stream = LearningOpponentStream(
        feature_dim=5,
        opponent_step_size=0.03,
        reset_interval=17,
        opponent_noise_std=0.2,
        target_noise_std=0.04,
        feature_std=1.25,
    )
    payload = stream.to_config()
    assert payload["config_schema"] == LEARNING_OPPONENT_CONFIG_SCHEMA
    assert payload["state_schema"] == LEARNING_OPPONENT_STATE_SCHEMA
    assert payload["input_schema"] == LEARNING_OPPONENT_INPUT_SCHEMA
    assert payload["result_schema"] == LEARNING_OPPONENT_RESULT_SCHEMA
    assert LearningOpponentStream.from_config(payload).to_config() == payload

    with pytest.raises(ValueError, match="fields"):
        LearningOpponentStream.from_config({**payload, "extra": 1})
    with pytest.raises(ValueError, match="schema"):
        LearningOpponentStream.from_config({**payload, "config_schema": "legacy"})
    with pytest.raises(ValueError, match="finite"):
        LearningOpponentStream(feature_dim=2, target_noise_std=float("nan"))
    with pytest.raises(ValueError, match="non-negative"):
        LearningOpponentStream(feature_dim=2, opponent_noise_std=-0.1)
    with pytest.raises(ValueError, match="integer"):
        LearningOpponentStream(feature_dim=True)  # type: ignore[arg-type]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (_UINT32_MAX, 1 << 32),
        ((1 << 32) + 11, (1 << 32) + 12),
        ((1 << 63) + 5, (1 << 63) + 6),
    ],
)
def test_learning_exact_clock_crosses_uint32_and_saturates_telemetry(
    value: int,
    expected: int,
) -> None:
    stream = LearningOpponentStream(feature_dim=3, reset_interval=0)
    state = _learning_at(stream, value)
    eager = stream.step_result(state, jnp.asarray(0, dtype=jnp.int32))
    compiled = jax.jit(stream.step_result)(state, jnp.asarray(0, dtype=jnp.int32))

    assert bool(eager.update_applied)
    chex.assert_trees_all_equal(jr.key_data(eager.state.key), jr.key_data(compiled.state.key))
    chex.assert_trees_all_equal(eager.state.w_star, compiled.state.w_star)
    chex.assert_trees_all_close(eager.state.w_opp, compiled.state.w_opp, rtol=1e-6, atol=1e-7)
    chex.assert_trees_all_equal(eager.pre_step_words, _words(value))
    chex.assert_trees_all_equal(eager.post_step_words, _words(expected))
    chex.assert_trees_all_equal(eager.state.step_count, _telemetry(expected))


@pytest.mark.unit
def test_learning_reset_schedule_uses_full_uint64_identity() -> None:
    stream = LearningOpponentStream(feature_dim=2, reset_interval=3)
    nonzero = jnp.asarray((4.0, -2.0), dtype=jnp.float32)

    not_reset = stream.step_result(
        _learning_at(stream, 1 << 32, w_opp=nonzero),
        jnp.asarray(0, dtype=jnp.int32),
    )
    at_reset = stream.step_result(
        _learning_at(stream, (1 << 32) + 2, w_opp=nonzero),
        jnp.asarray(0, dtype=jnp.int32),
    )
    assert not bool(not_reset.opponent_reset)
    assert bool(at_reset.opponent_reset)


@pytest.mark.unit
def test_learning_rejections_are_finite_and_bit_exact_under_jit() -> None:
    stream = LearningOpponentStream(feature_dim=4, reset_interval=5)
    terminal = _learning_at(stream, (1 << 64) - 1)
    terminal_result = jax.jit(stream.step_result)(
        terminal,
        jnp.asarray(0, dtype=jnp.int32),
    )
    assert bool(terminal_result.update_rejected)
    assert not bool(terminal_result.lifetime_capacity_available)
    chex.assert_trees_all_equal(terminal_result.state, terminal)
    assert bool(jnp.all(jnp.isfinite(terminal_result.timestep.observation)))
    assert bool(jnp.all(jnp.isfinite(terminal_result.timestep.target)))

    invalid_state = stream.init(jr.key(3)).replace(
        w_opp=jnp.full((4,), jnp.nan, dtype=jnp.float32)
    )
    invalid_result = stream.step_result(invalid_state, jnp.asarray(0, dtype=jnp.int32))
    assert bool(invalid_result.update_rejected)
    chex.assert_trees_all_equal(invalid_result.state, invalid_state)
    invalid_idx = stream.step_result(
        stream.init(jr.key(4)),
        jnp.asarray(jnp.nan, dtype=jnp.float32),
    )
    assert not bool(invalid_idx.input_valid)
    chex.assert_trees_all_equal(invalid_idx.pre_step_words, invalid_idx.post_step_words)


@pytest.mark.unit
def test_learning_state_and_input_contracts_reject_wrong_shape_or_dtype() -> None:
    stream = LearningOpponentStream(feature_dim=3)
    state = stream.init(jr.key(0))
    with pytest.raises(TypeError, match="step_words"):
        stream.step_result(
            state.replace(step_words=jnp.zeros((2,), dtype=jnp.int32)),
            jnp.asarray(0, dtype=jnp.int32),
        )
    with pytest.raises(ValueError, match="step_words"):
        stream.step_result(
            state.replace(step_words=jnp.zeros((1,), dtype=jnp.uint32)),
            jnp.asarray(0, dtype=jnp.int32),
        )
    with pytest.raises(ValueError, match="idx"):
        stream.step_result(state, jnp.zeros((1,), dtype=jnp.int32))
    with pytest.raises(TypeError, match="idx"):
        stream.step_result(state, jnp.asarray(True, dtype=jnp.bool_))


@pytest.mark.unit
def test_learning_legacy_migration_is_representable_only() -> None:
    stream = LearningOpponentStream(feature_dim=3)
    state = _learning_at(stream, 23)
    legacy = _legacy_fields(state, {"step_words"})
    migrated = migrate_legacy_learning_opponent_state(legacy, stream=stream)
    chex.assert_trees_all_equal(migrated, state)

    with pytest.raises(ValueError, match="ambiguous"):
        migrate_legacy_learning_opponent_state(
            {**legacy, "step_count": jnp.asarray(_INT32_MAX, dtype=jnp.int32)},
            stream=stream,
        )
    with pytest.raises(ValueError, match="fields"):
        migrate_legacy_learning_opponent_state({**legacy, "extra": 1}, stream=stream)
    with pytest.raises(ValueError, match="invalid"):
        migrate_legacy_learning_opponent_state(
            {**legacy, "w_star": jnp.full((3,), jnp.inf, dtype=jnp.float32)},
            stream=stream,
        )


@pytest.mark.unit
def test_pursuit_config_is_complete_strict_and_finite() -> None:
    stream = AdversarialPursuitStream(
        feature_dim=4,
        drift_budget=0.03,
        noise_std=0.08,
        feature_std=1.5,
    )
    payload = stream.to_config()
    assert payload["config_schema"] == ADVERSARIAL_PURSUIT_CONFIG_SCHEMA
    assert payload["state_schema"] == ADVERSARIAL_PURSUIT_STATE_SCHEMA
    assert payload["emit_input_schema"] == ADVERSARIAL_PURSUIT_EMIT_INPUT_SCHEMA
    assert payload["emit_result_schema"] == ADVERSARIAL_PURSUIT_EMIT_RESULT_SCHEMA
    assert payload["resolve_input_schema"] == ADVERSARIAL_PURSUIT_RESOLVE_INPUT_SCHEMA
    assert payload["resolve_result_schema"] == ADVERSARIAL_PURSUIT_RESOLVE_RESULT_SCHEMA
    assert AdversarialPursuitStream.from_config(payload).to_config() == payload
    with pytest.raises(ValueError, match="fields"):
        AdversarialPursuitStream.from_config({**payload, "extra": 1})
    with pytest.raises(ValueError, match="finite"):
        AdversarialPursuitStream(feature_dim=2, drift_budget=float("inf"))


@pytest.mark.unit
def test_pursuit_owned_protocol_rejects_duplicate_emit_stale_and_duplicate_resolve() -> None:
    stream = AdversarialPursuitStream(feature_dim=3, noise_std=0.0)
    initial = stream.init(jr.key(11))
    emitted = stream.emit_result(initial)
    assert bool(emitted.update_applied)
    assert bool(emitted.state.pending_armed)
    chex.assert_trees_all_equal(emitted.owner_words, initial.step_words)

    duplicate_emit = stream.emit_result(emitted.state)
    assert bool(duplicate_emit.update_rejected)
    chex.assert_trees_all_equal(duplicate_emit.state, emitted.state)
    assert not bool(jnp.array_equal(duplicate_emit.owner_words, emitted.owner_words))

    stale_owner = emitted.owner_words.at[1].add(jnp.asarray(1, dtype=jnp.uint32))
    stale = stream.resolve_result(
        emitted.state,
        jnp.asarray([0.0], dtype=jnp.float32),
        stale_owner,
    )
    assert bool(stale.update_rejected)
    assert not bool(stale.owner_matches)
    chex.assert_trees_all_equal(stale.state, emitted.state)

    resolved = stream.resolve_result(
        emitted.state,
        jnp.asarray([0.0], dtype=jnp.float32),
        emitted.owner_words,
    )
    assert bool(resolved.update_applied)
    assert not bool(resolved.state.pending_armed)
    chex.assert_trees_all_equal(resolved.post_step_words, _words(1))

    duplicate_resolve = stream.resolve_result(
        resolved.state,
        jnp.asarray([0.0], dtype=jnp.float32),
        emitted.owner_words,
    )
    assert bool(duplicate_resolve.update_rejected)
    chex.assert_trees_all_equal(duplicate_resolve.state, resolved.state)


@pytest.mark.unit
def test_pursuit_invalid_prediction_rolls_back_armed_rng_and_can_retry() -> None:
    stream = AdversarialPursuitStream(feature_dim=4)
    emitted = stream.emit_result(stream.init(jr.key(12)))
    rejected = jax.jit(stream.resolve_result)(
        emitted.state,
        jnp.asarray([jnp.nan], dtype=jnp.float32),
        emitted.owner_words,
    )
    assert bool(rejected.update_rejected)
    assert not bool(rejected.prediction_valid)
    chex.assert_trees_all_equal(rejected.state, emitted.state)
    assert bool(jnp.all(jnp.isfinite(rejected.target)))

    prediction = jnp.asarray([0.25], dtype=jnp.float32)
    retry = stream.resolve_result(rejected.state, prediction, emitted.owner_words)
    direct = stream.resolve_result(emitted.state, prediction, emitted.owner_words)
    chex.assert_trees_all_equal(retry, direct)


@pytest.mark.unit
def test_pursuit_clock_rollover_terminal_refusal_and_tuple_compatibility() -> None:
    stream = AdversarialPursuitStream(feature_dim=2)
    state = _pursuit_at(stream, _UINT32_MAX)
    emitted = jax.jit(stream.emit_result)(state)
    resolved = jax.jit(stream.resolve_result)(
        emitted.state,
        jnp.asarray([0.0], dtype=jnp.float32),
        emitted.owner_words,
    )
    assert bool(resolved.update_applied)
    chex.assert_trees_all_equal(resolved.post_step_words, _words(1 << 32))
    chex.assert_trees_all_equal(resolved.state.step_count, _telemetry(1 << 32))

    terminal = _pursuit_at(stream, (1 << 64) - 1)
    terminal_emit = stream.emit_result(terminal)
    assert bool(terminal_emit.update_rejected)
    assert not bool(terminal_emit.lifetime_capacity_available)
    chex.assert_trees_all_equal(terminal_emit.state, terminal)
    assert bool(jnp.all(jnp.isfinite(terminal_emit.observation)))

    compat_x, compat_armed = stream.emit(state)
    compat_target, compat_resolved = stream.resolve(
        compat_armed,
        jnp.asarray([0.0], dtype=jnp.float32),
    )
    chex.assert_trees_all_equal(compat_x, emitted.observation)
    chex.assert_trees_all_equal(compat_target, resolved.target)
    chex.assert_trees_all_equal(compat_resolved, resolved.state)


@pytest.mark.unit
def test_pursuit_state_owner_and_prediction_contracts_are_strict() -> None:
    stream = AdversarialPursuitStream(feature_dim=3)
    state = stream.init(jr.key(0))
    with pytest.raises(TypeError, match="pending_armed"):
        stream.emit_result(state.replace(pending_armed=jnp.asarray(0, dtype=jnp.int32)))
    emitted = stream.emit_result(state)
    with pytest.raises(ValueError, match="prediction"):
        stream.resolve_result(
            emitted.state,
            jnp.zeros((2,), dtype=jnp.float32),
            emitted.owner_words,
        )
    with pytest.raises(TypeError, match="prediction"):
        stream.resolve_result(
            emitted.state,
            jnp.asarray([0], dtype=jnp.int32),
            emitted.owner_words,
        )
    with pytest.raises(TypeError, match="owner_words"):
        stream.resolve_result(
            emitted.state,
            jnp.asarray([0.0], dtype=jnp.float32),
            emitted.owner_words.astype(jnp.int32),
        )


@pytest.mark.unit
def test_pursuit_legacy_migration_requires_explicit_protocol_ownership() -> None:
    stream = AdversarialPursuitStream(feature_dim=3)
    state = _pursuit_at(stream, 31).replace(
        pending_x=jnp.asarray((0.1, -0.2, 0.3), dtype=jnp.float32)
    )
    legacy = _legacy_fields(
        state,
        {"step_words", "pending_owner_words", "pending_armed"},
    )
    with pytest.raises(TypeError, match="pending_armed"):
        migrate_legacy_adversarial_pursuit_state(legacy, stream=stream)

    migrated = migrate_legacy_adversarial_pursuit_state(
        legacy,
        stream=stream,
        pending_armed=False,
    )
    chex.assert_trees_all_equal(migrated, state)
    armed = migrate_legacy_adversarial_pursuit_state(
        legacy,
        stream=stream,
        pending_armed=True,
    )
    assert bool(armed.pending_armed)
    chex.assert_trees_all_equal(armed.pending_owner_words, state.step_words)

    with pytest.raises(ValueError, match="ambiguous"):
        migrate_legacy_adversarial_pursuit_state(
            {**legacy, "step_count": jnp.asarray(_INT32_MAX, dtype=jnp.int32)},
            stream=stream,
            pending_armed=False,
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    "stream",
    [LearningOpponentStream(feature_dim=5), AdversarialPursuitStream(feature_dim=5)],
)
def test_opponent_resource_accounting_is_exact_and_strict(stream: Any) -> None:
    state = stream.init(jr.key(0))
    budget = stream.resource_budget
    assert budget.schema == OPPONENT_STREAM_RESOURCE_SCHEMA
    assert budget.state_nbytes == measure_opponent_stream_state_nbytes(state)
    assert budget.exact_clock_nbytes == OPPONENT_STREAM_CLOCK_NBYTES
    assert budget.exact_clock_delta_nbytes == OPPONENT_STREAM_CLOCK_DELTA_NBYTES
    assert OpponentStreamResourceBudget.from_dict(budget.to_dict()) == budget
    with pytest.raises(ValueError, match="fields"):
        OpponentStreamResourceBudget.from_dict({**budget.to_dict(), "extra": 1})
    with pytest.raises(ValueError, match="state_nbytes"):
        OpponentStreamResourceBudget.from_dict(
            {**budget.to_dict(), "state_nbytes": budget.state_nbytes + 1}
        )


@pytest.mark.unit
def test_safe_pursuit_loop_remains_scan_compatible() -> None:
    learner = LinearLearner(optimizer=LMS(step_size=0.01))
    stream = AdversarialPursuitStream(feature_dim=4)
    final, errors = run_pursuit_loop(learner, stream, 8, jr.key(0))
    assert final.weights.shape == (4,)
    assert errors.shape == (8,)
    assert bool(jnp.all(jnp.isfinite(errors)))


@pytest.mark.unit
def test_opponent_public_module_exports_exact_contract_surface() -> None:
    import alberta_framework.streams.opponent as namespace

    expected = {
        "ADVERSARIAL_PURSUIT_CONFIG_SCHEMA",
        "ADVERSARIAL_PURSUIT_EMIT_INPUT_SCHEMA",
        "ADVERSARIAL_PURSUIT_EMIT_RESULT_SCHEMA",
        "ADVERSARIAL_PURSUIT_RESOLVE_INPUT_SCHEMA",
        "ADVERSARIAL_PURSUIT_RESOLVE_RESULT_SCHEMA",
        "ADVERSARIAL_PURSUIT_STATE_SCHEMA",
        "LEARNING_OPPONENT_CONFIG_SCHEMA",
        "LEARNING_OPPONENT_INPUT_SCHEMA",
        "LEARNING_OPPONENT_RESULT_SCHEMA",
        "LEARNING_OPPONENT_STATE_SCHEMA",
        "OPPONENT_STREAM_CLOCK_DELTA_NBYTES",
        "OPPONENT_STREAM_CLOCK_NBYTES",
        "OPPONENT_STREAM_RESOURCE_SCHEMA",
        "AdversarialPursuitEmitResult",
        "AdversarialPursuitResolveResult",
        "AdversarialPursuitState",
        "AdversarialPursuitStream",
        "LearningOpponentState",
        "LearningOpponentStepResult",
        "LearningOpponentStream",
        "OpponentStreamResourceBudget",
        "measure_opponent_stream_state_nbytes",
        "migrate_legacy_adversarial_pursuit_state",
        "migrate_legacy_learning_opponent_state",
        "run_pursuit_loop",
    }
    assert expected <= set(namespace.__all__)
