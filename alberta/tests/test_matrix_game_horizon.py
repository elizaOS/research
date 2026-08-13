"""Exact finite-clock continual-horizon contracts for the recurring matrix game."""

from __future__ import annotations

from typing import cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import alberta_framework.streams as public_streams
from alberta_framework.streams.matrix_game import (
    CONVENTION_GAME_EXACT_CLOCK_DELTA_NBYTES,
    CONVENTION_GAME_EXACT_CLOCK_NBYTES,
    ConventionGameConfig,
    ConventionGameState,
    ConventionGameStepResult,
    RecurringConventionGame,
    measure_convention_game_state_nbytes,
)

pytestmark = pytest.mark.unit

_INT32_MAX = 2**31 - 1
_UINT32_MAX = 2**32 - 1


def _game() -> RecurringConventionGame:
    return RecurringConventionGame(
        ConventionGameConfig(
            n_actions=5,
            phase_length=7,
            offsets=(0, 2, 4),
            feature_mode="context",
        )
    )


def _state_at(
    game: RecurringConventionGame,
    high: int,
    low: int,
    *,
    last_actions: tuple[int, int] = (0, 0),
) -> ConventionGameState:
    exact = (high << 32) | low
    return cast(
        ConventionGameState,
        game.init(jr.key(11)).replace(  # type: ignore[attr-defined]
            step_count=jnp.asarray(min(exact, _INT32_MAX), dtype=jnp.int32),
            step_words=jnp.asarray((high, low), dtype=jnp.uint32),
            last_actions=jnp.asarray(last_actions, dtype=jnp.int32),
        ),
    )


def test_init_resources_and_public_exact_clock_surface() -> None:
    game = _game()
    state = game.init(jr.key(0))

    chex.assert_trees_all_equal(state.step_words, jnp.zeros((2,), dtype=jnp.uint32))
    assert int(state.step_count) == 0
    assert state.step_words.nbytes == CONVENTION_GAME_EXACT_CLOCK_NBYTES == 8
    assert CONVENTION_GAME_EXACT_CLOCK_DELTA_NBYTES == 8
    assert measure_convention_game_state_nbytes(state) - state.step_words.nbytes == 20
    assert public_streams.CONVENTION_GAME_EXACT_CLOCK_NBYTES == 8
    assert public_streams.CONVENTION_GAME_EXACT_CLOCK_DELTA_NBYTES == 8
    assert public_streams.ConventionGameStepResult is ConventionGameStepResult
    assert public_streams.measure_convention_game_state_nbytes is (
        measure_convention_game_state_nbytes
    )


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"n_actions": True}, "n_actions"),
        ({"n_actions": 2**31}, "n_actions"),
        ({"phase_length": True}, "phase_length"),
        ({"phase_length": 2**31}, "phase_length"),
        ({"offsets": [0, 1]}, "offsets"),
        ({"offsets": (0, True)}, "non-boolean"),
    ],
)
def test_config_rejects_ambiguous_or_out_of_contract_schedule_values(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        ConventionGameConfig(**kwargs)  # type: ignore[arg-type]


def test_non_power_of_two_high_word_schedule_math_is_exact() -> None:
    game = _game()
    for high, low in ((1, 0), (5, 123), (17, _UINT32_MAX - 3)):
        state = _state_at(game, high, low)
        exact = (high << 32) | low
        expected_phase = exact // game.config.phase_length
        expected_rule = expected_phase % game.config.n_rules

        phase_words = game.phase_words_of(state.step_words)
        actual_phase = (int(phase_words[0]) << 32) | int(phase_words[1])
        assert actual_phase == expected_phase
        assert int(game.rule_of(state.step_words)) == expected_rule
        assert int(game.phase_index_of(state.step_words)) == min(
            expected_phase,
            _INT32_MAX,
        )
        chex.assert_trees_all_equal(
            game.observe(state),
            jax.nn.one_hot(expected_rule, game.config.n_rules, dtype=jnp.float32),
        )

        offset = game.config.offsets[expected_rule]
        result = jax.jit(game.step_result)(
            state,
            jnp.asarray(offset, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
        )
        assert bool(result.update_applied)
        assert float(result.reward) == 1.0
        assert int(result.rule_index) == expected_rule


def test_large_cycle_fallback_uses_exact_phase_words() -> None:
    game = RecurringConventionGame(
        ConventionGameConfig(
            n_actions=2,
            phase_length=1_500_000_000,
            offsets=(0, 1),
        )
    )
    words = jnp.asarray((7, 1_234_567_890), dtype=jnp.uint32)
    exact = (7 << 32) | 1_234_567_890
    assert game.config.phase_length * game.config.n_rules > _INT32_MAX
    assert int(game.rule_of(words)) == (exact // game.config.phase_length) % 2


def test_low_word_carry_and_jitted_scan_remain_exact_after_telemetry_saturates() -> None:
    game = _game()
    initial = _state_at(game, 0, _UINT32_MAX - 1)
    actions = jnp.asarray(((0, 0), (2, 0), (4, 0)), dtype=jnp.int32)

    @jax.jit
    def scan_steps(
        state: ConventionGameState,
        joint_actions: jax.Array,
    ) -> tuple[ConventionGameState, tuple[jax.Array, jax.Array, jax.Array]]:
        def scan_step(
            carry: ConventionGameState,
            action: jax.Array,
        ) -> tuple[ConventionGameState, tuple[jax.Array, jax.Array, jax.Array]]:
            result = game.step_result(carry, action[0], action[1])
            return result.state, (
                result.post_step_words,
                result.update_applied,
                result.rule_index,
            )

        return jax.lax.scan(scan_step, state, joint_actions)

    final, (words, applied, rules) = scan_steps(initial, actions)
    chex.assert_trees_all_equal(
        words,
        jnp.asarray(
            ((0, _UINT32_MAX), (1, 0), (1, 1)),
            dtype=jnp.uint32,
        ),
    )
    assert bool(jnp.all(applied))
    expected_rules = [
        ((((0 << 32) | (_UINT32_MAX - 1 + i)) // 7) % 3)
        for i in range(3)
    ]
    chex.assert_trees_all_equal(rules, jnp.asarray(expected_rules, dtype=jnp.int32))
    chex.assert_trees_all_equal(final.step_words, jnp.asarray((1, 1), dtype=jnp.uint32))
    assert int(final.step_count) == _INT32_MAX


def test_terminal_all_ones_disarms_atomically_under_jit() -> None:
    game = _game()
    terminal = _state_at(game, _UINT32_MAX, _UINT32_MAX, last_actions=(3, 1))
    result = jax.jit(game.step_result)(
        terminal,
        jnp.asarray(4, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
    )

    assert bool(result.lifetime_counter_valid)
    assert not bool(result.lifetime_capacity_available)
    assert bool(result.state_valid)
    assert bool(result.input_valid)
    assert not bool(result.update_applied)
    assert float(result.reward) == 0.0
    assert int(result.rule_index) == -1
    chex.assert_trees_all_equal(result.state, terminal)
    chex.assert_trees_all_equal(result.pre_step_words, result.post_step_words)


def test_invalid_value_state_or_action_rolls_back_every_state_leaf() -> None:
    game = _game()
    valid = _state_at(game, 0, 3, last_actions=(2, 1))
    corrupt_clock = cast(
        ConventionGameState,
        valid.replace(step_count=jnp.asarray(2, dtype=jnp.int32)),  # type: ignore[attr-defined]
    )
    corrupt_history = cast(
        ConventionGameState,
        valid.replace(  # type: ignore[attr-defined]
            last_actions=jnp.asarray((5, 0), dtype=jnp.int32)
        ),
    )

    eager = game.step_result(
        corrupt_clock,
        jnp.asarray(1, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
    )
    assert not bool(eager.update_applied)
    chex.assert_trees_all_equal(eager.state, corrupt_clock)

    for invalid_state in (corrupt_clock, corrupt_history):
        result = jax.jit(game.step_result)(
            invalid_state,
            jnp.asarray(1, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
        )
        assert not bool(result.state_valid)
        assert not bool(result.update_applied)
        assert float(result.reward) == 0.0
        chex.assert_trees_all_equal(result.state, invalid_state)
        chex.assert_trees_all_equal(game.observe(invalid_state), jnp.zeros((3,)))

    bad_action = jax.jit(game.step_result)(
        valid,
        jnp.asarray(5, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
    )
    assert bool(bad_action.state_valid)
    assert not bool(bad_action.input_valid)
    assert not bool(bad_action.update_applied)
    chex.assert_trees_all_equal(bad_action.state, valid)


def test_structural_state_and_action_errors_are_rejected_before_mutation() -> None:
    game = _game()
    state = game.init(jr.key(0))
    bad_words = cast(
        ConventionGameState,
        state.replace(  # type: ignore[attr-defined]
            step_words=jnp.zeros((3,), dtype=jnp.uint32)
        ),
    )
    with pytest.raises(ValueError, match="step_words"):
        game.step_result(
            bad_words,
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
        )
    with pytest.raises(TypeError, match="action_0"):
        game.step_result(
            state,
            jnp.asarray(0.0, dtype=jnp.float32),
            jnp.asarray(0, dtype=jnp.int32),
        )


def test_invalid_transition_inside_scan_is_a_noop_then_later_work_can_resume() -> None:
    game = _game()
    initial = game.init(jr.key(2))
    actions = jnp.asarray(((0, 0), (9, 0), (2, 0)), dtype=jnp.int32)

    def scan_step(
        state: ConventionGameState,
        action: jax.Array,
    ) -> tuple[ConventionGameState, jax.Array]:
        result = game.step_result(state, action[0], action[1])
        return result.state, result.update_applied

    final, applied = jax.jit(lambda s: jax.lax.scan(scan_step, s, actions))(initial)
    chex.assert_trees_all_equal(applied, jnp.asarray((True, False, True)))
    chex.assert_trees_all_equal(final.step_words, jnp.asarray((0, 2), dtype=jnp.uint32))
    assert int(final.step_count) == 2


def test_short_horizon_rule_observation_and_reward_trajectory_is_legacy_identical() -> None:
    game = RecurringConventionGame(
        ConventionGameConfig(
            n_actions=5,
            phase_length=3,
            offsets=(0, 2),
            feature_mode="context",
        )
    )
    state = game.init(jr.key(3))
    actions = ((0, 0), (2, 0), (4, 2), (1, 1), (3, 1), (0, 3), (4, 4))
    for step, (action_0, action_1) in enumerate(actions):
        legacy_rule = (step // 3) % 2
        legacy_offset = (0, 2)[legacy_rule]
        legacy_reward = float(((action_0 - action_1) % 5) == legacy_offset)
        chex.assert_trees_all_equal(
            game.observe(state),
            jax.nn.one_hot(legacy_rule, 2, dtype=jnp.float32),
        )
        reward, state = game.step(
            state,
            jnp.asarray(action_0, dtype=jnp.int32),
            jnp.asarray(action_1, dtype=jnp.int32),
        )
        assert float(reward) == legacy_reward
        assert int(state.step_count) == step + 1
        chex.assert_trees_all_equal(
            state.step_words,
            jnp.asarray((0, step + 1), dtype=jnp.uint32),
        )


def test_legacy_scalar_schedule_queries_remain_short_horizon_compatible() -> None:
    game = _game()
    for step in range(100):
        scalar = jnp.asarray(step, dtype=jnp.int32)
        assert int(game.rule_of(scalar)) == (step // 7) % 3
        assert int(game.phase_index_of(scalar)) == step // 7
    assert int(game.rule_of(jnp.asarray(-1, dtype=jnp.int32))) == -1
    assert int(game.phase_index_of(jnp.asarray(-1, dtype=jnp.int32))) == -1
    assert int(game.rule_of(jnp.asarray(_INT32_MAX, dtype=jnp.int32))) == -1
    assert int(game.phase_index_of(jnp.asarray(_INT32_MAX, dtype=jnp.int32))) == -1
    assert int(game.rule_of(jnp.asarray(_UINT32_MAX, dtype=jnp.uint32))) == -1
    assert int(game.phase_index_of(jnp.asarray(_UINT32_MAX, dtype=jnp.uint32))) == -1
