"""Resumable atomic-composition contracts for the matrix-game runner."""

from __future__ import annotations

from typing import cast

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import alberta_framework.streams as public_streams
from alberta_framework.core.average_reward import (
    DifferentialSARSAAgent,
    DifferentialSARSAConfig,
    DifferentialSARSAState,
)
from alberta_framework.streams.matrix_game import (
    CONVENTION_GAME_RUNNER_EXACT_IDENTITY_NBYTES,
    CONVENTION_GAME_RUNNER_STATE_SCHEMA,
    ConventionGameConfig,
    ConventionGameRunnerState,
    RecurringConventionGame,
    convention_game_runner_resource_budget,
    init_matrix_game_runner,
    measure_convention_game_runner_state_nbytes,
    run_matrix_game,
    step_matrix_game_runner,
)

pytestmark = pytest.mark.unit

_INT32_MAX = 2**31 - 1
_UINT32_MAX = 2**32 - 1


def _agent() -> DifferentialSARSAAgent:
    return DifferentialSARSAAgent(
        DifferentialSARSAConfig(
            n_actions=3,
            q_step_size=0.1,
            average_reward_step_size=0.01,
            trace_decay=0.2,
            epsilon_start=0.1,
            epsilon_end=0.1,
            use_bias=False,
        )
    )


def _game() -> RecurringConventionGame:
    return RecurringConventionGame(
        ConventionGameConfig(
            n_actions=3,
            phase_length=2,
            offsets=(0, 1),
            feature_mode="context",
        )
    )


def _base(seed: int = 0) -> tuple[
    DifferentialSARSAAgent,
    RecurringConventionGame,
    ConventionGameRunnerState,
]:
    agent = _agent()
    game = _game()
    return agent, game, init_matrix_game_runner(agent, game, jax.random.key(seed))


def _set_exact_clock(
    state: ConventionGameRunnerState,
    words: tuple[int, int],
) -> ConventionGameRunnerState:
    exact = (words[0] << 32) | words[1]
    telemetry = min(exact, _INT32_MAX)
    word_array = jnp.asarray(words, dtype=jnp.uint32)
    count = jnp.asarray(telemetry, dtype=jnp.int32)
    return cast(
        ConventionGameRunnerState,
        state.replace(  # type: ignore[attr-defined]
            environment_state=state.environment_state.replace(  # type: ignore[attr-defined]
                step_count=count,
                step_words=word_array,
            ),
            agent_0_state=state.agent_0_state.replace(  # type: ignore[attr-defined]
                step_count=count,
                step_words=word_array,
            ),
            agent_1_state=state.agent_1_state.replace(  # type: ignore[attr-defined]
                step_count=count,
                step_words=word_array,
            ),
        ),
    )


def _assert_array_tree_equal(first: object, second: object) -> None:
    first_leaves, first_tree = jax.tree_util.tree_flatten(first)
    second_leaves, second_tree = jax.tree_util.tree_flatten(second)
    assert str(first_tree) == str(second_tree)
    for first_leaf, second_leaf in zip(first_leaves, second_leaves, strict=True):
        if isinstance(first_leaf, jax.Array) and isinstance(second_leaf, jax.Array):
            if jnp.issubdtype(first_leaf.dtype, jax.dtypes.prng_key):
                np.testing.assert_array_equal(
                    jax.random.key_data(first_leaf),
                    jax.random.key_data(second_leaf),
                )
            else:
                np.testing.assert_array_equal(first_leaf, second_leaf)


def test_resume_is_bit_exact_to_one_uninterrupted_run() -> None:
    agent, game, initial = _base(1)
    uninterrupted = run_matrix_game(
        agent,
        game,
        11,
        initial_state=initial,
    )
    first = run_matrix_game(agent, game, 4, initial_state=initial)
    resumed = run_matrix_game(agent, game, 7, initial_state=first.state)

    chex.assert_trees_all_equal(
        uninterrupted.rewards,
        jnp.concatenate((first.rewards, resumed.rewards)),
    )
    chex.assert_trees_all_equal(
        uninterrupted.actions,
        jnp.concatenate((first.actions, resumed.actions)),
    )
    chex.assert_trees_all_equal(
        uninterrupted.updates_applied,
        jnp.concatenate((first.updates_applied, resumed.updates_applied)),
    )
    _assert_array_tree_equal(uninterrupted.state, resumed.state)
    assert uninterrupted.environment_state is uninterrupted.state.environment_state
    assert uninterrupted.agent_states[0] is uninterrupted.state.agent_0_state
    assert uninterrupted.agent_states[1] is uninterrupted.state.agent_1_state


def test_one_child_refusal_rolls_back_environment_and_both_learners() -> None:
    agent, game, initial = _base(2)
    corrupted = cast(
        ConventionGameRunnerState,
        initial.replace(  # type: ignore[attr-defined]
            agent_1_state=initial.agent_1_state.replace(  # type: ignore[attr-defined]
                step_count=jnp.asarray(1, dtype=jnp.int32),
            )
        ),
    )

    result = jax.jit(step_matrix_game_runner, static_argnums=(0, 1))(
        agent,
        game,
        corrupted,
    )

    assert bool(result.environment_update_applied)
    chex.assert_trees_all_equal(
        result.learner_updates_applied,
        jnp.asarray((True, False)),
    )
    assert not bool(result.runner_state_valid)
    assert not bool(result.update_applied)
    assert float(result.reward) == 0.0
    _assert_array_tree_equal(result.state, corrupted)
    chex.assert_trees_all_equal(result.pre_step_words, result.post_step_words)


def test_finite_prestate_with_nonfinite_child_candidate_rolls_back_all_three() -> None:
    agent, game, initial = _base(3)
    huge = jnp.asarray(3.0e38, dtype=jnp.float32)
    overflowing_agent = cast(
        DifferentialSARSAState,
        initial.agent_0_state.replace(  # type: ignore[attr-defined]
            q_weights=jnp.full_like(initial.agent_0_state.q_weights, huge),
            last_observation=jnp.ones_like(initial.agent_0_state.last_observation),
        ),
    )
    hostile = cast(
        ConventionGameRunnerState,
        initial.replace(agent_0_state=overflowing_agent),  # type: ignore[attr-defined]
    )

    result = step_matrix_game_runner(agent, game, hostile)

    assert bool(result.runner_state_valid)
    assert not bool(result.candidate_state_finite)
    assert not bool(result.learner_updates_applied[0])
    assert bool(result.learner_updates_applied[1])
    assert not bool(result.update_applied)
    _assert_array_tree_equal(result.state, hostile)


def test_terminal_and_corrupt_alignment_are_diagnosed_three_way_noops() -> None:
    agent, game, initial = _base(4)
    terminal = _set_exact_clock(initial, (_UINT32_MAX, _UINT32_MAX))
    stopped = step_matrix_game_runner(agent, game, terminal)
    assert bool(stopped.runner_state_valid)
    assert bool(stopped.child_counters_aligned)
    assert not bool(stopped.lifetime_capacity_available)
    assert not bool(stopped.update_applied)
    _assert_array_tree_equal(stopped.state, terminal)

    misaligned = cast(
        ConventionGameRunnerState,
        initial.replace(  # type: ignore[attr-defined]
            agent_0_state=initial.agent_0_state.replace(  # type: ignore[attr-defined]
                step_count=jnp.asarray(1, dtype=jnp.int32),
                step_words=jnp.asarray((0, 1), dtype=jnp.uint32),
            )
        ),
    )
    refused = step_matrix_game_runner(agent, game, misaligned)
    assert not bool(refused.child_counters_aligned)
    assert not bool(refused.update_applied)
    _assert_array_tree_equal(refused.state, misaligned)


def test_runner_crosses_low_word_carry_beyond_int32_under_scan() -> None:
    agent, game, initial = _base(5)
    near_carry = _set_exact_clock(initial, (0, _UINT32_MAX - 1))
    result = run_matrix_game(agent, game, 3, initial_state=near_carry)

    chex.assert_trees_all_equal(result.updates_applied, jnp.ones((3,), dtype=jnp.bool_))
    chex.assert_trees_all_equal(
        result.post_step_words,
        jnp.asarray(
            ((0, _UINT32_MAX), (1, 0), (1, 1)),
            dtype=jnp.uint32,
        ),
    )
    for words in (
        result.state.environment_state.step_words,
        result.state.agent_0_state.step_words,
        result.state.agent_1_state.step_words,
    ):
        chex.assert_trees_all_equal(words, jnp.asarray((1, 1), dtype=jnp.uint32))
    assert int(result.state.environment_state.step_count) == _INT32_MAX
    assert int(result.state.agent_0_state.step_count) == _INT32_MAX
    assert int(result.state.agent_1_state.step_count) == _INT32_MAX


def test_short_run_matches_the_previous_scan_order_and_outputs() -> None:
    agent, game, initial = _base(6)
    current = run_matrix_game(agent, game, 6, initial_state=initial)

    def legacy_step(carry, unused):  # type: ignore[no-untyped-def]
        del unused
        environment, state_0, state_1 = carry
        action_0, action_1 = state_0.last_action, state_1.last_action
        reward, environment = game.step(environment, action_0, action_1)
        next_observation = game.observe(environment)
        state_0 = agent.update(state_0, reward, next_observation).state
        state_1 = agent.update(state_1, reward, next_observation).state
        return (
            environment,
            state_0,
            state_1,
        ), (reward, jnp.stack((action_0, action_1)))

    legacy_state, (legacy_rewards, legacy_actions) = jax.lax.scan(
        legacy_step,
        (
            initial.environment_state,
            initial.agent_0_state,
            initial.agent_1_state,
        ),
        jnp.arange(6, dtype=jnp.int32),
    )
    chex.assert_trees_all_equal(current.rewards, legacy_rewards)
    chex.assert_trees_all_equal(current.actions, legacy_actions)
    _assert_array_tree_equal(
        current.state,
        ConventionGameRunnerState(  # type: ignore[call-arg]
            environment_state=legacy_state[0],
            agent_0_state=legacy_state[1],
            agent_1_state=legacy_state[2],
        ),
    )


def test_runner_schema_resources_and_public_surface_are_exact() -> None:
    agent, game, initial = _base(7)
    budget = convention_game_runner_resource_budget(initial)
    measured = measure_convention_game_runner_state_nbytes(initial)

    assert CONVENTION_GAME_RUNNER_STATE_SCHEMA == "alberta.convention-game-runner-state.v1"
    assert budget.state_schema == CONVENTION_GAME_RUNNER_STATE_SCHEMA
    assert budget.environment_exact_identity_nbytes == 8
    assert budget.learner_exact_identity_nbytes == 16
    assert budget.exact_identity_nbytes == CONVENTION_GAME_RUNNER_EXACT_IDENTITY_NBYTES == 24
    assert budget.state_nbytes == measured
    assert (
        budget.environment_state_nbytes
        + budget.agent_0_state_nbytes
        + budget.agent_1_state_nbytes
        == budget.state_nbytes
    )
    advanced = run_matrix_game(agent, game, 2, initial_state=initial)
    assert measure_convention_game_runner_state_nbytes(advanced.state) == measured

    for name in (
        "CONVENTION_GAME_RUNNER_EXACT_IDENTITY_NBYTES",
        "CONVENTION_GAME_RUNNER_STATE_SCHEMA",
        "ConventionGameRunnerState",
        "convention_game_runner_resource_budget",
        "init_matrix_game_runner",
        "measure_convention_game_runner_state_nbytes",
        "step_matrix_game_runner",
    ):
        assert hasattr(public_streams, name)


def test_runner_requires_exactly_one_initialization_authority() -> None:
    agent, game, initial = _base(8)
    with pytest.raises(ValueError, match="exactly one"):
        run_matrix_game(agent, game, 1)
    with pytest.raises(ValueError, match="exactly one"):
        run_matrix_game(
            agent,
            game,
            1,
            jax.random.key(8),
            initial_state=initial,
        )
    with pytest.raises(ValueError, match="num_steps"):
        run_matrix_game(agent, game, True, initial_state=initial)
