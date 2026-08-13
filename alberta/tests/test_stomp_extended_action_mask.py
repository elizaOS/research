"""Behavior and bootstrap contracts for opt-in STOMP action masks."""

from __future__ import annotations

import chex
import jax.numpy as jnp
import jax.random as jr
import pytest

from alberta_framework.core.options import STOMPAgent, STOMPConfig, STOMPState, SubtaskSpec

pytestmark = pytest.mark.unit


def _config(*, epsilon: float, n_options: int = 1, backups: int = 0) -> STOMPConfig:
    return STOMPConfig(
        subtask_specs=tuple(
            SubtaskSpec(feature_index=0, threshold=100.0, max_option_steps=3)
            for _ in range(n_options)
        ),
        observation_dim=2,
        n_primitive_actions=2,
        epsilon_base=epsilon,
        epsilon_option=0.0,
        option_planning_backups_per_step=backups,
    )


def _with_constant_head_values(
    state: STOMPState,
    values: tuple[float, ...],
) -> STOMPState:
    learner = state.base_learner_state
    assert len(values) == len(learner.head_params.biases)
    params = learner.head_params.replace(
        weights=tuple(jnp.zeros_like(weight) for weight in learner.head_params.weights),
        biases=tuple(
            jnp.full_like(bias, value)
            for bias, value in zip(learner.head_params.biases, values, strict=True)
        ),
    )
    return state.replace(base_learner_state=learner.replace(head_params=params))


def _normalize_last_head(actual: STOMPState, expected: STOMPState) -> STOMPState:
    learner = actual.base_learner_state
    biases = list(learner.head_params.biases)
    biases[-1] = expected.base_learner_state.head_params.biases[-1]
    return actual.replace(
        base_learner_state=learner.replace(
            head_params=learner.head_params.replace(biases=tuple(biases))
        )
    )


def test_all_true_mask_is_exact_start_and_update_legacy_parity() -> None:
    agent = STOMPAgent(_config(epsilon=0.35, n_options=2, backups=1))
    initial = agent.init(jr.key(7))
    observation = jnp.asarray([0.2, -0.4], dtype=jnp.float32)
    all_true = jnp.ones((agent.config.n_total_actions,), dtype=jnp.bool_)

    legacy_start = agent.start(initial, observation)
    masked_start = agent.start_with_extended_action_mask(initial, observation, all_true)
    chex.assert_trees_all_equal(masked_start.state, legacy_start)
    chex.assert_trees_all_equal(masked_start.state.rng_key, legacy_start.rng_key)
    chex.assert_trees_all_equal(
        masked_start.primitive_action,
        legacy_start.last_primitive_action,
    )

    next_observation = jnp.asarray([-0.1, 0.8], dtype=jnp.float32)
    legacy_update = agent.update(
        legacy_start,
        jnp.asarray(0.3, dtype=jnp.float32),
        next_observation,
        jnp.asarray(0.9, dtype=jnp.float32),
    )
    masked_update = agent.update(
        legacy_start,
        jnp.asarray(0.3, dtype=jnp.float32),
        next_observation,
        jnp.asarray(0.9, dtype=jnp.float32),
        extended_action_mask=all_true,
    )
    chex.assert_trees_all_equal(masked_update, legacy_update)
    chex.assert_trees_all_equal(masked_update.state.rng_key, legacy_update.state.rng_key)


@pytest.mark.parametrize("epsilon", [0.0, 1.0])
def test_huge_inactive_head_neither_dispatches_nor_changes_real_td_target(
    epsilon: float,
) -> None:
    agent = STOMPAgent(_config(epsilon=epsilon))
    initial = agent.init(jr.key(11))
    low = _with_constant_head_values(initial, (0.25, 0.0, -3.0))
    huge = _with_constant_head_values(initial, (0.25, 0.0, 1.0e6))
    mask = jnp.asarray([True, True, False], dtype=jnp.bool_)
    observation = jnp.asarray([0.1, 0.2], dtype=jnp.float32)

    low_start = agent.start_with_extended_action_mask(low, observation, mask)
    huge_start = agent.start_with_extended_action_mask(huge, observation, mask)
    assert int(low_start.state.base_last_action) < agent.config.n_primitive_actions
    assert int(huge_start.state.base_last_action) < agent.config.n_primitive_actions
    assert int(low_start.state.executing_option) == -1
    assert int(huge_start.state.executing_option) == -1
    assert int(low_start.state.base_last_action) == int(huge_start.state.base_last_action)

    next_observation = jnp.asarray([-0.3, 0.7], dtype=jnp.float32)
    low_update = agent.update(
        low_start.state,
        jnp.asarray(0.4, dtype=jnp.float32),
        next_observation,
        jnp.asarray(0.8, dtype=jnp.float32),
        extended_action_mask=mask,
    )
    huge_update = agent.update(
        huge_start.state,
        jnp.asarray(0.4, dtype=jnp.float32),
        next_observation,
        jnp.asarray(0.8, dtype=jnp.float32),
        extended_action_mask=mask,
    )
    chex.assert_trees_all_equal(huge_update.td_error, low_update.td_error)
    chex.assert_trees_all_equal(huge_update.average_reward, low_update.average_reward)
    chex.assert_trees_all_equal(
        _normalize_last_head(huge_update.state, low_update.state),
        low_update.state,
    )
    assert int(huge_update.state.base_last_action) < agent.config.n_primitive_actions
    assert int(huge_update.state.executing_option) == -1


def test_inactive_head_cannot_enter_option_model_planning_bootstrap() -> None:
    agent = STOMPAgent(_config(epsilon=0.0, n_options=2, backups=1))
    initial = agent.init(jr.key(19))
    low = _with_constant_head_values(initial, (0.25, 0.0, -1.0, -3.0))
    huge = _with_constant_head_values(initial, (0.25, 0.0, -1.0, 1.0e6))
    mask = jnp.asarray([True, True, True, False], dtype=jnp.bool_)
    observation = jnp.asarray([0.2, 0.4], dtype=jnp.float32)
    low = agent.start_with_extended_action_mask(low, observation, mask).state
    huge = agent.start_with_extended_action_mask(huge, observation, mask).state

    def with_completed_model(state: STOMPState) -> STOMPState:
        models = state.option_models.replace(
            env_return_ema=state.option_models.env_return_ema.at[0].set(0.5),
            duration_ema=state.option_models.duration_ema.at[0].set(1.0),
            baseline_mass_ema=state.option_models.baseline_mass_ema.at[0].set(1.0),
            discount_ema=state.option_models.discount_ema.at[0].set(0.8),
            n_completions=state.option_models.n_completions.at[0].set(1),
        )
        return state.replace(option_models=models)

    low = with_completed_model(low)
    huge = with_completed_model(huge)
    next_observation = jnp.asarray([-0.2, 0.3], dtype=jnp.float32)
    low_result = agent.update(
        low,
        jnp.asarray(0.1, dtype=jnp.float32),
        next_observation,
        jnp.asarray(0.9, dtype=jnp.float32),
        extended_action_mask=mask,
    )
    huge_result = agent.update(
        huge,
        jnp.asarray(0.1, dtype=jnp.float32),
        next_observation,
        jnp.asarray(0.9, dtype=jnp.float32),
        extended_action_mask=mask,
    )
    assert int(low_result.planning_backups) == 1
    assert int(huge_result.planning_backups) == 1
    chex.assert_trees_all_equal(
        huge_result.planning_td_error,
        low_result.planning_td_error,
    )
    chex.assert_trees_all_equal(
        _normalize_last_head(huge_result.state, low_result.state),
        low_result.state,
    )


def test_scan_threads_cold_option_masks_without_dispatch() -> None:
    agent = STOMPAgent(_config(epsilon=1.0))
    initial = _with_constant_head_values(
        agent.init(jr.key(23)),
        (0.0, 0.0, 1.0e6),
    )
    mask = jnp.asarray([True, True, False], dtype=jnp.bool_)
    started = agent.start_with_extended_action_mask(
        initial,
        jnp.asarray([0.0, 0.0], dtype=jnp.float32),
        mask,
    ).state
    length = 4
    result = agent.scan(
        started,
        jnp.zeros((length,), dtype=jnp.float32),
        jnp.zeros((length, 2), dtype=jnp.float32),
        jnp.ones((length,), dtype=jnp.float32),
        extended_action_masks=jnp.broadcast_to(mask, (length, 3)),
    )
    assert bool(jnp.all(result.executing_options == -1))
    assert bool(jnp.all(result.primitive_actions < 2))
    assert int(result.state.executing_option) == -1
