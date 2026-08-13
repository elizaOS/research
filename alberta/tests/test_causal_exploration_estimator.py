# mypy: disable-error-code="attr-defined,call-arg,misc,no-untyped-call,type-var"
"""Unit contracts for causal WP5 exploration scores, shield, and world."""

from __future__ import annotations

import dataclasses
from typing import cast

import chex
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.causal_exploration_estimator import (
    CAUSAL_EXPLORATION_ASSESSMENT_STATUS,
    CAUSAL_EXPLORATION_OUTPUT_WRITE_AUTHORITY,
    CAUSAL_EXPLORATION_PROMOTION_AUTHORITY,
    COLLECT_ACTION,
    INVEST_ACTION,
    NOISY_TV_ACTION,
    CallerOwnedHardShieldConfig,
    CausalExplorationEstimator,
    CausalExplorationEstimatorConfig,
    ExecutedExplorationTransition,
    ExplorationExogenousEvent,
    RankedExplorationDecision,
    ShieldedExplorationDecision,
    StochasticTrapEnvironmentConfig,
    StochasticTrapEnvironmentState,
    apply_caller_owned_hard_shield,
    caller_owned_hard_shield_state_valid,
    initial_caller_owned_hard_shield,
    initial_stochastic_trap_environment,
    measure_causal_exploration_core_resources,
    stochastic_trap_environment_state_valid,
    stochastic_trap_environment_step,
    stochastic_trap_observation,
    stochastic_trap_safety_mask,
)

pytestmark = [pytest.mark.unit, pytest.mark.development]


def _digest(index: int) -> tuple[int, ...]:
    return (0xCA550000 + index, index, index + 1, 7, 9, 0, 0, 1)


def _estimator_config() -> CausalExplorationEstimatorConfig:
    return CausalExplorationEstimatorConfig(
        ensemble_size=3,
        discount=0.95,
        step_size=0.18,
        prior_scale=0.04,
        fast_error_rate=0.4,
        slow_error_rate=0.08,
        weight_clip=10.0,
        metric_cap=10.0,
        estimator_owner_digest=_digest(1),
        action_owner_digest=_digest(2),
        decision_owner_digest=_digest(3),
        environment_owner_digest=_digest(4),
    )


def _environment_config() -> StochasticTrapEnvironmentConfig:
    return StochasticTrapEnvironmentConfig(
        delayed_investments_required=3,
        stabilize_reward=0.08,
        invest_cost=-0.04,
        collect_reward=0.9,
        observation_noise_scale=5.0,
        reward_noise_scale=0.01,
        schedule_owner_digest=_digest(6),
        environment_owner_digest=_digest(4),
        estimator_owner_digest=_digest(1),
        action_owner_digest=_digest(2),
        decision_owner_digest=_digest(3),
        shield_owner_digest=_digest(5),
    )


def _shield_config() -> CallerOwnedHardShieldConfig:
    return CallerOwnedHardShieldConfig(
        estimator_owner_digest=_digest(1),
        action_owner_digest=_digest(2),
        decision_owner_digest=_digest(3),
        shield_owner_digest=_digest(5),
    )


def _transition(
    revision: int,
    *,
    action: int = 0,
    observation: jnp.ndarray | None = None,
) -> ExecutedExplorationTransition:
    obs = (
        jnp.zeros((5,), dtype=jnp.float32)
        if observation is None
        else jnp.asarray(observation, dtype=jnp.float32)
    )
    event = jnp.asarray((0, revision + 1), dtype=jnp.uint32)
    return ExecutedExplorationTransition(
        observation=obs,
        action=jnp.asarray(action, dtype=jnp.int32),
        reward=jnp.asarray(0.1, dtype=jnp.float32),
        next_observation=obs.at[0].set(jnp.float32(0.25)),
        source_event_words=event,
        decision_words=event,
        estimator_revision_words=jnp.asarray((0, revision), dtype=jnp.uint32),
        estimator_owner_digest=jnp.asarray(_digest(1), dtype=jnp.uint32),
        action_owner_digest=jnp.asarray(_digest(2), dtype=jnp.uint32),
        decision_owner_digest=jnp.asarray(_digest(3), dtype=jnp.uint32),
        environment_owner_digest=jnp.asarray(_digest(4), dtype=jnp.uint32),
    )


def _ranked(event: int, selected_action: int) -> RankedExplorationDecision:
    return RankedExplorationDecision(
        selected_action=jnp.asarray(selected_action, dtype=jnp.int32),
        host_action=jnp.asarray(0, dtype=jnp.int32),
        ranking_applied=jnp.asarray(True, dtype=jnp.bool_),
        source_event_words=jnp.asarray((0, event), dtype=jnp.uint32),
        pre_decision_words=jnp.asarray((0, event - 1), dtype=jnp.uint32),
        post_decision_words=jnp.asarray((0, event), dtype=jnp.uint32),
        estimator_revision_words=jnp.asarray((0, event - 1), dtype=jnp.uint32),
        estimator_owner_digest=jnp.asarray(_digest(1), dtype=jnp.uint32),
        decision_owner_digest=jnp.asarray(_digest(3), dtype=jnp.uint32),
    )


def _event(event: int, *, tv_noise: float = 0.0) -> ExplorationExogenousEvent:
    return ExplorationExogenousEvent(
        source_event_words=jnp.asarray((0, event), dtype=jnp.uint32),
        stable_noise=jnp.asarray(0.1, dtype=jnp.float32),
        reward_noise=jnp.asarray(0.0, dtype=jnp.float32),
        noisy_tv_noise=jnp.asarray(tv_noise, dtype=jnp.float32),
        schedule_owner_digest=jnp.asarray(_digest(6), dtype=jnp.uint32),
    )


def _executed(event: int, action: int) -> ShieldedExplorationDecision:
    return ShieldedExplorationDecision(
        action=jnp.asarray(action, dtype=jnp.int32),
        action_available=jnp.asarray(True, dtype=jnp.bool_),
        executed_action_safety_allowed=jnp.asarray(True, dtype=jnp.bool_),
        source_event_words=jnp.asarray((0, event), dtype=jnp.uint32),
        decision_words=jnp.asarray((0, event), dtype=jnp.uint32),
        estimator_revision_words=jnp.asarray((0, event - 1), dtype=jnp.uint32),
        estimator_owner_digest=jnp.asarray(_digest(1), dtype=jnp.uint32),
        action_owner_digest=jnp.asarray(_digest(2), dtype=jnp.uint32),
        decision_owner_digest=jnp.asarray(_digest(3), dtype=jnp.uint32),
        shield_owner_digest=jnp.asarray(_digest(5), dtype=jnp.uint32),
    )


def test_status_and_authority_boundary_is_strict() -> None:
    assert CAUSAL_EXPLORATION_ASSESSMENT_STATUS == "not_assessed"
    assert CAUSAL_EXPLORATION_OUTPUT_WRITE_AUTHORITY is False
    assert CAUSAL_EXPLORATION_PROMOTION_AUTHORITY is False


def test_scores_are_online_causal_and_progress_means_error_reduction_only() -> None:
    estimator = CausalExplorationEstimator(_estimator_config())
    state = estimator.init(jr.key(7, impl="threefry2x32"))
    observation = jnp.zeros((5,), dtype=jnp.float32)
    estimates = estimator.estimate(
        state,
        observation,
        jnp.asarray((0, 1), dtype=jnp.uint32),
    )
    assert bool(estimates.causal_online_estimate)
    assert not bool(estimates.oracle_input_used)
    assert estimates.candidate_actions.tolist() == [0, 1, 2, 3]
    assert np.all(np.isfinite(np.asarray(estimates.expected_improvement)))
    np.testing.assert_array_equal(estimates.learning_progress, jnp.zeros((4,)))

    rising_error = cast(
        type(state),
        state.replace(
            fast_absolute_td_error=jnp.asarray((2.0, 0.0, 0.0, 0.0), dtype=jnp.float32),
            slow_absolute_td_error=jnp.asarray((1.0, 0.0, 0.0, 0.0), dtype=jnp.float32),
        ),
    )
    rising = estimator.estimate(
        rising_error,
        observation,
        jnp.asarray((0, 1), dtype=jnp.uint32),
    )
    assert float(rising.learning_progress[0]) == 0.0

    falling_error = cast(
        type(state),
        state.replace(
            fast_absolute_td_error=jnp.asarray((1.0, 0.0, 0.0, 0.0), dtype=jnp.float32),
            slow_absolute_td_error=jnp.asarray((2.0, 0.0, 0.0, 0.0), dtype=jnp.float32),
        ),
    )
    falling = estimator.estimate(
        falling_error,
        observation,
        jnp.asarray((0, 1), dtype=jnp.uint32),
    )
    assert float(falling.learning_progress[0]) == 1.0


def test_estimator_exact_revision_stale_owner_and_numeric_failure_are_atomic() -> None:
    estimator = CausalExplorationEstimator(_estimator_config())
    initial = estimator.init(jr.key(8, impl="threefry2x32"))
    accepted = estimator.update(initial, _transition(0))
    assert bool(accepted.applied)
    assert accepted.state.revision_words.tolist() == [0, 1]
    assert int(jnp.sum(accepted.state.action_counts)) == 1

    stale = estimator.update(accepted.state, _transition(0))
    assert not bool(stale.applied)
    chex.assert_trees_all_equal(stale.state, accepted.state)

    aliased = dataclasses.replace(
        _transition(1),
        action_owner_digest=jnp.asarray(_digest(9), dtype=jnp.uint32),
    )
    rejected_alias = estimator.update(accepted.state, aliased)
    assert not bool(rejected_alias.applied)
    chex.assert_trees_all_equal(rejected_alias.state, accepted.state)

    huge = jnp.full((5,), jnp.finfo(jnp.float32).max, dtype=jnp.float32)
    rejected_numeric = estimator.update(
        accepted.state,
        _transition(1, action=1, observation=huge),
    )
    assert not bool(rejected_numeric.applied)
    chex.assert_trees_all_equal(rejected_numeric.state, accepted.state)
    assert rejected_numeric.state.revision_words.tolist() == [0, 1]
    assert int(jnp.sum(rejected_numeric.state.action_counts)) == 1

    out_of_bounds = cast(
        type(accepted.state),
        accepted.state.replace(
            weights=accepted.state.weights.at[0, 0, 0].set(jnp.float32(10.5))
        ),
    )
    assert not bool(estimator.state_valid(out_of_bounds))
    rejected_tamper = estimator.update(out_of_bounds, _transition(1))
    assert not bool(rejected_tamper.applied)
    chex.assert_trees_all_equal(rejected_tamper.state, out_of_bounds)


def test_hard_shield_is_actual_admissibility_owner_and_unavailable_is_noop() -> None:
    shield_config = _shield_config()
    initial = initial_caller_owned_hard_shield(shield_config)
    environment_config = _environment_config()
    environment = initial_stochastic_trap_environment(environment_config)
    locked_mask = stochastic_trap_safety_mask(environment_config, environment)
    assert locked_mask.tolist() == [True, True, True, False]

    fallback = apply_caller_owned_hard_shield(
        shield_config,
        initial,
        _ranked(1, COLLECT_ACTION),
        locked_mask,
    )
    assert bool(fallback.applied)
    assert bool(fallback.fallback_used)
    assert int(fallback.decision.action) == 0
    assert fallback.state.revision_words.tolist() == [0, 1]

    no_safe_action = apply_caller_owned_hard_shield(
        shield_config,
        initial,
        _ranked(1, COLLECT_ACTION),
        jnp.zeros((4,), dtype=jnp.bool_),
    )
    assert not bool(no_safe_action.applied)
    assert not bool(no_safe_action.decision.action_available)
    chex.assert_trees_all_equal(no_safe_action.state, initial)
    assert initial.revision_words.tolist() == [0, 0]

    tampered = cast(
        type(initial),
        initial.replace(revision_words=jnp.asarray((0, 1), dtype=jnp.uint32)),
    )
    assert not bool(caller_owned_hard_shield_state_valid(shield_config, tampered))
    rejected = apply_caller_owned_hard_shield(
        shield_config,
        tampered,
        _ranked(1, 0),
        locked_mask,
    )
    assert not bool(rejected.applied)
    chex.assert_trees_all_equal(rejected.state, tampered)


def test_delayed_benefit_noisy_tv_and_environment_state_validity() -> None:
    config = _environment_config()
    state = initial_stochastic_trap_environment(config)
    assert bool(stochastic_trap_environment_state_valid(config, state))
    rewards: list[float] = []
    for event_index in range(1, 4):
        result = stochastic_trap_environment_step(
            config,
            state,
            _event(event_index),
            _executed(event_index, INVEST_ACTION),
        )
        assert bool(result.applied)
        assert bool(result.delayed_investment_applied)
        rewards.append(float(result.reward))
        state = result.state
    assert rewards == pytest.approx([-0.04, -0.04, -0.04])
    assert int(state.delayed_progress) == 3
    assert stochastic_trap_safety_mask(config, state).tolist() == [True, True, True, True]

    collected = stochastic_trap_environment_step(
        config,
        state,
        _event(4),
        _executed(4, COLLECT_ACTION),
    )
    assert bool(collected.applied)
    assert bool(collected.delayed_collection_applied)
    assert float(collected.reward) == pytest.approx(0.9)
    assert int(collected.state.delayed_progress) == 0

    tv = stochastic_trap_environment_step(
        config,
        collected.state,
        _event(5, tv_noise=2.0),
        _executed(5, NOISY_TV_ACTION),
    )
    assert bool(tv.applied)
    assert bool(tv.noisy_tv_observed)
    assert float(tv.state.noisy_tv_channel) == pytest.approx(10.0)
    assert float(stochastic_trap_observation(config, tv.state)[3]) > 0.99

    initial = initial_stochastic_trap_environment(config)
    unsafe = stochastic_trap_environment_step(
        config,
        initial,
        _event(1),
        _executed(1, COLLECT_ACTION),
    )
    assert not bool(unsafe.hard_safety_valid)
    assert not bool(unsafe.applied)
    chex.assert_trees_all_equal(unsafe.state, initial)

    bad_progress = cast(
        StochasticTrapEnvironmentState,
        initial.replace(delayed_progress=jnp.asarray(4, dtype=jnp.int32)),
    )
    assert not bool(stochastic_trap_environment_state_valid(config, bad_progress))
    rejected = stochastic_trap_environment_step(
        config,
        bad_progress,
        _event(1),
        _executed(1, INVEST_ACTION),
    )
    assert not bool(rejected.applied)
    chex.assert_trees_all_equal(rejected.state, bad_progress)

    bad_clock = cast(
        StochasticTrapEnvironmentState,
        initial.replace(last_decision_words=jnp.asarray((0, 1), dtype=jnp.uint32)),
    )
    assert not bool(stochastic_trap_environment_state_valid(config, bad_clock))


def test_resource_partition_covers_every_persistent_core_leaf() -> None:
    estimator = CausalExplorationEstimator(_estimator_config()).init(
        jr.key(9, impl="threefry2x32")
    )
    environment = initial_stochastic_trap_environment(_environment_config())
    shield = initial_caller_owned_hard_shield(_shield_config())
    resources = measure_causal_exploration_core_resources(
        estimator,
        environment,
        shield,
    )
    assert resources.total_state_nbytes == (
        resources.estimator_state_nbytes
        + resources.environment_state_nbytes
        + resources.hard_shield_state_nbytes
    )
    assert min(dataclasses.asdict(resources).values()) > 0
