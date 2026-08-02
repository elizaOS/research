# mypy: disable-error-code="attr-defined,call-arg,no-any-return,arg-type,operator"
"""Contracts for the bounded continuous average-reward actor-critic core."""

from __future__ import annotations

import json
from dataclasses import fields
from typing import Any

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework import ContinuousAverageRewardActorCriticAgent as TopLevelAgent
from alberta_framework.core import ContinuousAverageRewardActorCriticAgent as CoreAgent
from alberta_framework.core.continuous_average_reward_actor_critic import (
    TRANSFORMED_LOG_DENSITY_MAX_ULPS,
    ContinuousAverageRewardActorCriticAgent,
    ContinuousAverageRewardActorCriticConfig,
    ContinuousAverageRewardActorCriticState,
    SquashedGaussianPolicySample,
    float32_ulp_distance,
    transformed_diagonal_gaussian_log_density,
)

pytestmark = pytest.mark.unit


def _config(**overrides: Any) -> ContinuousAverageRewardActorCriticConfig:
    values: dict[str, Any] = {
        "action_dim": 2,
        "action_low": (-2.0, -0.5),
        "action_high": (1.0, 2.5),
        "actor_step_size": 0.03,
        "critic_step_size": 0.08,
        "average_reward_step_size": 0.02,
        "actor_trace_lambda": 0.4,
        "critic_trace_lambda": 0.6,
        "target_log_std_init": -0.3,
        "target_log_std_min": -4.0,
        "target_log_std_max": 1.0,
        "behavior_std_scale": 1.0,
        "max_updates": 100,
    }
    values.update(overrides)
    return ContinuousAverageRewardActorCriticConfig(**values)


def _started(
    agent: ContinuousAverageRewardActorCriticAgent,
    *,
    key: int = 0,
    observation: jax.Array | None = None,
) -> ContinuousAverageRewardActorCriticState:
    state = agent.init(feature_dim=3, key=jr.key(key))
    obs = jnp.asarray([1.0, -0.5, 0.25], dtype=jnp.float32) if observation is None else observation
    result = agent.start(state, obs)
    assert bool(result.accepted)
    return result.state


def _assert_bit_exact(left: Any, right: Any) -> None:
    left_leaves, left_tree = jax.tree_util.tree_flatten(left)
    right_leaves, right_tree = jax.tree_util.tree_flatten(right)
    assert left_tree == right_tree
    for lhs, rhs in zip(left_leaves, right_leaves, strict=True):
        if jax.dtypes.issubdtype(lhs.dtype, jax.dtypes.prng_key):
            lhs = jr.key_data(lhs)
            rhs = jr.key_data(rhs)
        np.testing.assert_array_equal(np.asarray(lhs), np.asarray(rhs))


def _assert_close(left: Any, right: Any) -> None:
    left_leaves, left_tree = jax.tree_util.tree_flatten(left)
    right_leaves, right_tree = jax.tree_util.tree_flatten(right)
    assert left_tree == right_tree
    for lhs, rhs in zip(left_leaves, right_leaves, strict=True):
        if jax.dtypes.issubdtype(lhs.dtype, jax.dtypes.prng_key):
            np.testing.assert_array_equal(
                np.asarray(jr.key_data(lhs)), np.asarray(jr.key_data(rhs))
            )
        else:
            np.testing.assert_allclose(np.asarray(lhs), np.asarray(rhs), rtol=2e-6, atol=2e-6)


def test_strict_config_roundtrip_exports_and_l0_resource_shape() -> None:
    config = _config(behavior_std_scale=1.7)
    payload = config.to_config()
    json.dumps(payload)
    restored = ContinuousAverageRewardActorCriticConfig.from_config(payload)
    assert restored == config
    agent = ContinuousAverageRewardActorCriticAgent(config)
    assert ContinuousAverageRewardActorCriticAgent.from_config(agent.to_config()).config == config
    assert TopLevelAgent is ContinuousAverageRewardActorCriticAgent
    assert CoreAgent is ContinuousAverageRewardActorCriticAgent

    state = agent.init(3, jr.key(1))
    budget = agent.resource_budget(3)
    assert budget.optimizer_kind == "LMS"
    assert budget.evidence_level == "L0"
    assert budget.off_policy_state_distribution_correction is False
    assert budget.replay_capacity == 0
    expected_nbytes = 0
    for leaf in jax.tree_util.tree_leaves(state):
        if jax.dtypes.issubdtype(leaf.dtype, jax.dtypes.prng_key):
            leaf = jr.key_data(leaf)
        expected_nbytes += int(np.asarray(leaf).nbytes)
    assert budget.state_nbytes == expected_nbytes
    assert budget.trainable_float32_scalars == 2 * 3 + 2 + 2 + 3 + 1 + 1

    with pytest.raises(ValueError, match="fields"):
        ContinuousAverageRewardActorCriticConfig.from_config({**payload, "extra": 1})
    with pytest.raises(ValueError):
        _config(action_low=(-1.0, 0.0), action_high=(0.0, 0.0))
    with pytest.raises(ValueError):
        _config(behavior_std_scale=0.9)
    with pytest.raises(ValueError, match="exponentiate"):
        _config(target_log_std_min=-3.0e38)
    with pytest.raises(ValueError, match="exponentiate"):
        _config(target_log_std_max=3.0e38)
    with pytest.raises(ValueError, match="half-range"):
        _config(action_low=(-3.0e38, -1.0), action_high=(3.0e38, 1.0))


def test_transformed_density_matches_change_of_variables_and_contains_jacobian() -> None:
    latent = jnp.asarray([0.35, -0.7], dtype=jnp.float32)
    mean = jnp.asarray([-0.1, 0.2], dtype=jnp.float32)
    std = jnp.asarray([0.8, 1.3], dtype=jnp.float32)
    low = jnp.asarray([-2.0, 1.0], dtype=jnp.float32)
    high = jnp.asarray([2.0, 4.0], dtype=jnp.float32)
    actual = transformed_diagonal_gaussian_log_density(latent, mean, std, low, high)

    gaussian = np.sum(
        -0.5 * ((np.asarray(latent) - np.asarray(mean)) / np.asarray(std)) ** 2
        - np.log(np.asarray(std))
        - 0.5 * np.log(2.0 * np.pi)
    )
    jacobian = np.sum(
        np.log((np.asarray(high) - np.asarray(low)) / 2.0)
        + np.log(1.0 - np.tanh(np.asarray(latent)) ** 2)
    )
    assert float(actual) == pytest.approx(float(gaussian - jacobian), rel=2e-6, abs=2e-6)
    without_jacobian = jnp.sum(
        -0.5 * ((latent - mean) / std) ** 2 - jnp.log(std) - 0.5 * jnp.log(2.0 * jnp.pi)
    )
    assert not np.isclose(float(actual), float(without_jacobian))


def test_tanh_samples_equal_direct_transform_and_latents_are_never_clipped() -> None:
    agent = ContinuousAverageRewardActorCriticAgent(
        _config(target_log_std_init=0.5, behavior_std_scale=1.2)
    )
    state = agent.init(3, jr.key(2))
    obs = jnp.asarray([0.0, 1.0, -1.0], dtype=jnp.float32)
    samples: list[jax.Array] = []
    latents: list[jax.Array] = []
    expected_actions: list[jax.Array] = []
    low = jnp.asarray(agent.config.action_low, dtype=jnp.float32)
    high = jnp.asarray(agent.config.action_high, dtype=jnp.float32)
    midpoint = 0.5 * (high + low)
    half_range = 0.5 * (high - low)
    for _ in range(512):
        sample, key = agent.sample_policy(state, obs)
        samples.append(sample.action)
        latents.append(sample.pre_tanh_action)
        expected_actions.append(agent.squash_pre_tanh_action(sample.pre_tanh_action))
        state = state.replace(rng_key=key)
    actions = jnp.stack(samples)
    pre_tanh = jnp.stack(latents)
    assert bool(jnp.all(actions >= low))
    assert bool(jnp.all(actions <= high))
    chex.assert_trees_all_equal(actions, jnp.stack(expected_actions))
    chex.assert_trees_all_close(
        actions,
        midpoint + half_range * jnp.tanh(pre_tanh),
        rtol=0.0,
        atol=3.0e-7,
    )
    affine_clipped = jnp.clip(pre_tanh, low, high)
    assert not bool(jnp.any(jnp.all(actions == affine_clipped, axis=1)))
    assert np.unique(np.asarray(actions[:, 0])).size > 500


def test_target_score_matches_finite_difference_of_transformed_log_density() -> None:
    agent = ContinuousAverageRewardActorCriticAgent(
        _config(action_dim=1, action_low=-2.0, action_high=3.0)
    )
    state = agent.init(2, jr.key(3)).replace(
        actor_params=agent.init(2, jr.key(30)).actor_params.replace(
            mean_weights=jnp.asarray([[0.2, -0.1]], dtype=jnp.float32),
            mean_bias=jnp.asarray([0.05], dtype=jnp.float32),
            log_std=jnp.asarray([-0.25], dtype=jnp.float32),
        )
    )
    obs = jnp.asarray([0.7, -0.4], dtype=jnp.float32)
    sample, _ = agent.sample_policy(state, obs)
    score = agent.target_policy_score(state, sample)
    epsilon = 2.0e-3

    def density_for_bias(bias: float) -> float:
        mean = state.actor_params.mean_weights @ obs + jnp.asarray([bias], dtype=jnp.float32)
        return float(
            transformed_diagonal_gaussian_log_density(
                sample.pre_tanh_action,
                mean,
                jnp.exp(state.actor_params.log_std),
                jnp.asarray(agent.config.action_low, dtype=jnp.float32),
                jnp.asarray(agent.config.action_high, dtype=jnp.float32),
            )
        )

    def density_for_log_std(log_std: float) -> float:
        mean = state.actor_params.mean_weights @ obs + state.actor_params.mean_bias
        return float(
            transformed_diagonal_gaussian_log_density(
                sample.pre_tanh_action,
                mean,
                jnp.exp(jnp.asarray([log_std], dtype=jnp.float32)),
                jnp.asarray(agent.config.action_low, dtype=jnp.float32),
                jnp.asarray(agent.config.action_high, dtype=jnp.float32),
            )
        )

    bias = float(state.actor_params.mean_bias[0])
    log_std = float(state.actor_params.log_std[0])
    numeric_bias = (density_for_bias(bias + epsilon) - density_for_bias(bias - epsilon)) / (
        2 * epsilon
    )
    numeric_log_std = (
        density_for_log_std(log_std + epsilon) - density_for_log_std(log_std - epsilon)
    ) / (2 * epsilon)
    assert float(score.mean_bias[0]) == pytest.approx(numeric_bias, rel=2e-3, abs=2e-3)
    assert float(score.log_std[0]) == pytest.approx(numeric_log_std, rel=2e-3, abs=2e-3)
    chex.assert_trees_all_close(score.mean_weights[0], score.mean_bias[0] * obs)


def test_on_policy_ratio_is_exactly_one_and_broader_behavior_is_declared() -> None:
    on_policy = ContinuousAverageRewardActorCriticAgent(_config(behavior_std_scale=1.0))
    on_state = on_policy.init(3, jr.key(4))
    sample, _ = on_policy.sample_policy(on_state, jnp.asarray([1.0, 0.0, 0.0], dtype=jnp.float32))
    assert float(sample.target_log_density) == float(sample.behavior_log_density)
    assert float(sample.target_behavior_ratio) == 1.0

    wider = ContinuousAverageRewardActorCriticAgent(_config(behavior_std_scale=2.0))
    wide_state = wider.init(3, jr.key(4))
    wide_sample, _ = wider.sample_policy(
        wide_state, jnp.asarray([1.0, 0.0, 0.0], dtype=jnp.float32)
    )
    expected = jnp.exp(wide_sample.target_log_density - wide_sample.behavior_log_density)
    chex.assert_trees_all_close(wide_sample.target_behavior_ratio, expected, rtol=1e-6)
    assert bool(jnp.all(wide_sample.behavior_std > wide_sample.target_std))


def test_saturated_action_ratio_is_derived_before_the_shared_jacobian() -> None:
    agent = ContinuousAverageRewardActorCriticAgent(_config(behavior_std_scale=2.0))
    state = agent.init(3, jr.key(40)).replace(
        actor_params=agent.init(3, jr.key(41)).actor_params.replace(
            mean_bias=jnp.asarray([1.0e20, -1.0e20], dtype=jnp.float32)
        )
    )
    sample, _ = agent.sample_policy(state, jnp.asarray([0.0, 0.0, 0.0], dtype=jnp.float32))
    high = jnp.asarray(agent.config.action_high, dtype=jnp.float32)
    low = jnp.asarray(agent.config.action_low, dtype=jnp.float32)
    chex.assert_trees_all_equal(sample.pre_tanh_action, sample.target_mean)
    chex.assert_trees_all_equal(
        sample.action,
        jnp.asarray([high[0], low[1]], dtype=jnp.float32),
    )
    assert float(sample.target_log_density) == float(sample.behavior_log_density)
    assert float(sample.target_behavior_ratio) == pytest.approx(2.0**2, rel=1e-6)


def test_differential_td_reward_rate_and_separate_eligibility_owners() -> None:
    agent = ContinuousAverageRewardActorCriticAgent(
        _config(
            action_dim=1,
            action_low=-1.0,
            action_high=1.0,
            actor_step_size=0.0,
            critic_step_size=0.1,
            average_reward_step_size=0.25,
            actor_trace_lambda=0.0,
            critic_trace_lambda=0.5,
        )
    )
    state = _started(
        agent,
        key=5,
        observation=jnp.asarray([2.0, -1.0, 0.5], dtype=jnp.float32),
    ).replace(
        critic_params=agent.init(3, jr.key(50)).critic_params.replace(
            weights=jnp.asarray([0.4, -0.2, 0.1], dtype=jnp.float32),
            bias=jnp.asarray(0.3, dtype=jnp.float32),
        ),
        average_reward=jnp.asarray(0.2, dtype=jnp.float32),
    )
    old_obs = state.last_sample.observation
    next_obs = jnp.asarray([-1.0, 0.25, 2.0], dtype=jnp.float32)
    old_value = float(jnp.dot(state.critic_params.weights, old_obs) + state.critic_params.bias)
    next_value = float(jnp.dot(state.critic_params.weights, next_obs) + state.critic_params.bias)
    expected_delta = 1.4 - 0.2 + next_value - old_value
    result = agent.update(state, jnp.asarray(1.4, dtype=jnp.float32), next_obs)
    assert bool(result.accepted)
    assert float(result.td_error) == pytest.approx(expected_delta, rel=1e-6)
    assert float(result.state.average_reward) == pytest.approx(
        0.2 + 0.25 * expected_delta, rel=1e-6
    )
    chex.assert_trees_all_close(result.state.critic_trace.weights, old_obs)
    chex.assert_trees_all_close(result.state.critic_trace.bias, jnp.asarray(1.0))
    assert result.state.actor_optimizer_state.mean_weights.step_size.shape == ()
    assert result.state.critic_optimizer_state.weights.step_size.shape == ()
    assert (
        result.state.actor_optimizer_state.mean_weights.step_size
        != result.state.critic_optimizer_state.weights.step_size
    )


def test_cached_decision_owns_actor_trace_not_successor_observation() -> None:
    agent = ContinuousAverageRewardActorCriticAgent(
        _config(
            action_dim=1,
            action_low=-1.0,
            action_high=1.0,
            actor_trace_lambda=0.0,
            critic_trace_lambda=0.0,
        )
    )
    state = _started(
        agent,
        key=6,
        observation=jnp.asarray([1.0, 2.0, -1.0], dtype=jnp.float32),
    )
    owned_score = agent.target_policy_score(state, state.last_sample)
    next_obs = jnp.asarray([-4.0, 0.5, 3.0], dtype=jnp.float32)
    result = agent.update(state, jnp.asarray(0.7, dtype=jnp.float32), next_obs)
    correction = state.last_sample.target_behavior_ratio
    chex.assert_trees_all_close(
        result.state.actor_trace.mean_weights,
        correction * owned_score.mean_weights,
    )
    assert not np.allclose(
        np.asarray(result.state.actor_trace.mean_weights[0]),
        np.asarray(next_obs),
    )


def test_current_exact_ratio_scales_carried_trace_and_current_score() -> None:
    agent = ContinuousAverageRewardActorCriticAgent(
        _config(
            action_dim=1,
            action_low=-1.0,
            action_high=1.0,
            behavior_std_scale=2.0,
            actor_step_size=0.0,
            critic_step_size=0.0,
            average_reward_step_size=0.0,
            actor_trace_lambda=0.6,
        )
    )
    state = _started(agent, key=61)
    first = agent.update(
        state,
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray([0.0, 1.0, -0.5], dtype=jnp.float32),
    )
    assert bool(first.accepted)
    second_score = agent.target_policy_score(first.state, first.state.last_sample)
    second_ratio = first.state.last_sample.target_behavior_ratio
    expected = second_ratio * (
        0.6 * first.state.actor_trace.mean_weights + second_score.mean_weights
    )
    second = agent.update(
        first.state,
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray([-1.0, 0.25, 0.5], dtype=jnp.float32),
    )
    assert bool(second.accepted)
    chex.assert_trees_all_close(second.state.actor_trace.mean_weights, expected)


def test_successor_is_sampled_once_from_post_commit_parameters() -> None:
    agent = ContinuousAverageRewardActorCriticAgent(
        _config(action_dim=1, action_low=-1.0, action_high=1.0, behavior_std_scale=1.0)
    )
    state = _started(agent, key=7)
    old_rng = state.rng_key
    next_obs = jnp.asarray([-0.2, 0.8, 1.1], dtype=jnp.float32)
    result = agent.update(state, jnp.asarray(2.0, dtype=jnp.float32), next_obs)
    assert bool(result.accepted)
    assert not np.array_equal(
        np.asarray(result.state.actor_params.mean_weights),
        np.asarray(state.actor_params.mean_weights),
    )
    expected_sample, expected_key = agent.sample_policy(
        result.state.replace(rng_key=old_rng), next_obs
    )
    chex.assert_trees_all_equal(result.state.last_sample, expected_sample)
    chex.assert_trees_all_equal(result.state.rng_key, expected_key)
    chex.assert_trees_all_close(
        result.state.last_sample.target_mean,
        agent.target_policy_params(result.state, next_obs)[0],
    )


def test_invalid_input_corrupt_cache_and_nonfinite_candidate_roll_back_atomically() -> None:
    agent = ContinuousAverageRewardActorCriticAgent(
        _config(action_dim=1, action_low=-1.0, action_high=1.0)
    )
    state = _started(agent, key=8)
    invalid_input = agent.update(
        state,
        jnp.asarray(jnp.nan, dtype=jnp.float32),
        jnp.asarray([0.0, 1.0, 0.0], dtype=jnp.float32),
    )
    assert not bool(invalid_input.accepted)
    _assert_bit_exact(invalid_input.state, state)

    corrupt_sample = state.last_sample.replace(action=state.last_sample.action + 0.01)
    corrupt_state = state.replace(last_sample=corrupt_sample)
    corrupt = jax.jit(agent.update)(
        corrupt_state,
        jnp.asarray(1.0, dtype=jnp.float32),
        jnp.asarray([0.0, 1.0, 0.0], dtype=jnp.float32),
    )
    assert not bool(corrupt.accepted)
    assert not bool(corrupt.diagnostics.cached_decision_valid)
    _assert_bit_exact(corrupt.state, corrupt_state)

    explosive = ContinuousAverageRewardActorCriticAgent(
        _config(
            action_dim=1,
            action_low=-1.0,
            action_high=1.0,
            actor_step_size=3.0e38,
            critic_step_size=3.0e38,
            average_reward_step_size=3.0e38,
        )
    )
    explosive_state = _started(explosive, key=80)
    rejected = explosive.update(
        explosive_state,
        jnp.asarray(3.0e38, dtype=jnp.float32),
        jnp.asarray([1.0, 1.0, 1.0], dtype=jnp.float32),
    )
    assert not bool(rejected.accepted)
    assert not bool(rejected.diagnostics.candidate_finite)
    _assert_bit_exact(rejected.state, explosive_state)


def test_transformed_density_cache_uses_the_explicit_backend_ulp_bound() -> None:
    agent = ContinuousAverageRewardActorCriticAgent(
        _config(action_dim=1, action_low=-1.0, action_high=1.0, behavior_std_scale=1.4)
    )
    base = agent.init(3, jr.key(0))
    observation = jnp.asarray([1.0, 0.0, -0.5], dtype=jnp.float32)
    low = jnp.asarray([-1.0], dtype=jnp.float32)
    high = jnp.asarray([1.0], dtype=jnp.float32)
    assert TRANSFORMED_LOG_DENSITY_MAX_ULPS == 8
    for seed in range(128):
        started = agent.start(base.replace(rng_key=jr.key(seed)), observation)
        assert bool(started.accepted)
        sample = started.state.last_sample
        reconstructed_target = transformed_diagonal_gaussian_log_density(
            sample.pre_tanh_action, sample.target_mean, sample.target_std, low, high
        )
        reconstructed_behavior = transformed_diagonal_gaussian_log_density(
            sample.pre_tanh_action, sample.target_mean, sample.behavior_std, low, high
        )
        assert (
            int(float32_ulp_distance(sample.target_log_density, reconstructed_target))
            <= TRANSFORMED_LOG_DENSITY_MAX_ULPS
        )
        assert (
            int(float32_ulp_distance(sample.behavior_log_density, reconstructed_behavior))
            <= TRANSFORMED_LOG_DENSITY_MAX_ULPS
        )
        agent.checkpoint_payload(started.state)

    sample = agent.start(base.replace(rng_key=jr.key(33)), observation).state.last_sample
    reconstructed = np.float32(
        transformed_diagonal_gaussian_log_density(
            sample.pre_tanh_action, sample.target_mean, sample.target_std, low, high
        )
    )
    tampered = reconstructed
    for _ in range(TRANSFORMED_LOG_DENSITY_MAX_ULPS + 1):
        tampered = np.nextafter(tampered, np.float32(np.inf), dtype=np.float32)
    assert (
        int(float32_ulp_distance(jnp.asarray(tampered, dtype=jnp.float32), reconstructed))
        == TRANSFORMED_LOG_DENSITY_MAX_ULPS + 1
    )
    corrupted = base.replace(
        last_sample=sample.replace(target_log_density=jnp.asarray(tampered, dtype=jnp.float32)),
        decision_count=jnp.asarray(1, dtype=jnp.int32),
    )
    rejected_density = agent.update(
        corrupted,
        jnp.asarray(0.5, dtype=jnp.float32),
        jnp.asarray([0.0, 1.0, 0.0], dtype=jnp.float32),
    )
    assert not bool(rejected_density.accepted)
    assert not bool(rejected_density.diagnostics.cached_decision_valid)
    with pytest.raises(ValueError, match="cached decision"):
        agent.checkpoint_payload(corrupted)


def test_wrong_input_and_state_shapes_fail_closed_before_arithmetic() -> None:
    agent = ContinuousAverageRewardActorCriticAgent(
        _config(action_dim=1, action_low=-1.0, action_high=1.0)
    )
    state = _started(agent, key=81)
    wrong_observation = agent.update(
        state,
        jnp.asarray(1.0, dtype=jnp.float32),
        jnp.asarray([0.0, 1.0], dtype=jnp.float32),
    )
    assert not bool(wrong_observation.accepted)
    assert not bool(wrong_observation.diagnostics.input_valid)
    _assert_bit_exact(wrong_observation.state, state)

    wrong_reward = agent.update(
        state,
        jnp.asarray([1.0], dtype=jnp.float32),
        jnp.asarray([0.0, 1.0, 0.0], dtype=jnp.float32),
    )
    assert not bool(wrong_reward.accepted)
    assert not bool(wrong_reward.diagnostics.input_valid)
    _assert_bit_exact(wrong_reward.state, state)

    malformed = state.replace(
        critic_params=state.critic_params.replace(weights=jnp.zeros((2,), dtype=jnp.float32))
    )
    rejected = agent.update(
        malformed,
        jnp.asarray(1.0, dtype=jnp.float32),
        jnp.asarray([0.0, 1.0, 0.0], dtype=jnp.float32),
    )
    assert not bool(rejected.accepted)
    assert not bool(rejected.diagnostics.state_valid)
    _assert_bit_exact(rejected.state, malformed)


def test_update_capacity_and_all_counters_saturate_without_wraparound() -> None:
    agent = ContinuousAverageRewardActorCriticAgent(
        _config(action_dim=1, action_low=-1.0, action_high=1.0, max_updates=1)
    )
    state = _started(agent, key=9)
    first = agent.update(
        state,
        jnp.asarray(0.5, dtype=jnp.float32),
        jnp.asarray([0.0, 1.0, 0.0], dtype=jnp.float32),
    )
    assert bool(first.accepted)
    assert int(first.state.update_count) == 1
    second = agent.update(
        first.state,
        jnp.asarray(0.5, dtype=jnp.float32),
        jnp.asarray([1.0, 0.0, 0.0], dtype=jnp.float32),
    )
    assert not bool(second.accepted)
    assert not bool(second.diagnostics.capacity_available)
    _assert_bit_exact(second.state, first.state)

    maximum = jnp.asarray(np.iinfo(np.int32).max, dtype=jnp.int32)
    saturated = first.state.replace(decision_count=maximum)
    restarted = agent.start(saturated, jnp.asarray([0.0, 0.0, 1.0], dtype=jnp.float32))
    assert bool(restarted.accepted)
    assert int(restarted.state.decision_count) == np.iinfo(np.int32).max


def test_typed_scalar_rng_is_required_and_legacy_state_fails_closed() -> None:
    agent = ContinuousAverageRewardActorCriticAgent(
        _config(action_dim=1, action_low=-1.0, action_high=1.0)
    )
    with pytest.raises(ValueError, match="typed scalar"):
        agent.init(3, jr.PRNGKey(12))

    state = _started(agent, key=12)
    legacy_state = state.replace(rng_key=jr.PRNGKey(13))
    result = agent.update(
        legacy_state,
        jnp.asarray(0.5, dtype=jnp.float32),
        jnp.asarray([0.0, 1.0, 0.0], dtype=jnp.float32),
    )
    assert not bool(result.accepted)
    assert not bool(result.diagnostics.state_valid)
    _assert_bit_exact(result.state, legacy_state)


def test_eager_jit_and_lax_scan_have_matching_state_rng_and_diagnostics() -> None:
    agent = ContinuousAverageRewardActorCriticAgent(
        _config(action_dim=1, action_low=-1.0, action_high=1.0, behavior_std_scale=1.3)
    )
    initial = _started(agent, key=10)
    rewards = jnp.asarray([0.2, -0.1, 1.0, 0.4], dtype=jnp.float32)
    observations = jnp.asarray(
        [
            [0.0, 1.0, 0.0],
            [0.5, 0.0, -1.0],
            [1.0, 1.0, 0.5],
            [-0.5, 0.2, 1.0],
        ],
        dtype=jnp.float32,
    )

    eager_state = initial
    eager_deltas = []
    for reward, observation in zip(rewards, observations, strict=True):
        step = agent.update(eager_state, reward, observation)
        eager_state = step.state
        eager_deltas.append(step.td_error)

    @jax.jit
    def scanned(
        state: ContinuousAverageRewardActorCriticState,
    ) -> tuple[ContinuousAverageRewardActorCriticState, jax.Array]:
        def body(
            carry: ContinuousAverageRewardActorCriticState,
            item: tuple[jax.Array, jax.Array],
        ) -> tuple[ContinuousAverageRewardActorCriticState, jax.Array]:
            result = agent.update(carry, item[0], item[1])
            return result.state, result.td_error

        return jax.lax.scan(body, state, (rewards, observations))

    scan_state, scan_deltas = scanned(initial)
    _assert_close(scan_state, eager_state)
    chex.assert_trees_all_close(scan_deltas, jnp.stack(eager_deltas), rtol=2e-6, atol=2e-6)
    one_eager = agent.update(initial, rewards[0], observations[0])
    one_jit = jax.jit(agent.update)(initial, rewards[0], observations[0])
    _assert_close(one_jit, one_eager)


def test_json_checkpoint_roundtrip_is_exact_and_rejects_tampering() -> None:
    agent = ContinuousAverageRewardActorCriticAgent(
        _config(action_dim=1, action_low=-1.0, action_high=1.0, behavior_std_scale=1.4)
    )
    state = _started(agent, key=11)
    state = agent.update(
        state,
        jnp.asarray(0.75, dtype=jnp.float32),
        jnp.asarray([0.0, -1.0, 0.5], dtype=jnp.float32),
    ).state
    payload = json.loads(json.dumps(agent.checkpoint_payload(state)))
    restored_agent, restored_state = (
        ContinuousAverageRewardActorCriticAgent.from_checkpoint_payload(payload)
    )
    assert restored_agent.config == agent.config
    _assert_bit_exact(restored_state, state)

    malformed = dict(payload)
    malformed["unexpected"] = 1
    with pytest.raises(ValueError, match="fields"):
        ContinuousAverageRewardActorCriticAgent.from_checkpoint_payload(malformed)
    dishonest = json.loads(json.dumps(payload))
    dishonest["state"]["last_sample"]["target_log_density"] += 0.25
    with pytest.raises(ValueError, match="cached decision"):
        ContinuousAverageRewardActorCriticAgent.from_checkpoint_payload(dishonest)


def test_api_never_relabels_importance_correction_as_delight_or_kondo_selection() -> None:
    sample_fields = {field.name for field in fields(SquashedGaussianPolicySample)}
    assert "target_behavior_ratio" in sample_fields
    assert all("delight" not in name and "joy" not in name for name in sample_fields)
