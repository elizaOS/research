# mypy: disable-error-code="arg-type,call-arg,no-untyped-call,no-untyped-def,type-var"
"""Analytical contracts for the bounded nonlinear differential actor-critic."""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.nonlinear_average_reward_actor_critic import (
    NONLINEAR_AVERAGE_REWARD_ACTOR_CRITIC_CHECKPOINT_SCHEMA,
    NONLINEAR_AVERAGE_REWARD_ACTOR_CRITIC_CONFIG_SCHEMA,
    NONLINEAR_AVERAGE_REWARD_ACTOR_CRITIC_RESOURCE_SCHEMA,
    NONLINEAR_AVERAGE_REWARD_ACTOR_CRITIC_STATE_SCHEMA,
    DiscreteBehaviorPolicyReceipt,
    NonlinearAverageRewardActorCritic,
    NonlinearAverageRewardActorCriticConfig,
    measure_nonlinear_average_reward_actor_critic_state_nbytes,
)

_OWNER = tuple(range(8))
_OWNER_ARRAY = jnp.asarray(_OWNER, dtype=jnp.uint32)


def _agent(**overrides: object) -> NonlinearAverageRewardActorCritic:
    values: dict[str, object] = {
        "n_actions": 2,
        "behavior_owner_digest": _OWNER,
        "hidden_size": 1,
        "objective_mode": "clipped_target_importance",
        "ordinary_behavior_epsilon": 0.2,
        "actor_head_step_size": 0.1,
        "actor_trunk_step_size": 0.03,
        "critic_head_step_size": 0.2,
        "critic_trunk_step_size": 0.04,
        "average_reward_step_size": 0.1,
        "actor_trace_decay": 0.0,
        "critic_trace_decay": 0.0,
        "momentum": 0.0,
        "importance_clip": 8.0,
        "initialization_scale": 0.2,
        "utility_decay": 0.5,
    }
    values.update(overrides)
    return NonlinearAverageRewardActorCritic(
        NonlinearAverageRewardActorCriticConfig(**values)
    )


def _witness_state(agent: NonlinearAverageRewardActorCritic, *, key_seed: int = 1):
    state = agent.init(2, jr.key(key_seed))
    return dataclasses.replace(
        state,
        actor_trunk_w=jnp.asarray([[1.0, 0.0]], dtype=jnp.float32),
        actor_trunk_b=jnp.zeros((1,), dtype=jnp.float32),
        actor_head_w=jnp.zeros((2, 1), dtype=jnp.float32),
        actor_head_b=jnp.zeros((2,), dtype=jnp.float32),
        critic_trunk_w=jnp.asarray([[1.0, 0.0]], dtype=jnp.float32),
        critic_trunk_b=jnp.zeros((1,), dtype=jnp.float32),
        critic_head_w=jnp.asarray([0.5], dtype=jnp.float32),
        critic_head_b=jnp.asarray(0.0, dtype=jnp.float32),
        average_reward=jnp.asarray(0.25, dtype=jnp.float32),
    )


def _receipt(probabilities: list[float], revision: int) -> DiscreteBehaviorPolicyReceipt:
    policy = jnp.asarray(probabilities, dtype=jnp.float32)
    return DiscreteBehaviorPolicyReceipt(
        probabilities=policy,
        log_probabilities=jnp.log(policy),
        behavior_owner_digest=_OWNER_ARRAY,
        revision_words=jnp.asarray([0, revision], dtype=jnp.uint32),
    )


def _start(
    agent: NonlinearAverageRewardActorCritic,
    state,
    behavior: DiscreteBehaviorPolicyReceipt | None = None,
):
    if behavior is None:
        behavior = _receipt([0.25, 0.75], 3)
    return agent.start(
        state,
        jnp.asarray([1.0, 0.0], dtype=jnp.float32),
        behavior,
    )


@pytest.mark.unit
def test_ordinary_behavior_score_trace_and_differential_baseline_match_hand_derivation() -> None:
    agent = _agent(
        objective_mode="ordinary_behavior",
        actor_head_step_size=0.0,
        actor_trunk_step_size=0.0,
        actor_trace_decay=0.0,
        critic_trace_decay=0.0,
        ordinary_behavior_epsilon=0.2,
    )
    state = _witness_state(agent)
    behavior = agent.ordinary_behavior_receipt(
        state,
        jnp.asarray([1.0, 0.0], dtype=jnp.float32),
        jnp.asarray([0, 3], dtype=jnp.uint32),
    )
    start = _start(agent, state, behavior)
    assert bool(start.start_applied)

    next_behavior = agent.ordinary_behavior_receipt(
        start.state,
        jnp.asarray([0.0, 0.0], dtype=jnp.float32),
        jnp.asarray([0, 4], dtype=jnp.uint32),
    )
    result = agent.update(
        start.state,
        start.record,
        jnp.asarray(1.0, dtype=jnp.float32),
        jnp.asarray([0.0, 0.0], dtype=jnp.float32),
        next_behavior,
    )

    hidden = np.tanh(1.0)
    delta = 1.0 - 0.25 - 0.5 * hidden
    action = int(start.action)
    sign = 1.0 if action == 0 else -1.0
    expected_actor_bias_trace = 0.8 * np.asarray(
        [0.5 * sign, -0.5 * sign], dtype=np.float32
    )
    assert bool(result.update_applied)
    assert float(result.actor_score_multiplier) == pytest.approx(0.8, rel=2e-6)
    assert float(result.critic_trace_multiplier) == pytest.approx(1.0)
    assert float(result.reward_rate_multiplier) == pytest.approx(1.0)
    assert float(result.td_error) == pytest.approx(delta, rel=2e-6)
    np.testing.assert_allclose(
        np.asarray(result.state.actor_head_trace_b),
        expected_actor_bias_trace,
        rtol=2e-6,
        atol=2e-7,
    )
    assert float(result.state.critic_head_trace_b) == pytest.approx(1.0)
    assert float(result.state.average_reward) == pytest.approx(0.25 + 0.1 * delta, rel=2e-6)


@pytest.mark.unit
def test_clipped_target_ratio_scales_actor_critic_and_reward_rate_consistently() -> None:
    agent = _agent(importance_clip=1.5)
    state = _witness_state(agent, key_seed=1)
    start = _start(agent, state)
    assert int(start.action) == 0
    result = agent.update(
        start.state,
        start.record,
        jnp.asarray(1.0, dtype=jnp.float32),
        jnp.asarray([0.0, 0.0], dtype=jnp.float32),
        _receipt([0.5, 0.5], 4),
    )

    hidden = np.tanh(1.0)
    delta = 1.0 - 0.25 - 0.5 * hidden
    expected_actor_bias_trace = 1.5 * np.asarray([0.5, -0.5], dtype=np.float32)
    assert bool(result.update_applied)
    assert float(result.raw_importance_ratio) == pytest.approx(2.0, rel=2e-6)
    assert float(result.clipped_importance_ratio) == pytest.approx(1.5, rel=2e-6)
    assert float(result.ratio_truncation) == pytest.approx(0.5, rel=2e-6)
    assert float(result.actor_score_multiplier) == pytest.approx(1.5, rel=2e-6)
    assert float(result.critic_trace_multiplier) == pytest.approx(1.5, rel=2e-6)
    assert float(result.reward_rate_multiplier) == pytest.approx(1.5, rel=2e-6)
    np.testing.assert_allclose(
        np.asarray(result.state.actor_head_trace_b),
        expected_actor_bias_trace,
        rtol=2e-6,
        atol=2e-7,
    )
    assert float(result.state.critic_head_trace_b) == pytest.approx(1.5, rel=2e-6)
    assert float(result.state.average_reward) == pytest.approx(
        0.25 + 0.1 * 1.5 * delta,
        rel=2e-6,
    )


@pytest.mark.unit
def test_start_logs_full_policies_selected_probabilities_revision_and_identity() -> None:
    agent = _agent()
    state = _witness_state(agent)
    behavior = _receipt([0.25, 0.75], 37)
    result = _start(agent, state, behavior)

    action = int(result.action)
    assert bool(result.start_applied)
    chex.assert_trees_all_equal(result.record.behavior_policy, behavior.probabilities)
    chex.assert_trees_all_equal(
        result.record.behavior_log_policy,
        behavior.log_probabilities,
    )
    assert float(jnp.exp(result.record.behavior_log_probability)) == pytest.approx(
        [0.25, 0.75][action],
        rel=2e-6,
    )
    assert float(jnp.exp(result.record.target_log_probability)) == pytest.approx(
        float(result.record.target_policy[action]),
        rel=2e-6,
    )
    chex.assert_trees_all_equal(
        result.record.behavior_revision_words,
        jnp.asarray([0, 37], dtype=jnp.uint32),
    )
    chex.assert_trees_all_equal(
        result.record.behavior_owner_digest,
        _OWNER_ARRAY,
    )
    chex.assert_trees_all_equal(
        result.record.action_identity_words,
        jnp.asarray([0, 1], dtype=jnp.uint32),
    )


@pytest.mark.unit
def test_ordinary_mode_rejects_policy_that_is_not_declared_epsilon_mixture() -> None:
    agent = _agent(
        objective_mode="ordinary_behavior",
        ordinary_behavior_epsilon=0.2,
    )
    state = _witness_state(agent)
    result = _start(agent, state, _receipt([0.25, 0.75], 3))

    assert bool(result.behavior_receipt_valid)
    assert not bool(result.objective_policy_valid)
    assert not bool(result.start_applied)
    chex.assert_trees_all_equal(result.state, state)


@pytest.mark.unit
def test_corrected_trace_carry_matches_two_step_equation() -> None:
    agent = _agent(
        actor_head_step_size=0.0,
        actor_trunk_step_size=0.0,
        critic_head_step_size=0.0,
        critic_trunk_step_size=0.0,
        average_reward_step_size=0.0,
        actor_trace_decay=0.5,
        critic_trace_decay=0.25,
        importance_clip=8.0,
    )
    state = _witness_state(agent, key_seed=1)
    start = _start(agent, state)
    assert int(start.action) == 0
    first = agent.update(
        start.state,
        start.record,
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray([0.0, 1.0], dtype=jnp.float32),
        _receipt([0.5, 0.5], 4),
    )
    second = agent.update(
        first.state,
        first.successor_record,
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray([0.0, 0.0], dtype=jnp.float32),
        _receipt([0.5, 0.5], 5),
    )

    first_trace = np.asarray([1.0, -1.0], dtype=np.float32)
    action = int(first.successor_action)
    current_score = (
        np.asarray([0.5, -0.5], dtype=np.float32)
        if action == 0
        else np.asarray([-0.5, 0.5], dtype=np.float32)
    )
    np.testing.assert_allclose(
        np.asarray(first.state.actor_head_trace_b),
        first_trace,
        rtol=2e-6,
        atol=2e-7,
    )
    np.testing.assert_allclose(
        np.asarray(second.state.actor_head_trace_b),
        0.5 * first_trace + current_score,
        rtol=2e-6,
        atol=2e-7,
    )
    assert float(first.state.critic_head_trace_b) == pytest.approx(2.0, rel=2e-6)
    assert float(second.state.critic_head_trace_b) == pytest.approx(1.5, rel=2e-6)


@pytest.mark.unit
def test_successor_uses_post_commit_target_and_exactly_one_owned_draw() -> None:
    agent = _agent()
    state = _witness_state(agent, key_seed=1)
    start = _start(agent, state)
    pre_successor_key = start.state.rng_key
    expected_key, expected_sample_key = jr.split(pre_successor_key)
    next_behavior = _receipt([0.1, 0.9], 4)
    expected_action = jr.categorical(
        expected_sample_key,
        next_behavior.log_probabilities,
    ).astype(jnp.int32)
    result = agent.update(
        start.state,
        start.record,
        jnp.asarray(1.0, dtype=jnp.float32),
        jnp.asarray([0.3, -0.2], dtype=jnp.float32),
        next_behavior,
    )

    assert bool(result.update_applied)
    assert bool(result.successor_sampled)
    chex.assert_trees_all_equal(result.successor_action, expected_action)
    chex.assert_trees_all_equal(jr.key_data(result.state.rng_key), jr.key_data(expected_key))
    expected_target = agent.target_policy(
        result.state,
        jnp.asarray([0.3, -0.2], dtype=jnp.float32),
    )
    np.testing.assert_allclose(
        np.asarray(result.successor_record.target_policy),
        np.asarray(expected_target),
        rtol=1e-6,
        atol=1e-7,
    )
    chex.assert_trees_all_equal(
        result.successor_record.behavior_revision_words,
        next_behavior.revision_words,
    )
    chex.assert_trees_all_equal(result.state.decision_words, jnp.asarray([0, 2], jnp.uint32))
    chex.assert_trees_all_equal(result.state.update_words, jnp.asarray([0, 1], jnp.uint32))
    chex.assert_trees_all_equal(
        result.state.target_revision_words,
        jnp.asarray([0, 1], jnp.uint32),
    )


@pytest.mark.unit
def test_zero_support_next_receipt_rolls_back_learning_and_rng_atomically() -> None:
    agent = _agent()
    state = _witness_state(agent)
    start = _start(agent, state)
    probabilities = jnp.asarray([0.0, 1.0], dtype=jnp.float32)
    invalid = DiscreteBehaviorPolicyReceipt(
        probabilities=probabilities,
        log_probabilities=jnp.asarray([-jnp.inf, 0.0], dtype=jnp.float32),
        behavior_owner_digest=_OWNER_ARRAY,
        revision_words=jnp.asarray([0, 4], dtype=jnp.uint32),
    )
    result = agent.update(
        start.state,
        start.record,
        jnp.asarray(1.0, dtype=jnp.float32),
        jnp.asarray([0.0, 0.0], dtype=jnp.float32),
        invalid,
    )

    assert not bool(result.next_behavior_receipt_valid)
    assert not bool(result.successor_sampled)
    assert not bool(result.update_applied)
    chex.assert_trees_all_equal(result.state, start.state)


@pytest.mark.unit
def test_behavior_owner_is_required_at_start_and_every_successor() -> None:
    agent = _agent()
    state = _witness_state(agent)
    wrong_owner = _OWNER_ARRAY.at[0].add(jnp.uint32(1))
    initial = _receipt([0.25, 0.75], 3)
    rejected_start = agent.start(
        state,
        jnp.asarray([1.0, 0.0], dtype=jnp.float32),
        dataclasses.replace(initial, behavior_owner_digest=wrong_owner),
    )
    assert not bool(rejected_start.behavior_owner_valid)
    assert not bool(rejected_start.start_applied)
    chex.assert_trees_all_equal(rejected_start.state, state)

    start = _start(agent, state)
    successor = dataclasses.replace(
        _receipt([0.5, 0.5], 4),
        behavior_owner_digest=wrong_owner,
    )
    result = agent.update(
        start.state,
        start.record,
        jnp.asarray(1.0, dtype=jnp.float32),
        jnp.asarray([0.0, 0.0], dtype=jnp.float32),
        successor,
    )
    assert not bool(result.next_behavior_owner_valid)
    assert not bool(result.update_applied)
    chex.assert_trees_all_equal(result.state, start.state)


@pytest.mark.unit
@pytest.mark.parametrize("next_revision", [3, 5])
def test_behavior_revision_replay_or_skip_fails_closed(next_revision: int) -> None:
    agent = _agent()
    start = _start(agent, _witness_state(agent))
    result = agent.update(
        start.state,
        start.record,
        jnp.asarray(1.0, dtype=jnp.float32),
        jnp.asarray([0.0, 0.0], dtype=jnp.float32),
        _receipt([0.5, 0.5], next_revision),
    )
    assert bool(result.behavior_revision_capacity_available)
    assert not bool(result.next_behavior_revision_valid)
    assert not bool(result.update_applied)
    chex.assert_trees_all_equal(result.state, start.state)


@pytest.mark.unit
def test_exhausted_behavior_revision_is_a_fail_stop_boundary() -> None:
    agent = _agent()
    state = _witness_state(agent)
    policy = jnp.asarray([0.25, 0.75], dtype=jnp.float32)
    behavior = DiscreteBehaviorPolicyReceipt(
        probabilities=policy,
        log_probabilities=jnp.log(policy),
        behavior_owner_digest=_OWNER_ARRAY,
        revision_words=jnp.asarray([2**32 - 1, 2**32 - 1], dtype=jnp.uint32),
    )
    start = _start(agent, state, behavior)
    assert bool(start.start_applied)
    result = agent.update(
        start.state,
        start.record,
        jnp.asarray(1.0, dtype=jnp.float32),
        jnp.asarray([0.0, 0.0], dtype=jnp.float32),
        _receipt([0.5, 0.5], 0),
    )
    assert not bool(result.behavior_revision_capacity_available)
    assert not bool(result.next_behavior_revision_valid)
    assert not bool(result.update_applied)
    chex.assert_trees_all_equal(result.state, start.state)


@pytest.mark.unit
@pytest.mark.parametrize(
    "tamper",
    [
        "action",
        "identity",
        "behavior_owner",
        "behavior_revision",
        "behavior_policy",
        "target_policy",
    ],
)
def test_cached_record_tampering_is_an_atomic_noop(tamper: str) -> None:
    agent = _agent()
    state = _witness_state(agent)
    start = _start(agent, state)
    record = start.record
    if tamper == "action":
        record = dataclasses.replace(record, action=1 - record.action)
    elif tamper == "identity":
        record = dataclasses.replace(
            record,
            action_identity_words=record.action_identity_words.at[1].add(jnp.uint32(1)),
        )
    elif tamper == "behavior_owner":
        record = dataclasses.replace(
            record,
            behavior_owner_digest=record.behavior_owner_digest.at[0].add(jnp.uint32(1)),
        )
    elif tamper == "behavior_revision":
        record = dataclasses.replace(
            record,
            behavior_revision_words=record.behavior_revision_words.at[1].add(jnp.uint32(1)),
        )
    elif tamper == "behavior_policy":
        record = dataclasses.replace(
            record,
            behavior_policy=record.behavior_policy.at[0].add(jnp.float32(0.01)),
        )
    else:
        record = dataclasses.replace(
            record,
            target_policy=record.target_policy.at[0].add(jnp.float32(0.01)),
        )
    result = agent.update(
        start.state,
        record,
        jnp.asarray(1.0, dtype=jnp.float32),
        jnp.asarray([0.0, 0.0], dtype=jnp.float32),
        _receipt([0.5, 0.5], 4),
    )
    assert not bool(result.record_identity_valid)
    assert not bool(result.update_applied)
    chex.assert_trees_all_equal(result.state, start.state)


@pytest.mark.unit
def test_silent_actor_change_invalidates_cached_target_policy() -> None:
    agent = _agent()
    start = _start(agent, _witness_state(agent))
    changed = dataclasses.replace(
        start.state,
        actor_head_b=start.state.actor_head_b.at[0].add(jnp.float32(0.1)),
    )
    result = agent.update(
        changed,
        start.record,
        jnp.asarray(1.0, dtype=jnp.float32),
        jnp.asarray([0.0, 0.0], dtype=jnp.float32),
        _receipt([0.5, 0.5], 4),
    )
    assert bool(result.record_identity_valid)
    assert not bool(result.target_policy_valid)
    assert not bool(result.update_applied)
    chex.assert_trees_all_equal(result.state, changed)


@pytest.mark.unit
def test_four_plasticity_policies_have_independent_owned_state() -> None:
    agent = _agent(
        actor_head_plasticity="frozen",
        actor_trunk_plasticity="plastic",
        critic_head_plasticity="plastic",
        critic_trunk_plasticity="frozen",
    )
    base = _witness_state(agent)
    state = dataclasses.replace(
        base,
        actor_head_w=jnp.asarray([[0.4], [-0.2]], dtype=jnp.float32),
    )
    start = _start(agent, state)
    result = agent.update(
        start.state,
        start.record,
        jnp.asarray(1.0, dtype=jnp.float32),
        jnp.asarray([0.0, 0.0], dtype=jnp.float32),
        _receipt([0.5, 0.5], 4),
    )
    assert bool(result.update_applied)
    chex.assert_trees_all_equal(result.state.actor_head_w, state.actor_head_w)
    chex.assert_trees_all_equal(result.state.actor_head_trace_w, state.actor_head_trace_w)
    assert not np.array_equal(
        np.asarray(result.state.actor_trunk_w),
        np.asarray(state.actor_trunk_w),
    )
    assert not np.array_equal(
        np.asarray(result.state.critic_head_w),
        np.asarray(state.critic_head_w),
    )
    chex.assert_trees_all_equal(result.state.critic_trunk_w, state.critic_trunk_w)
    chex.assert_trees_all_equal(
        result.state.critic_trunk_trace_w,
        state.critic_trunk_trace_w,
    )


@pytest.mark.unit
def test_component_utility_emas_are_bounded_and_monitor_frozen_components() -> None:
    agent = _agent(
        utility_decay=0.0,
        max_component_utility=0.01,
        actor_head_plasticity="frozen",
        actor_trunk_plasticity="frozen",
        critic_head_plasticity="frozen",
        critic_trunk_plasticity="frozen",
    )
    state = dataclasses.replace(
        _witness_state(agent),
        actor_head_w=jnp.asarray([[0.4], [-0.2]], dtype=jnp.float32),
    )
    start = _start(agent, state)
    result = agent.update(
        start.state,
        start.record,
        jnp.asarray(10.0, dtype=jnp.float32),
        jnp.asarray([0.0, 0.0], dtype=jnp.float32),
        _receipt([0.5, 0.5], 4),
    )
    assert bool(result.update_applied)
    utilities = np.asarray(
        [
            result.state.actor_head_utility,
            result.state.actor_trunk_utility,
            result.state.critic_head_utility,
            result.state.critic_trunk_utility,
        ]
    )
    assert np.all(utilities > 0.0)
    assert np.all(utilities <= 0.01)
    chex.assert_trees_all_equal(result.state.actor_head_w, state.actor_head_w)
    chex.assert_trees_all_equal(result.state.critic_head_w, state.critic_head_w)
    chex.assert_trees_all_equal(
        result.state.target_revision_words,
        state.target_revision_words,
    )


@pytest.mark.unit
def test_ordinary_two_phase_proposal_breaks_successor_policy_circularity() -> None:
    agent = _agent(
        objective_mode="ordinary_behavior",
        ordinary_behavior_epsilon=0.2,
    )
    state = _witness_state(agent)
    observation = jnp.asarray([1.0, 0.0], dtype=jnp.float32)
    start = _start(
        agent,
        state,
        agent.ordinary_behavior_receipt(
            state,
            observation,
            jnp.asarray([0, 3], dtype=jnp.uint32),
        ),
    )
    next_observation = jnp.asarray([0.3, -0.2], dtype=jnp.float32)
    proposal = agent.propose_update(
        start.state,
        start.record,
        jnp.asarray(1.0, dtype=jnp.float32),
        next_observation,
    )
    assert bool(proposal.proposal_valid)
    chex.assert_trees_all_equal(
        jr.key_data(proposal.candidate_state.rng_key),
        jr.key_data(start.state.rng_key),
    )
    assert not np.array_equal(
        np.asarray(proposal.next_target_policy),
        np.asarray(agent.target_policy(start.state, next_observation)),
    )
    next_behavior = agent.ordinary_successor_behavior_receipt(proposal)
    result = agent.commit_update(
        start.state,
        start.record,
        jnp.asarray(1.0, dtype=jnp.float32),
        next_observation,
        proposal,
        next_behavior,
    )
    assert bool(result.proposal_identity_valid)
    assert bool(result.next_behavior_revision_valid)
    assert bool(result.update_applied)
    chex.assert_trees_all_equal(
        result.successor_record.behavior_policy,
        next_behavior.probabilities,
    )


@pytest.mark.unit
def test_tampered_or_stale_proposal_cannot_commit() -> None:
    agent = _agent()
    start = _start(agent, _witness_state(agent))
    reward = jnp.asarray(1.0, dtype=jnp.float32)
    next_observation = jnp.asarray([0.2, -0.1], dtype=jnp.float32)
    proposal = agent.propose_update(start.state, start.record, reward, next_observation)
    tampered = dataclasses.replace(
        proposal,
        next_target_policy=proposal.next_target_policy.at[0].add(jnp.float32(0.01)),
    )
    rejected = agent.commit_update(
        start.state,
        start.record,
        reward,
        next_observation,
        tampered,
        _receipt([0.5, 0.5], 4),
    )
    assert not bool(rejected.proposal_identity_valid)
    assert not bool(rejected.update_applied)
    chex.assert_trees_all_equal(rejected.state, start.state)

    accepted = agent.commit_update(
        start.state,
        start.record,
        reward,
        next_observation,
        proposal,
        _receipt([0.5, 0.5], 4),
    )
    stale = agent.commit_update(
        accepted.state,
        accepted.successor_record,
        reward,
        next_observation,
        proposal,
        _receipt([0.5, 0.5], 5),
    )
    assert bool(accepted.update_applied)
    assert not bool(stale.proposal_identity_valid)
    assert not bool(stale.update_applied)
    chex.assert_trees_all_equal(stale.state, accepted.state)


@pytest.mark.unit
@pytest.mark.parametrize("overflow_kind", ["largest_reward", "raw_velocity"])
def test_preclip_numeric_overflow_rolls_back_with_finite_masked_diagnostics(
    overflow_kind: str,
) -> None:
    if overflow_kind == "largest_reward":
        agent = _agent()
        reward = jnp.asarray(jnp.finfo(jnp.float32).max, dtype=jnp.float32)
    else:
        agent = _agent(actor_head_step_size=float(jnp.finfo(jnp.float32).max))
        reward = jnp.asarray(10.0, dtype=jnp.float32)
    start = _start(agent, _witness_state(agent))
    result = agent.update(
        start.state,
        start.record,
        reward,
        jnp.asarray([0.0, 0.0], dtype=jnp.float32),
        _receipt([0.5, 0.5], 4),
    )
    assert not bool(result.proposal_valid)
    assert not bool(result.update_applied)
    chex.assert_trees_all_equal(result.state, start.state)
    diagnostics = np.asarray(
        [
            result.value,
            result.next_value,
            result.td_error,
            result.raw_importance_ratio,
            result.clipped_importance_ratio,
            result.actor_score_multiplier,
            result.critic_trace_multiplier,
            result.reward_rate_multiplier,
            result.ratio_truncation,
        ],
        dtype=np.float32,
    )
    assert np.all(np.isfinite(diagnostics))
    np.testing.assert_array_equal(diagnostics, np.zeros_like(diagnostics))


@pytest.mark.unit
def test_target_softmax_underflow_to_zero_fails_closed() -> None:
    agent = _agent(max_parameter_magnitude=1_000.0)
    state = dataclasses.replace(
        agent.init(2, jr.key(3)),
        actor_head_b=jnp.asarray([1_000.0, -1_000.0], dtype=jnp.float32),
    )
    assert bool(agent.state_valid(state))
    result = _start(agent, state, _receipt([0.5, 0.5], 3))
    assert not bool(result.target_support_valid)
    assert not bool(result.start_applied)
    assert bool(jnp.all(jnp.isfinite(result.target_policy)))
    chex.assert_trees_all_equal(result.state, state)


@pytest.mark.unit
def test_strict_config_checkpoint_and_resource_contracts(tmp_path: Path) -> None:
    agent = _agent()
    state = agent.init(2, jr.key(19))
    payload = agent.to_config()
    rebuilt = NonlinearAverageRewardActorCritic.from_config(payload)
    chex.assert_trees_all_equal(rebuilt.init(2, jr.key(19)), state)
    assert payload["schema"] == NONLINEAR_AVERAGE_REWARD_ACTOR_CRITIC_CONFIG_SCHEMA
    assert payload["evidence_level"] == "L0"
    assert payload["outcome_status"] == "not_assessed"
    assert payload["behavior_owner_digest"] == list(_OWNER)
    with pytest.raises(ValueError, match="manifest"):
        NonlinearAverageRewardActorCritic.from_config({**payload, "extra": 1})
    with pytest.raises(ValueError, match="float32"):
        NonlinearAverageRewardActorCriticConfig(
            n_actions=2,
            behavior_owner_digest=_OWNER,
            actor_head_step_size=1.0e100,
        )
    with pytest.raises(ValueError, match="exactly 8"):
        NonlinearAverageRewardActorCriticConfig(
            n_actions=2,
            behavior_owner_digest=(1, 2),
        )
    with pytest.raises(TypeError, match="exact tuple"):
        NonlinearAverageRewardActorCriticConfig(
            n_actions=2,
            behavior_owner_digest=list(_OWNER),
        )

    budget = agent.resource_budget(state)
    assert budget.schema == NONLINEAR_AVERAGE_REWARD_ACTOR_CRITIC_RESOURCE_SCHEMA
    assert budget.total_state_nbytes == measure_nonlinear_average_reward_actor_critic_state_nbytes(
        state
    )
    assert budget.total_state_nbytes == (
        budget.parameter_nbytes
        + budget.trace_nbytes
        + budget.optimizer_nbytes
        + budget.utility_nbytes
        + budget.pending_cache_nbytes
        + budget.clock_nbytes
        + budget.rng_nbytes
    )

    path = tmp_path / "nonlinear-average-reward.ckpt"
    agent.save_checkpoint(state, path)
    loaded = agent.load_checkpoint(state, path)
    chex.assert_trees_all_equal(loaded, state)
    metadata = agent.checkpoint_metadata(path)
    assert metadata["schema"] == NONLINEAR_AVERAGE_REWARD_ACTOR_CRITIC_CHECKPOINT_SCHEMA
    assert metadata["state_schema"] == NONLINEAR_AVERAGE_REWARD_ACTOR_CRITIC_STATE_SCHEMA


@pytest.mark.unit
@pytest.mark.parametrize("clock", ["decision_words", "update_words", "target_revision_words"])
def test_exact_uint64_clock_exhaustion_preserves_current_pending_decision(clock: str) -> None:
    agent = _agent()
    state = _witness_state(agent)
    if clock == "decision_words":
        state = dataclasses.replace(
            state,
            decision_words=jnp.asarray([2**32 - 1, 2**32 - 1], dtype=jnp.uint32),
        )
        start = _start(agent, state)
        assert not bool(start.start_applied)
        chex.assert_trees_all_equal(start.state, state)
        return

    start = _start(agent, state)
    exhausted = dataclasses.replace(
        start.state,
        **{clock: jnp.asarray([2**32 - 1, 2**32 - 1], dtype=jnp.uint32)},
    )
    if clock == "target_revision_words":
        exhausted = dataclasses.replace(
            exhausted,
            pending_target_revision_words=exhausted.target_revision_words,
        )
        record = dataclasses.replace(
            start.record,
            target_revision_words=exhausted.target_revision_words,
        )
    else:
        record = start.record
    result = agent.update(
        exhausted,
        record,
        jnp.asarray(1.0, dtype=jnp.float32),
        jnp.asarray([0.0, 0.0], dtype=jnp.float32),
        _receipt([0.5, 0.5], 4),
    )
    assert not bool(result.lifetime_capacity_available)
    assert not bool(result.update_applied)
    chex.assert_trees_all_equal(result.state, exhausted)


@pytest.mark.unit
@pytest.mark.parametrize("clock", ["decision_words", "update_words", "target_revision_words"])
def test_initial_clock_tampering_invalidates_empty_state(clock: str) -> None:
    agent = _agent()
    state = agent.init(2, jr.key(11))
    tampered = dataclasses.replace(
        state,
        **{clock: jnp.asarray([0, 1], dtype=jnp.uint32)},
    )
    assert not bool(agent.state_valid(tampered))
    result = _start(agent, tampered)
    assert not bool(result.state_valid)
    assert not bool(result.start_applied)
    chex.assert_trees_all_equal(result.state, tampered)


@pytest.mark.unit
@pytest.mark.parametrize("clock", ["decision_words", "update_words", "target_revision_words"])
def test_armed_clock_tampering_breaks_exact_coherence(clock: str) -> None:
    agent = _agent()
    start = _start(agent, _witness_state(agent))
    tampered = dataclasses.replace(
        start.state,
        **{clock: getattr(start.state, clock).at[1].add(jnp.uint32(1))},
    )
    if clock == "target_revision_words":
        tampered = dataclasses.replace(
            tampered,
            pending_target_revision_words=tampered.target_revision_words,
        )
    assert not bool(agent.state_valid(tampered))


@pytest.mark.unit
def test_frozen_actor_requires_zero_target_revision_even_after_updates() -> None:
    agent = _agent(
        actor_head_plasticity="frozen",
        actor_trunk_plasticity="frozen",
    )
    start = _start(agent, _witness_state(agent))
    result = agent.update(
        start.state,
        start.record,
        jnp.asarray(1.0, dtype=jnp.float32),
        jnp.asarray([0.0, 0.0], dtype=jnp.float32),
        _receipt([0.5, 0.5], 4),
    )
    assert bool(result.update_applied)
    chex.assert_trees_all_equal(
        result.state.target_revision_words,
        jnp.asarray([0, 0], dtype=jnp.uint32),
    )
    assert bool(agent.state_valid(result.state))
    tampered = dataclasses.replace(
        result.state,
        target_revision_words=jnp.asarray([0, 1], dtype=jnp.uint32),
        pending_target_revision_words=jnp.asarray([0, 1], dtype=jnp.uint32),
    )
    assert not bool(agent.state_valid(tampered))


@pytest.mark.unit
def test_coherent_near_maximum_internal_clocks_fail_stop_before_wrap() -> None:
    agent = _agent()
    start = _start(agent, _witness_state(agent))
    maximum = jnp.asarray([2**32 - 1, 2**32 - 1], dtype=jnp.uint32)
    predecessor = jnp.asarray([2**32 - 1, 2**32 - 2], dtype=jnp.uint32)
    near_limit = dataclasses.replace(
        start.state,
        decision_words=maximum,
        update_words=predecessor,
        target_revision_words=predecessor,
        pending_action_identity_words=maximum,
        pending_target_revision_words=predecessor,
    )
    record = dataclasses.replace(
        start.record,
        action_identity_words=maximum,
        target_revision_words=predecessor,
    )
    assert bool(agent.state_valid(near_limit))
    result = agent.update(
        near_limit,
        record,
        jnp.asarray(1.0, dtype=jnp.float32),
        jnp.asarray([0.0, 0.0], dtype=jnp.float32),
        _receipt([0.5, 0.5], 4),
    )
    assert not bool(result.lifetime_capacity_available)
    assert not bool(result.update_applied)
    chex.assert_trees_all_equal(result.state, near_limit)


@pytest.mark.unit
def test_public_boundaries_reject_wrong_key_dtype_and_behavior_log_receipt() -> None:
    agent = _agent()
    with pytest.raises(TypeError, match="Threefry"):
        agent.init(2, jr.key(0, impl="rbg"))
    state = agent.init(2, jr.key(0))
    with pytest.raises(TypeError, match="float32"):
        agent.start(
            state,
            jnp.asarray([1, 0], dtype=jnp.int32),
            _receipt([0.5, 0.5], 1),
        )
    bad_log = DiscreteBehaviorPolicyReceipt(
        probabilities=jnp.asarray([0.5, 0.5], dtype=jnp.float32),
        log_probabilities=jnp.asarray([math.log(0.4), math.log(0.6)], dtype=jnp.float32),
        behavior_owner_digest=_OWNER_ARRAY,
        revision_words=jnp.asarray([0, 1], dtype=jnp.uint32),
    )
    result = agent.start(
        state,
        jnp.asarray([1.0, 0.0], dtype=jnp.float32),
        bad_log,
    )
    assert not bool(result.behavior_receipt_valid)
    assert not bool(result.start_applied)
    chex.assert_trees_all_equal(result.state, state)


@pytest.mark.unit
def test_no_delight_channel_or_state_visitation_claim_is_encoded() -> None:
    agent = _agent()
    payload = agent.to_config()
    assert set(payload).isdisjoint({"delight", "state_visitation_correction", "sota"})
    assert jax.tree.leaves(agent.init(2, jr.key(0)))
