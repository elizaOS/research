# mypy: disable-error-code="call-arg,no-untyped-call,no-untyped-def"
"""Compiled and scan parity for nonlinear average-reward transactions."""

from __future__ import annotations

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from alberta_framework.core.nonlinear_average_reward_actor_critic import (
    DiscreteBehaviorPolicyReceipt,
    NonlinearAverageRewardActorCritic,
    NonlinearAverageRewardActorCriticConfig,
    run_nonlinear_average_reward_actor_critic_from_arrays,
)

_OWNER = tuple(range(8))
_OWNER_ARRAY = jnp.asarray(_OWNER, dtype=jnp.uint32)


def _key_data_tree(tree):
    return jax.tree.map(
        lambda leaf: jr.key_data(leaf)
        if str(getattr(leaf, "dtype", "")).startswith("key<")
        else leaf,
        tree,
    )


def _assert_all_close(left, right) -> None:
    chex.assert_trees_all_close(
        _key_data_tree(left),
        _key_data_tree(right),
        rtol=1e-6,
        atol=1e-7,
    )


def _agent() -> NonlinearAverageRewardActorCritic:
    return NonlinearAverageRewardActorCritic(
        NonlinearAverageRewardActorCriticConfig(
            n_actions=2,
            behavior_owner_digest=_OWNER,
            hidden_size=3,
            objective_mode="clipped_target_importance",
            actor_head_step_size=0.03,
            actor_trunk_step_size=0.01,
            critic_head_step_size=0.05,
            critic_trunk_step_size=0.02,
            average_reward_step_size=0.01,
            actor_trace_decay=0.4,
            critic_trace_decay=0.6,
            momentum=0.2,
            importance_clip=2.0,
        )
    )


def _arrays():
    observations = jnp.asarray(
        [[1.0, 0.0], [0.25, -0.5], [-0.2, 0.8], [0.4, 0.1], [0.0, 0.0]],
        dtype=jnp.float32,
    )
    behavior_probabilities = jnp.asarray(
        [[0.5, 0.5], [0.25, 0.75], [0.7, 0.3], [0.4, 0.6], [0.8, 0.2]],
        dtype=jnp.float32,
    )
    behavior_logs = jnp.log(behavior_probabilities)
    owners = jnp.broadcast_to(_OWNER_ARRAY, (5, 8))
    revisions = jnp.asarray([[0, index] for index in range(5)], dtype=jnp.uint32)
    rewards = jnp.asarray([1.0, -0.2, 0.5, 0.0], dtype=jnp.float32)
    return observations, behavior_probabilities, behavior_logs, owners, revisions, rewards


@pytest.mark.integration
def test_one_full_transaction_has_eager_jit_parity() -> None:
    agent = _agent()
    state = agent.init(2, jr.key(31))
    observation = jnp.asarray([1.0, -0.5], dtype=jnp.float32)
    behavior = jnp.asarray([0.4, 0.6], dtype=jnp.float32)
    start_receipt = DiscreteBehaviorPolicyReceipt(
        probabilities=behavior,
        log_probabilities=jnp.log(behavior),
        behavior_owner_digest=_OWNER_ARRAY,
        revision_words=jnp.asarray([0, 7], dtype=jnp.uint32),
    )
    next_behavior = jnp.asarray([0.7, 0.3], dtype=jnp.float32)
    next_receipt = DiscreteBehaviorPolicyReceipt(
        probabilities=next_behavior,
        log_probabilities=jnp.log(next_behavior),
        behavior_owner_digest=_OWNER_ARRAY,
        revision_words=jnp.asarray([0, 8], dtype=jnp.uint32),
    )

    def transaction(carry):
        started = agent.start(carry, observation, start_receipt)
        proposal = agent.propose_update(
            started.state,
            started.record,
            jnp.asarray(0.75, dtype=jnp.float32),
            jnp.asarray([0.3, 0.2], dtype=jnp.float32),
        )
        return agent.commit_update(
            started.state,
            started.record,
            jnp.asarray(0.75, dtype=jnp.float32),
            jnp.asarray([0.3, 0.2], dtype=jnp.float32),
            proposal,
            next_receipt,
        )

    eager = transaction(state)
    compiled = jax.jit(transaction)(state)
    _assert_all_close(eager, compiled)
    assert bool(eager.update_applied)


@pytest.mark.integration
def test_scan_matches_explicit_eager_start_and_update_sequence() -> None:
    agent = _agent()
    initial = agent.init(2, jr.key(41))
    arrays = _arrays()
    scanned = run_nonlinear_average_reward_actor_critic_from_arrays(
        agent,
        initial,
        *arrays,
    )

    first_receipt = DiscreteBehaviorPolicyReceipt(
        probabilities=arrays[1][0],
        log_probabilities=arrays[2][0],
        behavior_owner_digest=arrays[3][0],
        revision_words=arrays[4][0],
    )
    started = agent.start(initial, arrays[0][0], first_receipt)
    state = started.state
    actions = [started.action]
    errors = []
    averages = []
    raw_ratios = []
    clipped_ratios = []
    record = started.record
    for index in range(arrays[5].shape[0]):
        receipt = DiscreteBehaviorPolicyReceipt(
            probabilities=arrays[1][index + 1],
            log_probabilities=arrays[2][index + 1],
            behavior_owner_digest=arrays[3][index + 1],
            revision_words=arrays[4][index + 1],
        )
        result = agent.update(
            state,
            record,
            arrays[5][index],
            arrays[0][index + 1],
            receipt,
        )
        state = result.state
        record = result.successor_record
        actions.append(result.successor_action)
        errors.append(result.td_error)
        averages.append(result.post_average_reward)
        raw_ratios.append(result.raw_importance_ratio)
        clipped_ratios.append(result.clipped_importance_ratio)

    _assert_all_close(scanned.state, state)
    chex.assert_trees_all_equal(scanned.actions, jnp.stack(actions))
    chex.assert_trees_all_close(scanned.td_errors, jnp.stack(errors), rtol=1e-6, atol=1e-7)
    chex.assert_trees_all_close(
        scanned.average_rewards,
        jnp.stack(averages),
        rtol=1e-6,
        atol=1e-7,
    )
    chex.assert_trees_all_close(
        scanned.raw_importance_ratios,
        jnp.stack(raw_ratios),
        rtol=1e-6,
        atol=1e-7,
    )
    chex.assert_trees_all_close(
        scanned.clipped_importance_ratios,
        jnp.stack(clipped_ratios),
        rtol=1e-6,
        atol=1e-7,
    )
    assert bool(scanned.start_applied)
    assert bool(jnp.all(scanned.update_applied))


@pytest.mark.integration
def test_scan_itself_has_compiled_parity() -> None:
    agent = _agent()
    initial = agent.init(2, jr.key(53))
    arrays = _arrays()
    eager = run_nonlinear_average_reward_actor_critic_from_arrays(agent, initial, *arrays)
    compiled = jax.jit(
        lambda state, *xs: run_nonlinear_average_reward_actor_critic_from_arrays(
            agent,
            state,
            *xs,
        )
    )(initial, *arrays)
    _assert_all_close(eager, compiled)
