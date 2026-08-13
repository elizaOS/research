"""Compiled and scan parity for nonlinear off-policy actor-critic transactions."""

from __future__ import annotations

import math

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from alberta_framework.core.nonlinear_off_policy_actor_critic import (
    NonlinearOffPolicyActorCritic,
    NonlinearOffPolicyActorCriticConfig,
    run_nonlinear_off_policy_actor_critic_from_arrays,
)


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


def _agent() -> NonlinearOffPolicyActorCritic:
    return NonlinearOffPolicyActorCritic(
        NonlinearOffPolicyActorCriticConfig(
            n_actions=2,
            hidden_size=3,
            actor_step_size=0.03,
            critic_step_size=0.05,
            trunk_actor_step_size=0.01,
            trunk_critic_step_size=0.02,
            actor_trace_decay=0.4,
            critic_trace_decay=0.6,
            momentum=0.2,
            importance_clip=2.0,
        )
    )


def _arrays():
    observations = jnp.asarray(
        [[1.0, 0.0], [0.25, -0.5], [-0.2, 0.8], [0.4, 0.1]], dtype=jnp.float32
    )
    next_observations = jnp.asarray(
        [[0.25, -0.5], [-0.2, 0.8], [0.4, 0.1], [0.0, 0.0]], dtype=jnp.float32
    )
    actions = jnp.asarray([0, 1, 1, 0], dtype=jnp.int32)
    behavior_log_probabilities = jnp.asarray(
        [math.log(0.5), math.log(0.25), math.log(0.75), math.log(0.4)],
        dtype=jnp.float32,
    )
    behavior_revisions = jnp.asarray([[0, i] for i in range(4)], dtype=jnp.uint32)
    rewards = jnp.asarray([1.0, -0.2, 0.5, 0.0], dtype=jnp.float32)
    discounts = jnp.asarray([0.9, 0.8, 0.95, 0.0], dtype=jnp.float32)
    return (
        observations,
        actions,
        behavior_log_probabilities,
        behavior_revisions,
        rewards,
        discounts,
        next_observations,
    )


@pytest.mark.integration
def test_one_transaction_has_eager_jit_parity() -> None:
    agent = _agent()
    state = agent.init(2, jr.key(31))
    observation = jnp.asarray([1.0, -0.5], dtype=jnp.float32)
    behavior_revision = jnp.asarray([0, 7], dtype=jnp.uint32)

    def transaction(carry):
        cache = agent.cache_executed_action(
            carry,
            observation,
            jnp.asarray(1, dtype=jnp.int32),
            jnp.asarray(math.log(0.4), dtype=jnp.float32),
            behavior_revision,
        )
        return agent.update(
            cache.state,
            cache.record,
            jnp.asarray(0.75, dtype=jnp.float32),
            jnp.asarray(0.9, dtype=jnp.float32),
            jnp.asarray([0.3, 0.2], dtype=jnp.float32),
        )

    eager = transaction(state)
    compiled = jax.jit(transaction)(state)
    _assert_all_close(eager, compiled)


@pytest.mark.integration
def test_scan_matches_explicit_eager_transaction_sequence() -> None:
    agent = _agent()
    initial = agent.init(2, jr.key(41))
    arrays = _arrays()
    scanned = run_nonlinear_off_policy_actor_critic_from_arrays(agent, initial, *arrays)

    state = initial
    eager_ratios = []
    eager_clipped = []
    eager_errors = []
    for index in range(arrays[0].shape[0]):
        cache = agent.cache_executed_action(
            state,
            arrays[0][index],
            arrays[1][index],
            arrays[2][index],
            arrays[3][index],
        )
        result = agent.update(
            cache.state,
            cache.record,
            arrays[4][index],
            arrays[5][index],
            arrays[6][index],
        )
        state = result.state
        eager_ratios.append(result.importance_ratio)
        eager_clipped.append(result.clipped_importance_ratio)
        eager_errors.append(result.td_error)

    _assert_all_close(scanned.state, state)
    chex.assert_trees_all_close(
        scanned.importance_ratios,
        jnp.stack(eager_ratios),
        rtol=1e-6,
        atol=1e-7,
    )
    chex.assert_trees_all_close(
        scanned.clipped_importance_ratios,
        jnp.stack(eager_clipped),
        rtol=1e-6,
        atol=1e-7,
    )
    chex.assert_trees_all_close(
        scanned.td_errors,
        jnp.stack(eager_errors),
        rtol=1e-6,
        atol=1e-7,
    )
    assert bool(jnp.all(scanned.cache_applied))
    assert bool(jnp.all(scanned.update_applied))


@pytest.mark.integration
def test_scan_itself_has_compiled_parity() -> None:
    agent = _agent()
    initial = agent.init(2, jr.key(53))
    arrays = _arrays()

    eager = run_nonlinear_off_policy_actor_critic_from_arrays(agent, initial, *arrays)
    compiled = jax.jit(
        lambda state, *xs: run_nonlinear_off_policy_actor_critic_from_arrays(
            agent, state, *xs
        )
    )(initial, *arrays)
    _assert_all_close(eager, compiled)
