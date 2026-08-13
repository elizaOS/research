"""Analytical contracts for the bounded nonlinear off-policy actor-critic."""

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

from alberta_framework.core.nonlinear_off_policy_actor_critic import (
    NONLINEAR_OFF_POLICY_ACTOR_CRITIC_CHECKPOINT_SCHEMA,
    NONLINEAR_OFF_POLICY_ACTOR_CRITIC_CONFIG_SCHEMA,
    NONLINEAR_OFF_POLICY_ACTOR_CRITIC_RESOURCE_SCHEMA,
    NONLINEAR_OFF_POLICY_ACTOR_CRITIC_STATE_SCHEMA,
    NonlinearOffPolicyActorCritic,
    NonlinearOffPolicyActorCriticConfig,
    measure_nonlinear_off_policy_actor_critic_state_nbytes,
)


def _agent(**overrides: object) -> NonlinearOffPolicyActorCritic:
    values: dict[str, object] = {
        "n_actions": 2,
        "hidden_size": 1,
        "actor_step_size": 0.1,
        "critic_step_size": 0.2,
        "trunk_actor_step_size": 0.03,
        "trunk_critic_step_size": 0.04,
        "actor_trace_decay": 0.0,
        "critic_trace_decay": 0.0,
        "momentum": 0.0,
        "importance_clip": 8.0,
        "initialization_scale": 0.2,
    }
    values.update(overrides)
    return NonlinearOffPolicyActorCritic(NonlinearOffPolicyActorCriticConfig(**values))


def _witness_state(agent: NonlinearOffPolicyActorCritic):
    state = agent.init(2, jr.key(7))
    return dataclasses.replace(
        state,
        trunk_w=jnp.asarray([[1.0, 0.0]], dtype=jnp.float32),
        trunk_b=jnp.zeros((1,), dtype=jnp.float32),
        actor_w=jnp.zeros((2, 1), dtype=jnp.float32),
        actor_b=jnp.zeros((2,), dtype=jnp.float32),
        critic_w=jnp.asarray([0.5], dtype=jnp.float32),
        critic_b=jnp.asarray(0.0, dtype=jnp.float32),
    )


def _cache_and_update(
    agent: NonlinearOffPolicyActorCritic,
    state,
    *,
    behavior_probability: float = 0.5,
):
    observation = jnp.asarray([1.0, 0.0], dtype=jnp.float32)
    cache = agent.cache_executed_action(
        state,
        observation,
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(math.log(behavior_probability), dtype=jnp.float32),
        jnp.asarray([4, 9], dtype=jnp.uint32),
    )
    update = agent.update(
        cache.state,
        cache.record,
        jnp.asarray(1.0, dtype=jnp.float32),
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray([0.0, 0.0], dtype=jnp.float32),
    )
    return cache, update


@pytest.mark.unit
def test_rho_one_matches_hand_derived_trace_free_update() -> None:
    agent = _agent()
    state = _witness_state(agent)
    cache, result = _cache_and_update(agent, state)

    hidden = np.tanh(1.0)
    td_error = 1.0 - 0.5 * hidden
    actor_score_w = np.asarray([[0.5 * hidden], [-0.5 * hidden]], dtype=np.float32)
    actor_score_b = np.asarray([0.5, -0.5], dtype=np.float32)
    critic_score_w = np.asarray([hidden], dtype=np.float32)
    critic_trunk_score = 0.5 * (1.0 - hidden**2)

    assert bool(cache.cache_applied)
    assert bool(result.update_applied)
    assert float(result.importance_ratio) == pytest.approx(1.0, abs=2e-7)
    assert float(result.clipped_importance_ratio) == pytest.approx(1.0, abs=2e-7)
    assert float(result.td_error) == pytest.approx(td_error, rel=2e-6)
    np.testing.assert_allclose(
        np.asarray(result.state.actor_w),
        0.1 * td_error * actor_score_w,
        rtol=2e-6,
        atol=2e-7,
    )
    np.testing.assert_allclose(
        np.asarray(result.state.actor_b),
        0.1 * td_error * actor_score_b,
        rtol=2e-6,
        atol=2e-7,
    )
    np.testing.assert_allclose(
        np.asarray(result.state.critic_w),
        np.asarray([0.5 + 0.2 * td_error * critic_score_w[0]], dtype=np.float32),
        rtol=2e-6,
        atol=2e-7,
    )
    assert float(result.state.critic_b) == pytest.approx(0.2 * td_error, rel=2e-6)
    assert float(result.state.trunk_w[0, 0]) == pytest.approx(
        1.0 + 0.04 * td_error * critic_trunk_score,
        rel=2e-6,
    )


@pytest.mark.unit
def test_behavior_mismatch_uses_explicit_clipped_per_decision_ratio() -> None:
    unit_agent = _agent(importance_clip=1.5)
    state = _witness_state(unit_agent)
    _, on_policy = _cache_and_update(unit_agent, state, behavior_probability=0.5)
    _, off_policy = _cache_and_update(unit_agent, state, behavior_probability=0.25)

    assert float(off_policy.importance_ratio) == pytest.approx(2.0, rel=2e-6)
    assert float(off_policy.clipped_importance_ratio) == pytest.approx(1.5, rel=2e-6)
    assert float(off_policy.ratio_truncation) == pytest.approx(0.5, rel=2e-6)
    assert bool(off_policy.ratio_was_clipped)
    on_delta = np.asarray(on_policy.state.actor_w) - np.asarray(state.actor_w)
    off_delta = np.asarray(off_policy.state.actor_w) - np.asarray(state.actor_w)
    np.testing.assert_allclose(off_delta, 1.5 * on_delta, rtol=2e-6, atol=2e-7)


@pytest.mark.unit
def test_importance_weighted_trace_carry_matches_two_step_equation() -> None:
    agent = _agent(
        actor_step_size=0.0,
        critic_step_size=0.0,
        trunk_actor_step_size=0.0,
        trunk_critic_step_size=0.0,
        actor_trace_decay=0.5,
        critic_trace_decay=0.25,
        importance_clip=8.0,
    )
    state = _witness_state(agent)
    first_cache = agent.cache_executed_action(
        state,
        jnp.asarray([1.0, 0.0], dtype=jnp.float32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(math.log(0.25), dtype=jnp.float32),
        jnp.asarray([0, 1], dtype=jnp.uint32),
    )
    first = agent.update(
        first_cache.state,
        first_cache.record,
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray(0.8, dtype=jnp.float32),
        jnp.asarray([0.0, 1.0], dtype=jnp.float32),
    )
    second_cache = agent.cache_executed_action(
        first.state,
        jnp.asarray([0.0, 1.0], dtype=jnp.float32),
        jnp.asarray(1, dtype=jnp.int32),
        jnp.asarray(math.log(0.5), dtype=jnp.float32),
        jnp.asarray([0, 2], dtype=jnp.uint32),
    )
    second = agent.update(
        second_cache.state,
        second_cache.record,
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray(0.6, dtype=jnp.float32),
        jnp.asarray([0.0, 0.0], dtype=jnp.float32),
    )

    # e_1 = 2 * score(action=0); e_2 = 1 * (0.6 * lambda * e_1 + score(action=1)).
    np.testing.assert_allclose(
        np.asarray(first.state.actor_head_trace_b),
        np.asarray([1.0, -1.0], dtype=np.float32),
        rtol=2e-6,
        atol=2e-7,
    )
    np.testing.assert_allclose(
        np.asarray(second.state.actor_head_trace_b),
        np.asarray([-0.2, 0.2], dtype=np.float32),
        rtol=2e-6,
        atol=2e-7,
    )
    assert float(first.state.critic_head_trace_b) == pytest.approx(2.0, rel=2e-6)
    assert float(second.state.critic_head_trace_b) == pytest.approx(1.3, rel=2e-6)


@pytest.mark.unit
def test_zero_behavior_support_fails_closed_without_consuming_decision() -> None:
    agent = _agent()
    state = _witness_state(agent)
    result = agent.cache_executed_action(
        state,
        jnp.asarray([1.0, 0.0], dtype=jnp.float32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(-jnp.inf, dtype=jnp.float32),
        jnp.asarray([0, 3], dtype=jnp.uint32),
    )

    assert not bool(result.behavior_support_valid)
    assert not bool(result.cache_applied)
    chex.assert_trees_all_equal(result.state, state)
    chex.assert_trees_all_equal(result.pre_decision_words, result.post_decision_words)


@pytest.mark.unit
@pytest.mark.parametrize(
    "tamper",
    ["action", "identity", "target_revision", "behavior_revision", "target_log_prob"],
)
def test_record_tampering_rolls_back_every_learning_array(tamper: str) -> None:
    agent = _agent()
    state = _witness_state(agent)
    cache = agent.cache_executed_action(
        state,
        jnp.asarray([1.0, 0.0], dtype=jnp.float32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(math.log(0.5), dtype=jnp.float32),
        jnp.asarray([2, 11], dtype=jnp.uint32),
    )
    record = cache.record
    if tamper == "action":
        record = dataclasses.replace(record, action=jnp.asarray(1, dtype=jnp.int32))
    elif tamper == "identity":
        record = dataclasses.replace(
            record,
            action_identity_words=record.action_identity_words.at[1].add(jnp.uint32(1)),
        )
    elif tamper == "target_revision":
        record = dataclasses.replace(
            record,
            target_revision_words=record.target_revision_words.at[1].add(jnp.uint32(1)),
        )
    elif tamper == "behavior_revision":
        record = dataclasses.replace(
            record,
            behavior_revision_words=record.behavior_revision_words.at[1].add(jnp.uint32(1)),
        )
    else:
        record = dataclasses.replace(
            record,
            target_log_probability=record.target_log_probability + jnp.float32(0.25),
        )

    result = agent.update(
        cache.state,
        record,
        jnp.asarray(1.0, dtype=jnp.float32),
        jnp.asarray(0.9, dtype=jnp.float32),
        jnp.asarray([0.2, -0.1], dtype=jnp.float32),
    )

    assert not bool(result.record_identity_valid)
    assert not bool(result.update_applied)
    chex.assert_trees_all_equal(result.state, cache.state)


@pytest.mark.unit
def test_policy_parameter_change_without_revision_invalidates_cached_target_log_prob() -> None:
    agent = _agent()
    state = _witness_state(agent)
    cache = agent.cache_executed_action(
        state,
        jnp.asarray([1.0, 0.0], dtype=jnp.float32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(math.log(0.5), dtype=jnp.float32),
        jnp.asarray([2, 11], dtype=jnp.uint32),
    )
    silently_changed = dataclasses.replace(
        cache.state,
        actor_b=cache.state.actor_b.at[0].add(jnp.float32(0.25)),
    )
    result = agent.update(
        silently_changed,
        cache.record,
        jnp.asarray(1.0, dtype=jnp.float32),
        jnp.asarray(0.9, dtype=jnp.float32),
        jnp.asarray([0.2, -0.1], dtype=jnp.float32),
    )

    assert bool(result.record_identity_valid)
    assert not bool(result.target_log_probability_valid)
    assert not bool(result.update_applied)
    chex.assert_trees_all_equal(result.state, silently_changed)


@pytest.mark.unit
def test_component_plasticity_policies_freeze_exact_owned_state() -> None:
    agent = _agent(
        actor_plasticity="frozen",
        critic_plasticity="plastic",
        trunk_plasticity="frozen",
    )
    state = _witness_state(agent)
    _, result = _cache_and_update(agent, state)

    assert bool(result.update_applied)
    chex.assert_trees_all_equal(result.state.actor_w, state.actor_w)
    chex.assert_trees_all_equal(result.state.actor_b, state.actor_b)
    chex.assert_trees_all_equal(result.state.actor_head_trace_w, state.actor_head_trace_w)
    chex.assert_trees_all_equal(result.state.actor_head_velocity_w, state.actor_head_velocity_w)
    chex.assert_trees_all_equal(result.state.trunk_w, state.trunk_w)
    chex.assert_trees_all_equal(result.state.actor_trunk_trace_w, state.actor_trunk_trace_w)
    assert not np.array_equal(np.asarray(result.state.critic_w), np.asarray(state.critic_w))


@pytest.mark.unit
def test_invalid_transition_is_an_atomic_noop() -> None:
    agent = _agent()
    state = _witness_state(agent)
    cache = agent.cache_executed_action(
        state,
        jnp.asarray([1.0, 0.0], dtype=jnp.float32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(math.log(0.5), dtype=jnp.float32),
        jnp.asarray([0, 1], dtype=jnp.uint32),
    )
    result = agent.update(
        cache.state,
        cache.record,
        jnp.asarray(jnp.nan, dtype=jnp.float32),
        jnp.asarray(0.9, dtype=jnp.float32),
        jnp.asarray([0.2, 0.3], dtype=jnp.float32),
    )

    assert not bool(result.source_valid)
    assert not bool(result.update_applied)
    chex.assert_trees_all_equal(result.state, cache.state)


@pytest.mark.unit
def test_sampling_consumes_owned_typed_threefry_key_and_binds_selected_probability() -> None:
    agent = _agent()
    state = _witness_state(agent)
    result = agent.sample_behavior_action(
        state,
        jnp.asarray([1.0, 0.0], dtype=jnp.float32),
        jnp.asarray([0.25, 0.75], dtype=jnp.float32),
        jnp.asarray([3, 5], dtype=jnp.uint32),
    )

    assert str(jr.key_impl(result.state.rng_key)) == "threefry2x32"
    assert bool(result.cache_applied)
    action = int(result.record.action)
    assert float(jnp.exp(result.record.behavior_log_probability)) == pytest.approx(
        [0.25, 0.75][action], rel=2e-6
    )
    assert not np.array_equal(
        np.asarray(jr.key_data(result.state.rng_key)),
        np.asarray(jr.key_data(state.rng_key)),
    )


@pytest.mark.unit
def test_config_resource_and_checkpoint_contracts(tmp_path: Path) -> None:
    agent = _agent()
    state = agent.init(2, jr.key(19))
    construction = agent.to_config()
    rebuilt = NonlinearOffPolicyActorCritic.from_config(construction)
    chex.assert_trees_all_equal(rebuilt.init(2, jr.key(19)), state)

    assert construction["schema"] == NONLINEAR_OFF_POLICY_ACTOR_CRITIC_CONFIG_SCHEMA
    budget = agent.resource_budget(state)
    assert budget.schema == NONLINEAR_OFF_POLICY_ACTOR_CRITIC_RESOURCE_SCHEMA
    assert budget.persistent_bytes_scope == (
        "all-persistent-state-array-leaves; excludes-host-object-overhead,"
        "temporaries,compiler-and-xla-workspaces; not-a-measured-device-peak"
    )
    assert budget.total_state_nbytes == measure_nonlinear_off_policy_actor_critic_state_nbytes(
        state
    )
    assert budget.total_state_nbytes == (
        budget.parameter_nbytes
        + budget.trace_nbytes
        + budget.optimizer_nbytes
        + budget.pending_cache_nbytes
        + budget.clock_nbytes
        + budget.rng_nbytes
    )

    path = tmp_path / "off-policy-ac.ckpt"
    agent.save_checkpoint(state, path)
    loaded = agent.load_checkpoint(state, path)
    chex.assert_trees_all_equal(loaded, state)
    metadata = agent.checkpoint_metadata(path)
    assert metadata["schema"] == NONLINEAR_OFF_POLICY_ACTOR_CRITIC_CHECKPOINT_SCHEMA
    assert metadata["state_schema"] == NONLINEAR_OFF_POLICY_ACTOR_CRITIC_STATE_SCHEMA


@pytest.mark.unit
def test_exact_clock_capacity_failure_preserves_pending_transaction() -> None:
    agent = _agent()
    state = dataclasses.replace(
        _witness_state(agent),
        decision_words=jnp.asarray([2**32 - 1, 2**32 - 1], dtype=jnp.uint32),
    )
    result = agent.cache_executed_action(
        state,
        jnp.asarray([1.0, 0.0], dtype=jnp.float32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(math.log(0.5), dtype=jnp.float32),
        jnp.asarray([0, 0], dtype=jnp.uint32),
    )

    assert not bool(result.lifetime_capacity_available)
    assert not bool(result.cache_applied)
    chex.assert_trees_all_equal(result.state, state)


@pytest.mark.unit
def test_update_and_target_revision_clocks_cross_uint32_carry_exactly() -> None:
    agent = _agent()
    state = dataclasses.replace(
        _witness_state(agent),
        update_words=jnp.asarray([0, 2**32 - 1], dtype=jnp.uint32),
        target_revision_words=jnp.asarray([0, 2**32 - 1], dtype=jnp.uint32),
    )
    _, result = _cache_and_update(agent, state)

    assert bool(result.update_applied)
    chex.assert_trees_all_equal(
        result.state.update_words,
        jnp.asarray([1, 0], dtype=jnp.uint32),
    )
    chex.assert_trees_all_equal(
        result.state.target_revision_words,
        jnp.asarray([1, 0], dtype=jnp.uint32),
    )


@pytest.mark.unit
@pytest.mark.parametrize("clock", ["update_words", "target_revision_words"])
def test_exhausted_update_or_revision_clock_preserves_pending_receipt(clock: str) -> None:
    agent = _agent()
    state = _witness_state(agent)
    cache = agent.cache_executed_action(
        state,
        jnp.asarray([1.0, 0.0], dtype=jnp.float32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(math.log(0.5), dtype=jnp.float32),
        jnp.asarray([0, 1], dtype=jnp.uint32),
    )
    exhausted = dataclasses.replace(
        cache.state,
        **{clock: jnp.asarray([2**32 - 1, 2**32 - 1], dtype=jnp.uint32)},
    )
    if clock == "target_revision_words":
        exhausted = dataclasses.replace(
            exhausted,
            pending_target_revision_words=exhausted.target_revision_words,
        )
        record = dataclasses.replace(
            cache.record,
            target_revision_words=exhausted.target_revision_words,
        )
    else:
        record = cache.record

    result = agent.update(
        exhausted,
        record,
        jnp.asarray(1.0, dtype=jnp.float32),
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray([0.0, 0.0], dtype=jnp.float32),
    )

    assert not bool(result.lifetime_capacity_available)
    assert not bool(result.update_applied)
    chex.assert_trees_all_equal(result.state, exhausted)


@pytest.mark.unit
def test_public_boundaries_reject_wrong_key_and_array_contracts() -> None:
    agent = _agent()
    with pytest.raises(TypeError, match="Threefry"):
        agent.init(2, jr.key(0, impl="rbg"))
    state = agent.init(2, jr.key(0))
    with pytest.raises(TypeError, match="float32"):
        agent.cache_executed_action(
            state,
            jnp.asarray([1, 0], dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(math.log(0.5), dtype=jnp.float32),
            jnp.asarray([0, 0], dtype=jnp.uint32),
        )


@pytest.mark.unit
def test_config_manifest_is_strict_and_no_claim_is_encoded() -> None:
    agent = _agent()
    config = agent.to_config()
    assert config["evidence_level"] == "L0"
    assert config["outcome_status"] == "not_assessed"
    with pytest.raises(ValueError, match="manifest"):
        NonlinearOffPolicyActorCritic.from_config({**config, "unknown": 1})
    assert jax.tree.leaves(agent.init(2, jr.key(0)))
