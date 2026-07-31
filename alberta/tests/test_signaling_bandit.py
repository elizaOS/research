"""Unit tests for the physically separate signaling-bandit learners."""

import dataclasses

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.signaling_bandit import (
    RoleBanditState,
    SignalingBanditAgent,
    SignalingBanditConfig,
    SignalingBanditState,
    greedy_role_action,
    role_resource_budget,
    signaling_bandit_keys,
    signaling_bandit_resource_budget,
)

pytestmark = pytest.mark.unit


def _agent_state() -> tuple[SignalingBanditAgent, SignalingBanditState]:
    agent = SignalingBanditAgent(SignalingBanditConfig(learning_rate=0.25, epsilon=0.0))
    return agent, agent.init(signaling_bandit_keys(jr.key(5)))


def test_config_validation_is_strict_and_finite() -> None:
    for value in (True, False, float("nan"), float("inf"), 0.0, -0.1, 1.1, "0.1"):
        with pytest.raises(ValueError, match="learning_rate"):
            SignalingBanditConfig(learning_rate=value)  # type: ignore[arg-type]
    for value in (True, False, float("nan"), float("inf"), -0.1, 1.1, "0.1"):
        with pytest.raises(ValueError, match="epsilon"):
            SignalingBanditConfig(epsilon=value)  # type: ignore[arg-type]
    assert SignalingBanditConfig(learning_rate=1, epsilon=0).to_dict() == {
        "learning_rate": 1.0,
        "epsilon": 0.0,
    }


def test_zero_initialization_and_role_resources_are_physically_separate() -> None:
    agent, state = _agent_state()
    del agent
    np.testing.assert_array_equal(state.helper.values, np.zeros((2, 2, 2), np.float32))
    np.testing.assert_array_equal(state.beneficiary.values, np.zeros((2, 2, 2), np.float32))
    assert not np.array_equal(jr.key_data(state.helper.key), jr.key_data(state.beneficiary.key))
    budget = signaling_bandit_resource_budget(state)
    assert budget.helper == role_resource_budget(state.helper)
    assert budget.beneficiary == role_resource_budget(state.beneficiary)
    assert budget.helper.value_scalars == 8
    assert budget.helper.key_scalars == 2
    assert budget.helper.state_scalars == 10
    assert budget.helper.state_bytes == 40
    assert budget.state_scalars == 20
    assert budget.state_bytes == 80


def test_decisions_use_only_their_old_physically_separate_tables() -> None:
    agent, initial = _agent_state()
    helper_values = initial.helper.values.at[0, 1].set(jnp.asarray((0.0, 2.0)))
    beneficiary_values = initial.beneficiary.values.at[0, 1].set(jnp.asarray((3.0, 0.0)))
    state = SignalingBanditState(
        helper=RoleBanditState(values=helper_values, key=initial.helper.key),
        beneficiary=RoleBanditState(values=beneficiary_values, key=initial.beneficiary.key),
    )
    helper = agent.select_helper(state.helper, jnp.int32(0), jnp.int32(1))
    assert int(helper.action) == 1
    # Beneficiary selection happens from the same old state, before either
    # reward update is committed, and does not read helper values or cue.
    beneficiary = agent.select_beneficiary(
        state.beneficiary,
        jnp.int32(0),
        helper.action,
    )
    assert int(beneficiary.action) == 0
    update = agent.update(state, helper, beneficiary, jnp.float32(1.0))
    assert float(update.helper_value_pre) == 2.0
    assert float(update.helper_value_post) == 1.75
    assert float(update.beneficiary_value_pre) == 3.0
    assert float(update.beneficiary_value_post) == 2.5
    untouched_helper = np.asarray(update.state.helper.values).copy()
    untouched_helper[0, 1, 1] = 2.0
    np.testing.assert_array_equal(untouched_helper, state.helper.values)


def test_false_write_mask_preserves_values_but_advances_policy_key() -> None:
    agent, state = _agent_state()
    helper = agent.select_helper(state.helper, jnp.int32(0), jnp.int32(0))
    beneficiary = agent.select_beneficiary(
        state.beneficiary,
        jnp.int32(0),
        helper.action,
    )
    update = agent.update(
        state,
        helper,
        beneficiary,
        jnp.float32(1.0),
        helper_write=False,
        beneficiary_write=True,
    )
    np.testing.assert_array_equal(update.state.helper.values, state.helper.values)
    np.testing.assert_array_equal(
        jr.key_data(update.state.helper.key),
        jr.key_data(helper.next_key),
    )
    assert not np.array_equal(jr.key_data(update.state.helper.key), jr.key_data(state.helper.key))
    assert not np.array_equal(
        jr.key_data(update.state.beneficiary.key),
        jr.key_data(state.beneficiary.key),
    )
    # A frozen zero table remains stochastic because its independent policy
    # stream advances even while its learning writes are suppressed.
    repeated = agent.select_helper(update.state.helper, jnp.int32(1), jnp.int32(1))
    assert not np.array_equal(jr.key_data(repeated.next_key), jr.key_data(helper.next_key))


def test_both_false_masks_leave_both_value_tables_bitwise_immutable() -> None:
    agent, state = _agent_state()
    helper = agent.select_helper(state.helper, jnp.int32(1), jnp.int32(0))
    beneficiary = agent.select_beneficiary(
        state.beneficiary,
        jnp.int32(1),
        helper.action,
    )
    update = agent.update(
        state,
        helper,
        beneficiary,
        jnp.float32(1.0),
        helper_write=False,
        beneficiary_write=False,
    )
    for field in dataclasses.fields(state):
        old_role = getattr(state, field.name)
        new_role = getattr(update.state, field.name)
        np.testing.assert_array_equal(new_role.values, old_role.values)
        assert not np.array_equal(jr.key_data(new_role.key), jr.key_data(old_role.key))


def test_greedy_probe_is_read_only_and_uses_fixed_tie_break() -> None:
    _, state = _agent_state()
    before_values = np.asarray(state.helper.values).copy()
    before_key = np.asarray(jr.key_data(state.helper.key)).copy()
    assert int(greedy_role_action(state.helper.values, jnp.int32(0), jnp.int32(0))) == 0
    np.testing.assert_array_equal(state.helper.values, before_values)
    np.testing.assert_array_equal(jr.key_data(state.helper.key), before_key)


def test_zero_table_ties_are_randomized_across_named_policy_keys() -> None:
    agent = SignalingBanditAgent(SignalingBanditConfig(epsilon=0.0))
    actions = {
        int(
            agent.select_helper(
                agent.init(signaling_bandit_keys(jr.key(seed))).helper,
                jnp.int32(0),
                jnp.int32(0),
            ).action
        )
        for seed in range(16)
    }
    assert actions == {0, 1}


def test_select_and_atomic_update_are_jittable_scannable_and_finite() -> None:
    agent = SignalingBanditAgent(SignalingBanditConfig(learning_rate=0.2, epsilon=0.1))
    state = agent.init(signaling_bandit_keys(jr.key(19)))

    @jax.jit
    def run(initial_state):
        def body(old_state, inputs):
            context, cue, reward = inputs
            helper = agent.select_helper(old_state.helper, context, cue)
            beneficiary = agent.select_beneficiary(
                old_state.beneficiary,
                context,
                helper.action,
            )
            update = agent.update(old_state, helper, beneficiary, reward)
            return update.state, jnp.stack(
                [update.helper_value_post, update.beneficiary_value_post]
            )

        inputs = (
            jnp.arange(32, dtype=jnp.int32) % 2,
            (jnp.arange(32, dtype=jnp.int32) // 2) % 2,
            jnp.ones((32,), dtype=jnp.float32),
        )
        return jax.lax.scan(body, initial_state, inputs)

    final_state, values = run(state)
    assert values.shape == (32, 2)
    assert bool(jnp.all(jnp.isfinite(values)))
    assert bool(jnp.all(jnp.isfinite(final_state.helper.values)))
    assert bool(jnp.all(jnp.isfinite(final_state.beneficiary.values)))
