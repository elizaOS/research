"""Contract tests for the bounded joint-action outcome model."""

from __future__ import annotations

import copy
import dataclasses
from typing import Any

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from alberta_framework.core.joint_partner_world import (
    JOINT_OUTCOME_STATE_SCHEMA,
    BoundedJointOutcomeConfig,
    BoundedJointOutcomeModel,
    joint_outcome_lifetime_counter_nbytes,
    measure_bounded_joint_outcome_state_nbytes,
    migrate_legacy_bounded_joint_outcome_state,
)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("n_actions", True),
        ("n_actions", 1),
        ("outcome_dim", True),
        ("outcome_dim", 0),
        ("step_size", float("nan")),
        ("step_size", 0.0),
        ("step_size", 1.01),
        ("reward_bound", float("inf")),
        ("outcome_bound", float("nan")),
        ("probability_tolerance", 0.0),
    ],
)
def test_config_rejects_invalid_values(field: str, value: Any) -> None:
    kwargs: dict[str, Any] = {field: value}
    with pytest.raises(ValueError):
        BoundedJointOutcomeConfig(**kwargs)


def test_config_and_model_roundtrip_are_strict() -> None:
    model = BoundedJointOutcomeModel(
        BoundedJointOutcomeConfig(
            n_actions=3,
            outcome_dim=2,
            step_size=0.4,
            reward_bound=2.0,
            outcome_bound=3.0,
            probability_tolerance=1e-4,
        )
    )
    restored = BoundedJointOutcomeModel.from_config(model.to_config())
    assert restored.to_config() == model.to_config()
    assert restored.to_config()["state_schema"] == JOINT_OUTCOME_STATE_SCHEMA

    changed = copy.deepcopy(model.to_config())
    changed["unexpected"] = 1
    with pytest.raises(ValueError, match="exactly"):
        BoundedJointOutcomeModel.from_config(changed)

    nested = copy.deepcopy(model.to_config())
    nested["config"]["unexpected"] = 1
    with pytest.raises(ValueError, match="schema"):
        BoundedJointOutcomeModel.from_config(nested)


def test_resource_budget_matches_state_arrays_exactly() -> None:
    model = BoundedJointOutcomeModel(BoundedJointOutcomeConfig(n_actions=3, outcome_dim=2))
    state = model.init()
    budget = model.resource_budget
    actual_nbytes = sum(int(leaf.nbytes) for leaf in jax.tree_util.tree_leaves(state))

    assert budget.joint_cells == 9
    assert budget.allocated_float32_scalars == 9 * 3
    assert budget.allocated_int32_scalars == 10
    assert budget.allocated_uint32_scalars == 2
    assert budget.state_nbytes == actual_nbytes
    assert budget.state_nbytes == measure_bounded_joint_outcome_state_nbytes(state)
    assert joint_outcome_lifetime_counter_nbytes() == 12
    assert budget.learned_float32_scalars_touched_per_update == 3
    assert budget.administrative_int32_scalars_touched_per_update == 2
    assert budget.administrative_uint32_scalars_touched_per_update == 2
    assert budget.planner_cell_evaluations_per_decision == 9
    assert budget.replay_capacity == 0


def test_update_is_predict_before_update_and_touches_only_executed_cell() -> None:
    model = BoundedJointOutcomeModel(
        BoundedJointOutcomeConfig(
            n_actions=2,
            outcome_dim=2,
            step_size=0.5,
        )
    )
    state = model.init().replace(
        reward_predictions=jnp.array(
            [[0.1, 0.2], [0.3, 0.4]],
            dtype=jnp.float32,
        ),
        outcome_predictions=jnp.array(
            [
                [[0.0, 0.1], [0.2, 0.3]],
                [[0.4, 0.5], [0.6, 0.7]],
            ],
            dtype=jnp.float32,
        ),
    )
    result = model.update(
        state,
        jnp.asarray(1, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.9, dtype=jnp.float32),
        jnp.asarray([0.8, -0.5], dtype=jnp.float32),
    )

    chex.assert_trees_all_close(result.prediction.reward, 0.3)
    chex.assert_trees_all_close(result.prediction.outcome, jnp.array([0.4, 0.5]))
    chex.assert_trees_all_close(result.reward_error, 0.6)
    chex.assert_trees_all_close(
        result.outcome_error,
        jnp.array([0.4, -1.0], dtype=jnp.float32),
    )
    assert bool(result.target_valid)
    assert int(result.prediction.visit_count) == 0
    assert int(result.visit_count_after) == 1
    assert int(result.state.step_count) == 1
    chex.assert_trees_all_equal(
        result.state.step_words,
        jnp.asarray((0, 1), dtype=jnp.uint32),
    )
    assert bool(result.update_applied)

    expected_reward = np.asarray(state.reward_predictions).copy()
    expected_reward[1, 0] = 0.6
    np.testing.assert_allclose(result.state.reward_predictions, expected_reward)
    expected_outcome = np.asarray(state.outcome_predictions).copy()
    expected_outcome[1, 0] = [0.6, 0.0]
    np.testing.assert_allclose(result.state.outcome_predictions, expected_outcome)
    expected_visits = np.zeros((2, 2), dtype=np.int32)
    expected_visits[1, 0] = 1
    np.testing.assert_array_equal(result.state.visit_counts, expected_visits)


def test_external_partner_marginalization_uses_every_joint_cell() -> None:
    model = BoundedJointOutcomeModel(BoundedJointOutcomeConfig(n_actions=2, outcome_dim=1))
    state = model.init().replace(
        reward_predictions=jnp.array(
            [[0.0, 1.0], [0.8, 0.2]],
            dtype=jnp.float32,
        ),
        outcome_predictions=jnp.array(
            [[[-1.0], [1.0]], [[0.5], [-0.5]]],
            dtype=jnp.float32,
        ),
    )

    result = model.marginalize(
        state,
        jnp.array([0.75, 0.25], dtype=jnp.float32),
    )

    assert bool(result.partner_probabilities_valid)
    chex.assert_trees_all_close(result.expected_rewards, jnp.array([0.25, 0.65]))
    chex.assert_trees_all_close(
        result.expected_outcomes,
        jnp.array([[-0.5], [0.25]], dtype=jnp.float32),
    )
    assert int(result.greedy_action) == 1
    assert int(result.cell_evaluations) == 4


@pytest.mark.parametrize(
    "probabilities",
    [
        jnp.array([-0.1, 1.1], dtype=jnp.float32),
        jnp.array([0.2, 0.2], dtype=jnp.float32),
        jnp.array([jnp.nan, 1.0], dtype=jnp.float32),
    ],
)
def test_invalid_partner_probabilities_are_reported(
    probabilities: jax.Array,
) -> None:
    model = BoundedJointOutcomeModel(BoundedJointOutcomeConfig())
    result = model.marginalize(model.init(), probabilities)
    assert not bool(result.partner_probabilities_valid)
    assert int(result.cell_evaluations) == 4


def test_invalid_target_is_bounded_but_fails_validity_diagnostic() -> None:
    model = BoundedJointOutcomeModel(
        BoundedJointOutcomeConfig(
            outcome_dim=2,
            step_size=1.0,
            reward_bound=1.0,
            outcome_bound=2.0,
        )
    )
    result = model.update(
        model.init(),
        jnp.asarray(0),
        jnp.asarray(1),
        jnp.asarray(float("inf")),
        jnp.asarray([float("nan"), 9.0]),
    )

    assert not bool(result.target_valid)
    chex.assert_tree_all_finite(result.state)
    chex.assert_trees_all_close(result.state.reward_predictions[0, 1], 1.0)
    chex.assert_trees_all_close(
        result.state.outcome_predictions[0, 1],
        jnp.array([0.0, 2.0], dtype=jnp.float32),
    )


def test_online_table_learns_all_joint_cells_without_replay() -> None:
    model = BoundedJointOutcomeModel(
        BoundedJointOutcomeConfig(
            n_actions=2,
            outcome_dim=1,
            step_size=0.2,
        )
    )
    truth_reward = jnp.array([[0.0, 1.0], [1.0, 0.0]], dtype=jnp.float32)
    truth_outcome = jnp.array([[[-1.0], [1.0]], [[1.0], [-1.0]]], dtype=jnp.float32)
    own = jnp.tile(jnp.array([0, 0, 1, 1], dtype=jnp.int32), 100)
    partner = jnp.tile(jnp.array([0, 1, 0, 1], dtype=jnp.int32), 100)
    rewards = truth_reward[own, partner]
    outcomes = truth_outcome[own, partner]

    def step(state, values):
        own_action, partner_action, reward, outcome = values
        result = model.update(
            state,
            own_action,
            partner_action,
            reward,
            outcome,
        )
        return result.state, result.prediction.reward

    final_state, prequential = jax.lax.scan(
        step,
        model.init(),
        (own, partner, rewards, outcomes),
    )

    assert prequential.shape == (400,)
    np.testing.assert_allclose(final_state.reward_predictions, truth_reward, atol=1e-6)
    np.testing.assert_allclose(final_state.outcome_predictions, truth_outcome, atol=1e-6)
    np.testing.assert_array_equal(
        final_state.visit_counts,
        np.full((2, 2), 100, dtype=np.int32),
    )
    assert int(final_state.step_count) == 400


def test_update_and_marginalization_are_jit_vmap_safe() -> None:
    model = BoundedJointOutcomeModel(BoundedJointOutcomeConfig(n_actions=2, outcome_dim=1))

    def one(reward):
        update = model.update(
            model.init(),
            jnp.asarray(1),
            jnp.asarray(0),
            reward,
            jnp.asarray([reward]),
        )
        decision = model.marginalize(
            update.state,
            jnp.asarray([0.6, 0.4]),
        )
        return update.state, decision

    states, decisions = jax.jit(jax.vmap(one))(jnp.asarray([0.2, 0.4, 0.6]))
    assert states.reward_predictions.shape == (3, 2, 2)
    assert decisions.expected_rewards.shape == (3, 2)
    assert bool(jnp.all(decisions.partner_probabilities_valid))
    chex.assert_tree_all_finite((states, decisions))


def test_exact_lifetime_clock_carries_and_refuses_corruption_or_exhaustion() -> None:
    model = BoundedJointOutcomeModel(BoundedJointOutcomeConfig())
    initial = model.init()
    args = (
        jnp.asarray(1, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.75, dtype=jnp.float32),
        jnp.asarray((1.0,), dtype=jnp.float32),
    )
    near_carry = initial.replace(
        step_count=jnp.asarray(2**31 - 1, dtype=jnp.int32),
        step_words=jnp.asarray((0, 2**32 - 1), dtype=jnp.uint32),
        visit_counts=initial.visit_counts.at[1, 0].set(2**31 - 1),
    )
    carried = jax.jit(model.update)(near_carry, *args)
    assert bool(carried.lifetime_counter_valid)
    assert bool(carried.lifetime_capacity_available)
    assert bool(carried.update_applied)
    chex.assert_trees_all_equal(
        carried.post_step_words,
        jnp.asarray((1, 0), dtype=jnp.uint32),
    )
    assert int(carried.state.step_count) == 2**31 - 1
    assert int(carried.state.visit_counts[1, 0]) == 2**31 - 1

    def scan_step(state, _):
        result = model.update(state, *args)
        return result.state, (result.update_applied, result.post_step_words)

    scanned, (applied, words) = jax.lax.scan(
        scan_step,
        near_carry,
        jnp.arange(2, dtype=jnp.int32),
    )
    chex.assert_trees_all_equal(applied, jnp.asarray((True, True), dtype=jnp.bool_))
    chex.assert_trees_all_equal(
        words,
        jnp.asarray(((1, 0), (1, 1)), dtype=jnp.uint32),
    )
    chex.assert_trees_all_equal(scanned.step_words, jnp.asarray((1, 1), dtype=jnp.uint32))

    exhausted = carried.state.replace(
        step_words=jnp.full((2,), 2**32 - 1, dtype=jnp.uint32),
    )
    stopped = model.update(exhausted, *args)
    assert bool(stopped.lifetime_counter_valid)
    assert not bool(stopped.lifetime_capacity_available)
    assert not bool(stopped.update_applied)
    chex.assert_trees_all_equal(stopped.state, exhausted)

    misaligned = initial.replace(step_count=jnp.asarray(1, dtype=jnp.int32))
    rejected = model.update(misaligned, *args)
    assert not bool(rejected.lifetime_counter_valid)
    assert not bool(rejected.update_applied)
    chex.assert_trees_all_equal(rejected.state, misaligned)

    negative_visit = initial.replace(
        visit_counts=initial.visit_counts.at[0, 0].set(-1)
    )
    rejected_visit = model.update(negative_visit, *args)
    assert not bool(rejected_visit.lifetime_counter_valid)
    assert not bool(rejected_visit.update_applied)
    chex.assert_trees_all_equal(rejected_visit.state, negative_visit)


def test_legacy_joint_outcome_clock_migration_checks_counter_alignment() -> None:
    model = BoundedJointOutcomeModel(BoundedJointOutcomeConfig())
    state = model.init()
    legacy = {
        field.name: getattr(state, field.name)
        for field in dataclasses.fields(type(state))
        if field.name != "step_words"
    }
    legacy["step_count"] = jnp.asarray(3, dtype=jnp.int32)
    legacy["visit_counts"] = jnp.asarray(((1, 0), (0, 2)), dtype=jnp.int32)
    migrated = migrate_legacy_bounded_joint_outcome_state(legacy)
    chex.assert_trees_all_equal(
        migrated.step_words,
        jnp.asarray((0, 3), dtype=jnp.uint32),
    )

    misaligned = dict(legacy)
    misaligned["visit_counts"] = jnp.asarray(((1, 0), (0, 1)), dtype=jnp.int32)
    with pytest.raises(ValueError, match="not aligned"):
        migrate_legacy_bounded_joint_outcome_state(misaligned)

    saturated = dict(legacy)
    saturated["step_count"] = jnp.asarray(2**31 - 1, dtype=jnp.int32)
    with pytest.raises(ValueError, match="ambiguous"):
        migrate_legacy_bounded_joint_outcome_state(saturated)
