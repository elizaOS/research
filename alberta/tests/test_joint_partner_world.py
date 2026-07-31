"""Contract tests for the bounded joint-action outcome model."""

from __future__ import annotations

import copy
from typing import Any

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from alberta_framework.core.joint_partner_world import (
    BoundedJointOutcomeConfig,
    BoundedJointOutcomeModel,
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
    assert budget.state_nbytes == actual_nbytes
    assert budget.learned_float32_scalars_touched_per_update == 3
    assert budget.administrative_int32_scalars_touched_per_update == 2
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
