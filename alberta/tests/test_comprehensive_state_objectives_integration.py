"""Integration contracts for comprehensive WP3 learned-state objectives."""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.comprehensive_state_objectives import (
    ComprehensiveStateObjectives,
    ComprehensiveStateObjectivesConfig,
    ComprehensiveStateObjectivesScanResult,
    ComprehensiveStateObjectivesState,
    load_comprehensive_state_objectives_checkpoint,
    run_comprehensive_state_objectives_scan,
    save_comprehensive_state_objectives_checkpoint,
)

pytestmark = pytest.mark.integration


def _objectives() -> ComprehensiveStateObjectives:
    return ComprehensiveStateObjectives(
        ComprehensiveStateObjectivesConfig(
            representation_dim=3,
            observation_target_dim=2,
            n_actions=2,
            gvf_discounts=(0.0, 0.5, 0.9),
            observation_step_size=0.02,
            latent_step_size=0.03,
            reward_step_size=0.04,
            termination_step_size=0.05,
            gvf_step_size=0.03,
            value_step_size=0.02,
            advantage_step_size=0.04,
            inverse_step_size=0.03,
            initialization_scale=0.08,
            representation_gradient_clip=10.0,
        )
    )


def _trajectory() -> tuple[jax.Array, ...]:
    current = jnp.asarray(
        [[0.2, -0.1, 0.5], [0.4, 0.3, -0.2], [-0.2, 0.6, 0.1], [0.1, 0.2, 0.7]],
        dtype=jnp.float32,
    )
    successor = jnp.asarray(
        [[0.4, 0.3, -0.2], [-0.2, 0.6, 0.1], [0.1, 0.2, 0.7], [0.3, -0.4, 0.2]],
        dtype=jnp.float32,
    )
    actions = jnp.asarray([0, 1, 1, 0], dtype=jnp.int32)
    observations = jnp.asarray(
        [[0.1, -0.2], [0.5, 0.4], [-0.3, 0.7], [0.2, 0.0]], dtype=jnp.float32
    )
    rewards = jnp.asarray([0.1, -0.2, 0.3, 0.0], dtype=jnp.float32)
    terminated = jnp.asarray([False, False, True, False], dtype=jnp.bool_)
    cumulants = jnp.asarray([0.2, -0.1, 0.4, 0.1], dtype=jnp.float32)
    continuations = jnp.asarray([1.0, 1.0, 0.0, 1.0], dtype=jnp.float32)
    values = jnp.asarray([0.3, -0.1, 0.0, 0.2], dtype=jnp.float32)
    advantages = jnp.asarray([0.2, -0.4, 0.1, 0.3], dtype=jnp.float32)
    current_revisions = jnp.asarray([[0, 0], [0, 0], [0, 1], [0, 1]], dtype=jnp.uint32)
    next_revisions = jnp.asarray([[0, 0], [0, 1], [0, 1], [0, 2]], dtype=jnp.uint32)
    return (
        current,
        successor,
        actions,
        observations,
        rewards,
        terminated,
        cumulants,
        continuations,
        values,
        advantages,
        current_revisions,
        next_revisions,
    )


def _assert_tree_allclose(left: object, right: object) -> None:
    left_leaves, left_tree = jax.tree.flatten(left)
    right_leaves, right_tree = jax.tree.flatten(right)
    assert str(left_tree) == str(right_tree)
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        if np.issubdtype(np.asarray(left_leaf).dtype, np.inexact):
            np.testing.assert_allclose(
                np.asarray(left_leaf), np.asarray(right_leaf), rtol=1e-6, atol=1e-7
            )
        else:
            np.testing.assert_array_equal(np.asarray(left_leaf), np.asarray(right_leaf))


def _run_scan(
    objectives: ComprehensiveStateObjectives,
    state: ComprehensiveStateObjectivesState,
    trajectory: tuple[jax.Array, ...],
) -> ComprehensiveStateObjectivesScanResult:
    return run_comprehensive_state_objectives_scan(objectives, state, *trajectory)


def test_eager_jit_scan_and_repeated_single_step_are_equivalent() -> None:
    objectives = _objectives()
    initial = objectives.init(jr.key(20))
    trajectory = _trajectory()
    with jax.disable_jit():
        eager = _run_scan(objectives, initial, trajectory)
    compiled = jax.jit(_run_scan, static_argnums=(0,))(objectives, initial, trajectory)
    _assert_tree_allclose(eager, compiled)
    assert bool(jnp.all(compiled.cache_applied))
    assert bool(jnp.all(compiled.update_applied))
    np.testing.assert_array_equal(
        compiled.action_identity_words,
        [[0, 1], [0, 2], [0, 3], [0, 4]],
    )

    state = initial
    losses = []
    current_gradients = []
    next_gradients = []
    for values in zip(*trajectory, strict=True):
        (
            current,
            successor,
            action,
            observation,
            reward,
            terminated,
            cumulant,
            continuation,
            value_target,
            advantage_target,
            current_revision,
            next_revision,
        ) = values
        cached = objectives.cache_action(state, current, action, current_revision)
        updated = objectives.update(
            cached.state,
            cached.receipt,
            successor,
            next_revision,
            observation,
            reward,
            terminated,
            cumulant,
            continuation,
            value_target,
            advantage_target,
        )
        state = updated.state
        losses.append(updated.balanced_loss)
        current_gradients.append(updated.current_representation_gradient)
        next_gradients.append(updated.next_representation_gradient)
    _assert_tree_allclose(state, compiled.state)
    np.testing.assert_allclose(jnp.stack(losses), compiled.balanced_losses, rtol=1e-6)
    np.testing.assert_allclose(
        jnp.stack(current_gradients), compiled.current_representation_gradients, rtol=1e-6
    )
    np.testing.assert_allclose(
        jnp.stack(next_gradients), compiled.next_representation_gradients, rtol=1e-6
    )


def test_checkpoint_resume_with_pending_owner_matches_uninterrupted(tmp_path: Path) -> None:
    objectives = _objectives()
    trajectory = _trajectory()
    initial = objectives.init(jr.key(21))
    prefix = _run_scan(objectives, initial, tuple(values[:2] for values in trajectory))
    values = tuple(value[2] for value in trajectory)
    (
        current,
        successor,
        action,
        observation,
        reward,
        terminated,
        cumulant,
        continuation,
        value_target,
        advantage_target,
        current_revision,
        next_revision,
    ) = values
    pending = objectives.cache_action(prefix.state, current, action, current_revision)
    assert bool(pending.cache_applied)

    checkpoint = tmp_path / "comprehensive-state-objectives"
    save_comprehensive_state_objectives_checkpoint(objectives, pending.state, checkpoint)
    restored_objectives, restored_state = load_comprehensive_state_objectives_checkpoint(
        checkpoint
    )
    assert restored_objectives.to_config() == objectives.to_config()
    _assert_tree_allclose(restored_state, pending.state)

    resumed = restored_objectives.update(
        restored_state,
        pending.receipt,
        successor,
        next_revision,
        observation,
        reward,
        terminated,
        cumulant,
        continuation,
        value_target,
        advantage_target,
    )
    assert bool(resumed.update_applied)
    resumed_tail = _run_scan(
        restored_objectives,
        resumed.state,
        tuple(values[3:] for values in trajectory),
    )
    uninterrupted = _run_scan(objectives, initial, trajectory)
    _assert_tree_allclose(resumed_tail.state, uninterrupted.state)
    np.testing.assert_allclose(
        jnp.concatenate(
            [
                prefix.balanced_losses,
                jnp.atleast_1d(resumed.balanced_loss),
                resumed_tail.balanced_losses,
            ]
        ),
        uninterrupted.balanced_losses,
        rtol=1e-6,
    )


def test_scan_rejection_is_atomic_per_event_and_preserves_retry_owner() -> None:
    objectives = _objectives()
    initial = objectives.init(jr.key(40))
    trajectory = list(_trajectory())
    continuations = trajectory[7].at[1].set(jnp.float32(1.5))
    trajectory[7] = continuations
    result = _run_scan(objectives, initial, tuple(trajectory))
    assert bool(result.cache_applied[0])
    assert bool(result.update_applied[0])
    assert bool(result.cache_applied[1])
    assert not bool(result.update_applied[1])
    assert not bool(result.cache_applied[2])
    assert not bool(result.update_applied[2])
    assert bool(result.state.pending_valid)
    np.testing.assert_array_equal(result.state.update_words, [0, 1])
    np.testing.assert_array_equal(result.state.decision_words, [0, 2])

