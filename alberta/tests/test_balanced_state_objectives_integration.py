"""Integration contracts for balanced learned-state auxiliary objectives."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.balanced_state_objectives import (
    BalancedStateObjectives,
    BalancedStateObjectivesConfig,
    BalancedStateObjectivesScanResult,
    BalancedStateObjectivesState,
    load_balanced_state_objectives_checkpoint,
    run_balanced_state_objectives_scan,
    save_balanced_state_objectives_checkpoint,
)
from alberta_framework.core.state_builder import (
    OnlineGatedStateBuilder,
    OnlineGatedStateBuilderConfig,
    OnlineGatedStateBuilderState,
)

pytestmark = pytest.mark.integration


def _objectives() -> BalancedStateObjectives:
    return BalancedStateObjectives(
        BalancedStateObjectivesConfig(
            representation_dim=3,
            n_actions=2,
            gvf_discounts=(0.0, 0.5, 0.9),
            gvf_step_size=0.03,
            inverse_step_size=0.04,
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
    cumulants = jnp.asarray([0.1, -0.2, 0.3, 0.0], dtype=jnp.float32)
    continuations = jnp.asarray([1.0, 1.0, 0.0, 1.0], dtype=jnp.float32)
    current_revisions = jnp.asarray([[0, 0], [0, 0], [0, 1], [0, 1]], dtype=jnp.uint32)
    next_revisions = jnp.asarray([[0, 0], [0, 1], [0, 1], [0, 2]], dtype=jnp.uint32)
    return (
        current,
        successor,
        actions,
        cumulants,
        continuations,
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
                np.asarray(left_leaf),
                np.asarray(right_leaf),
                rtol=1e-6,
                atol=1e-7,
            )
        else:
            np.testing.assert_array_equal(np.asarray(left_leaf), np.asarray(right_leaf))


def _run_scan(
    objectives: BalancedStateObjectives,
    state: BalancedStateObjectivesState,
    trajectory: tuple[jax.Array, ...],
) -> BalancedStateObjectivesScanResult:
    return run_balanced_state_objectives_scan(objectives, state, *trajectory)


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
        current, successor, action, cumulant, continuation, current_revision, next_revision = (
            values
        )
        cached = objectives.cache_action(state, current, action, current_revision)
        updated = objectives.update(
            cached.state,
            cached.receipt,
            successor,
            next_revision,
            cumulant,
            continuation,
        )
        state = updated.state
        losses.append(updated.balanced_loss)
        current_gradients.append(updated.current_representation_gradient)
        next_gradients.append(updated.next_representation_gradient)
    _assert_tree_allclose(state, compiled.state)
    np.testing.assert_allclose(jnp.stack(losses), compiled.balanced_losses, rtol=1e-6)
    np.testing.assert_allclose(
        jnp.stack(current_gradients),
        compiled.current_representation_gradients,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        jnp.stack(next_gradients),
        compiled.next_representation_gradients,
        rtol=1e-6,
    )


def test_checkpoint_resume_with_pending_owner_matches_uninterrupted(tmp_path: Path) -> None:
    objectives = _objectives()
    trajectory = _trajectory()
    initial = objectives.init(jr.key(21))
    prefix = _run_scan(
        objectives,
        initial,
        tuple(values[:2] for values in trajectory),
    )
    current, successor, action, cumulant, continuation, current_revision, next_revision = (
        value[2] for value in trajectory
    )
    pending = objectives.cache_action(prefix.state, current, action, current_revision)
    assert bool(pending.cache_applied)

    checkpoint = tmp_path / "balanced-state-objectives"
    save_balanced_state_objectives_checkpoint(objectives, pending.state, checkpoint)
    restored_objectives, restored_state = load_balanced_state_objectives_checkpoint(checkpoint)
    assert restored_objectives.to_config() == objectives.to_config()
    _assert_tree_allclose(restored_state, pending.state)

    resumed = restored_objectives.update(
        restored_state,
        pending.receipt,
        successor,
        next_revision,
        cumulant,
        continuation,
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


def test_owner_bound_gradient_commits_to_advanced_online_state_builder() -> None:
    builder = OnlineGatedStateBuilder(
        OnlineGatedStateBuilderConfig(
            observation_dim=2,
            n_actions=2,
            hidden_dim=1,
            include_raw_observation=True,
            step_size=0.1,
            gradient_clip=10.0,
        )
    )
    source_state, current_representation = builder.start(
        builder.init(jr.key(30)),
        jnp.asarray([0.4, -0.2], dtype=jnp.float32),
    )
    destination_state, next_representation = builder.update(
        source_state,
        jnp.asarray([0.1, 0.6], dtype=jnp.float32),
        jnp.asarray(1, dtype=jnp.int32),
        jnp.asarray(0.3, dtype=jnp.float32),
        jnp.asarray(1.0, dtype=jnp.float32),
    )
    assert builder.feature_dim() == 3
    objectives = _objectives()
    objective_state = objectives.init(jr.key(31))
    cached = objectives.cache_action(
        objective_state,
        current_representation,
        jnp.asarray(1, dtype=jnp.int32),
        source_state.update_words,
    )
    updated = objectives.update(
        cached.state,
        cached.receipt,
        next_representation,
        destination_state.update_words,
        jnp.asarray(0.3, dtype=jnp.float32),
        jnp.asarray(1.0, dtype=jnp.float32),
    )
    assert bool(updated.update_applied)
    np.testing.assert_array_equal(
        updated.current_representation_revision_words,
        source_state.update_words,
    )
    proposal = builder.propose_learning_update(
        source_state,
        updated.current_representation_gradient,
    )
    learned_state, diagnostics = builder.commit_learning_update(
        destination_state,
        proposal,
    )
    assert bool(diagnostics.source_matches)
    assert bool(diagnostics.update_applied)
    assert bool(builder.state_valid(learned_state))
    np.testing.assert_array_equal(learned_state.step_words, destination_state.step_words)
    np.testing.assert_array_equal(learned_state.update_words, [0, 1])
    assert not np.array_equal(
        np.asarray(learned_state.parameters),
        np.asarray(cast(OnlineGatedStateBuilderState, destination_state).parameters),
    )


def test_scan_rejection_is_atomic_per_event_and_preserves_retry_owner() -> None:
    objectives = _objectives()
    initial = objectives.init(jr.key(40))
    trajectory = list(_trajectory())
    continuations = trajectory[4].at[1].set(jnp.float32(1.5))
    trajectory[4] = continuations
    result = _run_scan(objectives, initial, tuple(trajectory))
    assert bool(result.cache_applied[0])
    assert bool(result.update_applied[0])
    assert bool(result.cache_applied[1])
    assert not bool(result.update_applied[1])
    # The rejected update keeps the pending owner, so later cache attempts are
    # rejected rather than silently skipping or overwriting that transition.
    assert not bool(result.cache_applied[2])
    assert not bool(result.update_applied[2])
    assert bool(result.state.pending_valid)
    np.testing.assert_array_equal(result.state.update_words, [0, 1])
    np.testing.assert_array_equal(result.state.decision_words, [0, 2])
