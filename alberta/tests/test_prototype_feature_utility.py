# mypy: disable-error-code="attr-defined,call-arg"
"""Standalone L0 contracts for causal feature-utility auditing."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from alberta_framework.core.prototype_feature_utility import (
    PROTOTYPE_FEATURE_UTILITY_CURATION_AUTHORITY,
    PROTOTYPE_FEATURE_UTILITY_MECHANISM_STATUS,
    PROTOTYPE_FEATURE_UTILITY_SCIENTIFIC_PROMOTION_ALLOWED,
    PrototypeFeatureUtilityAuditor,
    PrototypeFeatureUtilityConfig,
    PrototypeFeatureUtilityEvent,
)

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def _config(
    *,
    base_feature_dim: int = 4,
    active_pair_slots: int = 2,
    candidate_pair_slots: int = 3,
    managed_horde_demons: int = 2,
    utility_decay: float = 0.75,
    shadow_step_size: float = 0.2,
    second_moment_decay: float = 0.5,
    max_observations: int = 20,
) -> PrototypeFeatureUtilityConfig:
    return PrototypeFeatureUtilityConfig(
        base_feature_dim=base_feature_dim,
        active_pair_slots=active_pair_slots,
        candidate_pair_slots=candidate_pair_slots,
        managed_horde_demons=managed_horde_demons,
        utility_decay=utility_decay,
        shadow_step_size=shadow_step_size,
        second_moment_decay=second_moment_decay,
        scale_epsilon=1.0e-6,
        max_observations=max_observations,
    )


def _descriptors() -> tuple[jax.Array, jax.Array]:
    active = jnp.asarray([[0, 1], [1, 2]], dtype=jnp.int32)
    candidates = jnp.asarray([[0, 2], [2, 3], [0, 3]], dtype=jnp.int32)
    return active, candidates


def _auditor_and_state(
    config: PrototypeFeatureUtilityConfig | None = None,
) -> tuple[PrototypeFeatureUtilityAuditor, Any]:
    auditor = PrototypeFeatureUtilityAuditor(_config() if config is None else config)
    active, candidates = _descriptors()
    state = auditor.init(
        active_descriptors=active[: auditor.config.active_pair_slots],
        candidate_descriptors=candidates[: auditor.config.candidate_pair_slots],
        semantic_generation=jnp.asarray(3, dtype=jnp.int32),
        semantic_generation_words=jnp.asarray([0, 3], dtype=jnp.uint32),
    )
    return auditor, state


def _pair_values(base: jax.Array, descriptors: jax.Array) -> jax.Array:
    live = (
        (descriptors[:, 0] >= 0)
        & (descriptors[:, 0] < descriptors[:, 1])
        & (descriptors[:, 1] < base.shape[0])
    )
    left = jnp.where(live, descriptors[:, 0], 0)
    right = jnp.where(live, descriptors[:, 1], 0)
    return base[left] * base[right] * live.astype(jnp.float32)


def _event(
    state: Any,
    *,
    base: jax.Array | None = None,
    targets: jax.Array | None = None,
    predictions: jax.Array | None = None,
    available: jax.Array | None = None,
    weights: jax.Array | None = None,
    semantic_generation: jax.Array | None = None,
    semantic_generation_words: jax.Array | None = None,
    active_descriptors: jax.Array | None = None,
    candidate_descriptors: jax.Array | None = None,
) -> PrototypeFeatureUtilityEvent:
    base = (
        jnp.asarray([2.0, 3.0, 4.0, 5.0], dtype=jnp.float32)
        if base is None
        else base
    )
    active_descriptors = (
        state.active_descriptors if active_descriptors is None else active_descriptors
    )
    candidate_descriptors = (
        state.candidate_descriptors
        if candidate_descriptors is None
        else candidate_descriptors
    )
    tail = _pair_values(base, active_descriptors)
    n_tasks = int(state.active_task_utilities.shape[0])
    default_targets = jnp.arange(1, n_tasks + 1, dtype=jnp.float32) * 2.0
    default_predictions = default_targets - 1.0
    default_weights = jnp.reshape(
        jnp.linspace(
            -0.4,
            0.6,
            n_tasks * active_descriptors.shape[0],
            dtype=jnp.float32,
        ),
        (n_tasks, active_descriptors.shape[0]),
    )
    event_generation = (
        state.semantic_generation
        if semantic_generation is None
        else semantic_generation
    )
    event_generation_words = (
        state.semantic_generation_words
        if semantic_generation is None
        else jnp.stack(
            (
                jnp.asarray(0, dtype=jnp.uint32),
                event_generation.astype(jnp.uint32),
            )
        )
    )
    if semantic_generation_words is not None:
        event_generation_words = semantic_generation_words
    return PrototypeFeatureUtilityEvent(
        base_observation=base,
        augmented_observation=jnp.concatenate([base, tail]),
        targets=default_targets if targets is None else targets,
        predictions=default_predictions if predictions is None else predictions,
        target_available=(
            jnp.ones((n_tasks,), dtype=jnp.bool_)
            if available is None
            else available
        ),
        active_consumer_tail_weights=default_weights if weights is None else weights,
        semantic_generation=event_generation,
        semantic_generation_words=event_generation_words,
        active_descriptors=active_descriptors,
        candidate_descriptors=candidate_descriptors,
    )


def _assert_tree_exact(left: Any, right: Any) -> None:
    left_leaves, left_tree = jax.tree.flatten(left)
    right_leaves, right_tree = jax.tree.flatten(right)
    assert left_tree == right_tree  # type: ignore[operator]
    assert len(left_leaves) == len(right_leaves)
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        np.testing.assert_array_equal(np.asarray(left_leaf), np.asarray(right_leaf))


def _assert_tree_close(left: Any, right: Any) -> None:
    left_leaves, left_tree = jax.tree.flatten(left)
    right_leaves, right_tree = jax.tree.flatten(right)
    assert left_tree == right_tree  # type: ignore[operator]
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        left_array = np.asarray(left_leaf)
        right_array = np.asarray(right_leaf)
        if np.issubdtype(left_array.dtype, np.inexact):
            np.testing.assert_allclose(left_array, right_array, rtol=2e-6, atol=2e-7)
        else:
            np.testing.assert_array_equal(left_array, right_array)


def test_config_is_strict_versioned_and_group_balanced() -> None:
    config = _config()
    assert config.n_tasks == 3
    assert config.task_utility_weights == (0.5, 0.25, 0.25)
    assert config.total_feature_dim == 6
    assert PrototypeFeatureUtilityConfig.from_config(config.to_config()) == config

    raw = config.to_config()
    assert raw["schema_version"] == "alberta.prototype-feature-utility.config.v2"
    assert raw["state_schema"] == "alberta.prototype-feature-utility.state.v2"
    with pytest.raises(ValueError, match="keys"):
        PrototypeFeatureUtilityConfig.from_config({**raw, "extra": 1})

    invalid: tuple[Callable[[], PrototypeFeatureUtilityConfig], ...] = (
        lambda: dataclasses.replace(config, base_feature_dim=1),
        lambda: dataclasses.replace(config, base_feature_dim=True),
        lambda: dataclasses.replace(config, active_pair_slots=0),
        lambda: dataclasses.replace(config, candidate_pair_slots=-1),
        lambda: dataclasses.replace(config, managed_horde_demons=0),
        lambda: dataclasses.replace(config, utility_decay=1.0),
        lambda: dataclasses.replace(config, shadow_step_size=0.0),
        lambda: dataclasses.replace(config, shadow_step_size=1.1),
        lambda: dataclasses.replace(config, second_moment_decay=float("nan")),
        lambda: dataclasses.replace(config, scale_epsilon=1.0e-50),
        lambda: dataclasses.replace(config, max_observations=0),
        lambda: dataclasses.replace(config, max_observations=2**64),
        lambda: dataclasses.replace(config, active_pair_slots=4_093),
        lambda: dataclasses.replace(
            config,
            active_pair_slots=1_000,
            candidate_pair_slots=1_000,
            managed_horde_demons=1_000,
        ),
    )
    for construct in invalid:
        with pytest.raises(ValueError):
            construct()


def test_initial_state_neutral_diagnostics_and_exact_resource_claims() -> None:
    auditor, state = _auditor_and_state()
    assert bool(auditor.state_valid(state))
    assert state.active_task_utilities.shape == (3, 2)
    assert state.active_task_evidence_counts.shape == (3, 2)
    assert state.candidate_shadow_weights.shape == (3, 3)
    assert state.candidate_task_utilities.shape == (3, 3)
    assert state.candidate_task_evidence_counts.shape == (3, 3)

    diagnostics = auditor.unavailable_diagnostics(state)
    assert not bool(diagnostics.available)
    assert not bool(diagnostics.transaction_applied)
    assert int(diagnostics.semantic_generation_before) == 3
    assert int(diagnostics.observation_count_before) == 0
    assert np.all(np.isfinite(np.asarray(diagnostics.active_loss_changes)))
    assert np.all(np.asarray(diagnostics.active_loss_changes) == 0.0)

    budget = auditor.resource_budget()
    expected_scalars = 6 + 2 * 2 + 2 * 3 * 2 + 3 * 3 + 3 * 3 * 3 + 3
    assert budget.persistent_logical_scalars == expected_scalars
    assert budget.persistent_state_nbytes == 4 * expected_scalars
    assert budget.telemetry_counter_nbytes == 8
    assert budget.exact_counter_nbytes == 16
    assert budget.counter_delta_nbytes == 16
    assert budget.counter_nbytes == 24
    actual_nbytes = sum(int(leaf.nbytes) for leaf in jax.tree.leaves(state))
    assert actual_nbytes == budget.persistent_state_nbytes
    assert budget.task_feature_score_cells_per_observe == 3 * (2 + 3)
    assert budget.shadow_update_cells_per_observe == 3 * 3
    assert budget.pair_products_per_observe == 2 + 3
    assert budget.state_descriptor_validation_cells_per_observe == 2 * 2 + 3 * 3
    assert budget.event_descriptor_validation_cells_per_observe == 2 * 2 + 3 * 3
    assert budget.identity_rebind_cells_per_observe == 2 * 2 + 3 * 3
    assert budget.candidate_active_collision_cells_per_observe == 2 * 3
    assert budget.descriptor_comparison_cells_per_observe == 3 * (2 * 2 + 3 * 3) + 2 * 3
    assert budget.max_observations == auditor.config.max_observations
    assert budget.rng_draws_per_observe == 0
    assert budget.backward_passes_per_observe == 0
    assert budget.consumer_updates_per_observe == 0
    assert budget.router_calls_per_observe == 0
    assert budget.curation_decisions_per_observe == 0
    assert budget.mechanism_status == PROTOTYPE_FEATURE_UTILITY_MECHANISM_STATUS
    assert not budget.scientific_promotion_allowed
    assert not budget.curation_authority
    assert not PROTOTYPE_FEATURE_UTILITY_SCIENTIFIC_PROMOTION_ALLOWED
    assert not PROTOTYPE_FEATURE_UTILITY_CURATION_AUTHORITY


def test_scores_frozen_consumers_before_shadow_and_moment_updates() -> None:
    auditor, state = _auditor_and_state()
    shadow = jnp.asarray(
        [[0.20, -0.10, 0.05], [0.30, 0.25, -0.20], [-0.40, 0.10, 0.15]],
        dtype=jnp.float32,
    )
    candidate_moments = jnp.asarray([0.5, 2.0, 4.0], dtype=jnp.float32)
    target_moments = jnp.asarray([9.0, 1.0, 64.0], dtype=jnp.float32)
    state = state.replace(
        candidate_shadow_weights=shadow,
        candidate_second_moments=candidate_moments,
        target_second_moments=target_moments,
    )
    targets = jnp.asarray([2.0, 4.0, 6.0], dtype=jnp.float32)
    predictions = jnp.asarray([1.0, 2.0, 7.0], dtype=jnp.float32)
    availability = jnp.asarray([True, False, True], dtype=jnp.bool_)
    weights = jnp.asarray(
        [[0.5, -0.25], [0.25, 0.5], [0.5, 0.0]], dtype=jnp.float32
    )
    event = _event(
        state,
        targets=targets,
        predictions=predictions,
        available=availability,
        weights=weights,
    )

    result = auditor.observe(state, event)
    diagnostics = result.diagnostics
    assert bool(diagnostics.transaction_applied)

    base = np.asarray(event.base_observation)
    active_values = np.asarray(event.augmented_observation[4:])
    candidate_values = np.asarray(_pair_values(event.base_observation, event.candidate_descriptors))
    scale2 = np.maximum.reduce(
        [np.asarray(target_moments), np.asarray(targets) ** 2, np.asarray(predictions) ** 2,
         np.full((3,), 1.0e-6, dtype=np.float32)]
    )
    error = (np.asarray(targets) - np.asarray(predictions)) / np.sqrt(scale2)
    active_q = np.asarray(weights) * active_values[None, :] / np.sqrt(scale2)[:, None]
    deletion = 0.5 * ((error[:, None] + active_q) ** 2 - error[:, None] ** 2)
    deletion = np.where(np.asarray(availability)[:, None], deletion, 0.0)
    active_gain = np.maximum(deletion, 0.0)
    active_gain = active_gain / (1.0 + active_gain)

    candidate_z = np.asarray(shadow) * candidate_values[None, :]
    insertion = 0.5 * (error[:, None] ** 2 - (error[:, None] - candidate_z) ** 2)
    insertion = np.where(np.asarray(availability)[:, None], insertion, 0.0)
    candidate_gain = np.maximum(insertion, 0.0)
    candidate_gain = candidate_gain / (1.0 + candidate_gain)
    task_weights = np.asarray([0.5, 0.25, 0.25], dtype=np.float32)

    np.testing.assert_allclose(diagnostics.target_scale_second_moments, scale2, rtol=1e-6)
    np.testing.assert_allclose(diagnostics.normalized_errors, error, rtol=1e-6)
    np.testing.assert_allclose(diagnostics.active_normalized_contributions, active_q, rtol=1e-6)
    np.testing.assert_allclose(diagnostics.active_loss_changes, deletion, rtol=1e-6)
    np.testing.assert_allclose(diagnostics.active_bounded_gains, active_gain, rtol=1e-6)
    np.testing.assert_allclose(
        diagnostics.active_signed_scores,
        deletion / (1.0 + np.abs(deletion)),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        diagnostics.candidate_normalized_contributions,
        candidate_z,
        rtol=1e-6,
    )
    np.testing.assert_allclose(diagnostics.candidate_loss_changes, insertion, rtol=1e-6)
    np.testing.assert_allclose(diagnostics.candidate_bounded_gains, candidate_gain, rtol=1e-6)

    np.testing.assert_array_equal(diagnostics.targets, targets)
    np.testing.assert_array_equal(diagnostics.predictions, predictions)
    np.testing.assert_array_equal(diagnostics.target_available, availability)
    np.testing.assert_array_equal(
        diagnostics.source_active_descriptors,
        event.active_descriptors,
    )
    np.testing.assert_array_equal(
        diagnostics.source_candidate_descriptors,
        event.candidate_descriptors,
    )
    np.testing.assert_array_equal(diagnostics.active_values, active_values)
    np.testing.assert_array_equal(diagnostics.candidate_values, candidate_values)
    np.testing.assert_array_equal(diagnostics.task_weights, task_weights)
    assert bool(diagnostics.state_values_valid)
    assert bool(diagnostics.event_values_valid)
    assert bool(diagnostics.state_descriptors_valid)
    assert bool(diagnostics.event_descriptors_valid)
    assert bool(diagnostics.binding_valid)
    assert bool(diagnostics.observation_matches_source)
    assert bool(diagnostics.capacity_available)
    assert bool(diagnostics.numerical_update_valid)
    assert bool(diagnostics.any_task_available)
    np.testing.assert_array_equal(
        diagnostics.active_live_mask,
        jnp.ones((2,), dtype=jnp.bool_),
    )
    np.testing.assert_array_equal(
        diagnostics.candidate_eligible_mask,
        jnp.ones((3,), dtype=jnp.bool_),
    )

    np.testing.assert_allclose(
        diagnostics.active_aggregate_signal,
        np.sum(task_weights[:, None] * active_gain, axis=0),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        diagnostics.candidate_aggregate_signal,
        np.sum(task_weights[:, None] * candidate_gain, axis=0),
        rtol=1e-6,
    )
    # Missing demon one contributes exactly zero mass; available weights are
    # deliberately not renormalized from 0.75 back to 1.0.
    assert float(diagnostics.active_aggregate_signal[0]) < float(
        active_gain[0, 0] + active_gain[2, 0]
    )

    np.testing.assert_array_equal(diagnostics.candidate_shadow_weights_before, shadow)
    candidate_square = candidate_values**2
    h = np.maximum.reduce(
        [
            np.asarray(candidate_moments),
            candidate_square,
            np.full((3,), np.mean(candidate_square), dtype=np.float32),
            np.full((3,), 1.0e-6, dtype=np.float32),
        ]
    )
    lipschitz = 1.0 + candidate_square / h
    expected_shadow = np.asarray(shadow) + (
        0.2
        * np.asarray(availability)[:, None]
        * (error[:, None] - candidate_z)
        * candidate_values[None, :]
        / (h[None, :] * lipschitz[None, :])
    )
    np.testing.assert_allclose(
        diagnostics.candidate_shadow_weights_after,
        expected_shadow,
        rtol=2e-6,
        atol=2e-7,
    )
    np.testing.assert_array_equal(
        diagnostics.candidate_second_moments_before, candidate_moments
    )
    np.testing.assert_array_equal(diagnostics.target_second_moments_before, target_moments)
    assert not np.array_equal(
        np.asarray(diagnostics.target_second_moments_after), np.asarray(target_moments)
    )
    assert base.dtype == np.float32


def test_computed_nonfinite_update_is_an_atomic_finite_noop() -> None:
    config = _config(active_pair_slots=1, candidate_pair_slots=1, managed_horde_demons=1)
    auditor = PrototypeFeatureUtilityAuditor(config)
    state = auditor.init(
        active_descriptors=jnp.asarray([[0, 1]], dtype=jnp.int32),
        candidate_descriptors=jnp.asarray([[0, 2]], dtype=jnp.int32),
        semantic_generation=jnp.asarray(0, dtype=jnp.int32),
        semantic_generation_words=jnp.zeros((2,), dtype=jnp.uint32),
    )
    huge = jnp.asarray(1.0e19, dtype=jnp.float32)
    base = jnp.full((4,), huge, dtype=jnp.float32)
    event = _event(
        state,
        base=base,
        targets=jnp.asarray([1.0, 1.0], dtype=jnp.float32),
        predictions=jnp.asarray([0.0, 0.0], dtype=jnp.float32),
        weights=jnp.full((2, 1), huge, dtype=jnp.float32),
    )
    assert np.all(np.isfinite(np.asarray(event.augmented_observation)))
    result = auditor.observe(state, event)
    _assert_tree_exact(result.state, state)
    assert bool(result.diagnostics.event_values_valid)
    assert not bool(result.diagnostics.numerical_update_valid)
    assert not bool(result.diagnostics.transaction_applied)
    for leaf in jax.tree.leaves(result.diagnostics):
        if jnp.issubdtype(leaf.dtype, jnp.inexact):
            assert np.all(np.isfinite(np.asarray(leaf)))


def test_v2_rejects_inactive_rows_and_active_candidate_collisions_stay_zero() -> None:
    auditor = PrototypeFeatureUtilityAuditor(_config())
    inactive_active = jnp.asarray([[0, 1], [-1, -1]], dtype=jnp.int32)
    inactive_candidates = jnp.asarray(
        [[0, 2], [-1, -1], [-1, -1]],
        dtype=jnp.int32,
    )
    with pytest.raises(ValueError, match="descriptors"):
        auditor.init(
            active_descriptors=inactive_active,
            candidate_descriptors=inactive_candidates,
            semantic_generation=jnp.asarray(0, dtype=jnp.int32),
            semantic_generation_words=jnp.zeros((2,), dtype=jnp.uint32),
        )

    collision_state = auditor.init(
        active_descriptors=jnp.asarray([[0, 1], [1, 2]], dtype=jnp.int32),
        candidate_descriptors=jnp.asarray(
            [[0, 1], [0, 2], [2, 3]],
            dtype=jnp.int32,
        ),
        semantic_generation=jnp.asarray(0, dtype=jnp.int32),
        semantic_generation_words=jnp.zeros((2,), dtype=jnp.uint32),
    )
    corrupt_collision = collision_state.replace(
        candidate_shadow_weights=collision_state.candidate_shadow_weights.at[0, 0].set(
            0.1
        )
    )
    assert not bool(auditor.state_valid(corrupt_collision))
    collision_result = auditor.observe(collision_state, _event(collision_state))
    assert bool(collision_result.diagnostics.transaction_applied)
    assert bool(collision_result.diagnostics.candidate_collision_mask[0])
    assert not bool(collision_result.diagnostics.candidate_eligible_mask[0])
    np.testing.assert_array_equal(
        collision_result.state.candidate_shadow_weights[:, 0],
        jnp.zeros((3,), dtype=jnp.float32),
    )
    np.testing.assert_array_equal(
        collision_result.state.candidate_task_utilities[:, 0],
        jnp.zeros((3,), dtype=jnp.float32),
    )
    np.testing.assert_array_equal(
        collision_result.state.candidate_task_evidence_counts[:, 0],
        jnp.zeros((3,), dtype=jnp.int32),
    )
    assert float(collision_result.state.candidate_second_moments[0]) == 0.0

    collision_event = _event(collision_state)
    values = np.asarray(
        _pair_values(
            collision_event.base_observation,
            collision_event.candidate_descriptors,
        )
    )
    eligible = np.asarray([False, True, True], dtype=np.bool_)
    energy = float(np.mean(values[eligible] ** 2))
    normalizer = np.maximum.reduce(
        [
            values**2,
            np.full((3,), energy, dtype=np.float32),
            np.full((3,), 1.0e-6, dtype=np.float32),
        ]
    )
    scale2 = np.maximum(
        np.asarray(collision_event.targets) ** 2,
        np.asarray(collision_event.predictions) ** 2,
    )
    errors = (
        np.asarray(collision_event.targets)
        - np.asarray(collision_event.predictions)
    ) / np.sqrt(scale2)
    expected_shadow = (
        0.2
        * errors[:, None]
        * values[None, :]
        / (normalizer[None, :] * (1.0 + values[None, :] ** 2 / normalizer[None, :]))
    )
    expected_shadow[:, ~eligible] = 0.0
    np.testing.assert_allclose(
        collision_result.state.candidate_shadow_weights,
        expected_shadow,
        rtol=2e-6,
        atol=2e-7,
    )


def test_state_valid_enforces_bounded_utilities_and_evidence_counts() -> None:
    auditor, state = _auditor_and_state()
    assert not bool(
        auditor.state_valid(
            state.replace(
                active_task_utilities=state.active_task_utilities.at[0, 0].set(1.01)
            )
        )
    )
    assert not bool(
        auditor.state_valid(
            state.replace(
                candidate_task_utilities=state.candidate_task_utilities.at[0, 0].set(-0.01)
            )
        )
    )
    assert not bool(
        auditor.state_valid(
            state.replace(
                active_task_evidence_counts=state.active_task_evidence_counts.at[0, 0].set(1)
            )
        )
    )


def test_signed_scores_distinguish_harmful_from_neutral_without_curation() -> None:
    config = _config(active_pair_slots=1, candidate_pair_slots=1, managed_horde_demons=1)
    auditor = PrototypeFeatureUtilityAuditor(config)
    active = jnp.asarray([[0, 1]], dtype=jnp.int32)
    candidate = jnp.asarray([[0, 2]], dtype=jnp.int32)
    state = auditor.init(
        active_descriptors=active,
        candidate_descriptors=candidate,
        semantic_generation=jnp.asarray(0, dtype=jnp.int32),
        semantic_generation_words=jnp.zeros((2,), dtype=jnp.uint32),
    ).replace(candidate_shadow_weights=jnp.asarray([[-1.0], [0.0]], dtype=jnp.float32))
    targets = jnp.asarray([2.0, 1.0], dtype=jnp.float32)
    predictions = jnp.asarray([1.0, 1.0], dtype=jnp.float32)
    weights = jnp.asarray([[-1.0], [0.0]], dtype=jnp.float32)
    event = _event(
        state,
        targets=targets,
        predictions=predictions,
        weights=weights,
        base=jnp.asarray([1.0, 1.0, 1.0, 1.0], dtype=jnp.float32),
    )
    diagnostics = auditor.observe(state, event).diagnostics
    assert float(diagnostics.active_loss_changes[0, 0]) < 0.0
    assert float(diagnostics.active_signed_scores[0, 0]) < 0.0
    assert float(diagnostics.active_bounded_gains[0, 0]) == 0.0
    assert float(diagnostics.candidate_loss_changes[0, 0]) < 0.0
    assert float(diagnostics.candidate_signed_scores[0, 0]) < 0.0
    assert float(diagnostics.candidate_bounded_gains[0, 0]) == 0.0
    assert not PROTOTYPE_FEATURE_UTILITY_CURATION_AUTHORITY


def test_all_unavailable_decays_utility_without_incrementing_evidence() -> None:
    auditor, state = _auditor_and_state()
    state = state.replace(
        active_task_utilities=jnp.full_like(state.active_task_utilities, 0.8),
        candidate_task_utilities=jnp.full_like(state.candidate_task_utilities, 0.4),
    )
    event = _event(
        state,
        available=jnp.zeros((auditor.config.n_tasks,), dtype=jnp.bool_),
    )
    result = auditor.observe(state, event)
    assert bool(result.diagnostics.transaction_applied)
    assert int(result.state.observation_count) == 1
    np.testing.assert_allclose(result.state.active_task_utilities, 0.6, rtol=1e-6)
    np.testing.assert_allclose(result.state.candidate_task_utilities, 0.3, rtol=1e-6)
    np.testing.assert_array_equal(
        result.state.active_task_evidence_counts,
        state.active_task_evidence_counts,
    )
    np.testing.assert_array_equal(
        result.state.candidate_task_evidence_counts,
        state.candidate_task_evidence_counts,
    )


def test_cap_and_every_rejection_path_are_exact_atomic_noops() -> None:
    auditor, state = _auditor_and_state(_config(max_observations=1))
    event = _event(state)
    first = auditor.observe(state, event)
    assert bool(first.diagnostics.transaction_applied)
    capped = auditor.observe(first.state, _event(first.state))
    _assert_tree_exact(capped.state, first.state)
    assert bool(capped.diagnostics.capacity_capped)
    assert not bool(capped.diagnostics.transaction_applied)

    auditor, state = _auditor_and_state()
    valid = _event(state)
    nonfinite = valid.replace(
        targets=valid.targets.at[0].set(jnp.asarray(jnp.inf, dtype=jnp.float32)),
    )
    stale = valid.replace(
        semantic_generation=state.semantic_generation - jnp.asarray(1, dtype=jnp.int32),
    )
    fork = valid.replace(
        active_descriptors=valid.active_descriptors.at[0].set(
            jnp.asarray([0, 2], dtype=jnp.int32)
        ),
    )
    signed_zero_base = valid.base_observation.at[0].set(
        jnp.asarray(-0.0, dtype=jnp.float32)
    )
    signed_zero_augmented = valid.augmented_observation.at[0].set(
        jnp.asarray(0.0, dtype=jnp.float32)
    )
    signed_zero_mismatch = valid.replace(
        base_observation=signed_zero_base,
        augmented_observation=signed_zero_augmented,
    )
    for rejected in (nonfinite, stale, fork, signed_zero_mismatch):
        result = auditor.observe(state, rejected)
        _assert_tree_exact(result.state, state)
        assert not bool(result.diagnostics.transaction_applied)
        leaves = jax.tree.leaves(result.diagnostics)
        for leaf in leaves:
            if jnp.issubdtype(leaf.dtype, jnp.inexact):
                assert np.all(np.isfinite(np.asarray(leaf)))


def test_observe_rejects_new_generation_then_rebinds_identity_only() -> None:
    auditor, state = _auditor_and_state()
    state = state.replace(
        active_task_utilities=jnp.asarray(
            [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=jnp.float32
        ),
        active_task_evidence_counts=jnp.asarray(
            [[1, 2], [3, 4], [5, 6]], dtype=jnp.int32
        ),
        candidate_shadow_weights=jnp.asarray(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            dtype=jnp.float32,
        ),
        candidate_task_utilities=jnp.asarray(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
            dtype=jnp.float32,
        ),
        candidate_task_evidence_counts=jnp.asarray(
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=jnp.int32
        ),
        candidate_second_moments=jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float32),
        observation_count=jnp.asarray(10, dtype=jnp.int32),
        observation_words=jnp.asarray([0, 10], dtype=jnp.uint32),
    )
    new_active = jnp.asarray([[1, 2], [0, 2]], dtype=jnp.int32)
    # [2,3] survives candidate identity, [0,1] is reacquired from the old
    # active bank, and [0,2] collides with the newly deployed active bank.
    new_candidates = jnp.asarray([[2, 3], [0, 1], [0, 2]], dtype=jnp.int32)
    newer_event = _event(
        state,
        semantic_generation=jnp.asarray(4, dtype=jnp.int32),
        semantic_generation_words=jnp.asarray([0, 4], dtype=jnp.uint32),
        active_descriptors=new_active,
        candidate_descriptors=new_candidates,
        available=jnp.zeros((3,), dtype=jnp.bool_),
    )
    rejected = auditor.observe(state, newer_event)
    _assert_tree_exact(rejected.state, state)
    assert not bool(rejected.diagnostics.transaction_applied)
    assert bool(rejected.diagnostics.skipped_generation)

    rebound = auditor.rebind(
        state,
        active_descriptors=new_active,
        candidate_descriptors=new_candidates,
        semantic_generation=jnp.asarray(4, dtype=jnp.int32),
        semantic_generation_words=jnp.asarray([0, 4], dtype=jnp.uint32),
    )
    diagnostics = rebound.diagnostics
    assert bool(diagnostics.binding_rebound)
    assert bool(diagnostics.transaction_applied)
    assert int(rebound.state.observation_count) == 10
    np.testing.assert_array_equal(
        diagnostics.active_task_utilities_before[:, 0],
        state.active_task_utilities[:, 1],
    )
    np.testing.assert_array_equal(
        diagnostics.active_task_utilities_before[:, 1],
        jnp.zeros((3,), dtype=jnp.float32),
    )
    np.testing.assert_array_equal(
        diagnostics.candidate_shadow_weights_before[:, 0],
        state.candidate_shadow_weights[:, 1],
    )
    np.testing.assert_array_equal(
        diagnostics.candidate_shadow_weights_before[:, 1:],
        jnp.zeros((3, 2), dtype=jnp.float32),
    )
    np.testing.assert_array_equal(
        diagnostics.candidate_task_evidence_counts_before[:, 0],
        state.candidate_task_evidence_counts[:, 1],
    )
    np.testing.assert_array_equal(
        diagnostics.candidate_task_evidence_counts_before[:, 1:],
        jnp.zeros((3, 2), dtype=jnp.int32),
    )


def test_explicit_rebind_is_a_no_observation_identity_route() -> None:
    auditor, state = _auditor_and_state()
    observed = auditor.observe(state, _event(state)).state
    new_active = jnp.asarray([[1, 2], [0, 2]], dtype=jnp.int32)
    new_candidates = jnp.asarray([[2, 3], [0, 1], [0, 2]], dtype=jnp.int32)

    rebound = auditor.rebind(
        observed,
        active_descriptors=new_active,
        candidate_descriptors=new_candidates,
        semantic_generation=observed.semantic_generation
        + jnp.asarray(1, dtype=jnp.int32),
        semantic_generation_words=observed.semantic_generation_words.at[1].add(
            jnp.asarray(1, dtype=jnp.uint32)
        ),
    )
    assert bool(rebound.diagnostics.transaction_applied)
    assert int(rebound.state.observation_count) == int(observed.observation_count)
    np.testing.assert_array_equal(
        rebound.state.target_second_moments,
        observed.target_second_moments,
    )
    np.testing.assert_array_equal(
        rebound.state.active_task_utilities[:, 0],
        observed.active_task_utilities[:, 1],
    )
    np.testing.assert_array_equal(
        rebound.state.active_task_utilities[:, 1],
        jnp.zeros((auditor.config.n_tasks,), dtype=jnp.float32),
    )
    assert bool(rebound.diagnostics.active_survivor_mask[0])
    assert not bool(rebound.diagnostics.active_survivor_mask[1])
    assert bool(rebound.diagnostics.candidate_survivor_mask[0])
    assert not bool(rebound.diagnostics.candidate_survivor_mask[1])
    assert bool(rebound.diagnostics.candidate_collision_mask[2])

    compiled = jax.jit(auditor.rebind)(
        observed,
        active_descriptors=new_active,
        candidate_descriptors=new_candidates,
        semantic_generation=observed.semantic_generation
        + jnp.asarray(1, dtype=jnp.int32),
        semantic_generation_words=observed.semantic_generation_words.at[1].add(
            jnp.asarray(1, dtype=jnp.uint32)
        ),
    )
    _assert_tree_exact(compiled, rebound)

    skipped = auditor.rebind(
        observed,
        active_descriptors=new_active,
        candidate_descriptors=new_candidates,
        semantic_generation=observed.semantic_generation
        + jnp.asarray(2, dtype=jnp.int32),
        semantic_generation_words=observed.semantic_generation_words.at[1].add(
            jnp.asarray(2, dtype=jnp.uint32)
        ),
    )
    _assert_tree_exact(skipped.state, observed)
    assert bool(skipped.diagnostics.skipped_generation)
    assert not bool(skipped.diagnostics.transaction_applied)

    stale = auditor.rebind(
        observed,
        active_descriptors=new_active,
        candidate_descriptors=new_candidates,
        semantic_generation=observed.semantic_generation,
        semantic_generation_words=observed.semantic_generation_words,
    )
    _assert_tree_exact(stale.state, observed)
    assert bool(stale.diagnostics.stale_generation)
    assert not bool(stale.diagnostics.transaction_applied)

    newer_observation = _event(
        observed,
        semantic_generation=observed.semantic_generation
        + jnp.asarray(1, dtype=jnp.int32),
    )
    rejected_newer = auditor.observe(observed, newer_observation)
    _assert_tree_exact(rejected_newer.state, observed)
    assert bool(rejected_newer.diagnostics.skipped_generation)
    assert not bool(rejected_newer.diagnostics.binding_rebound)
    assert not bool(rejected_newer.diagnostics.transaction_applied)


def test_exact_static_array_contracts_reject_before_indexed_work() -> None:
    auditor, state = _auditor_and_state()
    event = _event(state)
    with pytest.raises(TypeError, match="base_observation"):
        auditor.observe(
            state,
            event.replace(
                base_observation=event.base_observation.astype(jnp.int32),
            ),
        )
    with pytest.raises(ValueError, match="augmented_observation"):
        auditor.observe(
            state,
            event.replace(
                augmented_observation=event.augmented_observation[:-1],
            ),
        )
    with pytest.raises(TypeError, match="state"):
        auditor.observe(cast(Any, object()), event)


def test_eager_jit_and_scan_have_matching_fixed_shape_semantics() -> None:
    auditor, state = _auditor_and_state()
    event = _event(state)
    eager = auditor.observe(state, event)
    compiled = jax.jit(auditor.observe)(state, event)
    _assert_tree_close(compiled, eager)

    events = jax.tree.map(lambda value: jnp.stack([value, value]), event)

    def scan_step(carry: Any, item: PrototypeFeatureUtilityEvent) -> tuple[Any, Any]:
        result = auditor.observe(carry, item)
        return result.state, result.diagnostics.active_aggregate_signal

    scanned_state, scanned_signals = jax.lax.scan(scan_step, state, events)
    first = auditor.observe(state, event)
    second = auditor.observe(first.state, event)
    _assert_tree_close(scanned_state, second.state)
    np.testing.assert_allclose(
        scanned_signals,
        jnp.stack(
            [
                first.diagnostics.active_aggregate_signal,
                second.diagnostics.active_aggregate_signal,
            ]
        ),
        rtol=2e-6,
        atol=2e-7,
    )
