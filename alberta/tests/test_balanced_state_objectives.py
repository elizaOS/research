"""Unit contracts for causal balanced learned-state objectives."""

from __future__ import annotations

import dataclasses
import json
from typing import Any, cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.balanced_state_objectives import (
    BALANCED_STATE_OBJECTIVES_EVIDENCE_LEVEL,
    BALANCED_STATE_OBJECTIVES_LIMITATIONS,
    BALANCED_STATE_OBJECTIVES_OUTCOME_STATUS,
    BALANCED_STATE_OBJECTIVES_OWNERSHIP,
    BalancedStateObjectives,
    BalancedStateObjectivesConfig,
    BalancedStateObjectivesState,
    StateObjectiveActionReceipt,
    measure_balanced_state_objectives_state_nbytes,
)

pytestmark = pytest.mark.unit


def _config(**updates: Any) -> BalancedStateObjectivesConfig:
    values: dict[str, Any] = {
        "representation_dim": 2,
        "n_actions": 2,
        "gvf_discounts": (0.0, 0.5),
        "gvf_step_size": 0.1,
        "inverse_step_size": 0.2,
        "representation_gradient_clip": 100.0,
    }
    values.update(updates)
    return BalancedStateObjectivesConfig(**values)


def _assert_tree_equal(left: object, right: object) -> None:
    left_leaves, left_tree = jax.tree.flatten(left)
    right_leaves, right_tree = jax.tree.flatten(right)
    assert str(left_tree) == str(right_tree)
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        np.testing.assert_array_equal(np.asarray(left_leaf), np.asarray(right_leaf))


def _known_state(objectives: BalancedStateObjectives) -> BalancedStateObjectivesState:
    state = objectives.init(jr.key(7))
    return dataclasses.replace(  # type: ignore[type-var]
        state,
        gvf_weights=jnp.asarray([[1.0, 0.0], [0.0, 2.0]], dtype=jnp.float32),
        inverse_current_weights=jnp.asarray(
            [[1.0, 0.0], [0.0, 1.0]], dtype=jnp.float32
        ),
        inverse_next_weights=jnp.asarray(
            [[0.5, 0.0], [0.0, 0.5]], dtype=jnp.float32
        ),
        inverse_bias=jnp.zeros((2,), dtype=jnp.float32),
    )


def _cache_known(
    objectives: BalancedStateObjectives,
    state: BalancedStateObjectivesState,
    *,
    revision: tuple[int, int] = (0, 3),
) -> tuple[BalancedStateObjectivesState, StateObjectiveActionReceipt]:
    result = objectives.cache_action(
        state,
        jnp.asarray([2.0, 1.0], dtype=jnp.float32),
        jnp.asarray(1, dtype=jnp.int32),
        jnp.asarray(revision, dtype=jnp.uint32),
    )
    assert bool(result.cache_applied)
    return result.state, result.receipt


def test_config_roundtrip_is_strict_json_safe_l0_and_honest() -> None:
    config = _config(
        gvf_discounts=(0.0, 0.25, 0.9),
        gvf_group_weight=0.4,
        inverse_group_weight=0.6,
    )
    payload = config.to_config()
    restored = BalancedStateObjectivesConfig.from_config(
        cast(dict[str, Any], json.loads(json.dumps(payload)))
    )
    assert restored == config
    assert payload["evidence_level"] == BALANCED_STATE_OBJECTIVES_EVIDENCE_LEVEL == "L0"
    assert payload["outcome_status"] == BALANCED_STATE_OBJECTIVES_OUTCOME_STATUS == "not_assessed"
    assert payload["ownership"] == BALANCED_STATE_OBJECTIVES_OWNERSHIP
    assert payload["limitations"] == list(BALANCED_STATE_OBJECTIVES_LIMITATIONS)
    assert "fixed-declared-not-empirically-calibrated-group-balance" in payload[
        "limitations"
    ]
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.n_actions = 3  # type: ignore[misc]

    for malformed in (
        {**payload, "unknown": 1},
        {name: value for name, value in payload.items() if name != "ownership"},
        {**payload, "evidence_level": "L2"},
        {**payload, "outcome_status": "accepted"},
        {**payload, "limitations": []},
    ):
        with pytest.raises(ValueError):
            BalancedStateObjectivesConfig.from_config(malformed)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"representation_dim": True},
        {"representation_dim": 0},
        {"n_actions": 1},
        {"gvf_discounts": [0.0, 0.5]},
        {"gvf_discounts": (0.5,)},
        {"gvf_discounts": (0.5, 0.5)},
        {"gvf_discounts": (0.7, 0.2)},
        {"gvf_discounts": (0.0, 1.0)},
        {"gvf_step_size": 0.0},
        {"inverse_step_size": float("nan")},
        {"gvf_group_weight": 0.0, "inverse_group_weight": 1.0},
        {"gvf_group_weight": 0.2, "inverse_group_weight": 0.2},
        {"representation_gradient_clip": 0.0},
    ],
)
def test_config_rejects_ambiguous_or_unbalanced_contracts(kwargs: dict[str, Any]) -> None:
    with pytest.raises((TypeError, ValueError)):
        _config(**kwargs)


def test_init_requires_typed_threefry_and_has_frozen_valid_state() -> None:
    objectives = BalancedStateObjectives(_config())
    state = objectives.init(jr.key(11))
    assert bool(objectives.state_valid(state))
    with pytest.raises(dataclasses.FrozenInstanceError):
        state.pending_valid = jnp.asarray(True)
    with pytest.raises(TypeError, match="typed Threefry"):
        objectives.init(jr.PRNGKey(11))


def test_cache_binds_exact_owner_and_rejects_overwrite_or_invalid_input_atomically() -> None:
    objectives = BalancedStateObjectives(_config())
    initial = objectives.init(jr.key(12))
    cached, receipt = _cache_known(objectives, initial)
    np.testing.assert_array_equal(receipt.representation, [2.0, 1.0])
    assert int(receipt.action) == 1
    np.testing.assert_array_equal(receipt.representation_revision_words, [0, 3])
    np.testing.assert_array_equal(receipt.action_identity_words, [0, 1])
    assert bool(objectives.state_valid(cached))

    overwrite = objectives.cache_action(
        cached,
        jnp.asarray([0.0, 0.0], dtype=jnp.float32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.zeros((2,), dtype=jnp.uint32),
    )
    assert not bool(overwrite.cache_applied)
    assert not bool(overwrite.cache_available)
    _assert_tree_equal(overwrite.state, cached)

    invalid = objectives.cache_action(
        initial,
        jnp.asarray([jnp.nan, 0.0], dtype=jnp.float32),
        jnp.asarray(4, dtype=jnp.int32),
        jnp.zeros((2,), dtype=jnp.uint32),
    )
    assert not bool(invalid.cache_applied)
    _assert_tree_equal(invalid.state, initial)


def test_update_math_uses_mean_gvf_group_separate_inverse_head_and_fixed_group_mass() -> None:
    objectives = BalancedStateObjectives(_config())
    initial = _known_state(objectives)
    cached, receipt = _cache_known(objectives, initial)
    result = objectives.update(
        cached,
        receipt,
        jnp.asarray([1.0, 3.0], dtype=jnp.float32),
        jnp.asarray([0, 4], dtype=jnp.uint32),
        jnp.asarray(0.25, dtype=jnp.float32),
        jnp.asarray(0.8, dtype=jnp.float32),
    )
    assert bool(result.update_applied)
    np.testing.assert_allclose(result.gvf_predictions, [2.0, 2.0], rtol=1e-6)
    np.testing.assert_allclose(result.gvf_targets, [0.25, 2.65], rtol=1e-6)
    np.testing.assert_allclose(result.gvf_td_errors, [-1.75, 0.65], rtol=1e-6)
    np.testing.assert_allclose(result.inverse_probabilities, [0.5, 0.5], rtol=1e-6)
    np.testing.assert_allclose(
        result.gvf_current_representation_gradient,
        [0.875, -0.65],
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        result.inverse_current_representation_gradient,
        [0.5, -0.5],
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        result.inverse_next_representation_gradient,
        [0.25, -0.25],
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        result.current_representation_gradient,
        [0.6875, -0.575],
        rtol=1e-6,
    )
    np.testing.assert_allclose(result.next_representation_gradient, [0.125, -0.125])
    expected_gvf_loss = 0.5 * np.mean(np.square([1.75, -0.65]))
    expected_inverse_loss = np.log(2.0)
    np.testing.assert_allclose(result.gvf_loss, expected_gvf_loss, rtol=1e-6)
    np.testing.assert_allclose(result.inverse_loss, expected_inverse_loss, rtol=1e-6)
    np.testing.assert_allclose(
        result.balanced_loss,
        0.5 * expected_gvf_loss + 0.5 * expected_inverse_loss,
        rtol=1e-6,
    )

    expected_gvf_gradient = np.asarray(
        [[[1.75 * 2.0, 1.75 * 1.0]], [[-0.65 * 2.0, -0.65 * 1.0]]]
    ).reshape(2, 2) / 2.0
    np.testing.assert_allclose(
        result.state.gvf_weights,
        np.asarray([[1.0, 0.0], [0.0, 2.0]]) - 0.1 * expected_gvf_gradient,
        rtol=1e-6,
    )
    assert not np.array_equal(
        np.asarray(result.state.inverse_current_weights),
        np.asarray(initial.inverse_current_weights),
    )
    assert not np.array_equal(
        np.asarray(result.state.inverse_next_weights),
        np.asarray(initial.inverse_next_weights),
    )
    np.testing.assert_array_equal(result.post_update_words, [0, 1])
    np.testing.assert_array_equal(result.post_head_revision_words, [0, 1])
    assert not bool(result.state.pending_valid)
    assert bool(objectives.state_valid(result.state))


def test_gvf_group_gradient_is_head_mean_not_head_count_sum() -> None:
    gradients = []
    for discounts in ((0.0, 0.2), (0.0, 0.2, 0.4, 0.6)):
        objectives = BalancedStateObjectives(_config(gvf_discounts=discounts))
        state = dataclasses.replace(  # type: ignore[type-var]
            objectives.init(jr.key(len(discounts))),
            gvf_weights=jnp.tile(
                jnp.asarray([[1.0, 0.0]], dtype=jnp.float32),
                (len(discounts), 1),
            ),
            inverse_current_weights=jnp.zeros((2, 2), dtype=jnp.float32),
            inverse_next_weights=jnp.zeros((2, 2), dtype=jnp.float32),
            inverse_bias=jnp.zeros((2,), dtype=jnp.float32),
        )
        cached = objectives.cache_action(
            state,
            jnp.asarray([1.0, 0.0], dtype=jnp.float32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.zeros((2,), dtype=jnp.uint32),
        )
        result = objectives.update(
            cached.state,
            cached.receipt,
            jnp.zeros((2,), dtype=jnp.float32),
            jnp.zeros((2,), dtype=jnp.uint32),
            jnp.asarray(0.0, dtype=jnp.float32),
            jnp.asarray(1.0, dtype=jnp.float32),
        )
        assert bool(result.update_applied)
        gradients.append(result.gvf_current_representation_gradient)
    np.testing.assert_allclose(gradients[0], [1.0, 0.0])
    np.testing.assert_allclose(gradients[1], gradients[0])


@pytest.mark.parametrize("failure", ["receipt", "successor", "revision", "continuation"])
def test_invalid_update_is_bit_exact_noop_and_preserves_pending_retry(failure: str) -> None:
    objectives = BalancedStateObjectives(_config())
    cached, receipt = _cache_known(objectives, _known_state(objectives), revision=(1, 0))
    successor = jnp.asarray([1.0, 3.0], dtype=jnp.float32)
    next_revision = jnp.asarray([1, 0], dtype=jnp.uint32)
    continuation = jnp.asarray(0.9, dtype=jnp.float32)
    if failure == "receipt":
        receipt = dataclasses.replace(  # type: ignore[type-var]
            receipt,
            action=jnp.asarray(0, dtype=jnp.int32),
        )
    elif failure == "successor":
        successor = jnp.asarray([jnp.inf, 3.0], dtype=jnp.float32)
    elif failure == "revision":
        next_revision = jnp.asarray([0, 2**32 - 1], dtype=jnp.uint32)
    else:
        continuation = jnp.asarray(1.1, dtype=jnp.float32)
    result = objectives.update(
        cached,
        receipt,
        successor,
        next_revision,
        jnp.asarray(0.2, dtype=jnp.float32),
        continuation,
    )
    assert not bool(result.update_applied)
    _assert_tree_equal(result.state, cached)
    np.testing.assert_array_equal(result.current_representation_gradient, [0.0, 0.0])
    np.testing.assert_array_equal(result.next_representation_gradient, [0.0, 0.0])
    assert bool(result.state.pending_valid)


def test_exact_clock_fails_stop_at_uint64_capacity_without_mutation() -> None:
    objectives = BalancedStateObjectives(_config())
    maximum = jnp.asarray([2**32 - 1, 2**32 - 1], dtype=jnp.uint32)
    saturated = dataclasses.replace(  # type: ignore[type-var]
        objectives.init(jr.key(13)),
        decision_words=maximum,
        update_words=maximum,
        head_revision_words=maximum,
    )
    assert bool(objectives.state_valid(saturated))
    result = objectives.cache_action(
        saturated,
        jnp.asarray([1.0, 0.0], dtype=jnp.float32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.zeros((2,), dtype=jnp.uint32),
    )
    assert not bool(result.lifetime_capacity_available)
    assert not bool(result.cache_applied)
    _assert_tree_equal(result.state, saturated)


def test_corrupt_clock_rejects_head_and_actionable_builder_gradient_together() -> None:
    objectives = BalancedStateObjectives(_config())
    cached, receipt = _cache_known(objectives, _known_state(objectives))
    corrupt = dataclasses.replace(  # type: ignore[type-var]
        cached,
        head_revision_words=jnp.asarray([0, 1], dtype=jnp.uint32),
    )
    assert not bool(objectives.state_valid(corrupt))
    result = objectives.update(
        corrupt,
        receipt,
        jnp.asarray([1.0, 3.0], dtype=jnp.float32),
        jnp.asarray([0, 3], dtype=jnp.uint32),
        jnp.asarray(0.25, dtype=jnp.float32),
        jnp.asarray(0.8, dtype=jnp.float32),
    )
    assert not bool(result.update_applied)
    _assert_tree_equal(result.state, corrupt)
    np.testing.assert_array_equal(result.current_representation_gradient, [0.0, 0.0])
    np.testing.assert_array_equal(result.next_representation_gradient, [0.0, 0.0])


def test_resource_budget_counts_every_persistent_byte_and_declares_scope() -> None:
    objectives = BalancedStateObjectives(_config(gvf_discounts=(0.0, 0.5, 0.9)))
    state = objectives.init(jr.key(14))
    budget = objectives.resource_budget(state)
    assert budget.total_state_nbytes == measure_balanced_state_objectives_state_nbytes(state)
    assert budget.parameter_nbytes == 4 * (3 * 2 + 2 * 2 * 2 + 2)
    assert budget.pending_cache_nbytes == 4 * 2 + 4 + 8 + 8 + 1
    assert budget.clock_and_revision_nbytes == 24
    assert budget.max_head_updates_per_transition == 1
    assert budget.temporary_bytes_scope.endswith("not-a-measured-device-peak")


def test_gradient_clip_is_disclosed_and_does_not_change_head_loss_updates() -> None:
    objectives = BalancedStateObjectives(_config(representation_gradient_clip=0.01))
    cached, receipt = _cache_known(objectives, _known_state(objectives))
    result = objectives.update(
        cached,
        receipt,
        jnp.asarray([1.0, 3.0], dtype=jnp.float32),
        jnp.asarray([0, 3], dtype=jnp.uint32),
        jnp.asarray(0.25, dtype=jnp.float32),
        jnp.asarray(0.8, dtype=jnp.float32),
    )
    assert bool(result.update_applied)
    assert bool(result.current_gradient_was_clipped)
    assert bool(result.next_gradient_was_clipped)
    np.testing.assert_allclose(
        jnp.linalg.norm(result.current_representation_gradient),
        0.01,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        jnp.linalg.norm(result.next_representation_gradient),
        0.01,
        rtol=1e-5,
    )
