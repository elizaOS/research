"""Unit contracts for comprehensive WP3 learned-state objectives."""

from __future__ import annotations

import dataclasses
import json
from typing import Any, cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.comprehensive_state_objectives import (
    COMPREHENSIVE_STATE_OBJECTIVES_EVIDENCE_LEVEL,
    COMPREHENSIVE_STATE_OBJECTIVES_HEADS,
    COMPREHENSIVE_STATE_OBJECTIVES_LIMITATIONS,
    COMPREHENSIVE_STATE_OBJECTIVES_OUTCOME_STATUS,
    COMPREHENSIVE_STATE_OBJECTIVES_OWNERSHIP,
    ComprehensiveStateObjectiveActionReceipt,
    ComprehensiveStateObjectives,
    ComprehensiveStateObjectivesConfig,
    ComprehensiveStateObjectivesState,
    measure_comprehensive_state_objectives_state_nbytes,
)

pytestmark = pytest.mark.unit


def _config(**updates: Any) -> ComprehensiveStateObjectivesConfig:
    values: dict[str, Any] = {
        "representation_dim": 2,
        "observation_target_dim": 2,
        "n_actions": 2,
        "gvf_discounts": (0.0, 0.5),
        "observation_step_size": 0.1,
        "latent_step_size": 0.1,
        "reward_step_size": 0.1,
        "termination_step_size": 0.1,
        "gvf_step_size": 0.1,
        "value_step_size": 0.1,
        "advantage_step_size": 0.1,
        "inverse_step_size": 0.1,
        "representation_gradient_clip": 100.0,
    }
    values.update(updates)
    return ComprehensiveStateObjectivesConfig(**values)


def _assert_tree_equal(left: object, right: object) -> None:
    left_leaves, left_tree = jax.tree.flatten(left)
    right_leaves, right_tree = jax.tree.flatten(right)
    assert str(left_tree) == str(right_tree)
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        np.testing.assert_array_equal(np.asarray(left_leaf), np.asarray(right_leaf))


def _known_state(
    objectives: ComprehensiveStateObjectives,
) -> ComprehensiveStateObjectivesState:
    state = objectives.init(jr.key(7))
    zeros_action_vector = jnp.zeros((2,), dtype=jnp.float32)
    return dataclasses.replace(  # type: ignore[type-var]
        state,
        observation_weights=jnp.asarray(
            [[[0.0, 0.0], [0.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]]],
            dtype=jnp.float32,
        ),
        observation_bias=jnp.zeros((2, 2), dtype=jnp.float32),
        latent_weights=jnp.asarray(
            [[[0.0, 0.0], [0.0, 0.0]], [[0.5, 0.0], [0.0, 2.0]]],
            dtype=jnp.float32,
        ),
        latent_bias=jnp.zeros((2, 2), dtype=jnp.float32),
        reward_weights=jnp.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=jnp.float32),
        reward_bias=zeros_action_vector,
        termination_weights=jnp.asarray(
            [[0.0, 0.0], [0.0, 1.0]], dtype=jnp.float32
        ),
        termination_bias=zeros_action_vector,
        gvf_weights=jnp.asarray([[1.0, 0.0], [0.0, 2.0]], dtype=jnp.float32),
        value_weights=jnp.asarray([1.0, -1.0], dtype=jnp.float32),
        value_bias=jnp.asarray(0.0, dtype=jnp.float32),
        advantage_weights=jnp.asarray(
            [[0.0, 0.0], [0.5, 0.5]], dtype=jnp.float32
        ),
        advantage_bias=zeros_action_vector,
        inverse_current_weights=jnp.eye(2, dtype=jnp.float32),
        inverse_next_weights=jnp.float32(0.5) * jnp.eye(2, dtype=jnp.float32),
        inverse_bias=zeros_action_vector,
    )


def _cache_known(
    objectives: ComprehensiveStateObjectives,
    state: ComprehensiveStateObjectivesState,
    *,
    representation: tuple[float, float] = (2.0, 1.0),
    revision: tuple[int, int] = (0, 3),
) -> tuple[ComprehensiveStateObjectivesState, ComprehensiveStateObjectiveActionReceipt]:
    cached = objectives.cache_action(
        state,
        jnp.asarray(representation, dtype=jnp.float32),
        jnp.asarray(1, dtype=jnp.int32),
        jnp.asarray(revision, dtype=jnp.uint32),
    )
    assert bool(cached.cache_applied)
    return cached.state, cached.receipt


def _update_known(
    objectives: ComprehensiveStateObjectives,
    state: ComprehensiveStateObjectivesState,
    receipt: ComprehensiveStateObjectiveActionReceipt,
    **updates: Any,
):
    values: dict[str, Any] = {
        "next_representation": jnp.asarray([1.0, 3.0], dtype=jnp.float32),
        "next_representation_revision_words": jnp.asarray([0, 4], dtype=jnp.uint32),
        "next_observation_target": jnp.asarray([1.0, 3.0], dtype=jnp.float32),
        "reward_target": jnp.asarray(0.5, dtype=jnp.float32),
        "terminated_target": jnp.asarray(True, dtype=jnp.bool_),
        "cumulant": jnp.asarray(0.25, dtype=jnp.float32),
        "continuation": jnp.asarray(0.8, dtype=jnp.float32),
        "control_value_target": jnp.asarray(0.25, dtype=jnp.float32),
        "advantage_target": jnp.asarray(-0.5, dtype=jnp.float32),
    }
    values.update(updates)
    return objectives.update(state, receipt, **values)


def test_config_roundtrip_is_strict_json_safe_l0_and_explicitly_limited() -> None:
    config = _config(
        gvf_discounts=(0.0, 0.25, 0.9),
        prediction_group_weight=0.2,
        reward_group_weight=0.1,
        termination_group_weight=0.1,
        gvf_group_weight=0.2,
        control_group_weight=0.2,
        inverse_group_weight=0.2,
    )
    payload = config.to_config()
    restored = ComprehensiveStateObjectivesConfig.from_config(
        cast(dict[str, Any], json.loads(json.dumps(payload)))
    )
    assert restored == config
    assert payload["evidence_level"] == COMPREHENSIVE_STATE_OBJECTIVES_EVIDENCE_LEVEL == "L0"
    assert payload["outcome_status"] == COMPREHENSIVE_STATE_OBJECTIVES_OUTCOME_STATUS
    assert payload["outcome_status"] == "not_assessed"
    assert payload["ownership"] == COMPREHENSIVE_STATE_OBJECTIVES_OWNERSHIP
    assert payload["heads"] == list(COMPREHENSIVE_STATE_OBJECTIVES_HEADS)
    assert payload["limitations"] == list(COMPREHENSIVE_STATE_OBJECTIVES_LIMITATIONS)
    assert "caller-supplied-control-targets-without-off-policy-correction" in payload[
        "limitations"
    ]
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.n_actions = 3  # type: ignore[misc]

    for malformed in (
        {**payload, "unknown": 1},
        {name: value for name, value in payload.items() if name != "ownership"},
        {**payload, "evidence_level": "L2"},
        {**payload, "outcome_status": "accepted"},
        {**payload, "heads": []},
        {**payload, "limitations": []},
    ):
        with pytest.raises(ValueError):
            ComprehensiveStateObjectivesConfig.from_config(malformed)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"representation_dim": True},
        {"representation_dim": 0},
        {"observation_target_dim": 0},
        {"n_actions": 1},
        {"gvf_discounts": [0.0, 0.5]},
        {"gvf_discounts": (0.5,)},
        {"gvf_discounts": (0.5, 0.5)},
        {"gvf_discounts": (0.7, 0.2)},
        {"gvf_discounts": (0.0, 1.0)},
        {"observation_step_size": 0.0},
        {"termination_step_size": float("nan")},
        {"prediction_group_weight": 0.0, "reward_group_weight": 1.0 / 3.0},
        {"prediction_group_weight": 0.2},
        {"representation_gradient_clip": 0.0},
        {"max_abs_control_target": float("inf")},
    ],
)
def test_config_rejects_ambiguous_or_unbalanced_contracts(kwargs: dict[str, Any]) -> None:
    with pytest.raises((TypeError, ValueError)):
        _config(**kwargs)


def test_init_requires_typed_threefry_and_resource_counts_every_leaf() -> None:
    objectives = ComprehensiveStateObjectives(_config(gvf_discounts=(0.0, 0.5, 0.9)))
    state = objectives.init(jr.key(11))
    assert bool(objectives.state_valid(state))
    budget = objectives.resource_budget(state)
    assert budget.total_state_nbytes == measure_comprehensive_state_objectives_state_nbytes(
        state
    )
    assert budget.max_parameter_head_updates_per_transition == len(
        COMPREHENSIVE_STATE_OBJECTIVES_HEADS
    )
    assert budget.max_atomic_transactions_per_transition == 1
    assert budget.temporary_bytes_scope.endswith("not-a-measured-device-peak")
    with pytest.raises(dataclasses.FrozenInstanceError):
        state.pending_valid = jnp.asarray(True)
    with pytest.raises(TypeError, match="typed Threefry"):
        objectives.init(jr.PRNGKey(11))


def test_cache_and_update_bind_exact_owners_and_all_separate_head_revisions() -> None:
    objectives = ComprehensiveStateObjectives(_config())
    initial = _known_state(objectives)
    cached, receipt = _cache_known(objectives, initial)
    np.testing.assert_array_equal(receipt.representation, [2.0, 1.0])
    np.testing.assert_array_equal(receipt.representation_revision_words, [0, 3])
    np.testing.assert_array_equal(receipt.action_identity_words, [0, 1])
    assert int(receipt.action) == 1

    result = _update_known(objectives, cached, receipt)
    assert bool(result.update_applied)
    np.testing.assert_allclose(result.observation_prediction, [2.0, 1.0])
    np.testing.assert_allclose(result.latent_prediction, [1.0, 2.0])
    np.testing.assert_allclose(result.reward_prediction, 2.0)
    np.testing.assert_allclose(result.termination_probability, 1.0 / (1.0 + np.exp(-1.0)))
    np.testing.assert_allclose(result.gvf_predictions, [2.0, 2.0])
    np.testing.assert_allclose(result.gvf_targets, [0.25, 2.65], rtol=1e-6)
    np.testing.assert_allclose(result.value_prediction, 1.0)
    np.testing.assert_allclose(result.advantage_prediction, 1.5)
    np.testing.assert_allclose(result.inverse_probabilities, [0.5, 0.5])

    expected_prediction_loss = 0.5 * (1.25 + 0.25)
    expected_control_loss = 0.5 * (0.5 * 0.75**2 + 0.5 * 2.0**2)
    expected_gvf_loss = 0.5 * np.mean(np.square([1.75, -0.65]))
    expected_termination_loss = np.log1p(np.exp(1.0)) - 1.0
    expected_balanced = np.mean(
        [
            expected_prediction_loss,
            0.5 * 1.5**2,
            expected_termination_loss,
            expected_gvf_loss,
            expected_control_loss,
            np.log(2.0),
        ]
    )
    np.testing.assert_allclose(result.prediction_group_loss, expected_prediction_loss)
    np.testing.assert_allclose(result.control_group_loss, expected_control_loss)
    np.testing.assert_allclose(result.balanced_loss, expected_balanced, rtol=1e-6)
    np.testing.assert_allclose(
        result.prediction_current_representation_gradient, [0.25, -1.0]
    )
    np.testing.assert_allclose(result.prediction_next_representation_gradient, [0.0, 0.25])
    np.testing.assert_allclose(result.control_current_representation_gradient, [0.875, 0.125])
    np.testing.assert_allclose(
        result.current_representation_gradient,
        [2.0 / 3.0, (-1.0 - (1.0 / (1.0 + np.exp(1.0))) - 0.65 + 0.125 - 0.5) / 6.0],
        rtol=1e-6,
    )
    np.testing.assert_allclose(result.next_representation_gradient, [1.0 / 24.0, 0.0])
    np.testing.assert_array_equal(result.current_representation_revision_words, [0, 3])
    np.testing.assert_array_equal(result.next_representation_revision_words, [0, 4])
    np.testing.assert_array_equal(result.post_update_words, [0, 1])
    np.testing.assert_array_equal(
        result.post_head_revision_words,
        np.tile(
            np.asarray([0, 1], dtype=np.uint32),
            (len(COMPREHENSIVE_STATE_OBJECTIVES_HEADS), 1),
        ),
    )
    assert not bool(result.state.pending_valid)
    assert bool(objectives.state_valid(result.state))

    parameter_names = (
        "observation_weights",
        "latent_weights",
        "reward_weights",
        "termination_weights",
        "gvf_weights",
        "value_weights",
        "advantage_weights",
        "inverse_current_weights",
    )
    for name in parameter_names:
        assert not np.array_equal(
            np.asarray(getattr(result.state, name)),
            np.asarray(getattr(initial, name)),
        )
    np.testing.assert_array_equal(
        result.state.observation_weights[0], initial.observation_weights[0]
    )
    np.testing.assert_array_equal(result.state.reward_weights[0], initial.reward_weights[0])
    np.testing.assert_array_equal(result.state.advantage_weights[0], initial.advantage_weights[0])


def test_vector_width_and_gvf_head_count_are_mean_normalized() -> None:
    observation_gradients = []
    for observation_dim in (2, 4):
        objectives = ComprehensiveStateObjectives(
            _config(observation_target_dim=observation_dim)
        )
        state = objectives.init(jr.key(observation_dim))
        state = dataclasses.replace(  # type: ignore[type-var]
            state,
            observation_weights=state.observation_weights.at[1].set(
                jnp.tile(
                    jnp.asarray([[1.0, 0.0]], dtype=jnp.float32),
                    (observation_dim, 1),
                )
            ),
            observation_bias=jnp.zeros((2, observation_dim), dtype=jnp.float32),
            latent_weights=jnp.zeros((2, 2, 2), dtype=jnp.float32),
            latent_bias=jnp.zeros((2, 2), dtype=jnp.float32),
        )
        cached, receipt = _cache_known(
            objectives,
            state,
            representation=(1.0, 0.0),
            revision=(0, 0),
        )
        result = _update_known(
            objectives,
            cached,
            receipt,
            next_representation=jnp.zeros((2,), dtype=jnp.float32),
            next_representation_revision_words=jnp.zeros((2,), dtype=jnp.uint32),
            next_observation_target=jnp.zeros((observation_dim,), dtype=jnp.float32),
        )
        assert bool(result.update_applied)
        observation_gradients.append(result.prediction_current_representation_gradient)
    np.testing.assert_allclose(observation_gradients[0], [0.5, 0.0])
    np.testing.assert_allclose(observation_gradients[1], observation_gradients[0])

    gvf_gradients = []
    for discounts in ((0.0, 0.2), (0.0, 0.2, 0.4, 0.6)):
        objectives = ComprehensiveStateObjectives(_config(gvf_discounts=discounts))
        state = dataclasses.replace(  # type: ignore[type-var]
            objectives.init(jr.key(len(discounts))),
            gvf_weights=jnp.tile(
                jnp.asarray([[1.0, 0.0]], dtype=jnp.float32),
                (len(discounts), 1),
            ),
        )
        cached, receipt = _cache_known(
            objectives,
            state,
            representation=(1.0, 0.0),
            revision=(0, 0),
        )
        result = _update_known(
            objectives,
            cached,
            receipt,
            next_representation=jnp.zeros((2,), dtype=jnp.float32),
            next_representation_revision_words=jnp.zeros((2,), dtype=jnp.uint32),
            cumulant=jnp.asarray(0.0, dtype=jnp.float32),
            continuation=jnp.asarray(1.0, dtype=jnp.float32),
        )
        assert bool(result.update_applied)
        gvf_gradients.append(result.gvf_current_representation_gradient)
    np.testing.assert_allclose(gvf_gradients[0], [1.0, 0.0])
    np.testing.assert_allclose(gvf_gradients[1], gvf_gradients[0])


def test_balanced_current_and_successor_gradients_match_finite_differences() -> None:
    objectives = ComprehensiveStateObjectives(_config())
    state = _known_state(objectives)
    cached, receipt = _cache_known(objectives, state)
    result = _update_known(objectives, cached, receipt)
    assert bool(result.update_applied)

    observation_target = jnp.asarray([1.0, 3.0], dtype=jnp.float32)
    reward_target = jnp.float32(0.5)
    terminated_target = jnp.float32(1.0)
    cumulant = jnp.float32(0.25)
    continuation = jnp.float32(0.8)
    value_target = jnp.float32(0.25)
    advantage_target = jnp.float32(-0.5)
    action = 1
    discounts = jnp.asarray((0.0, 0.5), dtype=jnp.float32)
    reference_successor = jnp.asarray([1.0, 3.0], dtype=jnp.float32)
    # The GVF is a stopped-bootstrap semi-gradient.  Freeze its baseline
    # targets outside the perturbed objective so finite differences measure
    # the implemented surrogate derivative rather than a residual gradient.
    frozen_gvf_targets = cumulant + continuation * discounts * (
        state.gvf_weights @ reference_successor
    )

    def balanced_loss(current: jax.Array, successor: jax.Array) -> jax.Array:
        observation = (
            state.observation_weights[action] @ current + state.observation_bias[action]
        )
        observation_loss = 0.5 * jnp.mean(jnp.square(observation - observation_target))
        latent = state.latent_weights[action] @ current + state.latent_bias[action]
        latent_loss = 0.5 * jnp.mean(jnp.square(latent - successor))
        reward = state.reward_weights[action] @ current + state.reward_bias[action]
        reward_loss = 0.5 * jnp.square(reward - reward_target)
        termination_logit = (
            state.termination_weights[action] @ current + state.termination_bias[action]
        )
        termination_loss = jax.nn.softplus(termination_logit) - (
            terminated_target * termination_logit
        )
        predictions = state.gvf_weights @ current
        gvf_loss = 0.5 * jnp.mean(jnp.square(predictions - frozen_gvf_targets))
        value = state.value_weights @ current + state.value_bias
        value_loss = 0.5 * jnp.square(value - value_target)
        advantage = (
            state.advantage_weights[action] @ current + state.advantage_bias[action]
        )
        advantage_loss = 0.5 * jnp.square(advantage - advantage_target)
        inverse_logits = (
            state.inverse_current_weights @ current
            + state.inverse_next_weights @ successor
            + state.inverse_bias
        )
        inverse_loss = -jax.nn.log_softmax(inverse_logits)[action]
        prediction_loss = 0.5 * (observation_loss + latent_loss)
        control_loss = 0.5 * (value_loss + advantage_loss)
        return (
            prediction_loss
            + reward_loss
            + termination_loss
            + gvf_loss
            + control_loss
            + inverse_loss
        ) / 6.0

    current = jnp.asarray([2.0, 1.0], dtype=jnp.float32)
    successor = reference_successor
    epsilon = jnp.float32(1.0e-3)

    def central_difference(argument: int, index: int) -> float:
        basis = jnp.zeros((2,), dtype=jnp.float32).at[index].set(epsilon)
        if argument == 0:
            positive = balanced_loss(current + basis, successor)
            negative = balanced_loss(current - basis, successor)
        else:
            positive = balanced_loss(current, successor + basis)
            negative = balanced_loss(current, successor - basis)
        return float((positive - negative) / (2.0 * epsilon))

    finite_current = np.asarray([central_difference(0, index) for index in range(2)])
    finite_successor = np.asarray([central_difference(1, index) for index in range(2)])
    np.testing.assert_allclose(
        result.current_representation_gradient,
        finite_current,
        rtol=2e-3,
        atol=2e-4,
    )
    np.testing.assert_allclose(
        result.next_representation_gradient,
        finite_successor,
        rtol=2e-3,
        atol=2e-4,
    )


@pytest.mark.parametrize(
    "failure",
    ["receipt_bits", "successor", "revision", "observation", "reward", "continuation"],
)
def test_invalid_update_is_bit_exact_noop_and_preserves_pending_retry(failure: str) -> None:
    objectives = ComprehensiveStateObjectives(_config(max_abs_reward_target=2.0))
    cached, receipt = _cache_known(
        objectives,
        _known_state(objectives),
        representation=(0.0, 1.0),
        revision=(1, 0),
    )
    updates: dict[str, Any] = {
        "next_representation_revision_words": jnp.asarray([1, 0], dtype=jnp.uint32)
    }
    if failure == "receipt_bits":
        receipt = dataclasses.replace(  # type: ignore[type-var]
            receipt,
            representation=jnp.asarray([-0.0, 1.0], dtype=jnp.float32),
        )
    elif failure == "successor":
        updates["next_representation"] = jnp.asarray([jnp.inf, 3.0], dtype=jnp.float32)
    elif failure == "revision":
        updates["next_representation_revision_words"] = jnp.asarray(
            [0, 2**32 - 1], dtype=jnp.uint32
        )
    elif failure == "observation":
        updates["next_observation_target"] = jnp.asarray([jnp.nan, 0.0], dtype=jnp.float32)
    elif failure == "reward":
        updates["reward_target"] = jnp.asarray(3.0, dtype=jnp.float32)
    else:
        updates["continuation"] = jnp.asarray(1.1, dtype=jnp.float32)
    result = _update_known(objectives, cached, receipt, **updates)
    assert not bool(result.update_applied)
    _assert_tree_equal(result.state, cached)
    for gradient_name in (
        "prediction_current_representation_gradient",
        "prediction_next_representation_gradient",
        "reward_current_representation_gradient",
        "termination_current_representation_gradient",
        "gvf_current_representation_gradient",
        "control_current_representation_gradient",
        "inverse_current_representation_gradient",
        "inverse_next_representation_gradient",
        "current_representation_gradient",
        "next_representation_gradient",
    ):
        np.testing.assert_array_equal(getattr(result, gradient_name), [0.0, 0.0])
    assert np.isfinite(float(result.balanced_loss))
    assert bool(result.state.pending_valid)


def test_same_revision_is_accepted_and_non_earlier_revision_is_rejected() -> None:
    objectives = ComprehensiveStateObjectives(_config())
    initial = _known_state(objectives)
    cached, receipt = _cache_known(objectives, initial, revision=(2, 7))
    same = _update_known(
        objectives,
        cached,
        receipt,
        next_representation_revision_words=jnp.asarray([2, 7], dtype=jnp.uint32),
    )
    assert bool(same.representation_revision_valid)
    assert bool(same.update_applied)

    cached_retry, retry_receipt = _cache_known(objectives, initial, revision=(2, 7))
    earlier = _update_known(
        objectives,
        cached_retry,
        retry_receipt,
        next_representation_revision_words=jnp.asarray([2, 6], dtype=jnp.uint32),
    )
    assert not bool(earlier.representation_revision_valid)
    assert not bool(earlier.update_applied)
    _assert_tree_equal(earlier.state, cached_retry)
    np.testing.assert_array_equal(earlier.current_representation_gradient, [0.0, 0.0])
    np.testing.assert_array_equal(earlier.next_representation_gradient, [0.0, 0.0])


def test_exact_clock_fail_stop_and_corrupt_head_revision_reject_atomically() -> None:
    objectives = ComprehensiveStateObjectives(_config())
    maximum = jnp.asarray([2**32 - 1, 2**32 - 1], dtype=jnp.uint32)
    saturated = dataclasses.replace(  # type: ignore[type-var]
        objectives.init(jr.key(13)),
        decision_words=maximum,
        update_words=maximum,
        head_revision_words=jnp.tile(
            maximum[None, :],
            (len(COMPREHENSIVE_STATE_OBJECTIVES_HEADS), 1),
        ),
    )
    assert bool(objectives.state_valid(saturated))
    rejected = objectives.cache_action(
        saturated,
        jnp.asarray([1.0, 0.0], dtype=jnp.float32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.zeros((2,), dtype=jnp.uint32),
    )
    assert not bool(rejected.lifetime_capacity_available)
    assert not bool(rejected.cache_applied)
    _assert_tree_equal(rejected.state, saturated)

    cached, receipt = _cache_known(objectives, _known_state(objectives))
    corrupt = dataclasses.replace(  # type: ignore[type-var]
        cached,
        head_revision_words=cached.head_revision_words.at[3, 1].set(jnp.uint32(1)),
    )
    assert not bool(objectives.state_valid(corrupt))
    result = _update_known(objectives, corrupt, receipt)
    assert not bool(result.update_applied)
    _assert_tree_equal(result.state, corrupt)
    np.testing.assert_array_equal(result.current_representation_gradient, [0.0, 0.0])
    np.testing.assert_array_equal(result.next_representation_gradient, [0.0, 0.0])


def test_termination_target_is_bool_and_independent_from_gvf_continuation() -> None:
    objectives = ComprehensiveStateObjectives(_config())
    cached, receipt = _cache_known(objectives, _known_state(objectives))
    with pytest.raises(TypeError):
        _update_known(
            objectives,
            cached,
            receipt,
            terminated_target=jnp.asarray(1.0, dtype=jnp.float32),
        )
    result = _update_known(
        objectives,
        cached,
        receipt,
        terminated_target=jnp.asarray(True, dtype=jnp.bool_),
        continuation=jnp.asarray(1.0, dtype=jnp.float32),
    )
    assert bool(result.update_applied)
    np.testing.assert_allclose(result.termination_probability, 1.0 / (1.0 + np.exp(-1.0)))
