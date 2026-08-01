# mypy: disable-error-code="call-arg"
"""L0 mechanism contracts for the shallow recursive-ridge world-model reference."""

from __future__ import annotations

import copy
import inspect
import json
from collections.abc import Callable, Mapping
from typing import Any, cast

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from alberta_framework.core.shallow_ridge_world_model import (
    EVIDENCE_LEVEL,
    SCIENTIFIC_PROMOTION_ALLOWED,
    ShallowRidgeWorldModel,
    ShallowRidgeWorldModelConfig,
    ShallowRidgeWorldModelState,
    run_shallow_ridge_world_model,
)

pytestmark = pytest.mark.unit


def _model(**overrides: Any) -> ShallowRidgeWorldModel:
    values: dict[str, object] = {
        "observation_dim": 1,
        "n_actions": 2,
        "ridge": 0.1,
        "max_updates": 1_000,
        "max_input_magnitude": 100.0,
        "max_statistic_magnitude": 100_000.0,
        "max_parameter_magnitude": 10_000.0,
        "max_prediction_magnitude": 10_000.0,
    }
    values.update(overrides)
    return ShallowRidgeWorldModel(ShallowRidgeWorldModelConfig(**values))  # type: ignore[arg-type]


def _transition(
    *,
    action: int = 1,
    observation: float = 2.0,
    next_observation: float = 5.0,
    reward: float = 3.0,
    continuation: float = 0.5,
) -> tuple[jax.Array, ...]:
    return (
        jnp.asarray([observation], dtype=jnp.float32),
        jnp.asarray(action, dtype=jnp.int32),
        jnp.asarray([next_observation], dtype=jnp.float32),
        jnp.asarray(reward, dtype=jnp.float32),
        jnp.asarray(continuation, dtype=jnp.float32),
    )


def _update(
    model: ShallowRidgeWorldModel,
    state: ShallowRidgeWorldModelState,
    **overrides: Any,
) -> Any:
    return model.update(state, *_transition(**overrides))


def _tree_bytes(tree: object) -> tuple[bytes, ...]:
    return tuple(np.asarray(leaf).tobytes() for leaf in jax.tree_util.tree_leaves(tree))


def _learning_stream(repetitions: int = 40) -> tuple[jax.Array, ...]:
    actions = jnp.tile(jnp.asarray([0, 1], dtype=jnp.int32), repetitions)
    observations = jnp.ones((2 * repetitions, 1), dtype=jnp.float32)
    action_zero = actions == 0
    next_observations = jnp.where(action_zero[:, None], 2.0, -1.0).astype(jnp.float32)
    rewards = jnp.where(action_zero, 1.0, -2.0).astype(jnp.float32)
    continuations = jnp.where(action_zero, 0.9, 0.0).astype(jnp.float32)
    return observations, actions, next_observations, rewards, continuations


_BASE_MODEL = _model()


@pytest.fixture(scope="module")
def learned_reference() -> tuple[ShallowRidgeWorldModel, ShallowRidgeWorldModelState, Any]:
    model = _BASE_MODEL
    result = run_shallow_ridge_world_model(model, model.init(), *_learning_stream())
    return model, result.state, result


@pytest.mark.parametrize(
    "overrides",
    (
        {"observation_dim": True},
        {"observation_dim": 0},
        {"n_actions": 0},
        {"ridge": 0.0},
        {"ridge": float("nan")},
        {"max_updates": True},
        {"max_updates": 0},
        {"max_updates": 2**24 + 1},
        {"max_input_magnitude": float("inf")},
        {"max_statistic_magnitude": 0.5},
        {"max_parameter_magnitude": -1.0},
        {"max_prediction_magnitude": 0.0},
        {"observation_dim": 100_000, "n_actions": 100_000},
    ),
)
def test_config_rejects_ambiguous_unbounded_or_nonfinite_values(
    overrides: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        _model(**overrides)


def test_config_is_exact_and_initialization_is_unique_rng_free_l0() -> None:
    model = _BASE_MODEL
    payload = model.to_config()
    restored = ShallowRidgeWorldModel.from_config(payload)

    assert restored.config == model.config
    assert restored.to_config() == payload
    assert EVIDENCE_LEVEL == "L0"
    assert SCIENTIFIC_PROMOTION_ALLOWED is False
    assert tuple(inspect.signature(model.init).parameters) == ()
    chex.assert_trees_all_equal(model.init(), model.init())

    extra = copy.deepcopy(payload)
    extra["seed"] = 0
    with pytest.raises(ValueError, match="fields"):
        ShallowRidgeWorldModel.from_config(extra)
    wrong_numeric_type = copy.deepcopy(payload)
    nested = cast(dict[str, object], wrong_numeric_type["config"])
    nested["ridge"] = 1
    with pytest.raises(ValueError, match="scalar types"):
        ShallowRidgeWorldModel.from_config(wrong_numeric_type)


def test_psd_validation_broadcasts_over_actions_not_feature_eigenvalues() -> None:
    model = _model(observation_dim=2, n_actions=3)
    state = model.init()
    assert bool(model.state_valid(state))
    result = model.update(
        state,
        jnp.asarray([0.25, -0.5], dtype=jnp.float32),
        jnp.asarray(2, dtype=jnp.int32),
        jnp.asarray([0.1, 0.2], dtype=jnp.float32),
        jnp.asarray(0.3, dtype=jnp.float32),
        jnp.asarray(0.9, dtype=jnp.float32),
    )
    assert bool(result.diagnostics.applied)
    assert bool(model.state_valid(result.state))


def test_cold_prediction_is_valid_zero_and_read_only() -> None:
    model = _BASE_MODEL
    state = model.init()
    before = _tree_bytes(state)
    prediction = model.predict(
        state,
        jnp.asarray([0.25], dtype=jnp.float32),
        jnp.asarray(0, dtype=jnp.int32),
    )

    assert bool(model.state_valid(state))
    assert bool(prediction.valid)
    assert int(prediction.action) == 0
    chex.assert_trees_all_equal(prediction.features, jnp.asarray([0.25, 1.0]))
    chex.assert_trees_all_equal(prediction.raw_outputs, jnp.zeros((3,), dtype=jnp.float32))
    assert float(prediction.reward) == 0.0
    assert float(prediction.continuation) == 0.0
    assert _tree_bytes(state) == before


def test_one_update_matches_hand_calculated_regularized_leader_and_is_prequential() -> None:
    model = _BASE_MODEL
    state = model.init()
    before = model.predict(state, _transition()[0], _transition()[1])
    result = _update(model, state)
    features = np.asarray([2.0, 1.0], dtype=np.float32)
    targets = np.asarray([5.0, 3.0, 0.5], dtype=np.float32)
    expected_gram = np.outer(features, features)
    expected_cross = np.outer(features, targets)
    expected_weights = np.linalg.solve(
        expected_gram + model.config.ridge * np.eye(2, dtype=np.float32),
        expected_cross,
    )

    assert bool(result.diagnostics.applied)
    chex.assert_trees_all_equal(result.prediction, before)
    np.testing.assert_array_equal(result.targets, targets)
    np.testing.assert_array_equal(result.errors, targets)
    assert float(result.squared_error) == pytest.approx(float(np.mean(targets**2)))
    np.testing.assert_array_equal(result.state.gram[1], expected_gram)
    np.testing.assert_array_equal(result.state.cross[1], expected_cross)
    np.testing.assert_allclose(result.state.weights[1], expected_weights, atol=2.0e-7)
    chex.assert_trees_all_equal(result.state.gram[0], state.gram[0])
    chex.assert_trees_all_equal(result.state.cross[0], state.cross[0])
    chex.assert_trees_all_equal(result.state.weights[0], state.weights[0])
    chex.assert_trees_all_equal(result.state.action_counts, jnp.asarray([0, 1]))
    assert int(result.state.update_count) == 1
    assert bool(model.state_valid(result.state))


def test_target_cannot_leak_into_the_preupdate_prediction() -> None:
    model = _BASE_MODEL
    state = model.init()
    first = _update(model, state, next_observation=5.0, reward=3.0, continuation=1.0)
    second = _update(model, state, next_observation=-4.0, reward=-2.0, continuation=0.0)

    chex.assert_trees_all_equal(first.prediction, second.prediction)
    assert not bool(jnp.array_equal(first.state.cross, second.state.cross))
    after_first = model.predict(first.state, _transition()[0], _transition()[1])
    after_second = model.predict(second.state, _transition()[0], _transition()[1])
    assert not bool(jnp.array_equal(after_first.raw_outputs, after_second.raw_outputs))


def test_online_learning_is_action_conditioned_and_handles_continuing_and_terminal_targets(
    learned_reference: tuple[ShallowRidgeWorldModel, ShallowRidgeWorldModelState, Any],
) -> None:
    model, state, learning = learned_reference
    observation = jnp.asarray([1.0], dtype=jnp.float32)
    continuing = model.predict(state, observation, jnp.asarray(0, dtype=jnp.int32))
    terminal = model.predict(state, observation, jnp.asarray(1, dtype=jnp.int32))

    np.testing.assert_allclose(continuing.next_observation, [2.0], atol=3.0e-3)
    assert float(continuing.reward) == pytest.approx(1.0, abs=2.0e-3)
    assert float(continuing.continuation) == pytest.approx(0.9, abs=2.0e-3)
    np.testing.assert_allclose(terminal.next_observation, [-1.0], atol=2.0e-3)
    assert float(terminal.reward) == pytest.approx(-2.0, abs=3.0e-3)
    assert float(terminal.continuation) == 0.0
    assert int(state.update_count) == 80
    np.testing.assert_array_equal(state.action_counts, [40, 40])
    assert bool(jnp.all(learning.applied))
    assert float(jnp.mean(learning.squared_errors[-10:])) < 0.01 * float(
        jnp.mean(learning.squared_errors[:10])
    )


def test_one_step_planning_scores_every_action_from_supplied_linear_successor_value(
    learned_reference: tuple[ShallowRidgeWorldModel, ShallowRidgeWorldModelState, Any],
) -> None:
    model, state, _ = learned_reference
    observation = jnp.asarray([1.0], dtype=jnp.float32)
    value_weights = jnp.asarray([2.0], dtype=jnp.float32)
    value_bias = jnp.asarray(0.5, dtype=jnp.float32)
    before = _tree_bytes(state)
    plan = model.score_actions(state, observation, value_weights, value_bias)

    assert bool(plan.valid)
    np.testing.assert_array_equal(plan.actions, [0, 1])
    for action in range(model.config.n_actions):
        prediction = model.predict(
            state,
            observation,
            jnp.asarray(action, dtype=jnp.int32),
        )
        expected_successor = float(prediction.next_observation @ value_weights + value_bias)
        expected_score = float(prediction.reward) + float(prediction.continuation) * (
            expected_successor
        )
        assert float(plan.successor_values[action]) == pytest.approx(expected_successor)
        assert float(plan.scores[action]) == pytest.approx(expected_score)
    assert int(plan.best_action) == int(jnp.argmax(plan.scores))
    assert _tree_bytes(state) == before


@pytest.mark.parametrize(
    "mutation",
    (
        lambda values: values.__setitem__(0, values[0].at[0].set(jnp.nan)),
        lambda values: values.__setitem__(1, jnp.asarray(2, dtype=jnp.int32)),
        lambda values: values.__setitem__(2, values[2].at[0].set(jnp.inf)),
        lambda values: values.__setitem__(4, jnp.asarray(1.2, dtype=jnp.float32)),
    ),
)
def test_invalid_dynamic_transition_is_atomic_and_returns_no_usable_signal(
    mutation: Callable[[list[jax.Array]], None],
) -> None:
    model = _BASE_MODEL
    state = model.init()
    values = list(_transition())
    mutation(values)
    result = model.update(state, *values)

    assert _tree_bytes(result.state) == _tree_bytes(state)
    assert not bool(result.diagnostics.applied)
    assert bool(result.diagnostics.rejected)
    assert not bool(result.prediction.valid)
    chex.assert_trees_all_equal(result.targets, jnp.zeros((3,), dtype=jnp.float32))
    chex.assert_trees_all_equal(result.errors, jnp.zeros((3,), dtype=jnp.float32))
    assert float(result.squared_error) == 0.0


def test_static_shape_and_dtype_errors_raise_before_dynamic_execution() -> None:
    model = _BASE_MODEL
    state = model.init()
    with pytest.raises(ValueError, match="observation"):
        model.predict(
            state,
            jnp.asarray([1.0], dtype=jnp.float16),
            jnp.asarray(0, dtype=jnp.int32),
        )
    with pytest.raises(ValueError, match="action"):
        model.predict(
            state,
            jnp.asarray([1.0], dtype=jnp.float32),
            jnp.asarray([0], dtype=jnp.int32),
        )
    with pytest.raises(ValueError, match="continuation"):
        model.update(
            state,
            *_transition()[:-1],
            jnp.asarray([0.5], dtype=jnp.float32),
        )
    wrong_state = state.replace(  # type: ignore[attr-defined]
        update_count=jnp.asarray(0.0, dtype=jnp.float32)
    )
    with pytest.raises(ValueError, match="state.update_count"):
        model.state_valid(wrong_state)


def test_corruption_capacity_and_candidate_bound_fail_closed_byte_exactly() -> None:
    model = _BASE_MODEL
    initial = model.init()
    asymmetric = initial.replace(  # type: ignore[attr-defined]
        gram=initial.gram.at[0, 0, 1].set(0.25)
    )
    mismatched_count = initial.replace(  # type: ignore[attr-defined]
        action_counts=initial.action_counts.at[0].set(1)
    )
    indefinite_gram = initial.replace(  # type: ignore[attr-defined]
        gram=initial.gram.at[0].set(
            jnp.asarray([[1.0, 2.0], [2.0, 1.0]], dtype=jnp.float32)
        ),
        action_counts=initial.action_counts.at[0].set(1),
        update_count=jnp.asarray(1, dtype=jnp.int32),
    )
    learned = _update(model, initial).state
    inconsistent_weight = learned.replace(weights=learned.weights.at[1, 0, 0].add(0.5))

    for corrupt in (
        asymmetric,
        mismatched_count,
        indefinite_gram,
        inconsistent_weight,
    ):
        assert not bool(model.state_valid(corrupt))
        prediction = model.predict(
            corrupt,
            jnp.asarray([1.0], dtype=jnp.float32),
            jnp.asarray(0, dtype=jnp.int32),
        )
        result = _update(model, corrupt)
        assert not bool(prediction.valid)
        assert not bool(result.diagnostics.state_valid)
        assert _tree_bytes(result.state) == _tree_bytes(corrupt)

    full = initial.replace(  # type: ignore[attr-defined]
        gram=initial.gram.at[0, -1, -1].set(float(model.config.max_updates)),
        action_counts=initial.action_counts.at[0].set(model.config.max_updates),
        update_count=jnp.asarray(model.config.max_updates, dtype=jnp.int32),
    )
    assert bool(model.state_valid(full))
    exhausted = _update(model, full)
    assert bool(exhausted.diagnostics.state_valid)
    assert not bool(exhausted.diagnostics.capacity_available)
    assert _tree_bytes(exhausted.state) == _tree_bytes(full)

    near_statistic_bound = initial.replace(  # type: ignore[attr-defined]
        gram=initial.gram.at[1, 0, 0]
        .set(model.config.max_statistic_magnitude)
        .at[1, -1, -1]
        .set(10.0),
        action_counts=initial.action_counts.at[1].set(10),
        update_count=jnp.asarray(10, dtype=jnp.int32),
    )
    assert bool(model.state_valid(near_statistic_bound))
    rejected = _update(
        model,
        near_statistic_bound,
        observation=1.0,
        next_observation=2.0,
        reward=2.0,
        continuation=1.0,
    )
    assert bool(rejected.diagnostics.state_valid)
    assert not bool(rejected.diagnostics.statistics_valid)
    assert not bool(rejected.diagnostics.applied)
    assert _tree_bytes(rejected.state) == _tree_bytes(near_statistic_bound)


def test_eager_jit_sequential_and_scan_paths_have_parity() -> None:
    model = _BASE_MODEL
    stream = tuple(value[:6] for value in _learning_stream(repetitions=3))
    initial = model.init()
    transition = tuple(value[0] for value in stream)
    with jax.disable_jit():
        eager = model.update(initial, *transition)
    compiled = jax.jit(model.update)(initial, *transition)
    chex.assert_trees_all_close(eager, compiled, atol=1.0e-6, rtol=1.0e-6)

    sequential_state = initial
    sequential_predictions = []
    for index in range(6):
        result = model.update(sequential_state, *(value[index] for value in stream))
        sequential_state = result.state
        sequential_predictions.append(result.prediction.raw_outputs)
    scan = run_shallow_ridge_world_model(model, initial, *stream)
    compiled_scan = jax.jit(lambda current: run_shallow_ridge_world_model(model, current, *stream))(
        initial
    )

    chex.assert_trees_all_close(scan, compiled_scan, atol=1.0e-6, rtol=1.0e-6)
    chex.assert_trees_all_close(scan.state, sequential_state, atol=1.0e-6, rtol=1.0e-6)
    expected_raw = jnp.stack(sequential_predictions)
    observed_raw = jnp.concatenate(
        (
            scan.next_observation_predictions,
            scan.reward_predictions[:, None],
            scan.continuation_predictions[:, None],
        ),
        axis=1,
    )
    chex.assert_trees_all_close(observed_raw, expected_raw, atol=1.0e-6, rtol=1.0e-6)
    assert bool(jnp.all(scan.applied))


def test_checkpoint_is_strict_exact_and_resume_has_no_rng_dependency() -> None:
    model = _BASE_MODEL
    first_stream = tuple(value[:6] for value in _learning_stream(repetitions=3))
    state = run_shallow_ridge_world_model(model, model.init(), *first_stream).state
    payload = json.loads(json.dumps(model.checkpoint_payload(state)))
    restored_model, restored_state = ShallowRidgeWorldModel.from_checkpoint_payload(payload)

    assert restored_model.config == model.config
    assert restored_model.checkpoint_payload(restored_state) == payload
    metadata = cast(Mapping[str, object], payload["metadata"])
    assert metadata["rng_state_nbytes"] == 0
    assert "rng" not in cast(Mapping[str, object], payload["state"])
    with jax.disable_jit():
        original_next = _update(
            model,
            state,
            observation=-0.5,
            action=0,
            next_observation=0.25,
        )
        restored_next = _update(
            restored_model,
            restored_state,
            observation=-0.5,
            action=0,
            next_observation=0.25,
        )
    chex.assert_trees_all_equal(original_next, restored_next)

    malformed = copy.deepcopy(payload)
    malformed["extra"] = True
    with pytest.raises(ValueError, match="fields"):
        ShallowRidgeWorldModel.from_checkpoint_payload(malformed)
    dishonest = copy.deepcopy(payload)
    cast(dict[str, object], dishonest["metadata"])["rng_state_nbytes"] = 4
    with pytest.raises(ValueError, match="RNG-free"):
        ShallowRidgeWorldModel.from_checkpoint_payload(dishonest)
    nonfinite = copy.deepcopy(payload)
    state_payload = cast(dict[str, object], nonfinite["state"])
    gram = cast(list[list[list[float]]], state_payload["gram"])
    gram[0][0][0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        ShallowRidgeWorldModel.from_checkpoint_payload(nonfinite)
    boolean_count = copy.deepcopy(payload)
    state_payload = cast(dict[str, object], boolean_count["state"])
    counts = cast(list[object], state_payload["action_counts"])
    counts[0] = True
    with pytest.raises(ValueError, match="integers"):
        ShallowRidgeWorldModel.from_checkpoint_payload(boolean_count)
    inconsistent = copy.deepcopy(payload)
    state_payload = cast(dict[str, object], inconsistent["state"])
    weights = cast(list[list[list[float]]], state_payload["weights"])
    weights[0][0][0] += 0.5
    with pytest.raises(ValueError, match="inconsistent"):
        ShallowRidgeWorldModel.from_checkpoint_payload(inconsistent)


def test_planning_invalid_reference_fails_closed_without_mutation(
    learned_reference: tuple[ShallowRidgeWorldModel, ShallowRidgeWorldModelState, Any],
) -> None:
    model, state, _ = learned_reference
    before = _tree_bytes(state)
    result = model.score_actions(
        state,
        jnp.asarray([1.0], dtype=jnp.float32),
        jnp.asarray([jnp.nan], dtype=jnp.float32),
        jnp.asarray(0.0, dtype=jnp.float32),
    )

    assert not bool(result.valid)
    assert int(result.best_action) == -1
    chex.assert_trees_all_equal(result.scores, jnp.zeros((2,), dtype=jnp.float32))
    assert _tree_bytes(state) == before


def test_resource_budget_exactly_matches_fixed_state_and_call_surfaces() -> None:
    model = _BASE_MODEL
    initial = model.init()
    budget = model.resource_budget
    actual_nbytes = sum(int(leaf.nbytes) for leaf in jax.tree_util.tree_leaves(initial))
    feature_dim = 2
    target_dim = 3
    expected_gram = 2 * feature_dim * feature_dim
    expected_cross = 2 * feature_dim * target_dim
    expected_weights = expected_cross
    expected_admin = 3

    assert budget.gram_float32_scalars == expected_gram
    assert budget.cross_float32_scalars == expected_cross
    assert budget.cached_weight_float32_scalars == expected_weights
    assert budget.administrative_int32_scalars == expected_admin
    assert budget.state_nbytes == model.config.state_nbytes == actual_nbytes
    assert actual_nbytes == 4 * (expected_gram + expected_cross + expected_weights + expected_admin)
    assert budget.selected_gram_float32_scalars_touched_per_update == feature_dim**2
    assert budget.selected_cross_float32_scalars_touched_per_update == feature_dim * target_dim
    assert budget.selected_weight_float32_scalars_solved_per_update == feature_dim * target_dim
    assert budget.administrative_int32_scalars_touched_per_update == 2
    assert budget.action_predictions_per_planning_call == 2
    assert budget.successor_value_evaluations_per_planning_call == 2
    assert budget.max_updates == 1_000
    assert budget.state_growth_nbytes_per_transition == 0
    assert budget.replay_capacity == 0
    assert budget.rng_state_nbytes == 0
    assert budget.to_dict()["state_nbytes"] == actual_nbytes

    result = _update(model, initial)
    assert bool(result.diagnostics.applied)
    assert (
        sum(int(leaf.nbytes) for leaf in jax.tree_util.tree_leaves(result.state)) == actual_nbytes
    )
    assert jax.tree.map(lambda leaf: leaf.shape, result.state) == jax.tree.map(
        lambda leaf: leaf.shape,
        initial,
    )
