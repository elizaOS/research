"""Tests for the representation-conditioned joint-action world model."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from alberta_framework.core.grounded_joint_world_model import (
    GROUNDED_JOINT_WORLD_STATE_SCHEMA,
    GroundedJointWorldModel,
    GroundedJointWorldModelConfig,
    grounded_joint_world_lifetime_counter_nbytes,
    measure_grounded_joint_world_state_nbytes,
    migrate_legacy_grounded_joint_world_state,
)

V6_REPRESENTATION_LOSS_WEIGHTS = (
    0.0,
    10.0 / 3.0,
    0.0,
    0.0,
    0.0,
    0.0,
    10.0 / 3.0,
    10.0 / 3.0,
    0.0,
    0.0,
)


def _model(**overrides: Any) -> GroundedJointWorldModel:
    values: dict[str, Any] = {
        "representation_dim": 4,
        "target_observation_dim": 3,
        "n_focal_actions": 2,
        "n_partner_actions": 3,
        "step_size": 0.4,
        "initialization_scale": 0.08,
        "max_input_magnitude": 100.0,
        "max_parameter_magnitude": 100.0,
    }
    values.update(overrides)
    return GroundedJointWorldModel(GroundedJointWorldModelConfig(**values))


def _transition() -> tuple[jax.Array, ...]:
    return (
        jnp.array([0.75, -0.25, 0.4, -0.6], dtype=jnp.float32),
        jnp.array(1, dtype=jnp.int32),
        jnp.array(2, dtype=jnp.int32),
        jnp.array([0.2, -0.7, 0.5], dtype=jnp.float32),
        jnp.array(0.6, dtype=jnp.float32),
        jnp.array(0.9, dtype=jnp.float32),
    )


def _call_update(model: GroundedJointWorldModel, state: Any) -> Any:
    representation, focal, partner, next_observation, reward, discount = _transition()
    return model.update(
        state,
        representation,
        focal,
        partner,
        next_observation,
        reward,
        discount,
    )


def test_joint_action_index_and_one_hot_are_exact_cartesian_encoding() -> None:
    model = _model()
    observed = []
    for focal_action in range(2):
        for partner_action in range(3):
            focal = jnp.array(focal_action, dtype=jnp.int32)
            partner = jnp.array(partner_action, dtype=jnp.int32)
            index = model.joint_action_index(focal, partner)
            one_hot = model.joint_action_one_hot(focal, partner)
            expected = focal_action * 3 + partner_action
            assert int(index) == expected
            np.testing.assert_array_equal(
                np.asarray(one_hot), np.eye(6, dtype=np.float32)[expected]
            )
            observed.append(int(index))

    assert observed == list(range(6))
    assert model.joint_action_count == 6


def test_real_online_learning_reduces_preupdate_loss_for_one_joint_action() -> None:
    model = _model()
    state = model.init(jax.random.key(11))
    initial = _call_update(model, state)
    selected = int(initial.prediction.joint_action_index)
    unselected_weights = state.weights.at[selected].get(mode="promise_in_bounds")

    def learn(_: int, carry: Any) -> Any:
        return _call_update(model, carry).state

    learned_state = jax.lax.fori_loop(0, 100, learn, state)
    final = _call_update(model, learned_state)

    assert bool(initial.diagnostics.applied)
    assert bool(final.diagnostics.applied)
    assert float(final.loss) < 0.01 * float(initial.loss)
    assert int(learned_state.update_count) == 100
    for index in range(model.joint_action_count):
        if index != selected:
            chex.assert_trees_all_equal(learned_state.weights[index], state.weights[index])
            chex.assert_trees_all_equal(learned_state.bias[index], state.bias[index])
    assert unselected_weights.shape == (model.target_dim, model.config.representation_dim)


def test_input_gradient_is_preupdate_finite_nonzero_and_matches_finite_difference() -> None:
    model = _model()
    state = model.init(jax.random.key(3))
    representation, focal, partner, next_observation, reward, discount = _transition()
    score = model.input_loss_gradient(
        state,
        representation,
        focal,
        partner,
        next_observation,
        reward,
        discount,
    )

    assert bool(score.valid)
    assert bool(jnp.all(jnp.isfinite(score.gradient)))
    assert float(score.gradient_norm) > 0.0

    epsilon = 1e-3
    finite_difference = []
    for coordinate in range(model.config.representation_dim):
        direction = jnp.zeros_like(representation).at[coordinate].set(epsilon)
        plus = model.input_loss_gradient(
            state,
            representation + direction,
            focal,
            partner,
            next_observation,
            reward,
            discount,
        ).loss
        minus = model.input_loss_gradient(
            state,
            representation - direction,
            focal,
            partner,
            next_observation,
            reward,
            discount,
        ).loss
        finite_difference.append((plus - minus) / (2.0 * epsilon))
    chex.assert_trees_all_close(
        score.gradient,
        jnp.stack(finite_difference),
        atol=2e-4,
        rtol=2e-3,
    )


def test_grounded_targets_are_stopped_for_representation_learning_signal() -> None:
    model = _model()
    state = model.init(jax.random.key(5))
    representation, focal, partner, next_observation, reward, discount = _transition()

    def objective(
        next_obs: jax.Array, reward_target: jax.Array, discount_target: jax.Array
    ) -> jax.Array:
        return model.input_loss_gradient(
            state,
            representation,
            focal,
            partner,
            next_obs,
            reward_target,
            discount_target,
        ).loss

    next_obs_gradient, reward_gradient, discount_gradient = jax.grad(
        objective,
        argnums=(0, 1, 2),
    )(
        next_observation,
        reward,
        discount,
    )
    chex.assert_trees_all_equal(next_obs_gradient, jnp.zeros_like(next_observation))
    chex.assert_trees_all_equal(reward_gradient, jnp.zeros_like(reward))
    chex.assert_trees_all_equal(discount_gradient, jnp.zeros_like(discount))


def test_update_reports_prediction_and_loss_from_preupdate_parameters() -> None:
    model = _model(step_size=1.0)
    state = model.init(jax.random.key(17))
    representation, focal, partner, *_ = _transition()
    before = model.predict(state, representation, focal, partner)
    result = _call_update(model, state)
    after = model.predict(result.state, representation, focal, partner)

    chex.assert_trees_all_close(result.prediction, before, atol=0.0, rtol=0.0)
    chex.assert_trees_all_close(
        result.errors,
        before.raw_predictions - result.targets,
        atol=1e-7,
        rtol=1e-7,
    )
    assert int(result.state.update_count) == 1
    assert not bool(jnp.allclose(after.raw_predictions, before.raw_predictions))
    assert float(_call_update(model, result.state).loss) < float(result.loss)


@pytest.mark.parametrize("feature_path_mode", ["affine", "row_bias_only"])
def test_prediction_exposes_exact_preupdate_feature_and_row_bias_decomposition(
    feature_path_mode: str,
) -> None:
    model = _model(feature_path_mode=feature_path_mode)
    state = model.init(jax.random.key(18))
    representation, focal, partner, *_ = _transition()
    joint_index = int(model.joint_action_index(focal, partner))
    row_bias = jnp.asarray((0.125, -0.25, 0.5, -0.75, 1.0), dtype=jnp.float32)
    state = state.replace(bias=state.bias.at[joint_index].set(row_bias))

    prediction = model.predict(state, representation, focal, partner)
    result = _call_update(model, state)
    expected_feature = state.weights[joint_index] @ representation

    chex.assert_trees_all_equal(prediction.feature_contribution, expected_feature)
    chex.assert_trees_all_equal(prediction.row_bias, row_bias)
    chex.assert_trees_all_equal(
        prediction.raw_predictions,
        prediction.feature_contribution + prediction.row_bias,
    )
    chex.assert_trees_all_equal(result.prediction, prediction)
    if feature_path_mode == "row_bias_only":
        chex.assert_trees_all_equal(
            prediction.feature_contribution,
            jnp.zeros((model.target_dim,), dtype=jnp.float32),
        )
        chex.assert_trees_all_equal(prediction.raw_predictions, row_bias)


@pytest.mark.parametrize("feature_path_mode", ["affine", "row_bias_only"])
def test_update_exposes_exact_proposed_row_masks_and_executed_head_deltas(
    feature_path_mode: str,
) -> None:
    model = _model(feature_path_mode=feature_path_mode)
    state = model.init(jax.random.key(20))
    result = _call_update(model, state)
    joint_index = int(result.prediction.joint_action_index)

    weight_bits_before = np.asarray(state.weights).view(np.uint32)
    weight_bits_after = np.asarray(result.state.weights).view(np.uint32)
    bias_bits_before = np.asarray(state.bias).view(np.uint32)
    bias_bits_after = np.asarray(result.state.bias).view(np.uint32)
    expected_weight_mask = np.any(weight_bits_before != weight_bits_after, axis=(1, 2))
    expected_bias_mask = np.any(bias_bits_before != bias_bits_after, axis=1)
    expected_weight_delta_norm = jnp.linalg.norm(
        result.state.weights[joint_index] - state.weights[joint_index],
        axis=1,
    )
    expected_bias_delta = result.state.bias[joint_index] - state.bias[joint_index]

    assert bool(result.diagnostics.row_update_isolated)
    np.testing.assert_array_equal(
        np.asarray(result.proposed_weight_row_bit_change_mask),
        expected_weight_mask,
    )
    np.testing.assert_array_equal(
        np.asarray(result.proposed_bias_row_bit_change_mask),
        expected_bias_mask,
    )
    chex.assert_trees_all_equal(
        result.executed_weight_row_delta_norm_by_head,
        expected_weight_delta_norm,
    )
    chex.assert_trees_all_equal(
        result.executed_bias_row_delta_by_head,
        expected_bias_delta,
    )
    assert not bool(
        jnp.any(
            result.proposed_weight_row_bit_change_mask.at[joint_index].set(False)
        )
    )
    assert not bool(
        jnp.any(result.proposed_bias_row_bit_change_mask.at[joint_index].set(False))
    )
    assert bool(result.proposed_bias_row_bit_change_mask[joint_index])
    if feature_path_mode == "affine":
        assert bool(result.proposed_weight_row_bit_change_mask[joint_index])
        assert bool(jnp.all(result.executed_weight_row_delta_norm_by_head > 0.0))
    else:
        assert not bool(jnp.any(result.proposed_weight_row_bit_change_mask))
        chex.assert_trees_all_equal(
            result.executed_weight_row_delta_norm_by_head,
            jnp.zeros((model.target_dim,), dtype=jnp.float32),
        )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda values: values.__setitem__(0, values[0].at[0].set(jnp.nan)),
        lambda values: values.__setitem__(1, jnp.array(2, dtype=jnp.int32)),
        lambda values: values.__setitem__(2, jnp.array(-1, dtype=jnp.int32)),
        lambda values: values.__setitem__(3, values[3].at[1].set(jnp.inf)),
        lambda values: values.__setitem__(4, jnp.array(jnp.inf, dtype=jnp.float32)),
        lambda values: values.__setitem__(5, jnp.array(1.1, dtype=jnp.float32)),
    ],
)
def test_invalid_transition_is_fail_closed_atomic_noop(
    mutate: Callable[[list[jax.Array]], None],
) -> None:
    model = _model()
    state = model.init(jax.random.key(19))
    values = list(_transition())
    mutate(values)
    result = model.update(state, *values)

    chex.assert_trees_all_equal(result.state, state)
    assert not bool(result.diagnostics.applied)
    assert bool(result.diagnostics.rejected)
    assert not bool(result.gradient_valid)
    chex.assert_trees_all_equal(
        result.representation_gradient,
        jnp.zeros((model.config.representation_dim,), dtype=jnp.float32),
    )
    chex.assert_trees_all_equal(
        result.prediction.raw_predictions,
        result.prediction.feature_contribution + result.prediction.row_bias,
    )
    chex.assert_trees_all_equal(
        result.proposed_weight_row_bit_change_mask,
        jnp.zeros((model.joint_action_count,), dtype=jnp.bool_),
    )
    chex.assert_trees_all_equal(
        result.proposed_bias_row_bit_change_mask,
        jnp.zeros((model.joint_action_count,), dtype=jnp.bool_),
    )
    chex.assert_trees_all_equal(
        result.executed_weight_row_delta_norm_by_head,
        jnp.zeros((model.target_dim,), dtype=jnp.float32),
    )
    chex.assert_trees_all_equal(
        result.executed_bias_row_delta_by_head,
        jnp.zeros((model.target_dim,), dtype=jnp.float32),
    )
    assert not bool(result.diagnostics.row_update_isolated)
    assert float(result.loss) == 0.0


@pytest.mark.parametrize(
    ("feature_path_mode", "representation_loss_weights"),
    [
        ("affine", None),
        ("row_bias_only", None),
        ("affine", (0.0, 2.5, 0.0, 0.0, 2.5)),
        ("row_bias_only", (0.0, 2.5, 0.0, 0.0, 2.5)),
    ],
)
def test_predict_update_and_input_gradient_have_jit_parity(
    feature_path_mode: str,
    representation_loss_weights: tuple[float, ...] | None,
) -> None:
    model = _model(
        feature_path_mode=feature_path_mode,
        representation_loss_weights=representation_loss_weights,
    )
    state = model.init(jax.random.key(23))
    args = _transition()

    with jax.disable_jit():
        eager_prediction = model.predict(state, args[0], args[1], args[2])
        eager_gradient = model.input_loss_gradient(state, *args)
        eager_update = model.update(state, *args)

    compiled_prediction = jax.jit(model.predict)(state, args[0], args[1], args[2])
    compiled_gradient = jax.jit(model.input_loss_gradient)(state, *args)
    compiled_update = jax.jit(model.update)(state, *args)

    chex.assert_trees_all_close(eager_prediction, compiled_prediction, atol=1e-7, rtol=1e-7)
    chex.assert_trees_all_close(eager_gradient, compiled_gradient, atol=1e-7, rtol=1e-7)
    chex.assert_trees_all_close(eager_update, compiled_update, atol=1e-7, rtol=1e-7)


@pytest.mark.parametrize("feature_path_mode", ["affine", "row_bias_only"])
def test_scanned_updates_preserve_exact_executed_row_isolation(
    feature_path_mode: str,
) -> None:
    model = _model(feature_path_mode=feature_path_mode)
    initial_state = model.init(jax.random.key(24))
    representation, _, _, next_observation, reward, discount = _transition()
    focal_actions = jnp.asarray((0, 1, 0, 1), dtype=jnp.int32)
    partner_actions = jnp.asarray((0, 2, 1, 0), dtype=jnp.int32)

    def run_scan(state: Any) -> tuple[Any, Any]:
        def step(carry: Any, actions: tuple[jax.Array, jax.Array]) -> tuple[Any, Any]:
            focal, partner = actions
            result = model.update(
                carry,
                representation,
                focal,
                partner,
                next_observation,
                reward,
                discount,
            )
            trace = (
                result.prediction.joint_action_index,
                result.proposed_weight_row_bit_change_mask,
                result.proposed_bias_row_bit_change_mask,
                result.executed_weight_row_delta_norm_by_head,
                result.executed_bias_row_delta_by_head,
                result.diagnostics.row_update_isolated,
            )
            return result.state, trace

        return jax.lax.scan(step, state, (focal_actions, partner_actions))

    final_state, trace = jax.jit(run_scan)(initial_state)
    indices, weight_masks, bias_masks, weight_delta_norms, bias_deltas, isolated = trace

    np.testing.assert_array_equal(np.asarray(indices), np.asarray((0, 5, 1, 3)))
    assert bool(jnp.all(isolated))
    assert not bool(
        jnp.any(
            weight_masks
            & ~jax.nn.one_hot(indices, model.joint_action_count, dtype=jnp.bool_)
        )
    )
    assert not bool(
        jnp.any(
            bias_masks
            & ~jax.nn.one_hot(indices, model.joint_action_count, dtype=jnp.bool_)
        )
    )
    assert bool(jnp.all(bias_masks[jnp.arange(indices.shape[0]), indices]))
    assert bool(jnp.all(jnp.isfinite(weight_delta_norms)))
    assert bool(jnp.all(jnp.isfinite(bias_deltas)))
    if feature_path_mode == "row_bias_only":
        assert not bool(jnp.any(weight_masks))
        chex.assert_trees_all_equal(weight_delta_norms, jnp.zeros_like(weight_delta_norms))
    assert int(final_state.update_count) == focal_actions.shape[0]


def test_strict_config_and_json_checkpoint_roundtrip() -> None:
    model = _model()
    config_payload = model.config.to_config()
    assert GroundedJointWorldModelConfig.from_config(config_payload) == model.config
    assert GroundedJointWorldModel.from_config(model.to_config()).config == model.config

    state = _call_update(model, model.init(jax.random.key(29))).state
    serialized = json.loads(json.dumps(model.checkpoint_payload(state)))
    restored_model, restored_state = GroundedJointWorldModel.from_checkpoint_payload(serialized)
    assert restored_model.config == model.config
    chex.assert_trees_all_equal(restored_state, state)

    malformed = dict(config_payload)
    malformed["unexpected"] = 1
    with pytest.raises(ValueError, match="fields"):
        GroundedJointWorldModelConfig.from_config(malformed)

    malformed_checkpoint = dict(serialized)
    malformed_checkpoint["unexpected"] = 1
    with pytest.raises(ValueError, match="fields"):
        GroundedJointWorldModel.from_checkpoint_payload(malformed_checkpoint)


def test_default_affine_objective_and_update_are_literal_legacy_formulas() -> None:
    model = _model()
    state = model.init(jax.random.key(41))
    representation, focal, partner, next_observation, reward, discount = _transition()
    joint_index = int(model.joint_action_index(focal, partner))
    targets = jnp.concatenate((next_observation, reward[None], discount[None]))

    def legacy_objective(current_representation: jax.Array) -> jax.Array:
        raw = state.weights[joint_index] @ current_representation + state.bias[joint_index]
        errors = raw - targets
        return 0.5 * jnp.mean(jnp.square(errors))

    expected_loss, expected_gradient = jax.value_and_grad(legacy_objective)(representation)
    expected_raw = state.weights[joint_index] @ representation + state.bias[joint_index]
    expected_errors = expected_raw - targets
    expected_error_gradient = expected_errors / jnp.asarray(model.target_dim, dtype=jnp.float32)
    expected_weight_row = state.weights[joint_index] - model.config.step_size * (
        expected_error_gradient[:, None] * representation[None, :]
    )
    expected_bias_row = state.bias[joint_index] - model.config.step_size * expected_error_gradient

    score = model.input_loss_gradient(state, *_transition())
    result = model.update(state, *_transition())

    chex.assert_trees_all_close(score.loss, expected_loss, atol=2e-8, rtol=0.0)
    chex.assert_trees_all_equal(score.representation_loss, score.loss)
    chex.assert_trees_all_equal(score.fit_loss, score.loss)
    chex.assert_trees_all_close(
        jnp.sum(score.fit_loss_by_head), score.fit_loss, atol=2e-8, rtol=0.0
    )
    chex.assert_trees_all_equal(
        score.representation_loss_by_head,
        score.fit_loss_by_head,
    )
    chex.assert_trees_all_close(
        jnp.sum(score.representation_gradient_by_head, axis=0),
        score.gradient,
        atol=2e-8,
        rtol=0.0,
    )
    chex.assert_trees_all_close(
        score.representation_gradient_norm_by_head,
        jnp.linalg.norm(score.representation_gradient_by_head, axis=1),
        atol=0.0,
        rtol=0.0,
    )
    assert bool(score.feature_path_enabled)
    assert bool(score.representation_credit_enabled)
    chex.assert_trees_all_close(score.gradient, expected_gradient, atol=2e-8, rtol=0.0)
    chex.assert_trees_all_equal(result.loss, score.loss)
    chex.assert_trees_all_equal(result.representation_loss, score.loss)
    chex.assert_trees_all_equal(result.fit_loss, score.loss)
    chex.assert_trees_all_equal(result.fit_loss_by_head, score.fit_loss_by_head)
    chex.assert_trees_all_equal(
        result.representation_loss_by_head,
        score.representation_loss_by_head,
    )
    chex.assert_trees_all_equal(
        result.representation_gradient_by_head,
        score.representation_gradient_by_head,
    )
    chex.assert_trees_all_equal(
        result.parameter_gradient_norm,
        result.computed_parameter_gradient_norm,
    )
    chex.assert_trees_all_equal(
        result.applied_parameter_gradient_norm,
        result.computed_parameter_gradient_norm,
    )
    chex.assert_trees_all_equal(result.errors, expected_errors)
    chex.assert_trees_all_close(
        result.state.weights[joint_index], expected_weight_row, atol=4e-9, rtol=0.0
    )
    chex.assert_trees_all_close(
        result.state.bias[joint_index], expected_bias_row, atol=4e-9, rtol=0.0
    )
    # Frozen CPU bit patterns from the literal pre-extension implementation.
    assert np.asarray(score.loss).view(np.uint32) == np.uint32(1044608382)
    np.testing.assert_array_equal(
        np.asarray(score.gradient).view(np.uint32),
        np.asarray((1010104720, 3129107039, 3165565490, 1003124053), dtype=np.uint32),
    )
    np.testing.assert_array_equal(
        np.asarray(result.state.weights[joint_index]).view(np.uint32),
        np.asarray(
            (
                (3159092726, 1025120921, 1026481992, 3179308714),
                (1007495447, 1032158762, 3164152204, 1035053235),
                (1036623453, 1003115200, 1023737320, 3167287797),
                (1029586233, 3155027280, 1033741768, 1027882199),
                (3156329297, 1007488568, 1036057124, 3181143408),
            ),
            dtype=np.uint32,
        ),
    )
    np.testing.assert_array_equal(
        np.asarray(result.state.bias[joint_index]).view(np.uint32),
        np.asarray(
            (1013756736, 3177413743, 1024477712, 1028144218, 1033174172),
            dtype=np.uint32,
        ),
    )


def test_v6_weighted_representation_gradient_matches_closed_form_and_target_order() -> None:
    model = _model(
        target_observation_dim=8,
        representation_loss_weights=V6_REPRESENTATION_LOSS_WEIGHTS,
    )
    state = model.init(jax.random.key(43))
    representation = jnp.asarray((0.4, -0.2, 0.7, -0.5), dtype=jnp.float32)
    focal = jnp.asarray(1, dtype=jnp.int32)
    partner = jnp.asarray(2, dtype=jnp.int32)
    # Exact v6 order: [x, y, previous partner action, history, u, v, cue 1, cue 2].
    next_observation = jnp.asarray(
        (0.9, -1.0, 1.0, 1.0, -0.4, 0.3, 1.0, -1.0),
        dtype=jnp.float32,
    )
    reward = jnp.asarray(0.0, dtype=jnp.float32)
    discount = jnp.asarray(1.0, dtype=jnp.float32)
    score = model.input_loss_gradient(
        state,
        representation,
        focal,
        partner,
        next_observation,
        reward,
        discount,
    )
    weights = jnp.asarray(V6_REPRESENTATION_LOSS_WEIGHTS, dtype=jnp.float32)
    joint_index = int(score.prediction.joint_action_index)
    expected_gradient = state.weights[joint_index].T @ (
        weights * score.errors / jnp.asarray(model.target_dim, dtype=jnp.float32)
    )
    expected_gradient_by_head = (
        state.weights[joint_index]
        * (weights * score.errors / jnp.asarray(model.target_dim, dtype=jnp.float32))[:, None]
    )
    expected_loss = 0.5 * jnp.mean(weights * jnp.square(score.errors))
    expected_fit_loss = 0.5 * jnp.mean(jnp.square(score.errors))
    expected_fit_by_head = (
        0.5 * jnp.square(score.errors) / jnp.asarray(model.target_dim, dtype=jnp.float32)
    )
    expected_representation_loss_by_head = weights * expected_fit_by_head

    np.testing.assert_array_equal(
        np.asarray(score.targets),
        np.asarray(
            (*next_observation.tolist(), float(reward), float(discount)),
            dtype=np.float32,
        ),
    )
    chex.assert_trees_all_close(score.gradient, expected_gradient, atol=1e-7, rtol=1e-7)
    chex.assert_trees_all_close(score.loss, expected_loss, atol=1e-7, rtol=1e-7)
    chex.assert_trees_all_equal(score.representation_loss, score.loss)
    chex.assert_trees_all_close(score.fit_loss, expected_fit_loss, atol=1e-7, rtol=1e-7)
    chex.assert_trees_all_close(
        score.fit_loss_by_head,
        expected_fit_by_head,
        atol=1e-7,
        rtol=1e-7,
    )
    chex.assert_trees_all_close(
        score.representation_loss_by_head,
        expected_representation_loss_by_head,
        atol=1e-7,
        rtol=1e-7,
    )
    chex.assert_trees_all_close(
        score.representation_gradient_by_head,
        expected_gradient_by_head,
        atol=1e-7,
        rtol=1e-7,
    )
    chex.assert_trees_all_close(
        jnp.sum(score.representation_gradient_by_head, axis=0),
        score.gradient,
        atol=1e-7,
        rtol=1e-7,
    )
    chex.assert_trees_all_close(
        jnp.sum(score.fit_loss_by_head),
        score.fit_loss,
        atol=1e-7,
        rtol=1e-7,
    )
    chex.assert_trees_all_close(
        jnp.sum(score.representation_loss_by_head),
        score.representation_loss,
        atol=1e-7,
        rtol=1e-7,
    )


def test_v6_zero_credit_heads_do_not_steer_representation_but_all_heads_learn() -> None:
    model = _model(
        target_observation_dim=8,
        representation_loss_weights=V6_REPRESENTATION_LOSS_WEIGHTS,
    )
    state = model.init(jax.random.key(47))
    representation = jnp.asarray((0.6, -0.5, 0.25, 0.8), dtype=jnp.float32)
    focal = jnp.asarray(0, dtype=jnp.int32)
    partner = jnp.asarray(1, dtype=jnp.int32)
    first_next = jnp.asarray(
        (-0.9, 1.0, -1.0, 1.0, 0.75, -0.65, 1.0, -1.0),
        dtype=jnp.float32,
    )
    second_next = first_next.at[jnp.asarray((0, 2, 3, 4, 5))].set(
        jnp.asarray((0.3, 1.0, 0.0, -0.2, 0.4), dtype=jnp.float32)
    )
    first = model.input_loss_gradient(
        state,
        representation,
        focal,
        partner,
        first_next,
        jnp.asarray(1.0, dtype=jnp.float32),
        jnp.asarray(1.0, dtype=jnp.float32),
    )
    second = model.input_loss_gradient(
        state,
        representation,
        focal,
        partner,
        second_next,
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray(0.5, dtype=jnp.float32),
    )
    result = model.update(
        state,
        representation,
        focal,
        partner,
        first_next,
        jnp.asarray(1.0, dtype=jnp.float32),
        jnp.asarray(1.0, dtype=jnp.float32),
    )
    joint_index = int(result.prediction.joint_action_index)
    weight_delta = result.state.weights[joint_index] - state.weights[joint_index]
    bias_delta = result.state.bias[joint_index] - state.bias[joint_index]

    chex.assert_trees_all_equal(first.gradient, second.gradient)
    zero_credit_heads = jnp.asarray((0, 2, 3, 4, 5, 8, 9), dtype=jnp.int32)
    chex.assert_trees_all_equal(
        first.representation_gradient_by_head[zero_credit_heads],
        jnp.zeros((7, model.config.representation_dim), dtype=jnp.float32),
    )
    chex.assert_trees_all_equal(
        first.representation_loss_by_head[zero_credit_heads],
        jnp.zeros((7,), dtype=jnp.float32),
    )
    assert float(first.fit_loss) != float(second.fit_loss)
    assert bool(jnp.all(jnp.linalg.norm(weight_delta, axis=1) > 0.0))
    assert bool(jnp.all(jnp.abs(bias_delta) > 0.0))


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"representation_loss_weights": [1.0] * 5}, "tuple"),
        ({"representation_loss_weights": (1.0,) * 4}, "length"),
        ({"representation_loss_weights": (1.0, 1.0, 1.0, 1.0, -1.0)}, "nonnegative"),
        (
            {"representation_loss_weights": (1.0, 1.0, 1.0, 1.0, float("nan"))},
            "finite",
        ),
        ({"representation_loss_weights": (1.0, 1.0, 1.0, 1.0, True)}, "boolean"),
        (
            {"representation_loss_weights": (1.0, 1.0, 1.0, 1.0, np.bool_(True))},
            "boolean",
        ),
        ({"representation_loss_weights": (0.0,) * 5}, "positive"),
        ({"representation_loss_weights": (1.0,) * 4 + (0.5,)}, "sum"),
        ({"representation_loss_weights": (5.0000005, 0.0, 0.0, 0.0, 0.0)}, "sum"),
        (
            {
                "representation_loss_weights": (
                    0.1,
                    0.1,
                    0.1,
                    3.9,
                    0.7999999999999998,
                )
            },
            "deterministic float32",
        ),
        (
            {
                "representation_loss_weights": (
                    float(np.finfo(np.float32).tiny) / 2.0,
                    5.0,
                    0.0,
                    0.0,
                    0.0,
                )
            },
            "normal float32",
        ),
        ({"feature_path_mode": "bias"}, "feature_path_mode"),
    ],
)
def test_representation_loss_config_rejects_invalid_values(
    overrides: dict[str, Any],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        _model(**overrides)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("step_size", True),
        ("initialization_scale", np.bool_(True)),
        ("max_input_magnitude", float(np.finfo(np.float32).tiny) / 2.0),
        ("max_parameter_magnitude", float(np.finfo(np.float32).max) * 2.0),
    ],
)
def test_positive_scalar_controls_are_strict_normal_float32(
    name: str,
    value: Any,
) -> None:
    with pytest.raises(ValueError):
        _model(**{name: value})


def test_positive_scalar_controls_canonicalize_to_json_python_floats() -> None:
    model = _model(
        step_size=np.float32(0.4),
        initialization_scale=np.float64(0.08),
        max_input_magnitude=np.int64(100),
        max_parameter_magnitude=np.float32(100.0),
    )

    assert type(model.config.step_size) is float
    assert type(model.config.initialization_scale) is float
    assert type(model.config.max_input_magnitude) is float
    assert type(model.config.max_parameter_magnitude) is float
    assert json.loads(json.dumps(model.config.to_config())) == model.config.to_config()


def test_weighted_config_and_v3_checkpoint_are_strict_json_roundtrips() -> None:
    model = _model(
        target_observation_dim=8,
        representation_loss_weights=V6_REPRESENTATION_LOSS_WEIGHTS,
    )
    config_payload = model.config.to_config()
    assert isinstance(config_payload["representation_loss_weights"], list)
    restored_config = GroundedJointWorldModelConfig.from_config(
        json.loads(json.dumps(config_payload))
    )
    assert restored_config == model.config

    state = model.init(jax.random.key(49))
    checkpoint = json.loads(json.dumps(model.checkpoint_payload(state)))
    assert checkpoint["schema"] == "alberta.grounded_joint_world_model.v3"
    assert checkpoint["model"]["state_schema"] == GROUNDED_JOINT_WORLD_STATE_SCHEMA
    restored_model, restored_state = GroundedJointWorldModel.from_checkpoint_payload(checkpoint)
    assert restored_model.config == model.config
    chex.assert_trees_all_equal(restored_state, state)

    legacy_v2 = json.loads(json.dumps(checkpoint))
    legacy_v2["schema"] = "alberta.grounded_joint_world_model.v2"
    legacy_v2["model"].pop("state_schema")
    legacy_v2["state"].pop("update_words")
    migrated_model, migrated_state = GroundedJointWorldModel.from_checkpoint_payload(
        legacy_v2
    )
    assert migrated_model.config == model.config
    chex.assert_trees_all_equal(migrated_state, state)

    ambiguous_v2 = json.loads(json.dumps(legacy_v2))
    ambiguous_v2["state"]["update_count"] = 2**31 - 1
    with pytest.raises(ValueError, match="ambiguous"):
        GroundedJointWorldModel.from_checkpoint_payload(ambiguous_v2)

    v1_checkpoint = json.loads(json.dumps(checkpoint))
    v1_checkpoint["schema"] = "alberta.grounded_joint_world_model.v1"
    with pytest.raises(ValueError, match="checkpoint schema"):
        GroundedJointWorldModel.from_checkpoint_payload(v1_checkpoint)

    boolean_checkpoint = json.loads(json.dumps(checkpoint))
    boolean_checkpoint["state"]["weights"][0][0][0] = False
    with pytest.raises(ValueError, match="JSON number"):
        GroundedJointWorldModel.from_checkpoint_payload(boolean_checkpoint)

    string_checkpoint = json.loads(json.dumps(checkpoint))
    string_checkpoint["state"]["bias"][0][0] = "0"
    with pytest.raises(ValueError, match="JSON number"):
        GroundedJointWorldModel.from_checkpoint_payload(string_checkpoint)

    tuple_checkpoint = json.loads(json.dumps(checkpoint))
    tuple_checkpoint["state"]["weights"] = tuple(tuple_checkpoint["state"]["weights"])
    with pytest.raises(ValueError, match="JSON list"):
        GroundedJointWorldModel.from_checkpoint_payload(tuple_checkpoint)

    tuple_wire_payload = dict(config_payload)
    tuple_wire_payload["representation_loss_weights"] = V6_REPRESENTATION_LOSS_WEIGHTS
    with pytest.raises(ValueError, match="JSON list"):
        GroundedJointWorldModelConfig.from_config(tuple_wire_payload)


def test_row_bias_only_is_zero_weight_fixed_resource_all_head_control() -> None:
    affine = _model(feature_path_mode="affine")
    model = _model(feature_path_mode="row_bias_only")
    state = model.init(jax.random.key(53))
    representation, focal, partner, next_observation, reward, discount = _transition()
    opposite_representation = -representation

    chex.assert_trees_all_equal(state.weights, jnp.zeros_like(state.weights))
    row_budget = model.resource_budget
    affine_budget = affine.resource_budget
    assert row_budget.allocated_float32_scalars == affine_budget.allocated_float32_scalars
    assert row_budget.state_nbytes == affine_budget.state_nbytes
    assert (
        row_budget.computed_parameter_gradient_float32_scalars_per_update
        == affine_budget.computed_parameter_gradient_float32_scalars_per_update
    )
    assert row_budget.trainable_float32_scalars == model.joint_action_count * model.target_dim
    assert row_budget.applied_trainable_float32_scalars_per_update == model.target_dim
    assert row_budget.learned_float32_scalars_touched_per_update == model.target_dim
    assert row_budget.trainable_float32_scalars < affine_budget.trainable_float32_scalars
    assert (
        row_budget.applied_trainable_float32_scalars_per_update
        < affine_budget.applied_trainable_float32_scalars_per_update
    )
    first_prediction = model.predict(state, representation, focal, partner)
    second_prediction = model.predict(state, opposite_representation, focal, partner)
    chex.assert_trees_all_equal(first_prediction.raw_predictions, second_prediction.raw_predictions)

    result = model.update(
        state,
        representation,
        focal,
        partner,
        next_observation,
        reward,
        discount,
    )
    joint_index = int(result.prediction.joint_action_index)
    chex.assert_trees_all_equal(result.state.weights, state.weights)
    chex.assert_trees_all_equal(
        result.representation_gradient,
        jnp.zeros((model.config.representation_dim,), dtype=jnp.float32),
    )
    chex.assert_trees_all_equal(
        result.representation_gradient_by_head,
        jnp.zeros(
            (model.target_dim, model.config.representation_dim),
            dtype=jnp.float32,
        ),
    )
    assert not bool(result.feature_path_enabled)
    assert not bool(result.representation_credit_enabled)
    assert not bool(result.diagnostics.feature_path_enabled)
    assert not bool(result.diagnostics.representation_credit_enabled)
    assert bool(result.gradient_valid)
    chex.assert_trees_all_equal(
        result.parameter_gradient_norm,
        result.computed_parameter_gradient_norm,
    )
    assert float(result.computed_parameter_gradient_norm) > float(
        result.applied_parameter_gradient_norm
    )
    assert bool(jnp.all(jnp.abs(result.state.bias[joint_index]) > 0.0))

    routed = result.state.replace(weights=result.state.weights[..., ::-1])
    chex.assert_trees_all_equal(routed.weights, jnp.zeros_like(routed.weights))
    routed_prediction = model.predict(routed, opposite_representation, focal, partner)
    chex.assert_trees_all_equal(
        routed_prediction.raw_predictions,
        result.state.bias[joint_index],
    )


def test_row_bias_only_rejects_nonzero_weight_state_and_checkpoint() -> None:
    model = _model(feature_path_mode="row_bias_only")
    state = model.init(jax.random.key(59))
    corrupt = state.replace(weights=state.weights.at[0, 0, 0].set(0.25))
    representation, focal, partner, *_ = _transition()
    prediction = model.predict(corrupt, representation, focal, partner)
    score = model.input_loss_gradient(corrupt, *_transition())
    result = _call_update(model, corrupt)

    assert not bool(prediction.valid)
    assert not bool(score.valid)
    chex.assert_trees_all_equal(
        score.gradient,
        jnp.zeros((model.config.representation_dim,), dtype=jnp.float32),
    )
    assert not bool(score.feature_path_enabled)
    assert not bool(score.representation_credit_enabled)
    chex.assert_trees_all_equal(result.state, corrupt)
    assert not bool(result.diagnostics.state_valid)
    assert not bool(result.diagnostics.applied)
    with pytest.raises(ValueError, match="invalid model state"):
        model.checkpoint_payload(corrupt)

    checkpoint = model.checkpoint_payload(state)
    checkpoint["state"]["weights"][0][0][0] = 0.25
    with pytest.raises(ValueError, match="zero weights"):
        GroundedJointWorldModel.from_checkpoint_payload(checkpoint)


def test_resource_budget_matches_state_bytes_and_exact_update_surface() -> None:
    model = _model()
    state = model.init(jax.random.key(31))
    budget = model.resource_budget
    actual_nbytes = sum(int(leaf.nbytes) for leaf in jax.tree_util.tree_leaves(state))
    expected_parameters = (
        model.joint_action_count * model.target_dim * (model.config.representation_dim + 1)
    )
    expected_touched = model.target_dim * (model.config.representation_dim + 1)

    assert state.weights.shape == (
        model.joint_action_count,
        model.target_dim,
        model.config.representation_dim,
    )
    assert state.bias.shape == (model.joint_action_count, model.target_dim)
    assert budget.allocated_float32_scalars == expected_parameters
    assert budget.trainable_float32_scalars == expected_parameters
    assert budget.administrative_int32_scalars == 1
    assert budget.administrative_uint32_scalars == 2
    assert budget.state_nbytes == actual_nbytes == 4 * (expected_parameters + 3)
    assert budget.state_nbytes == measure_grounded_joint_world_state_nbytes(state)
    assert grounded_joint_world_lifetime_counter_nbytes() == 12
    assert budget.computed_parameter_gradient_float32_scalars_per_update == expected_touched
    assert budget.learned_float32_scalars_touched_per_update == expected_touched
    assert budget.applied_trainable_float32_scalars_per_update == expected_touched
    assert budget.administrative_int32_scalars_touched_per_update == 1
    assert budget.administrative_uint32_scalars_touched_per_update == 2
    assert budget.replay_capacity == 0
    assert budget.to_dict()["state_nbytes"] == actual_nbytes
    assert model.representation_indexed_state_leaves == (("weights", -1),)


def test_representation_and_grounded_observation_widths_are_independent() -> None:
    model = _model(representation_dim=24, target_observation_dim=8)
    state = model.init(jax.random.key(33))
    representation = jnp.linspace(-0.5, 0.5, 24, dtype=jnp.float32)
    next_observation = jnp.linspace(0.4, -0.4, 8, dtype=jnp.float32)
    score = model.input_loss_gradient(
        state,
        representation,
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(1, dtype=jnp.int32),
        next_observation,
        jnp.asarray(0.25, dtype=jnp.float32),
        jnp.asarray(0.95, dtype=jnp.float32),
    )

    assert model.target_dim == 10
    assert state.weights.shape == (6, 10, 24)
    assert state.bias.shape == (6, 10)
    assert score.prediction.next_observation.shape == (8,)
    assert score.targets.shape == (10,)
    assert score.gradient.shape == (24,)
    assert bool(score.valid)
    assert 0.0 <= float(score.prediction.discount) <= 1.0


def test_corrupted_state_and_exhausted_counter_are_atomic_noops() -> None:
    model = _model()
    state = model.init(jax.random.key(35))
    corrupt = state.replace(weights=state.weights.at[0, 0, 0].set(jnp.nan))
    exhausted = state.replace(
        update_count=jnp.asarray(2**31 - 1, dtype=jnp.int32),
        update_words=jnp.full((2,), 2**32 - 1, dtype=jnp.uint32),
    )

    corrupt_result = _call_update(model, corrupt)
    exhausted_result = _call_update(model, exhausted)

    for before, after in zip(
        jax.tree_util.tree_leaves(corrupt),
        jax.tree_util.tree_leaves(corrupt_result.state),
        strict=True,
    ):
        assert np.asarray(before).tobytes() == np.asarray(after).tobytes()
    chex.assert_trees_all_equal(exhausted_result.state, exhausted)
    assert not bool(corrupt_result.diagnostics.state_valid)
    assert not bool(corrupt_result.diagnostics.applied)
    assert bool(corrupt_result.diagnostics.rejected)
    assert bool(exhausted_result.diagnostics.state_valid)
    assert bool(exhausted_result.diagnostics.lifetime_counter_valid)
    assert not bool(exhausted_result.diagnostics.capacity_available)
    assert not bool(exhausted_result.diagnostics.applied)
    assert not bool(exhausted_result.update_applied)
    assert bool(exhausted_result.diagnostics.rejected)


def test_exact_lifetime_clock_carries_scans_and_rejects_misalignment() -> None:
    model = _model()
    initial = model.init(jax.random.key(3_501))
    near_carry = initial.replace(
        update_count=jnp.asarray(2**31 - 1, dtype=jnp.int32),
        update_words=jnp.asarray((0, 2**32 - 1), dtype=jnp.uint32),
    )

    carried = _call_update(model, near_carry)
    assert bool(carried.diagnostics.lifetime_counter_valid)
    assert bool(carried.diagnostics.capacity_available)
    assert bool(carried.update_applied)
    chex.assert_trees_all_equal(
        carried.pre_update_words,
        jnp.asarray((0, 2**32 - 1), dtype=jnp.uint32),
    )
    chex.assert_trees_all_equal(
        carried.post_update_words,
        jnp.asarray((1, 0), dtype=jnp.uint32),
    )
    assert int(carried.state.update_count) == 2**31 - 1

    def step(state, _):
        result = _call_update(model, state)
        return result.state, (result.update_applied, result.post_update_words)

    scanned, (applied, words) = jax.lax.scan(
        step,
        near_carry,
        jnp.arange(2, dtype=jnp.int32),
    )
    chex.assert_trees_all_equal(applied, jnp.asarray((True, True), dtype=jnp.bool_))
    chex.assert_trees_all_equal(
        words,
        jnp.asarray(((1, 0), (1, 1)), dtype=jnp.uint32),
    )
    chex.assert_trees_all_equal(
        scanned.update_words,
        jnp.asarray((1, 1), dtype=jnp.uint32),
    )

    misaligned = initial.replace(update_count=jnp.asarray(1, dtype=jnp.int32))
    rejected = _call_update(model, misaligned)
    assert not bool(rejected.diagnostics.lifetime_counter_valid)
    assert not bool(rejected.diagnostics.state_valid)
    assert not bool(rejected.update_applied)
    chex.assert_trees_all_equal(rejected.state, misaligned)


def test_legacy_grounded_state_migration_is_fail_closed() -> None:
    model = _model()
    state = model.init(jax.random.key(3_502))
    legacy = {
        "weights": state.weights,
        "bias": state.bias,
        "update_count": jnp.asarray(19, dtype=jnp.int32),
    }
    migrated = migrate_legacy_grounded_joint_world_state(legacy)
    chex.assert_trees_all_equal(
        migrated.update_words,
        jnp.asarray((0, 19), dtype=jnp.uint32),
    )

    saturated = dict(legacy)
    saturated["update_count"] = jnp.asarray(2**31 - 1, dtype=jnp.int32)
    with pytest.raises(ValueError, match="ambiguous"):
        migrate_legacy_grounded_joint_world_state(saturated)

    negative = dict(legacy)
    negative["update_count"] = jnp.asarray(-1, dtype=jnp.int32)
    with pytest.raises(ValueError, match="wrap"):
        migrate_legacy_grounded_joint_world_state(negative)


def test_out_of_bound_candidate_parameter_update_is_atomic_noop() -> None:
    model = _model(
        step_size=100.0,
        initialization_scale=0.01,
        max_parameter_magnitude=0.1,
    )
    state = model.init(jax.random.key(36))
    result = _call_update(model, state)

    chex.assert_trees_all_equal(result.state, state)
    assert bool(result.diagnostics.state_valid)
    assert bool(result.diagnostics.input_valid)
    assert bool(result.diagnostics.parameter_update_valid)
    assert not bool(result.diagnostics.candidate_state_valid)
    assert not bool(result.diagnostics.applied)
    assert bool(result.diagnostics.rejected)
    assert not bool(result.gradient_valid)
    assert not bool(result.diagnostics.row_update_isolated)
    chex.assert_trees_all_equal(
        result.proposed_weight_row_bit_change_mask,
        jnp.zeros((model.joint_action_count,), dtype=jnp.bool_),
    )
    chex.assert_trees_all_equal(
        result.proposed_bias_row_bit_change_mask,
        jnp.zeros((model.joint_action_count,), dtype=jnp.bool_),
    )
    chex.assert_trees_all_equal(
        result.executed_weight_row_delta_norm_by_head,
        jnp.zeros((model.target_dim,), dtype=jnp.float32),
    )
    chex.assert_trees_all_equal(
        result.executed_bias_row_delta_by_head,
        jnp.zeros((model.target_dim,), dtype=jnp.float32),
    )


def test_initialization_is_deterministic_and_config_rejects_invalid_values() -> None:
    model = _model()
    chex.assert_trees_all_equal(model.init(jax.random.key(37)), model.init(jax.random.key(37)))
    assert not bool(
        jnp.array_equal(
            model.init(jax.random.key(37)).weights,
            model.init(jax.random.key(38)).weights,
        )
    )

    for kwargs in (
        {"representation_dim": True},
        {"representation_dim": np.int64(4)},
        {"target_observation_dim": 0},
        {"n_focal_actions": -1},
        {"n_partner_actions": 0},
        {"step_size": float("nan")},
        {"initialization_scale": float("inf")},
        {"max_input_magnitude": 0.0},
        {"max_parameter_magnitude": float("inf")},
    ):
        with pytest.raises(ValueError):
            _model(**kwargs)
