"""Tests for the online behavior/action prediction model."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import chex
import jax
import jax.numpy as jnp
import pytest

try:
    from alberta_framework.core.behavior_model import (
        BehaviorModel,
        BehaviorModelConfig,
        action_log_likelihoods,
        clipped_importance_ratios,
        epsilon_greedy_probabilities,
        floor_and_renormalize_probabilities,
        run_behavior_model_from_arrays,
        selected_action_probabilities,
    )
except ImportError:
    # Other in-flight Step 8/world-model lanes can temporarily break package
    # imports. Keep this focused behavior-model test runnable without touching
    # those files.
    module_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "alberta_framework"
        / "core"
        / "behavior_model.py"
    )
    spec = importlib.util.spec_from_file_location(
        "alberta_framework_behavior_model_under_test",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise
    behavior_model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(behavior_model_module)
    BehaviorModel = behavior_model_module.BehaviorModel
    BehaviorModelConfig = behavior_model_module.BehaviorModelConfig
    action_log_likelihoods = behavior_model_module.action_log_likelihoods
    clipped_importance_ratios = behavior_model_module.clipped_importance_ratios
    epsilon_greedy_probabilities = behavior_model_module.epsilon_greedy_probabilities
    floor_and_renormalize_probabilities = behavior_model_module.floor_and_renormalize_probabilities
    run_behavior_model_from_arrays = behavior_model_module.run_behavior_model_from_arrays
    selected_action_probabilities = behavior_model_module.selected_action_probabilities


def _assert_behavior_update_finite(result: Any) -> None:
    """Check numeric leaves while handling JAX typed PRNG keys explicitly."""
    chex.assert_tree_all_finite(
        (
            result.state.weights,
            result.state.bias,
            jax.random.key_data(result.state.rng_key),
            result.state.step_count,
            result.state.nll_ema,
            result.state.accuracy_ema,
            result.state.confidence_ema,
            result.logits,
            result.probabilities,
            result.action_probability,
            result.log_likelihood,
            result.loss,
            result.entropy,
            result.confidence,
            result.predicted_action,
            result.correct,
        )
    )


def test_init_predict_update_finite_and_shapes() -> None:
    model = BehaviorModel(BehaviorModelConfig(n_actions=3, step_size=0.1))
    state = model.init(feature_dim=4, key=jax.random.key(0))
    obs = jnp.array([1.0, -1.0, 0.5, 2.0], dtype=jnp.float32)

    logits = model.predict_logits(state, obs)
    probs = model.predict_probabilities(state, obs)
    result = model.update(state, obs, jnp.array(2, dtype=jnp.int32))

    chex.assert_shape(logits, (3,))
    chex.assert_shape(probs, (3,))
    chex.assert_shape(result.probabilities, (3,))
    chex.assert_shape(result.action_probability, ())
    _assert_behavior_update_finite(result)
    assert int(result.state.step_count) == 1
    assert float(result.loss) > 0.0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("step_size", float("nan")),
        ("step_size", float("inf")),
        ("temperature", float("nan")),
        ("temperature", float("inf")),
        ("l2_penalty", float("nan")),
        ("max_gradient_norm", float("inf")),
        ("min_probability", float("nan")),
        ("min_probability", 1.0),
        ("ratio_clip", float("inf")),
        ("diagnostic_decay", float("nan")),
    ],
)
def test_config_rejects_nonfinite_or_invalid_numeric_values(
    field: str,
    value: float,
) -> None:
    kwargs: dict[str, Any] = {"n_actions": 2, field: value}
    with pytest.raises(ValueError):
        BehaviorModelConfig(**kwargs)


def test_config_and_init_reject_boolean_or_nonpositive_dimensions() -> None:
    with pytest.raises(ValueError, match="n_actions"):
        BehaviorModelConfig(n_actions=True)
    model = BehaviorModel(BehaviorModelConfig(n_actions=2))
    for feature_dim in (True, 0, -1):
        with pytest.raises(ValueError, match="feature_dim"):
            model.init(feature_dim=feature_dim, key=jax.random.key(0))
        with pytest.raises(ValueError, match="feature_dim"):
            model.resource_budget(feature_dim)


def test_resource_budget_matches_initialized_state_arrays_exactly() -> None:
    model = BehaviorModel(BehaviorModelConfig(n_actions=3))
    feature_dim = 5
    state = model.init(feature_dim=feature_dim, key=jax.random.key(0))
    budget = model.resource_budget(feature_dim)
    actual_nbytes = sum(int(leaf.nbytes) for leaf in jax.tree_util.tree_leaves(state))

    assert budget.feature_dim == feature_dim
    assert budget.n_actions == 3
    assert budget.trainable_float32_scalars == 3 * 5 + 3
    assert budget.diagnostic_float32_scalars == 3
    assert budget.administrative_int32_scalars == 1
    assert budget.rng_uint32_scalars == 2
    assert budget.state_nbytes == actual_nbytes
    assert budget.learned_float32_scalars_touched_per_update == 3 * 5 + 3 + 3
    assert budget.replay_capacity == 0
    assert budget.to_dict()["state_nbytes"] == actual_nbytes


def test_preupdate_input_gradient_matches_autodiff_and_does_not_advance_state() -> None:
    model = BehaviorModel(
        BehaviorModelConfig(
            n_actions=3,
            step_size=0.1,
            temperature=0.7,
        )
    )
    state = model.init(feature_dim=4, key=jax.random.key(7)).replace(
        weights=jnp.array(
            [
                [0.2, -0.3, 0.5, 0.1],
                [-0.4, 0.6, 0.2, -0.5],
                [0.1, 0.4, -0.2, 0.3],
            ],
            dtype=jnp.float32,
        ),
        bias=jnp.array([0.1, -0.2, 0.05], dtype=jnp.float32),
    )
    observation = jnp.array([0.5, -1.0, 0.25, 2.0], dtype=jnp.float32)
    action = jnp.array(1, dtype=jnp.int32)

    def loss_fn(features):
        logits = (state.weights @ features + state.bias) / 0.7
        return -jax.nn.log_softmax(logits)[action]

    expected_loss, expected_gradient = jax.value_and_grad(loss_fn)(observation)
    before = jax.tree_util.tree_map(lambda value: value.copy(), state)
    result = jax.jit(model.input_loss_gradient)(state, observation, action)

    chex.assert_trees_all_close(result.loss, expected_loss, atol=1e-7, rtol=1e-6)
    chex.assert_trees_all_close(
        result.gradient,
        expected_gradient,
        atol=1e-7,
        rtol=1e-6,
    )
    chex.assert_trees_all_close(
        result.gradient_norm,
        jnp.linalg.norm(expected_gradient),
        atol=1e-7,
        rtol=1e-6,
    )
    chex.assert_trees_all_equal(state, before)
    assert int(state.step_count) == 0


def test_probability_simplex_and_helper_invariants() -> None:
    model = BehaviorModel(BehaviorModelConfig(n_actions=4))
    state = model.init(feature_dim=2, key=jax.random.key(1))
    probs = model.predict_probabilities(
        state,
        jnp.array([10.0, -3.0], dtype=jnp.float32),
    )
    floored = floor_and_renormalize_probabilities(
        jnp.array([0.0, 0.2, 0.3, 0.5], dtype=jnp.float32),
        min_probability=0.01,
    )

    chex.assert_trees_all_close(jnp.sum(probs), 1.0, atol=1e-6)
    chex.assert_trees_all_close(jnp.sum(floored), 1.0, atol=1e-6)
    assert float(jnp.min(floored)) >= 0.01 - 1e-7

    selected = selected_action_probabilities(
        jnp.array([[0.2, 0.8], [0.9, 0.1]], dtype=jnp.float32),
        jnp.array([1, 0], dtype=jnp.int32),
    )
    logs = action_log_likelihoods(
        jnp.array([[0.2, 0.8], [0.9, 0.1]], dtype=jnp.float32),
        jnp.array([1, 0], dtype=jnp.int32),
    )
    chex.assert_trees_all_close(selected, jnp.array([0.8, 0.9], dtype=jnp.float32))
    chex.assert_trees_all_close(logs, jnp.log(selected))


def test_likelihood_improves_on_deterministic_policy_stream() -> None:
    model = BehaviorModel(BehaviorModelConfig(n_actions=2, step_size=0.2, diagnostic_decay=0.9))
    state = model.init(feature_dim=2, key=jax.random.key(2))
    obs0 = jnp.array([1.0, 0.0], dtype=jnp.float32)
    obs1 = jnp.array([0.0, 1.0], dtype=jnp.float32)

    start_p0 = model.action_probability(state, obs0, jnp.array(0, dtype=jnp.int32))
    start_p1 = model.action_probability(state, obs1, jnp.array(1, dtype=jnp.int32))
    for _ in range(160):
        state = model.update(state, obs0, jnp.array(0, dtype=jnp.int32)).state
        state = model.update(state, obs1, jnp.array(1, dtype=jnp.int32)).state

    end_p0 = model.action_probability(state, obs0, jnp.array(0, dtype=jnp.int32))
    end_p1 = model.action_probability(state, obs1, jnp.array(1, dtype=jnp.int32))

    assert float(end_p0) > float(start_p0) + 0.35
    assert float(end_p1) > float(start_p1) + 0.35
    assert float(end_p0) > 0.85
    assert float(end_p1) > 0.85
    assert float(state.accuracy_ema) > 0.85


def test_scan_loop_and_jit_compatibility() -> None:
    model = BehaviorModel(BehaviorModelConfig(n_actions=3, step_size=0.05))
    state = model.init(feature_dim=3, key=jax.random.key(3))
    observations = jnp.eye(3, dtype=jnp.float32).repeat(4, axis=0)
    actions = jnp.array([0, 1, 2] * 4, dtype=jnp.int32)

    jitted_update = jax.jit(model.update)
    update_result = jitted_update(state, observations[0], actions[0])
    result = run_behavior_model_from_arrays(
        model,
        state,
        observations,
        actions,
    )

    _assert_behavior_update_finite(update_result)
    chex.assert_shape(result.probabilities, (12, 3))
    chex.assert_shape(result.action_probabilities, (12,))
    chex.assert_shape(result.log_likelihoods, (12,))
    chex.assert_shape(result.correct, (12,))
    assert int(result.state.step_count) == 12


def test_config_roundtrip_and_sampling() -> None:
    model = BehaviorModel(
        BehaviorModelConfig(
            n_actions=3,
            step_size=0.03,
            temperature=0.8,
            l2_penalty=0.01,
            max_gradient_norm=1.5,
            min_probability=1e-5,
            ratio_clip=3.0,
            diagnostic_decay=0.8,
        )
    )
    restored = BehaviorModel.from_config(model.to_config())
    assert restored.to_config() == model.to_config()

    state = restored.init(feature_dim=2, key=jax.random.key(4))
    sample = restored.sample_action(state, jnp.ones(2, dtype=jnp.float32))
    chex.assert_shape(sample.probabilities, (3,))
    chex.assert_trees_all_close(jnp.sum(sample.probabilities), 1.0, atol=1e-6)
    assert 0 <= int(sample.action) < 3


def test_importance_ratio_and_epsilon_greedy_helpers() -> None:
    target = jnp.array([[0.8, 0.2], [0.1, 0.9]], dtype=jnp.float32)
    behavior = jnp.array([[0.4, 0.6], [0.5, 0.5]], dtype=jnp.float32)
    actions = jnp.array([0, 1], dtype=jnp.int32)

    ratios = clipped_importance_ratios(
        target,
        behavior,
        actions,
        clip=1.5,
    )
    chex.assert_trees_all_close(ratios, jnp.array([1.5, 1.5], dtype=jnp.float32))

    q_values = jnp.array([1.0, 3.0, 3.0, 0.0], dtype=jnp.float32)
    probs = epsilon_greedy_probabilities(q_values, jnp.array(0.2, dtype=jnp.float32))
    expected = jnp.array([0.05, 0.45, 0.45, 0.05], dtype=jnp.float32)
    chex.assert_trees_all_close(probs, expected, atol=1e-6)

    model = BehaviorModel(BehaviorModelConfig(n_actions=2, ratio_clip=1.25))
    state = model.init(feature_dim=2, key=jax.random.key(5))
    ratio = model.importance_ratio(
        state,
        jnp.ones(2, dtype=jnp.float32),
        jnp.array(1, dtype=jnp.int32),
        jnp.array([0.1, 0.9], dtype=jnp.float32),
    )
    assert float(ratio) == 1.25
