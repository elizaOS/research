"""Tests for the online behavior/action prediction model."""

from __future__ import annotations

import dataclasses
import importlib.util
from pathlib import Path
from typing import Any

import chex
import jax
import jax.numpy as jnp
import pytest

try:
    from alberta_framework.core.behavior_model import (
        BEHAVIOR_MODEL_STATE_SCHEMA,
        BehaviorModel,
        BehaviorModelConfig,
        action_log_likelihoods,
        behavior_model_lifetime_counter_nbytes,
        clipped_importance_ratios,
        epsilon_greedy_probabilities,
        floor_and_renormalize_probabilities,
        measure_behavior_model_state_nbytes,
        migrate_legacy_behavior_model_state,
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
    BEHAVIOR_MODEL_STATE_SCHEMA = behavior_model_module.BEHAVIOR_MODEL_STATE_SCHEMA
    action_log_likelihoods = behavior_model_module.action_log_likelihoods
    behavior_model_lifetime_counter_nbytes = (
        behavior_model_module.behavior_model_lifetime_counter_nbytes
    )
    clipped_importance_ratios = behavior_model_module.clipped_importance_ratios
    epsilon_greedy_probabilities = behavior_model_module.epsilon_greedy_probabilities
    floor_and_renormalize_probabilities = behavior_model_module.floor_and_renormalize_probabilities
    measure_behavior_model_state_nbytes = behavior_model_module.measure_behavior_model_state_nbytes
    migrate_legacy_behavior_model_state = (
        behavior_model_module.migrate_legacy_behavior_model_state
    )
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
            result.state.step_words,
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
            result.pre_step_words,
            result.post_step_words,
            result.lifetime_counter_valid,
            result.lifetime_capacity_available,
            result.update_applied,
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
    chex.assert_trees_all_equal(result.state.step_words, jnp.asarray((0, 1), dtype=jnp.uint32))
    assert bool(result.update_applied)
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
    assert budget.lifetime_counter_uint32_scalars == 2
    assert budget.rng_uint32_scalars == 2
    assert budget.state_nbytes == actual_nbytes
    assert budget.state_nbytes == measure_behavior_model_state_nbytes(state)
    assert behavior_model_lifetime_counter_nbytes() == 12
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
    assert restored.to_config()["state_schema"] == BEHAVIOR_MODEL_STATE_SCHEMA

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


def test_exact_lifetime_clock_carries_and_refuses_invalid_or_exhausted_state() -> None:
    model = BehaviorModel(BehaviorModelConfig(n_actions=2, step_size=0.1))
    initial = model.init(feature_dim=2, key=jax.random.key(8))
    observation = jnp.asarray((0.5, -0.25), dtype=jnp.float32)
    action = jnp.asarray(1, dtype=jnp.int32)
    near_carry = initial.replace(
        step_count=jnp.asarray(2**31 - 1, dtype=jnp.int32),
        step_words=jnp.asarray((0, 2**32 - 1), dtype=jnp.uint32),
    )

    carried = jax.jit(model.update)(near_carry, observation, action)
    assert bool(carried.lifetime_counter_valid)
    assert bool(carried.lifetime_capacity_available)
    assert bool(carried.update_applied)
    chex.assert_trees_all_equal(
        carried.pre_step_words,
        jnp.asarray((0, 2**32 - 1), dtype=jnp.uint32),
    )
    chex.assert_trees_all_equal(
        carried.post_step_words,
        jnp.asarray((1, 0), dtype=jnp.uint32),
    )
    assert int(carried.state.step_count) == 2**31 - 1

    def scan_step(state, _):
        result = model.update(state, observation, action)
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
    stopped = model.update(exhausted, observation, action)
    assert bool(stopped.lifetime_counter_valid)
    assert not bool(stopped.lifetime_capacity_available)
    assert not bool(stopped.update_applied)
    chex.assert_trees_all_equal(stopped.state, exhausted)
    chex.assert_trees_all_equal(stopped.pre_step_words, stopped.post_step_words)

    misaligned = initial.replace(step_count=jnp.asarray(1, dtype=jnp.int32))
    rejected = model.update(misaligned, observation, action)
    assert not bool(rejected.lifetime_counter_valid)
    assert not bool(rejected.update_applied)
    chex.assert_trees_all_equal(rejected.state, misaligned)


def test_legacy_behavior_clock_migration_is_exact_and_fail_closed() -> None:
    model = BehaviorModel(BehaviorModelConfig(n_actions=2))
    state = model.init(feature_dim=3, key=jax.random.key(9))
    legacy = {
        field.name: getattr(state, field.name)
        for field in dataclasses.fields(type(state))
        if field.name != "step_words"
    }
    legacy["step_count"] = jnp.asarray(17, dtype=jnp.int32)
    migrated = migrate_legacy_behavior_model_state(legacy)
    chex.assert_trees_all_equal(
        migrated.step_words,
        jnp.asarray((0, 17), dtype=jnp.uint32),
    )

    saturated = dict(legacy)
    saturated["step_count"] = jnp.asarray(2**31 - 1, dtype=jnp.int32)
    with pytest.raises(ValueError, match="ambiguous"):
        migrate_legacy_behavior_model_state(saturated)

    negative = dict(legacy)
    negative["step_count"] = jnp.asarray(-1, dtype=jnp.int32)
    with pytest.raises(ValueError, match="wrap"):
        migrate_legacy_behavior_model_state(negative)

    incomplete = dict(legacy)
    incomplete.pop("confidence_ema")
    with pytest.raises(ValueError, match="manifest"):
        migrate_legacy_behavior_model_state(incomplete)
