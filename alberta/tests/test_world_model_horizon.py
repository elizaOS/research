"""Focused exact-clock contracts for direct MultiHead world-model wrappers."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from alberta_framework.core.normalizers import EMANormalizer, WelfordNormalizer
from alberta_framework.core.optimizers import LMS
from alberta_framework.core.world_model import (
    ACTION_CONDITIONED_WORLD_MODEL_STATE_SCHEMA,
    ONE_STEP_WORLD_MODEL_STATE_SCHEMA,
    ActionConditionedWorldModel,
    ActionConditionedWorldModelConfig,
    OneStepWorldModel,
    WorldModelConfig,
    action_conditioned_world_model_wrapper_state_nbytes_formula,
    measure_action_conditioned_world_model_wrapper_state_nbytes,
    measure_world_model_wrapper_state_nbytes,
    migrate_legacy_action_conditioned_world_model_state,
    migrate_legacy_world_model_state,
    one_step_world_model_wrapper_state_nbytes_formula,
    world_model_lifetime_counter_nbytes,
)

pytestmark = pytest.mark.unit

_FLOAT32_INTEGER_LIMIT = 2**24
_INT32_MAX = 2**31 - 1
_UINT32_MAX = 2**32 - 1
_OBSERVATION = jnp.asarray((0.25, -0.5), dtype=jnp.float32)
_NEXT_OBSERVATION = jnp.asarray((-0.75, 0.125), dtype=jnp.float32)
_ACTION = jnp.asarray(1, dtype=jnp.int32)
_REWARD = jnp.asarray(0.75, dtype=jnp.float32)
_DISCOUNT = jnp.asarray(0.6, dtype=jnp.float32)


def _action_model(*, normalizer=None) -> ActionConditionedWorldModel:  # type: ignore[no-untyped-def]
    return ActionConditionedWorldModel(
        ActionConditionedWorldModelConfig(
            observation_dim=2,
            n_actions=2,
            hidden_sizes=(),
            sparsity=0.0,
            use_layer_norm=False,
            error_decay=0.9,
        ),
        optimizer=LMS(step_size=0.05),
        normalizer=normalizer,
    )


def _one_step_model(*, normalizer=None) -> OneStepWorldModel:  # type: ignore[no-untyped-def]
    return OneStepWorldModel(
        WorldModelConfig(
            observation_dim=2,
            n_actions=2,
            hidden_sizes=(),
            sparsity=0.0,
            use_layer_norm=False,
        ),
        optimizer=LMS(step_size=0.05),
        normalizer=normalizer,
    )


def _action_update(model: ActionConditionedWorldModel, state):  # type: ignore[no-untyped-def]
    return model.update(
        state,
        _OBSERVATION,
        _ACTION,
        _REWARD,
        _DISCOUNT,
        _NEXT_OBSERVATION,
    )


def _one_step_update(model: OneStepWorldModel, state):  # type: ignore[no-untyped-def]
    return model.update(
        state,
        _OBSERVATION,
        _ACTION,
        _REWARD,
        _NEXT_OBSERVATION,
    )


def _assert_array_tree_bit_equal(first: object, second: object) -> None:
    first_leaves, first_tree = jax.tree_util.tree_flatten(first)
    second_leaves, second_tree = jax.tree_util.tree_flatten(second)
    assert first_tree == second_tree
    for first_leaf, second_leaf in zip(first_leaves, second_leaves, strict=True):
        if not isinstance(first_leaf, jax.Array) or not isinstance(
            second_leaf,
            jax.Array,
        ):
            continue
        np.testing.assert_array_equal(np.asarray(first_leaf), np.asarray(second_leaf))


def _assert_refused(result, state) -> None:  # type: ignore[no-untyped-def]
    assert not bool(result.update_applied)
    assert not bool(result.learner_result.update_applied)
    np.testing.assert_array_equal(
        result.pre_step_words,
        state.learner_state.step_words,
    )
    np.testing.assert_array_equal(
        result.post_step_words,
        state.learner_state.step_words,
    )
    np.testing.assert_array_equal(
        result.learner_result.post_step_words,
        state.learner_state.step_words,
    )
    _assert_array_tree_bit_equal(
        result.learner_result.state,
        state.learner_state,
    )
    _assert_array_tree_bit_equal(result.state, state)


def test_action_conditioned_early_update_matches_child_eager_and_jit() -> None:
    model = _action_model(normalizer=EMANormalizer(decay=0.9))
    state = model.init(jax.random.key(0))
    inputs = model.input_features(_OBSERVATION, _ACTION)
    targets = model.targets(
        _OBSERVATION,
        _REWARD,
        _DISCOUNT,
        _NEXT_OBSERVATION,
    )
    direct_child = model.learner.update(state.learner_state, inputs, targets)

    with jax.disable_jit():
        eager = _action_update(model, state)
    compiled = _action_update(model, state)

    for result in (eager, compiled):
        assert bool(result.wrapper_counter_aligned)
        assert bool(result.lifetime_counter_valid)
        assert bool(result.lifetime_capacity_available)
        assert bool(result.normalizer_counter_aligned)
        assert bool(result.normalizer_estimator_capacity_available)
        assert bool(result.update_applied)
        np.testing.assert_array_equal(result.pre_step_words, (0, 0))
        np.testing.assert_array_equal(result.post_step_words, (0, 1))
        np.testing.assert_array_equal(result.state.learner_state.step_words, (0, 1))
        np.testing.assert_array_equal(result.state.step_words, (0, 1))
        assert int(result.state.step_count) == 1
        _assert_array_tree_bit_equal(
            result.state.learner_state,
            direct_child.state,
        )
        np.testing.assert_array_equal(result.errors, direct_child.errors)
        np.testing.assert_array_equal(
            result.per_head_metrics,
            direct_child.per_head_metrics,
        )
        np.testing.assert_array_equal(
            result.state.observation_min,
            np.minimum(_OBSERVATION, _NEXT_OBSERVATION),
        )
        np.testing.assert_array_equal(
            result.state.observation_max,
            np.maximum(_OBSERVATION, _NEXT_OBSERVATION),
        )
        assert float(result.state.reward_min) == pytest.approx(float(_REWARD))
        assert float(result.state.reward_max) == pytest.approx(float(_REWARD))
        assert float(result.state.model_error_ema) == pytest.approx(
            float(result.prediction_error)
        )

    _assert_array_tree_bit_equal(eager.state, compiled.state)


def test_one_step_early_update_matches_child_and_propagates_transaction() -> None:
    model = _one_step_model()
    state = model.init(jax.random.key(1))
    inputs = model.input_features(_OBSERVATION, _ACTION)
    targets = model.targets(_OBSERVATION, _REWARD, _NEXT_OBSERVATION)
    direct_child = model.learner.update(state.learner_state, inputs, targets)

    result = _one_step_update(model, state)

    assert bool(result.update_applied)
    assert bool(result.wrapper_counter_aligned)
    assert bool(result.lifetime_counter_valid)
    assert bool(result.lifetime_capacity_available)
    assert bool(result.normalizer_counter_aligned)
    assert bool(result.normalizer_estimator_capacity_available)
    np.testing.assert_array_equal(result.pre_step_words, (0, 0))
    np.testing.assert_array_equal(result.post_step_words, (0, 1))
    np.testing.assert_array_equal(result.state.step_words, (0, 1))
    assert int(result.state.step_count) == 1
    _assert_array_tree_bit_equal(result.state.learner_state, direct_child.state)


def test_action_conditioned_scan_crosses_uint32_carry() -> None:
    model = _action_model()
    initial = model.init(jax.random.key(2))
    state = initial.replace(
        learner_state=initial.learner_state.replace(
            step_count=jnp.asarray(_INT32_MAX, dtype=jnp.int32),
            step_words=jnp.asarray((0, _UINT32_MAX), dtype=jnp.uint32),
        ),
        step_count=jnp.asarray(_INT32_MAX, dtype=jnp.int32),
        step_words=jnp.asarray((0, _UINT32_MAX), dtype=jnp.uint32),
    )

    def step(carry, unused):  # type: ignore[no-untyped-def]
        del unused
        result = _action_update(model, carry)
        return result.state, (
            result.pre_step_words,
            result.post_step_words,
            result.update_applied,
        )

    final_state, (pre_words, post_words, applied) = jax.lax.scan(
        step,
        state,
        jnp.arange(2, dtype=jnp.int32),
    )

    np.testing.assert_array_equal(applied, (True, True))
    np.testing.assert_array_equal(
        pre_words,
        np.asarray(((0, _UINT32_MAX), (1, 0)), dtype=np.uint32),
    )
    np.testing.assert_array_equal(
        post_words,
        np.asarray(((1, 0), (1, 1)), dtype=np.uint32),
    )
    np.testing.assert_array_equal(final_state.learner_state.step_words, (1, 1))
    np.testing.assert_array_equal(final_state.step_words, (1, 1))
    assert int(final_state.step_count) == _INT32_MAX


def test_nested_normalizer_misalignment_refuses_all_action_wrapper_mutation() -> None:
    model = _action_model(normalizer=EMANormalizer(decay=0.9))
    state = model.init(jax.random.key(3))
    normalizer_state = state.learner_state.normalizer_state
    assert normalizer_state is not None
    state = state.replace(
        learner_state=state.learner_state.replace(
            normalizer_state=normalizer_state.replace(
                sample_count=jnp.asarray(1, dtype=jnp.int32),
                sample_count_words=jnp.asarray((0, 1), dtype=jnp.uint32),
            )
        )
    )

    with jax.disable_jit():
        eager = _action_update(model, state)
    compiled = _action_update(model, state)

    for result in (eager, compiled):
        assert bool(result.wrapper_counter_aligned)
        assert not bool(result.lifetime_counter_valid)
        assert not bool(result.normalizer_counter_aligned)
        assert bool(result.lifetime_capacity_available)
        assert bool(result.normalizer_estimator_capacity_available)
        _assert_refused(result, state)


@pytest.mark.parametrize(
    "normalizer",
    (WelfordNormalizer(), EMANormalizer(decay=1.0)),
    ids=("welford", "ema-cumulative"),
)
def test_cumulative_estimator_horizon_refuses_all_action_wrapper_mutation(
    normalizer,
) -> None:  # type: ignore[no-untyped-def]
    model = _action_model(normalizer=normalizer)
    state = model.init(jax.random.key(4))
    normalizer_state = state.learner_state.normalizer_state
    assert normalizer_state is not None
    exact_words = jnp.asarray((0, _FLOAT32_INTEGER_LIMIT), dtype=jnp.uint32)
    state = state.replace(
        learner_state=state.learner_state.replace(
            step_count=jnp.asarray(_FLOAT32_INTEGER_LIMIT, dtype=jnp.int32),
            step_words=exact_words,
            normalizer_state=normalizer_state.replace(
                sample_count=jnp.asarray(_FLOAT32_INTEGER_LIMIT, dtype=jnp.int32),
                sample_count_words=exact_words,
            ),
        ),
        step_count=jnp.asarray(_FLOAT32_INTEGER_LIMIT, dtype=jnp.int32),
        step_words=exact_words,
    )

    result = _action_update(model, state)

    assert bool(result.lifetime_counter_valid)
    assert bool(result.wrapper_counter_aligned)
    assert bool(result.lifetime_capacity_available)
    assert bool(result.normalizer_counter_aligned)
    assert not bool(result.normalizer_estimator_capacity_available)
    _assert_refused(result, state)


def test_all_ones_clock_refuses_all_action_wrapper_mutation() -> None:
    model = _action_model(normalizer=EMANormalizer(decay=0.9))
    state = model.init(jax.random.key(5))
    normalizer_state = state.learner_state.normalizer_state
    assert normalizer_state is not None
    terminal_words = jnp.full((2,), _UINT32_MAX, dtype=jnp.uint32)
    state = state.replace(
        learner_state=state.learner_state.replace(
            step_count=jnp.asarray(_INT32_MAX, dtype=jnp.int32),
            step_words=terminal_words,
            normalizer_state=normalizer_state.replace(
                sample_count=jnp.asarray(_INT32_MAX, dtype=jnp.int32),
                sample_count_words=terminal_words,
            ),
        ),
        step_count=jnp.asarray(_INT32_MAX, dtype=jnp.int32),
        step_words=terminal_words,
    )

    result = _action_update(model, state)

    assert bool(result.lifetime_counter_valid)
    assert bool(result.wrapper_counter_aligned)
    assert not bool(result.lifetime_capacity_available)
    assert bool(result.normalizer_counter_aligned)
    assert bool(result.normalizer_estimator_capacity_available)
    _assert_refused(result, state)


def test_one_step_outer_counter_is_atomic_on_child_refusal() -> None:
    model = _one_step_model(normalizer=EMANormalizer(decay=0.9))
    state = model.init(jax.random.key(6))
    normalizer_state = state.learner_state.normalizer_state
    assert normalizer_state is not None
    state = state.replace(
        learner_state=state.learner_state.replace(
            normalizer_state=normalizer_state.replace(
                sample_count=jnp.asarray(1, dtype=jnp.int32),
                sample_count_words=jnp.asarray((0, 1), dtype=jnp.uint32),
            )
        )
    )

    result = _one_step_update(model, state)

    assert not bool(result.lifetime_counter_valid)
    assert bool(result.wrapper_counter_aligned)
    assert not bool(result.normalizer_counter_aligned)
    _assert_refused(result, state)


def test_action_wrapper_counter_mismatch_vetoes_child_in_eager_and_jit() -> None:
    model = _action_model(normalizer=EMANormalizer(decay=0.9))
    state = model.init(jax.random.key(7)).replace(
        step_count=jnp.asarray(1, dtype=jnp.int32)
    )

    with jax.disable_jit():
        eager = _action_update(model, state)
    compiled = _action_update(model, state)

    for result in (eager, compiled):
        assert not bool(result.wrapper_counter_aligned)
        assert not bool(result.lifetime_counter_valid)
        assert bool(result.lifetime_capacity_available)
        assert bool(result.normalizer_counter_aligned)
        assert bool(result.normalizer_estimator_capacity_available)
        assert not bool(result.learner_result.lifetime_counter_valid)
        _assert_refused(result, state)


def test_one_step_wrapper_counter_mismatch_vetoes_child_in_eager_and_jit() -> None:
    model = _one_step_model(normalizer=EMANormalizer(decay=0.9))
    state = model.init(jax.random.key(8)).replace(
        step_count=jnp.asarray(1, dtype=jnp.int32)
    )

    with jax.disable_jit():
        eager = _one_step_update(model, state)
    compiled = _one_step_update(model, state)

    for result in (eager, compiled):
        assert not bool(result.wrapper_counter_aligned)
        assert not bool(result.lifetime_counter_valid)
        assert bool(result.lifetime_capacity_available)
        assert bool(result.normalizer_counter_aligned)
        assert bool(result.normalizer_estimator_capacity_available)
        assert not bool(result.learner_result.lifetime_counter_valid)
        _assert_refused(result, state)


@pytest.mark.parametrize("model_kind", ("action", "one-step"))
def test_wrapper_exact_word_drift_vetoes_saturated_telemetry_eager_and_jit(
    model_kind: str,
) -> None:
    if model_kind == "action":
        model = _action_model()
        update = _action_update
    else:
        model = _one_step_model()
        update = _one_step_update
    initial = model.init(jax.random.key(9))
    child_words = jnp.asarray((1, 0), dtype=jnp.uint32)
    wrapper_words = jnp.asarray((0, _INT32_MAX), dtype=jnp.uint32)
    state = initial.replace(
        learner_state=initial.learner_state.replace(
            step_count=jnp.asarray(_INT32_MAX, dtype=jnp.int32),
            step_words=child_words,
        ),
        step_count=jnp.asarray(_INT32_MAX, dtype=jnp.int32),
        step_words=wrapper_words,
    )

    with jax.disable_jit():
        eager = update(model, state)
    compiled = update(model, state)

    for result in (eager, compiled):
        assert not bool(result.wrapper_counter_aligned)
        assert not bool(result.lifetime_counter_valid)
        assert bool(result.lifetime_capacity_available)
        assert bool(result.normalizer_counter_aligned)
        assert bool(result.normalizer_estimator_capacity_available)
        _assert_refused(result, state)


def test_world_model_wrapper_schema_migration_and_resource_contracts() -> None:
    action_model = _action_model(normalizer=EMANormalizer(decay=0.9))
    action_state = action_model.init(jax.random.key(10))
    action_learner = action_state.learner_state
    assert action_learner.normalizer_state is not None
    exact_words = jnp.asarray((0, 7), dtype=jnp.uint32)
    action_state = action_state.replace(
        learner_state=action_learner.replace(
            step_count=jnp.asarray(7, dtype=jnp.int32),
            step_words=exact_words,
            normalizer_state=action_learner.normalizer_state.replace(
                sample_count=jnp.asarray(7, dtype=jnp.int32),
                sample_count_words=exact_words,
            ),
        ),
        step_count=jnp.asarray(7, dtype=jnp.int32),
        step_words=exact_words,
    )
    action_legacy = {
        field.name: getattr(action_state, field.name)
        for field in dataclasses.fields(type(action_state))
        if field.name != "step_words"
    }
    migrated_action = migrate_legacy_action_conditioned_world_model_state(
        action_legacy
    )
    np.testing.assert_array_equal(migrated_action.step_words, exact_words)
    np.testing.assert_array_equal(
        migrated_action.step_words,
        migrated_action.learner_state.step_words,
    )

    one_step_model = _one_step_model()
    one_step_state = one_step_model.init(jax.random.key(11)).replace(
        learner_state=one_step_model.init(jax.random.key(11)).learner_state.replace(
            step_count=jnp.asarray(7, dtype=jnp.int32),
            step_words=exact_words,
        ),
        step_count=jnp.asarray(7, dtype=jnp.int32),
        step_words=exact_words,
    )
    one_step_legacy = {
        field.name: getattr(one_step_state, field.name)
        for field in dataclasses.fields(type(one_step_state))
        if field.name != "step_words"
    }
    migrated_one_step = migrate_legacy_world_model_state(one_step_legacy)
    np.testing.assert_array_equal(migrated_one_step.step_words, exact_words)

    action_config = action_model.to_config()
    one_step_config = one_step_model.to_config()
    assert action_config["state_schema"] == ACTION_CONDITIONED_WORLD_MODEL_STATE_SCHEMA
    assert one_step_config["state_schema"] == ONE_STEP_WORLD_MODEL_STATE_SCHEMA
    assert ActionConditionedWorldModel.from_config(action_config).to_config() == action_config
    assert OneStepWorldModel.from_config(one_step_config).to_config() == one_step_config
    bad_action_config = dict(action_config, state_schema="unsupported")
    with pytest.raises(ValueError, match="Unsupported.*state schema"):
        ActionConditionedWorldModel.from_config(bad_action_config)
    bad_one_step_config = dict(one_step_config, state_schema="unsupported")
    with pytest.raises(ValueError, match="Unsupported.*state schema"):
        OneStepWorldModel.from_config(bad_one_step_config)

    assert measure_action_conditioned_world_model_wrapper_state_nbytes(
        action_state
    ) == action_conditioned_world_model_wrapper_state_nbytes_formula(
        action_model.config
    )
    assert measure_world_model_wrapper_state_nbytes(
        one_step_state
    ) == one_step_world_model_wrapper_state_nbytes_formula()
    assert world_model_lifetime_counter_nbytes(learner_has_normalizer=False) == 24
    assert world_model_lifetime_counter_nbytes(learner_has_normalizer=True) == 36

    malformed = dict(action_legacy, unexpected=jnp.asarray(0))
    with pytest.raises(ValueError, match="field manifest is not exact"):
        migrate_legacy_action_conditioned_world_model_state(malformed)
    ambiguous = dict(one_step_legacy)
    ambiguous_current_learner = one_step_state.learner_state.replace(
        step_count=jnp.asarray(_INT32_MAX, dtype=jnp.int32)
    )
    ambiguous_learner = {
        field.name: getattr(ambiguous_current_learner, field.name)
        for field in dataclasses.fields(type(ambiguous_current_learner))
        if field.name != "step_words"
    }
    ambiguous["step_count"] = jnp.asarray(_INT32_MAX, dtype=jnp.int32)
    ambiguous["learner_state"] = ambiguous_learner
    with pytest.raises(ValueError, match="ambiguous"):
        migrate_legacy_world_model_state(ambiguous)
