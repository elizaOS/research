# mypy: disable-error-code="attr-defined,call-arg,no-any-return,operator"
"""Standalone routed linear ensemble over one authoritative changing bank."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import alberta_framework as alberta
import alberta_framework.core as core
from alberta_framework.core.feature_bank_router import FeatureBankRouterConfig
from alberta_framework.core.learning_signals import LearningSignalEstimatorConfig
from alberta_framework.core.prototype_feature_lifecycle import (
    PrototypeFeatureConsumerBinding,
    PrototypeFeatureLifecycle,
    PrototypeFeatureLifecycleConfig,
)
from alberta_framework.core.routed_linear_world_model_ensemble import (
    ROUTED_LINEAR_WORLD_MODEL_ENSEMBLE_CHECKPOINT_SCHEMA,
    ROUTED_LINEAR_WORLD_MODEL_ENSEMBLE_EVIDENCE_LEVEL,
    ROUTED_LINEAR_WORLD_MODEL_ENSEMBLE_OUTCOME_STATUS,
    RoutedLinearWorldModelEnsemble,
    RoutedLinearWorldModelEnsembleConfig,
    RoutedLinearWorldModelEnsembleTransition,
    load_routed_linear_world_model_ensemble_checkpoint,
    save_routed_linear_world_model_ensemble_checkpoint,
)
from alberta_framework.core.world_model import ActionConditionedWorldModelConfig

pytestmark = pytest.mark.integration

BASE_DIM = 3
ACTIVE_SLOTS = 2
TOTAL_DIM = BASE_DIM + ACTIVE_SLOTS
N_ACTIONS = 2
ENSEMBLE_SIZE = 2
TARGET_DIM = BASE_DIM + 2


def _lifecycle_config() -> PrototypeFeatureLifecycleConfig:
    return PrototypeFeatureLifecycleConfig(
        base_feature_dim=BASE_DIM,
        active_pair_slots=ACTIVE_SLOTS,
        candidate_pair_slots=1,
        n_tasks=1,
        n_options=1,
        n_primitive_actions=N_ACTIONS,
        option_subtask_feature_indices=(0,),
    )


def _config(*, ensemble_size: int = ENSEMBLE_SIZE) -> RoutedLinearWorldModelEnsembleConfig:
    return RoutedLinearWorldModelEnsembleConfig(
        router=FeatureBankRouterConfig(
            base_dim=BASE_DIM,
            active_slots=ACTIVE_SLOTS,
        ),
        world_model=ActionConditionedWorldModelConfig(
            observation_dim=BASE_DIM,
            n_actions=N_ACTIONS,
            gamma=0.99,
            hidden_sizes=(),
            step_size=0.1,
            sparsity=0.0,
            use_layer_norm=False,
            error_decay=0.5,
            include_action_interactions=False,
        ),
        signal_estimator=LearningSignalEstimatorConfig(
            ensemble_size=ensemble_size,
            target_dim=TARGET_DIM,
            progress_warmup_steps=2,
            change_calibration_steps=2,
            max_input_magnitude=1_000.0,
            max_predicted_variance=10_000.0,
            max_observed_loss=10_000.0,
        ),
        ensemble_size=ensemble_size,
        residual_variance_decay=0.8,
        residual_variance_warmup_steps=1,
        residual_variance_floor=1.0e-3,
        max_events=20,
        carry_survivors=True,
    )


def _setup() -> tuple[
    RoutedLinearWorldModelEnsemble,
    Any,
    PrototypeFeatureConsumerBinding,
    Any,
]:
    lifecycle = PrototypeFeatureLifecycle(_lifecycle_config())
    lifecycle_state, binding = lifecycle.init_bound(jr.key(10))
    model = RoutedLinearWorldModelEnsemble(_config())
    state = model.init(jr.key(11), binding, lifecycle_state.router_state)
    assert not bool(
        jnp.array_equal(
            _head_weights(state.member_states[0]),
            _head_weights(state.member_states[1]),
        )
    )
    return model, lifecycle_state.router_state, binding, state


def _destination(
    model: RoutedLinearWorldModelEnsemble,
    source_router: Any,
    descriptors: jax.Array | None = None,
) -> tuple[Any, PrototypeFeatureConsumerBinding]:
    if descriptors is None:
        descriptors = jnp.asarray(((0, 2), (1, 2)), dtype=jnp.int32)
    route = model.router.route(
        source_router,
        jnp.zeros((TOTAL_DIM,), dtype=jnp.float32),
        descriptors,
    )
    binding = PrototypeFeatureConsumerBinding(
        semantic_generation=route.state.generation_count,
        semantic_generation_words=route.state.generation_words,
        descriptors=route.state.descriptors,
    )
    return route.state, binding


def _event(
    model: RoutedLinearWorldModelEnsemble,
    state: Any,
    source_router: Any,
    destination_router: Any,
    destination_binding: PrototypeFeatureConsumerBinding,
) -> RoutedLinearWorldModelEnsembleTransition:
    prepared = model.prepare_transition(
        state,
        source_router,
        jnp.asarray((1.0, 2.0, 3.0), dtype=jnp.float32),
        jnp.asarray(1, dtype=jnp.int32),
    )
    assert bool(prepared.diagnostics.prepared)
    return RoutedLinearWorldModelEnsembleTransition(
        prepared=prepared.prepared,
        reward=jnp.asarray(0.75, dtype=jnp.float32),
        discount=jnp.asarray(0.9, dtype=jnp.float32),
        next_base_observation=jnp.asarray((1.5, 1.0, 4.0), dtype=jnp.float32),
        destination_router_state=destination_router,
        destination_binding=destination_binding,
    )


def _tree_equal(left: object, right: object) -> None:
    left_leaves, left_tree = jax.tree.flatten(left)
    right_leaves, right_tree = jax.tree.flatten(right)
    assert left_tree == right_tree
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        left_array = jnp.asarray(left_leaf)
        right_array = jnp.asarray(right_leaf)
        if jax.dtypes.issubdtype(left_array.dtype, jax.dtypes.prng_key):
            left_array = jr.key_data(left_array)
            right_array = jr.key_data(right_array)
        np.testing.assert_array_equal(np.asarray(left_array), np.asarray(right_array))


def _tree_close(left: object, right: object) -> None:
    left_leaves, left_tree = jax.tree.flatten(left)
    right_leaves, right_tree = jax.tree.flatten(right)
    assert left_tree == right_tree
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        left_array = jnp.asarray(left_leaf)
        right_array = jnp.asarray(right_leaf)
        if jax.dtypes.issubdtype(left_array.dtype, jax.dtypes.prng_key):
            left_array = jr.key_data(left_array)
            right_array = jr.key_data(right_array)
        if jnp.issubdtype(left_array.dtype, jnp.floating):
            np.testing.assert_allclose(
                np.asarray(left_array),
                np.asarray(right_array),
                rtol=2.0e-6,
                atol=2.0e-6,
            )
        else:
            np.testing.assert_array_equal(
                np.asarray(left_array),
                np.asarray(right_array),
            )


def _head_weights(member: Any) -> jax.Array:
    learner = getattr(member, "learner_state", member)
    return jnp.concatenate(learner.head_params.weights, axis=0)


def _head_weight_traces(member: Any) -> jax.Array:
    learner = getattr(member, "learner_state", member)
    return jnp.concatenate(
        tuple(trace[0] for trace in learner.head_traces),
        axis=0,
    )


def test_public_config_resources_and_checkpoint_own_no_lifecycle_or_authority(
    tmp_path: Path,
) -> None:
    model, router_state, binding, state = _setup()
    assert core.RoutedLinearWorldModelEnsemble is RoutedLinearWorldModelEnsemble
    assert alberta.ROUTED_LINEAR_WORLD_MODEL_ENSEMBLE_CHECKPOINT_SCHEMA == (
        ROUTED_LINEAR_WORLD_MODEL_ENSEMBLE_CHECKPOINT_SCHEMA
    )
    payload = model.to_config()
    restored = RoutedLinearWorldModelEnsemble.from_config(payload)
    assert restored.to_config() == payload
    assert "feature_lifecycle" not in payload
    assert payload["evidence_level"] == ROUTED_LINEAR_WORLD_MODEL_ENSEMBLE_EVIDENCE_LEVEL
    assert payload["outcome_status"] == ROUTED_LINEAR_WORLD_MODEL_ENSEMBLE_OUTCOME_STATUS
    assert payload["outcome_status"] == "not_assessed"
    assert model.state_valid(state)
    assert model.router_state_matches_binding(router_state, binding)
    budget = model.resource_budget
    assert budget.feature_lifecycle_state_owned == 0
    assert budget.router_state_owned == 0
    assert budget.planning_authority == 0
    assert budget.safety_authority == 0
    assert budget.max_router_evaluations_per_event == 1
    assert budget.max_member_updates_per_event == ENSEMBLE_SIZE

    checkpoint = tmp_path / "routed-ensemble"
    save_routed_linear_world_model_ensemble_checkpoint(model, state, checkpoint)
    loaded_model, loaded_state = load_routed_linear_world_model_ensemble_checkpoint(
        checkpoint
    )
    assert loaded_model.to_config() == payload
    _tree_equal(loaded_state, state)


def test_singleton_init_predict_update_config_and_resources_are_non_epistemic() -> None:
    config = _config(ensemble_size=1)
    model = RoutedLinearWorldModelEnsemble(config)
    assert RoutedLinearWorldModelEnsemble.from_config(model.to_config()).config == config
    with pytest.raises(ValueError, match=r">= 1"):
        dataclasses.replace(config, ensemble_size=0)

    lifecycle = PrototypeFeatureLifecycle(_lifecycle_config())
    lifecycle_state, binding = lifecycle.init_bound(jr.key(20))
    source_router = lifecycle_state.router_state
    state = model.init(jr.key(21), binding, source_router)
    assert len(state.member_states) == 1
    assert model.state_valid(state)
    assert state.residual_variances.shape == (1, TARGET_DIM)

    budget = model.resource_budget
    assert budget.ensemble_size == 1
    assert budget.max_member_updates_per_event == 1
    assert budget.max_router_evaluations_per_event == 1
    assert budget.persistent_state_scalars > 0
    assert budget.persistent_state_bytes > 0
    assert budget.prediction_scalars > 0
    assert budget.prediction_bytes > 0

    prediction = model.predict(
        state,
        source_router,
        jnp.asarray((1.0, 2.0, 3.0), dtype=jnp.float32),
        jnp.asarray(1, dtype=jnp.int32),
    )
    assert bool(prediction.valid)
    assert prediction.member_raw_predictions.shape == (1, TARGET_DIM)
    np.testing.assert_array_equal(
        jax.lax.bitcast_convert_type(
            prediction.per_head_epistemic_variance,
            jnp.uint32,
        ),
        jnp.zeros((TARGET_DIM,), dtype=jnp.uint32),
    )
    np.testing.assert_array_equal(
        jax.lax.bitcast_convert_type(prediction.epistemic_disagreement, jnp.uint32),
        jnp.asarray(0, dtype=jnp.uint32),
    )

    destination_router, destination_binding = _destination(model, source_router)
    event = _event(
        model,
        state,
        source_router,
        destination_router,
        destination_binding,
    )
    result = model.observe_and_route(state, source_router, event)
    assert bool(result.diagnostics.transaction_applied)
    assert bool(result.diagnostics.all_member_updates_applied)
    assert int(result.diagnostics.member_update_evaluations) == 1
    np.testing.assert_array_equal(result.member_updates_applied, [True])
    assert len(result.state.member_states) == 1
    assert model.state_valid(result.state)
    for value in (
        result.signals.epistemic_disagreement,
        result.signals.epistemic_surprise,
    ):
        np.testing.assert_array_equal(
            jax.lax.bitcast_convert_type(value, jnp.uint32),
            jnp.asarray(0, dtype=jnp.uint32),
        )
    assert bool(result.signals.availability.input_valid)
    assert not bool(result.signals.availability.epistemic)
    assert bool(result.signals.availability.aleatoric)
    assert bool(result.signals.availability.normalized_residual)
    assert not bool(result.signals.availability.learning_progress)
    assert bool(result.signals.counter_status.valid_event_recorded)
    np.testing.assert_array_equal(result.state.event_count_words, [0, 1])
    np.testing.assert_array_equal(result.state.signal_state.step_words, [0, 1])
    np.testing.assert_array_equal(result.state.signal_state.valid_words, [0, 1])
    assert bool(jnp.isfinite(result.signals.aleatoric_uncertainty))
    assert bool(jnp.isfinite(result.signals.normalized_residual))


def test_every_member_updates_old_input_then_one_route_moves_and_scrubs_columns() -> None:
    model, source_router, _, state = _setup()
    destination_router, destination_binding = _destination(model, source_router)
    event = _event(
        model,
        state,
        source_router,
        destination_router,
        destination_binding,
    )
    targets = model.targets(
        event.prepared.base_observation,
        event.reward,
        event.discount,
        event.next_base_observation,
    )
    ordinary_members = tuple(
        model.learner.update(
            member.learner_state,
            event.prepared.input_features,
            targets,
        ).state
        for member in state.member_states
    )
    result = model.observe_and_route(state, source_router, event)
    assert bool(result.diagnostics.transaction_applied)
    assert bool(result.diagnostics.one_authoritative_route_evaluated)
    assert int(result.diagnostics.router_evaluations) == 1
    np.testing.assert_array_equal(result.member_updates_applied, [True, True])
    assert int(result.state.event_count_words[1]) == 1
    assert int(result.state.signal_state.step_words[1]) == 1
    assert bool(result.signals.availability.epistemic)
    assert bool(result.signals.availability.aleatoric)
    assert not bool(result.signals.availability.learning_progress)
    np.testing.assert_array_equal(
        result.prediction.residual_variances,
        np.full((ENSEMBLE_SIZE, TARGET_DIM), 1.0e-3, dtype=np.float32),
    )
    expected_residuals = jnp.maximum(
        jnp.square(
            targets[None, :] - result.prediction.member_raw_predictions
        ),
        jnp.asarray(1.0e-3, dtype=jnp.float32),
    )
    np.testing.assert_array_equal(
        result.state.residual_variances,
        expected_residuals,
    )

    for member_index, routed_member in enumerate(result.state.member_states):
        ordinary = ordinary_members[member_index]
        ordinary_weights = _head_weights(ordinary)
        routed_weights = _head_weights(routed_member)
        ordinary_traces = _head_weight_traces(ordinary)
        routed_traces = _head_weight_traces(routed_member)
        # Stable base and action columns are untouched. Old generated slot 1
        # survives as new slot 0; the newborn destination slot is scrubbed.
        np.testing.assert_array_equal(
            routed_weights[:, :BASE_DIM], ordinary_weights[:, :BASE_DIM]
        )
        np.testing.assert_array_equal(
            routed_weights[:, TOTAL_DIM:], ordinary_weights[:, TOTAL_DIM:]
        )
        np.testing.assert_array_equal(
            routed_weights[:, BASE_DIM], ordinary_weights[:, BASE_DIM + 1]
        )
        np.testing.assert_array_equal(routed_weights[:, BASE_DIM + 1], 0.0)
        np.testing.assert_array_equal(
            routed_traces[:, BASE_DIM], ordinary_traces[:, BASE_DIM + 1]
        )
        np.testing.assert_array_equal(routed_traces[:, BASE_DIM + 1], 0.0)
        # Fixed physical heads never grow with the generated bank.
        assert len(routed_member.learner_state.head_params.weights) == TARGET_DIM
    assert int(result.diagnostics.physical_output_head_count) == TARGET_DIM
    assert int(result.diagnostics.generated_output_head_count) == 0


def test_preupdate_predictions_and_signal_inputs_ignore_destination_choice() -> None:
    model, source_router, _, state = _setup()
    destination_a, binding_a = _destination(model, source_router)
    destination_b, binding_b = _destination(
        model,
        source_router,
        jnp.asarray(((1, 2), (0, 1)), dtype=jnp.int32),
    )
    event_a = _event(
        model,
        state,
        source_router,
        destination_a,
        binding_a,
    )
    event_b = _event(
        model,
        state,
        source_router,
        destination_b,
        binding_b,
    )
    source_raw = jnp.stack(
        tuple(
            model.learner.predict(
                member.learner_state,
                event_a.prepared.input_features,
            )
            for member in state.member_states
        )
    )
    np.testing.assert_array_equal(
        event_a.prepared.prediction.member_raw_predictions,
        source_raw,
    )
    _tree_equal(event_a.prepared.prediction, event_b.prepared.prediction)

    result_a = model.observe_and_route(state, source_router, event_a)
    result_b = model.observe_and_route(state, source_router, event_b)
    assert bool(result_a.diagnostics.transaction_applied)
    assert bool(result_b.diagnostics.transaction_applied)
    _tree_equal(result_a.prediction, event_a.prepared.prediction)
    _tree_equal(result_b.prediction, event_a.prepared.prediction)
    # Signals consume only source-member means, pre-update residual proxies,
    # this outcome, and the prior signal state. Destination routing follows.
    _tree_equal(result_a.signals, result_b.signals)
    np.testing.assert_array_equal(
        result_a.member_prediction_losses,
        result_b.member_prediction_losses,
    )


def test_invalid_destination_route_rolls_back_members_signals_and_binding() -> None:
    model, source_router, binding, state = _setup()
    destination_router, destination_binding = _destination(model, source_router)
    invalid_binding = dataclasses.replace(
        destination_binding,
        descriptors=jnp.asarray(((0, 2), (0, 2)), dtype=jnp.int32),
    )
    event = _event(
        model,
        state,
        source_router,
        destination_router,
        invalid_binding,
    )
    result = model.observe_and_route(state, source_router, event)
    assert bool(result.diagnostics.rejected)
    assert not bool(result.signals.availability.input_valid)
    np.testing.assert_array_equal(result.member_updates_applied, [False, False])
    _tree_equal(result.state, state)
    _tree_equal(result.state.consumer_binding, binding)


def test_no_bank_change_is_one_exact_all_member_update_without_route_mutation() -> None:
    model, source_router, binding, state = _setup()
    event = _event(model, state, source_router, source_router, binding)
    result = model.observe_and_route(state, source_router, event)
    assert bool(result.diagnostics.transaction_applied)
    assert not bool(result.diagnostics.descriptors_changed)
    np.testing.assert_array_equal(result.member_updates_applied, [True, True])
    assert int(result.state.generation_update_words[1]) == 1
    _tree_equal(result.state.consumer_binding, binding)


def test_jit_matches_eager_and_checkpoint_continuation(tmp_path: Path) -> None:
    model, source_router, _, state = _setup()
    destination_router, destination_binding = _destination(model, source_router)
    event = _event(
        model,
        state,
        source_router,
        destination_router,
        destination_binding,
    )
    with jax.disable_jit():
        eager = model.observe_and_route(state, source_router, event)
    compiled = model.observe_and_route(state, source_router, event)
    _tree_close(compiled, eager)

    checkpoint = tmp_path / "continued"
    save_routed_linear_world_model_ensemble_checkpoint(
        model,
        compiled.state,
        checkpoint,
    )
    loaded_model, loaded_state = load_routed_linear_world_model_ensemble_checkpoint(
        checkpoint
    )
    next_event = _event(
        loaded_model,
        loaded_state,
        destination_router,
        destination_router,
        destination_binding,
    )
    resumed = loaded_model.observe_and_route(
        loaded_state,
        destination_router,
        next_event,
    )
    direct_event = _event(
        model,
        compiled.state,
        destination_router,
        destination_router,
        destination_binding,
    )
    direct = model.observe_and_route(
        compiled.state,
        destination_router,
        direct_event,
    )
    _tree_equal(resumed, direct)
    assert bool(resumed.signals.availability.learning_progress)
