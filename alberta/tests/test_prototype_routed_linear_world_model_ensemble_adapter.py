# mypy: disable-error-code="attr-defined,call-arg,no-any-return,operator"
"""Prototype-to-routed-ensemble ownership and atomic-adoption seam."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import alberta_framework as alberta
import alberta_framework.core as core
from alberta_framework.core.experiential_memory import ExperientialMemoryConfig
from alberta_framework.core.feature_bank_router import (
    FeatureBankRouterConfig,
    FeatureBankRouterState,
)
from alberta_framework.core.learning_signals import LearningSignalEstimatorConfig
from alberta_framework.core.oak import OaKConfig
from alberta_framework.core.options import STOMPConfig, SubtaskSpec
from alberta_framework.core.prototype_agent import (
    PrototypeAgentConfig,
    PrototypeAgentState,
    PrototypeAtomicFeatureWorldMemoryConfig,
    PrototypeExperientialMemoryInput,
    PrototypeFeatureOaKHordeState,
    PrototypeFeatureRepresentationState,
    PrototypeMemoryInteractionState,
    PrototypeTransition,
)
from alberta_framework.core.prototype_feature_lifecycle import (
    PrototypeFeatureConsumerBinding,
    PrototypeFeatureLifecycleConfig,
)
from alberta_framework.core.prototype_feature_memory import PrototypeFeatureMemoryState
from alberta_framework.core.prototype_routed_linear_world_model_ensemble_adapter import (
    PROTOTYPE_ROUTED_LINEAR_WORLD_MODEL_ENSEMBLE_ADAPTER_CHECKPOINT_SCHEMA,
    PROTOTYPE_ROUTED_LINEAR_WORLD_MODEL_ENSEMBLE_ADAPTER_EVIDENCE_LEVEL,
    PROTOTYPE_ROUTED_LINEAR_WORLD_MODEL_ENSEMBLE_ADAPTER_OUTCOME_STATUS,
    PrototypeRoutedLinearWorldModelEnsembleAdapter,
    PrototypeRoutedLinearWorldModelEnsembleAdapterConfig,
    PrototypeRoutedLinearWorldModelEnsembleAdapterState,
    load_prototype_routed_linear_world_model_ensemble_adapter_checkpoint,
    save_prototype_routed_linear_world_model_ensemble_adapter_checkpoint,
)
from alberta_framework.core.routed_linear_world_model_ensemble import (
    RoutedLinearWorldModelEnsembleConfig,
)
from alberta_framework.core.state_builder import IdentityStateBuilderConfig
from alberta_framework.core.types import (
    DemonType,
    GVFSpec,
    HordeSpec,
    create_horde_spec,
)
from alberta_framework.core.world_model import ActionConditionedWorldModelConfig

pytestmark = [pytest.mark.integration, pytest.mark.slow]

BASE_DIM = 3
PAIR_SLOTS = 2
TOTAL_DIM = BASE_DIM + PAIR_SLOTS
N_ACTIONS = 2
N_DEMONS = 1
ENSEMBLE_SIZE = 2
TARGET_DIM = BASE_DIM + 2


@pytest.fixture(autouse=True)
def _bounded_jax_execution(request: pytest.FixtureRequest):
    if request.node.name == "test_eager_jit_and_one_step_scan_boundaries":
        yield
    else:
        with jax.disable_jit():
            yield


def _feature_config(*, replacement_interval: int) -> PrototypeFeatureLifecycleConfig:
    return PrototypeFeatureLifecycleConfig(
        base_feature_dim=BASE_DIM,
        active_pair_slots=PAIR_SLOTS,
        candidate_pair_slots=3,
        n_tasks=1 + N_DEMONS,
        n_options=1,
        n_primitive_actions=N_ACTIONS,
        option_subtask_feature_indices=(0,),
        step_size_output=0.05,
        utility_decay=0.9,
        replacement_interval=replacement_interval,
        min_feature_age=0,
        candidate_min_age=0,
        promotion_margin=1.0,
        scale_normalizer_decay=0.9,
        scale_normalizer_epsilon=1.0e-6,
        carry_survivors=True,
        max_observations=100,
        managed_horde_demons=N_DEMONS,
    )


def _oak_config() -> OaKConfig:
    return OaKConfig(
        stomp=STOMPConfig(
            subtask_specs=(
                SubtaskSpec(
                    feature_index=0,
                    threshold=1_000_000.0,
                    max_option_steps=8,
                ),
            ),
            observation_dim=TOTAL_DIM,
            n_primitive_actions=N_ACTIONS,
            base_hidden_sizes=(),
            base_step_size=0.01,
            option_step_size=0.01,
            epsilon_base=0.0,
            epsilon_option=0.0,
            option_planning_backups_per_step=0,
        )
    )


def _horde_spec() -> HordeSpec:
    return create_horde_spec(
        (
            GVFSpec(
                name="prediction",
                demon_type=DemonType.PREDICTION,
                gamma=0.5,
                lamda=0.0,
                cumulant_index=0,
            ),
        )
    )


def _memory_config() -> ExperientialMemoryConfig:
    return ExperientialMemoryConfig(
        capacity=3,
        observation_dim=TOTAL_DIM,
        key_dim=TOTAL_DIM,
        action_dim=N_ACTIONS,
        outcome_dim=TOTAL_DIM + 1,
        top_k=1,
        min_neighbors=1,
        distance_scale=1.0,
        min_similarity=0.0,
        min_effective_reliability=0.01,
        max_uncertainty=1.0,
        max_safety_cost=1.0,
        max_age=100,
        staleness_scale=100.0,
        utility_decay=1.0,
        eviction_utility_weight=1.0,
        eviction_recency_weight=1.0,
        recency_scale=10.0,
    )


def _prototype_config(
    *,
    replacement_interval: int,
    memory: bool,
) -> PrototypeAgentConfig:
    return PrototypeAgentConfig(
        oak=_oak_config(),
        horde_spec=_horde_spec(),
        horde_hidden_sizes=(),
        horde_step_size=0.1,
        experiential_memory=_memory_config() if memory else None,
        state_builder=IdentityStateBuilderConfig(observation_dim=BASE_DIM),
        prototype_feature_lifecycle=_feature_config(
            replacement_interval=replacement_interval,
        ),
    )


def _ensemble_config() -> RoutedLinearWorldModelEnsembleConfig:
    return RoutedLinearWorldModelEnsembleConfig(
        router=FeatureBankRouterConfig(
            base_dim=BASE_DIM,
            active_slots=PAIR_SLOTS,
        ),
        world_model=ActionConditionedWorldModelConfig(
            observation_dim=BASE_DIM,
            n_actions=N_ACTIONS,
            gamma=0.99,
            hidden_sizes=(),
            step_size=0.02,
            sparsity=0.0,
            use_layer_norm=False,
            error_decay=0.5,
            include_action_interactions=False,
        ),
        signal_estimator=LearningSignalEstimatorConfig(
            ensemble_size=ENSEMBLE_SIZE,
            target_dim=TARGET_DIM,
            progress_warmup_steps=2,
            change_calibration_steps=2,
            max_input_magnitude=1_000.0,
            max_predicted_variance=10_000.0,
            max_observed_loss=10_000.0,
        ),
        ensemble_size=ENSEMBLE_SIZE,
        residual_variance_decay=0.8,
        residual_variance_warmup_steps=1,
        residual_variance_floor=1.0e-3,
        max_events=100,
        carry_survivors=True,
    )


def _adapter(
    *,
    replacement_interval: int = 0,
    memory: bool = False,
) -> PrototypeRoutedLinearWorldModelEnsembleAdapter:
    return PrototypeRoutedLinearWorldModelEnsembleAdapter(
        PrototypeRoutedLinearWorldModelEnsembleAdapterConfig(
            prototype=_prototype_config(
                replacement_interval=replacement_interval,
                memory=memory,
            ),
            ensemble=_ensemble_config(),
        )
    )


def _feature_state(state: PrototypeAgentState):
    wrapper = state.state_builder_state
    assert type(wrapper) is PrototypeFeatureRepresentationState
    return wrapper.feature_lifecycle_state


def _bundle(state: PrototypeAgentState) -> PrototypeFeatureOaKHordeState:
    assert type(state.oak_state) is PrototypeFeatureOaKHordeState
    return state.oak_state


def _binding(state: PrototypeAgentState) -> PrototypeFeatureConsumerBinding:
    return _bundle(state).consumer_binding


def _memory_wrapper(state: PrototypeAgentState) -> PrototypeFeatureMemoryState:
    interaction = state.ia_state
    assert type(interaction) is PrototypeMemoryInteractionState
    wrapper = interaction.experiential_memory_state
    assert type(wrapper) is PrototypeFeatureMemoryState
    return wrapper


def _start_idle(
    adapter: PrototypeRoutedLinearWorldModelEnsembleAdapter,
) -> PrototypeRoutedLinearWorldModelEnsembleAdapterState:
    observation = jnp.asarray((0.2, -0.1, 0.3), dtype=jnp.float32)
    for seed in range(32):
        state = adapter.start(adapter.init(jr.key(seed)), observation)
        if int(_bundle(state.prototype_state).oak_state.stomp_state.executing_option) == -1:
            return state
    raise AssertionError("could not initialize an idle primitive decision")


def _transition(state: PrototypeAgentState) -> PrototypeTransition:
    next_observation = jnp.asarray((0.1, 0.4, -0.2), dtype=jnp.float32)
    return PrototypeTransition(
        observation=state.current_raw_observation,
        action=state.current_action,
        decision_id=state.current_decision_id,
        reward=jnp.asarray(0.5, dtype=jnp.float32),
        discount=jnp.asarray(0.9, dtype=jnp.float32),
        terminated=jnp.asarray(False, dtype=jnp.bool_),
        truncated=jnp.asarray(False, dtype=jnp.bool_),
        next_observation=next_observation,
        next_decision_observation=next_observation,
        horde_cumulants=jnp.asarray((0.25,), dtype=jnp.float32),
    )


def _memory_input(state: PrototypeAgentState) -> PrototypeExperientialMemoryInput:
    binding = _binding(state)
    next_decision_id = state.current_decision_id.at[3].set(
        state.current_decision_id[3] + jnp.asarray(1, dtype=jnp.uint32)
    )
    return PrototypeExperientialMemoryInput(
        available=jnp.asarray(True, dtype=jnp.bool_),
        current_prototype_decision_id=state.current_decision_id,
        next_prototype_decision_id=next_decision_id,
        query_representation_version=binding.semantic_generation,
        entry_representation_version=binding.semantic_generation,
        query_uncertainty=jnp.asarray(0.1, dtype=jnp.float32),
        query_uncertainty_available=jnp.asarray(True, dtype=jnp.bool_),
        entry_uncertainty=jnp.asarray(0.1, dtype=jnp.float32),
        entry_uncertainty_available=jnp.asarray(True, dtype=jnp.bool_),
        safety_cost=jnp.asarray(0.0, dtype=jnp.float32),
        safety_cost_available=jnp.asarray(True, dtype=jnp.bool_),
        reliability=jnp.asarray(1.0, dtype=jnp.float32),
        utility=jnp.asarray(1.0, dtype=jnp.float32),
        utility_available=jnp.asarray(True, dtype=jnp.bool_),
        provenance_id=jnp.asarray(1, dtype=jnp.int32),
        source_id=jnp.asarray(9, dtype=jnp.int32),
        next_action_safety_mask=jnp.ones((N_ACTIONS,), dtype=jnp.bool_),
    )


def _force_promotion(
    state: PrototypeRoutedLinearWorldModelEnsembleAdapterState,
) -> PrototypeRoutedLinearWorldModelEnsembleAdapterState:
    prototype = state.prototype_state
    wrapper = cast(PrototypeFeatureRepresentationState, prototype.state_builder_state)
    feature_state = wrapper.feature_lifecycle_state
    learner = feature_state.learner_state
    active = set(
        zip(
            np.asarray(learner.feature_left).tolist(),
            np.asarray(learner.feature_right).tolist(),
            strict=True,
        )
    )
    candidates = list(
        zip(
            np.asarray(learner.candidate_left).tolist(),
            np.asarray(learner.candidate_right).tolist(),
            strict=True,
        )
    )
    candidate_index = next(
        index for index, pair in enumerate(candidates) if pair not in active
    )
    candidate_utilities = jnp.zeros_like(learner.candidate_utilities)
    candidate_utilities = candidate_utilities.at[candidate_index].set(0.9)
    feature_state = feature_state.replace(
        learner_state=learner.replace(
            utilities=jnp.asarray((0.0, 0.5), dtype=jnp.float32),
            candidate_utilities=candidate_utilities,
        )
    )
    prototype = prototype.replace(
        state_builder_state=wrapper.replace(feature_lifecycle_state=feature_state)
    )
    return state.replace(prototype_state=prototype)


def _tree_exact(left: object, right: object) -> None:
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


def _count_router_states(value: object) -> int:
    if type(value) is FeatureBankRouterState:
        return 1
    if dataclasses.is_dataclass(value):
        return sum(
            _count_router_states(getattr(value, field.name))
            for field in dataclasses.fields(value)
        )
    if isinstance(value, tuple):
        return sum(_count_router_states(item) for item in value)
    if isinstance(value, dict):
        return sum(_count_router_states(item) for item in value.values())
    return 0


def _head_weights(member: Any) -> jax.Array:
    learner = getattr(member, "learner_state", member)
    return jnp.concatenate(learner.head_params.weights, axis=0)


def _head_traces(member: Any) -> jax.Array:
    learner = getattr(member, "learner_state", member)
    return jnp.concatenate(tuple(trace[0] for trace in learner.head_traces), axis=0)


def test_feasibility_config_resources_checkpoint_and_exact_one_owner(
    tmp_path: Path,
) -> None:
    adapter = _adapter(memory=True)
    state = adapter.init(jr.key(3))
    assert core.PrototypeRoutedLinearWorldModelEnsembleAdapter is (
        PrototypeRoutedLinearWorldModelEnsembleAdapter
    )
    assert alberta.PROTOTYPE_ROUTED_LINEAR_WORLD_MODEL_ENSEMBLE_ADAPTER_CHECKPOINT_SCHEMA == (
        PROTOTYPE_ROUTED_LINEAR_WORLD_MODEL_ENSEMBLE_ADAPTER_CHECKPOINT_SCHEMA
    )
    payload = adapter.to_config()
    assert PrototypeRoutedLinearWorldModelEnsembleAdapter.from_config(
        payload
    ).to_config() == payload
    assert payload["prototype_v18_allowed"] is False
    assert payload["curation_recomputed"] is False
    assert payload["planning_authority"] is False
    assert payload["dispatch_authority"] is False
    assert payload["safety_authority"] is False
    assert payload["evidence_authority"] is False
    assert payload["evidence_level"] == (
        PROTOTYPE_ROUTED_LINEAR_WORLD_MODEL_ENSEMBLE_ADAPTER_EVIDENCE_LEVEL
    )
    assert payload["outcome_status"] == (
        PROTOTYPE_ROUTED_LINEAR_WORLD_MODEL_ENSEMBLE_ADAPTER_OUTCOME_STATUS
    )
    assert _count_router_states(state) == 1
    assert bool(adapter.state_valid(state))
    budget = adapter.resource_budget
    assert budget.feature_lifecycle_authority_count == 1
    assert budget.feature_router_authority_count == 1
    assert budget.external_ensemble_router_state_owned == 0
    assert budget.managed_linear_horde_count == 1
    assert budget.feature_bound_memory_count == 1
    assert budget.prototype_update_evaluations_per_event == 1
    assert budget.prototype_lifecycle_router_evaluations_per_event == 2
    assert budget.ensemble_router_evaluations_per_event == 1
    assert budget.total_bank_mapping_evaluations_per_event == 3
    assert budget.curation_recomputations_per_event == 0
    assert budget.memory_rebind_evaluations_per_event == 1
    assert budget.planning_authority == budget.dispatch_authority == 0
    assert budget.safety_authority == budget.evidence_authority == 0

    non_identity = _prototype_config(replacement_interval=0, memory=False)
    object.__setattr__(non_identity, "state_builder", None)
    with pytest.raises(ValueError, match="exact Identity"):
        PrototypeRoutedLinearWorldModelEnsembleAdapterConfig(
            prototype=non_identity,
            ensemble=_ensemble_config(),
        )
    v18 = _prototype_config(replacement_interval=0, memory=False)
    object.__setattr__(
        v18,
        "prototype_atomic_feature_world_memory",
        PrototypeAtomicFeatureWorldMemoryConfig(),
    )
    with pytest.raises(ValueError, match="separate from Prototype v18"):
        PrototypeRoutedLinearWorldModelEnsembleAdapterConfig(
            prototype=v18,
            ensemble=_ensemble_config(),
        )

    checkpoint = tmp_path / "prototype-routed-ensemble-adapter"
    save_prototype_routed_linear_world_model_ensemble_adapter_checkpoint(
        adapter,
        state,
        checkpoint,
    )
    restored_adapter, restored_state = (
        load_prototype_routed_linear_world_model_ensemble_adapter_checkpoint(
            checkpoint
        )
    )
    assert restored_adapter.to_config() == payload
    _tree_exact(restored_state, state)
    observation = jnp.asarray((0.2, -0.1, 0.3), dtype=jnp.float32)
    direct_source = adapter.start(state, observation)
    restored_source = restored_adapter.start(restored_state, observation)
    direct_transition = _transition(direct_source.prototype_state)
    restored_transition = _transition(restored_source.prototype_state)
    direct = adapter.step(
        direct_source,
        direct_transition,
        experiential_memory_input=_memory_input(direct_source.prototype_state),
    )
    resumed = restored_adapter.step(
        restored_source,
        restored_transition,
        experiential_memory_input=_memory_input(restored_source.prototype_state),
    )
    _tree_exact(resumed, direct)


def test_source_prediction_then_route_moves_and_scrubs_every_member() -> None:
    adapter = _adapter(replacement_interval=1)
    state = _force_promotion(_start_idle(adapter))
    transition = _transition(state.prototype_state)
    source_router = _feature_state(state.prototype_state).router_state
    direct_prediction = adapter.ensemble.predict(
        state.ensemble_state,
        source_router,
        state.prototype_state.current_raw_observation,
        transition.action,
    )
    prepared = adapter.prepare_transition(state, transition)
    _tree_exact(prepared.ensemble_prepared.prediction, direct_prediction)
    source_binding = _binding(state.prototype_state)
    destination_binding = _binding(prepared.prototype_result.state)
    assert not np.array_equal(
        np.asarray(source_binding.descriptors),
        np.asarray(destination_binding.descriptors),
    )
    targets = adapter.ensemble.targets(
        prepared.ensemble_prepared.base_observation,
        transition.reward,
        transition.discount,
        transition.next_observation,
    )
    ordinary = tuple(
        adapter.ensemble.learner.update(
            member.learner_state,
            prepared.ensemble_prepared.input_features,
            targets,
        ).state
        for member in state.ensemble_state.member_states
    )
    receipt = adapter.integrity_receipt(prepared)
    result = adapter.adopt_prepared_transition(state, prepared, receipt)
    assert bool(result.diagnostics.transaction_applied)
    assert int(result.diagnostics.prototype_update_evaluations) == 1
    assert int(result.diagnostics.prototype_lifecycle_router_evaluations) == 2
    assert int(result.diagnostics.ensemble_router_evaluations) == 1
    assert int(result.diagnostics.total_bank_mapping_evaluations) == 3
    assert int(result.diagnostics.curation_recomputations) == 0
    np.testing.assert_array_equal(
        result.ensemble_result.member_updates_applied,
        [True, True],
    )
    for member_index, routed_member in enumerate(
        result.state.ensemble_state.member_states
    ):
        ordinary_weights = _head_weights(ordinary[member_index])
        ordinary_traces = _head_traces(ordinary[member_index])
        routed_weights = _head_weights(routed_member)
        routed_traces = _head_traces(routed_member)
        np.testing.assert_array_equal(
            routed_weights[:, :BASE_DIM], ordinary_weights[:, :BASE_DIM]
        )
        np.testing.assert_array_equal(
            routed_weights[:, TOTAL_DIM:], ordinary_weights[:, TOTAL_DIM:]
        )
        for new_slot, descriptor in enumerate(
            np.asarray(destination_binding.descriptors).tolist()
        ):
            matching = np.flatnonzero(
                np.all(
                    np.asarray(source_binding.descriptors)
                    == np.asarray(descriptor),
                    axis=1,
                )
            )
            new_column = BASE_DIM + new_slot
            if matching.size:
                old_column = BASE_DIM + int(matching[0])
                np.testing.assert_array_equal(
                    routed_weights[:, new_column],
                    ordinary_weights[:, old_column],
                )
                np.testing.assert_array_equal(
                    routed_traces[:, new_column],
                    ordinary_traces[:, old_column],
                )
            else:
                np.testing.assert_array_equal(routed_weights[:, new_column], 0.0)
                np.testing.assert_array_equal(routed_traces[:, new_column], 0.0)


def test_unchanged_bank_updates_all_members_without_a_second_owner() -> None:
    adapter = _adapter()
    state = _start_idle(adapter)
    source_binding = _binding(state.prototype_state)
    result = adapter.step(state, _transition(state.prototype_state))
    assert bool(result.diagnostics.transaction_applied)
    assert not bool(result.diagnostics.descriptors_changed)
    assert _count_router_states(result.state) == 1
    _tree_exact(_binding(result.state.prototype_state), source_binding)
    _tree_exact(result.state.ensemble_state.consumer_binding, source_binding)
    np.testing.assert_array_equal(
        result.ensemble_result.member_updates_applied,
        [True, True],
    )
    assert int(result.state.ensemble_state.event_count_words[1]) == 1
    assert int(result.diagnostics.total_bank_mapping_evaluations) == 3


def test_stale_and_tampered_integrity_receipts_return_complete_source() -> None:
    adapter = _adapter(replacement_interval=1)
    state = _force_promotion(_start_idle(adapter))
    prepared = adapter.prepare_transition(state, _transition(state.prototype_state))
    receipt = adapter.integrity_receipt(prepared)
    tampered_route = dataclasses.replace(
        prepared.ensemble_prepared.source_router_state,
        route_words=prepared.ensemble_prepared.source_router_state.route_words.at[1].add(
            jnp.asarray(1, dtype=jnp.uint32)
        ),
    )
    tampered_prepared = prepared.replace(
        ensemble_prepared=prepared.ensemble_prepared.replace(
            source_router_state=tampered_route
        )
    )
    tampered_receipt = receipt.replace(prepared=tampered_prepared)
    refused = adapter.adopt_prepared_transition(state, prepared, tampered_receipt)
    assert bool(refused.diagnostics.rejected)
    assert not bool(refused.diagnostics.receipt_matches_preparation)
    _tree_exact(refused.state, state)
    _tree_exact(refused.prototype_result.state, state.prototype_state)
    _tree_exact(refused.ensemble_result.state, state.ensemble_state)

    accepted = adapter.adopt_prepared_transition(state, prepared, receipt)
    assert bool(accepted.diagnostics.transaction_applied)
    stale = adapter.adopt_prepared_transition(accepted.state, prepared, receipt)
    assert bool(stale.diagnostics.rejected)
    assert not bool(stale.diagnostics.source_state_matches)
    _tree_exact(stale.state, accepted.state)


def test_managed_horde_and_historical_feature_memory_coexist() -> None:
    adapter = _adapter(replacement_interval=1, memory=True)
    state = _force_promotion(_start_idle(adapter))
    source_horde_steps = _bundle(state.prototype_state).horde_state.step_words
    transition = _transition(state.prototype_state)
    memory_input = _memory_input(state.prototype_state)
    action_mask = jnp.ones((N_ACTIONS + 1,), dtype=jnp.bool_)
    direct_prototype = adapter.prototype.update_transition(
        state.prototype_state,
        transition,
        experiential_memory_input=memory_input,
        extended_action_mask=action_mask,
    )
    prepared = adapter.prepare_transition(
        state,
        transition,
        experiential_memory_input=memory_input,
        extended_action_mask=action_mask,
    )
    _tree_exact(prepared.prototype_result, direct_prototype)
    result = adapter.adopt_prepared_transition(
        state,
        prepared,
        adapter.integrity_receipt(prepared),
    )
    assert bool(result.diagnostics.transaction_applied)
    destination_binding = _binding(result.state.prototype_state)
    bundle = _bundle(result.state.prototype_state)
    memory = _memory_wrapper(result.state.prototype_state)
    assert int(bundle.horde_state.step_words[1]) == int(source_horde_steps[1]) + 1
    _tree_exact(bundle.consumer_binding, destination_binding)
    _tree_exact(memory.consumer_binding, destination_binding)
    _tree_exact(result.state.ensemble_state.consumer_binding, destination_binding)
    assert int(jnp.sum(memory.memory_state.entries.valid)) == 1
    assert int(memory.memory_state.entries.representation_versions[0]) == int(
        destination_binding.semantic_generation
    )
    assert result.prototype_result.prototype_feature_memory_diagnostics is not None


def test_eager_jit_and_one_step_scan_boundaries() -> None:
    adapter = _adapter()
    state = _start_idle(adapter)
    transition = _transition(state.prototype_state)
    with jax.disable_jit():
        eager = adapter.step(state, transition)
    compiled = jax.jit(adapter.step)(state, transition)
    _tree_close(compiled, eager)

    batched_transition = jax.tree.map(lambda value: value[None, ...], transition)
    scanned = adapter.scan_transitions(state, batched_transition)
    _tree_close(scanned.state, compiled.state)
    np.testing.assert_array_equal(
        scanned.transaction_applied,
        jnp.asarray((True,), dtype=jnp.bool_),
    )
