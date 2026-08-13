# mypy: disable-error-code="attr-defined,call-arg,no-untyped-def,override"
"""Atomic Prototype integration of one pair bank and every linear consumer."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import alberta_framework as alberta
import alberta_framework.core as core
import alberta_framework.core.prototype_agent as prototype_agent_module
from alberta_framework.core.checkpoints import (
    load_checkpoint_metadata,
    save_checkpoint,
)
from alberta_framework.core.experiential_memory import (
    ExperientialMemoryConfig,
    ExperientialMemoryState,
)
from alberta_framework.core.oak import OaKConfig
from alberta_framework.core.options import STOMPConfig, SubtaskSpec
from alberta_framework.core.prototype_agent import (
    PROTOTYPE_ATOMIC_FEATURE_WORLD_MEMORY_CHECKPOINT_SCHEMA,
    PROTOTYPE_ATOMIC_FEATURE_WORLD_MEMORY_CONFIG_SCHEMA,
    PrototypeAgent,
    PrototypeAgentConfig,
    PrototypeAgentState,
    PrototypeAtomicFeatureWorldMemoryConfig,
    PrototypeAtomicFeatureWorldMemoryState,
    PrototypeExperientialMemoryInput,
    PrototypeFeatureOaKHordeState,
    PrototypeFeatureRepresentationState,
    PrototypeMemoryInteractionState,
    PrototypeTransition,
    load_prototype_checkpoint,
    measure_prototype_agent_state_resources,
    save_prototype_checkpoint,
)
from alberta_framework.core.prototype_feature_lifecycle import (
    PrototypeFeatureConsumerBinding,
    PrototypeFeatureLifecycle,
    PrototypeFeatureLifecycleConfig,
)
from alberta_framework.core.prototype_feature_memory import (
    PrototypeFeatureMemory,
    PrototypeFeatureMemoryRebindResult,
    PrototypeFeatureMemoryState,
)
from alberta_framework.core.prototype_routed_linear_world_model import (
    PrototypeRoutedLinearWorldModel,
    PrototypeRoutedLinearWorldState,
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
N_OPTIONS = 1
N_DEMONS = 1


@pytest.fixture(autouse=True)
def _bounded_jax_execution(request: pytest.FixtureRequest):
    if request.node.name in {
        "test_atomic_transition_has_eager_jit_and_one_step_scan_parity",
    }:
        yield
    else:
        with jax.disable_jit():
            yield


def _feature_config(
    *,
    replacement_interval: int = 0,
    max_observations: int = 100,
) -> PrototypeFeatureLifecycleConfig:
    return PrototypeFeatureLifecycleConfig(
        base_feature_dim=BASE_DIM,
        active_pair_slots=PAIR_SLOTS,
        candidate_pair_slots=3,
        n_tasks=1 + N_DEMONS,
        n_options=N_OPTIONS,
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
        max_observations=max_observations,
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


def _world_config(*, hidden_sizes: tuple[int, ...] = ()) -> ActionConditionedWorldModelConfig:
    return ActionConditionedWorldModelConfig(
        observation_dim=BASE_DIM,
        n_actions=N_ACTIONS,
        hidden_sizes=hidden_sizes,
        step_size=0.02,
        sparsity=0.0,
        use_layer_norm=False,
        include_action_interactions=False,
    )


def _memory_config(*, capacity: int = 3) -> ExperientialMemoryConfig:
    return ExperientialMemoryConfig(
        capacity=capacity,
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


def _config(
    *,
    replacement_interval: int = 0,
    max_observations: int = 100,
    memory_capacity: int = 3,
    anchor_capacity: int = 3,
    world_model: ActionConditionedWorldModelConfig | None = None,
) -> PrototypeAgentConfig:
    return PrototypeAgentConfig(
        oak=_oak_config(),
        world_model=_world_config() if world_model is None else world_model,
        horde_spec=_horde_spec(),
        horde_hidden_sizes=(),
        horde_step_size=0.1,
        experiential_memory=_memory_config(capacity=memory_capacity),
        state_builder=IdentityStateBuilderConfig(observation_dim=BASE_DIM),
        prototype_feature_lifecycle=_feature_config(
            replacement_interval=replacement_interval,
            max_observations=max_observations,
        ),
        prototype_atomic_feature_world_memory=(
            PrototypeAtomicFeatureWorldMemoryConfig(
                anchor_capacity=anchor_capacity,
            )
        ),
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


def _world(state: PrototypeAgentState) -> PrototypeRoutedLinearWorldState:
    wrapper = state.world_model_state
    assert type(wrapper) is PrototypeAtomicFeatureWorldMemoryState
    return wrapper.world_state


def _memory_wrapper(state: PrototypeAgentState) -> PrototypeFeatureMemoryState:
    interaction = state.ia_state
    assert type(interaction) is PrototypeMemoryInteractionState
    wrapper = interaction.experiential_memory_state
    assert type(wrapper) is PrototypeFeatureMemoryState
    return wrapper


def _memory(state: PrototypeAgentState) -> ExperientialMemoryState:
    return _memory_wrapper(state).memory_state


def _start_idle(agent: PrototypeAgent) -> PrototypeAgentState:
    observation = jnp.asarray((0.2, -0.1, 0.3), dtype=jnp.float32)
    for seed in range(32):
        state = agent.start(agent.init(jr.key(seed)), observation)
        if int(_bundle(state).oak_state.stomp_state.executing_option) == -1:
            return state
    raise AssertionError("could not initialize an idle primitive decision")


def _next_decision_id(state: PrototypeAgentState) -> jax.Array:
    return state.current_decision_id.at[3].set(
        state.current_decision_id[3] + jnp.asarray(1, dtype=jnp.uint32)
    )


def _memory_input(
    state: PrototypeAgentState,
    *,
    provenance_id: int = 1,
) -> PrototypeExperientialMemoryInput:
    generation = _binding(state).semantic_generation
    return PrototypeExperientialMemoryInput(
        available=jnp.asarray(True, dtype=jnp.bool_),
        current_prototype_decision_id=state.current_decision_id,
        next_prototype_decision_id=_next_decision_id(state),
        query_representation_version=generation,
        entry_representation_version=generation,
        query_uncertainty=jnp.asarray(0.1, dtype=jnp.float32),
        query_uncertainty_available=jnp.asarray(True, dtype=jnp.bool_),
        entry_uncertainty=jnp.asarray(0.1, dtype=jnp.float32),
        entry_uncertainty_available=jnp.asarray(True, dtype=jnp.bool_),
        safety_cost=jnp.asarray(0.0, dtype=jnp.float32),
        safety_cost_available=jnp.asarray(True, dtype=jnp.bool_),
        reliability=jnp.asarray(1.0, dtype=jnp.float32),
        utility=jnp.asarray(1.0, dtype=jnp.float32),
        utility_available=jnp.asarray(True, dtype=jnp.bool_),
        provenance_id=jnp.asarray(provenance_id, dtype=jnp.int32),
        source_id=jnp.asarray(9, dtype=jnp.int32),
        next_action_safety_mask=jnp.ones((N_ACTIONS,), dtype=jnp.bool_),
    )


def _transition(
    state: PrototypeAgentState,
    *,
    reward: float = 0.5,
) -> PrototypeTransition:
    next_observation = jnp.asarray((0.1, 0.4, -0.2), dtype=jnp.float32)
    return PrototypeTransition(
        observation=state.current_raw_observation,
        action=state.current_action,
        decision_id=state.current_decision_id,
        reward=jnp.asarray(reward, dtype=jnp.float32),
        discount=jnp.asarray(0.9, dtype=jnp.float32),
        terminated=jnp.asarray(False, dtype=jnp.bool_),
        truncated=jnp.asarray(False, dtype=jnp.bool_),
        next_observation=next_observation,
        next_decision_observation=next_observation,
        horde_cumulants=jnp.asarray((0.25,), dtype=jnp.float32),
    )


def _force_promotion(state: PrototypeAgentState) -> PrototypeAgentState:
    wrapper = cast(PrototypeFeatureRepresentationState, state.state_builder_state)
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
    return cast(
        PrototypeAgentState,
        state.replace(
            state_builder_state=wrapper.replace(
                feature_lifecycle_state=feature_state,
            )
        ),
    )


def _force_next_option(state: PrototypeAgentState) -> PrototypeAgentState:
    bundle = _bundle(state)
    stomp = bundle.oak_state.stomp_state
    learner = stomp.base_learner_state
    option_action = N_ACTIONS
    biases = tuple(
        jnp.full_like(bias, 100.0 if index == option_action else -100.0)
        for index, bias in enumerate(learner.head_params.biases)
    )
    learner = learner.replace(
        head_params=learner.head_params.replace(biases=biases)
    )
    return cast(
        PrototypeAgentState,
        state.replace(
            oak_state=bundle.replace(
                oak_state=bundle.oak_state.replace(
                    stomp_state=stomp.replace(base_learner_state=learner),
                )
            )
        ),
    )


def _materialize_keys(tree: Any) -> Any:
    def convert(value: Any) -> Any:
        dtype = getattr(value, "dtype", None)
        if dtype is not None and jax.dtypes.issubdtype(dtype, jax.dtypes.prng_key):
            return jr.key_data(value)
        return value

    return jax.tree.map(convert, tree)


def _assert_tree_exact(left: Any, right: Any) -> None:
    left_leaves, left_tree = jax.tree.flatten(_materialize_keys(left))
    right_leaves, right_tree = jax.tree.flatten(_materialize_keys(right))
    assert left_tree == right_tree  # type: ignore[operator]
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        left_array = np.asarray(left_leaf)
        right_array = np.asarray(right_leaf)
        assert left_array.shape == right_array.shape
        assert left_array.dtype == right_array.dtype
        np.testing.assert_array_equal(
            np.frombuffer(left_array.tobytes(), dtype=np.uint8),
            np.frombuffer(right_array.tobytes(), dtype=np.uint8),
        )


def _assert_tree_portable(
    left: Any,
    right: Any,
    *,
    allowed_float_paths: frozenset[str],
) -> None:
    left_with_paths, left_tree = jax.tree_util.tree_flatten_with_path(
        _materialize_keys(left)
    )
    right_with_paths, right_tree = jax.tree_util.tree_flatten_with_path(
        _materialize_keys(right)
    )
    assert left_tree == right_tree  # type: ignore[operator]
    unexpected_paths: list[str] = []
    for (left_path, left_leaf), (right_path, right_leaf) in zip(
        left_with_paths,
        right_with_paths,
        strict=True,
    ):
        assert left_path == right_path
        left_array = np.asarray(left_leaf)
        right_array = np.asarray(right_leaf)
        assert left_array.shape == right_array.shape
        assert left_array.dtype == right_array.dtype
        left_bits = np.frombuffer(left_array.tobytes(), dtype=np.uint8)
        right_bits = np.frombuffer(right_array.tobytes(), dtype=np.uint8)
        if np.array_equal(left_bits, right_bits):
            continue
        path = jax.tree_util.keystr(left_path)
        if (
            not np.issubdtype(left_array.dtype, np.inexact)
            or path not in allowed_float_paths
        ):
            unexpected_paths.append(path)
            continue
        np.testing.assert_array_equal(np.isnan(left_array), np.isnan(right_array))
        np.testing.assert_array_equal(np.isposinf(left_array), np.isposinf(right_array))
        np.testing.assert_array_equal(np.isneginf(left_array), np.isneginf(right_array))
        left_zero = left_array == 0
        right_zero = right_array == 0
        np.testing.assert_array_equal(left_zero, right_zero)
        if np.any(left_zero):
            np.testing.assert_array_equal(
                np.signbit(left_array[left_zero]),
                np.signbit(right_array[right_zero]),
            )
        np.testing.assert_allclose(
            left_array,
            right_array,
            rtol=1.0e-6,
            atol=1.0e-7,
        )
    assert not unexpected_paths, f"unexpected non-bit-exact leaves: {unexpected_paths}"


class _DestinationVetoMemory(PrototypeFeatureMemory):
    """Reject only a real destination rebind while accepting source no-ops."""

    def rebind(self, state, source_binding, destination_binding):
        result = super().rebind(state, source_binding, destination_binding)
        required = result.diagnostics.rebind_required
        selected = jax.lax.cond(required, lambda _: state, lambda _: result.state, None)
        diagnostics = result.diagnostics.replace(
            candidate_state_valid=jnp.where(
                required,
                jnp.asarray(False, dtype=jnp.bool_),
                result.diagnostics.candidate_state_valid,
            ),
            transaction_applied=result.diagnostics.transaction_applied & ~required,
            valid_rows_reencoded=jnp.where(
                required,
                jnp.asarray(0, dtype=jnp.int32),
                result.diagnostics.valid_rows_reencoded,
            ),
            committed_generation_words=selected.consumer_binding.semantic_generation_words,
        )
        return PrototypeFeatureMemoryRebindResult(
            state=selected,
            diagnostics=diagnostics,
        )


class _TamperedReceiptLifecycle(PrototypeFeatureLifecycle):
    """Mint a receipt whose embedded preparation differs by one work count."""

    def horde_external_readiness_receipt(self, prepared, all_consumers_ready):
        receipt = super().horde_external_readiness_receipt(
            prepared,
            all_consumers_ready,
        )
        return receipt.replace(
            prepared_route=receipt.prepared_route.replace(
                preparation_learner_update_evaluations=jnp.asarray(
                    2,
                    dtype=jnp.int32,
                )
            )
        )


class _DestinationVetoWorld(PrototypeRoutedLinearWorldModel):
    """Keep the ordinary successor valid while refusing its routed sibling."""

    def prepare_observe_and_route(self, state, source_router_state, event):
        prepared = super().prepare_observe_and_route(
            state,
            source_router_state,
            event,
        )
        return prepared.replace(
            destination_valid=jnp.asarray(False, dtype=jnp.bool_),
        )


class _RouteInvalidLifecycle(PrototypeFeatureLifecycle):
    """Expose an internally rolled-back route while preserving ordinary work."""

    def prepare_observe_and_route_with_horde(
        self,
        state,
        oak_state,
        horde_state,
        consumer_binding,
        event,
        *,
        curation_priority_override=None,
    ):
        prepared = super().prepare_observe_and_route_with_horde(
            state,
            oak_state,
            horde_state,
            consumer_binding,
            event,
            curation_priority_override=curation_priority_override,
        )
        ordinary = prepared.ordinary_result
        destination = prepared.destination_result.replace(
            state=ordinary.state,
            oak_state=ordinary.oak_state,
            horde_state=ordinary.horde_state,
            consumer_binding=ordinary.consumer_binding,
            next_augmented_observation=ordinary.next_augmented_observation,
            diagnostics=prepared.destination_result.diagnostics.replace(
                routing_attempted=jnp.asarray(True, dtype=jnp.bool_),
                input_route_valid=jnp.asarray(False, dtype=jnp.bool_),
                curation_committed=jnp.asarray(False, dtype=jnp.bool_),
                curation_rolled_back=jnp.asarray(True, dtype=jnp.bool_),
                curation_deferred=jnp.asarray(False, dtype=jnp.bool_),
                transaction_applied=ordinary.diagnostics.transaction_applied,
                semantic_generation_after=(
                    ordinary.state.router_state.generation_count
                ),
                semantic_generation_words_after=(
                    ordinary.state.router_state.generation_words
                ),
            ),
            horde_diagnostics=ordinary.horde_diagnostics,
        )
        return prepared.replace(destination_result=destination)


def test_atomic_config_is_strict_opt_in_and_legacy_encoding_is_unchanged() -> None:
    public_names = (
        "PROTOTYPE_ATOMIC_FEATURE_WORLD_MEMORY_CHECKPOINT_SCHEMA",
        "PROTOTYPE_ATOMIC_FEATURE_WORLD_MEMORY_CONFIG_SCHEMA",
        "PROTOTYPE_ATOMIC_FEATURE_WORLD_MEMORY_MECHANISM_STATUS",
        "PROTOTYPE_ATOMIC_FEATURE_WORLD_MEMORY_SCIENTIFIC_PROMOTION_ALLOWED",
        "PrototypeAtomicFeatureWorldMemoryConfig",
        "PrototypeAtomicFeatureWorldMemoryDiagnostics",
        "PrototypeAtomicFeatureWorldMemoryResourceBudget",
        "PrototypeAtomicFeatureWorldMemoryState",
    )
    for name in public_names:
        value = getattr(prototype_agent_module, name)
        assert getattr(core, name) is value
        assert getattr(alberta, name) is value
        assert name in prototype_agent_module.__all__
        assert name in core.__all__
        assert name in alberta.__all__

    config = _config()
    payload = config.to_config()
    atomic_payload = payload["prototype_atomic_feature_world_memory"]
    assert atomic_payload["schema"] == PROTOTYPE_ATOMIC_FEATURE_WORLD_MEMORY_CONFIG_SCHEMA
    assert atomic_payload["planning_enabled"] is False
    assert PrototypeAgentConfig.from_config(payload).to_config() == payload

    legacy = PrototypeAgentConfig(oak=_oak_config())
    assert "prototype_atomic_feature_world_memory" not in legacy.to_config()
    with pytest.raises(ValueError, match="ordered managed linear-Horde"):
        PrototypeAgentConfig(
            oak=_oak_config(),
            prototype_atomic_feature_world_memory=(
                PrototypeAtomicFeatureWorldMemoryConfig()
            ),
        )
    with pytest.raises(ValueError, match="exact linear"):
        _config(world_model=_world_config(hidden_sizes=(4,)))


def test_atomic_state_ownership_and_resource_contract_are_exact() -> None:
    agent = PrototypeAgent(_config(anchor_capacity=2, memory_capacity=2))
    state = agent.init(jr.key(0))
    assert state.buffer_state is None
    assert state.horde_state is None
    assert type(state.world_model_state) is PrototypeAtomicFeatureWorldMemoryState
    assert type(_memory_wrapper(state)) is PrototypeFeatureMemoryState
    assert bool(agent.validate_state(state))
    budget = agent.prototype_atomic_feature_world_memory_resource_budget(state)
    assert budget.persistent_state_nbytes == (
        measure_prototype_agent_state_resources(state).total_nbytes
    )
    assert budget.persistent_capacity_growth == 0
    assert budget.lifecycle_authority_count == budget.router_authority_count == 1
    assert budget.oak_consumer_count == budget.ordered_linear_horde_count == 1
    assert budget.routed_world_count == budget.world_model_buffer_count == 1
    assert budget.experiential_memory_count == 1
    assert budget.mirrored_binding_cache_count == 3
    assert budget.feature_learner_update_evaluations_per_transition == 1
    assert budget.lifecycle_router_evaluations_per_transition == 2
    assert budget.world_learner_update_evaluations_per_transition == 1
    assert budget.world_router_evaluations_per_transition == 1
    assert budget.memory_rebind_evaluations_per_transition == 1
    assert budget.memory_step_evaluations_per_transition == 1
    assert budget.deterministic_prestate_memory_queries_per_transition == 2
    assert budget.memory_writes_attempted_per_transition == 1


def test_accepted_curation_updates_every_consumer_once_under_destination_identity() -> None:
    agent = PrototypeAgent(_config(replacement_interval=1))
    state = _force_promotion(_start_idle(agent))
    source_binding = _binding(state)
    result = agent.update_transition(
        state,
        _transition(state),
        experiential_memory_input=_memory_input(state),
    )
    diagnostics = result.prototype_atomic_feature_world_memory_diagnostics
    assert diagnostics is not None
    assert bool(diagnostics.descriptor_change_requested)
    assert bool(diagnostics.all_consumers_ready)
    assert bool(diagnostics.destination_adopted)
    assert not bool(diagnostics.ordinary_updates_retained)
    assert not bool(diagnostics.external_curation_rolled_back)
    assert int(diagnostics.oak_update_evaluations) == 1
    assert int(diagnostics.horde_update_evaluations) == 1
    assert int(diagnostics.feature_learner_update_evaluations) == 1
    assert int(diagnostics.lifecycle_router_evaluations) == 2
    assert int(diagnostics.world_learner_update_evaluations) == 1
    assert int(diagnostics.world_router_evaluations) == 1
    assert int(diagnostics.memory_rebind_evaluations) == 1
    assert int(diagnostics.memory_step_evaluations) == 1

    destination_binding = _binding(result.state)
    assert not np.array_equal(
        np.asarray(destination_binding.descriptors),
        np.asarray(source_binding.descriptors),
    )
    _assert_tree_exact(destination_binding, _world(result.state).consumer_binding)
    _assert_tree_exact(destination_binding, _memory_wrapper(result.state).consumer_binding)
    np.testing.assert_array_equal(
        np.asarray(_feature_state(result.state).router_state.descriptors),
        np.asarray(destination_binding.descriptors),
    )
    assert int(_bundle(result.state).oak_state.step_count) == int(state.step_count) + 1
    assert int(_bundle(result.state).horde_state.step_count) == int(state.step_count) + 1
    assert int(_world(result.state).model_state.step_count) == int(state.step_count) + 1
    assert int(_memory(result.state).step_count) == int(_memory(state).step_count) + 1
    assert int(jnp.sum(_memory(result.state).entries.valid)) == 1
    assert int(_memory(result.state).entries.representation_versions[0]) == int(
        destination_binding.semantic_generation
    )


def test_downstream_veto_retains_all_ordinary_updates_without_candidate_leak() -> None:
    agent = PrototypeAgent(_config(replacement_interval=1))
    state = _force_promotion(_start_idle(agent))
    source_binding = _binding(state)
    original_memory = agent.prototype_feature_memory
    assert original_memory is not None
    agent._prototype_feature_memory = _DestinationVetoMemory(original_memory.config)
    result = agent.update_transition(
        state,
        _transition(state),
        experiential_memory_input=_memory_input(state),
    )
    diagnostics = result.prototype_atomic_feature_world_memory_diagnostics
    assert diagnostics is not None
    assert bool(diagnostics.descriptor_change_requested)
    assert not bool(diagnostics.memory_destination_ready)
    assert not bool(diagnostics.all_consumers_ready)
    assert not bool(diagnostics.destination_adopted)
    assert bool(diagnostics.ordinary_updates_retained)
    assert bool(diagnostics.external_curation_rolled_back)
    lifecycle = result.prototype_feature_lifecycle_diagnostics
    assert lifecycle is not None
    assert bool(lifecycle.lifecycle.curation_rolled_back)
    assert not bool(lifecycle.lifecycle.curation_deferred)
    _assert_tree_exact(source_binding, _binding(result.state))
    _assert_tree_exact(source_binding, _world(result.state).consumer_binding)
    _assert_tree_exact(source_binding, _memory_wrapper(result.state).consumer_binding)
    assert int(_feature_state(result.state).observe_count) == int(
        _feature_state(state).observe_count
    ) + 1
    assert int(_feature_state(result.state).rolled_back_curation_count) == int(
        _feature_state(state).rolled_back_curation_count
    ) + 1
    assert int(_bundle(result.state).oak_state.step_count) == int(state.step_count) + 1
    assert int(_bundle(result.state).horde_state.step_count) == int(state.step_count) + 1
    assert int(_world(result.state).model_state.step_count) == int(state.step_count) + 1
    assert int(_world(result.state).generation_update_count) == (
        int(_world(state).generation_update_count) + 1
    )
    assert int(_memory(result.state).step_count) == int(_memory(state).step_count) + 1
    assert int(_memory(result.state).entries.representation_versions[0]) == int(
        source_binding.semantic_generation
    )


def test_world_veto_and_route_invalidity_both_choose_old_bank_successors() -> None:
    for veto_kind in ("world", "route"):
        agent = PrototypeAgent(_config(replacement_interval=1))
        state = _force_promotion(_start_idle(agent))
        source_binding = _binding(state)
        if veto_kind == "world":
            routed = agent.prototype_routed_linear_world_model
            assert routed is not None
            agent._prototype_routed_linear_world_model = _DestinationVetoWorld(
                routed.config
            )
        else:
            lifecycle = agent.prototype_feature_lifecycle
            assert lifecycle is not None
            agent._prototype_feature_lifecycle = _RouteInvalidLifecycle(
                lifecycle.config
            )
        result = agent.update_transition(
            state,
            _transition(state),
            experiential_memory_input=_memory_input(state),
        )
        atomic = result.prototype_atomic_feature_world_memory_diagnostics
        lifecycle_diagnostics = result.prototype_feature_lifecycle_diagnostics
        assert atomic is not None and lifecycle_diagnostics is not None
        assert not bool(atomic.all_consumers_ready)
        assert bool(atomic.ordinary_updates_retained)
        assert not bool(atomic.destination_adopted)
        assert bool(lifecycle_diagnostics.lifecycle.curation_rolled_back)
        assert not bool(lifecycle_diagnostics.lifecycle.curation_deferred)
        _assert_tree_exact(source_binding, _binding(result.state))
        _assert_tree_exact(source_binding, _world(result.state).consumer_binding)
        _assert_tree_exact(source_binding, _memory_wrapper(result.state).consumer_binding)
        assert int(_world(result.state).model_state.step_count) == int(state.step_count) + 1
        assert int(_memory(result.state).step_count) == int(_memory(state).step_count) + 1


def test_no_route_is_one_exact_old_bank_update_and_capacity_is_bounded() -> None:
    agent = PrototypeAgent(
        _config(
            replacement_interval=0,
            max_observations=1,
            memory_capacity=1,
            anchor_capacity=1,
        )
    )
    state = _start_idle(agent)
    source_binding = _binding(state)
    first = agent.update_transition(
        state,
        _transition(state),
        experiential_memory_input=_memory_input(state),
    )
    atomic = first.prototype_atomic_feature_world_memory_diagnostics
    assert atomic is not None
    assert not bool(atomic.descriptor_change_requested)
    assert bool(atomic.all_consumers_ready)
    assert not bool(atomic.external_curation_rolled_back)
    _assert_tree_exact(source_binding, _binding(first.state))
    assert int(_feature_state(first.state).observe_count) == 1
    assert int(_world(first.state).buffer_state.size) == 1
    assert int(_memory(first.state).entries.valid.sum()) == 1
    bytes_after_first = measure_prototype_agent_state_resources(
        first.state
    ).total_nbytes

    second = agent.update_transition(
        first.state,
        _transition(first.state),
        experiential_memory_input=_memory_input(first.state, provenance_id=2),
    )
    assert bool(second.transition_diagnostics.post_update_consistent)
    second_atomic = second.prototype_atomic_feature_world_memory_diagnostics
    second_lifecycle = second.prototype_feature_lifecycle_diagnostics
    assert second_atomic is not None and second_lifecycle is not None
    assert not bool(second_atomic.descriptor_change_requested)
    assert not bool(second_atomic.all_consumers_ready)
    assert not bool(second_atomic.destination_adopted)
    assert bool(second_atomic.ordinary_updates_retained)
    assert not bool(second_atomic.lifecycle_adoption.transaction_applied)
    assert bool(second_atomic.lifecycle_adoption.rejected)
    assert not bool(second_lifecycle.lifecycle.update_capacity_available)
    assert not bool(second_lifecycle.lifecycle.transaction_applied)
    assert not bool(second_lifecycle.lifecycle.curation_proposed)
    assert not bool(second_lifecycle.lifecycle.curation_deferred)
    assert not bool(second_lifecycle.lifecycle.routing_attempted)
    assert not bool(second_lifecycle.lifecycle.curation_committed)
    assert not bool(second_lifecycle.lifecycle.curation_rolled_back)
    _assert_tree_exact(source_binding, _binding(second.state))
    _assert_tree_exact(source_binding, _world(second.state).consumer_binding)
    _assert_tree_exact(source_binding, _memory_wrapper(second.state).consumer_binding)
    assert int(_feature_state(second.state).observe_count) == 1
    assert int(_feature_state(second.state).router_state.route_count) == 0
    lifecycle = agent.prototype_feature_lifecycle
    assert lifecycle is not None
    expected_representation = lifecycle.augment(
        _feature_state(second.state),
        second.state.current_raw_observation,
    )
    np.testing.assert_array_equal(
        np.asarray(second.state.current_representation),
        np.asarray(expected_representation),
    )
    assert int(_bundle(second.state).oak_state.step_count) == 2
    assert int(_bundle(second.state).horde_state.step_count) == 2
    assert int(_world(second.state).model_state.step_count) == 2
    assert int(_world(second.state).buffer_state.size) == 1
    assert int(_memory(second.state).entries.valid.sum()) == 1
    assert int(_memory(second.state).step_count) == int(_memory(first.state).step_count) + 1
    assert int(_memory(second.state).eviction_count) == 1
    assert int(_memory(second.state).entries.provenance_ids[0]) == 2
    assert measure_prototype_agent_state_resources(second.state).total_nbytes == (
        bytes_after_first
    )


def test_tampered_receipt_internal_invalidity_and_stale_transition_are_exact_noops() -> None:
    agent = PrototypeAgent(_config(replacement_interval=1))
    state = _force_promotion(_start_idle(agent))
    lifecycle = agent.prototype_feature_lifecycle
    assert lifecycle is not None
    agent._prototype_feature_lifecycle = _TamperedReceiptLifecycle(lifecycle.config)
    tampered = agent.update_transition(
        state,
        _transition(state),
        experiential_memory_input=_memory_input(state),
    )
    _assert_tree_exact(tampered.state, state)
    tampered_diagnostics = (
        tampered.prototype_atomic_feature_world_memory_diagnostics
    )
    assert tampered_diagnostics is not None
    assert not bool(
        tampered_diagnostics.lifecycle_adoption.receipt_matches_preparation
    )

    invalid_transition = _transition(state, reward=float("nan"))
    invalid = agent.update_transition(
        state,
        invalid_transition,
        experiential_memory_input=_memory_input(state),
    )
    _assert_tree_exact(invalid.state, state)
    assert not bool(invalid.transition_diagnostics.valid)

    clean_agent = PrototypeAgent(_config())
    clean_state = _start_idle(clean_agent)
    first = clean_agent.update_transition(
        clean_state,
        _transition(clean_state),
        experiential_memory_input=_memory_input(clean_state),
    )
    stale = clean_agent.update_transition(
        clean_state,
        _transition(first.state),
        experiential_memory_input=_memory_input(clean_state),
    )
    _assert_tree_exact(stale.state, clean_state)
    assert not bool(stale.transition_diagnostics.decision_id_matches)


def test_active_option_defers_internal_curation_without_downstream_route() -> None:
    agent = PrototypeAgent(_config(replacement_interval=1))
    state = _force_next_option(_start_idle(agent))
    first = agent.update_transition(
        state,
        _transition(state),
        experiential_memory_input=_memory_input(state),
    )
    assert int(_bundle(first.state).oak_state.stomp_state.executing_option) == 0
    active = _force_promotion(first.state)
    result = agent.update_transition(
        active,
        _transition(active),
        experiential_memory_input=_memory_input(active, provenance_id=2),
    )
    lifecycle = result.prototype_feature_lifecycle_diagnostics
    atomic = result.prototype_atomic_feature_world_memory_diagnostics
    assert lifecycle is not None and atomic is not None
    assert bool(lifecycle.lifecycle.curation_deferred)
    assert not bool(lifecycle.lifecycle.routing_attempted)
    assert not bool(atomic.descriptor_change_requested)
    assert bool(atomic.all_consumers_ready)


def test_v18_checkpoint_resume_and_schema_tamper_fail_closed(tmp_path: Path) -> None:
    agent = PrototypeAgent(_config())
    state = _start_idle(agent)
    result = agent.update_transition(
        state,
        _transition(state),
        experiential_memory_input=_memory_input(state),
    )
    path = tmp_path / "atomic"
    save_prototype_checkpoint(agent, result.state, path)
    metadata = load_checkpoint_metadata(path)
    assert metadata["schema"] == PROTOTYPE_ATOMIC_FEATURE_WORLD_MEMORY_CHECKPOINT_SCHEMA
    restored_agent, restored = load_prototype_checkpoint(path)
    assert restored_agent.to_config() == agent.to_config()
    _assert_tree_exact(restored, result.state)

    tampered_path = tmp_path / "tampered"
    save_checkpoint(
        result.state,
        tampered_path,
        metadata={
            **metadata,
            "atomic_feature_world_memory_schema_sha256": "00" * 32,
        },
    )
    with pytest.raises(ValueError, match="atomic feature/world/memory"):
        load_prototype_checkpoint(tampered_path)


def test_atomic_transition_has_eager_jit_and_one_step_scan_parity() -> None:
    agent = PrototypeAgent(_config())
    state = _start_idle(agent)
    transition = _transition(state)
    memory_input = _memory_input(state)
    with jax.disable_jit():
        eager = agent.update_transition(
            state,
            transition,
            experiential_memory_input=memory_input,
        )
    compiled = jax.jit(
        lambda source, event, sidecar: agent.update_transition(
            source,
            event,
            experiential_memory_input=sidecar,
        )
    )(state, transition, memory_input)
    # XLA may reassociate float32 linear-learner arithmetic by one ULP; all
    # identities, counters, descriptors, readiness bits, and schemas stay exact.
    _assert_tree_portable(
        eager,
        compiled,
        allowed_float_paths=frozenset(
            {
                ".behavior_gradient_result.diagnostics.gradient_norm",
                ".state.state_builder_state.feature_lifecycle_state.learner_state."
                "candidate_output_weights",
                ".state.state_builder_state.feature_lifecycle_state.learner_state."
                "output_biases",
                ".state.state_builder_state.feature_lifecycle_state.learner_state."
                "relevance_probe_biases",
                ".state.world_model_state.world_state.model_state.learner_state."
                "head_params.weights[0]",
            }
        ),
    )

    batched_transition = jax.tree.map(lambda value: value[None, ...], transition)
    batched_memory = jax.tree.map(lambda value: value[None, ...], memory_input)
    scanned = agent.scan_transitions(
        state,
        batched_transition,
        experiential_memory_input=batched_memory,
    )
    _assert_tree_portable(
        scanned.state,
        eager.state,
        allowed_float_paths=frozenset(
            {
                ".state_builder_state.feature_lifecycle_state.learner_state."
                "candidate_output_weights",
                ".state_builder_state.feature_lifecycle_state.learner_state."
                "output_biases",
                ".state_builder_state.feature_lifecycle_state.learner_state."
                "relevance_probe_biases",
                ".world_model_state.world_state.model_state.learner_state."
                "head_params.weights[0]",
            }
        ),
    )
