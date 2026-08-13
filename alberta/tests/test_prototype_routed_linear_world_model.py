# mypy: disable-error-code="attr-defined,call-arg,operator"
"""L0 tests for fixed-output world learning over routed generated inputs."""

from __future__ import annotations

import dataclasses
from typing import Any, cast

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest
from jax import Array

from alberta_framework.core.oak import OaKConfig, OaKState
from alberta_framework.core.options import STOMPConfig, SubtaskSpec
from alberta_framework.core.prototype_feature_lifecycle import (
    PrototypeFeatureConsumerBinding,
    PrototypeFeatureLifecycle,
    PrototypeFeatureLifecycleConfig,
)
from alberta_framework.core.prototype_routed_linear_world_model import (
    PROTOTYPE_ROUTED_LINEAR_WORLD_EVIDENCE_LEVEL,
    PROTOTYPE_ROUTED_LINEAR_WORLD_MECHANISM_STATUS,
    PrototypeRoutedLinearWorldConfig,
    PrototypeRoutedLinearWorldModel,
    PrototypeRoutedLinearWorldPlanRequest,
    PrototypeRoutedLinearWorldState,
    PrototypeRoutedLinearWorldTransition,
)
from alberta_framework.core.types import MLPParams
from alberta_framework.core.world_model import ActionConditionedWorldModelConfig

BASE_DIM = 3
ACTIVE_SLOTS = 2
TOTAL_DIM = BASE_DIM + ACTIVE_SLOTS
N_ACTIONS = 2
N_HEADS = BASE_DIM + 2
ANCHOR_CAPACITY = 3


def _feature_config() -> PrototypeFeatureLifecycleConfig:
    return PrototypeFeatureLifecycleConfig(
        base_feature_dim=BASE_DIM,
        active_pair_slots=ACTIVE_SLOTS,
        candidate_pair_slots=1,
        n_tasks=1,
        n_options=1,
        n_primitive_actions=N_ACTIONS,
        option_subtask_feature_indices=(0,),
    )


def _oak_config() -> OaKConfig:
    return OaKConfig(
        stomp=STOMPConfig(
            subtask_specs=(SubtaskSpec(feature_index=0),),
            observation_dim=TOTAL_DIM,
            n_primitive_actions=N_ACTIONS,
            base_hidden_sizes=(),
            option_planning_backups_per_step=0,
            epsilon_base=0.0,
            epsilon_option=0.0,
        )
    )


def _world_config(*, observation_dim: int = BASE_DIM) -> ActionConditionedWorldModelConfig:
    return ActionConditionedWorldModelConfig(
        observation_dim=observation_dim,
        n_actions=N_ACTIONS,
        gamma=0.99,
        predict_delta=True,
        hidden_sizes=(),
        step_size=0.1,
        sparsity=0.0,
        use_layer_norm=False,
        error_decay=0.5,
        observation_clip_margin=100.0,
        include_action_interactions=False,
    )


def _config(
    *,
    carry_survivors: bool = True,
    planning_enabled: bool = True,
) -> PrototypeRoutedLinearWorldConfig:
    return PrototypeRoutedLinearWorldConfig(
        feature_lifecycle=_feature_config(),
        world_model=_world_config(),
        oak=_oak_config(),
        anchor_capacity=ANCHOR_CAPACITY,
        planning_enabled=planning_enabled,
        planning_warmup_steps=1,
        max_generation_model_error=1.0e6,
        max_planned_backups=10,
        carry_survivors=carry_survivors,
    )


def _setup(
    *,
    carry_survivors: bool = True,
    planning_enabled: bool = True,
) -> tuple[
    PrototypeRoutedLinearWorldModel,
    Any,
    PrototypeFeatureConsumerBinding,
    PrototypeRoutedLinearWorldState,
]:
    model = PrototypeRoutedLinearWorldModel(
        _config(
            carry_survivors=carry_survivors,
            planning_enabled=planning_enabled,
        )
    )
    lifecycle = PrototypeFeatureLifecycle(_feature_config())
    lifecycle_state, binding = lifecycle.init_bound(jr.key(11))
    state = model.init(jr.key(12), binding, lifecycle_state.router_state)
    return model, lifecycle_state.router_state, binding, state


def _destination_receipt(
    model: PrototypeRoutedLinearWorldModel,
    source_router: Any,
) -> tuple[Any, PrototypeFeatureConsumerBinding]:
    # Source is [(0,1), (0,2)].  Destination moves survivor (0,2) from slot
    # one to slot zero and births (1,2) in slot one.
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
    model: PrototypeRoutedLinearWorldModel,
    state: PrototypeRoutedLinearWorldState,
    source_router: Any,
    destination_router: Any,
    destination_binding: PrototypeFeatureConsumerBinding,
    *,
    base: Array,
    next_base: Array,
    action: int,
    reward: float,
) -> PrototypeRoutedLinearWorldTransition:
    prepared = model.prepare_transition(
        state,
        source_router,
        base,
        jnp.asarray(action, dtype=jnp.int32),
    )
    assert bool(prepared.diagnostics.prepared)
    return PrototypeRoutedLinearWorldTransition(
        prepared=prepared.prepared,
        reward=jnp.asarray(reward, dtype=jnp.float32),
        discount=jnp.asarray(0.9, dtype=jnp.float32),
        next_base_observation=next_base,
        destination_router_state=destination_router,
        destination_binding=destination_binding,
    )


def _array_bits(value: Array) -> Array:
    array = jnp.asarray(value)
    if array.dtype == jnp.float32:
        return jax.lax.bitcast_convert_type(array, jnp.uint32)
    return array


def _assert_array_exact(left: Array, right: Array) -> None:
    left_array = jnp.asarray(left)
    right_array = jnp.asarray(right)
    assert left_array.shape == right_array.shape
    assert left_array.dtype == right_array.dtype
    assert bool(jnp.array_equal(_array_bits(left_array), _array_bits(right_array)))


def _assert_tree_exact(left: object, right: object) -> None:
    left_leaves, left_structure = jax.tree.flatten(left)
    right_leaves, right_structure = jax.tree.flatten(right)
    assert left_structure == right_structure
    assert len(left_leaves) == len(right_leaves)
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        if jax.dtypes.issubdtype(jnp.asarray(left_leaf).dtype, jax.dtypes.prng_key):
            _assert_array_exact(jr.key_data(left_leaf), jr.key_data(right_leaf))
        else:
            _assert_array_exact(jnp.asarray(left_leaf), jnp.asarray(right_leaf))


def _assert_tree_close(left: object, right: object) -> None:
    """Compare eager/compiled numerics while keeping discrete receipts exact."""

    left_leaves, left_structure = jax.tree.flatten(left)
    right_leaves, right_structure = jax.tree.flatten(right)
    assert left_structure == right_structure
    assert len(left_leaves) == len(right_leaves)
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        left_array = jnp.asarray(left_leaf)
        right_array = jnp.asarray(right_leaf)
        if jax.dtypes.issubdtype(left_array.dtype, jax.dtypes.prng_key):
            _assert_array_exact(jr.key_data(left_leaf), jr.key_data(right_leaf))
        elif jnp.issubdtype(left_array.dtype, jnp.inexact):
            assert bool(
                jnp.allclose(
                    left_array,
                    right_array,
                    rtol=1.0e-6,
                    atol=1.0e-7,
                    equal_nan=True,
                )
            )
        else:
            _assert_array_exact(left_array, right_array)


class _NoUpdateProxy:
    """Delegate learner metadata while forbidding adoption recomputation."""

    def __init__(self, delegate: Any) -> None:
        self._delegate = delegate

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)

    def update(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("adoption must not evaluate the learner")


class _NoRouteProxy:
    """Delegate router validation while forbidding adoption rerouting."""

    def __init__(self, delegate: Any) -> None:
        self._delegate = delegate

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)

    def route(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("adoption must not evaluate the router")


def _stack_weights(state: Any) -> Array:
    return jnp.concatenate(state.head_params.weights, axis=0)


def _stack_weight_traces(state: Any) -> Array:
    return jnp.concatenate(tuple(pair[0] for pair in state.head_traces), axis=0)


def _zero_dynamic_column(
    state: PrototypeRoutedLinearWorldState,
    column: int,
) -> PrototypeRoutedLinearWorldState:
    learner = state.model_state.learner_state
    weights = _stack_weights(learner).at[:, column].set(jnp.float32(0.0))
    traces = _stack_weight_traces(learner).at[:, column].set(jnp.float32(0.0))
    next_weights = tuple(weights[index : index + 1] for index in range(N_HEADS))
    next_traces = tuple(
        (traces[index : index + 1], learner.head_traces[index][1]) for index in range(N_HEADS)
    )
    next_learner = cast(
        Any,
        learner.replace(
            head_params=MLPParams(
                weights=next_weights,
                biases=learner.head_params.biases,
            ),
            head_traces=next_traces,
        ),
    )
    return cast(
        PrototypeRoutedLinearWorldState,
        state.replace(model_state=state.model_state.replace(learner_state=next_learner)),
    )


def test_config_keeps_outputs_physical_planning_default_off_and_resources_exact() -> None:
    default_off = dataclasses.replace(_config(), planning_enabled=False)
    assert default_off.planning_enabled is False
    assert default_off.to_config()["generated_output_semantics"] == ("unsupported-non-remappable")
    assert PrototypeRoutedLinearWorldConfig.from_config(default_off.to_config()) == default_off

    with pytest.raises(ValueError, match="unsupported and non-remappable"):
        PrototypeRoutedLinearWorldConfig(
            feature_lifecycle=_feature_config(),
            world_model=_world_config(observation_dim=TOTAL_DIM),
            oak=_oak_config(),
        )

    model, router, binding, state = _setup()
    budget = model.resource_budget()
    assert budget.mechanism_status == PROTOTYPE_ROUTED_LINEAR_WORLD_MECHANISM_STATUS
    assert budget.evidence_level == PROTOTYPE_ROUTED_LINEAR_WORLD_EVIDENCE_LEVEL
    assert budget.scientific_promotion_allowed is False
    assert budget.physical_output_heads == BASE_DIM + 2
    assert budget.generated_output_heads == 0
    assert budget.model_input_dim == TOTAL_DIM + N_ACTIONS
    assert budget.buffer_state_nbytes == 4 * ANCHOR_CAPACITY * BASE_DIM + 8
    assert budget.consumer_binding_nbytes == 8 * ACTIVE_SLOTS + 12
    assert budget.incremental_dynamic_input_nbytes == 8 * N_HEADS * ACTIVE_SLOTS
    assert budget.routed_input_feature_groups == 2 * N_HEADS
    assert budget.routed_input_scalars == 2 * N_HEADS * TOTAL_DIM
    assert budget.routed_dynamic_input_scalars == 2 * N_HEADS * ACTIVE_SLOTS
    assert budget.max_pair_products_per_transition_prepare == ACTIVE_SLOTS
    assert budget.max_pair_products_per_transition_consume == ACTIVE_SLOTS
    assert budget.max_pair_products_per_real_transition == 2 * ACTIVE_SLOTS
    assert budget.max_pair_products_per_plan_prepare == 2 * ACTIVE_SLOTS
    assert budget.max_pair_products_per_plan_consume == 2 * ACTIVE_SLOTS
    assert budget.max_pair_products_per_planning_call == 4 * ACTIVE_SLOTS
    assert budget.max_world_forwards_per_transition_prepare == 1
    assert budget.max_world_forwards_per_transition_consume == 2
    assert budget.max_world_forwards_per_real_transition == 3
    assert budget.max_world_forwards_per_plan_prepare == 1
    assert budget.max_world_forwards_per_plan_consume == 1
    assert budget.max_world_forwards_per_planning_call == 2
    assert budget.max_oak_base_backups_per_planning_call == 1
    assert budget.persistent_capacity_growth == 0
    assert budget.persistent_state_nbytes == sum(
        int(leaf.size) * int(leaf.dtype.itemsize)
        for leaf in jax.tree.leaves(state)
        if isinstance(leaf, Array)
    )
    prepared_transition = model.prepare_transition(
        state,
        router,
        jnp.zeros((BASE_DIM,), dtype=jnp.float32),
        jnp.asarray(0, dtype=jnp.int32),
    ).prepared
    assert budget.prepared_transition_cache_nbytes == sum(
        int(leaf.size) * int(leaf.dtype.itemsize)
        for leaf in jax.tree.leaves(prepared_transition)
        if isinstance(leaf, Array)
    )
    oak_state = model.oak.start(
        model.oak.init(jr.key(99)),
        model.augment(binding, jnp.zeros((BASE_DIM,), dtype=jnp.float32)),
    )
    plan_request = PrototypeRoutedLinearWorldPlanRequest(
        anchor_index=jnp.asarray(0, dtype=jnp.int32),
        primitive_action=jnp.asarray(0, dtype=jnp.int32),
        consumer_binding=binding,
        router_state=router,
        expected_model_step_words=state.model_state.step_words,
        expected_planned_backup_words=state.planned_backup_words,
        expected_oak_step_words=oak_state.step_words,
    )
    prepared_plan = model.prepare_plan(
        state,
        oak_state,
        router,
        plan_request,
    ).prepared
    assert budget.source_oak_state_nbytes == sum(
        int(leaf.size) * int(leaf.dtype.itemsize)
        for leaf in jax.tree.leaves(oak_state)
        if isinstance(leaf, Array)
    )
    assert budget.prepared_plan_cache_nbytes == sum(
        int(leaf.size) * int(leaf.dtype.itemsize)
        for leaf in jax.tree.leaves(prepared_plan)
        if isinstance(leaf, Array)
    )


def test_source_update_routes_survivor_scrubs_birth_and_changes_real_oak_backup() -> None:
    model, source_router, source_binding, source = _setup()
    destination_router, destination_binding = _destination_receipt(
        model,
        source_router,
    )
    base = jnp.asarray((1.0, 2.0, 3.0), dtype=jnp.float32)
    next_base = jnp.asarray((1.5, 1.0, 2.0), dtype=jnp.float32)
    event = _event(
        model,
        source,
        source_router,
        destination_router,
        destination_binding,
        base=base,
        next_base=next_base,
        action=1,
        reward=2.0,
    )

    # This direct update is the source-bank post-update state that must be
    # routed.  Its output count remains B+2; no pair target is invented.
    direct = model.learner.update(
        source.model_state.learner_state,
        event.prepared.input_features,
        model.targets(
            event.prepared.base_observation,
            event.reward,
            event.discount,
            event.next_base_observation,
        ),
    )
    assert direct.predictions.shape == (BASE_DIM + 2,)
    assert model.targets(base, event.reward, event.discount, next_base).shape == (BASE_DIM + 2,)
    old_weights = _stack_weights(direct.state)
    old_traces = _stack_weight_traces(direct.state)
    assert bool(jnp.any(_array_bits(old_weights[:, BASE_DIM + 0]) != 0))
    assert bool(jnp.any(_array_bits(old_weights[:, BASE_DIM + 1]) != 0))

    with jax.disable_jit():
        eager = model.observe_and_route(source, source_router, event)
    compiled = model.observe_and_route(source, source_router, event)
    _assert_tree_close(eager, compiled)
    result = compiled
    assert bool(result.diagnostics.transaction_applied)
    assert bool(result.diagnostics.prepared_source_state_matches)
    assert bool(result.diagnostics.prepared_source_router_matches)
    assert bool(result.diagnostics.source_cache_matches)
    assert bool(result.diagnostics.source_prediction_matches)
    _assert_tree_exact(result.prediction, event.prepared.prediction)
    assert bool(model.state_valid(result.state))
    assert result.targets.shape == (BASE_DIM + 2,)
    assert int(result.diagnostics.physical_output_head_count) == BASE_DIM + 2
    assert int(result.diagnostics.generated_output_head_count) == 0
    assert result.route_diagnostics.source_slots.tolist() == [1, -1]
    assert result.route_diagnostics.survivor_mask.tolist() == [True, False]
    assert result.route_diagnostics.new_mask.tolist() == [False, True]

    routed = result.state.model_state.learner_state
    routed_weights = _stack_weights(routed)
    routed_traces = _stack_weight_traces(routed)
    # Stable physical inputs and one-hot action inputs survive bit-for-bit.
    _assert_array_exact(routed_weights[:, :BASE_DIM], old_weights[:, :BASE_DIM])
    _assert_array_exact(routed_traces[:, :BASE_DIM], old_traces[:, :BASE_DIM])
    _assert_array_exact(routed_weights[:, TOTAL_DIM:], old_weights[:, TOTAL_DIM:])
    _assert_array_exact(routed_traces[:, TOTAL_DIM:], old_traces[:, TOTAL_DIM:])
    # Descriptor (0,2) moves from old dynamic slot one to new slot zero.
    _assert_array_exact(
        routed_weights[:, BASE_DIM + 0],
        old_weights[:, BASE_DIM + 1],
    )
    _assert_array_exact(
        routed_traces[:, BASE_DIM + 0],
        old_traces[:, BASE_DIM + 1],
    )
    # New descriptor (1,2) cannot inherit the evicted (0,1) column.
    _assert_array_exact(
        _array_bits(routed_weights[:, BASE_DIM + 1]),
        jnp.zeros((N_HEADS,), dtype=jnp.uint32),
    )
    _assert_array_exact(
        _array_bits(routed_traces[:, BASE_DIM + 1]),
        jnp.zeros((N_HEADS,), dtype=jnp.uint32),
    )
    for head in range(N_HEADS):
        _assert_array_exact(
            routed.head_params.biases[head],
            direct.state.head_params.biases[head],
        )
        _assert_array_exact(
            routed.head_traces[head][1],
            direct.state.head_traces[head][1],
        )
        _assert_tree_exact(
            routed.head_optimizer_states[head],
            direct.state.head_optimizer_states[head],
        )
    assert result.state.generation_update_words.tolist() == [0, 0]
    assert not bool(result.state.generation_error_valid)
    _assert_array_exact(
        result.state.generation_birth_model_step_words,
        result.state.model_state.step_words,
    )
    assert int(result.state.buffer_state.size) == 1
    _assert_array_exact(result.state.buffer_state.observations[0], next_base)

    cold_oak = model.oak.start(
        model.oak.init(jr.key(18)),
        model.augment(destination_binding, next_base),
    )
    cold_request = PrototypeRoutedLinearWorldPlanRequest(
        anchor_index=jnp.asarray(0, dtype=jnp.int32),
        primitive_action=jnp.asarray(0, dtype=jnp.int32),
        consumer_binding=destination_binding,
        router_state=destination_router,
        expected_model_step_words=result.state.model_state.step_words,
        expected_planned_backup_words=result.state.planned_backup_words,
        expected_oak_step_words=cold_oak.step_words,
    )
    cold_prepared = model.prepare_plan(
        result.state,
        cold_oak,
        destination_router,
        cold_request,
    )
    assert bool(cold_prepared.diagnostics.prepared)
    cold_plan = model.plan_one(
        result.state,
        cold_oak,
        destination_router,
        cold_prepared.prepared,
    )
    assert not bool(cold_plan.diagnostics.generation_warm)
    assert not bool(cold_plan.diagnostics.generation_error_ready)
    assert not bool(cold_plan.diagnostics.transaction_applied)
    _assert_tree_exact(cold_plan.state, result.state)
    _assert_tree_exact(cold_plan.oak_state, cold_oak)

    # The first destination-bank real event makes planning generation-local
    # rather than inheriting the old generation's error/warmup state.
    later_base = next_base
    later_next = jnp.asarray((2.0, 1.5, 1.0), dtype=jnp.float32)
    later_event = _event(
        model,
        result.state,
        destination_router,
        destination_router,
        destination_binding,
        base=later_base,
        next_base=later_next,
        action=0,
        reward=1.0,
    )
    later = model.observe_and_route(
        result.state,
        destination_router,
        later_event,
    )
    assert bool(later.diagnostics.transaction_applied)
    assert later.state.generation_update_words.tolist() == [0, 1]
    assert bool(later.state.generation_error_valid)

    oak_state = model.oak.start(
        model.oak.init(jr.key(19)),
        model.augment(destination_binding, later_next),
    )
    request = PrototypeRoutedLinearWorldPlanRequest(
        anchor_index=jnp.asarray(0, dtype=jnp.int32),
        primitive_action=jnp.asarray(0, dtype=jnp.int32),
        consumer_binding=destination_binding,
        router_state=destination_router,
        expected_model_step_words=later.state.model_state.step_words,
        expected_planned_backup_words=later.state.planned_backup_words,
        expected_oak_step_words=oak_state.step_words,
    )
    with jax.disable_jit():
        eager_prepared_plan = model.prepare_plan(
            later.state,
            oak_state,
            destination_router,
            request,
        )
        eager_plan = model.plan_one(
            later.state,
            oak_state,
            destination_router,
            eager_prepared_plan.prepared,
        )
    prepared_plan = model.prepare_plan(
        later.state,
        oak_state,
        destination_router,
        request,
    )
    _assert_tree_close(eager_prepared_plan, prepared_plan)
    assert bool(prepared_plan.diagnostics.prepared)
    assert bool(prepared_plan.diagnostics.request_router_matches_source)
    full_plan = model.plan_one(
        later.state,
        oak_state,
        destination_router,
        prepared_plan.prepared,
    )
    _assert_tree_close(eager_plan, full_plan)
    assert bool(full_plan.diagnostics.transaction_applied)
    assert bool(full_plan.diagnostics.source_oak_state_valid)
    assert bool(full_plan.diagnostics.prepared_source_state_matches)
    assert bool(full_plan.diagnostics.prepared_source_oak_state_matches)
    assert bool(full_plan.diagnostics.prepared_cache_matches)
    assert bool(full_plan.diagnostics.candidate_oak_state_valid)
    assert bool(full_plan.diagnostics.base_learner_changed)
    assert int(full_plan.diagnostics.pair_products_evaluated) == 2 * ACTIVE_SLOTS
    descriptors = destination_binding.descriptors
    predicted_base = full_plan.prediction.next_base_observation
    expected_pairs = predicted_base[descriptors[:, 0]] * predicted_base[descriptors[:, 1]]
    _assert_array_exact(
        full_plan.predicted_next_augmented_observation[:BASE_DIM],
        predicted_base,
    )
    _assert_array_exact(
        full_plan.predicted_next_augmented_observation[BASE_DIM:],
        expected_pairs,
    )

    # Only OaK's base learner is carried from the synthetic transition.
    restored_base = cast(
        OaKState,
        full_plan.oak_state.replace(
            stomp_state=full_plan.oak_state.stomp_state.replace(
                base_learner_state=oak_state.stomp_state.base_learner_state
            )
        ),
    )
    _assert_tree_exact(restored_base, oak_state)

    # The learned surviving generated column has real downstream influence:
    # zeroing only that column changes both the world proposal and the actual
    # OaK base backup, while the newborn did not inherit it at birth above.
    survivor_column = BASE_DIM
    assert bool(
        jnp.any(
            _array_bits(_stack_weights(later.state.model_state.learner_state)[:, survivor_column])
            != 0
        )
    )
    ablated_state = _zero_dynamic_column(later.state, survivor_column)
    assert bool(model.state_valid(ablated_state))
    ablated_prepared = model.prepare_plan(
        ablated_state,
        oak_state,
        destination_router,
        request,
    )
    assert bool(ablated_prepared.diagnostics.prepared)
    ablated_plan = model.plan_one(
        ablated_state,
        oak_state,
        destination_router,
        ablated_prepared.prepared,
    )
    assert bool(ablated_plan.diagnostics.transaction_applied)
    assert not bool(
        jnp.array_equal(
            _array_bits(full_plan.prediction.raw_predictions),
            _array_bits(ablated_plan.prediction.raw_predictions),
        )
    )
    assert not bool(
        jnp.array_equal(
            _array_bits(_stack_weights(full_plan.oak_state.stomp_state.base_learner_state)),
            _array_bits(_stack_weights(ablated_plan.oak_state.stomp_state.base_learner_state)),
        )
    )


def test_stale_tampered_nonfinite_and_disabled_planning_are_atomic_noops() -> None:
    model, source_router, _, source = _setup()
    destination_router, destination_binding = _destination_receipt(
        model,
        source_router,
    )
    base = jnp.asarray((1.0, 2.0, 3.0), dtype=jnp.float32)
    next_base = jnp.asarray((1.5, 1.0, 2.0), dtype=jnp.float32)
    valid_event = _event(
        model,
        source,
        source_router,
        destination_router,
        destination_binding,
        base=base,
        next_base=next_base,
        action=1,
        reward=2.0,
    )
    prepared = valid_event.prepared
    corrupt_cache = prepared.cached_augmented_observation.at[-1].add(jnp.float32(1.0))
    corrupt_input = prepared.input_features.at[-1].add(jnp.float32(1.0))
    corrupt_prediction = prepared.prediction.replace(
        raw_predictions=prepared.prediction.raw_predictions.at[0].add(jnp.float32(1.0))
    )
    tampered_destination = dataclasses.replace(
        destination_router, route_words=destination_router.route_words.at[1].add(jnp.uint32(1))
    )
    invalid_events = (
        valid_event.replace(prepared=prepared.replace(cached_augmented_observation=corrupt_cache)),
        valid_event.replace(prepared=prepared.replace(input_features=corrupt_input)),
        valid_event.replace(
            prepared=prepared.replace(
                base_observation=prepared.base_observation.at[0].add(jnp.float32(1.0))
            )
        ),
        valid_event.replace(
            prepared=prepared.replace(primitive_action=jnp.asarray(0, dtype=jnp.int32))
        ),
        valid_event.replace(prepared=prepared.replace(prediction=corrupt_prediction)),
        valid_event.replace(
            prepared=prepared.replace(prepared=jnp.asarray(False, dtype=jnp.bool_))
        ),
        valid_event.replace(next_base_observation=next_base.at[0].set(jnp.float32(jnp.nan))),
        valid_event.replace(destination_router_state=tampered_destination),
    )
    for invalid_event in invalid_events:
        rejected = model.observe_and_route(source, source_router, invalid_event)
        assert not bool(rejected.diagnostics.transaction_applied)
        _assert_tree_exact(rejected.state, source)

    # Full-state authentication rejects a same-clock, otherwise-valid model
    # substitution.  Explicit current-router authentication also rejects a
    # valid same-binding/same-descriptor router with a different route history.
    substituted_source = _zero_dynamic_column(source, BASE_DIM)
    assert bool(model.state_valid(substituted_source))
    state_rejected = model.observe_and_route(
        substituted_source,
        source_router,
        valid_event,
    )
    assert not bool(state_rejected.diagnostics.prepared_source_state_matches)
    assert not bool(state_rejected.diagnostics.transaction_applied)
    _assert_tree_exact(state_rejected.state, substituted_source)

    alternate_router = dataclasses.replace(
        source_router,
        route_count=source_router.route_count + jnp.int32(1),
        route_words=source_router.route_words.at[1].add(jnp.uint32(1)),
    )
    alternate_prepare = model.prepare_transition(
        source,
        alternate_router,
        base,
        jnp.asarray(1, dtype=jnp.int32),
    )
    assert bool(alternate_prepare.diagnostics.source_router_valid)
    assert bool(alternate_prepare.diagnostics.source_router_matches_binding)
    router_rejected = model.observe_and_route(
        source,
        alternate_router,
        valid_event,
    )
    assert bool(router_rejected.diagnostics.source_router_valid)
    assert bool(router_rejected.diagnostics.source_router_matches_binding)
    assert not bool(router_rejected.diagnostics.prepared_source_router_matches)
    assert not bool(router_rejected.diagnostics.transaction_applied)
    _assert_tree_exact(router_rejected.state, source)

    generation_router = dataclasses.replace(
        source_router,
        route_count=source_router.route_count + jnp.int32(1),
        route_words=source_router.route_words.at[1].add(jnp.uint32(1)),
        generation_count=source_router.generation_count + jnp.int32(1),
        generation_words=source_router.generation_words.at[1].add(jnp.uint32(1)),
    )
    generation_binding = PrototypeFeatureConsumerBinding(
        semantic_generation=generation_router.generation_count,
        semantic_generation_words=generation_router.generation_words,
        descriptors=generation_router.descriptors,
    )
    generation_source = cast(
        PrototypeRoutedLinearWorldState,
        source.replace(consumer_binding=generation_binding),
    )
    assert bool(model.state_valid(generation_source))
    generation_prepare = model.prepare_transition(
        generation_source,
        generation_router,
        base,
        jnp.asarray(1, dtype=jnp.int32),
    )
    assert bool(generation_prepare.diagnostics.prepared)
    generation_rejected = model.observe_and_route(
        generation_source,
        generation_router,
        valid_event,
    )
    assert bool(generation_rejected.diagnostics.source_router_valid)
    assert bool(generation_rejected.diagnostics.source_router_matches_binding)
    assert not bool(generation_rejected.diagnostics.prepared_source_state_matches)
    assert not bool(generation_rejected.diagnostics.prepared_source_router_matches)
    assert not bool(generation_rejected.diagnostics.transaction_applied)
    _assert_tree_exact(generation_rejected.state, generation_source)

    # Planning has the same full-snapshot boundary.  A valid baseline proves
    # that each rejection below is caused by provenance/cache substitution,
    # not by a warmup or planning-policy gate.
    warm = model.observe_and_route(
        source,
        source_router,
        _event(
            model,
            source,
            source_router,
            source_router,
            source.consumer_binding,
            base=base,
            next_base=next_base,
            action=0,
            reward=1.0,
        ),
    )
    assert bool(warm.diagnostics.transaction_applied)
    oak_state = model.oak.start(
        model.oak.init(jr.key(30)),
        model.augment(source.consumer_binding, next_base),
    )
    request = PrototypeRoutedLinearWorldPlanRequest(
        anchor_index=jnp.asarray(0, dtype=jnp.int32),
        primitive_action=jnp.asarray(0, dtype=jnp.int32),
        consumer_binding=source.consumer_binding,
        router_state=source_router,
        expected_model_step_words=warm.state.model_state.step_words,
        expected_planned_backup_words=warm.state.planned_backup_words,
        expected_oak_step_words=oak_state.step_words,
    )
    prepared_plan = model.prepare_plan(
        warm.state,
        oak_state,
        source_router,
        request,
    )
    assert bool(prepared_plan.diagnostics.prepared)
    mismatched_request_prepare = model.prepare_plan(
        warm.state,
        oak_state,
        source_router,
        request.replace(router_state=alternate_router),
    )
    assert not bool(mismatched_request_prepare.diagnostics.request_router_matches_source)
    assert not bool(mismatched_request_prepare.diagnostics.prepared)
    baseline_plan = model.plan_one(
        warm.state,
        oak_state,
        source_router,
        prepared_plan.prepared,
    )
    assert bool(baseline_plan.diagnostics.transaction_applied)
    assert bool(baseline_plan.diagnostics.candidate_oak_state_valid)

    alternate_oak = model.oak.start(
        model.oak.init(jr.key(31)),
        model.augment(source.consumer_binding, next_base),
    )
    _assert_array_exact(alternate_oak.step_words, oak_state.step_words)
    oak_rejected = model.plan_one(
        warm.state,
        alternate_oak,
        source_router,
        prepared_plan.prepared,
    )
    assert bool(oak_rejected.diagnostics.source_oak_state_valid)
    assert not bool(oak_rejected.diagnostics.prepared_source_oak_state_matches)
    assert not bool(oak_rejected.diagnostics.transaction_applied)
    _assert_tree_exact(oak_rejected.state, warm.state)
    _assert_tree_exact(oak_rejected.oak_state, alternate_oak)

    plan_router_rejected = model.plan_one(
        warm.state,
        oak_state,
        alternate_router,
        prepared_plan.prepared,
    )
    assert bool(plan_router_rejected.diagnostics.router_valid)
    assert bool(plan_router_rejected.diagnostics.router_matches_binding)
    assert not bool(plan_router_rejected.diagnostics.prepared_source_router_matches)
    assert not bool(plan_router_rejected.diagnostics.transaction_applied)
    _assert_tree_exact(plan_router_rejected.state, warm.state)
    _assert_tree_exact(plan_router_rejected.oak_state, oak_state)

    substituted_warm = _zero_dynamic_column(warm.state, BASE_DIM)
    assert bool(model.state_valid(substituted_warm))
    world_rejected = model.plan_one(
        substituted_warm,
        oak_state,
        source_router,
        prepared_plan.prepared,
    )
    assert not bool(world_rejected.diagnostics.prepared_source_state_matches)
    assert not bool(world_rejected.diagnostics.transaction_applied)
    _assert_tree_exact(world_rejected.state, substituted_warm)
    _assert_tree_exact(world_rejected.oak_state, oak_state)

    cached_plan = prepared_plan.prepared
    corrupt_plan_prediction = cached_plan.prediction.replace(
        raw_predictions=cached_plan.prediction.raw_predictions.at[0].add(jnp.float32(1.0))
    )
    invalid_prepared_plans = (
        cached_plan.replace(
            anchor_base_observation=cached_plan.anchor_base_observation.at[0].add(jnp.float32(1.0))
        ),
        cached_plan.replace(
            anchor_augmented_observation=(
                cached_plan.anchor_augmented_observation.at[-1].add(jnp.float32(1.0))
            )
        ),
        cached_plan.replace(
            predicted_next_augmented_observation=(
                cached_plan.predicted_next_augmented_observation.at[-1].add(jnp.float32(1.0))
            )
        ),
        cached_plan.replace(prediction=corrupt_plan_prediction),
        cached_plan.replace(
            request=cached_plan.request.replace(primitive_action=jnp.asarray(1, dtype=jnp.int32))
        ),
        cached_plan.replace(
            request=cached_plan.request.replace(anchor_index=jnp.asarray(1, dtype=jnp.int32))
        ),
        cached_plan.replace(prepared=jnp.asarray(False, dtype=jnp.bool_)),
    )
    for invalid_prepared_plan in invalid_prepared_plans:
        cache_rejected = model.plan_one(
            warm.state,
            oak_state,
            source_router,
            invalid_prepared_plan,
        )
        assert not bool(cache_rejected.diagnostics.transaction_applied)
        _assert_tree_exact(cache_rejected.state, warm.state)
        _assert_tree_exact(cache_rejected.oak_state, oak_state)

    # Build a warm state, then prove the default-off planning gate changes
    # neither its world/planning clock nor any OaK subtree.
    disabled_model, disabled_router, _, disabled_source = _setup(planning_enabled=False)
    same_binding = disabled_source.consumer_binding
    first = disabled_model.observe_and_route(
        disabled_source,
        disabled_router,
        _event(
            disabled_model,
            disabled_source,
            disabled_router,
            disabled_router,
            same_binding,
            base=base,
            next_base=next_base,
            action=0,
            reward=1.0,
        ),
    )
    assert bool(first.diagnostics.transaction_applied)
    oak_state = disabled_model.oak.start(
        disabled_model.oak.init(jr.key(29)),
        disabled_model.augment(same_binding, next_base),
    )
    request = PrototypeRoutedLinearWorldPlanRequest(
        anchor_index=jnp.asarray(0, dtype=jnp.int32),
        primitive_action=jnp.asarray(0, dtype=jnp.int32),
        consumer_binding=same_binding,
        router_state=disabled_router,
        expected_model_step_words=first.state.model_state.step_words,
        expected_planned_backup_words=first.state.planned_backup_words,
        expected_oak_step_words=oak_state.step_words,
    )
    prepared_plan = disabled_model.prepare_plan(
        first.state,
        oak_state,
        disabled_router,
        request,
    )
    assert bool(prepared_plan.diagnostics.prepared)
    rejected_plan = disabled_model.plan_one(
        first.state,
        oak_state,
        disabled_router,
        prepared_plan.prepared,
    )
    assert not bool(rejected_plan.diagnostics.planning_enabled)
    assert not bool(rejected_plan.diagnostics.transaction_applied)
    _assert_tree_exact(rejected_plan.state, first.state)
    _assert_tree_exact(rejected_plan.oak_state, oak_state)


def test_external_readiness_selects_routed_or_exact_source_updated_successor() -> None:
    model, source_router, _, source = _setup()
    destination_router, destination_binding = _destination_receipt(model, source_router)
    event = _event(
        model,
        source,
        source_router,
        destination_router,
        destination_binding,
        base=jnp.asarray((1.0, 2.0, 3.0), dtype=jnp.float32),
        next_base=jnp.asarray((1.5, 1.0, 2.0), dtype=jnp.float32),
        action=1,
        reward=2.0,
    )
    prepared = model.prepare_observe_and_route(source, source_router, event)
    assert bool(prepared.ordinary_valid)
    assert bool(prepared.destination_valid)
    model._learner = cast(Any, _NoUpdateProxy(model._learner))
    model._router = cast(Any, _NoRouteProxy(model._router))

    ready_receipt = model.external_readiness_receipt(
        prepared,
        jnp.asarray(True, dtype=jnp.bool_),
    )
    ready = model.adopt_prepared_route(source, source_router, prepared, ready_receipt)
    _assert_tree_exact(ready.result, prepared.destination_result)
    assert bool(ready.diagnostics.destination_adopted)

    veto_receipt = model.external_readiness_receipt(
        prepared,
        jnp.asarray(False, dtype=jnp.bool_),
    )
    veto = model.adopt_prepared_route(source, source_router, prepared, veto_receipt)
    compiled = jax.jit(model.adopt_prepared_route)(
        source,
        source_router,
        prepared,
        veto_receipt,
    )
    _assert_tree_exact(veto, compiled)
    _assert_tree_exact(veto.result, prepared.ordinary_result)
    _assert_tree_exact(veto.result.state.consumer_binding, source.consumer_binding)
    assert int(veto.result.state.model_state.step_count) == 1
    assert int(veto.result.state.generation_update_count) == 1
    assert int(veto.result.state.buffer_state.size) == 1
    assert bool(veto.diagnostics.ordinary_update_retained)
    assert bool(veto.diagnostics.external_route_rolled_back)
    assert int(veto.diagnostics.adoption_learner_update_evaluations) == 0
    assert int(veto.diagnostics.total_learner_update_evaluations) == 1
    assert int(veto.diagnostics.adoption_router_evaluations) == 0
    assert int(veto.diagnostics.total_router_evaluations) == 1

    transient = model.external_transaction_resource_budget(prepared, veto_receipt)
    prepared_nbytes = sum(
        int(leaf.size) * int(leaf.dtype.itemsize)
        for leaf in jax.tree.leaves(prepared)
        if isinstance(leaf, Array)
    )
    receipt_nbytes = sum(
        int(leaf.size) * int(leaf.dtype.itemsize)
        for leaf in jax.tree.leaves(veto_receipt)
        if isinstance(leaf, Array)
    )
    assert transient.prepared_adoption_logical_nbytes == prepared_nbytes
    assert transient.readiness_receipt_logical_nbytes == receipt_nbytes
    assert transient.simultaneous_logical_transient_nbytes == (
        prepared_nbytes + receipt_nbytes
    )
    assert (
        transient.persistent_state_nbytes_before
        == model.resource_budget().persistent_state_nbytes
    )
    assert transient.persistent_state_nbytes_after == transient.persistent_state_nbytes_before
    assert transient.learner_update_evaluations_per_prepare == 1
    assert transient.learner_update_evaluations_per_adopt == 0
    assert transient.router_evaluations_per_prepare == 1
    assert transient.router_evaluations_per_adopt == 0
    assert transient.persistent_capacity_growth == 0


def test_external_veto_retains_ordinary_update_when_destination_is_invalid() -> None:
    model, source_router, _, source = _setup()
    destination_router, destination_binding = _destination_receipt(model, source_router)
    duplicate_binding = destination_binding.replace(
        descriptors=destination_binding.descriptors.at[1].set(
            destination_binding.descriptors[0]
        )
    )
    event = _event(
        model,
        source,
        source_router,
        destination_router,
        duplicate_binding,
        base=jnp.asarray((1.0, 2.0, 3.0), dtype=jnp.float32),
        next_base=jnp.asarray((1.5, 1.0, 2.0), dtype=jnp.float32),
        action=1,
        reward=2.0,
    )
    prepared = model.prepare_observe_and_route(source, source_router, event)
    assert bool(prepared.ordinary_valid)
    assert not bool(prepared.destination_valid)
    assert bool(prepared.ordinary_result.diagnostics.transaction_applied)
    assert not bool(prepared.destination_result.diagnostics.transaction_applied)

    veto_receipt = model.external_readiness_receipt(
        prepared,
        jnp.asarray(False, dtype=jnp.bool_),
    )
    retained = model.adopt_prepared_route(
        source,
        source_router,
        prepared,
        veto_receipt,
    )
    assert bool(retained.diagnostics.transaction_applied)
    assert bool(retained.diagnostics.ordinary_update_retained)
    assert not bool(retained.diagnostics.destination_adopted)
    _assert_tree_exact(retained.result, prepared.ordinary_result)
    _assert_tree_exact(retained.result.state.consumer_binding, source.consumer_binding)
    assert int(retained.result.state.model_state.step_count) == 1

    ready_receipt = model.external_readiness_receipt(
        prepared,
        jnp.asarray(True, dtype=jnp.bool_),
    )
    rejected = model.adopt_prepared_route(
        source,
        source_router,
        prepared,
        ready_receipt,
    )
    assert bool(rejected.diagnostics.rejected)
    _assert_tree_exact(rejected.result.state, source)


def test_external_world_receipt_rejects_stale_and_tampered_preparations() -> None:
    model, source_router, _, source = _setup()
    destination_router, destination_binding = _destination_receipt(model, source_router)
    event = _event(
        model,
        source,
        source_router,
        destination_router,
        destination_binding,
        base=jnp.asarray((1.0, 2.0, 3.0), dtype=jnp.float32),
        next_base=jnp.asarray((1.5, 1.0, 2.0), dtype=jnp.float32),
        action=1,
        reward=2.0,
    )
    prepared = model.prepare_observe_and_route(source, source_router, event)
    receipt = model.external_readiness_receipt(
        prepared,
        jnp.asarray(True, dtype=jnp.bool_),
    )
    tampered = prepared.replace(
        destination_result=prepared.destination_result.replace(
            targets=prepared.destination_result.targets.at[0].add(jnp.float32(1.0))
        )
    )
    refused = model.adopt_prepared_route(source, source_router, tampered, receipt)
    assert not bool(refused.diagnostics.receipt_matches_preparation)
    assert bool(refused.diagnostics.rejected)
    _assert_tree_exact(refused.result.state, source)

    stale_source = prepared.destination_result.state
    stale = model.adopt_prepared_route(
        stale_source,
        source_router,
        prepared,
        receipt,
    )
    assert not bool(stale.diagnostics.source_state_matches)
    assert bool(stale.diagnostics.rejected)
    _assert_tree_exact(stale.result.state, stale_source)

    invalid_event = event.replace(reward=jnp.asarray(jnp.nan, dtype=jnp.float32))
    invalid_prepared = model.prepare_observe_and_route(
        source,
        source_router,
        invalid_event,
    )
    assert not bool(invalid_prepared.ordinary_valid)
    assert not bool(invalid_prepared.destination_valid)
    for ready in (False, True):
        invalid_receipt = model.external_readiness_receipt(
            invalid_prepared,
            jnp.asarray(ready, dtype=jnp.bool_),
        )
        invalid = model.adopt_prepared_route(
            source,
            source_router,
            invalid_prepared,
            invalid_receipt,
        )
        assert bool(invalid.diagnostics.rejected)
        _assert_tree_exact(invalid.result.state, source)
