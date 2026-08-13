# mypy: disable-error-code="arg-type,attr-defined,call-arg,no-any-return,type-var"
"""Red-first contracts for the external-coordinator repeated-option sidecar."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from test_authorized_option_replacement import (
    _context as _one_shot_context,
)
from test_authorized_option_replacement import (
    _transition as _swap_transition,
)

from alberta_framework.core.delight import CandidateUpdateAuditConfig
from alberta_framework.core.external_coordinator_repeated_option_sidecar import (
    EXTERNAL_COORDINATOR_REPEATED_OPTION_SIDECAR_STATUS,
    ExternalCoordinatorRepeatedOptionAtomicSwapPrepared,
    ExternalCoordinatorRepeatedOptionSidecar,
    ExternalCoordinatorRepeatedOptionSidecarState,
)
from alberta_framework.core.external_learned_state_router_audit_coordinator import (
    ExternalLearnedStateRouterAuditCoordinator,
    ExternalLearnedStateRouterAuditCoordinatorConfig,
    ExternalLearnedStateRouterAuditCoordinatorState,
    ExternalLearnedStateTransition,
)
from alberta_framework.core.feature_bank_router import FeatureBankRouterConfig
from alberta_framework.core.learning_signals import LearningSignalEstimatorConfig
from alberta_framework.core.learning_value_router import LearningValueRouterConfig
from alberta_framework.core.oak import OaKConfig
from alberta_framework.core.options import STOMPState
from alberta_framework.core.prototype_agent import PrototypeAgentConfig
from alberta_framework.core.prototype_feature_lifecycle import (
    PrototypeFeatureLifecycleConfig,
)
from alberta_framework.core.prototype_option_authority_bridge import (
    _prototype_oak_state,
    _replace_prototype_oak_state,
)
from alberta_framework.core.prototype_routed_linear_world_model_ensemble_adapter import (
    PrototypeRoutedLinearWorldModelEnsembleAdapterConfig,
)
from alberta_framework.core.repeated_option_lifecycle import (
    RepeatedOptionLifecycle,
    RepeatedOptionLifecycleConfig,
)
from alberta_framework.core.routed_linear_world_model_ensemble import (
    RoutedLinearWorldModelEnsembleConfig,
)
from alberta_framework.core.state_builder import (
    IdentityStateBuilderConfig,
    LearnableGRUStateBuilderConfig,
)
from alberta_framework.core.types import DemonType, GVFSpec, create_horde_spec
from alberta_framework.core.world_model import ActionConditionedWorldModelConfig

pytestmark = [pytest.mark.integration, pytest.mark.slow]

RAW_DIM = 6
HIDDEN_DIM = 1
BASE_DIM = RAW_DIM + HIDDEN_DIM
PAIR_SLOTS = 1
N_ACTIONS = 2
N_DEMONS = 1


@pytest.fixture(autouse=True)
def _bounded_jax_execution() -> object:
    with jax.disable_jit():
        yield


class _Context(NamedTuple):
    lower: Any
    coordinator: ExternalLearnedStateRouterAuditCoordinator
    lifecycle: RepeatedOptionLifecycle
    sidecar: ExternalCoordinatorRepeatedOptionSidecar
    source: ExternalCoordinatorRepeatedOptionSidecarState


def _coordinator_config(stomp: Any) -> ExternalLearnedStateRouterAuditCoordinatorConfig:
    assert stomp.observation_dim == BASE_DIM + PAIR_SLOTS
    n_options = stomp.n_options
    feature = PrototypeFeatureLifecycleConfig(
        base_feature_dim=BASE_DIM,
        active_pair_slots=PAIR_SLOTS,
        candidate_pair_slots=2,
        n_tasks=1 + N_DEMONS,
        n_options=n_options,
        n_primitive_actions=N_ACTIONS,
        option_subtask_feature_indices=tuple(
            spec.feature_index for spec in stomp.subtask_specs
        ),
        step_size_output=0.05,
        utility_decay=0.9,
        replacement_interval=0,
        min_feature_age=0,
        candidate_min_age=0,
        promotion_margin=1.0,
        scale_normalizer_decay=0.9,
        scale_normalizer_epsilon=1.0e-6,
        carry_survivors=True,
        max_observations=16,
        managed_horde_demons=N_DEMONS,
    )
    horde = create_horde_spec(
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
    prototype = PrototypeAgentConfig(
        oak=OaKConfig(stomp=stomp),
        horde_spec=horde,
        horde_hidden_sizes=(),
        horde_step_size=0.1,
        state_builder=IdentityStateBuilderConfig(observation_dim=BASE_DIM),
        prototype_feature_lifecycle=feature,
    )
    ensemble = RoutedLinearWorldModelEnsembleConfig(
        router=FeatureBankRouterConfig(base_dim=BASE_DIM, active_slots=PAIR_SLOTS),
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
            ensemble_size=2,
            target_dim=BASE_DIM + 2,
            progress_warmup_steps=2,
            change_calibration_steps=2,
            max_input_magnitude=1_000.0,
            max_predicted_variance=10_000.0,
            max_observed_loss=10_000.0,
        ),
        ensemble_size=2,
        residual_variance_decay=0.8,
        residual_variance_warmup_steps=1,
        residual_variance_floor=1.0e-3,
        max_events=16,
        carry_survivors=True,
    )
    return ExternalLearnedStateRouterAuditCoordinatorConfig(
        builder=LearnableGRUStateBuilderConfig(
            observation_dim=RAW_DIM,
            n_actions=N_ACTIONS,
            hidden_dim=HIDDEN_DIM,
            step_size=0.01,
            gradient_clip=10.0,
            initialization_scale=0.2,
            include_raw_observation=True,
        ),
        inner=PrototypeRoutedLinearWorldModelEnsembleAdapterConfig(
            prototype=prototype,
            ensemble=ensemble,
        ),
        learning_value_router=LearningValueRouterConfig(max_steps=16),
        candidate_audit=CandidateUpdateAuditConfig(candidate_semantics="update"),
        max_events=16,
    )


def _nested_stomp(state: ExternalLearnedStateRouterAuditCoordinatorState) -> STOMPState:
    prototype = state.inner_state.prototype_state
    return _prototype_oak_state(prototype.oak_state).stomp_state


def _bind_exact_owner(
    coordinator: ExternalLearnedStateRouterAuditCoordinator,
    state: ExternalLearnedStateRouterAuditCoordinatorState,
    stomp: STOMPState,
) -> ExternalLearnedStateRouterAuditCoordinatorState:
    prototype = state.inner_state.prototype_state
    oak = _prototype_oak_state(prototype.oak_state)
    rebound = _replace_prototype_oak_state(
        prototype,
        oak.replace(stomp_state=stomp),
    )
    candidate = state.replace(
        inner_state=state.inner_state.replace(prototype_state=rebound),
        current_action=rebound.current_action,
        current_decision_id=rebound.current_decision_id,
        cached_prototype_step_words=rebound.step_words,
    )
    assert bool(coordinator.state_valid(candidate))
    return candidate


@pytest.fixture(scope="module")
def context() -> _Context:
    lower = _one_shot_context(
        max_installations=8,
        reserved_observation_suffix=2,
    )
    stomp = lower.controller.scheduler.installation.stomp_agent.config
    coordinator = ExternalLearnedStateRouterAuditCoordinator(
        _coordinator_config(stomp)
    )
    coordinator_state = coordinator.init(
        jr.key(0xEC01),
        lifecycle_id=jnp.asarray((0xEC01, 0x0001), dtype=jnp.uint32),
    )
    canonical = (
        lower.pre_retirement_state.scheduler_state.installation_state
        .lifecycle_state.stomp_state
    )
    coordinator_state = _bind_exact_owner(coordinator, coordinator_state, canonical)
    lifecycle = RepeatedOptionLifecycle(
        lower.controller,
        RepeatedOptionLifecycleConfig(max_cycles=2),
    )
    repeated = lifecycle.init(lower.pre_retirement_state)
    sidecar = ExternalCoordinatorRepeatedOptionSidecar(coordinator, lifecycle)
    source = sidecar.init(coordinator_state, repeated)
    return _Context(lower, coordinator, lifecycle, sidecar, source)


def _count_exact_stomp_owners(value: object) -> int:
    if type(value) is STOMPState:
        return 1
    if dataclasses.is_dataclass(value):
        return sum(
            _count_exact_stomp_owners(getattr(value, field.name))
            for field in dataclasses.fields(value)
        )
    if isinstance(value, Mapping):
        return sum(_count_exact_stomp_owners(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_count_exact_stomp_owners(item) for item in value)
    return 0


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


def _raw_transition(
    state: ExternalLearnedStateRouterAuditCoordinatorState,
    *,
    next_observation: tuple[float, ...] = (0.2, -0.1, 0.3, 0.4, 0.15, -0.25),
) -> ExternalLearnedStateTransition:
    next_raw = jnp.asarray(next_observation, dtype=jnp.float32)
    return ExternalLearnedStateTransition(
        source_event_words=state.event_words,
        source_builder_step_words=state.cached_builder_step_words,
        source_prototype_step_words=state.cached_prototype_step_words,
        source_feature_generation_words=state.cached_feature_generation_words,
        observation=state.current_raw_observation,
        representation=state.current_representation,
        action=state.current_action,
        decision_id=state.current_decision_id,
        reward=jnp.asarray(0.5, dtype=jnp.float32),
        discount=jnp.asarray(0.9, dtype=jnp.float32),
        terminated=jnp.asarray(False, dtype=jnp.bool_),
        truncated=jnp.asarray(False, dtype=jnp.bool_),
        next_observation=next_raw,
        next_decision_observation=next_raw,
    )


def _evaluated(
    context: _Context,
    state: ExternalCoordinatorRepeatedOptionSidecarState,
) -> tuple[Any, Any]:
    prepared = context.coordinator.prepare_transition(
        state.coordinator_state,
        _raw_transition(state.coordinator_state),
        extended_action_mask=state.extended_action_mask,
    )
    evaluated = context.coordinator.evaluate_candidate(prepared)
    return evaluated, context.coordinator.integrity_receipt(evaluated)


def _started(context: _Context) -> ExternalCoordinatorRepeatedOptionSidecarState:
    result = context.sidecar.start(
        context.source,
        jnp.asarray((0.1, -0.2, 0.3, -0.4, 0.2, -0.1), dtype=jnp.float32),
    )
    assert bool(result.transaction_applied)
    assert bool(result.coordinator_started)
    assert bool(result.lifecycle_metadata_applied)
    return result.state


def test_exact_match_init_has_one_owner_and_delight_is_unavailable(
    context: _Context,
) -> None:
    state = context.source
    payload = context.sidecar.to_config()
    assert payload["mechanism_status"] == EXTERNAL_COORDINATOR_REPEATED_OPTION_SIDECAR_STATUS
    assert payload["coordinator_state_owners"] == 1
    assert payload["prototype_state_owners"] == 1
    assert payload["oak_state_owners"] == 1
    assert payload["stomp_state_owners"] == 1
    assert payload["additional_coordinator_state_owners"] == 0
    assert payload["additional_prototype_state_owners"] == 0
    assert payload["additional_bridge_state_owners"] == 0
    assert payload["detached_authority_metadata_stomp_state_owners"] == 0
    assert payload["repeated_overlay_stomp_state_owners"] == 0
    assert payload["atomic_swap_semantics"] == "all-installed-to-all-installed-only"
    assert payload["delight_available"] is False
    assert payload["delight_interpretation"] == "lifecycle-memory-metadata-only"
    assert payload["host_only"] is True
    assert payload["scan_supported"] is False

    assert bool(context.sidecar.state_valid(state))
    assert _count_exact_stomp_owners(state) == 1
    assert _count_exact_stomp_owners(state.coordinator_state) == 1
    assert _count_exact_stomp_owners(state.authority_metadata) == 0
    assert _count_exact_stomp_owners(state.lifecycle_metadata) == 0
    assert not hasattr(context.sidecar, "scan")

    mismatched_coordinator = context.coordinator.init(jr.key(0xBAD))
    repeated, attached = context.sidecar._attach_source(state)
    assert bool(attached)
    with pytest.raises(ValueError, match="exact|owner|match"):
        context.sidecar.init(mismatched_coordinator, repeated)


def test_raw_stomp_result_is_observed_once_and_adopted_atomically(
    context: _Context,
) -> None:
    source = _started(context)
    evaluated, receipt = _evaluated(context, source)
    result = context.sidecar.adopt_evaluated_transition(
        source,
        evaluated,
        receipt,
        downstream_candidate_valid=jnp.asarray(True, dtype=jnp.bool_),
    )
    prototype = evaluated.prepared.inner_result.prototype_result

    assert int(prototype.oak_real_stomp_update_evaluations) == 1
    assert int(result.raw_stomp_update_evaluations) == 1
    assert int(result.additional_stomp_update_evaluations) == 0
    assert int(result.lifecycle_observation_evaluations) == 1
    assert bool(result.raw_stomp_result_bound)
    assert bool(result.finalization_trace_bound)
    assert bool(result.raw_stomp_result_consumed)
    assert bool(result.coordinator_update_applied)
    assert bool(result.lifecycle_metadata_applied)
    assert bool(result.transaction_applied)
    assert not bool(result.delight_available)
    assert int(result.additional_delight_evaluations) == 0
    assert int(result.additional_actor_backward_calls) == 0
    _tree_exact(result.state.coordinator_state, evaluated.candidate_state)
    _tree_exact(
        prototype.oak_stomp_update_result.state,
        prototype.oak_owner_finalization_trace.raw_state,
    )
    _tree_exact(
        _nested_stomp(result.state.coordinator_state),
        prototype.oak_owner_finalization_trace.final_state,
    )
    assert _count_exact_stomp_owners(result.state) == 1
    assert bool(context.sidecar.state_valid(result.state))


def test_outer_rejection_tamper_and_replay_are_exact_source_noops(
    context: _Context,
) -> None:
    source = _started(context)
    evaluated, receipt = _evaluated(context, source)
    rejected = context.sidecar.adopt_evaluated_transition(
        source,
        evaluated,
        receipt,
        downstream_candidate_valid=jnp.asarray(False, dtype=jnp.bool_),
    )
    assert bool(rejected.coordinator_result.diagnostics.transaction_applied)
    assert bool(rejected.lifecycle_attempt.transaction_applied)
    assert not bool(rejected.coordinator_update_applied)
    assert not bool(rejected.lifecycle_metadata_applied)
    assert not bool(rejected.raw_stomp_result_consumed)
    assert not bool(rejected.transaction_applied)
    _tree_exact(rejected.state, source)

    tampered_receipt = receipt.replace(
        integrity_bound=jnp.asarray(False, dtype=jnp.bool_)
    )
    tampered = context.sidecar.adopt_evaluated_transition(
        source,
        evaluated,
        tampered_receipt,
        downstream_candidate_valid=jnp.asarray(True, dtype=jnp.bool_),
    )
    assert not bool(tampered.coordinator_result.diagnostics.transaction_applied)
    assert not bool(tampered.coordinator_update_applied)
    assert not bool(tampered.lifecycle_metadata_applied)
    assert not bool(tampered.transaction_applied)
    _tree_exact(tampered.state, source)

    accepted = context.sidecar.adopt_evaluated_transition(
        source,
        evaluated,
        receipt,
        downstream_candidate_valid=jnp.asarray(True, dtype=jnp.bool_),
    )
    assert bool(accepted.transaction_applied)
    replay = context.sidecar.adopt_evaluated_transition(
        accepted.state,
        evaluated,
        receipt,
        downstream_candidate_valid=jnp.asarray(True, dtype=jnp.bool_),
    )
    assert not bool(replay.coordinator_result.diagnostics.transaction_applied)
    assert not bool(replay.coordinator_update_applied)
    assert not bool(replay.lifecycle_metadata_applied)
    assert not bool(replay.transaction_applied)
    _tree_exact(replay.state, accepted.state)


def _prepared_swap(
    context: _Context,
) -> tuple[ExternalCoordinatorRepeatedOptionAtomicSwapPrepared, Any, jax.Array]:
    lower = context.lower
    arm_inputs, observation, live = _swap_transition(
        lower.retired_state.scheduler_state,
        lower.next_step,
    )
    cycle_key = jr.key(0xC1C1, impl="threefry2x32")
    prepared = context.sidecar.prepare_atomic_swap(
        context.source,
        cycle_key,
        lower.retirement_handoff,
        lower.retirement_authority,
        lower.phase_one_key,
        lower.phase_two_key,
        arm_inputs,
        observation,
        live,
    )
    authority = context.sidecar.authorize_atomic_swap(
        context.source,
        prepared,
        lower.installation_authority,
        cycle_key,
        swap_authorized=True,
    )
    return prepared, authority, cycle_key


def test_atomic_swap_rebinds_exact_oak_owner_without_persisting_cold_state(
    context: _Context,
) -> None:
    prepared, authority, cycle_key = _prepared_swap(context)
    result = context.sidecar.adopt_atomic_swap(
        context.source,
        prepared,
        authority,
        cycle_key,
        downstream_candidate_valid=jnp.asarray(True, dtype=jnp.bool_),
    )

    assert bool(result.atomic_swap_attempt.transaction_applied)
    assert not bool(result.atomic_swap_attempt.cold_state_persisted)
    assert bool(result.oak_rebind.transaction_applied)
    assert bool(result.exact_owner_rebind)
    assert bool(result.all_slots_installed_before)
    assert bool(result.all_slots_installed_after)
    assert bool(result.retirement_applied)
    assert bool(result.replacement_applied)
    assert bool(result.transaction_applied)
    assert int(result.state.lifecycle_metadata.completed_cycles) == 1
    assert bool(jnp.all(result.state.extended_action_mask))
    assert _count_exact_stomp_owners(result.state) == 1
    assert bool(context.sidecar.state_valid(result.state))
    rolled, attached = context.sidecar._attach_source(result.state)
    assert bool(attached)
    _tree_exact(
        _nested_stomp(result.state.coordinator_state),
        rolled.cycle_state.scheduler_state.installation_state.lifecycle_state.stomp_state,
    )

    replay = context.sidecar.adopt_atomic_swap(
        result.state,
        prepared,
        authority,
        cycle_key,
        downstream_candidate_valid=jnp.asarray(True, dtype=jnp.bool_),
    )
    assert not bool(replay.transaction_applied)
    assert not bool(replay.retirement_applied)
    assert not bool(replay.replacement_applied)
    _tree_exact(replay.state, result.state)


def test_atomic_swap_decline_tamper_and_outer_rejection_are_exact_noops(
    context: _Context,
) -> None:
    prepared, authority, cycle_key = _prepared_swap(context)
    outer_rejected = context.sidecar.adopt_atomic_swap(
        context.source,
        prepared,
        authority,
        cycle_key,
        downstream_candidate_valid=jnp.asarray(False, dtype=jnp.bool_),
    )
    assert bool(outer_rejected.atomic_swap_attempt.transaction_applied)
    assert not bool(outer_rejected.retirement_applied)
    assert not bool(outer_rejected.replacement_applied)
    assert not bool(outer_rejected.transaction_applied)
    _tree_exact(outer_rejected.state, context.source)

    declined_authority = context.sidecar.authorize_atomic_swap(
        context.source,
        prepared,
        context.lower.installation_authority,
        cycle_key,
        swap_authorized=False,
    )
    declined = context.sidecar.adopt_atomic_swap(
        context.source,
        prepared,
        declined_authority,
        cycle_key,
        downstream_candidate_valid=jnp.asarray(True, dtype=jnp.bool_),
    )
    assert not bool(declined.atomic_swap_attempt.transaction_applied)
    assert not bool(declined.transaction_applied)
    _tree_exact(declined.state, context.source)

    tampered = dataclasses.replace(
        prepared,
        prepared_checksum=prepared.prepared_checksum + jnp.uint32(1),
    )
    rejected = context.sidecar.adopt_atomic_swap(
        context.source,
        tampered,
        authority,
        cycle_key,
        downstream_candidate_valid=jnp.asarray(True, dtype=jnp.bool_),
    )
    assert not bool(rejected.prepared_integrity_valid)
    assert not bool(rejected.transaction_applied)
    _tree_exact(rejected.state, context.source)


def test_resource_contract_counts_only_borrowed_metadata(context: _Context) -> None:
    budget = context.sidecar.resource_budget(context.source)
    assert budget.persistent_state_nbytes == sum(
        int(jnp.asarray(leaf).nbytes)
        for leaf in jax.tree.leaves(context.source)
    )
    assert budget.coordinator_state_owners == 1
    assert budget.prototype_state_owners == 1
    assert budget.oak_state_owners == 1
    assert budget.stomp_state_owners == 1
    assert budget.detached_authority_metadata_stomp_state_owners == 0
    assert budget.repeated_overlay_stomp_state_owners == 0
    assert budget.borrowed_stomp_bindings == 1
    assert budget.additional_stomp_update_evaluations_per_adoption == 0
    assert budget.maximum_lifecycle_observations_per_adoption == 1
    assert budget.atomic_swap_prepare_host_only is True
    assert budget.atomic_swap_adopt_host_only is True
    assert budget.delight_available is False
    assert budget.additional_delight_evaluations == 0
    assert budget.additional_actor_backward_calls == 0
    assert budget.output_write_calls == 0
    assert budget.artifact_bytes_written == 0
