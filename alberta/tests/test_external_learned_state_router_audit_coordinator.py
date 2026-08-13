# mypy: disable-error-code="attr-defined,call-arg,no-any-return,operator"
"""Single-owner external learned-state/router/audit coordination."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import alberta_framework as alberta
import alberta_framework.core as core
from alberta_framework.core.delight import CandidateUpdateAuditConfig
from alberta_framework.core.experiential_memory import ExperientialMemoryConfig
from alberta_framework.core.external_builder_candidate_evidence_producer import (
    ExternalBuilderCandidateEvidenceProducer,
    ExternalBuilderRepresentationProbeEvidence,
)
from alberta_framework.core.external_learned_state_router_audit_coordinator import (
    EXTERNAL_LEARNED_STATE_ROUTER_AUDIT_COORDINATOR_CHECKPOINT_SCHEMA,
    EXTERNAL_LEARNED_STATE_ROUTER_AUDIT_COORDINATOR_EVIDENCE_LEVEL,
    EXTERNAL_LEARNED_STATE_ROUTER_AUDIT_COORDINATOR_OUTCOME_STATUS,
    ExternalBuilderCandidateAuditEvidence,
    ExternalLearnedStateRouterAuditCoordinator,
    ExternalLearnedStateRouterAuditCoordinatorConfig,
    ExternalLearnedStateRouterAuditCoordinatorState,
    ExternalLearnedStateTransition,
    load_external_learned_state_router_audit_coordinator_checkpoint,
    save_external_learned_state_router_audit_coordinator_checkpoint,
)
from alberta_framework.core.feature_bank_router import (
    FeatureBankRouterConfig,
    FeatureBankRouterState,
)
from alberta_framework.core.learning_signals import LearningSignalEstimatorConfig
from alberta_framework.core.learning_value_router import LearningValueRouterConfig
from alberta_framework.core.oak import OaKConfig
from alberta_framework.core.options import STOMPConfig, SubtaskSpec
from alberta_framework.core.prototype_agent import (
    PrototypeAgentConfig,
    PrototypeExperientialMemoryInput,
    PrototypeFeatureOaKHordeState,
    PrototypeMemoryInteractionState,
)
from alberta_framework.core.prototype_feature_lifecycle import (
    PrototypeFeatureConsumerBinding,
    PrototypeFeatureLifecycleConfig,
)
from alberta_framework.core.prototype_feature_memory import PrototypeFeatureMemoryState
from alberta_framework.core.prototype_routed_linear_world_model_ensemble_adapter import (
    PrototypeRoutedLinearWorldModelEnsembleAdapterConfig,
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

RAW_DIM = 2
HIDDEN_DIM = 1
BASE_DIM = RAW_DIM + HIDDEN_DIM
PAIR_SLOTS = 2
TOTAL_DIM = BASE_DIM + PAIR_SLOTS
N_ACTIONS = 2
N_DEMONS = 1
ENSEMBLE_SIZE = 2
TARGET_DIM = BASE_DIM + 2


@pytest.fixture(autouse=True)
def _bounded_jax_execution(request: pytest.FixtureRequest) -> object:
    if request.node.name == "test_direct_step_jit_and_monolithic_scan_rejection":
        yield
    else:
        with jax.disable_jit():
            yield


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


def _coordinator(
    *,
    memory: bool = False,
    replacement_interval: int = 0,
    max_events: int = 100,
    candidate_audit: CandidateUpdateAuditConfig | None = None,
) -> ExternalLearnedStateRouterAuditCoordinator:
    feature = PrototypeFeatureLifecycleConfig(
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
    oak = OaKConfig(
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
        oak=oak,
        horde_spec=horde,
        horde_hidden_sizes=(),
        horde_step_size=0.1,
        experiential_memory=_memory_config() if memory else None,
        state_builder=IdentityStateBuilderConfig(observation_dim=BASE_DIM),
        prototype_feature_lifecycle=feature,
    )
    ensemble = RoutedLinearWorldModelEnsembleConfig(
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
    return ExternalLearnedStateRouterAuditCoordinator(
        ExternalLearnedStateRouterAuditCoordinatorConfig(
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
            learning_value_router=LearningValueRouterConfig(max_steps=100),
            candidate_audit=(
                CandidateUpdateAuditConfig(candidate_semantics="update")
                if candidate_audit is None
                else candidate_audit
            ),
            max_events=max_events,
        )
    )


def _transition(
    state: ExternalLearnedStateRouterAuditCoordinatorState,
    *,
    next_observation: tuple[float, float] = (0.1, 0.4),
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
        horde_cumulants=jnp.asarray((0.25,), dtype=jnp.float32),
    )


def _audit_evidence(
    coordinator: ExternalLearnedStateRouterAuditCoordinator,
    prepared: object,
) -> ExternalBuilderCandidateAuditEvidence:
    source = prepared.source_state
    true = jnp.asarray(True, dtype=jnp.bool_)
    probes = ExternalBuilderRepresentationProbeEvidence(
        source_event_words=source.event_words,
        source_builder_step_words=source.cached_builder_step_words,
        source_prototype_step_words=source.cached_prototype_step_words,
        source_feature_generation_words=source.cached_feature_generation_words,
        decision_id=source.current_decision_id,
        objective_representation_gradient=(
            prepared.causal_target.representation_gradient
        ),
        retention_representation_gradient=(
            prepared.causal_target.representation_gradient
        ),
        safety_representation_gradient=(
            prepared.causal_target.representation_gradient
        ),
        objective_probe_available=true,
        retention_probe_available=true,
        safety_probe_available=true,
        probe_independence_attested=true,
        advantage=jnp.asarray(0.5, dtype=jnp.float32),
        action_surprisal=jnp.asarray(0.5, dtype=jnp.float32),
        safety_cost=jnp.asarray(0.0, dtype=jnp.float32),
        advantage_available=true,
        action_surprisal_available=true,
        safety_cost_available=true,
    )
    produced = ExternalBuilderCandidateEvidenceProducer(
        coordinator.config.builder
    ).produce(source, probes)
    assert bool(produced.diagnostics.evidence_ready)
    assert not bool(produced.diagnostics.delight_or_actor_backward)
    return produced.evidence


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


def _binding(
    state: ExternalLearnedStateRouterAuditCoordinatorState,
) -> PrototypeFeatureConsumerBinding:
    bundle = state.inner_state.prototype_state.oak_state
    assert type(bundle) is PrototypeFeatureOaKHordeState
    return bundle.consumer_binding


def _memory_input(
    state: ExternalLearnedStateRouterAuditCoordinatorState,
) -> PrototypeExperientialMemoryInput:
    prototype = state.inner_state.prototype_state
    binding = _binding(state)
    next_decision_id = prototype.current_decision_id.at[3].add(
        jnp.asarray(1, dtype=jnp.uint32)
    )
    return PrototypeExperientialMemoryInput(
        available=jnp.asarray(True, dtype=jnp.bool_),
        current_prototype_decision_id=prototype.current_decision_id,
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


def test_feasibility_contract_has_a_separate_coordinator_type() -> None:
    """The feasibility contract starts red until the external seam exists."""

    assert ExternalLearnedStateRouterAuditCoordinator.__module__.endswith(
        "external_learned_state_router_audit_coordinator"
    )


def test_one_real_event_advances_every_owner_once_with_learning_vetoed() -> None:
    coordinator = _coordinator()
    state = coordinator.init(jr.key(3))
    state = coordinator.start(
        state,
        jnp.asarray((0.2, -0.1), dtype=jnp.float32),
    )
    assert bool(coordinator.state_valid(state))

    prepared = coordinator.prepare_transition(
        state,
        _transition(state),
    )
    assert bool(prepared.preparation_valid)
    evaluated = coordinator.evaluate_candidate(prepared)
    result = coordinator.adopt_evaluated_transition(
        state,
        evaluated,
        coordinator.integrity_receipt(evaluated),
    )

    assert bool(result.diagnostics.transaction_applied)
    assert not bool(result.diagnostics.candidate_audit_accepted)
    assert bool(result.diagnostics.builder_learning_vetoed)
    assert not bool(result.diagnostics.builder_learning_applied)
    assert int(result.state.event_words[1]) == 1
    assert int(result.state.builder_state.step_words[1]) == 2
    assert int(result.state.inner_state.prototype_state.step_words[1]) == 1
    assert int(result.state.learning_value_router_state.step_count) == 1
    assert int(result.diagnostics.external_builder_transition_evaluations) == 1
    assert int(result.diagnostics.inner_prototype_update_evaluations) == 1
    assert int(result.diagnostics.learning_value_router_evaluations) == 1
    assert int(result.diagnostics.candidate_audit_evaluations) == 1
    assert int(result.diagnostics.ensemble_total_member_forward_evaluations) == 6
    assert int(result.diagnostics.additional_model_forward_evaluations) == 0
    assert bool(coordinator.state_valid(result.state))


def test_config_resources_exports_and_checkpoint_continuation(tmp_path: Path) -> None:
    coordinator = _coordinator(memory=True)
    state = coordinator.init(jr.key(7))
    assert core.ExternalLearnedStateRouterAuditCoordinator is (
        ExternalLearnedStateRouterAuditCoordinator
    )
    assert alberta.EXTERNAL_LEARNED_STATE_ROUTER_AUDIT_COORDINATOR_CHECKPOINT_SCHEMA == (
        EXTERNAL_LEARNED_STATE_ROUTER_AUDIT_COORDINATOR_CHECKPOINT_SCHEMA
    )
    payload = coordinator.to_config()
    assert ExternalLearnedStateRouterAuditCoordinator.from_config(
        payload
    ).to_config() == payload
    assert payload["evidence_level"] == (
        EXTERNAL_LEARNED_STATE_ROUTER_AUDIT_COORDINATOR_EVIDENCE_LEVEL
    )
    assert payload["outcome_status"] == (
        EXTERNAL_LEARNED_STATE_ROUTER_AUDIT_COORDINATOR_OUTCOME_STATUS
    )
    assert payload["caller_target_authority"] is False
    assert payload["prototype_v18_allowed"] is False
    assert payload["terminal_boundary_supported"] is False
    assert payload["direct_step_jit_supported"] is True
    assert payload["monolithic_scan_jit_supported"] is False
    assert payload["scan_execution"] == "host-loop-only"
    assert payload["learning_value_router_count"] == 1
    assert payload["planning_authority"] is False
    assert payload["dispatch_authority"] is False
    assert payload["safety_authority"] is False
    assert payload["evidence_authority"] is False
    assert _count_router_states(state) == 1

    budget = coordinator.resource_budget
    assert budget.persistent_state_bytes > 0
    assert budget.persistent_capacity_growth == 0
    assert budget.external_builder_owner_count == 1
    assert budget.inner_identity_builder_count == 1
    assert budget.feature_lifecycle_authority_count == 1
    assert budget.feature_router_authority_count == 1
    assert budget.learning_value_router_count == 1
    assert budget.managed_linear_horde_count == 1
    assert budget.feature_bound_memory_count == 1
    assert budget.routed_ensemble_count == 1
    assert budget.ensemble_total_member_forward_evaluations_per_event == 6
    assert budget.additional_model_forward_evaluations_for_pullback == 0
    assert budget.feature_bank_mapping_evaluations_per_event == 3
    assert budget.curation_recomputations_per_event == 0
    assert budget.caller_target_authority == 0
    assert budget.terminal_boundary_supported == 0
    assert budget.direct_step_jit_supported is True
    assert budget.monolithic_scan_jit_supported is False
    assert budget.scan_execution_host_loop_only is True
    assert budget.planning_authority == budget.dispatch_authority == 0
    assert budget.safety_authority == budget.evidence_authority == 0

    invalid_builder = dataclasses.replace(
        coordinator.config.builder,
        include_raw_observation=False,
    )
    with pytest.raises(ValueError, match="include_raw_observation"):
        ExternalLearnedStateRouterAuditCoordinatorConfig(
            builder=invalid_builder,
            inner=coordinator.config.inner,
            learning_value_router=coordinator.config.learning_value_router,
            candidate_audit=coordinator.config.candidate_audit,
            max_events=coordinator.config.max_events,
        )

    checkpoint = tmp_path / "external-learned-state-router-audit"
    save_external_learned_state_router_audit_coordinator_checkpoint(
        coordinator,
        state,
        checkpoint,
    )
    restored_owner, restored_state = (
        load_external_learned_state_router_audit_coordinator_checkpoint(checkpoint)
    )
    assert restored_owner.to_config() == payload
    _tree_exact(restored_state, state)

    observation = jnp.asarray((0.2, -0.1), dtype=jnp.float32)
    direct_source = coordinator.start(state, observation)
    restored_source = restored_owner.start(restored_state, observation)
    direct = coordinator.step(
        direct_source,
        _transition(direct_source),
        experiential_memory_input=_memory_input(direct_source),
    )
    resumed = restored_owner.step(
        restored_source,
        _transition(restored_source),
        experiential_memory_input=_memory_input(restored_source),
    )
    _tree_exact(resumed, direct)


def test_internal_causal_pullback_matches_cached_source_model_autodiff() -> None:
    coordinator = _coordinator()
    state = coordinator.start(
        coordinator.init(jr.key(11)),
        jnp.asarray((0.2, -0.1), dtype=jnp.float32),
    )
    prepared = coordinator.prepare_transition(state, _transition(state))
    lifecycle, _ = coordinator.inner._bank(state.inner_state.prototype_state)
    targets = prepared.causal_target.targets

    def source_objective(base: jax.Array) -> jax.Array:
        prediction = coordinator.inner.ensemble.predict(
            state.inner_state.ensemble_state,
            lifecycle.router_state,
            base,
            state.current_action,
        )
        return 0.5 * jnp.mean(
            jnp.square(prediction.member_raw_predictions - targets[None, :])
        )

    expected = jax.grad(source_objective)(state.current_representation)
    np.testing.assert_allclose(
        prepared.causal_target.representation_gradient,
        expected,
        rtol=2.0e-6,
        atol=2.0e-6,
    )
    np.testing.assert_allclose(
        prepared.causal_target.representation_objective,
        source_objective(state.current_representation),
        rtol=2.0e-6,
        atol=2.0e-6,
    )
    assert bool(prepared.causal_target.gradient_valid)
    assert not bool(prepared.causal_target.caller_target_supplied)
    assert int(prepared.causal_target.additional_model_forward_evaluations) == 0
    assert int(prepared.analytic_pullback_evaluations) == 1


def test_accepted_audit_updates_only_external_builder_for_the_next_event() -> None:
    coordinator = _coordinator(
        candidate_audit=CandidateUpdateAuditConfig(
            candidate_semantics="update",
            max_update_norm=100.0,
            max_retention_loss_increase=100.0,
            max_safety_cost_increase=100.0,
            min_objective_descent_alignment=-1.0,
            min_retention_descent_alignment=-1.0,
            min_safety_descent_alignment=-1.0,
        )
    )
    state = coordinator.start(
        coordinator.init(jr.key(13)),
        jnp.asarray((0.2, -0.1), dtype=jnp.float32),
    )
    for next_observation in ((0.1, 0.4), (-0.2, 0.3)):
        result = coordinator.step(
            state,
            _transition(state, next_observation=next_observation),
        )
        assert bool(result.diagnostics.transaction_applied)
        state = result.state

    prepared = coordinator.prepare_transition(
        state,
        _transition(state, next_observation=(0.35, -0.15)),
    )
    accepted_evaluation = coordinator.evaluate_candidate(
        prepared,
        _audit_evidence(coordinator, prepared),
    )
    vetoed_evaluation = coordinator.evaluate_candidate(prepared)
    assert bool(accepted_evaluation.candidate_audit_assessment.accepted)
    assert not bool(vetoed_evaluation.candidate_audit_assessment.accepted)

    accepted = coordinator.adopt_evaluated_transition(
        state,
        accepted_evaluation,
        coordinator.integrity_receipt(accepted_evaluation),
    )
    vetoed = coordinator.adopt_evaluated_transition(
        state,
        vetoed_evaluation,
        coordinator.integrity_receipt(vetoed_evaluation),
    )
    assert bool(accepted.diagnostics.builder_learning_applied)
    assert bool(vetoed.diagnostics.builder_learning_vetoed)
    assert not np.array_equal(
        np.asarray(accepted.state.builder_state.parameters),
        np.asarray(vetoed.state.builder_state.parameters),
    )
    np.testing.assert_array_equal(
        accepted.state.current_representation,
        vetoed.state.current_representation,
    )
    _tree_exact(accepted.state.inner_state, vetoed.state.inner_state)
    assert int(accepted.state.builder_state.update_words[1]) == 1
    assert int(vetoed.state.builder_state.update_words[1]) == 0
    assert bool(coordinator.state_valid(accepted.state))


def test_receipt_terminal_capacity_and_clock_failures_rollback_atomically() -> None:
    coordinator = _coordinator(max_events=1)
    state = coordinator.start(
        coordinator.init(jr.key(17)),
        jnp.asarray((0.2, -0.1), dtype=jnp.float32),
    )
    evaluated = coordinator.evaluate_candidate(
        coordinator.prepare_transition(state, _transition(state))
    )
    receipt = coordinator.integrity_receipt(evaluated)
    refused = coordinator.adopt_evaluated_transition(
        state,
        evaluated,
        receipt.replace(integrity_bound=jnp.asarray(False, dtype=jnp.bool_)),
    )
    assert bool(refused.diagnostics.rejected)
    _tree_exact(refused.state, state)

    accepted = coordinator.adopt_evaluated_transition(state, evaluated, receipt)
    assert bool(accepted.diagnostics.transaction_applied)
    stale = coordinator.adopt_evaluated_transition(accepted.state, evaluated, receipt)
    assert bool(stale.diagnostics.rejected)
    _tree_exact(stale.state, accepted.state)

    capacity = coordinator.step(
        accepted.state,
        _transition(accepted.state, next_observation=(-0.2, 0.3)),
    )
    assert bool(capacity.diagnostics.rejected)
    _tree_exact(capacity.state, accepted.state)

    terminal_transition = _transition(state).replace(
        terminated=jnp.asarray(True, dtype=jnp.bool_)
    )
    terminal = coordinator.step(state, terminal_transition)
    assert not bool(terminal.diagnostics.continuing_boundary_valid)
    assert bool(terminal.diagnostics.rejected)
    _tree_exact(terminal.state, state)

    divergent_count = accepted.state.replace(
        event_count=jnp.asarray(0, dtype=jnp.int32)
    )
    divergent_words = accepted.state.replace(
        event_words=jnp.asarray((0, 0), dtype=jnp.uint32)
    )
    assert not bool(coordinator.state_valid(divergent_count))
    assert not bool(coordinator.state_valid(divergent_words))


def test_managed_horde_feature_memory_and_shared_binding_coexist() -> None:
    coordinator = _coordinator(memory=True, replacement_interval=1)
    state = coordinator.start(
        coordinator.init(jr.key(19)),
        jnp.asarray((0.2, -0.1), dtype=jnp.float32),
    )
    prototype = state.inner_state.prototype_state
    source_bundle = prototype.oak_state
    assert type(source_bundle) is PrototypeFeatureOaKHordeState
    action_mask = jnp.ones((N_ACTIONS + 1,), dtype=jnp.bool_)
    result = coordinator.step(
        state,
        _transition(state),
        experiential_memory_input=_memory_input(state),
        extended_action_mask=action_mask,
    )
    assert bool(result.diagnostics.transaction_applied)
    destination = result.state.inner_state.prototype_state
    bundle = destination.oak_state
    assert type(bundle) is PrototypeFeatureOaKHordeState
    interaction = destination.ia_state
    assert type(interaction) is PrototypeMemoryInteractionState
    memory = interaction.experiential_memory_state
    assert type(memory) is PrototypeFeatureMemoryState
    binding = _binding(result.state)
    _tree_exact(bundle.consumer_binding, binding)
    _tree_exact(memory.consumer_binding, binding)
    _tree_exact(result.state.inner_state.ensemble_state.consumer_binding, binding)
    assert int(bundle.horde_state.step_words[1]) == int(
        source_bundle.horde_state.step_words[1]
    ) + 1
    assert int(jnp.sum(memory.memory_state.entries.valid)) == 1
    assert int(result.diagnostics.feature_bank_mapping_evaluations) == 3
    assert int(result.diagnostics.curation_recomputations) == 0


def test_host_one_step_scan_matches_eager_and_rejects_empty_batches() -> None:
    coordinator = _coordinator()
    state = coordinator.start(
        coordinator.init(jr.key(23)),
        jnp.asarray((0.2, -0.1), dtype=jnp.float32),
    )
    transition = _transition(state)
    eager = coordinator.step(state, transition)
    batched_transition = jax.tree.map(lambda value: value[None, ...], transition)
    scanned = coordinator.scan_transitions(state, batched_transition)
    _tree_exact(scanned.state, eager.state)
    np.testing.assert_array_equal(
        scanned.transaction_applied,
        jnp.asarray((True,), dtype=jnp.bool_),
    )
    np.testing.assert_array_equal(
        scanned.builder_learning_applied,
        jnp.asarray((False,), dtype=jnp.bool_),
    )
    empty = jax.tree.map(lambda value: value[:0], batched_transition)
    with pytest.raises(ValueError, match="length must be positive"):
        coordinator.scan_transitions(state, empty)


def test_direct_step_jit_and_monolithic_scan_rejection() -> None:
    coordinator = _coordinator()
    state = coordinator.start(
        coordinator.init(jr.key(29)),
        jnp.asarray((0.2, -0.1), dtype=jnp.float32),
    )
    transition = _transition(state)
    with jax.disable_jit():
        eager = coordinator.step(state, transition)
    compiled = jax.jit(coordinator.step)(state, transition)
    # The transient Prototype receipt hashes exact float bits, so ordinary
    # eager/XLA rounding may legitimately produce different content tags.
    # Each path validates its own receipt; destination state and public
    # diagnostics are the honest cross-execution equivalence boundary.
    _tree_close(compiled.state, eager.state)
    _tree_close(compiled.diagnostics, eager.diagnostics)
    assert bool(compiled.diagnostics.receipt_matches_evaluation)
    assert bool(eager.diagnostics.receipt_matches_evaluation)

    batched_transition = jax.tree.map(lambda value: value[None, ...], transition)
    with pytest.raises(RuntimeError, match="host-only"):
        jax.jit(coordinator.scan_transitions)(state, batched_transition)
