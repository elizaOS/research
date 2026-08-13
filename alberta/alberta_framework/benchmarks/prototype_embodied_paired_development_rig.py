# mypy: disable-error-code="arg-type,attr-defined,call-arg,no-any-return,type-var"
"""Deterministic production-only rig for the paired embodied L0 CLI.

The rig is deliberately tiny, synthetic, bounded, and nonpromoting.  It exists
only to make the development benchmark executable from an installed package;
it does not provide physical dispatch, safety, evidence, or deployment authority.
"""

from __future__ import annotations

import dataclasses
from typing import Any, cast

import jax
import jax.numpy as jnp
import jax.random as jr

from alberta_framework.core.consolidated_memory import (
    ConsolidatedMemoryConfig,
    ProceduralMemoryRequest,
    canonical_memory_digest,
)
from alberta_framework.core.consolidated_memory_controller import (
    ConsolidatedProceduralMemoryControllerConfig,
)
from alberta_framework.core.consolidated_memory_policy import (
    ConsolidatedProceduralMemoryPolicyConfig,
)
from alberta_framework.core.embodied_safety_envelope import (
    EmbodiedSafetyEnvelope,
    EmbodiedSafetyEnvelopeConfig,
    EmbodiedTelemetry,
)
from alberta_framework.core.ensemble_short_rollouts import (
    EnsembleShortRolloutConfig,
    EnsembleShortRolloutPlanner,
    ImaginedRolloutBatch,
)
from alberta_framework.core.grounded_imagination_composition import (
    GroundedImaginationComposition,
    GroundedImaginationCompositionState,
)
from alberta_framework.core.imagined_rollout_selection_gauge import (
    AuthorizedImaginedRolloutActorCritic,
    ImaginedRolloutActorCriticConfig,
    ImaginedRolloutSelectionGauge,
    ImaginedRolloutSelectionGaugeConfig,
    ImaginedRolloutSelectionGaugeState,
)
from alberta_framework.core.learning_signals import LearningSignalEstimatorConfig
from alberta_framework.core.oak import OaKConfig, OaKState
from alberta_framework.core.options import STOMPConfig, SubtaskSpec
from alberta_framework.core.prototype_agent import PrototypeAgentConfig
from alberta_framework.core.prototype_consolidated_memory import (
    PrototypeConsolidatedMemoryConfig,
    PrototypeConsolidatedMemoryDecisionInput,
    PrototypeConsolidatedMemoryState,
)
from alberta_framework.core.prototype_consolidated_semantic_memory import (
    PrototypeConsolidatedSemanticMemoryAgent,
    PrototypeConsolidatedSemanticMemoryConfig,
)
from alberta_framework.core.prototype_embodied_command_adapter import (
    DiscreteEmbodiedPrimitiveCommand,
    PrototypeEmbodiedCommandAdapter,
    PrototypeEmbodiedCommandAdapterConfig,
    PrototypeEmbodiedCommandPreparationInput,
)
from alberta_framework.core.prototype_embodied_development_harness import (
    DeterministicPrimitivePlant,
    DeterministicPrimitivePlantConfig,
    PrototypeEmbodiedDevelopmentHarness,
    PrototypeEmbodiedDevelopmentHarnessPreparationInput,
)
from alberta_framework.core.world_model import ActionConditionedWorldModelConfig
from alberta_framework.core.world_model_ensemble import (
    WorldModelEnsemble,
    WorldModelEnsembleConfig,
    WorldModelEnsembleState,
)

MODEL_VERSION = jnp.full((8,), 0x11, dtype=jnp.uint32)
OPTIMIZER_VERSION = jnp.full((8,), 0x22, dtype=jnp.uint32)
LIFECYCLE_VERSION = jnp.full((8,), 0x33, dtype=jnp.uint32)
PARTNER_DIGEST = jnp.full((8,), 0x44, dtype=jnp.uint32)
SOURCE_DIGEST = jnp.arange(11, 19, dtype=jnp.uint32)
REVISION_ONE = jnp.asarray((0, 1), dtype=jnp.uint32)
ACTION_SUPPORT = jnp.asarray((20, 20), dtype=jnp.int32)


def _words(value: int) -> jax.Array:
    return jnp.asarray((0, value), dtype=jnp.uint32)


def _digest(text: str) -> jax.Array:
    return canonical_memory_digest(
        "alberta.prototype-embodied-paired-development",
        text,
    )


def _procedural_request() -> ProceduralMemoryRequest:
    return ProceduralMemoryRequest(
        semantic_digest=_digest("skill"),
        generation=jnp.asarray(0, dtype=jnp.int32),
        provenance_digest=_digest("provenance"),
        representation_revision=jnp.asarray(0, dtype=jnp.int32),
        source_revision=jnp.asarray(0, dtype=jnp.int32),
        lifecycle_link_available=jnp.asarray(True, dtype=jnp.bool_),
        lifecycle_digest=_digest("option-lifecycle"),
        lifecycle_generation=jnp.asarray(3, dtype=jnp.int32),
        lifecycle_revision=jnp.asarray(5, dtype=jnp.int32),
    )


def _decision_input(
    state: PrototypeConsolidatedMemoryState,
) -> PrototypeConsolidatedMemoryDecisionInput:
    n_actions = state.controller.pending_hard_safety_mask.shape[0]
    return PrototypeConsolidatedMemoryDecisionInput(
        available=jnp.asarray(True, dtype=jnp.bool_),
        prototype_decision_id=state.prototype.current_decision_id,
        request=_procedural_request(),
        hard_safety_action_mask=jnp.ones((n_actions,), dtype=jnp.bool_),
    )


def _procedural_config() -> PrototypeConsolidatedMemoryConfig:
    n_actions = 2
    oak = OaKConfig(
        stomp=STOMPConfig(
            subtask_specs=(
                SubtaskSpec(
                    feature_index=0,
                    threshold=1.0e6,
                    max_option_steps=8,
                ),
            ),
            observation_dim=2,
            n_primitive_actions=n_actions,
            base_hidden_sizes=(),
            epsilon_base=0.0,
            epsilon_option=0.0,
        )
    )
    controller = ConsolidatedProceduralMemoryControllerConfig(
        memory=ConsolidatedMemoryConfig(
            semantic_capacity=1,
            procedural_capacity=2,
            semantic_payload_dim=1,
            procedural_payload_dim=n_actions,
            procedural_outcome_dim=1,
            semantic_max_age=20,
            procedural_max_age=20,
            max_operations=100,
            semantic_min_confidence=0.0,
            procedural_min_confidence=0.0,
        ),
        policy=ConsolidatedProceduralMemoryPolicyConfig(
            n_actions=n_actions,
            outcome_dim=1,
            min_evidence_count=2,
            min_success_lower_bound=0.0,
            wilson_z=1.0,
            max_outcome_standard_error=10.0,
            max_abs_outcome_mean=100.0,
        ),
    )
    return PrototypeConsolidatedMemoryConfig(
        prototype=PrototypeAgentConfig(oak=oak),
        controller=controller,
    )


def _canonicalize_semantic_host_metadata(state: Any) -> Any:
    prototype = state.composition.prototype
    oak = cast(OaKState, prototype.oak_state)
    learner = oak.stomp_state.base_learner_state.replace(
        birth_timestamp=jnp.asarray(0.0, dtype=jnp.float32),
        uptime_s=jnp.asarray(0.0, dtype=jnp.float32),
    )
    stomp = oak.stomp_state.replace(base_learner_state=learner)
    prototype = prototype.replace(oak_state=oak.replace(stomp_state=stomp))
    return state.replace(composition=state.composition.replace(prototype=prototype))


def _paired_semantics() -> tuple[Any, Any, Any, Any]:
    composition = _procedural_config()
    stomp = dataclasses.replace(
        composition.prototype.oak.stomp,
        observation_dim=3,
    )
    oak = dataclasses.replace(composition.prototype.oak, stomp=stomp)
    composition = dataclasses.replace(
        composition,
        prototype=dataclasses.replace(composition.prototype, oak=oak),
    )
    adaptive = PrototypeConsolidatedSemanticMemoryAgent(
        PrototypeConsolidatedSemanticMemoryConfig(
            composition=composition,
            raw_observation_dim=2,
        )
    )
    control_stomp = dataclasses.replace(
        stomp,
        base_step_size=0.0,
        base_avg_reward_step_size=0.0,
        option_step_size=0.0,
        option_avg_reward_step_size=0.0,
        option_model_step_size=0.0,
    )
    control_composition = dataclasses.replace(
        composition,
        prototype=dataclasses.replace(
            composition.prototype,
            oak=dataclasses.replace(composition.prototype.oak, stomp=control_stomp),
        ),
    )
    control = PrototypeConsolidatedSemanticMemoryAgent(
        PrototypeConsolidatedSemanticMemoryConfig(
            composition=control_composition,
            raw_observation_dim=2,
        )
    )

    def started(agent: PrototypeConsolidatedSemanticMemoryAgent) -> Any:
        initial = agent.init(
            jr.key(7),
            source_digest=_digest("source"),
            semantic_namespace_digest=_digest("namespace"),
            representation_revision=0,
            source_revision=0,
            lifecycle_id=jnp.asarray((17, 19), dtype=jnp.uint32),
        )
        with jax.disable_jit():
            result = agent.start(
                initial,
                jnp.zeros((2,), dtype=jnp.float32),
                decision_input=_decision_input(initial.composition),
            ).state
        return _canonicalize_semantic_host_metadata(result)

    return adaptive, started(adaptive), control, started(control)


def _world_model() -> tuple[WorldModelEnsemble, WorldModelEnsembleState]:
    ensemble = WorldModelEnsemble(
        WorldModelEnsembleConfig(
            model=ActionConditionedWorldModelConfig(
                observation_dim=2,
                n_actions=2,
                gamma=0.95,
                hidden_sizes=(),
                step_size=0.05,
                sparsity=0.0,
                use_layer_norm=False,
                error_decay=0.8,
            ),
            signal_estimator=LearningSignalEstimatorConfig(
                ensemble_size=2,
                target_dim=4,
                progress_warmup_steps=2,
                change_calibration_steps=2,
                max_input_magnitude=1_000.0,
                max_predicted_variance=10_000.0,
                max_observed_loss=10_000.0,
            ),
            ensemble_size=2,
            bootstrap_probability=0.5,
            residual_variance_warmup_steps=1,
            residual_variance_floor=1.0e-6,
        )
    )
    state = ensemble.init(jr.key(1, impl="threefry2x32"))
    members = []
    for member in state.member_states:
        learner = member.learner_state
        weights = []
        biases = []
        for head_index, value in enumerate((0.1, -0.1, 1.0, 0.5)):
            weight = jnp.zeros_like(learner.head_params.weights[head_index])
            weight = weight.at[0, 2].set(value)
            weight = weight.at[0, 3].set(value)
            weights.append(weight)
            biases.append(jnp.zeros((1,), dtype=jnp.float32))
        heads = learner.head_params.replace(weights=tuple(weights), biases=tuple(biases))
        members.append(member.replace(learner_state=learner.replace(head_params=heads)))
    state = cast(WorldModelEnsembleState, state.replace(member_states=tuple(members)))
    if not bool(ensemble.state_valid(state)):
        raise ValueError("paired development world-model rig is invalid")
    return ensemble, state


def _calibration_batch(
    planner: EnsembleShortRolloutPlanner,
    model_state: WorldModelEnsembleState,
    *,
    action: int,
    decision: int,
) -> ImaginedRolloutBatch:
    revision = jnp.asarray((0, decision), dtype=jnp.uint32)
    bias = (20.0, -20.0) if action == 0 else (-20.0, 20.0)
    authority = planner.bind_authority(
        policy_weights=jnp.zeros((2, 2), dtype=jnp.float32),
        policy_bias=jnp.asarray(bias, dtype=jnp.float32),
        value_weights=jnp.zeros((2,), dtype=jnp.float32),
        value_bias=jnp.asarray(0.0, dtype=jnp.float32),
        action_support_counts=ACTION_SUPPORT,
        source_revision_words=REVISION_ONE,
        model_state=model_state,
        policy_revision_words=revision,
        value_revision_words=revision,
    )
    planner_state = planner.init(
        jr.key(20 + decision, impl="threefry2x32"),
        model_state,
        authority,
    )
    anchor = planner.bind_real_anchor(
        jnp.asarray((float(decision), 0.0), dtype=jnp.float32),
        _words(decision),
        authority,
    )
    result = planner.propose(planner_state, model_state, authority, anchor)
    if not bool(result.diagnostics.transaction_applied):
        raise ValueError("paired development calibration rollout failed")
    if not bool(jnp.all(result.proposals.actions[result.proposals.transition_valid] == action)):
        raise ValueError("paired development calibration selected the wrong action")
    return result.proposals


def _grounded_system() -> tuple[
    GroundedImaginationComposition,
    GroundedImaginationCompositionState,
    WorldModelEnsembleState,
]:
    ensemble, model_state = _world_model()
    planner = EnsembleShortRolloutPlanner(
        ensemble,
        EnsembleShortRolloutConfig(
            rollout_horizon=2,
            rollout_budget=1,
            require_residual_proxy_ready=False,
            max_epistemic_disagreement=100.0,
            max_residual_variance=100.0,
            max_proposal_calls=16,
            max_rollout_attempts=16,
            max_imagined_steps=32,
        ),
    )
    action_zero = _calibration_batch(planner, model_state, action=0, decision=1)
    action_one = _calibration_batch(planner, model_state, action=1, decision=2)
    gauge = ImaginedRolloutSelectionGauge(
        planner,
        ImaginedRolloutSelectionGaugeConfig(
            audit_capacity=4,
            n_regions=1,
            min_evidence_count=1,
            min_realized_valid_fraction=1.0,
            max_mean_abs_reward_error=0.0,
            max_root_mean_square_next_observation_error=0.0,
            min_termination_accuracy=1.0,
            require_success_lcb=False,
            require_top_quantile_purity=False,
            max_authorizations=16,
        ),
    )
    gauge_state: ImaginedRolloutSelectionGaugeState = gauge.init(action_zero)
    for record_id, batch in enumerate((action_zero, action_one), start=1):
        record = gauge.bind_grounded_record(
            batch,
            rollout_index=jnp.asarray(0, dtype=jnp.int32),
            step_index=jnp.asarray(0, dtype=jnp.int32),
            region_id=jnp.asarray(0, dtype=jnp.int32),
            record_id_words=_words(record_id),
            realized_valid=jnp.asarray(True),
            realized_reward=batch.rewards[0, 0],
            realized_next_observation=batch.next_observations[0, 0],
            realized_terminated=batch.terminated[0, 0],
            realized_success=jnp.asarray(True),
        )
        result = gauge.record_grounded_outcome(gauge_state, record)
        if not bool(result.diagnostics.applied):
            raise ValueError("paired development gauge calibration failed")
        gauge_state = result.state
    actor_critic = AuthorizedImaginedRolloutActorCritic(
        gauge,
        ImaginedRolloutActorCriticConfig(
            initialization_scale=0.0,
            max_update_calls=16,
            max_backward_transitions=32,
        ),
    )
    composition = GroundedImaginationComposition(planner, gauge, actor_critic)
    state = composition.init(
        planner_key=jr.key(30, impl="threefry2x32"),
        actor_critic_key=jr.key(31, impl="threefry2x32"),
        model_state=model_state,
        action_support_counts=ACTION_SUPPORT,
        source_revision_words=REVISION_ONE,
        grounded_gauge_state=gauge_state,
    )
    return composition, state, model_state


def _safe_commands() -> tuple[
    DiscreteEmbodiedPrimitiveCommand,
    DiscreteEmbodiedPrimitiveCommand,
]:
    return (
        DiscreteEmbodiedPrimitiveCommand(
            joint_position=(0.2, 0.3),
            joint_velocity=(0.1, 0.2),
            joint_torque=(0.3, 0.4),
            workspace_position=(0.2, 0.2, 1.0),
            collision_clearance=0.4,
        ),
        DiscreteEmbodiedPrimitiveCommand(
            joint_position=(-0.2, -0.3),
            joint_velocity=(-0.1, -0.2),
            joint_torque=(-0.3, -0.4),
            workspace_position=(-0.2, -0.2, 1.0),
            collision_clearance=0.6,
        ),
    )


def _telemetry() -> EmbodiedTelemetry:
    return EmbodiedTelemetry(
        joint_position=jnp.zeros((2,), dtype=jnp.float32),
        joint_velocity=jnp.zeros((2,), dtype=jnp.float32),
        joint_torque=jnp.zeros((2,), dtype=jnp.float32),
        workspace_position=jnp.asarray((0.0, 0.0, 1.0), dtype=jnp.float32),
        collision_clearance=jnp.asarray(0.5, dtype=jnp.float32),
        bridge_connected=jnp.asarray(True, dtype=jnp.bool_),
        emergency_stop=jnp.asarray(False, dtype=jnp.bool_),
        telemetry_id=_words(1),
        sample_tick=_words(10),
    )


def build_prototype_embodied_paired_development_benchmark(
    *,
    development_key: int = 17,
) -> Any:
    """Build the fixed, synthetic v1 paired benchmark without test dependencies."""

    from alberta_framework.benchmarks.prototype_embodied_paired_development import (
        PrototypeEmbodiedPairedDevelopmentBenchmark,
        PrototypeEmbodiedPairedDevelopmentConfig,
    )

    grounded, grounded_state, model_state = _grounded_system()
    adaptive_semantic, adaptive_state, control_semantic, control_state = _paired_semantics()
    adaptive_selected = int(adaptive_state.composition.prototype.current_action)
    control_selected = int(control_state.composition.prototype.current_action)
    if adaptive_selected != control_selected:
        raise ValueError("paired development initial actions differ")
    commands = _safe_commands()
    fallback = commands[1 - adaptive_selected]
    envelope_config = EmbodiedSafetyEnvelopeConfig(
        n_joints=2,
        joint_position_lower=(-1.0, -2.0),
        joint_position_upper=(1.0, 2.0),
        max_abs_joint_velocity=(1.0, 2.0),
        max_abs_joint_torque=(3.0, 4.0),
        workspace_lower=(-1.0, -1.0, 0.0),
        workspace_upper=(1.0, 1.0, 2.0),
        min_collision_clearance=0.1,
        fallback_joint_position=fallback.joint_position,
        fallback_joint_velocity=fallback.joint_velocity,
        fallback_joint_torque=fallback.joint_torque,
        fallback_workspace_position=fallback.workspace_position,
        fallback_collision_clearance=fallback.collision_clearance,
        reset_stationary_velocity_tolerance=0.01,
        max_telemetry_age_ticks=5,
        max_control_deadline_ticks=3,
        shadow_window=4,
        min_shadow_samples=3,
        min_shadow_success_lcb=0.5,
        wilson_z=1.0,
        max_shadow_calibration_error=0.2,
        max_shadow_latency_ticks=4,
        max_decisions=6,
        max_committed_actions=6,
        max_shadow_records=32,
        max_handshakes_per_kind=8,
        reset_authority_digest=(1, 2, 3, 4, 5, 6, 7, 8),
        rollback_authority_digest=(8, 7, 6, 5, 4, 3, 2, 1),
    )

    def make_harness(semantic: Any, semantic_state: Any) -> tuple[Any, Any]:
        adapter = PrototypeEmbodiedCommandAdapter(
            PrototypeEmbodiedCommandAdapterConfig(
                semantic=semantic.config,
                envelope=envelope_config,
                command_bank=commands,
            )
        )
        envelope = EmbodiedSafetyEnvelope(envelope_config)
        adapter_state = adapter.init(
            semantic_state,
            envelope.init(source_digest=SOURCE_DIGEST),
        )
        plant = DeterministicPrimitivePlant(
            DeterministicPrimitivePlantConfig(
                observation_lower=(-10.0, -10.0),
                observation_upper=(10.0, 10.0),
                primitive_deltas=((1.0, 0.0), (0.0, 1.0)),
                primitive_rewards=(0.25, 0.5),
                max_transitions=4,
            )
        )
        harness = PrototypeEmbodiedDevelopmentHarness(adapter, plant, grounded)
        state = harness.init(
            adapter_state,
            plant.init(jnp.zeros((2,), dtype=jnp.float32)),
            grounded_state,
        )
        return harness, state

    adaptive_harness, adaptive_initial = make_harness(adaptive_semantic, adaptive_state)
    control_harness, control_initial = make_harness(control_semantic, control_state)
    envelope_input = PrototypeEmbodiedCommandPreparationInput(
        telemetry=_telemetry(),
        envelope_decision_id=_words(1),
        envelope_action_id=_words(1),
        control_tick=_words(12),
        control_deadline_tick=_words(15),
        model_version=MODEL_VERSION,
        optimizer_version=OPTIMIZER_VERSION,
        lifecycle_version=LIFECYCLE_VERSION,
        untrusted_reward=jnp.asarray(7.0, dtype=jnp.float32),
        partner_metadata_digest=PARTNER_DIGEST,
        learned_cost_estimate=jnp.asarray(-1_000.0, dtype=jnp.float32),
    )
    template = PrototypeEmbodiedDevelopmentHarnessPreparationInput(
        envelope=envelope_input,
        model_state=model_state,
        action_support_counts=ACTION_SUPPORT,
        source_revision_words=REVISION_ONE,
        region_ids=jnp.zeros((1, 2), dtype=jnp.int32),
        safety_admitted=jnp.ones((1, 2), dtype=jnp.bool_),
        protected=jnp.zeros((1, 2), dtype=jnp.bool_),
    )
    return PrototypeEmbodiedPairedDevelopmentBenchmark(
        PrototypeEmbodiedPairedDevelopmentConfig(development_key=development_key),
        adaptive_harness=adaptive_harness,
        adaptive_initial_state=adaptive_initial,
        zero_step_harness=control_harness,
        zero_step_initial_state=control_initial,
        common_preparation_template=template,
    )


__all__ = ["build_prototype_embodied_paired_development_benchmark"]
