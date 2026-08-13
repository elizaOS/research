# mypy: disable-error-code="call-arg"
"""Canonical construction for the primitive-only HCCL continual dyad.

This module owns the exact production configuration that was previously only
assembled by integration-test donors.  The default selects the full 8,998-event
HCCL schedule.  The mechanics-smoke profile changes only the authenticated
world schedule and intentionally retains 8,998-event learner capacities.  The
versioned Core-L2 and Core-L3 profiles select uninterrupted 71,984- and
1,007,776-event worlds and size every agent lifetime guard to the complete life.

The factory can construct and deterministically initialize the L0 transaction.
It does not execute an event, reserve a seed, write an artifact, or grant
benchmark, evidence, threshold, or promotion authority.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping
from typing import Final

from jax import Array

from alberta_framework.core.context_inference import ContextInferenceConfig
from alberta_framework.core.context_lineage_retention_seam import (
    ContextLineageRetentionSeamConfig,
)
from alberta_framework.core.delight import CandidateUpdateAuditConfig
from alberta_framework.core.experiential_memory import ExperientialMemoryConfig
from alberta_framework.core.external_learned_state_live_memory_action_stack_adapter import (
    ExternalLearnedStateLiveMemoryActionStackConfig,
)
from alberta_framework.core.external_learned_state_router_audit_coordinator import (
    ExternalLearnedStateRouterAuditCoordinatorConfig,
)
from alberta_framework.core.feature_bank_router import FeatureBankRouterConfig
from alberta_framework.core.hccl_continual_dyad_transaction import (
    HCCL_CONTINUAL_DYAD_CONFIG_SCHEMA,
    HCCLContinualDyadState,
    HCCLContinualDyadTransaction,
    HCCLContinualDyadTransactionConfig,
)
from alberta_framework.core.hccl_world_attribution_adapter import (
    HCCLWorldAttributionAdapterConfig,
)
from alberta_framework.core.learned_experiential_memory_controller import (
    LearnedExperientialMemoryControllerConfig,
)
from alberta_framework.core.learning_signals import LearningSignalEstimatorConfig
from alberta_framework.core.learning_value_router import LearningValueRouterConfig
from alberta_framework.core.oak import OaKConfig
from alberta_framework.core.options import STOMPConfig
from alberta_framework.core.prototype_agent import PrototypeAgentConfig
from alberta_framework.core.prototype_factorized_partner_planner import (
    PrototypeFactorizedPartnerPlannerConfig,
)
from alberta_framework.core.prototype_feature_lifecycle import (
    PrototypeFeatureLifecycleConfig,
)
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
from alberta_framework.core.types import (
    DemonType,
    GVFSpec,
    HordeSpec,
    TraceMode,
    create_horde_spec,
)
from alberta_framework.core.world_model import ActionConditionedWorldModelConfig
from alberta_framework.streams.hccl_causal_core import (
    HCCL_CAUSAL_CORE_CANONICAL_PROFILE,
    HCCL_CAUSAL_CORE_L2_PROFILE,
    HCCL_CAUSAL_CORE_L3_PROFILE,
    HCCL_CAUSAL_CORE_SMOKE_PROFILE,
    HCCLCausalCoreConfig,
    hccl_causal_core_lifetime_for_profile,
)

HCCL_CONTINUAL_DYAD_FACTORY_CONFIG_SCHEMA: Final = (
    "alberta.hccl-continual-dyad-factory.config.v1"
)
HCCL_CONTINUAL_DYAD_FACTORY_STATUS: Final = (
    "l0-development-config-and-deterministic-initialization-only"
)
HCCL_CONTINUAL_DYAD_FACTORY_EVIDENCE_LEVEL: Final = "L0"

HCCL_CONTINUAL_DYAD_PROPOSAL_OWNER_DIGEST: Final = (
    0x10203040,
    0x50607080,
    0x90A0B0C0,
    0xD0E0F001,
    0x12345678,
    0x9ABCDEF0,
    0x0F1E2D3C,
    0x4B5A6978,
)
HCCL_CONTINUAL_DYAD_AGENT_0_OWNER_DIGEST: Final = (
    0xA101A101,
    0xA202A202,
    0xA303A303,
    0xA404A404,
    0xA505A505,
    0xA606A606,
    0xA707A707,
    0xA808A808,
)
HCCL_CONTINUAL_DYAD_AGENT_1_OWNER_DIGEST: Final = (
    0xC101C101,
    0xC202C202,
    0xC303C303,
    0xC404C404,
    0xC505C505,
    0xC606C606,
    0xC707C707,
    0xC808C808,
)
HCCL_CONTINUAL_DYAD_BINDING_OWNER_DIGEST: Final = (
    0xD101D101,
    0xD202D202,
    0xD303D303,
    0xD404D404,
    0xD505D505,
    0xD606D606,
    0xD707D707,
    0xD808D808,
)

_N_ACTIONS = 2
_PHYSICAL_DIM = 16
_CONTEXT_DIM = 3
_FAST_DIM = 4
_EXTERNAL_RAW_DIM = _PHYSICAL_DIM + _CONTEXT_DIM
_BASE_DIM = _EXTERNAL_RAW_DIM + _FAST_DIM
_ACTIVE_PAIR_SLOTS = 12
_PAIR_CANDIDATE_SLOTS = 120
_ROUTED_DIM = _BASE_DIM + _ACTIVE_PAIR_SLOTS
_FEATURE_REPLACEMENT_INTERVAL = 64
_MEMORY_CAPACITY = 64
_CANONICAL_EVENTS = 8_998
_HORDE_NAMES = (
    "task_discount_0p5",
    "task_discount_0p9",
    "task_discount_0p99",
    "partner_action",
    "safety_cost",
    "tv_occupancy",
    "target_zone_occupancy",
    "option_success_unavailable",
)
_HORDE_GAMMAS = (0.5, 0.9, 0.99, 0.9, 0.9, 0.9, 0.9, 0.9)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"nonfinite JSON constant {value!r} is forbidden")


@dataclasses.dataclass(frozen=True, slots=True)
class HCCLContinualDyadFactoryConfig:
    """Select one fixed HCCL schedule for the exact primitive-only v2 dyad."""

    schedule_profile: str = HCCL_CAUSAL_CORE_CANONICAL_PROFILE

    def __post_init__(self) -> None:
        if type(self.schedule_profile) is not str or self.schedule_profile not in (
            HCCL_CAUSAL_CORE_CANONICAL_PROFILE,
            HCCL_CAUSAL_CORE_SMOKE_PROFILE,
            HCCL_CAUSAL_CORE_L2_PROFILE,
            HCCL_CAUSAL_CORE_L3_PROFILE,
        ):
            raise ValueError("schedule_profile must select one fixed versioned HCCL profile")

    @classmethod
    def mechanics_smoke(cls) -> HCCLContinualDyadFactoryConfig:
        """Select the opt-in authenticated 420-event mechanics schedule."""

        return cls(schedule_profile=HCCL_CAUSAL_CORE_SMOKE_PROFILE)

    @classmethod
    def core_l2(cls) -> HCCLContinualDyadFactoryConfig:
        """Select the uninterrupted eight-cycle 71,984-event Core-L2 life."""

        return cls(schedule_profile=HCCL_CAUSAL_CORE_L2_PROFILE)

    @classmethod
    def core_l3(cls) -> HCCLContinualDyadFactoryConfig:
        """Select the uninterrupted 112-cycle 1,007,776-event Core-L3 life."""

        return cls(schedule_profile=HCCL_CAUSAL_CORE_L3_PROFILE)

    @property
    def maximum_committed_transitions(self) -> int:
        return hccl_causal_core_lifetime_for_profile(self.schedule_profile)

    @property
    def agent_lifetime_events(self) -> int:
        """Return the full agent capacity; smoke intentionally retains 8,998."""

        if self.schedule_profile == HCCL_CAUSAL_CORE_SMOKE_PROFILE:
            return _CANONICAL_EVENTS
        return self.maximum_committed_transitions

    @property
    def mechanics_smoke_enabled(self) -> bool:
        return self.schedule_profile == HCCL_CAUSAL_CORE_SMOKE_PROFILE

    def to_config(self) -> dict[str, object]:
        """Return the strict JSON-compatible side-effect and authority contract."""

        return {
            "type": type(self).__name__,
            "schema": HCCL_CONTINUAL_DYAD_FACTORY_CONFIG_SCHEMA,
            "transaction_config_schema": HCCL_CONTINUAL_DYAD_CONFIG_SCHEMA,
            "mechanism_status": HCCL_CONTINUAL_DYAD_FACTORY_STATUS,
            "evidence_level": HCCL_CONTINUAL_DYAD_FACTORY_EVIDENCE_LEVEL,
            "schedule_profile": self.schedule_profile,
            "maximum_committed_transitions": self.maximum_committed_transitions,
            "mechanics_smoke_enabled": self.mechanics_smoke_enabled,
            "agent_lifetime_events": self.agent_lifetime_events,
            "deterministic_initialization": True,
            "configuration_and_initialization_only": True,
            "schedule_execution_authorized": False,
            "transaction_execution_authorized": False,
            "benchmark_execution_authorized": False,
            "seed_reservation_or_consumption_authorized": False,
            "artifact_authorized": False,
            "output_writes_authorized": False,
            "evidence_authorized": False,
            "promotion_authorized": False,
            "scientific_promotion_allowed": False,
            "artifact_write_calls": 0,
            "artifact_bytes_written": 0,
        }

    @classmethod
    def from_config(
        cls,
        payload: Mapping[str, object],
    ) -> HCCLContinualDyadFactoryConfig:
        """Reconstruct only one exact current factory manifest."""

        if type(payload) is not dict:
            raise TypeError("continual-dyad factory config must be an exact dict")
        schedule_profile = payload.get("schedule_profile")
        if type(schedule_profile) is not str:
            raise ValueError("continual-dyad factory schedule_profile must be an exact string")
        candidate = cls(schedule_profile=schedule_profile)
        if _canonical_json_bytes(payload) != _canonical_json_bytes(candidate.to_config()):
            raise ValueError("continual-dyad factory config is noncanonical or unsupported")
        return candidate

    def to_json(self) -> str:
        """Serialize the complete factory declaration as canonical strict JSON."""

        return _canonical_json_bytes(self.to_config()).decode("utf-8")

    @classmethod
    def from_json(cls, payload: str) -> HCCLContinualDyadFactoryConfig:
        """Parse a complete declaration while rejecting duplicates and nonfinite values."""

        if type(payload) is not str:
            raise TypeError("continual-dyad factory JSON must be an exact string")
        try:
            decoded = json.loads(
                payload,
                object_pairs_hook=_unique_json_object,
                parse_constant=_reject_json_constant,
            )
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError("continual-dyad factory JSON is invalid or non-strict") from error
        if type(decoded) is not dict:
            raise ValueError("continual-dyad factory JSON must encode one object")
        return cls.from_config(decoded)


def _horde_spec() -> HordeSpec:
    return create_horde_spec(
        tuple(
            GVFSpec(
                name=name,
                demon_type=DemonType.PREDICTION,
                gamma=gamma,
                lamda=0.0,
                cumulant_index=index,
                terminal_reward=0.0,
            )
            for index, (name, gamma) in enumerate(
                zip(_HORDE_NAMES, _HORDE_GAMMAS, strict=True)
            )
        )
    )


def _prototype_config(agent_lifetime_events: int) -> PrototypeAgentConfig:
    lifecycle = PrototypeFeatureLifecycleConfig(
        base_feature_dim=_BASE_DIM,
        active_pair_slots=_ACTIVE_PAIR_SLOTS,
        candidate_pair_slots=_PAIR_CANDIDATE_SLOTS,
        n_tasks=1 + len(_HORDE_NAMES),
        n_options=0,
        n_primitive_actions=_N_ACTIONS,
        option_subtask_feature_indices=(),
        step_size_output=0.05,
        utility_decay=0.9,
        replacement_interval=_FEATURE_REPLACEMENT_INTERVAL,
        min_feature_age=0,
        candidate_min_age=0,
        promotion_margin=1.0,
        scale_normalizer_decay=0.9,
        scale_normalizer_epsilon=1.0e-6,
        carry_survivors=True,
        max_observations=agent_lifetime_events,
        managed_horde_demons=len(_HORDE_NAMES),
        pair_source_feature_dim=_PHYSICAL_DIM,
    )
    stomp = STOMPConfig(
        subtask_specs=(),
        observation_dim=_ROUTED_DIM,
        n_primitive_actions=_N_ACTIONS,
        base_step_size=0.01,
        base_avg_reward_step_size=0.01,
        base_trace_decay=0.0,
        base_hidden_sizes=(),
        option_step_size=0.05,
        option_avg_reward_step_size=0.01,
        option_trace_decay=0.0,
        option_gamma=0.99,
        option_model_decay=0.95,
        option_model_step_size=0.1,
        option_planning_backups_per_step=0,
        epsilon_base=0.0,
        epsilon_option=0.1,
        option_target_epsilon=None,
        option_importance_clip=10.0,
    )
    oak = OaKConfig(
        stomp=stomp,
        utility_ema_decay=0.99,
        curation_threshold=0.0,
        min_steps_before_curation=0,
    )
    return PrototypeAgentConfig(
        oak=oak,
        world_model=None,
        world_model_ensemble=None,
        model_replay_rehearsal=None,
        recurrent_latent_world_model_ensemble=None,
        dreaming=None,
        buffer_capacity=200,
        n_dreams_per_step=0,
        dream_next_observation_mode="model_prediction",
        horde_spec=_horde_spec(),
        horde_hidden_sizes=(),
        horde_step_size=0.1,
        ia=None,
        partner_policy_fusion=None,
        experiential_memory=None,
        experiential_memory_advantage_gate=None,
        gru_perception=None,
        state_builder=IdentityStateBuilderConfig(observation_dim=_BASE_DIM),
        learn_state_builder_from_world_model=False,
        representation_gradient_mixer=None,
        gradient_joy=None,
        learning_value_router=None,
        auto_curate_every=0,
        option_search_control=None,
        prototype_feature_lifecycle=lifecycle,
        prototype_feature_utility=None,
        prototype_feature_utility_curation=None,
        prototype_atomic_feature_world_memory=None,
    )


def _ensemble_config(agent_lifetime_events: int) -> RoutedLinearWorldModelEnsembleConfig:
    return RoutedLinearWorldModelEnsembleConfig(
        router=FeatureBankRouterConfig(
            base_dim=_BASE_DIM,
            active_slots=_ACTIVE_PAIR_SLOTS,
        ),
        world_model=ActionConditionedWorldModelConfig(
            observation_dim=_BASE_DIM,
            n_actions=_N_ACTIONS,
            gamma=0.99,
            observation_scale=None,
            reward_scale=1.0,
            predict_delta=True,
            hidden_sizes=(),
            step_size=0.02,
            sparsity=0.0,
            leaky_relu_slope=0.01,
            use_layer_norm=False,
            trace_mode=TraceMode.ACCUMULATING,
            utility_decay=0.99,
            error_decay=0.5,
            observation_clip_margin=0.05,
            max_delta_scale=5.0,
            include_action_interactions=False,
        ),
        signal_estimator=LearningSignalEstimatorConfig(
            ensemble_size=1,
            target_dim=_BASE_DIM + 2,
            variance_floor=1.0e-6,
            fast_loss_decay=0.8,
            slow_loss_decay=0.99,
            progress_warmup_steps=2,
            change_calibration_steps=2,
            change_z_threshold=3.0,
            change_temperature=0.5,
            change_decay=0.95,
            calibration_scale_floor=0.25,
            max_normalized_residual=1.0e6,
            max_input_magnitude=1_000.0,
            max_predicted_variance=10_000.0,
            max_observed_loss=10_000.0,
        ),
        ensemble_size=1,
        residual_variance_decay=0.8,
        residual_variance_warmup_steps=1,
        residual_variance_floor=1.0e-3,
        max_events=agent_lifetime_events,
        carry_survivors=True,
    )


def _learning_value_router_config(agent_lifetime_events: int) -> LearningValueRouterConfig:
    return LearningValueRouterConfig(
        normalization_min_count=2,
        max_steps=agent_lifetime_events,
        max_abs_advantage=1.0e6,
        max_action_surprisal=1.0e6,
        max_abs_paper_dg_delight=1.0e12,
        max_epistemic_surprise=1.0e12,
        max_aleatoric_uncertainty=1.0e12,
        max_abs_learning_progress=1.0e12,
        max_safety_cost=1.0e12,
        advantage_scale_floor=1.0e-6,
        action_surprisal_scale_floor=1.0e-6,
        paper_dg_delight_scale_floor=1.0e-6,
        epistemic_surprise_scale_floor=1.0e-6,
        aleatoric_uncertainty_scale_floor=1.0e-6,
        learning_progress_scale_floor=1.0e-6,
        change_probability_scale_floor=1.0e-6,
        safety_cost_scale_floor=1.0e-6,
        normalization_clip=10.0,
    )


def _candidate_audit_config() -> CandidateUpdateAuditConfig:
    return CandidateUpdateAuditConfig(
        candidate_semantics="update",
        gradient_step_size=1.0,
        max_update_norm=1.0,
        min_objective_decrease=0.0,
        max_retention_loss_increase=0.0,
        max_safety_cost_increase=0.0,
        min_objective_descent_alignment=0.0,
        min_retention_descent_alignment=0.0,
        min_safety_descent_alignment=0.0,
        alignment_temperature=0.1,
        norm_temperature=0.1,
        diagnostics_epsilon=1.0e-8,
    )


def _coordinator_config(
    agent_lifetime_events: int,
) -> ExternalLearnedStateRouterAuditCoordinatorConfig:
    return ExternalLearnedStateRouterAuditCoordinatorConfig(
        builder=LearnableGRUStateBuilderConfig(
            observation_dim=_EXTERNAL_RAW_DIM,
            n_actions=_N_ACTIONS,
            hidden_dim=_FAST_DIM,
            step_size=0.01,
            gradient_clip=10.0,
            initialization_scale=0.2,
            include_raw_observation=True,
            initial_update_bias=1.0,
            initial_reset_bias=0.0,
        ),
        inner=PrototypeRoutedLinearWorldModelEnsembleAdapterConfig(
            prototype=_prototype_config(agent_lifetime_events),
            ensemble=_ensemble_config(agent_lifetime_events),
        ),
        learning_value_router=_learning_value_router_config(agent_lifetime_events),
        candidate_audit=_candidate_audit_config(),
        max_events=agent_lifetime_events,
    )


def _learned_memory_config(
    agent_lifetime_events: int,
) -> LearnedExperientialMemoryControllerConfig:
    memory = ExperientialMemoryConfig(
        capacity=_MEMORY_CAPACITY,
        observation_dim=_EXTERNAL_RAW_DIM,
        key_dim=_EXTERNAL_RAW_DIM,
        action_dim=_N_ACTIONS,
        outcome_dim=_EXTERNAL_RAW_DIM,
        top_k=1,
        min_neighbors=1,
        distance_scale=1.0,
        min_similarity=0.0,
        min_effective_reliability=0.01,
        max_uncertainty=1.0,
        max_safety_cost=1.0,
        max_age=agent_lifetime_events,
        staleness_scale=float(agent_lifetime_events),
        utility_decay=1.0,
        eviction_utility_weight=1.0,
        eviction_recency_weight=1.0,
        recency_scale=10.0,
    )
    return LearnedExperientialMemoryControllerConfig(
        memory=memory,
        admission_step_size=0.05,
        retention_step_size=0.1,
        admission_threshold=0.0,
        initial_admission_bias=0.0,
        max_abs_admission_weight=8.0,
        max_abs_counterfactual_delta=1.0,
        retention_prior=0.5,
    )


def _context_config() -> ContextLineageRetentionSeamConfig:
    return ContextLineageRetentionSeamConfig(
        context=ContextInferenceConfig(
            n_actions=_N_ACTIONS,
            observation_dim=_N_ACTIONS,
            max_contexts=_CONTEXT_DIM,
            model_step_size=1.0,
            error_decay=0.0,
            switch_threshold=0.75,
            novelty_prior_error=0.5,
            update_error_gate=0.75,
            min_dwell=0,
            initial_reward_estimate=0.5,
        )
    )


def _world_config(factory_config: HCCLContinualDyadFactoryConfig) -> HCCLCausalCoreConfig:
    if factory_config.schedule_profile == HCCL_CAUSAL_CORE_CANONICAL_PROFILE:
        return HCCLCausalCoreConfig()
    if factory_config.schedule_profile == HCCL_CAUSAL_CORE_SMOKE_PROFILE:
        return HCCLCausalCoreConfig.mechanics_smoke()
    if factory_config.schedule_profile == HCCL_CAUSAL_CORE_L2_PROFILE:
        return HCCLCausalCoreConfig.core_l2()
    if factory_config.schedule_profile == HCCL_CAUSAL_CORE_L3_PROFILE:
        return HCCLCausalCoreConfig.core_l3()
    raise AssertionError("unreachable fixed HCCL continual-dyad profile")


def build_hccl_continual_dyad_config(
    factory_config: HCCLContinualDyadFactoryConfig | None = None,
) -> HCCLContinualDyadTransactionConfig:
    """Build the exact outer transaction config without executing or writing."""

    selected = HCCLContinualDyadFactoryConfig() if factory_config is None else factory_config
    if type(selected) is not HCCLContinualDyadFactoryConfig:
        raise TypeError("factory_config must be an exact HCCLContinualDyadFactoryConfig")
    coordinator = _coordinator_config(selected.agent_lifetime_events)
    learned_memory = _learned_memory_config(selected.agent_lifetime_events)
    return HCCLContinualDyadTransactionConfig(
        hccl=HCCLWorldAttributionAdapterConfig(
            proposal_owner_digest=HCCL_CONTINUAL_DYAD_PROPOSAL_OWNER_DIGEST,
            world_config=_world_config(selected),
        ),
        agent_0=ExternalLearnedStateLiveMemoryActionStackConfig(
            coordinator=coordinator,
            learned_memory=learned_memory,
            final_action_owner_digest=HCCL_CONTINUAL_DYAD_AGENT_0_OWNER_DIGEST,
        ),
        agent_1=ExternalLearnedStateLiveMemoryActionStackConfig(
            coordinator=coordinator,
            learned_memory=learned_memory,
            final_action_owner_digest=HCCL_CONTINUAL_DYAD_AGENT_1_OWNER_DIGEST,
        ),
        planner=PrototypeFactorizedPartnerPlannerConfig(
            observation_dim=_BASE_DIM,
            prototype_representation_dim=_ROUTED_DIM,
            n_actions=_N_ACTIONS,
            behavior_step_size=0.05,
            grounded_step_size=0.02,
            grounded_initialization_scale=0.25,
            planning_enabled=True,
            uniform_partner_belief=False,
        ),
        context=_context_config(),
        binding_owner_digest=HCCL_CONTINUAL_DYAD_BINDING_OWNER_DIGEST,
    )


@dataclasses.dataclass(frozen=True, slots=True)
class HCCLContinualDyadFactoryInitialization:
    """One deterministic initialization result with no event-execution authority."""

    factory_config: HCCLContinualDyadFactoryConfig
    transaction: HCCLContinualDyadTransaction
    state: HCCLContinualDyadState


class HCCLContinualDyadFactory:
    """Build and initialize the exact production-owned primitive-only dyad."""

    def __init__(
        self,
        config: HCCLContinualDyadFactoryConfig | None = None,
    ) -> None:
        selected = HCCLContinualDyadFactoryConfig() if config is None else config
        if type(selected) is not HCCLContinualDyadFactoryConfig:
            raise TypeError("config must be an exact HCCLContinualDyadFactoryConfig")
        self._config = selected
        self._transaction_config = build_hccl_continual_dyad_config(selected)

    @property
    def config(self) -> HCCLContinualDyadFactoryConfig:
        return self._config

    @property
    def transaction_config(self) -> HCCLContinualDyadTransactionConfig:
        return self._transaction_config

    def to_config(self) -> dict[str, object]:
        return self._config.to_config()

    def build(self) -> HCCLContinualDyadTransaction:
        """Construct one transaction owner without executing an event."""

        return HCCLContinualDyadTransaction(self._transaction_config)

    def init(self, key: Array) -> HCCLContinualDyadFactoryInitialization:
        """Deterministically initialize all owners without advancing the schedule."""

        transaction = self.build()
        state = transaction.init(key)
        return HCCLContinualDyadFactoryInitialization(
            factory_config=self._config,
            transaction=transaction,
            state=state,
        )


__all__ = (
    "HCCL_CONTINUAL_DYAD_AGENT_0_OWNER_DIGEST",
    "HCCL_CONTINUAL_DYAD_AGENT_1_OWNER_DIGEST",
    "HCCL_CONTINUAL_DYAD_BINDING_OWNER_DIGEST",
    "HCCL_CONTINUAL_DYAD_FACTORY_CONFIG_SCHEMA",
    "HCCL_CONTINUAL_DYAD_FACTORY_EVIDENCE_LEVEL",
    "HCCL_CONTINUAL_DYAD_FACTORY_STATUS",
    "HCCL_CONTINUAL_DYAD_PROPOSAL_OWNER_DIGEST",
    "HCCLContinualDyadFactory",
    "HCCLContinualDyadFactoryConfig",
    "HCCLContinualDyadFactoryInitialization",
    "build_hccl_continual_dyad_config",
)
