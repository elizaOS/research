# mypy: disable-error-code="call-arg"
"""Development-only uninterrupted-life evaluation for the hidden partner.

The evaluator in this module is deliberately separated from the promoted
evidence registry.  It asks whether the bounded integration kernel can learn
online through one complete ``hidden-partner-mapping-v0`` cycle and exposes
matched mechanism ablations.  The other agent is scripted and stochastic, so
even a strong result is learning *about* another agent, not co-adaptation,
intelligence amplification, or L3 Alberta Plan evidence.

The learner receives :class:`LearnerHiddenPartnerTransition`, which has no
oracle field.  Hidden regime identities, intended partner actions, segment
boundaries, and counterfactual rewards stay in the evaluator trace only.

All reported prediction values are prequential: the partner probability used
at step ``t`` is the probability that selected the focal decision before the
observed partner action at ``t`` updates any learner state.
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
import time
from collections.abc import Callable, Mapping, Sequence
from numbers import Real
from typing import Any, Literal, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array
from jaxtyping import Bool, Float, Int

from alberta_framework.core.integrated_hidden_partner import (
    BASE_FEATURE_DIM,
    IntegratedHiddenPartnerAgent,
    IntegratedHiddenPartnerConfig,
    IntegratedHiddenPartnerResourceBudget,
    IntegratedHiddenPartnerState,
)
from alberta_framework.streams.hidden_partner_mapping import (
    DEFAULT_REGIME_SCHEDULE,
    REGIME_NAMES,
    HiddenPartnerMappingConfig,
    HiddenPartnerMappingState,
    HiddenPartnerMappingTransition,
    HiddenPartnerMappingWorld,
)

HIDDEN_PARTNER_DEVELOPMENT_SCHEMA_VERSION = (
    "alberta.hidden-partner-development.uninterrupted-life.v1"
)
DEVELOPMENT_ONLY = True
SCIENTIFIC_PROMOTION_ALLOWED = False

TUNING_SEED_NAMESPACE = "hidden-partner-v0-dev-tuning-v1"
ROBUSTNESS_SEED_NAMESPACE = "hidden-partner-v0-dev-robustness-v1"
DEFAULT_DEVELOPMENT_SEED_COUNT = 8

ConditionName = Literal[
    "full",
    "state_frozen",
    "memory_masked",
    "lifecycle_frozen",
    "no_carry",
    "no_retention",
    "no_planning",
    "uniform_partner",
    "random_curation",
]

# Observation-channel index pairs whose products equal the partner's intended
# sign in the nonlinear regimes: C uses ``x_t * b_{t-1}`` (channels 0 and 2)
# and D uses ``u_t * v_t`` (channels 4 and 5).  Regimes A/B are linear in the
# raw ``x`` channel, so only C and D have a critical pair feature to discover.
_CRITICAL_C_PAIR = (0, 2)
_CRITICAL_D_PAIR = (4, 5)
_FLOAT_EPSILON = 1e-7
_BLOCK_UNTIL_READY = cast(Callable[[object], object], jax.block_until_ready)


@dataclasses.dataclass(frozen=True)
class HiddenPartnerCondition:
    """One fixed-shape development condition and its isolated question."""

    name: ConditionName
    config: IntegratedHiddenPartnerConfig
    isolated_question: str

    def to_config(self) -> dict[str, object]:
        """Return a strict JSON-compatible condition record."""
        return {
            "name": self.name,
            "agent_config": self.config.to_config(),
            "isolated_question": self.isolated_question,
        }


HIDDEN_PARTNER_CONDITIONS: tuple[HiddenPartnerCondition, ...] = (
    HiddenPartnerCondition(
        name="full",
        config=IntegratedHiddenPartnerConfig(),
        isolated_question="bounded integrated reference",
    ),
    HiddenPartnerCondition(
        name="state_frozen",
        config=IntegratedHiddenPartnerConfig(state_learning_enabled=False),
        isolated_question="learned recurrent parameters versus fixed random recurrence",
    ),
    HiddenPartnerCondition(
        name="memory_masked",
        config=IntegratedHiddenPartnerConfig(memory_masked=True),
        isolated_question="deployed recurrent hidden coordinates versus raw channels only",
    ),
    HiddenPartnerCondition(
        name="lifecycle_frozen",
        config=IntegratedHiddenPartnerConfig(feature_lifecycle_enabled=False),
        isolated_question="deployed online pair discovery versus the distractor birth bank",
    ),
    HiddenPartnerCondition(
        name="no_carry",
        config=IntegratedHiddenPartnerConfig(carry_survivors=False),
        isolated_question="consumer-column retention across descriptor transactions",
    ),
    HiddenPartnerCondition(
        name="no_retention",
        config=IntegratedHiddenPartnerConfig(active_utility_retention_decay=None),
        isolated_question="slow active-utility retention versus ordinary utility decay",
    ),
    HiddenPartnerCondition(
        name="no_planning",
        config=IntegratedHiddenPartnerConfig(planning_enabled=False),
        isolated_question="combined partner/world model term beyond differential SARSA",
    ),
    HiddenPartnerCondition(
        name="uniform_partner",
        config=IntegratedHiddenPartnerConfig(uniform_partner_belief=True),
        isolated_question="learned partner belief versus uniform planner belief",
    ),
    HiddenPartnerCondition(
        name="random_curation",
        config=IntegratedHiddenPartnerConfig(random_feature_curation=True),
        isolated_question="learned utility ranking versus seeded random curation",
    ),
)

_CONDITIONS_BY_NAME = {condition.name: condition for condition in HIDDEN_PARTNER_CONDITIONS}


@dataclasses.dataclass(frozen=True)
class HiddenPartnerDevelopmentProtocol:
    """Static one-cycle development protocol.

    Thresholds here are falsification aids for development.  They are not
    preregistered acceptance criteria and cannot promote a scientific claim.
    """

    environment: HiddenPartnerMappingConfig = dataclasses.field(
        default_factory=HiddenPartnerMappingConfig
    )
    recovery_window: int = 128
    recovery_reward_threshold: float = 0.80
    early_late_window: int = 256
    retention_ratio_threshold: float = 0.90
    recurrent_early_reward_threshold: float = 0.75

    def __post_init__(self) -> None:
        if not isinstance(self.environment, HiddenPartnerMappingConfig):
            raise TypeError("environment must be a HiddenPartnerMappingConfig")
        minimum_length = min(self.environment.base_segment_lengths) - (
            self.environment.jitter_radius
        )
        for name in ("recovery_window", "early_late_window"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive non-boolean integer")
            if value > minimum_length:
                raise ValueError(
                    f"{name} must not exceed the minimum possible segment length ({minimum_length})"
                )
        for name in (
            "recovery_reward_threshold",
            "retention_ratio_threshold",
            "recurrent_early_reward_threshold",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
            ):
                raise ValueError(f"{name} must be finite and lie in [0, 1]")

    @property
    def maximum_cycle_steps(self) -> int:
        """Fixed scan length covering every possible jittered v0 cycle."""
        return sum(self.environment.base_segment_lengths) + (
            len(self.environment.regime_schedule) * self.environment.jitter_radius
        )

    def to_config(self) -> dict[str, object]:
        """Return the full nonpromoting protocol record."""
        return {
            "schema_version": HIDDEN_PARTNER_DEVELOPMENT_SCHEMA_VERSION,
            "development_only": True,
            "scientific_promotion_allowed": False,
            "claim_scope": (
                "one uninterrupted v0 life learning online about a scripted stochastic partner"
            ),
            "excluded_claims": [
                "learning-partner coadaptation",
                "intelligence amplification",
                "general feature finding",
                "catastrophic-forgetting resistance beyond this fixed v0 life",
                "L3 Alberta Plan integration",
            ],
            "environment": self.environment.to_config(),
            "maximum_cycle_steps": self.maximum_cycle_steps,
            "recovery_window": self.recovery_window,
            "recovery_reward_threshold": float(self.recovery_reward_threshold),
            "early_late_window": self.early_late_window,
            "retention_ratio_threshold": float(self.retention_ratio_threshold),
            "recurrent_early_reward_threshold": float(self.recurrent_early_reward_threshold),
            "conditions": [condition.to_config() for condition in HIDDEN_PARTNER_CONDITIONS],
        }

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, object],
    ) -> HiddenPartnerDevelopmentProtocol:
        """Reconstruct only an exact nonpromoting v1 record."""
        expected = {
            "schema_version",
            "development_only",
            "scientific_promotion_allowed",
            "claim_scope",
            "excluded_claims",
            "environment",
            "maximum_cycle_steps",
            "recovery_window",
            "recovery_reward_threshold",
            "early_late_window",
            "retention_ratio_threshold",
            "recurrent_early_reward_threshold",
            "conditions",
        }
        if set(config) != expected:
            raise ValueError("development protocol fields do not match the v1 schema")
        if config.get("schema_version") != HIDDEN_PARTNER_DEVELOPMENT_SCHEMA_VERSION:
            raise ValueError("unsupported hidden-partner development schema")
        if config.get("development_only") is not True:
            raise ValueError("hidden-partner v1 must remain development-only")
        if config.get("scientific_promotion_allowed") is not False:
            raise ValueError("hidden-partner v1 cannot promote scientific evidence")
        environment_payload = config.get("environment")
        if not isinstance(environment_payload, Mapping):
            raise ValueError("environment must be a serialized mapping")
        protocol = cls(
            environment=HiddenPartnerMappingConfig.from_config(environment_payload),
            recovery_window=cast(int, config["recovery_window"]),
            recovery_reward_threshold=cast(float, config["recovery_reward_threshold"]),
            early_late_window=cast(int, config["early_late_window"]),
            retention_ratio_threshold=cast(float, config["retention_ratio_threshold"]),
            recurrent_early_reward_threshold=cast(
                float,
                config["recurrent_early_reward_threshold"],
            ),
        )
        if protocol.to_config() != dict(config):
            raise ValueError("development protocol contains changed claims or conditions")
        return protocol


@dataclasses.dataclass(frozen=True)
class HiddenPartnerSeedPair:
    """Separated, deterministic environment and agent initialization seeds."""

    namespace: str
    index: int
    stream_seed: int
    initialization_seed: int

    def to_dict(self) -> dict[str, int | str]:
        return dataclasses.asdict(self)


def derive_hidden_partner_seed_pairs(
    namespace: str,
    count: int = DEFAULT_DEVELOPMENT_SEED_COUNT,
) -> tuple[HiddenPartnerSeedPair, ...]:
    """Derive stable paired uint32 keys from a named development namespace."""
    if not isinstance(namespace, str) or not namespace:
        raise ValueError("namespace must be a non-empty string")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("count must be a positive non-boolean integer")
    pairs: list[HiddenPartnerSeedPair] = []
    for index in range(count):
        seeds: list[int] = []
        for role in ("stream", "initialization"):
            payload = (
                f"{HIDDEN_PARTNER_DEVELOPMENT_SCHEMA_VERSION}|{namespace}|{index}|{role}"
            ).encode()
            digest = hashlib.sha256(payload).digest()
            seeds.append(int.from_bytes(digest[:4], byteorder="big", signed=False))
        pairs.append(
            HiddenPartnerSeedPair(
                namespace=namespace,
                index=index,
                stream_seed=seeds[0],
                initialization_seed=seeds[1],
            )
        )
    return tuple(pairs)


@chex.dataclass(frozen=True)
class LearnerHiddenPartnerTransition:
    """Oracle-free structural transition passed to the integrated learner."""

    observation: Float[Array, " 8"]
    focal_action: Int[Array, ""]
    partner_action: Int[Array, ""]
    reward: Float[Array, ""]
    outcome: Float[Array, ""]
    next_observation: Float[Array, " 8"]
    terminated: Bool[Array, ""]
    discount: Float[Array, ""]


def strip_hidden_partner_oracle(
    transition: HiddenPartnerMappingTransition,
) -> LearnerHiddenPartnerTransition:
    """Copy only causal learner-visible fields from an environment transition."""
    return LearnerHiddenPartnerTransition(
        observation=transition.observation,
        focal_action=transition.focal_action,
        partner_action=transition.partner_action,
        reward=transition.reward,
        outcome=transition.outcome,
        next_observation=transition.next_observation,
        terminated=transition.terminated,
        discount=transition.discount,
    )


@chex.dataclass(frozen=True)
class HiddenPartnerStepTrace:
    """Fixed-shape primitive trace for metric reconstruction."""

    active: Bool[Array, ""]
    step: Int[Array, ""]
    segment_index: Int[Array, ""]
    segment_step: Int[Array, ""]
    segment_length: Int[Array, ""]
    regime_id: Int[Array, ""]
    reward: Float[Array, ""]
    partner_action: Int[Array, ""]
    partner_intended_action: Int[Array, ""]
    focal_action: Int[Array, ""]
    behavior_logits_preupdate: Float[Array, " 2"]
    behavior_probabilities: Float[Array, " 2"]
    applied_partner_probabilities: Float[Array, " 2"]
    behavior_nll: Float[Array, ""]
    behavior_brier: Float[Array, ""]
    behavior_actual_correct: Bool[Array, ""]
    behavior_intended_correct: Bool[Array, ""]
    planner_greedy_action: Int[Array, ""]
    q_greedy_action: Int[Array, ""]
    planner_intended_correct: Bool[Array, ""]
    executed_intended_correct: Bool[Array, ""]
    realized_counterfactual_regret: Float[Array, ""]
    expected_greedy_regret: Float[Array, ""]
    model_intervened: Bool[Array, ""]
    model_expected_reward_effect: Float[Array, ""]
    world_reward_absolute_error: Float[Array, ""]
    world_outcome_squared_error: Float[Array, ""]
    interaction_phi_pre: Float[Array, " 12"]
    interaction_target: Float[Array, " 1"]
    deployed_pair_features: Float[Array, " 12"]
    behavior_pair_weights_pre: Float[Array, "2 12"]
    behavior_pair_weights_post: Float[Array, "2 12"]
    control_pair_weights_pre: Float[Array, "2 12"]
    control_pair_weights_post: Float[Array, "2 12"]
    control_pair_trace_weights_pre: Float[Array, "2 12"]
    control_pair_trace_weights_post: Float[Array, "2 12"]
    # ``active_*`` fields are retained as v1 aliases for deployed pre-state.
    active_descriptors: Int[Array, "12 2"]
    deployed_descriptors_post: Int[Array, "12 2"]
    shadow_descriptors_pre: Int[Array, "12 2"]
    shadow_descriptors_post: Int[Array, "12 2"]
    candidate_descriptors: Int[Array, "66 2"]
    candidate_descriptors_post: Int[Array, "66 2"]
    active_utilities: Float[Array, " 12"]
    shadow_utilities_post: Float[Array, " 12"]
    shadow_idle_steps_pre: Int[Array, " 12"]
    shadow_idle_steps_post: Int[Array, " 12"]
    candidate_utilities: Float[Array, " 66"]
    candidate_utilities_post: Float[Array, " 66"]
    candidate_output_weights_pre: Float[Array, "1 66"]
    candidate_output_weights_post: Float[Array, "1 66"]
    candidate_ages_pre: Int[Array, " 66"]
    candidate_ages_post: Int[Array, " 66"]
    interaction_feature_ages_pre: Int[Array, " 12"]
    interaction_feature_ages_post: Int[Array, " 12"]
    interaction_replaced_slot: Int[Array, ""]
    interaction_promoted_candidate: Int[Array, ""]
    interaction_retired_slot: Int[Array, ""]
    interaction_retired_left: Int[Array, ""]
    interaction_retired_right: Int[Array, ""]
    interaction_evidence_refreshed: Bool[Array, " 12"]
    interaction_retention_evidence_refreshed: Bool[Array, " 12"]
    interaction_relevance_probe_scores: Float[Array, " 12"]
    interaction_relevance_probe_errors: Float[Array, "1 12"]
    interaction_relevance_probe_weights_pre: Float[Array, "1 12"]
    interaction_relevance_probe_weights_post: Float[Array, "1 12"]
    interaction_relevance_probe_biases_pre: Float[Array, " 1"]
    interaction_relevance_probe_biases_post: Float[Array, " 1"]
    interaction_feature_second_moments_pre: Float[Array, " 12"]
    interaction_feature_second_moments_post: Float[Array, " 12"]
    interaction_candidate_second_moments_pre: Float[Array, " 66"]
    interaction_candidate_second_moments_post: Float[Array, " 66"]
    interaction_target_second_moments_pre: Float[Array, " 1"]
    interaction_target_second_moments_post: Float[Array, " 1"]
    interaction_candidate_promotion_signal: Float[Array, " 66"]
    interaction_candidate_promotion_raw_evidence: Bool[Array, " 66"]
    interaction_candidate_promotion_evidence_streak_pre: Int[Array, " 66"]
    interaction_candidate_promotion_evidence_streak_updated: Int[Array, " 66"]
    interaction_candidate_promotion_evidence_streak_post: Int[Array, " 66"]
    interaction_candidate_promotion_confirmed: Bool[Array, " 66"]
    interaction_candidate_reacquisition_required_pre: Bool[Array, " 66"]
    interaction_candidate_reacquisition_required_post: Bool[Array, " 66"]
    interaction_candidate_reacquisition_confirmed: Bool[Array, " 66"]
    interaction_durable_read_mask: Bool[Array, " 12"]
    interaction_output_weights_pre: Float[Array, "1 12"]
    interaction_output_weights_post: Float[Array, "1 12"]
    interaction_utility_evidence_streak_pre: Int[Array, " 12"]
    interaction_utility_evidence_streak_post: Int[Array, " 12"]
    interaction_active_output_memory_committed_pre: Bool[Array, " 12"]
    interaction_active_output_memory_committed_post: Bool[Array, " 12"]
    consumer_write_gate_pre: Bool[Array, " 12"]
    consumer_read_idle_steps_pre: Int[Array, " 12"]
    consumer_read_idle_steps_post: Int[Array, " 12"]
    consumer_active_mask_pre: Bool[Array, " 12"]
    consumer_active_mask_post: Bool[Array, " 12"]
    interaction_matching_candidate_reset_mask: Bool[Array, " 66"]
    interaction_matching_candidate_reset_count: Int[Array, ""]
    interaction_live_feature_count: Int[Array, ""]
    interaction_vacancy_count: Int[Array, ""]
    interaction_promoted_into_vacancy: Bool[Array, ""]
    c_masked_behavior_nll_increase: Float[Array, ""]
    d_masked_behavior_nll_increase: Float[Array, ""]
    c_masked_planner_margin_decrease: Float[Array, ""]
    d_masked_planner_margin_decrease: Float[Array, ""]
    descriptors_changed: Bool[Array, ""]
    state_builder_step_delta: Int[Array, ""]
    state_builder_learning_delta: Int[Array, ""]
    behavior_step_delta: Int[Array, ""]
    interaction_step_delta: Int[Array, ""]
    world_step_delta: Int[Array, ""]
    control_step_delta: Int[Array, ""]
    router_route_delta: Int[Array, ""]
    integrated_step_delta: Int[Array, ""]
    route_valid: Bool[Array, ""]
    causal_transition_valid: Bool[Array, ""]
    all_finite: Bool[Array, ""]


@dataclasses.dataclass(frozen=True)
class HiddenPartnerSegmentSummary:
    """Host summary for one evaluator-only schedule segment."""

    segment_index: int
    regime_id: int
    regime_name: str
    length: int
    mean_reward: float
    early_reward: float
    late_reward: float
    mean_behavior_nll: float
    late_behavior_nll: float
    mean_behavior_brier: float
    intended_prediction_accuracy: float
    late_intended_prediction_accuracy: float
    mean_realized_regret: float
    mean_expected_greedy_regret: float
    recovery_steps: int
    prior_same_regime_segment: int | None
    recurrent_early_to_prior_late_ratio: float | None
    recurrence_retained: bool | None

    def to_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class HiddenPartnerFeatureSummary:
    """Critical active-bank and candidate-archive diagnostics."""

    c_first_active_step: int | None
    d_first_active_step: int | None
    c_active_evictions: int
    d_active_evictions: int
    c_active_late_first_c: bool
    c_active_at_recurrent_c_entry: bool
    c_survived_first_to_recurrent_c: bool
    d_active_at_end_of_d: bool
    d_active_at_life_end: bool
    c_candidate_fraction: float
    d_candidate_fraction: float

    def to_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class HiddenPartnerRunSummary:
    """Compact, reconstructable result for one condition and seed pair."""

    condition: ConditionName
    seed_pair: HiddenPartnerSeedPair
    cycle_steps: int
    segment_lengths: tuple[int, ...]
    mean_reward: float
    normalized_control_score: float
    mean_behavior_nll: float
    mean_behavior_brier: float
    behavior_actual_accuracy: float
    behavior_intended_accuracy: float
    planner_intended_accuracy: float
    executed_intended_accuracy: float
    mean_realized_counterfactual_regret: float
    mean_expected_greedy_regret: float
    model_intervention_rate: float
    helpful_model_intervention_rate: float
    harmful_model_intervention_rate: float
    mean_world_reward_absolute_error: float
    mean_world_outcome_squared_error: float
    descriptor_transaction_count: int
    counter_contract_valid: bool
    causal_contract_valid: bool
    all_finite: bool
    initial_state_nbytes: int
    final_state_nbytes: int
    resource_shape_matched: bool
    compilation_wall_seconds: float
    execution_wall_seconds: float
    mean_execution_microseconds_per_step: float
    segments: tuple[HiddenPartnerSegmentSummary, ...]
    features: HiddenPartnerFeatureSummary

    def to_dict(self) -> dict[str, object]:
        payload = dataclasses.asdict(self)
        payload["development_only"] = True
        payload["scientific_promotion_allowed"] = False
        return payload


def _strict_summary_int(value: object, field: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{field} must be a non-boolean integer")
    return value


def _strict_summary_float(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _strict_summary_bool(value: object, field: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{field} must be a boolean")
    return value


def _strict_optional_summary_int(value: object, field: str) -> int | None:
    return None if value is None else _strict_summary_int(value, field)


def _strict_optional_summary_float(value: object, field: str) -> float | None:
    return None if value is None else _strict_summary_float(value, field)


def hidden_partner_run_summary_from_dict(
    payload: Mapping[str, object],
) -> HiddenPartnerRunSummary:
    """Strictly reconstruct one compact run summary from artifact JSON."""
    expected = {
        *(field.name for field in dataclasses.fields(HiddenPartnerRunSummary)),
        "development_only",
        "scientific_promotion_allowed",
    }
    if set(payload) != expected:
        raise ValueError("run summary fields do not match the v1 schema")
    if payload.get("development_only") is not True:
        raise ValueError("run summary must remain development-only")
    if payload.get("scientific_promotion_allowed") is not False:
        raise ValueError("run summary must forbid scientific promotion")

    condition_value = payload["condition"]
    if condition_value not in _CONDITIONS_BY_NAME:
        raise ValueError(f"unknown run condition: {condition_value!r}")
    condition = condition_value

    seed_payload = payload["seed_pair"]
    if not isinstance(seed_payload, Mapping) or set(seed_payload) != {
        "namespace",
        "index",
        "stream_seed",
        "initialization_seed",
    }:
        raise ValueError("run seed_pair fields do not match the v1 schema")
    namespace = seed_payload["namespace"]
    if not isinstance(namespace, str) or not namespace:
        raise ValueError("run seed namespace must be a non-empty string")
    seed_pair = HiddenPartnerSeedPair(
        namespace=namespace,
        index=_strict_summary_int(seed_payload["index"], "seed_pair.index"),
        stream_seed=_strict_summary_int(
            seed_payload["stream_seed"],
            "seed_pair.stream_seed",
        ),
        initialization_seed=_strict_summary_int(
            seed_payload["initialization_seed"],
            "seed_pair.initialization_seed",
        ),
    )
    if seed_pair.index < 0:
        raise ValueError("seed_pair.index must be non-negative")
    if not 0 <= seed_pair.stream_seed <= 0xFFFFFFFF:
        raise ValueError("seed_pair.stream_seed must be uint32")
    if not 0 <= seed_pair.initialization_seed <= 0xFFFFFFFF:
        raise ValueError("seed_pair.initialization_seed must be uint32")

    lengths_payload = payload["segment_lengths"]
    if not isinstance(lengths_payload, (list, tuple)):
        raise ValueError("segment_lengths must be an array")
    segment_lengths = tuple(
        _strict_summary_int(value, f"segment_lengths[{index}]")
        for index, value in enumerate(lengths_payload)
    )
    if len(segment_lengths) != len(DEFAULT_REGIME_SCHEDULE) or any(
        value <= 0 for value in segment_lengths
    ):
        raise ValueError("segment_lengths must contain nine positive values")

    segments_payload = payload["segments"]
    if not isinstance(segments_payload, (list, tuple)):
        raise ValueError("segments must be an array")
    if len(segments_payload) != len(DEFAULT_REGIME_SCHEDULE):
        raise ValueError("segments must contain the exact nine v0 records")
    segment_field_names = {field.name for field in dataclasses.fields(HiddenPartnerSegmentSummary)}
    segments: list[HiddenPartnerSegmentSummary] = []
    prior_by_regime: dict[int, int] = {}
    for index, raw_segment in enumerate(segments_payload):
        if not isinstance(raw_segment, Mapping) or set(raw_segment) != segment_field_names:
            raise ValueError(f"segments[{index}] fields do not match the v1 schema")
        regime_id = _strict_summary_int(
            raw_segment["regime_id"],
            f"segments[{index}].regime_id",
        )
        expected_regime = DEFAULT_REGIME_SCHEDULE[index]
        if regime_id != expected_regime:
            raise ValueError(f"segments[{index}] has the wrong regime")
        regime_name = raw_segment["regime_name"]
        if regime_name != REGIME_NAMES[regime_id]:
            raise ValueError(f"segments[{index}] has the wrong regime name")
        prior = _strict_optional_summary_int(
            raw_segment["prior_same_regime_segment"],
            f"segments[{index}].prior_same_regime_segment",
        )
        if prior != prior_by_regime.get(regime_id):
            raise ValueError(f"segments[{index}] has inconsistent recurrence linkage")
        retention_ratio = _strict_optional_summary_float(
            raw_segment["recurrent_early_to_prior_late_ratio"],
            f"segments[{index}].recurrent_early_to_prior_late_ratio",
        )
        retained_value = raw_segment["recurrence_retained"]
        if retained_value is not None and type(retained_value) is not bool:
            raise ValueError(f"segments[{index}].recurrence_retained must be boolean or null")
        if (prior is None) != (retention_ratio is None) or (prior is None) != (
            retained_value is None
        ):
            raise ValueError(f"segments[{index}] recurrence fields are inconsistent")
        segment = HiddenPartnerSegmentSummary(
            segment_index=_strict_summary_int(
                raw_segment["segment_index"],
                f"segments[{index}].segment_index",
            ),
            regime_id=regime_id,
            regime_name=cast(str, regime_name),
            length=_strict_summary_int(
                raw_segment["length"],
                f"segments[{index}].length",
            ),
            mean_reward=_strict_summary_float(
                raw_segment["mean_reward"],
                f"segments[{index}].mean_reward",
            ),
            early_reward=_strict_summary_float(
                raw_segment["early_reward"],
                f"segments[{index}].early_reward",
            ),
            late_reward=_strict_summary_float(
                raw_segment["late_reward"],
                f"segments[{index}].late_reward",
            ),
            mean_behavior_nll=_strict_summary_float(
                raw_segment["mean_behavior_nll"],
                f"segments[{index}].mean_behavior_nll",
            ),
            late_behavior_nll=_strict_summary_float(
                raw_segment["late_behavior_nll"],
                f"segments[{index}].late_behavior_nll",
            ),
            mean_behavior_brier=_strict_summary_float(
                raw_segment["mean_behavior_brier"],
                f"segments[{index}].mean_behavior_brier",
            ),
            intended_prediction_accuracy=_strict_summary_float(
                raw_segment["intended_prediction_accuracy"],
                f"segments[{index}].intended_prediction_accuracy",
            ),
            late_intended_prediction_accuracy=_strict_summary_float(
                raw_segment["late_intended_prediction_accuracy"],
                f"segments[{index}].late_intended_prediction_accuracy",
            ),
            mean_realized_regret=_strict_summary_float(
                raw_segment["mean_realized_regret"],
                f"segments[{index}].mean_realized_regret",
            ),
            mean_expected_greedy_regret=_strict_summary_float(
                raw_segment["mean_expected_greedy_regret"],
                f"segments[{index}].mean_expected_greedy_regret",
            ),
            recovery_steps=_strict_summary_int(
                raw_segment["recovery_steps"],
                f"segments[{index}].recovery_steps",
            ),
            prior_same_regime_segment=prior,
            recurrent_early_to_prior_late_ratio=retention_ratio,
            recurrence_retained=retained_value,
        )
        if segment.segment_index != index or segment.length != segment_lengths[index]:
            raise ValueError(f"segments[{index}] index or length is inconsistent")
        for name in (
            "mean_reward",
            "early_reward",
            "late_reward",
            "intended_prediction_accuracy",
            "late_intended_prediction_accuracy",
        ):
            if not 0.0 <= getattr(segment, name) <= 1.0:
                raise ValueError(f"segments[{index}].{name} must lie in [0, 1]")
        for name in (
            "mean_behavior_nll",
            "late_behavior_nll",
            "mean_behavior_brier",
            "mean_realized_regret",
            "mean_expected_greedy_regret",
        ):
            if getattr(segment, name) < 0.0:
                raise ValueError(f"segments[{index}].{name} must be non-negative")
        if not 0 < segment.recovery_steps <= segment.length:
            raise ValueError(f"segments[{index}].recovery_steps is outside the segment")
        if retention_ratio is not None and retention_ratio < 0.0:
            raise ValueError(f"segments[{index}] retention ratio must be non-negative")
        prior_by_regime[regime_id] = index
        segments.append(segment)
    feature_payload = payload["features"]
    feature_fields = {field.name for field in dataclasses.fields(HiddenPartnerFeatureSummary)}
    if not isinstance(feature_payload, Mapping) or set(feature_payload) != feature_fields:
        raise ValueError("feature summary fields do not match the v1 schema")
    features = HiddenPartnerFeatureSummary(
        c_first_active_step=_strict_optional_summary_int(
            feature_payload["c_first_active_step"],
            "features.c_first_active_step",
        ),
        d_first_active_step=_strict_optional_summary_int(
            feature_payload["d_first_active_step"],
            "features.d_first_active_step",
        ),
        c_active_evictions=_strict_summary_int(
            feature_payload["c_active_evictions"],
            "features.c_active_evictions",
        ),
        d_active_evictions=_strict_summary_int(
            feature_payload["d_active_evictions"],
            "features.d_active_evictions",
        ),
        c_active_late_first_c=_strict_summary_bool(
            feature_payload["c_active_late_first_c"],
            "features.c_active_late_first_c",
        ),
        c_active_at_recurrent_c_entry=_strict_summary_bool(
            feature_payload["c_active_at_recurrent_c_entry"],
            "features.c_active_at_recurrent_c_entry",
        ),
        c_survived_first_to_recurrent_c=_strict_summary_bool(
            feature_payload["c_survived_first_to_recurrent_c"],
            "features.c_survived_first_to_recurrent_c",
        ),
        d_active_at_end_of_d=_strict_summary_bool(
            feature_payload["d_active_at_end_of_d"],
            "features.d_active_at_end_of_d",
        ),
        d_active_at_life_end=_strict_summary_bool(
            feature_payload["d_active_at_life_end"],
            "features.d_active_at_life_end",
        ),
        c_candidate_fraction=_strict_summary_float(
            feature_payload["c_candidate_fraction"],
            "features.c_candidate_fraction",
        ),
        d_candidate_fraction=_strict_summary_float(
            feature_payload["d_candidate_fraction"],
            "features.d_candidate_fraction",
        ),
    )
    if features.c_survived_first_to_recurrent_c != (
        features.c_active_late_first_c and features.c_active_at_recurrent_c_entry
    ):
        raise ValueError("critical-C survival fields are inconsistent")
    if features.c_active_evictions < 0 or features.d_active_evictions < 0:
        raise ValueError("feature eviction counts must be non-negative")
    cycle_steps_from_segments = sum(segment_lengths)
    for name in ("c_first_active_step", "d_first_active_step"):
        step = getattr(features, name)
        if step is not None and not 0 <= step < cycle_steps_from_segments:
            raise ValueError(f"features.{name} must identify a step inside the life")
    if (
        features.c_active_evictions > cycle_steps_from_segments
        or features.d_active_evictions > cycle_steps_from_segments
    ):
        raise ValueError("feature eviction counts cannot exceed the life length")
    for name in ("c_candidate_fraction", "d_candidate_fraction"):
        if not 0.0 <= getattr(features, name) <= 1.0:
            raise ValueError(f"features.{name} must lie in [0, 1]")

    integer_fields = (
        "cycle_steps",
        "descriptor_transaction_count",
        "initial_state_nbytes",
        "final_state_nbytes",
    )
    integers = {name: _strict_summary_int(payload[name], name) for name in integer_fields}
    if integers["cycle_steps"] != cycle_steps_from_segments:
        raise ValueError("cycle_steps must equal the sum of segment_lengths")
    if not 0 <= integers["descriptor_transaction_count"] <= integers["cycle_steps"]:
        raise ValueError("descriptor_transaction_count is outside the life")
    if integers["initial_state_nbytes"] <= 0 or integers["final_state_nbytes"] <= 0:
        raise ValueError("state byte counts must be positive")

    float_fields = (
        "mean_reward",
        "normalized_control_score",
        "mean_behavior_nll",
        "mean_behavior_brier",
        "behavior_actual_accuracy",
        "behavior_intended_accuracy",
        "planner_intended_accuracy",
        "executed_intended_accuracy",
        "mean_realized_counterfactual_regret",
        "mean_expected_greedy_regret",
        "model_intervention_rate",
        "helpful_model_intervention_rate",
        "harmful_model_intervention_rate",
        "mean_world_reward_absolute_error",
        "mean_world_outcome_squared_error",
        "compilation_wall_seconds",
        "execution_wall_seconds",
        "mean_execution_microseconds_per_step",
    )
    floats = {name: _strict_summary_float(payload[name], name) for name in float_fields}
    for name in (
        "mean_reward",
        "behavior_actual_accuracy",
        "behavior_intended_accuracy",
        "planner_intended_accuracy",
        "executed_intended_accuracy",
        "model_intervention_rate",
        "helpful_model_intervention_rate",
        "harmful_model_intervention_rate",
    ):
        if not 0.0 <= floats[name] <= 1.0:
            raise ValueError(f"{name} must lie in [0, 1]")
    for name in (
        "mean_behavior_nll",
        "mean_behavior_brier",
        "mean_realized_counterfactual_regret",
        "mean_expected_greedy_regret",
        "mean_world_reward_absolute_error",
        "mean_world_outcome_squared_error",
        "compilation_wall_seconds",
        "execution_wall_seconds",
        "mean_execution_microseconds_per_step",
    ):
        if floats[name] < 0.0:
            raise ValueError(f"{name} must be non-negative")
    if (
        floats["helpful_model_intervention_rate"] + floats["harmful_model_intervention_rate"]
        > floats["model_intervention_rate"] + 1e-12
    ):
        raise ValueError("helpful and harmful intervention rates are inconsistent")
    weighted_segment_fields = {
        "mean_reward": "mean_reward",
        "mean_behavior_nll": "mean_behavior_nll",
        "mean_behavior_brier": "mean_behavior_brier",
        "behavior_intended_accuracy": "intended_prediction_accuracy",
        "mean_realized_counterfactual_regret": "mean_realized_regret",
        "mean_expected_greedy_regret": "mean_expected_greedy_regret",
    }
    for summary_field, segment_field in weighted_segment_fields.items():
        reconstructed = (
            sum(segment.length * getattr(segment, segment_field) for segment in segments)
            / cycle_steps_from_segments
        )
        if not math.isclose(
            floats[summary_field],
            reconstructed,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(f"{summary_field} does not reconstruct from segment summaries")

    boolean_fields = (
        "counter_contract_valid",
        "causal_contract_valid",
        "all_finite",
        "resource_shape_matched",
    )
    booleans = {name: _strict_summary_bool(payload[name], name) for name in boolean_fields}
    if booleans["resource_shape_matched"] != (
        integers["initial_state_nbytes"] == integers["final_state_nbytes"]
    ):
        raise ValueError("resource_shape_matched is inconsistent with byte counts")
    # Recompute the score's fixed anchors: chance reward is 0.5 and the
    # epsilon-greedy ceiling is (1 - eps) * 0.95 + eps * 0.5 (matching the
    # partner's intended sign earns E[r] = 0.95 under the 0.05 flip rate).
    expected_normalized = (floats["mean_reward"] - 0.5) / (
        (1.0 - _CONDITIONS_BY_NAME[condition].config.epsilon) * 0.95
        + _CONDITIONS_BY_NAME[condition].config.epsilon * 0.5
        - 0.5
    )
    if not math.isclose(
        floats["normalized_control_score"],
        expected_normalized,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError("normalized_control_score is inconsistent with mean_reward")
    expected_microseconds = floats["execution_wall_seconds"] * 1e6 / integers["cycle_steps"]
    if not math.isclose(
        floats["mean_execution_microseconds_per_step"],
        expected_microseconds,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError("mean execution time per step is inconsistent")

    return HiddenPartnerRunSummary(
        condition=condition,
        seed_pair=seed_pair,
        cycle_steps=integers["cycle_steps"],
        segment_lengths=segment_lengths,
        mean_reward=floats["mean_reward"],
        normalized_control_score=floats["normalized_control_score"],
        mean_behavior_nll=floats["mean_behavior_nll"],
        mean_behavior_brier=floats["mean_behavior_brier"],
        behavior_actual_accuracy=floats["behavior_actual_accuracy"],
        behavior_intended_accuracy=floats["behavior_intended_accuracy"],
        planner_intended_accuracy=floats["planner_intended_accuracy"],
        executed_intended_accuracy=floats["executed_intended_accuracy"],
        mean_realized_counterfactual_regret=floats["mean_realized_counterfactual_regret"],
        mean_expected_greedy_regret=floats["mean_expected_greedy_regret"],
        model_intervention_rate=floats["model_intervention_rate"],
        helpful_model_intervention_rate=floats["helpful_model_intervention_rate"],
        harmful_model_intervention_rate=floats["harmful_model_intervention_rate"],
        mean_world_reward_absolute_error=floats["mean_world_reward_absolute_error"],
        mean_world_outcome_squared_error=floats["mean_world_outcome_squared_error"],
        descriptor_transaction_count=integers["descriptor_transaction_count"],
        counter_contract_valid=booleans["counter_contract_valid"],
        causal_contract_valid=booleans["causal_contract_valid"],
        all_finite=booleans["all_finite"],
        initial_state_nbytes=integers["initial_state_nbytes"],
        final_state_nbytes=integers["final_state_nbytes"],
        resource_shape_matched=booleans["resource_shape_matched"],
        compilation_wall_seconds=floats["compilation_wall_seconds"],
        execution_wall_seconds=floats["execution_wall_seconds"],
        mean_execution_microseconds_per_step=floats["mean_execution_microseconds_per_step"],
        segments=tuple(segments),
        features=features,
    )


@dataclasses.dataclass(frozen=True)
class HiddenPartnerRunResult:
    """Full primitive trace plus compact development summary."""

    condition: HiddenPartnerCondition
    summary: HiddenPartnerRunSummary
    trace: HiddenPartnerStepTrace
    final_agent_state: IntegratedHiddenPartnerState
    final_environment_state: HiddenPartnerMappingState
    initial_resource: IntegratedHiddenPartnerResourceBudget
    final_resource: IntegratedHiddenPartnerResourceBudget


def _pair_present(descriptors: Array, left: int, right: int) -> Array:
    pair = jnp.asarray((left, right), dtype=jnp.int32)
    return jnp.any(jnp.all(descriptors == pair, axis=1))


def _select_carry(active: Array, proposed: Any, current: Any) -> Any:
    return jax.tree_util.tree_map(
        lambda new, old: jnp.where(active, new, old),
        proposed,
        current,
    )


class HiddenPartnerDevelopmentRunner:
    """Reusable compiled runner for one condition and one static protocol."""

    def __init__(
        self,
        condition: HiddenPartnerCondition | ConditionName,
        protocol: HiddenPartnerDevelopmentProtocol | None = None,
    ) -> None:
        if isinstance(condition, str):
            if condition not in _CONDITIONS_BY_NAME:
                raise ValueError(f"unknown hidden-partner condition: {condition!r}")
            resolved = _CONDITIONS_BY_NAME[condition]
        elif isinstance(condition, HiddenPartnerCondition):
            resolved = condition
        else:
            raise TypeError("condition must be a name or HiddenPartnerCondition")
        self.condition = resolved
        self.protocol = HiddenPartnerDevelopmentProtocol() if protocol is None else protocol
        if not isinstance(self.protocol, HiddenPartnerDevelopmentProtocol):
            raise TypeError("protocol must be a HiddenPartnerDevelopmentProtocol")
        self.agent = IntegratedHiddenPartnerAgent(resolved.config)
        self.world = HiddenPartnerMappingWorld(self.protocol.environment)
        self._jitted_life = jax.jit(self._scan_life)
        self._compiled_life: Any | None = None

    def _scan_life(
        self,
        initial_agent: IntegratedHiddenPartnerState,
        initial_environment: HiddenPartnerMappingState,
        cycle_length: Array,
    ) -> tuple[
        tuple[IntegratedHiddenPartnerState, HiddenPartnerMappingState],
        HiddenPartnerStepTrace,
    ]:
        def step(
            carry: tuple[IntegratedHiddenPartnerState, HiddenPartnerMappingState],
            _: None,
        ) -> tuple[
            tuple[IntegratedHiddenPartnerState, HiddenPartnerMappingState],
            HiddenPartnerStepTrace,
        ]:
            agent_state, environment_state = carry
            active = environment_state.step_count < cycle_length
            transition, proposed_environment = self.world.step(
                environment_state,
                agent_state.control.last_action,
            )
            learner_transition = strip_hidden_partner_oracle(transition)
            update = self.agent.update(agent_state, learner_transition)

            current_evaluation = update.diagnostics.current_evaluation
            probabilities = update.diagnostics.behavior_probabilities_preupdate
            partner_action = transition.partner_action
            intended_action = transition.oracle.partner_intended_action
            prediction = jnp.argmax(probabilities).astype(jnp.int32)
            selected_probability = probabilities[partner_action]
            one_hot = jax.nn.one_hot(partner_action, 2, dtype=jnp.float32)
            planner_action = update.diagnostics.current_evaluation.greedy_action
            q_action = jnp.argmax(update.diagnostics.current_evaluation.q_values).astype(jnp.int32)
            planner_expected_reward = jnp.where(
                planner_action == intended_action,
                jnp.asarray(0.95, dtype=jnp.float32),
                jnp.asarray(0.05, dtype=jnp.float32),
            )
            q_expected_reward = jnp.where(
                q_action == intended_action,
                jnp.asarray(0.95, dtype=jnp.float32),
                jnp.asarray(0.05, dtype=jnp.float32),
            )

            def critical_mask_effect(
                pair: tuple[int, int],
            ) -> tuple[Array, Array]:
                target_pair = jnp.asarray(pair, dtype=jnp.int32)
                matches = jnp.all(
                    agent_state.router.descriptors == target_pair,
                    axis=1,
                )
                present = jnp.any(matches)
                slot = jnp.argmax(matches).astype(jnp.int32)
                feature_index = BASE_FEATURE_DIM + slot
                masked_chi = agent_state.chi.at[feature_index].set(
                    jnp.where(
                        present,
                        jnp.asarray(0.0, dtype=jnp.float32),
                        agent_state.chi[feature_index],
                    )
                )
                masked = self.agent.evaluate_models(
                    agent_state.behavior,
                    agent_state.joint_world,
                    agent_state.control,
                    masked_chi,
                )
                full_intended_probability = probabilities[intended_action]
                masked_intended_probability = masked.predicted_partner_probabilities[
                    intended_action
                ]
                nll_increase = -jnp.log(
                    jnp.clip(
                        masked_intended_probability,
                        _FLOAT_EPSILON,
                        1.0,
                    )
                ) + jnp.log(
                    jnp.clip(
                        full_intended_probability,
                        _FLOAT_EPSILON,
                        1.0,
                    )
                )
                other_action = 1 - intended_action
                full_margin = (
                    current_evaluation.planner_scores[intended_action]
                    - current_evaluation.planner_scores[other_action]
                )
                masked_margin = (
                    masked.planner_scores[intended_action] - masked.planner_scores[other_action]
                )
                return (
                    jnp.where(present, nll_increase, 0.0),
                    jnp.where(present, full_margin - masked_margin, 0.0),
                )

            c_masked_nll, c_masked_margin = critical_mask_effect(_CRITICAL_C_PAIR)
            d_masked_nll, d_masked_margin = critical_mask_effect(_CRITICAL_D_PAIR)
            trace = HiddenPartnerStepTrace(
                active=active,
                step=transition.oracle.step_count,
                segment_index=transition.oracle.segment_index,
                segment_step=transition.oracle.segment_step,
                segment_length=transition.oracle.segment_length,
                regime_id=transition.oracle.regime_id,
                reward=transition.reward,
                partner_action=partner_action,
                partner_intended_action=intended_action,
                focal_action=transition.focal_action,
                behavior_logits_preupdate=self.agent.behavior_model.predict_logits(
                    agent_state.behavior,
                    agent_state.chi,
                ),
                behavior_probabilities=probabilities,
                applied_partner_probabilities=(
                    update.diagnostics.current_evaluation.partner_probabilities
                ),
                behavior_nll=-jnp.log(
                    jnp.clip(
                        selected_probability,
                        _FLOAT_EPSILON,
                        1.0,
                    )
                ),
                behavior_brier=jnp.sum((probabilities - one_hot) ** 2),
                behavior_actual_correct=prediction == partner_action,
                behavior_intended_correct=prediction == intended_action,
                planner_greedy_action=planner_action,
                q_greedy_action=q_action,
                planner_intended_correct=planner_action == intended_action,
                executed_intended_correct=transition.focal_action == intended_action,
                realized_counterfactual_regret=(
                    jnp.max(transition.oracle.counterfactual_rewards) - transition.reward
                ),
                # Disagreeing with the intended sign costs E[r] = 0.95 - 0.05
                # = 0.9 in expectation under the 0.05 partner-flip rate.
                expected_greedy_regret=jnp.where(
                    planner_action == intended_action,
                    jnp.asarray(0.0, dtype=jnp.float32),
                    jnp.asarray(0.9, dtype=jnp.float32),
                ),
                model_intervened=planner_action != q_action,
                model_expected_reward_effect=(planner_expected_reward - q_expected_reward),
                world_reward_absolute_error=jnp.abs(update.diagnostics.world_reward_error),
                world_outcome_squared_error=jnp.mean(update.diagnostics.world_outcome_error**2),
                interaction_phi_pre=agent_state.phi,
                interaction_target=jnp.reshape(
                    2.0 * partner_action.astype(jnp.float32) - 1.0,
                    (1,),
                ),
                deployed_pair_features=agent_state.chi[BASE_FEATURE_DIM:],
                behavior_pair_weights_pre=(agent_state.behavior.weights[:, BASE_FEATURE_DIM:]),
                behavior_pair_weights_post=(
                    update.state.behavior.weights[:, BASE_FEATURE_DIM:]
                ),
                control_pair_weights_pre=(
                    agent_state.control.q_weights[:, BASE_FEATURE_DIM:]
                ),
                control_pair_weights_post=(
                    update.state.control.q_weights[:, BASE_FEATURE_DIM:]
                ),
                control_pair_trace_weights_pre=(
                    agent_state.control.q_trace_weights[:, BASE_FEATURE_DIM:]
                ),
                control_pair_trace_weights_post=(
                    update.state.control.q_trace_weights[:, BASE_FEATURE_DIM:]
                ),
                active_descriptors=agent_state.router.descriptors,
                deployed_descriptors_post=update.state.router.descriptors,
                shadow_descriptors_pre=jnp.stack(
                    (
                        agent_state.interaction.feature_left,
                        agent_state.interaction.feature_right,
                    ),
                    axis=1,
                ),
                shadow_descriptors_post=jnp.stack(
                    (
                        update.state.interaction.feature_left,
                        update.state.interaction.feature_right,
                    ),
                    axis=1,
                ),
                candidate_descriptors=jnp.stack(
                    (
                        agent_state.interaction.candidate_left,
                        agent_state.interaction.candidate_right,
                    ),
                    axis=1,
                ),
                candidate_descriptors_post=jnp.stack(
                    (
                        update.state.interaction.candidate_left,
                        update.state.interaction.candidate_right,
                    ),
                    axis=1,
                ),
                active_utilities=agent_state.interaction.utilities,
                shadow_utilities_post=update.state.interaction.utilities,
                shadow_idle_steps_pre=(agent_state.interaction.evidence_idle_steps),
                shadow_idle_steps_post=(update.state.interaction.evidence_idle_steps),
                candidate_utilities=agent_state.interaction.candidate_utilities,
                candidate_utilities_post=(update.state.interaction.candidate_utilities),
                candidate_output_weights_pre=(
                    agent_state.interaction.candidate_output_weights
                ),
                candidate_output_weights_post=(update.state.interaction.candidate_output_weights),
                candidate_ages_pre=agent_state.interaction.candidate_ages,
                candidate_ages_post=update.state.interaction.candidate_ages,
                interaction_feature_ages_pre=agent_state.interaction.ages,
                interaction_feature_ages_post=update.state.interaction.ages,
                interaction_replaced_slot=(update.diagnostics.interaction_replaced_slot),
                interaction_promoted_candidate=(update.diagnostics.interaction_promoted_candidate),
                interaction_retired_slot=(update.diagnostics.interaction_retired_slot),
                interaction_retired_left=(update.diagnostics.interaction_retired_left),
                interaction_retired_right=(update.diagnostics.interaction_retired_right),
                interaction_evidence_refreshed=(update.diagnostics.interaction_evidence_refreshed),
                interaction_retention_evidence_refreshed=(
                    update.diagnostics.interaction_retention_evidence_refreshed
                ),
                interaction_relevance_probe_scores=(
                    update.diagnostics.interaction_relevance_probe_scores
                ),
                interaction_relevance_probe_errors=(
                    update.diagnostics.interaction_relevance_probe_errors
                ),
                interaction_relevance_probe_weights_pre=(
                    update.diagnostics.interaction_relevance_probe_weights_pre
                ),
                interaction_relevance_probe_weights_post=(
                    update.diagnostics.interaction_relevance_probe_weights_post
                ),
                interaction_relevance_probe_biases_pre=(
                    update.diagnostics.interaction_relevance_probe_biases_pre
                ),
                interaction_relevance_probe_biases_post=(
                    update.diagnostics.interaction_relevance_probe_biases_post
                ),
                interaction_feature_second_moments_pre=(
                    agent_state.interaction.feature_second_moments
                ),
                interaction_feature_second_moments_post=(
                    update.state.interaction.feature_second_moments
                ),
                interaction_candidate_second_moments_pre=(
                    agent_state.interaction.candidate_second_moments
                ),
                interaction_candidate_second_moments_post=(
                    update.state.interaction.candidate_second_moments
                ),
                interaction_target_second_moments_pre=(
                    agent_state.interaction.target_second_moments
                ),
                interaction_target_second_moments_post=(
                    update.state.interaction.target_second_moments
                ),
                interaction_candidate_promotion_signal=(
                    update.diagnostics.interaction_candidate_promotion_signal
                ),
                interaction_candidate_promotion_raw_evidence=(
                    update.diagnostics.interaction_candidate_promotion_raw_evidence
                ),
                interaction_candidate_promotion_evidence_streak_pre=(
                    update.diagnostics.interaction_candidate_promotion_evidence_streak_pre
                ),
                interaction_candidate_promotion_evidence_streak_updated=(
                    update.diagnostics.interaction_candidate_promotion_evidence_streak_updated
                ),
                interaction_candidate_promotion_evidence_streak_post=(
                    update.diagnostics.interaction_candidate_promotion_evidence_streak_post
                ),
                interaction_candidate_promotion_confirmed=(
                    update.diagnostics.interaction_candidate_promotion_confirmed
                ),
                interaction_candidate_reacquisition_required_pre=(
                    update.diagnostics.interaction_candidate_reacquisition_required_pre
                ),
                interaction_candidate_reacquisition_required_post=(
                    update.diagnostics.interaction_candidate_reacquisition_required_post
                ),
                interaction_candidate_reacquisition_confirmed=(
                    update.diagnostics.interaction_candidate_reacquisition_confirmed
                ),
                interaction_durable_read_mask=(
                    update.diagnostics.interaction_durable_read_mask
                ),
                interaction_output_weights_pre=agent_state.interaction.output_weights,
                interaction_output_weights_post=update.state.interaction.output_weights,
                interaction_utility_evidence_streak_pre=(
                    agent_state.interaction.utility_evidence_streak
                ),
                interaction_utility_evidence_streak_post=(
                    update.state.interaction.utility_evidence_streak
                ),
                interaction_active_output_memory_committed_pre=(
                    agent_state.interaction.active_output_memory_committed
                ),
                interaction_active_output_memory_committed_post=(
                    update.state.interaction.active_output_memory_committed
                ),
                consumer_write_gate_pre=(update.diagnostics.consumer_write_gate_pre),
                consumer_read_idle_steps_pre=(
                    update.diagnostics.consumer_read_idle_steps_pre
                ),
                consumer_read_idle_steps_post=(
                    update.diagnostics.consumer_read_idle_steps_post
                ),
                consumer_active_mask_pre=(update.diagnostics.consumer_active_mask_pre),
                consumer_active_mask_post=(update.diagnostics.consumer_active_mask_post),
                interaction_matching_candidate_reset_mask=(
                    update.diagnostics.interaction_matching_candidate_reset_mask
                ),
                interaction_matching_candidate_reset_count=(
                    update.diagnostics.interaction_matching_candidate_reset_count
                ),
                interaction_live_feature_count=(update.diagnostics.interaction_live_feature_count),
                interaction_vacancy_count=(update.diagnostics.interaction_vacancy_count),
                interaction_promoted_into_vacancy=(
                    update.diagnostics.interaction_promoted_into_vacancy
                ),
                c_masked_behavior_nll_increase=c_masked_nll,
                d_masked_behavior_nll_increase=d_masked_nll,
                c_masked_planner_margin_decrease=c_masked_margin,
                d_masked_planner_margin_decrease=d_masked_margin,
                descriptors_changed=update.diagnostics.route.descriptors_changed,
                state_builder_step_delta=update.diagnostics.state_builder_step_delta,
                state_builder_learning_delta=(update.diagnostics.state_builder_learning_delta),
                behavior_step_delta=update.diagnostics.behavior_step_delta,
                interaction_step_delta=update.diagnostics.interaction_step_delta,
                world_step_delta=update.diagnostics.world_step_delta,
                control_step_delta=update.diagnostics.control_step_delta,
                router_route_delta=update.diagnostics.router_route_delta,
                integrated_step_delta=update.diagnostics.integrated_step_delta,
                route_valid=update.diagnostics.route.valid,
                causal_transition_valid=(
                    update.diagnostics.transition_observation_matches
                    & update.diagnostics.transition_action_matches
                    & update.diagnostics.transition_semantics_valid
                ),
                all_finite=(
                    update.diagnostics.all_finite
                    & jnp.isfinite(c_masked_nll)
                    & jnp.isfinite(d_masked_nll)
                    & jnp.isfinite(c_masked_margin)
                    & jnp.isfinite(d_masked_margin)
                ),
            )
            next_carry = (
                _select_carry(active, update.state, agent_state),
                _select_carry(active, proposed_environment, environment_state),
            )
            return next_carry, trace

        return jax.lax.scan(
            step,
            (initial_agent, initial_environment),
            xs=None,
            length=self.protocol.maximum_cycle_steps,
        )

    def run(self, seed_pair: HiddenPartnerSeedPair) -> HiddenPartnerRunResult:
        """Run exactly one jittered cycle and reconstruct its development metrics."""
        if not isinstance(seed_pair, HiddenPartnerSeedPair):
            raise TypeError("seed_pair must be a HiddenPartnerSeedPair")
        environment_state = self.world.init(jr.key(seed_pair.stream_seed))
        start = self.agent.start(
            self.world.observe(environment_state),
            jr.key(seed_pair.initialization_seed),
        )
        _BLOCK_UNTIL_READY((environment_state, start.state))
        cycle_length = environment_state.segment_ends[-1]
        initial_resource = self.agent.resource_budget(start.state)

        compilation_seconds = 0.0
        if self._compiled_life is None:
            compile_start = time.perf_counter()
            self._compiled_life = self._jitted_life.lower(
                start.state,
                environment_state,
                cycle_length,
            ).compile()
            compilation_seconds = time.perf_counter() - compile_start

        execution_start = time.perf_counter()
        (final_agent, final_environment), trace = self._compiled_life(
            start.state,
            environment_state,
            cycle_length,
        )
        _BLOCK_UNTIL_READY((final_agent, final_environment, trace))
        execution_seconds = time.perf_counter() - execution_start
        final_resource = self.agent.resource_budget(final_agent)
        summary = summarize_hidden_partner_run(
            self.protocol,
            self.condition,
            seed_pair,
            trace,
            np.asarray(environment_state.segment_lengths, dtype=np.int64),
            initial_resource,
            final_resource,
            compilation_wall_seconds=compilation_seconds,
            execution_wall_seconds=execution_seconds,
        )
        return HiddenPartnerRunResult(
            condition=self.condition,
            summary=summary,
            trace=trace,
            final_agent_state=final_agent,
            final_environment_state=final_environment,
            initial_resource=initial_resource,
            final_resource=final_resource,
        )


def _presence_numpy(descriptors: np.ndarray, pair: tuple[int, int]) -> np.ndarray:
    target = np.asarray(pair, dtype=np.int32)
    return np.any(np.all(descriptors == target, axis=-1), axis=-1)


def _first_true_index(values: np.ndarray, steps: np.ndarray) -> int | None:
    indices = np.flatnonzero(values)
    return int(steps[indices[0]]) if indices.size else None


def _eviction_count(presence: np.ndarray) -> int:
    if presence.size < 2:
        return 0
    return int(np.sum(presence[:-1] & ~presence[1:]))


def _recovery_steps(
    rewards: np.ndarray,
    window: int,
    threshold: float,
) -> int:
    if rewards.size < window:
        return int(rewards.size)
    kernel = np.full(window, 1.0 / window, dtype=np.float64)
    forward_means = np.convolve(rewards, kernel, mode="valid")
    hits = np.flatnonzero(forward_means >= threshold)
    if hits.size == 0:
        return int(rewards.size)
    return int(hits[0] + window)


def summarize_hidden_partner_run(
    protocol: HiddenPartnerDevelopmentProtocol,
    condition: HiddenPartnerCondition,
    seed_pair: HiddenPartnerSeedPair,
    trace: HiddenPartnerStepTrace,
    segment_lengths: np.ndarray,
    initial_resource: IntegratedHiddenPartnerResourceBudget,
    final_resource: IntegratedHiddenPartnerResourceBudget,
    *,
    compilation_wall_seconds: float,
    execution_wall_seconds: float,
) -> HiddenPartnerRunSummary:
    """Recompute every compact metric from fixed-shape primitive trace arrays."""
    active = np.asarray(trace.active, dtype=np.bool_)
    cycle_steps = int(np.sum(active))
    if cycle_steps <= 0:
        raise ValueError("development trace contains no active transitions")

    def values(field: str, dtype: Any = np.float64) -> np.ndarray:
        return np.asarray(getattr(trace, field), dtype=dtype)[active]

    steps = values("step", np.int64)
    segments = values("segment_index", np.int64)
    rewards = values("reward")
    nll = values("behavior_nll")
    brier = values("behavior_brier")
    actual_correct = values("behavior_actual_correct", np.bool_)
    intended_correct = values("behavior_intended_correct", np.bool_)
    planner_correct = values("planner_intended_correct", np.bool_)
    executed_correct = values("executed_intended_correct", np.bool_)
    realized_regret = values("realized_counterfactual_regret")
    expected_regret = values("expected_greedy_regret")
    intervened = values("model_intervened", np.bool_)
    model_effect = values("model_expected_reward_effect")

    window = protocol.early_late_window
    segment_summaries: list[HiddenPartnerSegmentSummary] = []
    prior_by_regime: dict[int, HiddenPartnerSegmentSummary] = {}
    for segment_index, regime_id in enumerate(DEFAULT_REGIME_SCHEDULE):
        mask = segments == segment_index
        segment_rewards = rewards[mask]
        segment_nll = nll[mask]
        segment_brier = brier[mask]
        segment_intended = intended_correct[mask]
        segment_realized_regret = realized_regret[mask]
        segment_expected_regret = expected_regret[mask]
        if segment_rewards.size != int(segment_lengths[segment_index]):
            raise ValueError(f"segment {segment_index} primitive count does not match its length")
        early_count = min(window, segment_rewards.size)
        late_count = min(window, segment_rewards.size)
        early_reward = float(np.mean(segment_rewards[:early_count]))
        late_reward = float(np.mean(segment_rewards[-late_count:]))
        prior = prior_by_regime.get(int(regime_id))
        retention_ratio: float | None = None
        retained: bool | None = None
        if prior is not None:
            denominator = max(prior.late_reward, _FLOAT_EPSILON)
            retention_ratio = early_reward / denominator
            retained = (
                retention_ratio >= protocol.retention_ratio_threshold
                and early_reward >= protocol.recurrent_early_reward_threshold
            )
        summary = HiddenPartnerSegmentSummary(
            segment_index=segment_index,
            regime_id=int(regime_id),
            regime_name=REGIME_NAMES[int(regime_id)],
            length=int(segment_rewards.size),
            mean_reward=float(np.mean(segment_rewards)),
            early_reward=early_reward,
            late_reward=late_reward,
            mean_behavior_nll=float(np.mean(segment_nll)),
            late_behavior_nll=float(np.mean(segment_nll[-late_count:])),
            mean_behavior_brier=float(np.mean(segment_brier)),
            intended_prediction_accuracy=float(np.mean(segment_intended)),
            late_intended_prediction_accuracy=float(np.mean(segment_intended[-late_count:])),
            mean_realized_regret=float(np.mean(segment_realized_regret)),
            mean_expected_greedy_regret=float(np.mean(segment_expected_regret)),
            recovery_steps=_recovery_steps(
                segment_rewards,
                protocol.recovery_window,
                protocol.recovery_reward_threshold,
            ),
            prior_same_regime_segment=(prior.segment_index if prior is not None else None),
            recurrent_early_to_prior_late_ratio=retention_ratio,
            recurrence_retained=retained,
        )
        segment_summaries.append(summary)
        prior_by_regime[int(regime_id)] = summary

    active_descriptors = np.asarray(trace.active_descriptors, dtype=np.int32)[active]
    candidate_descriptors = np.asarray(
        trace.candidate_descriptors,
        dtype=np.int32,
    )[active]
    c_active = _presence_numpy(active_descriptors, _CRITICAL_C_PAIR)
    d_active = _presence_numpy(active_descriptors, _CRITICAL_D_PAIR)
    c_candidate = _presence_numpy(candidate_descriptors, _CRITICAL_C_PAIR)
    d_candidate = _presence_numpy(candidate_descriptors, _CRITICAL_D_PAIR)
    # Milestone positions in the frozen v0 schedule A B A D A C A B C:
    # segment 5 is the first C, 8 the recurrent C, 3 the sole D.  Safe to
    # hard-code: HiddenPartnerMappingConfig rejects any other schedule.
    first_c_mask = segments == 5
    recurrent_c_mask = segments == 8
    d_mask = segments == 3
    first_c_indices = np.flatnonzero(first_c_mask)
    recurrent_c_indices = np.flatnonzero(recurrent_c_mask)
    d_indices = np.flatnonzero(d_mask)
    c_late_first = bool(c_active[first_c_indices[-min(window, first_c_indices.size) :]].all())
    c_recurrent_entry = bool(c_active[recurrent_c_indices[0]])
    d_end = bool(d_active[d_indices[-1]])

    expected_state_learning_delta = 1 if condition.config.state_learning_enabled else 0
    counter_contract_valid = bool(
        np.all(values("state_builder_step_delta", np.int64) == 1)
        and np.all(
            values("state_builder_learning_delta", np.int64) == expected_state_learning_delta
        )
        and np.all(values("behavior_step_delta", np.int64) == 1)
        and np.all(values("interaction_step_delta", np.int64) == 1)
        and np.all(values("world_step_delta", np.int64) == 1)
        and np.all(values("control_step_delta", np.int64) == 1)
        and np.all(values("router_route_delta", np.int64) == 1)
        and np.all(values("integrated_step_delta", np.int64) == 1)
    )
    causal_contract_valid = bool(
        np.all(values("route_valid", np.bool_))
        and np.all(values("causal_transition_valid", np.bool_))
    )
    finite = bool(
        np.all(values("all_finite", np.bool_))
        and all(
            np.all(np.isfinite(metric))
            for metric in (
                rewards,
                nll,
                brier,
                realized_regret,
                expected_regret,
                model_effect,
                values("world_reward_absolute_error"),
                values("world_outcome_squared_error"),
            )
        )
    )
    helpful = intervened & (model_effect > 0.0)
    harmful = intervened & (model_effect < 0.0)
    # Epsilon-greedy ceiling: matching the partner's intended sign earns
    # E[r] = 0.95 (the 0.05 flip rate caps it), exploration earns chance 0.5.
    # The normalized score maps chance -> 0 and this ceiling -> 1.
    perfect_policy_reward = (1.0 - condition.config.epsilon) * 0.95 + condition.config.epsilon * 0.5
    normalized_control = (float(np.mean(rewards)) - 0.5) / max(
        perfect_policy_reward - 0.5, _FLOAT_EPSILON
    )
    return HiddenPartnerRunSummary(
        condition=condition.name,
        seed_pair=seed_pair,
        cycle_steps=cycle_steps,
        segment_lengths=tuple(int(value) for value in segment_lengths),
        mean_reward=float(np.mean(rewards)),
        normalized_control_score=float(normalized_control),
        mean_behavior_nll=float(np.mean(nll)),
        mean_behavior_brier=float(np.mean(brier)),
        behavior_actual_accuracy=float(np.mean(actual_correct)),
        behavior_intended_accuracy=float(np.mean(intended_correct)),
        planner_intended_accuracy=float(np.mean(planner_correct)),
        executed_intended_accuracy=float(np.mean(executed_correct)),
        mean_realized_counterfactual_regret=float(np.mean(realized_regret)),
        mean_expected_greedy_regret=float(np.mean(expected_regret)),
        model_intervention_rate=float(np.mean(intervened)),
        helpful_model_intervention_rate=float(np.mean(helpful)),
        harmful_model_intervention_rate=float(np.mean(harmful)),
        mean_world_reward_absolute_error=float(np.mean(values("world_reward_absolute_error"))),
        mean_world_outcome_squared_error=float(np.mean(values("world_outcome_squared_error"))),
        descriptor_transaction_count=int(np.sum(values("descriptors_changed", np.bool_))),
        counter_contract_valid=counter_contract_valid,
        causal_contract_valid=causal_contract_valid,
        all_finite=finite,
        initial_state_nbytes=initial_resource.total_state_nbytes,
        final_state_nbytes=final_resource.total_state_nbytes,
        resource_shape_matched=(
            initial_resource.total_state_nbytes == final_resource.total_state_nbytes
        ),
        compilation_wall_seconds=float(compilation_wall_seconds),
        execution_wall_seconds=float(execution_wall_seconds),
        mean_execution_microseconds_per_step=float(execution_wall_seconds * 1e6 / cycle_steps),
        segments=tuple(segment_summaries),
        features=HiddenPartnerFeatureSummary(
            c_first_active_step=_first_true_index(c_active, steps),
            d_first_active_step=_first_true_index(d_active, steps),
            c_active_evictions=_eviction_count(c_active),
            d_active_evictions=_eviction_count(d_active),
            c_active_late_first_c=c_late_first,
            c_active_at_recurrent_c_entry=c_recurrent_entry,
            c_survived_first_to_recurrent_c=(c_late_first and c_recurrent_entry),
            d_active_at_end_of_d=d_end,
            d_active_at_life_end=bool(d_active[-1]),
            c_candidate_fraction=float(np.mean(c_candidate)),
            d_candidate_fraction=float(np.mean(d_candidate)),
        ),
    )


def run_hidden_partner_development_suite(
    seed_pairs: Sequence[HiddenPartnerSeedPair],
    *,
    protocol: HiddenPartnerDevelopmentProtocol | None = None,
    condition_names: Sequence[ConditionName] = tuple(_CONDITIONS_BY_NAME),
) -> tuple[HiddenPartnerRunSummary, ...]:
    """Run paired development lives while reusing one compiled runner per arm."""
    resolved_protocol = HiddenPartnerDevelopmentProtocol() if protocol is None else protocol
    if not seed_pairs:
        raise ValueError("at least one development seed pair is required")
    if not condition_names:
        raise ValueError("at least one condition is required")
    runners = [HiddenPartnerDevelopmentRunner(name, resolved_protocol) for name in condition_names]
    summaries: list[HiddenPartnerRunSummary] = []
    for seed_pair in seed_pairs:
        for runner in runners:
            summaries.append(runner.run(seed_pair).summary)

    grouped_by_seed: dict[tuple[str, int], list[HiddenPartnerRunSummary]] = {}
    for summary in summaries:
        key = (summary.seed_pair.namespace, summary.seed_pair.index)
        grouped_by_seed.setdefault(key, []).append(summary)
    for seed_summaries in grouped_by_seed.values():
        state_bytes = {summary.initial_state_nbytes for summary in seed_summaries}
        steps = {summary.cycle_steps for summary in seed_summaries}
        lengths = {summary.segment_lengths for summary in seed_summaries}
        if len(state_bytes) != 1:
            raise RuntimeError("paired conditions do not have equal state bytes")
        if len(steps) != 1 or len(lengths) != 1:
            raise RuntimeError("paired conditions did not consume the same environment life")
    return tuple(summaries)


def hidden_partner_development_record(
    protocol: HiddenPartnerDevelopmentProtocol,
    summaries: Sequence[HiddenPartnerRunSummary],
) -> dict[str, object]:
    """Build a nonpromoting JSON-compatible development record."""
    if not summaries:
        raise ValueError("at least one summary is required")
    return {
        "schema_version": HIDDEN_PARTNER_DEVELOPMENT_SCHEMA_VERSION,
        "development_only": True,
        "scientific_promotion_allowed": False,
        "protocol": protocol.to_config(),
        "seed_roles": {
            "tuning_namespace": TUNING_SEED_NAMESPACE,
            "robustness_namespace": ROBUSTNESS_SEED_NAMESPACE,
            "consumed_if_run": True,
            "promoted_seed_role": "none",
        },
        "scope": (
            "scripted-partner v0 development diagnostic; no coadaptation, "
            "general continual-learning, or L3 claim"
        ),
        "runs": [summary.to_dict() for summary in summaries],
    }


def _development_bootstrap_mean_interval(
    values: np.ndarray,
    *,
    seed: int,
    replicates: int = 10_000,
) -> dict[str, float | int | str]:
    """Deterministic percentile interval for explicitly development-only effects."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < 2 or not np.all(np.isfinite(array)):
        raise ValueError("bootstrap values must be a finite vector of at least two")
    generator = np.random.default_rng(seed)
    indices = generator.integers(
        0,
        array.size,
        size=(replicates, array.size),
    )
    bootstrap = np.mean(array[indices], axis=1)
    return {
        "estimate": float(np.mean(array)),
        "lower": float(np.quantile(bootstrap, 0.025)),
        "upper": float(np.quantile(bootstrap, 0.975)),
        "confidence_level": 0.95,
        "method": "paired-seed-percentile-bootstrap",
        "replicates": replicates,
        "sample_size": int(array.size),
    }


def aggregate_hidden_partner_development(
    summaries: Sequence[HiddenPartnerRunSummary],
) -> dict[str, object]:
    """Aggregate paired development summaries without creating a promoted claim."""
    if not summaries:
        raise ValueError("at least one summary is required")
    conditions = tuple(condition.name for condition in HIDDEN_PARTNER_CONDITIONS)
    grouped: dict[str, dict[tuple[str, int], HiddenPartnerRunSummary]] = {
        name: {} for name in conditions
    }
    for summary in summaries:
        if summary.condition not in grouped:
            raise ValueError(f"unexpected condition in summaries: {summary.condition!r}")
        key = (summary.seed_pair.namespace, summary.seed_pair.index)
        if key in grouped[summary.condition]:
            raise ValueError(f"duplicate condition/seed summary: {summary.condition!r}, {key!r}")
        grouped[summary.condition][key] = summary
    seed_keys = tuple(sorted(grouped["full"]))
    if len(seed_keys) < 2:
        raise ValueError("development aggregation requires at least two paired seeds")
    for name, records_by_seed in grouped.items():
        if tuple(sorted(records_by_seed)) != seed_keys:
            raise ValueError(f"condition {name!r} does not contain the exact paired seeds")

    condition_records: dict[str, object] = {}
    full_rewards = np.asarray(
        [grouped["full"][key].mean_reward for key in seed_keys],
        dtype=np.float64,
    )
    for name in conditions:
        records = [grouped[name][key] for key in seed_keys]
        rewards = np.asarray(
            [record.mean_reward for record in records],
            dtype=np.float64,
        )
        condition_records[name] = {
            "seed_count": len(records),
            "mean_reward": float(np.mean(rewards)),
            "minimum_seed_reward": float(np.min(rewards)),
            "reward_standard_deviation": float(np.std(rewards, ddof=1)),
            "mean_normalized_control_score": float(
                np.mean([record.normalized_control_score for record in records])
            ),
            "mean_behavior_nll": float(np.mean([record.mean_behavior_nll for record in records])),
            "mean_behavior_brier": float(
                np.mean([record.mean_behavior_brier for record in records])
            ),
            "mean_planner_intended_accuracy": float(
                np.mean([record.planner_intended_accuracy for record in records])
            ),
            "finite_life_count": int(sum(record.all_finite for record in records)),
            "contract_valid_life_count": int(
                sum(
                    record.counter_contract_valid
                    and record.causal_contract_valid
                    and record.resource_shape_matched
                    for record in records
                )
            ),
            "c_survival_count": int(
                sum(record.features.c_survived_first_to_recurrent_c for record in records)
            ),
            "d_learned_by_end_of_d_count": int(
                sum(record.features.d_active_at_end_of_d for record in records)
            ),
            "d_absent_at_life_end_count": int(
                sum(not record.features.d_active_at_life_end for record in records)
            ),
            "mean_descriptor_transactions": float(
                np.mean([record.descriptor_transaction_count for record in records])
            ),
            "state_nbytes": records[0].initial_state_nbytes,
            "replay_capacity": 0,
            "rewards_by_seed": [float(value) for value in rewards],
        }

    effects: dict[str, object] = {}
    for arm_index, name in enumerate(conditions[1:], start=1):
        arm_rewards = np.asarray(
            [grouped[name][key].mean_reward for key in seed_keys],
            dtype=np.float64,
        )
        differences = full_rewards - arm_rewards
        effects[f"full_minus_{name}_mean_reward"] = {
            "interval": _development_bootstrap_mean_interval(
                differences,
                seed=0x48504400 + arm_index,
            ),
            "full_win_count": int(np.sum(differences > 0.0)),
            "tie_count": int(np.sum(differences == 0.0)),
            "arm_win_count": int(np.sum(differences < 0.0)),
            "paired_differences": [float(value) for value in differences],
        }

    full_records = [grouped["full"][key] for key in seed_keys]
    full_record = cast(dict[str, object], condition_records["full"])
    full_mean_reward = cast(float, full_record["mean_reward"])
    full_minimum_seed_reward = cast(float, full_record["minimum_seed_reward"])
    full_c_survival_count = cast(int, full_record["c_survival_count"])
    full_d_learning_count = cast(int, full_record["d_learned_by_end_of_d_count"])
    full_d_forgetting_count = cast(int, full_record["d_absent_at_life_end_count"])
    checks = {
        "all_full_lives_finite_and_contract_valid": (
            full_record["finite_life_count"] == len(seed_keys)
            and full_record["contract_valid_life_count"] == len(seed_keys)
        ),
        "full_mean_reward_at_least_0_85": full_mean_reward >= 0.85,
        "full_minimum_seed_reward_at_least_0_80": full_minimum_seed_reward >= 0.80,
        "full_c_survival_fraction_at_least_0_75": (full_c_survival_count / len(seed_keys) >= 0.75),
        "full_d_learning_fraction_at_least_0_75": (full_d_learning_count / len(seed_keys) >= 0.75),
        "full_d_forgetting_fraction_at_least_0_75": (
            full_d_forgetting_count / len(seed_keys) >= 0.75
        ),
        "full_beats_lifecycle_frozen_on_every_seed": (
            cast(
                dict[str, object],
                effects["full_minus_lifecycle_frozen_mean_reward"],
            )["full_win_count"]
            == len(seed_keys)
        ),
        "full_beats_uniform_partner_on_every_seed": (
            cast(
                dict[str, object],
                effects["full_minus_uniform_partner_mean_reward"],
            )["full_win_count"]
            == len(seed_keys)
        ),
        "full_beats_random_curation_on_every_seed": (
            cast(
                dict[str, object],
                effects["full_minus_random_curation_mean_reward"],
            )["full_win_count"]
            == len(seed_keys)
        ),
    }
    return {
        "development_only": True,
        "scientific_promotion_allowed": False,
        "seed_count": len(seed_keys),
        "conditions": condition_records,
        "paired_effects": effects,
        "development_checks": {
            **checks,
            "all_passed": all(checks.values()),
            "interpretation": (
                "falsification checks for v0 development only; failure guides "
                "mechanism work and passage cannot promote a scientific claim"
            ),
        },
        "known_scope_limits": [
            "scripted nonlearning partner",
            "fixed schedule family",
            "exhaustive closed 66-pair candidate archive",
            "no dormant downstream-weight archive after eviction",
            "eight development robustness lives are not a promoted sample",
        ],
        "full_recurrent_segment_retention": [
            {
                "seed_index": record.seed_pair.index,
                "segments": [
                    {
                        "segment_index": segment.segment_index,
                        "regime": segment.regime_name,
                        "retained": segment.recurrence_retained,
                        "recovery_steps": segment.recovery_steps,
                    }
                    for segment in record.segments
                    if segment.prior_same_regime_segment is not None
                ],
            }
            for record in full_records
        ],
    }


__all__ = [
    "DEFAULT_DEVELOPMENT_SEED_COUNT",
    "DEVELOPMENT_ONLY",
    "HIDDEN_PARTNER_CONDITIONS",
    "HIDDEN_PARTNER_DEVELOPMENT_SCHEMA_VERSION",
    "HiddenPartnerCondition",
    "HiddenPartnerDevelopmentProtocol",
    "HiddenPartnerDevelopmentRunner",
    "HiddenPartnerFeatureSummary",
    "HiddenPartnerRunResult",
    "HiddenPartnerRunSummary",
    "HiddenPartnerSeedPair",
    "HiddenPartnerSegmentSummary",
    "HiddenPartnerStepTrace",
    "LearnerHiddenPartnerTransition",
    "ROBUSTNESS_SEED_NAMESPACE",
    "SCIENTIFIC_PROMOTION_ALLOWED",
    "TUNING_SEED_NAMESPACE",
    "derive_hidden_partner_seed_pairs",
    "aggregate_hidden_partner_development",
    "hidden_partner_development_record",
    "hidden_partner_run_summary_from_dict",
    "run_hidden_partner_development_suite",
    "strip_hidden_partner_oracle",
    "summarize_hidden_partner_run",
]
