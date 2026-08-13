"""Development-only hidden-partner observation-horizon falsification.

The target-only selective-forgetting microcycle used 256-step regimes.  Its
fixed development seed acquired the recurring C-critical pair late enough to
leave only about 191 updates before C's first-exposure segment ended, and the
unchanged lifecycle-v2 C learning gate rejected the run.  This independent
successor spends one new namespaced development seed on one isolated question:
does doubling every regime to 512 steps provide a sufficient online
observation horizon for the same target-only learner?

There is exactly one arm.  All learner settings and lifecycle-v2 gates remain
unchanged except that the time-based active-retention grace is doubled from
640 to 1,280 steps along with C's absence gap.  That geometric adjustment
prevents a longer absence from turning this into a retention-grace test.  A
single new seed cannot support a between-run causal claim; it only records a
fixed, contract-valid development falsification of horizon sufficiency.

This module has no artifact writer, output path, threshold selection,
seed-search API, or scientific-promotion entry point.  A valid miss remains a
valid development rejection.
"""

from __future__ import annotations

import dataclasses
from typing import Literal

import numpy as np

from alberta_framework.core.integrated_hidden_partner import (
    IntegratedHiddenPartnerConfig,
)
from alberta_framework.core.interaction_features import (
    RELEVANCE_PROBE_MODE_TARGET_ONLY_V1,
)
from alberta_framework.evaluation.hidden_partner_development import (
    HiddenPartnerCondition,
    HiddenPartnerDevelopmentProtocol,
    HiddenPartnerDevelopmentRunner,
    HiddenPartnerRunResult,
    HiddenPartnerSeedPair,
    derive_hidden_partner_seed_pairs,
)
from alberta_framework.evaluation.hidden_partner_lifecycle_v2 import (
    CHANCE_REWARD,
    CONFIRMATION_NAMESPACE,
    CRITICAL_COLUMN_LEARNING_NLL_GAIN_THRESHOLD,
    CRITICAL_COLUMN_LEARNING_POSITIVE_FRACTION_THRESHOLD,
    CRITICAL_COLUMN_TARGET_CREATED_SHARE_THRESHOLD,
    CRITICAL_LATE_PREDICTION_ACCURACY_THRESHOLD,
    CRITICAL_MASKED_NLL_INCREASE_THRESHOLD,
    CRITICAL_MASKED_NLL_POSITIVE_FRACTION_THRESHOLD,
    FEATURE_LEARNING_WINDOW,
    FINAL_ABSENCE_WINDOW,
    INITIAL_LATE_REWARD_THRESHOLD,
    LEASE_TUNING_NAMESPACE,
    RECURRENT_EARLY_REWARD_THRESHOLD,
    RECURRENT_ENTRY_WINDOW,
    RETENTION_RATIO_THRESHOLD,
    RETIREMENT_CONFIRMATION_WINDOW,
    CriticalLifecycleV2Summary,
    summarize_critical_lifecycle_v2,
)
from alberta_framework.streams.hidden_partner_mapping import (
    DEFAULT_REGIME_SCHEDULE,
    HiddenPartnerMappingConfig,
)

HORIZON_SUFFICIENCY_DEVELOPMENT_SCHEMA = (
    "alberta.hidden-partner-horizon-sufficiency-development.falsification.v1"
)
DEVELOPMENT_ONLY = True
SCIENTIFIC_PROMOTION_ALLOWED = False
OUTPUT_WRITES_ALLOWED = False

HORIZON_SUFFICIENCY_NAMESPACE = (
    "hidden-partner-v0-dev-target-only-horizon-sufficiency-v1"
)
HORIZON_SUFFICIENCY_SEED = HiddenPartnerSeedPair(
    namespace=HORIZON_SUFFICIENCY_NAMESPACE,
    index=0,
    stream_seed=2_097_892_768,
    initialization_seed=3_606_366_503,
)

PREDECESSOR_SEGMENT_LENGTH = 256
PREDECESSOR_RETENTION_GRACE_STEPS = 640
SEGMENT_LENGTH = 512
SEGMENT_LENGTHS = (SEGMENT_LENGTH,) * 9
CYCLE_STEPS = sum(SEGMENT_LENGTHS)
RETENTION_GRACE_STEPS = 1_280
C_FIRST_EXPOSURE_START_STEP = 5 * SEGMENT_LENGTH
C_FIRST_EXPOSURE_END_STEP = 6 * SEGMENT_LENGTH
C_ABSENCE_GAP_STEPS = 2 * SEGMENT_LENGTH

HORIZON_SUFFICIENCY_PROTOCOL = HiddenPartnerDevelopmentProtocol(
    environment=HiddenPartnerMappingConfig(
        base_segment_lengths=SEGMENT_LENGTHS,
        jitter_radius=0,
        partner_flip_probability=0.05,
    ),
    recovery_window=128,
    early_late_window=128,
)

# These are the lifecycle-v2 authorities consumed by
# ``summarize_critical_lifecycle_v2``.  Pinning them here makes source-level
# threshold or window drift invalidate the development run instead of silently
# changing the question after the one seed has been consumed.
HORIZON_LIFECYCLE_GATE_CONTRACT: dict[str, float | int] = {
    "feature_learning_window": 128,
    "retirement_confirmation_window": 128,
    "final_absence_window": 256,
    "recurrent_entry_window": 128,
    "critical_late_prediction_accuracy_threshold": 0.80,
    "critical_column_learning_nll_gain_threshold": 0.05,
    "critical_column_learning_positive_fraction_threshold": 0.55,
    "critical_column_target_created_share_threshold": 0.50,
    "critical_masked_nll_increase_threshold": 0.005,
    "critical_masked_nll_positive_fraction_threshold": 0.55,
    "recurrent_early_reward_threshold": 0.75,
    "initial_late_reward_threshold": 0.75,
    "retention_ratio_threshold": 0.80,
    "chance_reward": 0.50,
}

# Exact logical resource contract.  The horizon and grace change values, not
# shapes, so this remains identical to the current target-only mechanism.
HORIZON_SUFFICIENCY_RESOURCE_CONTRACT: dict[str, int] = {
    "raw_observation_dim": 8,
    "base_feature_dim": 12,
    "active_pair_slots": 12,
    "candidate_pair_slots": 66,
    "deployed_feature_dim": 24,
    "state_builder_nbytes": 2_124,
    "interaction_nbytes": 3_342,
    "interaction_evidence_idle_nbytes": 48,
    "interaction_utility_evidence_streak_nbytes": 48,
    "interaction_active_output_memory_committed_nbytes": 12,
    "interaction_relevance_probe_nbytes": 52,
    "interaction_relevance_probe_bias_nbytes": 4,
    "interaction_candidate_promotion_evidence_streak_nbytes": 264,
    "interaction_candidate_reacquisition_required_nbytes": 66,
    "behavior_nbytes": 232,
    "joint_world_nbytes": 60,
    "grounded_world_nbytes": 0,
    "grounded_world_parameter_count": 0,
    "grounded_world_parameters_touched_per_update": 0,
    "grounded_world_update_counter_nbytes": 0,
    "control_nbytes": 536,
    "router_nbytes": 120,
    "consumer_active_mask_nbytes": 12,
    "consumer_evidence_streak_nbytes": 48,
    "consumer_read_idle_steps_nbytes": 48,
    "decision_cache_nbytes": 311,
    "integrated_transition_counter_nbytes": 12,
    "total_state_nbytes": 6_833,
    "legacy_joint_world_cells_per_decision": 4,
    "grounded_world_joint_cells_per_decision": 0,
    "planner_cell_evaluations_per_decision": 4,
    "replay_capacity": 0,
}

# Full serialized learner authority.  Defaults are included intentionally:
# changing an unrelated default must invalidate this consumed-seed lane rather
# than silently creating a second intervention.
HORIZON_AGENT_CONFIG_CONTRACT: dict[str, object] = {
    "type": "IntegratedHiddenPartnerConfig",
    "schema_version": "alberta.integrated-hidden-partner.l0.v16",
    "development_level": "L0",
    "accepted_scientific_evidence": False,
    "planning_enabled": True,
    "state_learning_enabled": True,
    "feature_lifecycle_enabled": True,
    "carry_survivors": True,
    "memory_masked": False,
    "uniform_partner_belief": False,
    "random_feature_curation": False,
    "action_selection_mode": "agent",
    "evidence_gated_feature_memory": True,
    "feature_evidence_confirmation_steps": 24,
    "independent_relevance_probe": True,
    "relevance_probe_mode": "target_only_v1",
    "evidence_gated_consumer_memory": True,
    "consumer_evidence_confirmation_steps": 12,
    "consumer_read_confirmation_steps": 4,
    "consumer_read_lease_steps": 4,
    "initial_active_descriptors": [
        [0, 6],
        [0, 8],
        [1, 4],
        [1, 7],
        [2, 5],
        [2, 9],
        [3, 6],
        [3, 10],
        [4, 7],
        [5, 6],
        [7, 11],
        [8, 10],
    ],
    "grounded_world_model": None,
    "representation_gradient_mixer": None,
    "grounded_world_learning_enabled": True,
    "grounded_world_planning_enabled": False,
    "planner_lambda": 2.0,
    "state_step_size": 0.005,
    "state_gradient_clip": 5.0,
    "interaction_step_size": 0.03,
    "interaction_utility_decay": 0.995,
    "active_utility_retention_decay": 0.9999,
    "active_utility_retention_grace_steps": 1_280,
    "active_utility_evidence_threshold": 0.1,
    "retire_stale_features": True,
    "candidate_promotion_floor": 0.1,
    "candidate_promotion_confirmation_steps": 1,
    "candidate_reacquisition_confirmation_steps": 8,
    "replacement_interval": 64,
    "min_feature_age": 256,
    "candidate_min_age": 128,
    "candidate_utility_retention_decay": 0.9995,
    "behavior_step_size": 0.05,
    "world_step_size": 0.25,
    "q_step_size": 0.03,
    "average_reward_step_size": 0.003,
    "trace_decay": 0.0,
    "epsilon": 0.1,
}


def _horizon_agent_config() -> IntegratedHiddenPartnerConfig:
    """Build the fixed target-only learner with only geometric grace scaling."""
    return IntegratedHiddenPartnerConfig(
        active_utility_retention_decay=0.9999,
        active_utility_retention_grace_steps=RETENTION_GRACE_STEPS,
        active_utility_evidence_threshold=0.10,
        retire_stale_features=True,
        candidate_promotion_floor=0.10,
        evidence_gated_feature_memory=True,
        feature_evidence_confirmation_steps=24,
        independent_relevance_probe=True,
        relevance_probe_mode=RELEVANCE_PROBE_MODE_TARGET_ONLY_V1,
        candidate_promotion_confirmation_steps=1,
        candidate_reacquisition_confirmation_steps=8,
        evidence_gated_consumer_memory=True,
        consumer_evidence_confirmation_steps=12,
        consumer_read_confirmation_steps=4,
        consumer_read_lease_steps=4,
        replacement_interval=64,
        min_feature_age=256,
        candidate_min_age=128,
    )


def _predecessor_agent_config() -> IntegratedHiddenPartnerConfig:
    """Build the config authority used solely for the one-field diff audit."""
    return dataclasses.replace(
        _horizon_agent_config(),
        active_utility_retention_grace_steps=PREDECESSOR_RETENTION_GRACE_STEPS,
    )


@dataclasses.dataclass(frozen=True)
class HorizonSufficiencyRunValidation:
    """Independent static, resource, trace, and lifecycle validity verdict."""

    valid: bool
    config_contract_valid: bool
    resource_contract_valid: bool
    trace_contract_valid: bool
    lifecycle_contract_valid: bool
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)


HorizonStatus = Literal[
    "passed_horizon_sufficiency_check",
    "valid_development_rejection",
    "invalid_development_run",
]


@dataclasses.dataclass(frozen=True)
class HorizonSufficiencyDevelopmentResult:
    """One fixed run with validity kept separate from empirical outcome."""

    schema_version: str
    development_only: bool
    scientific_promotion_allowed: bool
    output_writes_allowed: bool
    status: HorizonStatus
    lifecycle_requirement_failures: tuple[str, ...]
    c_horizon_requirement_failures: tuple[str, ...]
    c_post_acquisition_observation_steps: int | None
    run: HiddenPartnerRunResult
    lifecycle: CriticalLifecycleV2Summary
    validation: HorizonSufficiencyRunValidation

    def to_report(self) -> dict[str, object]:
        """Return a compact in-memory report with no evidence authority."""
        lifecycle = self.lifecycle
        return {
            "schema_version": self.schema_version,
            "development_only": self.development_only,
            "scientific_promotion_allowed": self.scientific_promotion_allowed,
            "output_writes_allowed": self.output_writes_allowed,
            "status": self.status,
            "lifecycle_requirement_failures": list(
                self.lifecycle_requirement_failures
            ),
            "c_horizon_requirement_failures": list(
                self.c_horizon_requirement_failures
            ),
            "c_post_acquisition_observation_steps": (
                self.c_post_acquisition_observation_steps
            ),
            "seed": HORIZON_SUFFICIENCY_SEED.to_dict(),
            "protocol": {
                "segment_lengths": list(SEGMENT_LENGTHS),
                "jitter_radius": 0,
                "partner_flip_probability": 0.05,
                "retention_grace_steps": RETENTION_GRACE_STEPS,
            },
            "validation": self.validation.to_dict(),
            "outcomes": {
                "mean_reward": self.run.summary.mean_reward,
                "c_promotion_event_steps": list(lifecycle.c_promotion_event_steps),
                "c_acquisition_step": lifecycle.c_acquisition_step,
                "c_first_late_reward": lifecycle.c_first_late_reward,
                "c_first_late_intended_accuracy": (
                    lifecycle.c_first_late_intended_accuracy
                ),
                "c_critical_column_learning_nll_gain": (
                    lifecycle.c_critical_column_learning_nll_gain
                ),
                "c_critical_column_learning_positive_fraction": (
                    lifecycle.c_critical_column_learning_positive_fraction
                ),
                "c_task_learned": lifecycle.c_task_learned,
                "c_continuously_survived": lifecycle.c_continuously_survived,
                "c_recurrent_early_reward": lifecycle.c_recurrent_early_reward,
                "c_retained_and_used": lifecycle.c_retained_and_used,
                "d_acquisition_step": lifecycle.d_acquisition_step,
                "d_task_learned": lifecycle.d_task_learned,
                "d_retirement_event_steps": list(
                    lifecycle.d_retirement_event_steps
                ),
                "d_learned_then_stably_retired": (
                    lifecycle.d_learned_then_stably_retired
                ),
                "joint_memory_management_success": (
                    lifecycle.joint_memory_management_success
                ),
            },
            "interpretation": (
                "one-seed development-only horizon falsification; no between-seed "
                "causal inference and no scientific promotion"
            ),
        }


def _live_lifecycle_gate_contract() -> dict[str, float | int]:
    return {
        "feature_learning_window": FEATURE_LEARNING_WINDOW,
        "retirement_confirmation_window": RETIREMENT_CONFIRMATION_WINDOW,
        "final_absence_window": FINAL_ABSENCE_WINDOW,
        "recurrent_entry_window": RECURRENT_ENTRY_WINDOW,
        "critical_late_prediction_accuracy_threshold": (
            CRITICAL_LATE_PREDICTION_ACCURACY_THRESHOLD
        ),
        "critical_column_learning_nll_gain_threshold": (
            CRITICAL_COLUMN_LEARNING_NLL_GAIN_THRESHOLD
        ),
        "critical_column_learning_positive_fraction_threshold": (
            CRITICAL_COLUMN_LEARNING_POSITIVE_FRACTION_THRESHOLD
        ),
        "critical_column_target_created_share_threshold": (
            CRITICAL_COLUMN_TARGET_CREATED_SHARE_THRESHOLD
        ),
        "critical_masked_nll_increase_threshold": (
            CRITICAL_MASKED_NLL_INCREASE_THRESHOLD
        ),
        "critical_masked_nll_positive_fraction_threshold": (
            CRITICAL_MASKED_NLL_POSITIVE_FRACTION_THRESHOLD
        ),
        "recurrent_early_reward_threshold": RECURRENT_EARLY_REWARD_THRESHOLD,
        "initial_late_reward_threshold": INITIAL_LATE_REWARD_THRESHOLD,
        "retention_ratio_threshold": RETENTION_RATIO_THRESHOLD,
        "chance_reward": CHANCE_REWARD,
    }


def validate_horizon_sufficiency_static_contract() -> tuple[str, ...]:
    """Fail closed on the fixed seed, mechanism, horizon, and gate contract."""
    errors: list[str] = []
    forbidden_namespaces = {
        LEASE_TUNING_NAMESPACE,
        CONFIRMATION_NAMESPACE,
        "hidden-partner-v0-dev-target-only-selective-forgetting-microcycle-v1",
        "hidden-partner-v0-dev-retirement-throughput-falsification-v1",
    }
    if HORIZON_SUFFICIENCY_NAMESPACE in forbidden_namespaces:
        errors.append("horizon namespace collides with a consumed namespace")
    expected_seed = derive_hidden_partner_seed_pairs(
        HORIZON_SUFFICIENCY_NAMESPACE,
        1,
    )[0]
    if expected_seed != HORIZON_SUFFICIENCY_SEED:
        errors.append("fixed development seed differs from its namespace derivation")
    if (DEVELOPMENT_ONLY, SCIENTIFIC_PROMOTION_ALLOWED, OUTPUT_WRITES_ALLOWED) != (
        True,
        False,
        False,
    ):
        errors.append("development-only nonpromotion contract changed")

    protocol = HORIZON_SUFFICIENCY_PROTOCOL
    if protocol.environment.base_segment_lengths != SEGMENT_LENGTHS:
        errors.append("segment lengths changed")
    if protocol.environment.jitter_radius != 0:
        errors.append("horizon run must remain jitterless")
    if protocol.environment.partner_flip_probability != 0.05:
        errors.append("partner flip probability changed")
    if protocol.maximum_cycle_steps != CYCLE_STEPS:
        errors.append("exact scan length changed")
    if protocol.recovery_window != 128 or protocol.early_late_window != 128:
        errors.append("development summary windows changed")
    if (
        protocol.recovery_reward_threshold != 0.80
        or protocol.retention_ratio_threshold != 0.90
        or protocol.recurrent_early_reward_threshold != 0.75
    ):
        errors.append("development summary thresholds changed")
    if SEGMENT_LENGTH != 2 * PREDECESSOR_SEGMENT_LENGTH:
        errors.append("observation horizon is no longer exactly doubled")
    if RETENTION_GRACE_STEPS != 2 * PREDECESSOR_RETENTION_GRACE_STEPS:
        errors.append("retention grace is no longer geometrically doubled")
    if C_ABSENCE_GAP_STEPS != 2 * 2 * PREDECESSOR_SEGMENT_LENGTH:
        errors.append("C absence-gap geometry changed")
    if _live_lifecycle_gate_contract() != HORIZON_LIFECYCLE_GATE_CONTRACT:
        errors.append("lifecycle-v2 gate or measurement-window contract changed")

    config = _horizon_agent_config()
    predecessor = _predecessor_agent_config()
    payload = config.to_config()
    predecessor_payload = predecessor.to_config()
    if payload != HORIZON_AGENT_CONFIG_CONTRACT:
        errors.append("serialized horizon agent config changed")
    expected_predecessor_payload = dict(HORIZON_AGENT_CONFIG_CONTRACT)
    expected_predecessor_payload["active_utility_retention_grace_steps"] = (
        PREDECESSOR_RETENTION_GRACE_STEPS
    )
    if predecessor_payload != expected_predecessor_payload:
        errors.append("serialized predecessor config authority changed")
    if set(payload) != set(predecessor_payload):
        errors.append("agent config field set changed across the horizon contrast")
    differing = {
        name
        for name in payload
        if payload.get(name) != predecessor_payload.get(name)
    }
    if differing != {"active_utility_retention_grace_steps"}:
        errors.append("agent differs from the predecessor outside retention grace")
    if config.relevance_probe_mode != RELEVANCE_PROBE_MODE_TARGET_ONLY_V1:
        errors.append("relevance probe is not target-only")
    if not config.evidence_gated_feature_memory:
        errors.append("feature memory gate is disabled")
    if not config.evidence_gated_consumer_memory:
        errors.append("consumer memory gate is disabled")
    if not config.independent_relevance_probe:
        errors.append("independent relevance probe is disabled")
    if not config.retire_stale_features:
        errors.append("explicit stale retirement is disabled")
    if config.active_utility_retention_grace_steps != RETENTION_GRACE_STEPS:
        errors.append("live retention grace differs from the fixed geometry")
    return tuple(errors)


def _trace_array(
    errors: list[str],
    result: HiddenPartnerRunResult,
    field: str,
    dtype: np.dtype[np.generic],
    tail_shape: tuple[int, ...],
) -> np.ndarray | None:
    raw = np.asarray(getattr(result.trace, field))
    expected_shape = (CYCLE_STEPS, *tail_shape)
    if raw.dtype != dtype or raw.shape != expected_shape:
        errors.append(
            f"trace.{field} must have dtype {dtype} and shape {expected_shape}, "
            f"got {raw.dtype} and {raw.shape}"
        )
        return None
    return np.ascontiguousarray(raw)


def _is_positive_zero_float32(value: np.ndarray) -> bool:
    contiguous = np.ascontiguousarray(value, dtype=np.float32)
    return bool(np.all(contiguous.view(np.uint32) == np.uint32(0)))


def _validate_horizon_trace(
    result: HiddenPartnerRunResult,
) -> tuple[bool, tuple[str, ...]]:
    """Authenticate the fixed schedule, target-only updates, and exact clocks."""
    errors: list[str] = []
    active = _trace_array(errors, result, "active", np.dtype(np.bool_), ())
    steps = _trace_array(errors, result, "step", np.dtype(np.int32), ())
    segment_index = _trace_array(
        errors, result, "segment_index", np.dtype(np.int32), ()
    )
    segment_step = _trace_array(
        errors, result, "segment_step", np.dtype(np.int32), ()
    )
    segment_length = _trace_array(
        errors, result, "segment_length", np.dtype(np.int32), ()
    )
    regimes = _trace_array(errors, result, "regime_id", np.dtype(np.int32), ())
    probe_errors = _trace_array(
        errors,
        result,
        "interaction_relevance_probe_errors",
        np.dtype(np.float32),
        (1, 12),
    )
    probe_weights_pre = _trace_array(
        errors,
        result,
        "interaction_relevance_probe_weights_pre",
        np.dtype(np.float32),
        (1, 12),
    )
    probe_weights_post = _trace_array(
        errors,
        result,
        "interaction_relevance_probe_weights_post",
        np.dtype(np.float32),
        (1, 12),
    )
    probe_biases_pre = _trace_array(
        errors,
        result,
        "interaction_relevance_probe_biases_pre",
        np.dtype(np.float32),
        (1,),
    )
    probe_biases_post = _trace_array(
        errors,
        result,
        "interaction_relevance_probe_biases_post",
        np.dtype(np.float32),
        (1,),
    )
    candidate_descriptors = _trace_array(
        errors,
        result,
        "candidate_descriptors",
        np.dtype(np.int32),
        (66, 2),
    )
    reset_mask = _trace_array(
        errors,
        result,
        "interaction_matching_candidate_reset_mask",
        np.dtype(np.bool_),
        (66,),
    )
    reset_count = _trace_array(
        errors,
        result,
        "interaction_matching_candidate_reset_count",
        np.dtype(np.int32),
        (),
    )
    candidate_utilities_post = _trace_array(
        errors,
        result,
        "candidate_utilities_post",
        np.dtype(np.float32),
        (66,),
    )
    candidate_weights_post = _trace_array(
        errors,
        result,
        "candidate_output_weights_post",
        np.dtype(np.float32),
        (1, 66),
    )
    candidate_ages_post = _trace_array(
        errors,
        result,
        "candidate_ages_post",
        np.dtype(np.int32),
        (66,),
    )

    if active is not None and not bool(np.all(active)):
        errors.append("trace active mask must be true for the exact full cycle")
    if steps is not None and not np.array_equal(
        steps, np.arange(CYCLE_STEPS, dtype=np.int32)
    ):
        errors.append("trace step identity is not contiguous from zero")
    expected_segment_index = np.repeat(
        np.arange(9, dtype=np.int32), SEGMENT_LENGTH
    )
    if segment_index is not None and not np.array_equal(
        segment_index, expected_segment_index
    ):
        errors.append("trace segment-index schedule changed")
    expected_segment_step = np.tile(
        np.arange(SEGMENT_LENGTH, dtype=np.int32), 9
    )
    if segment_step is not None and not np.array_equal(
        segment_step, expected_segment_step
    ):
        errors.append("trace segment-step schedule changed")
    if segment_length is not None and not bool(
        np.all(segment_length == np.int32(SEGMENT_LENGTH))
    ):
        errors.append("trace segment lengths changed")
    expected_regimes = np.repeat(
        np.asarray(DEFAULT_REGIME_SCHEDULE, dtype=np.int32), SEGMENT_LENGTH
    )
    if regimes is not None and not np.array_equal(regimes, expected_regimes):
        errors.append("trace hidden-regime schedule changed")

    for field in (
        "state_builder_step_delta",
        "state_builder_learning_delta",
        "behavior_step_delta",
        "interaction_step_delta",
        "world_step_delta",
        "control_step_delta",
        "router_route_delta",
        "integrated_step_delta",
    ):
        values = _trace_array(errors, result, field, np.dtype(np.int32), ())
        if values is not None and not bool(np.all(values == np.int32(1))):
            errors.append(f"trace.{field} must equal one on every transition")
    for field in ("route_valid", "causal_transition_valid", "all_finite"):
        values = _trace_array(errors, result, field, np.dtype(np.bool_), ())
        if values is not None and not bool(np.all(values)):
            errors.append(f"trace.{field} must remain true throughout")

    if probe_errors is not None:
        if not bool(np.all(np.isfinite(probe_errors))):
            errors.append("target-only relevance errors contain nonfinite values")
        elif not bool(
            np.all(
                probe_errors.view(np.uint32)
                == probe_errors[:, :, :1].view(np.uint32)
            )
        ):
            errors.append("target-only relevance baseline differs across columns")
    for name, values in (
        ("probe weights pre", probe_weights_pre),
        ("probe weights post", probe_weights_post),
        ("probe biases pre", probe_biases_pre),
        ("probe biases post", probe_biases_post),
    ):
        if values is not None and not bool(np.all(np.isfinite(values))):
            errors.append(f"{name} contain nonfinite values")
    if probe_weights_pre is not None and probe_weights_post is not None:
        if not np.array_equal(
            probe_weights_post[:-1].view(np.uint32),
            probe_weights_pre[1:].view(np.uint32),
        ):
            errors.append("target-only relevance-probe weight state is discontinuous")
    if probe_biases_pre is not None and probe_biases_post is not None:
        if not np.array_equal(
            probe_biases_post[:-1].view(np.uint32),
            probe_biases_pre[1:].view(np.uint32),
        ):
            errors.append("target-only relevance-probe bias state is discontinuous")

    retired_left = _trace_array(
        errors, result, "interaction_retired_left", np.dtype(np.int32), ()
    )
    retired_right = _trace_array(
        errors, result, "interaction_retired_right", np.dtype(np.int32), ()
    )
    reset_fields = (
        candidate_descriptors,
        reset_mask,
        reset_count,
        candidate_utilities_post,
        candidate_weights_post,
        candidate_ages_post,
        retired_left,
        retired_right,
    )
    if all(value is not None for value in reset_fields):
        assert candidate_descriptors is not None
        assert reset_mask is not None
        assert reset_count is not None
        assert candidate_utilities_post is not None
        assert candidate_weights_post is not None
        assert candidate_ages_post is not None
        assert retired_left is not None
        assert retired_right is not None
        d_events = np.flatnonzero((retired_left == 4) & (retired_right == 5))
        for event in d_events:
            matching = np.flatnonzero(
                np.all(
                    candidate_descriptors[event]
                    == np.asarray((4, 5), dtype=np.int32),
                    axis=1,
                )
            )
            if matching.size != 1:
                errors.append(f"D retirement at {event} lacks one canonical candidate")
                continue
            candidate = int(matching[0])
            if not bool(reset_mask[event, candidate]) or int(reset_count[event]) != 1:
                errors.append(f"D retirement at {event} lacks one exact candidate reset")
            if not _is_positive_zero_float32(
                candidate_utilities_post[event, candidate]
            ):
                errors.append(f"D retirement at {event} did not reset utility to +0.0")
            if not _is_positive_zero_float32(
                candidate_weights_post[event, :, candidate]
            ):
                errors.append(f"D retirement at {event} did not reset head to +0.0")
            if int(candidate_ages_post[event, candidate]) != 0:
                errors.append(f"D retirement at {event} did not reset candidate age")

    final = result.final_agent_state
    expected_outer = np.asarray((0, CYCLE_STEPS), dtype=np.uint32)
    expected_builder = np.asarray((0, CYCLE_STEPS + 1), dtype=np.uint32)
    exact_words = (
        ("integrated", final.step_words, expected_outer),
        ("behavior", final.behavior.step_words, expected_outer),
        ("joint_world", final.joint_world.step_words, expected_outer),
        ("control", final.control.step_words, expected_outer),
        ("router.route", final.router.route_words, expected_outer),
        ("interaction", final.interaction.step_words, expected_outer),
        ("state_builder.step", final.state_builder.step_words, expected_builder),
        ("state_builder.update", final.state_builder.update_words, expected_outer),
    )
    for name, actual, expected in exact_words:
        raw = np.asarray(actual)
        if raw.dtype != np.dtype(np.uint32) or not np.array_equal(raw, expected):
            errors.append(f"final {name} exact lifetime words changed")
    if int(np.asarray(final.step_count)) != CYCLE_STEPS:
        errors.append("final integrated saturating telemetry changed")
    return not errors, tuple(errors)


def validate_horizon_sufficiency_run(
    result: HiddenPartnerRunResult,
    lifecycle: CriticalLifecycleV2Summary,
) -> HorizonSufficiencyRunValidation:
    """Validate one live run without trusting producer outcome booleans."""
    config_errors = list(validate_horizon_sufficiency_static_contract())
    expected_config = _horizon_agent_config()
    if result.condition.name != "full":
        config_errors.append("underlying condition must remain the nonpromoting full role")
    if result.condition.config != expected_config:
        config_errors.append("live agent config differs from the fixed horizon config")
    if result.condition.isolated_question != (
        "double online observation horizon with geometrically scaled retention grace"
    ):
        config_errors.append("isolated question changed")
    if result.summary.seed_pair != HORIZON_SUFFICIENCY_SEED:
        config_errors.append("run used another development seed")
    if result.summary.cycle_steps != CYCLE_STEPS:
        config_errors.append("run length changed")
    if result.summary.segment_lengths != SEGMENT_LENGTHS:
        config_errors.append("realized segment lengths changed")

    resource_errors: list[str] = []
    if result.initial_resource.to_dict() != HORIZON_SUFFICIENCY_RESOURCE_CONTRACT:
        resource_errors.append("initial resource contract changed")
    if result.final_resource.to_dict() != HORIZON_SUFFICIENCY_RESOURCE_CONTRACT:
        resource_errors.append("final resource contract changed")
    expected_bytes = HORIZON_SUFFICIENCY_RESOURCE_CONTRACT["total_state_nbytes"]
    if result.summary.initial_state_nbytes != expected_bytes:
        resource_errors.append("initial summary bytes changed")
    if result.summary.final_state_nbytes != expected_bytes:
        resource_errors.append("final summary bytes changed")

    trace_valid, trace_errors = _validate_horizon_trace(result)
    lifecycle_errors: list[str] = []
    if lifecycle.cycle_steps != CYCLE_STEPS:
        lifecycle_errors.append("lifecycle summary run length changed")
    if lifecycle.decision_state_count != CYCLE_STEPS + 1:
        lifecycle_errors.append("lifecycle decision-state count changed")
    if not result.summary.all_finite:
        lifecycle_errors.append("run summary is nonfinite")
    if not result.summary.counter_contract_valid:
        lifecycle_errors.append("legacy counter contract failed")
    if not result.summary.causal_contract_valid:
        lifecycle_errors.append("causal transition contract failed")
    if not result.summary.resource_shape_matched:
        lifecycle_errors.append("resource shape changed across the life")
    if not lifecycle.representation_link_contract_valid:
        lifecycle_errors.append("representation link contract failed")
    if not lifecycle.consumer_gate_contract_valid:
        lifecycle_errors.append("consumer evidence gate contract failed")
    if not lifecycle.feature_memory_enabled:
        lifecycle_errors.append("feature memory is disabled")
    if not lifecycle.feature_memory_contract_valid:
        lifecycle_errors.append("feature memory contract failed")
    if not lifecycle.candidate_archive_contract_valid:
        lifecycle_errors.append("candidate archive contract failed")

    errors = (*config_errors, *resource_errors, *trace_errors, *lifecycle_errors)
    return HorizonSufficiencyRunValidation(
        valid=not errors,
        config_contract_valid=not config_errors,
        resource_contract_valid=not resource_errors,
        trace_contract_valid=trace_valid,
        lifecycle_contract_valid=not lifecycle_errors,
        errors=tuple(errors),
    )


def _lifecycle_requirement_failures(
    lifecycle: CriticalLifecycleV2Summary,
) -> tuple[str, ...]:
    """Apply the predecessor's full lifecycle checks without weakening them."""
    checks = (
        ("c_task_learned", lifecycle.c_task_learned),
        ("c_continuously_survived", lifecycle.c_continuously_survived),
        ("c_retained_and_used", lifecycle.c_retained_and_used),
        ("d_task_learned", lifecycle.d_task_learned),
        ("d_retirement_event_aligned", lifecycle.d_retirement_event_aligned),
        (
            "d_linked_matching_candidate_reset_count_is_one",
            lifecycle.d_linked_matching_candidate_reset_count == 1,
        ),
        (
            "d_linked_candidate_utility_is_positive_zero",
            lifecycle.d_linked_candidate_utility_post == 0.0,
        ),
        (
            "d_linked_candidate_head_is_positive_zero",
            lifecycle.d_linked_candidate_head_linf_post == 0.0,
        ),
        (
            "d_linked_candidate_age_is_zero",
            lifecycle.d_linked_candidate_age_post == 0,
        ),
        (
            "d_repromotions_after_retirement_is_zero",
            lifecycle.d_repromotions_after_retirement == 0,
        ),
        ("d_absent_entire_final_window", lifecycle.d_absent_entire_final_window),
        (
            "d_learned_then_stably_retired",
            lifecycle.d_learned_then_stably_retired,
        ),
        (
            "joint_memory_management_success",
            lifecycle.joint_memory_management_success,
        ),
    )
    return tuple(name for name, passed in checks if not passed)


def _c_horizon_requirement_failures(
    lifecycle: CriticalLifecycleV2Summary,
) -> tuple[str, ...]:
    checks = (
        ("c_task_learned", lifecycle.c_task_learned),
        ("c_continuously_survived", lifecycle.c_continuously_survived),
        ("c_retained_and_used", lifecycle.c_retained_and_used),
    )
    return tuple(name for name, passed in checks if not passed)


def run_horizon_sufficiency_development() -> HorizonSufficiencyDevelopmentResult:
    """Run exactly one fixed, one-arm, nonpromoting horizon falsification."""
    static_errors = validate_horizon_sufficiency_static_contract()
    if static_errors:
        raise RuntimeError(
            "invalid horizon-sufficiency development contract: "
            + "; ".join(static_errors)
        )
    condition = HiddenPartnerCondition(
        name="full",
        config=_horizon_agent_config(),
        isolated_question=(
            "double online observation horizon with geometrically scaled retention grace"
        ),
    )
    run = HiddenPartnerDevelopmentRunner(
        condition,
        HORIZON_SUFFICIENCY_PROTOCOL,
    ).run(HORIZON_SUFFICIENCY_SEED)
    lifecycle = summarize_critical_lifecycle_v2(run)
    validation = validate_horizon_sufficiency_run(run, lifecycle)
    failures = _lifecycle_requirement_failures(lifecycle)
    c_failures = _c_horizon_requirement_failures(lifecycle)
    if lifecycle.c_acquisition_step is None:
        post_acquisition_steps = None
    else:
        post_acquisition_steps = max(
            C_FIRST_EXPOSURE_END_STEP - lifecycle.c_acquisition_step,
            0,
        )
    if not validation.valid:
        status: HorizonStatus = "invalid_development_run"
    elif failures:
        status = "valid_development_rejection"
    else:
        status = "passed_horizon_sufficiency_check"
    return HorizonSufficiencyDevelopmentResult(
        schema_version=HORIZON_SUFFICIENCY_DEVELOPMENT_SCHEMA,
        development_only=DEVELOPMENT_ONLY,
        scientific_promotion_allowed=SCIENTIFIC_PROMOTION_ALLOWED,
        output_writes_allowed=OUTPUT_WRITES_ALLOWED,
        status=status,
        lifecycle_requirement_failures=failures,
        c_horizon_requirement_failures=c_failures,
        c_post_acquisition_observation_steps=post_acquisition_steps,
        run=run,
        lifecycle=lifecycle,
        validation=validation,
    )


__all__ = [
    "CYCLE_STEPS",
    "C_ABSENCE_GAP_STEPS",
    "C_FIRST_EXPOSURE_END_STEP",
    "C_FIRST_EXPOSURE_START_STEP",
    "DEVELOPMENT_ONLY",
    "HORIZON_LIFECYCLE_GATE_CONTRACT",
    "HORIZON_AGENT_CONFIG_CONTRACT",
    "HORIZON_SUFFICIENCY_DEVELOPMENT_SCHEMA",
    "HORIZON_SUFFICIENCY_NAMESPACE",
    "HORIZON_SUFFICIENCY_PROTOCOL",
    "HORIZON_SUFFICIENCY_RESOURCE_CONTRACT",
    "HORIZON_SUFFICIENCY_SEED",
    "HorizonStatus",
    "HorizonSufficiencyDevelopmentResult",
    "HorizonSufficiencyRunValidation",
    "OUTPUT_WRITES_ALLOWED",
    "PREDECESSOR_RETENTION_GRACE_STEPS",
    "PREDECESSOR_SEGMENT_LENGTH",
    "RETENTION_GRACE_STEPS",
    "SCIENTIFIC_PROMOTION_ALLOWED",
    "SEGMENT_LENGTH",
    "SEGMENT_LENGTHS",
    "run_horizon_sufficiency_development",
    "validate_horizon_sufficiency_run",
    "validate_horizon_sufficiency_static_contract",
]
