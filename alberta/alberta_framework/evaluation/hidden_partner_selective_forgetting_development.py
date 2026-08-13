"""Nonpromoting target-only selective-forgetting microcycle.

This successor is deliberately separate from the frozen v4 lease grid and
the frozen v5 confirmation plan.  It spends one explicit development seed on
one small, jitterless A->B->A->D->A->C->A->B->C life.  The primary question is
whether a target-only evidence lease can keep the recurring C-critical pair
continuously deployed while learning, explicitly retiring, and not
reacquiring the one-use D-critical pair.

The module has no artifact writer, promoted-evidence entry point, threshold
selection, or seed-search API.  Existing lifecycle-v2 gates are reused
without modification.  A contract-valid miss is reported as a valid
development rejection rather than being retuned on this consumed seed.
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
    CONFIRMATION_NAMESPACE,
    LEASE_TUNING_NAMESPACE,
    CriticalLifecycleV2Summary,
    summarize_critical_lifecycle_v2,
)
from alberta_framework.streams.hidden_partner_mapping import (
    DEFAULT_REGIME_SCHEDULE,
    HiddenPartnerMappingConfig,
)

SELECTIVE_FORGETTING_DEVELOPMENT_SCHEMA = (
    "alberta.hidden-partner-selective-forgetting-development.microcycle.v1"
)
DEVELOPMENT_ONLY = True
SCIENTIFIC_PROMOTION_ALLOWED = False
OUTPUT_WRITES_ALLOWED = False

SELECTIVE_FORGETTING_DEVELOPMENT_NAMESPACE = (
    "hidden-partner-v0-dev-target-only-selective-forgetting-microcycle-v1"
)
SELECTIVE_FORGETTING_DEVELOPMENT_SEED = HiddenPartnerSeedPair(
    namespace=SELECTIVE_FORGETTING_DEVELOPMENT_NAMESPACE,
    index=0,
    stream_seed=1_752_259_356,
    initialization_seed=1_288_584_077,
)

MICROCYCLE_SEGMENT_LENGTH = 256
MICROCYCLE_SEGMENT_LENGTHS = (MICROCYCLE_SEGMENT_LENGTH,) * 9
MICROCYCLE_STEPS = sum(MICROCYCLE_SEGMENT_LENGTHS)
MICROCYCLE_GRACE_STEPS = 640

SELECTIVE_FORGETTING_MICROCYCLE_PROTOCOL = HiddenPartnerDevelopmentProtocol(
    environment=HiddenPartnerMappingConfig(
        base_segment_lengths=MICROCYCLE_SEGMENT_LENGTHS,
        jitter_radius=0,
        partner_flip_probability=0.05,
    ),
    recovery_window=128,
    early_late_window=128,
)

ArmName = Literal[
    "selective_lease",
    "retirement_disabled",
    "reacquisition_one",
]
SELECTIVE_FORGETTING_ARM_ORDER: tuple[ArmName, ...] = (
    "selective_lease",
    "retirement_disabled",
    "reacquisition_one",
)

# Exact logical resource contract after the state-builder, interaction,
# behavior, joint-world, control, router, and integrated lifetime clocks were
# widened.  It is 76 bytes larger than the historical v5 total of 6,757.
SELECTIVE_FORGETTING_RESOURCE_CONTRACT: dict[str, int] = {
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


@dataclasses.dataclass(frozen=True)
class SelectiveForgettingDevelopmentArm:
    """One exact fixed-shape arm in the three-condition microcycle."""

    name: ArmName
    config: IntegratedHiddenPartnerConfig
    isolated_question: str

    def to_config(self) -> dict[str, object]:
        """Return an authority-free JSON-compatible arm record."""
        return {
            "name": self.name,
            "agent_config": self.config.to_config(),
            "isolated_question": self.isolated_question,
        }


def _selective_lease_config() -> IntegratedHiddenPartnerConfig:
    """Return the single fixed primary configuration without tuning."""
    return IntegratedHiddenPartnerConfig(
        active_utility_retention_decay=0.9999,
        active_utility_retention_grace_steps=MICROCYCLE_GRACE_STEPS,
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


def build_selective_forgetting_development_arms() -> tuple[
    SelectiveForgettingDevelopmentArm, ...
]:
    """Build the primary and two shape-matched causal controls."""
    primary = _selective_lease_config()
    return (
        SelectiveForgettingDevelopmentArm(
            name="selective_lease",
            config=primary,
            isolated_question=(
                "target-only evidence lease retains recurring C while explicitly "
                "retiring one-use D"
            ),
        ),
        SelectiveForgettingDevelopmentArm(
            name="retirement_disabled",
            config=dataclasses.replace(primary, retire_stale_features=False),
            isolated_question="explicit stale retirement versus ordinary replacement only",
        ),
        SelectiveForgettingDevelopmentArm(
            name="reacquisition_one",
            config=dataclasses.replace(
                primary,
                candidate_reacquisition_confirmation_steps=1,
            ),
            isolated_question="eight-step retired-identity confirmation versus one-step",
        ),
    )


SELECTIVE_FORGETTING_DEVELOPMENT_ARMS = build_selective_forgetting_development_arms()


@dataclasses.dataclass(frozen=True)
class SelectiveForgettingRunValidation:
    """Fail-closed static, resource, trace, and lifecycle contract verdict."""

    valid: bool
    config_contract_valid: bool
    resource_contract_valid: bool
    trace_contract_valid: bool
    lifecycle_contract_valid: bool
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class SelectiveForgettingArmResult:
    """One in-memory development run and its independently rebuilt lifecycle."""

    arm: SelectiveForgettingDevelopmentArm
    run: HiddenPartnerRunResult
    lifecycle: CriticalLifecycleV2Summary
    validation: SelectiveForgettingRunValidation

    def outcome_dict(self) -> dict[str, object]:
        """Return only descriptive arm outcomes, never an evidence claim."""
        lifecycle = self.lifecycle
        return {
            "arm": self.arm.name,
            "validation": self.validation.to_dict(),
            "mean_reward": self.run.summary.mean_reward,
            "c_promotion_event_steps": list(lifecycle.c_promotion_event_steps),
            "c_acquisition_step": lifecycle.c_acquisition_step,
            "c_task_learned": lifecycle.c_task_learned,
            "c_continuously_survived": lifecycle.c_continuously_survived,
            "c_retained_and_used": lifecycle.c_retained_and_used,
            "c_survival_gap_steps": lifecycle.c_survival_gap_steps,
            "d_promotion_event_steps": list(lifecycle.d_promotion_event_steps),
            "d_acquisition_step": lifecycle.d_acquisition_step,
            "d_task_learned": lifecycle.d_task_learned,
            "d_retirement_event_steps": list(lifecycle.d_retirement_event_steps),
            "d_retirement_event_aligned": lifecycle.d_retirement_event_aligned,
            "d_retirement_latency_steps": lifecycle.d_retirement_latency_steps,
            "d_post_exit_live_slot_steps": lifecycle.d_post_exit_live_slot_steps,
            "d_repromotions_after_retirement": lifecycle.d_repromotions_after_retirement,
            "d_absent_entire_final_window": lifecycle.d_absent_entire_final_window,
            "d_learned_then_stably_retired": lifecycle.d_learned_then_stably_retired,
            "joint_memory_management_success": lifecycle.joint_memory_management_success,
        }


PanelStatus = Literal[
    "passed_development_checks",
    "valid_development_rejection",
    "invalid_development_run",
]


@dataclasses.dataclass(frozen=True)
class SelectiveForgettingPanelResult:
    """Complete three-arm result with validity separate from outcome."""

    schema_version: str
    development_only: bool
    scientific_promotion_allowed: bool
    output_writes_allowed: bool
    status: PanelStatus
    primary_requirement_failures: tuple[str, ...]
    arms: tuple[SelectiveForgettingArmResult, ...]

    def arm_result(self, name: ArmName) -> SelectiveForgettingArmResult:
        """Return exactly one named arm."""
        matches = tuple(result for result in self.arms if result.arm.name == name)
        if len(matches) != 1:
            raise ValueError(f"panel must contain exactly one {name!r} arm")
        return matches[0]

    def to_report(self) -> dict[str, object]:
        """Return a compact nonpromoting diagnostic report."""
        return {
            "schema_version": self.schema_version,
            "development_only": self.development_only,
            "scientific_promotion_allowed": self.scientific_promotion_allowed,
            "output_writes_allowed": self.output_writes_allowed,
            "status": self.status,
            "primary_requirement_failures": list(self.primary_requirement_failures),
            "seed": SELECTIVE_FORGETTING_DEVELOPMENT_SEED.to_dict(),
            "protocol": {
                "segment_lengths": list(MICROCYCLE_SEGMENT_LENGTHS),
                "jitter_radius": 0,
                "partner_flip_probability": 0.05,
                "grace_steps": MICROCYCLE_GRACE_STEPS,
            },
            "arms": [result.outcome_dict() for result in self.arms],
            "interpretation": (
                "development-only mechanism falsification; passing cannot promote "
                "scientific evidence"
            ),
        }


def validate_selective_forgetting_static_contract() -> tuple[str, ...]:
    """Validate namespace, seed, protocol, arm, and nonpromotion contracts."""
    errors: list[str] = []
    if SELECTIVE_FORGETTING_DEVELOPMENT_NAMESPACE in (
        LEASE_TUNING_NAMESPACE,
        CONFIRMATION_NAMESPACE,
    ):
        errors.append("development namespace collides with a frozen forbidden namespace")
    expected_seed = derive_hidden_partner_seed_pairs(
        SELECTIVE_FORGETTING_DEVELOPMENT_NAMESPACE,
        1,
    )[0]
    if expected_seed != SELECTIVE_FORGETTING_DEVELOPMENT_SEED:
        errors.append("fixed development seed differs from its namespaced derivation")
    protocol = SELECTIVE_FORGETTING_MICROCYCLE_PROTOCOL
    if protocol.environment.base_segment_lengths != MICROCYCLE_SEGMENT_LENGTHS:
        errors.append("microcycle segment lengths changed")
    if protocol.environment.jitter_radius != 0:
        errors.append("microcycle must remain jitterless")
    if protocol.environment.partner_flip_probability != 0.05:
        errors.append("microcycle partner flip probability changed")
    if protocol.maximum_cycle_steps != MICROCYCLE_STEPS:
        errors.append("microcycle scan length changed")

    arms = SELECTIVE_FORGETTING_DEVELOPMENT_ARMS
    if tuple(arm.name for arm in arms) != SELECTIVE_FORGETTING_ARM_ORDER:
        errors.append("matched arm order changed")
    if arms != build_selective_forgetting_development_arms():
        errors.append("matched arm configuration changed")
    for arm in arms:
        config = arm.config
        if config.relevance_probe_mode != RELEVANCE_PROBE_MODE_TARGET_ONLY_V1:
            errors.append(f"{arm.name}: relevance probe is not target-only")
        if not config.evidence_gated_feature_memory:
            errors.append(f"{arm.name}: feature memory gate is disabled")
        if not config.evidence_gated_consumer_memory:
            errors.append(f"{arm.name}: consumer memory gate is disabled")
        if not config.independent_relevance_probe:
            errors.append(f"{arm.name}: independent relevance probe is disabled")
        if config.active_utility_retention_grace_steps != MICROCYCLE_GRACE_STEPS:
            errors.append(f"{arm.name}: evidence grace changed")
    if not arms[0].config.retire_stale_features:
        errors.append("primary arm must explicitly retire stale features")
    if arms[1].config.retire_stale_features:
        errors.append("retirement-disabled arm still retires stale features")
    if arms[2].config.candidate_reacquisition_confirmation_steps != 1:
        errors.append("reacquisition-one arm does not use one-step confirmation")
    return tuple(errors)


def _array_contract(
    errors: list[str],
    result: HiddenPartnerRunResult,
    field: str,
    dtype: np.dtype[np.generic],
    tail_shape: tuple[int, ...],
) -> np.ndarray | None:
    raw = np.asarray(getattr(result.trace, field))
    expected_shape = (MICROCYCLE_STEPS, *tail_shape)
    if raw.dtype != dtype or raw.shape != expected_shape:
        errors.append(
            f"trace.{field} must have dtype {dtype} and shape {expected_shape}, "
            f"got {raw.dtype} and {raw.shape}"
        )
        return None
    return np.ascontiguousarray(raw)


def _positive_zero_float32(value: np.ndarray) -> bool:
    array = np.ascontiguousarray(value, dtype=np.float32)
    return bool(np.all(array.view(np.uint32) == np.uint32(0)))


def _validate_target_only_trace(
    result: HiddenPartnerRunResult,
) -> tuple[bool, tuple[str, ...]]:
    errors: list[str] = []
    active = _array_contract(errors, result, "active", np.dtype(np.bool_), ())
    steps = _array_contract(errors, result, "step", np.dtype(np.int32), ())
    segment_index = _array_contract(
        errors,
        result,
        "segment_index",
        np.dtype(np.int32),
        (),
    )
    segment_step = _array_contract(
        errors,
        result,
        "segment_step",
        np.dtype(np.int32),
        (),
    )
    segment_length = _array_contract(
        errors,
        result,
        "segment_length",
        np.dtype(np.int32),
        (),
    )
    regimes = _array_contract(errors, result, "regime_id", np.dtype(np.int32), ())
    probe_errors = _array_contract(
        errors,
        result,
        "interaction_relevance_probe_errors",
        np.dtype(np.float32),
        (1, 12),
    )
    probe_weights_pre = _array_contract(
        errors,
        result,
        "interaction_relevance_probe_weights_pre",
        np.dtype(np.float32),
        (1, 12),
    )
    probe_weights_post = _array_contract(
        errors,
        result,
        "interaction_relevance_probe_weights_post",
        np.dtype(np.float32),
        (1, 12),
    )
    probe_biases_pre = _array_contract(
        errors,
        result,
        "interaction_relevance_probe_biases_pre",
        np.dtype(np.float32),
        (1,),
    )
    probe_biases_post = _array_contract(
        errors,
        result,
        "interaction_relevance_probe_biases_post",
        np.dtype(np.float32),
        (1,),
    )
    _array_contract(
        errors,
        result,
        "active_descriptors",
        np.dtype(np.int32),
        (12, 2),
    )
    _array_contract(
        errors,
        result,
        "deployed_descriptors_post",
        np.dtype(np.int32),
        (12, 2),
    )
    candidate_descriptors = _array_contract(
        errors,
        result,
        "candidate_descriptors",
        np.dtype(np.int32),
        (66, 2),
    )

    if active is not None and not bool(np.all(active)):
        errors.append("trace active mask must be an all-true exact microcycle")
    if steps is not None and not np.array_equal(
        steps,
        np.arange(MICROCYCLE_STEPS, dtype=np.int32),
    ):
        errors.append("trace step sequence is not contiguous from zero")
    expected_segments = np.repeat(
        np.arange(9, dtype=np.int32),
        MICROCYCLE_SEGMENT_LENGTH,
    )
    if segment_index is not None and not np.array_equal(segment_index, expected_segments):
        errors.append("trace segment-index schedule changed")
    expected_segment_steps = np.tile(
        np.arange(MICROCYCLE_SEGMENT_LENGTH, dtype=np.int32),
        9,
    )
    if segment_step is not None and not np.array_equal(segment_step, expected_segment_steps):
        errors.append("trace segment-step schedule changed")
    if segment_length is not None and not np.all(
        segment_length == np.int32(MICROCYCLE_SEGMENT_LENGTH)
    ):
        errors.append("trace segment lengths changed")
    expected_regimes = np.repeat(
        np.asarray(DEFAULT_REGIME_SCHEDULE, dtype=np.int32),
        MICROCYCLE_SEGMENT_LENGTH,
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
        values = _array_contract(errors, result, field, np.dtype(np.int32), ())
        if values is not None and not np.all(values == np.int32(1)):
            errors.append(f"trace.{field} must equal one on every transition")
    for field in ("route_valid", "causal_transition_valid", "all_finite"):
        values = _array_contract(errors, result, field, np.dtype(np.bool_), ())
        if values is not None and not bool(np.all(values)):
            errors.append(f"trace.{field} must be true on every transition")

    if probe_errors is not None:
        if not np.all(np.isfinite(probe_errors)):
            errors.append("target-only relevance errors contain nonfinite values")
        else:
            error_bits = probe_errors.view(np.uint32)
            if not np.all(error_bits == error_bits[:, :, :1]):
                errors.append("target-only relevance baseline differs across feature columns")
    for name, values in (
        ("probe weights pre", probe_weights_pre),
        ("probe weights post", probe_weights_post),
        ("probe biases pre", probe_biases_pre),
        ("probe biases post", probe_biases_post),
    ):
        if values is not None and not np.all(np.isfinite(values)):
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

    retired_left = np.asarray(result.trace.interaction_retired_left)
    retired_right = np.asarray(result.trace.interaction_retired_right)
    d_events = np.flatnonzero((retired_left == 4) & (retired_right == 5))
    if candidate_descriptors is not None:
        reset_mask = np.asarray(
            result.trace.interaction_matching_candidate_reset_mask,
            dtype=np.bool_,
        )
        reset_count = np.asarray(
            result.trace.interaction_matching_candidate_reset_count,
            dtype=np.int32,
        )
        candidate_utilities_post = np.asarray(
            result.trace.candidate_utilities_post,
            dtype=np.float32,
        )
        candidate_weights_post = np.asarray(
            result.trace.candidate_output_weights_post,
            dtype=np.float32,
        )
        candidate_ages_post = np.asarray(
            result.trace.candidate_ages_post,
            dtype=np.int32,
        )
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
            if not _positive_zero_float32(candidate_utilities_post[event, candidate]):
                errors.append(f"D retirement at {event} did not reset utility to +0.0")
            if not _positive_zero_float32(candidate_weights_post[event, :, candidate]):
                errors.append(f"D retirement at {event} did not reset candidate head to +0.0")
            if int(candidate_ages_post[event, candidate]) != 0:
                errors.append(f"D retirement at {event} did not reset candidate age")

    final = result.final_agent_state
    expected_outer = np.asarray((0, MICROCYCLE_STEPS), dtype=np.uint32)
    expected_builder_step = np.asarray((0, MICROCYCLE_STEPS + 1), dtype=np.uint32)
    exact_words = (
        ("integrated", final.step_words, expected_outer),
        ("behavior", final.behavior.step_words, expected_outer),
        ("joint_world", final.joint_world.step_words, expected_outer),
        ("control", final.control.step_words, expected_outer),
        ("router.route", final.router.route_words, expected_outer),
        ("interaction", final.interaction.step_words, expected_outer),
        ("state_builder.step", final.state_builder.step_words, expected_builder_step),
        ("state_builder.update", final.state_builder.update_words, expected_outer),
    )
    for name, actual, expected in exact_words:
        raw = np.asarray(actual)
        if raw.dtype != np.dtype(np.uint32) or not np.array_equal(raw, expected):
            errors.append(f"final {name} exact lifetime words changed")
    if int(np.asarray(final.step_count)) != MICROCYCLE_STEPS:
        errors.append("final integrated saturating telemetry changed")

    return not errors, tuple(errors)


def validate_selective_forgetting_run(
    arm: SelectiveForgettingDevelopmentArm,
    result: HiddenPartnerRunResult,
    lifecycle: CriticalLifecycleV2Summary,
) -> SelectiveForgettingRunValidation:
    """Validate one live run without trusting producer success booleans."""
    config_errors: list[str] = list(validate_selective_forgetting_static_contract())
    expected_arms = {
        candidate.name: candidate
        for candidate in SELECTIVE_FORGETTING_DEVELOPMENT_ARMS
    }
    expected_arm = expected_arms.get(arm.name)
    if expected_arm is None or arm != expected_arm:
        config_errors.append(f"{arm.name}: arm differs from its fixed successor config")
    if result.condition.name != "full":
        config_errors.append(
            "underlying development condition must remain the nonpromoting full role"
        )
    if result.condition.config != arm.config:
        config_errors.append(f"{arm.name}: live agent config differs from the matched arm")
    if result.condition.isolated_question != arm.isolated_question:
        config_errors.append(f"{arm.name}: isolated question changed")
    if result.summary.seed_pair != SELECTIVE_FORGETTING_DEVELOPMENT_SEED:
        config_errors.append(f"{arm.name}: run used a different development seed")
    if result.summary.cycle_steps != MICROCYCLE_STEPS:
        config_errors.append(f"{arm.name}: run length changed")
    if result.summary.segment_lengths != MICROCYCLE_SEGMENT_LENGTHS:
        config_errors.append(f"{arm.name}: realized segment lengths changed")

    resource_errors: list[str] = []
    expected_resource = SELECTIVE_FORGETTING_RESOURCE_CONTRACT
    if result.initial_resource.to_dict() != expected_resource:
        resource_errors.append(f"{arm.name}: initial resource contract changed")
    if result.final_resource.to_dict() != expected_resource:
        resource_errors.append(f"{arm.name}: final resource contract changed")
    if result.summary.initial_state_nbytes != expected_resource["total_state_nbytes"]:
        resource_errors.append(f"{arm.name}: initial summary bytes changed")
    if result.summary.final_state_nbytes != expected_resource["total_state_nbytes"]:
        resource_errors.append(f"{arm.name}: final summary bytes changed")

    trace_valid, trace_errors = _validate_target_only_trace(result)
    lifecycle_errors: list[str] = []
    if not result.summary.all_finite:
        lifecycle_errors.append(f"{arm.name}: run summary is nonfinite")
    if not result.summary.counter_contract_valid:
        lifecycle_errors.append(f"{arm.name}: legacy counter contract failed")
    if not result.summary.causal_contract_valid:
        lifecycle_errors.append(f"{arm.name}: causal transition contract failed")
    if not result.summary.resource_shape_matched:
        lifecycle_errors.append(f"{arm.name}: resource shape changed across the life")
    if not lifecycle.representation_link_contract_valid:
        lifecycle_errors.append(f"{arm.name}: representation link contract failed")
    if not lifecycle.consumer_gate_contract_valid:
        lifecycle_errors.append(f"{arm.name}: consumer evidence gate contract failed")
    if not lifecycle.feature_memory_enabled:
        lifecycle_errors.append(f"{arm.name}: feature memory must be enabled")
    if not lifecycle.feature_memory_contract_valid:
        lifecycle_errors.append(f"{arm.name}: feature memory contract failed")
    if not lifecycle.candidate_archive_contract_valid:
        lifecycle_errors.append(f"{arm.name}: candidate archive contract failed")

    errors = (*config_errors, *resource_errors, *trace_errors, *lifecycle_errors)
    return SelectiveForgettingRunValidation(
        valid=not errors,
        config_contract_valid=not config_errors,
        resource_contract_valid=not resource_errors,
        trace_contract_valid=trace_valid,
        lifecycle_contract_valid=not lifecycle_errors,
        errors=tuple(errors),
    )


def _primary_requirement_failures(
    lifecycle: CriticalLifecycleV2Summary,
) -> tuple[str, ...]:
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
        ("d_linked_candidate_age_is_zero", lifecycle.d_linked_candidate_age_post == 0),
        ("d_repromotions_after_retirement_is_zero", lifecycle.d_repromotions_after_retirement == 0),
        ("d_absent_entire_final_window", lifecycle.d_absent_entire_final_window),
        ("d_learned_then_stably_retired", lifecycle.d_learned_then_stably_retired),
        ("joint_memory_management_success", lifecycle.joint_memory_management_success),
    )
    return tuple(name for name, passed in checks if not passed)


def run_selective_forgetting_development_microcycle() -> SelectiveForgettingPanelResult:
    """Run exactly the fixed three-arm, one-seed development microcycle."""
    static_errors = validate_selective_forgetting_static_contract()
    if static_errors:
        raise RuntimeError(
            "invalid selective-forgetting development contract: " + "; ".join(static_errors)
        )

    arm_results: list[SelectiveForgettingArmResult] = []
    for arm in SELECTIVE_FORGETTING_DEVELOPMENT_ARMS:
        condition = HiddenPartnerCondition(
            name="full",
            config=arm.config,
            isolated_question=arm.isolated_question,
        )
        result = HiddenPartnerDevelopmentRunner(
            condition,
            SELECTIVE_FORGETTING_MICROCYCLE_PROTOCOL,
        ).run(SELECTIVE_FORGETTING_DEVELOPMENT_SEED)
        lifecycle = summarize_critical_lifecycle_v2(result)
        validation = validate_selective_forgetting_run(arm, result, lifecycle)
        arm_results.append(
            SelectiveForgettingArmResult(
                arm=arm,
                run=result,
                lifecycle=lifecycle,
                validation=validation,
            )
        )

    results = tuple(arm_results)
    primary_matches = tuple(
        result for result in results if result.arm.name == "selective_lease"
    )
    if len(primary_matches) != 1:
        raise RuntimeError("microcycle must contain exactly one primary arm")
    failures = _primary_requirement_failures(primary_matches[0].lifecycle)
    if not all(result.validation.valid for result in results):
        status: PanelStatus = "invalid_development_run"
    elif failures:
        status = "valid_development_rejection"
    else:
        status = "passed_development_checks"
    return SelectiveForgettingPanelResult(
        schema_version=SELECTIVE_FORGETTING_DEVELOPMENT_SCHEMA,
        development_only=DEVELOPMENT_ONLY,
        scientific_promotion_allowed=SCIENTIFIC_PROMOTION_ALLOWED,
        output_writes_allowed=OUTPUT_WRITES_ALLOWED,
        status=status,
        primary_requirement_failures=failures,
        arms=results,
    )


__all__ = [
    "DEVELOPMENT_ONLY",
    "MICROCYCLE_GRACE_STEPS",
    "MICROCYCLE_SEGMENT_LENGTHS",
    "MICROCYCLE_STEPS",
    "OUTPUT_WRITES_ALLOWED",
    "SCIENTIFIC_PROMOTION_ALLOWED",
    "SELECTIVE_FORGETTING_ARM_ORDER",
    "SELECTIVE_FORGETTING_DEVELOPMENT_ARMS",
    "SELECTIVE_FORGETTING_DEVELOPMENT_NAMESPACE",
    "SELECTIVE_FORGETTING_DEVELOPMENT_SCHEMA",
    "SELECTIVE_FORGETTING_DEVELOPMENT_SEED",
    "SELECTIVE_FORGETTING_MICROCYCLE_PROTOCOL",
    "SELECTIVE_FORGETTING_RESOURCE_CONTRACT",
    "SelectiveForgettingArmResult",
    "SelectiveForgettingDevelopmentArm",
    "SelectiveForgettingPanelResult",
    "SelectiveForgettingRunValidation",
    "build_selective_forgetting_development_arms",
    "run_selective_forgetting_development_microcycle",
    "validate_selective_forgetting_run",
    "validate_selective_forgetting_static_contract",
]
