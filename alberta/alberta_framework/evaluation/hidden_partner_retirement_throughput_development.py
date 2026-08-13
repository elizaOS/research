"""Development-only falsification of bounded stale-retirement throughput.

The target-only selective-forgetting v1 microcycle produced a valid rejection:
the one-use D pair became stale, but its retirement shared the 64-step ordinary
promotion cadence and landed inside the final-absence audit window. This module
spends one new namespaced development seed on one isolated mechanism change:
retirement is due every 31 exact transactions while ordinary replacement stays
due every 64. A due retirement may atomically refill its one new vacancy only
with a separately confirmed, nonmatching candidate under the existing
promotion criteria. If no candidate qualifies, the single vacancy blocks later
retirements, so the intervention cannot collapse the fixed feature bank.

The lifecycle-v2 gates are consumed unchanged. Only the D timing, linked reset,
and stable-final-absence question is judged here; C and joint memory-management
outcomes are descriptive. There is no artifact writer, seed-search API, or
scientific-promotion path.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Literal

import numpy as np

from alberta_framework.core.integrated_hidden_partner import (
    ACTIVE_PAIR_SLOTS,
    CANDIDATE_PAIR_SLOTS,
    IntegratedHiddenPartnerAgent,
    IntegratedHiddenPartnerConfig,
)
from alberta_framework.core.interaction_features import (
    RELEVANCE_PROBE_MODE_TARGET_ONLY_V1,
    FixedBudgetInteractionLearner,
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
    FINAL_ABSENCE_WINDOW,
    LEASE_TUNING_NAMESPACE,
    CriticalLifecycleV2Summary,
    summarize_critical_lifecycle_v2,
)
from alberta_framework.streams.hidden_partner_mapping import HiddenPartnerMappingConfig

RETIREMENT_THROUGHPUT_DEVELOPMENT_SCHEMA = (
    "alberta.hidden-partner-retirement-throughput-development.falsification.v1"
)
DEVELOPMENT_ONLY = True
SCIENTIFIC_PROMOTION_ALLOWED = False
OUTPUT_WRITES_ALLOWED = False

RETIREMENT_THROUGHPUT_NAMESPACE = (
    "hidden-partner-v0-dev-retirement-throughput-falsification-v1"
)
RETIREMENT_THROUGHPUT_SEED = HiddenPartnerSeedPair(
    namespace=RETIREMENT_THROUGHPUT_NAMESPACE,
    index=0,
    stream_seed=1_459_870_326,
    initialization_seed=942_973_286,
)

SEGMENT_LENGTH = 256
SEGMENT_LENGTHS = (SEGMENT_LENGTH,) * 9
CYCLE_STEPS = sum(SEGMENT_LENGTHS)
RETENTION_GRACE_STEPS = 640
ORDINARY_REPLACEMENT_INTERVAL = 64
PROMPT_RETIREMENT_INTERVAL = 31

# D exits after the fourth segment. The lease uses a strict ``idle > grace``
# check, and the lifecycle-v2 final window includes one decision state beyond
# its nominal transition count. The products below are coarse scheduling-
# capacity bounds conditional on every queued slot being stale and every
# retirement finding a separately confirmed refill. They do not predict D's
# queue rank, evidence eligibility, or cadence phase on a realized life.
D_EXIT_STEP = 4 * SEGMENT_LENGTH
EARLIEST_D_STALE_STEP = D_EXIT_STEP + RETENTION_GRACE_STEPS + 1
FINAL_ABSENCE_START_STEP = CYCLE_STEPS - (FINAL_ABSENCE_WINDOW + 1)
RETIREMENT_SLACK_STEPS = FINAL_ABSENCE_START_STEP - EARLIEST_D_STALE_STEP
PROMPT_FULL_BANK_QUEUE_BOUND_STEPS = ACTIVE_PAIR_SLOTS * PROMPT_RETIREMENT_INTERVAL
LEGACY_FULL_BANK_QUEUE_BOUND_STEPS = (
    ACTIVE_PAIR_SLOTS * ORDINARY_REPLACEMENT_INTERVAL
)

RETIREMENT_THROUGHPUT_PROTOCOL = HiddenPartnerDevelopmentProtocol(
    environment=HiddenPartnerMappingConfig(
        base_segment_lengths=SEGMENT_LENGTHS,
        jitter_radius=0,
        partner_flip_probability=0.05,
    ),
    recovery_window=128,
    early_late_window=128,
)

RETIREMENT_THROUGHPUT_RESOURCE_CONTRACT: dict[str, int] = {
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


def _base_agent_config() -> IntegratedHiddenPartnerConfig:
    """Return the unchanged target-only v1 mechanism configuration."""
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
        replacement_interval=ORDINARY_REPLACEMENT_INTERVAL,
        min_feature_age=256,
        candidate_min_age=128,
    )


ArmName = Literal["prompt_retirement", "legacy_throughput"]
ARM_ORDER: tuple[ArmName, ...] = ("prompt_retirement", "legacy_throughput")


@dataclasses.dataclass(frozen=True)
class RetirementThroughputArm:
    """One matched arm differing only in explicit retirement cadence."""

    name: ArmName
    stale_retirement_interval: int | None
    isolated_question: str

    def to_config(self) -> dict[str, object]:
        return dataclasses.asdict(self)


def build_retirement_throughput_arms() -> tuple[RetirementThroughputArm, ...]:
    """Build the fixed intervention and legacy-throughput control."""
    return (
        RetirementThroughputArm(
            name="prompt_retirement",
            stale_retirement_interval=PROMPT_RETIREMENT_INTERVAL,
            isolated_question=(
                "exact 31-step stale-retirement cadence with atomic confirmed refill"
            ),
        ),
        RetirementThroughputArm(
            name="legacy_throughput",
            stale_retirement_interval=None,
            isolated_question=(
                "legacy retirement coupled to the exact 64-step replacement cadence"
            ),
        ),
    )


RETIREMENT_THROUGHPUT_ARMS = build_retirement_throughput_arms()


def _interaction_learner_for_arm(
    arm: RetirementThroughputArm,
) -> FixedBudgetInteractionLearner:
    """Rebuild the integrated learner with exactly one serialized config edit."""
    baseline = IntegratedHiddenPartnerAgent(_base_agent_config()).interaction_learner
    payload = baseline.to_config()
    if payload.get("stale_retirement_interval") is not None:
        raise RuntimeError("integrated baseline no longer uses legacy retirement cadence")
    payload["stale_retirement_interval"] = arm.stale_retirement_interval
    learner = FixedBudgetInteractionLearner.from_config(payload)
    expected = dict(baseline.to_config())
    expected["stale_retirement_interval"] = arm.stale_retirement_interval
    if learner.to_config() != expected:
        raise RuntimeError("retirement arm changed more than its serialized cadence")
    return learner


def validate_retirement_throughput_static_contract() -> tuple[str, ...]:
    """Fail closed on namespace, seed, protocol, queue bound, and arm drift."""
    errors: list[str] = []
    if RETIREMENT_THROUGHPUT_NAMESPACE in (
        LEASE_TUNING_NAMESPACE,
        CONFIRMATION_NAMESPACE,
        "hidden-partner-v0-dev-target-only-selective-forgetting-microcycle-v1",
    ):
        errors.append("retirement-throughput namespace collides with a consumed namespace")
    expected_seed = derive_hidden_partner_seed_pairs(
        RETIREMENT_THROUGHPUT_NAMESPACE,
        1,
    )[0]
    if expected_seed != RETIREMENT_THROUGHPUT_SEED:
        errors.append("fixed development seed differs from its namespace derivation")
    protocol = RETIREMENT_THROUGHPUT_PROTOCOL
    if protocol.environment.base_segment_lengths != SEGMENT_LENGTHS:
        errors.append("segment lengths changed")
    if protocol.environment.jitter_radius != 0:
        errors.append("microcycle must remain jitterless")
    if protocol.environment.partner_flip_probability != 0.05:
        errors.append("partner flip probability changed")
    if protocol.maximum_cycle_steps != CYCLE_STEPS:
        errors.append("scan length changed")
    if FINAL_ABSENCE_START_STEP != 2_047 or RETIREMENT_SLACK_STEPS != 382:
        errors.append("lifecycle-v2 final-window timing changed")
    if not (
        PROMPT_FULL_BANK_QUEUE_BOUND_STEPS <= RETIREMENT_SLACK_STEPS
        < LEGACY_FULL_BANK_QUEUE_BOUND_STEPS
    ):
        errors.append("fixed prompt/legacy queue-bound contrast no longer holds")
    if PROMPT_RETIREMENT_INTERVAL != RETIREMENT_SLACK_STEPS // ACTIVE_PAIR_SLOTS:
        errors.append("prompt cadence is no longer the fixed largest coarse bound")

    arms = RETIREMENT_THROUGHPUT_ARMS
    if tuple(arm.name for arm in arms) != ARM_ORDER:
        errors.append("arm order changed")
    if arms != build_retirement_throughput_arms():
        errors.append("arm definitions changed")
    base = IntegratedHiddenPartnerAgent(_base_agent_config()).interaction_learner.to_config()
    for arm in arms:
        live = _interaction_learner_for_arm(arm).to_config()
        if set(live) != set(base):
            errors.append(f"{arm.name}: interaction config field set changed")
        differing = {key for key in base if live.get(key) != base.get(key)}
        expected_differing = (
            set()
            if arm.stale_retirement_interval is None
            else {"stale_retirement_interval"}
        )
        if differing != expected_differing:
            errors.append(f"{arm.name}: interaction config differs outside cadence")
        if live.get("stale_retirement_interval") != arm.stale_retirement_interval:
            errors.append(f"{arm.name}: serialized retirement cadence changed")
        if live.get("replacement_interval") != ORDINARY_REPLACEMENT_INTERVAL:
            errors.append(f"{arm.name}: ordinary replacement cadence changed")
        if live.get("n_features") != ACTIVE_PAIR_SLOTS:
            errors.append(f"{arm.name}: active feature budget changed")
        if live.get("candidate_count") != CANDIDATE_PAIR_SLOTS:
            errors.append(f"{arm.name}: candidate budget changed")
    return tuple(errors)


@dataclasses.dataclass(frozen=True)
class RetirementThroughputRunValidation:
    """Independent config, resource, trace, and lifecycle validity verdict."""

    valid: bool
    config_contract_valid: bool
    resource_contract_valid: bool
    trace_contract_valid: bool
    lifecycle_contract_valid: bool
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class RetirementThroughputArmResult:
    """One live matched run and its unchanged lifecycle-v2 readout."""

    arm: RetirementThroughputArm
    interaction_config: dict[str, Any]
    run: HiddenPartnerRunResult
    lifecycle: CriticalLifecycleV2Summary
    validation: RetirementThroughputRunValidation

    def outcome_dict(self) -> dict[str, object]:
        lifecycle = self.lifecycle
        live_counts = np.asarray(self.run.trace.interaction_live_feature_count)
        vacancy_counts = np.asarray(self.run.trace.interaction_vacancy_count)
        return {
            "arm": self.arm.name,
            "stale_retirement_interval": self.arm.stale_retirement_interval,
            "validation": self.validation.to_dict(),
            "mean_reward": self.run.summary.mean_reward,
            "minimum_live_feature_count": int(np.min(live_counts)),
            "maximum_vacancy_count": int(np.max(vacancy_counts)),
            "d_task_learned": lifecycle.d_task_learned,
            "d_retirement_event_steps": list(lifecycle.d_retirement_event_steps),
            "d_retirement_event_aligned": lifecycle.d_retirement_event_aligned,
            "d_retirement_latency_steps": lifecycle.d_retirement_latency_steps,
            "d_linked_matching_candidate_reset_count": (
                lifecycle.d_linked_matching_candidate_reset_count
            ),
            "d_absent_entire_final_window": lifecycle.d_absent_entire_final_window,
            "d_repromotions_after_retirement": lifecycle.d_repromotions_after_retirement,
            "d_learned_then_stably_retired": lifecycle.d_learned_then_stably_retired,
            # Explicitly descriptive: this falsification does not judge C.
            "c_task_learned_descriptive": lifecycle.c_task_learned,
            "c_continuously_survived_descriptive": lifecycle.c_continuously_survived,
            "c_retained_and_used_descriptive": lifecycle.c_retained_and_used,
            "joint_memory_management_success_descriptive": (
                lifecycle.joint_memory_management_success
            ),
        }


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


def _validate_retirement_trace(
    arm: RetirementThroughputArm,
    result: HiddenPartnerRunResult,
) -> tuple[bool, tuple[str, ...]]:
    """Authenticate exact cadence, scalar atomicity, reset, and utilization."""
    errors: list[str] = []
    steps = _trace_array(errors, result, "step", np.dtype(np.int32), ())
    retired_slots = _trace_array(
        errors, result, "interaction_retired_slot", np.dtype(np.int32), ()
    )
    retired_left = _trace_array(
        errors, result, "interaction_retired_left", np.dtype(np.int32), ()
    )
    retired_right = _trace_array(
        errors, result, "interaction_retired_right", np.dtype(np.int32), ()
    )
    promoted = _trace_array(
        errors, result, "interaction_promoted_candidate", np.dtype(np.int32), ()
    )
    promoted_into_vacancy = _trace_array(
        errors,
        result,
        "interaction_promoted_into_vacancy",
        np.dtype(np.bool_),
        (),
    )
    live_counts = _trace_array(
        errors,
        result,
        "interaction_live_feature_count",
        np.dtype(np.int32),
        (),
    )
    vacancy_counts = _trace_array(
        errors,
        result,
        "interaction_vacancy_count",
        np.dtype(np.int32),
        (),
    )
    descriptors_pre = _trace_array(
        errors, result, "shadow_descriptors_pre", np.dtype(np.int32), (12, 2)
    )
    descriptors_post = _trace_array(
        errors, result, "shadow_descriptors_post", np.dtype(np.int32), (12, 2)
    )
    candidate_descriptors = _trace_array(
        errors, result, "candidate_descriptors", np.dtype(np.int32), (66, 2)
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
    probe_errors = _trace_array(
        errors,
        result,
        "interaction_relevance_probe_errors",
        np.dtype(np.float32),
        (1, 12),
    )
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
            errors.append(f"trace.{field} must equal one throughout")
    for field in ("route_valid", "causal_transition_valid", "all_finite"):
        values = _trace_array(errors, result, field, np.dtype(np.bool_), ())
        if values is not None and not bool(np.all(values)):
            errors.append(f"trace.{field} must remain true throughout")

    if steps is not None and not np.array_equal(
        steps,
        np.arange(CYCLE_STEPS, dtype=np.int32),
    ):
        errors.append("trace step identity is not the exact contiguous life")
    if probe_errors is not None:
        if not bool(np.all(np.isfinite(probe_errors))):
            errors.append("target-only relevance errors contain nonfinite values")
        elif not bool(
            np.all(probe_errors.view(np.uint32) == probe_errors[:, :, :1].view(np.uint32))
        ):
            errors.append("target-only relevance baseline differs across columns")

    required = (
        retired_slots,
        retired_left,
        retired_right,
        promoted,
        promoted_into_vacancy,
        live_counts,
        vacancy_counts,
        descriptors_pre,
        descriptors_post,
        candidate_descriptors,
        reset_mask,
        reset_count,
    )
    if all(value is not None for value in required):
        assert retired_slots is not None
        assert retired_left is not None
        assert retired_right is not None
        assert promoted is not None
        assert promoted_into_vacancy is not None
        assert live_counts is not None
        assert vacancy_counts is not None
        assert descriptors_pre is not None
        assert descriptors_post is not None
        assert candidate_descriptors is not None
        assert reset_mask is not None
        assert reset_count is not None
        events = retired_slots >= 0
        interval = (
            ORDINARY_REPLACEMENT_INTERVAL
            if arm.stale_retirement_interval is None
            else arm.stale_retirement_interval
        )
        event_steps = np.flatnonzero(events)
        if not bool(np.all((event_steps + 1) % interval == 0)):
            errors.append("retirement event is not bound to its exact post-step cadence")
        if not bool(np.all((retired_slots[events] >= 0) & (retired_slots[events] < 12))):
            errors.append("retired slot lies outside the fixed active bank")
        if not bool(np.all(retired_left[~events] == -1)) or not bool(
            np.all(retired_right[~events] == -1)
        ):
            errors.append("non-retirement transaction reports a retired identity")
        if not bool(np.all(np.sum(reset_mask, axis=1) == reset_count)):
            errors.append("scalar matching-candidate reset count disagrees with mask")
        if not bool(np.all(reset_count[events] == 1)) or not bool(
            np.all(reset_count[~events] == 0)
        ):
            errors.append("retirement must reset exactly one canonical archive entry")
        if not bool(np.all(live_counts + vacancy_counts == ACTIVE_PAIR_SLOTS)):
            errors.append("live/vacancy accounting does not conserve the active bank")
        if not bool(np.all((live_counts >= ACTIVE_PAIR_SLOTS - 1) & (live_counts <= 12))):
            errors.append("prompt retirement collapsed more than one active slot")
        if not bool(np.all((vacancy_counts >= 0) & (vacancy_counts <= 1))):
            errors.append("prompt retirement exceeded its one-vacancy fail-stop bound")

        prior_vacancies = np.concatenate(
            (np.zeros((1,), dtype=np.int32), vacancy_counts[:-1])
        )
        if bool(np.any(events & (prior_vacancies > 0))):
            errors.append("retirement occurred while a prior vacancy was outstanding")
        expected_fill = (promoted >= 0) & (prior_vacancies > 0)
        # A same-transaction retirement creates a new vacancy after the prior
        # count is observed, so include its separately confirmed refill too.
        expected_fill |= events & (promoted >= 0)
        if not np.array_equal(promoted_into_vacancy, expected_fill):
            errors.append("promoted-into-vacancy telemetry is not transaction exact")

        for event in event_steps:
            slot = int(retired_slots[event])
            retired_pair = np.asarray(
                (retired_left[event], retired_right[event]),
                dtype=np.int32,
            )
            if not np.array_equal(descriptors_pre[event, slot], retired_pair):
                errors.append(f"retirement {event} identity does not match its pre-state slot")
            matching = np.flatnonzero(reset_mask[event])
            if matching.size != 1 or not np.array_equal(
                candidate_descriptors[event, int(matching[0])],
                retired_pair,
            ):
                errors.append(f"retirement {event} reset is not linked to its identity")
            candidate = int(promoted[event])
            if candidate >= 0:
                if bool(reset_mask[event, candidate]):
                    errors.append(f"retirement {event} reacquired its reset identity")
                promoted_pair = candidate_descriptors[event, candidate]
                if np.array_equal(promoted_pair, retired_pair):
                    errors.append(f"retirement {event} promoted the retired identity")
                if not np.array_equal(descriptors_post[event, slot], promoted_pair):
                    errors.append(f"retirement {event} refill descriptor is not exact")
                if int(vacancy_counts[event]) != 0:
                    errors.append(f"retirement {event} refill left an unexpected vacancy")
            else:
                if not np.array_equal(descriptors_post[event, slot], (-1, -1)):
                    errors.append(f"retirement {event} unfilled slot is not canonical")
                if int(vacancy_counts[event]) != 1:
                    errors.append(f"retirement {event} missing refill lacks one vacancy")

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
    return not errors, tuple(errors)


def validate_retirement_throughput_run(
    arm: RetirementThroughputArm,
    interaction_config: dict[str, Any],
    result: HiddenPartnerRunResult,
    lifecycle: CriticalLifecycleV2Summary,
) -> RetirementThroughputRunValidation:
    """Validate one live run without trusting producer outcome booleans."""
    config_errors = list(validate_retirement_throughput_static_contract())
    expected = {candidate.name: candidate for candidate in RETIREMENT_THROUGHPUT_ARMS}
    if expected.get(arm.name) != arm:
        config_errors.append(f"{arm.name}: arm differs from fixed definition")
    expected_interaction = _interaction_learner_for_arm(arm).to_config()
    if interaction_config != expected_interaction:
        config_errors.append(f"{arm.name}: live interaction config changed")
    if result.condition.name != "full" or result.condition.config != _base_agent_config():
        config_errors.append(f"{arm.name}: integrated matched condition changed")
    if result.condition.isolated_question != arm.isolated_question:
        config_errors.append(f"{arm.name}: isolated question changed")
    if result.summary.seed_pair != RETIREMENT_THROUGHPUT_SEED:
        config_errors.append(f"{arm.name}: run used another development seed")
    if result.summary.cycle_steps != CYCLE_STEPS:
        config_errors.append(f"{arm.name}: run length changed")
    if result.summary.segment_lengths != SEGMENT_LENGTHS:
        config_errors.append(f"{arm.name}: realized segment lengths changed")

    resource_errors: list[str] = []
    if result.initial_resource.to_dict() != RETIREMENT_THROUGHPUT_RESOURCE_CONTRACT:
        resource_errors.append(f"{arm.name}: initial resource contract changed")
    if result.final_resource.to_dict() != RETIREMENT_THROUGHPUT_RESOURCE_CONTRACT:
        resource_errors.append(f"{arm.name}: final resource contract changed")
    if result.summary.initial_state_nbytes != 6_833:
        resource_errors.append(f"{arm.name}: initial state bytes changed")
    if result.summary.final_state_nbytes != 6_833:
        resource_errors.append(f"{arm.name}: final state bytes changed")

    trace_valid, trace_errors = _validate_retirement_trace(arm, result)
    lifecycle_errors: list[str] = []
    if not result.summary.all_finite:
        lifecycle_errors.append(f"{arm.name}: run summary is nonfinite")
    if not result.summary.counter_contract_valid:
        lifecycle_errors.append(f"{arm.name}: counter contract failed")
    if not result.summary.causal_contract_valid:
        lifecycle_errors.append(f"{arm.name}: causal contract failed")
    if not result.summary.resource_shape_matched:
        lifecycle_errors.append(f"{arm.name}: resource shape changed")
    for name, passed in (
        ("representation link", lifecycle.representation_link_contract_valid),
        ("consumer gate", lifecycle.consumer_gate_contract_valid),
        ("feature memory", lifecycle.feature_memory_contract_valid),
        ("candidate archive", lifecycle.candidate_archive_contract_valid),
    ):
        if not passed:
            lifecycle_errors.append(f"{arm.name}: {name} lifecycle-v2 contract failed")
    if not lifecycle.feature_memory_enabled:
        lifecycle_errors.append(f"{arm.name}: feature memory is disabled")

    errors = (*config_errors, *resource_errors, *trace_errors, *lifecycle_errors)
    return RetirementThroughputRunValidation(
        valid=not errors,
        config_contract_valid=not config_errors,
        resource_contract_valid=not resource_errors,
        trace_contract_valid=trace_valid,
        lifecycle_contract_valid=not lifecycle_errors,
        errors=tuple(errors),
    )


PanelStatus = Literal[
    "passed_d_retirement_falsification",
    "valid_development_rejection",
    "invalid_development_run",
]


@dataclasses.dataclass(frozen=True)
class RetirementThroughputPanelResult:
    """Two-arm validity and D-only mechanism outcome."""

    schema_version: str
    development_only: bool
    scientific_promotion_allowed: bool
    output_writes_allowed: bool
    status: PanelStatus
    d_requirement_failures: tuple[str, ...]
    prompt_retirement_precedes_legacy: bool
    final_absence_isolated_to_prompt: bool
    cadence_alone_falsified_on_fixed_seed: bool
    arms: tuple[RetirementThroughputArmResult, ...]

    def arm_result(self, name: ArmName) -> RetirementThroughputArmResult:
        matches = tuple(result for result in self.arms if result.arm.name == name)
        if len(matches) != 1:
            raise ValueError(f"panel must contain exactly one {name!r} arm")
        return matches[0]

    def to_report(self) -> dict[str, object]:
        prompt = self.arm_result("prompt_retirement").lifecycle
        legacy = self.arm_result("legacy_throughput").lifecycle
        prompt_event = _first_post_exit_d_event(prompt)
        legacy_event = _first_post_exit_d_event(legacy)
        return {
            "schema_version": self.schema_version,
            "development_only": self.development_only,
            "scientific_promotion_allowed": self.scientific_promotion_allowed,
            "output_writes_allowed": self.output_writes_allowed,
            "status": self.status,
            "d_requirement_failures": list(self.d_requirement_failures),
            "prompt_retirement_precedes_legacy": (
                self.prompt_retirement_precedes_legacy
            ),
            "final_absence_isolated_to_prompt": (
                self.final_absence_isolated_to_prompt
            ),
            "cadence_alone_falsified_on_fixed_seed": (
                self.cadence_alone_falsified_on_fixed_seed
            ),
            "fixed_seed_contrast": {
                "prompt_d_retirement_event_step": prompt_event,
                "legacy_d_retirement_event_step": legacy_event,
                "prompt_minus_legacy_event_steps": (
                    None
                    if prompt_event is None or legacy_event is None
                    else prompt_event - legacy_event
                ),
                "prompt_d_absent_final": prompt.d_absent_entire_final_window,
                "legacy_d_absent_final": legacy.d_absent_entire_final_window,
                "prompt_d_task_learned": prompt.d_task_learned,
                "legacy_d_task_learned": legacy.d_task_learned,
            },
            "seed": RETIREMENT_THROUGHPUT_SEED.to_dict(),
            "queue_bound": {
                "earliest_d_stale_step": EARLIEST_D_STALE_STEP,
                "final_absence_start_step": FINAL_ABSENCE_START_STEP,
                "available_slack_steps": RETIREMENT_SLACK_STEPS,
                "prompt_full_bank_steps": PROMPT_FULL_BANK_QUEUE_BOUND_STEPS,
                "legacy_full_bank_steps": LEGACY_FULL_BANK_QUEUE_BOUND_STEPS,
                "scope": (
                    "coarse conditional capacity only; not a prediction of descriptor "
                    "ordering, eligibility, or cadence phase"
                ),
            },
            "arms": [result.outcome_dict() for result in self.arms],
            "claim_scope": (
                "D-only development falsification; C and joint success are descriptive, "
                "and no outcome can promote scientific evidence"
            ),
        }


def _first_post_exit_d_event(lifecycle: CriticalLifecycleV2Summary) -> int | None:
    return next(
        (step for step in lifecycle.d_retirement_event_steps if step >= D_EXIT_STEP),
        None,
    )


def _d_requirement_failures(
    prompt: CriticalLifecycleV2Summary,
) -> tuple[str, ...]:
    prompt_event = _first_post_exit_d_event(prompt)
    checks = (
        ("prompt_d_task_learned", prompt.d_task_learned),
        ("prompt_d_retirement_event_aligned", prompt.d_retirement_event_aligned),
        (
            "prompt_d_linked_reset_count_is_one",
            prompt.d_linked_matching_candidate_reset_count == 1,
        ),
        (
            "prompt_d_linked_candidate_utility_is_positive_zero",
            prompt.d_linked_candidate_utility_post == 0.0,
        ),
        (
            "prompt_d_linked_candidate_head_is_positive_zero",
            prompt.d_linked_candidate_head_linf_post == 0.0,
        ),
        ("prompt_d_linked_candidate_age_is_zero", prompt.d_linked_candidate_age_post == 0),
        ("prompt_d_repromotions_is_zero", prompt.d_repromotions_after_retirement == 0),
        ("prompt_d_absent_entire_final_window", prompt.d_absent_entire_final_window),
        ("prompt_d_learned_then_stably_retired", prompt.d_learned_then_stably_retired),
        (
            "prompt_d_event_precedes_final_window",
            prompt_event is not None and prompt_event < FINAL_ABSENCE_START_STEP,
        ),
    )
    return tuple(name for name, passed in checks if not passed)


def run_retirement_throughput_development() -> RetirementThroughputPanelResult:
    """Run exactly one fixed seed for the prompt and legacy-throughput arms."""
    static_errors = validate_retirement_throughput_static_contract()
    if static_errors:
        raise RuntimeError(
            "invalid retirement-throughput development contract: "
            + "; ".join(static_errors)
        )
    arm_results: list[RetirementThroughputArmResult] = []
    for arm in RETIREMENT_THROUGHPUT_ARMS:
        condition = HiddenPartnerCondition(
            name="full",
            config=_base_agent_config(),
            isolated_question=arm.isolated_question,
        )
        runner = HiddenPartnerDevelopmentRunner(
            condition,
            RETIREMENT_THROUGHPUT_PROTOCOL,
        )
        if runner._compiled_life is not None:  # noqa: SLF001 - precompile safety audit
            raise RuntimeError("runner compiled before retirement mechanism installation")
        learner = _interaction_learner_for_arm(arm)
        # The development runner has no learner-factory seam. Installation is
        # deliberately performed before JIT lowering, then authenticated from
        # the public property and serialized config below.
        runner.agent._interaction = learner  # noqa: SLF001
        live_config = runner.agent.interaction_learner.to_config()
        if live_config != learner.to_config():
            raise RuntimeError("installed interaction mechanism is not observable")
        run = runner.run(RETIREMENT_THROUGHPUT_SEED)
        lifecycle = summarize_critical_lifecycle_v2(run)
        validation = validate_retirement_throughput_run(
            arm,
            live_config,
            run,
            lifecycle,
        )
        arm_results.append(
            RetirementThroughputArmResult(
                arm=arm,
                interaction_config=live_config,
                run=run,
                lifecycle=lifecycle,
                validation=validation,
            )
        )

    results = tuple(arm_results)
    by_name = {result.arm.name: result for result in results}
    prompt_lifecycle = by_name["prompt_retirement"].lifecycle
    legacy_lifecycle = by_name["legacy_throughput"].lifecycle
    failures = _d_requirement_failures(prompt_lifecycle)
    prompt_event = _first_post_exit_d_event(prompt_lifecycle)
    legacy_event = _first_post_exit_d_event(legacy_lifecycle)
    prompt_precedes_legacy = bool(
        prompt_event is not None
        and legacy_event is not None
        and prompt_event < legacy_event
    )
    final_absence_isolated = bool(
        prompt_lifecycle.d_absent_entire_final_window
        and not legacy_lifecycle.d_absent_entire_final_window
    )
    cadence_alone_falsified = bool(
        failures
        and not final_absence_isolated
    )
    if not all(result.validation.valid for result in results):
        status: PanelStatus = "invalid_development_run"
    elif failures:
        status = "valid_development_rejection"
    else:
        status = "passed_d_retirement_falsification"
    return RetirementThroughputPanelResult(
        schema_version=RETIREMENT_THROUGHPUT_DEVELOPMENT_SCHEMA,
        development_only=DEVELOPMENT_ONLY,
        scientific_promotion_allowed=SCIENTIFIC_PROMOTION_ALLOWED,
        output_writes_allowed=OUTPUT_WRITES_ALLOWED,
        status=status,
        d_requirement_failures=failures,
        prompt_retirement_precedes_legacy=prompt_precedes_legacy,
        final_absence_isolated_to_prompt=final_absence_isolated,
        cadence_alone_falsified_on_fixed_seed=cadence_alone_falsified,
        arms=results,
    )


__all__ = [
    "ARM_ORDER",
    "CYCLE_STEPS",
    "DEVELOPMENT_ONLY",
    "EARLIEST_D_STALE_STEP",
    "FINAL_ABSENCE_START_STEP",
    "LEGACY_FULL_BANK_QUEUE_BOUND_STEPS",
    "ORDINARY_REPLACEMENT_INTERVAL",
    "OUTPUT_WRITES_ALLOWED",
    "PROMPT_FULL_BANK_QUEUE_BOUND_STEPS",
    "PROMPT_RETIREMENT_INTERVAL",
    "RETENTION_GRACE_STEPS",
    "RETIREMENT_SLACK_STEPS",
    "RETIREMENT_THROUGHPUT_ARMS",
    "RETIREMENT_THROUGHPUT_DEVELOPMENT_SCHEMA",
    "RETIREMENT_THROUGHPUT_NAMESPACE",
    "RETIREMENT_THROUGHPUT_PROTOCOL",
    "RETIREMENT_THROUGHPUT_RESOURCE_CONTRACT",
    "RETIREMENT_THROUGHPUT_SEED",
    "SCIENTIFIC_PROMOTION_ALLOWED",
    "SEGMENT_LENGTHS",
    "RetirementThroughputArm",
    "RetirementThroughputArmResult",
    "RetirementThroughputPanelResult",
    "RetirementThroughputRunValidation",
    "build_retirement_throughput_arms",
    "run_retirement_throughput_development",
    "validate_retirement_throughput_run",
    "validate_retirement_throughput_static_contract",
]
