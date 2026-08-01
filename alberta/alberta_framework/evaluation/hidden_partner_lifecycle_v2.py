"""Development-only v5 lifecycle metrics and frozen confirmation plan.

This module closes the endpoint loopholes in the original hidden-partner
diagnostic.  Feature presence is indexed by the representation available for
decision ``t``.  A lifecycle event produced while processing transition ``t``
therefore first changes presence at decision ``t + 1``.

The exhaustive 66-pair archive is reported separately from the scarce
deployed bank.  Retirement means release of deployed capacity and reset of the
candidate's learned head; it does not erase the enumerated descriptor.
"""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import json
import math
import zlib
from collections.abc import Sequence
from typing import Any, cast

import numpy as np

from alberta_framework.core.integrated_hidden_partner import (
    IntegratedHiddenPartnerConfig,
)
from alberta_framework.evaluation.hidden_partner_development import (
    HiddenPartnerCondition,
    HiddenPartnerDevelopmentProtocol,
    HiddenPartnerDevelopmentRunner,
    HiddenPartnerRunResult,
    HiddenPartnerRunSummary,
    HiddenPartnerSeedPair,
    derive_hidden_partner_seed_pairs,
)

HIDDEN_PARTNER_LIFECYCLE_V2_SCHEMA = "alberta.hidden-partner-development.lifecycle.v4"
CRITICAL_RUN_PRIMITIVES_SCHEMA = "alberta.hidden-partner-development.critical-run-primitives.v3"
HIDDEN_PARTNER_LIFECYCLE_V5_SCHEMA = "alberta.hidden-partner-development.lifecycle.v5"
CRITICAL_RUN_PRIMITIVES_V5_SCHEMA = (
    "alberta.hidden-partner-development.critical-run-primitives.v4"
)
# Earlier v3 development seeds were inspected while diagnosing downstream
# consumer interference.  This fresh namespace is reserved for the frozen
# evidence-gated grid and must not be reused for adaptive debugging.
LEASE_TUNING_NAMESPACE = "hidden-partner-v0-dev-v4-evidence-gated-lease-grid-a-v1"
LEASE_TUNING_NAMESPACE_STATUS = "FORBIDDEN/UNEXECUTED"
LEASE_TUNING_SEED_COUNT = 8
CONFIRMATION_NAMESPACE = (
    "hidden-partner-v0-dev-v5-retired-reacquisition-confirmation-a-v1"
)
# The namespace was reserved but never executed.  A concurrent core change to
# conditional leave-one-out probes no longer satisfies this v5 protocol's
# target-only probe contract, so fail closed and forbid future execution.
CONFIRMATION_NAMESPACE_STATUS = "FORBIDDEN/UNEXECUTED"


def require_lease_tuning_execution_allowed() -> None:
    """Fail closed unless an intentionally executable namespace is declared.

    The current v4 namespace was reserved but never executed.  Its explicit
    status is load-bearing: callers must not derive its seeds, construct a
    runner, or issue an artifact while it remains forbidden.
    """
    if LEASE_TUNING_NAMESPACE_STATUS != "EXECUTABLE":
        raise RuntimeError(
            "hidden-partner lease tuning execution is forbidden: namespace "
            f"{LEASE_TUNING_NAMESPACE!r} has status "
            f"{LEASE_TUNING_NAMESPACE_STATUS!r}"
        )


TUNING_SELECTION_RULE: dict[str, object] = {
    "feasibility": (
        "all finite/contracts; mean reward >=0.85; minimum reward "
        ">=0.80; and at least 75% per-life joint C-retain/D-retire success"
    ),
    "lexicographic_maximum": [
        "joint_success_count",
        "minimum_seed_reward",
        "mean_reward",
        "negative_total_d_repromotions",
        "negative_median_d_retirement_latency_steps",
        "negative_cell_index",
    ],
    "no_feasible_cell_result": None,
}

FEATURE_LEARNING_WINDOW = 128
RETIREMENT_CONFIRMATION_WINDOW = 128
FINAL_ABSENCE_WINDOW = 256
RECURRENT_ENTRY_WINDOW = 128
CRITICAL_LATE_PREDICTION_ACCURACY_THRESHOLD = 0.80
CRITICAL_COLUMN_LEARNING_NLL_GAIN_THRESHOLD = 0.05
CRITICAL_COLUMN_LEARNING_POSITIVE_FRACTION_THRESHOLD = 0.55
CRITICAL_COLUMN_TARGET_CREATED_SHARE_THRESHOLD = 0.50
CRITICAL_MASKED_NLL_INCREASE_THRESHOLD = 0.005
CRITICAL_MASKED_NLL_POSITIVE_FRACTION_THRESHOLD = 0.55
RECURRENT_EARLY_REWARD_THRESHOLD = 0.75
INITIAL_LATE_REWARD_THRESHOLD = 0.75
RETENTION_RATIO_THRESHOLD = 0.80
MINIMUM_JOINT_SUCCESS_FRACTION = 0.75
CHANCE_REWARD = 0.50

LEASE_TUNING_SCOPE_LIMITS = (
    "scripted nonlearning partner",
    "hard-coded C and D identities",
    "fixed schedule with unequal recurrence gaps",
    "exhaustive enumerated pair archive",
    "interaction-task utility only",
    (
        "uncommitted interaction heads bootstrap plastically until first confirmation "
        "or confirmed-evidence lease expiry/retirement; only committed durable heads "
        "are confirmation-gated"
    ),
    "pre-confirmation bootstrap plasticity is an intentional interference risk",
    "confirmed pair-specific evidence gates behavior/control durable writes",
    (
        "raw one-step pair-specific evidence acquires downstream behavior/control "
        "reads under a fixed lease"
    ),
    "feature masking is a same-state diagnostic, not an executed intervention",
    "retirement releases deployed capacity but preserves the descriptor archive",
    "artifact sha256 detects drift but is not an external signature or publication timestamp",
    "development tuning seeds cannot promote evidence",
)

_C_PAIR = (0, 2)
_D_PAIR = (4, 5)

_V5_EXPECTED_RESOURCE_CONTRACT: dict[str, int] = {
    "raw_observation_dim": 8,
    "base_feature_dim": 12,
    "active_pair_slots": 12,
    "candidate_pair_slots": 66,
    "deployed_feature_dim": 24,
    "state_builder_nbytes": 2_108,
    "interaction_nbytes": 3_330,
    "interaction_evidence_idle_nbytes": 48,
    "interaction_utility_evidence_streak_nbytes": 48,
    "interaction_active_output_memory_committed_nbytes": 12,
    "interaction_relevance_probe_nbytes": 52,
    "interaction_relevance_probe_bias_nbytes": 4,
    "interaction_candidate_promotion_evidence_streak_nbytes": 264,
    "interaction_candidate_reacquisition_required_nbytes": 66,
    "behavior_nbytes": 224,
    "joint_world_nbytes": 52,
    "grounded_world_nbytes": 0,
    "grounded_world_parameter_count": 0,
    "grounded_world_parameters_touched_per_update": 0,
    "grounded_world_update_counter_nbytes": 0,
    "control_nbytes": 528,
    "router_nbytes": 104,
    "consumer_active_mask_nbytes": 12,
    "consumer_evidence_streak_nbytes": 48,
    "consumer_read_idle_steps_nbytes": 48,
    "decision_cache_nbytes": 303,
    "total_state_nbytes": 6_757,
    "legacy_joint_world_cells_per_decision": 4,
    "grounded_world_joint_cells_per_decision": 0,
    "planner_cell_evaluations_per_decision": 4,
    "replay_capacity": 0,
}


@dataclasses.dataclass(frozen=True)
class EvidenceLeaseTuningCell:
    """One immutable cell in the reserved v4 confirmed-memory grid."""

    index: int
    grace_steps: int
    evidence_threshold: float
    candidate_promotion_floor: float = 0.01

    def agent_config(self) -> IntegratedHiddenPartnerConfig:
        return IntegratedHiddenPartnerConfig(
            active_utility_retention_grace_steps=self.grace_steps,
            active_utility_evidence_threshold=self.evidence_threshold,
            retire_stale_features=True,
            candidate_promotion_floor=self.candidate_promotion_floor,
            evidence_gated_feature_memory=True,
            feature_evidence_confirmation_steps=8,
            evidence_gated_consumer_memory=True,
            consumer_evidence_confirmation_steps=8,
            consumer_read_confirmation_steps=1,
            consumer_read_lease_steps=32,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "grace_steps": self.grace_steps,
            "evidence_threshold": self.evidence_threshold,
            "candidate_promotion_floor": self.candidate_promotion_floor,
            "agent_config": self.agent_config().to_config(),
        }


LEASE_TUNING_GRID: tuple[EvidenceLeaseTuningCell, ...] = tuple(
    EvidenceLeaseTuningCell(index=index, grace_steps=grace, evidence_threshold=threshold)
    for index, (grace, threshold) in enumerate(
        (
            (2_048, 0.075),
            (2_048, 0.10),
            (3_072, 0.10),
            (3_072, 0.15),
            (3_072, 0.20),
            (4_096, 0.10),
            (4_096, 0.20),
            (4_096, 0.40),
        )
    )
)


@dataclasses.dataclass(frozen=True)
class ReservedConfirmationCandidate:
    """The one frozen v5 candidate; this record is a plan, never a run."""

    grace_steps: int = 4_096
    evidence_threshold: float = 0.1
    candidate_promotion_floor: float = 0.1
    feature_confirmation_steps: int = 24
    candidate_promotion_confirmation_steps: int = 1
    candidate_reacquisition_confirmation_steps: int = 8
    consumer_write_confirmation_steps: int = 12
    consumer_read_confirmation_steps: int = 4
    consumer_read_lease_steps: int = 4

    def agent_config(self) -> IntegratedHiddenPartnerConfig:
        """Return the frozen development configuration without executing it."""
        return IntegratedHiddenPartnerConfig(
            active_utility_retention_grace_steps=self.grace_steps,
            active_utility_evidence_threshold=self.evidence_threshold,
            retire_stale_features=True,
            candidate_promotion_floor=self.candidate_promotion_floor,
            evidence_gated_feature_memory=True,
            feature_evidence_confirmation_steps=self.feature_confirmation_steps,
            independent_relevance_probe=True,
            candidate_promotion_confirmation_steps=(
                self.candidate_promotion_confirmation_steps
            ),
            candidate_reacquisition_confirmation_steps=(
                self.candidate_reacquisition_confirmation_steps
            ),
            evidence_gated_consumer_memory=True,
            consumer_evidence_confirmation_steps=(
                self.consumer_write_confirmation_steps
            ),
            consumer_read_confirmation_steps=self.consumer_read_confirmation_steps,
            consumer_read_lease_steps=self.consumer_read_lease_steps,
        )

    def to_dict(self) -> dict[str, object]:
        """Return the exact frozen, explicitly unexecuted plan record."""
        return {
            "namespace": CONFIRMATION_NAMESPACE,
            "namespace_status": CONFIRMATION_NAMESPACE_STATUS,
            "development_only": True,
            "scientific_promotion_allowed": False,
            "candidate_count": 1,
            "agent_config": self.agent_config().to_config(),
        }


RESERVED_CONFIRMATION_CANDIDATES: tuple[ReservedConfirmationCandidate, ...] = (
    ReservedConfirmationCandidate(),
)
RESERVED_CONFIRMATION_CONTROL = dataclasses.replace(
    RESERVED_CONFIRMATION_CANDIDATES[0],
    candidate_reacquisition_confirmation_steps=1,
)


@dataclasses.dataclass(frozen=True)
class CriticalPairLifecycleInterval:
    """Canonical half-open run of critical-pair bank locations."""

    start: int
    end_exclusive: int
    deployed_slot: int
    shadow_slot: int
    candidate_slot: int

    def to_dict(self) -> dict[str, int]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class CriticalLifecycleV2Summary:
    """Continuous C/D lifecycle metrics with v4 causal memory contracts."""

    cycle_steps: int
    decision_state_count: int
    representation_link_contract_valid: bool
    consumer_gate_contract_valid: bool
    feature_memory_enabled: bool
    feature_memory_contract_valid: bool
    c_shadow_deployed_mismatch_steps: int
    d_shadow_deployed_mismatch_steps: int
    c_promotion_event_steps: tuple[int, ...]
    c_target_evidence_refresh_steps: tuple[int, ...]
    c_acquisition_step: int | None
    c_first_late_reward: float
    c_first_late_intended_accuracy: float
    c_first_late_online_nll: float
    c_first_late_entry_frozen_critical_nll: float
    c_critical_column_learning_nll_gain: float
    c_critical_column_learning_positive_fraction: float
    c_critical_column_target_created_share: float
    c_first_late_entry_frozen_critical_accuracy: float
    c_critical_column_learning_accuracy_gain: float
    c_first_late_masked_nll_increase: float
    c_first_late_masked_nll_positive_fraction: float
    c_task_learned: bool
    c_survival_end_exclusive: int
    c_survival_gap_steps: int | None
    c_first_survival_gap_step: int | None
    c_evictions_after_acquisition: int
    c_repromotions_after_acquisition: int
    c_continuously_survived: bool
    c_recurrent_early_reward: float
    c_recurrent_early_excess_reward_retention: float
    c_recurrent_early_intended_accuracy: float
    c_recurrent_early_masked_nll_increase: float
    c_recurrent_early_masked_nll_positive_fraction: float
    c_retained_and_used: bool
    d_promotion_event_steps: tuple[int, ...]
    d_target_evidence_refresh_steps: tuple[int, ...]
    d_acquisition_step: int | None
    d_deployed_through_exit: bool
    d_late_reward: float
    d_late_intended_accuracy: float
    d_late_online_nll: float
    d_late_entry_frozen_critical_nll: float
    d_critical_column_learning_nll_gain: float
    d_critical_column_learning_positive_fraction: float
    d_critical_column_target_created_share: float
    d_late_entry_frozen_critical_accuracy: float
    d_critical_column_learning_accuracy_gain: float
    d_late_masked_nll_increase: float
    d_late_masked_nll_positive_fraction: float
    d_task_learned: bool
    d_retirement_event_step: int | None
    d_retirement_step: int | None
    d_retirement_event_latency_steps: int | None
    d_retirement_latency_steps: int | None
    d_post_exit_live_slot_steps: int
    d_post_exit_live_fraction: float
    d_post_exit_promotion_count: int
    d_repromotions_after_retirement: int
    d_absent_entire_final_window: bool
    d_retirement_event_steps: tuple[int, ...]
    d_retirement_event_reset_counts: tuple[int, ...]
    d_retirement_event_candidate_utility_post: tuple[float, ...]
    d_retirement_event_candidate_head_linf_post: tuple[float, ...]
    d_retirement_event_candidate_age_post: tuple[int, ...]
    d_retirement_event_count: int
    d_matching_candidate_reset_count: int
    d_linked_matching_candidate_reset_count: int | None
    d_linked_candidate_utility_post: float | None
    d_linked_candidate_head_linf_post: float | None
    d_linked_candidate_age_post: int | None
    d_retirement_event_aligned: bool
    d_learned_then_stably_retired: bool
    joint_memory_management_success: bool
    candidate_archive_contract_valid: bool
    c_candidate_utility_at_life_end: float
    d_candidate_utility_at_life_end: float
    c_lifecycle_rle: tuple[CriticalPairLifecycleInterval, ...]
    d_lifecycle_rle: tuple[CriticalPairLifecycleInterval, ...]

    def to_dict(self) -> dict[str, object]:
        payload = dataclasses.asdict(self)
        for field in (
            "c_promotion_event_steps",
            "c_target_evidence_refresh_steps",
            "d_promotion_event_steps",
            "d_target_evidence_refresh_steps",
            "d_retirement_event_steps",
            "d_retirement_event_reset_counts",
            "d_retirement_event_candidate_utility_post",
            "d_retirement_event_candidate_head_linf_post",
            "d_retirement_event_candidate_age_post",
        ):
            payload[field] = list(payload[field])
        payload["c_lifecycle_rle"] = [interval.to_dict() for interval in self.c_lifecycle_rle]
        payload["d_lifecycle_rle"] = [interval.to_dict() for interval in self.d_lifecycle_rle]
        return payload


def _descriptor_slots(
    descriptors: np.ndarray,
    pair: tuple[int, int],
    *,
    require_exactly_one: bool,
) -> np.ndarray:
    target = np.asarray(pair, dtype=np.int32)
    matches = np.all(descriptors == target, axis=-1)
    counts = np.sum(matches, axis=1)
    if np.any(counts > 1):
        raise ValueError(f"descriptor {pair!r} appears in multiple slots")
    if require_exactly_one and np.any(counts != 1):
        raise ValueError(f"candidate archive does not contain exactly one {pair!r}")
    return np.where(counts == 1, np.argmax(matches, axis=1), -1).astype(np.int32)


def _descriptor_live_mask(descriptors: np.ndarray) -> np.ndarray:
    left = descriptors[..., 0]
    right = descriptors[..., 1]
    return (left >= 0) & (right >= 0) & (left < 12) & (right < 12) & (left < right)


@dataclasses.dataclass(frozen=True)
class _ConsumerGateContractAudit:
    """Exact read/write audit plus per-slot durable-write violations."""

    valid: bool
    write_violation_bits: np.ndarray


def _invalid_consumer_gate_audit(steps: int) -> _ConsumerGateContractAudit:
    safe_steps = max(steps, 0)
    return _ConsumerGateContractAudit(
        valid=False,
        write_violation_bits=np.ones((safe_steps, 12), dtype=np.bool_),
    )


def _consumer_gate_contract_audit(
    deployed_pre: np.ndarray,
    deployed_post: np.ndarray,
    evidence: np.ndarray,
    write_gate: np.ndarray,
    read_idle_pre: np.ndarray,
    read_idle_post: np.ndarray,
    mask_pre: np.ndarray,
    mask_post: np.ndarray,
    representations: np.ndarray,
    behavior_weights_pre: np.ndarray,
    behavior_weights_post: np.ndarray,
    control_q_weights_pre: np.ndarray,
    control_q_weights_post: np.ndarray,
    control_q_trace_pre: np.ndarray,
    control_q_trace_post: np.ndarray,
    config: IntegratedHiddenPartnerConfig,
) -> _ConsumerGateContractAudit:
    """Reconstruct the gate and audit every persisted pair-column write.

    Durable columns are compared by descriptor identity across the curation
    transaction.  A closed confirmed-write gate must preserve behavior and Q
    columns bit-for-bit, while its eligibility trace is erased.  A new
    identity or vacancy starts at exact positive-zero bits.  With the gate
    disabled, survivor writes remain unrestricted, but finite-state,
    continuity, routing, and new-zero invariants still apply.
    """
    steps = deployed_pre.shape[0] if deployed_pre.ndim >= 1 else 0
    expected_shape = (steps, 12)
    if any(
        values.shape != expected_shape
        for values in (
            evidence,
            write_gate,
            read_idle_pre,
            read_idle_post,
            mask_pre,
            mask_post,
            representations,
        )
    ) or deployed_pre.shape != (steps, 12, 2) or deployed_post.shape != (
        steps,
        12,
        2,
    ) or any(
        values.shape != (steps, 2, 12)
        for values in (
            behavior_weights_pre,
            behavior_weights_post,
            control_q_weights_pre,
            control_q_weights_post,
            control_q_trace_pre,
            control_q_trace_post,
        )
    ):
        return _invalid_consumer_gate_audit(steps)

    write_violations = np.zeros(expected_shape, dtype=np.bool_)
    vacancy_pre = np.all(deployed_pre == -1, axis=2)
    vacancy_post = np.all(deployed_post == -1, axis=2)
    live_pre = _descriptor_live_mask(deployed_pre)
    live_post = _descriptor_live_mask(deployed_post)
    structural_valid = bool(
        np.all(live_pre | vacancy_pre) and np.all(live_post | vacancy_post)
    )
    for states in (deployed_pre, deployed_post):
        for bank in states:
            descriptors = [tuple(int(value) for value in pair) for pair in bank if pair[0] >= 0]
            if len(descriptors) != len(set(descriptors)):
                structural_valid = False
    linked_descriptors = bool(
        steps == 1 or np.array_equal(deployed_pre[1:], deployed_post[:-1])
    )
    linked_masks = bool(steps == 1 or np.array_equal(mask_pre[1:], mask_post[:-1]))
    finite_representations = bool(np.all(np.isfinite(representations)))
    closed_reads_zero = bool(
        finite_representations and np.all(representations[~mask_pre] == 0.0)
    )

    pre_codes = deployed_pre[..., 0] * 12 + deployed_pre[..., 1]
    post_codes = deployed_post[..., 0] * 12 + deployed_post[..., 1]
    identity_matches = (
        (post_codes[:, :, None] == pre_codes[:, None, :])
        & live_post[:, :, None]
        & live_pre[:, None, :]
    )
    post_has_source = np.any(identity_matches, axis=2)
    destination_for_pre = np.argmax(identity_matches, axis=1)
    pre_has_destination = np.any(identity_matches, axis=1)

    consumer_arrays = (
        behavior_weights_pre,
        behavior_weights_post,
        control_q_weights_pre,
        control_q_weights_post,
        control_q_trace_pre,
        control_q_trace_post,
    )
    consumer_bits = tuple(_float32_bits(values) for values in consumer_arrays)
    for values in consumer_arrays:
        write_violations |= np.any(~np.isfinite(values), axis=1)

    # Every initial pair column is newly initialized; vacancies remain exact
    # zero.  Post-curation identities without a pre-state source also begin at
    # zero rather than inheriting the physical slot's old contents.
    if steps > 0:
        for pre_bits in consumer_bits[::2]:
            write_violations[0] |= np.any(pre_bits[0] != 0, axis=0)
    no_source_post = ~post_has_source
    for post_bits in consumer_bits[1::2]:
        write_violations |= no_source_post & np.any(post_bits != 0, axis=1)
    for pre_bits in consumer_bits[::2]:
        write_violations |= vacancy_pre & np.any(pre_bits != 0, axis=1)

    # A state sequence is exact only when every next pre-state is the previous
    # post-state bit-for-bit, including signed zero and NaN payload bits.
    if steps > 1:
        for pre_bits, post_bits in zip(
            consumer_bits[::2],
            consumer_bits[1::2],
            strict=True,
        ):
            write_violations[1:] |= np.any(
                pre_bits[1:] != post_bits[:-1],
                axis=1,
            )

    if config.evidence_gated_consumer_memory:
        closed_survivor = live_pre & pre_has_destination & ~write_gate
        for pre_bits, post_bits in (
            (consumer_bits[0], consumer_bits[1]),
            (consumer_bits[2], consumer_bits[3]),
        ):
            routed_post = np.take_along_axis(
                post_bits,
                destination_for_pre[:, None, :],
                axis=2,
            )
            write_violations |= closed_survivor & np.any(
                pre_bits != routed_post,
                axis=1,
            )
        routed_trace_post = np.take_along_axis(
            consumer_bits[5],
            destination_for_pre[:, None, :],
            axis=2,
        )
        write_violations |= closed_survivor & np.any(
            routed_trace_post != 0,
            axis=1,
        )

    structural_valid = bool(
        structural_valid
        and linked_descriptors
        and linked_masks
        and closed_reads_zero
    )
    if not config.evidence_gated_consumer_memory:
        gate_valid = bool(
            np.all(write_gate)
            and not np.any(read_idle_pre)
            and not np.any(read_idle_post)
            and np.array_equal(mask_pre, live_pre)
            and np.array_equal(mask_post, live_post)
        )
        return _ConsumerGateContractAudit(
            valid=bool(structural_valid and gate_valid and not np.any(write_violations)),
            write_violation_bits=write_violations,
        )
    if np.any(evidence & ~live_pre) or (steps > 0 and np.any(mask_pre[0])):
        structural_valid = False

    int32_max = np.iinfo(np.int32).max
    current_streak = np.zeros((12,), dtype=np.int32)
    current_idle = np.zeros((12,), dtype=np.int32)
    current_mask = np.zeros((12,), dtype=np.bool_)
    expected_write = np.zeros(expected_shape, dtype=np.bool_)
    expected_idle_pre = np.zeros(expected_shape, dtype=np.int32)
    expected_idle_post = np.zeros(expected_shape, dtype=np.int32)
    expected_mask_pre = np.zeros(expected_shape, dtype=np.bool_)
    expected_mask_post = np.zeros(expected_shape, dtype=np.bool_)
    for step in range(steps):
        expected_mask_pre[step] = current_mask
        expected_idle_pre[step] = current_idle
        incremented_streak = np.minimum(
            np.maximum(current_streak, 0),
            int32_max - 1,
        ) + 1
        updated_streak = np.where(
            live_pre[step] & evidence[step],
            incremented_streak,
            0,
        ).astype(np.int32)
        confirmed_write = (
            live_pre[step]
            & evidence[step]
            & (
                updated_streak
                >= config.consumer_evidence_confirmation_steps
            )
        )
        read_acquire = (
            live_pre[step]
            & evidence[step]
            & (
                updated_streak
                >= config.consumer_read_confirmation_steps
            )
        )
        expected_write[step] = confirmed_write
        incremented_idle = np.minimum(
            np.maximum(current_idle, 0),
            int32_max - 1,
        ) + 1
        updated_idle = np.where(
            live_pre[step],
            np.where(evidence[step], 0, incremented_idle),
            0,
        ).astype(np.int32)

        next_streak = np.zeros((12,), dtype=np.int32)
        next_idle = np.zeros((12,), dtype=np.int32)
        next_mask = np.zeros((12,), dtype=np.bool_)
        for post_slot, descriptor in enumerate(deployed_post[step]):
            if not live_post[step, post_slot]:
                continue
            matches = np.flatnonzero(
                np.all(deployed_pre[step] == descriptor, axis=1)
            )
            if matches.size != 1:
                continue
            source = int(matches[0])
            next_streak[post_slot] = updated_streak[source]
            next_idle[post_slot] = updated_idle[source]
            next_mask[post_slot] = bool(
                (current_mask[source] or read_acquire[source])
                and next_idle[post_slot] <= config.consumer_read_lease_steps
            )
        expected_mask_post[step] = next_mask
        expected_idle_post[step] = next_idle
        current_streak = next_streak
        current_idle = next_idle
        current_mask = next_mask

    gate_valid = bool(
        np.array_equal(write_gate, expected_write)
        and np.array_equal(read_idle_pre, expected_idle_pre)
        and np.array_equal(read_idle_post, expected_idle_post)
        and np.array_equal(mask_pre, expected_mask_pre)
        and np.array_equal(mask_post, expected_mask_post)
    )
    return _ConsumerGateContractAudit(
        valid=bool(structural_valid and gate_valid and not np.any(write_violations)),
        write_violation_bits=write_violations,
    )


def _consumer_gate_contract_valid(
    deployed_pre: np.ndarray,
    deployed_post: np.ndarray,
    evidence: np.ndarray,
    write_gate: np.ndarray,
    read_idle_pre: np.ndarray,
    read_idle_post: np.ndarray,
    mask_pre: np.ndarray,
    mask_post: np.ndarray,
    representations: np.ndarray,
    behavior_weights_pre: np.ndarray,
    behavior_weights_post: np.ndarray,
    control_q_weights_pre: np.ndarray,
    control_q_weights_post: np.ndarray,
    control_q_trace_pre: np.ndarray,
    control_q_trace_post: np.ndarray,
    config: IntegratedHiddenPartnerConfig,
) -> bool:
    """Return the fail-closed consumer read/write contract predicate."""
    return _consumer_gate_contract_audit(
        deployed_pre,
        deployed_post,
        evidence,
        write_gate,
        read_idle_pre,
        read_idle_post,
        mask_pre,
        mask_post,
        representations,
        behavior_weights_pre,
        behavior_weights_post,
        control_q_weights_pre,
        control_q_weights_post,
        control_q_trace_pre,
        control_q_trace_post,
        config,
    ).valid


@dataclasses.dataclass(frozen=True)
class _FeatureMemoryContractAudit:
    """Exact causal audit plus compact per-slot violation primitives."""

    valid: bool
    identity_routed_head_changed: np.ndarray
    violation_bits: np.ndarray


@dataclasses.dataclass(frozen=True)
class HiddenPartnerLifecycleV5Audit:
    """Fail-closed host reconstruction for the reserved v5 mechanism."""

    schema_version: str
    development_only: bool
    scientific_promotion_allowed: bool
    config_role: str | None
    probe_contract_valid: bool
    candidate_reacquisition_contract_valid: bool
    durable_memory_contract_valid: bool
    lifecycle_contract_valid: bool
    resource_contract_valid: bool
    all_contracts_valid: bool
    feature_violation_bits: np.ndarray
    candidate_violation_bits: np.ndarray
    step_violation_bits: np.ndarray

    def to_dict(self) -> dict[str, object]:
        """Return compact JSON-compatible v5 audit metadata and bit payloads."""
        return {
            "schema_version": self.schema_version,
            "development_only": self.development_only,
            "scientific_promotion_allowed": self.scientific_promotion_allowed,
            "config_role": self.config_role,
            "probe_contract_valid": self.probe_contract_valid,
            "candidate_reacquisition_contract_valid": (
                self.candidate_reacquisition_contract_valid
            ),
            "durable_memory_contract_valid": self.durable_memory_contract_valid,
            "lifecycle_contract_valid": self.lifecycle_contract_valid,
            "resource_contract_valid": self.resource_contract_valid,
            "all_contracts_valid": self.all_contracts_valid,
            "feature_violation_bits": _packed_bool_payload(
                self.feature_violation_bits
            ),
            "candidate_violation_bits": _packed_bool_payload(
                self.candidate_violation_bits
            ),
            "step_violation_bits": _packed_bool_payload(self.step_violation_bits),
        }


def _v5_config_role(config: IntegratedHiddenPartnerConfig) -> str | None:
    serialized = config.to_config()
    candidate = RESERVED_CONFIRMATION_CANDIDATES[0].agent_config().to_config()
    control = RESERVED_CONFIRMATION_CONTROL.agent_config().to_config()
    if serialized == candidate:
        return "frozen_candidate"
    if serialized == control:
        return "matched_reacquisition_one_control"
    return None


def _v5_resource_contract_valid(result: HiddenPartnerRunResult) -> bool:
    """Pin every scalar and component of the frozen v5 resource budget."""
    expected = _V5_EXPECTED_RESOURCE_CONTRACT
    initial = result.initial_resource.to_dict()
    final = result.final_resource.to_dict()
    return bool(
        initial == expected
        and final == expected
        and result.summary.initial_state_nbytes == expected["total_state_nbytes"]
        and result.summary.final_state_nbytes == expected["total_state_nbytes"]
        and result.summary.resource_shape_matched
    )


def _float32_matches(
    actual: np.ndarray,
    *expected_alternatives: np.ndarray,
) -> np.ndarray:
    """Return elementwise exact matches for staged or justified FMA paths."""
    value = np.ascontiguousarray(actual, dtype=np.float32)
    matches = np.zeros(value.shape, dtype=np.bool_)
    for expected in expected_alternatives:
        candidate = np.ascontiguousarray(expected, dtype=np.float32)
        if candidate.shape != value.shape:
            continue
        matches |= value.view(np.uint32) == candidate.view(np.uint32)
    return np.isfinite(value) & matches


def _float32_ema_paths(
    old: np.ndarray,
    values: np.ndarray,
    decay: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return staged-float32 and single-rounding multiply-add EMA paths."""
    old32 = np.asarray(old, dtype=np.float32)
    values32 = np.asarray(values, dtype=np.float32)
    decay32 = np.float32(decay)
    one_minus32 = np.float32(1.0 - decay)
    squared = np.asarray(values32 * values32, dtype=np.float32)
    staged = np.asarray(
        np.asarray(decay32 * old32, dtype=np.float32)
        + np.asarray(one_minus32 * squared, dtype=np.float32),
        dtype=np.float32,
    )
    fused = np.asarray(
        np.asarray(decay32, dtype=np.float64) * old32.astype(np.float64)
        + np.asarray(one_minus32, dtype=np.float64) * squared.astype(np.float64),
        dtype=np.float32,
    )
    return staged, fused


def _float32_linear_ema_paths(
    old: np.ndarray,
    signal: np.ndarray,
    decay: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact staged and single-rounding linear EMA alternatives."""
    old32 = np.asarray(old, dtype=np.float32)
    signal32 = np.asarray(signal, dtype=np.float32)
    decay32 = np.float32(decay)
    one_minus32 = np.float32(1.0 - decay)
    staged = np.asarray(
        np.asarray(decay32 * old32, dtype=np.float32)
        + np.asarray(one_minus32 * signal32, dtype=np.float32),
        dtype=np.float32,
    )
    fused = np.asarray(
        np.asarray(decay32, dtype=np.float64) * old32.astype(np.float64)
        + np.asarray(one_minus32, dtype=np.float64) * signal32.astype(np.float64),
        dtype=np.float32,
    )
    return staged, fused


def _pair_values_host(descriptors: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Reconstruct fixed-bank pair products without trusting traced features."""
    live = _descriptor_live_mask(descriptors)
    left = np.clip(descriptors[..., 0], 0, phi.shape[1] - 1)
    right = np.clip(descriptors[..., 1], 0, phi.shape[1] - 1)
    rows = np.arange(phi.shape[0], dtype=np.int64)[:, None]
    products = np.asarray(
        phi[rows, left] * phi[rows, right],
        dtype=np.float32,
    )
    return np.where(live, products, np.float32(0.0)).astype(np.float32)


def _probe_reference_path(
    *,
    phi: np.ndarray,
    targets: np.ndarray,
    feature_values: np.ndarray,
    candidate_values: np.ndarray,
    probe_weights: np.ndarray,
    probe_biases: np.ndarray,
    candidate_weights: np.ndarray,
    feature_moments: np.ndarray,
    candidate_moments: np.ndarray,
    target_moments: np.ndarray,
    step_size: float,
    fused: bool,
) -> tuple[np.ndarray, ...]:
    """Independent one-task target-only probe and candidate recurrence."""
    del phi
    step32 = np.float32(step_size)
    epsilon = np.float32(1e-6)
    baseline = np.asarray(targets - probe_biases, dtype=np.float32)
    target_scale = np.asarray(
        np.sqrt(np.maximum(target_moments, epsilon)),
        dtype=np.float32,
    )
    feature_bank_energy = np.asarray(
        np.mean(feature_values * feature_values, axis=1),
        dtype=np.float32,
    )
    feature_denominator = np.maximum(
        np.maximum(
            np.maximum(feature_moments, feature_values * feature_values),
            feature_bank_energy[:, None],
        ),
        epsilon,
    ).astype(np.float32)
    candidate_bank_energy = np.asarray(
        np.mean(candidate_values * candidate_values, axis=1),
        dtype=np.float32,
    )
    candidate_denominator = np.maximum(
        np.maximum(
            np.maximum(candidate_moments, candidate_values * candidate_values),
            candidate_bank_energy[:, None],
        ),
        epsilon,
    ).astype(np.float32)

    feature_contribution = np.asarray(
        probe_weights[:, 0, :] * feature_values,
        dtype=np.float32,
    )
    candidate_contribution = np.asarray(
        candidate_weights[:, 0, :] * candidate_values,
        dtype=np.float32,
    )
    normalized_error = np.asarray(
        baseline / target_scale,
        dtype=np.float32,
    )
    normalized_feature = np.asarray(
        feature_contribution / target_scale,
        dtype=np.float32,
    )
    normalized_candidate = np.asarray(
        candidate_contribution / target_scale,
        dtype=np.float32,
    )
    normalized_error = np.clip(
        np.nan_to_num(normalized_error, nan=0.0, posinf=1e6, neginf=-1e6),
        -1e6,
        1e6,
    ).astype(np.float32)
    normalized_feature = np.clip(
        np.nan_to_num(normalized_feature, nan=0.0, posinf=1e6, neginf=-1e6),
        -1e6,
        1e6,
    ).astype(np.float32)
    normalized_candidate = np.clip(
        np.nan_to_num(normalized_candidate, nan=0.0, posinf=1e6, neginf=-1e6),
        -1e6,
        1e6,
    ).astype(np.float32)

    if fused:
        feature_gain = np.asarray(
            normalized_error.astype(np.float64)
            * normalized_feature.astype(np.float64)
            - 0.5 * normalized_feature.astype(np.float64) ** 2,
            dtype=np.float32,
        )
        candidate_gain = np.asarray(
            normalized_error.astype(np.float64)
            * normalized_candidate.astype(np.float64)
            - 0.5 * normalized_candidate.astype(np.float64) ** 2,
            dtype=np.float32,
        )
    else:
        feature_gain = np.asarray(
            np.asarray(normalized_error * normalized_feature, dtype=np.float32)
            - np.asarray(
                np.float32(0.5)
                * np.asarray(normalized_feature * normalized_feature, dtype=np.float32),
                dtype=np.float32,
            ),
            dtype=np.float32,
        )
        candidate_gain = np.asarray(
            np.asarray(normalized_error * normalized_candidate, dtype=np.float32)
            - np.asarray(
                np.float32(0.5)
                * np.asarray(
                    normalized_candidate * normalized_candidate,
                    dtype=np.float32,
                ),
                dtype=np.float32,
            ),
            dtype=np.float32,
        )
    feature_gain = np.maximum(feature_gain, np.float32(0.0)).astype(np.float32)
    candidate_gain = np.maximum(candidate_gain, np.float32(0.0)).astype(np.float32)
    feature_scores = np.asarray(
        feature_gain / np.asarray(np.float32(1.0) + feature_gain, dtype=np.float32),
        dtype=np.float32,
    )
    candidate_signal = np.asarray(
        candidate_gain
        / np.asarray(np.float32(1.0) + candidate_gain, dtype=np.float32),
        dtype=np.float32,
    )

    probe_error = np.asarray(
        baseline - feature_contribution,
        dtype=np.float32,
    )
    candidate_error = np.asarray(
        baseline - candidate_contribution,
        dtype=np.float32,
    )
    if fused:
        probe_delta = np.asarray(
            float(step32)
            * probe_error.astype(np.float64)
            * feature_values.astype(np.float64)
            / feature_denominator.astype(np.float64),
            dtype=np.float32,
        )
        candidate_delta = np.asarray(
            float(step32)
            * candidate_error.astype(np.float64)
            * candidate_values.astype(np.float64)
            / candidate_denominator.astype(np.float64),
            dtype=np.float32,
        )
        updated_probe = np.asarray(
            probe_weights[:, 0, :].astype(np.float64)
            + probe_delta.astype(np.float64),
            dtype=np.float32,
        )
        updated_candidate = np.asarray(
            candidate_weights[:, 0, :].astype(np.float64)
            + candidate_delta.astype(np.float64),
            dtype=np.float32,
        )
        updated_bias = np.asarray(
            probe_biases.astype(np.float64)
            + float(step32) * baseline.astype(np.float64),
            dtype=np.float32,
        )
    else:
        probe_delta = np.asarray(
            np.asarray(
                np.asarray(step32 * probe_error, dtype=np.float32)
                * feature_values,
                dtype=np.float32,
            )
            / feature_denominator,
            dtype=np.float32,
        )
        candidate_delta = np.asarray(
            np.asarray(
                np.asarray(step32 * candidate_error, dtype=np.float32)
                * candidate_values,
                dtype=np.float32,
            )
            / candidate_denominator,
            dtype=np.float32,
        )
        updated_probe = np.asarray(
            probe_weights[:, 0, :] + probe_delta,
            dtype=np.float32,
        )
        updated_candidate = np.asarray(
            candidate_weights[:, 0, :] + candidate_delta,
            dtype=np.float32,
        )
        updated_bias = np.asarray(
            probe_biases + np.asarray(step32 * baseline, dtype=np.float32),
            dtype=np.float32,
        )
    return (
        baseline,
        feature_scores,
        candidate_signal,
        updated_probe[:, None, :],
        updated_bias,
        updated_candidate[:, None, :],
    )


def _invalid_v5_audit(steps: int, role: str | None) -> HiddenPartnerLifecycleV5Audit:
    safe_steps = max(steps, 0)
    return HiddenPartnerLifecycleV5Audit(
        schema_version=HIDDEN_PARTNER_LIFECYCLE_V5_SCHEMA,
        development_only=True,
        scientific_promotion_allowed=False,
        config_role=role,
        probe_contract_valid=False,
        candidate_reacquisition_contract_valid=False,
        durable_memory_contract_valid=False,
        lifecycle_contract_valid=False,
        resource_contract_valid=False,
        all_contracts_valid=False,
        feature_violation_bits=np.ones((safe_steps, 12), dtype=np.bool_),
        candidate_violation_bits=np.ones((safe_steps, 66), dtype=np.bool_),
        step_violation_bits=np.ones((safe_steps,), dtype=np.bool_),
    )


def audit_hidden_partner_lifecycle_v5(
    result: HiddenPartnerRunResult,
) -> HiddenPartnerLifecycleV5Audit:
    """Independently reconstruct every frozen v5 probe/lifecycle recurrence.

    This host audit consumes primitive pre/post arrays, not producer contract
    flags.  It accepts only the one frozen candidate or its unexecuted matched
    reacquisition-one control.  It never derives a reserved seed or constructs
    a runner.
    """
    steps = result.summary.cycle_steps
    role = _v5_config_role(result.condition.config)
    if steps <= 0 or role is None:
        return _invalid_v5_audit(steps, role)
    trace = result.trace

    raw_active = np.asarray(trace.active)
    active_contract_valid = bool(
        raw_active.dtype == np.dtype(np.bool_)
        and raw_active.ndim == 1
        and steps <= raw_active.shape[0]
        and np.all(raw_active[:steps])
        and not np.any(raw_active[steps:])
        and np.count_nonzero(raw_active) == steps
    )
    if not active_contract_valid:
        return _invalid_v5_audit(steps, role)

    def exact_array(
        field: str,
        dtype: np.dtype[Any] | type[Any],
        tail: tuple[int, ...],
    ) -> np.ndarray:
        raw = np.asarray(getattr(trace, field))
        expected_dtype = np.dtype(dtype)
        if raw.dtype != expected_dtype or raw.shape != (
            result.trace.active.shape[0],
            *tail,
        ):
            raise ValueError(f"{field} has a noncanonical dtype or shape")
        return np.ascontiguousarray(raw[:steps])

    try:
        phi = exact_array("interaction_phi_pre", np.float32, (12,))
        targets = exact_array("interaction_target", np.float32, (1,))
        shadow_pre = exact_array("shadow_descriptors_pre", np.int32, (12, 2))
        shadow_post = exact_array("shadow_descriptors_post", np.int32, (12, 2))
        deployed_pre = exact_array("active_descriptors", np.int32, (12, 2))
        deployed_post = exact_array("deployed_descriptors_post", np.int32, (12, 2))
        candidate_pre = exact_array("candidate_descriptors", np.int32, (66, 2))
        candidate_post = exact_array("candidate_descriptors_post", np.int32, (66, 2))
        probe_weights_pre = exact_array(
            "interaction_relevance_probe_weights_pre",
            np.float32,
            (1, 12),
        )
        probe_weights_post = exact_array(
            "interaction_relevance_probe_weights_post",
            np.float32,
            (1, 12),
        )
        probe_biases_pre = exact_array(
            "interaction_relevance_probe_biases_pre",
            np.float32,
            (1,),
        )
        probe_biases_post = exact_array(
            "interaction_relevance_probe_biases_post",
            np.float32,
            (1,),
        )
        probe_scores = exact_array(
            "interaction_relevance_probe_scores",
            np.float32,
            (12,),
        )
        probe_errors = exact_array(
            "interaction_relevance_probe_errors",
            np.float32,
            (1, 12),
        )
        feature_moments_pre = exact_array(
            "interaction_feature_second_moments_pre",
            np.float32,
            (12,),
        )
        feature_moments_post = exact_array(
            "interaction_feature_second_moments_post",
            np.float32,
            (12,),
        )
        candidate_moments_pre = exact_array(
            "interaction_candidate_second_moments_pre",
            np.float32,
            (66,),
        )
        candidate_moments_post = exact_array(
            "interaction_candidate_second_moments_post",
            np.float32,
            (66,),
        )
        target_moments_pre = exact_array(
            "interaction_target_second_moments_pre",
            np.float32,
            (1,),
        )
        target_moments_post = exact_array(
            "interaction_target_second_moments_post",
            np.float32,
            (1,),
        )
        output_weights_pre = exact_array(
            "interaction_output_weights_pre",
            np.float32,
            (1, 12),
        )
        output_weights_post = exact_array(
            "interaction_output_weights_post",
            np.float32,
            (1, 12),
        )
        candidate_weights_pre = exact_array(
            "candidate_output_weights_pre",
            np.float32,
            (1, 66),
        )
        candidate_weights_post = exact_array(
            "candidate_output_weights_post",
            np.float32,
            (1, 66),
        )
        feature_utilities_pre = exact_array("active_utilities", np.float32, (12,))
        feature_utilities_post = exact_array(
            "shadow_utilities_post",
            np.float32,
            (12,),
        )
        candidate_utilities_pre = exact_array(
            "candidate_utilities",
            np.float32,
            (66,),
        )
        candidate_utilities_post = exact_array(
            "candidate_utilities_post",
            np.float32,
            (66,),
        )
        feature_idle_pre = exact_array("shadow_idle_steps_pre", np.int32, (12,))
        feature_idle_post = exact_array("shadow_idle_steps_post", np.int32, (12,))
        feature_ages_pre = exact_array(
            "interaction_feature_ages_pre",
            np.int32,
            (12,),
        )
        feature_ages_post = exact_array(
            "interaction_feature_ages_post",
            np.int32,
            (12,),
        )
        candidate_ages_pre = exact_array("candidate_ages_pre", np.int32, (66,))
        candidate_ages_post = exact_array("candidate_ages_post", np.int32, (66,))
        feature_raw = exact_array(
            "interaction_evidence_refreshed",
            np.bool_,
            (12,),
        )
        feature_confirmed = exact_array(
            "interaction_retention_evidence_refreshed",
            np.bool_,
            (12,),
        )
        feature_streak_pre = exact_array(
            "interaction_utility_evidence_streak_pre",
            np.int32,
            (12,),
        )
        feature_streak_post = exact_array(
            "interaction_utility_evidence_streak_post",
            np.int32,
            (12,),
        )
        feature_committed_pre = exact_array(
            "interaction_active_output_memory_committed_pre",
            np.bool_,
            (12,),
        )
        feature_committed_post = exact_array(
            "interaction_active_output_memory_committed_post",
            np.bool_,
            (12,),
        )
        consumer_mask_pre = exact_array(
            "consumer_active_mask_pre",
            np.bool_,
            (12,),
        )
        durable_read_mask = exact_array(
            "interaction_durable_read_mask",
            np.bool_,
            (12,),
        )
        candidate_signal = exact_array(
            "interaction_candidate_promotion_signal",
            np.float32,
            (66,),
        )
        candidate_raw = exact_array(
            "interaction_candidate_promotion_raw_evidence",
            np.bool_,
            (66,),
        )
        candidate_streak_pre = exact_array(
            "interaction_candidate_promotion_evidence_streak_pre",
            np.int32,
            (66,),
        )
        candidate_streak_updated = exact_array(
            "interaction_candidate_promotion_evidence_streak_updated",
            np.int32,
            (66,),
        )
        candidate_streak_post = exact_array(
            "interaction_candidate_promotion_evidence_streak_post",
            np.int32,
            (66,),
        )
        candidate_confirmed = exact_array(
            "interaction_candidate_promotion_confirmed",
            np.bool_,
            (66,),
        )
        reacquisition_pre = exact_array(
            "interaction_candidate_reacquisition_required_pre",
            np.bool_,
            (66,),
        )
        reacquisition_post = exact_array(
            "interaction_candidate_reacquisition_required_post",
            np.bool_,
            (66,),
        )
        reacquisition_confirmed = exact_array(
            "interaction_candidate_reacquisition_confirmed",
            np.bool_,
            (66,),
        )
        promoted = exact_array("interaction_promoted_candidate", np.int32, ())
        replaced = exact_array("interaction_replaced_slot", np.int32, ())
        retired_slot = exact_array("interaction_retired_slot", np.int32, ())
        retired_left = exact_array("interaction_retired_left", np.int32, ())
        retired_right = exact_array("interaction_retired_right", np.int32, ())
        reset_mask = exact_array(
            "interaction_matching_candidate_reset_mask",
            np.bool_,
            (66,),
        )
        reset_count = exact_array(
            "interaction_matching_candidate_reset_count",
            np.int32,
            (),
        )
        promoted_into_vacancy = exact_array(
            "interaction_promoted_into_vacancy",
            np.bool_,
            (),
        )
        live_feature_count = exact_array(
            "interaction_live_feature_count",
            np.int32,
            (),
        )
        vacancy_count = exact_array(
            "interaction_vacancy_count",
            np.int32,
            (),
        )
        descriptors_changed = exact_array("descriptors_changed", np.bool_, ())
    except (AttributeError, ValueError):
        return _invalid_v5_audit(steps, role)

    feature_violations = np.zeros((steps, 12), dtype=np.bool_)
    candidate_violations = np.zeros((steps, 66), dtype=np.bool_)
    step_violations = np.zeros((steps,), dtype=np.bool_)

    float_arrays = (
        phi,
        targets,
        probe_weights_pre,
        probe_weights_post,
        probe_biases_pre,
        probe_biases_post,
        probe_scores,
        probe_errors,
        feature_moments_pre,
        feature_moments_post,
        candidate_moments_pre,
        candidate_moments_post,
        target_moments_pre,
        target_moments_post,
        output_weights_pre,
        output_weights_post,
        candidate_weights_pre,
        candidate_weights_post,
        feature_utilities_pre,
        feature_utilities_post,
        candidate_utilities_pre,
        candidate_utilities_post,
        candidate_signal,
    )
    for values in float_arrays:
        noncanonical = ~np.isfinite(values) | (
            (values == np.float32(0.0)) & np.signbit(values)
        )
        step_violations |= np.any(noncanonical.reshape((steps, -1)), axis=1)
    if not np.all((targets == np.float32(-1.0)) | (targets == np.float32(1.0))):
        step_violations[:] = True

    feature_violations |= (
        (feature_idle_pre < 0)
        | (feature_idle_post < 0)
        | (feature_ages_pre < 0)
        | (feature_ages_post < 0)
        | (feature_streak_pre < 0)
        | (feature_streak_post < 0)
    )
    candidate_violations |= (
        (candidate_ages_pre < 0)
        | (candidate_ages_post < 0)
        | (candidate_streak_pre < 0)
        | (candidate_streak_updated < 0)
        | (candidate_streak_post < 0)
    )
    # The frozen runner initializes every learned head, statistic, utility,
    # age, and confirmation state canonically at positive zero.  Pinning the
    # initial state prevents a continuous but fabricated recurrence from being
    # accepted (including hostile int32-max age histories).
    feature_violations[0] |= (
        np.any(probe_weights_pre[0] != np.float32(0.0), axis=0)
        | (feature_moments_pre[0] != np.float32(0.0))
        | np.any(output_weights_pre[0] != np.float32(0.0), axis=0)
        | (feature_utilities_pre[0] != np.float32(0.0))
        | (feature_idle_pre[0] != 0)
        | (feature_ages_pre[0] != 0)
        | (feature_streak_pre[0] != 0)
        | feature_committed_pre[0]
    )
    candidate_violations[0] |= (
        np.any(candidate_weights_pre[0] != np.float32(0.0), axis=0)
        | (candidate_moments_pre[0] != np.float32(0.0))
        | (candidate_utilities_pre[0] != np.float32(0.0))
        | (candidate_ages_pre[0] != 0)
        | (candidate_streak_pre[0] != 0)
        | reacquisition_pre[0]
    )
    step_violations[0] |= bool(
        np.any(probe_biases_pre[0] != np.float32(0.0))
        or np.any(target_moments_pre[0] != np.float32(0.0))
    )

    continuity_pairs = (
        (shadow_pre, shadow_post),
        (deployed_pre, deployed_post),
        (candidate_pre, candidate_post),
        (probe_weights_pre, probe_weights_post),
        (probe_biases_pre, probe_biases_post),
        (feature_moments_pre, feature_moments_post),
        (candidate_moments_pre, candidate_moments_post),
        (target_moments_pre, target_moments_post),
        (output_weights_pre, output_weights_post),
        (candidate_weights_pre, candidate_weights_post),
        (feature_utilities_pre, feature_utilities_post),
        (candidate_utilities_pre, candidate_utilities_post),
        (feature_idle_pre, feature_idle_post),
        (feature_ages_pre, feature_ages_post),
        (candidate_ages_pre, candidate_ages_post),
        (feature_streak_pre, feature_streak_post),
        (feature_committed_pre, feature_committed_post),
        (candidate_streak_pre, candidate_streak_post),
        (reacquisition_pre, reacquisition_post),
    )
    for pre, post in continuity_pairs:
        if steps > 1 and not np.array_equal(pre[1:], post[:-1]):
            step_violations[1:] = True

    live_feature_pre = _descriptor_live_mask(shadow_pre)
    live_candidate = _descriptor_live_mask(candidate_pre)
    candidate_archive_valid = _candidate_archive_contract_valid(
        candidate_pre,
        candidate_post,
    )
    if not candidate_archive_valid:
        candidate_violations[:] = True
    if not np.array_equal(shadow_pre, deployed_pre) or not np.array_equal(
        shadow_post,
        deployed_post,
    ):
        feature_violations[:] = True

    promoted_in_domain = (promoted == -1) | ((promoted >= 0) & (promoted < 66))
    replaced_in_domain = (replaced == -1) | ((replaced >= 0) & (replaced < 12))
    retired_in_domain = (retired_slot == -1) | (
        (retired_slot >= 0) & (retired_slot < 12)
    )
    retired_descriptor_is_sentinel = (retired_left == -1) & (retired_right == -1)
    retired_descriptor_is_pair = (
        (retired_left >= 0)
        & (retired_left < retired_right)
        & (retired_right < 12)
    )
    event_domain_valid = (
        promoted_in_domain
        & replaced_in_domain
        & retired_in_domain
        & ((retired_slot == -1) == retired_descriptor_is_sentinel)
        & (retired_descriptor_is_sentinel | retired_descriptor_is_pair)
        & ((promoted == -1) == (replaced == -1))
        & ~((promoted >= 0) & (retired_slot >= 0))
        & ~((promoted == -1) & promoted_into_vacancy)
        & (reset_count >= 0)
        & (reset_count <= 66)
    )
    step_violations |= ~event_domain_valid
    expected_descriptor_change = np.any(shadow_pre != shadow_post, axis=(1, 2))
    expected_live_count = np.sum(
        _descriptor_live_mask(shadow_post),
        axis=1,
        dtype=np.int32,
    )
    step_violations |= descriptors_changed != expected_descriptor_change
    step_violations |= live_feature_count != expected_live_count
    step_violations |= vacancy_count != (12 - expected_live_count)

    feature_values = _pair_values_host(shadow_pre, phi)
    candidate_values = _pair_values_host(candidate_pre, phi)
    moment_decay = 0.99
    feature_moment_paths = _float32_ema_paths(
        feature_moments_pre,
        feature_values,
        moment_decay,
    )
    candidate_moment_paths = _float32_ema_paths(
        candidate_moments_pre,
        candidate_values,
        moment_decay,
    )
    target_moment_paths = _float32_ema_paths(
        target_moments_pre,
        targets,
        moment_decay,
    )

    expected_reset_mask = np.zeros_like(reset_mask)
    expected_feature_moment_paths = [values.copy() for values in feature_moment_paths]
    expected_candidate_moment_paths = [values.copy() for values in candidate_moment_paths]
    expected_target_moment_paths = [values.copy() for values in target_moment_paths]
    for step in range(steps):
        retirement = int(retired_slot[step])
        promotion = int(promoted[step])
        destination = int(replaced[step])
        if retirement >= 0:
            if not 0 <= retirement < 12 or promotion != -1 or destination != -1:
                step_violations[step] = True
                continue
            retired_pair = np.asarray(
                (retired_left[step], retired_right[step]),
                dtype=np.int32,
            )
            exact_retirement = bool(
                np.array_equal(shadow_pre[step, retirement], retired_pair)
                and np.array_equal(
                    shadow_post[step, retirement],
                    np.asarray((-1, -1), dtype=np.int32),
                )
            )
            unchanged = np.ones((12,), dtype=np.bool_)
            unchanged[retirement] = False
            exact_retirement &= bool(
                np.array_equal(
                    shadow_pre[step, unchanged],
                    shadow_post[step, unchanged],
                )
            )
            if not exact_retirement:
                step_violations[step] = True
            matching = np.all(candidate_pre[step] == retired_pair, axis=1)
            expected_reset_mask[step] = matching
            for path in expected_feature_moment_paths:
                path[step, retirement] = np.float32(0.0)
            for path in expected_candidate_moment_paths:
                path[step, matching] = np.float32(0.0)
        elif promotion >= 0:
            if not (0 <= promotion < 66 and 0 <= destination < 12):
                step_violations[step] = True
                continue
            expected_post = shadow_pre[step].copy()
            expected_post[destination] = candidate_pre[step, promotion]
            if not np.array_equal(expected_post, shadow_post[step]):
                step_violations[step] = True
            expected_vacancy = bool(np.all(shadow_pre[step, destination] == -1))
            if bool(promoted_into_vacancy[step]) != expected_vacancy:
                step_violations[step] = True
            for feature_path, candidate_path in zip(
                expected_feature_moment_paths,
                candidate_moment_paths,
                strict=True,
            ):
                feature_path[step, destination] = candidate_path[step, promotion]
        else:
            if destination != -1 or not np.array_equal(
                shadow_pre[step],
                shadow_post[step],
            ):
                step_violations[step] = True
    if not np.array_equal(reset_mask, expected_reset_mask) or not np.array_equal(
        reset_count,
        np.sum(expected_reset_mask, axis=1, dtype=np.int32),
    ):
        candidate_violations |= reset_mask != expected_reset_mask
        step_violations |= reset_count != np.sum(
            expected_reset_mask,
            axis=1,
            dtype=np.int32,
        )

    feature_violations |= ~_float32_matches(
        feature_moments_post,
        *expected_feature_moment_paths,
    )
    candidate_violations |= ~_float32_matches(
        candidate_moments_post,
        *expected_candidate_moment_paths,
    )
    if not np.all(
        _float32_matches(target_moments_post, *expected_target_moment_paths)
    ):
        step_violations[:] = True

    reference_paths = tuple(
        _probe_reference_path(
            phi=phi,
            targets=targets,
            feature_values=feature_values,
            candidate_values=candidate_values,
            probe_weights=probe_weights_pre,
            probe_biases=probe_biases_pre,
            candidate_weights=candidate_weights_pre,
            feature_moments=moment_path,
            candidate_moments=candidate_path,
            target_moments=target_path,
            step_size=result.condition.config.interaction_step_size,
            fused=fused,
        )
        for moment_path, candidate_path, target_path, fused in (
            (
                feature_moment_paths[0],
                candidate_moment_paths[0],
                target_moment_paths[0],
                False,
            ),
            (
                feature_moment_paths[1],
                candidate_moment_paths[1],
                target_moment_paths[1],
                True,
            ),
        )
    )
    expected_baselines = tuple(path[0] for path in reference_paths)
    expected_probe_scores = tuple(path[1] for path in reference_paths)
    expected_candidate_signals = tuple(path[2] for path in reference_paths)
    expected_probe_post = [path[3].copy() for path in reference_paths]
    expected_bias_post = tuple(path[4] for path in reference_paths)
    expected_candidate_post = [path[5].copy() for path in reference_paths]
    for step in range(steps):
        retirement = int(retired_slot[step])
        promotion = int(promoted[step])
        destination = int(replaced[step])
        if retirement >= 0:
            matching = expected_reset_mask[step]
            for probe_path, candidate_path in zip(
                expected_probe_post,
                expected_candidate_post,
                strict=True,
            ):
                probe_path[step, :, retirement] = np.float32(0.0)
                candidate_path[step, :, matching] = np.float32(0.0)
        elif promotion >= 0 and destination >= 0:
            for probe_path, candidate_path in zip(
                expected_probe_post,
                expected_candidate_post,
                strict=True,
            ):
                probe_path[step, :, destination] = candidate_path[
                    step,
                    :,
                    promotion,
                ]
                candidate_path[step, :, promotion] = np.float32(0.0)

    baseline_matrix = tuple(
        np.broadcast_to(values[:, :, None], (steps, 1, 12))
        for values in expected_baselines
    )
    feature_violations |= ~np.all(
        np.stack(
            (
                _float32_matches(probe_scores, *expected_probe_scores),
                np.all(
                    _float32_matches(probe_errors, *baseline_matrix),
                    axis=1,
                ),
                np.all(
                    _float32_matches(probe_weights_post, *expected_probe_post),
                    axis=1,
                ),
            ),
            axis=0,
        ),
        axis=0,
    )
    candidate_violations |= ~_float32_matches(
        candidate_signal,
        *expected_candidate_signals,
    )
    candidate_violations |= ~np.all(
        _float32_matches(candidate_weights_post, *expected_candidate_post),
        axis=1,
    )
    step_violations |= ~np.all(
        _float32_matches(probe_biases_post, *expected_bias_post),
        axis=1,
    )

    expected_feature_raw = live_feature_pre & (
        probe_scores
        >= np.float32(result.condition.config.active_utility_evidence_threshold)
    )
    feature_violations |= feature_raw != expected_feature_raw
    expected_candidate_raw = (
        live_candidate
        & np.isfinite(candidate_signal)
        & (candidate_signal > np.float32(0.0))
        & (
            candidate_signal
            >= np.float32(result.condition.config.active_utility_evidence_threshold)
        )
    )
    candidate_violations |= candidate_raw != expected_candidate_raw

    expected_candidate_streak_pre = np.zeros_like(candidate_streak_pre)
    expected_candidate_streak_updated = np.zeros_like(candidate_streak_updated)
    expected_candidate_streak_post = np.zeros_like(candidate_streak_post)
    expected_reacquisition_pre = np.zeros_like(reacquisition_pre)
    expected_reacquisition_post = np.zeros_like(reacquisition_post)
    expected_candidate_confirmed = np.zeros_like(candidate_confirmed)
    expected_reacquisition_confirmed = np.zeros_like(reacquisition_confirmed)
    current_streak = np.zeros((66,), dtype=np.int32)
    current_required = np.zeros((66,), dtype=np.bool_)
    generic_steps = result.condition.config.candidate_promotion_confirmation_steps
    reacquisition_steps = (
        result.condition.config.candidate_reacquisition_confirmation_steps
    )
    for step in range(steps):
        expected_candidate_streak_pre[step] = current_streak
        current_required = (
            current_required
            & live_candidate[step]
            & (reacquisition_steps > 1)
        )
        expected_reacquisition_pre[step] = current_required
        required_steps = np.where(
            current_required,
            max(generic_steps, reacquisition_steps),
            generic_steps,
        ).astype(np.int32)
        gate_active = required_steps > 1
        incremented = np.minimum(
            np.maximum(current_streak, 0),
            required_steps - 1,
        ) + 1
        updated = np.where(
            expected_candidate_raw[step] & gate_active,
            incremented,
            0,
        ).astype(np.int32)
        confirmed = live_candidate[step] & (
            ~gate_active | (updated >= required_steps)
        )
        reacquired = (
            live_candidate[step]
            & current_required
            & (updated >= required_steps)
        )
        expected_candidate_streak_updated[step] = updated
        expected_candidate_confirmed[step] = confirmed
        expected_reacquisition_confirmed[step] = reacquired
        next_streak = updated.copy()
        next_required = current_required.copy()
        retirement = int(retired_slot[step])
        promotion = int(promoted[step])
        if retirement >= 0:
            matching = expected_reset_mask[step]
            next_streak[matching] = 0
            next_required[matching] = reacquisition_steps > 1
        if promotion >= 0:
            if not confirmed[promotion]:
                candidate_violations[step, promotion] = True
            if current_required[promotion] and not reacquired[promotion]:
                candidate_violations[step, promotion] = True
            next_streak[promotion] = 0
            next_required[promotion] = False
        expected_candidate_streak_post[step] = next_streak
        expected_reacquisition_post[step] = next_required
        current_streak = next_streak
        current_required = next_required

    candidate_violations |= (
        candidate_streak_pre != expected_candidate_streak_pre
    )
    candidate_violations |= (
        candidate_streak_updated != expected_candidate_streak_updated
    )
    candidate_violations |= (
        candidate_streak_post != expected_candidate_streak_post
    )
    candidate_violations |= reacquisition_pre != expected_reacquisition_pre
    candidate_violations |= reacquisition_post != expected_reacquisition_post
    candidate_violations |= candidate_confirmed != expected_candidate_confirmed
    candidate_violations |= (
        reacquisition_confirmed != expected_reacquisition_confirmed
    )

    candidate_utility_paths = list(
        _float32_linear_ema_paths(
            candidate_utilities_pre,
            candidate_signal,
            result.condition.config.interaction_utility_decay,
        )
    )
    retention_decay = result.condition.config.candidate_utility_retention_decay
    retained_candidate_paths = (
        np.asarray(
            np.float32(retention_decay) * candidate_utilities_pre,
            dtype=np.float32,
        ),
        np.asarray(
            float(np.float32(retention_decay))
            * candidate_utilities_pre.astype(np.float64),
            dtype=np.float32,
        ),
    )
    candidate_utility_paths = [
        np.maximum(ema, retained).astype(np.float32)
        for ema, retained in zip(
            candidate_utility_paths,
            retained_candidate_paths,
            strict=True,
        )
    ]
    int32_max = np.iinfo(np.int32).max
    expected_candidate_ages = np.asarray(
        np.minimum(np.maximum(candidate_ages_pre, 0), int32_max - 1) + 1,
        dtype=np.int32,
    )
    for step in range(steps):
        if int(retired_slot[step]) >= 0:
            matching = expected_reset_mask[step]
            expected_candidate_ages[step, matching] = 0
            for path in candidate_utility_paths:
                path[step, matching] = np.float32(0.0)
        promotion = int(promoted[step])
        if promotion >= 0:
            expected_candidate_ages[step, promotion] = 0
            for path in candidate_utility_paths:
                path[step, promotion] = np.float32(0.0)
    candidate_violations |= candidate_ages_post != expected_candidate_ages
    candidate_violations |= ~_float32_matches(
        candidate_utilities_post,
        *candidate_utility_paths,
    )

    grace_steps = result.condition.config.active_utility_retention_grace_steps
    if grace_steps is None:
        return _invalid_v5_audit(steps, role)
    expected_idle = np.where(
        live_feature_pre,
        np.where(
            feature_confirmed,
            0,
            np.minimum(feature_idle_pre, int32_max - 1) + 1,
        ),
        0,
    ).astype(np.int32)
    feature_utility_ema_paths = list(
        _float32_linear_ema_paths(
            feature_utilities_pre,
            probe_scores,
            result.condition.config.interaction_utility_decay,
        )
    )
    feature_retention = result.condition.config.active_utility_retention_decay
    if feature_retention is None:
        feature_utility_paths = feature_utility_ema_paths
    else:
        retained_paths = (
            np.asarray(
                np.float32(feature_retention) * feature_utilities_pre,
                dtype=np.float32,
            ),
            np.asarray(
                float(np.float32(feature_retention))
                * feature_utilities_pre.astype(np.float64),
                dtype=np.float32,
            ),
        )
        protected = live_feature_pre & (
            expected_idle <= grace_steps
        )
        feature_utility_paths = [
            np.where(protected, np.maximum(ema, retained), ema).astype(np.float32)
            for ema, retained in zip(
                feature_utility_ema_paths,
                retained_paths,
                strict=True,
            )
        ]
    feature_age_expected = np.where(
        live_feature_pre,
        np.minimum(np.maximum(feature_ages_pre, 0), int32_max - 1) + 1,
        0,
    ).astype(np.int32)
    for step in range(steps):
        retirement = int(retired_slot[step])
        promotion = int(promoted[step])
        destination = int(replaced[step])
        if retirement >= 0:
            expected_idle[step, retirement] = 0
            feature_age_expected[step, retirement] = 0
            for path in feature_utility_paths:
                path[step, retirement] = np.float32(0.0)
        elif promotion >= 0 and destination >= 0:
            expected_idle[step, destination] = 0
            feature_age_expected[step, destination] = 0
            # Promotion copies the candidate utility before its archive reset.
            promoted_ema_paths = _float32_linear_ema_paths(
                candidate_utilities_pre[step, promotion],
                candidate_signal[step, promotion],
                result.condition.config.interaction_utility_decay,
            )
            promoted_retained_paths = (
                np.asarray(
                    np.float32(retention_decay)
                    * np.float32(candidate_utilities_pre[step, promotion]),
                    dtype=np.float32,
                ),
                np.asarray(
                    float(np.float32(retention_decay))
                    * float(candidate_utilities_pre[step, promotion]),
                    dtype=np.float32,
                ),
            )
            for index, feature_path in enumerate(feature_utility_paths):
                feature_path[step, destination] = np.maximum(
                    promoted_ema_paths[index],
                    promoted_retained_paths[index],
                ).astype(np.float32)
    feature_violations |= feature_idle_post != expected_idle
    feature_violations |= feature_ages_post != feature_age_expected
    feature_violations |= ~_float32_matches(
        feature_utilities_post,
        *feature_utility_paths,
    )

    feature_memory_audit = _feature_memory_contract_audit(
        deployed_pre,
        deployed_post,
        shadow_pre,
        shadow_post,
        feature_raw,
        feature_confirmed,
        output_weights_pre,
        output_weights_post,
        feature_streak_pre,
        feature_streak_post,
        feature_committed_pre,
        feature_committed_post,
        result.condition.config,
    )
    feature_violations |= feature_memory_audit.violation_bits
    expected_read = live_feature_pre & feature_committed_pre & consumer_mask_pre
    feature_violations |= durable_read_mask != expected_read

    strict_output_expected = np.where(
        feature_committed_pre[:, None, :],
        output_weights_pre,
        np.where(
            feature_confirmed[:, None, :],
            probe_weights_pre,
            np.float32(0.0),
        ),
    ).astype(np.float32)
    for step in range(steps):
        retirement = int(retired_slot[step])
        promotion = int(promoted[step])
        destination = int(replaced[step])
        if retirement >= 0:
            strict_output_expected[step, :, retirement] = np.float32(0.0)
        elif promotion >= 0 and destination >= 0:
            strict_output_expected[step, :, destination] = np.float32(0.0)
    feature_violations |= ~np.all(
        _float32_matches(output_weights_post, strict_output_expected),
        axis=1,
    )

    try:
        historical = summarize_critical_lifecycle_v2(result)
        historical_contracts = bool(
            historical.representation_link_contract_valid
            and historical.consumer_gate_contract_valid
            and historical.feature_memory_contract_valid
            and historical.candidate_archive_contract_valid
        )
    except (IndexError, ValueError):
        historical_contracts = False
    lifecycle_valid = bool(
        historical_contracts
        and not np.any(step_violations)
        and not np.any(reset_mask != expected_reset_mask)
        and result.summary.descriptor_transaction_count
        == int(np.count_nonzero(descriptors_changed))
    )
    probe_valid = bool(
        not np.any(feature_violations)
        and not np.any(candidate_violations & ~(
            (candidate_streak_pre != expected_candidate_streak_pre)
            | (candidate_streak_updated != expected_candidate_streak_updated)
            | (candidate_streak_post != expected_candidate_streak_post)
            | (reacquisition_pre != expected_reacquisition_pre)
            | (reacquisition_post != expected_reacquisition_post)
            | (candidate_confirmed != expected_candidate_confirmed)
            | (reacquisition_confirmed != expected_reacquisition_confirmed)
        ))
    )
    candidate_contract_valid = bool(not np.any(candidate_violations))
    durable_valid = bool(
        feature_memory_audit.valid
        and not np.any(durable_read_mask != expected_read)
        and np.all(_float32_matches(output_weights_post, strict_output_expected))
    )
    resource_valid = _v5_resource_contract_valid(result)
    all_valid = bool(
        probe_valid
        and candidate_contract_valid
        and durable_valid
        and lifecycle_valid
        and resource_valid
    )
    return HiddenPartnerLifecycleV5Audit(
        schema_version=HIDDEN_PARTNER_LIFECYCLE_V5_SCHEMA,
        development_only=True,
        scientific_promotion_allowed=False,
        config_role=role,
        probe_contract_valid=probe_valid,
        candidate_reacquisition_contract_valid=candidate_contract_valid,
        durable_memory_contract_valid=durable_valid,
        lifecycle_contract_valid=lifecycle_valid,
        resource_contract_valid=resource_valid,
        all_contracts_valid=all_valid,
        feature_violation_bits=feature_violations,
        candidate_violation_bits=candidate_violations,
        step_violation_bits=step_violations,
    )


def _invalid_feature_memory_audit(steps: int) -> _FeatureMemoryContractAudit:
    safe_steps = max(steps, 0)
    return _FeatureMemoryContractAudit(
        valid=False,
        identity_routed_head_changed=np.zeros(
            (safe_steps, 12),
            dtype=np.bool_,
        ),
        violation_bits=np.ones(
            (safe_steps, 12),
            dtype=np.bool_,
        ),
    )


def _duplicate_descriptor_mask(
    descriptors: np.ndarray,
    live: np.ndarray,
) -> np.ndarray:
    codes = descriptors[..., 0] * 12 + descriptors[..., 1]
    matches = codes[:, :, None] == codes[:, None, :]
    duplicate_counts = np.sum(matches & live[:, None, :], axis=2)
    return cast(np.ndarray, live & (duplicate_counts != 1))


def _float32_bits(values: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(values, dtype=np.float32)
    return cast(np.ndarray, contiguous.view(np.uint32))


def _feature_memory_contract_audit(
    deployed_pre: np.ndarray,
    deployed_post: np.ndarray,
    shadow_pre: np.ndarray,
    shadow_post: np.ndarray,
    raw_evidence: np.ndarray,
    confirmed_evidence: np.ndarray,
    output_weights_pre: np.ndarray,
    output_weights_post: np.ndarray,
    streak_pre: np.ndarray,
    streak_post: np.ndarray,
    committed_pre: np.ndarray,
    committed_post: np.ndarray,
    config: IntegratedHiddenPartnerConfig,
) -> _FeatureMemoryContractAudit:
    """Reconstruct confirmed feature memory without trusting its state trace.

    Evidence and confirmation are indexed by the pre-curation shadow identity.
    Streak and commitment state then follow a surviving descriptor into its
    post-curation slot.  A new, replaced, promoted, or retired identity has no
    source and therefore starts with zero streak and an uncommitted head.
    """
    steps = raw_evidence.shape[0] if raw_evidence.ndim >= 1 else 0
    descriptor_shape = (steps, 12, 2)
    state_shape = (steps, 12)
    if any(
        values.shape != descriptor_shape
        for values in (
            deployed_pre,
            deployed_post,
            shadow_pre,
            shadow_post,
        )
    ):
        return _invalid_feature_memory_audit(steps)
    if any(
        values.shape != state_shape
        for values in (
            raw_evidence,
            confirmed_evidence,
            streak_pre,
            streak_post,
            committed_pre,
            committed_post,
        )
    ):
        return _invalid_feature_memory_audit(steps)
    if (
        output_weights_pre.ndim != 3
        or output_weights_pre.shape[0] != steps
        or output_weights_pre.shape[2] != 12
        or output_weights_pre.shape != output_weights_post.shape
        or output_weights_pre.shape[1] <= 0
    ):
        return _invalid_feature_memory_audit(steps)

    live_deployed_pre = _descriptor_live_mask(deployed_pre)
    live_deployed_post = _descriptor_live_mask(deployed_post)
    live_shadow_pre = _descriptor_live_mask(shadow_pre)
    live_shadow_post = _descriptor_live_mask(shadow_post)
    descriptor_valid = (
        (live_deployed_pre | np.all(deployed_pre == -1, axis=2))
        & (live_deployed_post | np.all(deployed_post == -1, axis=2))
        & (live_shadow_pre | np.all(shadow_pre == -1, axis=2))
        & (live_shadow_post | np.all(shadow_post == -1, axis=2))
    )
    descriptor_valid &= ~_duplicate_descriptor_mask(
        deployed_pre,
        live_deployed_pre,
    )
    descriptor_valid &= ~_duplicate_descriptor_mask(
        deployed_post,
        live_deployed_post,
    )
    descriptor_valid &= ~_duplicate_descriptor_mask(
        shadow_pre,
        live_shadow_pre,
    )
    descriptor_valid &= ~_duplicate_descriptor_mask(
        shadow_post,
        live_shadow_post,
    )

    violation_bits = ~descriptor_valid
    violation_bits |= np.any(deployed_pre != shadow_pre, axis=2)
    violation_bits |= np.any(deployed_post != shadow_post, axis=2)
    if steps > 1:
        violation_bits[1:] |= np.any(
            deployed_pre[1:] != deployed_post[:-1],
            axis=2,
        )
        violation_bits[1:] |= np.any(
            shadow_pre[1:] != shadow_post[:-1],
            axis=2,
        )

    pre_codes = shadow_pre[..., 0] * 12 + shadow_pre[..., 1]
    post_codes = shadow_post[..., 0] * 12 + shadow_post[..., 1]
    identity_matches = (
        (post_codes[:, :, None] == pre_codes[:, None, :])
        & live_shadow_post[:, :, None]
        & live_shadow_pre[:, None, :]
    )
    source_for_post = np.argmax(identity_matches, axis=2)
    post_has_source = np.any(identity_matches, axis=2)
    destination_for_pre = np.argmax(identity_matches, axis=1)
    pre_has_destination = np.any(identity_matches, axis=1)

    expected_streak_pre = np.zeros(state_shape, dtype=np.int32)
    expected_streak_post = np.zeros(state_shape, dtype=np.int32)
    expected_confirmed = np.zeros(state_shape, dtype=np.bool_)
    expected_committed_pre = np.zeros(state_shape, dtype=np.bool_)
    expected_committed_post = np.zeros(state_shape, dtype=np.bool_)
    current_streak = np.zeros((12,), dtype=np.int32)
    current_committed = np.zeros((12,), dtype=np.bool_)
    int32_max = np.iinfo(np.int32).max
    for step in range(steps):
        expected_streak_pre[step] = current_streak
        expected_committed_pre[step] = current_committed
        if config.evidence_gated_feature_memory:
            incremented = np.minimum(
                np.maximum(current_streak, 0),
                int32_max - 1,
            ) + 1
            updated_streak = np.where(
                live_shadow_pre[step] & raw_evidence[step],
                incremented,
                0,
            ).astype(np.int32)
            confirmed = (
                live_shadow_pre[step]
                & raw_evidence[step]
                & (
                    updated_streak
                    >= config.feature_evidence_confirmation_steps
                )
            )
            updated_committed = live_shadow_pre[step] & (
                current_committed | confirmed
            )
        else:
            updated_streak = np.zeros((12,), dtype=np.int32)
            confirmed = live_shadow_pre[step] & raw_evidence[step]
            updated_committed = np.zeros((12,), dtype=np.bool_)
        expected_confirmed[step] = confirmed

        source = source_for_post[step]
        next_streak = np.where(
            live_shadow_post[step] & post_has_source[step],
            updated_streak[source],
            0,
        ).astype(np.int32)
        next_committed = (
            live_shadow_post[step]
            & post_has_source[step]
            & updated_committed[source]
        )
        expected_streak_post[step] = next_streak
        expected_committed_post[step] = next_committed
        current_streak = next_streak
        current_committed = next_committed

    violation_bits |= raw_evidence & ~live_shadow_pre
    violation_bits |= confirmed_evidence != expected_confirmed
    violation_bits |= streak_pre != expected_streak_pre
    violation_bits |= streak_post != expected_streak_post
    violation_bits |= committed_pre != expected_committed_pre
    violation_bits |= committed_post != expected_committed_post

    pre_weight_bits = _float32_bits(output_weights_pre)
    post_weight_bits = _float32_bits(output_weights_post)
    routed_post_bits = np.take_along_axis(
        post_weight_bits,
        destination_for_pre[:, None, :],
        axis=2,
    )
    identity_routed_head_changed = (
        live_shadow_pre
        & pre_has_destination
        & np.any(pre_weight_bits != routed_post_bits, axis=1)
    )
    if steps > 1:
        violation_bits[1:] |= np.any(
            pre_weight_bits[1:] != post_weight_bits[:-1],
            axis=1,
        )
    nonfinite_weights = (
        np.any(~np.isfinite(output_weights_pre), axis=1)
        | np.any(~np.isfinite(output_weights_post), axis=1)
    )
    violation_bits |= nonfinite_weights
    if config.evidence_gated_feature_memory:
        violation_bits |= (
            expected_committed_pre
            & ~expected_confirmed
            & identity_routed_head_changed
        )

    return _FeatureMemoryContractAudit(
        valid=bool(not np.any(violation_bits)),
        identity_routed_head_changed=identity_routed_head_changed,
        violation_bits=violation_bits,
    )


def _canonical_lifecycle_rle(
    deployed_slots: np.ndarray,
    shadow_slots: np.ndarray,
    candidate_slots: np.ndarray,
) -> tuple[CriticalPairLifecycleInterval, ...]:
    if not (
        deployed_slots.shape == shadow_slots.shape == candidate_slots.shape
        and deployed_slots.ndim == 1
        and deployed_slots.size > 0
    ):
        raise ValueError("lifecycle slot arrays must be equal non-empty vectors")
    rows = np.stack((deployed_slots, shadow_slots, candidate_slots), axis=1)
    starts = np.concatenate(
        (
            np.asarray([0], dtype=np.int64),
            np.flatnonzero(np.any(rows[1:] != rows[:-1], axis=1)) + 1,
        )
    )
    ends = np.concatenate((starts[1:], np.asarray([rows.shape[0]], dtype=np.int64)))
    return tuple(
        CriticalPairLifecycleInterval(
            start=int(start),
            end_exclusive=int(end),
            deployed_slot=int(rows[start, 0]),
            shadow_slot=int(rows[start, 1]),
            candidate_slot=int(rows[start, 2]),
        )
        for start, end in zip(starts, ends, strict=True)
    )


def _packed_bool_payload(values: np.ndarray) -> dict[str, object]:
    array = np.asarray(values, dtype=np.bool_)
    packed = np.packbits(array.reshape(-1), bitorder="little")
    return {
        "shape": list(array.shape),
        "bitorder": "little",
        "data_base64": base64.b64encode(packed.tobytes()).decode("ascii"),
    }


def _float32_state_sequence_payload(
    values_pre: np.ndarray,
    values_post: np.ndarray,
) -> dict[str, object]:
    """Encode an exact continuous T+1 float32 state sequence canonically."""
    pre = np.ascontiguousarray(values_pre, dtype="<f4")
    post = np.ascontiguousarray(values_post, dtype="<f4")
    if pre.shape != post.shape or pre.ndim < 1 or pre.shape[0] < 1:
        raise ValueError("float32 pre/post traces must have equal non-empty shapes")
    pre_bits = pre.view("<u4")
    post_bits = post.view("<u4")
    if not np.array_equal(pre_bits[1:], post_bits[:-1]):
        raise ValueError("float32 pre/post traces are not bitwise continuous")
    states = np.concatenate((pre[:1], post), axis=0)
    state_bits = np.ascontiguousarray(states, dtype="<f4").view("<u4")
    deltas = np.empty_like(state_bits)
    deltas[0] = state_bits[0]
    deltas[1:] = state_bits[1:] ^ state_bits[:-1]
    compressed = zlib.compress(deltas.tobytes(order="C"), level=9)
    return {
        "shape": list(states.shape),
        "dtype": "float32",
        "byteorder": "little",
        "delta": "uint32-xor",
        "codec": "zlib",
        "data_base64": base64.b64encode(compressed).decode("ascii"),
    }


def _numeric_array_payload(values: np.ndarray, *, dtype: str) -> dict[str, object]:
    """Encode one exact little-endian primitive array without JSON rounding."""
    array = np.ascontiguousarray(values, dtype=dtype)
    compressed = zlib.compress(array.tobytes(order="C"), level=9)
    return {
        "shape": list(array.shape),
        "dtype": "float32" if dtype == "<f4" else "int32",
        "byteorder": "little",
        "codec": "zlib",
        "data_base64": base64.b64encode(compressed).decode("ascii"),
    }


def _int32_state_sequence_payload(
    values_pre: np.ndarray,
    values_post: np.ndarray,
) -> dict[str, object]:
    """Encode an exact continuous T+1 int32 sequence using XOR deltas."""
    pre = np.ascontiguousarray(values_pre, dtype="<i4")
    post = np.ascontiguousarray(values_post, dtype="<i4")
    if pre.shape != post.shape or pre.ndim < 1 or pre.shape[0] < 1:
        raise ValueError("int32 pre/post traces must have equal non-empty shapes")
    if not np.array_equal(pre[1:], post[:-1]):
        raise ValueError("int32 pre/post traces are not exactly continuous")
    states = np.concatenate((pre[:1], post), axis=0)
    bits = np.ascontiguousarray(states, dtype="<i4").view("<u4")
    deltas = np.empty_like(bits)
    deltas[0] = bits[0]
    deltas[1:] = bits[1:] ^ bits[:-1]
    compressed = zlib.compress(deltas.tobytes(order="C"), level=9)
    return {
        "shape": list(states.shape),
        "dtype": "int32",
        "byteorder": "little",
        "delta": "uint32-xor",
        "codec": "zlib",
        "data_base64": base64.b64encode(compressed).decode("ascii"),
    }


def _canonical_candidate_descriptors() -> np.ndarray:
    return np.asarray(
        tuple(
            (left, right)
            for left in range(12)
            for right in range(left + 1, 12)
        ),
        dtype=np.int32,
    )


def _candidate_archive_contract_valid(
    candidate_pre: np.ndarray,
    candidate_post: np.ndarray,
) -> bool:
    steps = candidate_pre.shape[0] if candidate_pre.ndim >= 1 else 0
    if (
        candidate_pre.shape != (steps, 66, 2)
        or candidate_post.shape != candidate_pre.shape
        or steps < 1
    ):
        return False
    if not np.array_equal(candidate_pre[1:], candidate_post[:-1]):
        return False
    states = np.concatenate((candidate_pre[:1], candidate_post), axis=0)
    canonical = _canonical_candidate_descriptors()
    return bool(np.all(states == canonical[None, :, :]))


def _canonical_candidate_bank_state_rle(
    candidate_states: np.ndarray,
) -> list[dict[str, object]]:
    states = np.asarray(candidate_states, dtype=np.int32)
    if (
        states.ndim != 3
        or states.shape[1:] != (66, 2)
        or states.shape[0] < 2
    ):
        raise ValueError("candidate-bank state trace must have shape (T+1, 66, 2)")
    starts = np.concatenate(
        (
            np.asarray([0], dtype=np.int64),
            np.flatnonzero(np.any(states[1:] != states[:-1], axis=(1, 2))) + 1,
        )
    )
    ends = np.concatenate(
        (starts[1:], np.asarray([states.shape[0]], dtype=np.int64))
    )
    return [
        {
            "start": int(start),
            "end_exclusive": int(end),
            "candidate_descriptors": states[start].tolist(),
        }
        for start, end in zip(starts, ends, strict=True)
    ]


def _canonical_bank_state_rle(
    deployed_states: np.ndarray,
    shadow_states: np.ndarray,
) -> list[dict[str, object]]:
    if (
        deployed_states.shape != shadow_states.shape
        or deployed_states.ndim != 3
        or deployed_states.shape[1:] != (12, 2)
        or deployed_states.shape[0] < 2
    ):
        raise ValueError("full-bank state traces must have shape (T+1, 12, 2)")
    flattened = np.concatenate(
        (
            deployed_states.reshape((deployed_states.shape[0], -1)),
            shadow_states.reshape((shadow_states.shape[0], -1)),
        ),
        axis=1,
    )
    starts = np.concatenate(
        (
            np.asarray([0], dtype=np.int64),
            np.flatnonzero(np.any(flattened[1:] != flattened[:-1], axis=1)) + 1,
        )
    )
    ends = np.concatenate(
        (
            starts[1:],
            np.asarray([deployed_states.shape[0]], dtype=np.int64),
        )
    )
    return [
        {
            "start": int(start),
            "end_exclusive": int(end),
            "deployed_descriptors": deployed_states[start].tolist(),
            "shadow_descriptors": shadow_states[start].tolist(),
        }
        for start, end in zip(starts, ends, strict=True)
    ]


def _critical_window_primitives(
    *,
    behavior_logits: np.ndarray,
    behavior_weights: np.ndarray,
    representations: np.ndarray,
    descriptors: np.ndarray,
    intended_actions: np.ndarray,
    pair: tuple[int, int],
    entry_step: int,
    window_start: int,
    window_end_exclusive: int,
) -> dict[str, object]:
    target = np.asarray(pair, dtype=np.int32)
    entry_matches = np.all(descriptors[entry_step] == target, axis=1)
    if np.sum(entry_matches) > 1:
        raise ValueError("critical pair is duplicated at target entry")
    entry_margin = 0.0
    if np.any(entry_matches):
        entry_slot = int(np.argmax(entry_matches))
        entry_weights = behavior_weights[entry_step, :, entry_slot]
        entry_margin = float(entry_weights[1] - entry_weights[0])
    rows: list[dict[str, object]] = []
    for step in range(window_start, window_end_exclusive):
        matches = np.all(descriptors[step] == target, axis=1)
        if np.sum(matches) > 1:
            raise ValueError("critical pair is duplicated in an evidence window")
        activation = 0.0
        current_margin = 0.0
        if np.any(matches):
            slot = int(np.argmax(matches))
            activation = float(representations[step, slot])
            weights = behavior_weights[step, :, slot]
            current_margin = float(weights[1] - weights[0])
        logits = behavior_logits[step]
        rows.append(
            {
                "step": step,
                "intended_action": int(intended_actions[step]),
                "online_logit_margin": float(logits[1] - logits[0]),
                "critical_activation": activation,
                "current_critical_weight_margin": current_margin,
            }
        )
    return {
        "pair": list(pair),
        "entry_step": entry_step,
        "window_start": window_start,
        "window_end_exclusive": window_end_exclusive,
        "entry_critical_weight_margin": entry_margin,
        "rows": rows,
    }


def critical_run_primitives(
    result: HiddenPartnerRunResult,
) -> dict[str, object]:
    """Extract compact fixed-window and event primitives for independent audit."""
    trace = result.trace
    cycle_steps = result.summary.cycle_steps
    deployed_pre = np.asarray(
        trace.active_descriptors,
        dtype=np.int32,
    )[:cycle_steps]
    deployed_post = np.asarray(
        trace.deployed_descriptors_post,
        dtype=np.int32,
    )[:cycle_steps]
    shadow_pre = np.asarray(
        trace.shadow_descriptors_pre,
        dtype=np.int32,
    )[:cycle_steps]
    shadow_post = np.asarray(
        trace.shadow_descriptors_post,
        dtype=np.int32,
    )[:cycle_steps]
    candidate_pre = np.asarray(
        trace.candidate_descriptors,
        dtype=np.int32,
    )[:cycle_steps]
    candidate_post = np.asarray(
        trace.candidate_descriptors_post,
        dtype=np.int32,
    )[:cycle_steps]
    deployed_states = np.concatenate((deployed_pre, deployed_post[-1:]))
    shadow_states = np.concatenate((shadow_pre, shadow_post[-1:]))
    if (
        candidate_pre.shape != (cycle_steps, 66, 2)
        or candidate_post.shape != candidate_pre.shape
        or not np.array_equal(candidate_pre[1:], candidate_post[:-1])
    ):
        raise ValueError("candidate descriptor trace is not exactly continuous")
    candidate_states = np.concatenate((candidate_pre[:1], candidate_post), axis=0)
    link_violations = np.any(deployed_pre != shadow_pre, axis=(1, 2))
    link_violations |= np.any(deployed_post != shadow_post, axis=(1, 2))
    if cycle_steps > 1:
        link_violations[1:] |= np.any(
            deployed_pre[1:] != deployed_post[:-1],
            axis=(1, 2),
        )
        link_violations[1:] |= np.any(
            shadow_pre[1:] != shadow_post[:-1],
            axis=(1, 2),
        )

    rewards = np.asarray(trace.reward, dtype=np.float64)[:cycle_steps]
    if (
        rewards.shape != (cycle_steps,)
        or not np.all(np.isfinite(rewards))
        or np.any((rewards != 0.0) & (rewards != 1.0))
    ):
        raise ValueError("hidden-partner rewards must be finite binary values")
    evidence = np.asarray(
        trace.interaction_evidence_refreshed,
        dtype=np.bool_,
    )[:cycle_steps]
    confirmed_evidence = np.asarray(
        trace.interaction_retention_evidence_refreshed,
        dtype=np.bool_,
    )[:cycle_steps]
    interaction_output_weights_pre = np.asarray(
        trace.interaction_output_weights_pre,
        dtype=np.float32,
    )[:cycle_steps]
    interaction_output_weights_post = np.asarray(
        trace.interaction_output_weights_post,
        dtype=np.float32,
    )[:cycle_steps]
    feature_streak_pre = np.asarray(
        trace.interaction_utility_evidence_streak_pre,
        dtype=np.int32,
    )[:cycle_steps]
    feature_streak_post = np.asarray(
        trace.interaction_utility_evidence_streak_post,
        dtype=np.int32,
    )[:cycle_steps]
    feature_committed_pre = np.asarray(
        trace.interaction_active_output_memory_committed_pre,
        dtype=np.bool_,
    )[:cycle_steps]
    feature_committed_post = np.asarray(
        trace.interaction_active_output_memory_committed_post,
        dtype=np.bool_,
    )[:cycle_steps]
    consumer_write_gate = np.asarray(
        trace.consumer_write_gate_pre,
        dtype=np.bool_,
    )[:cycle_steps]
    consumer_read_idle_pre = np.asarray(
        trace.consumer_read_idle_steps_pre,
        dtype=np.int32,
    )[:cycle_steps]
    consumer_read_idle_post = np.asarray(
        trace.consumer_read_idle_steps_post,
        dtype=np.int32,
    )[:cycle_steps]
    consumer_mask_pre = np.asarray(
        trace.consumer_active_mask_pre,
        dtype=np.bool_,
    )[:cycle_steps]
    consumer_mask_post = np.asarray(
        trace.consumer_active_mask_post,
        dtype=np.bool_,
    )[:cycle_steps]
    behavior_weights_pre = np.asarray(
        trace.behavior_pair_weights_pre,
        dtype=np.float32,
    )[:cycle_steps]
    behavior_weights_post = np.asarray(
        trace.behavior_pair_weights_post,
        dtype=np.float32,
    )[:cycle_steps]
    control_q_weights_pre = np.asarray(
        trace.control_pair_weights_pre,
        dtype=np.float32,
    )[:cycle_steps]
    control_q_weights_post = np.asarray(
        trace.control_pair_weights_post,
        dtype=np.float32,
    )[:cycle_steps]
    control_q_trace_pre = np.asarray(
        trace.control_pair_trace_weights_pre,
        dtype=np.float32,
    )[:cycle_steps]
    control_q_trace_post = np.asarray(
        trace.control_pair_trace_weights_post,
        dtype=np.float32,
    )[:cycle_steps]
    if any(
        values.shape != (cycle_steps, 12)
        for values in (
            evidence,
            confirmed_evidence,
            feature_streak_pre,
            feature_streak_post,
            feature_committed_pre,
            feature_committed_post,
            consumer_write_gate,
            consumer_read_idle_pre,
            consumer_read_idle_post,
            consumer_mask_pre,
            consumer_mask_post,
        )
    ):
        raise ValueError("evidence/consumer gate trace has the wrong shape")
    feature_memory_audit = _feature_memory_contract_audit(
        deployed_pre,
        deployed_post,
        shadow_pre,
        shadow_post,
        evidence,
        confirmed_evidence,
        interaction_output_weights_pre,
        interaction_output_weights_post,
        feature_streak_pre,
        feature_streak_post,
        feature_committed_pre,
        feature_committed_post,
        result.condition.config,
    )
    representations = np.asarray(
        trace.deployed_pair_features,
        dtype=np.float32,
    )[:cycle_steps]
    consumer_gate_audit = _consumer_gate_contract_audit(
        deployed_pre,
        deployed_post,
        evidence,
        consumer_write_gate,
        consumer_read_idle_pre,
        consumer_read_idle_post,
        consumer_mask_pre,
        consumer_mask_post,
        representations,
        behavior_weights_pre,
        behavior_weights_post,
        control_q_weights_pre,
        control_q_weights_post,
        control_q_trace_pre,
        control_q_trace_post,
        result.condition.config,
    )

    behavior_logits = np.asarray(
        trace.behavior_logits_preupdate,
        dtype=np.float64,
    )[:cycle_steps]
    behavior_weights = behavior_weights_pre.astype(np.float64)
    representations_for_windows = representations.astype(np.float64)
    intended_actions = np.asarray(
        trace.partner_intended_action,
        dtype=np.int64,
    )[:cycle_steps]
    if (
        behavior_logits.shape != (cycle_steps, 2)
        or behavior_weights.shape != (cycle_steps, 2, 12)
        or representations_for_windows.shape != (cycle_steps, 12)
        or intended_actions.shape != (cycle_steps,)
        or not np.all(np.isfinite(behavior_logits))
        or not np.all(np.isfinite(behavior_weights))
        or not np.all(np.isfinite(representations_for_windows))
        or np.any((intended_actions < 0) | (intended_actions >= 2))
    ):
        raise ValueError("critical behavior primitive trace is invalid")
    closed_read_violations = ~consumer_mask_pre & (representations != 0.0)
    counter_violations = ~(
        (np.asarray(trace.state_builder_step_delta)[:cycle_steps] == 1)
        & (np.asarray(trace.state_builder_learning_delta)[:cycle_steps] == 1)
        & (np.asarray(trace.behavior_step_delta)[:cycle_steps] == 1)
        & (np.asarray(trace.interaction_step_delta)[:cycle_steps] == 1)
        & (np.asarray(trace.world_step_delta)[:cycle_steps] == 1)
        & (np.asarray(trace.control_step_delta)[:cycle_steps] == 1)
        & (np.asarray(trace.router_route_delta)[:cycle_steps] == 1)
        & (np.asarray(trace.integrated_step_delta)[:cycle_steps] == 1)
    )
    causal_violations = ~(
        np.asarray(trace.route_valid, dtype=np.bool_)[:cycle_steps]
        & np.asarray(
            trace.causal_transition_valid,
            dtype=np.bool_,
        )[:cycle_steps]
    )
    finite_violations = ~np.asarray(
        trace.all_finite,
        dtype=np.bool_,
    )[:cycle_steps]

    ends = np.cumsum(
        (0, *result.summary.segment_lengths),
        dtype=np.int64,
    )
    d_start, d_end = int(ends[3]), int(ends[4])
    c_start, c_end = int(ends[5]), int(ends[6])
    recurrent_c_start = int(ends[8])
    recurrent_c_end = recurrent_c_start + RECURRENT_ENTRY_WINDOW
    return {
        "schema_version": CRITICAL_RUN_PRIMITIVES_SCHEMA,
        "cycle_steps": cycle_steps,
        "reward_one_bits": _packed_bool_payload(rewards == 1.0),
        "evidence_refresh_bits": _packed_bool_payload(evidence),
        "retention_evidence_refresh_bits": _packed_bool_payload(
            confirmed_evidence,
        ),
        "feature_memory_committed_pre_bits": _packed_bool_payload(
            feature_committed_pre,
        ),
        "feature_memory_committed_post_bits": _packed_bool_payload(
            feature_committed_post,
        ),
        "identity_routed_head_changed_bits": _packed_bool_payload(
            feature_memory_audit.identity_routed_head_changed,
        ),
        "feature_memory_contract_violation_bits": _packed_bool_payload(
            feature_memory_audit.violation_bits,
        ),
        "feature_memory_enabled": result.condition.config.evidence_gated_feature_memory,
        "consumer_write_gate_bits": _packed_bool_payload(consumer_write_gate),
        "consumer_write_contract_violation_bits": _packed_bool_payload(
            consumer_gate_audit.write_violation_bits,
        ),
        "consumer_active_mask_pre_bits": _packed_bool_payload(consumer_mask_pre),
        "consumer_active_mask_post_bits": _packed_bool_payload(consumer_mask_post),
        "closed_consumer_read_violation_bits": _packed_bool_payload(closed_read_violations),
        "representation_link_violation_bits": _packed_bool_payload(link_violations),
        "counter_contract_violation_bits": _packed_bool_payload(counter_violations),
        "causal_contract_violation_bits": _packed_bool_payload(causal_violations),
        "finite_violation_bits": _packed_bool_payload(finite_violations),
        "bank_state_rle": _canonical_bank_state_rle(
            deployed_states,
            shadow_states,
        ),
        "candidate_bank_state_rle": _canonical_candidate_bank_state_rle(
            candidate_states,
        ),
        "feature_head_state_xor": _float32_state_sequence_payload(
            interaction_output_weights_pre,
            interaction_output_weights_post,
        ),
        "behavior_pair_weight_state_xor": _float32_state_sequence_payload(
            behavior_weights_pre,
            behavior_weights_post,
        ),
        "control_q_pair_weight_state_xor": _float32_state_sequence_payload(
            control_q_weights_pre,
            control_q_weights_post,
        ),
        "control_q_trace_state_xor": _float32_state_sequence_payload(
            control_q_trace_pre,
            control_q_trace_post,
        ),
        "critical_windows": {
            "c_first_late": _critical_window_primitives(
                behavior_logits=behavior_logits,
                behavior_weights=behavior_weights,
                representations=representations_for_windows,
                descriptors=deployed_pre,
                intended_actions=intended_actions,
                pair=_C_PAIR,
                entry_step=c_start,
                window_start=c_end - FEATURE_LEARNING_WINDOW,
                window_end_exclusive=c_end,
            ),
            "d_late": _critical_window_primitives(
                behavior_logits=behavior_logits,
                behavior_weights=behavior_weights,
                representations=representations_for_windows,
                descriptors=deployed_pre,
                intended_actions=intended_actions,
                pair=_D_PAIR,
                entry_step=d_start,
                window_start=d_end - FEATURE_LEARNING_WINDOW,
                window_end_exclusive=d_end,
            ),
            "c_recurrent_early": _critical_window_primitives(
                behavior_logits=behavior_logits,
                behavior_weights=behavior_weights,
                representations=representations_for_windows,
                descriptors=deployed_pre,
                intended_actions=intended_actions,
                pair=_C_PAIR,
                entry_step=c_start,
                window_start=recurrent_c_start,
                window_end_exclusive=recurrent_c_end,
            ),
        },
    }


def critical_run_primitives_v5(
    result: HiddenPartnerRunResult,
) -> dict[str, object]:
    """Serialize additive v5 primitives after independent host reconstruction."""
    audit = audit_hidden_partner_lifecycle_v5(result)
    if audit.config_role is None:
        raise ValueError("v5 primitives require the exact frozen candidate or control")
    if not audit.all_contracts_valid:
        raise ValueError("v5 primitives require a fully valid independent host audit")
    trace = result.trace
    steps = result.summary.cycle_steps

    def values(field: str, dtype: Any) -> np.ndarray:
        return np.ascontiguousarray(
            np.asarray(getattr(trace, field), dtype=dtype)[:steps]
        )

    required_pre = values(
        "interaction_candidate_reacquisition_required_pre",
        np.bool_,
    )
    required_post = values(
        "interaction_candidate_reacquisition_required_post",
        np.bool_,
    )
    if steps > 1 and not np.array_equal(required_pre[1:], required_post[:-1]):
        raise ValueError("candidate reacquisition-required trace is not continuous")
    required_states = np.concatenate((required_pre[:1], required_post), axis=0)
    promoted = values("interaction_promoted_candidate", np.int32)
    replaced = values("interaction_replaced_slot", np.int32)
    retired_slot = values("interaction_retired_slot", np.int32)
    retired_left = values("interaction_retired_left", np.int32)
    retired_right = values("interaction_retired_right", np.int32)
    reset_mask = values("interaction_matching_candidate_reset_mask", np.bool_)
    identity_events = [
        {
            "step": step,
            "replaced_slot": int(replaced[step]),
            "promoted_candidate": int(promoted[step]),
            "retired_slot": int(retired_slot[step]),
            "retired_descriptor": [
                int(retired_left[step]),
                int(retired_right[step]),
            ],
            "matching_candidate_reset_indices": [
                int(index) for index in np.flatnonzero(reset_mask[step])
            ],
        }
        for step in range(steps)
        if promoted[step] >= 0 or retired_slot[step] >= 0
    ]
    historical = critical_run_primitives(result)
    return {
        **historical,
        "schema_version": CRITICAL_RUN_PRIMITIVES_V5_SCHEMA,
        "historical_v4_schema": CRITICAL_RUN_PRIMITIVES_SCHEMA,
        "development_only": True,
        "scientific_promotion_allowed": False,
        "reserved_namespace": CONFIRMATION_NAMESPACE,
        "reserved_namespace_status": CONFIRMATION_NAMESPACE_STATUS,
        "config_role": audit.config_role,
        "agent_config": result.condition.config.to_config(),
        "v5_lifecycle_audit": audit.to_dict(),
        "interaction_phi": _numeric_array_payload(
            values("interaction_phi_pre", np.float32),
            dtype="<f4",
        ),
        "interaction_target": _numeric_array_payload(
            values("interaction_target", np.float32),
            dtype="<f4",
        ),
        "relevance_probe_score": _numeric_array_payload(
            values("interaction_relevance_probe_scores", np.float32),
            dtype="<f4",
        ),
        "relevance_probe_error": _numeric_array_payload(
            values("interaction_relevance_probe_errors", np.float32),
            dtype="<f4",
        ),
        "candidate_promotion_signal": _numeric_array_payload(
            values("interaction_candidate_promotion_signal", np.float32),
            dtype="<f4",
        ),
        "candidate_promotion_raw_evidence_bits": _packed_bool_payload(
            values("interaction_candidate_promotion_raw_evidence", np.bool_)
        ),
        "candidate_promotion_confirmed_bits": _packed_bool_payload(
            values("interaction_candidate_promotion_confirmed", np.bool_)
        ),
        "candidate_reacquisition_required_state_bits": _packed_bool_payload(
            required_states
        ),
        "candidate_reacquisition_confirmed_bits": _packed_bool_payload(
            values("interaction_candidate_reacquisition_confirmed", np.bool_)
        ),
        "durable_read_mask_bits": _packed_bool_payload(
            values("interaction_durable_read_mask", np.bool_)
        ),
        "matching_candidate_reset_bits": _packed_bool_payload(reset_mask),
        "candidate_promotion_streak_state_xor": _int32_state_sequence_payload(
            values(
                "interaction_candidate_promotion_evidence_streak_pre",
                np.int32,
            ),
            values(
                "interaction_candidate_promotion_evidence_streak_post",
                np.int32,
            ),
        ),
        "candidate_promotion_streak_updated": _numeric_array_payload(
            values(
                "interaction_candidate_promotion_evidence_streak_updated",
                np.int32,
            ),
            dtype="<i4",
        ),
        "candidate_head_state_xor": _float32_state_sequence_payload(
            values("candidate_output_weights_pre", np.float32),
            values("candidate_output_weights_post", np.float32),
        ),
        "relevance_probe_head_state_xor": _float32_state_sequence_payload(
            values("interaction_relevance_probe_weights_pre", np.float32),
            values("interaction_relevance_probe_weights_post", np.float32),
        ),
        "relevance_probe_bias_state_xor": _float32_state_sequence_payload(
            values("interaction_relevance_probe_biases_pre", np.float32),
            values("interaction_relevance_probe_biases_post", np.float32),
        ),
        "feature_moment_state_xor": _float32_state_sequence_payload(
            values("interaction_feature_second_moments_pre", np.float32),
            values("interaction_feature_second_moments_post", np.float32),
        ),
        "candidate_moment_state_xor": _float32_state_sequence_payload(
            values("interaction_candidate_second_moments_pre", np.float32),
            values("interaction_candidate_second_moments_post", np.float32),
        ),
        "target_moment_state_xor": _float32_state_sequence_payload(
            values("interaction_target_second_moments_pre", np.float32),
            values("interaction_target_second_moments_post", np.float32),
        ),
        "feature_utility_state_xor": _float32_state_sequence_payload(
            values("active_utilities", np.float32),
            values("shadow_utilities_post", np.float32),
        ),
        "candidate_utility_state_xor": _float32_state_sequence_payload(
            values("candidate_utilities", np.float32),
            values("candidate_utilities_post", np.float32),
        ),
        "feature_age_state_xor": _int32_state_sequence_payload(
            values("interaction_feature_ages_pre", np.int32),
            values("interaction_feature_ages_post", np.int32),
        ),
        "candidate_age_state_xor": _int32_state_sequence_payload(
            values("candidate_ages_pre", np.int32),
            values("candidate_ages_post", np.int32),
        ),
        "identity_events": identity_events,
        "resource_contract": {
            "expected": dict(_V5_EXPECTED_RESOURCE_CONTRACT),
            "initial": result.initial_resource.to_dict(),
            "final": result.final_resource.to_dict(),
            "summary_initial_state_nbytes": result.summary.initial_state_nbytes,
            "summary_final_state_nbytes": result.summary.final_state_nbytes,
            "summary_resource_shape_matched": result.summary.resource_shape_matched,
        },
    }


def _first_sustained(
    values: np.ndarray,
    start: int,
    end_exclusive: int,
    window: int,
) -> int | None:
    selected = np.asarray(values[start:end_exclusive], dtype=np.int8)
    if selected.size < window:
        return None
    totals = np.convolve(
        selected,
        np.ones(window, dtype=np.int32),
        mode="valid",
    )
    hits = np.flatnonzero(totals == window)
    return None if hits.size == 0 else int(start + hits[0])


def _transition_count(values: np.ndarray, *, rising: bool) -> int:
    previous = np.asarray(values[:-1], dtype=np.bool_)
    current = np.asarray(values[1:], dtype=np.bool_)
    selected = (~previous & current) if rising else (previous & ~current)
    return int(np.sum(selected))


def _promotion_steps_from_presence(
    present_states: np.ndarray,
) -> tuple[int, ...]:
    """Return transition indices whose post-state first contains the pair."""
    present = np.asarray(present_states, dtype=np.bool_)
    if present.ndim != 1 or present.size < 2:
        raise ValueError("pair presence must contain at least two decision states")
    return tuple(int(step) for step in np.flatnonzero(~present[:-1] & present[1:]))


def _target_attributed_acquisition(
    evidence_refresh_steps: tuple[int, ...],
    present_states: np.ndarray,
    start: int,
    end_exclusive: int,
) -> int | None:
    for event_step in evidence_refresh_steps:
        effective_step = event_step + 1
        confirmation_end = effective_step + FEATURE_LEARNING_WINDOW
        if (
            start <= event_step < end_exclusive
            and confirmation_end <= end_exclusive
            and bool(present_states[event_step])
            and bool(np.all(present_states[event_step:confirmation_end]))
        ):
            return effective_step
    return None


def _target_evidence_refresh_steps(
    descriptor_slots: np.ndarray,
    evidence_refreshed: np.ndarray,
    start: int,
    end_exclusive: int,
) -> tuple[int, ...]:
    if evidence_refreshed.ndim != 2 or evidence_refreshed.shape[1] != 12:
        raise ValueError("evidence refresh trace must have shape (steps, 12)")
    steps: list[int] = []
    for step in range(start, end_exclusive):
        slot = int(descriptor_slots[step])
        if slot >= 0 and bool(evidence_refreshed[step, slot]):
            steps.append(step)
    return tuple(steps)


@dataclasses.dataclass(frozen=True)
class _CriticalColumnWindowMetrics:
    online_nll: float
    entry_frozen_nll: float
    learning_nll_gain: float
    learning_positive_fraction: float
    entry_frozen_accuracy: float
    learning_accuracy_gain: float
    masked_nll_increase: float
    masked_nll_positive_fraction: float


def _softmax_nll_and_prediction(
    logits: np.ndarray,
    intended_action: int,
) -> tuple[float, int]:
    centered = np.asarray(logits, dtype=np.float64) - float(np.max(logits))
    log_normalizer = math.log(float(np.sum(np.exp(centered))))
    return (
        float(-centered[intended_action] + log_normalizer),
        int(np.argmax(centered)),
    )


def _critical_column_window_metrics(
    behavior_logits: np.ndarray,
    behavior_weights: np.ndarray,
    representations: np.ndarray,
    descriptors: np.ndarray,
    intended_actions: np.ndarray,
    pair: tuple[int, int],
    *,
    entry_step: int,
    window_start: int,
    window_end_exclusive: int,
    temperature: float = 1.0,
) -> _CriticalColumnWindowMetrics:
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("behavior temperature must be finite and positive")
    target = np.asarray(pair, dtype=np.int32)
    entry_matches = np.all(descriptors[entry_step] == target, axis=1)
    if np.sum(entry_matches) > 1:
        raise ValueError("critical pair is duplicated at target entry")
    entry_weights = (
        np.zeros((2,), dtype=np.float64)
        if not np.any(entry_matches)
        else behavior_weights[
            entry_step,
            :,
            int(np.argmax(entry_matches)),
        ]
    )
    online_nll: list[float] = []
    counterfactual_nll: list[float] = []
    zero_column_nll: list[float] = []
    online_correct: list[float] = []
    counterfactual_correct: list[float] = []
    for step in range(window_start, window_end_exclusive):
        matches = np.all(descriptors[step] == target, axis=1)
        if np.sum(matches) > 1:
            raise ValueError("critical pair is duplicated in an evaluation window")
        if np.any(matches):
            slot = int(np.argmax(matches))
            current_weights = behavior_weights[step, :, slot]
            feature_value = representations[step, slot]
        else:
            current_weights = np.zeros((2,), dtype=np.float64)
            feature_value = 0.0
        online_logits = behavior_logits[step] / temperature
        counterfactual_logits = (
            behavior_logits[step] + (entry_weights - current_weights) * feature_value
        ) / temperature
        zero_logits = (behavior_logits[step] - current_weights * feature_value) / temperature
        intended = int(intended_actions[step])
        online_loss, online_prediction = _softmax_nll_and_prediction(
            online_logits,
            intended,
        )
        counterfactual_loss, counterfactual_prediction = _softmax_nll_and_prediction(
            counterfactual_logits,
            intended,
        )
        zero_loss, _ = _softmax_nll_and_prediction(
            zero_logits,
            intended,
        )
        online_nll.append(online_loss)
        counterfactual_nll.append(counterfactual_loss)
        zero_column_nll.append(zero_loss)
        online_correct.append(float(online_prediction == intended))
        counterfactual_correct.append(float(counterfactual_prediction == intended))
    online_mean = float(np.mean(online_nll))
    counterfactual_mean = float(np.mean(counterfactual_nll))
    per_step_gain = np.asarray(counterfactual_nll) - np.asarray(online_nll)
    per_step_masked_increase = np.asarray(zero_column_nll) - np.asarray(online_nll)
    online_accuracy = float(np.mean(online_correct))
    counterfactual_accuracy = float(np.mean(counterfactual_correct))
    return _CriticalColumnWindowMetrics(
        online_nll=online_mean,
        entry_frozen_nll=counterfactual_mean,
        learning_nll_gain=counterfactual_mean - online_mean,
        learning_positive_fraction=float(np.mean(per_step_gain > 0.0)),
        entry_frozen_accuracy=counterfactual_accuracy,
        learning_accuracy_gain=(online_accuracy - counterfactual_accuracy),
        masked_nll_increase=float(np.mean(per_step_masked_increase)),
        masked_nll_positive_fraction=float(np.mean(per_step_masked_increase > 0.0)),
    )


def _window_mean(values: np.ndarray, start: int, end_exclusive: int) -> float:
    selected = np.asarray(values[start:end_exclusive], dtype=np.float64)
    if selected.size == 0 or not np.all(np.isfinite(selected)):
        raise ValueError("lifecycle performance window is empty or non-finite")
    return float(np.mean(selected))


def summarize_critical_lifecycle_v2(
    result: HiddenPartnerRunResult,
) -> CriticalLifecycleV2Summary:
    """Reconstruct continuous lifecycle facts from one primitive decision trace."""
    summary = result.summary
    trace = result.trace
    cycle_steps = summary.cycle_steps
    deployed_pre = np.asarray(
        trace.active_descriptors,
        dtype=np.int32,
    )[:cycle_steps]
    deployed_post = np.asarray(
        trace.deployed_descriptors_post,
        dtype=np.int32,
    )[:cycle_steps]
    shadow_pre = np.asarray(
        trace.shadow_descriptors_pre,
        dtype=np.int32,
    )[:cycle_steps]
    shadow_post = np.asarray(
        trace.shadow_descriptors_post,
        dtype=np.int32,
    )[:cycle_steps]
    candidates = np.asarray(trace.candidate_descriptors, dtype=np.int32)[:cycle_steps]
    candidates_post = np.asarray(
        trace.candidate_descriptors_post,
        dtype=np.int32,
    )[:cycle_steps]
    candidate_utilities_post = np.asarray(
        trace.candidate_utilities_post,
        dtype=np.float64,
    )[:cycle_steps]
    candidate_output_weights_post = np.asarray(
        trace.candidate_output_weights_post,
        dtype=np.float64,
    )[:cycle_steps]
    candidate_ages_post = np.asarray(
        trace.candidate_ages_post,
        dtype=np.int64,
    )[:cycle_steps]
    rewards = np.asarray(trace.reward, dtype=np.float64)[:cycle_steps]
    intended_correct = np.asarray(
        trace.behavior_intended_correct,
        dtype=np.bool_,
    )[:cycle_steps]
    intended_actions = np.asarray(
        trace.partner_intended_action,
        dtype=np.int64,
    )[:cycle_steps]
    behavior_logits = np.asarray(
        trace.behavior_logits_preupdate,
        dtype=np.float64,
    )[:cycle_steps]
    behavior_weights = np.asarray(
        trace.behavior_pair_weights_pre,
        dtype=np.float64,
    )[:cycle_steps]
    behavior_weights_post = np.asarray(
        trace.behavior_pair_weights_post,
        dtype=np.float32,
    )[:cycle_steps]
    control_q_weights_pre = np.asarray(
        trace.control_pair_weights_pre,
        dtype=np.float32,
    )[:cycle_steps]
    control_q_weights_post = np.asarray(
        trace.control_pair_weights_post,
        dtype=np.float32,
    )[:cycle_steps]
    control_q_trace_pre = np.asarray(
        trace.control_pair_trace_weights_pre,
        dtype=np.float32,
    )[:cycle_steps]
    control_q_trace_post = np.asarray(
        trace.control_pair_trace_weights_post,
        dtype=np.float32,
    )[:cycle_steps]
    representations = np.asarray(
        trace.deployed_pair_features,
        dtype=np.float64,
    )[:cycle_steps]
    evidence_refreshed = np.asarray(
        trace.interaction_evidence_refreshed,
        dtype=np.bool_,
    )[:cycle_steps]
    retention_evidence_refreshed = np.asarray(
        trace.interaction_retention_evidence_refreshed,
        dtype=np.bool_,
    )[:cycle_steps]
    interaction_output_weights_pre = np.asarray(
        trace.interaction_output_weights_pre,
        dtype=np.float32,
    )[:cycle_steps]
    interaction_output_weights_post = np.asarray(
        trace.interaction_output_weights_post,
        dtype=np.float32,
    )[:cycle_steps]
    feature_streak_pre = np.asarray(
        trace.interaction_utility_evidence_streak_pre,
        dtype=np.int32,
    )[:cycle_steps]
    feature_streak_post = np.asarray(
        trace.interaction_utility_evidence_streak_post,
        dtype=np.int32,
    )[:cycle_steps]
    feature_committed_pre = np.asarray(
        trace.interaction_active_output_memory_committed_pre,
        dtype=np.bool_,
    )[:cycle_steps]
    feature_committed_post = np.asarray(
        trace.interaction_active_output_memory_committed_post,
        dtype=np.bool_,
    )[:cycle_steps]
    consumer_write_gate = np.asarray(
        trace.consumer_write_gate_pre,
        dtype=np.bool_,
    )[:cycle_steps]
    consumer_read_idle_pre = np.asarray(
        trace.consumer_read_idle_steps_pre,
        dtype=np.int32,
    )[:cycle_steps]
    consumer_read_idle_post = np.asarray(
        trace.consumer_read_idle_steps_post,
        dtype=np.int32,
    )[:cycle_steps]
    consumer_mask_pre = np.asarray(
        trace.consumer_active_mask_pre,
        dtype=np.bool_,
    )[:cycle_steps]
    consumer_mask_post = np.asarray(
        trace.consumer_active_mask_post,
        dtype=np.bool_,
    )[:cycle_steps]
    expected_descriptor_shape = (cycle_steps, 12, 2)
    for name, values in (
        ("deployed_pre", deployed_pre),
        ("deployed_post", deployed_post),
        ("shadow_pre", shadow_pre),
        ("shadow_post", shadow_post),
    ):
        if values.shape != expected_descriptor_shape:
            raise ValueError(f"{name} descriptor trace has the wrong shape")
    if cycle_steps <= 0:
        raise ValueError("cycle must contain at least one decision")
    deployed_states = np.concatenate(
        (deployed_pre, deployed_post[-1:]),
        axis=0,
    )
    shadow_states = np.concatenate(
        (shadow_pre, shadow_post[-1:]),
        axis=0,
    )
    decision_state_count = cycle_steps + 1
    reset_mask = np.asarray(
        trace.interaction_matching_candidate_reset_mask,
        dtype=np.bool_,
    )[:cycle_steps]
    reset_count = np.asarray(
        trace.interaction_matching_candidate_reset_count,
        dtype=np.int64,
    )[:cycle_steps]
    representation_link_contract = bool(
        np.array_equal(deployed_pre[1:], deployed_post[:-1])
        and np.array_equal(shadow_pre[1:], shadow_post[:-1])
        and np.array_equal(deployed_pre, shadow_pre)
        and np.array_equal(deployed_post, shadow_post)
        and reset_mask.shape == (cycle_steps, 66)
        and np.array_equal(
            reset_count,
            np.sum(reset_mask, axis=1, dtype=np.int64),
        )
    )
    consumer_gate_contract = _consumer_gate_contract_valid(
        deployed_pre,
        deployed_post,
        evidence_refreshed,
        consumer_write_gate,
        consumer_read_idle_pre,
        consumer_read_idle_post,
        consumer_mask_pre,
        consumer_mask_post,
        representations,
        behavior_weights.astype(np.float32),
        behavior_weights_post,
        control_q_weights_pre,
        control_q_weights_post,
        control_q_trace_pre,
        control_q_trace_post,
        result.condition.config,
    )
    feature_memory_enabled = result.condition.config.evidence_gated_feature_memory
    feature_memory_contract = _feature_memory_contract_audit(
        deployed_pre,
        deployed_post,
        shadow_pre,
        shadow_post,
        evidence_refreshed,
        retention_evidence_refreshed,
        interaction_output_weights_pre,
        interaction_output_weights_post,
        feature_streak_pre,
        feature_streak_post,
        feature_committed_pre,
        feature_committed_post,
        result.condition.config,
    ).valid

    c_deployed_slots = _descriptor_slots(
        deployed_states,
        _C_PAIR,
        require_exactly_one=False,
    )
    d_deployed_slots = _descriptor_slots(
        deployed_states,
        _D_PAIR,
        require_exactly_one=False,
    )
    c_shadow_slots = _descriptor_slots(
        shadow_states,
        _C_PAIR,
        require_exactly_one=False,
    )
    d_shadow_slots = _descriptor_slots(
        shadow_states,
        _D_PAIR,
        require_exactly_one=False,
    )
    candidate_archive_contract = _candidate_archive_contract_valid(
        candidates,
        candidates_post,
    )
    c_candidate_slots = _descriptor_slots(
        candidates,
        _C_PAIR,
        require_exactly_one=False,
    )
    d_candidate_slots = _descriptor_slots(
        candidates,
        _D_PAIR,
        require_exactly_one=False,
    )
    canonical_candidates = _canonical_candidate_descriptors()
    c_canonical_slot = int(
        np.flatnonzero(
            np.all(canonical_candidates == np.asarray(_C_PAIR), axis=1)
        )[0]
    )
    d_canonical_slot = int(
        np.flatnonzero(
            np.all(canonical_candidates == np.asarray(_D_PAIR), axis=1)
        )[0]
    )
    c_candidate_slots = np.where(
        c_candidate_slots >= 0,
        c_candidate_slots,
        c_canonical_slot,
    ).astype(np.int32)
    d_candidate_slots = np.where(
        d_candidate_slots >= 0,
        d_candidate_slots,
        d_canonical_slot,
    ).astype(np.int32)
    c_candidate_state_slots = np.concatenate(
        (c_candidate_slots, c_candidate_slots[-1:]),
    )
    d_candidate_state_slots = np.concatenate(
        (d_candidate_slots, d_candidate_slots[-1:]),
    )
    c_present = c_deployed_slots >= 0
    d_present = d_deployed_slots >= 0
    c_mismatch_steps = int(np.sum(c_deployed_slots != c_shadow_slots))
    d_mismatch_steps = int(np.sum(d_deployed_slots != d_shadow_slots))

    ends = np.cumsum((0, *summary.segment_lengths), dtype=np.int64)
    d_start, d_end = int(ends[3]), int(ends[4])
    first_c_start, first_c_end = int(ends[5]), int(ends[6])
    recurrent_c_start = int(ends[8])

    # Promotion is a representation state transition, not a trusted event
    # label.  Deriving it from the T+1 decision-state sequence prevents an
    # omitted repromotion diagnostic from manufacturing stable retirement.
    c_promotion_steps = _promotion_steps_from_presence(c_present)
    d_promotion_steps = _promotion_steps_from_presence(d_present)
    c_target_refresh_steps = _target_evidence_refresh_steps(
        c_deployed_slots,
        evidence_refreshed,
        first_c_start,
        first_c_end,
    )
    d_target_refresh_steps = _target_evidence_refresh_steps(
        d_deployed_slots,
        evidence_refreshed,
        d_start,
        d_end,
    )
    c_acquisition = _target_attributed_acquisition(
        c_target_refresh_steps,
        c_present,
        first_c_start,
        first_c_end,
    )
    c_first_late_start = first_c_end - FEATURE_LEARNING_WINDOW
    c_first_late_reward = _window_mean(
        rewards,
        c_first_late_start,
        first_c_end,
    )
    c_first_late_accuracy = _window_mean(
        intended_correct,
        c_first_late_start,
        first_c_end,
    )
    c_first_window = _critical_column_window_metrics(
        behavior_logits,
        behavior_weights,
        representations,
        deployed_pre,
        intended_actions,
        _C_PAIR,
        entry_step=first_c_start,
        window_start=c_first_late_start,
        window_end_exclusive=first_c_end,
    )
    c_target_created_share = c_first_window.learning_nll_gain / max(
        c_first_window.masked_nll_increase,
        1e-12,
    )
    c_task_learned = (
        representation_link_contract
        and c_mismatch_steps == 0
        and c_acquisition is not None
        and c_first_late_reward >= INITIAL_LATE_REWARD_THRESHOLD
        and c_first_late_accuracy >= CRITICAL_LATE_PREDICTION_ACCURACY_THRESHOLD
        and c_first_window.learning_nll_gain >= CRITICAL_COLUMN_LEARNING_NLL_GAIN_THRESHOLD
        and c_first_window.learning_positive_fraction
        >= CRITICAL_COLUMN_LEARNING_POSITIVE_FRACTION_THRESHOLD
        and c_target_created_share >= CRITICAL_COLUMN_TARGET_CREATED_SHARE_THRESHOLD
        and c_first_window.masked_nll_increase >= CRITICAL_MASKED_NLL_INCREASE_THRESHOLD
        and c_first_window.masked_nll_positive_fraction
        >= CRITICAL_MASKED_NLL_POSITIVE_FRACTION_THRESHOLD
    )
    c_survival_end = min(
        cycle_steps,
        recurrent_c_start + RECURRENT_ENTRY_WINDOW,
    )
    if c_acquisition is None:
        c_gap_steps = None
        c_first_gap = None
        c_evictions = 0
        c_repromotions = 0
        c_continuous = False
    else:
        c_interval = c_present[c_acquisition:c_survival_end]
        missing = np.flatnonzero(~c_interval)
        c_gap_steps = int(missing.size)
        c_first_gap = None if missing.size == 0 else int(c_acquisition + missing[0])
        c_evictions = _transition_count(c_interval, rising=False)
        c_repromotions = sum(
            c_acquisition < event_step + 1 < c_survival_end for event_step in c_promotion_steps
        )
        c_continuous = c_gap_steps == 0

    recurrent_c_end = min(
        cycle_steps,
        recurrent_c_start + RECURRENT_ENTRY_WINDOW,
    )
    c_early_reward = _window_mean(
        rewards,
        recurrent_c_start,
        recurrent_c_end,
    )
    c_retention_ratio = (c_early_reward - CHANCE_REWARD) / max(
        c_first_late_reward - CHANCE_REWARD,
        1e-7,
    )
    c_early_accuracy = _window_mean(
        intended_correct,
        recurrent_c_start,
        recurrent_c_end,
    )
    c_recurrent_window = _critical_column_window_metrics(
        behavior_logits,
        behavior_weights,
        representations,
        deployed_pre,
        intended_actions,
        _C_PAIR,
        entry_step=first_c_start,
        window_start=recurrent_c_start,
        window_end_exclusive=recurrent_c_end,
    )
    c_retained_and_used = (
        c_task_learned
        and c_continuous
        and c_early_reward >= RECURRENT_EARLY_REWARD_THRESHOLD
        and c_retention_ratio >= RETENTION_RATIO_THRESHOLD
        and c_early_accuracy >= CRITICAL_LATE_PREDICTION_ACCURACY_THRESHOLD
        and c_recurrent_window.masked_nll_increase >= CRITICAL_MASKED_NLL_INCREASE_THRESHOLD
        and c_recurrent_window.masked_nll_positive_fraction
        >= CRITICAL_MASKED_NLL_POSITIVE_FRACTION_THRESHOLD
    )

    d_acquisition = _target_attributed_acquisition(
        d_target_refresh_steps,
        d_present,
        d_start,
        d_end,
    )
    d_through_exit = bool(np.all(d_present[d_end - FEATURE_LEARNING_WINDOW : d_end]))
    d_late_start = d_end - FEATURE_LEARNING_WINDOW
    d_late_reward = _window_mean(rewards, d_late_start, d_end)
    d_late_accuracy = _window_mean(
        intended_correct,
        d_late_start,
        d_end,
    )
    d_late_window = _critical_column_window_metrics(
        behavior_logits,
        behavior_weights,
        representations,
        deployed_pre,
        intended_actions,
        _D_PAIR,
        entry_step=d_start,
        window_start=d_late_start,
        window_end_exclusive=d_end,
    )
    d_target_created_share = d_late_window.learning_nll_gain / max(
        d_late_window.masked_nll_increase,
        1e-12,
    )
    d_task_learned = (
        representation_link_contract
        and d_mismatch_steps == 0
        and d_acquisition is not None
        and d_through_exit
        and d_late_reward >= INITIAL_LATE_REWARD_THRESHOLD
        and d_late_accuracy >= CRITICAL_LATE_PREDICTION_ACCURACY_THRESHOLD
        and d_late_window.learning_nll_gain >= CRITICAL_COLUMN_LEARNING_NLL_GAIN_THRESHOLD
        and d_late_window.learning_positive_fraction
        >= CRITICAL_COLUMN_LEARNING_POSITIVE_FRACTION_THRESHOLD
        and d_target_created_share >= CRITICAL_COLUMN_TARGET_CREATED_SHARE_THRESHOLD
        and d_late_window.masked_nll_increase >= CRITICAL_MASKED_NLL_INCREASE_THRESHOLD
        and d_late_window.masked_nll_positive_fraction
        >= CRITICAL_MASKED_NLL_POSITIVE_FRACTION_THRESHOLD
    )
    d_post_exit = d_present[d_end:cycle_steps]
    d_post_exit_steps = int(np.sum(d_post_exit))
    d_post_exit_promotions = sum(event_step >= d_end for event_step in d_promotion_steps)
    d_final_absent = bool(np.all(~d_present[-(FINAL_ABSENCE_WINDOW + 1) :]))

    retired_left = np.asarray(trace.interaction_retired_left, dtype=np.int32)[:cycle_steps]
    retired_right = np.asarray(trace.interaction_retired_right, dtype=np.int32)[:cycle_steps]
    d_retired_events = (retired_left == _D_PAIR[0]) & (retired_right == _D_PAIR[1])
    d_event_steps = tuple(int(step) for step in np.flatnonzero(d_retired_events))
    d_event_reset_counts: list[int] = []
    d_event_candidate_utilities: list[float] = []
    d_event_candidate_head_linf: list[float] = []
    d_event_candidate_ages: list[int] = []
    for event_step in d_event_steps:
        candidate_index = int(d_candidate_slots[event_step])
        d_event_reset_counts.append(int(reset_mask[event_step, candidate_index]))
        d_event_candidate_utilities.append(
            float(candidate_utilities_post[event_step, candidate_index])
        )
        d_event_candidate_head_linf.append(
            float(
                np.max(
                    np.abs(
                        candidate_output_weights_post[
                            event_step,
                            :,
                            candidate_index,
                        ]
                    )
                )
            )
        )
        d_event_candidate_ages.append(int(candidate_ages_post[event_step, candidate_index]))

    d_retirement_event_step: int | None = None
    d_retirement: int | None = None
    d_event_latency: int | None = None
    d_latency: int | None = None
    d_linked_reset: int | None = None
    d_linked_utility: float | None = None
    d_linked_head_linf: float | None = None
    d_linked_age: int | None = None
    for event_index, event_step in enumerate(d_event_steps):
        effective_step = event_step + 1
        confirmation_end = effective_step + RETIREMENT_CONFIRMATION_WINDOW
        aligned = (
            event_step >= d_end
            and confirmation_end <= decision_state_count
            and bool(d_present[event_step])
            and not bool(d_present[effective_step])
            and bool(np.all(~d_present[effective_step:confirmation_end]))
            and d_event_reset_counts[event_index] == 1
            and d_event_candidate_utilities[event_index] == 0.0
            and d_event_candidate_head_linf[event_index] == 0.0
            and d_event_candidate_ages[event_index] == 0
            and event_step not in d_promotion_steps
        )
        if aligned:
            d_retirement_event_step = event_step
            d_retirement = effective_step
            d_event_latency = event_step - d_end
            d_latency = effective_step - d_end
            d_linked_reset = d_event_reset_counts[event_index]
            d_linked_utility = d_event_candidate_utilities[event_index]
            d_linked_head_linf = d_event_candidate_head_linf[event_index]
            d_linked_age = d_event_candidate_ages[event_index]
            break
    d_repromotions = (
        0
        if d_retirement_event_step is None
        else sum(event_step > d_retirement_event_step for event_step in d_promotion_steps)
    )
    d_retirement_event_count = len(d_event_steps)
    d_candidate_reset_count = int(sum(d_event_reset_counts))
    d_event_aligned = d_retirement_event_step is not None
    d_stably_retired = d_task_learned and d_event_aligned and d_final_absent and d_repromotions == 0

    c_candidate_index = int(c_candidate_slots[-1])
    d_candidate_index = int(d_candidate_slots[-1])
    joint_memory_management_success = bool(
        representation_link_contract
        and consumer_gate_contract
        and feature_memory_enabled
        and feature_memory_contract
        and candidate_archive_contract
        and c_retained_and_used
        and d_stably_retired
    )
    return CriticalLifecycleV2Summary(
        cycle_steps=cycle_steps,
        decision_state_count=decision_state_count,
        representation_link_contract_valid=representation_link_contract,
        consumer_gate_contract_valid=consumer_gate_contract,
        feature_memory_enabled=feature_memory_enabled,
        feature_memory_contract_valid=feature_memory_contract,
        c_shadow_deployed_mismatch_steps=c_mismatch_steps,
        d_shadow_deployed_mismatch_steps=d_mismatch_steps,
        c_promotion_event_steps=c_promotion_steps,
        c_target_evidence_refresh_steps=c_target_refresh_steps,
        c_acquisition_step=c_acquisition,
        c_first_late_reward=c_first_late_reward,
        c_first_late_intended_accuracy=c_first_late_accuracy,
        c_first_late_online_nll=c_first_window.online_nll,
        c_first_late_entry_frozen_critical_nll=(c_first_window.entry_frozen_nll),
        c_critical_column_learning_nll_gain=(c_first_window.learning_nll_gain),
        c_critical_column_learning_positive_fraction=(c_first_window.learning_positive_fraction),
        c_critical_column_target_created_share=c_target_created_share,
        c_first_late_entry_frozen_critical_accuracy=(c_first_window.entry_frozen_accuracy),
        c_critical_column_learning_accuracy_gain=(c_first_window.learning_accuracy_gain),
        c_first_late_masked_nll_increase=(c_first_window.masked_nll_increase),
        c_first_late_masked_nll_positive_fraction=(c_first_window.masked_nll_positive_fraction),
        c_task_learned=c_task_learned,
        c_survival_end_exclusive=c_survival_end,
        c_survival_gap_steps=c_gap_steps,
        c_first_survival_gap_step=c_first_gap,
        c_evictions_after_acquisition=c_evictions,
        c_repromotions_after_acquisition=c_repromotions,
        c_continuously_survived=c_continuous,
        c_recurrent_early_reward=c_early_reward,
        c_recurrent_early_excess_reward_retention=c_retention_ratio,
        c_recurrent_early_intended_accuracy=c_early_accuracy,
        c_recurrent_early_masked_nll_increase=(c_recurrent_window.masked_nll_increase),
        c_recurrent_early_masked_nll_positive_fraction=(
            c_recurrent_window.masked_nll_positive_fraction
        ),
        c_retained_and_used=c_retained_and_used,
        d_promotion_event_steps=d_promotion_steps,
        d_target_evidence_refresh_steps=d_target_refresh_steps,
        d_acquisition_step=d_acquisition,
        d_deployed_through_exit=d_through_exit,
        d_late_reward=d_late_reward,
        d_late_intended_accuracy=d_late_accuracy,
        d_late_online_nll=d_late_window.online_nll,
        d_late_entry_frozen_critical_nll=(d_late_window.entry_frozen_nll),
        d_critical_column_learning_nll_gain=(d_late_window.learning_nll_gain),
        d_critical_column_learning_positive_fraction=(d_late_window.learning_positive_fraction),
        d_critical_column_target_created_share=d_target_created_share,
        d_late_entry_frozen_critical_accuracy=(d_late_window.entry_frozen_accuracy),
        d_critical_column_learning_accuracy_gain=(d_late_window.learning_accuracy_gain),
        d_late_masked_nll_increase=d_late_window.masked_nll_increase,
        d_late_masked_nll_positive_fraction=(d_late_window.masked_nll_positive_fraction),
        d_task_learned=d_task_learned,
        d_retirement_event_step=d_retirement_event_step,
        d_retirement_step=d_retirement,
        d_retirement_event_latency_steps=d_event_latency,
        d_retirement_latency_steps=d_latency,
        d_post_exit_live_slot_steps=d_post_exit_steps,
        d_post_exit_live_fraction=d_post_exit_steps / max(cycle_steps - d_end, 1),
        d_post_exit_promotion_count=d_post_exit_promotions,
        d_repromotions_after_retirement=d_repromotions,
        d_absent_entire_final_window=d_final_absent,
        d_retirement_event_steps=d_event_steps,
        d_retirement_event_reset_counts=tuple(d_event_reset_counts),
        d_retirement_event_candidate_utility_post=tuple(d_event_candidate_utilities),
        d_retirement_event_candidate_head_linf_post=tuple(d_event_candidate_head_linf),
        d_retirement_event_candidate_age_post=tuple(d_event_candidate_ages),
        d_retirement_event_count=d_retirement_event_count,
        d_matching_candidate_reset_count=d_candidate_reset_count,
        d_linked_matching_candidate_reset_count=d_linked_reset,
        d_linked_candidate_utility_post=d_linked_utility,
        d_linked_candidate_head_linf_post=d_linked_head_linf,
        d_linked_candidate_age_post=d_linked_age,
        d_retirement_event_aligned=d_event_aligned,
        d_learned_then_stably_retired=d_stably_retired,
        joint_memory_management_success=joint_memory_management_success,
        candidate_archive_contract_valid=candidate_archive_contract,
        c_candidate_utility_at_life_end=float(candidate_utilities_post[-1, c_candidate_index]),
        d_candidate_utility_at_life_end=float(candidate_utilities_post[-1, d_candidate_index]),
        c_lifecycle_rle=_canonical_lifecycle_rle(
            c_deployed_slots,
            c_shadow_slots,
            c_candidate_state_slots,
        ),
        d_lifecycle_rle=_canonical_lifecycle_rle(
            d_deployed_slots,
            d_shadow_slots,
            d_candidate_state_slots,
        ),
    )


def _cell_digest(cell: EvidenceLeaseTuningCell) -> str:
    serialized = json.dumps(
        cell.to_dict(),
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(serialized).hexdigest()


def _all_run_contracts_valid(
    summary: HiddenPartnerRunSummary,
    lifecycle: CriticalLifecycleV2Summary,
) -> bool:
    """Return the fail-closed contract predicate used by grid feasibility."""
    return bool(
        summary.all_finite
        and summary.counter_contract_valid
        and summary.causal_contract_valid
        and summary.resource_shape_matched
        and lifecycle.representation_link_contract_valid
        and lifecycle.consumer_gate_contract_valid
        and lifecycle.feature_memory_enabled
        and lifecycle.feature_memory_contract_valid
        and lifecycle.candidate_archive_contract_valid
    )


def run_evidence_lease_tuning_grid(
    *,
    seed_pairs: Sequence[HiddenPartnerSeedPair] | None = None,
    protocol: HiddenPartnerDevelopmentProtocol | None = None,
) -> dict[str, object]:
    """Execute the reserved v4 development grid and select only feasible cells."""
    require_lease_tuning_execution_allowed()
    resolved_seeds = (
        derive_hidden_partner_seed_pairs(
            LEASE_TUNING_NAMESPACE,
            LEASE_TUNING_SEED_COUNT,
        )
        if seed_pairs is None
        else tuple(seed_pairs)
    )
    expected_seeds = derive_hidden_partner_seed_pairs(
        LEASE_TUNING_NAMESPACE,
        LEASE_TUNING_SEED_COUNT,
    )
    if resolved_seeds != expected_seeds:
        raise ValueError("tuning seeds must be the exact derived eight-pair v4 grid")
    resolved_protocol = HiddenPartnerDevelopmentProtocol() if protocol is None else protocol
    if resolved_protocol.to_config() != HiddenPartnerDevelopmentProtocol().to_config():
        raise ValueError("lease tuning requires the exact default hidden-partner protocol")
    run_records: list[dict[str, object]] = []
    aggregates: list[dict[str, object]] = []
    required_successes = math.ceil(MINIMUM_JOINT_SUCCESS_FRACTION * len(resolved_seeds))
    for cell in LEASE_TUNING_GRID:
        condition = HiddenPartnerCondition(
            name="full",
            config=cell.agent_config(),
            isolated_question=f"evidence-lease tuning cell {cell.index}",
        )
        runner = HiddenPartnerDevelopmentRunner(condition, resolved_protocol)
        cell_digest = _cell_digest(cell)
        condition_config = condition.to_config()
        cell_records: list[tuple[HiddenPartnerRunResult, CriticalLifecycleV2Summary]] = []
        for seed_pair in resolved_seeds:
            result = runner.run(seed_pair)
            lifecycle = summarize_critical_lifecycle_v2(result)
            run_primitives = critical_run_primitives(result)
            cell_records.append((result, lifecycle))
            run_records.append(
                {
                    "cell_index": cell.index,
                    "cell_digest": cell_digest,
                    "condition_config": condition_config,
                    "seed_pair": seed_pair.to_dict(),
                    "run_summary": result.summary.to_dict(),
                    "critical_lifecycle": lifecycle.to_dict(),
                    "critical_run_primitives": run_primitives,
                }
            )
        summaries = [item[0].summary for item in cell_records]
        lifecycles = [item[1] for item in cell_records]
        rewards = np.asarray(
            [summary.mean_reward for summary in summaries],
            dtype=np.float64,
        )
        joint_count = sum(lifecycle.joint_memory_management_success for lifecycle in lifecycles)
        c_count = sum(lifecycle.c_retained_and_used for lifecycle in lifecycles)
        d_count = sum(lifecycle.d_learned_then_stably_retired for lifecycle in lifecycles)
        latencies = [
            lifecycle.d_retirement_latency_steps
            for lifecycle in lifecycles
            if lifecycle.d_retirement_latency_steps is not None
        ]
        contracts_valid = all(
            _all_run_contracts_valid(summary, lifecycle)
            for summary, lifecycle in zip(
                summaries,
                lifecycles,
                strict=True,
            )
        )
        feasible = (
            contracts_valid
            and float(np.mean(rewards)) >= 0.85
            and float(np.min(rewards)) >= 0.80
            and joint_count >= required_successes
            and c_count >= required_successes
            and d_count >= required_successes
        )
        aggregates.append(
            {
                "cell_index": cell.index,
                "cell_digest": cell_digest,
                "seed_count": len(resolved_seeds),
                "required_success_count": required_successes,
                "finite_contract_valid": contracts_valid,
                "mean_reward": float(np.mean(rewards)),
                "minimum_seed_reward": float(np.min(rewards)),
                "joint_success_count": joint_count,
                "c_retained_and_used_count": c_count,
                "d_learned_then_stably_retired_count": d_count,
                "total_d_repromotions": int(
                    sum(lifecycle.d_repromotions_after_retirement for lifecycle in lifecycles)
                ),
                "median_d_retirement_latency_steps": (
                    None if not latencies else float(np.median(latencies))
                ),
                "feasible": feasible,
            }
        )

    feasible_cells = [record for record in aggregates if record["feasible"]]
    selected: dict[str, object] | None = None
    if feasible_cells:
        selected = max(
            feasible_cells,
            key=lambda record: (
                cast(int, record["joint_success_count"]),
                cast(float, record["minimum_seed_reward"]),
                cast(float, record["mean_reward"]),
                -cast(int, record["total_d_repromotions"]),
                -cast(
                    float,
                    record["median_d_retirement_latency_steps"],
                ),
                -cast(int, record["cell_index"]),
            ),
        )
    return {
        "schema_version": HIDDEN_PARTNER_LIFECYCLE_V2_SCHEMA,
        "development_only": True,
        "scientific_promotion_allowed": False,
        "protocol": resolved_protocol.to_config(),
        "seed_namespace": LEASE_TUNING_NAMESPACE,
        "seed_pairs": [pair.to_dict() for pair in resolved_seeds],
        "grid": [cell.to_dict() for cell in LEASE_TUNING_GRID],
        "selection_rule": TUNING_SELECTION_RULE,
        "aggregates": aggregates,
        "selected_cell": selected,
        "runs": run_records,
        "scope_limits": list(LEASE_TUNING_SCOPE_LIMITS),
    }


__all__ = [
    "CRITICAL_LATE_PREDICTION_ACCURACY_THRESHOLD",
    "CRITICAL_RUN_PRIMITIVES_SCHEMA",
    "CRITICAL_RUN_PRIMITIVES_V5_SCHEMA",
    "CONFIRMATION_NAMESPACE",
    "CONFIRMATION_NAMESPACE_STATUS",
    "CriticalLifecycleV2Summary",
    "CriticalPairLifecycleInterval",
    "EvidenceLeaseTuningCell",
    "FEATURE_LEARNING_WINDOW",
    "FINAL_ABSENCE_WINDOW",
    "HIDDEN_PARTNER_LIFECYCLE_V2_SCHEMA",
    "HIDDEN_PARTNER_LIFECYCLE_V5_SCHEMA",
    "HiddenPartnerLifecycleV5Audit",
    "LEASE_TUNING_GRID",
    "LEASE_TUNING_NAMESPACE",
    "LEASE_TUNING_NAMESPACE_STATUS",
    "LEASE_TUNING_SEED_COUNT",
    "MINIMUM_JOINT_SUCCESS_FRACTION",
    "RECURRENT_EARLY_REWARD_THRESHOLD",
    "RECURRENT_ENTRY_WINDOW",
    "RESERVED_CONFIRMATION_CANDIDATES",
    "RESERVED_CONFIRMATION_CONTROL",
    "ReservedConfirmationCandidate",
    "RETENTION_RATIO_THRESHOLD",
    "RETIREMENT_CONFIRMATION_WINDOW",
    "TUNING_SELECTION_RULE",
    "audit_hidden_partner_lifecycle_v5",
    "critical_run_primitives",
    "critical_run_primitives_v5",
    "run_evidence_lease_tuning_grid",
    "require_lease_tuning_execution_allowed",
    "summarize_critical_lifecycle_v2",
]
