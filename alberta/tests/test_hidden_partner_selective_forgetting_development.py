"""Slow development regression for target-only selective forgetting."""

from __future__ import annotations

import functools

import pytest

from alberta_framework.core.interaction_features import (
    RELEVANCE_PROBE_MODE_TARGET_ONLY_V1,
)
from alberta_framework.evaluation.hidden_partner_selective_forgetting_development import (
    MICROCYCLE_GRACE_STEPS,
    MICROCYCLE_SEGMENT_LENGTHS,
    MICROCYCLE_STEPS,
    SELECTIVE_FORGETTING_ARM_ORDER,
    SELECTIVE_FORGETTING_DEVELOPMENT_ARMS,
    SELECTIVE_FORGETTING_DEVELOPMENT_SEED,
    SELECTIVE_FORGETTING_RESOURCE_CONTRACT,
    SelectiveForgettingPanelResult,
    run_selective_forgetting_development_microcycle,
    validate_selective_forgetting_static_contract,
)

pytestmark = [pytest.mark.development, pytest.mark.slow]


@functools.lru_cache(maxsize=1)
def _panel() -> SelectiveForgettingPanelResult:
    return run_selective_forgetting_development_microcycle()


def test_selective_forgetting_successor_is_fixed_shape_and_nonpromoting() -> None:
    assert validate_selective_forgetting_static_contract() == ()
    assert MICROCYCLE_SEGMENT_LENGTHS == (256,) * 9
    assert MICROCYCLE_STEPS == 2_304
    assert MICROCYCLE_GRACE_STEPS == 640
    assert SELECTIVE_FORGETTING_DEVELOPMENT_SEED.namespace.endswith(
        "target-only-selective-forgetting-microcycle-v1"
    )
    assert tuple(arm.name for arm in SELECTIVE_FORGETTING_DEVELOPMENT_ARMS) == (
        SELECTIVE_FORGETTING_ARM_ORDER
    )
    primary, retirement_disabled, reacquisition_one = SELECTIVE_FORGETTING_DEVELOPMENT_ARMS
    assert primary.config.relevance_probe_mode == RELEVANCE_PROBE_MODE_TARGET_ONLY_V1
    assert primary.config.retire_stale_features
    assert primary.config.candidate_reacquisition_confirmation_steps == 8
    assert not retirement_disabled.config.retire_stale_features
    assert retirement_disabled.config.candidate_reacquisition_confirmation_steps == 8
    assert reacquisition_one.config.retire_stale_features
    assert reacquisition_one.config.candidate_reacquisition_confirmation_steps == 1
    assert SELECTIVE_FORGETTING_RESOURCE_CONTRACT["total_state_nbytes"] == 6_833


def test_target_only_microcycle_records_unchanged_gate_rejection() -> None:
    panel = _panel()
    assert panel.development_only
    assert not panel.scientific_promotion_allowed
    assert not panel.output_writes_allowed
    assert tuple(result.arm.name for result in panel.arms) == SELECTIVE_FORGETTING_ARM_ORDER
    assert all(result.validation.valid for result in panel.arms), panel.to_report()

    primary = panel.arm_result("selective_lease")
    lifecycle = primary.lifecycle
    assert lifecycle.c_continuously_survived, panel.to_report()
    assert lifecycle.c_survival_gap_steps == 0, panel.to_report()
    assert lifecycle.d_task_learned, panel.to_report()
    assert lifecycle.d_repromotions_after_retirement == 0, panel.to_report()
    assert lifecycle.d_retirement_event_steps == (2_239,), panel.to_report()
    assert panel.primary_requirement_failures == (
        "c_task_learned",
        "c_retained_and_used",
        "d_retirement_event_aligned",
        "d_linked_matching_candidate_reset_count_is_one",
        "d_linked_candidate_utility_is_positive_zero",
        "d_linked_candidate_head_is_positive_zero",
        "d_linked_candidate_age_is_zero",
        "d_absent_entire_final_window",
        "d_learned_then_stably_retired",
        "joint_memory_management_success",
    ), panel.to_report()
    assert panel.status == "valid_development_rejection", panel.to_report()

    # The matched arms are descriptive causal diagnostics. Their empirical
    # outcomes are deliberately not converted into required directionality.
    assert panel.arm_result("retirement_disabled").validation.valid
    assert panel.arm_result("reacquisition_one").validation.valid
