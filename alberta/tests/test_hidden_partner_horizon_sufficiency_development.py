"""Slow development regression for the fixed observation-horizon check."""

from __future__ import annotations

import functools

import pytest

from alberta_framework.evaluation.hidden_partner_horizon_sufficiency_development import (
    C_ABSENCE_GAP_STEPS,
    CYCLE_STEPS,
    HORIZON_LIFECYCLE_GATE_CONTRACT,
    HORIZON_SUFFICIENCY_NAMESPACE,
    HORIZON_SUFFICIENCY_RESOURCE_CONTRACT,
    HORIZON_SUFFICIENCY_SEED,
    OUTPUT_WRITES_ALLOWED,
    RETENTION_GRACE_STEPS,
    SCIENTIFIC_PROMOTION_ALLOWED,
    SEGMENT_LENGTHS,
    HorizonSufficiencyDevelopmentResult,
    run_horizon_sufficiency_development,
    validate_horizon_sufficiency_static_contract,
)

pytestmark = [pytest.mark.development, pytest.mark.slow]


@functools.lru_cache(maxsize=1)
def _result() -> HorizonSufficiencyDevelopmentResult:
    return run_horizon_sufficiency_development()


def test_horizon_sufficiency_lane_is_fixed_shape_and_nonpromoting() -> None:
    assert validate_horizon_sufficiency_static_contract() == ()
    assert HORIZON_SUFFICIENCY_NAMESPACE == (
        "hidden-partner-v0-dev-target-only-horizon-sufficiency-v1"
    )
    assert HORIZON_SUFFICIENCY_SEED.to_dict() == {
        "namespace": HORIZON_SUFFICIENCY_NAMESPACE,
        "index": 0,
        "stream_seed": 2_097_892_768,
        "initialization_seed": 3_606_366_503,
    }
    assert SEGMENT_LENGTHS == (512,) * 9
    assert CYCLE_STEPS == 4_608
    assert C_ABSENCE_GAP_STEPS == 1_024
    assert RETENTION_GRACE_STEPS == 1_280
    assert HORIZON_LIFECYCLE_GATE_CONTRACT == {
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
    assert HORIZON_SUFFICIENCY_RESOURCE_CONTRACT["total_state_nbytes"] == 6_833
    assert not SCIENTIFIC_PROMOTION_ALLOWED
    assert not OUTPUT_WRITES_ALLOWED


def test_fixed_horizon_run_records_valid_unchanged_gate_pass() -> None:
    result = _result()
    lifecycle = result.lifecycle
    assert result.validation.valid, result.to_report()
    assert result.validation.config_contract_valid
    assert result.validation.resource_contract_valid
    assert result.validation.trace_contract_valid
    assert result.validation.lifecycle_contract_valid
    assert result.development_only
    assert not result.scientific_promotion_allowed
    assert not result.output_writes_allowed

    assert result.status == "passed_horizon_sufficiency_check", result.to_report()
    assert result.lifecycle_requirement_failures == (), result.to_report()
    assert result.c_horizon_requirement_failures == (), result.to_report()
    assert result.c_post_acquisition_observation_steps == 383, result.to_report()

    assert lifecycle.c_promotion_event_steps == (2_687,), result.to_report()
    assert lifecycle.c_acquisition_step == 2_689, result.to_report()
    assert lifecycle.c_first_late_reward == pytest.approx(0.8671875)
    assert lifecycle.c_first_late_intended_accuracy == pytest.approx(1.0)
    assert lifecycle.c_critical_column_learning_nll_gain == pytest.approx(
        0.6834743965472907
    )
    assert lifecycle.c_critical_column_learning_positive_fraction == pytest.approx(
        1.0
    )
    assert lifecycle.c_task_learned
    assert lifecycle.c_survival_gap_steps == 0
    assert lifecycle.c_continuously_survived
    assert lifecycle.c_recurrent_early_reward == pytest.approx(0.828125)
    assert lifecycle.c_retained_and_used

    assert lifecycle.d_acquisition_step == 1_601, result.to_report()
    assert lifecycle.d_task_learned
    assert lifecycle.d_retirement_event_steps == (3_391,), result.to_report()
    assert lifecycle.d_retirement_event_aligned
    assert lifecycle.d_linked_matching_candidate_reset_count == 1
    assert lifecycle.d_linked_candidate_utility_post == 0.0
    assert lifecycle.d_linked_candidate_head_linf_post == 0.0
    assert lifecycle.d_linked_candidate_age_post == 0
    assert lifecycle.d_repromotions_after_retirement == 0
    assert lifecycle.d_absent_entire_final_window
    assert lifecycle.d_learned_then_stably_retired
    assert lifecycle.joint_memory_management_success
