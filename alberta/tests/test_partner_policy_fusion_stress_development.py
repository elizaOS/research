from __future__ import annotations

import copy
import dataclasses
import json
from typing import Any

import numpy as np
import pytest

from alberta_framework.evaluation import partner_policy_fusion_stress_development as stress

pytestmark = pytest.mark.unit


def _phase_mean(
    report: stress.PartnerPolicyFusionStressReport,
    condition: stress.Condition,
    phase: int,
) -> float:
    values = [
        record.net_utility
        for record in report.traces[condition]
        if record.event.phase == phase
    ]
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def test_protocol_is_frozen_development_only_and_nonpromoting() -> None:
    assert stress.CONFIG.num_events == 96
    assert stress.CONDITIONS == (
        "learned_fusion",
        "outcome_blinded_fusion",
        "base_only",
    )
    assert stress.ASSESSMENT == "not_assessed"
    assert stress.DEVELOPMENT_ONLY
    assert not stress.SCIENTIFIC_PROMOTION_ALLOWED
    assert not stress.OUTPUT_WRITES_ALLOWED
    assert stress.RNG_DRAWS_PER_EVENT == 0
    assert all("write" not in name for name in stress.__all__)


@pytest.mark.parametrize(
    "change",
    [
        {"phase_length": 11},
        {"num_phases": 7},
        {"reversal_phase": 3},
        {"declared_confidence": 0.8},
        {"partner_costs": (0.1, 0.1)},
    ],
)
def test_protocol_configuration_rejects_retuning(change: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="frozen"):
        stress.PartnerPolicyFusionStressConfig(**change)


def test_schedule_hides_reliability_shift_behind_repeated_observable_contexts() -> None:
    schedule = stress.build_stress_schedule()
    assert schedule == stress.SCHEDULE
    for context_id in (0, 1):
        before = next(
            event
            for event in schedule
            if event.context_id == context_id and not event.after_reversal
        )
        after = next(
            event
            for event in schedule
            if event.context_id == context_id and event.after_reversal
        )
        assert before.context_features == after.context_features
        assert before.reliable_partner != after.reliable_partner
        assert before.correct_action != after.correct_action
    assert all(event.safety_action_mask[0] for event in schedule)


@pytest.fixture(scope="module")
def report() -> stress.PartnerPolicyFusionStressReport:
    return stress.run_partner_policy_fusion_stress_development()


def test_all_arms_share_exact_events_shapes_state_budget_and_call_counts(
    report: stress.PartnerPolicyFusionStressReport,
) -> None:
    expected_events = tuple(event.to_dict() for event in stress.SCHEDULE)
    budgets: set[int] = set()
    for condition in stress.CONDITIONS:
        trace = report.traces[condition]
        summary = report.summaries[condition]
        assert tuple(record.event.to_dict() for record in trace) == expected_events
        assert summary.event_count == stress.CONFIG.num_events
        assert summary.decision_calls == stress.CONFIG.num_events
        assert summary.feedback_calls == stress.CONFIG.num_events
        assert summary.fixed_message_slots_per_call == stress.CONFIG.max_partners
        budgets.add(summary.persistent_state_bytes)
    assert len(budgets) == 1


def test_hard_mask_is_never_overridden_and_failures_fall_back_to_base(
    report: stress.PartnerPolicyFusionStressReport,
) -> None:
    for condition in stress.CONDITIONS:
        for record in report.traces[condition]:
            assert record.event.safety_action_mask[record.effective_action]
            if not any(record.event.message_available):
                assert record.effective_action == 0
                assert not record.partner_influenced
            for partner, valid in enumerate(record.valid_messages):
                proposed_action = partner + 1
                if not record.event.safety_action_mask[proposed_action]:
                    assert not valid


def test_outcome_blinding_runs_feedback_but_cannot_learn_partner_value(
    report: stress.PartnerPolicyFusionStressReport,
) -> None:
    learned = report.summaries[stress.LEARNED_FUSION]
    blinded = report.summaries[stress.OUTCOME_BLINDED_FUSION]
    assert learned.feedback_applied_count > 0
    assert blinded.feedback_applied_count > 0
    assert any(
        abs(value) > 0.0
        for row in learned.final_reliability_weights
        for value in row
    )
    assert all(
        value == 0.0
        for row in blinded.final_reliability_weights
        for value in row
    )
    assert all(
        record.feedback_target in {0.0, 0.5}
        for record in report.traces[stress.OUTCOME_BLINDED_FUSION]
    )


def test_reversal_and_recovery_are_raw_descriptions_not_a_gate(
    report: stress.PartnerPolicyFusionStressReport,
) -> None:
    assert _phase_mean(report, stress.LEARNED_FUSION, 4) < _phase_mean(
        report, stress.LEARNED_FUSION, 3
    )
    assert _phase_mean(report, stress.LEARNED_FUSION, 7) > _phase_mean(
        report, stress.LEARNED_FUSION, 4
    )
    assert report.assessment == "not_assessed"
    assert not report.scientific_promotion_allowed


def test_descriptive_summaries_reconstruct_from_primitive_trace(
    report: stress.PartnerPolicyFusionStressReport,
) -> None:
    for condition in stress.CONDITIONS:
        trace = report.traces[condition]
        summary = report.summaries[condition]
        assert summary.mean_task_reward == pytest.approx(
            np.mean([record.task_reward for record in trace])
        )
        assert summary.mean_net_utility == pytest.approx(
            np.mean([record.net_utility for record in trace])
        )
        assert summary.partner_influenced_count == sum(
            record.partner_influenced for record in trace
        )
        assert summary.feedback_applied_count == sum(
            record.feedback_applied for record in trace
        )
        assert summary.total_communication_failure_count == 8


def test_report_validator_rejects_digest_tamper_and_resealed_causal_tamper(
    report: stress.PartnerPolicyFusionStressReport,
) -> None:
    changed = dataclasses.replace(report, assessment="accepted")
    changed_errors = stress.validate_partner_policy_fusion_stress_report(changed)
    assert "assessment must remain not_assessed" in changed_errors

    summaries = dict(report.summaries)
    summaries[stress.LEARNED_FUSION] = dataclasses.replace(
        summaries[stress.LEARNED_FUSION], mean_net_utility=123.0
    )
    provisional = dataclasses.replace(report, summaries=summaries)
    resealed = dataclasses.replace(
        provisional,
        deterministic_payload_digest=stress._digest(
            provisional.payload(include_digest=False)
        ),
    )
    resealed_errors = stress.validate_partner_policy_fusion_stress_report(resealed)
    assert "report differs from exact causal replay" in resealed_errors


def test_report_payload_is_finite_canonical_json(
    report: stress.PartnerPolicyFusionStressReport,
) -> None:
    encoded = json.dumps(
        report.payload(), allow_nan=False, separators=(",", ":"), sort_keys=True
    )
    assert report.deterministic_payload_digest in encoded
    assert "not_assessed" in encoded


def test_checkpoint_creation_rejects_boolean_and_out_of_range_splits() -> None:
    for value in (-1, stress.CONFIG.num_events + 1, True, 1.0):
        with pytest.raises(ValueError, match="strict integer"):
            stress.make_partner_policy_fusion_stress_checkpoint(value)  # type: ignore[arg-type]


def test_forged_checkpoint_prefix_is_rejected_even_after_outer_reseal() -> None:
    checkpoint = stress.make_partner_policy_fusion_stress_checkpoint(3)
    forged = copy.deepcopy(checkpoint)
    conditions = forged["conditions"]
    assert isinstance(conditions, dict)
    learned = conditions[stress.LEARNED_FUSION]
    assert isinstance(learned, dict)
    trace = learned["trace_prefix"]
    assert isinstance(trace, list)
    first = trace[0]
    assert isinstance(first, dict)
    first["net_utility"] = 999.0
    unsigned = dict(forged)
    unsigned.pop("checkpoint_digest")
    forged["checkpoint_digest"] = stress._digest(unsigned)
    with pytest.raises(ValueError, match="causal prefix replay"):
        stress.resume_partner_policy_fusion_stress_checkpoint(forged)


def test_checkpoint_json_roundtrip_preserves_exact_resume(
    report: stress.PartnerPolicyFusionStressReport,
) -> None:
    checkpoint = stress.make_partner_policy_fusion_stress_checkpoint(48)
    transported = json.loads(json.dumps(checkpoint, allow_nan=False))
    resumed = stress.resume_partner_policy_fusion_stress_checkpoint(transported)
    assert resumed.payload() == report.payload()
