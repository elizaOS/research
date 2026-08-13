"""Consumed-result locks for Stage A H=2 latent-expert quarantine."""

from __future__ import annotations

import copy
import json
from typing import Any, cast

import pytest

from alberta_framework.evaluation.two_event_latent_context_expert_recurrence_development import (
    ASSESSMENT_STATUS,
    LIMITATIONS,
    TWO_EVENT_LATENT_CONTEXT_EXPERT_RECURRENCE_REPORT_SCHEMA,
    run_two_event_latent_context_expert_recurrence_development,
    two_event_latent_context_expert_recurrence_report_json,
    validate_two_event_latent_context_expert_recurrence_report,
)

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def report() -> dict[str, Any]:
    return cast(
        dict[str, Any],
        run_two_event_latent_context_expert_recurrence_development(),
    )


def _runs(report: dict[str, Any], engine: str = "jax_jit_scan") -> list[dict[str, Any]]:
    return cast(list[dict[str, Any]], report["executions"][engine])


def test_report_is_strict_in_memory_consumed_stage_a_and_nonpromoting(
    report: dict[str, Any],
) -> None:
    validation = validate_two_event_latent_context_expert_recurrence_report(report)

    assert validation.valid, validation.errors
    assert report["schema_version"] == (
        TWO_EVENT_LATENT_CONTEXT_EXPERT_RECURRENCE_REPORT_SCHEMA
    )
    assert report["development_only"] is True
    assert report["scientific_promotion_allowed"] is False
    assert report["output_writes_allowed"] is False
    assert report["assessment_status"] == ASSESSMENT_STATUS == "not_assessed"
    assert report["consumed_development_result"] is True
    assert report["stage"] == "A"
    assert report["stage_b_executed"] is False
    assert report["descriptive_only"] is True
    assert report["winner_or_default_selected"] is False
    assert report["limitations"] == list(LIMITATIONS)
    assert report["protocol"]["confirmation_horizon"] == 2
    assert report["protocol"]["margin_or_dwell_parameter_present"] is False
    assert report["protocol"]["seed_or_hyperparameter_search_performed"] is False
    assert len(report["report_sha256"]) == 64
    encoded = two_event_latent_context_expert_recurrence_report_json(report)
    assert json.loads(encoded) == report
    assert encoded == two_event_latent_context_expert_recurrence_report_json(report)


def test_matched_arms_have_recomputed_resources_and_fixed_work(
    report: dict[str, Any],
) -> None:
    comparison = report["arm_comparison"]
    assert comparison["initial_state_equal"] is True
    assert comparison["resources_equal"] is True
    assert comparison["fixed_work_equal"] is True
    assert comparison["only_config_differences"] == {
        "confirmation_routing_enabled": {
            "two_event_confirmation_routing_disabled": False,
            "two_event_confirmation_routing_enabled": True,
        }
    }
    assert comparison["winner_selected"] is False

    for engine in ("python_eager", "jax_jit_scan"):
        for run in _runs(report, engine):
            resources = run["resources"]
            work = run["work"]
            assert resources["initial_state"] == resources["final_state"]
            assert resources["logical_peak_state_nbytes"] == 53
            assert resources["logical_prediction_cache_nbytes"] == 70
            assert work["confirmation_horizon"] == 2
            assert work["logical_updates"] == 1536
            assert work["logical_pending_transition_evaluations"] == 1536
            assert work["logical_expert_predictions"] == 6144
            assert work["logical_expert_losses"] == 3072
            assert work["logical_candidate_gradients"] == 3072
            assert work["online_random_draws"] == 0
            assert run["checkpoints"]["A2_tail"]["step_words"] == [0, 1536]
            for event in run["trace"]:
                owner = event["pre_update_owner"]
                assert event["prediction"] == event["expert_predictions"][owner]
                assert event["current_error_relabelled_after_target"] is False
                assert event["parameter_subtree_commit_count"] == sum(
                    event["expert_update_mask"]
                )
                if event["quarantine_opened"]:
                    assert event["zero_parameter_commit"] is True
                    assert event["zero_commit_reason"] == "quarantine_opened"
                    assert event["pending_after"]["valid"] is True
                if event["ambiguous_challenger_abstention"]:
                    assert event["zero_parameter_commit"] is True
                    assert event["zero_commit_reason"] == (
                        "ambiguous_challenger_abstention"
                    )
                if event["quarantine_second_evidence"]:
                    assert event["pending_before"]["valid"] is True
                    assert event["pending_after"]["valid"] is False


def test_consumed_h_two_routing_counts_and_phase_metrics_are_exact(
    report: dict[str, Any],
) -> None:
    enabled, disabled = _runs(report)

    assert enabled["observed_routing_counts"] == {
        "ambiguous_challenger_abstentions": 0,
        "confirmations": 2,
        "parameter_subtree_commits": 1532,
        "quarantine_openings": 4,
        "rejections": 2,
        "second_evidence_events": 4,
        "zero_parameter_commits": 4,
    }
    assert disabled["observed_routing_counts"] == {
        "ambiguous_challenger_abstentions": 0,
        "confirmations": 25,
        "parameter_subtree_commits": 1496,
        "quarantine_openings": 40,
        "rejections": 15,
        "second_evidence_events": 40,
        "zero_parameter_commits": 40,
    }

    enabled_phase = enabled["metrics"]["phase"]
    disabled_phase = disabled["metrics"]["phase"]
    assert {
        name: enabled_phase[name]["prequential_mse"]
        for name in ("A1", "B", "A2")
    } == {
        "A1": 0.015666176583638062,
        "B": 0.01452195047199631,
        "A2": 0.017470371040697825,
    }
    assert {
        name: disabled_phase[name]["prequential_mse"]
        for name in ("A1", "B", "A2")
    } == {
        "A1": 0.015666176583638062,
        "B": 0.0994502382110847,
        "A2": 0.07548175956553796,
    }
    assert {
        name: enabled_phase[name]["early_prequential_mse"]
        for name in ("A1", "B", "A2")
    } == {
        "A1": 0.12532941205202097,
        "B": 0.11617250464878454,
        "A2": 0.1397629683255826,
    }
    assert {
        name: disabled_phase[name]["early_prequential_mse"]
        for name in ("A1", "B", "A2")
    } == {
        "A1": 0.12532941205202097,
        "B": 0.7955090700830885,
        "A2": 0.5980502745331571,
    }


def test_enabled_routing_preserves_a_bit_exact_through_b_on_consumed_root(
    report: dict[str, Any],
) -> None:
    enabled = report["consumed_findings"]["confirmation_routing_enabled"]
    disabled = report["consumed_findings"]["confirmation_routing_disabled"]
    enabled_probe = enabled["direct_a_memory_probe"]
    disabled_probe = disabled["direct_a_memory_probe"]

    assert enabled["learned_a_expert_identity"] == 0
    assert enabled_probe["a1_end_a_expert_mse"] == 4.743384504624082e-20
    assert enabled_probe["b_end_a_expert_mse"] == 4.743384504624082e-20
    assert enabled_probe["b_end_minus_a1_end_a_expert_mse"] == 0.0
    assert enabled_probe["subtree_bit_exact_across_b"] is True
    assert enabled_probe["selected_update_count_during_b"] == 0
    assert enabled_probe["b_phase_steps_updating_a_expert"] == {
        "count": 0,
        "first": None,
        "last": None,
        "phase_steps_if_at_most_16": [],
        "phase_steps_sha256": (
            "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945"
        ),
    }
    assert enabled["a2_reactivation"]["a1_owner_was_dormant_at_a2_entry"] is True
    assert enabled["a2_reactivation"][
        "observed_a2_outcomes_until_a1_owner_selected"
    ] == 2

    assert disabled_probe["a1_end_a_expert_mse"] == 4.743384504624082e-20
    assert disabled_probe["b_end_a_expert_mse"] == 4.037702599687327
    assert disabled_probe["b_end_minus_a1_end_a_expert_mse"] == 4.037702599687327
    assert disabled_probe["subtree_bit_exact_across_b"] is False
    assert disabled_probe["selected_update_count_during_b"] == 498
    assert disabled["a2_reactivation"]["a1_owner_was_dormant_at_a2_entry"] is False


def test_first_b_window_is_causal_open_then_second_evidence(
    report: dict[str, Any],
) -> None:
    enabled = report["consumed_findings"]["confirmation_routing_enabled"]
    disabled = report["consumed_findings"]["confirmation_routing_disabled"]
    enabled_first, enabled_second = enabled["first_b_two_event_window"]
    disabled_first, disabled_second = disabled["first_b_two_event_window"]

    assert enabled_first == disabled_first
    assert enabled_first["phase_step"] == 0
    assert enabled_first["prediction"] == -0.5878112316131592
    assert enabled_first["quarantine_opened"] is True
    assert enabled_first["parameter_subtree_commit_count"] == 0
    assert enabled_first["pending_after"] == {
        "birth_words": [0, 513],
        "candidate": 1,
        "owner": 0,
        "valid": True,
    }
    assert enabled_second["phase_step"] == disabled_second["phase_step"] == 1
    assert enabled_second["quarantine_second_evidence"] is True
    assert enabled_second["quarantine_confirmed"] is True
    assert enabled_second["prediction"] == disabled_second["prediction"]
    assert enabled_second["selected_next_expert"] == 1
    assert disabled_second["selected_next_expert"] == 0
    assert enabled_second["expert_update_mask"] == [False, True]
    assert disabled_second["expert_update_mask"] == [True, False]
    assert enabled_second["pending_after"]["valid"] is False
    assert disabled_second["pending_after"]["valid"] is False


def test_eager_compiled_parity_is_within_declared_technical_tolerance(
    report: dict[str, Any],
) -> None:
    parity = report["eager_compiled_parity"]["arms"]
    assert parity["two_event_confirmation_routing_enabled"] == {
        "checkpoint_state_max_abs_difference": 2.6837067013119054e-09,
        "declared_numeric_tolerance": 2e-06,
        "observed_max_abs_difference": 1.9073486328125e-06,
        "observed_routing_counts_exact": True,
        "resources_exact": True,
        "technical_tolerance_is_not_an_outcome_threshold": True,
        "trace_discrete_fields_exact": True,
        "trace_float_max_abs_difference": 1.9073486328125e-06,
        "within_declared_numeric_tolerance": True,
        "work_exact": True,
    }
    assert parity["two_event_confirmation_routing_disabled"] == {
        "checkpoint_state_max_abs_difference": 2.6837067013119054e-09,
        "declared_numeric_tolerance": 2e-06,
        "observed_max_abs_difference": 1.443084329366684e-06,
        "observed_routing_counts_exact": True,
        "resources_exact": True,
        "technical_tolerance_is_not_an_outcome_threshold": True,
        "trace_discrete_fields_exact": True,
        "trace_float_max_abs_difference": 1.443084329366684e-06,
        "within_declared_numeric_tolerance": True,
        "work_exact": True,
    }


def test_validator_rejects_tampering_without_a_writer(report: dict[str, Any]) -> None:
    tampered = copy.deepcopy(report)
    tampered["executions"]["jax_jit_scan"][0]["trace"][512][
        "quarantine_confirmed"
    ] = True
    validation = validate_two_event_latent_context_expert_recurrence_report(tampered)

    assert not validation.valid
    assert "report does not match" in "; ".join(validation.errors)
