"""Consumed-result checks for latent-context expert recurrence."""

from __future__ import annotations

import copy
import json
from typing import Any, cast

import pytest

from alberta_framework.evaluation.latent_context_expert_recurrence_development import (
    ASSESSMENT_STATUS,
    LATENT_CONTEXT_EXPERT_RECURRENCE_REPORT_SCHEMA,
    LIMITATIONS,
    latent_context_expert_recurrence_report_json,
    run_latent_context_expert_recurrence_development,
    validate_latent_context_expert_recurrence_report,
)

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def report() -> dict[str, Any]:
    return cast(dict[str, Any], run_latent_context_expert_recurrence_development())


def _runs(report: dict[str, Any], engine: str = "jax_jit_scan") -> list[dict[str, Any]]:
    return cast(list[dict[str, Any]], report["executions"][engine])


def test_report_is_strict_in_memory_consumed_and_nonpromoting(report: dict[str, Any]) -> None:
    validation = validate_latent_context_expert_recurrence_report(report)

    assert validation.valid, validation.errors
    assert report["schema_version"] == LATENT_CONTEXT_EXPERT_RECURRENCE_REPORT_SCHEMA
    assert report["development_only"] is True
    assert report["scientific_promotion_allowed"] is False
    assert report["output_writes_allowed"] is False
    assert report["assessment_status"] == ASSESSMENT_STATUS == "not_assessed"
    assert report["consumed_development_result"] is True
    assert report["limitations"] == list(LIMITATIONS)
    assert report["design_record"]["conceptual_novelty_claimed"] is False
    assert report["protocol"]["seed_or_hyperparameter_search_performed"] is False
    for field in (
        "protocol_sha256",
        "source_manifest_sha256",
        "causal_reconstruction_sha256",
        "report_sha256",
    ):
        assert len(report[field]) == 64
    encoded = latent_context_expert_recurrence_report_json(report)
    assert json.loads(encoded) == report
    assert encoded == latent_context_expert_recurrence_report_json(report)


def test_matched_arms_have_one_commit_and_causal_prequential_ownership(
    report: dict[str, Any],
) -> None:
    comparison = report["arm_comparison"]
    assert comparison["initial_state_equal"] is True
    assert comparison["resources_equal"] is True
    assert comparison["work_equal"] is True
    assert comparison["only_config_differences"] == {
        "selective_gating": {
            "latent_context_no_selective_gating": False,
            "latent_context_selective_gating": True,
        }
    }
    assert comparison["winner_selected"] is False

    for engine in ("python_eager", "jax_jit_scan"):
        for run in _runs(report, engine):
            assert run["resources"]["initial_state"] == run["resources"]["final_state"]
            assert run["resources"]["logical_peak_state_nbytes"] == 32
            assert run["work"]["logical_updates"] == 1536
            assert run["work"]["logical_expert_predictions"] == 6144
            assert run["work"]["logical_expert_losses"] == 3072
            assert run["work"]["logical_candidate_gradients"] == 3072
            assert run["work"]["logical_expert_subtree_commits"] == 1536
            assert run["work"]["online_random_draws"] == 0
            assert run["checkpoints"]["A2_tail"]["step_words"] == [0, 1536]
            for event in run["trace"]:
                owner = event["pre_update_owner"]
                assert event["prediction"] == event["expert_predictions"][owner]
                assert sum(event["expert_update_mask"]) == 1


def test_learned_identity_dormancy_reactivation_and_fragmentation_are_recorded(
    report: dict[str, Any],
) -> None:
    audit = report["consumed_findings"]["selective_gating"]
    ablation_audit = report["consumed_findings"]["no_selective_gating"]
    ordinary, ablation = _runs(report)
    learned_a = audit["learned_a_expert_identity"]

    assert learned_a == ordinary["checkpoints"]["A1_end"]["active_expert"]
    assert learned_a == 0
    assert audit["identity_was_not_hard_coded"] is True
    assert audit["direct_dormant_a_probe"] == {
        "a1_end_a_expert_mse": 3.7269449679189215e-20,
        "a1_end_subtree_sha256": (
            "f04285fcb6b68485ee65cacb6a14033f803e2f53481fd6014eb644a578175e96"
        ),
        "b_end_a_expert_mse": 1.9001388680843692e-05,
        "b_end_minus_a1_end_a_expert_mse": 1.9001388680843654e-05,
        "b_end_owner": 1,
        "b_end_subtree_sha256": (
            "22f5baa8a5f2c803d85d08292c54dd25626a4e10e5a6e1b598558f351e53457f"
        ),
        "contains_any_a2_update": False,
        "selected_update_count_during_b": 1,
        "subtree_bit_exact_across_b": False,
    }
    assert audit["first_b_outcome_routing"] == {
        "evidence_best_expert_after_target": 1,
        "pre_update_owner": 0,
        "preoutcome_context_identification_claimed": False,
        "prequential_prediction": -0.5878112316131592,
        "selected_different_expert_than_pre_update_owner": True,
        "selected_expert_received_current_outcome_update": True,
        "selected_next_expert_after_target": 1,
        "selection_and_training_are_post_outcome": True,
    }
    assert audit["a2_reactivation"] == {
        "a1_owner_reactivated_during_a2": True,
        "a1_owner_was_dormant_at_a2_entry": True,
        "a2_entry_state_equals_b_end": True,
        "a2_tail_a_expert_mse": 3.7269449679189215e-20,
        "a2_tail_subtree_sha256": (
            "c1c9cbe91a8e1e5831cbedf8451b84ae415a3c5ec23bdf8215026c30f728c54d"
        ),
        "counted_as_retention_through_b": False,
        "first_a2_pre_update_owner": 1,
        "first_a2_prediction": -0.3040594756603241,
        "first_a2_prediction_phase_step_using_reactivated_owner": 1,
        "first_a2_prediction_precedes_first_a2_outcome": True,
        "first_a2_selected_next_expert": 0,
        "first_a2_squared_error": 0.3698086589553249,
        "latency_is_descriptive_not_thresholded": True,
        "observed_a2_outcomes_until_a1_owner_selected": 1,
        "reactivation_term_applies": True,
    }
    assert audit["fragmentation_audit"] == {
        "a1_context_switch_count": 10,
        "a1_distinct_pre_update_owners": [0, 1],
        "b_context_switch_count": 3,
        "b_distinct_pre_update_owners": [0, 1],
        "b_distinct_selected_next_experts": [0, 1],
        "b_phase_steps_selecting_a1_owner_for_update": {
            "count": 1,
            "first": 2,
            "last": 2,
            "phase_steps_if_at_most_16": [2],
            "phase_steps_sha256": (
                "038966de9f6b9a901b20b4c6ca8b2a46009feebe031babc842d43690c0bc222b"
            ),
        },
        "clean_a_expert_dormancy_across_b": False,
        "fragmentation_threshold_or_verdict_applied": False,
        "per_transaction_nonselected_subtree_preservation_contract": True,
    }
    assert audit["performance_threshold_or_verdict_applied"] is False

    assert ablation_audit["direct_dormant_a_probe"] == {
        "a1_end_a_expert_mse": 4.743384504624082e-20,
        "a1_end_subtree_sha256": (
            "e7517d82138eb79cdae70f41fd5bcc88389eea5036fa71936d9a55f65d5dbcca"
        ),
        "b_end_a_expert_mse": 4.037702599687506,
        "b_end_minus_a1_end_a_expert_mse": 4.037702599687506,
        "b_end_owner": 0,
        "b_end_subtree_sha256": (
            "4346b1199ee6fda29d17d904aae96135ed78a49af4ef9a873a1fedff98aac7cb"
        ),
        "contains_any_a2_update": False,
        "selected_update_count_during_b": 512,
        "subtree_bit_exact_across_b": False,
    }
    assert ablation_audit["first_b_outcome_routing"] == {
        "evidence_best_expert_after_target": 1,
        "pre_update_owner": 0,
        "preoutcome_context_identification_claimed": False,
        "prequential_prediction": -0.5878112316131592,
        "selected_different_expert_than_pre_update_owner": False,
        "selected_expert_received_current_outcome_update": True,
        "selected_next_expert_after_target": 0,
        "selection_and_training_are_post_outcome": True,
    }
    assert ablation_audit["fragmentation_audit"][
        "b_phase_steps_selecting_a1_owner_for_update"
    ] == {
        "count": 512,
        "first": 0,
        "last": 511,
        "phase_steps_if_at_most_16": None,
        "phase_steps_sha256": (
            "61f3f6fd3aa109e05aa31cc4d74f333d17c8f73ced22284db2209e96a39884af"
        ),
    }
    assert ablation_audit["a2_reactivation"]["reactivation_term_applies"] is False
    assert ablation_audit["a2_reactivation"]["a1_owner_was_dormant_at_a2_entry"] is False
    assert ablation["checkpoints"]["B_end"] == ablation["checkpoints"]["A2_entry"] | {
        "label": "B_end"
    }
    assert ordinary["checkpoints"]["B_end"] == ordinary["checkpoints"]["A2_entry"] | {
        "label": "B_end"
    }


def test_consumed_prequential_phase_metrics_are_exact_but_not_a_verdict(
    report: dict[str, Any],
) -> None:
    ordinary, ablation = _runs(report)
    ordinary_phase = ordinary["metrics"]["phase"]
    ablation_phase = ablation["metrics"]["phase"]

    assert {
        name: (
            ordinary_phase[name]["prequential_mse"],
            ordinary_phase[name]["early_prequential_mse"],
            ordinary_phase[name]["tail_prequential_mse"],
            ordinary_phase[name]["context_switch_count"],
        )
        for name in ("A1", "B", "A2")
    } == {
        "A1": (0.02006674230304454, 0.16053392594143914, 0.0, 10),
        "B": (0.014963522183536051, 0.11970498050687987, 2.168404344971009e-19, 3),
        "A2": (0.0007224462825537357, 0.005779570126036035, 3.049318610115481e-20, 1),
    }
    assert {
        name: (
            ablation_phase[name]["prequential_mse"],
            ablation_phase[name]["early_prequential_mse"],
            ablation_phase[name]["tail_prequential_mse"],
            ablation_phase[name]["context_switch_count"],
        )
        for name in ("A1", "B", "A2")
    } == {
        "A1": (0.011575251259982683, 0.092602006238942, 0.0, 0),
        "B": (0.0436876796129365, 0.3494897454786507, 2.168404344971009e-19, 0),
        "A2": (0.04535282710706045, 0.3626772259381271, 3.049318610115481e-20, 0),
    }
    assert report["assessment_status"] == "not_assessed"
    assert report["arm_comparison"]["winner_selected"] is False


def test_eager_compiled_state_parity_and_validator_tamper_rejection(
    report: dict[str, Any],
) -> None:
    for arm in report["eager_compiled_parity"]["arms"].values():
        assert arm["trace_discrete_fields_exact"] is True
        assert arm["checkpoint_state_max_abs_difference"] == 0.0
        assert arm["resources_exact"] is True
        assert arm["work_exact"] is True
        assert arm["within_declared_numeric_tolerance"] is True
        assert arm["technical_tolerance_is_not_an_outcome_threshold"] is True

    corrupted = copy.deepcopy(report)
    corrupted["consumed_findings"]["selective_gating"][
        "performance_threshold_or_verdict_applied"
    ] = True
    validation = validate_latent_context_expert_recurrence_report(corrupted)
    assert not validation.valid
    assert "causal reconstruction" in validation.errors[0]
    with pytest.raises(ValueError, match="invalid latent-context expert report"):
        latent_context_expert_recurrence_report_json(corrupted)
