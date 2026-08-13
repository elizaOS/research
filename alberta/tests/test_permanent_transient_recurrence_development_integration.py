"""Consumed-result checks for the permanent/transient recurrence diagnostic."""

from __future__ import annotations

import copy
import json
from typing import Any, cast

import pytest

from alberta_framework.evaluation.permanent_transient_recurrence_development import (
    ASSESSMENT_STATUS,
    LIMITATIONS,
    PARITY_FLOAT_MAX_ABS_TOLERANCE,
    PERMANENT_TRANSIENT_RECURRENCE_REPORT_SCHEMA,
    permanent_transient_recurrence_report_json,
    run_permanent_transient_recurrence_development,
    validate_permanent_transient_recurrence_report,
)

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def report() -> dict[str, Any]:
    return cast(dict[str, Any], run_permanent_transient_recurrence_development())


def _runs(report: dict[str, Any], engine: str = "jax_jit_scan") -> list[dict[str, Any]]:
    return cast(list[dict[str, Any]], report["executions"][engine])


def test_report_is_deterministic_in_memory_and_strictly_nonpromoting(
    report: dict[str, Any],
) -> None:
    validation = validate_permanent_transient_recurrence_report(report)

    assert validation.valid, validation.errors
    assert report["schema_version"] == PERMANENT_TRANSIENT_RECURRENCE_REPORT_SCHEMA
    assert report["development_only"] is True
    assert report["scientific_promotion_allowed"] is False
    assert report["output_writes_allowed"] is False
    assert report["assessment_status"] == ASSESSMENT_STATUS == "not_assessed"
    assert report["consumed_development_result"] is True
    assert report["limitations"] == list(LIMITATIONS)
    assert report["design_record"]["source_faithful"] is False
    assert report["protocol"]["seed_or_hyperparameter_search_performed"] is False
    for field in (
        "protocol_sha256",
        "source_manifest_sha256",
        "causal_reconstruction_sha256",
        "report_sha256",
    ):
        assert len(report[field]) == 64
    encoded = permanent_transient_recurrence_report_json(report)
    assert json.loads(encoded) == report
    assert encoded == permanent_transient_recurrence_report_json(report)


def test_ablation_matches_state_work_and_freezes_only_permanent_subtree(
    report: dict[str, Any],
) -> None:
    comparison = report["arm_comparison"]
    assert comparison["ordinary_and_ablation_initial_state_sha256_equal"] is True
    assert comparison["ordinary_and_ablation_resources_equal"] is True
    assert comparison["ordinary_and_ablation_work_equal"] is True
    assert comparison["only_config_differences"] == {
        "permanent_encoder_step_size": {
            "alberta_pt_no_consolidation": 0.0,
            "alberta_pt_online_consolidation": 0.001,
        },
        "permanent_head_step_size": {
            "alberta_pt_no_consolidation": 0.0,
            "alberta_pt_online_consolidation": 0.01,
        },
    }
    assert comparison["winner_selected"] is False

    for engine in ("python_eager", "jax_jit_scan"):
        ordinary, ablation = _runs(report, engine)
        assert ordinary["resources"] == ablation["resources"]
        assert ordinary["work"] == ablation["work"]
        for run in (ordinary, ablation):
            assert run["resources"]["initial_state"] == run["resources"]["final_state"]
            assert run["resources"]["logical_peak_state_nbytes"] == 788
            assert run["work"]["logical_updates"] == 1536
            assert run["work"]["logical_gradient_evaluations"] == 3072
            assert run["work"]["replay_samples"] == 0
            assert run["work"]["online_random_draws"] == 0
            assert run["checkpoints"]["A2_tail"]["step_words"] == [0, 1536]
        permanent_hashes = {
            checkpoint["permanent_subtree_sha256"]
            for checkpoint in ablation["checkpoints"].values()
        }
        assert len(permanent_hashes) == 1
        assert ordinary["checkpoints"]["initial"]["permanent_subtree_sha256"] != ordinary[
            "checkpoints"
        ]["A1_end"]["permanent_subtree_sha256"]


def test_direct_a_probes_record_overwrite_without_counting_reacquisition(
    report: dict[str, Any],
) -> None:
    ordinary, ablation = _runs(report)
    finding = report["consumed_findings"]
    direct = finding["direct_pre_post_b_a_probe"]

    assert direct == {
        "a1_end_combined_a_mse": 0.034571646329729026,
        "a1_end_permanent_a_mse": 0.04346209138762533,
        "a2_entry_combined_a_mse": 4.631621755402419,
        "a2_entry_equals_b_end": True,
        "a2_entry_permanent_a_mse": 3.925718401691044,
        "b_end_combined_a_mse": 4.631621755402419,
        "b_end_minus_a1_end_combined_a_mse": 4.59705010907269,
        "b_end_minus_a1_end_permanent_a_mse": 3.8822563103034184,
        "b_end_over_a1_end_permanent_a_mse": 90.32511497614648,
        "b_end_permanent_a_mse": 3.925718401691044,
        "contains_any_a2_update": False,
        "retention_threshold_or_verdict_applied": False,
    }
    assert finding["a2_reacquisition"] == {
        "a2_entry_minus_a2_tail_combined_a_mse": 4.580307810035493,
        "a2_entry_minus_a2_tail_permanent_a_mse": 3.8753229831526608,
        "a2_tail_combined_a_mse": 0.0513139453669253,
        "a2_tail_permanent_a_mse": 0.05039541853838296,
        "counted_as_pre_b_retention": False,
    }
    assert ordinary["checkpoints"]["B_end"]["state_sha256"] == ordinary[
        "checkpoints"
    ]["A2_entry"]["state_sha256"]
    assert ordinary["checkpoints"]["B_end"]["a_probe"] == ordinary["checkpoints"][
        "A2_entry"
    ]["a_probe"]

    ablation_direct = report["no_consolidation_findings"]["direct_pre_post_b_a_probe"]
    assert ablation_direct["a1_end_permanent_a_mse"] == 1.0094256499246534
    assert ablation_direct["b_end_permanent_a_mse"] == 1.0094256499246534
    assert ablation_direct["a2_entry_permanent_a_mse"] == 1.0094256499246534
    assert ablation["checkpoints"]["A2_tail"]["a_probe"]["permanent_a_mse"] == (
        1.0094256499246534
    )


def test_prequential_readouts_and_permanent_path_intervention_are_explicit(
    report: dict[str, Any],
) -> None:
    ordinary = _runs(report)[0]
    phase = ordinary["metrics"]["phase"]

    assert {
        name: (
            phase[name]["combined_readout"]["prequential_mse"],
            phase[name]["permanent_only_readout"]["prequential_mse"],
        )
        for name in ("A1", "B", "A2")
    } == {
        "A1": (0.07749909220795403, 0.07808464757539782),
        "B": (0.13304984464196268, 0.14229154605228578),
        "A2": (0.10721928072130138, 0.11507264832216002),
    }
    intervention = report["consumed_findings"]["permanent_path_readout_ablation"]
    assert intervention["intervention"] == (
        "remove the additive permanent prediction at readout only"
    )
    assert intervention["counterfactual_retraining_claimed"] is False
    assert intervention["success_threshold_or_verdict_applied"] is False
    assert intervention["A1_end"] == {
        "combined_a_mse": 0.034571646329729026,
        "transient_only_a_mse": 1.009415144318261,
        "transient_only_minus_combined_a_mse": 0.9748434979885321,
    }
    assert intervention["B_end"] == {
        "combined_a_mse": 4.631621755402419,
        "transient_only_a_mse": 1.385489355382096,
        "transient_only_minus_combined_a_mse": -3.246132400020323,
    }


def test_cross_family_boundary_and_eager_compiled_parity_are_honest(
    report: dict[str, Any],
) -> None:
    boundary = report["fast_slow_sibling_comparison_boundary"]
    assert boundary == {
        "cross_family_performance_winner_allowed": False,
        "fast_slow_gradient_evaluations_per_update": 1,
        "fast_slow_state_nbytes": 1304,
        "permanent_transient_gradient_evaluations_per_update": 2,
        "permanent_transient_state_nbytes": 788,
        "same_consumed_source_manifest": True,
        "same_input_and_output_dims": True,
        "same_total_hidden_feature_count": True,
        "shape_or_gradient_work_matched": False,
    }
    for arm in report["eager_compiled_parity"]["arms"].values():
        assert arm["trace_discrete_fields_exact"] is True
        assert arm["checkpoint_state_max_abs_difference"] == 0.0
        assert arm["resources_exact"] is True
        assert arm["work_exact"] is True
        assert arm["declared_numeric_tolerance"] == PARITY_FLOAT_MAX_ABS_TOLERANCE
        assert arm["within_declared_numeric_tolerance"] is True
        assert arm["technical_tolerance_is_not_an_outcome_threshold"] is True


def test_validator_rejects_interpretation_and_hash_tampering(report: dict[str, Any]) -> None:
    corrupted = copy.deepcopy(report)
    corrupted["consumed_findings"]["direct_pre_post_b_a_probe"][
        "retention_threshold_or_verdict_applied"
    ] = True

    validation = validate_permanent_transient_recurrence_report(corrupted)
    assert not validation.valid
    assert "causal reconstruction" in validation.errors[0]
    with pytest.raises(ValueError, match="invalid permanent/transient report"):
        permanent_transient_recurrence_report_json(corrupted)
