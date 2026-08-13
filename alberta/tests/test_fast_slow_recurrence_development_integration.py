"""Consumed-result integration checks for the FastSlow recurrence diagnostic."""

from __future__ import annotations

import copy
import json
from typing import Any, cast

import pytest

from alberta_framework.evaluation.fast_slow_recurrence_development import (
    ASSESSMENT_STATUS,
    FAST_SLOW_RECURRENCE_REPORT_SCHEMA,
    LIMITATIONS,
    PARITY_FLOAT_MAX_ABS_TOLERANCE,
    fast_slow_recurrence_report_json,
    run_fast_slow_recurrence_development,
    validate_fast_slow_recurrence_report,
)

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def report() -> dict[str, Any]:
    return cast(dict[str, Any], run_fast_slow_recurrence_development())


def _runs(report: dict[str, Any], engine: str = "jax_jit_scan") -> list[dict[str, Any]]:
    return cast(list[dict[str, Any]], report["executions"][engine])


def test_consumed_report_is_strict_in_memory_and_nonpromoting(
    report: dict[str, Any],
) -> None:
    validation = validate_fast_slow_recurrence_report(report)

    assert validation.valid, validation.errors
    assert report["schema_version"] == FAST_SLOW_RECURRENCE_REPORT_SCHEMA
    assert report["development_only"] is True
    assert report["scientific_promotion_allowed"] is False
    assert report["output_writes_allowed"] is False
    assert report["assessment_status"] == ASSESSMENT_STATUS == "not_assessed"
    assert report["consumed_development_result"] is True
    assert report["limitations"] == list(LIMITATIONS)
    assert len(report["protocol_sha256"]) == 64
    assert len(report["source_manifest_sha256"]) == 64
    assert len(report["causal_reconstruction_sha256"]) == 64
    assert len(report["report_sha256"]) == 64
    encoded = fast_slow_recurrence_report_json(report)
    assert json.loads(encoded) == report
    assert encoded == fast_slow_recurrence_report_json(report)


def test_configs_resources_work_and_rng_are_exactly_matched(
    report: dict[str, Any],
) -> None:
    comparison = report["arm_comparison"]
    assert comparison["only_config_differences"] == {
        "fast_step_size": {
            "ordinary_fast_slow": 0.05,
            "slow_only_matched_state": 0.0,
        },
        "gate_step_size": {
            "ordinary_fast_slow": 0.01,
            "slow_only_matched_state": 0.0,
        },
    }
    assert comparison["winner_selected"] is False
    assert comparison["default_selected"] is False

    for engine in ("python_eager", "jax_jit_scan"):
        runs = _runs(report, engine)
        assert runs[0]["resources"] == runs[1]["resources"]
        assert runs[0]["work"] == runs[1]["work"]
        assert runs[0]["rng"] == runs[1]["rng"]
        for run in runs:
            assert run["resources"]["initial_state"] == run["resources"]["final_state"]
            assert run["resources"]["logical_peak_state_nbytes"] == 1304
            assert run["resources"]["initial_state"]["step_count_dtype"] == "int32"
            assert run["work"] == {
                "logical_updates": 1536,
                "logical_online_predict_before_update_examples": 1536,
                "logical_probe_prediction_examples": 2560,
                "logical_explicit_prediction_examples": 4096,
                "logical_forward_examples_inside_updates": 1536,
                "logical_total_forward_examples": 5632,
                "update_random_draws": 0,
                "prediction_random_draws": 0,
            }
            assert run["rng"] == {
                "initialization_owned_by": "FastSlowLearner.init",
                "common_initialization_key_across_arms": True,
                "encoder_float32_values_drawn": 64,
                "gate_float32_values_drawn": 64,
                "readout_initialization_random_draws": 0,
                "online_random_draws": 0,
            }


def test_consumed_full_arm_records_slow_retention_failure_without_relabeling(
    report: dict[str, Any],
) -> None:
    ordinary, slow_only = _runs(report)
    audit = ordinary["metrics"]["decisive_slow_component_audit"]

    assert audit["a1_end_slow_component_a_mse"] == 0.018893112003101138
    assert audit["b_end_slow_component_a_mse"] == 4.0191857787912575
    assert audit["b_end_minus_a1_end_slow_component_a_mse"] == 4.000292666788156
    assert audit["b_end_over_a1_end_slow_component_a_mse"] == 212.7328614857914
    assert audit["a2_entry_slow_component_a_mse"] == 4.0191857787912575
    assert audit["a2_tail_slow_component_a_mse"] == 0.026916519126731542
    assert audit["a2_tail_relearning_reduction"] == 3.9922692596645257
    assert audit["a2_tail_relearning_counted_as_retention"] is False
    assert audit["threshold_or_verdict_applied"] is False

    finding = report["consumed_findings"]["ordinary_full_arm"]
    assert finding["slow_component_retention_through_b_demonstrated"] is False
    assert finding["a2_relearning_used_as_retention_evidence"] is False
    assert finding["outcome_threshold_applied"] is False
    assert "did not preserve" in report["consumed_findings"]["negative_finding"]

    for run in (ordinary, slow_only):
        checkpoints = run["checkpoints"]
        assert checkpoints["B_end"]["state_sha256"] == checkpoints["A2_entry"][
            "state_sha256"
        ]
        assert checkpoints["B_end"]["a_probe"] == checkpoints["A2_entry"][
            "a_probe"
        ]
        assert checkpoints["A2_entry"]["parameter_drift_from_previous_checkpoint"] == {
            "all": 0.0,
            "encoder": 0.0,
            "fast": 0.0,
            "gate": 0.0,
            "slow": 0.0,
        }
    assert slow_only["checkpoints"]["A2_tail"]["parameter_norms"]["fast"] == 0.0
    assert slow_only["checkpoints"]["A2_tail"]["parameter_drift_from_initial"][
        "gate"
    ] == 0.0


def test_prequential_windows_switches_and_decomposition_are_exact(
    report: dict[str, Any],
) -> None:
    ordinary, slow_only = _runs(report)
    ordinary_phase = ordinary["metrics"]["phase"]
    slow_phase = slow_only["metrics"]["phase"]

    assert {
        phase: (
            ordinary_phase[phase]["early"]["prequential_mse"],
            ordinary_phase[phase]["tail"]["prequential_mse"],
            ordinary_phase[phase]["mean"]["prequential_mse"],
        )
        for phase in ("A1", "B", "A2")
    } == {
        "A1": (0.14879887298457106, 0.024695930265494326, 0.05142238980767949),
        "B": (0.1852027628878261, 0.03844795155278913, 0.11954193647870887),
        "A2": (0.31054127362535944, 0.03547665175223358, 0.09669453453763459),
    }
    assert {
        phase: (
            slow_phase[phase]["early"]["prequential_mse"],
            slow_phase[phase]["tail"]["prequential_mse"],
            slow_phase[phase]["mean"]["prequential_mse"],
        )
        for phase in ("A1", "B", "A2")
    } == {
        "A1": (0.16661773500653965, 0.017074317550950013, 0.048013824587115436),
        "B": (0.24202190769051413, 0.03321684995735197, 0.12173865538928075),
        "A2": (0.3609846546610811, 0.028071144652287217, 0.10035130029928022),
    }
    assert ordinary["metrics"]["switch_adaptation"] == {
        "b_entry_minus_a1_tail_mse": 0.16050683262233179,
        "b_tail_minus_b_entry_mse": -0.14675481133503698,
        "a2_entry_minus_b_tail_mse": 0.2720933220725703,
        "a2_tail_minus_a2_entry_mse": -0.27506462187312586,
    }
    assert ordinary["metrics"]["recurrence"] == {
        "a2_entry_minus_a1_tail_mse": 0.2858453433598651,
        "a2_tail_minus_a1_tail_mse": 0.010780721486739253,
        "a2_within_phase_relearning_mse": 0.27506462187312586,
        "relearning_is_not_retention": True,
    }
    assert max(
        window["decomposition_max_abs_error"]
        for phase in ordinary_phase.values()
        for name, window in phase.items()
        if name in {"early", "tail", "mean"}
    ) <= 1.1920928955078125e-07


def test_eager_compiled_parity_is_bounded_and_state_exact(
    report: dict[str, Any],
) -> None:
    parity = report["eager_compiled_parity"]
    assert parity["technical_float_tolerance_not_an_outcome_threshold"] is True
    ordinary = parity["arms"]["ordinary_fast_slow"]
    slow_only = parity["arms"]["slow_only_matched_state"]

    assert ordinary["checkpoint_state_max_abs_difference"] == 0.0
    assert slow_only["checkpoint_state_max_abs_difference"] == 0.0
    assert ordinary["observed_max_abs_difference"] == 2.38008617259311e-07
    assert slow_only["observed_max_abs_difference"] == 4.799440915803643e-07
    for arm in (ordinary, slow_only):
        assert arm["trace_discrete_fields_exact"] is True
        assert arm["config_exact"] is True
        assert arm["resources_exact"] is True
        assert arm["work_exact"] is True
        assert arm["declared_numeric_tolerance"] == PARITY_FLOAT_MAX_ABS_TOLERANCE
        assert arm["within_declared_numeric_tolerance"] is True
        assert arm["full_state_digest_equality_claimed"] is False


def test_validator_rejects_metric_and_hash_tampering(report: dict[str, Any]) -> None:
    corrupted = copy.deepcopy(report)
    corrupted["consumed_findings"]["ordinary_full_arm"][
        "slow_component_retention_through_b_demonstrated"
    ] = True

    validation = validate_fast_slow_recurrence_report(corrupted)
    assert not validation.valid
    assert "causal reconstruction" in validation.errors[0]
    with pytest.raises(ValueError, match="invalid FastSlow recurrence report"):
        fast_slow_recurrence_report_json(corrupted)
