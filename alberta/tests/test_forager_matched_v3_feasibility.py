"""Tests for the non-authorizing matched-v3 feasibility receipt."""

from __future__ import annotations

import hashlib
import json
import math

import pytest

from alberta_framework.benchmarks import forager_matched_v3_feasibility as feasibility
from alberta_framework.benchmarks import forager_matched_v3_statistics as statistics


@pytest.mark.unit
def test_current_panel_range_arithmetic_and_interaction_ceiling_are_exact() -> None:
    variance, range_term, total = feasibility.finite_sample_penalty(
        candidate_count=11,
        alpha=0.05,
        block_count=30,
        variance_upper_bound=0,
    )
    assert variance == 0.0
    assert range_term.hex() == "0x1.3f1bebee8343ep+24"
    assert total == range_term

    variance, range_term, total = feasibility.finite_sample_penalty(
        candidate_count=11,
        alpha=0.05,
        block_count=4_545,
        variance_upper_bound=0,
    )
    assert variance == 0.0
    assert range_term.hex() == "0x1.04ae3eee1be82p+17"
    assert total == range_term
    assert 11 * 4_545 * statistics.HORIZON == 24_983_101_440


@pytest.mark.unit
def test_worst_case_variance_receipt_recomputes_minimum_n_and_exact_resources() -> None:
    design = feasibility.FeasibilityDesign(
        candidate_count=11,
        alpha=0.05,
        practical_margin=0,
        target_gap=2_000_000,
        variance_upper_bound=statistics.CONTRAST_RANGE_WIDTH**2 // 2,
        variance_bound_source="bounded_sample_variance_maximum",
        calibration_artifact_sha256=None,
        compute_cap_interactions=100_000_000_000,
    )
    receipt = feasibility.build_feasibility_receipt(design)
    arithmetic = receipt["arithmetic"]

    minimum = arithmetic["minimum_statistical_block_count"]
    assert minimum == 2_585
    previous_penalty = feasibility.finite_sample_penalty(
        candidate_count=11,
        alpha=0.05,
        block_count=minimum - 1,
        variance_upper_bound=design.variance_upper_bound,
    )[2]
    selected_penalty = feasibility.finite_sample_penalty(
        candidate_count=11,
        alpha=0.05,
        block_count=minimum,
        variance_upper_bound=design.variance_upper_bound,
    )[2]
    assert design.target_gap <= design.practical_margin + previous_penalty
    assert design.target_gap > design.practical_margin + selected_penalty
    assert arithmetic[
        "required_confirmatory_inferential_interactions_before_retries"
    ] == 14_209_310_720
    assert arithmetic["q_hex"] == "0x1.0c75a2b99bf9cp+3"
    assert arithmetic["variance_correction_hex"] == "0x1.aee8ba3bbb700p+20"
    assert arithmetic["range_correction_hex"] == "0x1.ca6919dbc8748p+17"
    assert arithmetic["total_correction_hex"] == "0x1.e835dd77347e9p+20"
    assert arithmetic["conditional_gap_cleared"] is True
    assert arithmetic["inferential_interaction_cap_cleared"] is True
    assert arithmetic["preliminary_conditional_design_cleared"] is True
    assert arithmetic["power_or_beta_modeled"] is False
    assert receipt["status"] == "preliminary_conditional_precision_cleared"
    assert receipt["payload_sha256"] == (
        "4c30b0e07ed4b18cb6ea55e3515c82b9077ba89cdf6a8afac5af9219bb3c62ed"
    )
    assert receipt["conditional_assumptions"] == {
        "hypothetical_observed_gap_equals_target_gap": True,
        "power_or_beta_modeled": False,
        "variance_does_not_exceed_asserted_ceiling": True,
    }


@pytest.mark.unit
def test_infeasible_target_uses_maximum_n_without_claiming_required_resources() -> None:
    design = feasibility.FeasibilityDesign(
        candidate_count=11,
        alpha=0.05,
        practical_margin=0,
        target_gap=100_000,
        variance_upper_bound=statistics.CONTRAST_RANGE_WIDTH**2 // 2,
        variance_bound_source="bounded_sample_variance_maximum",
        calibration_artifact_sha256=None,
        compute_cap_interactions=100_000_000_000,
    )
    receipt = feasibility.build_feasibility_receipt(design)
    arithmetic = receipt["arithmetic"]

    assert arithmetic["maximum_blocks_by_score_cell_cap"] == 4_545
    assert arithmetic["evaluated_block_count"] == 4_545
    assert arithmetic["minimum_statistical_block_count"] is None
    assert arithmetic[
        "required_confirmatory_inferential_interactions_before_retries"
    ] is None
    assert arithmetic[
        "maximum_confirmatory_inferential_interactions_before_retries"
    ] == 24_983_101_440
    assert arithmetic["conditional_gap_cleared"] is False
    assert arithmetic["inferential_interaction_cap_cleared"] is False
    assert arithmetic["preliminary_conditional_design_cleared"] is False
    assert receipt["status"] == "preliminary_conditional_precision_not_cleared"


@pytest.mark.unit
def test_compute_cap_is_independently_fail_closed() -> None:
    ample = feasibility.FeasibilityDesign(
        candidate_count=11,
        alpha=0.05,
        practical_margin=0,
        target_gap=2_000_000,
        variance_upper_bound=statistics.CONTRAST_RANGE_WIDTH**2 // 2,
        variance_bound_source="bounded_sample_variance_maximum",
        calibration_artifact_sha256=None,
        compute_cap_interactions=100_000_000_000,
    )
    minimum = feasibility.build_feasibility_receipt(ample)["arithmetic"][
        "minimum_statistical_block_count"
    ]
    assert type(minimum) is int
    required = 11 * minimum * statistics.HORIZON
    constrained = feasibility.FeasibilityDesign(
        candidate_count=11,
        alpha=0.05,
        practical_margin=0,
        target_gap=2_000_000,
        variance_upper_bound=statistics.CONTRAST_RANGE_WIDTH**2 // 2,
        variance_bound_source="bounded_sample_variance_maximum",
        calibration_artifact_sha256=None,
        compute_cap_interactions=required - 1,
    )
    receipt = feasibility.build_feasibility_receipt(constrained)

    assert receipt["arithmetic"]["conditional_gap_cleared"] is True
    assert receipt["arithmetic"]["inferential_interaction_cap_cleared"] is False
    assert receipt["arithmetic"]["preliminary_conditional_design_cleared"] is False


@pytest.mark.unit
def test_calibration_variance_requires_a_bound_artifact_digest() -> None:
    with pytest.raises(feasibility.V3FeasibilityError, match="calibration_artifact_sha256"):
        feasibility.FeasibilityDesign(
            candidate_count=11,
            alpha=0.05,
            practical_margin=0,
            target_gap=500_000,
            variance_upper_bound=1_000_000,
            variance_bound_source="development_calibration_upper_bound",
            calibration_artifact_sha256=None,
            compute_cap_interactions=1_000_000_000,
        )

    design = feasibility.FeasibilityDesign(
        candidate_count=11,
        alpha=0.05,
        practical_margin=0,
        target_gap=500_000,
        variance_upper_bound=1_000_000,
        variance_bound_source="development_calibration_upper_bound",
        calibration_artifact_sha256="a" * 64,
        compute_cap_interactions=1_000_000_000,
    )
    assert design.calibration_artifact_sha256 == "a" * 64
    receipt = feasibility.build_feasibility_receipt(design)
    assert receipt["status"] == "unvalidated_calibration_assertion_not_gate_eligible"
    assert receipt["arithmetic"]["global_variance_ceiling_used"] is False
    assert receipt["arithmetic"]["preliminary_conditional_design_cleared"] is False
    assert receipt["authority"]["campaign_gate_eligible"] is False


@pytest.mark.unit
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("candidate_count", True),
        ("alpha", 1),
        ("practical_margin", 0.0),
        ("target_gap", True),
        ("variance_upper_bound", -1),
        ("compute_cap_interactions", 1.0),
    ],
)
def test_design_rejects_type_aliases_and_out_of_range_values(
    field: str,
    value: object,
) -> None:
    values: dict[str, object] = {
        "candidate_count": 11,
        "alpha": 0.05,
        "practical_margin": 0,
        "target_gap": 2_000_000,
        "variance_upper_bound": statistics.CONTRAST_RANGE_WIDTH**2 // 2,
        "variance_bound_source": "bounded_sample_variance_maximum",
        "calibration_artifact_sha256": None,
        "compute_cap_interactions": 100_000_000_000,
    }
    values[field] = value
    with pytest.raises(feasibility.V3FeasibilityError):
        feasibility.FeasibilityDesign(**values)  # type: ignore[arg-type]


@pytest.mark.unit
def test_receipt_is_canonical_replayable_and_mutation_resistant() -> None:
    design = feasibility.FeasibilityDesign(
        candidate_count=11,
        alpha=0.05,
        practical_margin=0,
        target_gap=100_000,
        variance_upper_bound=statistics.CONTRAST_RANGE_WIDTH**2 // 2,
        variance_bound_source="bounded_sample_variance_maximum",
        calibration_artifact_sha256=None,
        compute_cap_interactions=10_000_000_000,
    )
    raw = feasibility.canonical_feasibility_receipt_bytes(design)
    receipt = feasibility.parse_feasibility_receipt(raw)

    assert receipt == feasibility.build_feasibility_receipt(design)
    assert receipt["payload_sha256"] == hashlib.sha256(
        feasibility.canonical_feasibility_receipt_body_bytes(design)
    ).hexdigest()

    mutated = json.loads(raw)
    mutated["arithmetic"]["preliminary_conditional_design_cleared"] = True
    body = dict(mutated)
    body.pop("payload_sha256")
    mutated["payload_sha256"] = hashlib.sha256(
        json.dumps(body, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    tampered = json.dumps(mutated, separators=(",", ":"), sort_keys=True).encode()
    with pytest.raises(feasibility.V3FeasibilityError):
        feasibility.parse_feasibility_receipt(tampered)


@pytest.mark.unit
def test_even_feasible_arithmetic_never_grants_execution_or_heldout_authority() -> None:
    design = feasibility.FeasibilityDesign(
        candidate_count=11,
        alpha=0.05,
        practical_margin=0,
        target_gap=2_000_000,
        variance_upper_bound=statistics.CONTRAST_RANGE_WIDTH**2 // 2,
        variance_bound_source="bounded_sample_variance_maximum",
        calibration_artifact_sha256=None,
        compute_cap_interactions=100_000_000_000,
    )
    receipt = feasibility.build_feasibility_receipt(design)

    assert receipt["arithmetic"]["preliminary_conditional_design_cleared"] is True
    assert receipt["authority"] == {
        "campaign_gate_eligible": False,
        "compute_matched_claim_allowed": False,
        "execution_authorized": False,
        "heldout_tokens_may_be_requested": False,
        "literature_sota_claim_allowed": False,
        "named_panel_superiority_claim_allowed": False,
        "performance_claim_allowed": False,
        "preregistration_artifact": False,
        "resource_matched_claim_allowed": False,
        "scientific_promotion_allowed": False,
        "standalone_result_use_allowed": False,
        "universal_sota_claim_allowed": False,
    }
    assert math.isfinite(float.fromhex(receipt["arithmetic"]["q_hex"]))


@pytest.mark.unit
def test_formula_descriptor_is_content_addressed_and_denies_power_claim() -> None:
    raw = feasibility.canonical_conditional_precision_implementation_descriptor_bytes()
    descriptor = feasibility.conditional_precision_implementation_descriptor()

    assert json.loads(raw) == descriptor
    assert hashlib.sha256(raw).hexdigest() == (
        feasibility.CONDITIONAL_PRECISION_IMPLEMENTATION_SHA256
    )
    assert feasibility.CONDITIONAL_PRECISION_IMPLEMENTATION_SHA256 == (
        "b8f0d2698d58a4b019adafbf620124dc8880a72c6d8b6ad4211a219d1e30c2cd"
    )
    assert descriptor["power_or_beta_modeled"] is False
    assert descriptor["campaign_panel_bound"] is False
    descriptor["power_or_beta_modeled"] = True
    assert feasibility.conditional_precision_implementation_descriptor()[
        "power_or_beta_modeled"
    ] is False


@pytest.mark.unit
def test_parser_rejects_duplicate_keys_noncanonical_json_and_unknown_fields() -> None:
    design = feasibility.FeasibilityDesign(
        candidate_count=11,
        alpha=0.05,
        practical_margin=0,
        target_gap=100_000,
        variance_upper_bound=statistics.CONTRAST_RANGE_WIDTH**2 // 2,
        variance_bound_source="bounded_sample_variance_maximum",
        calibration_artifact_sha256=None,
        compute_cap_interactions=10_000_000_000,
    )
    canonical = feasibility.canonical_feasibility_receipt_bytes(design)
    with pytest.raises(feasibility.V3FeasibilityError, match="canonical"):
        feasibility.parse_feasibility_receipt(b" " + canonical)

    duplicate = canonical[:-1] + b',"status":"forged"}'
    with pytest.raises(feasibility.V3FeasibilityError, match="duplicate"):
        feasibility.parse_feasibility_receipt(duplicate)

    unknown = json.loads(canonical)
    unknown["authority"]["forged"] = False
    body = dict(unknown)
    body.pop("payload_sha256")
    unknown["payload_sha256"] = hashlib.sha256(
        json.dumps(body, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    raw = json.dumps(unknown, separators=(",", ":"), sort_keys=True).encode()
    with pytest.raises(feasibility.V3FeasibilityError, match="replay"):
        feasibility.parse_feasibility_receipt(raw)
