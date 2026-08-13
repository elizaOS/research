"""Contracts for the bounded prospective-retention L0 outcome panel."""

from __future__ import annotations

import dataclasses

import pytest

from alberta_framework.evaluation import (
    prospective_lineage_retention_outcome_development as outcome,
)

pytestmark = pytest.mark.development


@pytest.fixture(scope="module")
def report() -> outcome.ProspectiveLineageRetentionOutcomeDevelopmentReport:
    return outcome.build_prospective_lineage_retention_outcome_development_report()


def _cells(
    report: outcome.ProspectiveLineageRetentionOutcomeDevelopmentReport,
    panel: str,
    routed: bool,
) -> tuple[outcome.ProspectiveLineageRetentionOutcomeCell, ...]:
    return tuple(
        cell for cell in report.cells if cell.prior_panel == panel and cell.routed is routed
    )


def test_config_is_exact_l0_nonpromoting_and_round_trips() -> None:
    config = outcome.ProspectiveLineageRetentionOutcomeDevelopmentConfig()
    payload = config.to_config()

    assert payload["evidence_level"] == "L0"
    assert payload["status"] == "descriptive-development-outcome-not-evidence"
    assert payload["scientific_promotion_allowed"] is False
    assert payload["thresholds_used"] is False
    assert payload["selection_performed"] is False
    assert (
        outcome.ProspectiveLineageRetentionOutcomeDevelopmentConfig.from_config(payload) == config
    )
    assert outcome.PROSPECTIVE_LINEAGE_RETENTION_OUTCOME_SCIENTIFIC_PROMOTION_ALLOWED is False

    wrong_type = dict(payload)
    wrong_type["confirmation_horizon"] = 2.0
    with pytest.raises(ValueError, match="types are not canonical"):
        outcome.ProspectiveLineageRetentionOutcomeDevelopmentConfig.from_config(wrong_type)

    promoted = dict(payload)
    promoted["scientific_promotion_allowed"] = True
    with pytest.raises(ValueError, match="not canonical"):
        outcome.ProspectiveLineageRetentionOutcomeDevelopmentConfig.from_config(promoted)

    with pytest.raises(ValueError, match="round-trip exactly through float32"):
        outcome.ProspectiveLineageRetentionOutcomeDevelopmentConfig(cost_ema_decay=0.1)
    with pytest.raises(ValueError, match="finite float32 range"):
        outcome.ProspectiveLineageRetentionOutcomeDevelopmentConfig(
            max_abs_cost=float.fromhex("0x1p+128")
        )


def test_report_is_nonwriting_descriptive_and_source_bound(
    report: outcome.ProspectiveLineageRetentionOutcomeDevelopmentReport,
) -> None:
    assert report.schema == outcome.PROSPECTIVE_LINEAGE_RETENTION_OUTCOME_REPORT_SCHEMA
    assert report.evidence_level == "L0"
    assert report.status == outcome.PROSPECTIVE_LINEAGE_RETENTION_OUTCOME_STATUS
    assert len(report.cells) == 8
    assert len(report.prior_panels) == 2
    assert len(report.config_sha256) == 64
    assert len(report.protocol_sha256) == 64
    assert len(report.core_source_sha256) == 64
    assert len(report.evaluator_source_sha256) == 64
    assert len(report.true_future_law_sha256) == 64
    assert report.thresholds_used is False
    assert report.winner_selected is False
    assert report.default_selected is False
    assert report.expected_direction_asserted is False
    assert report.artifact_written is False
    assert report.evidence_claimed is False
    assert report.scientific_promotion_allowed is False


def test_priors_are_birth_visible_law_is_evaluator_only_and_twins_share_prefix(
    report: outcome.ProspectiveLineageRetentionOutcomeDevelopmentReport,
) -> None:
    audit = report.matched_audit
    assert audit.priors_bound_at_birth_before_preparation
    assert audit.true_law_absent_from_raw_cells
    assert audit.ordinary_lru_b_fixed_before_outcomes
    assert audit.future_twins_bit_exact_through_eviction_commit
    assert audit.prefix_groups_checked == 4
    assert audit.first_future_observation_is_first_branch_divergence

    for panel in ("calibrated", "reversed_misspecified"):
        for routed in (True, False):
            twins = _cells(report, panel, routed)
            assert len(twins) == 2
            assert {cell.future_lineage for cell in twins} == {"B", "D"}
            assert len({cell.prefix_sha256 for cell in twins}) == 1
            assert len({cell.eviction_state_sha256 for cell in twins}) == 1
            assert len({cell.prior_declaration_sha256 for cell in twins}) == 1
            assert all(cell.future_observations_seen_before_eviction == 0 for cell in twins)
            assert all(not cell.true_evaluation_law_supplied_to_cell for cell in twins)
            assert twins[0].future_observations[0] != twins[1].future_observations[0]

    assert all(
        not cell.routed and cell.committed_victim_slot == 1
        for cell in report.cells
        if not cell.routed
    )


def test_calibrated_and_reversed_priors_are_distinct_declared_inputs(
    report: outcome.ProspectiveLineageRetentionOutcomeDevelopmentReport,
) -> None:
    calibrated = _cells(report, "calibrated", True)
    reversed_control = _cells(report, "reversed_misspecified", True)

    assert {cell.supplied_return_priors for cell in calibrated} == {(1.0, 0.75, 0.25)}
    assert {cell.supplied_return_priors for cell in reversed_control} == {(1.0, 0.25, 0.75)}
    assert {cell.initial_selected_victim_slot for cell in calibrated} == {2}
    assert {cell.initial_selected_victim_slot for cell in reversed_control} == {1}
    assert {cell.evicted_lineage for cell in calibrated} == {"D"}
    assert {cell.evicted_lineage for cell in reversed_control} == {"B"}
    assert len({cell.prior_declaration_sha256 for cell in report.cells}) == 2


def test_raw_cells_precede_exact_expected_and_minimax_aggregation(
    report: outcome.ProspectiveLineageRetentionOutcomeDevelopmentReport,
) -> None:
    law = {"B": 0.75, "D": 0.25}
    for panel in report.prior_panels:
        assert panel.true_future_law == (("B", 0.75), ("D", 0.25))
        assert panel.true_future_law_evaluator_only
        assert panel.threshold_used is False
        assert panel.winner_selected is False
        assert panel.default_selected is False
        assert panel.expected_direction_asserted is False
        for routed, expected_field, minimax_field in (
            (
                True,
                "routed_expected_recurrence_cost",
                "routed_minimax_recurrence_cost",
            ),
            (
                False,
                "unrouted_expected_recurrence_cost",
                "unrouted_minimax_recurrence_cost",
            ),
        ):
            raw = _cells(report, panel.prior_panel, routed)
            expected = sum(law[cell.future_lineage] * cell.raw_recurrence_cost for cell in raw)
            worst = max(cell.raw_recurrence_cost for cell in raw)
            assert getattr(panel, expected_field) == expected
            assert getattr(panel, minimax_field) == worst


def test_h2_confirmation_is_observation_bound_and_affects_only_later_preparation(
    report: outcome.ProspectiveLineageRetentionOutcomeDevelopmentReport,
) -> None:
    assert report.matched_audit.restoration_settled_after_h2
    assert report.matched_audit.restoration_effect_visible_only_to_later_preparation
    for cell in report.cells:
        expected_confirmation = cell.future_lineage == cell.evicted_lineage
        assert cell.future_lineage_archived_after_eviction is expected_confirmation
        assert cell.future_lineage_retained_after_eviction is (not expected_confirmation)
        assert cell.strict_h2_confirmation is expected_confirmation
        assert cell.core_confirmation_requested is expected_confirmation
        assert cell.restoration_applied is expected_confirmation
        assert cell.prior_restored is expected_confirmation
        assert cell.cost_updated is expected_confirmation
        assert cell.parameter_transplanted is False
        assert cell.pre_confirmation_target_prior == 0.0
        assert cell.pre_confirmation_target_score == 0.0
        if expected_confirmation:
            assert cell.archived_loss == 0.0
            assert cell.fresh_loss == 1.0
            assert cell.core_cost_observation == 1.0
            assert cell.restored_reacquisition_cost == 0.75
            assert cell.post_confirmation_target_prior in (0.25, 0.75)
            assert cell.post_confirmation_target_score > 0.0
            assert cell.restoration_count == 1
            assert cell.lineage_rebind_count == 1
        else:
            assert cell.core_cost_observation == 0.0
            assert cell.restored_reacquisition_cost == 0.5
            assert cell.post_confirmation_target_prior == 0.0
            assert cell.post_confirmation_target_score == 0.0
            assert cell.restoration_count == 0
            assert cell.lineage_rebind_count == 0


def test_route_and_future_cells_have_exact_equal_work_rng_and_byte_sizes(
    report: outcome.ProspectiveLineageRetentionOutcomeDevelopmentReport,
) -> None:
    audit = report.matched_audit
    assert audit.all_core_work_equal
    assert audit.core_preparations_per_cell == 3
    assert audit.core_settlements_per_cell == 2
    assert audit.core_authentication_repreparations_per_cell == 2
    assert audit.core_total_score_preparations_per_cell == 5
    assert audit.core_score_products_per_cell == 15
    assert audit.all_host_work_equal
    assert audit.host_squared_error_cells_per_cell == 10
    assert audit.all_rng_streams_equal
    assert audit.rng_stream_nbytes_per_cell == 0
    assert audit.rng_stream_sha256 == report.cells[0].rng_stream_sha256
    assert audit.all_state_nbytes_equal
    assert audit.state_nbytes_per_cell == 295
    assert audit.all_fixed_output_nbytes_equal
    assert audit.fixed_output_nbytes_per_cell == 256
    assert audit.persistent_capacity_growth == 0
    assert audit.replay_capacity == 0
    assert audit.archive_capacity == 1

    assert {cell.preparations for cell in report.cells} == {3}
    assert {cell.settlements for cell in report.cells} == {2}
    assert {cell.host_squared_error_cells for cell in report.cells} == {10}
    assert {cell.random_draws for cell in report.cells} == {0}
    assert {cell.rng_stream_nbytes for cell in report.cells} == {0}
    assert {cell.state_nbytes for cell in report.cells} == {295}
    assert {cell.fixed_output_nbytes for cell in report.cells} == {256}
    assert all(cell.every_core_update_applied for cell in report.cells)


def test_scaling_and_limitations_scope_the_finite_panel(
    report: outcome.ProspectiveLineageRetentionOutcomeDevelopmentReport,
) -> None:
    scaling = report.scaling
    assert scaling.persistent_state_formula_bytes == "64 + 57 * (K + 1) + K"
    assert scaling.measured_state_bytes_at_k3 == 295
    assert scaling.archive_capacity == 1
    assert scaling.core_prepare_work == "O(K)"
    assert scaling.core_settle_work_including_authentication == "O(K)"
    assert scaling.host_confirmation_work == "O(H * (K + 2))"
    assert scaling.exhaustive_panel_work == "O(P * R * F * H * K)"
    assert scaling.report_cell_count_formula == "P * R * F"
    assert scaling.realized_report_cell_count == 8
    assert scaling.unbounded_history_retained is False
    joined = " ".join(report.limitations)
    assert "not learned hazards" in joined
    assert "cannot authenticate host losses" in joined
    assert "metadata only" in joined
    assert "not demonstrated behavioral recovery" in joined
    assert "whole-agent forgetting" in joined
    assert "No threshold, winner, default, artifact, evidence, or promotion" in joined


def test_validator_reconstructs_exactly_and_rejects_finite_tamper(
    report: outcome.ProspectiveLineageRetentionOutcomeDevelopmentReport,
) -> None:
    receipt = outcome.validate_prospective_lineage_retention_outcome_development_report(report)
    assert receipt.valid
    assert receipt.raw_cell_count == 8
    assert receipt.deterministic_reconstruction_exact
    assert receipt.future_prefixes_exact
    assert receipt.equal_work_rng_and_bytes
    assert receipt.evidence_level == "L0"
    assert receipt.scientific_promotion_allowed is False
    assert len(receipt.report_sha256) == 64

    forged_cell = dataclasses.replace(
        report.cells[0],
        raw_recurrence_cost=report.cells[0].raw_recurrence_cost + 0.125,
    )
    forged = dataclasses.replace(report, cells=(forged_cell, *report.cells[1:]))
    with pytest.raises(ValueError, match="canonical bytes differ"):
        outcome.validate_prospective_lineage_retention_outcome_development_report(forged)

    zero_cell = report.cells[0]
    assert zero_cell.core_cost_observation == 0.0
    signed_zero_cell = dataclasses.replace(zero_cell, core_cost_observation=-0.0)
    signed_zero_report = dataclasses.replace(
        report,
        cells=(signed_zero_cell, *report.cells[1:]),
    )
    with pytest.raises(ValueError, match="canonical bytes differ"):
        outcome.validate_prospective_lineage_retention_outcome_development_report(
            signed_zero_report
        )

    bool_int_alias = dataclasses.replace(
        report,
        artifact_written=0,  # type: ignore[arg-type]
    )
    with pytest.raises(ValueError, match="exact reconstructed scalar type"):
        outcome.validate_prospective_lineage_retention_outcome_development_report(bool_int_alias)


def test_builder_and_validator_reject_wrong_outer_types() -> None:
    with pytest.raises(TypeError, match="config must be"):
        outcome.build_prospective_lineage_retention_outcome_development_report(
            object()  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="report must be"):
        outcome.validate_prospective_lineage_retention_outcome_development_report(
            object()  # type: ignore[arg-type]
        )
