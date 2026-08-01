"""Contracts for the inert hidden-partner world lifecycle v6 design."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path

import pytest

from alberta_framework.evaluation import hidden_partner_lifecycle_world_v6 as design

pytestmark = pytest.mark.unit


EXPECTED_BIRTH_BANK = (
    (0, 4),
    (0, 5),
    (1, 4),
    (1, 5),
    (1, 6),
    (1, 7),
    (2, 4),
    (2, 5),
    (2, 6),
    (2, 7),
    (4, 6),
    (5, 7),
)

EXPECTED_PRIMARY_CONDITIONS = (
    "full",
    "grounded_model_frozen",
    "world_credit_off",
    "behavior_credit_off",
    "all_representation_credit_off",
    "state_frozen",
    "recurrent_memory_masked",
    "table_planner",
    "no_planning",
    "uniform_partner",
    "lifecycle_frozen",
    "no_identity_carry",
    "no_retention_floor",
    "retirement_disabled",
    "random_curation",
)

EXPECTED_CONDITION_OVERRIDES = (
    {},
    {"grounded_world_learning_enabled": False},
    {"representation_gradient_mode": "behavior_only"},
    {"representation_gradient_mode": "world_only"},
    {"representation_gradient_mode": "discard"},
    {"state_learning_enabled": False},
    {"memory_masked": True},
    {"grounded_world_planning_enabled": False},
    {"planning_enabled": False},
    {"uniform_partner_belief": True},
    {"feature_lifecycle_enabled": False},
    {"carry_survivors": False},
    {"active_utility_retention_decay": None},
    {"retire_stale_features": False},
    {"random_feature_curation": True},
)

EXPECTED_CAUSAL_AUDITS = (
    "observation_t_precedes_decision_t",
    "decision_t_is_cached_before_transition_t",
    "transition_t_is_scored_before_any_update",
    "representation_credit_uses_preupdate_predictions",
    "retirement_at_transition_t_changes_decision_t_plus_1",
    "reacquisition_follows_an_observed_retirement",
    "descriptor_transactions_are_atomic",
    "survivor_columns_follow_descriptor_identity_not_slot",
    "retired_identity_heads_are_reset_before_reacquisition",
    "stream_rng_is_action_independent",
    "primary_conditions_share_stream_and_initialization_realizations",
    "hidden_world_regime_boundaries_and_noise_bits_never_enter_agent_inputs",
    "bayes_filter_is_evaluator_only_and_cannot_change_actions_or_updates",
)

EXPECTED_RESOURCE_AUDITS = (
    "fixed_state_tree_structure",
    "initial_peak_and_final_state_nbytes",
    "trainable_and_administrative_scalar_counts",
    "twelve_active_sixty_six_candidate_twenty_four_deployed_dimensions",
    "router_and_consumer_feature_axis_counts",
    "per_decision_prediction_cell_counts",
    "per_transition_update_attempt_counts",
    "rng_split_and_draw_counts",
    "zero_replay_and_zero_dormant_weight_archive",
    "matched_compute_except_the_named_static_intervention",
)

EXPECTED_FILTER_AUDITS = (
    "stream_and_filter_contract_versions_match_manifest",
    "stream_and_filter_world_cue_and_outcome_probabilities_match",
    "initial_posterior_uses_only_current_ordinary_cues",
    "feedback_is_outcome_times_previous_focal_and_partner_signs",
    "posterior_order_is_observe_then_markov_propagate_then_next_cues",
    "filter_never_reads_hidden_world_regime_boundary_or_noise_bits",
    "posterior_and_reward_cells_are_finite_and_normalized",
    "filter_decisions_report_ties_regret_margin_and_optimal_value",
    "every_active_step_has_one_prequential_filter_record",
    "filter_outputs_are_diagnostic_only_and_never_agent_inputs",
)

EXPECTED_COVERAGE_AUDITS = (
    "at_least_two_complete_schedule_cycles",
    "every_cycle_contains_all_A_B_C_D_regimes",
    "every_regime_occurrence_has_full_128_step_entry_and_tail_windows",
    "terminal_run_has_a_full_256_step_final_window",
    "windows_do_not_cross_segment_or_run_boundaries",
    "D_is_acquired_in_cycle_one_retired_between_cycles_and_reacquired_in_cycle_two",
    "D_retirement_is_observed_while_D_is_absent",
    "C_retention_is_measured_across_recurrences",
    "all_fifteen_primary_conditions_have_matched_complete_coverage",
    "uniform_action_equal_cue_and_row_bias_diagnostics_have_complete_coverage",
)

EXPECTED_PER_HEAD_AUDITS = (
    "all_four_joint_action_rows_are_reported",
    "all_eight_observation_heads_plus_reward_and_discount_are_reported",
    "prequential_loss_is_reported_for_every_row_and_head",
    "only_the_executed_joint_action_row_updates",
    "unexecuted_rows_remain_bit_exact",
    "weights_and_row_biases_are_audited_separately",
    "target_weights_are_explicit_finite_and_bound",
    "observation_reward_and_discount_aggregates_cannot_hide_head_failures",
    "representation_gradient_contribution_is_reported_per_head",
    "nonfinite_or_missing_row_head_values_fail_closed",
    "minimum_support_is_reported_for_every_row_and_head",
    "uniform_action_diagnostic_reports_row_support",
    "equal_cue_diagnostic_reports_tie_aware_posterior_metrics",
    "row_bias_diagnostic_separates_bias_only_and_feature_weight_predictions",
)


def test_design_is_explicitly_nonexecuting_unseeded_and_nonpromoting() -> None:
    payload = design.build_hidden_partner_lifecycle_world_v6_design()

    assert payload["schema"] == "alberta.hidden-partner-lifecycle-world.design.v6"
    assert payload["status"] == "NONEXECUTABLE_DESIGN"
    assert payload["seed_namespace"] is None
    assert payload["seed_count"] == 0
    assert payload["seed_pairs"] == []
    assert payload["seeds_derived"] is False
    assert payload["execution_authorized"] is False
    assert payload["evidence_authorized"] is False
    assert payload["scientific_promotion_allowed"] is False
    assert payload["claim_accepted"] is False
    assert payload["thresholds_frozen"] is False


def test_noisy_stream_filter_and_lifecycle_geometry_are_exact() -> None:
    payload = design.build_hidden_partner_lifecycle_world_v6_design()
    environment = payload["environment"]
    lifecycle = payload["lifecycle"]

    assert environment == {
        "stream_contract": "hidden-partner-world-feedback-v1",
        "filter_contract": "hidden-partner-world-filter-v1",
        "regime_schedule": ["A", "B", "A", "D", "A", "C", "A", "B", "C"],
        "base_segment_lengths": [1536, 1792, 1536, 1280, 1536, 1792, 1536, 1792, 1792],
        "jitter_radius": 63,
        "partner_flip_probability": 0.05,
        "world_flip_probability": 0.03,
        "cue_flip_probabilities": [0.25, 0.35],
        "outcome_flip_probability": 0.15,
    }
    assert lifecycle["minimum_complete_cycles"] == 2
    assert lifecycle["entry_window_steps"] == 128
    assert lifecycle["tail_window_steps"] == 128
    assert lifecycle["final_window_steps"] == 256
    assert lifecycle["required_D_sequence"] == [
        "cycle_one_acquisition",
        "intercycle_retirement_while_D_absent",
        "cycle_two_reacquisition",
    ]
    assert lifecycle["critical_pairs"] == {"C": [0, 2], "D": [4, 5]}


def test_birth_bank_is_exact_canonical_and_contains_no_initial_shortcut() -> None:
    payload = design.build_hidden_partner_lifecycle_world_v6_design()
    bank = tuple(tuple(pair) for pair in payload["lifecycle"]["initial_active_descriptors"])

    assert bank == EXPECTED_BIRTH_BANK
    assert len(bank) == len(set(bank)) == 12
    assert all(0 <= left < right < 12 for left, right in bank)
    assert (0, 2) not in bank
    assert (4, 5) not in bank
    assert (0, 6) not in bank
    assert (0, 7) not in bank
    assert all(left < 8 and right < 8 for left, right in bank)


def test_exact_fifteen_primary_arms_are_ordered_and_matched() -> None:
    payload = design.build_hidden_partner_lifecycle_world_v6_design()
    conditions = payload["primary_conditions"]

    assert tuple(condition["name"] for condition in conditions) == EXPECTED_PRIMARY_CONDITIONS
    assert tuple(condition["static_overrides"] for condition in conditions) == (
        EXPECTED_CONDITION_OVERRIDES
    )
    assert len({condition["name"] for condition in conditions}) == 15
    assert all(condition["primary"] is True for condition in conditions)
    assert all(condition["matched_compute_required"] is True for condition in conditions)
    assert payload["matched_primary_requirements"] == [
        "same_future_seed_pair_and_stream_realization_for_every_arm",
        "same_initial_parameter_draws_for_shared_components",
        "same_fixed_shapes_state_budget_and_candidate_archive",
        "same_prediction_update_and_curation_attempt_schedule",
        "disabled_learning_paths_compute_the_proposed_update_then_discard_it",
        "disabled_read_paths_still_compute_the_predictions_and_diagnostics",
        "only_the_named_static_intervention_may_differ",
    ]


def test_diagnostics_are_required_but_excluded_from_primary_inference() -> None:
    diagnostics = design.build_hidden_partner_lifecycle_world_v6_design()["diagnostics"]

    assert tuple(item["name"] for item in diagnostics) == (
        "uniform_action",
        "equal_cue",
        "row_bias",
    )
    assert all(item["required"] is True for item in diagnostics)
    assert all(item["primary"] is False for item in diagnostics)
    assert all(item["scientific_inference_allowed"] is False for item in diagnostics)
    assert diagnostics[0]["requirement"] == (
        "balanced_external_focal_actions_with_complete_joint_action_row_support"
    )
    assert diagnostics[1]["cue_flip_probabilities"] == [0.30, 0.30]
    assert diagnostics[1]["requirement"] == (
        "preserve_mean_cue_noise_and_report_explicit_tie_aware_posterior_metrics"
    )
    assert diagnostics[2]["requirement"] == (
        "separate_joint_action_row_bias_from_feature_weight_and_target_weight_effects"
    )


def test_all_mandatory_audit_families_are_exact_and_unique() -> None:
    audits = design.build_hidden_partner_lifecycle_world_v6_design()["mandatory_audits"]

    assert set(audits) == {"causal", "resource", "filter", "coverage", "per_head"}
    assert tuple(audits["causal"]) == EXPECTED_CAUSAL_AUDITS
    assert tuple(audits["resource"]) == EXPECTED_RESOURCE_AUDITS
    assert tuple(audits["filter"]) == EXPECTED_FILTER_AUDITS
    assert tuple(audits["coverage"]) == EXPECTED_COVERAGE_AUDITS
    assert tuple(audits["per_head"]) == EXPECTED_PER_HEAD_AUDITS
    flat = [audit for family in audits.values() for audit in family]
    assert len(flat) == len(set(flat))


def test_old_namespaces_are_denied_and_every_execution_prerequisite_is_open() -> None:
    payload = design.build_hidden_partner_lifecycle_world_v6_design()

    assert payload["forbidden_seed_namespaces"] == [
        "hidden-partner-v0-dev-v4-evidence-gated-lease-grid-a-v1",
        "hidden-partner-v0-dev-v5-retired-reacquisition-confirmation-a-v1",
    ]
    prerequisites = payload["open_prerequisites"]
    assert tuple(item["id"] for item in prerequisites) == (
        "core_custom_initial_descriptor_bank_config",
        "matched_freeze_control",
        "grounded_world_row_bias_and_target_weights",
        "v6_runner",
        "v6_strict_validator",
        "development_threshold_calibration",
        "fresh_seed_namespace_and_seed_snapshot",
    )
    assert all(item["status"] == "NOT_CERTIFIED" for item in prerequisites)
    assert payload["scope_limitations"]
    assert "not_an_Alberta_Plan_completion_claim" in payload["scope_limitations"]


def test_canonical_roundtrip_is_exact_digest_bound_and_fail_closed() -> None:
    payload = design.build_hidden_partner_lifecycle_world_v6_design()
    encoded = design.canonical_hidden_partner_lifecycle_world_v6_bytes(payload)
    decoded = design.load_hidden_partner_lifecycle_world_v6_design(encoded)

    assert decoded == payload
    assert encoded == design.canonical_hidden_partner_lifecycle_world_v6_bytes(decoded)
    body = dict(payload)
    digest = body.pop("payload_sha256")
    canonical_body = json.dumps(
        body,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    assert digest == design.DESIGN_PAYLOAD_SHA256
    assert digest == hashlib.sha256(canonical_body).hexdigest()

    tampered = copy.deepcopy(payload)
    tampered["seed_count"] = 1
    with pytest.raises(ValueError, match="exact frozen v6 design"):
        design.validate_hidden_partner_lifecycle_world_v6_design(tampered)
    with pytest.raises(ValueError, match="canonical"):
        design.load_hidden_partner_lifecycle_world_v6_design(encoded + b"\n")
    duplicate = b'{"schema":"a","schema":"b"}'
    with pytest.raises(ValueError, match="duplicate"):
        design.load_hidden_partner_lifecycle_world_v6_design(duplicate)
    with pytest.raises(ValueError, match="canonical"):
        design.load_hidden_partner_lifecycle_world_v6_design(b'{"value":NaN}')


def test_source_has_no_execution_seed_namespace_or_evidence_authority() -> None:
    source = Path(design.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)

    forbidden_source_fragments = (
        "hidden_partner_lifecycle_v2",
        "hidden_partner_development",
        "derive_hidden_partner_seed_pairs",
        "HiddenPartnerDevelopmentRunner",
        "HiddenPartnerDevelopmentProtocol",
        "jax.random",
        "subprocess",
    )
    assert all(fragment not in source for fragment in forbidden_source_fragments)

    imported_roots: set[str] = set()
    called_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_roots.add(node.module.split(".")[0])
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                called_names.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                called_names.add(node.func.attr)

    assert imported_roots <= {"__future__", "copy", "hashlib", "json", "collections", "typing"}
    assert called_names.isdisjoint(
        {
            "open",
            "run",
            "start",
            "update",
            "initialize",
            "derive_hidden_partner_seed_pairs",
            "issue_namespace",
            "publish",
            "write_text",
            "write_bytes",
        }
    )
    assert "if __name__" not in source
