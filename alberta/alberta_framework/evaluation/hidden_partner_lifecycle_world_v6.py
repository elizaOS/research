"""Frozen, nonexecuting design for a noisy-world hidden-partner lifecycle study.

The object in this module is a design manifest, not a protocol or capability.
It has no learner construction, environment construction, random-key creation,
artifact writer, command-line entry point, or promotion decision.  Every
execution-bearing field is deliberately closed until all named prerequisites
exist and a later, separately reviewed protocol is frozen.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from typing import NoReturn, cast

HIDDEN_PARTNER_LIFECYCLE_WORLD_V6_SCHEMA = (
    "alberta.hidden-partner-lifecycle-world.design.v6"
)
DESIGN_STATUS = "NONEXECUTABLE_DESIGN"
CANONICALIZATION = "utf8-json-sort-keys-no-whitespace-no-nan-v1"

V6_INITIAL_ACTIVE_DESCRIPTORS: tuple[tuple[int, int], ...] = (
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

PRIMARY_CONDITION_ORDER: tuple[str, ...] = (
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

PRIMARY_CONDITION_OVERRIDES: tuple[tuple[tuple[str, object], ...], ...] = (
    (),
    (("grounded_world_learning_enabled", False),),
    (("representation_gradient_mode", "behavior_only"),),
    (("representation_gradient_mode", "world_only"),),
    (("representation_gradient_mode", "discard"),),
    (("state_learning_enabled", False),),
    (("memory_masked", True),),
    (("grounded_world_planning_enabled", False),),
    (("planning_enabled", False),),
    (("uniform_partner_belief", True),),
    (("feature_lifecycle_enabled", False),),
    (("carry_survivors", False),),
    (("active_utility_retention_decay", None),),
    (("retire_stale_features", False),),
    (("random_feature_curation", True),),
)

MATCHED_PRIMARY_REQUIREMENTS: tuple[str, ...] = (
    "same_future_seed_pair_and_stream_realization_for_every_arm",
    "same_initial_parameter_draws_for_shared_components",
    "same_fixed_shapes_state_budget_and_candidate_archive",
    "same_prediction_update_and_curation_attempt_schedule",
    "disabled_learning_paths_compute_the_proposed_update_then_discard_it",
    "disabled_read_paths_still_compute_the_predictions_and_diagnostics",
    "only_the_named_static_intervention_may_differ",
)

MANDATORY_CAUSAL_AUDITS: tuple[str, ...] = (
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

MANDATORY_RESOURCE_AUDITS: tuple[str, ...] = (
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

MANDATORY_FILTER_AUDITS: tuple[str, ...] = (
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

MANDATORY_COVERAGE_AUDITS: tuple[str, ...] = (
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

MANDATORY_PER_HEAD_AUDITS: tuple[str, ...] = (
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

FORBIDDEN_SEED_NAMESPACES: tuple[str, ...] = (
    "hidden-partner-v0-dev-v4-evidence-gated-lease-grid-a-v1",
    "hidden-partner-v0-dev-v5-retired-reacquisition-confirmation-a-v1",
)

SCOPE_LIMITATIONS: tuple[str, ...] = (
    "development_design_without_empirical_outcomes",
    "scripted_nonlearning_partner_not_coadaptation",
    "fixed_A_B_C_D_partner_identities_and_schedule_family",
    "binary_markov_world_with_two_binary_noisy_cues",
    "pairwise_degree_two_feature_archive_only",
    "twelve_active_slots_and_sixty_six_enumerated_candidates",
    "two_cycles_is_a_minimum_lifecycle_demonstration_not_a_scaling_result",
    "bayes_filter_is_an_evaluator_reference_not_agent_input",
    "exact_descriptor_identity_is_narrower_than_functional_feature_equivalence",
    "no_replay_and_no_dormant_downstream_weight_archive",
    "not_a_general_multiagent_or_learned_partner_claim",
    "not_an_Alberta_Plan_completion_claim",
)

OPEN_PREREQUISITES: tuple[tuple[str, str], ...] = (
    (
        "core_custom_initial_descriptor_bank_config",
        "exact_core_bank_contract_is_not_yet_source_bound_and_certified",
    ),
    (
        "matched_freeze_control",
        "matched_freeze_semantics_are_not_yet_source_bound_and_certified",
    ),
    (
        "grounded_world_row_bias_and_target_weights",
        "row_bias_and_per_target_loss_weights_are_not_yet_source_bound_and_certified",
    ),
    ("v6_runner", "runner_is_not_yet_source_bound_and_certified"),
    ("v6_strict_validator", "validator_is_not_yet_source_bound_and_certified"),
    (
        "development_threshold_calibration",
        "thresholds_are_not_yet_calibrated_frozen_and_certified",
    ),
    (
        "fresh_seed_namespace_and_seed_snapshot",
        "fresh_namespace_and_seed_snapshot_are_not_yet_reserved_and_certified",
    ),
)

_MAX_MANIFEST_BYTES = 256 * 1024


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise ValueError("manifest must contain canonical finite JSON data") from exc


def _design_body() -> dict[str, object]:
    conditions = [
        {
            "name": name,
            "static_overrides": dict(overrides),
            "primary": True,
            "matched_compute_required": True,
        }
        for name, overrides in zip(
            PRIMARY_CONDITION_ORDER,
            PRIMARY_CONDITION_OVERRIDES,
            strict=True,
        )
    ]
    return {
        "schema": HIDDEN_PARTNER_LIFECYCLE_WORLD_V6_SCHEMA,
        "status": DESIGN_STATUS,
        "canonicalization": CANONICALIZATION,
        "seed_namespace": None,
        "seed_count": 0,
        "seed_pairs": [],
        "seeds_derived": False,
        "execution_authorized": False,
        "evidence_authorized": False,
        "scientific_promotion_allowed": False,
        "claim_accepted": False,
        "thresholds_frozen": False,
        "environment": {
            "stream_contract": "hidden-partner-world-feedback-v1",
            "filter_contract": "hidden-partner-world-filter-v1",
            "regime_schedule": ["A", "B", "A", "D", "A", "C", "A", "B", "C"],
            "base_segment_lengths": [
                1536,
                1792,
                1536,
                1280,
                1536,
                1792,
                1536,
                1792,
                1792,
            ],
            "jitter_radius": 63,
            "partner_flip_probability": 0.05,
            "world_flip_probability": 0.03,
            "cue_flip_probabilities": [0.25, 0.35],
            "outcome_flip_probability": 0.15,
        },
        "lifecycle": {
            "initial_active_descriptors": [
                [left, right] for left, right in V6_INITIAL_ACTIVE_DESCRIPTORS
            ],
            "active_pair_slots": 12,
            "candidate_pair_slots": 66,
            "deployed_feature_dim": 24,
            "critical_pairs": {"C": [0, 2], "D": [4, 5]},
            "minimum_complete_cycles": 2,
            "entry_window_steps": 128,
            "tail_window_steps": 128,
            "final_window_steps": 256,
            "required_D_sequence": [
                "cycle_one_acquisition",
                "intercycle_retirement_while_D_absent",
                "cycle_two_reacquisition",
            ],
        },
        "primary_conditions": conditions,
        "matched_primary_requirements": list(MATCHED_PRIMARY_REQUIREMENTS),
        "diagnostics": [
            {
                "name": "uniform_action",
                "required": True,
                "primary": False,
                "scientific_inference_allowed": False,
                "requirement": (
                    "balanced_external_focal_actions_with_complete_joint_action_row_support"
                ),
            },
            {
                "name": "equal_cue",
                "required": True,
                "primary": False,
                "scientific_inference_allowed": False,
                "cue_flip_probabilities": [0.30, 0.30],
                "requirement": (
                    "preserve_mean_cue_noise_and_report_explicit_tie_aware_posterior_metrics"
                ),
            },
            {
                "name": "row_bias",
                "required": True,
                "primary": False,
                "scientific_inference_allowed": False,
                "requirement": (
                    "separate_joint_action_row_bias_from_feature_weight_and_target_weight_effects"
                ),
            },
        ],
        "mandatory_audits": {
            "causal": list(MANDATORY_CAUSAL_AUDITS),
            "resource": list(MANDATORY_RESOURCE_AUDITS),
            "filter": list(MANDATORY_FILTER_AUDITS),
            "coverage": list(MANDATORY_COVERAGE_AUDITS),
            "per_head": list(MANDATORY_PER_HEAD_AUDITS),
        },
        "forbidden_seed_namespaces": list(FORBIDDEN_SEED_NAMESPACES),
        "scope_limitations": list(SCOPE_LIMITATIONS),
        "open_prerequisites": [
            {
                "id": identifier,
                "status": "NOT_CERTIFIED",
                "requirement": requirement,
            }
            for identifier, requirement in OPEN_PREREQUISITES
        ],
    }


DESIGN_PAYLOAD_SHA256 = "62106deded06a85412b621a8b558c29a1c7db3fc2a135a8888fb63eedad98748"


def build_hidden_partner_lifecycle_world_v6_design() -> dict[str, object]:
    """Return a fresh copy of the exact inert v6 design manifest."""

    body = _design_body()
    return {**body, "payload_sha256": DESIGN_PAYLOAD_SHA256}


def validate_hidden_partner_lifecycle_world_v6_design(
    payload: Mapping[str, object],
) -> dict[str, object]:
    """Accept only the exact frozen manifest and return a detached copy."""

    if type(payload) is not dict:
        raise ValueError("payload must be a plain object containing the exact frozen v6 design")
    supplied = cast(dict[str, object], payload)
    expected = build_hidden_partner_lifecycle_world_v6_design()
    if supplied != expected or _canonical_json_bytes(supplied) != _canonical_json_bytes(expected):
        raise ValueError("payload differs from the exact frozen v6 design")
    body = dict(supplied)
    digest = body.pop("payload_sha256", None)
    if digest != hashlib.sha256(_canonical_json_bytes(body)).hexdigest():
        raise ValueError("payload differs from the exact frozen v6 design digest")
    return copy.deepcopy(expected)


def canonical_hidden_partner_lifecycle_world_v6_bytes(
    payload: Mapping[str, object],
) -> bytes:
    """Return canonical bytes only after exact frozen-design validation."""

    return _canonical_json_bytes(validate_hidden_partner_lifecycle_world_v6_design(payload))


def _duplicate_rejecting_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> NoReturn:
    raise ValueError(f"canonical JSON cannot contain {value}")


def load_hidden_partner_lifecycle_world_v6_design(data: bytes | str) -> dict[str, object]:
    """Decode canonical bytes and accept only the exact inert design."""

    if isinstance(data, str):
        try:
            raw = data.encode("ascii")
        except UnicodeEncodeError as exc:
            raise ValueError("manifest is not canonical ASCII JSON") from exc
    elif isinstance(data, bytes):
        raw = data
    else:
        raise TypeError("manifest must be bytes or text")
    if len(raw) > _MAX_MANIFEST_BYTES:
        raise ValueError("manifest exceeds the canonical size limit")
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ValueError("manifest is not canonical ASCII JSON") from exc
    try:
        value = json.loads(
            text,
            object_pairs_hook=_duplicate_rejecting_object,
            parse_constant=_reject_json_constant,
        )
    except json.JSONDecodeError as exc:
        raise ValueError("manifest is not canonical JSON") from exc
    if type(value) is not dict:
        raise ValueError("manifest must be a canonical JSON object")
    validated = validate_hidden_partner_lifecycle_world_v6_design(
        cast(dict[str, object], value)
    )
    if raw != _canonical_json_bytes(validated):
        raise ValueError("manifest bytes are not canonical")
    return validated


__all__ = [
    "CANONICALIZATION",
    "DESIGN_PAYLOAD_SHA256",
    "DESIGN_STATUS",
    "FORBIDDEN_SEED_NAMESPACES",
    "HIDDEN_PARTNER_LIFECYCLE_WORLD_V6_SCHEMA",
    "MANDATORY_CAUSAL_AUDITS",
    "MANDATORY_COVERAGE_AUDITS",
    "MANDATORY_FILTER_AUDITS",
    "MANDATORY_PER_HEAD_AUDITS",
    "MANDATORY_RESOURCE_AUDITS",
    "MATCHED_PRIMARY_REQUIREMENTS",
    "OPEN_PREREQUISITES",
    "PRIMARY_CONDITION_ORDER",
    "PRIMARY_CONDITION_OVERRIDES",
    "SCOPE_LIMITATIONS",
    "V6_INITIAL_ACTIVE_DESCRIPTORS",
    "build_hidden_partner_lifecycle_world_v6_design",
    "canonical_hidden_partner_lifecycle_world_v6_bytes",
    "load_hidden_partner_lifecycle_world_v6_design",
    "validate_hidden_partner_lifecycle_world_v6_design",
]
