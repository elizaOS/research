"""Contracts for the factorized pre-action decision-utility L0 probe."""

from __future__ import annotations

import copy
import dataclasses
import inspect
import math
from types import MappingProxyType
from typing import cast

import pytest

from alberta_framework.evaluation import factorized_preaction_decision_utility_development as lane
from alberta_framework.evaluation.factorized_preaction_decision_utility_development import (
    ARM_NAMES,
    ARTIFACT_AUTHORITY,
    ASSESSMENT_STATUS,
    BENCHMARK_EXECUTION_AUTHORITY,
    BRANCH_NAMES,
    DEVELOPMENT_ONLY,
    DEVELOPMENT_SCHEMA,
    EVIDENCE_CLAIMED,
    EVIDENCE_LEVEL,
    OUTPUT_WRITES_ALLOWED,
    POST_REVEAL_COMPARATOR_NAMES,
    PREACTION_ARM_NAMES,
    SCIENTIFIC_PROMOTION_ALLOWED,
    THRESHOLDS_DEFINED,
    FactorizedPreactionDecisionUtilityConfig,
    run_factorized_preaction_decision_utility_development,
    validate_factorized_preaction_decision_utility_report,
)

pytestmark = [pytest.mark.unit, pytest.mark.development]


@pytest.fixture(scope="module")
def report() -> dict[str, object]:
    return run_factorized_preaction_decision_utility_development(
        FactorizedPreactionDecisionUtilityConfig(
            prefix_steps=32,
            evaluation_steps=16,
        )
    )


def _branches(report: dict[str, object]) -> dict[str, dict[str, object]]:
    return {
        cast(str, branch["branch"]): branch
        for branch in cast(list[dict[str, object]], report["branches"])
    }


def _summaries(report: dict[str, object]) -> dict[str, dict[str, object]]:
    return cast(dict[str, dict[str, object]], report["branch_summaries"])


def _arm_summary(
    report: dict[str, object],
    branch: str,
    arm: str,
) -> dict[str, object]:
    arms = cast(dict[str, dict[str, object]], _summaries(report)[branch]["arms"])
    return arms[arm]


@pytest.mark.parametrize(
    ("field", "value", "exception"),
    [
        ("run_id", "unversioned", ValueError),
        ("prefix_steps", True, TypeError),
        ("prefix_steps", 15, ValueError),
        ("prefix_steps", 17, ValueError),
        ("evaluation_steps", 8, ValueError),
        ("evaluation_steps", 17, ValueError),
        ("pseudocount", 1, TypeError),
        ("pseudocount", float("nan"), ValueError),
        ("pseudocount", 1_000_001.0, ValueError),
        ("max_total_source_events", 2, ValueError),
        ("max_logical_state_bytes", 2, ValueError),
        ("max_raw_trajectory_bytes", False, TypeError),
        ("max_report_bytes", 0, ValueError),
    ],
)
def test_config_is_exact_bounded_and_roundtrips(
    field: str,
    value: object,
    exception: type[Exception],
) -> None:
    with pytest.raises(exception):
        FactorizedPreactionDecisionUtilityConfig(**{field: value})  # type: ignore[arg-type]

    config = FactorizedPreactionDecisionUtilityConfig()
    assert FactorizedPreactionDecisionUtilityConfig.from_config(config.to_config()) == config
    with pytest.raises(TypeError, match="exact JSON object"):
        FactorizedPreactionDecisionUtilityConfig.from_config(
            MappingProxyType(config.to_config())
        )
    malformed = config.to_config()
    malformed["extra"] = 1
    with pytest.raises(ValueError, match="fields differ"):
        FactorizedPreactionDecisionUtilityConfig.from_config(malformed)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("prefix_steps", 64.0),
        ("evaluation_steps", 32.0),
        ("pseudocount", 1),
        ("max_total_source_events", True),
        ("max_report_bytes", 4_194_304.0),
    ],
)
def test_config_rejects_numeric_and_boolean_aliases(
    field: str,
    replacement: object,
) -> None:
    payload = FactorizedPreactionDecisionUtilityConfig().to_config()
    payload[field] = replacement
    with pytest.raises((TypeError, ValueError), match="invalid config payload"):
        FactorizedPreactionDecisionUtilityConfig.from_config(payload)


def test_common_prefix_has_exhaustive_balanced_joint_support_before_evaluation(
    report: dict[str, object],
) -> None:
    support = cast(dict[str, object], report["conditional_support_receipt"])
    assert support["own_partner_counts"] == [8, 8, 8, 8]
    assert support["minimum_cell_count"] == 8
    assert support["complete_before_evaluation"] is True
    assert support["support_cell_order"] == [[0, 0], [0, 1], [1, 0], [1, 1]]
    prefix = cast(dict[str, object], report["common_prefix"])
    assert prefix["source_event_count"] == 32
    assert prefix["passes_over_source"] == 1
    assert prefix["learner_updates"] == 32
    assert prefix["source_events_retained_in_report"] == 0


def test_preaction_api_cannot_receive_partner_action_outcome_or_branch() -> None:
    assert tuple(inspect.signature(lane._freeze_preaction_surface).parameters) == (
        "state",
        "cue",
        "pseudocount",
    )
    assert tuple(inspect.signature(lane._form_preaction_decisions).parameters) == (
        "surface",
    )
    inverse_fields = {field.name for field in dataclasses.fields(lane._InverseObservationPair)}
    assert "partner_action" not in inverse_fields
    assert tuple(inspect.signature(lane._retrospective_inverse_distribution).parameters) == (
        "state",
        "observation_pair",
        "pseudocount",
    )

    state = lane._initial_state()
    surface = lane._freeze_preaction_surface(state, 0, 1.0)
    assert surface.partner_action_revealed is False
    assert surface.outcome_revealed is False
    unrevealed = lane._InverseObservationPair(
        own_action=0,
        post_physical_bit=0,
        outcome_revealed=False,
    )
    with pytest.raises(ValueError, match="revealed outcome"):
        lane._retrospective_inverse_distribution(state, unrevealed, 1.0)


def test_decision_timing_places_all_causal_actions_before_reveal(
    report: dict[str, object],
) -> None:
    for branch in _branches(report).values():
        for event in cast(list[dict[str, object]], branch["raw_events"]):
            order = cast(list[str], event["decision_operation_order"])
            reveal = order.index("reveal_partner_action")
            assert order.index("form_learned_behavior_marginal_action") < reveal
            assert order.index("form_uniform_belief_control_action") < reveal
            assert (
                order.index(
                    "record_inverse_input_unavailable_and_use_fixed_uniform_fallback"
                )
                < reveal
            )
            assert order.index("form_actual_partner_conditional_model_comparator") > reveal
            assert order.index("form_evaluator_true_reward_comparator") > reveal
            arms = cast(dict[str, dict[str, object]], event["arms"])
            for arm in PREACTION_ARM_NAMES:
                assert arms[arm]["causally_valid_preaction"] is True
                assert arms[arm]["partner_action_consumed_before_action"] is False
                assert arms[arm]["post_observation_consumed_before_action"] is False
            for arm in POST_REVEAL_COMPARATOR_NAMES:
                assert arms[arm]["causally_valid_preaction"] is False
                assert arms[arm]["evaluator_only"] is True


def test_inverse_misuse_has_fixed_causal_fallback_and_never_uses_later_output(
    report: dict[str, object],
) -> None:
    contract = cast(dict[str, object], report["inverse_action_misuse_contract"])
    assert contract == {
        "candidate_signal": "retrospective_inverse_action_distribution",
        "available_before_simultaneous_action": False,
        "missing_preaction_input": "post_physical_observation",
        "fixed_fallback_belief": [0.5, 0.5],
        "fallback_uses_post_outcome": False,
        "later_inverse_output_reused_for_decision": False,
        "fallback_matches_uniform_belief_control": True,
    }
    assert report["inverse_action_misuse_contract_sha256"] == lane._sha256(contract)
    for branch in _branches(report).values():
        for event in cast(list[dict[str, object]], branch["raw_events"]):
            arms = cast(dict[str, dict[str, object]], event["arms"])
            inverse = arms["inverse_action_unavailable_fallback"]
            uniform = arms["uniform_belief_control"]
            assert inverse["belief"] == uniform["belief"] == [0.5, 0.5]
            assert inverse["chosen_action"] == uniform["chosen_action"]
            assert inverse["fixed_inverse_unavailable_fallback_used"] is True
            assert inverse["retrospective_inverse_output_consumed"] is False
            for row in cast(
                list[dict[str, object]],
                event["retrospective_inverse_diagnostics_by_own_action"],
            ):
                assert row["formed_after_outcome_reveal"] is True
                assert row["fed_back_into_any_decision"] is False


def test_all_model_arms_share_byte_identical_snapshot_and_joint_cell_work(
    report: dict[str, object],
) -> None:
    model_arms = (*PREACTION_ARM_NAMES, "actual_partner_conditional_model_ceiling")
    expected_order = [[0, 0], [0, 1], [1, 0], [1, 1]]
    events_by_branch = {
        name: cast(list[dict[str, object]], branch["raw_events"])
        for name, branch in _branches(report).items()
    }
    for step in range(16):
        event_surface_hashes = {
            events_by_branch[branch][step]["frozen_preaction_surface_sha256"]
            for branch in BRANCH_NAMES
        }
        event_table_hashes = {
            events_by_branch[branch][step]["frozen_complete_conditional_table_sha256"]
            for branch in BRANCH_NAMES
        }
        assert len(event_surface_hashes) == 1
        assert len(event_table_hashes) == 1
        for branch in BRANCH_NAMES:
            event = events_by_branch[branch][step]
            arms = cast(dict[str, dict[str, object]], event["arms"])
            for arm in model_arms:
                assert arms[arm]["learned_predictor_snapshot_consumed"] is True
                assert arms[arm]["frozen_preaction_surface_sha256"] in event_surface_hashes
                assert arms[arm]["frozen_complete_conditional_table_sha256"] in event_table_hashes
                assert arms[arm]["joint_cell_evaluation_order"] == expected_order
            true_ceiling = arms["evaluator_true_reward_ceiling"]
            assert true_ceiling["learned_predictor_snapshot_consumed"] is False
            assert true_ceiling["joint_cell_evaluation_order"] == []


def test_branch_interventions_change_only_the_named_source_factor(
    report: dict[str, object],
) -> None:
    interventions = {
        name: branch["evaluator_only_intervention"]
        for name, branch in _branches(report).items()
    }
    assert interventions == {
        "control": {
            "partner_policy_mapping_changed": False,
            "physical_reward_law_changed": False,
        },
        "partner_policy_drift": {
            "partner_policy_mapping_changed": True,
            "physical_reward_law_changed": False,
        },
        "physical_reward_law_drift": {
            "partner_policy_mapping_changed": False,
            "physical_reward_law_changed": True,
        },
    }
    branches = _branches(report)
    control_events = cast(list[dict[str, object]], branches["control"]["raw_events"])
    policy_events = cast(
        list[dict[str, object]],
        branches["partner_policy_drift"]["raw_events"],
    )
    law_events = cast(
        list[dict[str, object]],
        branches["physical_reward_law_drift"]["raw_events"],
    )
    for control, policy, law in zip(
        control_events,
        policy_events,
        law_events,
        strict=True,
    ):
        assert control["cue"] == policy["cue"] == law["cue"]
        assert policy["partner_action_revealed_after_preaction_decisions"] == 1 - cast(
            int,
            control["partner_action_revealed_after_preaction_decisions"],
        )
        assert law["partner_action_revealed_after_preaction_decisions"] == control[
            "partner_action_revealed_after_preaction_decisions"
        ]


def test_every_proposed_action_is_scored_on_the_same_counterfactual_event(
    report: dict[str, object],
) -> None:
    for branch in _branches(report).values():
        assert branch["evaluation_updates"] == 0
        assert branch["starts_from_state_sha256"] == branch["ends_with_state_sha256"]
        for event in cast(list[dict[str, object]], branch["raw_events"]):
            outcomes = cast(
                list[dict[str, object]],
                event["counterfactual_true_outcomes_by_own_action"],
            )
            assert [row["own_action"] for row in outcomes] == [0, 1]
            assert {cast(int, row["reward"]) for row in outcomes} == {0, 1}
            arms = cast(dict[str, dict[str, object]], event["arms"])
            assert tuple(arms) == ARM_NAMES
            for arm in arms.values():
                action = cast(int, arm["chosen_action"])
                assert arm["realized_reward"] == outcomes[action]["reward"]
                assert arm["realized_post_physical_bit"] == outcomes[action][
                    "post_physical_bit"
                ]
                assert arm["realized_regret"] == 1 - cast(int, arm["realized_reward"])
                for field in (
                    "decision_reward_prediction_squared_error",
                    "decision_physical_prediction_squared_error",
                    "learned_conditional_reward_prediction_squared_error",
                    "learned_conditional_physical_prediction_squared_error",
                ):
                    assert math.isfinite(cast(float, arm[field]))


def test_raw_one_step_utility_outcomes_cover_all_three_branches(
    report: dict[str, object],
) -> None:
    expected_rewards = {
        "control": {
            "learned_behavior_marginal": 0.75,
            "uniform_belief_control": 0.5,
            "inverse_action_unavailable_fallback": 0.5,
            "actual_partner_conditional_model_ceiling": 1.0,
            "evaluator_true_reward_ceiling": 1.0,
        },
        "partner_policy_drift": {
            "learned_behavior_marginal": 0.25,
            "uniform_belief_control": 0.5,
            "inverse_action_unavailable_fallback": 0.5,
            "actual_partner_conditional_model_ceiling": 1.0,
            "evaluator_true_reward_ceiling": 1.0,
        },
        "physical_reward_law_drift": {
            "learned_behavior_marginal": 0.25,
            "uniform_belief_control": 0.5,
            "inverse_action_unavailable_fallback": 0.5,
            "actual_partner_conditional_model_ceiling": 0.0,
            "evaluator_true_reward_ceiling": 1.0,
        },
    }
    for branch, arms in expected_rewards.items():
        for arm, reward in arms.items():
            summary = _arm_summary(report, branch, arm)
            assert summary["mean_realized_reward"] == reward
            assert summary["mean_realized_regret"] == 1.0 - reward
            chosen = cast(list[int], summary["chosen_actions"])
            assert len(chosen) == 16
            assert sum(cast(list[int], summary["chosen_action_counts"])) == 16


def test_factor_metrics_localize_policy_and_law_changes(
    report: dict[str, object],
) -> None:
    summaries = _summaries(report)
    control_behavior = cast(dict[str, object], summaries["control"]["behavior"])
    policy_behavior = cast(
        dict[str, object],
        summaries["partner_policy_drift"]["behavior"],
    )
    law_behavior = cast(
        dict[str, object],
        summaries["physical_reward_law_drift"]["behavior"],
    )
    assert policy_behavior["mean_nll"] > control_behavior["mean_nll"]  # type: ignore[operator]
    assert policy_behavior["mean_brier"] > control_behavior["mean_brier"]  # type: ignore[operator]
    assert law_behavior == control_behavior

    control_table = summaries["control"]["grounded_complete_conditional_table"]
    policy_table = summaries["partner_policy_drift"][
        "grounded_complete_conditional_table"
    ]
    law_table = summaries["physical_reward_law_drift"][
        "grounded_complete_conditional_table"
    ]
    assert policy_table == control_table
    assert cast(dict[str, object], law_table)[
        "reward_mse_over_all_own_partner_cells"
    ] > cast(dict[str, object], control_table)[  # type: ignore[operator]
        "reward_mse_over_all_own_partner_cells"
    ]

    control_inverse = summaries["control"][
        "retrospective_inverse_over_all_counterfactual_own_actions"
    ]
    policy_inverse = summaries["partner_policy_drift"][
        "retrospective_inverse_over_all_counterfactual_own_actions"
    ]
    law_inverse = cast(
        dict[str, object],
        summaries["physical_reward_law_drift"][
            "retrospective_inverse_over_all_counterfactual_own_actions"
        ],
    )
    assert policy_inverse == control_inverse
    assert law_inverse["mean_nll"] > cast(dict[str, object], control_inverse)[  # type: ignore[operator]
        "mean_nll"
    ]
    assert law_inverse["decision_feedback_count"] == 0


def test_action_changes_and_branch_deltas_are_raw_and_reconstructable(
    report: dict[str, object],
) -> None:
    for branch in BRANCH_NAMES:
        for arm in ARM_NAMES:
            summary = _arm_summary(report, branch, arm)
            actions = cast(list[int], summary["chosen_actions"])
            expected_changes = sum(
                int(actions[index] != actions[index - 1])
                for index in range(1, len(actions))
            )
            assert summary["action_changes_from_previous_event"] == expected_changes

    deltas = cast(dict[str, dict[str, object]], report["branch_minus_control_deltas"])
    summaries = _summaries(report)
    for branch in ("partner_policy_drift", "physical_reward_law_drift"):
        assert deltas[branch] == lane._summary_delta(
            summaries[branch],
            summaries["control"],
        )


def test_work_state_trajectory_and_scaling_receipts_are_exact(
    report: dict[str, object],
) -> None:
    resource = cast(dict[str, object], report["resource"])
    work = cast(dict[str, object], report["work"])
    assert resource["behavior_count_cells"] == 4
    assert resource["conditional_count_cells"] == 4
    assert resource["conditional_reward_one_cells"] == 4
    assert resource["conditional_physical_one_cells"] == 4
    assert resource["retrospective_inverse_count_cells"] == 8
    assert resource["persistent_integer_scalars"] == 25
    assert resource["logical_preallocated_state_nbytes"] == 200
    assert resource["state_size_fixed"] is True
    assert resource["raw_evaluation_events_retained"] == 48
    assert resource["randomness_calls"] == 0
    assert resource["replay_capacity"] == 0
    assert resource["persistent_state_scaling"] == "O(C*P + U*P + U*Z*P)"
    assert resource["per_event_decision_scaling"] == "O(arms*U*P)"
    assert work["prefix_source_events_consumed"] == 32
    assert work["evaluation_source_events_consumed"] == 48
    assert work["total_source_events_consumed"] == 80
    assert work["evaluation_model_updates"] == 0
    assert work["complete_conditional_cells_frozen"] == 192
    assert work["causal_preaction_arm_decisions"] == 144
    assert work["post_reveal_comparator_decisions"] == 96
    assert work["own_action_scores_per_arm_per_event"] == 2
    assert work["equal_own_action_score_count_across_arms"] is True
    assert work["learned_model_arms_per_event"] == 4
    assert work["learned_joint_cells_scored_per_model_arm"] == 4
    assert work["learned_joint_cell_score_evaluations"] == 768
    assert work["evaluator_true_law_cells_scored_per_event"] == 2
    assert work["own_actions_counterfactually_evaluated"] == 96
    assert work["retrospective_inverse_distributions"] == 96
    unhashed_work = dict(work)
    work_sha = unhashed_work.pop("work_contract_sha256")
    assert work_sha == lane._sha256(unhashed_work)

    total_raw_bytes = 0
    for branch in _branches(report).values():
        raw = cast(list[dict[str, object]], branch["raw_events"])
        assert branch["raw_trajectory_sha256"] == lane._sha256(raw)
        assert branch["raw_trajectory_canonical_nbytes"] == lane._canonical_nbytes(raw)
        total_raw_bytes += branch["raw_trajectory_canonical_nbytes"]
    assert resource["raw_trajectory_canonical_nbytes"] == total_raw_bytes
    assert len(cast(str, report["source_manifest_sha256"])) == 64
    assert len(cast(str, report["trajectory_manifest_sha256"])) == 64
    assert resource["final_report_canonical_nbytes"] == lane._canonical_nbytes(report)


def test_report_is_deterministic_strict_reconstructable_and_nonpromoting(
    report: dict[str, object],
) -> None:
    rerun = run_factorized_preaction_decision_utility_development(
        FactorizedPreactionDecisionUtilityConfig(
            prefix_steps=32,
            evaluation_steps=16,
        )
    )
    assert rerun == report
    assert validate_factorized_preaction_decision_utility_report(report) == ()
    assert report["schema"] == DEVELOPMENT_SCHEMA
    assert report["development_only"] is DEVELOPMENT_ONLY is True
    assert report["assessment_status"] == ASSESSMENT_STATUS == "not_assessed"
    assert report["evidence_level"] == EVIDENCE_LEVEL == "L0"
    assert report["scientific_promotion_allowed"] is SCIENTIFIC_PROMOTION_ALLOWED is False
    assert report["benchmark_execution_authority"] is BENCHMARK_EXECUTION_AUTHORITY is False
    assert report["artifact_authority"] is ARTIFACT_AUTHORITY is False
    assert report["output_writes_allowed"] is OUTPUT_WRITES_ALLOWED is False
    assert report["evidence_claimed"] is EVIDENCE_CLAIMED is False
    assert report["thresholds_defined"] is THRESHOLDS_DEFINED is False
    assert report["task_identifiers_exposed"] is False
    assert report["descriptive_claims_only"] is True
    assert tuple(
        inspect.signature(run_factorized_preaction_decision_utility_development).parameters
    ) == ("config",)


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("task_identifiers_exposed",), 0),
        (("thresholds_defined",), 0),
        (("resource", "randomness_calls"), False),
        (("resource", "logical_preallocated_state_nbytes"), 200.0),
        (("branch_summaries", "control", "behavior", "mean_nll"), -0.0),
    ],
)
def test_resealed_boolean_numeric_and_signed_zero_tampering_is_rejected(
    report: dict[str, object],
    path: tuple[str, ...],
    replacement: object,
) -> None:
    tampered = copy.deepcopy(report)
    tampered.pop("integrity")
    cursor = tampered
    for key in path[:-1]:
        cursor = cast(dict[str, object], cursor[key])
    cursor[path[-1]] = replacement
    resealed = lane._seal_report(tampered)
    errors = validate_factorized_preaction_decision_utility_report(resealed)
    assert "report does not reconstruct with exact canonical types and bytes" in errors


def test_resealed_state_source_and_trajectory_hash_tampering_is_rejected(
    report: dict[str, object],
) -> None:
    paths: tuple[tuple[str | int, ...], ...] = (
        ("source_contract_sha256",),
        ("states", "after_common_prefix_frozen", "content_sha256"),
        ("branches", 0, "raw_trajectory_sha256"),
    )
    for path in paths:
        tampered = copy.deepcopy(report)
        tampered.pop("integrity")
        cursor: object = tampered
        for key in path[:-1]:
            if type(key) is int:
                cursor = cast(list[object], cursor)[key]
            else:
                cursor = cast(dict[str, object], cursor)[cast(str, key)]
        final_key = path[-1]
        assert type(final_key) is str
        cast(dict[str, object], cursor)[final_key] = "f" * 64
        resealed = lane._seal_report(tampered)
        assert validate_factorized_preaction_decision_utility_report(resealed)


def test_runner_rejects_falsey_or_nonexact_config_substitutes() -> None:
    class FalseySubstitute:
        def __bool__(self) -> bool:
            return False

    malformed_values: tuple[object, ...] = (False, 0, {}, FalseySubstitute())
    for malformed in malformed_values:
        with pytest.raises(
            TypeError,
            match="exact FactorizedPreactionDecisionUtilityConfig",
        ):
            run_factorized_preaction_decision_utility_development(
                malformed  # type: ignore[arg-type]
            )


def test_raw_trajectory_and_report_size_bounds_are_enforced() -> None:
    with pytest.raises(ValueError, match="raw trajectories exceed"):
        run_factorized_preaction_decision_utility_development(
            FactorizedPreactionDecisionUtilityConfig(max_raw_trajectory_bytes=1)
        )
    with pytest.raises(ValueError, match="report exceeds"):
        run_factorized_preaction_decision_utility_development(
            FactorizedPreactionDecisionUtilityConfig(max_report_bytes=1)
        )
