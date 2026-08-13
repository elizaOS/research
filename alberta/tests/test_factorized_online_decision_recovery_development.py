"""Contracts for factorized online decision recovery with matched routing arms."""

from __future__ import annotations

import copy
import dataclasses
import inspect
import math
from types import MappingProxyType
from typing import cast

import pytest

from alberta_framework.evaluation import factorized_online_decision_recovery_development as lane
from alberta_framework.evaluation.factorized_online_decision_recovery_development import (
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
    SCIENTIFIC_PROMOTION_ALLOWED,
    THRESHOLDS_DEFINED,
    FactorizedOnlineDecisionRecoveryConfig,
    run_factorized_online_decision_recovery_development,
    validate_factorized_online_decision_recovery_report,
)

pytestmark = [pytest.mark.unit, pytest.mark.development]


@pytest.fixture(scope="module")
def report() -> dict[str, object]:
    return run_factorized_online_decision_recovery_development(
        FactorizedOnlineDecisionRecoveryConfig(
            prefix_steps=16,
            evaluation_steps=32,
            summary_window=16,
        )
    )


def _branches(report: dict[str, object]) -> dict[str, dict[str, object]]:
    return {
        cast(str, branch["branch"]): branch
        for branch in cast(list[dict[str, object]], report["branches"])
    }


def _arms(branch: dict[str, object]) -> dict[str, dict[str, object]]:
    return {
        cast(str, arm["arm"]): arm
        for arm in cast(list[dict[str, object]], branch["arms"])
    }


def _summaries(report: dict[str, object]) -> dict[str, dict[str, dict[str, object]]]:
    return cast(
        dict[str, dict[str, dict[str, object]]],
        report["branch_summaries"],
    )


@pytest.mark.parametrize(
    ("field", "value", "exception"),
    [
        ("run_id", "unversioned", ValueError),
        ("prefix_steps", True, TypeError),
        ("prefix_steps", 15, ValueError),
        ("prefix_steps", 17, ValueError),
        ("evaluation_steps", 8, ValueError),
        ("evaluation_steps", 17, ValueError),
        ("summary_window", 8, ValueError),
        ("summary_window", 17, ValueError),
        ("summary_window", 112, ValueError),
        ("pseudocount", 1, TypeError),
        ("pseudocount", float("nan"), ValueError),
        ("pseudocount", 1_000_001.0, ValueError),
        ("prefix_steps", 65_552, ValueError),
        ("max_total_arm_events", 2, ValueError),
        ("max_total_arm_events", 65_537, ValueError),
        ("max_logical_state_bytes", 2, ValueError),
        ("max_logical_state_bytes", 1_048_577, ValueError),
        ("max_raw_trajectory_bytes", False, TypeError),
        ("max_raw_trajectory_bytes", 67_108_865, ValueError),
        ("max_report_bytes", 0, ValueError),
        ("max_report_bytes", 100_663_297, ValueError),
    ],
)
def test_config_is_exact_bounded_and_roundtrips(
    field: str,
    value: object,
    exception: type[Exception],
) -> None:
    with pytest.raises(exception):
        FactorizedOnlineDecisionRecoveryConfig(**{field: value})  # type: ignore[arg-type]

    config = FactorizedOnlineDecisionRecoveryConfig()
    assert FactorizedOnlineDecisionRecoveryConfig.from_config(config.to_config()) == config
    with pytest.raises(TypeError, match="exact JSON object"):
        FactorizedOnlineDecisionRecoveryConfig.from_config(
            MappingProxyType(config.to_config())
        )
    malformed = config.to_config()
    malformed["extra"] = 1
    with pytest.raises(ValueError, match="fields differ"):
        FactorizedOnlineDecisionRecoveryConfig.from_config(malformed)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("prefix_steps", 64.0),
        ("evaluation_steps", 96.0),
        ("summary_window", True),
        ("pseudocount", 1),
        ("max_report_bytes", 12_582_912.0),
    ],
)
def test_config_rejects_boolean_and_numeric_aliases(
    field: str,
    replacement: object,
) -> None:
    payload = FactorizedOnlineDecisionRecoveryConfig().to_config()
    payload[field] = replacement
    with pytest.raises((TypeError, ValueError), match="invalid config payload"):
        FactorizedOnlineDecisionRecoveryConfig.from_config(payload)


def test_common_prefix_matches_the_exhaustive_factorized_construction(
    report: dict[str, object],
) -> None:
    support = cast(dict[str, object], report["common_prefix_support"])
    assert support["own_partner_counts"] == [4, 4, 4, 4]
    assert support["complete"] is True
    prefix = cast(dict[str, object], report["common_prefix"])
    assert prefix["source_event_count"] == 16
    assert prefix["passes_over_source"] == 1
    assert prefix["behavior_updates"] == 16
    assert prefix["world_updates"] == 16
    assert prefix["source_events_retained_in_report"] == 0


def test_runner_reconstructs_config_before_any_work() -> None:
    config = FactorizedOnlineDecisionRecoveryConfig()
    object.__setattr__(config, "pseudocount", 1)

    with pytest.raises(TypeError, match="invalid config payload"):
        run_factorized_online_decision_recovery_development(config)


def test_preaction_api_has_no_branch_action_label_or_outcome_surface() -> None:
    assert tuple(inspect.signature(lane._freeze_preaction_decision).parameters) == (
        "state",
        "cue",
        "pseudocount",
    )
    fields = {field.name for field in dataclasses.fields(lane._FrozenPreactionDecision)}
    assert "partner_action" not in fields
    assert "reward" not in fields
    assert "post_physical_bit" not in fields
    frozen = lane._freeze_preaction_decision(lane._initial_state(), 0, 1.0)
    assert frozen.partner_action_revealed is False
    assert frozen.outcome_revealed is False
    assert tuple(inspect.signature(lane._exploration_instruction).parameters) == ("step",)


def test_periodic_exploration_is_branch_independent_and_guarantees_joint_support() -> None:
    expected_offsets = [0, 1, 2, 0]
    expected_actions = [0, 0, 1, 1]
    scheduled_steps = []
    for step in range(16):
        receipt = lane._exploration_instruction(step)
        assert receipt["depends_on_branch"] is False
        assert receipt["depends_on_arm"] is False
        assert receipt["depends_on_prediction"] is False
        assert receipt["randomness_calls"] == 0
        if receipt["scheduled"]:
            scheduled_steps.append(step)
    assert scheduled_steps == [4 * block + expected_offsets[block] for block in range(4)]
    assert [
        lane._exploration_instruction(step)["forced_action"] for step in scheduled_steps
    ] == expected_actions

    for policy_drift in (False, True):
        support = [0, 0, 0, 0]
        for step in scheduled_steps:
            receipt = lane._exploration_instruction(step)
            own_action = cast(int, receipt["forced_action"])
            baseline = lane._baseline_partner_action(step)
            partner_action = 1 - baseline if policy_drift else baseline
            support[lane._world_index(own_action, partner_action)] += 1
        assert support == [1, 1, 1, 1]


def test_all_arms_start_from_one_exact_prefix_copy_in_every_branch(
    report: dict[str, object],
) -> None:
    prefix_state = cast(
        dict[str, object],
        cast(dict[str, object], report["states"])["after_common_prefix"],
    )
    prefix_sha = prefix_state["content_sha256"]
    for branch in _branches(report).values():
        assert branch["common_prefix_state_sha256"] == prefix_sha
        arms = _arms(branch)
        assert tuple(arms) == ARM_NAMES
        assert {arm["initial_state_sha256"] for arm in arms.values()} == {prefix_sha}


def test_routing_masks_are_the_only_candidate_application_difference(
    report: dict[str, object],
) -> None:
    expected = {
        "both_adaptive": (True, True),
        "behavior_frozen_world_adaptive": (False, True),
        "behavior_adaptive_world_frozen": (True, False),
        "both_frozen": (False, False),
    }
    routing = cast(dict[str, dict[str, object]], report["routing_contract"])
    for routing_arm, (behavior, world) in expected.items():
        assert routing[routing_arm]["apply_behavior"] is behavior
        assert routing[routing_arm]["apply_world"] is world
        assert routing[routing_arm]["behavior_candidate_computed_every_event"] is True
        assert routing[routing_arm]["world_candidate_computed_every_event"] is True
    for branch in _branches(report).values():
        for arm_name, arm in _arms(branch).items():
            behavior, world = expected[arm_name]
            assert arm["candidate_counts"] == {"behavior": 32, "world": 32}
            assert arm["applied_update_counts"] == {
                "behavior": 32 if behavior else 0,
                "world": 32 if world else 0,
            }
            for event in cast(list[dict[str, object]], arm["raw_events"]):
                behavior_receipt = cast(dict[str, object], event["behavior_candidate"])
                world_receipt = cast(dict[str, object], event["world_candidate"])
                assert behavior_receipt["computed"] is True
                assert world_receipt["computed"] is True
                assert behavior_receipt["applied"] is behavior
                assert world_receipt["applied"] is world
                assert len(cast(str, behavior_receipt["content_sha256"])) == 64
                assert len(cast(str, world_receipt["content_sha256"])) == 64

    runner_source = inspect.getsource(lane._run_arm)
    first_mask_read = runner_source.index("_ROUTING_MASKS[arm]")
    assert runner_source.index("_compute_behavior_candidate") < first_mask_read
    assert runner_source.index("_compute_world_candidate") < first_mask_read


def test_exploration_replaces_only_execution_and_retains_both_utilities(
    report: dict[str, object],
) -> None:
    contract = cast(dict[str, object], report["exploration_contract"])
    assert contract["one_scheduled_event_per_four_event_block"] is True
    assert contract["replacement_semantics"] == (
        "forced action replaces proposal only for execution"
    )
    assert contract["proposed_counterfactual_utility_retained"] is True
    assert contract["executed_utility_retained"] is True
    assert contract["joint_candidate_support_per_cycle_under_each_partner_mapping"] is True
    for branch in _branches(report).values():
        for arm in _arms(branch).values():
            scheduled_count = 0
            for event in cast(list[dict[str, object]], arm["raw_events"]):
                exploration = cast(dict[str, object], event["exploration"])
                outcomes = cast(
                    list[dict[str, object]],
                    event["counterfactual_outcomes_by_own_action"],
                )
                proposed = cast(int, exploration["proposed_action"])
                executed = cast(int, exploration["executed_action"])
                assert event["proposed_counterfactual_reward"] == outcomes[proposed]["reward"]
                assert event["executed_reward"] == outcomes[executed]["reward"]
                if exploration["scheduled"]:
                    scheduled_count += 1
                    assert executed == exploration["forced_action"]
                else:
                    assert executed == proposed
                    assert exploration["forced_action"] is None
                    assert exploration["forced_action_replaced_proposal"] is False
            assert scheduled_count == 8


def test_exploration_support_counts_are_exact_and_separate_from_application(
    report: dict[str, object],
) -> None:
    for branch in _branches(report).values():
        for arm_name, arm in _arms(branch).items():
            assert arm["exploration_world_candidate_support_counts"] == [2, 2, 2, 2]
            counts = cast(list[int], arm["world_candidate_support_counts"])
            assert all(count >= 2 for count in counts)
            if arm_name in ("both_adaptive", "behavior_frozen_world_adaptive"):
                assert arm["exploration_world_applied_support_counts"] == [2, 2, 2, 2]
                assert arm["world_applied_support_counts"] == counts
            else:
                assert arm["exploration_world_applied_support_counts"] == [0, 0, 0, 0]
                assert arm["world_applied_support_counts"] == [0, 0, 0, 0]


def test_branch_interventions_are_matched_and_evaluator_only(
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
    assert report["task_identifiers_exposed"] is False


def test_behavior_routing_is_the_only_policy_drift_recovery_factor(
    report: dict[str, object],
) -> None:
    summaries = _summaries(report)["partner_policy_drift"]
    both = summaries["both_adaptive"]
    behavior_only = summaries["behavior_adaptive_world_frozen"]
    world_only = summaries["behavior_frozen_world_adaptive"]
    frozen = summaries["both_frozen"]
    assert both["raw_curves"]  # exact raw curves are retained, not only endpoints
    assert cast(dict[str, object], both["early_window"])["behavior_nll"] == cast(
        dict[str, object],
        behavior_only["early_window"],
    )["behavior_nll"]
    assert cast(dict[str, object], both["late_window"])["behavior_nll"] == cast(
        dict[str, object],
        behavior_only["late_window"],
    )["behavior_nll"]
    assert cast(dict[str, object], world_only["early_window"])["behavior_nll"] == cast(
        dict[str, object],
        world_only["late_window"],
    )["behavior_nll"]
    assert cast(dict[str, object], frozen["early_window"])["behavior_nll"] == cast(
        dict[str, object],
        frozen["late_window"],
    )["behavior_nll"]
    assert cast(dict[str, float], both["late_window"])["behavior_nll"] < cast(
        dict[str, float],
        both["early_window"],
    )["behavior_nll"]
    assert cast(dict[str, float], behavior_only["late_window"])[
        "behavior_nll"
    ] < cast(dict[str, float], behavior_only["early_window"])["behavior_nll"]
    assert cast(dict[str, float], both["exploit_only"])["executed_reward"] == cast(
        dict[str, float],
        behavior_only["exploit_only"],
    )["executed_reward"]
    assert cast(dict[str, float], both["exploit_only"])["executed_reward"] > cast(
        dict[str, float],
        world_only["exploit_only"],
    )["executed_reward"]


def test_world_routing_is_the_only_law_drift_recovery_factor(
    report: dict[str, object],
) -> None:
    summaries = _summaries(report)["physical_reward_law_drift"]
    both = summaries["both_adaptive"]
    world_only = summaries["behavior_frozen_world_adaptive"]
    behavior_only = summaries["behavior_adaptive_world_frozen"]
    frozen = summaries["both_frozen"]
    for metric in ("complete_table_reward_mse", "complete_table_physical_mse"):
        assert cast(dict[str, float], both["early_window"])[metric] == cast(
            dict[str, float],
            world_only["early_window"],
        )[metric]
        assert cast(dict[str, float], both["late_window"])[metric] == cast(
            dict[str, float],
            world_only["late_window"],
        )[metric]
        assert cast(dict[str, float], both["late_window"])[metric] < cast(
            dict[str, float],
            both["early_window"],
        )[metric]
        assert cast(dict[str, float], behavior_only["early_window"])[metric] == cast(
            dict[str, float],
            behavior_only["late_window"],
        )[metric]
        assert cast(dict[str, float], frozen["early_window"])[metric] == cast(
            dict[str, float],
            frozen["late_window"],
        )[metric]
    assert cast(dict[str, float], both["exploit_only"])["executed_reward"] == cast(
        dict[str, float],
        world_only["exploit_only"],
    )["executed_reward"]
    assert cast(dict[str, float], both["exploit_only"])["executed_reward"] > cast(
        dict[str, float],
        behavior_only["exploit_only"],
    )["executed_reward"]


def test_exploration_and_exploit_utility_are_separate_raw_summaries(
    report: dict[str, object],
) -> None:
    for branch in BRANCH_NAMES:
        for arm in ARM_NAMES:
            summary = _summaries(report)[branch][arm]
            assert summary["event_count"] == 32
            assert summary["exploration_event_count"] == 8
            assert summary["exploit_event_count"] == 24
            assert len(cast(list[int], summary["proposed_actions"])) == 32
            assert len(cast(list[int], summary["executed_actions"])) == 32
            for scope in (
                "full",
                "exploration_only",
                "exploit_only",
                "early_window",
                "late_window",
                "late_minus_early",
            ):
                metrics = cast(dict[str, float], summary[scope])
                assert all(math.isfinite(value) for value in metrics.values())
            curves = cast(dict[str, list[float | int]], summary["raw_curves"])
            assert all(len(curve) == 32 for curve in curves.values())


def test_branch_minus_control_summaries_recompute_exactly(
    report: dict[str, object],
) -> None:
    summaries = _summaries(report)
    deltas = cast(
        dict[str, dict[str, dict[str, object]]],
        report["branch_minus_control_deltas"],
    )
    for branch in ("partner_policy_drift", "physical_reward_law_drift"):
        for arm in ARM_NAMES:
            assert deltas[branch][arm] == lane._summary_delta(
                summaries[branch][arm],
                summaries["control"][arm],
            )


def test_work_state_hash_and_resource_receipts_are_exact(
    report: dict[str, object],
) -> None:
    work = cast(dict[str, object], report["work"])
    resource = cast(dict[str, object], report["resource"])
    assert work["prefix_events"] == 16
    assert work["arm_events"] == 384
    assert work["total_events_evaluated"] == 400
    assert work["arm_preaction_decisions"] == 384
    assert work["arm_complete_world_cells_frozen"] == 1536
    assert work["prefix_behavior_candidates_computed"] == 16
    assert work["prefix_world_candidates_computed"] == 16
    assert work["arm_behavior_candidates_computed"] == 384
    assert work["arm_world_candidates_computed"] == 384
    assert work["total_behavior_candidates_computed"] == 400
    assert work["total_world_candidates_computed"] == 400
    assert work["prefix_behavior_candidates_applied"] == 16
    assert work["prefix_world_candidates_applied"] == 16
    assert work["arm_behavior_candidates_applied"] == 192
    assert work["arm_world_candidates_applied"] == 192
    assert work["total_behavior_candidates_applied"] == 208
    assert work["total_world_candidates_applied"] == 208
    assert work["counterfactual_own_actions_scored"] == 768
    assert work["scheduled_exploration_events"] == 96
    unhashed_work = dict(work)
    work_sha = unhashed_work.pop("work_contract_sha256")
    assert work_sha == lane._sha256(unhashed_work)
    assert resource["behavior_count_cells"] == 4
    assert resource["conditional_count_cells"] == 4
    assert resource["reward_one_count_cells"] == 4
    assert resource["physical_one_count_cells"] == 4
    assert resource["persistent_integer_scalars_per_arm"] == 16
    assert resource["logical_state_nbytes_per_arm"] == 128
    assert resource["counterfactual_arm_state_copies"] == 12
    assert resource["raw_arm_events_retained"] == 384
    assert (
        resource["raw_report_memory_scaling"]
        == "O(branches*arms*evaluation_steps*U*P)"
    )
    assert resource["randomness_calls"] == 0
    assert resource["replay_capacity"] == 0

    total_raw_bytes = 0
    for branch in _branches(report).values():
        for arm in _arms(branch).values():
            raw = cast(list[dict[str, object]], arm["raw_events"])
            assert arm["raw_trajectory_sha256"] == lane._sha256(raw)
            assert arm["raw_trajectory_canonical_nbytes"] == lane._canonical_nbytes(raw)
            total_raw_bytes += arm["raw_trajectory_canonical_nbytes"]
    assert resource["raw_trajectory_canonical_nbytes"] == total_raw_bytes
    assert len(cast(str, report["source_manifest_sha256"])) == 64
    assert len(cast(str, report["trajectory_manifest_sha256"])) == 64
    assert resource["final_report_canonical_nbytes"] == lane._canonical_nbytes(report)


def test_report_is_strict_deterministic_reconstructable_and_nonpromoting(
    report: dict[str, object],
) -> None:
    rerun = run_factorized_online_decision_recovery_development(
        FactorizedOnlineDecisionRecoveryConfig(
            prefix_steps=16,
            evaluation_steps=32,
            summary_window=16,
        )
    )
    assert rerun == report
    assert validate_factorized_online_decision_recovery_report(report) == ()
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
    assert report["descriptive_claims_only"] is True
    assert tuple(
        inspect.signature(run_factorized_online_decision_recovery_development).parameters
    ) == ("config",)


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("task_identifiers_exposed",), 0),
        (("thresholds_defined",), 0),
        (("resource", "randomness_calls"), False),
        (("resource", "logical_state_nbytes_per_arm"), 128.0),
        (
            (
                "branch_summaries",
                "control",
                "both_frozen",
                "full",
                "behavior_nll",
            ),
            -0.0,
        ),
    ],
)
def test_resealed_type_and_signed_zero_tampering_is_rejected(
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
    errors = validate_factorized_online_decision_recovery_report(resealed)
    assert "report does not reconstruct with exact canonical types and bytes" in errors


@pytest.mark.parametrize(
    "path",
    (
        ("event_timing_contract",),
        ("branches", 0, "arms", 0, "raw_events"),
    ),
)
def test_original_integrity_cannot_hide_tuple_for_list_alias(
    report: dict[str, object],
    path: tuple[str | int, ...],
) -> None:
    tampered = copy.deepcopy(report)
    cursor: object = tampered
    for key in path[:-1]:
        if type(key) is int:
            cursor = cast(list[object], cursor)[key]
        else:
            cursor = cast(dict[str, object], cursor)[cast(str, key)]
    final_key = path[-1]
    assert type(final_key) is str
    original = cast(dict[str, object], cursor)[final_key]
    assert type(original) is list
    cast(dict[str, object], cursor)[final_key] = tuple(cast(list[object], original))

    assert tampered["integrity"] == report["integrity"]
    errors = validate_factorized_online_decision_recovery_report(tampered)
    assert "report does not reconstruct with exact canonical types and bytes" in errors


def test_resealed_state_source_and_trajectory_hash_tampering_is_rejected(
    report: dict[str, object],
) -> None:
    paths: tuple[tuple[str | int, ...], ...] = (
        ("source_contract_sha256",),
        ("states", "after_common_prefix", "content_sha256"),
        ("branches", 0, "arms", 0, "raw_trajectory_sha256"),
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
        assert validate_factorized_online_decision_recovery_report(resealed)


def test_runner_rejects_falsey_or_nonexact_config_substitutes() -> None:
    class FalseySubstitute:
        def __bool__(self) -> bool:
            return False

    malformed_values: tuple[object, ...] = (False, 0, {}, FalseySubstitute())
    for malformed in malformed_values:
        with pytest.raises(
            TypeError,
            match="exact FactorizedOnlineDecisionRecoveryConfig",
        ):
            run_factorized_online_decision_recovery_development(
                malformed  # type: ignore[arg-type]
            )


def test_raw_trajectory_and_report_bounds_are_enforced() -> None:
    with pytest.raises(ValueError, match="raw trajectories exceed"):
        run_factorized_online_decision_recovery_development(
            FactorizedOnlineDecisionRecoveryConfig(max_raw_trajectory_bytes=1)
        )
    with pytest.raises(ValueError, match="report exceeds"):
        run_factorized_online_decision_recovery_development(
            FactorizedOnlineDecisionRecoveryConfig(max_report_bytes=1)
        )
