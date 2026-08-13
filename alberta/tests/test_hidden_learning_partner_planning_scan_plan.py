"""Contracts for the nonexecuting hidden learning-partner matched scan plan."""

from __future__ import annotations

import dataclasses
from typing import Any, cast

import jax.random as jr
import numpy as np
import pytest

import alberta_framework.evaluation.hidden_learning_partner_planning_development as bridge_module
from alberta_framework.evaluation.hidden_learning_partner_planning_development import (
    BEHAVIOR_FROZEN,
    BENEFICIARY_FROZEN,
    BOTH_MODELS_FROZEN,
    BOTH_ROLES_FROZEN,
    GROUNDED_FROZEN,
    HELPER_FROZEN,
    JOINT_ADAPTIVE,
    MATCHED_CONDITIONS,
    SHUFFLED_DELIVERY,
    HiddenLearningPartnerPhaseDiagnostics,
    HiddenLearningPartnerPlanningConfig,
    HiddenLearningPartnerPlanningMetrics,
    HiddenLearningPartnerPlanningState,
    HiddenLearningPartnerPlanningTrace,
)
from alberta_framework.evaluation.hidden_learning_partner_planning_scan_plan import (
    CANONICAL_CONDITION_ORDER,
    DIAGNOSTIC_CONDITIONS,
    PAIRED_DEVELOPMENT_SEEDS,
    PRIMARY_CONDITIONS,
    HiddenLearningPartnerPlanningScanPlan,
    HiddenPlanningArm,
    HiddenPlanningContrast,
    HiddenPlanningExactChildClock,
    build_hidden_learning_partner_planning_scan_plan,
    require_valid_hidden_learning_partner_planning_scan_plan,
    validate_hidden_learning_partner_planning_scan_plan,
)

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def scan_plan() -> HiddenLearningPartnerPlanningScanPlan:
    return build_hidden_learning_partner_planning_scan_plan()


def _operation_map(arm: HiddenPlanningArm) -> dict[str, int]:
    return {operation.name: operation.per_run_total for operation in arm.named_operation_totals}


def _replace_arm(
    plan: HiddenLearningPartnerPlanningScanPlan,
    index: int,
    arm: HiddenPlanningArm,
) -> HiddenLearningPartnerPlanningScanPlan:
    arms = list(plan.arms)
    arms[index] = arm
    return dataclasses.replace(plan, arms=tuple(arms))


def test_build_is_canonical_nonexecuting_and_does_not_call_life_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_runner(*args: object, **kwargs: object) -> None:
        raise AssertionError("scan-plan construction must not execute a life")

    monkeypatch.setattr(
        bridge_module,
        "run_hidden_learning_partner_planning",
        forbidden_runner,
    )
    plan = build_hidden_learning_partner_planning_scan_plan()

    assert plan.schema == "alberta.hidden-learning-partner-planning.scan-plan.development.v1"
    assert plan.status == "RUNNER_AND_CRN_AUDIT_IMPLEMENTED_EXECUTION_PERMIT_REQUIRED"
    assert plan.config == HiddenLearningPartnerPlanningConfig()
    assert plan.bridge_schema == "alberta.hidden-learning-partner-planning.development.v1"
    assert plan.config.phase_length == 512
    assert plan.config.n_phases == 6
    assert plan.life_steps == 3_072
    assert not plan.execution_authorized
    assert not plan.runner_authorized
    assert not plan.campaign_authorized
    assert not plan.artifact_writes_authorized
    assert not plan.evidence_authorized
    assert not plan.scientific_promotion_allowed
    assert plan.thresholds is None
    assert plan.outcomes is None
    assert plan.artifact_output_path is None
    assert validate_hidden_learning_partner_planning_scan_plan(plan) == ()
    assert require_valid_hidden_learning_partner_planning_scan_plan(plan) is plan
    assert build_hidden_learning_partner_planning_scan_plan() == plan
    assert len(plan.plan_sha256) == 64


def test_arm_taxonomy_and_contrasts_cover_the_exact_bridge_surface(
    scan_plan: HiddenLearningPartnerPlanningScanPlan,
) -> None:
    assert CANONICAL_CONDITION_ORDER == tuple(MATCHED_CONDITIONS)
    assert tuple(arm.condition for arm in scan_plan.arms) == CANONICAL_CONDITION_ORDER
    assert tuple(arm.serialization_index for arm in scan_plan.arms) == tuple(range(11))
    assert tuple(arm.condition for arm in scan_plan.arms if arm.family == "primary") == (
        PRIMARY_CONDITIONS
    )
    assert tuple(arm.condition for arm in scan_plan.arms if arm.family == "diagnostic") == (
        DIAGNOSTIC_CONDITIONS
    )
    assert scan_plan.arms[0].condition == JOINT_ADAPTIVE
    assert scan_plan.arms[0].role == "reference"
    assert scan_plan.arms[0].contrast_id is None

    assert len(scan_plan.contrasts) == 10
    assert (
        tuple(contrast.intervention_condition for contrast in scan_plan.contrasts)
        == (CANONICAL_CONDITION_ORDER[1:])
    )
    primary = tuple(
        contrast for contrast in scan_plan.contrasts if contrast.family == "primary_causal"
    )
    diagnostics = tuple(
        contrast for contrast in scan_plan.contrasts if contrast.family == "diagnostic_only"
    )
    assert len(primary) == 7
    assert len(diagnostics) == 3
    assert all(contrast.reference_condition == JOINT_ADAPTIVE for contrast in scan_plan.contrasts)
    assert all(
        contrast.difference_direction == "intervention_minus_reference_per_paired_seed"
        for contrast in scan_plan.contrasts
    )
    assert {contrast.causal_claim_scope for contrast in primary} == {
        "finite_unexecuted_development_intervention_only"
    }
    assert {contrast.causal_claim_scope for contrast in diagnostics} == {
        "diagnostic_no_primary_causal_claim"
    }

    for dataclass_type in (
        HiddenLearningPartnerPlanningScanPlan,
        HiddenPlanningArm,
        HiddenPlanningContrast,
    ):
        names = tuple(field.name for field in dataclasses.fields(dataclass_type))
        assert len(names) == len(set(names))


def test_paired_seed_roots_named_key_ownership_and_order_independence_are_exact(
    scan_plan: HiddenLearningPartnerPlanningScanPlan,
) -> None:
    seeds = scan_plan.seed_contract
    assert tuple(binding.seed for binding in seeds.bindings) == PAIRED_DEVELOPMENT_SEEDS
    assert len(set(PAIRED_DEVELOPMENT_SEEDS)) == 4
    assert seeds.paired_across_every_condition
    assert seeds.development_only
    assert not seeds.held_out
    assert not seeds.evidence_eligible
    assert not seeds.executed
    stream_names = tuple(stream.name for stream in scan_plan.key_streams)
    assert stream_names == (
        "world.cue",
        "world.channel",
        "learner.helper",
        "learner.beneficiary",
        "behavior.initialization",
        "grounded.initialization",
        "planner",
        "intervention",
    )
    for index, binding in enumerate(seeds.bindings):
        assert binding.seed_index == index
        expected_root = np.asarray(jr.key_data(jr.key(binding.seed)), dtype=np.uint32)
        np.testing.assert_array_equal(np.asarray(binding.root_key_data), expected_root)
        assert tuple(name for name, _ in binding.named_key_data) == stream_names
        assert len({words for _, words in binding.named_key_data}) == len(stream_names)

    assert len({arm.seed_manifest_sha256 for arm in scan_plan.arms}) == 1
    assert scan_plan.arms[0].seed_manifest_sha256 == seeds.seed_manifest_sha256
    crn = scan_plan.common_random_numbers
    assert crn.same_seed_set_every_arm
    assert crn.same_root_key_for_seed_every_arm
    assert not crn.condition_is_key_derivation_input
    assert not crn.arm_order_is_key_derivation_input
    assert crn.fresh_state_per_seed_condition
    assert not crn.cross_arm_state_reuse_allowed
    assert crn.allowed_initial_state_difference_fields == ("config_token",)
    assert crn.required_equal_named_key_streams == stream_names
    assert crn.result_join_key == ("seed", "condition")
    assert crn.shuffled_channel_output_binding_required
    assert crn.cross_arm_rng_audit_implemented


def test_resource_transition_trace_and_named_operation_totals_are_exact(
    scan_plan: HiddenLearningPartnerPlanningScanPlan,
) -> None:
    counts = scan_plan.counts
    assert counts.primary_arm_count == 8
    assert counts.diagnostic_arm_count == 3
    assert counts.contrast_count == 10
    assert counts.paired_seed_count == 4
    assert counts.planned_run_count == 44
    assert counts.steps_per_run == 3_072
    assert counts.planned_transition_count == 44 * 3_072
    assert counts.initial_state_record_count == 44
    assert counts.final_state_record_count == 44
    assert counts.trace_row_count == counts.planned_transition_count
    assert counts.trace_fields_per_row == 58
    assert counts.metric_record_count == 44
    assert counts.scalar_metric_fields_per_record == 15
    assert counts.phase_diagnostic_container_count == 44
    assert counts.phase_rows_per_container == 6
    assert counts.phase_diagnostic_phase_row_count == 44 * 6
    assert counts.phase_diagnostic_field_count == 20
    assert counts.named_operation_accounting_scope == (
        "selected_bridge_runner_calls_static_write_masks_and_key_advances_"
        "not_flop_hlo_or_all_nested_primitives"
    )
    assert not counts.flop_or_hlo_equivalence_claimed
    assert counts.state_fields_per_snapshot == 9
    assert counts.persistent_state_bytes_per_run == 321
    assert counts.summed_logical_persistent_state_bytes == 44 * 321
    assert all(arm.resource_budget == scan_plan.resource_budget for arm in scan_plan.arms)
    assert scan_plan.resource_budget.signaling_state_nbytes == 80
    assert scan_plan.resource_budget.behavior_state_nbytes == 48
    assert scan_plan.resource_budget.grounded_state_nbytes == 108
    assert scan_plan.resource_budget.learner_model_state_nbytes == 236
    assert scan_plan.resource_budget.world_state_nbytes == 32
    assert scan_plan.resource_budget.metadata_state_nbytes == 53
    assert scan_plan.resource_budget.total_state_nbytes == 321
    assert scan_plan.resource_budget.replay_capacity == 0
    assert scan_plan.resource_budget.exact_tree_match

    for arm in scan_plan.arms:
        assert all(type(clock) is HiddenPlanningExactChildClock for clock in arm.exact_child_clocks)
        assert tuple(clock.name for clock in arm.exact_child_clocks) == (
            "behavior",
            "grounded",
        )
        behavior, grounded = arm.exact_child_clocks
        assert behavior.words_state_path == "behavior.step_words"
        assert behavior.telemetry_state_path == "behavior.step_count"
        assert grounded.words_state_path == "grounded.update_words"
        assert grounded.telemetry_state_path == "grounded.update_count"
        for clock in arm.exact_child_clocks:
            assert clock.words_dtype == "uint32"
            assert clock.words_shape == (2,)
            assert clock.initial_words == (0, 0)
            assert clock.initial_telemetry == 0
        behavior_writes = 3_072 if arm.condition_spec.behavior_write else 0
        grounded_writes = 3_072 if arm.condition_spec.grounded_write else 0
        assert behavior.final_words == (0, behavior_writes)
        assert behavior.final_telemetry == behavior_writes
        assert grounded.final_words == (0, grounded_writes)
        assert grounded.final_telemetry == grounded_writes

    operations = {arm.condition: _operation_map(arm) for arm in scan_plan.arms}
    for arm_operations in operations.values():
        assert arm_operations["bridge_step_calls"] == 3_072
        assert arm_operations["planner_consumption_gate_draws"] == 3_072
        assert arm_operations["behavior_update_proposal_opportunities"] == 3_072
        assert arm_operations["grounded_update_proposal_opportunities"] == 3_072
        assert arm_operations["helper_value_update_proposal_opportunities"] == 3_072
        assert arm_operations["beneficiary_value_update_proposal_opportunities"] == 3_072
    assert operations[SHUFFLED_DELIVERY]["shuffled_channel_output_bindings"] == 3_072
    assert all(
        operation["shuffled_channel_output_bindings"] == 0
        for condition, operation in operations.items()
        if condition != SHUFFLED_DELIVERY
    )

    assert operations[HELPER_FROZEN]["helper_value_committed_writes_on_required_valid_trace"] == 0
    assert (
        operations[HELPER_FROZEN]["beneficiary_value_committed_writes_on_required_valid_trace"]
        == 3_072
    )
    assert (
        operations[BENEFICIARY_FROZEN]["beneficiary_value_committed_writes_on_required_valid_trace"]
        == 0
    )
    assert (
        operations[BOTH_ROLES_FROZEN]["helper_value_committed_writes_on_required_valid_trace"] == 0
    )
    assert (
        operations[BOTH_ROLES_FROZEN]["beneficiary_value_committed_writes_on_required_valid_trace"]
        == 0
    )
    assert (
        operations[BEHAVIOR_FROZEN]["behavior_model_committed_writes_on_required_valid_trace"] == 0
    )
    assert (
        operations[GROUNDED_FROZEN]["grounded_model_committed_writes_on_required_valid_trace"] == 0
    )
    assert (
        operations[BOTH_MODELS_FROZEN]["behavior_model_committed_writes_on_required_valid_trace"]
        == 0
    )
    assert (
        operations[BOTH_MODELS_FROZEN]["grounded_model_committed_writes_on_required_valid_trace"]
        == 0
    )

    suite_operations = dict(counts.suite_named_operation_totals)
    assert suite_operations["bridge_step_calls"] == counts.planned_transition_count
    assert suite_operations["shuffled_channel_output_bindings"] == 4 * 3_072


def test_requested_outputs_include_exact_trace_metrics_and_phase_diagnostics(
    scan_plan: HiddenLearningPartnerPlanningScanPlan,
) -> None:
    requested = scan_plan.requested_outputs
    state_fields = tuple(
        field.name for field in dataclasses.fields(cast(Any, HiddenLearningPartnerPlanningState))
    )
    trace_fields = tuple(
        field.name for field in dataclasses.fields(cast(Any, HiddenLearningPartnerPlanningTrace))
    )
    metric_fields = tuple(
        field.name
        for field in dataclasses.fields(HiddenLearningPartnerPlanningMetrics)
        if field.name != "phase_diagnostics"
    )
    phase_fields = tuple(
        field.name for field in dataclasses.fields(HiddenLearningPartnerPhaseDiagnostics)
    )
    assert requested.initial_state_fields == state_fields
    assert requested.final_state_fields == state_fields
    assert requested.trace_fields == trace_fields
    assert requested.scalar_metric_fields == metric_fields
    assert requested.phase_diagnostic_fields == phase_fields
    assert "switch_cost" in requested.phase_diagnostic_fields
    assert "recurrence_savings" in requested.phase_diagnostic_fields
    assert requested.resource_budget_requested
    assert requested.strict_run_validation_errors_requested
    assert not requested.aggregate_statistics_requested
    assert not requested.outcomes_present
    assert not requested.thresholds_defined
    assert not requested.artifact_output_requested


def test_readiness_truthfully_reports_runner_audit_replay_and_live_permit_blockers(
    scan_plan: HiddenLearningPartnerPlanningScanPlan,
) -> None:
    readiness = scan_plan.readiness
    assert readiness.plan_validator_implemented
    assert readiness.canonical_bridge_conditions_bound
    assert readiness.paired_seed_and_key_contract_complete
    assert readiness.named_operation_accounting_complete_in_declared_scope
    assert not readiness.ready_for_runner_implementation
    assert readiness.suite_runner_implemented
    assert readiness.cross_arm_rng_audit_implemented
    assert readiness.execution_request_and_permit_implemented
    assert readiness.authenticated_source_replay_implemented
    assert not readiness.default_life_executed
    assert not readiness.outcomes_present
    assert readiness.quiescence_required
    assert not readiness.quiescence_checked
    assert not readiness.quiescence_verified
    assert not readiness.ready_for_execution
    assert readiness.blockers == (
        "exact_authenticated_execution_permit_not_issued",
        "external_host_quiescence_not_verified_live",
    )


def test_validator_fails_closed_on_duplicate_missing_unsupported_and_reordered_arms(
    scan_plan: HiddenLearningPartnerPlanningScanPlan,
) -> None:
    duplicate = dataclasses.replace(
        scan_plan,
        arms=(scan_plan.arms[0], scan_plan.arms[0], *scan_plan.arms[2:]),
    )
    duplicate_errors = validate_hidden_learning_partner_planning_scan_plan(duplicate)
    assert "scan plan contains duplicate condition arms" in duplicate_errors
    assert "scan plan is missing canonical bridge conditions" in duplicate_errors

    missing = dataclasses.replace(scan_plan, arms=scan_plan.arms[:-1])
    assert "scan plan is missing canonical bridge conditions" in (
        validate_hidden_learning_partner_planning_scan_plan(missing)
    )

    unsupported_arm = dataclasses.replace(scan_plan.arms[1], condition="unsupported")
    unsupported = _replace_arm(scan_plan, 1, unsupported_arm)
    unsupported_errors = validate_hidden_learning_partner_planning_scan_plan(unsupported)
    assert "scan plan contains unsupported bridge conditions" in unsupported_errors
    assert "scan plan is missing canonical bridge conditions" in unsupported_errors

    reordered = dataclasses.replace(scan_plan, arms=tuple(reversed(scan_plan.arms)))
    assert "arm order differs; evaluator order must remain serialization-only" in (
        validate_hidden_learning_partner_planning_scan_plan(reordered)
    )


def test_validator_fails_closed_on_resource_config_seed_and_order_key_mismatch(
    scan_plan: HiddenLearningPartnerPlanningScanPlan,
) -> None:
    alternate_config = HiddenLearningPartnerPlanningConfig(phase_length=256, n_phases=6)
    config_tamper = dataclasses.replace(
        scan_plan,
        config=alternate_config,
        life_steps=alternate_config.num_steps,
    )
    assert "config or exact life length differs from the canonical default" in (
        validate_hidden_learning_partner_planning_scan_plan(config_tamper)
    )

    bad_resource = dataclasses.replace(
        scan_plan.resource_budget,
        total_state_nbytes=scan_plan.resource_budget.total_state_nbytes + 4,
    )
    resource_arm = dataclasses.replace(scan_plan.arms[0], resource_budget=bad_resource)
    resource_tamper = _replace_arm(scan_plan, 0, resource_arm)
    assert "arm resource/config/seed pairing mismatch: joint_adaptive" in (
        validate_hidden_learning_partner_planning_scan_plan(resource_tamper)
    )

    first_seed = scan_plan.seed_contract.bindings[0]
    forged_seed = dataclasses.replace(first_seed, root_key_data=(1, first_seed.seed))
    forged_seed_contract = dataclasses.replace(
        scan_plan.seed_contract,
        bindings=(forged_seed, *scan_plan.seed_contract.bindings[1:]),
    )
    seed_tamper = dataclasses.replace(scan_plan, seed_contract=forged_seed_contract)
    assert "paired development seed or named key ownership contract differs" in (
        validate_hidden_learning_partner_planning_scan_plan(seed_tamper)
    )

    order_crn = dataclasses.replace(
        scan_plan.common_random_numbers,
        arm_order_is_key_derivation_input=True,
    )
    order_tamper = dataclasses.replace(scan_plan, common_random_numbers=order_crn)
    assert "common-random-number or evaluator-order requirements differ" in (
        validate_hidden_learning_partner_planning_scan_plan(order_tamper)
    )

    first_clock = scan_plan.arms[0].exact_child_clocks[0]
    bad_clock = dataclasses.replace(first_clock, final_words=(0, 0))
    bad_clock_arm = dataclasses.replace(
        scan_plan.arms[0],
        exact_child_clocks=(bad_clock, scan_plan.arms[0].exact_child_clocks[1]),
    )
    clock_tamper = _replace_arm(scan_plan, 0, bad_clock_arm)
    assert "arm exact child-clock binding differs: joint_adaptive" in (
        validate_hidden_learning_partner_planning_scan_plan(clock_tamper)
    )


def test_builder_fails_closed_on_live_source_schema_or_resource_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        bridge_module,
        "HIDDEN_LEARNING_PARTNER_PLANNING_SCHEMA",
        f"{bridge_module.HIDDEN_LEARNING_PARTNER_PLANNING_SCHEMA}.tampered",
    )
    with pytest.raises(RuntimeError, match="source kernel schema"):
        build_hidden_learning_partner_planning_scan_plan()

    monkeypatch.undo()
    monkeypatch.setattr(bridge_module, "_EXPECTED_BEHAVIOR_BYTES", 40)
    monkeypatch.setattr(bridge_module, "_EXPECTED_GROUNDED_BYTES", 100)
    monkeypatch.setattr(
        bridge_module,
        "_EXPECTED_TOTAL_BYTES",
        bridge_module._EXPECTED_TOTAL_BYTES - 16,
    )
    with pytest.raises(RuntimeError, match="source kernel persistent resources"):
        build_hidden_learning_partner_planning_scan_plan()


def test_validator_rejects_threshold_artifact_evidence_promotion_and_false_readiness(
    scan_plan: HiddenLearningPartnerPlanningScanPlan,
) -> None:
    authority = dataclasses.replace(
        scan_plan,
        execution_authorized=True,
        evidence_authorized=True,
        scientific_promotion_allowed=True,
    )
    assert "plan carries execution, artifact, evidence, or promotion authority" in (
        validate_hidden_learning_partner_planning_scan_plan(authority)
    )

    threshold = dataclasses.replace(
        scan_plan,
        thresholds=cast(Any, {"mean_reward": 0.9}),
    )
    assert "thresholds are forbidden in the development scan plan" in (
        validate_hidden_learning_partner_planning_scan_plan(threshold)
    )
    artifact = dataclasses.replace(
        scan_plan,
        artifact_output_path=cast(Any, "outputs/forbidden.json"),
    )
    assert "artifact output paths are forbidden in the scan plan" in (
        validate_hidden_learning_partner_planning_scan_plan(artifact)
    )
    outcomes = dataclasses.replace(
        scan_plan,
        outcomes=cast(Any, {"mean_reward": 1.0}),
    )
    assert "outcomes are forbidden in the nonexecuting scan plan" in (
        validate_hidden_learning_partner_planning_scan_plan(outcomes)
    )

    false_readiness = dataclasses.replace(
        scan_plan.readiness,
        default_life_executed=True,
        outcomes_present=True,
        quiescence_verified=True,
        ready_for_execution=True,
    )
    readiness_tamper = dataclasses.replace(scan_plan, readiness=false_readiness)
    assert "readiness falsely claims execution, quiescence, or outcomes" in (
        validate_hidden_learning_partner_planning_scan_plan(readiness_tamper)
    )
