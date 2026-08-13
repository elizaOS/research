"""Strict synthetic tests for the v3 descriptive report gate."""

from __future__ import annotations

import ast
import copy
from pathlib import Path

import pytest

from alberta_framework.evaluation import (
    _compositional_future_utility_v3_report_gate as gate,
)

MODULE_PATH = (
    Path(__file__).parents[1]
    / "alberta_framework/evaluation/_compositional_future_utility_v3_report_gate.py"
)

_LEARNER_SHA256 = (
    "5bca00ecc8a3c14dff9eb1afbd7af2e0d6cfc371e80fad21da4a5239af7548e7",
    "34d98992313753d1e810a22714cd22bf4199cfcdb9359eff1b4e887564ca1392",
    "590a9e5f757cffcc9ca8aac120a57b34ebf7ffce53f57b96974433f3e9c1778f",
    "f1ddcfde6a7d3ed6cf5f238afa95e1846bf2367315c112e5b9cc811d3590a269",
    "defe82edf61c6e7fbbd3f5dce7c4353738bfead2f5e13858245c9ecd393dc12e",
)
_TRACE_BITS = ("3f733333", "3f733333", "3f733333", "3f733333", "3f7fcc93")
_ENERGY_BITS = (1_101_004_788,) * 4 + (1_151_281_462,)
_NORMALIZATION = (
    "disabled-exact-zero",
    "disabled-exact-zero",
    "disabled-exact-zero",
    "enabled-bounded-endpoint",
    "enabled-bounded-endpoint",
)
_NONCLAIMS = (
    "per-step-contribution-transition-not-proven",
    "candidate-trace-transition-not-proven",
    "mixed-utility-equation-not-proven",
    "normalization-use-in-ranking-not-proven",
    "trace-reset-and-promotion-transfer-not-proven",
)
_RECURRENCES = (
    ("A", 2, 2, 1_584),
    ("A", 3, 4, 3_300),
    ("A", 4, 6, 5_144),
    ("B", 2, 7, 6_111),
    ("C", 2, 8, 7_110),
    ("A", 5, 9, 8_130),
)
_MUTATION_NAMES = (
    "decision_should_promote",
    "decision_should_refresh",
    "proposal_formed",
    "has_event",
    "promotion_applied",
    "root_change_applied",
    "root_change_mask",
    "cascade_refill_mask",
    "active_change_mask",
    "ordinary_candidate_refresh_mask",
    "post_promotion_candidate_refresh_mask",
    "candidate_refresh_mask",
    "candidate_rebound_mask",
    "candidate_overdepth_regeneration_mask",
)
_TOTAL_NAMES = (
    "proposal",
    "root_change",
    "promotion",
    "cascade_refill",
    "ordinary_candidate_refresh",
    "post_promotion_candidate_refresh",
    "candidate_refresh",
    "candidate_rebound",
    "candidate_overdepth_regeneration",
)


def _sha(character: str) -> str:
    return character * 64


def _expected_bindings() -> gate.ExpectedExecutionBindings:
    return gate.ExpectedExecutionBindings(
        execution_source_closure_sha256=_sha("a"),
        bootstrap_sha256=_sha("b"),
        ledger_primitive_sha256=_sha("c"),
        declared_loader_sha256=_sha("d"),
        genesis_sha256=_sha("e"),
        started_sha256=_sha("f"),
    )


def _reward_record(steps: int) -> dict[str, int]:
    reward_sum = steps % 2
    return {
        "steps": steps,
        "executed_reward_sum": reward_sum,
        "greedy_reward_sum": reward_sum,
        "executed_action_one_count": 0,
        "greedy_action_one_count": 0,
        "explored_count": 0,
    }


def _reward_counts() -> dict[str, object]:
    phases = [_reward_record(steps) for steps in gate.PHASE_LENGTHS]
    lifetime = {
        field: sum(phase[field] for phase in phases)
        for field in gate.REWARD_RECORD_FIELDS
    }
    return {
        "schema": gate.REWARD_COUNT_SCHEMA,
        "phase_order": list(gate.PHASE_ORDER),
        "lifetime": lifetime,
        "whole_phases": phases,
        "entry_windows": [_reward_record(gate.ENTRY_WINDOW) for _ in gate.PHASE_ORDER],
        "tail_windows": [_reward_record(gate.TAIL_WINDOW) for _ in gate.PHASE_ORDER],
        "experience_semantics_validated": True,
        "development_only": True,
        "execution_authorized": False,
        "output_writes_allowed": False,
        "evidence_authorized": False,
        "scientific_promotion_allowed": False,
    }


def _absent_rank(slot_field: str) -> dict[str, object]:
    return {
        "present": False,
        slot_field: [],
        "matching_score_f32_bits": [],
        "best_score_f32_bits": None,
        "descending_rank_interval": None,
    }


def _primary_endpoints() -> dict[str, object]:
    presence: list[dict[str, object]] = []
    rank_records: list[dict[str, object]] = []
    for target, occurrence, phase_index, post_step in _RECURRENCES:
        common = {
            "target": target,
            "occurrence": occurrence,
            "pre_recurrence_post_step": post_step,
            "active_present": False,
            "candidate_present": False,
            "active_slot_count": 0,
            "candidate_slot_count": 0,
        }
        presence.append(dict(common))
        rank_records.append(
            {
                "target": target,
                "occurrence": occurrence,
                "recurrence_phase_index": phase_index,
                "pre_recurrence_post_step": post_step,
                "active_present": False,
                "candidate_present": False,
                "active_slot_count": 0,
                "candidate_slot_count": 0,
                "matching_active_slots": [],
                "matching_candidate_slots": [],
                "direct_rank": _absent_rank("matching_composed_slots"),
                "ancestor_backed_rank": _absent_rank("matching_composed_slots"),
                "candidate_direct_rank": _absent_rank("matching_candidate_slots"),
                "candidate_augmented_rank": _absent_rank("matching_candidate_slots"),
            }
        )
    zero_partition = {
        "all_step_count": 0,
        "due_opportunity_count": 0,
        "off_opportunity_count": 0,
    }
    lifecycle = {
        target: {
            "direct_candidate_admission_count": 0,
            "admission_episode_count": 0,
            "loss_episode_count": 0,
            "present_at_end": False,
            "structural_reacquisition_count": 0,
        }
        for target in gate.TARGET_NAMES
    }
    cascade = {
        target: {
            "loss_episode_count": 0,
            "root_replacement_lost_slot_count": 0,
            "cascade_dependency_refill_lost_slot_count": 0,
            "all_changed_slots_accounted": True,
        }
        for target in gate.TARGET_NAMES
    }
    occupancy = {
        target: {
            "active_present_post_steps": 0,
            "active_presence_fraction": 0.0,
            "active_slot_step_cells": 0,
            "candidate_present_post_steps": 0,
            "candidate_presence_fraction": 0.0,
            "candidate_slot_step_cells": 0,
        }
        for target in gate.TARGET_NAMES
    }
    coexistence = {
        "target_order": list(gate.TARGET_NAMES),
        "steps": gate.TOTAL_STEPS,
        "steps_by_active_target_count": [gate.TOTAL_STEPS, 0, 0, 0],
        "maximum_active_target_count": 0,
        "all_targets_present_steps": 0,
        "all_targets_presence_fraction": 0.0,
        "first_all_targets_post_step": None,
        "last_all_targets_post_step": None,
        "active_targets_at_end": [],
    }
    cadence = {
        "diagnostic_partitions": {
            "decision_margin_passed": dict(zero_partition),
            "decision_candidate_margin_eligible": dict(zero_partition),
        },
        "mutation_partitions": {
            name: dict(zero_partition) for name in _MUTATION_NAMES
        },
        "all_mutations_off_opportunity_count": 0,
        "curation_counts_close": True,
        "curation_count_closure": {
            "all_checked_counts_close": True,
            "curation_due_count": gate.TOTAL_CURATION_OPPORTUNITIES,
            "mutation_counts": {name: 0 for name in _TOTAL_NAMES},
            "logical_event_count": 0,
            "event_bearing_opportunity_count": 0,
        },
        "eventwise_curation_closure": {
            "all_eventwise_curation_semantics_match": True,
            "promotion_event_count": 0,
            "ordinary_refresh_event_count": 0,
            "event_bearing_opportunity_count": 0,
        },
    }
    return {
        "endpoint_order": list(gate.PRIMARY_ENDPOINT_ORDER),
        "margin_passes": {
            "selected_strict_margin_pass_count": 0,
            "selected_strict_margin_all_step_diagnostic_count": 0,
            "selected_strict_margin_off_opportunity_diagnostic_count": 0,
            "candidate_destination_strict_margin_pair_count": 0,
            "candidate_destination_strict_margin_all_step_diagnostic_count": 0,
            "candidate_destination_strict_margin_off_opportunity_diagnostic_count": 0,
            "due_curation_event_count": gate.TOTAL_CURATION_OPPORTUNITIES,
        },
        "promotions": {"event_count": 0},
        "cascade_refill_slot_count": 0,
        "candidate_refreshes": {
            "decision_should_refresh_event_count": 0,
            "ordinary_refreshed_slot_count": 0,
            "post_promotion_refreshed_slot_count": 0,
            "total_refreshed_slot_count": 0,
        },
        "cascade_losses": cascade,
        "cascade_loss_definition": (
            "target-signature lost slots whose exact decision audit cause is "
            "cascade_dependency_refill"
        ),
        "target_admission_loss_end": lifecycle,
        "pre_recurrence_presence": presence,
        "target_retention": {
            "A": {
                "pre_recurrence_phase_indices": [2, 4, 6, 9],
                "pre_recurrence_presence": [False, False, False, False],
                "present_at_end": False,
            },
            "B": {
                "pre_recurrence_phase_indices": [7],
                "pre_recurrence_presence": [False],
                "present_at_end": False,
            },
            "C": {
                "pre_recurrence_phase_indices": [8],
                "pre_recurrence_presence": [False],
                "present_at_end": False,
            },
        },
        "target_occupancy": {
            "post_update_state_count": gate.TOTAL_STEPS,
            "per_target": occupancy,
            "coexistence": coexistence,
            "steps_by_distinct_active_target_count": [gate.TOTAL_STEPS, 0, 0, 0],
            "maximum_distinct_active_target_count": 0,
            "final_active_targets": [],
        },
        "pre_recurrence_ranks": {
            "active_definition": (
                "best matching target slot among composed slots RAW_DIM:ACTIVE_SLOTS; "
                "tie-aware descending rank interval, with rank 1 highest"
            ),
            "candidate_definition": (
                "best matching target slot among all candidate slots; direct and "
                "novelty-augmented scores each use a tie-aware descending rank interval, "
                "with rank 1 highest"
            ),
            "records": rank_records,
        },
        "cadence_integrity": cadence,
        "identity_reacquisition_claimed": False,
    }


def _execution_receipt(arm_index: int) -> dict[str, object]:
    return {
        "schema": gate.ARM_EXECUTION_RECEIPT_SCHEMA,
        "total_steps": gate.TOTAL_STEPS,
        "initial_state_sha256": _sha("1"),
        "final_state_sha256": _sha("34567"[arm_index]),
        "trace_sha256": _sha("89abc"[arm_index]),
        "expected_persistent_state_nbytes": gate.PERSISTENT_STATE_NBYTES,
        "initial_persistent_state_nbytes": gate.PERSISTENT_STATE_NBYTES,
        "final_persistent_state_nbytes": gate.PERSISTENT_STATE_NBYTES,
        "final_step_count": gate.TOTAL_STEPS,
        "final_step_words_uint32": [0, gate.TOTAL_STEPS],
        "final_replacement_phase": gate.FINAL_REPLACEMENT_PHASE,
        "initial_state_finite": True,
        "final_state_finite": True,
        "all_lifetime_counters_valid": True,
        "all_lifetime_capacity_available": True,
        "all_ranking_contracts_valid": True,
        "all_core_predictions_match_full_q": True,
        "initial_target_signature_counts_zero": True,
        "scientific_promotion_allowed": False,
        "evidence_authorized": False,
        "output_writes_allowed": False,
    }


def _state_receipt(arm_index: int) -> dict[str, object]:
    return {
        "schema": gate.STATE_GATE_SCHEMA,
        "steps": gate.TOTAL_STEPS,
        "trace_decay_f32_bits": _TRACE_BITS[arm_index],
        "expected_raw_energy_f32_bits": _ENERGY_BITS[arm_index],
        "normalization_moment_policy": _NORMALIZATION[arm_index],
        "field_manifest_sha256": (
            "834498ba4ed937d814590c2852d756164a80377124ae11fc15ad22ed17cfc9bd"
        ),
        "initial_subset_sha256": _sha("2"),
        "final_subset_sha256": _sha("89abc"[arm_index]),
        "initial_fields_all_zero": True,
        "all_fields_finite": True,
        "contribution_mode_zero_marginal_traces": True,
        "raw_slots_untouched_by_curation": True,
        "raw_energy_bits_exact": True,
        "normalization_moment_policy_exact": True,
        "utility_event_final_rows_exact": True,
        "nonclaims": list(_NONCLAIMS),
        "development_only": True,
        "panel_execution_authorized": False,
        "result_authorized": False,
        "output_writes_allowed": False,
        "evidence_authorized": False,
        "scientific_promotion_allowed": False,
    }


def _run(arm_index: int) -> dict[str, object]:
    body: dict[str, object] = {
        "arm": gate.ARM_ORDER[arm_index],
        "source_arm_name": gate.SOURCE_ARM_NAME,
        "learner_config_sha256": _LEARNER_SHA256[arm_index],
        "execution_receipt": _execution_receipt(arm_index),
        "state_gate_receipt": _state_receipt(arm_index),
        "primary_endpoints": _primary_endpoints(),
        "reward_counts": _reward_counts(),
    }
    return {**body, "arm_record_sha256": gate.canonical_json_sha256(body)}


def _report() -> dict[str, object]:
    bindings = {
        "development_root": 317_707_403,
        "development_root_hex": "0x12EFD48B",
        "protocol_config_sha256": (
            "09b7d06ae720f1a2aeb167ae10e4dbde46dff5437659e431bfff79a8445dc16c"
        ),
        "control_protocol_config_sha256": (
            "208afe0b0b91603e1da73f4b87116259a814d2332bdb107102b403e81ce667ca"
        ),
        "runtime_config_sha256": (
            "48f769d8b53c652b7f6ab251ca31be74ada978af53f9e8e15d04ea6b538720b6"
        ),
        "consumed_history_sha256": (
            "0c61ae4ae11e1e1b056cb481a0c652e37ba7119af9d8b6a5516856e0798c58e6"
        ),
        "key_manifest_sha256": (
            "ae8ad5a84b6d8f1449e90e71925184ffef46b74edf1a231948475fcf0fe11fd5"
        ),
        "stream_sha256": (
            "f8fdc3a73c06726686e1b285686219806401e2ff6179cb46ed14200d78bc3758"
        ),
        "cadence_bound_stream_sha256": (
            "ac4447b3c86c2f53acf3731d9e6a2d0b39a8e2552b3968748295700e6cbdebf1"
        ),
        "source_envelope_sha256": (
            "25d10d556df131be2822adb2879720b0624fc4af873458a285ee8a7bfd9e6e41"
        ),
        **_expected_bindings().to_config(),
    }
    runs = [_run(index) for index in range(len(gate.ARM_ORDER))]
    body: dict[str, object] = {
        "schema": gate.REPORT_SCHEMA,
        "status": gate.REPORT_STATUS,
        "bindings": bindings,
        "execution": {
            "attempt_index": 1,
            "attempts_authorized": 1,
            "attempts_consumed": 1,
            "root_consumed": True,
            "attempt_consumed_before_evaluator_import": True,
            "retry_or_recovery_authorized": False,
            "panel_completed": True,
            "arm_count": len(gate.ARM_ORDER),
        },
        "authority": {
            "development_only": True,
            "descriptive_result_available": True,
            "scientific_promotion_allowed": False,
            "evidence_authorized": False,
            "experiment_output_writes_allowed": False,
            "artifact_authorized": False,
            "threshold_defined_or_applied": False,
            "winner_or_default_selected": False,
            "search_or_tuning_performed": False,
            "retry_or_recovery_authorized": False,
        },
        "arm_order": list(gate.ARM_ORDER),
        "primary_endpoint_order": list(gate.PRIMARY_ENDPOINT_ORDER),
        "reward_metric_order": list(gate.REWARD_RECORD_FIELDS),
        "runs": runs,
        "cross_arm_contract": {
            "shared_initial_state_sha256": _sha("1"),
            "shared_initial_subset_sha256": _sha("2"),
            "shared_protocol_source_and_genesis": True,
            "shared_base_logical_work_matched": True,
            "stream_shapes_and_update_opportunities_matched": True,
            "persistent_shapes_and_bytes_matched": True,
            "intervention_specific_logical_work_matched": False,
            "total_named_logical_work_equivalence_claimed": False,
            "behavior_dependent_branch_work_equivalence_claimed": False,
            "behavioral_experience_matching_claimed": False,
            "compiled_flop_equivalence_claimed": False,
            "work_resource_contract_embedded": True,
            "work_resource_contract_sha256_bound": True,
        },
        "work_resource_contract": gate.work_resource_contract_config(),
        "work_resource_contract_sha256": gate.WORK_RESOURCE_CONTRACT_SHA256,
    }
    return {**body, "report_sha256": gate.canonical_json_sha256(body)}


def _reseal(report: dict[str, object], arm_index: int | None = None) -> None:
    if arm_index is not None:
        runs = report["runs"]
        assert type(runs) is list
        run = runs[arm_index]
        assert type(run) is dict
        run["arm_record_sha256"] = gate.canonical_json_sha256(
            {key: value for key, value in run.items() if key != "arm_record_sha256"}
        )
    report["report_sha256"] = gate.canonical_json_sha256(
        {key: value for key, value in report.items() if key != "report_sha256"}
    )


def _set_path(root: dict[str, object], path: tuple[str | int, ...], value: object) -> None:
    current: object = root
    for part in path[:-1]:
        if type(part) is int:
            assert type(current) is list
            current = current[part]
        else:
            assert type(current) is dict
            current = current[part]
    final = path[-1]
    if type(final) is int:
        assert type(current) is list
        current[final] = value
    else:
        assert type(current) is dict
        current[final] = value


def test_report_gate_is_pure_stdlib() -> None:
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    imported_roots = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    assert imported_roots <= {
        "__future__",
        "dataclasses",
        "hashlib",
        "json",
        "math",
        "struct",
        "typing",
    }
    assert gate.REPORT_SCHEMA.endswith(".v1")


def test_valid_zero_mutation_structural_report_closes_and_serializes() -> None:
    report = _report()

    gate.validate_v3_descriptive_report(report, _expected_bindings())
    serialized = gate.serialize_v3_descriptive_report(report, _expected_bindings())

    assert serialized == gate.canonical_json(report)
    assert '"winner_or_default_selected":false' in serialized
    assert gate.canonical_json_sha256(gate.work_resource_contract_config()) == (
        gate.WORK_RESOURCE_CONTRACT_SHA256
    )


def test_expected_execution_bindings_are_exact_and_lowercase() -> None:
    with pytest.raises(ValueError, match="lowercase"):
        gate.ExpectedExecutionBindings(
            execution_source_closure_sha256="A" * 64,
            bootstrap_sha256=_sha("b"),
            ledger_primitive_sha256=_sha("c"),
            declared_loader_sha256=_sha("d"),
            genesis_sha256=_sha("e"),
            started_sha256=_sha("f"),
        )
    with pytest.raises(TypeError, match="ExpectedExecutionBindings"):
        gate.validate_v3_descriptive_report(_report(), _expected_bindings().to_config())  # type: ignore[arg-type]


def test_gate_exports_no_writer_executor_or_selection_surface() -> None:
    public = set(gate.__all__)

    assert not any(
        token in name.lower()
        for name in public
        for token in ("write", "execute", "issue", "select", "winner", "threshold")
    )


@pytest.mark.parametrize(
    ("path", "replacement", "arm_index"),
    [
        pytest.param(
            ("bindings", "protocol_config_sha256"), _sha("0"), None, id="protocol-binding"
        ),
        pytest.param(
            ("bindings", "consumed_history_sha256"), _sha("0"), None, id="history-binding"
        ),
        pytest.param(
            ("bindings", "source_envelope_sha256"), _sha("0"), None, id="source-envelope"
        ),
        pytest.param(("bindings", "genesis_sha256"), _sha("0"), None, id="genesis-binding"),
        pytest.param(("execution", "attempt_index"), 2, None, id="attempt-index"),
        pytest.param(("execution", "attempt_index"), True, None, id="attempt-bool"),
        pytest.param(("execution", "attempts_consumed"), 0, None, id="attempt-consumption"),
        pytest.param(("execution", "panel_completed"), False, None, id="panel-completion"),
        pytest.param(("authority", "development_only"), False, None, id="development-only"),
        pytest.param(
            ("authority", "scientific_promotion_allowed"), True, None, id="promotion-authority"
        ),
        pytest.param(
            ("authority", "winner_or_default_selected"), True, None, id="winner-selection"
        ),
        pytest.param(("arm_order",), list(reversed(gate.ARM_ORDER)), None, id="arm-order"),
        pytest.param(
            ("primary_endpoint_order",),
            list(reversed(gate.PRIMARY_ENDPOINT_ORDER)),
            None,
            id="endpoint-order",
        ),
        pytest.param(
            ("reward_metric_order",),
            list(reversed(gate.REWARD_RECORD_FIELDS)),
            None,
            id="reward-order",
        ),
        pytest.param(("runs", 0, "arm"), gate.ARM_ORDER[1], 0, id="run-arm"),
        pytest.param(("runs", 0, "source_arm_name"), "other", 0, id="source-arm"),
        pytest.param(("runs", 0, "learner_config_sha256"), _sha("0"), 0, id="learner-hash"),
        pytest.param(
            ("runs", 0, "execution_receipt", "schema"), "other", 0, id="execution-schema"
        ),
        pytest.param(
            ("runs", 0, "execution_receipt", "total_steps"), 8_997, 0, id="execution-steps"
        ),
        pytest.param(
            ("runs", 0, "execution_receipt", "final_state_sha256"),
            _sha("1"),
            0,
            id="unadvanced-state",
        ),
        pytest.param(
            ("runs", 0, "execution_receipt", "trace_sha256"),
            "A" * 64,
            0,
            id="uppercase-trace-hash",
        ),
        pytest.param(
            ("runs", 0, "execution_receipt", "final_persistent_state_nbytes"),
            2_073,
            0,
            id="state-bytes",
        ),
        pytest.param(
            ("runs", 0, "execution_receipt", "final_step_words_uint32"),
            [0, 8_997],
            0,
            id="lifetime-words",
        ),
        pytest.param(
            ("runs", 0, "execution_receipt", "final_replacement_phase"),
            7,
            0,
            id="replacement-phase",
        ),
        pytest.param(
            ("runs", 0, "execution_receipt", "all_ranking_contracts_valid"),
            False,
            0,
            id="ranking-closure",
        ),
        pytest.param(
            ("runs", 0, "execution_receipt", "evidence_authorized"),
            True,
            0,
            id="execution-evidence",
        ),
        pytest.param(
            ("runs", 0, "state_gate_receipt", "trace_decay_f32_bits"),
            "3f7fcc93",
            0,
            id="state-decay-pin",
        ),
        pytest.param(
            ("runs", 0, "state_gate_receipt", "steps"),
            float(gate.TOTAL_STEPS),
            0,
            id="state-step-float",
        ),
        pytest.param(
            ("runs", 0, "state_gate_receipt", "expected_raw_energy_f32_bits"),
            1_151_281_462,
            0,
            id="state-energy-pin",
        ),
        pytest.param(
            ("runs", 0, "state_gate_receipt", "normalization_moment_policy"),
            "enabled-bounded-endpoint",
            0,
            id="normalization-pin",
        ),
        pytest.param(
            ("runs", 0, "state_gate_receipt", "initial_fields_all_zero"),
            False,
            0,
            id="state-genesis",
        ),
        pytest.param(
            ("runs", 0, "state_gate_receipt", "nonclaims"),
            list(reversed(_NONCLAIMS)),
            0,
            id="state-nonclaims",
        ),
        pytest.param(
            ("runs", 0, "reward_counts", "schema"), "other", 0, id="reward-schema"
        ),
        pytest.param(
            ("runs", 0, "reward_counts", "lifetime", "executed_reward_sum"),
            10,
            0,
            id="lifetime-phase-closure",
        ),
        pytest.param(
            ("runs", 0, "reward_counts", "whole_phases", 0, "executed_reward_sum"),
            0,
            0,
            id="reward-parity",
        ),
        pytest.param(
            ("runs", 0, "reward_counts", "whole_phases", 0, "greedy_reward_sum"),
            775,
            0,
            id="reward-range",
        ),
        pytest.param(
            ("runs", 0, "reward_counts", "whole_phases", 0, "explored_count"),
            774,
            0,
            id="reward-count-bound",
        ),
        pytest.param(
            ("runs", 0, "reward_counts", "entry_windows", 0, "steps"),
            63,
            0,
            id="entry-window",
        ),
        pytest.param(
            ("runs", 0, "reward_counts", "experience_semantics_validated"),
            False,
            0,
            id="experience-semantics",
        ),
        pytest.param(
            ("runs", 0, "primary_endpoints", "margin_passes", "due_curation_event_count"),
            280,
            0,
            id="margin-cadence",
        ),
        pytest.param(
            (
                "runs",
                0,
                "primary_endpoints",
                "margin_passes",
                "selected_strict_margin_pass_count",
            ),
            0.0,
            0,
            id="margin-float",
        ),
        pytest.param(
            ("runs", 0, "primary_endpoints", "cascade_refill_slot_count"),
            1,
            0,
            id="cascade-count",
        ),
        pytest.param(
            (
                "runs",
                0,
                "primary_endpoints",
                "target_admission_loss_end",
                "A",
                "admission_episode_count",
            ),
            1,
            0,
            id="lifecycle-algebra",
        ),
        pytest.param(
            (
                "runs",
                0,
                "primary_endpoints",
                "target_admission_loss_end",
                "A",
                "direct_candidate_admission_count",
            ),
            1,
            0,
            id="admission-cadence-closure",
        ),
        pytest.param(
            ("runs", 0, "primary_endpoints", "pre_recurrence_presence", 0, "target"),
            "B",
            0,
            id="recurrence-schedule",
        ),
        pytest.param(
            (
                "runs",
                0,
                "primary_endpoints",
                "pre_recurrence_ranks",
                "records",
                0,
                "active_slot_count",
            ),
            1,
            0,
            id="rank-slot-closure",
        ),
        pytest.param(
            (
                "runs",
                0,
                "primary_endpoints",
                "target_occupancy",
                "per_target",
                "A",
                "active_presence_fraction",
            ),
            0.5,
            0,
            id="occupancy-fraction",
        ),
        pytest.param(
            (
                "runs",
                0,
                "primary_endpoints",
                "cadence_integrity",
                "curation_count_closure",
                "curation_due_count",
            ),
            280,
            0,
            id="cadence-due-count",
        ),
        pytest.param(
            (
                "runs",
                0,
                "primary_endpoints",
                "cadence_integrity",
                "curation_count_closure",
                "logical_event_count",
            ),
            1,
            0,
            id="logical-event-closure",
        ),
        pytest.param(
            (
                "runs",
                0,
                "primary_endpoints",
                "cadence_integrity",
                "eventwise_curation_closure",
                "all_eventwise_curation_semantics_match",
            ),
            False,
            0,
            id="eventwise-semantics",
        ),
        pytest.param(
            ("runs", 0, "primary_endpoints", "identity_reacquisition_claimed"),
            True,
            0,
            id="identity-overclaim",
        ),
        pytest.param(
            ("cross_arm_contract", "shared_initial_state_sha256"),
            _sha("0"),
            None,
            id="cross-arm-genesis",
        ),
        pytest.param(
            ("cross_arm_contract", "behavioral_experience_matching_claimed"),
            True,
            None,
            id="experience-matching-overclaim",
        ),
        pytest.param(
            ("cross_arm_contract", "shared_protocol_source_and_genesis"),
            1,
            None,
            id="cross-arm-bool-integer",
        ),
        pytest.param(
            ("work_resource_contract", "panel_learner_updates"),
            44_989,
            None,
            id="work-contract",
        ),
        pytest.param(
            ("work_resource_contract", "panel_learner_updates"),
            44_990.0,
            None,
            id="work-contract-float",
        ),
        pytest.param(
            ("work_resource_contract_sha256",), _sha("0"), None, id="work-contract-hash"
        ),
    ],
)
def test_semantic_mutations_fail_closed(
    path: tuple[str | int, ...],
    replacement: object,
    arm_index: int | None,
) -> None:
    report = copy.deepcopy(_report())
    _set_path(report, path, replacement)
    _reseal(report, arm_index)

    with pytest.raises((TypeError, ValueError)):
        gate.validate_v3_descriptive_report(report, _expected_bindings())


@pytest.mark.parametrize(
    ("container_path", "field", "arm_index"),
    [
        ((), "unexpected", None),
        (("runs", 0), "unexpected", 0),
        (("runs", 0, "execution_receipt"), "unexpected", 0),
        (("runs", 0, "state_gate_receipt"), "unexpected", 0),
        (("runs", 0, "reward_counts"), "unexpected", 0),
        (("runs", 0, "reward_counts", "lifetime"), "unexpected", 0),
        (("runs", 0, "primary_endpoints"), "unexpected", 0),
        (("runs", 0, "primary_endpoints", "cadence_integrity"), "unexpected", 0),
        (("cross_arm_contract",), "unexpected", None),
        (("authority",), "unexpected", None),
    ],
)
def test_extra_fields_fail_closed(
    container_path: tuple[str | int, ...],
    field: str,
    arm_index: int | None,
) -> None:
    report = copy.deepcopy(_report())
    container: object = report
    for part in container_path:
        if type(part) is int:
            assert type(container) is list
            container = container[part]
        else:
            assert type(container) is dict
            container = container[part]
    assert type(container) is dict
    container[field] = None
    _reseal(report, arm_index)

    with pytest.raises(ValueError, match="field set"):
        gate.validate_v3_descriptive_report(report, _expected_bindings())


def test_arm_and_report_hashes_each_fail_closed() -> None:
    arm_corruption = _report()
    runs = arm_corruption["runs"]
    assert type(runs) is list and type(runs[0]) is dict
    runs[0]["arm_record_sha256"] = _sha("0")
    _reseal(arm_corruption)
    with pytest.raises(ValueError, match="arm_record_sha256"):
        gate.validate_v3_descriptive_report(arm_corruption, _expected_bindings())

    report_corruption = _report()
    report_corruption["report_sha256"] = _sha("0")
    with pytest.raises(ValueError, match="report_sha256"):
        gate.validate_v3_descriptive_report(report_corruption, _expected_bindings())


@pytest.mark.parametrize("replacement", [("A", "B"), float("nan"), -0.0, True])
def test_noncanonical_json_and_bool_integer_confusion_fail_closed(replacement: object) -> None:
    report = _report()
    if replacement is True:
        _set_path(report, ("runs", 0, "reward_counts", "lifetime", "steps"), True)
    else:
        _set_path(report, ("arm_order",), replacement)

    with pytest.raises((TypeError, ValueError)):
        gate.validate_v3_descriptive_report(report, _expected_bindings())
