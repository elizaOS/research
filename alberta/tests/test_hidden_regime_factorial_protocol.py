"""Contract tests for the non-executing hidden-regime calibration design."""

from __future__ import annotations

import builtins
import copy
import dataclasses
from collections import Counter
from collections.abc import Callable

import pytest

import alberta_framework.evaluation.hidden_regime_factorial_protocol as protocol
from alberta_framework.core.slot_signaling_agent import (
    N_DURABLE_SLOTS,
    N_SIGNAL_SYMBOLS,
    N_SLOT_ACTIONS,
    N_SLOT_INPUTS,
    N_SLOTS,
    SCRATCH_SLOT,
    SlotSignalingConfig,
)
from alberta_framework.evaluation.hidden_regime_factorial_protocol import (
    CALIBRATION_DESIGN_PAYLOAD_SHA256,
    CALIBRATION_MANIFEST_ORDER,
    CANONICAL_CONDITION_ORDER,
    CONSUMED_CALIBRATION_NAMESPACE,
    FACTORIAL_CELL_ORDER,
    FROZEN_SEED_PAIRS,
    N_MATCHED_CASES,
    PRIMARY_LEVEL_METRIC_IDS,
    PROTOCOL_STATUS,
    SEED_SNAPSHOT_SHA256,
    BaseEvaluatorConfigBinding,
    CalibrationAssignment,
    FactorialCell,
    FrozenSeedPair,
    MatchedCalibrationCase,
    MetricContract,
    PriorCommitLineageAuditRecord,
    audit_prior_commit_lineage_serialization,
    build_hidden_regime_factorial_calibration_design,
    calibration_design_envelope,
    calibration_design_payload,
    calibration_design_payload_sha256,
    canonical_json_bytes,
    canonical_sha256,
    derive_seed_pair_for_audit,
    seed_snapshot_payload,
    validate_calibration_design_envelope,
    validate_calibration_design_payload,
)
from alberta_framework.evaluation.hidden_regime_signaling_development import (
    HIDDEN_REGIME_DEVELOPMENT_SCHEMA,
    HIDDEN_REGIME_TRACE_SCHEMA,
    CommitGenerationLineage,
    DormantGenerationProbe,
    HiddenRegimeDevelopmentConfig,
    HiddenRegimeRunSummary,
    RecurrenceLineageProbe,
    RecurrenceRetentionRecord,
    RetentionAggregateSummary,
    condition_spec,
)
from alberta_framework.streams.hidden_regime_signaling import (
    CALIBRATION_ONLY_PARTITION,
    HIDDEN_REGIME_CALIBRATION_MANIFESTS,
    HIDDEN_REGIME_MANIFEST_USE_LEDGER,
    HIDDEN_REGIME_STRUCTURAL_MANIFESTS,
    PROTECTED_CANDIDATE_LEARNER_OUTCOMES_EXECUTED,
    PROTECTED_CANDIDATE_PARTITION,
    HiddenRegimeScheduleManifest,
)

pytestmark = pytest.mark.unit


def _ledger_snapshot() -> dict[str, dict[str, object]]:
    return {
        name: entry.to_dict()
        for name, entry in HIDDEN_REGIME_MANIFEST_USE_LEDGER.items()
    }


def _as_dict(value: object) -> dict[str, object]:
    assert type(value) is dict
    return value


def _as_list(value: object) -> list[object]:
    assert type(value) is list
    return value


def test_seed_snapshot_is_literal_unique_uint32_and_matches_disclosed_derivation() -> None:
    assert CONSUMED_CALIBRATION_NAMESPACE == (
        "hidden-regime-factorial-calibration-v1-consumed-nonpromoting"
    )
    assert len(FROZEN_SEED_PAIRS) == 30
    assert tuple(pair.index for pair in FROZEN_SEED_PAIRS) == tuple(range(30))
    assert FROZEN_SEED_PAIRS[0] == FrozenSeedPair(0, 1468689570, 1546104370)
    assert FROZEN_SEED_PAIRS[-1] == FrozenSeedPair(29, 3585556973, 2948504861)

    seeds = [
        seed
        for pair in FROZEN_SEED_PAIRS
        for seed in (pair.world_seed, pair.learner_seed)
    ]
    assert len(set(seeds)) == 60
    assert all(0 <= seed <= (1 << 32) - 1 for seed in seeds)
    assert all(
        derive_seed_pair_for_audit(pair.index) == (pair.world_seed, pair.learner_seed)
        for pair in FROZEN_SEED_PAIRS
    )
    assert canonical_sha256(seed_snapshot_payload()) == SEED_SNAPSHOT_SHA256
    assert SEED_SNAPSHOT_SHA256 == (
        "1733afb917902d180c1c784563e7b557162eb36c6904dc6bc79b4b721ce008f3"
    )


@pytest.mark.parametrize("bad_index", [-1, 30, True, 1.0, "1"])
def test_seed_derivation_rejects_noncanonical_indices(bad_index: object) -> None:
    with pytest.raises(ValueError, match="strict integer"):
        derive_seed_pair_for_audit(bad_index)  # type: ignore[arg-type]


def test_round_robin_assignment_is_balanced_and_cases_are_fully_matched() -> None:
    design = build_hidden_regime_factorial_calibration_design()
    assert len(design.assignments) == 30
    assert Counter(item.manifest_name for item in design.assignments) == Counter(
        {name: 10 for name in CALIBRATION_MANIFEST_ORDER}
    )
    assert tuple(item.manifest_name for item in design.assignments[:6]) == (
        CALIBRATION_MANIFEST_ORDER * 2
    )

    assert len(design.cases) == N_MATCHED_CASES == 240
    assert tuple(case.case_index for case in design.cases) == tuple(range(240))
    for seed_index in range(30):
        block = design.cases[seed_index * 8 : (seed_index + 1) * 8]
        assert tuple(case.condition for case in block) == CANONICAL_CONDITION_ORDER
        assert len({case.world_seed for case in block}) == 1
        assert len({case.learner_seed for case in block}) == 1
        assert len({case.manifest_name for case in block}) == 1
        assert {case.seed_index for case in block} == {seed_index}

    cross_tab = Counter((case.manifest_name, case.condition) for case in design.cases)
    assert set(cross_tab.values()) == {10}
    assert len(cross_tab) == 3 * 8


def test_condition_and_factorial_cell_order_is_exact() -> None:
    payload = calibration_design_payload()
    assert payload["condition_order"] == [
        "selective_full",
        "writable_evidence",
        "selective_lru",
        "writable_lru",
        "helper_frozen",
        "beneficiary_frozen",
        "constant_channel_0",
        "shuffled_channel",
    ]
    design = build_hidden_regime_factorial_calibration_design()
    assert tuple(cell.code for cell in design.factorial_cells) == FACTORIAL_CELL_ORDER
    assert tuple(cell.condition for cell in design.factorial_cells) == (
        "selective_full",
        "writable_evidence",
        "selective_lru",
        "writable_lru",
    )
    assert tuple(
        (cell.durable_write_policy, cell.replacement_target_policy)
        for cell in design.factorial_cells
    ) == (
        ("selective", "evidence"),
        ("writable", "evidence"),
        ("selective", "lru"),
        ("writable", "lru"),
    )


def test_all_eight_runtime_bindings_match_actual_evaluator_literals() -> None:
    design = build_hidden_regime_factorial_calibration_design()
    assert tuple(item.condition for item in design.condition_runtime_bindings) == (
        CANONICAL_CONDITION_ORDER
    )
    assert len(design.condition_runtime_bindings) == 8
    for binding in design.condition_runtime_bindings:
        actual = condition_spec(binding.condition)  # type: ignore[arg-type]
        assert actual.durable_write_policy == binding.durable_write_policy
        assert actual.replacement_target_policy == binding.replacement_target_policy
        assert actual.helper_write is binding.helper_learning_enabled
        assert actual.beneficiary_write is binding.beneficiary_learning_enabled
        assert actual.channel == binding.delivery_mode

    bindings = {item.condition: item for item in design.condition_runtime_bindings}
    constant = bindings["constant_channel_0"]
    assert constant.delivery_mode == "constant_0"
    assert constant.constant_delivery_symbol == 0
    assert constant.shuffle_low_inclusive is None
    shuffled = bindings["shuffled_channel"]
    assert shuffled.delivery_mode == "shuffled"
    assert (
        shuffled.shuffle_low_inclusive,
        shuffled.shuffle_high_exclusive,
        shuffled.shuffle_dtype,
    ) == (0, 3, "int32")
    assert "jax.random.split(world.channel_key)[0]" in str(shuffled.shuffle_key_rule)
    assert all(
        item.channel_key_advance_rule
        == "world channel_key advances exactly once on every transition"
        for item in design.condition_runtime_bindings
    )


def test_base_config_binding_matches_live_config_without_running_a_world() -> None:
    binding = build_hidden_regime_factorial_calibration_design().base_config_binding
    learner = SlotSignalingConfig(
        learning_rate=float(binding.learning_rate_decimal),
        epsilon=float(binding.epsilon_decimal),
        relevance_rate=float(binding.relevance_rate_decimal),
        lease_length=binding.lease_length,
        confirmation_steps=binding.confirmation_steps,
        durable_retrieval_threshold=float(binding.durable_retrieval_threshold_decimal),
        candidate_confirmation_threshold=float(
            binding.candidate_confirmation_threshold_decimal
        ),
        candidate_confirmation_leases=binding.candidate_confirmation_leases,
        scratch_training_leases_before_retest=(
            binding.scratch_training_leases_before_retest
        ),
        writable_lru_ablation=binding.writable_lru_ablation,
        durable_write_policy=binding.requested_durable_write_policy,
        replacement_target_policy=binding.requested_replacement_target_policy,
    )
    assert learner.to_dict() == {
        "learning_rate": 0.25,
        "epsilon": 0.1,
        "relevance_rate": 0.1,
        "lease_length": 16,
        "confirmation_steps": 8,
        "durable_retrieval_threshold": 0.5,
        "candidate_confirmation_threshold": 0.75,
        "candidate_confirmation_leases": 3,
        "scratch_training_leases_before_retest": 16,
        "writable_lru_ablation": False,
        "requested_durable_write_policy": None,
        "requested_replacement_target_policy": None,
        "effective_durable_write_policy": "selective",
        "effective_replacement_target_policy": "evidence",
        "development_only": True,
        "scientific_promotion_allowed": False,
    }
    world_config = HIDDEN_REGIME_CALIBRATION_MANIFESTS[
        CALIBRATION_MANIFEST_ORDER[0]
    ].to_world_config(repeat_schedule=False)
    live = HiddenRegimeDevelopmentConfig(
        world=world_config,
        learner=learner,
        metric_window=binding.metric_window,
    )
    assert live.num_steps == binding.expected_total_steps_per_manifest == 16_528
    assert live.world.repeat_schedule is binding.repeat_schedule is False
    assert (
        N_SIGNAL_SYMBOLS,
        N_SLOT_INPUTS,
        N_SLOT_ACTIONS,
        N_DURABLE_SLOTS,
        N_SLOTS,
        SCRATCH_SLOT,
    ) == (
        binding.signal_symbols,
        binding.slot_inputs,
        binding.slot_actions,
        binding.durable_slots,
        binding.total_slots_per_role,
        binding.scratch_slot,
    )


@pytest.mark.parametrize(
    "field_name",
    [field.name for field in dataclasses.fields(BaseEvaluatorConfigBinding)],
)
def test_every_base_config_field_mutation_is_rejected(field_name: str) -> None:
    binding = build_hidden_regime_factorial_calibration_design().base_config_binding
    old_value = getattr(binding, field_name)
    if old_value is None:
        mutated: object = "non-null"
    elif type(old_value) is bool:
        mutated = not old_value
    elif type(old_value) is int:
        mutated = old_value + 1
    else:
        assert type(old_value) is str
        mutated = old_value + "-mutated"
    with pytest.raises(ValueError, match="base evaluator configuration|base requested policies"):
        dataclasses.replace(binding, **{field_name: mutated})  # type: ignore[arg-type]


def test_only_calibration_manifests_are_bound_and_no_candidate_name_is_serialized() -> None:
    design = build_hidden_regime_factorial_calibration_design()
    assert tuple(binding.name for binding in design.manifest_bindings) == (
        CALIBRATION_MANIFEST_ORDER
    )
    assert all(
        binding.use_partition == CALIBRATION_ONLY_PARTITION
        for binding in design.manifest_bindings
    )
    serialized = canonical_json_bytes(design.to_payload()).decode("ascii")
    assert all(name not in serialized for name in HIDDEN_REGIME_STRUCTURAL_MANIFESTS)
    assert all("hidden-regime-structural-" not in case.manifest_name for case in design.cases)
    assert "heldout" not in CONSUMED_CALIBRATION_NAMESPACE
    assert "reserved" not in CONSUMED_CALIBRATION_NAMESPACE
    assert "protected" not in CONSUMED_CALIBRATION_NAMESPACE


def test_manifest_content_bindings_match_the_live_calibration_registry() -> None:
    design = build_hidden_regime_factorial_calibration_design()
    for binding in design.manifest_bindings:
        manifest = HIDDEN_REGIME_CALIBRATION_MANIFESTS[binding.name]
        assert manifest.use_partition == CALIBRATION_ONLY_PARTITION
        assert canonical_sha256(manifest.to_dict()) == binding.manifest_payload_sha256


def test_recurrence_bindings_coalesce_adjacent_equal_regime_segments() -> None:
    design = build_hidden_regime_factorial_calibration_design()
    bindings = {item.manifest_name: item for item in design.recurrence_bindings}
    a = bindings["hidden-regime-calibration-a-v1"]
    b = bindings["hidden-regime-calibration-b-v1"]
    c = bindings["hidden-regime-calibration-c-v1"]

    assert 5 in a.coalesced_episode_start_segment_indices
    assert 6 not in a.coalesced_episode_start_segment_indices
    assert 5 in a.eligible_recurrence_start_segment_indices
    assert 6 not in a.eligible_recurrence_start_segment_indices
    assert 2 in b.coalesced_episode_start_segment_indices
    assert 3 not in b.coalesced_episode_start_segment_indices
    assert 3 not in b.eligible_recurrence_start_segment_indices
    assert c.coalesced_episode_start_segment_indices == tuple(range(17))

    assert tuple(
        len(item.eligible_recurrence_start_segment_indices)
        for item in design.recurrence_bindings
    ) == (11, 11, 12)
    assert tuple(
        item.eligible_recurrence_counts_by_regime for item in design.recurrence_bindings
    ) == ((4, 4, 1, 2, 0), (4, 4, 1, 2, 0), (5, 4, 1, 2, 0))
    assert a.eligible_recurrence_identities == (
        (3, 2, 1),
        (4, 1, 1),
        (5, 0, 1),
        (7, 1, 2),
        (9, 0, 2),
        (10, 1, 3),
        (11, 3, 1),
        (12, 0, 3),
        (14, 3, 2),
        (15, 1, 4),
        (16, 0, 4),
    )
    payload = calibration_design_payload()
    serialized = _as_list(payload["recurrence_eligibility_bindings"])
    assert all(
        "adjacent equal-regime" in str(_as_dict(item)["eligibility_rule"])
        for item in serialized
    )
    assert all(
        _as_dict(item)["required_runtime_helper"]
        == "hidden_regime_lineage_recurrence_segments(world)"
        for item in serialized
    )
    assert all(_as_dict(item)["expected_total_steps"] == 16_528 for item in serialized)


def test_protocol_status_execution_policy_and_thresholds_are_fail_closed() -> None:
    payload = calibration_design_payload()
    assert payload["protocol_status"] == PROTOCOL_STATUS
    assert PROTOCOL_STATUS == "calibration_design_frozen_outcomes_unexecuted"
    assert payload["outcomes_observed_at_design_freeze"] is False
    assert payload["design_module_execution_api_available"] is False
    assert payload["design_module_artifact_writer_available"] is False
    assert payload["outcome_artifact_written_at_design_freeze"] is False
    assert payload["scientific_promotion_allowed"] is False
    schema_bindings = _as_dict(payload["runtime_schema_bindings"])
    assert schema_bindings == {
        "development_summary_schema": HIDDEN_REGIME_DEVELOPMENT_SCHEMA,
        "primitive_trace_schema": HIDDEN_REGIME_TRACE_SCHEMA,
    }
    assert HIDDEN_REGIME_DEVELOPMENT_SCHEMA == (
        "alberta.hidden-regime-signaling.development.v5"
    )
    assert HIDDEN_REGIME_TRACE_SCHEMA == (
        "alberta.hidden-regime-signaling.primitive-trace.v3"
    )
    assert payload["frozen_thresholds"] == []
    assert payload["threshold_freeze_receipt"] is None

    policy = _as_dict(payload["execution_policy"])
    assert policy["permitted_partition"] == CALIBRATION_ONLY_PARTITION
    assert "if and only if" in str(policy["authorization_rule"])
    assert "content-addressed readiness receipt" in str(policy["authorization_rule"])
    assert policy["protected_candidate_execution_permitted"] is False
    assert policy["promotion_permitted"] is False
    certifications = _as_list(policy["readiness_receipt_required_certifications"])
    assert len(certifications) == 8
    assert any("coalesced_nonadjacent_recurrence" in str(item) for item in certifications)
    assert any("actual_terminated_and_discount" in str(item) for item in certifications)

    namespace = _as_dict(payload["namespace"])
    assert namespace["promotion_eligible"] is False
    assert namespace["protected_seed_namespace"] is None
    assert "permanently_consumed_nonpromoting" in str(namespace["disposition"])


def test_primary_retention_is_predeclared_lineage_not_posthoc_best() -> None:
    design = build_hidden_regime_factorial_calibration_design()
    metrics = {metric.metric_id: metric for metric in design.metrics}
    expected_primary = {
        "qualified_first_entry_window_error_rate",
        "latest_prior_qualified_lineage_survival_rate",
        "selected_lineage_joint_bit_exact_preservation_rate",
        "selected_lineage_entry_composed_accuracy",
        "selected_lineage_commit_to_entry_accuracy_change",
        "selected_lineage_exact_generation_relock_rate",
        "selected_lineage_retrieval_before_scratch_rate",
        "recurrence_minus_latest_qualified_acquisition_error_rate",
    }
    assert {metric.metric_id for metric in design.metrics if metric.role == "primary"} == (
        expected_primary
    )
    assert all(metrics[name].role == "diagnostic" for name in metrics if name.startswith("best_"))
    assert all(
        metrics[name].gate_mode == "diagnostic_only" for name in metrics if name.startswith("best_")
    )

    lineage = _as_dict(design.to_payload()["lineage_selection_contract"])
    assert lineage["observer"] == "evaluator_only_not_learner_visible"
    assert lineage["all_prior_generation_lineages_serialized"] == (
        "all_including_unqualified_and_evicted"
    )
    assert lineage["selection_uses_recurrence_performance"] is False
    assert lineage["posthoc_best_dormant_primary_allowed"] is False
    assert lineage["prior_generation_lineage_order"] == [
        "commit_step",
        "slot",
        "generation",
        "lineage_index",
    ]
    assert "exact ordered identifier equality" in str(lineage["omission_detection"])
    assert "derived filter" in str(lineage["qualified_prior_lineages"])
    assert "committed_composed_greedy_accuracy=1.0" in str(
        lineage["acquisition_qualification"]
    )
    assert "committed_composed_greedy_tie_free=true" in str(
        lineage["acquisition_qualification"]
    )
    assert lineage["latest_prior_qualified_lineage_index_path"] == (
        "summary.recurrence_retention[*].latest_prior_qualified_lineage_index"
    )
    assert lineage["latest_prior_qualified_commit_step_path"] == (
        "summary.recurrence_retention[*].latest_prior_qualified_commit_step"
    )
    assert "exact lineage_index" in str(lineage["latest_prior_qualified_lineage"])
    assert "join exactly" in str(lineage["latest_prior_qualified_lineage"])
    assert "commit_segment_index" in str(
        lineage["latest_qualified_acquisition_episode_binding"]
    )
    assert "maximum commit_step among qualified" in str(lineage["selected_probe_lineage"])
    assert "acquisition-unqualified" in str(lineage["retention_denominator"])
    assert "no endpoint can substitute" in str(lineage["non_substitution_rule"])
    bit_metric = metrics["selected_lineage_joint_bit_exact_preservation_rate"]
    assert bit_metric.role == "primary"
    assert bit_metric.gate_mode == "level_and_contrast"
    assert "summary.retention.selected_joint_bit_exact_preservation_count" in (
        bit_metric.source_fields
    )
    assert (
        "summary.recurrence_retention[*].selected_lineage_joint_bit_exact_preserved"
        in bit_metric.source_fields
    )
    coverage = metrics["acquisition_qualified_recurrence_coverage_rate"]
    latest = metrics["latest_prior_qualified_lineage_survival_rate"]
    any_survival = metrics["any_qualified_lineage_survival_rate"]
    assert coverage.role == "secondary"
    assert latest.role == "primary"
    assert any_survival.role == "secondary"
    assert "evaluator-known episode entry" in coverage.eligibility
    assert "committed_composed_greedy_accuracy=1.0" in latest.eligibility
    assert "committed_composed_greedy_tie_free=true" in latest.eligibility
    assert "latest prior qualified" in latest.aggregation
    assert (
        "summary.recurrence_retention[*].latest_prior_qualified_lineage_index"
        in latest.source_fields
    )
    assert "at least one qualified prior" in any_survival.aggregation
    acquisition_comparison = metrics[
        "recurrence_minus_latest_qualified_acquisition_error_rate"
    ]
    assert acquisition_comparison.gate_mode == "contrast_only"
    assert acquisition_comparison.null_value_decimal is None
    assert acquisition_comparison.metric_id not in PRIMARY_LEVEL_METRIC_IDS
    assert (
        "summary.retention.recurrence_minus_latest_qualified_acquisition_error_rate_delta_mean"
        in acquisition_comparison.source_fields
    )
    assert (
        "summary.recurrence_retention[*].latest_qualified_acquisition_segment_index"
        in acquisition_comparison.source_fields
    )
    assert (
        "summary.recurrence_retention[*].latest_prior_qualified_commit_step"
        in acquisition_comparison.source_fields
    )
    assert (
        "summary.recurrence_retention[*].latest_prior_qualified_lineage_index"
        in acquisition_comparison.source_fields
    )
    assert (
        "summary.recurrence_retention[*].prior_same_regime_lineages[*].commit_segment_index"
        in acquisition_comparison.source_fields
    )
    assert all("legacy_first_exposure" not in path for path in acquisition_comparison.source_fields)


def test_every_metric_source_path_resolves_against_live_v4_serialized_dataclasses() -> None:
    design = build_hidden_regime_factorial_calibration_design()
    summary_fields = {field.name for field in dataclasses.fields(HiddenRegimeRunSummary)}
    retention_fields = {
        field.name for field in dataclasses.fields(RetentionAggregateSummary)
    }
    recurrence_fields = {
        field.name for field in dataclasses.fields(RecurrenceRetentionRecord)
    }
    commit_fields = {
        field.name for field in dataclasses.fields(CommitGenerationLineage)
    }
    lineage_probe_fields = {
        field.name for field in dataclasses.fields(RecurrenceLineageProbe)
    }
    dormant_probe_fields = {
        field.name for field in dataclasses.fields(DormantGenerationProbe)
    }

    source_paths = tuple(
        path for contract in design.metrics for path in contract.source_fields
    ) + tuple(
        path
        for contract in design.paired_population_support_metrics
        for path in contract.source_fields
    )
    assert source_paths
    for path in source_paths:
        parts = path.split(".")
        assert parts[0] == "summary", path
        if parts[1] == "retention":
            assert "retention" in summary_fields, path
            assert len(parts) == 3, path
            assert parts[2] in retention_fields, path
        elif parts[1] == "commit_generation_lineages[*]":
            assert "commit_generation_lineages" in summary_fields, path
            assert len(parts) == 3, path
            assert parts[2] in commit_fields, path
        elif parts[1] == "recurrence_retention[*]":
            assert "recurrence_retention" in summary_fields, path
            if len(parts) == 3:
                assert parts[2] in recurrence_fields, path
            elif parts[2] == "prior_same_regime_lineages[*]":
                assert "prior_same_regime_lineages" in recurrence_fields, path
                assert len(parts) == 4, path
                assert parts[3] in lineage_probe_fields, path
            elif parts[2] == "eligible_dormant_generations[*]":
                assert "eligible_dormant_generations" in recurrence_fields, path
                assert len(parts) == 4, path
                assert parts[3] in dormant_probe_fields, path
            else:
                pytest.fail(f"unknown nested recurrence source path: {path}")
        else:
            assert len(parts) == 2, path
            assert parts[1] in summary_fields, path

    payload = calibration_design_payload()
    lineage = _as_dict(payload["lineage_selection_contract"])
    assert lineage["global_commit_collection_path"] == (
        "summary.commit_generation_lineages"
    )
    assert tuple(_as_list(lineage["global_commit_record_fields"])) == tuple(
        field.name for field in dataclasses.fields(CommitGenerationLineage)
    )
    assert lineage["recurrence_lineage_collection_path"] == (
        "summary.recurrence_retention[*].prior_same_regime_lineages"
    )
    assert tuple(_as_list(lineage["recurrence_lineage_probe_fields"])) == tuple(
        field.name for field in dataclasses.fields(RecurrenceLineageProbe)
    )
    assert set(_as_list(lineage["global_commit_count_paths"])) == {
        "summary.synchronized_commit_lineage_count",
        "summary.acquisition_qualified_commit_lineage_count",
        "summary.acquisition_unqualified_commit_lineage_count",
    }
    assert {
        str(path).removeprefix("summary.")
        for path in _as_list(lineage["global_commit_count_paths"])
    }.issubset(summary_fields)
    for contract_path_key in (
        "latest_prior_qualified_lineage_index_path",
        "latest_prior_qualified_commit_step_path",
    ):
        contract_path = str(lineage[contract_path_key])
        contract_parts = contract_path.split(".")
        assert contract_parts[:2] == ["summary", "recurrence_retention[*]"]
        assert len(contract_parts) == 3
        assert contract_parts[2] in recurrence_fields

    serialized = canonical_json_bytes(payload).decode("ascii")
    for obsolete_token in (
        "prior_commit_lineages",
        "selected_surviving_qualified_lineage",
        "dormant_generation_probes",
    ):
        assert obsolete_token not in serialized


def test_lineage_audit_retains_unqualified_and_evicted_records_and_rejects_omission() -> None:
    ledger = (
        PriorCommitLineageAuditRecord(0, 10, 3, 0, 1, True, False, False),
        PriorCommitLineageAuditRecord(1, 20, 3, 1, 2, True, True, True),
        PriorCommitLineageAuditRecord(2, 30, 3, 2, 3, True, True, False),
        PriorCommitLineageAuditRecord(3, 35, 1, 0, 4, True, True, True),
    )
    audit = audit_prior_commit_lineage_serialization(
        ledger,
        recurrence_regime_id=3,
        recurrence_entry_step=40,
        serialized_prior_lineage_indices=(0, 1, 2),
    )
    assert audit.all_prior_lineage_indices == (0, 1, 2)
    assert audit.qualified_prior_lineage_indices == (1, 2)
    assert audit.latest_prior_qualified_lineage_index == 2
    assert audit.surviving_qualified_lineage_indices == (1,)
    assert audit.selected_latest_surviving_qualified_lineage_index == 1

    with pytest.raises(ValueError, match="including unqualified and evicted"):
        audit_prior_commit_lineage_serialization(
            ledger,
            recurrence_regime_id=3,
            recurrence_entry_step=40,
            serialized_prior_lineage_indices=(1, 2),
        )


def test_all_probe_zero_role_and_role_swapped_diagnostics_are_predeclared() -> None:
    metrics = {
        metric.metric_id: metric
        for metric in build_hidden_regime_factorial_calibration_design().metrics
    }
    for name in (
        "all_dormant_probe_composed_accuracy",
        "all_dormant_probe_composed_minus_zero_helper_accuracy",
        "all_dormant_probe_composed_minus_zero_beneficiary_accuracy",
        "all_dormant_probe_composed_minus_role_swapped_accuracy",
    ):
        assert metrics[name].role == "secondary"
        assert "every" in metrics[name].aggregation or "mean paired" in metrics[name].aggregation
    assert "zero_helper_accuracy" in metrics[
        "all_dormant_probe_composed_minus_zero_helper_accuracy"
    ].source_fields[1]
    assert "zero_beneficiary_accuracy" in metrics[
        "all_dormant_probe_composed_minus_zero_beneficiary_accuracy"
    ].source_fields[1]
    assert "role_swapped_accuracy" in metrics[
        "all_dormant_probe_composed_minus_role_swapped_accuracy"
    ].source_fields[1]


def test_factorial_estimands_have_exact_oriented_formulas() -> None:
    design = build_hidden_regime_factorial_calibration_design()
    estimands = {item.estimand_id: item for item in design.factorial_estimands}
    assert tuple(estimands) == (
        "immutability_evidence_primary",
        "immutability_lru_replication",
        "replacement_target_selective_secondary",
        "replacement_target_writable_secondary",
        "write_by_replacement_interaction_secondary",
    )
    assert estimands["immutability_evidence_primary"].role == "primary"
    assert "z_m(SE)-z_m(WE)" in estimands["immutability_evidence_primary"].formula
    assert estimands["immutability_lru_replication"].role == "replication"
    assert "z_m(SL)-z_m(WL)" in estimands["immutability_lru_replication"].formula
    assert "z_m(SE)-z_m(SL)" in estimands[
        "replacement_target_selective_secondary"
    ].formula
    assert "z_m(WE)-z_m(WL)" in estimands[
        "replacement_target_writable_secondary"
    ].formula
    assert "[z_m(SE)-z_m(WE)]-[z_m(SL)-z_m(WL)]" in estimands[
        "write_by_replacement_interaction_secondary"
    ].formula
    assert estimands["immutability_evidence_primary"].condition_terms == (
        ("selective_full", 1),
        ("writable_evidence", -1),
    )
    assert estimands["write_by_replacement_interaction_secondary"].condition_terms == (
        ("selective_full", 1),
        ("writable_evidence", -1),
        ("selective_lru", -1),
        ("writable_lru", 1),
    )
    assert all("exact intersection" in item.population_rule for item in design.factorial_estimands)
    assert all(
        set(item.metrics).issuperset(
            {
                "qualified_first_entry_window_error_rate",
                "latest_prior_qualified_lineage_survival_rate",
                "selected_lineage_joint_bit_exact_preservation_rate",
                "selected_lineage_entry_composed_accuracy",
            }
        )
        for item in design.factorial_estimands
    )


def test_causal_control_estimands_are_matched_and_complete() -> None:
    design = build_hidden_regime_factorial_calibration_design()
    assert tuple(item.condition_terms[1][0] for item in design.control_estimands) == (
        "helper_frozen",
        "beneficiary_frozen",
        "constant_channel_0",
        "shuffled_channel",
    )
    assert all(
        item.condition_terms[0] == ("selective_full", 1)
        and item.condition_terms[1][1] == -1
        for item in design.control_estimands
    )
    assert all("paired z_m(SE)" in item.formula for item in design.control_estimands)
    assert all(
        item.metrics
        == (
            "mean_prequential_reward",
            "all_recurrence_first_entry_window_error_rate",
            "acquisition_qualified_recurrence_coverage_rate",
        )
        for item in design.control_estimands
    )


def test_audits_cover_learning_replacement_resources_lifecycle_and_controls() -> None:
    design = build_hidden_regime_factorial_calibration_design()
    audit_ids = {item.requirement_id for item in design.audits}
    assert audit_ids == {
        "lineage_serialization",
        "both_roles_learning",
        "atomic_c_old_to_c_new_replacement",
        "d_short_non_displacement",
        "constant_resource",
        "complete_role_lifecycle_oracle",
        "complete_world_oracle",
        "source_bound_trace_contract",
        "decentralized_role_equivalence",
        "checkpoint_resume_equivalence",
        "frozen_role_causal_controls",
        "channel_causal_controls",
    }
    audits = {item.requirement_id: item for item in design.audits}
    assert "latest_prior_qualified_lineage_index" in audits[
        "lineage_serialization"
    ].predicate
    assert audits["atomic_c_old_to_c_new_replacement"].scope.startswith("SE in every")
    assert audits["d_short_non_displacement"].scope.startswith("SE in every")
    assert "descriptive" in audits["atomic_c_old_to_c_new_replacement"].scope
    resource = _as_dict(design.to_payload()["resource_contract"])
    assert resource == {
        "per_role_scalars": 69,
        "per_role_bytes": 276,
        "dyad_scalars": 138,
        "dyad_bytes": 552,
        "constant_at_every_step": True,
        "matched_across_all_conditions": True,
    }


def test_statistical_plan_predeclares_paired_lcbs_strata_wins_and_missingness() -> None:
    plan = build_hidden_regime_factorial_calibration_design().statistical_plan
    assert plan.pooled_expected_n == 30
    assert plan.manifest_expected_n == 10
    assert plan.confidence_basis_points == 9500
    assert "n-1" in plan.standard_deviation
    assert "t_quantile(0.95,df=n-1)" in plan.one_sided_lower_bound
    assert "strictly greater than zero" in plan.win_definition
    assert "exactly equal to zero" in plan.tie_definition
    assert "missing_n=0" in plan.missingness_policy
    assert "minimum_manifest_mean_oriented_delta" in plan.worst_manifest_summary
    assert "minimum_manifest_one_sided_95_percent_lower_confidence_bound" in (
        plan.worst_manifest_summary
    )
    assert "no familywise probability claim" in plan.multiplicity_scope
    assert "never compare conditional means" in plan.paired_recurrence_alignment
    assert "selected-survival intersection coverage" in plan.paired_recurrence_alignment


def test_gate_matrix_fixes_mandatory_scope_before_outcomes() -> None:
    design = build_hidden_regime_factorial_calibration_design()
    families = {family.gate_family_id: family for family in design.gate_families}
    assert tuple(families) == (
        "candidate_primary_level_gates",
        "acquisition_coverage_level_gates",
        "primary_paired_population_support_gates",
        "primary_immutability_contrast_gates",
        "replication_paired_population_support_gates",
        "lru_immutability_replication_gates",
        "selective_lru_absolute_levels_descriptive",
        "causal_control_contrast_gates",
        "mandatory_trace_and_lifecycle_audits",
        "replacement_and_interaction_descriptive",
        "probe_and_posthoc_best_diagnostics",
    )
    assert families["candidate_primary_level_gates"].conditions == ("selective_full",)
    assert families["acquisition_coverage_level_gates"].conditions == ("selective_full",)
    assert families["primary_paired_population_support_gates"].metric_ids == (
        "se_we_paired_qualification_intersection_coverage_rate",
        "se_we_paired_selected_survival_intersection_coverage_rate",
    )
    assert families["replication_paired_population_support_gates"].metric_ids == (
        "sl_wl_paired_qualification_intersection_coverage_rate",
        "sl_wl_paired_selected_survival_intersection_coverage_rate",
    )
    assert "selected_lineage_commit_to_entry_accuracy_change" not in families[
        "candidate_primary_level_gates"
    ].metric_ids
    assert "recurrence_minus_latest_qualified_acquisition_error_rate" not in families[
        "candidate_primary_level_gates"
    ].metric_ids
    for family_name in (
        "primary_immutability_contrast_gates",
        "lru_immutability_replication_gates",
    ):
        family = families[family_name]
        assert family.mandatory is True
        assert "selected_lineage_commit_to_entry_accuracy_change" in family.metric_ids
        assert "exact null 0" in family.null_rule
        assert "pooled_one_sided_95_percent_paired_t_lower_bound" in (
            family.required_components
        )
        assert "worst_manifest_mean_and_one_sided_95_percent_paired_t_lower_bound" in (
            family.required_components
        )
    assert families["lru_immutability_replication_gates"].conditions == (
        "selective_lru",
        "writable_lru",
    )
    assert "does not require absolute LRU success" in families[
        "lru_immutability_replication_gates"
    ].null_rule
    assert families["selective_lru_absolute_levels_descriptive"].mandatory is False
    assert families["replacement_and_interaction_descriptive"].mandatory is False
    assert families["probe_and_posthoc_best_diagnostics"].mandatory is False
    assert families["causal_control_contrast_gates"].mandatory is True

    support = {
        item.metric_id: item for item in design.paired_population_support_metrics
    }
    assert set(support) == {
        "se_we_paired_qualification_intersection_coverage_rate",
        "se_we_paired_selected_survival_intersection_coverage_rate",
        "sl_wl_paired_qualification_intersection_coverage_rate",
        "sl_wl_paired_selected_survival_intersection_coverage_rate",
    }
    assert all(item.mandatory is True for item in support.values())
    assert all("11 for calibration A" in item.denominator for item in support.values())
    assert support[
        "se_we_paired_selected_survival_intersection_coverage_rate"
    ].conditions == ("selective_full", "writable_evidence")

    payload = calibration_design_payload()
    gate_payload = _as_list(payload["gate_families"])
    assert payload["gate_matrix_sha256"] == canonical_sha256(gate_payload)
    semantics = _as_dict(payload["gate_mode_semantics"])
    assert "separate level and paired-contrast" in str(semantics["level_and_contrast"])
    assert "no level gate" in str(semantics["contrast_only"])
    assert semantics["unlisted_metric_estimand_pair"] == "descriptive_only"
    rationale = _as_dict(payload["gate_scope_rationale"])
    assert rationale["mandatory_absolute_levels"] == "SE only"
    assert "SL-vs-WL" in str(rationale["lru_axis"])
    assert "cannot reject SE" in str(rationale["lru_axis"])
    acquisition_delta_rationale = str(rationale["latest_qualified_acquisition_delta"])
    assert "contrast_only" in acquisition_delta_rationale
    assert "initial acquisition, relearning" in acquisition_delta_rationale
    assert "already-mastered refresh" in acquisition_delta_rationale
    assert "no level null" in acquisition_delta_rationale


def test_threshold_rule_is_deterministic_twofold_and_has_immutable_receipt_contract() -> None:
    rule = build_hidden_regime_factorial_calibration_design().threshold_rule
    assert rule.frozen_thresholds == ()
    assert rule.threshold_freeze_receipt is None
    assert (
        rule.minimum_margin_ratio_numerator,
        rule.minimum_margin_ratio_denominator,
    ) == (2, 1)
    assert "floor_to_0.0001" in rule.higher_is_better_rule
    assert "rounding toward N" in rule.lower_is_better_rule
    assert "W0=n/2" in rule.wins_rule
    assert "pooled n=30 gives 15" in rule.wins_rule
    assert "manifest n=10 gives 5" in rule.wins_rule
    assert "q>=1" in rule.wins_rule
    assert rule.mandatory_missingness_rule.endswith("exactly zero")
    assert "pooled one-sided 95% bound" in rule.conservative_bound_rule
    assert "worst of the three manifest-stratified" in rule.conservative_bound_rule
    assert "created exclusively at a new path" in rule.receipt_immutability
    assert "never overwritten" in rule.receipt_immutability
    assert "forbidden" in rule.post_protected_adjustment
    assert "failed gate remains a valid rejection" in rule.post_protected_adjustment
    assert {
        "protocol_payload_sha256",
        "gate_matrix_sha256",
        "calibration_outcomes_payload_sha256",
        "mandatory_gate_results",
        "protected_outcomes_observed",
        "amendments_allowed",
        "receipt_payload_sha256",
    }.issubset(set(rule.receipt_required_fields))


def test_protected_candidate_ledger_is_false_before_and_after_construction() -> None:
    before = _ledger_snapshot()
    assert PROTECTED_CANDIDATE_LEARNER_OUTCOMES_EXECUTED is False
    assert all(
        entry.learner_outcomes_executed is False
        for entry in HIDDEN_REGIME_MANIFEST_USE_LEDGER.values()
        if entry.use_partition == PROTECTED_CANDIDATE_PARTITION
    )
    payload = calibration_design_payload()
    after = _ledger_snapshot()
    assert before == after
    guard = _as_dict(payload["protected_candidate_guard"])
    assert guard == {
        "learner_outcomes_observed_at_design_freeze": False,
        "outcome_ledger_all_false_at_design_freeze": True,
        "manifest_names_serialized": False,
    }


def test_construction_neither_builds_a_world_runs_outcomes_nor_writes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = _ledger_snapshot()

    def forbidden_world_build(
        self: HiddenRegimeScheduleManifest,
        *,
        repeat_schedule: bool = False,
    ) -> object:
        del self, repeat_schedule
        raise AssertionError("world construction is forbidden in protocol construction")

    def forbidden_open(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("file access is forbidden in protocol construction")

    monkeypatch.setattr(HiddenRegimeScheduleManifest, "to_world_config", forbidden_world_build)
    monkeypatch.setattr(builtins, "open", forbidden_open)
    design = build_hidden_regime_factorial_calibration_design()
    payload = design.to_payload()

    assert payload["outcomes_observed_at_design_freeze"] is False
    assert payload["design_module_execution_api_available"] is False
    assert _ledger_snapshot() == before
    assert not hasattr(protocol, "run_hidden_regime_condition")
    assert not hasattr(protocol, "HiddenRegimeSignalingWorld")


def test_payload_and_envelope_round_trip_against_frozen_digests() -> None:
    payload = calibration_design_payload()
    assert calibration_design_payload_sha256() == CALIBRATION_DESIGN_PAYLOAD_SHA256
    assert CALIBRATION_DESIGN_PAYLOAD_SHA256 == (
        "735ceb533717e8b71c0159372b44041b2fd533ec14b62e78234de2c3552dd47d"
    )
    assert validate_calibration_design_payload(copy.deepcopy(payload)).to_payload() == payload

    envelope = calibration_design_envelope()
    assert envelope["payload_sha256"] == CALIBRATION_DESIGN_PAYLOAD_SHA256
    assert validate_calibration_design_envelope(copy.deepcopy(envelope)).to_payload() == payload


PayloadMutation = Callable[[dict[str, object]], None]


def _mutate_status(payload: dict[str, object]) -> None:
    payload["protocol_status"] = "executed"


def _mutate_seed(payload: dict[str, object]) -> None:
    first = _as_dict(_as_list(payload["seed_pairs"])[0])
    world_seed = first["world_seed"]
    assert type(world_seed) is int
    first["world_seed"] = world_seed + 1


def _mutate_seed_to_bool(payload: dict[str, object]) -> None:
    first = _as_dict(_as_list(payload["seed_pairs"])[0])
    first["world_seed"] = True


def _mutate_assignment(payload: dict[str, object]) -> None:
    first = _as_dict(_as_list(payload["assignments"])[0])
    first["manifest_name"] = CALIBRATION_MANIFEST_ORDER[1]


def _mutate_condition_order(payload: dict[str, object]) -> None:
    conditions = _as_list(payload["condition_order"])
    conditions[0], conditions[1] = conditions[1], conditions[0]


def _mutate_case_count(payload: dict[str, object]) -> None:
    payload["matched_case_count"] = 239


def _mutate_manifest_name(payload: dict[str, object]) -> None:
    first = _as_dict(_as_list(payload["manifest_bindings"])[0])
    first["name"] = "hidden-regime-calibration-z-v1"


def _mutate_threshold(payload: dict[str, object]) -> None:
    _as_list(payload["frozen_thresholds"]).append({"metric": "reward", "value": 0})


def _mutate_lineage_selection(payload: dict[str, object]) -> None:
    lineage = _as_dict(payload["lineage_selection_contract"])
    lineage["posthoc_best_dormant_primary_allowed"] = True


def _mutate_runtime_binding(payload: dict[str, object]) -> None:
    first = _as_dict(_as_list(payload["condition_runtime_bindings"])[0])
    first["helper_learning_enabled"] = False


def _mutate_runtime_schema_binding(payload: dict[str, object]) -> None:
    schemas = _as_dict(payload["runtime_schema_bindings"])
    schemas["development_summary_schema"] = (
        "alberta.hidden-regime-signaling.development.v3"
    )


def _mutate_metric_source_path(payload: dict[str, object]) -> None:
    first = _as_dict(_as_list(payload["metric_contracts"])[0])
    _as_list(first["source_fields"])[0] = "summary.nonexistent"


def _mutate_paired_support_source_path(payload: dict[str, object]) -> None:
    first = _as_dict(_as_list(payload["paired_population_support_metrics"])[0])
    _as_list(first["source_fields"])[0] = "summary.nonexistent"


def _mutate_base_config(payload: dict[str, object]) -> None:
    base = _as_dict(payload["base_config_binding"])
    base["lease_length"] = 17


def _mutate_recurrence_identity(payload: dict[str, object]) -> None:
    first = _as_dict(_as_list(payload["recurrence_eligibility_bindings"])[0])
    first_identity = _as_list(_as_list(first["eligible_recurrence_identities"])[0])
    first_identity[0] = 6


def _mutate_gate_scope(payload: dict[str, object]) -> None:
    first = _as_dict(_as_list(payload["gate_families"])[0])
    first["mandatory"] = False


def _mutate_interaction_term(payload: dict[str, object]) -> None:
    interaction = _as_dict(_as_list(payload["factorial_estimands"])[4])
    fourth = _as_dict(_as_list(interaction["condition_terms"])[3])
    fourth["coefficient"] = 2


def _add_unknown_field(payload: dict[str, object]) -> None:
    payload["unknown"] = None


@pytest.mark.parametrize(
    "mutate",
    [
        _mutate_status,
        _mutate_seed,
        _mutate_seed_to_bool,
        _mutate_assignment,
        _mutate_condition_order,
        _mutate_case_count,
        _mutate_manifest_name,
        _mutate_threshold,
        _mutate_lineage_selection,
        _mutate_runtime_binding,
        _mutate_runtime_schema_binding,
        _mutate_metric_source_path,
        _mutate_paired_support_source_path,
        _mutate_base_config,
        _mutate_recurrence_identity,
        _mutate_gate_scope,
        _mutate_interaction_term,
        _add_unknown_field,
    ],
)
def test_strict_payload_validation_rejects_nested_mutations(mutate: PayloadMutation) -> None:
    payload = calibration_design_payload()
    mutate(payload)
    with pytest.raises(ValueError, match="differs from the frozen canonical design"):
        validate_calibration_design_payload(payload)


def test_envelope_rejects_tampering_even_when_attacker_recomputes_digest() -> None:
    envelope = calibration_design_envelope()
    payload = _as_dict(envelope["payload"])
    payload["protocol_status"] = "tampered"
    envelope["payload_sha256"] = canonical_sha256(payload)
    with pytest.raises(ValueError, match="differs from frozen source digest"):
        validate_calibration_design_envelope(envelope)


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda envelope: envelope.update(extra=None), "keys are not exact"),
        (
            lambda envelope: envelope.update(envelope_schema="unknown"),
            "schema is unsupported",
        ),
        (lambda envelope: envelope.update(payload_sha256="A" * 64), "lowercase SHA-256"),
        (lambda envelope: envelope.update(payload=[]), "payload must be a plain dict"),
    ],
)
def test_envelope_validation_is_strict(
    mutate: Callable[[dict[str, object]], object],
    match: str,
) -> None:
    envelope = calibration_design_envelope()
    mutate(envelope)
    with pytest.raises((TypeError, ValueError), match=match):
        validate_calibration_design_envelope(envelope)


def test_canonical_json_rejects_floats_tuples_and_nonstring_keys() -> None:
    with pytest.raises(TypeError, match="float"):
        canonical_json_bytes({"value": 0.5})
    with pytest.raises(TypeError, match="tuple"):
        canonical_json_bytes({"value": (1, 2)})
    with pytest.raises(TypeError, match="non-string mapping key"):
        canonical_json_bytes({1: "value"})
    assert canonical_json_bytes({"b": 2, "a": 1}) == b'{"a":1,"b":2}'


def test_dataclasses_reject_bool_aliases_and_semantic_mutations() -> None:
    with pytest.raises(ValueError, match="strict uint32"):
        FrozenSeedPair(0, True, 3)
    with pytest.raises(ValueError, match="round-robin"):
        CalibrationAssignment(0, CALIBRATION_MANIFEST_ORDER[1])
    with pytest.raises(ValueError, match="semantics"):
        FactorialCell("SE", "writable_evidence", "selective", "evidence")
    with pytest.raises(ValueError, match="case index"):
        MatchedCalibrationCase(
            case_index=1,
            seed_index=0,
            manifest_name=CALIBRATION_MANIFEST_ORDER[0],
            world_seed=FROZEN_SEED_PAIRS[0].world_seed,
            learner_seed=FROZEN_SEED_PAIRS[0].learner_seed,
            condition="selective_full",
        )
    with pytest.raises(ValueError, match="diagnostic metrics"):
        MetricContract(
            metric_id="bad",
            role="diagnostic",
            orientation="higher",
            gate_mode="contrast_only",
            source_fields=("x",),
            aggregation="x",
            eligibility="x",
            missingness="x",
            null_value_decimal=None,
        )


def test_design_dataclass_rejects_reordered_or_incomplete_cases() -> None:
    design = build_hidden_regime_factorial_calibration_design()
    with pytest.raises(ValueError, match="240 matched cases"):
        dataclasses.replace(design, cases=design.cases[:-1])
    swapped = (design.cases[1], design.cases[0], *design.cases[2:])
    with pytest.raises(ValueError, match="Cartesian schedule"):
        dataclasses.replace(design, cases=swapped)
