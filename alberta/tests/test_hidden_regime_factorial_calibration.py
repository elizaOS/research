"""Fail-closed tests for hidden-regime calibration execution and aggregation."""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any, cast

import pytest
from scipy.stats import t as student_t

import alberta_framework.evaluation.hidden_regime_factorial_calibration as calibration
from alberta_framework.evaluation.hidden_regime_factorial_protocol import (
    CALIBRATION_DESIGN_PAYLOAD_SHA256,
    CALIBRATION_MANIFEST_ORDER,
    CANONICAL_CONDITION_ORDER,
    EstimandContract,
    MetricContract,
    build_hidden_regime_factorial_calibration_design,
    canonical_json_bytes,
)

pytestmark = pytest.mark.unit


def _record(
    identity: tuple[int, int, int, int],
    *,
    qualified: bool = True,
    selected: bool = True,
    errors: int = 0,
) -> dict[str, object]:
    segment, regime, occurrence, raw_occurrence = identity
    selected = selected and qualified
    probe = {
        "acquisition_qualified": True,
        "synchronized_generation_survives": True,
        "entry_composed_greedy_accuracy": 1.0,
    }
    dormant = {
        "composed_greedy_accuracy": 1.0,
        "zero_helper_accuracy": 1.0 / 3.0,
        "zero_beneficiary_accuracy": 1.0 / 3.0,
        "role_swapped_accuracy": 1.0 / 3.0,
    }
    return {
        "segment_index": segment,
        "regime_id": regime,
        "occurrence_index": occurrence,
        "raw_segment_occurrence_index": raw_occurrence,
        "lineage_retention_applicable": qualified,
        "acquisition_coverage_failure": not qualified,
        "first_world_window_complete": True,
        "first_world_window_errors": errors,
        "first_world_window_length": 16,
        "latest_prior_qualified_survived": True if qualified else None,
        "any_prior_qualified_survived": True if qualified else None,
        "selected_lineage_available": selected,
        "selected_lineage_joint_bit_exact_preserved": True if selected else None,
        "selected_exact_generation_relock_observed": True if selected else None,
        "selected_durable_retrieval_before_scratch": True if selected else None,
        "selected_lineage_entry_composed_greedy_accuracy": 1.0 if selected else None,
        "selected_lineage_entry_minus_commit_accuracy": 0.0 if selected else None,
        "latest_qualified_acquisition_comparison_available": qualified,
        "recurrence_minus_latest_qualified_acquisition_error_rate_delta": (
            errors / 16.0 if qualified else None
        ),
        "prior_same_regime_lineages": [probe] if qualified else [],
        "eligible_dormant_generations": [dormant],
        "best_dormant_composed_greedy_accuracy": 1.0,
        "best_dormant_zero_helper_accuracy": 1.0 / 3.0,
        "best_dormant_zero_beneficiary_accuracy": 1.0 / 3.0,
        "best_dormant_role_swapped_accuracy": 1.0 / 3.0,
        "selected_lineage_entry_activity_status": "dormant" if selected else None,
    }


def _summary(records: list[dict[str, object]], *, reward: float = 0.75) -> dict[str, object]:
    return {"mean_prequential_reward": reward, "recurrence_retention": records}


def _metric(metric_id: str, *, orientation: str = "higher") -> MetricContract:
    return MetricContract(
        metric_id=metric_id,
        role="primary",
        orientation=cast(Any, orientation),
        gate_mode="level_and_contrast",
        source_fields=("summary.test",),
        aggregation="test",
        eligibility="test",
        missingness="test",
        null_value_decimal="0",
    )


def _estimand(metric_id: str) -> EstimandContract:
    return EstimandContract(
        estimand_id="test_pair",
        role="primary",
        formula="test",
        condition_terms=(("selective_full", 1), ("writable_evidence", -1)),
        population_rule="test",
        metrics=(metric_id,),
    )


def _compact_shard(records: list[dict[str, object]], *, reward: float = 0.75) -> dict[str, object]:
    return {"summary": _summary(records, reward=reward)}


def test_protocol_digest_and_exact_cardinality_are_bound() -> None:
    design = calibration._design()
    assert len(design.cases) == 240
    assert len(design.seed_pairs) == 30
    assert len(design.condition_runtime_bindings) == 8
    assert CALIBRATION_DESIGN_PAYLOAD_SHA256 == (
        "735ceb533717e8b71c0159372b44041b2fd533ec14b62e78234de2c3552dd47d"
    )


def test_float_hex_round_trip_is_exact_and_nonfinite_fails() -> None:
    value = math.nextafter(1.0, 2.0)
    encoded = calibration._float_hex(value, "value")
    assert calibration._parse_float_hex(encoded, "value") == value
    with pytest.raises(calibration.CalibrationError, match="canonical"):
        calibration._parse_float_hex("0x1p+0", "value")
    with pytest.raises(calibration.CalibrationError, match="finite"):
        calibration._float_hex(float("nan"), "value")
    with pytest.raises(calibration.CalibrationError, match="finite"):
        calibration._float_hex(float("inf"), "value")


def test_payload_digest_rejects_outcome_tamper() -> None:
    payload = calibration._payload_with_digest({"schema": "test", "value_hex": (1.0).hex()})
    assert calibration._validate_payload_digest(payload, "test")["value_hex"] == (1.0).hex()
    tampered = dict(payload)
    tampered["value_hex"] = (0.0).hex()
    with pytest.raises(calibration.CalibrationError, match="digest mismatch"):
        calibration._validate_payload_digest(tampered, "test")


def test_all_predeclared_metrics_have_direct_extractors() -> None:
    design = build_hidden_regime_factorial_calibration_design()
    records = [_record((3, 0, 1, 1))]
    summary = _summary(records)
    observations = [
        calibration._metric_observation(contract, records, summary) for contract in design.metrics
    ]
    assert tuple(item["metric_id"] for item in observations) == tuple(
        contract.metric_id for contract in design.metrics
    )
    assert all(item["structural_missing_n"] == 0 for item in observations)


def test_qualified_failures_are_zero_not_best_lineage_substitution() -> None:
    records = [
        _record((3, 0, 1, 1), qualified=True, selected=False),
        _record((5, 1, 1, 1), qualified=False, selected=False),
    ]
    value, eligible, observed = calibration._metric_value(
        "selected_lineage_joint_bit_exact_preservation_rate",
        records,
        _summary(records),
    )
    assert value == 0.0
    assert eligible == ((3, 0, 1, 1),)
    assert observed == eligible


def test_paired_metric_uses_exact_qualification_intersection() -> None:
    identity_a = (3, 0, 1, 1)
    identity_b = (5, 1, 1, 1)
    conditions = {
        "selective_full": _compact_shard(
            [
                _record(identity_a, qualified=True, errors=0),
                _record(identity_b, qualified=False, errors=0),
            ]
        ),
        "writable_evidence": _compact_shard(
            [
                _record(identity_a, qualified=True, errors=16),
                _record(identity_b, qualified=True, errors=16),
            ]
        ),
    }
    paired = calibration._paired_seed_metric(
        _metric("qualified_first_entry_window_error_rate", orientation="lower"),
        _estimand("qualified_first_entry_window_error_rate"),
        conditions,
    )
    assert paired["included_recurrence_identities"] == [list(identity_a)]
    assert paired["excluded_recurrence_identities"] == [list(identity_b)]
    assert calibration._parse_float_hex(paired["oriented_delta_hex"], "delta") == 1.0


def test_paired_metric_rejects_misaligned_recurrence_identity() -> None:
    left = _compact_shard([_record((3, 0, 1, 1))])
    right = _compact_shard([_record((4, 0, 1, 1))])
    with pytest.raises(calibration.CalibrationError, match="populations differ"):
        calibration._paired_seed_metric(
            _metric("qualified_first_entry_window_error_rate", orientation="lower"),
            _estimand("qualified_first_entry_window_error_rate"),
            {"selective_full": left, "writable_evidence": right},
        )


def test_statistics_use_sample_sd_se_and_one_sided_t_bound() -> None:
    values = [0.25, 0.5, 0.75, 1.0]
    result = calibration._sample_statistics(values, null=0.0)
    mean = sum(values) / len(values)
    sample_sd = math.sqrt(sum((value - mean) ** 2 for value in values) / 3)
    se = sample_sd / math.sqrt(4)
    expected_bound = mean - float(student_t.ppf(0.95, 3)) * se
    assert calibration._parse_float_hex(result["mean_hex"], "mean") == mean
    assert calibration._parse_float_hex(
        result["sample_standard_deviation_hex"], "sd"
    ) == pytest.approx(sample_sd, rel=0.0, abs=2e-16)
    assert calibration._parse_float_hex(result["standard_error_hex"], "se") == pytest.approx(
        se, rel=0.0, abs=2e-16
    )
    assert calibration._parse_float_hex(
        result["one_sided_95_percent_lower_confidence_bound_hex"], "bound"
    ) == pytest.approx(expected_bound, rel=0.0, abs=2e-16)
    assert (result["wins"], result["ties"], result["losses"]) == (4, 0, 0)


def test_strata_are_exact_n30_and_n10_with_explicit_missingness() -> None:
    rows: list[tuple[int, str, float | None, bool, bool]] = []
    for seed in range(30):
        manifest = CALIBRATION_MANIFEST_ORDER[seed % 3]
        if seed == 0:
            rows.append((seed, manifest, None, True, False))
        elif seed == 1:
            rows.append((seed, manifest, None, False, True))
        else:
            rows.append((seed, manifest, float(seed), False, False))
    result = calibration._stratified_statistics(rows, null=0.0)
    pooled = cast(dict[str, object], result["pooled"])
    assert pooled["eligible_n"] == 30
    assert pooled["observed_n"] == 28
    assert pooled["conditional_unobserved_seed_indices"] == [0]
    assert pooled["structural_missing_seed_indices"] == [1]
    assert [cast(dict[str, object], item)["eligible_n"] for item in result["by_manifest"]] == [
        10,
        10,
        10,
    ]


def test_worker_result_parser_rejects_trailing_or_noncanonical_data() -> None:
    payload = calibration._payload_with_digest({"schema": "synthetic"})
    raw = canonical_json_bytes(payload)
    encoded = calibration.WORKER_RESULT_PREFIX + calibration.base64.b64encode(raw)
    assert calibration._parse_worker_result(encoded) == payload
    with pytest.raises(calibration.CalibrationError, match="output differs"):
        calibration._parse_worker_result(encoded + b"\n")


def test_coordinator_refuses_execution_without_explicit_boolean(tmp_path: Path) -> None:
    with pytest.raises(calibration.CalibrationError, match="explicit authorization"):
        calibration.run_calibration_case_subprocess(
            case_index=0,
            readiness_directory=tmp_path / "missing-readiness",
            managed_ledger_directory=tmp_path / "missing-ledger",
            shard_publication_root=tmp_path,
            explicit_acknowledgement=calibration.EXECUTION_ACKNOWLEDGEMENT,
            authorize_calibration_execution=False,
        )


def test_request_refuses_wrong_acknowledgement_before_any_execution() -> None:
    with pytest.raises(calibration.CalibrationError, match="exact explicit"):
        calibration.build_calibration_case_request(
            0,
            cast("calibration.ValidatedReadinessBundle", object()),
            managed_ledger_directory=Path("unused"),
            explicit_acknowledgement="yes",
        )


def test_immutable_reader_rejects_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    target.write_bytes(b"{}")
    target.chmod(0o444)
    link = tmp_path / "link.json"
    link.symlink_to(target)
    with pytest.raises(calibration.CalibrationError, match="symlinked"):
        calibration._read_regular_file(link, max_bytes=100, label="symlink")


def test_immutable_reader_rejects_symlinked_ancestor(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    target = real / "target.json"
    target.write_bytes(b"{}")
    target.chmod(0o444)
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)
    with pytest.raises(calibration.CalibrationError, match="directory component"):
        calibration._read_regular_file(linked / "target.json", max_bytes=100, label="ancestor")


def test_new_only_shard_duplicate_must_be_byte_identical(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    readiness = {"readiness_receipt_sha256": "a" * 64}
    body = {
        "case": {"case_index": 0},
        "readiness_binding": readiness,
    }
    shard = calibration._payload_with_digest(body)
    monkeypatch.setattr(
        calibration,
        "validate_calibration_case_shard",
        lambda payload, **_: dict(payload),
    )
    path = calibration.publish_calibration_case_shard_new_only(
        tmp_path,
        shard,
        expected_readiness_binding=readiness,
    )
    assert path.stat().st_mode & 0o777 == 0o444
    assert (
        calibration.publish_calibration_case_shard_new_only(
            tmp_path,
            shard,
            expected_readiness_binding=readiness,
        )
        == path
    )
    tampered = calibration._payload_with_digest(
        {"case": {"case_index": 0}, "readiness_binding": readiness, "extra": True}
    )
    with pytest.raises(calibration.CalibrationError, match="not byte-identical"):
        calibration.publish_calibration_case_shard_new_only(
            tmp_path,
            tampered,
            expected_readiness_binding=readiness,
        )


def test_shard_publication_rejects_symlinked_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    readiness = {"readiness_receipt_sha256": "a" * 64}
    shard = calibration._payload_with_digest(
        {"case": {"case_index": 0}, "readiness_binding": readiness}
    )
    real_root = tmp_path / "real-root"
    real_root.mkdir()
    linked_root = tmp_path / "linked-root"
    linked_root.symlink_to(real_root, target_is_directory=True)
    monkeypatch.setattr(
        calibration,
        "validate_calibration_case_shard",
        lambda payload, **_: dict(payload),
    )
    with pytest.raises(calibration.CalibrationError, match="directory component"):
        calibration.publish_calibration_case_shard_new_only(
            linked_root,
            shard,
            expected_readiness_binding=readiness,
        )


def test_aggregate_publication_is_new_only_content_addressed_and_symlink_safe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    digest = "a" * 64
    aggregate = {"schema": "synthetic", "payload_sha256": digest, "value": 1}
    monkeypatch.setattr(
        calibration,
        "validate_calibration_aggregate",
        lambda payload, shards, **kwargs: dict(payload),
    )
    published = calibration.publish_calibration_aggregate_new_only(
        tmp_path,
        aggregate,
        [],
        managed_ledger_snapshot={},
        managed_ledger_directory=tmp_path,
        authorize_publication=True,
    )
    assert published.path.name == f"{digest}.json"
    assert published.path.stat().st_mode & 0o777 == 0o444
    duplicate = calibration.publish_calibration_aggregate_new_only(
        tmp_path,
        aggregate,
        [],
        managed_ledger_snapshot={},
        managed_ledger_directory=tmp_path,
        authorize_publication=True,
    )
    assert duplicate.path == published.path
    with pytest.raises(calibration.CalibrationError, match="not byte-identical"):
        calibration.publish_calibration_aggregate_new_only(
            tmp_path,
            {**aggregate, "value": 2},
            [],
            managed_ledger_snapshot={},
            managed_ledger_directory=tmp_path,
            authorize_publication=True,
        )

    real_root = tmp_path / "aggregate-real"
    real_root.mkdir()
    linked_root = tmp_path / "aggregate-linked"
    linked_root.symlink_to(real_root, target_is_directory=True)
    with pytest.raises(calibration.CalibrationError, match="directory component"):
        calibration.publish_calibration_aggregate_new_only(
            linked_root,
            aggregate,
            [],
            managed_ledger_snapshot={},
            managed_ledger_directory=tmp_path,
            authorize_publication=True,
        )


def test_exact_240_ledger_rejects_missing_and_duplicate_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(calibration.CalibrationError, match="exactly 240"):
        calibration._validated_shard_bodies([])
    duplicate = calibration._payload_with_digest(
        {"case": {"case_index": 0}, "readiness_binding": {}}
    )
    monkeypatch.setattr(
        calibration,
        "validate_calibration_case_shard",
        lambda payload, **_: dict(payload),
    )
    with pytest.raises(calibration.CalibrationError, match="duplicate shard payload"):
        calibration._validated_shard_bodies([copy.deepcopy(duplicate) for _ in range(240)])


def test_execution_record_binding_includes_direct_component_digests() -> None:
    inventory = {
        "genesis_sha256": "1" * 64,
        "started_records": [
            {
                "case_index": 7,
                "started_record_sha256": "2" * 64,
            }
        ],
        "completed_records": [
            {
                "case_index": 7,
                "started_record_sha256": "2" * 64,
                "completed_record_sha256": "3" * 64,
                "summary_sha256": "4" * 64,
                "resource_sha256": "5" * 64,
                "primitive_trace_sha256": "6" * 64,
                "outcome_sha256": "7" * 64,
            }
        ],
    }
    assert calibration._execution_record_binding(inventory, 7) == {
        "case_index": 7,
        "genesis_sha256": "1" * 64,
        "started_record_sha256": "2" * 64,
        "completed_record_sha256": "3" * 64,
        "summary_sha256": "4" * 64,
        "resource_sha256": "5" * 64,
        "primitive_trace_sha256": "6" * 64,
        "outcome_sha256": "7" * 64,
    }


def test_batch_coordinator_refuses_defaults_before_touching_readiness(tmp_path: Path) -> None:
    with pytest.raises(calibration.CalibrationError, match="explicit authorization"):
        calibration.run_calibration_cases_subprocess(
            case_indices=range(2),
            max_workers=1,
            readiness_directory=tmp_path / "missing",
            managed_ledger_directory=tmp_path / "missing",
            shard_publication_root=tmp_path,
        )


def test_batch_coordinator_returns_caller_order_without_retries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []

    def run_case(**kwargs: object) -> dict[str, object]:
        index = cast(int, kwargs["case_index"])
        calls.append(index)
        return {"case_index": index}

    monkeypatch.setattr(calibration, "run_calibration_case_subprocess", run_case)
    shards = calibration.run_calibration_cases_subprocess(
        case_indices=(2, 0, 1),
        max_workers=2,
        readiness_directory=tmp_path,
        managed_ledger_directory=tmp_path,
        shard_publication_root=tmp_path,
        explicit_acknowledgement=calibration.EXECUTION_ACKNOWLEDGEMENT,
        authorize_calibration_execution=True,
    )
    assert tuple(item["case_index"] for item in shards) == (2, 0, 1)
    assert sorted(calls) == [0, 1, 2]


def test_batch_failure_does_not_start_a_later_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []

    def run_case(**kwargs: object) -> dict[str, object]:
        index = cast(int, kwargs["case_index"])
        calls.append(index)
        if index == 1:
            raise RuntimeError("synthetic failure")
        return {"case_index": index}

    monkeypatch.setattr(calibration, "run_calibration_case_subprocess", run_case)
    with pytest.raises(calibration.CalibrationError, match="without retry or substitution"):
        calibration.run_calibration_cases_subprocess(
            case_indices=(0, 1, 2),
            max_workers=2,
            readiness_directory=tmp_path,
            managed_ledger_directory=tmp_path,
            shard_publication_root=tmp_path,
            explicit_acknowledgement=calibration.EXECUTION_ACKNOWLEDGEMENT,
            authorize_calibration_execution=True,
        )
    assert sorted(calls) == [0, 1]


def test_preflight_refuses_defaults_before_touching_readiness(tmp_path: Path) -> None:
    with pytest.raises(calibration.CalibrationError, match="explicit authorization"):
        calibration.run_calibration_preflight_subprocess(
            readiness_directory=tmp_path / "missing",
            managed_ledger_directory=tmp_path / "missing",
        )


def test_preflight_case_rows_cover_all_frozen_bindings_without_seed_material(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    readiness = {"execution_governance": {"genesis_sha256": "a" * 64}}
    request = calibration._payload_with_digest(
        {
            "readiness_binding": readiness,
            "managed_ledger_genesis_sha256": "a" * 64,
        }
    )
    monkeypatch.setattr(
        calibration,
        "validate_calibration_case_request",
        lambda payload, bundle: cast(Any, payload),
    )
    rows = calibration._preflight_case_binding_rows(request, cast(Any, object()))
    assert tuple(item["case_index"] for item in rows) == tuple(range(240))
    assert len({cast(str, item["case_request_payload_sha256"]) for item in rows}) == 240
    assert all(
        set(item)
        == {
            "case_index",
            "case_binding_sha256",
            "condition_runtime_binding_sha256",
            "manifest_binding_sha256",
            "recurrence_binding_sha256",
            "configuration_sha256",
            "seed_pair_binding_sha256",
            "case_request_payload_sha256",
        }
        for item in rows
    )


def test_zip_preflight_never_calls_learner_and_leaves_inventory_equal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inventory = {
        "schema": calibration.CALIBRATION_EXECUTION_INVENTORY_SCHEMA,
        "genesis_sha256": "a" * 64,
        "expected_case_count": 240,
        "started_case_indices": [],
        "completed_case_indices": [],
        "interrupted_case_indices": [],
        "started_record_count": 0,
        "completed_record_count": 0,
        "protected_started_record_count": 0,
        "protected_completed_record_count": 0,
        "pristine": True,
        "started_records": [],
        "completed_records": [],
        "managed_boundary_scope": "synthetic",
        "inventory_sha256": "b" * 64,
    }
    request = calibration._payload_with_digest(
        {
            "managed_ledger_genesis_sha256": "a" * 64,
            "pristine_inventory_sha256": "b" * 64,
            "issue_process_local_authorizations": True,
        }
    )
    bundle = type(
        "Bundle",
        (),
        {
            "receipt_sha256": "c" * 64,
            "source_archive_sha256": "d" * 64,
            "source_manifest_sha256": "e" * 64,
        },
    )()
    rows = tuple({"case_request_payload_sha256": f"{index:064x}"} for index in range(240))
    issued: list[int] = []
    monkeypatch.setattr(calibration, "load_validated_readiness_bundle", lambda *a, **k: bundle)
    monkeypatch.setattr(
        calibration,
        "validate_calibration_preflight_request",
        lambda payload, validated_bundle: dict(payload),
    )
    monkeypatch.setattr(
        calibration,
        "snapshot_calibration_execution_inventory",
        lambda directory: copy.deepcopy(inventory),
    )
    monkeypatch.setattr(
        calibration,
        "require_valid_calibration_execution_inventory",
        lambda snapshot, directory: dict(snapshot),
    )
    monkeypatch.setattr(
        calibration,
        "_zip_worker_provenance",
        lambda archive, validated_bundle: {"source_archive_sha256": "d" * 64},
    )
    monkeypatch.setattr(calibration, "_read_regular_file", lambda *a, **k: b"zip")
    monkeypatch.setattr(
        calibration,
        "_preflight_case_binding_rows",
        lambda payload, validated_bundle: rows,
    )
    monkeypatch.setattr(
        calibration,
        "issue_calibration_execution_authorization",
        lambda **kwargs: issued.append(cast(int, kwargs["case_index"])),
    )
    monkeypatch.setattr(
        calibration,
        "run_hidden_regime_condition",
        lambda *a, **k: pytest.fail("preflight called learner"),
    )
    report = calibration._worker_preflight(
        readiness_directory=tmp_path,
        ledger_directory=tmp_path,
        request_payload=request,
    )
    body = calibration._validate_payload_digest(report, "preflight report")
    assert issued == list(range(240))
    assert body["learner_execution_called"] is False
    assert body["outcome_observed"] is False
    assert body["authorization_material_serialized"] is False
    assert body["inventory_byte_equal_before_after"] is True


def test_gate_matrix_never_decides_while_thresholds_are_unset() -> None:
    design = build_hidden_regime_factorial_calibration_design()
    levels = [
        {"condition": condition, "metric_id": metric.metric_id, "statistics": {}}
        for condition in CANONICAL_CONDITION_ORDER
        for metric in design.metrics
    ]
    estimands = [
        {
            "estimand_id": estimand.estimand_id,
            "metrics": [
                {"metric_id": metric_id, "statistics": {}} for metric_id in estimand.metrics
            ],
        }
        for estimand in design.factorial_estimands + design.control_estimands
    ]
    supports = [
        {
            "metric_id": support.metric_id,
            "estimand_id": support.estimand_id,
            "statistics": {},
        }
        for support in design.paired_population_support_metrics
    ]
    mandatory, descriptive = calibration._gate_result_matrix(
        design,
        levels,
        estimands,
        supports,
    )
    assert len(mandatory) + len(descriptive) == len(design.gate_families)
    assert all(item["decision"] == "not_evaluated_no_thresholds" for item in mandatory)
    assert all(
        item["threshold_status"] == "unset_pending_consumed_calibration_outcomes"
        for item in mandatory + descriptive
    )
