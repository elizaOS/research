# mypy: disable-error-code="arg-type,call-arg,index,type-var"
"""Integrity-bound report and exact replay tests for consolidated-memory transfer."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from typing import cast

import pytest

from alberta_framework.evaluation.consolidated_memory_transfer import (
    CONSOLIDATED_MEMORY_TRANSFER_REPORT_SCHEMA,
    build_consolidated_memory_transfer_report,
    canonical_consolidated_memory_transfer_report_bytes,
    load_consolidated_memory_transfer_report_bytes,
    reconstruct_consolidated_memory_transfer_summary,
    validate_consolidated_memory_transfer_report,
)

pytestmark = [pytest.mark.integration, pytest.mark.slow]


@pytest.fixture(scope="module")
def report() -> dict[str, object]:
    return build_consolidated_memory_transfer_report()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _payload(report: Mapping[str, object]) -> dict[str, object]:
    return cast(dict[str, object], report["payload"])


def test_report_is_not_assessed_and_reconstructs_requested_stress_metrics(
    report: dict[str, object],
) -> None:
    assert report["schema"] == CONSOLIDATED_MEMORY_TRANSFER_REPORT_SCHEMA
    payload = _payload(report)
    assert payload["development_status"] == "development-only-not-assessed"
    assert payload["assessment_status"] == "not-assessed"
    assert payload["performance_thresholds_applied"] is False
    assert payload["promotion_authority"] is False
    assert payload["scientific_promotion_allowed"] is False

    summary = cast(dict[str, object], payload["summary"])
    retrieval = cast(dict[str, object], summary["retrieval"])
    assert retrieval["opportunities"] == 17
    assert retrieval["accepted"] == 7
    assert retrieval["exact_target_matches"] == 5
    assert retrieval["exact_match_precision"] == pytest.approx(5.0 / 7.0)
    assert retrieval["abstentions"] == 10
    assert retrieval["provenance_mismatches_on_accepted_retrieval"] == 0
    assert cast(dict[str, int], retrieval["abstention_reasons"]) == {
        "generation-mismatch": 2,
        "identity-missing": 6,
        "provenance-mismatch": 1,
        "stale-or-invalidated": 1,
    }

    harm = cast(dict[str, object], summary["harm"])
    assert harm["events"] == 2
    assert harm["total_excess_squared_error"] == pytest.approx(6.0)
    assert harm["event_ids"] == [
        "interference-success-failure-shift",
        "interference-misleading-probe",
    ]
    transfer = cast(dict[str, object], summary["forward_transfer_and_recovery"])
    assert transfer["semantic_generation_shift_gain"] == 0.0
    assert transfer["semantic_new_generation_recurrence_gain"] == 4.0
    assert transfer["semantic_return_to_recurrence_recovery"] == 4.0
    assert transfer["procedural_stale_to_recurrence_recovery"] == 1.0
    retained = cast(dict[str, object], summary["retained_semantic_utility"])
    assert retained["return_signed_gain"] == 0.0
    assert retained["recurrence_signed_gain"] == 4.0
    stale = cast(dict[str, object], summary["stale_skill_harm"])
    assert stale == {
        "events": 1,
        "accepted_stale_retrievals": 0,
        "harm_events": 0,
        "total_excess_squared_error": 0.0,
    }
    eviction = cast(dict[str, object], summary["eviction_and_provenance"])
    assert eviction["replacements"] == 3
    assert eviction["provenance_gate_abstentions"] == 1
    assert len(cast(list[str], eviction["evicted_provenance_sha256"])) == 3


def test_trace_proves_query_before_write_and_matched_comparators(
    report: dict[str, object],
) -> None:
    payload = _payload(report)
    trace = cast(list[dict[str, object]], payload["raw_trace"])
    assert len(trace) == 17
    assert all(event["query_precedes_write"] is True for event in trace)
    assert all(event["evaluator_annotations_visible_to_memory"] is False for event in trace)
    assert all(
        cast(dict[str, object], event["comparators"])["matched_external_event"] is True
        for event in trace
    )
    assert all(cast(dict[str, object], event["write"])["wrote"] is True for event in trace)
    assert [event["post_operation_count"] for event in trace] == list(range(1, 18))

    misleading = next(event for event in trace if event["role"] == "misleading-probe")
    assert cast(dict[str, object], misleading["retrieval"])["accepted"] is True
    assert misleading["retrieval_harm"] is True
    stale = next(event for event in trace if event["role"] == "stale-skill-probe")
    assert cast(dict[str, object], stale["retrieval"])["accepted"] is False
    assert (
        cast(dict[str, object], stale["retrieval"])["abstention_reason"] == "stale-or-invalidated"
    )
    provenance = next(event for event in trace if event["role"] == "provenance-mismatch")
    assert (
        cast(dict[str, object], provenance["retrieval"])["abstention_reason"]
        == "provenance-mismatch"
    )
    reconstructed = reconstruct_consolidated_memory_transfer_summary(trace)
    assert reconstructed == payload["summary"]


def test_resources_are_exact_matched_and_declare_no_memory_difference(
    report: dict[str, object],
) -> None:
    payload = _payload(report)
    resources = cast(dict[str, object], payload["resource_accounting"])
    execution = cast(dict[str, object], payload["execution"])
    snapshot = cast(dict[str, object], payload["initial_snapshot"])
    assert snapshot["empty"] is True
    assert snapshot["source_bound"] is True
    assert resources["persistent_state_bytes_per_memory_arm"] == snapshot["state_bytes"]
    assert resources["logical_memory_arm_count"] == 2
    assert resources["logical_memory_bytes_across_full_and_ablation"] == (
        2 * cast(int, snapshot["state_bytes"])
    )
    assert resources["no_memory_persistent_state_bytes"] == 0
    assert resources["dynamic_persistent_growth_bytes"] == 0
    assert resources["external_event_count_per_arm"] == 17
    assert resources["query_opportunities_per_arm"] == 17
    assert resources["write_opportunities_per_arm"] == 17
    assert resources["full_memory_kernel_calls"] == 17
    assert resources["retrieval_ablation_kernel_calls"] == 17
    assert resources["no_memory_kernel_calls"] == 0
    assert resources["compiled_parity_diagnostic_kernel_calls"] == 17
    assert resources["total_physical_memory_kernel_calls"] == 51
    assert resources["matched_external_experience"] is True
    assert resources["matched_full_and_ablation_compute"] is True
    assert resources["no_memory_compute_difference_declared"] is True
    assert resources["evaluator_random_generator_calls"] == 0
    assert resources["agent_parameter_mutations"] == 0
    assert resources["action_selection_calls"] == 0
    assert resources["promotion_decisions"] == 0
    full = cast(dict[str, int], resources["full_memory_accounting"])
    ablation = cast(dict[str, int], resources["retrieval_ablation_accounting"])
    assert full == ablation
    assert full["operation_count"] == 17
    assert full["semantic_writes"] + full["procedural_writes"] == 17
    full_lifetime = cast(dict[str, int], resources["full_memory_lifetime_counters"])
    ablation_lifetime = cast(dict[str, int], resources["retrieval_ablation_lifetime_counters"])
    assert full_lifetime == ablation_lifetime
    assert full_lifetime["semantic_retirement_count"] == 1
    assert full_lifetime["procedural_retirement_count"] == 1
    assert full_lifetime["semantic_replacement_count"] == 3
    assert execution["compiled_schedule_parity_checked"] is True
    assert execution["compiled_schedule_parity_exact"] is True
    assert execution["full_ablation_state_parity_exact"] is True
    assert execution["external_snapshot_mutations"] == 0
    limitations = cast(list[str], payload["limitations"])
    assert any("signed-int32 max_operations" in value for value in limitations)


def test_report_integrity_canonical_roundtrip_and_exact_replay(
    report: dict[str, object],
) -> None:
    validation = validate_consolidated_memory_transfer_report(report)
    assert validation.valid
    assert validation.status == "not-assessed"
    assert validation.exact_replay_checked
    assert validation.exact_replay_matches
    encoded = canonical_consolidated_memory_transfer_report_bytes(report)
    assert len(encoded) == cast(
        int, _payload(report)["resource_accounting"]["canonical_report_bytes"]
    )
    restored = load_consolidated_memory_transfer_report_bytes(encoded)
    assert restored == report


def test_validator_rejects_hash_source_protocol_runtime_and_rehashed_trace_tamper(
    report: dict[str, object],
) -> None:
    bad_hash = copy.deepcopy(report)
    bad_hash["payload_sha256"] = "0" * 64
    assert not validate_consolidated_memory_transfer_report(bad_hash).valid

    source = copy.deepcopy(report)
    source_payload = _payload(source)
    source_bindings = cast(dict[str, object], source_payload["bindings"])
    source_bindings["source_manifest_sha256"] = "0" * 64
    source["payload_sha256"] = _canonical_sha256(source_payload)
    assert not validate_consolidated_memory_transfer_report(source).valid

    runtime = copy.deepcopy(report)
    runtime_payload = _payload(runtime)
    runtime_bindings = cast(dict[str, object], runtime_payload["bindings"])
    runtime_bindings["runtime_sha256"] = "f" * 64
    runtime["payload_sha256"] = _canonical_sha256(runtime_payload)
    assert not validate_consolidated_memory_transfer_report(runtime).valid

    protocol = copy.deepcopy(report)
    protocol_payload = _payload(protocol)
    protocol_body = cast(dict[str, object], protocol_payload["protocol"])
    events = cast(list[dict[str, object]], protocol_body["events"])
    events[0]["expected_target"] = [9.0, 9.0]
    protocol_body["external_experience_sha256"] = _canonical_sha256(events)
    protocol_bindings = cast(dict[str, object], protocol_payload["bindings"])
    protocol_bindings["protocol_sha256"] = _canonical_sha256(protocol_body)
    protocol["payload_sha256"] = _canonical_sha256(protocol_payload)
    assert not validate_consolidated_memory_transfer_report(protocol).valid

    replay_tamper = copy.deepcopy(report)
    replay_payload = _payload(replay_tamper)
    replay_trace = cast(list[dict[str, object]], replay_payload["raw_trace"])
    replay_trace[0]["record_payload"] = [8.0, -8.0]
    replay_tamper["payload_sha256"] = _canonical_sha256(replay_payload)
    validation = validate_consolidated_memory_transfer_report(replay_tamper)
    assert not validation.valid
    assert validation.exact_replay_checked
    assert not validation.exact_replay_matches


def test_strict_loader_rejects_noncanonical_and_duplicate_json(
    report: dict[str, object],
) -> None:
    encoded = canonical_consolidated_memory_transfer_report_bytes(report)
    with pytest.raises(ValueError, match="not canonical"):
        load_consolidated_memory_transfer_report_bytes(encoded + b"\n")
    duplicate = b'{"schema":"a","schema":"b"}'
    with pytest.raises(ValueError, match="duplicate"):
        load_consolidated_memory_transfer_report_bytes(duplicate)
