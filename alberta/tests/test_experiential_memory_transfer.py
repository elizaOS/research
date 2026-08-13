"""Development-only matched transfer diagnostics for experiential memory."""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import jax
import numpy as np
import pytest

from alberta_framework.core.checkpoints import save_checkpoint
from alberta_framework.core.experiential_memory import (
    ExperientialMemory,
    ExperientialMemoryState,
)
from alberta_framework.evaluation.experiential_memory_transfer import (
    EXPERIENTIAL_MEMORY_TRANSFER_CHECKPOINT_SCHEMA,
    EXPERIENTIAL_MEMORY_TRANSFER_REPORT_SCHEMA,
    MAX_ABSOLUTE_EVENTS,
    MAX_ABSOLUTE_PHASES,
    MAX_ABSOLUTE_REPORT_BYTES,
    MAX_ABSOLUTE_SNAPSHOT_BYTES,
    ExperientialMemoryTransferConfig,
    ExperientialMemoryTransferEvaluator,
    ExperientialMemoryTransferProtocol,
    build_experiential_memory_transfer_report,
    canonical_experiential_memory_transfer_report_bytes,
    default_experiential_memory_transfer_config,
    default_experiential_memory_transfer_protocol,
    experiential_memory_transfer_source_snapshot,
    frozen_experiential_memory_state_sha256,
    load_experiential_memory_transfer_report,
    load_experiential_memory_transfer_snapshot_checkpoint,
    save_experiential_memory_transfer_report,
    save_experiential_memory_transfer_snapshot_checkpoint,
    validate_experiential_memory_transfer_report,
)

pytestmark = pytest.mark.development


@pytest.fixture(scope="module")
def fixture() -> tuple[
    ExperientialMemory,
    ExperientialMemoryState,
    ExperientialMemoryTransferConfig,
    ExperientialMemoryTransferProtocol,
    dict[str, object],
]:
    config = default_experiential_memory_transfer_config()
    protocol = default_experiential_memory_transfer_protocol()
    memory = ExperientialMemory(config.memory_config)
    state = memory.init()
    report = build_experiential_memory_transfer_report(memory, state, config, protocol)
    return memory, state, config, protocol, report


def _assert_trees_equal(left: object, right: object) -> None:
    left_leaves = jax.tree.leaves(left)
    right_leaves = jax.tree.leaves(right)
    assert len(left_leaves) == len(right_leaves)
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        np.testing.assert_array_equal(np.asarray(left_leaf), np.asarray(right_leaf))


def _rehash(report: dict[str, object]) -> None:
    import hashlib

    payload = report["payload"]
    report["payload_sha256"] = hashlib.sha256(
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def test_config_protocol_and_evaluator_roundtrip_are_strict() -> None:
    config = default_experiential_memory_transfer_config()
    protocol = default_experiential_memory_transfer_protocol()

    restored_config = ExperientialMemoryTransferConfig.from_config(
        cast(dict[str, object], json.loads(json.dumps(config.to_config())))
    )
    restored_protocol = ExperientialMemoryTransferProtocol.from_config(
        cast(dict[str, object], json.loads(json.dumps(protocol.to_config())))
    )
    assert restored_config == config
    assert restored_protocol == protocol
    assert ExperientialMemoryTransferEvaluator(config, protocol).to_config() == {
        "config": config.to_config(),
        "protocol": protocol.to_config(),
    }

    serialized = protocol.to_config()
    assert serialized["learner_visible_fields"] == [
        "query_key",
        "representation_version",
        "query_uncertainty",
        "query_uncertainty_available",
        "entry",
    ]
    assert serialized["evaluator_only_fields"] == [
        "phase_id",
        "evaluator_regime_id",
        "case_id",
        "expected_outcome",
    ]
    assert serialized["regime_identifiers_visible_to_memory"] is False

    bad = dict(config.to_config())
    bad["success_threshold"] = 0.9
    with pytest.raises(ValueError, match="fields"):
        ExperientialMemoryTransferConfig.from_config(bad)


def test_protocol_is_exact_recurring_aba_and_rejects_identity_or_shape_drift() -> None:
    protocol = default_experiential_memory_transfer_protocol()
    assert [phase.evaluator_regime_id for phase in protocol.phases] == [
        "context-a",
        "context-b",
        "context-a",
    ]
    first, _, returned = protocol.phase_events()
    assert len(first) == len(returned) == 4
    assert [event.case_id for event in first] == [event.case_id for event in returned]
    assert [event.learner_case_config() for event in first] == [
        event.learner_case_config() for event in returned
    ]
    assert len({event.entry_provenance_id for event in protocol.events}) == len(
        protocol.events
    )

    duplicate = replace(
        protocol.events[-1],
        entry_provenance_id=protocol.events[0].entry_provenance_id,
    )
    with pytest.raises(ValueError, match="provenance"):
        replace(protocol, events=(*protocol.events[:-1], duplicate))

    drift = replace(protocol.events[-1], query_key=(9.0, 9.0))
    with pytest.raises(ValueError, match="exact learner-visible cases"):
        replace(protocol, events=(*protocol.events[:-1], drift))


def test_report_exercises_transfer_abstention_eviction_and_loophole_diagnostics(
    fixture: tuple[
        ExperientialMemory,
        ExperientialMemoryState,
        ExperientialMemoryTransferConfig,
        ExperientialMemoryTransferProtocol,
        dict[str, object],
    ],
) -> None:
    _, _, _, protocol, report = fixture
    assert report["schema"] == EXPERIENTIAL_MEMORY_TRANSFER_REPORT_SCHEMA
    validation = validate_experiential_memory_transfer_report(report)
    assert validation.valid, validation.errors
    assert validation.assessment_status == "not-assessed"

    payload = cast(dict[str, Any], report["payload"])
    assert payload["development_only"] is True
    assert payload["scientific_promotion_allowed"] is False
    assert payload["performance_thresholds_applied"] is False
    assert payload["retention_claimed"] is False
    assert payload["sota_claimed"] is False
    summary = payload["summary"]
    abstentions = summary["abstention_diagnostics"]
    assert abstentions["representation_version_mismatch_count"] >= 1
    assert abstentions["stale_count"] >= 1
    assert abstentions["unsafe_count"] >= 1
    assert abstentions["uncertain_or_unavailable_count"] >= 1
    assert summary["retrieval_diagnostics"]["accepted_count"] >= 2
    assert summary["negative_transfer"]["count"] >= 1
    assert summary["negative_transfer"]["total_excess_squared_error"] > 0.0
    assert summary["loophole_diagnostics"]["always_abstained"] is False
    assert summary["loophole_diagnostics"]["memory_prediction_constant"] is False
    assert summary["loophole_diagnostics"]["expected_outcome_constant"] is False
    assert summary["claims"] == {
        "transfer_established": False,
        "retention_established": False,
        "efficacy_established": False,
        "scientific_promotion_allowed": False,
        "performance_thresholds_applied": False,
        "sota_claimed": False,
    }
    assert len(summary["forward_transfer_descriptions"]) == 2
    assert len(summary["return_transfer_descriptions"]) == 1
    assert len(payload["raw_query_before_write_trace"]) == len(protocol.events)


def test_trace_binds_query_before_write_neighbors_and_negative_transfer(
    fixture: tuple[
        ExperientialMemory,
        ExperientialMemoryState,
        ExperientialMemoryTransferConfig,
        ExperientialMemoryTransferProtocol,
        dict[str, object],
    ],
) -> None:
    _, _, _, protocol, report = fixture
    trace = cast(dict[str, Any], report["payload"])["raw_query_before_write_trace"]
    assert trace[0]["query_before_write"] is True
    assert trace[0]["retrieval"]["accepted"] is False
    for event, record in zip(protocol.events, trace, strict=True):
        assert event.entry_provenance_id not in [
            provenance
            for provenance, used in zip(
                record["retrieval"]["neighbor_provenance_ids"],
                record["retrieval"]["neighbor_mask"],
                strict=True,
            )
            if used
        ]
        assert record["memory_prediction"]["used_retrieval"] is record["retrieval"][
            "accepted"
        ]

    harmful = [record for record in trace if record["negative_transfer"]["occurred"]]
    assert harmful
    for record in harmful:
        assert record["negative_transfer"]["excess_squared_error"] > 0.0
        assert record["retrieval"]["accepted"] is True
        assert record["retrieval_provenance"]["correct_neighbor_weight"] < 1.0


def test_operation_and_byte_accounting_are_hard_bounded_and_matched(
    fixture: tuple[
        ExperientialMemory,
        ExperientialMemoryState,
        ExperientialMemoryTransferConfig,
        ExperientialMemoryTransferProtocol,
        dict[str, object],
    ],
) -> None:
    memory, _, config, protocol, report = fixture
    resources = cast(dict[str, Any], report["payload"])["resource_accounting"]
    event_count = len(protocol.events)
    assert resources["recorded_event_count"] == event_count
    assert resources["memory_query_opportunities"] == event_count
    assert resources["memory_write_opportunities"] == event_count
    assert resources["reference_query_opportunities"] == event_count
    assert resources["reference_write_opportunities"] == event_count
    assert resources["matched_event_query_write_opportunity_budgets"] is True
    assert resources["reference_persistent_state_bytes"] == 0
    assert resources["memory_persistent_state_bytes"] == memory.persistent_bytes
    assert resources["memory_capacity_entries"] == memory.config.capacity
    assert resources["memory_evictions"] >= 1
    assert resources["canonical_report_bytes"] <= config.max_report_bytes
    assert resources["initial_snapshot_state_bytes"] <= config.max_initial_snapshot_bytes
    assert resources["compiled_kernel_parity_checked"] is True
    assert resources["compiled_kernel_parity_exact"] is True
    assert resources["external_snapshot_mutations"] == 0


def test_build_and_live_replay_are_exact_and_do_not_mutate_snapshot(
    fixture: tuple[
        ExperientialMemory,
        ExperientialMemoryState,
        ExperientialMemoryTransferConfig,
        ExperientialMemoryTransferProtocol,
        dict[str, object],
    ],
) -> None:
    memory, state, config, protocol, report = fixture
    before = frozen_experiential_memory_state_sha256(state)
    replayed = build_experiential_memory_transfer_report(memory, state, config, protocol)
    assert replayed == report
    assert frozen_experiential_memory_state_sha256(state) == before
    _assert_trees_equal(state, memory.init())

    validation = validate_experiential_memory_transfer_report(
        report,
        memory=memory,
        state=state,
        protocol=protocol,
    )
    assert validation.valid, validation.errors
    mismatched = replace(protocol, protocol_id="different-protocol")
    validation = validate_experiential_memory_transfer_report(
        report,
        memory=memory,
        state=state,
        protocol=mismatched,
    )
    assert not validation.valid
    assert any("live replay" in error for error in validation.errors)


def test_validator_rejects_rehashed_trace_summary_and_source_tampering(
    fixture: tuple[
        ExperientialMemory,
        ExperientialMemoryState,
        ExperientialMemoryTransferConfig,
        ExperientialMemoryTransferProtocol,
        dict[str, object],
    ],
) -> None:
    _, _, _, _, report = fixture

    trace_tampered = copy.deepcopy(report)
    trace = cast(dict[str, Any], trace_tampered["payload"])[
        "raw_query_before_write_trace"
    ]
    trace[0]["memory_prediction"]["mean_squared_error"] += 1.0
    _rehash(trace_tampered)
    validation = validate_experiential_memory_transfer_report(trace_tampered)
    assert not validation.valid
    assert any("reconstruct" in error or "hash" in error for error in validation.errors)

    source_tampered = copy.deepcopy(report)
    sources = cast(dict[str, Any], source_tampered["payload"])["source_sha256"]
    sources[next(iter(sources))] = "0" * 64
    _rehash(source_tampered)
    validation = validate_experiential_memory_transfer_report(source_tampered)
    assert not validation.valid
    assert any("source" in error for error in validation.errors)


def test_canonical_report_save_load_and_no_overwrite(
    fixture: tuple[
        ExperientialMemory,
        ExperientialMemoryState,
        ExperientialMemoryTransferConfig,
        ExperientialMemoryTransferProtocol,
        dict[str, object],
    ],
    tmp_path: Path,
) -> None:
    _, _, _, _, report = fixture
    path = tmp_path / "memory-transfer.json"
    save_experiential_memory_transfer_report(path, report)
    assert path.read_bytes() == canonical_experiential_memory_transfer_report_bytes(report)
    assert load_experiential_memory_transfer_report(path) == report
    with pytest.raises(FileExistsError, match="overwrite"):
        save_experiential_memory_transfer_report(path, report)

    path.write_bytes(b" " + path.read_bytes())
    with pytest.raises(ValueError, match="canonical"):
        load_experiential_memory_transfer_report(path)


def test_snapshot_checkpoint_is_source_config_resource_and_state_bound(
    fixture: tuple[
        ExperientialMemory,
        ExperientialMemoryState,
        ExperientialMemoryTransferConfig,
        ExperientialMemoryTransferProtocol,
        dict[str, object],
    ],
    tmp_path: Path,
) -> None:
    memory, state, _, _, _ = fixture
    path = tmp_path / "memory-snapshot"
    save_experiential_memory_transfer_snapshot_checkpoint(memory, state, path)
    restored_memory, restored_state = load_experiential_memory_transfer_snapshot_checkpoint(path)
    assert restored_memory.to_config() == memory.to_config()
    _assert_trees_equal(restored_state, state)
    with pytest.raises(FileExistsError, match="overwrite"):
        save_experiential_memory_transfer_snapshot_checkpoint(memory, state, path)

    bad = tmp_path / "bad-memory-snapshot"
    save_checkpoint(
        state,
        bad,
        metadata={
            "schema": EXPERIENTIAL_MEMORY_TRANSFER_CHECKPOINT_SCHEMA,
            "memory_config": memory.to_config(),
        },
    )
    with pytest.raises(ValueError, match="metadata fields"):
        load_experiential_memory_transfer_snapshot_checkpoint(bad)


def test_hard_resource_bounds_fail_closed() -> None:
    config = default_experiential_memory_transfer_config()
    protocol = default_experiential_memory_transfer_protocol()
    memory = ExperientialMemory(config.memory_config)
    state = memory.init()

    with pytest.raises(ValueError, match="event"):
        build_experiential_memory_transfer_report(
            memory,
            state,
            replace(config, max_events=len(protocol.events) - 1),
            protocol,
        )
    with pytest.raises(ValueError, match="snapshot"):
        build_experiential_memory_transfer_report(
            memory,
            state,
            replace(config, max_initial_snapshot_bytes=1),
            protocol,
        )
    with pytest.raises(ValueError, match="report"):
        build_experiential_memory_transfer_report(
            memory,
            state,
            replace(config, max_report_bytes=1),
            protocol,
        )
    for field, value in (
        ("max_events", MAX_ABSOLUTE_EVENTS + 1),
        ("max_phases", MAX_ABSOLUTE_PHASES + 1),
        ("max_initial_snapshot_bytes", MAX_ABSOLUTE_SNAPSHOT_BYTES + 1),
        ("max_report_bytes", MAX_ABSOLUTE_REPORT_BYTES + 1),
    ):
        with pytest.raises(ValueError, match="hard evaluator ceiling"):
            replace(config, **cast(Any, {field: value}))


def test_source_manifest_is_exact_and_report_has_no_nonfinite_json(
    fixture: tuple[
        ExperientialMemory,
        ExperientialMemoryState,
        ExperientialMemoryTransferConfig,
        ExperientialMemoryTransferProtocol,
        dict[str, object],
    ],
) -> None:
    _, _, _, _, report = fixture
    payload = cast(dict[str, Any], report["payload"])
    assert payload["source_sha256"] == experiential_memory_transfer_source_snapshot()
    encoded = canonical_experiential_memory_transfer_report_bytes(report)
    assert b"NaN" not in encoded
    assert b"Infinity" not in encoded
    assert json.loads(encoded) == report
