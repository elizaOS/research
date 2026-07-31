"""Fail-closed contracts for the managed hidden-regime execution boundary."""

from __future__ import annotations

import base64
import dataclasses
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import alberta_framework.evaluation.hidden_regime_execution_governance as governance
from alberta_framework.evaluation.hidden_regime_execution_governance import (
    EXECUTION_AUTHORIZATION_ACKNOWLEDGEMENT,
    PROCESS_LOCAL_AUTHORIZATION_SCOPE,
    CalibrationExecutionAuthorization,
    HiddenRegimeCaseConsumedError,
    HiddenRegimeExecutionGovernanceError,
    HiddenRegimeProtectedExecutionError,
    begin_managed_hidden_regime_execution,
    build_calibration_execution_genesis,
    calibration_execution_configuration_sha256,
    calibration_execution_genesis_receipt_binding,
    calibration_execution_primitive_trace_sha256,
    calibration_execution_resource_sha256,
    calibration_execution_summary_sha256,
    canonical_json_bytes,
    canonical_sha256,
    classify_hidden_regime_world,
    complete_managed_hidden_regime_execution,
    initialize_calibration_execution_ledger,
    issue_calibration_execution_authorization,
    require_valid_calibration_execution_genesis,
    require_valid_calibration_execution_inventory,
    require_valid_calibration_execution_started_record,
    snapshot_calibration_execution_inventory,
    validate_completed_calibration_ledger_snapshot,
)
from alberta_framework.evaluation.hidden_regime_factorial_protocol import (
    CALIBRATION_DESIGN_PAYLOAD_SHA256,
    CONSUMED_CALIBRATION_NAMESPACE,
    N_MATCHED_CASES,
    SEED_SNAPSHOT_SHA256,
    build_hidden_regime_factorial_calibration_design,
)
from alberta_framework.evaluation.hidden_regime_signaling_development import (
    SELECTIVE_FULL,
    HiddenRegimeDevelopmentConfig,
    HiddenRegimeSeedPair,
    run_hidden_regime_condition,
)
from alberta_framework.streams.hidden_regime_signaling import (
    HIDDEN_REGIME_CALIBRATION_MANIFESTS,
    HIDDEN_REGIME_STRUCTURAL_MANIFESTS,
    HiddenRegimeWorldConfig,
)

pytestmark = pytest.mark.development

_SOURCE_ARCHIVE_DIGEST = "1" * 64
_SOURCE_MANIFEST_DIGEST = "2" * 64
_RUNTIME_DIGEST = "3" * 64
_READINESS_DIGEST = "4" * 64


@dataclasses.dataclass(frozen=True)
class _TinySummary:
    num_steps: int


@dataclasses.dataclass(frozen=True)
class _TinyResource:
    state_bytes: int = 552


@dataclasses.dataclass(frozen=True)
class _TinyTrace:
    step: np.ndarray
    reward: np.ndarray


@dataclasses.dataclass(frozen=True)
class _TinyResult:
    condition: str
    seed_pair: HiddenRegimeSeedPair
    config: HiddenRegimeDevelopmentConfig
    summary: _TinySummary
    resource: _TinyResource
    trace: _TinyTrace


def _genesis() -> dict[str, object]:
    return build_calibration_execution_genesis(
        source_archive_sha256=_SOURCE_ARCHIVE_DIGEST,
        source_manifest_sha256=_SOURCE_MANIFEST_DIGEST,
        runtime_identity_sha256=_RUNTIME_DIGEST,
    )


def _case_inputs(case_index: int = 0) -> tuple[str, HiddenRegimeSeedPair, Any]:
    case = build_hidden_regime_factorial_calibration_design().cases[case_index]
    seed_pair = HiddenRegimeSeedPair(
        namespace=CONSUMED_CALIBRATION_NAMESPACE,
        index=case.seed_index,
        world_seed=case.world_seed,
        learner_seed=case.learner_seed,
    )
    return case.condition, seed_pair, governance._expected_case_config(case)


def _fake_readiness(genesis: dict[str, object]) -> SimpleNamespace:
    body = {
        "authorization": {
            "ready_for_calibration": True,
            "protected_candidate_execution_permitted": False,
        },
        "source_snapshot": {"archive": {"sha256": _SOURCE_ARCHIVE_DIGEST}},
        governance.READINESS_EXECUTION_GOVERNANCE_FIELD: (
            calibration_execution_genesis_receipt_binding(genesis)
        ),
    }
    return SimpleNamespace(
        payload={"body": body, "receipt_sha256": _READINESS_DIGEST},
        receipt_sha256=_READINESS_DIGEST,
        source_archive_sha256=_SOURCE_ARCHIVE_DIGEST,
        source_manifest_sha256=_SOURCE_MANIFEST_DIGEST,
        runtime_identity_sha256=_RUNTIME_DIGEST,
    )


def _initialize(tmp_path: Path) -> tuple[Path, dict[str, object], SimpleNamespace]:
    genesis = _genesis()
    publication_root = tmp_path / "ledgers"
    publication_root.mkdir(parents=True)
    ledger = initialize_calibration_execution_ledger(
        publication_root,
        genesis,
        authorize_initialization=True,
    )
    return ledger.directory, genesis, _fake_readiness(genesis)


def _issue(
    monkeypatch: pytest.MonkeyPatch,
    ledger: Path,
    readiness: SimpleNamespace,
    *,
    case_index: int = 0,
    request_payload_sha256: str = "5" * 64,
    allow_exact_replay: bool = False,
) -> tuple[CalibrationExecutionAuthorization, str, HiddenRegimeSeedPair, Any]:
    monkeypatch.setattr(governance, "_validate_readiness_bundle", lambda bundle, archive: bundle)
    condition, seed_pair, config = _case_inputs(case_index)
    authorization = issue_calibration_execution_authorization(
        ledger_directory=ledger,
        readiness_bundle=readiness,
        readiness_source_archive=b"test-bound-archive",
        case_index=case_index,
        condition=condition,
        seed_pair=seed_pair,
        config=config,
        request_payload_sha256=request_payload_sha256,
        explicit_acknowledgement=EXECUTION_AUTHORIZATION_ACKNOWLEDGEMENT,
        allow_exact_replay=allow_exact_replay,
    )
    return authorization, condition, seed_pair, config


def test_pristine_genesis_and_inventory_are_deterministic_and_zero_entry(
    tmp_path: Path,
) -> None:
    first = _genesis()
    second = _genesis()
    assert first == second
    assert require_valid_calibration_execution_genesis(first) == first
    assert first["genesis_sha256"] == canonical_sha256(
        {key: value for key, value in first.items() if key != "genesis_sha256"}
    )
    ledger, _, _ = _initialize(tmp_path)
    snapshot = snapshot_calibration_execution_inventory(ledger)
    assert snapshot["expected_case_count"] == N_MATCHED_CASES
    assert snapshot["started_case_indices"] == []
    assert snapshot["completed_case_indices"] == []
    assert snapshot["interrupted_case_indices"] == []
    assert snapshot["protected_started_record_count"] == 0
    assert snapshot["protected_completed_record_count"] == 0
    assert snapshot["pristine"] is True
    assert require_valid_calibration_execution_inventory(snapshot, ledger) == snapshot
    assert (ledger / "genesis.json").stat().st_mode & 0o777 == 0o444
    assert len(tuple((ledger / "cases").iterdir())) == N_MATCHED_CASES

    with pytest.raises(FileExistsError, match="overwrite"):
        initialize_calibration_execution_ledger(
            tmp_path / "ledgers",
            first,
            authorize_initialization=True,
        )
    changed = dict(first)
    changed["source_manifest_sha256"] = "9" * 64
    with pytest.raises(HiddenRegimeExecutionGovernanceError, match="digest"):
        require_valid_calibration_execution_genesis(changed)


def test_frozen_case_configuration_digest_uses_exact_float_hex_payload() -> None:
    _, _, config = _case_inputs(0)
    encoded = governance._exact_json_value(config.to_dict(), label="configuration")
    exact_digest = calibration_execution_configuration_sha256(config)
    assert exact_digest == canonical_sha256(encoded)
    assert exact_digest != canonical_sha256(config.to_dict())


def test_exact_calibration_requires_sealed_authorization_and_consumes_before_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger, _, readiness = _initialize(tmp_path)
    authorization, condition, seed_pair, config = _issue(monkeypatch, ledger, readiness)
    assert authorization.payload["authorization_scope"] == PROCESS_LOCAL_AUTHORIZATION_SCOPE
    assert not hasattr(authorization, "to_json")

    with pytest.raises(HiddenRegimeExecutionGovernanceError, match="sealed"):
        begin_managed_hidden_regime_execution(
            condition=condition,
            seed_pair=seed_pair,
            config=config,
            authorization=None,
        )

    tampered_payload = dict(authorization.payload)
    tampered_payload["execution_mode"] = "exact_replay_after_interruption"
    tampered = dataclasses.replace(authorization, payload=tampered_payload)
    with pytest.raises(HiddenRegimeExecutionGovernanceError, match="seal"):
        begin_managed_hidden_regime_execution(
            condition=condition,
            seed_pair=seed_pair,
            config=config,
            authorization=tampered,
        )

    class _StopBeforeScanError(RuntimeError):
        pass

    def observe_consumption_before_scan(*_args: object, **_kwargs: object) -> object:
        snapshot = snapshot_calibration_execution_inventory(ledger)
        assert snapshot["started_case_indices"] == [0]
        assert snapshot["completed_case_indices"] == []
        raise _StopBeforeScanError

    import alberta_framework.evaluation.hidden_regime_signaling_development as development

    monkeypatch.setattr(development, "_scan_runner", observe_consumption_before_scan)
    with pytest.raises(_StopBeforeScanError):
        run_hidden_regime_condition(
            condition,  # type: ignore[arg-type]
            seed_pair=seed_pair,
            config=config,
            execution_authorization=authorization,
        )
    snapshot = snapshot_calibration_execution_inventory(ledger)
    assert snapshot["started_case_indices"] == [0]
    assert snapshot["interrupted_case_indices"] == [0]


def test_authorization_seal_is_rejected_by_a_fresh_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger, _, readiness = _initialize(tmp_path)
    authorization, _, _, _ = _issue(monkeypatch, ledger, readiness)
    encoded = base64.b64encode(canonical_json_bytes(authorization.payload)).decode("ascii")
    script = """
import base64
import hmac
import json
import sys
from alberta_framework.evaluation.hidden_regime_execution_governance import _seal

payload = json.loads(base64.b64decode(sys.argv[1], validate=True))
valid = hmac.compare_digest(
    sys.argv[2],
    _seal("calibration-execution-authorization-v1", payload),
)
raise SystemExit(0 if valid else 23)
"""
    completed = subprocess.run(
        (sys.executable, "-c", script, encoded, authorization.seal),
        check=False,
        capture_output=True,
    )
    assert completed.returncode == 23


def test_protected_manifests_and_tail_extensions_fail_closed() -> None:
    _, seed_pair, ordinary_config = _case_inputs(0)
    for manifest in HIDDEN_REGIME_STRUCTURAL_MANIFESTS.values():
        worlds = [manifest.to_world_config(repeat_schedule=False)]
        worlds.extend(
            HiddenRegimeWorldConfig(
                segment_lengths=(
                    *manifest.segment_lengths[:-1],
                    manifest.segment_lengths[-1] + extension,
                ),
                segment_regimes=manifest.segment_regimes,
                regime_permutations=manifest.regime_permutations,
                repeat_schedule=False,
            )
            for extension in range(1, 16)
        )
        for world in worlds:
            classification = classify_hidden_regime_world(world)
            assert classification.sensitivity == "protected"
            config = dataclasses.replace(ordinary_config, world=world)
            with pytest.raises(HiddenRegimeProtectedExecutionError, match="no learner-execution"):
                begin_managed_hidden_regime_execution(
                    condition=SELECTIVE_FULL,
                    seed_pair=seed_pair,
                    config=config,
                    authorization=None,
                )

    for manifest in HIDDEN_REGIME_CALIBRATION_MANIFESTS.values():
        classification = classify_hidden_regime_world(
            manifest.to_world_config(repeat_schedule=False)
        )
        assert classification.sensitivity == "calibration"
        assert classification.manifest_name == manifest.name

    changed_world = dataclasses.replace(
        ordinary_config.world,
        segment_lengths=(
            ordinary_config.world.segment_lengths[0] + 1,
            *ordinary_config.world.segment_lengths[1:],
        ),
    )
    assert classify_hidden_regime_world(changed_world).sensitivity == "ordinary"


def test_concurrent_first_execution_has_exactly_one_winner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger, _, readiness = _initialize(tmp_path)
    first, condition, seed_pair, config = _issue(monkeypatch, ledger, readiness)
    second, _, _, _ = _issue(monkeypatch, ledger, readiness)

    def begin(authorization: CalibrationExecutionAuthorization) -> str:
        try:
            ticket = begin_managed_hidden_regime_execution(
                condition=condition,
                seed_pair=seed_pair,
                config=config,
                authorization=authorization,
            )
        except HiddenRegimeCaseConsumedError:
            return "consumed"
        assert ticket is not None
        return "won"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = tuple(executor.map(begin, (first, second)))
    assert sorted(outcomes) == ["consumed", "won"]
    snapshot = snapshot_calibration_execution_inventory(ledger)
    assert snapshot["started_record_count"] == 1
    assert snapshot["completed_record_count"] == 0


def test_crash_consumption_allows_only_exact_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger, _, readiness = _initialize(tmp_path)
    authorization, condition, seed_pair, config = _issue(monkeypatch, ledger, readiness)
    ticket = begin_managed_hidden_regime_execution(
        condition=condition,
        seed_pair=seed_pair,
        config=config,
        authorization=authorization,
    )
    assert ticket is not None

    with pytest.raises(HiddenRegimeCaseConsumedError, match="exact replay"):
        _issue(monkeypatch, ledger, readiness)
    different_pair = dataclasses.replace(seed_pair, learner_seed=seed_pair.learner_seed ^ 1)
    with pytest.raises(HiddenRegimeExecutionGovernanceError, match="seed pair"):
        issue_calibration_execution_authorization(
            ledger_directory=ledger,
            readiness_bundle=readiness,
            readiness_source_archive=b"test-bound-archive",
            case_index=0,
            condition=condition,
            seed_pair=different_pair,
            config=config,
            request_payload_sha256="5" * 64,
            explicit_acknowledgement=EXECUTION_AUTHORIZATION_ACKNOWLEDGEMENT,
            allow_exact_replay=True,
        )

    replay, _, _, _ = _issue(
        monkeypatch,
        ledger,
        readiness,
        allow_exact_replay=True,
    )
    replay_ticket = begin_managed_hidden_regime_execution(
        condition=condition,
        seed_pair=seed_pair,
        config=config,
        authorization=replay,
    )
    assert replay_ticket is not None
    assert replay_ticket.execution_mode == "exact_replay_after_interruption"

    result = _TinyResult(
        condition=condition,
        seed_pair=seed_pair,
        config=config,
        summary=_TinySummary(config.num_steps),
        resource=_TinyResource(),
        trace=_TinyTrace(
            step=np.asarray([0, 1], dtype=np.int32),
            reward=np.asarray([0.0, 1.0], dtype=np.float32),
        ),
    )
    completed = complete_managed_hidden_regime_execution(replay_ticket, result)
    assert completed is not None
    assert completed["execution_state"] == "learner_execution_completed"
    snapshot = snapshot_calibration_execution_inventory(ledger)
    assert snapshot["completed_case_indices"] == [0]
    assert snapshot["interrupted_case_indices"] == []

    completed_replay, _, _, _ = _issue(
        monkeypatch,
        ledger,
        readiness,
        allow_exact_replay=True,
    )
    completed_ticket = begin_managed_hidden_regime_execution(
        condition=condition,
        seed_pair=seed_pair,
        config=config,
        authorization=completed_replay,
    )
    assert completed_ticket is not None
    assert completed_ticket.execution_mode == "exact_replay_after_completion"
    assert complete_managed_hidden_regime_execution(completed_ticket, result) == completed

    changed_result = dataclasses.replace(
        result,
        trace=_TinyTrace(
            step=np.asarray([0, 1], dtype=np.int32),
            reward=np.asarray([1.0, 1.0], dtype=np.float32),
        ),
    )
    with pytest.raises(HiddenRegimeExecutionGovernanceError, match="outcome differs"):
        complete_managed_hidden_regime_execution(completed_ticket, changed_result)


def test_completed_snapshot_joins_request_summary_resource_and_trace_digests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger, genesis, readiness = _initialize(tmp_path)
    governance_binding = calibration_execution_genesis_receipt_binding(genesis)
    readiness_binding = {
        "readiness_receipt_sha256": _READINESS_DIGEST,
        "source_archive_sha256": _SOURCE_ARCHIVE_DIGEST,
        "source_manifest_sha256": _SOURCE_MANIFEST_DIGEST,
        "runtime_identity_sha256": _RUNTIME_DIGEST,
        governance.READINESS_EXECUTION_GOVERNANCE_FIELD: governance_binding,
    }
    case_sources: dict[int, tuple[str, HiddenRegimeDevelopmentConfig, str]] = {}
    trace = _TinyTrace(
        step=np.asarray([0, 1], dtype=np.int32),
        reward=np.asarray([0.0, 1.0], dtype=np.float32),
    )
    summary = _TinySummary(16_528)
    resource = _TinyResource()
    for case_index in range(N_MATCHED_CASES):
        request_digest = canonical_sha256({"case_index": case_index, "request": "test"})
        authorization, condition, seed_pair, config = _issue(
            monkeypatch,
            ledger,
            readiness,
            case_index=case_index,
            request_payload_sha256=request_digest,
        )
        ticket = begin_managed_hidden_regime_execution(
            condition=condition,
            seed_pair=seed_pair,
            config=config,
            authorization=authorization,
        )
        assert ticket is not None
        result = _TinyResult(
            condition=condition,
            seed_pair=seed_pair,
            config=config,
            summary=summary,
            resource=resource,
            trace=trace,
        )
        assert complete_managed_hidden_regime_execution(ticket, result) is not None
        case_sources[case_index] = (request_digest, config, condition)

    snapshot = snapshot_calibration_execution_inventory(ledger)
    completed_by_case = {
        item["case_index"]: item
        for item in snapshot["completed_records"]  # type: ignore[union-attr]
    }
    started_by_case = {
        item["case_index"]: item
        for item in snapshot["started_records"]  # type: ignore[union-attr]
    }
    design = build_hidden_regime_factorial_calibration_design()
    summary_payload = dataclasses.asdict(summary)
    resource_payload = dataclasses.asdict(resource)
    summary_digest = calibration_execution_summary_sha256(summary)
    resource_digest = calibration_execution_resource_sha256(resource)
    trace_digest = calibration_execution_primitive_trace_sha256(trace)
    shards: dict[int, dict[str, object]] = {}
    for case_index, case in enumerate(design.cases):
        request_digest, config, condition = case_sources[case_index]
        assert condition == case.condition
        started = started_by_case[case_index]
        completed = completed_by_case[case_index]
        shards[case_index] = {
            "case": case.to_payload(),
            "protocol_payload_sha256": CALIBRATION_DESIGN_PAYLOAD_SHA256,
            "seed_snapshot_sha256": SEED_SNAPSHOT_SHA256,
            "request_payload_sha256": request_digest,
            "configuration": governance._exact_json_value(
                config.to_dict(),
                label="configuration",
            ),
            "configuration_sha256": calibration_execution_configuration_sha256(config),
            "readiness_binding": readiness_binding,
            "executed_steps": summary.num_steps,
            "summary": summary_payload,
            "summary_sha256": summary_digest,
            "resource": resource_payload,
            "resource_sha256": resource_digest,
            "primitive_trace": {"sha256": trace_digest},
            "execution_record_binding": {
                "case_index": case_index,
                "genesis_sha256": snapshot["genesis_sha256"],
                "started_record_sha256": started["started_record_sha256"],
                "completed_record_sha256": completed["completed_record_sha256"],
                "summary_sha256": completed["summary_sha256"],
                "resource_sha256": completed["resource_sha256"],
                "primitive_trace_sha256": completed["primitive_trace_sha256"],
                "outcome_sha256": completed["outcome_sha256"],
            },
        }
        shards[case_index]["payload_sha256"] = canonical_sha256(shards[case_index])
    assert validate_completed_calibration_ledger_snapshot(snapshot, shards) == snapshot

    summary_tamper = deepcopy(shards)
    summary_tamper[0]["summary"] = {"num_steps": 16_527}
    summary_tamper[0]["summary_sha256"] = canonical_sha256(summary_tamper[0]["summary"])
    summary_body = dict(summary_tamper[0])
    summary_body.pop("payload_sha256")
    summary_tamper[0]["payload_sha256"] = canonical_sha256(summary_body)
    with pytest.raises(HiddenRegimeExecutionGovernanceError, match="immutable completion"):
        validate_completed_calibration_ledger_snapshot(snapshot, summary_tamper)

    resource_tamper = deepcopy(shards)
    resource_tamper[0]["resource"] = {"state_bytes": 551}
    resource_tamper[0]["resource_sha256"] = canonical_sha256(resource_tamper[0]["resource"])
    resource_body = dict(resource_tamper[0])
    resource_body.pop("payload_sha256")
    resource_tamper[0]["payload_sha256"] = canonical_sha256(resource_body)
    with pytest.raises(HiddenRegimeExecutionGovernanceError, match="immutable completion"):
        validate_completed_calibration_ledger_snapshot(snapshot, resource_tamper)


def test_ledger_tamper_symlink_and_started_record_mutation_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    publication_root = tmp_path / "publication"
    publication_root.mkdir()
    target = tmp_path / "target"
    target.mkdir()
    symlink = tmp_path / "publication-link"
    symlink.symlink_to(target, target_is_directory=True)
    with pytest.raises(HiddenRegimeExecutionGovernanceError, match="symlink"):
        initialize_calibration_execution_ledger(
            symlink,
            _genesis(),
            authorize_initialization=True,
        )

    real_ancestor = tmp_path / "real-ancestor"
    nested_root = real_ancestor / "nested" / "publication"
    nested_root.mkdir(parents=True)
    linked_ancestor = tmp_path / "linked-ancestor"
    linked_ancestor.symlink_to(real_ancestor, target_is_directory=True)
    with pytest.raises(HiddenRegimeExecutionGovernanceError, match="traverses a symlink"):
        initialize_calibration_execution_ledger(
            linked_ancestor / "nested" / "publication",
            _genesis(),
            authorize_initialization=True,
        )

    ledger, _, readiness = _initialize(tmp_path / "separate")
    authorization, condition, seed_pair, config = _issue(monkeypatch, ledger, readiness)
    assert begin_managed_hidden_regime_execution(
        condition=condition,
        seed_pair=seed_pair,
        config=config,
        authorization=authorization,
    ) is not None
    started_path = ledger / "cases" / "case-000" / "started.json"
    started = json.loads(started_path.read_text(encoding="ascii"))
    started["execution_state"] = "unconsumed"
    started["started_record_sha256"] = canonical_sha256(
        {key: value for key, value in started.items() if key != "started_record_sha256"}
    )
    with pytest.raises(HiddenRegimeExecutionGovernanceError, match="state"):
        require_valid_calibration_execution_started_record(started)

    os.chmod(started_path, 0o644)
    with pytest.raises(HiddenRegimeExecutionGovernanceError, match="mode"):
        snapshot_calibration_execution_inventory(ledger)
