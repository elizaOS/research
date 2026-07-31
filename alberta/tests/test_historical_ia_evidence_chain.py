"""Fail-closed tests for the continual-IA v1 historical rejection chain."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
import zipfile
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

import pytest

from alberta_framework.evaluation import evidence_manifest as manifest_module
from alberta_framework.evaluation.evidence_manifest import (
    EVIDENCE_SPECS,
    EvidenceSpec,
    ValidationResult,
    build_evidence_manifest,
)

pytestmark = pytest.mark.scientific

_SOURCE_DRIFT_ERROR = "content.source_provenance does not match the current pinned sources"
_SNAPSHOT_PATH = Path("outputs/continual_ia/source_snapshot_v1/manifest.json")
_ATTESTATION_PATH = Path("outputs/continual_ia/reproductions/attestation.json")
_REPLAY_PATH = Path("outputs/continual_ia/reproductions/nonpromoting_consumed_seed_replay.json")


@dataclass(frozen=True)
class _Validation:
    valid: bool
    accepted: bool
    errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class _Fixture:
    root: Path
    spec: EvidenceSpec
    validator: object
    replay_path: Path
    attestation_path: Path
    mutable_source: Path


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_sha256(value: object) -> str:
    return _sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    )


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_json(path: Path, value: object) -> bytes:
    raw = _json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return raw


def _reduced_sha256(artifact: Mapping[str, object]) -> str:
    reduced = deepcopy(dict(artifact))
    del reduced["content_digest"]
    del reduced["operational_diagnostics"]
    content = cast(dict[str, object], reduced["content"])
    del content["source_provenance"]
    return _canonical_sha256(reduced)


def _artifact(
    spec: EvidenceSpec,
    *,
    source_hashes: Mapping[str, str],
    runtime_label: str,
) -> dict[str, object]:
    content: dict[str, object] = {
        "protocol": spec.protocol,
        "configuration": spec.configuration,
        "thresholds": spec.thresholds,
        "seed_summaries": [{"seed": 30, "fixture_record": [1, 2, 3]}],
        "aggregate": {"fixture_metric": 0.08727777777777779},
        "acceptance": {
            "passed": False,
            "primary_passed": False,
            "secondary_passed": True,
            "checks": [
                {
                    "name": "changed_action_intervention_rate",
                    "passed": False,
                    "actual": 0.08727777777777779,
                    "threshold": 0.1,
                }
            ],
        },
        "source_provenance": {
            "repository_subtree": "research/alberta",
            "source_sha256": dict(source_hashes),
        },
    }
    return {
        "schema_version": spec.expected_schema,
        "content": content,
        "operational_diagnostics": {
            "digest_exclusion_reason": (
                "host environment and wall-clock timing are non-deterministic"
            ),
            "environment": {
                "python": {"version": runtime_label},
                "platform": {"system": "test"},
                "packages": {"fixture": "1"},
            },
            "condition_timings": [],
            "overall_acceptance_passed": False,
        },
        "content_digest": {
            "algorithm": "sha256",
            "scope": "$.content",
            "canonicalization": "utf8-json-sort-keys-compact-no-nan",
            "sha256": _canonical_sha256(content),
        },
    }


def _write_archives(
    root: Path,
    historical_sources: Mapping[str, bytes],
) -> dict[str, dict[str, str]]:
    snapshot_dir = root / _SNAPSHOT_PATH.parent
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    wheel_name = "alberta_framework-0.26.0-py3-none-any.whl"
    wheel_path = snapshot_dir / wheel_name
    with zipfile.ZipFile(wheel_path, mode="w", compression=zipfile.ZIP_DEFLATED) as wheel:
        for relative_path, raw in historical_sources.items():
            wheel.writestr(relative_path, raw)

    sdist_name = "alberta_framework-0.26.0.tar.gz"
    sdist_path = snapshot_dir / sdist_name
    with tarfile.open(sdist_path, mode="w:gz") as source_distribution:
        for relative_path, raw in historical_sources.items():
            member = tarfile.TarInfo(f"alberta_framework-0.26.0/{relative_path}")
            member.size = len(raw)
            member.mtime = 0
            source_distribution.addfile(member, io.BytesIO(raw))

    return {
        wheel_name: {
            "sha256": _sha256(wheel_path.read_bytes()),
            "role": (
                "exact recoverable package snapshot containing every artifact-pinned source path"
            ),
        },
        sdist_name: {
            "sha256": _sha256(sdist_path.read_bytes()),
            "role": "matching source distribution",
        },
    }


def _sync_replay_attestation(fixture: _Fixture) -> None:
    replay = json.loads(fixture.replay_path.read_text(encoding="utf-8"))
    replay_content = cast(dict[str, object], replay["content"])
    replay_digest = _canonical_sha256(replay_content)
    cast(dict[str, object], replay["content_digest"])["sha256"] = replay_digest
    replay_raw = _write_json(fixture.replay_path, replay)

    attestation = json.loads(fixture.attestation_path.read_text(encoding="utf-8"))
    replay_record = cast(dict[str, object], attestation["current_source_replay"])
    replay_record["bytes_sha256"] = _sha256(replay_raw)
    replay_record["scientific_content_sha256"] = replay_digest
    comparison = cast(dict[str, object], attestation["comparison"])
    comparison["current_replay_reduced_sha256"] = _reduced_sha256(replay)
    _write_json(fixture.attestation_path, attestation)


def _build_fixture(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> _Fixture:
    registered = next(
        spec for spec in EVIDENCE_SPECS if spec.name == "continual_intelligence_amplification"
    )
    historical_sources = {
        path.as_posix(): f"historical source: {path.as_posix()}\n".encode()
        for path in registered.source_paths
    }
    for relative_path in historical_sources:
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"current source: {relative_path}\n".encode())
    historical_hashes = {
        relative_path: _sha256(raw) for relative_path, raw in historical_sources.items()
    }
    current_hashes = {
        path.as_posix(): _sha256((root / path).read_bytes()) for path in registered.source_paths
    }

    def _loader(path: Path) -> Mapping[str, object]:
        parsed = json.loads(path.read_text(encoding="utf-8"))
        assert isinstance(parsed, dict)
        return cast(dict[str, object], parsed)

    def _validator(artifact: Mapping[str, object]) -> ValidationResult:
        content = cast(Mapping[str, object], artifact["content"])
        provenance = cast(Mapping[str, object], content["source_provenance"])
        observed = provenance["source_sha256"]
        live = {
            path.as_posix(): _sha256((root / path).read_bytes()) for path in registered.source_paths
        }
        if observed != live:
            return _Validation(
                valid=False,
                accepted=False,
                errors=(_SOURCE_DRIFT_ERROR,),
            )
        acceptance = cast(Mapping[str, object], content["acceptance"])
        accepted = acceptance["passed"] is True
        return _Validation(valid=True, accepted=accepted)

    spec = replace(registered, loader=_loader, validator=_validator)
    historical = _artifact(
        spec,
        source_hashes=historical_hashes,
        runtime_label="historical",
    )
    replay = _artifact(
        spec,
        source_hashes=current_hashes,
        runtime_label="current replay",
    )
    historical_raw = _write_json(root / spec.relative_path, historical)
    replay_path = root / _REPLAY_PATH
    replay_raw = _write_json(replay_path, replay)
    archives = _write_archives(root, historical_sources)
    historical_content_digest = cast(
        str,
        cast(Mapping[str, object], historical["content_digest"])["sha256"],
    )
    replay_content_digest = cast(
        str,
        cast(Mapping[str, object], replay["content_digest"])["sha256"],
    )
    snapshot = {
        "schema_version": "alberta.historical_source_snapshot.v1",
        "artifact": {
            "path": spec.relative_path.as_posix(),
            "schema_version": spec.expected_schema,
            "bytes_sha256": _sha256(historical_raw),
            "scientific_content_sha256": historical_content_digest,
            "result": "valid-rejection-at-generation",
        },
        "archives": archives,
        "artifact_pinned_source_sha256": historical_hashes,
        "verification": {
            "all_artifact_pinned_paths_present_in_wheel": True,
            "all_artifact_pinned_path_hashes_match_wheel": True,
            "artifact_preserved_byte_for_byte": True,
        },
        "limitations": [
            (
                "The v1 artifact pinned eight selected files, not every byte "
                "imported by Python or JAX."
            ),
            (
                "SHA-256 identifiers establish integrity and reproducibility "
                "context, not authorship or authenticity."
            ),
            (
                "The live worktree has since changed; this snapshot preserves "
                "the historical rejected result and does not validate a claim "
                "about current code."
            ),
            (
                "A replay on consumed v1 seeds can test compatibility but "
                "cannot become fresh held-out evidence."
            ),
            (
                "A changed IA protocol or promoted current-code claim requires "
                "a preregistered v2 and untouched seeds."
            ),
        ],
    }
    _write_json(root / _SNAPSHOT_PATH, snapshot)
    historical_reduced = _reduced_sha256(historical)
    replay_reduced = _reduced_sha256(replay)
    assert historical_reduced == replay_reduced
    attestation = {
        "schema_version": "alberta.ia_consumed_seed_source_compatibility.v1",
        "historical_artifact": {
            "path": spec.relative_path.as_posix(),
            "bytes_sha256": _sha256(historical_raw),
            "scientific_content_sha256": historical_content_digest,
            "source_snapshot_manifest": _SNAPSHOT_PATH.as_posix(),
        },
        "current_source_replay": {
            "path": _REPLAY_PATH.as_posix(),
            "bytes_sha256": _sha256(replay_raw),
            "scientific_content_sha256": replay_content_digest,
            "command": [
                "python",
                "-m",
                "alberta_framework.evaluation.continual_ia_cli",
                "--output",
                _REPLAY_PATH.as_posix(),
            ],
        },
        "comparison": {
            "excluded_paths": [
                "$.content.source_provenance",
                "$.content_digest",
                "$.operational_diagnostics",
            ],
            "included_scope": (
                "the complete remaining artifact, including schema, frozen "
                "protocol, configuration, thresholds, all primitive per-seed "
                "traces, aggregates, and acceptance decisions"
            ),
            "canonicalization": "utf8-json-sort-keys-compact-no-nan",
            "historical_reduced_sha256": historical_reduced,
            "current_replay_reduced_sha256": replay_reduced,
            "exact_match": True,
        },
        "interpretation": {
            "historical_result": (
                "the original v1 artifact remains a frozen valid rejection "
                "under its archived source snapshot"
            ),
            "compatibility_result": (
                "current pinned sources reproduce every v1 scientific field "
                "exactly on the consumed v1 schedule"
            ),
            "promotion_result": (
                "none; this replay is a source-compatibility check, not new held-out evidence"
            ),
        },
        "limitations": [
            "Seeds 30-59 were already consumed by the original v1 experiment.",
            (
                "The replay cannot support retuning, a stronger claim, or "
                "promotion of a changed protocol."
            ),
            (
                "The frozen 0.10 action-changing intervention-rate threshold "
                "remains failed and was not lowered."
            ),
            (
                "A changed IA protocol or new current-code promotion requires "
                "preregistration and untouched seeds."
            ),
        ],
    }
    attestation_path = root / _ATTESTATION_PATH
    _write_json(attestation_path, attestation)
    monkeypatch.setattr(
        manifest_module,
        "validate_ia_evidence_artifact",
        _validator,
    )
    return _Fixture(
        root=root,
        spec=spec,
        validator=_validator,
        replay_path=replay_path,
        attestation_path=attestation_path,
        mutable_source=root / registered.source_paths[0],
    )


def _claim(fixture: _Fixture) -> dict[str, object]:
    manifest = build_evidence_manifest(fixture.root, specs=(fixture.spec,))
    claims = cast(list[dict[str, object]], manifest["claims"])
    assert len(claims) == 1
    return claims[0]


@pytest.mark.unit
def test_exact_historical_chain_retains_only_a_valid_rejection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _build_fixture(tmp_path, monkeypatch)

    claim = _claim(fixture)

    assert claim["status"] == "valid-rejection"
    assert claim["valid"] is True
    assert claim["accepted"] is False
    assert claim["scientific_claim_supported"] is False
    assert claim["validation_basis"] == "archived-source-historical-rejection-chain"
    assert claim["primary_validator_valid"] is False
    assert claim["errors"] == []
    historical = cast(dict[str, object], claim["historical_validation"])
    assert historical["valid"] is True
    assert historical["scientific_promotion_allowed"] is False
    replay = cast(dict[str, object], historical["current_source_replay"])
    assert replay["role"] == "nonpromoting_consumed_seed_replay"
    assert replay["strict_validator_valid"] is True
    assert replay["validator_accepted"] is False
    assert replay["scientific_acceptance"] is False
    artifact_records = cast(list[dict[str, object]], claim["artifacts"])
    replay_record = next(
        record
        for record in artifact_records
        if record["role"] == "nonpromoting_consumed_seed_replay"
    )
    assert replay_record["scientific_promotion_allowed"] is False


@pytest.mark.unit
def test_any_non_source_primary_failure_cannot_enter_historical_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _build_fixture(tmp_path, monkeypatch)

    def _extra_failure(_: Mapping[str, object]) -> ValidationResult:
        return _Validation(
            valid=False,
            accepted=False,
            errors=(_SOURCE_DRIFT_ERROR, "primitive reconstruction failed"),
        )

    fixture = replace(
        fixture,
        spec=replace(fixture.spec, validator=_extra_failure),
    )
    claim = _claim(fixture)

    assert claim["status"] == "invalid"
    assert claim["valid"] is False
    assert claim["validation_basis"] == "current-registered-sources"
    assert "historical_validation" not in claim
    assert "primitive reconstruction failed" in cast(list[str], claim["errors"])


@pytest.mark.unit
def test_current_replay_source_drift_is_a_hard_failure_with_exact_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _build_fixture(tmp_path, monkeypatch)
    fixture.mutable_source.write_text("changed after replay\n", encoding="utf-8")

    claim = _claim(fixture)

    assert claim["status"] == "invalid"
    assert claim["validation_basis"] == "historical-rejection-chain-failed"
    errors = cast(list[str], claim["errors"])
    assert any(
        "current replay source hash mismatch" in error
        and fixture.mutable_source.relative_to(tmp_path).as_posix() in error
        and "replay " in error
        and "current " in error
        for error in errors
    )
    historical = cast(dict[str, object], claim["historical_validation"])
    replay = cast(dict[str, object], historical["current_source_replay"])
    assert replay["strict_validator_valid"] is False
    assert replay["validator_accepted"] is False


@pytest.mark.unit
@pytest.mark.parametrize(
    "tamper",
    [
        "archive",
        "missing_wheel_source",
        "attestation_wording",
        "scientific_field",
        "acceptance",
    ],
)
def test_every_historical_chain_link_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
) -> None:
    fixture = _build_fixture(tmp_path, monkeypatch)
    if tamper == "archive":
        archive = tmp_path / _SNAPSHOT_PATH.parent / "alberta_framework-0.26.0-py3-none-any.whl"
        archive.write_bytes(b"not the attested wheel")
    elif tamper == "missing_wheel_source":
        archive = tmp_path / _SNAPSHOT_PATH.parent / ("alberta_framework-0.26.0-py3-none-any.whl")
        with zipfile.ZipFile(archive) as wheel:
            retained = {
                name: wheel.read(name)
                for name in wheel.namelist()
                if name != fixture.spec.source_paths[0].as_posix()
            }
        with zipfile.ZipFile(
            archive,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
        ) as wheel:
            for name, raw in retained.items():
                wheel.writestr(name, raw)
        snapshot_path = tmp_path / _SNAPSHOT_PATH
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        archives = cast(dict[str, object], snapshot["archives"])
        wheel_record = cast(
            dict[str, object],
            archives["alberta_framework-0.26.0-py3-none-any.whl"],
        )
        wheel_record["sha256"] = _sha256(archive.read_bytes())
        _write_json(snapshot_path, snapshot)
    elif tamper == "attestation_wording":
        attestation = json.loads(fixture.attestation_path.read_text(encoding="utf-8"))
        cast(list[str], attestation["limitations"])[0] = "Consumed seeds are somehow fresh."
        _write_json(fixture.attestation_path, attestation)
    elif tamper == "scientific_field":
        replay = json.loads(fixture.replay_path.read_text(encoding="utf-8"))
        content = cast(dict[str, object], replay["content"])
        cast(dict[str, object], content["aggregate"])["fixture_metric"] = 0.2
        _write_json(fixture.replay_path, replay)
        _sync_replay_attestation(fixture)
    else:
        replay = json.loads(fixture.replay_path.read_text(encoding="utf-8"))
        content = cast(dict[str, object], replay["content"])
        cast(dict[str, object], content["acceptance"])["passed"] = True
        cast(dict[str, object], replay["operational_diagnostics"])["overall_acceptance_passed"] = (
            True
        )
        _write_json(fixture.replay_path, replay)
        _sync_replay_attestation(fixture)

    claim = _claim(fixture)

    assert claim["status"] == "invalid"
    assert claim["accepted"] is False
    assert claim["validation_basis"] == "historical-rejection-chain-failed"
    historical = cast(dict[str, object], claim["historical_validation"])
    assert historical["valid"] is False
    assert historical["scientific_promotion_allowed"] is False
