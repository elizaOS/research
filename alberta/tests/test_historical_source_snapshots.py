"""Integrity checks for source bundles retained with historical evidence."""

from __future__ import annotations

import hashlib
import json
import zipfile
from copy import deepcopy
from pathlib import Path

import pytest

pytestmark = pytest.mark.scientific

_ROOT = Path(__file__).resolve().parents[1]
_IA_DIR = _ROOT / "outputs" / "continual_ia"
_SNAPSHOT_DIR = _IA_DIR / "source_snapshot_v1"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _canonical_sha256(value: object) -> str:
    return _sha256_bytes(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    )


def _ia_reduced_artifact_sha256(artifact: dict[str, object]) -> str:
    reduced = deepcopy(artifact)
    del reduced["content_digest"]
    del reduced["operational_diagnostics"]
    content = reduced["content"]
    assert isinstance(content, dict)
    del content["source_provenance"]
    return _canonical_sha256(reduced)


def test_ia_v1_historical_artifact_and_source_archives_are_intact() -> None:
    manifest = json.loads((_SNAPSHOT_DIR / "manifest.json").read_text(encoding="utf-8"))
    artifact_path = _ROOT / manifest["artifact"]["path"]
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert _sha256_file(artifact_path) == manifest["artifact"]["bytes_sha256"]
    canonical_content = json.dumps(
        artifact["content"],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    assert (
        _sha256_bytes(canonical_content)
        == manifest["artifact"]["scientific_content_sha256"]
        == artifact["content_digest"]["sha256"]
    )

    archives = manifest["archives"]
    for archive_name, record in archives.items():
        assert _sha256_file(_SNAPSHOT_DIR / archive_name) == record["sha256"]

    pinned = manifest["artifact_pinned_source_sha256"]
    assert artifact["content"]["source_provenance"]["source_sha256"] == pinned
    wheel_path = _SNAPSHOT_DIR / "alberta_framework-0.26.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel_path) as wheel:
        for relative_path, expected_hash in pinned.items():
            assert _sha256_bytes(wheel.read(relative_path)) == expected_hash


def test_ia_v1_current_source_replay_exactly_matches_scientific_records() -> None:
    attestation_path = _IA_DIR / "reproductions" / "attestation.json"
    attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    historical_record = attestation["historical_artifact"]
    replay_record = attestation["current_source_replay"]
    historical_path = _ROOT / historical_record["path"]
    replay_path = _ROOT / replay_record["path"]
    historical = json.loads(historical_path.read_text(encoding="utf-8"))
    replay = json.loads(replay_path.read_text(encoding="utf-8"))

    assert _sha256_file(historical_path) == historical_record["bytes_sha256"]
    assert _sha256_file(replay_path) == replay_record["bytes_sha256"]
    assert (
        _canonical_sha256(historical["content"])
        == historical_record["scientific_content_sha256"]
        == historical["content_digest"]["sha256"]
    )
    assert (
        _canonical_sha256(replay["content"])
        == replay_record["scientific_content_sha256"]
        == replay["content_digest"]["sha256"]
    )

    historical_reduced = _ia_reduced_artifact_sha256(historical)
    replay_reduced = _ia_reduced_artifact_sha256(replay)
    comparison = attestation["comparison"]
    assert comparison["exact_match"] is True
    assert historical_reduced == comparison["historical_reduced_sha256"]
    assert replay_reduced == comparison["current_replay_reduced_sha256"]
    assert historical_reduced == replay_reduced
