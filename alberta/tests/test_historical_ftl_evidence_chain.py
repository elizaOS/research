"""Fail-closed tests for the historical FTL/current-source compatibility chain."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

import pytest

from alberta_framework.evaluation.evidence_manifest import (
    EVIDENCE_SPECS,
    EvidenceSpec,
    build_evidence_manifest,
)
from alberta_framework.evaluation.ftl_decision_artifact import (
    load_ftl_decision_artifact,
    validate_ftl_decision_artifact,
)

pytestmark = pytest.mark.scientific

_ROOT = Path(__file__).resolve().parents[1]
_ORIGINAL_PATH = Path("outputs/ftl_decision/evidence.v1.json")
_REPLAY_PATH = Path(
    "outputs/ftl_decision/reproductions/nonpromoting_consumed_seed_replay.json"
)
_ATTESTATION_PATH = Path("outputs/ftl_decision/reproductions/attestation.json")
_BUILDER_PATH = Path("alberta_framework/evaluation/ftl_decision_artifact.py")
_INVARIANT_PATHS = (
    Path("alberta_framework/core/ftl_world_model.py"),
    Path("alberta_framework/evaluation/ftl_decision_fidelity.py"),
    Path("alberta_framework/evaluation/ftl_decision_cli.py"),
)
_SOURCE_DRIFT_ERROR = (
    "scientific_payload.source_provenance does not match the current pinned "
    "source hashes"
)


@dataclass(frozen=True)
class _Validation:
    valid: bool
    accepted: bool
    errors: tuple[str, ...] = ()


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


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


def _reduced_ftl_sha256(artifact: dict[str, object]) -> str:
    reduced = deepcopy(artifact)
    del reduced["operational_metadata"]
    scientific = cast(dict[str, object], reduced["scientific_payload"])
    source = cast(dict[str, object], scientific["source_provenance"])
    hashes = cast(dict[str, object], source["source_sha256"])
    del hashes[_BUILDER_PATH.as_posix()]
    digest = cast(dict[str, object], reduced["scientific_digest"])
    del digest["sha256"]
    return _canonical_sha256(reduced)


def _registered_spec() -> EvidenceSpec:
    return next(
        spec
        for spec in EVIDENCE_SPECS
        if spec.name == "ftl_world_model_decision_fidelity"
    )


def _copy_fixture(root: Path) -> EvidenceSpec:
    spec = _registered_spec()
    for relative_path in (
        *spec.source_paths,
        _ORIGINAL_PATH,
        _REPLAY_PATH,
        _ATTESTATION_PATH,
    ):
        destination = root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((_ROOT / relative_path).read_bytes())
    return spec


def _claim(root: Path, spec: EvidenceSpec) -> dict[str, object]:
    manifest = build_evidence_manifest(root, specs=(spec,))
    claims = cast(list[dict[str, object]], manifest["claims"])
    assert len(claims) == 1
    return claims[0]


def _sync_replay_attestation(root: Path) -> None:
    replay_path = root / _REPLAY_PATH
    replay = cast(
        dict[str, object],
        json.loads(replay_path.read_text(encoding="utf-8")),
    )
    scientific = cast(dict[str, object], replay["scientific_payload"])
    digest = _canonical_sha256(scientific)
    cast(dict[str, object], replay["scientific_digest"])["sha256"] = digest
    replay_raw = _write_json(replay_path, replay)

    attestation_path = root / _ATTESTATION_PATH
    attestation = cast(
        dict[str, object],
        json.loads(attestation_path.read_text(encoding="utf-8")),
    )
    replay_record = cast(dict[str, object], attestation["current_source_replay"])
    replay_record["bytes_sha256"] = _sha256(replay_raw)
    replay_record["scientific_payload_sha256"] = digest
    comparison = cast(dict[str, object], attestation["comparison"])
    comparison["current_replay_reduced_sha256"] = _reduced_ftl_sha256(replay)
    _write_json(attestation_path, attestation)


@pytest.mark.unit
def test_exact_chain_fails_closed_after_invariant_source_drift(
    tmp_path: Path,
) -> None:
    spec = _copy_fixture(tmp_path)

    original = load_ftl_decision_artifact(tmp_path / _ORIGINAL_PATH)
    original_validation = validate_ftl_decision_artifact(original)
    assert original_validation.valid is False
    assert original_validation.accepted is False
    assert original_validation.errors == (_SOURCE_DRIFT_ERROR,)

    replay = load_ftl_decision_artifact(tmp_path / _REPLAY_PATH)
    replay_validation = validate_ftl_decision_artifact(replay)
    assert replay_validation.valid is False
    assert replay_validation.accepted is False
    assert replay_validation.errors == (_SOURCE_DRIFT_ERROR,)

    claim = _claim(tmp_path, spec)

    assert claim["status"] == "invalid"
    assert claim["valid"] is False
    assert claim["accepted"] is False
    assert claim["scientific_claim_supported"] is False
    assert (
        claim["validation_basis"]
        == "historical-acceptance-compatibility-chain-failed"
    )
    assert claim["primary_validator_valid"] is False
    assert claim["primary_validator_accepted"] is False
    assert claim["validator_accepted"] is False
    errors = cast(list[str], claim["errors"])
    assert any(
        "historical/current FTL source drift must be exactly" in error
        for error in errors
    )
    assert any(
        _INVARIANT_PATHS[1].as_posix() in error
        for error in errors
    )
    assert any(
        _INVARIANT_PATHS[2].as_posix() in error
        for error in errors
    )

    historical = cast(dict[str, object], claim["historical_validation"])
    assert historical["valid"] is False
    assert historical["classification"] == "invalid"
    assert historical["historical_scientific_claim_supported"] is False
    assert historical["current_replay_scientific_promotion_allowed"] is False
    original_record = cast(dict[str, object], historical["historical_artifact"])
    assert original_record["reconstructed_validator_valid"] is False
    assert original_record["reconstructed_validator_accepted"] is False
    replay_record = cast(dict[str, object], historical["current_source_replay"])
    assert replay_record["role"] == "nonpromoting_consumed_seed_replay"
    assert replay_record["consumed_seed_schedule"] is True
    assert replay_record["strict_validator_valid"] is False
    assert replay_record["validator_accepted"] is False
    recoverability = cast(
        dict[str, object],
        historical["historical_source_recoverability"],
    )
    assert recoverability["exact_artifact_builder_source_archived"] is False

    artifacts = cast(list[dict[str, object]], claim["artifacts"])
    replay_artifact = next(
        record
        for record in artifacts
        if record["role"] == "nonpromoting_consumed_seed_replay"
    )
    assert replay_artifact["scientific_promotion_allowed"] is False


@pytest.mark.unit
@pytest.mark.parametrize("relative_path", _INVARIANT_PATHS)
def test_world_evaluator_or_cli_drift_cannot_pass(
    tmp_path: Path,
    relative_path: Path,
) -> None:
    spec = _copy_fixture(tmp_path)
    path = tmp_path / relative_path
    path.write_bytes(path.read_bytes() + b"\n# post-replay scientific drift\n")

    claim = _claim(tmp_path, spec)

    assert claim["status"] == "invalid"
    assert claim["valid"] is False
    assert claim["accepted"] is False
    assert (
        claim["validation_basis"]
        == "historical-acceptance-compatibility-chain-failed"
    )
    errors = cast(list[str], claim["errors"])
    assert any(
        "historical/current FTL source drift must be exactly" in error
        and relative_path.as_posix() in error
        for error in errors
    )
    assert any(relative_path.as_posix() in error for error in errors)


@pytest.mark.unit
def test_additional_builder_drift_after_replay_cannot_pass(tmp_path: Path) -> None:
    spec = _copy_fixture(tmp_path)
    builder = tmp_path / _BUILDER_PATH
    builder.write_bytes(builder.read_bytes() + b"\n# drift after replay\n")

    claim = _claim(tmp_path, spec)

    assert claim["status"] == "invalid"
    assert claim["accepted"] is False
    errors = cast(list[str], claim["errors"])
    assert any("strict current FTL replay" in error for error in errors)
    assert any(_BUILDER_PATH.as_posix() in error for error in errors)


@pytest.mark.unit
def test_extra_primary_validator_error_cannot_enter_ftl_fallback(
    tmp_path: Path,
) -> None:
    spec = _copy_fixture(tmp_path)
    original_validator = spec.validator

    def _extra_failure(artifact: Mapping[str, object]) -> _Validation:
        validation = original_validator(artifact)
        assert validation.errors == (_SOURCE_DRIFT_ERROR,)
        return _Validation(
            valid=False,
            accepted=False,
            errors=(
                _SOURCE_DRIFT_ERROR,
                "primitive reconstruction failed",
            ),
        )

    changed_spec = replace(spec, validator=_extra_failure)
    claim = _claim(tmp_path, changed_spec)

    assert claim["status"] == "invalid"
    assert claim["accepted"] is False
    assert claim["validation_basis"] == "current-registered-sources"
    assert "historical_validation" not in claim
    assert "primitive reconstruction failed" in cast(list[str], claim["errors"])


@pytest.mark.unit
def test_changed_protocol_is_ineligible_for_historical_ftl_fallback(
    tmp_path: Path,
) -> None:
    spec = _copy_fixture(tmp_path)
    changed_protocol = dict(spec.protocol)
    changed_protocol["supported_claim"] = "post-hoc expanded FTL claim"
    changed_spec = replace(spec, protocol=changed_protocol)

    claim = _claim(tmp_path, changed_spec)

    assert claim["status"] == "invalid"
    assert claim["accepted"] is False
    assert claim["validation_basis"] == "current-registered-sources"
    assert "historical_validation" not in claim
    errors = cast(list[str], claim["errors"])
    assert any("registered frozen protocol" in error for error in errors)


@pytest.mark.unit
@pytest.mark.parametrize(
    "tamper",
    (
        "historical_bytes",
        "replay_scientific_field",
        "replay_acceptance",
        "attestation_wording",
    ),
)
def test_every_ftl_compatibility_link_fails_closed(
    tmp_path: Path,
    tamper: str,
) -> None:
    spec = _copy_fixture(tmp_path)
    if tamper == "historical_bytes":
        original_path = tmp_path / _ORIGINAL_PATH
        original = cast(
            dict[str, object],
            json.loads(original_path.read_text(encoding="utf-8")),
        )
        operational = cast(dict[str, object], original["operational_metadata"])
        operational["evaluation_wall_seconds"] = 999.0
        original_raw = _write_json(original_path, original)
        attestation_path = tmp_path / _ATTESTATION_PATH
        attestation = cast(
            dict[str, object],
            json.loads(attestation_path.read_text(encoding="utf-8")),
        )
        historical_record = cast(
            dict[str, object],
            attestation["historical_artifact"],
        )
        historical_record["bytes_sha256"] = _sha256(original_raw)
        _write_json(attestation_path, attestation)
    elif tamper == "replay_scientific_field":
        replay_path = tmp_path / _REPLAY_PATH
        replay = cast(
            dict[str, object],
            json.loads(replay_path.read_text(encoding="utf-8")),
        )
        scientific = cast(dict[str, object], replay["scientific_payload"])
        aggregate = cast(dict[str, object], scientific["aggregate"])
        aggregate["seed_count"] = 29
        _write_json(replay_path, replay)
        _sync_replay_attestation(tmp_path)
    elif tamper == "replay_acceptance":
        replay_path = tmp_path / _REPLAY_PATH
        replay = cast(
            dict[str, object],
            json.loads(replay_path.read_text(encoding="utf-8")),
        )
        scientific = cast(dict[str, object], replay["scientific_payload"])
        acceptance = cast(dict[str, object], scientific["acceptance"])
        acceptance["passed"] = False
        _write_json(replay_path, replay)
        _sync_replay_attestation(tmp_path)
    else:
        attestation_path = tmp_path / _ATTESTATION_PATH
        attestation = cast(
            dict[str, object],
            json.loads(attestation_path.read_text(encoding="utf-8")),
        )
        limitations = cast(list[str], attestation["limitations"])
        limitations[0] = "Consumed seeds are fresh evidence."
        _write_json(attestation_path, attestation)

    claim = _claim(tmp_path, spec)

    assert claim["status"] == "invalid"
    assert claim["valid"] is False
    assert claim["accepted"] is False
    assert (
        claim["validation_basis"]
        == "historical-acceptance-compatibility-chain-failed"
    )
    historical = cast(dict[str, object], claim["historical_validation"])
    assert historical["valid"] is False
    assert historical["historical_scientific_claim_supported"] is False
