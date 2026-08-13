"""Contracts for the fail-closed 18-row complete-prototype manifest."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from alberta_framework.evaluation.complete_prototype_manifest import (
    COMPLETE_PROTOTYPE_CONTRACT_VERSION,
    COMPLETE_PROTOTYPE_INDEX_SCHEMA,
    COMPLETE_PROTOTYPE_MANIFEST_SCHEMA,
    SCORECARD_ROWS,
    CompletePrototypeValidationReceipt,
    RegisteredCompletePrototypeValidator,
    build_complete_prototype_manifest,
    complete_prototype_manifest_exit_code,
    complete_prototype_manifest_json,
    empty_complete_prototype_evidence_index,
    validate_complete_prototype_manifest,
)

pytestmark = pytest.mark.unit

CONFIGURATION_SHA256 = "1" * 64
PROTOCOL_SHA256 = "2" * 64
SCIENTIFIC_SHA256 = "3" * 64
ARTIFACT_SCHEMA = "test.complete-prototype-artifact.v1"
VALIDATOR_ID = "test.strict-complete-prototype-validator.v1"


def _roles(property_id: str, *, delight: bool, kondo: bool) -> tuple[str, ...]:
    row = next(row for row in SCORECARD_ROWS if row.property_id == property_id)
    roles = row.required_roles
    if property_id == "candidate_update_audit":
        if delight:
            roles += ("paper_delight_actor_learning_and_guardrails",)
        if kondo:
            roles += ("kondo_measured_compute_and_guardrails",)
    return roles


def _receipt(
    *,
    evidence_role: str,
    accepted: bool = True,
) -> CompletePrototypeValidationReceipt:
    return CompletePrototypeValidationReceipt(
        valid=True,
        accepted=accepted,
        evidence_role=evidence_role,
        prototype_configuration_sha256=CONFIGURATION_SHA256,
        schema_version=ARTIFACT_SCHEMA,
        protocol_sha256=PROTOCOL_SHA256,
        scientific_digest_sha256=SCIENTIFIC_SHA256,
        evidence_class="scientific",
        evidence_level="L3",
        frozen_protocol_valid=True,
        untouched_held_out_seeds_valid=True,
        source_closure_valid=True,
    )


def _fixture(
    root: Path,
    *,
    delight: bool = False,
    kondo: bool = False,
    modes: dict[tuple[str, str], str] | None = None,
) -> tuple[
    dict[str, object],
    dict[str, RegisteredCompletePrototypeValidator],
]:
    source_path = Path("validators/complete.py")
    absolute_source = root / source_path
    absolute_source.parent.mkdir(parents=True, exist_ok=True)
    absolute_source.write_text("# frozen test validator\n", encoding="utf-8")
    source_sha256 = hashlib.sha256(absolute_source.read_bytes()).hexdigest()

    def validate(path: Path) -> CompletePrototypeValidationReceipt:
        payload = json.loads(path.read_text(encoding="utf-8"))
        mode = payload["mode"]
        receipt = _receipt(
            evidence_role=cast(str, payload["role"]),
            accepted=mode != "rejection",
        )
        if mode == "invalid":
            return replace(receipt, valid=False, accepted=False)
        if mode == "l2":
            return replace(receipt, evidence_level="L2")
        if mode == "development":
            return replace(receipt, evidence_class="development")
        if mode == "reused-seeds":
            return replace(receipt, untouched_held_out_seeds_valid=False)
        if mode == "protocol-drift":
            return replace(receipt, protocol_sha256="4" * 64)
        if mode == "schema-drift":
            return replace(receipt, schema_version="test.wrong.v1")
        if mode == "source-gap":
            return replace(receipt, source_closure_valid=False)
        if mode == "configuration-drift":
            return replace(receipt, prototype_configuration_sha256="5" * 64)
        if mode == "role-drift":
            return replace(receipt, evidence_role="wrong_role")
        if mode == "scientific-digest-drift":
            return replace(receipt, scientific_digest_sha256="6" * 64)
        return receipt

    registered = RegisteredCompletePrototypeValidator(
        validator_id=VALIDATOR_ID,
        source_paths=(source_path,),
        validate=validate,
    )
    index = empty_complete_prototype_evidence_index(
        prototype_configuration_sha256=CONFIGURATION_SHA256,
        paper_delight_enabled=delight,
        kondo_enabled=kondo,
    )
    rows = cast(list[dict[str, object]], index["rows"])
    for row in rows:
        property_id = cast(str, row["property_id"])
        references: list[dict[str, object]] = []
        for role in _roles(property_id, delight=delight, kondo=kondo):
            mode = "accepted" if modes is None else modes.get((property_id, role), "accepted")
            relative_path = Path("outputs/complete") / f"{property_id}.{role}.json"
            absolute_path = root / relative_path
            absolute_path.parent.mkdir(parents=True, exist_ok=True)
            raw = (
                json.dumps({"mode": mode, "role": role}, sort_keys=True) + "\n"
            ).encode()
            absolute_path.write_bytes(raw)
            references.append(
                {
                    "role": role,
                    "relative_path": relative_path.as_posix(),
                    "artifact_sha256": hashlib.sha256(raw).hexdigest(),
                    "expected_schema": ARTIFACT_SCHEMA,
                    "expected_protocol_sha256": PROTOCOL_SHA256,
                    "expected_scientific_digest_sha256": SCIENTIFIC_SHA256,
                    "validator_id": VALIDATOR_ID,
                    "validator_source_sha256": {
                        source_path.as_posix(): source_sha256,
                    },
                }
            )
        row["evidence"] = references
    return index, {VALIDATOR_ID: registered}


def _row(manifest: dict[str, object], property_id: str) -> dict[str, object]:
    rows = cast(list[dict[str, object]], manifest["rows"])
    return next(row for row in rows if row["property_id"] == property_id)


def test_empty_index_is_exact_and_cannot_claim_completion(tmp_path: Path) -> None:
    index = empty_complete_prototype_evidence_index(
        prototype_configuration_sha256=CONFIGURATION_SHA256,
        paper_delight_enabled=True,
        kondo_enabled=True,
    )

    manifest = build_complete_prototype_manifest(tmp_path, index, validators={})

    assert index["schema_version"] == COMPLETE_PROTOTYPE_INDEX_SCHEMA
    assert index["contract_version"] == COMPLETE_PROTOTYPE_CONTRACT_VERSION
    assert len(cast(list[object], index["rows"])) == 18
    assert manifest["schema_version"] == COMPLETE_PROTOTYPE_MANIFEST_SCHEMA
    assert manifest["overall_status"] == "not-ready"
    assert manifest["all_18_rows_registered"] is False
    assert manifest["all_18_rows_accepted"] is False
    assert complete_prototype_manifest_exit_code(manifest) == 1
    candidate = _row(manifest, "candidate_update_audit")
    assert candidate["required_roles"] == [
        "audited_update_realized_outcomes",
        "paper_delight_actor_learning_and_guardrails",
        "kondo_measured_compute_and_guardrails",
    ]


def test_every_exact_source_pinned_l3_acceptance_is_required(tmp_path: Path) -> None:
    index, validators = _fixture(tmp_path, delight=True, kondo=True)

    first = build_complete_prototype_manifest(tmp_path, index, validators=validators)
    second = build_complete_prototype_manifest(tmp_path, index, validators=validators)

    assert first == second
    assert first["overall_status"] == "accepted"
    assert first["all_18_rows_registered"] is True
    assert first["all_18_rows_accepted"] is True
    assert all(
        row["status"] == "accepted"
        for row in cast(list[dict[str, object]], first["rows"])
    )
    assert complete_prototype_manifest_exit_code(first) == 0
    assert validate_complete_prototype_manifest(first) == ()
    assert len(cast(str, first["manifest_sha256"])) == 64
    rendered = complete_prototype_manifest_json(first)
    assert rendered.endswith("\n")
    assert json.loads(rendered) == first


def test_manifest_consumer_reconstructs_rows_flags_and_self_digest(tmp_path: Path) -> None:
    index, validators = _fixture(tmp_path)
    manifest = build_complete_prototype_manifest(tmp_path, index, validators=validators)

    tampered = cast(dict[str, object], json.loads(json.dumps(manifest)))
    cast(list[dict[str, object]], tampered["rows"])[0]["status"] = "not-run"
    errors = validate_complete_prototype_manifest(tampered)
    assert any("row status does not reconstruct" in error for error in errors)
    assert complete_prototype_manifest_exit_code(tampered) == 2

    forged_flag = cast(dict[str, object], json.loads(json.dumps(manifest)))
    forged_flag["all_18_rows_accepted"] = False
    errors = validate_complete_prototype_manifest(forged_flag)
    assert "manifest all_18_rows_accepted does not reconstruct" in errors
    assert "manifest self-digest does not match its payload" in errors

    malformed = {"schema_version": COMPLETE_PROTOTYPE_MANIFEST_SCHEMA,
                 "overall_status": "accepted"}
    assert complete_prototype_manifest_exit_code(malformed) == 2


def test_missing_artifact_is_not_run_and_rejection_is_preserved(tmp_path: Path) -> None:
    first_role = SCORECARD_ROWS[0].required_roles[0]
    reject_role = SCORECARD_ROWS[1].required_roles[0]
    index, validators = _fixture(
        tmp_path,
        modes={(SCORECARD_ROWS[1].property_id, reject_role): "rejection"},
    )
    rows = cast(list[dict[str, object]], index["rows"])
    first_reference = cast(list[dict[str, object]], rows[0]["evidence"])[0]
    (tmp_path / cast(str, first_reference["relative_path"])).unlink()

    manifest = build_complete_prototype_manifest(tmp_path, index, validators=validators)

    assert manifest["overall_status"] == "not-ready"
    first = _row(manifest, SCORECARD_ROWS[0].property_id)
    assert first["required_roles"] == [first_role]
    assert first["status"] == "not-run"
    assert _row(manifest, SCORECARD_ROWS[1].property_id)["status"] == "valid-rejection"
    assert complete_prototype_manifest_exit_code(manifest) == 1


@pytest.mark.parametrize(
    "mode",
    [
        "invalid",
        "l2",
        "development",
        "reused-seeds",
        "protocol-drift",
        "schema-drift",
        "source-gap",
        "configuration-drift",
        "role-drift",
        "scientific-digest-drift",
    ],
)
def test_nonpromoting_or_invalid_receipts_fail_closed(tmp_path: Path, mode: str) -> None:
    property_id = SCORECARD_ROWS[0].property_id
    role = SCORECARD_ROWS[0].required_roles[0]
    index, validators = _fixture(tmp_path, modes={(property_id, role): mode})

    manifest = build_complete_prototype_manifest(tmp_path, index, validators=validators)

    assert manifest["overall_status"] == "invalid"
    assert _row(manifest, property_id)["status"] == "invalid"
    assert complete_prototype_manifest_exit_code(manifest) == 2


def test_artifact_and_validator_source_tampering_are_invalid(tmp_path: Path) -> None:
    index, validators = _fixture(tmp_path)
    rows = cast(list[dict[str, object]], index["rows"])
    first_reference = cast(list[dict[str, object]], rows[0]["evidence"])[0]
    artifact_path = tmp_path / cast(str, first_reference["relative_path"])
    artifact_path.write_text('{"mode":"accepted","tampered":true}\n', encoding="utf-8")

    artifact_tamper = build_complete_prototype_manifest(
        tmp_path,
        index,
        validators=validators,
    )
    assert artifact_tamper["overall_status"] == "invalid"

    clean_index, clean_validators = _fixture(tmp_path)
    (tmp_path / "validators/complete.py").write_text("# drift\n", encoding="utf-8")
    source_tamper = build_complete_prototype_manifest(
        tmp_path,
        clean_index,
        validators=clean_validators,
    )
    assert source_tamper["overall_status"] == "invalid"
    first_evidence = cast(
        list[dict[str, object]],
        _row(source_tamper, SCORECARD_ROWS[0].property_id)["evidence"],
    )[0]
    assert "validator source hash mismatch" in cast(list[str], first_evidence["errors"])[0]


def test_unknown_validator_and_validator_exception_are_invalid(tmp_path: Path) -> None:
    index, validators = _fixture(tmp_path)
    rows = cast(list[dict[str, object]], index["rows"])
    first_reference = cast(list[dict[str, object]], rows[0]["evidence"])[0]
    first_reference["validator_id"] = "unknown"
    unknown = build_complete_prototype_manifest(tmp_path, index, validators=validators)
    assert unknown["overall_status"] == "invalid"

    failing_index, failing_validators = _fixture(tmp_path)
    registered = failing_validators[VALIDATOR_ID]

    def fail(_: Path) -> CompletePrototypeValidationReceipt:
        raise RuntimeError("deliberate validator failure")

    failing_validators[VALIDATOR_ID] = replace(registered, validate=fail)
    failed = build_complete_prototype_manifest(
        tmp_path,
        failing_index,
        validators=failing_validators,
    )
    assert failed["overall_status"] == "invalid"


def test_index_shape_roles_paths_and_digests_are_strict(tmp_path: Path) -> None:
    index = empty_complete_prototype_evidence_index(
        prototype_configuration_sha256=CONFIGURATION_SHA256,
        paper_delight_enabled=False,
        kondo_enabled=False,
    )
    index["extra"] = True
    with pytest.raises(ValueError, match="fields are not exact"):
        build_complete_prototype_manifest(tmp_path, index, validators={})

    index, validators = _fixture(tmp_path)
    rows = cast(list[dict[str, object]], index["rows"])
    rows[0], rows[1] = rows[1], rows[0]
    with pytest.raises(ValueError, match="canonical order"):
        build_complete_prototype_manifest(tmp_path, index, validators=validators)

    index, validators = _fixture(tmp_path)
    rows = cast(list[dict[str, object]], index["rows"])
    reference = cast(list[dict[str, object]], rows[0]["evidence"])[0]
    reference["relative_path"] = "../outside.json"
    with pytest.raises(ValueError, match="repository-relative"):
        build_complete_prototype_manifest(tmp_path, index, validators=validators)

    index, validators = _fixture(tmp_path)
    rows = cast(list[dict[str, object]], index["rows"])
    reference = cast(list[dict[str, object]], rows[0]["evidence"])[0]
    reference["relative_path"] = "outputs//complete/artifact.json"
    with pytest.raises(ValueError, match="repository-relative"):
        build_complete_prototype_manifest(tmp_path, index, validators=validators)

    with pytest.raises(ValueError, match="SHA-256"):
        empty_complete_prototype_evidence_index(
            prototype_configuration_sha256="not-a-digest",
            paper_delight_enabled=False,
            kondo_enabled=False,
        )


def test_partial_or_reordered_roles_cannot_be_renormalized(tmp_path: Path) -> None:
    index, validators = _fixture(tmp_path)
    rows = cast(list[dict[str, object]], index["rows"])
    resource_row = next(
        row for row in rows if row["property_id"] == "temporal_resource_bounds"
    )
    evidence = cast(list[dict[str, object]], resource_row["evidence"])
    resource_row["evidence"] = evidence[:1]
    with pytest.raises(ValueError, match="roles must be exact"):
        build_complete_prototype_manifest(tmp_path, index, validators=validators)

    index, validators = _fixture(tmp_path)
    rows = cast(list[dict[str, object]], index["rows"])
    resource_row = next(
        row for row in rows if row["property_id"] == "temporal_resource_bounds"
    )
    evidence = cast(list[dict[str, object]], resource_row["evidence"])
    resource_row["evidence"] = list(reversed(evidence))
    with pytest.raises(ValueError, match="roles must be exact"):
        build_complete_prototype_manifest(tmp_path, index, validators=validators)


def test_invalid_manifest_schema_maps_to_exit_two() -> None:
    assert complete_prototype_manifest_exit_code({"overall_status": "accepted"}) == 2


def test_symlink_escape_and_malformed_receipt_fail_closed(tmp_path: Path) -> None:
    index, validators = _fixture(tmp_path)
    outside = tmp_path.parent / f"{tmp_path.name}.outside.json"
    outside.write_text('{"mode":"accepted"}\n', encoding="utf-8")
    escape = tmp_path / "outputs/complete/escape.json"
    escape.symlink_to(outside)
    rows = cast(list[dict[str, object]], index["rows"])
    reference = cast(list[dict[str, object]], rows[0]["evidence"])[0]
    reference["relative_path"] = "outputs/complete/escape.json"
    reference["artifact_sha256"] = hashlib.sha256(outside.read_bytes()).hexdigest()
    escaped = build_complete_prototype_manifest(tmp_path, index, validators=validators)
    assert escaped["overall_status"] == "invalid"

    clean_index, clean_validators = _fixture(tmp_path)
    registered = clean_validators[VALIDATOR_ID]

    def malformed(_: Path) -> CompletePrototypeValidationReceipt:
        return replace(_receipt(), valid=cast(bool, 1))

    clean_validators[VALIDATOR_ID] = replace(registered, validate=malformed)
    malformed_manifest = build_complete_prototype_manifest(
        tmp_path,
        clean_index,
        validators=clean_validators,
    )
    assert malformed_manifest["overall_status"] == "invalid"
