"""Adversarial tests for non-authorizing matched-v3 external source publication.

All bundles are small synthetic canonical USTARs.  No candidate source is imported or
executed, no benchmark runs, and no evidence or qualification authority is created.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pytest

from alberta_framework.benchmarks import (
    forager_matched_v3_external_materialization as materialization,
)
from alberta_framework.benchmarks import (
    forager_matched_v3_external_source_publication as publication,
)
from alberta_framework.benchmarks import forager_matched_v3_external_staging as staging

pytestmark = pytest.mark.unit


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical_json(value: object, *, newline: bool = True) -> bytes:
    raw = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return raw + (b"\n" if newline else b"")


@dataclass(frozen=True)
class _SyntheticStage:
    archive_raw: bytes
    archive_sha256: str
    manifest: dict[str, Any]
    manifest_raw: bytes
    manifest_sha256: str
    base_manifest_raw: bytes
    base_manifest: dict[str, Any]


def _synthetic_stage(monkeypatch: pytest.MonkeyPatch) -> _SyntheticStage:
    frozen_records: list[tuple[str, str, str, str]] = []
    overlays: list[dict[str, Any]] = []
    inventory: list[dict[str, Any]] = []
    member_raw_by_path: dict[str, bytes] = {}
    original_total = 0
    for index, frozen in enumerate(staging._FROZEN_EXECUTION_RECORDS):
        candidate_id, path, _original_sha256, _derived_sha256 = frozen
        original_raw = f"original-{index}".encode("ascii")
        derived_raw = _canonical_json({"candidate": candidate_id, "index": index})
        original_sha256 = _sha256(original_raw)
        derived_sha256 = _sha256(derived_raw)
        frozen_records.append((candidate_id, path, original_sha256, derived_sha256))
        overlays.append(
            {
                "archive_mode": "0444",
                "candidate_id": candidate_id,
                "derived_sha256": derived_sha256,
                "derived_size_bytes": len(derived_raw),
                "original_sha256": original_sha256,
                "original_size_bytes": len(original_raw),
                "path": path,
                "transform_descriptor_sha256": staging._expected_transform_sha256(candidate_id),
            }
        )
        inventory.append(
            staging._inventory_record(
                path=path,
                size_bytes=len(derived_raw),
                sha256=derived_sha256,
                mode="0444",
                provenance="derived_configuration_overlay",
            )
        )
        member_raw_by_path[path] = derived_raw
        original_total += len(original_raw)

    alias_raw = b"same portable-alias bytes\n"
    for alias_path in staging._PINNED_PORTABLE_ALIAS_GROUPS[0]:
        inventory.append(
            staging._inventory_record(
                path=alias_path,
                size_bytes=len(alias_raw),
                sha256=_sha256(alias_raw),
                mode="0444",
                provenance="materializer_v2_regular_file",
            )
        )
        member_raw_by_path[alias_path] = alias_raw
        original_total += len(alias_raw)
    source_file_count = len(overlays) + len(staging._PINNED_PORTABLE_ALIAS_GROUPS[0])

    base_manifest: dict[str, Any] = {
        "schema_version": materialization.EXTERNAL_MATERIALIZATION_SCHEMA_VERSION,
        "identity_sha256": materialization.PINNED_EXTERNAL_MATERIALIZATION_IDENTITY_SHA256,
        "payload_sha256": _sha256(b"synthetic-base-body"),
        "identity": {
            "commit_git_sha1": "9710f60fa30da5badc451ad7ce3ff296d5070830",
            "tree_git_sha1": "a5ad878ac4be0567c43dfd9177471c4b5a910bfa",
        },
        "source_tree": {
            "tracked_entry_count": source_file_count + 1,
            "materialized_regular_file_count": source_file_count,
            "excluded_gitlink_count": 1,
            "materialized_total_size_bytes": original_total,
        },
        "claims": {"authority_granted": False},
    }
    base_manifest_raw = _canonical_json(base_manifest, newline=False)
    inventory.append(
        staging._inventory_record(
            path=staging.EXTERNAL_STAGING_MATERIALIZER_MANIFEST_PATH,
            size_bytes=len(base_manifest_raw),
            sha256=_sha256(base_manifest_raw),
            mode="0444",
            provenance="relocated_exact_materializer_v2_manifest",
        )
    )
    member_raw_by_path[staging.EXTERNAL_STAGING_MATERIALIZER_MANIFEST_PATH] = base_manifest_raw
    inventory.sort(key=lambda item: cast(str, item["path"]).encode("utf-8"))
    nonself_bytes = sum(cast(int, item["size_bytes"]) for item in inventory)
    monkeypatch.setattr(staging, "_FROZEN_EXECUTION_RECORDS", tuple(frozen_records))
    body: dict[str, Any] = {
        "archive_layout": {
            "complete_member_count": len(inventory) + 1,
            "final_manifest_mode": "0444",
            "final_manifest_path": staging.EXTERNAL_STAGING_FINAL_MANIFEST_PATH,
            "final_manifest_self_excluded_from_payload_inventory": True,
            "format": "canonical_posix_ustar_uncompressed",
            "member_order": "ascending_utf8_path_bytes",
            "nonself_member_count": len(inventory),
            "nonself_payload_bytes": nonself_bytes,
            "record_size_bytes": 10_240,
        },
        "base_materialization": {
            "identity_sha256": staging._MATERIALIZER_IDENTITY_SHA256,
            "manifest_attestation_path": staging.EXTERNAL_STAGING_MATERIALIZER_MANIFEST_PATH,
            "manifest_root_path_removed": staging._MATERIALIZER_MANIFEST_FILENAME,
            "manifest_schema_version": staging._MATERIALIZER_SCHEMA_VERSION,
            "manifest_sha256": _sha256(base_manifest_raw),
            "manifest_size_bytes": len(base_manifest_raw),
            "source_materialized_total_size_bytes": original_total,
            "source_regular_file_count": source_file_count,
        },
        "claims": staging._claims(),
        "classification": "sealed_external_source_staging_non_authorizing",
        "configuration_overlays": overlays,
        "execution_contract": {
            "candidate_count": len(staging._CANDIDATE_IDS),
            "candidate_order": list(staging._CANDIDATE_IDS),
            "descriptor_sha256": staging._EXECUTION_CONTRACT_SHA256,
            "schema_version": staging._EXECUTION_CONTRACT_SCHEMA_VERSION,
        },
        "implementation_source_sha256": staging._IMPORTED_IMPLEMENTATION_SOURCE_SHA256,
        "limitations": staging._limitations(),
        "payload_inventory": inventory,
        "schema_version": staging.EXTERNAL_STAGING_MANIFEST_SCHEMA_VERSION,
        "staging_contract_descriptor_sha256": (staging.EXTERNAL_STAGING_CONTRACT_DESCRIPTOR_SHA256),
        "status": staging.EXTERNAL_STAGING_STATUS,
    }
    body["manifest_body_sha256"] = _sha256(_canonical_json(body))
    manifest_raw = _canonical_json(body)
    manifest_sha256 = _sha256(manifest_raw)
    manifest = staging.parse_external_staging_manifest(
        manifest_raw,
        expected_manifest_sha256=manifest_sha256,
    )
    member_raw_by_path[staging.EXTERNAL_STAGING_FINAL_MANIFEST_PATH] = manifest_raw
    members = [
        staging._ArchiveMember(
            path=path,
            size_bytes=len(raw),
            sha256=_sha256(raw),
            mode=0o444,
            raw=raw,
        )
        for path, raw in sorted(
            member_raw_by_path.items(), key=lambda item: item[0].encode("utf-8")
        )
    ]
    descriptor = staging._create_private_memfd("synthetic-external-source")
    try:
        archive_size, archive_sha256 = staging._write_canonical_ustar(
            descriptor,
            members,
            allowed_alias_groups=staging._PINNED_PORTABLE_ALIAS_GROUPS,
        )
        archive_raw = os.pread(descriptor, archive_size, 0)
    finally:
        os.close(descriptor)

    def parse_synthetic_base(raw: bytes, *, expected_manifest_sha256: str) -> dict[str, Any]:
        assert raw == base_manifest_raw
        assert expected_manifest_sha256 == _sha256(base_manifest_raw)
        return copy.deepcopy(base_manifest)

    monkeypatch.setattr(
        materialization,
        "parse_matched_v3_external_materialization_manifest",
        parse_synthetic_base,
    )
    return _SyntheticStage(
        archive_raw=archive_raw,
        archive_sha256=archive_sha256,
        manifest=manifest,
        manifest_raw=manifest_raw,
        manifest_sha256=manifest_sha256,
        base_manifest_raw=base_manifest_raw,
        base_manifest=base_manifest,
    )


@pytest.fixture
def synthetic_stage(monkeypatch: pytest.MonkeyPatch) -> _SyntheticStage:
    return _synthetic_stage(monkeypatch)


def _retained(
    synthetic: _SyntheticStage,
) -> staging.RetainedExternalStagingBundle:
    descriptor = staging._sealed_bytes_fd(
        synthetic.archive_raw,
        "synthetic-external-source-retained",
        len(synthetic.archive_raw),
    )
    metadata = os.fstat(descriptor)
    return staging.RetainedExternalStagingBundle(
        staging._RETAINED_BUNDLE_CREATION_TOKEN,
        descriptor,
        metadata.st_dev,
        metadata.st_ino,
        len(synthetic.archive_raw),
        synthetic.archive_sha256,
        synthetic.manifest_raw,
        synthetic.manifest_sha256,
    )


def _root(tmp_path: Path) -> Path:
    root = tmp_path / "publications"
    root.mkdir(mode=0o755)
    return root


def _publish(
    synthetic: _SyntheticStage,
    root: Path,
) -> publication.PublishedMatchedV3ExternalSource:
    retained = _retained(synthetic)
    try:
        return publication.publish_matched_v3_external_source(
            retained,
            root,
            authorize_non_evidence_publication=True,
        )
    finally:
        retained.close()


def _receipt_payload(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="ascii")))


def _rewrite_receipt(payload: dict[str, Any]) -> tuple[bytes, str]:
    body = copy.deepcopy(payload)
    body.pop("receipt_body_sha256", None)
    body_sha256 = _sha256(_canonical_json(body, newline=False))
    body["receipt_body_sha256"] = body_sha256
    raw = _canonical_json(body)
    return raw, _sha256(raw)


def test_descriptor_and_receipt_claims_are_exact_false() -> None:
    descriptor = publication.external_source_publication_contract_descriptor()
    assert descriptor["claims"]
    assert all(value is False for value in descriptor["claims"].values())
    assert descriptor["dependency"]["staging_descriptor_sha256"] == (
        staging.EXTERNAL_STAGING_CONTRACT_DESCRIPTOR_SHA256
    )
    raw = publication.canonical_external_source_publication_contract_descriptor_bytes()
    assert _sha256(raw) == publication.EXTERNAL_SOURCE_PUBLICATION_CONTRACT_DESCRIPTOR_SHA256
    assert publication.parse_external_source_publication_contract_descriptor(raw) == descriptor


def test_publisher_rejects_missing_authorization_ducks_subclasses_and_closed_capability(
    synthetic_stage: _SyntheticStage,
    tmp_path: Path,
) -> None:
    root = _root(tmp_path)
    retained = _retained(synthetic_stage)
    try:
        with pytest.raises(publication.ForagerMatchedV3ExternalSourcePublicationError):
            publication.publish_matched_v3_external_source(
                retained,
                root,
                authorize_non_evidence_publication=False,
            )
        with pytest.raises(publication.ForagerMatchedV3ExternalSourcePublicationError):
            publication.publish_matched_v3_external_source(
                cast(Any, object()),
                root,
                authorize_non_evidence_publication=True,
            )

        class Subclass(staging.RetainedExternalStagingBundle):
            pass

        subclass = object.__new__(Subclass)
        with pytest.raises(publication.ForagerMatchedV3ExternalSourcePublicationError):
            publication.publish_matched_v3_external_source(
                cast(Any, subclass),
                root,
                authorize_non_evidence_publication=True,
            )
    finally:
        retained.close()
    with pytest.raises(publication.ForagerMatchedV3ExternalSourcePublicationError):
        publication.publish_matched_v3_external_source(
            retained,
            root,
            authorize_non_evidence_publication=True,
        )


def test_valid_publication_is_content_addressed_read_only_and_fully_replays(
    synthetic_stage: _SyntheticStage,
    tmp_path: Path,
) -> None:
    result = _publish(synthetic_stage, _root(tmp_path))
    assert result.directory.name == synthetic_stage.archive_sha256
    assert result.archive.name == publication.EXTERNAL_SOURCE_ARCHIVE_FILENAME
    assert result.receipt.name == publication.EXTERNAL_SOURCE_RECEIPT_FILENAME
    assert sorted(path.name for path in result.directory.iterdir()) == [
        publication.EXTERNAL_SOURCE_ARCHIVE_FILENAME,
        publication.EXTERNAL_SOURCE_RECEIPT_FILENAME,
    ]
    assert stat.S_IMODE(result.directory.stat().st_mode) == 0o555
    for path in (result.archive, result.receipt):
        metadata = path.stat()
        assert stat.S_ISREG(metadata.st_mode)
        assert metadata.st_nlink == 1
        assert stat.S_IMODE(metadata.st_mode) == 0o444
    assert _sha256(result.archive.read_bytes()) == synthetic_stage.archive_sha256
    receipt = publication.validate_published_matched_v3_external_source(
        result.directory,
        expected_receipt_sha256=result.receipt_sha256,
        expected_archive_sha256=result.archive_sha256,
    )
    assert (
        receipt["archive"]["member_count"]
        == (synthetic_stage.manifest["archive_layout"]["complete_member_count"])
    )
    assert all(value is False for value in receipt["claims"].values())


def test_receipt_binds_stage_base_inventory_payload_and_archive(
    synthetic_stage: _SyntheticStage,
    tmp_path: Path,
) -> None:
    result = _publish(synthetic_stage, _root(tmp_path))
    raw = result.receipt.read_bytes()
    receipt = publication.parse_external_source_publication_receipt(
        raw,
        expected_file_sha256=result.receipt_sha256,
    )
    assert receipt["archive"]["sha256"] == synthetic_stage.archive_sha256
    assert receipt["archive"]["size_bytes"] == len(synthetic_stage.archive_raw)
    assert receipt["archive"]["payload_size_bytes"] == sum(
        item["size_bytes"] for item in receipt["archive"]["members"]
    )
    assert receipt["staging_manifest"] == {
        "body_sha256": synthetic_stage.manifest["manifest_body_sha256"],
        "full_file_sha256": synthetic_stage.manifest_sha256,
        "schema_version": staging.EXTERNAL_STAGING_MANIFEST_SCHEMA_VERSION,
        "size_bytes": len(synthetic_stage.manifest_raw),
        "status": staging.EXTERNAL_STAGING_STATUS,
    }
    assert receipt["external_source_manifest"]["full_file_sha256"] == _sha256(
        synthetic_stage.base_manifest_raw
    )
    assert (
        receipt["external_source_manifest"]["payload_sha256"]
        == (synthetic_stage.base_manifest["payload_sha256"])
    )


def test_collision_refuses_even_identical_existing_destination_and_cleans_partial(
    synthetic_stage: _SyntheticStage,
    tmp_path: Path,
) -> None:
    root = _root(tmp_path)
    first = _publish(synthetic_stage, root)
    before = first.receipt.read_bytes()
    retained = _retained(synthetic_stage)
    try:
        with pytest.raises(FileExistsError, match="overwrite"):
            publication.publish_matched_v3_external_source(
                retained,
                root,
                authorize_non_evidence_publication=True,
            )
    finally:
        retained.close()
    assert first.receipt.read_bytes() == before
    namespace = root / "sha256"
    assert not tuple(namespace.glob(".staging-*"))


def test_partial_archive_write_failure_cleans_owned_staging(
    synthetic_stage: _SyntheticStage,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _root(tmp_path)
    retained = _retained(synthetic_stage)
    real_write = os.write
    calls = 0

    def partial_then_fail(descriptor: int, raw: object) -> int:
        nonlocal calls
        calls += 1
        if calls == 1:
            view = memoryview(cast(Any, raw))
            return int(real_write(descriptor, view[: max(1, len(view) // 2)]))
        raise OSError("injected partial write")

    monkeypatch.setattr(os, "write", partial_then_fail)
    try:
        with pytest.raises(publication.ForagerMatchedV3ExternalSourcePublicationError):
            publication.publish_matched_v3_external_source(
                retained,
                root,
                authorize_non_evidence_publication=True,
            )
    finally:
        retained.close()
    namespace = root / "sha256"
    assert namespace.exists()
    assert list(namespace.iterdir()) == []


def test_published_archive_substitution_is_rejected(
    synthetic_stage: _SyntheticStage,
    tmp_path: Path,
) -> None:
    result = _publish(synthetic_stage, _root(tmp_path))
    os.chmod(result.directory, 0o755)
    os.chmod(result.archive, 0o644)
    with result.archive.open("r+b") as stream:
        first = stream.read(1)
        stream.seek(0)
        stream.write(bytes([first[0] ^ 1]))
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(result.archive, 0o444)
    os.chmod(result.directory, 0o555)
    with pytest.raises(publication.ForagerMatchedV3ExternalSourcePublicationError):
        publication.validate_published_matched_v3_external_source(
            result.directory,
            expected_receipt_sha256=result.receipt_sha256,
            expected_archive_sha256=result.archive_sha256,
        )


def test_independent_raw_ustar_replay_rejects_structural_mutations(
    synthetic_stage: _SyntheticStage,
) -> None:
    members = publication._complete_inventory(
        synthetic_stage.manifest_raw,
        synthetic_stage.manifest,
    )
    content_size = sum(512 + item["size_bytes"] + (-item["size_bytes"]) % 512 for item in members)
    padding_offset: int | None = None
    cursor = 0
    for item in members:
        cursor += 512 + item["size_bytes"]
        if item["size_bytes"] % 512:
            padding_offset = cursor
            break
        cursor += (-item["size_bytes"]) % 512
    assert padding_offset is not None
    assert content_size + 1024 < len(synthetic_stage.archive_raw)

    for offset in (0, 512, padding_offset, content_size, len(synthetic_stage.archive_raw) - 1):
        mutated = bytearray(synthetic_stage.archive_raw)
        mutated[offset] ^= 1
        raw = bytes(mutated)
        descriptor = staging._sealed_bytes_fd(raw, "mutated-ustar", len(raw))
        try:
            with pytest.raises(publication.ForagerMatchedV3ExternalSourcePublicationError):
                publication._verify_external_source_ustar_fd(
                    descriptor,
                    expected_size=len(raw),
                    expected_sha256=_sha256(raw),
                    members=members,
                )
        finally:
            os.close(descriptor)


def test_coherent_receipt_substitution_still_requires_external_file_pin(
    synthetic_stage: _SyntheticStage,
    tmp_path: Path,
) -> None:
    result = _publish(synthetic_stage, _root(tmp_path))
    payload = _receipt_payload(result.receipt)
    payload["limitations"] = list(payload["limitations"]) + ["attacker addition"]
    raw, changed_sha256 = _rewrite_receipt(payload)
    os.chmod(result.directory, 0o755)
    os.chmod(result.receipt, 0o644)
    result.receipt.write_bytes(raw)
    os.chmod(result.receipt, 0o444)
    os.chmod(result.directory, 0o555)
    assert changed_sha256 != result.receipt_sha256
    with pytest.raises(publication.ForagerMatchedV3ExternalSourcePublicationError):
        publication.validate_published_matched_v3_external_source(
            result.directory,
            expected_receipt_sha256=result.receipt_sha256,
            expected_archive_sha256=result.archive_sha256,
        )


def test_coherent_inventory_receipt_substitution_fails_archive_replay(
    synthetic_stage: _SyntheticStage,
    tmp_path: Path,
) -> None:
    result = _publish(synthetic_stage, _root(tmp_path))
    payload = _receipt_payload(result.receipt)
    members = cast(list[dict[str, Any]], payload["archive"]["members"])
    ordinary = next(
        item
        for item in members
        if item["path"]
        not in {
            staging.EXTERNAL_STAGING_FINAL_MANIFEST_PATH,
            staging.EXTERNAL_STAGING_MATERIALIZER_MANIFEST_PATH,
        }
    )
    ordinary["sha256"] = "0" * 64
    payload["archive"]["inventory_sha256"] = _sha256(_canonical_json(members, newline=False))
    raw, changed_sha256 = _rewrite_receipt(payload)
    publication.parse_external_source_publication_receipt(
        raw,
        expected_file_sha256=changed_sha256,
    )

    os.chmod(result.directory, 0o755)
    os.chmod(result.receipt, 0o644)
    result.receipt.write_bytes(raw)
    os.chmod(result.receipt, 0o444)
    os.chmod(result.directory, 0o555)
    with pytest.raises(publication.ForagerMatchedV3ExternalSourcePublicationError):
        publication.validate_published_matched_v3_external_source(
            result.directory,
            expected_receipt_sha256=changed_sha256,
            expected_archive_sha256=result.archive_sha256,
        )


def test_receipt_parser_rejects_duplicates_floats_and_noncanonical_bytes() -> None:
    for raw in (
        b'{"a":1,"a":2}\n',
        b'{"a":1.0}\n',
        b'{ "a":1}\n',
    ):
        with pytest.raises(publication.ForagerMatchedV3ExternalSourcePublicationError):
            publication.parse_external_source_publication_receipt(
                raw,
                expected_file_sha256=_sha256(raw),
            )


def test_group_writable_or_symlink_publication_root_is_rejected(
    synthetic_stage: _SyntheticStage,
    tmp_path: Path,
) -> None:
    unsafe = tmp_path / "unsafe"
    unsafe.mkdir(mode=0o775)
    unsafe.chmod(0o775)
    retained = _retained(synthetic_stage)
    try:
        with pytest.raises(publication.ForagerMatchedV3ExternalSourcePublicationError):
            publication.publish_matched_v3_external_source(
                retained,
                unsafe,
                authorize_non_evidence_publication=True,
            )
        symlink = tmp_path / "link"
        symlink.symlink_to(unsafe, target_is_directory=True)
        with pytest.raises(publication.ForagerMatchedV3ExternalSourcePublicationError):
            publication.publish_matched_v3_external_source(
                retained,
                symlink,
                authorize_non_evidence_publication=True,
            )
    finally:
        retained.close()


def test_symlink_sha256_namespace_is_rejected(
    synthetic_stage: _SyntheticStage,
    tmp_path: Path,
) -> None:
    root = _root(tmp_path)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir(mode=0o755)
    (root / "sha256").symlink_to(elsewhere, target_is_directory=True)
    retained = _retained(synthetic_stage)
    try:
        with pytest.raises(publication.ForagerMatchedV3ExternalSourcePublicationError):
            publication.publish_matched_v3_external_source(
                retained,
                root,
                authorize_non_evidence_publication=True,
            )
    finally:
        retained.close()
    assert list(elsewhere.iterdir()) == []


@pytest.mark.parametrize(
    "attack",
    ["archive_symlink", "receipt_hardlink", "archive_wrong_mode", "extra_entry"],
)
def test_published_filesystem_substitution_is_rejected(
    synthetic_stage: _SyntheticStage,
    tmp_path: Path,
    attack: str,
) -> None:
    result = _publish(synthetic_stage, _root(tmp_path))
    os.chmod(result.directory, 0o755)
    if attack == "archive_symlink":
        outside = tmp_path / "outside.tar"
        outside.write_bytes(synthetic_stage.archive_raw)
        outside.chmod(0o444)
        result.archive.unlink()
        result.archive.symlink_to(outside)
    elif attack == "receipt_hardlink":
        os.link(result.receipt, tmp_path / "receipt-hardlink.json")
    elif attack == "archive_wrong_mode":
        result.archive.chmod(0o644)
    else:
        (result.directory / "unexpected").write_bytes(b"unexpected\n")
    os.chmod(result.directory, 0o555)
    with pytest.raises(publication.ForagerMatchedV3ExternalSourcePublicationError):
        publication.validate_published_matched_v3_external_source(
            result.directory,
            expected_receipt_sha256=result.receipt_sha256,
            expected_archive_sha256=result.archive_sha256,
        )


def test_rename_moves_then_reports_error_is_uncertain_and_preserves_bytes(
    synthetic_stage: _SyntheticStage,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _root(tmp_path)
    real_rename = publication._rename_new_only

    def move_then_fail(directory_fd: int, source: str, target: str) -> None:
        real_rename(directory_fd, source, target)
        raise OSError("injected ambiguous rename report")

    monkeypatch.setattr(publication, "_rename_new_only", move_then_fail)
    retained = _retained(synthetic_stage)
    try:
        with pytest.raises(publication.PublishedMatchedV3ExternalSourceUncertainError) as caught:
            publication.publish_matched_v3_external_source(
                retained,
                root,
                authorize_non_evidence_publication=True,
            )
    finally:
        retained.close()
    destination = root / "sha256" / synthetic_stage.archive_sha256
    assert caught.value.destination == destination
    assert destination.is_dir()
    assert sorted(path.name for path in destination.iterdir()) == [
        publication.EXTERNAL_SOURCE_ARCHIVE_FILENAME,
        publication.EXTERNAL_SOURCE_RECEIPT_FILENAME,
    ]


def test_postcommit_replay_failure_is_uncertain_and_preserves_destination(
    synthetic_stage: _SyntheticStage,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _root(tmp_path)
    real_validate = publication._validate_published_directory_fd
    calls = 0

    def fail_second(*args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected postcommit replay failure")
        return real_validate(*args, **kwargs)

    monkeypatch.setattr(publication, "_validate_published_directory_fd", fail_second)
    retained = _retained(synthetic_stage)
    try:
        with pytest.raises(publication.PublishedMatchedV3ExternalSourceUncertainError):
            publication.publish_matched_v3_external_source(
                retained,
                root,
                authorize_non_evidence_publication=True,
            )
    finally:
        retained.close()
    destination = root / "sha256" / synthetic_stage.archive_sha256
    assert destination.is_dir()
    assert not tuple((root / "sha256").glob(".staging-*"))


def test_precommit_source_capability_invalidation_cleans_partial(
    synthetic_stage: _SyntheticStage,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _root(tmp_path)
    real_copy = publication._copy_retained_archive

    def copy_then_close(
        retained: staging.RetainedExternalStagingBundle,
        directory_fd: int,
    ) -> None:
        real_copy(retained, directory_fd)
        retained.close()

    monkeypatch.setattr(publication, "_copy_retained_archive", copy_then_close)
    retained = _retained(synthetic_stage)
    with pytest.raises(publication.ForagerMatchedV3ExternalSourcePublicationError):
        publication.publish_matched_v3_external_source(
            retained,
            root,
            authorize_non_evidence_publication=True,
        )
    assert list((root / "sha256").iterdir()) == []


def test_precommit_cleanup_failure_is_explicit_and_preserves_residual_for_audit(
    synthetic_stage: _SyntheticStage,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _root(tmp_path)

    def fail_copy(
        retained: staging.RetainedExternalStagingBundle,
        directory_fd: int,
    ) -> None:
        del retained, directory_fd
        raise publication.ForagerMatchedV3ExternalSourcePublicationError(
            "injected precommit failure"
        )

    monkeypatch.setattr(publication, "_copy_retained_archive", fail_copy)
    monkeypatch.setattr(publication, "_cleanup_owned_staging", lambda *_args: False)
    retained = _retained(synthetic_stage)
    try:
        with pytest.raises(
            publication.ForagerMatchedV3ExternalSourcePublicationError,
            match="injected precommit failure",
        ) as caught:
            publication.publish_matched_v3_external_source(
                retained,
                root,
                authorize_non_evidence_publication=True,
            )
    finally:
        retained.close()
    assert any("partial staging directory may remain" in note for note in caught.value.__notes__)
    assert len(tuple((root / "sha256").glob(".staging-*"))) == 1
