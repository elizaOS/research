"""Adversarial unit tests for the matched-v3 external staging boundary.

All archive fixtures are synthetic and bounded.  These tests do not inspect a
real external checkout, launch candidate code, execute a workload, write below
``outputs/``, or grant execution or acceptance authority.
"""

from __future__ import annotations

import ast
import copy
import fcntl
import hashlib
import inspect
import json
import os
import pickle
import signal
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from alberta_framework.benchmarks import (
    forager_matched_v3_external_materialization as materialization,
)
from alberta_framework.benchmarks import forager_matched_v3_external_staging as staging

pytestmark = pytest.mark.unit

_EXPECTED_DESCRIPTOR_SHA256 = "ceea86b38822f3add0465788003d349dd221a49fba5f3fa069bfec985537caea"
_EXPECTED_SOURCE_SHA256 = "675d54edcf2f87c1847712e7a480e2e5134312d040a68a1102c10c4829f8fba0"


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _fd_count() -> int:
    return len(os.listdir("/proc/self/fd"))


def _reencode_manifest(value: dict[str, Any]) -> bytes:
    body = copy.deepcopy(value)
    body.pop("manifest_body_sha256", None)
    body["manifest_body_sha256"] = _sha256(staging._canonical_json(body))
    return staging._canonical_json(body)


@dataclass(frozen=True)
class _SyntheticClosure:
    manifest: dict[str, Any]
    manifest_raw: bytes
    manifest_sha256: str
    archive_raw: bytes
    archive_sha256: str
    records: tuple[tuple[str, str, str, str], ...]
    member_raw_by_path: dict[str, bytes]


def _synthetic_closure(monkeypatch: pytest.MonkeyPatch) -> _SyntheticClosure:
    records: list[tuple[str, str, str, str]] = []
    overlays: list[dict[str, Any]] = []
    inventory: list[dict[str, Any]] = []
    member_raw_by_path: dict[str, bytes] = {}
    source_materialized_total = 0

    for index, frozen in enumerate(staging._FROZEN_EXECUTION_RECORDS):
        candidate_id, path, _frozen_original, _frozen_derived = frozen
        original_raw = f"synthetic-original-{index}".encode("ascii")
        derived_raw = json.dumps(
            {"candidate": candidate_id, "index": index},
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        original_sha256 = _sha256(original_raw)
        derived_sha256 = _sha256(derived_raw)
        records.append((candidate_id, path, original_sha256, derived_sha256))
        overlays.append(
            {
                "candidate_id": candidate_id,
                "path": path,
                "original_size_bytes": len(original_raw),
                "original_sha256": original_sha256,
                "derived_size_bytes": len(derived_raw),
                "derived_sha256": derived_sha256,
                "transform_descriptor_sha256": staging._expected_transform_sha256(candidate_id),
                "archive_mode": "0444",
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
        source_materialized_total += len(original_raw)

    alias_raw = b"same bytes, separate members\n"
    for path in staging._PINNED_PORTABLE_ALIAS_GROUPS[0]:
        inventory.append(
            staging._inventory_record(
                path=path,
                size_bytes=len(alias_raw),
                sha256=_sha256(alias_raw),
                mode="0444",
                provenance="materializer_v2_regular_file",
            )
        )
        member_raw_by_path[path] = alias_raw
        source_materialized_total += len(alias_raw)

    executable_path = "src/synthetic_tool.py"
    executable_raw = b"#!/usr/bin/env python3\nraise SystemExit(99)\n"
    inventory.append(
        staging._inventory_record(
            path=executable_path,
            size_bytes=len(executable_raw),
            sha256=_sha256(executable_raw),
            mode="0555",
            provenance="materializer_v2_regular_file",
        )
    )
    member_raw_by_path[executable_path] = executable_raw
    source_materialized_total += len(executable_raw)

    base_manifest_raw = b'{"synthetic_materializer_manifest":true}'
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
    inventory.sort(key=lambda item: item["path"].encode("utf-8"))
    source_count = len(records) + len(staging._PINNED_PORTABLE_ALIAS_GROUPS[0]) + 1
    nonself_total = sum(item["size_bytes"] for item in inventory)

    monkeypatch.setattr(staging, "_FROZEN_EXECUTION_RECORDS", tuple(records))
    body: dict[str, Any] = {
        "schema_version": staging.EXTERNAL_STAGING_MANIFEST_SCHEMA_VERSION,
        "status": staging.EXTERNAL_STAGING_STATUS,
        "classification": "sealed_external_source_staging_non_authorizing",
        "staging_contract_descriptor_sha256": (staging.EXTERNAL_STAGING_CONTRACT_DESCRIPTOR_SHA256),
        "implementation_source_sha256": staging._IMPORTED_IMPLEMENTATION_SOURCE_SHA256,
        "execution_contract": {
            "schema_version": staging._EXECUTION_CONTRACT_SCHEMA_VERSION,
            "descriptor_sha256": staging._EXECUTION_CONTRACT_SHA256,
            "candidate_count": len(staging._CANDIDATE_IDS),
            "candidate_order": list(staging._CANDIDATE_IDS),
        },
        "base_materialization": {
            "manifest_schema_version": staging._MATERIALIZER_SCHEMA_VERSION,
            "identity_sha256": staging._MATERIALIZER_IDENTITY_SHA256,
            "manifest_root_path_removed": staging._MATERIALIZER_MANIFEST_FILENAME,
            "manifest_attestation_path": (staging.EXTERNAL_STAGING_MATERIALIZER_MANIFEST_PATH),
            "manifest_size_bytes": len(base_manifest_raw),
            "manifest_sha256": _sha256(base_manifest_raw),
            "source_regular_file_count": source_count,
            "source_materialized_total_size_bytes": source_materialized_total,
        },
        "configuration_overlays": overlays,
        "payload_inventory": inventory,
        "archive_layout": {
            "format": "canonical_posix_ustar_uncompressed",
            "nonself_member_count": len(inventory),
            "nonself_payload_bytes": nonself_total,
            "final_manifest_path": staging.EXTERNAL_STAGING_FINAL_MANIFEST_PATH,
            "final_manifest_mode": "0444",
            "final_manifest_self_excluded_from_payload_inventory": True,
            "complete_member_count": len(inventory) + 1,
            "member_order": "ascending_utf8_path_bytes",
            "record_size_bytes": staging._USTAR_RECORD_BYTES,
        },
        "claims": staging._claims(),
        "limitations": staging._limitations(),
    }
    manifest_raw = _reencode_manifest(body)
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
            mode=(0o555 if path == executable_path else 0o444),
            raw=raw,
        )
        for path, raw in sorted(
            member_raw_by_path.items(), key=lambda item: item[0].encode("utf-8")
        )
    ]
    descriptor = staging._create_private_memfd("synthetic-ustar")
    try:
        archive_size, archive_sha256 = staging._write_canonical_ustar(
            descriptor,
            members,
            allowed_alias_groups=staging._PINNED_PORTABLE_ALIAS_GROUPS,
        )
        archive_raw = os.pread(descriptor, archive_size, 0)
    finally:
        os.close(descriptor)
    assert _sha256(archive_raw) == archive_sha256
    return _SyntheticClosure(
        manifest=manifest,
        manifest_raw=manifest_raw,
        manifest_sha256=manifest_sha256,
        archive_raw=archive_raw,
        archive_sha256=archive_sha256,
        records=tuple(records),
        member_raw_by_path=member_raw_by_path,
    )


@pytest.fixture
def synthetic_closure(monkeypatch: pytest.MonkeyPatch) -> _SyntheticClosure:
    return _synthetic_closure(monkeypatch)


def _sealed_archive_descriptor(raw: bytes) -> int:
    return staging._sealed_bytes_fd(raw, "synthetic-sealed-archive", len(raw))


def _verify_archive(raw: bytes, closure: _SyntheticClosure) -> dict[str, Any]:
    descriptor = _sealed_archive_descriptor(raw)
    try:
        return staging._verify_canonical_ustar_fd(
            descriptor,
            expected_size=len(raw),
            expected_sha256=_sha256(raw),
            manifest_raw=closure.manifest_raw,
            manifest_sha256=closure.manifest_sha256,
        )
    finally:
        os.close(descriptor)


def test_descriptor_is_literal_digest_bound_and_detached() -> None:
    raw = staging.canonical_external_staging_contract_descriptor_bytes()
    assert _sha256(raw) == _EXPECTED_DESCRIPTOR_SHA256
    assert staging.EXTERNAL_STAGING_CONTRACT_DESCRIPTOR_SHA256 == (_EXPECTED_DESCRIPTOR_SHA256)
    assert staging.external_staging_contract_descriptor_sha256() == (_EXPECTED_DESCRIPTOR_SHA256)
    assert staging.parse_external_staging_contract_descriptor(raw) == (
        staging.external_staging_contract_descriptor()
    )

    first = staging.external_staging_contract_descriptor()
    second = staging.external_staging_contract_descriptor()
    first["claims"]["execution_authority_granted"] = True
    assert second["claims"]["execution_authority_granted"] is False
    changed = bytearray(raw)
    changed[-2] ^= 1
    with pytest.raises(staging.ForagerMatchedV3ExternalStagingError):
        staging.parse_external_staging_contract_descriptor(bytes(changed))


def test_source_identity_and_final_manifest_bind_imported_bytes(
    synthetic_closure: _SyntheticClosure,
) -> None:
    source_path = Path(staging.__file__)
    source_raw = source_path.read_bytes()
    assert _sha256(source_raw) == _EXPECTED_SOURCE_SHA256
    assert staging._IMPORTED_IMPLEMENTATION_SOURCE_SHA256 == _EXPECTED_SOURCE_SHA256
    assert synthetic_closure.manifest["implementation_source_sha256"] == (_EXPECTED_SOURCE_SHA256)
    reparsed = staging.parse_external_staging_manifest(
        synthetic_closure.manifest_raw,
        expected_manifest_sha256=synthetic_closure.manifest_sha256,
    )
    reparsed["claims"]["execution_authority_granted"] = True
    assert synthetic_closure.manifest["claims"]["execution_authority_granted"] is False


def test_descriptor_manifest_and_public_surface_deny_all_authority(
    synthetic_closure: _SyntheticClosure,
) -> None:
    descriptor = staging.external_staging_contract_descriptor()
    assert descriptor["claims"]
    assert all(value is False for value in descriptor["claims"].values())
    assert descriptor["apis"]
    assert all(value is False for value in descriptor["apis"].values())
    assert all(value is False for value in synthetic_closure.manifest["claims"].values())

    signature = inspect.signature(staging.stage_matched_v3_external_workload)
    assert list(signature.parameters) == ["retained_materialization"]
    prohibited_public = {
        "execute",
        "extract",
        "publish",
        "accept",
        "issue_seed",
        "load_result",
    }
    assert prohibited_public.isdisjoint(staging.__all__)
    assert not hasattr(staging.RetainedExternalStagingBundle, "extract")
    assert not hasattr(staging.RetainedExternalStagingBundle, "execute")
    assert not hasattr(staging.RetainedExternalStagingBundle, "publish")


def test_source_ast_has_no_runner_network_dynamic_execution_or_output_root() -> None:
    source = Path(staging.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_modules: set[str] = set()
    called_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_modules.add(node.module)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            called_names.add(node.func.id)
    assert not any(
        module.startswith(("docker", "socket", "requests", "urllib", "httpx"))
        for module in imported_modules
    )
    assert not any(
        token in source
        for token in (
            "forager_matched_executor",
            "forager_matched_v3_configuration_plan",
            "forager_matched_v3_local_runner",
            "forager_matched_v3_ppo_gru_runner",
            "forager_matched_v3_full_rainbow_runner",
        )
    )
    assert called_names.isdisjoint({"eval", "exec", "compile"})
    assert "outputs/" not in source


def test_strict_json_rejects_duplicates_floats_aliases_and_noncanonical_bytes() -> None:
    with pytest.raises(staging.ForagerMatchedV3ExternalStagingError, match="duplicate"):
        staging._strict_json_load(b'{"a":1,"a":2}\n', maximum_bytes=100, trailing_newline=True)
    with pytest.raises(staging.ForagerMatchedV3ExternalStagingError, match="float"):
        staging._strict_json_load(b'{"a":1.0}\n', maximum_bytes=100, trailing_newline=True)
    with pytest.raises(staging.ForagerMatchedV3ExternalStagingError, match="canonical"):
        staging._strict_json_load(b'{ "a":1}\n', maximum_bytes=100, trailing_newline=True)
    aliased: dict[str, Any] = {"x": []}
    aliased["y"] = aliased["x"]
    with pytest.raises(staging.ForagerMatchedV3ExternalStagingError, match="unaliased"):
        staging._canonical_json(aliased)


@pytest.mark.parametrize(
    "path",
    [
        "../escape",
        "/absolute",
        "a/./b",
        "a//b",
        "a\\b",
        "a\x00b",
        "Cafe\u0301/file",
        ".git/config",
        "AUX.txt",
        "tail./file",
        ("a" * 101) + "/" + ("b" * 101),
    ],
)
def test_path_validation_rejects_traversal_aliases_and_ustar_overflow(path: str) -> None:
    with pytest.raises(staging.ForagerMatchedV3ExternalStagingError):
        staging._validate_relative_path(path, "test path")


def test_canonical_ustar_prefix_split_and_exact_alias_policy() -> None:
    long_path = ("p" * 120) + "/" + ("n" * 80)
    prefix, name = staging._split_ustar_path(long_path)
    assert prefix == ("p" * 120).encode()
    assert name == ("n" * 80).encode()

    alias_paths = list(staging._PINNED_PORTABLE_ALIAS_GROUPS[0])
    staging._validate_path_set(
        alias_paths + ["safe/file.txt"],
        allowed_alias_groups=staging._PINNED_PORTABLE_ALIAS_GROUPS,
    )
    with pytest.raises(staging.ForagerMatchedV3ExternalStagingError, match="aliases"):
        staging._validate_path_set(alias_paths, allowed_alias_groups=())
    third = alias_paths[0].replace("NTKRank", "NtKrAnK")
    with pytest.raises(staging.ForagerMatchedV3ExternalStagingError, match="aliases"):
        staging._validate_path_set(
            alias_paths + [third],
            allowed_alias_groups=staging._PINNED_PORTABLE_ALIAS_GROUPS,
        )
    with pytest.raises(staging.ForagerMatchedV3ExternalStagingError, match="collision"):
        staging._validate_path_set(
            ["a", "a/b"],
            allowed_alias_groups=(),
        )


def test_canonical_ustar_header_fields_order_and_record_padding() -> None:
    long_path = ("prefix" * 18) + "/member.txt"
    header = staging._canonical_ustar_header(long_path, 7, 0o444)
    assert len(header) == 512
    assert header[257:263] == b"ustar\0"
    assert header[263:265] == b"00"
    assert header[100:108] == b"0000444\0"
    assert header[108:124] == b"0000000\0" * 2
    assert header[136:148] == b"00000000000\0"
    assert header[156:157] == b"0"
    assert header[157:257] == bytes(100)
    assert header[265:345] == bytes(80)
    assert header[500:512] == bytes(12)
    checksum_header = bytearray(header)
    checksum_header[148:156] = b"        "
    assert int(header[148:154], 8) == sum(checksum_header)

    members = [
        staging._ArchiveMember("a", 1, _sha256(b"a"), 0o444, b"a"),
        staging._ArchiveMember("b", 1, _sha256(b"b"), 0o555, b"b"),
    ]
    descriptor = staging._create_private_memfd("header-test")
    try:
        size, digest = staging._write_canonical_ustar(descriptor, members)
        raw = os.pread(descriptor, size, 0)
    finally:
        os.close(descriptor)
    assert size % staging._USTAR_RECORD_BYTES == 0
    assert raw[-1024:] == bytes(1024)
    assert _sha256(raw) == digest
    with pytest.raises(staging.ForagerMatchedV3ExternalStagingError, match="order"):
        descriptor = staging._create_private_memfd("order-test")
        try:
            staging._write_canonical_ustar(descriptor, list(reversed(members)))
        finally:
            os.close(descriptor)


def test_synthetic_archive_roundtrips_and_is_deterministic(
    synthetic_closure: _SyntheticClosure,
) -> None:
    assert _verify_archive(synthetic_closure.archive_raw, synthetic_closure) == (
        synthetic_closure.manifest
    )
    second_patcher = pytest.MonkeyPatch()
    second = _synthetic_closure(second_patcher)
    try:
        assert second.manifest_raw == synthetic_closure.manifest_raw
        assert second.archive_raw == synthetic_closure.archive_raw
        assert second.archive_sha256 == synthetic_closure.archive_sha256
    finally:
        second_patcher.undo()


def _mutated_archive_variants(closure: _SyntheticClosure) -> list[bytes]:
    variants: list[bytes] = []
    header = bytearray(closure.archive_raw)
    header[0] ^= 1
    variants.append(bytes(header))

    inventory, _aliases = staging._expected_complete_inventory(
        closure.manifest_raw, closure.manifest
    )
    first_size = inventory[0]["size_bytes"]
    padding_offset = staging._USTAR_BLOCK_BYTES + first_size
    padding_size = (-first_size) % staging._USTAR_BLOCK_BYTES
    assert padding_size > 0
    nonzero_padding = bytearray(closure.archive_raw)
    nonzero_padding[padding_offset] = 1
    variants.append(bytes(nonzero_padding))

    alternate_octal = bytearray(closure.archive_raw)
    alternate_octal[100:108] = b"0000444 "
    alternate_octal[148:156] = b"        "
    checksum = sum(alternate_octal[:512])
    alternate_octal[148:156] = f"{checksum:06o}".encode() + b"\0 "
    variants.append(bytes(alternate_octal))

    first_span = staging._USTAR_BLOCK_BYTES + first_size + padding_size
    second_size = inventory[1]["size_bytes"]
    second_padding = (-second_size) % staging._USTAR_BLOCK_BYTES
    second_span = staging._USTAR_BLOCK_BYTES + second_size + second_padding
    reordered = (
        closure.archive_raw[first_span : first_span + second_span]
        + closure.archive_raw[:first_span]
        + closure.archive_raw[first_span + second_span :]
    )
    variants.append(reordered)
    variants.append(closure.archive_raw + bytes(staging._USTAR_RECORD_BYTES))
    variants.append(closure.archive_raw[:-1])
    return variants


def test_raw_ustar_replay_rejects_header_order_padding_and_tail_variants(
    synthetic_closure: _SyntheticClosure,
) -> None:
    for raw in _mutated_archive_variants(synthetic_closure):
        with pytest.raises(staging.ForagerMatchedV3ExternalStagingError):
            _verify_archive(raw, synthetic_closure)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("candidate_id", "unknown_candidate"),
        ("path", "experiments/changed.json"),
        ("original_sha256", "0" * 64),
        ("derived_sha256", "1" * 64),
        ("transform_descriptor_sha256", "2" * 64),
    ],
)
def test_manifest_rejects_exact_overlay_semantic_mutations(
    synthetic_closure: _SyntheticClosure,
    field: str,
    replacement: str,
) -> None:
    changed = copy.deepcopy(synthetic_closure.manifest)
    changed["configuration_overlays"][0][field] = replacement
    raw = _reencode_manifest(changed)
    with pytest.raises(
        staging.ForagerMatchedV3ExternalStagingError,
        match="overlay execution semantics",
    ):
        staging.parse_external_staging_manifest(
            raw,
            expected_manifest_sha256=_sha256(raw),
        )


def test_manifest_rejects_recomputed_authority_and_inventory_mutations(
    synthetic_closure: _SyntheticClosure,
) -> None:
    authority = copy.deepcopy(synthetic_closure.manifest)
    authority["claims"]["execution_authority_granted"] = True
    authority_raw = _reencode_manifest(authority)
    with pytest.raises(staging.ForagerMatchedV3ExternalStagingError, match="authority"):
        staging.parse_external_staging_manifest(
            authority_raw,
            expected_manifest_sha256=_sha256(authority_raw),
        )

    inventory = copy.deepcopy(synthetic_closure.manifest)
    inventory["payload_inventory"][0]["sha256"] = "f" * 64
    inventory_raw = _reencode_manifest(inventory)
    with pytest.raises(staging.ForagerMatchedV3ExternalStagingError):
        staging.parse_external_staging_manifest(
            inventory_raw,
            expected_manifest_sha256=_sha256(inventory_raw),
        )


def _retained_bundle(
    closure: _SyntheticClosure,
) -> tuple[staging.RetainedExternalStagingBundle, int]:
    descriptor = _sealed_archive_descriptor(closure.archive_raw)
    metadata = os.fstat(descriptor)
    bundle = staging.RetainedExternalStagingBundle(
        staging._RETAINED_BUNDLE_CREATION_TOKEN,
        descriptor,
        metadata.st_dev,
        metadata.st_ino,
        len(closure.archive_raw),
        closure.archive_sha256,
        closure.manifest_raw,
        closure.manifest_sha256,
    )
    return bundle, descriptor


def test_retained_bundle_is_sealed_read_only_nonserializable_and_closes(
    synthetic_closure: _SyntheticClosure,
) -> None:
    bundle, descriptor = _retained_bundle(synthetic_closure)
    assert bundle.reverify() == synthetic_closure.manifest
    assert bundle.archive_size_bytes == len(synthetic_closure.archive_raw)
    assert bundle.archive_sha256 == synthetic_closure.archive_sha256
    assert bundle.manifest_bytes == synthetic_closure.manifest_raw
    assert bundle.manifest_sha256 == synthetic_closure.manifest_sha256
    assert bundle.subprocess_pass_fds == (descriptor,)
    assert bundle.proc_fd_path == f"/proc/self/fd/{descriptor}"
    metadata = os.fstat(descriptor)
    assert stat.S_ISREG(metadata.st_mode)
    assert metadata.st_nlink == 0
    assert stat.S_IMODE(metadata.st_mode) == 0o400
    assert fcntl.fcntl(descriptor, fcntl.F_GETFL) & os.O_ACCMODE == os.O_RDONLY
    assert fcntl.fcntl(descriptor, fcntl.F_GETFD) & fcntl.FD_CLOEXEC
    assert fcntl.fcntl(descriptor, fcntl.F_GET_SEALS) & staging._required_seals() == (
        staging._required_seals()
    )
    duplicate = os.dup(descriptor)
    try:
        assert not os.get_inheritable(duplicate)
        assert fcntl.fcntl(duplicate, fcntl.F_GETFL) & os.O_ACCMODE == os.O_RDONLY
        with pytest.raises(OSError):
            os.write(duplicate, b"x")
        with pytest.raises(OSError):
            os.ftruncate(duplicate, 0)
    finally:
        os.close(duplicate)
    with pytest.raises(TypeError):
        copy.copy(bundle)
    with pytest.raises(TypeError):
        copy.deepcopy(bundle)
    with pytest.raises(TypeError):
        pickle.dumps(bundle)
    bundle.close()
    assert bundle.closed is True
    with pytest.raises(staging.ForagerMatchedV3ExternalStagingError, match="closed"):
        _ = bundle.archive_sha256


def test_retained_bundle_rejects_pid_and_descriptor_reuse_without_closing_replacement(
    synthetic_closure: _SyntheticClosure,
) -> None:
    bundle, descriptor = _retained_bundle(synthetic_closure)
    owner_pid = bundle.owner_pid
    local_patch = pytest.MonkeyPatch()
    local_patch.setattr(os, "getpid", lambda: owner_pid + 1)
    try:
        assert bundle.closed is True
    finally:
        local_patch.undo()
        os.close(descriptor)

    bundle, descriptor = _retained_bundle(synthetic_closure)
    os.close(descriptor)
    replacements: list[int] = []
    try:
        while not replacements or replacements[-1] < descriptor:
            replacements.append(os.open("/dev/null", os.O_RDONLY))
        replacement = replacements[-1]
        assert replacement == descriptor
        with pytest.raises(staging.ForagerMatchedV3ExternalStagingError, match="descriptor"):
            _ = bundle.archive_sha256
        os.fstat(replacement)
    finally:
        for replacement in replacements:
            os.close(replacement)


def test_retained_bundle_constructor_rejects_external_creation() -> None:
    with pytest.raises(TypeError, match="staging context"):
        staging.RetainedExternalStagingBundle(object(), 3, 1, 1, 1, "0" * 64, b"{}\n", "0" * 64)


class _DuckCapability:
    subprocess_pass_fds = (99,)

    def reverify(self) -> dict[str, Any]:
        return {}


def test_public_stage_rejects_paths_ducks_subclasses_and_closed_exact_capability(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(staging, "_validated_execution_records", lambda: (b"x", ()))
    for value in (tmp_path, str(tmp_path), 7, _DuckCapability()):
        with pytest.raises(staging.ForagerMatchedV3ExternalStagingError, match="exact live"):
            with staging.stage_matched_v3_external_workload(value):
                raise AssertionError("unreachable")

    class _Subclass(materialization.RetainedExternalMaterializationTree):
        pass

    subclass = object.__new__(_Subclass)
    with pytest.raises(staging.ForagerMatchedV3ExternalStagingError, match="exact live"):
        with staging.stage_matched_v3_external_workload(subclass):
            raise AssertionError("unreachable")

    directory_descriptor = os.open(
        tmp_path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    metadata = os.fstat(directory_descriptor)
    closed = materialization.RetainedExternalMaterializationTree(
        materialization._RETAINED_CAPABILITY_CREATION_TOKEN,
        directory_descriptor,
        metadata.st_dev,
        metadata.st_ino,
        b"{}",
        "0" * 64,
        require_matched_v3_identity=True,
    )
    closed.close()
    with pytest.raises(staging.ForagerMatchedV3ExternalStagingError, match="not active"):
        with staging.stage_matched_v3_external_workload(closed):
            raise AssertionError("unreachable")


def test_import_time_source_substitution_is_rejected_without_fd_leak(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = _fd_count()
    monkeypatch.setattr(staging, "_IMPORTED_IMPLEMENTATION_SOURCE_SHA256", "0" * 64)
    with pytest.raises(
        staging.ForagerMatchedV3ExternalStagingError,
        match="changed since module import",
    ):
        staging._open_bound_worker_source()
    assert _fd_count() == before


def test_bound_worker_source_initial_fstat_failure_closes_named_descriptor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = _fd_count()

    def fail_fstat(_descriptor: int) -> os.stat_result:
        raise OSError("injected worker-source fstat failure")

    monkeypatch.setattr(os, "fstat", fail_fstat)
    with pytest.raises(
        staging.ForagerMatchedV3ExternalStagingError,
        match="cannot be retained",
    ):
        staging._open_bound_worker_source()
    assert _fd_count() == baseline


def test_bound_worker_source_named_close_failure_closes_sealed_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_metadata = os.stat(staging.__file__, follow_symlinks=False)
    source_identity = (source_metadata.st_dev, source_metadata.st_ino)
    baseline = _fd_count()
    real_close = os.close
    injected = False

    def fail_after_named_close(descriptor: int) -> None:
        nonlocal injected
        metadata = os.fstat(descriptor)
        real_close(descriptor)
        if not injected and (metadata.st_dev, metadata.st_ino) == source_identity:
            injected = True
            raise OSError("injected named worker-source close failure")

    monkeypatch.setattr(os, "close", fail_after_named_close)
    with pytest.raises(OSError, match="injected named worker-source close failure"):
        staging._open_bound_worker_source()
    assert injected
    assert _fd_count() == baseline


def test_bound_worker_source_is_an_immutable_sealed_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_raw = Path(staging.__file__).read_bytes()
    source_path = tmp_path / "forager_matched_v3_external_staging.py"
    source_path.write_bytes(source_raw)
    source_path.chmod(0o644)
    source_metadata = os.stat(source_path, follow_symlinks=False)
    monkeypatch.setattr(staging, "_implementation_source_path", lambda: str(source_path))
    monkeypatch.setattr(
        staging,
        "_IMPORTED_IMPLEMENTATION_SOURCE_SIGNATURE",
        staging._stat_signature(source_metadata),
    )
    monkeypatch.setattr(
        staging,
        "_IMPORTED_IMPLEMENTATION_SOURCE_SHA256",
        _sha256(source_raw),
    )
    baseline = _fd_count()
    bound = staging._open_bound_worker_source()
    try:
        writer = os.open(source_path, os.O_WRONLY)
        try:
            os.pwrite(writer, bytes([source_raw[0] ^ 1]), 0)
            os.fsync(writer)
        finally:
            os.close(writer)
        metadata = os.fstat(bound.descriptor)
        assert metadata.st_nlink == 0
        assert stat.S_IMODE(metadata.st_mode) == 0o400
        assert fcntl.fcntl(bound.descriptor, fcntl.F_GETFL) & os.O_ACCMODE == os.O_RDONLY
        assert fcntl.fcntl(bound.descriptor, fcntl.F_GET_SEALS) & staging._required_seals() == (
            staging._required_seals()
        )
        assert staging._hash_fd(
            bound.descriptor,
            metadata.st_size,
            "sealed worker source",
        ) == _sha256(source_raw)
    finally:
        os.close(bound.descriptor)
    assert _fd_count() == baseline


def test_proc_fd_import_rejects_unsealed_source_and_duplicates_exact_snapshot(
    tmp_path: Path,
) -> None:
    ordinary_path = tmp_path / "ordinary.py"
    ordinary_path.write_bytes(b"pass\n")
    ordinary = os.open(ordinary_path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    baseline = _fd_count()
    try:
        with pytest.raises(staging.ForagerMatchedV3ExternalStagingError):
            staging._open_imported_implementation_source(f"/proc/self/fd/{ordinary}")
        os.fstat(ordinary)
        assert _fd_count() == baseline
    finally:
        os.close(ordinary)

    bound = staging._open_bound_worker_source()
    duplicate = -1
    try:
        duplicate, duplicate_metadata = staging._open_imported_implementation_source(
            f"/proc/self/fd/{bound.descriptor}"
        )
        assert staging._stat_signature(duplicate_metadata) == bound.signature
        assert not os.get_inheritable(duplicate)
        assert (
            staging._hash_fd(
                duplicate,
                duplicate_metadata.st_size,
                "duplicated sealed worker source",
            )
            == bound.sha256
        )
    finally:
        if duplicate >= 0:
            os.close(duplicate)
        os.close(bound.descriptor)


def test_private_memfd_fchmod_failure_closes_created_descriptor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = _fd_count()

    def fail_fchmod(_descriptor: int, _mode: int) -> None:
        raise OSError("injected fchmod failure")

    monkeypatch.setattr(os, "fchmod", fail_fchmod)
    with pytest.raises(staging.ForagerMatchedV3ExternalStagingError):
        staging._create_private_memfd("failure")
    assert _fd_count() == before


def test_sealed_bytes_writable_close_failure_closes_readonly_descriptor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = _fd_count()
    real_close = os.close
    injected = False

    def fail_after_first_close(descriptor: int) -> None:
        nonlocal injected
        real_close(descriptor)
        if not injected:
            injected = True
            raise OSError("injected writable memfd close failure")

    monkeypatch.setattr(os, "close", fail_after_first_close)
    with pytest.raises(OSError, match="injected writable memfd close failure"):
        staging._sealed_bytes_fd(b"payload", "close-failure", 100)
    assert injected
    assert _fd_count() == baseline


def test_seal_reopen_failure_closes_readonly_duplicate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writable = staging._create_private_memfd("seal-failure")
    staging._write_all(writable, b"payload")
    baseline = _fd_count()
    real_fstat = os.fstat

    def fail_duplicate_fstat(descriptor: int) -> os.stat_result:
        if descriptor != writable:
            raise OSError("injected readonly fstat failure")
        return real_fstat(descriptor)

    monkeypatch.setattr(os, "fstat", fail_duplicate_fstat)
    try:
        with pytest.raises(staging.ForagerMatchedV3ExternalStagingError):
            staging._seal_and_reopen_readonly(writable, expected_size=7)
        assert _fd_count() == baseline
    finally:
        os.close(writable)


def test_duplicate_directory_failure_closes_duplicate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = os.open(
        tmp_path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    baseline = _fd_count()
    real_fstat = os.fstat
    calls = 0

    def fail_third_fstat(descriptor: int) -> os.stat_result:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise OSError("injected duplicate fstat failure")
        return real_fstat(descriptor)

    monkeypatch.setattr(os, "fstat", fail_third_fstat)
    try:
        with pytest.raises(staging.ForagerMatchedV3ExternalStagingError):
            staging._duplicate_directory_descriptor(source, "test")
        assert _fd_count() == baseline
    finally:
        os.close(source)


def test_open_relative_parent_failure_closes_child_and_duplicate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    child = tmp_path / "child"
    child.mkdir()
    child.chmod(0o755)
    source = os.open(
        tmp_path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    baseline = _fd_count()
    real_fstat = os.fstat
    calls = 0

    def fail_child_fstat(descriptor: int) -> os.stat_result:
        nonlocal calls
        calls += 1
        if calls == 4:
            raise OSError("injected child fstat failure")
        return real_fstat(descriptor)

    monkeypatch.setattr(os, "fstat", fail_child_fstat)
    try:
        with pytest.raises(staging.ForagerMatchedV3ExternalStagingError):
            staging._open_relative_parent(source, "child/file")
        assert _fd_count() == baseline
    finally:
        os.close(source)


def test_worker_source_bind_failure_cleans_receipt_pipe_and_selector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptors = [os.open("/dev/null", os.O_RDONLY) for _ in range(3)]
    baseline = _fd_count()

    def fail_source() -> staging._BoundWorkerSource:
        raise staging.ForagerMatchedV3ExternalStagingError("injected source bind failure")

    monkeypatch.setattr(staging, "_open_bound_worker_source", fail_source)
    try:
        with pytest.raises(staging.ForagerMatchedV3ExternalStagingError):
            staging._run_isolated_worker(
                source_descriptor=descriptors[0],
                request_descriptor=descriptors[1],
                output_descriptor=descriptors[2],
                base_manifest_sha256="0" * 64,
                stage_manifest_sha256="1" * 64,
                expected_member_count=1,
            )
        assert _fd_count() == baseline
    finally:
        for descriptor in descriptors:
            os.close(descriptor)


def test_worker_argv_request_and_sealed_request_bounds(tmp_path: Path) -> None:
    assert staging._worker_entrypoint([]) == 64
    assert staging._worker_entrypoint(["--isolated-stage-worker", "0", "2", "3"]) == 64
    with pytest.raises(staging.ForagerMatchedV3ExternalStagingError):
        staging._parse_worker_request(b"{}\n")
    with pytest.raises(staging.ForagerMatchedV3ExternalStagingError):
        staging._strict_base64("***", "invalid", 10)

    unsealed = os.open(tmp_path / "request", os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(unsealed, b"{}\n")
        with pytest.raises(staging.ForagerMatchedV3ExternalStagingError, match="metadata"):
            staging._read_bounded_fd(unsealed, 100, "request")
    finally:
        os.close(unsealed)


def test_worker_rejects_invalid_sealed_request_without_reading_source(tmp_path: Path) -> None:
    source = os.open(
        tmp_path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    request = staging._sealed_bytes_fd(b"{}\n", "invalid-request", 100)
    receipt_read, receipt_write = os.pipe2(getattr(os, "O_CLOEXEC", 0))
    try:
        with pytest.raises(staging.ForagerMatchedV3ExternalStagingError, match="request fields"):
            staging._worker_stage(source, request, receipt_write)
        os.close(receipt_write)
        receipt_write = -1
        assert os.read(receipt_read, 1) == b""
    finally:
        if receipt_write >= 0:
            os.close(receipt_write)
        os.close(receipt_read)
        os.close(request)
        os.close(source)


def test_isolated_worker_imports_exact_inherited_source_fd_before_rejecting_request(
    tmp_path: Path,
) -> None:
    source = os.open(
        tmp_path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    request = staging._sealed_bytes_fd(b"{}\n", "invalid-isolated-request", 100)
    output = staging._create_private_memfd("invalid-isolated-output")
    baseline = _fd_count()
    try:
        with pytest.raises(
            staging.ForagerMatchedV3ExternalStagingError,
            match="isolated staging request fields differ",
        ):
            staging._run_isolated_worker(
                source_descriptor=source,
                request_descriptor=request,
                output_descriptor=output,
                base_manifest_sha256="1" * 64,
                stage_manifest_sha256="2" * 64,
                expected_member_count=1,
            )
        assert os.fstat(output).st_size == 0
        assert _fd_count() == baseline
    finally:
        os.close(output)
        os.close(request)
        os.close(source)


def test_process_group_cleanup_targets_descendants_after_direct_child_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[tuple[int, int]] = []

    class _ExitedProcess:
        pid = 424242

        @staticmethod
        def poll() -> int:
            return 0

    def record_killpg(pid: int, requested_signal: int) -> None:
        observed.append((pid, requested_signal))

    monkeypatch.setattr(os, "killpg", record_killpg)
    staging._terminate_process_group(_ExitedProcess())
    assert observed == [
        (_ExitedProcess.pid, signal.SIGTERM),
        (_ExitedProcess.pid, signal.SIGKILL),
    ]
