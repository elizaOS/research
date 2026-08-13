"""Tests for the source-only matched-v3 Quicknet archive inventory contract."""

from __future__ import annotations

import ast
import copy
import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path
from typing import Any, cast

import pytest

from alberta_framework.benchmarks import (
    forager_matched_v3_quicknet_source_materialization as materialization,
)

pytestmark = pytest.mark.unit

_EXPECTED_DESCRIPTOR_SHA256 = "61345825673afb16bc1942c4b8c84e763fb14530a68225caffa94d98e733a03d"


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        + b"\n"
    )


def _tar_bytes(
    prefix: str,
    entries: list[tuple[str, bytes, bytes, int]],
    *,
    include_root: bool = True,
    tar_format: int = tarfile.USTAR_FORMAT,
) -> bytes:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w", format=tar_format) as archive:
        if include_root:
            root = tarfile.TarInfo(prefix + "/")
            root.type = tarfile.DIRTYPE
            root.mode = 0o755
            root.uid = 0
            root.gid = 0
            root.mtime = 0
            archive.addfile(root)
        for path, payload, entry_type, mode in entries:
            info = tarfile.TarInfo(path)
            info.type = entry_type
            info.mode = mode
            info.uid = 0
            info.gid = 0
            info.mtime = 0
            if entry_type in {tarfile.REGTYPE, tarfile.AREGTYPE}:
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))
            else:
                info.size = 0
                if entry_type in {tarfile.SYMTYPE, tarfile.LNKTYPE}:
                    info.linkname = f"{prefix}/target"
                archive.addfile(info)
    return stream.getvalue()


def _gzip_tar(
    prefix: str,
    entries: list[tuple[str, bytes, bytes, int]],
    *,
    include_root: bool = True,
    tar_format: int = tarfile.USTAR_FORMAT,
) -> bytes:
    return gzip.compress(
        _tar_bytes(
            prefix,
            entries,
            include_root=include_root,
            tar_format=tar_format,
        ),
        mtime=0,
    )


def _spec(archive_id: str, prefix: str, raw: bytes) -> materialization._ArchiveSpec:
    return materialization._ArchiveSpec(
        archive_id=archive_id,
        sha256=hashlib.sha256(raw).hexdigest(),
        size_bytes=len(raw),
        top_level_prefix=prefix,
    )


def _inventory(
    archive_id: str,
    prefix: str,
    raw: bytes,
) -> tuple[materialization._ArchiveSpec, dict[str, Any]]:
    spec = _spec(archive_id, prefix, raw)
    return spec, materialization._inventory_gzip_tar_archive(raw, spec=spec)


def _pair_artifacts() -> tuple[
    tuple[materialization._ArchiveSpec, materialization._ArchiveSpec],
    tuple[materialization._RelevantFilePin, ...],
    bytes,
    bytes,
]:
    commit_prefix = "commit-root"
    crate_prefix = "crate-root"
    commit_raw = _gzip_tar(
        commit_prefix,
        [
            (f"{commit_prefix}/Cargo.lock", b"lock", tarfile.REGTYPE, 0o644),
            (f"{commit_prefix}/only-commit", b"commit", tarfile.REGTYPE, 0o644),
            (f"{commit_prefix}/src/", b"", tarfile.DIRTYPE, 0o755),
            (f"{commit_prefix}/src/lib.rs", b"same", tarfile.REGTYPE, 0o644),
        ],
    )
    crate_raw = _gzip_tar(
        crate_prefix,
        [
            (
                f"{crate_prefix}/.cargo_vcs_info.json",
                b"vcs",
                tarfile.REGTYPE,
                0o644,
            ),
            (f"{crate_prefix}/only-crate", b"crate", tarfile.REGTYPE, 0o644),
            (f"{crate_prefix}/src/", b"", tarfile.DIRTYPE, 0o755),
            (f"{crate_prefix}/src/lib.rs", b"same", tarfile.REGTYPE, 0o644),
        ],
        include_root=False,
    )
    commit_spec, commit_inventory = _inventory("commit", commit_prefix, commit_raw)
    crate_spec, crate_inventory = _inventory("crate", crate_prefix, crate_raw)
    specs = (commit_spec, crate_spec)
    pins = (
        materialization._RelevantFilePin(
            archive_id="commit",
            path="Cargo.lock",
            sha256=hashlib.sha256(b"lock").hexdigest(),
        ),
        materialization._RelevantFilePin(
            archive_id="crate",
            path=".cargo_vcs_info.json",
            sha256=hashlib.sha256(b"vcs").hexdigest(),
        ),
    )
    manifest = materialization._build_manifest(
        [commit_inventory, crate_inventory],
        pins=pins,
        plan_bytes=materialization._PLAN_BYTES,
    )
    receipt = materialization._build_receipt(
        manifest,
        producer_source_bytes=b"frozen materializer source",
        specs=specs,
        pins=pins,
        plan_bytes=materialization._PLAN_BYTES,
    )
    return specs, pins, manifest, receipt


def _replace_body_digest(value: dict[str, Any], field: str) -> bytes:
    body = {key: item for key, item in value.items() if key != field}
    value[field] = hashlib.sha256(_canonical(body)).hexdigest()
    return _canonical(value)


def _mutate_first_tar_header(raw: bytes, mutator: Any) -> bytes:
    unpacked = bytearray(gzip.decompress(raw))
    header = bytearray(unpacked[:512])
    mutator(header)
    header[148:156] = b" " * 8
    checksum = sum(header)
    header[148:156] = f"{checksum:06o}\0 ".encode("ascii")
    unpacked[:512] = header
    return gzip.compress(bytes(unpacked), mtime=0)


def test_descriptor_is_canonical_detached_and_frozen() -> None:
    raw = materialization.canonical_matched_v3_quicknet_source_materialization_descriptor_bytes()
    assert raw == _canonical(json.loads(raw))
    assert materialization.MATCHED_V3_QUICKNET_SOURCE_MATERIALIZATION_DESCRIPTOR_SHA256 == (
        _EXPECTED_DESCRIPTOR_SHA256
    )
    assert (
        materialization.parse_matched_v3_quicknet_source_materialization_descriptor(raw)
        == materialization.matched_v3_quicknet_source_materialization_descriptor()
    )

    first = materialization.matched_v3_quicknet_source_materialization_descriptor()
    first["authority"]["execution_authority_granted"] = True
    first["state"]["qualification_ready"] = True
    second = materialization.matched_v3_quicknet_source_materialization_descriptor()
    assert set(second["authority"].values()) == {False}
    assert second["state"]["qualification_ready"] is False


def test_frozen_plan_is_available_without_archives_and_binds_existing_source_registry() -> None:
    raw = materialization.canonical_matched_v3_quicknet_source_materialization_plan_bytes()
    plan = materialization.parse_matched_v3_quicknet_source_materialization_plan(
        raw,
        expected_plan_sha256=hashlib.sha256(raw).hexdigest(),
    )
    assert plan == materialization.matched_v3_quicknet_source_materialization_plan()
    assert plan["source_registry"] == {
        "descriptor_schema_version": (
            "alberta.forager_matched_v3.quicknet_verifier_source_descriptor.v2"
        ),
        "descriptor_sha256": ("4d2241ebf8e4e431e33addf317c116531a6605a391906f6bddf18491e0764fdd"),
        "source_module_sha256": (
            "3e13009c1843c3341e5a0eb8b2f84ea903b8e5315fbdef347549757710fd3623"
        ),
        "source_module_bytes_read_here": False,
        "descriptor_imported_here": False,
    }
    assert plan["state"] == {
        "archive_bytes_supplied": False,
        "archives_inventory_completed": False,
        "cross_archive_comparison_completed": False,
        "filesystem_materialization_performed": False,
        "primary_build_source_selected": False,
        "dependency_vendor_closure_available": False,
        "qualification_ready": False,
    }
    assert set(plan["authority"].values()) == {False}


def test_plan_exactly_pins_both_archives_and_safe_inventory_policy() -> None:
    plan = materialization.matched_v3_quicknet_source_materialization_plan()
    assert plan["archive_inputs"] == [
        {
            "archive_id": "upstream_commit_archive",
            "compression": "gzip_single_member",
            "container": "ustar_regular_and_directory_entries_only",
            "sha256": ("633408b2d2adca4d9986e765ee2ece148b26de50f7440db5c5f3f7054edfe760"),
            "size_bytes": 18_727,
            "top_level_prefix": ("drand-verify-1db2248afac44fc2e5c9c78f896b4412d8679914"),
        },
        {
            "archive_id": "crates_io_package_archive",
            "compression": "gzip_single_member",
            "container": "ustar_regular_and_directory_entries_only",
            "sha256": ("4c1d531704590bbfce3433cd735378d135cabc9e318d8aa52c5dccf7b80178ee"),
            "size_bytes": 18_961,
            "top_level_prefix": "drand-verify-0.6.2",
        },
    ]
    policy = plan["inventory_policy"]
    assert policy["gzip_member_count_required"] == 1
    assert policy["gzip_trailing_bytes_allowed"] is False
    assert policy["accepted_tar_entry_types"] == ["directory", "regular_file"]
    for denial in (
        "pax_entries_allowed",
        "gnu_long_name_entries_allowed",
        "sparse_entries_allowed",
        "link_entries_allowed",
        "special_entries_allowed",
        "path_aliases_allowed",
        "archive_extraction_performed",
        "filesystem_write_performed",
    ):
        assert policy[denial] is False


def test_public_inventory_rejects_unpinned_bytes_before_any_manifest() -> None:
    with pytest.raises(
        materialization.ForagerMatchedV3QuicknetSourceMaterializationError,
        match="exact archive identity",
    ):
        materialization.inventory_matched_v3_quicknet_source_archives(
            commit_archive_bytes=b"not the commit archive",
            crate_archive_bytes=b"not the crate archive",
            producer_source_bytes=b"source",
        )


def test_bounded_inventory_records_exact_path_type_mode_size_digest_and_tree() -> None:
    prefix = "source-root"
    raw = _gzip_tar(
        prefix,
        [
            (f"{prefix}/bin/tool", b"#!tool\n", tarfile.REGTYPE, 0o755),
            (f"{prefix}/empty/", b"", tarfile.DIRTYPE, 0o755),
            (f"{prefix}/src/lib.rs", b"pub fn x() {{}}\n", tarfile.REGTYPE, 0o644),
        ],
    )
    spec, inventory = _inventory("fixture", prefix, raw)
    parsed = materialization._validate_archive_inventory(
        inventory,
        spec=spec,
        label="fixture",
    )
    assert parsed["raw_archive_sha256"] == hashlib.sha256(raw).hexdigest()
    assert parsed["root_directory_entry_present"] is True
    assert parsed["raw_member_count"] == 4
    assert parsed["entry_count"] == 3
    assert parsed["regular_file_count"] == 2
    assert parsed["directory_count"] == 1
    assert [entry["path"] for entry in parsed["entries"]] == [
        "bin/tool",
        "empty",
        "src/lib.rs",
    ]
    by_path = {entry["path"]: entry for entry in parsed["entries"]}
    assert by_path["bin/tool"]["mode"] == "0755"
    assert by_path["bin/tool"]["size_bytes"] == 7
    assert by_path["bin/tool"]["sha256"] == hashlib.sha256(b"#!tool\n").hexdigest()
    assert by_path["empty"]["entry_type"] == "directory"
    assert by_path["empty"]["sha256"] == hashlib.sha256(b"").hexdigest()
    assert parsed["tree_sha256"] == materialization._entry_tree_sha256(parsed["entries"])


def test_root_directory_entry_may_be_absent_without_weakening_prefix_check() -> None:
    prefix = "source-root"
    raw = _gzip_tar(
        prefix,
        [(f"{prefix}/src/lib.rs", b"source", tarfile.REGTYPE, 0o644)],
        include_root=False,
    )
    _, inventory = _inventory("fixture", prefix, raw)
    assert inventory["root_directory_entry_present"] is False
    assert inventory["raw_member_count"] == inventory["entry_count"] == 1
    assert inventory["entries"][0]["path"] == "src/lib.rs"


def test_manifest_receipt_and_cross_archive_comparison_replay_exactly() -> None:
    specs, pins, manifest_raw, receipt_raw = _pair_artifacts()
    manifest_sha = hashlib.sha256(manifest_raw).hexdigest()
    manifest = materialization._parse_manifest(
        manifest_raw,
        expected_manifest_sha256=manifest_sha,
        specs=specs,
        pins=pins,
        plan_bytes=materialization._PLAN_BYTES,
    )
    comparison = manifest["cross_archive_comparison"]
    assert comparison["commit_only_paths"] == ["Cargo.lock", "only-commit"]
    assert comparison["crate_only_paths"] == [".cargo_vcs_info.json", "only-crate"]
    assert [record["path"] for record in comparison["common_entries"]] == [
        "src",
        "src/lib.rs",
    ]
    assert comparison["common_entry_count"] == 2
    assert comparison["content_identical_common_entry_count"] == 2
    assert comparison["archives_declared_equivalent"] is False
    assert comparison["build_source_selected"] is False
    assert all(check["matched"] is True for check in manifest["relevant_file_pin_checks"])

    receipt_sha = hashlib.sha256(receipt_raw).hexdigest()
    receipt = materialization._parse_receipt(
        receipt_raw,
        expected_receipt_sha256=receipt_sha,
        manifest_bytes=manifest_raw,
        expected_manifest_sha256=manifest_sha,
        specs=specs,
        pins=pins,
        plan_bytes=materialization._PLAN_BYTES,
    )
    assert receipt["manifest"]["sha256"] == manifest_sha
    assert receipt["state"]["source_archive_inventory_receipt_issued"] is True
    assert receipt["state"]["filesystem_materialization_receipt_issued"] is False
    assert receipt["state"]["dependency_vendor_closure_receipt_issued"] is False
    assert receipt["state"]["qualification_receipt_issued"] is False
    assert set(receipt["authority"].values()) == {False}


@pytest.mark.parametrize("suffix", [b"trailing", gzip.compress(b"second", mtime=0)])
def test_gzip_rejects_trailing_bytes_and_concatenated_members(suffix: bytes) -> None:
    raw = gzip.compress(b"payload", mtime=0) + suffix
    with pytest.raises(
        materialization.ForagerMatchedV3QuicknetSourceMaterializationError,
        match="concatenated member or trailing bytes",
    ):
        materialization._decompress_single_gzip_member(raw, label="fixture")


def test_gzip_rejects_crc_corruption_truncation_and_decompression_bomb(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    valid = gzip.compress(b"payload", mtime=0)
    corrupt = valid[:-1] + bytes([valid[-1] ^ 1])
    for raw in (corrupt, valid[:-3]):
        with pytest.raises(materialization.ForagerMatchedV3QuicknetSourceMaterializationError):
            materialization._decompress_single_gzip_member(raw, label="fixture")

    monkeypatch.setattr(materialization, "_MAX_UNCOMPRESSED_TAR_BYTES", 1_024)
    bomb = gzip.compress(b"x" * 1_025, mtime=0)
    with pytest.raises(
        materialization.ForagerMatchedV3QuicknetSourceMaterializationError,
        match="exceeds its byte bound",
    ):
        materialization._decompress_single_gzip_member(bomb, label="fixture")


@pytest.mark.parametrize(
    "relative",
    [
        "../escape",
        "./same",
        "dir//file",
        "dir/../file",
        "dir/./file",
        "dir\\file",
        "/absolute",
        "name.",
        "CON",
    ],
)
def test_tar_inventory_rejects_traversal_and_portable_path_aliases(relative: str) -> None:
    prefix = "source-root"
    raw = _gzip_tar(
        prefix,
        [(f"{prefix}/{relative}", b"x", tarfile.REGTYPE, 0o644)],
    )
    spec = _spec("fixture", prefix, raw)
    with pytest.raises(materialization.ForagerMatchedV3QuicknetSourceMaterializationError):
        materialization._inventory_gzip_tar_archive(raw, spec=spec)


def test_tar_inventory_rejects_duplicate_casefold_and_file_ancestor_aliases() -> None:
    prefix = "source-root"
    cases = [
        [
            (f"{prefix}/same", b"a", tarfile.REGTYPE, 0o644),
            (f"{prefix}/same", b"b", tarfile.REGTYPE, 0o644),
        ],
        [
            (f"{prefix}/Name", b"a", tarfile.REGTYPE, 0o644),
            (f"{prefix}/name", b"b", tarfile.REGTYPE, 0o644),
        ],
        [
            (f"{prefix}/parent", b"a", tarfile.REGTYPE, 0o644),
            (f"{prefix}/parent/child", b"b", tarfile.REGTYPE, 0o644),
        ],
    ]
    for entries in cases:
        raw = _gzip_tar(prefix, entries)
        spec = _spec("fixture", prefix, raw)
        with pytest.raises(materialization.ForagerMatchedV3QuicknetSourceMaterializationError):
            materialization._inventory_gzip_tar_archive(raw, spec=spec)


@pytest.mark.parametrize(
    "entry_type",
    [
        tarfile.SYMTYPE,
        tarfile.LNKTYPE,
        tarfile.CHRTYPE,
        tarfile.BLKTYPE,
        tarfile.FIFOTYPE,
        tarfile.CONTTYPE,
        tarfile.XHDTYPE,
        tarfile.XGLTYPE,
        tarfile.GNUTYPE_LONGNAME,
        tarfile.GNUTYPE_LONGLINK,
        tarfile.GNUTYPE_SPARSE,
    ],
)
def test_tar_inventory_rejects_links_special_sparse_pax_and_gnu_extensions(
    entry_type: bytes,
) -> None:
    prefix = "source-root"
    raw = _gzip_tar(
        prefix,
        [(f"{prefix}/forbidden", b"", entry_type, 0o644)],
    )
    spec = _spec("fixture", prefix, raw)
    with pytest.raises(
        materialization.ForagerMatchedV3QuicknetSourceMaterializationError,
        match="forbidden link, special, sparse, PAX, or GNU extension",
    ):
        materialization._inventory_gzip_tar_archive(raw, spec=spec)


def test_tar_inventory_rejects_pax_format_even_when_logical_member_is_regular() -> None:
    prefix = "source-root"
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w", format=tarfile.PAX_FORMAT) as archive:
        info = tarfile.TarInfo(f"{prefix}/file")
        info.size = 1
        info.pax_headers = {"comment": "force-pax-header"}
        archive.addfile(info, io.BytesIO(b"x"))
    raw = gzip.compress(stream.getvalue(), mtime=0)
    spec = _spec("fixture", prefix, raw)
    with pytest.raises(materialization.ForagerMatchedV3QuicknetSourceMaterializationError):
        materialization._inventory_gzip_tar_archive(raw, spec=spec)


def test_tar_inventory_rejects_special_mode_bad_checksum_nonzero_padding_and_tail() -> None:
    prefix = "source-root"
    special_mode = _gzip_tar(
        prefix,
        [(f"{prefix}/file", b"x", tarfile.REGTYPE, 0o4755)],
        include_root=False,
    )
    cases = [special_mode]

    valid = _gzip_tar(
        prefix,
        [(f"{prefix}/file", b"x", tarfile.REGTYPE, 0o644)],
        include_root=False,
    )
    bad_checksum_tar = bytearray(gzip.decompress(valid))
    bad_checksum_tar[0] ^= 1
    cases.append(gzip.compress(bytes(bad_checksum_tar), mtime=0))

    nonzero_padding_tar = bytearray(gzip.decompress(valid))
    nonzero_padding_tar[513] = 1
    cases.append(gzip.compress(bytes(nonzero_padding_tar), mtime=0))

    nonzero_tail_tar = bytearray(gzip.decompress(valid))
    nonzero_tail_tar[-1] = 1
    cases.append(gzip.compress(bytes(nonzero_tail_tar), mtime=0))

    for raw in cases:
        spec = _spec("fixture", prefix, raw)
        with pytest.raises(materialization.ForagerMatchedV3QuicknetSourceMaterializationError):
            materialization._inventory_gzip_tar_archive(raw, spec=spec)


def test_tar_inventory_rejects_base256_numbers_and_wrong_top_level_prefix() -> None:
    prefix = "source-root"
    valid = _gzip_tar(
        prefix,
        [(f"{prefix}/file", b"x", tarfile.REGTYPE, 0o644)],
        include_root=False,
    )

    base256 = _mutate_first_tar_header(valid, lambda header: header.__setitem__(100, 0x80))
    with pytest.raises(
        materialization.ForagerMatchedV3QuicknetSourceMaterializationError,
        match="base-256",
    ):
        materialization._inventory_gzip_tar_archive(
            base256,
            spec=_spec("fixture", prefix, base256),
        )

    with pytest.raises(
        materialization.ForagerMatchedV3QuicknetSourceMaterializationError,
        match="top-level prefix",
    ):
        materialization._inventory_gzip_tar_archive(
            valid,
            spec=_spec("fixture", "different-root", valid),
        )


def test_relevant_file_pin_mismatch_fails_before_manifest_creation() -> None:
    prefix = "source-root"
    raw = _gzip_tar(
        prefix,
        [(f"{prefix}/Cargo.lock", b"lock", tarfile.REGTYPE, 0o644)],
    )
    _, inventory = _inventory("commit", prefix, raw)
    bad_pin = materialization._RelevantFilePin(
        archive_id="commit",
        path="Cargo.lock",
        sha256="f" * 64,
    )
    with pytest.raises(
        materialization.ForagerMatchedV3QuicknetSourceMaterializationError,
        match="pinned relevant file",
    ):
        materialization._pin_checks([inventory], [bad_pin])


def test_manifest_rejects_archive_crosswire_and_stale_cross_comparison() -> None:
    specs, pins, manifest_raw, _ = _pair_artifacts()
    value = cast(dict[str, Any], json.loads(manifest_raw))
    value["archive_inventories"] = list(reversed(value["archive_inventories"]))
    crosswired = _replace_body_digest(value, "manifest_body_sha256")
    with pytest.raises(materialization.ForagerMatchedV3QuicknetSourceMaterializationError):
        materialization._parse_manifest(
            crosswired,
            expected_manifest_sha256=hashlib.sha256(crosswired).hexdigest(),
            specs=specs,
            pins=pins,
            plan_bytes=materialization._PLAN_BYTES,
        )

    value = cast(dict[str, Any], json.loads(manifest_raw))
    commit_inventory = value["archive_inventories"][0]
    target = next(entry for entry in commit_inventory["entries"] if entry["path"] == "src/lib.rs")
    target["mode"] = "0755"
    commit_inventory["tree_sha256"] = materialization._entry_tree_sha256(
        commit_inventory["entries"]
    )
    stale_comparison = _replace_body_digest(value, "manifest_body_sha256")
    with pytest.raises(
        materialization.ForagerMatchedV3QuicknetSourceMaterializationError,
        match="cross-archive comparison",
    ):
        materialization._parse_manifest(
            stale_comparison,
            expected_manifest_sha256=hashlib.sha256(stale_comparison).hexdigest(),
            specs=specs,
            pins=pins,
            plan_bytes=materialization._PLAN_BYTES,
        )


def test_receipt_rejects_manifest_and_tree_binding_substitution() -> None:
    specs, pins, manifest_raw, receipt_raw = _pair_artifacts()
    receipt = cast(dict[str, Any], json.loads(receipt_raw))
    receipt["manifest"]["sha256"] = "f" * 64
    mutated = _replace_body_digest(receipt, "receipt_body_sha256")
    with pytest.raises(materialization.ForagerMatchedV3QuicknetSourceMaterializationError):
        materialization._parse_receipt(
            mutated,
            expected_receipt_sha256=hashlib.sha256(mutated).hexdigest(),
            manifest_bytes=manifest_raw,
            expected_manifest_sha256=hashlib.sha256(manifest_raw).hexdigest(),
            specs=specs,
            pins=pins,
            plan_bytes=materialization._PLAN_BYTES,
        )

    receipt = cast(dict[str, Any], json.loads(receipt_raw))
    receipt["archive_tree_bindings"][0]["tree_sha256"] = "f" * 64
    mutated = _replace_body_digest(receipt, "receipt_body_sha256")
    with pytest.raises(
        materialization.ForagerMatchedV3QuicknetSourceMaterializationError,
        match="archive-tree binding",
    ):
        materialization._parse_receipt(
            mutated,
            expected_receipt_sha256=hashlib.sha256(mutated).hexdigest(),
            manifest_bytes=manifest_raw,
            expected_manifest_sha256=hashlib.sha256(manifest_raw).hexdigest(),
            specs=specs,
            pins=pins,
            plan_bytes=materialization._PLAN_BYTES,
        )


@pytest.mark.parametrize(
    "raw",
    [
        b'{"x":1,"x":2}\n',
        b'{"x":NaN}\n',
        b'{"x":Infinity}\n',
        b'{"x":1.0}\n',
        b'{"x":true}\n',
        b'{"x":999999999999999999999}\n',
        b'{"x":"\xff"}\n',
        b"[]\n",
    ],
)
def test_artifact_parsers_reject_noncanonical_or_wrong_identity_json(raw: bytes) -> None:
    with pytest.raises(materialization.ForagerMatchedV3QuicknetSourceMaterializationError):
        materialization.parse_matched_v3_quicknet_source_materialization_plan(
            raw,
            expected_plan_sha256=hashlib.sha256(raw).hexdigest(),
        )


def test_plan_parser_requires_exact_canonical_bytes_and_caller_digest() -> None:
    raw = materialization.canonical_matched_v3_quicknet_source_materialization_plan_bytes()
    for mutated in (b" " + raw, raw[:-1], raw + b"\n", raw.replace(b":", b": ", 1)):
        with pytest.raises(materialization.ForagerMatchedV3QuicknetSourceMaterializationError):
            materialization.parse_matched_v3_quicknet_source_materialization_plan(
                mutated,
                expected_plan_sha256=hashlib.sha256(mutated).hexdigest(),
            )
    with pytest.raises(
        materialization.ForagerMatchedV3QuicknetSourceMaterializationError,
        match="full-file digest",
    ):
        materialization.parse_matched_v3_quicknet_source_materialization_plan(
            raw,
            expected_plan_sha256="f" * 64,
        )


def test_boolean_integer_and_container_aliases_fail_closed() -> None:
    specs, pins, manifest_raw, _ = _pair_artifacts()
    value = cast(dict[str, Any], json.loads(manifest_raw))
    value["archive_inventories"][0]["entry_count"] = True
    mutated = _replace_body_digest(value, "manifest_body_sha256")
    with pytest.raises(materialization.ForagerMatchedV3QuicknetSourceMaterializationError):
        materialization._parse_manifest(
            mutated,
            expected_manifest_sha256=hashlib.sha256(mutated).hexdigest(),
            specs=specs,
            pins=pins,
            plan_bytes=materialization._PLAN_BYTES,
        )


def test_manifest_replay_rejects_noncanonical_paths_oversize_files_and_changed_limitations() -> (
    None
):
    specs, pins, manifest_raw, _ = _pair_artifacts()

    value = cast(dict[str, Any], json.loads(manifest_raw))
    directory = next(
        entry
        for entry in value["archive_inventories"][0]["entries"]
        if entry["entry_type"] == "directory"
    )
    directory["path"] += "/"
    value["archive_inventories"][0]["tree_sha256"] = materialization._entry_tree_sha256(
        value["archive_inventories"][0]["entries"]
    )
    mutated = _replace_body_digest(value, "manifest_body_sha256")
    with pytest.raises(
        materialization.ForagerMatchedV3QuicknetSourceMaterializationError,
        match="canonical recorded form",
    ):
        materialization._parse_manifest(
            mutated,
            expected_manifest_sha256=hashlib.sha256(mutated).hexdigest(),
            specs=specs,
            pins=pins,
            plan_bytes=materialization._PLAN_BYTES,
        )

    value = cast(dict[str, Any], json.loads(manifest_raw))
    target = next(
        entry
        for entry in value["archive_inventories"][0]["entries"]
        if entry["path"] == "only-commit"
    )
    target["size_bytes"] = materialization._MAX_REGULAR_FILE_BYTES + 1
    value["archive_inventories"][0]["total_regular_file_bytes"] += (
        materialization._MAX_REGULAR_FILE_BYTES + 1 - len(b"commit")
    )
    value["archive_inventories"][0]["tree_sha256"] = materialization._entry_tree_sha256(
        value["archive_inventories"][0]["entries"]
    )
    mutated = _replace_body_digest(value, "manifest_body_sha256")
    with pytest.raises(materialization.ForagerMatchedV3QuicknetSourceMaterializationError):
        materialization._parse_manifest(
            mutated,
            expected_manifest_sha256=hashlib.sha256(mutated).hexdigest(),
            specs=specs,
            pins=pins,
            plan_bytes=materialization._PLAN_BYTES,
        )

    value = cast(dict[str, Any], json.loads(manifest_raw))
    value["limitations"] = ["caller-selected prose"]
    mutated = _replace_body_digest(value, "manifest_body_sha256")
    with pytest.raises(
        materialization.ForagerMatchedV3QuicknetSourceMaterializationError,
        match="limitations differ",
    ):
        materialization._parse_manifest(
            mutated,
            expected_manifest_sha256=hashlib.sha256(mutated).hexdigest(),
            specs=specs,
            pins=pins,
            plan_bytes=materialization._PLAN_BYTES,
        )

    value = cast(dict[str, Any], json.loads(manifest_raw))
    value["archive_inventories"][0]["entries"] = {
        entry["path"]: entry for entry in value["archive_inventories"][0]["entries"]
    }
    mutated = _replace_body_digest(value, "manifest_body_sha256")
    with pytest.raises(materialization.ForagerMatchedV3QuicknetSourceMaterializationError):
        materialization._parse_manifest(
            mutated,
            expected_manifest_sha256=hashlib.sha256(mutated).hexdigest(),
            specs=specs,
            pins=pins,
            plan_bytes=materialization._PLAN_BYTES,
        )


def test_source_module_is_stdlib_only_and_exposes_no_operational_imports() -> None:
    module_path = Path(materialization.__file__)
    raw_source = module_path.read_text(encoding="utf-8")
    tree = ast.parse(raw_source)
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_roots.add(node.module.split(".", 1)[0])
    assert imported_roots <= {
        "__future__",
        "collections",
        "dataclasses",
        "hashlib",
        "hmac",
        "json",
        "re",
        "typing",
        "zlib",
    }
    assert imported_roots.isdisjoint(
        {
            "aiohttp",
            "gzip",
            "httpx",
            "os",
            "pathlib",
            "requests",
            "socket",
            "subprocess",
            "tarfile",
            "tempfile",
            "urllib",
        }
    )


def test_descriptor_denies_filesystem_process_rust_vendor_and_qualification_authority() -> None:
    descriptor = materialization.matched_v3_quicknet_source_materialization_descriptor()
    capabilities = descriptor["capabilities"]
    assert capabilities["caller_supplied_archive_inventory_api_exposed"] is True
    for key, value in capabilities.items():
        if key != "caller_supplied_archive_inventory_api_exposed":
            assert value is False
    assert set(descriptor["authority"].values()) == {False}
    assert descriptor["state"]["archive_bytes_embedded"] is False
    assert descriptor["state"]["archive_bytes_fetched"] is False
    assert descriptor["state"]["archive_bytes_materialized"] is False
    assert descriptor["state"]["filesystem_materialization_performed"] is False
    assert descriptor["state"]["dependency_vendor_closure_available"] is False
    assert descriptor["state"]["rust_built"] is False
    assert descriptor["state"]["verifier_invoked"] is False
    assert descriptor["state"]["qualification_ready"] is False


def test_result_dataclass_is_frozen_and_detached() -> None:
    _, _, manifest_raw, receipt_raw = _pair_artifacts()
    result = materialization.MatchedV3QuicknetSourceArchiveInventory(
        manifest_bytes=manifest_raw,
        manifest_sha256=hashlib.sha256(manifest_raw).hexdigest(),
        receipt_bytes=receipt_raw,
        receipt_sha256=hashlib.sha256(receipt_raw).hexdigest(),
    )
    first = copy.deepcopy(cast(dict[str, Any], json.loads(result.manifest_bytes)))
    first["state"]["qualification_ready"] = True
    assert json.loads(result.manifest_bytes)["state"]["qualification_ready"] is False
    with pytest.raises((AttributeError, TypeError)):
        result.manifest_sha256 = "f" * 64  # type: ignore[misc]
