"""Bounded in-memory inventory contract for the pinned Quicknet verifier source.

This source-only module defines a frozen plan and can inspect two exact,
caller-supplied gzip-compressed tar archives entirely in memory.  It never
fetches or writes bytes, extracts a path, starts a process, invokes Rust, or
selects a build input.  A successful receipt means only that the two pinned
archives passed the bounded structural inventory and comparison described by
the plan.  It is not a filesystem-materialization, dependency-vendor, build,
verification, trust, chronology, seed, qualification, or evidence receipt.

After Python has initialized this module's parent packages, this leaf module
body constructs deterministic in-memory constants only.  Normal dotted import
still executes parent package initializers first; their behavior is outside
this module's leaf-body claim.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, NoReturn, cast

MATCHED_V3_QUICKNET_SOURCE_MATERIALIZATION_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.quicknet_source_materialization_descriptor.v1"
)
MATCHED_V3_QUICKNET_SOURCE_MATERIALIZATION_PLAN_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.quicknet_source_materialization_plan.v1"
)
MATCHED_V3_QUICKNET_SOURCE_MATERIALIZATION_MANIFEST_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.quicknet_source_materialization_manifest.v1"
)
MATCHED_V3_QUICKNET_SOURCE_MATERIALIZATION_RECEIPT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.quicknet_source_materialization_receipt.v1"
)

MATCHED_V3_QUICKNET_SOURCE_MATERIALIZATION_STATUS: Final = (
    "source_only_in_memory_archive_inventory_nonauthorizing"
)
MATCHED_V3_QUICKNET_VERIFIER_SOURCE_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.quicknet_verifier_source_descriptor.v2"
)
MATCHED_V3_QUICKNET_VERIFIER_SOURCE_DESCRIPTOR_SHA256: Final = (
    "4d2241ebf8e4e431e33addf317c116531a6605a391906f6bddf18491e0764fdd"
)
MATCHED_V3_QUICKNET_VERIFIER_SOURCE_MODULE_SHA256: Final = (
    "3e13009c1843c3341e5a0eb8b2f84ea903b8e5315fbdef347549757710fd3623"
)

UPSTREAM_COMMIT_GIT_SHA1: Final = "1db2248afac44fc2e5c9c78f896b4412d8679914"
UPSTREAM_COMMIT_ARCHIVE_SHA256: Final = (
    "633408b2d2adca4d9986e765ee2ece148b26de50f7440db5c5f3f7054edfe760"
)
UPSTREAM_COMMIT_ARCHIVE_SIZE_BYTES: Final = 18_727
UPSTREAM_COMMIT_ARCHIVE_TOP_LEVEL_PREFIX: Final = (
    "drand-verify-1db2248afac44fc2e5c9c78f896b4412d8679914"
)
UPSTREAM_CRATE_NAME: Final = "drand-verify"
UPSTREAM_CRATE_VERSION: Final = "0.6.2"
UPSTREAM_CRATE_ARCHIVE_SHA256: Final = (
    "4c1d531704590bbfce3433cd735378d135cabc9e318d8aa52c5dccf7b80178ee"
)
UPSTREAM_CRATE_ARCHIVE_SIZE_BYTES: Final = 18_961
UPSTREAM_CRATE_ARCHIVE_TOP_LEVEL_PREFIX: Final = "drand-verify-0.6.2"

_COMMIT_ARCHIVE_ID: Final = "upstream_commit_archive"
_CRATE_ARCHIVE_ID: Final = "crates_io_package_archive"
_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_MODE_RE: Final = re.compile(r"0[0-7]{3}\Z")
_DOS_RESERVED_COMPONENT_RE: Final = re.compile(
    r"(?:con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\..*)?\Z", re.IGNORECASE
)

_MAX_COMPRESSED_ARCHIVE_BYTES: Final = 64 * 1024 * 1024
_MAX_UNCOMPRESSED_TAR_BYTES: Final = 16 * 1024 * 1024
_MAX_REGULAR_FILE_BYTES: Final = 8 * 1024 * 1024
_MAX_TOTAL_REGULAR_FILE_BYTES: Final = 16 * 1024 * 1024
_MAX_PRODUCER_SOURCE_BYTES: Final = 4 * 1024 * 1024
_MAX_ARCHIVE_MEMBERS: Final = 4_096
_MAX_PATH_BYTES: Final = 1_024
_MAX_PATH_COMPONENT_BYTES: Final = 255
_MAX_PATH_COMPONENTS: Final = 128
_GZIP_INPUT_CHUNK_BYTES: Final = 64 * 1024
_TAR_BLOCK_BYTES: Final = 512
_MAX_ARTIFACT_BYTES: Final = 8 * 1024 * 1024
_MAX_JSON_DEPTH: Final = 64
_MAX_JSON_NODES: Final = 500_000
_MAX_JSON_TEXT_LENGTH: Final = 16_384
_MAX_JSON_INTEGER: Final = (1 << 63) - 1

_PINNED_RELEVANT_FILES: Final = (
    (
        _CRATE_ARCHIVE_ID,
        ".cargo_vcs_info.json",
        "f304ef56e003d4cf1c29f052279ffafed2fd83d7a49ffce07377fc66687060fa",
    ),
    (
        _COMMIT_ARCHIVE_ID,
        "CHANGELOG.md",
        "f0393d5e35a54a5ad9181307fb4b242912925542bf15636cb817da12dafeab94",
    ),
    (
        _COMMIT_ARCHIVE_ID,
        "Cargo.lock",
        "6dd200178128e6e02788b194c856ff3668abf2916b321431790760c950739767",
    ),
    (
        _COMMIT_ARCHIVE_ID,
        "Cargo.toml",
        "499d25339dc90d107633cab975404ea1fdac8e03b4f784fa64e5f06dccf04cc1",
    ),
    (
        _COMMIT_ARCHIVE_ID,
        "LICENSE",
        "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30",
    ),
    (
        _COMMIT_ARCHIVE_ID,
        "MAINTENANCE.md",
        "f6b31cf4f31d718253bfa3a1ddbcc32cf83ff71a745006f7fc6b9cd89c0a5272",
    ),
    (
        _COMMIT_ARCHIVE_ID,
        "NOTICE",
        "3322338486638129acbae928b2025eac84395088ff92cdc2f07528e00312ac8e",
    ),
    (
        _COMMIT_ARCHIVE_ID,
        "README.md",
        "b45615e1648c152d60bc784b28e9ba0d0ec2618ecc84acef7eb09d7e9618b3a5",
    ),
    (
        _COMMIT_ARCHIVE_ID,
        "src/lib.rs",
        "5aab4357c622f089cb0a25825356f6cece18259c42af8aca1069c8708a15b97c",
    ),
    (
        _COMMIT_ARCHIVE_ID,
        "src/points.rs",
        "7fa15e818aead5758a44306b304d35c8ff3ab3d9491e8370fa4698546f545eee",
    ),
    (
        _COMMIT_ARCHIVE_ID,
        "src/randomness.rs",
        "d66781a3e78b61fe64b3e734b8f159e73926f13dc4badd546498446c1418575b",
    ),
    (
        _COMMIT_ARCHIVE_ID,
        "src/verify.rs",
        "47c7a755b4bc226371df83ec8dc430c7a37f0f0ef2bcf55f898cad52e100f9a6",
    ),
)


class ForagerMatchedV3QuicknetSourceMaterializationError(ValueError):
    """The source-only Quicknet archive contract failed closed."""


@dataclass(frozen=True)
class _ArchiveSpec:
    archive_id: str
    sha256: str
    size_bytes: int
    top_level_prefix: str


@dataclass(frozen=True)
class _RelevantFilePin:
    archive_id: str
    path: str
    sha256: str


@dataclass(frozen=True)
class MatchedV3QuicknetSourceArchiveInventory:
    """Detached canonical artifacts produced from caller-supplied source bytes."""

    manifest_bytes: bytes
    manifest_sha256: str
    receipt_bytes: bytes
    receipt_sha256: str

    def manifest(self) -> dict[str, Any]:
        """Replay the retained manifest under its full-file digest."""

        return parse_matched_v3_quicknet_source_materialization_manifest(
            self.manifest_bytes,
            expected_manifest_sha256=self.manifest_sha256,
        )

    def receipt(self) -> dict[str, Any]:
        """Replay the retained receipt and its exact manifest cross-link."""

        return parse_matched_v3_quicknet_source_materialization_receipt(
            self.receipt_bytes,
            expected_receipt_sha256=self.receipt_sha256,
            manifest_bytes=self.manifest_bytes,
            expected_manifest_sha256=self.manifest_sha256,
        )


_PRODUCTION_ARCHIVE_SPECS: Final = (
    _ArchiveSpec(
        archive_id=_COMMIT_ARCHIVE_ID,
        sha256=UPSTREAM_COMMIT_ARCHIVE_SHA256,
        size_bytes=UPSTREAM_COMMIT_ARCHIVE_SIZE_BYTES,
        top_level_prefix=UPSTREAM_COMMIT_ARCHIVE_TOP_LEVEL_PREFIX,
    ),
    _ArchiveSpec(
        archive_id=_CRATE_ARCHIVE_ID,
        sha256=UPSTREAM_CRATE_ARCHIVE_SHA256,
        size_bytes=UPSTREAM_CRATE_ARCHIVE_SIZE_BYTES,
        top_level_prefix=UPSTREAM_CRATE_ARCHIVE_TOP_LEVEL_PREFIX,
    ),
)
_PRODUCTION_RELEVANT_PINS: Final = tuple(
    _RelevantFilePin(archive_id=archive_id, path=path, sha256=sha256)
    for archive_id, path, sha256 in _PINNED_RELEVANT_FILES
)


def _fail(message: str) -> NoReturn:
    raise ForagerMatchedV3QuicknetSourceMaterializationError(message)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        _fail(f"{label} must be one lowercase SHA-256 digest")
    return value


def _require_bool(value: object, *, label: str) -> bool:
    if type(value) is not bool:
        _fail(f"{label} must be one exact boolean")
    return value


def _require_int(
    value: object,
    *,
    label: str,
    minimum: int = 0,
    maximum: int = _MAX_JSON_INTEGER,
) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        _fail(f"{label} must be an exact bounded integer")
    return value


def _require_str(value: object, *, label: str) -> str:
    if type(value) is not str or not value:
        _fail(f"{label} must be one nonempty exact string")
    return value


def _exact_keys(value: object, expected: set[str], *, label: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != expected:
        _fail(f"{label} keys differ")
    return cast(dict[str, Any], value)


def _reject_json_constant(value: str) -> NoReturn:
    _fail(f"Quicknet materialization JSON contains non-finite constant {value!r}")


def _reject_json_float(value: str) -> NoReturn:
    _fail(f"Quicknet materialization JSON contains forbidden float {value!r}")


def _parse_json_int(value: str) -> int:
    if len(value.lstrip("-")) > 19:
        _fail("Quicknet materialization JSON integer exceeds its lexical bound")
    parsed = int(value)
    if not -_MAX_JSON_INTEGER <= parsed <= _MAX_JSON_INTEGER:
        _fail("Quicknet materialization JSON integer exceeds its value bound")
    return parsed


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"Quicknet materialization JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _validate_json_tree(value: object) -> None:
    nodes = 0

    def visit(item: object, depth: int) -> None:
        nonlocal nodes
        nodes += 1
        if nodes > _MAX_JSON_NODES:
            _fail("Quicknet materialization JSON exceeds its node bound")
        if depth > _MAX_JSON_DEPTH:
            _fail("Quicknet materialization JSON exceeds its depth bound")
        if type(item) is str:
            text = item
            if len(text) > _MAX_JSON_TEXT_LENGTH or any(
                ord(character) < 0x20 or ord(character) > 0x7E for character in text
            ):
                _fail("Quicknet materialization JSON strings must be bounded printable ASCII")
            return
        if item is None or type(item) is bool:
            return
        if type(item) is int:
            _require_int(item, label="Quicknet materialization JSON integer", minimum=0)
            return
        if type(item) is list:
            for child in cast(list[object], item):
                visit(child, depth + 1)
            return
        if type(item) is dict:
            for key, child in cast(dict[object, object], item).items():
                if type(key) is not str:
                    _fail("Quicknet materialization JSON object keys must be exact strings")
                visit(key, depth + 1)
                visit(child, depth + 1)
            return
        _fail("Quicknet materialization JSON contains a non-JSON value")

    visit(value, 0)


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    if type(value) is not dict:
        _fail("Quicknet materialization JSON root must be one plain object")
    _validate_json_tree(value)
    try:
        raw = (
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("ascii")
            + b"\n"
        )
    except (OverflowError, RecursionError, TypeError, UnicodeError, ValueError) as exc:
        raise ForagerMatchedV3QuicknetSourceMaterializationError(
            "Quicknet materialization artifact is not canonical ASCII JSON"
        ) from exc
    if len(raw) > _MAX_ARTIFACT_BYTES:
        _fail("Quicknet materialization artifact exceeds its byte bound")
    return raw


def _strict_json_load(raw: bytes, *, label: str) -> dict[str, Any]:
    if type(raw) is not bytes:
        _fail(f"{label} must be exact bytes")
    if not raw or len(raw) > _MAX_ARTIFACT_BYTES:
        _fail(f"{label} violates its byte bound")
    try:
        decoded = raw.decode("ascii", errors="strict")
        value = json.loads(
            decoded,
            object_pairs_hook=_object_without_duplicates,
            parse_constant=_reject_json_constant,
            parse_float=_reject_json_float,
            parse_int=_parse_json_int,
        )
    except ForagerMatchedV3QuicknetSourceMaterializationError:
        raise
    except (RecursionError, UnicodeError, ValueError) as exc:
        raise ForagerMatchedV3QuicknetSourceMaterializationError(
            f"{label} is not strict ASCII JSON"
        ) from exc
    if type(value) is not dict:
        _fail(f"{label} root must be one plain object")
    result = cast(dict[str, Any], value)
    _validate_json_tree(result)
    if not hmac.compare_digest(_canonical_json_bytes(result), raw):
        _fail(f"{label} bytes are not canonical")
    return result


def _body_bound_artifact(body: dict[str, Any], *, digest_field: str) -> dict[str, Any]:
    artifact = dict(body)
    artifact[digest_field] = _sha256(_canonical_json_bytes(body))
    return artifact


def _verify_body_digest(
    value: dict[str, Any],
    *,
    digest_field: str,
    label: str,
) -> None:
    supplied = _require_sha256(value[digest_field], label=f"{label} body digest")
    body = {key: item for key, item in value.items() if key != digest_field}
    if not hmac.compare_digest(supplied, _sha256(_canonical_json_bytes(body))):
        _fail(f"{label} body digest differs")


def _denied_authority() -> dict[str, bool]:
    return {
        "acceptance_authority_granted": False,
        "build_authority_granted": False,
        "chronology_authority_granted": False,
        "execution_authority_granted": False,
        "network_authority_granted": False,
        "publication_authority_granted": False,
        "qualification_authority_granted": False,
        "scientific_evidence_authority_granted": False,
        "seed_issuer_authority_granted": False,
        "trust_root_authority_granted": False,
        "promotion_authority_granted": False,
        "performance_or_sota_claim_authority_granted": False,
    }


def _source_registry_identity() -> dict[str, Any]:
    return {
        "descriptor_schema_version": (
            MATCHED_V3_QUICKNET_VERIFIER_SOURCE_DESCRIPTOR_SCHEMA_VERSION
        ),
        "descriptor_sha256": MATCHED_V3_QUICKNET_VERIFIER_SOURCE_DESCRIPTOR_SHA256,
        "source_module_sha256": MATCHED_V3_QUICKNET_VERIFIER_SOURCE_MODULE_SHA256,
        "source_module_bytes_read_here": False,
        "descriptor_imported_here": False,
    }


def _archive_spec_record(spec: _ArchiveSpec) -> dict[str, Any]:
    return {
        "archive_id": spec.archive_id,
        "compression": "gzip_single_member",
        "container": "ustar_regular_and_directory_entries_only",
        "sha256": spec.sha256,
        "size_bytes": spec.size_bytes,
        "top_level_prefix": spec.top_level_prefix,
    }


def _relevant_pin_records(pins: Sequence[_RelevantFilePin]) -> list[dict[str, Any]]:
    return [
        {
            "archive_id": pin.archive_id,
            "path": pin.path,
            "sha256": pin.sha256,
        }
        for pin in pins
    ]


def _inventory_policy() -> dict[str, Any]:
    return {
        "archive_bytes_origin": "caller_supplied_only",
        "compressed_archive_maximum_bytes": _MAX_COMPRESSED_ARCHIVE_BYTES,
        "decompressed_tar_maximum_bytes": _MAX_UNCOMPRESSED_TAR_BYTES,
        "regular_file_maximum_bytes": _MAX_REGULAR_FILE_BYTES,
        "total_regular_file_maximum_bytes": _MAX_TOTAL_REGULAR_FILE_BYTES,
        "archive_member_maximum_count": _MAX_ARCHIVE_MEMBERS,
        "path_maximum_bytes": _MAX_PATH_BYTES,
        "path_component_maximum_bytes": _MAX_PATH_COMPONENT_BYTES,
        "path_component_maximum_count": _MAX_PATH_COMPONENTS,
        "gzip_member_count_required": 1,
        "gzip_trailing_bytes_allowed": False,
        "tar_terminal_zero_blocks_minimum": 2,
        "accepted_tar_entry_types": ["directory", "regular_file"],
        "pax_entries_allowed": False,
        "gnu_long_name_entries_allowed": False,
        "sparse_entries_allowed": False,
        "link_entries_allowed": False,
        "special_entries_allowed": False,
        "path_aliases_allowed": False,
        "archive_extraction_performed": False,
        "filesystem_write_performed": False,
    }


def _plan_body() -> dict[str, Any]:
    return {
        "schema_version": MATCHED_V3_QUICKNET_SOURCE_MATERIALIZATION_PLAN_SCHEMA_VERSION,
        "status": "frozen_source_only_plan_archives_not_supplied",
        "producer": {
            "descriptor_schema_version": (
                MATCHED_V3_QUICKNET_SOURCE_MATERIALIZATION_DESCRIPTOR_SCHEMA_VERSION
            ),
            "descriptor_sha256": (MATCHED_V3_QUICKNET_SOURCE_MATERIALIZATION_DESCRIPTOR_SHA256),
        },
        "source_registry": _source_registry_identity(),
        "archive_inputs": [_archive_spec_record(spec) for spec in _PRODUCTION_ARCHIVE_SPECS],
        "relevant_file_pins": _relevant_pin_records(_PRODUCTION_RELEVANT_PINS),
        "inventory_policy": _inventory_policy(),
        "cross_archive_policy": {
            "compare_after_exact_top_level_prefix_removal": True,
            "compare_union_of_all_inventory_paths": True,
            "compare_entry_type": True,
            "compare_mode": True,
            "compare_size_bytes": True,
            "compare_content_sha256": True,
            "archive_equivalence_required": False,
            "crate_selected_as_build_source": False,
            "commit_archive_selected_as_build_source": False,
        },
        "state": {
            "archive_bytes_supplied": False,
            "archives_inventory_completed": False,
            "cross_archive_comparison_completed": False,
            "filesystem_materialization_performed": False,
            "primary_build_source_selected": False,
            "dependency_vendor_closure_available": False,
            "qualification_ready": False,
        },
        "authority": _denied_authority(),
        "limitations": [
            "The plan is useful before either archive is supplied and proves no materialization.",
            "Relevant-file pins are a required subset and are not a full archive inventory.",
            "A full primary-crate inventory is not a Cargo dependency vendor closure.",
            "Cross-archive differences are recorded and do not choose a build source.",
            "No successful artifact from this module grants execution or qualification authority.",
        ],
    }


def _descriptor() -> dict[str, Any]:
    return {
        "schema_version": (MATCHED_V3_QUICKNET_SOURCE_MATERIALIZATION_DESCRIPTOR_SCHEMA_VERSION),
        "status": MATCHED_V3_QUICKNET_SOURCE_MATERIALIZATION_STATUS,
        "classification": "source_only_archive_inventory_contract_nonqualifying",
        "schemas": {
            "plan": MATCHED_V3_QUICKNET_SOURCE_MATERIALIZATION_PLAN_SCHEMA_VERSION,
            "manifest": MATCHED_V3_QUICKNET_SOURCE_MATERIALIZATION_MANIFEST_SCHEMA_VERSION,
            "receipt": MATCHED_V3_QUICKNET_SOURCE_MATERIALIZATION_RECEIPT_SCHEMA_VERSION,
        },
        "import_boundary": {
            "claim_scope": "leaf_module_body_after_parent_package_initialization",
            "leaf_module_body_constructs_deterministic_in_memory_constants_only": True,
            "dotted_import_is_hermetic": False,
            "parent_package_initializers_execute_first": True,
            "parent_initializer_behavior_audited_here": False,
        },
        "source_registry": _source_registry_identity(),
        "archive_inputs": [_archive_spec_record(spec) for spec in _PRODUCTION_ARCHIVE_SPECS],
        "inventory_policy": _inventory_policy(),
        "capabilities": {
            "caller_supplied_archive_inventory_api_exposed": True,
            "filesystem_read_api_exposed": False,
            "filesystem_write_api_exposed": False,
            "network_api_exposed": False,
            "process_api_exposed": False,
            "rust_api_exposed": False,
            "archive_extraction_api_exposed": False,
            "dependency_vendor_api_exposed": False,
            "build_source_selection_api_exposed": False,
            "verifier_api_exposed": False,
            "seed_issuer_api_exposed": False,
        },
        "state": {
            "plan_defined": True,
            "archive_bytes_embedded": False,
            "archive_bytes_fetched": False,
            "archive_bytes_materialized": False,
            "filesystem_materialization_performed": False,
            "dependency_vendor_closure_available": False,
            "rust_built": False,
            "verifier_invoked": False,
            "qualification_ready": False,
        },
        "authority": _denied_authority(),
        "limitations": [
            "Importing the leaf module does not supply, fetch, or inventory either archive.",
            "An inventory receipt records bounded in-memory inspection, not path extraction.",
            "Archive SHA-256 pins do not authenticate repository or registry transport.",
            (
                "The two archives may differ; the comparison records differences "
                "without resolving them."
            ),
            "No dependency source closure, Cargo feature policy, toolchain, or binary is defined.",
            (
                "No receipt from this module is a trust root, chronology proof, "
                "seed, or qualification."
            ),
        ],
    }


_DESCRIPTOR_BYTES: Final = _canonical_json_bytes(_descriptor())
MATCHED_V3_QUICKNET_SOURCE_MATERIALIZATION_DESCRIPTOR_SHA256: Final = (
    "61345825673afb16bc1942c4b8c84e763fb14530a68225caffa94d98e733a03d"
)
_PLAN_BYTES: Final = _canonical_json_bytes(
    _body_bound_artifact(_plan_body(), digest_field="plan_body_sha256")
)
MATCHED_V3_QUICKNET_SOURCE_MATERIALIZATION_PLAN_SHA256: Final = _sha256(_PLAN_BYTES)


def matched_v3_quicknet_source_materialization_descriptor() -> dict[str, Any]:
    """Return a detached copy of the frozen authority-denying descriptor."""

    return _strict_json_load(_DESCRIPTOR_BYTES, label="Quicknet materialization descriptor")


def canonical_matched_v3_quicknet_source_materialization_descriptor_bytes() -> bytes:
    """Return the exact canonical materialization descriptor bytes."""

    return _DESCRIPTOR_BYTES


def parse_matched_v3_quicknet_source_materialization_descriptor(
    raw: bytes,
) -> dict[str, Any]:
    """Accept only the exact frozen descriptor."""

    value = _strict_json_load(raw, label="Quicknet materialization descriptor")
    if not hmac.compare_digest(raw, _DESCRIPTOR_BYTES):
        _fail("Quicknet materialization descriptor identity differs")
    return value


def matched_v3_quicknet_source_materialization_plan() -> dict[str, Any]:
    """Return the frozen plan; no source archive is needed for this artifact."""

    return _parse_plan(
        _PLAN_BYTES,
        expected_plan_sha256=MATCHED_V3_QUICKNET_SOURCE_MATERIALIZATION_PLAN_SHA256,
    )


def canonical_matched_v3_quicknet_source_materialization_plan_bytes() -> bytes:
    """Return the exact frozen, nonexecuting plan bytes."""

    return _PLAN_BYTES


def parse_matched_v3_quicknet_source_materialization_plan(
    raw: bytes,
    *,
    expected_plan_sha256: str,
) -> dict[str, Any]:
    """Replay only the exact frozen plan under a caller-carried full digest."""

    return _parse_plan(raw, expected_plan_sha256=expected_plan_sha256)


def _parse_plan(raw: bytes, *, expected_plan_sha256: str) -> dict[str, Any]:
    expected = _require_sha256(expected_plan_sha256, label="expected plan")
    if not hmac.compare_digest(_sha256(raw), expected):
        _fail("Quicknet materialization plan full-file digest differs")
    value = _strict_json_load(raw, label="Quicknet materialization plan")
    _exact_keys(
        value,
        {
            "archive_inputs",
            "authority",
            "cross_archive_policy",
            "inventory_policy",
            "limitations",
            "plan_body_sha256",
            "producer",
            "relevant_file_pins",
            "schema_version",
            "source_registry",
            "state",
            "status",
        },
        label="Quicknet materialization plan",
    )
    _verify_body_digest(value, digest_field="plan_body_sha256", label="plan")
    if not hmac.compare_digest(raw, _PLAN_BYTES):
        _fail("Quicknet materialization plan identity differs")
    return value


def _decompress_single_gzip_member(raw: bytes, *, label: str) -> bytes:
    if type(raw) is not bytes:
        _fail(f"{label} must be exact bytes")
    if not raw or len(raw) > _MAX_COMPRESSED_ARCHIVE_BYTES:
        _fail(f"{label} compressed size violates its bound")
    decompressor = zlib.decompressobj(zlib.MAX_WBITS | 16)
    output = bytearray()
    offset = 0
    try:
        while offset < len(raw) and not decompressor.eof:
            chunk = raw[offset : offset + _GZIP_INPUT_CHUNK_BYTES]
            offset += len(chunk)
            while chunk and not decompressor.eof:
                remaining = _MAX_UNCOMPRESSED_TAR_BYTES - len(output)
                produced = decompressor.decompress(chunk, remaining + 1)
                output.extend(produced)
                if len(output) > _MAX_UNCOMPRESSED_TAR_BYTES:
                    _fail(f"{label} decompressed tar exceeds its byte bound")
                chunk = decompressor.unconsumed_tail
        if not decompressor.eof:
            _fail(f"{label} gzip stream is truncated")
        trailing = decompressor.unused_data + raw[offset:]
        if trailing:
            _fail(f"{label} contains a concatenated member or trailing bytes")
        flushed = decompressor.flush()
        output.extend(flushed)
        if len(output) > _MAX_UNCOMPRESSED_TAR_BYTES:
            _fail(f"{label} decompressed tar exceeds its byte bound")
    except zlib.error as exc:
        raise ForagerMatchedV3QuicknetSourceMaterializationError(
            f"{label} is not one valid gzip member"
        ) from exc
    return bytes(output)


def _decode_tar_text(field: bytes, *, label: str, allow_empty: bool = True) -> str:
    terminator = field.find(b"\0")
    if terminator >= 0:
        encoded = field[:terminator]
        if any(field[terminator:]):
            _fail(f"{label} has nonzero bytes after its NUL terminator")
    else:
        encoded = field
    if not encoded:
        if allow_empty:
            return ""
        _fail(f"{label} is empty")
    if any(byte < 0x20 or byte > 0x7E for byte in encoded):
        _fail(f"{label} must be printable ASCII")
    return encoded.decode("ascii")


def _parse_tar_number(field: bytes, *, label: str) -> int:
    if field and field[0] & 0x80:
        _fail(f"{label} uses forbidden base-256 encoding")
    stripped = field.strip(b"\0 ")
    if not stripped:
        return 0
    if any(byte < ord("0") or byte > ord("7") for byte in stripped):
        _fail(f"{label} is not canonical nonnegative octal")
    value = int(stripped, 8)
    return _require_int(value, label=label)


def _portable_path_key(path: str) -> str:
    return "/".join(component.casefold().rstrip(". ") for component in path.split("/"))


def _canonical_archive_path(raw_path: str, *, entry_type: str, label: str) -> str:
    if not raw_path:
        _fail(f"{label} path is empty")
    if entry_type == "directory":
        if raw_path.endswith("//"):
            _fail(f"{label} directory path has repeated trailing separators")
        path = raw_path[:-1] if raw_path.endswith("/") else raw_path
    else:
        if raw_path.endswith("/"):
            _fail(f"{label} regular-file path ends with a separator")
        path = raw_path
    if (
        not path
        or path.startswith("/")
        or "\\" in path
        or "//" in path
        or any(character in '<>:"|?*' for character in path)
    ):
        _fail(f"{label} path is not a canonical portable relative path")
    encoded = path.encode("ascii", errors="strict")
    if len(encoded) > _MAX_PATH_BYTES:
        _fail(f"{label} path exceeds its byte bound")
    components = path.split("/")
    if len(components) > _MAX_PATH_COMPONENTS:
        _fail(f"{label} path exceeds its component bound")
    for component in components:
        component_bytes = component.encode("ascii")
        if (
            component in {"", ".", ".."}
            or len(component_bytes) > _MAX_PATH_COMPONENT_BYTES
            or component.endswith((".", " "))
            or _DOS_RESERVED_COMPONENT_RE.fullmatch(component) is not None
        ):
            _fail(f"{label} path contains a forbidden or aliased component")
    return path


def _entry_tree_sha256(entries: Sequence[Mapping[str, Any]]) -> str:
    return _sha256(_canonical_json_bytes({"entries": list(entries)}))


def _validate_inventory_paths(entries: Sequence[Mapping[str, Any]], *, label: str) -> None:
    previous_path: str | None = None
    paths: dict[str, str] = {}
    types: dict[str, str] = {}
    for index, entry in enumerate(entries):
        path = _require_str(entry["path"], label=f"{label} entry {index} path")
        entry_type = _require_str(entry["entry_type"], label=f"{label} entry {index} type")
        _canonical_archive_path(path, entry_type=entry_type, label=f"{label} entry {index}")
        if previous_path is not None and previous_path.encode("ascii") >= path.encode("ascii"):
            _fail(f"{label} entries are not strictly bytewise sorted")
        previous_path = path
        alias = _portable_path_key(path)
        if alias in paths:
            _fail(f"{label} paths {paths[alias]!r} and {path!r} alias")
        paths[alias] = path
        types[path] = entry_type
    for path in types:
        components = path.split("/")
        for stop in range(1, len(components)):
            ancestor = "/".join(components[:stop])
            if types.get(ancestor) == "regular_file":
                _fail(f"{label} regular file {ancestor!r} is an ancestor of another entry")


def _inventory_gzip_tar_archive(raw: bytes, *, spec: _ArchiveSpec) -> dict[str, Any]:
    if type(raw) is not bytes:
        _fail(f"{spec.archive_id} must be supplied as exact bytes")
    if len(raw) != spec.size_bytes or not hmac.compare_digest(_sha256(raw), spec.sha256):
        _fail(f"{spec.archive_id} exact archive identity differs")
    tar_bytes = _decompress_single_gzip_member(raw, label=spec.archive_id)
    if not tar_bytes or len(tar_bytes) % _TAR_BLOCK_BYTES != 0:
        _fail(f"{spec.archive_id} tar size is not a positive 512-byte multiple")

    records: list[dict[str, Any]] = []
    raw_member_count = 0
    total_regular_file_bytes = 0
    root_directory_entry_present = False
    seen_paths: dict[str, str] = {}
    offset = 0
    terminal_zero_blocks = 0

    while offset < len(tar_bytes):
        header = tar_bytes[offset : offset + _TAR_BLOCK_BYTES]
        if header == b"\0" * _TAR_BLOCK_BYTES:
            terminal_zero_blocks = (len(tar_bytes) - offset) // _TAR_BLOCK_BYTES
            if terminal_zero_blocks < 2 or any(tar_bytes[offset:]):
                _fail(f"{spec.archive_id} has an invalid tar terminator")
            break
        if raw_member_count >= _MAX_ARCHIVE_MEMBERS:
            _fail(f"{spec.archive_id} exceeds its member-count bound")
        raw_member_count += 1

        supplied_checksum = _parse_tar_number(
            header[148:156], label=f"{spec.archive_id} member checksum"
        )
        checksum_header = header[:148] + b" " * 8 + header[156:]
        if supplied_checksum != sum(checksum_header):
            _fail(f"{spec.archive_id} member header checksum differs")
        magic = header[257:263]
        version = header[263:265]
        if (magic, version) not in {(b"ustar\0", b"00"), (b"ustar ", b" \0")}:
            _fail(f"{spec.archive_id} member is not an admitted USTAR header")
        if any(header[500:512]):
            _fail(f"{spec.archive_id} member uses forbidden tar extension bytes")

        name = _decode_tar_text(
            header[0:100], label=f"{spec.archive_id} member name", allow_empty=False
        )
        prefix = _decode_tar_text(header[345:500], label=f"{spec.archive_id} member prefix")
        raw_path = f"{prefix}/{name}" if prefix else name
        typeflag = header[156:157]
        if typeflag in {b"\0", b"0"}:
            entry_type = "regular_file"
        elif typeflag == b"5":
            entry_type = "directory"
        else:
            _fail(
                f"{spec.archive_id} contains a forbidden link, special, sparse, PAX, "
                "or GNU extension entry"
            )
        path = _canonical_archive_path(
            raw_path,
            entry_type=entry_type,
            label=f"{spec.archive_id} member",
        )
        link_name = _decode_tar_text(header[157:257], label=f"{spec.archive_id} member link name")
        if link_name:
            _fail(f"{spec.archive_id} admitted entry has a nonempty link name")

        mode = _parse_tar_number(header[100:108], label=f"{spec.archive_id} member mode")
        if mode > 0o777:
            _fail(f"{spec.archive_id} member mode contains special bits")
        uid = _parse_tar_number(header[108:116], label=f"{spec.archive_id} member uid")
        gid = _parse_tar_number(header[116:124], label=f"{spec.archive_id} member gid")
        size = _parse_tar_number(header[124:136], label=f"{spec.archive_id} member size")
        mtime = _parse_tar_number(header[136:148], label=f"{spec.archive_id} member mtime")
        device_major = _parse_tar_number(
            header[329:337], label=f"{spec.archive_id} member device major"
        )
        device_minor = _parse_tar_number(
            header[337:345], label=f"{spec.archive_id} member device minor"
        )
        if device_major != 0 or device_minor != 0:
            _fail(f"{spec.archive_id} admitted entry has device metadata")
        uname = _decode_tar_text(header[265:297], label=f"{spec.archive_id} member uname")
        gname = _decode_tar_text(header[297:329], label=f"{spec.archive_id} member gname")

        if entry_type == "directory" and size != 0:
            _fail(f"{spec.archive_id} directory has a nonzero payload")
        if entry_type == "regular_file" and size > _MAX_REGULAR_FILE_BYTES:
            _fail(f"{spec.archive_id} regular file exceeds its byte bound")
        data_start = offset + _TAR_BLOCK_BYTES
        data_end = data_start + size
        padded_end = data_start + ((size + _TAR_BLOCK_BYTES - 1) // _TAR_BLOCK_BYTES) * 512
        if data_end > len(tar_bytes) or padded_end > len(tar_bytes):
            _fail(f"{spec.archive_id} member payload is truncated")
        if any(tar_bytes[data_end:padded_end]):
            _fail(f"{spec.archive_id} member has nonzero tar padding")
        payload = tar_bytes[data_start:data_end]
        offset = padded_end

        prefix_path = spec.top_level_prefix
        if path == prefix_path:
            if entry_type != "directory" or root_directory_entry_present:
                _fail(f"{spec.archive_id} top-level prefix entry differs")
            root_directory_entry_present = True
            continue
        required_prefix = prefix_path + "/"
        if not path.startswith(required_prefix):
            _fail(f"{spec.archive_id} member escapes its exact top-level prefix")
        relative = path[len(required_prefix) :]
        relative = _canonical_archive_path(
            relative,
            entry_type=entry_type,
            label=f"{spec.archive_id} relative member",
        )
        alias = _portable_path_key(relative)
        if alias in seen_paths:
            _fail(
                f"{spec.archive_id} paths {seen_paths[alias]!r} and {relative!r} "
                "are duplicates or portable aliases"
            )
        seen_paths[alias] = relative
        if entry_type == "regular_file":
            total_regular_file_bytes += size
            if total_regular_file_bytes > _MAX_TOTAL_REGULAR_FILE_BYTES:
                _fail(f"{spec.archive_id} total regular-file bytes exceed their bound")
        records.append(
            {
                "entry_type": entry_type,
                "gid": gid,
                "gname": gname,
                "header_sha256": _sha256(header),
                "mode": f"0{mode:03o}",
                "mtime": mtime,
                "path": relative,
                "sha256": _sha256(payload),
                "size_bytes": size,
                "uid": uid,
                "uname": uname,
            }
        )
    else:
        _fail(f"{spec.archive_id} tar is missing its terminal zero blocks")

    if terminal_zero_blocks < 2:
        _fail(f"{spec.archive_id} tar is missing its terminal zero blocks")
    records.sort(key=lambda record: cast(str, record["path"]).encode("ascii"))
    _validate_inventory_paths(records, label=spec.archive_id)
    regular_file_count = sum(record["entry_type"] == "regular_file" for record in records)
    directory_count = len(records) - regular_file_count
    if regular_file_count == 0:
        _fail(f"{spec.archive_id} contains no regular files")
    return {
        "archive_id": spec.archive_id,
        "compression": "gzip_single_member",
        "gzip_member_count": 1,
        "raw_archive_sha256": spec.sha256,
        "raw_archive_size_bytes": spec.size_bytes,
        "tar_format": "bounded_ustar",
        "tar_uncompressed_size_bytes": len(tar_bytes),
        "tar_terminal_zero_block_count": terminal_zero_blocks,
        "top_level_prefix": spec.top_level_prefix,
        "root_directory_entry_present": root_directory_entry_present,
        "raw_member_count": raw_member_count,
        "entry_count": len(records),
        "regular_file_count": regular_file_count,
        "directory_count": directory_count,
        "total_regular_file_bytes": total_regular_file_bytes,
        "entries": records,
        "tree_sha256": _entry_tree_sha256(records),
    }


def _entry_projection(entry: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "entry_type": entry["entry_type"],
        "mode": entry["mode"],
        "sha256": entry["sha256"],
        "size_bytes": entry["size_bytes"],
    }


def _cross_archive_comparison(
    commit_inventory: Mapping[str, Any],
    crate_inventory: Mapping[str, Any],
) -> dict[str, Any]:
    commit_entries = {
        cast(str, entry["path"]): cast(Mapping[str, Any], entry)
        for entry in cast(list[dict[str, Any]], commit_inventory["entries"])
    }
    crate_entries = {
        cast(str, entry["path"]): cast(Mapping[str, Any], entry)
        for entry in cast(list[dict[str, Any]], crate_inventory["entries"])
    }
    common: list[dict[str, Any]] = []
    common_paths = sorted(
        set(commit_entries) & set(crate_entries), key=lambda item: item.encode("ascii")
    )
    for path in common_paths:
        commit = _entry_projection(commit_entries[path])
        crate = _entry_projection(crate_entries[path])
        common.append(
            {
                "commit": commit,
                "content_identity_equal": (
                    commit["entry_type"] == crate["entry_type"]
                    and commit["size_bytes"] == crate["size_bytes"]
                    and commit["sha256"] == crate["sha256"]
                ),
                "crate": crate,
                "mode_equal": commit["mode"] == crate["mode"],
                "path": path,
            }
        )
    comparison_body: dict[str, Any] = {
        "commit_archive_id": commit_inventory["archive_id"],
        "crate_archive_id": crate_inventory["archive_id"],
        "common_entries": common,
        "common_entry_count": len(common),
        "content_identical_common_entry_count": sum(
            record["content_identity_equal"] is True for record in common
        ),
        "mode_identical_common_entry_count": sum(record["mode_equal"] is True for record in common),
        "commit_only_paths": sorted(
            set(commit_entries) - set(crate_entries), key=lambda item: item.encode("ascii")
        ),
        "crate_only_paths": sorted(
            set(crate_entries) - set(commit_entries), key=lambda item: item.encode("ascii")
        ),
        "archives_declared_equivalent": False,
        "build_source_selected": False,
    }
    return _body_bound_artifact(comparison_body, digest_field="comparison_body_sha256")


def _pin_checks(
    inventories: Sequence[Mapping[str, Any]],
    pins: Sequence[_RelevantFilePin],
) -> list[dict[str, Any]]:
    by_archive: dict[str, dict[str, Mapping[str, Any]]] = {}
    for inventory in inventories:
        archive_id = cast(str, inventory["archive_id"])
        by_archive[archive_id] = {
            cast(str, entry["path"]): cast(Mapping[str, Any], entry)
            for entry in cast(list[dict[str, Any]], inventory["entries"])
        }
    checks: list[dict[str, Any]] = []
    for pin in pins:
        archive_entries = by_archive.get(pin.archive_id)
        entry = None if archive_entries is None else archive_entries.get(pin.path)
        if entry is None or entry["entry_type"] != "regular_file" or entry["sha256"] != pin.sha256:
            _fail(f"pinned relevant file {pin.archive_id}:{pin.path} differs")
        checks.append(
            {
                "archive_id": pin.archive_id,
                "entry_type": "regular_file",
                "expected_sha256": pin.sha256,
                "matched": True,
                "observed_sha256": entry["sha256"],
                "path": pin.path,
            }
        )
    return checks


def _manifest_state() -> dict[str, bool]:
    return {
        "archive_bytes_supplied_by_caller": True,
        "archives_inventory_completed": True,
        "cross_archive_comparison_completed": True,
        "relevant_file_pins_matched": True,
        "filesystem_materialization_performed": False,
        "archive_member_extraction_performed": False,
        "primary_build_source_selected": False,
        "dependency_vendor_closure_available": False,
        "rust_built": False,
        "verifier_invoked": False,
        "qualification_ready": False,
    }


def _manifest_limitations() -> list[str]:
    return [
        "All archive processing occurred in memory; no member was extracted to a path.",
        "The manifest records source archives but does not select either as build input.",
        "The manifest is not a dependency vendor closure or security-audit receipt.",
        "Matching relevant-file pins does not authenticate archive transport.",
        "This manifest grants no execution, trust, seed, qualification, or evidence authority.",
    ]


def _build_manifest(
    inventories: Sequence[Mapping[str, Any]],
    *,
    pins: Sequence[_RelevantFilePin],
    plan_bytes: bytes,
) -> bytes:
    if len(inventories) != 2:
        _fail("Quicknet materialization requires exactly two archive inventories")
    checks = _pin_checks(inventories, pins)
    body: dict[str, Any] = {
        "schema_version": MATCHED_V3_QUICKNET_SOURCE_MATERIALIZATION_MANIFEST_SCHEMA_VERSION,
        "status": "caller_bytes_safely_inventoried_in_memory_nonauthorizing",
        "plan": {
            "schema_version": MATCHED_V3_QUICKNET_SOURCE_MATERIALIZATION_PLAN_SCHEMA_VERSION,
            "sha256": _sha256(plan_bytes),
        },
        "source_registry": _source_registry_identity(),
        "archive_inventories": list(inventories),
        "relevant_file_pin_checks": checks,
        "cross_archive_comparison": _cross_archive_comparison(inventories[0], inventories[1]),
        "state": _manifest_state(),
        "authority": _denied_authority(),
        "limitations": _manifest_limitations(),
    }
    return _canonical_json_bytes(_body_bound_artifact(body, digest_field="manifest_body_sha256"))


def _validate_entry(entry: object, *, label: str) -> dict[str, Any]:
    record = _exact_keys(
        entry,
        {
            "entry_type",
            "gid",
            "gname",
            "header_sha256",
            "mode",
            "mtime",
            "path",
            "sha256",
            "size_bytes",
            "uid",
            "uname",
        },
        label=label,
    )
    entry_type = _require_str(record["entry_type"], label=f"{label} type")
    if entry_type not in {"directory", "regular_file"}:
        _fail(f"{label} type differs")
    path = _require_str(record["path"], label=f"{label} path")
    if _canonical_archive_path(path, entry_type=entry_type, label=label) != path:
        _fail(f"{label} path is not in canonical recorded form")
    mode = _require_str(record["mode"], label=f"{label} mode")
    if _MODE_RE.fullmatch(mode) is None:
        _fail(f"{label} mode differs")
    for field in ("gid", "mtime", "uid"):
        _require_int(record[field], label=f"{label} {field}")
    _require_int(
        record["size_bytes"],
        label=f"{label} size_bytes",
        maximum=_MAX_REGULAR_FILE_BYTES if entry_type == "regular_file" else 0,
    )
    for field in ("gname", "uname"):
        if type(record[field]) is not str:
            _fail(f"{label} {field} must be an exact string")
        if any(ord(character) < 0x20 or ord(character) > 0x7E for character in record[field]):
            _fail(f"{label} {field} must be printable ASCII")
    _require_sha256(record["header_sha256"], label=f"{label} header")
    _require_sha256(record["sha256"], label=f"{label} content")
    if entry_type == "directory" and (
        record["size_bytes"] != 0 or record["sha256"] != _sha256(b"")
    ):
        _fail(f"{label} directory payload identity differs")
    return record


def _validate_archive_inventory(
    inventory: object,
    *,
    spec: _ArchiveSpec,
    label: str,
) -> dict[str, Any]:
    record = _exact_keys(
        inventory,
        {
            "archive_id",
            "compression",
            "directory_count",
            "entries",
            "entry_count",
            "gzip_member_count",
            "raw_archive_sha256",
            "raw_archive_size_bytes",
            "raw_member_count",
            "regular_file_count",
            "root_directory_entry_present",
            "tar_format",
            "tar_terminal_zero_block_count",
            "tar_uncompressed_size_bytes",
            "top_level_prefix",
            "total_regular_file_bytes",
            "tree_sha256",
        },
        label=label,
    )
    if (
        record["archive_id"] != spec.archive_id
        or record["compression"] != "gzip_single_member"
        or record["gzip_member_count"] != 1
        or record["raw_archive_sha256"] != spec.sha256
        or record["raw_archive_size_bytes"] != spec.size_bytes
        or record["tar_format"] != "bounded_ustar"
        or record["top_level_prefix"] != spec.top_level_prefix
    ):
        _fail(f"{label} fixed archive identity differs")
    _require_bool(record["root_directory_entry_present"], label=f"{label} root directory state")
    raw_members = _require_int(
        record["raw_member_count"],
        label=f"{label} raw members",
        maximum=_MAX_ARCHIVE_MEMBERS,
    )
    entry_count = _require_int(record["entry_count"], label=f"{label} entry count")
    regular_count = _require_int(
        record["regular_file_count"], label=f"{label} regular-file count", minimum=1
    )
    directory_count = _require_int(record["directory_count"], label=f"{label} directory count")
    total_size = _require_int(
        record["total_regular_file_bytes"],
        label=f"{label} total regular-file bytes",
        maximum=_MAX_TOTAL_REGULAR_FILE_BYTES,
    )
    tar_size = _require_int(
        record["tar_uncompressed_size_bytes"],
        label=f"{label} tar size",
        minimum=_TAR_BLOCK_BYTES * 2,
        maximum=_MAX_UNCOMPRESSED_TAR_BYTES,
    )
    terminal_blocks = _require_int(
        record["tar_terminal_zero_block_count"],
        label=f"{label} terminal zero blocks",
        minimum=2,
    )
    if tar_size % _TAR_BLOCK_BYTES != 0 or terminal_blocks * 512 > tar_size:
        _fail(f"{label} tar bounds differ")
    entries_raw = record["entries"]
    if type(entries_raw) is not list or len(entries_raw) > _MAX_ARCHIVE_MEMBERS:
        _fail(f"{label} entries violate their count bound")
    entries = [
        _validate_entry(entry, label=f"{label} entry {index}")
        for index, entry in enumerate(cast(list[object], entries_raw))
    ]
    _validate_inventory_paths(entries, label=label)
    if (
        entry_count != len(entries)
        or regular_count != sum(entry["entry_type"] == "regular_file" for entry in entries)
        or directory_count != sum(entry["entry_type"] == "directory" for entry in entries)
        or total_size
        != sum(
            cast(int, entry["size_bytes"])
            for entry in entries
            if entry["entry_type"] == "regular_file"
        )
        or raw_members != entry_count + (1 if record["root_directory_entry_present"] is True else 0)
    ):
        _fail(f"{label} inventory counts differ")
    expected_tree = _entry_tree_sha256(entries)
    if record["tree_sha256"] != expected_tree:
        _fail(f"{label} tree digest differs")
    return record


def _validate_denials(value: object, *, label: str) -> None:
    authority = _exact_keys(value, set(_denied_authority()), label=label)
    if authority != _denied_authority():
        _fail(f"{label} must deny every authority")


def _parse_manifest(
    raw: bytes,
    *,
    expected_manifest_sha256: str,
    specs: Sequence[_ArchiveSpec],
    pins: Sequence[_RelevantFilePin],
    plan_bytes: bytes,
) -> dict[str, Any]:
    expected = _require_sha256(expected_manifest_sha256, label="expected manifest")
    if not hmac.compare_digest(_sha256(raw), expected):
        _fail("Quicknet materialization manifest full-file digest differs")
    value = _strict_json_load(raw, label="Quicknet materialization manifest")
    _exact_keys(
        value,
        {
            "archive_inventories",
            "authority",
            "cross_archive_comparison",
            "limitations",
            "manifest_body_sha256",
            "plan",
            "relevant_file_pin_checks",
            "schema_version",
            "source_registry",
            "state",
            "status",
        },
        label="Quicknet materialization manifest",
    )
    if (
        value["schema_version"]
        != MATCHED_V3_QUICKNET_SOURCE_MATERIALIZATION_MANIFEST_SCHEMA_VERSION
        or value["status"] != "caller_bytes_safely_inventoried_in_memory_nonauthorizing"
        or value["source_registry"] != _source_registry_identity()
        or value["plan"]
        != {
            "schema_version": MATCHED_V3_QUICKNET_SOURCE_MATERIALIZATION_PLAN_SCHEMA_VERSION,
            "sha256": _sha256(plan_bytes),
        }
        or value["state"] != _manifest_state()
    ):
        _fail("Quicknet materialization manifest fixed metadata differs")
    _validate_denials(value["authority"], label="manifest authority")
    _verify_body_digest(value, digest_field="manifest_body_sha256", label="manifest")
    inventories_raw = value["archive_inventories"]
    if type(inventories_raw) is not list or len(inventories_raw) != len(specs):
        _fail("Quicknet materialization manifest archive count differs")
    inventories = [
        _validate_archive_inventory(
            inventory,
            spec=spec,
            label=f"manifest archive {index}",
        )
        for index, (inventory, spec) in enumerate(
            zip(cast(list[object], inventories_raw), specs, strict=True)
        )
    ]
    if value["relevant_file_pin_checks"] != _pin_checks(inventories, pins):
        _fail("Quicknet materialization manifest relevant-file checks differ")
    if value["cross_archive_comparison"] != _cross_archive_comparison(
        inventories[0], inventories[1]
    ):
        _fail("Quicknet materialization manifest cross-archive comparison differs")
    if value["limitations"] != _manifest_limitations():
        _fail("Quicknet materialization manifest limitations differ")
    return value


def parse_matched_v3_quicknet_source_materialization_manifest(
    raw: bytes,
    *,
    expected_manifest_sha256: str,
) -> dict[str, Any]:
    """Replay a production manifest under exact plan, archive, and full-file pins."""

    return _parse_manifest(
        raw,
        expected_manifest_sha256=expected_manifest_sha256,
        specs=_PRODUCTION_ARCHIVE_SPECS,
        pins=_PRODUCTION_RELEVANT_PINS,
        plan_bytes=_PLAN_BYTES,
    )


def _receipt_state() -> dict[str, bool]:
    return {
        "source_archive_inventory_receipt_issued": True,
        "filesystem_materialization_receipt_issued": False,
        "build_source_selection_receipt_issued": False,
        "dependency_vendor_closure_receipt_issued": False,
        "rust_build_receipt_issued": False,
        "verifier_receipt_issued": False,
        "trust_root_receipt_issued": False,
        "chronology_receipt_issued": False,
        "seed_receipt_issued": False,
        "qualification_receipt_issued": False,
    }


def _receipt_limitations() -> list[str]:
    return [
        "This receipt binds an in-memory inventory and does not attest path extraction.",
        "The caller-supplied producer source digest is bound but not an authority decision.",
        "No primary build source or dependency closure has been selected or produced.",
        "This receipt cannot substitute for build, verifier, trust, chronology, or seed receipts.",
        "This receipt grants no qualification, evidence, promotion, or performance authority.",
    ]


def _build_receipt(
    manifest_bytes: bytes,
    *,
    producer_source_bytes: bytes,
    specs: Sequence[_ArchiveSpec],
    pins: Sequence[_RelevantFilePin],
    plan_bytes: bytes,
) -> bytes:
    if (
        type(producer_source_bytes) is not bytes
        or not producer_source_bytes
        or len(producer_source_bytes) > _MAX_PRODUCER_SOURCE_BYTES
    ):
        _fail("materializer producer source must be supplied as bounded nonempty exact bytes")
    manifest_sha256 = _sha256(manifest_bytes)
    manifest = _parse_manifest(
        manifest_bytes,
        expected_manifest_sha256=manifest_sha256,
        specs=specs,
        pins=pins,
        plan_bytes=plan_bytes,
    )
    inventories = cast(list[dict[str, Any]], manifest["archive_inventories"])
    body: dict[str, Any] = {
        "schema_version": MATCHED_V3_QUICKNET_SOURCE_MATERIALIZATION_RECEIPT_SCHEMA_VERSION,
        "status": "in_memory_archive_inventory_receipt_nonauthorizing",
        "producer": {
            "descriptor_schema_version": (
                MATCHED_V3_QUICKNET_SOURCE_MATERIALIZATION_DESCRIPTOR_SCHEMA_VERSION
            ),
            "descriptor_sha256": (MATCHED_V3_QUICKNET_SOURCE_MATERIALIZATION_DESCRIPTOR_SHA256),
            "source_sha256": _sha256(producer_source_bytes),
            "source_size_bytes": len(producer_source_bytes),
            "source_bytes_supplied_by_caller": True,
            "source_identity_authorized_here": False,
        },
        "plan": {
            "schema_version": MATCHED_V3_QUICKNET_SOURCE_MATERIALIZATION_PLAN_SCHEMA_VERSION,
            "sha256": _sha256(plan_bytes),
            "body_sha256": _strict_json_load(plan_bytes, label="Quicknet materialization plan")[
                "plan_body_sha256"
            ],
        },
        "manifest": {
            "schema_version": MATCHED_V3_QUICKNET_SOURCE_MATERIALIZATION_MANIFEST_SCHEMA_VERSION,
            "sha256": manifest_sha256,
            "body_sha256": manifest["manifest_body_sha256"],
        },
        "archive_tree_bindings": [
            {
                "archive_id": inventory["archive_id"],
                "raw_archive_sha256": inventory["raw_archive_sha256"],
                "raw_archive_size_bytes": inventory["raw_archive_size_bytes"],
                "tree_sha256": inventory["tree_sha256"],
            }
            for inventory in inventories
        ],
        "state": _receipt_state(),
        "authority": _denied_authority(),
        "limitations": _receipt_limitations(),
    }
    return _canonical_json_bytes(_body_bound_artifact(body, digest_field="receipt_body_sha256"))


def _parse_receipt(
    raw: bytes,
    *,
    expected_receipt_sha256: str,
    manifest_bytes: bytes,
    expected_manifest_sha256: str,
    specs: Sequence[_ArchiveSpec],
    pins: Sequence[_RelevantFilePin],
    plan_bytes: bytes,
) -> dict[str, Any]:
    expected = _require_sha256(expected_receipt_sha256, label="expected receipt")
    if not hmac.compare_digest(_sha256(raw), expected):
        _fail("Quicknet materialization receipt full-file digest differs")
    manifest = _parse_manifest(
        manifest_bytes,
        expected_manifest_sha256=expected_manifest_sha256,
        specs=specs,
        pins=pins,
        plan_bytes=plan_bytes,
    )
    value = _strict_json_load(raw, label="Quicknet materialization receipt")
    _exact_keys(
        value,
        {
            "archive_tree_bindings",
            "authority",
            "limitations",
            "manifest",
            "plan",
            "producer",
            "receipt_body_sha256",
            "schema_version",
            "state",
            "status",
        },
        label="Quicknet materialization receipt",
    )
    if (
        value["schema_version"] != MATCHED_V3_QUICKNET_SOURCE_MATERIALIZATION_RECEIPT_SCHEMA_VERSION
        or value["status"] != "in_memory_archive_inventory_receipt_nonauthorizing"
        or value["state"] != _receipt_state()
    ):
        _fail("Quicknet materialization receipt fixed metadata differs")
    _validate_denials(value["authority"], label="receipt authority")
    _verify_body_digest(value, digest_field="receipt_body_sha256", label="receipt")
    plan = _exact_keys(
        value["plan"], {"body_sha256", "schema_version", "sha256"}, label="receipt plan"
    )
    parsed_plan = _strict_json_load(plan_bytes, label="Quicknet materialization plan")
    if plan != {
        "schema_version": MATCHED_V3_QUICKNET_SOURCE_MATERIALIZATION_PLAN_SCHEMA_VERSION,
        "sha256": _sha256(plan_bytes),
        "body_sha256": parsed_plan["plan_body_sha256"],
    }:
        _fail("Quicknet materialization receipt plan binding differs")
    manifest_binding = _exact_keys(
        value["manifest"],
        {"body_sha256", "schema_version", "sha256"},
        label="receipt manifest",
    )
    if manifest_binding != {
        "schema_version": MATCHED_V3_QUICKNET_SOURCE_MATERIALIZATION_MANIFEST_SCHEMA_VERSION,
        "sha256": expected_manifest_sha256,
        "body_sha256": manifest["manifest_body_sha256"],
    }:
        _fail("Quicknet materialization receipt manifest binding differs")
    producer = _exact_keys(
        value["producer"],
        {
            "descriptor_schema_version",
            "descriptor_sha256",
            "source_bytes_supplied_by_caller",
            "source_identity_authorized_here",
            "source_sha256",
            "source_size_bytes",
        },
        label="receipt producer",
    )
    if (
        producer["descriptor_schema_version"]
        != MATCHED_V3_QUICKNET_SOURCE_MATERIALIZATION_DESCRIPTOR_SCHEMA_VERSION
        or producer["descriptor_sha256"]
        != MATCHED_V3_QUICKNET_SOURCE_MATERIALIZATION_DESCRIPTOR_SHA256
        or producer["source_bytes_supplied_by_caller"] is not True
        or producer["source_identity_authorized_here"] is not False
    ):
        _fail("Quicknet materialization receipt producer binding differs")
    _require_sha256(producer["source_sha256"], label="receipt producer source")
    _require_int(
        producer["source_size_bytes"],
        label="receipt producer source size",
        minimum=1,
        maximum=_MAX_PRODUCER_SOURCE_BYTES,
    )
    inventories = cast(list[dict[str, Any]], manifest["archive_inventories"])
    expected_tree_bindings = [
        {
            "archive_id": inventory["archive_id"],
            "raw_archive_sha256": inventory["raw_archive_sha256"],
            "raw_archive_size_bytes": inventory["raw_archive_size_bytes"],
            "tree_sha256": inventory["tree_sha256"],
        }
        for inventory in inventories
    ]
    if value["archive_tree_bindings"] != expected_tree_bindings:
        _fail("Quicknet materialization receipt archive-tree binding differs")
    if value["limitations"] != _receipt_limitations():
        _fail("Quicknet materialization receipt limitations differ")
    return value


def parse_matched_v3_quicknet_source_materialization_receipt(
    raw: bytes,
    *,
    expected_receipt_sha256: str,
    manifest_bytes: bytes,
    expected_manifest_sha256: str,
) -> dict[str, Any]:
    """Replay a receipt with its exact production manifest and plan cross-links."""

    return _parse_receipt(
        raw,
        expected_receipt_sha256=expected_receipt_sha256,
        manifest_bytes=manifest_bytes,
        expected_manifest_sha256=expected_manifest_sha256,
        specs=_PRODUCTION_ARCHIVE_SPECS,
        pins=_PRODUCTION_RELEVANT_PINS,
        plan_bytes=_PLAN_BYTES,
    )


def inventory_matched_v3_quicknet_source_archives(
    *,
    commit_archive_bytes: bytes,
    crate_archive_bytes: bytes,
    producer_source_bytes: bytes,
) -> MatchedV3QuicknetSourceArchiveInventory:
    """Inventory the two exact pinned archives without extraction or authority.

    Every byte is caller-supplied.  The function performs bounded in-memory
    gzip and USTAR parsing only and returns detached canonical JSON artifacts.
    It has no path, URL, process, clock, environment, randomness, or Rust input.
    """

    inventories = [
        _inventory_gzip_tar_archive(commit_archive_bytes, spec=_PRODUCTION_ARCHIVE_SPECS[0]),
        _inventory_gzip_tar_archive(crate_archive_bytes, spec=_PRODUCTION_ARCHIVE_SPECS[1]),
    ]
    manifest_bytes = _build_manifest(
        inventories,
        pins=_PRODUCTION_RELEVANT_PINS,
        plan_bytes=_PLAN_BYTES,
    )
    manifest_sha256 = _sha256(manifest_bytes)
    _parse_manifest(
        manifest_bytes,
        expected_manifest_sha256=manifest_sha256,
        specs=_PRODUCTION_ARCHIVE_SPECS,
        pins=_PRODUCTION_RELEVANT_PINS,
        plan_bytes=_PLAN_BYTES,
    )
    receipt_bytes = _build_receipt(
        manifest_bytes,
        producer_source_bytes=producer_source_bytes,
        specs=_PRODUCTION_ARCHIVE_SPECS,
        pins=_PRODUCTION_RELEVANT_PINS,
        plan_bytes=_PLAN_BYTES,
    )
    receipt_sha256 = _sha256(receipt_bytes)
    _parse_receipt(
        receipt_bytes,
        expected_receipt_sha256=receipt_sha256,
        manifest_bytes=manifest_bytes,
        expected_manifest_sha256=manifest_sha256,
        specs=_PRODUCTION_ARCHIVE_SPECS,
        pins=_PRODUCTION_RELEVANT_PINS,
        plan_bytes=_PLAN_BYTES,
    )
    return MatchedV3QuicknetSourceArchiveInventory(
        manifest_bytes=manifest_bytes,
        manifest_sha256=manifest_sha256,
        receipt_bytes=receipt_bytes,
        receipt_sha256=receipt_sha256,
    )


if not hmac.compare_digest(
    _sha256(_DESCRIPTOR_BYTES),
    MATCHED_V3_QUICKNET_SOURCE_MATERIALIZATION_DESCRIPTOR_SHA256,
):
    raise AssertionError("matched-v3 Quicknet source-materialization descriptor drifted")


__all__ = [
    "ForagerMatchedV3QuicknetSourceMaterializationError",
    "MATCHED_V3_QUICKNET_SOURCE_MATERIALIZATION_DESCRIPTOR_SCHEMA_VERSION",
    "MATCHED_V3_QUICKNET_SOURCE_MATERIALIZATION_DESCRIPTOR_SHA256",
    "MATCHED_V3_QUICKNET_SOURCE_MATERIALIZATION_MANIFEST_SCHEMA_VERSION",
    "MATCHED_V3_QUICKNET_SOURCE_MATERIALIZATION_PLAN_SCHEMA_VERSION",
    "MATCHED_V3_QUICKNET_SOURCE_MATERIALIZATION_PLAN_SHA256",
    "MATCHED_V3_QUICKNET_SOURCE_MATERIALIZATION_RECEIPT_SCHEMA_VERSION",
    "MATCHED_V3_QUICKNET_SOURCE_MATERIALIZATION_STATUS",
    "MATCHED_V3_QUICKNET_VERIFIER_SOURCE_DESCRIPTOR_SHA256",
    "MATCHED_V3_QUICKNET_VERIFIER_SOURCE_MODULE_SHA256",
    "MatchedV3QuicknetSourceArchiveInventory",
    "UPSTREAM_COMMIT_ARCHIVE_SHA256",
    "UPSTREAM_COMMIT_ARCHIVE_SIZE_BYTES",
    "UPSTREAM_CRATE_ARCHIVE_SHA256",
    "UPSTREAM_CRATE_ARCHIVE_SIZE_BYTES",
    "canonical_matched_v3_quicknet_source_materialization_descriptor_bytes",
    "canonical_matched_v3_quicknet_source_materialization_plan_bytes",
    "inventory_matched_v3_quicknet_source_archives",
    "matched_v3_quicknet_source_materialization_descriptor",
    "matched_v3_quicknet_source_materialization_plan",
    "parse_matched_v3_quicknet_source_materialization_descriptor",
    "parse_matched_v3_quicknet_source_materialization_manifest",
    "parse_matched_v3_quicknet_source_materialization_plan",
    "parse_matched_v3_quicknet_source_materialization_receipt",
]
