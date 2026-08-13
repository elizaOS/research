"""Canonical retained local-source USTAR for matched-v3 CPU image assembly.

This pure-stdlib module converts one caller-pinned local source snapshot into a
sealed anonymous POSIX USTAR content capability.  The archive contains exactly
the regular-file records in the snapshot manifest: it contains no synthetic
manifest, directory entry, cache file, extraction instruction, or execution
token.  A separate canonical receipt binds the complete archive identity and
member inventory to the source-manifest and tree identities.

The producer validates the expected artifact before observing the filesystem,
opens the repository through no-follow directory descriptors, performs complete
pre- and post-archive measurements, streams every archive member through a
checked single-link regular-file descriptor, replays the raw USTAR, and seals it
before returning.  This is a nonauthorizing content boundary.  It does not build
an image, execute source, qualify a runtime, publish an artifact, or create
scientific evidence.
"""

from __future__ import annotations

import fcntl
import hashlib
import hmac
import json
import os
import re
import stat
from collections.abc import Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Never, NoReturn, SupportsIndex, cast

LOCAL_SOURCE_BUNDLE_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.local_source_bundle_descriptor.v1"
)
LOCAL_SOURCE_BUNDLE_RECEIPT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.local_source_bundle_receipt.v1"
)
LOCAL_SOURCE_BUNDLE_STATUS: Final = (
    "implemented_retained_payload_unexecuted_unqualified_non_authorizing"
)

_SNAPSHOT_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.local_source_snapshot_descriptor.v1"
)
_SNAPSHOT_DESCRIPTOR_SHA256: Final = (
    "5ba69445a00dfc0bc36a4d05dafcc534b291430d491c3f71560570d7eb862899"
)
_SNAPSHOT_MANIFEST_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.local_source_snapshot_manifest.v1"
)
_SNAPSHOT_TREE_SCHEMA_VERSION: Final = "alberta.forager_matched_v3.local_source_snapshot_tree.v1"

_ROOT_FILE_NAMES: Final = ("pyproject.toml", "uv.lock", "FORAGER_BENCHMARK.md")
_FRAMEWORK_DIRECTORY_NAME: Final = "alberta_framework"
_CACHE_DIRECTORY_NAMES: Final = frozenset(
    {".mypy_cache", ".pytest_cache", ".ruff_cache", "__pycache__"}
)
_CACHE_FILE_SUFFIXES: Final = (".pyc", ".pyo")

_MAX_FILES: Final = 20_000
_MAX_DIRECTORIES: Final = 10_000
_MAX_ENTRIES: Final = 50_000
_MAX_DEPTH: Final = 64
_MAX_FILE_BYTES: Final = 128 * 1024 * 1024
_MAX_TOTAL_BYTES: Final = 2 * 1024 * 1024 * 1024
_MAX_SNAPSHOT_MANIFEST_BYTES: Final = 16 * 1024 * 1024
_MAX_RECEIPT_BYTES: Final = 32 * 1024 * 1024
_MAX_JSON_DEPTH: Final = 64
_MAX_JSON_NODES: Final = 300_000
_MAX_JSON_STRING_BYTES: Final = 2_048
_MAX_PATH_BYTES: Final = 1_024
_MAX_COMPONENT_BYTES: Final = 255
_READ_CHUNK_BYTES: Final = 1024 * 1024
_USTAR_BLOCK_BYTES: Final = 512
_USTAR_RECORD_BYTES: Final = 10 * 1024
_MAX_ARCHIVE_BYTES: Final = (
    _MAX_TOTAL_BYTES + (_MAX_FILES * 2 * _USTAR_BLOCK_BYTES) + _USTAR_RECORD_BYTES
)

_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_ASCII_COMPONENT_RE: Final = re.compile(r"[A-Za-z0-9_.+-]{1,255}\Z")


class ForagerMatchedV3LocalSourceBundleError(RuntimeError):
    """The expected snapshot, source capability, or canonical USTAR failed closed."""


def _claims() -> dict[str, bool]:
    return {
        "execution_authority_granted": False,
        "execution_linkage_established": False,
        "executed_bytecode_attested": False,
        "filesystem_publication_authority_granted": False,
        "image_build_authority_granted": False,
        "import_behavior_attested": False,
        "performance_claim_allowed": False,
        "publication_authority_granted": False,
        "qualification_granted": False,
        "runtime_qualified": False,
        "scientific_evidence_created": False,
        "scientific_promotion_allowed": False,
        "source_bundle_qualified": False,
        "source_snapshot_qualified": False,
        "universal_sota_claim_allowed": False,
    }


def _snapshot_claims() -> dict[str, bool]:
    return {
        "execution_authority_granted": False,
        "execution_linkage_established": False,
        "executed_bytecode_attested": False,
        "import_behavior_attested": False,
        "publication_authority_granted": False,
        "qualification_granted": False,
        "runtime_qualified": False,
        "source_snapshot_qualified": False,
        "scientific_evidence_created": False,
        "scientific_promotion_allowed": False,
        "performance_claim_allowed": False,
        "universal_sota_claim_allowed": False,
    }


def _snapshot_limitations() -> list[str]:
    return [
        "A measured manifest describes caller-supplied local source only.",
        "No current-repository manifest or source identity is embedded here.",
        (
            "Excluded cache subtrees and pyc/pyo files are not bound; this snapshot "
            "does not attest executed bytecode or import behavior and cannot establish "
            "execution linkage alone."
        ),
        (
            "A standalone execution bootstrap must separately neutralize bytecode "
            "caches before linking execution to this source snapshot."
        ),
        "A manifest grants no execution, publication, qualification, or promotion authority.",
        "Runtime, dependency, toolchain, and hardware closure remain external.",
    ]


def _limitations() -> list[str]:
    return [
        "The retained USTAR is a content capability, not execution or image-build authority.",
        (
            "The receipt is out of band and is not an archive member; the archive contains "
            "exactly the source snapshot regular-file inventory."
        ),
        (
            "Snapshot manifests do not bind source filesystem modes; every archive member is "
            "therefore normalized to read-only mode 0444."
        ),
        (
            "Excluded cache subtrees and pyc/pyo files remain outside the bound payload and "
            "cannot attest import behavior or executed bytecode."
        ),
        (
            "Identical pre/post observations do not prove continuous source immutability "
            "between observation endpoints."
        ),
        "Dependency wheels, runtime lock, base image, toolchain, and hardware remain external.",
        "No descriptor, receipt, digest, or retained file descriptor grants publication.",
        "No bundle result grants qualification, promotion, performance, or SOTA claims.",
    ]


def _descriptor() -> dict[str, Any]:
    return {
        "schema_version": LOCAL_SOURCE_BUNDLE_DESCRIPTOR_SCHEMA_VERSION,
        "status": LOCAL_SOURCE_BUNDLE_STATUS,
        "classification": "retained_local_source_payload_non_authorizing",
        "source_snapshot": {
            "descriptor_schema_version": _SNAPSHOT_DESCRIPTOR_SCHEMA_VERSION,
            "descriptor_sha256": _SNAPSHOT_DESCRIPTOR_SHA256,
            "manifest_schema_version": _SNAPSHOT_MANIFEST_SCHEMA_VERSION,
            "tree_schema_version": _SNAPSHOT_TREE_SCHEMA_VERSION,
            "caller_carried_manifest_bytes_required": True,
            "caller_carried_manifest_sha256_required": True,
            "caller_carried_tree_sha256_required": True,
            "strict_independent_revalidation_before_filesystem_observation": True,
            "manifest_or_tree_identity_is_payload": False,
        },
        "source_read": {
            "repository_root_caller_supplied": True,
            "repository_root_default": False,
            "directory_descriptor_anchored": True,
            "relative_nofollow_reads": True,
            "single_link_regular_files_only": True,
            "pre_open_stat_open_fstat_post_read_fstat_restat_required": True,
            "complete_pre_archive_measurement_required": True,
            "complete_post_archive_measurement_required": True,
            "exact_manifest_inventory_match_required": True,
            "mutation_and_capability_drift_rejected": True,
        },
        "archive": {
            "format": "canonical_posix_ustar_uncompressed",
            "members": "exact_snapshot_regular_file_inventory_only",
            "synthetic_members": False,
            "directory_members": False,
            "path_encoding": "ASCII_POSIX_USTAR_name_prefix_exact",
            "member_order": "ascending_ascii_path_bytes",
            "member_mode": "0444",
            "uid_gid_mtime": 0,
            "uname_gname": "empty",
            "typeflag": "0",
            "payload_padding": "zero_to_512_byte_block",
            "end_blocks": 2,
            "record_size_bytes": _USTAR_RECORD_BYTES,
            "record_padding": "zero_to_10240_byte_record",
            "raw_replay_before_and_after_sealing": True,
        },
        "receipt": {
            "schema_version": LOCAL_SOURCE_BUNDLE_RECEIPT_SCHEMA_VERSION,
            "storage": "out_of_band_canonical_json",
            "binds_archive_sha256_size_and_complete_member_inventory": True,
            "binds_source_manifest_and_tree_sha256": True,
            "body_digest": "sha256_of_canonical_receipt_without_body_digest",
            "full_digest": "sha256_of_canonical_receipt_bytes",
        },
        "retention": {
            "sealed_anonymous_read_only_descriptor": True,
            "pid_bound": True,
            "copyable": False,
            "serializable": False,
            "descriptor_reverified_on_every_access": True,
            "bounded_exact_archive_byte_read": True,
            "extraction_api": False,
            "publication_api": False,
            "execution_api": False,
        },
        "limits": {
            "maximum_files": _MAX_FILES,
            "maximum_directories": _MAX_DIRECTORIES,
            "maximum_entries": _MAX_ENTRIES,
            "maximum_depth": _MAX_DEPTH,
            "maximum_file_bytes": _MAX_FILE_BYTES,
            "maximum_total_source_bytes": _MAX_TOTAL_BYTES,
            "maximum_archive_bytes": _MAX_ARCHIVE_BYTES,
            "maximum_receipt_bytes": _MAX_RECEIPT_BYTES,
        },
        "claims": _claims(),
        "limitations": _limitations(),
    }


def _raise_json_constant(value: str) -> NoReturn:
    raise ForagerMatchedV3LocalSourceBundleError(
        f"local source bundle JSON contains non-finite constant {value!r}"
    )


def _parse_bounded_int(value: str) -> int:
    if len(value.lstrip("-")) > 20:
        raise ForagerMatchedV3LocalSourceBundleError(
            "local source bundle JSON integer exceeds its lexical bound"
        )
    return int(value)


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ForagerMatchedV3LocalSourceBundleError(
                f"local source bundle JSON contains duplicate key {key!r}"
            )
        result[key] = value
    return result


def _assert_plain_unaliased_json(value: Any) -> None:
    seen: set[int] = set()
    nodes = 0

    def visit(item: Any, depth: int) -> None:
        nonlocal nodes
        nodes += 1
        if nodes > _MAX_JSON_NODES:
            raise ForagerMatchedV3LocalSourceBundleError(
                "local source bundle JSON exceeds its node bound"
            )
        if depth > _MAX_JSON_DEPTH:
            raise ForagerMatchedV3LocalSourceBundleError(
                "local source bundle JSON exceeds its depth bound"
            )
        if type(item) is str:
            try:
                encoded = item.encode("ascii")
            except UnicodeEncodeError as exc:
                raise ForagerMatchedV3LocalSourceBundleError(
                    "local source bundle JSON strings must be ASCII"
                ) from exc
            if len(encoded) > _MAX_JSON_STRING_BYTES or any(
                byte < 0x20 or byte > 0x7E for byte in encoded
            ):
                raise ForagerMatchedV3LocalSourceBundleError(
                    "local source bundle JSON strings must be bounded printable ASCII"
                )
            return
        if item is None or type(item) in {bool, int}:
            return
        if type(item) not in {dict, list}:
            raise ForagerMatchedV3LocalSourceBundleError(
                "local source bundle JSON contains a non-plain value"
            )
        identity = id(item)
        if identity in seen:
            raise ForagerMatchedV3LocalSourceBundleError(
                "local source bundle JSON contains a container alias"
            )
        seen.add(identity)
        if type(item) is list:
            for child in item:
                visit(child, depth + 1)
        else:
            for key, child in cast(dict[Any, Any], item).items():
                if type(key) is not str:
                    raise ForagerMatchedV3LocalSourceBundleError(
                        "local source bundle JSON object keys must be strings"
                    )
                visit(key, depth + 1)
                visit(child, depth + 1)

    visit(value, 0)


def _canonical_json(value: dict[str, Any], *, maximum_bytes: int) -> bytes:
    if type(value) is not dict:
        raise ForagerMatchedV3LocalSourceBundleError("canonical JSON root must be an object")
    _assert_plain_unaliased_json(value)
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
        raise ForagerMatchedV3LocalSourceBundleError(
            "local source bundle JSON is not finite canonical ASCII"
        ) from exc
    if not 0 < len(raw) <= maximum_bytes:
        raise ForagerMatchedV3LocalSourceBundleError(
            "local source bundle JSON exceeds its byte bound"
        )
    return raw


def _strict_json_load(raw: bytes, *, maximum_bytes: int) -> dict[str, Any]:
    if type(raw) is not bytes or not raw or len(raw) > maximum_bytes:
        raise ForagerMatchedV3LocalSourceBundleError("JSON artifact is not bounded exact bytes")
    if not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
        raise ForagerMatchedV3LocalSourceBundleError(
            "JSON artifact must have exactly one trailing newline"
        )
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ForagerMatchedV3LocalSourceBundleError("JSON artifact must be ASCII") from exc
    try:
        value = json.loads(
            text,
            object_pairs_hook=_object_without_duplicates,
            parse_constant=_raise_json_constant,
            parse_int=_parse_bounded_int,
        )
    except ForagerMatchedV3LocalSourceBundleError:
        raise
    except (RecursionError, ValueError, json.JSONDecodeError) as exc:
        raise ForagerMatchedV3LocalSourceBundleError("JSON artifact is not strict JSON") from exc
    if type(value) is not dict:
        raise ForagerMatchedV3LocalSourceBundleError("JSON artifact root must be an object")
    result = cast(dict[str, Any], value)
    _assert_plain_unaliased_json(result)
    if not hmac.compare_digest(_canonical_json(result, maximum_bytes=maximum_bytes), raw):
        raise ForagerMatchedV3LocalSourceBundleError("JSON artifact is not exactly canonical")
    return result


def _exact_json_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if left is None or type(left) in {bool, int, str}:
        return bool(left == right)
    if type(left) is list:
        exact_left = left
        exact_right = cast(list[Any], right)
        return len(exact_left) == len(exact_right) and all(
            _exact_json_equal(left_item, right_item)
            for left_item, right_item in zip(exact_left, exact_right, strict=True)
        )
    if type(left) is dict:
        exact_left_map = cast(dict[str, Any], left)
        exact_right_map = cast(dict[str, Any], right)
        return exact_left_map.keys() == exact_right_map.keys() and all(
            _exact_json_equal(exact_left_map[key], exact_right_map[key]) for key in exact_left_map
        )
    return False


def _require_exact_keys(value: Any, keys: frozenset[str], label: str) -> dict[str, Any]:
    if type(value) is not dict or frozenset(value) != keys:
        raise ForagerMatchedV3LocalSourceBundleError(f"{label} keys are not exact")
    return cast(dict[str, Any], value)


def _require_sha256(value: Any, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None or value == "0" * 64:
        raise ForagerMatchedV3LocalSourceBundleError(
            f"{label} must be one nonzero lowercase SHA-256"
        )
    return value


def _require_int(value: Any, label: str, *, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ForagerMatchedV3LocalSourceBundleError(
            f"{label} must be an exact integer in [{minimum}, {maximum}]"
        )
    return value


_DESCRIPTOR_BYTES: Final = _canonical_json(_descriptor(), maximum_bytes=_MAX_RECEIPT_BYTES)
LOCAL_SOURCE_BUNDLE_DESCRIPTOR_SHA256: Final = (
    "52a48f3258aff9c7f2e80033187b85dd2924dc843d991ba7c2bac829f10c5e89"
)
if not hmac.compare_digest(
    hashlib.sha256(_DESCRIPTOR_BYTES).hexdigest(),
    LOCAL_SOURCE_BUNDLE_DESCRIPTOR_SHA256,
):
    raise AssertionError("matched-v3 local source bundle descriptor identity drifted")


def _validate_component(component: Any, label: str) -> str:
    if (
        type(component) is not str
        or component in {"", ".", ".."}
        or "/" in component
        or "\\" in component
        or "\x00" in component
        or _ASCII_COMPONENT_RE.fullmatch(component) is None
    ):
        raise ForagerMatchedV3LocalSourceBundleError(
            f"{label} is not one unambiguous ASCII path component"
        )
    if len(component.encode("ascii")) > _MAX_COMPONENT_BYTES:
        raise ForagerMatchedV3LocalSourceBundleError(f"{label} exceeds its component bound")
    return component


def _validate_relative_path(path: Any, label: str) -> str:
    if type(path) is not str:
        raise ForagerMatchedV3LocalSourceBundleError(f"{label} must be one relative path")
    try:
        encoded = path.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ForagerMatchedV3LocalSourceBundleError(f"{label} must be ASCII") from exc
    if not encoded or len(encoded) > _MAX_PATH_BYTES or path.startswith("/"):
        raise ForagerMatchedV3LocalSourceBundleError(f"{label} is outside its path bounds")
    parts = path.split("/")
    if any(not part for part in parts):
        raise ForagerMatchedV3LocalSourceBundleError(f"{label} has an empty component")
    for part in parts:
        _validate_component(part, label)
    if path in _ROOT_FILE_NAMES:
        return path
    if len(parts) < 2 or parts[0] != _FRAMEWORK_DIRECTORY_NAME:
        raise ForagerMatchedV3LocalSourceBundleError(
            f"{label} is outside the frozen source inventory roots"
        )
    if len(parts) - 1 > _MAX_DEPTH:
        raise ForagerMatchedV3LocalSourceBundleError(f"{label} exceeds the depth bound")
    if any(part in _CACHE_DIRECTORY_NAMES for part in parts[:-1]) or path.endswith(
        _CACHE_FILE_SUFFIXES
    ):
        raise ForagerMatchedV3LocalSourceBundleError(f"{label} names excluded cache material")
    _split_ustar_path(path)
    return path


def _validate_relative_directory(path: Any, label: str) -> str:
    if type(path) is not str:
        raise ForagerMatchedV3LocalSourceBundleError(f"{label} must be one directory path")
    try:
        encoded = path.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ForagerMatchedV3LocalSourceBundleError(f"{label} must be ASCII") from exc
    if not encoded or len(encoded) > _MAX_PATH_BYTES or path.startswith("/"):
        raise ForagerMatchedV3LocalSourceBundleError(f"{label} is outside its path bounds")
    parts = path.split("/")
    if any(not part for part in parts):
        raise ForagerMatchedV3LocalSourceBundleError(f"{label} has an empty component")
    for part in parts:
        _validate_component(part, label)
    if parts[0] != _FRAMEWORK_DIRECTORY_NAME or len(parts) > _MAX_DEPTH:
        raise ForagerMatchedV3LocalSourceBundleError(
            f"{label} is outside the recursive directory or depth bound"
        )
    if any(part in _CACHE_DIRECTORY_NAMES for part in parts):
        raise ForagerMatchedV3LocalSourceBundleError(f"{label} names an excluded cache directory")
    return path


def _split_ustar_path(path: str) -> tuple[bytes, bytes]:
    encoded = path.encode("ascii")
    if len(encoded) <= 100:
        return b"", encoded
    for index in reversed([i for i, byte in enumerate(encoded) if byte == ord("/")]):
        prefix = encoded[:index]
        name = encoded[index + 1 :]
        if prefix and name and len(prefix) <= 155 and len(name) <= 100:
            return prefix, name
    raise ForagerMatchedV3LocalSourceBundleError(
        f"source path is not exactly representable in POSIX USTAR: {path}"
    )


@dataclass(frozen=True, slots=True)
class _SnapshotIdentity:
    manifest: dict[str, Any]
    manifest_sha256: str
    tree_sha256: str
    directory_count: int
    file_count: int
    total_size_bytes: int


def _validate_expected_snapshot(
    raw: bytes,
    *,
    expected_manifest_sha256: str,
    expected_tree_sha256: str,
) -> _SnapshotIdentity:
    manifest_sha256 = _require_sha256(expected_manifest_sha256, "expected source snapshot manifest")
    tree_pin = _require_sha256(expected_tree_sha256, "expected source snapshot tree")
    if type(raw) is not bytes or not hmac.compare_digest(
        hashlib.sha256(raw).hexdigest(), manifest_sha256
    ):
        raise ForagerMatchedV3LocalSourceBundleError(
            "source snapshot manifest full-file digest disagrees"
        )
    manifest = _strict_json_load(raw, maximum_bytes=_MAX_SNAPSHOT_MANIFEST_BYTES)
    _require_exact_keys(
        manifest,
        frozenset(
            {
                "schema_version",
                "status",
                "classification",
                "descriptor_binding",
                "observation",
                "inventory",
                "directories",
                "files",
                "tree",
                "claims",
                "limitations",
                "manifest_body_sha256",
            }
        ),
        "source snapshot manifest",
    )
    if (
        manifest["schema_version"] != _SNAPSHOT_MANIFEST_SCHEMA_VERSION
        or manifest["status"] != "measured_unqualified_non_authorizing"
        or manifest["classification"] != "local_source_snapshot_content_only"
    ):
        raise ForagerMatchedV3LocalSourceBundleError("source snapshot identity fields drifted")
    if not _exact_json_equal(
        manifest["descriptor_binding"],
        {
            "schema_version": _SNAPSHOT_DESCRIPTOR_SCHEMA_VERSION,
            "sha256": _SNAPSHOT_DESCRIPTOR_SHA256,
        },
    ):
        raise ForagerMatchedV3LocalSourceBundleError("source snapshot descriptor binding drifted")
    if not _exact_json_equal(
        manifest["observation"],
        {
            "measurement_passes": 2,
            "identical_passes_required": True,
            "repository_path_recorded": False,
        },
    ):
        raise ForagerMatchedV3LocalSourceBundleError("source snapshot observation drifted")

    raw_directories = manifest["directories"]
    if type(raw_directories) is not list or not 1 <= len(raw_directories) <= _MAX_DIRECTORIES:
        raise ForagerMatchedV3LocalSourceBundleError("source snapshot directories are invalid")
    directories: list[str] = []
    directory_aliases: set[str] = set()
    for index, item in enumerate(raw_directories):
        path = _validate_relative_directory(item, f"source snapshot directory {index}")
        alias = path.casefold()
        if alias in directory_aliases:
            raise ForagerMatchedV3LocalSourceBundleError(
                "source snapshot contains duplicate or aliased directories"
            )
        directories.append(path)
        directory_aliases.add(alias)
    if directories != sorted(directories, key=lambda value: value.encode("ascii")):
        raise ForagerMatchedV3LocalSourceBundleError("source snapshot directories are not ordered")
    if _FRAMEWORK_DIRECTORY_NAME not in directories:
        raise ForagerMatchedV3LocalSourceBundleError("source snapshot omits framework root")
    directory_set = set(directories)
    for path in directories:
        if path != _FRAMEWORK_DIRECTORY_NAME and path.rsplit("/", 1)[0] not in directory_set:
            raise ForagerMatchedV3LocalSourceBundleError(
                "source snapshot directory has no bound parent"
            )

    raw_files = manifest["files"]
    if type(raw_files) is not list or not 2 <= len(raw_files) <= _MAX_FILES:
        raise ForagerMatchedV3LocalSourceBundleError("source snapshot files are invalid")
    files: list[dict[str, Any]] = []
    paths: list[str] = []
    aliases: set[str] = set()
    total_size = 0
    for index, item in enumerate(raw_files):
        record = _require_exact_keys(
            item,
            frozenset({"path", "size_bytes", "sha256"}),
            f"source snapshot file {index}",
        )
        path = _validate_relative_path(record["path"], f"source snapshot file {index} path")
        alias = path.casefold()
        if alias in aliases:
            raise ForagerMatchedV3LocalSourceBundleError(
                "source snapshot contains duplicate or aliased file paths"
            )
        size = _require_int(
            record["size_bytes"],
            f"source snapshot file {index} size",
            minimum=0,
            maximum=_MAX_FILE_BYTES,
        )
        digest = _require_sha256(record["sha256"], f"source snapshot file {index}")
        total_size += size
        if total_size > _MAX_TOTAL_BYTES:
            raise ForagerMatchedV3LocalSourceBundleError(
                "source snapshot payload exceeds its byte bound"
            )
        files.append({"path": path, "size_bytes": size, "sha256": digest})
        paths.append(path)
        aliases.add(alias)
    if paths != sorted(paths, key=lambda value: value.encode("ascii")):
        raise ForagerMatchedV3LocalSourceBundleError("source snapshot files are not ordered")
    if not all(name in paths for name in _ROOT_FILE_NAMES):
        raise ForagerMatchedV3LocalSourceBundleError("source snapshot omits a required root file")
    if f"{_FRAMEWORK_DIRECTORY_NAME}/__init__.py" not in paths:
        raise ForagerMatchedV3LocalSourceBundleError("source snapshot omits framework initializer")
    for path in paths:
        if path not in _ROOT_FILE_NAMES and path.rsplit("/", 1)[0] not in directory_set:
            raise ForagerMatchedV3LocalSourceBundleError("source snapshot file has no bound parent")
    if aliases & directory_aliases:
        raise ForagerMatchedV3LocalSourceBundleError(
            "source snapshot path identifies both file and directory"
        )

    inventory = _require_exact_keys(
        manifest["inventory"],
        frozenset({"directory_count", "file_count", "total_size_bytes"}),
        "source snapshot inventory",
    )
    directory_count = _require_int(
        inventory["directory_count"],
        "source snapshot directory count",
        minimum=1,
        maximum=_MAX_DIRECTORIES,
    )
    file_count = _require_int(
        inventory["file_count"],
        "source snapshot file count",
        minimum=2,
        maximum=_MAX_FILES,
    )
    recorded_total = _require_int(
        inventory["total_size_bytes"],
        "source snapshot total bytes",
        minimum=0,
        maximum=_MAX_TOTAL_BYTES,
    )
    if (
        directory_count != len(directories)
        or file_count != len(files)
        or recorded_total != total_size
    ):
        raise ForagerMatchedV3LocalSourceBundleError("source snapshot totals drifted")

    tree = _require_exact_keys(
        manifest["tree"], frozenset({"schema_version", "sha256"}), "source snapshot tree"
    )
    recorded_tree = _require_sha256(tree["sha256"], "source snapshot tree")
    computed_tree = hashlib.sha256(
        _canonical_json(
            {
                "schema_version": _SNAPSHOT_TREE_SCHEMA_VERSION,
                "directories": directories,
                "files": files,
            },
            maximum_bytes=_MAX_SNAPSHOT_MANIFEST_BYTES,
        )
    ).hexdigest()
    if (
        tree["schema_version"] != _SNAPSHOT_TREE_SCHEMA_VERSION
        or not hmac.compare_digest(recorded_tree, computed_tree)
        or not hmac.compare_digest(recorded_tree, tree_pin)
    ):
        raise ForagerMatchedV3LocalSourceBundleError("source snapshot tree identity drifted")
    claims = _require_exact_keys(
        manifest["claims"], frozenset(_snapshot_claims()), "source snapshot claims"
    )
    if not _exact_json_equal(claims, _snapshot_claims()) or any(
        value is not False for value in claims.values()
    ):
        raise ForagerMatchedV3LocalSourceBundleError("source snapshot claim became true")
    if not _exact_json_equal(manifest["limitations"], _snapshot_limitations()):
        raise ForagerMatchedV3LocalSourceBundleError("source snapshot limitations drifted")
    body = dict(manifest)
    body_digest = _require_sha256(body.pop("manifest_body_sha256"), "source snapshot manifest body")
    if not hmac.compare_digest(
        body_digest,
        hashlib.sha256(
            _canonical_json(body, maximum_bytes=_MAX_SNAPSHOT_MANIFEST_BYTES)
        ).hexdigest(),
    ):
        raise ForagerMatchedV3LocalSourceBundleError("source snapshot body digest drifted")
    return _SnapshotIdentity(
        manifest=manifest,
        manifest_sha256=manifest_sha256,
        tree_sha256=recorded_tree,
        directory_count=directory_count,
        file_count=file_count,
        total_size_bytes=total_size,
    )


def _stat_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _locator_identity(metadata: os.stat_result) -> tuple[int, int, int]:
    return (metadata.st_dev, metadata.st_ino, stat.S_IFMT(metadata.st_mode))


def _close_untransferred_descriptor(
    descriptor: int,
    *,
    failure: BaseException | None,
    label: str,
) -> None:
    if descriptor < 0:
        return
    try:
        os.close(descriptor)
    except BaseException as cleanup_error:
        if failure is not None:
            failure.add_note(f"{label} cleanup close also failed: {cleanup_error!r}")
            return
        raise


def _merge_cleanup_failure(
    primary: BaseException | None,
    cleanup_error: BaseException,
    *,
    label: str,
) -> BaseException:
    if primary is None:
        cleanup_error.add_note(f"while {label}")
        return cleanup_error
    primary.add_note(f"{label} also failed: {cleanup_error!r}")
    return primary


def _directory_flags() -> int:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if type(nofollow) is not int or type(directory) is not int:
        raise ForagerMatchedV3LocalSourceBundleError(
            "local source bundling requires O_NOFOLLOW and O_DIRECTORY"
        )
    return (
        os.O_RDONLY
        | nofollow
        | directory
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )


def _file_flags() -> int:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if type(nofollow) is not int:
        raise ForagerMatchedV3LocalSourceBundleError("local source bundling requires O_NOFOLLOW")
    return os.O_RDONLY | nofollow | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0)


@dataclass(slots=True)
class _AnchoredRoot:
    descriptors: list[int]
    components: list[str]
    metadata: list[os.stat_result]

    @property
    def root_descriptor(self) -> int:
        return self.descriptors[-1]

    def verify(self) -> None:
        for index in range(1, len(self.descriptors)):
            parent = self.descriptors[index - 1]
            descriptor = self.descriptors[index]
            component = self.components[index]
            expected = self.metadata[index]
            try:
                current = os.stat(component, dir_fd=parent, follow_symlinks=False)
                opened = os.fstat(descriptor)
            except OSError as exc:
                raise ForagerMatchedV3LocalSourceBundleError(
                    "repository root locator changed during bundling"
                ) from exc
            if _locator_identity(current) != _locator_identity(expected) or _locator_identity(
                opened
            ) != _locator_identity(expected):
                raise ForagerMatchedV3LocalSourceBundleError(
                    "repository root locator changed during bundling"
                )

    def close(self) -> None:
        descriptors = self.descriptors
        self.descriptors = []
        failure: BaseException | None = None
        for descriptor in reversed(descriptors):
            try:
                os.close(descriptor)
            except BaseException as cleanup_error:
                if failure is None:
                    failure = cleanup_error
                    failure.add_note("while closing an anchored repository descriptor")
                else:
                    failure.add_note(
                        "another anchored repository descriptor close also failed: "
                        f"{cleanup_error!r}"
                    )
        if failure is not None:
            raise failure


def _open_anchored_repository_root(repository_root: Any) -> _AnchoredRoot:
    concrete_path_type = type(Path())
    if type(repository_root) is not concrete_path_type:
        raise ForagerMatchedV3LocalSourceBundleError(
            "repository_root must be one exact concrete pathlib.Path"
        )
    root = repository_root
    if not root.is_absolute() or root.anchor != os.sep or root == Path(root.anchor):
        raise ForagerMatchedV3LocalSourceBundleError(
            "repository_root must be a non-root absolute path"
        )
    if os.path.abspath(str(root)) != str(root):
        raise ForagerMatchedV3LocalSourceBundleError(
            "repository_root must not contain aliases or traversal"
        )
    components = [root.anchor]
    for index, component in enumerate(root.parts[1:]):
        components.append(_validate_component(component, f"repository_root component {index}"))
    descriptors: list[int] = []
    metadata: list[os.stat_result] = []
    pending_descriptor = -1
    try:
        pending_descriptor = os.open(root.anchor, _directory_flags())
        anchor = pending_descriptor
        descriptors.append(anchor)
        pending_descriptor = -1
        metadata.append(os.fstat(anchor))
        for component in components[1:]:
            parent = descriptors[-1]
            before = os.stat(component, dir_fd=parent, follow_symlinks=False)
            if not stat.S_ISDIR(before.st_mode) or stat.S_ISLNK(before.st_mode):
                raise ForagerMatchedV3LocalSourceBundleError(
                    "repository_root contains a link or non-directory component"
                )
            pending_descriptor = os.open(component, _directory_flags(), dir_fd=parent)
            child = pending_descriptor
            descriptors.append(child)
            pending_descriptor = -1
            opened = os.fstat(child)
            after = os.stat(component, dir_fd=parent, follow_symlinks=False)
            if _stat_identity(before) != _stat_identity(opened) or _stat_identity(
                opened
            ) != _stat_identity(after):
                raise ForagerMatchedV3LocalSourceBundleError(
                    "repository_root changed while being opened"
                )
            metadata.append(opened)
    except OSError as exc:
        failure = ForagerMatchedV3LocalSourceBundleError(
            "repository_root cannot be opened without following links"
        )
        if pending_descriptor >= 0:
            _close_untransferred_descriptor(
                pending_descriptor,
                failure=failure,
                label="unrecorded repository root descriptor",
            )
        for descriptor in reversed(descriptors):
            try:
                os.close(descriptor)
            except BaseException as cleanup_error:
                failure.add_note(
                    f"repository root descriptor cleanup also failed: {cleanup_error!r}"
                )
        raise failure from exc
    except BaseException as failure:
        if pending_descriptor >= 0:
            _close_untransferred_descriptor(
                pending_descriptor,
                failure=failure,
                label="unrecorded repository root descriptor",
            )
        for descriptor in reversed(descriptors):
            try:
                os.close(descriptor)
            except BaseException as cleanup_error:
                failure.add_note(
                    f"repository root descriptor cleanup also failed: {cleanup_error!r}"
                )
        raise
    anchored = _AnchoredRoot(descriptors, components, metadata)
    try:
        anchored.verify()
    except BaseException as failure:
        try:
            anchored.close()
        except BaseException as cleanup_error:
            failure.add_note(f"anchored repository cleanup also failed: {cleanup_error!r}")
        raise
    return anchored


@dataclass(slots=True)
class _WalkState:
    files: int = 0
    directories: int = 0
    entries: int = 0
    total_bytes: int = 0

    def add_entries(self, count: int) -> None:
        self.entries += count
        if self.entries > _MAX_ENTRIES:
            raise ForagerMatchedV3LocalSourceBundleError("source exceeds its entry bound")

    def add_directory(self, depth: int) -> None:
        self.directories += 1
        if self.directories > _MAX_DIRECTORIES or depth > _MAX_DEPTH:
            raise ForagerMatchedV3LocalSourceBundleError("source exceeds its directory/depth bound")

    def add_file(self, size: int) -> None:
        if size < 0 or size > _MAX_FILE_BYTES:
            raise ForagerMatchedV3LocalSourceBundleError("source file exceeds its byte bound")
        self.files += 1
        self.total_bytes += size
        if self.files > _MAX_FILES or self.total_bytes > _MAX_TOTAL_BYTES:
            raise ForagerMatchedV3LocalSourceBundleError("source exceeds its file/byte bound")


def _safe_sorted_names(descriptor: int, *, maximum_entries: int) -> list[str]:
    if type(maximum_entries) is not int or not 0 <= maximum_entries <= _MAX_ENTRIES:
        raise ForagerMatchedV3LocalSourceBundleError("source entry budget is invalid")
    names: list[str] = []
    aliases: set[str] = set()
    try:
        with os.scandir(descriptor) as entries:
            for index, entry in enumerate(entries):
                if index >= maximum_entries:
                    raise ForagerMatchedV3LocalSourceBundleError(
                        "source directory exceeds the entry bound"
                    )
                name = _validate_component(entry.name, f"source entry {index}")
                alias = name.casefold()
                if alias in aliases:
                    raise ForagerMatchedV3LocalSourceBundleError(
                        "source directory contains duplicate or casefold-aliased entries"
                    )
                names.append(name)
                aliases.add(alias)
    except ForagerMatchedV3LocalSourceBundleError:
        raise
    except OSError as exc:
        raise ForagerMatchedV3LocalSourceBundleError(
            "anchored source directory cannot be enumerated"
        ) from exc
    return sorted(names, key=lambda value: value.encode("ascii"))


def _open_checked_child(
    parent: int, name: str, before: os.stat_result, *, directory: bool
) -> tuple[int, os.stat_result]:
    descriptor = -1
    failure: BaseException | None = None
    try:
        try:
            descriptor = os.open(
                name,
                _directory_flags() if directory else _file_flags(),
                dir_fd=parent,
            )
            opened = os.fstat(descriptor)
            after = os.stat(name, dir_fd=parent, follow_symlinks=False)
            expected_type = stat.S_ISDIR if directory else stat.S_ISREG
            if (
                not expected_type(opened.st_mode)
                or _stat_identity(before) != _stat_identity(opened)
                or _stat_identity(opened) != _stat_identity(after)
            ):
                raise ForagerMatchedV3LocalSourceBundleError("source entry changed while opening")
        except OSError as exc:
            raise ForagerMatchedV3LocalSourceBundleError(
                "source entry changed while opening"
            ) from exc
        result = (descriptor, opened)
        descriptor = -1
        return result
    except BaseException as exc:
        failure = exc
        raise
    finally:
        if descriptor >= 0:
            _close_untransferred_descriptor(
                descriptor,
                failure=failure,
                label="source entry descriptor",
            )


def _verify_opened_child(parent: int, name: str, descriptor: int, expected: os.stat_result) -> None:
    try:
        opened = os.fstat(descriptor)
        after = os.stat(name, dir_fd=parent, follow_symlinks=False)
    except OSError as exc:
        raise ForagerMatchedV3LocalSourceBundleError("source entry changed while reading") from exc
    if _stat_identity(opened) != _stat_identity(expected) or _stat_identity(
        after
    ) != _stat_identity(expected):
        raise ForagerMatchedV3LocalSourceBundleError("source entry changed while reading")


def _measure_regular_file(
    parent: int,
    name: str,
    relative_path: str,
    before: os.stat_result,
    state: _WalkState,
    *,
    include: bool,
) -> dict[str, Any] | None:
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise ForagerMatchedV3LocalSourceBundleError(
            "source inventory contains a hardlink, symlink, or special file"
        )
    state.add_file(before.st_size)
    descriptor, opened = _open_checked_child(parent, name, before, directory=False)
    failure: BaseException | None = None
    try:
        if opened.st_nlink != 1:
            raise ForagerMatchedV3LocalSourceBundleError(
                "source inventory contains a multiply linked file"
            )
        digest = hashlib.sha256()
        remaining = opened.st_size
        while remaining:
            try:
                chunk = os.read(descriptor, min(remaining, _READ_CHUNK_BYTES))
            except InterruptedError:
                continue
            except OSError as exc:
                raise ForagerMatchedV3LocalSourceBundleError(
                    "source file could not be read exactly"
                ) from exc
            if not chunk:
                raise ForagerMatchedV3LocalSourceBundleError("source file ended while reading")
            digest.update(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ForagerMatchedV3LocalSourceBundleError("source file grew while reading")
        _verify_opened_child(parent, name, descriptor, opened)
    except BaseException as exc:
        failure = exc
        raise
    finally:
        closing_descriptor = descriptor
        descriptor = -1
        _close_untransferred_descriptor(
            closing_descriptor,
            failure=failure,
            label="measured source file descriptor",
        )
    if not include:
        return None
    return {
        "path": _validate_relative_path(relative_path, "measured source path"),
        "size_bytes": opened.st_size,
        "sha256": digest.hexdigest(),
    }


def _walk_framework(
    descriptor: int,
    prefix: str,
    depth: int,
    opened: os.stat_result,
    state: _WalkState,
    directories: list[str],
    records: list[dict[str, Any]],
) -> None:
    state.add_directory(depth)
    directories.append(_validate_relative_directory(prefix, "measured source directory"))
    names = _safe_sorted_names(descriptor, maximum_entries=_MAX_ENTRIES - state.entries)
    state.add_entries(len(names))
    for name in names:
        relative = f"{prefix}/{name}"
        try:
            before = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        except OSError as exc:
            raise ForagerMatchedV3LocalSourceBundleError(
                "anchored source entry cannot be inspected"
            ) from exc
        if stat.S_ISLNK(before.st_mode):
            raise ForagerMatchedV3LocalSourceBundleError("source contains a symbolic link")
        if stat.S_ISDIR(before.st_mode):
            child, child_opened = _open_checked_child(descriptor, name, before, directory=True)
            child_failure: BaseException | None = None
            try:
                if name not in _CACHE_DIRECTORY_NAMES:
                    _walk_framework(
                        child,
                        relative,
                        depth + 1,
                        child_opened,
                        state,
                        directories,
                        records,
                    )
                _verify_opened_child(descriptor, name, child, child_opened)
            except BaseException as exc:
                child_failure = exc
                raise
            finally:
                closing_child = child
                child = -1
                _close_untransferred_descriptor(
                    closing_child,
                    failure=child_failure,
                    label="measured source directory descriptor",
                )
            continue
        if not stat.S_ISREG(before.st_mode):
            raise ForagerMatchedV3LocalSourceBundleError(
                "source contains a nonregular filesystem entry"
            )
        record = _measure_regular_file(
            descriptor,
            name,
            relative,
            before,
            state,
            include=not name.endswith(_CACHE_FILE_SUFFIXES),
        )
        if record is not None:
            records.append(record)
    names_after = _safe_sorted_names(descriptor, maximum_entries=len(names))
    if names != names_after or _stat_identity(os.fstat(descriptor)) != _stat_identity(opened):
        raise ForagerMatchedV3LocalSourceBundleError("source directory changed while reading")


def _measure_tree(repository_descriptor: int) -> dict[str, Any]:
    state = _WalkState()
    directories: list[str] = []
    records: list[dict[str, Any]] = []
    for root_name in _ROOT_FILE_NAMES:
        state.add_entries(1)
        try:
            before = os.stat(root_name, dir_fd=repository_descriptor, follow_symlinks=False)
        except OSError as exc:
            raise ForagerMatchedV3LocalSourceBundleError(
                f"required root source file is unavailable: {root_name}"
            ) from exc
        record = _measure_regular_file(
            repository_descriptor,
            root_name,
            root_name,
            before,
            state,
            include=True,
        )
        if record is None:  # pragma: no cover
            raise AssertionError("required source record was excluded")
        records.append(record)
    state.add_entries(1)
    try:
        before = os.stat(
            _FRAMEWORK_DIRECTORY_NAME,
            dir_fd=repository_descriptor,
            follow_symlinks=False,
        )
    except OSError as exc:
        raise ForagerMatchedV3LocalSourceBundleError(
            "framework source root is unavailable"
        ) from exc
    if not stat.S_ISDIR(before.st_mode) or stat.S_ISLNK(before.st_mode):
        raise ForagerMatchedV3LocalSourceBundleError("framework source root is not a directory")
    framework, opened = _open_checked_child(
        repository_descriptor,
        _FRAMEWORK_DIRECTORY_NAME,
        before,
        directory=True,
    )
    failure: BaseException | None = None
    try:
        _walk_framework(
            framework,
            _FRAMEWORK_DIRECTORY_NAME,
            1,
            opened,
            state,
            directories,
            records,
        )
        _verify_opened_child(
            repository_descriptor,
            _FRAMEWORK_DIRECTORY_NAME,
            framework,
            opened,
        )
    except BaseException as exc:
        failure = exc
        raise
    finally:
        closing_framework = framework
        framework = -1
        _close_untransferred_descriptor(
            closing_framework,
            failure=failure,
            label="framework source root descriptor",
        )
    records.sort(key=lambda item: cast(str, item["path"]).encode("ascii"))
    directories.sort(key=lambda value: value.encode("ascii"))
    return {"directories": directories, "files": records}


def _expected_tree_inventory(snapshot: _SnapshotIdentity) -> dict[str, Any]:
    return {
        "directories": list(cast(list[str], snapshot.manifest["directories"])),
        "files": [dict(item) for item in cast(list[dict[str, Any]], snapshot.manifest["files"])],
    }


def _require_expected_tree(observed: dict[str, Any], expected: dict[str, Any], label: str) -> None:
    if not _exact_json_equal(observed, expected):
        raise ForagerMatchedV3LocalSourceBundleError(
            f"{label} source tree differs from the caller-pinned snapshot"
        )


def _duplicate_directory_descriptor(descriptor: int, label: str) -> int:
    duplicate = -1
    failure: BaseException | None = None
    try:
        try:
            before = os.fstat(descriptor)
            duplicate = fcntl.fcntl(descriptor, fcntl.F_DUPFD_CLOEXEC, 3)
            after = os.fstat(descriptor)
            retained = os.fstat(duplicate)
            if (
                not stat.S_ISDIR(before.st_mode)
                or _locator_identity(before) != _locator_identity(after)
                or _locator_identity(before) != _locator_identity(retained)
                or os.get_inheritable(duplicate)
            ):
                raise ForagerMatchedV3LocalSourceBundleError(f"{label} identity changed")
        except OSError as exc:
            raise ForagerMatchedV3LocalSourceBundleError(f"{label} cannot be retained") from exc
        result = duplicate
        duplicate = -1
        return result
    except BaseException as exc:
        failure = exc
        raise
    finally:
        if duplicate >= 0:
            _close_untransferred_descriptor(
                duplicate,
                failure=failure,
                label=f"{label} duplicate descriptor",
            )


def _open_relative_parent(root_descriptor: int, path: str) -> tuple[int, str]:
    parts = path.split("/")
    current = -1
    failure: BaseException | None = None
    try:
        current = _duplicate_directory_descriptor(root_descriptor, "source root")
        for part in parts[:-1]:
            child = -1
            child_failure: BaseException | None = None
            try:
                try:
                    before = os.stat(part, dir_fd=current, follow_symlinks=False)
                    child = os.open(part, _directory_flags(), dir_fd=current)
                    opened = os.fstat(child)
                    after = os.stat(part, dir_fd=current, follow_symlinks=False)
                    if (
                        not stat.S_ISDIR(before.st_mode)
                        or stat.S_ISLNK(before.st_mode)
                        or _stat_identity(before) != _stat_identity(opened)
                        or _stat_identity(before) != _stat_identity(after)
                    ):
                        raise ForagerMatchedV3LocalSourceBundleError(
                            f"source ancestor changed: {path}"
                        )
                except OSError as exc:
                    raise ForagerMatchedV3LocalSourceBundleError(
                        f"source ancestor is inaccessible: {path}"
                    ) from exc
                previous = current
                current = -1
                _close_untransferred_descriptor(
                    previous,
                    failure=None,
                    label="source ancestor descriptor",
                )
                current = child
                child = -1
            except BaseException as exc:
                child_failure = exc
                raise
            finally:
                if child >= 0:
                    _close_untransferred_descriptor(
                        child,
                        failure=child_failure,
                        label="source ancestor child descriptor",
                    )
        result = (current, parts[-1])
        current = -1
        return result
    except BaseException as exc:
        failure = exc
        raise
    finally:
        if current >= 0:
            _close_untransferred_descriptor(
                current,
                failure=failure,
                label="source parent descriptor",
            )


def _ustar_octal(value: int, width: int, label: str) -> bytes:
    if type(value) is not int or value < 0:
        raise ForagerMatchedV3LocalSourceBundleError(f"{label} is invalid")
    token = format(value, "o").encode("ascii")
    if len(token) > width - 1:
        raise ForagerMatchedV3LocalSourceBundleError(f"{label} exceeds its USTAR field")
    return token.rjust(width - 1, b"0") + b"\0"


def _canonical_ustar_header(path: str, size: int) -> bytes:
    prefix, name = _split_ustar_path(path)
    header = bytearray(_USTAR_BLOCK_BYTES)
    header[0 : len(name)] = name
    header[100:108] = _ustar_octal(0o444, 8, "member mode")
    header[108:116] = _ustar_octal(0, 8, "member uid")
    header[116:124] = _ustar_octal(0, 8, "member gid")
    header[124:136] = _ustar_octal(size, 12, "member size")
    header[136:148] = _ustar_octal(0, 12, "member mtime")
    header[148:156] = b"        "
    header[156:157] = b"0"
    header[257:263] = b"ustar\0"
    header[263:265] = b"00"
    header[345 : 345 + len(prefix)] = prefix
    checksum = format(sum(header), "06o").encode("ascii")
    if len(checksum) != 6:
        raise ForagerMatchedV3LocalSourceBundleError("USTAR checksum overflowed")
    header[148:156] = checksum + b"\0 "
    return bytes(header)


def _write_all(descriptor: int, raw: bytes) -> None:
    view = memoryview(raw)
    while view:
        try:
            written = os.write(descriptor, view)
        except InterruptedError:
            continue
        if written <= 0:
            raise ForagerMatchedV3LocalSourceBundleError("archive write made no progress")
        view = view[written:]


class _HashingWriter:
    def __init__(self, descriptor: int) -> None:
        self.descriptor = descriptor
        self.size = 0
        self.digest = hashlib.sha256()

    def write(self, raw: bytes) -> None:
        if type(raw) is not bytes or self.size + len(raw) > _MAX_ARCHIVE_BYTES:
            raise ForagerMatchedV3LocalSourceBundleError("canonical USTAR exceeds its bound")
        _write_all(self.descriptor, raw)
        self.digest.update(raw)
        self.size += len(raw)


def _stream_source_member(
    root_descriptor: int,
    record: Mapping[str, Any],
    writer: _HashingWriter,
) -> None:
    path = cast(str, record["path"])
    expected_size = cast(int, record["size_bytes"])
    expected_sha256 = cast(str, record["sha256"])
    parent, name = _open_relative_parent(root_descriptor, path)
    descriptor = -1
    failure: BaseException | None = None
    try:
        try:
            name_before = os.stat(name, dir_fd=parent, follow_symlinks=False)
            descriptor = os.open(name, _file_flags(), dir_fd=parent)
            opened = os.fstat(descriptor)
        except OSError as exc:
            raise ForagerMatchedV3LocalSourceBundleError(
                f"archive source member is inaccessible: {path}"
            ) from exc
        if (
            not stat.S_ISREG(name_before.st_mode)
            or stat.S_ISLNK(name_before.st_mode)
            or opened.st_nlink != 1
            or _stat_identity(name_before) != _stat_identity(opened)
            or opened.st_size != expected_size
        ):
            raise ForagerMatchedV3LocalSourceBundleError(
                f"archive source member identity differs: {path}"
            )
        writer.write(_canonical_ustar_header(path, expected_size))
        digest = hashlib.sha256()
        remaining = expected_size
        while remaining:
            try:
                chunk = os.read(descriptor, min(remaining, _READ_CHUNK_BYTES))
            except InterruptedError:
                continue
            if not chunk:
                raise ForagerMatchedV3LocalSourceBundleError(
                    f"archive source member ended early: {path}"
                )
            digest.update(chunk)
            writer.write(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ForagerMatchedV3LocalSourceBundleError(f"archive source member grew: {path}")
        opened_after = os.fstat(descriptor)
        name_after = os.stat(name, dir_fd=parent, follow_symlinks=False)
        if (
            _stat_identity(opened) != _stat_identity(opened_after)
            or _stat_identity(opened) != _stat_identity(name_after)
            or not hmac.compare_digest(digest.hexdigest(), expected_sha256)
        ):
            raise ForagerMatchedV3LocalSourceBundleError(
                f"archive source member changed or differs from snapshot: {path}"
            )
        padding = (-expected_size) % _USTAR_BLOCK_BYTES
        if padding:
            writer.write(bytes(padding))
    except BaseException as exc:
        failure = exc
        raise
    finally:
        cleanup_failure = failure
        if descriptor >= 0:
            closing_descriptor = descriptor
            descriptor = -1
            try:
                os.close(closing_descriptor)
            except BaseException as cleanup_error:
                cleanup_failure = _merge_cleanup_failure(
                    cleanup_failure,
                    cleanup_error,
                    label=f"closing archive source member descriptor {path}",
                )
        closing_parent = parent
        parent = -1
        try:
            os.close(closing_parent)
        except BaseException as cleanup_error:
            cleanup_failure = _merge_cleanup_failure(
                cleanup_failure,
                cleanup_error,
                label=f"closing archive source parent descriptor {path}",
            )
        if failure is None and cleanup_failure is not None:
            raise cleanup_failure


def _write_archive(
    descriptor: int,
    repository_descriptor: int,
    records: Sequence[Mapping[str, Any]],
) -> tuple[int, str]:
    if type(records) not in {list, tuple} or not 0 < len(records) <= _MAX_FILES:
        raise ForagerMatchedV3LocalSourceBundleError("source member inventory is invalid")
    paths = [cast(str, record["path"]) for record in records]
    if paths != sorted(paths, key=lambda value: value.encode("ascii")) or len(paths) != len(
        set(paths)
    ):
        raise ForagerMatchedV3LocalSourceBundleError("source member order is not exact")
    writer = _HashingWriter(descriptor)
    for record in records:
        _stream_source_member(repository_descriptor, record, writer)
    writer.write(bytes(2 * _USTAR_BLOCK_BYTES))
    record_padding = (-writer.size) % _USTAR_RECORD_BYTES
    if record_padding:
        writer.write(bytes(record_padding))
    return writer.size, writer.digest.hexdigest()


def _pread_exact(descriptor: int, size: int, offset: int, label: str) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    position = offset
    while remaining:
        try:
            chunk = os.pread(descriptor, min(remaining, _READ_CHUNK_BYTES), position)
        except InterruptedError:
            continue
        except OSError as exc:
            raise ForagerMatchedV3LocalSourceBundleError(f"{label} could not be read") from exc
        if not chunk:
            raise ForagerMatchedV3LocalSourceBundleError(f"{label} ended early")
        chunks.append(chunk)
        remaining -= len(chunk)
        position += len(chunk)
    return b"".join(chunks)


def _hash_fd(descriptor: int, size: int) -> str:
    digest = hashlib.sha256()
    offset = 0
    while offset < size:
        raw = _pread_exact(
            descriptor,
            min(_READ_CHUNK_BYTES, size - offset),
            offset,
            "retained source bundle",
        )
        digest.update(raw)
        offset += len(raw)
    return digest.hexdigest()


def _build_receipt(
    snapshot: _SnapshotIdentity,
    *,
    archive_size_bytes: int,
    archive_sha256: str,
) -> tuple[bytes, str]:
    members = [
        {
            "path": cast(str, item["path"]),
            "size_bytes": cast(int, item["size_bytes"]),
            "sha256": cast(str, item["sha256"]),
            "mode": "0444",
        }
        for item in cast(list[dict[str, Any]], snapshot.manifest["files"])
    ]
    body: dict[str, Any] = {
        "schema_version": LOCAL_SOURCE_BUNDLE_RECEIPT_SCHEMA_VERSION,
        "status": LOCAL_SOURCE_BUNDLE_STATUS,
        "classification": "retained_local_source_payload_non_authorizing",
        "descriptor_binding": {
            "schema_version": LOCAL_SOURCE_BUNDLE_DESCRIPTOR_SCHEMA_VERSION,
            "sha256": LOCAL_SOURCE_BUNDLE_DESCRIPTOR_SHA256,
        },
        "source_snapshot": {
            "manifest_schema_version": _SNAPSHOT_MANIFEST_SCHEMA_VERSION,
            "manifest_sha256": snapshot.manifest_sha256,
            "tree_schema_version": _SNAPSHOT_TREE_SCHEMA_VERSION,
            "tree_sha256": snapshot.tree_sha256,
            "directory_count": snapshot.directory_count,
            "file_count": snapshot.file_count,
            "total_size_bytes": snapshot.total_size_bytes,
        },
        "archive": {
            "format": "canonical_posix_ustar_uncompressed",
            "size_bytes": archive_size_bytes,
            "sha256": archive_sha256,
            "member_count": len(members),
            "member_mode": "0444",
            "record_size_bytes": _USTAR_RECORD_BYTES,
        },
        "members": members,
        "claims": _claims(),
        "limitations": _limitations(),
    }
    receipt = {
        **body,
        "receipt_body_sha256": hashlib.sha256(
            _canonical_json(body, maximum_bytes=_MAX_RECEIPT_BYTES)
        ).hexdigest(),
    }
    raw = _canonical_json(receipt, maximum_bytes=_MAX_RECEIPT_BYTES)
    return raw, hashlib.sha256(raw).hexdigest()


def _validate_receipt(value: dict[str, Any]) -> None:
    receipt = _require_exact_keys(
        value,
        frozenset(
            {
                "schema_version",
                "status",
                "classification",
                "descriptor_binding",
                "source_snapshot",
                "archive",
                "members",
                "claims",
                "limitations",
                "receipt_body_sha256",
            }
        ),
        "local source bundle receipt",
    )
    if (
        receipt["schema_version"] != LOCAL_SOURCE_BUNDLE_RECEIPT_SCHEMA_VERSION
        or receipt["status"] != LOCAL_SOURCE_BUNDLE_STATUS
        or receipt["classification"] != "retained_local_source_payload_non_authorizing"
    ):
        raise ForagerMatchedV3LocalSourceBundleError("source bundle receipt identity drifted")
    if not _exact_json_equal(
        receipt["descriptor_binding"],
        {
            "schema_version": LOCAL_SOURCE_BUNDLE_DESCRIPTOR_SCHEMA_VERSION,
            "sha256": LOCAL_SOURCE_BUNDLE_DESCRIPTOR_SHA256,
        },
    ):
        raise ForagerMatchedV3LocalSourceBundleError("source bundle descriptor binding drifted")
    source = _require_exact_keys(
        receipt["source_snapshot"],
        frozenset(
            {
                "manifest_schema_version",
                "manifest_sha256",
                "tree_schema_version",
                "tree_sha256",
                "directory_count",
                "file_count",
                "total_size_bytes",
            }
        ),
        "source bundle snapshot binding",
    )
    if (
        source["manifest_schema_version"] != _SNAPSHOT_MANIFEST_SCHEMA_VERSION
        or source["tree_schema_version"] != _SNAPSHOT_TREE_SCHEMA_VERSION
    ):
        raise ForagerMatchedV3LocalSourceBundleError("source bundle snapshot schema drifted")
    _require_sha256(source["manifest_sha256"], "source bundle manifest")
    _require_sha256(source["tree_sha256"], "source bundle tree")
    directory_count = _require_int(
        source["directory_count"],
        "source bundle directory count",
        minimum=1,
        maximum=_MAX_DIRECTORIES,
    )
    file_count = _require_int(
        source["file_count"], "source bundle file count", minimum=2, maximum=_MAX_FILES
    )
    total_size = _require_int(
        source["total_size_bytes"],
        "source bundle total size",
        minimum=0,
        maximum=_MAX_TOTAL_BYTES,
    )
    archive = _require_exact_keys(
        receipt["archive"],
        frozenset(
            {
                "format",
                "size_bytes",
                "sha256",
                "member_count",
                "member_mode",
                "record_size_bytes",
            }
        ),
        "source bundle archive",
    )
    if (
        archive["format"] != "canonical_posix_ustar_uncompressed"
        or archive["member_mode"] != "0444"
        or archive["record_size_bytes"] != _USTAR_RECORD_BYTES
    ):
        raise ForagerMatchedV3LocalSourceBundleError("source bundle archive layout drifted")
    archive_size = _require_int(
        archive["size_bytes"], "source bundle archive size", minimum=1, maximum=_MAX_ARCHIVE_BYTES
    )
    if archive_size % _USTAR_RECORD_BYTES:
        raise ForagerMatchedV3LocalSourceBundleError("source bundle archive size is not canonical")
    _require_sha256(archive["sha256"], "source bundle archive")
    member_count = _require_int(
        archive["member_count"], "source bundle member count", minimum=2, maximum=_MAX_FILES
    )
    raw_members = receipt["members"]
    if (
        type(raw_members) is not list
        or len(raw_members) != member_count
        or member_count != file_count
    ):
        raise ForagerMatchedV3LocalSourceBundleError("source bundle member count drifted")
    paths: list[str] = []
    payload_total = 0
    aliases: set[str] = set()
    for index, item in enumerate(raw_members):
        member = _require_exact_keys(
            item,
            frozenset({"path", "size_bytes", "sha256", "mode"}),
            f"source bundle member {index}",
        )
        path = _validate_relative_path(member["path"], f"source bundle member {index} path")
        if path.casefold() in aliases or member["mode"] != "0444":
            raise ForagerMatchedV3LocalSourceBundleError("source bundle member path/mode drifted")
        aliases.add(path.casefold())
        paths.append(path)
        payload_total += _require_int(
            member["size_bytes"],
            f"source bundle member {index} size",
            minimum=0,
            maximum=_MAX_FILE_BYTES,
        )
        _require_sha256(member["sha256"], f"source bundle member {index}")
    if paths != sorted(paths, key=lambda value: value.encode("ascii")):
        raise ForagerMatchedV3LocalSourceBundleError("source bundle members are not ordered")
    if payload_total != total_size or directory_count < 1:
        raise ForagerMatchedV3LocalSourceBundleError("source bundle payload total drifted")
    payload_end = sum(
        _USTAR_BLOCK_BYTES
        + cast(int, member["size_bytes"])
        + ((-cast(int, member["size_bytes"])) % _USTAR_BLOCK_BYTES)
        for member in raw_members
    ) + (2 * _USTAR_BLOCK_BYTES)
    canonical_archive_size = payload_end + ((-payload_end) % _USTAR_RECORD_BYTES)
    if archive_size != canonical_archive_size:
        raise ForagerMatchedV3LocalSourceBundleError(
            "source bundle archive size differs from its complete member inventory"
        )
    claims = _require_exact_keys(receipt["claims"], frozenset(_claims()), "source bundle claims")
    if not _exact_json_equal(claims, _claims()) or any(
        value is not False for value in claims.values()
    ):
        raise ForagerMatchedV3LocalSourceBundleError("source bundle claim became true")
    if not _exact_json_equal(receipt["limitations"], _limitations()):
        raise ForagerMatchedV3LocalSourceBundleError("source bundle limitations drifted")
    body = dict(receipt)
    body_digest = _require_sha256(body.pop("receipt_body_sha256"), "source bundle receipt body")
    if not hmac.compare_digest(
        body_digest,
        hashlib.sha256(_canonical_json(body, maximum_bytes=_MAX_RECEIPT_BYTES)).hexdigest(),
    ):
        raise ForagerMatchedV3LocalSourceBundleError("source bundle receipt body drifted")


def _verify_archive_fd(
    descriptor: int,
    *,
    expected_size: int,
    expected_sha256: str,
    receipt: Mapping[str, Any],
) -> None:
    expected_size = _require_int(
        expected_size, "expected source bundle archive size", minimum=1, maximum=_MAX_ARCHIVE_BYTES
    )
    expected_sha256 = _require_sha256(expected_sha256, "expected source bundle archive")
    try:
        before = os.fstat(descriptor)
    except OSError as exc:
        raise ForagerMatchedV3LocalSourceBundleError(
            "source bundle descriptor is inaccessible"
        ) from exc
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_size != expected_size
        or expected_size % _USTAR_RECORD_BYTES
    ):
        raise ForagerMatchedV3LocalSourceBundleError("source bundle descriptor metadata differs")
    members = cast(list[dict[str, Any]], receipt["members"])
    offset = 0
    for record in members:
        path = cast(str, record["path"])
        size = cast(int, record["size_bytes"])
        header = _pread_exact(descriptor, _USTAR_BLOCK_BYTES, offset, f"USTAR header {path}")
        if not hmac.compare_digest(header, _canonical_ustar_header(path, size)):
            raise ForagerMatchedV3LocalSourceBundleError(f"USTAR header is not canonical: {path}")
        offset += _USTAR_BLOCK_BYTES
        digest = hashlib.sha256()
        remaining = size
        while remaining:
            chunk = _pread_exact(
                descriptor,
                min(remaining, _READ_CHUNK_BYTES),
                offset,
                f"USTAR payload {path}",
            )
            digest.update(chunk)
            offset += len(chunk)
            remaining -= len(chunk)
        if not hmac.compare_digest(digest.hexdigest(), cast(str, record["sha256"])):
            raise ForagerMatchedV3LocalSourceBundleError(f"USTAR payload differs: {path}")
        padding = (-size) % _USTAR_BLOCK_BYTES
        if padding and any(_pread_exact(descriptor, padding, offset, f"USTAR padding {path}")):
            raise ForagerMatchedV3LocalSourceBundleError(f"USTAR padding is nonzero: {path}")
        offset += padding
    if any(_pread_exact(descriptor, 2 * _USTAR_BLOCK_BYTES, offset, "USTAR end blocks")):
        raise ForagerMatchedV3LocalSourceBundleError("USTAR end blocks are nonzero")
    offset += 2 * _USTAR_BLOCK_BYTES
    canonical_size = offset + ((-offset) % _USTAR_RECORD_BYTES)
    if canonical_size != expected_size:
        raise ForagerMatchedV3LocalSourceBundleError("USTAR record padding length differs")
    tail = expected_size - offset
    if tail and any(_pread_exact(descriptor, tail, offset, "USTAR record padding")):
        raise ForagerMatchedV3LocalSourceBundleError("USTAR record padding is nonzero")
    actual_sha256 = _hash_fd(descriptor, expected_size)
    after = os.fstat(descriptor)
    if not hmac.compare_digest(actual_sha256, expected_sha256) or _stat_identity(
        before
    ) != _stat_identity(after):
        raise ForagerMatchedV3LocalSourceBundleError(
            "source bundle digest or descriptor stability differs"
        )


def _memfd_flags() -> int:
    cloexec = getattr(os, "MFD_CLOEXEC", None)
    sealing = getattr(os, "MFD_ALLOW_SEALING", None)
    if type(cloexec) is not int or type(sealing) is not int:
        raise ForagerMatchedV3LocalSourceBundleError(
            "local source bundling requires sealed anonymous memfd support"
        )
    return cloexec | sealing


def _required_seals() -> int:
    values = [
        getattr(fcntl, name, None)
        for name in ("F_SEAL_WRITE", "F_SEAL_GROW", "F_SEAL_SHRINK", "F_SEAL_SEAL")
    ]
    if any(type(value) is not int for value in values):
        raise ForagerMatchedV3LocalSourceBundleError(
            "local source bundling requires full memfd seals"
        )
    return sum(cast(int, value) for value in values)


def _create_private_memfd() -> int:
    creator = getattr(os, "memfd_create", None)
    if creator is None:
        raise ForagerMatchedV3LocalSourceBundleError("os.memfd_create is required")
    descriptor = -1
    failure: BaseException | None = None
    try:
        try:
            descriptor = creator("alberta-local-source-bundle", _memfd_flags())
            os.fchmod(descriptor, 0o600)
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 0
                or stat.S_IMODE(metadata.st_mode) != 0o600
                or os.get_inheritable(descriptor)
            ):
                raise ForagerMatchedV3LocalSourceBundleError(
                    "private source bundle descriptor metadata differs"
                )
        except OSError as exc:
            raise ForagerMatchedV3LocalSourceBundleError(
                "private source bundle descriptor cannot be created"
            ) from exc
        result = int(descriptor)
        descriptor = -1
        return result
    except BaseException as exc:
        failure = exc
        raise
    finally:
        if descriptor >= 0:
            _close_untransferred_descriptor(
                descriptor,
                failure=failure,
                label="private source bundle descriptor",
            )


def _seal_and_reopen_readonly(descriptor: int, *, expected_size: int) -> int:
    readonly = -1
    failure: BaseException | None = None
    try:
        try:
            os.fsync(descriptor)
            os.fchmod(descriptor, 0o400)
            fcntl.fcntl(descriptor, fcntl.F_ADD_SEALS, _required_seals())
            seals = fcntl.fcntl(descriptor, fcntl.F_GET_SEALS)
            before = os.fstat(descriptor)
            readonly = os.open(
                f"/proc/self/fd/{descriptor}",
                os.O_RDONLY | getattr(os, "O_CLOEXEC", 0),
            )
            after = os.fstat(readonly)
            if (
                seals & _required_seals() != _required_seals()
                or not stat.S_ISREG(before.st_mode)
                or before.st_nlink != 0
                or before.st_size != expected_size
                or stat.S_IMODE(before.st_mode) != 0o400
                or (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino)
                or fcntl.fcntl(readonly, fcntl.F_GETFL) & os.O_ACCMODE != os.O_RDONLY
                or os.get_inheritable(readonly)
            ):
                raise ForagerMatchedV3LocalSourceBundleError(
                    "sealed source bundle descriptor metadata differs"
                )
        except OSError as exc:
            raise ForagerMatchedV3LocalSourceBundleError(
                "source bundle cannot be sealed and reopened read-only"
            ) from exc
        result = readonly
        readonly = -1
        return result
    except BaseException as exc:
        failure = exc
        raise
    finally:
        if readonly >= 0:
            _close_untransferred_descriptor(
                readonly,
                failure=failure,
                label="sealed read-only source bundle descriptor",
            )


_CAPABILITY_CREATION_TOKEN: Final = object()


class RetainedMatchedV3LocalSourceBundle:
    """PID-bound sealed content capability for one exact local-source USTAR."""

    __slots__ = (
        "_archive_sha256",
        "_archive_size_bytes",
        "_descriptor",
        "_device",
        "_inode",
        "_owner_pid",
        "_receipt_bytes",
        "_receipt_sha256",
        "_snapshot_manifest_sha256",
        "_snapshot_tree_sha256",
    )

    def __init__(
        self,
        creation_token: object,
        descriptor: int,
        device: int,
        inode: int,
        archive_size_bytes: int,
        archive_sha256: str,
        receipt_bytes: bytes,
        receipt_sha256: str,
        snapshot_manifest_sha256: str,
        snapshot_tree_sha256: str,
    ) -> None:
        if creation_token is not _CAPABILITY_CREATION_TOKEN:
            raise TypeError("retained source bundles require the producer context")
        self._descriptor = descriptor
        self._device = device
        self._inode = inode
        self._archive_size_bytes = archive_size_bytes
        self._archive_sha256 = archive_sha256
        self._receipt_bytes = receipt_bytes
        self._receipt_sha256 = receipt_sha256
        self._snapshot_manifest_sha256 = snapshot_manifest_sha256
        self._snapshot_tree_sha256 = snapshot_tree_sha256
        self._owner_pid = os.getpid()

    def __reduce__(self) -> Never:
        raise TypeError("retained source bundles cannot be serialized")

    def __reduce_ex__(self, protocol: SupportsIndex) -> Never:
        del protocol
        raise TypeError("retained source bundles cannot be serialized")

    def __copy__(self) -> Never:
        raise TypeError("retained source bundles cannot be copied")

    def __deepcopy__(self, memo: object) -> Never:
        del memo
        raise TypeError("retained source bundles cannot be copied")

    def _invalidate(self, *, close_if_owned: bool) -> None:
        descriptor = self._descriptor
        self._descriptor = -1
        if descriptor < 0 or not close_if_owned:
            return
        try:
            metadata = os.fstat(descriptor)
        except OSError:
            return
        if (metadata.st_dev, metadata.st_ino) != (self._device, self._inode):
            return
        try:
            os.close(descriptor)
        except OSError:
            pass

    def _require_active(self) -> int:
        if os.getpid() != self._owner_pid:
            self._invalidate(close_if_owned=True)
            raise ForagerMatchedV3LocalSourceBundleError(
                "retained source bundle is invalid after a PID change"
            )
        descriptor = self._descriptor
        if descriptor < 0:
            raise ForagerMatchedV3LocalSourceBundleError("retained source bundle is closed")
        try:
            metadata = os.fstat(descriptor)
            flags = fcntl.fcntl(descriptor, fcntl.F_GETFL)
            descriptor_flags = fcntl.fcntl(descriptor, fcntl.F_GETFD)
            seals = fcntl.fcntl(descriptor, fcntl.F_GET_SEALS)
        except OSError as exc:
            self._invalidate(close_if_owned=True)
            raise ForagerMatchedV3LocalSourceBundleError(
                "retained source bundle descriptor became inaccessible"
            ) from exc
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 0
            or stat.S_IMODE(metadata.st_mode) != 0o400
            or metadata.st_size != self._archive_size_bytes
            or (metadata.st_dev, metadata.st_ino) != (self._device, self._inode)
            or flags & os.O_ACCMODE != os.O_RDONLY
            or descriptor_flags & fcntl.FD_CLOEXEC == 0
            or seals & _required_seals() != _required_seals()
            or os.get_inheritable(descriptor)
        ):
            same = (metadata.st_dev, metadata.st_ino) == (self._device, self._inode)
            self._invalidate(close_if_owned=same)
            raise ForagerMatchedV3LocalSourceBundleError(
                "retained source bundle descriptor identity drifted"
            )
        return descriptor

    @property
    def closed(self) -> bool:
        if self._descriptor >= 0 and os.getpid() != self._owner_pid:
            self._invalidate(close_if_owned=True)
        return self._descriptor < 0

    @property
    def proc_fd_path(self) -> str:
        return f"/proc/self/fd/{self._require_active()}"

    @property
    def subprocess_pass_fds(self) -> tuple[int, ...]:
        return (self._require_active(),)

    @property
    def owner_pid(self) -> int:
        self._require_active()
        return self._owner_pid

    @property
    def archive_size_bytes(self) -> int:
        self._require_active()
        return self._archive_size_bytes

    @property
    def archive_sha256(self) -> str:
        self._require_active()
        return self._archive_sha256

    @property
    def receipt_bytes(self) -> bytes:
        self._require_active()
        return self._receipt_bytes

    @property
    def receipt_sha256(self) -> str:
        self._require_active()
        return self._receipt_sha256

    @property
    def member_count(self) -> int:
        self._require_active()
        receipt = parse_matched_v3_local_source_bundle_receipt(
            self._receipt_bytes, expected_receipt_sha256=self._receipt_sha256
        )
        return cast(int, cast(dict[str, Any], receipt["archive"])["member_count"])

    @property
    def source_snapshot_manifest_sha256(self) -> str:
        self._require_active()
        return self._snapshot_manifest_sha256

    @property
    def source_manifest_sha256(self) -> str:
        """Return the exact caller-pinned source snapshot manifest digest."""

        self._require_active()
        return self._snapshot_manifest_sha256

    @property
    def source_snapshot_tree_sha256(self) -> str:
        self._require_active()
        return self._snapshot_tree_sha256

    @property
    def source_tree_sha256(self) -> str:
        """Return the exact caller-pinned source snapshot tree digest."""

        self._require_active()
        return self._snapshot_tree_sha256

    def read_archive_bytes(self) -> bytes:
        """Return exact bounded USTAR bytes after full receipt-bound replay."""

        self.reverify()
        try:
            descriptor = self._require_active()
            before = os.fstat(descriptor)
            raw = _pread_exact(
                descriptor,
                self._archive_size_bytes,
                0,
                "retained source bundle byte read",
            )
            after = os.fstat(descriptor)
            if _stat_identity(before) != _stat_identity(after) or not hmac.compare_digest(
                hashlib.sha256(raw).hexdigest(), self._archive_sha256
            ):
                raise ForagerMatchedV3LocalSourceBundleError(
                    "retained source bundle changed during exact byte read"
                )
            return raw
        except BaseException:
            self._invalidate(close_if_owned=True)
            raise

    def receipt(self) -> dict[str, Any]:
        self._require_active()
        return parse_matched_v3_local_source_bundle_receipt(
            self._receipt_bytes, expected_receipt_sha256=self._receipt_sha256
        )

    def reverify(self) -> dict[str, Any]:
        descriptor = self._require_active()
        try:
            receipt = verify_matched_v3_local_source_bundle_archive(
                descriptor=descriptor,
                expected_archive_size_bytes=self._archive_size_bytes,
                expected_archive_sha256=self._archive_sha256,
                expected_receipt_bytes=self._receipt_bytes,
                expected_receipt_sha256=self._receipt_sha256,
                expected_source_snapshot_manifest_sha256=self._snapshot_manifest_sha256,
                expected_source_snapshot_tree_sha256=self._snapshot_tree_sha256,
            )
            self._require_active()
            return receipt
        except BaseException:
            self._invalidate(close_if_owned=True)
            raise

    def close(self) -> None:
        self._invalidate(close_if_owned=True)


def matched_v3_local_source_bundle_descriptor() -> dict[str, Any]:
    """Return a detached copy of the frozen nonauthorizing bundle descriptor."""

    return _strict_json_load(_DESCRIPTOR_BYTES, maximum_bytes=_MAX_RECEIPT_BYTES)


def canonical_matched_v3_local_source_bundle_descriptor_bytes() -> bytes:
    """Return exact canonical bundle-descriptor bytes."""

    return _DESCRIPTOR_BYTES


def matched_v3_local_source_bundle_descriptor_sha256() -> str:
    """Return the frozen bundle-descriptor SHA-256."""

    return LOCAL_SOURCE_BUNDLE_DESCRIPTOR_SHA256


def parse_matched_v3_local_source_bundle_descriptor(raw: bytes) -> dict[str, Any]:
    """Parse only the exact frozen bundle descriptor."""

    value = _strict_json_load(raw, maximum_bytes=_MAX_RECEIPT_BYTES)
    if not _exact_json_equal(value, _descriptor()) or not hmac.compare_digest(
        hashlib.sha256(raw).hexdigest(), LOCAL_SOURCE_BUNDLE_DESCRIPTOR_SHA256
    ):
        raise ForagerMatchedV3LocalSourceBundleError("source bundle descriptor differs")
    return value


def parse_matched_v3_local_source_bundle_receipt(
    raw: bytes, *, expected_receipt_sha256: str
) -> dict[str, Any]:
    """Parse one canonical receipt with its caller-carried exact full digest."""

    expected = _require_sha256(expected_receipt_sha256, "expected source bundle receipt")
    if type(raw) is not bytes or not hmac.compare_digest(hashlib.sha256(raw).hexdigest(), expected):
        raise ForagerMatchedV3LocalSourceBundleError("source bundle receipt digest differs")
    value = _strict_json_load(raw, maximum_bytes=_MAX_RECEIPT_BYTES)
    _validate_receipt(value)
    return value


def verify_matched_v3_local_source_bundle_archive(
    *,
    descriptor: int,
    expected_archive_size_bytes: int,
    expected_archive_sha256: str,
    expected_receipt_bytes: bytes,
    expected_receipt_sha256: str,
    expected_source_snapshot_manifest_sha256: str,
    expected_source_snapshot_tree_sha256: str,
) -> dict[str, Any]:
    """Independently replay an exact receipt-bound USTAR descriptor.

    Verification grants no capability or authority and does not close the
    caller-owned descriptor.
    """

    if type(descriptor) is not int or descriptor < 0:
        raise ForagerMatchedV3LocalSourceBundleError("source bundle descriptor is invalid")
    archive_sha256 = _require_sha256(expected_archive_sha256, "expected source bundle archive")
    manifest_sha256 = _require_sha256(
        expected_source_snapshot_manifest_sha256, "expected source snapshot manifest"
    )
    tree_sha256 = _require_sha256(
        expected_source_snapshot_tree_sha256, "expected source snapshot tree"
    )
    receipt = parse_matched_v3_local_source_bundle_receipt(
        expected_receipt_bytes,
        expected_receipt_sha256=expected_receipt_sha256,
    )
    archive = cast(dict[str, Any], receipt["archive"])
    source = cast(dict[str, Any], receipt["source_snapshot"])
    if (
        archive["size_bytes"] != expected_archive_size_bytes
        or archive["sha256"] != archive_sha256
        or source["manifest_sha256"] != manifest_sha256
        or source["tree_sha256"] != tree_sha256
    ):
        raise ForagerMatchedV3LocalSourceBundleError(
            "source bundle caller-carried identities differ from the receipt"
        )
    _verify_archive_fd(
        descriptor,
        expected_size=expected_archive_size_bytes,
        expected_sha256=archive_sha256,
        receipt=receipt,
    )
    return receipt


@contextmanager
def _retain_matched_v3_local_source_bundle(
    *,
    repository_root: Path,
    expected_canonical_snapshot_manifest_bytes: bytes,
    expected_snapshot_manifest_sha256: str,
    expected_snapshot_tree_sha256: str,
) -> Iterator[RetainedMatchedV3LocalSourceBundle]:
    snapshot = _validate_expected_snapshot(
        expected_canonical_snapshot_manifest_bytes,
        expected_manifest_sha256=expected_snapshot_manifest_sha256,
        expected_tree_sha256=expected_snapshot_tree_sha256,
    )
    expected_tree = _expected_tree_inventory(snapshot)
    anchored: _AnchoredRoot | None = None
    writable = -1
    readonly = -1
    retained: RetainedMatchedV3LocalSourceBundle | None = None
    failure: BaseException | None = None
    try:
        anchored = _open_anchored_repository_root(repository_root)
        root_before = os.fstat(anchored.root_descriptor)
        observed_before = _measure_tree(anchored.root_descriptor)
        anchored.verify()
        _require_expected_tree(observed_before, expected_tree, "pre-archive")
        writable = _create_private_memfd()
        records = cast(list[dict[str, Any]], expected_tree["files"])
        archive_size, archive_sha256 = _write_archive(writable, anchored.root_descriptor, records)
        receipt_bytes, receipt_sha256 = _build_receipt(
            snapshot,
            archive_size_bytes=archive_size,
            archive_sha256=archive_sha256,
        )
        receipt = parse_matched_v3_local_source_bundle_receipt(
            receipt_bytes, expected_receipt_sha256=receipt_sha256
        )
        _verify_archive_fd(
            writable,
            expected_size=archive_size,
            expected_sha256=archive_sha256,
            receipt=receipt,
        )
        observed_after = _measure_tree(anchored.root_descriptor)
        anchored.verify()
        root_after = os.fstat(anchored.root_descriptor)
        _require_expected_tree(observed_after, expected_tree, "post-archive")
        if _stat_identity(root_before) != _stat_identity(root_after):
            raise ForagerMatchedV3LocalSourceBundleError(
                "repository root changed across source bundling"
            )
        readonly = _seal_and_reopen_readonly(writable, expected_size=archive_size)
        closing_writable = writable
        writable = -1
        _close_untransferred_descriptor(
            closing_writable,
            failure=None,
            label="writable source bundle descriptor",
        )
        verify_matched_v3_local_source_bundle_archive(
            descriptor=readonly,
            expected_archive_size_bytes=archive_size,
            expected_archive_sha256=archive_sha256,
            expected_receipt_bytes=receipt_bytes,
            expected_receipt_sha256=receipt_sha256,
            expected_source_snapshot_manifest_sha256=snapshot.manifest_sha256,
            expected_source_snapshot_tree_sha256=snapshot.tree_sha256,
        )
        metadata = os.fstat(readonly)
        retained = RetainedMatchedV3LocalSourceBundle(
            _CAPABILITY_CREATION_TOKEN,
            readonly,
            metadata.st_dev,
            metadata.st_ino,
            archive_size,
            archive_sha256,
            receipt_bytes,
            receipt_sha256,
            snapshot.manifest_sha256,
            snapshot.tree_sha256,
        )
        readonly = -1
        retained.reverify()
        closing_anchored = anchored
        anchored = None
        closing_anchored.close()
        del expected_canonical_snapshot_manifest_bytes
        del snapshot
        del expected_tree
        del observed_before
        del records
        del receipt
        del observed_after
        del receipt_bytes
        yield retained
    except BaseException as exc:
        failure = exc
        raise
    finally:
        cleanup_failure = failure
        if retained is not None:
            closing_retained = retained
            retained = None
            try:
                closing_retained.close()
            except BaseException as cleanup_error:
                cleanup_failure = _merge_cleanup_failure(
                    cleanup_failure,
                    cleanup_error,
                    label="closing the retained source bundle",
                )
        if readonly >= 0:
            closing_readonly = readonly
            readonly = -1
            try:
                os.close(closing_readonly)
            except BaseException as cleanup_error:
                cleanup_failure = _merge_cleanup_failure(
                    cleanup_failure,
                    cleanup_error,
                    label="closing the untransferred read-only source bundle descriptor",
                )
        if writable >= 0:
            closing_writable = writable
            writable = -1
            try:
                os.close(closing_writable)
            except BaseException as cleanup_error:
                cleanup_failure = _merge_cleanup_failure(
                    cleanup_failure,
                    cleanup_error,
                    label="closing the untransferred writable source bundle descriptor",
                )
        if anchored is not None:
            closing_anchored = anchored
            anchored = None
            try:
                closing_anchored.close()
            except BaseException as cleanup_error:
                cleanup_failure = _merge_cleanup_failure(
                    cleanup_failure,
                    cleanup_error,
                    label="closing the anchored repository root",
                )
        if failure is None and cleanup_failure is not None:
            raise cleanup_failure


def retain_matched_v3_local_source_bundle(
    *,
    repository_root: Path,
    expected_canonical_snapshot_manifest_bytes: bytes,
    expected_snapshot_manifest_sha256: str,
    expected_snapshot_tree_sha256: str,
) -> AbstractContextManager[RetainedMatchedV3LocalSourceBundle]:
    """Produce one sealed exact-source USTAR within a bounded context.

    Every argument is explicit and caller pinned.  The returned capability has
    no extraction, publication, image-build, execution, or qualification method.
    """

    return _retain_matched_v3_local_source_bundle(
        repository_root=repository_root,
        expected_canonical_snapshot_manifest_bytes=expected_canonical_snapshot_manifest_bytes,
        expected_snapshot_manifest_sha256=expected_snapshot_manifest_sha256,
        expected_snapshot_tree_sha256=expected_snapshot_tree_sha256,
    )


__all__ = [
    "ForagerMatchedV3LocalSourceBundleError",
    "LOCAL_SOURCE_BUNDLE_DESCRIPTOR_SCHEMA_VERSION",
    "LOCAL_SOURCE_BUNDLE_DESCRIPTOR_SHA256",
    "LOCAL_SOURCE_BUNDLE_RECEIPT_SCHEMA_VERSION",
    "LOCAL_SOURCE_BUNDLE_STATUS",
    "RetainedMatchedV3LocalSourceBundle",
    "canonical_matched_v3_local_source_bundle_descriptor_bytes",
    "matched_v3_local_source_bundle_descriptor",
    "matched_v3_local_source_bundle_descriptor_sha256",
    "parse_matched_v3_local_source_bundle_descriptor",
    "parse_matched_v3_local_source_bundle_receipt",
    "retain_matched_v3_local_source_bundle",
    "verify_matched_v3_local_source_bundle_archive",
]
