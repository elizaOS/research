"""Atomic, content-only publication of matched-v3 adapter reward bundles.

The adapter runners prove completion only through process-local capabilities that are
deliberately absent from their serialized receipts.  This module therefore publishes and
reloads structural bundle content without reconstructing execution, qualification,
campaign-ingestion, evidence, promotion, or performance authority.

Publication is a Linux-local durability operation: a fully replayed sibling staging
directory is fsynced, moved with ``renameat2(RENAME_NOREPLACE)``, fsynced through its held
parent descriptor, and replayed again.  A caller-supplied full-file SHA-256 for
``publication.json`` is required when loading so downstream code must carry an external
content pin rather than trusting the manifest's self-digest.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import hmac
import json
import os
import re
import secrets
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, NoReturn, cast

from alberta_framework.benchmarks import _forager_matched_v3_scorer as scorer
from alberta_framework.benchmarks import (
    forager_matched_v3_adapter_reward_bundle as reward_bundle,
)

ADAPTER_REWARD_PUBLICATION_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.adapter_reward_publication_descriptor.v1"
)
ADAPTER_REWARD_PUBLICATION_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.adapter_reward_publication.v1"
)
ADAPTER_REWARD_PUBLICATION_STATUS: Final = "implemented_unexecuted"

PUBLICATION_MANIFEST_FILENAME: Final = "publication.json"
PUBLICATION_FILENAME: Final = PUBLICATION_MANIFEST_FILENAME
ADAPTER_BUNDLE_MANIFEST_FILENAME: Final = "adapter-bundle-manifest.json"
RUNNER_RESULT_RECEIPT_FILENAME: Final = "runner-result-receipt.json"
REWARD_TRACE_FILENAME: Final = "reward-trace.npz"
SCORE_RECEIPT_FILENAME: Final = "score-receipt.json"

_ADAPTER_BUNDLE_SOURCE_PATH: Final = (
    "alberta_framework/benchmarks/forager_matched_v3_adapter_reward_bundle.py"
)
_ADAPTER_BUNDLE_SOURCE_SHA256: Final = (
    "22199838219cfb5610d83fb71cb828f087b1a4754132f1c325388571e8aa2469"
)
_WRITER_CONTRACT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.adapter_reward_publication_writer.linux.v1"
)

_DIRECTORY_MODE: Final = 0o700
_FILE_MODE: Final = 0o600
_MAX_PUBLICATION_BYTES: Final = 64 * 1024
_MAX_ADAPTER_MANIFEST_BYTES: Final = 256 * 1024
_MAX_RUNNER_RECEIPT_BYTES: Final = 4 * 1024 * 1024
_MAX_SCORE_RECEIPT_BYTES: Final = 64 * 1024
_MAX_ROOT_ENTRIES: Final = 5
_MAX_JSON_NODES: Final = 4_096
_MAX_JSON_DEPTH: Final = 32
_MAX_JSON_INTEGER_DIGITS: Final = 16
_STAGING_ATTEMPTS: Final = 64
_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_PORTABLE_NAME_RE: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")

_FILE_PATHS: Final = MappingProxyType(
    {
        "adapter_bundle_manifest": ADAPTER_BUNDLE_MANIFEST_FILENAME,
        "runner_result_receipt": RUNNER_RESULT_RECEIPT_FILENAME,
        "reward_trace": REWARD_TRACE_FILENAME,
        "score_receipt": SCORE_RECEIPT_FILENAME,
    }
)


class ForagerMatchedV3AdapterRewardPublicationError(ValueError):
    """Publication content, filesystem structure, or durability failed closed."""


class PublishedAdapterRewardPublicationUncertainError(
    ForagerMatchedV3AdapterRewardPublicationError
):
    """The destination became visible, but final durability or replay is uncertain."""

    def __init__(
        self,
        destination: Path,
        detail: str,
        *,
        publication_file_sha256: str,
        publication_body_sha256: str,
    ) -> None:
        self.destination = destination
        self.publication_file_sha256 = publication_file_sha256
        self.publication_body_sha256 = publication_body_sha256
        super().__init__(f"adapter reward publication at {destination}, but {detail}")


@dataclass(frozen=True, slots=True)
class ContentVerifiedAdapterRewardPublication:
    """Structurally replayed persisted bytes with no execution or ingestion authority."""

    output_root: Path
    candidate_id: str
    publication_file_sha256: str
    publication_body_sha256: str
    manifest: Mapping[str, Any]
    bundle: reward_bundle.MatchedV3AdapterRewardBundle


@dataclass(frozen=True, slots=True)
class _OpenDirectory:
    path: Path
    descriptor: int
    inode_identity: tuple[int, int, int]


def _source_sha256(module_file: object, expected_suffix: str) -> str:
    if type(module_file) is not str or not module_file.endswith(expected_suffix):
        raise RuntimeError(f"cannot resolve exact source path for {expected_suffix}")
    try:
        raw = Path(module_file).read_bytes()
    except OSError as exc:
        raise RuntimeError(f"cannot read exact source bytes for {expected_suffix}") from exc
    return hashlib.sha256(raw).hexdigest()


if not hmac.compare_digest(
    _source_sha256(reward_bundle.__file__, _ADAPTER_BUNDLE_SOURCE_PATH),
    _ADAPTER_BUNDLE_SOURCE_SHA256,
):
    raise RuntimeError("adapter reward publication dependency source binding drifted")


def _claims() -> dict[str, bool]:
    return {
        "authority_granted": False,
        "campaign_ingestion_authorized": False,
        "evidence_authority": False,
        "execution_authorized": False,
        "execution_ready": False,
        "performance_claim_allowed": False,
        "production_result_accepted": False,
        "runtime_qualified": False,
        "scientific_promotion_allowed": False,
        "universal_sota_claim_allowed": False,
    }


def _limitations() -> list[str]:
    return [
        "The runner's live process completion capability is not serialized or reconstructed.",
        "Content hashes and structural replay do not independently prove agent execution.",
        "Atomic publication is a writer contract, not execution or scientific attestation.",
        "Seed provenance and source/runtime qualification remain unverified.",
        "Loading grants no campaign ingestion, evidence, promotion, or performance authority.",
    ]


def _writer_contract() -> dict[str, Any]:
    return {
        "schema_version": _WRITER_CONTRACT_SCHEMA_VERSION,
        "layout": "one_flat_directory_with_five_exact_files",
        "staging_directory_mode": "0700",
        "artifact_file_mode": "0600",
        "exclusive_move": "renameat2_RENAME_NOREPLACE",
        "staged_files_fsynced": True,
        "staging_directory_fsynced": True,
        "publication_parent_fsynced": True,
        "publication_parent_owned_by_effective_uid": True,
        "publication_parent_group_or_world_writable": False,
        "structural_replay_before_and_after_move": True,
        "writer_contract_independently_attests_execution": False,
    }


def _descriptor() -> dict[str, Any]:
    return {
        "schema_version": ADAPTER_REWARD_PUBLICATION_DESCRIPTOR_SCHEMA_VERSION,
        "status": ADAPTER_REWARD_PUBLICATION_STATUS,
        "classification": "durable_content_publication_non_authorizing",
        "dependency": {
            "source_path": _ADAPTER_BUNDLE_SOURCE_PATH,
            "source_sha256": _ADAPTER_BUNDLE_SOURCE_SHA256,
            "descriptor_schema_version": (
                reward_bundle.ADAPTER_REWARD_BUNDLE_DESCRIPTOR_SCHEMA_VERSION
            ),
            "descriptor_sha256": (
                reward_bundle.ADAPTER_REWARD_BUNDLE_DESCRIPTOR_SHA256
            ),
            "manifest_schema_version": (
                reward_bundle.ADAPTER_REWARD_BUNDLE_MANIFEST_SCHEMA_VERSION
            ),
        },
        "candidate_consumers": ["adapted_full_rainbow", "adapted_ppo_gru"],
        "exact_files": {
            "publication_manifest": PUBLICATION_MANIFEST_FILENAME,
            **dict(_FILE_PATHS),
        },
        "bounds": {
            "root_entries": _MAX_ROOT_ENTRIES,
            "publication_manifest_bytes": _MAX_PUBLICATION_BYTES,
            "adapter_bundle_manifest_bytes": _MAX_ADAPTER_MANIFEST_BYTES,
            "runner_result_receipt_bytes": _MAX_RUNNER_RECEIPT_BYTES,
            "reward_trace_bytes": scorer.CANONICAL_NPZ_SIZE_BYTES,
            "score_receipt_bytes": _MAX_SCORE_RECEIPT_BYTES,
        },
        "filesystem_contract": _writer_contract(),
        "loading": {
            "caller_supplied_publication_file_sha256_required": True,
            "exact_inventory_required": True,
            "regular_single_link_files_required": True,
            "stable_descriptor_relative_reads_required": True,
            "adapter_bundle_structural_replay_required": True,
            "live_capability_reconstruction": False,
        },
        "claims": _claims(),
        "limitations": _limitations(),
    }


def _canonical_json(value: object, *, label: str, maximum: int) -> bytes:
    try:
        raw = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (OverflowError, RecursionError, TypeError, UnicodeError, ValueError) as exc:
        raise ForagerMatchedV3AdapterRewardPublicationError(
            f"{label} is not finite canonical JSON"
        ) from exc
    if not raw or len(raw) > maximum:
        raise ForagerMatchedV3AdapterRewardPublicationError(
            f"{label} violates its canonical byte bound"
        )
    return raw


_DESCRIPTOR_BYTES: Final = _canonical_json(
    _descriptor(),
    label="adapter reward publication descriptor",
    maximum=_MAX_PUBLICATION_BYTES,
)
ADAPTER_REWARD_PUBLICATION_DESCRIPTOR_SHA256: Final = (
    "5ca0f236a7b6ac58a67578282ca2091f1a443a72502c81fe08b2ecf850ec7905"
)
if not hmac.compare_digest(
    hashlib.sha256(_DESCRIPTOR_BYTES).hexdigest(),
    ADAPTER_REWARD_PUBLICATION_DESCRIPTOR_SHA256,
):
    raise RuntimeError("adapter reward publication descriptor identity drifted")


def adapter_reward_publication_descriptor() -> dict[str, Any]:
    """Return a detached copy of the exact publication descriptor."""

    return cast(dict[str, Any], json.loads(_DESCRIPTOR_BYTES.decode("ascii")))


def canonical_adapter_reward_publication_descriptor_bytes() -> bytes:
    """Return the exact frozen publication descriptor bytes."""

    return bytes(_DESCRIPTOR_BYTES)


def parse_adapter_reward_publication_descriptor(raw: bytes) -> dict[str, Any]:
    """Accept only the exact frozen publication descriptor bytes."""

    if type(raw) is not bytes or raw != _DESCRIPTOR_BYTES:
        raise ForagerMatchedV3AdapterRewardPublicationError(
            "adapter reward publication descriptor bytes differ"
        )
    if not hmac.compare_digest(
        hashlib.sha256(raw).hexdigest(),
        ADAPTER_REWARD_PUBLICATION_DESCRIPTOR_SHA256,
    ):
        raise ForagerMatchedV3AdapterRewardPublicationError(
            "adapter reward publication descriptor digest differs"
        )
    return adapter_reward_publication_descriptor()


def _require_exact_keys(
    value: Mapping[str, Any], expected: frozenset[str], label: str
) -> None:
    actual = set(value)
    if actual != expected:
        raise ForagerMatchedV3AdapterRewardPublicationError(
            f"{label} fields differ; missing={sorted(expected - actual)!r}, "
            f"extra={sorted(actual - expected)!r}"
        )


def _require_object(value: object, label: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise ForagerMatchedV3AdapterRewardPublicationError(
            f"{label} must be a plain object"
        )
    return cast(dict[str, Any], value)


def _require_sha256(value: object, label: str) -> str:
    if (
        type(value) is not str
        or _SHA256_RE.fullmatch(value) is None
        or value == "0" * 64
    ):
        raise ForagerMatchedV3AdapterRewardPublicationError(
            f"{label} must be a lowercase SHA-256 digest"
        )
    return value


def _require_exact_int(
    value: object,
    label: str,
    *,
    minimum: int,
    maximum: int,
) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ForagerMatchedV3AdapterRewardPublicationError(
            f"{label} must be an exact integer in [{minimum}, {maximum}]"
        )
    return value


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ForagerMatchedV3AdapterRewardPublicationError(
                f"duplicate JSON key {key!r} is forbidden"
            )
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> NoReturn:
    raise ForagerMatchedV3AdapterRewardPublicationError(
        f"non-finite JSON number {value!r} is forbidden"
    )


def _reject_float(value: str) -> NoReturn:
    raise ForagerMatchedV3AdapterRewardPublicationError(
        f"floating-point JSON number {value!r} is forbidden"
    )


def _bounded_json_integer(value: str) -> int:
    digits = value[1:] if value.startswith("-") else value
    if not digits or len(digits) > _MAX_JSON_INTEGER_DIGITS:
        raise ForagerMatchedV3AdapterRewardPublicationError(
            "JSON integer exceeds the publication digit bound"
        )
    return int(value)


def _validate_json_complexity(value: object, label: str) -> None:
    pending: list[tuple[object, int]] = [(value, 0)]
    nodes = 0
    while pending:
        item, depth = pending.pop()
        nodes += 1
        if nodes > _MAX_JSON_NODES:
            raise ForagerMatchedV3AdapterRewardPublicationError(
                f"{label} exceeds the JSON node bound"
            )
        if depth > _MAX_JSON_DEPTH:
            raise ForagerMatchedV3AdapterRewardPublicationError(
                f"{label} exceeds the JSON depth bound"
            )
        if type(item) is dict:
            pending.extend(
                (child, depth + 1)
                for child in cast(dict[str, Any], item).values()
            )
        elif type(item) is list:
            pending.extend((child, depth + 1) for child in item)


def _decode_canonical_json(raw: bytes, *, label: str, maximum: int) -> dict[str, Any]:
    if type(raw) is not bytes or not raw or len(raw) > maximum:
        raise ForagerMatchedV3AdapterRewardPublicationError(
            f"{label} violates its JSON byte bound"
        )
    try:
        decoded = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
            parse_float=_reject_float,
            parse_int=_bounded_json_integer,
        )
    except ForagerMatchedV3AdapterRewardPublicationError:
        raise
    except (OverflowError, RecursionError, TypeError, UnicodeError, ValueError) as exc:
        raise ForagerMatchedV3AdapterRewardPublicationError(
            f"{label} is not strict ASCII JSON"
        ) from exc
    _validate_json_complexity(decoded, label)
    payload = _require_object(decoded, label)
    if _canonical_json(payload, label=label, maximum=maximum) != raw:
        raise ForagerMatchedV3AdapterRewardPublicationError(
            f"{label} is not canonical JSON"
        )
    return payload


def _payload_bytes(
    bundle: reward_bundle.MatchedV3AdapterRewardBundle,
) -> dict[str, bytes]:
    return {
        "adapter_bundle_manifest": bundle.manifest_bytes,
        "runner_result_receipt": bundle.runner_receipt_bytes,
        "reward_trace": bundle.reward_artifact_bytes,
        "score_receipt": bundle.score_receipt_bytes,
    }


def _file_records(
    bundle: reward_bundle.MatchedV3AdapterRewardBundle,
) -> dict[str, dict[str, Any]]:
    payloads = _payload_bytes(bundle)
    return {
        role: {
            "path": _FILE_PATHS[role],
            "role": role,
            "sha256": hashlib.sha256(payloads[role]).hexdigest(),
            "size_bytes": len(payloads[role]),
        }
        for role in _FILE_PATHS
    }


def _publication_body(
    bundle: reward_bundle.MatchedV3AdapterRewardBundle,
) -> dict[str, Any]:
    return {
        "schema_version": ADAPTER_REWARD_PUBLICATION_SCHEMA_VERSION,
        "classification": "durable_content_publication_non_authorizing",
        "candidate_id": bundle.candidate_id,
        "publication_descriptor": {
            "schema_version": ADAPTER_REWARD_PUBLICATION_DESCRIPTOR_SCHEMA_VERSION,
            "sha256": ADAPTER_REWARD_PUBLICATION_DESCRIPTOR_SHA256,
        },
        "adapter_reward_bundle": {
            "descriptor_schema_version": (
                reward_bundle.ADAPTER_REWARD_BUNDLE_DESCRIPTOR_SCHEMA_VERSION
            ),
            "descriptor_sha256": (
                reward_bundle.ADAPTER_REWARD_BUNDLE_DESCRIPTOR_SHA256
            ),
            "manifest_schema_version": (
                reward_bundle.ADAPTER_REWARD_BUNDLE_MANIFEST_SCHEMA_VERSION
            ),
            "implementation_path": _ADAPTER_BUNDLE_SOURCE_PATH,
            "implementation_source_sha256": _ADAPTER_BUNDLE_SOURCE_SHA256,
            "manifest_body_sha256": bundle.manifest_sha256,
            "manifest_file_sha256": hashlib.sha256(bundle.manifest_bytes).hexdigest(),
        },
        "files": _file_records(bundle),
        "writer_contract": _writer_contract(),
        "claims": _claims(),
        "limitations": _limitations(),
    }


def _publication_manifest_bytes(
    bundle: reward_bundle.MatchedV3AdapterRewardBundle,
) -> tuple[bytes, str, str]:
    body = _publication_body(bundle)
    body_bytes = _canonical_json(
        body,
        label="adapter reward publication body",
        maximum=_MAX_PUBLICATION_BYTES,
    )
    body_sha256 = hashlib.sha256(body_bytes).hexdigest()
    payload = dict(body)
    payload["publication_body_sha256"] = body_sha256
    raw = _canonical_json(
        payload,
        label="adapter reward publication manifest",
        maximum=_MAX_PUBLICATION_BYTES,
    )
    return raw, body_sha256, hashlib.sha256(raw).hexdigest()


def _validate_file_records(value: object) -> None:
    files = _require_object(value, "publication files")
    _require_exact_keys(files, frozenset(_FILE_PATHS), "publication files")
    bounds = {
        "adapter_bundle_manifest": (1, _MAX_ADAPTER_MANIFEST_BYTES),
        "runner_result_receipt": (1, _MAX_RUNNER_RECEIPT_BYTES),
        "reward_trace": (
            scorer.CANONICAL_NPZ_SIZE_BYTES,
            scorer.CANONICAL_NPZ_SIZE_BYTES,
        ),
        "score_receipt": (1, _MAX_SCORE_RECEIPT_BYTES),
    }
    for role, path in _FILE_PATHS.items():
        record = _require_object(files[role], f"publication file {role}")
        _require_exact_keys(
            record,
            frozenset({"path", "role", "sha256", "size_bytes"}),
            f"publication file {role}",
        )
        if record["path"] != path or record["role"] != role:
            raise ForagerMatchedV3AdapterRewardPublicationError(
                f"publication file {role} path or role drifted"
            )
        _require_sha256(record["sha256"], f"publication file {role} digest")
        minimum, maximum = bounds[role]
        _require_exact_int(
            record["size_bytes"],
            f"publication file {role} size",
            minimum=minimum,
            maximum=maximum,
        )


def _validate_publication_body(body: dict[str, Any]) -> None:
    _require_exact_keys(
        body,
        frozenset(
            {
                "schema_version",
                "classification",
                "candidate_id",
                "publication_descriptor",
                "adapter_reward_bundle",
                "files",
                "writer_contract",
                "claims",
                "limitations",
            }
        ),
        "adapter reward publication body",
    )
    if (
        body["schema_version"] != ADAPTER_REWARD_PUBLICATION_SCHEMA_VERSION
        or body["classification"] != "durable_content_publication_non_authorizing"
        or body["candidate_id"]
        not in {"adapted_full_rainbow", "adapted_ppo_gru"}
    ):
        raise ForagerMatchedV3AdapterRewardPublicationError(
            "adapter reward publication identity drifted"
        )
    descriptor = _require_object(
        body["publication_descriptor"], "publication descriptor binding"
    )
    if descriptor != {
        "schema_version": ADAPTER_REWARD_PUBLICATION_DESCRIPTOR_SCHEMA_VERSION,
        "sha256": ADAPTER_REWARD_PUBLICATION_DESCRIPTOR_SHA256,
    }:
        raise ForagerMatchedV3AdapterRewardPublicationError(
            "publication descriptor binding drifted"
        )
    dependency = _require_object(
        body["adapter_reward_bundle"], "adapter reward bundle binding"
    )
    _require_exact_keys(
        dependency,
        frozenset(
            {
                "descriptor_schema_version",
                "descriptor_sha256",
                "manifest_schema_version",
                "implementation_path",
                "implementation_source_sha256",
                "manifest_body_sha256",
                "manifest_file_sha256",
            }
        ),
        "adapter reward bundle binding",
    )
    fixed_dependency = {
        "descriptor_schema_version": (
            reward_bundle.ADAPTER_REWARD_BUNDLE_DESCRIPTOR_SCHEMA_VERSION
        ),
        "descriptor_sha256": reward_bundle.ADAPTER_REWARD_BUNDLE_DESCRIPTOR_SHA256,
        "manifest_schema_version": (
            reward_bundle.ADAPTER_REWARD_BUNDLE_MANIFEST_SCHEMA_VERSION
        ),
        "implementation_path": _ADAPTER_BUNDLE_SOURCE_PATH,
        "implementation_source_sha256": _ADAPTER_BUNDLE_SOURCE_SHA256,
    }
    if any(dependency[key] != expected for key, expected in fixed_dependency.items()):
        raise ForagerMatchedV3AdapterRewardPublicationError(
            "adapter reward bundle fixed binding drifted"
        )
    _require_sha256(
        dependency["manifest_body_sha256"], "adapter bundle manifest body digest"
    )
    _require_sha256(
        dependency["manifest_file_sha256"], "adapter bundle manifest file digest"
    )
    _validate_file_records(body["files"])
    files = cast(dict[str, Any], body["files"])
    adapter_manifest_record = cast(
        dict[str, Any], files["adapter_bundle_manifest"]
    )
    if dependency["manifest_file_sha256"] != adapter_manifest_record["sha256"]:
        raise ForagerMatchedV3AdapterRewardPublicationError(
            "adapter bundle manifest digest bindings disagree"
        )
    writer_contract = _require_object(
        body["writer_contract"], "adapter reward publication writer contract"
    )
    expected_writer_contract = _writer_contract()
    _require_exact_keys(
        writer_contract,
        frozenset(expected_writer_contract),
        "adapter reward publication writer contract",
    )
    if any(
        type(writer_contract[key]) is not type(expected)
        or writer_contract[key] != expected
        for key, expected in expected_writer_contract.items()
    ):
        raise ForagerMatchedV3AdapterRewardPublicationError(
            "adapter reward publication writer contract drifted"
        )
    claims = _require_object(body["claims"], "publication claims")
    _require_exact_keys(claims, frozenset(_claims()), "publication claims")
    if any(value is not False for value in claims.values()):
        raise ForagerMatchedV3AdapterRewardPublicationError(
            "publication claims must be exact false booleans"
        )
    limitations = body["limitations"]
    if (
        type(limitations) is not list
        or any(type(item) is not str for item in limitations)
        or limitations != _limitations()
    ):
        raise ForagerMatchedV3AdapterRewardPublicationError(
            "publication limitations drifted"
        )


def parse_adapter_reward_publication_manifest(
    raw: bytes,
    *,
    expected_publication_file_sha256: str,
) -> dict[str, Any]:
    """Parse a canonical outer manifest under a required external file digest."""

    expected = _require_sha256(
        expected_publication_file_sha256,
        "expected publication manifest file digest",
    )
    if type(raw) is not bytes or not hmac.compare_digest(
        hashlib.sha256(raw).hexdigest(), expected
    ):
        raise ForagerMatchedV3AdapterRewardPublicationError(
            "publication manifest does not match its external file digest"
        )
    payload = _decode_canonical_json(
        raw,
        label="adapter reward publication manifest",
        maximum=_MAX_PUBLICATION_BYTES,
    )
    _require_exact_keys(
        payload,
        frozenset(
            {
                "schema_version",
                "classification",
                "candidate_id",
                "publication_descriptor",
                "adapter_reward_bundle",
                "files",
                "writer_contract",
                "claims",
                "limitations",
                "publication_body_sha256",
            }
        ),
        "adapter reward publication manifest",
    )
    supplied_body_sha256 = _require_sha256(
        payload["publication_body_sha256"], "publication body digest"
    )
    body = dict(payload)
    del body["publication_body_sha256"]
    body_bytes = _canonical_json(
        body,
        label="adapter reward publication body",
        maximum=_MAX_PUBLICATION_BYTES,
    )
    if not hmac.compare_digest(
        hashlib.sha256(body_bytes).hexdigest(), supplied_body_sha256
    ):
        raise ForagerMatchedV3AdapterRewardPublicationError(
            "publication body digest does not replay"
        )
    _validate_publication_body(body)
    return payload


def _freeze_json(value: Any) -> Any:
    if type(value) is dict:
        return MappingProxyType(
            {key: _freeze_json(item) for key, item in cast(dict[str, Any], value).items()}
        )
    if type(value) is list:
        return tuple(_freeze_json(item) for item in value)
    return value


def _stat_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _inode_identity(metadata: os.stat_result) -> tuple[int, int, int]:
    return metadata.st_dev, metadata.st_ino, stat.S_IFMT(metadata.st_mode)


def _directory_open_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )


def _file_read_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )


def _close_no_raise(descriptor: int) -> None:
    """Attempt one close without masking a primary result or exception.

    All writable content and containing directories are explicitly fsynced before
    publication.  A later close error on these read-only/directory descriptors therefore
    affects process resource cleanup, not the already verified content commitment.  POSIX
    also leaves descriptor state unspecified after an interrupted close, so retrying could
    close a reused descriptor.
    """

    try:
        os.close(descriptor)
    except OSError:
        pass


def _open_stable_directory(
    path: Path,
    label: str,
    *,
    required_mode: int | None = None,
) -> _OpenDirectory:
    try:
        path_metadata = path.lstat()
        canonical = path.resolve(strict=True)
        descriptor = os.open(canonical, _directory_open_flags())
    except (OSError, RuntimeError) as exc:
        raise ForagerMatchedV3AdapterRewardPublicationError(
            f"cannot safely open {label}"
        ) from exc
    try:
        opened = os.fstat(descriptor)
        current = os.lstat(canonical)
        identity = _inode_identity(opened)
        if (
            not stat.S_ISDIR(path_metadata.st_mode)
            or not stat.S_ISDIR(opened.st_mode)
            or path.is_symlink()
            or _inode_identity(path_metadata) != identity
            or _inode_identity(current) != identity
            or (
                required_mode is not None
                and stat.S_IMODE(opened.st_mode) != required_mode
            )
        ):
            raise ForagerMatchedV3AdapterRewardPublicationError(
                f"{label} changed or has an unsafe mode"
            )
        return _OpenDirectory(canonical, descriptor, identity)
    except ForagerMatchedV3AdapterRewardPublicationError:
        _close_no_raise(descriptor)
        raise
    except OSError as exc:
        _close_no_raise(descriptor)
        raise ForagerMatchedV3AdapterRewardPublicationError(
            f"cannot safely verify {label}"
        ) from exc
    except BaseException:
        _close_no_raise(descriptor)
        raise


def _open_stable_directory_at(
    parent: _OpenDirectory,
    name: str,
    path: Path,
    label: str,
    *,
    required_mode: int | None,
) -> _OpenDirectory:
    try:
        path_metadata = os.stat(name, dir_fd=parent.descriptor, follow_symlinks=False)
        descriptor = os.open(name, _directory_open_flags(), dir_fd=parent.descriptor)
    except OSError as exc:
        raise ForagerMatchedV3AdapterRewardPublicationError(
            f"cannot safely open {label}"
        ) from exc
    try:
        opened = os.fstat(descriptor)
        current = os.stat(name, dir_fd=parent.descriptor, follow_symlinks=False)
        identity = _inode_identity(opened)
        if (
            not stat.S_ISDIR(path_metadata.st_mode)
            or not stat.S_ISDIR(opened.st_mode)
            or _inode_identity(path_metadata) != identity
            or _inode_identity(current) != identity
            or (
                required_mode is not None
                and stat.S_IMODE(opened.st_mode) != required_mode
            )
        ):
            raise ForagerMatchedV3AdapterRewardPublicationError(
                f"{label} changed or has an unsafe mode"
            )
        return _OpenDirectory(path, descriptor, identity)
    except ForagerMatchedV3AdapterRewardPublicationError:
        _close_no_raise(descriptor)
        raise
    except OSError as exc:
        _close_no_raise(descriptor)
        raise ForagerMatchedV3AdapterRewardPublicationError(
            f"cannot safely verify {label}"
        ) from exc
    except BaseException:
        _close_no_raise(descriptor)
        raise


def _assert_open_directory_path(root: _OpenDirectory, label: str) -> None:
    try:
        opened = os.fstat(root.descriptor)
        current = os.lstat(root.path)
    except OSError as exc:
        raise ForagerMatchedV3AdapterRewardPublicationError(
            f"{label} is no longer reachable"
        ) from exc
    if (
        not stat.S_ISDIR(opened.st_mode)
        or not stat.S_ISDIR(current.st_mode)
        or root.path.is_symlink()
        or _inode_identity(opened) != root.inode_identity
        or _inode_identity(current) != root.inode_identity
    ):
        raise ForagerMatchedV3AdapterRewardPublicationError(
            f"{label} no longer names the opened inode"
        )


def _root_inventory(root: _OpenDirectory) -> dict[str, tuple[int, ...]]:
    opened_root = os.fstat(root.descriptor)
    if (
        not stat.S_ISDIR(opened_root.st_mode)
        or stat.S_IMODE(opened_root.st_mode) != _DIRECTORY_MODE
    ):
        raise ForagerMatchedV3AdapterRewardPublicationError(
            "adapter publication root mode or type is unsafe"
        )
    expected = {PUBLICATION_MANIFEST_FILENAME, *_FILE_PATHS.values()}
    names: set[str] = set()
    inventory: dict[str, tuple[int, ...]] = {}
    try:
        iterator = os.scandir(root.descriptor)
    except OSError as exc:
        raise ForagerMatchedV3AdapterRewardPublicationError(
            "cannot enumerate adapter publication root"
        ) from exc
    with iterator:
        for entry in iterator:
            if len(names) >= _MAX_ROOT_ENTRIES:
                raise ForagerMatchedV3AdapterRewardPublicationError(
                    "adapter publication root exceeds its entry bound"
                )
            if entry.name in names:
                raise ForagerMatchedV3AdapterRewardPublicationError(
                    "adapter publication root repeats an entry"
                )
            names.add(entry.name)
            try:
                metadata = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise ForagerMatchedV3AdapterRewardPublicationError(
                    "cannot inspect adapter publication entry"
                ) from exc
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or stat.S_IMODE(metadata.st_mode) != _FILE_MODE
                or (metadata.st_uid, metadata.st_gid)
                != (opened_root.st_uid, opened_root.st_gid)
            ):
                raise ForagerMatchedV3AdapterRewardPublicationError(
                    "adapter publication contains a link, special file, unsafe mode, "
                    "or owner mismatch"
                )
            inventory[entry.name] = _stat_identity(metadata)
    if names != expected:
        raise ForagerMatchedV3AdapterRewardPublicationError(
            "adapter publication inventory differs; "
            f"missing={sorted(expected - names)!r}, extra={sorted(names - expected)!r}"
        )
    return inventory


def _read_stable_regular_at(
    root: _OpenDirectory,
    name: str,
    label: str,
    *,
    maximum: int,
    exact_size: int | None = None,
) -> bytes:
    try:
        path_metadata = os.stat(name, dir_fd=root.descriptor, follow_symlinks=False)
        descriptor = os.open(name, _file_read_flags(), dir_fd=root.descriptor)
    except OSError as exc:
        raise ForagerMatchedV3AdapterRewardPublicationError(
            f"cannot safely open {label}"
        ) from exc
    try:
        before = os.fstat(descriptor)
        root_metadata = os.fstat(root.descriptor)
        valid_size = (
            before.st_size == exact_size
            if exact_size is not None
            else 0 < before.st_size <= maximum
        )
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or not valid_size
            or stat.S_IMODE(before.st_mode) != _FILE_MODE
            or (before.st_uid, before.st_gid)
            != (root_metadata.st_uid, root_metadata.st_gid)
            or _stat_identity(path_metadata) != _stat_identity(before)
        ):
            raise ForagerMatchedV3AdapterRewardPublicationError(
                f"{label} is not a bounded single-link publication file"
            )
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise ForagerMatchedV3AdapterRewardPublicationError(
                    f"{label} ended while being read"
                )
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ForagerMatchedV3AdapterRewardPublicationError(
                f"{label} grew while being read"
            )
        after = os.fstat(descriptor)
        current = os.stat(name, dir_fd=root.descriptor, follow_symlinks=False)
        if (
            _stat_identity(before) != _stat_identity(after)
            or _stat_identity(after) != _stat_identity(current)
        ):
            raise ForagerMatchedV3AdapterRewardPublicationError(
                f"{label} changed while being read"
            )
        return b"".join(chunks)
    finally:
        _close_no_raise(descriptor)


def _load_from_open_root(
    root: _OpenDirectory,
    *,
    expected_publication_file_sha256: str,
) -> ContentVerifiedAdapterRewardPublication:
    initial_inventory = _root_inventory(root)
    publication_bytes = _read_stable_regular_at(
        root,
        PUBLICATION_MANIFEST_FILENAME,
        "publication manifest",
        maximum=_MAX_PUBLICATION_BYTES,
    )
    manifest = parse_adapter_reward_publication_manifest(
        publication_bytes,
        expected_publication_file_sha256=expected_publication_file_sha256,
    )
    files = cast(dict[str, Any], manifest["files"])
    loaded: dict[str, bytes] = {}
    limits = {
        "adapter_bundle_manifest": (_MAX_ADAPTER_MANIFEST_BYTES, None),
        "runner_result_receipt": (_MAX_RUNNER_RECEIPT_BYTES, None),
        "reward_trace": (
            scorer.CANONICAL_NPZ_SIZE_BYTES,
            scorer.CANONICAL_NPZ_SIZE_BYTES,
        ),
        "score_receipt": (_MAX_SCORE_RECEIPT_BYTES, None),
    }
    for role, path in _FILE_PATHS.items():
        maximum, exact_size = limits[role]
        raw = _read_stable_regular_at(
            root,
            path,
            f"publication file {role}",
            maximum=maximum,
            exact_size=exact_size,
        )
        record = cast(dict[str, Any], files[role])
        if (
            len(raw) != record["size_bytes"]
            or not hmac.compare_digest(
                hashlib.sha256(raw).hexdigest(), cast(str, record["sha256"])
            )
        ):
            raise ForagerMatchedV3AdapterRewardPublicationError(
                f"publication file {role} differs from its outer binding"
            )
        loaded[role] = raw
    dependency = cast(dict[str, Any], manifest["adapter_reward_bundle"])
    candidate_id = cast(str, manifest["candidate_id"])
    reconstructed = reward_bundle.MatchedV3AdapterRewardBundle(
        candidate_id=candidate_id,
        runner_receipt_bytes=loaded["runner_result_receipt"],
        reward_artifact_bytes=loaded["reward_trace"],
        score_receipt_bytes=loaded["score_receipt"],
        manifest_bytes=loaded["adapter_bundle_manifest"],
        manifest_sha256=cast(str, dependency["manifest_body_sha256"]),
    )
    try:
        validated = reward_bundle.validate_adapter_reward_bundle(reconstructed)
    except reward_bundle.ForagerMatchedV3AdapterRewardBundleError as exc:
        raise ForagerMatchedV3AdapterRewardPublicationError(
            "persisted adapter reward bundle failed structural replay"
        ) from exc
    replayed_bytes, replayed_body_sha256, replayed_file_sha256 = (
        _publication_manifest_bytes(validated)
    )
    if (
        replayed_bytes != publication_bytes
        or replayed_file_sha256 != expected_publication_file_sha256
        or replayed_body_sha256 != manifest["publication_body_sha256"]
    ):
        raise ForagerMatchedV3AdapterRewardPublicationError(
            "publication manifest does not replay from its exact payload files"
        )
    if _root_inventory(root) != initial_inventory:
        raise ForagerMatchedV3AdapterRewardPublicationError(
            "adapter publication inventory changed during replay"
        )
    return ContentVerifiedAdapterRewardPublication(
        output_root=root.path,
        candidate_id=candidate_id,
        publication_file_sha256=replayed_file_sha256,
        publication_body_sha256=replayed_body_sha256,
        manifest=cast(Mapping[str, Any], _freeze_json(manifest)),
        bundle=validated,
    )


def load_adapter_reward_bundle_publication(
    root: Path,
    *,
    expected_publication_file_sha256: str,
) -> ContentVerifiedAdapterRewardPublication:
    """Load one exact publication under a required externally carried file digest."""

    _require_sha256(
        expected_publication_file_sha256,
        "expected publication manifest file digest",
    )
    if not isinstance(root, Path):
        raise ForagerMatchedV3AdapterRewardPublicationError(
            "adapter publication root must be a pathlib Path"
        )
    opened = _open_stable_directory(
        root,
        "adapter publication root",
        required_mode=_DIRECTORY_MODE,
    )
    try:
        result = _load_from_open_root(
            opened,
            expected_publication_file_sha256=expected_publication_file_sha256,
        )
        _assert_open_directory_path(opened, "adapter publication root")
        return result
    except ForagerMatchedV3AdapterRewardPublicationError:
        raise
    except OSError as exc:
        raise ForagerMatchedV3AdapterRewardPublicationError(
            "adapter publication filesystem replay failed"
        ) from exc
    finally:
        _close_no_raise(opened.descriptor)


def _write_exclusive_at(root: _OpenDirectory, name: str, raw: bytes) -> None:
    if type(raw) is not bytes or not raw:
        raise ForagerMatchedV3AdapterRewardPublicationError(
            "empty or non-byte publication files are forbidden"
        )
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(name, flags, _FILE_MODE, dir_fd=root.descriptor)
    except OSError as exc:
        raise ForagerMatchedV3AdapterRewardPublicationError(
            f"cannot stage publication file {name!r}"
        ) from exc
    try:
        os.fchmod(descriptor, _FILE_MODE)
        remaining = memoryview(raw)
        while remaining:
            written = os.write(descriptor, remaining)
            if written < 1:
                raise ForagerMatchedV3AdapterRewardPublicationError(
                    f"cannot completely stage publication file {name!r}"
                )
            remaining = remaining[written:]
        os.fsync(descriptor)
        opened = os.fstat(descriptor)
        current = os.stat(name, dir_fd=root.descriptor, follow_symlinks=False)
        root_metadata = os.fstat(root.descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or opened.st_size != len(raw)
            or stat.S_IMODE(opened.st_mode) != _FILE_MODE
            or (opened.st_uid, opened.st_gid)
            != (root_metadata.st_uid, root_metadata.st_gid)
            or _stat_identity(opened) != _stat_identity(current)
        ):
            raise ForagerMatchedV3AdapterRewardPublicationError(
                f"staged publication file {name!r} changed while writing"
            )
    except OSError as exc:
        raise ForagerMatchedV3AdapterRewardPublicationError(
            f"cannot stage publication file {name!r}"
        ) from exc
    finally:
        _close_no_raise(descriptor)


def _durably_sync_open_tree(root: _OpenDirectory) -> None:
    _assert_open_directory_path(root, "adapter publication staging directory")
    initial_inventory = _root_inventory(root)
    try:
        for name in sorted(initial_inventory, key=os.fsencode):
            expected = os.stat(name, dir_fd=root.descriptor, follow_symlinks=False)
            descriptor = os.open(name, _file_read_flags(), dir_fd=root.descriptor)
            try:
                opened = os.fstat(descriptor)
                if _stat_identity(expected) != _stat_identity(opened):
                    raise ForagerMatchedV3AdapterRewardPublicationError(
                        "staged publication file changed before fsync"
                    )
                os.fsync(descriptor)
                current = os.stat(name, dir_fd=root.descriptor, follow_symlinks=False)
                if (
                    _stat_identity(opened) != _stat_identity(os.fstat(descriptor))
                    or _stat_identity(opened) != _stat_identity(current)
                ):
                    raise ForagerMatchedV3AdapterRewardPublicationError(
                        "staged publication file changed during fsync"
                    )
            finally:
                _close_no_raise(descriptor)
        os.fsync(root.descriptor)
    except OSError as exc:
        raise ForagerMatchedV3AdapterRewardPublicationError(
            "cannot durably sync adapter publication staging tree"
        ) from exc
    if _root_inventory(root) != initial_inventory:
        raise ForagerMatchedV3AdapterRewardPublicationError(
            "staged publication inventory changed during fsync"
        )
    _assert_open_directory_path(root, "adapter publication staging directory")


def _parent_entry_matches_open_directory(
    parent: _OpenDirectory,
    name: str,
    opened: _OpenDirectory,
) -> bool:
    try:
        entry = os.stat(name, dir_fd=parent.descriptor, follow_symlinks=False)
        descriptor = os.fstat(opened.descriptor)
    except OSError:
        return False
    return (
        stat.S_ISDIR(entry.st_mode)
        and _inode_identity(entry) == opened.inode_identity
        and _inode_identity(descriptor) == opened.inode_identity
    )


def _sync_publication_parent(parent: _OpenDirectory) -> None:
    os.fsync(parent.descriptor)


def _rename_no_replace(
    parent: _OpenDirectory,
    source_name: str,
    destination_name: str,
) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise ForagerMatchedV3AdapterRewardPublicationError(
            "renameat2 is required for exclusive adapter publication"
        )
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        parent.descriptor,
        os.fsencode(source_name),
        parent.descriptor,
        os.fsencode(destination_name),
        1,
    )
    if result != 0:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise ForagerMatchedV3AdapterRewardPublicationError(
                "adapter publication destination was created concurrently"
            )
        raise ForagerMatchedV3AdapterRewardPublicationError(
            f"exclusive adapter publication failed with errno {error}"
        )


def _entry_exists(parent: _OpenDirectory, name: str) -> bool:
    try:
        os.stat(name, dir_fd=parent.descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return False
    except OSError:
        return True
    return True


def _publish_verified_no_replace(
    parent: _OpenDirectory,
    staging: _OpenDirectory,
    source_name: str,
    destination_name: str,
    destination: Path,
    *,
    publication_file_sha256: str,
    publication_body_sha256: str,
) -> None:
    _assert_open_directory_path(parent, "adapter publication parent")
    if not _parent_entry_matches_open_directory(parent, source_name, staging):
        raise ForagerMatchedV3AdapterRewardPublicationError(
            "staging name no longer refers to the verified publication inode"
        )
    try:
        _rename_no_replace(parent, source_name, destination_name)
    except BaseException as exc:
        destination_matches = _parent_entry_matches_open_directory(
            parent, destination_name, staging
        )
        source_matches = _parent_entry_matches_open_directory(parent, source_name, staging)
        if destination_matches or not source_matches:
            raise PublishedAdapterRewardPublicationUncertainError(
                destination,
                "exclusive move outcome is uncertain",
                publication_file_sha256=publication_file_sha256,
                publication_body_sha256=publication_body_sha256,
            ) from exc
        raise
    try:
        if not _parent_entry_matches_open_directory(parent, destination_name, staging):
            raise ForagerMatchedV3AdapterRewardPublicationError(
                "published destination differs from the verified staging inode"
            )
        if _entry_exists(parent, source_name):
            raise ForagerMatchedV3AdapterRewardPublicationError(
                "staging name survived exclusive publication"
            )
        _sync_publication_parent(parent)
        if not _parent_entry_matches_open_directory(parent, destination_name, staging):
            raise ForagerMatchedV3AdapterRewardPublicationError(
                "published destination changed during parent fsync"
            )
        _assert_open_directory_path(parent, "adapter publication parent")
    except BaseException as exc:
        raise PublishedAdapterRewardPublicationUncertainError(
            destination,
            "publication durability or inode verification is uncertain",
            publication_file_sha256=publication_file_sha256,
            publication_body_sha256=publication_body_sha256,
        ) from exc


def _cleanup_owned_staging(
    parent: _OpenDirectory,
    name: str,
    staging: _OpenDirectory,
) -> None:
    if not _parent_entry_matches_open_directory(parent, name, staging):
        return
    entries: list[tuple[str, tuple[int, ...]]] = []
    try:
        with os.scandir(staging.descriptor) as iterator:
            for entry in iterator:
                if len(entries) >= _MAX_ROOT_ENTRIES:
                    return
                metadata = entry.stat(follow_symlinks=False)
                if (
                    not stat.S_ISREG(metadata.st_mode)
                    or metadata.st_nlink != 1
                    or stat.S_IMODE(metadata.st_mode) != _FILE_MODE
                ):
                    return
                entries.append((entry.name, _stat_identity(metadata)))
        for entry_name, expected in entries:
            current = os.stat(
                entry_name,
                dir_fd=staging.descriptor,
                follow_symlinks=False,
            )
            if _stat_identity(current) != expected:
                return
        for entry_name, _ in entries:
            os.unlink(entry_name, dir_fd=staging.descriptor)
        if _parent_entry_matches_open_directory(parent, name, staging):
            os.rmdir(name, dir_fd=parent.descriptor)
    except OSError:
        return


def _open_destination_parent(
    output_root: Path,
) -> tuple[_OpenDirectory, Path]:
    if not isinstance(output_root, Path):
        raise ForagerMatchedV3AdapterRewardPublicationError(
            "adapter publication output root must be a pathlib Path"
        )
    if _PORTABLE_NAME_RE.fullmatch(output_root.name) is None:
        raise ForagerMatchedV3AdapterRewardPublicationError(
            "adapter publication output name is not portable"
        )
    if output_root.exists() or output_root.is_symlink():
        raise ForagerMatchedV3AdapterRewardPublicationError(
            "adapter publication output root already exists"
        )
    try:
        prospective = output_root.resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise ForagerMatchedV3AdapterRewardPublicationError(
            "adapter publication output path cannot be resolved"
        ) from exc
    parent = _open_stable_directory(output_root.parent, "adapter publication parent")
    try:
        parent_metadata = os.fstat(parent.descriptor)
        if (
            parent_metadata.st_uid != os.geteuid()
            or stat.S_IMODE(parent_metadata.st_mode) & 0o022
        ):
            raise ForagerMatchedV3AdapterRewardPublicationError(
                "adapter publication parent must be effective-UID-owned and not "
                "group/world writable"
            )
        destination = parent.path / output_root.name
        if destination != prospective:
            raise ForagerMatchedV3AdapterRewardPublicationError(
                "adapter publication parent was redirected before publication"
            )
        if _entry_exists(parent, output_root.name):
            raise ForagerMatchedV3AdapterRewardPublicationError(
                "adapter publication output root already exists"
            )
        return parent, destination
    except ForagerMatchedV3AdapterRewardPublicationError:
        _close_no_raise(parent.descriptor)
        raise
    except OSError as exc:
        _close_no_raise(parent.descriptor)
        raise ForagerMatchedV3AdapterRewardPublicationError(
            "cannot verify adapter publication parent namespace"
        ) from exc
    except BaseException:
        _close_no_raise(parent.descriptor)
        raise


def _create_owned_staging(
    parent: _OpenDirectory,
    destination: Path,
) -> tuple[str, _OpenDirectory]:
    for _ in range(_STAGING_ATTEMPTS):
        name = f".forager-v3-adapter-partial-{secrets.token_hex(16)}"
        try:
            os.mkdir(name, _DIRECTORY_MODE, dir_fd=parent.descriptor)
        except FileExistsError:
            continue
        except OSError as exc:
            raise ForagerMatchedV3AdapterRewardPublicationError(
                "cannot create adapter publication staging directory"
            ) from exc
        path = parent.path / name
        staging: _OpenDirectory | None = None
        created_identity: tuple[int, int, int] | None = None
        try:
            created = os.stat(name, dir_fd=parent.descriptor, follow_symlinks=False)
            created_identity = _inode_identity(created)
            if not stat.S_ISDIR(created.st_mode) or created.st_uid != os.geteuid():
                raise ForagerMatchedV3AdapterRewardPublicationError(
                    "new adapter publication staging inode is not owned"
                )
            os.chmod(
                name,
                _DIRECTORY_MODE,
                dir_fd=parent.descriptor,
                follow_symlinks=False,
            )
            normalized = os.stat(
                name,
                dir_fd=parent.descriptor,
                follow_symlinks=False,
            )
            if (
                _inode_identity(normalized) != created_identity
                or stat.S_IMODE(normalized.st_mode) != _DIRECTORY_MODE
            ):
                raise ForagerMatchedV3AdapterRewardPublicationError(
                    "adapter publication staging inode changed during mode normalization"
                )
            staging = _open_stable_directory_at(
                parent,
                name,
                path,
                "adapter publication staging directory",
                required_mode=_DIRECTORY_MODE,
            )
            os.fchmod(staging.descriptor, _DIRECTORY_MODE)
            metadata = os.fstat(staging.descriptor)
            if (
                _inode_identity(metadata) != created_identity
                or stat.S_IMODE(metadata.st_mode) != _DIRECTORY_MODE
                or metadata.st_uid != os.geteuid()
            ):
                raise ForagerMatchedV3AdapterRewardPublicationError(
                    "adapter publication staging ownership or mode is unsafe"
                )
            if destination.parent != parent.path:
                raise ForagerMatchedV3AdapterRewardPublicationError(
                    "adapter publication destination escaped its opened parent"
                )
            return name, staging
        except BaseException as exc:
            if staging is not None:
                _cleanup_owned_staging(parent, name, staging)
                _close_no_raise(staging.descriptor)
            elif created_identity is not None:
                cleanup: _OpenDirectory | None = None
                try:
                    cleanup = _open_stable_directory_at(
                        parent,
                        name,
                        path,
                        "adapter publication failed staging directory",
                        required_mode=None,
                    )
                    if cleanup.inode_identity == created_identity:
                        _cleanup_owned_staging(parent, name, cleanup)
                except (ForagerMatchedV3AdapterRewardPublicationError, OSError):
                    pass
                finally:
                    if cleanup is not None:
                        _close_no_raise(cleanup.descriptor)
            else:
                # The parent namespace is effective-UID-owned and non-writable by
                # other users.  If the first metadata read itself failed, removing
                # only the empty name just created is the sole safe bounded cleanup.
                try:
                    os.rmdir(name, dir_fd=parent.descriptor)
                except OSError:
                    pass
            if isinstance(exc, OSError):
                raise ForagerMatchedV3AdapterRewardPublicationError(
                    "cannot initialize adapter publication staging directory"
                ) from exc
            raise
    raise ForagerMatchedV3AdapterRewardPublicationError(
        "cannot allocate a unique adapter publication staging directory"
    )


def publish_adapter_reward_bundle(
    bundle: reward_bundle.MatchedV3AdapterRewardBundle,
    output_root: Path,
) -> ContentVerifiedAdapterRewardPublication:
    """Atomically publish one validated structural bundle without granting authority."""

    try:
        validated = reward_bundle.validate_adapter_reward_bundle(bundle)
    except reward_bundle.ForagerMatchedV3AdapterRewardBundleError as exc:
        raise ForagerMatchedV3AdapterRewardPublicationError(
            "adapter reward bundle failed structural validation before publication"
        ) from exc
    publication_bytes, publication_body_sha256, publication_file_sha256 = (
        _publication_manifest_bytes(validated)
    )
    payloads = _payload_bytes(validated)
    parent, destination = _open_destination_parent(output_root)
    staging: _OpenDirectory | None = None
    staging_name = ""
    published = False
    try:
        staging_name, staging = _create_owned_staging(parent, destination)
        for role, path in _FILE_PATHS.items():
            _write_exclusive_at(staging, path, payloads[role])
        _write_exclusive_at(
            staging,
            PUBLICATION_MANIFEST_FILENAME,
            publication_bytes,
        )
        _load_from_open_root(
            staging,
            expected_publication_file_sha256=publication_file_sha256,
        )
        _durably_sync_open_tree(staging)
        _publish_verified_no_replace(
            parent,
            staging,
            staging_name,
            destination.name,
            destination,
            publication_file_sha256=publication_file_sha256,
            publication_body_sha256=publication_body_sha256,
        )
        published = True
        final_root = _open_stable_directory_at(
            parent,
            destination.name,
            destination,
            "published adapter reward directory",
            required_mode=_DIRECTORY_MODE,
        )
        try:
            if not _parent_entry_matches_open_directory(
                parent, destination.name, staging
            ) or not _parent_entry_matches_open_directory(
                parent, destination.name, final_root
            ):
                raise ForagerMatchedV3AdapterRewardPublicationError(
                    "published adapter reward inode changed before final replay"
                )
            result = _load_from_open_root(
                final_root,
                expected_publication_file_sha256=publication_file_sha256,
            )
            _assert_open_directory_path(final_root, "published adapter reward directory")
            _assert_open_directory_path(parent, "adapter publication parent")
            return result
        except BaseException as exc:
            raise PublishedAdapterRewardPublicationUncertainError(
                destination,
                "final content replay is uncertain",
                publication_file_sha256=publication_file_sha256,
                publication_body_sha256=publication_body_sha256,
            ) from exc
        finally:
            _close_no_raise(final_root.descriptor)
    except PublishedAdapterRewardPublicationUncertainError:
        raise
    except BaseException as exc:
        destination_matches = (
            staging is not None
            and _parent_entry_matches_open_directory(
                parent, destination.name, staging
            )
        )
        source_matches = (
            staging is not None
            and bool(staging_name)
            and _parent_entry_matches_open_directory(parent, staging_name, staging)
        )
        if published or destination_matches or (staging is not None and not source_matches):
            raise PublishedAdapterRewardPublicationUncertainError(
                destination,
                "final publication state is uncertain",
                publication_file_sha256=publication_file_sha256,
                publication_body_sha256=publication_body_sha256,
            ) from exc
        if staging is not None:
            _cleanup_owned_staging(parent, staging_name, staging)
        if isinstance(exc, OSError):
            raise ForagerMatchedV3AdapterRewardPublicationError(
                "adapter publication filesystem operation failed before publication"
            ) from exc
        raise
    finally:
        if staging is not None:
            _close_no_raise(staging.descriptor)
        _close_no_raise(parent.descriptor)


__all__ = [
    "ADAPTER_BUNDLE_MANIFEST_FILENAME",
    "ADAPTER_REWARD_PUBLICATION_DESCRIPTOR_SCHEMA_VERSION",
    "ADAPTER_REWARD_PUBLICATION_DESCRIPTOR_SHA256",
    "ADAPTER_REWARD_PUBLICATION_SCHEMA_VERSION",
    "ADAPTER_REWARD_PUBLICATION_STATUS",
    "ContentVerifiedAdapterRewardPublication",
    "ForagerMatchedV3AdapterRewardPublicationError",
    "PUBLICATION_FILENAME",
    "PUBLICATION_MANIFEST_FILENAME",
    "PublishedAdapterRewardPublicationUncertainError",
    "REWARD_TRACE_FILENAME",
    "RUNNER_RESULT_RECEIPT_FILENAME",
    "SCORE_RECEIPT_FILENAME",
    "adapter_reward_publication_descriptor",
    "canonical_adapter_reward_publication_descriptor_bytes",
    "load_adapter_reward_bundle_publication",
    "parse_adapter_reward_publication_descriptor",
    "parse_adapter_reward_publication_manifest",
    "publish_adapter_reward_bundle",
]
