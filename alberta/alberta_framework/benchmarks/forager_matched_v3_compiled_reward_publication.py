"""Atomic, content-only publication of a compiled PPO-GRU reward bundle.

The compiled runner's process-local outcome capability is deliberately absent from the
bundle.  This module publishes and reloads only the bundle's structurally replayable bytes;
it never reconstructs that capability and grants no execution, runtime, ingestion,
qualification, evidence, promotion, publication-policy, or performance authority.

Publication is Linux-local and fail-closed.  Six exact files are written into an exclusive
owned sibling directory, replayed, individually fsynced, directory-fsynced, moved through a
held safe parent descriptor with ``renameat2(RENAME_NOREPLACE)``, parent-fsynced, and replayed
again.  Loading requires a caller-carried full-file SHA-256 for ``publication.json``.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import hmac
import json
import math
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
    forager_matched_v3_compiled_reward_bundle as compiled_bundle,
)

COMPILED_REWARD_PUBLICATION_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.compiled_reward_publication_descriptor.v1"
)
COMPILED_REWARD_PUBLICATION_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.compiled_reward_publication.v1"
)
COMPILED_REWARD_PUBLICATION_STATUS: Final = "implemented_unexecuted"

PUBLICATION_MANIFEST_FILENAME: Final = "publication.json"
PUBLICATION_FILENAME: Final = PUBLICATION_MANIFEST_FILENAME
COMPILED_BUNDLE_MANIFEST_FILENAME: Final = "compiled-bundle-manifest.json"
RUNNER_RESULT_RECEIPT_FILENAME: Final = "runner-result-receipt.json"
RUNTIME_IDENTITY_FILENAME: Final = "runtime-identity.json"
REWARD_TRACE_FILENAME: Final = "reward-trace.npz"
SCORE_RECEIPT_FILENAME: Final = "score-receipt.json"

_COMPILED_BUNDLE_SOURCE_PATH: Final = (
    "alberta_framework/benchmarks/forager_matched_v3_compiled_reward_bundle.py"
)
_COMPILED_BUNDLE_SOURCE_SHA256: Final = (
    "e50466c185d66334f629915944407d72cb4aff4aa611dffbbe20de8aa8146f6e"
)
_COMPILED_BUNDLE_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.compiled_reward_bundle_descriptor.v1"
)
_COMPILED_BUNDLE_DESCRIPTOR_SHA256: Final = (
    "cc9e2ad605496682ff2870bb6db312f56ad4926f4805a4a90fbacac4f648cf08"
)
_COMPILED_BUNDLE_MANIFEST_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.compiled_reward_bundle_manifest.v1"
)
_SCORE_RECEIPT_SCHEMA_VERSION: Final = "alberta.forager_matched_v3_score_receipt.v1"
_NPZ_CONTAINER_SCHEMA_VERSION: Final = "alberta.forager_matched_v3_reward_npz.v1"
_CANONICAL_NPZ_SIZE_BYTES: Final = 499_980
_WRITER_CONTRACT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.compiled_reward_publication_writer.linux.v1"
)

_DIRECTORY_MODE: Final = 0o700
_FILE_MODE: Final = 0o600
_MAX_SOURCE_BYTES: Final = 4 * 1024 * 1024
_MAX_PUBLICATION_BYTES: Final = 64 * 1024
_MAX_BUNDLE_MANIFEST_BYTES: Final = 256 * 1024
_MAX_RUNNER_RECEIPT_BYTES: Final = 1024 * 1024
_MAX_RUNTIME_IDENTITY_BYTES: Final = 256 * 1024
_MAX_SCORE_RECEIPT_BYTES: Final = 64 * 1024
_MAX_ROOT_ENTRIES: Final = 6
_MAX_JSON_NODES: Final = 8_192
_MAX_JSON_DEPTH: Final = 32
_MAX_JSON_STRING_CHARACTERS: Final = 16 * 1024
_MAX_JSON_INTEGER_DIGITS: Final = 19
_STAGING_ATTEMPTS: Final = 64
_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_PORTABLE_NAME_RE: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_CONCRETE_PATH_TYPE: Final = type(Path())

_FILE_PATHS: Final = MappingProxyType(
    {
        "compiled_bundle_manifest": COMPILED_BUNDLE_MANIFEST_FILENAME,
        "runner_result_receipt": RUNNER_RESULT_RECEIPT_FILENAME,
        "runtime_identity": RUNTIME_IDENTITY_FILENAME,
        "reward_trace": REWARD_TRACE_FILENAME,
        "score_receipt": SCORE_RECEIPT_FILENAME,
    }
)


class ForagerMatchedV3CompiledRewardPublicationError(ValueError):
    """Publication content, filesystem structure, or durability failed closed."""


class PublishedCompiledRewardPublicationUncertainError(
    ForagerMatchedV3CompiledRewardPublicationError
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
        super().__init__(f"compiled reward publication at {destination}, but {detail}")


@dataclass(frozen=True, slots=True)
class ContentVerifiedCompiledRewardPublication:
    """Structurally replayed persisted bytes with no execution or ingestion authority."""

    output_root: Path
    candidate_id: str
    publication_file_sha256: str
    publication_body_sha256: str
    manifest: Mapping[str, Any]
    bundle: compiled_bundle.MatchedV3CompiledRewardBundle


@dataclass(frozen=True, slots=True)
class _OpenDirectory:
    path: Path
    descriptor: int
    inode_identity: tuple[int, int, int]


def _source_stat_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _source_sha256(module_file: object, expected_suffix: str) -> str:
    if type(module_file) is not str or not module_file.endswith(expected_suffix):
        raise RuntimeError(f"cannot resolve exact source path for {expected_suffix}")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(module_file, flags)
    except OSError as exc:
        raise RuntimeError(f"cannot open exact source bytes for {expected_suffix}") from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or not 0 < before.st_size <= _MAX_SOURCE_BYTES
        ):
            raise RuntimeError(
                f"source is not one bounded single-link regular file: {expected_suffix}"
            )
        remaining = before.st_size
        digest = hashlib.sha256()
        while remaining:
            chunk = os.read(descriptor, min(remaining, 64 * 1024))
            if not chunk:
                raise RuntimeError(f"source ended while reading {expected_suffix}")
            digest.update(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise RuntimeError(f"source grew while reading {expected_suffix}")
        after = os.fstat(descriptor)
        if _source_stat_identity(before) != _source_stat_identity(after):
            raise RuntimeError(f"source changed while reading {expected_suffix}")
        return digest.hexdigest()
    except OSError as exc:
        raise RuntimeError(f"cannot read exact source bytes for {expected_suffix}") from exc
    finally:
        os.close(descriptor)


def _check_frozen_bindings() -> None:
    if not hmac.compare_digest(
        _source_sha256(compiled_bundle.__file__, _COMPILED_BUNDLE_SOURCE_PATH),
        _COMPILED_BUNDLE_SOURCE_SHA256,
    ):
        raise RuntimeError("compiled reward publication dependency source binding drifted")
    if (
        compiled_bundle.COMPILED_REWARD_BUNDLE_DESCRIPTOR_SCHEMA_VERSION
        != _COMPILED_BUNDLE_DESCRIPTOR_SCHEMA_VERSION
        or compiled_bundle.COMPILED_REWARD_BUNDLE_DESCRIPTOR_SHA256
        != _COMPILED_BUNDLE_DESCRIPTOR_SHA256
        or compiled_bundle.COMPILED_REWARD_BUNDLE_MANIFEST_SCHEMA_VERSION
        != _COMPILED_BUNDLE_MANIFEST_SCHEMA_VERSION
        or scorer.SCORE_RECEIPT_SCHEMA_VERSION != _SCORE_RECEIPT_SCHEMA_VERSION
        or scorer.NPZ_CONTAINER_SCHEMA_VERSION != _NPZ_CONTAINER_SCHEMA_VERSION
        or scorer.CANONICAL_NPZ_SIZE_BYTES != _CANONICAL_NPZ_SIZE_BYTES
    ):
        raise RuntimeError("compiled reward publication descriptor or scorer binding drifted")


def _claims() -> dict[str, bool]:
    return {
        "authority_granted": False,
        "campaign_ingestion_authorized": False,
        "evidence_authority": False,
        "execution_authorized": False,
        "execution_ready": False,
        "live_outcome_capability_reconstructed": False,
        "performance_claim_allowed": False,
        "production_result_accepted": False,
        "qualification_claim_allowed": False,
        "runtime_qualified": False,
        "scientific_promotion_allowed": False,
        "universal_sota_claim_allowed": False,
    }


def _limitations() -> list[str]:
    return [
        "The live compiled-outcome capability is not serialized or reconstructed.",
        "Content hashes and structural replay do not independently prove agent execution.",
        "Atomic publication is a writer contract, not execution or scientific attestation.",
        "Seed provenance and source/runtime qualification remain unverified.",
        "Loading grants no campaign ingestion, evidence, promotion, or performance authority.",
    ]


def _writer_contract() -> dict[str, Any]:
    return {
        "schema_version": _WRITER_CONTRACT_SCHEMA_VERSION,
        "layout": "one_flat_directory_with_six_exact_files",
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
        "schema_version": COMPILED_REWARD_PUBLICATION_DESCRIPTOR_SCHEMA_VERSION,
        "status": COMPILED_REWARD_PUBLICATION_STATUS,
        "classification": "durable_compiled_content_publication_non_authorizing",
        "dependency": {
            "source_path": _COMPILED_BUNDLE_SOURCE_PATH,
            "source_sha256": _COMPILED_BUNDLE_SOURCE_SHA256,
            "descriptor_schema_version": _COMPILED_BUNDLE_DESCRIPTOR_SCHEMA_VERSION,
            "descriptor_sha256": _COMPILED_BUNDLE_DESCRIPTOR_SHA256,
            "manifest_schema_version": _COMPILED_BUNDLE_MANIFEST_SCHEMA_VERSION,
        },
        "candidate_consumer": "adapted_ppo_gru",
        "strict_scorer": {
            "score_receipt_schema_version": _SCORE_RECEIPT_SCHEMA_VERSION,
            "npz_container_schema_version": _NPZ_CONTAINER_SCHEMA_VERSION,
            "canonical_npz_size_bytes": _CANONICAL_NPZ_SIZE_BYTES,
            "score_receipt_maximum_bytes": _MAX_SCORE_RECEIPT_BYTES,
        },
        "exact_files": {
            "publication_manifest": PUBLICATION_MANIFEST_FILENAME,
            **dict(_FILE_PATHS),
        },
        "bounds": {
            "root_entries": _MAX_ROOT_ENTRIES,
            "publication_manifest_bytes": _MAX_PUBLICATION_BYTES,
            "compiled_bundle_manifest_bytes": _MAX_BUNDLE_MANIFEST_BYTES,
            "runner_result_receipt_bytes": _MAX_RUNNER_RECEIPT_BYTES,
            "runtime_identity_bytes": _MAX_RUNTIME_IDENTITY_BYTES,
            "reward_trace_bytes": _CANONICAL_NPZ_SIZE_BYTES,
            "score_receipt_bytes": _MAX_SCORE_RECEIPT_BYTES,
        },
        "filesystem_contract": _writer_contract(),
        "loading": {
            "caller_supplied_publication_file_sha256_required": True,
            "exact_inventory_required": True,
            "regular_single_link_files_required": True,
            "stable_descriptor_relative_reads_required": True,
            "compiled_bundle_structural_replay_required": True,
            "live_capability_reconstruction": False,
        },
        "claims": _claims(),
        "limitations": _limitations(),
    }


def _assert_plain_unaliased_json(value: object, *, label: str) -> None:
    pending = [(value, 0)]
    seen: set[int] = set()
    nodes = 0
    while pending:
        item, depth = pending.pop()
        nodes += 1
        if nodes > _MAX_JSON_NODES:
            raise ForagerMatchedV3CompiledRewardPublicationError(
                f"{label} exceeds its JSON node bound"
            )
        if depth > _MAX_JSON_DEPTH:
            raise ForagerMatchedV3CompiledRewardPublicationError(
                f"{label} exceeds its JSON depth bound"
            )
        if type(item) is dict:
            identity = id(item)
            if identity in seen:
                raise ForagerMatchedV3CompiledRewardPublicationError(
                    f"{label} contains aliased or cyclic containers"
                )
            seen.add(identity)
            mapping = cast(dict[object, object], item)
            if any(type(key) is not str for key in mapping):
                raise ForagerMatchedV3CompiledRewardPublicationError(
                    f"{label} contains a non-string object key"
                )
            pending.extend((child, depth + 1) for child in mapping.values())
        elif type(item) is list:
            identity = id(item)
            if identity in seen:
                raise ForagerMatchedV3CompiledRewardPublicationError(
                    f"{label} contains aliased or cyclic containers"
                )
            seen.add(identity)
            pending.extend((child, depth + 1) for child in cast(list[object], item))
        elif type(item) is str:
            if len(item) > _MAX_JSON_STRING_CHARACTERS:
                raise ForagerMatchedV3CompiledRewardPublicationError(
                    f"{label} contains an oversized string"
                )
        elif type(item) is float:
            if not math.isfinite(item):
                raise ForagerMatchedV3CompiledRewardPublicationError(
                    f"{label} contains a non-finite float"
                )
            raise ForagerMatchedV3CompiledRewardPublicationError(
                f"{label} contains a forbidden float"
            )
        elif item is not None and type(item) not in {bool, int}:
            raise ForagerMatchedV3CompiledRewardPublicationError(
                f"{label} contains non-plain type {type(item).__name__}"
            )


def _canonical_json(value: object, *, label: str, maximum: int) -> bytes:
    if type(value) is not dict:
        raise ForagerMatchedV3CompiledRewardPublicationError(f"{label} root must be a plain object")
    _assert_plain_unaliased_json(value, label=label)
    try:
        raw = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (OverflowError, RecursionError, TypeError, UnicodeError, ValueError) as exc:
        raise ForagerMatchedV3CompiledRewardPublicationError(
            f"{label} is not finite canonical ASCII JSON"
        ) from exc
    if not 0 < len(raw) <= maximum:
        raise ForagerMatchedV3CompiledRewardPublicationError(
            f"{label} violates its canonical byte bound"
        )
    return raw


_check_frozen_bindings()
_DESCRIPTOR_BYTES: Final = _canonical_json(
    _descriptor(),
    label="compiled reward publication descriptor",
    maximum=_MAX_PUBLICATION_BYTES,
)
COMPILED_REWARD_PUBLICATION_DESCRIPTOR_SHA256: Final = (
    "a7827fd32b526c1ad3f9d22549a66fa054c3785c75891560356db82791a3f500"
)
if not hmac.compare_digest(
    hashlib.sha256(_DESCRIPTOR_BYTES).hexdigest(),
    COMPILED_REWARD_PUBLICATION_DESCRIPTOR_SHA256,
):
    raise RuntimeError("compiled reward publication descriptor identity drifted")


def compiled_reward_publication_descriptor() -> dict[str, Any]:
    """Return a detached copy of the frozen nonauthorizing descriptor."""

    return cast(dict[str, Any], json.loads(_DESCRIPTOR_BYTES.decode("ascii")))


def canonical_compiled_reward_publication_descriptor_bytes() -> bytes:
    """Return the exact canonical publication descriptor bytes."""

    return bytes(_DESCRIPTOR_BYTES)


def parse_compiled_reward_publication_descriptor(raw: bytes) -> dict[str, Any]:
    """Accept only the exact frozen publication descriptor identity."""

    if type(raw) is not bytes or not hmac.compare_digest(raw, _DESCRIPTOR_BYTES):
        raise ForagerMatchedV3CompiledRewardPublicationError(
            "compiled reward publication descriptor bytes differ"
        )
    if not hmac.compare_digest(
        hashlib.sha256(raw).hexdigest(),
        COMPILED_REWARD_PUBLICATION_DESCRIPTOR_SHA256,
    ):
        raise ForagerMatchedV3CompiledRewardPublicationError(
            "compiled reward publication descriptor digest differs"
        )
    return compiled_reward_publication_descriptor()


def _raise_json_constant(value: str) -> NoReturn:
    raise ForagerMatchedV3CompiledRewardPublicationError(
        f"publication JSON contains non-finite constant {value!r}"
    )


def _raise_json_float(value: str) -> NoReturn:
    raise ForagerMatchedV3CompiledRewardPublicationError(
        f"publication JSON contains forbidden float {value!r}"
    )


def _parse_bounded_int(value: str) -> int:
    if len(value.lstrip("-")) > _MAX_JSON_INTEGER_DIGITS:
        raise ForagerMatchedV3CompiledRewardPublicationError(
            "publication JSON integer exceeds its lexical bound"
        )
    return int(value)


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ForagerMatchedV3CompiledRewardPublicationError(
                f"publication JSON contains duplicate key {key!r}"
            )
        result[key] = value
    return result


def _validate_json_lexical_bounds(text: str, *, label: str) -> None:
    depth = 0
    nodes = 0
    in_string = False
    escaped = False
    in_primitive = False
    for character in text:
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
            in_primitive = False
            nodes += 1
        elif character in "[{":
            depth += 1
            nodes += 1
            in_primitive = False
            if depth > _MAX_JSON_DEPTH:
                raise ForagerMatchedV3CompiledRewardPublicationError(
                    f"{label} exceeds its JSON depth bound"
                )
        elif character in "]}":
            depth -= 1
            in_primitive = False
        elif character in ",:":
            in_primitive = False
        elif character in " \t\r\n":
            in_primitive = False
        elif not in_primitive:
            nodes += 1
            in_primitive = True
        if nodes > _MAX_JSON_NODES:
            raise ForagerMatchedV3CompiledRewardPublicationError(
                f"{label} exceeds its JSON node bound"
            )


def _strict_json_object(raw: bytes, *, label: str, maximum: int) -> dict[str, Any]:
    if type(raw) is not bytes or not 0 < len(raw) <= maximum:
        raise ForagerMatchedV3CompiledRewardPublicationError(
            f"{label} violates its JSON byte bound"
        )
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ForagerMatchedV3CompiledRewardPublicationError(f"{label} must be ASCII") from exc
    _validate_json_lexical_bounds(text, label=label)
    try:
        parsed = json.loads(
            text,
            object_pairs_hook=_object_without_duplicates,
            parse_constant=_raise_json_constant,
            parse_float=_raise_json_float,
            parse_int=_parse_bounded_int,
        )
    except ForagerMatchedV3CompiledRewardPublicationError:
        raise
    except (RecursionError, json.JSONDecodeError, ValueError) as exc:
        raise ForagerMatchedV3CompiledRewardPublicationError(f"{label} is not strict JSON") from exc
    if type(parsed) is not dict:
        raise ForagerMatchedV3CompiledRewardPublicationError(f"{label} root must be a plain object")
    result = cast(dict[str, Any], parsed)
    _assert_plain_unaliased_json(result, label=label)
    if not hmac.compare_digest(_canonical_json(result, label=label, maximum=maximum), raw):
        raise ForagerMatchedV3CompiledRewardPublicationError(f"{label} is not exactly canonical")
    return result


def _exact_json_equal(first: object, second: object) -> bool:
    pending = [(first, second)]
    while pending:
        left, right = pending.pop()
        if type(left) is not type(right):
            return False
        if type(left) is dict:
            left_mapping = cast(dict[str, object], left)
            right_mapping = cast(dict[str, object], right)
            if set(left_mapping) != set(right_mapping):
                return False
            pending.extend((left_mapping[key], right_mapping[key]) for key in left_mapping)
        elif type(left) is list:
            left_list = cast(list[object], left)
            right_list = cast(list[object], right)
            if len(left_list) != len(right_list):
                return False
            pending.extend(zip(left_list, right_list, strict=True))
        elif left != right:
            return False
    return True


def _require_exact_keys(value: object, expected: frozenset[str], label: str) -> dict[str, Any]:
    if type(value) is not dict or frozenset(value) != expected:
        raise ForagerMatchedV3CompiledRewardPublicationError(f"{label} fields are not exact")
    return cast(dict[str, Any], value)


def _require_sha256(value: object, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None or value == "0" * 64:
        raise ForagerMatchedV3CompiledRewardPublicationError(
            f"{label} must be a nonzero lowercase SHA-256 digest"
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
        raise ForagerMatchedV3CompiledRewardPublicationError(
            f"{label} must be an exact integer in [{minimum}, {maximum}]"
        )
    return value


def _payload_bytes(
    bundle: compiled_bundle.MatchedV3CompiledRewardBundle,
) -> dict[str, bytes]:
    return {
        "compiled_bundle_manifest": bundle.manifest_bytes,
        "runner_result_receipt": bundle.runner_receipt_bytes,
        "runtime_identity": bundle.runtime_identity_bytes,
        "reward_trace": bundle.reward_artifact_bytes,
        "score_receipt": bundle.score_receipt_bytes,
    }


def _file_records(
    bundle: compiled_bundle.MatchedV3CompiledRewardBundle,
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
    bundle: compiled_bundle.MatchedV3CompiledRewardBundle,
) -> dict[str, Any]:
    return {
        "schema_version": COMPILED_REWARD_PUBLICATION_SCHEMA_VERSION,
        "classification": "durable_compiled_content_publication_non_authorizing",
        "candidate_id": "adapted_ppo_gru",
        "publication_descriptor": {
            "schema_version": COMPILED_REWARD_PUBLICATION_DESCRIPTOR_SCHEMA_VERSION,
            "sha256": COMPILED_REWARD_PUBLICATION_DESCRIPTOR_SHA256,
        },
        "compiled_reward_bundle": {
            "descriptor_schema_version": _COMPILED_BUNDLE_DESCRIPTOR_SCHEMA_VERSION,
            "descriptor_sha256": _COMPILED_BUNDLE_DESCRIPTOR_SHA256,
            "manifest_schema_version": _COMPILED_BUNDLE_MANIFEST_SCHEMA_VERSION,
            "implementation_path": _COMPILED_BUNDLE_SOURCE_PATH,
            "implementation_source_sha256": _COMPILED_BUNDLE_SOURCE_SHA256,
            "manifest_body_sha256": bundle.manifest_sha256,
            "manifest_file_sha256": hashlib.sha256(bundle.manifest_bytes).hexdigest(),
        },
        "files": _file_records(bundle),
        "writer_contract": _writer_contract(),
        "claims": _claims(),
        "limitations": _limitations(),
    }


def _publication_manifest_bytes(
    bundle: compiled_bundle.MatchedV3CompiledRewardBundle,
) -> tuple[bytes, str, str]:
    body = _publication_body(bundle)
    body_bytes = _canonical_json(
        body,
        label="compiled reward publication body",
        maximum=_MAX_PUBLICATION_BYTES,
    )
    body_sha256 = hashlib.sha256(body_bytes).hexdigest()
    payload = dict(body)
    payload["publication_body_sha256"] = body_sha256
    raw = _canonical_json(
        payload,
        label="compiled reward publication manifest",
        maximum=_MAX_PUBLICATION_BYTES,
    )
    return raw, body_sha256, hashlib.sha256(raw).hexdigest()


def _validate_file_records(value: object) -> None:
    files = _require_exact_keys(value, frozenset(_FILE_PATHS), "publication files")
    bounds = {
        "compiled_bundle_manifest": (1, _MAX_BUNDLE_MANIFEST_BYTES),
        "runner_result_receipt": (1, _MAX_RUNNER_RECEIPT_BYTES),
        "runtime_identity": (1, _MAX_RUNTIME_IDENTITY_BYTES),
        "reward_trace": (_CANONICAL_NPZ_SIZE_BYTES, _CANONICAL_NPZ_SIZE_BYTES),
        "score_receipt": (1, _MAX_SCORE_RECEIPT_BYTES),
    }
    for role, path in _FILE_PATHS.items():
        record = _require_exact_keys(
            files[role],
            frozenset({"path", "role", "sha256", "size_bytes"}),
            f"publication file {role}",
        )
        if type(record["path"]) is not str or type(record["role"]) is not str:
            raise ForagerMatchedV3CompiledRewardPublicationError(
                f"publication file {role} path and role must be exact strings"
            )
        if record["path"] != path or record["role"] != role:
            raise ForagerMatchedV3CompiledRewardPublicationError(
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
                "compiled_reward_bundle",
                "files",
                "writer_contract",
                "claims",
                "limitations",
            }
        ),
        "compiled reward publication body",
    )
    if (
        type(body["schema_version"]) is not str
        or body["schema_version"] != COMPILED_REWARD_PUBLICATION_SCHEMA_VERSION
        or type(body["classification"]) is not str
        or body["classification"] != "durable_compiled_content_publication_non_authorizing"
        or type(body["candidate_id"]) is not str
        or body["candidate_id"] != "adapted_ppo_gru"
    ):
        raise ForagerMatchedV3CompiledRewardPublicationError(
            "compiled reward publication fixed identity drifted"
        )
    publication_descriptor = _require_exact_keys(
        body["publication_descriptor"],
        frozenset({"schema_version", "sha256"}),
        "publication descriptor binding",
    )
    if not _exact_json_equal(
        publication_descriptor,
        {
            "schema_version": COMPILED_REWARD_PUBLICATION_DESCRIPTOR_SCHEMA_VERSION,
            "sha256": COMPILED_REWARD_PUBLICATION_DESCRIPTOR_SHA256,
        },
    ):
        raise ForagerMatchedV3CompiledRewardPublicationError(
            "publication descriptor binding drifted"
        )
    dependency = _require_exact_keys(
        body["compiled_reward_bundle"],
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
        "compiled reward bundle binding",
    )
    fixed_dependency = {
        "descriptor_schema_version": _COMPILED_BUNDLE_DESCRIPTOR_SCHEMA_VERSION,
        "descriptor_sha256": _COMPILED_BUNDLE_DESCRIPTOR_SHA256,
        "manifest_schema_version": _COMPILED_BUNDLE_MANIFEST_SCHEMA_VERSION,
        "implementation_path": _COMPILED_BUNDLE_SOURCE_PATH,
        "implementation_source_sha256": _COMPILED_BUNDLE_SOURCE_SHA256,
    }
    if not all(
        type(dependency[key]) is str and dependency[key] == expected
        for key, expected in fixed_dependency.items()
    ):
        raise ForagerMatchedV3CompiledRewardPublicationError(
            "compiled reward bundle fixed binding drifted"
        )
    _require_sha256(dependency["manifest_body_sha256"], "compiled bundle manifest body digest")
    _require_sha256(dependency["manifest_file_sha256"], "compiled bundle manifest file digest")
    _validate_file_records(body["files"])
    files = cast(dict[str, Any], body["files"])
    manifest_record = cast(dict[str, Any], files["compiled_bundle_manifest"])
    if dependency["manifest_file_sha256"] != manifest_record["sha256"]:
        raise ForagerMatchedV3CompiledRewardPublicationError(
            "compiled bundle manifest digest bindings disagree"
        )
    if not _exact_json_equal(body["writer_contract"], _writer_contract()):
        raise ForagerMatchedV3CompiledRewardPublicationError(
            "compiled reward publication writer contract drifted"
        )
    claims = _require_exact_keys(body["claims"], frozenset(_claims()), "publication claims")
    if any(value is not False for value in claims.values()):
        raise ForagerMatchedV3CompiledRewardPublicationError(
            "publication claims must be exact false booleans"
        )
    if not _exact_json_equal(body["limitations"], _limitations()):
        raise ForagerMatchedV3CompiledRewardPublicationError("publication limitations drifted")


def parse_compiled_reward_publication_manifest(
    raw: bytes,
    *,
    expected_publication_file_sha256: str,
) -> dict[str, Any]:
    """Parse a canonical outer manifest under a required external full-file digest."""

    expected = _require_sha256(
        expected_publication_file_sha256,
        "expected publication manifest file digest",
    )
    if type(raw) is not bytes or not hmac.compare_digest(hashlib.sha256(raw).hexdigest(), expected):
        raise ForagerMatchedV3CompiledRewardPublicationError(
            "publication manifest does not match its external full-file digest"
        )
    payload = _strict_json_object(
        raw,
        label="compiled reward publication manifest",
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
                "compiled_reward_bundle",
                "files",
                "writer_contract",
                "claims",
                "limitations",
                "publication_body_sha256",
            }
        ),
        "compiled reward publication manifest",
    )
    supplied_body_sha256 = _require_sha256(
        payload["publication_body_sha256"], "publication body digest"
    )
    body = dict(payload)
    del body["publication_body_sha256"]
    calculated = hashlib.sha256(
        _canonical_json(
            body,
            label="compiled reward publication body",
            maximum=_MAX_PUBLICATION_BYTES,
        )
    ).hexdigest()
    if not hmac.compare_digest(calculated, supplied_body_sha256):
        raise ForagerMatchedV3CompiledRewardPublicationError(
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
        raise ForagerMatchedV3CompiledRewardPublicationError(f"cannot safely open {label}") from exc
    try:
        opened = os.fstat(descriptor)
        current = os.lstat(canonical)
        identity = _inode_identity(opened)
        if (
            not stat.S_ISDIR(path_metadata.st_mode)
            or not stat.S_ISDIR(opened.st_mode)
            or _inode_identity(path_metadata) != identity
            or _inode_identity(current) != identity
            or (required_mode is not None and stat.S_IMODE(opened.st_mode) != required_mode)
        ):
            raise ForagerMatchedV3CompiledRewardPublicationError(
                f"{label} changed or has an unsafe mode"
            )
        return _OpenDirectory(canonical, descriptor, identity)
    except ForagerMatchedV3CompiledRewardPublicationError:
        _close_no_raise(descriptor)
        raise
    except OSError as exc:
        _close_no_raise(descriptor)
        raise ForagerMatchedV3CompiledRewardPublicationError(
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
        raise ForagerMatchedV3CompiledRewardPublicationError(f"cannot safely open {label}") from exc
    try:
        opened = os.fstat(descriptor)
        current = os.stat(name, dir_fd=parent.descriptor, follow_symlinks=False)
        identity = _inode_identity(opened)
        if (
            not stat.S_ISDIR(path_metadata.st_mode)
            or not stat.S_ISDIR(opened.st_mode)
            or _inode_identity(path_metadata) != identity
            or _inode_identity(current) != identity
            or (required_mode is not None and stat.S_IMODE(opened.st_mode) != required_mode)
        ):
            raise ForagerMatchedV3CompiledRewardPublicationError(
                f"{label} changed or has an unsafe mode"
            )
        return _OpenDirectory(path, descriptor, identity)
    except ForagerMatchedV3CompiledRewardPublicationError:
        _close_no_raise(descriptor)
        raise
    except OSError as exc:
        _close_no_raise(descriptor)
        raise ForagerMatchedV3CompiledRewardPublicationError(
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
        raise ForagerMatchedV3CompiledRewardPublicationError(
            f"{label} is no longer reachable"
        ) from exc
    if (
        not stat.S_ISDIR(opened.st_mode)
        or not stat.S_ISDIR(current.st_mode)
        or _inode_identity(opened) != root.inode_identity
        or _inode_identity(current) != root.inode_identity
    ):
        raise ForagerMatchedV3CompiledRewardPublicationError(
            f"{label} no longer names the opened inode"
        )


def _root_inventory(root: _OpenDirectory) -> dict[str, tuple[int, ...]]:
    opened_root = os.fstat(root.descriptor)
    if (
        not stat.S_ISDIR(opened_root.st_mode)
        or stat.S_IMODE(opened_root.st_mode) != _DIRECTORY_MODE
    ):
        raise ForagerMatchedV3CompiledRewardPublicationError(
            "compiled publication root mode or type is unsafe"
        )
    expected = {PUBLICATION_MANIFEST_FILENAME, *_FILE_PATHS.values()}
    names: set[str] = set()
    inventory: dict[str, tuple[int, ...]] = {}
    try:
        iterator = os.scandir(root.descriptor)
    except OSError as exc:
        raise ForagerMatchedV3CompiledRewardPublicationError(
            "cannot enumerate compiled publication root"
        ) from exc
    with iterator:
        for entry in iterator:
            if len(names) >= _MAX_ROOT_ENTRIES:
                raise ForagerMatchedV3CompiledRewardPublicationError(
                    "compiled publication root exceeds its entry bound"
                )
            if entry.name in names:
                raise ForagerMatchedV3CompiledRewardPublicationError(
                    "compiled publication root repeats an entry"
                )
            names.add(entry.name)
            try:
                metadata = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise ForagerMatchedV3CompiledRewardPublicationError(
                    "cannot inspect compiled publication entry"
                ) from exc
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or stat.S_IMODE(metadata.st_mode) != _FILE_MODE
                or (metadata.st_uid, metadata.st_gid) != (opened_root.st_uid, opened_root.st_gid)
            ):
                raise ForagerMatchedV3CompiledRewardPublicationError(
                    "compiled publication contains a link, special file, unsafe mode, "
                    "or owner mismatch"
                )
            inventory[entry.name] = _stat_identity(metadata)
    if names != expected:
        raise ForagerMatchedV3CompiledRewardPublicationError(
            "compiled publication inventory differs; "
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
        raise ForagerMatchedV3CompiledRewardPublicationError(f"cannot safely open {label}") from exc
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
            or (before.st_uid, before.st_gid) != (root_metadata.st_uid, root_metadata.st_gid)
            or _stat_identity(path_metadata) != _stat_identity(before)
        ):
            raise ForagerMatchedV3CompiledRewardPublicationError(
                f"{label} is not a bounded single-link publication file"
            )
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise ForagerMatchedV3CompiledRewardPublicationError(
                    f"{label} ended while being read"
                )
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ForagerMatchedV3CompiledRewardPublicationError(f"{label} grew while being read")
        after = os.fstat(descriptor)
        current = os.stat(name, dir_fd=root.descriptor, follow_symlinks=False)
        if _stat_identity(before) != _stat_identity(after) or _stat_identity(
            after
        ) != _stat_identity(current):
            raise ForagerMatchedV3CompiledRewardPublicationError(
                f"{label} changed while being read"
            )
        return b"".join(chunks)
    finally:
        _close_no_raise(descriptor)


def _load_from_open_root(
    root: _OpenDirectory,
    *,
    expected_publication_file_sha256: str,
) -> ContentVerifiedCompiledRewardPublication:
    initial_inventory = _root_inventory(root)
    publication_bytes = _read_stable_regular_at(
        root,
        PUBLICATION_MANIFEST_FILENAME,
        "publication manifest",
        maximum=_MAX_PUBLICATION_BYTES,
    )
    manifest = parse_compiled_reward_publication_manifest(
        publication_bytes,
        expected_publication_file_sha256=expected_publication_file_sha256,
    )
    files = cast(dict[str, Any], manifest["files"])
    limits = {
        "compiled_bundle_manifest": (_MAX_BUNDLE_MANIFEST_BYTES, None),
        "runner_result_receipt": (_MAX_RUNNER_RECEIPT_BYTES, None),
        "runtime_identity": (_MAX_RUNTIME_IDENTITY_BYTES, None),
        "reward_trace": (_CANONICAL_NPZ_SIZE_BYTES, _CANONICAL_NPZ_SIZE_BYTES),
        "score_receipt": (_MAX_SCORE_RECEIPT_BYTES, None),
    }
    loaded: dict[str, bytes] = {}
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
        if len(raw) != record["size_bytes"] or not hmac.compare_digest(
            hashlib.sha256(raw).hexdigest(), cast(str, record["sha256"])
        ):
            raise ForagerMatchedV3CompiledRewardPublicationError(
                f"publication file {role} differs from its outer binding"
            )
        loaded[role] = raw
    dependency = cast(dict[str, Any], manifest["compiled_reward_bundle"])
    reconstructed = compiled_bundle.MatchedV3CompiledRewardBundle(
        candidate_id="adapted_ppo_gru",
        runner_receipt_bytes=loaded["runner_result_receipt"],
        runtime_identity_bytes=loaded["runtime_identity"],
        reward_artifact_bytes=loaded["reward_trace"],
        score_receipt_bytes=loaded["score_receipt"],
        manifest_bytes=loaded["compiled_bundle_manifest"],
        manifest_sha256=cast(str, dependency["manifest_body_sha256"]),
    )
    try:
        validated = compiled_bundle.validate_compiled_reward_bundle(reconstructed)
    except compiled_bundle.ForagerMatchedV3CompiledRewardBundleError as exc:
        raise ForagerMatchedV3CompiledRewardPublicationError(
            "persisted compiled reward bundle failed structural replay"
        ) from exc
    replayed_bytes, replayed_body_sha256, replayed_file_sha256 = _publication_manifest_bytes(
        validated
    )
    if (
        not hmac.compare_digest(replayed_bytes, publication_bytes)
        or not hmac.compare_digest(replayed_file_sha256, expected_publication_file_sha256)
        or not hmac.compare_digest(
            replayed_body_sha256, cast(str, manifest["publication_body_sha256"])
        )
    ):
        raise ForagerMatchedV3CompiledRewardPublicationError(
            "publication manifest does not replay from its exact payload files"
        )
    if _root_inventory(root) != initial_inventory:
        raise ForagerMatchedV3CompiledRewardPublicationError(
            "compiled publication inventory changed during replay"
        )
    return ContentVerifiedCompiledRewardPublication(
        output_root=root.path,
        candidate_id="adapted_ppo_gru",
        publication_file_sha256=replayed_file_sha256,
        publication_body_sha256=replayed_body_sha256,
        manifest=cast(Mapping[str, Any], _freeze_json(manifest)),
        bundle=validated,
    )


def load_compiled_reward_bundle_publication(
    root: Path,
    *,
    expected_publication_file_sha256: str,
) -> ContentVerifiedCompiledRewardPublication:
    """Load one exact publication under a caller-carried full-file digest."""

    _require_sha256(
        expected_publication_file_sha256,
        "expected publication manifest file digest",
    )
    if type(root) is not _CONCRETE_PATH_TYPE:
        raise ForagerMatchedV3CompiledRewardPublicationError(
            "compiled publication root must be an exact pathlib Path"
        )
    opened = _open_stable_directory(
        root,
        "compiled publication root",
        required_mode=_DIRECTORY_MODE,
    )
    try:
        result = _load_from_open_root(
            opened,
            expected_publication_file_sha256=expected_publication_file_sha256,
        )
        _assert_open_directory_path(opened, "compiled publication root")
        return result
    except ForagerMatchedV3CompiledRewardPublicationError:
        raise
    except OSError as exc:
        raise ForagerMatchedV3CompiledRewardPublicationError(
            "compiled publication filesystem replay failed"
        ) from exc
    finally:
        _close_no_raise(opened.descriptor)


def _write_exclusive_at(root: _OpenDirectory, name: str, raw: bytes) -> None:
    if type(raw) is not bytes or not raw:
        raise ForagerMatchedV3CompiledRewardPublicationError(
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
        raise ForagerMatchedV3CompiledRewardPublicationError(
            f"cannot stage publication file {name!r}"
        ) from exc
    try:
        os.fchmod(descriptor, _FILE_MODE)
        remaining = memoryview(raw)
        while remaining:
            written = os.write(descriptor, remaining)
            if written < 1:
                raise ForagerMatchedV3CompiledRewardPublicationError(
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
            or (opened.st_uid, opened.st_gid) != (root_metadata.st_uid, root_metadata.st_gid)
            or _stat_identity(opened) != _stat_identity(current)
        ):
            raise ForagerMatchedV3CompiledRewardPublicationError(
                f"staged publication file {name!r} changed while writing"
            )
    except OSError as exc:
        raise ForagerMatchedV3CompiledRewardPublicationError(
            f"cannot stage publication file {name!r}"
        ) from exc
    finally:
        _close_no_raise(descriptor)


def _durably_sync_open_tree(root: _OpenDirectory) -> None:
    _assert_open_directory_path(root, "compiled publication staging directory")
    initial_inventory = _root_inventory(root)
    try:
        for name in sorted(initial_inventory, key=os.fsencode):
            expected = os.stat(name, dir_fd=root.descriptor, follow_symlinks=False)
            descriptor = os.open(name, _file_read_flags(), dir_fd=root.descriptor)
            try:
                opened = os.fstat(descriptor)
                if _stat_identity(expected) != _stat_identity(opened):
                    raise ForagerMatchedV3CompiledRewardPublicationError(
                        "staged publication file changed before fsync"
                    )
                os.fsync(descriptor)
                current = os.stat(name, dir_fd=root.descriptor, follow_symlinks=False)
                if _stat_identity(opened) != _stat_identity(os.fstat(descriptor)) or _stat_identity(
                    opened
                ) != _stat_identity(current):
                    raise ForagerMatchedV3CompiledRewardPublicationError(
                        "staged publication file changed during fsync"
                    )
            finally:
                _close_no_raise(descriptor)
        os.fsync(root.descriptor)
    except OSError as exc:
        raise ForagerMatchedV3CompiledRewardPublicationError(
            "cannot durably sync compiled publication staging tree"
        ) from exc
    if _root_inventory(root) != initial_inventory:
        raise ForagerMatchedV3CompiledRewardPublicationError(
            "staged publication inventory changed during fsync"
        )
    _assert_open_directory_path(root, "compiled publication staging directory")


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
        raise ForagerMatchedV3CompiledRewardPublicationError(
            "renameat2 is required for exclusive compiled publication"
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
            raise ForagerMatchedV3CompiledRewardPublicationError(
                "compiled publication destination was created concurrently"
            )
        raise ForagerMatchedV3CompiledRewardPublicationError(
            f"exclusive compiled publication failed with errno {error}"
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
    _assert_open_directory_path(parent, "compiled publication parent")
    if not _parent_entry_matches_open_directory(parent, source_name, staging):
        raise ForagerMatchedV3CompiledRewardPublicationError(
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
            raise PublishedCompiledRewardPublicationUncertainError(
                destination,
                "exclusive move outcome is uncertain",
                publication_file_sha256=publication_file_sha256,
                publication_body_sha256=publication_body_sha256,
            ) from exc
        raise
    try:
        if not _parent_entry_matches_open_directory(parent, destination_name, staging):
            raise ForagerMatchedV3CompiledRewardPublicationError(
                "published destination differs from the verified staging inode"
            )
        if _entry_exists(parent, source_name):
            raise ForagerMatchedV3CompiledRewardPublicationError(
                "staging name survived exclusive publication"
            )
        _sync_publication_parent(parent)
        if not _parent_entry_matches_open_directory(parent, destination_name, staging):
            raise ForagerMatchedV3CompiledRewardPublicationError(
                "published destination changed during parent fsync"
            )
        _assert_open_directory_path(parent, "compiled publication parent")
    except BaseException as exc:
        raise PublishedCompiledRewardPublicationUncertainError(
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
        root_metadata = os.fstat(staging.descriptor)
        with os.scandir(staging.descriptor) as iterator:
            for entry in iterator:
                if len(entries) >= _MAX_ROOT_ENTRIES:
                    return
                metadata = entry.stat(follow_symlinks=False)
                if (
                    not stat.S_ISREG(metadata.st_mode)
                    or metadata.st_nlink != 1
                    or stat.S_IMODE(metadata.st_mode) != _FILE_MODE
                    or (metadata.st_uid, metadata.st_gid)
                    != (root_metadata.st_uid, root_metadata.st_gid)
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


def _open_destination_parent(output_root: Path) -> tuple[_OpenDirectory, Path]:
    if type(output_root) is not _CONCRETE_PATH_TYPE:
        raise ForagerMatchedV3CompiledRewardPublicationError(
            "compiled publication destination must be an exact pathlib Path"
        )
    if _PORTABLE_NAME_RE.fullmatch(output_root.name) is None:
        raise ForagerMatchedV3CompiledRewardPublicationError(
            "compiled publication output name is not portable"
        )
    if output_root.exists() or output_root.is_symlink():
        raise ForagerMatchedV3CompiledRewardPublicationError(
            "compiled publication output root already exists"
        )
    try:
        prospective = output_root.resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise ForagerMatchedV3CompiledRewardPublicationError(
            "compiled publication output path cannot be resolved"
        ) from exc
    parent = _open_stable_directory(output_root.parent, "compiled publication parent")
    try:
        parent_metadata = os.fstat(parent.descriptor)
        if parent_metadata.st_uid != os.geteuid() or stat.S_IMODE(parent_metadata.st_mode) & 0o022:
            raise ForagerMatchedV3CompiledRewardPublicationError(
                "compiled publication parent must be effective-UID-owned and not "
                "group/world writable"
            )
        destination = parent.path / output_root.name
        if destination != prospective:
            raise ForagerMatchedV3CompiledRewardPublicationError(
                "compiled publication parent was redirected before publication"
            )
        if _entry_exists(parent, output_root.name):
            raise ForagerMatchedV3CompiledRewardPublicationError(
                "compiled publication output root already exists"
            )
        return parent, destination
    except ForagerMatchedV3CompiledRewardPublicationError:
        _close_no_raise(parent.descriptor)
        raise
    except OSError as exc:
        _close_no_raise(parent.descriptor)
        raise ForagerMatchedV3CompiledRewardPublicationError(
            "cannot verify compiled publication parent namespace"
        ) from exc
    except BaseException:
        _close_no_raise(parent.descriptor)
        raise


def _create_owned_staging(
    parent: _OpenDirectory,
    destination: Path,
) -> tuple[str, _OpenDirectory]:
    for _ in range(_STAGING_ATTEMPTS):
        name = f".forager-v3-compiled-partial-{secrets.token_hex(16)}"
        try:
            os.mkdir(name, _DIRECTORY_MODE, dir_fd=parent.descriptor)
        except FileExistsError:
            continue
        except OSError as exc:
            raise ForagerMatchedV3CompiledRewardPublicationError(
                "cannot create compiled publication staging directory"
            ) from exc
        path = parent.path / name
        staging: _OpenDirectory | None = None
        created_identity: tuple[int, int, int] | None = None
        try:
            created = os.stat(name, dir_fd=parent.descriptor, follow_symlinks=False)
            created_identity = _inode_identity(created)
            if not stat.S_ISDIR(created.st_mode) or created.st_uid != os.geteuid():
                raise ForagerMatchedV3CompiledRewardPublicationError(
                    "new compiled publication staging inode is not owned"
                )
            os.chmod(
                name,
                _DIRECTORY_MODE,
                dir_fd=parent.descriptor,
                follow_symlinks=False,
            )
            normalized = os.stat(name, dir_fd=parent.descriptor, follow_symlinks=False)
            if (
                _inode_identity(normalized) != created_identity
                or stat.S_IMODE(normalized.st_mode) != _DIRECTORY_MODE
            ):
                raise ForagerMatchedV3CompiledRewardPublicationError(
                    "compiled publication staging inode changed during mode normalization"
                )
            staging = _open_stable_directory_at(
                parent,
                name,
                path,
                "compiled publication staging directory",
                required_mode=_DIRECTORY_MODE,
            )
            os.fchmod(staging.descriptor, _DIRECTORY_MODE)
            metadata = os.fstat(staging.descriptor)
            if (
                _inode_identity(metadata) != created_identity
                or stat.S_IMODE(metadata.st_mode) != _DIRECTORY_MODE
                or metadata.st_uid != os.geteuid()
            ):
                raise ForagerMatchedV3CompiledRewardPublicationError(
                    "compiled publication staging ownership or mode is unsafe"
                )
            if destination.parent != parent.path:
                raise ForagerMatchedV3CompiledRewardPublicationError(
                    "compiled publication destination escaped its opened parent"
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
                        "compiled publication failed staging directory",
                        required_mode=None,
                    )
                    if cleanup.inode_identity == created_identity:
                        _cleanup_owned_staging(parent, name, cleanup)
                except (ForagerMatchedV3CompiledRewardPublicationError, OSError):
                    pass
                finally:
                    if cleanup is not None:
                        _close_no_raise(cleanup.descriptor)
            else:
                try:
                    os.rmdir(name, dir_fd=parent.descriptor)
                except OSError:
                    pass
            if isinstance(exc, OSError):
                raise ForagerMatchedV3CompiledRewardPublicationError(
                    "cannot initialize compiled publication staging directory"
                ) from exc
            raise
    raise ForagerMatchedV3CompiledRewardPublicationError(
        "cannot allocate a unique compiled publication staging directory"
    )


def publish_compiled_reward_bundle(
    bundle: compiled_bundle.MatchedV3CompiledRewardBundle,
    output_root: Path,
) -> ContentVerifiedCompiledRewardPublication:
    """Atomically publish one validated structural bundle without granting authority."""

    try:
        validated = compiled_bundle.validate_compiled_reward_bundle(bundle)
    except compiled_bundle.ForagerMatchedV3CompiledRewardBundleError as exc:
        raise ForagerMatchedV3CompiledRewardPublicationError(
            "compiled reward bundle failed structural validation before publication"
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
        _write_exclusive_at(staging, PUBLICATION_MANIFEST_FILENAME, publication_bytes)
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
            "published compiled reward directory",
            required_mode=_DIRECTORY_MODE,
        )
        try:
            if not _parent_entry_matches_open_directory(
                parent, destination.name, staging
            ) or not _parent_entry_matches_open_directory(parent, destination.name, final_root):
                raise ForagerMatchedV3CompiledRewardPublicationError(
                    "published compiled reward inode changed before final replay"
                )
            result = _load_from_open_root(
                final_root,
                expected_publication_file_sha256=publication_file_sha256,
            )
            _assert_open_directory_path(final_root, "published compiled reward directory")
            _assert_open_directory_path(parent, "compiled publication parent")
            return result
        except BaseException as exc:
            raise PublishedCompiledRewardPublicationUncertainError(
                destination,
                "final content replay is uncertain",
                publication_file_sha256=publication_file_sha256,
                publication_body_sha256=publication_body_sha256,
            ) from exc
        finally:
            _close_no_raise(final_root.descriptor)
    except PublishedCompiledRewardPublicationUncertainError:
        raise
    except BaseException as exc:
        destination_matches = staging is not None and _parent_entry_matches_open_directory(
            parent, destination.name, staging
        )
        source_matches = (
            staging is not None
            and bool(staging_name)
            and _parent_entry_matches_open_directory(parent, staging_name, staging)
        )
        if published or destination_matches or (staging is not None and not source_matches):
            raise PublishedCompiledRewardPublicationUncertainError(
                destination,
                "final publication state is uncertain",
                publication_file_sha256=publication_file_sha256,
                publication_body_sha256=publication_body_sha256,
            ) from exc
        if staging is not None:
            _cleanup_owned_staging(parent, staging_name, staging)
        if isinstance(exc, OSError):
            raise ForagerMatchedV3CompiledRewardPublicationError(
                "compiled publication filesystem operation failed before publication"
            ) from exc
        raise
    finally:
        if staging is not None:
            _close_no_raise(staging.descriptor)
        _close_no_raise(parent.descriptor)


__all__ = [
    "COMPILED_BUNDLE_MANIFEST_FILENAME",
    "COMPILED_REWARD_PUBLICATION_DESCRIPTOR_SCHEMA_VERSION",
    "COMPILED_REWARD_PUBLICATION_DESCRIPTOR_SHA256",
    "COMPILED_REWARD_PUBLICATION_SCHEMA_VERSION",
    "COMPILED_REWARD_PUBLICATION_STATUS",
    "ContentVerifiedCompiledRewardPublication",
    "ForagerMatchedV3CompiledRewardPublicationError",
    "PUBLICATION_FILENAME",
    "PUBLICATION_MANIFEST_FILENAME",
    "PublishedCompiledRewardPublicationUncertainError",
    "REWARD_TRACE_FILENAME",
    "RUNNER_RESULT_RECEIPT_FILENAME",
    "RUNTIME_IDENTITY_FILENAME",
    "SCORE_RECEIPT_FILENAME",
    "canonical_compiled_reward_publication_descriptor_bytes",
    "compiled_reward_publication_descriptor",
    "load_compiled_reward_bundle_publication",
    "parse_compiled_reward_publication_descriptor",
    "parse_compiled_reward_publication_manifest",
    "publish_compiled_reward_bundle",
]
