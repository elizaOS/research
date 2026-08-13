"""Fail-closed materialization of the matched-v3 external source derivative.

The production entrypoint accepts only a clean checkout at the frozen
``continual-foragax-agents`` commit and tree.  It validates every tracked
regular worktree file against the Git tree, verifies the one exact pinned
gitlink without entering it, replays the existing four-file two-seed transform,
and publishes a Git-metadata-free derived source tree with an exact canonical
manifest.

Materialized source closure is deliberately narrower than runtime
qualification.  A valid manifest binds every regular tracked file and the
identity of every explicitly excluded gitlink, but it never authorizes
execution, result ingestion, scientific promotion, or a performance claim.
The pinned archive identity is carried as provenance; an ordinary Git checkout
cannot independently attest the archive bytes.
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
import selectors
import signal
import stat
import subprocess
import time
import unicodedata
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Final, Never, SupportsIndex, cast

from alberta_framework.benchmarks import (
    forager_matched_v3_external_seed_transport as seed_transport,
)

EXTERNAL_MATERIALIZATION_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3_external_materialization.v2"
)
EXTERNAL_MATERIALIZATION_IDENTITY_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3_external_materialization_identity.v2"
)
EXTERNAL_MATERIALIZATION_MANIFEST_FILENAME: Final = (
    ".alberta-forager-matched-v3-external-materialization.v2.json"
)
_V1_EXTERNAL_MATERIALIZATION_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3_external_materialization.v1"
)
_V1_EXTERNAL_MATERIALIZATION_IDENTITY_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3_external_materialization_identity.v1"
)
_V1_EXTERNAL_MATERIALIZATION_MANIFEST_FILENAME: Final = (
    ".alberta-forager-matched-v3-external-materialization.v1.json"
)

_MAX_MANIFEST_BYTES: Final = 64 * 1024 * 1024
_MAX_IDENTITY_BYTES: Final = 1024 * 1024
_MAX_TRACKED_FILES: Final = 1_000_000
_COPY_CHUNK_BYTES: Final = 1024 * 1024
_MAX_PATH_COMPONENTS: Final = 256
_MAX_PATH_UTF8_BYTES: Final = 4095
_MAX_PATH_COMPONENT_UTF8_BYTES: Final = 255
_MAX_GIT_IDENTITY_OUTPUT_BYTES: Final = 64 * 1024
_MAX_GIT_CONFIG_OUTPUT_BYTES: Final = 16 * 1024 * 1024
_MAX_GIT_TREE_OUTPUT_BYTES: Final = 256 * 1024 * 1024
_MAX_GIT_STDERR_BYTES: Final = 1024 * 1024
_GIT_TIMEOUT_SECONDS: Final = 30.0
_PROCESS_CLEANUP_GRACE_SECONDS: Final = 1.0
_PROCESS_POLL_INTERVAL_SECONDS: Final = 0.01
_GIT_SHA1_RE: Final = re.compile(r"[0-9a-f]{40}\Z")
_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_NTFS_DOT_GIT_SHORT_NAME_RE: Final = re.compile(r"\.?git~[0-9]{1,6}\Z")
_RENAME_NOREPLACE: Final = 1
_AT_FDCWD: Final = -100


class ExternalMaterializationError(ValueError):
    """The checkout, materialization identity, destination, or manifest is invalid."""


@dataclass(frozen=True)
class SourceTransformPin:
    """Frozen original and derived identities for one transformed source file."""

    path: str
    upstream_size_bytes: int
    upstream_sha256: str
    derived_size_bytes: int
    derived_sha256: str


@dataclass(frozen=True)
class GitlinkPin:
    """Exact Git tree identity for content deliberately excluded from materialization."""

    path: str
    commit_git_sha1: str


@dataclass(frozen=True)
class PortablePathAliasPin:
    """Exact regular blob admitted despite an identity-bound portable path alias."""

    path: str
    blob_git_sha1: str


@dataclass(frozen=True)
class ExternalCheckoutIdentity:
    """Exact source and derivation identity accepted by the generic materializer."""

    repository_id: str
    canonical_url: str
    commit_git_sha1: str
    tree_git_sha1: str
    archive_sha256: str
    archive_size_bytes: int
    transport_schema_version: str
    transport_descriptor_sha256: str
    source_transforms: tuple[SourceTransformPin, ...]
    excluded_gitlinks: tuple[GitlinkPin, ...] = ()
    portable_path_aliases: tuple[PortablePathAliasPin, ...] = ()


@dataclass(frozen=True)
class ExternalMaterialization:
    """Published destination and its exact, non-authorizing manifest bytes.

    The returned paths are names, not retained filesystem capabilities.  A
    process running as the same OS user can replace or mutate them after this
    function returns; callers that need current assurance must verify again.
    """

    destination: Path
    manifest_path: Path
    manifest_bytes: bytes
    manifest_sha256: str

    def manifest(self) -> dict[str, Any]:
        """Return a detached manifest after revalidating the frozen bytes."""
        return parse_external_materialization_manifest(
            self.manifest_bytes,
            expected_manifest_sha256=self.manifest_sha256,
        )


@dataclass(frozen=True)
class _GitTreeFile:
    path: str
    git_mode: str
    blob_git_sha1: str
    upstream_size_bytes: int
    upstream_sha256: str
    source_device: int | None = None
    source_inode: int | None = None


@dataclass(frozen=True)
class _DerivedSourceSet:
    sources: Mapping[str, bytes]
    transport_schema_version: str
    transport_descriptor_sha256: str


@dataclass(frozen=True)
class _TreeBlob:
    git_mode: str
    blob_git_sha1: str


@dataclass
class _TreeDirectory:
    children: dict[str, _TreeDirectory | _TreeBlob]


_DeriveSources = Callable[[dict[str, bytes]], _DerivedSourceSet]

_MANIFEST_LIMITATIONS_CONSTRUCTION: Final = (
    ("The archive identity is bound as provenance; archive bytes were not supplied or verified."),
    (
        "Source closure binds every regular tracked file plus the exact identities of "
        "explicitly excluded gitlinks at the pinned Git commit."
    ),
    (
        "Gitlink worktree placeholders must be absent or stable empty real directories; "
        "their content is outside the materialized source closure."
    ),
    "Materialization never initializes, imports, copies, or executes gitlink content.",
    "Materialization does not import or execute the derived source tree.",
    ("Runtime dependencies, capabilities, RNG traces, and result handling remain unqualified."),
    (
        "Named-path stability assumes no concurrent process with the materializing "
        "effective user ID mutates the checkout, staging namespace, destination "
        "parent, or published path."
    ),
    (
        "Returned destination and manifest paths can later be replaced or mutated "
        "by a process running as the same OS user."
    ),
    (
        "Only a live PID-bound retained-descriptor context grants a filesystem "
        "capability; serialized manifests and returned paths grant none."
    ),
    (
        "The materialized tree is qualified only on a case-sensitive Linux filesystem; "
        "its exact identity-bound case-colliding paths are nonportable and must remain distinct."
    ),
    "A valid manifest grants no execution, ingestion, scientific, or promotion authority.",
)
_FROZEN_MANIFEST_LIMITATIONS_BYTES: Final = json.dumps(
    _MANIFEST_LIMITATIONS_CONSTRUCTION,
    ensure_ascii=True,
    separators=(",", ":"),
).encode("ascii")
_FROZEN_MANIFEST_LIMITATIONS_SHA256: Final = (
    "1aee1987836514360a3f252a1df53ecf7d293775bca5f8f8dc04e2846e47bd40"
)


def _frozen_manifest_limitations() -> list[str]:
    if not hmac.compare_digest(
        _sha256(_FROZEN_MANIFEST_LIMITATIONS_BYTES),
        _FROZEN_MANIFEST_LIMITATIONS_SHA256,
    ):
        raise AssertionError("frozen external-materialization limitations drifted")
    value = json.loads(_FROZEN_MANIFEST_LIMITATIONS_BYTES.decode("ascii"))
    if type(value) is not list or any(type(item) is not str for item in value):
        raise AssertionError("frozen external-materialization limitations are invalid")
    return cast(list[str], value)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            dict(value),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (OverflowError, RecursionError, TypeError, UnicodeError, ValueError) as exc:
        raise ExternalMaterializationError("value is not finite canonical JSON") from exc


def _portable_path_key(path: str) -> str:
    return "/".join(
        unicodedata.normalize("NFKC", part).casefold() for part in PurePosixPath(path).parts
    )


def _validate_relative_path(path: object, *, context: str) -> str:
    if type(path) is not str or not path:
        raise ExternalMaterializationError(f"{context} must be a nonempty exact string")
    if unicodedata.normalize("NFKC", path) != path:
        raise ExternalMaterializationError(f"{context} must use NFKC Unicode")
    if "\\" in path or any(ord(character) < 32 or ord(character) == 127 for character in path):
        raise ExternalMaterializationError(f"{context} contains a nonportable character")
    if any(character in '<>:"|?*' for character in path):
        raise ExternalMaterializationError(f"{context} contains a reserved portable character")
    pure = PurePosixPath(path)
    if (
        pure.is_absolute()
        or pure.as_posix() != path
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise ExternalMaterializationError(f"{context} is not a canonical relative path")
    try:
        encoded_path = path.encode("utf-8", "strict")
        encoded_parts = tuple(part.encode("utf-8", "strict") for part in pure.parts)
    except UnicodeEncodeError as exc:
        raise ExternalMaterializationError(f"{context} is not strict UTF-8") from exc
    if len(pure.parts) > _MAX_PATH_COMPONENTS:
        raise ExternalMaterializationError(f"{context} exceeds the component-depth limit")
    if len(encoded_path) > _MAX_PATH_UTF8_BYTES or any(
        len(part) > _MAX_PATH_COMPONENT_UTF8_BYTES for part in encoded_parts
    ):
        raise ExternalMaterializationError(f"{context} exceeds the portable byte limit")
    windows_devices = {
        "con",
        "prn",
        "aux",
        "nul",
        *(f"com{index}" for index in range(1, 10)),
        *(f"lpt{index}" for index in range(1, 10)),
    }
    for part in pure.parts:
        normalized = unicodedata.normalize("NFKC", part).casefold()
        if part.endswith((".", " ")):
            raise ExternalMaterializationError(f"{context} has a component ending in dot or space")
        if normalized == ".git":
            raise ExternalMaterializationError(f"{context} aliases reserved Git metadata")
        if _NTFS_DOT_GIT_SHORT_NAME_RE.fullmatch(normalized) is not None:
            raise ExternalMaterializationError(f"{context} aliases reserved Git metadata")
        if normalized.split(".", 1)[0] in windows_devices:
            raise ExternalMaterializationError(f"{context} aliases a Windows device")
    for manifest_filename in (
        EXTERNAL_MATERIALIZATION_MANIFEST_FILENAME,
        _V1_EXTERNAL_MATERIALIZATION_MANIFEST_FILENAME,
    ):
        if _portable_path_key(path) == _portable_path_key(manifest_filename):
            raise ExternalMaterializationError(f"{context} aliases a reserved manifest path")
    return path


def _require_sha256(value: object, *, context: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ExternalMaterializationError(f"{context} must be a lowercase SHA-256")
    return value


def _require_git_sha1(value: object, *, context: str) -> str:
    if type(value) is not str or _GIT_SHA1_RE.fullmatch(value) is None:
        raise ExternalMaterializationError(f"{context} must be a lowercase Git SHA-1")
    return value


def _require_exact_nonnegative_int(value: object, *, context: str) -> int:
    if type(value) is not int or value < 0:
        raise ExternalMaterializationError(f"{context} must be an exact nonnegative integer")
    return value


def _portable_alias_expectations(
    pins: tuple[PortablePathAliasPin, ...],
) -> dict[str, dict[str, str]]:
    if type(pins) is not tuple:
        raise ExternalMaterializationError("portable path aliases must be an exact tuple")
    groups: dict[str, dict[str, str]] = {}
    previous_path: str | None = None
    for index, alias_pin in enumerate(pins):
        if type(alias_pin) is not PortablePathAliasPin:
            raise ExternalMaterializationError(f"portable path alias {index} has the wrong type")
        path = _validate_relative_path(
            alias_pin.path,
            context=f"portable path alias {index} path",
        )
        if previous_path is not None and path <= previous_path:
            raise ExternalMaterializationError(
                "portable path aliases must be path-sorted and unique"
            )
        blob_git_sha1 = _require_git_sha1(
            alias_pin.blob_git_sha1,
            context=f"portable path alias {path} blob",
        )
        groups.setdefault(_portable_path_key(path), {})[path] = blob_git_sha1
        previous_path = path
    if any(len(group) < 2 for group in groups.values()):
        raise ExternalMaterializationError(
            "each portable path alias exception must contain at least two colliding paths"
        )
    return groups


def _validate_observed_portable_aliases(
    records: Sequence[tuple[str, str, str]],
    pins: tuple[PortablePathAliasPin, ...],
    *,
    context: str,
) -> None:
    expected = _portable_alias_expectations(pins)
    observed: dict[str, dict[str, tuple[str, str]]] = {}
    observed_nodes: dict[str, set[tuple[str, str]]] = {}
    observed_paths: set[str] = set()
    for path, mode, object_id in records:
        path = _validate_relative_path(path, context=f"{context} path")
        if path in observed_paths:
            raise ExternalMaterializationError(f"{context} contains duplicate paths")
        observed_paths.add(path)
        observed.setdefault(_portable_path_key(path), {})[path] = (mode, object_id)
        parts = PurePosixPath(path).parts
        for depth in range(1, len(parts)):
            ancestor = "/".join(parts[:depth])
            observed_nodes.setdefault(_portable_path_key(ancestor), set()).add(
                ("directory", ancestor)
            )
        leaf_kind = "regular" if mode in {"100644", "100755"} else mode
        observed_nodes.setdefault(_portable_path_key(path), set()).add((leaf_kind, path))
    for alias_key, nodes in observed_nodes.items():
        if len(nodes) < 2:
            continue
        expected_group = expected.get(alias_key, {})
        regular_paths = {path for kind, path in nodes if kind == "regular"}
        if len(nodes) != len(expected_group) or regular_paths != set(expected_group):
            raise ExternalMaterializationError(
                f"{context} portable path aliases contain component collisions outside "
                "the exact identity exception"
            )
    for alias_key in set(expected) | set(observed):
        expected_group = expected.get(alias_key, {})
        observed_group = observed.get(alias_key, {})
        if len(observed_group) < 2 and not expected_group:
            continue
        exact_observed = {
            path: object_id
            for path, (mode, object_id) in observed_group.items()
            if mode in {"100644", "100755"}
        }
        if exact_observed != expected_group or len(exact_observed) != len(observed_group):
            raise ExternalMaterializationError(
                f"{context} portable path aliases do not match the exact identity exception"
            )


def _verify_distinct_portable_alias_identities(
    identities: Mapping[str, tuple[int, int]],
    pins: tuple[PortablePathAliasPin, ...],
    *,
    context: str,
) -> None:
    for group in _portable_alias_expectations(pins).values():
        try:
            group_identities = [identities[path] for path in group]
        except KeyError as exc:
            raise ExternalMaterializationError(
                f"{context} omits an identity-bound portable path alias"
            ) from exc
        if len(set(group_identities)) != len(group_identities):
            raise ExternalMaterializationError(
                f"{context} does not preserve portable path aliases as distinct inodes"
            )


def _identity_payload(identity: ExternalCheckoutIdentity) -> dict[str, Any]:
    if type(identity) is not ExternalCheckoutIdentity:
        raise ExternalMaterializationError("identity must be an exact ExternalCheckoutIdentity")
    for field_name in (
        "repository_id",
        "canonical_url",
        "transport_schema_version",
    ):
        value = getattr(identity, field_name)
        if type(value) is not str or not value:
            raise ExternalMaterializationError(f"identity {field_name} must be nonempty")
    _require_git_sha1(identity.commit_git_sha1, context="identity commit")
    _require_git_sha1(identity.tree_git_sha1, context="identity tree")
    _require_sha256(identity.archive_sha256, context="identity archive")
    _require_exact_nonnegative_int(identity.archive_size_bytes, context="identity archive size")
    _require_sha256(
        identity.transport_descriptor_sha256,
        context="identity transport descriptor",
    )
    if type(identity.source_transforms) is not tuple or not identity.source_transforms:
        raise ExternalMaterializationError("identity source transforms must be a nonempty tuple")

    records: list[dict[str, Any]] = []
    path_keys: set[str] = set()
    previous_path: str | None = None
    for index, transform_pin in enumerate(identity.source_transforms):
        if type(transform_pin) is not SourceTransformPin:
            raise ExternalMaterializationError(
                f"identity source transform {index} has the wrong type"
            )
        path = _validate_relative_path(
            transform_pin.path,
            context=f"source transform {index} path",
        )
        key = _portable_path_key(path)
        if key in path_keys:
            raise ExternalMaterializationError("identity source transforms contain aliases")
        if previous_path is not None and path <= previous_path:
            raise ExternalMaterializationError(
                "identity source transforms must be path-sorted and unique"
            )
        path_keys.add(key)
        previous_path = path
        records.append(
            {
                "path": path,
                "upstream_size_bytes": _require_exact_nonnegative_int(
                    transform_pin.upstream_size_bytes,
                    context=f"source transform {path} upstream size",
                ),
                "upstream_sha256": _require_sha256(
                    transform_pin.upstream_sha256,
                    context=f"source transform {path} upstream digest",
                ),
                "derived_size_bytes": _require_exact_nonnegative_int(
                    transform_pin.derived_size_bytes,
                    context=f"source transform {path} derived size",
                ),
                "derived_sha256": _require_sha256(
                    transform_pin.derived_sha256,
                    context=f"source transform {path} derived digest",
                ),
            }
        )
    if type(identity.excluded_gitlinks) is not tuple:
        raise ExternalMaterializationError("identity excluded gitlinks must be an exact tuple")
    gitlink_records: list[dict[str, Any]] = []
    previous_gitlink_path: str | None = None
    gitlink_paths: list[str] = []
    for index, gitlink_pin in enumerate(identity.excluded_gitlinks):
        if type(gitlink_pin) is not GitlinkPin:
            raise ExternalMaterializationError(
                f"identity excluded gitlink {index} has the wrong type"
            )
        path = _validate_relative_path(
            gitlink_pin.path,
            context=f"excluded gitlink {index} path",
        )
        if path == ".gitmodules":
            raise ExternalMaterializationError(".gitmodules cannot itself be a gitlink")
        key = _portable_path_key(path)
        if key in path_keys:
            raise ExternalMaterializationError("identity paths contain duplicate aliases")
        if previous_gitlink_path is not None and path <= previous_gitlink_path:
            raise ExternalMaterializationError(
                "identity excluded gitlinks must be path-sorted and unique"
            )
        path_keys.add(key)
        previous_gitlink_path = path
        gitlink_paths.append(path)
        gitlink_records.append(
            {
                "path": path,
                "commit_git_sha1": _require_git_sha1(
                    gitlink_pin.commit_git_sha1,
                    context=f"excluded gitlink {path} commit",
                ),
            }
        )
    for index, path in enumerate(gitlink_paths):
        prefix = f"{path}/"
        if any(other.startswith(prefix) for other in gitlink_paths[index + 1 :]):
            raise ExternalMaterializationError("identity gitlinks cannot contain one another")
        if any(
            transform.path == path
            or transform.path.startswith(prefix)
            or path.startswith(f"{transform.path}/")
            for transform in identity.source_transforms
        ):
            raise ExternalMaterializationError(
                "identity source transforms cannot be inside an excluded gitlink"
            )
    portable_alias_groups = _portable_alias_expectations(identity.portable_path_aliases)
    for alias_group in portable_alias_groups.values():
        for alias_path in alias_group:
            alias_key = _portable_path_key(alias_path)
            for gitlink_path in gitlink_paths:
                gitlink_key = _portable_path_key(gitlink_path)
                if (
                    alias_key == gitlink_key
                    or alias_key.startswith(f"{gitlink_key}/")
                    or gitlink_key.startswith(f"{alias_key}/")
                ):
                    raise ExternalMaterializationError(
                        "portable path alias exceptions must identify regular files "
                        "outside gitlinks"
                    )
    portable_alias_records = [
        {
            "path": item.path,
            "blob_git_sha1": item.blob_git_sha1,
        }
        for item in identity.portable_path_aliases
    ]
    return {
        "schema_version": EXTERNAL_MATERIALIZATION_IDENTITY_SCHEMA_VERSION,
        "repository_id": identity.repository_id,
        "canonical_url": identity.canonical_url,
        "commit_git_sha1": identity.commit_git_sha1,
        "tree_git_sha1": identity.tree_git_sha1,
        "archive_sha256": identity.archive_sha256,
        "archive_size_bytes": identity.archive_size_bytes,
        "transport_schema_version": identity.transport_schema_version,
        "transport_descriptor_sha256": identity.transport_descriptor_sha256,
        "source_transforms": records,
        "excluded_gitlinks": gitlink_records,
        "portable_path_aliases": portable_alias_records,
    }


def _identity_from_payload(value: object) -> ExternalCheckoutIdentity:
    if type(value) is not dict:
        raise ExternalMaterializationError("identity payload must be a plain object")
    payload = cast(dict[str, Any], value)
    expected_keys = {
        "schema_version",
        "repository_id",
        "canonical_url",
        "commit_git_sha1",
        "tree_git_sha1",
        "archive_sha256",
        "archive_size_bytes",
        "transport_schema_version",
        "transport_descriptor_sha256",
        "source_transforms",
        "excluded_gitlinks",
        "portable_path_aliases",
    }
    if set(payload) != expected_keys:
        raise ExternalMaterializationError("identity payload fields do not match")
    if payload["schema_version"] != EXTERNAL_MATERIALIZATION_IDENTITY_SCHEMA_VERSION:
        raise ExternalMaterializationError("identity schema version does not match")
    raw_transforms = payload["source_transforms"]
    if type(raw_transforms) is not list:
        raise ExternalMaterializationError("identity source transforms must be a list")
    transforms: list[SourceTransformPin] = []
    for index, raw_item in enumerate(raw_transforms):
        if type(raw_item) is not dict:
            raise ExternalMaterializationError(f"source transform {index} must be an object")
        item = cast(dict[str, Any], raw_item)
        if set(item) != {
            "path",
            "upstream_size_bytes",
            "upstream_sha256",
            "derived_size_bytes",
            "derived_sha256",
        }:
            raise ExternalMaterializationError(f"source transform {index} fields do not match")
        transforms.append(
            SourceTransformPin(
                path=item["path"],
                upstream_size_bytes=item["upstream_size_bytes"],
                upstream_sha256=item["upstream_sha256"],
                derived_size_bytes=item["derived_size_bytes"],
                derived_sha256=item["derived_sha256"],
            )
        )
    raw_gitlinks = payload["excluded_gitlinks"]
    if type(raw_gitlinks) is not list:
        raise ExternalMaterializationError("identity excluded gitlinks must be a list")
    gitlinks: list[GitlinkPin] = []
    for index, raw_item in enumerate(raw_gitlinks):
        if type(raw_item) is not dict:
            raise ExternalMaterializationError(f"excluded gitlink {index} must be an object")
        item = cast(dict[str, Any], raw_item)
        if set(item) != {"path", "commit_git_sha1"}:
            raise ExternalMaterializationError(f"excluded gitlink {index} fields do not match")
        gitlinks.append(
            GitlinkPin(
                path=item["path"],
                commit_git_sha1=item["commit_git_sha1"],
            )
        )
    raw_portable_aliases = payload["portable_path_aliases"]
    if type(raw_portable_aliases) is not list:
        raise ExternalMaterializationError("identity portable path aliases must be a list")
    portable_aliases: list[PortablePathAliasPin] = []
    for index, raw_item in enumerate(raw_portable_aliases):
        if type(raw_item) is not dict:
            raise ExternalMaterializationError(f"portable path alias {index} must be an object")
        item = cast(dict[str, Any], raw_item)
        if set(item) != {"path", "blob_git_sha1"}:
            raise ExternalMaterializationError(f"portable path alias {index} fields do not match")
        portable_aliases.append(
            PortablePathAliasPin(
                path=item["path"],
                blob_git_sha1=item["blob_git_sha1"],
            )
        )
    identity = ExternalCheckoutIdentity(
        repository_id=payload["repository_id"],
        canonical_url=payload["canonical_url"],
        commit_git_sha1=payload["commit_git_sha1"],
        tree_git_sha1=payload["tree_git_sha1"],
        archive_sha256=payload["archive_sha256"],
        archive_size_bytes=payload["archive_size_bytes"],
        transport_schema_version=payload["transport_schema_version"],
        transport_descriptor_sha256=payload["transport_descriptor_sha256"],
        source_transforms=tuple(transforms),
        excluded_gitlinks=tuple(gitlinks),
        portable_path_aliases=tuple(portable_aliases),
    )
    if _identity_payload(identity) != payload:
        raise ExternalMaterializationError("identity payload is not exact")
    return identity


_PINNED_IDENTITY_CONSTRUCTION: Final = ExternalCheckoutIdentity(
    repository_id="foragax_agents",
    canonical_url="https://github.com/steventango/continual-foragax-agents",
    commit_git_sha1="9710f60fa30da5badc451ad7ce3ff296d5070830",
    tree_git_sha1="a5ad878ac4be0567c43dfd9177471c4b5a910bfa",
    archive_sha256=("1f6976de38f34a697c947891de26ad3373b294195fe82094e9d1d5b8ddfd43b6"),
    archive_size_bytes=314_961_920,
    transport_schema_version="alberta.forager_matched_v3_external_seed_transport.v1",
    transport_descriptor_sha256=(
        "66be593917a47c8eca4e1a3227407e060ebb52ac835e4207dc32fc81de7d13ad"
    ),
    source_transforms=(
        SourceTransformPin(
            path="src/continuing_main.py",
            upstream_size_bytes=32_190,
            upstream_sha256=("681c2dae9569a0bbd72c8f47a3a63d51176071308f9762f3d81855da79c3aebf"),
            derived_size_bytes=33_029,
            derived_sha256=("ca9748cf92107b41c1d1e6cd17d4a1a3c517fa5921c55469c1e66a73ef8d2551"),
        ),
        SourceTransformPin(
            path="src/problems/BaseProblem.py",
            upstream_size_bytes=1_548,
            upstream_sha256=("1985825dfa257570c605a4f3704f4dc648775398008507761d76bc46d7c835d0"),
            derived_size_bytes=1_719,
            derived_sha256=("a4ab77408c1bb38dd3f4e72d830765176c38bba4b73b69fe296765a0272d87dc"),
        ),
        SourceTransformPin(
            path="src/problems/Foragax.py",
            upstream_size_bytes=1_069,
            upstream_sha256=("f901d20109a35791c6ed8a8b3ddad97707645eea49461470a4bfa63ae3b40fea"),
            derived_size_bytes=1_316,
            derived_sha256=("ff6e875511fcc574bafde7f114382dccf5303dba96f4154d5abbc16744d8e7c9"),
        ),
        SourceTransformPin(
            path="src/rtu_ppo.py",
            upstream_size_bytes=89_937,
            upstream_sha256=("e75a6762690832067a24a649559a55e0aa89abba005d600f090b1bf284b3fc24"),
            derived_size_bytes=91_286,
            derived_sha256=("1859b4cde5695fcedd5cd21280caa0df029057e1b90e364f3bace225d127f3f1"),
        ),
    ),
    excluded_gitlinks=(
        GitlinkPin(
            path="continual-foragax-loss-of-plasticity",
            commit_git_sha1="8880f3f241ec441e584416b61b0579fca3bc1ef4",
        ),
    ),
    portable_path_aliases=(
        PortablePathAliasPin(
            path=(
                "experiments/R2-plasticity/foragax/ForagaxSquareWaveTwoBiome-v11/"
                "metrics/NTKRank_LOP_vs_NoLOP.png"
            ),
            blob_git_sha1="566e89612c822a72f39fa84f8f1c4ed65d1c2788",
        ),
        PortablePathAliasPin(
            path=(
                "experiments/R2-plasticity/foragax/ForagaxSquareWaveTwoBiome-v11/"
                "metrics/ntkrank_LOP_vs_NoLOP.png"
            ),
            blob_git_sha1="566e89612c822a72f39fa84f8f1c4ed65d1c2788",
        ),
    ),
)
_PINNED_IDENTITY_BYTES: Final = _canonical_json(_identity_payload(_PINNED_IDENTITY_CONSTRUCTION))
PINNED_EXTERNAL_MATERIALIZATION_IDENTITY_SHA256: Final = (
    "74cf45b9d09b06c17dd38c8713940f32a04e887259bb027c75bfa680e7b43192"
)
if not hmac.compare_digest(
    _sha256(_PINNED_IDENTITY_BYTES),
    PINNED_EXTERNAL_MATERIALIZATION_IDENTITY_SHA256,
):
    raise AssertionError("pinned external materialization identity drifted")
_PINNED_TRANSPORT_DESCRIPTOR_BYTES: Final = (
    seed_transport.canonical_matched_v3_external_seed_transport_descriptor_bytes()
)


def pinned_external_checkout_identity() -> ExternalCheckoutIdentity:
    """Decode the production identity from its frozen canonical bytes."""
    return parse_pinned_external_checkout_identity(_PINNED_IDENTITY_BYTES)


def parse_pinned_external_checkout_identity(raw: bytes) -> ExternalCheckoutIdentity:
    """Accept only the exact canonical production checkout identity."""
    if type(raw) is not bytes:
        raise ExternalMaterializationError("pinned checkout identity must be exact bytes")
    if not 0 < len(raw) <= _MAX_IDENTITY_BYTES:
        raise ExternalMaterializationError("pinned checkout identity byte length is invalid")
    if not hmac.compare_digest(
        _sha256(raw),
        PINNED_EXTERNAL_MATERIALIZATION_IDENTITY_SHA256,
    ):
        raise ExternalMaterializationError("pinned checkout identity digest does not match")
    if raw != _PINNED_IDENTITY_BYTES:
        raise ExternalMaterializationError("pinned checkout identity bytes do not match")
    try:
        value = json.loads(raw.decode("ascii"))
    except (RecursionError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExternalMaterializationError("pinned checkout identity is not JSON") from exc
    return _identity_from_payload(value)


def canonical_pinned_external_checkout_identity_bytes() -> bytes:
    """Return the exact production identity bytes."""
    pinned_external_checkout_identity()
    return _PINNED_IDENTITY_BYTES


@dataclass(frozen=True)
class _DirectoryAnchor:
    path: Path
    descriptor: int
    device: int
    inode: int


@dataclass(frozen=True)
class _CheckoutAnchor:
    root: _DirectoryAnchor
    git: _DirectoryAnchor


@dataclass(frozen=True)
class _HermeticGitEnvironment:
    values: Mapping[str, str]
    config_descriptor: int


def _path_text(value: os.PathLike[str] | str, *, context: str) -> Path:
    try:
        text = os.fspath(value)
        if type(text) is not str or "\0" in text:
            raise ValueError
        return Path(os.path.abspath(text))
    except (TypeError, ValueError) as exc:
        raise ExternalMaterializationError(f"{context} is not a valid path") from exc


def _open_directory_path_descriptor(path: Path, *, context: str) -> int:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = -1
    try:
        descriptor = os.open("/", flags)
        for part in path.parts[1:]:
            next_descriptor = os.open(part, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
    except OSError as exc:
        if descriptor >= 0:
            os.close(descriptor)
        raise ExternalMaterializationError(
            f"{context} and every ancestor must be real directories"
        ) from exc
    return descriptor


def _open_directory_anchor(path: Path, *, context: str) -> _DirectoryAnchor:
    descriptor = _open_directory_path_descriptor(path, context=context)
    try:
        item_stat = os.fstat(descriptor)
    except BaseException:
        os.close(descriptor)
        raise
    if not stat.S_ISDIR(item_stat.st_mode):
        os.close(descriptor)
        raise ExternalMaterializationError(f"{context} must be a directory")
    anchor = _DirectoryAnchor(path, descriptor, item_stat.st_dev, item_stat.st_ino)
    try:
        _recheck_directory_anchor(anchor, context=context)
    except BaseException:
        os.close(descriptor)
        raise
    return anchor


def _recheck_directory_anchor(anchor: _DirectoryAnchor, *, context: str) -> None:
    reopened = -1
    try:
        descriptor_stat = os.fstat(anchor.descriptor)
        reopened = _open_directory_path_descriptor(anchor.path, context=context)
        path_stat = os.fstat(reopened)
    except (OSError, ExternalMaterializationError) as exc:
        raise ExternalMaterializationError(f"{context} anchor became inaccessible") from exc
    finally:
        if reopened >= 0:
            os.close(reopened)
    identity = (anchor.device, anchor.inode)
    if (
        not stat.S_ISDIR(path_stat.st_mode)
        or (descriptor_stat.st_dev, descriptor_stat.st_ino) != identity
        or (path_stat.st_dev, path_stat.st_ino) != identity
    ):
        raise ExternalMaterializationError(f"{context} anchor identity changed")


def _directory_anchor_is_within(
    child: _DirectoryAnchor,
    ancestor: _DirectoryAnchor,
) -> bool:
    """Return whether an anchored directory is at or below another anchor."""
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.dup(child.descriptor)
    visited: set[tuple[int, int]] = set()
    ancestor_identity = (ancestor.device, ancestor.inode)
    try:
        while True:
            current_stat = os.fstat(descriptor)
            current_identity = (current_stat.st_dev, current_stat.st_ino)
            if current_identity == ancestor_identity:
                return True
            if current_identity in visited:
                raise ExternalMaterializationError(
                    "destination directory ancestry contains a cycle"
                )
            visited.add(current_identity)
            parent_descriptor = -1
            try:
                parent_descriptor = os.open("..", flags, dir_fd=descriptor)
                parent_stat = os.fstat(parent_descriptor)
            except BaseException:
                if parent_descriptor >= 0:
                    os.close(parent_descriptor)
                raise
            parent_identity = (parent_stat.st_dev, parent_stat.st_ino)
            if parent_identity == current_identity:
                os.close(parent_descriptor)
                return False
            os.close(descriptor)
            descriptor = parent_descriptor
    except OSError as exc:
        raise ExternalMaterializationError(
            "destination directory ancestry became inaccessible"
        ) from exc
    finally:
        os.close(descriptor)


def _checkout_root(checkout: os.PathLike[str] | str) -> _CheckoutAnchor:
    root = _open_directory_anchor(
        _path_text(checkout, context="checkout root"),
        context="checkout root",
    )
    git_descriptor = -1
    try:
        git_descriptor = os.open(
            ".git",
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=root.descriptor,
        )
        git_stat = os.fstat(git_descriptor)
        git_name_stat = os.stat(".git", dir_fd=root.descriptor, follow_symlinks=False)
    except OSError as exc:
        if git_descriptor >= 0:
            os.close(git_descriptor)
        os.close(root.descriptor)
        raise ExternalMaterializationError("checkout has no direct Git metadata directory") from exc
    git_identity = (git_stat.st_dev, git_stat.st_ino)
    if (
        not stat.S_ISDIR(git_stat.st_mode)
        or not stat.S_ISDIR(git_name_stat.st_mode)
        or stat.S_ISLNK(git_name_stat.st_mode)
        or (git_name_stat.st_dev, git_name_stat.st_ino) != git_identity
    ):
        os.close(git_descriptor)
        os.close(root.descriptor)
        raise ExternalMaterializationError("checkout .git metadata must be a real directory")
    git = _DirectoryAnchor(
        root.path / ".git",
        git_descriptor,
        git_stat.st_dev,
        git_stat.st_ino,
    )
    checkout_anchor = _CheckoutAnchor(root, git)
    try:
        _recheck_checkout_anchor(checkout_anchor)
    except BaseException:
        os.close(git_descriptor)
        os.close(root.descriptor)
        raise
    return checkout_anchor


def _recheck_checkout_anchor(checkout: _CheckoutAnchor) -> None:
    _recheck_directory_anchor(checkout.root, context="checkout root")
    git_descriptor_stat = os.fstat(checkout.git.descriptor)
    git_name_stat = os.stat(
        ".git",
        dir_fd=checkout.root.descriptor,
        follow_symlinks=False,
    )
    identity = (checkout.git.device, checkout.git.inode)
    if (
        not stat.S_ISDIR(git_name_stat.st_mode)
        or stat.S_ISLNK(git_name_stat.st_mode)
        or (git_descriptor_stat.st_dev, git_descriptor_stat.st_ino) != identity
        or (git_name_stat.st_dev, git_name_stat.st_ino) != identity
    ):
        raise ExternalMaterializationError("checkout .git anchor identity changed")


def _close_checkout_anchor(checkout: _CheckoutAnchor) -> None:
    os.close(checkout.git.descriptor)
    os.close(checkout.root.descriptor)


@contextmanager
def _hermetic_git_environment() -> Any:
    temp_parent = _open_directory_anchor(
        Path("/tmp"),
        context="temporary directory",
    )
    name: str | None = None
    config_descriptor = -1
    config_identity: tuple[int, int] | None = None
    primary_error: BaseException | None = None
    try:
        name, config_descriptor, config_identity = _create_anchored_temp_directory(
            temp_parent.descriptor,
            prefix=".alberta-v3-git-",
        )
        os.mkdir("home", 0o700, dir_fd=config_descriptor)
        os.mkdir("xdg", 0o700, dir_fd=config_descriptor)
        root_reference = f"/proc/self/fd/{config_descriptor}"
        environment = MappingProxyType(
            {
                "PATH": os.defpath,
                "HOME": f"{root_reference}/home",
                "XDG_CONFIG_HOME": f"{root_reference}/xdg",
                "LC_ALL": "C",
                "LANG": "C",
                "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_CONFIG_GLOBAL": "/dev/null",
                "GIT_ATTR_NOSYSTEM": "1",
                "GIT_NO_REPLACE_OBJECTS": "1",
                "GIT_NO_LAZY_FETCH": "1",
                "GIT_OPTIONAL_LOCKS": "0",
                "GIT_TERMINAL_PROMPT": "0",
                "GIT_PROTOCOL_FROM_USER": "0",
                "GIT_ALLOW_PROTOCOL": "",
                "GCM_INTERACTIVE": "never",
                "GIT_PAGER": "cat",
                "PAGER": "cat",
            }
        )
        try:
            yield _HermeticGitEnvironment(environment, config_descriptor)
        except BaseException as exc:
            primary_error = exc
            raise
    except BaseException as exc:
        if primary_error is None:
            primary_error = exc
        raise
    finally:
        cleanup_error: OSError | ExternalMaterializationError | None = None
        if config_descriptor >= 0 and config_identity is not None and name is not None:
            try:
                _safe_remove_open_directory(config_descriptor)
                os.close(config_descriptor)
                config_descriptor = -1
                _remove_anchored_directory_name(
                    temp_parent.descriptor,
                    name,
                    config_identity,
                )
            except (OSError, ExternalMaterializationError) as exc:
                cleanup_error = exc
        if config_descriptor >= 0:
            os.close(config_descriptor)
        os.close(temp_parent.descriptor)
        if cleanup_error is not None and primary_error is not None:
            primary_error.add_note(f"hermetic Git cleanup also failed: {cleanup_error}")
        elif cleanup_error is not None:
            raise ExternalMaterializationError(
                "hermetic Git configuration cleanup failed"
            ) from cleanup_error


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _run_bounded_process(
    command: Sequence[str],
    *,
    cwd: str,
    environment: Mapping[str, str],
    pass_fds: Sequence[int],
    maximum_stdout_bytes: int,
    maximum_stderr_bytes: int,
    timeout_seconds: float,
) -> tuple[int, bytes, bytes]:
    """Run a local inspection command with bounded output and wall time."""
    if maximum_stdout_bytes < 0 or maximum_stderr_bytes < 0 or timeout_seconds <= 0.0:
        raise ExternalMaterializationError("bounded process limits are invalid")
    try:
        process = subprocess.Popen(
            list(command),
            cwd=cwd,
            env=dict(environment),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            pass_fds=tuple(pass_fds),
            start_new_session=True,
        )
    except OSError as exc:
        raise ExternalMaterializationError("inspection process could not start") from exc
    if process.stdout is None or process.stderr is None:
        _terminate_process_group(process)
        process.wait()
        raise ExternalMaterializationError("inspection process pipes are unavailable")

    stdout_buffer = bytearray()
    stderr_buffer = bytearray()
    overflow: str | None = None
    reader_error: OSError | None = None
    selector = selectors.DefaultSelector()
    timed_out = False
    cleanup_timed_out = False
    deadline = time.monotonic() + timeout_seconds
    try:
        for stream, destination, maximum_bytes, label in (
            (process.stdout, stdout_buffer, maximum_stdout_bytes, "stdout"),
            (process.stderr, stderr_buffer, maximum_stderr_bytes, "stderr"),
        ):
            descriptor = stream.fileno()
            os.set_blocking(descriptor, False)
            selector.register(
                descriptor,
                selectors.EVENT_READ,
                (destination, maximum_bytes, label),
            )
        while selector.get_map():
            leader_exited = process.poll() is not None
            if leader_exited:
                # Ordinary descendants remain in this session and are killed.
                # A hostile descendant may have escaped with setsid(); never
                # wait for an inherited pipe after the leader has exited.
                _terminate_process_group(process)
                selection_timeout = 0.0
            else:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    timed_out = True
                    break
                selection_timeout = min(remaining, _PROCESS_POLL_INTERVAL_SECONDS)
            events = selector.select(selection_timeout)
            if not events:
                if leader_exited:
                    break
                if time.monotonic() >= deadline:
                    timed_out = True
                    break
                continue
            stop_reading = False
            for key, _ in events:
                destination, maximum_bytes, label = cast(tuple[bytearray, int, str], key.data)
                while True:
                    try:
                        chunk = os.read(key.fd, 64 * 1024)
                    except BlockingIOError:
                        break
                    except OSError as exc:
                        reader_error = exc
                        stop_reading = True
                        break
                    if not chunk:
                        selector.unregister(key.fd)
                        break
                    if len(destination) + len(chunk) > maximum_bytes:
                        overflow = label
                        stop_reading = True
                        break
                    destination.extend(chunk)
                if stop_reading:
                    break
            if stop_reading:
                break
    except OSError as exc:
        reader_error = exc
    finally:
        # The leader can exit after spawning a descendant that inherits a pipe.
        # Always kill the entire session, even when the leader already has a
        # return code, so pipe readers and wall time remain bounded.
        _terminate_process_group(process)
        cleanup_deadline = time.monotonic() + _PROCESS_CLEANUP_GRACE_SECONDS
        try:
            process.wait(timeout=max(0.001, cleanup_deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            cleanup_timed_out = True
        selector.close()
        process.stdout.close()
        process.stderr.close()
    if cleanup_timed_out:
        raise ExternalMaterializationError(
            "inspection process did not stop within its cleanup bound"
        )
    if reader_error is not None:
        raise ExternalMaterializationError("inspection process output could not be read") from (
            reader_error
        )
    if timed_out:
        raise ExternalMaterializationError("inspection process exceeded its time limit")
    if overflow is not None:
        raise ExternalMaterializationError(f"inspection process {overflow} exceeded its byte limit")
    if type(process.returncode) is not int:
        raise ExternalMaterializationError("inspection process has no exact return code")
    return process.returncode, bytes(stdout_buffer), bytes(stderr_buffer)


def _git(
    checkout: _CheckoutAnchor,
    git_environment: _HermeticGitEnvironment,
    arguments: Sequence[str],
    *,
    maximum_stdout_bytes: int = _MAX_GIT_IDENTITY_OUTPUT_BYTES,
) -> bytes:
    git_executable = "/usr/bin/git"
    if not os.path.isfile(git_executable) or not os.access(git_executable, os.X_OK):
        git_executable = "/bin/git"
    root_reference = f"/proc/self/fd/{checkout.root.descriptor}"
    git_reference = f"/proc/self/fd/{checkout.git.descriptor}"
    command = [
        git_executable,
        "--no-pager",
        "--no-replace-objects",
        f"--git-dir={git_reference}",
        f"--work-tree={root_reference}",
        "-c",
        "core.fsmonitor=false",
        "-c",
        "core.hooksPath=/dev/null",
        "-c",
        "core.attributesFile=/dev/null",
        "-c",
        "core.alternateRefsCommand=",
        "-c",
        "diff.external=",
        "-c",
        "submodule.recurse=false",
        "-c",
        "fetch.recurseSubmodules=false",
        "-c",
        "protocol.ext.allow=never",
        *arguments,
    ]
    returncode, stdout, stderr = _run_bounded_process(
        command,
        cwd=root_reference,
        environment=git_environment.values,
        pass_fds=(
            checkout.root.descriptor,
            checkout.git.descriptor,
            git_environment.config_descriptor,
        ),
        maximum_stdout_bytes=maximum_stdout_bytes,
        maximum_stderr_bytes=_MAX_GIT_STDERR_BYTES,
        timeout_seconds=_GIT_TIMEOUT_SECONDS,
    )
    if returncode != 0:
        detail = stderr.decode("utf-8", "replace").strip()
        raise ExternalMaterializationError(f"Git checkout inspection failed: {detail}")
    return stdout


def _parse_tree(
    raw: bytes,
    expected_gitlinks: tuple[GitlinkPin, ...] = (),
    portable_path_aliases: tuple[PortablePathAliasPin, ...] = (),
) -> tuple[tuple[str, str, str], ...]:
    if not raw or not raw.endswith(b"\0"):
        raise ExternalMaterializationError("Git tree enumeration is empty or malformed")
    if type(expected_gitlinks) is not tuple:
        raise ExternalMaterializationError("expected gitlinks must be an exact tuple")
    expected_by_path: dict[str, GitlinkPin] = {}
    expected_aliases: set[str] = set()
    previous_expected_path: str | None = None
    for index, expected_pin in enumerate(expected_gitlinks):
        if type(expected_pin) is not GitlinkPin:
            raise ExternalMaterializationError(f"expected gitlink {index} has the wrong type")
        path = _validate_relative_path(
            expected_pin.path,
            context=f"expected gitlink {index} path",
        )
        alias = _portable_path_key(path)
        if alias in expected_aliases:
            raise ExternalMaterializationError("expected gitlinks contain duplicate aliases")
        if previous_expected_path is not None and path <= previous_expected_path:
            raise ExternalMaterializationError("expected gitlinks are not strictly path-sorted")
        _require_git_sha1(
            expected_pin.commit_git_sha1,
            context=f"expected gitlink {path} commit",
        )
        expected_aliases.add(alias)
        expected_by_path[path] = expected_pin
        previous_expected_path = path
    records: list[tuple[str, str, str]] = []
    seen_gitlinks: set[str] = set()
    previous_path: str | None = None
    for index, record in enumerate(raw[:-1].split(b"\0")):
        try:
            metadata, path_raw = record.split(b"\t", 1)
            mode_raw, kind_raw, object_raw = metadata.split(b" ")
            path = path_raw.decode("utf-8", "strict")
            mode = mode_raw.decode("ascii", "strict")
            kind = kind_raw.decode("ascii", "strict")
            object_id = object_raw.decode("ascii", "strict")
        except (UnicodeDecodeError, ValueError) as exc:
            raise ExternalMaterializationError(f"Git tree entry {index} is malformed") from exc
        path = _validate_relative_path(path, context=f"Git tree entry {index} path")
        if previous_path is not None and path <= previous_path:
            raise ExternalMaterializationError("Git tree paths are not strictly sorted")
        previous_path = path
        if mode == "160000":
            if kind != "commit":
                raise ExternalMaterializationError(f"Git tree gitlink is malformed: {path}")
            _require_git_sha1(object_id, context=f"Git tree gitlink {path}")
            matched_pin = expected_by_path.get(path)
            if matched_pin is None:
                raise ExternalMaterializationError(
                    f"Git tree contains an unexpected gitlink: {path}"
                )
            if not hmac.compare_digest(object_id, matched_pin.commit_git_sha1):
                raise ExternalMaterializationError(
                    f"Git tree gitlink commit does not match the identity: {path}"
                )
            seen_gitlinks.add(path)
        elif mode in {"100644", "100755"} and kind == "blob":
            _require_git_sha1(object_id, context=f"Git tree blob {path}")
        else:
            label = "symlink" if mode == "120000" else "non-regular entry"
            raise ExternalMaterializationError(f"Git tree contains a {label}: {path}")
        records.append((path, mode, object_id))
        if len(records) > _MAX_TRACKED_FILES:
            raise ExternalMaterializationError("Git tree exceeds the tracked-file limit")
    missing_gitlinks = set(expected_by_path) - seen_gitlinks
    if missing_gitlinks:
        raise ExternalMaterializationError(
            f"Git tree lacks expected gitlinks: {sorted(missing_gitlinks)}"
        )
    _validate_observed_portable_aliases(
        records,
        portable_path_aliases,
        context="Git tree",
    )
    return tuple(records)


def _git_tree_sha1(
    files: Sequence[tuple[str, str, str]],
    portable_path_aliases: tuple[PortablePathAliasPin, ...] = (),
) -> str:
    """Reconstruct a Git SHA-1 root tree from a complete flat tracked inventory."""
    _validate_observed_portable_aliases(
        files,
        portable_path_aliases,
        context="Git tree reconstruction",
    )
    root = _TreeDirectory(children={})
    for path, mode, blob_git_sha1 in files:
        _validate_relative_path(path, context="Git tree reconstruction path")
        if mode not in {"100644", "100755", "160000"}:
            raise ExternalMaterializationError("Git tree reconstruction mode is invalid")
        _require_git_sha1(blob_git_sha1, context="Git tree reconstruction object")
        parts = PurePosixPath(path).parts
        directory = root
        for part in parts[:-1]:
            existing = directory.children.get(part)
            if existing is None:
                child = _TreeDirectory(children={})
                directory.children[part] = child
                directory = child
            elif isinstance(existing, _TreeDirectory):
                directory = existing
            else:
                raise ExternalMaterializationError(
                    "Git tree reconstruction contains a file/directory conflict"
                )
        filename = parts[-1]
        if filename in directory.children:
            raise ExternalMaterializationError("Git tree reconstruction contains a duplicate path")
        directory.children[filename] = _TreeBlob(mode, blob_git_sha1)

    def hash_directory(directory: _TreeDirectory) -> str:
        entries: list[tuple[bytes, bytes]] = []
        for name, item in directory.children.items():
            name_raw = name.encode("utf-8")
            if isinstance(item, _TreeDirectory):
                mode_raw = b"40000"
                object_id = hash_directory(item)
                sort_key = name_raw + b"/"
            else:
                mode_raw = item.git_mode.encode("ascii")
                object_id = item.blob_git_sha1
                sort_key = name_raw + b"\0"
            serialized = mode_raw + b" " + name_raw + b"\0" + bytes.fromhex(object_id)
            entries.append((sort_key, serialized))
        payload = b"".join(serialized for _, serialized in sorted(entries))
        header = f"tree {len(payload)}\0".encode("ascii")
        return hashlib.sha1(header + payload).hexdigest()

    return hash_directory(root)


def _open_relative_directory(root_descriptor: int, parts: Sequence[str]) -> int:
    descriptor = os.dup(root_descriptor)
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        for part in parts:
            next_descriptor = os.open(part, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _lstat_relative(root_descriptor: int, parts: Sequence[str]) -> os.stat_result | None:
    if not parts:
        return os.fstat(root_descriptor)
    parent_descriptor = _open_relative_directory(root_descriptor, parts[:-1])
    try:
        try:
            return os.stat(parts[-1], dir_fd=parent_descriptor, follow_symlinks=False)
        except FileNotFoundError:
            return None
    finally:
        os.close(parent_descriptor)


def _read_relative_bytes(
    root_descriptor: int,
    parts: Sequence[str],
    *,
    maximum_bytes: int,
) -> bytes | None:
    item_stat = _lstat_relative(root_descriptor, parts)
    if item_stat is None:
        return None
    if not stat.S_ISREG(item_stat.st_mode) or stat.S_ISLNK(item_stat.st_mode):
        raise ExternalMaterializationError("Git metadata entry is not a regular file")
    if item_stat.st_nlink != 1:
        raise ExternalMaterializationError("Git metadata entry is hardlinked")
    if item_stat.st_size > maximum_bytes:
        raise ExternalMaterializationError("Git metadata entry exceeds its byte limit")
    parent_descriptor = _open_relative_directory(root_descriptor, parts[:-1])
    try:
        descriptor = os.open(
            parts[-1],
            os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_descriptor,
        )
    finally:
        os.close(parent_descriptor)
    chunks: list[bytes] = []
    observed = 0
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or (before.st_dev, before.st_ino) != (item_stat.st_dev, item_stat.st_ino)
        ):
            raise ExternalMaterializationError("Git metadata entry identity changed")
        while True:
            chunk = os.read(descriptor, min(_COPY_CHUNK_BYTES, maximum_bytes + 1 - observed))
            if not chunk:
                break
            chunks.append(chunk)
            observed += len(chunk)
            if observed > maximum_bytes:
                raise ExternalMaterializationError("Git metadata entry exceeds its byte limit")
        after = os.fstat(descriptor)
        if (
            observed != before.st_size
            or after.st_size != before.st_size
            or after.st_mtime_ns != before.st_mtime_ns
            or after.st_ctime_ns != before.st_ctime_ns
        ):
            raise ExternalMaterializationError("Git metadata changed during inspection")
    finally:
        os.close(descriptor)
    return b"".join(chunks)


def _reject_git_metadata_path(
    git_descriptor: int,
    relative: tuple[str, ...],
    *,
    context: str,
) -> None:
    try:
        observed = _lstat_relative(git_descriptor, relative)
    except OSError as exc:
        raise ExternalMaterializationError(f"unsafe Git metadata path: {context}") from exc
    if observed is not None:
        raise ExternalMaterializationError(f"unsupported Git metadata is present: {context}")


def _validate_git_metadata(
    checkout: _CheckoutAnchor,
    git_environment: _HermeticGitEnvironment,
) -> None:
    for relative, context in (
        (("commondir",), "linked-worktree common directory"),
        (("gitdir",), "linked-worktree back pointer"),
        (("worktrees",), "linked-worktree metadata"),
        (("refs", "replace"), "replace refs"),
        (("objects", "info", "alternates"), "object alternates"),
        (("objects", "info", "http-alternates"), "HTTP object alternates"),
        (("info", "grafts"), "legacy grafts"),
        (("shallow",), "shallow repository marker"),
        (("modules",), "submodule metadata"),
        (("config.worktree",), "worktree-specific config"),
    ):
        _reject_git_metadata_path(checkout.git.descriptor, relative, context=context)

    direct_config = _read_relative_bytes(
        checkout.git.descriptor,
        ("config",),
        maximum_bytes=_MAX_GIT_CONFIG_OUTPUT_BYTES,
    )
    direct_head = _read_relative_bytes(
        checkout.git.descriptor,
        ("HEAD",),
        maximum_bytes=_MAX_GIT_IDENTITY_OUTPUT_BYTES,
    )
    if direct_config is None or direct_head is None:
        raise ExternalMaterializationError("checkout lacks direct regular Git config or HEAD")

    packed_refs = _read_relative_bytes(
        checkout.git.descriptor,
        ("packed-refs",),
        maximum_bytes=64 * 1024 * 1024,
    )
    if packed_refs is not None and b" refs/replace/" in packed_refs:
        raise ExternalMaterializationError("packed replace refs are unsupported")

    for directory_parts, forbidden_suffix, context in (
        (("objects", "pack"), ".promisor", "promisor pack metadata"),
        (("hooks",), None, "active Git hook"),
    ):
        directory_stat = _lstat_relative(checkout.git.descriptor, directory_parts)
        if directory_stat is None:
            continue
        if not stat.S_ISDIR(directory_stat.st_mode) or stat.S_ISLNK(directory_stat.st_mode):
            raise ExternalMaterializationError(f"unsafe Git metadata directory: {context}")
        descriptor = _open_relative_directory(checkout.git.descriptor, directory_parts)
        try:
            for name in os.listdir(descriptor):
                item_stat = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                if (
                    not stat.S_ISREG(item_stat.st_mode)
                    or stat.S_ISLNK(item_stat.st_mode)
                    or item_stat.st_nlink != 1
                ):
                    raise ExternalMaterializationError(
                        f"unsafe non-regular or hardlinked Git metadata: {context}"
                    )
                if forbidden_suffix is not None:
                    if name.endswith(forbidden_suffix):
                        raise ExternalMaterializationError(
                            f"unsupported Git metadata is present: {context}"
                        )
                    continue
                if not name.endswith(".sample"):
                    raise ExternalMaterializationError(
                        f"unsupported Git metadata is present: {context}"
                    )
        finally:
            os.close(descriptor)

    raw_config = _git(
        checkout,
        git_environment,
        ["config", "--local", "--no-includes", "--null", "--list"],
        maximum_stdout_bytes=_MAX_GIT_CONFIG_OUTPUT_BYTES,
    )
    allowed_exact = {
        "core.repositoryformatversion",
        "core.filemode",
        "core.bare",
        "core.logallrefupdates",
        "core.ignorecase",
        "core.precomposeunicode",
        "gc.auto",
        "user.name",
        "user.email",
    }
    for record in raw_config.rstrip(b"\0").split(b"\0") if raw_config else ():
        key_raw, separator, value_raw = record.partition(b"\n")
        if not separator:
            raise ExternalMaterializationError("repository config output is malformed")
        try:
            key = key_raw.decode("utf-8", "strict").casefold()
            value = value_raw.decode("utf-8", "strict")
        except UnicodeDecodeError as exc:
            raise ExternalMaterializationError("repository config is not UTF-8") from exc
        key_parts = key.split(".")
        safe_remote_url = (
            len(key_parts) == 3
            and key_parts[0] == "remote"
            and key_parts[2] == "url"
            and value.startswith("https://")
        )
        safe_remote_fetch = (
            len(key_parts) == 3
            and key_parts[0] == "remote"
            and key_parts[2] == "fetch"
            and value.startswith(("+refs/", "refs/"))
            and "\n" not in value
        )
        safe_branch = (
            len(key_parts) == 3
            and key_parts[0] == "branch"
            and key_parts[2] in {"remote", "merge"}
            and "\n" not in value
        )
        if key not in allowed_exact and not (safe_remote_url or safe_remote_fetch or safe_branch):
            raise ExternalMaterializationError(
                f"repository config is outside the inert inspection allowlist: {key}"
            )
        if key == "core.repositoryformatversion" and value != "0":
            raise ExternalMaterializationError("extended repository formats are unsupported")
        if key == "core.bare" and value.casefold() not in {"false", "no", "off", "0"}:
            raise ExternalMaterializationError("bare repositories are unsupported")


def _git_reported_directory_identity(raw_path: bytes, *, context: str) -> tuple[int, int]:
    try:
        text = raw_path.decode("utf-8", "strict")
    except UnicodeDecodeError as exc:
        raise ExternalMaterializationError(f"Git-reported {context} is not UTF-8") from exc
    if not text or "\0" in text or "\n" in text or not os.path.isabs(text):
        raise ExternalMaterializationError(f"Git-reported {context} is not one absolute path")
    descriptor = -1
    try:
        descriptor = os.open(
            text,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        item_stat = os.fstat(descriptor)
    except OSError as exc:
        raise ExternalMaterializationError(
            f"Git-reported {context} is not an accessible directory"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if not stat.S_ISDIR(item_stat.st_mode):
        raise ExternalMaterializationError(f"Git-reported {context} is not a directory")
    return item_stat.st_dev, item_stat.st_ino


def _validate_git_layout_binding(
    checkout: _CheckoutAnchor,
    git_environment: _HermeticGitEnvironment,
) -> None:
    raw = _git(
        checkout,
        git_environment,
        [
            "rev-parse",
            "--path-format=absolute",
            "--git-dir",
            "--git-common-dir",
            "--show-toplevel",
        ],
        maximum_stdout_bytes=_MAX_GIT_IDENTITY_OUTPUT_BYTES,
    )
    if not raw.endswith(b"\n"):
        raise ExternalMaterializationError("Git layout report is malformed")
    records = raw[:-1].split(b"\n")
    if len(records) != 3:
        raise ExternalMaterializationError("Git layout report does not contain three paths")
    observed = tuple(
        _git_reported_directory_identity(record, context=context)
        for record, context in zip(
            records,
            ("Git directory", "common Git directory", "worktree top level"),
            strict=True,
        )
    )
    expected_git = (checkout.git.device, checkout.git.inode)
    expected_root = (checkout.root.device, checkout.root.inode)
    if observed != (expected_git, expected_git, expected_root):
        raise ExternalMaterializationError(
            "Git directory, common directory, or top-level binding escaped its anchor"
        )


def _worktree_inventory(
    checkout: _CheckoutAnchor,
    expected_files: set[str],
    expected_gitlinks: tuple[GitlinkPin, ...],
) -> None:
    gitlink_paths = {item.path for item in expected_gitlinks}
    if (".gitmodules" in expected_files) is not bool(gitlink_paths):
        raise ExternalMaterializationError(
            "tracked .gitmodules is permitted exactly when gitlinks are explicitly expected"
        )
    expected_directories = {
        ancestor.as_posix()
        for path in expected_files
        for ancestor in PurePosixPath(path).parents
        if ancestor.as_posix() != "."
    }
    observed_files: set[str] = set()
    observed_directories: set[str] = set()
    observed_placeholders: set[str] = set()

    def validate_empty_gitlink_placeholder(
        parent_descriptor: int,
        name: str,
        path: str,
        name_stat: os.stat_result,
    ) -> None:
        if not stat.S_ISDIR(name_stat.st_mode) or stat.S_ISLNK(name_stat.st_mode):
            raise ExternalMaterializationError(
                f"gitlink placeholder is not an empty real directory: {path}"
            )
        descriptor = -1
        try:
            descriptor = os.open(
                name,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=parent_descriptor,
            )
            before = os.fstat(descriptor)
            if (
                not stat.S_ISDIR(before.st_mode)
                or before.st_nlink != 2
                or (before.st_dev, before.st_ino) != (name_stat.st_dev, name_stat.st_ino)
            ):
                raise ExternalMaterializationError(
                    f"gitlink placeholder has ambiguous identity: {path}"
                )
            if os.listdir(descriptor):
                raise ExternalMaterializationError(
                    f"gitlink placeholder is not exactly empty: {path}"
                )
            after = os.fstat(descriptor)
            current_name_stat = os.stat(
                name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            stable_before = (
                before.st_dev,
                before.st_ino,
                before.st_mode,
                before.st_nlink,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            )
            stable_after = (
                after.st_dev,
                after.st_ino,
                after.st_mode,
                after.st_nlink,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            )
            stable_name = (
                current_name_stat.st_dev,
                current_name_stat.st_ino,
                current_name_stat.st_mode,
                current_name_stat.st_nlink,
                current_name_stat.st_size,
                current_name_stat.st_mtime_ns,
                current_name_stat.st_ctime_ns,
            )
            if stable_before != stable_after or stable_after != stable_name:
                raise ExternalMaterializationError(
                    f"gitlink placeholder changed during inspection: {path}"
                )
        except OSError as exc:
            raise ExternalMaterializationError(
                f"gitlink placeholder is inaccessible or unstable: {path}"
            ) from exc
        finally:
            if descriptor >= 0:
                os.close(descriptor)

    def walk(directory_descriptor: int, prefix: tuple[str, ...]) -> None:
        with os.scandir(directory_descriptor) as entries:
            for entry in entries:
                name = entry.name
                if not prefix and name == ".git":
                    continue
                path = "/".join((*prefix, name))
                _validate_relative_path(path, context="worktree path")
                item_stat = entry.stat(follow_symlinks=False)
                if path in gitlink_paths:
                    validate_empty_gitlink_placeholder(
                        directory_descriptor,
                        name,
                        path,
                        item_stat,
                    )
                    observed_directories.add(path)
                    observed_placeholders.add(path)
                    continue
                if stat.S_ISDIR(item_stat.st_mode) and not stat.S_ISLNK(item_stat.st_mode):
                    observed_directories.add(path)
                    child = os.open(
                        name,
                        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
                        dir_fd=directory_descriptor,
                    )
                    try:
                        child_stat = os.fstat(child)
                        if (child_stat.st_dev, child_stat.st_ino) != (
                            item_stat.st_dev,
                            item_stat.st_ino,
                        ):
                            raise ExternalMaterializationError(
                                "worktree directory changed during enumeration"
                            )
                        walk(child, (*prefix, name))
                    finally:
                        os.close(child)
                elif stat.S_ISREG(item_stat.st_mode) and not stat.S_ISLNK(item_stat.st_mode):
                    if item_stat.st_nlink != 1:
                        raise ExternalMaterializationError(f"tracked source is hardlinked: {path}")
                    observed_files.add(path)
                else:
                    raise ExternalMaterializationError(
                        f"worktree contains a non-regular entry: {path}"
                    )

    walk(checkout.root.descriptor, ())
    allowed_directories = expected_directories | observed_placeholders
    for path in observed_placeholders:
        allowed_directories.update(
            ancestor.as_posix()
            for ancestor in PurePosixPath(path).parents
            if ancestor.as_posix() != "."
        )
    if observed_files != expected_files or observed_directories != allowed_directories:
        raise ExternalMaterializationError(
            "checkout contains dirty, untracked, ignored, or missing content"
        )


def _read_exact_file(
    root_descriptor: int,
    path: str,
    mode: str,
    blob_git_sha1: str,
    *,
    capture: bool,
) -> tuple[int, str, bytes | None, tuple[int, int]]:
    relative = PurePosixPath(path).parts
    parent_descriptor = _open_relative_directory(root_descriptor, relative[:-1])
    descriptor = -1
    try:
        name_before = os.stat(
            relative[-1],
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        descriptor = os.open(
            relative[-1],
            os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_descriptor,
        )
    except OSError as exc:
        os.close(parent_descriptor)
        raise ExternalMaterializationError(f"tracked file is inaccessible: {path}") from exc
    captured = bytearray() if capture else None
    sha256 = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_ISLNK(name_before.st_mode)
            or (before.st_dev, before.st_ino) != (name_before.st_dev, name_before.st_ino)
        ):
            raise ExternalMaterializationError(f"tracked entry is not a regular file: {path}")
        if before.st_nlink != 1:
            raise ExternalMaterializationError(f"tracked source is hardlinked: {path}")
        executable = bool(stat.S_IMODE(before.st_mode) & 0o111)
        if executable != (mode == "100755"):
            raise ExternalMaterializationError(f"tracked executable bit drifted: {path}")
        blob_sha1 = hashlib.sha1(f"blob {before.st_size}\0".encode("ascii"))
        observed_size = 0
        while True:
            chunk = os.read(descriptor, _COPY_CHUNK_BYTES)
            if not chunk:
                break
            observed_size += len(chunk)
            blob_sha1.update(chunk)
            sha256.update(chunk)
            if captured is not None:
                captured.extend(chunk)
        after = os.fstat(descriptor)
        name_after = os.stat(
            relative[-1],
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (
            observed_size != before.st_size
            or after.st_size != before.st_size
            or after.st_mtime_ns != before.st_mtime_ns
            or after.st_ctime_ns != before.st_ctime_ns
            or (after.st_dev, after.st_ino) != (before.st_dev, before.st_ino)
            or (name_after.st_dev, name_after.st_ino) != (before.st_dev, before.st_ino)
            or name_after.st_mode != before.st_mode
            or name_after.st_nlink != before.st_nlink
        ):
            raise ExternalMaterializationError(f"tracked file changed during inspection: {path}")
    finally:
        os.close(descriptor)
        os.close(parent_descriptor)
    if not hmac.compare_digest(blob_sha1.hexdigest(), blob_git_sha1):
        raise ExternalMaterializationError(f"tracked worktree bytes differ from Git blob: {path}")
    return (
        observed_size,
        sha256.hexdigest(),
        bytes(captured) if captured is not None else None,
        (before.st_dev, before.st_ino),
    )


def _inspect_checkout(
    checkout: _CheckoutAnchor,
    identity: ExternalCheckoutIdentity,
    git_environment: _HermeticGitEnvironment,
) -> tuple[tuple[_GitTreeFile, ...], dict[str, bytes]]:
    _recheck_checkout_anchor(checkout)
    _validate_git_metadata(checkout, git_environment)
    _validate_git_layout_binding(checkout, git_environment)
    commit = (
        _git(
            checkout,
            git_environment,
            ["rev-parse", "--verify", "HEAD^{commit}"],
        )
        .strip()
        .decode("ascii")
    )
    tree = (
        _git(
            checkout,
            git_environment,
            ["rev-parse", "--verify", "HEAD^{tree}"],
        )
        .strip()
        .decode("ascii")
    )
    if not hmac.compare_digest(commit, identity.commit_git_sha1):
        raise ExternalMaterializationError("checkout commit does not match the identity")
    if not hmac.compare_digest(tree, identity.tree_git_sha1):
        raise ExternalMaterializationError("checkout tree does not match the identity")
    tree_entries = _parse_tree(
        _git(
            checkout,
            git_environment,
            ["ls-tree", "-r", "-z", "--full-tree", "HEAD"],
            maximum_stdout_bytes=_MAX_GIT_TREE_OUTPUT_BYTES,
        ),
        identity.excluded_gitlinks,
        identity.portable_path_aliases,
    )
    if not hmac.compare_digest(
        _git_tree_sha1(tree_entries, identity.portable_path_aliases),
        identity.tree_git_sha1,
    ):
        raise ExternalMaterializationError("enumerated Git tree does not match the identity")
    transform_paths = {item.path for item in identity.source_transforms}
    regular_tree_entries = tuple(item for item in tree_entries if item[1] in {"100644", "100755"})
    observed_paths = {path for path, _, _ in regular_tree_entries}
    if not transform_paths <= observed_paths:
        missing = sorted(transform_paths - observed_paths)
        raise ExternalMaterializationError(f"checkout lacks transformed sources: {missing}")
    _worktree_inventory(checkout, observed_paths, identity.excluded_gitlinks)

    files: list[_GitTreeFile] = []
    captured: dict[str, bytes] = {}
    source_identities: dict[str, tuple[int, int]] = {}
    for path, mode, blob_git_sha1 in regular_tree_entries:
        size, sha256, raw, source_identity = _read_exact_file(
            checkout.root.descriptor,
            path,
            mode,
            blob_git_sha1,
            capture=path in transform_paths,
        )
        files.append(
            _GitTreeFile(
                path,
                mode,
                blob_git_sha1,
                size,
                sha256,
                source_identity[0],
                source_identity[1],
            )
        )
        source_identities[path] = source_identity
        if raw is not None:
            captured[path] = raw
    _verify_distinct_portable_alias_identities(
        source_identities,
        identity.portable_path_aliases,
        context="checkout",
    )

    pins = {item.path: item for item in identity.source_transforms}
    for path in sorted(transform_paths):
        record = files[[item.path for item in files].index(path)]
        pin = pins[path]
        if record.upstream_size_bytes != pin.upstream_size_bytes:
            raise ExternalMaterializationError(f"audited source size drifted: {path}")
        if not hmac.compare_digest(record.upstream_sha256, pin.upstream_sha256):
            raise ExternalMaterializationError(f"audited source SHA-256 drifted: {path}")
    _validate_git_layout_binding(checkout, git_environment)
    _validate_git_metadata(checkout, git_environment)
    _recheck_checkout_anchor(checkout)
    return tuple(files), captured


def _production_derive(sources: dict[str, bytes]) -> _DerivedSourceSet:
    expected_identity = pinned_external_checkout_identity()
    if not hmac.compare_digest(
        _sha256(_PINNED_TRANSPORT_DESCRIPTOR_BYTES),
        expected_identity.transport_descriptor_sha256,
    ):
        raise AssertionError("pinned seed-transport descriptor drifted")
    try:
        derived = seed_transport.replay_matched_v3_external_seed_transport(
            sources,
            _PINNED_TRANSPORT_DESCRIPTOR_BYTES,
        )
    except seed_transport.ExternalSeedTransportError as exc:
        raise ExternalMaterializationError("seed-transport derivation failed") from exc
    return _DerivedSourceSet(
        sources=MappingProxyType(dict(derived.sources)),
        transport_schema_version=seed_transport.SCHEMA_VERSION,
        transport_descriptor_sha256=derived.descriptor_sha256,
    )


def _output_parent_descriptor(root_descriptor: int, path: str) -> tuple[int, str]:
    parts = PurePosixPath(path).parts
    descriptor = os.dup(root_descriptor)
    try:
        for part in parts[:-1]:
            try:
                os.mkdir(part, 0o700, dir_fd=descriptor)
            except FileExistsError:
                pass
            next_descriptor = os.open(
                part,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor, parts[-1]
    except BaseException:
        os.close(descriptor)
        raise


def _write_exact_bytes_at(root_descriptor: int, path: str, raw: bytes, mode: int) -> None:
    parent_descriptor, filename = _output_parent_descriptor(root_descriptor, path)
    try:
        descriptor = os.open(
            filename,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            mode,
            dir_fd=parent_descriptor,
        )
    finally:
        os.close(parent_descriptor)
    try:
        _write_all(descriptor, raw)
        os.fchmod(descriptor, mode)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_all(descriptor: int, raw: bytes) -> None:
    offset = 0
    while offset < len(raw):
        written = os.write(descriptor, raw[offset:])
        if written <= 0:
            raise ExternalMaterializationError("file write made no progress")
        offset += written


def _copy_exact_file_at(
    source_root_descriptor: int,
    destination_root_descriptor: int,
    record: _GitTreeFile,
) -> str:
    source_parts = PurePosixPath(record.path).parts
    source_parent = _open_relative_directory(
        source_root_descriptor,
        source_parts[:-1],
    )
    try:
        source_descriptor = os.open(
            source_parts[-1],
            os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=source_parent,
        )
    finally:
        os.close(source_parent)
    mode = 0o755 if record.git_mode == "100755" else 0o644
    destination_descriptor = -1
    try:
        destination_parent, destination_name = _output_parent_descriptor(
            destination_root_descriptor,
            record.path,
        )
        try:
            destination_descriptor = os.open(
                destination_name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                mode,
                dir_fd=destination_parent,
            )
        finally:
            os.close(destination_parent)
        sha256 = hashlib.sha256()
        blob_sha1 = hashlib.sha1(f"blob {record.upstream_size_bytes}\0".encode("ascii"))
        observed_size = 0
        before = os.fstat(source_descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ExternalMaterializationError(f"tracked entry changed type: {record.path}")
        if before.st_nlink != 1:
            raise ExternalMaterializationError(f"tracked source is hardlinked: {record.path}")
        if (
            record.source_device is not None
            and record.source_inode is not None
            and (before.st_dev, before.st_ino) != (record.source_device, record.source_inode)
        ):
            raise ExternalMaterializationError(
                f"tracked source inode changed before materialization: {record.path}"
            )
        while True:
            chunk = os.read(source_descriptor, _COPY_CHUNK_BYTES)
            if not chunk:
                break
            observed_size += len(chunk)
            blob_sha1.update(chunk)
            sha256.update(chunk)
            _write_all(destination_descriptor, chunk)
        after = os.fstat(source_descriptor)
        if (
            observed_size != record.upstream_size_bytes
            or after.st_size != before.st_size
            or after.st_mtime_ns != before.st_mtime_ns
            or after.st_ctime_ns != before.st_ctime_ns
        ):
            raise ExternalMaterializationError(
                f"tracked file changed during materialization: {record.path}"
            )
        if not hmac.compare_digest(blob_sha1.hexdigest(), record.blob_git_sha1):
            raise ExternalMaterializationError(
                f"tracked file no longer matches its Git blob: {record.path}"
            )
        os.fchmod(destination_descriptor, mode)
        os.fsync(destination_descriptor)
    finally:
        os.close(source_descriptor)
        if destination_descriptor >= 0:
            os.close(destination_descriptor)
    digest = sha256.hexdigest()
    if not hmac.compare_digest(digest, record.upstream_sha256):
        raise ExternalMaterializationError(
            f"tracked file no longer matches its inspected digest: {record.path}"
        )
    return digest


def _manifest(
    identity: ExternalCheckoutIdentity,
    files: tuple[_GitTreeFile, ...],
    derived: _DerivedSourceSet,
) -> dict[str, Any]:
    transforms = {item.path: item for item in identity.source_transforms}
    file_records: list[dict[str, Any]] = []
    upstream_total = 0
    materialized_total = 0
    for item in files:
        pin = transforms.get(item.path)
        materialized_size = pin.derived_size_bytes if pin is not None else item.upstream_size_bytes
        materialized_sha256 = pin.derived_sha256 if pin is not None else item.upstream_sha256
        upstream_total += item.upstream_size_bytes
        materialized_total += materialized_size
        file_records.append(
            {
                "path": item.path,
                "git_mode": item.git_mode,
                "upstream_blob_git_sha1": item.blob_git_sha1,
                "upstream_size_bytes": item.upstream_size_bytes,
                "upstream_sha256": item.upstream_sha256,
                "materialized_size_bytes": materialized_size,
                "materialized_sha256": materialized_sha256,
                "transformed": pin is not None,
            }
        )
    identity_payload = _identity_payload(identity)
    excluded_gitlink_records = [
        {
            "path": item.path,
            "git_mode": "160000",
            "commit_git_sha1": item.commit_git_sha1,
            "content_materialized": False,
        }
        for item in identity.excluded_gitlinks
    ]
    portable_alias_records = [
        {
            "path": item.path,
            "upstream_blob_git_sha1": item.blob_git_sha1,
            "materialized_as_distinct_path": True,
        }
        for item in identity.portable_path_aliases
    ]
    payload: dict[str, Any] = {
        "schema_version": EXTERNAL_MATERIALIZATION_SCHEMA_VERSION,
        "status": (
            "materialized_tracked_regular_source_closure_with_excluded_gitlinks_unqualified"
        ),
        "classification": "nonpromoting_external_derived_checkout",
        "identity": identity_payload,
        "identity_sha256": _sha256(_canonical_json(identity_payload)),
        "checkout_attestation": {
            "commit_verified": True,
            "tree_verified": True,
            "clean_worktree_verified": True,
            "every_tracked_regular_blob_verified": True,
            "exact_gitlinks_verified": True,
            "gitlink_placeholders_absent_or_empty_verified": True,
            "gitlink_content_initialized": False,
            "gitlink_content_imported": False,
            "gitlink_content_copied": False,
            "gitlink_content_executed": False,
            "portable_path_aliases_verified": True,
            "portable_path_aliases_have_distinct_inodes": True,
            "archive_bytes_verified": False,
            "archive_identity_binding_only": True,
        },
        "derivation": {
            "transport_schema_version": derived.transport_schema_version,
            "transport_descriptor_sha256": derived.transport_descriptor_sha256,
            "transformed_source_count": len(transforms),
        },
        "source_tree": {
            "scope": (
                "complete_tracked_tree_with_materialized_regular_files_and_excluded_gitlinks"
            ),
            "source_closure_bound": True,
            "git_metadata_included": False,
            "symlinks_included": False,
            "gitlink_content_included": False,
            "normalized_directory_mode": "0755",
            "tracked_entry_count": len(file_records) + len(excluded_gitlink_records),
            "materialized_regular_file_count": len(file_records),
            "excluded_gitlink_count": len(excluded_gitlink_records),
            "portable_path_alias_count": len(portable_alias_records),
            "upstream_total_size_bytes": upstream_total,
            "materialized_total_size_bytes": materialized_total,
            "excluded_gitlinks": excluded_gitlink_records,
            "portable_path_aliases": portable_alias_records,
            "files": file_records,
        },
        "claims": {
            "runtime_import_qualified": False,
            "runtime_execution_qualified": False,
            "execution_ready": False,
            "execution_authorized": False,
            "result_ingestion_authorized": False,
            "scientific_promotion_allowed": False,
            "performance_claim_allowed": False,
            "universal_sota_claim_allowed": False,
            "filesystem_capability_granted": False,
            "authority_granted": False,
        },
        "limitations": _frozen_manifest_limitations(),
    }
    payload["payload_sha256"] = _sha256(_canonical_json(payload))
    return payload


def _reject_duplicate_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ExternalMaterializationError(f"manifest contains duplicate key: {key}")
        result[key] = value
    return result


def parse_external_materialization_manifest(
    raw: bytes,
    *,
    expected_manifest_sha256: str,
) -> dict[str, Any]:
    """Parse only exact canonical, digest-bound, non-authorizing manifest bytes."""
    if type(raw) is not bytes:
        raise ExternalMaterializationError("manifest must be exact bytes")
    if not 0 < len(raw) <= _MAX_MANIFEST_BYTES:
        raise ExternalMaterializationError("manifest byte length is invalid")
    expected_digest = _require_sha256(expected_manifest_sha256, context="expected manifest digest")
    if not hmac.compare_digest(_sha256(raw), expected_digest):
        raise ExternalMaterializationError("manifest digest does not match")
    try:
        value = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_reject_duplicate_object_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ExternalMaterializationError(f"manifest contains {token}")
            ),
        )
    except (RecursionError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExternalMaterializationError("manifest is not strict ASCII JSON") from exc
    if type(value) is not dict:
        raise ExternalMaterializationError("manifest root must be a plain object")
    manifest = cast(dict[str, Any], value)
    if raw != _canonical_json(manifest):
        raise ExternalMaterializationError("manifest is not exact canonical JSON")
    expected_root_keys = {
        "schema_version",
        "status",
        "classification",
        "identity",
        "identity_sha256",
        "checkout_attestation",
        "derivation",
        "source_tree",
        "claims",
        "limitations",
        "payload_sha256",
    }
    if set(manifest) != expected_root_keys:
        raise ExternalMaterializationError("manifest root fields do not match")
    if manifest["schema_version"] != EXTERNAL_MATERIALIZATION_SCHEMA_VERSION:
        raise ExternalMaterializationError("manifest schema version does not match")
    if manifest["status"] != (
        "materialized_tracked_regular_source_closure_with_excluded_gitlinks_unqualified"
    ):
        raise ExternalMaterializationError("manifest status does not match")
    if manifest["classification"] != "nonpromoting_external_derived_checkout":
        raise ExternalMaterializationError("manifest classification does not match")
    payload_sha256 = manifest["payload_sha256"]
    _require_sha256(payload_sha256, context="manifest payload digest")
    without_payload = dict(manifest)
    without_payload.pop("payload_sha256")
    if not hmac.compare_digest(_sha256(_canonical_json(without_payload)), payload_sha256):
        raise ExternalMaterializationError("manifest payload digest does not match")

    identity = _identity_from_payload(manifest["identity"])
    identity_digest = _sha256(_canonical_json(_identity_payload(identity)))
    manifest_identity_digest = _require_sha256(
        manifest["identity_sha256"], context="manifest identity digest"
    )
    if not hmac.compare_digest(identity_digest, manifest_identity_digest):
        raise ExternalMaterializationError("manifest identity digest does not match")
    attestation = manifest["checkout_attestation"]
    expected_attestation = {
        "commit_verified": True,
        "tree_verified": True,
        "clean_worktree_verified": True,
        "every_tracked_regular_blob_verified": True,
        "exact_gitlinks_verified": True,
        "gitlink_placeholders_absent_or_empty_verified": True,
        "gitlink_content_initialized": False,
        "gitlink_content_imported": False,
        "gitlink_content_copied": False,
        "gitlink_content_executed": False,
        "portable_path_aliases_verified": True,
        "portable_path_aliases_have_distinct_inodes": True,
        "archive_bytes_verified": False,
        "archive_identity_binding_only": True,
    }
    if (
        type(attestation) is not dict
        or set(attestation) != set(expected_attestation)
        or any(attestation[key] is not expected for key, expected in expected_attestation.items())
    ):
        raise ExternalMaterializationError("manifest checkout attestation does not match")
    derivation = manifest["derivation"]
    expected_derivation = {
        "transport_schema_version": identity.transport_schema_version,
        "transport_descriptor_sha256": identity.transport_descriptor_sha256,
        "transformed_source_count": len(identity.source_transforms),
    }
    if (
        type(derivation) is not dict
        or set(derivation) != set(expected_derivation)
        or type(derivation.get("transport_schema_version")) is not str
        or type(derivation.get("transport_descriptor_sha256")) is not str
        or type(derivation.get("transformed_source_count")) is not int
        or derivation != expected_derivation
    ):
        raise ExternalMaterializationError("manifest derivation binding does not match")

    claims = manifest["claims"]
    expected_claim_keys = {
        "runtime_import_qualified",
        "runtime_execution_qualified",
        "execution_ready",
        "execution_authorized",
        "result_ingestion_authorized",
        "scientific_promotion_allowed",
        "performance_claim_allowed",
        "universal_sota_claim_allowed",
        "filesystem_capability_granted",
        "authority_granted",
    }
    if (
        type(claims) is not dict
        or set(claims) != expected_claim_keys
        or any(value is not False for value in claims.values())
    ):
        raise ExternalMaterializationError("manifest authority denial does not match")
    limitations = manifest["limitations"]
    if type(limitations) is not list or limitations != _frozen_manifest_limitations():
        raise ExternalMaterializationError("manifest limitations do not match")

    source_tree = manifest["source_tree"]
    if type(source_tree) is not dict or set(source_tree) != {
        "scope",
        "source_closure_bound",
        "git_metadata_included",
        "symlinks_included",
        "gitlink_content_included",
        "normalized_directory_mode",
        "tracked_entry_count",
        "materialized_regular_file_count",
        "excluded_gitlink_count",
        "portable_path_alias_count",
        "upstream_total_size_bytes",
        "materialized_total_size_bytes",
        "excluded_gitlinks",
        "portable_path_aliases",
        "files",
    }:
        raise ExternalMaterializationError("manifest source-tree fields do not match")
    if (
        source_tree["scope"]
        != "complete_tracked_tree_with_materialized_regular_files_and_excluded_gitlinks"
        or source_tree["source_closure_bound"] is not True
        or source_tree["git_metadata_included"] is not False
        or source_tree["symlinks_included"] is not False
        or source_tree["gitlink_content_included"] is not False
        or source_tree["normalized_directory_mode"] != "0755"
    ):
        raise ExternalMaterializationError("manifest source-tree scope does not match")
    raw_files = source_tree["files"]
    if type(raw_files) is not list or not raw_files or len(raw_files) > _MAX_TRACKED_FILES:
        raise ExternalMaterializationError("manifest files must be a bounded nonempty list")
    tracked_entry_count = _require_exact_nonnegative_int(
        source_tree["tracked_entry_count"], context="manifest tracked-entry count"
    )
    materialized_regular_file_count = _require_exact_nonnegative_int(
        source_tree["materialized_regular_file_count"],
        context="manifest materialized regular-file count",
    )
    excluded_gitlink_count = _require_exact_nonnegative_int(
        source_tree["excluded_gitlink_count"],
        context="manifest excluded-gitlink count",
    )
    portable_path_alias_count = _require_exact_nonnegative_int(
        source_tree["portable_path_alias_count"],
        context="manifest portable-path-alias count",
    )
    declared_upstream_total = _require_exact_nonnegative_int(
        source_tree["upstream_total_size_bytes"], context="manifest upstream total"
    )
    declared_materialized_total = _require_exact_nonnegative_int(
        source_tree["materialized_total_size_bytes"],
        context="manifest materialized total",
    )
    transforms = {item.path: item for item in identity.source_transforms}
    previous_path: str | None = None
    upstream_total = 0
    materialized_total = 0
    transformed_paths: set[str] = set()
    upstream_tree_records: list[tuple[str, str, str]] = []
    for index, raw_item in enumerate(raw_files):
        if type(raw_item) is not dict:
            raise ExternalMaterializationError(f"manifest file {index} must be an object")
        item = cast(dict[str, Any], raw_item)
        if set(item) != {
            "path",
            "git_mode",
            "upstream_blob_git_sha1",
            "upstream_size_bytes",
            "upstream_sha256",
            "materialized_size_bytes",
            "materialized_sha256",
            "transformed",
        }:
            raise ExternalMaterializationError(f"manifest file {index} fields do not match")
        path = _validate_relative_path(item["path"], context=f"manifest file {index} path")
        if previous_path is not None and path <= previous_path:
            raise ExternalMaterializationError("manifest files are not strictly path-sorted")
        previous_path = path
        if type(item["git_mode"]) is not str or item["git_mode"] not in {
            "100644",
            "100755",
        }:
            raise ExternalMaterializationError(f"manifest file mode is invalid: {path}")
        _require_git_sha1(item["upstream_blob_git_sha1"], context=f"manifest blob {path}")
        upstream_tree_records.append((path, item["git_mode"], item["upstream_blob_git_sha1"]))
        upstream_size = _require_exact_nonnegative_int(
            item["upstream_size_bytes"], context=f"manifest upstream size {path}"
        )
        upstream_sha256 = _require_sha256(
            item["upstream_sha256"], context=f"manifest upstream digest {path}"
        )
        materialized_size = _require_exact_nonnegative_int(
            item["materialized_size_bytes"], context=f"manifest materialized size {path}"
        )
        materialized_sha256 = _require_sha256(
            item["materialized_sha256"], context=f"manifest materialized digest {path}"
        )
        if type(item["transformed"]) is not bool:
            raise ExternalMaterializationError(f"manifest transformed flag is invalid: {path}")
        transform_pin = transforms.get(path)
        if transform_pin is None:
            if item["transformed"] is not False or (
                materialized_size,
                materialized_sha256,
            ) != (upstream_size, upstream_sha256):
                raise ExternalMaterializationError(
                    f"untransformed manifest identity does not match: {path}"
                )
        else:
            if item["transformed"] is not True or (
                upstream_size,
                upstream_sha256,
                materialized_size,
                materialized_sha256,
            ) != (
                transform_pin.upstream_size_bytes,
                transform_pin.upstream_sha256,
                transform_pin.derived_size_bytes,
                transform_pin.derived_sha256,
            ):
                raise ExternalMaterializationError(
                    f"transformed manifest identity does not match: {path}"
                )
            transformed_paths.add(path)
        upstream_total += upstream_size
        materialized_total += materialized_size
    raw_gitlinks = source_tree["excluded_gitlinks"]
    if type(raw_gitlinks) is not list or len(raw_gitlinks) > _MAX_TRACKED_FILES:
        raise ExternalMaterializationError("manifest excluded gitlinks must be a bounded list")
    if len(raw_gitlinks) != len(identity.excluded_gitlinks):
        raise ExternalMaterializationError(
            "manifest excluded-gitlink count does not match identity"
        )
    for index, (raw_item, excluded_pin) in enumerate(
        zip(raw_gitlinks, identity.excluded_gitlinks, strict=True)
    ):
        if type(raw_item) is not dict:
            raise ExternalMaterializationError(
                f"manifest excluded gitlink {index} must be an object"
            )
        item = cast(dict[str, Any], raw_item)
        if set(item) != {
            "path",
            "git_mode",
            "commit_git_sha1",
            "content_materialized",
        }:
            raise ExternalMaterializationError(
                f"manifest excluded gitlink {index} fields do not match"
            )
        path = _validate_relative_path(
            item["path"],
            context=f"manifest excluded gitlink {index} path",
        )
        if (
            path != excluded_pin.path
            or item["git_mode"] != "160000"
            or item["commit_git_sha1"] != excluded_pin.commit_git_sha1
            or item["content_materialized"] is not False
        ):
            raise ExternalMaterializationError(
                f"manifest excluded gitlink does not match identity: {path}"
            )
        _require_git_sha1(
            item["commit_git_sha1"],
            context=f"manifest excluded gitlink {path}",
        )
        upstream_tree_records.append((path, "160000", item["commit_git_sha1"]))
    raw_portable_aliases = source_tree["portable_path_aliases"]
    if type(raw_portable_aliases) is not list or len(raw_portable_aliases) > _MAX_TRACKED_FILES:
        raise ExternalMaterializationError("manifest portable path aliases must be a bounded list")
    if len(raw_portable_aliases) != len(identity.portable_path_aliases):
        raise ExternalMaterializationError(
            "manifest portable-path-alias count does not match identity"
        )
    for index, (raw_item, alias_pin) in enumerate(
        zip(raw_portable_aliases, identity.portable_path_aliases, strict=True)
    ):
        if type(raw_item) is not dict:
            raise ExternalMaterializationError(
                f"manifest portable path alias {index} must be an object"
            )
        item = cast(dict[str, Any], raw_item)
        if set(item) != {
            "path",
            "upstream_blob_git_sha1",
            "materialized_as_distinct_path",
        }:
            raise ExternalMaterializationError(
                f"manifest portable path alias {index} fields do not match"
            )
        if (
            item["path"] != alias_pin.path
            or item["upstream_blob_git_sha1"] != alias_pin.blob_git_sha1
            or item["materialized_as_distinct_path"] is not True
        ):
            raise ExternalMaterializationError(
                f"manifest portable path alias {index} does not match identity"
            )
    if transformed_paths != set(transforms):
        raise ExternalMaterializationError("manifest omits transformed source records")
    regular_paths = {item["path"] for item in raw_files}
    if (".gitmodules" in regular_paths) is not bool(identity.excluded_gitlinks):
        raise ExternalMaterializationError(
            "manifest .gitmodules presence does not match its excluded gitlinks"
        )
    if not hmac.compare_digest(
        _git_tree_sha1(upstream_tree_records, identity.portable_path_aliases),
        identity.tree_git_sha1,
    ):
        raise ExternalMaterializationError("manifest file inventory does not reconstruct the tree")
    if (
        tracked_entry_count != len(raw_files) + len(raw_gitlinks)
        or materialized_regular_file_count != len(raw_files)
        or excluded_gitlink_count != len(raw_gitlinks)
        or portable_path_alias_count != len(raw_portable_aliases)
        or declared_upstream_total != upstream_total
        or declared_materialized_total != materialized_total
    ):
        raise ExternalMaterializationError("manifest source-tree totals do not match")
    return cast(dict[str, Any], json.loads(raw.decode("ascii")))


def parse_matched_v3_external_materialization_manifest(
    raw: bytes,
    *,
    expected_manifest_sha256: str,
) -> dict[str, Any]:
    """Parse a manifest only when it embeds the exact frozen production identity."""
    manifest = parse_external_materialization_manifest(
        raw,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    if (
        manifest["identity_sha256"] != PINNED_EXTERNAL_MATERIALIZATION_IDENTITY_SHA256
        or _canonical_json(manifest["identity"]) != _PINNED_IDENTITY_BYTES
    ):
        raise ExternalMaterializationError(
            "manifest does not bind the pinned matched-v3 checkout identity"
        )
    return manifest


def _rename_no_replace_at(
    parent_descriptor: int,
    source_name: str,
    destination_name: str,
) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise ExternalMaterializationError(
            "atomic no-replace publication requires renameat2 support"
        )
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        parent_descriptor,
        os.fsencode(source_name),
        parent_descriptor,
        os.fsencode(destination_name),
        _RENAME_NOREPLACE,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
            raise ExternalMaterializationError("destination already exists")
        if error_number in {errno.ENOSYS, errno.EINVAL, errno.ENOTSUP}:
            raise ExternalMaterializationError("atomic no-replace publication is unsupported")
        raise ExternalMaterializationError(f"atomic publication failed with errno {error_number}")


def _hash_materialized_file_at(
    root_descriptor: int,
    path: str,
) -> tuple[int, str, int, str, tuple[int, int]]:
    parts = PurePosixPath(path).parts
    parent_descriptor = _open_relative_directory(root_descriptor, parts[:-1])
    descriptor = -1
    try:
        name_before = os.stat(
            parts[-1],
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        descriptor = os.open(
            parts[-1],
            os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_descriptor,
        )
    except OSError as exc:
        os.close(parent_descriptor)
        raise ExternalMaterializationError(f"materialized file is inaccessible: {path}") from exc
    digest = hashlib.sha256()
    observed_size = 0
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_ISLNK(name_before.st_mode)
            or (before.st_dev, before.st_ino) != (name_before.st_dev, name_before.st_ino)
        ):
            raise ExternalMaterializationError(f"materialized file is not regular: {path}")
        if before.st_nlink != 1:
            raise ExternalMaterializationError(f"materialized file is hardlinked: {path}")
        blob_digest = hashlib.sha1(f"blob {before.st_size}\0".encode("ascii"))
        while True:
            chunk = os.read(descriptor, _COPY_CHUNK_BYTES)
            if not chunk:
                break
            observed_size += len(chunk)
            digest.update(chunk)
            blob_digest.update(chunk)
        after = os.fstat(descriptor)
        name_after = os.stat(
            parts[-1],
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (
            observed_size != before.st_size
            or after.st_size != before.st_size
            or after.st_mtime_ns != before.st_mtime_ns
            or after.st_ctime_ns != before.st_ctime_ns
            or (after.st_dev, after.st_ino) != (before.st_dev, before.st_ino)
            or (name_after.st_dev, name_after.st_ino) != (before.st_dev, before.st_ino)
            or name_after.st_mode != before.st_mode
            or name_after.st_nlink != before.st_nlink
        ):
            raise ExternalMaterializationError(f"materialized file changed: {path}")
        mode = stat.S_IMODE(before.st_mode)
    finally:
        os.close(descriptor)
        os.close(parent_descriptor)
    return (
        observed_size,
        digest.hexdigest(),
        mode,
        blob_digest.hexdigest(),
        (before.st_dev, before.st_ino),
    )


def _materialized_inventory(root_descriptor: int) -> tuple[set[str], set[str]]:
    observed_files: set[str] = set()
    observed_directories: set[str] = set()

    def walk(directory_descriptor: int, prefix: tuple[str, ...]) -> None:
        directory_stat = os.fstat(directory_descriptor)
        if stat.S_IMODE(directory_stat.st_mode) != 0o755:
            label = "/".join(prefix) if prefix else "."
            raise ExternalMaterializationError(
                f"materialized directory mode does not match: {label}"
            )
        with os.scandir(directory_descriptor) as entries:
            for entry in entries:
                path = "/".join((*prefix, entry.name))
                if path != EXTERNAL_MATERIALIZATION_MANIFEST_FILENAME:
                    _validate_relative_path(path, context="materialized path")
                item_stat = entry.stat(follow_symlinks=False)
                if stat.S_ISDIR(item_stat.st_mode) and not stat.S_ISLNK(item_stat.st_mode):
                    observed_directories.add(path)
                    child = os.open(
                        entry.name,
                        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
                        dir_fd=directory_descriptor,
                    )
                    try:
                        child_stat = os.fstat(child)
                        if (child_stat.st_dev, child_stat.st_ino) != (
                            item_stat.st_dev,
                            item_stat.st_ino,
                        ):
                            raise ExternalMaterializationError(
                                "materialized directory changed during enumeration"
                            )
                        walk(child, (*prefix, entry.name))
                    finally:
                        os.close(child)
                elif stat.S_ISREG(item_stat.st_mode) and not stat.S_ISLNK(item_stat.st_mode):
                    if item_stat.st_nlink != 1:
                        raise ExternalMaterializationError(
                            f"materialized file is hardlinked: {path}"
                        )
                    observed_files.add(path)
                else:
                    raise ExternalMaterializationError(f"materialized entry is not regular: {path}")

    walk(root_descriptor, ())
    return observed_files, observed_directories


def _verify_external_materialization_tree_fd(
    root_descriptor: int,
    manifest_raw: bytes,
    *,
    expected_manifest_sha256: str,
) -> dict[str, Any]:
    manifest = parse_external_materialization_manifest(
        manifest_raw,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    file_records = manifest["source_tree"]["files"]
    expected_files = {item["path"]: item for item in file_records}
    expected_files[EXTERNAL_MATERIALIZATION_MANIFEST_FILENAME] = {
        "materialized_size_bytes": len(manifest_raw),
        "materialized_sha256": expected_manifest_sha256,
        "git_mode": "100644",
        "transformed": True,
    }
    expected_directories = {
        ancestor.as_posix()
        for path in expected_files
        for ancestor in PurePosixPath(path).parents
        if ancestor.as_posix() != "."
    }
    observed_files, observed_directories = _materialized_inventory(root_descriptor)
    if observed_files != set(expected_files) or observed_directories != expected_directories:
        raise ExternalMaterializationError("materialized tree contents do not match the manifest")
    materialized_identities: dict[str, tuple[int, int]] = {}
    for path, record in expected_files.items():
        size, digest, mode, blob_git_sha1, file_identity = _hash_materialized_file_at(
            root_descriptor,
            path,
        )
        materialized_identities[path] = file_identity
        expected_mode = 0o755 if record["git_mode"] == "100755" else 0o644
        if mode != expected_mode:
            raise ExternalMaterializationError(f"materialized mode does not match: {path}")
        if size != record["materialized_size_bytes"] or not hmac.compare_digest(
            digest, record["materialized_sha256"]
        ):
            raise ExternalMaterializationError(f"materialized bytes do not match: {path}")
        if path != EXTERNAL_MATERIALIZATION_MANIFEST_FILENAME and not record["transformed"]:
            if (
                record["upstream_size_bytes"] != record["materialized_size_bytes"]
                or record["upstream_sha256"] != record["materialized_sha256"]
                or not hmac.compare_digest(
                    blob_git_sha1,
                    record["upstream_blob_git_sha1"],
                )
            ):
                raise ExternalMaterializationError(
                    f"untransformed materialized blob does not match: {path}"
                )
    identity = _identity_from_payload(manifest["identity"])
    _verify_distinct_portable_alias_identities(
        materialized_identities,
        identity.portable_path_aliases,
        context="materialized tree",
    )
    return manifest


def verify_external_materialization_tree(
    destination: os.PathLike[str] | str,
    manifest_raw: bytes,
    *,
    expected_manifest_sha256: str,
) -> dict[str, Any]:
    """Verify an exact materialized tree and return its detached manifest.

    Verification binds the named root through the end of this call.  The path
    can still be replaced or mutated afterward by a process running as the
    same OS user.
    """
    root = _open_directory_anchor(
        _path_text(destination, context="materialized root"),
        context="materialized root",
    )
    try:
        manifest = _verify_external_materialization_tree_fd(
            root.descriptor,
            manifest_raw,
            expected_manifest_sha256=expected_manifest_sha256,
        )
        _recheck_directory_anchor(root, context="materialized root")
        return manifest
    finally:
        os.close(root.descriptor)


_RETAINED_CAPABILITY_CREATION_TOKEN: Final = object()


class RetainedExternalMaterializationTree:
    """Live, PID-bound descriptor capability for one verified materialized tree.

    Instances are created only by the retained-verification context managers.
    The descriptor is inherited by a child only when the caller explicitly
    supplies :attr:`subprocess_pass_fds`; serialized manifests and integer file
    descriptor values do not recreate this capability.
    """

    __slots__ = (
        "_descriptor",
        "_device",
        "_inode",
        "_manifest_raw",
        "_manifest_sha256",
        "_owner_pid",
        "_require_matched_v3_identity",
    )

    def __init__(
        self,
        creation_token: object,
        descriptor: int,
        device: int,
        inode: int,
        manifest_raw: bytes,
        manifest_sha256: str,
        *,
        require_matched_v3_identity: bool,
    ) -> None:
        if creation_token is not _RETAINED_CAPABILITY_CREATION_TOKEN:
            raise TypeError("retained capabilities can only be created by a verification context")
        self._descriptor = descriptor
        self._device = device
        self._inode = inode
        self._manifest_raw = manifest_raw
        self._manifest_sha256 = manifest_sha256
        self._owner_pid = os.getpid()
        self._require_matched_v3_identity = require_matched_v3_identity

    def __reduce__(self) -> Never:
        raise TypeError("retained filesystem capabilities cannot be serialized")

    def __reduce_ex__(self, protocol: SupportsIndex) -> Never:
        del protocol
        raise TypeError("retained filesystem capabilities cannot be serialized")

    def __copy__(self) -> Never:
        raise TypeError("retained filesystem capabilities cannot be copied")

    def __deepcopy__(self, memo: object) -> Never:
        del memo
        raise TypeError("retained filesystem capabilities cannot be copied")

    def _invalidate(self) -> None:
        descriptor = self._descriptor
        self._descriptor = -1
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass

    def _require_active(self) -> int:
        if os.getpid() != self._owner_pid:
            self._invalidate()
            raise ExternalMaterializationError(
                "retained filesystem capability is invalid after a PID change"
            )
        descriptor = self._descriptor
        if descriptor < 0:
            raise ExternalMaterializationError("retained filesystem capability is closed")
        try:
            item_stat = os.fstat(descriptor)
        except OSError as exc:
            self._invalidate()
            raise ExternalMaterializationError(
                "retained filesystem capability descriptor became inaccessible"
            ) from exc
        if not stat.S_ISDIR(item_stat.st_mode) or (item_stat.st_dev, item_stat.st_ino) != (
            self._device,
            self._inode,
        ):
            self._invalidate()
            raise ExternalMaterializationError("retained filesystem capability identity changed")
        return descriptor

    @property
    def closed(self) -> bool:
        """Return whether the context has closed or invalidated the capability."""
        if self._descriptor >= 0 and os.getpid() != self._owner_pid:
            self._invalidate()
        return self._descriptor < 0

    @property
    def proc_fd_path(self) -> str:
        """Return the stable procfs path for use while this context is active."""
        return f"/proc/self/fd/{self._require_active()}"

    @property
    def subprocess_pass_fds(self) -> tuple[int, ...]:
        """Return the narrow ``subprocess`` ``pass_fds`` fact for this capability."""
        return (self._require_active(),)

    @property
    def owner_pid(self) -> int:
        """Return the only process ID in which this capability is valid."""
        self._require_active()
        return self._owner_pid

    def reverify(self) -> dict[str, Any]:
        """Reverify the retained inode, independently of later path replacement."""
        descriptor = self._require_active()
        try:
            manifest = _verify_external_materialization_tree_fd(
                descriptor,
                self._manifest_raw,
                expected_manifest_sha256=self._manifest_sha256,
            )
            if self._require_matched_v3_identity:
                parse_matched_v3_external_materialization_manifest(
                    self._manifest_raw,
                    expected_manifest_sha256=self._manifest_sha256,
                )
            self._require_active()
            return manifest
        except BaseException:
            self._invalidate()
            raise

    def close(self) -> None:
        """Close and permanently invalidate this retained capability."""
        self._invalidate()


@contextmanager
def _retain_verified_external_materialization_tree(
    destination: os.PathLike[str] | str,
    manifest_raw: bytes,
    *,
    expected_manifest_sha256: str,
    require_matched_v3_identity: bool,
) -> Iterator[RetainedExternalMaterializationTree]:
    root = _open_directory_anchor(
        _path_text(destination, context="materialized root"),
        context="materialized root",
    )
    capability: RetainedExternalMaterializationTree | None = None
    try:
        _verify_external_materialization_tree_fd(
            root.descriptor,
            manifest_raw,
            expected_manifest_sha256=expected_manifest_sha256,
        )
        if require_matched_v3_identity:
            parse_matched_v3_external_materialization_manifest(
                manifest_raw,
                expected_manifest_sha256=expected_manifest_sha256,
            )
        _recheck_directory_anchor(root, context="materialized root")
        capability = RetainedExternalMaterializationTree(
            _RETAINED_CAPABILITY_CREATION_TOKEN,
            root.descriptor,
            root.device,
            root.inode,
            manifest_raw,
            expected_manifest_sha256,
            require_matched_v3_identity=require_matched_v3_identity,
        )
        yield capability
    finally:
        if capability is None:
            os.close(root.descriptor)
        else:
            capability.close()


def retain_verified_external_materialization_tree(
    destination: os.PathLike[str] | str,
    manifest_raw: bytes,
    *,
    expected_manifest_sha256: str,
) -> AbstractContextManager[RetainedExternalMaterializationTree]:
    """Retain a verified generic materialization inode for one context lifetime."""
    return _retain_verified_external_materialization_tree(
        destination,
        manifest_raw,
        expected_manifest_sha256=expected_manifest_sha256,
        require_matched_v3_identity=False,
    )


def retain_verified_matched_v3_external_materialization_tree(
    destination: os.PathLike[str] | str,
    manifest_raw: bytes,
    *,
    expected_manifest_sha256: str,
) -> AbstractContextManager[RetainedExternalMaterializationTree]:
    """Retain a verified tree only when it binds the exact production identity."""
    return _retain_verified_external_materialization_tree(
        destination,
        manifest_raw,
        expected_manifest_sha256=expected_manifest_sha256,
        require_matched_v3_identity=True,
    )


def _normalize_and_fsync_directory_tree(root_descriptor: int) -> None:
    def visit(directory_descriptor: int) -> None:
        with os.scandir(directory_descriptor) as entries:
            directory_names = [
                entry.name
                for entry in entries
                if stat.S_ISDIR(entry.stat(follow_symlinks=False).st_mode)
                and not stat.S_ISLNK(entry.stat(follow_symlinks=False).st_mode)
            ]
        for name in directory_names:
            child = os.open(
                name,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=directory_descriptor,
            )
            try:
                visit(child)
                os.fchmod(child, 0o755)
                os.fsync(child)
            finally:
                os.close(child)

    visit(root_descriptor)
    os.fchmod(root_descriptor, 0o755)
    os.fsync(root_descriptor)


def _safe_remove_open_directory(directory_descriptor: int) -> None:
    for name in os.listdir(directory_descriptor):
        item_stat = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
        if stat.S_ISDIR(item_stat.st_mode) and not stat.S_ISLNK(item_stat.st_mode):
            child = os.open(
                name,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=directory_descriptor,
            )
            try:
                child_stat = os.fstat(child)
                if (child_stat.st_dev, child_stat.st_ino) != (
                    item_stat.st_dev,
                    item_stat.st_ino,
                ):
                    raise ExternalMaterializationError("cleanup directory identity changed")
                _safe_remove_open_directory(child)
            finally:
                os.close(child)
            current_stat = os.stat(
                name,
                dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
            if (current_stat.st_dev, current_stat.st_ino) != (
                item_stat.st_dev,
                item_stat.st_ino,
            ):
                raise ExternalMaterializationError("cleanup directory name was replaced")
            os.rmdir(name, dir_fd=directory_descriptor)
        else:
            current_stat = os.stat(
                name,
                dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
            if (current_stat.st_dev, current_stat.st_ino) != (
                item_stat.st_dev,
                item_stat.st_ino,
            ):
                raise ExternalMaterializationError("cleanup file name was replaced")
            os.unlink(name, dir_fd=directory_descriptor)


def _remove_anchored_directory_name(
    parent_descriptor: int,
    name: str,
    identity: tuple[int, int],
) -> None:
    item_stat = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    if (
        not stat.S_ISDIR(item_stat.st_mode)
        or stat.S_ISLNK(item_stat.st_mode)
        or (item_stat.st_dev, item_stat.st_ino) != identity
    ):
        raise ExternalMaterializationError("temporary directory name was replaced")
    os.rmdir(name, dir_fd=parent_descriptor)


def _create_anchored_temp_directory(
    parent_descriptor: int,
    *,
    prefix: str = ".alberta-materialize-",
) -> tuple[str, int, tuple[int, int]]:
    if (
        type(prefix) is not str
        or not prefix
        or "/" in prefix
        or "\0" in prefix
        or len((prefix + "0" * 32).encode("utf-8", "strict")) > _MAX_PATH_COMPONENT_UTF8_BYTES
    ):
        raise ExternalMaterializationError("temporary directory prefix is invalid")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    for _ in range(128):
        name = f"{prefix}{secrets.token_hex(16)}"
        try:
            os.mkdir(name, 0o700, dir_fd=parent_descriptor)
        except FileExistsError:
            continue
        descriptor = -1
        created_identity: tuple[int, int] | None = None
        try:
            name_stat = os.stat(
                name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            created_identity = (name_stat.st_dev, name_stat.st_ino)
            if not stat.S_ISDIR(name_stat.st_mode) or stat.S_ISLNK(name_stat.st_mode):
                raise ExternalMaterializationError("new temporary directory name changed type")
            descriptor = os.open(name, flags, dir_fd=parent_descriptor)
            item_stat = os.fstat(descriptor)
            if (
                not stat.S_ISDIR(item_stat.st_mode)
                or (item_stat.st_dev, item_stat.st_ino) != created_identity
            ):
                raise ExternalMaterializationError("new temporary directory identity changed")
        except BaseException as exc:
            if descriptor >= 0:
                os.close(descriptor)
                descriptor = -1
            cleanup_error: BaseException | None = None
            if created_identity is None:
                try:
                    cleanup_stat = os.stat(
                        name,
                        dir_fd=parent_descriptor,
                        follow_symlinks=False,
                    )
                    created_identity = (cleanup_stat.st_dev, cleanup_stat.st_ino)
                except BaseException as cleanup_exc:
                    cleanup_error = cleanup_exc
            if created_identity is not None:
                try:
                    _remove_anchored_directory_name(
                        parent_descriptor,
                        name,
                        created_identity,
                    )
                except BaseException as cleanup_exc:
                    cleanup_error = cleanup_exc
            surfaced_error: BaseException
            if isinstance(exc, OSError):
                surfaced_error = ExternalMaterializationError(
                    "new temporary directory could not be anchored"
                )
            else:
                surfaced_error = exc
            if cleanup_error is not None:
                surfaced_error.add_note(
                    f"new temporary directory cleanup also failed: {cleanup_error}"
                )
            if surfaced_error is not exc:
                raise surfaced_error from exc
            raise
        if created_identity is None:
            os.close(descriptor)
            raise AssertionError("temporary directory identity was not retained")
        return name, descriptor, created_identity
    raise ExternalMaterializationError("could not allocate a unique temporary directory")


def _recheck_temp_name(
    parent_descriptor: int,
    name: str,
    descriptor: int,
    identity: tuple[int, int],
) -> None:
    descriptor_stat = os.fstat(descriptor)
    name_stat = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    if (
        not stat.S_ISDIR(name_stat.st_mode)
        or stat.S_ISLNK(name_stat.st_mode)
        or (descriptor_stat.st_dev, descriptor_stat.st_ino) != identity
        or (name_stat.st_dev, name_stat.st_ino) != identity
    ):
        raise ExternalMaterializationError("temporary directory anchor changed")


def _recheck_published_name(
    parent_descriptor: int,
    name: str,
    descriptor: int,
    identity: tuple[int, int],
) -> None:
    descriptor_stat = os.fstat(descriptor)
    name_stat = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    if (
        not stat.S_ISDIR(name_stat.st_mode)
        or stat.S_ISLNK(name_stat.st_mode)
        or (descriptor_stat.st_dev, descriptor_stat.st_ino) != identity
        or (name_stat.st_dev, name_stat.st_ino) != identity
    ):
        raise ExternalMaterializationError("published destination anchor changed")


def _fsync_parent_after_publish(parent_descriptor: int) -> None:
    os.fsync(parent_descriptor)


def _materialize_external_checkout_with_identity(
    checkout: os.PathLike[str] | str,
    destination: os.PathLike[str] | str,
    identity: ExternalCheckoutIdentity,
    derive_sources: _DeriveSources,
) -> ExternalMaterialization:
    """Internal generic core used by production and miniature-repository tests."""
    identity_payload = _identity_payload(identity)
    identity = _identity_from_payload(identity_payload)
    destination_path = _path_text(destination, context="destination")
    if not destination_path.name or destination_path.name in {".", ".."}:
        raise ExternalMaterializationError("destination must name one new directory")
    checkout_root = _checkout_root(checkout)
    try:
        parent = _open_directory_anchor(
            destination_path.parent,
            context="destination parent",
        )
    except BaseException:
        _close_checkout_anchor(checkout_root)
        raise
    if not callable(derive_sources):
        _close_checkout_anchor(checkout_root)
        os.close(parent.descriptor)
        raise ExternalMaterializationError("source derivation must be callable")
    destination_name = destination_path.name
    temp_name: str | None = None
    temp_descriptor = -1
    temp_identity: tuple[int, int] | None = None
    published = False
    try:
        _recheck_checkout_anchor(checkout_root)
        if _directory_anchor_is_within(parent, checkout_root.root):
            raise ExternalMaterializationError("destination cannot be nested within the checkout")
        try:
            existing = os.stat(
                destination_name,
                dir_fd=parent.descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            existing = None
        if existing is not None:
            raise ExternalMaterializationError("destination already exists")
        with _hermetic_git_environment() as git_environment:
            files, audited_sources = _inspect_checkout(
                checkout_root,
                identity,
                git_environment,
            )
            try:
                derived = derive_sources(dict(audited_sources))
            except ExternalMaterializationError:
                raise
            except Exception as exc:
                raise ExternalMaterializationError("source derivation failed") from exc
            if type(derived) is not _DerivedSourceSet:
                raise ExternalMaterializationError("source derivation returned the wrong type")
            if (
                derived.transport_schema_version != identity.transport_schema_version
                or not hmac.compare_digest(
                    derived.transport_descriptor_sha256,
                    identity.transport_descriptor_sha256,
                )
            ):
                raise ExternalMaterializationError("source derivation identity does not match")
            if set(derived.sources) != {item.path for item in identity.source_transforms}:
                raise ExternalMaterializationError("source derivation returned the wrong path set")
            for pin in identity.source_transforms:
                raw = derived.sources.get(pin.path)
                if type(raw) is not bytes:
                    raise ExternalMaterializationError(
                        f"derived source must be exact bytes: {pin.path}"
                    )
                if len(raw) != pin.derived_size_bytes or not hmac.compare_digest(
                    _sha256(raw), pin.derived_sha256
                ):
                    raise ExternalMaterializationError(
                        f"derived source identity drifted: {pin.path}"
                    )

            temp_name, temp_descriptor, temp_identity = _create_anchored_temp_directory(
                parent.descriptor
            )
            transformed_paths = set(derived.sources)
            for record in files:
                if record.path in transformed_paths:
                    raw = derived.sources[record.path]
                    mode = 0o755 if record.git_mode == "100755" else 0o644
                    _write_exact_bytes_at(temp_descriptor, record.path, raw, mode)
                else:
                    _copy_exact_file_at(
                        checkout_root.root.descriptor,
                        temp_descriptor,
                        record,
                    )
            manifest = _manifest(identity, files, derived)
            manifest_raw = _canonical_json(manifest)
            manifest_sha256 = _sha256(manifest_raw)
            _write_exact_bytes_at(
                temp_descriptor,
                EXTERNAL_MATERIALIZATION_MANIFEST_FILENAME,
                manifest_raw,
                0o644,
            )

            # Recheck the source before verifying the staging tree, then bind and
            # verify the same open directory again after its atomic publication.
            second_files, second_sources = _inspect_checkout(
                checkout_root,
                identity,
                git_environment,
            )
            if second_files != files or second_sources != audited_sources:
                raise ExternalMaterializationError("checkout changed during materialization")
            _normalize_and_fsync_directory_tree(temp_descriptor)
            _verify_external_materialization_tree_fd(
                temp_descriptor,
                manifest_raw,
                expected_manifest_sha256=manifest_sha256,
            )
            _recheck_directory_anchor(parent, context="destination parent")
            _recheck_temp_name(
                parent.descriptor,
                temp_name,
                temp_descriptor,
                temp_identity,
            )
            _rename_no_replace_at(
                parent.descriptor,
                temp_name,
                destination_name,
            )
            published = True
            _recheck_directory_anchor(parent, context="destination parent")
            _recheck_published_name(
                parent.descriptor,
                destination_name,
                temp_descriptor,
                temp_identity,
            )
            _verify_external_materialization_tree_fd(
                temp_descriptor,
                manifest_raw,
                expected_manifest_sha256=manifest_sha256,
            )
            _recheck_published_name(
                parent.descriptor,
                destination_name,
                temp_descriptor,
                temp_identity,
            )
            try:
                _fsync_parent_after_publish(parent.descriptor)
            except OSError as exc:
                raise ExternalMaterializationError(
                    "publication completed but destination-parent fsync failed"
                ) from exc
            _recheck_directory_anchor(parent, context="destination parent")
            _recheck_published_name(
                parent.descriptor,
                destination_name,
                temp_descriptor,
                temp_identity,
            )
            return ExternalMaterialization(
                destination=destination_path,
                manifest_path=(destination_path / EXTERNAL_MATERIALIZATION_MANIFEST_FILENAME),
                manifest_bytes=manifest_raw,
                manifest_sha256=manifest_sha256,
            )
    except BaseException as exc:
        surfaced_error: BaseException
        if isinstance(exc, OSError):
            surfaced_error = ExternalMaterializationError(f"external materialization failed: {exc}")
        else:
            surfaced_error = exc
        if temp_descriptor >= 0 and not published:
            cleanup_error: BaseException | None = None
            try:
                _safe_remove_open_directory(temp_descriptor)
            except BaseException as cleanup_exc:
                cleanup_error = cleanup_exc
            finally:
                os.close(temp_descriptor)
                temp_descriptor = -1
            if cleanup_error is None and temp_name is not None and temp_identity is not None:
                try:
                    _remove_anchored_directory_name(
                        parent.descriptor,
                        temp_name,
                        temp_identity,
                    )
                except BaseException as cleanup_exc:
                    cleanup_error = cleanup_exc
            if cleanup_error is not None:
                surfaced_error.add_note(f"temporary cleanup also failed: {cleanup_error}")
        if surfaced_error is not exc:
            raise surfaced_error from exc
        raise
    finally:
        if temp_descriptor >= 0:
            os.close(temp_descriptor)
        _close_checkout_anchor(checkout_root)
        os.close(parent.descriptor)


def materialize_matched_v3_external_checkout(
    checkout: os.PathLike[str] | str,
    destination: os.PathLike[str] | str,
) -> ExternalMaterialization:
    """Atomically materialize only the exact frozen matched-v3 external checkout.

    The returned paths identify the publication at return time.  A process
    running as the same OS user can subsequently replace or mutate them.
    """
    return _materialize_external_checkout_with_identity(
        checkout,
        destination,
        pinned_external_checkout_identity(),
        _production_derive,
    )


__all__ = [
    "EXTERNAL_MATERIALIZATION_IDENTITY_SCHEMA_VERSION",
    "EXTERNAL_MATERIALIZATION_MANIFEST_FILENAME",
    "EXTERNAL_MATERIALIZATION_SCHEMA_VERSION",
    "PINNED_EXTERNAL_MATERIALIZATION_IDENTITY_SHA256",
    "ExternalCheckoutIdentity",
    "ExternalMaterialization",
    "ExternalMaterializationError",
    "GitlinkPin",
    "PortablePathAliasPin",
    "RetainedExternalMaterializationTree",
    "SourceTransformPin",
    "canonical_pinned_external_checkout_identity_bytes",
    "materialize_matched_v3_external_checkout",
    "parse_external_materialization_manifest",
    "parse_matched_v3_external_materialization_manifest",
    "parse_pinned_external_checkout_identity",
    "pinned_external_checkout_identity",
    "retain_verified_external_materialization_tree",
    "retain_verified_matched_v3_external_materialization_tree",
    "verify_external_materialization_tree",
]
