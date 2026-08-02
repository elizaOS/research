#!/usr/bin/env python3
"""Nonpromoting public-seed probe for the frozen causal-map q grid.

The matched-current panel contains three ``respawn_safety_quantile`` values at
one fixed exploration probability.  This probe checks whether that grid shows
any behavioral divergence on the exact public Foragax task before the expensive
matched campaign is launched.  It is deliberately outside the frozen
candidate source tree: the candidate tree and configuration bytes are checked
against the pinned qualification, then executed in an isolated child process.

Only action-tree digests, canonical first-divergence proofs, at most two action
scalars per divergent pair, and bounded respawn-estimator aggregates cross the
child boundary.  The action scalars are in the public four-action domain and
are necessary to verify the indexed Merkle leaves.  Rewards are consumed one
scalar at a time by the frozen policy and are never returned, retained as a
host array, scored, or written.

This is permanently open-development and nonpromoting.  A run with no
pairwise action divergence writes a rejection receipt and exits nonzero.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import io
import json
import math
import os
import platform
import secrets
import selectors
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from statistics import NormalDist
from typing import Any, Final, cast

SCHEMA_VERSION: Final = "alberta.forager_causal_grid_divergence_probe.v1"
CHILD_SCHEMA_VERSION: Final = (
    "alberta.forager_causal_grid_divergence_probe_child.v1"
)
CLASSIFICATION: Final = "open_development_nonpromoting"
SEED_CLASS: Final = "public_nonbenchmark_seed"
FIXED_SEED: Final = 0
FIXED_EXPLORATION_PROBABILITY: Final = 0.05
FIXED_STEPS: Final = 10_000
DIAGNOSTIC_CHANNEL_COUNT: Final = 3
MATCHED_HORIZON: Final = 499_712
QUALIFIED_IMAGE_SHA256: Final = (
    "5ecaabefce6439a8731c19e7a55fedb666788242baf035e6ffca86eb31299768"
)
QUALIFICATION_MANIFEST_SHA256: Final = (
    "90182e6d9d79c4543648881f67d969d567c42163b93b2440377e5d36b2fb4d9a"
)
SOURCE_INVENTORY_SHA256: Final = (
    "e1cd51e16db0533b8a55c99cb343705be14b11084b7c8a02fb8ced66558cee6f"
)
EXECUTOR_INVENTORY_SHA256: Final = (
    "3fd69fdc2f5ab373dfe8a99c494bcd41e79cbdbd8d7d5a6f5b90ee918eb6eeea"
)
SOURCE_ARCHIVE_SHA256: Final = (
    "8f66a8cb2357e4d003adf2ac8084c75c7c46ac07cbbb8dddd6cce6e39f88bd79"
)
SNAPSHOT_DESCRIPTOR_SHA256: Final = (
    "8a390e0ed1c88e373b0e0c9a682e2e9dec79370dc02e58a3a0ff4f8233827fa7"
)
RUNTIME_PROFILE_SHA256: Final = (
    "7170418e8082babbf17ebfbbb639ee75fcd8b5ae3931d35b3fb9199ea2bfd9b3"
)
TASK_IDENTITY_SHA256: Final = (
    "3a353233a7eb48915220a0387d41ecafd1028b0316b04e32c09a30c70bbcb159"
)
ENVIRONMENT_RNG_SCHEDULE_SHA256: Final = (
    "51d811e6fccd2b015b1703f22775f880089bbca3fc8938421ad3e18526882cb0"
)
OPEN_PROTOCOL_SHA256: Final = (
    "b17da8af19cac570c426c74ff6bbc0e4ee0a4b95a4486c3ad5da19ceb3f8176e"
)
MAXIMUM_JSON_BYTES: Final = 16 * 1024 * 1024
MAXIMUM_STDERR_BYTES: Final = 64 * 1024
MAXIMUM_SOURCE_FILES: Final = 10_000
MAXIMUM_SOURCE_BYTES: Final = 512 * 1024**2
ACTION_MERKLE_ENCODING: Final = (
    "indexed_uint8_actions_sha256_binary_tree_16384_v1"
)
ACTION_MERKLE_TREE_LEAF_COUNT: Final = 16_384
ACTION_MERKLE_ROOT_LEVEL: Final = 14
_ACTION_MERKLE_LEAF_DOMAIN: Final = b"alberta.causal_q.action_leaf.v1\0"
_ACTION_MERKLE_PADDING_DOMAIN: Final = b"alberta.causal_q.padding_leaf.v1\0"
_ACTION_MERKLE_NODE_DOMAIN: Final = b"alberta.causal_q.action_node.v1\0"
_ACTION_MERKLE_ROOT_DOMAIN: Final = b"alberta.causal_q.action_root.v1\0"

_REPOSITORY_ROOT = Path(__file__).resolve().parent
DEFAULT_QUALIFICATION_ROOT = (
    _REPOSITORY_ROOT
    / "outputs"
    / "forager"
    / "matched_current_qualification_2c3b214c_v1"
)
DEFAULT_OUTPUT_ROOT = (
    _REPOSITORY_ROOT
    / "outputs"
    / "forager"
    / "development"
    / "causal_q_grid_divergence_seed0_v1"
)
DEFAULT_OCI_RUNTIME = Path(shutil.which("docker") or "/usr/bin/docker")

_CANDIDATE_CONFIGURATIONS: Final[tuple[tuple[str, float, str], ...]] = (
    (
        "causal_e050_q050",
        0.50,
        "916bd37e04c39dc16c19153032fc1c3baf12a941efb3df95860ee9f03c1ef331",
    ),
    (
        "causal_e050_q075",
        0.75,
        "afaa3ea47cd410a43541c85976fa6f718c5f70504494f70496385ec37ea84a63",
    ),
    (
        "causal_e050_q090",
        0.90,
        "ab555510e08a98e733d01a9b145d19073bb17ba31681a459a55a978d5a4faf33",
    ),
)


class CausalGridDivergenceProbeError(RuntimeError):
    """The public-seed divergence probe violated a frozen contract."""


class ReceiptPublishedButUncertainError(CausalGridDivergenceProbeError):
    """The final path was published but its durability could not be confirmed."""


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CausalGridDivergenceProbeError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite(token: str) -> Any:
    raise CausalGridDivergenceProbeError(
        f"non-finite JSON number {token!r} is forbidden"
    )


def _parse_finite_json_float(token: str) -> float:
    try:
        value = float(token)
    except (ValueError, OverflowError) as exc:
        raise CausalGridDivergenceProbeError(
            f"invalid JSON number {token!r}"
        ) from exc
    if not math.isfinite(value):
        raise CausalGridDivergenceProbeError(
            f"non-finite JSON number {token!r} is forbidden"
        )
    return value


def _decode_strict_json(raw: bytes, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
            parse_float=_parse_finite_json_float,
        )
    except CausalGridDivergenceProbeError:
        raise
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise CausalGridDivergenceProbeError(f"{label} is not strict UTF-8 JSON") from exc
    if type(payload) is not dict:
        raise CausalGridDivergenceProbeError(f"{label} must be a JSON object")
    return payload


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


def _read_stable_regular_descriptor(
    descriptor: int,
    *,
    label: str,
    maximum: int,
    allow_empty: bool = False,
) -> bytes:
    before = os.fstat(descriptor)
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_size < (0 if allow_empty else 1)
        or before.st_size > maximum
    ):
        raise CausalGridDivergenceProbeError(
            f"{label} violates the bounded single-link regular-file contract"
        )
    chunks: list[bytes] = []
    total = 0
    while chunk := os.read(descriptor, min(1024 * 1024, maximum + 1 - total)):
        chunks.append(chunk)
        total += len(chunk)
        if total > maximum:
            raise CausalGridDivergenceProbeError(f"{label} exceeds its byte bound")
    after = os.fstat(descriptor)
    if _stat_identity(before) != _stat_identity(after):
        raise CausalGridDivergenceProbeError(f"{label} changed while being read")
    return b"".join(chunks)


def _read_stable_regular_file(
    path: Path,
    *,
    label: str,
    maximum: int,
    allow_empty: bool = False,
) -> bytes:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise CausalGridDivergenceProbeError(f"cannot open {label}") from exc
    try:
        return _read_stable_regular_descriptor(
            descriptor,
            label=label,
            maximum=maximum,
            allow_empty=allow_empty,
        )
    finally:
        os.close(descriptor)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _read_probe_source(*, expected_sha256: str | None = None) -> bytes:
    raw = _read_stable_regular_file(
        Path(__file__).resolve(strict=True),
        label="causal-grid divergence probe source",
        maximum=MAXIMUM_JSON_BYTES,
    )
    observed_sha256 = _sha256(raw)
    if expected_sha256 is not None and observed_sha256 != expected_sha256:
        raise CausalGridDivergenceProbeError(
            "causal-grid divergence probe source changed during execution"
        )
    return raw


def _action_leaf_digest(index: int, action: int) -> bytes:
    if (
        type(index) is not int
        or not 0 <= index < FIXED_STEPS
        or type(action) is not int
        or not 0 <= action < 4
    ):
        raise CausalGridDivergenceProbeError("action leaf index or value is invalid")
    digest = hashlib.sha256(_ACTION_MERKLE_LEAF_DOMAIN)
    digest.update(index.to_bytes(8, byteorder="big", signed=False))
    digest.update(bytes((action,)))
    return digest.digest()


def _padding_leaf_digest(index: int) -> bytes:
    if type(index) is not int or not FIXED_STEPS <= index < ACTION_MERKLE_TREE_LEAF_COUNT:
        raise CausalGridDivergenceProbeError("action-tree padding index is invalid")
    digest = hashlib.sha256(_ACTION_MERKLE_PADDING_DOMAIN)
    digest.update(index.to_bytes(8, byteorder="big", signed=False))
    return digest.digest()


def _action_parent_digest(level: int, left: bytes, right: bytes) -> bytes:
    if (
        type(level) is not int
        or not 1 <= level <= ACTION_MERKLE_ROOT_LEVEL
        or type(left) is not bytes
        or type(right) is not bytes
        or len(left) != 32
        or len(right) != 32
    ):
        raise CausalGridDivergenceProbeError("action-tree parent inputs are invalid")
    digest = hashlib.sha256(_ACTION_MERKLE_NODE_DOMAIN)
    digest.update(level.to_bytes(2, byteorder="big", signed=False))
    digest.update(left)
    digest.update(right)
    return digest.digest()


def _action_root_digest(raw_tree_root: bytes) -> bytes:
    if type(raw_tree_root) is not bytes or len(raw_tree_root) != 32:
        raise CausalGridDivergenceProbeError("raw action-tree root is invalid")
    digest = hashlib.sha256(_ACTION_MERKLE_ROOT_DOMAIN)
    digest.update(FIXED_STEPS.to_bytes(8, byteorder="big", signed=False))
    digest.update(
        ACTION_MERKLE_TREE_LEAF_COUNT.to_bytes(8, byteorder="big", signed=False)
    )
    encoding = ACTION_MERKLE_ENCODING.encode("ascii")
    digest.update(len(encoding).to_bytes(2, byteorder="big", signed=False))
    digest.update(encoding)
    digest.update(raw_tree_root)
    return digest.digest()


def _build_action_merkle_tree(
    action_bytes: bytes,
) -> tuple[tuple[bytes, ...], ...]:
    if type(action_bytes) is not bytes or len(action_bytes) != FIXED_STEPS:
        raise CausalGridDivergenceProbeError(
            "action tree requires the exact fixed-length byte trace"
        )
    if any(action >= 4 for action in action_bytes):
        raise CausalGridDivergenceProbeError("action tree contains an invalid action")
    leaves = tuple(
        _action_leaf_digest(index, action)
        if index < FIXED_STEPS
        else _padding_leaf_digest(index)
        for index, action in enumerate(
            action_bytes + bytes(ACTION_MERKLE_TREE_LEAF_COUNT - FIXED_STEPS)
        )
    )
    levels: list[tuple[bytes, ...]] = [leaves]
    current = leaves
    for level in range(1, ACTION_MERKLE_ROOT_LEVEL + 1):
        current = tuple(
            _action_parent_digest(level, current[index], current[index + 1])
            for index in range(0, len(current), 2)
        )
        levels.append(current)
    if len(current) != 1:
        raise CausalGridDivergenceProbeError("action tree did not reduce to one root")
    return tuple(levels)


def _action_merkle_root_sha256(tree: tuple[tuple[bytes, ...], ...]) -> str:
    if len(tree) != ACTION_MERKLE_ROOT_LEVEL + 1 or len(tree[-1]) != 1:
        raise CausalGridDivergenceProbeError("action tree shape is invalid")
    return _action_root_digest(tree[-1][0]).hex()


def _first_divergence_merkle_proof(
    left_tree: tuple[tuple[bytes, ...], ...],
    right_tree: tuple[tuple[bytes, ...], ...],
    *,
    divergence_index: int,
    left_action: int,
    right_action: int,
) -> dict[str, Any]:
    if (
        type(divergence_index) is not int
        or not 0 <= divergence_index < FIXED_STEPS
        or left_action == right_action
    ):
        raise CausalGridDivergenceProbeError(
            "first-divergence proof inputs are inconsistent"
        )
    levels: list[dict[str, Any]] = []
    left_index = 0
    right_index = 0
    for level in range(ACTION_MERKLE_ROOT_LEVEL, 0, -1):
        left_children = left_tree[level - 1]
        right_children = right_tree[level - 1]
        left_child_index = left_index * 2
        right_child_index = right_index * 2
        followed_child = (
            "right" if divergence_index & (1 << (level - 1)) else "left"
        )
        levels.append(
            {
                "followed_child": followed_child,
                "level": level,
                "left_arm_left_child_sha256": left_children[
                    left_child_index
                ].hex(),
                "left_arm_right_child_sha256": left_children[
                    left_child_index + 1
                ].hex(),
                "right_arm_left_child_sha256": right_children[
                    right_child_index
                ].hex(),
                "right_arm_right_child_sha256": right_children[
                    right_child_index + 1
                ].hex(),
            }
        )
        branch_offset = 1 if followed_child == "right" else 0
        left_index = left_child_index + branch_offset
        right_index = right_child_index + branch_offset
    return {
        "divergence_index": divergence_index,
        "encoding": ACTION_MERKLE_ENCODING,
        "left_action": left_action,
        "levels": levels,
        "right_action": right_action,
        "tree_leaf_count": ACTION_MERKLE_TREE_LEAF_COUNT,
    }


def _require_sha256(value: Any, *, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CausalGridDivergenceProbeError(f"{label} must be a lowercase SHA-256")
    return value


def _require_int(
    value: Any,
    *,
    label: str,
    minimum: int,
    maximum: int,
) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise CausalGridDivergenceProbeError(
            f"{label} must be an integer in [{minimum}, {maximum}]"
        )
    return value


def _source_inventory_records(source_root: Path) -> list[dict[str, Any]]:
    try:
        root_metadata = source_root.lstat()
    except OSError as exc:
        raise CausalGridDivergenceProbeError("frozen source root is missing") from exc
    if not stat.S_ISDIR(root_metadata.st_mode) or source_root.is_symlink():
        raise CausalGridDivergenceProbeError(
            "frozen source root must be a non-symlink directory"
        )
    records: list[dict[str, Any]] = []
    total_bytes = 0
    for path in sorted(source_root.rglob("*"), key=lambda item: item.as_posix()):
        try:
            metadata = path.lstat()
        except OSError as exc:
            raise CausalGridDivergenceProbeError("cannot inspect frozen source") from exc
        if stat.S_ISDIR(metadata.st_mode):
            if path.is_symlink():
                raise CausalGridDivergenceProbeError("frozen source contains a symlink")
            continue
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise CausalGridDivergenceProbeError(
                "frozen source contains a link or special file"
            )
        relative = path.relative_to(source_root).as_posix()
        raw = _read_stable_regular_file(
            path,
            label=f"frozen source file {relative}",
            maximum=MAXIMUM_SOURCE_BYTES,
            allow_empty=True,
        )
        total_bytes += len(raw)
        if len(records) >= MAXIMUM_SOURCE_FILES or total_bytes > MAXIMUM_SOURCE_BYTES:
            raise CausalGridDivergenceProbeError(
                "frozen source exceeds its file-count or total-byte bound"
            )
        records.append(
            {
                "mode": stat.S_IMODE(metadata.st_mode),
                "path": relative,
                "sha256": _sha256(raw),
                "size_bytes": len(raw),
            }
        )
    if not records:
        raise CausalGridDivergenceProbeError("frozen source inventory is empty")
    records.sort(key=lambda record: str(record["path"]))
    return records


def _verify_source_identity(
    source_root: Path,
    inventory_path: Path,
    *,
    expected_inventory_sha256: str = EXECUTOR_INVENTORY_SHA256,
) -> None:
    files = _load_source_inventory_records(
        inventory_path,
        expected_inventory_sha256=expected_inventory_sha256,
    )
    if list(files) != _source_inventory_records(source_root):
        raise CausalGridDivergenceProbeError(
            "frozen Alberta source bytes differ from the qualified inventory"
        )


def _load_source_inventory_records(
    inventory_path: Path,
    *,
    expected_inventory_sha256: str = EXECUTOR_INVENTORY_SHA256,
) -> tuple[dict[str, Any], ...]:
    raw = _read_stable_regular_file(
        inventory_path,
        label="frozen Alberta source inventory",
        maximum=MAXIMUM_JSON_BYTES,
    )
    if _sha256(raw) != expected_inventory_sha256:
        raise CausalGridDivergenceProbeError(
            "frozen Alberta source inventory digest differs from qualification"
        )
    payload = _decode_strict_json(raw, label="frozen Alberta source inventory")
    if raw != _canonical_json_bytes(payload):
        raise CausalGridDivergenceProbeError(
            "frozen Alberta source inventory is not canonical JSON"
        )
    if set(payload) != {"schema_version", "files"} or payload["schema_version"] != (
        "alberta.forager_source_inventory.v1"
    ):
        raise CausalGridDivergenceProbeError(
            "frozen Alberta source inventory schema is unsupported"
        )
    files = payload["files"]
    if type(files) is not list or not files or len(files) > MAXIMUM_SOURCE_FILES:
        raise CausalGridDivergenceProbeError("frozen source inventory file list is invalid")
    checked: list[dict[str, Any]] = []
    total_bytes = 0
    previous_path = ""
    for index, record in enumerate(files):
        if type(record) is not dict or set(record) != {
            "mode",
            "path",
            "sha256",
            "size_bytes",
        }:
            raise CausalGridDivergenceProbeError(
                f"frozen source inventory record[{index}] fields drifted"
            )
        path_text = record["path"]
        relative = PurePosixPath(path_text) if type(path_text) is str else None
        if (
            relative is None
            or relative.is_absolute()
            or path_text != relative.as_posix()
            or not relative.parts
            or any(part in {"", ".", ".."} for part in relative.parts)
            or path_text <= previous_path
        ):
            raise CausalGridDivergenceProbeError(
                f"frozen source inventory record[{index}] path is unsafe or unordered"
            )
        mode = _require_int(
            record["mode"],
            label=f"source inventory record[{index}] mode",
            minimum=0,
            maximum=0o777,
        )
        size = _require_int(
            record["size_bytes"],
            label=f"source inventory record[{index}] size",
            minimum=0,
            maximum=MAXIMUM_SOURCE_BYTES,
        )
        digest = _require_sha256(
            record["sha256"],
            label=f"source inventory record[{index}] digest",
        )
        total_bytes += size
        if total_bytes > MAXIMUM_SOURCE_BYTES:
            raise CausalGridDivergenceProbeError(
                "frozen source inventory exceeds its total-byte bound"
            )
        checked.append(
            {
                "mode": mode,
                "path": path_text,
                "sha256": digest,
                "size_bytes": size,
            }
        )
        previous_path = path_text
    return tuple(checked)


def _extract_pinned_source_archive(
    qualification_root: Path,
    destination_root: Path,
) -> None:
    """Reconstruct the exact source snapshot inside the child's private tmpfs."""
    if destination_root.exists() or destination_root.is_symlink():
        raise CausalGridDivergenceProbeError(
            "private source extraction destination unexpectedly exists"
        )
    inventory_path = qualification_root / "sources" / "alberta" / "inventory.json"
    records = _load_source_inventory_records(inventory_path)
    expected_by_path = {str(record["path"]): record for record in records}
    expected_directories = {
        parent.as_posix()
        for path_text in expected_by_path
        for parent in PurePosixPath(path_text).parents
        if parent.as_posix() != "."
    }
    archive_raw = _read_stable_regular_file(
        qualification_root / "sources" / "alberta" / "source.tar",
        label="private source extraction archive",
        maximum=MAXIMUM_SOURCE_BYTES,
    )
    if _sha256(archive_raw) != SOURCE_ARCHIVE_SHA256:
        raise CausalGridDivergenceProbeError(
            "private source extraction archive differs from qualification"
        )
    destination_root.mkdir(mode=0o700, parents=False)
    seen_files: set[str] = set()
    seen_directories: set[str] = set()
    try:
        with tarfile.open(fileobj=io.BytesIO(archive_raw), mode="r:") as archive:
            for member in archive:
                name = (
                    member.name[:-1]
                    if member.isdir() and member.name.endswith("/")
                    else member.name
                )
                relative = PurePosixPath(name)
                if (
                    not name
                    or relative.is_absolute()
                    or name != relative.as_posix()
                    or any(part in {"", ".", ".."} for part in relative.parts)
                    or member.pax_headers
                    or member.linkname
                ):
                    raise CausalGridDivergenceProbeError(
                        "pinned source archive contains an unsafe member"
                    )
                destination = destination_root.joinpath(*relative.parts)
                if member.isdir():
                    if (
                        name not in expected_directories
                        or name in seen_directories
                        or member.size != 0
                    ):
                        raise CausalGridDivergenceProbeError(
                            "pinned source archive directory inventory drifted"
                        )
                    destination.mkdir(mode=0o755, parents=True, exist_ok=True)
                    seen_directories.add(name)
                    continue
                record = expected_by_path.get(name)
                if (
                    record is None
                    or name in seen_files
                    or not member.isreg()
                    or member.size != record["size_bytes"]
                    or member.mode != record["mode"]
                ):
                    raise CausalGridDivergenceProbeError(
                        "pinned source archive file inventory drifted"
                    )
                source = archive.extractfile(member)
                if source is None:
                    raise CausalGridDivergenceProbeError(
                        "pinned source archive file has no payload"
                    )
                raw = source.read(member.size + 1)
                if len(raw) != member.size or source.read(1):
                    raise CausalGridDivergenceProbeError(
                        "pinned source archive member length drifted"
                    )
                if _sha256(raw) != record["sha256"]:
                    raise CausalGridDivergenceProbeError(
                        "pinned source archive member digest drifted"
                    )
                destination.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
                descriptor = os.open(
                    destination,
                    os.O_WRONLY
                    | os.O_CREAT
                    | os.O_EXCL
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_NOFOLLOW", 0),
                    mode=int(record["mode"]),
                )
                try:
                    view = memoryview(raw)
                    offset = 0
                    while offset < len(view):
                        written = os.write(descriptor, view[offset:])
                        if written < 1:
                            raise OSError("short write while extracting pinned source")
                        offset += written
                    os.fchmod(descriptor, int(record["mode"]))
                finally:
                    os.close(descriptor)
                seen_files.add(name)
    except (tarfile.TarError, OSError) as exc:
        if isinstance(exc, CausalGridDivergenceProbeError):
            raise
        raise CausalGridDivergenceProbeError(
            "cannot reconstruct the pinned source archive"
        ) from exc
    if seen_files != set(expected_by_path) or seen_directories != expected_directories:
        raise CausalGridDivergenceProbeError(
            "pinned source archive omits inventory members"
        )
    if _source_inventory_records(destination_root) != list(records):
        raise CausalGridDivergenceProbeError(
            "private extracted source differs from the pinned inventory"
        )


def _configuration_path(qualification_root: Path, candidate_id: str) -> Path:
    return qualification_root / "configurations" / candidate_id / "derived.json"


def _load_bound_inputs(
    qualification_root: Path,
    source_root: Path,
) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
    manifest_path = qualification_root / "manifest.json"
    manifest_raw = _read_stable_regular_file(
        manifest_path,
        label="matched qualification manifest",
        maximum=MAXIMUM_JSON_BYTES,
    )
    if _sha256(manifest_raw) != QUALIFICATION_MANIFEST_SHA256:
        raise CausalGridDivergenceProbeError(
            "matched qualification manifest differs from the frozen v1 identity"
        )
    manifest = _decode_strict_json(manifest_raw, label="matched qualification manifest")
    if manifest_raw != _canonical_json_bytes(manifest):
        raise CausalGridDivergenceProbeError(
            "matched qualification manifest is not canonical JSON"
        )
    try:
        alberta_source = manifest["sources"]["alberta"]
        binding = alberta_source["binding"]
        inventory_binding = alberta_source["inventory"]
        archive_binding = alberta_source["archive"]
        runtime_qualification = manifest["runtime_qualification"]
        candidates = manifest["candidates"]
    except (KeyError, TypeError) as exc:
        raise CausalGridDivergenceProbeError(
            "matched qualification manifest omits required Alberta bindings"
        ) from exc
    if (
        manifest.get("classification") != "content_only_unendorsed_nonpromoting"
        or manifest.get("promotion_authorized") is not False
        or manifest.get("open_protocol_sha256") != OPEN_PROTOCOL_SHA256
        or binding.get("inventory_sha256") != SOURCE_INVENTORY_SHA256
        or binding.get("archive_sha256") != SOURCE_ARCHIVE_SHA256
        or binding.get("snapshot_descriptor_sha256")
        != SNAPSHOT_DESCRIPTOR_SHA256
        or inventory_binding.get("canonical_sha256") != EXECUTOR_INVENTORY_SHA256
        or inventory_binding.get("path") != "sources/alberta/inventory.json"
        or archive_binding.get("path") != "sources/alberta/source.tar"
        or archive_binding.get("sha256") != SOURCE_ARCHIVE_SHA256
        or runtime_qualification.get("image_sha256") != QUALIFIED_IMAGE_SHA256
        or runtime_qualification.get("runtime_profile_sha256")
        != RUNTIME_PROFILE_SHA256
    ):
        raise CausalGridDivergenceProbeError(
            "matched qualification manifest Alberta identity drifted"
        )
    inventory_path = qualification_root / "sources" / "alberta" / "inventory.json"
    _verify_source_identity(source_root, inventory_path)
    archive_raw = _read_stable_regular_file(
        qualification_root / "sources" / "alberta" / "source.tar",
        label="frozen Alberta source archive",
        maximum=MAXIMUM_SOURCE_BYTES,
    )
    if _sha256(archive_raw) != SOURCE_ARCHIVE_SHA256:
        raise CausalGridDivergenceProbeError(
            "frozen Alberta source archive differs from qualification"
        )
    snapshot_raw = _read_stable_regular_file(
        qualification_root / "sources" / "alberta" / "snapshot-descriptor.json",
        label="frozen Alberta snapshot descriptor",
        maximum=MAXIMUM_JSON_BYTES,
    )
    if _sha256(snapshot_raw) != SNAPSHOT_DESCRIPTOR_SHA256:
        raise CausalGridDivergenceProbeError(
            "frozen Alberta snapshot descriptor differs from qualification"
        )

    loaded: list[dict[str, Any]] = []
    base_configuration: dict[str, Any] | None = None
    for candidate_id, quantile, expected_digest in _CANDIDATE_CONFIGURATIONS:
        path = _configuration_path(qualification_root, candidate_id)
        raw = _read_stable_regular_file(
            path,
            label=f"frozen configuration {candidate_id}",
            maximum=MAXIMUM_JSON_BYTES,
        )
        if _sha256(raw) != expected_digest:
            raise CausalGridDivergenceProbeError(
                f"frozen configuration {candidate_id} digest drifted"
            )
        payload = _decode_strict_json(raw, label=f"frozen configuration {candidate_id}")
        if raw != _canonical_json_bytes(payload):
            raise CausalGridDivergenceProbeError(
                f"frozen configuration {candidate_id} is not canonical JSON"
            )
        original_path = qualification_root / "configurations" / candidate_id / "original.json"
        original_raw = _read_stable_regular_file(
            original_path,
            label=f"original frozen configuration {candidate_id}",
            maximum=MAXIMUM_JSON_BYTES,
        )
        if original_raw != raw or _sha256(original_raw) != expected_digest:
            raise CausalGridDivergenceProbeError(
                f"original and derived configuration bytes differ for {candidate_id}"
            )
        try:
            candidate_manifest = candidates[candidate_id]
            manifest_configuration = candidate_manifest["configuration"]
            manifest_binding = manifest_configuration["binding"]
            configuration = payload["configuration"]
        except (KeyError, TypeError) as exc:
            raise CausalGridDivergenceProbeError(
                f"frozen configuration binding for {candidate_id} is incomplete"
            ) from exc
        if (
            set(payload)
            != {"schema_version", "implementation_kind", "configuration"}
            or payload["schema_version"]
            != "alberta.forager_matched_worker_configuration.v1"
            or payload["implementation_kind"] != "alberta_causal_map"
            or manifest_configuration.get("derived_path")
            != f"configurations/{candidate_id}/derived.json"
            or manifest_configuration.get("original_path")
            != f"configurations/{candidate_id}/original.json"
            or manifest_binding.get("derived_sha256") != expected_digest
            or manifest_binding.get("original_sha256") != expected_digest
            or manifest_binding.get("allowed_transforms") != []
            or type(configuration) is not dict
            or configuration.get("exploration_probability")
            != FIXED_EXPLORATION_PROBABILITY
            or configuration.get("respawn_safety_quantile") != quantile
            or configuration.get("world_shape") != [15, 15]
        ):
            raise CausalGridDivergenceProbeError(
                f"frozen configuration semantics for {candidate_id} drifted"
            )
        expected_z = 0.0 if quantile == 0.5 else NormalDist().inv_cdf(quantile)
        declared_z = configuration.get("respawn_quantile_z")
        if type(declared_z) not in (int, float) or not math.isclose(
            float(cast(int | float, declared_z)),
            expected_z,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise CausalGridDivergenceProbeError(
                f"frozen configuration z binding for {candidate_id} drifted"
            )
        comparison = dict(configuration)
        comparison.pop("respawn_safety_quantile")
        comparison.pop("respawn_quantile_z")
        if base_configuration is None:
            base_configuration = comparison
        elif comparison != base_configuration:
            raise CausalGridDivergenceProbeError(
                "q-grid configurations differ outside the frozen quantile fields"
            )
        capability = candidate_manifest.get("capability_receipt")
        if type(capability) is not dict or set(capability) != {"path", "sha256"}:
            raise CausalGridDivergenceProbeError(
                f"capability receipt binding for {candidate_id} is incomplete"
            )
        capability_path = qualification_root / str(capability["path"])
        capability_raw = _read_stable_regular_file(
            capability_path,
            label=f"capability receipt {candidate_id}",
            maximum=MAXIMUM_JSON_BYTES,
        )
        if _sha256(capability_raw) != capability["sha256"]:
            raise CausalGridDivergenceProbeError(
                f"capability receipt digest drifted for {candidate_id}"
            )
        capability_payload = _decode_strict_json(
            capability_raw,
            label=f"capability receipt {candidate_id}",
        )
        if capability_raw != _canonical_json_bytes(capability_payload) or (
            capability_payload.get("candidate_id") != candidate_id
            or capability_payload.get("configuration_sha256") != expected_digest
            or capability_payload.get("environment_rng_schedule_sha256")
            != ENVIRONMENT_RNG_SCHEDULE_SHA256
            or capability_payload.get("image_sha256") != QUALIFIED_IMAGE_SHA256
            or capability_payload.get("runtime_profile_sha256")
            != RUNTIME_PROFILE_SHA256
            or capability_payload.get("task_identity_sha256") != TASK_IDENTITY_SHA256
            or capability_payload.get("status") != "qualified"
        ):
            raise CausalGridDivergenceProbeError(
                f"capability receipt semantics drifted for {candidate_id}"
            )
        loaded.append(payload)
    return tuple(loaded), manifest


def _shared_step_keys(step_key: Any, count: int) -> tuple[Any, ...]:
    """Return the same exogenous environment key object for every q arm."""
    if type(count) is not int or count < 1:
        raise CausalGridDivergenceProbeError("shared-key lane count must be positive")
    return tuple(step_key for _ in range(count))


def _coupled_lane_resets(
    environment_reset: Any,
    reset_key: Any,
    environment_parameters: Any,
    lane_count: int,
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    results = tuple(
        environment_reset(shared_key, environment_parameters)
        for shared_key in _shared_step_keys(reset_key, lane_count)
    )
    return (
        tuple(result[0] for result in results),
        tuple(result[1] for result in results),
    )


def _coupled_lane_step(
    *,
    environment_step: Any,
    agent_step: Any,
    delay_for: Any,
    step_key: Any,
    environment_parameters: Any,
    environment_states: tuple[Any, ...],
    agent_states: tuple[Any, ...],
    actions: tuple[Any, ...],
    configurations: tuple[Any, ...],
) -> tuple[
    tuple[Any, ...],
    tuple[Any, ...],
    tuple[Any, ...],
    tuple[Any, ...],
    tuple[Any, ...],
    tuple[Any, ...],
]:
    lane_count = len(configurations)
    if not (
        lane_count >= 1
        and len(environment_states) == lane_count
        and len(agent_states) == lane_count
        and len(actions) == lane_count
    ):
        raise CausalGridDivergenceProbeError(
            "coupled lane state/configuration lengths disagree"
        )
    shared_step_keys = _shared_step_keys(step_key, lane_count)
    next_environment_states: list[Any] = []
    next_agent_states: list[Any] = []
    next_actions: list[Any] = []
    next_delays: list[Any] = []
    dones: list[Any] = []
    executed_actions = tuple(actions)
    for lane, configuration in enumerate(configurations):
        (
            next_observation,
            next_environment_state,
            reward,
            done,
            _info,
        ) = environment_step(
            shared_step_keys[lane],
            environment_states[lane],
            executed_actions[lane],
            environment_parameters,
        )
        next_agent_state, next_action, _diagnostics = agent_step(
            agent_states[lane],
            reward,
            next_observation,
            configuration,
        )
        next_environment_states.append(next_environment_state)
        next_agent_states.append(next_agent_state)
        next_actions.append(next_action)
        next_delays.append(delay_for(next_agent_state, configuration))
        dones.append(done)
    return (
        tuple(next_environment_states),
        tuple(next_agent_states),
        tuple(next_actions),
        tuple(next_delays),
        tuple(dones),
        executed_actions,
    )


def _child_runtime_payload(
    source_root: Path,
    configuration_payloads: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    """Run the bounded frozen workload; called only in an isolated child."""
    if any(
        name == "alberta_framework" or name.startswith("alberta_framework.")
        for name in sys.modules
    ):
        raise CausalGridDivergenceProbeError(
            "isolated child imported alberta_framework before binding frozen source"
        )
    sys.path.insert(0, source_root.as_posix())

    import jax  # pylint: disable=import-outside-toplevel
    import jax.numpy as jnp  # pylint: disable=import-outside-toplevel
    import jax.random as jr  # pylint: disable=import-outside-toplevel
    import jaxlib  # pylint: disable=import-outside-toplevel
    import numpy as np  # pylint: disable=import-outside-toplevel

    from alberta_framework.benchmarks import (  # pylint: disable=import-outside-toplevel
        causal_map_forager as causal,
    )
    from alberta_framework.benchmarks import forager  # pylint: disable=import-outside-toplevel

    for module, label in ((causal, "causal map"), (forager, "Forager runner")):
        module_file = getattr(module, "__file__", None)
        if not isinstance(module_file, str):
            raise CausalGridDivergenceProbeError(
                f"{label} module has no regular import path"
            )
        module_path = Path(module_file).resolve(strict=True)
        try:
            module_path.relative_to(source_root.resolve(strict=True))
        except ValueError as exc:
            raise CausalGridDivergenceProbeError(
                f"{label} module was not imported from the frozen source root"
            ) from exc
    if forager.environment_rng_schedule_sha256() != ENVIRONMENT_RNG_SCHEDULE_SHA256:
        raise CausalGridDivergenceProbeError(
            "frozen environment RNG schedule identity drifted"
        )
    if str(jax.config.jax_default_prng_impl) != "threefry2x32":
        raise CausalGridDivergenceProbeError(
            "probe requires the qualified threefry2x32 default PRNG"
        )
    if not bool(jax.config.jax_threefry_partitionable):
        raise CausalGridDivergenceProbeError(
            "probe requires the qualified partitionable Threefry mode"
        )
    if bool(jax.config.jax_enable_x64):
        raise CausalGridDivergenceProbeError("probe requires qualified x64-disabled JAX")
    jaxlib_version = getattr(jaxlib, "__version__", None)
    if (
        jax.__version__ != "0.9.0.1"
        or jaxlib_version != "0.9.0.1"
        or platform.python_version() != "3.12.3"
    ):
        raise CausalGridDivergenceProbeError(
            "probe runtime differs from the qualified JAX/Python versions"
        )
    if jax.default_backend() != "cpu" or {
        device.platform for device in jax.devices()
    } != {"cpu"}:
        raise CausalGridDivergenceProbeError("probe requires the qualified CPU backend")

    configurations = tuple(
        causal.CausalMapForagerConfig.from_dict(dict(payload["configuration"]))
        for payload in configuration_payloads
    )
    for parsed, payload in zip(configurations, configuration_payloads, strict=True):
        if parsed.to_dict() != payload["configuration"]:
            raise CausalGridDivergenceProbeError(
                "frozen causal-map configuration did not round-trip exactly"
            )

    environment, environment_parameters = (
        forager.ForagerEnvConfig.paper_field_of_view(aperture_size=9).make()
    )
    environment_key = jr.key(FIXED_SEED)
    environment_key, reset_key = jr.split(environment_key)
    observations, environment_states = _coupled_lane_resets(
        environment.reset,
        reset_key,
        environment_parameters,
        len(configurations),
    )
    agent_pairs = tuple(
        causal.causal_map_start(observation, configuration, FIXED_SEED)
        for observation, configuration in zip(
            observations,
            configurations,
            strict=True,
        )
    )
    agent_states = tuple(pair[0] for pair in agent_pairs)
    actions = tuple(pair[1] for pair in agent_pairs)
    channel_count = int(agent_states[0].respawn_exact_count.shape[0])
    if channel_count != DIAGNOSTIC_CHANNEL_COUNT or any(
        int(state.respawn_exact_count.shape[0]) != channel_count
        for state in agent_states
    ):
        raise CausalGridDivergenceProbeError(
            "frozen q arms disagree with the exact diagnostic channel count"
        )
    channel_indices = tuple(jnp.asarray(index, dtype=jnp.int32) for index in range(channel_count))

    def delays_for(state: Any, configuration: Any) -> Any:
        return jnp.stack(
            tuple(
                causal._estimated_respawn_delay(state, channel, configuration)
                for channel in channel_indices
            )
        ).astype(jnp.int32)

    initial_delays = jnp.stack(
        tuple(
            delays_for(state, configuration)
            for state, configuration in zip(agent_states, configurations, strict=True)
        )
    )
    delay_minimum = jnp.full_like(initial_delays, 2_147_483_647)
    delay_maximum = jnp.zeros_like(initial_delays)
    delay_sum = jnp.zeros_like(initial_delays)
    delay_change_count = jnp.zeros_like(initial_delays)
    done_seen = jnp.asarray(False)

    def scan_body(carry: Any, _: Any) -> tuple[Any, Any]:
        (
            current_environment_key,
            current_environment_states,
            current_agent_states,
            current_actions,
            current_minimum,
            current_maximum,
            current_sum,
            current_change_count,
            previous_delays,
            current_done_seen,
        ) = carry
        next_environment_key, step_key = jr.split(current_environment_key)
        (
            next_environment_states,
            next_agent_states,
            next_actions,
            next_delays,
            dones,
            executed_actions,
        ) = _coupled_lane_step(
            environment_step=environment.step,
            agent_step=causal.causal_map_step,
            delay_for=delays_for,
            step_key=step_key,
            environment_parameters=environment_parameters,
            environment_states=current_environment_states,
            agent_states=current_agent_states,
            actions=current_actions,
            configurations=configurations,
        )
        stacked_delays = jnp.stack(next_delays)
        next_carry = (
            next_environment_key,
            next_environment_states,
            next_agent_states,
            next_actions,
            jnp.minimum(current_minimum, stacked_delays),
            jnp.maximum(current_maximum, stacked_delays),
            current_sum + stacked_delays,
            current_change_count + (stacked_delays != previous_delays).astype(jnp.int32),
            stacked_delays,
            current_done_seen | jnp.any(jnp.stack(dones)),
        )
        return next_carry, jnp.stack(executed_actions).astype(jnp.int32)

    initial_carry = (
        environment_key,
        environment_states,
        agent_states,
        actions,
        delay_minimum,
        delay_maximum,
        delay_sum,
        delay_change_count,
        initial_delays,
        done_seen,
    )
    final_carry, action_matrix = jax.jit(
        lambda carry: jax.lax.scan(scan_body, carry, xs=None, length=FIXED_STEPS)
    )(initial_carry)
    jax.block_until_ready((final_carry, action_matrix))  # type: ignore[no-untyped-call]
    (
        _final_environment_key,
        _final_environment_states,
        final_agent_states,
        _next_actions,
        final_minimum,
        final_maximum,
        final_sum,
        final_change_count,
        final_delays,
        final_done_seen,
    ) = final_carry
    finite_flags = [
        jnp.all(jnp.isfinite(leaf))
        for leaf in jax.tree_util.tree_leaves(
            (_final_environment_states, final_agent_states)
        )
        if jnp.issubdtype(jnp.asarray(leaf).dtype, jnp.inexact)
    ]
    if finite_flags and not bool(np.asarray(jnp.all(jnp.stack(finite_flags)))):
        raise CausalGridDivergenceProbeError(
            "compiled probe produced a non-finite environment or agent state"
        )
    if bool(np.asarray(final_done_seen)):
        raise CausalGridDivergenceProbeError(
            "ForagaxTwoBiomeLarge-v1 unexpectedly terminated during the probe"
        )
    action_array = np.asarray(action_matrix)
    if (
        action_array.shape != (FIXED_STEPS, len(configurations))
        or action_array.dtype != np.int32
    ):
        raise CausalGridDivergenceProbeError("compiled action matrix contract drifted")
    if not bool(np.all((action_array >= 0) & (action_array < 4))):
        raise CausalGridDivergenceProbeError("compiled probe produced an invalid action")

    minimum_host = np.asarray(final_minimum, dtype=np.int64)
    maximum_host = np.asarray(final_maximum, dtype=np.int64)
    sum_host = np.asarray(final_sum, dtype=np.int64)
    changes_host = np.asarray(final_change_count, dtype=np.int64)
    final_delays_host = np.asarray(final_delays, dtype=np.int64)
    action_bytes_by_lane = tuple(
        action_array[:, lane].astype(np.uint8, copy=False).tobytes(order="C")
        for lane in range(len(configurations))
    )
    action_trees = tuple(
        _build_action_merkle_tree(action_bytes)
        for action_bytes in action_bytes_by_lane
    )
    candidate_records: list[dict[str, Any]] = []
    for lane, (candidate_id, quantile, configuration_sha256) in enumerate(
        _CANDIDATE_CONFIGURATIONS
    ):
        exact_counts = np.asarray(
            final_agent_states[lane].respawn_exact_count,
            dtype=np.int64,
        )
        channels: list[dict[str, Any]] = []
        for channel in range(channel_count):
            delay_sum_value = int(sum_host[lane, channel])
            channels.append(
                {
                    "channel_index": channel,
                    "exact_count_final": int(exact_counts[channel]),
                    "estimated_delay": {
                        "change_count": int(changes_host[lane, channel]),
                        "final": int(final_delays_host[lane, channel]),
                        "maximum": int(maximum_host[lane, channel]),
                        "mean_hex": float(delay_sum_value / FIXED_STEPS).hex(),
                        "minimum": int(minimum_host[lane, channel]),
                        "sample_count": FIXED_STEPS,
                        "sum": delay_sum_value,
                    },
                }
            )
        candidate_records.append(
            {
                "action_count": FIXED_STEPS,
                "action_trace_encoding": ACTION_MERKLE_ENCODING,
                "action_trace_sha256": _action_merkle_root_sha256(
                    action_trees[lane]
                ),
                "candidate_id": candidate_id,
                "configuration_sha256": configuration_sha256,
                "per_channel_diagnostics": channels,
                "respawn_safety_quantile": quantile,
            }
        )

    pairwise: list[dict[str, Any]] = []
    for left in range(len(configurations)):
        for right in range(left + 1, len(configurations)):
            differing = np.flatnonzero(action_array[:, left] != action_array[:, right])
            divergence_index = int(differing[0]) if differing.size else None
            left_bytes = action_bytes_by_lane[left]
            right_bytes = action_bytes_by_lane[right]
            if left_bytes[:divergence_index] != right_bytes[:divergence_index]:
                raise CausalGridDivergenceProbeError(
                    "first-divergence reduction produced an inconsistent common prefix"
                )
            pairwise.append(
                {
                    "first_action_divergence_step": (
                        None if divergence_index is None else divergence_index + 1
                    ),
                    "first_divergence_merkle_proof": (
                        None
                        if divergence_index is None
                        else _first_divergence_merkle_proof(
                            action_trees[left],
                            action_trees[right],
                            divergence_index=divergence_index,
                            left_action=left_bytes[divergence_index],
                            right_action=right_bytes[divergence_index],
                        )
                    ),
                    "left_candidate_id": _CANDIDATE_CONFIGURATIONS[left][0],
                    "right_candidate_id": _CANDIDATE_CONFIGURATIONS[right][0],
                }
            )
    # Do not retain a host action trace beyond the bounded hash/divergence reduction.
    del action_array
    return {
        "schema_version": CHILD_SCHEMA_VERSION,
        "candidates": candidate_records,
        "pairwise_action_divergence": pairwise,
        "runtime_observation": {
            "device_platforms": sorted({device.platform for device in jax.devices()}),
            "jax_backend": jax.default_backend(),
            "jax_version": jax.__version__,
            "jax_enable_x64": bool(jax.config.jax_enable_x64),
            "jaxlib_version": jaxlib_version,
            "numpy_version": np.__version__,
            "python_version": platform.python_version(),
            "threefry_partitionable": bool(jax.config.jax_threefry_partitionable),
        },
    }


def _child_arguments(
    source_root: Path,
    qualification_root: Path,
    probe_source_sha256: str,
) -> tuple[str, ...]:
    return (
        "_child",
        "--source-root",
        source_root.as_posix(),
        "--qualification-root",
        qualification_root.as_posix(),
        "--probe-source-sha256",
        probe_source_sha256,
    )


def _docker_mount_source(path: Path, *, label: str) -> str:
    resolved = path.resolve(strict=True)
    text = resolved.as_posix()
    if any(character in text for character in (",", "\n", "\r", "\x00")):
        raise CausalGridDivergenceProbeError(
            f"{label} cannot be represented safely in an OCI mount"
        )
    return text


def _run_bounded_command(
    command: tuple[str, ...],
    *,
    environment: Mapping[str, str],
    timeout_seconds: float,
    maximum_stdout_bytes: int,
    maximum_stderr_bytes: int,
) -> subprocess.CompletedProcess[bytes]:
    """Run a process while streaming both pipes into strictly bounded buffers."""
    if (
        timeout_seconds <= 0
        or maximum_stdout_bytes < 0
        or maximum_stderr_bytes < 0
    ):
        raise ValueError("bounded command limits must be nonnegative and finite")
    process = subprocess.Popen(
        command,
        env=dict(environment),
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    stdout = process.stdout
    stderr = process.stderr
    selector: selectors.BaseSelector | None = None
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    limits = {
        "stdout": maximum_stdout_bytes,
        "stderr": maximum_stderr_bytes,
    }
    deadline = time.monotonic() + timeout_seconds
    try:
        if stdout is None or stderr is None:
            raise CausalGridDivergenceProbeError("cannot create bounded child pipes")
        selector = selectors.DefaultSelector()
        selector.register(stdout, selectors.EVENT_READ, "stdout")
        selector.register(stderr, selectors.EVENT_READ, "stderr")
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(
                    command,
                    timeout_seconds,
                    output=bytes(buffers["stdout"]),
                    stderr=bytes(buffers["stderr"]),
                )
            for key, _mask in selector.select(min(remaining, 1.0)):
                label = str(key.data)
                buffer = buffers[label]
                maximum = limits[label]
                read_size = min(64 * 1024, maximum + 1 - len(buffer))
                chunk = os.read(key.fd, max(read_size, 1))
                if not chunk:
                    stream = stdout if label == "stdout" else stderr
                    selector.unregister(stream)
                    stream.close()
                    continue
                buffer.extend(chunk)
                if len(buffer) > maximum:
                    raise CausalGridDivergenceProbeError(
                        f"isolated probe {label} exceeded its hard byte bound"
                    )
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired(
                command,
                timeout_seconds,
                output=bytes(buffers["stdout"]),
                stderr=bytes(buffers["stderr"]),
            )
        returncode = process.wait(timeout=remaining)
        return subprocess.CompletedProcess(
            command,
            returncode,
            bytes(buffers["stdout"]),
            bytes(buffers["stderr"]),
        )
    except BaseException:
        if process.poll() is None:
            process.kill()
        try:
            process.wait(timeout=10)
        except (OSError, subprocess.SubprocessError):
            pass
        raise
    finally:
        if selector is not None:
            selector.close()
        if stdout is not None and not stdout.closed:
            stdout.close()
        if stderr is not None and not stderr.closed:
            stderr.close()


def _cleanup_named_container(
    runtime_path: Path,
    container_name: str,
    environment: Mapping[str, str],
) -> bool:
    try:
        cleanup = _run_bounded_command(
            (
                runtime_path.as_posix(),
                "rm",
                "--force",
                container_name,
            ),
            environment=environment,
            timeout_seconds=120,
            maximum_stdout_bytes=64 * 1024,
            maximum_stderr_bytes=64 * 1024,
        )
    except (OSError, subprocess.SubprocessError, CausalGridDivergenceProbeError):
        return False
    return cleanup.returncode == 0 or b"No such container" in cleanup.stderr


def _qualification_mount_relative_paths() -> tuple[str, ...]:
    paths = [
        "manifest.json",
        "sources/alberta/inventory.json",
        "sources/alberta/source.tar",
        "sources/alberta/snapshot-descriptor.json",
    ]
    for candidate_id, _quantile, _configuration_sha256 in _CANDIDATE_CONFIGURATIONS:
        paths.extend(
            (
                f"configurations/{candidate_id}/derived.json",
                f"configurations/{candidate_id}/original.json",
                f"receipts/{candidate_id}.json",
            )
        )
    return tuple(paths)


def _materialize_readable_qualification_mount(
    qualification_root: Path,
    destination_root: Path,
) -> dict[str, str]:
    """Copy only the exact pinned inputs into a nonsecret OCI-readable mirror."""
    try:
        destination_metadata = destination_root.lstat()
    except OSError as exc:
        raise CausalGridDivergenceProbeError(
            "temporary qualification mount root is missing"
        ) from exc
    if not stat.S_ISDIR(destination_metadata.st_mode) or destination_root.is_symlink():
        raise CausalGridDivergenceProbeError(
            "temporary qualification mount root must be a non-symlink directory"
        )
    destination_root.chmod(0o755)
    digests: dict[str, str] = {}
    for relative_text in _qualification_mount_relative_paths():
        relative = Path(relative_text)
        if relative.is_absolute() or ".." in relative.parts:
            raise CausalGridDivergenceProbeError(
                "temporary qualification mount path is unsafe"
            )
        maximum = (
            MAXIMUM_SOURCE_BYTES
            if relative_text == "sources/alberta/source.tar"
            else MAXIMUM_JSON_BYTES
        )
        raw = _read_stable_regular_file(
            qualification_root / relative,
            label=f"qualification mount input {relative_text}",
            maximum=maximum,
        )
        destination = destination_root / relative
        destination.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
        for parent in (destination.parent, *destination.parent.parents):
            if parent == destination_root.parent:
                break
            parent.chmod(0o755)
            if parent == destination_root:
                break
        descriptor = os.open(
            destination,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            mode=0o444,
        )
        try:
            view = memoryview(raw)
            offset = 0
            while offset < len(view):
                written = os.write(descriptor, view[offset:])
                if written < 1:
                    raise OSError("short write while mirroring qualification input")
                offset += written
            os.fchmod(descriptor, 0o444)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        digests[relative_text] = _sha256(raw)
    return digests


def _materialize_readonly_probe_mount(
    destination: Path,
    *,
    expected_probe_source_sha256: str,
) -> None:
    expected = _require_sha256(
        expected_probe_source_sha256,
        label="expected probe source digest",
    )
    raw = _read_probe_source(expected_sha256=expected)
    if destination.exists() or destination.is_symlink():
        raise CausalGridDivergenceProbeError(
            "private probe-source mount unexpectedly exists"
        )
    descriptor = os.open(
        destination,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        mode=0o444,
    )
    try:
        view = memoryview(raw)
        offset = 0
        while offset < len(view):
            written = os.write(descriptor, view[offset:])
            if written < 1:
                raise OSError("short write while snapshotting probe source")
            offset += written
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    copied = _read_stable_regular_file(
        destination,
        label="private probe-source mount",
        maximum=MAXIMUM_JSON_BYTES,
    )
    if copied != raw or _sha256(copied) != expected:
        raise CausalGridDivergenceProbeError(
            "private probe-source mount differs from the exact source snapshot"
        )


def _run_child_with_qualification_mount(
    qualification_root: Path,
    probe_path: Path,
    *,
    oci_runtime: Path,
    expected_probe_source_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    environment = dict(os.environ)
    environment.update(
        {
            "JAX_ENABLE_COMPILATION_CACHE": "false",
            "JAX_PLATFORM_NAME": "cpu",
            "JAX_PLATFORMS": "cpu",
            "PYTHONHASHSEED": "0",
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": "",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    expected_probe_source_sha256 = _require_sha256(
        expected_probe_source_sha256,
        label="expected probe source digest",
    )
    probe_raw_before = _read_stable_regular_file(
        probe_path,
        label="private probe-source mount",
        maximum=MAXIMUM_JSON_BYTES,
    )
    if _sha256(probe_raw_before) != expected_probe_source_sha256:
        raise CausalGridDivergenceProbeError(
            "private probe-source mount differs from the expected source digest"
        )
    runtime_path = oci_runtime.resolve(strict=True)
    runtime_raw = _read_stable_regular_file(
        runtime_path,
        label="OCI runtime executable",
        maximum=512 * 1024 * 1024,
    )
    qualification_text = _docker_mount_source(
        qualification_root,
        label="qualification root",
    )
    probe_text = _docker_mount_source(probe_path, label="probe source")
    container_name = f"alberta-causal-q-probe-{secrets.token_hex(12)}"
    tmpfs_spec = (
        "--tmpfs=/run/alberta:rw,noexec,nosuid,nodev,size=8g,"
        "uid=65532,gid=65532,mode=0700"
    )
    command = (
        runtime_path.as_posix(),
        "run",
        "--rm",
        f"--name={container_name}",
        "--pull=never",
        "--network=none",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--user=65532:65532",
        "--cpus=4.0",
        "--memory=16g",
        "--memory-swap=16g",
        "--pids-limit=512",
        tmpfs_spec,
        "--env=HOME=/run/alberta",
        "--env=JAX_ENABLE_COMPILATION_CACHE=false",
        "--env=JAX_PLATFORM_NAME=cpu",
        "--env=JAX_PLATFORMS=cpu",
        "--env=JAX_THREEFRY_PARTITIONABLE=true",
        "--env=LANG=C.UTF-8",
        "--env=LC_ALL=C.UTF-8",
        "--env=NVIDIA_VISIBLE_DEVICES=void",
        "--env=PYTHONHASHSEED=0",
        "--env=PYTHONNOUSERSITE=1",
        "--env=PYTHONPATH=",
        "--env=PYTHONDONTWRITEBYTECODE=1",
        "--env=TMPDIR=/run/alberta",
        "--env=TZ=UTC",
        "--mount=type=bind,source="
        f"{qualification_text},destination=/inputs/qualification,readonly",
        "--mount=type=bind,source="
        f"{probe_text},destination=/harness/causal_q_grid_probe.py,readonly",
        "--workdir=/run/alberta",
        f"sha256:{QUALIFIED_IMAGE_SHA256}",
        "/opt/alberta-runtime/bin/python",
        "-I",
        "-B",
        "/harness/causal_q_grid_probe.py",
        *_child_arguments(
            Path("/run/alberta/source"),
            Path("/inputs/qualification"),
            expected_probe_source_sha256,
        ),
    )
    execution_envelope = {
        "image_sha256": QUALIFIED_IMAGE_SHA256,
        "kind": "sha256_pinned_qualified_oci",
        "network": "none",
        "oci_runtime_executable_sha256": _sha256(runtime_raw),
        "probe_mount": "private_exact_readonly_snapshot_v1",
        "qualification_mount": "minimal_exact_readable_snapshot_v1",
        "qualified_image_executed": True,
        "root_filesystem": "read_only",
        "source_mount": "pinned_archive_extracted_in_private_tmpfs_v1",
    }
    try:
        completed = _run_bounded_command(
            command,
            environment=environment,
            timeout_seconds=60 * 60,
            maximum_stdout_bytes=MAXIMUM_JSON_BYTES,
            maximum_stderr_bytes=MAXIMUM_STDERR_BYTES,
        )
    except BaseException as exc:
        if not _cleanup_named_container(runtime_path, container_name, environment):
            raise CausalGridDivergenceProbeError(
                "isolated probe failed and named-container cleanup was not confirmed"
            ) from exc
        if isinstance(exc, subprocess.TimeoutExpired):
            raise CausalGridDivergenceProbeError(
                "isolated probe timed out; named-container cleanup completed"
            ) from exc
        raise
    if completed.returncode != 0 or completed.stderr:
        if not _cleanup_named_container(runtime_path, container_name, environment):
            raise CausalGridDivergenceProbeError(
                "isolated probe completed unsuccessfully and named-container cleanup "
                "was not confirmed"
            )
        raise CausalGridDivergenceProbeError(
            "isolated frozen-source probe failed with status "
            f"{completed.returncode}; named-container cleanup completed"
        )
    probe_raw_after = _read_stable_regular_file(
        probe_path,
        label="private probe-source mount",
        maximum=MAXIMUM_JSON_BYTES,
    )
    if (
        probe_raw_after != probe_raw_before
        or _sha256(probe_raw_after) != expected_probe_source_sha256
    ):
        raise CausalGridDivergenceProbeError(
            "causal-grid divergence probe source bytes changed across OCI execution"
        )
    payload = _decode_strict_json(completed.stdout, label="isolated probe output")
    if completed.stdout != _canonical_json_bytes(payload):
        raise CausalGridDivergenceProbeError("isolated probe output is not canonical JSON")
    return payload, execution_envelope


def _run_child(
    source_root: Path,
    qualification_root: Path,
    *,
    oci_runtime: Path,
    expected_probe_source_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run only private snapshots while leaving every frozen input unchanged."""
    _verify_source_identity(
        source_root,
        qualification_root / "sources" / "alberta" / "inventory.json",
    )
    with tempfile.TemporaryDirectory(
        prefix="alberta-causal-q-qualification-mount-"
    ) as temporary:
        private_root = Path(temporary)
        private_root.chmod(0o755)
        qualification_mount = private_root / "qualification"
        qualification_mount.mkdir(mode=0o700)
        probe_mount = private_root / "causal_q_grid_probe.py"
        _materialize_readable_qualification_mount(
            qualification_root,
            qualification_mount,
        )
        _materialize_readonly_probe_mount(
            probe_mount,
            expected_probe_source_sha256=expected_probe_source_sha256,
        )
        return _run_child_with_qualification_mount(
            qualification_mount,
            probe_mount,
            oci_runtime=oci_runtime,
            expected_probe_source_sha256=expected_probe_source_sha256,
        )


def _sha256_bytes(value: Any, *, label: str) -> bytes:
    return bytes.fromhex(_require_sha256(value, label=label))


def _validate_first_divergence_merkle_proof(
    proof: Any,
    *,
    divergence_index: int,
    left_root_sha256: str,
    right_root_sha256: str,
) -> None:
    if type(proof) is not dict or set(proof) != {
        "divergence_index",
        "encoding",
        "left_action",
        "levels",
        "right_action",
        "tree_leaf_count",
    }:
        raise CausalGridDivergenceProbeError(
            "first-divergence Merkle proof fields drifted"
        )
    if (
        type(proof["divergence_index"]) is not int
        or proof["divergence_index"] != divergence_index
        or proof["encoding"] != ACTION_MERKLE_ENCODING
        or proof["tree_leaf_count"] != ACTION_MERKLE_TREE_LEAF_COUNT
    ):
        raise CausalGridDivergenceProbeError(
            "first-divergence Merkle proof identity drifted"
        )
    left_action = _require_int(
        proof["left_action"],
        label="left divergence action",
        minimum=0,
        maximum=3,
    )
    right_action = _require_int(
        proof["right_action"],
        label="right divergence action",
        minimum=0,
        maximum=3,
    )
    if left_action == right_action:
        raise CausalGridDivergenceProbeError(
            "first-divergence Merkle proof actions are equal"
        )
    levels = proof["levels"]
    if type(levels) is not list or len(levels) != ACTION_MERKLE_ROOT_LEVEL:
        raise CausalGridDivergenceProbeError(
            "first-divergence Merkle proof level count drifted"
        )
    left_current: bytes | None = None
    right_current: bytes | None = None
    left_root = _sha256_bytes(left_root_sha256, label="left action Merkle root")
    right_root = _sha256_bytes(right_root_sha256, label="right action Merkle root")
    if left_root == right_root:
        raise CausalGridDivergenceProbeError(
            "divergent pair has equal action Merkle roots"
        )
    for offset, record in enumerate(levels):
        level = ACTION_MERKLE_ROOT_LEVEL - offset
        if type(record) is not dict or set(record) != {
            "followed_child",
            "level",
            "left_arm_left_child_sha256",
            "left_arm_right_child_sha256",
            "right_arm_left_child_sha256",
            "right_arm_right_child_sha256",
        }:
            raise CausalGridDivergenceProbeError(
                "first-divergence Merkle level fields drifted"
            )
        expected_child = (
            "right" if divergence_index & (1 << (level - 1)) else "left"
        )
        if (
            type(record["level"]) is not int
            or record["level"] != level
            or record["followed_child"] != expected_child
        ):
            raise CausalGridDivergenceProbeError(
                "first-divergence Merkle path is noncanonical"
            )
        left_left = _sha256_bytes(
            record["left_arm_left_child_sha256"],
            label="left-arm left-child digest",
        )
        left_right = _sha256_bytes(
            record["left_arm_right_child_sha256"],
            label="left-arm right-child digest",
        )
        right_left = _sha256_bytes(
            record["right_arm_left_child_sha256"],
            label="right-arm left-child digest",
        )
        right_right = _sha256_bytes(
            record["right_arm_right_child_sha256"],
            label="right-arm right-child digest",
        )
        left_parent = _action_parent_digest(level, left_left, left_right)
        right_parent = _action_parent_digest(level, right_left, right_right)
        if offset == 0:
            if (
                _action_root_digest(left_parent) != left_root
                or _action_root_digest(right_parent) != right_root
            ):
                raise CausalGridDivergenceProbeError(
                    "first-divergence proof does not reconstruct candidate roots"
                )
        elif left_parent != left_current or right_parent != right_current:
            raise CausalGridDivergenceProbeError(
                "first-divergence Merkle descent is disconnected"
            )
        if expected_child == "right":
            if left_left != right_left:
                raise CausalGridDivergenceProbeError(
                    "first-divergence proof has an unequal earlier subtree"
                )
            left_current = left_right
            right_current = right_right
        else:
            left_current = left_left
            right_current = right_left
        if left_current == right_current:
            raise CausalGridDivergenceProbeError(
                "claimed first-divergence path becomes equal before its leaf"
            )
    if (
        left_current != _action_leaf_digest(divergence_index, left_action)
        or right_current != _action_leaf_digest(divergence_index, right_action)
    ):
        raise CausalGridDivergenceProbeError(
            "first-divergence Merkle leaves do not match the declared actions"
        )


def _validate_child_payload(payload: Mapping[str, Any]) -> bool:
    if type(payload) is not dict or set(payload) != {
        "schema_version",
        "candidates",
        "pairwise_action_divergence",
        "runtime_observation",
    }:
        raise CausalGridDivergenceProbeError("isolated probe output fields drifted")
    if payload["schema_version"] != CHILD_SCHEMA_VERSION:
        raise CausalGridDivergenceProbeError("isolated probe output schema drifted")
    runtime = payload["runtime_observation"]
    if type(runtime) is not dict or set(runtime) != {
        "device_platforms",
        "jax_backend",
        "jax_enable_x64",
        "jax_version",
        "jaxlib_version",
        "numpy_version",
        "python_version",
        "threefry_partitionable",
    }:
        raise CausalGridDivergenceProbeError("runtime observation fields drifted")
    if (
        runtime["jax_backend"] != "cpu"
        or runtime["device_platforms"] != ["cpu"]
        or runtime["jax_enable_x64"] is not False
        or runtime["jax_version"] != "0.9.0.1"
        or runtime["jaxlib_version"] != "0.9.0.1"
        or runtime["python_version"] != "3.12.3"
        or runtime["threefry_partitionable"] is not True
        or any(
            type(runtime[name]) is not str or not runtime[name]
            for name in ("numpy_version",)
        )
    ):
        raise CausalGridDivergenceProbeError(
            "probe did not execute on the required CPU runtime"
        )
    candidates = payload["candidates"]
    if type(candidates) is not list or len(candidates) != len(_CANDIDATE_CONFIGURATIONS):
        raise CausalGridDivergenceProbeError("isolated probe candidate block drifted")
    for index, (record, expected) in enumerate(
        zip(candidates, _CANDIDATE_CONFIGURATIONS, strict=True)
    ):
        if type(record) is not dict or set(record) != {
            "action_count",
            "action_trace_encoding",
            "action_trace_sha256",
            "candidate_id",
            "configuration_sha256",
            "per_channel_diagnostics",
            "respawn_safety_quantile",
        }:
            raise CausalGridDivergenceProbeError(
                f"isolated probe candidate[{index}] fields drifted"
            )
        candidate_id, quantile, configuration_sha256 = expected
        if (
            record["candidate_id"] != candidate_id
            or record["configuration_sha256"] != configuration_sha256
            or record["respawn_safety_quantile"] != quantile
            or record["action_count"] != FIXED_STEPS
            or record["action_trace_encoding"] != ACTION_MERKLE_ENCODING
        ):
            raise CausalGridDivergenceProbeError(
                f"isolated probe candidate[{index}] identity drifted"
            )
        _require_sha256(
            record["action_trace_sha256"],
            label=f"candidate[{index}] action trace digest",
        )
        channels = record["per_channel_diagnostics"]
        if type(channels) is not list or len(channels) != DIAGNOSTIC_CHANNEL_COUNT:
            raise CausalGridDivergenceProbeError(
                f"isolated probe candidate[{index}] channel count drifted"
            )
        for channel_index, channel in enumerate(channels):
            if type(channel) is not dict or set(channel) != {
                "channel_index",
                "exact_count_final",
                "estimated_delay",
            }:
                raise CausalGridDivergenceProbeError("channel diagnostic fields drifted")
            if (
                type(channel["channel_index"]) is not int
                or channel["channel_index"] != channel_index
            ):
                raise CausalGridDivergenceProbeError("channel diagnostic order drifted")
            _require_int(
                channel["exact_count_final"],
                label="exact_count_final",
                minimum=0,
                maximum=FIXED_STEPS * 81,
            )
            delay = channel["estimated_delay"]
            if type(delay) is not dict or set(delay) != {
                "change_count",
                "final",
                "maximum",
                "mean_hex",
                "minimum",
                "sample_count",
                "sum",
            }:
                raise CausalGridDivergenceProbeError(
                    "estimated-delay aggregate fields drifted"
                )
            minimum = _require_int(
                delay["minimum"], label="delay minimum", minimum=1, maximum=4096
            )
            maximum = _require_int(
                delay["maximum"], label="delay maximum", minimum=1, maximum=4096
            )
            final = _require_int(
                delay["final"], label="delay final", minimum=1, maximum=4096
            )
            total = _require_int(
                delay["sum"],
                label="delay sum",
                minimum=FIXED_STEPS,
                maximum=FIXED_STEPS * 4096,
            )
            _require_int(
                delay["change_count"],
                label="delay change count",
                minimum=0,
                maximum=FIXED_STEPS,
            )
            if (
                minimum > maximum
                or not minimum <= final <= maximum
                or not minimum * FIXED_STEPS <= total <= maximum * FIXED_STEPS
                or delay["sample_count"] != FIXED_STEPS
                or type(delay["mean_hex"]) is not str
            ):
                raise CausalGridDivergenceProbeError(
                    "estimated-delay aggregate values are inconsistent"
                )
            try:
                mean = float.fromhex(delay["mean_hex"])
            except (TypeError, ValueError, OverflowError) as exc:
                raise CausalGridDivergenceProbeError(
                    "estimated-delay mean is not a hexadecimal float"
                ) from exc
            if (
                not math.isfinite(mean)
                or mean.hex() != delay["mean_hex"]
                or mean != total / FIXED_STEPS
            ):
                raise CausalGridDivergenceProbeError(
                    "estimated-delay mean does not match its exact aggregate"
                )
    pairwise = payload["pairwise_action_divergence"]
    expected_pairs = len(_CANDIDATE_CONFIGURATIONS) * (
        len(_CANDIDATE_CONFIGURATIONS) - 1
    ) // 2
    if type(pairwise) is not list or len(pairwise) != expected_pairs:
        raise CausalGridDivergenceProbeError("pairwise divergence block drifted")
    expected_pair_ids = [
        (left[0], right[0])
        for left_index, left in enumerate(_CANDIDATE_CONFIGURATIONS)
        for right in _CANDIDATE_CONFIGURATIONS[left_index + 1 :]
    ]
    candidate_by_id = {
        candidate["candidate_id"]: candidate for candidate in candidates
    }
    any_divergence = False
    for record, (left_id, right_id) in zip(pairwise, expected_pair_ids, strict=True):
        if type(record) is not dict or set(record) != {
            "first_action_divergence_step",
            "first_divergence_merkle_proof",
            "left_candidate_id",
            "right_candidate_id",
        }:
            raise CausalGridDivergenceProbeError("pairwise divergence fields drifted")
        divergence = record["first_action_divergence_step"]
        if (
            record["left_candidate_id"] != left_id
            or record["right_candidate_id"] != right_id
            or (
                divergence is not None
                and (
                    type(divergence) is not int
                    or not 1 <= divergence <= FIXED_STEPS
                )
            )
        ):
            raise CausalGridDivergenceProbeError("pairwise divergence identity drifted")
        proof = record["first_divergence_merkle_proof"]
        left_root = candidate_by_id[left_id]["action_trace_sha256"]
        right_root = candidate_by_id[right_id]["action_trace_sha256"]
        if divergence is None:
            if proof is not None:
                raise CausalGridDivergenceProbeError(
                    "nondivergent pair unexpectedly includes a divergence witness"
                )
            if left_root != right_root:
                raise CausalGridDivergenceProbeError(
                    "nondivergent pair has unequal action Merkle roots"
                )
        else:
            _validate_first_divergence_merkle_proof(
                proof,
                divergence_index=divergence - 1,
                left_root_sha256=left_root,
                right_root_sha256=right_root,
            )
        any_divergence = any_divergence or divergence is not None
    return any_divergence


def _assemble_receipt(
    child_payload: dict[str, Any],
    *,
    probe_source_sha256: str,
    execution_envelope: Mapping[str, Any],
) -> tuple[dict[str, Any], bool]:
    any_divergence = _validate_child_payload(child_payload)
    if type(execution_envelope) is not dict or set(execution_envelope) != {
        "image_sha256",
        "kind",
        "network",
        "oci_runtime_executable_sha256",
        "probe_mount",
        "qualification_mount",
        "qualified_image_executed",
        "root_filesystem",
        "source_mount",
    }:
        raise CausalGridDivergenceProbeError("execution envelope fields drifted")
    if (
        execution_envelope["image_sha256"] != QUALIFIED_IMAGE_SHA256
        or execution_envelope["kind"] != "sha256_pinned_qualified_oci"
        or execution_envelope["network"] != "none"
        or execution_envelope["probe_mount"]
        != "private_exact_readonly_snapshot_v1"
        or execution_envelope["qualification_mount"]
        != "minimal_exact_readable_snapshot_v1"
        or execution_envelope["qualified_image_executed"] is not True
        or execution_envelope["root_filesystem"] != "read_only"
        or execution_envelope["source_mount"]
        != "pinned_archive_extracted_in_private_tmpfs_v1"
    ):
        raise CausalGridDivergenceProbeError(
            "probe did not execute in the required qualified OCI envelope"
        )
    _require_sha256(
        execution_envelope["oci_runtime_executable_sha256"],
        label="OCI runtime executable digest",
    )
    pair_count = len(child_payload["pairwise_action_divergence"])
    divergent_pair_count = sum(
        record["first_action_divergence_step"] is not None
        for record in child_payload["pairwise_action_divergence"]
    )
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "classification": CLASSIFICATION,
        "promotion_authorized": False,
        "performance_claim": False,
        "protocol_relationship": {
            "engineering_diagnostic_only": True,
            "frozen_v1_protocol_modified": False,
            "retroactive_evidence_gate": False,
            "valid_rejection_effect": "may_motivate_abandoning_v1_only",
        },
        "status": (
            "action_divergence_observed"
            if any_divergence
            else "rejected_no_action_divergence"
        ),
        "seed": FIXED_SEED,
        "seed_class": SEED_CLASS,
        "steps": FIXED_STEPS,
        "fixed_exploration_probability": FIXED_EXPLORATION_PROBABILITY,
        "task": {
            "aperture_size": 9,
            "environment_id": "ForagaxTwoBiomeLarge-v1",
            "observation_type": "color",
        },
        "environment_rng": {
            "agent_environment_key_shared": False,
            "environment_rng_schedule_sha256": ENVIRONMENT_RNG_SCHEDULE_SHA256,
            "pairing": (
                "identical reset key and identical per-transition environment key "
                "for every q arm"
            ),
        },
        "divergence_summary": {
            "all_pairs_diverged": divergent_pair_count == pair_count,
            "any_pair_diverged": any_divergence,
            "divergent_pair_count": divergent_pair_count,
            "gate": "at_least_one_pair_must_diverge",
            "pair_count": pair_count,
        },
        "action_boundary": {
            "action_arrays_emitted": False,
            "action_arrays_persisted": False,
            "commitment": ACTION_MERKLE_ENCODING,
            "first_divergence_proof": "canonical_paired_merkle_descent_v1",
            "maximum_disclosed_action_scalars": 2 * divergent_pair_count,
            "scalar_domain": [0, 1, 2, 3],
        },
        "execution_envelope": dict(execution_envelope),
        "runtime_observation": child_payload["runtime_observation"],
        "frozen_inputs": {
            "candidate_source_inventory_sha256": SOURCE_INVENTORY_SHA256,
            "executor_inventory_sha256": EXECUTOR_INVENTORY_SHA256,
            "matched_horizon": MATCHED_HORIZON,
            "open_protocol_sha256": OPEN_PROTOCOL_SHA256,
            "qualification_manifest_sha256": QUALIFICATION_MANIFEST_SHA256,
            "qualified_image_lock_sha256": QUALIFIED_IMAGE_SHA256,
            "runtime_profile_sha256": RUNTIME_PROFILE_SHA256,
            "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
            "snapshot_descriptor_sha256": SNAPSHOT_DESCRIPTOR_SHA256,
            "task_identity_sha256": TASK_IDENTITY_SHA256,
        },
        "probe_source_sha256": _require_sha256(
            probe_source_sha256, label="probe source digest"
        ),
        "reward_boundary": {
            "reward_arrays_emitted": False,
            "reward_arrays_persisted": False,
            "reward_arrays_read_by_host": False,
            "reward_scalar_use": "online_policy_update_only",
            "scoring_performed": False,
        },
        "candidates": child_payload["candidates"],
        "pairwise_action_divergence": child_payload["pairwise_action_divergence"],
    }
    # Round-trip under the exact persisted encoding before returning a result.
    if _decode_strict_json(_canonical_json_bytes(receipt), label="probe receipt") != receipt:
        raise CausalGridDivergenceProbeError("probe receipt is not canonically round-trippable")
    return receipt, any_divergence


def _atomic_rename_directory_noreplace(
    parent_descriptor: int,
    source_name: str,
    destination_name: str,
) -> None:
    for value, label in (
        (source_name, "staging directory name"),
        (destination_name, "destination directory name"),
    ):
        if (
            type(value) is not str
            or not value
            or value in {".", ".."}
            or "/" in value
            or "\x00" in value
        ):
            raise CausalGridDivergenceProbeError(f"unsafe {label}")
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = libc.renameat2
    except (AttributeError, OSError) as exc:
        raise CausalGridDivergenceProbeError(
            "atomic no-replace directory publication is unavailable"
        ) from exc
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    rename_noreplace = 1
    result = renameat2(
        parent_descriptor,
        os.fsencode(source_name),
        parent_descriptor,
        os.fsencode(destination_name),
        rename_noreplace,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in (errno.EEXIST, errno.ENOTEMPTY):
        raise CausalGridDivergenceProbeError(
            "development output root already exists; probe outputs are immutable"
        )
    raise CausalGridDivergenceProbeError(
        "cannot atomically publish the development receipt"
    ) from OSError(error_number, os.strerror(error_number))


def _directory_descriptor_matches_path(descriptor: int, path: Path) -> bool:
    try:
        descriptor_metadata = os.fstat(descriptor)
        path_metadata = path.stat(follow_symlinks=False)
    except OSError:
        return False
    return (
        stat.S_ISDIR(descriptor_metadata.st_mode)
        and stat.S_ISDIR(path_metadata.st_mode)
        and descriptor_metadata.st_dev == path_metadata.st_dev
        and descriptor_metadata.st_ino == path_metadata.st_ino
    )


def _directory_entry_matches_descriptor(
    parent_descriptor: int,
    entry_name: str,
    held_descriptor: int,
) -> bool:
    try:
        entry_metadata = os.stat(
            entry_name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        held_metadata = os.fstat(held_descriptor)
    except OSError:
        return False
    return (
        stat.S_ISDIR(entry_metadata.st_mode)
        and stat.S_ISDIR(held_metadata.st_mode)
        and entry_metadata.st_dev == held_metadata.st_dev
        and entry_metadata.st_ino == held_metadata.st_ino
    )


def _create_staging_directory(
    parent_descriptor: int,
    *,
    output_name: str,
) -> tuple[str, int]:
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    for _attempt in range(32):
        staging_name = f".{output_name}.staging-{secrets.token_hex(12)}"
        try:
            os.mkdir(staging_name, mode=0o700, dir_fd=parent_descriptor)
        except FileExistsError:
            continue
        try:
            staging_descriptor = os.open(
                staging_name,
                directory_flags,
                dir_fd=parent_descriptor,
            )
        except BaseException:
            os.rmdir(staging_name, dir_fd=parent_descriptor)
            raise
        return staging_name, staging_descriptor
    raise CausalGridDivergenceProbeError(
        "cannot allocate a unique sibling staging directory"
    )


def _write_regular_file_at(
    directory_descriptor: int,
    name: str,
    raw: bytes,
) -> None:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(name, flags, mode=0o600, dir_fd=directory_descriptor)
    try:
        view = memoryview(raw)
        offset = 0
        while offset < len(view):
            written = os.write(descriptor, view[offset:])
            if written < 1:
                raise OSError("short write while sealing receipt")
            offset += written
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o400)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_regular_file_at(
    directory_descriptor: int,
    name: str,
    *,
    label: str,
    maximum: int,
) -> bytes:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    descriptor = os.open(name, flags, dir_fd=directory_descriptor)
    try:
        return _read_stable_regular_descriptor(
            descriptor,
            label=label,
            maximum=maximum,
        )
    finally:
        os.close(descriptor)


def _remove_staging_directory(
    parent_descriptor: int,
    staging_descriptor: int,
    staging_name: str,
) -> None:
    for name in ("receipt.json", "receipt.json.sha256"):
        try:
            os.unlink(name, dir_fd=staging_descriptor)
        except FileNotFoundError:
            pass
    os.rmdir(staging_name, dir_fd=parent_descriptor)


def _write_receipt(output_root: Path, receipt: Mapping[str, Any]) -> Path:
    if (
        not output_root.is_absolute()
        or output_root != output_root.resolve(strict=False)
        or output_root.exists()
        or output_root.is_symlink()
    ):
        raise CausalGridDivergenceProbeError(
            "development output root must be canonical, absent, and write-once"
        )
    output_name = output_root.name
    if output_name in {"", ".", ".."} or any(
        character in output_name for character in ("/", "\x00", "\n", "\r")
    ):
        raise CausalGridDivergenceProbeError("development output name is unsafe")
    raw = _canonical_json_bytes(dict(receipt))
    if len(raw) > MAXIMUM_JSON_BYTES:
        raise CausalGridDivergenceProbeError(
            "development receipt exceeds its byte bound before publication"
        )
    digest_raw = f"{_sha256(raw)}\n".encode("ascii")
    parent_descriptor = -1
    staging_descriptor = -1
    published_descriptor = -1
    staging_name: str | None = None
    published = False
    try:
        output_root.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        parent_descriptor = os.open(
            output_root.parent,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        if not _directory_descriptor_matches_path(
            parent_descriptor,
            output_root.parent,
        ):
            raise CausalGridDivergenceProbeError(
                "development output parent changed while being opened"
            )
        staging_name, staging_descriptor = _create_staging_directory(
            parent_descriptor,
            output_name=output_name,
        )
        _write_regular_file_at(staging_descriptor, "receipt.json", raw)
        _write_regular_file_at(
            staging_descriptor,
            "receipt.json.sha256",
            digest_raw,
        )
        os.fsync(staging_descriptor)
        if not _directory_descriptor_matches_path(
            parent_descriptor,
            output_root.parent,
        ):
            raise CausalGridDivergenceProbeError(
                "development output parent changed before publication"
            )
        _atomic_rename_directory_noreplace(
            parent_descriptor,
            staging_name,
            output_name,
        )
        published = True
        os.fsync(parent_descriptor)
        if not _directory_descriptor_matches_path(
            parent_descriptor,
            output_root.parent,
        ):
            raise CausalGridDivergenceProbeError(
                "development output parent changed after publication"
            )
        published_descriptor = os.open(
            output_name,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_descriptor,
        )
        if not _directory_entry_matches_descriptor(
            parent_descriptor,
            output_name,
            staging_descriptor,
        ) or not _directory_entry_matches_descriptor(
            parent_descriptor,
            output_name,
            published_descriptor,
        ):
            raise CausalGridDivergenceProbeError(
                "published destination entry does not match the held staging inode"
            )
        persisted = _read_regular_file_at(
            published_descriptor,
            "receipt.json",
            label="published development receipt",
            maximum=MAXIMUM_JSON_BYTES,
        )
        persisted_digest = _read_regular_file_at(
            published_descriptor,
            "receipt.json.sha256",
            label="published development receipt digest",
            maximum=65,
        )
        if persisted != raw or persisted_digest != digest_raw:
            raise CausalGridDivergenceProbeError(
                "published development receipt differs from the sealed bytes"
            )
        if not _directory_entry_matches_descriptor(
            parent_descriptor,
            output_name,
            published_descriptor,
        ):
            raise CausalGridDivergenceProbeError(
                "published destination entry changed during replay verification"
            )
    except BaseException as exc:
        if not published and (
            parent_descriptor >= 0
            and staging_descriptor >= 0
            and staging_name is not None
        ):
            try:
                _remove_staging_directory(
                    parent_descriptor,
                    staging_descriptor,
                    staging_name,
                )
            except OSError:
                pass
        if published:
            raise ReceiptPublishedButUncertainError(
                f"receipt was published at {output_root} but durability or replay "
                "verification is uncertain; do not reuse this output root"
            ) from exc
        if isinstance(exc, CausalGridDivergenceProbeError):
            raise
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        raise CausalGridDivergenceProbeError("cannot seal development receipt") from exc
    finally:
        if published_descriptor >= 0:
            os.close(published_descriptor)
        if staging_descriptor >= 0:
            os.close(staging_descriptor)
        if parent_descriptor >= 0:
            os.close(parent_descriptor)
    return output_root / "receipt.json"


def load_receipt(output_root: Path) -> dict[str, Any]:
    """Strictly replay one published receipt and its detached digest."""
    try:
        before = output_root.lstat()
    except OSError as exc:
        raise CausalGridDivergenceProbeError(
            "published development output root is missing"
        ) from exc
    if not stat.S_ISDIR(before.st_mode) or output_root.is_symlink():
        raise CausalGridDivergenceProbeError(
            "published development output root must be a non-symlink directory"
        )
    try:
        entry_names = sorted(path.name for path in output_root.iterdir())
    except OSError as exc:
        raise CausalGridDivergenceProbeError(
            "cannot enumerate the published development output root"
        ) from exc
    if entry_names != ["receipt.json", "receipt.json.sha256"]:
        raise CausalGridDivergenceProbeError(
            "published development output must contain exactly the receipt and sidecar"
        )
    raw = _read_stable_regular_file(
        output_root / "receipt.json",
        label="published development receipt",
        maximum=MAXIMUM_JSON_BYTES,
    )
    sidecar = _read_stable_regular_file(
        output_root / "receipt.json.sha256",
        label="published development receipt digest",
        maximum=65,
    )
    if sidecar != f"{_sha256(raw)}\n".encode("ascii"):
        raise CausalGridDivergenceProbeError(
            "published development receipt sidecar does not match its bytes"
        )
    payload = _decode_strict_json(raw, label="published development receipt")
    if raw != _canonical_json_bytes(payload):
        raise CausalGridDivergenceProbeError(
            "published development receipt is not canonical JSON"
        )
    try:
        child_payload = {
            "schema_version": CHILD_SCHEMA_VERSION,
            "candidates": payload["candidates"],
            "pairwise_action_divergence": payload["pairwise_action_divergence"],
            "runtime_observation": payload["runtime_observation"],
        }
        reconstructed, _passed = _assemble_receipt(
            child_payload,
            probe_source_sha256=payload["probe_source_sha256"],
            execution_envelope=payload["execution_envelope"],
        )
    except (KeyError, TypeError) as exc:
        raise CausalGridDivergenceProbeError(
            "published development receipt omits required schema fields"
        ) from exc
    if reconstructed != payload or _canonical_json_bytes(reconstructed) != raw:
        raise CausalGridDivergenceProbeError(
            "published development receipt fails exact schema replay"
        )
    try:
        after = output_root.lstat()
    except OSError as exc:
        raise CausalGridDivergenceProbeError(
            "published development output root disappeared during replay"
        ) from exc
    if _stat_identity(before) != _stat_identity(after):
        raise CausalGridDivergenceProbeError(
            "published development output root changed during replay"
        )
    return payload


def _paths_overlap(left: Path, right: Path) -> bool:
    left_resolved = left.resolve(strict=False)
    right_resolved = right.resolve(strict=False)
    return (
        left_resolved == right_resolved
        or left_resolved in right_resolved.parents
        or right_resolved in left_resolved.parents
    )


def _validate_output_root(
    output_root: Path,
    *,
    qualification_root: Path,
    source_root: Path,
) -> None:
    if not output_root.is_absolute() or output_root != output_root.resolve(strict=False):
        raise CausalGridDivergenceProbeError(
            "development output root must be a canonical absolute path"
        )
    if any(
        part in {"", ".", ".."}
        or any(character in part for character in ("\x00", "\n", "\r"))
        for part in output_root.parts
    ):
        raise CausalGridDivergenceProbeError("development output root is unsafe")
    protected_paths = (
        qualification_root,
        source_root,
        Path(__file__).resolve(strict=True),
    )
    if any(_paths_overlap(output_root, protected) for protected in protected_paths):
        raise CausalGridDivergenceProbeError(
            "development output root overlaps a frozen input or probe source"
        )
    if output_root.exists() or output_root.is_symlink():
        raise CausalGridDivergenceProbeError(
            "development output root already exists; choose a new immutable path"
        )


def run_probe(
    *,
    qualification_root: Path = DEFAULT_QUALIFICATION_ROOT,
    source_root: Path | None = None,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    oci_runtime: Path = DEFAULT_OCI_RUNTIME,
) -> tuple[Path, dict[str, Any], bool]:
    """Execute, validate, and immutably persist one bounded development probe."""
    if not isinstance(qualification_root, Path) or not isinstance(output_root, Path):
        raise TypeError("qualification_root and output_root must be Path objects")
    if not isinstance(oci_runtime, Path):
        raise TypeError("oci_runtime must be a Path")
    probe_raw_before = _read_probe_source()
    probe_source_sha256 = _sha256(probe_raw_before)
    bound_source_root = (
        qualification_root / "sources" / "alberta" / "source"
        if source_root is None
        else source_root
    )
    if not isinstance(bound_source_root, Path):
        raise TypeError("source_root must be a Path or None")
    _validate_output_root(
        output_root,
        qualification_root=qualification_root,
        source_root=bound_source_root,
    )
    configuration_payloads, _manifest = _load_bound_inputs(
        qualification_root,
        bound_source_root,
    )
    before_source_records = _source_inventory_records(bound_source_root)
    child_payload, execution_envelope = _run_child(
        bound_source_root,
        qualification_root,
        oci_runtime=oci_runtime,
        expected_probe_source_sha256=probe_source_sha256,
    )
    # Rebind every load-bearing input after execution to catch mutable-host drift.
    after_payloads, _manifest = _load_bound_inputs(
        qualification_root,
        bound_source_root,
    )
    if after_payloads != configuration_payloads or (
        _source_inventory_records(bound_source_root) != before_source_records
    ):
        raise CausalGridDivergenceProbeError("frozen probe inputs changed during execution")
    probe_raw_after = _read_probe_source(
        expected_sha256=probe_source_sha256,
    )
    if probe_raw_after != probe_raw_before:
        raise CausalGridDivergenceProbeError(
            "causal-grid divergence probe source bytes changed across the run"
        )
    receipt, passed = _assemble_receipt(
        child_payload,
        probe_source_sha256=probe_source_sha256,
        execution_envelope=execution_envelope,
    )
    _validate_output_root(
        output_root,
        qualification_root=qualification_root,
        source_root=bound_source_root,
    )
    probe_raw_final = _read_probe_source(
        expected_sha256=probe_source_sha256,
    )
    if probe_raw_final != probe_raw_before:
        raise CausalGridDivergenceProbeError(
            "causal-grid divergence probe source bytes changed before publication"
        )
    receipt_path = _write_receipt(output_root, receipt)
    return receipt_path, receipt, passed


def _child_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--qualification-root", type=Path, required=True)
    parser.add_argument("--probe-source-sha256", required=True)
    return parser


def _public_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the frozen seed-0 causal q-grid divergence probe",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--qualification-root",
        type=Path,
        default=DEFAULT_QUALIFICATION_ROOT,
    )
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser


def _child_main(argv: Sequence[str]) -> int:
    args = _child_parser().parse_args(argv)
    if (
        args.source_root != Path("/run/alberta/source")
        or args.qualification_root != Path("/inputs/qualification")
    ):
        raise CausalGridDivergenceProbeError(
            "isolated child input paths differ from the private mount contract"
        )
    _read_probe_source(
        expected_sha256=_require_sha256(
            args.probe_source_sha256,
            label="child probe source digest",
        )
    )
    _extract_pinned_source_archive(
        args.qualification_root,
        args.source_root,
    )
    configurations, _manifest = _load_bound_inputs(
        args.qualification_root,
        args.source_root,
    )
    payload = _child_runtime_payload(
        args.source_root,
        configurations,
    )
    sys.stdout.buffer.write(_canonical_json_bytes(payload))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    arguments = tuple(sys.argv[1:] if argv is None else argv)
    try:
        if arguments and arguments[0] == "_child":
            return _child_main(arguments[1:])
        args = _public_parser().parse_args(arguments)
        _path, receipt, passed = run_probe(
            qualification_root=args.qualification_root,
            source_root=args.source_root,
            output_root=args.output_root,
        )
        sys.stdout.buffer.write(_canonical_json_bytes(receipt) + b"\n")
        return 0 if passed else 1
    except ReceiptPublishedButUncertainError as exc:
        sys.stderr.write(f"causal q-grid divergence probe: PUBLISHED-UNCERTAIN: {exc}\n")
        return 3
    except (CausalGridDivergenceProbeError, OSError, subprocess.SubprocessError) as exc:
        sys.stderr.write(f"causal q-grid divergence probe: {exc}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
