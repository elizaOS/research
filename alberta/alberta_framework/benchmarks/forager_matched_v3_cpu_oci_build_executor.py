"""Explicit, bounded-output Docker executor for a sealed matched-v3 CPU OCI context.

Execution requires a PID-bound, context-bound, single-use authorization object.
The executor first proves that the digest-pinned base is already local on the
required platform, proves that the reserved default builder is the local Docker
driver before and after the build, streams only the sealed USTAR on stdin to the
exact Buildx command carried by the context receipt, captures one exclusive
iidfile, and replays ``docker image inspect`` into a canonical nonauthorizing
receipt.

This module never issues a tag, publication, explicit pull, prune, or image
removal command.  ``--pull=false`` disables forced refresh, but neither that nor
Docker ``--network=none`` attests that the daemon lacked implicit registry
contact or egress; the receipt says so explicitly.
"""

from __future__ import annotations

import copy
import hashlib
import hmac
import json
import os
import re
import selectors
import signal
import stat
import subprocess
import tempfile
import threading
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Never, NoReturn, Protocol, SupportsIndex, cast

from alberta_framework.benchmarks import (
    forager_matched_v3_cpu_oci_build_context as context_contract,
)

CPU_OCI_BUILD_EXECUTION_RECEIPT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.cpu_oci_build_execution_receipt.v1"
)
CPU_OCI_BUILD_EXECUTION_STATUS: Final = "docker_image_built_inspected_unqualified_non_authorizing"
CPU_OCI_BUILD_EXECUTION_CLASSIFICATION: Final = "local_cpu_oci_build_observation_non_authorizing"
CPU_OCI_BUILD_EXECUTION_ACKNOWLEDGEMENT: Final = (
    "AUTHORIZE ONE LOCAL MATCHED-V3 CPU OCI BUILD FROM THIS SEALED CONTEXT"
)

_IMAGE_ID_RE: Final = re.compile(r"sha256:[0-9a-f]{64}\Z")
_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_MAX_INSPECT_BYTES: Final = 8 * 1024 * 1024
_MAX_CONTEXT_RECEIPT_BYTES: Final = 8 * 1024 * 1024
_MAX_RECEIPT_BYTES: Final = 16 * 1024 * 1024
_MAX_PROCESS_OUTPUT_BYTES: Final = 64 * 1024 * 1024
_DEFAULT_BUILD_TIMEOUT_SECONDS: Final = 7200
_MIN_BUILD_TIMEOUT_SECONDS: Final = 60
_MAX_BUILD_TIMEOUT_SECONDS: Final = 21_600
_INSPECT_TIMEOUT_SECONDS: Final = 60
_BUILDER_LS_TIMEOUT_SECONDS: Final = 60
_MAX_BUILDER_LS_BYTES: Final = 8 * 1024 * 1024
_MAX_BUILDER_LS_LINES: Final = 4096
_MAX_BUILDER_LS_LINE_BYTES: Final = 2 * 1024 * 1024
_MAX_JSON_DEPTH: Final = 64
_MAX_JSON_NODES: Final = 500_000
_MAX_JSON_TEXT: Final = 2 * 1024 * 1024
_MAX_INTEGER: Final = 2**63 - 1
_IID_FILENAME: Final = "image.id"
_DOCKER_DAEMON_HOST: Final = "unix:///var/run/docker.sock"
_DOCKER_HOST_ARGUMENT: Final = f"--host={_DOCKER_DAEMON_HOST}"
_BUILDX_BUILDER_NAME: Final = "default"
_BUILDX_BUILDER_ARGUMENT: Final = f"--builder={_BUILDX_BUILDER_NAME}"
_DOCKER_CLI_PATH: Final = "/usr/bin/docker"
_DOCKER_CLI_SHA256: Final = "d767d00af09e69cf053e9d923550fda999c2b5911c7a0a0a920b964e86b32d25"
_DOCKER_CLI_SIZE_BYTES: Final = 45_355_843
_DOCKER_LOADER_PATH: Final = "/usr/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2"
_DOCKER_LOADER_SHA256: Final = "cd4df4f3c7b83673d61189bf2eaebd33ca4f2853ab9772b8a25e025ef99b1e81"
_DOCKER_LOADER_SIZE_BYTES: Final = 236_616
_DOCKER_LIBC_PATH: Final = "/usr/lib/x86_64-linux-gnu/libc.so.6"
_DOCKER_LIBC_SHA256: Final = "8db37cf3f2169f59a0f07ef1fea308c35656668c64c8ff294e1860f4121eb161"
_DOCKER_LIBC_SIZE_BYTES: Final = 2_125_328
_DOCKER_LD_SO_CACHE_PATH: Final = "/etc/ld.so.cache"
_DOCKER_LD_SO_CACHE_SHA256: Final = (
    "8b230c5aaa5dc9c53a2133cc37522be42a63f9230596be98c2e5510c07615749"
)
_DOCKER_LD_SO_CACHE_SIZE_BYTES: Final = 103_319
_BUILDX_PLUGIN_PATH: Final = "/usr/libexec/docker/cli-plugins/docker-buildx"
_BUILDX_PLUGIN_SHA256: Final = "84554b12c90d21f3627d7f6a99aeabf69dc3aab8c2de41ed6f85097ed7cfd2c0"
_BUILDX_PLUGIN_SIZE_BYTES: Final = 72_456_104
_DOCKER_COMMAND_PREFIX: Final = (_DOCKER_CLI_PATH, _DOCKER_HOST_ARGUMENT)
_BUILDX_COMMAND_PREFIX: Final = (_BUILDX_PLUGIN_PATH, _BUILDX_BUILDER_ARGUMENT)
_CLI_FIXED_ENVIRONMENT: Final = {
    "DOCKER_HOST": _DOCKER_DAEMON_HOST,
    "LC_ALL": "C.UTF-8",
    "PATH": "/usr/bin:/bin",
}
_CLI_PRIVATE_DIRECTORIES: Final = {
    "BUILDX_CONFIG": "buildx",
    "DOCKER_CONFIG": "docker",
    "HOME": "home",
    "TMPDIR": "tmp",
    "XDG_CONFIG_HOME": "xdg",
}
_ROUTING_ENVIRONMENT_KEYS: Final = frozenset(
    {
        "BUILDKIT_HOST",
        "BUILDX_BUILDER",
        "BUILDX_CONFIG",
        "DOCKER_API_VERSION",
        "DOCKER_CLI_PLUGIN_EXTRA_DIRS",
        "DOCKER_CONFIG",
        "DOCKER_CONTEXT",
        "DOCKER_HOST",
    }
)


class ForagerMatchedV3CpuOciBuildExecutorError(RuntimeError):
    """An authorization, process, iidfile, inspect, or receipt failed closed."""

    def __init__(self, message: str, *, image_state_uncertain: bool = False) -> None:
        if image_state_uncertain and "image state is uncertain" not in message:
            message = f"{message}; image state is uncertain and no cleanup was attempted"
        super().__init__(message)
        self.image_state_uncertain = image_state_uncertain


def _fail(message: str, *, image_state_uncertain: bool = False) -> NoReturn:
    raise ForagerMatchedV3CpuOciBuildExecutorError(
        message,
        image_state_uncertain=image_state_uncertain,
    )


def _hash(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256(value: Any, *, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None or value == "0" * 64:
        _fail(f"{label} must be one nonzero lowercase SHA-256")
    return value


def _image_id(value: Any, *, label: str) -> str:
    if type(value) is not str or _IMAGE_ID_RE.fullmatch(value) is None:
        _fail(f"{label} must be one exact sha256 image ID")
    return value


def _integer(
    value: Any,
    *,
    label: str,
    minimum: int = 0,
    maximum: int = _MAX_INTEGER,
) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        _fail(f"{label} must be one bounded exact integer")
    return value


def _string(value: Any, *, label: str) -> str:
    if type(value) is not str or not value:
        _fail(f"{label} must be one nonempty exact string")
    return value


def _exact(value: Any, fields: frozenset[str], *, label: str) -> dict[str, Any]:
    if type(value) is not dict or frozenset(value) != fields:
        _fail(f"{label} fields are not exact")
    return cast(dict[str, Any], value)


def _raise_float(value: str) -> Never:
    _fail(f"execution receipt JSON contains float {value!r}")


def _raise_constant(value: str) -> Never:
    _fail(f"execution receipt JSON contains non-finite constant {value!r}")


def _parse_int(value: str) -> int:
    if len(value.lstrip("-")) > 19:
        _fail("execution receipt JSON integer exceeds its lexical bound")
    return int(value)


def _without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _assert_plain_json(value: Any, *, label: str) -> None:
    pending: list[tuple[Any, int]] = [(value, 0)]
    seen: set[int] = set()
    nodes = 0
    while pending:
        item, depth = pending.pop()
        nodes += 1
        if nodes > _MAX_JSON_NODES or depth > _MAX_JSON_DEPTH:
            _fail(f"{label} exceeds its structure bound")
        if type(item) is str:
            if len(item) > _MAX_JSON_TEXT or any(
                ord(character) < 0x20 and character not in "\n\r\t" for character in item
            ):
                _fail(f"{label} contains an invalid or oversized string")
            continue
        if item is None or type(item) in {bool, int}:
            if type(item) is int and not -_MAX_INTEGER <= item <= _MAX_INTEGER:
                _fail(f"{label} integer exceeds its value bound")
            continue
        if type(item) not in {dict, list}:
            _fail(f"{label} contains a non-JSON value")
        identity = id(item)
        if identity in seen:
            _fail(f"{label} contains a container alias")
        seen.add(identity)
        if type(item) is list:
            pending.extend((child, depth + 1) for child in item)
        else:
            for key, child in cast(dict[Any, Any], item).items():
                if type(key) is not str:
                    _fail(f"{label} object key is not an exact string")
                pending.append((key, depth + 1))
                pending.append((child, depth + 1))


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    exact = copy.deepcopy(dict(value))
    _assert_plain_json(exact, label="execution receipt")
    try:
        raw = (
            json.dumps(
                exact,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("ascii")
            + b"\n"
        )
    except (OverflowError, RecursionError, TypeError, UnicodeError, ValueError) as exc:
        raise ForagerMatchedV3CpuOciBuildExecutorError(
            "execution receipt is not canonical finite ASCII JSON"
        ) from exc
    if len(raw) > _MAX_RECEIPT_BYTES:
        _fail("execution receipt exceeds its byte bound")
    return raw


def _strict_receipt(raw: bytes) -> dict[str, Any]:
    if type(raw) is not bytes or not raw or len(raw) > _MAX_RECEIPT_BYTES:
        _fail("execution receipt must be nonempty exact bounded bytes")
    if not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
        _fail("execution receipt must have one canonical trailing newline")
    try:
        value = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_without_duplicates,
            parse_constant=_raise_constant,
            parse_float=_raise_float,
            parse_int=_parse_int,
        )
    except ForagerMatchedV3CpuOciBuildExecutorError:
        raise
    except (RecursionError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ForagerMatchedV3CpuOciBuildExecutorError(
            "execution receipt is not bounded strict ASCII JSON"
        ) from exc
    if type(value) is not dict:
        _fail("execution receipt root must be one object")
    result = cast(dict[str, Any], value)
    _assert_plain_json(result, label="execution receipt")
    if not hmac.compare_digest(_canonical_json(result), raw):
        _fail("execution receipt bytes are not canonical")
    return result


def _strict_inspect(raw: bytes, *, label: str) -> dict[str, Any]:
    if type(raw) is not bytes or not raw or len(raw) > _MAX_INSPECT_BYTES:
        _fail(f"{label} output is absent or exceeds its byte bound")
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_without_duplicates)
    except ForagerMatchedV3CpuOciBuildExecutorError:
        raise
    except (RecursionError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ForagerMatchedV3CpuOciBuildExecutorError(
            f"{label} output is not one strict JSON object"
        ) from exc
    if type(value) is not dict:
        _fail(f"{label} output root is not one object")
    result = cast(dict[str, Any], value)
    _assert_plain_json(result, label=label)
    return result


def _strict_json_lines(raw: bytes, *, label: str) -> list[dict[str, Any]]:
    if (
        type(raw) is not bytes
        or not raw
        or len(raw) > _MAX_BUILDER_LS_BYTES
        or not raw.endswith(b"\n")
        or raw.endswith(b"\n\n")
        or b"\r" in raw
    ):
        _fail(f"{label} must be bounded newline-terminated JSONL")
    lines = raw[:-1].split(b"\n")
    if not lines or len(lines) > _MAX_BUILDER_LS_LINES:
        _fail(f"{label} JSONL record count differs")
    records: list[dict[str, Any]] = []
    for line in lines:
        if not line or len(line) > _MAX_BUILDER_LS_LINE_BYTES:
            _fail(f"{label} contains an absent or oversized JSONL record")
        try:
            value = json.loads(
                line.decode("utf-8"),
                object_pairs_hook=_without_duplicates,
                parse_constant=_raise_constant,
                parse_float=_raise_float,
                parse_int=_parse_int,
            )
        except ForagerMatchedV3CpuOciBuildExecutorError:
            raise
        except (RecursionError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise ForagerMatchedV3CpuOciBuildExecutorError(
                f"{label} contains invalid strict JSONL"
            ) from exc
        if type(value) is not dict:
            _fail(f"{label} JSONL record root is not one object")
        record = cast(dict[str, Any], value)
        _assert_plain_json(record, label=label)
        records.append(record)
    return records


def _bounded_topology_string(value: Any, *, label: str, maximum: int = 256) -> str:
    if (
        type(value) is not str
        or not 1 <= len(value) <= maximum
        or any(ord(character) < 0x21 or ord(character) > 0x7E for character in value)
    ):
        _fail(f"{label} must be one bounded printable ASCII token")
    return value


def _validate_builder_locality_projection(value: Any) -> dict[str, Any]:
    projection = _exact(
        value,
        frozenset({"driver", "dynamic", "name", "node"}),
        label="builder locality projection",
    )
    node = _exact(
        projection["node"],
        frozenset({"endpoint", "name", "platforms", "status"}),
        label="builder locality node projection",
    )
    platforms = node["platforms"]
    if (
        projection["name"] != _BUILDX_BUILDER_NAME
        or projection["driver"] != "docker"
        or projection["dynamic"] is not False
        or node["name"] != _BUILDX_BUILDER_NAME
        or node["endpoint"] != _BUILDX_BUILDER_NAME
        or node["status"] != "running"
        or type(platforms) is not list
        or not platforms
        or len(platforms) > 256
    ):
        _fail("builder locality projection is not the running local default Docker builder")
    validated_platforms = [
        _bounded_topology_string(platform, label="builder locality platform", maximum=128)
        for platform in platforms
    ]
    if (
        validated_platforms != sorted(set(validated_platforms), key=str.encode)
        or "linux/amd64" not in validated_platforms
    ):
        _fail("builder locality platforms are noncanonical or omit linux/amd64")
    return copy.deepcopy(projection)


def _builder_locality_projection(raw: bytes, *, label: str) -> dict[str, Any]:
    records = _strict_json_lines(raw, label=label)
    default_records: list[dict[str, Any]] = []
    for record in records:
        name = _bounded_topology_string(record.get("Name"), label=f"{label} builder name")
        if name == _BUILDX_BUILDER_NAME:
            default_records.append(record)
    if len(default_records) != 1:
        _fail(f"{label} must contain exactly one default builder record")
    builder = default_records[0]
    if "Err" in builder and builder["Err"] != "":
        _fail(f"{label} default builder contains an error")
    nodes = builder.get("Nodes")
    if (
        builder.get("Driver") != "docker"
        or builder.get("Dynamic") is not False
        or type(nodes) is not list
        or len(nodes) != 1
        or type(nodes[0]) is not dict
    ):
        _fail(f"{label} default builder is not one static Docker-driver node")
    node = cast(dict[str, Any], nodes[0])
    if "Err" in node and node["Err"] != "":
        _fail(f"{label} default builder node contains an error")
    empty_only_fields: dict[str, type[dict[Any, Any]] | type[list[Any]]] = {
        "DriverOpts": dict,
        "Files": dict,
        "Flags": list,
        "ProxyConfig": dict,
    }
    for field, expected_type in empty_only_fields.items():
        if field in node and (type(node[field]) is not expected_type or node[field]):
            _fail(f"{label} default builder node {field} must be absent or empty")
    raw_platforms = node.get("Platforms")
    if type(raw_platforms) is not list or not raw_platforms or len(raw_platforms) > 256:
        _fail(f"{label} default builder platforms differ")
    platforms = [
        _bounded_topology_string(platform, label=f"{label} platform", maximum=128)
        for platform in raw_platforms
    ]
    if len(platforms) != len(set(platforms)) or "linux/amd64" not in platforms:
        _fail(f"{label} default builder platforms are duplicated or omit linux/amd64")
    projection = {
        "driver": "docker",
        "dynamic": False,
        "name": _BUILDX_BUILDER_NAME,
        "node": {
            "endpoint": _bounded_topology_string(
                node.get("Endpoint"), label=f"{label} default builder endpoint"
            ),
            "name": _bounded_topology_string(
                node.get("Name"), label=f"{label} default builder node name"
            ),
            "platforms": sorted(platforms, key=str.encode),
            "status": _bounded_topology_string(
                node.get("Status"), label=f"{label} default builder node status"
            ),
        },
    }
    return _validate_builder_locality_projection(projection)


def _canonical_object_sha256(value: Mapping[str, Any]) -> str:
    return _hash(_canonical_json(value))


@dataclass(frozen=True, slots=True)
class BoundedProcessResult:
    """One runner result with explicit timeout and output-bound observations."""

    returncode: int
    stdout: bytes
    stderr: bytes
    timed_out: bool = False
    output_limit_exceeded: bool = False

    def __post_init__(self) -> None:
        if (
            type(self.returncode) is not int
            or type(self.stdout) is not bytes
            or type(self.stderr) is not bytes
            or type(self.timed_out) is not bool
            or type(self.output_limit_exceeded) is not bool
        ):
            raise TypeError("bounded process results require exact scalar field types")


class ProcessRunner(Protocol):
    """Injected process boundary used by tests and the default bounded runner."""

    def __call__(
        self,
        argv: tuple[str, ...],
        *,
        environment: Mapping[str, str],
        executable_descriptor: int,
        inherited_descriptors: tuple[int, ...],
        working_directory: str,
        stdin_descriptor: int | None,
        timeout_seconds: int,
        stdout_limit_bytes: int,
        stderr_limit_bytes: int,
    ) -> BoundedProcessResult: ...


def _routing_environment_keys(environment: Mapping[str, str]) -> tuple[str, ...]:
    return tuple(
        sorted(
            (
                key
                for key in environment
                if key in _ROUTING_ENVIRONMENT_KEYS or key.startswith("DOCKER_TLS")
            ),
            key=str.encode,
        )
    )


def _require_routing_environment_absent(environment: Mapping[str, str]) -> None:
    keys = _routing_environment_keys(environment)
    if keys:
        _fail("ambient Docker or BuildKit routing variables are forbidden: " + ", ".join(keys))


def _execution_toolchain_contract() -> dict[str, Any]:
    return {
        "buildx_plugin": {
            "gid": 0,
            "mode": "0755",
            "path": _BUILDX_PLUGIN_PATH,
            "sha256": _BUILDX_PLUGIN_SHA256,
            "size_bytes": _BUILDX_PLUGIN_SIZE_BYTES,
            "uid": 0,
        },
        "docker_cli": {
            "gid": 0,
            "mode": "0755",
            "path": _DOCKER_CLI_PATH,
            "sha256": _DOCKER_CLI_SHA256,
            "size_bytes": _DOCKER_CLI_SIZE_BYTES,
            "uid": 0,
        },
        "docker_dynamic_runtime": {
            "ld_so_preload": {"path": "/etc/ld.so.preload", "required_state": "absent"},
            "regular_files": {
                "ld_so_cache": {
                    "gid": 0,
                    "mode": "0644",
                    "path": _DOCKER_LD_SO_CACHE_PATH,
                    "sha256": _DOCKER_LD_SO_CACHE_SHA256,
                    "size_bytes": _DOCKER_LD_SO_CACHE_SIZE_BYTES,
                    "uid": 0,
                },
                "libc": {
                    "gid": 0,
                    "mode": "0755",
                    "path": _DOCKER_LIBC_PATH,
                    "sha256": _DOCKER_LIBC_SHA256,
                    "size_bytes": _DOCKER_LIBC_SIZE_BYTES,
                    "uid": 0,
                },
                "loader": {
                    "gid": 0,
                    "mode": "0755",
                    "path": _DOCKER_LOADER_PATH,
                    "sha256": _DOCKER_LOADER_SHA256,
                    "size_bytes": _DOCKER_LOADER_SIZE_BYTES,
                    "uid": 0,
                },
            },
            "symlinks": [
                {"gid": 0, "path": "/lib", "target": "usr/lib", "uid": 0},
                {"gid": 0, "path": "/lib64", "target": "usr/lib64", "uid": 0},
                {
                    "gid": 0,
                    "path": "/usr/lib64/ld-linux-x86-64.so.2",
                    "target": "../lib/x86_64-linux-gnu/ld-linux-x86-64.so.2",
                    "uid": 0,
                },
            ],
        },
        "environment": {
            "fixed": dict(_CLI_FIXED_ENVIRONMENT),
            "private_directories": dict(_CLI_PRIVATE_DIRECTORIES),
            "working_directory": "tmp",
        },
        "invocation": "direct_pinned_executables_via_open_readonly_descriptors",
    }


def _stable_stat_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_uid,
        value.st_gid,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _require_secure_tool_path_ancestors(path: str, *, image_state_uncertain: bool) -> None:
    candidate = Path(path)
    if not candidate.is_absolute() or not candidate.name:
        _fail("pinned executable path is not absolute", image_state_uncertain=image_state_uncertain)
    for ancestor in reversed(candidate.parents):
        try:
            metadata = os.lstat(ancestor)
        except OSError as exc:
            raise ForagerMatchedV3CpuOciBuildExecutorError(
                "pinned executable ancestor metadata could not be read",
                image_state_uncertain=image_state_uncertain,
            ) from exc
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != 0
            or metadata.st_gid != 0
            or stat.S_IMODE(metadata.st_mode) & 0o022
        ):
            _fail(
                "pinned executable ancestor is not a root-owned non-writable directory",
                image_state_uncertain=image_state_uncertain,
            )


def _descriptor_sha256(descriptor: int, *, expected_size: int) -> str:
    digest = hashlib.sha256()
    offset = 0
    while offset < expected_size:
        block = os.pread(descriptor, min(1024 * 1024, expected_size - offset), offset)
        if not block:
            _fail("pinned executable ended before its exact size")
        digest.update(block)
        offset += len(block)
    if os.pread(descriptor, 1, expected_size):
        _fail("pinned executable exceeds its exact size")
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class _RetainedExecutable:
    key: str
    contract: Mapping[str, Any]
    path: str
    descriptor: int
    device: int
    inode: int

    def reverify(self, *, image_state_uncertain: bool) -> dict[str, Any]:
        contract = self.contract
        try:
            _require_secure_tool_path_ancestors(
                self.path,
                image_state_uncertain=image_state_uncertain,
            )
            before = os.lstat(self.path)
            opened = os.fstat(self.descriptor)
            digest = _descriptor_sha256(
                self.descriptor,
                expected_size=cast(int, contract["size_bytes"]),
            )
            after = os.lstat(self.path)
        except ForagerMatchedV3CpuOciBuildExecutorError as exc:
            if image_state_uncertain and not exc.image_state_uncertain:
                raise ForagerMatchedV3CpuOciBuildExecutorError(
                    str(exc),
                    image_state_uncertain=True,
                ) from exc
            raise
        except BaseException as exc:
            raise ForagerMatchedV3CpuOciBuildExecutorError(
                "pinned executable could not be stably verified",
                image_state_uncertain=image_state_uncertain,
            ) from exc
        if (
            not stat.S_ISREG(opened.st_mode)
            or _stable_stat_identity(before) != _stable_stat_identity(opened)
            or _stable_stat_identity(opened) != _stable_stat_identity(after)
            or (opened.st_dev, opened.st_ino) != (self.device, self.inode)
            or stat.S_IMODE(opened.st_mode) != int(cast(str, contract["mode"]), 8)
            or opened.st_nlink != 1
            or opened.st_uid != contract["uid"]
            or opened.st_gid != contract["gid"]
            or opened.st_size != contract["size_bytes"]
            or digest != contract["sha256"]
            or os.get_inheritable(self.descriptor)
        ):
            _fail(
                "pinned executable metadata or content identity differs",
                image_state_uncertain=image_state_uncertain,
            )
        return copy.deepcopy(dict(contract))


def _reverify_dynamic_runtime_routes(*, image_state_uncertain: bool) -> dict[str, Any]:
    runtime = cast(
        Mapping[str, Any],
        _execution_toolchain_contract()["docker_dynamic_runtime"],
    )
    expected_symlinks = cast(list[dict[str, Any]], runtime["symlinks"])
    observed_symlinks: list[dict[str, Any]] = []
    try:
        for expected in expected_symlinks:
            path = cast(str, expected["path"])
            _require_secure_tool_path_ancestors(
                path,
                image_state_uncertain=image_state_uncertain,
            )
            before = os.lstat(path)
            target = os.readlink(path)
            after = os.lstat(path)
            if (
                not stat.S_ISLNK(before.st_mode)
                or _stable_stat_identity(before) != _stable_stat_identity(after)
                or before.st_nlink != 1
                or before.st_uid != expected["uid"]
                or before.st_gid != expected["gid"]
                or target != expected["target"]
            ):
                _fail(
                    "Docker dynamic-loader symlink identity differs",
                    image_state_uncertain=image_state_uncertain,
                )
            observed_symlinks.append(copy.deepcopy(expected))
        preload = cast(Mapping[str, Any], runtime["ld_so_preload"])
        preload_path = cast(str, preload["path"])
        _require_secure_tool_path_ancestors(
            preload_path,
            image_state_uncertain=image_state_uncertain,
        )
        try:
            os.lstat(preload_path)
        except FileNotFoundError:
            pass
        else:
            _fail(
                "Docker dynamic-loader preload file must remain absent",
                image_state_uncertain=image_state_uncertain,
            )
    except ForagerMatchedV3CpuOciBuildExecutorError:
        raise
    except BaseException as exc:
        raise ForagerMatchedV3CpuOciBuildExecutorError(
            "Docker dynamic-loader routing could not be stably verified",
            image_state_uncertain=image_state_uncertain,
        ) from exc
    return {
        "ld_so_preload": copy.deepcopy(runtime["ld_so_preload"]),
        "symlinks": observed_symlinks,
    }


@dataclass(frozen=True, slots=True)
class _RetainedExecutionToolchain:
    docker_cli: _RetainedExecutable
    buildx_plugin: _RetainedExecutable
    docker_loader: _RetainedExecutable
    docker_libc: _RetainedExecutable
    docker_ld_so_cache: _RetainedExecutable

    def descriptor_for(self, argv: tuple[str, ...]) -> int:
        if argv and argv[0] == self.docker_cli.path:
            return self.docker_cli.descriptor
        if argv and argv[0] == self.buildx_plugin.path:
            return self.buildx_plugin.descriptor
        _fail("process argv does not select one retained pinned executable")

    def reverify(self, *, image_state_uncertain: bool) -> dict[str, Any]:
        routes = _reverify_dynamic_runtime_routes(image_state_uncertain=image_state_uncertain)
        records = {
            "buildx_plugin": self.buildx_plugin.reverify(
                image_state_uncertain=image_state_uncertain
            ),
            "docker_cli": self.docker_cli.reverify(image_state_uncertain=image_state_uncertain),
            "docker_dynamic_runtime": {
                "ld_so_preload": routes["ld_so_preload"],
                "regular_files": {
                    "ld_so_cache": self.docker_ld_so_cache.reverify(
                        image_state_uncertain=image_state_uncertain
                    ),
                    "libc": self.docker_libc.reverify(image_state_uncertain=image_state_uncertain),
                    "loader": self.docker_loader.reverify(
                        image_state_uncertain=image_state_uncertain
                    ),
                },
                "symlinks": routes["symlinks"],
            },
        }
        if records != {
            key: value
            for key, value in _execution_toolchain_contract().items()
            if key in {"buildx_plugin", "docker_cli", "docker_dynamic_runtime"}
        }:
            _fail(
                "retained executable records differ from the execution contract",
                image_state_uncertain=image_state_uncertain,
            )
        return records


def _open_retained_executable(
    key: str,
    contract: Mapping[str, Any],
) -> _RetainedExecutable:
    path = cast(str, contract["path"])
    _require_secure_tool_path_ancestors(path, image_state_uncertain=False)
    descriptor = -1
    primary_failure: BaseException | None = None
    transferred = False
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
        )
        opened = os.fstat(descriptor)
        retained = _RetainedExecutable(
            key,
            copy.deepcopy(contract),
            path,
            descriptor,
            opened.st_dev,
            opened.st_ino,
        )
        retained.reverify(image_state_uncertain=False)
        transferred = True
        return retained
    except BaseException as exc:
        primary_failure = exc
        raise
    finally:
        if descriptor >= 0 and not transferred:
            try:
                os.close(descriptor)
            except BaseException as exc:
                if primary_failure is not None:
                    primary_failure.add_note(
                        "untransferred pinned executable cleanup also failed: "
                        f"{type(exc).__name__}: {exc}"
                    )
                else:
                    raise


@contextmanager
def _retain_execution_toolchain() -> Iterator[_RetainedExecutionToolchain]:
    docker_cli: _RetainedExecutable | None = None
    buildx_plugin: _RetainedExecutable | None = None
    docker_loader: _RetainedExecutable | None = None
    docker_libc: _RetainedExecutable | None = None
    docker_ld_so_cache: _RetainedExecutable | None = None
    primary_failure: BaseException | None = None
    try:
        contract = _execution_toolchain_contract()
        dynamic_files = cast(
            Mapping[str, Mapping[str, Any]],
            cast(Mapping[str, Any], contract["docker_dynamic_runtime"])["regular_files"],
        )
        docker_cli = _open_retained_executable("docker_cli", contract["docker_cli"])
        buildx_plugin = _open_retained_executable(
            "buildx_plugin",
            contract["buildx_plugin"],
        )
        docker_loader = _open_retained_executable("docker_loader", dynamic_files["loader"])
        docker_libc = _open_retained_executable("docker_libc", dynamic_files["libc"])
        docker_ld_so_cache = _open_retained_executable(
            "docker_ld_so_cache",
            dynamic_files["ld_so_cache"],
        )
        yield _RetainedExecutionToolchain(
            docker_cli,
            buildx_plugin,
            docker_loader,
            docker_libc,
            docker_ld_so_cache,
        )
    except BaseException as exc:
        primary_failure = exc
        raise
    finally:
        cleanup_failures: list[BaseException] = []
        for executable in (
            docker_ld_so_cache,
            docker_libc,
            docker_loader,
            buildx_plugin,
            docker_cli,
        ):
            if executable is None:
                continue
            try:
                os.close(executable.descriptor)
            except BaseException as exc:
                cleanup_failures.append(exc)
        if cleanup_failures:
            if primary_failure is not None:
                for cleanup_failure in cleanup_failures:
                    primary_failure.add_note(
                        "pinned executable cleanup also failed: "
                        f"{type(cleanup_failure).__name__}: {cleanup_failure}"
                    )
            else:
                failure = ForagerMatchedV3CpuOciBuildExecutorError(
                    "pinned executable cleanup failed"
                )
                for cleanup_failure in cleanup_failures[1:]:
                    failure.add_note(
                        "additional pinned executable cleanup failure: "
                        f"{type(cleanup_failure).__name__}: {cleanup_failure}"
                    )
                raise failure from cleanup_failures[0]


@dataclass(frozen=True, slots=True)
class _PrivateCliState:
    directory_path: Path
    directory_descriptor: int
    directory_device: int
    directory_inode: int
    child_identities: Mapping[str, tuple[int, int]]
    environment: Mapping[str, str]
    working_directory: str

    def reverify(self, *, image_state_uncertain: bool) -> None:
        try:
            root = os.fstat(self.directory_descriptor)
            named_root = os.lstat(self.directory_path)
            if (
                not stat.S_ISDIR(root.st_mode)
                or (root.st_dev, root.st_ino) != (self.directory_device, self.directory_inode)
                or (named_root.st_dev, named_root.st_ino)
                != (self.directory_device, self.directory_inode)
                or root.st_uid != os.geteuid()
                or stat.S_IMODE(root.st_mode) != 0o700
                or os.get_inheritable(self.directory_descriptor)
            ):
                _fail(
                    "private CLI state root identity differs",
                    image_state_uncertain=image_state_uncertain,
                )
            for name, identity in self.child_identities.items():
                child = os.stat(
                    name,
                    dir_fd=self.directory_descriptor,
                    follow_symlinks=False,
                )
                if (
                    not stat.S_ISDIR(child.st_mode)
                    or (child.st_dev, child.st_ino) != identity
                    or child.st_uid != os.geteuid()
                    or stat.S_IMODE(child.st_mode) != 0o700
                ):
                    _fail(
                        "private CLI state directory identity differs",
                        image_state_uncertain=image_state_uncertain,
                    )
        except ForagerMatchedV3CpuOciBuildExecutorError:
            raise
        except BaseException as exc:
            raise ForagerMatchedV3CpuOciBuildExecutorError(
                "private CLI state could not be reverified",
                image_state_uncertain=image_state_uncertain,
            ) from exc


def _clear_directory_contents(descriptor: int) -> None:
    failures: list[BaseException] = []
    try:
        with os.scandir(descriptor) as entries:
            names = sorted((entry.name for entry in entries), key=os.fsencode)
    except BaseException as exc:
        raise ForagerMatchedV3CpuOciBuildExecutorError(
            "private CLI state directory could not be enumerated for cleanup"
        ) from exc
    for name in names:
        child_descriptor = -1
        try:
            metadata = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            if stat.S_ISDIR(metadata.st_mode):
                child_descriptor = os.open(name, _directory_flags(), dir_fd=descriptor)
                opened = os.fstat(child_descriptor)
                if not stat.S_ISDIR(opened.st_mode) or (opened.st_dev, opened.st_ino) != (
                    metadata.st_dev,
                    metadata.st_ino,
                ):
                    _fail("private CLI state child changed during anchored cleanup")
                _clear_directory_contents(child_descriptor)
                os.rmdir(name, dir_fd=descriptor)
            else:
                os.unlink(name, dir_fd=descriptor)
        except FileNotFoundError:
            pass
        except BaseException as exc:
            failures.append(exc)
        finally:
            if child_descriptor >= 0:
                try:
                    os.close(child_descriptor)
                except BaseException as exc:
                    failures.append(exc)
    if failures:
        failure = ForagerMatchedV3CpuOciBuildExecutorError(
            "private CLI state anchored content cleanup failed"
        )
        for cleanup_failure in failures[1:]:
            failure.add_note(
                "additional anchored content cleanup failure: "
                f"{type(cleanup_failure).__name__}: {cleanup_failure}"
            )
        raise failure from failures[0]


@contextmanager
def _exclusive_cli_state() -> Iterator[_PrivateCliState]:
    directory_path = Path(tempfile.mkdtemp(prefix="alberta-matched-v3-oci-cli-"))
    descriptor = -1
    identity: tuple[int, int] | None = None
    primary_failure: BaseException | None = None
    try:
        descriptor = os.open(directory_path, _directory_flags())
        opened = os.fstat(descriptor)
        identity = (opened.st_dev, opened.st_ino)
        created = os.lstat(directory_path)
        if (
            not stat.S_ISDIR(created.st_mode)
            or stat.S_IMODE(created.st_mode) != 0o700
            or created.st_uid != os.geteuid()
            or (created.st_dev, created.st_ino) != identity
            or stat.S_IMODE(opened.st_mode) != 0o700
            or opened.st_uid != os.geteuid()
            or os.get_inheritable(descriptor)
        ):
            _fail("private CLI state root metadata differs")
        child_identities: dict[str, tuple[int, int]] = {}
        for name in sorted(set(_CLI_PRIVATE_DIRECTORIES.values()), key=str.encode):
            os.mkdir(name, mode=0o700, dir_fd=descriptor)
            child = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            if (
                not stat.S_ISDIR(child.st_mode)
                or stat.S_IMODE(child.st_mode) != 0o700
                or child.st_uid != os.geteuid()
            ):
                _fail("private CLI state child creation metadata differs")
            child_identities[name] = (child.st_dev, child.st_ino)
        descriptor_root = f"/proc/self/fd/{descriptor}"
        environment = {
            **_CLI_FIXED_ENVIRONMENT,
            **{
                key: f"{descriptor_root}/{relative}"
                for key, relative in _CLI_PRIVATE_DIRECTORIES.items()
            },
        }
        state = _PrivateCliState(
            directory_path,
            descriptor,
            opened.st_dev,
            opened.st_ino,
            child_identities,
            environment,
            f"{descriptor_root}/tmp",
        )
        state.reverify(image_state_uncertain=False)
        yield state
    except BaseException as exc:
        primary_failure = exc
        raise
    finally:
        cleanup_failures: list[BaseException] = []
        if descriptor >= 0:
            try:
                _clear_directory_contents(descriptor)
            except BaseException as exc:
                cleanup_failures.append(exc)
        current: os.stat_result | None
        try:
            current = os.lstat(directory_path)
        except FileNotFoundError:
            current = None
        except BaseException as exc:
            current = None
            cleanup_failures.append(exc)
        if current is not None and identity == (current.st_dev, current.st_ino):
            try:
                os.rmdir(directory_path)
            except BaseException as exc:
                cleanup_failures.append(exc)
        elif current is not None:
            cleanup_failures.append(
                ForagerMatchedV3CpuOciBuildExecutorError(
                    "private CLI state root path identity changed before cleanup"
                )
            )
        elif identity is not None:
            cleanup_failures.append(
                ForagerMatchedV3CpuOciBuildExecutorError(
                    "private CLI state root path vanished before cleanup"
                )
            )
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except BaseException as exc:
                cleanup_failures.append(exc)
        if cleanup_failures:
            if primary_failure is not None:
                for cleanup_failure in cleanup_failures:
                    primary_failure.add_note(
                        "private CLI state cleanup also failed: "
                        f"{type(cleanup_failure).__name__}: {cleanup_failure}"
                    )
            else:
                failure = ForagerMatchedV3CpuOciBuildExecutorError(
                    "private CLI state cleanup failed"
                )
                for cleanup_failure in cleanup_failures[1:]:
                    failure.add_note(
                        "additional private CLI state cleanup failure: "
                        f"{type(cleanup_failure).__name__}: {cleanup_failure}"
                    )
                raise failure from cleanup_failures[0]


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    process_group = process.pid

    def group_exists() -> bool:
        try:
            os.killpg(process_group, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    try:
        os.killpg(process_group, signal.SIGTERM)
    except (OSError, ProcessLookupError):
        pass
    deadline = time.monotonic() + 0.5
    while group_exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    if group_exists():
        try:
            os.killpg(process_group, signal.SIGKILL)
        except (OSError, ProcessLookupError):
            pass


def _live_process_group_member_pids(process_group: int, *, leader_pid: int) -> tuple[int, ...]:
    members: list[int] = []
    try:
        entries = os.scandir("/proc")
    except OSError as exc:
        raise ForagerMatchedV3CpuOciBuildExecutorError(
            "anchored process-group membership could not be inspected"
        ) from exc
    with entries:
        for entry in entries:
            if not entry.name.isdecimal():
                continue
            pid = int(entry.name)
            if pid == leader_pid:
                continue
            try:
                raw = Path(entry.path, "stat").read_text(encoding="ascii")
            except (FileNotFoundError, ProcessLookupError):
                continue
            except (OSError, UnicodeError) as exc:
                raise ForagerMatchedV3CpuOciBuildExecutorError(
                    "anchored process-group member metadata could not be read"
                ) from exc
            _prefix, separator, suffix = raw.rpartition(") ")
            fields = suffix.split()
            if separator != ") " or len(fields) < 4:
                _fail("anchored process-group member metadata is malformed")
            try:
                member_process_group = int(fields[2])
            except ValueError as exc:
                raise ForagerMatchedV3CpuOciBuildExecutorError(
                    "anchored process-group member identity is malformed"
                ) from exc
            if member_process_group == process_group and fields[0] not in {"X", "Z", "x"}:
                members.append(pid)
    return tuple(sorted(members))


def _wait_for_leader_exit_unreaped(process: subprocess.Popen[bytes], *, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    options = os.WEXITED | os.WNOHANG | os.WNOWAIT
    while True:
        try:
            result = os.waitid(os.P_PID, process.pid, options)
        except (ChildProcessError, OSError) as exc:
            raise ForagerMatchedV3CpuOciBuildExecutorError(
                "process leader could not be observed before anchored cleanup"
            ) from exc
        if result is not None:
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired(process.args, timeout)
        time.sleep(min(0.01, remaining))


def _terminate_anchored_process_group(process: subprocess.Popen[bytes]) -> None:
    """Terminate live group members while the exited leader remains unreaped."""

    process_group = process.pid
    try:
        os.killpg(process_group, signal.SIGTERM)
    except ProcessLookupError:
        return
    except OSError as exc:
        raise ForagerMatchedV3CpuOciBuildExecutorError(
            "anchored process group could not receive SIGTERM"
        ) from exc
    inspection_failure: BaseException | None = None
    deadline = time.monotonic() + 0.5
    while time.monotonic() < deadline:
        try:
            if not _live_process_group_member_pids(process_group, leader_pid=process.pid):
                if inspection_failure is not None:
                    raise inspection_failure
                return
        except BaseException as exc:
            if inspection_failure is None:
                inspection_failure = exc
        time.sleep(0.01)
    try:
        os.killpg(process_group, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except OSError as exc:
        raise ForagerMatchedV3CpuOciBuildExecutorError(
            "anchored process group could not receive SIGKILL"
        ) from exc
    deadline = time.monotonic() + 0.5
    while time.monotonic() < deadline:
        try:
            if not _live_process_group_member_pids(process_group, leader_pid=process.pid):
                if inspection_failure is not None:
                    raise inspection_failure
                return
        except BaseException as exc:
            if inspection_failure is None:
                inspection_failure = exc
        time.sleep(0.01)
    try:
        retained_members = _live_process_group_member_pids(
            process_group,
            leader_pid=process.pid,
        )
    except BaseException as exc:
        if inspection_failure is None:
            inspection_failure = exc
        retained_members = ()
    if retained_members:
        _fail("anchored process group retained live descendants after SIGKILL")
    if inspection_failure is not None:
        raise inspection_failure


def _default_process_runner(
    argv: tuple[str, ...],
    *,
    environment: Mapping[str, str],
    executable_descriptor: int,
    inherited_descriptors: tuple[int, ...],
    working_directory: str,
    stdin_descriptor: int | None,
    timeout_seconds: int,
    stdout_limit_bytes: int,
    stderr_limit_bytes: int,
) -> BoundedProcessResult:
    """Bound one argv's time/output and terminate its original process group."""

    if type(environment) is not dict or any(
        type(key) is not str or type(value) is not str for key, value in environment.items()
    ):
        _fail("bounded process environment must be one exact string mapping")
    if type(executable_descriptor) is not int or executable_descriptor < 0:
        _fail("bounded process executable descriptor is invalid")
    if type(inherited_descriptors) is not tuple or any(
        type(descriptor) is not int or descriptor < 0 for descriptor in inherited_descriptors
    ):
        _fail("bounded process inherited descriptors are invalid")
    if (
        type(working_directory) is not str
        or not working_directory.startswith("/")
        or "\x00" in working_directory
    ):
        _fail("bounded process working directory is invalid")
    pass_descriptors = tuple(sorted({executable_descriptor, *inherited_descriptors}))
    for descriptor in pass_descriptors:
        try:
            metadata = os.fstat(descriptor)
        except OSError as exc:
            raise ForagerMatchedV3CpuOciBuildExecutorError(
                "bounded process inherited descriptor is not open"
            ) from exc
        if descriptor == executable_descriptor and not stat.S_ISREG(metadata.st_mode):
            _fail("bounded process executable descriptor is not a regular file")
    process_environment = dict(environment)
    try:
        process = subprocess.Popen(
            list(argv),
            executable=f"/proc/self/fd/{executable_descriptor}",
            stdin=subprocess.DEVNULL if stdin_descriptor is None else stdin_descriptor,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            close_fds=True,
            cwd=working_directory,
            pass_fds=pass_descriptors,
            start_new_session=True,
            env=process_environment,
        )
    except (OSError, ValueError) as exc:
        raise ForagerMatchedV3CpuOciBuildExecutorError(
            f"process could not be started: {argv[0]!r}"
        ) from exc
    selector: selectors.BaseSelector | None = None
    streams: dict[int, tuple[Any, bytearray, int]] = {}
    stdout_buffer = bytearray()
    stderr_buffer = bytearray()
    returncode: int | None = None
    timed_out = False
    output_limit_exceeded = False
    failure: BaseException | None = None
    leader_reaped = False
    command_deadline = time.monotonic() + timeout_seconds

    def bounded_wait(*, terminate_first: bool = False) -> int:
        nonlocal leader_reaped, timed_out
        if terminate_first:
            _terminate_process_group(process)
        wait_timeout = 2.0
        if not terminate_first and not timed_out and not output_limit_exceeded:
            wait_timeout = max(0.0, command_deadline - time.monotonic())
        try:
            _wait_for_leader_exit_unreaped(process, timeout=wait_timeout)
        except subprocess.TimeoutExpired:
            if not terminate_first and not output_limit_exceeded:
                timed_out = True
            _terminate_process_group(process)
            try:
                _wait_for_leader_exit_unreaped(process, timeout=2.0)
            except subprocess.TimeoutExpired as exc:
                raise ForagerMatchedV3CpuOciBuildExecutorError(
                    "process leader did not exit within the bounded cleanup interval"
                ) from exc
        cleanup_failure: BaseException | None = None
        try:
            _terminate_anchored_process_group(process)
        except BaseException as exc:
            cleanup_failure = exc
        try:
            returncode = process.wait(timeout=2.0)
            leader_reaped = True
        except BaseException as exc:
            if cleanup_failure is not None:
                raise cleanup_failure from exc
            raise
        if cleanup_failure is not None:
            raise cleanup_failure
        return returncode

    try:
        if process.stdout is None or process.stderr is None:
            _fail("bounded process pipes were not created")
        stdout_descriptor = process.stdout.fileno()
        stderr_descriptor = process.stderr.fileno()
        streams = {
            stdout_descriptor: (process.stdout, stdout_buffer, stdout_limit_bytes),
            stderr_descriptor: (process.stderr, stderr_buffer, stderr_limit_bytes),
        }
        selector = selectors.DefaultSelector()
        for descriptor, (stream, _buffer, _limit) in streams.items():
            os.set_blocking(descriptor, False)
            selector.register(stream, selectors.EVENT_READ, descriptor)
        forced_deadline: float | None = None
        while selector.get_map():
            now = time.monotonic()
            if not timed_out and not output_limit_exceeded and now >= command_deadline:
                timed_out = True
                _terminate_process_group(process)
                forced_deadline = time.monotonic() + 3.0
            if forced_deadline is not None and now >= forced_deadline:
                for key in list(selector.get_map().values()):
                    selector.unregister(key.fileobj)
                    descriptor = cast(int, key.data)
                    streams[descriptor][0].close()
                break
            timeout = 0.1
            if not timed_out and not output_limit_exceeded:
                timeout = max(0.0, min(timeout, command_deadline - now))
            for key, _mask in selector.select(timeout):
                descriptor = cast(int, key.data)
                stream, buffer, limit = streams[descriptor]
                try:
                    block = os.read(descriptor, 64 * 1024)
                except BlockingIOError:
                    continue
                if not block:
                    selector.unregister(stream)
                    stream.close()
                    continue
                available = max(0, limit - len(buffer))
                if available:
                    buffer.extend(block[:available])
                if len(block) > available and not output_limit_exceeded:
                    output_limit_exceeded = True
                    _terminate_process_group(process)
                    forced_deadline = time.monotonic() + 3.0
        returncode = bounded_wait()
    except BaseException as exc:
        failure = exc
    finally:
        if selector is not None:
            try:
                selector.close()
            except BaseException as exc:
                if failure is None:
                    failure = exc
                else:
                    failure.add_note(f"selector cleanup also failed: {type(exc).__name__}: {exc}")
        for stream, _buffer, _limit in streams.values():
            try:
                if not stream.closed:
                    stream.close()
            except BaseException as exc:
                if failure is None:
                    failure = exc
                else:
                    failure.add_note(
                        f"process stream cleanup also failed: {type(exc).__name__}: {exc}"
                    )
        if not leader_reaped:
            try:
                returncode = bounded_wait(terminate_first=True)
            except BaseException as exc:
                if failure is None:
                    failure = exc
                else:
                    failure.add_note(
                        f"bounded process-group cleanup also failed: {type(exc).__name__}: {exc}"
                    )
    if failure is not None:
        if isinstance(failure, ForagerMatchedV3CpuOciBuildExecutorError):
            raise failure
        raise ForagerMatchedV3CpuOciBuildExecutorError(
            "bounded process setup, monitoring, or cleanup failed"
        ) from failure
    if returncode is None:
        _fail("bounded process return status is absent")
    return BoundedProcessResult(
        returncode,
        bytes(stdout_buffer),
        bytes(stderr_buffer),
        timed_out=timed_out,
        output_limit_exceeded=output_limit_exceeded,
    )


def _run(
    runner: ProcessRunner,
    argv: tuple[str, ...],
    *,
    environment: Mapping[str, str],
    executable_descriptor: int,
    inherited_descriptors: tuple[int, ...],
    working_directory: str,
    stdin_descriptor: int | None,
    timeout_seconds: int,
    label: str,
    image_state_uncertain: bool,
) -> BoundedProcessResult:
    try:
        result = runner(
            argv,
            environment=environment,
            executable_descriptor=executable_descriptor,
            inherited_descriptors=inherited_descriptors,
            working_directory=working_directory,
            stdin_descriptor=stdin_descriptor,
            timeout_seconds=timeout_seconds,
            stdout_limit_bytes=_MAX_PROCESS_OUTPUT_BYTES,
            stderr_limit_bytes=_MAX_PROCESS_OUTPUT_BYTES,
        )
    except ForagerMatchedV3CpuOciBuildExecutorError as exc:
        if image_state_uncertain and not exc.image_state_uncertain:
            raise ForagerMatchedV3CpuOciBuildExecutorError(
                str(exc), image_state_uncertain=True
            ) from exc
        raise
    except BaseException as exc:
        raise ForagerMatchedV3CpuOciBuildExecutorError(
            f"{label} runner failed",
            image_state_uncertain=image_state_uncertain,
        ) from exc
    if type(result) is not BoundedProcessResult:
        _fail(
            f"{label} runner returned a noncanonical result",
            image_state_uncertain=image_state_uncertain,
        )
    if (
        len(result.stdout) > _MAX_PROCESS_OUTPUT_BYTES
        or len(result.stderr) > _MAX_PROCESS_OUTPUT_BYTES
    ):
        _fail(f"{label} exceeded its output bound", image_state_uncertain=image_state_uncertain)
    if result.timed_out:
        _fail(f"{label} timed out", image_state_uncertain=image_state_uncertain)
    if result.output_limit_exceeded:
        _fail(f"{label} exceeded its output bound", image_state_uncertain=image_state_uncertain)
    if result.returncode != 0:
        _fail(f"{label} failed with a nonzero status", image_state_uncertain=image_state_uncertain)
    return result


_AUTHORIZATION_TOKEN: Final = object()


class MatchedV3CpuOciBuildAuthorization:
    """Opaque PID-bound, context-bound, single-use local build authorization."""

    __slots__ = (
        "_archive_sha256",
        "_consumed",
        "_context_capability",
        "_lock",
        "_owner_pid",
        "_plan_sha256",
        "_receipt_sha256",
    )

    def __init__(
        self,
        token: object,
        context_capability: context_contract.RetainedMatchedV3CpuOciBuildContext,
        archive_sha256: str,
        receipt_sha256: str,
        plan_sha256: str,
    ) -> None:
        if token is not _AUTHORIZATION_TOKEN:
            raise TypeError("OCI build authorizations require the explicit authorization factory")
        self._context_capability: context_contract.RetainedMatchedV3CpuOciBuildContext | None = (
            context_capability
        )
        self._archive_sha256 = archive_sha256
        self._receipt_sha256 = receipt_sha256
        self._plan_sha256 = plan_sha256
        self._owner_pid = os.getpid()
        self._consumed = False
        self._lock = threading.Lock()

    def __reduce__(self) -> Never:
        raise TypeError("OCI build authorizations cannot be serialized")

    def __reduce_ex__(self, protocol: SupportsIndex) -> Never:
        del protocol
        raise TypeError("OCI build authorizations cannot be serialized")

    def __copy__(self) -> Never:
        raise TypeError("OCI build authorizations cannot be copied")

    def __deepcopy__(self, memo: object) -> Never:
        del memo
        raise TypeError("OCI build authorizations cannot be copied")

    @property
    def consumed(self) -> bool:
        return self._consumed

    def _consume(
        self,
        context_capability: context_contract.RetainedMatchedV3CpuOciBuildContext,
    ) -> None:
        if os.getpid() != self._owner_pid:
            self._consumed = True
            self._context_capability = None
            _fail("explicit OCI build authorization is invalid after a PID change")
        with self._lock:
            if os.getpid() != self._owner_pid:
                self._consumed = True
                self._context_capability = None
                _fail("explicit OCI build authorization is invalid after a PID change")
            if self._consumed:
                _fail("explicit OCI build authorization was already consumed")
            authorized_context = self._context_capability
            self._consumed = True
            self._context_capability = None
            if (
                context_capability is not authorized_context
                or context_capability.archive_sha256 != self._archive_sha256
                or context_capability.receipt_sha256 != self._receipt_sha256
                or context_capability.plan_sha256 != self._plan_sha256
            ):
                _fail("explicit OCI build authorization is bound to a different context")


def authorize_matched_v3_cpu_oci_build(
    *,
    context_capability: context_contract.RetainedMatchedV3CpuOciBuildContext,
    exact_acknowledgement: str,
) -> MatchedV3CpuOciBuildAuthorization:
    """Create one explicit single-use authorization bound to a sealed context."""

    if type(context_capability) is not context_contract.RetainedMatchedV3CpuOciBuildContext:
        _fail("explicit OCI build authorization requires the exact context capability")
    if type(exact_acknowledgement) is not str or not hmac.compare_digest(
        exact_acknowledgement,
        CPU_OCI_BUILD_EXECUTION_ACKNOWLEDGEMENT,
    ):
        _fail("explicit OCI build acknowledgement differs")
    context_capability.reverify()
    return MatchedV3CpuOciBuildAuthorization(
        _AUTHORIZATION_TOKEN,
        context_capability,
        context_capability.archive_sha256,
        context_capability.receipt_sha256,
        context_capability.plan_sha256,
    )


@dataclass(frozen=True, slots=True)
class _PrivateIidfile:
    directory_path: Path
    directory_descriptor: int
    directory_device: int
    directory_inode: int
    file_path: Path


def _directory_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )


@contextmanager
def _exclusive_iidfile() -> Iterator[_PrivateIidfile]:
    directory_path = Path(tempfile.mkdtemp(prefix="alberta-matched-v3-oci-iid-"))
    directory_descriptor = -1
    identity: tuple[int, int] | None = None
    primary_failure: BaseException | None = None
    try:
        created = os.lstat(directory_path)
        identity = (created.st_dev, created.st_ino)
        if (
            not stat.S_ISDIR(created.st_mode)
            or stat.S_IMODE(created.st_mode) != 0o700
            or created.st_uid != os.geteuid()
        ):
            _fail("exclusive iidfile directory creation metadata differs")
        os.chmod(directory_path, 0o700, follow_symlinks=False)
        before = os.lstat(directory_path)
        directory_descriptor = os.open(directory_path, _directory_flags())
        opened = os.fstat(directory_descriptor)
        if (
            not stat.S_ISDIR(before.st_mode)
            or identity != (before.st_dev, before.st_ino)
            or (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino)
            or stat.S_IMODE(opened.st_mode) != 0o700
            or opened.st_uid != os.geteuid()
            or os.get_inheritable(directory_descriptor)
        ):
            _fail("exclusive iidfile directory metadata differs")
        try:
            os.stat(_IID_FILENAME, dir_fd=directory_descriptor, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            _fail("exclusive iidfile path already exists")
        yield _PrivateIidfile(
            directory_path,
            directory_descriptor,
            opened.st_dev,
            opened.st_ino,
            directory_path / _IID_FILENAME,
        )
    except BaseException as exc:
        primary_failure = exc
        raise
    finally:
        cleanup_failures: list[BaseException] = []
        if directory_descriptor >= 0:
            try:
                os.unlink(_IID_FILENAME, dir_fd=directory_descriptor)
            except FileNotFoundError:
                pass
            except IsADirectoryError:
                pass
            except BaseException as exc:
                cleanup_failures.append(exc)
            try:
                os.close(directory_descriptor)
            except BaseException as exc:
                cleanup_failures.append(exc)
        current: os.stat_result | None
        try:
            current = os.lstat(directory_path)
        except FileNotFoundError:
            current = None
        except BaseException as exc:
            current = None
            cleanup_failures.append(exc)
        if current is not None and identity == (current.st_dev, current.st_ino):
            try:
                os.rmdir(directory_path)
            except BaseException as exc:
                cleanup_failures.append(exc)
        elif current is not None:
            cleanup_failures.append(
                ForagerMatchedV3CpuOciBuildExecutorError(
                    "exclusive iidfile directory path identity changed before cleanup"
                )
            )
        elif identity is not None:
            cleanup_failures.append(
                ForagerMatchedV3CpuOciBuildExecutorError(
                    "exclusive iidfile directory path vanished before cleanup"
                )
            )
        if cleanup_failures:
            if primary_failure is not None:
                for cleanup_failure in cleanup_failures:
                    primary_failure.add_note(
                        "exclusive iidfile cleanup also failed: "
                        f"{type(cleanup_failure).__name__}: {cleanup_failure}"
                    )
            else:
                failure = ForagerMatchedV3CpuOciBuildExecutorError(
                    "exclusive iidfile cleanup failed"
                )
                for cleanup_failure in cleanup_failures[1:]:
                    failure.add_note(
                        "additional exclusive iidfile cleanup failure: "
                        f"{type(cleanup_failure).__name__}: {cleanup_failure}"
                    )
                raise failure from cleanup_failures[0]


def _stable_iidfile(private: _PrivateIidfile) -> str:
    try:
        before = os.stat(
            _IID_FILENAME,
            dir_fd=private.directory_descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        _fail("exclusive iidfile is missing", image_state_uncertain=True)
    except OSError as exc:
        raise ForagerMatchedV3CpuOciBuildExecutorError(
            "exclusive iidfile metadata read failed",
            image_state_uncertain=True,
        ) from exc
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_uid != os.geteuid()
        or before.st_size != 71
    ):
        _fail("exclusive iidfile metadata or bytes differ", image_state_uncertain=True)
    descriptor = -1
    primary_failure: BaseException | None = None
    try:
        descriptor = os.open(
            _IID_FILENAME,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
            dir_fd=private.directory_descriptor,
        )
        opened = os.fstat(descriptor)
        raw = os.read(descriptor, 72)
        after = os.fstat(descriptor)
        restated = os.stat(
            _IID_FILENAME,
            dir_fd=private.directory_descriptor,
            follow_symlinks=False,
        )
        directory = os.fstat(private.directory_descriptor)
    except BaseException as exc:
        primary_failure = ForagerMatchedV3CpuOciBuildExecutorError(
            "exclusive iidfile stable read failed",
            image_state_uncertain=True,
        )
        raise primary_failure from exc
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except BaseException as exc:
                if primary_failure is not None:
                    primary_failure.add_note(
                        "exclusive iidfile descriptor cleanup also failed: "
                        f"{type(exc).__name__}: {exc}"
                    )
                else:
                    raise ForagerMatchedV3CpuOciBuildExecutorError(
                        "exclusive iidfile descriptor cleanup failed",
                        image_state_uncertain=True,
                    ) from exc

    def identity(item: os.stat_result) -> tuple[int, ...]:
        return (
            item.st_dev,
            item.st_ino,
            item.st_mode,
            item.st_nlink,
            item.st_uid,
            item.st_gid,
            item.st_size,
            item.st_mtime_ns,
            item.st_ctime_ns,
        )

    if (
        identity(before) != identity(opened)
        or identity(opened) != identity(after)
        or identity(after) != identity(restated)
        or (directory.st_dev, directory.st_ino)
        != (private.directory_device, private.directory_inode)
    ):
        _fail("exclusive iidfile changed during stable read", image_state_uncertain=True)
    try:
        value = raw.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ForagerMatchedV3CpuOciBuildExecutorError(
            "exclusive iidfile is not ASCII",
            image_state_uncertain=True,
        ) from exc
    return _image_id(value, label="exclusive iidfile")


def _require_context_streamed_to_eof(descriptor: int, *, expected_size: int) -> None:
    try:
        offset = os.lseek(descriptor, 0, os.SEEK_CUR)
        tail = os.read(descriptor, 1)
    except OSError as exc:
        raise ForagerMatchedV3CpuOciBuildExecutorError(
            "Docker Buildx context EOF check failed",
            image_state_uncertain=True,
        ) from exc
    if offset != expected_size or tail != b"":
        _fail(
            "Docker Buildx did not consume the exact context through EOF",
            image_state_uncertain=True,
        )


def _inspect_command(reference: str) -> tuple[str, ...]:
    return (*_DOCKER_COMMAND_PREFIX, "image", "inspect", "--format={{json .}}", reference)


def _builder_ls_command() -> tuple[str, ...]:
    return (*_BUILDX_COMMAND_PREFIX, "ls", "--format=json")


def _normalize_docker_repository(value: str) -> str:
    repository = value
    if repository.startswith("docker.io/"):
        repository = repository.removeprefix("docker.io/")
    if "/" not in repository:
        repository = f"library/{repository}"
    return f"docker.io/{repository}"


def _digest_reference(value: Any, *, label: str) -> tuple[str, str]:
    reference = _string(value, label=label)
    repository, separator, digest = reference.rpartition("@")
    if (
        separator != "@"
        or not repository
        or not digest.startswith("sha256:")
        or _SHA256_RE.fullmatch(digest.removeprefix("sha256:")) is None
        or any(character.isspace() or ord(character) > 0x7E for character in repository)
    ):
        _fail(f"{label} is not one exact digest reference")
    normalized = _normalize_docker_repository(repository)
    components = normalized.removeprefix("docker.io/").split("/")
    if any(
        not component
        or component in {".", ".."}
        or re.fullmatch(r"[a-z0-9]+(?:[._-][a-z0-9]+)*", component) is None
        for component in components
    ):
        _fail(f"{label} repository is not canonical")
    return normalized, digest


def _matching_repo_digest(inspect: Mapping[str, Any], expected_reference: str) -> str:
    expected_repository, expected_digest = _digest_reference(
        expected_reference,
        label="expected base image reference",
    )
    raw_digests = inspect.get("RepoDigests")
    if type(raw_digests) is not list:
        _fail("base image inspect omits repository digests")
    matches: list[str] = []
    for value in raw_digests:
        if type(value) is not str:
            _fail("base image inspect contains a non-string repository digest")
        repository, digest = _digest_reference(value, label="local base repository digest")
        if digest == expected_digest and repository == expected_repository:
            matches.append(value)
    if len(matches) != 1:
        _fail("base image preflight did not find the exact local repository digest")
    return matches[0]


def _validate_base_inspect(
    inspect: Mapping[str, Any],
    *,
    expected_reference: str,
    expected_platform: str,
) -> tuple[str, str]:
    operating_system, separator, architecture = expected_platform.partition("/")
    if separator != "/" or not operating_system or not architecture:
        _fail("planned base platform is invalid")
    if inspect.get("Os") != operating_system or inspect.get("Architecture") != architecture:
        _fail("base image preflight platform differs")
    image = _image_id(inspect.get("Id"), label="base image ID")
    return image, _matching_repo_digest(inspect, expected_reference)


def _validate_built_inspect(
    inspect: Mapping[str, Any],
    *,
    expected_image_id: str,
    expected_platform: str,
    expected_labels: Mapping[str, Any],
) -> tuple[str, str]:
    operating_system, _, architecture = expected_platform.partition("/")
    if inspect.get("Id") != expected_image_id:
        _fail("built image inspect image ID differs", image_state_uncertain=True)
    if inspect.get("Os") != operating_system or inspect.get("Architecture") != architecture:
        _fail("built image inspect platform differs", image_state_uncertain=True)
    config = inspect.get("Config")
    if type(config) is not dict:
        _fail("built image inspect configuration is absent", image_state_uncertain=True)
    labels = cast(dict[str, Any], config).get("Labels")
    if type(labels) is not dict or cast(dict[str, Any], labels) != dict(expected_labels):
        _fail("built image inspect labels differ", image_state_uncertain=True)
    if config.get("User") != "65532:65532" or config.get("WorkingDir") != "/work":
        _fail("built image inspect runtime user or workdir differs", image_state_uncertain=True)
    return cast(str, config["User"]), cast(str, config["WorkingDir"])


def _claims() -> dict[str, bool]:
    return {
        "further_execution_authority_granted": False,
        "publication_authority_granted": False,
        "qualification_granted": False,
        "runtime_qualified": False,
        "scientific_evidence_created": False,
        "scientific_promotion_allowed": False,
        "universal_sota_claim_allowed": False,
    }


def _limitations() -> list[str]:
    return [
        (
            "This receipt observes one image build on the explicit local Unix daemon through "
            "its reserved default Docker-driver builder; it grants no further authority."
        ),
        (
            "BuildKit network=none constrains build steps, but Docker daemon registry egress "
            "is not attested by this executor."
        ),
        (
            "No explicit pull command was issued and pull refresh was disabled; implicit "
            "registry contact and daemon egress remain unobserved and unattested."
        ),
        "The image is unqualified; no benchmark workload or runtime qualification ran here.",
        (
            "No image tag, publication, prune, evidence, promotion, or SOTA operation was "
            "issued or authorized."
        ),
        (
            "Process cleanup terminates the launched process group; it does not contain a "
            "descendant that successfully creates a different session or process group. The "
            "executed Docker and Buildx binaries are exact pinned inputs, not arbitrary code."
        ),
        (
            "The Docker CLI loader, libc, loader cache, interpreter symlinks, and absent "
            "ld.so preload file are pinned; the host kernel, vDSO, and daemon remain external."
        ),
    ]


def _stream_record(raw: bytes) -> dict[str, Any]:
    return {"sha256": _hash(raw), "size_bytes": len(raw)}


def _builder_locality_record(
    *,
    command: tuple[str, ...],
    result: BoundedProcessResult,
    projection: Mapping[str, Any],
) -> dict[str, Any]:
    validated_projection = _validate_builder_locality_projection(projection)
    projection_bytes = _canonical_json(validated_projection)
    return {
        "command": list(command),
        "projection": validated_projection,
        "projection_sha256": _hash(projection_bytes),
        "projection_size_bytes": len(projection_bytes),
        "stderr": _stream_record(result.stderr),
        "stdout": _stream_record(result.stdout),
    }


def _build_receipt(
    *,
    context_receipt: Mapping[str, Any],
    context_receipt_bytes: bytes,
    context_receipt_sha256: str,
    base_command: tuple[str, ...],
    base_result: BoundedProcessResult,
    base_inspect: Mapping[str, Any],
    base_image_id: str,
    matched_repo_digest: str,
    builder_ls_command: tuple[str, ...],
    builder_preflight_result: BoundedProcessResult,
    builder_preflight_projection: Mapping[str, Any],
    builder_postflight_result: BoundedProcessResult,
    builder_postflight_projection: Mapping[str, Any],
    toolchain_preflight: Mapping[str, Any],
    toolchain_postflight: Mapping[str, Any],
    build_command: tuple[str, ...],
    build_result: BoundedProcessResult,
    timeout_seconds: int,
    image_id: str,
    inspect_command: tuple[str, ...],
    inspect_result: BoundedProcessResult,
    inspect: Mapping[str, Any],
    user: str,
    working_directory: str,
) -> tuple[bytes, str]:
    archive = cast(Mapping[str, Any], context_receipt["archive"])
    plan = cast(Mapping[str, Any], context_receipt["plan"])
    base = cast(Mapping[str, Any], context_receipt["base_image"])
    if (
        type(context_receipt_bytes) is not bytes
        or len(context_receipt_bytes) > _MAX_CONTEXT_RECEIPT_BYTES
        or _hash(context_receipt_bytes) != context_receipt_sha256
    ):
        _fail("canonical context receipt identity differs", image_state_uncertain=True)
    try:
        canonical_context_receipt = context_receipt_bytes.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ForagerMatchedV3CpuOciBuildExecutorError(
            "canonical context receipt is not ASCII",
            image_state_uncertain=True,
        ) from exc
    receipt: dict[str, Any] = {
        "base_preflight": {
            "architecture": cast(str, base["platform"]).split("/", 1)[1],
            "command": list(base_command),
            "image_id": base_image_id,
            "inspect_object_sha256": _canonical_object_sha256(base_inspect),
            "inspect_output": _stream_record(base_result.stdout),
            "matched_repository_digest": matched_repo_digest,
            "operating_system": cast(str, base["platform"]).split("/", 1)[0],
            "stderr": _stream_record(base_result.stderr),
        },
        "build": {
            "command": list(build_command),
            "image_id": image_id,
            "stderr": _stream_record(build_result.stderr),
            "stdin_context_sha256": archive["sha256"],
            "stdin_context_size_bytes": archive["size_bytes"],
            "stdout": _stream_record(build_result.stdout),
            "timeout_seconds": timeout_seconds,
        },
        "builder_locality": {
            "builder_name": _BUILDX_BUILDER_NAME,
            "daemon_host": _DOCKER_DAEMON_HOST,
            "postflight": _builder_locality_record(
                command=builder_ls_command,
                result=builder_postflight_result,
                projection=builder_postflight_projection,
            ),
            "preflight": _builder_locality_record(
                command=builder_ls_command,
                result=builder_preflight_result,
                projection=builder_preflight_projection,
            ),
        },
        "claims": _claims(),
        "classification": CPU_OCI_BUILD_EXECUTION_CLASSIFICATION,
        "context": {
            "archive_sha256": archive["sha256"],
            "archive_size_bytes": archive["size_bytes"],
            "canonical_receipt": canonical_context_receipt,
            "canonical_receipt_size_bytes": len(context_receipt_bytes),
            "execution_projection_sha256": context_receipt["execution_projection_sha256"],
            "expected_image_labels_sha256": _hash(
                _canonical_json({"expected_image_labels": context_receipt["expected_image_labels"]})
            ),
            "execution_toolchain_sha256": _hash(
                _canonical_json({"execution_toolchain": context_receipt["execution_toolchain"]})
            ),
            "plan_sha256": plan["full_file_sha256"],
            "receipt_sha256": context_receipt_sha256,
        },
        "image_inspect": {
            "architecture": cast(str, base["platform"]).split("/", 1)[1],
            "command": list(inspect_command),
            "expected_labels": copy.deepcopy(context_receipt["expected_image_labels"]),
            "image_id": image_id,
            "inspect_object_sha256": _canonical_object_sha256(inspect),
            "inspect_output": _stream_record(inspect_result.stdout),
            "operating_system": cast(str, base["platform"]).split("/", 1)[0],
            "stderr": _stream_record(inspect_result.stderr),
            "user": user,
            "working_directory": working_directory,
        },
        "execution_toolchain": {
            "contract": copy.deepcopy(context_receipt["execution_toolchain"]),
            "contract_sha256": _hash(
                _canonical_json({"execution_toolchain": context_receipt["execution_toolchain"]})
            ),
            "postflight": copy.deepcopy(dict(toolchain_postflight)),
            "preflight": copy.deepcopy(dict(toolchain_preflight)),
        },
        "limitations": _limitations(),
        "observation": {
            "ambient_routing_environment_absent": True,
            "base_digest_preloaded": True,
            "builder_locality_postflight_matched": True,
            "builder_locality_preflight_matched": True,
            "container_image_built": True,
            "daemon_egress_isolation_attested": False,
            "exact_context_streamed": True,
            "explicit_pull_command_issued": False,
            "image_inspect_matched": True,
            "isolated_cli_environment_used": True,
            "pinned_executable_descriptors_used": True,
            "pull_refresh_disabled": True,
            "registry_contact_absence_attested": False,
        },
        "receipt_body_sha256": "0" * 64,
        "schema_version": CPU_OCI_BUILD_EXECUTION_RECEIPT_SCHEMA_VERSION,
        "status": CPU_OCI_BUILD_EXECUTION_STATUS,
    }
    body = copy.deepcopy(receipt)
    body.pop("receipt_body_sha256")
    receipt["receipt_body_sha256"] = _hash(_canonical_json(body))
    raw = _canonical_json(receipt)
    parse_matched_v3_cpu_oci_build_execution_receipt(raw, expected_receipt_sha256=_hash(raw))
    return raw, _hash(raw)


def _validate_stream_record(value: Any, *, label: str) -> dict[str, Any]:
    record = _exact(value, frozenset({"sha256", "size_bytes"}), label=label)
    _sha256(record["sha256"], label=f"{label} digest")
    _integer(record["size_bytes"], label=f"{label} size", maximum=_MAX_PROCESS_OUTPUT_BYTES)
    return record


def _validate_builder_locality_record(value: Any, *, label: str) -> dict[str, Any]:
    record = _exact(
        value,
        frozenset(
            {
                "command",
                "projection",
                "projection_sha256",
                "projection_size_bytes",
                "stderr",
                "stdout",
            }
        ),
        label=label,
    )
    if record["command"] != list(_builder_ls_command()):
        _fail(f"{label} command differs from the exact local builder query")
    projection = _validate_builder_locality_projection(record["projection"])
    projection_bytes = _canonical_json(projection)
    if record["projection_sha256"] != _hash(projection_bytes) or record[
        "projection_size_bytes"
    ] != len(projection_bytes):
        _fail(f"{label} normalized projection identity differs")
    _sha256(record["projection_sha256"], label=f"{label} projection")
    _integer(
        record["projection_size_bytes"],
        label=f"{label} projection size",
        minimum=1,
        maximum=_MAX_BUILDER_LS_LINE_BYTES,
    )
    stdout = _validate_stream_record(record["stdout"], label=f"{label} stdout")
    if cast(int, stdout["size_bytes"]) > _MAX_BUILDER_LS_BYTES:
        _fail(f"{label} stdout exceeds the builder-list byte bound")
    _validate_stream_record(record["stderr"], label=f"{label} stderr")
    return record


def _validate_execution_toolchain_record(value: Any) -> dict[str, Any]:
    record = _exact(
        value,
        frozenset({"contract", "contract_sha256", "postflight", "preflight"}),
        label="execution toolchain record",
    )
    expected_contract = _execution_toolchain_contract()
    if record["contract"] != expected_contract:
        _fail("execution toolchain contract differs")
    expected_contract_sha256 = _hash(_canonical_json({"execution_toolchain": expected_contract}))
    if record["contract_sha256"] != expected_contract_sha256:
        _fail("execution toolchain contract identity differs")
    expected_executables = {
        key: copy.deepcopy(value)
        for key, value in expected_contract.items()
        if key in {"buildx_plugin", "docker_cli", "docker_dynamic_runtime"}
    }
    preflight = _exact(
        record["preflight"],
        frozenset(expected_executables),
        label="execution toolchain preflight",
    )
    postflight = _exact(
        record["postflight"],
        frozenset(expected_executables),
        label="execution toolchain postflight",
    )
    if preflight != expected_executables or postflight != expected_executables:
        _fail("execution toolchain executable identity differs")
    return record


def _validate_execution_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    receipt = _exact(
        value,
        frozenset(
            {
                "base_preflight",
                "build",
                "builder_locality",
                "claims",
                "classification",
                "context",
                "execution_toolchain",
                "image_inspect",
                "limitations",
                "observation",
                "receipt_body_sha256",
                "schema_version",
                "status",
            }
        ),
        label="execution receipt",
    )
    if (
        receipt["schema_version"] != CPU_OCI_BUILD_EXECUTION_RECEIPT_SCHEMA_VERSION
        or receipt["status"] != CPU_OCI_BUILD_EXECUTION_STATUS
        or receipt["classification"] != CPU_OCI_BUILD_EXECUTION_CLASSIFICATION
    ):
        _fail("execution receipt schema, status, or classification differs")
    claims = _exact(receipt["claims"], frozenset(_claims()), label="execution claims")
    if claims != _claims() or any(value is not False for value in claims.values()):
        _fail("execution receipt authority claim became true")
    if receipt["limitations"] != _limitations():
        _fail("execution receipt limitations differ")
    observation = _exact(
        receipt["observation"],
        frozenset(
            {
                "ambient_routing_environment_absent",
                "base_digest_preloaded",
                "builder_locality_postflight_matched",
                "builder_locality_preflight_matched",
                "container_image_built",
                "daemon_egress_isolation_attested",
                "exact_context_streamed",
                "explicit_pull_command_issued",
                "image_inspect_matched",
                "isolated_cli_environment_used",
                "pinned_executable_descriptors_used",
                "pull_refresh_disabled",
                "registry_contact_absence_attested",
            }
        ),
        label="execution observation",
    )
    if observation != {
        "ambient_routing_environment_absent": True,
        "base_digest_preloaded": True,
        "builder_locality_postflight_matched": True,
        "builder_locality_preflight_matched": True,
        "container_image_built": True,
        "daemon_egress_isolation_attested": False,
        "exact_context_streamed": True,
        "explicit_pull_command_issued": False,
        "image_inspect_matched": True,
        "isolated_cli_environment_used": True,
        "pinned_executable_descriptors_used": True,
        "pull_refresh_disabled": True,
        "registry_contact_absence_attested": False,
    }:
        _fail("execution receipt observation differs")
    locality = _exact(
        receipt["builder_locality"],
        frozenset({"builder_name", "daemon_host", "postflight", "preflight"}),
        label="builder locality",
    )
    if (
        locality["builder_name"] != _BUILDX_BUILDER_NAME
        or locality["daemon_host"] != _DOCKER_DAEMON_HOST
    ):
        _fail("builder locality daemon or builder binding differs")
    preflight = _validate_builder_locality_record(
        locality["preflight"], label="builder locality preflight"
    )
    postflight = _validate_builder_locality_record(
        locality["postflight"], label="builder locality postflight"
    )
    preflight_projection_bytes = _canonical_json(cast(Mapping[str, Any], preflight["projection"]))
    postflight_projection_bytes = _canonical_json(cast(Mapping[str, Any], postflight["projection"]))
    if (
        not hmac.compare_digest(preflight_projection_bytes, postflight_projection_bytes)
        or preflight["projection_sha256"] != postflight["projection_sha256"]
    ):
        _fail("builder locality normalized projection drifted across the build")
    context = _exact(
        receipt["context"],
        frozenset(
            {
                "archive_sha256",
                "archive_size_bytes",
                "canonical_receipt",
                "canonical_receipt_size_bytes",
                "execution_projection_sha256",
                "expected_image_labels_sha256",
                "execution_toolchain_sha256",
                "plan_sha256",
                "receipt_sha256",
            }
        ),
        label="execution context",
    )
    for field in (
        "archive_sha256",
        "execution_projection_sha256",
        "expected_image_labels_sha256",
        "execution_toolchain_sha256",
        "plan_sha256",
        "receipt_sha256",
    ):
        _sha256(context[field], label=f"execution context {field}")
    _integer(
        context["archive_size_bytes"],
        label="execution context archive size",
        minimum=10_240,
        maximum=7 * 1024 * 1024 * 1024,
    )
    context_receipt_size = _integer(
        context["canonical_receipt_size_bytes"],
        label="canonical context receipt size",
        minimum=1,
        maximum=_MAX_CONTEXT_RECEIPT_BYTES,
    )
    canonical_context_receipt = _string(
        context["canonical_receipt"],
        label="canonical context receipt",
    )
    try:
        canonical_context_receipt_bytes = canonical_context_receipt.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ForagerMatchedV3CpuOciBuildExecutorError(
            "canonical context receipt is not ASCII"
        ) from exc
    if (
        len(canonical_context_receipt_bytes) != context_receipt_size
        or _hash(canonical_context_receipt_bytes) != context["receipt_sha256"]
    ):
        _fail("embedded canonical context receipt identity differs")
    try:
        parsed_context_receipt = context_contract.parse_matched_v3_cpu_oci_build_context_receipt(
            canonical_context_receipt_bytes,
            expected_receipt_sha256=cast(str, context["receipt_sha256"]),
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ForagerMatchedV3CpuOciBuildExecutorError(
            "embedded canonical context receipt failed its authoritative parser"
        ) from exc
    context_archive = cast(Mapping[str, Any], parsed_context_receipt["archive"])
    context_plan = cast(Mapping[str, Any], parsed_context_receipt["plan"])
    context_base = cast(Mapping[str, Any], parsed_context_receipt["base_image"])
    context_labels = cast(Mapping[str, Any], parsed_context_receipt["expected_image_labels"])
    context_build_template = cast(list[str], parsed_context_receipt["build_command_template"])
    context_toolchain = cast(Mapping[str, Any], parsed_context_receipt["execution_toolchain"])
    toolchain_record = _validate_execution_toolchain_record(receipt["execution_toolchain"])
    expected_toolchain_sha256 = _hash(_canonical_json({"execution_toolchain": context_toolchain}))
    if (
        context["archive_sha256"] != context_archive["sha256"]
        or context["archive_size_bytes"] != context_archive["size_bytes"]
        or context["plan_sha256"] != context_plan["full_file_sha256"]
        or context["execution_projection_sha256"]
        != parsed_context_receipt["execution_projection_sha256"]
        or context["expected_image_labels_sha256"]
        != _hash(_canonical_json({"expected_image_labels": context_labels}))
        or context["execution_toolchain_sha256"] != expected_toolchain_sha256
        or toolchain_record["contract_sha256"] != expected_toolchain_sha256
        or toolchain_record["contract"] != context_toolchain
    ):
        _fail("execution context differs from its embedded canonical context receipt")
    expected_reference = cast(str, context_base["reference"])
    expected_platform = cast(str, context_base["platform"])
    expected_manifest_digest = cast(str, context_base["manifest_digest"])
    expected_repository, expected_digest = _digest_reference(
        expected_reference,
        label="embedded context base reference",
    )
    if expected_digest != expected_manifest_digest:
        _fail("embedded context base reference differs from its manifest digest")
    expected_operating_system, separator, expected_architecture = expected_platform.partition("/")
    if separator != "/" or not expected_operating_system or not expected_architecture:
        _fail("embedded context platform is invalid")
    base = _exact(
        receipt["base_preflight"],
        frozenset(
            {
                "architecture",
                "command",
                "image_id",
                "inspect_object_sha256",
                "inspect_output",
                "matched_repository_digest",
                "operating_system",
                "stderr",
            }
        ),
        label="base preflight",
    )
    _image_id(base["image_id"], label="base preflight image ID")
    _sha256(base["inspect_object_sha256"], label="base inspect object")
    _validate_stream_record(base["inspect_output"], label="base inspect output")
    _validate_stream_record(base["stderr"], label="base inspect stderr")
    expected_base_command = list(_inspect_command(expected_reference))
    if (
        base["operating_system"] != expected_operating_system
        or base["architecture"] != expected_architecture
        or base["command"] != expected_base_command
    ):
        _fail("base preflight command or platform differs")
    matched_repository, matched_digest = _digest_reference(
        base["matched_repository_digest"],
        label="base preflight matched repository digest",
    )
    if matched_repository != expected_repository or matched_digest != expected_digest:
        _fail("base preflight repository digest differs from the embedded context")
    build = _exact(
        receipt["build"],
        frozenset(
            {
                "command",
                "image_id",
                "stderr",
                "stdin_context_sha256",
                "stdin_context_size_bytes",
                "stdout",
                "timeout_seconds",
            }
        ),
        label="build observation",
    )
    image = _image_id(build["image_id"], label="built image ID")
    if (
        build["stdin_context_sha256"] != context["archive_sha256"]
        or build["stdin_context_size_bytes"] != context["archive_size_bytes"]
    ):
        _fail("build stdin context differs from the sealed context")
    _validate_stream_record(build["stdout"], label="build stdout")
    _validate_stream_record(build["stderr"], label="build stderr")
    _integer(
        build["timeout_seconds"],
        label="build timeout",
        minimum=_MIN_BUILD_TIMEOUT_SECONDS,
        maximum=_MAX_BUILD_TIMEOUT_SECONDS,
    )
    command = build["command"]
    expected_build_prefix = [*_BUILDX_COMMAND_PREFIX, "build"]
    if (
        type(command) is not list
        or len(command) < len(expected_build_prefix)
        or command[: len(expected_build_prefix)] != expected_build_prefix
    ):
        _fail("build receipt command differs from Buildx")
    iid_arguments = [
        item for item in command if type(item) is str and item.startswith("--iidfile=")
    ]
    if (
        len(iid_arguments) != 1
        or command[-1] != "-"
        or command[-2] != iid_arguments[0]
        or not Path(iid_arguments[0].split("=", 1)[1]).is_absolute()
        or Path(iid_arguments[0].split("=", 1)[1]).name != _IID_FILENAME
    ):
        _fail("build receipt iidfile or sealed-stdin command differs")
    if command != [*context_build_template[:-1], iid_arguments[0], "-"]:
        _fail("build receipt reproducibility or isolation arguments differ")
    inspect = _exact(
        receipt["image_inspect"],
        frozenset(
            {
                "architecture",
                "command",
                "expected_labels",
                "image_id",
                "inspect_object_sha256",
                "inspect_output",
                "operating_system",
                "stderr",
                "user",
                "working_directory",
            }
        ),
        label="image inspect observation",
    )
    if (
        inspect["image_id"] != image
        or inspect["operating_system"] != expected_operating_system
        or inspect["architecture"] != expected_architecture
        or inspect["user"] != "65532:65532"
        or inspect["working_directory"] != "/work"
        or inspect["command"] != list(_inspect_command(image))
    ):
        _fail("built image inspect identity, platform, or runtime configuration differs")
    _sha256(inspect["inspect_object_sha256"], label="built inspect object")
    _validate_stream_record(inspect["inspect_output"], label="built inspect output")
    _validate_stream_record(inspect["stderr"], label="built inspect stderr")
    labels = inspect["expected_labels"]
    expected_label_keys = {
        "io.elizaos.alberta.forager-matched-v3.base-manifest",
        "io.elizaos.alberta.forager-matched-v3.cas-manifest-sha256",
        "io.elizaos.alberta.forager-matched-v3.external-source-sha256",
        "io.elizaos.alberta.forager-matched-v3.local-source-sha256",
        "io.elizaos.alberta.forager-matched-v3.runtime-lock-sha256",
        "io.elizaos.alberta.forager-matched-v3.wheelhouse-sha256",
    }
    if (
        type(labels) is not dict
        or frozenset(labels) != expected_label_keys
        or labels != context_labels
    ):
        _fail("built inspect expected labels differ")
    if context["expected_image_labels_sha256"] != _hash(
        _canonical_json({"expected_image_labels": labels})
    ):
        _fail("built inspect labels differ from the sealed context binding")
    for key, label_value in cast(dict[str, Any], labels).items():
        if key.endswith("base-manifest"):
            if (
                type(label_value) is not str
                or not label_value.startswith("sha256:")
                or _SHA256_RE.fullmatch(label_value.removeprefix("sha256:")) is None
            ):
                _fail("built inspect base-manifest label differs")
        else:
            _sha256(label_value, label=f"built inspect expected label {key}")
    if (
        cast(dict[str, Any], labels)["io.elizaos.alberta.forager-matched-v3.base-manifest"]
        != expected_manifest_digest
    ):
        _fail("built inspect base-manifest label differs from the embedded context")
    _sha256(receipt["receipt_body_sha256"], label="execution receipt body")
    body = copy.deepcopy(receipt)
    supplied = cast(str, body.pop("receipt_body_sha256"))
    if not hmac.compare_digest(supplied, _hash(_canonical_json(body))):
        _fail("execution receipt body digest differs")
    return copy.deepcopy(receipt)


def parse_matched_v3_cpu_oci_build_execution_receipt(
    raw: bytes,
    *,
    expected_receipt_sha256: str,
) -> dict[str, Any]:
    """Parse one canonical successful-build receipt under an exact full digest."""

    expected = _sha256(expected_receipt_sha256, label="expected execution receipt")
    if type(raw) is not bytes or not hmac.compare_digest(_hash(raw), expected):
        _fail("execution receipt full-file SHA-256 differs")
    return _validate_execution_receipt(_strict_receipt(raw))


@dataclass(frozen=True, slots=True)
class CpuOciBuildExecutionArtifacts:
    """Canonical successful-build receipt and the exact inspected image ID."""

    image_id: str
    receipt_bytes: bytes
    receipt_sha256: str

    def __post_init__(self) -> None:
        image = _image_id(self.image_id, label="execution artifact image ID")
        receipt = parse_matched_v3_cpu_oci_build_execution_receipt(
            self.receipt_bytes,
            expected_receipt_sha256=self.receipt_sha256,
        )
        if cast(Mapping[str, Any], receipt["build"])["image_id"] != image:
            _fail("execution artifact image ID differs from its receipt")


def execute_matched_v3_cpu_oci_build(
    *,
    context_capability: context_contract.RetainedMatchedV3CpuOciBuildContext,
    authorization: MatchedV3CpuOciBuildAuthorization | None,
    timeout_seconds: int = _DEFAULT_BUILD_TIMEOUT_SECONDS,
) -> CpuOciBuildExecutionArtifacts:
    """Execute one exact local build after consuming explicit single-use authority."""

    if type(context_capability) is not context_contract.RetainedMatchedV3CpuOciBuildContext:
        _fail("OCI build execution requires the exact sealed context capability")
    if type(authorization) is not MatchedV3CpuOciBuildAuthorization:
        _fail("explicit single-use OCI build authorization is required")
    timeout = _integer(
        timeout_seconds,
        label="OCI build timeout",
        minimum=_MIN_BUILD_TIMEOUT_SECONDS,
        maximum=_MAX_BUILD_TIMEOUT_SECONDS,
    )
    authorization._consume(context_capability)
    context_receipt = context_capability.reverify()
    context_archive = cast(Mapping[str, Any], context_receipt["archive"])
    context_archive_size_bytes = cast(int, context_archive["size_bytes"])
    context_receipt_bytes = context_capability.receipt_bytes
    context_receipt_sha256 = context_capability.receipt_sha256
    _require_routing_environment_absent(dict(os.environ))
    if context_receipt["execution_toolchain"] != _execution_toolchain_contract():
        _fail("sealed context execution toolchain differs")
    runner = _default_process_runner
    base = cast(Mapping[str, Any], context_receipt["base_image"])
    reference = cast(str, base["reference"])
    platform = cast(str, base["platform"])
    build_started = False
    try:
        with _retain_execution_toolchain() as toolchain:
            toolchain_preflight = toolchain.reverify(image_state_uncertain=False)
            with _exclusive_cli_state() as cli_state:

                def run_bound(
                    argv: tuple[str, ...],
                    *,
                    stdin_descriptor: int | None,
                    timeout_seconds: int,
                    label: str,
                    image_state_uncertain: bool,
                ) -> BoundedProcessResult:
                    cli_state.reverify(image_state_uncertain=image_state_uncertain)
                    toolchain.reverify(image_state_uncertain=image_state_uncertain)
                    return _run(
                        runner,
                        argv,
                        environment=cli_state.environment,
                        executable_descriptor=toolchain.descriptor_for(argv),
                        inherited_descriptors=(cli_state.directory_descriptor,),
                        working_directory=cli_state.working_directory,
                        stdin_descriptor=stdin_descriptor,
                        timeout_seconds=timeout_seconds,
                        label=label,
                        image_state_uncertain=image_state_uncertain,
                    )

                builder_ls_command = _builder_ls_command()
                builder_preflight_result = run_bound(
                    builder_ls_command,
                    stdin_descriptor=None,
                    timeout_seconds=_BUILDER_LS_TIMEOUT_SECONDS,
                    label="builder locality preflight",
                    image_state_uncertain=False,
                )
                builder_preflight_projection = _builder_locality_projection(
                    builder_preflight_result.stdout,
                    label="builder locality preflight",
                )
                base_command = _inspect_command(reference)
                base_result = run_bound(
                    base_command,
                    stdin_descriptor=None,
                    timeout_seconds=_INSPECT_TIMEOUT_SECONDS,
                    label="base image preflight",
                    image_state_uncertain=False,
                )
                base_inspect = _strict_inspect(base_result.stdout, label="base image inspect")
                base_image_id, matched_repo_digest = _validate_base_inspect(
                    base_inspect,
                    expected_reference=reference,
                    expected_platform=platform,
                )
                template = context_receipt["build_command_template"]
                if type(template) is not list or not template or template[-1] != "-":
                    _fail("sealed context build command template differs")
                with _exclusive_iidfile() as private:
                    iid_argument = f"--iidfile={private.file_path}"
                    build_command = tuple([*cast(list[str], template[:-1]), iid_argument, "-"])
                    context_descriptor = context_capability.duplicate_readonly_descriptor()
                    descriptor_primary_failure: BaseException | None = None
                    try:
                        build_started = True
                        build_result = run_bound(
                            build_command,
                            stdin_descriptor=context_descriptor,
                            timeout_seconds=timeout,
                            label="Docker Buildx build",
                            image_state_uncertain=True,
                        )
                        _require_context_streamed_to_eof(
                            context_descriptor,
                            expected_size=context_archive_size_bytes,
                        )
                    except BaseException as exc:
                        descriptor_primary_failure = exc
                    try:
                        os.close(context_descriptor)
                    except BaseException as exc:
                        if descriptor_primary_failure is not None:
                            descriptor_primary_failure.add_note(
                                "build-context descriptor cleanup also failed: "
                                f"{type(exc).__name__}: {exc}"
                            )
                        else:
                            raise ForagerMatchedV3CpuOciBuildExecutorError(
                                "build-context descriptor cleanup failed",
                                image_state_uncertain=True,
                            ) from exc
                    if descriptor_primary_failure is not None:
                        raise descriptor_primary_failure
                    image = _stable_iidfile(private)
                    builder_postflight_result = run_bound(
                        builder_ls_command,
                        stdin_descriptor=None,
                        timeout_seconds=_BUILDER_LS_TIMEOUT_SECONDS,
                        label="builder locality postflight",
                        image_state_uncertain=True,
                    )
                    builder_postflight_projection = _builder_locality_projection(
                        builder_postflight_result.stdout,
                        label="builder locality postflight",
                    )
                    if not hmac.compare_digest(
                        _canonical_json(builder_preflight_projection),
                        _canonical_json(builder_postflight_projection),
                    ):
                        _fail(
                            "builder locality normalized projection drifted across the build",
                            image_state_uncertain=True,
                        )
                    inspect_command = _inspect_command(image)
                    inspect_result = run_bound(
                        inspect_command,
                        stdin_descriptor=None,
                        timeout_seconds=_INSPECT_TIMEOUT_SECONDS,
                        label="built image inspect",
                        image_state_uncertain=True,
                    )
                    inspect = _strict_inspect(
                        inspect_result.stdout,
                        label="built image inspect",
                    )
                    user, working_directory = _validate_built_inspect(
                        inspect,
                        expected_image_id=image,
                        expected_platform=platform,
                        expected_labels=cast(
                            Mapping[str, Any],
                            context_receipt["expected_image_labels"],
                        ),
                    )
                    cli_state.reverify(image_state_uncertain=True)
                    toolchain_postflight = toolchain.reverify(image_state_uncertain=True)
                    receipt_bytes, receipt_sha256 = _build_receipt(
                        context_receipt=context_receipt,
                        context_receipt_bytes=context_receipt_bytes,
                        context_receipt_sha256=context_receipt_sha256,
                        base_command=base_command,
                        base_result=base_result,
                        base_inspect=base_inspect,
                        base_image_id=base_image_id,
                        matched_repo_digest=matched_repo_digest,
                        builder_ls_command=builder_ls_command,
                        builder_preflight_result=builder_preflight_result,
                        builder_preflight_projection=builder_preflight_projection,
                        builder_postflight_result=builder_postflight_result,
                        builder_postflight_projection=builder_postflight_projection,
                        toolchain_preflight=toolchain_preflight,
                        toolchain_postflight=toolchain_postflight,
                        build_command=build_command,
                        build_result=build_result,
                        timeout_seconds=timeout,
                        image_id=image,
                        inspect_command=inspect_command,
                        inspect_result=inspect_result,
                        inspect=inspect,
                        user=user,
                        working_directory=working_directory,
                    )
                    artifacts = CpuOciBuildExecutionArtifacts(
                        image,
                        receipt_bytes,
                        receipt_sha256,
                    )
    except BaseException as exc:
        if build_started:
            if (
                isinstance(exc, ForagerMatchedV3CpuOciBuildExecutorError)
                and exc.image_state_uncertain
            ):
                raise
            message = (
                str(exc)
                if isinstance(exc, ForagerMatchedV3CpuOciBuildExecutorError)
                else f"post-build execution or cleanup failed: {type(exc).__name__}: {exc}"
            )
            raise ForagerMatchedV3CpuOciBuildExecutorError(
                message,
                image_state_uncertain=True,
            ) from exc
        raise
    return artifacts


__all__ = [
    "BoundedProcessResult",
    "CPU_OCI_BUILD_EXECUTION_ACKNOWLEDGEMENT",
    "CPU_OCI_BUILD_EXECUTION_CLASSIFICATION",
    "CPU_OCI_BUILD_EXECUTION_RECEIPT_SCHEMA_VERSION",
    "CPU_OCI_BUILD_EXECUTION_STATUS",
    "CpuOciBuildExecutionArtifacts",
    "ForagerMatchedV3CpuOciBuildExecutorError",
    "MatchedV3CpuOciBuildAuthorization",
    "authorize_matched_v3_cpu_oci_build",
    "execute_matched_v3_cpu_oci_build",
    "parse_matched_v3_cpu_oci_build_execution_receipt",
]
