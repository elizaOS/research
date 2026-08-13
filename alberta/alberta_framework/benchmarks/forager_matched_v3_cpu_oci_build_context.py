"""Canonical sealed Docker build context for matched-v3 CPU qualification.

The producer accepts an independently pinned, already-valid CPU OCI build plan
and the three exact raw archives named by that plan.  It revalidates the plan,
cross-checks every supplied payload, emits exactly the planned twelve files as
canonical POSIX USTAR, fully replays the archive, and returns a PID-bound sealed
read-only content capability.  It does not extract, publish, build, or execute
anything.
"""

from __future__ import annotations

import copy
import fcntl
import hashlib
import hmac
import json
import os
import re
import stat
from collections.abc import Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from typing import Any, Final, Never, NoReturn, SupportsIndex, cast

from alberta_framework.benchmarks import (
    forager_matched_v3_cpu_oci_build_plan as plan_contract,
)

CPU_OCI_BUILD_CONTEXT_RECEIPT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.cpu_oci_build_context_receipt.v1"
)
CPU_OCI_BUILD_CONTEXT_STATUS: Final = (
    "retained_canonical_context_unexecuted_unqualified_non_authorizing"
)
CPU_OCI_BUILD_CONTEXT_CLASSIFICATION: Final = "sealed_exact_oci_build_context_content_capability"

_EXPECTED_GENERATED_PATHS: Final = frozenset(
    {
        "Dockerfile",
        "generated/archive-checksums.txt",
        "generated/elizaos-forager-sources.pth",
        "generated/materialize-wheelhouse.py",
        "generated/requirements.lock",
        "generated/runtime-inventory.json",
        "generated/verify-sources.py",
        "generated/verify-runtime.py",
        "generated/wheel-map.json",
    }
)
_EXPECTED_INPUT_ROLES: Final = {
    "inputs/external-foragax-source.v1.tar": "external_foragax_source",
    "inputs/local-alberta-source.v1.tar": "local_alberta_source",
    "inputs/wheelhouse.v1.tar": "wheelhouse_archive",
}
_EXPECTED_PATHS: Final = _EXPECTED_GENERATED_PATHS | frozenset(_EXPECTED_INPUT_ROLES)
_EXPECTED_MEMBER_COUNT: Final = 12
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
_BUILD_COMMAND_PREFIX: Final = [
    _BUILDX_PLUGIN_PATH,
    "--builder=default",
    "build",
    "--network=none",
    "--pull=false",
]
_BUILD_COMMAND_SUFFIX: Final = [
    "--file=Dockerfile",
    "--build-arg=SOURCE_DATE_EPOCH=0",
    "--build-arg=BUILDKIT_MULTI_PLATFORM=1",
    "--provenance=false",
    "--sbom=false",
    "--load",
    "--no-cache",
    "--progress=plain",
    "-",
]

_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_PATH_COMPONENT_RE: Final = re.compile(r"[A-Za-z0-9_.+-]{1,255}\Z")
_USTAR_BLOCK_BYTES: Final = 512
_USTAR_RECORD_BYTES: Final = 10_240
_READ_CHUNK_BYTES: Final = 1024 * 1024
_MAX_PLAN_BYTES: Final = 32 * 1024 * 1024
_MAX_RECEIPT_BYTES: Final = 8 * 1024 * 1024
_MAX_GENERATED_FILE_BYTES: Final = 2 * 1024 * 1024
_MAX_INPUT_FILE_BYTES: Final = 2 * 1024 * 1024 * 1024
_MAX_CONTEXT_BYTES: Final = 7 * 1024 * 1024 * 1024
_MAX_JSON_DEPTH: Final = 48
_MAX_JSON_NODES: Final = 250_000
_MAX_JSON_TEXT: Final = 512 * 1024
_MAX_INTEGER: Final = 2**63 - 1


def _execution_toolchain() -> dict[str, Any]:
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
            "fixed": {
                "DOCKER_HOST": "unix:///var/run/docker.sock",
                "LC_ALL": "C.UTF-8",
                "PATH": "/usr/bin:/bin",
            },
            "private_directories": {
                "BUILDX_CONFIG": "buildx",
                "DOCKER_CONFIG": "docker",
                "HOME": "home",
                "TMPDIR": "tmp",
                "XDG_CONFIG_HOME": "xdg",
            },
            "working_directory": "tmp",
        },
        "invocation": "direct_pinned_executables_via_open_readonly_descriptors",
    }


class ForagerMatchedV3CpuOciBuildContextError(RuntimeError):
    """The plan, context payload, USTAR, receipt, or capability failed closed."""


def _fail(message: str) -> NoReturn:
    raise ForagerMatchedV3CpuOciBuildContextError(message)


def _hash(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256(value: Any, *, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None or value == "0" * 64:
        _fail(f"{label} must be one nonzero lowercase SHA-256")
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
    _fail(f"context JSON contains float {value!r}")


def _raise_constant(value: str) -> Never:
    _fail(f"context JSON contains non-finite constant {value!r}")


def _parse_int(value: str) -> int:
    if len(value.lstrip("-")) > 19:
        _fail("context JSON integer exceeds its lexical bound")
    return int(value)


def _without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"context JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _assert_plain_json(value: Any) -> None:
    pending: list[tuple[Any, int]] = [(value, 0)]
    seen: set[int] = set()
    nodes = 0
    while pending:
        item, depth = pending.pop()
        nodes += 1
        if nodes > _MAX_JSON_NODES or depth > _MAX_JSON_DEPTH:
            _fail("context JSON exceeds its structure bound")
        if type(item) is str:
            if len(item) > _MAX_JSON_TEXT or any(ord(character) < 0x20 for character in item):
                _fail("context JSON contains an invalid or oversized string")
            continue
        if item is None or type(item) in {bool, int}:
            if type(item) is int and not -_MAX_INTEGER <= item <= _MAX_INTEGER:
                _fail("context JSON integer exceeds its value bound")
            continue
        if type(item) not in {dict, list}:
            _fail("context JSON contains a non-JSON value")
        identity = id(item)
        if identity in seen:
            _fail("context JSON contains a container alias")
        seen.add(identity)
        if type(item) is list:
            pending.extend((child, depth + 1) for child in item)
        else:
            for key, child in cast(dict[Any, Any], item).items():
                if type(key) is not str:
                    _fail("context JSON object key is not an exact string")
                pending.append((key, depth + 1))
                pending.append((child, depth + 1))


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    exact = copy.deepcopy(dict(value))
    _assert_plain_json(exact)
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
        raise ForagerMatchedV3CpuOciBuildContextError(
            "context JSON is not canonical finite ASCII"
        ) from exc
    if len(raw) > _MAX_RECEIPT_BYTES:
        _fail("context JSON exceeds its byte bound")
    return raw


def _strict_json(raw: bytes) -> dict[str, Any]:
    if type(raw) is not bytes or not raw or len(raw) > _MAX_RECEIPT_BYTES:
        _fail("context receipt must be nonempty exact bounded bytes")
    if not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
        _fail("context receipt must have one trailing newline")
    try:
        value = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_without_duplicates,
            parse_constant=_raise_constant,
            parse_float=_raise_float,
            parse_int=_parse_int,
        )
    except ForagerMatchedV3CpuOciBuildContextError:
        raise
    except (RecursionError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ForagerMatchedV3CpuOciBuildContextError(
            "context receipt is not bounded strict ASCII JSON"
        ) from exc
    if type(value) is not dict:
        _fail("context receipt root must be one object")
    result = cast(dict[str, Any], value)
    _assert_plain_json(result)
    if not hmac.compare_digest(_canonical_json(result), raw):
        _fail("context receipt bytes are not canonical")
    return result


def _body_sha256(value: Mapping[str, Any], field: str) -> str:
    body = copy.deepcopy(dict(value))
    supplied = _sha256(body.pop(field, None), label=f"context receipt {field}")
    observed = _hash(_canonical_json(body))
    if not hmac.compare_digest(supplied, observed):
        _fail(f"context receipt {field} differs from its canonical body")
    return observed


def _execution_projection_sha256(value: Mapping[str, Any]) -> str:
    return _hash(
        _canonical_json(
            {
                "base_image": value["base_image"],
                "build_command_template": value["build_command_template"],
                "execution_toolchain": value["execution_toolchain"],
                "expected_image_labels": value["expected_image_labels"],
            }
        )
    )


def _canonical_path(value: Any, *, label: str) -> str:
    path = _string(value, label=label)
    if path.startswith("/") or path.endswith("/") or "\\" in path:
        _fail(f"{label} is not a canonical relative path")
    try:
        encoded = path.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ForagerMatchedV3CpuOciBuildContextError(f"{label} is not ASCII") from exc
    if len(encoded) > 255:
        _fail(f"{label} exceeds the POSIX USTAR path bound")
    if any(
        component in {"", ".", ".."} or _PATH_COMPONENT_RE.fullmatch(component) is None
        for component in path.split("/")
    ):
        _fail(f"{label} contains an unsafe component")
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
    _fail(f"context path is not representable in POSIX USTAR: {path}")


def _ustar_octal(value: int, width: int, *, label: str) -> bytes:
    token = format(value, "o").encode("ascii")
    if value < 0 or len(token) > width - 1:
        _fail(f"{label} exceeds its POSIX USTAR field")
    return token.rjust(width - 1, b"0") + b"\0"


def _canonical_ustar_header(path: str, size: int) -> bytes:
    _canonical_path(path, label="context USTAR path")
    prefix, name = _split_ustar_path(path)
    header = bytearray(_USTAR_BLOCK_BYTES)
    header[: len(name)] = name
    header[100:108] = _ustar_octal(0o444, 8, label="context USTAR mode")
    header[108:116] = _ustar_octal(0, 8, label="context USTAR uid")
    header[116:124] = _ustar_octal(0, 8, label="context USTAR gid")
    header[124:136] = _ustar_octal(size, 12, label="context USTAR size")
    header[136:148] = _ustar_octal(0, 12, label="context USTAR mtime")
    header[148:156] = b"        "
    header[156:157] = b"0"
    header[257:263] = b"ustar\0"
    header[263:265] = b"00"
    header[345 : 345 + len(prefix)] = prefix
    checksum = format(sum(header), "06o").encode("ascii")
    if len(checksum) != 6:
        _fail("context USTAR checksum exceeds its field")
    header[148:156] = checksum + b"\0 "
    return bytes(header)


def _claims() -> dict[str, bool]:
    return {
        "build_execution_authority_granted": False,
        "container_image_built": False,
        "daemon_egress_isolation_attested": False,
        "execution_authority_granted": False,
        "image_identity_issued": False,
        "publication_authority_granted": False,
        "qualification_granted": False,
        "runtime_qualified": False,
        "scientific_evidence_created": False,
        "scientific_promotion_allowed": False,
    }


def _limitations() -> list[str]:
    return [
        "The retained USTAR is an exact content capability, not Docker build authority.",
        "The receipt and sealed descriptor grant no execution, publication, or qualification.",
        (
            "The command binds the local Unix daemon and reserved default builder, but daemon "
            "availability, exact topology, and egress remain external until execution."
        ),
        "No image, runtime behavior, scientific result, or promotion claim is created here.",
    ]


def _expected_labels(plan: Mapping[str, Any]) -> dict[str, str]:
    bindings = cast(Mapping[str, Any], plan["bindings"])
    wheelhouse = cast(Mapping[str, Any], bindings["wheelhouse"])
    runtime_lock = cast(Mapping[str, Any], bindings["runtime_lock"])
    raw_sources = bindings["sources"]
    if type(raw_sources) is not list or len(raw_sources) != 2:
        _fail("validated plan source bindings are unavailable")
    sources = {
        _string(item.get("role"), label="plan source role"): cast(Mapping[str, Any], item)
        for item in raw_sources
        if type(item) is dict
    }
    if frozenset(sources) != {"external_foragax", "local_alberta"}:
        _fail("validated plan source roles differ")
    base = cast(Mapping[str, Any], plan["base_image"])
    return {
        "io.elizaos.alberta.forager-matched-v3.base-manifest": _string(
            base["manifest_digest"], label="base manifest digest"
        ),
        "io.elizaos.alberta.forager-matched-v3.cas-manifest-sha256": _sha256(
            wheelhouse["cas_manifest_sha256"], label="CAS manifest"
        ),
        "io.elizaos.alberta.forager-matched-v3.external-source-sha256": _sha256(
            sources["external_foragax"]["archive_sha256"], label="external source archive"
        ),
        "io.elizaos.alberta.forager-matched-v3.local-source-sha256": _sha256(
            sources["local_alberta"]["archive_sha256"], label="local source archive"
        ),
        "io.elizaos.alberta.forager-matched-v3.runtime-lock-sha256": _sha256(
            runtime_lock["sha256"], label="runtime lock"
        ),
        "io.elizaos.alberta.forager-matched-v3.wheelhouse-sha256": _sha256(
            wheelhouse["archive_sha256"], label="wheelhouse archive"
        ),
    }


@dataclass(frozen=True, slots=True)
class _ContextMember:
    path: str
    payload: bytes

    def receipt_record(self) -> dict[str, Any]:
        return {
            "mode": "0444",
            "path": self.path,
            "sha256": _hash(self.payload),
            "size_bytes": len(self.payload),
        }


def _validated_members(
    plan: Mapping[str, Any],
    *,
    wheelhouse_archive_bytes: bytes,
    external_foragax_source_archive_bytes: bytes,
    local_alberta_source_archive_bytes: bytes,
) -> list[_ContextMember]:
    build = cast(Mapping[str, Any], plan["build"])
    raw_generated = build.get("generated_files")
    if type(raw_generated) is not list or len(raw_generated) != len(_EXPECTED_GENERATED_PATHS):
        _fail("build context does not contain the exact planned generated paths")
    generated: dict[str, bytes] = {}
    for index, item in enumerate(raw_generated):
        if type(item) is not dict:
            _fail(f"planned generated file {index} is not one object")
        record = cast(dict[str, Any], item)
        path = _canonical_path(record.get("path"), label=f"planned generated path {index}")
        content = record.get("content")
        if type(content) is not str:
            _fail(f"planned generated content {path} is not text")
        try:
            payload = content.encode("ascii")
        except UnicodeEncodeError as exc:
            raise ForagerMatchedV3CpuOciBuildContextError(
                f"planned generated content is not ASCII: {path}"
            ) from exc
        if (
            len(payload) > _MAX_GENERATED_FILE_BYTES
            or record.get("size_bytes") != len(payload)
            or record.get("sha256") != _hash(payload)
            or path in generated
        ):
            _fail(f"planned generated file identity differs: {path}")
        generated[path] = payload
    if frozenset(generated) != _EXPECTED_GENERATED_PATHS:
        _fail("build context does not contain the exact planned generated paths")

    supplied_inputs = {
        "inputs/external-foragax-source.v1.tar": external_foragax_source_archive_bytes,
        "inputs/local-alberta-source.v1.tar": local_alberta_source_archive_bytes,
        "inputs/wheelhouse.v1.tar": wheelhouse_archive_bytes,
    }
    if any(type(payload) is not bytes for payload in supplied_inputs.values()):
        _fail("every build-context archive must be exact bytes")
    raw_inputs = build.get("context_inputs")
    if type(raw_inputs) is not list or len(raw_inputs) != len(_EXPECTED_INPUT_ROLES):
        _fail("build context does not contain the exact planned input paths")
    planned_inputs: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(raw_inputs):
        if type(item) is not dict:
            _fail(f"planned context input {index} is not one object")
        record = cast(dict[str, Any], item)
        path = _canonical_path(record.get("path"), label=f"planned input path {index}")
        if path in planned_inputs:
            _fail("build context contains a duplicate planned input path")
        planned_inputs[path] = record
    if frozenset(planned_inputs) != frozenset(_EXPECTED_INPUT_ROLES):
        _fail("build context does not contain the exact planned input paths")
    for path, payload in supplied_inputs.items():
        record = planned_inputs[path]
        if (
            record.get("role") != _EXPECTED_INPUT_ROLES[path]
            or len(payload) > _MAX_INPUT_FILE_BYTES
            or record.get("size_bytes") != len(payload)
            or record.get("sha256") != _hash(payload)
        ):
            _fail(f"planned context input identity differs: {path}")

    payloads = generated | supplied_inputs
    if frozenset(payloads) != _EXPECTED_PATHS or len(payloads) != _EXPECTED_MEMBER_COUNT:
        _fail("build context does not contain the exact planned twelve paths")
    total = sum(len(payload) for payload in payloads.values())
    if total > _MAX_CONTEXT_BYTES:
        _fail("build context payload exceeds its global byte bound")
    return [_ContextMember(path, payloads[path]) for path in sorted(payloads, key=str.encode)]


def _write_all(descriptor: int, raw: bytes | memoryview) -> None:
    view = memoryview(raw)
    while view:
        try:
            count = os.write(descriptor, view[:_READ_CHUNK_BYTES])
        except OSError as exc:
            raise ForagerMatchedV3CpuOciBuildContextError(
                "canonical build-context write failed"
            ) from exc
        if count <= 0:
            _fail("canonical build-context write made no progress")
        view = view[count:]


class _HashingWriter:
    def __init__(self, descriptor: int) -> None:
        self.descriptor = descriptor
        self.digest = hashlib.sha256()
        self.size = 0

    def write(self, raw: bytes | memoryview) -> None:
        view = memoryview(raw)
        self.digest.update(view)
        self.size += len(view)
        if self.size > _MAX_CONTEXT_BYTES:
            _fail("canonical build-context USTAR exceeds its byte bound")
        _write_all(self.descriptor, view)


def _write_archive(descriptor: int, members: list[_ContextMember]) -> tuple[int, str]:
    writer = _HashingWriter(descriptor)
    for member in members:
        writer.write(_canonical_ustar_header(member.path, len(member.payload)))
        writer.write(member.payload)
        padding = (-len(member.payload)) % _USTAR_BLOCK_BYTES
        if padding:
            writer.write(bytes(padding))
    writer.write(bytes(2 * _USTAR_BLOCK_BYTES))
    record_padding = (-writer.size) % _USTAR_RECORD_BYTES
    if record_padding:
        writer.write(bytes(record_padding))
    return writer.size, writer.digest.hexdigest()


def _pread_exact(descriptor: int, size: int, offset: int, *, label: str) -> bytes:
    result = bytearray()
    position = offset
    while len(result) < size:
        try:
            block = os.pread(descriptor, min(size - len(result), _READ_CHUNK_BYTES), position)
        except OSError as exc:
            raise ForagerMatchedV3CpuOciBuildContextError(f"{label} read failed") from exc
        if not block:
            _fail(f"{label} is truncated")
        result.extend(block)
        position += len(block)
    return bytes(result)


def _hash_fd(descriptor: int, size: int) -> str:
    digest = hashlib.sha256()
    offset = 0
    while offset < size:
        block = _pread_exact(
            descriptor,
            min(size - offset, _READ_CHUNK_BYTES),
            offset,
            label="canonical build context",
        )
        digest.update(block)
        offset += len(block)
    return digest.hexdigest()


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


def _verify_archive_fd(
    descriptor: int,
    *,
    expected_size: int,
    expected_sha256: str,
    members: list[dict[str, Any]],
) -> None:
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode) or before.st_size != expected_size:
        _fail("canonical build-context descriptor metadata differs")
    offset = 0
    for record in members:
        path = cast(str, record["path"])
        size = cast(int, record["size_bytes"])
        header = _pread_exact(
            descriptor, _USTAR_BLOCK_BYTES, offset, label=f"context USTAR header {path}"
        )
        if not hmac.compare_digest(header, _canonical_ustar_header(path, size)):
            _fail(f"context USTAR header is not canonical: {path}")
        offset += _USTAR_BLOCK_BYTES
        digest = hashlib.sha256()
        remaining = size
        while remaining:
            block = _pread_exact(
                descriptor,
                min(remaining, _READ_CHUNK_BYTES),
                offset,
                label=f"context USTAR payload {path}",
            )
            digest.update(block)
            offset += len(block)
            remaining -= len(block)
        if not hmac.compare_digest(digest.hexdigest(), cast(str, record["sha256"])):
            _fail(f"context USTAR payload differs: {path}")
        padding = (-size) % _USTAR_BLOCK_BYTES
        if padding and any(
            _pread_exact(descriptor, padding, offset, label=f"context USTAR padding {path}")
        ):
            _fail(f"context USTAR padding is nonzero: {path}")
        offset += padding
    if any(
        _pread_exact(
            descriptor,
            2 * _USTAR_BLOCK_BYTES,
            offset,
            label="context USTAR end blocks",
        )
    ):
        _fail("context USTAR end blocks are nonzero")
    offset += 2 * _USTAR_BLOCK_BYTES
    final_size = offset + (-offset) % _USTAR_RECORD_BYTES
    if final_size != expected_size:
        _fail("context USTAR record padding length differs")
    if expected_size > offset and any(
        _pread_exact(
            descriptor,
            expected_size - offset,
            offset,
            label="context USTAR record padding",
        )
    ):
        _fail("context USTAR record padding is nonzero")
    after = os.fstat(descriptor)
    if not hmac.compare_digest(
        _hash_fd(descriptor, expected_size), expected_sha256
    ) or _stat_identity(before) != _stat_identity(after):
        _fail("canonical build-context digest or descriptor stability differs")


def _required_seals() -> int:
    values = [
        getattr(fcntl, name, None)
        for name in ("F_SEAL_WRITE", "F_SEAL_GROW", "F_SEAL_SHRINK", "F_SEAL_SEAL")
    ]
    if any(type(value) is not int for value in values):
        _fail("canonical build contexts require full memfd seals")
    return sum(cast(int, value) for value in values)


def _create_memfd() -> int:
    creator = getattr(os, "memfd_create", None)
    cloexec = getattr(os, "MFD_CLOEXEC", None)
    sealing = getattr(os, "MFD_ALLOW_SEALING", None)
    if creator is None or type(cloexec) is not int or type(sealing) is not int:
        _fail("canonical build contexts require sealed anonymous memfd support")
    descriptor = -1
    primary_failure: BaseException | None = None
    try:
        descriptor = creator("alberta-matched-v3-oci-context", cloexec | sealing)
        os.fchmod(descriptor, 0o600)
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 0
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or os.get_inheritable(descriptor)
        ):
            _fail("private build-context descriptor metadata differs")
        result = int(descriptor)
        descriptor = -1
        return result
    except OSError as exc:
        primary_failure = ForagerMatchedV3CpuOciBuildContextError(
            "private build-context descriptor cannot be created"
        )
        raise primary_failure from exc
    except BaseException as exc:
        primary_failure = exc
        raise
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except BaseException as cleanup_failure:
                if primary_failure is not None:
                    primary_failure.add_note(
                        "private build-context descriptor cleanup also failed: "
                        f"{type(cleanup_failure).__name__}: {cleanup_failure}"
                    )
                else:
                    raise ForagerMatchedV3CpuOciBuildContextError(
                        "private build-context descriptor cleanup failed"
                    ) from cleanup_failure


def _seal_and_reopen(descriptor: int, *, expected_size: int) -> int:
    readonly = -1
    primary_failure: BaseException | None = None
    try:
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o400)
        fcntl.fcntl(descriptor, fcntl.F_ADD_SEALS, _required_seals())
        before = os.fstat(descriptor)
        readonly = os.open(f"/proc/self/fd/{descriptor}", os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
        after = os.fstat(readonly)
        if (
            fcntl.fcntl(readonly, fcntl.F_GET_SEALS) & _required_seals() != _required_seals()
            or before.st_nlink != 0
            or before.st_size != expected_size
            or stat.S_IMODE(before.st_mode) != 0o400
            or (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino)
            or fcntl.fcntl(readonly, fcntl.F_GETFL) & os.O_ACCMODE != os.O_RDONLY
            or os.get_inheritable(readonly)
        ):
            _fail("sealed build-context descriptor metadata differs")
        result = readonly
        readonly = -1
        return result
    except OSError as exc:
        primary_failure = ForagerMatchedV3CpuOciBuildContextError(
            "canonical build context cannot be sealed read-only"
        )
        raise primary_failure from exc
    except BaseException as exc:
        primary_failure = exc
        raise
    finally:
        if readonly >= 0:
            try:
                os.close(readonly)
            except BaseException as cleanup_failure:
                if primary_failure is not None:
                    primary_failure.add_note(
                        "sealed build-context descriptor cleanup also failed: "
                        f"{type(cleanup_failure).__name__}: {cleanup_failure}"
                    )
                else:
                    raise ForagerMatchedV3CpuOciBuildContextError(
                        "sealed build-context descriptor cleanup failed"
                    ) from cleanup_failure


def _build_receipt(
    *,
    plan: Mapping[str, Any],
    plan_sha256: str,
    archive_size: int,
    archive_sha256: str,
    members: list[dict[str, Any]],
) -> tuple[bytes, str]:
    base = cast(Mapping[str, Any], plan["base_image"])
    build = cast(Mapping[str, Any], plan["build"])
    command = build.get("command")
    expected_command = [
        *_BUILD_COMMAND_PREFIX,
        f"--platform={_string(base['platform'], label='plan base platform')}",
        *_BUILD_COMMAND_SUFFIX,
    ]
    if command != expected_command:
        _fail("validated plan build command is not the exact sealed-stdin template")
    member_inventory_sha256 = _hash(_canonical_json({"members": members}))
    receipt: dict[str, Any] = {
        "archive": {
            "format": "canonical_posix_ustar_uncompressed",
            "member_count": len(members),
            "member_inventory_sha256": member_inventory_sha256,
            "member_mode": "0444",
            "sha256": archive_sha256,
            "size_bytes": archive_size,
            "uid_gid_mtime": 0,
        },
        "base_image": {
            "manifest_digest": _string(base["manifest_digest"], label="plan base manifest digest"),
            "platform": _string(base["platform"], label="plan base platform"),
            "reference": _string(base["reference"], label="plan base reference"),
        },
        "build_command_template": copy.deepcopy(expected_command),
        "claims": _claims(),
        "classification": CPU_OCI_BUILD_CONTEXT_CLASSIFICATION,
        "execution_toolchain": copy.deepcopy(plan["execution_toolchain"]),
        "execution_projection_sha256": "0" * 64,
        "expected_image_labels": _expected_labels(plan),
        "limitations": _limitations(),
        "members": copy.deepcopy(members),
        "plan": {
            "body_sha256": _sha256(plan["plan_body_sha256"], label="plan body"),
            "full_file_sha256": plan_sha256,
            "schema_version": _string(plan["schema_version"], label="plan schema"),
        },
        "receipt_body_sha256": "0" * 64,
        "schema_version": CPU_OCI_BUILD_CONTEXT_RECEIPT_SCHEMA_VERSION,
        "status": CPU_OCI_BUILD_CONTEXT_STATUS,
    }
    receipt["execution_projection_sha256"] = _execution_projection_sha256(receipt)
    body = copy.deepcopy(receipt)
    body.pop("receipt_body_sha256")
    receipt["receipt_body_sha256"] = _hash(_canonical_json(body))
    raw = _canonical_json(receipt)
    parse_matched_v3_cpu_oci_build_context_receipt(raw, expected_receipt_sha256=_hash(raw))
    return raw, _hash(raw)


def _validate_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    receipt = _exact(
        value,
        frozenset(
            {
                "archive",
                "base_image",
                "build_command_template",
                "claims",
                "classification",
                "execution_toolchain",
                "execution_projection_sha256",
                "expected_image_labels",
                "limitations",
                "members",
                "plan",
                "receipt_body_sha256",
                "schema_version",
                "status",
            }
        ),
        label="context receipt",
    )
    if (
        receipt["schema_version"] != CPU_OCI_BUILD_CONTEXT_RECEIPT_SCHEMA_VERSION
        or receipt["status"] != CPU_OCI_BUILD_CONTEXT_STATUS
        or receipt["classification"] != CPU_OCI_BUILD_CONTEXT_CLASSIFICATION
    ):
        _fail("context receipt schema, status, or classification differs")
    claims = _exact(receipt["claims"], frozenset(_claims()), label="context receipt claims")
    if claims != _claims() or any(value is not False for value in claims.values()):
        _fail("context receipt authority claim became true")
    if receipt["limitations"] != _limitations():
        _fail("context receipt limitations differ")
    if receipt["execution_toolchain"] != _execution_toolchain():
        _fail("context receipt execution toolchain or isolated environment differs")
    plan = _exact(
        receipt["plan"],
        frozenset({"body_sha256", "full_file_sha256", "schema_version"}),
        label="context receipt plan",
    )
    _sha256(plan["body_sha256"], label="context receipt plan body")
    _sha256(plan["full_file_sha256"], label="context receipt plan file")
    plan_schema = _string(plan["schema_version"], label="context receipt plan schema")
    if plan_schema != plan_contract.CPU_OCI_BUILD_PLAN_SCHEMA_VERSION:
        _fail("context receipt plan schema differs")
    base = _exact(
        receipt["base_image"],
        frozenset({"manifest_digest", "platform", "reference"}),
        label="context receipt base image",
    )
    manifest_digest = _string(base["manifest_digest"], label="context receipt base manifest")
    platform = _string(base["platform"], label="context receipt base platform")
    reference = _string(base["reference"], label="context receipt base reference")
    if (
        not manifest_digest.startswith("sha256:")
        or _SHA256_RE.fullmatch(manifest_digest.removeprefix("sha256:")) is None
        or manifest_digest != plan_contract.BASE_IMAGE_MANIFEST_DIGEST
        or platform != plan_contract.BASE_IMAGE_PLATFORM
        or reference != plan_contract.BASE_IMAGE_REFERENCE
    ):
        _fail("context receipt base digest reference or platform differs")
    expected_command = [
        *_BUILD_COMMAND_PREFIX,
        f"--platform={base['platform']}",
        *_BUILD_COMMAND_SUFFIX,
    ]
    if receipt["build_command_template"] != expected_command:
        _fail("context receipt build command is not the exact sealed-stdin template")
    labels = receipt["expected_image_labels"]
    if type(labels) is not dict or frozenset(labels) != frozenset(
        {
            "io.elizaos.alberta.forager-matched-v3.base-manifest",
            "io.elizaos.alberta.forager-matched-v3.cas-manifest-sha256",
            "io.elizaos.alberta.forager-matched-v3.external-source-sha256",
            "io.elizaos.alberta.forager-matched-v3.local-source-sha256",
            "io.elizaos.alberta.forager-matched-v3.runtime-lock-sha256",
            "io.elizaos.alberta.forager-matched-v3.wheelhouse-sha256",
        }
    ):
        _fail("context receipt expected image labels differ")
    for key, label_value in cast(dict[str, Any], labels).items():
        if key.endswith("base-manifest"):
            if label_value != base["manifest_digest"]:
                _fail("context receipt base label differs")
        else:
            _sha256(label_value, label=f"context receipt image label {key}")
    projection_sha256 = _sha256(
        receipt["execution_projection_sha256"],
        label="context receipt execution projection",
    )
    if projection_sha256 != _execution_projection_sha256(receipt):
        _fail("context receipt execution projection identity differs")
    raw_members = receipt["members"]
    if type(raw_members) is not list or len(raw_members) != _EXPECTED_MEMBER_COUNT:
        _fail("context receipt must bind exactly twelve members")
    members: list[dict[str, Any]] = []
    paths: list[str] = []
    for index, item in enumerate(raw_members):
        record = _exact(
            item,
            frozenset({"mode", "path", "sha256", "size_bytes"}),
            label=f"context receipt member {index}",
        )
        path = _canonical_path(record["path"], label=f"context receipt member {index} path")
        if record["mode"] != "0444":
            _fail("context receipt member mode differs")
        _sha256(record["sha256"], label=f"context receipt member {index} digest")
        _integer(
            record["size_bytes"],
            label=f"context receipt member {index} size",
            maximum=_MAX_INPUT_FILE_BYTES,
        )
        members.append(record)
        paths.append(path)
    if paths != sorted(paths, key=str.encode) or frozenset(paths) != _EXPECTED_PATHS:
        _fail("context receipt member paths are not the exact sorted planned paths")
    member_by_path = {cast(str, member["path"]): member for member in members}
    payload_label_paths = {
        "io.elizaos.alberta.forager-matched-v3.external-source-sha256": (
            "inputs/external-foragax-source.v1.tar"
        ),
        "io.elizaos.alberta.forager-matched-v3.local-source-sha256": (
            "inputs/local-alberta-source.v1.tar"
        ),
        "io.elizaos.alberta.forager-matched-v3.wheelhouse-sha256": ("inputs/wheelhouse.v1.tar"),
    }
    if any(
        cast(dict[str, Any], labels)[label] != member_by_path[path]["sha256"]
        for label, path in payload_label_paths.items()
    ):
        _fail("context receipt source or wheelhouse label differs from its exact member")
    archive = _exact(
        receipt["archive"],
        frozenset(
            {
                "format",
                "member_count",
                "member_inventory_sha256",
                "member_mode",
                "sha256",
                "size_bytes",
                "uid_gid_mtime",
            }
        ),
        label="context receipt archive",
    )
    if (
        archive["format"] != "canonical_posix_ustar_uncompressed"
        or archive["member_count"] != _EXPECTED_MEMBER_COUNT
        or archive["member_mode"] != "0444"
        or archive["uid_gid_mtime"] != 0
        or archive["member_inventory_sha256"] != _hash(_canonical_json({"members": members}))
    ):
        _fail("context receipt archive inventory differs")
    _sha256(archive["sha256"], label="context receipt archive")
    size = _integer(
        archive["size_bytes"],
        label="context receipt archive size",
        minimum=_USTAR_RECORD_BYTES,
        maximum=_MAX_CONTEXT_BYTES,
    )
    if size % _USTAR_RECORD_BYTES:
        _fail("context receipt archive size is not a USTAR record multiple")
    _body_sha256(receipt, "receipt_body_sha256")
    return copy.deepcopy(receipt)


def parse_matched_v3_cpu_oci_build_context_receipt(
    raw: bytes,
    *,
    expected_receipt_sha256: str,
) -> dict[str, Any]:
    """Parse one exact canonical receipt under a caller-carried full digest."""

    expected = _sha256(expected_receipt_sha256, label="expected context receipt")
    if type(raw) is not bytes or not hmac.compare_digest(_hash(raw), expected):
        _fail("context receipt full-file SHA-256 differs")
    return _validate_receipt(_strict_json(raw))


_CAPABILITY_TOKEN: Final = object()


class RetainedMatchedV3CpuOciBuildContext:
    """PID-bound sealed read-only capability for one exact Docker context USTAR."""

    __slots__ = (
        "_archive_sha256",
        "_archive_size",
        "_descriptor",
        "_device",
        "_execution_projection_sha256",
        "_inode",
        "_owner_pid",
        "_plan_sha256",
        "_receipt_bytes",
        "_receipt_sha256",
    )

    def __init__(
        self,
        token: object,
        descriptor: int,
        device: int,
        inode: int,
        archive_size: int,
        archive_sha256: str,
        plan_sha256: str,
        execution_projection_sha256: str,
        receipt_bytes: bytes,
        receipt_sha256: str,
    ) -> None:
        if token is not _CAPABILITY_TOKEN:
            raise TypeError("retained OCI build contexts require the producer context")
        self._descriptor = descriptor
        self._device = device
        self._inode = inode
        self._archive_size = archive_size
        self._archive_sha256 = archive_sha256
        self._plan_sha256 = plan_sha256
        self._execution_projection_sha256 = execution_projection_sha256
        self._receipt_bytes = receipt_bytes
        self._receipt_sha256 = receipt_sha256
        self._owner_pid = os.getpid()

    def __reduce__(self) -> Never:
        raise TypeError("retained OCI build contexts cannot be serialized")

    def __reduce_ex__(self, protocol: SupportsIndex) -> Never:
        del protocol
        raise TypeError("retained OCI build contexts cannot be serialized")

    def __copy__(self) -> Never:
        raise TypeError("retained OCI build contexts cannot be copied")

    def __deepcopy__(self, memo: object) -> Never:
        del memo
        raise TypeError("retained OCI build contexts cannot be copied")

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
            _fail("retained OCI build context is invalid after a PID change")
        descriptor = self._descriptor
        if descriptor < 0:
            _fail("retained OCI build context is closed")
        try:
            metadata = os.fstat(descriptor)
            flags = fcntl.fcntl(descriptor, fcntl.F_GETFL)
            descriptor_flags = fcntl.fcntl(descriptor, fcntl.F_GETFD)
            seals = fcntl.fcntl(descriptor, fcntl.F_GET_SEALS)
        except OSError as exc:
            self._invalidate(close_if_owned=True)
            raise ForagerMatchedV3CpuOciBuildContextError(
                "retained OCI build context became inaccessible"
            ) from exc
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 0
            or stat.S_IMODE(metadata.st_mode) != 0o400
            or metadata.st_size != self._archive_size
            or (metadata.st_dev, metadata.st_ino) != (self._device, self._inode)
            or flags & os.O_ACCMODE != os.O_RDONLY
            or descriptor_flags & fcntl.FD_CLOEXEC == 0
            or seals & _required_seals() != _required_seals()
            or os.get_inheritable(descriptor)
        ):
            same = (metadata.st_dev, metadata.st_ino) == (self._device, self._inode)
            self._invalidate(close_if_owned=same)
            _fail("retained OCI build-context descriptor identity drifted")
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
    def archive_size_bytes(self) -> int:
        self._require_active()
        return self._archive_size

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
    def plan_sha256(self) -> str:
        self._require_active()
        return self._plan_sha256

    @property
    def execution_projection_sha256(self) -> str:
        self._require_active()
        return self._execution_projection_sha256

    @property
    def member_count(self) -> int:
        self._require_active()
        return cast(int, cast(Mapping[str, Any], self.receipt()["archive"])["member_count"])

    def receipt(self) -> dict[str, Any]:
        self._require_active()
        return parse_matched_v3_cpu_oci_build_context_receipt(
            self._receipt_bytes,
            expected_receipt_sha256=self._receipt_sha256,
        )

    def reverify(self) -> dict[str, Any]:
        descriptor = self._require_active()
        try:
            receipt = self.receipt()
            archive = cast(Mapping[str, Any], receipt["archive"])
            plan = cast(Mapping[str, Any], receipt["plan"])
            if plan["full_file_sha256"] != self._plan_sha256:
                _fail("retained OCI build-context plan identity differs")
            if (
                receipt["execution_projection_sha256"] != self._execution_projection_sha256
                or _execution_projection_sha256(receipt) != self._execution_projection_sha256
            ):
                _fail("retained OCI build-context execution projection or label inventory differs")
            if (
                archive["sha256"] != self._archive_sha256
                or archive["size_bytes"] != self._archive_size
            ):
                _fail("retained OCI build-context archive identity differs")
            _verify_archive_fd(
                descriptor,
                expected_size=cast(int, archive["size_bytes"]),
                expected_sha256=cast(str, archive["sha256"]),
                members=cast(list[dict[str, Any]], receipt["members"]),
            )
            self._require_active()
            return receipt
        except BaseException:
            self._invalidate(close_if_owned=True)
            raise

    def read_context_bytes(self) -> bytes:
        """Return exact context bytes after a complete receipt-bound replay."""

        self.reverify()
        descriptor = self._require_active()
        try:
            raw = _pread_exact(
                descriptor,
                self._archive_size,
                0,
                label="retained OCI build context",
            )
            if not hmac.compare_digest(_hash(raw), self._archive_sha256):
                _fail("retained OCI build-context byte read differs")
            return raw
        except BaseException:
            self._invalidate(close_if_owned=True)
            raise

    def duplicate_readonly_descriptor(self) -> int:
        """Return a caller-owned, offset-zero descriptor for one exact stream."""

        source = self._require_active()
        duplicate = -1
        try:
            duplicate = os.open(
                f"/proc/self/fd/{source}", os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
            )
            metadata = os.fstat(duplicate)
            if (
                (metadata.st_dev, metadata.st_ino) != (self._device, self._inode)
                or metadata.st_size != self._archive_size
                or fcntl.fcntl(duplicate, fcntl.F_GETFL) & os.O_ACCMODE != os.O_RDONLY
                or fcntl.fcntl(duplicate, fcntl.F_GET_SEALS) & _required_seals()
                != _required_seals()
                or os.get_inheritable(duplicate)
                or os.lseek(duplicate, 0, os.SEEK_CUR) != 0
            ):
                _fail("duplicated OCI build-context descriptor identity differs")
            return duplicate
        except BaseException:
            if duplicate >= 0:
                os.close(duplicate)
            raise

    def close(self) -> None:
        self._invalidate(close_if_owned=True)


@contextmanager
def _retain(
    *,
    plan_bytes: bytes,
    expected_plan_sha256: str,
    wheelhouse_archive_bytes: bytes,
    external_foragax_source_archive_bytes: bytes,
    local_alberta_source_archive_bytes: bytes,
) -> Iterator[RetainedMatchedV3CpuOciBuildContext]:
    expected_plan = _sha256(expected_plan_sha256, label="expected OCI build plan")
    if (
        type(plan_bytes) is not bytes
        or not plan_bytes
        or len(plan_bytes) > _MAX_PLAN_BYTES
        or not hmac.compare_digest(_hash(plan_bytes), expected_plan)
    ):
        _fail("OCI build-plan full-file identity differs")
    try:
        plan = plan_contract.parse_cpu_oci_build_plan(
            plan_bytes,
            expected_file_sha256=expected_plan,
        )
    except ForagerMatchedV3CpuOciBuildContextError:
        raise
    except (TypeError, ValueError) as exc:
        raise ForagerMatchedV3CpuOciBuildContextError(
            "OCI build plan failed its authoritative parser"
        ) from exc
    members = _validated_members(
        plan,
        wheelhouse_archive_bytes=wheelhouse_archive_bytes,
        external_foragax_source_archive_bytes=external_foragax_source_archive_bytes,
        local_alberta_source_archive_bytes=local_alberta_source_archive_bytes,
    )
    member_records = [member.receipt_record() for member in members]
    writable = -1
    readonly = -1
    retained: RetainedMatchedV3CpuOciBuildContext | None = None
    primary_failure: BaseException | None = None
    try:
        writable = _create_memfd()
        archive_size, archive_sha256 = _write_archive(writable, members)
        _verify_archive_fd(
            writable,
            expected_size=archive_size,
            expected_sha256=archive_sha256,
            members=member_records,
        )
        receipt_bytes, receipt_sha256 = _build_receipt(
            plan=plan,
            plan_sha256=expected_plan,
            archive_size=archive_size,
            archive_sha256=archive_sha256,
            members=member_records,
        )
        created_receipt = parse_matched_v3_cpu_oci_build_context_receipt(
            receipt_bytes,
            expected_receipt_sha256=receipt_sha256,
        )
        readonly = _seal_and_reopen(writable, expected_size=archive_size)
        os.close(writable)
        writable = -1
        _verify_archive_fd(
            readonly,
            expected_size=archive_size,
            expected_sha256=archive_sha256,
            members=member_records,
        )
        metadata = os.fstat(readonly)
        retained = RetainedMatchedV3CpuOciBuildContext(
            _CAPABILITY_TOKEN,
            readonly,
            metadata.st_dev,
            metadata.st_ino,
            archive_size,
            archive_sha256,
            expected_plan,
            _execution_projection_sha256(created_receipt),
            receipt_bytes,
            receipt_sha256,
        )
        readonly = -1
        retained.reverify()
        # The sealed memfd and canonical receipt are now the sole retained context.
        # Drop every potentially large caller payload and member wrapper before the
        # context manager yields for a long-running Docker build.
        del plan_bytes
        del wheelhouse_archive_bytes
        del external_foragax_source_archive_bytes
        del local_alberta_source_archive_bytes
        del plan
        del members
        del member_records
        del created_receipt
        yield retained
    except BaseException as exc:
        primary_failure = exc
        raise
    finally:
        cleanup_failures: list[BaseException] = []
        if retained is not None:
            try:
                retained.close()
            except BaseException as exc:
                cleanup_failures.append(exc)
        if readonly >= 0:
            try:
                os.close(readonly)
            except BaseException as exc:
                cleanup_failures.append(exc)
        if writable >= 0:
            try:
                os.close(writable)
            except BaseException as exc:
                cleanup_failures.append(exc)
        if cleanup_failures:
            if primary_failure is not None:
                for cleanup_failure in cleanup_failures:
                    primary_failure.add_note(
                        "OCI build-context cleanup also failed: "
                        f"{type(cleanup_failure).__name__}: {cleanup_failure}"
                    )
            else:
                failure = ForagerMatchedV3CpuOciBuildContextError(
                    "OCI build-context cleanup failed"
                )
                for cleanup_failure in cleanup_failures[1:]:
                    failure.add_note(
                        "additional OCI build-context cleanup failure: "
                        f"{type(cleanup_failure).__name__}: {cleanup_failure}"
                    )
                raise failure from cleanup_failures[0]


def retain_matched_v3_cpu_oci_build_context(
    *,
    plan_bytes: bytes,
    expected_plan_sha256: str,
    wheelhouse_archive_bytes: bytes,
    external_foragax_source_archive_bytes: bytes,
    local_alberta_source_archive_bytes: bytes,
) -> AbstractContextManager[RetainedMatchedV3CpuOciBuildContext]:
    """Return a bounded context manager retaining one exact sealed USTAR."""

    return _retain(
        plan_bytes=plan_bytes,
        expected_plan_sha256=expected_plan_sha256,
        wheelhouse_archive_bytes=wheelhouse_archive_bytes,
        external_foragax_source_archive_bytes=external_foragax_source_archive_bytes,
        local_alberta_source_archive_bytes=local_alberta_source_archive_bytes,
    )


__all__ = [
    "CPU_OCI_BUILD_CONTEXT_CLASSIFICATION",
    "CPU_OCI_BUILD_CONTEXT_RECEIPT_SCHEMA_VERSION",
    "CPU_OCI_BUILD_CONTEXT_STATUS",
    "ForagerMatchedV3CpuOciBuildContextError",
    "RetainedMatchedV3CpuOciBuildContext",
    "parse_matched_v3_cpu_oci_build_context_receipt",
    "retain_matched_v3_cpu_oci_build_context",
]
