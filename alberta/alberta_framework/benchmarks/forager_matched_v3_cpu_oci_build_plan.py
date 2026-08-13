"""Pure-content offline OCI build plan for matched-v3 CPU qualification.

This module accepts already-issued runtime-lock content, the exact canonical
wheelhouse USTAR, and two separately pinned canonical source USTAR payloads with
their exact producer receipts.  It rehashes and structurally replays those
bytes, cross-binds the wheel closure and source provenance, and returns canonical
JSON containing an exact Dockerfile and build-context files.  It performs no
filesystem, network, subprocess, container, install, import, qualification,
evidence, or publication operation.

In particular, a source snapshot manifest or tree digest is provenance only.
It is never accepted in place of the corresponding source archive bytes.
"""

from __future__ import annotations

import copy
import hashlib
import hmac
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, Never, cast

from alberta_framework.benchmarks import (
    forager_matched_v3_cpu_runtime_lock as runtime_lock_contract,
)
from alberta_framework.benchmarks import (
    forager_matched_v3_cpu_runtime_lock_issuer as runtime_lock_issuer,
)
from alberta_framework.benchmarks import (
    forager_matched_v3_cpu_wheelhouse as wheelhouse_contract,
)
from alberta_framework.benchmarks import (
    forager_matched_v3_external_materialization as external_materialization_contract,
)
from alberta_framework.benchmarks import (
    forager_matched_v3_external_source_publication as external_source_publication_contract,
)
from alberta_framework.benchmarks import (
    forager_matched_v3_local_source_bundle as local_source_bundle_contract,
)

CPU_OCI_BUILD_PLAN_SCHEMA_VERSION: Final = "alberta.forager_matched_v3.cpu_oci_build_plan.v1"
CPU_OCI_BUILD_PLAN_STATUS: Final = "offline_build_plan_unexecuted_non_authorizing"
CPU_OCI_BUILD_PLAN_CLASSIFICATION: Final = "pure_content_oci_plan_non_authorizing"

BASE_IMAGE_REPOSITORY: Final = "docker.io/library/python"
BASE_IMAGE_INFORMATIONAL_TAG: Final = "3.12.3-slim-bookworm"
BASE_IMAGE_PLATFORM: Final = "linux/amd64"
BASE_IMAGE_MANIFEST_DIGEST: Final = (
    "sha256:fd3817f3a855f6c2ada16ac9468e5ee93e361005bd226fd5a5ee1a504e038c84"
)
BASE_IMAGE_REFERENCE: Final = f"{BASE_IMAGE_REPOSITORY}@{BASE_IMAGE_MANIFEST_DIGEST}"

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
_BUILDX_BUILDER_ARGUMENT: Final = "--builder=default"
_BUILD_COMMAND_PREFIX: Final = [
    _BUILDX_PLUGIN_PATH,
    _BUILDX_BUILDER_ARGUMENT,
    "build",
]

_WHEELHOUSE_CONTEXT_PATH: Final = "inputs/wheelhouse.v1.tar"
_EXTERNAL_SOURCE_CONTEXT_PATH: Final = "inputs/external-foragax-source.v1.tar"
_LOCAL_SOURCE_CONTEXT_PATH: Final = "inputs/local-alberta-source.v1.tar"
_EXTERNAL_SOURCE_ROOT: Final = "/opt/elizaos/src/external-foragax"
_LOCAL_SOURCE_ROOT: Final = "/opt/elizaos/src/local-alberta"
_WHEEL_CAS_ROOT: Final = "/opt/elizaos/wheel-cas"
_WHEEL_INSTALL_ROOT: Final = "/opt/elizaos/wheelhouse"

_MAX_JSON_BYTES: Final = 32 * 1024 * 1024
_MAX_JSON_DEPTH: Final = 48
_MAX_JSON_NODES: Final = 1_000_000
_MAX_TEXT_LENGTH: Final = 512 * 1024
_MAX_INTEGER: Final = 2**63 - 1
_MAX_SOURCE_ARCHIVE_BYTES: Final = 2 * 1024 * 1024 * 1024
_MAX_SOURCE_RECEIPT_BYTES: Final = 64 * 1024 * 1024
_MAX_WHEELHOUSE_ARCHIVE_BYTES: Final = 2 * 1024 * 1024 * 1024
_MAX_SOURCE_MEMBERS: Final = 20_000
_MAX_SOURCE_MEMBER_BYTES: Final = 256 * 1024 * 1024
_USTAR_BLOCK_BYTES: Final = 512
_USTAR_RECORD_BYTES: Final = 10_240

_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_GIT_SHA1_RE: Final = re.compile(r"[0-9a-f]{40}\Z")
_SOURCE_PATH_COMPONENT_RE: Final = re.compile(r"[A-Za-z0-9_.+-]{1,255}\Z")
_BUILD_FILE_PATH_RE: Final = re.compile(r"(?:Dockerfile|generated/[A-Za-z0-9_.+-]+)\Z")
_DISTRIBUTION_NAME_RE: Final = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\Z")
_CAS_KEY_RE: Final = re.compile(
    r"sha256/(?P<prefix>[0-9a-f]{2})/(?P<sha256>[0-9a-f]{64})/"
    r"(?P<filename>[^/]+\.whl)\Z"
)
_PYPI_WHEEL_URL_RE: Final = re.compile(
    r"https://files\.pythonhosted\.org/packages/[0-9a-f]{2}/[0-9a-f]{2}/"
    r"[0-9a-f]{60}/(?P<filename>[A-Za-z0-9_.+-]+\.whl)\Z"
)
_REQUIRED_RUNTIME_IMPORTS: Final = [
    "alberta_framework",
    "chex",
    "distrax",
    "flax",
    "gymnax",
    "haiku",
    "jax",
    "jaxlib",
    "optax",
]


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


_REQUIRED_FUNCTIONAL_PROBES: Final = [
    "flax.linen.Dense.init_apply_jit",
    "haiku.Linear.init_apply_jit",
]
_EXTERNAL_MATERIALIZATION_IDENTITY_SHA256: Final = (
    external_materialization_contract.PINNED_EXTERNAL_MATERIALIZATION_IDENTITY_SHA256
)
_EXTERNAL_PUBLICATION_DESCRIPTOR_SHA256: Final = (
    external_source_publication_contract.EXTERNAL_SOURCE_PUBLICATION_CONTRACT_DESCRIPTOR_SHA256
)
_LOCAL_SOURCE_BUNDLE_DESCRIPTOR_SHA256: Final = (
    local_source_bundle_contract.LOCAL_SOURCE_BUNDLE_DESCRIPTOR_SHA256
)


class ForagerMatchedV3CpuOciBuildPlanError(ValueError):
    """An OCI build-plan input, binding, or canonical artifact failed closed."""


@dataclass(frozen=True, slots=True)
class CanonicalSourceBundleInput:
    """Exact source bytes plus caller-pinned payload and provenance identities.

    The exact producer receipt is mandatory and independently digest-pinned.
    ``source_manifest_sha256`` and ``source_tree_sha256`` are caller-carried
    provenance pins: the external role binds them to the materializer manifest
    and tree while the local role binds them to the source-snapshot manifest and
    tree.  ``staging_manifest_sha256`` is mandatory for the external role and
    must be ``None`` for the local role.  No provenance hash can stand in for the
    independently replayed ``archive_bytes``.
    """

    archive_bytes: bytes
    expected_archive_sha256: str
    expected_archive_size_bytes: int
    expected_member_count: int
    receipt_bytes: bytes
    expected_receipt_sha256: str
    source_manifest_sha256: str
    source_tree_sha256: str
    staging_manifest_sha256: str | None


@dataclass(frozen=True, slots=True)
class CpuOciBuildPlanArtifacts:
    """Detached canonical plan bytes emitted by the pure-content builder."""

    plan_bytes: bytes
    plan_sha256: str


@dataclass(frozen=True, slots=True)
class _ValidatedSourceBundle:
    role: str
    context_path: str
    archive_sha256: str
    archive_size_bytes: int
    member_count: int
    member_inventory_sha256: str
    receipt_sha256: str
    receipt_size_bytes: int
    producer_descriptor_sha256: str
    materialization_identity_sha256: str | None
    source_manifest_sha256: str
    source_tree_sha256: str
    staging_manifest_sha256: str | None
    commit_git_sha1: str | None
    tree_git_sha1: str | None
    members: tuple[dict[str, Any], ...]


def _fail(message: str) -> Never:
    raise ForagerMatchedV3CpuOciBuildPlanError(message)


def _raise_float(value: str) -> Never:
    _fail(f"OCI plan JSON contains a float {value!r}")


def _raise_constant(value: str) -> Never:
    _fail(f"OCI plan JSON contains a non-finite constant {value!r}")


def _parse_int(value: str) -> int:
    if len(value.lstrip("-")) > 19:
        _fail("OCI plan JSON integer exceeds its lexical bound")
    return int(value)


def _without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"OCI plan JSON contains duplicate key {key!r}")
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
            _fail("OCI plan JSON exceeds its structure bound")
        if type(item) is str:
            if len(item) > _MAX_TEXT_LENGTH or any(
                ord(character) < 0x20 and character not in "\n\t" or ord(character) > 0x7E
                for character in item
            ):
                _fail("OCI plan JSON strings must be bounded printable ASCII text")
            continue
        if item is None or type(item) in {bool, int}:
            if type(item) is int and not -_MAX_INTEGER <= item <= _MAX_INTEGER:
                _fail("OCI plan JSON integer exceeds its value bound")
            continue
        if type(item) not in {dict, list}:
            _fail("OCI plan JSON contains a non-JSON value")
        identity = id(item)
        if identity in seen:
            _fail("OCI plan JSON contains a container alias")
        seen.add(identity)
        if type(item) is list:
            pending.extend((child, depth + 1) for child in item)
        else:
            for key, child in cast(dict[Any, Any], item).items():
                if type(key) is not str:
                    _fail("OCI plan JSON object key is not an exact string")
                pending.append((key, depth + 1))
                pending.append((child, depth + 1))


def _canonical_compact(value: Any) -> bytes:
    _assert_plain_json(value)
    try:
        result = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (OverflowError, RecursionError, TypeError, UnicodeError, ValueError) as exc:
        raise ForagerMatchedV3CpuOciBuildPlanError(
            "OCI build-plan value is not canonical finite ASCII JSON"
        ) from exc
    if len(result) > _MAX_JSON_BYTES:
        _fail("OCI build-plan artifact exceeds its byte bound")
    return result


def _canonical_json(value: Any) -> bytes:
    if type(value) is not dict:
        _fail("canonical OCI build-plan root must be one plain object")
    return _canonical_compact(value) + b"\n"


def _strict_json(raw: bytes, *, label: str) -> dict[str, Any]:
    if type(raw) is not bytes or not raw or len(raw) > _MAX_JSON_BYTES:
        _fail(f"{label} must be nonempty exact bytes within the byte bound")
    if not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
        _fail(f"{label} must have one canonical trailing newline")
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ForagerMatchedV3CpuOciBuildPlanError(f"{label} must be ASCII") from exc
    try:
        value = json.loads(
            text,
            object_pairs_hook=_without_duplicate_keys,
            parse_constant=_raise_constant,
            parse_float=_raise_float,
            parse_int=_parse_int,
        )
    except ForagerMatchedV3CpuOciBuildPlanError:
        raise
    except (RecursionError, json.JSONDecodeError, ValueError) as exc:
        raise ForagerMatchedV3CpuOciBuildPlanError(f"{label} is not bounded strict JSON") from exc
    if type(value) is not dict:
        _fail(f"{label} root must be one object")
    result = cast(dict[str, Any], value)
    _assert_plain_json(result)
    if not hmac.compare_digest(_canonical_json(result), raw):
        _fail(f"{label} bytes are not canonical")
    return result


def _exact(value: Any, fields: frozenset[str], *, label: str) -> dict[str, Any]:
    if type(value) is not dict or frozenset(value) != fields:
        _fail(f"{label} fields are not exact")
    return cast(dict[str, Any], value)


def _string(value: Any, *, label: str) -> str:
    if type(value) is not str or not value:
        _fail(f"{label} must be one nonempty exact string")
    return value


def _integer(value: Any, *, label: str, minimum: int = 0, maximum: int = _MAX_INTEGER) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        _fail(f"{label} must be one bounded exact integer")
    return value


def _sha256(value: Any, *, label: str) -> str:
    result = _string(value, label=label)
    if _SHA256_RE.fullmatch(result) is None or result == "0" * 64:
        _fail(f"{label} must be one nonzero lowercase SHA-256")
    return result


def _git_sha1(value: Any, *, label: str) -> str:
    result = _string(value, label=label)
    if _GIT_SHA1_RE.fullmatch(result) is None or result == "0" * 40:
        _fail(f"{label} must be one nonzero lowercase Git SHA-1")
    return result


def _hash(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _wheel_filename_matches_identity(filename: str, name: str, version: str) -> bool:
    try:
        prefix, _python_tag, _abi_tag, _platform_tag = filename.removesuffix(".whl").rsplit("-", 3)
    except ValueError:
        return False
    expected = f"{name.replace('-', '_')}-{version.replace('-', '_')}"
    return (
        prefix.casefold() == expected.casefold()
        or re.fullmatch(
            re.escape(expected) + r"-[0-9][A-Za-z0-9_.]*",
            prefix,
            re.IGNORECASE,
        )
        is not None
    )


def _body_sha256(value: Mapping[str, Any], field: str, *, label: str) -> str:
    body = copy.deepcopy(dict(value))
    supplied = _sha256(body.pop(field, None), label=f"{label} {field}")
    observed = _hash(_canonical_json(body))
    if not hmac.compare_digest(supplied, observed):
        _fail(f"{label} {field} differs from its canonical body")
    return observed


def _canonical_source_path(value: str, *, label: str) -> str:
    if (
        not value
        or value.startswith("/")
        or value.endswith("/")
        or "\\" in value
        or len(value.encode("ascii", errors="ignore")) != len(value)
        or len(value.encode("ascii")) > 255
    ):
        _fail(f"{label} is not a canonical relative ASCII source path")
    parts = value.split("/")
    if any(
        part in {"", ".", ".."} or _SOURCE_PATH_COMPONENT_RE.fullmatch(part) is None
        for part in parts
    ):
        _fail(f"{label} contains an unsafe source path component")
    if any(
        part in {"__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache"} for part in parts
    ) or value.endswith((".pyc", ".pyo")):
        _fail(f"{label} contains an excluded cache artifact")
    return value


def _split_ustar_path(path: str) -> tuple[bytes, bytes]:
    encoded = path.encode("ascii")
    if len(encoded) <= 100:
        return b"", encoded
    positions = [index for index, byte in enumerate(encoded) if byte == ord("/")]
    for index in reversed(positions):
        prefix = encoded[:index]
        name = encoded[index + 1 :]
        if prefix and name and len(prefix) <= 155 and len(name) <= 100:
            return prefix, name
    _fail("source path is not exactly representable in POSIX USTAR")


def _ustar_octal(value: int, width: int, *, label: str) -> bytes:
    token = format(value, "o").encode("ascii")
    if value < 0 or len(token) > width - 1:
        _fail(f"{label} exceeds its POSIX USTAR field")
    return token.rjust(width - 1, b"0") + b"\0"


def _canonical_ustar_header(path: str, size: int, mode: int) -> bytes:
    _canonical_source_path(path, label="USTAR path")
    if mode not in {0o444, 0o555}:
        _fail("USTAR member mode must be exactly 0444 or 0555")
    prefix, name = _split_ustar_path(path)
    header = bytearray(_USTAR_BLOCK_BYTES)
    header[0 : len(name)] = name
    header[100:108] = _ustar_octal(mode, 8, label="USTAR mode")
    header[108:116] = _ustar_octal(0, 8, label="USTAR uid")
    header[116:124] = _ustar_octal(0, 8, label="USTAR gid")
    header[124:136] = _ustar_octal(size, 12, label="USTAR size")
    header[136:148] = _ustar_octal(0, 12, label="USTAR mtime")
    header[148:156] = b"        "
    header[156:157] = b"0"
    header[257:263] = b"ustar\0"
    header[263:265] = b"00"
    header[345 : 345 + len(prefix)] = prefix
    checksum = format(sum(header), "06o").encode("ascii")
    if len(checksum) != 6:
        _fail("USTAR checksum exceeds its field")
    header[148:156] = checksum + b"\0 "
    return bytes(header)


def _decode_ustar_text(raw: bytes, *, label: str) -> str:
    if b"\0" not in raw:
        try:
            return raw.decode("ascii")
        except UnicodeDecodeError as exc:
            raise ForagerMatchedV3CpuOciBuildPlanError(f"{label} is not ASCII") from exc
    token, separator, tail = raw.partition(b"\0")
    if separator != b"\0" or any(tail) or not token:
        _fail(f"{label} is not canonical NUL-padded USTAR text")
    try:
        return token.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ForagerMatchedV3CpuOciBuildPlanError(f"{label} is not ASCII") from exc


def _decode_ustar_octal(raw: bytes, *, label: str) -> int:
    if (
        not raw.endswith(b"\0")
        or not raw[:-1]
        or any(byte < ord("0") or byte > ord("7") for byte in raw[:-1])
    ):
        _fail(f"{label} is not canonical NUL-terminated octal")
    return int(raw[:-1], 8)


def _source_header_identity(header: bytes) -> tuple[str, int, int]:
    name = _decode_ustar_text(header[0:100], label="USTAR name")
    prefix_field = header[345:500]
    if any(prefix_field):
        prefix = _decode_ustar_text(prefix_field, label="USTAR prefix")
        path = f"{prefix}/{name}"
    else:
        path = name
    path = _canonical_source_path(path, label="USTAR member path")
    mode = _decode_ustar_octal(header[100:108], label=f"USTAR mode {path}")
    size = _decode_ustar_octal(header[124:136], label=f"USTAR size {path}")
    if size > _MAX_SOURCE_MEMBER_BYTES:
        _fail(f"USTAR member exceeds its byte bound: {path}")
    if not hmac.compare_digest(header, _canonical_ustar_header(path, size, mode)):
        _fail(f"USTAR member header is not canonical: {path}")
    return path, size, mode


def _verify_ustar_tail(raw: bytes, offset: int, *, label: str) -> None:
    if len(raw) % _USTAR_RECORD_BYTES != 0:
        _fail(f"{label} size is not a canonical USTAR record multiple")
    if offset + 2 * _USTAR_BLOCK_BYTES > len(raw):
        _fail(f"{label} is missing its two USTAR end blocks")
    if any(raw[offset : offset + 2 * _USTAR_BLOCK_BYTES]):
        _fail(f"{label} USTAR end blocks are not zero")
    offset += 2 * _USTAR_BLOCK_BYTES
    final_size = offset + (-offset) % _USTAR_RECORD_BYTES
    if final_size != len(raw) or any(raw[offset:]):
        _fail(f"{label} USTAR record padding is not canonical")


def _parse_source_producer_receipt(
    source: CanonicalSourceBundleInput,
    *,
    role: str,
) -> tuple[dict[str, Any], str]:
    receipt_sha256 = _sha256(
        source.expected_receipt_sha256,
        label=f"{role} source producer receipt",
    )
    raw = source.receipt_bytes
    if (
        type(raw) is not bytes
        or not raw
        or len(raw) > _MAX_SOURCE_RECEIPT_BYTES
        or not hmac.compare_digest(_hash(raw), receipt_sha256)
    ):
        _fail(f"{role} source producer receipt full-file identity differs")
    try:
        if role == "external_foragax":
            receipt = (
                external_source_publication_contract.parse_external_source_publication_receipt(
                    raw,
                    expected_file_sha256=receipt_sha256,
                )
            )
        elif role == "local_alberta":
            receipt = local_source_bundle_contract.parse_matched_v3_local_source_bundle_receipt(
                raw,
                expected_receipt_sha256=receipt_sha256,
            )
        else:
            _fail("source-bundle role is unsupported")
    except ForagerMatchedV3CpuOciBuildPlanError:
        raise
    except (RuntimeError, TypeError, ValueError) as exc:
        raise ForagerMatchedV3CpuOciBuildPlanError(
            f"{role} source producer receipt parser rejected the exact bytes"
        ) from exc
    if type(receipt) is not dict:
        _fail(f"{role} source producer receipt parser returned a non-object")
    return receipt, receipt_sha256


def _normalized_receipt_members(
    receipt: Mapping[str, Any],
    *,
    role: str,
) -> tuple[Mapping[str, Any], list[dict[str, Any]]]:
    archive = receipt.get("archive")
    if type(archive) is not dict:
        _fail(f"{role} source producer receipt archive binding is absent")
    archive_value = cast(dict[str, Any], archive)
    raw_members = (
        archive_value.get("members") if role == "external_foragax" else receipt.get("members")
    )
    if type(raw_members) is not list:
        _fail(f"{role} source producer receipt member inventory is absent")
    normalized: list[dict[str, Any]] = []
    for index, raw_member in enumerate(raw_members):
        if type(raw_member) is not dict:
            _fail(f"{role} source producer receipt member {index} is not one object")
        member = cast(dict[str, Any], raw_member)
        path = _canonical_source_path(
            _string(member.get("path"), label=f"{role} receipt member {index} path"),
            label=f"{role} receipt member {index} path",
        )
        mode = _string(member.get("mode"), label=f"{role} receipt member {index} mode")
        if mode not in {"0444", "0555"}:
            _fail(f"{role} source producer receipt member mode differs")
        normalized.append(
            {
                "mode": mode,
                "path": path,
                "sha256": _sha256(member.get("sha256"), label=f"{role} receipt member {index}"),
                "size_bytes": _integer(
                    member.get("size_bytes"),
                    label=f"{role} receipt member {index} size",
                    maximum=_MAX_SOURCE_MEMBER_BYTES,
                ),
            }
        )
    return archive_value, normalized


def _validate_source_bundle(
    source: CanonicalSourceBundleInput,
    *,
    role: str,
    context_path: str,
) -> _ValidatedSourceBundle:
    if type(source) is not CanonicalSourceBundleInput:
        _fail(f"{role} source requires one exact canonical source-bundle input")
    if type(source.archive_bytes) is not bytes or not source.archive_bytes:
        _fail(f"{role} source archive bytes are absent")
    expected_size = _integer(
        source.expected_archive_size_bytes,
        label=f"{role} source archive size",
        minimum=_USTAR_RECORD_BYTES,
        maximum=_MAX_SOURCE_ARCHIVE_BYTES,
    )
    expected_sha = _sha256(
        source.expected_archive_sha256,
        label=f"{role} source archive",
    )
    expected_count = _integer(
        source.expected_member_count,
        label=f"{role} source member count",
        minimum=1,
        maximum=_MAX_SOURCE_MEMBERS,
    )
    source_manifest_sha = _sha256(
        source.source_manifest_sha256,
        label=f"{role} source manifest",
    )
    source_tree_sha = _sha256(
        source.source_tree_sha256,
        label=f"{role} source tree",
    )
    if role == "external_foragax":
        staging_manifest_sha: str | None = _sha256(
            source.staging_manifest_sha256,
            label="external_foragax staging manifest",
        )
    elif role == "local_alberta":
        if source.staging_manifest_sha256 is not None:
            _fail("local Alberta source staging-manifest pin must be absent")
        staging_manifest_sha = None
    else:
        _fail("source-bundle role is unsupported")
    receipt, receipt_sha256 = _parse_source_producer_receipt(source, role=role)
    receipt_archive, receipt_members = _normalized_receipt_members(receipt, role=role)
    raw = source.archive_bytes
    if len(raw) != expected_size or not hmac.compare_digest(_hash(raw), expected_sha):
        _fail(f"{role} source archive full-file identity differs")
    members: list[dict[str, Any]] = []
    paths: set[str] = set()
    offset = 0
    while offset + _USTAR_BLOCK_BYTES <= len(raw):
        header = raw[offset : offset + _USTAR_BLOCK_BYTES]
        if not any(header):
            break
        if len(members) >= _MAX_SOURCE_MEMBERS:
            _fail(f"{role} source archive exceeds its member bound")
        path, size, mode = _source_header_identity(header)
        if path in paths:
            _fail(f"{role} source archive contains an exact duplicate member path")
        if any(path.startswith(ancestor + "/") for ancestor in paths):
            _fail(f"{role} source archive contains a file/descendant collision")
        paths.add(path)
        offset += _USTAR_BLOCK_BYTES
        end = offset + size
        if end > len(raw):
            _fail(f"{role} source USTAR payload is truncated: {path}")
        payload = raw[offset:end]
        offset = end
        padding = (-size) % _USTAR_BLOCK_BYTES
        if offset + padding > len(raw) or any(raw[offset : offset + padding]):
            _fail(f"{role} source USTAR payload padding is not zero: {path}")
        offset += padding
        members.append(
            {
                "mode": "0444" if mode == 0o444 else "0555",
                "path": path,
                "sha256": _hash(payload),
                "size_bytes": size,
            }
        )
    _verify_ustar_tail(raw, offset, label=f"{role} source archive")
    if len(members) != expected_count:
        _fail(f"{role} source member count differs")
    if [member["path"] for member in members] != sorted(paths, key=str.encode):
        _fail(f"{role} source members are not in exact ASCII path order")
    required = (
        {"pyproject.toml", "alberta_framework/__init__.py"}
        if role == "local_alberta"
        else {"pyproject.toml"}
    )
    if not required.issubset(paths):
        _fail(f"{role} source archive omits its required project payload")
    if (
        receipt_archive.get("sha256") != expected_sha
        or receipt_archive.get("size_bytes") != expected_size
        or receipt_archive.get("member_count") != expected_count
    ):
        _fail(f"{role} source producer receipt archive identity differs from raw USTAR")
    if receipt_members != members:
        _fail(f"{role} source producer receipt member inventory differs from raw USTAR")
    if role == "external_foragax":
        raw_external = receipt.get("external_source_manifest")
        raw_stage = receipt.get("staging_manifest")
        raw_contract = receipt.get("publication_contract")
        if (
            type(raw_external) is not dict
            or type(raw_stage) is not dict
            or type(raw_contract) is not dict
        ):
            _fail("external_foragax source producer receipt provenance is incomplete")
        external = cast(dict[str, Any], raw_external)
        stage = cast(dict[str, Any], raw_stage)
        contract = cast(dict[str, Any], raw_contract)
        producer_descriptor_sha = _sha256(
            contract.get("descriptor_sha256"),
            label="external publication descriptor",
        )
        materialization_identity_sha: str | None = _sha256(
            external.get("identity_sha256"),
            label="external materialization identity",
        )
        commit_git_sha1: str | None = _git_sha1(
            external.get("commit_git_sha1"),
            label="external source receipt commit",
        )
        tree_git_sha1: str | None = _git_sha1(
            external.get("tree_git_sha1"),
            label="external source receipt tree",
        )
        if (
            external.get("full_file_sha256") != source_manifest_sha
            or external.get("source_tree_sha256") != source_tree_sha
        ):
            _fail("external_foragax source provenance differs from its producer receipt")
        if stage.get("full_file_sha256") != staging_manifest_sha:
            _fail("external_foragax staging-manifest provenance differs from its producer receipt")
    else:
        raw_snapshot = receipt.get("source_snapshot")
        raw_descriptor = receipt.get("descriptor_binding")
        if type(raw_snapshot) is not dict or type(raw_descriptor) is not dict:
            _fail("local_alberta source producer receipt provenance is incomplete")
        snapshot = cast(dict[str, Any], raw_snapshot)
        descriptor = cast(dict[str, Any], raw_descriptor)
        producer_descriptor_sha = _sha256(
            descriptor.get("sha256"),
            label="local source bundle descriptor",
        )
        materialization_identity_sha = None
        commit_git_sha1 = None
        tree_git_sha1 = None
        if (
            snapshot.get("manifest_sha256") != source_manifest_sha
            or snapshot.get("tree_sha256") != source_tree_sha
        ):
            _fail("local_alberta source provenance differs from its producer receipt")
    inventory_sha = _hash(_canonical_json({"members": members}))
    return _ValidatedSourceBundle(
        role=role,
        context_path=context_path,
        archive_sha256=expected_sha,
        archive_size_bytes=expected_size,
        member_count=expected_count,
        member_inventory_sha256=inventory_sha,
        receipt_sha256=receipt_sha256,
        receipt_size_bytes=len(source.receipt_bytes),
        producer_descriptor_sha256=producer_descriptor_sha,
        materialization_identity_sha256=materialization_identity_sha,
        source_manifest_sha256=source_manifest_sha,
        source_tree_sha256=source_tree_sha,
        staging_manifest_sha256=staging_manifest_sha,
        commit_git_sha1=commit_git_sha1,
        tree_git_sha1=tree_git_sha1,
        members=tuple(copy.deepcopy(members)),
    )


def _verify_wheelhouse_archive(
    raw: bytes,
    entries: Sequence[Mapping[str, Any]],
    *,
    expected_sha256: str,
    expected_size_bytes: int,
) -> list[dict[str, Any]]:
    if type(raw) is not bytes or not raw or len(raw) > _MAX_WHEELHOUSE_ARCHIVE_BYTES:
        _fail("wheelhouse archive bytes are absent or outside their bound")
    if len(raw) != expected_size_bytes or not hmac.compare_digest(_hash(raw), expected_sha256):
        _fail("wheelhouse archive full-file identity differs")
    expected = sorted(entries, key=lambda entry: cast(str, entry["archive_name"]).encode("ascii"))
    observed: list[dict[str, Any]] = []
    offset = 0
    for entry in expected:
        archive_name = cast(str, entry["archive_name"])
        size = cast(int, entry["size_bytes"])
        if offset + _USTAR_BLOCK_BYTES > len(raw):
            _fail(f"wheelhouse USTAR header is truncated: {archive_name}")
        header = raw[offset : offset + _USTAR_BLOCK_BYTES]
        if not hmac.compare_digest(header, _canonical_ustar_header(archive_name, size, 0o444)):
            _fail(f"wheelhouse USTAR header differs from CAS entry: {archive_name}")
        offset += _USTAR_BLOCK_BYTES
        end = offset + size
        if end > len(raw):
            _fail(f"wheelhouse USTAR payload is truncated: {archive_name}")
        payload = raw[offset:end]
        if not hmac.compare_digest(_hash(payload), cast(str, entry["sha256"])):
            _fail(f"wheelhouse USTAR payload differs from CAS entry: {archive_name}")
        offset = end
        padding = (-size) % _USTAR_BLOCK_BYTES
        if offset + padding > len(raw) or any(raw[offset : offset + padding]):
            _fail(f"wheelhouse USTAR padding differs: {archive_name}")
        offset += padding
        observed.append(
            {
                "archive_name": archive_name,
                "sha256": entry["sha256"],
                "size_bytes": size,
            }
        )
    _verify_ustar_tail(raw, offset, label="wheelhouse archive")
    return observed


def _claims() -> dict[str, bool]:
    return {
        "base_image_available": False,
        "build_context_materialized": False,
        "container_image_built": False,
        "dependency_installation_reproduced": False,
        "execution_authority_granted": False,
        "image_identity_issued": False,
        "imports_qualified": False,
        "network_isolation_observed": False,
        "qualification_granted": False,
        "runtime_qualified": False,
        "scientific_evidence_created": False,
        "scientific_promotion_allowed": False,
        "source_linkage_executed": False,
        "universal_sota_claim_allowed": False,
    }


def _limitations() -> list[str]:
    return [
        "This artifact is a pure-content plan; it does not pull, build, install, import, or run.",
        (
            "The base linux/amd64 manifest digest is pinned, but its OCI content is not "
            "downloaded or inspected by this module."
        ),
        (
            "Source receipt, manifest, staging, and tree digests are bindings only; actual "
            "canonical USTAR payload bytes are independently mandatory."
        ),
        (
            "Each source USTAR is cross-bound to its exact producer receipt; the external "
            "receipt commit/tree is also cross-bound to runtime-lock upstream provenance."
        ),
        (
            "Source trees are linked through a fixed site .pth file without invoking an "
            "unlocked PEP 517 build backend."
        ),
        (
            "Source USTAR extraction and runtime access require case-sensitive Linux "
            "filesystem semantics; case-distinct POSIX member paths are intentionally distinct."
        ),
        (
            "A later build executor must materialize the exact context paths and hashes, "
            "bind the local Unix Docker daemon and its default builder, run with network=none "
            "and pull=false, and inspect the resulting image."
        ),
        (
            "Docker build network=none governs build steps, not registry resolution; pull=false "
            "disables refresh but does not prove absence of implicit registry contact. An "
            "executor must prove the digest-pinned base is preloaded and report daemon egress "
            "as unobserved and unattested."
        ),
        "Canonical planning grants no execution, qualification, evidence, or promotion authority.",
    ]


_MATERIALIZE_WHEELHOUSE_SOURCE: Final = """from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    mapping_path = Path("/opt/elizaos/build/wheel-map.json")
    source_root = Path("/opt/elizaos/wheel-cas")
    target_root = Path("/opt/elizaos/wheelhouse")
    value = json.loads(mapping_path.read_bytes())
    entries = value["entries"]
    expected_sources = {entry["archive_name"] for entry in entries}
    if {path.name for path in source_root.iterdir()} != expected_sources:
        raise SystemExit("wheel CAS member set differs")
    for entry in entries:
        source = source_root / entry["archive_name"]
        target = target_root / entry["filename"]
        if source.stat().st_size != entry["size_bytes"] or _digest(source) != entry["sha256"]:
            raise SystemExit("wheel CAS member identity differs")
        os.replace(source, target)
    expected_targets = {entry["filename"] for entry in entries}
    if {path.name for path in target_root.iterdir()} != expected_targets:
        raise SystemExit("install wheel member set differs")


if __name__ == "__main__":
    main()
"""


_VERIFY_SOURCES_SOURCE: Final = """from __future__ import annotations

import hashlib
import os
import re
import stat
from pathlib import Path


_BLOCK_BYTES = 512
_RECORD_BYTES = 10_240
_MAX_ARCHIVE_BYTES = 2 * 1024 * 1024 * 1024
_MAX_MEMBER_BYTES = 256 * 1024 * 1024
_MAX_MEMBERS = 20_000
_COMPONENT_RE = re.compile(r"[A-Za-z0-9_.+-]{1,255}\\Z")


def _fail(message: str) -> None:
    raise SystemExit(f"source verification failed: {message}")


def _canonical_path(value: str) -> str:
    try:
        encoded = value.encode("ascii")
    except UnicodeEncodeError:
        _fail("a source path is not ASCII")
    if not value or value.startswith("/") or value.endswith("/") or "\\\\" in value:
        _fail("a source path is not canonical and relative")
    if len(encoded) > 255:
        _fail("a source path exceeds its byte bound")
    parts = value.split("/")
    if any(
        part in {"", ".", ".."} or _COMPONENT_RE.fullmatch(part) is None for part in parts
    ):
        _fail("a source path contains an unsafe component")
    if any(
        part in {"__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache"}
        for part in parts
    ) or value.endswith((".pyc", ".pyo")):
        _fail("a source path contains an excluded cache artifact")
    return value


def _split_path(path: str) -> tuple[bytes, bytes]:
    encoded = path.encode("ascii")
    if len(encoded) <= 100:
        return b"", encoded
    positions = [index for index, byte in enumerate(encoded) if byte == ord("/")]
    for index in reversed(positions):
        prefix = encoded[:index]
        name = encoded[index + 1 :]
        if prefix and name and len(prefix) <= 155 and len(name) <= 100:
            return prefix, name
    _fail("a source path is not exactly representable in POSIX USTAR")


def _octal(value: int, width: int) -> bytes:
    token = format(value, "o").encode("ascii")
    if value < 0 or len(token) > width - 1:
        _fail("a POSIX USTAR integer exceeds its field")
    return token.rjust(width - 1, b"0") + b"\\0"


def _canonical_header(path: str, size: int, mode: int) -> bytes:
    _canonical_path(path)
    if mode not in {0o444, 0o555}:
        _fail("a source member mode is not exactly 0444 or 0555")
    prefix, name = _split_path(path)
    header = bytearray(_BLOCK_BYTES)
    header[0 : len(name)] = name
    header[100:108] = _octal(mode, 8)
    header[108:116] = _octal(0, 8)
    header[116:124] = _octal(0, 8)
    header[124:136] = _octal(size, 12)
    header[136:148] = _octal(0, 12)
    header[148:156] = b"        "
    header[156:157] = b"0"
    header[257:263] = b"ustar\\0"
    header[263:265] = b"00"
    header[345 : 345 + len(prefix)] = prefix
    checksum = format(sum(header), "06o").encode("ascii")
    if len(checksum) != 6:
        _fail("a POSIX USTAR checksum exceeds its field")
    header[148:156] = checksum + b"\\0 "
    return bytes(header)


def _decode_text(raw: bytes) -> str:
    if b"\\0" not in raw:
        try:
            return raw.decode("ascii")
        except UnicodeDecodeError:
            _fail("a POSIX USTAR text field is not ASCII")
    token, separator, tail = raw.partition(b"\\0")
    if separator != b"\\0" or not token or any(tail):
        _fail("a POSIX USTAR text field is not canonically NUL-padded")
    try:
        return token.decode("ascii")
    except UnicodeDecodeError:
        _fail("a POSIX USTAR text field is not ASCII")


def _decode_octal(raw: bytes) -> int:
    if (
        not raw.endswith(b"\\0")
        or not raw[:-1]
        or any(byte < ord("0") or byte > ord("7") for byte in raw[:-1])
    ):
        _fail("a POSIX USTAR integer is not canonical octal")
    return int(raw[:-1], 8)


def _header_identity(header: bytes) -> tuple[str, int, int]:
    name = _decode_text(header[0:100])
    prefix_field = header[345:500]
    path = f"{_decode_text(prefix_field)}/{name}" if any(prefix_field) else name
    path = _canonical_path(path)
    mode = _decode_octal(header[100:108])
    size = _decode_octal(header[124:136])
    if size > _MAX_MEMBER_BYTES:
        _fail("a source member exceeds its byte bound")
    if header != _canonical_header(path, size, mode):
        _fail("a source member header is not canonical POSIX USTAR")
    return path, size, mode


def _read_exact(stream: object, size: int) -> bytes:
    result = bytearray()
    while len(result) < size:
        block = stream.read(size - len(result))
        if not block:
            _fail("a source USTAR is truncated")
        result.extend(block)
    return bytes(result)


def _payload_digest(stream: object, size: int) -> str:
    value = hashlib.sha256()
    remaining = size
    while remaining:
        block = _read_exact(stream, min(1024 * 1024, remaining))
        value.update(block)
        remaining -= len(block)
    return value.hexdigest()


def _archive_inventory(archive_path: Path) -> tuple[dict[str, tuple[int, int, str]], set[str]]:
    archive_stat = archive_path.lstat()
    archive_size = archive_stat.st_size
    if (
        not stat.S_ISREG(archive_stat.st_mode)
        or archive_size < _RECORD_BYTES
        or archive_size > _MAX_ARCHIVE_BYTES
        or archive_size % _RECORD_BYTES != 0
    ):
        _fail("a source archive is not one bounded canonical USTAR regular file")
    members: dict[str, tuple[int, int, str]] = {}
    directories: set[str] = set()
    previous_path: bytes | None = None
    offset = 0
    terminated = False
    with archive_path.open("rb") as stream:
        while offset + _BLOCK_BYTES <= archive_size:
            header = _read_exact(stream, _BLOCK_BYTES)
            header_offset = offset
            offset += _BLOCK_BYTES
            if not any(header):
                if any(_read_exact(stream, _BLOCK_BYTES)):
                    _fail("a source USTAR second end block is not zero")
                offset += _BLOCK_BYTES
                final_size = offset + (-offset) % _RECORD_BYTES
                if final_size != archive_size:
                    _fail("a source USTAR record padding size is not canonical")
                while offset < archive_size:
                    block = _read_exact(stream, min(1024 * 1024, archive_size - offset))
                    if any(block):
                        _fail("a source USTAR record padding is not zero")
                    offset += len(block)
                terminated = True
                break
            if len(members) >= _MAX_MEMBERS:
                _fail("a source USTAR exceeds its member bound")
            path, size, mode = _header_identity(header)
            encoded_path = path.encode("ascii")
            if previous_path is not None and encoded_path <= previous_path:
                _fail("source USTAR member paths are duplicate or not in exact ASCII order")
            parts = path.split("/")
            if any("/".join(parts[:index]) in members for index in range(1, len(parts))):
                _fail("a source USTAR contains a file/descendant collision")
            previous_path = encoded_path
            for index in range(1, len(parts)):
                directories.add("/".join(parts[:index]))
            digest = _payload_digest(stream, size)
            offset += size
            padding_size = (-size) % _BLOCK_BYTES
            if any(_read_exact(stream, padding_size)):
                _fail("a source USTAR payload padding is not zero")
            offset += padding_size
            members[path] = (mode, size, digest)
            if offset <= header_offset:
                _fail("a source USTAR offset did not advance")
    if not terminated or offset != archive_size or not members:
        _fail("a source USTAR is missing members or its canonical end blocks")
    return members, directories


def _file_identity(path: Path) -> tuple[tuple[int, int, str], tuple[int, int]]:
    if not hasattr(os, "O_NOFOLLOW"):
        _fail("Linux O_NOFOLLOW semantics are unavailable")
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        observed = os.fstat(descriptor)
        if not stat.S_ISREG(observed.st_mode) or observed.st_nlink != 1:
            _fail("an extracted source entry is not one unlinked regular file")
        value = hashlib.sha256()
        while block := os.read(descriptor, 1024 * 1024):
            value.update(block)
        return (
            (stat.S_IMODE(observed.st_mode), observed.st_size, value.hexdigest()),
            (observed.st_dev, observed.st_ino),
        )
    finally:
        os.close(descriptor)


def _extracted_inventory(root: Path) -> tuple[dict[str, tuple[int, int, str]], set[str]]:
    root_stat = root.lstat()
    if not stat.S_ISDIR(root_stat.st_mode):
        _fail("an extracted source root is not one directory")
    members: dict[str, tuple[int, int, str]] = {}
    directories: set[str] = set()
    file_identities: set[tuple[int, int]] = set()
    pending = [(root, "")]
    while pending:
        directory, prefix = pending.pop()
        try:
            entries = list(os.scandir(directory))
        except OSError:
            _fail("an extracted source directory cannot be enumerated")
        for entry in entries:
            path = _canonical_path(f"{prefix}/{entry.name}" if prefix else entry.name)
            observed = entry.stat(follow_symlinks=False)
            if stat.S_ISDIR(observed.st_mode):
                directories.add(path)
                pending.append((Path(entry.path), path))
            elif stat.S_ISREG(observed.st_mode):
                member_identity, file_identity = _file_identity(Path(entry.path))
                if file_identity in file_identities:
                    _fail("two extracted source members share one filesystem inode")
                file_identities.add(file_identity)
                members[path] = member_identity
            else:
                _fail("an extracted source entry is a link or special file")
    return members, directories


def _verify_source(archive_path: Path, root: Path) -> None:
    expected_members, expected_directories = _archive_inventory(Path(archive_path))
    observed_members, observed_directories = _extracted_inventory(Path(root))
    if set(observed_members) != set(expected_members):
        _fail("the extracted source member set or exact path spelling differs")
    if observed_directories != expected_directories:
        _fail("the extracted source directory set or exact path spelling differs")
    for path, expected in expected_members.items():
        if observed_members[path] != expected:
            _fail(f"the extracted source mode, size, or SHA-256 differs: {path}")


def main() -> None:
    pairs = [
        (
            Path("/opt/elizaos/input/external-foragax-source.v1.tar"),
            Path("/opt/elizaos/src/external-foragax"),
        ),
        (
            Path("/opt/elizaos/input/local-alberta-source.v1.tar"),
            Path("/opt/elizaos/src/local-alberta"),
        ),
    ]
    for archive_path, root in pairs:
        _verify_source(archive_path, root)


if __name__ == "__main__":
    main()
"""


_VERIFY_RUNTIME_SOURCE: Final = """from __future__ import annotations

import importlib
import importlib.metadata
import json
from pathlib import Path


def main() -> None:
    expected = json.loads(Path("/opt/elizaos/build/runtime-inventory.json").read_bytes())
    observed = {
        item["name"]: importlib.metadata.version(item["name"])
        for item in expected["distributions"]
    }
    wanted = {item["name"]: item["version"] for item in expected["distributions"]}
    if observed != wanted:
        raise SystemExit("installed distribution inventory differs")
    for module_name in expected["required_imports"]:
        importlib.import_module(module_name)
    import jax
    import jaxlib
    import jax.numpy as jnp

    if jax.__version__ != "0.11.0" or jaxlib.__version__ != "0.11.0":
        raise SystemExit("JAX runtime version differs")
    if jax.default_backend() != "cpu" or not jax.devices():
        raise SystemExit("JAX CPU backend is unavailable")
    if any(device.platform != "cpu" for device in jax.devices()):
        raise SystemExit("non-CPU JAX device is active")
    import flax.linen as nn

    inputs = jnp.ones((1, 3), dtype=jnp.float32)
    flax_layer = nn.Dense(features=2)
    flax_parameters = flax_layer.init(jax.random.key(1), inputs)
    flax_output = jax.jit(flax_layer.apply)(flax_parameters, inputs)
    flax_output.block_until_ready()
    if flax_output.shape != (1, 2) or not bool(jnp.all(jnp.isfinite(flax_output))):
        raise SystemExit("Flax Dense init/apply/JIT probe differs")
    import haiku as hk

    def haiku_forward(value: object) -> object:
        return hk.Linear(2)(value)

    haiku_layer = hk.without_apply_rng(hk.transform(haiku_forward))
    haiku_parameters = haiku_layer.init(jax.random.key(2), inputs)
    haiku_output = jax.jit(haiku_layer.apply)(haiku_parameters, inputs)
    haiku_output.block_until_ready()
    if haiku_output.shape != (1, 2) or not bool(jnp.all(jnp.isfinite(haiku_output))):
        raise SystemExit("Haiku Linear init/apply/JIT probe differs")
    import alberta_framework

    origin = Path(alberta_framework.__file__).resolve(strict=True)
    root = Path("/opt/elizaos/src/local-alberta").resolve(strict=True)
    if not origin.is_relative_to(root):
        raise SystemExit("Alberta import origin differs from the source bundle")


if __name__ == "__main__":
    main()
"""


def _requirements_text(entries: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        f"{entry['name']}=={entry['version']} --hash=sha256:{entry['sha256']}" for entry in entries
    ]
    return "\n".join(lines) + "\n"


def _wheel_map_text(entries: Sequence[Mapping[str, Any]]) -> str:
    mapping = {
        "entries": [
            {
                "archive_name": entry["archive_name"],
                "filename": entry["filename"],
                "sha256": entry["sha256"],
                "size_bytes": entry["size_bytes"],
            }
            for entry in entries
        ]
    }
    return _canonical_json(mapping).decode("ascii")


def _runtime_inventory_text(entries: Sequence[Mapping[str, Any]]) -> str:
    return _canonical_json(
        {
            "distributions": [
                {"name": entry["name"], "version": entry["version"]} for entry in entries
            ],
            "python_version": runtime_lock_contract.PRODUCTION_PYTHON_VERSION,
            "required_functional_probes": _REQUIRED_FUNCTIONAL_PROBES,
            "required_imports": _REQUIRED_RUNTIME_IMPORTS,
            "runtime": "cpu",
        }
    ).decode("ascii")


def _archive_checksums_text(
    wheelhouse_sha256: str,
    external_sha256: str,
    local_sha256: str,
) -> str:
    return (
        f"{wheelhouse_sha256}  /opt/elizaos/input/wheelhouse.v1.tar\n"
        f"{external_sha256}  /opt/elizaos/input/external-foragax-source.v1.tar\n"
        f"{local_sha256}  /opt/elizaos/input/local-alberta-source.v1.tar\n"
    )


def _source_pth_text() -> str:
    return f"{_EXTERNAL_SOURCE_ROOT}\n{_LOCAL_SOURCE_ROOT}\n"


def _render_dockerfile(bindings: Mapping[str, Any]) -> str:
    runtime = cast(Mapping[str, Any], bindings["runtime_lock"])
    wheelhouse = cast(Mapping[str, Any], bindings["wheelhouse"])
    sources = cast(list[dict[str, Any]], bindings["sources"])
    source_by_role = {cast(str, source["role"]): source for source in sources}
    labels = {
        "io.elizaos.alberta.forager-matched-v3.base-manifest": BASE_IMAGE_MANIFEST_DIGEST,
        "io.elizaos.alberta.forager-matched-v3.cas-manifest-sha256": cast(
            str, wheelhouse["cas_manifest_sha256"]
        ),
        "io.elizaos.alberta.forager-matched-v3.external-source-sha256": cast(
            str, source_by_role["external_foragax"]["archive_sha256"]
        ),
        "io.elizaos.alberta.forager-matched-v3.local-source-sha256": cast(
            str, source_by_role["local_alberta"]["archive_sha256"]
        ),
        "io.elizaos.alberta.forager-matched-v3.runtime-lock-sha256": cast(str, runtime["sha256"]),
        "io.elizaos.alberta.forager-matched-v3.wheelhouse-sha256": cast(
            str, wheelhouse["archive_sha256"]
        ),
    }
    label_separator = " \\" + "\n    "
    label_text = label_separator.join(f'{key}="{labels[key]}"' for key in sorted(labels))
    pip_command = [
        "/usr/local/bin/python",
        "-m",
        "pip",
        "--isolated",
        "install",
        "--no-index",
        f"--find-links={_WHEEL_INSTALL_ROOT}",
        "--require-hashes",
        "--only-binary=:all:",
        "--no-deps",
        "--no-cache-dir",
        "--disable-pip-version-check",
        "--requirement=/opt/elizaos/build/requirements.lock",
    ]

    def json_argv(argv: Sequence[str]) -> str:
        return json.dumps(argv, ensure_ascii=True, separators=(",", ":"))

    lines = [
        f"FROM --platform={BASE_IMAGE_PLATFORM} {BASE_IMAGE_REFERENCE}",
        "ENV JAX_PLATFORMS=cpu \\",
        "    PIP_CONFIG_FILE=/dev/null \\",
        "    PIP_DISABLE_PIP_VERSION_CHECK=1 \\",
        "    PIP_NO_INDEX=1 \\",
        "    PYTHONDONTWRITEBYTECODE=1 \\",
        "    PYTHONHASHSEED=0 \\",
        "    TAR_OPTIONS= \\",
        "    XLA_FLAGS=--xla_force_host_platform_device_count=1",
        f"LABEL {label_text}",
        "WORKDIR /opt/elizaos",
        "COPY generated/archive-checksums.txt /opt/elizaos/build/archive-checksums.txt",
        "COPY generated/requirements.lock /opt/elizaos/build/requirements.lock",
        "COPY generated/wheel-map.json /opt/elizaos/build/wheel-map.json",
        ("COPY generated/materialize-wheelhouse.py /opt/elizaos/build/materialize-wheelhouse.py"),
        "COPY generated/runtime-inventory.json /opt/elizaos/build/runtime-inventory.json",
        "COPY generated/verify-sources.py /opt/elizaos/build/verify-sources.py",
        "COPY generated/verify-runtime.py /opt/elizaos/build/verify-runtime.py",
        f"COPY {_WHEELHOUSE_CONTEXT_PATH} /opt/elizaos/input/wheelhouse.v1.tar",
        (f"COPY {_EXTERNAL_SOURCE_CONTEXT_PATH} /opt/elizaos/input/external-foragax-source.v1.tar"),
        (f"COPY {_LOCAL_SOURCE_CONTEXT_PATH} /opt/elizaos/input/local-alberta-source.v1.tar"),
        ('RUN ["sha256sum","--check","--strict","/opt/elizaos/build/archive-checksums.txt"]'),
        (
            'RUN ["mkdir","-p","/opt/elizaos/wheel-cas","/opt/elizaos/wheelhouse",'
            '"/opt/elizaos/src/external-foragax","/opt/elizaos/src/local-alberta","/work"]'
        ),
        (
            'RUN ["/usr/bin/tar","--extract","--file",'
            '"/opt/elizaos/input/external-foragax-source.v1.tar","--directory",'
            '"/opt/elizaos/src/external-foragax","--no-same-owner","--no-same-permissions",'
            '"--keep-old-files"]'
        ),
        (
            'RUN ["/usr/bin/tar","--extract","--file",'
            '"/opt/elizaos/input/local-alberta-source.v1.tar","--directory",'
            '"/opt/elizaos/src/local-alberta","--no-same-owner","--no-same-permissions",'
            '"--keep-old-files"]'
        ),
        ('RUN ["/usr/local/bin/python","-I","-S","-B","/opt/elizaos/build/verify-sources.py"]'),
        (
            'RUN ["/usr/bin/tar","--extract","--file",'
            '"/opt/elizaos/input/wheelhouse.v1.tar",'
            '"--directory","/opt/elizaos/wheel-cas","--no-same-owner","--no-same-permissions"]'
        ),
        (
            'RUN ["/usr/local/bin/python","-I","-S","-B",'
            '"/opt/elizaos/build/materialize-wheelhouse.py"]'
        ),
        f"RUN {json_argv(pip_command)}",
        (
            "COPY generated/elizaos-forager-sources.pth "
            "/usr/local/lib/python3.12/site-packages/elizaos-forager-sources.pth"
        ),
        ('RUN ["/usr/local/bin/python","-I","-B","/opt/elizaos/build/verify-runtime.py"]'),
        'RUN ["chmod","0555","/work"]',
        "USER 65532:65532",
        "WORKDIR /work",
        'CMD ["/usr/local/bin/python","--version"]',
    ]
    return "\n".join(lines) + "\n"


def _build_file(path: str, content: str) -> dict[str, Any]:
    if _BUILD_FILE_PATH_RE.fullmatch(path) is None:
        _fail("generated build file path is invalid")
    try:
        raw = content.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ForagerMatchedV3CpuOciBuildPlanError("generated build file is not ASCII") from exc
    return {
        "content": content,
        "path": path,
        "sha256": _hash(raw),
        "size_bytes": len(raw),
    }


def _generated_files(
    bindings: Mapping[str, Any],
    entries: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    wheelhouse = cast(Mapping[str, Any], bindings["wheelhouse"])
    sources = cast(list[dict[str, Any]], bindings["sources"])
    source_by_role = {cast(str, source["role"]): source for source in sources}
    contents = {
        "Dockerfile": _render_dockerfile(bindings),
        "generated/archive-checksums.txt": _archive_checksums_text(
            cast(str, wheelhouse["archive_sha256"]),
            cast(str, source_by_role["external_foragax"]["archive_sha256"]),
            cast(str, source_by_role["local_alberta"]["archive_sha256"]),
        ),
        "generated/elizaos-forager-sources.pth": _source_pth_text(),
        "generated/materialize-wheelhouse.py": _MATERIALIZE_WHEELHOUSE_SOURCE,
        "generated/requirements.lock": _requirements_text(entries),
        "generated/runtime-inventory.json": _runtime_inventory_text(entries),
        "generated/verify-sources.py": _VERIFY_SOURCES_SOURCE,
        "generated/verify-runtime.py": _VERIFY_RUNTIME_SOURCE,
        "generated/wheel-map.json": _wheel_map_text(entries),
    }
    return [_build_file(path, contents[path]) for path in sorted(contents, key=str.encode)]


def _source_binding(source: _ValidatedSourceBundle, *, install_order: int) -> dict[str, Any]:
    return {
        "archive_format": "canonical_posix_ustar_uncompressed",
        "archive_sha256": source.archive_sha256,
        "archive_size_bytes": source.archive_size_bytes,
        "commit_git_sha1": source.commit_git_sha1,
        "context_path": source.context_path,
        "install_order": install_order,
        "link_mode": "site_pth_path_without_pep517_build",
        "member_count": source.member_count,
        "member_inventory_sha256": source.member_inventory_sha256,
        "materialization_identity_sha256": source.materialization_identity_sha256,
        "producer_descriptor_sha256": source.producer_descriptor_sha256,
        "receipt_sha256": source.receipt_sha256,
        "receipt_size_bytes": source.receipt_size_bytes,
        "role": source.role,
        "source_manifest_sha256": source.source_manifest_sha256,
        "source_tree_sha256": source.source_tree_sha256,
        "source_tree_payload_present": True,
        "staging_manifest_sha256": source.staging_manifest_sha256,
        "tree_git_sha1": source.tree_git_sha1,
    }


def _cross_bind_wheels(
    lock: Mapping[str, Any],
    cas: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> list[dict[str, Any]]:
    packages = cast(list[dict[str, Any]], lock["packages"])
    lock_by_name = {cast(str, package["name"]): package for package in packages}
    cas_entries = cast(list[dict[str, Any]], cas["entries"])
    receipt_archive = cast(Mapping[str, Any], receipt["archive"])
    receipt_members = {
        cast(str, member["filename"]): member
        for member in cast(list[dict[str, Any]], receipt_archive["members"])
    }
    if len(lock_by_name) != len(packages) or len(cas_entries) != len(packages):
        _fail("runtime lock and CAS distribution cardinalities differ")
    result: list[dict[str, Any]] = []
    for entry in cas_entries:
        name = cast(str, entry["name"])
        package = lock_by_name.get(name)
        if package is None:
            _fail(f"CAS entry is absent from the runtime lock: {name}")
        wheels = cast(list[dict[str, Any]], package["wheels"])
        if len(wheels) != 1:
            _fail(f"runtime lock does not select exactly one wheel: {name}")
        wheel = wheels[0]
        member = receipt_members.get(cast(str, entry["filename"]))
        expected = {
            "archive_name": None if member is None else member["archive_name"],
            "cas_key": wheel["cas_key"],
            "filename": wheel["filename"],
            "name": package["name"],
            "sha256": wheel["sha256"],
            "size_bytes": wheel["size_bytes"],
            "version": package["version"],
            "wheel_body_sha256": wheel["wheel_body_sha256"],
        }
        observed = {field: entry[field] for field in expected}
        if (
            observed != expected
            or member is None
            or (
                member["sha256"],
                member["size_bytes"],
            )
            != (
                entry["sha256"],
                entry["size_bytes"],
            )
        ):
            _fail(f"CAS entry substitution differs from lock or archive receipt: {name}")
        result.append(copy.deepcopy(entry))
    if [entry["name"] for entry in result] != sorted(lock_by_name):
        _fail("cross-bound CAS entries are not in runtime-lock distribution order")
    return result


def _validate_binding_graph(
    artifacts: runtime_lock_issuer.CpuRuntimeLockIssuanceArtifacts,
    lock: Mapping[str, Any],
    cas: Mapping[str, Any],
    receipt: Mapping[str, Any],
    wheelhouse_archive_bytes: bytes,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if (
        _hash(artifacts.runtime_lock_bytes) != artifacts.runtime_lock_sha256
        or _hash(artifacts.cas_manifest_bytes) != artifacts.cas_manifest_sha256
        or _hash(artifacts.wheelhouse_receipt_bytes) != artifacts.wheelhouse_receipt_sha256
    ):
        _fail("issued runtime-lock artifact full-file identity differs")
    wheelhouse = cast(Mapping[str, Any], lock["wheelhouse"])
    lock_manifest = cast(Mapping[str, Any], wheelhouse["manifest"])
    lock_archive = cast(Mapping[str, Any], wheelhouse["archive"])
    cas_receipt = cast(Mapping[str, Any], cas["source_receipt"])
    cas_archive = cast(Mapping[str, Any], cas["source_archive"])
    receipt_archive = cast(Mapping[str, Any], receipt["archive"])
    if (
        lock_manifest["sha256"] != artifacts.cas_manifest_sha256
        or lock_manifest["size_bytes"] != len(artifacts.cas_manifest_bytes)
        or lock_manifest["body_sha256"] != cas["manifest_body_sha256"]
        or lock_manifest["entry_count"] != cas["entry_count"]
        or lock_manifest["total_bytes"] != cas["total_bytes"]
        or lock_manifest["inventory_sha256"] != cas["wheel_inventory_sha256"]
        or cas_receipt["full_file_sha256"] != artifacts.wheelhouse_receipt_sha256
        or cas_receipt["body_sha256"] != receipt["receipt_body_sha256"]
    ):
        _fail("runtime lock, CAS manifest, and wheelhouse receipt identities differ")
    archive_identity = (
        lock_archive["sha256"],
        lock_archive["size_bytes"],
    )
    if archive_identity != (
        cas_archive["sha256"],
        cas_archive["size_bytes"],
    ) or archive_identity != (
        receipt_archive["sha256"],
        receipt_archive["size_bytes"],
    ):
        _fail("wheelhouse archive identity differs across lock, CAS, and receipt")
    if cas_archive["inventory_sha256"] != receipt_archive["inventory_sha256"]:
        _fail("wheelhouse archive inventory differs across CAS and receipt")
    entries = _cross_bind_wheels(lock, cas, receipt)
    archive_members = _verify_wheelhouse_archive(
        wheelhouse_archive_bytes,
        entries,
        expected_sha256=cast(str, lock_archive["sha256"]),
        expected_size_bytes=cast(int, lock_archive["size_bytes"]),
    )
    archive_by_name = {cast(str, member["archive_name"]): member for member in archive_members}
    for entry in entries:
        member = archive_by_name.get(cast(str, entry["archive_name"]))
        if member is None or (member["sha256"], member["size_bytes"]) != (
            entry["sha256"],
            entry["size_bytes"],
        ):
            _fail("CAS entry is not independently bound to verified archive bytes")
    return entries, archive_members


def _source_provenance(lock: Mapping[str, Any]) -> dict[str, Any]:
    upstream = cast(Mapping[str, Any], lock["upstream"])
    archive = cast(Mapping[str, Any], upstream["archive"])
    return {
        "archive_sha256": archive["sha256"],
        "archive_size_bytes": archive["size_bytes"],
        "commit_git_sha1": upstream["commit_git_sha1"],
        "repository_id": upstream["repository_id"],
        "repository_url": upstream["repository_url"],
        "tree_git_sha1": upstream["tree_git_sha1"],
    }


def _cross_bind_external_source_to_runtime_lock(
    external: _ValidatedSourceBundle,
    lock: Mapping[str, Any],
) -> None:
    upstream = cast(Mapping[str, Any], lock["upstream"])
    if (
        external.commit_git_sha1 is None
        or external.tree_git_sha1 is None
        or external.commit_git_sha1 != upstream["commit_git_sha1"]
        or external.tree_git_sha1 != upstream["tree_git_sha1"]
    ):
        _fail("external source receipt Git identity differs from runtime-lock upstream")


def build_matched_v3_cpu_oci_build_plan(
    *,
    issuance_artifacts: runtime_lock_issuer.CpuRuntimeLockIssuanceArtifacts,
    expected_root_pin_inventory_sha256: str,
    expected_selected_wheel_inventory_sha256: str,
    expected_resolution_lock_sha256: str,
    expected_resolution_lock_size_bytes: int,
    wheelhouse_archive_bytes: bytes,
    external_foragax_source: CanonicalSourceBundleInput,
    local_alberta_source: CanonicalSourceBundleInput | None,
) -> CpuOciBuildPlanArtifacts:
    """Return an exact offline plan after replaying every supplied content boundary.

    ``local_alberta_source`` is intentionally optional in the type so callers get
    a specific fail-closed diagnostic while the canonical local bundle is not yet
    available.  ``None`` can never produce a plan.
    """

    if local_alberta_source is None:
        _fail(
            "a local Alberta canonical USTAR payload is mandatory; snapshot manifest/tree "
            "identities alone are not source bytes"
        )
    external = _validate_source_bundle(
        external_foragax_source,
        role="external_foragax",
        context_path=_EXTERNAL_SOURCE_CONTEXT_PATH,
    )
    local = _validate_source_bundle(
        local_alberta_source,
        role="local_alberta",
        context_path=_LOCAL_SOURCE_CONTEXT_PATH,
    )
    try:
        validated_artifacts = runtime_lock_issuer.validate_production_cpu_runtime_lock_issuance(
            issuance_artifacts,
            expected_root_pin_inventory_sha256=expected_root_pin_inventory_sha256,
            expected_selected_wheel_inventory_sha256=(expected_selected_wheel_inventory_sha256),
            expected_resolution_lock_sha256=expected_resolution_lock_sha256,
            expected_resolution_lock_size_bytes=expected_resolution_lock_size_bytes,
        )
        lock = runtime_lock_contract.parse_cpu_runtime_lock(
            validated_artifacts.runtime_lock_bytes,
            expected_file_sha256=validated_artifacts.runtime_lock_sha256,
        )
        cas = runtime_lock_issuer.parse_cpu_runtime_wheelhouse_cas_manifest(
            validated_artifacts.cas_manifest_bytes,
            expected_file_sha256=validated_artifacts.cas_manifest_sha256,
        )
        receipt = wheelhouse_contract.parse_cpu_wheelhouse_receipt(
            validated_artifacts.wheelhouse_receipt_bytes,
            expected_file_sha256=validated_artifacts.wheelhouse_receipt_sha256,
        )
    except ForagerMatchedV3CpuOciBuildPlanError:
        raise
    except (TypeError, ValueError) as exc:
        raise ForagerMatchedV3CpuOciBuildPlanError(
            "production runtime-lock issuance or one final parser rejected the build input"
        ) from exc
    _cross_bind_external_source_to_runtime_lock(external, lock)
    entries, archive_members = _validate_binding_graph(
        validated_artifacts,
        lock,
        cas,
        receipt,
        wheelhouse_archive_bytes,
    )
    if len(entries) != runtime_lock_contract.PRODUCTION_DISTRIBUTION_COUNT:
        _fail("OCI build plan requires the exact production distribution count")
    wheelhouse = cast(Mapping[str, Any], lock["wheelhouse"])
    lock_archive = cast(Mapping[str, Any], wheelhouse["archive"])
    sources = [_source_binding(external, install_order=1), _source_binding(local, install_order=2)]
    bindings: dict[str, Any] = {
        "runtime_lock": {
            "body_sha256": lock["lock_body_sha256"],
            "sha256": validated_artifacts.runtime_lock_sha256,
            "size_bytes": len(validated_artifacts.runtime_lock_bytes),
        },
        "runtime_lock_issuance": {
            "capture_manifest_sha256": validated_artifacts.capture_manifest_sha256,
            "issuance_envelope_sha256": validated_artifacts.issuance_envelope_sha256,
            "root_pin_count": validated_artifacts.root_pin_count,
            "root_pin_inventory_sha256": validated_artifacts.root_pin_inventory_sha256,
        },
        "sources": sources,
        "upstream_foragax_provenance": _source_provenance(lock),
        "wheelhouse": {
            "archive_inventory_sha256": cast(Mapping[str, Any], cas["source_archive"])[
                "inventory_sha256"
            ],
            "archive_member_count": len(archive_members),
            "archive_sha256": lock_archive["sha256"],
            "archive_size_bytes": lock_archive["size_bytes"],
            "cas_manifest_body_sha256": cas["manifest_body_sha256"],
            "cas_manifest_sha256": validated_artifacts.cas_manifest_sha256,
            "cas_manifest_size_bytes": len(validated_artifacts.cas_manifest_bytes),
            "receipt_body_sha256": receipt["receipt_body_sha256"],
            "receipt_sha256": validated_artifacts.wheelhouse_receipt_sha256,
            "receipt_size_bytes": len(validated_artifacts.wheelhouse_receipt_bytes),
        },
    }
    generated_files = _generated_files(bindings, entries)
    plan: dict[str, Any] = {
        "base_image": {
            "informational_tag": BASE_IMAGE_INFORMATIONAL_TAG,
            "manifest_digest": BASE_IMAGE_MANIFEST_DIGEST,
            "platform": BASE_IMAGE_PLATFORM,
            "pull_by_tag_allowed": False,
            "reference": BASE_IMAGE_REFERENCE,
            "repository": BASE_IMAGE_REPOSITORY,
        },
        "bindings": bindings,
        "build": {
            "command": [
                *_BUILD_COMMAND_PREFIX,
                "--network=none",
                "--pull=false",
                f"--platform={BASE_IMAGE_PLATFORM}",
                "--file=Dockerfile",
                "--build-arg=SOURCE_DATE_EPOCH=0",
                "--build-arg=BUILDKIT_MULTI_PLATFORM=1",
                "--provenance=false",
                "--sbom=false",
                "--load",
                "--no-cache",
                "--progress=plain",
                "-",
            ],
            "context_inputs": [
                {
                    "path": _WHEELHOUSE_CONTEXT_PATH,
                    "role": "wheelhouse_archive",
                    "sha256": lock_archive["sha256"],
                    "size_bytes": lock_archive["size_bytes"],
                },
                {
                    "path": _EXTERNAL_SOURCE_CONTEXT_PATH,
                    "role": "external_foragax_source",
                    "sha256": external.archive_sha256,
                    "size_bytes": external.archive_size_bytes,
                },
                {
                    "path": _LOCAL_SOURCE_CONTEXT_PATH,
                    "role": "local_alberta_source",
                    "sha256": local.archive_sha256,
                    "size_bytes": local.archive_size_bytes,
                },
            ],
            "generated_files": generated_files,
            "network_mode": "none",
            "pull": False,
        },
        "claims": _claims(),
        "classification": CPU_OCI_BUILD_PLAN_CLASSIFICATION,
        "dependency_install": {
            "distribution_count": len(entries),
            "index_access": "disabled",
            "pip_argv": [
                "/usr/local/bin/python",
                "-m",
                "pip",
                "--isolated",
                "install",
                "--no-index",
                f"--find-links={_WHEEL_INSTALL_ROOT}",
                "--require-hashes",
                "--only-binary=:all:",
                "--no-deps",
                "--no-cache-dir",
                "--disable-pip-version-check",
                "--requirement=/opt/elizaos/build/requirements.lock",
            ],
            "source_builds_allowed": False,
            "wheel_entries": copy.deepcopy(entries),
        },
        "execution_toolchain": _execution_toolchain(),
        "limitations": _limitations(),
        "plan_body_sha256": "0" * 64,
        "runtime_verification": {
            "commands": [
                [
                    "sha256sum",
                    "--check",
                    "--strict",
                    "/opt/elizaos/build/archive-checksums.txt",
                ],
                [
                    "/usr/local/bin/python",
                    "-I",
                    "-S",
                    "-B",
                    "/opt/elizaos/build/verify-sources.py",
                ],
                [
                    "/usr/local/bin/python",
                    "-I",
                    "-B",
                    "/opt/elizaos/build/verify-runtime.py",
                ],
            ],
            "expected_jax_backend": "cpu",
            "expected_jax_version": "0.11.0",
            "expected_jaxlib_version": "0.11.0",
            "expected_python_version": runtime_lock_contract.PRODUCTION_PYTHON_VERSION,
            "executed": False,
            "required_functional_probes": _REQUIRED_FUNCTIONAL_PROBES,
            "required_imports": _REQUIRED_RUNTIME_IMPORTS,
        },
        "schema_version": CPU_OCI_BUILD_PLAN_SCHEMA_VERSION,
        "source_install": {
            "build_backends_invoked": False,
            "order": ["external_foragax", "local_alberta"],
            "site_path_file": (
                "/usr/local/lib/python3.12/site-packages/elizaos-forager-sources.pth"
            ),
        },
        "status": CPU_OCI_BUILD_PLAN_STATUS,
    }
    plan["plan_body_sha256"] = _hash(
        _canonical_json({key: value for key, value in plan.items() if key != "plan_body_sha256"})
    )
    validated = validate_cpu_oci_build_plan(plan)
    raw = _canonical_json(validated)
    return CpuOciBuildPlanArtifacts(plan_bytes=raw, plan_sha256=_hash(raw))


def _validate_build_file(value: Any, *, label: str) -> dict[str, Any]:
    record = _exact(
        value,
        frozenset({"content", "path", "sha256", "size_bytes"}),
        label=label,
    )
    path = _string(record["path"], label=f"{label} path")
    content = _string(record["content"], label=f"{label} content")
    if _BUILD_FILE_PATH_RE.fullmatch(path) is None:
        _fail(f"{label} path is invalid")
    try:
        raw = content.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ForagerMatchedV3CpuOciBuildPlanError(f"{label} is not ASCII") from exc
    if record["size_bytes"] != len(raw) or _sha256(
        record["sha256"], label=f"{label} digest"
    ) != _hash(raw):
        _fail(f"{label} content identity differs")
    return record


def validate_cpu_oci_build_plan(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and detach one exact nonauthorizing offline OCI build plan."""

    plan = _exact(
        value,
        frozenset(
            {
                "base_image",
                "bindings",
                "build",
                "claims",
                "classification",
                "dependency_install",
                "execution_toolchain",
                "limitations",
                "plan_body_sha256",
                "runtime_verification",
                "schema_version",
                "source_install",
                "status",
            }
        ),
        label="CPU OCI build plan",
    )
    if (
        plan["schema_version"] != CPU_OCI_BUILD_PLAN_SCHEMA_VERSION
        or plan["status"] != CPU_OCI_BUILD_PLAN_STATUS
        or plan["classification"] != CPU_OCI_BUILD_PLAN_CLASSIFICATION
    ):
        _fail("CPU OCI build-plan schema, status, or classification differs")
    base = _exact(
        plan["base_image"],
        frozenset(
            {
                "informational_tag",
                "manifest_digest",
                "platform",
                "pull_by_tag_allowed",
                "reference",
                "repository",
            }
        ),
        label="base image",
    )
    if base != {
        "informational_tag": BASE_IMAGE_INFORMATIONAL_TAG,
        "manifest_digest": BASE_IMAGE_MANIFEST_DIGEST,
        "platform": BASE_IMAGE_PLATFORM,
        "pull_by_tag_allowed": False,
        "reference": BASE_IMAGE_REFERENCE,
        "repository": BASE_IMAGE_REPOSITORY,
    }:
        _fail("base image differs from the exact linux/amd64 manifest digest")
    claims = _exact(plan["claims"], frozenset(_claims()), label="OCI plan claims")
    if claims != _claims() or any(item is not False for item in claims.values()):
        _fail("OCI build plan must keep every authority claim false")
    if plan["limitations"] != _limitations():
        _fail("OCI build-plan limitations differ")
    if plan["execution_toolchain"] != _execution_toolchain():
        _fail("OCI build execution toolchain or isolated environment differs")
    bindings = _exact(
        plan["bindings"],
        frozenset(
            {
                "runtime_lock",
                "runtime_lock_issuance",
                "sources",
                "upstream_foragax_provenance",
                "wheelhouse",
            }
        ),
        label="OCI plan bindings",
    )
    runtime = _exact(
        bindings["runtime_lock"],
        frozenset({"body_sha256", "sha256", "size_bytes"}),
        label="runtime-lock binding",
    )
    _sha256(runtime["body_sha256"], label="runtime-lock body")
    _sha256(runtime["sha256"], label="runtime-lock file")
    _integer(runtime["size_bytes"], label="runtime-lock size", minimum=1)
    issuance = _exact(
        bindings["runtime_lock_issuance"],
        frozenset(
            {
                "capture_manifest_sha256",
                "issuance_envelope_sha256",
                "root_pin_count",
                "root_pin_inventory_sha256",
            }
        ),
        label="runtime-lock issuance binding",
    )
    _sha256(issuance["capture_manifest_sha256"], label="capture manifest")
    _sha256(issuance["issuance_envelope_sha256"], label="issuance envelope")
    _sha256(issuance["root_pin_inventory_sha256"], label="root-pin inventory")
    if issuance["root_pin_count"] != runtime_lock_issuer.PRODUCTION_ROOT_PIN_COUNT:
        _fail("OCI plan root-pin count differs from production")
    upstream = _exact(
        bindings["upstream_foragax_provenance"],
        frozenset(
            {
                "archive_sha256",
                "archive_size_bytes",
                "commit_git_sha1",
                "repository_id",
                "repository_url",
                "tree_git_sha1",
            }
        ),
        label="upstream Foragax provenance",
    )
    _sha256(upstream["archive_sha256"], label="upstream archive")
    _integer(upstream["archive_size_bytes"], label="upstream archive size", minimum=1)
    _git_sha1(upstream["commit_git_sha1"], label="upstream commit")
    _git_sha1(upstream["tree_git_sha1"], label="upstream tree")
    _string(upstream["repository_id"], label="upstream repository ID")
    _string(upstream["repository_url"], label="upstream repository URL")
    upstream_contract = cast(
        Mapping[str, Any], runtime_lock_contract.cpu_runtime_lock_descriptor()["upstream"]
    )
    if upstream != {
        "archive_sha256": upstream_contract["archive_sha256"],
        "archive_size_bytes": upstream_contract["archive_size_bytes"],
        "commit_git_sha1": upstream_contract["commit_git_sha1"],
        "repository_id": upstream_contract["repository_id"],
        "repository_url": upstream_contract["repository_url"],
        "tree_git_sha1": upstream_contract["tree_git_sha1"],
    }:
        _fail("upstream Foragax provenance differs from the runtime-lock contract")
    raw_sources = bindings["sources"]
    if type(raw_sources) is not list or len(raw_sources) != 2:
        _fail("OCI plan requires exactly two source-bundle bindings")
    sources = cast(list[dict[str, Any]], raw_sources)
    expected_roles = ["external_foragax", "local_alberta"]
    for index, source in enumerate(sources):
        item = _exact(
            source,
            frozenset(
                {
                    "archive_format",
                    "archive_sha256",
                    "archive_size_bytes",
                    "commit_git_sha1",
                    "context_path",
                    "install_order",
                    "link_mode",
                    "member_count",
                    "member_inventory_sha256",
                    "materialization_identity_sha256",
                    "producer_descriptor_sha256",
                    "receipt_sha256",
                    "receipt_size_bytes",
                    "role",
                    "source_manifest_sha256",
                    "source_tree_sha256",
                    "source_tree_payload_present",
                    "staging_manifest_sha256",
                    "tree_git_sha1",
                }
            ),
            label=f"source binding {index}",
        )
        expected_role = expected_roles[index]
        expected_path = (
            _EXTERNAL_SOURCE_CONTEXT_PATH
            if expected_role == "external_foragax"
            else _LOCAL_SOURCE_CONTEXT_PATH
        )
        if (
            item["role"] != expected_role
            or item["context_path"] != expected_path
            or item["install_order"] != index + 1
            or item["archive_format"] != "canonical_posix_ustar_uncompressed"
            or item["link_mode"] != "site_pth_path_without_pep517_build"
            or item["source_tree_payload_present"] is not True
        ):
            _fail("source-bundle role, path, order, format, or payload declaration differs")
        for field in (
            "archive_sha256",
            "member_inventory_sha256",
            "producer_descriptor_sha256",
            "receipt_sha256",
            "source_manifest_sha256",
            "source_tree_sha256",
        ):
            _sha256(item[field], label=f"source binding {field}")
        _integer(item["archive_size_bytes"], label="source archive size", minimum=1)
        _integer(item["member_count"], label="source member count", minimum=1)
        _integer(item["receipt_size_bytes"], label="source receipt size", minimum=1)
        if expected_role == "external_foragax":
            materialization_identity = _sha256(
                item["materialization_identity_sha256"], label="external materialization identity"
            )
            _sha256(item["staging_manifest_sha256"], label="external staging manifest")
            source_commit = _git_sha1(item["commit_git_sha1"], label="external source commit")
            source_tree = _git_sha1(item["tree_git_sha1"], label="external source Git tree")
            if (
                item["producer_descriptor_sha256"] != _EXTERNAL_PUBLICATION_DESCRIPTOR_SHA256
                or materialization_identity != _EXTERNAL_MATERIALIZATION_IDENTITY_SHA256
                or source_commit != upstream["commit_git_sha1"]
                or source_tree != upstream["tree_git_sha1"]
            ):
                _fail(
                    "external source producer descriptor, materialization identity, or Git "
                    "provenance differs"
                )
        elif (
            item["materialization_identity_sha256"] is not None
            or item["staging_manifest_sha256"] is not None
            or item["commit_git_sha1"] is not None
            or item["tree_git_sha1"] is not None
        ):
            _fail("local source binding cannot carry external materialization provenance")
        elif item["producer_descriptor_sha256"] != _LOCAL_SOURCE_BUNDLE_DESCRIPTOR_SHA256:
            _fail("local source producer descriptor differs")
    wheelhouse = _exact(
        bindings["wheelhouse"],
        frozenset(
            {
                "archive_inventory_sha256",
                "archive_member_count",
                "archive_sha256",
                "archive_size_bytes",
                "cas_manifest_body_sha256",
                "cas_manifest_sha256",
                "cas_manifest_size_bytes",
                "receipt_body_sha256",
                "receipt_sha256",
                "receipt_size_bytes",
            }
        ),
        label="wheelhouse binding",
    )
    for field in (
        "archive_inventory_sha256",
        "archive_sha256",
        "cas_manifest_body_sha256",
        "cas_manifest_sha256",
        "receipt_body_sha256",
        "receipt_sha256",
    ):
        _sha256(wheelhouse[field], label=f"wheelhouse {field}")
    for field in (
        "archive_member_count",
        "archive_size_bytes",
        "cas_manifest_size_bytes",
        "receipt_size_bytes",
    ):
        _integer(wheelhouse[field], label=f"wheelhouse {field}", minimum=1)
    dependency = _exact(
        plan["dependency_install"],
        frozenset(
            {
                "distribution_count",
                "index_access",
                "pip_argv",
                "source_builds_allowed",
                "wheel_entries",
            }
        ),
        label="dependency install",
    )
    raw_entries = dependency["wheel_entries"]
    if type(raw_entries) is not list or not raw_entries:
        _fail("dependency install wheel entries are absent")
    entries = cast(list[dict[str, Any]], raw_entries)
    names: set[str] = set()
    filenames: set[str] = set()
    archive_names: set[str] = set()
    for index, entry in enumerate(entries):
        item = _exact(
            entry,
            frozenset(
                {
                    "archive_name",
                    "cas_key",
                    "filename",
                    "name",
                    "sha256",
                    "size_bytes",
                    "source_url",
                    "version",
                    "wheel_body_sha256",
                }
            ),
            label=f"dependency wheel {index}",
        )
        name = _string(item["name"], label="dependency distribution name")
        filename = _string(item["filename"], label="dependency wheel filename")
        archive_name = _string(item["archive_name"], label="dependency archive name")
        wheel_sha = _sha256(item["sha256"], label="dependency wheel")
        version = _string(item["version"], label="dependency version")
        cas_key = _string(item["cas_key"], label="dependency CAS key")
        source_url = _string(item["source_url"], label="dependency source URL")
        cas_match = _CAS_KEY_RE.fullmatch(cas_key)
        source_match = _PYPI_WHEEL_URL_RE.fullmatch(source_url)
        if (
            _DISTRIBUTION_NAME_RE.fullmatch(name) is None
            or name in names
            or filename in filenames
            or archive_name in archive_names
            or archive_name != f"{wheel_sha}.whl"
            or not filename.endswith(".whl")
            or "/" in filename
            or "\\" in filename
            or not _wheel_filename_matches_identity(filename, name, version)
            or cas_match is None
            or cas_match.group("prefix") != wheel_sha[:2]
            or cas_match.group("sha256") != wheel_sha
            or cas_match.group("filename") != filename
            or source_match is None
            or source_match.group("filename") != filename
        ):
            _fail(
                "dependency wheel identity, source URL, CAS key, ordering key, or archive "
                "name differs"
            )
        names.add(name)
        filenames.add(filename)
        archive_names.add(archive_name)
        _sha256(item["wheel_body_sha256"], label="dependency wheel body")
        _integer(item["size_bytes"], label="dependency wheel size", minimum=1)
    if [entry["name"] for entry in entries] != sorted(names):
        _fail("dependency wheel entries are not sorted by canonical distribution name")
    expected_pip = [
        "/usr/local/bin/python",
        "-m",
        "pip",
        "--isolated",
        "install",
        "--no-index",
        f"--find-links={_WHEEL_INSTALL_ROOT}",
        "--require-hashes",
        "--only-binary=:all:",
        "--no-deps",
        "--no-cache-dir",
        "--disable-pip-version-check",
        "--requirement=/opt/elizaos/build/requirements.lock",
    ]
    if (
        dependency["distribution_count"] != len(entries)
        or wheelhouse["archive_member_count"] != len(entries)
        or dependency["index_access"] != "disabled"
        or dependency["pip_argv"] != expected_pip
        or dependency["source_builds_allowed"] is not False
    ):
        _fail("dependency install count, network policy, or exact pip argv differs")
    source_install = _exact(
        plan["source_install"],
        frozenset({"build_backends_invoked", "order", "site_path_file"}),
        label="source install",
    )
    if source_install != {
        "build_backends_invoked": False,
        "order": expected_roles,
        "site_path_file": ("/usr/local/lib/python3.12/site-packages/elizaos-forager-sources.pth"),
    }:
        _fail("source installation order or no-build linkage policy differs")
    runtime_verification = _exact(
        plan["runtime_verification"],
        frozenset(
            {
                "commands",
                "expected_jax_backend",
                "expected_jax_version",
                "expected_jaxlib_version",
                "expected_python_version",
                "executed",
                "required_functional_probes",
                "required_imports",
            }
        ),
        label="runtime verification",
    )
    expected_verification = {
        "commands": [
            [
                "sha256sum",
                "--check",
                "--strict",
                "/opt/elizaos/build/archive-checksums.txt",
            ],
            [
                "/usr/local/bin/python",
                "-I",
                "-S",
                "-B",
                "/opt/elizaos/build/verify-sources.py",
            ],
            [
                "/usr/local/bin/python",
                "-I",
                "-B",
                "/opt/elizaos/build/verify-runtime.py",
            ],
        ],
        "expected_jax_backend": "cpu",
        "expected_jax_version": "0.11.0",
        "expected_jaxlib_version": "0.11.0",
        "expected_python_version": runtime_lock_contract.PRODUCTION_PYTHON_VERSION,
        "executed": False,
        "required_functional_probes": _REQUIRED_FUNCTIONAL_PROBES,
        "required_imports": _REQUIRED_RUNTIME_IMPORTS,
    }
    if runtime_verification != expected_verification:
        _fail("runtime verification command or expected CPU runtime differs")
    build = _exact(
        plan["build"],
        frozenset(
            {
                "command",
                "context_inputs",
                "generated_files",
                "network_mode",
                "pull",
            }
        ),
        label="OCI build",
    )
    expected_command = [
        *_BUILD_COMMAND_PREFIX,
        "--network=none",
        "--pull=false",
        f"--platform={BASE_IMAGE_PLATFORM}",
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
    if (
        build["command"] != expected_command
        or build["network_mode"] != "none"
        or build["pull"] is not False
    ):
        _fail("OCI build command is not exact, pull-disabled, and networkless")
    expected_context = [
        {
            "path": _WHEELHOUSE_CONTEXT_PATH,
            "role": "wheelhouse_archive",
            "sha256": wheelhouse["archive_sha256"],
            "size_bytes": wheelhouse["archive_size_bytes"],
        },
        {
            "path": _EXTERNAL_SOURCE_CONTEXT_PATH,
            "role": "external_foragax_source",
            "sha256": sources[0]["archive_sha256"],
            "size_bytes": sources[0]["archive_size_bytes"],
        },
        {
            "path": _LOCAL_SOURCE_CONTEXT_PATH,
            "role": "local_alberta_source",
            "sha256": sources[1]["archive_sha256"],
            "size_bytes": sources[1]["archive_size_bytes"],
        },
    ]
    if build["context_inputs"] != expected_context:
        _fail("OCI build-context input path, role, or identity differs")
    raw_files = build["generated_files"]
    if type(raw_files) is not list:
        _fail("OCI generated build files must be one array")
    files = [
        _validate_build_file(item, label=f"generated build file {index}")
        for index, item in enumerate(raw_files)
    ]
    if [item["path"] for item in files] != sorted(
        [cast(str, item["path"]) for item in files], key=str.encode
    ) or len({cast(str, item["path"]) for item in files}) != len(files):
        _fail("generated build files are not unique and sorted")
    expected_files = _generated_files(bindings, entries)
    if files != expected_files:
        _fail("generated Dockerfile or auxiliary build content differs from the exact plan")
    dockerfile = next(cast(str, item["content"]) for item in files if item["path"] == "Dockerfile")
    forbidden = ("http://", "https://", "apt-get", "curl ", "wget ", "git clone")
    if any(token in dockerfile for token in forbidden) or not dockerfile.startswith(
        f"FROM --platform={BASE_IMAGE_PLATFORM} {BASE_IMAGE_REFERENCE}\n"
    ):
        _fail("Dockerfile contains a network fetch or non-digest base authority")
    _body_sha256(plan, "plan_body_sha256", label="CPU OCI build plan")
    return _strict_json(_canonical_json(plan), label="CPU OCI build plan")


def canonical_cpu_oci_build_plan_bytes(value: Mapping[str, Any]) -> bytes:
    """Validate and canonically encode one CPU OCI build plan."""

    return _canonical_json(validate_cpu_oci_build_plan(value))


def parse_cpu_oci_build_plan(raw: bytes, *, expected_file_sha256: str) -> dict[str, Any]:
    """Parse canonical plan bytes under an independent full-file digest pin."""

    expected = _sha256(expected_file_sha256, label="expected CPU OCI build plan")
    if type(raw) is not bytes or not hmac.compare_digest(_hash(raw), expected):
        _fail("CPU OCI build-plan full-file SHA-256 differs")
    return validate_cpu_oci_build_plan(_strict_json(raw, label="CPU OCI build plan"))


__all__ = [
    "BASE_IMAGE_INFORMATIONAL_TAG",
    "BASE_IMAGE_MANIFEST_DIGEST",
    "BASE_IMAGE_PLATFORM",
    "BASE_IMAGE_REFERENCE",
    "BASE_IMAGE_REPOSITORY",
    "CPU_OCI_BUILD_PLAN_CLASSIFICATION",
    "CPU_OCI_BUILD_PLAN_SCHEMA_VERSION",
    "CPU_OCI_BUILD_PLAN_STATUS",
    "CanonicalSourceBundleInput",
    "CpuOciBuildPlanArtifacts",
    "ForagerMatchedV3CpuOciBuildPlanError",
    "build_matched_v3_cpu_oci_build_plan",
    "canonical_cpu_oci_build_plan_bytes",
    "parse_cpu_oci_build_plan",
    "validate_cpu_oci_build_plan",
]
