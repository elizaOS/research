#!/usr/bin/env python3
"""Isolated wheel-byte verifier for the matched-v3 CPU wheelhouse.

This helper is executed by :mod:`forager_matched_v3_cpu_wheelhouse` under an
exact caller-bound CPython with ``-I -S -B``.  Its first-party logic imports no
Alberta module and does not resolve, download, install, extract, or import a
candidate wheel.
PEP 440/508/425 parsing is provided only by a separately supplied, exact-hash
``packaging`` tool wheel added to ``sys.path`` after its bytes are verified.

The helper consumes a canonical request from an inherited descriptor, opens
candidate wheels relative to an inherited directory descriptor, and emits one
canonical JSON validation report on stdout.  It grants no artifact, execution,
qualification, evidence, or promotion authority.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import csv
import fcntl
import hashlib
import io
import json
import os
import re
import stat
import struct
import sys
import unicodedata
import zipfile
from email import policy
from email.parser import BytesParser
from pathlib import PurePosixPath
from types import ModuleType
from typing import Any, Final, NoReturn, cast

CAPTURE_MANIFEST_SCHEMA: Final = "alberta.forager_matched_v3.cpu_wheel_capture_manifest.v1"
VALIDATION_REPORT_SCHEMA: Final = "alberta.forager_matched_v3.cpu_wheel_validation_report.v1"
HELPER_REQUEST_SCHEMA: Final = "alberta.forager_matched_v3.cpu_wheel_helper_request.v1"

MAX_REQUEST_BYTES: Final = 16 * 1024 * 1024
MAX_REPORT_BYTES: Final = 16 * 1024 * 1024
MAX_WHEELS: Final = 256
MAX_WHEEL_BYTES: Final = 256 * 1024 * 1024
MAX_TOTAL_WHEEL_BYTES: Final = 1024 * 1024 * 1024
MAX_ZIP_MEMBERS_PER_WHEEL: Final = 100_000
MAX_ZIP_MEMBERS_TOTAL: Final = 1_000_000
MAX_ZIP_PATH_BYTES: Final = 4096
MAX_UNCOMPRESSED_BYTES_PER_WHEEL: Final = 2 * 1024 * 1024 * 1024
MAX_UNCOMPRESSED_BYTES_TOTAL: Final = 8 * 1024 * 1024 * 1024
MAX_METADATA_BYTES: Final = 8 * 1024 * 1024
MAX_WHEEL_METADATA_BYTES: Final = 256 * 1024
MAX_RECORD_BYTES: Final = 64 * 1024 * 1024
MAX_REQUIRES_DIST: Final = 4096
MAX_EXTRAS: Final = 512
READ_CHUNK_BYTES: Final = 1024 * 1024

_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_WHEEL_FILENAME_RE: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.+-]{0,250}\.whl\Z")
_ACCELERATOR_SEGMENT_RE: Final = re.compile(
    r"(?:cuda|rocm|nvidia|cublas|cufft|curand|cusolver|cusparse|cudnn|nccl|hip|gpu|xpu|tpu|cupy)"
    r"[0-9a-z]*\Z"
)
_SUPPORTED_RECORD_HASHES: Final = frozenset({"sha256", "sha384", "sha512"})
_SUPPORTED_COMPRESSION: Final = frozenset({zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED})
_SUPPORTED_METADATA_VERSIONS: Final = frozenset({"2.1", "2.2", "2.3", "2.4", "2.5"})
_MARKER_KEYS: Final = frozenset(
    {
        "implementation_name",
        "implementation_version",
        "os_name",
        "platform_machine",
        "platform_python_implementation",
        "platform_release",
        "platform_system",
        "platform_version",
        "python_full_version",
        "python_version",
        "sys_platform",
    }
)
_CRITICAL_VERSIONS: Final = {
    "continual-foragax": "0.55.0",
    "jax": "0.11.0",
    "jaxlib": "0.11.0",
}
_FORBIDDEN_DISTRIBUTIONS: Final = frozenset(
    {
        "alberta-framework",
        "continual-foragax-agents",
        "jax-cuda12-pjrt",
        "jax-cuda12-plugin",
    }
)
_CLAIM_KEYS: Final = frozenset(
    {
        "artifact_accepted",
        "execution_authority_granted",
        "image_qualified",
        "network_isolation_attested",
        "publication_authority_granted",
        "qualification_granted",
        "runtime_qualified",
        "scientific_evidence_created",
        "scientific_promotion_allowed",
        "wheelhouse_installation_reproduced",
    }
)
_EOCD = struct.Struct("<4s4H2LH")
_LOCAL_HEADER = struct.Struct("<4s5H3L2H")
_CENTRAL_HEADER = struct.Struct("<4s6H3L5H2L")


class WheelhouseHelperError(RuntimeError):
    """A candidate wheel or isolated verification request failed closed."""


def _fail(message: str) -> NoReturn:
    raise WheelhouseHelperError(message)


def _is_forbidden_accelerator_distribution(name: str) -> bool:
    return name in _FORBIDDEN_DISTRIBUTIONS or any(
        _ACCELERATOR_SEGMENT_RE.fullmatch(segment) is not None for segment in name.split("-")
    )


def _required_snapshot_seals() -> int:
    values = [
        getattr(fcntl, name, None)
        for name in ("F_SEAL_WRITE", "F_SEAL_GROW", "F_SEAL_SHRINK", "F_SEAL_SEAL")
    ]
    if any(type(value) is not int for value in values):
        _fail("helper requires full Linux memfd sealing support")
    return sum(cast(int, value) for value in values)


def _canonical_json(value: Any, *, newline: bool = True) -> bytes:
    raw = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return raw + (b"\n" if newline else b"")


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _claims() -> dict[str, bool]:
    return {key: False for key in sorted(_CLAIM_KEYS)}


def _hash_descriptor(descriptor: int, size: int) -> str:
    digest = hashlib.sha256()
    offset = 0
    while offset < size:
        block = os.pread(descriptor, min(READ_CHUNK_BYTES, size - offset), offset)
        if not block:
            _fail("bound file was truncated while hashed")
        digest.update(block)
        offset += len(block)
    if os.pread(descriptor, 1, size):
        _fail("bound file exceeds its declared size")
    return digest.hexdigest()


def _strict_json(raw: bytes, *, label: str, maximum: int) -> Any:
    if type(raw) is not bytes or not raw or len(raw) > maximum:
        _fail(f"{label} bytes exceed their bound")

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                _fail(f"{label} repeats JSON key {key!r}")
            result[key] = value
        return result

    def constant(value: str) -> NoReturn:
        _fail(f"{label} contains non-finite JSON constant {value}")

    try:
        return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WheelhouseHelperError(f"{label} is not strict JSON") from exc


def _exact_keys(value: dict[str, Any], expected: set[str], *, label: str) -> None:
    actual = set(value)
    if actual != expected:
        _fail(
            f"{label} keys differ: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _require_sha256(value: Any, *, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        _fail(f"{label} must be one lowercase SHA-256")
    return value


def _require_int(value: Any, *, label: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        _fail(f"{label} is outside its integer bound")
    return value


def _read_fd(descriptor: int, *, maximum: int, label: str) -> bytes:
    try:
        before = os.fstat(descriptor)
    except OSError as exc:
        raise WheelhouseHelperError(f"cannot stat {label}") from exc
    if not stat.S_ISREG(before.st_mode) or before.st_size <= 0 or before.st_size > maximum:
        _fail(f"{label} descriptor identity is invalid")
    chunks: list[bytes] = []
    offset = 0
    while offset < before.st_size:
        block = os.pread(
            descriptor,
            min(READ_CHUNK_BYTES, before.st_size - offset),
            offset,
        )
        if not block:
            _fail(f"{label} descriptor was truncated")
        chunks.append(block)
        offset += len(block)
    after = os.fstat(descriptor)
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        _fail(f"{label} descriptor changed while read")
    return b"".join(chunks)


class _PackagingApi:
    """Narrow dynamically loaded view of the exact supplied packaging tool."""

    def __init__(self, modules: dict[str, ModuleType]) -> None:
        self.package = modules["packaging"]
        self.markers = modules["packaging.markers"]
        self.requirements = modules["packaging.requirements"]
        self.specifiers = modules["packaging.specifiers"]
        self.tags = modules["packaging.tags"]
        self.utils = modules["packaging.utils"]
        self.version = modules["packaging.version"]

    def canonicalize_name(self, value: str) -> str:
        return str(self.utils.canonicalize_name(value))

    def version_value(self, value: str) -> Any:
        return self.version.Version(value)

    def canonical_version(self, value: str) -> str:
        return str(self.version.Version(value))

    def parse_wheel_filename(self, value: str) -> tuple[Any, Any, Any, Any]:
        return cast(tuple[Any, Any, Any, Any], self.utils.parse_wheel_filename(value))

    def parse_requirement(self, value: str) -> Any:
        return self.requirements.Requirement(value)

    def parse_specifier(self, value: str) -> Any:
        return self.specifiers.SpecifierSet(value)

    def parse_tag(self, value: str) -> set[Any]:
        return set(self.tags.parse_tag(value))

    def evaluate_marker(self, marker: Any, environment: dict[str, str]) -> bool:
        try:
            result = marker.evaluate(environment=environment, context="metadata")
        except TypeError:
            result = marker.evaluate(environment=environment)
        if type(result) is not bool:
            _fail("packaging marker evaluation returned a non-boolean")
        return result


def _load_packaging_tool(
    descriptor: int,
    *,
    expected_sha256: str,
    expected_version: str,
) -> _PackagingApi:
    before = os.fstat(descriptor)
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 0
        or stat.S_IMODE(before.st_mode) != 0o400
        or not 1 <= before.st_size <= MAX_WHEEL_BYTES
        or fcntl.fcntl(descriptor, fcntl.F_GETFL) & os.O_ACCMODE != os.O_RDONLY
        or fcntl.fcntl(descriptor, fcntl.F_GET_SEALS) & _required_snapshot_seals()
        != _required_snapshot_seals()
    ):
        _fail("packaging tool wheel is not an exact sealed read-only snapshot")
    observed = _hash_descriptor(descriptor, before.st_size)
    if observed != expected_sha256:
        _fail("packaging tool wheel SHA-256 differs")
    if any(name == "packaging" or name.startswith("packaging.") for name in sys.modules):
        _fail("ambient packaging was imported before the exact tool binding")
    tool_path = f"/proc/self/fd/{descriptor}"
    sys.path.insert(0, tool_path)
    try:
        imported: dict[str, ModuleType] = {}
        for name in (
            "packaging",
            "packaging.markers",
            "packaging.requirements",
            "packaging.specifiers",
            "packaging.tags",
            "packaging.utils",
            "packaging.version",
        ):
            imported[name] = __import__(name, fromlist=["*"])
    except Exception as exc:
        raise WheelhouseHelperError("exact packaging tool wheel cannot be imported") from exc
    package_version = getattr(imported["packaging"], "__version__", None)
    if package_version != expected_version:
        _fail("packaging tool version differs from its exact binding")
    packaging_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "packaging" or name.startswith("packaging.")
    }
    origins = {str(getattr(module, "__file__", "")) for module in packaging_modules.values()}
    after = os.fstat(descriptor)
    if (
        not origins
        or any(not origin.startswith(tool_path + "/") for origin in origins)
        or (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        or fcntl.fcntl(descriptor, fcntl.F_GETFL) & os.O_ACCMODE != os.O_RDONLY
        or fcntl.fcntl(descriptor, fcntl.F_GET_SEALS) & _required_snapshot_seals()
        != _required_snapshot_seals()
    ):
        _fail("packaging tool import escaped the exact supplied wheel")
    return _PackagingApi(imported)


def _safe_zip_path(value: str, *, directory: bool) -> PurePosixPath:
    if (
        not value
        or len(value.encode("utf-8")) > MAX_ZIP_PATH_BYTES
        or "\x00" in value
        or "\\" in value
        or unicodedata.normalize("NFC", value) != value
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        _fail("wheel ZIP contains a noncanonical member path")
    raw = value[:-1] if directory and value.endswith("/") else value
    path = PurePosixPath(raw)
    if (
        not raw
        or path.is_absolute()
        or path.as_posix() != raw
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        _fail("wheel ZIP contains an unsafe member path")
    return path


def _validate_eocd(handle: io.BufferedReader, size: int) -> tuple[int, int, int]:
    if size < _EOCD.size:
        _fail("wheel ZIP is shorter than its end record")
    window_size = min(size, 65_535 + _EOCD.size)
    handle.seek(size - window_size)
    tail = handle.read(window_size)
    offset = tail.rfind(b"PK\x05\x06")
    if offset < 0 or offset + _EOCD.size != len(tail):
        _fail("wheel ZIP EOCD is absent, commented, or not at EOF")
    (
        signature,
        disk_number,
        central_disk,
        disk_entries,
        total_entries,
        central_size,
        central_offset,
        comment_size,
    ) = _EOCD.unpack_from(tail, offset)
    if (
        signature != b"PK\x05\x06"
        or disk_number != 0
        or central_disk != 0
        or disk_entries != total_entries
        or total_entries in {0, 0xFFFF}
        or central_size == 0xFFFFFFFF
        or central_offset == 0xFFFFFFFF
        or comment_size != 0
    ):
        _fail("wheel ZIP EOCD uses an unsupported disk/ZIP64/comment layout")
    eocd_absolute = size - window_size + offset
    if central_offset + central_size != eocd_absolute:
        _fail("wheel ZIP central directory is not contiguous with EOCD")
    handle.seek(0)
    if handle.read(4) != b"PK\x03\x04":
        _fail("wheel ZIP has a prefix or lacks a local header at byte zero")
    return total_entries, central_offset, central_size


def _read_at(handle: io.BufferedReader, offset: int, size: int, *, label: str) -> bytes:
    handle.seek(offset)
    raw = handle.read(size)
    if len(raw) != size:
        _fail(f"wheel ZIP {label} is truncated")
    return raw


def _zip_name(raw: bytes, flag_bits: int, *, label: str) -> str:
    encoding = "utf-8" if flag_bits & 0x0800 else "cp437"
    try:
        return raw.decode(encoding)
    except UnicodeDecodeError as exc:
        raise WheelhouseHelperError(f"wheel ZIP {label} filename is malformed") from exc


def _local_record_span(
    handle: io.BufferedReader,
    info: zipfile.ZipInfo,
    *,
    central_offset: int,
) -> tuple[int, int]:
    raw = _read_at(
        handle,
        info.header_offset,
        _LOCAL_HEADER.size,
        label=f"local header {info.filename}",
    )
    (
        signature,
        _version_needed,
        flag_bits,
        compression,
        _modified_time,
        _modified_date,
        crc32,
        compressed_size,
        uncompressed_size,
        filename_size,
        extra_size,
    ) = _LOCAL_HEADER.unpack(raw)
    if (
        signature != b"PK\x03\x04"
        or flag_bits != info.flag_bits
        or compression != info.compress_type
    ):
        _fail("wheel ZIP local header differs from its central entry")
    raw_name = _read_at(
        handle,
        info.header_offset + _LOCAL_HEADER.size,
        filename_size,
        label=f"local filename {info.filename}",
    )
    if _zip_name(raw_name, flag_bits, label="local") != info.filename:
        _fail("wheel ZIP local filename differs from its central entry")
    data_offset = info.header_offset + _LOCAL_HEADER.size + filename_size + extra_size
    data_end = data_offset + info.compress_size
    if data_offset < info.header_offset or data_end > central_offset:
        _fail("wheel ZIP local payload crosses the central directory")
    if flag_bits & 0x0008:
        if (
            crc32 not in {0, info.CRC}
            or compressed_size
            not in {
                0,
                info.compress_size,
            }
            or uncompressed_size not in {0, info.file_size}
        ):
            _fail("wheel ZIP deferred local sizes contradict the central entry")
        descriptor_prefix = _read_at(
            handle,
            data_end,
            4,
            label=f"data descriptor {info.filename}",
        )
        signed = descriptor_prefix == b"PK\x07\x08"
        descriptor_size = 16 if signed else 12
        descriptor_raw = _read_at(
            handle,
            data_end + (4 if signed else 0),
            12,
            label=f"data descriptor {info.filename}",
        )
        descriptor_crc, descriptor_compressed, descriptor_uncompressed = struct.unpack(
            "<3L", descriptor_raw
        )
        if (
            descriptor_crc != info.CRC
            or descriptor_compressed != info.compress_size
            or descriptor_uncompressed != info.file_size
        ):
            _fail("wheel ZIP data descriptor differs from its central entry")
        data_end += descriptor_size
        if data_end > central_offset:
            _fail("wheel ZIP data descriptor crosses the central directory")
    elif (crc32, compressed_size, uncompressed_size) != (
        info.CRC,
        info.compress_size,
        info.file_size,
    ):
        _fail("wheel ZIP local sizes/hash differ from its central entry")
    return info.header_offset, data_end


def _validate_zip_layout(
    handle: io.BufferedReader,
    infos: list[zipfile.ZipInfo],
    *,
    central_offset: int,
    central_size: int,
) -> None:
    spans = sorted(
        (_local_record_span(handle, info, central_offset=central_offset) for info in infos),
        key=lambda item: item[0],
    )
    cursor = 0
    for start, end in spans:
        if start != cursor or end <= start:
            _fail("wheel ZIP local records contain a gap, overlap, or reordered alias")
        cursor = end
    if cursor != central_offset:
        _fail("wheel ZIP local records leave a gap before the central directory")

    cursor = central_offset
    central_end = central_offset + central_size
    for info in infos:
        raw = _read_at(
            handle,
            cursor,
            _CENTRAL_HEADER.size,
            label=f"central header {info.filename}",
        )
        fields = _CENTRAL_HEADER.unpack(raw)
        (
            signature,
            _version_made,
            _version_needed,
            flag_bits,
            compression,
            _modified_time,
            _modified_date,
            crc32,
            compressed_size,
            uncompressed_size,
            filename_size,
            extra_size,
            comment_size,
            disk_start,
            _internal_attr,
            _external_attr,
            local_offset,
        ) = fields
        if (
            signature != b"PK\x01\x02"
            or disk_start != 0
            or flag_bits != info.flag_bits
            or compression != info.compress_type
            or crc32 != info.CRC
            or compressed_size != info.compress_size
            or uncompressed_size != info.file_size
            or local_offset != info.header_offset
        ):
            _fail("wheel ZIP central header differs from its parsed entry")
        raw_name = _read_at(
            handle,
            cursor + _CENTRAL_HEADER.size,
            filename_size,
            label=f"central filename {info.filename}",
        )
        if _zip_name(raw_name, flag_bits, label="central") != info.filename:
            _fail("wheel ZIP central filename differs from its parsed entry")
        cursor += _CENTRAL_HEADER.size + filename_size + extra_size + comment_size
        if cursor > central_end:
            _fail("wheel ZIP central entry exceeds the central directory")
    if cursor != central_end:
        _fail("wheel ZIP central directory contains unindexed bytes")


def _decode_record_hash(value: str, *, algorithm: str) -> bytes:
    if algorithm not in _SUPPORTED_RECORD_HASHES or not value or "=" in value:
        _fail("wheel RECORD uses an unsupported or malformed hash")
    padding = "=" * (-len(value) % 4)
    try:
        decoded = base64.b64decode(
            (value + padding).encode("ascii"),
            altchars=b"-_",
            validate=True,
        )
    except (binascii.Error, UnicodeEncodeError, ValueError) as exc:
        raise WheelhouseHelperError("wheel RECORD hash encoding is malformed") from exc
    if len(decoded) != hashlib.new(algorithm).digest_size:
        _fail("wheel RECORD hash has the wrong length")
    return decoded


def _parse_record(
    raw: bytes,
    *,
    record_path: str,
    regular_paths: set[str],
) -> dict[str, tuple[str | None, bytes | None, int | None]]:
    if len(raw) > MAX_RECORD_BYTES:
        _fail("wheel RECORD exceeds its byte bound")
    try:
        reader = csv.reader(io.StringIO(raw.decode("utf-8"), newline=""), strict=True)
        rows = list(reader)
    except (UnicodeDecodeError, csv.Error) as exc:
        raise WheelhouseHelperError("wheel RECORD is not strict UTF-8 CSV") from exc
    if not rows:
        _fail("wheel RECORD is empty")
    parsed: dict[str, tuple[str | None, bytes | None, int | None]] = {}
    for index, row in enumerate(rows):
        if len(row) != 3:
            _fail(f"wheel RECORD row {index} does not have three fields")
        path_text, hash_text, size_text = row
        path = _safe_zip_path(path_text, directory=False).as_posix()
        if path in parsed:
            _fail("wheel RECORD repeats a path")
        if path == record_path:
            if hash_text or size_text:
                _fail("wheel RECORD must leave its own hash and size empty")
            parsed[path] = (None, None, None)
            continue
        if not hash_text or hash_text.count("=") != 1:
            _fail("wheel RECORD omits or malforms a payload hash")
        algorithm, encoded = hash_text.split("=", 1)
        digest = _decode_record_hash(encoded, algorithm=algorithm)
        if (
            not size_text
            or not size_text.isascii()
            or not size_text.isdecimal()
            or (len(size_text) > 1 and size_text.startswith("0"))
        ):
            _fail("wheel RECORD payload size is not canonical decimal")
        parsed[path] = (algorithm, digest, int(size_text))
    if set(parsed) != regular_paths or record_path not in parsed:
        _fail("wheel RECORD path set differs from regular ZIP members")
    return parsed


def _parse_headers(raw: bytes, *, label: str) -> Any:
    try:
        raw.decode("utf-8")
        message = BytesParser(policy=policy.default).parsebytes(raw, headersonly=True)
    except (UnicodeDecodeError, ValueError) as exc:
        raise WheelhouseHelperError(f"{label} is not strict UTF-8 metadata") from exc
    if message.defects:
        _fail(f"{label} contains email parser defects")
    return message


def _single_header(message: Any, name: str, *, label: str, optional: bool = False) -> str | None:
    values = message.get_all(name, [])
    if optional and not values:
        return None
    if len(values) != 1:
        _fail(f"{label} must contain exactly one {name} header")
    value = str(values[0]).strip()
    if not value or "\x00" in value:
        _fail(f"{label} {name} header is empty or unsafe")
    return value


def _dist_info_identity(
    directory: str,
    *,
    expected_name: str,
    expected_version: str,
    api: _PackagingApi,
) -> None:
    if not directory.endswith(".dist-info"):
        _fail("wheel dist-info directory suffix is invalid")
    stem = directory.removesuffix(".dist-info")
    if "-" not in stem:
        _fail("wheel dist-info directory lacks a version separator")
    raw_name, raw_version = stem.rsplit("-", 1)
    try:
        normalized_name = api.canonicalize_name(raw_name)
        normalized_version = api.canonical_version(raw_version)
    except Exception as exc:
        raise WheelhouseHelperError("wheel dist-info identity is invalid") from exc
    if normalized_name != expected_name or normalized_version != expected_version:
        _fail("wheel dist-info identity differs from filename/METADATA")


def _validate_data_paths(
    regular_paths: set[str],
    *,
    expected_name: str,
    expected_version: str,
    api: _PackagingApi,
) -> None:
    allowed = {"data", "headers", "platlib", "purelib", "scripts"}
    data_roots = {path.split("/", 1)[0] for path in regular_paths if ".data/" in path}
    for root in data_roots:
        if not root.endswith(".data") or "-" not in root.removesuffix(".data"):
            _fail("wheel contains a malformed .data root")
        raw_name, raw_version = root.removesuffix(".data").rsplit("-", 1)
        try:
            same = (
                api.canonicalize_name(raw_name) == expected_name
                and api.canonical_version(raw_version) == expected_version
            )
        except Exception as exc:
            raise WheelhouseHelperError("wheel .data identity is invalid") from exc
        if not same:
            _fail("wheel .data identity differs from its distribution")
        for member in regular_paths:
            if member.startswith(root + "/"):
                parts = PurePosixPath(member).parts
                if len(parts) < 3 or parts[1] not in allowed:
                    _fail("wheel .data member uses an unsupported installation role")


def _requirement_record(raw: str, *, api: _PackagingApi) -> dict[str, Any]:
    try:
        requirement = api.parse_requirement(raw)
        name = api.canonicalize_name(str(requirement.name))
        extras = sorted(api.canonicalize_name(str(item)) for item in requirement.extras)
    except Exception as exc:
        raise WheelhouseHelperError(f"invalid Requires-Dist requirement: {raw!r}") from exc
    if requirement.url is not None:
        _fail("direct URL/VCS/path requirements are forbidden")
    if len(extras) != len(set(extras)):
        _fail("requirement extras normalize to duplicates")
    return {
        "extras": extras,
        "marker": None if requirement.marker is None else str(requirement.marker),
        "name": name,
        "raw": raw,
        "specifier": str(requirement.specifier),
    }


def _verify_metadata(
    raw: bytes,
    *,
    filename_name: str,
    filename_version: str,
    target_python: str,
    api: _PackagingApi,
) -> dict[str, Any]:
    if len(raw) > MAX_METADATA_BYTES:
        _fail("wheel METADATA exceeds its byte bound")
    message = _parse_headers(raw, label="wheel METADATA")
    metadata_version = cast(str, _single_header(message, "Metadata-Version", label="METADATA"))
    if metadata_version not in _SUPPORTED_METADATA_VERSIONS:
        _fail("wheel METADATA version is unsupported")
    raw_name = cast(str, _single_header(message, "Name", label="METADATA"))
    raw_version = cast(str, _single_header(message, "Version", label="METADATA"))
    try:
        name = api.canonicalize_name(raw_name)
        version = api.canonical_version(raw_version)
    except Exception as exc:
        raise WheelhouseHelperError("wheel METADATA name/version is invalid") from exc
    if name != filename_name or version != filename_version:
        _fail("wheel METADATA identity differs from its filename")
    requires_python = _single_header(
        message,
        "Requires-Python",
        label="METADATA",
        optional=True,
    )
    if requires_python is not None:
        try:
            specifier = api.parse_specifier(requires_python)
            compatible = api.version_value(target_python) in specifier
        except Exception as exc:
            raise WheelhouseHelperError("wheel Requires-Python is invalid") from exc
        if not compatible:
            _fail("wheel Requires-Python excludes the frozen target")
    raw_extras = [str(item).strip() for item in message.get_all("Provides-Extra", [])]
    if len(raw_extras) > MAX_EXTRAS:
        _fail("wheel METADATA declares too many extras")
    try:
        normalized_extras = [api.canonicalize_name(item) for item in raw_extras]
    except Exception as exc:
        raise WheelhouseHelperError("wheel Provides-Extra is invalid") from exc
    if any(not item for item in normalized_extras):
        _fail("wheel Provides-Extra contains an empty value")
    extras = sorted(set(normalized_extras))
    raw_requirements = [str(item).strip() for item in message.get_all("Requires-Dist", [])]
    if len(raw_requirements) > MAX_REQUIRES_DIST:
        _fail("wheel METADATA declares too many Requires-Dist fields")
    requirements: list[dict[str, Any]] = []
    requirement_identities: set[bytes] = set()
    for item in raw_requirements:
        record = _requirement_record(item, api=api)
        identity = _canonical_json(record, newline=False)
        if identity not in requirement_identities:
            requirement_identities.add(identity)
            requirements.append(record)
    return {
        "metadata_version": metadata_version,
        "name": name,
        "provides_extra": extras,
        "requires_dist": requirements,
        "requires_python": requires_python,
        "version": version,
    }


def _verify_wheel_metadata(
    raw: bytes,
    *,
    filename_tags: set[str],
    build_text: str | None,
    api: _PackagingApi,
) -> dict[str, Any]:
    if len(raw) > MAX_WHEEL_METADATA_BYTES:
        _fail("wheel WHEEL metadata exceeds its byte bound")
    message = _parse_headers(raw, label="wheel WHEEL metadata")
    wheel_version = _single_header(message, "Wheel-Version", label="WHEEL")
    if wheel_version != "1.0":
        _fail("wheel WHEEL version is unsupported")
    generator = cast(str, _single_header(message, "Generator", label="WHEEL"))
    if len(generator) > 512 or any(not 32 <= ord(character) <= 126 for character in generator):
        _fail("wheel WHEEL Generator is not bounded printable ASCII")
    root_is_pure = _single_header(message, "Root-Is-Purelib", label="WHEEL")
    if root_is_pure not in {"true", "false"}:
        _fail("wheel Root-Is-Purelib is not canonical")
    tag_headers = [str(item).strip() for item in message.get_all("Tag", [])]
    if not tag_headers:
        _fail("wheel WHEEL metadata contains no Tag")
    try:
        wheel_tags = {str(tag) for header in tag_headers for tag in api.parse_tag(header)}
    except Exception as exc:
        raise WheelhouseHelperError("wheel WHEEL Tag header is invalid") from exc
    if wheel_tags != filename_tags:
        _fail("wheel WHEEL Tag set differs from its filename")
    if any(
        abi != "none" and platform == "any"
        for _python, abi, platform in (tag.rsplit("-", 2) for tag in filename_tags)
    ):
        _fail("wheel platform-any tag must use the none ABI")
    build = _single_header(message, "Build", label="WHEEL", optional=True)
    if build != build_text:
        _fail("wheel WHEEL Build differs from its filename build tag")
    return {
        "build": build,
        "generator": generator,
        "root_is_purelib": root_is_pure == "true",
        "tags": sorted(wheel_tags),
        "wheel_version": wheel_version,
    }


def _open_wheel(directory_fd: int, filename: str) -> tuple[int, os.stat_result]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        before = os.stat(filename, dir_fd=directory_fd, follow_symlinks=False)
        descriptor = os.open(filename, flags, dir_fd=directory_fd)
        opened = os.fstat(descriptor)
    except OSError as exc:
        raise WheelhouseHelperError(f"cannot safely open candidate wheel {filename!r}") from exc
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_nlink,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    identity_opened = (
        opened.st_dev,
        opened.st_ino,
        opened.st_mode,
        opened.st_nlink,
        opened.st_size,
        opened.st_mtime_ns,
        opened.st_ctime_ns,
    )
    if (
        identity_before != identity_opened
        or not stat.S_ISREG(opened.st_mode)
        or opened.st_nlink != 1
        or not 1 <= opened.st_size <= MAX_WHEEL_BYTES
    ):
        os.close(descriptor)
        _fail(f"candidate wheel {filename!r} is not a stable single-link regular file")
    return descriptor, opened


def _recheck_wheel(
    directory_fd: int,
    filename: str,
    descriptor: int,
    expected: os.stat_result,
) -> None:
    try:
        opened = os.fstat(descriptor)
        located = os.stat(filename, dir_fd=directory_fd, follow_symlinks=False)
    except OSError as exc:
        raise WheelhouseHelperError(f"candidate wheel {filename!r} changed") from exc
    expected_identity = (
        expected.st_dev,
        expected.st_ino,
        expected.st_mode,
        expected.st_nlink,
        expected.st_size,
        expected.st_mtime_ns,
        expected.st_ctime_ns,
    )
    for observed in (opened, located):
        identity = (
            observed.st_dev,
            observed.st_ino,
            observed.st_mode,
            observed.st_nlink,
            observed.st_size,
            observed.st_mtime_ns,
            observed.st_ctime_ns,
        )
        if identity != expected_identity:
            _fail(f"candidate wheel {filename!r} changed during verification")


def _verify_one_wheel(
    directory_fd: int,
    expected: dict[str, Any],
    target: dict[str, Any],
    *,
    api: _PackagingApi,
) -> dict[str, Any]:
    filename = expected["filename"]
    if (
        type(filename) is not str
        or len(filename.encode("ascii", "ignore")) != len(filename)
        or len(filename.encode("ascii")) > 255
        or _WHEEL_FILENAME_RE.fullmatch(filename) is None
    ):
        _fail("candidate wheel filename is not canonical bounded ASCII")
    descriptor, metadata = _open_wheel(directory_fd, filename)
    try:
        if metadata.st_size != expected["size_bytes"]:
            _fail(f"candidate wheel size differs: {filename}")
        observed_sha = _hash_descriptor(descriptor, metadata.st_size)
        if observed_sha != expected["sha256"]:
            _fail(f"candidate wheel SHA-256 differs: {filename}")
        try:
            parsed_name, parsed_version, parsed_build, parsed_tags = api.parse_wheel_filename(
                filename
            )
            name = api.canonicalize_name(str(parsed_name))
            version = str(parsed_version)
            filename_tags = {str(tag) for tag in parsed_tags}
        except Exception as exc:
            raise WheelhouseHelperError(
                f"candidate wheel filename cannot be parsed: {filename}"
            ) from exc
        if not filename_tags:
            _fail("candidate wheel filename contains no tags")
        compatible_tags = cast(list[str], target["compatible_tags"])
        compatible_set = set(compatible_tags)
        intersections = filename_tags.intersection(compatible_set)
        if not intersections:
            _fail(f"candidate wheel is incompatible with the frozen target: {filename}")
        best_rank = min(compatible_tags.index(tag) for tag in intersections)
        build_text: str | None = None
        if parsed_build:
            build_tuple = cast(tuple[Any, Any], parsed_build)
            build_text = f"{build_tuple[0]}{build_tuple[1]}"

        duplicate = os.dup(descriptor)
        try:
            with os.fdopen(duplicate, "rb", closefd=True) as raw_handle:
                buffered = raw_handle
                expected_entries, central_offset, central_size = _validate_eocd(
                    buffered,
                    metadata.st_size,
                )
                buffered.seek(0)
                with zipfile.ZipFile(buffered, mode="r", allowZip64=False) as archive:
                    if archive.comment:
                        _fail("wheel ZIP comment is forbidden")
                    infos = archive.infolist()
                    if (
                        len(infos) != expected_entries
                        or not 1 <= len(infos) <= MAX_ZIP_MEMBERS_PER_WHEEL
                    ):
                        _fail("wheel ZIP member count differs or exceeds its bound")
                    regular: dict[str, zipfile.ZipInfo] = {}
                    aliases: set[str] = set()
                    logical_paths: list[tuple[str, bool]] = []
                    structures: list[dict[str, Any]] = []
                    uncompressed_total = 0
                    compressed_total = 0
                    header_offsets: set[int] = set()
                    dist_info_roots: set[str] = set()
                    for info in infos:
                        directory = info.is_dir()
                        path = _safe_zip_path(info.filename, directory=directory)
                        canonical = path.as_posix() + ("/" if directory else "")
                        alias = canonical.casefold()
                        if alias in aliases:
                            _fail("wheel ZIP repeats or casefold-aliases a member")
                        aliases.add(alias)
                        logical_paths.append((path.as_posix(), directory))
                        if (
                            info.flag_bits & 1
                            or info.compress_type not in _SUPPORTED_COMPRESSION
                            or info.header_offset in header_offsets
                            or not 0 <= info.header_offset < central_offset
                            or info.comment
                        ):
                            _fail("wheel ZIP member structure is unsupported")
                        header_offsets.add(info.header_offset)
                        mode = (info.external_attr >> 16) & 0xFFFF
                        node_type = stat.S_IFMT(mode)
                        if directory:
                            if info.file_size != 0 or node_type not in {0, stat.S_IFDIR}:
                                _fail("wheel ZIP directory entry is noncanonical")
                        elif node_type not in {0, stat.S_IFREG}:
                            _fail("wheel ZIP contains a symlink or special member")
                        else:
                            regular[path.as_posix()] = info
                            top = path.parts[0]
                            if top.endswith(".dist-info"):
                                dist_info_roots.add(top)
                        uncompressed_total += info.file_size
                        compressed_total += info.compress_size
                        if uncompressed_total > MAX_UNCOMPRESSED_BYTES_PER_WHEEL:
                            _fail("wheel ZIP uncompressed bytes exceed their bound")
                        structures.append(
                            {
                                "compressed_size": info.compress_size,
                                "compression": info.compress_type,
                                "crc32": info.CRC,
                                "external_attr": info.external_attr,
                                "path": canonical,
                                "size": info.file_size,
                            }
                        )
                    _validate_zip_layout(
                        buffered,
                        infos,
                        central_offset=central_offset,
                        central_size=central_size,
                    )
                    regular_folded = {path.casefold() for path in regular}
                    for logical, directory in logical_paths:
                        folded = logical.casefold()
                        if directory and folded in regular_folded:
                            _fail("wheel ZIP contains a regular-path prefix conflict")
                        parts = folded.split("/")
                        for stop in range(1, len(parts)):
                            if "/".join(parts[:stop]) in regular_folded:
                                _fail("wheel ZIP contains a regular-path prefix conflict")
                    if len(dist_info_roots) != 1:
                        _fail("wheel must contain exactly one dist-info directory")
                    dist_info = next(iter(dist_info_roots))
                    metadata_path = f"{dist_info}/METADATA"
                    wheel_path = f"{dist_info}/WHEEL"
                    record_path = f"{dist_info}/RECORD"
                    required_metadata = {metadata_path, wheel_path, record_path}
                    if not required_metadata.issubset(regular):
                        _fail("wheel omits METADATA, WHEEL, or RECORD")
                    if any(
                        path in regular
                        for path in (f"{dist_info}/RECORD.jws", f"{dist_info}/RECORD.p7s")
                    ):
                        _fail("signed wheel RECORD sidecars are unsupported")
                    if regular[record_path].file_size > MAX_RECORD_BYTES:
                        _fail("wheel RECORD exceeds its byte bound")
                    if regular[metadata_path].file_size > MAX_METADATA_BYTES:
                        _fail("wheel METADATA exceeds its byte bound")
                    if regular[wheel_path].file_size > MAX_WHEEL_METADATA_BYTES:
                        _fail("wheel WHEEL metadata exceeds its byte bound")
                    record_raw = archive.read(record_path)
                    record = _parse_record(
                        record_raw,
                        record_path=record_path,
                        regular_paths=set(regular),
                    )
                    captured: dict[str, bytes] = {record_path: record_raw}
                    payload_inventory: list[dict[str, Any]] = []
                    for member_path in sorted(regular):
                        info = regular[member_path]
                        record_algorithm, record_digest, declared_size = record[member_path]
                        sha256 = hashlib.sha256()
                        record_hasher = (
                            None if record_algorithm is None else hashlib.new(record_algorithm)
                        )
                        count = 0
                        try:
                            with archive.open(info, mode="r") as member:
                                while True:
                                    block = member.read(READ_CHUNK_BYTES)
                                    if not block:
                                        break
                                    count += len(block)
                                    if count > info.file_size:
                                        _fail("wheel ZIP member exceeds its declared size")
                                    sha256.update(block)
                                    if record_hasher is not None:
                                        record_hasher.update(block)
                                    if member_path in {metadata_path, wheel_path}:
                                        captured[member_path] = (
                                            captured.get(member_path, b"") + block
                                        )
                        except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
                            raise WheelhouseHelperError(
                                f"wheel ZIP payload is corrupt: {member_path}"
                            ) from exc
                        if count != info.file_size:
                            _fail("wheel ZIP member size differs after streaming")
                        if declared_size is not None and declared_size != count:
                            _fail("wheel RECORD size differs from ZIP payload")
                        if (
                            record_hasher is not None
                            and record_digest is not None
                            and record_hasher.digest() != record_digest
                        ):
                            _fail("wheel RECORD hash differs from ZIP payload")
                        payload_inventory.append(
                            {
                                "path": member_path,
                                "sha256": sha256.hexdigest(),
                                "size_bytes": count,
                            }
                        )
                    metadata_value = _verify_metadata(
                        captured[metadata_path],
                        filename_name=name,
                        filename_version=version,
                        target_python=cast(str, target["python_version"]),
                        api=api,
                    )
                    wheel_value = _verify_wheel_metadata(
                        captured[wheel_path],
                        filename_tags=filename_tags,
                        build_text=build_text,
                        api=api,
                    )
                    _dist_info_identity(
                        dist_info,
                        expected_name=name,
                        expected_version=version,
                        api=api,
                    )
                    _validate_data_paths(
                        set(regular),
                        expected_name=name,
                        expected_version=version,
                        api=api,
                    )
        finally:
            # os.fdopen closes the duplicate on the normal path and most failures.
            try:
                os.close(duplicate)
            except OSError:
                pass
        _recheck_wheel(directory_fd, filename, descriptor, metadata)
    finally:
        os.close(descriptor)

    metadata_sha256 = _sha256_bytes(captured[metadata_path])
    wheel_metadata_sha256 = _sha256_bytes(captured[wheel_path])
    record_sha256 = _sha256_bytes(record_raw)
    payload_inventory_sha256 = _sha256_bytes(_canonical_json(payload_inventory, newline=False))
    metadata_record = {
        **metadata_value,
        "path": metadata_path,
        "sha256": metadata_sha256,
        "size_bytes": len(captured[metadata_path]),
    }
    wheel_record = {
        **wheel_value,
        "path": wheel_path,
        "sha256": wheel_metadata_sha256,
        "size_bytes": len(captured[wheel_path]),
    }
    record_record = {
        "entries_sha256": payload_inventory_sha256,
        "entry_count": len(payload_inventory),
        "path": record_path,
        "sha256": record_sha256,
        "size_bytes": len(record_raw),
    }
    return {
        "best_compatible_tag_rank": best_rank,
        "build_tag": build_text,
        "compressed_size_bytes": compressed_total,
        "dist_info_directory": dist_info,
        "filename": filename,
        "metadata": metadata_record,
        "metadata_sha256": metadata_sha256,
        "name": name,
        "payload_file_count": len(payload_inventory),
        "payload_inventory_sha256": payload_inventory_sha256,
        "record": record_record,
        "record_sha256": record_sha256,
        "sha256": expected["sha256"],
        "size_bytes": expected["size_bytes"],
        "tags": sorted(filename_tags),
        "uncompressed_size_bytes": uncompressed_total,
        "version": version,
        "wheel": wheel_record,
        "wheel_metadata": wheel_value,
        "wheel_metadata_sha256": wheel_metadata_sha256,
        "zip_member_count": len(structures),
        "zip_structure_sha256": _sha256_bytes(_canonical_json(structures, newline=False)),
    }


def _requirement_object(record: dict[str, Any], *, api: _PackagingApi) -> Any:
    try:
        return api.parse_requirement(cast(str, record["raw"]))
    except Exception as exc:
        raise WheelhouseHelperError("validated requirement cannot be replayed") from exc


def _requirement_active_contexts(
    requirement: Any,
    *,
    extras: set[str],
    environment: dict[str, str],
    api: _PackagingApi,
) -> list[str]:
    contexts = ["", *sorted(extras)]
    if requirement.marker is None:
        return contexts
    active: list[str] = []
    for extra in contexts:
        marker_environment = dict(environment)
        marker_environment["extra"] = extra
        try:
            if api.evaluate_marker(requirement.marker, marker_environment):
                active.append(extra)
        except Exception as exc:
            raise WheelhouseHelperError("dependency marker evaluation failed") from exc
    return active


def _assert_requirement_target(
    requirement: Any,
    packages: dict[str, dict[str, Any]],
    *,
    api: _PackagingApi,
) -> tuple[str, set[str]]:
    name = api.canonicalize_name(str(requirement.name))
    target = packages.get(name)
    if target is None:
        _fail(f"active dependency is absent from wheelhouse: {name}")
    try:
        compatible = api.version_value(cast(str, target["version"])) in requirement.specifier
    except Exception as exc:
        raise WheelhouseHelperError("dependency version constraint cannot be evaluated") from exc
    if not compatible:
        _fail(f"active dependency version constraint is unsatisfied: {name}")
    extras = {api.canonicalize_name(str(item)) for item in requirement.extras}
    declared = set(cast(list[str], cast(dict[str, Any], target["metadata"])["provides_extra"]))
    if not extras.issubset(declared):
        _fail(f"active dependency requests an undeclared extra: {name}")
    return name, extras


def _validate_closure(
    packages: list[dict[str, Any]],
    root_requirements: list[str],
    marker_environment: dict[str, str],
    *,
    api: _PackagingApi,
) -> dict[str, Any]:
    by_name: dict[str, dict[str, Any]] = {}
    for package in packages:
        name = cast(str, package["name"])
        if name in by_name:
            _fail(f"wheelhouse repeats normalized distribution {name}")
        by_name[name] = package
    for name, version in _CRITICAL_VERSIONS.items():
        if name not in by_name or by_name[name]["version"] != version:
            _fail(f"wheelhouse critical version differs: {name}=={version}")
    if any(_is_forbidden_accelerator_distribution(name) for name in by_name):
        _fail("wheelhouse contains a forbidden root/accelerator distribution")

    reachable: set[str] = set()
    activated: dict[str, set[str]] = {}
    parsed_roots: list[dict[str, Any]] = []
    for raw in root_requirements:
        record = _requirement_record(raw, api=api)
        requirement = _requirement_object(record, api=api)
        contexts = _requirement_active_contexts(
            requirement,
            extras=set(),
            environment=marker_environment,
            api=api,
        )
        if not contexts:
            _fail("root requirement marker is false for the frozen target")
        name, extras = _assert_requirement_target(requirement, by_name, api=api)
        reachable.add(name)
        activated.setdefault(name, set()).update(extras)
        parsed_roots.append(record)

    edge_map: dict[bytes, dict[str, Any]] = {}
    budget = (
        len(by_name)
        + sum(
            len(cast(list[str], cast(dict[str, Any], package["metadata"])["provides_extra"]))
            for package in packages
        )
        + 1
    )
    for _iteration in range(budget):
        before = (
            frozenset(reachable),
            tuple(sorted((name, tuple(sorted(values))) for name, values in activated.items())),
        )
        for source in sorted(reachable):
            source_extras = activated.setdefault(source, set())
            requirements = cast(
                list[dict[str, Any]],
                cast(dict[str, Any], by_name[source]["metadata"])["requires_dist"],
            )
            for record in requirements:
                requirement = _requirement_object(record, api=api)
                contexts = _requirement_active_contexts(
                    requirement,
                    extras=source_extras,
                    environment=marker_environment,
                    api=api,
                )
                if not contexts:
                    continue
                target, requested_extras = _assert_requirement_target(
                    requirement,
                    by_name,
                    api=api,
                )
                reachable.add(target)
                activated.setdefault(target, set()).update(requested_extras)
                edge = {
                    "active_contexts": contexts,
                    "requirement": record,
                    "source": source,
                    "target": target,
                }
                edge_identity = {
                    "requirement": record,
                    "source": source,
                    "target": target,
                }
                edge_map[_canonical_json(edge_identity, newline=False)] = edge
        after = (
            frozenset(reachable),
            tuple(sorted((name, tuple(sorted(values))) for name, values in activated.items())),
        )
        if after == before:
            break
    else:
        _fail("dependency extras fixed point exceeded its bound")
    if reachable != set(by_name):
        _fail(
            "wheelhouse contains unreachable extra distributions: "
            + ", ".join(sorted(set(by_name) - reachable))
        )
    edges = sorted(edge_map.values(), key=lambda item: _canonical_json(item, newline=False))
    extras_value = {name: sorted(activated.get(name, set())) for name in sorted(reachable)}
    graph_identity = {
        "activated_extras": extras_value,
        "edges": edges,
        "root_requirements": parsed_roots,
    }
    return {
        **graph_identity,
        "dependency_graph_sha256": _sha256_bytes(_canonical_json(graph_identity, newline=False)),
        "reachable_distributions": sorted(reachable),
    }


def verify_request(
    request: dict[str, Any],
    *,
    wheel_directory_fd: int,
    packaging_tool_fd: int,
) -> dict[str, Any]:
    _exact_keys(
        request,
        {"manifest", "packaging_tool", "schema_version"},
        label="helper request",
    )
    if request["schema_version"] != HELPER_REQUEST_SCHEMA:
        _fail("helper request schema is unsupported")
    manifest = request["manifest"]
    tool = request["packaging_tool"]
    if type(manifest) is not dict or type(tool) is not dict:
        _fail("helper request manifest/tool must be objects")
    tool_value = cast(dict[str, Any], tool)
    _exact_keys(tool_value, {"sha256", "version"}, label="packaging tool")
    api = _load_packaging_tool(
        packaging_tool_fd,
        expected_sha256=_require_sha256(tool_value["sha256"], label="packaging tool"),
        expected_version=cast(str, tool_value["version"]),
    )
    manifest_value = cast(dict[str, Any], manifest)
    if manifest_value.get("schema_version") != CAPTURE_MANIFEST_SCHEMA:
        _fail("helper manifest schema is unsupported")
    target = manifest_value.get("target")
    wheels = manifest_value.get("wheels")
    roots = manifest_value.get("root_requirements")
    if type(target) is not dict or type(wheels) is not list or type(roots) is not list:
        _fail("helper manifest target/wheels/root requirements are invalid")
    wheel_values = cast(list[dict[str, Any]], wheels)
    if not 1 <= len(wheel_values) <= MAX_WHEELS:
        _fail("helper wheel count is outside its bound")
    expected_names = [cast(str, wheel["filename"]) for wheel in wheel_values]
    try:
        observed_names = sorted(os.listdir(wheel_directory_fd))
    except OSError as exc:
        raise WheelhouseHelperError("cannot enumerate staged wheel directory") from exc
    if observed_names != expected_names:
        _fail("staged wheel directory file set differs from capture manifest")
    target_value = cast(dict[str, Any], target)
    marker_environment = target_value.get("marker_environment")
    if type(marker_environment) is not dict or set(marker_environment) != _MARKER_KEYS:
        _fail("frozen marker environment is incomplete")
    marker_value = cast(dict[str, str], marker_environment)
    packages: list[dict[str, Any]] = []
    total_bytes = 0
    total_uncompressed = 0
    total_members = 0
    for wheel in wheel_values:
        package = _verify_one_wheel(
            wheel_directory_fd,
            wheel,
            target_value,
            api=api,
        )
        packages.append(package)
        total_bytes += cast(int, package["size_bytes"])
        total_uncompressed += cast(int, package["uncompressed_size_bytes"])
        total_members += cast(int, package["zip_member_count"])
        if total_bytes > MAX_TOTAL_WHEEL_BYTES:
            _fail("wheelhouse bytes exceed their global bound")
        if total_uncompressed > MAX_UNCOMPRESSED_BYTES_TOTAL:
            _fail("wheelhouse uncompressed bytes exceed their global bound")
        if total_members > MAX_ZIP_MEMBERS_TOTAL:
            _fail("wheelhouse ZIP members exceed their global bound")
    packages.sort(key=lambda item: cast(str, item["name"]))
    closure = _validate_closure(
        packages,
        cast(list[str], roots),
        marker_value,
        api=api,
    )
    inventory = [
        {
            "filename": package["filename"],
            "name": package["name"],
            "sha256": package["sha256"],
            "size_bytes": package["size_bytes"],
            "version": package["version"],
        }
        for package in packages
    ]
    report_body: dict[str, Any] = {
        "capture_manifest_body_sha256": manifest_value["manifest_body_sha256"],
        "claims": _claims(),
        "classification": "disconnected_wheel_bytes_validation_non_authorizing",
        "closure": closure,
        "inventory_sha256": _sha256_bytes(_canonical_json(inventory, newline=False)),
        "package_count": len(packages),
        "packages": packages,
        "packaging_tool": {
            "sha256": tool_value["sha256"],
            "version": tool_value["version"],
        },
        "schema_version": VALIDATION_REPORT_SCHEMA,
        "status": "content_verified_unqualified_non_authorizing",
        "total_uncompressed_bytes": total_uncompressed,
        "total_wheel_bytes": total_bytes,
        "zip_member_count": total_members,
    }
    report_body["report_body_sha256"] = _sha256_bytes(_canonical_json(report_body, newline=False))
    return report_body


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("command", choices=("verify",))
    parser.add_argument("--request-fd", required=True, type=int)
    parser.add_argument("--wheel-directory-fd", required=True, type=int)
    parser.add_argument("--packaging-wheel-fd", required=True, type=int)
    return parser


def main() -> int:
    try:
        arguments = _parser().parse_args()
        request_raw = _read_fd(
            arguments.request_fd,
            maximum=MAX_REQUEST_BYTES,
            label="helper request",
        )
        request = _strict_json(
            request_raw,
            label="helper request",
            maximum=MAX_REQUEST_BYTES,
        )
        if type(request) is not dict:
            _fail("helper request must be one JSON object")
        report = verify_request(
            cast(dict[str, Any], request),
            wheel_directory_fd=arguments.wheel_directory_fd,
            packaging_tool_fd=arguments.packaging_wheel_fd,
        )
        raw = _canonical_json(report)
        if len(raw) > MAX_REPORT_BYTES:
            _fail("helper validation report exceeds its byte bound")
        sys.stdout.buffer.write(raw)
        return 0
    except (WheelhouseHelperError, OSError, ValueError, zipfile.BadZipFile) as exc:
        sys.stderr.write(f"wheelhouse-helper: {exc}\n")
        return 2


if __name__ == "__main__":  # pragma: no cover - exercised through the host boundary
    raise SystemExit(main())
