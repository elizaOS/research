"""Strict, authority-free matched-v3 raw-reward score ingestion.

The accepted artifact is intentionally narrower than general NPZ: exactly one
uncompressed ZIP64-local member named ``rewards.npy`` whose bytes are the
canonical NumPy-v1 header for a C-order ``|i1`` vector of length 499,712,
followed by exactly one signed-int8 byte per reward.  The ZIP metadata and
record layout are fixed to the deterministic bytes emitted by ``numpy.savez``
for that array shape.  General ZIP/NPY parsing and decompression are forbidden.

Ingestion accepts an already-open regular-file descriptor, reads it with
``pread`` without changing its offset, and compares descriptor identity before
and after the bounded read.  It grants no task, configuration, candidate,
evidence, execution, promotion, or qualification authority.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import struct
import zlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final, cast

from alberta_framework.benchmarks import forager_matched_v3_protocol as protocol

SCORE_RECEIPT_SCHEMA_VERSION: Final = "alberta.forager_matched_v3_score_receipt.v1"
NPZ_CONTAINER_SCHEMA_VERSION: Final = "alberta.forager_matched_v3_reward_npz.v1"
RAW_TRACE_ENCODING_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3_raw_reward_trace.int8.v1"
)
RAW_TRACE_ENCODING: Final = "signed_int8_twos_complement_c_order_one_byte_per_step"
RAW_TRACE_DIGEST_DOMAIN: Final = b"alberta.forager.matched_v3.raw_reward_trace.int8.v1"
NPZ_MEMBER_NAME: Final = "rewards.npy"

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_MAX_RECEIPT_BYTES: Final = 64 * 1024
_READ_CHUNK_BYTES: Final = 64 * 1024
# This value is embedded in the v1 NPY header bytes.  _bound_metric_descriptor
# rejects any protocol horizon drift before ingestion.
_NPY_FORMAT_HORIZON: Final = 499_712
_NPY_MAGIC_AND_VERSION: Final = b"\x93NUMPY\x01\x00"
_NPY_HEADER_DICTIONARY: Final = (
    "{'descr': '|i1', 'fortran_order': False, "
    f"'shape': ({_NPY_FORMAT_HORIZON},), }}"
).encode("ascii")
_NPY_HEADER_PAYLOAD_SIZE: Final = 118
_NPY_HEADER_PAYLOAD: Final = (
    _NPY_HEADER_DICTIONARY
    + b" " * (_NPY_HEADER_PAYLOAD_SIZE - len(_NPY_HEADER_DICTIONARY) - 1)
    + b"\n"
)
_CANONICAL_NPY_HEADER: Final = (
    _NPY_MAGIC_AND_VERSION
    + struct.pack("<H", _NPY_HEADER_PAYLOAD_SIZE)
    + _NPY_HEADER_PAYLOAD
)
_NPY_HEADER_SIZE: Final = len(_CANONICAL_NPY_HEADER)
_NPY_MEMBER_SIZE: Final = _NPY_HEADER_SIZE + protocol.MATCHED_V3_HORIZON

_ZIP_LOCAL_HEADER = struct.Struct("<IHHHHHIIIHH")
_ZIP_CENTRAL_HEADER = struct.Struct("<IHHHHHHIIIHHHHHII")
_ZIP_END_RECORD = struct.Struct("<IHHHHIIH")
_ZIP64_LOCAL_EXTRA = struct.pack("<HHQQ", 1, 16, _NPY_MEMBER_SIZE, _NPY_MEMBER_SIZE)
_MEMBER_NAME_BYTES: Final = NPZ_MEMBER_NAME.encode("ascii")
_ZIP_LOCAL_SIGNATURE: Final = 0x04034B50
_ZIP_CENTRAL_SIGNATURE: Final = 0x02014B50
_ZIP_END_SIGNATURE: Final = 0x06054B50
_ZIP_DOS_DATE_1980_01_01: Final = 33
_ZIP_EXTERNAL_ATTR: Final = 0x01800000
_ZIP_DATA_OFFSET: Final = (
    _ZIP_LOCAL_HEADER.size + len(_MEMBER_NAME_BYTES) + len(_ZIP64_LOCAL_EXTRA)
)
_ZIP_CENTRAL_OFFSET: Final = _ZIP_DATA_OFFSET + _NPY_MEMBER_SIZE
_ZIP_CENTRAL_SIZE: Final = _ZIP_CENTRAL_HEADER.size + len(_MEMBER_NAME_BYTES)
_ZIP_END_OFFSET: Final = _ZIP_CENTRAL_OFFSET + _ZIP_CENTRAL_SIZE
CANONICAL_NPZ_SIZE_BYTES: Final = _ZIP_END_OFFSET + _ZIP_END_RECORD.size

def _authority_denial() -> dict[str, bool]:
    """Construct the fixed denial object without consulting mutable module state."""

    return {
        "task_identity_authority": False,
        "configuration_identity_authority": False,
        "candidate_identity_authority": False,
        "scientific_evidence_authority": False,
        "qualification_authority": False,
        "execution_authority": False,
        "promotion_authority": False,
    }


class ForagerMatchedV3ScorerError(ValueError):
    """A reward artifact or detached score receipt violated the v3 contract."""


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
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
        raise ForagerMatchedV3ScorerError("value is not canonical JSON") from exc
    if len(raw) > _MAX_RECEIPT_BYTES:
        raise ForagerMatchedV3ScorerError("score receipt exceeds the JSON byte limit")
    return raw


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _require_sha256(value: object, path: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ForagerMatchedV3ScorerError(f"{path} must be a lowercase SHA-256 digest")
    return value


def _require_exact_integer(value: object, path: str) -> int:
    if type(value) is not int:
        raise ForagerMatchedV3ScorerError(f"{path} must be an exact integer")
    return value


def _require_object(value: object, path: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise ForagerMatchedV3ScorerError(f"{path} must be a plain JSON object")
    return cast(dict[str, Any], value)


def _require_exact_keys(value: Mapping[str, Any], path: str, expected: set[str]) -> None:
    if set(value) != expected:
        raise ForagerMatchedV3ScorerError(f"{path} fields are not exact")


def _bound_metric_descriptor() -> dict[str, Any]:
    descriptor = protocol.cumulative_reward_metric_descriptor()
    canonical = protocol.canonical_cumulative_reward_metric_bytes()
    if (
        _sha256(canonical) != protocol.CUMULATIVE_REWARD_METRIC_SHA256
        or protocol.MATCHED_V3_HORIZON != _NPY_FORMAT_HORIZON
        or descriptor.get("schema_version")
        != protocol.CUMULATIVE_REWARD_METRIC_SCHEMA_VERSION
        or descriptor.get("horizon") != protocol.MATCHED_V3_HORIZON
        or descriptor.get("raw_reward_values")
        != list(protocol.MATCHED_V3_RAW_REWARD_VALUES)
        or descriptor.get("accumulation") != "ordered_exact_integer_sum"
        or descriptor.get("trace_completeness_required") is not True
        or descriptor.get("out_of_set_reward_rejected") is not True
    ):
        raise ForagerMatchedV3ScorerError("v3 metric descriptor binding is invalid")
    return descriptor


@dataclass(frozen=True, slots=True)
class MatchedV3ScoreReceipt:
    """Content-addressed scalar result with no scientific or identity authority."""

    cumulative_score: int
    raw_trace_sha256: str
    artifact_sha256: str
    artifact_size_bytes: int

    def __post_init__(self) -> None:
        score = _require_exact_integer(self.cumulative_score, "cumulative_score")
        if not protocol.MATCHED_V3_SCORE_MINIMUM <= score <= protocol.MATCHED_V3_SCORE_MAXIMUM:
            raise ForagerMatchedV3ScorerError("cumulative_score is outside metric bounds")
        _require_sha256(self.raw_trace_sha256, "raw_trace_sha256")
        _require_sha256(self.artifact_sha256, "artifact_sha256")
        size = _require_exact_integer(self.artifact_size_bytes, "artifact_size_bytes")
        if size != CANONICAL_NPZ_SIZE_BYTES:
            raise ForagerMatchedV3ScorerError("artifact_size_bytes is not canonical")

    def to_body(self) -> dict[str, Any]:
        """Return a detached receipt body without its self-digest."""
        return {
            "schema_version": SCORE_RECEIPT_SCHEMA_VERSION,
            "metric": {
                "descriptor": _bound_metric_descriptor(),
                "sha256": protocol.CUMULATIVE_REWARD_METRIC_SHA256,
            },
            "score": {
                "accumulation": "ordered_exact_integer_sum",
                "cumulative_reward": self.cumulative_score,
            },
            "raw_trace": {
                "encoding_schema_version": RAW_TRACE_ENCODING_SCHEMA_VERSION,
                "encoding": RAW_TRACE_ENCODING,
                "digest_domain": RAW_TRACE_DIGEST_DOMAIN.decode("ascii"),
                "digest_framing": (
                    "uint32be_domain_length_then_ascii_domain_then_"
                    "uint64be_trace_length_then_trace_bytes"
                ),
                "horizon": protocol.MATCHED_V3_HORIZON,
                "raw_reward_values": list(protocol.MATCHED_V3_RAW_REWARD_VALUES),
                "sha256": self.raw_trace_sha256,
            },
            "artifact": {
                "container_schema_version": NPZ_CONTAINER_SCHEMA_VERSION,
                "member_name": NPZ_MEMBER_NAME,
                "compression": "stored",
                "npy_format_version": "1.0",
                "dtype": "|i1",
                "fortran_order": False,
                "shape": [protocol.MATCHED_V3_HORIZON],
                "sha256": self.artifact_sha256,
                "size_bytes": self.artifact_size_bytes,
            },
            "authority": _authority_denial(),
        }

    def canonical_body(self) -> bytes:
        """Return canonical bytes covered by :attr:`receipt_sha256`."""
        return _canonical_json_bytes(self.to_body())

    @property
    def receipt_sha256(self) -> str:
        """Return the content address of the canonical receipt body."""
        return _sha256(self.canonical_body())

    def to_payload(self) -> dict[str, Any]:
        """Return a detached receipt including its content address."""
        payload = self.to_body()
        payload["receipt_sha256"] = self.receipt_sha256
        return payload

    def canonical_json(self) -> bytes:
        """Return canonical replay bytes including the receipt content address."""
        return _canonical_json_bytes(self.to_payload())


def _trace_score_and_sha256(raw_trace: bytes) -> tuple[int, str]:
    if type(raw_trace) is not bytes:
        raise ForagerMatchedV3ScorerError("raw reward trace must be exact bytes")
    if len(raw_trace) != protocol.MATCHED_V3_HORIZON:
        raise ForagerMatchedV3ScorerError(
            f"raw reward trace horizon must equal {protocol.MATCHED_V3_HORIZON}"
        )
    cumulative_score = 0
    for index, encoded in enumerate(raw_trace):
        if encoded == 255:
            cumulative_score -= 1
        elif encoded == 0:
            continue
        elif encoded == 1:
            cumulative_score += 1
        elif encoded == 30:
            cumulative_score += 30
        else:
            raise ForagerMatchedV3ScorerError(
                f"raw reward support violation at trace index {index}"
            )
    try:
        protocol.validate_cumulative_reward_score(cumulative_score)
    except protocol.ForagerMatchedV3ProtocolError as exc:
        raise ForagerMatchedV3ScorerError("cumulative reward is outside metric bounds") from exc
    preimage = b"".join(
        (
            len(RAW_TRACE_DIGEST_DOMAIN).to_bytes(4, "big"),
            RAW_TRACE_DIGEST_DOMAIN,
            len(raw_trace).to_bytes(8, "big"),
            raw_trace,
        )
    )
    return cumulative_score, _sha256(preimage)


def canonical_raw_reward_trace_sha256(raw_trace: bytes) -> str:
    """Validate one complete int8 trace and return its version-framed digest."""
    return _trace_score_and_sha256(raw_trace)[1]


def canonical_reward_npz_bytes(raw_trace: bytes) -> bytes:
    """Encode one validated trace as the sole canonical v3 reward artifact.

    This is the inverse of :func:`ingest_reward_npz_descriptor`, not a general
    ZIP/NPY writer.  Its fixed headers intentionally reproduce the one exact
    uncompressed ``numpy.savez`` byte layout accepted by the scorer.
    """

    _trace_score_and_sha256(raw_trace)
    member = _CANONICAL_NPY_HEADER + raw_trace
    if len(member) != _NPY_MEMBER_SIZE:
        raise AssertionError("canonical reward member size drifted")
    crc32 = zlib.crc32(member) & 0xFFFFFFFF
    local = _ZIP_LOCAL_HEADER.pack(
        _ZIP_LOCAL_SIGNATURE,
        45,
        0,
        0,
        0,
        _ZIP_DOS_DATE_1980_01_01,
        crc32,
        0xFFFFFFFF,
        0xFFFFFFFF,
        len(_MEMBER_NAME_BYTES),
        len(_ZIP64_LOCAL_EXTRA),
    )
    central = _ZIP_CENTRAL_HEADER.pack(
        _ZIP_CENTRAL_SIGNATURE,
        (3 << 8) | 45,
        45,
        0,
        0,
        0,
        _ZIP_DOS_DATE_1980_01_01,
        crc32,
        _NPY_MEMBER_SIZE,
        _NPY_MEMBER_SIZE,
        len(_MEMBER_NAME_BYTES),
        0,
        0,
        0,
        0,
        _ZIP_EXTERNAL_ATTR,
        0,
    )
    end = _ZIP_END_RECORD.pack(
        _ZIP_END_SIGNATURE,
        0,
        0,
        1,
        1,
        _ZIP_CENTRAL_SIZE,
        _ZIP_CENTRAL_OFFSET,
        0,
    )
    artifact = b"".join(
        (
            local,
            _MEMBER_NAME_BYTES,
            _ZIP64_LOCAL_EXTRA,
            member,
            central,
            _MEMBER_NAME_BYTES,
            end,
        )
    )
    if len(artifact) != CANONICAL_NPZ_SIZE_BYTES:
        raise AssertionError("canonical reward artifact size drifted")
    return artifact


def _descriptor_identity(metadata: Any) -> tuple[int, int, int, int, int, int, int]:
    return (
        int(metadata.st_mode),
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_nlink),
        int(metadata.st_size),
        int(metadata.st_mtime_ns),
        int(metadata.st_ctime_ns),
    )


def _read_descriptor_bytes(descriptor: int, byte_count: int) -> bytes:
    chunks: list[bytes] = []
    offset = 0
    while offset < byte_count:
        requested = min(_READ_CHUNK_BYTES, byte_count - offset)
        try:
            chunk = os.pread(descriptor, requested, offset)
        except InterruptedError:
            continue
        except OSError as exc:
            raise ForagerMatchedV3ScorerError(
                "could not read reward artifact descriptor"
            ) from exc
        if not chunk:
            raise ForagerMatchedV3ScorerError("reward artifact changed or truncated during read")
        chunks.append(chunk)
        offset += len(chunk)
    return b"".join(chunks)


def _snapshot_regular_descriptor(descriptor: int) -> bytes:
    if type(descriptor) is not int or descriptor < 0:
        raise ForagerMatchedV3ScorerError("reward artifact descriptor must be an exact integer")
    try:
        before = os.fstat(descriptor)
    except OSError as exc:
        raise ForagerMatchedV3ScorerError("could not inspect reward artifact descriptor") from exc
    if not stat.S_ISREG(before.st_mode):
        raise ForagerMatchedV3ScorerError("reward artifact descriptor must name a regular file")
    if before.st_nlink != 1:
        raise ForagerMatchedV3ScorerError(
            "reward artifact descriptor must have exactly one filesystem link"
        )
    if before.st_size != CANONICAL_NPZ_SIZE_BYTES:
        raise ForagerMatchedV3ScorerError(
            f"reward artifact byte size must equal {CANONICAL_NPZ_SIZE_BYTES}"
        )
    identity = _descriptor_identity(before)
    raw = _read_descriptor_bytes(descriptor, CANONICAL_NPZ_SIZE_BYTES)
    try:
        after = os.fstat(descriptor)
    except OSError as exc:
        raise ForagerMatchedV3ScorerError(
            "could not re-inspect reward artifact descriptor"
        ) from exc
    if _descriptor_identity(after) != identity:
        raise ForagerMatchedV3ScorerError("reward artifact changed during ingestion")
    if len(raw) != CANONICAL_NPZ_SIZE_BYTES:
        raise ForagerMatchedV3ScorerError("reward artifact read length is invalid")
    return raw


def _unpack_exact(record: struct.Struct, raw: bytes, offset: int, path: str) -> tuple[Any, ...]:
    try:
        return record.unpack_from(raw, offset)
    except struct.error as exc:
        raise ForagerMatchedV3ScorerError(f"reward artifact {path} is malformed") from exc


def _extract_canonical_trace(raw: bytes) -> bytes:
    if len(raw) != CANONICAL_NPZ_SIZE_BYTES:
        raise ForagerMatchedV3ScorerError("reward artifact byte size is noncanonical")

    end = _unpack_exact(_ZIP_END_RECORD, raw, _ZIP_END_OFFSET, "ZIP end record")
    if end != (
        _ZIP_END_SIGNATURE,
        0,
        0,
        1,
        1,
        _ZIP_CENTRAL_SIZE,
        _ZIP_CENTRAL_OFFSET,
        0,
    ):
        raise ForagerMatchedV3ScorerError("reward artifact ZIP end record is noncanonical")

    local = _unpack_exact(_ZIP_LOCAL_HEADER, raw, 0, "ZIP local header")
    (
        local_signature,
        extract_version,
        flags,
        compression,
        dos_time,
        dos_date,
        local_crc32,
        compressed_size,
        file_size,
        name_size,
        extra_size,
    ) = local
    if (
        local_signature != _ZIP_LOCAL_SIGNATURE
        or extract_version != 45
        or flags != 0
        or compression != 0
        or dos_time != 0
        or dos_date != _ZIP_DOS_DATE_1980_01_01
        or compressed_size != 0xFFFFFFFF
        or file_size != 0xFFFFFFFF
        or name_size != len(_MEMBER_NAME_BYTES)
        or extra_size != len(_ZIP64_LOCAL_EXTRA)
        or raw[_ZIP_LOCAL_HEADER.size : _ZIP_LOCAL_HEADER.size + name_size]
        != _MEMBER_NAME_BYTES
        or raw[
            _ZIP_LOCAL_HEADER.size + name_size : _ZIP_LOCAL_HEADER.size + name_size + extra_size
        ]
        != _ZIP64_LOCAL_EXTRA
    ):
        raise ForagerMatchedV3ScorerError("reward artifact ZIP local record is noncanonical")

    central = _unpack_exact(
        _ZIP_CENTRAL_HEADER,
        raw,
        _ZIP_CENTRAL_OFFSET,
        "ZIP central record",
    )
    (
        central_signature,
        create_version,
        central_extract_version,
        central_flags,
        central_compression,
        central_time,
        central_date,
        central_crc32,
        central_compressed_size,
        central_file_size,
        central_name_size,
        central_extra_size,
        central_comment_size,
        disk_number,
        internal_attr,
        external_attr,
        header_offset,
    ) = central
    central_name_offset = _ZIP_CENTRAL_OFFSET + _ZIP_CENTRAL_HEADER.size
    if (
        central_signature != _ZIP_CENTRAL_SIGNATURE
        or create_version != ((3 << 8) | 45)
        or central_extract_version != 45
        or central_flags != 0
        or central_compression != 0
        or central_time != 0
        or central_date != _ZIP_DOS_DATE_1980_01_01
        or central_compressed_size != _NPY_MEMBER_SIZE
        or central_file_size != _NPY_MEMBER_SIZE
        or central_name_size != len(_MEMBER_NAME_BYTES)
        or central_extra_size != 0
        or central_comment_size != 0
        or disk_number != 0
        or internal_attr != 0
        or external_attr != _ZIP_EXTERNAL_ATTR
        or header_offset != 0
        or raw[central_name_offset : central_name_offset + central_name_size]
        != _MEMBER_NAME_BYTES
    ):
        raise ForagerMatchedV3ScorerError("reward artifact ZIP central record is noncanonical")

    member = raw[_ZIP_DATA_OFFSET:_ZIP_CENTRAL_OFFSET]
    crc32 = zlib.crc32(member) & 0xFFFFFFFF
    if local_crc32 != crc32 or central_crc32 != crc32:
        raise ForagerMatchedV3ScorerError("reward artifact ZIP CRC is invalid")
    if member[:_NPY_HEADER_SIZE] != _CANONICAL_NPY_HEADER:
        raise ForagerMatchedV3ScorerError(
            "reward artifact has a noncanonical NPY header, dtype, or shape"
        )
    trace = member[_NPY_HEADER_SIZE:]
    if len(trace) != protocol.MATCHED_V3_HORIZON:
        raise ForagerMatchedV3ScorerError("reward artifact trace horizon is invalid")
    return trace


def ingest_reward_npz_bytes(artifact: bytes) -> MatchedV3ScoreReceipt:
    """Ingest one complete canonical NPZ byte string without filesystem authority."""

    trace = extract_canonical_reward_trace(artifact)
    score, trace_sha256 = _trace_score_and_sha256(trace)
    return MatchedV3ScoreReceipt(
        cumulative_score=score,
        raw_trace_sha256=trace_sha256,
        artifact_sha256=_sha256(artifact),
        artifact_size_bytes=len(artifact),
    )


def extract_canonical_reward_trace(artifact: bytes) -> bytes:
    """Return the exact trace from the sole accepted NPZ layout after validation."""

    if type(artifact) is not bytes:
        raise ForagerMatchedV3ScorerError("reward artifact must be exact bytes")
    _bound_metric_descriptor()
    trace = _extract_canonical_trace(artifact)
    _trace_score_and_sha256(trace)
    return trace


def ingest_reward_npz_descriptor(descriptor: int) -> MatchedV3ScoreReceipt:
    """Ingest one canonical NPZ from an already-open stable regular descriptor."""

    return ingest_reward_npz_bytes(_snapshot_regular_descriptor(descriptor))


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ForagerMatchedV3ScorerError(
                f"duplicate JSON object key {key!r} is forbidden"
            )
        result[key] = value
    return result


def _reject_nonfinite(token: str) -> Any:
    raise ForagerMatchedV3ScorerError(f"non-finite JSON number {token!r} is forbidden")


def _validate_json_tree(value: Any) -> None:
    pending = [value]
    nodes = 0
    while pending:
        item = pending.pop()
        nodes += 1
        if nodes > 10_000:
            raise ForagerMatchedV3ScorerError("score receipt exceeds the JSON node limit")
        if isinstance(item, dict):
            if any(type(key) is not str for key in item):
                raise ForagerMatchedV3ScorerError("score receipt keys must be strings")
            pending.extend(item.values())
        elif isinstance(item, list):
            pending.extend(item)
        elif isinstance(item, float):
            if not math.isfinite(item):
                raise ForagerMatchedV3ScorerError(
                    "score receipt contains a non-finite number"
                )
        elif item is not None and not isinstance(item, (str, bool, int)):
            raise ForagerMatchedV3ScorerError(
                f"score receipt contains non-JSON value type {type(item).__name__}"
            )


def _decode_receipt(value: Mapping[str, Any] | bytes | str) -> tuple[dict[str, Any], bytes | None]:
    supplied_bytes: bytes | None = None
    try:
        if isinstance(value, bytes):
            if len(value) > _MAX_RECEIPT_BYTES:
                raise ForagerMatchedV3ScorerError("score receipt exceeds the JSON byte limit")
            supplied_bytes = value
            text = value.decode("ascii")
        elif isinstance(value, str):
            supplied_bytes = value.encode("ascii")
            if len(supplied_bytes) > _MAX_RECEIPT_BYTES:
                raise ForagerMatchedV3ScorerError("score receipt exceeds the JSON byte limit")
            text = value
        elif isinstance(value, Mapping):
            canonical = _canonical_json_bytes(value)
            text = canonical.decode("ascii")
        else:
            raise ForagerMatchedV3ScorerError(
                "score receipt must be canonical JSON bytes, text, or an object"
            )
        decoded = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except ForagerMatchedV3ScorerError:
        raise
    except (OverflowError, RecursionError, UnicodeError, ValueError) as exc:
        raise ForagerMatchedV3ScorerError("score receipt is not strict ASCII JSON") from exc
    _validate_json_tree(decoded)
    return _require_object(decoded, "score receipt"), supplied_bytes


def parse_score_receipt(
    value: Mapping[str, Any] | bytes | str,
) -> MatchedV3ScoreReceipt:
    """Validate and detach a content-addressed score receipt."""
    payload, supplied_bytes = _decode_receipt(value)
    _require_exact_keys(
        payload,
        "score receipt",
        {
            "schema_version",
            "metric",
            "score",
            "raw_trace",
            "artifact",
            "authority",
            "receipt_sha256",
        },
    )
    if payload["schema_version"] != SCORE_RECEIPT_SCHEMA_VERSION:
        raise ForagerMatchedV3ScorerError("score receipt schema_version is invalid")
    supplied_digest = _require_sha256(payload["receipt_sha256"], "receipt_sha256")
    body = {key: item for key, item in payload.items() if key != "receipt_sha256"}
    if _sha256(_canonical_json_bytes(body)) != supplied_digest:
        raise ForagerMatchedV3ScorerError("receipt_sha256 does not bind the receipt body")

    metric = _require_object(payload["metric"], "score receipt metric")
    _require_exact_keys(metric, "score receipt metric", {"descriptor", "sha256"})
    if (
        metric["descriptor"] != _bound_metric_descriptor()
        or metric["sha256"] != protocol.CUMULATIVE_REWARD_METRIC_SHA256
    ):
        raise ForagerMatchedV3ScorerError("score receipt metric binding is invalid")

    score = _require_object(payload["score"], "score receipt score")
    _require_exact_keys(
        score,
        "score receipt score",
        {"accumulation", "cumulative_reward"},
    )
    if score["accumulation"] != "ordered_exact_integer_sum":
        raise ForagerMatchedV3ScorerError("score receipt accumulation is invalid")
    cumulative_score = _require_exact_integer(
        score["cumulative_reward"],
        "score receipt cumulative_reward",
    )

    raw_trace = _require_object(payload["raw_trace"], "score receipt raw_trace")
    _require_exact_keys(
        raw_trace,
        "score receipt raw_trace",
        {
            "encoding_schema_version",
            "encoding",
            "digest_domain",
            "digest_framing",
            "horizon",
            "raw_reward_values",
            "sha256",
        },
    )
    expected_trace_constants = {
        "encoding_schema_version": RAW_TRACE_ENCODING_SCHEMA_VERSION,
        "encoding": RAW_TRACE_ENCODING,
        "digest_domain": RAW_TRACE_DIGEST_DOMAIN.decode("ascii"),
        "digest_framing": (
            "uint32be_domain_length_then_ascii_domain_then_"
            "uint64be_trace_length_then_trace_bytes"
        ),
        "horizon": protocol.MATCHED_V3_HORIZON,
        "raw_reward_values": list(protocol.MATCHED_V3_RAW_REWARD_VALUES),
    }
    if any(raw_trace[key] != expected for key, expected in expected_trace_constants.items()):
        raise ForagerMatchedV3ScorerError("score receipt raw_trace constants are invalid")
    trace_sha256 = _require_sha256(raw_trace["sha256"], "score receipt raw_trace sha256")

    artifact = _require_object(payload["artifact"], "score receipt artifact")
    _require_exact_keys(
        artifact,
        "score receipt artifact",
        {
            "container_schema_version",
            "member_name",
            "compression",
            "npy_format_version",
            "dtype",
            "fortran_order",
            "shape",
            "sha256",
            "size_bytes",
        },
    )
    expected_artifact_constants = {
        "container_schema_version": NPZ_CONTAINER_SCHEMA_VERSION,
        "member_name": NPZ_MEMBER_NAME,
        "compression": "stored",
        "npy_format_version": "1.0",
        "dtype": "|i1",
        "fortran_order": False,
        "shape": [protocol.MATCHED_V3_HORIZON],
    }
    if any(artifact[key] != expected for key, expected in expected_artifact_constants.items()):
        raise ForagerMatchedV3ScorerError("score receipt artifact constants are invalid")
    artifact_sha256 = _require_sha256(
        artifact["sha256"],
        "score receipt artifact sha256",
    )
    artifact_size = _require_exact_integer(
        artifact["size_bytes"],
        "score receipt artifact size_bytes",
    )

    authority = _require_object(payload["authority"], "score receipt authority")
    if authority != _authority_denial():
        raise ForagerMatchedV3ScorerError("score receipt authority must remain entirely denied")

    receipt = MatchedV3ScoreReceipt(
        cumulative_score=cumulative_score,
        raw_trace_sha256=trace_sha256,
        artifact_sha256=artifact_sha256,
        artifact_size_bytes=artifact_size,
    )
    if _canonical_json_bytes(payload) != receipt.canonical_json():
        raise ForagerMatchedV3ScorerError("score receipt fields do not replay exactly")
    if supplied_bytes is not None and supplied_bytes != receipt.canonical_json():
        raise ForagerMatchedV3ScorerError("score receipt JSON bytes are not canonical")
    return receipt


def replay_reward_npz_descriptor(
    descriptor: int,
    receipt: Mapping[str, Any] | bytes | str,
) -> MatchedV3ScoreReceipt:
    """Re-ingest an artifact and require exact equality with a detached receipt."""
    expected = parse_score_receipt(receipt)
    actual = ingest_reward_npz_descriptor(descriptor)
    if actual.canonical_json() != expected.canonical_json():
        raise ForagerMatchedV3ScorerError("score receipt does not replay from reward artifact")
    return actual


__all__ = [
    "CANONICAL_NPZ_SIZE_BYTES",
    "MatchedV3ScoreReceipt",
    "NPZ_CONTAINER_SCHEMA_VERSION",
    "NPZ_MEMBER_NAME",
    "RAW_TRACE_DIGEST_DOMAIN",
    "RAW_TRACE_ENCODING",
    "RAW_TRACE_ENCODING_SCHEMA_VERSION",
    "SCORE_RECEIPT_SCHEMA_VERSION",
    "ForagerMatchedV3ScorerError",
    "canonical_reward_npz_bytes",
    "canonical_raw_reward_trace_sha256",
    "extract_canonical_reward_trace",
    "ingest_reward_npz_bytes",
    "ingest_reward_npz_descriptor",
    "parse_score_receipt",
    "replay_reward_npz_descriptor",
]
