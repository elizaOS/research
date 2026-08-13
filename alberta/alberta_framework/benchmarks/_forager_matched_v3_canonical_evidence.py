"""Private canonical codecs and frozen nested records for matched-v3 evidence.

This module is deliberately limited to pure in-memory validation.  It does not
read or write files, inspect a host, launch a process, contact a container
runtime, issue a case, or dispatch public artifact schemas.  Artifact bytes and
their two independent identities are always supplied by the caller.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final, Literal, Never, cast, final

from alberta_framework.benchmarks.forager_matched_v3_qualification_plan_v3 import (
    MATCHED_V3_ADAPTER_CANDIDATE_IDS as _AUTHORITATIVE_ADAPTER_CANDIDATE_IDS,
)
from alberta_framework.benchmarks.forager_matched_v3_qualification_plan_v3 import (
    MATCHED_V3_EXTERNAL_CANDIDATE_IDS as _AUTHORITATIVE_EXTERNAL_CANDIDATE_IDS,
)
from alberta_framework.benchmarks.forager_matched_v3_qualification_plan_v3 import (
    MATCHED_V3_LOCAL_CANDIDATE_IDS as _AUTHORITATIVE_LOCAL_CANDIDATE_IDS,
)
from alberta_framework.benchmarks.forager_matched_v3_qualification_plan_v3 import (
    MATCHED_V3_QUALIFICATION_V3_CANDIDATE_IDS as _AUTHORITATIVE_CANDIDATE_IDS,
)

MAX_CANONICAL_FILE_BYTES: Final = 4 * 1024 * 1024
MAX_RAW_CAPTURE_BYTES: Final = 2 * 1024 * 1024
MAX_JSON_DEPTH: Final = 64
MAX_JSON_NODES: Final = 20_000
MAX_JSON_STRING_LENGTH: Final = 3 * 1024 * 1024
MAX_EXACT_INTEGER: Final = (1 << 63) - 1

RAW_BYTE_CAPTURE_V1_SCHEMA_VERSION: Final = "alberta.forager_matched_v3.raw_byte_capture.v1"

# These aliases deliberately have one authority: qualification-plan v3.  The
# codec adds no competing candidate registry and the plan does not import this
# lower-level helper, so the dependency remains acyclic.
MATCHED_V3_LOCAL_CANDIDATE_IDS: Final = _AUTHORITATIVE_LOCAL_CANDIDATE_IDS
MATCHED_V3_EXTERNAL_CANDIDATE_IDS: Final = _AUTHORITATIVE_EXTERNAL_CANDIDATE_IDS
MATCHED_V3_ADAPTER_CANDIDATE_IDS: Final = _AUTHORITATIVE_ADAPTER_CANDIDATE_IDS
MATCHED_V3_CANDIDATE_IDS: Final = _AUTHORITATIVE_CANDIDATE_IDS

ProducerRole = Literal[
    "measurement_producer",
    "terminal_relay",
    "nonstorage_channel",
    "write_seal_producer",
    "runtime_storage_escape_gate",
    "namespace_cleanup_producer",
]
PRODUCER_ROLES: Final[tuple[ProducerRole, ...]] = (
    "measurement_producer",
    "terminal_relay",
    "nonstorage_channel",
    "write_seal_producer",
    "runtime_storage_escape_gate",
    "namespace_cleanup_producer",
)
PRODUCER_DESCRIPTOR_SCHEMA_BY_ROLE: Final[Mapping[ProducerRole, str]] = MappingProxyType(
    {
        "measurement_producer": (
            "alberta.forager_matched_v3.qualification_storage_measurement_producer_descriptor.v2"
        ),
        "terminal_relay": (
            "alberta.forager_matched_v3.qualification_storage_terminal_relay_descriptor.v1"
        ),
        "nonstorage_channel": (
            "alberta.forager_matched_v3.qualification_storage_nonstorage_channel_descriptor.v1"
        ),
        "write_seal_producer": (
            "alberta.forager_matched_v3.qualification_storage_write_seal_producer_descriptor.v2"
        ),
        "runtime_storage_escape_gate": (
            "alberta.forager_matched_v3.runtime_storage_escape_gate_descriptor.v1"
        ),
        "namespace_cleanup_producer": (
            "alberta.forager_matched_v3.namespace_cleanup_producer_descriptor.v1"
        ),
    }
)

_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER_RE: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}\Z")
_ZERO_SHA256: Final = "0" * 64


class CanonicalEvidenceError(ValueError):
    """Canonical evidence bytes or a frozen nested record failed closed."""


def _fail(message: str) -> Never:
    raise CanonicalEvidenceError(message)


def _require_exact_int(
    value: object,
    label: str,
    *,
    minimum: int = 0,
    maximum: int = MAX_EXACT_INTEGER,
) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        _fail(f"{label} must be one bounded exact integer")
    return value


def _require_identifier(value: object, label: str) -> str:
    if type(value) is not str or _IDENTIFIER_RE.fullmatch(value) is None:
        _fail(f"{label} must be one bounded portable identifier")
    return value


def require_nonzero_sha256(value: object, label: str) -> str:
    """Return one exact lowercase nonzero SHA-256 digest or fail closed."""

    if type(value) is not str or _SHA256_RE.fullmatch(value) is None or value == _ZERO_SHA256:
        _fail(f"{label} must be one lowercase nonzero SHA-256")
    return value


def require_distinct_sha256s(values: Sequence[object], label: str) -> tuple[str, ...]:
    """Validate an exact sequence of nonzero digests and reject cross-kind aliases."""

    if type(values) not in {tuple, list} or not values:
        _fail(f"{label} must be one nonempty exact digest sequence")
    digests = tuple(
        require_nonzero_sha256(value, f"{label} item {index}") for index, value in enumerate(values)
    )
    if len(set(digests)) != len(digests):
        _fail(f"{label} contains aliased SHA-256 identities")
    return digests


def _reject_json_constant(value: str) -> Never:
    _fail(f"JSON contains forbidden constant {value!r}")


def _reject_json_float(value: str) -> Never:
    _fail(f"JSON contains forbidden float {value!r}")


def _parse_json_int(value: str) -> int:
    digits = value[1:] if value.startswith("-") else value
    if not digits or len(digits) > 19:
        _fail("JSON integer exceeds its lexical bound")
    parsed = int(value, 10)
    if not -MAX_EXACT_INTEGER <= parsed <= MAX_EXACT_INTEGER:
        _fail("JSON integer exceeds its value bound")
    return parsed


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if type(key) is not str or key in result:
            _fail("JSON contains a duplicate or non-text key")
        result[key] = value
    return result


def _canonical_json_size_preflight(value: object, *, final_lf: bool) -> int:
    """Validate a plain JSON tree and return its exact encoded byte size."""

    seen_containers: set[int] = set()
    node_count = 0

    def checked_size(size: int) -> int:
        if size > MAX_CANONICAL_FILE_BYTES:
            _fail("canonical JSON violates its aggregate encoded byte bound")
        return size

    def visit(item: object, depth: int) -> int:
        nonlocal node_count
        node_count += 1
        if depth > MAX_JSON_DEPTH:
            _fail("JSON structure exceeds its depth bound")
        if node_count > MAX_JSON_NODES:
            _fail("JSON structure exceeds its node bound")
        if item is None:
            return 4
        if type(item) is bool:
            return 4 if item else 5
        if type(item) is int:
            _require_exact_int(
                item,
                "JSON integer",
                minimum=-MAX_EXACT_INTEGER,
                maximum=MAX_EXACT_INTEGER,
            )
            return len(str(item))
        if type(item) is str:
            if len(item) > MAX_JSON_STRING_LENGTH or any(
                ord(character) < 0x20 or ord(character) > 0x7E for character in item
            ):
                _fail("JSON strings must be bounded printable ASCII")
            return checked_size(2 + len(item) + item.count('"') + item.count("\\"))
        if type(item) not in {dict, list}:
            _fail("JSON contains a float, subclass, or other non-plain value")
        identity = id(item)
        if identity in seen_containers:
            _fail("JSON contains an alias or cycle")
        seen_containers.add(identity)
        if type(item) is list:
            encoded_size = 2
            for index, child in enumerate(cast(list[object], item)):
                if index:
                    encoded_size = checked_size(encoded_size + 1)
                encoded_size = checked_size(encoded_size + visit(child, depth + 1))
            return encoded_size
        encoded_size = 2
        for index, (key, child) in enumerate(cast(dict[object, object], item).items()):
            if type(key) is not str:
                _fail("JSON keys must be exact strings")
            if index:
                encoded_size = checked_size(encoded_size + 1)
            encoded_size = checked_size(encoded_size + visit(key, depth + 1))
            encoded_size = checked_size(encoded_size + 1)
            encoded_size = checked_size(encoded_size + visit(child, depth + 1))
        return encoded_size

    encoded_size = visit(value, 0)
    if final_lf:
        encoded_size = checked_size(encoded_size + 1)
    return encoded_size


def _preflight_canonical_json(value: object, *, final_lf: bool) -> int:
    try:
        return _canonical_json_size_preflight(value, final_lf=final_lf)
    except CanonicalEvidenceError:
        raise
    except (MemoryError, RecursionError) as exc:
        raise CanonicalEvidenceError("canonical JSON preflight exhausted safe bounds") from exc


def _assert_plain_unaliased_json(value: object) -> None:
    _preflight_canonical_json(value, final_lf=False)


def canonical_json_bytes(value: object, *, final_lf: bool = True) -> bytes:
    """Encode bounded, duplicate-free, printable-ASCII canonical JSON.

    ``final_lf=True`` creates full-file framing with exactly one trailing LF.
    ``final_lf=False`` is reserved for BODY identity calculation.
    """

    if type(final_lf) is not bool:
        _fail("canonical JSON final-LF selector must be one exact boolean")
    expected_size = _preflight_canonical_json(value, final_lf=final_lf)
    try:
        raw = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        if final_lf:
            raw += b"\n"
    except (MemoryError, RecursionError, TypeError, ValueError) as exc:
        raise CanonicalEvidenceError("value is not canonical evidence JSON") from exc
    if len(raw) != expected_size:
        _fail("canonical JSON serializer differs from its exact size preflight")
    return raw


class _CanonicalJsonByteScanner:
    """Bounded canonical-JSON grammar preflight that materializes no value tree."""

    __slots__ = ("_index", "_limit", "_node_count", "_raw")

    def __init__(self, raw: bytes) -> None:
        self._raw = raw
        self._index = 0
        self._limit = len(raw) - 1
        self._node_count = 0

    def scan_root_object(self) -> None:
        if self._limit < 2 or self._raw[-1] != 0x0A:
            _fail("canonical file must have exactly one final LF")
        if self._raw[0] != 0x7B:
            _fail("canonical file root must be one exact object")
        self._parse_value(0)
        if self._index != self._limit:
            _fail("canonical file has trailing or noncanonical bytes before its final LF")

    def _bump_node(self, depth: int) -> None:
        self._node_count += 1
        if depth > MAX_JSON_DEPTH:
            _fail("JSON structure exceeds its depth bound before decoding")
        if self._node_count > MAX_JSON_NODES:
            _fail("JSON structure exceeds its node bound before decoding")

    def _current(self) -> int:
        if self._index >= self._limit:
            _fail("canonical JSON value is truncated")
        return self._raw[self._index]

    def _parse_value(self, depth: int) -> None:
        self._bump_node(depth)
        token = self._current()
        if token == 0x7B:
            self._parse_object(depth)
            return
        if token == 0x5B:
            self._parse_array(depth)
            return
        if token == 0x22:
            self._parse_string()
            return
        if token == 0x74:
            self._parse_literal(b"true")
            return
        if token == 0x66:
            self._parse_literal(b"false")
            return
        if token == 0x6E:
            self._parse_literal(b"null")
            return
        if token == 0x2D or 0x30 <= token <= 0x39:
            self._parse_integer()
            return
        _fail("canonical JSON contains whitespace, a float, a constant, or an invalid token")

    def _parse_object(self, depth: int) -> None:
        self._index += 1
        if self._current() == 0x7D:
            self._index += 1
            return
        while True:
            if self._current() != 0x22:
                _fail("canonical JSON object key must be one exact string")
            self._bump_node(depth + 1)
            self._parse_string()
            if self._current() != 0x3A:
                _fail("canonical JSON object key must be followed by one colon")
            self._index += 1
            self._parse_value(depth + 1)
            delimiter = self._current()
            if delimiter == 0x7D:
                self._index += 1
                return
            if delimiter != 0x2C:
                _fail("canonical JSON object member delimiter differs")
            self._index += 1

    def _parse_array(self, depth: int) -> None:
        self._index += 1
        if self._current() == 0x5D:
            self._index += 1
            return
        while True:
            self._parse_value(depth + 1)
            delimiter = self._current()
            if delimiter == 0x5D:
                self._index += 1
                return
            if delimiter != 0x2C:
                _fail("canonical JSON array element delimiter differs")
            self._index += 1

    def _parse_string(self) -> None:
        if self._current() != 0x22:
            _fail("canonical JSON string opening quote differs")
        self._index += 1
        decoded_length = 0
        while True:
            token = self._current()
            if token == 0x22:
                self._index += 1
                return
            if token == 0x5C:
                self._index += 1
                escaped = self._current()
                if escaped not in {0x22, 0x5C}:
                    _fail("canonical printable-ASCII JSON permits only quote/backslash escapes")
                self._index += 1
            else:
                if not 0x20 <= token <= 0x7E:
                    _fail("canonical JSON strings must decode to printable ASCII")
                self._index += 1
            decoded_length += 1
            if decoded_length > MAX_JSON_STRING_LENGTH:
                _fail("canonical JSON string exceeds its decoded length bound")

    def _parse_literal(self, literal: bytes) -> None:
        end = self._index + len(literal)
        if end > self._limit or not self._raw.startswith(literal, self._index):
            _fail("canonical JSON literal differs")
        self._index = end

    def _parse_integer(self) -> None:
        negative = self._current() == 0x2D
        if negative:
            self._index += 1
        first = self._current()
        if first == 0x30:
            if negative:
                _fail("canonical JSON forbids negative zero")
            self._index += 1
            if self._index < self._limit and 0x30 <= self._raw[self._index] <= 0x39:
                _fail("canonical JSON integer has a leading zero")
            return
        if not 0x31 <= first <= 0x39:
            _fail("canonical JSON integer has no canonical digit sequence")
        magnitude = 0
        digits = 0
        while self._index < self._limit:
            token = self._raw[self._index]
            if not 0x30 <= token <= 0x39:
                break
            digits += 1
            if digits > 19:
                _fail("canonical JSON integer exceeds its lexical bound")
            magnitude = magnitude * 10 + token - 0x30
            if magnitude > MAX_EXACT_INTEGER:
                _fail("canonical JSON integer exceeds its value bound")
            self._index += 1


def _scan_canonical_json_file(raw: bytes) -> None:
    try:
        _CanonicalJsonByteScanner(raw).scan_root_object()
    except CanonicalEvidenceError:
        raise
    except (MemoryError, RecursionError) as exc:
        raise CanonicalEvidenceError("canonical JSON byte preflight exhausted safe bounds") from exc


def decode_canonical_json_file(raw: bytes) -> dict[str, Any]:
    """Decode one canonical object framed by exactly one final LF."""

    if type(raw) is not bytes or not 1 <= len(raw) <= MAX_CANONICAL_FILE_BYTES:
        _fail("canonical file bytes violate their bound")
    _scan_canonical_json_file(raw)
    try:
        value = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_object_without_duplicates,
            parse_constant=_reject_json_constant,
            parse_float=_reject_json_float,
            parse_int=_parse_json_int,
        )
    except CanonicalEvidenceError:
        raise
    except (
        MemoryError,
        RecursionError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        ValueError,
    ) as exc:
        raise CanonicalEvidenceError("artifact is not strict ASCII JSON") from exc
    if type(value) is not dict:
        _fail("canonical file root must be one exact object")
    result = cast(dict[str, Any], value)
    _assert_plain_unaliased_json(result)
    if not hmac.compare_digest(canonical_json_bytes(result), raw):
        _fail("artifact is noncanonical or lacks exactly one final LF")
    return result


def canonical_body_sha256(body: object) -> str:
    """Return the SHA-256 of one canonical object BODY without file framing."""

    if type(body) is not dict:
        _fail("canonical BODY must be one exact object")
    exact = cast(dict[str, Any], body)
    return hashlib.sha256(canonical_json_bytes(exact, final_lf=False)).hexdigest()


def canonical_file_bytes(body: object, *, body_digest_field: str) -> bytes:
    """Add one BODY identity and encode the resulting canonical full file."""

    if type(body) is not dict:
        _fail("canonical file BODY must be one exact object")
    field = _require_identifier(body_digest_field, "BODY digest field")
    exact = cast(dict[str, Any], body)
    _assert_plain_unaliased_json(exact)
    if field in exact:
        _fail("canonical file BODY already contains its digest field")
    document = dict(exact)
    document[field] = canonical_body_sha256(exact)
    return canonical_json_bytes(document)


def validate_file_sha256(raw: bytes, expected_file_sha256: object) -> str:
    """Validate only the independently supplied full-file identity."""

    if type(raw) is not bytes or not 1 <= len(raw) <= MAX_CANONICAL_FILE_BYTES:
        _fail("canonical file bytes violate their bound")
    expected = require_nonzero_sha256(expected_file_sha256, "expected FILE identity")
    observed = hashlib.sha256(raw).hexdigest()
    if not hmac.compare_digest(observed, expected):
        _fail("canonical file differs from its expected FILE identity")
    return observed


def validate_body_sha256(
    document: object,
    *,
    body_digest_field: str,
    expected_body_sha256: object,
) -> dict[str, Any]:
    """Validate only the embedded and independently supplied BODY identities."""

    if type(document) is not dict:
        _fail("canonical document must be one exact object")
    exact = cast(dict[str, Any], document)
    _assert_plain_unaliased_json(exact)
    field = _require_identifier(body_digest_field, "BODY digest field")
    if frozenset(exact).isdisjoint({field}):
        _fail("canonical document lacks its BODY digest field")
    supplied = require_nonzero_sha256(exact[field], "embedded BODY identity")
    expected = require_nonzero_sha256(expected_body_sha256, "expected BODY identity")
    body = dict(exact)
    del body[field]
    observed = canonical_body_sha256(body)
    if not hmac.compare_digest(supplied, observed):
        _fail("embedded BODY identity differs from canonical BODY bytes")
    if not hmac.compare_digest(supplied, expected):
        _fail("canonical BODY differs from its expected BODY identity")
    return body


def validate_canonical_file(
    raw: bytes,
    *,
    expected_file_sha256: object,
    expected_body_sha256: object,
    body_digest_field: str,
) -> dict[str, Any]:
    """Validate independent, nonaliasing FILE and BODY identities and decode the BODY."""

    file_identity, body_identity = require_distinct_sha256s(
        (expected_file_sha256, expected_body_sha256),
        "expected FILE and BODY identities",
    )
    validate_file_sha256(raw, file_identity)
    document = decode_canonical_json_file(raw)
    return validate_body_sha256(
        document,
        body_digest_field=body_digest_field,
        expected_body_sha256=body_identity,
    )


@final
@dataclass(frozen=True, slots=True)
class ArtifactRefV1:
    """One canonical artifact's independently carried FILE and BODY identities."""

    schema_version: str
    file_sha256: str
    body_sha256: str

    def __init_subclass__(cls, **kwargs: Any) -> None:
        raise TypeError("ArtifactRefV1 is runtime-final")

    def __post_init__(self) -> None:
        _require_identifier(self.schema_version, "artifact schema")
        require_distinct_sha256s(
            (self.file_sha256, self.body_sha256),
            "artifact FILE and BODY identities",
        )

    def to_dict(self) -> dict[str, str]:
        ArtifactRefV1.__post_init__(self)
        return {
            "body_sha256": self.body_sha256,
            "file_sha256": self.file_sha256,
            "schema_version": self.schema_version,
        }


@final
@dataclass(frozen=True, slots=True)
class ProducerRefV1:
    """One of six exact producer roles with three nonaliasing identities."""

    role: ProducerRole
    descriptor_schema_version: str
    descriptor_file_sha256: str
    descriptor_body_sha256: str
    source_sha256: str

    def __init_subclass__(cls, **kwargs: Any) -> None:
        raise TypeError("ProducerRefV1 is runtime-final")

    def __post_init__(self) -> None:
        role = _require_identifier(self.role, "producer role")
        if role not in PRODUCER_ROLES:
            _fail("producer role differs from the frozen six-role universe")
        exact_role = role
        descriptor_schema = _require_identifier(
            self.descriptor_schema_version,
            "producer descriptor schema",
        )
        if descriptor_schema != PRODUCER_DESCRIPTOR_SCHEMA_BY_ROLE[exact_role]:
            _fail("producer descriptor schema differs for its role")
        require_distinct_sha256s(
            (
                self.descriptor_file_sha256,
                self.descriptor_body_sha256,
                self.source_sha256,
            ),
            "producer descriptor FILE, descriptor BODY, and source identities",
        )

    def to_dict(self) -> dict[str, str]:
        ProducerRefV1.__post_init__(self)
        return {
            "descriptor_body_sha256": self.descriptor_body_sha256,
            "descriptor_file_sha256": self.descriptor_file_sha256,
            "descriptor_schema_version": self.descriptor_schema_version,
            "role": self.role,
            "source_sha256": self.source_sha256,
        }


def validate_producer_refs_v1(producers: object) -> tuple[ProducerRefV1, ...]:
    """Validate the exact ordered six-role tuple and global identity separation."""

    if type(producers) is not tuple or len(producers) != len(PRODUCER_ROLES):
        _fail("producers must be one exact ordered six-role tuple")
    exact = cast(tuple[object, ...], producers)
    if any(type(producer) is not ProducerRefV1 for producer in exact):
        _fail("producer tuple contains a nonexact producer reference")
    typed = cast(tuple[ProducerRefV1, ...], exact)
    for producer in typed:
        ProducerRefV1.__post_init__(producer)
    if tuple(producer.role for producer in typed) != PRODUCER_ROLES:
        _fail("producer role order differs from the frozen six-role order")
    require_distinct_sha256s(
        tuple(producer.descriptor_file_sha256 for producer in typed)
        + tuple(producer.descriptor_body_sha256 for producer in typed)
        + tuple(producer.source_sha256 for producer in typed),
        "all producer identities",
    )
    return typed


CandidateFamily = Literal["local", "external", "adapter"]


def candidate_family(candidate_id: object) -> CandidateFamily:
    """Return the authoritative matched-v3 family for one exact candidate ID."""

    candidate = _require_identifier(candidate_id, "candidate ID")
    if candidate in MATCHED_V3_LOCAL_CANDIDATE_IDS:
        return "local"
    if candidate in MATCHED_V3_EXTERNAL_CANDIDATE_IDS:
        return "external"
    if candidate in MATCHED_V3_ADAPTER_CANDIDATE_IDS:
        return "adapter"
    _fail("candidate ID is outside the frozen matched-v3 universe")


@final
@dataclass(frozen=True, slots=True)
class CaseSubjectV1:
    """The exact ordinal, candidate, family, and ID projection for one v3 case."""

    case_ordinal: int
    candidate_id: str
    candidate_family: CandidateFamily
    qualification_case_id: str

    def __init_subclass__(cls, **kwargs: Any) -> None:
        raise TypeError("CaseSubjectV1 is runtime-final")

    def __post_init__(self) -> None:
        ordinal = _require_exact_int(
            self.case_ordinal,
            "case ordinal",
            maximum=len(MATCHED_V3_CANDIDATE_IDS) - 1,
        )
        expected_candidate = MATCHED_V3_CANDIDATE_IDS[ordinal]
        candidate = _require_identifier(self.candidate_id, "candidate ID")
        if candidate != expected_candidate:
            _fail("case ordinal and candidate differ from the frozen order")
        family = _require_identifier(self.candidate_family, "candidate family")
        if family != candidate_family(candidate):
            _fail("candidate family differs from the frozen family projection")
        case_id = _require_identifier(self.qualification_case_id, "qualification case ID")
        if case_id != f"qualification_{ordinal:02d}_{candidate}":
            _fail("qualification case ID differs from the frozen case projection")

    @staticmethod
    def for_ordinal(case_ordinal: int) -> CaseSubjectV1:
        """Construct the unique frozen case subject for ``case_ordinal``."""

        ordinal = _require_exact_int(
            case_ordinal,
            "case ordinal",
            maximum=len(MATCHED_V3_CANDIDATE_IDS) - 1,
        )
        candidate = MATCHED_V3_CANDIDATE_IDS[ordinal]
        return CaseSubjectV1(
            case_ordinal=ordinal,
            candidate_id=candidate,
            candidate_family=candidate_family(candidate),
            qualification_case_id=f"qualification_{ordinal:02d}_{candidate}",
        )

    def to_dict(self) -> dict[str, int | str]:
        CaseSubjectV1.__post_init__(self)
        return {
            "candidate_family": self.candidate_family,
            "candidate_id": self.candidate_id,
            "case_ordinal": self.case_ordinal,
            "qualification_case_id": self.qualification_case_id,
        }


@final
@dataclass(frozen=True, slots=True)
class RawByteCaptureV1:
    """One bounded raw byte string carried as verified canonical base64 metadata."""

    raw_bytes_base64: str
    raw_size_bytes: int
    raw_sha256: str
    schema_version: str = RAW_BYTE_CAPTURE_V1_SCHEMA_VERSION

    def __init_subclass__(cls, **kwargs: Any) -> None:
        raise TypeError("RawByteCaptureV1 is runtime-final")

    def __post_init__(self) -> None:
        if type(self.schema_version) is not str or (
            self.schema_version != RAW_BYTE_CAPTURE_V1_SCHEMA_VERSION
        ):
            _fail("raw-byte capture schema differs")
        if type(self.raw_bytes_base64) is not str:
            _fail("raw-byte capture base64 must be one exact string")
        if len(self.raw_bytes_base64) > ((MAX_RAW_CAPTURE_BYTES + 2) // 3) * 4:
            _fail("raw-byte capture base64 exceeds its lexical bound")
        try:
            decoded = base64.b64decode(self.raw_bytes_base64.encode("ascii"), validate=True)
        except (UnicodeEncodeError, binascii.Error, ValueError) as exc:
            raise CanonicalEvidenceError("raw-byte capture is not strict base64") from exc
        if not 0 <= len(decoded) <= MAX_RAW_CAPTURE_BYTES:
            _fail("raw-byte capture violates its decoded byte bound")
        canonical_base64 = base64.b64encode(decoded).decode("ascii")
        if not hmac.compare_digest(canonical_base64, self.raw_bytes_base64):
            _fail("raw-byte capture base64 is not canonical RFC 4648 encoding")
        expected_size = _require_exact_int(
            self.raw_size_bytes,
            "raw-byte capture size",
            minimum=0,
            maximum=MAX_RAW_CAPTURE_BYTES,
        )
        if len(decoded) != expected_size:
            _fail("raw-byte capture decoded size differs")
        expected_sha256 = require_nonzero_sha256(
            self.raw_sha256,
            "raw-byte capture identity",
        )
        if not hmac.compare_digest(hashlib.sha256(decoded).hexdigest(), expected_sha256):
            _fail("raw-byte capture decoded identity differs")

    @staticmethod
    def from_bytes(raw: bytes) -> RawByteCaptureV1:
        """Create verified capture metadata from exact caller-owned bytes."""

        if type(raw) is not bytes or not 0 <= len(raw) <= MAX_RAW_CAPTURE_BYTES:
            _fail("raw-byte capture input violates its byte bound")
        return RawByteCaptureV1(
            raw_bytes_base64=base64.b64encode(raw).decode("ascii"),
            raw_size_bytes=len(raw),
            raw_sha256=hashlib.sha256(raw).hexdigest(),
        )

    def decoded_bytes(self) -> bytes:
        """Return the bytes reverified by construction."""

        RawByteCaptureV1.__post_init__(self)
        return base64.b64decode(self.raw_bytes_base64.encode("ascii"), validate=True)

    def to_dict(self) -> dict[str, int | str]:
        RawByteCaptureV1.__post_init__(self)
        return {
            "raw_bytes_base64": self.raw_bytes_base64,
            "raw_sha256": self.raw_sha256,
            "raw_size_bytes": self.raw_size_bytes,
            "schema_version": self.schema_version,
        }


def _exact_dict(value: object, keys: frozenset[str], label: str) -> dict[str, Any]:
    if type(value) is not dict or frozenset(value) != keys:
        _fail(f"{label} keys differ")
    exact = cast(dict[str, Any], value)
    _assert_plain_unaliased_json(exact)
    return exact


def artifact_ref_v1_from_dict(value: object) -> ArtifactRefV1:
    """Parse one exact artifact-reference nested record."""

    exact = _exact_dict(
        value,
        frozenset({"body_sha256", "file_sha256", "schema_version"}),
        "artifact reference",
    )
    return ArtifactRefV1(
        schema_version=exact["schema_version"],
        file_sha256=exact["file_sha256"],
        body_sha256=exact["body_sha256"],
    )


def producer_ref_v1_from_dict(value: object) -> ProducerRefV1:
    """Parse one exact producer-reference nested record."""

    exact = _exact_dict(
        value,
        frozenset(
            {
                "descriptor_body_sha256",
                "descriptor_file_sha256",
                "descriptor_schema_version",
                "role",
                "source_sha256",
            }
        ),
        "producer reference",
    )
    return ProducerRefV1(
        role=exact["role"],
        descriptor_schema_version=exact["descriptor_schema_version"],
        descriptor_file_sha256=exact["descriptor_file_sha256"],
        descriptor_body_sha256=exact["descriptor_body_sha256"],
        source_sha256=exact["source_sha256"],
    )


def producer_refs_v1_from_json(value: object) -> tuple[ProducerRefV1, ...]:
    """Parse and validate one exact JSON array containing all six producers."""

    if type(value) is not list:
        _fail("producer references must be one exact JSON array")
    producers = tuple(producer_ref_v1_from_dict(item) for item in cast(list[object], value))
    return validate_producer_refs_v1(producers)


def case_subject_v1_from_dict(value: object) -> CaseSubjectV1:
    """Parse one exact matched-v3 case-subject nested record."""

    exact = _exact_dict(
        value,
        frozenset({"candidate_family", "candidate_id", "case_ordinal", "qualification_case_id"}),
        "case subject",
    )
    return CaseSubjectV1(
        case_ordinal=exact["case_ordinal"],
        candidate_id=exact["candidate_id"],
        candidate_family=exact["candidate_family"],
        qualification_case_id=exact["qualification_case_id"],
    )


def raw_byte_capture_v1_from_dict(value: object) -> RawByteCaptureV1:
    """Parse one exact verified raw-byte-capture nested record."""

    exact = _exact_dict(
        value,
        frozenset({"raw_bytes_base64", "raw_sha256", "raw_size_bytes", "schema_version"}),
        "raw-byte capture",
    )
    return RawByteCaptureV1(
        raw_bytes_base64=exact["raw_bytes_base64"],
        raw_size_bytes=exact["raw_size_bytes"],
        raw_sha256=exact["raw_sha256"],
        schema_version=exact["schema_version"],
    )


if (
    MATCHED_V3_CANDIDATE_IDS
    != MATCHED_V3_LOCAL_CANDIDATE_IDS
    + MATCHED_V3_EXTERNAL_CANDIDATE_IDS[:9]
    + MATCHED_V3_ADAPTER_CANDIDATE_IDS
    + MATCHED_V3_EXTERNAL_CANDIDATE_IDS[9:]
    or len(MATCHED_V3_CANDIDATE_IDS) != 28
    or len(set(MATCHED_V3_CANDIDATE_IDS)) != 28
):
    raise AssertionError("frozen matched-v3 candidate order drifted")
if frozenset(PRODUCER_DESCRIPTOR_SCHEMA_BY_ROLE) != frozenset(PRODUCER_ROLES):
    raise AssertionError("frozen producer role/schema mapping drifted")


__all__ = [
    "ArtifactRefV1",
    "CandidateFamily",
    "CanonicalEvidenceError",
    "CaseSubjectV1",
    "MATCHED_V3_ADAPTER_CANDIDATE_IDS",
    "MATCHED_V3_CANDIDATE_IDS",
    "MATCHED_V3_EXTERNAL_CANDIDATE_IDS",
    "MATCHED_V3_LOCAL_CANDIDATE_IDS",
    "MAX_CANONICAL_FILE_BYTES",
    "MAX_EXACT_INTEGER",
    "MAX_JSON_DEPTH",
    "MAX_JSON_NODES",
    "MAX_JSON_STRING_LENGTH",
    "MAX_RAW_CAPTURE_BYTES",
    "PRODUCER_DESCRIPTOR_SCHEMA_BY_ROLE",
    "PRODUCER_ROLES",
    "ProducerRefV1",
    "ProducerRole",
    "RAW_BYTE_CAPTURE_V1_SCHEMA_VERSION",
    "RawByteCaptureV1",
    "artifact_ref_v1_from_dict",
    "candidate_family",
    "canonical_body_sha256",
    "canonical_file_bytes",
    "canonical_json_bytes",
    "case_subject_v1_from_dict",
    "decode_canonical_json_file",
    "producer_ref_v1_from_dict",
    "producer_refs_v1_from_json",
    "raw_byte_capture_v1_from_dict",
    "require_distinct_sha256s",
    "require_nonzero_sha256",
    "validate_body_sha256",
    "validate_canonical_file",
    "validate_file_sha256",
    "validate_producer_refs_v1",
]
