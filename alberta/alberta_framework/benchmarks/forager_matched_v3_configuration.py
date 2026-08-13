"""Strict typed configuration derivation for future matched-current v3 arms.

This module is deliberately independent from the v2 dotted-key, integer-only
transform machinery.  It accepts an external configuration as raw bytes,
validates a versioned descriptor, applies scalar replacements through exact
RFC 6901 JSON Pointers, and returns content-addressed in-memory bytes.  It does
not read or write files and does not import or execute candidate code.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Final, Literal, cast

DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3_configuration_transform.v1"
)

_MAX_JSON_BYTES: Final = 2 * 1024 * 1024
_MAX_JSON_DEPTH: Final = 128
_MAX_JSON_NODES: Final = 100_000
_MAX_POINTER_LENGTH: Final = 4_096
_MAX_TRANSFORMS: Final = 512

type ScalarType = Literal["string", "integer", "number", "boolean", "null"]
type JsonScalar = str | int | float | Decimal | bool | None


class ForagerMatchedV3ConfigurationError(ValueError):
    """A v3 external configuration or transformation descriptor is invalid."""


@dataclass(frozen=True)
class TypedJsonPointerTransform:
    """One exact, scalar-valued RFC 6901 configuration replacement.

    Parsed values declared as ``number`` are stored as :class:`Decimal` so
    authored decimal values never pass through binary64.  A directly
    constructed instance may contain a finite Python ``float``; validation
    interprets that value through Python's shortest round-tripping JSON token
    before converting it to ``Decimal``.  Callers needing an authored decimal
    value that binary64 cannot represent must use strict JSON bytes/text or a
    ``Decimal`` value.
    """

    pointer: str
    value_type: ScalarType
    expected_original: JsonScalar
    replacement: JsonScalar

    def to_dict(self) -> dict[str, Any]:
        """Return a detached representation accepted by this canonicalizer."""
        return {
            "pointer": self.pointer,
            "value_type": self.value_type,
            "expected_original": self.expected_original,
            "replacement": self.replacement,
        }


@dataclass(frozen=True)
class DerivedConfigurationDescriptor:
    """Versioned, canonically ordered external-configuration transform plan."""

    schema_version: str
    transforms: tuple[TypedJsonPointerTransform, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a detached representation accepted by this canonicalizer."""
        return {
            "schema_version": self.schema_version,
            "transforms": [transform.to_dict() for transform in self.transforms],
        }


@dataclass(frozen=True)
class DerivedConfiguration:
    """Raw source identity and canonical derived configuration identity."""

    original_bytes: bytes
    original_sha256: str
    descriptor: DerivedConfigurationDescriptor
    descriptor_canonical_bytes: bytes
    descriptor_sha256: str
    derived_canonical_bytes: bytes
    derived_sha256: str


type DescriptorInput = DerivedConfigurationDescriptor | Mapping[str, Any] | bytes | str


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ForagerMatchedV3ConfigurationError(
                f"duplicate JSON object key {key!r} is forbidden"
            )
        result[key] = value
    return result


def _reject_nonfinite(token: str) -> Any:
    raise ForagerMatchedV3ConfigurationError(
        f"non-finite JSON number {token!r} is forbidden"
    )


def _parse_exact_decimal(token: str) -> Decimal:
    try:
        value = Decimal(token)
    except (InvalidOperation, ValueError) as exc:
        raise ForagerMatchedV3ConfigurationError(
            f"invalid JSON number {token!r}"
        ) from exc
    if not value.is_finite():
        raise ForagerMatchedV3ConfigurationError(
            f"non-finite JSON number {token!r} is forbidden"
        )
    return value


def _validate_json_tree(value: Any, *, label: str) -> None:
    pending: list[tuple[Any, int]] = [(value, 0)]
    seen_containers: set[int] = set()
    nodes = 0
    while pending:
        item, depth = pending.pop()
        nodes += 1
        if nodes > _MAX_JSON_NODES:
            raise ForagerMatchedV3ConfigurationError(
                f"{label} exceeds the JSON node limit"
            )
        if depth > _MAX_JSON_DEPTH:
            raise ForagerMatchedV3ConfigurationError(
                f"{label} exceeds the JSON nesting limit"
            )
        if isinstance(item, Mapping):
            identity = id(item)
            if identity in seen_containers:
                raise ForagerMatchedV3ConfigurationError(
                    f"{label} must be an unaliased acyclic JSON tree"
                )
            seen_containers.add(identity)
            for key, child in item.items():
                if not isinstance(key, str):
                    raise ForagerMatchedV3ConfigurationError(
                        f"{label} JSON object keys must be strings"
                    )
                try:
                    key.encode("utf-8")
                except UnicodeEncodeError as exc:
                    raise ForagerMatchedV3ConfigurationError(
                        f"{label} JSON object keys must contain valid Unicode"
                    ) from exc
                pending.append((child, depth + 1))
        elif isinstance(item, list):
            identity = id(item)
            if identity in seen_containers:
                raise ForagerMatchedV3ConfigurationError(
                    f"{label} must be an unaliased acyclic JSON tree"
                )
            seen_containers.add(identity)
            pending.extend((child, depth + 1) for child in item)
        elif isinstance(item, str):
            try:
                item.encode("utf-8")
            except UnicodeEncodeError as exc:
                raise ForagerMatchedV3ConfigurationError(
                    f"{label} JSON strings must contain valid Unicode"
                ) from exc
        elif isinstance(item, float):
            if not math.isfinite(item):
                raise ForagerMatchedV3ConfigurationError(
                    f"{label} contains a non-finite JSON number"
                )
        elif isinstance(item, Decimal):
            if not item.is_finite():
                raise ForagerMatchedV3ConfigurationError(
                    f"{label} contains a non-finite JSON number"
                )
        elif item is not None and not isinstance(item, (bool, int)):
            raise ForagerMatchedV3ConfigurationError(
                f"{label} contains non-JSON value type {type(item).__name__}"
            )


def _decode_strict_json(data: bytes | str, *, label: str) -> Any:
    try:
        if isinstance(data, bytes):
            if len(data) > _MAX_JSON_BYTES:
                raise ForagerMatchedV3ConfigurationError(
                    f"{label} exceeds the JSON byte limit"
                )
            text = data.decode("utf-8")
        elif isinstance(data, str):
            encoded = data.encode("utf-8")
            if len(encoded) > _MAX_JSON_BYTES:
                raise ForagerMatchedV3ConfigurationError(
                    f"{label} exceeds the JSON byte limit"
                )
            text = data
        else:
            raise ForagerMatchedV3ConfigurationError(
                f"{label} must be strict UTF-8 JSON bytes or text"
            )
        decoded = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
            parse_float=_parse_exact_decimal,
        )
    except ForagerMatchedV3ConfigurationError:
        raise
    except (UnicodeError, ValueError, RecursionError, OverflowError) as exc:
        raise ForagerMatchedV3ConfigurationError(
            f"{label} is not strict UTF-8 JSON: {exc}"
        ) from exc
    _validate_json_tree(decoded, label=label)
    return decoded


def _canonical_decimal_token(value: Decimal, *, label: str) -> str:
    """Encode one finite decimal without rounding or exponent expansion.

    Numerically equal nonzero decimals have one representation: insignificant
    trailing zeroes are removed, then the shorter of plain and scientific
    notation is selected (plain wins ties).  Integral decimals retain either
    a decimal point or exponent so strict JSON consumers continue to distinguish
    declared ``number`` values from ``integer`` values.  Signed zero remains
    distinct and uses ``-0.0`` or ``0.0`` so replay continues through
    ``parse_float``.
    """
    if not value.is_finite():
        raise ForagerMatchedV3ConfigurationError(
            f"{label} contains a non-finite JSON number"
        )
    sign, raw_digits, raw_exponent = value.as_tuple()
    if not raw_digits or all(digit == 0 for digit in raw_digits):
        return "-0.0" if sign else "0.0"
    exponent = cast(int, raw_exponent)
    digits = list(raw_digits)
    while digits[-1] == 0:
        digits.pop()
        exponent += 1
    digit_text = "".join(str(digit) for digit in digits)
    adjusted = len(digit_text) + exponent - 1

    if exponent >= 0:
        plain_length = len(digit_text) + exponent + 2
    elif -exponent < len(digit_text):
        plain_length = len(digit_text) + 1
    else:
        plain_length = 2 - exponent
    exponent_text = str(adjusted)
    coefficient_length = len(digit_text) if len(digit_text) == 1 else len(digit_text) + 1
    scientific_length = coefficient_length + 1 + len(exponent_text)

    prefix = "-" if sign else ""
    if plain_length <= scientific_length:
        if exponent >= 0:
            body = digit_text + ("0" * exponent) + ".0"
        else:
            split = len(digit_text) + exponent
            if split > 0:
                body = f"{digit_text[:split]}.{digit_text[split:]}"
            else:
                body = f"0.{('0' * -split)}{digit_text}"
        return prefix + body

    coefficient = (
        digit_text
        if len(digit_text) == 1
        else f"{digit_text[0]}.{digit_text[1:]}"
    )
    return f"{prefix}{coefficient}e{exponent_text}"


def _decimal_from_float(value: float, *, label: str) -> Decimal:
    if not math.isfinite(value):
        raise ForagerMatchedV3ConfigurationError(
            f"{label} contains a non-finite JSON number"
        )
    try:
        token = json.dumps(value, allow_nan=False)
        return Decimal(token)
    except (InvalidOperation, TypeError, ValueError, OverflowError) as exc:
        raise ForagerMatchedV3ConfigurationError(
            f"{label} is not a finite JSON number"
        ) from exc


def _canonical_json_bytes(value: Any, *, label: str) -> bytes:
    """Encode bounded canonical JSON, including exact :class:`Decimal` values."""
    _validate_json_tree(value, label=label)
    chunks: list[bytes] = []
    encoded_size = 0

    def append(text: str) -> None:
        nonlocal encoded_size
        raw = text.encode("utf-8")
        encoded_size += len(raw)
        if encoded_size > _MAX_JSON_BYTES:
            raise ForagerMatchedV3ConfigurationError(
                f"{label} exceeds the canonical JSON byte limit"
            )
        chunks.append(raw)

    def encode(item: Any) -> None:
        if isinstance(item, Mapping):
            append("{")
            for index, key in enumerate(sorted(item)):
                if index:
                    append(",")
                append(json.dumps(key, allow_nan=False, ensure_ascii=False))
                append(":")
                encode(item[key])
            append("}")
        elif isinstance(item, list):
            append("[")
            for index, child in enumerate(item):
                if index:
                    append(",")
                encode(child)
            append("]")
        elif isinstance(item, str):
            append(json.dumps(item, allow_nan=False, ensure_ascii=False))
        elif item is None:
            append("null")
        elif type(item) is bool:
            append("true" if item else "false")
        elif type(item) is int:
            append(str(item))
        elif type(item) is float:
            append(
                _canonical_decimal_token(
                    _decimal_from_float(item, label=label),
                    label=label,
                )
            )
        elif isinstance(item, Decimal):
            append(_canonical_decimal_token(item, label=label))
        else:
            raise ForagerMatchedV3ConfigurationError(
                f"{label} contains non-JSON value type {type(item).__name__}"
            )

    try:
        encode(value)
    except ForagerMatchedV3ConfigurationError:
        raise
    except (TypeError, ValueError, UnicodeError, RecursionError, OverflowError) as exc:
        raise ForagerMatchedV3ConfigurationError(
            f"{label} is not canonical JSON data: {exc}"
        ) from exc
    return b"".join(chunks)


def _require_object(value: Any, *, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ForagerMatchedV3ConfigurationError(f"{path} must be a JSON object")
    return cast(dict[str, Any], value)


def _require_exact_keys(
    value: Mapping[str, Any],
    *,
    path: str,
    required: frozenset[str],
) -> None:
    actual = set(value)
    missing = sorted(required - actual)
    unknown = sorted(actual - required)
    if missing:
        raise ForagerMatchedV3ConfigurationError(
            f"{path} is missing required keys: {', '.join(missing)}"
        )
    if unknown:
        raise ForagerMatchedV3ConfigurationError(
            f"{path} contains unknown keys: {', '.join(unknown)}"
        )


def _decode_pointer(pointer: Any, *, path: str) -> tuple[str, ...]:
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        raise ForagerMatchedV3ConfigurationError(
            f"{path} must be a non-root RFC 6901 JSON Pointer beginning with '/'"
        )
    if len(pointer) > _MAX_POINTER_LENGTH:
        raise ForagerMatchedV3ConfigurationError(f"{path} is too long")
    try:
        pointer.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ForagerMatchedV3ConfigurationError(
            f"{path} must contain valid Unicode"
        ) from exc

    decoded: list[str] = []
    for raw_token in pointer[1:].split("/"):
        token: list[str] = []
        index = 0
        while index < len(raw_token):
            character = raw_token[index]
            if character != "~":
                token.append(character)
                index += 1
                continue
            if index + 1 >= len(raw_token) or raw_token[index + 1] not in {"0", "1"}:
                raise ForagerMatchedV3ConfigurationError(
                    f"{path} contains an invalid RFC 6901 escape"
                )
            token.append("~" if raw_token[index + 1] == "0" else "/")
            index += 2
        decoded.append("".join(token))
    return tuple(decoded)


def _normalize_scalar(value: Any, value_type: ScalarType, *, path: str) -> JsonScalar:
    if value_type == "string":
        if not isinstance(value, str):
            raise ForagerMatchedV3ConfigurationError(f"{path} must be a string")
        try:
            value.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ForagerMatchedV3ConfigurationError(
                f"{path} must contain valid Unicode"
            ) from exc
        return value
    if value_type == "integer":
        if type(value) is not int:
            raise ForagerMatchedV3ConfigurationError(
                f"{path} must be an integer (boolean aliases are forbidden)"
            )
        return value
    if value_type == "number":
        if type(value) is int:
            return Decimal(value)
        if type(value) is float:
            return _decimal_from_float(value, label=path)
        if isinstance(value, Decimal) and value.is_finite():
            return value
        if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
            raise ForagerMatchedV3ConfigurationError(
                f"{path} must be a finite number (boolean aliases are forbidden)"
            )
        raise ForagerMatchedV3ConfigurationError(f"{path} must be a finite number")
    if value_type == "boolean":
        if type(value) is not bool:
            raise ForagerMatchedV3ConfigurationError(f"{path} must be a boolean")
        return value
    if value is not None:
        raise ForagerMatchedV3ConfigurationError(f"{path} must be null")
    return None


def _parse_transform(value: Any, *, path: str) -> TypedJsonPointerTransform:
    payload = _require_object(value, path=path)
    _require_exact_keys(
        payload,
        path=path,
        required=frozenset(
            {"pointer", "value_type", "expected_original", "replacement"}
        ),
    )
    raw_value_type = payload["value_type"]
    allowed_types = ("string", "integer", "number", "boolean", "null")
    if not isinstance(raw_value_type, str) or raw_value_type not in allowed_types:
        raise ForagerMatchedV3ConfigurationError(
            f"{path}.value_type must be one of {', '.join(allowed_types)}"
        )
    value_type = cast(ScalarType, raw_value_type)
    pointer = payload["pointer"]
    _decode_pointer(pointer, path=f"{path}.pointer")
    return TypedJsonPointerTransform(
        pointer=cast(str, pointer),
        value_type=value_type,
        expected_original=_normalize_scalar(
            payload["expected_original"],
            value_type,
            path=f"{path}.expected_original",
        ),
        replacement=_normalize_scalar(
            payload["replacement"],
            value_type,
            path=f"{path}.replacement",
        ),
    )


def _descriptor_payload(value: DescriptorInput) -> Any:
    if isinstance(value, DerivedConfigurationDescriptor):
        return value.to_dict()
    if isinstance(value, (bytes, str)):
        return _decode_strict_json(value, label="transformation descriptor")
    if isinstance(value, Mapping):
        canonical = _canonical_json_bytes(value, label="transformation descriptor")
        return _decode_strict_json(canonical, label="transformation descriptor")
    raise ForagerMatchedV3ConfigurationError(
        "transformation descriptor must be a descriptor, JSON object, bytes, or text"
    )


def parse_transform_descriptor(value: DescriptorInput) -> DerivedConfigurationDescriptor:
    """Parse, detach, and strictly validate a versioned transform descriptor.

    Transform pointers must be unique and strictly ascending by their raw JSON
    Pointer strings using Unicode code-point order.  The canonical order makes
    semantically equivalent transform sets byte-identical and makes order drift
    fail closed.
    """
    payload = _require_object(_descriptor_payload(value), path="descriptor")
    _require_exact_keys(
        payload,
        path="descriptor",
        required=frozenset({"schema_version", "transforms"}),
    )
    if payload["schema_version"] != DESCRIPTOR_SCHEMA_VERSION:
        raise ForagerMatchedV3ConfigurationError(
            "descriptor.schema_version must equal " f"{DESCRIPTOR_SCHEMA_VERSION!r}"
        )
    raw_transforms = payload["transforms"]
    if not isinstance(raw_transforms, list):
        raise ForagerMatchedV3ConfigurationError("descriptor.transforms must be a JSON array")
    if len(raw_transforms) > _MAX_TRANSFORMS:
        raise ForagerMatchedV3ConfigurationError(
            "descriptor.transforms contains too many entries"
        )
    transforms = tuple(
        _parse_transform(item, path=f"descriptor.transforms[{index}]")
        for index, item in enumerate(raw_transforms)
    )
    pointers = tuple(transform.pointer for transform in transforms)
    if any(current <= previous for previous, current in zip(pointers, pointers[1:])):
        raise ForagerMatchedV3ConfigurationError(
            "descriptor.transforms pointer order must be unique and strictly ascending"
        )
    descriptor = DerivedConfigurationDescriptor(
        schema_version=DESCRIPTOR_SCHEMA_VERSION,
        transforms=transforms,
    )
    _canonical_json_bytes(descriptor.to_dict(), label="transformation descriptor")
    return descriptor


def canonical_descriptor_bytes(value: DescriptorInput) -> bytes:
    """Return canonical UTF-8 JSON bytes after replaying descriptor validation."""
    descriptor = parse_transform_descriptor(value)
    return _canonical_json_bytes(
        descriptor.to_dict(),
        label="transformation descriptor",
    )


def canonical_descriptor_sha256(value: DescriptorInput) -> str:
    """Return the SHA-256 digest of :func:`canonical_descriptor_bytes`."""
    return hashlib.sha256(canonical_descriptor_bytes(value)).hexdigest()


def _resolve_scalar_parent(
    root: dict[str, Any],
    *,
    pointer: str,
    path: str,
) -> tuple[dict[str, Any], str, Any]:
    tokens = _decode_pointer(pointer, path=path)
    current: Any = root
    for token in tokens[:-1]:
        if not isinstance(current, dict):
            raise ForagerMatchedV3ConfigurationError(
                f"{path} may traverse JSON objects only"
            )
        if token not in current:
            raise ForagerMatchedV3ConfigurationError(
                f"{path} does not exist in the original configuration"
            )
        current = current[token]
    if not isinstance(current, dict):
        raise ForagerMatchedV3ConfigurationError(
            f"{path} may traverse JSON objects only"
        )
    leaf = tokens[-1]
    if leaf not in current:
        raise ForagerMatchedV3ConfigurationError(
            f"{path} does not exist in the original configuration"
        )
    actual = current[leaf]
    if isinstance(actual, (dict, list)):
        raise ForagerMatchedV3ConfigurationError(
            f"{path} must select a scalar JSON value"
        )
    return current, leaf, actual


def _exact_scalar_bytes(value: JsonScalar) -> bytes:
    return _canonical_json_bytes(value, label="scalar comparison")


def derive_configuration(
    original_bytes: bytes,
    descriptor: DescriptorInput,
) -> DerivedConfiguration:
    """Derive canonical configuration bytes without filesystem or runtime effects.

    The original digest covers the caller's exact bytes, including insignificant
    whitespace.  Every transform first proves the selected source value has the
    declared scalar type and exact expected value.  The derived digest covers
    canonical JSON bytes after all replacements.
    """
    if not isinstance(original_bytes, bytes):
        raise ForagerMatchedV3ConfigurationError(
            "original configuration must be bytes so its raw identity is exact"
        )
    raw_original = bytes(original_bytes)
    decoded = _decode_strict_json(raw_original, label="original configuration")
    if not isinstance(decoded, dict):
        raise ForagerMatchedV3ConfigurationError(
            "configuration root must be a JSON object"
        )
    root = cast(dict[str, Any], decoded)
    parsed_descriptor = parse_transform_descriptor(descriptor)
    for index, transform in enumerate(parsed_descriptor.transforms):
        path = f"descriptor.transforms[{index}].pointer"
        parent, leaf, actual = _resolve_scalar_parent(
            root,
            pointer=transform.pointer,
            path=path,
        )
        try:
            normalized_actual = _normalize_scalar(
                actual,
                transform.value_type,
                path=f"{path} actual value",
            )
        except ForagerMatchedV3ConfigurationError as exc:
            raise ForagerMatchedV3ConfigurationError(
                f"{path} actual value type does not match declared "
                f"{transform.value_type!r}"
            ) from exc
        if _exact_scalar_bytes(normalized_actual) != _exact_scalar_bytes(
            transform.expected_original
        ):
            raise ForagerMatchedV3ConfigurationError(
                f"{path} expected_original does not exactly match the source value"
            )
        parent[leaf] = transform.replacement

    descriptor_bytes = canonical_descriptor_bytes(parsed_descriptor)
    derived_bytes = _canonical_json_bytes(root, label="derived configuration")
    return DerivedConfiguration(
        original_bytes=raw_original,
        original_sha256=hashlib.sha256(raw_original).hexdigest(),
        descriptor=parsed_descriptor,
        descriptor_canonical_bytes=descriptor_bytes,
        descriptor_sha256=hashlib.sha256(descriptor_bytes).hexdigest(),
        derived_canonical_bytes=derived_bytes,
        derived_sha256=hashlib.sha256(derived_bytes).hexdigest(),
    )


__all__ = [
    "DESCRIPTOR_SCHEMA_VERSION",
    "DerivedConfiguration",
    "DerivedConfigurationDescriptor",
    "ForagerMatchedV3ConfigurationError",
    "TypedJsonPointerTransform",
    "canonical_descriptor_bytes",
    "canonical_descriptor_sha256",
    "derive_configuration",
    "parse_transform_descriptor",
]
