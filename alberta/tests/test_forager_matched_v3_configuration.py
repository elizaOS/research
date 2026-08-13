"""Tests for strict matched-current v3 external configuration transforms."""

from __future__ import annotations

import hashlib
import json
from dataclasses import FrozenInstanceError
from decimal import Decimal
from typing import Any

import pytest

from alberta_framework.benchmarks import forager_matched_v3_configuration as configuration


def _transform(
    pointer: str,
    value_type: str,
    expected_original: object,
    replacement: object,
) -> dict[str, object]:
    return {
        "pointer": pointer,
        "value_type": value_type,
        "expected_original": expected_original,
        "replacement": replacement,
    }


def _descriptor(*transforms: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": configuration.DESCRIPTOR_SCHEMA_VERSION,
        "transforms": list(transforms),
    }


@pytest.mark.unit
def test_derivation_hashes_raw_source_and_emits_canonical_detached_bytes() -> None:
    original = (
        b'{ "nil" : null, "n": 1.25, "count": 1, "bool": false, '
        b'"a/b": {"~key": "old"} }\n'
    )
    descriptor = _descriptor(
        _transform("/a~1b/~0key", "string", "old", "new"),
        _transform("/bool", "boolean", False, True),
        _transform("/count", "integer", 1, 2),
        _transform("/n", "number", 1.25, 2),
        _transform("/nil", "null", None, None),
    )

    result = configuration.derive_configuration(original, descriptor)

    expected = {
        "a/b": {"~key": "new"},
        "bool": True,
        "count": 2,
        "n": 2.0,
        "nil": None,
    }
    expected_bytes = json.dumps(
        expected,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    assert result.original_bytes == original
    assert result.original_sha256 == hashlib.sha256(original).hexdigest()
    assert result.derived_canonical_bytes == expected_bytes
    assert result.derived_sha256 == hashlib.sha256(expected_bytes).hexdigest()
    assert result.descriptor_sha256 == hashlib.sha256(
        result.descriptor_canonical_bytes
    ).hexdigest()
    assert original != expected_bytes


@pytest.mark.unit
def test_descriptor_canonical_bytes_and_digest_replay_strict_validation() -> None:
    raw = json.dumps(
        _descriptor(_transform("/x", "integer", 1, 3)),
        indent=2,
    ).encode("utf-8")

    parsed = configuration.parse_transform_descriptor(raw)
    canonical = configuration.canonical_descriptor_bytes(parsed)

    assert canonical == configuration.canonical_descriptor_bytes(raw)
    assert configuration.canonical_descriptor_sha256(raw) == hashlib.sha256(
        canonical
    ).hexdigest()
    assert configuration.parse_transform_descriptor(canonical) == parsed


@pytest.mark.unit
def test_descriptor_and_results_are_frozen_and_detached_from_callers() -> None:
    payload = _descriptor(_transform("/x", "integer", 1, 2))
    parsed = configuration.parse_transform_descriptor(payload)
    payload["transforms"][0]["replacement"] = 99  # type: ignore[index]
    detached = parsed.to_dict()
    detached["transforms"][0]["replacement"] = 77

    assert parsed.transforms[0].replacement == 2
    with pytest.raises(FrozenInstanceError):
        parsed.schema_version = "changed"  # type: ignore[misc]

    result = configuration.derive_configuration(
        b'{"x":1}',
        _descriptor(_transform("/x", "integer", 1, 2)),
    )
    with pytest.raises(FrozenInstanceError):
        result.derived_sha256 = "0" * 64  # type: ignore[misc]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("raw", "match"),
    [
        (b'{"x":1,"x":2}', "duplicate JSON object key"),
        (b'{"x":NaN}', "non-finite JSON number"),
        (b'{"x":Infinity}', "non-finite JSON number"),
        (b'\xff', "strict UTF-8 JSON"),
        (b'[]', "configuration root must be a JSON object"),
    ],
)
def test_original_configuration_must_be_strict_object_json(raw: bytes, match: str) -> None:
    with pytest.raises(configuration.ForagerMatchedV3ConfigurationError, match=match):
        configuration.derive_configuration(raw, _descriptor())


@pytest.mark.unit
@pytest.mark.parametrize(
    ("raw", "match"),
    [
        (
            b'{"schema_version":"alberta.forager_matched_v3_configuration_transform.v1",'
            b'"schema_version":"other","transforms":[]}',
            "duplicate JSON object key",
        ),
        (
            b'{"schema_version":"alberta.forager_matched_v3_configuration_transform.v1",'
            b'"transforms":[{"pointer":"/x","value_type":"number",'
            b'"expected_original":0,"replacement":NaN}]}',
            "non-finite JSON number",
        ),
        (b'\xff', "strict UTF-8 JSON"),
    ],
)
def test_descriptor_bytes_must_be_strict_json(raw: bytes, match: str) -> None:
    with pytest.raises(configuration.ForagerMatchedV3ConfigurationError, match=match):
        configuration.parse_transform_descriptor(raw)


@pytest.mark.unit
@pytest.mark.parametrize(
    "payload",
    [
        {"schema_version": configuration.DESCRIPTOR_SCHEMA_VERSION},
        {
            "schema_version": configuration.DESCRIPTOR_SCHEMA_VERSION,
            "transforms": [],
            "unknown": None,
        },
        {"schema_version": "v0", "transforms": []},
        {"schema_version": configuration.DESCRIPTOR_SCHEMA_VERSION, "transforms": {}},
        {
            "schema_version": configuration.DESCRIPTOR_SCHEMA_VERSION,
            "transforms": [{"pointer": "/x"}],
        },
        {
            "schema_version": configuration.DESCRIPTOR_SCHEMA_VERSION,
            "transforms": [
                {
                    **_transform("/x", "integer", 1, 2),
                    "unknown": None,
                }
            ],
        },
    ],
)
def test_descriptor_schema_uses_exact_keys_and_version(payload: dict[str, object]) -> None:
    with pytest.raises(configuration.ForagerMatchedV3ConfigurationError):
        configuration.parse_transform_descriptor(payload)


@pytest.mark.unit
@pytest.mark.parametrize(
    "transforms",
    [
        (
            _transform("/x", "integer", 1, 2),
            _transform("/x", "integer", 1, 3),
        ),
        (
            _transform("/z", "integer", 1, 2),
            _transform("/a", "integer", 1, 3),
        ),
    ],
)
def test_duplicate_pointers_and_noncanonical_order_are_rejected(
    transforms: tuple[dict[str, object], ...],
) -> None:
    with pytest.raises(configuration.ForagerMatchedV3ConfigurationError, match="pointer order"):
        configuration.parse_transform_descriptor(_descriptor(*transforms))


@pytest.mark.unit
@pytest.mark.parametrize(
    ("pointer", "match"),
    [
        ("", "non-root RFC 6901"),
        ("x", "non-root RFC 6901"),
        ("/bad~", "invalid RFC 6901 escape"),
        ("/bad~2escape", "invalid RFC 6901 escape"),
        ("/missing", "does not exist"),
        ("/items/0", "objects only"),
        ("/object", "must select a scalar"),
    ],
)
def test_pointer_resolution_is_exact_rfc6901_and_object_only(pointer: str, match: str) -> None:
    original = b'{"items":[1],"object":{"child":1},"x":1}'
    descriptor = _descriptor(_transform(pointer, "integer", 1, 2))

    with pytest.raises(configuration.ForagerMatchedV3ConfigurationError, match=match):
        configuration.derive_configuration(original, descriptor)


@pytest.mark.unit
def test_rfc6901_empty_key_and_escape_order_are_supported() -> None:
    original = b'{"":1,"~1":2}'
    descriptor = _descriptor(
        _transform("/", "integer", 1, 3),
        _transform("/~01", "integer", 2, 4),
    )

    result = configuration.derive_configuration(original, descriptor)

    assert json.loads(result.derived_canonical_bytes) == {"": 3, "~1": 4}


@pytest.mark.unit
@pytest.mark.parametrize(
    ("value_type", "expected", "replacement"),
    [
        ("integer", True, 2),
        ("integer", 1, False),
        ("number", True, 2.0),
        ("number", 1.0, False),
        ("boolean", 1, True),
        ("string", "x", 2),
        ("null", None, 0),
        ("object", 1, 2),
    ],
)
def test_declared_scalar_types_reject_aliases_and_mismatches(
    value_type: str,
    expected: object,
    replacement: object,
) -> None:
    descriptor = _descriptor(_transform("/x", value_type, expected, replacement))

    with pytest.raises(
        configuration.ForagerMatchedV3ConfigurationError,
        match="value_type|must be",
    ):
        configuration.parse_transform_descriptor(descriptor)


@pytest.mark.unit
def test_actual_value_type_and_expected_value_must_match_exactly() -> None:
    integer_for_float = _descriptor(_transform("/x", "integer", 1, 2))
    wrong_expected = _descriptor(_transform("/x", "integer", 2, 3))

    with pytest.raises(configuration.ForagerMatchedV3ConfigurationError, match="actual value type"):
        configuration.derive_configuration(b'{"x":1.0}', integer_for_float)
    with pytest.raises(configuration.ForagerMatchedV3ConfigurationError, match="expected_original"):
        configuration.derive_configuration(b'{"x":1}', wrong_expected)


@pytest.mark.unit
def test_number_comparison_does_not_collapse_large_integers_through_float() -> None:
    descriptor = _descriptor(
        _transform("/x", "number", 9_007_199_254_740_992, 2),
    )

    with pytest.raises(configuration.ForagerMatchedV3ConfigurationError, match="expected_original"):
        configuration.derive_configuration(b'{"x":9007199254740993}', descriptor)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("source_number", "expected_number"),
    [
        ("9007199254740993.0", "9007199254740992.0"),
        ("1e-4000", "0.0"),
    ],
)
def test_decimal_json_numbers_never_alias_through_binary64(
    source_number: str,
    expected_number: str,
) -> None:
    descriptor = (
        '{"schema_version":"alberta.forager_matched_v3_configuration_transform.v1",'
        '"transforms":[{"expected_original":'
        f"{expected_number},"
        '"pointer":"/x","replacement":2.5,"value_type":"number"}]}'
    ).encode()
    source = f'{{"x":{source_number}}}'.encode()

    with pytest.raises(configuration.ForagerMatchedV3ConfigurationError, match="expected_original"):
        configuration.derive_configuration(source, descriptor)


@pytest.mark.unit
def test_exact_decimal_replacement_has_deterministic_strict_json_bytes() -> None:
    descriptor = (
        b'{"schema_version":"alberta.forager_matched_v3_configuration_transform.v1",'
        b'"transforms":[{"expected_original":0.1,"pointer":"/x",'
        b'"replacement":9007199254740993.0,"value_type":"number"}]}'
    )

    parsed = configuration.parse_transform_descriptor(descriptor)
    result = configuration.derive_configuration(b'{"x":1e-1}', parsed)

    assert parsed.transforms[0].expected_original == Decimal("0.1")
    assert parsed.transforms[0].replacement == Decimal("9007199254740993.0")
    assert result.derived_canonical_bytes == b'{"x":9007199254740993.0}'
    assert isinstance(json.loads(result.derived_canonical_bytes)["x"], float)
    assert json.loads(
        result.derived_canonical_bytes,
        parse_float=Decimal,
    ) == {"x": Decimal("9007199254740993.0")}


@pytest.mark.unit
def test_untransformed_extreme_decimal_remains_exact_and_strict() -> None:
    result = configuration.derive_configuration(b'{"tiny":1e-4000}', _descriptor())

    assert result.derived_canonical_bytes == b'{"tiny":1e-4000}'
    assert json.loads(
        result.derived_canonical_bytes,
        parse_float=Decimal,
    ) == {"tiny": Decimal("1e-4000")}


@pytest.mark.unit
def test_parse_success_guarantees_direct_descriptor_is_canonicalizable_within_limit() -> None:
    oversized = configuration.DerivedConfigurationDescriptor(
        schema_version=configuration.DESCRIPTOR_SCHEMA_VERSION,
        transforms=(
            configuration.TypedJsonPointerTransform(
                pointer="/x",
                value_type="string",
                expected_original="",
                replacement="x" * (2 * 1024 * 1024),
            ),
        ),
    )

    with pytest.raises(
        configuration.ForagerMatchedV3ConfigurationError,
        match="canonical JSON byte limit",
    ):
        configuration.parse_transform_descriptor(oversized)


@pytest.mark.unit
def test_source_scalar_type_must_match_declared_type_without_bool_aliasing() -> None:
    descriptor = _descriptor(_transform("/x", "number", 1, 2))

    with pytest.raises(configuration.ForagerMatchedV3ConfigurationError, match="actual value type"):
        configuration.derive_configuration(b'{"x":true}', descriptor)


@pytest.mark.unit
def test_str_inputs_are_validated_but_source_must_remain_raw_bytes() -> None:
    descriptor_text = json.dumps(_descriptor(_transform("/x", "integer", 1, 2)))
    assert configuration.parse_transform_descriptor(descriptor_text).transforms[0].pointer == "/x"

    with pytest.raises(configuration.ForagerMatchedV3ConfigurationError, match="must be bytes"):
        configuration.derive_configuration('{"x":1}', descriptor_text)  # type: ignore[arg-type]


@pytest.mark.unit
def test_non_json_values_in_mapping_descriptor_are_rejected() -> None:
    cyclic: dict[str, Any] = _descriptor()
    cyclic["transforms"] = [
        _transform("/x", "string", "a", object()),
    ]
    with pytest.raises(configuration.ForagerMatchedV3ConfigurationError, match="non-JSON value"):
        configuration.parse_transform_descriptor(cyclic)
