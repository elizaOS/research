"""Cheap adversarial tests for the private matched-v3 canonical-evidence codec."""

from __future__ import annotations

import base64
import binascii
import dataclasses
import hashlib
import json
import random
from typing import Any

import pytest

from alberta_framework.benchmarks import _forager_matched_v3_canonical_evidence as codec
from alberta_framework.benchmarks import forager_matched_v3_qualification_plan_v3 as plan_v3

EXPECTED_CANDIDATE_ORDER = (
    "causal_e025_q050",
    "causal_e025_q075",
    "causal_e025_q090",
    "causal_e050_q050",
    "causal_e050_q075",
    "causal_e050_q090",
    "causal_e100_q050",
    "causal_e100_q075",
    "causal_e100_q090",
    "alberta_horde_default",
    "alberta_horde_eps05",
    "alberta_horde_recurrent64",
    "alberta_horde_step3e3",
    "alberta_rtu_h08_taylor",
    "external_dqn_plain",
    "external_dqn_crelu",
    "external_dqn_redo",
    "external_dqn_reward_trace",
    "external_dqn_l2_init",
    "external_pt_dqn_xfinal",
    "external_drqn_xfinal",
    "isolated_ppo_generic",
    "isolated_rtu_paper_scale",
    "adapted_full_rainbow",
    "adapted_ppo_gru",
    "random_policy",
    "search_nearest",
    "search_oracle",
)
EXPECTED_PRODUCER_SCHEMAS = {
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
EXPECTED_ALL = [
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


class _StrSubclass(str):
    pass


class _IntSubclass(int):
    pass


class _DictSubclass(dict[str, Any]):
    pass


class _EqualityProxy:
    def __init__(self, target: object) -> None:
        self.target = target

    def __eq__(self, other: object) -> bool:
        return bool(other == self.target)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _artifact() -> codec.ArtifactRefV1:
    return codec.ArtifactRefV1(
        schema_version="alberta.test.artifact.v1",
        file_sha256=_sha("artifact file"),
        body_sha256=_sha("artifact body"),
    )


def _producer(role: codec.ProducerRole, index: int) -> codec.ProducerRefV1:
    return codec.ProducerRefV1(
        role=role,
        descriptor_schema_version=codec.PRODUCER_DESCRIPTOR_SCHEMA_BY_ROLE[role],
        descriptor_file_sha256=_sha(f"producer {index} descriptor file"),
        descriptor_body_sha256=_sha(f"producer {index} descriptor body"),
        source_sha256=_sha(f"producer {index} source"),
    )


def _producers() -> tuple[codec.ProducerRefV1, ...]:
    return tuple(_producer(role, index) for index, role in enumerate(codec.PRODUCER_ROLES))


def test_canonical_json_has_sorted_ascii_and_exact_final_lf_framing() -> None:
    payload = {"z": True, "a": [None, -2, "plain ASCII"]}
    assert codec.canonical_json_bytes(payload) == b'{"a":[null,-2,"plain ASCII"],"z":true}\n'
    assert codec.canonical_json_bytes(payload, final_lf=False) == (
        b'{"a":[null,-2,"plain ASCII"],"z":true}'
    )
    assert codec.decode_canonical_json_file(codec.canonical_json_bytes(payload)) == payload


def test_canonical_size_preflight_accounts_for_every_escaped_ascii_byte() -> None:
    payload = {'key"\\': 'value"\\'}
    raw = codec.canonical_json_bytes(payload)
    assert raw == b'{"key\\"\\\\":"value\\"\\\\"}\n'
    assert len(raw) == 24


def test_canonical_size_preflight_accepts_exact_framed_and_unframed_boundaries() -> None:
    framed_overhead = len(b'{"a":"","b":""}\n')
    shared_length = (codec.MAX_CANONICAL_FILE_BYTES - framed_overhead) // 2
    assert framed_overhead + 2 * shared_length == codec.MAX_CANONICAL_FILE_BYTES
    shared = "x" * shared_length
    framed = codec.canonical_json_bytes({"a": shared, "b": shared})
    assert len(framed) == codec.MAX_CANONICAL_FILE_BYTES
    assert framed.endswith(b"\n")

    unframed = codec.canonical_json_bytes(
        {"a": shared, "b": shared + "x"},
        final_lf=False,
    )
    assert len(unframed) == codec.MAX_CANONICAL_FILE_BYTES
    with pytest.raises(codec.CanonicalEvidenceError, match="aggregate encoded byte bound"):
        codec.canonical_json_bytes({"a": shared, "b": shared + "x"})


def test_canonical_size_preflight_stops_shared_string_amplification_before_dump(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shared = "x" * codec.MAX_JSON_STRING_LENGTH
    payload = {"items": [shared] * 10_000}

    def unexpected_dump(*args: object, **kwargs: object) -> str:
        raise AssertionError("json.dumps must not run after preflight rejection")

    monkeypatch.setattr(json, "dumps", unexpected_dump)
    with pytest.raises(codec.CanonicalEvidenceError, match="aggregate encoded byte bound"):
        codec.canonical_json_bytes(payload)


def test_canonical_serializer_memory_error_is_normalized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def exhausted_dump(*args: object, **kwargs: object) -> str:
        raise MemoryError("synthetic serializer exhaustion")

    monkeypatch.setattr(json, "dumps", exhausted_dump)
    with pytest.raises(codec.CanonicalEvidenceError) as caught:
        codec.canonical_json_bytes({"small": "payload"})
    assert isinstance(caught.value.__cause__, MemoryError)


@pytest.mark.parametrize(
    "raw",
    [
        b'{"a":1}',
        b'{"a":1}\n\n',
        b'{"a":1}\r\n',
        b' {"a":1}\n',
        b'{ "a":1}\n',
        b'{"a":1 }\n',
        b'{"a":1,"a":1}\n',
        b'{"a":1.0}\n',
        b'{"a":1e0}\n',
        b'{"a":NaN}\n',
        b'{"a":Infinity}\n',
        b'{"a":9223372036854775808}\n',
        b'{"a":"\\u00e9"}\n',
        b'{"a":"\\u000a"}\n',
        b"[]\n",
        b'"object"\n',
    ],
)
def test_decoder_rejects_noncanonical_duplicate_float_unbounded_or_nonobject_json(
    raw: bytes,
) -> None:
    with pytest.raises(codec.CanonicalEvidenceError):
        codec.decode_canonical_json_file(raw)


@pytest.mark.parametrize("raw", [bytearray(b"{}\n"), "{}\n", b"", b"\xff\n"])
def test_decoder_rejects_nonexact_or_nonascii_file_bytes(raw: object) -> None:
    with pytest.raises(codec.CanonicalEvidenceError):
        codec.decode_canonical_json_file(raw)  # type: ignore[arg-type]


def test_decoder_rejects_negative_zero_as_noncanonical() -> None:
    with pytest.raises(codec.CanonicalEvidenceError, match="negative zero"):
        codec.decode_canonical_json_file(b'{"value":-0}\n')


def test_decoder_normalizes_json_loads_recursion_error_after_safe_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def recursive_loads(*args: object, **kwargs: object) -> object:
        return recursive_loads(*args, **kwargs)

    monkeypatch.setattr(json, "loads", recursive_loads)
    with pytest.raises(codec.CanonicalEvidenceError) as caught:
        codec.decode_canonical_json_file(b"{}\n")
    assert isinstance(caught.value.__cause__, RecursionError)


def test_scanner_rejects_deep_input_before_json_loads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nesting = 10_000
    raw = b'{"value":' + b"[" * nesting + b"null" + b"]" * nesting + b"}\n"

    def unexpected_loads(*args: object, **kwargs: object) -> object:
        raise AssertionError("json.loads must not run after scanner rejection")

    monkeypatch.setattr(json, "loads", unexpected_loads)
    with pytest.raises(codec.CanonicalEvidenceError, match="depth"):
        codec.decode_canonical_json_file(raw)


def test_scanner_stops_million_container_and_duplicate_key_amplification_before_loads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_loads(*args: object, **kwargs: object) -> object:
        raise AssertionError("json.loads must not run after scanner rejection")

    monkeypatch.setattr(json, "loads", unexpected_loads)
    container_count = 1_300_000
    containers = b'{"items":[' + b"{}," * (container_count - 1) + b"{}]}\n"
    assert 3_800_000 < len(containers) < codec.MAX_CANONICAL_FILE_BYTES
    with pytest.raises(codec.CanonicalEvidenceError, match="node bound before decoding"):
        codec.decode_canonical_json_file(containers)

    duplicate_count = 300_000
    duplicate_keys = b"{" + b'"same":0,' * (duplicate_count - 1) + b'"same":0}\n'
    assert len(duplicate_keys) < codec.MAX_CANONICAL_FILE_BYTES
    with pytest.raises(codec.CanonicalEvidenceError, match="node bound before decoding"):
        codec.decode_canonical_json_file(duplicate_keys)


def test_scanner_rejects_overlong_decoded_string_before_json_loads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = b'{"value":"' + b"x" * (codec.MAX_JSON_STRING_LENGTH + 1) + b'"}\n'

    def unexpected_loads(*args: object, **kwargs: object) -> object:
        raise AssertionError("json.loads must not run after scanner rejection")

    monkeypatch.setattr(json, "loads", unexpected_loads)
    with pytest.raises(codec.CanonicalEvidenceError, match="decoded length bound"):
        codec.decode_canonical_json_file(raw)


def test_encoder_rejects_aliased_containers_and_cycles() -> None:
    shared: dict[str, object] = {"value": 1}
    with pytest.raises(codec.CanonicalEvidenceError, match="alias or cycle"):
        codec.canonical_json_bytes({"first": shared, "second": shared})

    cyclic: list[object] = []
    cyclic.append(cyclic)
    with pytest.raises(codec.CanonicalEvidenceError, match="alias or cycle"):
        codec.canonical_json_bytes({"cycle": cyclic})


def test_encoder_rejects_depth_and_node_overflow() -> None:
    root: list[object] = []
    cursor = root
    for _ in range(codec.MAX_JSON_DEPTH + 1):
        child: list[object] = []
        cursor.append(child)
        cursor = child
    with pytest.raises(codec.CanonicalEvidenceError, match="depth"):
        codec.canonical_json_bytes(root)

    too_many_nodes = [None] * codec.MAX_JSON_NODES
    with pytest.raises(codec.CanonicalEvidenceError, match="node"):
        codec.canonical_json_bytes(too_many_nodes)


def test_decoder_independently_rejects_depth_and_node_overflow() -> None:
    too_deep = (
        b'{"value":'
        + b"[" * (codec.MAX_JSON_DEPTH + 1)
        + b"null"
        + b"]" * (codec.MAX_JSON_DEPTH + 1)
        + b"}\n"
    )
    with pytest.raises(codec.CanonicalEvidenceError, match="depth"):
        codec.decode_canonical_json_file(too_deep)

    too_many_nodes = b'{"value":[' + b",".join([b"null"] * codec.MAX_JSON_NODES) + b"]}\n"
    with pytest.raises(codec.CanonicalEvidenceError, match="node"):
        codec.decode_canonical_json_file(too_many_nodes)


def test_scanner_accepts_exact_node_and_depth_boundaries() -> None:
    scalar_count = codec.MAX_JSON_NODES - 3
    exact_nodes = b'{"value":[' + b",".join([b"null"] * scalar_count) + b"]}\n"
    decoded_nodes = codec.decode_canonical_json_file(exact_nodes)
    assert len(decoded_nodes["value"]) == scalar_count

    exact_depth = b'{"value":' + b"[" * codec.MAX_JSON_DEPTH + b"]" * codec.MAX_JSON_DEPTH + b"}\n"
    decoded_depth = codec.decode_canonical_json_file(exact_depth)
    cursor: object = decoded_depth["value"]
    observed_depth = 0
    while type(cursor) is list:
        observed_depth += 1
        exact_cursor = cursor
        if not exact_cursor:
            break
        cursor = exact_cursor[0]
    assert observed_depth == codec.MAX_JSON_DEPTH


def test_scanner_and_decoder_agree_on_deterministic_canonical_fuzz_corpus() -> None:
    generator = random.Random(20_260_804)

    def generated_value(depth: int) -> object:
        leaf_values: tuple[object, ...] = (
            None,
            False,
            True,
            0,
            -codec.MAX_EXACT_INTEGER,
            codec.MAX_EXACT_INTEGER,
            "",
            'quote"backslash\\ space',
        )
        if depth >= 4 or generator.randrange(3) == 0:
            return leaf_values[generator.randrange(len(leaf_values))]
        if generator.randrange(2) == 0:
            return [generated_value(depth + 1) for _ in range(generator.randrange(5))]
        return {
            f'key_{depth}_{index}_"\\': generated_value(depth + 1)
            for index in range(generator.randrange(5))
        }

    for corpus_index in range(200):
        payload = {
            "corpus_index": corpus_index,
            "value": generated_value(0),
        }
        raw = codec.canonical_json_bytes(payload)
        assert codec.decode_canonical_json_file(raw) == payload


@pytest.mark.parametrize(
    "value",
    [
        {"float": 1.0},
        {"large": codec.MAX_EXACT_INTEGER + 1},
        {"small": -codec.MAX_EXACT_INTEGER - 1},
        {"tuple": (1, 2)},
        {_StrSubclass("key"): 1},
        {"value": _IntSubclass(1)},
        {"unicode": "caf\N{LATIN SMALL LETTER E WITH ACUTE}"},
        {"control": "line\nfeed"},
    ],
)
def test_encoder_rejects_nonplain_or_out_of_domain_values(value: object) -> None:
    with pytest.raises(codec.CanonicalEvidenceError):
        codec.canonical_json_bytes(value)


@pytest.mark.parametrize("selector", [0, 1, None, "true"])
def test_final_lf_selector_rejects_bool_impersonators(selector: object) -> None:
    with pytest.raises(codec.CanonicalEvidenceError):
        codec.canonical_json_bytes({}, final_lf=selector)  # type: ignore[arg-type]


def test_file_and_body_validation_uses_two_independent_distinct_identities() -> None:
    body = {"schema_version": "alberta.test.document.v1", "value": 7}
    digest_field = "document_body_sha256"
    raw = codec.canonical_file_bytes(body, body_digest_field=digest_field)
    file_sha256 = hashlib.sha256(raw).hexdigest()
    body_sha256 = codec.canonical_body_sha256(body)
    assert file_sha256 != body_sha256
    document = codec.decode_canonical_json_file(raw)
    assert document[digest_field] == body_sha256
    assert codec.validate_file_sha256(raw, file_sha256) == file_sha256
    assert (
        codec.validate_body_sha256(
            document,
            body_digest_field=digest_field,
            expected_body_sha256=body_sha256,
        )
        == body
    )
    assert (
        codec.validate_canonical_file(
            raw,
            expected_file_sha256=file_sha256,
            expected_body_sha256=body_sha256,
            body_digest_field=digest_field,
        )
        == body
    )


def test_file_and_body_validation_rejects_each_tamper_frontier() -> None:
    body = {"schema_version": "alberta.test.document.v1", "value": 7}
    digest_field = "document_body_sha256"
    raw = codec.canonical_file_bytes(body, body_digest_field=digest_field)
    file_sha256 = hashlib.sha256(raw).hexdigest()
    body_sha256 = codec.canonical_body_sha256(body)

    with pytest.raises(codec.CanonicalEvidenceError, match="FILE"):
        codec.validate_canonical_file(
            raw,
            expected_file_sha256=_sha("wrong file"),
            expected_body_sha256=body_sha256,
            body_digest_field=digest_field,
        )
    with pytest.raises(codec.CanonicalEvidenceError, match="BODY"):
        codec.validate_canonical_file(
            raw,
            expected_file_sha256=file_sha256,
            expected_body_sha256=_sha("wrong body"),
            body_digest_field=digest_field,
        )
    with pytest.raises(codec.CanonicalEvidenceError, match="nonzero"):
        codec.validate_canonical_file(
            raw,
            expected_file_sha256="0" * 64,
            expected_body_sha256=body_sha256,
            body_digest_field=digest_field,
        )
    with pytest.raises(codec.CanonicalEvidenceError, match="aliased"):
        codec.validate_canonical_file(
            raw,
            expected_file_sha256=file_sha256,
            expected_body_sha256=file_sha256,
            body_digest_field=digest_field,
        )

    damaged_document = codec.decode_canonical_json_file(raw)
    damaged_document[digest_field] = _sha("damaged embedded body")
    damaged_raw = codec.canonical_json_bytes(damaged_document)
    with pytest.raises(codec.CanonicalEvidenceError, match="embedded BODY"):
        codec.validate_canonical_file(
            damaged_raw,
            expected_file_sha256=hashlib.sha256(damaged_raw).hexdigest(),
            expected_body_sha256=body_sha256,
            body_digest_field=digest_field,
        )

    changed_body = dict(body)
    changed_body["value"] = 8
    changed_raw = codec.canonical_file_bytes(changed_body, body_digest_field=digest_field)
    with pytest.raises(codec.CanonicalEvidenceError, match="expected BODY"):
        codec.validate_canonical_file(
            changed_raw,
            expected_file_sha256=hashlib.sha256(changed_raw).hexdigest(),
            expected_body_sha256=body_sha256,
            body_digest_field=digest_field,
        )


def test_file_helpers_reject_preexisting_digest_missing_digest_and_nonplain_body() -> None:
    digest_field = "document_body_sha256"
    with pytest.raises(codec.CanonicalEvidenceError, match="already contains"):
        codec.canonical_file_bytes({digest_field: _sha("body")}, body_digest_field=digest_field)
    with pytest.raises(codec.CanonicalEvidenceError, match="exact object"):
        codec.canonical_file_bytes(_DictSubclass(), body_digest_field=digest_field)
    with pytest.raises(codec.CanonicalEvidenceError, match="lacks"):
        codec.validate_body_sha256(
            {"schema_version": "alberta.test.v1"},
            body_digest_field=digest_field,
            expected_body_sha256=_sha("body"),
        )


def test_artifact_reference_is_frozen_exact_and_round_trips() -> None:
    artifact = _artifact()
    assert codec.artifact_ref_v1_from_dict(artifact.to_dict()) == artifact
    with pytest.raises(dataclasses.FrozenInstanceError):
        artifact.file_sha256 = _sha("replacement")  # type: ignore[misc]


@pytest.mark.parametrize(
    "record_type",
    [
        codec.ArtifactRefV1,
        codec.ProducerRefV1,
        codec.CaseSubjectV1,
        codec.RawByteCaptureV1,
    ],
)
def test_nested_record_classes_are_runtime_final(record_type: type[object]) -> None:
    with pytest.raises(TypeError, match="runtime-final"):
        type("SubclassBypass", (record_type,), {})


def test_every_nested_to_dict_method_preserves_exact_parser_parity() -> None:
    producer = _producers()[0]
    subject = codec.CaseSubjectV1.for_ordinal(0)
    capture = codec.RawByteCaptureV1.from_bytes(b"parser parity")
    records_and_parsers: tuple[tuple[Any, Any], ...] = (
        (_artifact(), codec.artifact_ref_v1_from_dict),
        (producer, codec.producer_ref_v1_from_dict),
        (subject, codec.case_subject_v1_from_dict),
        (capture, codec.raw_byte_capture_v1_from_dict),
    )
    for record, parser in records_and_parsers:
        assert parser(record.to_dict()) == record


def test_every_artifact_field_is_revalidated_from_current_state_before_serialization() -> None:
    baseline = _artifact()
    for field, value in (
        ("schema_version", "bad schema with spaces"),
        ("schema_version", _EqualityProxy(baseline.schema_version)),
        ("file_sha256", "0" * 64),
        ("file_sha256", _EqualityProxy(baseline.file_sha256)),
        ("body_sha256", baseline.file_sha256),
        ("body_sha256", _EqualityProxy(baseline.body_sha256)),
    ):
        artifact = _artifact()
        object.__setattr__(artifact, field, value)
        with pytest.raises(codec.CanonicalEvidenceError):
            artifact.to_dict()


def test_every_producer_field_is_revalidated_from_current_state() -> None:
    baseline = _producers()[0]
    for field, value in (
        ("role", _StrSubclass(baseline.role)),
        ("role", _EqualityProxy(baseline.role)),
        ("descriptor_schema_version", "alberta.test.wrong.v1"),
        (
            "descriptor_schema_version",
            _EqualityProxy(baseline.descriptor_schema_version),
        ),
        ("descriptor_file_sha256", "0" * 64),
        (
            "descriptor_file_sha256",
            _EqualityProxy(baseline.descriptor_file_sha256),
        ),
        ("descriptor_body_sha256", baseline.descriptor_file_sha256),
        (
            "descriptor_body_sha256",
            _EqualityProxy(baseline.descriptor_body_sha256),
        ),
        ("source_sha256", baseline.descriptor_file_sha256),
        ("source_sha256", _EqualityProxy(baseline.source_sha256)),
    ):
        producers = _producers()
        object.__setattr__(producers[0], field, value)
        with pytest.raises(codec.CanonicalEvidenceError):
            producers[0].to_dict()
        with pytest.raises(codec.CanonicalEvidenceError):
            codec.validate_producer_refs_v1(producers)


def test_every_case_subject_field_is_revalidated_before_serialization() -> None:
    baseline = codec.CaseSubjectV1.for_ordinal(0)
    for field, value in (
        ("case_ordinal", True),
        ("case_ordinal", _IntSubclass(baseline.case_ordinal)),
        ("case_ordinal", _EqualityProxy(baseline.case_ordinal)),
        ("candidate_id", _EqualityProxy(baseline.candidate_id)),
        ("candidate_family", _EqualityProxy(baseline.candidate_family)),
        ("qualification_case_id", _EqualityProxy(baseline.qualification_case_id)),
    ):
        subject = codec.CaseSubjectV1.for_ordinal(0)
        object.__setattr__(subject, field, value)
        with pytest.raises(codec.CanonicalEvidenceError):
            subject.to_dict()


def test_every_raw_capture_field_is_revalidated_before_serialization_and_decode() -> None:
    baseline = codec.RawByteCaptureV1.from_bytes(b"current bytes")
    for field, value in (
        ("raw_bytes_base64", base64.b64encode(b"changed bytes").decode("ascii")),
        ("raw_bytes_base64", _EqualityProxy(baseline.raw_bytes_base64)),
        ("raw_size_bytes", baseline.raw_size_bytes + 1),
        ("raw_size_bytes", _IntSubclass(baseline.raw_size_bytes)),
        ("raw_size_bytes", _EqualityProxy(baseline.raw_size_bytes)),
        ("raw_sha256", _sha("wrong current bytes")),
        ("raw_sha256", _EqualityProxy(baseline.raw_sha256)),
        ("schema_version", "alberta.test.wrong.v1"),
        ("schema_version", _EqualityProxy(baseline.schema_version)),
    ):
        capture = codec.RawByteCaptureV1.from_bytes(b"current bytes")
        object.__setattr__(capture, field, value)
        with pytest.raises(codec.CanonicalEvidenceError):
            capture.to_dict()
        with pytest.raises(codec.CanonicalEvidenceError):
            capture.decoded_bytes()


def test_raw_capture_decoding_normalizes_current_malformed_base64() -> None:
    capture = codec.RawByteCaptureV1.from_bytes(b"current bytes")
    object.__setattr__(capture, "raw_bytes_base64", "%%%")
    with pytest.raises(codec.CanonicalEvidenceError) as caught:
        capture.decoded_bytes()
    assert isinstance(caught.value.__cause__, binascii.Error)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", "bad schema with spaces"),
        ("schema_version", _StrSubclass("alberta.test.artifact.v1")),
        ("file_sha256", "0" * 64),
        ("file_sha256", _StrSubclass(_sha("artifact file"))),
        ("file_sha256", _EqualityProxy(_sha("artifact file"))),
        ("body_sha256", "not-a-hash"),
    ],
)
def test_artifact_reference_rejects_schema_and_digest_adversaries(
    field: str,
    value: object,
) -> None:
    kwargs: dict[str, object] = {
        "schema_version": "alberta.test.artifact.v1",
        "file_sha256": _sha("artifact file"),
        "body_sha256": _sha("artifact body"),
    }
    kwargs[field] = value
    with pytest.raises(codec.CanonicalEvidenceError):
        codec.ArtifactRefV1(**kwargs)  # type: ignore[arg-type]


def test_artifact_reference_rejects_file_body_alias_and_parser_key_drift() -> None:
    digest = _sha("same")
    with pytest.raises(codec.CanonicalEvidenceError, match="aliased"):
        codec.ArtifactRefV1("alberta.test.artifact.v1", digest, digest)
    value = _artifact().to_dict()
    value["extra"] = "field"
    with pytest.raises(codec.CanonicalEvidenceError, match="keys differ"):
        codec.artifact_ref_v1_from_dict(value)
    with pytest.raises(codec.CanonicalEvidenceError, match="keys differ"):
        codec.artifact_ref_v1_from_dict(_DictSubclass(_artifact().to_dict()))


def test_six_producer_roles_and_schemas_are_frozen_exact_and_round_trip() -> None:
    assert tuple(codec.PRODUCER_ROLES) == tuple(EXPECTED_PRODUCER_SCHEMAS)
    assert dict(codec.PRODUCER_DESCRIPTOR_SCHEMA_BY_ROLE) == EXPECTED_PRODUCER_SCHEMAS
    producers = _producers()
    assert codec.validate_producer_refs_v1(producers) == producers
    assert codec.producer_refs_v1_from_json([item.to_dict() for item in producers]) == producers
    for producer in producers:
        assert codec.producer_ref_v1_from_dict(producer.to_dict()) == producer


@pytest.mark.parametrize("index", range(6))
def test_each_producer_role_rejects_the_wrong_descriptor_schema(index: int) -> None:
    producer = _producers()[index]
    with pytest.raises(codec.CanonicalEvidenceError, match="schema differs"):
        dataclasses.replace(producer, descriptor_schema_version="alberta.test.wrong.v1")


def test_producer_tuple_rejects_order_type_count_and_cross_role_hash_aliases() -> None:
    producers = _producers()
    with pytest.raises(codec.CanonicalEvidenceError, match="exact ordered"):
        codec.validate_producer_refs_v1(list(producers))
    with pytest.raises(codec.CanonicalEvidenceError, match="exact ordered"):
        codec.validate_producer_refs_v1(producers[:-1])
    with pytest.raises(codec.CanonicalEvidenceError, match="role order"):
        codec.validate_producer_refs_v1((producers[1], producers[0], *producers[2:]))
    aliased_source = dataclasses.replace(
        producers[1],
        source_sha256=producers[0].source_sha256,
    )
    with pytest.raises(codec.CanonicalEvidenceError, match="aliased"):
        codec.validate_producer_refs_v1((producers[0], aliased_source, *producers[2:]))
    aliased_cross_kind = dataclasses.replace(
        producers[1],
        source_sha256=producers[0].descriptor_body_sha256,
    )
    with pytest.raises(codec.CanonicalEvidenceError, match="aliased"):
        codec.validate_producer_refs_v1((producers[0], aliased_cross_kind, *producers[2:]))


def test_producer_reference_rejects_exact_type_and_parser_adversaries() -> None:
    producer = _producers()[0]
    with pytest.raises(codec.CanonicalEvidenceError):
        dataclasses.replace(producer, role=_StrSubclass(producer.role))  # type: ignore[arg-type]
    with pytest.raises(codec.CanonicalEvidenceError):
        dataclasses.replace(
            producer,
            descriptor_file_sha256=_EqualityProxy(producer.descriptor_file_sha256),  # type: ignore[arg-type]
        )
    changed = producer.to_dict()
    changed["extra"] = "field"
    with pytest.raises(codec.CanonicalEvidenceError, match="keys differ"):
        codec.producer_ref_v1_from_dict(changed)


def test_matched_v3_candidate_order_family_and_case_projection_are_exact() -> None:
    assert codec.MATCHED_V3_CANDIDATE_IDS == EXPECTED_CANDIDATE_ORDER
    assert codec.MATCHED_V3_CANDIDATE_IDS is plan_v3.MATCHED_V3_QUALIFICATION_V3_CANDIDATE_IDS
    assert codec.MATCHED_V3_LOCAL_CANDIDATE_IDS is plan_v3.MATCHED_V3_LOCAL_CANDIDATE_IDS
    assert codec.MATCHED_V3_EXTERNAL_CANDIDATE_IDS is plan_v3.MATCHED_V3_EXTERNAL_CANDIDATE_IDS
    assert codec.MATCHED_V3_ADAPTER_CANDIDATE_IDS is plan_v3.MATCHED_V3_ADAPTER_CANDIDATE_IDS
    for ordinal, candidate_id in enumerate(EXPECTED_CANDIDATE_ORDER):
        subject = codec.CaseSubjectV1.for_ordinal(ordinal)
        expected_family = (
            "adapter"
            if candidate_id.startswith("adapted_")
            else (
                "external"
                if candidate_id.startswith(("external_", "isolated_", "random_", "search_"))
                else "local"
            )
        )
        assert subject == codec.CaseSubjectV1(
            case_ordinal=ordinal,
            candidate_id=candidate_id,
            candidate_family=expected_family,  # type: ignore[arg-type]
            qualification_case_id=f"qualification_{ordinal:02d}_{candidate_id}",
        )
        assert codec.candidate_family(candidate_id) == expected_family
        assert codec.case_subject_v1_from_dict(subject.to_dict()) == subject


@pytest.mark.parametrize("ordinal", [True, False, 0.0, _IntSubclass(0), -1, 28])
def test_case_subject_rejects_nonexact_or_out_of_range_ordinals(ordinal: object) -> None:
    with pytest.raises(codec.CanonicalEvidenceError):
        codec.CaseSubjectV1(
            case_ordinal=ordinal,  # type: ignore[arg-type]
            candidate_id=EXPECTED_CANDIDATE_ORDER[0],
            candidate_family="local",
            qualification_case_id=f"qualification_00_{EXPECTED_CANDIDATE_ORDER[0]}",
        )


def test_case_subject_rejects_candidate_family_case_id_and_text_subclass_drift() -> None:
    subject = codec.CaseSubjectV1.for_ordinal(0)
    for changes in (
        {"candidate_id": EXPECTED_CANDIDATE_ORDER[1]},
        {"candidate_family": "external"},
        {"qualification_case_id": "qualification_00_wrong"},
    ):
        with pytest.raises(codec.CanonicalEvidenceError):
            dataclasses.replace(subject, **changes)  # type: ignore[arg-type]
    with pytest.raises(codec.CanonicalEvidenceError):
        codec.CaseSubjectV1(
            case_ordinal=0,
            candidate_id=_StrSubclass(subject.candidate_id),
            candidate_family="local",
            qualification_case_id=subject.qualification_case_id,
        )
    with pytest.raises(codec.CanonicalEvidenceError):
        codec.candidate_family("unknown_candidate")


def test_raw_byte_capture_round_trips_verified_binary_bytes_and_json() -> None:
    raw = b"\x00binary\xff\n"
    capture = codec.RawByteCaptureV1.from_bytes(raw)
    assert capture.raw_bytes_base64 == base64.b64encode(raw).decode("ascii")
    assert capture.raw_size_bytes == len(raw)
    assert capture.raw_sha256 == hashlib.sha256(raw).hexdigest()
    assert capture.decoded_bytes() == raw
    assert codec.raw_byte_capture_v1_from_dict(capture.to_dict()) == capture
    assert codec.decode_canonical_json_file(codec.canonical_json_bytes(capture.to_dict())) == (
        capture.to_dict()
    )


def test_raw_byte_capture_allows_verified_empty_absence_proof_stream() -> None:
    capture = codec.RawByteCaptureV1.from_bytes(b"")
    assert capture.raw_bytes_base64 == ""
    assert capture.raw_size_bytes == 0
    assert capture.raw_sha256 == hashlib.sha256(b"").hexdigest()
    assert capture.decoded_bytes() == b""
    assert codec.raw_byte_capture_v1_from_dict(capture.to_dict()) == capture


@pytest.mark.parametrize("raw", [bytearray(b"x"), "x"])
def test_raw_byte_capture_factory_rejects_nonexact_bytes(raw: object) -> None:
    with pytest.raises(codec.CanonicalEvidenceError):
        codec.RawByteCaptureV1.from_bytes(raw)  # type: ignore[arg-type]


def test_raw_byte_capture_factory_rejects_oversize_bytes() -> None:
    with pytest.raises(codec.CanonicalEvidenceError, match="bound"):
        codec.RawByteCaptureV1.from_bytes(b"x" * (codec.MAX_RAW_CAPTURE_BYTES + 1))


def test_raw_byte_capture_accepts_exact_maximum_size() -> None:
    raw = b"x" * codec.MAX_RAW_CAPTURE_BYTES
    capture = codec.RawByteCaptureV1.from_bytes(raw)
    assert capture.raw_size_bytes == codec.MAX_RAW_CAPTURE_BYTES
    assert capture.decoded_bytes() == raw


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("raw_bytes_base64", "Zh=="),
        ("raw_bytes_base64", "Zg==\n"),
        ("raw_bytes_base64", "Zg"),
        ("raw_bytes_base64", _StrSubclass("Zg==")),
        ("raw_size_bytes", 2),
        ("raw_size_bytes", True),
        ("raw_size_bytes", 1.0),
        ("raw_sha256", "0" * 64),
        ("raw_sha256", _sha("wrong raw")),
        ("schema_version", "alberta.test.wrong.v1"),
        ("schema_version", _StrSubclass(codec.RAW_BYTE_CAPTURE_V1_SCHEMA_VERSION)),
    ],
)
def test_raw_byte_capture_rejects_base64_size_hash_and_schema_adversaries(
    field: str,
    value: object,
) -> None:
    kwargs: dict[str, object] = dict(codec.RawByteCaptureV1.from_bytes(b"f").to_dict())
    kwargs[field] = value
    with pytest.raises(codec.CanonicalEvidenceError):
        codec.RawByteCaptureV1(**kwargs)  # type: ignore[arg-type]


def test_raw_byte_capture_parser_rejects_extra_or_nonexact_mapping() -> None:
    value = codec.RawByteCaptureV1.from_bytes(b"raw").to_dict()
    value["extra"] = "field"
    with pytest.raises(codec.CanonicalEvidenceError, match="keys differ"):
        codec.raw_byte_capture_v1_from_dict(value)
    with pytest.raises(codec.CanonicalEvidenceError, match="keys differ"):
        codec.raw_byte_capture_v1_from_dict(
            _DictSubclass(codec.RawByteCaptureV1.from_bytes(b"raw").to_dict())
        )


def test_public_surface_has_no_schema_dispatch_or_operational_entrypoint() -> None:
    assert type(codec.__all__) is list
    assert codec.__all__ == EXPECTED_ALL
    assert all(type(name) is str and hasattr(codec, name) for name in codec.__all__)
    assert len(codec.__all__) == len(set(codec.__all__))
    assert "MAX_JSON_STRING_LENGTH" in codec.__all__
    assert not any(
        name in codec.__all__
        for name in (
            "dispatch_schema",
            "execute",
            "issue",
            "launch",
            "load_path",
            "read_file",
            "write_file",
        )
    )
