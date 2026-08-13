"""Adversarial tests for the pure matched-v3 write-seal provider protocol."""

from __future__ import annotations

import ast
import hashlib
import re
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest

from alberta_framework.benchmarks import (
    forager_matched_v3_storage_write_seal_protocol_v2 as seal,
)
from alberta_framework.benchmarks._forager_matched_v3_canonical_evidence import (
    ArtifactRefV1,
    CanonicalEvidenceError,
    CaseSubjectV1,
    ProducerRefV1,
    RawByteCaptureV1,
    canonical_body_sha256,
    canonical_file_bytes,
    canonical_json_bytes,
    decode_canonical_json_file,
)

SOURCE = (
    Path(__file__).parents[1]
    / "alberta_framework"
    / "benchmarks"
    / "forager_matched_v3_storage_write_seal_protocol_v2.py"
)
ERROR = seal.ForagerMatchedV3StorageWriteSealProtocolV2Error


class _EqualitySpoof:
    def __eq__(self, other: object) -> bool:
        return True

    def __ne__(self, other: object) -> bool:
        return False


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _artifact(schema: str, label: str) -> ArtifactRefV1:
    return ArtifactRefV1(
        schema_version=schema,
        file_sha256=_sha(f"{label}:file"),
        body_sha256=_sha(f"{label}:body"),
    )


def _replace_unchecked(value: Any, /, **changes: Any) -> Any:
    """Exercise invalid runtime values that intentionally violate static types."""

    return replace(value, **changes)


def _artifact_with_alias(
    artifact: ArtifactRefV1,
    schema: str,
    label: str,
    *,
    kind: str,
) -> ArtifactRefV1:
    return ArtifactRefV1(
        schema_version=schema,
        file_sha256=(artifact.file_sha256 if kind == "file" else _sha(f"{label}:file")),
        body_sha256=(artifact.body_sha256 if kind == "body" else _sha(f"{label}:body")),
    )


def _producer(label: str = "write-seal") -> ProducerRefV1:
    return ProducerRefV1(
        role="write_seal_producer",
        descriptor_schema_version=seal.WRITE_SEAL_PRODUCER_DESCRIPTOR_SCHEMA_VERSION,
        descriptor_file_sha256=_sha(f"{label}:descriptor:file"),
        descriptor_body_sha256=_sha(f"{label}:descriptor:body"),
        source_sha256=_sha(f"{label}:source"),
    )


def _runtime(label: str = "runtime") -> seal.WriteSealRuntimeEnvelopeV1:
    return seal.WriteSealRuntimeEnvelopeV1(
        campaign_id="campaign_2026q3",
        case_subject=CaseSubjectV1.for_ordinal(0),
        runtime_intent=_artifact(seal.STORAGE_BOUNDARY_RUNTIME_INTENT_SCHEMA_VERSION, label),
        host_go=_artifact(seal.HOST_GO_V3_SCHEMA_VERSION, f"{label}:go"),
        image_id=f"sha256:{_sha(f'{label}:image')}",
        container_name="alberta-qualified-case-00",
        container_id_commitment_sha256=_sha(f"{label}:container"),
        outer_cgroup_identity_sha256=_sha(f"{label}:cgroup"),
    )


def _raw_inputs(**changes: Any) -> dict[str, Any]:
    facts: dict[str, Any] = {
        "runtime": _runtime(),
        "publication_reload_validation": _artifact(
            seal.PUBLICATION_RELOAD_VALIDATION_SCHEMA_VERSION,
            "reload",
        ),
        "terminal_relay_preseal_attestation": _artifact(
            seal.TERMINAL_RELAY_PRESEAL_ATTESTATION_SCHEMA_VERSION,
            "relay",
        ),
        "nonstorage_channel_preseal_attestation": _artifact(
            seal.NONSTORAGE_CHANNEL_PRESEAL_ATTESTATION_SCHEMA_VERSION,
            "channel",
        ),
        "tmpfs_hard_limit_mount_mutation_closure": _artifact(
            seal.TMPFS_HARD_LIMIT_MOUNT_MUTATION_CLOSURE_SCHEMA_VERSION,
            "tmpfs-closure",
        ),
        "writable_fd_lifetime_inventory": _artifact(
            seal.WRITABLE_FD_LIFETIME_INVENTORY_SCHEMA_VERSION,
            "fd-inventory",
        ),
        "docker_api_operation_journal": _artifact(
            seal.DOCKER_API_OPERATION_JOURNAL_SCHEMA_VERSION,
            "docker-journal",
        ),
        "rootfs_upperdir_interval_delta": _artifact(
            seal.ROOTFS_UPPERDIR_INTERVAL_DELTA_SCHEMA_VERSION,
            "upperdir-delta",
        ),
        "docker_volume_inventory_delta": _artifact(
            seal.DOCKER_VOLUME_INVENTORY_DELTA_SCHEMA_VERSION,
            "volume-delta",
        ),
        "producer": _producer(),
        "precommit_monotonic_ns": 900,
    }
    facts.update(changes)
    return facts


def _expected_observation_bytes(facts: dict[str, Any]) -> bytes:
    return seal.canonical_raw_write_seal_capture_bytes(**facts)


def _raw(**changes: Any) -> seal.RawWriteSealV1:
    facts = _raw_inputs(**changes)
    return seal.build_raw_write_seal_v1(
        observation_bytes=_expected_observation_bytes(facts),
        **facts,
    )


def _sealed(raw: seal.RawWriteSealV1, **changes: Any) -> seal.WriteQuiescenceSealV1:
    facts: dict[str, Any] = {
        "runtime": raw.runtime,
        "publication_reload_validation": raw.publication_reload_validation,
        "terminal_relay_preseal_attestation": raw.terminal_relay_preseal_attestation,
        "nonstorage_channel_preseal_attestation": (raw.nonstorage_channel_preseal_attestation),
        "raw_write_seal": seal.raw_write_seal_identity_v1(raw),
        "producer": raw.producer,
        "seal_monotonic_ns": raw.precommit_monotonic_ns + 1,
    }
    facts.update(changes)
    return seal.WriteQuiescenceSealV1(**facts)


def _encoded_body(
    body: dict[str, Any],
    digest_field: str,
) -> tuple[bytes, str, str]:
    raw = canonical_file_bytes(body, body_digest_field=digest_field)
    return raw, hashlib.sha256(raw).hexdigest(), canonical_body_sha256(body)


def _parse_mutated_raw(body: dict[str, Any]) -> seal.RawWriteSealV1:
    raw, file_pin, body_pin = _encoded_body(body, seal.RAW_WRITE_SEAL_BODY_SHA256_FIELD)
    return seal.parse_raw_write_seal_v1(
        raw,
        expected_file_sha256=file_pin,
        expected_body_sha256=body_pin,
    )


def _parse_mutated_seal(body: dict[str, Any]) -> seal.WriteQuiescenceSealV1:
    raw, file_pin, body_pin = _encoded_body(
        body,
        seal.IRREVERSIBLE_WRITE_SEAL_BODY_SHA256_FIELD,
    )
    return seal.parse_write_quiescence_seal_v1(
        raw,
        expected_file_sha256=file_pin,
        expected_body_sha256=body_pin,
    )


def _parser_fixture(
    kind: str,
) -> tuple[bytes, str, dict[str, Any], Callable[..., object]]:
    raw = _raw()
    parser: Callable[..., object]
    if kind == "raw":
        value: Any = raw
        encoded = seal.canonical_raw_write_seal_v1_file_bytes(raw)
        body_bytes = seal.canonical_raw_write_seal_v1_body_bytes(raw)
        parser = seal.parse_raw_write_seal_v1
    elif kind == "seal":
        value = _sealed(raw)
        encoded = seal.canonical_write_quiescence_seal_v1_file_bytes(value)
        body_bytes = seal.canonical_write_quiescence_seal_v1_body_bytes(value)
        parser = seal.parse_write_quiescence_seal_v1
    else:
        value = seal.WriteSealProducerDescriptorV2()
        encoded = seal.canonical_write_seal_producer_descriptor_v2_file_bytes(value)
        body_bytes = seal.canonical_write_seal_producer_descriptor_v2_body_bytes(value)
        parser = seal.parse_write_seal_producer_descriptor_v2
    return encoded, hashlib.sha256(body_bytes).hexdigest(), value.to_body_dict(), parser


def _replace_first_integer_with_float(match: re.Match[bytes]) -> bytes:
    return match.group(1) + match.group(2) + b".0"


def test_write_seal_dual_pin_roundtrip_and_chain() -> None:
    raw = _raw()
    committed = _sealed(raw)
    raw_bytes = seal.canonical_raw_write_seal_v1_file_bytes(raw)
    raw_body = seal.canonical_raw_write_seal_v1_body_bytes(raw)
    parsed_raw = seal.parse_raw_write_seal_v1(
        raw_bytes,
        expected_file_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        expected_body_sha256=hashlib.sha256(raw_body).hexdigest(),
    )
    seal_bytes = seal.canonical_write_quiescence_seal_v1_file_bytes(committed)
    seal_body = seal.canonical_write_quiescence_seal_v1_body_bytes(committed)
    parsed_seal = seal.parse_write_quiescence_seal_v1(
        seal_bytes,
        expected_file_sha256=hashlib.sha256(seal_bytes).hexdigest(),
        expected_body_sha256=hashlib.sha256(seal_body).hexdigest(),
    )
    assert parsed_raw == raw
    assert parsed_seal == committed
    seal.validate_write_seal_chain_v2(raw, committed)
    projection = seal.raw_write_seal_projection_v1(raw, committed)
    assert projection.raw_artifact == seal.raw_write_seal_identity_v1(raw)
    assert projection.first_class_artifact == seal.write_quiescence_seal_identity_v1(committed)


def test_first_class_bodies_use_the_exact_flat_run_projection() -> None:
    raw = _raw()
    committed = _sealed(raw)
    expected_runtime = raw.runtime.to_dict()
    raw_body = raw.to_body_dict()
    seal_body = committed.to_body_dict()
    for body in (raw_body, seal_body):
        assert "runtime" not in body
        assert {key: body[key] for key in expected_runtime} == expected_runtime
        assert body["producer"] == raw.producer.to_dict()
    runtime_keys = set(expected_runtime)
    assert set(raw_body) == runtime_keys | {
        "capture",
        "descendant_process_count",
        "docker_api_operation_journal",
        "docker_volume_inventory_delta",
        "later_writer_count",
        "measured_writable_fd_count",
        "measured_writable_namespace_holder_count",
        "mount_mutation_enabled",
        "nonstorage_channel_preseal_attestation",
        "precommit_monotonic_ns",
        "producer",
        "publication_reload_validation",
        "rootfs_upperdir_interval_delta",
        "schema_version",
        "status",
        "terminal_relay_preseal_attestation",
        "tmpfs_hard_limit_mount_mutation_closure",
        "worker_exit_observed",
        "writable_fd_lifetime_inventory",
        "write_quiescence_irreversible",
    }
    assert set(seal_body) == runtime_keys | {
        "architecture_kind",
        "channel_ready_before_seal",
        "container_writes_disabled",
        "descendant_writes_disabled",
        "later_allocation_possible",
        "later_copy_up_possible",
        "later_peak_increase_possible",
        "no_later_writer_exists",
        "nonstorage_channel_preseal_attestation",
        "producer",
        "publication_committed_before_seal",
        "publication_reload_validation",
        "raw_write_seal",
        "relay_ready_before_seal",
        "reload_validated_before_seal",
        "schema_version",
        "seal_monotonic_ns",
        "status",
        "teardown_can_increase_measured_usage",
        "teardown_deletion_only",
        "terminal_relay_preseal_attestation",
        "write_quiescence_irreversible",
    }
    assert raw_body["precommit_monotonic_ns"] == raw.precommit_monotonic_ns
    assert "precommit_monotonic_ns" not in seal_body


@pytest.mark.parametrize("kind", ["raw", "seal", "descriptor"])
def test_artifacts_require_independent_file_and_body_pins(kind: str) -> None:
    raw = _raw()
    committed = _sealed(raw)
    parser: Callable[..., object]
    expected_value: object
    if kind == "raw":
        expected_value = raw
        encoded = seal.canonical_raw_write_seal_v1_file_bytes(raw)
        body = seal.canonical_raw_write_seal_v1_body_bytes(raw)
        parser = seal.parse_raw_write_seal_v1
    elif kind == "seal":
        expected_value = committed
        encoded = seal.canonical_write_quiescence_seal_v1_file_bytes(committed)
        body = seal.canonical_write_quiescence_seal_v1_body_bytes(committed)
        parser = seal.parse_write_quiescence_seal_v1
    else:
        descriptor = seal.WriteSealProducerDescriptorV2()
        expected_value = descriptor
        encoded = seal.canonical_write_seal_producer_descriptor_v2_file_bytes(descriptor)
        body = seal.canonical_write_seal_producer_descriptor_v2_body_bytes(descriptor)
        parser = seal.parse_write_seal_producer_descriptor_v2
    file_pin = hashlib.sha256(encoded).hexdigest()
    body_pin = hashlib.sha256(body).hexdigest()
    assert file_pin != body_pin
    assert (
        parser(
            encoded,
            expected_file_sha256=file_pin,
            expected_body_sha256=body_pin,
        )
        == expected_value
    )
    with pytest.raises(ValueError):
        parser(
            encoded,
            expected_file_sha256=_sha(f"wrong:{kind}:file"),
            expected_body_sha256=body_pin,
        )
    with pytest.raises(ValueError):
        parser(
            encoded,
            expected_file_sha256=body_pin,
            expected_body_sha256=file_pin,
        )
    with pytest.raises(ValueError):
        parser(
            encoded,
            expected_file_sha256=file_pin,
            expected_body_sha256=file_pin,
        )
    with pytest.raises(ValueError):
        parser(
            encoded,
            expected_file_sha256=file_pin,
            expected_body_sha256=_sha(f"wrong:{kind}:body"),
        )


@pytest.mark.parametrize("kind", ["raw", "seal", "descriptor"])
def test_every_parser_rejects_noncanonical_encoding_and_nonbytes(kind: str) -> None:
    encoded, body_pin, body, parser = _parser_fixture(kind)
    schema_member = canonical_json_bytes(
        {"schema_version": body["schema_version"]},
        final_lf=False,
    )[1:-1]
    duplicate_key = b"{" + schema_member + b"," + encoded[1:]
    float_value = re.sub(
        rb"(:)(-?[0-9]+)(?=[,}])",
        _replace_first_integer_with_float,
        encoded,
        count=1,
    )
    adversaries = (
        encoded[:-1],
        encoded + b"\n",
        b" " + encoded,
        encoded.replace(b"alberta", b"\\u0061lberta", 1),
        duplicate_key,
        float_value,
        encoded.replace(b"alberta", b"\xfflberta", 1),
    )
    assert all(adversary != encoded for adversary in adversaries)
    for adversary in adversaries:
        with pytest.raises(ERROR):
            parser(
                adversary,
                expected_file_sha256=hashlib.sha256(adversary).hexdigest(),
                expected_body_sha256=body_pin,
            )
    with pytest.raises(ERROR):
        parser(
            bytearray(encoded),
            expected_file_sha256=hashlib.sha256(encoded).hexdigest(),
            expected_body_sha256=body_pin,
        )


@pytest.mark.parametrize("kind", ["raw", "seal", "descriptor"])
def test_every_parser_enforces_file_depth_and_node_bounds(kind: str) -> None:
    _, body_pin, _, parser = _parser_fixture(kind)
    depth = seal.LIMITS["maximum_json_depth"] + 2
    depth_adversary = b'{"x":' + b"[" * depth + b"0" + b"]" * depth + b"}\n"
    node_adversary = (
        b'{"x":[' + b",".join([b"0"] * (seal.LIMITS["maximum_json_nodes"] + 1)) + b"]}\n"
    )
    oversize_adversary = b"x" * (seal.LIMITS["maximum_canonical_file_bytes"] + 1)
    for adversary in (depth_adversary, node_adversary, oversize_adversary):
        with pytest.raises(ERROR):
            parser(
                adversary,
                expected_file_sha256=hashlib.sha256(adversary).hexdigest(),
                expected_body_sha256=body_pin,
            )


def test_repository_producer_pins_are_explicitly_zero_without_a_minting_api() -> None:
    assert seal.DESCRIPTOR_SERIALIZED_SELF_PIN_SENTINEL == "0" * 64
    assert seal.PINNED_WRITE_SEAL_PRODUCER_DESCRIPTOR_FILE_SHA256 == "0" * 64
    assert seal.PINNED_WRITE_SEAL_PRODUCER_DESCRIPTOR_BODY_SHA256 == "0" * 64
    assert seal.PINNED_WRITE_SEAL_PRODUCER_SOURCE_SHA256 == "0" * 64
    assert not hasattr(seal, "write_seal_producer_ref_v2")
    assert "write_seal_producer_ref_v2" not in seal.__all__


def test_raw_capture_is_exact_bounded_schema_specific_replay() -> None:
    raw = _raw()
    captured = decode_canonical_json_file(raw.capture.decoded_bytes())
    assert set(captured) == {
        "capture_format",
        "descendant_process_count",
        "docker_api_operation_journal",
        "docker_volume_inventory_delta",
        "later_writer_count",
        "measured_writable_fd_count",
        "measured_writable_namespace_holder_count",
        "mount_mutation_enabled",
        "nonstorage_channel_preseal_attestation",
        "precommit_monotonic_ns",
        "producer",
        "publication_reload_validation",
        "rootfs_upperdir_interval_delta",
        "runtime",
        "terminal_relay_preseal_attestation",
        "tmpfs_hard_limit_mount_mutation_closure",
        "worker_exit_observed",
        "writable_fd_lifetime_inventory",
        "write_quiescence_irreversible",
    }
    assert captured["capture_format"] == seal.RAW_WRITE_SEAL_CAPTURE_FORMAT
    captured["descendant_process_count"] = 1
    mutated_capture = RawByteCaptureV1.from_bytes(canonical_json_bytes(captured))
    with pytest.raises(ERROR, match="capture differs"):
        replace(raw, capture=mutated_capture)
    with pytest.raises(ERROR, match="capture differs"):
        replace(raw, capture=RawByteCaptureV1.from_bytes(b"{}\n"))


def test_raw_builder_requires_and_preserves_actual_caller_observation_bytes() -> None:
    facts = _raw_inputs()
    observation_bytes = _expected_observation_bytes(facts)
    built = seal.build_raw_write_seal_v1(
        observation_bytes=observation_bytes,
        **facts,
    )
    assert built.capture.decoded_bytes() == observation_bytes
    untyped_builder: Any = seal.build_raw_write_seal_v1
    with pytest.raises(TypeError):
        untyped_builder(**facts)


def test_raw_builder_rejects_noncanonical_tampered_and_oversize_actual_bytes() -> None:
    facts = _raw_inputs()
    observation_bytes = _expected_observation_bytes(facts)
    decoded = decode_canonical_json_file(observation_bytes)
    decoded["later_writer_count"] = 1
    adversaries = (
        observation_bytes.removesuffix(b"\n"),
        canonical_json_bytes(decoded),
        b"x" * (seal.LIMITS["maximum_raw_capture_bytes"] + 1),
    )
    for adversary in adversaries:
        with pytest.raises(ValueError):
            seal.build_raw_write_seal_v1(
                observation_bytes=adversary,
                **facts,
            )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("worker_exit_observed", False),
        ("descendant_process_count", 1),
        ("measured_writable_namespace_holder_count", 1),
        ("measured_writable_fd_count", 1),
        ("later_writer_count", 1),
        ("mount_mutation_enabled", True),
        ("write_quiescence_irreversible", False),
        ("precommit_monotonic_ns", True),
    ],
)
def test_every_raw_quiescence_fact_is_frozen(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        _replace_unchecked(_raw(), **{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("architecture_kind", "irreversible_seccomp_fd_closure"),
        ("publication_committed_before_seal", False),
        ("reload_validated_before_seal", False),
        ("relay_ready_before_seal", False),
        ("channel_ready_before_seal", False),
        ("container_writes_disabled", False),
        ("descendant_writes_disabled", False),
        ("later_allocation_possible", True),
        ("later_copy_up_possible", True),
        ("later_peak_increase_possible", True),
        ("no_later_writer_exists", False),
        ("teardown_deletion_only", False),
        ("teardown_can_increase_measured_usage", True),
        ("write_quiescence_irreversible", False),
    ],
)
def test_every_irreversible_seal_fact_is_frozen(field: str, value: object) -> None:
    raw = _raw()
    with pytest.raises(ValueError):
        _replace_unchecked(_sealed(raw), **{field: value})


@pytest.mark.parametrize("seal_time", [900, 899])
def test_chain_is_the_sole_authority_for_precommit_chronology(seal_time: int) -> None:
    raw = _raw()
    committed = replace(_sealed(raw), seal_monotonic_ns=seal_time)
    seal.validate_write_quiescence_seal_v1(committed)
    with pytest.raises(ValueError, match="follow raw7 precommit strictly"):
        seal.validate_write_seal_chain_v2(raw, committed)


@pytest.mark.parametrize("seal_time", [True, 0])
def test_standalone_seal_time_is_one_positive_exact_integer(seal_time: object) -> None:
    raw = _raw()
    with pytest.raises(ValueError):
        _replace_unchecked(_sealed(raw), seal_monotonic_ns=seal_time)


@pytest.mark.parametrize(
    "field",
    [
        "campaign_id",
        "case_subject",
        "runtime_intent",
        "host_go",
        "image_id",
        "container_name",
        "container_id_commitment_sha256",
        "outer_cgroup_identity_sha256",
    ],
)
def test_runtime_envelope_crosswires_fail_the_raw_to_seal_chain(field: str) -> None:
    raw = _raw()
    committed = _sealed(raw)
    alternate = _runtime("alternate-runtime")
    if field == "campaign_id":
        alternate = replace(raw.runtime, campaign_id="other_campaign")
    elif field == "case_subject":
        alternate = replace(raw.runtime, case_subject=CaseSubjectV1.for_ordinal(1))
    elif field == "container_name":
        alternate = replace(raw.runtime, container_name="alternate-qualified-case-00")
    else:
        alternate = replace(raw.runtime, **{field: getattr(alternate, field)})
    crosswired = replace(committed, runtime=alternate)
    with pytest.raises(ERROR, match="runtime envelope crosswires"):
        seal.validate_write_seal_chain_v2(raw, crosswired)


@pytest.mark.parametrize(
    ("field", "schema_version"),
    [
        ("publication_reload_validation", seal.PUBLICATION_RELOAD_VALIDATION_SCHEMA_VERSION),
        (
            "terminal_relay_preseal_attestation",
            seal.TERMINAL_RELAY_PRESEAL_ATTESTATION_SCHEMA_VERSION,
        ),
        (
            "nonstorage_channel_preseal_attestation",
            seal.NONSTORAGE_CHANNEL_PRESEAL_ATTESTATION_SCHEMA_VERSION,
        ),
    ],
)
def test_every_predecessor_crosswire_fails_chain(field: str, schema_version: str) -> None:
    raw = _raw()
    committed = _sealed(raw)
    crosswired = _replace_unchecked(
        committed,
        **{field: _artifact(schema_version, f"other:{field}")},
    )
    with pytest.raises(ERROR, match="crosswires raw7"):
        seal.validate_write_seal_chain_v2(raw, crosswired)


def test_producer_and_raw_identity_crosswires_fail_chain() -> None:
    raw = _raw()
    committed = _sealed(raw)
    other_producer = _producer("other")
    with pytest.raises(ERROR, match="producer crosswires"):
        seal.validate_write_seal_chain_v2(
            raw,
            replace(committed, producer=other_producer),
        )
    other_raw = _raw(precommit_monotonic_ns=800)
    with pytest.raises(ERROR, match="raw7 identity"):
        seal.validate_write_seal_chain_v2(
            raw,
            replace(committed, raw_write_seal=seal.raw_write_seal_identity_v1(other_raw)),
        )


@pytest.mark.parametrize(
    ("field", "wrong_schema"),
    [
        ("runtime_intent", seal.HOST_GO_V3_SCHEMA_VERSION),
        ("host_go", seal.STORAGE_BOUNDARY_RUNTIME_INTENT_SCHEMA_VERSION),
    ],
)
def test_runtime_artifact_schemas_are_exact(field: str, wrong_schema: str) -> None:
    runtime = _runtime()
    with pytest.raises(ERROR):
        _replace_unchecked(runtime, **{field: _artifact(wrong_schema, f"wrong:{field}")})


@pytest.mark.parametrize(
    ("field", "wrong_schema"),
    [
        ("publication_reload_validation", seal.RAW_WRITE_SEAL_SCHEMA_VERSION),
        (
            "terminal_relay_preseal_attestation",
            seal.NONSTORAGE_CHANNEL_PRESEAL_ATTESTATION_SCHEMA_VERSION,
        ),
        (
            "nonstorage_channel_preseal_attestation",
            seal.TERMINAL_RELAY_PRESEAL_ATTESTATION_SCHEMA_VERSION,
        ),
        ("tmpfs_hard_limit_mount_mutation_closure", seal.RAW_WRITE_SEAL_SCHEMA_VERSION),
        ("writable_fd_lifetime_inventory", seal.RAW_WRITE_SEAL_SCHEMA_VERSION),
        ("docker_api_operation_journal", seal.RAW_WRITE_SEAL_SCHEMA_VERSION),
        ("rootfs_upperdir_interval_delta", seal.RAW_WRITE_SEAL_SCHEMA_VERSION),
        ("docker_volume_inventory_delta", seal.RAW_WRITE_SEAL_SCHEMA_VERSION),
    ],
)
def test_every_raw_dependency_schema_is_exact(field: str, wrong_schema: str) -> None:
    with pytest.raises(ERROR):
        _raw(**{field: _artifact(wrong_schema, f"wrong:{field}")})


@pytest.mark.parametrize(
    ("field", "wrong_schema"),
    [
        ("publication_reload_validation", seal.RAW_WRITE_SEAL_SCHEMA_VERSION),
        (
            "terminal_relay_preseal_attestation",
            seal.NONSTORAGE_CHANNEL_PRESEAL_ATTESTATION_SCHEMA_VERSION,
        ),
        (
            "nonstorage_channel_preseal_attestation",
            seal.TERMINAL_RELAY_PRESEAL_ATTESTATION_SCHEMA_VERSION,
        ),
        ("raw_write_seal", seal.PUBLICATION_RELOAD_VALIDATION_SCHEMA_VERSION),
    ],
)
def test_every_seal_dependency_schema_is_exact(field: str, wrong_schema: str) -> None:
    raw = _raw()
    committed = _sealed(raw)
    with pytest.raises(ERROR):
        _replace_unchecked(
            committed,
            **{field: _artifact(wrong_schema, f"wrong:sealed:{field}")},
        )


@pytest.mark.parametrize("field", ["role", "descriptor_schema_version"])
def test_producer_role_and_descriptor_schema_are_recursively_exact(field: str) -> None:
    raw = _raw()
    replacement = (
        "terminal_relay"
        if field == "role"
        else seal.TERMINAL_RELAY_PRESEAL_ATTESTATION_SCHEMA_VERSION
    )
    object.__setattr__(raw.producer, field, replacement)
    with pytest.raises(ERROR):
        seal.validate_raw_write_seal_v1(raw)

    raw = _raw()
    committed = _sealed(raw)
    object.__setattr__(committed.producer, field, replacement)
    with pytest.raises(ERROR):
        seal.validate_write_quiescence_seal_v1(committed)


@pytest.mark.parametrize("kind", ["file", "body"])
def test_cross_artifact_file_and_body_aliases_are_rejected(kind: str) -> None:
    raw = _raw()
    aliased = _artifact_with_alias(
        raw.publication_reload_validation,
        seal.TERMINAL_RELAY_PRESEAL_ATTESTATION_SCHEMA_VERSION,
        "aliased-relay",
        kind=kind,
    )
    with pytest.raises(ValueError, match="aliased"):
        _raw(terminal_relay_preseal_attestation=aliased)


def test_runtime_scalar_and_artifact_identity_aliases_are_rejected() -> None:
    runtime = _runtime()
    with pytest.raises(ValueError, match="aliased"):
        replace(
            runtime,
            outer_cgroup_identity_sha256=runtime.container_id_commitment_sha256,
        )
    with pytest.raises(ValueError, match="aliased"):
        replace(
            runtime,
            image_id=f"sha256:{runtime.container_id_commitment_sha256}",
        )
    aliased_reload = ArtifactRefV1(
        schema_version=seal.PUBLICATION_RELOAD_VALIDATION_SCHEMA_VERSION,
        file_sha256=runtime.container_id_commitment_sha256,
        body_sha256=_sha("aliased-reload:body"),
    )
    with pytest.raises(ValueError, match="aliased"):
        _raw(runtime=runtime, publication_reload_validation=aliased_reload)
    base_producer = _producer()
    aliased_producer = ProducerRefV1(
        role=base_producer.role,
        descriptor_schema_version=base_producer.descriptor_schema_version,
        descriptor_file_sha256=base_producer.descriptor_file_sha256,
        descriptor_body_sha256=base_producer.descriptor_body_sha256,
        source_sha256=runtime.host_go.file_sha256,
    )
    with pytest.raises(ValueError, match="aliased"):
        _raw(runtime=runtime, producer=aliased_producer)


@pytest.mark.parametrize("kind", ["file", "body"])
@pytest.mark.parametrize(
    "field",
    [
        "runtime_intent",
        "host_go",
        "publication_reload_validation",
        "terminal_relay_preseal_attestation",
        "nonstorage_channel_preseal_attestation",
        "tmpfs_hard_limit_mount_mutation_closure",
        "writable_fd_lifetime_inventory",
        "docker_api_operation_journal",
        "rootfs_upperdir_interval_delta",
        "docker_volume_inventory_delta",
    ],
)
def test_every_raw_artifact_identity_rejects_scalar_alias(field: str, kind: str) -> None:
    facts = _raw_inputs()
    runtime = facts["runtime"]
    donor = runtime.container_id_commitment_sha256
    if field in {"runtime_intent", "host_go"}:
        artifact = getattr(runtime, field)
        aliased = _replace_unchecked(artifact, **{f"{kind}_sha256": donor})
        with pytest.raises(ERROR, match="aliased"):
            _replace_unchecked(runtime, **{field: aliased})
        return
    facts[field] = _replace_unchecked(
        facts[field],
        **{f"{kind}_sha256": donor},
    )
    with pytest.raises(ERROR, match="aliased"):
        seal.canonical_raw_write_seal_capture_bytes(**facts)


@pytest.mark.parametrize(
    "producer_field",
    ["descriptor_file_sha256", "descriptor_body_sha256", "source_sha256"],
)
def test_every_raw_producer_identity_rejects_scalar_alias(producer_field: str) -> None:
    facts = _raw_inputs()
    donor = facts["runtime"].container_id_commitment_sha256
    facts["producer"] = _replace_unchecked(
        facts["producer"],
        **{producer_field: donor},
    )
    with pytest.raises(ERROR, match="aliased"):
        seal.canonical_raw_write_seal_capture_bytes(**facts)


@pytest.mark.parametrize("kind", ["file", "body"])
@pytest.mark.parametrize(
    "field",
    [
        "publication_reload_validation",
        "terminal_relay_preseal_attestation",
        "nonstorage_channel_preseal_attestation",
        "raw_write_seal",
    ],
)
def test_every_seal_artifact_identity_rejects_scalar_alias(field: str, kind: str) -> None:
    raw = _raw()
    committed = _sealed(raw)
    artifact = getattr(committed, field)
    aliased = _replace_unchecked(
        artifact,
        **{f"{kind}_sha256": raw.runtime.container_id_commitment_sha256},
    )
    with pytest.raises(ERROR, match="aliased"):
        _replace_unchecked(committed, **{field: aliased})


@pytest.mark.parametrize(
    "producer_field",
    ["descriptor_file_sha256", "descriptor_body_sha256", "source_sha256"],
)
def test_every_seal_producer_identity_rejects_scalar_alias(producer_field: str) -> None:
    raw = _raw()
    committed = _sealed(raw)
    producer = _replace_unchecked(
        committed.producer,
        **{producer_field: raw.runtime.container_id_commitment_sha256},
    )
    with pytest.raises(ERROR, match="aliased"):
        _replace_unchecked(committed, producer=producer)


@pytest.mark.parametrize("kind", ["file", "body"])
@pytest.mark.parametrize("field", ["first_class_artifact", "raw_artifact"])
def test_projection_first_class_identities_reject_alias(field: str, kind: str) -> None:
    raw = _raw()
    committed = _sealed(raw)
    projection = seal.raw_write_seal_projection_v1(raw, committed)
    artifact = getattr(projection, field)
    donor = projection.predecessors[0].file_sha256
    aliased = _replace_unchecked(artifact, **{f"{kind}_sha256": donor})
    with pytest.raises(ERROR, match="aliased"):
        _replace_unchecked(projection, **{field: aliased})


@pytest.mark.parametrize("kind", ["file", "body"])
@pytest.mark.parametrize("index", [0, 1, 2])
def test_every_projection_predecessor_identity_rejects_alias(index: int, kind: str) -> None:
    raw = _raw()
    committed = _sealed(raw)
    projection = seal.raw_write_seal_projection_v1(raw, committed)
    predecessors = list(projection.predecessors)
    predecessors[index] = _replace_unchecked(
        predecessors[index],
        **{f"{kind}_sha256": projection.raw_artifact.file_sha256},
    )
    with pytest.raises(ERROR, match="aliased"):
        replace(projection, predecessors=tuple(predecessors))


@pytest.mark.parametrize(
    "producer_field",
    ["descriptor_file_sha256", "descriptor_body_sha256", "source_sha256"],
)
def test_every_projection_producer_identity_rejects_alias(producer_field: str) -> None:
    raw = _raw()
    committed = _sealed(raw)
    projection = seal.raw_write_seal_projection_v1(raw, committed)
    producer = _replace_unchecked(
        projection.producer,
        **{producer_field: projection.raw_artifact.file_sha256},
    )
    with pytest.raises(ERROR, match="aliased"):
        replace(projection, producer=producer)


def test_raw_capture_and_outer_body_reject_key_type_schema_and_case_mutations() -> None:
    raw = _raw()
    body = raw.to_body_dict()
    for mutation in ("extra", "missing", "bool_count", "schema", "case"):
        changed = dict(body)
        if mutation == "extra":
            changed["future_seal"] = _artifact(
                seal.IRREVERSIBLE_WRITE_SEAL_SCHEMA_VERSION,
                "forbidden-future",
            ).to_dict()
        elif mutation == "missing":
            changed.pop("capture")
        elif mutation == "bool_count":
            changed["descendant_process_count"] = False
        elif mutation == "schema":
            changed["schema_version"] = seal.IRREVERSIBLE_WRITE_SEAL_SCHEMA_VERSION
        else:
            changed["case_subject"] = CaseSubjectV1.for_ordinal(1).to_dict()
        with pytest.raises(ValueError):
            _parse_mutated_raw(changed)


def test_seal_parser_rejects_key_type_schema_and_runtime_mutations() -> None:
    raw = _raw()
    committed = _sealed(raw)
    body = committed.to_body_dict()
    for mutation in ("extra", "missing", "bool_time", "schema", "container"):
        changed = dict(body)
        if mutation == "extra":
            changed["lifecycle"] = _artifact(seal.RAW_WRITE_SEAL_SCHEMA_VERSION, "later").to_dict()
        elif mutation == "missing":
            changed.pop("raw_write_seal")
        elif mutation == "bool_time":
            changed["seal_monotonic_ns"] = True
        elif mutation == "schema":
            changed["schema_version"] = seal.RAW_WRITE_SEAL_SCHEMA_VERSION
        else:
            changed["container_name"] = "not/a/container/name"
        with pytest.raises(ValueError):
            _parse_mutated_seal(changed)


def test_raw7_has_no_future_seal_dependency_or_cycle_surface() -> None:
    raw = _raw()
    assert "seal" not in raw.__dataclass_fields__
    assert "raw_write_seal" not in raw.__dataclass_fields__
    outer = raw.to_body_dict()
    captured = decode_canonical_json_file(raw.capture.decoded_bytes())
    assert "seal" not in outer
    assert "future_seal" not in outer
    assert "seal" not in captured
    assert "future_seal" not in captured
    assert seal.IRREVERSIBLE_WRITE_SEAL_SCHEMA_VERSION.encode("ascii") not in (
        raw.capture.decoded_bytes()
    )
    captured["future_seal"] = _artifact(
        seal.IRREVERSIBLE_WRITE_SEAL_SCHEMA_VERSION,
        "future-seal",
    ).to_dict()
    with pytest.raises(ERROR, match="capture differs"):
        replace(
            raw,
            capture=RawByteCaptureV1.from_bytes(canonical_json_bytes(captured)),
        )


def test_raw7_projection_is_exact_ordered_and_producer_bound() -> None:
    raw = _raw()
    committed = _sealed(raw)
    projection = seal.raw_write_seal_projection_v1(raw, committed)
    assert projection.kind == "write_seal"
    assert projection.predecessors == (
        raw.publication_reload_validation,
        raw.terminal_relay_preseal_attestation,
        raw.nonstorage_channel_preseal_attestation,
    )
    assert projection.producer == raw.producer
    assert projection.exact_projection is True
    assert projection.predecessor_identity_bound is True
    with pytest.raises(ERROR):
        replace(projection, predecessors=tuple(reversed(projection.predecessors)))
    with pytest.raises(ERROR):
        replace(projection, exact_projection=False)
    with pytest.raises(ERROR):
        replace(projection, predecessor_identity_bound=False)


@pytest.mark.parametrize(
    "mutation",
    [
        "schema_version",
        "status",
        "role",
        "owned_artifact_schemas",
        "required_input_schemas",
        "canonical_policy",
        "limits",
        "capabilities",
        "readiness",
        "authority",
        "claims",
        "public_canonical_builders",
        "public_parsers",
        "public_validators",
        "operational_apis",
        "descriptor_self_pin_sha256",
        "source_file_sha256_pin",
    ],
)
def test_descriptor_rejects_every_semantic_mutation(mutation: str) -> None:
    descriptor = seal.WriteSealProducerDescriptorV2()
    changes: dict[str, object]
    if mutation == "schema_version":
        changes = {mutation: seal.RAW_WRITE_SEAL_SCHEMA_VERSION}
    elif mutation == "status":
        changes = {mutation: "production_ready"}
    elif mutation == "role":
        changes = {mutation: "terminal_relay"}
    elif mutation == "owned_artifact_schemas":
        changes = {mutation: tuple(reversed(descriptor.owned_artifact_schemas))}
    elif mutation == "required_input_schemas":
        changes = {mutation: descriptor.required_input_schemas[:-1]}
    elif mutation in {
        "canonical_policy",
        "capabilities",
        "readiness",
        "authority",
        "claims",
    }:
        changed = dict(getattr(descriptor, mutation))
        key = next(iter(changed))
        changed[key] = not changed[key]
        changes = {mutation: MappingProxyType(changed)}
    elif mutation == "limits":
        changed_limits = dict(descriptor.limits)
        changed_limits["maximum_raw_capture_bytes"] += 1
        changes = {mutation: MappingProxyType(changed_limits)}
    elif mutation in {
        "public_canonical_builders",
        "public_parsers",
        "public_validators",
    }:
        changes = {mutation: getattr(descriptor, mutation)[:-1]}
    elif mutation == "operational_apis":
        changes = {mutation: ("execute_write_seal",)}
    elif mutation == "descriptor_self_pin_sha256":
        changes = {mutation: _sha("embedded-self-pin")}
    else:
        changes = {mutation: _sha("embedded-source-pin")}
    with pytest.raises(ValueError):
        _replace_unchecked(descriptor, **changes)


def test_descriptor_posture_is_exactly_source_only_and_defensive() -> None:
    descriptor = seal.WriteSealProducerDescriptorV2()
    assert descriptor.operational_apis == ()
    assert all(value is False for value in descriptor.capabilities.values())
    assert all(value is False for value in descriptor.readiness.values())
    assert all(value is False for value in descriptor.authority.values())
    assert all(value is False for value in descriptor.claims.values())
    assert type(descriptor.capabilities) is type(MappingProxyType({}))
    assert descriptor.descriptor_self_pin_sha256 == "0" * 64
    assert descriptor.source_file_sha256_pin is None
    observed_file, observed_body = seal.write_seal_producer_descriptor_observed_hashes_v2()
    assert observed_file != observed_body
    assert observed_file != "0" * 64
    assert observed_body != "0" * 64


def test_descriptor_inventories_are_exact_unique_disjoint_exported_and_callable() -> None:
    assert seal.OWNED_ARTIFACT_SCHEMAS == (
        seal.RAW_WRITE_SEAL_SCHEMA_VERSION,
        seal.IRREVERSIBLE_WRITE_SEAL_SCHEMA_VERSION,
    )
    assert seal.REQUIRED_INPUT_SCHEMAS == (
        seal.STORAGE_BOUNDARY_RUNTIME_INTENT_SCHEMA_VERSION,
        seal.HOST_GO_V3_SCHEMA_VERSION,
        seal.PUBLICATION_RELOAD_VALIDATION_SCHEMA_VERSION,
        seal.TERMINAL_RELAY_PRESEAL_ATTESTATION_SCHEMA_VERSION,
        seal.NONSTORAGE_CHANNEL_PRESEAL_ATTESTATION_SCHEMA_VERSION,
        seal.TMPFS_HARD_LIMIT_MOUNT_MUTATION_CLOSURE_SCHEMA_VERSION,
        seal.WRITABLE_FD_LIFETIME_INVENTORY_SCHEMA_VERSION,
        seal.DOCKER_API_OPERATION_JOURNAL_SCHEMA_VERSION,
        seal.ROOTFS_UPPERDIR_INTERVAL_DELTA_SCHEMA_VERSION,
        seal.DOCKER_VOLUME_INVENTORY_DELTA_SCHEMA_VERSION,
    )
    assert seal.PUBLIC_CANONICAL_BUILDERS == (
        "canonical_raw_write_seal_capture_bytes",
        "canonical_raw_write_seal_v1_body_bytes",
        "canonical_raw_write_seal_v1_file_bytes",
        "canonical_write_quiescence_seal_v1_body_bytes",
        "canonical_write_quiescence_seal_v1_file_bytes",
        "canonical_write_seal_producer_descriptor_v2_body_bytes",
        "canonical_write_seal_producer_descriptor_v2_file_bytes",
    )
    assert seal.PUBLIC_PARSERS == (
        "parse_raw_write_seal_v1",
        "parse_write_quiescence_seal_v1",
        "parse_write_seal_producer_descriptor_v2",
    )
    assert seal.PUBLIC_VALIDATORS == (
        "validate_raw_write_seal_v1",
        "validate_raw_write_seal_projection_v1",
        "validate_write_quiescence_seal_v1",
        "validate_write_seal_chain_v2",
    )
    assert set(seal.OWNED_ARTIFACT_SCHEMAS).isdisjoint(seal.REQUIRED_INPUT_SCHEMAS)
    for inventory in (
        seal.OWNED_ARTIFACT_SCHEMAS,
        seal.REQUIRED_INPUT_SCHEMAS,
        seal.PUBLIC_CANONICAL_BUILDERS,
        seal.PUBLIC_PARSERS,
        seal.PUBLIC_VALIDATORS,
    ):
        assert len(inventory) == len(set(inventory))
        assert all(type(item) is str for item in inventory)
    for name in (
        *seal.PUBLIC_CANONICAL_BUILDERS,
        *seal.PUBLIC_PARSERS,
        *seal.PUBLIC_VALIDATORS,
    ):
        assert name in seal.__all__
        assert callable(getattr(seal, name))


def test_descriptor_parser_rejects_extra_key_and_mutated_posture() -> None:
    descriptor = seal.WriteSealProducerDescriptorV2()
    body = descriptor.to_body_dict()
    for mutation in ("extra", "readiness", "operational"):
        changed = dict(body)
        if mutation == "extra":
            changed["execute"] = True
        elif mutation == "readiness":
            readiness = dict(changed["readiness"])
            readiness["production_ready"] = True
            changed["readiness"] = readiness
        else:
            changed["operational_apis"] = ["execute_write_seal"]
        encoded, file_pin, body_pin = _encoded_body(
            changed,
            seal.WRITE_SEAL_PRODUCER_DESCRIPTOR_BODY_SHA256_FIELD,
        )
        with pytest.raises(ValueError):
            seal.parse_write_seal_producer_descriptor_v2(
                encoded,
                expected_file_sha256=file_pin,
                expected_body_sha256=body_pin,
            )


def test_equality_spoofs_cannot_satisfy_any_exact_semantic_field() -> None:
    spoof = _EqualitySpoof()
    raw = _raw()
    committed = _sealed(raw)
    projection = seal.raw_write_seal_projection_v1(raw, committed)
    for value, changes in (
        (raw, {"schema_version": spoof}),
        (raw, {"status": spoof}),
        (committed, {"schema_version": spoof}),
        (committed, {"status": spoof}),
        (committed, {"architecture_kind": spoof}),
        (projection, {"kind": spoof}),
    ):
        with pytest.raises(ERROR):
            _replace_unchecked(value, **changes)

    descriptor = seal.WriteSealProducerDescriptorV2()
    for field in (
        "schema_version",
        "status",
        "role",
        "descriptor_self_pin_sha256",
    ):
        with pytest.raises(ERROR):
            _replace_unchecked(descriptor, **{field: spoof})
    for field in (
        "owned_artifact_schemas",
        "required_input_schemas",
        "public_canonical_builders",
        "public_parsers",
        "public_validators",
    ):
        original = getattr(descriptor, field)
        poisoned = (spoof, *original[1:])
        with pytest.raises(ERROR):
            _replace_unchecked(descriptor, **{field: poisoned})
    for field in (
        "canonical_policy",
        "limits",
        "capabilities",
        "readiness",
        "authority",
        "claims",
    ):
        poisoned_mapping = {key: spoof for key in getattr(descriptor, field)}
        with pytest.raises(ERROR):
            _replace_unchecked(
                descriptor,
                **{field: MappingProxyType(poisoned_mapping)},
            )


def test_descriptor_defensively_detaches_every_mapping_backing_store() -> None:
    baseline = seal.WriteSealProducerDescriptorV2()
    for field, expected in (
        ("canonical_policy", seal.CANONICAL_POLICY),
        ("capabilities", seal.SOURCE_ONLY_CAPABILITIES),
        ("readiness", seal.SOURCE_ONLY_READINESS),
        ("authority", seal.SOURCE_ONLY_AUTHORITY),
        ("claims", seal.SOURCE_ONLY_CLAIMS),
    ):
        backing = dict(expected)
        descriptor = _replace_unchecked(
            baseline,
            **{field: MappingProxyType(backing)},
        )
        key = next(iter(backing))
        backing[key] = not backing[key]
        assert dict(getattr(descriptor, field)) == dict(expected)

    limit_backing = dict(seal.LIMITS)
    descriptor = replace(baseline, limits=MappingProxyType(limit_backing))
    limit_backing["maximum_raw_capture_bytes"] += 1
    assert dict(descriptor.limits) == dict(seal.LIMITS)
    encoded = seal.canonical_write_seal_producer_descriptor_v2_file_bytes(descriptor)
    body = seal.canonical_write_seal_producer_descriptor_v2_body_bytes(descriptor)
    assert (
        seal.parse_write_seal_producer_descriptor_v2(
            encoded,
            expected_file_sha256=hashlib.sha256(encoded).hexdigest(),
            expected_body_sha256=hashlib.sha256(body).hexdigest(),
        )
        == descriptor
    )


def test_nested_runtime_case_artifact_producer_and_capture_tamper_fail_closed() -> None:
    runtime = _runtime()
    object.__setattr__(runtime, "campaign_id", "bad/campaign")
    with pytest.raises(ERROR):
        runtime.to_dict()

    runtime = _runtime()
    object.__setattr__(runtime.case_subject, "case_ordinal", True)
    with pytest.raises(ERROR):
        runtime.to_dict()

    runtime = _runtime()
    object.__setattr__(runtime.runtime_intent, "file_sha256", "0" * 64)
    with pytest.raises(ERROR):
        runtime.to_dict()

    raw = _raw()
    object.__setattr__(raw.producer, "role", "terminal_relay")
    with pytest.raises(ERROR):
        seal.validate_raw_write_seal_v1(raw)

    raw = _raw()
    object.__setattr__(raw.capture, "raw_sha256", _sha("tampered capture identity"))
    for operation in (
        lambda: seal.validate_raw_write_seal_v1(raw),
        lambda: seal.canonical_raw_write_seal_v1_body_bytes(raw),
        lambda: seal.canonical_raw_write_seal_v1_file_bytes(raw),
        lambda: seal.raw_write_seal_identity_v1(raw),
    ):
        with pytest.raises(ERROR):
            operation()

    raw = _raw()
    object.__setattr__(
        raw.capture,
        "raw_sha256",
        raw.runtime.container_id_commitment_sha256,
    )
    with pytest.raises(ERROR):
        seal.validate_raw_write_seal_v1(raw)


@pytest.mark.parametrize("kind", ["artifact", "producer", "case", "capture"])
def test_nested_codec_current_state_errors_are_translated_exactly(kind: str) -> None:
    raw = _raw()
    if kind == "artifact":
        object.__setattr__(raw.runtime.runtime_intent, "file_sha256", "0" * 64)
    elif kind == "producer":
        object.__setattr__(raw.producer, "role", "terminal_relay")
    elif kind == "case":
        object.__setattr__(raw.runtime.case_subject, "case_ordinal", True)
    else:
        object.__setattr__(raw.capture, "raw_size_bytes", raw.capture.raw_size_bytes + 1)

    with pytest.raises(ERROR) as caught:
        seal.validate_raw_write_seal_v1(raw)
    assert type(caught.value) is ERROR
    assert type(caught.value.__cause__) is CanonicalEvidenceError


def test_nested_capture_decode_error_is_translated_exactly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _raw()

    def reject_current_capture(_capture: RawByteCaptureV1) -> bytes:
        raise CanonicalEvidenceError("synthetic current-state decode rejection")

    monkeypatch.setattr(RawByteCaptureV1, "decoded_bytes", reject_current_capture)
    with pytest.raises(ERROR) as caught:
        seal.validate_raw_write_seal_v1(raw)
    assert type(caught.value) is ERROR
    assert type(caught.value.__cause__) is CanonicalEvidenceError


def test_nested_seal_and_projection_tamper_cannot_serialize_or_validate() -> None:
    raw = _raw()
    committed = _sealed(raw)
    object.__setattr__(committed.runtime, "container_name", "bad/container")
    for operation in (
        lambda: seal.validate_write_quiescence_seal_v1(committed),
        lambda: seal.canonical_write_quiescence_seal_v1_body_bytes(committed),
        lambda: seal.canonical_write_quiescence_seal_v1_file_bytes(committed),
        lambda: seal.write_quiescence_seal_identity_v1(committed),
    ):
        with pytest.raises(ERROR):
            operation()

    raw = _raw()
    committed = _sealed(raw)
    projection = seal.raw_write_seal_projection_v1(raw, committed)
    object.__setattr__(projection.raw_artifact, "schema_version", seal.HOST_GO_V3_SCHEMA_VERSION)
    with pytest.raises(ERROR):
        seal.validate_raw_write_seal_projection_v1(projection)
    with pytest.raises(ERROR):
        projection.to_dict()


def test_public_record_types_are_runtime_final_and_exact() -> None:
    for name, parent in (
        ("RuntimeSubclass", seal.WriteSealRuntimeEnvelopeV1),
        ("RawSubclass", seal.RawWriteSealV1),
        ("SealSubclass", seal.WriteQuiescenceSealV1),
        ("ProjectionSubclass", seal.RawWriteSealProjectionV1),
        ("DescriptorSubclass", seal.WriteSealProducerDescriptorV2),
    ):
        with pytest.raises(TypeError):
            type(name, (parent,), {})


def test_exports_are_complete_and_no_operational_surface_is_exported() -> None:
    required = {
        "WriteSealRuntimeEnvelopeV1",
        "RawWriteSealV1",
        "WriteQuiescenceSealV1",
        "RawWriteSealProjectionV1",
        "WriteSealProducerDescriptorV2",
        "build_raw_write_seal_v1",
        "parse_raw_write_seal_v1",
        "parse_write_quiescence_seal_v1",
        "parse_write_seal_producer_descriptor_v2",
        "validate_raw_write_seal_projection_v1",
        "validate_write_seal_chain_v2",
        "raw_write_seal_projection_v1",
    }
    assert required <= set(seal.__all__)
    assert "write_seal_producer_ref_v2" not in seal.__all__
    assert not any(name.startswith(("execute_", "run_", "launch_")) for name in seal.__all__)


def test_source_ast_bans_operational_imports_calls_and_reverse_bindings() -> None:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    forbidden_imports = {
        "asyncio",
        "builtins",
        "concurrent",
        "ctypes",
        "docker",
        "fcntl",
        "ftplib",
        "http",
        "importlib",
        "io",
        "mmap",
        "multiprocessing",
        "os",
        "pathlib",
        "posix",
        "resource",
        "selectors",
        "shutil",
        "signal",
        "socket",
        "ssl",
        "subprocess",
        "sys",
        "tempfile",
        "threading",
        "urllib",
    }
    forbidden_calls = {
        "__import__",
        "breakpoint",
        "compile",
        "eval",
        "exec",
        "input",
        "open",
        "system",
    }
    forbidden_attributes = {
        "Popen",
        "call",
        "check_call",
        "check_output",
        "connect",
        "create_connection",
        "execv",
        "execve",
        "makedirs",
        "mkdir",
        "open",
        "popen",
        "read_bytes",
        "read_text",
        "remove",
        "rmtree",
        "run",
        "spawn",
        "system",
        "unlink",
        "write_bytes",
        "write_text",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert not ({alias.name.split(".")[0] for alias in node.names} & forbidden_imports)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            assert node.module.split(".")[0] not in forbidden_imports
            assert "storage_backend" not in node.module
            assert "host_qualification_executor" not in node.module
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in forbidden_calls
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr not in forbidden_attributes
    source_text = SOURCE.read_text(encoding="utf-8")
    for forbidden in (
        "evaluation_receipt",
        "issuance_receipt",
        "lifecycle_receipt",
        "merger_receipt",
        "terminal_receipt",
        "write_seal_producer_ref_v2",
    ):
        assert forbidden not in source_text
