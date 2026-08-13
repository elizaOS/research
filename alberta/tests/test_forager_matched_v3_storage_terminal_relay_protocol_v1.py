"""Cheap adversarial tests for the pure matched-v3 terminal-relay schemas."""

from __future__ import annotations

import ast
import dataclasses
import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from alberta_framework.benchmarks import _forager_matched_v3_canonical_evidence as codec
from alberta_framework.benchmarks import (
    forager_matched_v3_storage_terminal_relay_protocol_v1 as relay,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _replace(instance: Any, **changes: Any) -> Any:
    return dataclasses.replace(instance, **changes)


class _AlwaysEqual:
    def __eq__(self, _other: object) -> bool:
        return True


def _ref(schema: str, label: str) -> codec.ArtifactRefV1:
    return codec.ArtifactRefV1(
        schema_version=schema,
        file_sha256=_sha(f"{label} file"),
        body_sha256=_sha(f"{label} body"),
    )


def _descriptor_and_producer() -> tuple[
    relay.TerminalRelayDescriptorV1,
    codec.ProducerRefV1,
]:
    descriptor = relay.TerminalRelayDescriptorV1()
    identity = relay.terminal_relay_descriptor_identity_v1(descriptor)
    producer = codec.ProducerRefV1(
        role="terminal_relay",
        descriptor_schema_version=relay.TERMINAL_RELAY_DESCRIPTOR_SCHEMA_VERSION,
        descriptor_file_sha256=identity.file_sha256,
        descriptor_body_sha256=identity.body_sha256,
        source_sha256=_sha("terminal relay source"),
    )
    return descriptor, producer


def _pre_wrapper_run_kwargs() -> dict[str, Any]:
    _, producer = _descriptor_and_producer()
    return {
        "campaign_id": "matched_v3_campaign_2026",
        "case_subject": codec.CaseSubjectV1.for_ordinal(0),
        "runtime_intent": _ref(
            relay.STORAGE_BOUNDARY_RUNTIME_INTENT_SCHEMA_VERSION,
            "runtime intent",
        ),
        "host_go": _ref(relay.HOST_GO_V3_SCHEMA_VERSION, "host GO"),
        "image_id": f"sha256:{_sha('image')}",
        "container_name": "alberta-qualification-00",
        "container_id_commitment_sha256": _sha("container commitment"),
        "outer_cgroup_identity_sha256": _sha("outer cgroup"),
        "producer": producer,
        "worker_exit_monotonic_ns": 20,
    }


def _actual_reload_observation_bytes(label: str = "primary") -> bytes:
    return codec.canonical_json_bytes(
        {
            "file_count": 4,
            "file_inventory_sha256": _sha(f"{label} file inventory"),
            "publication_manifest_body_sha256": _sha(f"{label} manifest body"),
            "publication_manifest_file_sha256": _sha(f"{label} manifest file"),
            "published_bundle_sha256": _sha(f"{label} published bundle"),
            "schema_version": "alberta.forager_matched_v3.actual_publication_reload_observation.v1",
            "total_size_bytes": 4096,
        }
    )


def _raw_kwargs() -> dict[str, Any]:
    run = _pre_wrapper_run_kwargs()
    # The observation exists first.  Only its digest is subsequently committed
    # by the independently identified wrapper and reload-validation artifacts.
    capture = relay.build_publication_reload_observation_capture_v1(
        observation_bytes=_actual_reload_observation_bytes()
    )
    kwargs: dict[str, Any] = {
        **run,
        "publication_wrapper": _ref(
            relay.NORMALIZED_PUBLICATION_WRAPPER_SCHEMA_VERSION,
            "publication wrapper",
        ),
        "publication_reload_validation": _ref(
            relay.PUBLICATION_RELOAD_VALIDATION_SCHEMA_VERSION,
            "reload validation",
        ),
        "publication_wrapper_monotonic_ns": 30,
        "reload_validation_monotonic_ns": 40,
        "reload_performed": True,
        "reload_read_only": True,
        "capture": capture,
        "expected_reload_observation_sha256": capture.raw_sha256,
        "actual_reload_observation_sha256": capture.raw_sha256,
    }
    return kwargs


def _raw() -> relay.RawPublicationReloadV1:
    return relay.RawPublicationReloadV1(**_raw_kwargs())


def _relay(
    raw: relay.RawPublicationReloadV1 | None = None,
) -> relay.TerminalRelayPresealAttestationV1:
    exact_raw = _raw() if raw is None else raw
    return relay.TerminalRelayPresealAttestationV1(
        campaign_id=exact_raw.campaign_id,
        case_subject=exact_raw.case_subject,
        runtime_intent=exact_raw.runtime_intent,
        host_go=exact_raw.host_go,
        image_id=exact_raw.image_id,
        container_name=exact_raw.container_name,
        container_id_commitment_sha256=exact_raw.container_id_commitment_sha256,
        outer_cgroup_identity_sha256=exact_raw.outer_cgroup_identity_sha256,
        producer=exact_raw.producer,
        publication_wrapper=exact_raw.publication_wrapper,
        publication_reload_validation=exact_raw.publication_reload_validation,
        raw_publication_reload=relay.raw_publication_reload_identity_v1(exact_raw),
        terminal_relay_process_identity_sha256=_sha("relay process"),
        nonstorage_channel_commitment_sha256=_sha("channel commitment"),
        attestation_monotonic_ns=50,
    )


def _resign(body: dict[str, Any], body_field: str) -> tuple[bytes, str, str]:
    raw = codec.canonical_file_bytes(body, body_digest_field=body_field)
    return raw, hashlib.sha256(raw).hexdigest(), codec.canonical_body_sha256(body)


def _assert_current_state_rejected(calls: tuple[Callable[[], object], ...]) -> None:
    for call in calls:
        with pytest.raises(relay.ForagerMatchedV3StorageTerminalRelayProtocolV1Error):
            call()


def test_descriptor_round_trip_identity_binding_and_zero_repository_pins() -> None:
    descriptor, producer = _descriptor_and_producer()
    identity = relay.terminal_relay_descriptor_identity_v1(descriptor)
    raw = relay.build_terminal_relay_descriptor_file_v1(descriptor)
    assert (
        relay.parse_terminal_relay_descriptor_file_v1(
            raw,
            expected_file_sha256=identity.file_sha256,
            expected_body_sha256=identity.body_sha256,
        )
        == descriptor
    )
    relay.validate_terminal_relay_descriptor_binding_v1(descriptor, producer)
    assert {
        "canonical_terminal_relay_descriptor_v1_body_bytes",
        "canonical_raw_publication_reload_v1_body_bytes",
        "canonical_terminal_relay_preseal_attestation_v1_body_bytes",
    } <= set(descriptor.public_canonical_builders)
    assert {
        relay.PINNED_TERMINAL_RELAY_DESCRIPTOR_FILE_SHA256,
        relay.PINNED_TERMINAL_RELAY_DESCRIPTOR_BODY_SHA256,
        relay.PINNED_TERMINAL_RELAY_SOURCE_SHA256,
    } == {"0" * 64}


def test_raw6_and_relay_round_trip_with_independent_file_and_body_pins() -> None:
    raw6 = _raw()
    relay_attestation = _relay(raw6)
    cases: tuple[tuple[Any, Any, Any, Any], ...] = (
        (
            raw6,
            relay.build_raw_publication_reload_file_v1,
            relay.raw_publication_reload_identity_v1,
            relay.parse_raw_publication_reload_file_v1,
        ),
        (
            relay_attestation,
            relay.build_terminal_relay_preseal_attestation_file_v1,
            relay.terminal_relay_preseal_attestation_identity_v1,
            relay.parse_terminal_relay_preseal_attestation_file_v1,
        ),
    )
    for artifact, builder, identity_builder, parser in cases:
        encoded = builder(artifact)
        identity = identity_builder(artifact)
        assert identity.file_sha256 != identity.body_sha256
        assert (
            parser(
                encoded,
                expected_file_sha256=identity.file_sha256,
                expected_body_sha256=identity.body_sha256,
            )
            == artifact
        )
    relay.validate_terminal_relay_chain_v1(raw6, relay_attestation)


def test_canonical_body_byte_functions_are_unframed_exact_replays() -> None:
    descriptor = relay.TerminalRelayDescriptorV1()
    raw6 = _raw()
    attestation = _relay(raw6)
    cases: tuple[tuple[Any, Any], ...] = (
        (descriptor, relay.canonical_terminal_relay_descriptor_v1_body_bytes),
        (raw6, relay.canonical_raw_publication_reload_v1_body_bytes),
        (
            attestation,
            relay.canonical_terminal_relay_preseal_attestation_v1_body_bytes,
        ),
    )
    for artifact, body_builder in cases:
        body_bytes = body_builder(artifact)
        assert not body_bytes.endswith(b"\n")
        assert body_bytes == codec.canonical_json_bytes(
            artifact.to_body_dict(),
            final_lf=False,
        )


@pytest.mark.parametrize("kind", ["raw", "relay", "descriptor"])
def test_file_and_body_pin_swaps_and_aliases_fail_closed(kind: str) -> None:
    artifact: Any
    parser: Any
    if kind == "raw":
        artifact = _raw()
        encoded = relay.build_raw_publication_reload_file_v1(artifact)
        identity = relay.raw_publication_reload_identity_v1(artifact)
        parser = relay.parse_raw_publication_reload_file_v1
    elif kind == "relay":
        artifact = _relay()
        encoded = relay.build_terminal_relay_preseal_attestation_file_v1(artifact)
        identity = relay.terminal_relay_preseal_attestation_identity_v1(artifact)
        parser = relay.parse_terminal_relay_preseal_attestation_file_v1
    else:
        artifact = relay.TerminalRelayDescriptorV1()
        encoded = relay.build_terminal_relay_descriptor_file_v1(artifact)
        identity = relay.terminal_relay_descriptor_identity_v1(artifact)
        parser = relay.parse_terminal_relay_descriptor_file_v1
    with pytest.raises(codec.CanonicalEvidenceError):
        parser(
            encoded,
            expected_file_sha256=identity.body_sha256,
            expected_body_sha256=identity.file_sha256,
        )
    with pytest.raises(codec.CanonicalEvidenceError):
        parser(
            encoded,
            expected_file_sha256=identity.file_sha256,
            expected_body_sha256=identity.file_sha256,
        )


@pytest.mark.parametrize("mutation", ["extra", "missing", "schema", "bool_type"])
def test_raw6_parser_rejects_exact_key_schema_and_type_mutations(mutation: str) -> None:
    body = _raw().to_body_dict()
    if mutation == "extra":
        body["extra"] = False
    elif mutation == "missing":
        del body["campaign_id"]
    elif mutation == "schema":
        body["schema_version"] = "alberta.wrong.v1"
    else:
        body["reload_performed"] = 1
    encoded, file_pin, body_pin = _resign(body, "raw_publication_reload_body_sha256")
    with pytest.raises(relay.ForagerMatchedV3StorageTerminalRelayProtocolV1Error):
        relay.parse_raw_publication_reload_file_v1(
            encoded,
            expected_file_sha256=file_pin,
            expected_body_sha256=body_pin,
        )


@pytest.mark.parametrize("mutation", ["extra", "missing", "schema", "bool_type"])
def test_relay_parser_rejects_exact_key_schema_and_type_mutations(mutation: str) -> None:
    body = _relay().to_body_dict()
    if mutation == "extra":
        body["extra"] = False
    elif mutation == "missing":
        del body["campaign_id"]
    elif mutation == "schema":
        body["schema_version"] = "alberta.wrong.v1"
    else:
        body["ready_before_write_seal"] = 1
    encoded, file_pin, body_pin = _resign(
        body,
        "terminal_relay_preseal_attestation_body_sha256",
    )
    with pytest.raises(relay.ForagerMatchedV3StorageTerminalRelayProtocolV1Error):
        relay.parse_terminal_relay_preseal_attestation_file_v1(
            encoded,
            expected_file_sha256=file_pin,
            expected_body_sha256=body_pin,
        )


def test_capture_binds_caller_observed_content_to_expected_and_actual_digests() -> None:
    raw6 = _raw()
    assert raw6.capture.raw_sha256 == raw6.expected_reload_observation_sha256
    assert raw6.capture.raw_sha256 == raw6.actual_reload_observation_sha256
    altered_capture = relay.build_publication_reload_observation_capture_v1(
        observation_bytes=_actual_reload_observation_bytes("different actual content")
    )
    assert altered_capture.raw_sha256 != raw6.capture.raw_sha256
    with pytest.raises(relay.ForagerMatchedV3StorageTerminalRelayProtocolV1Error):
        _replace(raw6, capture=altered_capture)
    changed = _replace(
        raw6,
        capture=altered_capture,
        expected_reload_observation_sha256=altered_capture.raw_sha256,
        actual_reload_observation_sha256=altered_capture.raw_sha256,
    )
    assert changed.capture.decoded_bytes() == _actual_reload_observation_bytes(
        "different actual content"
    )
    with pytest.raises(relay.ForagerMatchedV3StorageTerminalRelayProtocolV1Error):
        _replace(raw6, actual_reload_observation_sha256=_sha("different observation"))


def test_capture_is_caller_observed_content_and_precedes_future_artifact_identities() -> None:
    first_bytes = _actual_reload_observation_bytes("first")
    second_bytes = _actual_reload_observation_bytes("second")
    first_capture = relay.build_publication_reload_observation_capture_v1(
        observation_bytes=first_bytes
    )
    second_capture = relay.build_publication_reload_observation_capture_v1(
        observation_bytes=second_bytes
    )
    assert first_capture.raw_sha256 != second_capture.raw_sha256
    with pytest.raises(TypeError):
        relay.build_publication_reload_observation_capture_v1(  # type: ignore[call-arg]
            campaign_id="metadata_only_is_not_an_observation"
        )
    with pytest.raises(codec.CanonicalEvidenceError):
        relay.build_publication_reload_observation_capture_v1(
            observation_bytes=b"opaque noncanonical publication bytes"
        )

    # Construction is strictly acyclic: observed bytes and their digest exist
    # before either future first-class artifact identity is supplied.
    wrapper = _ref(relay.NORMALIZED_PUBLICATION_WRAPPER_SCHEMA_VERSION, "later wrapper")
    validation = _ref(
        relay.PUBLICATION_RELOAD_VALIDATION_SCHEMA_VERSION,
        "later validation",
    )
    raw6 = relay.RawPublicationReloadV1(
        **_pre_wrapper_run_kwargs(),
        publication_wrapper=wrapper,
        publication_reload_validation=validation,
        expected_reload_observation_sha256=first_capture.raw_sha256,
        actual_reload_observation_sha256=first_capture.raw_sha256,
        publication_wrapper_monotonic_ns=30,
        reload_validation_monotonic_ns=40,
        capture=first_capture,
    )
    assert raw6.capture.decoded_bytes() == first_bytes


@pytest.mark.parametrize(
    "field,value",
    [
        ("reload_performed", False),
        ("reload_performed", 1),
        ("reload_read_only", False),
        ("reload_read_only", 1),
    ],
)
def test_raw6_frozen_reload_posture_rejects_false_and_bool_impersonators(
    field: str,
    value: object,
) -> None:
    with pytest.raises(relay.ForagerMatchedV3StorageTerminalRelayProtocolV1Error):
        _replace(_raw(), **{field: value})


@pytest.mark.parametrize(
    "worker,wrapper,reload_time",
    [(20, 20, 40), (20, 40, 40), (30, 20, 40), (20, 50, 40)],
)
def test_raw6_strict_chronology_rejects_equality_and_reversal(
    worker: int,
    wrapper: int,
    reload_time: int,
) -> None:
    kwargs = _raw_kwargs()
    kwargs.update(
        {
            "worker_exit_monotonic_ns": worker,
            "publication_wrapper_monotonic_ns": wrapper,
            "reload_validation_monotonic_ns": reload_time,
        }
    )
    with pytest.raises(relay.ForagerMatchedV3StorageTerminalRelayProtocolV1Error):
        relay.RawPublicationReloadV1(**kwargs)


@pytest.mark.parametrize(
    "field,value",
    [
        ("worker_exit_observed", False),
        ("worker_has_measured_writable_namespace", True),
        ("worker_has_measured_writable_fd", True),
        ("relay_has_measured_writable_namespace", True),
        ("relay_has_measured_writable_fd", True),
        ("terminal_transport_outside_measured_storage", False),
        ("terminal_transport_can_allocate_measured_storage", True),
        ("terminal_emission_performed", True),
        ("ready_before_write_seal", False),
        ("worker_exit_observed", 1),
        ("relay_has_measured_writable_fd", 0),
    ],
)
def test_relay_frozen_posture_rejects_every_substitution(field: str, value: object) -> None:
    with pytest.raises(relay.ForagerMatchedV3StorageTerminalRelayProtocolV1Error):
        _replace(_relay(), **{field: value})


def test_relay_input_policy_is_exact_and_terminal_emission_is_not_preseal() -> None:
    artifact = _relay()
    assert artifact.input_policy == "exact_committed_storage_receipt_identity_only"
    assert artifact.terminal_emission_performed is False
    with pytest.raises(relay.ForagerMatchedV3StorageTerminalRelayProtocolV1Error):
        _replace(artifact, input_policy="full_receipt_payload")


@pytest.mark.parametrize(
    "field,schema",
    [
        ("runtime_intent", relay.STORAGE_BOUNDARY_RUNTIME_INTENT_SCHEMA_VERSION),
        ("host_go", relay.HOST_GO_V3_SCHEMA_VERSION),
        ("publication_wrapper", relay.NORMALIZED_PUBLICATION_WRAPPER_SCHEMA_VERSION),
        (
            "publication_reload_validation",
            relay.PUBLICATION_RELOAD_VALIDATION_SCHEMA_VERSION,
        ),
    ],
)
def test_raw6_rejects_file_or_body_identity_aliases_and_schema_crosswires(
    field: str,
    schema: str,
) -> None:
    raw6 = _raw()
    original = getattr(raw6, field)
    assert isinstance(original, codec.ArtifactRefV1)
    alias_source = raw6.host_go if field != "host_go" else raw6.runtime_intent
    aliased = codec.ArtifactRefV1(
        schema_version=schema,
        file_sha256=alias_source.file_sha256,
        body_sha256=_sha(f"{field} replacement body"),
    )
    with pytest.raises(
        (
            relay.ForagerMatchedV3StorageTerminalRelayProtocolV1Error,
            codec.CanonicalEvidenceError,
        )
    ):
        _replace(raw6, **{field: aliased})
    wrong_schema = _ref("alberta.wrong.artifact.v1", f"{field} wrong schema")
    with pytest.raises(relay.ForagerMatchedV3StorageTerminalRelayProtocolV1Error):
        _replace(raw6, **{field: wrong_schema})


def test_case_campaign_image_container_cgroup_and_go_crosswires_fail_closed() -> None:
    raw6 = _raw()
    mutations: tuple[dict[str, object], ...] = (
        {"campaign_id": "not portable/"},
        {"image_id": _sha("not image framing")},
        {"container_name": "bad/name"},
        {"container_id_commitment_sha256": "0" * 64},
        {"outer_cgroup_identity_sha256": "0" * 64},
        {"host_go": _ref("alberta.wrong.go.v1", "wrong GO")},
    )
    for mutation in mutations:
        with pytest.raises(
            (
                relay.ForagerMatchedV3StorageTerminalRelayProtocolV1Error,
                codec.CanonicalEvidenceError,
            )
        ):
            _replace(raw6, **mutation)
    altered_case = _replace(raw6, case_subject=codec.CaseSubjectV1.for_ordinal(1))
    with pytest.raises(relay.ForagerMatchedV3StorageTerminalRelayProtocolV1Error):
        relay.validate_terminal_relay_chain_v1(altered_case, _relay(raw6))


def test_producer_role_crosswire_and_descriptor_identity_crosswire_fail() -> None:
    raw6 = _raw()
    wrong = codec.ProducerRefV1(
        role="nonstorage_channel",
        descriptor_schema_version=(
            "alberta.forager_matched_v3.qualification_storage_nonstorage_channel_descriptor.v1"
        ),
        descriptor_file_sha256=_sha("wrong descriptor file"),
        descriptor_body_sha256=_sha("wrong descriptor body"),
        source_sha256=_sha("wrong source"),
    )
    with pytest.raises(relay.ForagerMatchedV3StorageTerminalRelayProtocolV1Error):
        _replace(raw6, producer=wrong)
    descriptor, producer = _descriptor_and_producer()
    with pytest.raises(relay.ForagerMatchedV3StorageTerminalRelayProtocolV1Error):
        relay.validate_terminal_relay_descriptor_binding_v1(
            descriptor,
            _replace(
                producer,
                descriptor_file_sha256=_sha("different descriptor file"),
            ),
        )


def test_raw6_projection_and_chain_reject_wrapper_raw_producer_and_chronology_crosswires() -> None:
    raw6 = _raw()
    attestation = _relay(raw6)
    relay.validate_terminal_relay_chain_v1(raw6, attestation)
    for mutation in (
        {"publication_wrapper": _ref(relay.NORMALIZED_PUBLICATION_WRAPPER_SCHEMA_VERSION, "x")},
        {
            "publication_reload_validation": _ref(
                relay.PUBLICATION_RELOAD_VALIDATION_SCHEMA_VERSION,
                "x reload",
            )
        },
        {"raw_publication_reload": _ref(relay.RAW_PUBLICATION_RELOAD_SCHEMA_VERSION, "x raw")},
        {"attestation_monotonic_ns": raw6.reload_validation_monotonic_ns},
        {"attestation_monotonic_ns": raw6.reload_validation_monotonic_ns - 1},
    ):
        altered = _replace(attestation, **mutation)
        with pytest.raises(relay.ForagerMatchedV3StorageTerminalRelayProtocolV1Error):
            relay.validate_terminal_relay_chain_v1(raw6, altered)


@pytest.mark.parametrize(
    "field",
    [
        "terminal_relay_process_identity_sha256",
        "nonstorage_channel_commitment_sha256",
    ],
)
def test_terminal_chain_rejects_raw_capture_digest_aliases_with_relay_only_identities(
    field: str,
) -> None:
    raw6 = _raw()
    attestation = _replace(_relay(raw6), **{field: raw6.capture.raw_sha256})
    with pytest.raises(
        (
            relay.ForagerMatchedV3StorageTerminalRelayProtocolV1Error,
            codec.CanonicalEvidenceError,
        )
    ):
        relay.validate_terminal_relay_chain_v1(raw6, attestation)


@pytest.mark.parametrize(
    "field",
    [
        "terminal_relay_process_identity_sha256",
        "nonstorage_channel_commitment_sha256",
    ],
)
@pytest.mark.parametrize("raw_identity_field", ["file_sha256", "body_sha256"])
def test_relay_rejects_raw6_file_and_body_aliases_with_relay_only_identities(
    field: str,
    raw_identity_field: str,
) -> None:
    raw6 = _raw()
    raw_identity = relay.raw_publication_reload_identity_v1(raw6)
    with pytest.raises(
        (
            relay.ForagerMatchedV3StorageTerminalRelayProtocolV1Error,
            codec.CanonicalEvidenceError,
        )
    ):
        _replace(
            _relay(raw6),
            **{field: getattr(raw_identity, raw_identity_field)},
        )


def test_forced_descriptor_inventory_is_rejected_by_every_public_consumer() -> None:
    descriptor, producer = _descriptor_and_producer()
    object.__setattr__(descriptor, "required_input_schemas", ("alberta.invalid.input.v1",))
    _assert_current_state_rejected(
        (
            lambda: relay.build_terminal_relay_descriptor_file_v1(descriptor),
            lambda: relay.canonical_terminal_relay_descriptor_v1_body_bytes(descriptor),
            lambda: relay.terminal_relay_descriptor_identity_v1(descriptor),
            lambda: relay.validate_terminal_relay_descriptor_binding_v1(descriptor, producer),
        )
    )
    encoded, file_pin, body_pin = _resign(
        descriptor.to_body_dict(),
        "terminal_relay_descriptor_body_sha256",
    )
    with pytest.raises(relay.ForagerMatchedV3StorageTerminalRelayProtocolV1Error):
        relay.parse_terminal_relay_descriptor_file_v1(
            encoded,
            expected_file_sha256=file_pin,
            expected_body_sha256=body_pin,
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", "alberta.invalid.raw6.v1"),
        ("reload_performed", False),
        ("expected_reload_observation_sha256", _sha("false expected observation")),
        ("worker_exit_monotonic_ns", 0),
    ],
)
def test_forced_raw6_facts_are_rejected_by_builders_identity_chain_and_parser(
    field: str,
    value: object,
) -> None:
    raw6 = _raw()
    attestation = _relay(raw6)
    object.__setattr__(raw6, field, value)
    _assert_current_state_rejected(
        (
            lambda: relay.build_raw_publication_reload_file_v1(raw6),
            lambda: relay.canonical_raw_publication_reload_v1_body_bytes(raw6),
            lambda: relay.raw_publication_reload_identity_v1(raw6),
            lambda: relay.validate_terminal_relay_chain_v1(raw6, attestation),
        )
    )
    encoded, file_pin, body_pin = _resign(
        raw6.to_body_dict(),
        "raw_publication_reload_body_sha256",
    )
    with pytest.raises(relay.ForagerMatchedV3StorageTerminalRelayProtocolV1Error):
        relay.parse_raw_publication_reload_file_v1(
            encoded,
            expected_file_sha256=file_pin,
            expected_body_sha256=body_pin,
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("raw_size_bytes", 0),
        ("raw_sha256", _sha("false capture identity")),
        ("raw_bytes_base64", "e30K"),
        ("schema_version", "alberta.invalid.raw_capture.v1"),
    ],
)
def test_forced_raw_capture_metadata_is_rejected_everywhere(
    field: str,
    value: object,
) -> None:
    raw6 = _raw()
    attestation = _relay(raw6)
    object.__setattr__(raw6.capture, field, value)
    with pytest.raises(codec.CanonicalEvidenceError):
        codec.raw_byte_capture_v1_from_dict(raw6.capture.to_dict())
    _assert_current_state_rejected(
        (
            lambda: relay.build_raw_publication_reload_file_v1(raw6),
            lambda: relay.canonical_raw_publication_reload_v1_body_bytes(raw6),
            lambda: relay.raw_publication_reload_identity_v1(raw6),
            lambda: relay.validate_terminal_relay_chain_v1(raw6, attestation),
        )
    )
    encoded, file_pin, body_pin = _resign(
        raw6.to_body_dict(),
        "raw_publication_reload_body_sha256",
    )
    with pytest.raises(relay.ForagerMatchedV3StorageTerminalRelayProtocolV1Error):
        relay.parse_raw_publication_reload_file_v1(
            encoded,
            expected_file_sha256=file_pin,
            expected_body_sha256=body_pin,
        )


def test_exact_nested_equality_spoofs_are_rejected_by_relay_consumers() -> None:
    raw6 = _raw()
    cases: list[tuple[str, object]] = []

    case_subject = codec.CaseSubjectV1.for_ordinal(1)
    for subject_field in dataclasses.fields(case_subject):
        object.__setattr__(case_subject, subject_field.name, _AlwaysEqual())
    cases.append(("case_subject", case_subject))

    runtime_intent = _replace(raw6.runtime_intent)
    object.__setattr__(runtime_intent, "schema_version", _AlwaysEqual())
    cases.append(("runtime_intent", runtime_intent))

    producer = _replace(raw6.producer)
    object.__setattr__(producer, "role", _AlwaysEqual())
    object.__setattr__(producer, "descriptor_schema_version", _AlwaysEqual())
    cases.append(("producer", producer))

    for record_field, spoofed in cases:
        attestation = _relay(raw6)
        object.__setattr__(attestation, record_field, spoofed)
        _assert_current_state_rejected(
            (
                lambda: relay.build_terminal_relay_preseal_attestation_file_v1(attestation),
                lambda: relay.canonical_terminal_relay_preseal_attestation_v1_body_bytes(
                    attestation
                ),
                lambda: relay.terminal_relay_preseal_attestation_identity_v1(attestation),
                lambda: relay.validate_terminal_relay_chain_v1(
                    raw6,
                    attestation,
                ),
            )
        )


def test_forced_relay_posture_is_rejected_by_builders_identity_chain_and_parser() -> None:
    mutations: tuple[tuple[str, object], ...] = (
        ("worker_exit_observed", False),
        ("worker_has_measured_writable_namespace", True),
        ("worker_has_measured_writable_fd", True),
        ("relay_has_measured_writable_namespace", True),
        ("relay_has_measured_writable_fd", True),
        ("terminal_transport_outside_measured_storage", False),
        ("terminal_transport_can_allocate_measured_storage", True),
        ("input_policy", _AlwaysEqual()),
        ("terminal_emission_performed", True),
        ("ready_before_write_seal", False),
        ("attestation_monotonic_ns", 0),
    )
    for field, value in mutations:
        raw6 = _raw()
        attestation = _relay(raw6)
        object.__setattr__(attestation, field, value)
        _assert_current_state_rejected(
            (
                lambda: relay.build_terminal_relay_preseal_attestation_file_v1(attestation),
                lambda: relay.canonical_terminal_relay_preseal_attestation_v1_body_bytes(
                    attestation
                ),
                lambda: relay.terminal_relay_preseal_attestation_identity_v1(attestation),
                lambda: relay.validate_terminal_relay_chain_v1(raw6, attestation),
            )
        )
        if isinstance(value, _AlwaysEqual):
            with pytest.raises(codec.CanonicalEvidenceError):
                _resign(
                    attestation.to_body_dict(),
                    "terminal_relay_preseal_attestation_body_sha256",
                )
            continue
        encoded, file_pin, body_pin = _resign(
            attestation.to_body_dict(),
            "terminal_relay_preseal_attestation_body_sha256",
        )
        with pytest.raises(relay.ForagerMatchedV3StorageTerminalRelayProtocolV1Error):
            relay.parse_terminal_relay_preseal_attestation_file_v1(
                encoded,
                expected_file_sha256=file_pin,
                expected_body_sha256=body_pin,
            )


def test_descriptor_posture_is_false_immutable_exact_and_mutation_rejected() -> None:
    capabilities = dict(relay.CAPABILITIES)
    descriptor = relay.TerminalRelayDescriptorV1(capabilities=capabilities)
    capabilities["filesystem_access"] = True
    assert descriptor.capabilities["filesystem_access"] is False
    for field_name, posture in (
        ("capabilities", relay.CAPABILITIES),
        ("readiness", relay.READINESS),
        ("authority", relay.AUTHORITY),
        ("claims", relay.CLAIMS),
    ):
        mutated = dict(posture)
        first = next(iter(mutated))
        mutated[first] = True
        with pytest.raises(relay.ForagerMatchedV3StorageTerminalRelayProtocolV1Error):
            _replace(descriptor, **{field_name: mutated})
    with pytest.raises(relay.ForagerMatchedV3StorageTerminalRelayProtocolV1Error):
        _replace(descriptor, operational_apis=("open",))
    with pytest.raises(relay.ForagerMatchedV3StorageTerminalRelayProtocolV1Error):
        _replace(descriptor, descriptor_self_pin_sha256=_sha("fake pin"))
    with pytest.raises(relay.ForagerMatchedV3StorageTerminalRelayProtocolV1Error):
        _replace(descriptor, source_file_sha256_pin=_sha("fake source pin"))


def test_descriptor_parser_rejects_extra_missing_and_bool_as_limit() -> None:
    descriptor = relay.TerminalRelayDescriptorV1()
    for mutation in ("extra", "missing", "bool_limit"):
        body = descriptor.to_body_dict()
        if mutation == "extra":
            body["extra"] = False
        elif mutation == "missing":
            del body["role"]
        else:
            limits = dict(body["limits"])
            limits["max_json_depth"] = True
            body["limits"] = limits
        encoded, file_pin, body_pin = _resign(body, "terminal_relay_descriptor_body_sha256")
        with pytest.raises(relay.ForagerMatchedV3StorageTerminalRelayProtocolV1Error):
            relay.parse_terminal_relay_descriptor_file_v1(
                encoded,
                expected_file_sha256=file_pin,
                expected_body_sha256=body_pin,
            )


def test_public_frozen_records_are_immutable_and_runtime_sealed() -> None:
    raw6 = _raw()
    with pytest.raises(dataclasses.FrozenInstanceError):
        raw6.status = "changed"  # type: ignore[misc]
    for base in (
        relay.TerminalRelayDescriptorV1,
        relay.RawPublicationReloadV1,
        relay.TerminalRelayPresealAttestationV1,
    ):
        with pytest.raises(TypeError, match="runtime-sealed"):
            type("Bypass", (base,), {})


def test_source_ast_has_only_shared_codec_and_no_operational_surface() -> None:
    source_path = Path(relay.__file__)
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_modules: set[str] = set()
    called_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_modules.add(node.module)
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                called_names.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                called_names.add(node.func.attr)
    allowed_import_roots = {
        "__future__",
        "collections",
        "collections.abc",
        "dataclasses",
        "hashlib",
        "hmac",
        "re",
        "types",
        "typing",
        "alberta_framework.benchmarks._forager_matched_v3_canonical_evidence",
    }
    assert imported_modules <= allowed_import_roots
    assert not {
        "open",
        "read_text",
        "read_bytes",
        "write_text",
        "write_bytes",
        "run",
        "Popen",
        "system",
        "socket",
        "connect",
        "mount",
        "exec",
        "eval",
    }.intersection(called_names)
    assert "qualification_storage_backend_v2" not in source
    assert "host_qualification_executor" not in source
    assert "issuer" not in source.lower()
    assert "evaluator" not in source.lower()
    assert "merger" not in source.lower()
