"""Cheap adversarial tests for the pure matched-v3 nonstorage channel schema."""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import inspect
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from alberta_framework.benchmarks import _forager_matched_v3_canonical_evidence as codec
from alberta_framework.benchmarks import (
    forager_matched_v3_storage_nonstorage_channel_protocol_v1 as channel,
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


def _channel_descriptor_and_producer() -> tuple[
    channel.NonstorageChannelDescriptorV1,
    codec.ProducerRefV1,
]:
    descriptor = channel.NonstorageChannelDescriptorV1()
    identity = channel.nonstorage_channel_descriptor_identity_v1(descriptor)
    producer = codec.ProducerRefV1(
        role="nonstorage_channel",
        descriptor_schema_version=channel.NONSTORAGE_CHANNEL_DESCRIPTOR_SCHEMA_VERSION,
        descriptor_file_sha256=identity.file_sha256,
        descriptor_body_sha256=identity.body_sha256,
        source_sha256=_sha("nonstorage channel source"),
    )
    return descriptor, producer


def _terminal_producer() -> codec.ProducerRefV1:
    return codec.ProducerRefV1(
        role="terminal_relay",
        descriptor_schema_version=channel.TERMINAL_RELAY_DESCRIPTOR_SCHEMA_VERSION,
        descriptor_file_sha256=_sha("terminal descriptor file"),
        descriptor_body_sha256=_sha("terminal descriptor body"),
        source_sha256=_sha("terminal source"),
    )


def _base_kwargs() -> dict[str, Any]:
    _, producer = _channel_descriptor_and_producer()
    return {
        "campaign_id": "matched_v3_campaign_2026",
        "case_subject": codec.CaseSubjectV1.for_ordinal(0),
        "runtime_intent": _ref(
            channel.STORAGE_BOUNDARY_RUNTIME_INTENT_SCHEMA_VERSION,
            "runtime intent",
        ),
        "host_go": _ref(channel.HOST_GO_V3_SCHEMA_VERSION, "host GO"),
        "image_id": f"sha256:{_sha('image')}",
        "container_name": "alberta-qualification-00",
        "container_id_commitment_sha256": _sha("container commitment"),
        "outer_cgroup_identity_sha256": _sha("outer cgroup"),
        "producer": producer,
        "terminal_relay_preseal_attestation": _ref(
            channel.TERMINAL_RELAY_PRESEAL_ATTESTATION_SCHEMA_VERSION,
            "relay attestation",
        ),
        "terminal_relay_producer": _terminal_producer(),
        "attestation_monotonic_ns": 60,
    }


def _channel() -> channel.NonstorageChannelPresealAttestationV1:
    kwargs = _base_kwargs()
    kwargs["channel_commitment_sha256"] = channel.nonstorage_channel_commitment_sha256_v1(
        **{
            key: value
            for key, value in kwargs.items()
            if key
            not in {
                "terminal_relay_preseal_attestation",
                "attestation_monotonic_ns",
            }
        }
    )
    return channel.NonstorageChannelPresealAttestationV1(**kwargs)


def _resign(body: dict[str, Any], body_field: str) -> tuple[bytes, str, str]:
    raw = codec.canonical_file_bytes(body, body_digest_field=body_field)
    return raw, hashlib.sha256(raw).hexdigest(), codec.canonical_body_sha256(body)


def _assert_current_state_rejected(calls: tuple[Callable[[], object], ...]) -> None:
    for call in calls:
        with pytest.raises(channel.ForagerMatchedV3StorageNonstorageChannelProtocolV1Error):
            call()


def _validate_dependency(
    artifact: channel.NonstorageChannelPresealAttestationV1,
    **overrides: object,
) -> None:
    values: dict[str, object] = {
        "relay_attestation": artifact.terminal_relay_preseal_attestation,
        "relay_producer": artifact.terminal_relay_producer,
        "relay_campaign_id": artifact.campaign_id,
        "relay_case_subject": artifact.case_subject,
        "relay_runtime_intent": artifact.runtime_intent,
        "relay_host_go": artifact.host_go,
        "relay_image_id": artifact.image_id,
        "relay_container_name": artifact.container_name,
        "relay_container_id_commitment_sha256": artifact.container_id_commitment_sha256,
        "relay_outer_cgroup_identity_sha256": artifact.outer_cgroup_identity_sha256,
        "relay_nonstorage_channel_commitment_sha256": artifact.channel_commitment_sha256,
        "relay_attestation_monotonic_ns": 50,
    }
    values.update(overrides)
    channel.validate_nonstorage_channel_dependency_v1(artifact, **values)  # type: ignore[arg-type]


def test_descriptor_round_trip_identity_binding_and_zero_repository_pins() -> None:
    descriptor, producer = _channel_descriptor_and_producer()
    identity = channel.nonstorage_channel_descriptor_identity_v1(descriptor)
    raw = channel.build_nonstorage_channel_descriptor_file_v1(descriptor)
    assert (
        channel.parse_nonstorage_channel_descriptor_file_v1(
            raw,
            expected_file_sha256=identity.file_sha256,
            expected_body_sha256=identity.body_sha256,
        )
        == descriptor
    )
    channel.validate_nonstorage_channel_descriptor_binding_v1(descriptor, producer)
    assert {
        "canonical_nonstorage_channel_descriptor_v1_body_bytes",
        "canonical_nonstorage_channel_preseal_attestation_v1_body_bytes",
    } <= set(descriptor.public_canonical_builders)
    assert {
        channel.PINNED_NONSTORAGE_CHANNEL_DESCRIPTOR_FILE_SHA256,
        channel.PINNED_NONSTORAGE_CHANNEL_DESCRIPTOR_BODY_SHA256,
        channel.PINNED_NONSTORAGE_CHANNEL_SOURCE_SHA256,
    } == {"0" * 64}


def test_descriptor_required_inputs_are_complete_exact_and_ordered() -> None:
    descriptor = channel.NonstorageChannelDescriptorV1()
    expected = (
        channel.STORAGE_BOUNDARY_RUNTIME_INTENT_SCHEMA_VERSION,
        channel.HOST_GO_V3_SCHEMA_VERSION,
        channel.TERMINAL_RELAY_DESCRIPTOR_SCHEMA_VERSION,
        channel.TERMINAL_RELAY_PRESEAL_ATTESTATION_SCHEMA_VERSION,
        channel.STORAGE_BOUNDARY_RECEIPT_V2_SCHEMA_VERSION,
    )
    assert descriptor.required_input_schemas == expected
    for mutated in (
        expected[:-1],
        expected[:2] + (expected[3], expected[2]) + expected[4:],
    ):
        with pytest.raises(channel.ForagerMatchedV3StorageNonstorageChannelProtocolV1Error):
            _replace(descriptor, required_input_schemas=mutated)

    body = descriptor.to_body_dict()
    body["required_input_schemas"] = list(reversed(expected))
    encoded, file_pin, body_pin = _resign(
        body,
        "nonstorage_channel_descriptor_body_sha256",
    )
    with pytest.raises(channel.ForagerMatchedV3StorageNonstorageChannelProtocolV1Error):
        channel.parse_nonstorage_channel_descriptor_file_v1(
            encoded,
            expected_file_sha256=file_pin,
            expected_body_sha256=body_pin,
        )


def test_channel_round_trip_independent_file_body_pins_and_dependency() -> None:
    artifact = _channel()
    identity = channel.nonstorage_channel_preseal_attestation_identity_v1(artifact)
    raw = channel.build_nonstorage_channel_preseal_attestation_file_v1(artifact)
    assert identity.file_sha256 != identity.body_sha256
    assert (
        channel.parse_nonstorage_channel_preseal_attestation_file_v1(
            raw,
            expected_file_sha256=identity.file_sha256,
            expected_body_sha256=identity.body_sha256,
        )
        == artifact
    )
    _validate_dependency(artifact)


def test_canonical_body_byte_functions_are_unframed_exact_replays() -> None:
    descriptor = channel.NonstorageChannelDescriptorV1()
    attestation = _channel()
    cases: tuple[tuple[Any, Any], ...] = (
        (descriptor, channel.canonical_nonstorage_channel_descriptor_v1_body_bytes),
        (
            attestation,
            channel.canonical_nonstorage_channel_preseal_attestation_v1_body_bytes,
        ),
    )
    for artifact, body_builder in cases:
        body_bytes = body_builder(artifact)
        assert not body_bytes.endswith(b"\n")
        assert body_bytes == codec.canonical_json_bytes(
            artifact.to_body_dict(),
            final_lf=False,
        )


@pytest.mark.parametrize("kind", ["descriptor", "attestation"])
def test_file_body_pin_swaps_and_aliases_fail_closed(kind: str) -> None:
    artifact: Any
    parser: Any
    if kind == "descriptor":
        artifact = channel.NonstorageChannelDescriptorV1()
        raw = channel.build_nonstorage_channel_descriptor_file_v1(artifact)
        identity = channel.nonstorage_channel_descriptor_identity_v1(artifact)
        parser = channel.parse_nonstorage_channel_descriptor_file_v1
    else:
        artifact = _channel()
        raw = channel.build_nonstorage_channel_preseal_attestation_file_v1(artifact)
        identity = channel.nonstorage_channel_preseal_attestation_identity_v1(artifact)
        parser = channel.parse_nonstorage_channel_preseal_attestation_file_v1
    with pytest.raises(codec.CanonicalEvidenceError):
        parser(
            raw,
            expected_file_sha256=identity.body_sha256,
            expected_body_sha256=identity.file_sha256,
        )
    with pytest.raises(codec.CanonicalEvidenceError):
        parser(
            raw,
            expected_file_sha256=identity.file_sha256,
            expected_body_sha256=identity.file_sha256,
        )


@pytest.mark.parametrize("mutation", ["extra", "missing", "schema", "bool_type", "int_type"])
def test_channel_parser_rejects_exact_key_schema_and_type_mutations(mutation: str) -> None:
    body = _channel().to_body_dict()
    if mutation == "extra":
        body["extra"] = False
    elif mutation == "missing":
        del body["campaign_id"]
    elif mutation == "schema":
        body["schema_version"] = "alberta.wrong.v1"
    elif mutation == "bool_type":
        body["close_on_exec"] = 1
    else:
        body["relay_endpoint_count"] = True
    encoded, file_pin, body_pin = _resign(
        body,
        "nonstorage_channel_preseal_attestation_body_sha256",
    )
    with pytest.raises(channel.ForagerMatchedV3StorageNonstorageChannelProtocolV1Error):
        channel.parse_nonstorage_channel_preseal_attestation_file_v1(
            encoded,
            expected_file_sha256=file_pin,
            expected_body_sha256=body_pin,
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("channel_kind", "filesystem_socket"),
        ("relay_endpoint_count", 0),
        ("relay_endpoint_count", True),
        ("trusted_host_endpoint_count", 0),
        ("candidate_endpoint_count", 1),
        ("filesystem_path_present", True),
        ("filesystem_backing_present", True),
        ("close_on_exec", False),
        ("passed_to_candidate", True),
        ("candidate_accessible", True),
        ("can_allocate_measured_storage", True),
        ("frame_policy", "arbitrary_payload"),
        ("maximum_frame_bytes", 511),
        ("maximum_frame_bytes", 513),
        ("ready_for_post_receipt_terminal", False),
        ("terminal_emission_performed", True),
        ("candidate_accessible", 0),
    ],
)
def test_every_frozen_topology_confinement_frame_and_emission_field_rejects_substitution(
    field: str,
    value: object,
) -> None:
    with pytest.raises(channel.ForagerMatchedV3StorageNonstorageChannelProtocolV1Error):
        _replace(_channel(), **{field: value})


def test_frozen_channel_constants_and_exact_512_byte_policy() -> None:
    artifact = _channel()
    assert artifact.channel_kind == "anonymous_unix_seqpacket_socketpair"
    assert artifact.frame_policy == "one_exact_committed_storage_receipt_identity_frame"
    assert artifact.relay_endpoint_count == 1
    assert artifact.trusted_host_endpoint_count == 1
    assert artifact.candidate_endpoint_count == 0
    assert artifact.maximum_frame_bytes == 512
    assert artifact.ready_for_post_receipt_terminal is True
    assert artifact.terminal_emission_performed is False


def test_only_canonical_storage_receipt_identity_frame_fits_frozen_bound() -> None:
    receipt = _ref(channel.STORAGE_BOUNDARY_RECEIPT_V2_SCHEMA_VERSION, "storage receipt")
    frame = channel.build_storage_receipt_identity_frame_v1(receipt)
    assert len(frame) < channel.MAXIMUM_FRAME_BYTES == 512
    assert codec.decode_canonical_json_file(frame) == {"storage_receipt": receipt.to_dict()}
    with pytest.raises(channel.ForagerMatchedV3StorageNonstorageChannelProtocolV1Error):
        channel.build_storage_receipt_identity_frame_v1(
            _ref("alberta.wrong.receipt.v1", "wrong receipt")
        )
    with pytest.raises(channel.ForagerMatchedV3StorageNonstorageChannelProtocolV1Error):
        _replace(_channel(), maximum_frame_bytes=511)


def test_channel_commitment_is_acyclic_but_binds_every_precommitted_context() -> None:
    artifact = _channel()
    signature = inspect.signature(channel.nonstorage_channel_commitment_sha256_v1)
    assert "terminal_relay_preseal_attestation" not in signature.parameters
    assert "attestation_monotonic_ns" not in signature.parameters
    different_relay = _ref(
        channel.TERMINAL_RELAY_PRESEAL_ATTESTATION_SCHEMA_VERSION,
        "different future relay",
    )
    altered = _replace(
        artifact,
        terminal_relay_preseal_attestation=different_relay,
    )
    assert altered.channel_commitment_sha256 == artifact.channel_commitment_sha256
    assert channel.nonstorage_channel_preseal_attestation_identity_v1(
        altered
    ) != channel.nonstorage_channel_preseal_attestation_identity_v1(artifact)


@pytest.mark.parametrize(
    "field,replacement",
    [
        ("campaign_id", "another_campaign"),
        ("case_subject", codec.CaseSubjectV1.for_ordinal(1)),
        (
            "runtime_intent",
            _ref(channel.STORAGE_BOUNDARY_RUNTIME_INTENT_SCHEMA_VERSION, "different intent"),
        ),
        ("host_go", _ref(channel.HOST_GO_V3_SCHEMA_VERSION, "different GO")),
        ("image_id", f"sha256:{_sha('different image')}"),
        ("container_name", "different-container"),
        ("container_id_commitment_sha256", _sha("different container")),
        ("outer_cgroup_identity_sha256", _sha("different cgroup")),
    ],
)
def test_channel_commitment_changes_for_each_precommitted_run_field(
    field: str,
    replacement: object,
) -> None:
    artifact = _channel()
    kwargs = _base_kwargs()
    kwargs[field] = replacement
    observed = channel.nonstorage_channel_commitment_sha256_v1(
        **{
            key: value
            for key, value in kwargs.items()
            if key
            not in {
                "terminal_relay_preseal_attestation",
                "attestation_monotonic_ns",
            }
        }
    )
    assert observed != artifact.channel_commitment_sha256


def test_channel_commitment_binds_both_exact_producer_identities() -> None:
    artifact = _channel()
    for field, producer in (
        (
            "producer",
            _replace(artifact.producer, source_sha256=_sha("different channel source")),
        ),
        (
            "terminal_relay_producer",
            _replace(
                artifact.terminal_relay_producer,
                source_sha256=_sha("different terminal source"),
            ),
        ),
    ):
        kwargs = _base_kwargs()
        kwargs[field] = producer
        observed = channel.nonstorage_channel_commitment_sha256_v1(
            **{
                key: value
                for key, value in kwargs.items()
                if key
                not in {
                    "terminal_relay_preseal_attestation",
                    "attestation_monotonic_ns",
                }
            }
        )
        assert observed != artifact.channel_commitment_sha256


def test_bad_or_stale_channel_commitment_is_rejected() -> None:
    with pytest.raises(
        channel.ForagerMatchedV3StorageNonstorageChannelProtocolV1Error,
        match="commitment differs",
    ):
        _replace(_channel(), channel_commitment_sha256=_sha("stale commitment"))


@pytest.mark.parametrize(
    "mutation",
    [
        {"campaign_id": "bad/campaign"},
        {"case_subject": _AlwaysEqual()},
        {"runtime_intent": _ref("alberta.wrong.runtime_intent.v1", "wrong intent")},
        {"host_go": _ref("alberta.wrong.host_go.v1", "wrong GO")},
        {"image_id": _sha("unframed image")},
        {"container_name": "bad/container"},
        {"container_id_commitment_sha256": "0" * 64},
        {"outer_cgroup_identity_sha256": "0" * 64},
        {"producer": _terminal_producer()},
        {"terminal_relay_producer": _channel_descriptor_and_producer()[1]},
        {"channel_kind": "filesystem_socket"},
        {"candidate_endpoint_count": 1, "candidate_accessible": True},
        {"close_on_exec": 1},
        {"maximum_frame_bytes": 2**63},
    ],
)
def test_commitment_builder_rejects_malformed_configuration(
    mutation: dict[str, object],
) -> None:
    kwargs = {
        key: value
        for key, value in _base_kwargs().items()
        if key not in {"terminal_relay_preseal_attestation", "attestation_monotonic_ns"}
    }
    kwargs.update(mutation)
    with pytest.raises(
        (
            channel.ForagerMatchedV3StorageNonstorageChannelProtocolV1Error,
            codec.CanonicalEvidenceError,
        )
    ):
        channel.nonstorage_channel_commitment_sha256_v1(**kwargs)


@pytest.mark.parametrize(
    "override",
    [
        {"relay_campaign_id": "other_campaign"},
        {"relay_case_subject": codec.CaseSubjectV1.for_ordinal(1)},
        {
            "relay_runtime_intent": _ref(
                channel.STORAGE_BOUNDARY_RUNTIME_INTENT_SCHEMA_VERSION,
                "other runtime intent",
            )
        },
        {"relay_host_go": _ref(channel.HOST_GO_V3_SCHEMA_VERSION, "other GO")},
        {"relay_image_id": f"sha256:{_sha('other image')}"},
        {"relay_container_name": "other-container"},
        {"relay_container_id_commitment_sha256": _sha("other container")},
        {"relay_outer_cgroup_identity_sha256": _sha("other cgroup")},
        {"relay_nonstorage_channel_commitment_sha256": _sha("other commitment")},
        {"relay_attestation_monotonic_ns": 60},
        {"relay_attestation_monotonic_ns": 61},
    ],
)
def test_dependency_linker_rejects_context_commitment_and_chronology_crosswires(
    override: dict[str, object],
) -> None:
    with pytest.raises(channel.ForagerMatchedV3StorageNonstorageChannelProtocolV1Error):
        _validate_dependency(_channel(), **override)


@pytest.mark.parametrize(
    "field",
    [
        "relay_campaign_id",
        "relay_case_subject",
        "relay_runtime_intent",
        "relay_host_go",
        "relay_image_id",
        "relay_container_name",
        "relay_container_id_commitment_sha256",
        "relay_outer_cgroup_identity_sha256",
    ],
)
def test_dependency_linker_rejects_equality_spoofs_for_every_relay_context_field(
    field: str,
) -> None:
    with pytest.raises(
        (
            channel.ForagerMatchedV3StorageNonstorageChannelProtocolV1Error,
            codec.CanonicalEvidenceError,
        )
    ):
        _validate_dependency(_channel(), **{field: _AlwaysEqual()})


def test_dependency_linker_rejects_relay_identity_and_producer_crosswires() -> None:
    artifact = _channel()
    with pytest.raises(channel.ForagerMatchedV3StorageNonstorageChannelProtocolV1Error):
        _validate_dependency(
            artifact,
            relay_attestation=_ref(
                channel.TERMINAL_RELAY_PRESEAL_ATTESTATION_SCHEMA_VERSION,
                "other relay",
            ),
        )
    wrong_terminal = _replace(
        artifact.terminal_relay_producer,
        source_sha256=_sha("different terminal source"),
    )
    with pytest.raises(channel.ForagerMatchedV3StorageNonstorageChannelProtocolV1Error):
        _validate_dependency(artifact, relay_producer=wrong_terminal)


def test_case_campaign_image_container_cgroup_go_and_role_crosswires_fail_closed() -> None:
    artifact = _channel()
    mutations: tuple[dict[str, object], ...] = (
        {"campaign_id": "not portable/"},
        {"case_subject": codec.CaseSubjectV1.for_ordinal(1)},
        {"image_id": _sha("unframed image")},
        {"container_name": "bad/name"},
        {"container_id_commitment_sha256": "0" * 64},
        {"outer_cgroup_identity_sha256": "0" * 64},
        {"host_go": _ref("alberta.wrong.go.v1", "wrong GO")},
        {
            "terminal_relay_preseal_attestation": _ref(
                "alberta.wrong.relay.v1",
                "wrong relay schema",
            )
        },
    )
    for mutation in mutations:
        with pytest.raises(
            (
                channel.ForagerMatchedV3StorageNonstorageChannelProtocolV1Error,
                codec.CanonicalEvidenceError,
            )
        ):
            _replace(artifact, **mutation)


def test_producer_role_crosswires_and_all_identity_aliases_fail_closed() -> None:
    artifact = _channel()
    with pytest.raises(channel.ForagerMatchedV3StorageNonstorageChannelProtocolV1Error):
        _replace(artifact, producer=artifact.terminal_relay_producer)
    aliased_terminal = codec.ProducerRefV1(
        role="terminal_relay",
        descriptor_schema_version=channel.TERMINAL_RELAY_DESCRIPTOR_SCHEMA_VERSION,
        descriptor_file_sha256=artifact.producer.descriptor_file_sha256,
        descriptor_body_sha256=_sha("aliased terminal descriptor body"),
        source_sha256=_sha("aliased terminal source"),
    )
    with pytest.raises(
        (
            channel.ForagerMatchedV3StorageNonstorageChannelProtocolV1Error,
            codec.CanonicalEvidenceError,
        )
    ):
        _replace(artifact, terminal_relay_producer=aliased_terminal)


def test_forced_descriptor_inventory_is_rejected_by_every_public_consumer() -> None:
    descriptor, producer = _channel_descriptor_and_producer()
    object.__setattr__(descriptor, "required_input_schemas", ("alberta.invalid.input.v1",))
    _assert_current_state_rejected(
        (
            lambda: channel.build_nonstorage_channel_descriptor_file_v1(descriptor),
            lambda: channel.canonical_nonstorage_channel_descriptor_v1_body_bytes(descriptor),
            lambda: channel.nonstorage_channel_descriptor_identity_v1(descriptor),
            lambda: channel.validate_nonstorage_channel_descriptor_binding_v1(
                descriptor,
                producer,
            ),
        )
    )
    encoded, file_pin, body_pin = _resign(
        descriptor.to_body_dict(),
        "nonstorage_channel_descriptor_body_sha256",
    )
    with pytest.raises(channel.ForagerMatchedV3StorageNonstorageChannelProtocolV1Error):
        channel.parse_nonstorage_channel_descriptor_file_v1(
            encoded,
            expected_file_sha256=file_pin,
            expected_body_sha256=body_pin,
        )


def test_commitment_recursively_revalidates_subject_artifacts_and_producers() -> None:
    malformed_kwargs: list[dict[str, Any]] = []

    inconsistent_case = codec.CaseSubjectV1.for_ordinal(0)
    object.__setattr__(inconsistent_case, "case_ordinal", 1)
    with pytest.raises(codec.CanonicalEvidenceError):
        codec.case_subject_v1_from_dict(inconsistent_case.to_dict())
    malformed_kwargs.append({"case_subject": inconsistent_case})

    runtime_intent = _ref(
        channel.STORAGE_BOUNDARY_RUNTIME_INTENT_SCHEMA_VERSION,
        "spoofed runtime intent",
    )
    object.__setattr__(runtime_intent, "schema_version", _AlwaysEqual())
    malformed_kwargs.append({"runtime_intent": runtime_intent})

    producer = _channel_descriptor_and_producer()[1]
    object.__setattr__(producer, "role", _AlwaysEqual())
    object.__setattr__(producer, "descriptor_schema_version", _AlwaysEqual())
    malformed_kwargs.append({"producer": producer})

    terminal_producer = _terminal_producer()
    object.__setattr__(terminal_producer, "role", _AlwaysEqual())
    object.__setattr__(terminal_producer, "descriptor_schema_version", _AlwaysEqual())
    malformed_kwargs.append({"terminal_relay_producer": terminal_producer})

    for mutation in malformed_kwargs:
        kwargs = {
            key: value
            for key, value in _base_kwargs().items()
            if key not in {"terminal_relay_preseal_attestation", "attestation_monotonic_ns"}
        }
        kwargs.update(mutation)
        with pytest.raises(channel.ForagerMatchedV3StorageNonstorageChannelProtocolV1Error):
            channel.nonstorage_channel_commitment_sha256_v1(**kwargs)


def test_storage_receipt_frame_recursively_revalidates_exact_reference() -> None:
    receipt = _ref(channel.STORAGE_BOUNDARY_RECEIPT_V2_SCHEMA_VERSION, "storage receipt")
    object.__setattr__(receipt, "file_sha256", "not-a-sha256")
    with pytest.raises(codec.CanonicalEvidenceError):
        codec.artifact_ref_v1_from_dict(receipt.to_dict())
    with pytest.raises(channel.ForagerMatchedV3StorageNonstorageChannelProtocolV1Error):
        channel.build_storage_receipt_identity_frame_v1(receipt)


def test_exact_nested_equality_spoofs_are_rejected_by_channel_consumers() -> None:
    artifact = _channel()

    case_subject = codec.CaseSubjectV1.for_ordinal(1)
    for subject_field in dataclasses.fields(case_subject):
        object.__setattr__(case_subject, subject_field.name, _AlwaysEqual())

    runtime_intent = _replace(artifact.runtime_intent)
    object.__setattr__(runtime_intent, "schema_version", _AlwaysEqual())

    producer = _replace(artifact.producer)
    object.__setattr__(producer, "role", _AlwaysEqual())
    object.__setattr__(producer, "descriptor_schema_version", _AlwaysEqual())

    relay_attestation = _replace(artifact.terminal_relay_preseal_attestation)
    object.__setattr__(relay_attestation, "schema_version", _AlwaysEqual())

    for record_field, spoofed in (
        ("case_subject", case_subject),
        ("runtime_intent", runtime_intent),
        ("producer", producer),
        ("terminal_relay_preseal_attestation", relay_attestation),
    ):
        altered = _channel()
        object.__setattr__(altered, record_field, spoofed)
        _assert_current_state_rejected(
            (
                lambda: channel.build_nonstorage_channel_preseal_attestation_file_v1(altered),
                lambda: channel.canonical_nonstorage_channel_preseal_attestation_v1_body_bytes(
                    altered
                ),
                lambda: channel.nonstorage_channel_preseal_attestation_identity_v1(altered),
                lambda: _validate_dependency(altered),
            )
        )

    with pytest.raises(channel.ForagerMatchedV3StorageNonstorageChannelProtocolV1Error):
        _validate_dependency(artifact, relay_case_subject=case_subject)
    with pytest.raises(channel.ForagerMatchedV3StorageNonstorageChannelProtocolV1Error):
        _validate_dependency(artifact, relay_runtime_intent=runtime_intent)

    terminal_producer = _replace(artifact.terminal_relay_producer)
    object.__setattr__(terminal_producer, "role", _AlwaysEqual())
    object.__setattr__(terminal_producer, "descriptor_schema_version", _AlwaysEqual())
    with pytest.raises(channel.ForagerMatchedV3StorageNonstorageChannelProtocolV1Error):
        _validate_dependency(artifact, relay_producer=terminal_producer)


def test_forced_channel_posture_is_rejected_by_builders_identity_linker_and_parser() -> None:
    mutations: tuple[tuple[str, object], ...] = (
        ("channel_kind", _AlwaysEqual()),
        ("relay_endpoint_count", 0),
        ("trusted_host_endpoint_count", 0),
        ("candidate_endpoint_count", 1),
        ("filesystem_path_present", True),
        ("filesystem_backing_present", True),
        ("close_on_exec", False),
        ("passed_to_candidate", True),
        ("candidate_accessible", True),
        ("can_allocate_measured_storage", True),
        ("frame_policy", "arbitrary_payload"),
        ("maximum_frame_bytes", 511),
        ("ready_for_post_receipt_terminal", False),
        ("terminal_emission_performed", True),
        ("channel_commitment_sha256", _sha("stale forced commitment")),
        ("attestation_monotonic_ns", 0),
    )
    for field, value in mutations:
        artifact = _channel()
        object.__setattr__(artifact, field, value)
        _assert_current_state_rejected(
            (
                lambda: channel.build_nonstorage_channel_preseal_attestation_file_v1(artifact),
                lambda: channel.canonical_nonstorage_channel_preseal_attestation_v1_body_bytes(
                    artifact
                ),
                lambda: channel.nonstorage_channel_preseal_attestation_identity_v1(artifact),
                lambda: _validate_dependency(artifact),
            )
        )
        if isinstance(value, _AlwaysEqual):
            with pytest.raises(codec.CanonicalEvidenceError):
                _resign(
                    artifact.to_body_dict(),
                    "nonstorage_channel_preseal_attestation_body_sha256",
                )
            continue
        encoded, file_pin, body_pin = _resign(
            artifact.to_body_dict(),
            "nonstorage_channel_preseal_attestation_body_sha256",
        )
        with pytest.raises(channel.ForagerMatchedV3StorageNonstorageChannelProtocolV1Error):
            channel.parse_nonstorage_channel_preseal_attestation_file_v1(
                encoded,
                expected_file_sha256=file_pin,
                expected_body_sha256=body_pin,
            )


def test_descriptor_posture_is_all_false_defensively_frozen_and_exact() -> None:
    capabilities = dict(channel.CAPABILITIES)
    descriptor = channel.NonstorageChannelDescriptorV1(capabilities=capabilities)
    capabilities["filesystem_access"] = True
    assert descriptor.capabilities["filesystem_access"] is False
    for field_name, posture in (
        ("capabilities", channel.CAPABILITIES),
        ("readiness", channel.READINESS),
        ("authority", channel.AUTHORITY),
        ("claims", channel.CLAIMS),
    ):
        mutated = dict(posture)
        mutated[next(iter(mutated))] = True
        with pytest.raises(channel.ForagerMatchedV3StorageNonstorageChannelProtocolV1Error):
            _replace(descriptor, **{field_name: mutated})
    with pytest.raises(channel.ForagerMatchedV3StorageNonstorageChannelProtocolV1Error):
        _replace(descriptor, operational_apis=("socket",))
    with pytest.raises(channel.ForagerMatchedV3StorageNonstorageChannelProtocolV1Error):
        _replace(descriptor, descriptor_self_pin_sha256=_sha("fake pin"))
    with pytest.raises(channel.ForagerMatchedV3StorageNonstorageChannelProtocolV1Error):
        _replace(descriptor, source_file_sha256_pin=_sha("fake source pin"))


def test_descriptor_parser_rejects_extra_missing_and_bool_as_limit() -> None:
    descriptor = channel.NonstorageChannelDescriptorV1()
    for mutation in ("extra", "missing", "bool_limit"):
        body = descriptor.to_body_dict()
        if mutation == "extra":
            body["extra"] = False
        elif mutation == "missing":
            del body["role"]
        else:
            limits = dict(body["limits"])
            limits["maximum_frame_bytes"] = True
            body["limits"] = limits
        encoded, file_pin, body_pin = _resign(
            body,
            "nonstorage_channel_descriptor_body_sha256",
        )
        with pytest.raises(channel.ForagerMatchedV3StorageNonstorageChannelProtocolV1Error):
            channel.parse_nonstorage_channel_descriptor_file_v1(
                encoded,
                expected_file_sha256=file_pin,
                expected_body_sha256=body_pin,
            )


def test_public_frozen_records_are_immutable_and_runtime_sealed() -> None:
    artifact = _channel()
    with pytest.raises(dataclasses.FrozenInstanceError):
        artifact.status = "changed"  # type: ignore[misc]
    for base in (
        channel.NonstorageChannelDescriptorV1,
        channel.NonstorageChannelPresealAttestationV1,
    ):
        with pytest.raises(TypeError, match="runtime-sealed"):
            type("Bypass", (base,), {})


def test_source_ast_has_only_shared_codec_and_no_operational_surface() -> None:
    source_path = Path(channel.__file__)
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
