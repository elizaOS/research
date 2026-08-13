"""Pure nonstorage-channel evidence schema for matched Forager v3 storage.

The records here are bounded canonical metadata only.  They do not create a
socket, open an endpoint, inspect a namespace, transmit a frame, or authorize a
run.  The channel commitment deliberately excludes the future terminal-relay
attestation identity, while the later channel attestation names that relay;
this makes the dependency graph one-way and free of hash cycles.
"""

from __future__ import annotations

import hashlib
import hmac
import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final, Never, cast

from alberta_framework.benchmarks._forager_matched_v3_canonical_evidence import (
    MAX_CANONICAL_FILE_BYTES,
    MAX_JSON_DEPTH,
    MAX_JSON_NODES,
    MAX_RAW_CAPTURE_BYTES,
    ArtifactRefV1,
    CaseSubjectV1,
    ProducerRefV1,
    artifact_ref_v1_from_dict,
    canonical_body_sha256,
    canonical_file_bytes,
    canonical_json_bytes,
    case_subject_v1_from_dict,
    producer_ref_v1_from_dict,
    require_distinct_sha256s,
    require_nonzero_sha256,
    validate_canonical_file,
)

NONSTORAGE_CHANNEL_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.qualification_storage_nonstorage_channel_descriptor.v1"
)
NONSTORAGE_CHANNEL_PRESEAL_ATTESTATION_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.qualification_storage_nonstorage_channel_readiness_attestation.v1"
)
TERMINAL_RELAY_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.qualification_storage_terminal_relay_descriptor.v1"
)
TERMINAL_RELAY_PRESEAL_ATTESTATION_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.qualification_storage_terminal_relay_preseal_attestation.v1"
)
STORAGE_BOUNDARY_RUNTIME_INTENT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.storage_boundary_runtime_intent.v1"
)
HOST_GO_V3_SCHEMA_VERSION: Final = "alberta.forager_matched_v3.host_qualification_go_commitment.v3"
STORAGE_BOUNDARY_RECEIPT_V2_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.storage_boundary_receipt.v2"
)

NONSTORAGE_CHANNEL_ROLE: Final = "nonstorage_channel"
TERMINAL_RELAY_ROLE: Final = "terminal_relay"
NONSTORAGE_CHANNEL_DESCRIPTOR_STATUS: Final = (
    "pure_source_only_unfinalized_uninvoked_no_production_artifact"
)
NONSTORAGE_CHANNEL_PRESEAL_STATUS: Final = (
    "pre_seal_nonstorage_channel_readiness_attestation_non_authorizing"
)
CHANNEL_KIND: Final = "anonymous_unix_seqpacket_socketpair"
FRAME_POLICY: Final = "one_exact_committed_storage_receipt_identity_frame"
MAXIMUM_FRAME_BYTES: Final = 512
RELAY_ENDPOINT_COUNT: Final = 1
TRUSTED_HOST_ENDPOINT_COUNT: Final = 1
CANDIDATE_ENDPOINT_COUNT: Final = 0
_CHANNEL_COMMITMENT_DOMAIN: Final = (
    b"alberta.forager_matched_v3.nonstorage_channel_commitment.v1\x00"
)

_ZERO_SHA256: Final = "0" * 64
PINNED_NONSTORAGE_CHANNEL_DESCRIPTOR_FILE_SHA256: Final = _ZERO_SHA256
PINNED_NONSTORAGE_CHANNEL_DESCRIPTOR_BODY_SHA256: Final = _ZERO_SHA256
PINNED_NONSTORAGE_CHANNEL_SOURCE_SHA256: Final = _ZERO_SHA256

_DESCRIPTOR_BODY_FIELD: Final = "nonstorage_channel_descriptor_body_sha256"
_ATTESTATION_BODY_FIELD: Final = "nonstorage_channel_preseal_attestation_body_sha256"
_IDENTIFIER_RE: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}\Z")
_CONTAINER_NAME_RE: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_IMAGE_ID_RE: Final = re.compile(r"sha256:[0-9a-f]{64}\Z")
_MAX_EXACT_INTEGER: Final = (1 << 63) - 1

NONSTORAGE_CHANNEL_OWNED_ARTIFACT_SCHEMAS: Final = (
    NONSTORAGE_CHANNEL_PRESEAL_ATTESTATION_SCHEMA_VERSION,
)
NONSTORAGE_CHANNEL_REQUIRED_INPUT_SCHEMAS: Final = (
    STORAGE_BOUNDARY_RUNTIME_INTENT_SCHEMA_VERSION,
    HOST_GO_V3_SCHEMA_VERSION,
    TERMINAL_RELAY_DESCRIPTOR_SCHEMA_VERSION,
    TERMINAL_RELAY_PRESEAL_ATTESTATION_SCHEMA_VERSION,
    STORAGE_BOUNDARY_RECEIPT_V2_SCHEMA_VERSION,
)
CANONICAL_POLICY: Final = (
    "bounded_printable_ascii_json",
    "sorted_keys_compact_separators",
    "exactly_one_final_lf",
    "duplicate_keys_rejected",
    "exact_schema_keys_and_types",
    "independent_nonaliasing_file_and_body_sha256",
)
LIMITS: Final[Mapping[str, int]] = MappingProxyType(
    {
        "max_canonical_file_bytes": MAX_CANONICAL_FILE_BYTES,
        "max_json_depth": MAX_JSON_DEPTH,
        "max_json_nodes": MAX_JSON_NODES,
        "max_raw_capture_bytes": MAX_RAW_CAPTURE_BYTES,
        "maximum_frame_bytes": MAXIMUM_FRAME_BYTES,
    }
)
CAPABILITIES: Final[Mapping[str, bool]] = MappingProxyType(
    {
        "container_control": False,
        "filesystem_access": False,
        "network_access": False,
        "process_control": False,
        "transport_creation": False,
        "transport_execution": False,
    }
)
READINESS: Final[Mapping[str, bool]] = MappingProxyType(
    {
        "execution_ready": False,
        "producer_schema_closure_complete": False,
        "production_ready": False,
        "qualification_ready": False,
    }
)
AUTHORITY: Final[Mapping[str, bool]] = MappingProxyType(
    {
        "execution_authorized": False,
        "publication_authorized": False,
        "qualification_granted": False,
        "receipt_endorsed": False,
    }
)
CLAIMS: Final[Mapping[str, bool]] = MappingProxyType(
    {
        "nonstorage_channel_proven": False,
        "performance_claim_allowed": False,
        "promotion_allowed": False,
        "scientific_evidence_created": False,
    }
)
PUBLIC_CANONICAL_BUILDERS: Final = (
    "build_nonstorage_channel_descriptor_file_v1",
    "build_nonstorage_channel_preseal_attestation_file_v1",
    "build_storage_receipt_identity_frame_v1",
    "canonical_nonstorage_channel_descriptor_v1_body_bytes",
    "canonical_nonstorage_channel_preseal_attestation_v1_body_bytes",
    "nonstorage_channel_commitment_sha256_v1",
    "nonstorage_channel_descriptor_identity_v1",
    "nonstorage_channel_preseal_attestation_identity_v1",
)
PUBLIC_PARSERS: Final = (
    "parse_nonstorage_channel_descriptor_file_v1",
    "parse_nonstorage_channel_preseal_attestation_file_v1",
)
PUBLIC_VALIDATORS: Final = (
    "validate_nonstorage_channel_dependency_v1",
    "validate_nonstorage_channel_descriptor_binding_v1",
)


class ForagerMatchedV3StorageNonstorageChannelProtocolV1Error(ValueError):
    """One nonstorage-channel protocol record failed closed."""


def _fail(message: str) -> Never:
    raise ForagerMatchedV3StorageNonstorageChannelProtocolV1Error(message)


def _require_identifier(value: object, label: str) -> str:
    if type(value) is not str or _IDENTIFIER_RE.fullmatch(value) is None:
        _fail(f"{label} must be one bounded portable identifier")
    return value


def _require_container_name(value: object) -> str:
    if type(value) is not str or _CONTAINER_NAME_RE.fullmatch(value) is None:
        _fail("container name must be one bounded portable name")
    return value


def _require_image_id(value: object) -> str:
    if type(value) is not str or _IMAGE_ID_RE.fullmatch(value) is None:
        _fail("image ID must be one exact sha256 runtime identity")
    return value


def _require_exact_int(
    value: object,
    label: str,
    *,
    minimum: int = 0,
    maximum: int = _MAX_EXACT_INTEGER,
) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        _fail(f"{label} must be one bounded exact integer")
    return value


def _require_exact_dict(value: object, keys: frozenset[str], label: str) -> dict[str, Any]:
    if type(value) is not dict or frozenset(value) != keys:
        _fail(f"{label} keys differ")
    return cast(dict[str, Any], value)


def _require_case_subject(value: object, label: str) -> CaseSubjectV1:
    if type(value) is not CaseSubjectV1:
        _fail(f"{label} type differs")
    try:
        return case_subject_v1_from_dict(value.to_dict())
    except ValueError as exc:
        _fail(f"{label} differs: {exc}")


def _require_artifact(value: object, schema: str, label: str) -> ArtifactRefV1:
    if type(value) is not ArtifactRefV1:
        _fail(f"{label} identity differs")
    try:
        exact = artifact_ref_v1_from_dict(value.to_dict())
    except ValueError as exc:
        _fail(f"{label} identity differs: {exc}")
    if exact.schema_version != schema:
        _fail(f"{label} identity differs")
    return exact


def _require_channel_producer(value: object) -> ProducerRefV1:
    if type(value) is not ProducerRefV1:
        _fail("nonstorage channel producer role differs")
    try:
        exact = producer_ref_v1_from_dict(value.to_dict())
    except ValueError as exc:
        _fail(f"nonstorage channel producer differs: {exc}")
    if exact.role != NONSTORAGE_CHANNEL_ROLE:
        _fail("nonstorage channel producer role differs")
    if exact.descriptor_schema_version != NONSTORAGE_CHANNEL_DESCRIPTOR_SCHEMA_VERSION:
        _fail("nonstorage channel producer descriptor schema differs")
    return exact


def _require_terminal_producer(value: object) -> ProducerRefV1:
    if type(value) is not ProducerRefV1:
        _fail("terminal relay producer role differs")
    try:
        exact = producer_ref_v1_from_dict(value.to_dict())
    except ValueError as exc:
        _fail(f"terminal relay producer differs: {exc}")
    if exact.role != TERMINAL_RELAY_ROLE:
        _fail("terminal relay producer role differs")
    if exact.descriptor_schema_version != TERMINAL_RELAY_DESCRIPTOR_SCHEMA_VERSION:
        _fail("terminal relay producer descriptor schema differs")
    return exact


def _case_subject_from_dict(value: object, label: str) -> CaseSubjectV1:
    try:
        parsed = case_subject_v1_from_dict(value)
    except ValueError as exc:
        _fail(f"{label} differs: {exc}")
    return _require_case_subject(parsed, label)


def _artifact_from_dict(value: object, schema: str, label: str) -> ArtifactRefV1:
    try:
        parsed = artifact_ref_v1_from_dict(value)
    except ValueError as exc:
        _fail(f"{label} identity differs: {exc}")
    return _require_artifact(parsed, schema, label)


def _channel_producer_from_dict(value: object) -> ProducerRefV1:
    try:
        parsed = producer_ref_v1_from_dict(value)
    except ValueError as exc:
        _fail(f"nonstorage channel producer differs: {exc}")
    return _require_channel_producer(parsed)


def _terminal_producer_from_dict(value: object) -> ProducerRefV1:
    try:
        parsed = producer_ref_v1_from_dict(value)
    except ValueError as exc:
        _fail(f"terminal relay producer differs: {exc}")
    return _require_terminal_producer(parsed)


def _artifact_identity_values(*artifacts: ArtifactRefV1) -> tuple[str, ...]:
    return tuple(
        identity
        for artifact in artifacts
        for identity in (artifact.file_sha256, artifact.body_sha256)
    )


def _seal_subclasses(name: str) -> Never:
    raise TypeError(f"{name} is runtime-sealed against subclasses")


def _validate_channel_configuration_v1(
    *,
    campaign_id: object,
    case_subject: object,
    runtime_intent: object,
    host_go: object,
    image_id: object,
    container_name: object,
    container_id_commitment_sha256: object,
    outer_cgroup_identity_sha256: object,
    producer: object,
    terminal_relay_producer: object,
    channel_kind: object,
    relay_endpoint_count: object,
    trusted_host_endpoint_count: object,
    candidate_endpoint_count: object,
    filesystem_path_present: object,
    filesystem_backing_present: object,
    close_on_exec: object,
    passed_to_candidate: object,
    candidate_accessible: object,
    can_allocate_measured_storage: object,
    frame_policy: object,
    maximum_frame_bytes: object,
) -> tuple[str, ...]:
    """Validate and return every digest in one pre-relay channel configuration."""

    _require_identifier(campaign_id, "campaign ID")
    _require_case_subject(case_subject, "case subject")
    exact_runtime_intent = _require_artifact(
        runtime_intent,
        STORAGE_BOUNDARY_RUNTIME_INTENT_SCHEMA_VERSION,
        "storage runtime intent",
    )
    exact_host_go = _require_artifact(host_go, HOST_GO_V3_SCHEMA_VERSION, "host GO")
    _require_image_id(image_id)
    _require_container_name(container_name)
    container_commitment = require_nonzero_sha256(
        container_id_commitment_sha256,
        "container identity commitment",
    )
    outer_cgroup = require_nonzero_sha256(
        outer_cgroup_identity_sha256,
        "outer cgroup identity",
    )
    channel_producer = _require_channel_producer(producer)
    terminal_producer = _require_terminal_producer(terminal_relay_producer)
    if (
        type(channel_kind) is not str
        or channel_kind != CHANNEL_KIND
        or _require_exact_int(
            relay_endpoint_count,
            "relay endpoint count",
            minimum=RELAY_ENDPOINT_COUNT,
            maximum=RELAY_ENDPOINT_COUNT,
        )
        != RELAY_ENDPOINT_COUNT
        or _require_exact_int(
            trusted_host_endpoint_count,
            "trusted host endpoint count",
            minimum=TRUSTED_HOST_ENDPOINT_COUNT,
            maximum=TRUSTED_HOST_ENDPOINT_COUNT,
        )
        != TRUSTED_HOST_ENDPOINT_COUNT
        or _require_exact_int(
            candidate_endpoint_count,
            "candidate endpoint count",
            minimum=CANDIDATE_ENDPOINT_COUNT,
            maximum=CANDIDATE_ENDPOINT_COUNT,
        )
        != CANDIDATE_ENDPOINT_COUNT
        or filesystem_path_present is not False
        or filesystem_backing_present is not False
        or close_on_exec is not True
        or passed_to_candidate is not False
        or candidate_accessible is not False
        or can_allocate_measured_storage is not False
        or type(frame_policy) is not str
        or frame_policy != FRAME_POLICY
        or _require_exact_int(
            maximum_frame_bytes,
            "maximum frame bytes",
            minimum=MAXIMUM_FRAME_BYTES,
            maximum=MAXIMUM_FRAME_BYTES,
        )
        != MAXIMUM_FRAME_BYTES
    ):
        _fail("nonstorage channel topology, confinement, or frame policy differs")
    return require_distinct_sha256s(
        (
            *_artifact_identity_values(exact_runtime_intent, exact_host_go),
            channel_producer.descriptor_file_sha256,
            channel_producer.descriptor_body_sha256,
            channel_producer.source_sha256,
            terminal_producer.descriptor_file_sha256,
            terminal_producer.descriptor_body_sha256,
            terminal_producer.source_sha256,
            container_commitment,
            outer_cgroup,
        ),
        "nonstorage channel configuration identities",
    )


def _channel_commitment_body_v1(
    *,
    campaign_id: str,
    case_subject: CaseSubjectV1,
    runtime_intent: ArtifactRefV1,
    host_go: ArtifactRefV1,
    image_id: str,
    container_name: str,
    container_id_commitment_sha256: str,
    outer_cgroup_identity_sha256: str,
    producer: ProducerRefV1,
    terminal_relay_producer: ProducerRefV1,
    channel_kind: str,
    relay_endpoint_count: int,
    trusted_host_endpoint_count: int,
    candidate_endpoint_count: int,
    filesystem_path_present: bool,
    filesystem_backing_present: bool,
    close_on_exec: bool,
    passed_to_candidate: bool,
    candidate_accessible: bool,
    can_allocate_measured_storage: bool,
    frame_policy: str,
    maximum_frame_bytes: int,
) -> dict[str, Any]:
    return {
        "campaign_id": campaign_id,
        "candidate_accessible": candidate_accessible,
        "candidate_endpoint_count": candidate_endpoint_count,
        "can_allocate_measured_storage": can_allocate_measured_storage,
        "case_subject": case_subject.to_dict(),
        "channel_kind": channel_kind,
        "close_on_exec": close_on_exec,
        "container_id_commitment_sha256": container_id_commitment_sha256,
        "container_name": container_name,
        "filesystem_backing_present": filesystem_backing_present,
        "filesystem_path_present": filesystem_path_present,
        "frame_policy": frame_policy,
        "host_go": host_go.to_dict(),
        "image_id": image_id,
        "maximum_frame_bytes": maximum_frame_bytes,
        "outer_cgroup_identity_sha256": outer_cgroup_identity_sha256,
        "passed_to_candidate": passed_to_candidate,
        "producer": producer.to_dict(),
        "relay_endpoint_count": relay_endpoint_count,
        "runtime_intent": runtime_intent.to_dict(),
        "terminal_relay_producer": terminal_relay_producer.to_dict(),
        "trusted_host_endpoint_count": trusted_host_endpoint_count,
    }


def nonstorage_channel_commitment_sha256_v1(
    *,
    campaign_id: str,
    case_subject: CaseSubjectV1,
    runtime_intent: ArtifactRefV1,
    host_go: ArtifactRefV1,
    image_id: str,
    container_name: str,
    container_id_commitment_sha256: str,
    outer_cgroup_identity_sha256: str,
    producer: ProducerRefV1,
    terminal_relay_producer: ProducerRefV1,
    channel_kind: str = CHANNEL_KIND,
    relay_endpoint_count: int = RELAY_ENDPOINT_COUNT,
    trusted_host_endpoint_count: int = TRUSTED_HOST_ENDPOINT_COUNT,
    candidate_endpoint_count: int = CANDIDATE_ENDPOINT_COUNT,
    filesystem_path_present: bool = False,
    filesystem_backing_present: bool = False,
    close_on_exec: bool = True,
    passed_to_candidate: bool = False,
    candidate_accessible: bool = False,
    can_allocate_measured_storage: bool = False,
    frame_policy: str = FRAME_POLICY,
    maximum_frame_bytes: int = MAXIMUM_FRAME_BYTES,
) -> str:
    """Commit pre-relay channel configuration without a future relay identity."""

    configuration_identities = _validate_channel_configuration_v1(
        campaign_id=campaign_id,
        case_subject=case_subject,
        runtime_intent=runtime_intent,
        host_go=host_go,
        image_id=image_id,
        container_name=container_name,
        container_id_commitment_sha256=container_id_commitment_sha256,
        outer_cgroup_identity_sha256=outer_cgroup_identity_sha256,
        producer=producer,
        terminal_relay_producer=terminal_relay_producer,
        channel_kind=channel_kind,
        relay_endpoint_count=relay_endpoint_count,
        trusted_host_endpoint_count=trusted_host_endpoint_count,
        candidate_endpoint_count=candidate_endpoint_count,
        filesystem_path_present=filesystem_path_present,
        filesystem_backing_present=filesystem_backing_present,
        close_on_exec=close_on_exec,
        passed_to_candidate=passed_to_candidate,
        candidate_accessible=candidate_accessible,
        can_allocate_measured_storage=can_allocate_measured_storage,
        frame_policy=frame_policy,
        maximum_frame_bytes=maximum_frame_bytes,
    )
    body = _channel_commitment_body_v1(
        campaign_id=campaign_id,
        case_subject=case_subject,
        runtime_intent=runtime_intent,
        host_go=host_go,
        image_id=image_id,
        container_name=container_name,
        container_id_commitment_sha256=container_id_commitment_sha256,
        outer_cgroup_identity_sha256=outer_cgroup_identity_sha256,
        producer=producer,
        terminal_relay_producer=terminal_relay_producer,
        channel_kind=channel_kind,
        relay_endpoint_count=relay_endpoint_count,
        trusted_host_endpoint_count=trusted_host_endpoint_count,
        candidate_endpoint_count=candidate_endpoint_count,
        filesystem_path_present=filesystem_path_present,
        filesystem_backing_present=filesystem_backing_present,
        close_on_exec=close_on_exec,
        passed_to_candidate=passed_to_candidate,
        candidate_accessible=candidate_accessible,
        can_allocate_measured_storage=can_allocate_measured_storage,
        frame_policy=frame_policy,
        maximum_frame_bytes=maximum_frame_bytes,
    )
    commitment = hashlib.sha256(
        _CHANNEL_COMMITMENT_DOMAIN + canonical_json_bytes(body, final_lf=False)
    ).hexdigest()
    require_distinct_sha256s(
        (*configuration_identities, commitment),
        "nonstorage channel commitment identities",
    )
    return commitment


@dataclass(frozen=True, slots=True)
class NonstorageChannelDescriptorV1:
    """Nonoperational descriptor for the nonstorage-channel schema provider."""

    schema_version: str = NONSTORAGE_CHANNEL_DESCRIPTOR_SCHEMA_VERSION
    status: str = NONSTORAGE_CHANNEL_DESCRIPTOR_STATUS
    role: str = NONSTORAGE_CHANNEL_ROLE
    owned_artifact_schemas: tuple[str, ...] = NONSTORAGE_CHANNEL_OWNED_ARTIFACT_SCHEMAS
    required_input_schemas: tuple[str, ...] = NONSTORAGE_CHANNEL_REQUIRED_INPUT_SCHEMAS
    canonical_policy: tuple[str, ...] = CANONICAL_POLICY
    limits: Mapping[str, int] = LIMITS
    capabilities: Mapping[str, bool] = CAPABILITIES
    readiness: Mapping[str, bool] = READINESS
    authority: Mapping[str, bool] = AUTHORITY
    claims: Mapping[str, bool] = CLAIMS
    public_canonical_builders: tuple[str, ...] = PUBLIC_CANONICAL_BUILDERS
    public_parsers: tuple[str, ...] = PUBLIC_PARSERS
    public_validators: tuple[str, ...] = PUBLIC_VALIDATORS
    operational_apis: tuple[()] = ()
    descriptor_self_pin_sha256: str = _ZERO_SHA256
    source_file_sha256_pin: None = None

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not str
            or self.schema_version != NONSTORAGE_CHANNEL_DESCRIPTOR_SCHEMA_VERSION
        ):
            _fail("nonstorage channel descriptor schema differs")
        if (
            type(self.status) is not str
            or self.status != NONSTORAGE_CHANNEL_DESCRIPTOR_STATUS
            or type(self.role) is not str
            or self.role != NONSTORAGE_CHANNEL_ROLE
        ):
            _fail("nonstorage channel descriptor status or role differs")
        if (
            type(self.owned_artifact_schemas) is not tuple
            or self.owned_artifact_schemas != NONSTORAGE_CHANNEL_OWNED_ARTIFACT_SCHEMAS
            or any(type(value) is not str for value in self.owned_artifact_schemas)
            or type(self.required_input_schemas) is not tuple
            or self.required_input_schemas != NONSTORAGE_CHANNEL_REQUIRED_INPUT_SCHEMAS
            or any(type(value) is not str for value in self.required_input_schemas)
            or type(self.canonical_policy) is not tuple
            or self.canonical_policy != CANONICAL_POLICY
            or any(type(value) is not str for value in self.canonical_policy)
        ):
            _fail("nonstorage channel descriptor schema inventory or canonical policy differs")
        for field_name, supplied, expected in (
            ("limits", self.limits, LIMITS),
            ("capabilities", self.capabilities, CAPABILITIES),
            ("readiness", self.readiness, READINESS),
            ("authority", self.authority, AUTHORITY),
            ("claims", self.claims, CLAIMS),
        ):
            if (
                type(supplied) not in {dict, type(LIMITS)}
                or dict(supplied) != dict(expected)
                or any(type(key) is not str for key in supplied)
            ):
                _fail(f"nonstorage channel descriptor {field_name} differs")
            if field_name == "limits" and any(
                type(value) is not int for value in supplied.values()
            ):
                _fail("nonstorage channel descriptor limits must be exact integers")
            object.__setattr__(self, field_name, MappingProxyType(dict(supplied)))
        for posture_name, posture in (
            ("capabilities", self.capabilities),
            ("readiness", self.readiness),
            ("authority", self.authority),
            ("claims", self.claims),
        ):
            if any(type(value) is not bool or value is not False for value in posture.values()):
                _fail(f"nonstorage channel descriptor {posture_name} must be all false")
        if (
            type(self.public_canonical_builders) is not tuple
            or self.public_canonical_builders != PUBLIC_CANONICAL_BUILDERS
            or any(type(value) is not str for value in self.public_canonical_builders)
            or type(self.public_parsers) is not tuple
            or self.public_parsers != PUBLIC_PARSERS
            or any(type(value) is not str for value in self.public_parsers)
            or type(self.public_validators) is not tuple
            or self.public_validators != PUBLIC_VALIDATORS
            or any(type(value) is not str for value in self.public_validators)
        ):
            _fail("nonstorage channel descriptor public API inventory differs")
        if type(self.operational_apis) is not tuple or self.operational_apis != ():
            _fail("nonstorage channel descriptor operational APIs must be empty")
        if (
            type(self.descriptor_self_pin_sha256) is not str
            or self.descriptor_self_pin_sha256 != _ZERO_SHA256
            or self.source_file_sha256_pin is not None
        ):
            _fail("nonstorage channel descriptor repository pins must remain unfinalized")

    @classmethod
    def __init_subclass__(cls, **kwargs: object) -> None:
        _seal_subclasses(cls.__name__)

    def to_body_dict(self) -> dict[str, Any]:
        return {
            "authority": dict(self.authority),
            "canonical_policy": list(self.canonical_policy),
            "capabilities": dict(self.capabilities),
            "claims": dict(self.claims),
            "descriptor_self_pin_sha256": self.descriptor_self_pin_sha256,
            "limits": dict(self.limits),
            "operational_apis": list(self.operational_apis),
            "owned_artifact_schemas": list(self.owned_artifact_schemas),
            "public_canonical_builders": list(self.public_canonical_builders),
            "public_parsers": list(self.public_parsers),
            "public_validators": list(self.public_validators),
            "readiness": dict(self.readiness),
            "required_input_schemas": list(self.required_input_schemas),
            "role": self.role,
            "schema_version": self.schema_version,
            "source_file_sha256_pin": self.source_file_sha256_pin,
            "status": self.status,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class NonstorageChannelPresealAttestationV1:
    """Pre-seal channel readiness bound to an already-created relay attestation."""

    schema_version: str = NONSTORAGE_CHANNEL_PRESEAL_ATTESTATION_SCHEMA_VERSION
    status: str = NONSTORAGE_CHANNEL_PRESEAL_STATUS
    campaign_id: str
    case_subject: CaseSubjectV1
    runtime_intent: ArtifactRefV1
    host_go: ArtifactRefV1
    image_id: str
    container_name: str
    container_id_commitment_sha256: str
    outer_cgroup_identity_sha256: str
    producer: ProducerRefV1
    terminal_relay_preseal_attestation: ArtifactRefV1
    terminal_relay_producer: ProducerRefV1
    channel_commitment_sha256: str
    channel_kind: str = CHANNEL_KIND
    relay_endpoint_count: int = RELAY_ENDPOINT_COUNT
    trusted_host_endpoint_count: int = TRUSTED_HOST_ENDPOINT_COUNT
    candidate_endpoint_count: int = CANDIDATE_ENDPOINT_COUNT
    filesystem_path_present: bool = False
    filesystem_backing_present: bool = False
    close_on_exec: bool = True
    passed_to_candidate: bool = False
    candidate_accessible: bool = False
    can_allocate_measured_storage: bool = False
    frame_policy: str = FRAME_POLICY
    maximum_frame_bytes: int = MAXIMUM_FRAME_BYTES
    ready_for_post_receipt_terminal: bool = True
    terminal_emission_performed: bool = False
    attestation_monotonic_ns: int

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not str
            or self.schema_version != NONSTORAGE_CHANNEL_PRESEAL_ATTESTATION_SCHEMA_VERSION
        ):
            _fail("nonstorage channel attestation schema differs")
        if type(self.status) is not str or self.status != NONSTORAGE_CHANNEL_PRESEAL_STATUS:
            _fail("nonstorage channel attestation status differs")
        configuration_identities = _validate_channel_configuration_v1(
            campaign_id=self.campaign_id,
            case_subject=self.case_subject,
            runtime_intent=self.runtime_intent,
            host_go=self.host_go,
            image_id=self.image_id,
            container_name=self.container_name,
            container_id_commitment_sha256=self.container_id_commitment_sha256,
            outer_cgroup_identity_sha256=self.outer_cgroup_identity_sha256,
            producer=self.producer,
            terminal_relay_producer=self.terminal_relay_producer,
            channel_kind=self.channel_kind,
            relay_endpoint_count=self.relay_endpoint_count,
            trusted_host_endpoint_count=self.trusted_host_endpoint_count,
            candidate_endpoint_count=self.candidate_endpoint_count,
            filesystem_path_present=self.filesystem_path_present,
            filesystem_backing_present=self.filesystem_backing_present,
            close_on_exec=self.close_on_exec,
            passed_to_candidate=self.passed_to_candidate,
            candidate_accessible=self.candidate_accessible,
            can_allocate_measured_storage=self.can_allocate_measured_storage,
            frame_policy=self.frame_policy,
            maximum_frame_bytes=self.maximum_frame_bytes,
        )
        relay_attestation = _require_artifact(
            self.terminal_relay_preseal_attestation,
            TERMINAL_RELAY_PRESEAL_ATTESTATION_SCHEMA_VERSION,
            "terminal relay pre-seal attestation",
        )
        channel_commitment = require_nonzero_sha256(
            self.channel_commitment_sha256,
            "nonstorage channel commitment",
        )
        require_distinct_sha256s(
            (
                *configuration_identities,
                *_artifact_identity_values(relay_attestation),
                channel_commitment,
            ),
            "nonstorage channel run identities",
        )
        if (
            self.ready_for_post_receipt_terminal is not True
            or self.terminal_emission_performed is not False
        ):
            _fail("nonstorage channel readiness or pre-seal emission posture differs")
        _require_exact_int(
            self.attestation_monotonic_ns,
            "nonstorage channel attestation monotonic timestamp",
            minimum=1,
        )
        observed_commitment = nonstorage_channel_commitment_sha256_v1(
            campaign_id=self.campaign_id,
            case_subject=self.case_subject,
            runtime_intent=self.runtime_intent,
            host_go=self.host_go,
            image_id=self.image_id,
            container_name=self.container_name,
            container_id_commitment_sha256=self.container_id_commitment_sha256,
            outer_cgroup_identity_sha256=self.outer_cgroup_identity_sha256,
            producer=self.producer,
            terminal_relay_producer=self.terminal_relay_producer,
            channel_kind=self.channel_kind,
            relay_endpoint_count=self.relay_endpoint_count,
            trusted_host_endpoint_count=self.trusted_host_endpoint_count,
            candidate_endpoint_count=self.candidate_endpoint_count,
            filesystem_path_present=self.filesystem_path_present,
            filesystem_backing_present=self.filesystem_backing_present,
            close_on_exec=self.close_on_exec,
            passed_to_candidate=self.passed_to_candidate,
            candidate_accessible=self.candidate_accessible,
            can_allocate_measured_storage=self.can_allocate_measured_storage,
            frame_policy=self.frame_policy,
            maximum_frame_bytes=self.maximum_frame_bytes,
        )
        if not hmac.compare_digest(channel_commitment, observed_commitment):
            _fail("nonstorage channel commitment differs from its acyclic configuration")

    @classmethod
    def __init_subclass__(cls, **kwargs: object) -> None:
        _seal_subclasses(cls.__name__)

    def to_body_dict(self) -> dict[str, Any]:
        return {
            "attestation_monotonic_ns": self.attestation_monotonic_ns,
            "campaign_id": self.campaign_id,
            "candidate_accessible": self.candidate_accessible,
            "candidate_endpoint_count": self.candidate_endpoint_count,
            "can_allocate_measured_storage": self.can_allocate_measured_storage,
            "case_subject": self.case_subject.to_dict(),
            "channel_commitment_sha256": self.channel_commitment_sha256,
            "channel_kind": self.channel_kind,
            "close_on_exec": self.close_on_exec,
            "container_id_commitment_sha256": self.container_id_commitment_sha256,
            "container_name": self.container_name,
            "filesystem_backing_present": self.filesystem_backing_present,
            "filesystem_path_present": self.filesystem_path_present,
            "frame_policy": self.frame_policy,
            "host_go": self.host_go.to_dict(),
            "image_id": self.image_id,
            "maximum_frame_bytes": self.maximum_frame_bytes,
            "outer_cgroup_identity_sha256": self.outer_cgroup_identity_sha256,
            "passed_to_candidate": self.passed_to_candidate,
            "producer": self.producer.to_dict(),
            "ready_for_post_receipt_terminal": self.ready_for_post_receipt_terminal,
            "relay_endpoint_count": self.relay_endpoint_count,
            "runtime_intent": self.runtime_intent.to_dict(),
            "schema_version": self.schema_version,
            "status": self.status,
            "terminal_emission_performed": self.terminal_emission_performed,
            "terminal_relay_preseal_attestation": (
                self.terminal_relay_preseal_attestation.to_dict()
            ),
            "terminal_relay_producer": self.terminal_relay_producer.to_dict(),
            "trusted_host_endpoint_count": self.trusted_host_endpoint_count,
        }


def _validate_current_nonstorage_channel_descriptor_v1(
    descriptor: object,
) -> NonstorageChannelDescriptorV1:
    if type(descriptor) is not NonstorageChannelDescriptorV1:
        _fail("nonstorage channel descriptor type differs")
    NonstorageChannelDescriptorV1.__post_init__(descriptor)
    return descriptor


def _validate_current_nonstorage_channel_preseal_attestation_v1(
    artifact: object,
) -> NonstorageChannelPresealAttestationV1:
    if type(artifact) is not NonstorageChannelPresealAttestationV1:
        _fail("nonstorage channel attestation type differs")
    NonstorageChannelPresealAttestationV1.__post_init__(artifact)
    return artifact


def build_storage_receipt_identity_frame_v1(storage_receipt: ArtifactRefV1) -> bytes:
    """Build the only frame shape allowed by the frozen channel policy."""

    receipt = _require_artifact(
        storage_receipt,
        STORAGE_BOUNDARY_RECEIPT_V2_SCHEMA_VERSION,
        "storage receipt",
    )
    frame = canonical_json_bytes({"storage_receipt": receipt.to_dict()})
    if len(frame) > MAXIMUM_FRAME_BYTES:
        _fail("storage receipt identity frame exceeds its frozen bound")
    return frame


def build_nonstorage_channel_descriptor_file_v1(
    descriptor: NonstorageChannelDescriptorV1,
) -> bytes:
    """Serialize one exact nonstorage-channel descriptor."""

    _validate_current_nonstorage_channel_descriptor_v1(descriptor)
    return canonical_file_bytes(
        descriptor.to_body_dict(),
        body_digest_field=_DESCRIPTOR_BODY_FIELD,
    )


def canonical_nonstorage_channel_descriptor_v1_body_bytes(
    descriptor: NonstorageChannelDescriptorV1,
) -> bytes:
    """Return canonical unframed nonstorage-channel descriptor BODY bytes."""

    _validate_current_nonstorage_channel_descriptor_v1(descriptor)
    return canonical_json_bytes(descriptor.to_body_dict(), final_lf=False)


def build_nonstorage_channel_preseal_attestation_file_v1(
    artifact: NonstorageChannelPresealAttestationV1,
) -> bytes:
    """Serialize one exact nonstorage-channel pre-seal attestation."""

    _validate_current_nonstorage_channel_preseal_attestation_v1(artifact)
    return canonical_file_bytes(
        artifact.to_body_dict(),
        body_digest_field=_ATTESTATION_BODY_FIELD,
    )


def canonical_nonstorage_channel_preseal_attestation_v1_body_bytes(
    artifact: NonstorageChannelPresealAttestationV1,
) -> bytes:
    """Return canonical unframed nonstorage-channel attestation BODY bytes."""

    _validate_current_nonstorage_channel_preseal_attestation_v1(artifact)
    return canonical_json_bytes(artifact.to_body_dict(), final_lf=False)


def _tuple_of_strings(value: object, label: str) -> tuple[str, ...]:
    if type(value) is not list or any(type(item) is not str for item in value):
        _fail(f"{label} must be one exact string array")
    return tuple(cast(list[str], value))


def _mapping_of_exact_ints(value: object, label: str) -> dict[str, int]:
    if type(value) is not dict or any(
        type(key) is not str or type(item) is not int for key, item in value.items()
    ):
        _fail(f"{label} must be one exact integer object")
    return cast(dict[str, int], value)


def _mapping_of_exact_bools(value: object, label: str) -> dict[str, bool]:
    if type(value) is not dict or any(
        type(key) is not str or type(item) is not bool for key, item in value.items()
    ):
        _fail(f"{label} must be one exact boolean object")
    return cast(dict[str, bool], value)


def parse_nonstorage_channel_descriptor_file_v1(
    raw: bytes,
    *,
    expected_file_sha256: object,
    expected_body_sha256: object,
) -> NonstorageChannelDescriptorV1:
    """Parse a descriptor using independent caller-supplied FILE and BODY pins."""

    data = _require_exact_dict(
        validate_canonical_file(
            raw,
            expected_file_sha256=expected_file_sha256,
            expected_body_sha256=expected_body_sha256,
            body_digest_field=_DESCRIPTOR_BODY_FIELD,
        ),
        frozenset(
            {
                "authority",
                "canonical_policy",
                "capabilities",
                "claims",
                "descriptor_self_pin_sha256",
                "limits",
                "operational_apis",
                "owned_artifact_schemas",
                "public_canonical_builders",
                "public_parsers",
                "public_validators",
                "readiness",
                "required_input_schemas",
                "role",
                "schema_version",
                "source_file_sha256_pin",
                "status",
            }
        ),
        "nonstorage channel descriptor",
    )
    return NonstorageChannelDescriptorV1(
        schema_version=data["schema_version"],
        status=data["status"],
        role=data["role"],
        owned_artifact_schemas=_tuple_of_strings(
            data["owned_artifact_schemas"],
            "owned artifact schemas",
        ),
        required_input_schemas=_tuple_of_strings(
            data["required_input_schemas"],
            "required input schemas",
        ),
        canonical_policy=_tuple_of_strings(data["canonical_policy"], "canonical policy"),
        limits=_mapping_of_exact_ints(data["limits"], "limits"),
        capabilities=_mapping_of_exact_bools(data["capabilities"], "capabilities"),
        readiness=_mapping_of_exact_bools(data["readiness"], "readiness"),
        authority=_mapping_of_exact_bools(data["authority"], "authority"),
        claims=_mapping_of_exact_bools(data["claims"], "claims"),
        public_canonical_builders=_tuple_of_strings(
            data["public_canonical_builders"],
            "public canonical builders",
        ),
        public_parsers=_tuple_of_strings(data["public_parsers"], "public parsers"),
        public_validators=_tuple_of_strings(data["public_validators"], "public validators"),
        operational_apis=cast(
            tuple[()],
            _tuple_of_strings(data["operational_apis"], "operational APIs"),
        ),
        descriptor_self_pin_sha256=data["descriptor_self_pin_sha256"],
        source_file_sha256_pin=data["source_file_sha256_pin"],
    )


_ATTESTATION_KEYS: Final = frozenset(
    {
        "attestation_monotonic_ns",
        "campaign_id",
        "candidate_accessible",
        "candidate_endpoint_count",
        "can_allocate_measured_storage",
        "case_subject",
        "channel_commitment_sha256",
        "channel_kind",
        "close_on_exec",
        "container_id_commitment_sha256",
        "container_name",
        "filesystem_backing_present",
        "filesystem_path_present",
        "frame_policy",
        "host_go",
        "image_id",
        "maximum_frame_bytes",
        "outer_cgroup_identity_sha256",
        "passed_to_candidate",
        "producer",
        "ready_for_post_receipt_terminal",
        "relay_endpoint_count",
        "runtime_intent",
        "schema_version",
        "status",
        "terminal_emission_performed",
        "terminal_relay_preseal_attestation",
        "terminal_relay_producer",
        "trusted_host_endpoint_count",
    }
)


def parse_nonstorage_channel_preseal_attestation_file_v1(
    raw: bytes,
    *,
    expected_file_sha256: object,
    expected_body_sha256: object,
) -> NonstorageChannelPresealAttestationV1:
    """Parse one channel attestation under independent FILE and BODY pins."""

    data = _require_exact_dict(
        validate_canonical_file(
            raw,
            expected_file_sha256=expected_file_sha256,
            expected_body_sha256=expected_body_sha256,
            body_digest_field=_ATTESTATION_BODY_FIELD,
        ),
        _ATTESTATION_KEYS,
        "nonstorage channel attestation",
    )
    return NonstorageChannelPresealAttestationV1(
        campaign_id=data["campaign_id"],
        case_subject=_case_subject_from_dict(data["case_subject"], "case subject"),
        runtime_intent=_artifact_from_dict(
            data["runtime_intent"],
            STORAGE_BOUNDARY_RUNTIME_INTENT_SCHEMA_VERSION,
            "storage runtime intent",
        ),
        host_go=_artifact_from_dict(data["host_go"], HOST_GO_V3_SCHEMA_VERSION, "host GO"),
        image_id=data["image_id"],
        container_name=data["container_name"],
        container_id_commitment_sha256=data["container_id_commitment_sha256"],
        outer_cgroup_identity_sha256=data["outer_cgroup_identity_sha256"],
        producer=_channel_producer_from_dict(data["producer"]),
        terminal_relay_preseal_attestation=_artifact_from_dict(
            data["terminal_relay_preseal_attestation"],
            TERMINAL_RELAY_PRESEAL_ATTESTATION_SCHEMA_VERSION,
            "terminal relay pre-seal attestation",
        ),
        terminal_relay_producer=_terminal_producer_from_dict(data["terminal_relay_producer"]),
        channel_commitment_sha256=data["channel_commitment_sha256"],
        attestation_monotonic_ns=data["attestation_monotonic_ns"],
        schema_version=data["schema_version"],
        status=data["status"],
        channel_kind=data["channel_kind"],
        relay_endpoint_count=data["relay_endpoint_count"],
        trusted_host_endpoint_count=data["trusted_host_endpoint_count"],
        candidate_endpoint_count=data["candidate_endpoint_count"],
        filesystem_path_present=data["filesystem_path_present"],
        filesystem_backing_present=data["filesystem_backing_present"],
        close_on_exec=data["close_on_exec"],
        passed_to_candidate=data["passed_to_candidate"],
        candidate_accessible=data["candidate_accessible"],
        can_allocate_measured_storage=data["can_allocate_measured_storage"],
        frame_policy=data["frame_policy"],
        maximum_frame_bytes=data["maximum_frame_bytes"],
        ready_for_post_receipt_terminal=data["ready_for_post_receipt_terminal"],
        terminal_emission_performed=data["terminal_emission_performed"],
    )


def nonstorage_channel_descriptor_identity_v1(
    descriptor: NonstorageChannelDescriptorV1,
) -> ArtifactRefV1:
    """Derive canonical descriptor FILE and BODY identities in memory."""

    _validate_current_nonstorage_channel_descriptor_v1(descriptor)
    raw = build_nonstorage_channel_descriptor_file_v1(descriptor)
    return ArtifactRefV1(
        schema_version=NONSTORAGE_CHANNEL_DESCRIPTOR_SCHEMA_VERSION,
        file_sha256=hashlib.sha256(raw).hexdigest(),
        body_sha256=canonical_body_sha256(descriptor.to_body_dict()),
    )


def nonstorage_channel_preseal_attestation_identity_v1(
    artifact: NonstorageChannelPresealAttestationV1,
) -> ArtifactRefV1:
    """Derive canonical channel-attestation FILE and BODY identities in memory."""

    _validate_current_nonstorage_channel_preseal_attestation_v1(artifact)
    raw = build_nonstorage_channel_preseal_attestation_file_v1(artifact)
    return ArtifactRefV1(
        schema_version=NONSTORAGE_CHANNEL_PRESEAL_ATTESTATION_SCHEMA_VERSION,
        file_sha256=hashlib.sha256(raw).hexdigest(),
        body_sha256=canonical_body_sha256(artifact.to_body_dict()),
    )


def validate_nonstorage_channel_descriptor_binding_v1(
    descriptor: NonstorageChannelDescriptorV1,
    producer: ProducerRefV1,
) -> None:
    """Bind one exact channel producer to canonical descriptor identities."""

    _validate_current_nonstorage_channel_descriptor_v1(descriptor)
    exact_producer = _require_channel_producer(producer)
    identity = nonstorage_channel_descriptor_identity_v1(descriptor)
    if not hmac.compare_digest(
        exact_producer.descriptor_file_sha256, identity.file_sha256
    ) or not hmac.compare_digest(exact_producer.descriptor_body_sha256, identity.body_sha256):
        _fail("nonstorage channel producer differs from its descriptor identities")


def validate_nonstorage_channel_dependency_v1(
    channel: NonstorageChannelPresealAttestationV1,
    *,
    relay_attestation: ArtifactRefV1,
    relay_producer: ProducerRefV1,
    relay_campaign_id: object,
    relay_case_subject: object,
    relay_runtime_intent: ArtifactRefV1,
    relay_host_go: ArtifactRefV1,
    relay_image_id: object,
    relay_container_name: object,
    relay_container_id_commitment_sha256: object,
    relay_outer_cgroup_identity_sha256: object,
    relay_nonstorage_channel_commitment_sha256: object,
    relay_attestation_monotonic_ns: object,
) -> None:
    """Validate the typed relay projection without importing its owner module."""

    _validate_current_nonstorage_channel_preseal_attestation_v1(channel)
    exact_relay_artifact = _require_artifact(
        relay_attestation,
        TERMINAL_RELAY_PRESEAL_ATTESTATION_SCHEMA_VERSION,
        "terminal relay pre-seal attestation",
    )
    exact_relay_producer = _require_terminal_producer(relay_producer)
    exact_relay_campaign_id = _require_identifier(relay_campaign_id, "relay campaign ID")
    exact_relay_case_subject = _require_case_subject(
        relay_case_subject,
        "relay case subject",
    )
    exact_relay_runtime_intent = _require_artifact(
        relay_runtime_intent,
        STORAGE_BOUNDARY_RUNTIME_INTENT_SCHEMA_VERSION,
        "relay storage runtime intent",
    )
    exact_relay_host_go = _require_artifact(
        relay_host_go,
        HOST_GO_V3_SCHEMA_VERSION,
        "relay host GO",
    )
    exact_relay_image_id = _require_image_id(relay_image_id)
    exact_relay_container_name = _require_container_name(relay_container_name)
    exact_relay_container_commitment = require_nonzero_sha256(
        relay_container_id_commitment_sha256,
        "relay container identity commitment",
    )
    exact_relay_outer_cgroup = require_nonzero_sha256(
        relay_outer_cgroup_identity_sha256,
        "relay outer cgroup identity",
    )
    relay_commitment = require_nonzero_sha256(
        relay_nonstorage_channel_commitment_sha256,
        "relay nonstorage channel commitment",
    )
    relay_time = _require_exact_int(
        relay_attestation_monotonic_ns,
        "relay attestation monotonic timestamp",
        minimum=1,
    )
    relay_context = (
        exact_relay_campaign_id,
        exact_relay_case_subject,
        exact_relay_runtime_intent,
        exact_relay_host_go,
        exact_relay_image_id,
        exact_relay_container_name,
        exact_relay_container_commitment,
        exact_relay_outer_cgroup,
    )
    channel_context = (
        channel.campaign_id,
        channel.case_subject,
        channel.runtime_intent,
        channel.host_go,
        channel.image_id,
        channel.container_name,
        channel.container_id_commitment_sha256,
        channel.outer_cgroup_identity_sha256,
    )
    if relay_context != channel_context:
        _fail("terminal relay and nonstorage channel run contexts differ")
    if (
        channel.terminal_relay_preseal_attestation != exact_relay_artifact
        or channel.terminal_relay_producer != exact_relay_producer
        or not hmac.compare_digest(channel.channel_commitment_sha256, relay_commitment)
    ):
        _fail("nonstorage channel relay identity, producer, or commitment differs")
    if not relay_time < channel.attestation_monotonic_ns:
        _fail("relay attestation must strictly precede channel attestation")


if any(
    pin != _ZERO_SHA256
    for pin in (
        PINNED_NONSTORAGE_CHANNEL_DESCRIPTOR_FILE_SHA256,
        PINNED_NONSTORAGE_CHANNEL_DESCRIPTOR_BODY_SHA256,
        PINNED_NONSTORAGE_CHANNEL_SOURCE_SHA256,
    )
):
    raise AssertionError("nonstorage channel repository pins must remain unfinalized")
if set(NONSTORAGE_CHANNEL_OWNED_ARTIFACT_SCHEMAS).intersection(
    NONSTORAGE_CHANNEL_REQUIRED_INPUT_SCHEMAS
):
    raise AssertionError("nonstorage channel owned and input schemas must be disjoint")


__all__ = [
    "AUTHORITY",
    "ArtifactRefV1",
    "CANDIDATE_ENDPOINT_COUNT",
    "CANONICAL_POLICY",
    "CAPABILITIES",
    "CHANNEL_KIND",
    "CLAIMS",
    "FRAME_POLICY",
    "ForagerMatchedV3StorageNonstorageChannelProtocolV1Error",
    "HOST_GO_V3_SCHEMA_VERSION",
    "LIMITS",
    "MAXIMUM_FRAME_BYTES",
    "NONSTORAGE_CHANNEL_DESCRIPTOR_SCHEMA_VERSION",
    "NONSTORAGE_CHANNEL_DESCRIPTOR_STATUS",
    "NONSTORAGE_CHANNEL_OWNED_ARTIFACT_SCHEMAS",
    "NONSTORAGE_CHANNEL_PRESEAL_ATTESTATION_SCHEMA_VERSION",
    "NONSTORAGE_CHANNEL_PRESEAL_STATUS",
    "NONSTORAGE_CHANNEL_REQUIRED_INPUT_SCHEMAS",
    "NONSTORAGE_CHANNEL_ROLE",
    "NonstorageChannelDescriptorV1",
    "NonstorageChannelPresealAttestationV1",
    "PINNED_NONSTORAGE_CHANNEL_DESCRIPTOR_BODY_SHA256",
    "PINNED_NONSTORAGE_CHANNEL_DESCRIPTOR_FILE_SHA256",
    "PINNED_NONSTORAGE_CHANNEL_SOURCE_SHA256",
    "PUBLIC_CANONICAL_BUILDERS",
    "PUBLIC_PARSERS",
    "PUBLIC_VALIDATORS",
    "ProducerRefV1",
    "READINESS",
    "RELAY_ENDPOINT_COUNT",
    "STORAGE_BOUNDARY_RECEIPT_V2_SCHEMA_VERSION",
    "STORAGE_BOUNDARY_RUNTIME_INTENT_SCHEMA_VERSION",
    "TERMINAL_RELAY_DESCRIPTOR_SCHEMA_VERSION",
    "TERMINAL_RELAY_PRESEAL_ATTESTATION_SCHEMA_VERSION",
    "TERMINAL_RELAY_ROLE",
    "TRUSTED_HOST_ENDPOINT_COUNT",
    "build_nonstorage_channel_descriptor_file_v1",
    "build_nonstorage_channel_preseal_attestation_file_v1",
    "build_storage_receipt_identity_frame_v1",
    "canonical_nonstorage_channel_descriptor_v1_body_bytes",
    "canonical_nonstorage_channel_preseal_attestation_v1_body_bytes",
    "nonstorage_channel_commitment_sha256_v1",
    "nonstorage_channel_descriptor_identity_v1",
    "nonstorage_channel_preseal_attestation_identity_v1",
    "parse_nonstorage_channel_descriptor_file_v1",
    "parse_nonstorage_channel_preseal_attestation_file_v1",
    "validate_nonstorage_channel_dependency_v1",
    "validate_nonstorage_channel_descriptor_binding_v1",
]
