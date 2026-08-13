"""Pure terminal-relay evidence schemas for matched Forager v3 storage.

This module only validates caller-supplied canonical metadata and byte captures.
It does not observe a host, read or write a file, create a transport, relay a
terminal artifact, or authorize execution.  The host-GO artifact is an exact
structural predecessor; its numeric GO-to-worker chronology remains the host
provider linker's responsibility because this schema carries no duplicate GO
timestamp.  Reload-observation bytes are caller-supplied canonical content;
this metadata-only provider can bind their digest but cannot establish that an
operational reload produced them.  The typed publication linker must compare
the wrapper and reload-validation commitments independently.
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
    RawByteCaptureV1,
    artifact_ref_v1_from_dict,
    canonical_body_sha256,
    canonical_file_bytes,
    canonical_json_bytes,
    case_subject_v1_from_dict,
    decode_canonical_json_file,
    producer_ref_v1_from_dict,
    raw_byte_capture_v1_from_dict,
    require_distinct_sha256s,
    require_nonzero_sha256,
    validate_canonical_file,
)

TERMINAL_RELAY_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.qualification_storage_terminal_relay_descriptor.v1"
)
RAW_PUBLICATION_RELOAD_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.raw.publication_reload.v1"
)
TERMINAL_RELAY_PRESEAL_ATTESTATION_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.qualification_storage_terminal_relay_preseal_attestation.v1"
)

STORAGE_BOUNDARY_RUNTIME_INTENT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.storage_boundary_runtime_intent.v1"
)
HOST_GO_V3_SCHEMA_VERSION: Final = "alberta.forager_matched_v3.host_qualification_go_commitment.v3"
NORMALIZED_PUBLICATION_WRAPPER_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.qualification_publication_commitment_wrapper.v1"
)
PUBLICATION_RELOAD_VALIDATION_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.qualification_publication_reload_validation.v1"
)

TERMINAL_RELAY_ROLE: Final = "terminal_relay"
TERMINAL_RELAY_DESCRIPTOR_STATUS: Final = (
    "pure_source_only_unfinalized_uninvoked_no_production_artifact"
)
RAW_PUBLICATION_RELOAD_STATUS: Final = "post_reload_raw_capture_non_authorizing"
TERMINAL_RELAY_PRESEAL_STATUS: Final = "pre_seal_terminal_relay_attestation_non_authorizing"
TERMINAL_RELAY_INPUT_POLICY: Final = "exact_committed_storage_receipt_identity_only"

_ZERO_SHA256: Final = "0" * 64
PINNED_TERMINAL_RELAY_DESCRIPTOR_FILE_SHA256: Final = _ZERO_SHA256
PINNED_TERMINAL_RELAY_DESCRIPTOR_BODY_SHA256: Final = _ZERO_SHA256
PINNED_TERMINAL_RELAY_SOURCE_SHA256: Final = _ZERO_SHA256

_DESCRIPTOR_BODY_FIELD: Final = "terminal_relay_descriptor_body_sha256"
_RAW_RELOAD_BODY_FIELD: Final = "raw_publication_reload_body_sha256"
_RELAY_ATTESTATION_BODY_FIELD: Final = "terminal_relay_preseal_attestation_body_sha256"

_IDENTIFIER_RE: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}\Z")
_CONTAINER_NAME_RE: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_IMAGE_ID_RE: Final = re.compile(r"sha256:[0-9a-f]{64}\Z")
_MAX_EXACT_INTEGER: Final = (1 << 63) - 1

TERMINAL_RELAY_OWNED_ARTIFACT_SCHEMAS: Final = (
    RAW_PUBLICATION_RELOAD_SCHEMA_VERSION,
    TERMINAL_RELAY_PRESEAL_ATTESTATION_SCHEMA_VERSION,
)
TERMINAL_RELAY_REQUIRED_INPUT_SCHEMAS: Final = (
    STORAGE_BOUNDARY_RUNTIME_INTENT_SCHEMA_VERSION,
    HOST_GO_V3_SCHEMA_VERSION,
    NORMALIZED_PUBLICATION_WRAPPER_SCHEMA_VERSION,
    PUBLICATION_RELOAD_VALIDATION_SCHEMA_VERSION,
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
    }
)
CAPABILITIES: Final[Mapping[str, bool]] = MappingProxyType(
    {
        "container_control": False,
        "filesystem_access": False,
        "network_access": False,
        "process_control": False,
        "terminal_relay_execution": False,
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
        "performance_claim_allowed": False,
        "promotion_allowed": False,
        "scientific_evidence_created": False,
        "terminal_relay_proven": False,
    }
)
PUBLIC_CANONICAL_BUILDERS: Final = (
    "build_raw_publication_reload_file_v1",
    "build_terminal_relay_descriptor_file_v1",
    "build_terminal_relay_preseal_attestation_file_v1",
    "build_publication_reload_observation_capture_v1",
    "canonical_raw_publication_reload_v1_body_bytes",
    "canonical_terminal_relay_descriptor_v1_body_bytes",
    "canonical_terminal_relay_preseal_attestation_v1_body_bytes",
    "raw_publication_reload_identity_v1",
    "terminal_relay_descriptor_identity_v1",
    "terminal_relay_preseal_attestation_identity_v1",
)
PUBLIC_PARSERS: Final = (
    "parse_raw_publication_reload_file_v1",
    "parse_terminal_relay_descriptor_file_v1",
    "parse_terminal_relay_preseal_attestation_file_v1",
)
PUBLIC_VALIDATORS: Final = (
    "validate_terminal_relay_descriptor_binding_v1",
    "validate_terminal_relay_chain_v1",
)


class ForagerMatchedV3StorageTerminalRelayProtocolV1Error(ValueError):
    """One terminal-relay protocol record failed closed."""


def _fail(message: str) -> Never:
    raise ForagerMatchedV3StorageTerminalRelayProtocolV1Error(message)


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


def _require_exact_int(value: object, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or not minimum <= value <= _MAX_EXACT_INTEGER:
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


def _require_raw_capture(value: object, label: str) -> RawByteCaptureV1:
    if type(value) is not RawByteCaptureV1:
        _fail(f"{label} type differs")
    try:
        return raw_byte_capture_v1_from_dict(value.to_dict())
    except ValueError as exc:
        _fail(f"{label} differs: {exc}")


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


def _terminal_producer_from_dict(value: object) -> ProducerRefV1:
    try:
        parsed = producer_ref_v1_from_dict(value)
    except ValueError as exc:
        _fail(f"terminal relay producer differs: {exc}")
    return _require_terminal_producer(parsed)


def _raw_capture_from_dict(value: object, label: str) -> RawByteCaptureV1:
    try:
        parsed = raw_byte_capture_v1_from_dict(value)
    except ValueError as exc:
        _fail(f"{label} differs: {exc}")
    return _require_raw_capture(parsed, label)


def _validate_flat_run_envelope(
    *,
    schema_version: object,
    expected_schema: str,
    status: object,
    expected_status: str,
    campaign_id: object,
    case_subject: object,
    runtime_intent: object,
    host_go: object,
    image_id: object,
    container_name: object,
    container_id_commitment_sha256: object,
    outer_cgroup_identity_sha256: object,
    producer: object,
    additional_identity_values: tuple[object, ...],
) -> None:
    if type(schema_version) is not str or schema_version != expected_schema:
        _fail("artifact schema differs")
    if type(status) is not str or status != expected_status:
        _fail("artifact status differs")
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
    exact_producer = _require_terminal_producer(producer)
    require_distinct_sha256s(
        (
            exact_runtime_intent.file_sha256,
            exact_runtime_intent.body_sha256,
            exact_host_go.file_sha256,
            exact_host_go.body_sha256,
            exact_producer.descriptor_file_sha256,
            exact_producer.descriptor_body_sha256,
            exact_producer.source_sha256,
            container_commitment,
            outer_cgroup,
            *additional_identity_values,
        ),
        "terminal-relay run identities",
    )


def _artifact_identity_values(*artifacts: ArtifactRefV1) -> tuple[str, ...]:
    return tuple(
        identity
        for artifact in artifacts
        for identity in (artifact.file_sha256, artifact.body_sha256)
    )


def _common_body(
    record: RawPublicationReloadV1 | TerminalRelayPresealAttestationV1,
) -> dict[str, Any]:
    return {
        "campaign_id": record.campaign_id,
        "case_subject": record.case_subject.to_dict(),
        "container_id_commitment_sha256": record.container_id_commitment_sha256,
        "container_name": record.container_name,
        "host_go": record.host_go.to_dict(),
        "image_id": record.image_id,
        "outer_cgroup_identity_sha256": record.outer_cgroup_identity_sha256,
        "producer": record.producer.to_dict(),
        "runtime_intent": record.runtime_intent.to_dict(),
        "schema_version": record.schema_version,
        "status": record.status,
    }


def _seal_subclasses(name: str) -> Never:
    raise TypeError(f"{name} is runtime-sealed against subclasses")


@dataclass(frozen=True, slots=True)
class TerminalRelayDescriptorV1:
    """Nonoperational descriptor for the terminal-relay schema provider."""

    schema_version: str = TERMINAL_RELAY_DESCRIPTOR_SCHEMA_VERSION
    status: str = TERMINAL_RELAY_DESCRIPTOR_STATUS
    role: str = TERMINAL_RELAY_ROLE
    owned_artifact_schemas: tuple[str, ...] = TERMINAL_RELAY_OWNED_ARTIFACT_SCHEMAS
    required_input_schemas: tuple[str, ...] = TERMINAL_RELAY_REQUIRED_INPUT_SCHEMAS
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
            or self.schema_version != TERMINAL_RELAY_DESCRIPTOR_SCHEMA_VERSION
        ):
            _fail("terminal relay descriptor schema differs")
        if (
            type(self.status) is not str
            or self.status != TERMINAL_RELAY_DESCRIPTOR_STATUS
            or type(self.role) is not str
            or self.role != TERMINAL_RELAY_ROLE
        ):
            _fail("terminal relay descriptor status or role differs")
        if (
            type(self.owned_artifact_schemas) is not tuple
            or self.owned_artifact_schemas != TERMINAL_RELAY_OWNED_ARTIFACT_SCHEMAS
            or any(type(value) is not str for value in self.owned_artifact_schemas)
            or type(self.required_input_schemas) is not tuple
            or self.required_input_schemas != TERMINAL_RELAY_REQUIRED_INPUT_SCHEMAS
            or any(type(value) is not str for value in self.required_input_schemas)
            or type(self.canonical_policy) is not tuple
            or self.canonical_policy != CANONICAL_POLICY
            or any(type(value) is not str for value in self.canonical_policy)
        ):
            _fail("terminal relay descriptor schema inventory or canonical policy differs")
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
                _fail(f"terminal relay descriptor {field_name} differs")
            if field_name == "limits" and any(
                type(value) is not int for value in supplied.values()
            ):
                _fail("terminal relay descriptor limits must be exact integers")
            object.__setattr__(self, field_name, MappingProxyType(dict(supplied)))
        for posture_name, posture in (
            ("capabilities", self.capabilities),
            ("readiness", self.readiness),
            ("authority", self.authority),
            ("claims", self.claims),
        ):
            if any(type(value) is not bool or value is not False for value in posture.values()):
                _fail(f"terminal relay descriptor {posture_name} must be all false")
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
            _fail("terminal relay descriptor public API inventory differs")
        if type(self.operational_apis) is not tuple or self.operational_apis != ():
            _fail("terminal relay descriptor operational APIs must be empty")
        if (
            type(self.descriptor_self_pin_sha256) is not str
            or self.descriptor_self_pin_sha256 != _ZERO_SHA256
            or self.source_file_sha256_pin is not None
        ):
            _fail("terminal relay descriptor repository pins must remain unfinalized")

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
class RawPublicationReloadV1:
    """Raw6 capture replay for one exact read-only publication reload."""

    schema_version: str = RAW_PUBLICATION_RELOAD_SCHEMA_VERSION
    status: str = RAW_PUBLICATION_RELOAD_STATUS
    campaign_id: str
    case_subject: CaseSubjectV1
    runtime_intent: ArtifactRefV1
    host_go: ArtifactRefV1
    image_id: str
    container_name: str
    container_id_commitment_sha256: str
    outer_cgroup_identity_sha256: str
    producer: ProducerRefV1
    publication_wrapper: ArtifactRefV1
    publication_reload_validation: ArtifactRefV1
    expected_reload_observation_sha256: str
    actual_reload_observation_sha256: str
    reload_performed: bool = True
    reload_read_only: bool = True
    worker_exit_monotonic_ns: int
    publication_wrapper_monotonic_ns: int
    reload_validation_monotonic_ns: int
    capture: RawByteCaptureV1

    def __post_init__(self) -> None:
        wrapper = _require_artifact(
            self.publication_wrapper,
            NORMALIZED_PUBLICATION_WRAPPER_SCHEMA_VERSION,
            "publication wrapper",
        )
        reload_validation = _require_artifact(
            self.publication_reload_validation,
            PUBLICATION_RELOAD_VALIDATION_SCHEMA_VERSION,
            "publication reload validation",
        )
        _validate_flat_run_envelope(
            schema_version=self.schema_version,
            expected_schema=RAW_PUBLICATION_RELOAD_SCHEMA_VERSION,
            status=self.status,
            expected_status=RAW_PUBLICATION_RELOAD_STATUS,
            campaign_id=self.campaign_id,
            case_subject=self.case_subject,
            runtime_intent=self.runtime_intent,
            host_go=self.host_go,
            image_id=self.image_id,
            container_name=self.container_name,
            container_id_commitment_sha256=self.container_id_commitment_sha256,
            outer_cgroup_identity_sha256=self.outer_cgroup_identity_sha256,
            producer=self.producer,
            additional_identity_values=_artifact_identity_values(wrapper, reload_validation),
        )
        if self.reload_performed is not True or self.reload_read_only is not True:
            _fail("raw publication reload must be performed read-only")
        worker_exit = _require_exact_int(
            self.worker_exit_monotonic_ns,
            "worker-exit monotonic timestamp",
            minimum=1,
        )
        wrapper_time = _require_exact_int(
            self.publication_wrapper_monotonic_ns,
            "publication-wrapper monotonic timestamp",
            minimum=1,
        )
        reload_time = _require_exact_int(
            self.reload_validation_monotonic_ns,
            "reload-validation monotonic timestamp",
            minimum=1,
        )
        if not worker_exit < wrapper_time < reload_time:
            _fail("worker-exit/publication-wrapper/reload-validation chronology differs")
        expected = require_nonzero_sha256(
            self.expected_reload_observation_sha256,
            "expected reload observation identity",
        )
        actual = require_nonzero_sha256(
            self.actual_reload_observation_sha256,
            "actual reload observation identity",
        )
        exact_capture = _require_raw_capture(self.capture, "raw publication reload capture")
        if not (
            hmac.compare_digest(expected, actual)
            and hmac.compare_digest(expected, exact_capture.raw_sha256)
        ):
            _fail("expected, actual, and captured reload observations differ")
        # Re-decode the exact captured bytes here, even though the nested record
        # already verifies their size and SHA-256.  This rejects opaque binary,
        # noncanonical JSON, duplicate keys, and non-object observations while
        # deliberately avoiding a synthetic metadata-derived observation.
        decode_canonical_json_file(exact_capture.decoded_bytes())
        other_identities = (
            self.runtime_intent.file_sha256,
            self.runtime_intent.body_sha256,
            self.host_go.file_sha256,
            self.host_go.body_sha256,
            self.publication_wrapper.file_sha256,
            self.publication_wrapper.body_sha256,
            self.publication_reload_validation.file_sha256,
            self.publication_reload_validation.body_sha256,
            self.producer.descriptor_file_sha256,
            self.producer.descriptor_body_sha256,
            self.producer.source_sha256,
            self.container_id_commitment_sha256,
            self.outer_cgroup_identity_sha256,
        )
        if expected in other_identities:
            _fail("reload observation identity aliases a run identity")

    @classmethod
    def __init_subclass__(cls, **kwargs: object) -> None:
        _seal_subclasses(cls.__name__)

    def to_body_dict(self) -> dict[str, Any]:
        body = _common_body(self)
        body.update(
            {
                "actual_reload_observation_sha256": self.actual_reload_observation_sha256,
                "capture": self.capture.to_dict(),
                "expected_reload_observation_sha256": self.expected_reload_observation_sha256,
                "publication_reload_validation": self.publication_reload_validation.to_dict(),
                "publication_wrapper": self.publication_wrapper.to_dict(),
                "publication_wrapper_monotonic_ns": self.publication_wrapper_monotonic_ns,
                "reload_performed": self.reload_performed,
                "reload_read_only": self.reload_read_only,
                "reload_validation_monotonic_ns": self.reload_validation_monotonic_ns,
                "worker_exit_monotonic_ns": self.worker_exit_monotonic_ns,
            }
        )
        return body


@dataclass(frozen=True, slots=True, kw_only=True)
class TerminalRelayPresealAttestationV1:
    """Pre-seal relay posture bound to raw6 without emitting terminal data."""

    schema_version: str = TERMINAL_RELAY_PRESEAL_ATTESTATION_SCHEMA_VERSION
    status: str = TERMINAL_RELAY_PRESEAL_STATUS
    campaign_id: str
    case_subject: CaseSubjectV1
    runtime_intent: ArtifactRefV1
    host_go: ArtifactRefV1
    image_id: str
    container_name: str
    container_id_commitment_sha256: str
    outer_cgroup_identity_sha256: str
    producer: ProducerRefV1
    publication_wrapper: ArtifactRefV1
    publication_reload_validation: ArtifactRefV1
    raw_publication_reload: ArtifactRefV1
    terminal_relay_process_identity_sha256: str
    nonstorage_channel_commitment_sha256: str
    worker_exit_observed: bool = True
    worker_has_measured_writable_namespace: bool = False
    worker_has_measured_writable_fd: bool = False
    relay_has_measured_writable_namespace: bool = False
    relay_has_measured_writable_fd: bool = False
    terminal_transport_outside_measured_storage: bool = True
    terminal_transport_can_allocate_measured_storage: bool = False
    input_policy: str = TERMINAL_RELAY_INPUT_POLICY
    terminal_emission_performed: bool = False
    ready_before_write_seal: bool = True
    attestation_monotonic_ns: int

    def __post_init__(self) -> None:
        wrapper = _require_artifact(
            self.publication_wrapper,
            NORMALIZED_PUBLICATION_WRAPPER_SCHEMA_VERSION,
            "publication wrapper",
        )
        reload_validation = _require_artifact(
            self.publication_reload_validation,
            PUBLICATION_RELOAD_VALIDATION_SCHEMA_VERSION,
            "publication reload validation",
        )
        raw_reload = _require_artifact(
            self.raw_publication_reload,
            RAW_PUBLICATION_RELOAD_SCHEMA_VERSION,
            "raw publication reload",
        )
        process_identity = require_nonzero_sha256(
            self.terminal_relay_process_identity_sha256,
            "terminal relay process identity",
        )
        channel_commitment = require_nonzero_sha256(
            self.nonstorage_channel_commitment_sha256,
            "nonstorage channel commitment",
        )
        _validate_flat_run_envelope(
            schema_version=self.schema_version,
            expected_schema=TERMINAL_RELAY_PRESEAL_ATTESTATION_SCHEMA_VERSION,
            status=self.status,
            expected_status=TERMINAL_RELAY_PRESEAL_STATUS,
            campaign_id=self.campaign_id,
            case_subject=self.case_subject,
            runtime_intent=self.runtime_intent,
            host_go=self.host_go,
            image_id=self.image_id,
            container_name=self.container_name,
            container_id_commitment_sha256=self.container_id_commitment_sha256,
            outer_cgroup_identity_sha256=self.outer_cgroup_identity_sha256,
            producer=self.producer,
            additional_identity_values=(
                *_artifact_identity_values(wrapper, reload_validation, raw_reload),
                process_identity,
                channel_commitment,
            ),
        )
        _require_exact_int(
            self.attestation_monotonic_ns,
            "terminal relay attestation monotonic timestamp",
            minimum=1,
        )
        if (
            self.worker_exit_observed is not True
            or self.worker_has_measured_writable_namespace is not False
            or self.worker_has_measured_writable_fd is not False
            or self.relay_has_measured_writable_namespace is not False
            or self.relay_has_measured_writable_fd is not False
            or self.terminal_transport_outside_measured_storage is not True
            or self.terminal_transport_can_allocate_measured_storage is not False
            or type(self.input_policy) is not str
            or self.input_policy != TERMINAL_RELAY_INPUT_POLICY
            or self.terminal_emission_performed is not False
            or self.ready_before_write_seal is not True
        ):
            _fail("terminal relay pre-seal posture differs")

    @classmethod
    def __init_subclass__(cls, **kwargs: object) -> None:
        _seal_subclasses(cls.__name__)

    def to_body_dict(self) -> dict[str, Any]:
        body = _common_body(self)
        body.update(
            {
                "attestation_monotonic_ns": self.attestation_monotonic_ns,
                "input_policy": self.input_policy,
                "nonstorage_channel_commitment_sha256": (self.nonstorage_channel_commitment_sha256),
                "publication_reload_validation": self.publication_reload_validation.to_dict(),
                "publication_wrapper": self.publication_wrapper.to_dict(),
                "raw_publication_reload": self.raw_publication_reload.to_dict(),
                "ready_before_write_seal": self.ready_before_write_seal,
                "relay_has_measured_writable_fd": self.relay_has_measured_writable_fd,
                "relay_has_measured_writable_namespace": (
                    self.relay_has_measured_writable_namespace
                ),
                "terminal_emission_performed": self.terminal_emission_performed,
                "terminal_relay_process_identity_sha256": (
                    self.terminal_relay_process_identity_sha256
                ),
                "terminal_transport_can_allocate_measured_storage": (
                    self.terminal_transport_can_allocate_measured_storage
                ),
                "terminal_transport_outside_measured_storage": (
                    self.terminal_transport_outside_measured_storage
                ),
                "worker_exit_observed": self.worker_exit_observed,
                "worker_has_measured_writable_fd": self.worker_has_measured_writable_fd,
                "worker_has_measured_writable_namespace": (
                    self.worker_has_measured_writable_namespace
                ),
            }
        )
        return body


def _validate_current_terminal_relay_descriptor_v1(
    descriptor: object,
) -> TerminalRelayDescriptorV1:
    if type(descriptor) is not TerminalRelayDescriptorV1:
        _fail("terminal relay descriptor type differs")
    TerminalRelayDescriptorV1.__post_init__(descriptor)
    return descriptor


def _validate_current_raw_publication_reload_v1(
    artifact: object,
) -> RawPublicationReloadV1:
    if type(artifact) is not RawPublicationReloadV1:
        _fail("raw publication reload type differs")
    RawPublicationReloadV1.__post_init__(artifact)
    return artifact


def _validate_current_terminal_relay_preseal_attestation_v1(
    artifact: object,
) -> TerminalRelayPresealAttestationV1:
    if type(artifact) is not TerminalRelayPresealAttestationV1:
        _fail("terminal relay pre-seal attestation type differs")
    TerminalRelayPresealAttestationV1.__post_init__(artifact)
    return artifact


def build_publication_reload_observation_capture_v1(
    *,
    observation_bytes: bytes,
) -> RawByteCaptureV1:
    """Wrap exact caller-observed canonical reload content without deriving it."""

    decode_canonical_json_file(observation_bytes)
    return RawByteCaptureV1.from_bytes(observation_bytes)


def build_terminal_relay_descriptor_file_v1(
    descriptor: TerminalRelayDescriptorV1,
) -> bytes:
    """Serialize one exact terminal-relay descriptor with an embedded BODY identity."""

    _validate_current_terminal_relay_descriptor_v1(descriptor)
    return canonical_file_bytes(
        descriptor.to_body_dict(),
        body_digest_field=_DESCRIPTOR_BODY_FIELD,
    )


def canonical_terminal_relay_descriptor_v1_body_bytes(
    descriptor: TerminalRelayDescriptorV1,
) -> bytes:
    """Return the canonical unframed descriptor BODY bytes."""

    _validate_current_terminal_relay_descriptor_v1(descriptor)
    return canonical_json_bytes(descriptor.to_body_dict(), final_lf=False)


def build_raw_publication_reload_file_v1(artifact: RawPublicationReloadV1) -> bytes:
    """Serialize one exact raw6 artifact with an embedded BODY identity."""

    _validate_current_raw_publication_reload_v1(artifact)
    return canonical_file_bytes(artifact.to_body_dict(), body_digest_field=_RAW_RELOAD_BODY_FIELD)


def canonical_raw_publication_reload_v1_body_bytes(
    artifact: RawPublicationReloadV1,
) -> bytes:
    """Return canonical unframed raw6 BODY bytes."""

    _validate_current_raw_publication_reload_v1(artifact)
    return canonical_json_bytes(artifact.to_body_dict(), final_lf=False)


def build_terminal_relay_preseal_attestation_file_v1(
    artifact: TerminalRelayPresealAttestationV1,
) -> bytes:
    """Serialize one exact relay pre-seal attestation."""

    _validate_current_terminal_relay_preseal_attestation_v1(artifact)
    return canonical_file_bytes(
        artifact.to_body_dict(),
        body_digest_field=_RELAY_ATTESTATION_BODY_FIELD,
    )


def canonical_terminal_relay_preseal_attestation_v1_body_bytes(
    artifact: TerminalRelayPresealAttestationV1,
) -> bytes:
    """Return canonical unframed relay-attestation BODY bytes."""

    _validate_current_terminal_relay_preseal_attestation_v1(artifact)
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


def parse_terminal_relay_descriptor_file_v1(
    raw: bytes,
    *,
    expected_file_sha256: object,
    expected_body_sha256: object,
) -> TerminalRelayDescriptorV1:
    """Parse a descriptor using independent caller-supplied FILE and BODY pins."""

    body = validate_canonical_file(
        raw,
        expected_file_sha256=expected_file_sha256,
        expected_body_sha256=expected_body_sha256,
        body_digest_field=_DESCRIPTOR_BODY_FIELD,
    )
    data = _require_exact_dict(
        body,
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
        "terminal relay descriptor",
    )
    owned_schemas = _tuple_of_strings(data["owned_artifact_schemas"], "owned schemas")
    return TerminalRelayDescriptorV1(
        schema_version=data["schema_version"],
        status=data["status"],
        role=data["role"],
        owned_artifact_schemas=owned_schemas,
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


_COMMON_KEYS: Final = frozenset(
    {
        "campaign_id",
        "case_subject",
        "container_id_commitment_sha256",
        "container_name",
        "host_go",
        "image_id",
        "outer_cgroup_identity_sha256",
        "producer",
        "runtime_intent",
        "schema_version",
        "status",
    }
)
_RAW_KEYS: Final = _COMMON_KEYS | frozenset(
    {
        "actual_reload_observation_sha256",
        "capture",
        "expected_reload_observation_sha256",
        "publication_reload_validation",
        "publication_wrapper",
        "publication_wrapper_monotonic_ns",
        "reload_performed",
        "reload_read_only",
        "reload_validation_monotonic_ns",
        "worker_exit_monotonic_ns",
    }
)
_RELAY_KEYS: Final = _COMMON_KEYS | frozenset(
    {
        "attestation_monotonic_ns",
        "input_policy",
        "nonstorage_channel_commitment_sha256",
        "publication_reload_validation",
        "publication_wrapper",
        "raw_publication_reload",
        "ready_before_write_seal",
        "relay_has_measured_writable_fd",
        "relay_has_measured_writable_namespace",
        "terminal_emission_performed",
        "terminal_relay_process_identity_sha256",
        "terminal_transport_can_allocate_measured_storage",
        "terminal_transport_outside_measured_storage",
        "worker_exit_observed",
        "worker_has_measured_writable_fd",
        "worker_has_measured_writable_namespace",
    }
)


def parse_raw_publication_reload_file_v1(
    raw: bytes,
    *,
    expected_file_sha256: object,
    expected_body_sha256: object,
) -> RawPublicationReloadV1:
    """Parse raw6 using two independently supplied canonical identities."""

    data = _require_exact_dict(
        validate_canonical_file(
            raw,
            expected_file_sha256=expected_file_sha256,
            expected_body_sha256=expected_body_sha256,
            body_digest_field=_RAW_RELOAD_BODY_FIELD,
        ),
        _RAW_KEYS,
        "raw publication reload",
    )
    return RawPublicationReloadV1(
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
        producer=_terminal_producer_from_dict(data["producer"]),
        publication_wrapper=_artifact_from_dict(
            data["publication_wrapper"],
            NORMALIZED_PUBLICATION_WRAPPER_SCHEMA_VERSION,
            "publication wrapper",
        ),
        publication_reload_validation=_artifact_from_dict(
            data["publication_reload_validation"],
            PUBLICATION_RELOAD_VALIDATION_SCHEMA_VERSION,
            "publication reload validation",
        ),
        expected_reload_observation_sha256=data["expected_reload_observation_sha256"],
        actual_reload_observation_sha256=data["actual_reload_observation_sha256"],
        worker_exit_monotonic_ns=data["worker_exit_monotonic_ns"],
        publication_wrapper_monotonic_ns=data["publication_wrapper_monotonic_ns"],
        reload_validation_monotonic_ns=data["reload_validation_monotonic_ns"],
        capture=_raw_capture_from_dict(data["capture"], "raw publication reload capture"),
        schema_version=data["schema_version"],
        status=data["status"],
        reload_performed=data["reload_performed"],
        reload_read_only=data["reload_read_only"],
    )


def parse_terminal_relay_preseal_attestation_file_v1(
    raw: bytes,
    *,
    expected_file_sha256: object,
    expected_body_sha256: object,
) -> TerminalRelayPresealAttestationV1:
    """Parse one pre-seal relay attestation under independent pins."""

    data = _require_exact_dict(
        validate_canonical_file(
            raw,
            expected_file_sha256=expected_file_sha256,
            expected_body_sha256=expected_body_sha256,
            body_digest_field=_RELAY_ATTESTATION_BODY_FIELD,
        ),
        _RELAY_KEYS,
        "terminal relay pre-seal attestation",
    )
    return TerminalRelayPresealAttestationV1(
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
        producer=_terminal_producer_from_dict(data["producer"]),
        publication_wrapper=_artifact_from_dict(
            data["publication_wrapper"],
            NORMALIZED_PUBLICATION_WRAPPER_SCHEMA_VERSION,
            "publication wrapper",
        ),
        publication_reload_validation=_artifact_from_dict(
            data["publication_reload_validation"],
            PUBLICATION_RELOAD_VALIDATION_SCHEMA_VERSION,
            "publication reload validation",
        ),
        raw_publication_reload=_artifact_from_dict(
            data["raw_publication_reload"],
            RAW_PUBLICATION_RELOAD_SCHEMA_VERSION,
            "raw publication reload",
        ),
        terminal_relay_process_identity_sha256=data["terminal_relay_process_identity_sha256"],
        nonstorage_channel_commitment_sha256=data["nonstorage_channel_commitment_sha256"],
        attestation_monotonic_ns=data["attestation_monotonic_ns"],
        schema_version=data["schema_version"],
        status=data["status"],
        worker_exit_observed=data["worker_exit_observed"],
        worker_has_measured_writable_namespace=data["worker_has_measured_writable_namespace"],
        worker_has_measured_writable_fd=data["worker_has_measured_writable_fd"],
        relay_has_measured_writable_namespace=data["relay_has_measured_writable_namespace"],
        relay_has_measured_writable_fd=data["relay_has_measured_writable_fd"],
        terminal_transport_outside_measured_storage=data[
            "terminal_transport_outside_measured_storage"
        ],
        terminal_transport_can_allocate_measured_storage=data[
            "terminal_transport_can_allocate_measured_storage"
        ],
        input_policy=data["input_policy"],
        terminal_emission_performed=data["terminal_emission_performed"],
        ready_before_write_seal=data["ready_before_write_seal"],
    )


def terminal_relay_descriptor_identity_v1(
    descriptor: TerminalRelayDescriptorV1,
) -> ArtifactRefV1:
    """Derive the descriptor's canonical FILE and BODY identities in memory."""

    _validate_current_terminal_relay_descriptor_v1(descriptor)
    raw = build_terminal_relay_descriptor_file_v1(descriptor)
    return ArtifactRefV1(
        schema_version=TERMINAL_RELAY_DESCRIPTOR_SCHEMA_VERSION,
        file_sha256=hashlib.sha256(raw).hexdigest(),
        body_sha256=canonical_body_sha256(descriptor.to_body_dict()),
    )


def raw_publication_reload_identity_v1(artifact: RawPublicationReloadV1) -> ArtifactRefV1:
    """Derive canonical raw6 FILE and BODY identities in memory."""

    _validate_current_raw_publication_reload_v1(artifact)
    raw = build_raw_publication_reload_file_v1(artifact)
    return ArtifactRefV1(
        schema_version=RAW_PUBLICATION_RELOAD_SCHEMA_VERSION,
        file_sha256=hashlib.sha256(raw).hexdigest(),
        body_sha256=canonical_body_sha256(artifact.to_body_dict()),
    )


def terminal_relay_preseal_attestation_identity_v1(
    artifact: TerminalRelayPresealAttestationV1,
) -> ArtifactRefV1:
    """Derive canonical relay-attestation FILE and BODY identities in memory."""

    _validate_current_terminal_relay_preseal_attestation_v1(artifact)
    raw = build_terminal_relay_preseal_attestation_file_v1(artifact)
    return ArtifactRefV1(
        schema_version=TERMINAL_RELAY_PRESEAL_ATTESTATION_SCHEMA_VERSION,
        file_sha256=hashlib.sha256(raw).hexdigest(),
        body_sha256=canonical_body_sha256(artifact.to_body_dict()),
    )


def validate_terminal_relay_descriptor_binding_v1(
    descriptor: TerminalRelayDescriptorV1,
    producer: ProducerRefV1,
) -> None:
    """Bind one exact terminal producer to canonical descriptor identities."""

    _validate_current_terminal_relay_descriptor_v1(descriptor)
    exact_producer = _require_terminal_producer(producer)
    identity = terminal_relay_descriptor_identity_v1(descriptor)
    if not hmac.compare_digest(
        exact_producer.descriptor_file_sha256, identity.file_sha256
    ) or not hmac.compare_digest(exact_producer.descriptor_body_sha256, identity.body_sha256):
        _fail("terminal relay producer differs from its descriptor identities")


def _run_context(
    record: RawPublicationReloadV1 | TerminalRelayPresealAttestationV1,
) -> tuple[object, ...]:
    return (
        record.campaign_id,
        record.case_subject,
        record.runtime_intent,
        record.host_go,
        record.image_id,
        record.container_name,
        record.container_id_commitment_sha256,
        record.outer_cgroup_identity_sha256,
        record.producer,
    )


def validate_terminal_relay_chain_v1(
    raw_publication_reload: RawPublicationReloadV1,
    relay_attestation: TerminalRelayPresealAttestationV1,
) -> None:
    """Validate the one-way wrapper -> reload/raw6 -> relay dependency chain."""

    _validate_current_raw_publication_reload_v1(raw_publication_reload)
    _validate_current_terminal_relay_preseal_attestation_v1(relay_attestation)
    if _run_context(raw_publication_reload) != _run_context(relay_attestation):
        _fail("raw6 and relay run contexts differ")
    raw_identity = raw_publication_reload_identity_v1(raw_publication_reload)
    if (
        relay_attestation.publication_wrapper != raw_publication_reload.publication_wrapper
        or relay_attestation.publication_reload_validation
        != raw_publication_reload.publication_reload_validation
        or relay_attestation.raw_publication_reload != raw_identity
    ):
        _fail("relay wrapper, reload validation, or raw6 projection differs")
    require_distinct_sha256s(
        (
            raw_publication_reload.capture.raw_sha256,
            *_artifact_identity_values(
                raw_publication_reload.runtime_intent,
                raw_publication_reload.host_go,
                raw_publication_reload.publication_wrapper,
                raw_publication_reload.publication_reload_validation,
                raw_identity,
            ),
            raw_publication_reload.producer.descriptor_file_sha256,
            raw_publication_reload.producer.descriptor_body_sha256,
            raw_publication_reload.producer.source_sha256,
            raw_publication_reload.container_id_commitment_sha256,
            raw_publication_reload.outer_cgroup_identity_sha256,
            relay_attestation.terminal_relay_process_identity_sha256,
            relay_attestation.nonstorage_channel_commitment_sha256,
        ),
        "terminal-relay full-chain identities",
    )
    if not (
        raw_publication_reload.reload_validation_monotonic_ns
        < relay_attestation.attestation_monotonic_ns
    ):
        _fail("reload validation must strictly precede relay attestation")


if any(
    pin != _ZERO_SHA256
    for pin in (
        PINNED_TERMINAL_RELAY_DESCRIPTOR_FILE_SHA256,
        PINNED_TERMINAL_RELAY_DESCRIPTOR_BODY_SHA256,
        PINNED_TERMINAL_RELAY_SOURCE_SHA256,
    )
):
    raise AssertionError("terminal relay repository pins must remain unfinalized")
if len(set(TERMINAL_RELAY_OWNED_ARTIFACT_SCHEMAS)) != len(TERMINAL_RELAY_OWNED_ARTIFACT_SCHEMAS):
    raise AssertionError("terminal relay owned schemas must be distinct")
if set(TERMINAL_RELAY_OWNED_ARTIFACT_SCHEMAS).intersection(TERMINAL_RELAY_REQUIRED_INPUT_SCHEMAS):
    raise AssertionError("terminal relay owned and input schemas must be disjoint")


__all__ = [
    "AUTHORITY",
    "ArtifactRefV1",
    "CANONICAL_POLICY",
    "CAPABILITIES",
    "CLAIMS",
    "ForagerMatchedV3StorageTerminalRelayProtocolV1Error",
    "HOST_GO_V3_SCHEMA_VERSION",
    "LIMITS",
    "NORMALIZED_PUBLICATION_WRAPPER_SCHEMA_VERSION",
    "PINNED_TERMINAL_RELAY_DESCRIPTOR_BODY_SHA256",
    "PINNED_TERMINAL_RELAY_DESCRIPTOR_FILE_SHA256",
    "PINNED_TERMINAL_RELAY_SOURCE_SHA256",
    "PUBLICATION_RELOAD_VALIDATION_SCHEMA_VERSION",
    "PUBLIC_CANONICAL_BUILDERS",
    "PUBLIC_PARSERS",
    "PUBLIC_VALIDATORS",
    "ProducerRefV1",
    "RAW_PUBLICATION_RELOAD_SCHEMA_VERSION",
    "RAW_PUBLICATION_RELOAD_STATUS",
    "READINESS",
    "RawPublicationReloadV1",
    "STORAGE_BOUNDARY_RUNTIME_INTENT_SCHEMA_VERSION",
    "TERMINAL_RELAY_DESCRIPTOR_SCHEMA_VERSION",
    "TERMINAL_RELAY_DESCRIPTOR_STATUS",
    "TERMINAL_RELAY_INPUT_POLICY",
    "TERMINAL_RELAY_OWNED_ARTIFACT_SCHEMAS",
    "TERMINAL_RELAY_PRESEAL_ATTESTATION_SCHEMA_VERSION",
    "TERMINAL_RELAY_PRESEAL_STATUS",
    "TERMINAL_RELAY_REQUIRED_INPUT_SCHEMAS",
    "TERMINAL_RELAY_ROLE",
    "TerminalRelayDescriptorV1",
    "TerminalRelayPresealAttestationV1",
    "build_publication_reload_observation_capture_v1",
    "build_raw_publication_reload_file_v1",
    "build_terminal_relay_descriptor_file_v1",
    "build_terminal_relay_preseal_attestation_file_v1",
    "canonical_raw_publication_reload_v1_body_bytes",
    "canonical_terminal_relay_descriptor_v1_body_bytes",
    "canonical_terminal_relay_preseal_attestation_v1_body_bytes",
    "parse_raw_publication_reload_file_v1",
    "parse_terminal_relay_descriptor_file_v1",
    "parse_terminal_relay_preseal_attestation_file_v1",
    "raw_publication_reload_identity_v1",
    "terminal_relay_descriptor_identity_v1",
    "terminal_relay_preseal_attestation_identity_v1",
    "validate_terminal_relay_chain_v1",
    "validate_terminal_relay_descriptor_binding_v1",
]
