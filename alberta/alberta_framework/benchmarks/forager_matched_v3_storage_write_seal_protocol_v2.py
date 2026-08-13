"""Pure write-seal metadata protocol for matched Forager v3.

This module defines canonical records and validators only.  It cannot observe a
process, inspect a namespace, contact a container runtime, quiesce a writer, or
commit a production seal.  Raw write-seal bytes are supplied by a separately
pinned producer and replayed against an exact schema before the metadata record
can be constructed.

The dependency direction is deliberately one way: raw7 binds only already
committed predecessors and preseal evidence.  The irreversible seal then binds
raw7.  Raw7 never names, predicts, or contains the future seal identity.
"""

from __future__ import annotations

import hashlib
import hmac
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final, Literal, Never, cast, final

from alberta_framework.benchmarks import _forager_matched_v3_canonical_evidence as codec

WRITE_SEAL_PRODUCER_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.qualification_storage_write_seal_producer_descriptor.v2"
)
RAW_WRITE_SEAL_SCHEMA_VERSION: Final = "alberta.forager_matched_v3.raw.write_seal.v1"
IRREVERSIBLE_WRITE_SEAL_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.qualification_storage_write_quiescence_seal.v1"
)
STORAGE_BOUNDARY_RUNTIME_INTENT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.storage_boundary_runtime_intent.v1"
)
HOST_GO_V3_SCHEMA_VERSION: Final = "alberta.forager_matched_v3.host_qualification_go_commitment.v3"
PUBLICATION_RELOAD_VALIDATION_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.qualification_publication_reload_validation.v1"
)
TERMINAL_RELAY_PRESEAL_ATTESTATION_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.qualification_storage_terminal_relay_preseal_attestation.v1"
)
NONSTORAGE_CHANNEL_PRESEAL_ATTESTATION_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.qualification_storage_nonstorage_channel_readiness_attestation.v1"
)
TMPFS_HARD_LIMIT_MOUNT_MUTATION_CLOSURE_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.raw.tmpfs_hard_limit_mount_mutation_closure.v1"
)
WRITABLE_FD_LIFETIME_INVENTORY_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.raw.allocatable_writable_fd_lifetime_inventory.v1"
)
DOCKER_API_OPERATION_JOURNAL_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.raw.docker_storage_api_operation_journal.v1"
)
ROOTFS_UPPERDIR_INTERVAL_DELTA_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.raw.rootfs_upperdir_go_to_seal_delta.v1"
)
DOCKER_VOLUME_INVENTORY_DELTA_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.raw.docker_volume_inventory_delta.v1"
)

RAW_WRITE_SEAL_CAPTURE_FORMAT: Final = "alberta.forager_matched_v3.raw.write_seal.capture.v1"
WRITE_SEAL_ARCHITECTURE: Final = "worker_exit_then_isolated_terminal_relay"

WRITE_SEAL_DESCRIPTOR_STATUS: Final = (
    "pure_source_only_unfinalized_uninvoked_no_production_artifact"
)
RAW_WRITE_SEAL_STATUS: Final = "preseal_raw_write_quiescence_observation_non_authorizing"
IRREVERSIBLE_WRITE_SEAL_STATUS: Final = "irreversible_preterminal_write_seal_non_authorizing"

WRITE_SEAL_PRODUCER_DESCRIPTOR_BODY_SHA256_FIELD: Final = (
    "write_seal_producer_descriptor_body_sha256"
)
RAW_WRITE_SEAL_BODY_SHA256_FIELD: Final = "raw_write_seal_body_sha256"
IRREVERSIBLE_WRITE_SEAL_BODY_SHA256_FIELD: Final = "write_quiescence_seal_body_sha256"

ZERO_SHA256: Final = "0" * 64
DESCRIPTOR_SERIALIZED_SELF_PIN_SENTINEL: Final = ZERO_SHA256
PINNED_WRITE_SEAL_PRODUCER_DESCRIPTOR_FILE_SHA256: Final = ZERO_SHA256
PINNED_WRITE_SEAL_PRODUCER_DESCRIPTOR_BODY_SHA256: Final = ZERO_SHA256
PINNED_WRITE_SEAL_PRODUCER_SOURCE_SHA256: Final = ZERO_SHA256

OWNED_ARTIFACT_SCHEMAS: Final = (
    RAW_WRITE_SEAL_SCHEMA_VERSION,
    IRREVERSIBLE_WRITE_SEAL_SCHEMA_VERSION,
)
REQUIRED_INPUT_SCHEMAS: Final = (
    STORAGE_BOUNDARY_RUNTIME_INTENT_SCHEMA_VERSION,
    HOST_GO_V3_SCHEMA_VERSION,
    PUBLICATION_RELOAD_VALIDATION_SCHEMA_VERSION,
    TERMINAL_RELAY_PRESEAL_ATTESTATION_SCHEMA_VERSION,
    NONSTORAGE_CHANNEL_PRESEAL_ATTESTATION_SCHEMA_VERSION,
    TMPFS_HARD_LIMIT_MOUNT_MUTATION_CLOSURE_SCHEMA_VERSION,
    WRITABLE_FD_LIFETIME_INVENTORY_SCHEMA_VERSION,
    DOCKER_API_OPERATION_JOURNAL_SCHEMA_VERSION,
    ROOTFS_UPPERDIR_INTERVAL_DELTA_SCHEMA_VERSION,
    DOCKER_VOLUME_INVENTORY_DELTA_SCHEMA_VERSION,
)

CANONICAL_POLICY: Final[Mapping[str, bool]] = MappingProxyType(
    {
        "ascii_only": True,
        "body_and_file_pins_independent": True,
        "duplicate_keys_rejected": True,
        "exact_key_sets": True,
        "final_lf_required": True,
        "floats_rejected": True,
        "raw_capture_replayed": True,
    }
)
LIMITS: Final[Mapping[str, int]] = MappingProxyType(
    {
        "maximum_canonical_file_bytes": codec.MAX_CANONICAL_FILE_BYTES,
        "maximum_exact_integer": codec.MAX_EXACT_INTEGER,
        "maximum_json_depth": codec.MAX_JSON_DEPTH,
        "maximum_json_nodes": codec.MAX_JSON_NODES,
        "maximum_raw_capture_bytes": codec.MAX_RAW_CAPTURE_BYTES,
    }
)
SOURCE_ONLY_CAPABILITIES: Final[Mapping[str, bool]] = MappingProxyType(
    {
        "container_control": False,
        "filesystem_access": False,
        "mount_control": False,
        "namespace_control": False,
        "network_access": False,
        "process_control": False,
        "storage_measurement": False,
        "write_seal_execution": False,
    }
)
SOURCE_ONLY_READINESS: Final[Mapping[str, bool]] = MappingProxyType(
    {
        "execution_ready": False,
        "producer_schema_closure_complete": False,
        "production_ready": False,
        "qualification_ready": False,
        "write_seal_ready": False,
    }
)
SOURCE_ONLY_AUTHORITY: Final[Mapping[str, bool]] = MappingProxyType(
    {
        "execution_authorized": False,
        "publication_authorized": False,
        "qualification_granted": False,
        "receipt_endorsed": False,
    }
)
SOURCE_ONLY_CLAIMS: Final[Mapping[str, bool]] = MappingProxyType(
    {
        "performance_claim_allowed": False,
        "promotion_allowed": False,
        "resource_matched": False,
        "scientific_evidence_created": False,
        "write_quiescence_established_by_this_module": False,
    }
)

PUBLIC_CANONICAL_BUILDERS: Final = (
    "canonical_raw_write_seal_capture_bytes",
    "canonical_raw_write_seal_v1_body_bytes",
    "canonical_raw_write_seal_v1_file_bytes",
    "canonical_write_quiescence_seal_v1_body_bytes",
    "canonical_write_quiescence_seal_v1_file_bytes",
    "canonical_write_seal_producer_descriptor_v2_body_bytes",
    "canonical_write_seal_producer_descriptor_v2_file_bytes",
)
PUBLIC_PARSERS: Final = (
    "parse_raw_write_seal_v1",
    "parse_write_quiescence_seal_v1",
    "parse_write_seal_producer_descriptor_v2",
)
PUBLIC_VALIDATORS: Final = (
    "validate_raw_write_seal_v1",
    "validate_raw_write_seal_projection_v1",
    "validate_write_quiescence_seal_v1",
    "validate_write_seal_chain_v2",
)

_IDENTIFIER_RE: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}\Z")
_CONTAINER_NAME_RE: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_IMAGE_ID_RE: Final = re.compile(r"sha256:[0-9a-f]{64}\Z")
_MAPPING_PROXY_TYPE: Final[type] = type(MappingProxyType({}))

_RUNTIME_KEYS: Final = frozenset(
    {
        "campaign_id",
        "case_subject",
        "container_id_commitment_sha256",
        "container_name",
        "host_go",
        "image_id",
        "outer_cgroup_identity_sha256",
        "runtime_intent",
    }
)
_RAW_BODY_KEYS: Final = _RUNTIME_KEYS | frozenset(
    {
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
)
_SEAL_BODY_KEYS: Final = _RUNTIME_KEYS | frozenset(
    {
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
)
_DESCRIPTOR_BODY_KEYS: Final = frozenset(
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
)
_CAPTURE_KEYS: Final = frozenset(
    _RAW_BODY_KEYS - _RUNTIME_KEYS - {"capture", "schema_version", "status"}
) | {"capture_format", "runtime"}


class ForagerMatchedV3StorageWriteSealProtocolV2Error(ValueError):
    """One pure write-seal protocol record failed closed."""


def _fail(message: str) -> Never:
    raise ForagerMatchedV3StorageWriteSealProtocolV2Error(message)


def _codec_call[T](label: str, callback: Callable[[], T]) -> T:
    """Translate one trusted-codec rejection into this protocol's error surface."""

    try:
        return callback()
    except codec.CanonicalEvidenceError as exc:
        raise ForagerMatchedV3StorageWriteSealProtocolV2Error(
            f"{label} failed canonical nested validation"
        ) from exc


def _artifact_from_dict(value: object, label: str) -> codec.ArtifactRefV1:
    return _codec_call(label, lambda: codec.artifact_ref_v1_from_dict(value))


def _producer_from_dict(value: object, label: str) -> codec.ProducerRefV1:
    return _codec_call(label, lambda: codec.producer_ref_v1_from_dict(value))


def _case_subject_from_dict(value: object, label: str) -> codec.CaseSubjectV1:
    return _codec_call(label, lambda: codec.case_subject_v1_from_dict(value))


def _raw_capture_from_dict(value: object, label: str) -> codec.RawByteCaptureV1:
    return _codec_call(label, lambda: codec.raw_byte_capture_v1_from_dict(value))


def _require_exact_text(value: object, expected: str, label: str) -> str:
    if type(value) is not str or value != expected:
        _fail(f"{label} differs")
    return value


def _require_exact_text_tuple(
    value: object,
    expected: tuple[str, ...],
    label: str,
) -> tuple[str, ...]:
    if (
        type(value) is not tuple
        or any(type(item) is not str for item in value)
        or value != expected
    ):
        _fail(f"{label} differs")
    return cast(tuple[str, ...], value)


def _require_int(
    value: object,
    label: str,
    *,
    minimum: int = 0,
    maximum: int = codec.MAX_EXACT_INTEGER,
) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        _fail(f"{label} must be one bounded exact integer")
    return value


def _require_bool(value: object, label: str, *, expected: bool) -> bool:
    if type(value) is not bool or value is not expected:
        _fail(f"{label} must be exact {expected!r}")
    return value


def _require_identifier(value: object, label: str) -> str:
    if type(value) is not str or _IDENTIFIER_RE.fullmatch(value) is None:
        _fail(f"{label} must be one bounded portable identifier")
    return value


def _require_artifact(
    value: object,
    schema_version: str,
    label: str,
) -> codec.ArtifactRefV1:
    if type(value) is not codec.ArtifactRefV1:
        _fail(f"{label} identity differs")
    replayed = _codec_call(
        label,
        lambda: codec.artifact_ref_v1_from_dict(value.to_dict()),
    )
    if replayed != value or type(value.schema_version) is not str:
        _fail(f"{label} identity differs")
    _require_exact_text(value.schema_version, schema_version, f"{label} schema")
    return value


def _require_producer(value: object, label: str) -> codec.ProducerRefV1:
    if type(value) is not codec.ProducerRefV1:
        _fail(f"{label} must use the exact write-seal producer role")
    replayed = _codec_call(
        label,
        lambda: codec.producer_ref_v1_from_dict(value.to_dict()),
    )
    if replayed != value:
        _fail(f"{label} producer identity differs")
    _require_exact_text(value.role, "write_seal_producer", f"{label} producer role")
    _require_exact_text(
        value.descriptor_schema_version,
        WRITE_SEAL_PRODUCER_DESCRIPTOR_SCHEMA_VERSION,
        f"{label} producer descriptor schema",
    )
    return value


def _require_case_subject(value: object, label: str) -> codec.CaseSubjectV1:
    if type(value) is not codec.CaseSubjectV1:
        _fail(f"{label} type differs")
    replayed = _codec_call(
        label,
        lambda: codec.case_subject_v1_from_dict(value.to_dict()),
    )
    if replayed != value:
        _fail(f"{label} identity differs")
    return value


def _require_raw_capture(value: object, label: str) -> codec.RawByteCaptureV1:
    if type(value) is not codec.RawByteCaptureV1:
        _fail(f"{label} type differs")
    replayed = _codec_call(
        label,
        lambda: codec.raw_byte_capture_v1_from_dict(value.to_dict()),
    )
    if replayed != value:
        _fail(f"{label} identity differs")
    return value


def _require_nonzero_sha256(value: object, label: str) -> str:
    try:
        return codec.require_nonzero_sha256(value, label)
    except codec.CanonicalEvidenceError as exc:
        raise ForagerMatchedV3StorageWriteSealProtocolV2Error(
            f"{label} must be one exact nonzero SHA-256"
        ) from exc


def _require_distinct_sha256s(values: tuple[object, ...], label: str) -> tuple[str, ...]:
    try:
        return codec.require_distinct_sha256s(values, label)
    except codec.CanonicalEvidenceError as exc:
        raise ForagerMatchedV3StorageWriteSealProtocolV2Error(
            f"{label} contain an invalid or aliased SHA-256 identity"
        ) from exc


def _exact_dict(value: object, keys: frozenset[str], label: str) -> dict[str, Any]:
    if (
        type(value) is not dict
        or any(type(key) is not str for key in value)
        or frozenset(value) != keys
    ):
        _fail(f"{label} keys differ")
    return cast(dict[str, Any], value)


def _exact_text_tuple(value: object, label: str) -> tuple[str, ...]:
    if type(value) is not list or any(type(item) is not str for item in value):
        _fail(f"{label} must be one exact text array")
    return tuple(cast(list[str], value))


def _exact_bool_mapping(
    value: object,
    expected: Mapping[str, bool],
    label: str,
) -> Mapping[str, bool]:
    item = _exact_dict(value, frozenset(expected), label)
    if any(type(entry) is not bool for entry in item.values()) or item != dict(expected):
        _fail(f"{label} differs")
    return MappingProxyType(cast(dict[str, bool], dict(item)))


def _exact_int_mapping(
    value: object,
    expected: Mapping[str, int],
    label: str,
) -> Mapping[str, int]:
    item = _exact_dict(value, frozenset(expected), label)
    if any(type(entry) is not int for entry in item.values()) or item != dict(expected):
        _fail(f"{label} differs")
    return MappingProxyType(cast(dict[str, int], dict(item)))


def _detached_descriptor_bool_mapping(
    value: object,
    expected: Mapping[str, bool],
    label: str,
) -> Mapping[str, bool]:
    if type(value) is not _MAPPING_PROXY_TYPE:
        _fail(f"write-seal descriptor {label} differ")
    observed = cast(Mapping[object, object], value)
    if (
        any(type(key) is not str for key in observed)
        or frozenset(cast(str, key) for key in observed) != frozenset(expected)
        or any(type(entry) is not bool for entry in observed.values())
        or dict(observed) != dict(expected)
    ):
        _fail(f"write-seal descriptor {label} differ")
    return MappingProxyType(cast(dict[str, bool], dict(observed)))


def _detached_descriptor_int_mapping(
    value: object,
    expected: Mapping[str, int],
    label: str,
) -> Mapping[str, int]:
    if type(value) is not _MAPPING_PROXY_TYPE:
        _fail(f"write-seal descriptor {label} differ")
    observed = cast(Mapping[object, object], value)
    if (
        any(type(key) is not str for key in observed)
        or frozenset(cast(str, key) for key in observed) != frozenset(expected)
        or any(type(entry) is not int for entry in observed.values())
        or dict(observed) != dict(expected)
    ):
        _fail(f"write-seal descriptor {label} differ")
    return MappingProxyType(cast(dict[str, int], dict(observed)))


def _identity_hashes(artifacts: tuple[codec.ArtifactRefV1, ...]) -> tuple[str, ...]:
    return tuple(artifact.file_sha256 for artifact in artifacts) + tuple(
        artifact.body_sha256 for artifact in artifacts
    )


def _runtime_identity_hashes(runtime: WriteSealRuntimeEnvelopeV1) -> tuple[str, ...]:
    return _identity_hashes((runtime.runtime_intent, runtime.host_go)) + (
        runtime.image_id.removeprefix("sha256:"),
        runtime.container_id_commitment_sha256,
        runtime.outer_cgroup_identity_sha256,
    )


@final
@dataclass(frozen=True, slots=True)
class WriteSealRuntimeEnvelopeV1:
    """Exact case/runtime/GO projection shared by raw7 and its later seal."""

    campaign_id: str
    case_subject: codec.CaseSubjectV1
    runtime_intent: codec.ArtifactRefV1
    host_go: codec.ArtifactRefV1
    image_id: str
    container_name: str
    container_id_commitment_sha256: str
    outer_cgroup_identity_sha256: str

    def __init_subclass__(cls, **kwargs: Any) -> None:
        raise TypeError("WriteSealRuntimeEnvelopeV1 is runtime-final")

    def __post_init__(self) -> None:
        _require_identifier(self.campaign_id, "write-seal campaign ID")
        _require_case_subject(self.case_subject, "write-seal case subject")
        runtime_intent = _require_artifact(
            self.runtime_intent,
            STORAGE_BOUNDARY_RUNTIME_INTENT_SCHEMA_VERSION,
            "storage runtime intent",
        )
        host_go = _require_artifact(self.host_go, HOST_GO_V3_SCHEMA_VERSION, "host GO")
        if type(self.image_id) is not str or _IMAGE_ID_RE.fullmatch(self.image_id) is None:
            _fail("write-seal image ID differs")
        if (
            type(self.container_name) is not str
            or _CONTAINER_NAME_RE.fullmatch(self.container_name) is None
        ):
            _fail("write-seal container name differs")
        container = _require_nonzero_sha256(
            self.container_id_commitment_sha256,
            "write-seal container commitment",
        )
        cgroup = _require_nonzero_sha256(
            self.outer_cgroup_identity_sha256,
            "write-seal outer-cgroup identity",
        )
        _require_distinct_sha256s(
            _identity_hashes((runtime_intent, host_go))
            + (self.image_id.removeprefix("sha256:"), container, cgroup),
            "write-seal runtime identities",
        )

    def to_dict(self) -> dict[str, Any]:
        WriteSealRuntimeEnvelopeV1.__post_init__(self)
        return {
            "campaign_id": self.campaign_id,
            "case_subject": self.case_subject.to_dict(),
            "container_id_commitment_sha256": self.container_id_commitment_sha256,
            "container_name": self.container_name,
            "host_go": self.host_go.to_dict(),
            "image_id": self.image_id,
            "outer_cgroup_identity_sha256": self.outer_cgroup_identity_sha256,
            "runtime_intent": self.runtime_intent.to_dict(),
        }


def _require_runtime(
    value: object,
    label: str,
) -> WriteSealRuntimeEnvelopeV1:
    if type(value) is not WriteSealRuntimeEnvelopeV1:
        _fail(f"{label} type differs")
    WriteSealRuntimeEnvelopeV1.__post_init__(value)
    return value


def _runtime_from_dict(value: object) -> WriteSealRuntimeEnvelopeV1:
    item = _exact_dict(value, _RUNTIME_KEYS, "write-seal runtime envelope")
    return WriteSealRuntimeEnvelopeV1(
        campaign_id=item["campaign_id"],
        case_subject=_case_subject_from_dict(item["case_subject"], "write-seal case subject"),
        runtime_intent=_artifact_from_dict(item["runtime_intent"], "storage runtime intent"),
        host_go=_artifact_from_dict(item["host_go"], "host GO"),
        image_id=item["image_id"],
        container_name=item["container_name"],
        container_id_commitment_sha256=item["container_id_commitment_sha256"],
        outer_cgroup_identity_sha256=item["outer_cgroup_identity_sha256"],
    )


def _runtime_from_flat_body(value: Mapping[str, Any]) -> WriteSealRuntimeEnvelopeV1:
    return _runtime_from_dict({key: value[key] for key in _RUNTIME_KEYS})


def _validate_raw_fields(
    *,
    runtime: object,
    publication_reload_validation: object,
    terminal_relay_preseal_attestation: object,
    nonstorage_channel_preseal_attestation: object,
    tmpfs_hard_limit_mount_mutation_closure: object,
    writable_fd_lifetime_inventory: object,
    docker_api_operation_journal: object,
    rootfs_upperdir_interval_delta: object,
    docker_volume_inventory_delta: object,
    producer: object,
    precommit_monotonic_ns: object,
) -> tuple[
    WriteSealRuntimeEnvelopeV1,
    tuple[codec.ArtifactRefV1, ...],
    codec.ProducerRefV1,
    int,
]:
    exact_runtime = _require_runtime(runtime, "raw write-seal runtime envelope")
    artifacts = (
        _require_artifact(
            publication_reload_validation,
            PUBLICATION_RELOAD_VALIDATION_SCHEMA_VERSION,
            "publication reload validation",
        ),
        _require_artifact(
            terminal_relay_preseal_attestation,
            TERMINAL_RELAY_PRESEAL_ATTESTATION_SCHEMA_VERSION,
            "terminal relay preseal attestation",
        ),
        _require_artifact(
            nonstorage_channel_preseal_attestation,
            NONSTORAGE_CHANNEL_PRESEAL_ATTESTATION_SCHEMA_VERSION,
            "nonstorage channel preseal attestation",
        ),
        _require_artifact(
            tmpfs_hard_limit_mount_mutation_closure,
            TMPFS_HARD_LIMIT_MOUNT_MUTATION_CLOSURE_SCHEMA_VERSION,
            "tmpfs hard-limit mount-mutation closure",
        ),
        _require_artifact(
            writable_fd_lifetime_inventory,
            WRITABLE_FD_LIFETIME_INVENTORY_SCHEMA_VERSION,
            "writable-FD lifetime inventory",
        ),
        _require_artifact(
            docker_api_operation_journal,
            DOCKER_API_OPERATION_JOURNAL_SCHEMA_VERSION,
            "Docker API operation journal",
        ),
        _require_artifact(
            rootfs_upperdir_interval_delta,
            ROOTFS_UPPERDIR_INTERVAL_DELTA_SCHEMA_VERSION,
            "rootfs upperdir interval delta",
        ),
        _require_artifact(
            docker_volume_inventory_delta,
            DOCKER_VOLUME_INVENTORY_DELTA_SCHEMA_VERSION,
            "Docker volume inventory delta",
        ),
    )
    exact_producer = _require_producer(producer, "raw write seal")
    exact_precommit = _require_int(
        precommit_monotonic_ns,
        "raw write-seal precommit monotonic time",
        minimum=1,
    )
    _require_distinct_sha256s(
        _runtime_identity_hashes(exact_runtime)
        + _identity_hashes(artifacts)
        + (
            exact_producer.descriptor_file_sha256,
            exact_producer.descriptor_body_sha256,
            exact_producer.source_sha256,
        ),
        "raw write-seal artifact and producer identities",
    )
    return exact_runtime, artifacts, exact_producer, exact_precommit


def _raw_capture_dict(
    *,
    runtime: WriteSealRuntimeEnvelopeV1,
    publication_reload_validation: codec.ArtifactRefV1,
    terminal_relay_preseal_attestation: codec.ArtifactRefV1,
    nonstorage_channel_preseal_attestation: codec.ArtifactRefV1,
    tmpfs_hard_limit_mount_mutation_closure: codec.ArtifactRefV1,
    writable_fd_lifetime_inventory: codec.ArtifactRefV1,
    docker_api_operation_journal: codec.ArtifactRefV1,
    rootfs_upperdir_interval_delta: codec.ArtifactRefV1,
    docker_volume_inventory_delta: codec.ArtifactRefV1,
    producer: codec.ProducerRefV1,
    precommit_monotonic_ns: int,
) -> dict[str, Any]:
    return {
        "capture_format": RAW_WRITE_SEAL_CAPTURE_FORMAT,
        "descendant_process_count": 0,
        "docker_api_operation_journal": docker_api_operation_journal.to_dict(),
        "docker_volume_inventory_delta": docker_volume_inventory_delta.to_dict(),
        "later_writer_count": 0,
        "measured_writable_fd_count": 0,
        "measured_writable_namespace_holder_count": 0,
        "mount_mutation_enabled": False,
        "nonstorage_channel_preseal_attestation": (
            nonstorage_channel_preseal_attestation.to_dict()
        ),
        "precommit_monotonic_ns": precommit_monotonic_ns,
        "producer": producer.to_dict(),
        "publication_reload_validation": publication_reload_validation.to_dict(),
        "rootfs_upperdir_interval_delta": rootfs_upperdir_interval_delta.to_dict(),
        "runtime": runtime.to_dict(),
        "terminal_relay_preseal_attestation": terminal_relay_preseal_attestation.to_dict(),
        "tmpfs_hard_limit_mount_mutation_closure": (
            tmpfs_hard_limit_mount_mutation_closure.to_dict()
        ),
        "worker_exit_observed": True,
        "writable_fd_lifetime_inventory": writable_fd_lifetime_inventory.to_dict(),
        "write_quiescence_irreversible": True,
    }


def canonical_raw_write_seal_capture_bytes(
    *,
    runtime: WriteSealRuntimeEnvelopeV1,
    publication_reload_validation: codec.ArtifactRefV1,
    terminal_relay_preseal_attestation: codec.ArtifactRefV1,
    nonstorage_channel_preseal_attestation: codec.ArtifactRefV1,
    tmpfs_hard_limit_mount_mutation_closure: codec.ArtifactRefV1,
    writable_fd_lifetime_inventory: codec.ArtifactRefV1,
    docker_api_operation_journal: codec.ArtifactRefV1,
    rootfs_upperdir_interval_delta: codec.ArtifactRefV1,
    docker_volume_inventory_delta: codec.ArtifactRefV1,
    producer: codec.ProducerRefV1,
    precommit_monotonic_ns: int,
) -> bytes:
    """Encode the exact bounded raw7 capture supplied by the pinned producer."""

    exact_runtime, artifacts, exact_producer, exact_precommit = _validate_raw_fields(
        runtime=runtime,
        publication_reload_validation=publication_reload_validation,
        terminal_relay_preseal_attestation=terminal_relay_preseal_attestation,
        nonstorage_channel_preseal_attestation=nonstorage_channel_preseal_attestation,
        tmpfs_hard_limit_mount_mutation_closure=(tmpfs_hard_limit_mount_mutation_closure),
        writable_fd_lifetime_inventory=writable_fd_lifetime_inventory,
        docker_api_operation_journal=docker_api_operation_journal,
        rootfs_upperdir_interval_delta=rootfs_upperdir_interval_delta,
        docker_volume_inventory_delta=docker_volume_inventory_delta,
        producer=producer,
        precommit_monotonic_ns=precommit_monotonic_ns,
    )
    return codec.canonical_json_bytes(
        _raw_capture_dict(
            runtime=exact_runtime,
            publication_reload_validation=artifacts[0],
            terminal_relay_preseal_attestation=artifacts[1],
            nonstorage_channel_preseal_attestation=artifacts[2],
            tmpfs_hard_limit_mount_mutation_closure=artifacts[3],
            writable_fd_lifetime_inventory=artifacts[4],
            docker_api_operation_journal=artifacts[5],
            rootfs_upperdir_interval_delta=artifacts[6],
            docker_volume_inventory_delta=artifacts[7],
            producer=exact_producer,
            precommit_monotonic_ns=exact_precommit,
        )
    )


@final
@dataclass(frozen=True, slots=True)
class RawWriteSealV1:
    """Raw position 7: a replayed preseal quiescence observation."""

    runtime: WriteSealRuntimeEnvelopeV1
    publication_reload_validation: codec.ArtifactRefV1
    terminal_relay_preseal_attestation: codec.ArtifactRefV1
    nonstorage_channel_preseal_attestation: codec.ArtifactRefV1
    tmpfs_hard_limit_mount_mutation_closure: codec.ArtifactRefV1
    writable_fd_lifetime_inventory: codec.ArtifactRefV1
    docker_api_operation_journal: codec.ArtifactRefV1
    rootfs_upperdir_interval_delta: codec.ArtifactRefV1
    docker_volume_inventory_delta: codec.ArtifactRefV1
    producer: codec.ProducerRefV1
    precommit_monotonic_ns: int
    capture: codec.RawByteCaptureV1
    schema_version: str = RAW_WRITE_SEAL_SCHEMA_VERSION
    status: str = RAW_WRITE_SEAL_STATUS
    worker_exit_observed: bool = True
    descendant_process_count: int = 0
    measured_writable_namespace_holder_count: int = 0
    measured_writable_fd_count: int = 0
    later_writer_count: int = 0
    mount_mutation_enabled: bool = False
    write_quiescence_irreversible: bool = True

    def __init_subclass__(cls, **kwargs: Any) -> None:
        raise TypeError("RawWriteSealV1 is runtime-final")

    def __post_init__(self) -> None:
        _require_exact_text(
            self.schema_version,
            RAW_WRITE_SEAL_SCHEMA_VERSION,
            "raw write-seal schema",
        )
        _require_exact_text(self.status, RAW_WRITE_SEAL_STATUS, "raw write-seal status")
        runtime, artifacts, producer, _ = _validate_raw_fields(
            runtime=self.runtime,
            publication_reload_validation=self.publication_reload_validation,
            terminal_relay_preseal_attestation=self.terminal_relay_preseal_attestation,
            nonstorage_channel_preseal_attestation=(self.nonstorage_channel_preseal_attestation),
            tmpfs_hard_limit_mount_mutation_closure=(self.tmpfs_hard_limit_mount_mutation_closure),
            writable_fd_lifetime_inventory=self.writable_fd_lifetime_inventory,
            docker_api_operation_journal=self.docker_api_operation_journal,
            rootfs_upperdir_interval_delta=self.rootfs_upperdir_interval_delta,
            docker_volume_inventory_delta=self.docker_volume_inventory_delta,
            producer=self.producer,
            precommit_monotonic_ns=self.precommit_monotonic_ns,
        )
        _require_bool(self.worker_exit_observed, "raw worker-exit fact", expected=True)
        for value, label in (
            (self.descendant_process_count, "raw descendant process count"),
            (
                self.measured_writable_namespace_holder_count,
                "raw measured-writable namespace-holder count",
            ),
            (self.measured_writable_fd_count, "raw measured-writable FD count"),
            (self.later_writer_count, "raw later-writer count"),
        ):
            _require_int(value, label, maximum=0)
        _require_bool(self.mount_mutation_enabled, "raw mount-mutation enabled", expected=False)
        _require_bool(
            self.write_quiescence_irreversible,
            "raw irreversible write-quiescence fact",
            expected=True,
        )
        capture = _require_raw_capture(self.capture, "raw write-seal capture")
        _require_distinct_sha256s(
            (
                capture.raw_sha256,
                *_runtime_identity_hashes(runtime),
                *_identity_hashes(artifacts),
                producer.descriptor_file_sha256,
                producer.descriptor_body_sha256,
                producer.source_sha256,
            ),
            "raw write-seal capture and producer identities",
        )
        expected_capture = canonical_raw_write_seal_capture_bytes(
            runtime=self.runtime,
            publication_reload_validation=self.publication_reload_validation,
            terminal_relay_preseal_attestation=self.terminal_relay_preseal_attestation,
            nonstorage_channel_preseal_attestation=(self.nonstorage_channel_preseal_attestation),
            tmpfs_hard_limit_mount_mutation_closure=(self.tmpfs_hard_limit_mount_mutation_closure),
            writable_fd_lifetime_inventory=self.writable_fd_lifetime_inventory,
            docker_api_operation_journal=self.docker_api_operation_journal,
            rootfs_upperdir_interval_delta=self.rootfs_upperdir_interval_delta,
            docker_volume_inventory_delta=self.docker_volume_inventory_delta,
            producer=self.producer,
            precommit_monotonic_ns=self.precommit_monotonic_ns,
        )
        observed_capture = _codec_call(
            "raw write-seal capture",
            capture.decoded_bytes,
        )
        if not hmac.compare_digest(expected_capture, observed_capture):
            _fail("raw write-seal capture differs from its exact replay")
        decoded = _codec_call(
            "raw write-seal capture",
            lambda: codec.decode_canonical_json_file(observed_capture),
        )
        if frozenset(decoded) != _CAPTURE_KEYS:
            _fail("raw write-seal capture key set differs")

    def to_body_dict(self) -> dict[str, Any]:
        RawWriteSealV1.__post_init__(self)
        result = _raw_capture_dict(
            runtime=self.runtime,
            publication_reload_validation=self.publication_reload_validation,
            terminal_relay_preseal_attestation=self.terminal_relay_preseal_attestation,
            nonstorage_channel_preseal_attestation=(self.nonstorage_channel_preseal_attestation),
            tmpfs_hard_limit_mount_mutation_closure=(self.tmpfs_hard_limit_mount_mutation_closure),
            writable_fd_lifetime_inventory=self.writable_fd_lifetime_inventory,
            docker_api_operation_journal=self.docker_api_operation_journal,
            rootfs_upperdir_interval_delta=self.rootfs_upperdir_interval_delta,
            docker_volume_inventory_delta=self.docker_volume_inventory_delta,
            producer=self.producer,
            precommit_monotonic_ns=self.precommit_monotonic_ns,
        )
        result.pop("capture_format")
        result.pop("runtime")
        result.update(self.runtime.to_dict())
        result.update(
            {
                "capture": self.capture.to_dict(),
                "schema_version": self.schema_version,
                "status": self.status,
            }
        )
        return result


def build_raw_write_seal_v1(
    *,
    observation_bytes: bytes,
    runtime: WriteSealRuntimeEnvelopeV1,
    publication_reload_validation: codec.ArtifactRefV1,
    terminal_relay_preseal_attestation: codec.ArtifactRefV1,
    nonstorage_channel_preseal_attestation: codec.ArtifactRefV1,
    tmpfs_hard_limit_mount_mutation_closure: codec.ArtifactRefV1,
    writable_fd_lifetime_inventory: codec.ArtifactRefV1,
    docker_api_operation_journal: codec.ArtifactRefV1,
    rootfs_upperdir_interval_delta: codec.ArtifactRefV1,
    docker_volume_inventory_delta: codec.ArtifactRefV1,
    producer: codec.ProducerRefV1,
    precommit_monotonic_ns: int,
) -> RawWriteSealV1:
    """Create raw7 only from caller-supplied, exactly replayed observation bytes."""

    if (
        type(observation_bytes) is not bytes
        or not observation_bytes
        or len(observation_bytes) > codec.MAX_RAW_CAPTURE_BYTES
    ):
        _fail("raw write-seal observation must be one nonempty bounded exact byte string")
    decoded = _codec_call(
        "raw write-seal observation",
        lambda: codec.decode_canonical_json_file(observation_bytes),
    )
    if frozenset(decoded) != _CAPTURE_KEYS:
        _fail("raw write-seal observation key set differs")
    expected_observation = canonical_raw_write_seal_capture_bytes(
        runtime=runtime,
        publication_reload_validation=publication_reload_validation,
        terminal_relay_preseal_attestation=terminal_relay_preseal_attestation,
        nonstorage_channel_preseal_attestation=nonstorage_channel_preseal_attestation,
        tmpfs_hard_limit_mount_mutation_closure=tmpfs_hard_limit_mount_mutation_closure,
        writable_fd_lifetime_inventory=writable_fd_lifetime_inventory,
        docker_api_operation_journal=docker_api_operation_journal,
        rootfs_upperdir_interval_delta=rootfs_upperdir_interval_delta,
        docker_volume_inventory_delta=docker_volume_inventory_delta,
        producer=producer,
        precommit_monotonic_ns=precommit_monotonic_ns,
    )
    if not hmac.compare_digest(expected_observation, observation_bytes):
        _fail("raw write-seal observation differs from its exact structured replay")
    capture = _codec_call(
        "raw write-seal observation capture",
        lambda: codec.RawByteCaptureV1.from_bytes(observation_bytes),
    )
    return RawWriteSealV1(
        runtime=runtime,
        publication_reload_validation=publication_reload_validation,
        terminal_relay_preseal_attestation=terminal_relay_preseal_attestation,
        nonstorage_channel_preseal_attestation=nonstorage_channel_preseal_attestation,
        tmpfs_hard_limit_mount_mutation_closure=(tmpfs_hard_limit_mount_mutation_closure),
        writable_fd_lifetime_inventory=writable_fd_lifetime_inventory,
        docker_api_operation_journal=docker_api_operation_journal,
        rootfs_upperdir_interval_delta=rootfs_upperdir_interval_delta,
        docker_volume_inventory_delta=docker_volume_inventory_delta,
        producer=producer,
        precommit_monotonic_ns=precommit_monotonic_ns,
        capture=capture,
    )


@final
@dataclass(frozen=True, slots=True)
class WriteQuiescenceSealV1:
    """Irreversible seal that binds raw7 after all measured writers are closed."""

    runtime: WriteSealRuntimeEnvelopeV1
    publication_reload_validation: codec.ArtifactRefV1
    terminal_relay_preseal_attestation: codec.ArtifactRefV1
    nonstorage_channel_preseal_attestation: codec.ArtifactRefV1
    raw_write_seal: codec.ArtifactRefV1
    producer: codec.ProducerRefV1
    seal_monotonic_ns: int
    schema_version: str = IRREVERSIBLE_WRITE_SEAL_SCHEMA_VERSION
    status: str = IRREVERSIBLE_WRITE_SEAL_STATUS
    architecture_kind: str = WRITE_SEAL_ARCHITECTURE
    publication_committed_before_seal: bool = True
    reload_validated_before_seal: bool = True
    relay_ready_before_seal: bool = True
    channel_ready_before_seal: bool = True
    container_writes_disabled: bool = True
    descendant_writes_disabled: bool = True
    later_allocation_possible: bool = False
    later_copy_up_possible: bool = False
    later_peak_increase_possible: bool = False
    no_later_writer_exists: bool = True
    teardown_deletion_only: bool = True
    teardown_can_increase_measured_usage: bool = False
    write_quiescence_irreversible: bool = True

    def __init_subclass__(cls, **kwargs: Any) -> None:
        raise TypeError("WriteQuiescenceSealV1 is runtime-final")

    def __post_init__(self) -> None:
        _require_exact_text(
            self.schema_version,
            IRREVERSIBLE_WRITE_SEAL_SCHEMA_VERSION,
            "write-quiescence seal schema",
        )
        _require_exact_text(
            self.status,
            IRREVERSIBLE_WRITE_SEAL_STATUS,
            "write-quiescence seal status",
        )
        runtime = _require_runtime(
            self.runtime,
            "write-quiescence seal runtime envelope",
        )
        artifacts = (
            runtime.runtime_intent,
            runtime.host_go,
            _require_artifact(
                self.publication_reload_validation,
                PUBLICATION_RELOAD_VALIDATION_SCHEMA_VERSION,
                "sealed publication reload validation",
            ),
            _require_artifact(
                self.terminal_relay_preseal_attestation,
                TERMINAL_RELAY_PRESEAL_ATTESTATION_SCHEMA_VERSION,
                "sealed terminal relay preseal attestation",
            ),
            _require_artifact(
                self.nonstorage_channel_preseal_attestation,
                NONSTORAGE_CHANNEL_PRESEAL_ATTESTATION_SCHEMA_VERSION,
                "sealed nonstorage channel preseal attestation",
            ),
            _require_artifact(
                self.raw_write_seal,
                RAW_WRITE_SEAL_SCHEMA_VERSION,
                "sealed raw write-seal observation",
            ),
        )
        producer = _require_producer(self.producer, "write-quiescence seal")
        _require_distinct_sha256s(
            _identity_hashes(artifacts)
            + (
                runtime.image_id.removeprefix("sha256:"),
                runtime.container_id_commitment_sha256,
                runtime.outer_cgroup_identity_sha256,
            )
            + (
                producer.descriptor_file_sha256,
                producer.descriptor_body_sha256,
                producer.source_sha256,
            ),
            "write-quiescence seal artifact and producer identities",
        )
        _require_int(
            self.seal_monotonic_ns,
            "write-seal commit monotonic time",
            minimum=1,
        )
        _require_exact_text(
            self.architecture_kind,
            WRITE_SEAL_ARCHITECTURE,
            "write-seal architecture",
        )
        for value, label in (
            (self.publication_committed_before_seal, "publication committed before seal"),
            (self.reload_validated_before_seal, "reload validated before seal"),
            (self.relay_ready_before_seal, "terminal relay ready before seal"),
            (self.channel_ready_before_seal, "nonstorage channel ready before seal"),
            (self.container_writes_disabled, "container writes disabled"),
            (self.descendant_writes_disabled, "descendant writes disabled"),
            (self.no_later_writer_exists, "no later writer exists"),
            (self.teardown_deletion_only, "teardown deletion-only"),
            (self.write_quiescence_irreversible, "write quiescence irreversible"),
        ):
            _require_bool(value, label, expected=True)
        for value, label in (
            (self.later_allocation_possible, "later allocation possible"),
            (self.later_copy_up_possible, "later copy-up possible"),
            (self.later_peak_increase_possible, "later peak increase possible"),
            (
                self.teardown_can_increase_measured_usage,
                "teardown can increase measured usage",
            ),
        ):
            _require_bool(value, label, expected=False)

    def to_body_dict(self) -> dict[str, Any]:
        WriteQuiescenceSealV1.__post_init__(self)
        result = self.runtime.to_dict()
        result.update(
            {
                "architecture_kind": self.architecture_kind,
                "channel_ready_before_seal": self.channel_ready_before_seal,
                "container_writes_disabled": self.container_writes_disabled,
                "descendant_writes_disabled": self.descendant_writes_disabled,
                "later_allocation_possible": self.later_allocation_possible,
                "later_copy_up_possible": self.later_copy_up_possible,
                "later_peak_increase_possible": self.later_peak_increase_possible,
                "no_later_writer_exists": self.no_later_writer_exists,
                "nonstorage_channel_preseal_attestation": (
                    self.nonstorage_channel_preseal_attestation.to_dict()
                ),
                "producer": self.producer.to_dict(),
                "publication_committed_before_seal": self.publication_committed_before_seal,
                "publication_reload_validation": self.publication_reload_validation.to_dict(),
                "raw_write_seal": self.raw_write_seal.to_dict(),
                "relay_ready_before_seal": self.relay_ready_before_seal,
                "reload_validated_before_seal": self.reload_validated_before_seal,
                "schema_version": self.schema_version,
                "seal_monotonic_ns": self.seal_monotonic_ns,
                "status": self.status,
                "teardown_can_increase_measured_usage": (self.teardown_can_increase_measured_usage),
                "teardown_deletion_only": self.teardown_deletion_only,
                "terminal_relay_preseal_attestation": (
                    self.terminal_relay_preseal_attestation.to_dict()
                ),
                "write_quiescence_irreversible": self.write_quiescence_irreversible,
            }
        )
        return result


@final
@dataclass(frozen=True, slots=True)
class RawWriteSealProjectionV1:
    """In-memory exact raw7 projection used by the storage receipt."""

    first_class_artifact: codec.ArtifactRefV1
    raw_artifact: codec.ArtifactRefV1
    predecessors: tuple[codec.ArtifactRefV1, ...]
    producer: codec.ProducerRefV1
    kind: Literal["write_seal"] = "write_seal"
    exact_projection: bool = True
    predecessor_identity_bound: bool = True

    def __init_subclass__(cls, **kwargs: Any) -> None:
        raise TypeError("RawWriteSealProjectionV1 is runtime-final")

    def __post_init__(self) -> None:
        first = _require_artifact(
            self.first_class_artifact,
            IRREVERSIBLE_WRITE_SEAL_SCHEMA_VERSION,
            "raw7 projected seal",
        )
        raw = _require_artifact(self.raw_artifact, RAW_WRITE_SEAL_SCHEMA_VERSION, "raw7 projection")
        if type(self.predecessors) is not tuple or len(self.predecessors) != 3:
            _fail("raw7 projection predecessor tuple differs")
        schemas = (
            PUBLICATION_RELOAD_VALIDATION_SCHEMA_VERSION,
            TERMINAL_RELAY_PRESEAL_ATTESTATION_SCHEMA_VERSION,
            NONSTORAGE_CHANNEL_PRESEAL_ATTESTATION_SCHEMA_VERSION,
        )
        exact_predecessors = tuple(
            _require_artifact(artifact, schema, "raw7 projection predecessor")
            for artifact, schema in zip(self.predecessors, schemas, strict=True)
        )
        producer = _require_producer(self.producer, "raw7 projection")
        _require_distinct_sha256s(
            _identity_hashes((first, raw, *exact_predecessors))
            + (
                producer.descriptor_file_sha256,
                producer.descriptor_body_sha256,
                producer.source_sha256,
            ),
            "raw7 projection identities",
        )
        _require_exact_text(self.kind, "write_seal", "raw7 projection kind")
        _require_bool(self.exact_projection, "raw7 exact projection", expected=True)
        _require_bool(
            self.predecessor_identity_bound,
            "raw7 predecessor identity binding",
            expected=True,
        )

    def to_dict(self) -> dict[str, Any]:
        RawWriteSealProjectionV1.__post_init__(self)
        return {
            "exact_projection": self.exact_projection,
            "first_class_artifact": self.first_class_artifact.to_dict(),
            "kind": self.kind,
            "predecessor_identity_bound": self.predecessor_identity_bound,
            "predecessors": [artifact.to_dict() for artifact in self.predecessors],
            "producer": self.producer.to_dict(),
            "raw_artifact": self.raw_artifact.to_dict(),
        }


@final
@dataclass(frozen=True, slots=True)
class WriteSealProducerDescriptorV2:
    """Source-only self-description with no operational capability or authority."""

    schema_version: str = WRITE_SEAL_PRODUCER_DESCRIPTOR_SCHEMA_VERSION
    status: str = WRITE_SEAL_DESCRIPTOR_STATUS
    role: Literal["write_seal_producer"] = "write_seal_producer"
    owned_artifact_schemas: tuple[str, ...] = OWNED_ARTIFACT_SCHEMAS
    required_input_schemas: tuple[str, ...] = REQUIRED_INPUT_SCHEMAS
    canonical_policy: Mapping[str, bool] = CANONICAL_POLICY
    limits: Mapping[str, int] = LIMITS
    capabilities: Mapping[str, bool] = SOURCE_ONLY_CAPABILITIES
    readiness: Mapping[str, bool] = SOURCE_ONLY_READINESS
    authority: Mapping[str, bool] = SOURCE_ONLY_AUTHORITY
    claims: Mapping[str, bool] = SOURCE_ONLY_CLAIMS
    public_canonical_builders: tuple[str, ...] = PUBLIC_CANONICAL_BUILDERS
    public_parsers: tuple[str, ...] = PUBLIC_PARSERS
    public_validators: tuple[str, ...] = PUBLIC_VALIDATORS
    operational_apis: tuple[()] = ()
    descriptor_self_pin_sha256: str = DESCRIPTOR_SERIALIZED_SELF_PIN_SENTINEL
    source_file_sha256_pin: None = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        raise TypeError("WriteSealProducerDescriptorV2 is runtime-final")

    def __post_init__(self) -> None:
        _require_exact_text(
            self.schema_version,
            WRITE_SEAL_PRODUCER_DESCRIPTOR_SCHEMA_VERSION,
            "write-seal producer descriptor schema",
        )
        _require_exact_text(
            self.status,
            WRITE_SEAL_DESCRIPTOR_STATUS,
            "write-seal producer descriptor status",
        )
        _require_exact_text(
            self.role,
            "write_seal_producer",
            "write-seal producer descriptor role",
        )
        for tuple_observed, tuple_expected, tuple_label in (
            (self.owned_artifact_schemas, OWNED_ARTIFACT_SCHEMAS, "owned artifact schemas"),
            (self.required_input_schemas, REQUIRED_INPUT_SCHEMAS, "required input schemas"),
            (
                self.public_canonical_builders,
                PUBLIC_CANONICAL_BUILDERS,
                "public canonical builders",
            ),
            (self.public_parsers, PUBLIC_PARSERS, "public parsers"),
            (self.public_validators, PUBLIC_VALIDATORS, "public validators"),
        ):
            _require_exact_text_tuple(
                tuple_observed,
                tuple_expected,
                f"write-seal descriptor {tuple_label}",
            )
        canonical_policy = _detached_descriptor_bool_mapping(
            self.canonical_policy,
            CANONICAL_POLICY,
            "canonical policy",
        )
        limits = _detached_descriptor_int_mapping(self.limits, LIMITS, "limits")
        capabilities = _detached_descriptor_bool_mapping(
            self.capabilities,
            SOURCE_ONLY_CAPABILITIES,
            "capabilities",
        )
        readiness = _detached_descriptor_bool_mapping(
            self.readiness,
            SOURCE_ONLY_READINESS,
            "readiness",
        )
        authority = _detached_descriptor_bool_mapping(
            self.authority,
            SOURCE_ONLY_AUTHORITY,
            "authority",
        )
        claims = _detached_descriptor_bool_mapping(
            self.claims,
            SOURCE_ONLY_CLAIMS,
            "claims",
        )
        object.__setattr__(self, "canonical_policy", canonical_policy)
        object.__setattr__(self, "limits", limits)
        object.__setattr__(self, "capabilities", capabilities)
        object.__setattr__(self, "readiness", readiness)
        object.__setattr__(self, "authority", authority)
        object.__setattr__(self, "claims", claims)
        if type(self.operational_apis) is not tuple or self.operational_apis:
            _fail("write-seal descriptor operational APIs must be exact empty")
        _require_exact_text(
            self.descriptor_self_pin_sha256,
            DESCRIPTOR_SERIALIZED_SELF_PIN_SENTINEL,
            "write-seal descriptor serialized self-pin sentinel",
        )
        if self.source_file_sha256_pin is not None:
            _fail("write-seal descriptor must not embed its source-file identity")

    def to_body_dict(self) -> dict[str, Any]:
        WriteSealProducerDescriptorV2.__post_init__(self)
        return {
            "authority": dict(self.authority),
            "canonical_policy": dict(self.canonical_policy),
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


def canonical_raw_write_seal_v1_body_bytes(value: RawWriteSealV1) -> bytes:
    if type(value) is not RawWriteSealV1:
        _fail("raw write-seal BODY value type differs")
    validate_raw_write_seal_v1(value)
    return codec.canonical_json_bytes(value.to_body_dict(), final_lf=False)


def canonical_raw_write_seal_v1_file_bytes(value: RawWriteSealV1) -> bytes:
    if type(value) is not RawWriteSealV1:
        _fail("raw write-seal file value type differs")
    validate_raw_write_seal_v1(value)
    return codec.canonical_file_bytes(
        value.to_body_dict(),
        body_digest_field=RAW_WRITE_SEAL_BODY_SHA256_FIELD,
    )


def canonical_write_quiescence_seal_v1_body_bytes(value: WriteQuiescenceSealV1) -> bytes:
    if type(value) is not WriteQuiescenceSealV1:
        _fail("write-quiescence seal BODY value type differs")
    validate_write_quiescence_seal_v1(value)
    return codec.canonical_json_bytes(value.to_body_dict(), final_lf=False)


def canonical_write_quiescence_seal_v1_file_bytes(value: WriteQuiescenceSealV1) -> bytes:
    if type(value) is not WriteQuiescenceSealV1:
        _fail("write-quiescence seal file value type differs")
    validate_write_quiescence_seal_v1(value)
    return codec.canonical_file_bytes(
        value.to_body_dict(),
        body_digest_field=IRREVERSIBLE_WRITE_SEAL_BODY_SHA256_FIELD,
    )


def canonical_write_seal_producer_descriptor_v2_body_bytes(
    value: WriteSealProducerDescriptorV2,
) -> bytes:
    if type(value) is not WriteSealProducerDescriptorV2:
        _fail("write-seal producer descriptor BODY value type differs")
    WriteSealProducerDescriptorV2.__post_init__(value)
    return codec.canonical_json_bytes(value.to_body_dict(), final_lf=False)


def canonical_write_seal_producer_descriptor_v2_file_bytes(
    value: WriteSealProducerDescriptorV2,
) -> bytes:
    if type(value) is not WriteSealProducerDescriptorV2:
        _fail("write-seal producer descriptor file value type differs")
    WriteSealProducerDescriptorV2.__post_init__(value)
    return codec.canonical_file_bytes(
        value.to_body_dict(),
        body_digest_field=WRITE_SEAL_PRODUCER_DESCRIPTOR_BODY_SHA256_FIELD,
    )


def raw_write_seal_identity_v1(value: RawWriteSealV1) -> codec.ArtifactRefV1:
    validate_raw_write_seal_v1(value)
    body = canonical_raw_write_seal_v1_body_bytes(value)
    file = canonical_raw_write_seal_v1_file_bytes(value)
    return codec.ArtifactRefV1(
        schema_version=RAW_WRITE_SEAL_SCHEMA_VERSION,
        file_sha256=hashlib.sha256(file).hexdigest(),
        body_sha256=hashlib.sha256(body).hexdigest(),
    )


def write_quiescence_seal_identity_v1(
    value: WriteQuiescenceSealV1,
) -> codec.ArtifactRefV1:
    validate_write_quiescence_seal_v1(value)
    body = canonical_write_quiescence_seal_v1_body_bytes(value)
    file = canonical_write_quiescence_seal_v1_file_bytes(value)
    return codec.ArtifactRefV1(
        schema_version=IRREVERSIBLE_WRITE_SEAL_SCHEMA_VERSION,
        file_sha256=hashlib.sha256(file).hexdigest(),
        body_sha256=hashlib.sha256(body).hexdigest(),
    )


def _raw_from_body(body: dict[str, Any]) -> RawWriteSealV1:
    item = _exact_dict(body, _RAW_BODY_KEYS, "raw write-seal BODY")
    return RawWriteSealV1(
        runtime=_runtime_from_flat_body(item),
        publication_reload_validation=_artifact_from_dict(
            item["publication_reload_validation"],
            "publication reload validation",
        ),
        terminal_relay_preseal_attestation=_artifact_from_dict(
            item["terminal_relay_preseal_attestation"],
            "terminal relay preseal attestation",
        ),
        nonstorage_channel_preseal_attestation=_artifact_from_dict(
            item["nonstorage_channel_preseal_attestation"],
            "nonstorage channel preseal attestation",
        ),
        tmpfs_hard_limit_mount_mutation_closure=_artifact_from_dict(
            item["tmpfs_hard_limit_mount_mutation_closure"],
            "tmpfs hard-limit mount-mutation closure",
        ),
        writable_fd_lifetime_inventory=_artifact_from_dict(
            item["writable_fd_lifetime_inventory"],
            "writable-FD lifetime inventory",
        ),
        docker_api_operation_journal=_artifact_from_dict(
            item["docker_api_operation_journal"],
            "Docker API operation journal",
        ),
        rootfs_upperdir_interval_delta=_artifact_from_dict(
            item["rootfs_upperdir_interval_delta"],
            "rootfs upperdir interval delta",
        ),
        docker_volume_inventory_delta=_artifact_from_dict(
            item["docker_volume_inventory_delta"],
            "Docker volume inventory delta",
        ),
        producer=_producer_from_dict(item["producer"], "raw write seal producer"),
        precommit_monotonic_ns=item["precommit_monotonic_ns"],
        capture=_raw_capture_from_dict(item["capture"], "raw write-seal capture"),
        schema_version=item["schema_version"],
        status=item["status"],
        worker_exit_observed=item["worker_exit_observed"],
        descendant_process_count=item["descendant_process_count"],
        measured_writable_namespace_holder_count=(item["measured_writable_namespace_holder_count"]),
        measured_writable_fd_count=item["measured_writable_fd_count"],
        later_writer_count=item["later_writer_count"],
        mount_mutation_enabled=item["mount_mutation_enabled"],
        write_quiescence_irreversible=item["write_quiescence_irreversible"],
    )


def _seal_from_body(body: dict[str, Any]) -> WriteQuiescenceSealV1:
    item = _exact_dict(body, _SEAL_BODY_KEYS, "write-quiescence seal BODY")
    return WriteQuiescenceSealV1(
        runtime=_runtime_from_flat_body(item),
        publication_reload_validation=_artifact_from_dict(
            item["publication_reload_validation"],
            "sealed publication reload validation",
        ),
        terminal_relay_preseal_attestation=_artifact_from_dict(
            item["terminal_relay_preseal_attestation"],
            "sealed terminal relay preseal attestation",
        ),
        nonstorage_channel_preseal_attestation=_artifact_from_dict(
            item["nonstorage_channel_preseal_attestation"],
            "sealed nonstorage channel preseal attestation",
        ),
        raw_write_seal=_artifact_from_dict(item["raw_write_seal"], "sealed raw write seal"),
        producer=_producer_from_dict(item["producer"], "write-quiescence seal producer"),
        seal_monotonic_ns=item["seal_monotonic_ns"],
        schema_version=item["schema_version"],
        status=item["status"],
        architecture_kind=item["architecture_kind"],
        publication_committed_before_seal=item["publication_committed_before_seal"],
        reload_validated_before_seal=item["reload_validated_before_seal"],
        relay_ready_before_seal=item["relay_ready_before_seal"],
        channel_ready_before_seal=item["channel_ready_before_seal"],
        container_writes_disabled=item["container_writes_disabled"],
        descendant_writes_disabled=item["descendant_writes_disabled"],
        later_allocation_possible=item["later_allocation_possible"],
        later_copy_up_possible=item["later_copy_up_possible"],
        later_peak_increase_possible=item["later_peak_increase_possible"],
        no_later_writer_exists=item["no_later_writer_exists"],
        teardown_deletion_only=item["teardown_deletion_only"],
        teardown_can_increase_measured_usage=item["teardown_can_increase_measured_usage"],
        write_quiescence_irreversible=item["write_quiescence_irreversible"],
    )


def _descriptor_from_body(body: dict[str, Any]) -> WriteSealProducerDescriptorV2:
    item = _exact_dict(body, _DESCRIPTOR_BODY_KEYS, "write-seal producer descriptor BODY")
    source_pin = item["source_file_sha256_pin"]
    if source_pin is not None:
        _fail("write-seal producer descriptor source pin differs")
    operational = item["operational_apis"]
    if type(operational) is not list or operational:
        _fail("write-seal producer descriptor operational APIs differ")
    return WriteSealProducerDescriptorV2(
        schema_version=item["schema_version"],
        status=item["status"],
        role=item["role"],
        owned_artifact_schemas=_exact_text_tuple(
            item["owned_artifact_schemas"], "owned artifact schemas"
        ),
        required_input_schemas=_exact_text_tuple(
            item["required_input_schemas"], "required input schemas"
        ),
        canonical_policy=_exact_bool_mapping(
            item["canonical_policy"], CANONICAL_POLICY, "canonical policy"
        ),
        limits=_exact_int_mapping(item["limits"], LIMITS, "limits"),
        capabilities=_exact_bool_mapping(
            item["capabilities"], SOURCE_ONLY_CAPABILITIES, "capabilities"
        ),
        readiness=_exact_bool_mapping(item["readiness"], SOURCE_ONLY_READINESS, "readiness"),
        authority=_exact_bool_mapping(item["authority"], SOURCE_ONLY_AUTHORITY, "authority"),
        claims=_exact_bool_mapping(item["claims"], SOURCE_ONLY_CLAIMS, "claims"),
        public_canonical_builders=_exact_text_tuple(
            item["public_canonical_builders"], "public canonical builders"
        ),
        public_parsers=_exact_text_tuple(item["public_parsers"], "public parsers"),
        public_validators=_exact_text_tuple(item["public_validators"], "public validators"),
        operational_apis=(),
        descriptor_self_pin_sha256=item["descriptor_self_pin_sha256"],
        source_file_sha256_pin=None,
    )


def parse_raw_write_seal_v1(
    raw: bytes,
    *,
    expected_file_sha256: str,
    expected_body_sha256: str,
) -> RawWriteSealV1:
    body = _codec_call(
        "raw write-seal file",
        lambda: codec.validate_canonical_file(
            raw,
            expected_file_sha256=expected_file_sha256,
            expected_body_sha256=expected_body_sha256,
            body_digest_field=RAW_WRITE_SEAL_BODY_SHA256_FIELD,
        ),
    )
    value = _raw_from_body(body)
    if not hmac.compare_digest(canonical_raw_write_seal_v1_file_bytes(value), raw):
        _fail("raw write-seal canonical replay differs")
    return value


def parse_write_quiescence_seal_v1(
    raw: bytes,
    *,
    expected_file_sha256: str,
    expected_body_sha256: str,
) -> WriteQuiescenceSealV1:
    body = _codec_call(
        "write-quiescence seal file",
        lambda: codec.validate_canonical_file(
            raw,
            expected_file_sha256=expected_file_sha256,
            expected_body_sha256=expected_body_sha256,
            body_digest_field=IRREVERSIBLE_WRITE_SEAL_BODY_SHA256_FIELD,
        ),
    )
    value = _seal_from_body(body)
    if not hmac.compare_digest(canonical_write_quiescence_seal_v1_file_bytes(value), raw):
        _fail("write-quiescence seal canonical replay differs")
    return value


def parse_write_seal_producer_descriptor_v2(
    raw: bytes,
    *,
    expected_file_sha256: str,
    expected_body_sha256: str,
) -> WriteSealProducerDescriptorV2:
    body = _codec_call(
        "write-seal producer descriptor file",
        lambda: codec.validate_canonical_file(
            raw,
            expected_file_sha256=expected_file_sha256,
            expected_body_sha256=expected_body_sha256,
            body_digest_field=WRITE_SEAL_PRODUCER_DESCRIPTOR_BODY_SHA256_FIELD,
        ),
    )
    value = _descriptor_from_body(body)
    if not hmac.compare_digest(canonical_write_seal_producer_descriptor_v2_file_bytes(value), raw):
        _fail("write-seal producer descriptor canonical replay differs")
    return value


def validate_raw_write_seal_v1(value: RawWriteSealV1) -> None:
    if type(value) is not RawWriteSealV1:
        _fail("raw write-seal validation type differs")
    RawWriteSealV1.__post_init__(value)


def validate_raw_write_seal_projection_v1(value: RawWriteSealProjectionV1) -> None:
    if type(value) is not RawWriteSealProjectionV1:
        _fail("raw7 projection validation type differs")
    RawWriteSealProjectionV1.__post_init__(value)


def validate_write_quiescence_seal_v1(value: WriteQuiescenceSealV1) -> None:
    if type(value) is not WriteQuiescenceSealV1:
        _fail("write-quiescence seal validation type differs")
    WriteQuiescenceSealV1.__post_init__(value)


def validate_write_seal_chain_v2(
    raw: RawWriteSealV1,
    value: WriteQuiescenceSealV1,
) -> None:
    validate_raw_write_seal_v1(raw)
    validate_write_quiescence_seal_v1(value)
    if value.runtime != raw.runtime:
        _fail("write-seal runtime envelope crosswires raw7")
    for observed, expected, label in (
        (
            value.publication_reload_validation,
            raw.publication_reload_validation,
            "publication reload validation",
        ),
        (
            value.terminal_relay_preseal_attestation,
            raw.terminal_relay_preseal_attestation,
            "terminal relay preseal attestation",
        ),
        (
            value.nonstorage_channel_preseal_attestation,
            raw.nonstorage_channel_preseal_attestation,
            "nonstorage channel preseal attestation",
        ),
        (value.producer, raw.producer, "producer"),
    ):
        if observed != expected:
            _fail(f"write-seal {label} crosswires raw7")
    if value.raw_write_seal != raw_write_seal_identity_v1(raw):
        _fail("write-seal raw7 identity differs")
    if value.seal_monotonic_ns <= raw.precommit_monotonic_ns:
        _fail("write-seal commit must follow raw7 precommit strictly")


def raw_write_seal_projection_v1(
    raw: RawWriteSealV1,
    value: WriteQuiescenceSealV1,
) -> RawWriteSealProjectionV1:
    validate_write_seal_chain_v2(raw, value)
    return RawWriteSealProjectionV1(
        first_class_artifact=write_quiescence_seal_identity_v1(value),
        raw_artifact=raw_write_seal_identity_v1(raw),
        predecessors=(
            raw.publication_reload_validation,
            raw.terminal_relay_preseal_attestation,
            raw.nonstorage_channel_preseal_attestation,
        ),
        producer=raw.producer,
    )


def write_seal_producer_descriptor_observed_hashes_v2() -> tuple[str, str]:
    descriptor = WriteSealProducerDescriptorV2()
    file = canonical_write_seal_producer_descriptor_v2_file_bytes(descriptor)
    body = canonical_write_seal_producer_descriptor_v2_body_bytes(descriptor)
    return hashlib.sha256(file).hexdigest(), hashlib.sha256(body).hexdigest()


if (
    DESCRIPTOR_SERIALIZED_SELF_PIN_SENTINEL,
    PINNED_WRITE_SEAL_PRODUCER_DESCRIPTOR_FILE_SHA256,
    PINNED_WRITE_SEAL_PRODUCER_DESCRIPTOR_BODY_SHA256,
    PINNED_WRITE_SEAL_PRODUCER_SOURCE_SHA256,
) != (ZERO_SHA256,) * 4:
    raise AssertionError("write-seal producer pins must remain explicitly unfinalized")


__all__ = (
    "WRITE_SEAL_PRODUCER_DESCRIPTOR_SCHEMA_VERSION",
    "RAW_WRITE_SEAL_SCHEMA_VERSION",
    "IRREVERSIBLE_WRITE_SEAL_SCHEMA_VERSION",
    "STORAGE_BOUNDARY_RUNTIME_INTENT_SCHEMA_VERSION",
    "HOST_GO_V3_SCHEMA_VERSION",
    "PUBLICATION_RELOAD_VALIDATION_SCHEMA_VERSION",
    "TERMINAL_RELAY_PRESEAL_ATTESTATION_SCHEMA_VERSION",
    "NONSTORAGE_CHANNEL_PRESEAL_ATTESTATION_SCHEMA_VERSION",
    "TMPFS_HARD_LIMIT_MOUNT_MUTATION_CLOSURE_SCHEMA_VERSION",
    "WRITABLE_FD_LIFETIME_INVENTORY_SCHEMA_VERSION",
    "DOCKER_API_OPERATION_JOURNAL_SCHEMA_VERSION",
    "ROOTFS_UPPERDIR_INTERVAL_DELTA_SCHEMA_VERSION",
    "DOCKER_VOLUME_INVENTORY_DELTA_SCHEMA_VERSION",
    "RAW_WRITE_SEAL_CAPTURE_FORMAT",
    "WRITE_SEAL_ARCHITECTURE",
    "WRITE_SEAL_DESCRIPTOR_STATUS",
    "RAW_WRITE_SEAL_STATUS",
    "IRREVERSIBLE_WRITE_SEAL_STATUS",
    "WRITE_SEAL_PRODUCER_DESCRIPTOR_BODY_SHA256_FIELD",
    "RAW_WRITE_SEAL_BODY_SHA256_FIELD",
    "IRREVERSIBLE_WRITE_SEAL_BODY_SHA256_FIELD",
    "ZERO_SHA256",
    "DESCRIPTOR_SERIALIZED_SELF_PIN_SENTINEL",
    "PINNED_WRITE_SEAL_PRODUCER_DESCRIPTOR_FILE_SHA256",
    "PINNED_WRITE_SEAL_PRODUCER_DESCRIPTOR_BODY_SHA256",
    "PINNED_WRITE_SEAL_PRODUCER_SOURCE_SHA256",
    "OWNED_ARTIFACT_SCHEMAS",
    "REQUIRED_INPUT_SCHEMAS",
    "CANONICAL_POLICY",
    "LIMITS",
    "SOURCE_ONLY_CAPABILITIES",
    "SOURCE_ONLY_READINESS",
    "SOURCE_ONLY_AUTHORITY",
    "SOURCE_ONLY_CLAIMS",
    "PUBLIC_CANONICAL_BUILDERS",
    "PUBLIC_PARSERS",
    "PUBLIC_VALIDATORS",
    "ForagerMatchedV3StorageWriteSealProtocolV2Error",
    "WriteSealRuntimeEnvelopeV1",
    "RawWriteSealV1",
    "WriteQuiescenceSealV1",
    "RawWriteSealProjectionV1",
    "WriteSealProducerDescriptorV2",
    "canonical_raw_write_seal_capture_bytes",
    "build_raw_write_seal_v1",
    "canonical_raw_write_seal_v1_body_bytes",
    "canonical_raw_write_seal_v1_file_bytes",
    "canonical_write_quiescence_seal_v1_body_bytes",
    "canonical_write_quiescence_seal_v1_file_bytes",
    "canonical_write_seal_producer_descriptor_v2_body_bytes",
    "canonical_write_seal_producer_descriptor_v2_file_bytes",
    "raw_write_seal_identity_v1",
    "write_quiescence_seal_identity_v1",
    "parse_raw_write_seal_v1",
    "parse_write_quiescence_seal_v1",
    "parse_write_seal_producer_descriptor_v2",
    "validate_raw_write_seal_v1",
    "validate_raw_write_seal_projection_v1",
    "validate_write_quiescence_seal_v1",
    "validate_write_seal_chain_v2",
    "raw_write_seal_projection_v1",
    "write_seal_producer_descriptor_observed_hashes_v2",
)
