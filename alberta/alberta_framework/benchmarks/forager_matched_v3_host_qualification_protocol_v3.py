"""Pure additive host-qualification protocol metadata for matched Forager v3.

This module closes the typed pre-GO metadata boundary that the storage-v2
contract expects.  It performs no signature operation, host inspection,
filesystem access, container operation, process creation, issuance, evaluation,
resource merge, or workload execution.  A structurally valid artifact records
only what independently pinned producers report; parsing grants no authority.

The dependency direction is intentionally acyclic.  Host GO directly names the
already committed storage runtime-intent FILE and BODY identities and the exact
six storage producers.  There is no serialized host-GO binding artifact, and GO
does not name the future write seal.  The write seal is a later endpoint owned by
its independent producer.
"""

from __future__ import annotations

import hashlib
import hmac
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final, Literal, Never, Protocol, cast, final

from alberta_framework.benchmarks import (
    forager_matched_v3_host_provisioning_v3 as provisioning_v3,
)
from alberta_framework.benchmarks import (
    forager_matched_v3_host_qualification_executor_v2 as executor_v2,
)
from alberta_framework.benchmarks import (
    forager_matched_v3_qualification_plan_v3 as plan_v3,
)
from alberta_framework.benchmarks._forager_matched_v3_canonical_evidence import (
    PRODUCER_ROLES,
    ArtifactRefV1,
    CanonicalEvidenceError,
    CaseSubjectV1,
    ProducerRefV1,
    artifact_ref_v1_from_dict,
    canonical_file_bytes,
    canonical_json_bytes,
    case_subject_v1_from_dict,
    producer_ref_v1_from_dict,
    producer_refs_v1_from_json,
    require_distinct_sha256s,
    require_nonzero_sha256,
    validate_canonical_file,
    validate_producer_refs_v1,
)

# Public first-class schemas: one descriptor, four six-role trust artifacts,
# one validated pre-GO prefix, and the five additive host handshake artifacts.
HOST_QUALIFICATION_PROTOCOL_DESCRIPTOR_V3_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.host_qualification_protocol_descriptor.v3"
)
HOST_STORAGE_PRODUCER_TRUST_POLICY_V1_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.host_storage_producer_trust_policy.v1"
)
HOST_STORAGE_PRODUCER_INVENTORY_STATEMENT_V1_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.host_storage_producer_inventory_statement.v1"
)
HOST_STORAGE_PRODUCER_INVENTORY_SIGNATURE_VERIFICATION_RECEIPT_V1_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.host_storage_producer_inventory_signature_verification_receipt.v1"
)
HOST_STORAGE_PRODUCER_LIVE_VALIDATION_RECEIPT_V1_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.host_storage_producer_live_validation_receipt.v1"
)
HOST_PROVISIONING_VALIDATED_PRE_GO_PREFIX_V3_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.host_provisioning_validated_pre_go_prefix.v3"
)
HOST_CASE_REQUEST_V3_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.host_qualification_case_request.v3"
)
HOST_CASE_INTENT_V3_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.host_qualification_case_intent.v3"
)
HOST_READY_V3_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.in_container_qualification_driver_ready.v3"
)
HOST_OBSERVER_ANCHOR_V3_SCHEMA_VERSION: Final = "alberta.forager_matched_v3.host_observer_anchor.v3"
HOST_GO_V3_SCHEMA_VERSION: Final = "alberta.forager_matched_v3.host_qualification_go_commitment.v3"

# External schemas are duplicated as expected dependency literals so this module
# never imports storage-v2.  Storage-v2 must later assert literal equality.
QUALIFICATION_PLAN_V3_SCHEMA_VERSION: Final = plan_v3.QUALIFICATION_PLAN_V3_SCHEMA_VERSION
QUALIFICATION_PLAN_V3_DESCRIPTOR_SCHEMA_VERSION: Final = (
    plan_v3.QUALIFICATION_PLAN_V3_DESCRIPTOR_SCHEMA_VERSION
)
CASE_EXECUTION_TICKET_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.qualification_case_execution_ticket.v1"
)
RUNTIME_QUALIFICATION_RECEIPT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.runtime_qualification_receipt.v1"
)
STORAGE_BACKEND_CONTRACT_DESCRIPTOR_V2_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.qualification_storage_backend_contract_descriptor.v2"
)
STORAGE_BACKEND_POLICY_V1_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.storage_backend_policy.v1"
)
STORAGE_BOUNDARY_RUNTIME_INTENT_V1_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.storage_boundary_runtime_intent.v1"
)
STORAGE_BOUNDARY_RECEIPT_V2_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.storage_boundary_receipt.v2"
)
IRREVERSIBLE_WRITE_SEAL_V1_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.qualification_storage_write_quiescence_seal.v1"
)

HOST_CASE_REQUEST_V2_SCHEMA_VERSION: Final = executor_v2.HOST_CASE_REQUEST_SCHEMA_VERSION
HOST_CASE_INTENT_V2_SCHEMA_VERSION: Final = executor_v2.HOST_CASE_INTENT_SCHEMA_VERSION
HOST_INITIAL_CGROUP_SAMPLE_V2_SCHEMA_VERSION: Final = (
    executor_v2.HOST_INITIAL_CGROUP_SAMPLE_SCHEMA_VERSION
)
HOST_READY_V2_SCHEMA_VERSION: Final = executor_v2.HOST_READY_SCHEMA_VERSION
HOST_OBSERVER_ANCHOR_V2_SCHEMA_VERSION: Final = executor_v2.HOST_OBSERVER_ANCHOR_SCHEMA_VERSION
HOST_GO_V2_SCHEMA_VERSION: Final = executor_v2.HOST_GO_SCHEMA_VERSION

HOST_STORAGE_PRODUCER_SIGNATURE_ALGORITHM: Final = "ed25519"
HOST_STORAGE_PRODUCER_SIGNATURE_DOMAIN_LABEL: Final = (
    HOST_STORAGE_PRODUCER_INVENTORY_STATEMENT_V1_SCHEMA_VERSION
)
HOST_STORAGE_PRODUCER_SIGNATURE_DOMAIN: Final = (
    HOST_STORAGE_PRODUCER_SIGNATURE_DOMAIN_LABEL.encode("ascii") + b"\x00"
)
HOST_STORAGE_LIVE_VALIDATION_CHECKPOINTS: Final = ("pre_capability", "pre_go")

HOST_OPERATIONAL_PHASES_V3: Final = executor_v2.OPERATIONAL_PHASES
HOST_REQUEST_PHASE_PREFIX: Final = HOST_OPERATIONAL_PHASES_V3[:1]
HOST_INTENT_PHASE_PREFIX: Final = HOST_OPERATIONAL_PHASES_V3[:3]
HOST_READY_PHASE_PREFIX: Final = HOST_OPERATIONAL_PHASES_V3[:9]
HOST_ANCHOR_PHASE_PREFIX: Final = HOST_OPERATIONAL_PHASES_V3[:10]
HOST_GO_PHASE_PREFIX: Final = HOST_OPERATIONAL_PHASES_V3[:11]
HOST_GO_PHASE_ORDINAL: Final = 10
HOST_WRITE_SEAL_PHASE_ORDINAL: Final = 17
HOST_STORAGE_RECEIPT_PHASE_ORDINAL: Final = 18

RESOURCE_FIELDS: Final = executor_v2.RESOURCE_FIELDS
RESOURCE_FIELD_ORDER_SHA256: Final = (
    "8048ec1a1402b45d8bb4c67684ee7216b242bfb6d3ed9e196c0cfb262c3b93cc"
)
CANDIDATE_ORDER_SHA256: Final = "d93aaf66053aaf9a7b1c6d268a47740078dd2c1007f7287bd80908707e40b858"
MATCHED_V3_HORIZON: Final = plan_v3.MATCHED_V3_HORIZON

AUDITED_HOST_PROVISIONING_V3_DESCRIPTOR_FILE_SHA256: Final = (
    "1ff3b76662504333749529926120c0f9a49dfd7aa010f5fc5951282feed4cf56"
)
AUDITED_HOST_PROVISIONING_V3_DESCRIPTOR_BODY_SHA256: Final = (
    "0e75dc103dc9b5b4f6d50b35e0832a11396a5f18b839deb05604548b1aacc54a"
)
AUDITED_HOST_PROVISIONING_V3_SOURCE_SHA256: Final = (
    "9a5eb7dede9dc8a48f3a130ae7451a24bed3aedb4d2a56de4fa19427eef42c80"
)
AUDITED_HOST_EXECUTOR_V2_DESCRIPTOR_FILE_SHA256: Final = (
    "24c205cc4e3d189b4580512c281a84764d81dce51f43e5d959b8650a516343a4"
)
AUDITED_HOST_EXECUTOR_V2_DESCRIPTOR_BODY_SHA256: Final = (
    "e3aa649722e306a4b869db18854a9bc79508cf76f4693468e4abbd3347e5007f"
)
AUDITED_HOST_EXECUTOR_V2_SOURCE_SHA256: Final = (
    "deddb6a2386b9ece4eef3e70fa3b805d0cbdaa01ff142b7c77b0ae0606a2a96e"
)

# Zero is retained only as a local invalid-identity sentinel.  This source-only
# protocol deliberately carries no self/source/storage repository pin literals:
# the later typed provider bundle owns that complete repository closure.
_ZERO_SHA256: Final = "0" * 64

SOURCE_ONLY_CAPABILITIES: Final[Mapping[str, bool]] = MappingProxyType(
    {
        "container_control": False,
        "cryptographic_signature_verification": False,
        "filesystem_access": False,
        "host_inspection": False,
        "network_access": False,
        "process_control": False,
        "storage_measurement": False,
        "write_seal_execution": False,
    }
)
SOURCE_ONLY_READINESS: Final[Mapping[str, bool]] = MappingProxyType(
    {
        "descriptor_pinned": False,
        "plan_descriptor_pinned": False,
        "production_ready": False,
        "provider_schema_closure_complete": False,
        "storage_descriptor_pinned": False,
    }
)
SOURCE_ONLY_AUTHORITY: Final[Mapping[str, bool]] = MappingProxyType(
    {
        "evaluation_authority_granted": False,
        "execution_authority_granted": False,
        "issuance_authority_granted": False,
        "promotion_authority_granted": False,
        "qualification_authority_granted": False,
    }
)
SOURCE_ONLY_CLAIMS: Final[Mapping[str, bool]] = MappingProxyType(
    {
        "cryptographic_signature_verified_by_parser": False,
        "host_qualified": False,
        "performance_claim_allowed": False,
        "resource_matched": False,
        "scientific_evidence_created": False,
    }
)

_DESCRIPTOR_BODY_FIELD: Final = "host_qualification_protocol_descriptor_v3_body_sha256"
_PRODUCER_POLICY_BODY_FIELD: Final = "host_storage_producer_trust_policy_body_sha256"
_PRODUCER_STATEMENT_BODY_FIELD: Final = "host_storage_producer_inventory_statement_body_sha256"
_PRODUCER_VERIFICATION_BODY_FIELD: Final = (
    "host_storage_producer_inventory_signature_verification_receipt_body_sha256"
)
_PRODUCER_LIVE_BODY_FIELD: Final = "host_storage_producer_live_validation_receipt_body_sha256"
_PREFIX_BODY_FIELD: Final = "host_provisioning_validated_pre_go_prefix_body_sha256"
_REQUEST_BODY_FIELD: Final = "host_qualification_case_request_v3_body_sha256"
_INTENT_BODY_FIELD: Final = "host_qualification_case_intent_v3_body_sha256"
_READY_BODY_FIELD: Final = "host_ready_v3_body_sha256"
_ANCHOR_BODY_FIELD: Final = "host_observer_anchor_v3_body_sha256"
_GO_BODY_FIELD: Final = "host_go_v3_body_sha256"

_MAX_INTEGER: Final = (1 << 63) - 1
_MAX_TEXT_LENGTH: Final = 16_384
_IDENTIFIER_RE: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}\Z")
_IMAGE_RE: Final = re.compile(r"sha256:[0-9a-f]{64}\Z")
_CONTAINER_ID_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_SIGNATURE_RE: Final = re.compile(r"[0-9a-f]{128}\Z")


class ForagerMatchedV3HostQualificationProtocolV3Error(ValueError):
    """One host-qualification protocol-v3 artifact failed closed."""


def _fail(message: str) -> Never:
    raise ForagerMatchedV3HostQualificationProtocolV3Error(message)


class _RuntimeSealed:
    """Permit direct records while rejecting runtime subclass substitution."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if any(
            base is not _RuntimeSealed and issubclass(base, _RuntimeSealed)
            for base in cls.__bases__
        ):
            raise TypeError("host protocol frozen records cannot be subclassed")


class _BodyArtifact(Protocol):
    def to_body_dict(self) -> dict[str, Any]: ...


def _require_bool(value: object, label: str, *, expected: bool | None = None) -> bool:
    if type(value) is not bool:
        _fail(f"{label} must be one exact bool")
    exact = value
    if expected is not None and exact is not expected:
        _fail(f"{label} must be exactly {expected!r}")
    return exact


def _require_int(
    value: object,
    label: str,
    *,
    minimum: int = 0,
    maximum: int = _MAX_INTEGER,
) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        _fail(f"{label} must be one bounded exact integer")
    return value


def _require_text(value: object, label: str, *, maximum: int = _MAX_TEXT_LENGTH) -> str:
    if (
        type(value) is not str
        or not value
        or len(value) > maximum
        or any(ord(character) < 0x20 or ord(character) > 0x7E for character in value)
    ):
        _fail(f"{label} must be bounded non-empty printable ASCII text")
    return value


def _require_identifier(value: object, label: str) -> str:
    exact = _require_text(value, label, maximum=256)
    if _IDENTIFIER_RE.fullmatch(exact) is None:
        _fail(f"{label} must be one portable identifier")
    return exact


def _require_sha256(value: object, label: str, *, permit_zero: bool = False) -> str:
    try:
        return require_nonzero_sha256(value, label)
    except CanonicalEvidenceError as exc:
        if permit_zero and value == _ZERO_SHA256:
            return _ZERO_SHA256
        raise ForagerMatchedV3HostQualificationProtocolV3Error(str(exc)) from exc


def _require_image_id(value: object, label: str) -> str:
    if (
        type(value) is not str
        or _IMAGE_RE.fullmatch(value) is None
        or value == "sha256:" + _ZERO_SHA256
    ):
        _fail(f"{label} must be one nonzero canonical image ID")
    return value


def _require_container_id(value: object, label: str) -> str:
    if type(value) is not str or _CONTAINER_ID_RE.fullmatch(value) is None or value == _ZERO_SHA256:
        _fail(f"{label} must be one nonzero 64-hex container ID")
    return value


def _require_signature(value: object, label: str) -> str:
    if type(value) is not str or _SIGNATURE_RE.fullmatch(value) is None or value == "0" * 128:
        _fail(f"{label} must be one nonzero 64-byte lowercase-hex signature")
    return value


def _require_exact_type(value: object, expected: type[Any], label: str) -> None:
    if type(value) is not expected:
        _fail(f"{label} must use exact type {expected.__name__}")
    post_init = getattr(value, "__post_init__", None)
    if callable(post_init):
        try:
            post_init()
        except ForagerMatchedV3HostQualificationProtocolV3Error:
            raise
        except (
            CanonicalEvidenceError,
            provisioning_v3.ForagerMatchedV3HostProvisioningV3Error,
            executor_v2.ForagerMatchedV3HostQualificationExecutorV2Error,
        ) as exc:
            raise ForagerMatchedV3HostQualificationProtocolV3Error(str(exc)) from exc


def _revalidate_artifact_ref(value: object, label: str) -> ArtifactRefV1:
    if type(value) is not ArtifactRefV1:
        _fail(f"{label} must use exact type ArtifactRefV1")
    exact = value
    try:
        return ArtifactRefV1(
            schema_version=exact.schema_version,
            file_sha256=exact.file_sha256,
            body_sha256=exact.body_sha256,
        )
    except CanonicalEvidenceError as exc:
        raise ForagerMatchedV3HostQualificationProtocolV3Error(str(exc)) from exc


def _require_artifact(value: object, schema: str, label: str) -> ArtifactRefV1:
    exact = _revalidate_artifact_ref(value, label)
    if exact.schema_version != schema:
        _fail(f"{label} identity or schema differs")
    return exact


def _require_phase_prefix(value: object, expected: tuple[str, ...], label: str) -> None:
    if (
        type(value) is not tuple
        or any(type(item) is not str for item in cast(tuple[object, ...], value))
        or value != expected
    ):
        _fail(f"{label} differs from the exact host phase prefix")


def _require_strict_times(values: Sequence[object], label: str) -> tuple[int, ...]:
    if type(values) not in {tuple, list} or not values:
        _fail(f"{label} must be one exact nonempty sequence")
    exact = tuple(_require_int(value, f"{label} time", minimum=1) for value in values)
    if any(later <= earlier for earlier, later in zip(exact, exact[1:], strict=False)):
        _fail(f"{label} must be strictly increasing")
    return exact


def _require_distinct_artifacts(values: Sequence[ArtifactRefV1], label: str) -> None:
    exact = tuple(
        _revalidate_artifact_ref(value, f"{label} item {index}")
        for index, value in enumerate(values)
    )
    if len(set(exact)) != len(exact):
        _fail(f"{label} contains an exact duplicate artifact identity")
    try:
        require_distinct_sha256s(
            tuple(item.file_sha256 for item in exact) + tuple(item.body_sha256 for item in exact),
            label,
        )
    except CanonicalEvidenceError as exc:
        raise ForagerMatchedV3HostQualificationProtocolV3Error(str(exc)) from exc


def _authority_dict() -> dict[str, bool]:
    return dict(SOURCE_ONLY_AUTHORITY)


def _claims_dict() -> dict[str, bool]:
    return dict(SOURCE_ONLY_CLAIMS)


def _safety_posture_dict() -> dict[str, dict[str, bool]]:
    return {
        "capabilities": dict(SOURCE_ONLY_CAPABILITIES),
        "readiness": dict(SOURCE_ONLY_READINESS),
        "authority": dict(SOURCE_ONLY_AUTHORITY),
        "claims": dict(SOURCE_ONLY_CLAIMS),
    }


def _exact_dict(value: object, keys: frozenset[str], label: str) -> dict[str, Any]:
    if type(value) is not dict or frozenset(value) != keys:
        _fail(f"{label} keys differ")
    return dict(cast(dict[str, Any], value))


def _artifact_from_json(value: object, schema: str, label: str) -> ArtifactRefV1:
    try:
        result = artifact_ref_v1_from_dict(value)
    except CanonicalEvidenceError as exc:
        raise ForagerMatchedV3HostQualificationProtocolV3Error(str(exc)) from exc
    return _require_artifact(result, schema, label)


def _subject_from_json(value: object) -> CaseSubjectV1:
    try:
        return case_subject_v1_from_dict(value)
    except CanonicalEvidenceError as exc:
        raise ForagerMatchedV3HostQualificationProtocolV3Error(str(exc)) from exc


def _producers_from_json(value: object) -> tuple[ProducerRefV1, ...]:
    try:
        return producer_refs_v1_from_json(value)
    except CanonicalEvidenceError as exc:
        raise ForagerMatchedV3HostQualificationProtocolV3Error(str(exc)) from exc


def _validate_producers(value: object) -> tuple[ProducerRefV1, ...]:
    if type(value) is not tuple:
        _fail("producers must be one exact ordered six-role tuple")
    for index, producer in enumerate(cast(tuple[object, ...], value)):
        if type(producer) is not ProducerRefV1:
            _fail(f"producer {index} must use exact type ProducerRefV1")
        exact = producer
        try:
            ProducerRefV1(
                role=exact.role,
                descriptor_schema_version=exact.descriptor_schema_version,
                descriptor_file_sha256=exact.descriptor_file_sha256,
                descriptor_body_sha256=exact.descriptor_body_sha256,
                source_sha256=exact.source_sha256,
            )
        except CanonicalEvidenceError as exc:
            raise ForagerMatchedV3HostQualificationProtocolV3Error(str(exc)) from exc
    try:
        return validate_producer_refs_v1(value)
    except CanonicalEvidenceError as exc:
        raise ForagerMatchedV3HostQualificationProtocolV3Error(str(exc)) from exc


def _canonical_body_bytes[ArtifactT: _BodyArtifact](
    value: object,
    expected: type[ArtifactT],
) -> bytes:
    _require_exact_type(value, expected, "canonical BODY artifact")
    try:
        return canonical_json_bytes(cast(_BodyArtifact, value).to_body_dict(), final_lf=False)
    except CanonicalEvidenceError as exc:
        raise ForagerMatchedV3HostQualificationProtocolV3Error(str(exc)) from exc


def _canonical_file_bytes[ArtifactT: _BodyArtifact](
    value: object,
    expected: type[ArtifactT],
    body_field: str,
) -> bytes:
    _require_exact_type(value, expected, "canonical FILE artifact")
    try:
        return canonical_file_bytes(
            cast(_BodyArtifact, value).to_body_dict(),
            body_digest_field=body_field,
        )
    except CanonicalEvidenceError as exc:
        raise ForagerMatchedV3HostQualificationProtocolV3Error(str(exc)) from exc


def _parse_body(
    raw: bytes,
    *,
    expected_file_sha256: object,
    expected_body_sha256: object,
    body_field: str,
    body_keys: frozenset[str],
    label: str,
) -> dict[str, Any]:
    try:
        body = validate_canonical_file(
            raw,
            expected_file_sha256=expected_file_sha256,
            expected_body_sha256=expected_body_sha256,
            body_digest_field=body_field,
        )
    except CanonicalEvidenceError as exc:
        raise ForagerMatchedV3HostQualificationProtocolV3Error(str(exc)) from exc
    return _exact_dict(body, body_keys, label)


def _identity(
    schema_version: str,
    body_bytes: bytes,
    file_bytes: bytes,
) -> ArtifactRefV1:
    return ArtifactRefV1(
        schema_version=schema_version,
        file_sha256=hashlib.sha256(file_bytes).hexdigest(),
        body_sha256=hashlib.sha256(body_bytes).hexdigest(),
    )


def _ref_from_provisioning(identity: provisioning_v3.ArtifactIdentityV1) -> ArtifactRefV1:
    _require_exact_type(identity, provisioning_v3.ArtifactIdentityV1, "provisioning identity")
    return ArtifactRefV1(identity.schema_version, identity.file_sha256, identity.body_sha256)


def _ref_from_executor_identity(identity: executor_v2.ArtifactIdentityV2) -> ArtifactRefV1:
    _require_exact_type(identity, executor_v2.ArtifactIdentityV2, "executor identity")
    return ArtifactRefV1(identity.schema_version, identity.file_sha256, identity.body_sha256)


def _executor_artifact_ref(
    value: object,
    expected: type[Any],
    schema: str,
    body_builder: Callable[[Any], bytes],
    file_builder: Callable[[Any], bytes],
) -> ArtifactRefV1:
    if type(value) is not expected:
        _fail(f"executor artifact must use exact type {expected.__name__}")
    return _identity(schema, body_builder(value), file_builder(value))


def _inventory_sha256(facts: tuple[HostStorageProducerRuntimeFactV1, ...]) -> str:
    return hashlib.sha256(
        canonical_json_bytes([fact.to_dict() for fact in facts], final_lf=False)
    ).hexdigest()


@final
@dataclass(frozen=True, slots=True)
class PinnedHostComponentRefV1(_RuntimeSealed):
    """One exact host verifier, validator, or observer component."""

    component_id: str
    descriptor_schema_version: str
    descriptor_file_sha256: str
    descriptor_body_sha256: str
    source_sha256: str
    runtime_artifact_sha256: str

    def __post_init__(self) -> None:
        _require_identifier(self.component_id, "host component ID")
        _require_identifier(self.descriptor_schema_version, "host component descriptor schema")
        try:
            require_distinct_sha256s(
                (
                    self.descriptor_file_sha256,
                    self.descriptor_body_sha256,
                    self.source_sha256,
                    self.runtime_artifact_sha256,
                ),
                "host component descriptor/source/runtime identities",
            )
        except CanonicalEvidenceError as exc:
            raise ForagerMatchedV3HostQualificationProtocolV3Error(str(exc)) from exc

    def to_dict(self) -> dict[str, str]:
        self.__post_init__()
        return {
            "component_id": self.component_id,
            "descriptor_body_sha256": self.descriptor_body_sha256,
            "descriptor_file_sha256": self.descriptor_file_sha256,
            "descriptor_schema_version": self.descriptor_schema_version,
            "runtime_artifact_sha256": self.runtime_artifact_sha256,
            "source_sha256": self.source_sha256,
        }

    def matches_legacy(self, value: provisioning_v3.PinnedComponentIdentityV1) -> bool:
        self.__post_init__()
        _require_exact_type(
            value,
            provisioning_v3.PinnedComponentIdentityV1,
            "legacy host component",
        )
        return (
            self.component_id == value.component_id
            and self.descriptor_schema_version == value.descriptor_schema_version
            and self.descriptor_file_sha256 == value.descriptor_file_sha256
            and self.source_sha256 == value.source_sha256
            and self.runtime_artifact_sha256 == value.runtime_artifact_sha256
        )


def _pinned_host_component_from_json(value: object) -> PinnedHostComponentRefV1:
    item = _exact_dict(
        value,
        frozenset(
            {
                "component_id",
                "descriptor_body_sha256",
                "descriptor_file_sha256",
                "descriptor_schema_version",
                "runtime_artifact_sha256",
                "source_sha256",
            }
        ),
        "pinned host component",
    )
    return PinnedHostComponentRefV1(**item)


@final
@dataclass(frozen=True, slots=True)
class HostStorageProducerRuntimeFactV1(_RuntimeSealed):
    """One signed storage producer plus its live runtime artifact identity."""

    producer: ProducerRefV1
    component_id: str
    runtime_artifact_sha256: str

    def __post_init__(self) -> None:
        _require_exact_type(self.producer, ProducerRefV1, "storage producer")
        _require_identifier(self.component_id, "storage producer component ID")
        _require_sha256(self.runtime_artifact_sha256, "storage producer runtime artifact")
        if self.runtime_artifact_sha256 in {
            self.producer.descriptor_file_sha256,
            self.producer.descriptor_body_sha256,
            self.producer.source_sha256,
        }:
            _fail("storage producer runtime artifact aliases descriptor or source")

    def to_dict(self) -> dict[str, Any]:
        self.__post_init__()
        return {
            "component_id": self.component_id,
            "producer": self.producer.to_dict(),
            "runtime_artifact_sha256": self.runtime_artifact_sha256,
        }


def _producer_fact_from_json(value: object) -> HostStorageProducerRuntimeFactV1:
    item = _exact_dict(
        value,
        frozenset({"component_id", "producer", "runtime_artifact_sha256"}),
        "storage producer runtime fact",
    )
    try:
        exact_producer = producer_ref_v1_from_dict(item.pop("producer"))
    except CanonicalEvidenceError as exc:
        raise ForagerMatchedV3HostQualificationProtocolV3Error(str(exc)) from exc
    return HostStorageProducerRuntimeFactV1(producer=exact_producer, **item)


def _validate_producer_facts(
    value: object,
) -> tuple[HostStorageProducerRuntimeFactV1, ...]:
    if type(value) is not tuple or len(value) != len(PRODUCER_ROLES):
        _fail("storage producer facts must be one exact ordered six-role tuple")
    exact = cast(tuple[object, ...], value)
    if any(type(item) is not HostStorageProducerRuntimeFactV1 for item in exact):
        _fail("storage producer fact type differs")
    facts = cast(tuple[HostStorageProducerRuntimeFactV1, ...], exact)
    for index, fact in enumerate(facts):
        _require_exact_type(
            fact,
            HostStorageProducerRuntimeFactV1,
            f"storage producer fact {index}",
        )
    producers = _validate_producers(tuple(item.producer for item in facts))
    if tuple(item.producer for item in facts) != producers:
        _fail("storage producer fact role order differs")
    try:
        require_distinct_sha256s(
            tuple(item.runtime_artifact_sha256 for item in facts)
            + tuple(item.producer.descriptor_file_sha256 for item in facts)
            + tuple(item.producer.descriptor_body_sha256 for item in facts)
            + tuple(item.producer.source_sha256 for item in facts),
            "all storage producer fact identities",
        )
    except CanonicalEvidenceError as exc:
        raise ForagerMatchedV3HostQualificationProtocolV3Error(str(exc)) from exc
    if len({item.component_id for item in facts}) != len(facts):
        _fail("storage producer component IDs alias")
    return facts


def _producer_facts_from_json(value: object) -> tuple[HostStorageProducerRuntimeFactV1, ...]:
    if type(value) is not list:
        _fail("storage producer facts must be one exact JSON list")
    return _validate_producer_facts(
        tuple(_producer_fact_from_json(item) for item in cast(list[object], value))
    )


@final
@dataclass(frozen=True, slots=True)
class HostStorageProducerTrustPolicyV1(_RuntimeSealed):
    """Preissued policy for the separate six-role inventory signature domain."""

    policy_id: str
    policy_nonce_sha256: str
    qualification_plan: ArtifactRefV1
    storage_backend_policy: ArtifactRefV1
    host_trust_policy: ArtifactRefV1
    issued_at_unix_ns: int
    valid_from_unix_ns: int
    valid_until_unix_ns: int
    signer_key_id: str
    signer_public_key_sha256: str
    independent_verifier: PinnedHostComponentRefV1
    live_validator: PinnedHostComponentRefV1
    expected_producer_facts: tuple[HostStorageProducerRuntimeFactV1, ...]

    def __post_init__(self) -> None:
        _require_identifier(self.policy_id, "storage producer policy ID")
        _require_sha256(self.policy_nonce_sha256, "storage producer policy nonce")
        _require_artifact(
            self.qualification_plan,
            QUALIFICATION_PLAN_V3_SCHEMA_VERSION,
            "storage producer policy qualification plan",
        )
        _require_artifact(
            self.storage_backend_policy,
            STORAGE_BACKEND_POLICY_V1_SCHEMA_VERSION,
            "storage producer policy storage policy",
        )
        _require_artifact(
            self.host_trust_policy,
            provisioning_v3.HOST_TRUST_POLICY_SCHEMA_VERSION,
            "storage producer policy host trust policy",
        )
        issued = _require_int(self.issued_at_unix_ns, "storage producer policy issuance", minimum=1)
        valid_from = _require_int(
            self.valid_from_unix_ns,
            "storage producer policy valid-from",
            minimum=1,
        )
        valid_until = _require_int(
            self.valid_until_unix_ns,
            "storage producer policy valid-until",
            minimum=1,
        )
        if not issued <= valid_from < valid_until:
            _fail("storage producer policy chronology differs")
        _require_identifier(self.signer_key_id, "storage producer policy signer key ID")
        _require_sha256(
            self.signer_public_key_sha256,
            "storage producer policy signer public key",
        )
        _require_exact_type(
            self.independent_verifier,
            PinnedHostComponentRefV1,
            "storage producer policy verifier",
        )
        _require_exact_type(
            self.live_validator,
            PinnedHostComponentRefV1,
            "storage producer policy live validator",
        )
        if self.independent_verifier.component_id == self.live_validator.component_id:
            _fail("storage producer verifier and live validator must be distinct")
        facts = _validate_producer_facts(self.expected_producer_facts)
        component_ids = {item.component_id for item in facts}
        if {
            self.independent_verifier.component_id,
            self.live_validator.component_id,
        } & component_ids:
            _fail("storage producer verifier or validator aliases one producer")

    @property
    def expected_producer_inventory_sha256(self) -> str:
        self.__post_init__()
        return _inventory_sha256(self.expected_producer_facts)

    @property
    def storage_producers(self) -> tuple[ProducerRefV1, ...]:
        self.__post_init__()
        return tuple(item.producer for item in self.expected_producer_facts)

    def to_body_dict(self) -> dict[str, Any]:
        self.__post_init__()
        return {
            "authority": _authority_dict(),
            "claims": _claims_dict(),
            "expected_producer_facts": [item.to_dict() for item in self.expected_producer_facts],
            "expected_producer_inventory_sha256": (self.expected_producer_inventory_sha256),
            "host_trust_policy": self.host_trust_policy.to_dict(),
            "independent_verifier": self.independent_verifier.to_dict(),
            "issued_at_unix_ns": self.issued_at_unix_ns,
            "live_validator": self.live_validator.to_dict(),
            "policy_id": self.policy_id,
            "policy_nonce_sha256": self.policy_nonce_sha256,
            "qualification_plan": self.qualification_plan.to_dict(),
            "schema_version": HOST_STORAGE_PRODUCER_TRUST_POLICY_V1_SCHEMA_VERSION,
            "signature_algorithm": HOST_STORAGE_PRODUCER_SIGNATURE_ALGORITHM,
            "signature_domain": HOST_STORAGE_PRODUCER_SIGNATURE_DOMAIN_LABEL,
            "signer_key_id": self.signer_key_id,
            "signer_public_key_sha256": self.signer_public_key_sha256,
            "status": "preissued_six_role_policy_non_authorizing",
            "storage_backend_policy": self.storage_backend_policy.to_dict(),
            "valid_from_unix_ns": self.valid_from_unix_ns,
            "valid_until_unix_ns": self.valid_until_unix_ns,
        }


@final
@dataclass(frozen=True, slots=True)
class HostStorageProducerInventoryStatementV1(_RuntimeSealed):
    """Detached signature metadata over the complete six-role runtime inventory."""

    policy: ArtifactRefV1
    qualification_plan: ArtifactRefV1
    storage_backend_policy: ArtifactRefV1
    observed_at_unix_ns: int
    observed_at_monotonic_ns: int
    signer_key_id: str
    signer_public_key_sha256: str
    producer_facts: tuple[HostStorageProducerRuntimeFactV1, ...]
    signature_hex: str

    def __post_init__(self) -> None:
        _require_artifact(
            self.policy,
            HOST_STORAGE_PRODUCER_TRUST_POLICY_V1_SCHEMA_VERSION,
            "producer statement policy",
        )
        _require_artifact(
            self.qualification_plan,
            QUALIFICATION_PLAN_V3_SCHEMA_VERSION,
            "producer statement qualification plan",
        )
        _require_artifact(
            self.storage_backend_policy,
            STORAGE_BACKEND_POLICY_V1_SCHEMA_VERSION,
            "producer statement storage policy",
        )
        _require_int(self.observed_at_unix_ns, "producer statement Unix time", minimum=1)
        _require_int(
            self.observed_at_monotonic_ns,
            "producer statement monotonic time",
            minimum=1,
        )
        _require_identifier(self.signer_key_id, "producer statement signer key ID")
        _require_sha256(
            self.signer_public_key_sha256,
            "producer statement signer public key",
        )
        _validate_producer_facts(self.producer_facts)
        _require_signature(self.signature_hex, "producer statement signature")

    @property
    def producer_inventory_sha256(self) -> str:
        self.__post_init__()
        return _inventory_sha256(self.producer_facts)

    @property
    def storage_producers(self) -> tuple[ProducerRefV1, ...]:
        self.__post_init__()
        return tuple(item.producer for item in self.producer_facts)

    def to_unsigned_dict(self) -> dict[str, Any]:
        self.__post_init__()
        return {
            "executor_held_signing_secret": False,
            "hmac_used": False,
            "observed_at_monotonic_ns": self.observed_at_monotonic_ns,
            "observed_at_unix_ns": self.observed_at_unix_ns,
            "policy": self.policy.to_dict(),
            "producer_facts": [item.to_dict() for item in self.producer_facts],
            "producer_inventory_sha256": self.producer_inventory_sha256,
            "qualification_plan": self.qualification_plan.to_dict(),
            "schema_version": (HOST_STORAGE_PRODUCER_INVENTORY_STATEMENT_V1_SCHEMA_VERSION),
            "signature_algorithm": HOST_STORAGE_PRODUCER_SIGNATURE_ALGORITHM,
            "signature_domain": HOST_STORAGE_PRODUCER_SIGNATURE_DOMAIN_LABEL,
            "signer_key_id": self.signer_key_id,
            "signer_public_key_sha256": self.signer_public_key_sha256,
            "status": "six_role_inventory_signed_metadata_non_authorizing",
            "storage_backend_policy": self.storage_backend_policy.to_dict(),
        }

    @property
    def signed_payload_sha256(self) -> str:
        self.__post_init__()
        return hashlib.sha256(
            HOST_STORAGE_PRODUCER_SIGNATURE_DOMAIN
            + canonical_json_bytes(self.to_unsigned_dict(), final_lf=False)
        ).hexdigest()

    def to_body_dict(self) -> dict[str, Any]:
        self.__post_init__()
        return {
            **self.to_unsigned_dict(),
            "authority": _authority_dict(),
            "claims": _claims_dict(),
            "signature_hex": self.signature_hex,
            "signature_verified_by_parser": False,
            "signed_payload_sha256": self.signed_payload_sha256,
        }


@final
@dataclass(frozen=True, slots=True)
class HostStorageProducerInventorySignatureVerificationReceiptV1(_RuntimeSealed):
    """Report from the policy-pinned verifier; parsing performs no Ed25519 work."""

    policy: ArtifactRefV1
    statement: ArtifactRefV1
    verifier: PinnedHostComponentRefV1
    verification_run_id_sha256: str
    verification_started_at_unix_ns: int
    verification_completed_at_unix_ns: int
    verification_started_at_monotonic_ns: int
    verification_completed_at_monotonic_ns: int
    signer_key_id: str
    signer_public_key_sha256: str
    signed_payload_sha256: str
    signature_sha256: str

    def __post_init__(self) -> None:
        _require_artifact(
            self.policy,
            HOST_STORAGE_PRODUCER_TRUST_POLICY_V1_SCHEMA_VERSION,
            "producer verification policy",
        )
        _require_artifact(
            self.statement,
            HOST_STORAGE_PRODUCER_INVENTORY_STATEMENT_V1_SCHEMA_VERSION,
            "producer verification statement",
        )
        _require_exact_type(
            self.verifier,
            PinnedHostComponentRefV1,
            "producer signature verifier",
        )
        _require_sha256(self.verification_run_id_sha256, "producer verification run ID")
        started_unix, completed_unix = _require_strict_times(
            (self.verification_started_at_unix_ns, self.verification_completed_at_unix_ns),
            "producer verification Unix chronology",
        )
        started_mono, completed_mono = _require_strict_times(
            (
                self.verification_started_at_monotonic_ns,
                self.verification_completed_at_monotonic_ns,
            ),
            "producer verification monotonic chronology",
        )
        del started_unix, completed_unix, started_mono, completed_mono
        _require_identifier(self.signer_key_id, "verified producer signer key ID")
        _require_sha256(
            self.signer_public_key_sha256,
            "verified producer signer public key",
        )
        _require_sha256(self.signed_payload_sha256, "verified producer signed payload")
        _require_sha256(self.signature_sha256, "verified producer signature")

    def to_body_dict(self) -> dict[str, Any]:
        self.__post_init__()
        return {
            "authority": _authority_dict(),
            "claims": _claims_dict(),
            "cryptographic_verification_performed_by_parser": False,
            "policy": self.policy.to_dict(),
            "schema_version": (
                HOST_STORAGE_PRODUCER_INVENTORY_SIGNATURE_VERIFICATION_RECEIPT_V1_SCHEMA_VERSION
            ),
            "signature_algorithm": HOST_STORAGE_PRODUCER_SIGNATURE_ALGORITHM,
            "signature_domain": HOST_STORAGE_PRODUCER_SIGNATURE_DOMAIN_LABEL,
            "signature_sha256": self.signature_sha256,
            "signed_payload_sha256": self.signed_payload_sha256,
            "signer_key_id": self.signer_key_id,
            "signer_public_key_sha256": self.signer_public_key_sha256,
            "statement": self.statement.to_dict(),
            "status": "independent_signature_verifier_report_non_authorizing",
            "verification_completed_at_monotonic_ns": (self.verification_completed_at_monotonic_ns),
            "verification_completed_at_unix_ns": self.verification_completed_at_unix_ns,
            "verification_method": "independently_pinned_ed25519_verifier",
            "verification_result": "verifier_reports_signature_valid",
            "verification_run_id_sha256": self.verification_run_id_sha256,
            "verification_started_at_monotonic_ns": (self.verification_started_at_monotonic_ns),
            "verification_started_at_unix_ns": self.verification_started_at_unix_ns,
            "verifier": self.verifier.to_dict(),
        }


@final
@dataclass(frozen=True, slots=True)
class HostStorageProducerLiveValidationReceiptV1(_RuntimeSealed):
    """Pinned validator report for one of the two exact six-role checkpoints."""

    checkpoint: Literal["pre_capability", "pre_go"]
    checkpoint_ordinal: int
    policy: ArtifactRefV1
    statement: ArtifactRefV1
    signature_verification_receipt: ArtifactRefV1
    previous_live_validation_receipt: ArtifactRefV1 | None
    validator: PinnedHostComponentRefV1
    validation_run_id_sha256: str
    validated_at_unix_ns: int
    validated_at_monotonic_ns: int
    observed_producer_facts: tuple[HostStorageProducerRuntimeFactV1, ...]

    def __post_init__(self) -> None:
        if (
            type(self.checkpoint) is not str
            or self.checkpoint not in HOST_STORAGE_LIVE_VALIDATION_CHECKPOINTS
        ):
            _fail("producer live checkpoint differs")
        ordinal = _require_int(
            self.checkpoint_ordinal,
            "producer live checkpoint ordinal",
            maximum=1,
        )
        if HOST_STORAGE_LIVE_VALIDATION_CHECKPOINTS[ordinal] != self.checkpoint:
            _fail("producer live checkpoint and ordinal differ")
        _require_artifact(
            self.policy,
            HOST_STORAGE_PRODUCER_TRUST_POLICY_V1_SCHEMA_VERSION,
            "producer live policy",
        )
        _require_artifact(
            self.statement,
            HOST_STORAGE_PRODUCER_INVENTORY_STATEMENT_V1_SCHEMA_VERSION,
            "producer live statement",
        )
        _require_artifact(
            self.signature_verification_receipt,
            HOST_STORAGE_PRODUCER_INVENTORY_SIGNATURE_VERIFICATION_RECEIPT_V1_SCHEMA_VERSION,
            "producer live verification receipt",
        )
        if ordinal == 0:
            if self.previous_live_validation_receipt is not None:
                _fail("producer pre-capability live receipt cannot have a predecessor")
        else:
            _require_artifact(
                self.previous_live_validation_receipt,
                HOST_STORAGE_PRODUCER_LIVE_VALIDATION_RECEIPT_V1_SCHEMA_VERSION,
                "producer previous live receipt",
            )
        _require_exact_type(
            self.validator,
            PinnedHostComponentRefV1,
            "producer live validator",
        )
        _require_sha256(self.validation_run_id_sha256, "producer validation run ID")
        _require_int(self.validated_at_unix_ns, "producer validation Unix time", minimum=1)
        _require_int(
            self.validated_at_monotonic_ns,
            "producer validation monotonic time",
            minimum=1,
        )
        _validate_producer_facts(self.observed_producer_facts)

    @property
    def observed_producer_inventory_sha256(self) -> str:
        self.__post_init__()
        return _inventory_sha256(self.observed_producer_facts)

    def to_body_dict(self) -> dict[str, Any]:
        self.__post_init__()
        return {
            "authority": _authority_dict(),
            "checkpoint": self.checkpoint,
            "checkpoint_ordinal": self.checkpoint_ordinal,
            "claims": _claims_dict(),
            "live_inspection_performed_by_parser": False,
            "observed_producer_facts": [item.to_dict() for item in self.observed_producer_facts],
            "observed_producer_inventory_sha256": (self.observed_producer_inventory_sha256),
            "policy": self.policy.to_dict(),
            "previous_live_validation_receipt": (
                None
                if self.previous_live_validation_receipt is None
                else self.previous_live_validation_receipt.to_dict()
            ),
            "schema_version": (HOST_STORAGE_PRODUCER_LIVE_VALIDATION_RECEIPT_V1_SCHEMA_VERSION),
            "signature_verification_receipt": (self.signature_verification_receipt.to_dict()),
            "statement": self.statement.to_dict(),
            "status": "six_role_live_match_report_non_authorizing",
            "validated_at_monotonic_ns": self.validated_at_monotonic_ns,
            "validated_at_unix_ns": self.validated_at_unix_ns,
            "validation_result": "validator_reports_exact_six_role_inventory_match",
            "validation_run_id_sha256": self.validation_run_id_sha256,
            "validator": self.validator.to_dict(),
        }


def _validate_case_context(
    *,
    campaign_id: object,
    case_spine_sha256: object,
    subject: object,
    qualification_plan: object,
    storage_backend_policy: object,
    storage_producers: object,
    image_id: object,
    container_name: object,
    max_temporary_peak_bytes: object,
) -> tuple[CaseSubjectV1, tuple[ProducerRefV1, ...]]:
    _require_identifier(campaign_id, "campaign ID")
    _require_sha256(case_spine_sha256, "case spine")
    if type(subject) is not CaseSubjectV1:
        _fail("case subject must use exact type CaseSubjectV1")
    exact_subject = subject
    try:
        CaseSubjectV1(
            case_ordinal=exact_subject.case_ordinal,
            candidate_id=exact_subject.candidate_id,
            candidate_family=exact_subject.candidate_family,
            qualification_case_id=exact_subject.qualification_case_id,
        )
    except CanonicalEvidenceError as exc:
        raise ForagerMatchedV3HostQualificationProtocolV3Error(str(exc)) from exc
    _require_artifact(
        qualification_plan,
        QUALIFICATION_PLAN_V3_SCHEMA_VERSION,
        "qualification plan",
    )
    _require_artifact(
        storage_backend_policy,
        STORAGE_BACKEND_POLICY_V1_SCHEMA_VERSION,
        "storage backend policy",
    )
    producers = _validate_producers(storage_producers)
    _require_image_id(image_id, "case image ID")
    expected_name = executor_v2.expected_container_name_v2(case_spine_sha256)
    if _require_text(container_name, "case container name", maximum=128) != expected_name:
        _fail("container name differs from the complete case spine")
    _require_int(max_temporary_peak_bytes, "temporary hard ceiling", minimum=1)
    return exact_subject, producers


@final
@dataclass(frozen=True, slots=True)
class HostProvisioningValidatedPreGoPrefixV3(_RuntimeSealed):
    """Case-specific union of both independently validated pre-GO trust chains."""

    campaign_id: str
    case_spine_sha256: str
    subject: CaseSubjectV1
    qualification_plan: ArtifactRefV1
    storage_backend_policy: ArtifactRefV1
    request: ArtifactRefV1
    intent: ArtifactRefV1
    ready: ArtifactRefV1
    observer_anchor: ArtifactRefV1
    host_provisioning_descriptor: ArtifactRefV1
    host_provisioning_source_sha256: str
    host_trust_policy: ArtifactRefV1
    host_provisioning_statement: ArtifactRefV1
    host_signature_verification_receipt: ArtifactRefV1
    host_pre_capability_live_validation: ArtifactRefV1
    host_pre_go_live_validation: ArtifactRefV1
    storage_producer_trust_policy: ArtifactRefV1
    storage_producer_inventory_statement: ArtifactRefV1
    storage_producer_signature_verification_receipt: ArtifactRefV1
    storage_producer_pre_capability_live_validation: ArtifactRefV1
    storage_producer_pre_go_live_validation: ArtifactRefV1
    storage_producers: tuple[ProducerRefV1, ...]
    image_id: str
    container_name: str
    container_id_commitment_sha256: str
    outer_cgroup_identity_sha256: str
    max_temporary_peak_bytes: int
    aggregate_root_case_exclusive: bool
    prefix_committed_monotonic_ns: int
    committed_host_phase_prefix: tuple[str, ...] = HOST_ANCHOR_PHASE_PREFIX

    def __post_init__(self) -> None:
        _validate_case_context(
            campaign_id=self.campaign_id,
            case_spine_sha256=self.case_spine_sha256,
            subject=self.subject,
            qualification_plan=self.qualification_plan,
            storage_backend_policy=self.storage_backend_policy,
            storage_producers=self.storage_producers,
            image_id=self.image_id,
            container_name=self.container_name,
            max_temporary_peak_bytes=self.max_temporary_peak_bytes,
        )
        expected_artifacts = (
            (self.request, HOST_CASE_REQUEST_V3_SCHEMA_VERSION, "prefix request"),
            (self.intent, HOST_CASE_INTENT_V3_SCHEMA_VERSION, "prefix intent"),
            (self.ready, HOST_READY_V3_SCHEMA_VERSION, "prefix READY"),
            (
                self.observer_anchor,
                HOST_OBSERVER_ANCHOR_V3_SCHEMA_VERSION,
                "prefix observer anchor",
            ),
            (
                self.host_provisioning_descriptor,
                provisioning_v3.HOST_PROVISIONING_TRUST_CONTRACT_DESCRIPTOR_SCHEMA_VERSION,
                "prefix host provisioning descriptor",
            ),
            (
                self.host_trust_policy,
                provisioning_v3.HOST_TRUST_POLICY_SCHEMA_VERSION,
                "prefix host trust policy",
            ),
            (
                self.host_provisioning_statement,
                provisioning_v3.HOST_PROVISIONING_STATEMENT_SCHEMA_VERSION,
                "prefix host provisioning statement",
            ),
            (
                self.host_signature_verification_receipt,
                provisioning_v3.HOST_SIGNATURE_VERIFICATION_RECEIPT_SCHEMA_VERSION,
                "prefix host signature verification",
            ),
            (
                self.host_pre_capability_live_validation,
                provisioning_v3.HOST_LIVE_VALIDATION_RECEIPT_SCHEMA_VERSION,
                "prefix host pre-capability live validation",
            ),
            (
                self.host_pre_go_live_validation,
                provisioning_v3.HOST_LIVE_VALIDATION_RECEIPT_SCHEMA_VERSION,
                "prefix host pre-GO live validation",
            ),
            (
                self.storage_producer_trust_policy,
                HOST_STORAGE_PRODUCER_TRUST_POLICY_V1_SCHEMA_VERSION,
                "prefix storage producer policy",
            ),
            (
                self.storage_producer_inventory_statement,
                HOST_STORAGE_PRODUCER_INVENTORY_STATEMENT_V1_SCHEMA_VERSION,
                "prefix storage producer statement",
            ),
            (
                self.storage_producer_signature_verification_receipt,
                HOST_STORAGE_PRODUCER_INVENTORY_SIGNATURE_VERIFICATION_RECEIPT_V1_SCHEMA_VERSION,
                "prefix storage producer verification",
            ),
            (
                self.storage_producer_pre_capability_live_validation,
                HOST_STORAGE_PRODUCER_LIVE_VALIDATION_RECEIPT_V1_SCHEMA_VERSION,
                "prefix producer pre-capability live validation",
            ),
            (
                self.storage_producer_pre_go_live_validation,
                HOST_STORAGE_PRODUCER_LIVE_VALIDATION_RECEIPT_V1_SCHEMA_VERSION,
                "prefix producer pre-GO live validation",
            ),
        )
        for artifact, schema, label in expected_artifacts:
            _require_artifact(artifact, schema, label)
        _require_distinct_artifacts(
            (
                self.qualification_plan,
                self.storage_backend_policy,
                *(artifact for artifact, _, _ in expected_artifacts),
            ),
            "prefix first-class artifact identities",
        )
        if self.host_provisioning_descriptor != ArtifactRefV1(
            provisioning_v3.HOST_PROVISIONING_TRUST_CONTRACT_DESCRIPTOR_SCHEMA_VERSION,
            AUDITED_HOST_PROVISIONING_V3_DESCRIPTOR_FILE_SHA256,
            AUDITED_HOST_PROVISIONING_V3_DESCRIPTOR_BODY_SHA256,
        ):
            _fail("prefix host provisioning descriptor differs from its final audit")
        if (
            _require_sha256(
                self.host_provisioning_source_sha256,
                "prefix host provisioning source",
            )
            != AUDITED_HOST_PROVISIONING_V3_SOURCE_SHA256
        ):
            _fail("prefix host provisioning source differs from its final audit")
        _require_sha256(
            self.container_id_commitment_sha256,
            "prefix container identity commitment",
        )
        _require_sha256(self.outer_cgroup_identity_sha256, "prefix outer-cgroup identity")
        if self.container_id_commitment_sha256 == self.outer_cgroup_identity_sha256:
            _fail("prefix container and outer-cgroup identities alias")
        _require_bool(
            self.aggregate_root_case_exclusive,
            "prefix aggregate-root case exclusivity",
            expected=True,
        )
        _require_int(self.prefix_committed_monotonic_ns, "prefix commitment time", minimum=1)
        _require_phase_prefix(
            self.committed_host_phase_prefix,
            HOST_ANCHOR_PHASE_PREFIX,
            "prefix committed phases",
        )

    def to_body_dict(self) -> dict[str, Any]:
        self.__post_init__()
        return {
            "aggregate_root_case_exclusive": self.aggregate_root_case_exclusive,
            "authority": _authority_dict(),
            "campaign_id": self.campaign_id,
            "case_spine_sha256": self.case_spine_sha256,
            "claims": _claims_dict(),
            "committed_host_phase_prefix": list(self.committed_host_phase_prefix),
            "container_id_commitment_sha256": self.container_id_commitment_sha256,
            "container_name": self.container_name,
            "host_pre_capability_live_validation": (
                self.host_pre_capability_live_validation.to_dict()
            ),
            "host_pre_go_live_validation": self.host_pre_go_live_validation.to_dict(),
            "host_provisioning_descriptor": self.host_provisioning_descriptor.to_dict(),
            "host_provisioning_source_sha256": self.host_provisioning_source_sha256,
            "host_provisioning_statement": self.host_provisioning_statement.to_dict(),
            "host_signature_verification_receipt": (
                self.host_signature_verification_receipt.to_dict()
            ),
            "host_trust_policy": self.host_trust_policy.to_dict(),
            "image_id": self.image_id,
            "intent": self.intent.to_dict(),
            "max_temporary_peak_bytes": self.max_temporary_peak_bytes,
            "observer_anchor": self.observer_anchor.to_dict(),
            "outer_cgroup_identity_sha256": self.outer_cgroup_identity_sha256,
            "prefix_committed_monotonic_ns": self.prefix_committed_monotonic_ns,
            "qualification_plan": self.qualification_plan.to_dict(),
            "ready": self.ready.to_dict(),
            "request": self.request.to_dict(),
            "schema_version": HOST_PROVISIONING_VALIDATED_PRE_GO_PREFIX_V3_SCHEMA_VERSION,
            "status": "two_chain_pre_go_prefix_validated_non_authorizing",
            "storage_backend_policy": self.storage_backend_policy.to_dict(),
            "storage_producer_inventory_statement": (
                self.storage_producer_inventory_statement.to_dict()
            ),
            "storage_producer_pre_capability_live_validation": (
                self.storage_producer_pre_capability_live_validation.to_dict()
            ),
            "storage_producer_pre_go_live_validation": (
                self.storage_producer_pre_go_live_validation.to_dict()
            ),
            "storage_producer_signature_verification_receipt": (
                self.storage_producer_signature_verification_receipt.to_dict()
            ),
            "storage_producer_trust_policy": self.storage_producer_trust_policy.to_dict(),
            "storage_producers": [item.to_dict() for item in self.storage_producers],
            "subject": self.subject.to_dict(),
        }


def _validate_declared_ceilings(value: object) -> tuple[tuple[str, int], ...]:
    if type(value) is not tuple or len(value) != len(RESOURCE_FIELDS):
        _fail("declared ceilings must be one exact 28-field tuple")
    exact = cast(tuple[object, ...], value)
    pairs: list[tuple[str, int]] = []
    for index, item in enumerate(exact):
        if type(item) is not tuple or len(item) != 2:
            _fail("declared ceiling entry must be one exact pair")
        name, ceiling = item
        if type(name) is not str or name != RESOURCE_FIELDS[index]:
            _fail("declared ceiling field order differs")
        pairs.append((name, _require_int(ceiling, f"declared ceiling {name}")))
    return tuple(pairs)


def _declared_ceilings_from_json(value: object) -> tuple[tuple[str, int], ...]:
    if type(value) is not list:
        _fail("declared ceilings must be one exact JSON list")
    pairs: list[tuple[str, int]] = []
    for item in cast(list[object], value):
        record = _exact_dict(
            item,
            frozenset({"declared_ceiling", "field_name"}),
            "declared ceiling",
        )
        pairs.append((record["field_name"], record["declared_ceiling"]))
    return _validate_declared_ceilings(tuple(pairs))


@final
@dataclass(frozen=True, slots=True)
class HostQualificationCaseRequestV3(_RuntimeSealed):
    """Additive phase-0 request that binds every pre-capability dependency."""

    campaign_id: str
    case_spine_sha256: str
    subject: CaseSubjectV1
    base_request_v2: ArtifactRefV1
    qualification_plan: ArtifactRefV1
    qualification_plan_descriptor: ArtifactRefV1
    case_execution_ticket: ArtifactRefV1
    runtime_qualification_receipt: ArtifactRefV1
    storage_backend_descriptor: ArtifactRefV1
    storage_backend_source_sha256: str
    storage_backend_policy: ArtifactRefV1
    host_provisioning_descriptor: ArtifactRefV1
    host_provisioning_source_sha256: str
    host_executor_v2_descriptor: ArtifactRefV1
    host_executor_v2_source_sha256: str
    host_protocol_descriptor: ArtifactRefV1
    host_protocol_source_sha256: str
    host_trust_policy: ArtifactRefV1
    host_provisioning_statement: ArtifactRefV1
    host_signature_verification_receipt: ArtifactRefV1
    host_pre_capability_live_validation: ArtifactRefV1
    storage_producer_trust_policy: ArtifactRefV1
    storage_producer_inventory_statement: ArtifactRefV1
    storage_producer_signature_verification_receipt: ArtifactRefV1
    storage_producer_pre_capability_live_validation: ArtifactRefV1
    storage_producers: tuple[ProducerRefV1, ...]
    resource_field_order_sha256: str
    candidate_order_sha256: str
    resource_requirement_body_sha256: str
    declared_ceilings: tuple[tuple[str, int], ...]
    image_id: str
    container_name: str
    max_temporary_peak_bytes: int
    request_validated_monotonic_ns: int
    committed_host_phase_prefix: tuple[str, ...] = HOST_REQUEST_PHASE_PREFIX

    def __post_init__(self) -> None:
        _validate_case_context(
            campaign_id=self.campaign_id,
            case_spine_sha256=self.case_spine_sha256,
            subject=self.subject,
            qualification_plan=self.qualification_plan,
            storage_backend_policy=self.storage_backend_policy,
            storage_producers=self.storage_producers,
            image_id=self.image_id,
            container_name=self.container_name,
            max_temporary_peak_bytes=self.max_temporary_peak_bytes,
        )
        expected = (
            (self.base_request_v2, HOST_CASE_REQUEST_V2_SCHEMA_VERSION, "base request v2"),
            (
                self.qualification_plan_descriptor,
                QUALIFICATION_PLAN_V3_DESCRIPTOR_SCHEMA_VERSION,
                "qualification plan descriptor",
            ),
            (
                self.case_execution_ticket,
                CASE_EXECUTION_TICKET_SCHEMA_VERSION,
                "case execution ticket",
            ),
            (
                self.runtime_qualification_receipt,
                RUNTIME_QUALIFICATION_RECEIPT_SCHEMA_VERSION,
                "runtime qualification receipt",
            ),
            (
                self.storage_backend_descriptor,
                STORAGE_BACKEND_CONTRACT_DESCRIPTOR_V2_SCHEMA_VERSION,
                "storage backend descriptor",
            ),
            (
                self.host_provisioning_descriptor,
                provisioning_v3.HOST_PROVISIONING_TRUST_CONTRACT_DESCRIPTOR_SCHEMA_VERSION,
                "host provisioning descriptor",
            ),
            (
                self.host_executor_v2_descriptor,
                executor_v2.HOST_EXECUTOR_DESCRIPTOR_SCHEMA_VERSION,
                "host executor v2 descriptor",
            ),
            (
                self.host_protocol_descriptor,
                HOST_QUALIFICATION_PROTOCOL_DESCRIPTOR_V3_SCHEMA_VERSION,
                "host protocol descriptor",
            ),
            (
                self.host_trust_policy,
                provisioning_v3.HOST_TRUST_POLICY_SCHEMA_VERSION,
                "host trust policy",
            ),
            (
                self.host_provisioning_statement,
                provisioning_v3.HOST_PROVISIONING_STATEMENT_SCHEMA_VERSION,
                "host provisioning statement",
            ),
            (
                self.host_signature_verification_receipt,
                provisioning_v3.HOST_SIGNATURE_VERIFICATION_RECEIPT_SCHEMA_VERSION,
                "host signature verification",
            ),
            (
                self.host_pre_capability_live_validation,
                provisioning_v3.HOST_LIVE_VALIDATION_RECEIPT_SCHEMA_VERSION,
                "host pre-capability live validation",
            ),
            (
                self.storage_producer_trust_policy,
                HOST_STORAGE_PRODUCER_TRUST_POLICY_V1_SCHEMA_VERSION,
                "storage producer trust policy",
            ),
            (
                self.storage_producer_inventory_statement,
                HOST_STORAGE_PRODUCER_INVENTORY_STATEMENT_V1_SCHEMA_VERSION,
                "storage producer inventory statement",
            ),
            (
                self.storage_producer_signature_verification_receipt,
                HOST_STORAGE_PRODUCER_INVENTORY_SIGNATURE_VERIFICATION_RECEIPT_V1_SCHEMA_VERSION,
                "storage producer signature verification",
            ),
            (
                self.storage_producer_pre_capability_live_validation,
                HOST_STORAGE_PRODUCER_LIVE_VALIDATION_RECEIPT_V1_SCHEMA_VERSION,
                "storage producer pre-capability live validation",
            ),
        )
        for artifact, schema, label in expected:
            _require_artifact(artifact, schema, label)
        _require_distinct_artifacts(
            (
                self.qualification_plan,
                self.storage_backend_policy,
                *(artifact for artifact, _, _ in expected),
            ),
            "request first-class artifact identities",
        )
        _require_sha256(self.storage_backend_source_sha256, "storage backend source")
        host_provisioning_source = _require_sha256(
            self.host_provisioning_source_sha256,
            "request host provisioning source",
        )
        if (
            self.host_provisioning_descriptor
            != ArtifactRefV1(
                provisioning_v3.HOST_PROVISIONING_TRUST_CONTRACT_DESCRIPTOR_SCHEMA_VERSION,
                AUDITED_HOST_PROVISIONING_V3_DESCRIPTOR_FILE_SHA256,
                AUDITED_HOST_PROVISIONING_V3_DESCRIPTOR_BODY_SHA256,
            )
            or host_provisioning_source != AUDITED_HOST_PROVISIONING_V3_SOURCE_SHA256
        ):
            _fail("request host provisioning dependency differs from its final audit")
        host_executor_source = _require_sha256(
            self.host_executor_v2_source_sha256,
            "request host executor-v2 source",
        )
        if (
            self.host_executor_v2_descriptor
            != ArtifactRefV1(
                executor_v2.HOST_EXECUTOR_DESCRIPTOR_SCHEMA_VERSION,
                AUDITED_HOST_EXECUTOR_V2_DESCRIPTOR_FILE_SHA256,
                AUDITED_HOST_EXECUTOR_V2_DESCRIPTOR_BODY_SHA256,
            )
            or host_executor_source != AUDITED_HOST_EXECUTOR_V2_SOURCE_SHA256
        ):
            _fail("request host executor-v2 dependency differs from its final audit")
        if self.host_protocol_descriptor != host_qualification_protocol_descriptor_v3_identity():
            _fail("request host protocol descriptor identity differs")
        _require_sha256(self.host_protocol_source_sha256, "host protocol source")
        if (
            _require_sha256(
                self.resource_field_order_sha256,
                "request resource-field order identity",
            )
            != RESOURCE_FIELD_ORDER_SHA256
        ):
            _fail("request resource-field order identity differs")
        if (
            _require_sha256(
                self.candidate_order_sha256,
                "request candidate-order identity",
            )
            != CANDIDATE_ORDER_SHA256
        ):
            _fail("request candidate-order identity differs")
        _require_sha256(
            self.resource_requirement_body_sha256,
            "request resource requirement BODY",
        )
        ceilings = _validate_declared_ceilings(self.declared_ceilings)
        if ceilings[23][1] != self.max_temporary_peak_bytes:
            _fail("request field-24 ceiling differs from storage hard ceiling")
        _require_int(
            self.request_validated_monotonic_ns,
            "request validation time",
            minimum=1,
        )
        _require_phase_prefix(
            self.committed_host_phase_prefix,
            HOST_REQUEST_PHASE_PREFIX,
            "request committed phases",
        )

    def to_body_dict(self) -> dict[str, Any]:
        self.__post_init__()
        artifacts = {
            name: cast(ArtifactRefV1, getattr(self, name)).to_dict()
            for name in (
                "base_request_v2",
                "case_execution_ticket",
                "host_executor_v2_descriptor",
                "host_pre_capability_live_validation",
                "host_protocol_descriptor",
                "host_provisioning_descriptor",
                "host_provisioning_statement",
                "host_signature_verification_receipt",
                "host_trust_policy",
                "qualification_plan",
                "qualification_plan_descriptor",
                "runtime_qualification_receipt",
                "storage_backend_descriptor",
                "storage_backend_policy",
                "storage_producer_inventory_statement",
                "storage_producer_pre_capability_live_validation",
                "storage_producer_signature_verification_receipt",
                "storage_producer_trust_policy",
            )
        }
        return {
            **artifacts,
            "authority": _authority_dict(),
            "campaign_id": self.campaign_id,
            "candidate_order_sha256": self.candidate_order_sha256,
            "case_spine_sha256": self.case_spine_sha256,
            "claims": _claims_dict(),
            "committed_host_phase_prefix": list(self.committed_host_phase_prefix),
            "container_name": self.container_name,
            "declared_ceilings": [
                {"declared_ceiling": ceiling, "field_name": name}
                for name, ceiling in self.declared_ceilings
            ],
            "host_executor_v2_source_sha256": self.host_executor_v2_source_sha256,
            "host_protocol_source_sha256": self.host_protocol_source_sha256,
            "host_provisioning_source_sha256": self.host_provisioning_source_sha256,
            "image_id": self.image_id,
            "max_temporary_peak_bytes": self.max_temporary_peak_bytes,
            "request_validated_monotonic_ns": self.request_validated_monotonic_ns,
            "resource_field_order_sha256": self.resource_field_order_sha256,
            "resource_requirement_body_sha256": self.resource_requirement_body_sha256,
            "schema_version": HOST_CASE_REQUEST_V3_SCHEMA_VERSION,
            "status": "phase0_additive_request_validated_non_authorizing",
            "storage_backend_source_sha256": self.storage_backend_source_sha256,
            "storage_producers": [item.to_dict() for item in self.storage_producers],
            "subject": self.subject.to_dict(),
        }


@final
@dataclass(frozen=True, slots=True)
class HostQualificationCaseIntentV3(_RuntimeSealed):
    """Additive phases 1--2 commitment after both pre-capability checks."""

    campaign_id: str
    case_spine_sha256: str
    subject: CaseSubjectV1
    request: ArtifactRefV1
    base_intent_v2: ArtifactRefV1
    qualification_plan: ArtifactRefV1
    case_execution_ticket: ArtifactRefV1
    storage_backend_policy: ArtifactRefV1
    storage_producer_trust_policy: ArtifactRefV1
    storage_producer_inventory_statement: ArtifactRefV1
    storage_producer_signature_verification_receipt: ArtifactRefV1
    storage_producer_pre_capability_live_validation: ArtifactRefV1
    storage_producers: tuple[ProducerRefV1, ...]
    image_id: str
    container_name: str
    max_temporary_peak_bytes: int
    authorization_validated_monotonic_ns: int
    intent_committed_monotonic_ns: int
    intent_committed: bool
    same_case_retry_permitted: bool
    committed_host_phase_prefix: tuple[str, ...] = HOST_INTENT_PHASE_PREFIX

    def __post_init__(self) -> None:
        _validate_case_context(
            campaign_id=self.campaign_id,
            case_spine_sha256=self.case_spine_sha256,
            subject=self.subject,
            qualification_plan=self.qualification_plan,
            storage_backend_policy=self.storage_backend_policy,
            storage_producers=self.storage_producers,
            image_id=self.image_id,
            container_name=self.container_name,
            max_temporary_peak_bytes=self.max_temporary_peak_bytes,
        )
        expected = (
            (self.request, HOST_CASE_REQUEST_V3_SCHEMA_VERSION, "intent request"),
            (self.base_intent_v2, HOST_CASE_INTENT_V2_SCHEMA_VERSION, "base intent v2"),
            (
                self.case_execution_ticket,
                CASE_EXECUTION_TICKET_SCHEMA_VERSION,
                "intent case ticket",
            ),
            (
                self.storage_producer_trust_policy,
                HOST_STORAGE_PRODUCER_TRUST_POLICY_V1_SCHEMA_VERSION,
                "intent producer policy",
            ),
            (
                self.storage_producer_inventory_statement,
                HOST_STORAGE_PRODUCER_INVENTORY_STATEMENT_V1_SCHEMA_VERSION,
                "intent producer statement",
            ),
            (
                self.storage_producer_signature_verification_receipt,
                HOST_STORAGE_PRODUCER_INVENTORY_SIGNATURE_VERIFICATION_RECEIPT_V1_SCHEMA_VERSION,
                "intent producer verification",
            ),
            (
                self.storage_producer_pre_capability_live_validation,
                HOST_STORAGE_PRODUCER_LIVE_VALIDATION_RECEIPT_V1_SCHEMA_VERSION,
                "intent producer pre-capability validation",
            ),
        )
        for artifact, schema, label in expected:
            _require_artifact(artifact, schema, label)
        _require_distinct_artifacts(
            (
                self.qualification_plan,
                self.storage_backend_policy,
                *(artifact for artifact, _, _ in expected),
            ),
            "intent first-class artifact identities",
        )
        _require_strict_times(
            (
                self.authorization_validated_monotonic_ns,
                self.intent_committed_monotonic_ns,
            ),
            "intent phase chronology",
        )
        _require_bool(self.intent_committed, "intent commitment", expected=True)
        _require_bool(
            self.same_case_retry_permitted,
            "intent same-case retry",
            expected=False,
        )
        _require_phase_prefix(
            self.committed_host_phase_prefix,
            HOST_INTENT_PHASE_PREFIX,
            "intent committed phases",
        )

    def to_body_dict(self) -> dict[str, Any]:
        self.__post_init__()
        return {
            "authority": _authority_dict(),
            "authorization_validated_monotonic_ns": (self.authorization_validated_monotonic_ns),
            "base_intent_v2": self.base_intent_v2.to_dict(),
            "campaign_id": self.campaign_id,
            "case_execution_ticket": self.case_execution_ticket.to_dict(),
            "case_spine_sha256": self.case_spine_sha256,
            "claims": _claims_dict(),
            "committed_host_phase_prefix": list(self.committed_host_phase_prefix),
            "container_name": self.container_name,
            "image_id": self.image_id,
            "intent_committed": self.intent_committed,
            "intent_committed_monotonic_ns": self.intent_committed_monotonic_ns,
            "max_temporary_peak_bytes": self.max_temporary_peak_bytes,
            "qualification_plan": self.qualification_plan.to_dict(),
            "request": self.request.to_dict(),
            "same_case_retry_permitted": self.same_case_retry_permitted,
            "schema_version": HOST_CASE_INTENT_V3_SCHEMA_VERSION,
            "status": "phases1_2_additive_intent_committed_non_authorizing",
            "storage_backend_policy": self.storage_backend_policy.to_dict(),
            "storage_producer_inventory_statement": (
                self.storage_producer_inventory_statement.to_dict()
            ),
            "storage_producer_pre_capability_live_validation": (
                self.storage_producer_pre_capability_live_validation.to_dict()
            ),
            "storage_producer_signature_verification_receipt": (
                self.storage_producer_signature_verification_receipt.to_dict()
            ),
            "storage_producer_trust_policy": (self.storage_producer_trust_policy.to_dict()),
            "storage_producers": [item.to_dict() for item in self.storage_producers],
            "subject": self.subject.to_dict(),
        }


@final
@dataclass(frozen=True, slots=True)
class HostReadyV3(_RuntimeSealed):
    """Additive phase-8 READY with the exact phases 3--8 projection."""

    campaign_id: str
    case_spine_sha256: str
    subject: CaseSubjectV1
    intent: ArtifactRefV1
    base_initial_cgroup_sample_v2: ArtifactRefV1
    base_ready_v2: ArtifactRefV1
    qualification_plan: ArtifactRefV1
    storage_backend_policy: ArtifactRefV1
    storage_producers: tuple[ProducerRefV1, ...]
    image_id: str
    container_id: str
    container_name: str
    container_id_commitment_sha256: str
    outer_cgroup_identity_sha256: str
    max_temporary_peak_bytes: int
    aggregate_root_case_exclusive: bool
    fresh_cgroup_created_monotonic_ns: int
    retained_counter_fds_opened_monotonic_ns: int
    initial_cgroup_sample_committed_monotonic_ns: int
    container_created_monotonic_ns: int
    container_started_monotonic_ns: int
    driver_ready_monotonic_ns: int
    candidate_code_loaded: bool
    go_committed: bool
    committed_host_phase_prefix: tuple[str, ...] = HOST_READY_PHASE_PREFIX

    def __post_init__(self) -> None:
        _validate_case_context(
            campaign_id=self.campaign_id,
            case_spine_sha256=self.case_spine_sha256,
            subject=self.subject,
            qualification_plan=self.qualification_plan,
            storage_backend_policy=self.storage_backend_policy,
            storage_producers=self.storage_producers,
            image_id=self.image_id,
            container_name=self.container_name,
            max_temporary_peak_bytes=self.max_temporary_peak_bytes,
        )
        for artifact, schema, label in (
            (self.intent, HOST_CASE_INTENT_V3_SCHEMA_VERSION, "READY intent"),
            (
                self.base_initial_cgroup_sample_v2,
                HOST_INITIAL_CGROUP_SAMPLE_V2_SCHEMA_VERSION,
                "READY base initial sample",
            ),
            (self.base_ready_v2, HOST_READY_V2_SCHEMA_VERSION, "READY base READY"),
        ):
            _require_artifact(artifact, schema, label)
        _require_distinct_artifacts(
            (
                self.qualification_plan,
                self.storage_backend_policy,
                self.intent,
                self.base_initial_cgroup_sample_v2,
                self.base_ready_v2,
            ),
            "READY first-class artifact identities",
        )
        _require_container_id(self.container_id, "READY container ID")
        _require_sha256(
            self.container_id_commitment_sha256,
            "READY container identity commitment",
        )
        expected_commitment = executor_v2.container_runtime_identity_sha256_v2(
            self.case_spine_sha256,
            self.container_name,
            self.container_id,
        )
        if not hmac.compare_digest(
            self.container_id_commitment_sha256,
            expected_commitment,
        ):
            _fail("READY container identity commitment differs")
        _require_sha256(self.outer_cgroup_identity_sha256, "READY outer-cgroup identity")
        if self.outer_cgroup_identity_sha256 == self.container_id_commitment_sha256:
            _fail("READY container and outer-cgroup identities alias")
        _require_bool(
            self.aggregate_root_case_exclusive,
            "READY aggregate-root case exclusivity",
            expected=True,
        )
        _require_strict_times(
            (
                self.fresh_cgroup_created_monotonic_ns,
                self.retained_counter_fds_opened_monotonic_ns,
                self.initial_cgroup_sample_committed_monotonic_ns,
                self.container_created_monotonic_ns,
                self.container_started_monotonic_ns,
                self.driver_ready_monotonic_ns,
            ),
            "READY phase chronology",
        )
        _require_bool(
            self.candidate_code_loaded,
            "READY candidate-code-loaded",
            expected=False,
        )
        _require_bool(self.go_committed, "READY GO commitment", expected=False)
        _require_phase_prefix(
            self.committed_host_phase_prefix,
            HOST_READY_PHASE_PREFIX,
            "READY committed phases",
        )

    def to_body_dict(self) -> dict[str, Any]:
        self.__post_init__()
        return {
            "aggregate_root_case_exclusive": self.aggregate_root_case_exclusive,
            "authority": _authority_dict(),
            "base_initial_cgroup_sample_v2": (self.base_initial_cgroup_sample_v2.to_dict()),
            "base_ready_v2": self.base_ready_v2.to_dict(),
            "campaign_id": self.campaign_id,
            "candidate_code_loaded": self.candidate_code_loaded,
            "case_spine_sha256": self.case_spine_sha256,
            "claims": _claims_dict(),
            "committed_host_phase_prefix": list(self.committed_host_phase_prefix),
            "container_created_monotonic_ns": self.container_created_monotonic_ns,
            "container_id": self.container_id,
            "container_id_commitment_sha256": self.container_id_commitment_sha256,
            "container_name": self.container_name,
            "container_started_monotonic_ns": self.container_started_monotonic_ns,
            "driver_ready_monotonic_ns": self.driver_ready_monotonic_ns,
            "fresh_cgroup_created_monotonic_ns": self.fresh_cgroup_created_monotonic_ns,
            "go_committed": self.go_committed,
            "image_id": self.image_id,
            "initial_cgroup_sample_committed_monotonic_ns": (
                self.initial_cgroup_sample_committed_monotonic_ns
            ),
            "intent": self.intent.to_dict(),
            "max_temporary_peak_bytes": self.max_temporary_peak_bytes,
            "outer_cgroup_identity_sha256": self.outer_cgroup_identity_sha256,
            "qualification_plan": self.qualification_plan.to_dict(),
            "retained_counter_fds_opened_monotonic_ns": (
                self.retained_counter_fds_opened_monotonic_ns
            ),
            "schema_version": HOST_READY_V3_SCHEMA_VERSION,
            "status": "phase8_additive_driver_ready_non_authorizing",
            "storage_backend_policy": self.storage_backend_policy.to_dict(),
            "storage_producers": [item.to_dict() for item in self.storage_producers],
            "subject": self.subject.to_dict(),
        }


@final
@dataclass(frozen=True, slots=True)
class HostObserverAnchorV3(_RuntimeSealed):
    """Additive phase-9 observer anchor bound to the pinned live component."""

    campaign_id: str
    case_spine_sha256: str
    subject: CaseSubjectV1
    ready: ArtifactRefV1
    base_observer_anchor_v2: ArtifactRefV1
    qualification_plan: ArtifactRefV1
    storage_backend_policy: ArtifactRefV1
    storage_producers: tuple[ProducerRefV1, ...]
    image_id: str
    container_name: str
    container_id_commitment_sha256: str
    outer_cgroup_identity_sha256: str
    max_temporary_peak_bytes: int
    aggregate_root_case_exclusive: bool
    membership_observer: PinnedHostComponentRefV1
    observer_anchored_monotonic_ns: int
    observation_loss_detected: bool
    committed_host_phase_prefix: tuple[str, ...] = HOST_ANCHOR_PHASE_PREFIX

    def __post_init__(self) -> None:
        _validate_case_context(
            campaign_id=self.campaign_id,
            case_spine_sha256=self.case_spine_sha256,
            subject=self.subject,
            qualification_plan=self.qualification_plan,
            storage_backend_policy=self.storage_backend_policy,
            storage_producers=self.storage_producers,
            image_id=self.image_id,
            container_name=self.container_name,
            max_temporary_peak_bytes=self.max_temporary_peak_bytes,
        )
        _require_artifact(self.ready, HOST_READY_V3_SCHEMA_VERSION, "anchor READY")
        _require_artifact(
            self.base_observer_anchor_v2,
            HOST_OBSERVER_ANCHOR_V2_SCHEMA_VERSION,
            "anchor base observer anchor",
        )
        _require_distinct_artifacts(
            (
                self.qualification_plan,
                self.storage_backend_policy,
                self.ready,
                self.base_observer_anchor_v2,
            ),
            "anchor first-class artifact identities",
        )
        _require_sha256(
            self.container_id_commitment_sha256,
            "anchor container identity commitment",
        )
        _require_sha256(self.outer_cgroup_identity_sha256, "anchor outer-cgroup identity")
        if self.outer_cgroup_identity_sha256 == self.container_id_commitment_sha256:
            _fail("anchor container and outer-cgroup identities alias")
        _require_bool(
            self.aggregate_root_case_exclusive,
            "anchor aggregate-root case exclusivity",
            expected=True,
        )
        _require_exact_type(
            self.membership_observer,
            PinnedHostComponentRefV1,
            "anchor membership observer",
        )
        _require_int(
            self.observer_anchored_monotonic_ns,
            "observer anchor time",
            minimum=1,
        )
        _require_bool(
            self.observation_loss_detected,
            "anchor observation loss",
            expected=False,
        )
        _require_phase_prefix(
            self.committed_host_phase_prefix,
            HOST_ANCHOR_PHASE_PREFIX,
            "anchor committed phases",
        )

    def to_body_dict(self) -> dict[str, Any]:
        self.__post_init__()
        return {
            "aggregate_root_case_exclusive": self.aggregate_root_case_exclusive,
            "authority": _authority_dict(),
            "base_observer_anchor_v2": self.base_observer_anchor_v2.to_dict(),
            "campaign_id": self.campaign_id,
            "case_spine_sha256": self.case_spine_sha256,
            "claims": _claims_dict(),
            "committed_host_phase_prefix": list(self.committed_host_phase_prefix),
            "container_id_commitment_sha256": self.container_id_commitment_sha256,
            "container_name": self.container_name,
            "image_id": self.image_id,
            "max_temporary_peak_bytes": self.max_temporary_peak_bytes,
            "membership_observer": self.membership_observer.to_dict(),
            "observation_loss_detected": self.observation_loss_detected,
            "observer_anchored_monotonic_ns": self.observer_anchored_monotonic_ns,
            "outer_cgroup_identity_sha256": self.outer_cgroup_identity_sha256,
            "qualification_plan": self.qualification_plan.to_dict(),
            "ready": self.ready.to_dict(),
            "schema_version": HOST_OBSERVER_ANCHOR_V3_SCHEMA_VERSION,
            "status": "phase9_additive_observer_anchored_non_authorizing",
            "storage_backend_policy": self.storage_backend_policy.to_dict(),
            "storage_producers": [item.to_dict() for item in self.storage_producers],
            "subject": self.subject.to_dict(),
        }


def host_go_payload_sha256_v3(
    *,
    campaign_id: str,
    case_spine_sha256: str,
    subject: CaseSubjectV1,
    base_go_v2: ArtifactRefV1,
    qualification_plan: ArtifactRefV1,
    storage_backend_policy: ArtifactRefV1,
    ready: ArtifactRefV1,
    observer_anchor: ArtifactRefV1,
    validated_pre_go_prefix: ArtifactRefV1,
    storage_runtime_intent: ArtifactRefV1,
    storage_producers: tuple[ProducerRefV1, ...],
    image_id: str,
    container_name: str,
    container_id_commitment_sha256: str,
    outer_cgroup_identity_sha256: str,
    max_temporary_peak_bytes: int,
    resource_field_order_sha256: str,
    candidate_order_sha256: str,
) -> str:
    """Derive the exact additive one-way GO payload identity."""

    _validate_case_context(
        campaign_id=campaign_id,
        case_spine_sha256=case_spine_sha256,
        subject=subject,
        qualification_plan=qualification_plan,
        storage_backend_policy=storage_backend_policy,
        storage_producers=storage_producers,
        image_id=image_id,
        container_name=container_name,
        max_temporary_peak_bytes=max_temporary_peak_bytes,
    )
    for artifact, schema, label in (
        (base_go_v2, HOST_GO_V2_SCHEMA_VERSION, "GO payload base GO"),
        (ready, HOST_READY_V3_SCHEMA_VERSION, "GO payload READY"),
        (
            observer_anchor,
            HOST_OBSERVER_ANCHOR_V3_SCHEMA_VERSION,
            "GO payload observer anchor",
        ),
        (
            validated_pre_go_prefix,
            HOST_PROVISIONING_VALIDATED_PRE_GO_PREFIX_V3_SCHEMA_VERSION,
            "GO payload validated prefix",
        ),
        (
            storage_runtime_intent,
            STORAGE_BOUNDARY_RUNTIME_INTENT_V1_SCHEMA_VERSION,
            "GO payload storage runtime intent",
        ),
    ):
        _require_artifact(artifact, schema, label)
    _require_sha256(container_id_commitment_sha256, "GO payload container identity")
    _require_sha256(outer_cgroup_identity_sha256, "GO payload outer-cgroup identity")
    if container_id_commitment_sha256 == outer_cgroup_identity_sha256:
        _fail("GO payload container and outer-cgroup identities alias")
    if (
        _require_sha256(
            resource_field_order_sha256,
            "GO payload resource-field order identity",
        )
        != RESOURCE_FIELD_ORDER_SHA256
    ):
        _fail("GO payload resource-field order identity differs")
    if (
        _require_sha256(
            candidate_order_sha256,
            "GO payload candidate-order identity",
        )
        != CANDIDATE_ORDER_SHA256
    ):
        _fail("GO payload candidate-order identity differs")

    payload = {
        "base_go_v2": base_go_v2.to_dict(),
        "campaign_id": campaign_id,
        "candidate_order_sha256": candidate_order_sha256,
        "case_spine_sha256": case_spine_sha256,
        "container_id_commitment_sha256": container_id_commitment_sha256,
        "container_name": container_name,
        "image_id": image_id,
        "max_temporary_peak_bytes": max_temporary_peak_bytes,
        "observer_anchor": observer_anchor.to_dict(),
        "outer_cgroup_identity_sha256": outer_cgroup_identity_sha256,
        "qualification_plan": qualification_plan.to_dict(),
        "ready": ready.to_dict(),
        "resource_field_order_sha256": resource_field_order_sha256,
        "storage_backend_policy": storage_backend_policy.to_dict(),
        "storage_producers": [item.to_dict() for item in storage_producers],
        "storage_runtime_intent": storage_runtime_intent.to_dict(),
        "subject": subject.to_dict(),
        "validated_pre_go_prefix": validated_pre_go_prefix.to_dict(),
    }
    return hashlib.sha256(canonical_json_bytes(payload, final_lf=False)).hexdigest()


@final
@dataclass(frozen=True, slots=True)
class HostGoCommitmentV3(_RuntimeSealed):
    """Additive phase-10 GO naming runtime intent and all six producers directly."""

    campaign_id: str
    case_spine_sha256: str
    subject: CaseSubjectV1
    base_go_v2: ArtifactRefV1
    qualification_plan: ArtifactRefV1
    storage_backend_policy: ArtifactRefV1
    ready: ArtifactRefV1
    observer_anchor: ArtifactRefV1
    validated_pre_go_prefix: ArtifactRefV1
    storage_runtime_intent: ArtifactRefV1
    storage_producers: tuple[ProducerRefV1, ...]
    image_id: str
    container_name: str
    container_id_commitment_sha256: str
    outer_cgroup_identity_sha256: str
    max_temporary_peak_bytes: int
    aggregate_root_case_exclusive: bool
    resource_field_order_sha256: str
    candidate_order_sha256: str
    storage_runtime_intent_committed_monotonic_ns: int
    go_committed_monotonic_ns: int
    go_payload_sha256: str
    go_commit_count: int
    one_way: bool
    same_case_retry_permitted: bool
    storage_runtime_intent_committed_before_go: bool
    exact_six_producers_committed: bool
    committed_host_phase_prefix: tuple[str, ...] = HOST_GO_PHASE_PREFIX

    def __post_init__(self) -> None:
        _validate_case_context(
            campaign_id=self.campaign_id,
            case_spine_sha256=self.case_spine_sha256,
            subject=self.subject,
            qualification_plan=self.qualification_plan,
            storage_backend_policy=self.storage_backend_policy,
            storage_producers=self.storage_producers,
            image_id=self.image_id,
            container_name=self.container_name,
            max_temporary_peak_bytes=self.max_temporary_peak_bytes,
        )
        expected = (
            (self.base_go_v2, HOST_GO_V2_SCHEMA_VERSION, "GO base GO"),
            (self.ready, HOST_READY_V3_SCHEMA_VERSION, "GO READY"),
            (
                self.observer_anchor,
                HOST_OBSERVER_ANCHOR_V3_SCHEMA_VERSION,
                "GO observer anchor",
            ),
            (
                self.validated_pre_go_prefix,
                HOST_PROVISIONING_VALIDATED_PRE_GO_PREFIX_V3_SCHEMA_VERSION,
                "GO validated pre-GO prefix",
            ),
            (
                self.storage_runtime_intent,
                STORAGE_BOUNDARY_RUNTIME_INTENT_V1_SCHEMA_VERSION,
                "GO storage runtime intent",
            ),
        )
        for artifact, schema, label in expected:
            _require_artifact(artifact, schema, label)
        _require_distinct_artifacts(
            (
                self.qualification_plan,
                self.storage_backend_policy,
                *(artifact for artifact, _, _ in expected),
            ),
            "GO first-class artifact identities",
        )
        _require_sha256(
            self.container_id_commitment_sha256,
            "GO container identity commitment",
        )
        _require_sha256(self.outer_cgroup_identity_sha256, "GO outer-cgroup identity")
        if self.outer_cgroup_identity_sha256 == self.container_id_commitment_sha256:
            _fail("GO container and outer-cgroup identities alias")
        _require_bool(
            self.aggregate_root_case_exclusive,
            "GO aggregate-root case exclusivity",
            expected=True,
        )
        if (
            _require_sha256(
                self.resource_field_order_sha256,
                "GO resource-field order identity",
            )
            != RESOURCE_FIELD_ORDER_SHA256
        ):
            _fail("GO resource-field order identity differs")
        if (
            _require_sha256(
                self.candidate_order_sha256,
                "GO candidate-order identity",
            )
            != CANDIDATE_ORDER_SHA256
        ):
            _fail("GO candidate-order identity differs")
        _require_strict_times(
            (
                self.storage_runtime_intent_committed_monotonic_ns,
                self.go_committed_monotonic_ns,
            ),
            "GO runtime-intent chronology",
        )
        expected_payload = host_go_payload_sha256_v3(
            campaign_id=self.campaign_id,
            case_spine_sha256=self.case_spine_sha256,
            subject=self.subject,
            base_go_v2=self.base_go_v2,
            qualification_plan=self.qualification_plan,
            storage_backend_policy=self.storage_backend_policy,
            ready=self.ready,
            observer_anchor=self.observer_anchor,
            validated_pre_go_prefix=self.validated_pre_go_prefix,
            storage_runtime_intent=self.storage_runtime_intent,
            storage_producers=self.storage_producers,
            image_id=self.image_id,
            container_name=self.container_name,
            container_id_commitment_sha256=self.container_id_commitment_sha256,
            outer_cgroup_identity_sha256=self.outer_cgroup_identity_sha256,
            max_temporary_peak_bytes=self.max_temporary_peak_bytes,
            resource_field_order_sha256=self.resource_field_order_sha256,
            candidate_order_sha256=self.candidate_order_sha256,
        )
        if not hmac.compare_digest(self.go_payload_sha256, expected_payload):
            _fail("GO payload identity differs from its complete additive projection")
        if _require_int(self.go_commit_count, "GO commitment count", maximum=1) != 1:
            _fail("GO commitment count must be exactly one")
        _require_bool(self.one_way, "GO one-way commitment", expected=True)
        _require_bool(
            self.same_case_retry_permitted,
            "GO same-case retry",
            expected=False,
        )
        _require_bool(
            self.storage_runtime_intent_committed_before_go,
            "GO runtime intent predecessor",
            expected=True,
        )
        _require_bool(
            self.exact_six_producers_committed,
            "GO six-producer commitment",
            expected=True,
        )
        _require_phase_prefix(
            self.committed_host_phase_prefix,
            HOST_GO_PHASE_PREFIX,
            "GO committed phases",
        )

    def to_body_dict(self) -> dict[str, Any]:
        self.__post_init__()
        return {
            "aggregate_root_case_exclusive": self.aggregate_root_case_exclusive,
            "authority": _authority_dict(),
            "base_go_v2": self.base_go_v2.to_dict(),
            "campaign_id": self.campaign_id,
            "candidate_order_sha256": self.candidate_order_sha256,
            "case_spine_sha256": self.case_spine_sha256,
            "claims": _claims_dict(),
            "committed_host_phase_prefix": list(self.committed_host_phase_prefix),
            "container_id_commitment_sha256": self.container_id_commitment_sha256,
            "container_name": self.container_name,
            "exact_six_producers_committed": self.exact_six_producers_committed,
            "go_commit_count": self.go_commit_count,
            "go_committed_monotonic_ns": self.go_committed_monotonic_ns,
            "go_payload_sha256": self.go_payload_sha256,
            "image_id": self.image_id,
            "max_temporary_peak_bytes": self.max_temporary_peak_bytes,
            "observer_anchor": self.observer_anchor.to_dict(),
            "one_way": self.one_way,
            "outer_cgroup_identity_sha256": self.outer_cgroup_identity_sha256,
            "qualification_plan": self.qualification_plan.to_dict(),
            "ready": self.ready.to_dict(),
            "resource_field_order_sha256": self.resource_field_order_sha256,
            "same_case_retry_permitted": self.same_case_retry_permitted,
            "schema_version": HOST_GO_V3_SCHEMA_VERSION,
            "status": "phase10_additive_one_way_go_committed_non_authorizing",
            "storage_backend_policy": self.storage_backend_policy.to_dict(),
            "storage_producers": [item.to_dict() for item in self.storage_producers],
            "storage_runtime_intent": self.storage_runtime_intent.to_dict(),
            "storage_runtime_intent_committed_before_go": (
                self.storage_runtime_intent_committed_before_go
            ),
            "storage_runtime_intent_committed_monotonic_ns": (
                self.storage_runtime_intent_committed_monotonic_ns
            ),
            "subject": self.subject.to_dict(),
            "validated_pre_go_prefix": self.validated_pre_go_prefix.to_dict(),
        }


@final
@dataclass(frozen=True, slots=True)
class HostQualificationProtocolDescriptorV3(_RuntimeSealed):
    """Source-only descriptor for the eleven new public artifact schemas."""

    def to_body_dict(self) -> dict[str, Any]:
        schemas = (
            (
                HOST_QUALIFICATION_PROTOCOL_DESCRIPTOR_V3_SCHEMA_VERSION,
                _DESCRIPTOR_BODY_FIELD,
            ),
            (
                HOST_STORAGE_PRODUCER_TRUST_POLICY_V1_SCHEMA_VERSION,
                _PRODUCER_POLICY_BODY_FIELD,
            ),
            (
                HOST_STORAGE_PRODUCER_INVENTORY_STATEMENT_V1_SCHEMA_VERSION,
                _PRODUCER_STATEMENT_BODY_FIELD,
            ),
            (
                HOST_STORAGE_PRODUCER_INVENTORY_SIGNATURE_VERIFICATION_RECEIPT_V1_SCHEMA_VERSION,
                _PRODUCER_VERIFICATION_BODY_FIELD,
            ),
            (
                HOST_STORAGE_PRODUCER_LIVE_VALIDATION_RECEIPT_V1_SCHEMA_VERSION,
                _PRODUCER_LIVE_BODY_FIELD,
            ),
            (
                HOST_PROVISIONING_VALIDATED_PRE_GO_PREFIX_V3_SCHEMA_VERSION,
                _PREFIX_BODY_FIELD,
            ),
            (HOST_CASE_REQUEST_V3_SCHEMA_VERSION, _REQUEST_BODY_FIELD),
            (HOST_CASE_INTENT_V3_SCHEMA_VERSION, _INTENT_BODY_FIELD),
            (HOST_READY_V3_SCHEMA_VERSION, _READY_BODY_FIELD),
            (HOST_OBSERVER_ANCHOR_V3_SCHEMA_VERSION, _ANCHOR_BODY_FIELD),
            (HOST_GO_V3_SCHEMA_VERSION, _GO_BODY_FIELD),
        )
        return {
            "authority": _authority_dict(),
            "claims": _claims_dict(),
            "external_endpoint_schemas": {
                "irreversible_write_seal": IRREVERSIBLE_WRITE_SEAL_V1_SCHEMA_VERSION,
                "storage_boundary_receipt": STORAGE_BOUNDARY_RECEIPT_V2_SCHEMA_VERSION,
                "storage_runtime_intent": STORAGE_BOUNDARY_RUNTIME_INTENT_V1_SCHEMA_VERSION,
            },
            "frozen_host_operational_phases": list(HOST_OPERATIONAL_PHASES_V3),
            "go_committed_phase_ordinal": HOST_GO_PHASE_ORDINAL,
            "go_names_future_write_seal": False,
            "go_serializes_host_storage_binding_artifact": False,
            "go_serializes_runtime_intent_directly": True,
            "go_serializes_six_producers_directly": True,
            "matched_v3_horizon": MATCHED_V3_HORIZON,
            "operational_apis": [],
            "producer_roles": list(PRODUCER_ROLES),
            "repository_closure": {
                "delegated_to": "later_typed_provider_bundle",
                "full_descriptor_source_dependency_closure_delegated": True,
                "host_protocol_repository_pins_serialized": False,
                "storage_repository_pins_serialized": False,
            },
            "public_schemas": [
                {"body_sha256_field": body_field, "schema_version": schema}
                for schema, body_field in schemas
            ],
            "resource_field_order_sha256": RESOURCE_FIELD_ORDER_SHA256,
            "candidate_order_sha256": CANDIDATE_ORDER_SHA256,
            "safety_posture": _safety_posture_dict(),
            "schema_version": HOST_QUALIFICATION_PROTOCOL_DESCRIPTOR_V3_SCHEMA_VERSION,
            "status": "source_only_delegated_closure_non_authorizing_protocol_descriptor",
            "storage_boundary_receipt_phase_ordinal": HOST_STORAGE_RECEIPT_PHASE_ORDINAL,
            "storage_write_seal_phase_ordinal": HOST_WRITE_SEAL_PHASE_ORDINAL,
        }


@final
@dataclass(frozen=True, slots=True)
class HostGoWriteSealEndpointProjectionV1(_RuntimeSealed):
    """Nonserialized projection connecting phase-10 GO to the later phase-17 seal."""

    host_go: ArtifactRefV1
    write_seal: ArtifactRefV1
    host_go_phase_ordinal: int = HOST_GO_PHASE_ORDINAL
    write_seal_phase_ordinal: int = HOST_WRITE_SEAL_PHASE_ORDINAL

    def __post_init__(self) -> None:
        _require_artifact(self.host_go, HOST_GO_V3_SCHEMA_VERSION, "endpoint host GO")
        _require_artifact(
            self.write_seal,
            IRREVERSIBLE_WRITE_SEAL_V1_SCHEMA_VERSION,
            "endpoint write seal",
        )
        _require_distinct_artifacts((self.host_go, self.write_seal), "GO/seal endpoints")
        if (
            _require_int(
                self.host_go_phase_ordinal,
                "GO endpoint phase ordinal",
                maximum=HOST_GO_PHASE_ORDINAL,
            )
            != HOST_GO_PHASE_ORDINAL
        ):
            _fail("GO endpoint phase ordinal differs")
        if (
            _require_int(
                self.write_seal_phase_ordinal,
                "write-seal endpoint phase ordinal",
                maximum=HOST_WRITE_SEAL_PHASE_ORDINAL,
            )
            != HOST_WRITE_SEAL_PHASE_ORDINAL
        ):
            _fail("write-seal endpoint phase ordinal differs")


@final
@dataclass(frozen=True, slots=True)
class HostProvisioningPreGoBundleV3(_RuntimeSealed):
    """Typed original five-role provisioning chain through pre-GO."""

    policy: provisioning_v3.HostTrustPolicyV1
    statement: provisioning_v3.HostProvisioningStatementV1
    signature_verification_receipt: provisioning_v3.HostSignatureVerificationReceiptV1
    pre_capability_live_validation: provisioning_v3.HostLiveValidationReceiptV1
    pre_go_live_validation: provisioning_v3.HostLiveValidationReceiptV1

    def __post_init__(self) -> None:
        for value, expected, label in (
            (self.policy, provisioning_v3.HostTrustPolicyV1, "host policy"),
            (
                self.statement,
                provisioning_v3.HostProvisioningStatementV1,
                "host statement",
            ),
            (
                self.signature_verification_receipt,
                provisioning_v3.HostSignatureVerificationReceiptV1,
                "host verification",
            ),
            (
                self.pre_capability_live_validation,
                provisioning_v3.HostLiveValidationReceiptV1,
                "host pre-capability validation",
            ),
            (
                self.pre_go_live_validation,
                provisioning_v3.HostLiveValidationReceiptV1,
                "host pre-GO validation",
            ),
        ):
            _require_exact_type(value, expected, label)
        try:
            provisioning_v3.validate_host_live_validation_receipt_v1(
                self.policy,
                self.statement,
                self.signature_verification_receipt,
                self.pre_capability_live_validation,
                previous_receipt=None,
            )
            provisioning_v3.validate_host_live_validation_receipt_v1(
                self.policy,
                self.statement,
                self.signature_verification_receipt,
                self.pre_go_live_validation,
                previous_receipt=self.pre_capability_live_validation,
            )
        except provisioning_v3.ForagerMatchedV3HostProvisioningV3Error as exc:
            raise ForagerMatchedV3HostQualificationProtocolV3Error(str(exc)) from exc
        if (
            self.pre_capability_live_validation.checkpoint != "pre_capability"
            or self.pre_go_live_validation.checkpoint != "pre_go"
        ):
            _fail("host pre-GO bundle checkpoint order differs")

    @property
    def policy_ref(self) -> ArtifactRefV1:
        self.__post_init__()
        return _ref_from_provisioning(provisioning_v3.host_trust_policy_identity_v1(self.policy))

    @property
    def statement_ref(self) -> ArtifactRefV1:
        self.__post_init__()
        return _ref_from_provisioning(
            provisioning_v3.host_provisioning_statement_identity_v1(self.statement)
        )

    @property
    def verification_ref(self) -> ArtifactRefV1:
        self.__post_init__()
        return _ref_from_provisioning(
            provisioning_v3.host_signature_verification_receipt_identity_v1(
                self.signature_verification_receipt
            )
        )

    @property
    def pre_capability_ref(self) -> ArtifactRefV1:
        self.__post_init__()
        return _ref_from_provisioning(
            provisioning_v3.host_live_validation_receipt_identity_v1(
                self.pre_capability_live_validation
            )
        )

    @property
    def pre_go_ref(self) -> ArtifactRefV1:
        self.__post_init__()
        return _ref_from_provisioning(
            provisioning_v3.host_live_validation_receipt_identity_v1(self.pre_go_live_validation)
        )


@final
@dataclass(frozen=True, slots=True)
class HostStorageProducerPreGoBundleV1(_RuntimeSealed):
    """Typed separate six-role signature and two-checkpoint live chain."""

    policy: HostStorageProducerTrustPolicyV1
    statement: HostStorageProducerInventoryStatementV1
    signature_verification_receipt: HostStorageProducerInventorySignatureVerificationReceiptV1
    pre_capability_live_validation: HostStorageProducerLiveValidationReceiptV1
    pre_go_live_validation: HostStorageProducerLiveValidationReceiptV1

    def __post_init__(self) -> None:
        validate_host_storage_producer_pre_go_bundle_v1(self)

    @property
    def policy_ref(self) -> ArtifactRefV1:
        self.__post_init__()
        return host_storage_producer_trust_policy_v1_identity(self.policy)

    @property
    def statement_ref(self) -> ArtifactRefV1:
        self.__post_init__()
        return host_storage_producer_inventory_statement_v1_identity(self.statement)

    @property
    def verification_ref(self) -> ArtifactRefV1:
        self.__post_init__()
        return host_storage_producer_inventory_signature_verification_receipt_v1_identity(
            self.signature_verification_receipt
        )

    @property
    def pre_capability_ref(self) -> ArtifactRefV1:
        self.__post_init__()
        return host_storage_producer_live_validation_receipt_v1_identity(
            self.pre_capability_live_validation
        )

    @property
    def pre_go_ref(self) -> ArtifactRefV1:
        self.__post_init__()
        return host_storage_producer_live_validation_receipt_v1_identity(
            self.pre_go_live_validation
        )


@final
@dataclass(frozen=True, slots=True)
class HostQualificationPreGoBundleV3(_RuntimeSealed):
    """Typed pair of independent pre-GO chains and their case-specific prefix."""

    host_provisioning: HostProvisioningPreGoBundleV3
    storage_producers: HostStorageProducerPreGoBundleV1
    validated_prefix: HostProvisioningValidatedPreGoPrefixV3

    def __post_init__(self) -> None:
        _require_exact_type(
            self.host_provisioning,
            HostProvisioningPreGoBundleV3,
            "pre-GO host bundle",
        )
        _require_exact_type(
            self.storage_producers,
            HostStorageProducerPreGoBundleV1,
            "pre-GO storage producer bundle",
        )
        _require_exact_type(
            self.validated_prefix,
            HostProvisioningValidatedPreGoPrefixV3,
            "validated pre-GO prefix",
        )
        validate_host_pre_go_prefix_v3(
            self.validated_prefix,
            self.host_provisioning,
            self.storage_producers,
        )


def canonical_host_qualification_protocol_descriptor_v3_body_bytes(
    descriptor: HostQualificationProtocolDescriptorV3,
) -> bytes:
    return _canonical_body_bytes(descriptor, HostQualificationProtocolDescriptorV3)


def canonical_host_qualification_protocol_descriptor_v3_file_bytes(
    descriptor: HostQualificationProtocolDescriptorV3,
) -> bytes:
    return _canonical_file_bytes(
        descriptor,
        HostQualificationProtocolDescriptorV3,
        _DESCRIPTOR_BODY_FIELD,
    )


def host_qualification_protocol_descriptor_v3_identity() -> ArtifactRefV1:
    descriptor = HostQualificationProtocolDescriptorV3()
    return _identity(
        HOST_QUALIFICATION_PROTOCOL_DESCRIPTOR_V3_SCHEMA_VERSION,
        canonical_host_qualification_protocol_descriptor_v3_body_bytes(descriptor),
        canonical_host_qualification_protocol_descriptor_v3_file_bytes(descriptor),
    )


def canonical_host_storage_producer_trust_policy_v1_body_bytes(
    policy: HostStorageProducerTrustPolicyV1,
) -> bytes:
    return _canonical_body_bytes(policy, HostStorageProducerTrustPolicyV1)


def canonical_host_storage_producer_trust_policy_v1_file_bytes(
    policy: HostStorageProducerTrustPolicyV1,
) -> bytes:
    return _canonical_file_bytes(
        policy,
        HostStorageProducerTrustPolicyV1,
        _PRODUCER_POLICY_BODY_FIELD,
    )


def host_storage_producer_trust_policy_v1_identity(
    policy: HostStorageProducerTrustPolicyV1,
) -> ArtifactRefV1:
    return _identity(
        HOST_STORAGE_PRODUCER_TRUST_POLICY_V1_SCHEMA_VERSION,
        canonical_host_storage_producer_trust_policy_v1_body_bytes(policy),
        canonical_host_storage_producer_trust_policy_v1_file_bytes(policy),
    )


def canonical_host_storage_producer_inventory_statement_v1_signed_payload_bytes(
    statement: HostStorageProducerInventoryStatementV1,
) -> bytes:
    _require_exact_type(
        statement,
        HostStorageProducerInventoryStatementV1,
        "producer inventory statement",
    )
    return HOST_STORAGE_PRODUCER_SIGNATURE_DOMAIN + canonical_json_bytes(
        statement.to_unsigned_dict(),
        final_lf=False,
    )


def canonical_host_storage_producer_inventory_statement_v1_body_bytes(
    statement: HostStorageProducerInventoryStatementV1,
) -> bytes:
    return _canonical_body_bytes(statement, HostStorageProducerInventoryStatementV1)


def canonical_host_storage_producer_inventory_statement_v1_file_bytes(
    statement: HostStorageProducerInventoryStatementV1,
) -> bytes:
    return _canonical_file_bytes(
        statement,
        HostStorageProducerInventoryStatementV1,
        _PRODUCER_STATEMENT_BODY_FIELD,
    )


def host_storage_producer_inventory_statement_v1_identity(
    statement: HostStorageProducerInventoryStatementV1,
) -> ArtifactRefV1:
    return _identity(
        HOST_STORAGE_PRODUCER_INVENTORY_STATEMENT_V1_SCHEMA_VERSION,
        canonical_host_storage_producer_inventory_statement_v1_body_bytes(statement),
        canonical_host_storage_producer_inventory_statement_v1_file_bytes(statement),
    )


def canonical_host_storage_producer_inventory_signature_verification_receipt_v1_body_bytes(
    receipt: HostStorageProducerInventorySignatureVerificationReceiptV1,
) -> bytes:
    return _canonical_body_bytes(
        receipt,
        HostStorageProducerInventorySignatureVerificationReceiptV1,
    )


def canonical_host_storage_producer_inventory_signature_verification_receipt_v1_file_bytes(
    receipt: HostStorageProducerInventorySignatureVerificationReceiptV1,
) -> bytes:
    return _canonical_file_bytes(
        receipt,
        HostStorageProducerInventorySignatureVerificationReceiptV1,
        _PRODUCER_VERIFICATION_BODY_FIELD,
    )


def host_storage_producer_inventory_signature_verification_receipt_v1_identity(
    receipt: HostStorageProducerInventorySignatureVerificationReceiptV1,
) -> ArtifactRefV1:
    return _identity(
        HOST_STORAGE_PRODUCER_INVENTORY_SIGNATURE_VERIFICATION_RECEIPT_V1_SCHEMA_VERSION,
        canonical_host_storage_producer_inventory_signature_verification_receipt_v1_body_bytes(
            receipt
        ),
        canonical_host_storage_producer_inventory_signature_verification_receipt_v1_file_bytes(
            receipt
        ),
    )


def canonical_host_storage_producer_live_validation_receipt_v1_body_bytes(
    receipt: HostStorageProducerLiveValidationReceiptV1,
) -> bytes:
    return _canonical_body_bytes(receipt, HostStorageProducerLiveValidationReceiptV1)


def canonical_host_storage_producer_live_validation_receipt_v1_file_bytes(
    receipt: HostStorageProducerLiveValidationReceiptV1,
) -> bytes:
    return _canonical_file_bytes(
        receipt,
        HostStorageProducerLiveValidationReceiptV1,
        _PRODUCER_LIVE_BODY_FIELD,
    )


def host_storage_producer_live_validation_receipt_v1_identity(
    receipt: HostStorageProducerLiveValidationReceiptV1,
) -> ArtifactRefV1:
    return _identity(
        HOST_STORAGE_PRODUCER_LIVE_VALIDATION_RECEIPT_V1_SCHEMA_VERSION,
        canonical_host_storage_producer_live_validation_receipt_v1_body_bytes(receipt),
        canonical_host_storage_producer_live_validation_receipt_v1_file_bytes(receipt),
    )


def canonical_host_provisioning_validated_pre_go_prefix_v3_body_bytes(
    prefix: HostProvisioningValidatedPreGoPrefixV3,
) -> bytes:
    return _canonical_body_bytes(prefix, HostProvisioningValidatedPreGoPrefixV3)


def canonical_host_provisioning_validated_pre_go_prefix_v3_file_bytes(
    prefix: HostProvisioningValidatedPreGoPrefixV3,
) -> bytes:
    return _canonical_file_bytes(
        prefix,
        HostProvisioningValidatedPreGoPrefixV3,
        _PREFIX_BODY_FIELD,
    )


def host_provisioning_validated_pre_go_prefix_v3_identity(
    prefix: HostProvisioningValidatedPreGoPrefixV3,
) -> ArtifactRefV1:
    return _identity(
        HOST_PROVISIONING_VALIDATED_PRE_GO_PREFIX_V3_SCHEMA_VERSION,
        canonical_host_provisioning_validated_pre_go_prefix_v3_body_bytes(prefix),
        canonical_host_provisioning_validated_pre_go_prefix_v3_file_bytes(prefix),
    )


def canonical_host_case_request_v3_body_bytes(
    request: HostQualificationCaseRequestV3,
) -> bytes:
    return _canonical_body_bytes(request, HostQualificationCaseRequestV3)


def canonical_host_case_request_v3_file_bytes(
    request: HostQualificationCaseRequestV3,
) -> bytes:
    return _canonical_file_bytes(request, HostQualificationCaseRequestV3, _REQUEST_BODY_FIELD)


def host_case_request_v3_identity(request: HostQualificationCaseRequestV3) -> ArtifactRefV1:
    return _identity(
        HOST_CASE_REQUEST_V3_SCHEMA_VERSION,
        canonical_host_case_request_v3_body_bytes(request),
        canonical_host_case_request_v3_file_bytes(request),
    )


def canonical_host_case_intent_v3_body_bytes(
    intent: HostQualificationCaseIntentV3,
) -> bytes:
    return _canonical_body_bytes(intent, HostQualificationCaseIntentV3)


def canonical_host_case_intent_v3_file_bytes(
    intent: HostQualificationCaseIntentV3,
) -> bytes:
    return _canonical_file_bytes(intent, HostQualificationCaseIntentV3, _INTENT_BODY_FIELD)


def host_case_intent_v3_identity(intent: HostQualificationCaseIntentV3) -> ArtifactRefV1:
    return _identity(
        HOST_CASE_INTENT_V3_SCHEMA_VERSION,
        canonical_host_case_intent_v3_body_bytes(intent),
        canonical_host_case_intent_v3_file_bytes(intent),
    )


def canonical_host_ready_v3_body_bytes(ready: HostReadyV3) -> bytes:
    return _canonical_body_bytes(ready, HostReadyV3)


def canonical_host_ready_v3_file_bytes(ready: HostReadyV3) -> bytes:
    return _canonical_file_bytes(ready, HostReadyV3, _READY_BODY_FIELD)


def host_ready_v3_identity(ready: HostReadyV3) -> ArtifactRefV1:
    return _identity(
        HOST_READY_V3_SCHEMA_VERSION,
        canonical_host_ready_v3_body_bytes(ready),
        canonical_host_ready_v3_file_bytes(ready),
    )


def canonical_host_observer_anchor_v3_body_bytes(anchor: HostObserverAnchorV3) -> bytes:
    return _canonical_body_bytes(anchor, HostObserverAnchorV3)


def canonical_host_observer_anchor_v3_file_bytes(anchor: HostObserverAnchorV3) -> bytes:
    return _canonical_file_bytes(anchor, HostObserverAnchorV3, _ANCHOR_BODY_FIELD)


def host_observer_anchor_v3_identity(anchor: HostObserverAnchorV3) -> ArtifactRefV1:
    return _identity(
        HOST_OBSERVER_ANCHOR_V3_SCHEMA_VERSION,
        canonical_host_observer_anchor_v3_body_bytes(anchor),
        canonical_host_observer_anchor_v3_file_bytes(anchor),
    )


def canonical_host_go_v3_body_bytes(go: HostGoCommitmentV3) -> bytes:
    return _canonical_body_bytes(go, HostGoCommitmentV3)


def canonical_host_go_v3_file_bytes(go: HostGoCommitmentV3) -> bytes:
    return _canonical_file_bytes(go, HostGoCommitmentV3, _GO_BODY_FIELD)


def host_go_v3_identity(go: HostGoCommitmentV3) -> ArtifactRefV1:
    return _identity(
        HOST_GO_V3_SCHEMA_VERSION,
        canonical_host_go_v3_body_bytes(go),
        canonical_host_go_v3_file_bytes(go),
    )


def validate_repository_protocol_pins_v3() -> None:
    """Compatibility shim that permanently delegates repository closure."""

    _fail(
        "repository protocol readiness is permanently delegated to the later typed provider bundle"
    )


def repository_protocol_ready_v3() -> bool:
    """Compatibility shim: this source-only module is permanently nonready."""

    return False


def validate_host_storage_producer_pre_go_bundle_v1(
    bundle: HostStorageProducerPreGoBundleV1,
) -> None:
    """Validate all signed six-role crosslinks without performing cryptography."""

    if type(bundle) is not HostStorageProducerPreGoBundleV1:
        raise TypeError("storage producer pre-GO bundle must use its exact type")
    policy = bundle.policy
    statement = bundle.statement
    verification = bundle.signature_verification_receipt
    pre_capability = bundle.pre_capability_live_validation
    pre_go = bundle.pre_go_live_validation
    for value, expected, label in (
        (policy, HostStorageProducerTrustPolicyV1, "producer policy"),
        (statement, HostStorageProducerInventoryStatementV1, "producer statement"),
        (
            verification,
            HostStorageProducerInventorySignatureVerificationReceiptV1,
            "producer verification",
        ),
        (
            pre_capability,
            HostStorageProducerLiveValidationReceiptV1,
            "producer pre-capability validation",
        ),
        (
            pre_go,
            HostStorageProducerLiveValidationReceiptV1,
            "producer pre-GO validation",
        ),
    ):
        _require_exact_type(value, expected, label)
    policy_ref = host_storage_producer_trust_policy_v1_identity(policy)
    statement_ref = host_storage_producer_inventory_statement_v1_identity(statement)
    verification_ref = host_storage_producer_inventory_signature_verification_receipt_v1_identity(
        verification
    )
    pre_capability_ref = host_storage_producer_live_validation_receipt_v1_identity(pre_capability)
    if (
        statement.policy != policy_ref
        or verification.policy != policy_ref
        or pre_capability.policy != policy_ref
        or pre_go.policy != policy_ref
    ):
        _fail("storage producer chain crosswires its trust policy")
    if (
        verification.statement != statement_ref
        or pre_capability.statement != statement_ref
        or pre_go.statement != statement_ref
    ):
        _fail("storage producer chain crosswires its inventory statement")
    if (
        pre_capability.signature_verification_receipt != verification_ref
        or pre_go.signature_verification_receipt != verification_ref
    ):
        _fail("storage producer live chain crosswires signature verification")
    if pre_capability.previous_live_validation_receipt is not None or (
        pre_go.previous_live_validation_receipt != pre_capability_ref
    ):
        _fail("storage producer live-chain predecessor differs")
    if (
        pre_capability.checkpoint != "pre_capability"
        or pre_capability.checkpoint_ordinal != 0
        or pre_go.checkpoint != "pre_go"
        or pre_go.checkpoint_ordinal != 1
    ):
        _fail("storage producer live checkpoint order differs")
    if (
        statement.qualification_plan != policy.qualification_plan
        or statement.storage_backend_policy != policy.storage_backend_policy
        or statement.producer_facts != policy.expected_producer_facts
        or pre_capability.observed_producer_facts != policy.expected_producer_facts
        or pre_go.observed_producer_facts != policy.expected_producer_facts
    ):
        _fail("storage producer plan, policy, or six-role inventory drifted")
    if (
        statement.signer_key_id != policy.signer_key_id
        or statement.signer_public_key_sha256 != policy.signer_public_key_sha256
        or verification.signer_key_id != policy.signer_key_id
        or verification.signer_public_key_sha256 != policy.signer_public_key_sha256
    ):
        _fail("storage producer signer projection differs")
    if verification.verifier != policy.independent_verifier:
        _fail("storage producer verification uses an unpinned verifier")
    if (
        pre_capability.validator != policy.live_validator
        or pre_go.validator != policy.live_validator
    ):
        _fail("storage producer live validation uses an unpinned validator")
    if verification.signed_payload_sha256 != statement.signed_payload_sha256:
        _fail("storage producer signed-payload identity differs")
    signature_sha256 = hashlib.sha256(bytes.fromhex(statement.signature_hex)).hexdigest()
    if verification.signature_sha256 != signature_sha256:
        _fail("storage producer signature identity differs")
    if not (
        policy.valid_from_unix_ns
        <= statement.observed_at_unix_ns
        < verification.verification_started_at_unix_ns
        < verification.verification_completed_at_unix_ns
        < pre_capability.validated_at_unix_ns
        < pre_go.validated_at_unix_ns
        <= policy.valid_until_unix_ns
    ):
        _fail("storage producer Unix chronology differs or leaves policy validity")
    _require_strict_times(
        (
            statement.observed_at_monotonic_ns,
            verification.verification_started_at_monotonic_ns,
            verification.verification_completed_at_monotonic_ns,
            pre_capability.validated_at_monotonic_ns,
            pre_go.validated_at_monotonic_ns,
        ),
        "storage producer monotonic chronology",
    )


def _producer_fact_matches_legacy_component(
    fact: HostStorageProducerRuntimeFactV1,
    component: provisioning_v3.PinnedComponentIdentityV1,
) -> bool:
    return (
        fact.component_id == component.component_id
        and fact.producer.descriptor_schema_version == component.descriptor_schema_version
        and fact.producer.descriptor_file_sha256 == component.descriptor_file_sha256
        and fact.producer.source_sha256 == component.source_sha256
        and fact.runtime_artifact_sha256 == component.runtime_artifact_sha256
    )


def validate_host_pre_go_prefix_v3(
    prefix: HostProvisioningValidatedPreGoPrefixV3,
    host_bundle: HostProvisioningPreGoBundleV3,
    producer_bundle: HostStorageProducerPreGoBundleV1,
) -> None:
    """Validate the exact union of the original and separate six-role chains."""

    for value, expected, label in (
        (prefix, HostProvisioningValidatedPreGoPrefixV3, "validated prefix"),
        (host_bundle, HostProvisioningPreGoBundleV3, "host pre-GO bundle"),
        (
            producer_bundle,
            HostStorageProducerPreGoBundleV1,
            "storage producer pre-GO bundle",
        ),
    ):
        _require_exact_type(value, expected, label)
    producer_policy = producer_bundle.policy
    if producer_policy.host_trust_policy != host_bundle.policy_ref:
        _fail("six-role policy crosswires the original host trust policy")
    expected_plan = _ref_from_provisioning(host_bundle.policy.qualification_plan)
    if (
        prefix.qualification_plan != expected_plan
        or producer_policy.qualification_plan != expected_plan
    ):
        _fail("pre-GO chains crosswire the qualification plan")
    if not producer_policy.independent_verifier.matches_legacy(
        host_bundle.policy.independent_verifier
    ):
        _fail("six-role policy verifier differs from the host-pinned verifier")
    if not producer_policy.live_validator.matches_legacy(host_bundle.policy.live_validator):
        _fail("six-role policy validator differs from the host-pinned validator")
    measurement = host_bundle.policy.expected_facts.components.storage_measurement_producer
    terminal = host_bundle.policy.expected_facts.components.storage_terminal_relay
    if not _producer_fact_matches_legacy_component(
        producer_policy.expected_producer_facts[0],
        measurement,
    ) or not _producer_fact_matches_legacy_component(
        producer_policy.expected_producer_facts[1],
        terminal,
    ):
        _fail("six-role policy does not extend the two host-pinned storage roles")
    if (
        prefix.host_trust_policy != host_bundle.policy_ref
        or prefix.host_provisioning_statement != host_bundle.statement_ref
        or prefix.host_signature_verification_receipt != host_bundle.verification_ref
        or prefix.host_pre_capability_live_validation != host_bundle.pre_capability_ref
        or prefix.host_pre_go_live_validation != host_bundle.pre_go_ref
    ):
        _fail("validated prefix crosswires the original host provisioning chain")
    if (
        prefix.storage_producer_trust_policy != producer_bundle.policy_ref
        or prefix.storage_producer_inventory_statement != producer_bundle.statement_ref
        or prefix.storage_producer_signature_verification_receipt
        != producer_bundle.verification_ref
        or prefix.storage_producer_pre_capability_live_validation
        != producer_bundle.pre_capability_ref
        or prefix.storage_producer_pre_go_live_validation != producer_bundle.pre_go_ref
    ):
        _fail("validated prefix crosswires the six-role producer chain")
    if (
        prefix.storage_backend_policy != producer_policy.storage_backend_policy
        or prefix.storage_producers != producer_policy.storage_producers
    ):
        _fail("validated prefix storage policy or six-role projection differs")
    if prefix.prefix_committed_monotonic_ns <= max(
        host_bundle.pre_go_live_validation.validated_at_monotonic_ns,
        producer_bundle.pre_go_live_validation.validated_at_monotonic_ns,
    ):
        _fail("validated prefix precedes a pre-GO live validation")


def _request_v2_ref(value: executor_v2.HostQualificationCaseRequestV2) -> ArtifactRefV1:
    return _executor_artifact_ref(
        value,
        executor_v2.HostQualificationCaseRequestV2,
        HOST_CASE_REQUEST_V2_SCHEMA_VERSION,
        executor_v2.canonical_host_case_request_v2_body_bytes,
        executor_v2.canonical_host_case_request_v2_file_bytes,
    )


def _intent_v2_ref(value: executor_v2.HostQualificationCaseIntentV2) -> ArtifactRefV1:
    return _executor_artifact_ref(
        value,
        executor_v2.HostQualificationCaseIntentV2,
        HOST_CASE_INTENT_V2_SCHEMA_VERSION,
        executor_v2.canonical_host_case_intent_v2_body_bytes,
        executor_v2.canonical_host_case_intent_v2_file_bytes,
    )


def _initial_sample_v2_ref(value: executor_v2.HostInitialCgroupSampleV2) -> ArtifactRefV1:
    return _executor_artifact_ref(
        value,
        executor_v2.HostInitialCgroupSampleV2,
        HOST_INITIAL_CGROUP_SAMPLE_V2_SCHEMA_VERSION,
        executor_v2.canonical_host_initial_cgroup_sample_v2_body_bytes,
        executor_v2.canonical_host_initial_cgroup_sample_v2_file_bytes,
    )


def _ready_v2_ref(value: executor_v2.HostReadyV2) -> ArtifactRefV1:
    return _executor_artifact_ref(
        value,
        executor_v2.HostReadyV2,
        HOST_READY_V2_SCHEMA_VERSION,
        executor_v2.canonical_host_ready_v2_body_bytes,
        executor_v2.canonical_host_ready_v2_file_bytes,
    )


def _anchor_v2_ref(value: executor_v2.HostObserverAnchorV2) -> ArtifactRefV1:
    return _executor_artifact_ref(
        value,
        executor_v2.HostObserverAnchorV2,
        HOST_OBSERVER_ANCHOR_V2_SCHEMA_VERSION,
        executor_v2.canonical_host_observer_anchor_v2_body_bytes,
        executor_v2.canonical_host_observer_anchor_v2_file_bytes,
    )


def _go_v2_ref(value: executor_v2.HostGoCommitmentV2) -> ArtifactRefV1:
    return _executor_artifact_ref(
        value,
        executor_v2.HostGoCommitmentV2,
        HOST_GO_V2_SCHEMA_VERSION,
        executor_v2.canonical_host_go_v2_body_bytes,
        executor_v2.canonical_host_go_v2_file_bytes,
    )


def _assert_same_projection(
    values: Sequence[object],
    labels: Sequence[str],
) -> None:
    if len(values) != len(labels) or not values:
        _fail("projection comparison is empty or malformed")
    if any(value != values[0] for value in values[1:]):
        _fail("case projection differs for " + ", ".join(labels))


def validate_host_request_intent_v3_chain(
    request: HostQualificationCaseRequestV3,
    intent: HostQualificationCaseIntentV3,
) -> None:
    """Validate the additive phase-0 through phase-2 crosslinks."""

    _require_exact_type(request, HostQualificationCaseRequestV3, "request")
    _require_exact_type(intent, HostQualificationCaseIntentV3, "intent")
    if intent.request != host_case_request_v3_identity(request):
        _fail("intent does not bind the exact request-v3 FILE and BODY")
    for field in (
        "campaign_id",
        "case_spine_sha256",
        "subject",
        "qualification_plan",
        "case_execution_ticket",
        "storage_backend_policy",
        "storage_producer_trust_policy",
        "storage_producer_inventory_statement",
        "storage_producer_signature_verification_receipt",
        "storage_producer_pre_capability_live_validation",
        "storage_producers",
        "image_id",
        "container_name",
        "max_temporary_peak_bytes",
    ):
        if getattr(intent, field) != getattr(request, field):
            _fail(f"request-to-intent v3 projection differs for {field}")
    _require_strict_times(
        (
            request.request_validated_monotonic_ns,
            intent.authorization_validated_monotonic_ns,
            intent.intent_committed_monotonic_ns,
        ),
        "request/intent first-three-phase chronology",
    )


def validate_host_ready_anchor_go_v3_chain(
    intent: HostQualificationCaseIntentV3,
    ready: HostReadyV3,
    anchor: HostObserverAnchorV3,
    prefix: HostProvisioningValidatedPreGoPrefixV3,
    go: HostGoCommitmentV3,
) -> None:
    """Validate the v3 READY/anchor/prefix/GO handshake and shared projections."""

    expected_types = (
        (intent, HostQualificationCaseIntentV3),
        (ready, HostReadyV3),
        (anchor, HostObserverAnchorV3),
        (prefix, HostProvisioningValidatedPreGoPrefixV3),
        (go, HostGoCommitmentV3),
    )
    for value, expected in expected_types:
        _require_exact_type(value, expected, "READY/anchor/prefix/GO artifact")
    if ready.intent != host_case_intent_v3_identity(intent):
        _fail("READY does not bind the exact intent-v3 FILE and BODY")
    if anchor.ready != host_ready_v3_identity(ready):
        _fail("observer anchor does not bind the exact READY-v3 FILE and BODY")
    if (
        prefix.intent != host_case_intent_v3_identity(intent)
        or prefix.ready != host_ready_v3_identity(ready)
        or prefix.observer_anchor != host_observer_anchor_v3_identity(anchor)
    ):
        _fail("validated prefix crosswires intent, READY, or observer anchor")
    if prefix.request != intent.request:
        _fail("validated prefix crosswires the request carried by intent")
    for field in (
        "storage_producer_trust_policy",
        "storage_producer_inventory_statement",
        "storage_producer_signature_verification_receipt",
        "storage_producer_pre_capability_live_validation",
    ):
        if getattr(prefix, field) != getattr(intent, field):
            _fail(f"validated prefix crosswires intent predecessor {field}")
    if (
        go.ready != host_ready_v3_identity(ready)
        or go.observer_anchor != host_observer_anchor_v3_identity(anchor)
        or go.validated_pre_go_prefix
        != host_provisioning_validated_pre_go_prefix_v3_identity(prefix)
    ):
        _fail("GO crosswires READY, observer anchor, or validated prefix")
    for field in (
        "campaign_id",
        "case_spine_sha256",
        "subject",
        "qualification_plan",
        "storage_backend_policy",
        "storage_producers",
        "image_id",
        "container_name",
        "max_temporary_peak_bytes",
    ):
        values = tuple(getattr(item, field) for item in (intent, ready, anchor, prefix, go))
        _assert_same_projection(values, (field,) * len(values))
    for field in (
        "container_id_commitment_sha256",
        "outer_cgroup_identity_sha256",
        "aggregate_root_case_exclusive",
    ):
        values = tuple(getattr(item, field) for item in (ready, anchor, prefix, go))
        _assert_same_projection(values, (field,) * len(values))
    _require_strict_times(
        (
            intent.intent_committed_monotonic_ns,
            ready.fresh_cgroup_created_monotonic_ns,
            ready.retained_counter_fds_opened_monotonic_ns,
            ready.initial_cgroup_sample_committed_monotonic_ns,
            ready.container_created_monotonic_ns,
            ready.container_started_monotonic_ns,
            ready.driver_ready_monotonic_ns,
            anchor.observer_anchored_monotonic_ns,
            prefix.prefix_committed_monotonic_ns,
            go.storage_runtime_intent_committed_monotonic_ns,
            go.go_committed_monotonic_ns,
        ),
        "intent-through-GO chronology",
    )


def validate_host_qualification_v3_chain(
    *,
    base_request: executor_v2.HostQualificationCaseRequestV2,
    base_intent: executor_v2.HostQualificationCaseIntentV2,
    base_initial_sample: executor_v2.HostInitialCgroupSampleV2,
    base_ready: executor_v2.HostReadyV2,
    base_anchor: executor_v2.HostObserverAnchorV2,
    base_go: executor_v2.HostGoCommitmentV2,
    request: HostQualificationCaseRequestV3,
    intent: HostQualificationCaseIntentV3,
    ready: HostReadyV3,
    anchor: HostObserverAnchorV3,
    pre_go: HostQualificationPreGoBundleV3,
    go: HostGoCommitmentV3,
) -> None:
    """Validate exact base-v2 closure and the additive first-eleven-phase chain."""

    exact_types = (
        (base_request, executor_v2.HostQualificationCaseRequestV2),
        (base_intent, executor_v2.HostQualificationCaseIntentV2),
        (base_initial_sample, executor_v2.HostInitialCgroupSampleV2),
        (base_ready, executor_v2.HostReadyV2),
        (base_anchor, executor_v2.HostObserverAnchorV2),
        (base_go, executor_v2.HostGoCommitmentV2),
        (request, HostQualificationCaseRequestV3),
        (intent, HostQualificationCaseIntentV3),
        (ready, HostReadyV3),
        (anchor, HostObserverAnchorV3),
        (pre_go, HostQualificationPreGoBundleV3),
        (go, HostGoCommitmentV3),
    )
    for value, expected in exact_types:
        _require_exact_type(value, expected, "host-v3 chain artifact")
    try:
        executor_v2.validate_host_request_intent_v2_chain(base_request, base_intent)
        executor_v2.validate_host_ready_anchor_go_v2_chain(
            base_intent,
            base_initial_sample,
            base_ready,
            base_anchor,
            base_go,
        )
    except executor_v2.ForagerMatchedV3HostQualificationExecutorV2Error as exc:
        raise ForagerMatchedV3HostQualificationProtocolV3Error(str(exc)) from exc

    validate_host_pre_go_prefix_v3(
        pre_go.validated_prefix,
        pre_go.host_provisioning,
        pre_go.storage_producers,
    )
    validate_host_request_intent_v3_chain(request, intent)
    validate_host_ready_anchor_go_v3_chain(intent, ready, anchor, pre_go.validated_prefix, go)

    subject_projection = (
        base_request.case_ordinal,
        base_request.candidate_id,
        base_request.candidate_family,
        base_request.qualification_case_id,
    )
    if subject_projection != (
        request.subject.case_ordinal,
        request.subject.candidate_id,
        request.subject.candidate_family,
        request.subject.qualification_case_id,
    ):
        _fail("request subject differs from base-v2")
    base_plan = _ref_from_executor_identity(base_request.qualification_plan)
    base_ticket = _ref_from_executor_identity(base_request.case_execution_ticket)
    base_runtime = _ref_from_executor_identity(base_request.runtime_qualification_receipt)
    if (
        request.base_request_v2 != _request_v2_ref(base_request)
        or request.qualification_plan != base_plan
        or request.case_execution_ticket != base_ticket
        or request.runtime_qualification_receipt != base_runtime
        or request.resource_requirement_body_sha256 != base_request.resource_requirement_body_sha256
        or request.declared_ceilings != base_request.declared_ceilings
        or request.image_id != base_request.image_id
        or request.container_name != base_request.container_name
        or request.max_temporary_peak_bytes != base_request.declared_ceilings[23][1]
    ):
        _fail("request-v3 differs from its exact base-v2 projection")
    if (
        base_request.host_executor.descriptor_sha256
        != request.host_executor_v2_descriptor.file_sha256
        or base_request.host_executor.source_sha256 != request.host_executor_v2_source_sha256
    ):
        _fail("request-v3 host executor differs from base-v2")
    host_bundle = pre_go.host_provisioning
    producer_bundle = pre_go.storage_producers
    if (
        request.host_trust_policy != host_bundle.policy_ref
        or request.host_provisioning_statement != host_bundle.statement_ref
        or request.host_signature_verification_receipt != host_bundle.verification_ref
        or request.host_pre_capability_live_validation != host_bundle.pre_capability_ref
        or request.storage_producer_trust_policy != producer_bundle.policy_ref
        or request.storage_producer_inventory_statement != producer_bundle.statement_ref
        or request.storage_producer_signature_verification_receipt
        != producer_bundle.verification_ref
        or request.storage_producer_pre_capability_live_validation
        != producer_bundle.pre_capability_ref
    ):
        _fail("request-v3 pre-capability trust references differ")
    if (
        intent.request != host_case_request_v3_identity(request)
        or intent.base_intent_v2 != _intent_v2_ref(base_intent)
        or intent.qualification_plan != request.qualification_plan
        or intent.case_execution_ticket != request.case_execution_ticket
        or intent.storage_backend_policy != request.storage_backend_policy
        or intent.storage_producer_trust_policy != request.storage_producer_trust_policy
        or intent.storage_producer_inventory_statement
        != request.storage_producer_inventory_statement
        or intent.storage_producer_signature_verification_receipt
        != request.storage_producer_signature_verification_receipt
        or intent.storage_producer_pre_capability_live_validation
        != request.storage_producer_pre_capability_live_validation
    ):
        _fail("request-to-intent v3 crosslink or trust projection differs")
    if (
        ready.intent != host_case_intent_v3_identity(intent)
        or ready.base_initial_cgroup_sample_v2 != _initial_sample_v2_ref(base_initial_sample)
        or ready.base_ready_v2 != _ready_v2_ref(base_ready)
        or ready.container_id != base_ready.container_id
        or ready.container_id_commitment_sha256 != base_ready.container_identity_sha256
        or ready.outer_cgroup_identity_sha256 != base_ready.cgroup_identity_sha256
        or ready.initial_cgroup_sample_committed_monotonic_ns
        != base_initial_sample.facts.monotonic_ns
        or ready.driver_ready_monotonic_ns != base_ready.ready_monotonic_ns
    ):
        _fail("intent-to-READY v3 crosslink or runtime projection differs")
    membership_observer = host_bundle.policy.expected_facts.components.membership_observer
    if (
        anchor.ready != host_ready_v3_identity(ready)
        or anchor.base_observer_anchor_v2 != _anchor_v2_ref(base_anchor)
        or anchor.membership_observer.matches_legacy(membership_observer) is not True
        or anchor.observer_anchored_monotonic_ns != base_anchor.observer_started_monotonic_ns
    ):
        _fail("READY-to-anchor v3 crosslink or observer projection differs")
    prefix = pre_go.validated_prefix
    if (
        prefix.request != host_case_request_v3_identity(request)
        or prefix.intent != host_case_intent_v3_identity(intent)
        or prefix.ready != host_ready_v3_identity(ready)
        or prefix.observer_anchor != host_observer_anchor_v3_identity(anchor)
    ):
        _fail("validated pre-GO prefix crosswires the v3 handshake")
    if (
        go.base_go_v2 != _go_v2_ref(base_go)
        or go.ready != prefix.ready
        or go.observer_anchor != prefix.observer_anchor
        or go.validated_pre_go_prefix
        != host_provisioning_validated_pre_go_prefix_v3_identity(prefix)
        or go.go_committed_monotonic_ns != base_go.go_committed_monotonic_ns
    ):
        _fail("anchor-to-GO v3 crosslink or base-v2 projection differs")
    shared_values = (
        request.campaign_id,
        intent.campaign_id,
        ready.campaign_id,
        anchor.campaign_id,
        prefix.campaign_id,
        go.campaign_id,
    )
    _assert_same_projection(shared_values, ("request", "intent", "READY", "anchor", "prefix", "GO"))
    for field in (
        "case_spine_sha256",
        "subject",
        "qualification_plan",
        "storage_backend_policy",
        "storage_producers",
        "image_id",
        "container_name",
        "max_temporary_peak_bytes",
    ):
        values = tuple(
            getattr(item, field) for item in (request, intent, ready, anchor, prefix, go)
        )
        _assert_same_projection(values, (field,) * len(values))
    for field in (
        "container_id_commitment_sha256",
        "outer_cgroup_identity_sha256",
        "aggregate_root_case_exclusive",
    ):
        values = tuple(getattr(item, field) for item in (ready, anchor, prefix, go))
        _assert_same_projection(values, (field,) * len(values))
    phase_times = (
        request.request_validated_monotonic_ns,
        intent.authorization_validated_monotonic_ns,
        intent.intent_committed_monotonic_ns,
        ready.fresh_cgroup_created_monotonic_ns,
        ready.retained_counter_fds_opened_monotonic_ns,
        ready.initial_cgroup_sample_committed_monotonic_ns,
        ready.container_created_monotonic_ns,
        ready.container_started_monotonic_ns,
        ready.driver_ready_monotonic_ns,
        anchor.observer_anchored_monotonic_ns,
        go.go_committed_monotonic_ns,
    )
    _require_strict_times(phase_times, "first eleven host phases")
    if not (
        host_bundle.pre_capability_live_validation.validated_at_monotonic_ns
        < request.request_validated_monotonic_ns
        and producer_bundle.pre_capability_live_validation.validated_at_monotonic_ns
        < request.request_validated_monotonic_ns
        and anchor.observer_anchored_monotonic_ns
        < host_bundle.pre_go_live_validation.validated_at_monotonic_ns
        < prefix.prefix_committed_monotonic_ns
        and anchor.observer_anchored_monotonic_ns
        < producer_bundle.pre_go_live_validation.validated_at_monotonic_ns
        < prefix.prefix_committed_monotonic_ns
        < go.storage_runtime_intent_committed_monotonic_ns
        < go.go_committed_monotonic_ns
    ):
        _fail("host trust checkpoints, prefix, runtime intent, and GO chronology differs")


def _pop_common_envelope(
    item: dict[str, Any],
    *,
    schema: str,
    status: str,
    label: str,
) -> None:
    if (
        item.pop("schema_version") != schema
        or item.pop("status") != status
        or item.pop("authority") != _authority_dict()
        or item.pop("claims") != _claims_dict()
    ):
        _fail(f"{label} envelope differs")


def _phase_prefix_from_json(value: object, label: str) -> tuple[str, ...]:
    if type(value) is not list or any(type(item) is not str for item in value):
        _fail(f"{label} must be one exact JSON string list")
    return tuple(cast(list[str], value))


def parse_host_qualification_protocol_descriptor_v3(
    raw: bytes,
    *,
    expected_file_sha256: str,
    expected_body_sha256: str,
) -> HostQualificationProtocolDescriptorV3:
    descriptor = HostQualificationProtocolDescriptorV3()
    body = _parse_body(
        raw,
        expected_file_sha256=expected_file_sha256,
        expected_body_sha256=expected_body_sha256,
        body_field=_DESCRIPTOR_BODY_FIELD,
        body_keys=frozenset(descriptor.to_body_dict()),
        label="host protocol descriptor",
    )
    if body != descriptor.to_body_dict():
        _fail("host protocol descriptor content differs")
    if raw != canonical_host_qualification_protocol_descriptor_v3_file_bytes(descriptor):
        _fail("host protocol descriptor canonical replay differs")
    return descriptor


def parse_host_storage_producer_trust_policy_v1(
    raw: bytes,
    *,
    expected_file_sha256: str,
    expected_body_sha256: str,
) -> HostStorageProducerTrustPolicyV1:
    body_keys = frozenset(HostStorageProducerTrustPolicyV1.__dataclass_fields__) | {
        "authority",
        "claims",
        "expected_producer_inventory_sha256",
        "schema_version",
        "signature_algorithm",
        "signature_domain",
        "status",
    }
    item = _parse_body(
        raw,
        expected_file_sha256=expected_file_sha256,
        expected_body_sha256=expected_body_sha256,
        body_field=_PRODUCER_POLICY_BODY_FIELD,
        body_keys=body_keys,
        label="storage producer trust policy",
    )
    _pop_common_envelope(
        item,
        schema=HOST_STORAGE_PRODUCER_TRUST_POLICY_V1_SCHEMA_VERSION,
        status="preissued_six_role_policy_non_authorizing",
        label="storage producer trust policy",
    )
    if (
        item.pop("signature_algorithm") != HOST_STORAGE_PRODUCER_SIGNATURE_ALGORITHM
        or item.pop("signature_domain") != HOST_STORAGE_PRODUCER_SIGNATURE_DOMAIN_LABEL
    ):
        _fail("storage producer policy signature domain differs")
    supplied_inventory = item.pop("expected_producer_inventory_sha256")
    item["qualification_plan"] = _artifact_from_json(
        item["qualification_plan"],
        QUALIFICATION_PLAN_V3_SCHEMA_VERSION,
        "producer policy plan",
    )
    item["storage_backend_policy"] = _artifact_from_json(
        item["storage_backend_policy"],
        STORAGE_BACKEND_POLICY_V1_SCHEMA_VERSION,
        "producer policy storage policy",
    )
    item["host_trust_policy"] = _artifact_from_json(
        item["host_trust_policy"],
        provisioning_v3.HOST_TRUST_POLICY_SCHEMA_VERSION,
        "producer policy host trust policy",
    )
    item["independent_verifier"] = _pinned_host_component_from_json(item["independent_verifier"])
    item["live_validator"] = _pinned_host_component_from_json(item["live_validator"])
    item["expected_producer_facts"] = _producer_facts_from_json(item["expected_producer_facts"])
    result = HostStorageProducerTrustPolicyV1(**item)
    if supplied_inventory != result.expected_producer_inventory_sha256:
        _fail("storage producer policy inventory identity differs")
    if raw != canonical_host_storage_producer_trust_policy_v1_file_bytes(result):
        _fail("storage producer trust policy canonical replay differs")
    return result


def parse_host_storage_producer_inventory_statement_v1(
    raw: bytes,
    *,
    expected_file_sha256: str,
    expected_body_sha256: str,
) -> HostStorageProducerInventoryStatementV1:
    body_keys = frozenset(HostStorageProducerInventoryStatementV1.__dataclass_fields__) | {
        "authority",
        "claims",
        "executor_held_signing_secret",
        "hmac_used",
        "producer_inventory_sha256",
        "schema_version",
        "signature_algorithm",
        "signature_domain",
        "signature_verified_by_parser",
        "signed_payload_sha256",
        "status",
    }
    item = _parse_body(
        raw,
        expected_file_sha256=expected_file_sha256,
        expected_body_sha256=expected_body_sha256,
        body_field=_PRODUCER_STATEMENT_BODY_FIELD,
        body_keys=body_keys,
        label="storage producer inventory statement",
    )
    _pop_common_envelope(
        item,
        schema=HOST_STORAGE_PRODUCER_INVENTORY_STATEMENT_V1_SCHEMA_VERSION,
        status="six_role_inventory_signed_metadata_non_authorizing",
        label="storage producer inventory statement",
    )
    if (
        item.pop("signature_algorithm") != HOST_STORAGE_PRODUCER_SIGNATURE_ALGORITHM
        or item.pop("signature_domain") != HOST_STORAGE_PRODUCER_SIGNATURE_DOMAIN_LABEL
        or item.pop("executor_held_signing_secret") is not False
        or item.pop("hmac_used") is not False
        or item.pop("signature_verified_by_parser") is not False
    ):
        _fail("storage producer inventory statement signature posture differs")
    supplied_inventory = item.pop("producer_inventory_sha256")
    supplied_payload = item.pop("signed_payload_sha256")
    item["policy"] = _artifact_from_json(
        item["policy"],
        HOST_STORAGE_PRODUCER_TRUST_POLICY_V1_SCHEMA_VERSION,
        "producer statement policy",
    )
    item["qualification_plan"] = _artifact_from_json(
        item["qualification_plan"],
        QUALIFICATION_PLAN_V3_SCHEMA_VERSION,
        "producer statement plan",
    )
    item["storage_backend_policy"] = _artifact_from_json(
        item["storage_backend_policy"],
        STORAGE_BACKEND_POLICY_V1_SCHEMA_VERSION,
        "producer statement storage policy",
    )
    item["producer_facts"] = _producer_facts_from_json(item["producer_facts"])
    result = HostStorageProducerInventoryStatementV1(**item)
    if (
        supplied_inventory != result.producer_inventory_sha256
        or supplied_payload != result.signed_payload_sha256
    ):
        _fail("storage producer statement derived identity differs")
    if raw != canonical_host_storage_producer_inventory_statement_v1_file_bytes(result):
        _fail("storage producer inventory statement canonical replay differs")
    return result


def parse_host_storage_producer_inventory_signature_verification_receipt_v1(
    raw: bytes,
    *,
    expected_file_sha256: str,
    expected_body_sha256: str,
) -> HostStorageProducerInventorySignatureVerificationReceiptV1:
    body_keys = frozenset(
        HostStorageProducerInventorySignatureVerificationReceiptV1.__dataclass_fields__
    ) | {
        "authority",
        "claims",
        "cryptographic_verification_performed_by_parser",
        "schema_version",
        "signature_algorithm",
        "signature_domain",
        "status",
        "verification_method",
        "verification_result",
    }
    item = _parse_body(
        raw,
        expected_file_sha256=expected_file_sha256,
        expected_body_sha256=expected_body_sha256,
        body_field=_PRODUCER_VERIFICATION_BODY_FIELD,
        body_keys=body_keys,
        label="storage producer signature verification receipt",
    )
    _pop_common_envelope(
        item,
        schema=(HOST_STORAGE_PRODUCER_INVENTORY_SIGNATURE_VERIFICATION_RECEIPT_V1_SCHEMA_VERSION),
        status="independent_signature_verifier_report_non_authorizing",
        label="storage producer signature verification receipt",
    )
    if (
        item.pop("signature_algorithm") != HOST_STORAGE_PRODUCER_SIGNATURE_ALGORITHM
        or item.pop("signature_domain") != HOST_STORAGE_PRODUCER_SIGNATURE_DOMAIN_LABEL
        or item.pop("verification_method") != "independently_pinned_ed25519_verifier"
        or item.pop("verification_result") != "verifier_reports_signature_valid"
        or item.pop("cryptographic_verification_performed_by_parser") is not False
    ):
        _fail("storage producer signature-verification report envelope differs")
    item["policy"] = _artifact_from_json(
        item["policy"],
        HOST_STORAGE_PRODUCER_TRUST_POLICY_V1_SCHEMA_VERSION,
        "producer verification policy",
    )
    item["statement"] = _artifact_from_json(
        item["statement"],
        HOST_STORAGE_PRODUCER_INVENTORY_STATEMENT_V1_SCHEMA_VERSION,
        "producer verification statement",
    )
    item["verifier"] = _pinned_host_component_from_json(item["verifier"])
    result = HostStorageProducerInventorySignatureVerificationReceiptV1(**item)
    if (
        raw
        != canonical_host_storage_producer_inventory_signature_verification_receipt_v1_file_bytes(
            result
        )
    ):
        _fail("storage producer signature verification canonical replay differs")
    return result


def parse_host_storage_producer_live_validation_receipt_v1(
    raw: bytes,
    *,
    expected_file_sha256: str,
    expected_body_sha256: str,
) -> HostStorageProducerLiveValidationReceiptV1:
    body_keys = frozenset(HostStorageProducerLiveValidationReceiptV1.__dataclass_fields__) | {
        "authority",
        "claims",
        "live_inspection_performed_by_parser",
        "observed_producer_inventory_sha256",
        "schema_version",
        "status",
        "validation_result",
    }
    item = _parse_body(
        raw,
        expected_file_sha256=expected_file_sha256,
        expected_body_sha256=expected_body_sha256,
        body_field=_PRODUCER_LIVE_BODY_FIELD,
        body_keys=body_keys,
        label="storage producer live validation receipt",
    )
    _pop_common_envelope(
        item,
        schema=HOST_STORAGE_PRODUCER_LIVE_VALIDATION_RECEIPT_V1_SCHEMA_VERSION,
        status="six_role_live_match_report_non_authorizing",
        label="storage producer live validation receipt",
    )
    if (
        item.pop("live_inspection_performed_by_parser") is not False
        or item.pop("validation_result") != "validator_reports_exact_six_role_inventory_match"
    ):
        _fail("storage producer live validation report envelope differs")
    supplied_inventory = item.pop("observed_producer_inventory_sha256")
    for field, schema in (
        ("policy", HOST_STORAGE_PRODUCER_TRUST_POLICY_V1_SCHEMA_VERSION),
        ("statement", HOST_STORAGE_PRODUCER_INVENTORY_STATEMENT_V1_SCHEMA_VERSION),
        (
            "signature_verification_receipt",
            HOST_STORAGE_PRODUCER_INVENTORY_SIGNATURE_VERIFICATION_RECEIPT_V1_SCHEMA_VERSION,
        ),
    ):
        item[field] = _artifact_from_json(item[field], schema, f"producer live {field}")
    previous = item["previous_live_validation_receipt"]
    item["previous_live_validation_receipt"] = (
        None
        if previous is None
        else _artifact_from_json(
            previous,
            HOST_STORAGE_PRODUCER_LIVE_VALIDATION_RECEIPT_V1_SCHEMA_VERSION,
            "producer previous live receipt",
        )
    )
    item["validator"] = _pinned_host_component_from_json(item["validator"])
    item["observed_producer_facts"] = _producer_facts_from_json(item["observed_producer_facts"])
    result = HostStorageProducerLiveValidationReceiptV1(**item)
    if supplied_inventory != result.observed_producer_inventory_sha256:
        _fail("storage producer live inventory identity differs")
    if raw != canonical_host_storage_producer_live_validation_receipt_v1_file_bytes(result):
        _fail("storage producer live validation canonical replay differs")
    return result


def parse_host_provisioning_validated_pre_go_prefix_v3(
    raw: bytes,
    *,
    expected_file_sha256: str,
    expected_body_sha256: str,
) -> HostProvisioningValidatedPreGoPrefixV3:
    body_keys = frozenset(HostProvisioningValidatedPreGoPrefixV3.__dataclass_fields__) | {
        "authority",
        "claims",
        "schema_version",
        "status",
    }
    item = _parse_body(
        raw,
        expected_file_sha256=expected_file_sha256,
        expected_body_sha256=expected_body_sha256,
        body_field=_PREFIX_BODY_FIELD,
        body_keys=body_keys,
        label="validated pre-GO prefix",
    )
    _pop_common_envelope(
        item,
        schema=HOST_PROVISIONING_VALIDATED_PRE_GO_PREFIX_V3_SCHEMA_VERSION,
        status="two_chain_pre_go_prefix_validated_non_authorizing",
        label="validated pre-GO prefix",
    )
    item["subject"] = _subject_from_json(item["subject"])
    schemas = {
        "qualification_plan": QUALIFICATION_PLAN_V3_SCHEMA_VERSION,
        "storage_backend_policy": STORAGE_BACKEND_POLICY_V1_SCHEMA_VERSION,
        "request": HOST_CASE_REQUEST_V3_SCHEMA_VERSION,
        "intent": HOST_CASE_INTENT_V3_SCHEMA_VERSION,
        "ready": HOST_READY_V3_SCHEMA_VERSION,
        "observer_anchor": HOST_OBSERVER_ANCHOR_V3_SCHEMA_VERSION,
        "host_provisioning_descriptor": (
            provisioning_v3.HOST_PROVISIONING_TRUST_CONTRACT_DESCRIPTOR_SCHEMA_VERSION
        ),
        "host_trust_policy": provisioning_v3.HOST_TRUST_POLICY_SCHEMA_VERSION,
        "host_provisioning_statement": (provisioning_v3.HOST_PROVISIONING_STATEMENT_SCHEMA_VERSION),
        "host_signature_verification_receipt": (
            provisioning_v3.HOST_SIGNATURE_VERIFICATION_RECEIPT_SCHEMA_VERSION
        ),
        "host_pre_capability_live_validation": (
            provisioning_v3.HOST_LIVE_VALIDATION_RECEIPT_SCHEMA_VERSION
        ),
        "host_pre_go_live_validation": (
            provisioning_v3.HOST_LIVE_VALIDATION_RECEIPT_SCHEMA_VERSION
        ),
        "storage_producer_trust_policy": (HOST_STORAGE_PRODUCER_TRUST_POLICY_V1_SCHEMA_VERSION),
        "storage_producer_inventory_statement": (
            HOST_STORAGE_PRODUCER_INVENTORY_STATEMENT_V1_SCHEMA_VERSION
        ),
        "storage_producer_signature_verification_receipt": (
            HOST_STORAGE_PRODUCER_INVENTORY_SIGNATURE_VERIFICATION_RECEIPT_V1_SCHEMA_VERSION
        ),
        "storage_producer_pre_capability_live_validation": (
            HOST_STORAGE_PRODUCER_LIVE_VALIDATION_RECEIPT_V1_SCHEMA_VERSION
        ),
        "storage_producer_pre_go_live_validation": (
            HOST_STORAGE_PRODUCER_LIVE_VALIDATION_RECEIPT_V1_SCHEMA_VERSION
        ),
    }
    for field, schema in schemas.items():
        item[field] = _artifact_from_json(item[field], schema, f"prefix {field}")
    item["storage_producers"] = _producers_from_json(item["storage_producers"])
    item["committed_host_phase_prefix"] = _phase_prefix_from_json(
        item["committed_host_phase_prefix"],
        "prefix committed phases",
    )
    result = HostProvisioningValidatedPreGoPrefixV3(**item)
    if raw != canonical_host_provisioning_validated_pre_go_prefix_v3_file_bytes(result):
        _fail("validated pre-GO prefix canonical replay differs")
    return result


def parse_host_case_request_v3(
    raw: bytes,
    *,
    expected_file_sha256: str,
    expected_body_sha256: str,
) -> HostQualificationCaseRequestV3:
    body_keys = frozenset(HostQualificationCaseRequestV3.__dataclass_fields__) | {
        "authority",
        "claims",
        "schema_version",
        "status",
    }
    item = _parse_body(
        raw,
        expected_file_sha256=expected_file_sha256,
        expected_body_sha256=expected_body_sha256,
        body_field=_REQUEST_BODY_FIELD,
        body_keys=body_keys,
        label="host case request v3",
    )
    _pop_common_envelope(
        item,
        schema=HOST_CASE_REQUEST_V3_SCHEMA_VERSION,
        status="phase0_additive_request_validated_non_authorizing",
        label="host case request v3",
    )
    item["subject"] = _subject_from_json(item["subject"])
    schemas = {
        "base_request_v2": HOST_CASE_REQUEST_V2_SCHEMA_VERSION,
        "qualification_plan": QUALIFICATION_PLAN_V3_SCHEMA_VERSION,
        "qualification_plan_descriptor": QUALIFICATION_PLAN_V3_DESCRIPTOR_SCHEMA_VERSION,
        "case_execution_ticket": CASE_EXECUTION_TICKET_SCHEMA_VERSION,
        "runtime_qualification_receipt": RUNTIME_QUALIFICATION_RECEIPT_SCHEMA_VERSION,
        "storage_backend_descriptor": STORAGE_BACKEND_CONTRACT_DESCRIPTOR_V2_SCHEMA_VERSION,
        "storage_backend_policy": STORAGE_BACKEND_POLICY_V1_SCHEMA_VERSION,
        "host_provisioning_descriptor": (
            provisioning_v3.HOST_PROVISIONING_TRUST_CONTRACT_DESCRIPTOR_SCHEMA_VERSION
        ),
        "host_executor_v2_descriptor": executor_v2.HOST_EXECUTOR_DESCRIPTOR_SCHEMA_VERSION,
        "host_protocol_descriptor": (HOST_QUALIFICATION_PROTOCOL_DESCRIPTOR_V3_SCHEMA_VERSION),
        "host_trust_policy": provisioning_v3.HOST_TRUST_POLICY_SCHEMA_VERSION,
        "host_provisioning_statement": (provisioning_v3.HOST_PROVISIONING_STATEMENT_SCHEMA_VERSION),
        "host_signature_verification_receipt": (
            provisioning_v3.HOST_SIGNATURE_VERIFICATION_RECEIPT_SCHEMA_VERSION
        ),
        "host_pre_capability_live_validation": (
            provisioning_v3.HOST_LIVE_VALIDATION_RECEIPT_SCHEMA_VERSION
        ),
        "storage_producer_trust_policy": (HOST_STORAGE_PRODUCER_TRUST_POLICY_V1_SCHEMA_VERSION),
        "storage_producer_inventory_statement": (
            HOST_STORAGE_PRODUCER_INVENTORY_STATEMENT_V1_SCHEMA_VERSION
        ),
        "storage_producer_signature_verification_receipt": (
            HOST_STORAGE_PRODUCER_INVENTORY_SIGNATURE_VERIFICATION_RECEIPT_V1_SCHEMA_VERSION
        ),
        "storage_producer_pre_capability_live_validation": (
            HOST_STORAGE_PRODUCER_LIVE_VALIDATION_RECEIPT_V1_SCHEMA_VERSION
        ),
    }
    for field, schema in schemas.items():
        item[field] = _artifact_from_json(item[field], schema, f"request {field}")
    item["storage_producers"] = _producers_from_json(item["storage_producers"])
    item["declared_ceilings"] = _declared_ceilings_from_json(item["declared_ceilings"])
    item["committed_host_phase_prefix"] = _phase_prefix_from_json(
        item["committed_host_phase_prefix"],
        "request committed phases",
    )
    result = HostQualificationCaseRequestV3(**item)
    if raw != canonical_host_case_request_v3_file_bytes(result):
        _fail("host case request v3 canonical replay differs")
    return result


def parse_host_case_intent_v3(
    raw: bytes,
    *,
    expected_file_sha256: str,
    expected_body_sha256: str,
) -> HostQualificationCaseIntentV3:
    body_keys = frozenset(HostQualificationCaseIntentV3.__dataclass_fields__) | {
        "authority",
        "claims",
        "schema_version",
        "status",
    }
    item = _parse_body(
        raw,
        expected_file_sha256=expected_file_sha256,
        expected_body_sha256=expected_body_sha256,
        body_field=_INTENT_BODY_FIELD,
        body_keys=body_keys,
        label="host case intent v3",
    )
    _pop_common_envelope(
        item,
        schema=HOST_CASE_INTENT_V3_SCHEMA_VERSION,
        status="phases1_2_additive_intent_committed_non_authorizing",
        label="host case intent v3",
    )
    item["subject"] = _subject_from_json(item["subject"])
    schemas = {
        "request": HOST_CASE_REQUEST_V3_SCHEMA_VERSION,
        "base_intent_v2": HOST_CASE_INTENT_V2_SCHEMA_VERSION,
        "qualification_plan": QUALIFICATION_PLAN_V3_SCHEMA_VERSION,
        "case_execution_ticket": CASE_EXECUTION_TICKET_SCHEMA_VERSION,
        "storage_backend_policy": STORAGE_BACKEND_POLICY_V1_SCHEMA_VERSION,
        "storage_producer_trust_policy": (HOST_STORAGE_PRODUCER_TRUST_POLICY_V1_SCHEMA_VERSION),
        "storage_producer_inventory_statement": (
            HOST_STORAGE_PRODUCER_INVENTORY_STATEMENT_V1_SCHEMA_VERSION
        ),
        "storage_producer_signature_verification_receipt": (
            HOST_STORAGE_PRODUCER_INVENTORY_SIGNATURE_VERIFICATION_RECEIPT_V1_SCHEMA_VERSION
        ),
        "storage_producer_pre_capability_live_validation": (
            HOST_STORAGE_PRODUCER_LIVE_VALIDATION_RECEIPT_V1_SCHEMA_VERSION
        ),
    }
    for field, schema in schemas.items():
        item[field] = _artifact_from_json(item[field], schema, f"intent {field}")
    item["storage_producers"] = _producers_from_json(item["storage_producers"])
    item["committed_host_phase_prefix"] = _phase_prefix_from_json(
        item["committed_host_phase_prefix"],
        "intent committed phases",
    )
    result = HostQualificationCaseIntentV3(**item)
    if raw != canonical_host_case_intent_v3_file_bytes(result):
        _fail("host case intent v3 canonical replay differs")
    return result


def parse_host_ready_v3(
    raw: bytes,
    *,
    expected_file_sha256: str,
    expected_body_sha256: str,
) -> HostReadyV3:
    body_keys = frozenset(HostReadyV3.__dataclass_fields__) | {
        "authority",
        "claims",
        "schema_version",
        "status",
    }
    item = _parse_body(
        raw,
        expected_file_sha256=expected_file_sha256,
        expected_body_sha256=expected_body_sha256,
        body_field=_READY_BODY_FIELD,
        body_keys=body_keys,
        label="host READY v3",
    )
    _pop_common_envelope(
        item,
        schema=HOST_READY_V3_SCHEMA_VERSION,
        status="phase8_additive_driver_ready_non_authorizing",
        label="host READY v3",
    )
    item["subject"] = _subject_from_json(item["subject"])
    for field, schema in {
        "intent": HOST_CASE_INTENT_V3_SCHEMA_VERSION,
        "base_initial_cgroup_sample_v2": HOST_INITIAL_CGROUP_SAMPLE_V2_SCHEMA_VERSION,
        "base_ready_v2": HOST_READY_V2_SCHEMA_VERSION,
        "qualification_plan": QUALIFICATION_PLAN_V3_SCHEMA_VERSION,
        "storage_backend_policy": STORAGE_BACKEND_POLICY_V1_SCHEMA_VERSION,
    }.items():
        item[field] = _artifact_from_json(item[field], schema, f"READY {field}")
    item["storage_producers"] = _producers_from_json(item["storage_producers"])
    item["committed_host_phase_prefix"] = _phase_prefix_from_json(
        item["committed_host_phase_prefix"],
        "READY committed phases",
    )
    result = HostReadyV3(**item)
    if raw != canonical_host_ready_v3_file_bytes(result):
        _fail("host READY v3 canonical replay differs")
    return result


def parse_host_observer_anchor_v3(
    raw: bytes,
    *,
    expected_file_sha256: str,
    expected_body_sha256: str,
) -> HostObserverAnchorV3:
    body_keys = frozenset(HostObserverAnchorV3.__dataclass_fields__) | {
        "authority",
        "claims",
        "schema_version",
        "status",
    }
    item = _parse_body(
        raw,
        expected_file_sha256=expected_file_sha256,
        expected_body_sha256=expected_body_sha256,
        body_field=_ANCHOR_BODY_FIELD,
        body_keys=body_keys,
        label="host observer anchor v3",
    )
    _pop_common_envelope(
        item,
        schema=HOST_OBSERVER_ANCHOR_V3_SCHEMA_VERSION,
        status="phase9_additive_observer_anchored_non_authorizing",
        label="host observer anchor v3",
    )
    item["subject"] = _subject_from_json(item["subject"])
    for field, schema in {
        "ready": HOST_READY_V3_SCHEMA_VERSION,
        "base_observer_anchor_v2": HOST_OBSERVER_ANCHOR_V2_SCHEMA_VERSION,
        "qualification_plan": QUALIFICATION_PLAN_V3_SCHEMA_VERSION,
        "storage_backend_policy": STORAGE_BACKEND_POLICY_V1_SCHEMA_VERSION,
    }.items():
        item[field] = _artifact_from_json(item[field], schema, f"anchor {field}")
    item["storage_producers"] = _producers_from_json(item["storage_producers"])
    item["membership_observer"] = _pinned_host_component_from_json(item["membership_observer"])
    item["committed_host_phase_prefix"] = _phase_prefix_from_json(
        item["committed_host_phase_prefix"],
        "anchor committed phases",
    )
    result = HostObserverAnchorV3(**item)
    if raw != canonical_host_observer_anchor_v3_file_bytes(result):
        _fail("host observer anchor v3 canonical replay differs")
    return result


def parse_host_go_v3(
    raw: bytes,
    *,
    expected_file_sha256: str,
    expected_body_sha256: str,
) -> HostGoCommitmentV3:
    body_keys = frozenset(HostGoCommitmentV3.__dataclass_fields__) | {
        "authority",
        "claims",
        "schema_version",
        "status",
    }
    item = _parse_body(
        raw,
        expected_file_sha256=expected_file_sha256,
        expected_body_sha256=expected_body_sha256,
        body_field=_GO_BODY_FIELD,
        body_keys=body_keys,
        label="host GO v3",
    )
    _pop_common_envelope(
        item,
        schema=HOST_GO_V3_SCHEMA_VERSION,
        status="phase10_additive_one_way_go_committed_non_authorizing",
        label="host GO v3",
    )
    item["subject"] = _subject_from_json(item["subject"])
    for field, schema in {
        "base_go_v2": HOST_GO_V2_SCHEMA_VERSION,
        "qualification_plan": QUALIFICATION_PLAN_V3_SCHEMA_VERSION,
        "storage_backend_policy": STORAGE_BACKEND_POLICY_V1_SCHEMA_VERSION,
        "ready": HOST_READY_V3_SCHEMA_VERSION,
        "observer_anchor": HOST_OBSERVER_ANCHOR_V3_SCHEMA_VERSION,
        "validated_pre_go_prefix": (HOST_PROVISIONING_VALIDATED_PRE_GO_PREFIX_V3_SCHEMA_VERSION),
        "storage_runtime_intent": STORAGE_BOUNDARY_RUNTIME_INTENT_V1_SCHEMA_VERSION,
    }.items():
        item[field] = _artifact_from_json(item[field], schema, f"GO {field}")
    item["storage_producers"] = _producers_from_json(item["storage_producers"])
    item["committed_host_phase_prefix"] = _phase_prefix_from_json(
        item["committed_host_phase_prefix"],
        "GO committed phases",
    )
    result = HostGoCommitmentV3(**item)
    if raw != canonical_host_go_v3_file_bytes(result):
        _fail("host GO v3 canonical replay differs")
    return result


__all__ = [
    "AUDITED_HOST_EXECUTOR_V2_DESCRIPTOR_BODY_SHA256",
    "AUDITED_HOST_EXECUTOR_V2_DESCRIPTOR_FILE_SHA256",
    "AUDITED_HOST_EXECUTOR_V2_SOURCE_SHA256",
    "AUDITED_HOST_PROVISIONING_V3_DESCRIPTOR_BODY_SHA256",
    "AUDITED_HOST_PROVISIONING_V3_DESCRIPTOR_FILE_SHA256",
    "AUDITED_HOST_PROVISIONING_V3_SOURCE_SHA256",
    "CANDIDATE_ORDER_SHA256",
    "ForagerMatchedV3HostQualificationProtocolV3Error",
    "HOST_ANCHOR_PHASE_PREFIX",
    "HOST_CASE_INTENT_V3_SCHEMA_VERSION",
    "HOST_CASE_REQUEST_V3_SCHEMA_VERSION",
    "HOST_GO_PHASE_ORDINAL",
    "HOST_GO_PHASE_PREFIX",
    "HOST_GO_V3_SCHEMA_VERSION",
    "HOST_INTENT_PHASE_PREFIX",
    "HOST_OBSERVER_ANCHOR_V3_SCHEMA_VERSION",
    "HOST_OPERATIONAL_PHASES_V3",
    "HOST_PROVISIONING_VALIDATED_PRE_GO_PREFIX_V3_SCHEMA_VERSION",
    "HOST_QUALIFICATION_PROTOCOL_DESCRIPTOR_V3_SCHEMA_VERSION",
    "HOST_READY_PHASE_PREFIX",
    "HOST_READY_V3_SCHEMA_VERSION",
    "HOST_REQUEST_PHASE_PREFIX",
    "HOST_STORAGE_PRODUCER_INVENTORY_SIGNATURE_VERIFICATION_RECEIPT_V1_SCHEMA_VERSION",
    "HOST_STORAGE_PRODUCER_INVENTORY_STATEMENT_V1_SCHEMA_VERSION",
    "HOST_STORAGE_PRODUCER_LIVE_VALIDATION_RECEIPT_V1_SCHEMA_VERSION",
    "HOST_STORAGE_PRODUCER_SIGNATURE_ALGORITHM",
    "HOST_STORAGE_PRODUCER_SIGNATURE_DOMAIN",
    "HOST_STORAGE_PRODUCER_SIGNATURE_DOMAIN_LABEL",
    "HOST_STORAGE_PRODUCER_TRUST_POLICY_V1_SCHEMA_VERSION",
    "HOST_STORAGE_RECEIPT_PHASE_ORDINAL",
    "HOST_WRITE_SEAL_PHASE_ORDINAL",
    "HostGoCommitmentV3",
    "HostGoWriteSealEndpointProjectionV1",
    "HostObserverAnchorV3",
    "HostProvisioningPreGoBundleV3",
    "HostProvisioningValidatedPreGoPrefixV3",
    "HostQualificationCaseIntentV3",
    "HostQualificationCaseRequestV3",
    "HostQualificationPreGoBundleV3",
    "HostQualificationProtocolDescriptorV3",
    "HostReadyV3",
    "HostStorageProducerInventorySignatureVerificationReceiptV1",
    "HostStorageProducerInventoryStatementV1",
    "HostStorageProducerLiveValidationReceiptV1",
    "HostStorageProducerPreGoBundleV1",
    "HostStorageProducerRuntimeFactV1",
    "HostStorageProducerTrustPolicyV1",
    "PinnedHostComponentRefV1",
    "RESOURCE_FIELD_ORDER_SHA256",
    "STORAGE_BOUNDARY_RUNTIME_INTENT_V1_SCHEMA_VERSION",
    "canonical_host_case_intent_v3_body_bytes",
    "canonical_host_case_intent_v3_file_bytes",
    "canonical_host_case_request_v3_body_bytes",
    "canonical_host_case_request_v3_file_bytes",
    "canonical_host_go_v3_body_bytes",
    "canonical_host_go_v3_file_bytes",
    "canonical_host_observer_anchor_v3_body_bytes",
    "canonical_host_observer_anchor_v3_file_bytes",
    "canonical_host_provisioning_validated_pre_go_prefix_v3_body_bytes",
    "canonical_host_provisioning_validated_pre_go_prefix_v3_file_bytes",
    "canonical_host_qualification_protocol_descriptor_v3_body_bytes",
    "canonical_host_qualification_protocol_descriptor_v3_file_bytes",
    "canonical_host_ready_v3_body_bytes",
    "canonical_host_ready_v3_file_bytes",
    "canonical_host_storage_producer_inventory_signature_verification_receipt_v1_body_bytes",
    "canonical_host_storage_producer_inventory_signature_verification_receipt_v1_file_bytes",
    "canonical_host_storage_producer_inventory_statement_v1_body_bytes",
    "canonical_host_storage_producer_inventory_statement_v1_file_bytes",
    "canonical_host_storage_producer_inventory_statement_v1_signed_payload_bytes",
    "canonical_host_storage_producer_live_validation_receipt_v1_body_bytes",
    "canonical_host_storage_producer_live_validation_receipt_v1_file_bytes",
    "canonical_host_storage_producer_trust_policy_v1_body_bytes",
    "canonical_host_storage_producer_trust_policy_v1_file_bytes",
    "host_case_intent_v3_identity",
    "host_case_request_v3_identity",
    "host_go_payload_sha256_v3",
    "host_go_v3_identity",
    "host_observer_anchor_v3_identity",
    "host_provisioning_validated_pre_go_prefix_v3_identity",
    "host_qualification_protocol_descriptor_v3_identity",
    "host_ready_v3_identity",
    "host_storage_producer_inventory_signature_verification_receipt_v1_identity",
    "host_storage_producer_inventory_statement_v1_identity",
    "host_storage_producer_live_validation_receipt_v1_identity",
    "host_storage_producer_trust_policy_v1_identity",
    "parse_host_case_intent_v3",
    "parse_host_case_request_v3",
    "parse_host_go_v3",
    "parse_host_observer_anchor_v3",
    "parse_host_provisioning_validated_pre_go_prefix_v3",
    "parse_host_qualification_protocol_descriptor_v3",
    "parse_host_ready_v3",
    "parse_host_storage_producer_inventory_signature_verification_receipt_v1",
    "parse_host_storage_producer_inventory_statement_v1",
    "parse_host_storage_producer_live_validation_receipt_v1",
    "parse_host_storage_producer_trust_policy_v1",
    "repository_protocol_ready_v3",
    "validate_host_pre_go_prefix_v3",
    "validate_host_qualification_v3_chain",
    "validate_host_ready_anchor_go_v3_chain",
    "validate_host_request_intent_v3_chain",
    "validate_host_storage_producer_pre_go_bundle_v1",
    "validate_repository_protocol_pins_v3",
]
