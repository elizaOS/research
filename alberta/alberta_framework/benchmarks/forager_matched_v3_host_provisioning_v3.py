"""Pure source-only trust contract for matched-v3 host provisioning v3.

The four artifacts in this module describe, but do not establish, a future
production host trust chain:

* a preissued trust policy pins one host-facts inventory, signer public-key
  identity, independent verifier, live validator, and supported host tuple;
* a provisioning statement carries a detached, domain-separated Ed25519
  signature over canonical statement bytes;
* a verification receipt records what an independently pinned verifier
  reported; and
* four live-validation receipts record exact facts at ``pre_capability``,
  ``pre_go``, ``post_workload``, and ``post_cleanup``.

This module deliberately implements no Ed25519 operation and holds no signing
secret.  Parsing well-formed signature metadata is not cryptographic
verification.  It also contains no filesystem, process, socket, Docker, OCI,
cgroup, issuer, workload, evaluator, or qualification operation.  Every
parser requires a nonzero caller-supplied full-file SHA-256.
"""

from __future__ import annotations

import hashlib
import json
import re
import secrets
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final, Never, NoReturn, Protocol, cast

__all__ = (
    "HOST_PROVISIONING_TRUST_CONTRACT_DESCRIPTOR_SCHEMA_VERSION",
    "HOST_TRUST_POLICY_SCHEMA_VERSION",
    "HOST_PROVISIONING_STATEMENT_SCHEMA_VERSION",
    "HOST_SIGNATURE_VERIFICATION_RECEIPT_SCHEMA_VERSION",
    "HOST_LIVE_VALIDATION_RECEIPT_SCHEMA_VERSION",
    "QUALIFICATION_PLAN_V3_SCHEMA_VERSION",
    "ED25519_SIGNATURE_ALGORITHM",
    "ED25519_SIGNATURE_DOMAIN_LABEL",
    "ED25519_SIGNATURE_DOMAIN",
    "LIVE_VALIDATION_CHECKPOINTS",
    "REQUIRED_RETAINED_RESOURCE_EVENT_COUNTERS",
    "RESOURCE_EVENT_COUNTER_MONOTONICITY_POLICY",
    "POSITIVE_RESOURCE_EVENT_DELTA_POLICY",
    "PINNED_HOST_PROVISIONING_V3_DESCRIPTOR_FILE_SHA256",
    "SOURCE_ONLY_CAPABILITIES",
    "SOURCE_ONLY_READINESS",
    "SOURCE_ONLY_AUTHORITY",
    "SOURCE_ONLY_CLAIMS",
    "SOURCE_ONLY_SAFETY_POSTURE",
    "ForagerMatchedV3HostProvisioningV3Error",
    "ArtifactIdentityV1",
    "PinnedComponentIdentityV1",
    "HostKernelIdentityV1",
    "CgroupV2IdentityV1",
    "DockerDaemonIdentityV1",
    "HostComponentInventoryV1",
    "HostFactsInventoryV1",
    "SupportedHostTupleV1",
    "HostProvisioningTrustContractDescriptorV1",
    "HostTrustPolicyV1",
    "HostProvisioningStatementV1",
    "HostSignatureVerificationReceiptV1",
    "HostLiveValidationReceiptV1",
    "canonical_provisioning_json_bytes",
    "decode_canonical_provisioning_json",
    "canonical_host_provisioning_trust_contract_descriptor_v1_body_bytes",
    "canonical_host_provisioning_trust_contract_descriptor_v1_file_bytes",
    "canonical_host_trust_policy_v1_body_bytes",
    "canonical_host_trust_policy_v1_file_bytes",
    "canonical_host_provisioning_statement_v1_signed_payload_bytes",
    "canonical_host_provisioning_statement_v1_body_bytes",
    "canonical_host_provisioning_statement_v1_file_bytes",
    "canonical_host_signature_verification_receipt_v1_body_bytes",
    "canonical_host_signature_verification_receipt_v1_file_bytes",
    "canonical_host_live_validation_receipt_v1_body_bytes",
    "canonical_host_live_validation_receipt_v1_file_bytes",
    "host_provisioning_v3_descriptor_sha256",
    "host_trust_policy_identity_v1",
    "host_provisioning_statement_identity_v1",
    "host_signature_verification_receipt_identity_v1",
    "host_live_validation_receipt_identity_v1",
    "parse_host_provisioning_trust_contract_descriptor_v1",
    "parse_host_trust_policy_v1",
    "parse_host_provisioning_statement_v1",
    "parse_host_signature_verification_receipt_v1",
    "parse_host_live_validation_receipt_v1",
    "validate_host_provisioning_statement_against_policy_v1",
    "validate_host_signature_verification_receipt_v1",
    "validate_host_live_validation_receipt_v1",
    "validate_host_provisioning_trust_chain_v1",
)

HOST_PROVISIONING_TRUST_CONTRACT_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.host_provisioning_trust_contract_descriptor.v1"
)
HOST_TRUST_POLICY_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.host_trust_policy.v1"
)
HOST_PROVISIONING_STATEMENT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.host_provisioning_statement.v1"
)
HOST_SIGNATURE_VERIFICATION_RECEIPT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.host_provisioning_signature_verification_receipt.v1"
)
HOST_LIVE_VALIDATION_RECEIPT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.host_provisioning_live_validation_receipt.v1"
)
QUALIFICATION_PLAN_V3_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.qualification_plan.v3"
)

ED25519_SIGNATURE_ALGORITHM: Final = "ed25519"
ED25519_SIGNATURE_DOMAIN_LABEL: Final = (
    "alberta.forager_matched_v3.host_provisioning_statement.v1"
)
ED25519_SIGNATURE_DOMAIN: Final = ED25519_SIGNATURE_DOMAIN_LABEL.encode("ascii") + b"\x00"
LIVE_VALIDATION_CHECKPOINTS: Final = (
    "pre_capability",
    "pre_go",
    "post_workload",
    "post_cleanup",
)
REQUIRED_RETAINED_RESOURCE_EVENT_COUNTERS: Final = (
    "memory.events:oom",
    "memory.events:oom_kill",
    "pids.events:max",
)
RESOURCE_EVENT_COUNTER_MONOTONICITY_POLICY: Final = (
    "retained_initial_and_terminal_unsigned_counters_must_be_monotonic"
)
POSITIVE_RESOURCE_EVENT_DELTA_POLICY: Final = (
    "any_positive_delta_makes_case_resource_ineligible_even_when_worker_exit_zero"
)

PINNED_HOST_PROVISIONING_V3_DESCRIPTOR_FILE_SHA256: Final = (
    "1ff3b76662504333749529926120c0f9a49dfd7aa010f5fc5951282feed4cf56"
)

_DESCRIPTOR_BODY_FIELD: Final = "host_provisioning_trust_contract_descriptor_body_sha256"
_POLICY_BODY_FIELD: Final = "host_trust_policy_body_sha256"
_STATEMENT_BODY_FIELD: Final = "host_provisioning_statement_body_sha256"
_VERIFICATION_BODY_FIELD: Final = "host_signature_verification_receipt_body_sha256"
_LIVE_BODY_FIELD: Final = "host_live_validation_receipt_body_sha256"

_POLICY_STATUS: Final = "preissued_source_only_non_authorizing_host_trust_policy"
_STATEMENT_STATUS: Final = "signed_metadata_unverified_by_this_parser_non_authorizing"
_VERIFICATION_STATUS: Final = (
    "independent_verifier_report_recorded_unendorsed_by_this_parser_non_authorizing"
)
_LIVE_STATUS: Final = "live_fact_comparison_recorded_non_authorizing"
_VERIFICATION_METHOD: Final = "independently_pinned_ed25519_verifier"
_VERIFICATION_RESULT: Final = "verifier_reports_ed25519_signature_valid"
_LIVE_RESULT: Final = "validator_reports_exact_match_no_live_drift"

_MAX_ARTIFACT_BYTES: Final = 4 * 1024 * 1024
_MAX_JSON_DEPTH: Final = 64
_MAX_JSON_NODES: Final = 100_000
_MAX_TEXT_LENGTH: Final = 16_384
_MAX_INTEGER: Final = 2**63 - 1
_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_SIGNATURE_RE: Final = re.compile(r"[0-9a-f]{128}\Z")
_IDENTIFIER_RE: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}\Z")
_BOOT_ID_RE: Final = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\Z"
)
_CGROUP_FILESYSTEM_MAGIC: Final = "0x63677270"
_CGROUP_MOUNT_PATH: Final = "/sys/fs/cgroup"
_CGROUP_DELEGATE_PATH: Final = "/sys/fs/cgroup/alberta-qualified-host"
_REQUIRED_CONTROLLERS: Final = ("cpu", "memory", "pids")

SOURCE_ONLY_CAPABILITIES: Final[Mapping[str, bool]] = MappingProxyType(
    {
        "accesses_signing_secret": False,
        "executes_host_operations": False,
        "inspects_live_host": False,
        "issues_execution_capability": False,
        "mutates_filesystem": False,
        "verifies_ed25519_signatures": False,
    }
)
SOURCE_ONLY_READINESS: Final[Mapping[str, bool]] = MappingProxyType(
    {
        "descriptor_pinned": False,
        "host_qualified": False,
        "production_ready": False,
        "verifier_audited": False,
    }
)
SOURCE_ONLY_AUTHORITY: Final[Mapping[str, bool]] = MappingProxyType(
    {
        "evidence_authority_granted": False,
        "execution_authority_granted": False,
        "issuance_authority_granted": False,
        "promotion_authority_granted": False,
        "qualification_authority_granted": False,
    }
)
SOURCE_ONLY_CLAIMS: Final[Mapping[str, bool]] = MappingProxyType(
    {
        "candidate_qualified": False,
        "host_facts_true_by_parsing": False,
        "performance_claim_allowed": False,
        "scientific_evidence_created": False,
        "signature_cryptographically_verified_by_parser": False,
        "universal_sota_claim_allowed": False,
    }
)
SOURCE_ONLY_SAFETY_POSTURE: Final[Mapping[str, Mapping[str, bool]]] = MappingProxyType(
    {
        "capabilities": SOURCE_ONLY_CAPABILITIES,
        "readiness": SOURCE_ONLY_READINESS,
        "authority": SOURCE_ONLY_AUTHORITY,
        "claims": SOURCE_ONLY_CLAIMS,
    }
)


class ForagerMatchedV3HostProvisioningV3Error(ValueError):
    """A host-provisioning v3 canonical artifact or chain failed closed."""


class _BodyArtifact(Protocol):
    def to_body_dict(self) -> dict[str, Any]: ...


def _fail(message: str) -> NoReturn:
    raise ForagerMatchedV3HostProvisioningV3Error(message)


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
        _fail(f"{label} is outside its exact integer bounds")
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


def _require_sha256(value: object, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None or value == "0" * 64:
        _fail(f"{label} must be one nonzero lowercase SHA-256")
    return value


def _require_boot_id(value: object, label: str) -> str:
    if type(value) is not str or _BOOT_ID_RE.fullmatch(value) is None:
        _fail(f"{label} must be one canonical lowercase UUID")
    return value


def _require_signature(value: object, label: str) -> str:
    if type(value) is not str or _SIGNATURE_RE.fullmatch(value) is None or value == "00" * 64:
        _fail(f"{label} must be one nonzero 64-byte lowercase-hex Ed25519 signature")
    return value


def _require_absolute_path(value: object, label: str) -> str:
    exact = _require_text(value, label, maximum=4096)
    if not exact.startswith("/") or exact == "/" or "//" in exact:
        _fail(f"{label} must be one normalized non-root absolute path")
    if any(part in {"", ".", ".."} for part in exact.split("/")[1:]):
        _fail(f"{label} must not contain empty, dot, or parent segments")
    return exact


def _require_exact_tuple(value: object, expected: tuple[str, ...], label: str) -> tuple[str, ...]:
    if type(value) is not tuple or value != expected:
        _fail(f"{label} differs from the exact ordered tuple")
    return cast(tuple[str, ...], value)


def _require_exact_type(value: object, expected: type[Any], label: str) -> None:
    if type(value) is not expected:
        _fail(f"{label} must use exact type {expected.__name__}")


def _reject_constant(value: str) -> Never:
    _fail(f"canonical provisioning JSON contains forbidden constant {value!r}")


def _reject_float(value: str) -> Never:
    _fail(f"canonical provisioning JSON contains forbidden float {value!r}")


def _parse_int(value: str) -> int:
    if len(value.lstrip("-")) > 19:
        _fail("canonical provisioning JSON integer exceeds its lexical bound")
    parsed = int(value)
    if abs(parsed) > _MAX_INTEGER:
        _fail("canonical provisioning JSON integer exceeds its value bound")
    return parsed


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if type(key) is not str or key in result:
            _fail("canonical provisioning JSON contains a duplicate or non-text key")
        result[key] = value
    return result


def _assert_plain_unaliased_json(value: object) -> None:
    pending: list[tuple[object, int]] = [(value, 0)]
    containers: set[int] = set()
    nodes = 0
    while pending:
        item, depth = pending.pop()
        nodes += 1
        if nodes > _MAX_JSON_NODES or depth > _MAX_JSON_DEPTH:
            _fail("canonical provisioning JSON exceeds its structural bound")
        if item is None or type(item) is bool:
            continue
        if type(item) is int:
            _require_int(item, "canonical JSON integer", minimum=-_MAX_INTEGER)
            continue
        if type(item) is str:
            _require_text(item, "canonical JSON text")
            continue
        if type(item) not in {dict, list}:
            _fail("canonical provisioning JSON contains a non-plain value")
        identity = id(item)
        if identity in containers:
            _fail("canonical provisioning JSON contains an alias or cycle")
        containers.add(identity)
        if type(item) is list:
            pending.extend((child, depth + 1) for child in cast(list[object], item))
        else:
            mapping = cast(dict[object, object], item)
            for key, child in mapping.items():
                if type(key) is not str:
                    _fail("canonical provisioning JSON key is not one exact string")
                _require_text(key, "canonical JSON key")
                pending.append((child, depth + 1))


def canonical_provisioning_json_bytes(value: object, *, trailing_lf: bool) -> bytes:
    """Encode plain unaliased data as bounded canonical ASCII JSON."""

    _require_bool(trailing_lf, "canonical JSON trailing-LF selection")
    _assert_plain_unaliased_json(value)
    try:
        raw = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError, RecursionError) as exc:
        raise ForagerMatchedV3HostProvisioningV3Error(
            "value is not canonical ASCII provisioning JSON"
        ) from exc
    if trailing_lf:
        raw += b"\n"
    if len(raw) > _MAX_ARTIFACT_BYTES:
        _fail("canonical provisioning JSON exceeds its byte bound")
    return raw


def decode_canonical_provisioning_json(raw: bytes) -> dict[str, Any]:
    """Decode one canonical ASCII object with exactly one trailing LF."""

    if type(raw) is not bytes or not raw or len(raw) > _MAX_ARTIFACT_BYTES:
        _fail("provisioning artifact bytes violate their bound")
    try:
        value = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_object_without_duplicates,
            parse_float=_reject_float,
            parse_constant=_reject_constant,
            parse_int=_parse_int,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ForagerMatchedV3HostProvisioningV3Error(
            "provisioning artifact bytes are not strict ASCII JSON"
        ) from exc
    if type(value) is not dict:
        _fail("provisioning artifact JSON root must be one object")
    exact = cast(dict[str, Any], value)
    _assert_plain_unaliased_json(exact)
    if raw != canonical_provisioning_json_bytes(exact, trailing_lf=True):
        _fail("provisioning artifact is not canonical one-LF JSON")
    return exact


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _body_sha256(body: Mapping[str, Any]) -> str:
    return _sha256(canonical_provisioning_json_bytes(dict(body), trailing_lf=False))


def _file_dict(body: Mapping[str, Any], body_field: str) -> dict[str, Any]:
    return {**dict(body), body_field: _body_sha256(body)}


def _require_exact_keys(value: object, expected: frozenset[str], label: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != expected:
        _fail(f"{label} keys differ from the exact schema")
    return dict(cast(dict[str, Any], value))


def _parse_artifact_file(
    raw: bytes,
    *,
    expected_file_sha256: object,
    body_field: str,
    body_keys: frozenset[str],
    label: str,
) -> dict[str, Any]:
    expected = _require_sha256(expected_file_sha256, f"{label} caller FILE pin")
    if type(raw) is not bytes or not raw or len(raw) > _MAX_ARTIFACT_BYTES:
        _fail(f"{label} artifact bytes violate their bound")
    if not secrets.compare_digest(_sha256(raw), expected):
        _fail(f"{label} full-file SHA-256 differs")
    item = decode_canonical_provisioning_json(raw)
    _require_exact_keys(item, body_keys | {body_field}, label)
    supplied = _require_sha256(item[body_field], f"{label} BODY digest")
    body = {key: value for key, value in item.items() if key != body_field}
    if not secrets.compare_digest(supplied, _body_sha256(body)):
        _fail(f"{label} BODY digest differs")
    return body


def _canonical_artifact_body_bytes(value: object, expected_type: type[Any]) -> bytes:
    if type(value) is not expected_type:
        raise TypeError(f"artifact must use exact type {expected_type.__name__}")
    body = cast(_BodyArtifact, value).to_body_dict()
    return canonical_provisioning_json_bytes(body, trailing_lf=False)


def _canonical_artifact_file_bytes(
    value: object,
    expected_type: type[Any],
    body_field: str,
) -> bytes:
    if type(value) is not expected_type:
        raise TypeError(f"artifact must use exact type {expected_type.__name__}")
    body = cast(_BodyArtifact, value).to_body_dict()
    return canonical_provisioning_json_bytes(_file_dict(body, body_field), trailing_lf=True)


def _safety_posture_dict() -> dict[str, dict[str, bool]]:
    return {
        role: dict(values)
        for role, values in SOURCE_ONLY_SAFETY_POSTURE.items()
    }


def _parse_false_map(
    value: object,
    expected: Mapping[str, bool],
    label: str,
) -> None:
    item = _require_exact_keys(value, frozenset(expected), label)
    for key in expected:
        if _require_bool(item[key], f"{label}.{key}") is not False:
            _fail(f"{label}.{key} must remain false")


def _parse_safety_posture(value: object, label: str) -> None:
    item = _require_exact_keys(
        value,
        frozenset({"capabilities", "readiness", "authority", "claims"}),
        label,
    )
    _parse_false_map(item["capabilities"], SOURCE_ONLY_CAPABILITIES, label + ".capabilities")
    _parse_false_map(item["readiness"], SOURCE_ONLY_READINESS, label + ".readiness")
    _parse_false_map(item["authority"], SOURCE_ONLY_AUTHORITY, label + ".authority")
    _parse_false_map(item["claims"], SOURCE_ONLY_CLAIMS, label + ".claims")


@dataclass(frozen=True, slots=True)
class ArtifactIdentityV1:
    """Schema, full-file, and BODY identity for one canonical artifact."""

    schema_version: str
    file_sha256: str
    body_sha256: str

    def __post_init__(self) -> None:
        _require_identifier(self.schema_version, "artifact schema version")
        _require_sha256(self.file_sha256, "artifact FILE SHA-256")
        _require_sha256(self.body_sha256, "artifact BODY SHA-256")

    def to_dict(self) -> dict[str, str]:
        return {
            "schema_version": self.schema_version,
            "file_sha256": self.file_sha256,
            "body_sha256": self.body_sha256,
        }


@dataclass(frozen=True, slots=True)
class PinnedComponentIdentityV1:
    """Exact descriptor, source, and runtime-artifact identity of one role."""

    component_id: str
    descriptor_schema_version: str
    descriptor_file_sha256: str
    source_sha256: str
    runtime_artifact_sha256: str

    def __post_init__(self) -> None:
        _require_identifier(self.component_id, "component ID")
        _require_identifier(self.descriptor_schema_version, "component descriptor schema")
        _require_sha256(self.descriptor_file_sha256, "component descriptor FILE SHA-256")
        _require_sha256(self.source_sha256, "component source SHA-256")
        _require_sha256(self.runtime_artifact_sha256, "component runtime artifact SHA-256")

    def to_dict(self) -> dict[str, str]:
        return {
            "component_id": self.component_id,
            "descriptor_schema_version": self.descriptor_schema_version,
            "descriptor_file_sha256": self.descriptor_file_sha256,
            "source_sha256": self.source_sha256,
            "runtime_artifact_sha256": self.runtime_artifact_sha256,
        }


@dataclass(frozen=True, slots=True)
class HostKernelIdentityV1:
    """Stable host, boot, architecture, and kernel facts."""

    host_identity_sha256: str
    machine_id_sha256: str
    boot_id: str
    architecture: str
    kernel_release: str
    kernel_build_sha256: str
    kernel_command_line_sha256: str

    def __post_init__(self) -> None:
        _require_sha256(self.host_identity_sha256, "host identity SHA-256")
        _require_sha256(self.machine_id_sha256, "machine-id SHA-256")
        _require_boot_id(self.boot_id, "boot ID")
        _require_identifier(self.architecture, "host architecture")
        _require_text(self.kernel_release, "kernel release", maximum=256)
        _require_sha256(self.kernel_build_sha256, "kernel build SHA-256")
        _require_sha256(self.kernel_command_line_sha256, "kernel command-line SHA-256")

    def to_dict(self) -> dict[str, Any]:
        return {
            "host_identity_sha256": self.host_identity_sha256,
            "machine_id_sha256": self.machine_id_sha256,
            "boot_id": self.boot_id,
            "architecture": self.architecture,
            "kernel_release": self.kernel_release,
            "kernel_build_sha256": self.kernel_build_sha256,
            "kernel_command_line_sha256": self.kernel_command_line_sha256,
        }


@dataclass(frozen=True, slots=True)
class CgroupV2IdentityV1:
    """Unified cgroup-v2 mount and exact delegated-root facts."""

    mount_path: str
    mount_device_major: int
    mount_device_minor: int
    mount_inode: int
    filesystem_magic: str
    unified_hierarchy: bool
    delegate_path: str
    delegate_device_major: int
    delegate_device_minor: int
    delegate_inode: int
    delegate_uid: int
    delegate_gid: int
    delegate_mode: int
    delegated_controllers: tuple[str, ...]
    subtree_control: tuple[str, ...]

    def __post_init__(self) -> None:
        if _require_absolute_path(self.mount_path, "cgroup mount path") != _CGROUP_MOUNT_PATH:
            _fail("cgroup mount path differs from the unified host mount")
        _require_int(self.mount_device_major, "cgroup mount device major")
        _require_int(self.mount_device_minor, "cgroup mount device minor")
        _require_int(self.mount_inode, "cgroup mount inode", minimum=1)
        if (
            _require_text(self.filesystem_magic, "cgroup filesystem magic")
            != _CGROUP_FILESYSTEM_MAGIC
        ):
            _fail("cgroup filesystem magic is not cgroup v2")
        _require_bool(self.unified_hierarchy, "unified cgroup hierarchy", expected=True)
        if (
            _require_absolute_path(self.delegate_path, "cgroup delegate path")
            != _CGROUP_DELEGATE_PATH
        ):
            _fail("cgroup delegate path differs from the dedicated qualified-host root")
        _require_int(self.delegate_device_major, "cgroup delegate device major")
        _require_int(self.delegate_device_minor, "cgroup delegate device minor")
        _require_int(self.delegate_inode, "cgroup delegate inode", minimum=1)
        if (
            self.delegate_device_major != self.mount_device_major
            or self.delegate_device_minor != self.mount_device_minor
        ):
            _fail("cgroup mount and delegate must share one filesystem device")
        _require_int(self.delegate_uid, "cgroup delegate UID")
        _require_int(self.delegate_gid, "cgroup delegate GID")
        _require_int(self.delegate_mode, "cgroup delegate mode", maximum=0o7777)
        _require_exact_tuple(
            self.delegated_controllers,
            _REQUIRED_CONTROLLERS,
            "delegated cgroup controllers",
        )
        _require_exact_tuple(
            self.subtree_control,
            _REQUIRED_CONTROLLERS,
            "cgroup subtree control",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "mount_path": self.mount_path,
            "mount_device_major": self.mount_device_major,
            "mount_device_minor": self.mount_device_minor,
            "mount_inode": self.mount_inode,
            "filesystem_magic": self.filesystem_magic,
            "unified_hierarchy": self.unified_hierarchy,
            "delegate_path": self.delegate_path,
            "delegate_device_major": self.delegate_device_major,
            "delegate_device_minor": self.delegate_device_minor,
            "delegate_inode": self.delegate_inode,
            "delegate_uid": self.delegate_uid,
            "delegate_gid": self.delegate_gid,
            "delegate_mode": self.delegate_mode,
            "delegated_controllers": list(self.delegated_controllers),
            "subtree_control": list(self.subtree_control),
        }


@dataclass(frozen=True, slots=True)
class DockerDaemonIdentityV1:
    """Rootful Docker socket, daemon lifetime, configuration, and root facts."""

    socket_path: str
    socket_device_major: int
    socket_device_minor: int
    socket_inode: int
    socket_uid: int
    socket_gid: int
    socket_mode: int
    daemon_id: str
    daemon_pid: int
    daemon_start_ticks: int
    rootful: bool
    cgroup_driver: str
    version: str
    api_version: str
    config_sha256: str
    root_dir_path: str
    root_dir_device_major: int
    root_dir_device_minor: int
    root_dir_inode: int

    def __post_init__(self) -> None:
        _require_absolute_path(self.socket_path, "Docker socket path")
        _require_int(self.socket_device_major, "Docker socket device major")
        _require_int(self.socket_device_minor, "Docker socket device minor")
        _require_int(self.socket_inode, "Docker socket inode", minimum=1)
        _require_int(self.socket_uid, "Docker socket UID")
        _require_int(self.socket_gid, "Docker socket GID")
        _require_int(self.socket_mode, "Docker socket mode", maximum=0o7777)
        _require_identifier(self.daemon_id, "Docker daemon ID")
        _require_int(self.daemon_pid, "Docker daemon PID", minimum=1)
        _require_int(self.daemon_start_ticks, "Docker daemon start ticks", minimum=1)
        _require_bool(self.rootful, "Docker rootful fact", expected=True)
        if _require_text(self.cgroup_driver, "Docker cgroup driver") != "cgroupfs":
            _fail("Docker cgroup driver must be exactly cgroupfs")
        _require_text(self.version, "Docker version", maximum=256)
        _require_text(self.api_version, "Docker API version", maximum=256)
        _require_sha256(self.config_sha256, "Docker config SHA-256")
        _require_absolute_path(self.root_dir_path, "Docker root-dir path")
        _require_int(self.root_dir_device_major, "Docker root-dir device major")
        _require_int(self.root_dir_device_minor, "Docker root-dir device minor")
        _require_int(self.root_dir_inode, "Docker root-dir inode", minimum=1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "socket_path": self.socket_path,
            "socket_device_major": self.socket_device_major,
            "socket_device_minor": self.socket_device_minor,
            "socket_inode": self.socket_inode,
            "socket_uid": self.socket_uid,
            "socket_gid": self.socket_gid,
            "socket_mode": self.socket_mode,
            "daemon_id": self.daemon_id,
            "daemon_pid": self.daemon_pid,
            "daemon_start_ticks": self.daemon_start_ticks,
            "rootful": self.rootful,
            "cgroup_driver": self.cgroup_driver,
            "version": self.version,
            "api_version": self.api_version,
            "config_sha256": self.config_sha256,
            "root_dir_path": self.root_dir_path,
            "root_dir_device_major": self.root_dir_device_major,
            "root_dir_device_minor": self.root_dir_device_minor,
            "root_dir_inode": self.root_dir_inode,
        }


@dataclass(frozen=True, slots=True)
class HostComponentInventoryV1:
    """Pinned operational components whose identities are live host facts."""

    oci_runtime: PinnedComponentIdentityV1
    membership_observer: PinnedComponentIdentityV1
    storage_measurement_producer: PinnedComponentIdentityV1
    storage_terminal_relay: PinnedComponentIdentityV1
    security_profile: PinnedComponentIdentityV1

    def __post_init__(self) -> None:
        values = (
            self.oci_runtime,
            self.membership_observer,
            self.storage_measurement_producer,
            self.storage_terminal_relay,
            self.security_profile,
        )
        for index, value in enumerate(values):
            _require_exact_type(
                value,
                PinnedComponentIdentityV1,
                f"component inventory role {index}",
            )
        if len({value.component_id for value in values}) != len(values):
            _fail("host component roles must have distinct component IDs")

    def to_dict(self) -> dict[str, Any]:
        return {
            "oci_runtime": self.oci_runtime.to_dict(),
            "membership_observer": self.membership_observer.to_dict(),
            "storage_measurement_producer": self.storage_measurement_producer.to_dict(),
            "storage_terminal_relay": self.storage_terminal_relay.to_dict(),
            "security_profile": self.security_profile.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class HostFactsInventoryV1:
    """Exact facts that must remain identical at all live checkpoints."""

    kernel: HostKernelIdentityV1
    cgroup: CgroupV2IdentityV1
    docker: DockerDaemonIdentityV1
    components: HostComponentInventoryV1

    def __post_init__(self) -> None:
        _require_exact_type(self.kernel, HostKernelIdentityV1, "host kernel facts")
        _require_exact_type(self.cgroup, CgroupV2IdentityV1, "host cgroup facts")
        _require_exact_type(self.docker, DockerDaemonIdentityV1, "host Docker facts")
        _require_exact_type(self.components, HostComponentInventoryV1, "host component facts")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kernel": self.kernel.to_dict(),
            "cgroup": self.cgroup.to_dict(),
            "docker": self.docker.to_dict(),
            "components": self.components.to_dict(),
        }

    @property
    def inventory_sha256(self) -> str:
        return _sha256(canonical_provisioning_json_bytes(self.to_dict(), trailing_lf=False))


@dataclass(frozen=True, slots=True)
class SupportedHostTupleV1:
    """Exact software/host tuple authorized by a preissued policy."""

    tuple_id: str
    architecture: str
    kernel_release: str
    kernel_build_sha256: str
    kernel_command_line_sha256: str
    cgroup_filesystem_magic: str
    docker_cgroup_driver: str
    docker_version: str
    docker_api_version: str
    docker_config_sha256: str
    oci_runtime: PinnedComponentIdentityV1
    security_profile: PinnedComponentIdentityV1

    def __post_init__(self) -> None:
        _require_identifier(self.tuple_id, "supported host tuple ID")
        _require_identifier(self.architecture, "supported host architecture")
        _require_text(self.kernel_release, "supported kernel release", maximum=256)
        _require_sha256(self.kernel_build_sha256, "supported kernel build SHA-256")
        _require_sha256(
            self.kernel_command_line_sha256,
            "supported kernel command-line SHA-256",
        )
        if self.cgroup_filesystem_magic != _CGROUP_FILESYSTEM_MAGIC:
            _fail("supported host tuple must require cgroup v2")
        if self.docker_cgroup_driver != "cgroupfs":
            _fail("supported host tuple must require the Docker cgroupfs driver")
        _require_text(self.docker_version, "supported Docker version", maximum=256)
        _require_text(self.docker_api_version, "supported Docker API version", maximum=256)
        _require_sha256(self.docker_config_sha256, "supported Docker config SHA-256")
        _require_exact_type(self.oci_runtime, PinnedComponentIdentityV1, "supported OCI runtime")
        _require_exact_type(
            self.security_profile,
            PinnedComponentIdentityV1,
            "supported security profile",
        )

    @classmethod
    def from_facts(cls, tuple_id: str, facts: HostFactsInventoryV1) -> SupportedHostTupleV1:
        _require_exact_type(facts, HostFactsInventoryV1, "supported-tuple source facts")
        return cls(
            tuple_id=tuple_id,
            architecture=facts.kernel.architecture,
            kernel_release=facts.kernel.kernel_release,
            kernel_build_sha256=facts.kernel.kernel_build_sha256,
            kernel_command_line_sha256=facts.kernel.kernel_command_line_sha256,
            cgroup_filesystem_magic=facts.cgroup.filesystem_magic,
            docker_cgroup_driver=facts.docker.cgroup_driver,
            docker_version=facts.docker.version,
            docker_api_version=facts.docker.api_version,
            docker_config_sha256=facts.docker.config_sha256,
            oci_runtime=facts.components.oci_runtime,
            security_profile=facts.components.security_profile,
        )

    def matches_facts(self, facts: HostFactsInventoryV1) -> bool:
        _require_exact_type(facts, HostFactsInventoryV1, "supported-tuple comparison facts")
        return self == SupportedHostTupleV1.from_facts(self.tuple_id, facts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tuple_id": self.tuple_id,
            "architecture": self.architecture,
            "kernel_release": self.kernel_release,
            "kernel_build_sha256": self.kernel_build_sha256,
            "kernel_command_line_sha256": self.kernel_command_line_sha256,
            "cgroup_filesystem_magic": self.cgroup_filesystem_magic,
            "docker_cgroup_driver": self.docker_cgroup_driver,
            "docker_version": self.docker_version,
            "docker_api_version": self.docker_api_version,
            "docker_config_sha256": self.docker_config_sha256,
            "oci_runtime": self.oci_runtime.to_dict(),
            "security_profile": self.security_profile.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class HostProvisioningTrustContractDescriptorV1:
    """Self-description of this nonoperational trust contract."""

    schema_version: str = HOST_PROVISIONING_TRUST_CONTRACT_DESCRIPTOR_SCHEMA_VERSION
    status: str = "source_only_nonoperational_trust_contract"
    signature_algorithm: str = ED25519_SIGNATURE_ALGORITHM
    signature_domain: str = ED25519_SIGNATURE_DOMAIN_LABEL
    live_validation_checkpoints: tuple[str, ...] = LIVE_VALIDATION_CHECKPOINTS
    operational_apis: tuple[str, ...] = ()
    audit_pin_state: str = "descriptor_file_pin_required_source_identity_external"

    def __post_init__(self) -> None:
        if self.schema_version != HOST_PROVISIONING_TRUST_CONTRACT_DESCRIPTOR_SCHEMA_VERSION:
            _fail("host-provisioning descriptor schema differs")
        if self.status != "source_only_nonoperational_trust_contract":
            _fail("host-provisioning descriptor status differs")
        if self.signature_algorithm != ED25519_SIGNATURE_ALGORITHM:
            _fail("host-provisioning descriptor algorithm differs")
        if self.signature_domain != ED25519_SIGNATURE_DOMAIN_LABEL:
            _fail("host-provisioning descriptor domain differs")
        _require_exact_tuple(
            self.live_validation_checkpoints,
            LIVE_VALIDATION_CHECKPOINTS,
            "descriptor live-validation checkpoints",
        )
        if type(self.operational_apis) is not tuple or self.operational_apis:
            _fail("source-only host-provisioning descriptor cannot expose operations")
        if self.audit_pin_state != "descriptor_file_pin_required_source_identity_external":
            _fail("host-provisioning descriptor audit pin state differs")

    @property
    def capabilities(self) -> Mapping[str, bool]:
        return SOURCE_ONLY_CAPABILITIES

    @property
    def readiness(self) -> Mapping[str, bool]:
        return SOURCE_ONLY_READINESS

    @property
    def authority(self) -> Mapping[str, bool]:
        return SOURCE_ONLY_AUTHORITY

    @property
    def claims(self) -> Mapping[str, bool]:
        return SOURCE_ONLY_CLAIMS

    @property
    def safety_posture(self) -> Mapping[str, Mapping[str, bool]]:
        return SOURCE_ONLY_SAFETY_POSTURE

    def to_body_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "artifact_schema_versions": [
                HOST_TRUST_POLICY_SCHEMA_VERSION,
                HOST_PROVISIONING_STATEMENT_SCHEMA_VERSION,
                HOST_SIGNATURE_VERIFICATION_RECEIPT_SCHEMA_VERSION,
                HOST_LIVE_VALIDATION_RECEIPT_SCHEMA_VERSION,
            ],
            "signature_algorithm": self.signature_algorithm,
            "signature_domain": self.signature_domain,
            "signature_length_bytes": 64,
            "live_validation_checkpoints": list(self.live_validation_checkpoints),
            "required_retained_resource_event_counters": list(
                REQUIRED_RETAINED_RESOURCE_EVENT_COUNTERS
            ),
            "resource_event_counter_monotonicity_policy": (
                RESOURCE_EVENT_COUNTER_MONOTONICITY_POLICY
            ),
            "positive_resource_event_delta_policy": POSITIVE_RESOURCE_EVENT_DELTA_POLICY,
            "operational_apis": list(self.operational_apis),
            "executor_signing_secret_policy": "executor_never_holds_signing_secret",
            "hmac_policy": "forbidden",
            "parsing_semantics": (
                "structural_parsing_never_equates_to_cryptographic_verification"
            ),
            "audit_pin_state": self.audit_pin_state,
            "safety_posture": _safety_posture_dict(),
        }


@dataclass(frozen=True, slots=True)
class HostTrustPolicyV1:
    """Preissued exact host policy; valid metadata grants no authority."""

    policy_id: str
    policy_nonce_sha256: str
    qualification_plan: ArtifactIdentityV1
    issued_at_unix_ns: int
    valid_from_unix_ns: int
    valid_until_unix_ns: int
    signer_key_id: str
    signer_public_key_sha256: str
    independent_verifier: PinnedComponentIdentityV1
    live_validator: PinnedComponentIdentityV1
    supported_host_tuple: SupportedHostTupleV1
    expected_facts: HostFactsInventoryV1

    def __post_init__(self) -> None:
        _require_identifier(self.policy_id, "host trust policy ID")
        _require_sha256(self.policy_nonce_sha256, "host trust policy nonce SHA-256")
        _require_exact_type(
            self.qualification_plan,
            ArtifactIdentityV1,
            "host trust policy qualification plan",
        )
        if self.qualification_plan.schema_version != QUALIFICATION_PLAN_V3_SCHEMA_VERSION:
            _fail("host trust policy requires the exact qualification-plan v3 schema")
        issued = _require_int(
            self.issued_at_unix_ns,
            "host trust policy issuance time",
            minimum=1,
        )
        valid_from = _require_int(
            self.valid_from_unix_ns,
            "host trust policy valid-from time",
            minimum=1,
        )
        valid_until = _require_int(
            self.valid_until_unix_ns,
            "host trust policy valid-until time",
            minimum=1,
        )
        if not issued <= valid_from < valid_until:
            _fail("host trust policy chronology is invalid")
        _require_identifier(self.signer_key_id, "host provisioner signer key ID")
        _require_sha256(
            self.signer_public_key_sha256,
            "host provisioner signer public-key SHA-256",
        )
        _require_exact_type(
            self.independent_verifier,
            PinnedComponentIdentityV1,
            "independent signature verifier",
        )
        _require_exact_type(
            self.live_validator,
            PinnedComponentIdentityV1,
            "live host validator",
        )
        if self.independent_verifier.component_id == self.live_validator.component_id:
            _fail("signature verifier and live validator roles must be distinct")
        _require_exact_type(
            self.supported_host_tuple,
            SupportedHostTupleV1,
            "supported host tuple",
        )
        _require_exact_type(self.expected_facts, HostFactsInventoryV1, "expected host facts")
        if not self.supported_host_tuple.matches_facts(self.expected_facts):
            _fail("supported host tuple differs from the policy facts inventory")
        host_role_ids = {
            self.expected_facts.components.oci_runtime.component_id,
            self.expected_facts.components.membership_observer.component_id,
            self.expected_facts.components.storage_measurement_producer.component_id,
            self.expected_facts.components.storage_terminal_relay.component_id,
            self.expected_facts.components.security_profile.component_id,
        }
        if {
            self.independent_verifier.component_id,
            self.live_validator.component_id,
        } & host_role_ids:
            _fail("verification and validation roles must be separate from host components")

    @property
    def expected_facts_inventory_sha256(self) -> str:
        return self.expected_facts.inventory_sha256

    @property
    def safety_posture(self) -> Mapping[str, Mapping[str, bool]]:
        return SOURCE_ONLY_SAFETY_POSTURE

    def to_body_dict(self) -> dict[str, Any]:
        return {
            "schema_version": HOST_TRUST_POLICY_SCHEMA_VERSION,
            "status": _POLICY_STATUS,
            "policy_id": self.policy_id,
            "policy_nonce_sha256": self.policy_nonce_sha256,
            "qualification_plan": self.qualification_plan.to_dict(),
            "issued_at_unix_ns": self.issued_at_unix_ns,
            "valid_from_unix_ns": self.valid_from_unix_ns,
            "valid_until_unix_ns": self.valid_until_unix_ns,
            "signature_algorithm": ED25519_SIGNATURE_ALGORITHM,
            "signature_domain": ED25519_SIGNATURE_DOMAIN_LABEL,
            "signer_key_id": self.signer_key_id,
            "signer_public_key_sha256": self.signer_public_key_sha256,
            "independent_verifier": self.independent_verifier.to_dict(),
            "live_validator": self.live_validator.to_dict(),
            "supported_host_tuple": self.supported_host_tuple.to_dict(),
            "expected_facts": self.expected_facts.to_dict(),
            "expected_facts_inventory_sha256": self.expected_facts_inventory_sha256,
            "required_executor_handoff_resource_event_counters": list(
                REQUIRED_RETAINED_RESOURCE_EVENT_COUNTERS
            ),
            "resource_event_counter_monotonicity_policy": (
                RESOURCE_EVENT_COUNTER_MONOTONICITY_POLICY
            ),
            "positive_resource_event_delta_policy": POSITIVE_RESOURCE_EVENT_DELTA_POLICY,
            "executor_held_signing_secret": False,
            "hmac_allowed": False,
            "safety_posture": _safety_posture_dict(),
        }


@dataclass(frozen=True, slots=True)
class HostProvisioningStatementV1:
    """Detached Ed25519 signature metadata over one exact host inventory."""

    policy: ArtifactIdentityV1
    observed_at_unix_ns: int
    observed_at_monotonic_ns: int
    signer_key_id: str
    signer_public_key_sha256: str
    facts: HostFactsInventoryV1
    signature_hex: str

    def __post_init__(self) -> None:
        _require_exact_type(self.policy, ArtifactIdentityV1, "statement policy identity")
        if self.policy.schema_version != HOST_TRUST_POLICY_SCHEMA_VERSION:
            _fail("provisioning statement policy schema differs")
        _require_int(
            self.observed_at_unix_ns,
            "provisioning statement observation Unix time",
            minimum=1,
        )
        _require_int(
            self.observed_at_monotonic_ns,
            "provisioning statement observation monotonic time",
            minimum=1,
        )
        _require_identifier(self.signer_key_id, "provisioning statement signer key ID")
        _require_sha256(
            self.signer_public_key_sha256,
            "provisioning statement signer public-key SHA-256",
        )
        _require_exact_type(self.facts, HostFactsInventoryV1, "provisioning statement facts")
        _require_signature(self.signature_hex, "provisioning statement signature")

    @property
    def facts_inventory_sha256(self) -> str:
        return self.facts.inventory_sha256

    @property
    def signed_payload_sha256(self) -> str:
        return _sha256(canonical_host_provisioning_statement_v1_signed_payload_bytes(self))

    @property
    def safety_posture(self) -> Mapping[str, Mapping[str, bool]]:
        return SOURCE_ONLY_SAFETY_POSTURE

    def to_unsigned_dict(self) -> dict[str, Any]:
        return {
            "schema_version": HOST_PROVISIONING_STATEMENT_SCHEMA_VERSION,
            "status": _STATEMENT_STATUS,
            "policy": self.policy.to_dict(),
            "observed_at_unix_ns": self.observed_at_unix_ns,
            "observed_at_monotonic_ns": self.observed_at_monotonic_ns,
            "signature_algorithm": ED25519_SIGNATURE_ALGORITHM,
            "signature_domain": ED25519_SIGNATURE_DOMAIN_LABEL,
            "signer_key_id": self.signer_key_id,
            "signer_public_key_sha256": self.signer_public_key_sha256,
            "facts": self.facts.to_dict(),
            "facts_inventory_sha256": self.facts_inventory_sha256,
            "executor_held_signing_secret": False,
            "hmac_used": False,
            "signature_verified_by_parser": False,
            "safety_posture": _safety_posture_dict(),
        }

    def to_body_dict(self) -> dict[str, Any]:
        return {
            **self.to_unsigned_dict(),
            "signed_payload_sha256": self.signed_payload_sha256,
            "signature_hex": self.signature_hex,
        }


@dataclass(frozen=True, slots=True)
class HostSignatureVerificationReceiptV1:
    """Report from a policy-pinned verifier; this parser does no verification."""

    policy: ArtifactIdentityV1
    statement: ArtifactIdentityV1
    verifier: PinnedComponentIdentityV1
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
        _require_exact_type(self.policy, ArtifactIdentityV1, "verification policy identity")
        if self.policy.schema_version != HOST_TRUST_POLICY_SCHEMA_VERSION:
            _fail("signature-verification policy schema differs")
        _require_exact_type(
            self.statement,
            ArtifactIdentityV1,
            "verification statement identity",
        )
        if self.statement.schema_version != HOST_PROVISIONING_STATEMENT_SCHEMA_VERSION:
            _fail("signature-verification statement schema differs")
        _require_exact_type(self.verifier, PinnedComponentIdentityV1, "signature verifier")
        _require_sha256(self.verification_run_id_sha256, "signature verification run ID")
        started_unix = _require_int(
            self.verification_started_at_unix_ns,
            "signature verification start Unix time",
            minimum=1,
        )
        completed_unix = _require_int(
            self.verification_completed_at_unix_ns,
            "signature verification completion Unix time",
            minimum=1,
        )
        started_monotonic = _require_int(
            self.verification_started_at_monotonic_ns,
            "signature verification start monotonic time",
            minimum=1,
        )
        completed_monotonic = _require_int(
            self.verification_completed_at_monotonic_ns,
            "signature verification completion monotonic time",
            minimum=1,
        )
        if completed_unix < started_unix or completed_monotonic < started_monotonic:
            _fail("signature-verification chronology is invalid")
        _require_identifier(self.signer_key_id, "verified signer key ID")
        _require_sha256(
            self.signer_public_key_sha256,
            "verified signer public-key SHA-256",
        )
        _require_sha256(self.signed_payload_sha256, "verified signed-payload SHA-256")
        _require_sha256(self.signature_sha256, "verified signature SHA-256")

    @property
    def safety_posture(self) -> Mapping[str, Mapping[str, bool]]:
        return SOURCE_ONLY_SAFETY_POSTURE

    def to_body_dict(self) -> dict[str, Any]:
        return {
            "schema_version": HOST_SIGNATURE_VERIFICATION_RECEIPT_SCHEMA_VERSION,
            "status": _VERIFICATION_STATUS,
            "policy": self.policy.to_dict(),
            "statement": self.statement.to_dict(),
            "verifier": self.verifier.to_dict(),
            "verification_run_id_sha256": self.verification_run_id_sha256,
            "verification_started_at_unix_ns": self.verification_started_at_unix_ns,
            "verification_completed_at_unix_ns": self.verification_completed_at_unix_ns,
            "verification_started_at_monotonic_ns": (
                self.verification_started_at_monotonic_ns
            ),
            "verification_completed_at_monotonic_ns": (
                self.verification_completed_at_monotonic_ns
            ),
            "signature_algorithm": ED25519_SIGNATURE_ALGORITHM,
            "signature_domain": ED25519_SIGNATURE_DOMAIN_LABEL,
            "verification_method": _VERIFICATION_METHOD,
            "verification_result": _VERIFICATION_RESULT,
            "signer_key_id": self.signer_key_id,
            "signer_public_key_sha256": self.signer_public_key_sha256,
            "signed_payload_sha256": self.signed_payload_sha256,
            "signature_sha256": self.signature_sha256,
            "executor_held_signing_secret": False,
            "hmac_used": False,
            "cryptographic_verification_performed_by_parser": False,
            "safety_posture": _safety_posture_dict(),
        }


@dataclass(frozen=True, slots=True)
class HostLiveValidationReceiptV1:
    """One exact-checkpoint report of the full live host facts inventory."""

    checkpoint: str
    checkpoint_ordinal: int
    policy: ArtifactIdentityV1
    statement: ArtifactIdentityV1
    signature_verification_receipt: ArtifactIdentityV1
    previous_live_validation_receipt: ArtifactIdentityV1 | None
    validator: PinnedComponentIdentityV1
    validation_run_id_sha256: str
    validated_at_unix_ns: int
    validated_at_monotonic_ns: int
    facts: HostFactsInventoryV1

    def __post_init__(self) -> None:
        if type(self.checkpoint) is not str or self.checkpoint not in LIVE_VALIDATION_CHECKPOINTS:
            _fail("live-validation checkpoint is outside the exact checkpoint order")
        ordinal = _require_int(
            self.checkpoint_ordinal,
            "live-validation checkpoint ordinal",
            maximum=len(LIVE_VALIDATION_CHECKPOINTS) - 1,
        )
        if LIVE_VALIDATION_CHECKPOINTS[ordinal] != self.checkpoint:
            _fail("live-validation checkpoint and ordinal differ")
        _require_exact_type(self.policy, ArtifactIdentityV1, "live-validation policy identity")
        if self.policy.schema_version != HOST_TRUST_POLICY_SCHEMA_VERSION:
            _fail("live-validation policy schema differs")
        _require_exact_type(
            self.statement,
            ArtifactIdentityV1,
            "live-validation statement identity",
        )
        if self.statement.schema_version != HOST_PROVISIONING_STATEMENT_SCHEMA_VERSION:
            _fail("live-validation statement schema differs")
        _require_exact_type(
            self.signature_verification_receipt,
            ArtifactIdentityV1,
            "live-validation signature-verification identity",
        )
        if (
            self.signature_verification_receipt.schema_version
            != HOST_SIGNATURE_VERIFICATION_RECEIPT_SCHEMA_VERSION
        ):
            _fail("live-validation signature-verification schema differs")
        if ordinal == 0:
            if self.previous_live_validation_receipt is not None:
                _fail("pre-capability validation cannot name a previous live receipt")
        else:
            _require_exact_type(
                self.previous_live_validation_receipt,
                ArtifactIdentityV1,
                "previous live-validation identity",
            )
            previous = cast(ArtifactIdentityV1, self.previous_live_validation_receipt)
            if previous.schema_version != HOST_LIVE_VALIDATION_RECEIPT_SCHEMA_VERSION:
                _fail("previous live-validation schema differs")
        _require_exact_type(self.validator, PinnedComponentIdentityV1, "live validator")
        _require_sha256(self.validation_run_id_sha256, "live-validation run ID")
        _require_int(
            self.validated_at_unix_ns,
            "live-validation Unix time",
            minimum=1,
        )
        _require_int(
            self.validated_at_monotonic_ns,
            "live-validation monotonic time",
            minimum=1,
        )
        _require_exact_type(self.facts, HostFactsInventoryV1, "live-validation facts")

    @property
    def facts_inventory_sha256(self) -> str:
        return self.facts.inventory_sha256

    @property
    def safety_posture(self) -> Mapping[str, Mapping[str, bool]]:
        return SOURCE_ONLY_SAFETY_POSTURE

    def to_body_dict(self) -> dict[str, Any]:
        return {
            "schema_version": HOST_LIVE_VALIDATION_RECEIPT_SCHEMA_VERSION,
            "status": _LIVE_STATUS,
            "checkpoint": self.checkpoint,
            "checkpoint_ordinal": self.checkpoint_ordinal,
            "policy": self.policy.to_dict(),
            "statement": self.statement.to_dict(),
            "signature_verification_receipt": (
                self.signature_verification_receipt.to_dict()
            ),
            "previous_live_validation_receipt": (
                None
                if self.previous_live_validation_receipt is None
                else self.previous_live_validation_receipt.to_dict()
            ),
            "validator": self.validator.to_dict(),
            "validation_run_id_sha256": self.validation_run_id_sha256,
            "validated_at_unix_ns": self.validated_at_unix_ns,
            "validated_at_monotonic_ns": self.validated_at_monotonic_ns,
            "facts": self.facts.to_dict(),
            "facts_inventory_sha256": self.facts_inventory_sha256,
            "validation_result": _LIVE_RESULT,
            "cryptographic_verification_performed_by_parser": False,
            "execution_authority_granted": False,
            "safety_posture": _safety_posture_dict(),
        }


def canonical_host_provisioning_trust_contract_descriptor_v1_body_bytes(
    descriptor: HostProvisioningTrustContractDescriptorV1,
) -> bytes:
    return _canonical_artifact_body_bytes(
        descriptor,
        HostProvisioningTrustContractDescriptorV1,
    )


def canonical_host_provisioning_trust_contract_descriptor_v1_file_bytes(
    descriptor: HostProvisioningTrustContractDescriptorV1,
) -> bytes:
    return _canonical_artifact_file_bytes(
        descriptor,
        HostProvisioningTrustContractDescriptorV1,
        _DESCRIPTOR_BODY_FIELD,
    )


def _guard_host_provisioning_v3_descriptor_pin() -> str:
    descriptor = HostProvisioningTrustContractDescriptorV1()
    raw = canonical_host_provisioning_trust_contract_descriptor_v1_file_bytes(descriptor)
    pin = PINNED_HOST_PROVISIONING_V3_DESCRIPTOR_FILE_SHA256
    if type(pin) is not str or _SHA256_RE.fullmatch(pin) is None or pin == "0" * 64:
        _fail("host-provisioning descriptor FILE pin is not independently finalized")
    if not secrets.compare_digest(pin, _sha256(raw)):
        _fail("host-provisioning descriptor FILE drifted from its repository literal")
    return pin


def host_provisioning_v3_descriptor_sha256() -> str:
    """Return the descriptor FILE identity after its repository pin is finalized."""

    return _guard_host_provisioning_v3_descriptor_pin()


def canonical_host_trust_policy_v1_body_bytes(policy: HostTrustPolicyV1) -> bytes:
    return _canonical_artifact_body_bytes(policy, HostTrustPolicyV1)


def canonical_host_trust_policy_v1_file_bytes(policy: HostTrustPolicyV1) -> bytes:
    return _canonical_artifact_file_bytes(policy, HostTrustPolicyV1, _POLICY_BODY_FIELD)


def canonical_host_provisioning_statement_v1_signed_payload_bytes(
    statement: HostProvisioningStatementV1,
) -> bytes:
    if type(statement) is not HostProvisioningStatementV1:
        raise TypeError("statement must use exact type HostProvisioningStatementV1")
    return ED25519_SIGNATURE_DOMAIN + canonical_provisioning_json_bytes(
        statement.to_unsigned_dict(),
        trailing_lf=False,
    )


def canonical_host_provisioning_statement_v1_body_bytes(
    statement: HostProvisioningStatementV1,
) -> bytes:
    return _canonical_artifact_body_bytes(statement, HostProvisioningStatementV1)


def canonical_host_provisioning_statement_v1_file_bytes(
    statement: HostProvisioningStatementV1,
) -> bytes:
    return _canonical_artifact_file_bytes(
        statement,
        HostProvisioningStatementV1,
        _STATEMENT_BODY_FIELD,
    )


def canonical_host_signature_verification_receipt_v1_body_bytes(
    receipt: HostSignatureVerificationReceiptV1,
) -> bytes:
    return _canonical_artifact_body_bytes(receipt, HostSignatureVerificationReceiptV1)


def canonical_host_signature_verification_receipt_v1_file_bytes(
    receipt: HostSignatureVerificationReceiptV1,
) -> bytes:
    return _canonical_artifact_file_bytes(
        receipt,
        HostSignatureVerificationReceiptV1,
        _VERIFICATION_BODY_FIELD,
    )


def canonical_host_live_validation_receipt_v1_body_bytes(
    receipt: HostLiveValidationReceiptV1,
) -> bytes:
    return _canonical_artifact_body_bytes(receipt, HostLiveValidationReceiptV1)


def canonical_host_live_validation_receipt_v1_file_bytes(
    receipt: HostLiveValidationReceiptV1,
) -> bytes:
    return _canonical_artifact_file_bytes(
        receipt,
        HostLiveValidationReceiptV1,
        _LIVE_BODY_FIELD,
    )


def _identity(
    schema_version: str,
    body_bytes: bytes,
    file_bytes: bytes,
) -> ArtifactIdentityV1:
    return ArtifactIdentityV1(
        schema_version=schema_version,
        file_sha256=_sha256(file_bytes),
        body_sha256=_sha256(body_bytes),
    )


def host_trust_policy_identity_v1(policy: HostTrustPolicyV1) -> ArtifactIdentityV1:
    return _identity(
        HOST_TRUST_POLICY_SCHEMA_VERSION,
        canonical_host_trust_policy_v1_body_bytes(policy),
        canonical_host_trust_policy_v1_file_bytes(policy),
    )


def host_provisioning_statement_identity_v1(
    statement: HostProvisioningStatementV1,
) -> ArtifactIdentityV1:
    return _identity(
        HOST_PROVISIONING_STATEMENT_SCHEMA_VERSION,
        canonical_host_provisioning_statement_v1_body_bytes(statement),
        canonical_host_provisioning_statement_v1_file_bytes(statement),
    )


def host_signature_verification_receipt_identity_v1(
    receipt: HostSignatureVerificationReceiptV1,
) -> ArtifactIdentityV1:
    return _identity(
        HOST_SIGNATURE_VERIFICATION_RECEIPT_SCHEMA_VERSION,
        canonical_host_signature_verification_receipt_v1_body_bytes(receipt),
        canonical_host_signature_verification_receipt_v1_file_bytes(receipt),
    )


def host_live_validation_receipt_identity_v1(
    receipt: HostLiveValidationReceiptV1,
) -> ArtifactIdentityV1:
    return _identity(
        HOST_LIVE_VALIDATION_RECEIPT_SCHEMA_VERSION,
        canonical_host_live_validation_receipt_v1_body_bytes(receipt),
        canonical_host_live_validation_receipt_v1_file_bytes(receipt),
    )


def _expect_literal(value: object, expected: str, label: str) -> None:
    if value != expected:
        _fail(f"{label} differs from the exact contract literal")


def _parse_artifact_identity(value: object, label: str) -> ArtifactIdentityV1:
    item = _require_exact_keys(
        value,
        frozenset({"schema_version", "file_sha256", "body_sha256"}),
        label,
    )
    return ArtifactIdentityV1(
        schema_version=_require_identifier(item["schema_version"], label + " schema"),
        file_sha256=_require_sha256(item["file_sha256"], label + " FILE SHA-256"),
        body_sha256=_require_sha256(item["body_sha256"], label + " BODY SHA-256"),
    )


def _parse_component(value: object, label: str) -> PinnedComponentIdentityV1:
    item = _require_exact_keys(
        value,
        frozenset(
            {
                "component_id",
                "descriptor_schema_version",
                "descriptor_file_sha256",
                "source_sha256",
                "runtime_artifact_sha256",
            }
        ),
        label,
    )
    return PinnedComponentIdentityV1(
        component_id=_require_identifier(item["component_id"], label + " component ID"),
        descriptor_schema_version=_require_identifier(
            item["descriptor_schema_version"],
            label + " descriptor schema",
        ),
        descriptor_file_sha256=_require_sha256(
            item["descriptor_file_sha256"],
            label + " descriptor FILE SHA-256",
        ),
        source_sha256=_require_sha256(item["source_sha256"], label + " source SHA-256"),
        runtime_artifact_sha256=_require_sha256(
            item["runtime_artifact_sha256"],
            label + " runtime artifact SHA-256",
        ),
    )


def _parse_kernel(value: object, label: str) -> HostKernelIdentityV1:
    item = _require_exact_keys(
        value,
        frozenset(
            {
                "host_identity_sha256",
                "machine_id_sha256",
                "boot_id",
                "architecture",
                "kernel_release",
                "kernel_build_sha256",
                "kernel_command_line_sha256",
            }
        ),
        label,
    )
    return HostKernelIdentityV1(
        host_identity_sha256=_require_sha256(
            item["host_identity_sha256"],
            label + " host identity SHA-256",
        ),
        machine_id_sha256=_require_sha256(
            item["machine_id_sha256"],
            label + " machine-id SHA-256",
        ),
        boot_id=_require_boot_id(item["boot_id"], label + " boot ID"),
        architecture=_require_identifier(item["architecture"], label + " architecture"),
        kernel_release=_require_text(
            item["kernel_release"],
            label + " kernel release",
            maximum=256,
        ),
        kernel_build_sha256=_require_sha256(
            item["kernel_build_sha256"],
            label + " kernel build SHA-256",
        ),
        kernel_command_line_sha256=_require_sha256(
            item["kernel_command_line_sha256"],
            label + " kernel command-line SHA-256",
        ),
    )


def _parse_controller_list(value: object, label: str) -> tuple[str, ...]:
    if type(value) is not list:
        _fail(f"{label} must be one exact JSON array")
    result = tuple(_require_identifier(item, label + " entry") for item in value)
    return _require_exact_tuple(result, _REQUIRED_CONTROLLERS, label)


def _parse_cgroup(value: object, label: str) -> CgroupV2IdentityV1:
    item = _require_exact_keys(
        value,
        frozenset(
            {
                "mount_path",
                "mount_device_major",
                "mount_device_minor",
                "mount_inode",
                "filesystem_magic",
                "unified_hierarchy",
                "delegate_path",
                "delegate_device_major",
                "delegate_device_minor",
                "delegate_inode",
                "delegate_uid",
                "delegate_gid",
                "delegate_mode",
                "delegated_controllers",
                "subtree_control",
            }
        ),
        label,
    )
    return CgroupV2IdentityV1(
        mount_path=_require_absolute_path(item["mount_path"], label + " mount path"),
        mount_device_major=_require_int(
            item["mount_device_major"],
            label + " mount device major",
        ),
        mount_device_minor=_require_int(
            item["mount_device_minor"],
            label + " mount device minor",
        ),
        mount_inode=_require_int(item["mount_inode"], label + " mount inode", minimum=1),
        filesystem_magic=_require_text(
            item["filesystem_magic"],
            label + " filesystem magic",
        ),
        unified_hierarchy=_require_bool(
            item["unified_hierarchy"],
            label + " unified hierarchy",
        ),
        delegate_path=_require_absolute_path(
            item["delegate_path"],
            label + " delegate path",
        ),
        delegate_device_major=_require_int(
            item["delegate_device_major"],
            label + " delegate device major",
        ),
        delegate_device_minor=_require_int(
            item["delegate_device_minor"],
            label + " delegate device minor",
        ),
        delegate_inode=_require_int(
            item["delegate_inode"],
            label + " delegate inode",
            minimum=1,
        ),
        delegate_uid=_require_int(item["delegate_uid"], label + " delegate UID"),
        delegate_gid=_require_int(item["delegate_gid"], label + " delegate GID"),
        delegate_mode=_require_int(
            item["delegate_mode"],
            label + " delegate mode",
            maximum=0o7777,
        ),
        delegated_controllers=_parse_controller_list(
            item["delegated_controllers"],
            label + " delegated controllers",
        ),
        subtree_control=_parse_controller_list(
            item["subtree_control"],
            label + " subtree control",
        ),
    )


def _parse_docker(value: object, label: str) -> DockerDaemonIdentityV1:
    item = _require_exact_keys(
        value,
        frozenset(
            {
                "socket_path",
                "socket_device_major",
                "socket_device_minor",
                "socket_inode",
                "socket_uid",
                "socket_gid",
                "socket_mode",
                "daemon_id",
                "daemon_pid",
                "daemon_start_ticks",
                "rootful",
                "cgroup_driver",
                "version",
                "api_version",
                "config_sha256",
                "root_dir_path",
                "root_dir_device_major",
                "root_dir_device_minor",
                "root_dir_inode",
            }
        ),
        label,
    )
    return DockerDaemonIdentityV1(
        socket_path=_require_absolute_path(item["socket_path"], label + " socket path"),
        socket_device_major=_require_int(
            item["socket_device_major"],
            label + " socket device major",
        ),
        socket_device_minor=_require_int(
            item["socket_device_minor"],
            label + " socket device minor",
        ),
        socket_inode=_require_int(item["socket_inode"], label + " socket inode", minimum=1),
        socket_uid=_require_int(item["socket_uid"], label + " socket UID"),
        socket_gid=_require_int(item["socket_gid"], label + " socket GID"),
        socket_mode=_require_int(
            item["socket_mode"],
            label + " socket mode",
            maximum=0o7777,
        ),
        daemon_id=_require_identifier(item["daemon_id"], label + " daemon ID"),
        daemon_pid=_require_int(item["daemon_pid"], label + " daemon PID", minimum=1),
        daemon_start_ticks=_require_int(
            item["daemon_start_ticks"],
            label + " daemon start ticks",
            minimum=1,
        ),
        rootful=_require_bool(item["rootful"], label + " rootful fact"),
        cgroup_driver=_require_text(item["cgroup_driver"], label + " cgroup driver"),
        version=_require_text(item["version"], label + " version", maximum=256),
        api_version=_require_text(
            item["api_version"],
            label + " API version",
            maximum=256,
        ),
        config_sha256=_require_sha256(item["config_sha256"], label + " config SHA-256"),
        root_dir_path=_require_absolute_path(
            item["root_dir_path"],
            label + " root-dir path",
        ),
        root_dir_device_major=_require_int(
            item["root_dir_device_major"],
            label + " root-dir device major",
        ),
        root_dir_device_minor=_require_int(
            item["root_dir_device_minor"],
            label + " root-dir device minor",
        ),
        root_dir_inode=_require_int(
            item["root_dir_inode"],
            label + " root-dir inode",
            minimum=1,
        ),
    )


def _parse_components(value: object, label: str) -> HostComponentInventoryV1:
    item = _require_exact_keys(
        value,
        frozenset(
            {
                "oci_runtime",
                "membership_observer",
                "storage_measurement_producer",
                "storage_terminal_relay",
                "security_profile",
            }
        ),
        label,
    )
    return HostComponentInventoryV1(
        oci_runtime=_parse_component(item["oci_runtime"], label + " OCI runtime"),
        membership_observer=_parse_component(
            item["membership_observer"],
            label + " membership observer",
        ),
        storage_measurement_producer=_parse_component(
            item["storage_measurement_producer"],
            label + " storage measurement producer",
        ),
        storage_terminal_relay=_parse_component(
            item["storage_terminal_relay"],
            label + " storage terminal relay",
        ),
        security_profile=_parse_component(
            item["security_profile"],
            label + " security profile",
        ),
    )


def _parse_facts(value: object, label: str) -> HostFactsInventoryV1:
    item = _require_exact_keys(
        value,
        frozenset({"kernel", "cgroup", "docker", "components"}),
        label,
    )
    return HostFactsInventoryV1(
        kernel=_parse_kernel(item["kernel"], label + " kernel"),
        cgroup=_parse_cgroup(item["cgroup"], label + " cgroup"),
        docker=_parse_docker(item["docker"], label + " Docker"),
        components=_parse_components(item["components"], label + " components"),
    )


def _parse_supported_tuple(value: object, label: str) -> SupportedHostTupleV1:
    item = _require_exact_keys(
        value,
        frozenset(
            {
                "tuple_id",
                "architecture",
                "kernel_release",
                "kernel_build_sha256",
                "kernel_command_line_sha256",
                "cgroup_filesystem_magic",
                "docker_cgroup_driver",
                "docker_version",
                "docker_api_version",
                "docker_config_sha256",
                "oci_runtime",
                "security_profile",
            }
        ),
        label,
    )
    return SupportedHostTupleV1(
        tuple_id=_require_identifier(item["tuple_id"], label + " ID"),
        architecture=_require_identifier(item["architecture"], label + " architecture"),
        kernel_release=_require_text(
            item["kernel_release"],
            label + " kernel release",
            maximum=256,
        ),
        kernel_build_sha256=_require_sha256(
            item["kernel_build_sha256"],
            label + " kernel build SHA-256",
        ),
        kernel_command_line_sha256=_require_sha256(
            item["kernel_command_line_sha256"],
            label + " kernel command-line SHA-256",
        ),
        cgroup_filesystem_magic=_require_text(
            item["cgroup_filesystem_magic"],
            label + " cgroup filesystem magic",
        ),
        docker_cgroup_driver=_require_text(
            item["docker_cgroup_driver"],
            label + " Docker cgroup driver",
        ),
        docker_version=_require_text(
            item["docker_version"],
            label + " Docker version",
            maximum=256,
        ),
        docker_api_version=_require_text(
            item["docker_api_version"],
            label + " Docker API version",
            maximum=256,
        ),
        docker_config_sha256=_require_sha256(
            item["docker_config_sha256"],
            label + " Docker config SHA-256",
        ),
        oci_runtime=_parse_component(item["oci_runtime"], label + " OCI runtime"),
        security_profile=_parse_component(
            item["security_profile"],
            label + " security profile",
        ),
    )


def parse_host_provisioning_trust_contract_descriptor_v1(
    raw: bytes,
    *,
    expected_file_sha256: str,
) -> HostProvisioningTrustContractDescriptorV1:
    pinned = _guard_host_provisioning_v3_descriptor_pin()
    caller_pin = _require_sha256(
        expected_file_sha256,
        "host-provisioning descriptor caller file pin",
    )
    if not secrets.compare_digest(caller_pin, pinned):
        _fail("host-provisioning descriptor caller pin differs from repository literal")
    body_keys = frozenset(
        {
            "schema_version",
            "status",
            "artifact_schema_versions",
            "signature_algorithm",
            "signature_domain",
            "signature_length_bytes",
            "live_validation_checkpoints",
            "required_retained_resource_event_counters",
            "resource_event_counter_monotonicity_policy",
            "positive_resource_event_delta_policy",
            "operational_apis",
            "executor_signing_secret_policy",
            "hmac_policy",
            "parsing_semantics",
            "audit_pin_state",
            "safety_posture",
        }
    )
    body = _parse_artifact_file(
        raw,
        expected_file_sha256=pinned,
        body_field=_DESCRIPTOR_BODY_FIELD,
        body_keys=body_keys,
        label="host-provisioning descriptor",
    )
    _expect_literal(
        body["schema_version"],
        HOST_PROVISIONING_TRUST_CONTRACT_DESCRIPTOR_SCHEMA_VERSION,
        "host-provisioning descriptor schema",
    )
    _expect_literal(
        body["status"],
        "source_only_nonoperational_trust_contract",
        "descriptor status",
    )
    if body["artifact_schema_versions"] != [
        HOST_TRUST_POLICY_SCHEMA_VERSION,
        HOST_PROVISIONING_STATEMENT_SCHEMA_VERSION,
        HOST_SIGNATURE_VERIFICATION_RECEIPT_SCHEMA_VERSION,
        HOST_LIVE_VALIDATION_RECEIPT_SCHEMA_VERSION,
    ]:
        _fail("descriptor artifact schema order differs")
    _expect_literal(
        body["signature_algorithm"],
        ED25519_SIGNATURE_ALGORITHM,
        "descriptor algorithm",
    )
    _expect_literal(body["signature_domain"], ED25519_SIGNATURE_DOMAIN_LABEL, "descriptor domain")
    if body["signature_length_bytes"] != 64:
        _fail("descriptor signature length differs")
    if body["live_validation_checkpoints"] != list(LIVE_VALIDATION_CHECKPOINTS):
        _fail("descriptor live-validation checkpoint order differs")
    if body["required_retained_resource_event_counters"] != list(
        REQUIRED_RETAINED_RESOURCE_EVENT_COUNTERS
    ):
        _fail("descriptor retained resource-event counter order differs")
    _expect_literal(
        body["resource_event_counter_monotonicity_policy"],
        RESOURCE_EVENT_COUNTER_MONOTONICITY_POLICY,
        "descriptor resource-event counter monotonicity policy",
    )
    _expect_literal(
        body["positive_resource_event_delta_policy"],
        POSITIVE_RESOURCE_EVENT_DELTA_POLICY,
        "descriptor positive resource-event delta policy",
    )
    if body["operational_apis"] != []:
        _fail("source-only descriptor cannot expose operational APIs")
    _expect_literal(
        body["executor_signing_secret_policy"],
        "executor_never_holds_signing_secret",
        "descriptor signing-secret policy",
    )
    _expect_literal(body["hmac_policy"], "forbidden", "descriptor HMAC policy")
    _expect_literal(
        body["parsing_semantics"],
        "structural_parsing_never_equates_to_cryptographic_verification",
        "descriptor parsing semantics",
    )
    _expect_literal(
        body["audit_pin_state"],
        "descriptor_file_pin_required_source_identity_external",
        "descriptor audit-pin state",
    )
    _parse_safety_posture(body["safety_posture"], "descriptor safety posture")
    return HostProvisioningTrustContractDescriptorV1()


def parse_host_trust_policy_v1(
    raw: bytes,
    *,
    expected_file_sha256: str,
) -> HostTrustPolicyV1:
    body_keys = frozenset(
        {
            "schema_version",
            "status",
            "policy_id",
            "policy_nonce_sha256",
            "qualification_plan",
            "issued_at_unix_ns",
            "valid_from_unix_ns",
            "valid_until_unix_ns",
            "signature_algorithm",
            "signature_domain",
            "signer_key_id",
            "signer_public_key_sha256",
            "independent_verifier",
            "live_validator",
            "supported_host_tuple",
            "expected_facts",
            "expected_facts_inventory_sha256",
            "required_executor_handoff_resource_event_counters",
            "resource_event_counter_monotonicity_policy",
            "positive_resource_event_delta_policy",
            "executor_held_signing_secret",
            "hmac_allowed",
            "safety_posture",
        }
    )
    body = _parse_artifact_file(
        raw,
        expected_file_sha256=expected_file_sha256,
        body_field=_POLICY_BODY_FIELD,
        body_keys=body_keys,
        label="host trust policy",
    )
    _expect_literal(body["schema_version"], HOST_TRUST_POLICY_SCHEMA_VERSION, "policy schema")
    _expect_literal(body["status"], _POLICY_STATUS, "policy status")
    _expect_literal(body["signature_algorithm"], ED25519_SIGNATURE_ALGORITHM, "policy algorithm")
    _expect_literal(body["signature_domain"], ED25519_SIGNATURE_DOMAIN_LABEL, "policy domain")
    if body["required_executor_handoff_resource_event_counters"] != list(
        REQUIRED_RETAINED_RESOURCE_EVENT_COUNTERS
    ):
        _fail("policy retained resource-event counter order differs")
    _expect_literal(
        body["resource_event_counter_monotonicity_policy"],
        RESOURCE_EVENT_COUNTER_MONOTONICITY_POLICY,
        "policy resource-event counter monotonicity policy",
    )
    _expect_literal(
        body["positive_resource_event_delta_policy"],
        POSITIVE_RESOURCE_EVENT_DELTA_POLICY,
        "policy positive resource-event delta policy",
    )
    _require_bool(
        body["executor_held_signing_secret"],
        "policy executor-held-signing-secret fact",
        expected=False,
    )
    _require_bool(body["hmac_allowed"], "policy HMAC allowance", expected=False)
    _parse_safety_posture(body["safety_posture"], "policy safety posture")
    result = HostTrustPolicyV1(
        policy_id=_require_identifier(body["policy_id"], "policy ID"),
        policy_nonce_sha256=_require_sha256(body["policy_nonce_sha256"], "policy nonce SHA-256"),
        qualification_plan=_parse_artifact_identity(
            body["qualification_plan"],
            "policy qualification plan",
        ),
        issued_at_unix_ns=_require_int(
            body["issued_at_unix_ns"],
            "policy issuance time",
            minimum=1,
        ),
        valid_from_unix_ns=_require_int(
            body["valid_from_unix_ns"],
            "policy valid-from time",
            minimum=1,
        ),
        valid_until_unix_ns=_require_int(
            body["valid_until_unix_ns"],
            "policy valid-until time",
            minimum=1,
        ),
        signer_key_id=_require_identifier(body["signer_key_id"], "policy signer key ID"),
        signer_public_key_sha256=_require_sha256(
            body["signer_public_key_sha256"],
            "policy signer public-key SHA-256",
        ),
        independent_verifier=_parse_component(
            body["independent_verifier"],
            "policy independent verifier",
        ),
        live_validator=_parse_component(body["live_validator"], "policy live validator"),
        supported_host_tuple=_parse_supported_tuple(
            body["supported_host_tuple"],
            "policy supported host tuple",
        ),
        expected_facts=_parse_facts(body["expected_facts"], "policy expected facts"),
    )
    supplied_inventory = _require_sha256(
        body["expected_facts_inventory_sha256"],
        "policy expected-facts inventory SHA-256",
    )
    if not secrets.compare_digest(supplied_inventory, result.expected_facts_inventory_sha256):
        _fail("policy expected-facts inventory SHA-256 differs")
    return result


def parse_host_provisioning_statement_v1(
    raw: bytes,
    *,
    expected_file_sha256: str,
) -> HostProvisioningStatementV1:
    body_keys = frozenset(
        {
            "schema_version",
            "status",
            "policy",
            "observed_at_unix_ns",
            "observed_at_monotonic_ns",
            "signature_algorithm",
            "signature_domain",
            "signer_key_id",
            "signer_public_key_sha256",
            "facts",
            "facts_inventory_sha256",
            "executor_held_signing_secret",
            "hmac_used",
            "signature_verified_by_parser",
            "safety_posture",
            "signed_payload_sha256",
            "signature_hex",
        }
    )
    body = _parse_artifact_file(
        raw,
        expected_file_sha256=expected_file_sha256,
        body_field=_STATEMENT_BODY_FIELD,
        body_keys=body_keys,
        label="host provisioning statement",
    )
    _expect_literal(
        body["schema_version"],
        HOST_PROVISIONING_STATEMENT_SCHEMA_VERSION,
        "provisioning statement schema",
    )
    _expect_literal(body["status"], _STATEMENT_STATUS, "provisioning statement status")
    _expect_literal(
        body["signature_algorithm"],
        ED25519_SIGNATURE_ALGORITHM,
        "provisioning statement algorithm",
    )
    _expect_literal(
        body["signature_domain"],
        ED25519_SIGNATURE_DOMAIN_LABEL,
        "provisioning statement domain",
    )
    _require_bool(
        body["executor_held_signing_secret"],
        "statement executor-held-signing-secret fact",
        expected=False,
    )
    _require_bool(body["hmac_used"], "statement HMAC fact", expected=False)
    _require_bool(
        body["signature_verified_by_parser"],
        "statement parser-verification claim",
        expected=False,
    )
    _parse_safety_posture(body["safety_posture"], "statement safety posture")
    result = HostProvisioningStatementV1(
        policy=_parse_artifact_identity(body["policy"], "statement policy"),
        observed_at_unix_ns=_require_int(
            body["observed_at_unix_ns"],
            "statement observation Unix time",
            minimum=1,
        ),
        observed_at_monotonic_ns=_require_int(
            body["observed_at_monotonic_ns"],
            "statement observation monotonic time",
            minimum=1,
        ),
        signer_key_id=_require_identifier(
            body["signer_key_id"],
            "statement signer key ID",
        ),
        signer_public_key_sha256=_require_sha256(
            body["signer_public_key_sha256"],
            "statement signer public-key SHA-256",
        ),
        facts=_parse_facts(body["facts"], "statement facts"),
        signature_hex=_require_signature(body["signature_hex"], "statement signature"),
    )
    supplied_facts = _require_sha256(
        body["facts_inventory_sha256"],
        "statement facts inventory SHA-256",
    )
    if not secrets.compare_digest(supplied_facts, result.facts_inventory_sha256):
        _fail("statement facts inventory SHA-256 differs")
    supplied_payload = _require_sha256(
        body["signed_payload_sha256"],
        "statement signed-payload SHA-256",
    )
    if not secrets.compare_digest(supplied_payload, result.signed_payload_sha256):
        _fail("statement signed-payload SHA-256 differs")
    return result


def parse_host_signature_verification_receipt_v1(
    raw: bytes,
    *,
    expected_file_sha256: str,
) -> HostSignatureVerificationReceiptV1:
    body_keys = frozenset(
        {
            "schema_version",
            "status",
            "policy",
            "statement",
            "verifier",
            "verification_run_id_sha256",
            "verification_started_at_unix_ns",
            "verification_completed_at_unix_ns",
            "verification_started_at_monotonic_ns",
            "verification_completed_at_monotonic_ns",
            "signature_algorithm",
            "signature_domain",
            "verification_method",
            "verification_result",
            "signer_key_id",
            "signer_public_key_sha256",
            "signed_payload_sha256",
            "signature_sha256",
            "executor_held_signing_secret",
            "hmac_used",
            "cryptographic_verification_performed_by_parser",
            "safety_posture",
        }
    )
    body = _parse_artifact_file(
        raw,
        expected_file_sha256=expected_file_sha256,
        body_field=_VERIFICATION_BODY_FIELD,
        body_keys=body_keys,
        label="host signature-verification receipt",
    )
    _expect_literal(
        body["schema_version"],
        HOST_SIGNATURE_VERIFICATION_RECEIPT_SCHEMA_VERSION,
        "signature-verification schema",
    )
    _expect_literal(body["status"], _VERIFICATION_STATUS, "signature-verification status")
    _expect_literal(
        body["signature_algorithm"],
        ED25519_SIGNATURE_ALGORITHM,
        "signature-verification algorithm",
    )
    _expect_literal(
        body["signature_domain"],
        ED25519_SIGNATURE_DOMAIN_LABEL,
        "signature-verification domain",
    )
    _expect_literal(
        body["verification_method"],
        _VERIFICATION_METHOD,
        "signature-verification method",
    )
    _expect_literal(
        body["verification_result"],
        _VERIFICATION_RESULT,
        "signature-verification result",
    )
    _require_bool(
        body["executor_held_signing_secret"],
        "verification executor-held-signing-secret fact",
        expected=False,
    )
    _require_bool(body["hmac_used"], "verification HMAC fact", expected=False)
    _require_bool(
        body["cryptographic_verification_performed_by_parser"],
        "verification parser-cryptography claim",
        expected=False,
    )
    _parse_safety_posture(body["safety_posture"], "verification safety posture")
    return HostSignatureVerificationReceiptV1(
        policy=_parse_artifact_identity(body["policy"], "verification policy"),
        statement=_parse_artifact_identity(
            body["statement"],
            "verification statement",
        ),
        verifier=_parse_component(body["verifier"], "verification verifier"),
        verification_run_id_sha256=_require_sha256(
            body["verification_run_id_sha256"],
            "verification run ID",
        ),
        verification_started_at_unix_ns=_require_int(
            body["verification_started_at_unix_ns"],
            "verification start Unix time",
            minimum=1,
        ),
        verification_completed_at_unix_ns=_require_int(
            body["verification_completed_at_unix_ns"],
            "verification completion Unix time",
            minimum=1,
        ),
        verification_started_at_monotonic_ns=_require_int(
            body["verification_started_at_monotonic_ns"],
            "verification start monotonic time",
            minimum=1,
        ),
        verification_completed_at_monotonic_ns=_require_int(
            body["verification_completed_at_monotonic_ns"],
            "verification completion monotonic time",
            minimum=1,
        ),
        signer_key_id=_require_identifier(
            body["signer_key_id"],
            "verification signer key ID",
        ),
        signer_public_key_sha256=_require_sha256(
            body["signer_public_key_sha256"],
            "verification signer public-key SHA-256",
        ),
        signed_payload_sha256=_require_sha256(
            body["signed_payload_sha256"],
            "verification signed-payload SHA-256",
        ),
        signature_sha256=_require_sha256(
            body["signature_sha256"],
            "verification signature SHA-256",
        ),
    )


def parse_host_live_validation_receipt_v1(
    raw: bytes,
    *,
    expected_file_sha256: str,
) -> HostLiveValidationReceiptV1:
    body_keys = frozenset(
        {
            "schema_version",
            "status",
            "checkpoint",
            "checkpoint_ordinal",
            "policy",
            "statement",
            "signature_verification_receipt",
            "previous_live_validation_receipt",
            "validator",
            "validation_run_id_sha256",
            "validated_at_unix_ns",
            "validated_at_monotonic_ns",
            "facts",
            "facts_inventory_sha256",
            "validation_result",
            "cryptographic_verification_performed_by_parser",
            "execution_authority_granted",
            "safety_posture",
        }
    )
    body = _parse_artifact_file(
        raw,
        expected_file_sha256=expected_file_sha256,
        body_field=_LIVE_BODY_FIELD,
        body_keys=body_keys,
        label="host live-validation receipt",
    )
    _expect_literal(
        body["schema_version"],
        HOST_LIVE_VALIDATION_RECEIPT_SCHEMA_VERSION,
        "live-validation schema",
    )
    _expect_literal(body["status"], _LIVE_STATUS, "live-validation status")
    _expect_literal(
        body["validation_result"],
        _LIVE_RESULT,
        "live-validation result",
    )
    _require_bool(
        body["cryptographic_verification_performed_by_parser"],
        "live-validation parser-cryptography claim",
        expected=False,
    )
    _require_bool(
        body["execution_authority_granted"],
        "live-validation execution-authority claim",
        expected=False,
    )
    _parse_safety_posture(body["safety_posture"], "live-validation safety posture")
    previous_raw = body["previous_live_validation_receipt"]
    previous = (
        None
        if previous_raw is None
        else _parse_artifact_identity(previous_raw, "previous live-validation receipt")
    )
    result = HostLiveValidationReceiptV1(
        checkpoint=_require_text(body["checkpoint"], "live-validation checkpoint"),
        checkpoint_ordinal=_require_int(
            body["checkpoint_ordinal"],
            "live-validation checkpoint ordinal",
            maximum=len(LIVE_VALIDATION_CHECKPOINTS) - 1,
        ),
        policy=_parse_artifact_identity(body["policy"], "live-validation policy"),
        statement=_parse_artifact_identity(
            body["statement"],
            "live-validation statement",
        ),
        signature_verification_receipt=_parse_artifact_identity(
            body["signature_verification_receipt"],
            "live-validation signature-verification receipt",
        ),
        previous_live_validation_receipt=previous,
        validator=_parse_component(body["validator"], "live-validation validator"),
        validation_run_id_sha256=_require_sha256(
            body["validation_run_id_sha256"],
            "live-validation run ID",
        ),
        validated_at_unix_ns=_require_int(
            body["validated_at_unix_ns"],
            "live-validation Unix time",
            minimum=1,
        ),
        validated_at_monotonic_ns=_require_int(
            body["validated_at_monotonic_ns"],
            "live-validation monotonic time",
            minimum=1,
        ),
        facts=_parse_facts(body["facts"], "live-validation facts"),
    )
    supplied_inventory = _require_sha256(
        body["facts_inventory_sha256"],
        "live-validation facts inventory SHA-256",
    )
    if not secrets.compare_digest(supplied_inventory, result.facts_inventory_sha256):
        _fail("live-validation facts inventory SHA-256 differs")
    return result


def validate_host_provisioning_statement_against_policy_v1(
    policy: HostTrustPolicyV1,
    statement: HostProvisioningStatementV1,
) -> None:
    """Validate structural policy/statement closure without doing cryptography."""

    _require_exact_type(policy, HostTrustPolicyV1, "host trust policy")
    _require_exact_type(statement, HostProvisioningStatementV1, "provisioning statement")
    if statement.policy != host_trust_policy_identity_v1(policy):
        _fail("provisioning statement crosswires its trust policy")
    if statement.signer_key_id != policy.signer_key_id:
        _fail("provisioning statement signer key ID differs from policy")
    if not secrets.compare_digest(
        statement.signer_public_key_sha256,
        policy.signer_public_key_sha256,
    ):
        _fail("provisioning statement signer public-key hash differs from policy")
    if statement.facts != policy.expected_facts:
        _fail("provisioning statement facts differ from the preissued policy")
    if not policy.supported_host_tuple.matches_facts(statement.facts):
        _fail("provisioning statement facts differ from the supported host tuple")
    if not policy.valid_from_unix_ns <= statement.observed_at_unix_ns <= policy.valid_until_unix_ns:
        _fail("provisioning statement observation is outside policy validity")


def validate_host_signature_verification_receipt_v1(
    policy: HostTrustPolicyV1,
    statement: HostProvisioningStatementV1,
    receipt: HostSignatureVerificationReceiptV1,
) -> None:
    """Validate receipt closure; trust remains with the independently pinned verifier."""

    validate_host_provisioning_statement_against_policy_v1(policy, statement)
    _require_exact_type(
        receipt,
        HostSignatureVerificationReceiptV1,
        "signature-verification receipt",
    )
    if receipt.policy != host_trust_policy_identity_v1(policy):
        _fail("signature-verification receipt crosswires its trust policy")
    if receipt.statement != host_provisioning_statement_identity_v1(statement):
        _fail("signature-verification receipt crosswires its provisioning statement")
    if receipt.verifier != policy.independent_verifier:
        _fail("signature-verification receipt uses an unpinned verifier")
    if receipt.signer_key_id != statement.signer_key_id:
        _fail("signature-verification receipt signer key ID differs")
    if not secrets.compare_digest(
        receipt.signer_public_key_sha256,
        statement.signer_public_key_sha256,
    ):
        _fail("signature-verification receipt public-key hash differs")
    if not secrets.compare_digest(
        receipt.signed_payload_sha256,
        statement.signed_payload_sha256,
    ):
        _fail("signature-verification receipt signed-payload hash differs")
    expected_signature_sha256 = _sha256(bytes.fromhex(statement.signature_hex))
    if not secrets.compare_digest(receipt.signature_sha256, expected_signature_sha256):
        _fail("signature-verification receipt signature hash differs")
    if receipt.verification_started_at_unix_ns < statement.observed_at_unix_ns:
        _fail("signature verification starts before the statement observation")
    if receipt.verification_started_at_monotonic_ns < statement.observed_at_monotonic_ns:
        _fail("signature verification monotonic time precedes statement observation")
    if receipt.verification_completed_at_unix_ns > policy.valid_until_unix_ns:
        _fail("signature verification completes after policy validity")


def _validate_live_receipt_semantic_closure_v1(
    policy: HostTrustPolicyV1,
    statement: HostProvisioningStatementV1,
    signature_verification_receipt: HostSignatureVerificationReceiptV1,
    receipt: HostLiveValidationReceiptV1,
    *,
    label: str,
) -> None:
    """Validate one receipt's direct closure without following predecessor links."""

    _require_exact_type(receipt, HostLiveValidationReceiptV1, label)
    if receipt.policy != host_trust_policy_identity_v1(policy):
        _fail(f"{label} crosswires its trust policy")
    if receipt.statement != host_provisioning_statement_identity_v1(statement):
        _fail(f"{label} crosswires its provisioning statement")
    expected_verification = host_signature_verification_receipt_identity_v1(
        signature_verification_receipt
    )
    if receipt.signature_verification_receipt != expected_verification:
        _fail(f"{label} crosswires signature verification")
    if receipt.validator != policy.live_validator:
        _fail(f"{label} uses an unpinned live validator")
    if receipt.facts != statement.facts:
        _fail(f"{label} reports material host-fact drift")
    if not policy.supported_host_tuple.matches_facts(receipt.facts):
        _fail(f"{label} facts differ from the supported host tuple")
    if not policy.valid_from_unix_ns <= receipt.validated_at_unix_ns <= policy.valid_until_unix_ns:
        _fail(f"{label} time is outside policy validity")
    if (
        receipt.validated_at_unix_ns
        < signature_verification_receipt.verification_completed_at_unix_ns
        or receipt.validated_at_monotonic_ns
        < signature_verification_receipt.verification_completed_at_monotonic_ns
    ):
        _fail(f"{label} precedes signature verification")


def validate_host_live_validation_receipt_v1(
    policy: HostTrustPolicyV1,
    statement: HostProvisioningStatementV1,
    signature_verification_receipt: HostSignatureVerificationReceiptV1,
    receipt: HostLiveValidationReceiptV1,
    *,
    previous_receipt: HostLiveValidationReceiptV1 | None,
) -> None:
    """Validate one live checkpoint and its exact predecessor link."""

    validate_host_signature_verification_receipt_v1(
        policy,
        statement,
        signature_verification_receipt,
    )
    _validate_live_receipt_semantic_closure_v1(
        policy,
        statement,
        signature_verification_receipt,
        receipt,
        label="live-validation receipt",
    )
    if receipt.checkpoint_ordinal == 0:
        if previous_receipt is not None or receipt.previous_live_validation_receipt is not None:
            _fail("pre-capability validation must be the live-chain genesis")
        return
    _require_exact_type(
        previous_receipt,
        HostLiveValidationReceiptV1,
        "previous live-validation receipt",
    )
    previous = cast(HostLiveValidationReceiptV1, previous_receipt)
    _validate_live_receipt_semantic_closure_v1(
        policy,
        statement,
        signature_verification_receipt,
        previous,
        label="previous live-validation receipt",
    )
    if previous.checkpoint_ordinal + 1 != receipt.checkpoint_ordinal:
        _fail("live-validation checkpoint order is not consecutive")
    if (
        receipt.previous_live_validation_receipt
        != host_live_validation_receipt_identity_v1(previous)
    ):
        _fail("live-validation previous-receipt identity differs")
    if (
        receipt.validated_at_unix_ns <= previous.validated_at_unix_ns
        or receipt.validated_at_monotonic_ns <= previous.validated_at_monotonic_ns
    ):
        _fail("live-validation chronology is not strictly increasing")


def validate_host_provisioning_trust_chain_v1(
    policy: HostTrustPolicyV1,
    statement: HostProvisioningStatementV1,
    signature_verification_receipt: HostSignatureVerificationReceiptV1,
    live_validation_receipts: tuple[HostLiveValidationReceiptV1, ...],
) -> None:
    """Validate all structural crosslinks and the exact four-checkpoint chain."""

    validate_host_signature_verification_receipt_v1(
        policy,
        statement,
        signature_verification_receipt,
    )
    if type(live_validation_receipts) is not tuple or len(live_validation_receipts) != len(
        LIVE_VALIDATION_CHECKPOINTS
    ):
        _fail("complete live-validation chain must contain the exact checkpoint tuple")
    previous: HostLiveValidationReceiptV1 | None = None
    for ordinal, receipt in enumerate(live_validation_receipts):
        _require_exact_type(receipt, HostLiveValidationReceiptV1, "live-chain receipt")
        if receipt.checkpoint != LIVE_VALIDATION_CHECKPOINTS[ordinal]:
            _fail("complete live-validation chain checkpoint order differs")
        validate_host_live_validation_receipt_v1(
            policy,
            statement,
            signature_verification_receipt,
            receipt,
            previous_receipt=previous,
        )
        previous = receipt
