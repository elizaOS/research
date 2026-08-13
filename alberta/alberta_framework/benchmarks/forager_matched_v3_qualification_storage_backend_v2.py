"""Pure production-storage metadata contract for matched Forager v3.

The artifacts in this module are fail-closed canonical metadata.  They do not
inspect or mutate a filesystem, mount a tmpfs, launch a container, enforce a
quota, relay a terminal record, or authorize execution.  In particular, a
well-formed receipt says only what independently pinned producers attest; this
module cannot create such an attestation.

The dependency graph is one way.  A score-blind phase-4 policy precedes all
runtime observations.  A phase-6 runtime intent binds the concrete pre-GO
storage boundary.  A receipt binds the host-v3 GO chain, publication reload,
relay/channel readiness, and the irreversible write seal.  Cleanup follows a
committed receipt or an operational failure.  Terminal, lifecycle, and merger
identities are deliberately outside every schema here.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final, Literal, Never, cast

STORAGE_BACKEND_CONTRACT_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.qualification_storage_backend_contract_descriptor.v2"
)
STORAGE_BACKEND_POLICY_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.storage_backend_policy.v1"
)
STORAGE_BOUNDARY_RUNTIME_INTENT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.storage_boundary_runtime_intent.v1"
)
STORAGE_BOUNDARY_RECEIPT_V2_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.storage_boundary_receipt.v2"
)
STORAGE_CLEANUP_RECONCILIATION_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.storage_cleanup_reconciliation.v1"
)
QUALIFICATION_PLAN_V3_SCHEMA_VERSION: Final = "alberta.forager_matched_v3.qualification_plan.v3"
RUNTIME_QUALIFICATION_RECEIPT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.runtime_qualification_receipt.v1"
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
NORMALIZED_PUBLICATION_WRAPPER_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.qualification_publication_commitment_wrapper.v1"
)
PUBLICATION_RELOAD_VALIDATION_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.qualification_publication_reload_validation.v1"
)
TERMINAL_RELAY_PRESEAL_ATTESTATION_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.qualification_storage_terminal_relay_preseal_attestation.v1"
)
NONSTORAGE_CHANNEL_PRESEAL_ATTESTATION_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.qualification_storage_nonstorage_channel_readiness_attestation.v1"
)
IRREVERSIBLE_WRITE_SEAL_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.qualification_storage_write_quiescence_seal.v1"
)
STORAGE_OPERATIONAL_FAILURE_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.storage_operational_failure_frontier.v1"
)
STORAGE_NAMESPACE_CLEANUP_RECEIPT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.storage_namespace_cleanup_receipt.v1"
)
STORAGE_CLEANUP_FAILURE_FRONTIER_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.storage_cleanup_failure_frontier.v1"
)
MOUNT_NAMESPACE_IDENTITY_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.storage_mount_namespace_identity.v1"
)
ROOTFS_MOUNT_IDENTITY_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.storage_rootfs_mount_identity.v1"
)
TMPFS_MOUNT_IDENTITY_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.raw.tmpfs_mountinfo_pre_go.v1"
)
TMPFS_BACKING_IDENTITY_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.storage_tmpfs_backing_identity.v1"
)
MOUNT_INVENTORY_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.raw.disk_mount_inventory_pre_go.v1"
)
PATH_INVENTORY_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.raw.disk_structural_absence_scan_pre_go.v1"
)
STORAGE_ROOT_INVENTORY_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.storage_root_inventory.v2"
)
STORAGE_FIELD_INVENTORY_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.storage_field_inventory.v2"
)
RAW_SCHEMA_INVENTORY_IDENTITY_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.storage_raw_schema_inventory.v2"
)
DOCKER_CREATE_INSPECT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.raw.docker_create_inspect_storage_projection.v1"
)
FINAL_OCI_SPEC_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.raw.final_oci_spec_storage_projection.v1"
)
CONSOLE_STDIO_INVENTORY_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.raw.console_fifo_stdio_inventory.v1"
)
DOCKER_API_OPERATION_JOURNAL_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.raw.docker_storage_api_operation_journal.v1"
)
ROOTFS_UPPERDIR_ACCESSIBILITY_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.raw.rootfs_upperdir_pre_go_baseline.v1"
)
ROOTFS_UPPERDIR_INTERVAL_DELTA_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.raw.rootfs_upperdir_go_to_seal_delta.v1"
)
DOCKER_VOLUME_INVENTORY_DELTA_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.raw.docker_volume_inventory_delta.v1"
)
DOCKER_VOLUME_INVENTORY_PRE_GO_BASELINE_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.storage_docker_volume_inventory_pre_go_baseline.v1"
)

MEASUREMENT_PRODUCER_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.qualification_storage_measurement_producer_descriptor.v2"
)
TERMINAL_RELAY_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.qualification_storage_terminal_relay_descriptor.v1"
)
NONSTORAGE_CHANNEL_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.qualification_storage_nonstorage_channel_descriptor.v1"
)
WRITE_SEAL_PRODUCER_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.qualification_storage_write_seal_producer_descriptor.v2"
)
RUNTIME_STORAGE_ESCAPE_GATE_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.runtime_storage_escape_gate_descriptor.v1"
)
NAMESPACE_CLEANUP_PRODUCER_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.namespace_cleanup_producer_descriptor.v1"
)

RESOURCE_FIELDS: Final = (
    "max_environment_interactions",
    "max_optimizer_updates",
    "max_gradient_updates",
    "max_sample_updates",
    "max_trainable_parameters",
    "max_frozen_parameters",
    "max_optimizer_state_elements",
    "max_optimizer_state_bytes",
    "max_target_copy_elements",
    "max_target_copy_bytes",
    "max_replay_capacity_transitions",
    "max_replay_peak_bytes",
    "max_rollout_storage_elements",
    "max_rollout_peak_bytes",
    "max_recurrent_carry_elements",
    "max_recurrent_carry_bytes",
    "max_rtrl_sensitivity_elements",
    "max_rtrl_sensitivity_bytes",
    "max_eligibility_elements",
    "max_eligibility_bytes",
    "max_peak_rss_bytes",
    "max_cpu_time_ns",
    "max_wall_time_ns",
    "max_temporary_peak_bytes",
    "max_disk_peak_bytes",
    "max_thread_count",
    "max_attempt_count",
    "max_failure_count",
)
RESOURCE_FIELD_ORDER_SHA256: Final = (
    "8048ec1a1402b45d8bb4c67684ee7216b242bfb6d3ed9e196c0cfb262c3b93cc"
)
CANDIDATE_ORDER_SHA256: Final = "d93aaf66053aaf9a7b1c6d268a47740078dd2c1007f7287bd80908707e40b858"
MATCHED_V3_CANDIDATE_IDS: Final = (
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
STORAGE_COMPONENT_ROLES: Final = (
    "measurement_producer",
    "terminal_relay",
    "nonstorage_channel",
    "write_seal_producer",
    "runtime_storage_escape_gate",
    "namespace_cleanup_producer",
)
STORAGE_COMPONENT_DESCRIPTOR_SCHEMAS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "measurement_producer": MEASUREMENT_PRODUCER_DESCRIPTOR_SCHEMA_VERSION,
        "terminal_relay": TERMINAL_RELAY_DESCRIPTOR_SCHEMA_VERSION,
        "nonstorage_channel": NONSTORAGE_CHANNEL_DESCRIPTOR_SCHEMA_VERSION,
        "write_seal_producer": WRITE_SEAL_PRODUCER_DESCRIPTOR_SCHEMA_VERSION,
        "runtime_storage_escape_gate": RUNTIME_STORAGE_ESCAPE_GATE_DESCRIPTOR_SCHEMA_VERSION,
        "namespace_cleanup_producer": NAMESPACE_CLEANUP_PRODUCER_DESCRIPTOR_SCHEMA_VERSION,
    }
)
STORAGE_COMPONENT_DESCRIPTOR_SCHEMA_INVENTORY: Final = tuple(
    STORAGE_COMPONENT_DESCRIPTOR_SCHEMAS[role] for role in STORAGE_COMPONENT_ROLES
)
RAW_ARTIFACT_SCHEMA_INVENTORY: Final = (
    TMPFS_MOUNT_IDENTITY_SCHEMA_VERSION,
    "alberta.forager_matched_v3.raw.tmpfs_statfs_diagnostic_samples.v1",
    "alberta.forager_matched_v3.raw.tmpfs_hard_limit_mount_mutation_closure.v1",
    MOUNT_INVENTORY_SCHEMA_VERSION,
    PATH_INVENTORY_SCHEMA_VERSION,
    "alberta.forager_matched_v3.raw.allocatable_writable_fd_lifetime_inventory.v1",
    "alberta.forager_matched_v3.raw.publication_reload.v1",
    "alberta.forager_matched_v3.raw.write_seal.v1",
    "alberta.forager_matched_v3.raw.outer_cgroup_memory_swap_max_pre_go.v1",
    "alberta.forager_matched_v3.raw.outer_cgroup_swap_counters_initial.v1",
    "alberta.forager_matched_v3.raw.outer_cgroup_swap_counters_terminal.v1",
    "alberta.forager_matched_v3.raw.outer_cgroup_memory_zswap_writeback_pre_go.v1",
    "alberta.forager_matched_v3.raw.docker_implicit_mount_inventory_pre_go.v1",
    DOCKER_CREATE_INSPECT_SCHEMA_VERSION,
    FINAL_OCI_SPEC_SCHEMA_VERSION,
    CONSOLE_STDIO_INVENTORY_SCHEMA_VERSION,
    DOCKER_API_OPERATION_JOURNAL_SCHEMA_VERSION,
    ROOTFS_UPPERDIR_ACCESSIBILITY_SCHEMA_VERSION,
    ROOTFS_UPPERDIR_INTERVAL_DELTA_SCHEMA_VERSION,
    DOCKER_VOLUME_INVENTORY_DELTA_SCHEMA_VERSION,
)
REQUIRED_EXTERNAL_ARTIFACT_SCHEMAS: Final = (
    QUALIFICATION_PLAN_V3_SCHEMA_VERSION,
    RUNTIME_QUALIFICATION_RECEIPT_SCHEMA_VERSION,
    HOST_PROVISIONING_VALIDATED_PRE_GO_PREFIX_V3_SCHEMA_VERSION,
    HOST_CASE_REQUEST_V3_SCHEMA_VERSION,
    HOST_CASE_INTENT_V3_SCHEMA_VERSION,
    HOST_READY_V3_SCHEMA_VERSION,
    HOST_OBSERVER_ANCHOR_V3_SCHEMA_VERSION,
    HOST_GO_V3_SCHEMA_VERSION,
    NORMALIZED_PUBLICATION_WRAPPER_SCHEMA_VERSION,
    PUBLICATION_RELOAD_VALIDATION_SCHEMA_VERSION,
    TERMINAL_RELAY_PRESEAL_ATTESTATION_SCHEMA_VERSION,
    NONSTORAGE_CHANNEL_PRESEAL_ATTESTATION_SCHEMA_VERSION,
    IRREVERSIBLE_WRITE_SEAL_SCHEMA_VERSION,
    STORAGE_OPERATIONAL_FAILURE_SCHEMA_VERSION,
    STORAGE_NAMESPACE_CLEANUP_RECEIPT_SCHEMA_VERSION,
    STORAGE_CLEANUP_FAILURE_FRONTIER_SCHEMA_VERSION,
    MOUNT_NAMESPACE_IDENTITY_SCHEMA_VERSION,
    ROOTFS_MOUNT_IDENTITY_SCHEMA_VERSION,
    TMPFS_BACKING_IDENTITY_SCHEMA_VERSION,
    STORAGE_ROOT_INVENTORY_SCHEMA_VERSION,
    STORAGE_FIELD_INVENTORY_SCHEMA_VERSION,
    RAW_SCHEMA_INVENTORY_IDENTITY_SCHEMA_VERSION,
    DOCKER_VOLUME_INVENTORY_PRE_GO_BASELINE_SCHEMA_VERSION,
)
if len(set(STORAGE_COMPONENT_DESCRIPTOR_SCHEMA_INVENTORY)) != len(
    STORAGE_COMPONENT_DESCRIPTOR_SCHEMA_INVENTORY
):
    raise AssertionError("storage component descriptor schemas must be distinct")
if len(set(REQUIRED_EXTERNAL_ARTIFACT_SCHEMAS)) != len(REQUIRED_EXTERNAL_ARTIFACT_SCHEMAS):
    raise AssertionError("required external artifact schemas must be distinct")
if set(REQUIRED_EXTERNAL_ARTIFACT_SCHEMAS).intersection(
    RAW_ARTIFACT_SCHEMA_INVENTORY,
    STORAGE_COMPONENT_DESCRIPTOR_SCHEMA_INVENTORY,
    (
        STORAGE_BACKEND_POLICY_SCHEMA_VERSION,
        STORAGE_BOUNDARY_RUNTIME_INTENT_SCHEMA_VERSION,
        STORAGE_BOUNDARY_RECEIPT_V2_SCHEMA_VERSION,
        STORAGE_CLEANUP_RECONCILIATION_SCHEMA_VERSION,
    ),
):
    raise AssertionError("storage descriptor schema inventories must be disjoint")
TMPFS_AGGREGATE_ROOT: Final = "/run/alberta"
PERSISTENT_STORAGE_SCOPE: Final = "candidate_addressable_allocatable_persistent_storage"
PERSISTENT_STORAGE_MEASUREMENT_INTERVAL: Final = (
    "host_go_commit_inclusive_through_irreversible_write_seal_exclusive"
)
TRUSTED_RUNTIME_BOOKKEEPING_EXCLUSIONS: Final = (
    "candidate_inaccessible_docker_daemon_control_state",
    "candidate_inaccessible_graph_driver_control_metadata",
    "candidate_inaccessible_generated_read_only_etc_backing_files",
    "post_seal_host_archival",
)
DOCKER_IMPLICIT_CONFIG_BIND_PATHS: Final = (
    "/etc/hostname",
    "/etc/hosts",
    "/etc/resolv.conf",
)
FIELD24_POSITION: Final = 24
FIELD25_POSITION: Final = 25
_ZERO_SHA256: Final = "0" * 64
DESCRIPTOR_SERIALIZED_SELF_PIN_SENTINEL: Final = _ZERO_SHA256
PINNED_STORAGE_BACKEND_V2_DESCRIPTOR_FILE_SHA256: Final = _ZERO_SHA256
PINNED_STORAGE_BACKEND_V2_DESCRIPTOR_BODY_SHA256: Final = _ZERO_SHA256

STORAGE_POLICY_STATUS: Final = "phase4_pre_runtime_score_blind_storage_policy_non_authorizing"
STORAGE_RUNTIME_INTENT_STATUS: Final = "phase6_pre_go_runtime_storage_intent_non_authorizing"
STORAGE_RECEIPT_STATUS: Final = "post_seal_storage_receipt_non_authorizing"
STORAGE_CLEANUP_STATUS: Final = "post_receipt_or_failure_deletion_only_cleanup_non_authorizing"
STORAGE_DESCRIPTOR_STATUS: Final = "pure_source_only_unfinalized_uninvoked_no_production_receipt"

STORAGE_PHASE_SPLIT: Final = (
    "phase4_policy_before_runtime",
    "phase6_runtime_intent_before_go",
    "post_go_receipt_after_irreversible_seal",
    "post_receipt_or_failure_deletion_only_cleanup",
)
STORAGE_CHRONOLOGY: Final = (
    "policy",
    "runtime_intent",
    "host_v3_request_intent_ready_anchor_go",
    "publication_wrapper",
    "read_only_reload_validation",
    "terminal_relay_preseal_attestation",
    "nonstorage_channel_preseal_attestation",
    "irreversible_write_seal",
    "storage_receipt",
    "cleanup_reconciliation",
)

_MAX_ARTIFACT_BYTES: Final = 4 * 1024 * 1024
_MAX_JSON_DEPTH: Final = 64
_MAX_JSON_NODES: Final = 100_000
_MAX_TEXT_LENGTH: Final = 16_384
_MAX_INTEGER: Final = 2**63 - 1
_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_IMAGE_ID_RE: Final = re.compile(r"sha256:[0-9a-f]{64}\Z")
_IDENTIFIER_RE: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}\Z")

_DESCRIPTOR_BODY_FIELD: Final = "storage_backend_contract_descriptor_body_sha256"
_POLICY_BODY_FIELD: Final = "storage_backend_policy_body_sha256"
_INTENT_BODY_FIELD: Final = "storage_boundary_runtime_intent_body_sha256"
_RECEIPT_BODY_FIELD: Final = "storage_boundary_receipt_v2_body_sha256"
_CLEANUP_BODY_FIELD: Final = "storage_cleanup_reconciliation_body_sha256"

_FORBIDDEN_PHASE4_KEYS: Final = frozenset(
    {
        "actual_container_id",
        "container_id_commitment_sha256",
        "container_name",
        "go",
        "host_go",
        "host_ready",
        "mount_namespace",
        "observed_peak",
        "receipt",
        "runtime",
        "runtime_intent",
        "seal",
    }
)
_FORBIDDEN_REVERSE_KEYS: Final = frozenset(
    {
        "evaluation_receipt",
        "evaluator",
        "host_success",
        "issuance_receipt",
        "issuer",
        "lifecycle",
        "merger",
        "resource_merger",
        "terminal",
        "terminal_receipt",
    }
)

SOURCE_ONLY_CAPABILITIES: Final[Mapping[str, bool]] = MappingProxyType(
    {
        "container_control": False,
        "filesystem_access": False,
        "mount_control": False,
        "network_access": False,
        "process_control": False,
        "quota_enforcement": False,
        "storage_measurement": False,
        "terminal_relay": False,
        "write_seal_execution": False,
    }
)
SOURCE_ONLY_READINESS: Final[Mapping[str, bool]] = MappingProxyType(
    {
        "execution_ready": False,
        "measurement_ready": False,
        "producer_schema_closure_complete": False,
        "production_ready": False,
        "qualification_ready": False,
        "storage_backend_ready": False,
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
        "foreign_schema_contracts_validated": False,
        "promotion_allowed": False,
        "resource_matched": False,
        "scientific_evidence_created": False,
        "storage_peak_established_by_this_module": False,
    }
)


class ForagerMatchedV3QualificationStorageBackendV2Error(ValueError):
    """One storage-backend-v2 artifact failed closed."""


def _fail(message: str) -> Never:
    raise ForagerMatchedV3QualificationStorageBackendV2Error(message)


def _translate_current_state_error(label: str, validator: Callable[[], None]) -> None:
    """Run one in-memory validator and keep this protocol's error boundary exact."""

    try:
        validator()
    except ForagerMatchedV3QualificationStorageBackendV2Error:
        raise
    except Exception as exc:
        raise ForagerMatchedV3QualificationStorageBackendV2Error(
            f"{label} current state is invalid"
        ) from exc


def _require_sha256(value: object, label: str, *, permit_zero: bool = False) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        _fail(f"{label} must be one lowercase SHA-256")
    if not permit_zero and value == "0" * 64:
        _fail(f"{label} must be nonzero")
    return value


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


def _require_bool(value: object, label: str) -> bool:
    if type(value) is not bool:
        _fail(f"{label} must be one exact boolean")
    return value


def _require_text(value: object, label: str) -> str:
    if (
        type(value) is not str
        or not value
        or len(value) > _MAX_TEXT_LENGTH
        or any(ord(character) < 0x20 or ord(character) > 0x7E for character in value)
    ):
        _fail(f"{label} must be bounded nonempty printable ASCII")
    return value


def _require_identifier(value: object, label: str) -> str:
    text = _require_text(value, label)
    if _IDENTIFIER_RE.fullmatch(text) is None:
        _fail(f"{label} must be one portable identifier")
    return text


def _require_path(value: object, label: str) -> str:
    path = _require_text(value, label)
    if not path.startswith("/") or path.startswith("//") or (path != "/" and path.endswith("/")):
        _fail(f"{label} must be one canonical absolute path")
    if any(part in {"", ".", ".."} for part in path.split("/")[1:]):
        _fail(f"{label} contains a forbidden path component")
    return path


def _require_image_id(value: object, label: str) -> str:
    if (
        type(value) is not str
        or _IMAGE_ID_RE.fullmatch(value) is None
        or value == "sha256:" + "0" * 64
    ):
        _fail(f"{label} must be one nonzero immutable sha256 image ID")
    return value


def _require_exact_keys(value: object, expected: frozenset[str], label: str) -> dict[str, Any]:
    if type(value) is not dict or frozenset(value) != expected:
        _fail(f"{label} keys differ")
    return cast(dict[str, Any], value)


def _string_tuple_from_json(value: object, label: str) -> tuple[str, ...]:
    if type(value) is not list:
        _fail(f"{label} must be one JSON array")
    result: list[str] = []
    for index, item in enumerate(cast(list[object], value)):
        result.append(_require_text(item, f"{label} item {index}"))
    return tuple(result)


def _int_tuple_from_json(value: object, label: str) -> tuple[int, ...]:
    if type(value) is not list:
        _fail(f"{label} must be one JSON array")
    return tuple(
        _require_int(item, f"{label} item {index}")
        for index, item in enumerate(cast(list[object], value))
    )


def _reject_constant(value: str) -> Never:
    _fail(f"JSON contains forbidden constant {value!r}")


def _reject_float(value: str) -> Never:
    _fail(f"JSON contains forbidden float {value!r}")


def _parse_int(value: str) -> int:
    if len(value.lstrip("-")) > 19:
        _fail("JSON integer exceeds its lexical bound")
    parsed = int(value, 10)
    if not -_MAX_INTEGER <= parsed <= _MAX_INTEGER:
        _fail("JSON integer exceeds its value bound")
    return parsed


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if type(key) is not str or key in result:
            _fail("JSON contains a duplicate or non-text key")
        result[key] = value
    return result


def _assert_plain_unaliased_json(value: object) -> None:
    seen: set[int] = set()
    nodes = 0

    def visit(item: object, depth: int) -> None:
        nonlocal nodes
        nodes += 1
        if depth > _MAX_JSON_DEPTH or nodes > _MAX_JSON_NODES:
            _fail("JSON structure exceeds its depth or node bound")
        if item is None or type(item) is bool:
            return
        if type(item) is int:
            _require_int(item, "JSON integer", minimum=-_MAX_INTEGER)
            return
        if type(item) is str:
            if len(item) > _MAX_TEXT_LENGTH or any(
                ord(character) < 0x20 or ord(character) > 0x7E for character in item
            ):
                _fail("JSON strings must be bounded printable ASCII")
            return
        if type(item) not in {dict, list}:
            _fail("JSON contains a non-plain value")
        identity = id(item)
        if identity in seen:
            _fail("JSON contains an alias or cycle")
        seen.add(identity)
        if type(item) is list:
            for child in cast(list[object], item):
                visit(child, depth + 1)
        else:
            for key, child in cast(dict[object, object], item).items():
                if type(key) is not str:
                    _fail("JSON keys must be exact strings")
                visit(key, depth + 1)
                visit(child, depth + 1)

    visit(value, 0)


def _reject_keys(value: object, forbidden: frozenset[str], label: str) -> None:
    if type(value) is list:
        for child in cast(list[object], value):
            _reject_keys(child, forbidden, label)
        return
    if type(value) is not dict:
        return
    for key, child in cast(dict[str, object], value).items():
        if key in forbidden:
            _fail(f"{label} key {key!r} is forbidden")
        _reject_keys(child, forbidden, label)


def canonical_storage_backend_json_bytes(value: object, *, final_lf: bool = True) -> bytes:
    """Return strict sorted printable-ASCII JSON, optionally with one final LF."""

    _require_bool(final_lf, "canonical JSON final-LF selector")
    _assert_plain_unaliased_json(value)
    try:
        raw = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (RecursionError, TypeError, ValueError) as exc:
        raise ForagerMatchedV3QualificationStorageBackendV2Error(
            "value is not canonical storage-backend JSON"
        ) from exc
    if final_lf:
        raw += b"\n"
    if not 1 <= len(raw) <= _MAX_ARTIFACT_BYTES:
        _fail("canonical JSON exceeds its byte bound")
    return raw


def decode_canonical_storage_backend_json(raw: bytes) -> dict[str, Any]:
    """Decode one canonical full-file object and reject noncanonical encodings."""

    if type(raw) is not bytes or not 1 <= len(raw) <= _MAX_ARTIFACT_BYTES:
        _fail("artifact bytes violate their bound")
    try:
        value = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_object_without_duplicates,
            parse_constant=_reject_constant,
            parse_float=_reject_float,
            parse_int=_parse_int,
        )
    except ForagerMatchedV3QualificationStorageBackendV2Error:
        raise
    except (RecursionError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ForagerMatchedV3QualificationStorageBackendV2Error(
            "artifact is not strict ASCII JSON"
        ) from exc
    if type(value) is not dict:
        _fail("artifact root must be one object")
    result = cast(dict[str, Any], value)
    _assert_plain_unaliased_json(result)
    _reject_keys(result, _FORBIDDEN_REVERSE_KEYS, "reverse binding")
    if not hmac.compare_digest(canonical_storage_backend_json_bytes(result), raw):
        _fail("artifact bytes are not canonical or lack exactly one final LF")
    return result


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _body_sha256(body: Mapping[str, Any]) -> str:
    return _sha256(canonical_storage_backend_json_bytes(dict(body), final_lf=False))


def _file_dict(body: Mapping[str, Any], digest_field: str) -> dict[str, Any]:
    result = dict(body)
    result[digest_field] = _body_sha256(body)
    return result


def _validate_file(
    raw: bytes,
    expected_file_sha256: object,
    expected_body_sha256: object,
    digest_field: str,
    label: str,
) -> dict[str, Any]:
    if type(raw) is not bytes or not 1 <= len(raw) <= _MAX_ARTIFACT_BYTES:
        _fail(f"{label} artifact bytes violate their bound")
    expected = _require_sha256(expected_file_sha256, f"{label} caller file pin")
    expected_body = _require_sha256(expected_body_sha256, f"{label} caller BODY pin")
    if not hmac.compare_digest(_sha256(raw), expected):
        _fail(f"{label} file differs from its caller pin")
    value = decode_canonical_storage_backend_json(raw)
    supplied = _require_sha256(value.get(digest_field), f"{label} BODY")
    body = dict(value)
    body.pop(digest_field, None)
    if not hmac.compare_digest(supplied, _body_sha256(body)):
        _fail(f"{label} BODY digest differs")
    if not hmac.compare_digest(supplied, expected_body):
        _fail(f"{label} BODY differs from its caller pin")
    return body


@dataclass(frozen=True, slots=True)
class ArtifactIdentityV1:
    """One canonical artifact identity; this module never loads its bytes."""

    schema_version: str
    file_sha256: str
    body_sha256: str

    def __post_init__(self) -> None:
        _require_identifier(self.schema_version, "artifact schema")
        _require_sha256(self.file_sha256, "artifact file")
        _require_sha256(self.body_sha256, "artifact BODY")
        if hmac.compare_digest(self.file_sha256, self.body_sha256):
            _fail("artifact FILE and BODY identities alias")
        _assert_plain_unaliased_json(self.to_dict())

    def to_dict(self) -> dict[str, str]:
        return {
            "body_sha256": self.body_sha256,
            "file_sha256": self.file_sha256,
            "schema_version": self.schema_version,
        }


def _require_artifact_identity(
    artifact: object,
    label: str,
    *,
    expected_schema: str | None = None,
) -> ArtifactIdentityV1:
    if type(artifact) is not ArtifactIdentityV1:
        _fail(f"{label} identity type differs")
    exact = artifact
    _translate_current_state_error(
        label,
        lambda: ArtifactIdentityV1.__post_init__(exact),
    )
    if expected_schema is not None and not hmac.compare_digest(
        exact.schema_version,
        expected_schema,
    ):
        _fail(f"{label} identity schema differs")
    return exact


def _require_distinct_artifact_digests(
    artifacts: tuple[ArtifactIdentityV1, ...],
    label: str,
) -> None:
    exact = tuple(
        _require_artifact_identity(artifact, f"{label} artifact {index}")
        for index, artifact in enumerate(artifacts)
    )
    digests = tuple(
        digest
        for artifact in exact
        for digest in (artifact.file_sha256, artifact.body_sha256)
    )
    if len(set(digests)) != len(digests):
        _fail(f"{label} FILE or BODY identities alias")


@dataclass(frozen=True, slots=True)
class PinnedStorageComponentIdentityV1:
    """One of six independently pinned, mutually distinct storage roles."""

    role: Literal[
        "measurement_producer",
        "terminal_relay",
        "nonstorage_channel",
        "write_seal_producer",
        "runtime_storage_escape_gate",
        "namespace_cleanup_producer",
    ]
    descriptor_schema_version: str
    descriptor_file_sha256: str
    descriptor_body_sha256: str
    source_sha256: str

    def __post_init__(self) -> None:
        role = _require_identifier(self.role, "component role")
        if role not in STORAGE_COMPONENT_ROLES:
            _fail("component role differs")
        descriptor_schema = _require_identifier(
            self.descriptor_schema_version,
            "component descriptor schema",
        )
        if descriptor_schema != STORAGE_COMPONENT_DESCRIPTOR_SCHEMAS[role]:
            _fail("component descriptor schema differs for its role")
        _require_sha256(self.descriptor_file_sha256, "component descriptor file")
        _require_sha256(self.descriptor_body_sha256, "component descriptor BODY")
        _require_sha256(self.source_sha256, "component source")
        if (
            len(
                {
                    self.descriptor_file_sha256,
                    self.descriptor_body_sha256,
                    self.source_sha256,
                }
            )
            != 3
        ):
            _fail("component source, descriptor FILE, or descriptor BODY identities alias")
        _assert_plain_unaliased_json(self.to_dict())

    def to_dict(self) -> dict[str, str]:
        return {
            "descriptor_body_sha256": self.descriptor_body_sha256,
            "descriptor_file_sha256": self.descriptor_file_sha256,
            "descriptor_schema_version": self.descriptor_schema_version,
            "role": self.role,
            "source_sha256": self.source_sha256,
        }


def _require_component_identity(
    component: object,
    label: str,
    *,
    expected_role: str | None = None,
) -> PinnedStorageComponentIdentityV1:
    if type(component) is not PinnedStorageComponentIdentityV1:
        _fail(f"{label} component type differs")
    exact = component
    _translate_current_state_error(
        label,
        lambda: PinnedStorageComponentIdentityV1.__post_init__(exact),
    )
    if expected_role is not None and not hmac.compare_digest(exact.role, expected_role):
        _fail(f"{label} producer role differs")
    return exact


def _validate_components(components: object) -> tuple[PinnedStorageComponentIdentityV1, ...]:
    if type(components) is not tuple or len(components) != len(STORAGE_COMPONENT_ROLES):
        _fail("storage components must be one exact ordered role tuple")
    exact = cast(tuple[object, ...], components)
    if any(type(component) is not PinnedStorageComponentIdentityV1 for component in exact):
        _fail("storage component type differs")
    typed = tuple(
        _require_component_identity(component, f"storage component {index}")
        for index, component in enumerate(exact)
    )
    if tuple(component.role for component in typed) != STORAGE_COMPONENT_ROLES:
        _fail("storage component role order differs")
    if len({component.source_sha256 for component in typed}) != len(STORAGE_COMPONENT_ROLES):
        _fail("storage component roles alias one source implementation")
    if len({component.descriptor_file_sha256 for component in typed}) != len(
        STORAGE_COMPONENT_ROLES
    ):
        _fail("storage component roles alias one descriptor FILE")
    if len({component.descriptor_body_sha256 for component in typed}) != len(
        STORAGE_COMPONENT_ROLES
    ):
        _fail("storage component roles alias one descriptor BODY")
    all_component_hashes = (
        tuple(component.source_sha256 for component in typed)
        + tuple(component.descriptor_file_sha256 for component in typed)
        + tuple(component.descriptor_body_sha256 for component in typed)
    )
    if len(set(all_component_hashes)) != len(all_component_hashes):
        _fail("storage component source, descriptor FILE, or descriptor BODY identities alias")
    return typed


def _require_component_role(
    component: object,
    role: str,
    label: str,
) -> PinnedStorageComponentIdentityV1:
    return _require_component_identity(component, label, expected_role=role)


def _candidate_projection(
    ordinal: object,
    candidate_id: object,
    candidate_family: object,
    qualification_case_id: object,
) -> tuple[int, str, str, str]:
    exact_ordinal = _require_int(
        ordinal,
        "case ordinal",
        maximum=len(MATCHED_V3_CANDIDATE_IDS) - 1,
    )
    expected_candidate = MATCHED_V3_CANDIDATE_IDS[exact_ordinal]
    exact_candidate = _require_identifier(candidate_id, "candidate ID")
    if exact_candidate != expected_candidate:
        _fail("case ordinal and candidate ID differ from the frozen order")
    if exact_candidate.startswith("adapted_"):
        expected_family = "adapter"
    elif exact_candidate.startswith(("external_", "isolated_", "random_", "search_")):
        expected_family = "external"
    else:
        expected_family = "local"
    exact_family = _require_identifier(candidate_family, "candidate family")
    if exact_family != expected_family:
        _fail("candidate family differs")
    expected_case = f"qualification_{exact_ordinal:02d}_{exact_candidate}"
    exact_case = _require_identifier(qualification_case_id, "qualification case ID")
    if exact_case != expected_case:
        _fail("qualification case ID differs")
    return exact_ordinal, exact_candidate, exact_family, exact_case


@dataclass(frozen=True, slots=True)
class StorageBackendPolicyV1:
    """Phase-4 score-blind policy, frozen before any case runtime fact exists."""

    qualification_plan: ArtifactIdentityV1
    max_temporary_peak_bytes: int
    components: tuple[PinnedStorageComponentIdentityV1, ...]
    schema_version: str = STORAGE_BACKEND_POLICY_SCHEMA_VERSION
    status: str = STORAGE_POLICY_STATUS
    resource_field_order_sha256: str = RESOURCE_FIELD_ORDER_SHA256
    candidate_order_sha256: str = CANDIDATE_ORDER_SHA256
    temporary_field_position: int = FIELD24_POSITION
    disk_field_position: int = FIELD25_POSITION
    aggregate_root_path: str = TMPFS_AGGREGATE_ROOT
    aggregate_root_count: int = 1
    aggregate_root_case_exclusive: bool = True
    aggregate_root_application_writable: bool = True
    aggregate_root_allocatable: bool = True
    tmpfs_hard_size_limit_bytes: int = 0
    tmpfs_exact_size_readback_required: bool = True
    tmpfs_noswap_required: bool = True
    tmpfs_mount_mutation_closure_required: bool = True
    disk_published_value_bytes: int = 0
    disk_scope: str = "structurally_absent"
    persistent_storage_scope: str = PERSISTENT_STORAGE_SCOPE
    persistent_storage_measurement_interval: str = PERSISTENT_STORAGE_MEASUREMENT_INTERVAL
    trusted_runtime_bookkeeping_exclusions: tuple[str, ...] = TRUSTED_RUNTIME_BOOKKEEPING_EXCLUSIONS
    rootfs_read_only: bool = True
    rootfs_copy_up_enabled: bool = False
    container_log_driver: str = "none"
    application_bind_mount_count: int = 0
    volume_count: int = 0
    added_device_count: int = 0
    network_enabled: bool = False
    inherited_allocatable_storage_fd_count: int = 0
    alternate_writable_path_count: int = 0
    application_writable_tmpfs_count: int = 1
    docker_implicit_config_bind_paths: tuple[str, ...] = DOCKER_IMPLICIT_CONFIG_BIND_PATHS
    implicit_readonly_etc_bind_count: int = 3
    writable_persistent_mount_count: int = 0
    writable_persistent_fd_count: int = 0
    candidate_stdio_transport_count: int = 0
    docker_implicit_mount_inventory_required: bool = True
    docker_implicit_mounts_all_read_only: bool = True
    docker_implicit_mounts_all_nonallocatable: bool = True
    docker_implicit_mounts_have_application_writable_path: bool = False
    docker_ipc_mode: str = "none"
    docker_shm_mount_present: bool = False
    docker_tty_enabled: bool = False
    docker_stdin_open: bool = False
    image_declared_volume_count: int = 0
    docker_exec_permitted: bool = False
    docker_archive_api_permitted: bool = False
    docker_api_candidate_accessible: bool = False
    container_console_or_fifo_candidate_accessible: bool = False
    default_device_inventory_required: bool = True
    default_devices_can_allocate_storage: bool = False
    device_open_ioctl_confinement_required: bool = True
    memfd_posix_or_sysv_shm_permitted: bool = False
    post_go_mount_mutation_permitted: bool = False
    candidate_cgroup_mutation_permitted: bool = False
    rootfs_upperdir_candidate_writable: bool = False
    daemon_runtime_storage_candidate_accessible: bool = False
    host_archival_candidate_accessible: bool = False
    outer_cgroup_memory_swap_max_bytes: int = 0
    outer_cgroup_memory_zswap_writeback_enabled: bool = False
    architecture: str = "worker_exit_then_isolated_terminal_relay"
    raw_artifact_schema_inventory: tuple[str, ...] = RAW_ARTIFACT_SCHEMA_INVENTORY
    measurement_failure_policy: str = "fail_closed_no_value_no_retry"
    cleanup_policy: str = "deletion_only_after_receipt_or_failure"
    phase_number: int = 4
    runtime_observations_present: bool = False

    def __post_init__(self) -> None:
        if self.schema_version != STORAGE_BACKEND_POLICY_SCHEMA_VERSION:
            _fail("storage policy schema differs")
        if self.status != STORAGE_POLICY_STATUS:
            _fail("storage policy status differs")
        _require_artifact_identity(
            self.qualification_plan,
            "qualification plan",
            expected_schema=QUALIFICATION_PLAN_V3_SCHEMA_VERSION,
        )
        if self.resource_field_order_sha256 != RESOURCE_FIELD_ORDER_SHA256:
            _fail("resource field order hash differs")
        if self.candidate_order_sha256 != CANDIDATE_ORDER_SHA256:
            _fail("candidate order hash differs")
        ceiling = _require_int(self.max_temporary_peak_bytes, "temporary hard ceiling", minimum=1)
        if (
            _require_int(self.tmpfs_hard_size_limit_bytes, "tmpfs hard size limit", minimum=1)
            != ceiling
        ):
            _fail("tmpfs hard size limit must equal field-24 hard ceiling")
        if not all(
            (
                self.tmpfs_exact_size_readback_required is True,
                self.tmpfs_noswap_required is True,
                self.tmpfs_mount_mutation_closure_required is True,
            )
        ):
            _fail("tmpfs pre-GO hard-limit, noswap, or mount-mutation policy differs")
        _require_int(
            self.temporary_field_position,
            "temporary field position",
            minimum=FIELD24_POSITION,
            maximum=FIELD24_POSITION,
        )
        _require_int(
            self.disk_field_position,
            "disk field position",
            minimum=FIELD25_POSITION,
            maximum=FIELD25_POSITION,
        )
        if _require_path(self.aggregate_root_path, "aggregate root") != TMPFS_AGGREGATE_ROOT:
            _fail("aggregate root must be /run/alberta")
        if (
            _require_int(self.aggregate_root_count, "aggregate root count", minimum=1, maximum=1)
            != 1
            or self.aggregate_root_case_exclusive is not True
            or self.aggregate_root_application_writable is not True
            or self.aggregate_root_allocatable is not True
        ):
            _fail("exactly one case-exclusive tmpfs aggregate root is required")
        _require_int(self.disk_published_value_bytes, "disk published value", maximum=0)
        if (
            self.disk_scope != "structurally_absent"
            or self.persistent_storage_scope != PERSISTENT_STORAGE_SCOPE
            or self.persistent_storage_measurement_interval
            != PERSISTENT_STORAGE_MEASUREMENT_INTERVAL
            or self.trusted_runtime_bookkeeping_exclusions != TRUSTED_RUNTIME_BOOKKEEPING_EXCLUSIONS
        ):
            _fail("field 25 requires exact structural absence")
        expected_topology = (
            self.rootfs_read_only is True,
            self.rootfs_copy_up_enabled is False,
            self.container_log_driver == "none",
            self.application_bind_mount_count == 0,
            self.volume_count == 0,
            self.added_device_count == 0,
            self.network_enabled is False,
            self.inherited_allocatable_storage_fd_count == 0,
            self.alternate_writable_path_count == 0,
        )
        if not all(expected_topology):
            _fail("storage policy admits a writable disk or transport escape")
        for value, label in (
            (self.application_bind_mount_count, "application bind mount count"),
            (self.volume_count, "volume count"),
            (self.added_device_count, "added device count"),
            (
                self.inherited_allocatable_storage_fd_count,
                "inherited allocatable-storage FD count",
            ),
            (self.alternate_writable_path_count, "alternate path count"),
        ):
            _require_int(value, label, maximum=0)
        _require_int(
            self.application_writable_tmpfs_count,
            "application-writable tmpfs count",
            minimum=1,
            maximum=1,
        )
        if self.docker_implicit_config_bind_paths != DOCKER_IMPLICIT_CONFIG_BIND_PATHS:
            _fail("Docker implicit configuration bind paths differ")
        _require_int(
            self.implicit_readonly_etc_bind_count,
            "implicit read-only /etc bind count",
            minimum=3,
            maximum=3,
        )
        for value, label in (
            (self.writable_persistent_mount_count, "writable persistent mount count"),
            (self.writable_persistent_fd_count, "writable persistent FD count"),
            (self.candidate_stdio_transport_count, "candidate stdio transport count"),
        ):
            _require_int(value, label, maximum=0)
        if not all(
            (
                self.docker_implicit_mount_inventory_required is True,
                self.docker_implicit_mounts_all_read_only is True,
                self.docker_implicit_mounts_all_nonallocatable is True,
                self.docker_implicit_mounts_have_application_writable_path is False,
            )
        ):
            _fail("Docker implicit mounts are not closed as nonapplication surfaces")
        if self.docker_ipc_mode != "none":
            _fail("Docker IPC mode must be exact none")
        _require_int(self.image_declared_volume_count, "image-declared volume count", maximum=0)
        if not all(
            (
                self.docker_shm_mount_present is False,
                self.docker_tty_enabled is False,
                self.docker_stdin_open is False,
                self.docker_exec_permitted is False,
                self.docker_archive_api_permitted is False,
                self.docker_api_candidate_accessible is False,
                self.container_console_or_fifo_candidate_accessible is False,
                self.default_device_inventory_required is True,
                self.default_devices_can_allocate_storage is False,
                self.device_open_ioctl_confinement_required is True,
                self.memfd_posix_or_sysv_shm_permitted is False,
                self.post_go_mount_mutation_permitted is False,
                self.candidate_cgroup_mutation_permitted is False,
                self.rootfs_upperdir_candidate_writable is False,
                self.daemon_runtime_storage_candidate_accessible is False,
                self.host_archival_candidate_accessible is False,
            )
        ):
            _fail("Docker implicit storage, API, device, or namespace escape policy differs")
        _require_int(
            self.outer_cgroup_memory_swap_max_bytes,
            "outer-cgroup memory.swap.max",
            maximum=0,
        )
        if self.outer_cgroup_memory_zswap_writeback_enabled is not False:
            _fail("outer-cgroup memory.zswap.writeback must be disabled before GO")
        if self.architecture != "worker_exit_then_isolated_terminal_relay":
            _fail("storage architecture differs")
        _validate_components(self.components)
        if self.raw_artifact_schema_inventory != RAW_ARTIFACT_SCHEMA_INVENTORY:
            _fail("raw artifact schema inventory differs")
        if self.measurement_failure_policy != "fail_closed_no_value_no_retry":
            _fail("measurement failure policy differs")
        if self.cleanup_policy != "deletion_only_after_receipt_or_failure":
            _fail("cleanup policy differs")
        _require_int(self.phase_number, "storage policy phase", minimum=4, maximum=4)
        if self.runtime_observations_present is not False:
            _fail("phase-4 policy contains a runtime observation")
        _assert_plain_unaliased_json(self.to_body_dict())

    def to_body_dict(self) -> dict[str, Any]:
        return {
            "aggregate_root_case_exclusive": self.aggregate_root_case_exclusive,
            "aggregate_root_allocatable": self.aggregate_root_allocatable,
            "aggregate_root_application_writable": self.aggregate_root_application_writable,
            "aggregate_root_count": self.aggregate_root_count,
            "aggregate_root_path": self.aggregate_root_path,
            "alternate_writable_path_count": self.alternate_writable_path_count,
            "application_writable_tmpfs_count": self.application_writable_tmpfs_count,
            "architecture": self.architecture,
            "application_bind_mount_count": self.application_bind_mount_count,
            "candidate_order_sha256": self.candidate_order_sha256,
            "cleanup_policy": self.cleanup_policy,
            "components": [component.to_dict() for component in self.components],
            "container_log_driver": self.container_log_driver,
            "added_device_count": self.added_device_count,
            "disk_field_position": self.disk_field_position,
            "disk_published_value_bytes": self.disk_published_value_bytes,
            "disk_scope": self.disk_scope,
            "persistent_storage_scope": self.persistent_storage_scope,
            "persistent_storage_measurement_interval": (
                self.persistent_storage_measurement_interval
            ),
            "trusted_runtime_bookkeeping_exclusions": list(
                self.trusted_runtime_bookkeeping_exclusions
            ),
            "inherited_allocatable_storage_fd_count": (self.inherited_allocatable_storage_fd_count),
            "max_temporary_peak_bytes": self.max_temporary_peak_bytes,
            "measurement_failure_policy": self.measurement_failure_policy,
            "network_enabled": self.network_enabled,
            "docker_implicit_mounts_have_application_writable_path": (
                self.docker_implicit_mounts_have_application_writable_path
            ),
            "docker_implicit_mount_inventory_required": (
                self.docker_implicit_mount_inventory_required
            ),
            "docker_implicit_mounts_all_nonallocatable": (
                self.docker_implicit_mounts_all_nonallocatable
            ),
            "docker_implicit_mounts_all_read_only": self.docker_implicit_mounts_all_read_only,
            "docker_implicit_config_bind_paths": list(self.docker_implicit_config_bind_paths),
            "implicit_readonly_etc_bind_count": self.implicit_readonly_etc_bind_count,
            "writable_persistent_mount_count": self.writable_persistent_mount_count,
            "writable_persistent_fd_count": self.writable_persistent_fd_count,
            "candidate_stdio_transport_count": self.candidate_stdio_transport_count,
            "docker_ipc_mode": self.docker_ipc_mode,
            "docker_shm_mount_present": self.docker_shm_mount_present,
            "docker_tty_enabled": self.docker_tty_enabled,
            "docker_stdin_open": self.docker_stdin_open,
            "image_declared_volume_count": self.image_declared_volume_count,
            "docker_exec_permitted": self.docker_exec_permitted,
            "docker_archive_api_permitted": self.docker_archive_api_permitted,
            "docker_api_candidate_accessible": self.docker_api_candidate_accessible,
            "container_console_or_fifo_candidate_accessible": (
                self.container_console_or_fifo_candidate_accessible
            ),
            "default_device_inventory_required": self.default_device_inventory_required,
            "default_devices_can_allocate_storage": self.default_devices_can_allocate_storage,
            "device_open_ioctl_confinement_required": (self.device_open_ioctl_confinement_required),
            "memfd_posix_or_sysv_shm_permitted": self.memfd_posix_or_sysv_shm_permitted,
            "post_go_mount_mutation_permitted": self.post_go_mount_mutation_permitted,
            "candidate_cgroup_mutation_permitted": self.candidate_cgroup_mutation_permitted,
            "rootfs_upperdir_candidate_writable": self.rootfs_upperdir_candidate_writable,
            "daemon_runtime_storage_candidate_accessible": (
                self.daemon_runtime_storage_candidate_accessible
            ),
            "host_archival_candidate_accessible": self.host_archival_candidate_accessible,
            "outer_cgroup_memory_swap_max_bytes": self.outer_cgroup_memory_swap_max_bytes,
            "outer_cgroup_memory_zswap_writeback_enabled": (
                self.outer_cgroup_memory_zswap_writeback_enabled
            ),
            "phase_number": self.phase_number,
            "qualification_plan": self.qualification_plan.to_dict(),
            "raw_artifact_schema_inventory": list(self.raw_artifact_schema_inventory),
            "resource_field_order_sha256": self.resource_field_order_sha256,
            "rootfs_copy_up_enabled": self.rootfs_copy_up_enabled,
            "rootfs_read_only": self.rootfs_read_only,
            "runtime_observations_present": self.runtime_observations_present,
            "schema_version": self.schema_version,
            "status": self.status,
            "temporary_field_position": self.temporary_field_position,
            "tmpfs_hard_size_limit_bytes": self.tmpfs_hard_size_limit_bytes,
            "tmpfs_exact_size_readback_required": self.tmpfs_exact_size_readback_required,
            "tmpfs_mount_mutation_closure_required": (self.tmpfs_mount_mutation_closure_required),
            "tmpfs_noswap_required": self.tmpfs_noswap_required,
            "volume_count": self.volume_count,
        }


def _validate_current_storage_backend_policy_v1(
    policy: object,
) -> StorageBackendPolicyV1:
    if type(policy) is not StorageBackendPolicyV1:
        _fail("storage policy type differs")
    exact = policy
    _translate_current_state_error(
        "storage policy",
        lambda: StorageBackendPolicyV1.__post_init__(exact),
    )
    return exact


@dataclass(frozen=True, slots=True)
class StorageBoundaryRuntimeIntentV1:
    """Phase-6 exact pre-GO runtime storage-boundary commitment."""

    campaign_id: str
    case_ordinal: int
    candidate_id: str
    candidate_family: str
    qualification_case_id: str
    qualification_plan: ArtifactIdentityV1
    policy: ArtifactIdentityV1
    image_id: str
    runtime_qualification_receipt: ArtifactIdentityV1
    host_provisioning_v3_validated_pre_go_prefix: ArtifactIdentityV1
    container_name: str
    container_id_commitment_sha256: str
    outer_cgroup_identity_sha256: str
    mount_namespace_identity: ArtifactIdentityV1
    rootfs_mount_identity: ArtifactIdentityV1
    tmpfs_mount_identity: ArtifactIdentityV1
    tmpfs_backing_identity: ArtifactIdentityV1
    mount_inventory: ArtifactIdentityV1
    path_inventory: ArtifactIdentityV1
    storage_root_inventory: ArtifactIdentityV1
    field_inventory: ArtifactIdentityV1
    raw_schema_inventory: ArtifactIdentityV1
    outer_cgroup_memory_swap_max_pre_go: ArtifactIdentityV1
    outer_cgroup_swap_counters_initial: ArtifactIdentityV1
    outer_cgroup_memory_zswap_writeback_pre_go: ArtifactIdentityV1
    docker_implicit_mount_inventory: ArtifactIdentityV1
    docker_create_inspect: ArtifactIdentityV1
    final_oci_spec: ArtifactIdentityV1
    console_stdio_inventory: ArtifactIdentityV1
    rootfs_upperdir_pre_go_baseline: ArtifactIdentityV1
    docker_volume_inventory_pre_go_baseline: ArtifactIdentityV1
    max_temporary_peak_bytes: int
    components: tuple[PinnedStorageComponentIdentityV1, ...]
    schema_version: str = STORAGE_BOUNDARY_RUNTIME_INTENT_SCHEMA_VERSION
    status: str = STORAGE_RUNTIME_INTENT_STATUS
    resource_field_order_sha256: str = RESOURCE_FIELD_ORDER_SHA256
    candidate_order_sha256: str = CANDIDATE_ORDER_SHA256
    aggregate_root_path: str = TMPFS_AGGREGATE_ROOT
    aggregate_root_count: int = 1
    aggregate_root_case_exclusive: bool = True
    aggregate_root_application_writable: bool = True
    aggregate_root_allocatable: bool = True
    tmpfs_hard_size_limit_bytes: int = 0
    tmpfs_exact_size_readback_matches_ceiling_pre_go: bool = True
    tmpfs_noswap_active_pre_go: bool = True
    mount_mutation_disabled_before_go: bool = True
    disk_published_value_bytes: int = 0
    disk_scope: str = "structurally_absent"
    persistent_storage_scope: str = PERSISTENT_STORAGE_SCOPE
    persistent_storage_measurement_interval: str = PERSISTENT_STORAGE_MEASUREMENT_INTERVAL
    trusted_runtime_bookkeeping_exclusions: tuple[str, ...] = TRUSTED_RUNTIME_BOOKKEEPING_EXCLUSIONS
    rootfs_read_only: bool = True
    rootfs_copy_up_enabled: bool = False
    container_log_driver: str = "none"
    application_bind_mount_count: int = 0
    volume_count: int = 0
    added_device_count: int = 0
    network_enabled: bool = False
    inherited_allocatable_storage_fd_count: int = 0
    alternate_writable_path_count: int = 0
    application_writable_tmpfs_count: int = 1
    docker_implicit_config_bind_paths: tuple[str, ...] = DOCKER_IMPLICIT_CONFIG_BIND_PATHS
    implicit_readonly_etc_bind_count: int = 3
    docker_implicit_mount_inventory_required: bool = True
    writable_persistent_mount_count: int = 0
    writable_persistent_fd_count: int = 0
    candidate_stdio_transport_count: int = 0
    docker_implicit_mounts_all_read_only: bool = True
    docker_implicit_mounts_all_nonallocatable: bool = True
    docker_implicit_mounts_have_application_writable_path: bool = False
    docker_ipc_mode: str = "none"
    docker_shm_mount_present: bool = False
    docker_tty_enabled: bool = False
    docker_stdin_open: bool = False
    image_declared_volume_count: int = 0
    docker_exec_permitted: bool = False
    docker_archive_api_permitted: bool = False
    docker_api_candidate_accessible: bool = False
    container_console_or_fifo_candidate_accessible: bool = False
    default_device_inventory_exact: bool = True
    default_devices_can_allocate_storage: bool = False
    device_open_ioctl_confinement_active: bool = True
    memfd_posix_or_sysv_shm_permitted: bool = False
    post_go_mount_mutation_permitted: bool = False
    candidate_cgroup_mutation_permitted: bool = False
    rootfs_upperdir_candidate_writable: bool = False
    daemon_runtime_storage_candidate_accessible: bool = False
    host_archival_candidate_accessible: bool = False
    outer_cgroup_memory_swap_max_bytes_pre_go: int = 0
    outer_cgroup_memory_zswap_writeback_enabled_pre_go: bool = False
    phase_number: int = 6
    committed_before_go: bool = True
    future_receipt_bound: bool = False
    future_seal_bound: bool = False
    future_terminal_bound: bool = False

    def __post_init__(self) -> None:
        if self.schema_version != STORAGE_BOUNDARY_RUNTIME_INTENT_SCHEMA_VERSION:
            _fail("runtime intent schema differs")
        if self.status != STORAGE_RUNTIME_INTENT_STATUS:
            _fail("runtime intent status differs")
        _require_identifier(self.campaign_id, "campaign ID")
        _candidate_projection(
            self.case_ordinal,
            self.candidate_id,
            self.candidate_family,
            self.qualification_case_id,
        )
        expected_artifacts = (
            (self.qualification_plan, QUALIFICATION_PLAN_V3_SCHEMA_VERSION, "qualification plan"),
            (self.policy, STORAGE_BACKEND_POLICY_SCHEMA_VERSION, "storage policy"),
            (
                self.runtime_qualification_receipt,
                RUNTIME_QUALIFICATION_RECEIPT_SCHEMA_VERSION,
                "runtime qualification receipt",
            ),
            (
                self.host_provisioning_v3_validated_pre_go_prefix,
                HOST_PROVISIONING_VALIDATED_PRE_GO_PREFIX_V3_SCHEMA_VERSION,
                "host provisioning pre-GO prefix",
            ),
        )
        for artifact, schema, label in expected_artifacts:
            _require_artifact_identity(artifact, label, expected_schema=schema)
        for artifact, schema, label in (
            (
                self.mount_namespace_identity,
                MOUNT_NAMESPACE_IDENTITY_SCHEMA_VERSION,
                "mount namespace",
            ),
            (self.rootfs_mount_identity, ROOTFS_MOUNT_IDENTITY_SCHEMA_VERSION, "rootfs mount"),
            (self.tmpfs_mount_identity, TMPFS_MOUNT_IDENTITY_SCHEMA_VERSION, "tmpfs mount"),
            (
                self.tmpfs_backing_identity,
                TMPFS_BACKING_IDENTITY_SCHEMA_VERSION,
                "tmpfs backing",
            ),
            (self.mount_inventory, MOUNT_INVENTORY_SCHEMA_VERSION, "mount inventory"),
            (self.path_inventory, PATH_INVENTORY_SCHEMA_VERSION, "path inventory"),
            (
                self.storage_root_inventory,
                STORAGE_ROOT_INVENTORY_SCHEMA_VERSION,
                "storage root inventory",
            ),
            (self.field_inventory, STORAGE_FIELD_INVENTORY_SCHEMA_VERSION, "field inventory"),
            (
                self.raw_schema_inventory,
                RAW_SCHEMA_INVENTORY_IDENTITY_SCHEMA_VERSION,
                "raw schema inventory",
            ),
        ):
            _require_artifact_identity(artifact, label, expected_schema=schema)
        boundary_artifacts = (
            self.mount_namespace_identity,
            self.rootfs_mount_identity,
            self.tmpfs_mount_identity,
            self.tmpfs_backing_identity,
            self.mount_inventory,
            self.path_inventory,
            self.storage_root_inventory,
            self.field_inventory,
            self.raw_schema_inventory,
        )
        _require_distinct_artifact_digests(boundary_artifacts, "runtime storage-boundary")
        for artifact, schema, label in (
            (
                self.outer_cgroup_memory_swap_max_pre_go,
                RAW_ARTIFACT_SCHEMA_INVENTORY[8],
                "outer-cgroup memory.swap.max pre-GO",
            ),
            (
                self.outer_cgroup_swap_counters_initial,
                RAW_ARTIFACT_SCHEMA_INVENTORY[9],
                "outer-cgroup initial swap counters",
            ),
            (
                self.outer_cgroup_memory_zswap_writeback_pre_go,
                RAW_ARTIFACT_SCHEMA_INVENTORY[11],
                "outer-cgroup memory.zswap.writeback pre-GO",
            ),
            (
                self.docker_implicit_mount_inventory,
                RAW_ARTIFACT_SCHEMA_INVENTORY[12],
                "Docker implicit mount inventory",
            ),
            (
                self.docker_create_inspect,
                RAW_ARTIFACT_SCHEMA_INVENTORY[13],
                "Docker create/inspect",
            ),
            (self.final_oci_spec, RAW_ARTIFACT_SCHEMA_INVENTORY[14], "final OCI spec"),
            (
                self.console_stdio_inventory,
                RAW_ARTIFACT_SCHEMA_INVENTORY[15],
                "console/stdin/stdout inventory",
            ),
            (
                self.rootfs_upperdir_pre_go_baseline,
                RAW_ARTIFACT_SCHEMA_INVENTORY[17],
                "rootfs upperdir pre-GO baseline",
            ),
            (
                self.docker_volume_inventory_pre_go_baseline,
                DOCKER_VOLUME_INVENTORY_PRE_GO_BASELINE_SCHEMA_VERSION,
                "Docker volume inventory pre-GO baseline",
            ),
        ):
            _require_artifact_identity(artifact, label, expected_schema=schema)
        all_runtime_artifacts = (
            tuple(artifact for artifact, _, _ in expected_artifacts)
            + boundary_artifacts
            + (
                self.outer_cgroup_memory_swap_max_pre_go,
                self.outer_cgroup_swap_counters_initial,
                self.outer_cgroup_memory_zswap_writeback_pre_go,
                self.docker_implicit_mount_inventory,
                self.docker_create_inspect,
                self.final_oci_spec,
                self.console_stdio_inventory,
                self.rootfs_upperdir_pre_go_baseline,
                self.docker_volume_inventory_pre_go_baseline,
            )
        )
        _require_distinct_artifact_digests(
            all_runtime_artifacts,
            "runtime first-class, boundary, or pre-GO raw artifact",
        )
        _require_image_id(self.image_id, "image ID")
        _require_identifier(self.container_name, "container name")
        _require_sha256(self.container_id_commitment_sha256, "container ID commitment")
        _require_sha256(self.outer_cgroup_identity_sha256, "outer-cgroup identity")
        if hmac.compare_digest(
            self.container_id_commitment_sha256,
            self.outer_cgroup_identity_sha256,
        ):
            _fail("container commitment and outer-cgroup identity alias")
        if self.resource_field_order_sha256 != RESOURCE_FIELD_ORDER_SHA256:
            _fail("resource field order hash differs")
        if self.candidate_order_sha256 != CANDIDATE_ORDER_SHA256:
            _fail("candidate order hash differs")
        ceiling = _require_int(self.max_temporary_peak_bytes, "temporary hard ceiling", minimum=1)
        if _require_int(self.tmpfs_hard_size_limit_bytes, "tmpfs hard limit", minimum=1) != ceiling:
            _fail("runtime tmpfs hard limit differs from field-24 ceiling")
        if not all(
            (
                self.tmpfs_exact_size_readback_matches_ceiling_pre_go is True,
                self.tmpfs_noswap_active_pre_go is True,
                self.mount_mutation_disabled_before_go is True,
            )
        ):
            _fail("runtime tmpfs size, noswap, or mount-mutation closure differs")
        if (
            self.aggregate_root_path != TMPFS_AGGREGATE_ROOT
            or _require_int(self.aggregate_root_count, "aggregate root count", minimum=1, maximum=1)
            != 1
            or self.aggregate_root_case_exclusive is not True
            or self.aggregate_root_application_writable is not True
            or self.aggregate_root_allocatable is not True
        ):
            _fail("runtime storage boundary is not one /run/alberta aggregate root")
        _require_int(self.disk_published_value_bytes, "runtime disk published value", maximum=0)
        if (
            self.disk_scope != "structurally_absent"
            or self.persistent_storage_scope != PERSISTENT_STORAGE_SCOPE
            or self.persistent_storage_measurement_interval
            != PERSISTENT_STORAGE_MEASUREMENT_INTERVAL
            or self.trusted_runtime_bookkeeping_exclusions != TRUSTED_RUNTIME_BOOKKEEPING_EXCLUSIONS
        ):
            _fail("runtime field 25 does not commit exact structural absence")
        if not all(
            (
                self.rootfs_read_only is True,
                self.rootfs_copy_up_enabled is False,
                self.container_log_driver == "none",
                self.network_enabled is False,
                self.docker_implicit_mount_inventory_required is True,
            )
        ):
            _fail("runtime rootfs, log, network, or implicit-mount topology differs")
        for value, label in (
            (self.application_bind_mount_count, "runtime application bind mount count"),
            (self.volume_count, "runtime volume count"),
            (self.added_device_count, "runtime added device count"),
            (
                self.inherited_allocatable_storage_fd_count,
                "runtime inherited allocatable-storage FD count",
            ),
            (self.alternate_writable_path_count, "runtime alternate writable path count"),
        ):
            _require_int(value, label, maximum=0)
        _require_int(
            self.application_writable_tmpfs_count,
            "runtime application-writable tmpfs count",
            minimum=1,
            maximum=1,
        )
        if not all(
            (
                self.docker_implicit_mounts_all_read_only is True,
                self.docker_implicit_mounts_all_nonallocatable is True,
                self.docker_implicit_mounts_have_application_writable_path is False,
            )
        ):
            _fail("runtime Docker implicit mounts admit application allocation")
        if self.docker_implicit_config_bind_paths != DOCKER_IMPLICIT_CONFIG_BIND_PATHS:
            _fail("runtime Docker implicit /etc bind paths differ")
        _require_int(
            self.implicit_readonly_etc_bind_count,
            "runtime implicit read-only /etc bind count",
            minimum=3,
            maximum=3,
        )
        for value, label in (
            (self.writable_persistent_mount_count, "runtime writable persistent mount count"),
            (self.writable_persistent_fd_count, "runtime writable persistent FD count"),
            (self.candidate_stdio_transport_count, "runtime candidate stdio transport count"),
            (self.image_declared_volume_count, "runtime image-declared volume count"),
        ):
            _require_int(value, label, maximum=0)
        if self.docker_ipc_mode != "none":
            _fail("runtime Docker IPC mode differs")
        if not all(
            (
                self.docker_shm_mount_present is False,
                self.docker_tty_enabled is False,
                self.docker_stdin_open is False,
                self.docker_exec_permitted is False,
                self.docker_archive_api_permitted is False,
                self.docker_api_candidate_accessible is False,
                self.container_console_or_fifo_candidate_accessible is False,
                self.default_device_inventory_exact is True,
                self.default_devices_can_allocate_storage is False,
                self.device_open_ioctl_confinement_active is True,
                self.memfd_posix_or_sysv_shm_permitted is False,
                self.post_go_mount_mutation_permitted is False,
                self.candidate_cgroup_mutation_permitted is False,
                self.rootfs_upperdir_candidate_writable is False,
                self.daemon_runtime_storage_candidate_accessible is False,
                self.host_archival_candidate_accessible is False,
            )
        ):
            _fail("runtime Docker storage/API/device/namespace escape closure differs")
        _require_int(
            self.outer_cgroup_memory_swap_max_bytes_pre_go,
            "runtime outer-cgroup memory.swap.max",
            maximum=0,
        )
        if self.outer_cgroup_memory_zswap_writeback_enabled_pre_go is not False:
            _fail("runtime outer-cgroup memory.zswap.writeback must be disabled")
        _validate_components(self.components)
        _require_int(self.phase_number, "runtime intent phase", minimum=6, maximum=6)
        if (
            self.committed_before_go is not True
            or self.future_receipt_bound is not False
            or self.future_seal_bound is not False
            or self.future_terminal_bound is not False
        ):
            _fail("runtime intent violates the pre-GO one-way phase boundary")
        _assert_plain_unaliased_json(self.to_body_dict())

    def to_body_dict(self) -> dict[str, Any]:
        return {
            "aggregate_root_count": self.aggregate_root_count,
            "aggregate_root_case_exclusive": self.aggregate_root_case_exclusive,
            "aggregate_root_allocatable": self.aggregate_root_allocatable,
            "aggregate_root_application_writable": self.aggregate_root_application_writable,
            "aggregate_root_path": self.aggregate_root_path,
            "application_writable_tmpfs_count": self.application_writable_tmpfs_count,
            "campaign_id": self.campaign_id,
            "candidate_family": self.candidate_family,
            "candidate_id": self.candidate_id,
            "candidate_order_sha256": self.candidate_order_sha256,
            "case_ordinal": self.case_ordinal,
            "committed_before_go": self.committed_before_go,
            "components": [component.to_dict() for component in self.components],
            "container_id_commitment_sha256": self.container_id_commitment_sha256,
            "container_name": self.container_name,
            "disk_published_value_bytes": self.disk_published_value_bytes,
            "disk_scope": self.disk_scope,
            "persistent_storage_scope": self.persistent_storage_scope,
            "persistent_storage_measurement_interval": (
                self.persistent_storage_measurement_interval
            ),
            "trusted_runtime_bookkeeping_exclusions": list(
                self.trusted_runtime_bookkeeping_exclusions
            ),
            "rootfs_read_only": self.rootfs_read_only,
            "rootfs_copy_up_enabled": self.rootfs_copy_up_enabled,
            "container_log_driver": self.container_log_driver,
            "application_bind_mount_count": self.application_bind_mount_count,
            "volume_count": self.volume_count,
            "added_device_count": self.added_device_count,
            "network_enabled": self.network_enabled,
            "inherited_allocatable_storage_fd_count": (self.inherited_allocatable_storage_fd_count),
            "alternate_writable_path_count": self.alternate_writable_path_count,
            "docker_implicit_mounts_have_application_writable_path": (
                self.docker_implicit_mounts_have_application_writable_path
            ),
            "docker_implicit_mount_inventory": self.docker_implicit_mount_inventory.to_dict(),
            "docker_implicit_mounts_all_nonallocatable": (
                self.docker_implicit_mounts_all_nonallocatable
            ),
            "docker_implicit_mounts_all_read_only": self.docker_implicit_mounts_all_read_only,
            "docker_create_inspect": self.docker_create_inspect.to_dict(),
            "final_oci_spec": self.final_oci_spec.to_dict(),
            "console_stdio_inventory": self.console_stdio_inventory.to_dict(),
            "rootfs_upperdir_pre_go_baseline": (self.rootfs_upperdir_pre_go_baseline.to_dict()),
            "docker_volume_inventory_pre_go_baseline": (
                self.docker_volume_inventory_pre_go_baseline.to_dict()
            ),
            "docker_implicit_config_bind_paths": list(self.docker_implicit_config_bind_paths),
            "implicit_readonly_etc_bind_count": self.implicit_readonly_etc_bind_count,
            "docker_implicit_mount_inventory_required": (
                self.docker_implicit_mount_inventory_required
            ),
            "writable_persistent_mount_count": self.writable_persistent_mount_count,
            "writable_persistent_fd_count": self.writable_persistent_fd_count,
            "candidate_stdio_transport_count": self.candidate_stdio_transport_count,
            "docker_ipc_mode": self.docker_ipc_mode,
            "docker_shm_mount_present": self.docker_shm_mount_present,
            "docker_tty_enabled": self.docker_tty_enabled,
            "docker_stdin_open": self.docker_stdin_open,
            "image_declared_volume_count": self.image_declared_volume_count,
            "docker_exec_permitted": self.docker_exec_permitted,
            "docker_archive_api_permitted": self.docker_archive_api_permitted,
            "docker_api_candidate_accessible": self.docker_api_candidate_accessible,
            "container_console_or_fifo_candidate_accessible": (
                self.container_console_or_fifo_candidate_accessible
            ),
            "default_device_inventory_exact": self.default_device_inventory_exact,
            "default_devices_can_allocate_storage": self.default_devices_can_allocate_storage,
            "device_open_ioctl_confinement_active": self.device_open_ioctl_confinement_active,
            "memfd_posix_or_sysv_shm_permitted": self.memfd_posix_or_sysv_shm_permitted,
            "post_go_mount_mutation_permitted": self.post_go_mount_mutation_permitted,
            "candidate_cgroup_mutation_permitted": self.candidate_cgroup_mutation_permitted,
            "rootfs_upperdir_candidate_writable": self.rootfs_upperdir_candidate_writable,
            "daemon_runtime_storage_candidate_accessible": (
                self.daemon_runtime_storage_candidate_accessible
            ),
            "host_archival_candidate_accessible": self.host_archival_candidate_accessible,
            "field_inventory": self.field_inventory.to_dict(),
            "future_receipt_bound": self.future_receipt_bound,
            "future_seal_bound": self.future_seal_bound,
            "future_terminal_bound": self.future_terminal_bound,
            "host_provisioning_v3_validated_pre_go_prefix": (
                self.host_provisioning_v3_validated_pre_go_prefix.to_dict()
            ),
            "image_id": self.image_id,
            "max_temporary_peak_bytes": self.max_temporary_peak_bytes,
            "mount_inventory": self.mount_inventory.to_dict(),
            "mount_namespace_identity": self.mount_namespace_identity.to_dict(),
            "outer_cgroup_identity_sha256": self.outer_cgroup_identity_sha256,
            "outer_cgroup_memory_swap_max_bytes_pre_go": (
                self.outer_cgroup_memory_swap_max_bytes_pre_go
            ),
            "outer_cgroup_memory_swap_max_pre_go": (
                self.outer_cgroup_memory_swap_max_pre_go.to_dict()
            ),
            "outer_cgroup_swap_counters_initial": (
                self.outer_cgroup_swap_counters_initial.to_dict()
            ),
            "outer_cgroup_memory_zswap_writeback_enabled_pre_go": (
                self.outer_cgroup_memory_zswap_writeback_enabled_pre_go
            ),
            "outer_cgroup_memory_zswap_writeback_pre_go": (
                self.outer_cgroup_memory_zswap_writeback_pre_go.to_dict()
            ),
            "path_inventory": self.path_inventory.to_dict(),
            "phase_number": self.phase_number,
            "policy": self.policy.to_dict(),
            "qualification_case_id": self.qualification_case_id,
            "qualification_plan": self.qualification_plan.to_dict(),
            "raw_schema_inventory": self.raw_schema_inventory.to_dict(),
            "resource_field_order_sha256": self.resource_field_order_sha256,
            "rootfs_mount_identity": self.rootfs_mount_identity.to_dict(),
            "runtime_qualification_receipt": self.runtime_qualification_receipt.to_dict(),
            "schema_version": self.schema_version,
            "status": self.status,
            "storage_root_inventory": self.storage_root_inventory.to_dict(),
            "tmpfs_backing_identity": self.tmpfs_backing_identity.to_dict(),
            "tmpfs_exact_size_readback_matches_ceiling_pre_go": (
                self.tmpfs_exact_size_readback_matches_ceiling_pre_go
            ),
            "tmpfs_hard_size_limit_bytes": self.tmpfs_hard_size_limit_bytes,
            "tmpfs_mount_identity": self.tmpfs_mount_identity.to_dict(),
            "tmpfs_noswap_active_pre_go": self.tmpfs_noswap_active_pre_go,
            "mount_mutation_disabled_before_go": self.mount_mutation_disabled_before_go,
        }


def _validate_current_storage_boundary_runtime_intent_v1(
    intent: object,
) -> StorageBoundaryRuntimeIntentV1:
    if type(intent) is not StorageBoundaryRuntimeIntentV1:
        _fail("runtime intent type differs")
    exact = intent
    _translate_current_state_error(
        "runtime intent",
        lambda: StorageBoundaryRuntimeIntentV1.__post_init__(exact),
    )
    return exact


@dataclass(frozen=True, slots=True)
class HostV3HandshakeProjectionV1:
    """Exact host-v3 request/intent/READY/anchor/GO projection."""

    request: ArtifactIdentityV1
    intent: ArtifactIdentityV1
    ready: ArtifactIdentityV1
    observer_anchor: ArtifactIdentityV1
    go: ArtifactIdentityV1
    campaign_id: str
    case_ordinal: int
    candidate_id: str
    qualification_case_id: str
    image_id: str
    container_name: str
    container_id_commitment_sha256: str
    exact_projection_validated: bool = True
    order: tuple[str, ...] = ("request", "intent", "ready", "observer_anchor", "go")

    def __post_init__(self) -> None:
        handshake_artifacts = (
            (self.request, HOST_CASE_REQUEST_V3_SCHEMA_VERSION, "host request"),
            (self.intent, HOST_CASE_INTENT_V3_SCHEMA_VERSION, "host intent"),
            (self.ready, HOST_READY_V3_SCHEMA_VERSION, "host READY"),
            (self.observer_anchor, HOST_OBSERVER_ANCHOR_V3_SCHEMA_VERSION, "host anchor"),
            (self.go, HOST_GO_V3_SCHEMA_VERSION, "host GO"),
        )
        for artifact, schema, label in handshake_artifacts:
            _require_artifact_identity(artifact, label, expected_schema=schema)
        _require_distinct_artifact_digests(
            tuple(artifact for artifact, _, _ in handshake_artifacts),
            "host handshake artifact",
        )
        _require_identifier(self.campaign_id, "handshake campaign ID")
        ordinal = _require_int(self.case_ordinal, "handshake case ordinal", maximum=27)
        candidate = _require_identifier(self.candidate_id, "handshake candidate ID")
        case_id = _require_identifier(self.qualification_case_id, "handshake case ID")
        expected_candidate = MATCHED_V3_CANDIDATE_IDS[ordinal]
        if candidate != expected_candidate:
            _fail("host handshake candidate differs from the frozen case ordinal")
        if case_id != f"qualification_{ordinal:02d}_{expected_candidate}":
            _fail("host handshake case ID differs from the frozen projection")
        _require_image_id(self.image_id, "handshake image ID")
        _require_identifier(self.container_name, "handshake container name")
        _require_sha256(self.container_id_commitment_sha256, "handshake container commitment")
        if self.exact_projection_validated is not True:
            _fail("host handshake projection is not exact")
        if self.order != ("request", "intent", "ready", "observer_anchor", "go"):
            _fail("host handshake chronology differs")
        _assert_plain_unaliased_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "campaign_id": self.campaign_id,
            "candidate_id": self.candidate_id,
            "case_ordinal": self.case_ordinal,
            "container_id_commitment_sha256": self.container_id_commitment_sha256,
            "container_name": self.container_name,
            "exact_projection_validated": self.exact_projection_validated,
            "go": self.go.to_dict(),
            "image_id": self.image_id,
            "intent": self.intent.to_dict(),
            "observer_anchor": self.observer_anchor.to_dict(),
            "order": list(self.order),
            "qualification_case_id": self.qualification_case_id,
            "ready": self.ready.to_dict(),
            "request": self.request.to_dict(),
        }


def _validate_current_host_v3_handshake_projection_v1(
    handshake: object,
) -> HostV3HandshakeProjectionV1:
    if type(handshake) is not HostV3HandshakeProjectionV1:
        _fail("host handshake projection type differs")
    exact = handshake
    _translate_current_state_error(
        "host handshake projection",
        lambda: HostV3HandshakeProjectionV1.__post_init__(exact),
    )
    return exact


@dataclass(frozen=True, slots=True)
class RawArtifactProjectionV1:
    """Bind one first-class reload/seal artifact to raw evidence and its producer."""

    kind: Literal["publication_reload", "write_seal"]
    first_class_artifact: ArtifactIdentityV1
    raw_artifact: ArtifactIdentityV1
    predecessors: tuple[ArtifactIdentityV1, ...]
    producer: PinnedStorageComponentIdentityV1
    exact_projection: bool = True
    predecessor_identity_bound: bool = True

    def __post_init__(self) -> None:
        predecessor_schemas: tuple[str, ...]
        if self.kind == "publication_reload":
            first_schema = PUBLICATION_RELOAD_VALIDATION_SCHEMA_VERSION
            raw_schema = RAW_ARTIFACT_SCHEMA_INVENTORY[6]
            producer_role = "terminal_relay"
            predecessor_schemas = (NORMALIZED_PUBLICATION_WRAPPER_SCHEMA_VERSION,)
        elif self.kind == "write_seal":
            first_schema = IRREVERSIBLE_WRITE_SEAL_SCHEMA_VERSION
            raw_schema = RAW_ARTIFACT_SCHEMA_INVENTORY[7]
            producer_role = "write_seal_producer"
            predecessor_schemas = (
                PUBLICATION_RELOAD_VALIDATION_SCHEMA_VERSION,
                TERMINAL_RELAY_PRESEAL_ATTESTATION_SCHEMA_VERSION,
                NONSTORAGE_CHANNEL_PRESEAL_ATTESTATION_SCHEMA_VERSION,
            )
        else:
            _fail("raw artifact projection kind differs")
        for artifact, schema, label in (
            (self.first_class_artifact, first_schema, "first-class projection artifact"),
            (self.raw_artifact, raw_schema, "raw projection artifact"),
        ):
            _require_artifact_identity(artifact, label, expected_schema=schema)
        if (
            type(self.predecessors) is not tuple
            or len(self.predecessors) != len(predecessor_schemas)
        ):
            _fail("raw projection predecessor inventory differs")
        for index, (artifact, schema) in enumerate(
            zip(self.predecessors, predecessor_schemas, strict=True)
        ):
            _require_artifact_identity(
                artifact,
                f"raw projection predecessor {index}",
                expected_schema=schema,
            )
        bound_artifacts = (
            self.first_class_artifact,
            self.raw_artifact,
            *self.predecessors,
        )
        _require_distinct_artifact_digests(bound_artifacts, "raw projection artifact")
        _require_component_role(self.producer, producer_role, "raw artifact projection")
        if self.exact_projection is not True or self.predecessor_identity_bound is not True:
            _fail("raw artifact projection is not exact and predecessor-bound")
        _assert_plain_unaliased_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "exact_projection": self.exact_projection,
            "first_class_artifact": self.first_class_artifact.to_dict(),
            "kind": self.kind,
            "predecessor_identity_bound": self.predecessor_identity_bound,
            "predecessors": [artifact.to_dict() for artifact in self.predecessors],
            "producer": self.producer.to_dict(),
            "raw_artifact": self.raw_artifact.to_dict(),
        }


def _validate_current_raw_artifact_projection_v1(
    projection: object,
) -> RawArtifactProjectionV1:
    if type(projection) is not RawArtifactProjectionV1:
        _fail("raw artifact projection type differs")
    exact = projection
    _translate_current_state_error(
        "raw artifact projection",
        lambda: RawArtifactProjectionV1.__post_init__(exact),
    )
    return exact


@dataclass(frozen=True, slots=True)
class HostGoStorageIntentBindingV1:
    """Projection derived from parsed host GO naming the exact phase-6 intent."""

    host_go: ArtifactIdentityV1
    runtime_intent: ArtifactIdentityV1
    verifier: PinnedStorageComponentIdentityV1
    exact_bidirectional_projection: bool = True
    intent_committed_before_go: bool = True

    def __post_init__(self) -> None:
        for artifact, schema, label in (
            (self.host_go, HOST_GO_V3_SCHEMA_VERSION, "host GO"),
            (
                self.runtime_intent,
                STORAGE_BOUNDARY_RUNTIME_INTENT_SCHEMA_VERSION,
                "storage runtime intent",
            ),
        ):
            _require_artifact_identity(artifact, label, expected_schema=schema)
        bound_artifacts = (self.host_go, self.runtime_intent)
        _require_distinct_artifact_digests(bound_artifacts, "host-GO binding artifact")
        _require_component_role(
            self.verifier,
            "measurement_producer",
            "host-GO storage-intent binding",
        )
        if (
            self.exact_bidirectional_projection is not True
            or self.intent_committed_before_go is not True
        ):
            _fail("host GO does not exactly bind a prior storage runtime intent")
        _assert_plain_unaliased_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "exact_bidirectional_projection": self.exact_bidirectional_projection,
            "host_go": self.host_go.to_dict(),
            "intent_committed_before_go": self.intent_committed_before_go,
            "runtime_intent": self.runtime_intent.to_dict(),
            "verifier": self.verifier.to_dict(),
        }


def _validate_current_host_go_storage_intent_binding_v1(
    binding: object,
) -> HostGoStorageIntentBindingV1:
    if type(binding) is not HostGoStorageIntentBindingV1:
        _fail("host-GO storage-intent binding type differs")
    exact = binding
    _translate_current_state_error(
        "host-GO storage-intent binding",
        lambda: HostGoStorageIntentBindingV1.__post_init__(exact),
    )
    return exact


@dataclass(frozen=True, slots=True)
class TmpfsConservativeBoundEvidenceV1:
    """Immutable tmpfs hard-bound proof with diagnostic occupancy samples only."""

    aggregate_root_path: str
    hard_limit_bytes: int
    initial_observed_used_bytes: int
    terminal_observed_used_bytes: int
    measurement_interval_start: ArtifactIdentityV1
    measurement_interval_end: ArtifactIdentityV1
    raw_mountinfo: ArtifactIdentityV1
    raw_statfs_diagnostic_samples: ArtifactIdentityV1
    raw_hard_limit_mount_mutation_closure: ArtifactIdentityV1
    measurement_producer: PinnedStorageComponentIdentityV1
    field_position: int = FIELD24_POSITION
    accounting_mode: str = "immutable_non_bypass_pre_go_tmpfs_hard_limit"
    publication_semantics: str = "conservative_enforced_upper_bound"
    aggregate_root_count: int = 1
    aggregate_root_case_exclusive: bool = True
    exact_size_readback_equals_ceiling_pre_go: bool = True
    hard_limit_unchanged_through_write_seal: bool = True
    noswap_active_pre_go: bool = True
    noswap_unchanged_through_write_seal: bool = True
    mount_mutation_disabled_before_go: bool = True
    mount_mutation_disabled_through_write_seal: bool = True
    aggregate_root_non_bypassable: bool = True
    statfs_samples_are_diagnostic_only: bool = True

    def __post_init__(self) -> None:
        if (
            self.aggregate_root_path != TMPFS_AGGREGATE_ROOT
            or _require_int(self.aggregate_root_count, "aggregate root count", minimum=1, maximum=1)
            != 1
        ):
            _fail("tmpfs bound evidence does not cover one aggregate root")
        hard = _require_int(self.hard_limit_bytes, "tmpfs hard limit", minimum=1)
        for value, label in (
            (self.initial_observed_used_bytes, "initial diagnostic tmpfs occupancy"),
            (self.terminal_observed_used_bytes, "terminal diagnostic tmpfs occupancy"),
        ):
            if _require_int(value, label) > hard:
                _fail("diagnostic tmpfs occupancy exceeds the retained hard limit")
        expected = (
            (self.measurement_interval_start, HOST_GO_V3_SCHEMA_VERSION),
            (self.measurement_interval_end, IRREVERSIBLE_WRITE_SEAL_SCHEMA_VERSION),
            (self.raw_mountinfo, RAW_ARTIFACT_SCHEMA_INVENTORY[0]),
            (self.raw_statfs_diagnostic_samples, RAW_ARTIFACT_SCHEMA_INVENTORY[1]),
            (
                self.raw_hard_limit_mount_mutation_closure,
                RAW_ARTIFACT_SCHEMA_INVENTORY[2],
            ),
        )
        for artifact, schema in expected:
            _require_artifact_identity(
                artifact,
                "tmpfs raw evidence",
                expected_schema=schema,
            )
        tmpfs_artifacts = tuple(artifact for artifact, _ in expected)
        _require_distinct_artifact_digests(tmpfs_artifacts, "tmpfs raw evidence")
        _require_component_role(
            self.measurement_producer,
            "measurement_producer",
            "tmpfs conservative-bound evidence",
        )
        _require_int(
            self.field_position,
            "tmpfs evidence field position",
            minimum=FIELD24_POSITION,
            maximum=FIELD24_POSITION,
        )
        if self.accounting_mode != "immutable_non_bypass_pre_go_tmpfs_hard_limit":
            _fail("tmpfs accounting mode differs")
        if self.publication_semantics != "conservative_enforced_upper_bound":
            _fail("tmpfs publication semantics differ")
        for flag, label in (
            (self.aggregate_root_case_exclusive, "tmpfs aggregate-root case exclusivity"),
            (
                self.exact_size_readback_equals_ceiling_pre_go,
                "exact pre-GO tmpfs size readback",
            ),
            (
                self.hard_limit_unchanged_through_write_seal,
                "tmpfs hard limit unchanged through seal",
            ),
            (self.noswap_active_pre_go, "tmpfs noswap active before GO"),
            (
                self.noswap_unchanged_through_write_seal,
                "tmpfs noswap unchanged through seal",
            ),
            (self.mount_mutation_disabled_before_go, "mount mutation disabled before GO"),
            (
                self.mount_mutation_disabled_through_write_seal,
                "mount mutation disabled through seal",
            ),
            (self.aggregate_root_non_bypassable, "tmpfs aggregate root non-bypassability"),
            (self.statfs_samples_are_diagnostic_only, "statfs diagnostic-only semantics"),
        ):
            if flag is not True:
                _fail(f"{label} differs")
        _assert_plain_unaliased_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "accounting_mode": self.accounting_mode,
            "aggregate_root_count": self.aggregate_root_count,
            "aggregate_root_case_exclusive": self.aggregate_root_case_exclusive,
            "aggregate_root_path": self.aggregate_root_path,
            "field_position": self.field_position,
            "hard_limit_bytes": self.hard_limit_bytes,
            "hard_limit_unchanged_through_write_seal": (
                self.hard_limit_unchanged_through_write_seal
            ),
            "initial_observed_used_bytes": self.initial_observed_used_bytes,
            "measurement_interval_start": self.measurement_interval_start.to_dict(),
            "measurement_interval_end": self.measurement_interval_end.to_dict(),
            "mount_mutation_disabled_before_go": self.mount_mutation_disabled_before_go,
            "mount_mutation_disabled_through_write_seal": (
                self.mount_mutation_disabled_through_write_seal
            ),
            "noswap_active_pre_go": self.noswap_active_pre_go,
            "noswap_unchanged_through_write_seal": (self.noswap_unchanged_through_write_seal),
            "publication_semantics": self.publication_semantics,
            "aggregate_root_non_bypassable": self.aggregate_root_non_bypassable,
            "exact_size_readback_equals_ceiling_pre_go": (
                self.exact_size_readback_equals_ceiling_pre_go
            ),
            "statfs_samples_are_diagnostic_only": self.statfs_samples_are_diagnostic_only,
            "raw_mountinfo": self.raw_mountinfo.to_dict(),
            "raw_hard_limit_mount_mutation_closure": (
                self.raw_hard_limit_mount_mutation_closure.to_dict()
            ),
            "raw_statfs_diagnostic_samples": self.raw_statfs_diagnostic_samples.to_dict(),
            "measurement_producer": self.measurement_producer.to_dict(),
            "terminal_observed_used_bytes": self.terminal_observed_used_bytes,
        }


def _validate_current_tmpfs_conservative_bound_evidence_v1(
    evidence: object,
) -> TmpfsConservativeBoundEvidenceV1:
    if type(evidence) is not TmpfsConservativeBoundEvidenceV1:
        _fail("tmpfs conservative-bound evidence type differs")
    exact = evidence
    _translate_current_state_error(
        "tmpfs conservative-bound evidence",
        lambda: TmpfsConservativeBoundEvidenceV1.__post_init__(exact),
    )
    return exact


@dataclass(frozen=True, slots=True)
class DiskStructuralAbsenceEvidenceV1:
    """Structural non-addressability proof for scoped field-25 exact zero."""

    measurement_interval_start: ArtifactIdentityV1
    measurement_interval_end: ArtifactIdentityV1
    raw_mountinfo: ArtifactIdentityV1
    raw_absence_scan: ArtifactIdentityV1
    raw_writable_fd_inventory: ArtifactIdentityV1
    measurement_producer: PinnedStorageComponentIdentityV1
    field_position: int = FIELD25_POSITION
    published_value_bytes: int = 0
    absence_kind: str = "disk_storage_scope_structurally_absent"
    persistent_storage_scope: str = PERSISTENT_STORAGE_SCOPE
    measurement_interval: str = PERSISTENT_STORAGE_MEASUREMENT_INTERVAL
    trusted_runtime_bookkeeping_exclusions: tuple[str, ...] = TRUSTED_RUNTIME_BOOKKEEPING_EXCLUSIONS
    measurement_mode: str = "structural_nonaddressability_with_api_and_security_closure"
    rootfs_read_only: bool = True
    copy_up_disabled: bool = True
    no_bind_volume_device_network_log_paths: bool = True
    no_inherited_or_alternate_writable_paths: bool = True
    writable_persistent_mount_count: int = 0
    writable_persistent_fd_count: int = 0
    transient_persistent_open_structurally_impossible: bool = True
    structural_nonaddressability_complete: bool = True

    def __post_init__(self) -> None:
        disk_artifacts = (
            (self.measurement_interval_start, HOST_GO_V3_SCHEMA_VERSION),
            (self.measurement_interval_end, IRREVERSIBLE_WRITE_SEAL_SCHEMA_VERSION),
            (self.raw_mountinfo, RAW_ARTIFACT_SCHEMA_INVENTORY[3]),
            (self.raw_absence_scan, RAW_ARTIFACT_SCHEMA_INVENTORY[4]),
            (self.raw_writable_fd_inventory, RAW_ARTIFACT_SCHEMA_INVENTORY[5]),
        )
        for artifact, schema in disk_artifacts:
            _require_artifact_identity(
                artifact,
                "disk absence raw evidence",
                expected_schema=schema,
            )
        _require_distinct_artifact_digests(
            tuple(artifact for artifact, _ in disk_artifacts),
            "disk absence artifact",
        )
        _require_component_role(
            self.measurement_producer,
            "measurement_producer",
            "disk structural-absence evidence",
        )
        _require_int(
            self.field_position,
            "disk evidence field position",
            minimum=FIELD25_POSITION,
            maximum=FIELD25_POSITION,
        )
        _require_int(self.published_value_bytes, "disk published value", maximum=0)
        if self.absence_kind != "disk_storage_scope_structurally_absent":
            _fail("disk structural-absence kind differs")
        if (
            self.persistent_storage_scope != PERSISTENT_STORAGE_SCOPE
            or self.measurement_interval != PERSISTENT_STORAGE_MEASUREMENT_INTERVAL
            or self.trusted_runtime_bookkeeping_exclusions != TRUSTED_RUNTIME_BOOKKEEPING_EXCLUSIONS
            or self.measurement_mode != "structural_nonaddressability_with_api_and_security_closure"
        ):
            _fail("disk structural-absence scope or measurement semantics differ")
        _require_int(
            self.writable_persistent_mount_count,
            "disk writable persistent mount count",
            maximum=0,
        )
        _require_int(
            self.writable_persistent_fd_count,
            "disk writable persistent FD count",
            maximum=0,
        )
        if not all(
            (
                self.rootfs_read_only is True,
                self.copy_up_disabled is True,
                self.no_bind_volume_device_network_log_paths is True,
                self.no_inherited_or_alternate_writable_paths is True,
                self.transient_persistent_open_structurally_impossible is True,
                self.structural_nonaddressability_complete is True,
            )
        ):
            _fail("disk structural-absence evidence admits an escape")
        _assert_plain_unaliased_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "absence_kind": self.absence_kind,
            "copy_up_disabled": self.copy_up_disabled,
            "measurement_interval_start": self.measurement_interval_start.to_dict(),
            "measurement_interval_end": self.measurement_interval_end.to_dict(),
            "persistent_storage_scope": self.persistent_storage_scope,
            "measurement_interval": self.measurement_interval,
            "trusted_runtime_bookkeeping_exclusions": list(
                self.trusted_runtime_bookkeeping_exclusions
            ),
            "measurement_mode": self.measurement_mode,
            "measurement_producer": self.measurement_producer.to_dict(),
            "field_position": self.field_position,
            "no_bind_volume_device_network_log_paths": (
                self.no_bind_volume_device_network_log_paths
            ),
            "no_inherited_or_alternate_writable_paths": (
                self.no_inherited_or_alternate_writable_paths
            ),
            "published_value_bytes": self.published_value_bytes,
            "raw_absence_scan": self.raw_absence_scan.to_dict(),
            "raw_mountinfo": self.raw_mountinfo.to_dict(),
            "raw_writable_fd_inventory": self.raw_writable_fd_inventory.to_dict(),
            "rootfs_read_only": self.rootfs_read_only,
            "writable_persistent_mount_count": self.writable_persistent_mount_count,
            "writable_persistent_fd_count": self.writable_persistent_fd_count,
            "transient_persistent_open_structurally_impossible": (
                self.transient_persistent_open_structurally_impossible
            ),
            "structural_nonaddressability_complete": self.structural_nonaddressability_complete,
        }


def _validate_current_disk_structural_absence_evidence_v1(
    evidence: object,
) -> DiskStructuralAbsenceEvidenceV1:
    if type(evidence) is not DiskStructuralAbsenceEvidenceV1:
        _fail("disk structural-absence evidence type differs")
    exact = evidence
    _translate_current_state_error(
        "disk structural-absence evidence",
        lambda: DiskStructuralAbsenceEvidenceV1.__post_init__(exact),
    )
    return exact


@dataclass(frozen=True, slots=True)
class SwapAndImplicitMountClosureEvidenceV1:
    """Retained exact-zero swap and hardened Docker implicit-mount proof."""

    outer_cgroup_identity_sha256: str
    measurement_interval_start: ArtifactIdentityV1
    measurement_interval_end: ArtifactIdentityV1
    docker_volume_inventory_pre_go_baseline: ArtifactIdentityV1
    raw_memory_swap_max_pre_go: ArtifactIdentityV1
    raw_swap_counters_initial: ArtifactIdentityV1
    raw_swap_counters_terminal: ArtifactIdentityV1
    raw_memory_zswap_writeback_pre_go: ArtifactIdentityV1
    raw_docker_implicit_mount_inventory: ArtifactIdentityV1
    raw_docker_create_inspect: ArtifactIdentityV1
    raw_final_oci_spec: ArtifactIdentityV1
    raw_console_stdio_inventory: ArtifactIdentityV1
    raw_docker_api_operation_journal: ArtifactIdentityV1
    raw_rootfs_upperdir_pre_go_baseline: ArtifactIdentityV1
    raw_rootfs_upperdir_interval_delta: ArtifactIdentityV1
    raw_docker_volume_inventory_delta: ArtifactIdentityV1
    measurement_producer: PinnedStorageComponentIdentityV1
    runtime_storage_escape_gate: PinnedStorageComponentIdentityV1
    initial_observation_monotonic_ns: int
    go_commit_monotonic_ns: int
    worker_exit_monotonic_ns: int
    publication_wrapper_monotonic_ns: int
    reload_validation_monotonic_ns: int
    relay_preseal_monotonic_ns: int
    channel_preseal_monotonic_ns: int
    write_seal_monotonic_ns: int
    terminal_observation_monotonic_ns: int
    receipt_precommit_monotonic_ns: int
    memory_swap_max_bytes_pre_go: int = 0
    memory_zswap_writeback_enabled_pre_go: bool = False
    memory_swap_current_initial_bytes: int = 0
    memory_swap_current_terminal_bytes: int = 0
    memory_swap_peak_initial_bytes: int = 0
    memory_swap_peak_terminal_bytes: int = 0
    memory_zswap_current_initial_bytes: int = 0
    memory_zswap_current_terminal_bytes: int = 0
    retained_counter_endpoints_same_outer_cgroup: bool = True
    counters_retained_from_pre_go_through_terminal: bool = True
    application_writable_tmpfs_count: int = 1
    application_writable_tmpfs_is_aggregate_root: bool = True
    application_writable_tmpfs_allocatable: bool = True
    docker_implicit_mount_inventory_complete: bool = True
    docker_implicit_mounts_all_read_only: bool = True
    docker_implicit_mounts_all_nonallocatable: bool = True
    docker_implicit_mounts_have_application_writable_path: bool = False
    persistent_storage_scope: str = PERSISTENT_STORAGE_SCOPE
    persistent_storage_measurement_interval: str = PERSISTENT_STORAGE_MEASUREMENT_INTERVAL
    trusted_runtime_bookkeeping_exclusions: tuple[str, ...] = TRUSTED_RUNTIME_BOOKKEEPING_EXCLUSIONS
    docker_implicit_config_bind_paths: tuple[str, ...] = DOCKER_IMPLICIT_CONFIG_BIND_PATHS
    implicit_readonly_etc_bind_count: int = 3
    user_bind_mount_count: int = 0
    user_volume_mount_count: int = 0
    image_declared_volume_count: int = 0
    added_device_count: int = 0
    writable_persistent_mount_count: int = 0
    writable_persistent_fd_count: int = 0
    candidate_stdio_transport_count: int = 0
    forbidden_docker_api_operation_count: int = 0
    rootfs_upperdir_interval_delta_bytes: int = 0
    docker_volume_inventory_delta: int = 0
    docker_ipc_mode: str = "none"
    docker_shm_mount_present: bool = False
    docker_tty_enabled: bool = False
    docker_stdin_open: bool = False
    docker_exec_permitted: bool = False
    docker_archive_api_permitted: bool = False
    docker_api_candidate_accessible: bool = False
    container_console_or_fifo_candidate_accessible: bool = False
    default_device_inventory_exact: bool = True
    default_devices_can_allocate_storage: bool = False
    memfd_posix_or_sysv_shm_permitted: bool = False
    post_go_mount_mutation_permitted: bool = False
    candidate_cgroup_mutation_permitted: bool = False
    daemon_runtime_storage_candidate_accessible: bool = False
    host_archival_candidate_accessible: bool = False
    final_oci_spec_exact: bool = True
    custom_runtime_neutralized_stock_writable_implicit_mounts: bool = True
    structural_nonaddressability_complete: bool = True
    docker_api_allowlist_complete_and_lossless: bool = True
    device_and_ipc_confinement_active: bool = True
    rootfs_upperdir_candidate_inaccessible: bool = True

    def __post_init__(self) -> None:
        _require_sha256(self.outer_cgroup_identity_sha256, "outer-cgroup identity")
        closure_artifacts = (
            (
                self.measurement_interval_start,
                HOST_GO_V3_SCHEMA_VERSION,
                "swap/implicit-mount measurement interval start",
            ),
            (
                self.measurement_interval_end,
                IRREVERSIBLE_WRITE_SEAL_SCHEMA_VERSION,
                "swap/implicit-mount measurement interval end",
            ),
            (
                self.docker_volume_inventory_pre_go_baseline,
                DOCKER_VOLUME_INVENTORY_PRE_GO_BASELINE_SCHEMA_VERSION,
                "Docker volume inventory pre-GO baseline",
            ),
            (
                self.raw_memory_swap_max_pre_go,
                RAW_ARTIFACT_SCHEMA_INVENTORY[8],
                "raw memory.swap.max pre-GO",
            ),
            (
                self.raw_swap_counters_initial,
                RAW_ARTIFACT_SCHEMA_INVENTORY[9],
                "raw initial swap counters",
            ),
            (
                self.raw_swap_counters_terminal,
                RAW_ARTIFACT_SCHEMA_INVENTORY[10],
                "raw terminal swap counters",
            ),
            (
                self.raw_memory_zswap_writeback_pre_go,
                RAW_ARTIFACT_SCHEMA_INVENTORY[11],
                "raw memory.zswap.writeback pre-GO",
            ),
            (
                self.raw_docker_implicit_mount_inventory,
                RAW_ARTIFACT_SCHEMA_INVENTORY[12],
                "raw Docker implicit mount inventory",
            ),
            (
                self.raw_docker_create_inspect,
                RAW_ARTIFACT_SCHEMA_INVENTORY[13],
                "raw Docker create/inspect projection",
            ),
            (self.raw_final_oci_spec, RAW_ARTIFACT_SCHEMA_INVENTORY[14], "raw final OCI spec"),
            (
                self.raw_console_stdio_inventory,
                RAW_ARTIFACT_SCHEMA_INVENTORY[15],
                "raw console/stdin/stdout inventory",
            ),
            (
                self.raw_docker_api_operation_journal,
                RAW_ARTIFACT_SCHEMA_INVENTORY[16],
                "raw Docker API operation journal",
            ),
            (
                self.raw_rootfs_upperdir_pre_go_baseline,
                RAW_ARTIFACT_SCHEMA_INVENTORY[17],
                "raw rootfs upperdir pre-GO baseline",
            ),
            (
                self.raw_rootfs_upperdir_interval_delta,
                RAW_ARTIFACT_SCHEMA_INVENTORY[18],
                "raw rootfs upperdir interval delta",
            ),
            (
                self.raw_docker_volume_inventory_delta,
                RAW_ARTIFACT_SCHEMA_INVENTORY[19],
                "raw Docker volume inventory delta",
            ),
        )
        for artifact, schema, label in closure_artifacts:
            _require_artifact_identity(artifact, label, expected_schema=schema)
        _require_distinct_artifact_digests(
            tuple(artifact for artifact, _, _ in closure_artifacts),
            "swap/implicit-mount raw artifact",
        )
        _require_component_role(
            self.measurement_producer,
            "measurement_producer",
            "swap and implicit-mount evidence",
        )
        _require_component_role(
            self.runtime_storage_escape_gate,
            "runtime_storage_escape_gate",
            "swap and implicit-mount evidence",
        )
        initial = _require_int(
            self.initial_observation_monotonic_ns,
            "initial swap observation monotonic time",
            minimum=1,
        )
        chronology = (
            initial,
            _require_int(self.go_commit_monotonic_ns, "GO commit monotonic time", minimum=1),
            _require_int(self.worker_exit_monotonic_ns, "worker exit monotonic time", minimum=1),
            _require_int(
                self.publication_wrapper_monotonic_ns,
                "publication wrapper monotonic time",
                minimum=1,
            ),
            _require_int(
                self.reload_validation_monotonic_ns,
                "reload validation monotonic time",
                minimum=1,
            ),
            _require_int(
                self.relay_preseal_monotonic_ns,
                "relay preseal monotonic time",
                minimum=1,
            ),
            _require_int(
                self.channel_preseal_monotonic_ns,
                "channel preseal monotonic time",
                minimum=1,
            ),
            _require_int(self.write_seal_monotonic_ns, "write seal monotonic time", minimum=1),
            _require_int(
                self.terminal_observation_monotonic_ns,
                "terminal swap observation monotonic time",
                minimum=1,
            ),
            _require_int(
                self.receipt_precommit_monotonic_ns,
                "storage receipt precommit monotonic time",
                minimum=1,
            ),
        )
        if any(right <= left for left, right in zip(chronology, chronology[1:])):
            _fail("storage evidence chronology is not strictly ordered")
        for value, label in (
            (self.memory_swap_max_bytes_pre_go, "memory.swap.max pre-GO"),
            (self.memory_swap_current_initial_bytes, "initial memory.swap.current"),
            (self.memory_swap_current_terminal_bytes, "terminal memory.swap.current"),
            (self.memory_swap_peak_initial_bytes, "initial memory.swap.peak"),
            (self.memory_swap_peak_terminal_bytes, "terminal memory.swap.peak"),
            (self.memory_zswap_current_initial_bytes, "initial memory.zswap.current"),
            (self.memory_zswap_current_terminal_bytes, "terminal memory.zswap.current"),
        ):
            _require_int(value, label, maximum=0)
        if self.memory_zswap_writeback_enabled_pre_go is not False:
            _fail("memory.zswap.writeback must be disabled before GO")
        if not all(
            (
                self.retained_counter_endpoints_same_outer_cgroup is True,
                self.counters_retained_from_pre_go_through_terminal is True,
            )
        ):
            _fail("swap counters were not retained on one exact outer cgroup")
        _require_int(
            self.application_writable_tmpfs_count,
            "application-writable tmpfs count",
            minimum=1,
            maximum=1,
        )
        if not all(
            (
                self.application_writable_tmpfs_is_aggregate_root is True,
                self.application_writable_tmpfs_allocatable is True,
            )
        ):
            _fail("the sole application-writable tmpfs is not the allocatable aggregate root")
        if not all(
            (
                self.docker_implicit_mount_inventory_complete is True,
                self.docker_implicit_mounts_all_read_only is True,
                self.docker_implicit_mounts_all_nonallocatable is True,
                self.docker_implicit_mounts_have_application_writable_path is False,
            )
        ):
            _fail("Docker implicit-mount closure evidence admits application allocation")
        if (
            self.persistent_storage_scope != PERSISTENT_STORAGE_SCOPE
            or self.persistent_storage_measurement_interval
            != PERSISTENT_STORAGE_MEASUREMENT_INTERVAL
            or self.trusted_runtime_bookkeeping_exclusions != TRUSTED_RUNTIME_BOOKKEEPING_EXCLUSIONS
        ):
            _fail("candidate-addressable persistent-storage scope differs")
        if self.docker_implicit_config_bind_paths != DOCKER_IMPLICIT_CONFIG_BIND_PATHS:
            _fail("Docker implicit configuration bind paths differ")
        _require_int(
            self.implicit_readonly_etc_bind_count,
            "implicit read-only /etc bind count",
            minimum=3,
            maximum=3,
        )
        for value, label in (
            (self.user_bind_mount_count, "user bind mount count"),
            (self.user_volume_mount_count, "user volume mount count"),
            (self.image_declared_volume_count, "image-declared volume count"),
            (self.added_device_count, "added device count"),
            (self.writable_persistent_mount_count, "writable persistent mount count"),
            (self.writable_persistent_fd_count, "writable persistent FD count"),
            (self.candidate_stdio_transport_count, "candidate stdio transport count"),
            (
                self.forbidden_docker_api_operation_count,
                "forbidden Docker API operation count",
            ),
            (self.rootfs_upperdir_interval_delta_bytes, "rootfs upperdir interval delta"),
            (self.docker_volume_inventory_delta, "Docker volume inventory delta"),
        ):
            _require_int(value, label, maximum=0)
        if self.docker_ipc_mode != "none":
            _fail("Docker IPC mode differs")
        if not all(
            (
                self.docker_shm_mount_present is False,
                self.docker_tty_enabled is False,
                self.docker_stdin_open is False,
                self.docker_exec_permitted is False,
                self.docker_archive_api_permitted is False,
                self.docker_api_candidate_accessible is False,
                self.container_console_or_fifo_candidate_accessible is False,
                self.default_device_inventory_exact is True,
                self.default_devices_can_allocate_storage is False,
                self.memfd_posix_or_sysv_shm_permitted is False,
                self.post_go_mount_mutation_permitted is False,
                self.candidate_cgroup_mutation_permitted is False,
                self.daemon_runtime_storage_candidate_accessible is False,
                self.host_archival_candidate_accessible is False,
                self.final_oci_spec_exact is True,
                self.custom_runtime_neutralized_stock_writable_implicit_mounts is True,
                self.structural_nonaddressability_complete is True,
                self.docker_api_allowlist_complete_and_lossless is True,
                self.device_and_ipc_confinement_active is True,
                self.rootfs_upperdir_candidate_inaccessible is True,
            )
        ):
            _fail("Docker/OCI structural storage closure differs")
        _assert_plain_unaliased_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "application_writable_tmpfs_count": self.application_writable_tmpfs_count,
            "application_writable_tmpfs_allocatable": (self.application_writable_tmpfs_allocatable),
            "application_writable_tmpfs_is_aggregate_root": (
                self.application_writable_tmpfs_is_aggregate_root
            ),
            "counters_retained_from_pre_go_through_terminal": (
                self.counters_retained_from_pre_go_through_terminal
            ),
            "docker_implicit_mounts_all_nonallocatable": (
                self.docker_implicit_mounts_all_nonallocatable
            ),
            "docker_implicit_mounts_all_read_only": self.docker_implicit_mounts_all_read_only,
            "docker_implicit_mounts_have_application_writable_path": (
                self.docker_implicit_mounts_have_application_writable_path
            ),
            "docker_implicit_mount_inventory_complete": (
                self.docker_implicit_mount_inventory_complete
            ),
            "persistent_storage_scope": self.persistent_storage_scope,
            "persistent_storage_measurement_interval": (
                self.persistent_storage_measurement_interval
            ),
            "trusted_runtime_bookkeeping_exclusions": list(
                self.trusted_runtime_bookkeeping_exclusions
            ),
            "docker_implicit_config_bind_paths": list(self.docker_implicit_config_bind_paths),
            "implicit_readonly_etc_bind_count": self.implicit_readonly_etc_bind_count,
            "user_bind_mount_count": self.user_bind_mount_count,
            "user_volume_mount_count": self.user_volume_mount_count,
            "image_declared_volume_count": self.image_declared_volume_count,
            "added_device_count": self.added_device_count,
            "writable_persistent_mount_count": self.writable_persistent_mount_count,
            "writable_persistent_fd_count": self.writable_persistent_fd_count,
            "candidate_stdio_transport_count": self.candidate_stdio_transport_count,
            "forbidden_docker_api_operation_count": (self.forbidden_docker_api_operation_count),
            "rootfs_upperdir_interval_delta_bytes": self.rootfs_upperdir_interval_delta_bytes,
            "docker_volume_inventory_delta": self.docker_volume_inventory_delta,
            "docker_ipc_mode": self.docker_ipc_mode,
            "docker_shm_mount_present": self.docker_shm_mount_present,
            "docker_tty_enabled": self.docker_tty_enabled,
            "docker_stdin_open": self.docker_stdin_open,
            "docker_exec_permitted": self.docker_exec_permitted,
            "docker_archive_api_permitted": self.docker_archive_api_permitted,
            "docker_api_candidate_accessible": self.docker_api_candidate_accessible,
            "container_console_or_fifo_candidate_accessible": (
                self.container_console_or_fifo_candidate_accessible
            ),
            "default_device_inventory_exact": self.default_device_inventory_exact,
            "default_devices_can_allocate_storage": (self.default_devices_can_allocate_storage),
            "memfd_posix_or_sysv_shm_permitted": self.memfd_posix_or_sysv_shm_permitted,
            "post_go_mount_mutation_permitted": self.post_go_mount_mutation_permitted,
            "candidate_cgroup_mutation_permitted": (self.candidate_cgroup_mutation_permitted),
            "daemon_runtime_storage_candidate_accessible": (
                self.daemon_runtime_storage_candidate_accessible
            ),
            "host_archival_candidate_accessible": self.host_archival_candidate_accessible,
            "final_oci_spec_exact": self.final_oci_spec_exact,
            "custom_runtime_neutralized_stock_writable_implicit_mounts": (
                self.custom_runtime_neutralized_stock_writable_implicit_mounts
            ),
            "structural_nonaddressability_complete": self.structural_nonaddressability_complete,
            "docker_api_allowlist_complete_and_lossless": (
                self.docker_api_allowlist_complete_and_lossless
            ),
            "device_and_ipc_confinement_active": self.device_and_ipc_confinement_active,
            "rootfs_upperdir_candidate_inaccessible": (self.rootfs_upperdir_candidate_inaccessible),
            "initial_observation_monotonic_ns": self.initial_observation_monotonic_ns,
            "measurement_interval_start": self.measurement_interval_start.to_dict(),
            "measurement_interval_end": self.measurement_interval_end.to_dict(),
            "docker_volume_inventory_pre_go_baseline": (
                self.docker_volume_inventory_pre_go_baseline.to_dict()
            ),
            "go_commit_monotonic_ns": self.go_commit_monotonic_ns,
            "worker_exit_monotonic_ns": self.worker_exit_monotonic_ns,
            "publication_wrapper_monotonic_ns": self.publication_wrapper_monotonic_ns,
            "reload_validation_monotonic_ns": self.reload_validation_monotonic_ns,
            "relay_preseal_monotonic_ns": self.relay_preseal_monotonic_ns,
            "channel_preseal_monotonic_ns": self.channel_preseal_monotonic_ns,
            "write_seal_monotonic_ns": self.write_seal_monotonic_ns,
            "memory_swap_current_initial_bytes": self.memory_swap_current_initial_bytes,
            "memory_swap_current_terminal_bytes": self.memory_swap_current_terminal_bytes,
            "memory_swap_max_bytes_pre_go": self.memory_swap_max_bytes_pre_go,
            "memory_swap_peak_initial_bytes": self.memory_swap_peak_initial_bytes,
            "memory_swap_peak_terminal_bytes": self.memory_swap_peak_terminal_bytes,
            "memory_zswap_current_initial_bytes": self.memory_zswap_current_initial_bytes,
            "memory_zswap_current_terminal_bytes": self.memory_zswap_current_terminal_bytes,
            "memory_zswap_writeback_enabled_pre_go": (self.memory_zswap_writeback_enabled_pre_go),
            "outer_cgroup_identity_sha256": self.outer_cgroup_identity_sha256,
            "raw_docker_implicit_mount_inventory": (
                self.raw_docker_implicit_mount_inventory.to_dict()
            ),
            "raw_docker_create_inspect": self.raw_docker_create_inspect.to_dict(),
            "raw_final_oci_spec": self.raw_final_oci_spec.to_dict(),
            "raw_console_stdio_inventory": self.raw_console_stdio_inventory.to_dict(),
            "raw_docker_api_operation_journal": (self.raw_docker_api_operation_journal.to_dict()),
            "raw_rootfs_upperdir_pre_go_baseline": (
                self.raw_rootfs_upperdir_pre_go_baseline.to_dict()
            ),
            "raw_rootfs_upperdir_interval_delta": (
                self.raw_rootfs_upperdir_interval_delta.to_dict()
            ),
            "raw_docker_volume_inventory_delta": (self.raw_docker_volume_inventory_delta.to_dict()),
            "measurement_producer": self.measurement_producer.to_dict(),
            "runtime_storage_escape_gate": self.runtime_storage_escape_gate.to_dict(),
            "raw_memory_swap_max_pre_go": self.raw_memory_swap_max_pre_go.to_dict(),
            "raw_memory_zswap_writeback_pre_go": (self.raw_memory_zswap_writeback_pre_go.to_dict()),
            "raw_swap_counters_initial": self.raw_swap_counters_initial.to_dict(),
            "raw_swap_counters_terminal": self.raw_swap_counters_terminal.to_dict(),
            "retained_counter_endpoints_same_outer_cgroup": (
                self.retained_counter_endpoints_same_outer_cgroup
            ),
            "terminal_observation_monotonic_ns": self.terminal_observation_monotonic_ns,
            "receipt_precommit_monotonic_ns": self.receipt_precommit_monotonic_ns,
        }


def _validate_current_swap_and_implicit_mount_closure_evidence_v1(
    evidence: object,
) -> SwapAndImplicitMountClosureEvidenceV1:
    if type(evidence) is not SwapAndImplicitMountClosureEvidenceV1:
        _fail("swap and implicit-mount closure evidence type differs")
    exact = evidence
    _translate_current_state_error(
        "swap and implicit-mount closure evidence",
        lambda: SwapAndImplicitMountClosureEvidenceV1.__post_init__(exact),
    )
    return exact


@dataclass(frozen=True, slots=True)
class StorageBoundaryReceiptV2:
    """Post-seal storage receipt; terminal/lifecycle/merger bindings are forbidden."""

    campaign_id: str
    case_ordinal: int
    candidate_id: str
    candidate_family: str
    qualification_case_id: str
    image_id: str
    handshake: HostV3HandshakeProjectionV1
    policy: ArtifactIdentityV1
    runtime_intent: ArtifactIdentityV1
    host_go_storage_intent_binding: HostGoStorageIntentBindingV1
    publication_wrapper: ArtifactIdentityV1
    publication_reload_validation: ArtifactIdentityV1
    terminal_relay_preseal_attestation: ArtifactIdentityV1
    nonstorage_channel_preseal_attestation: ArtifactIdentityV1
    irreversible_write_seal: ArtifactIdentityV1
    tmpfs_conservative_bound_evidence: TmpfsConservativeBoundEvidenceV1
    disk_absence_evidence: DiskStructuralAbsenceEvidenceV1
    swap_and_implicit_mount_closure_evidence: SwapAndImplicitMountClosureEvidenceV1
    raw_publication_reload: ArtifactIdentityV1
    raw_write_seal: ArtifactIdentityV1
    publication_reload_projection: RawArtifactProjectionV1
    write_seal_projection: RawArtifactProjectionV1
    terminal_relay_preseal_producer: PinnedStorageComponentIdentityV1
    nonstorage_channel_preseal_producer: PinnedStorageComponentIdentityV1
    raw_artifacts: tuple[ArtifactIdentityV1, ...]
    max_temporary_peak_bytes: int
    max_disk_peak_bytes: int
    components: tuple[PinnedStorageComponentIdentityV1, ...]
    schema_version: str = STORAGE_BOUNDARY_RECEIPT_V2_SCHEMA_VERSION
    status: str = STORAGE_RECEIPT_STATUS
    publication_wrapper_order: int = 0
    reload_validation_order: int = 1
    relay_preseal_order: int = 2
    channel_preseal_order: int = 3
    write_seal_order: int = 4
    receipt_commit_order: int = 5
    reload_read_only: bool = True
    publication_projection_exact: bool = True
    write_seal_irreversible: bool = True
    later_measured_writes_possible: bool = False
    terminal_bound: bool = False
    lifecycle_bound: bool = False
    merger_bound: bool = False

    def __post_init__(self) -> None:
        if self.schema_version != STORAGE_BOUNDARY_RECEIPT_V2_SCHEMA_VERSION:
            _fail("storage receipt schema differs")
        if self.status != STORAGE_RECEIPT_STATUS:
            _fail("storage receipt status differs")
        _require_identifier(self.campaign_id, "receipt campaign ID")
        _candidate_projection(
            self.case_ordinal,
            self.candidate_id,
            self.candidate_family,
            self.qualification_case_id,
        )
        _require_image_id(self.image_id, "receipt image ID")
        _validate_current_host_v3_handshake_projection_v1(self.handshake)
        for name in (
            "campaign_id",
            "case_ordinal",
            "candidate_id",
            "qualification_case_id",
            "image_id",
        ):
            if getattr(self.handshake, name) != getattr(self, name):
                _fail(f"receipt and host handshake {name} differ")
        _validate_current_host_go_storage_intent_binding_v1(
            self.host_go_storage_intent_binding
        )
        if (
            self.host_go_storage_intent_binding.host_go != self.handshake.go
            or self.host_go_storage_intent_binding.runtime_intent != self.runtime_intent
        ):
            _fail("host GO does not bind this receipt's exact runtime intent")
        for artifact, schema, label in (
            (self.policy, STORAGE_BACKEND_POLICY_SCHEMA_VERSION, "storage policy"),
            (
                self.runtime_intent,
                STORAGE_BOUNDARY_RUNTIME_INTENT_SCHEMA_VERSION,
                "runtime intent",
            ),
            (
                self.publication_wrapper,
                NORMALIZED_PUBLICATION_WRAPPER_SCHEMA_VERSION,
                "publication wrapper",
            ),
            (
                self.publication_reload_validation,
                PUBLICATION_RELOAD_VALIDATION_SCHEMA_VERSION,
                "reload validation",
            ),
            (
                self.terminal_relay_preseal_attestation,
                TERMINAL_RELAY_PRESEAL_ATTESTATION_SCHEMA_VERSION,
                "relay preseal attestation",
            ),
            (
                self.nonstorage_channel_preseal_attestation,
                NONSTORAGE_CHANNEL_PRESEAL_ATTESTATION_SCHEMA_VERSION,
                "channel preseal attestation",
            ),
            (
                self.irreversible_write_seal,
                IRREVERSIBLE_WRITE_SEAL_SCHEMA_VERSION,
                "irreversible write seal",
            ),
        ):
            _require_artifact_identity(artifact, label, expected_schema=schema)
        first_class_artifacts = (
            self.handshake.request,
            self.handshake.intent,
            self.handshake.ready,
            self.handshake.observer_anchor,
            self.handshake.go,
            self.policy,
            self.runtime_intent,
            self.publication_wrapper,
            self.publication_reload_validation,
            self.terminal_relay_preseal_attestation,
            self.nonstorage_channel_preseal_attestation,
            self.irreversible_write_seal,
        )
        _require_distinct_artifact_digests(
            first_class_artifacts,
            "receipt first-class artifact",
        )
        _validate_current_raw_artifact_projection_v1(self.publication_reload_projection)
        _validate_current_raw_artifact_projection_v1(self.write_seal_projection)
        if (
            self.publication_reload_projection.kind != "publication_reload"
            or self.publication_reload_projection.first_class_artifact
            != self.publication_reload_validation
            or self.publication_reload_projection.raw_artifact != self.raw_publication_reload
            or self.publication_reload_projection.predecessors != (self.publication_wrapper,)
            or self.write_seal_projection.kind != "write_seal"
            or self.write_seal_projection.first_class_artifact != self.irreversible_write_seal
            or self.write_seal_projection.raw_artifact != self.raw_write_seal
            or self.write_seal_projection.predecessors
            != (
                self.publication_reload_validation,
                self.terminal_relay_preseal_attestation,
                self.nonstorage_channel_preseal_attestation,
            )
        ):
            _fail("reload or write-seal raw projection crosswires receipt artifacts")
        _validate_current_tmpfs_conservative_bound_evidence_v1(
            self.tmpfs_conservative_bound_evidence
        )
        _validate_current_disk_structural_absence_evidence_v1(self.disk_absence_evidence)
        _validate_current_swap_and_implicit_mount_closure_evidence_v1(
            self.swap_and_implicit_mount_closure_evidence
        )
        for evidence, label in (
            (self.tmpfs_conservative_bound_evidence, "tmpfs"),
            (self.disk_absence_evidence, "disk"),
            (self.swap_and_implicit_mount_closure_evidence, "swap/implicit-mount"),
        ):
            if (
                evidence.measurement_interval_start != self.handshake.go
                or evidence.measurement_interval_end != self.irreversible_write_seal
            ):
                _fail(f"{label} interval does not bind host GO through write seal")
        for artifact, schema, label in (
            (
                self.raw_publication_reload,
                RAW_ARTIFACT_SCHEMA_INVENTORY[6],
                "raw publication reload",
            ),
            (self.raw_write_seal, RAW_ARTIFACT_SCHEMA_INVENTORY[7], "raw write seal"),
        ):
            _require_artifact_identity(artifact, label, expected_schema=schema)
        if type(self.raw_artifacts) is not tuple:
            _fail("raw artifacts must be one exact tuple")
        if len(self.raw_artifacts) != len(RAW_ARTIFACT_SCHEMA_INVENTORY):
            _fail("raw artifact inventory differs or is incomplete")
        for index, artifact in enumerate(self.raw_artifacts):
            _require_artifact_identity(artifact, f"raw artifact {index}")
        if tuple(artifact.schema_version for artifact in self.raw_artifacts) != (
            RAW_ARTIFACT_SCHEMA_INVENTORY
        ):
            _fail("raw artifact inventory differs or is incomplete")
        _require_distinct_artifact_digests(self.raw_artifacts, "raw artifact")
        all_receipt_artifacts = first_class_artifacts + self.raw_artifacts
        _require_distinct_artifact_digests(
            all_receipt_artifacts,
            "receipt first-class and raw artifact",
        )
        typed_raw = (
            self.tmpfs_conservative_bound_evidence.raw_mountinfo,
            self.tmpfs_conservative_bound_evidence.raw_statfs_diagnostic_samples,
            self.tmpfs_conservative_bound_evidence.raw_hard_limit_mount_mutation_closure,
            self.disk_absence_evidence.raw_mountinfo,
            self.disk_absence_evidence.raw_absence_scan,
            self.disk_absence_evidence.raw_writable_fd_inventory,
            self.raw_publication_reload,
            self.raw_write_seal,
            self.swap_and_implicit_mount_closure_evidence.raw_memory_swap_max_pre_go,
            self.swap_and_implicit_mount_closure_evidence.raw_swap_counters_initial,
            self.swap_and_implicit_mount_closure_evidence.raw_swap_counters_terminal,
            self.swap_and_implicit_mount_closure_evidence.raw_memory_zswap_writeback_pre_go,
            self.swap_and_implicit_mount_closure_evidence.raw_docker_implicit_mount_inventory,
            self.swap_and_implicit_mount_closure_evidence.raw_docker_create_inspect,
            self.swap_and_implicit_mount_closure_evidence.raw_final_oci_spec,
            self.swap_and_implicit_mount_closure_evidence.raw_console_stdio_inventory,
            self.swap_and_implicit_mount_closure_evidence.raw_docker_api_operation_journal,
            self.swap_and_implicit_mount_closure_evidence.raw_rootfs_upperdir_pre_go_baseline,
            self.swap_and_implicit_mount_closure_evidence.raw_rootfs_upperdir_interval_delta,
            self.swap_and_implicit_mount_closure_evidence.raw_docker_volume_inventory_delta,
        )
        if typed_raw != self.raw_artifacts:
            _fail("typed storage evidence differs from its ordered raw inventory")
        temporary = _require_int(
            self.max_temporary_peak_bytes, "field-24 published value", minimum=1
        )
        if temporary != self.tmpfs_conservative_bound_evidence.hard_limit_bytes:
            _fail("field 24 must publish the conservative enforced hard limit")
        _require_int(self.max_disk_peak_bytes, "field-25 published value", maximum=0)
        _validate_components(self.components)
        measurement_producer = self.components[0]
        terminal_relay = self.components[1]
        nonstorage_channel = self.components[2]
        write_seal_producer = self.components[3]
        runtime_escape_gate = self.components[4]
        if (
            self.host_go_storage_intent_binding.verifier != measurement_producer
            or self.tmpfs_conservative_bound_evidence.measurement_producer != measurement_producer
            or self.disk_absence_evidence.measurement_producer != measurement_producer
            or self.swap_and_implicit_mount_closure_evidence.measurement_producer
            != measurement_producer
            or self.swap_and_implicit_mount_closure_evidence.runtime_storage_escape_gate
            != runtime_escape_gate
            or self.publication_reload_projection.producer != terminal_relay
            or self.write_seal_projection.producer != write_seal_producer
            or self.terminal_relay_preseal_producer != terminal_relay
            or self.nonstorage_channel_preseal_producer != nonstorage_channel
        ):
            _fail("storage evidence or attestation producer binding differs")
        _require_component_role(
            self.terminal_relay_preseal_producer,
            "terminal_relay",
            "terminal relay preseal attestation",
        )
        _require_component_role(
            self.nonstorage_channel_preseal_producer,
            "nonstorage_channel",
            "nonstorage channel preseal attestation",
        )
        chronology = (
            self.publication_wrapper_order,
            self.reload_validation_order,
            self.relay_preseal_order,
            self.channel_preseal_order,
            self.write_seal_order,
            self.receipt_commit_order,
        )
        if any(type(value) is not int for value in chronology) or chronology != (0, 1, 2, 3, 4, 5):
            _fail("publication/reload/relay/channel/seal/receipt chronology differs")
        if not all(
            (
                self.reload_read_only is True,
                self.publication_projection_exact is True,
                self.write_seal_irreversible is True,
                self.later_measured_writes_possible is False,
                self.terminal_bound is False,
                self.lifecycle_bound is False,
                self.merger_bound is False,
            )
        ):
            _fail("receipt seal or reverse-binding posture differs")
        _assert_plain_unaliased_json(self.to_body_dict())

    def to_body_dict(self) -> dict[str, Any]:
        return {
            "campaign_id": self.campaign_id,
            "candidate_family": self.candidate_family,
            "candidate_id": self.candidate_id,
            "case_ordinal": self.case_ordinal,
            "channel_preseal_order": self.channel_preseal_order,
            "components": [component.to_dict() for component in self.components],
            "disk_absence_evidence": self.disk_absence_evidence.to_dict(),
            "handshake": self.handshake.to_dict(),
            "host_go_storage_intent_binding": (self.host_go_storage_intent_binding.to_dict()),
            "image_id": self.image_id,
            "irreversible_write_seal": self.irreversible_write_seal.to_dict(),
            "later_measured_writes_possible": self.later_measured_writes_possible,
            "lifecycle_bound": self.lifecycle_bound,
            "max_disk_peak_bytes": self.max_disk_peak_bytes,
            "max_temporary_peak_bytes": self.max_temporary_peak_bytes,
            "merger_bound": self.merger_bound,
            "nonstorage_channel_preseal_attestation": (
                self.nonstorage_channel_preseal_attestation.to_dict()
            ),
            "policy": self.policy.to_dict(),
            "publication_projection_exact": self.publication_projection_exact,
            "publication_reload_validation": self.publication_reload_validation.to_dict(),
            "publication_wrapper": self.publication_wrapper.to_dict(),
            "publication_wrapper_order": self.publication_wrapper_order,
            "qualification_case_id": self.qualification_case_id,
            "raw_publication_reload": self.raw_publication_reload.to_dict(),
            "publication_reload_projection": self.publication_reload_projection.to_dict(),
            "raw_artifacts": [artifact.to_dict() for artifact in self.raw_artifacts],
            "raw_write_seal": self.raw_write_seal.to_dict(),
            "write_seal_projection": self.write_seal_projection.to_dict(),
            "terminal_relay_preseal_producer": self.terminal_relay_preseal_producer.to_dict(),
            "nonstorage_channel_preseal_producer": (
                self.nonstorage_channel_preseal_producer.to_dict()
            ),
            "receipt_commit_order": self.receipt_commit_order,
            "relay_preseal_order": self.relay_preseal_order,
            "reload_read_only": self.reload_read_only,
            "reload_validation_order": self.reload_validation_order,
            "runtime_intent": self.runtime_intent.to_dict(),
            "schema_version": self.schema_version,
            "status": self.status,
            "swap_and_implicit_mount_closure_evidence": (
                self.swap_and_implicit_mount_closure_evidence.to_dict()
            ),
            "terminal_bound": self.terminal_bound,
            "terminal_relay_preseal_attestation": (
                self.terminal_relay_preseal_attestation.to_dict()
            ),
            "tmpfs_conservative_bound_evidence": (self.tmpfs_conservative_bound_evidence.to_dict()),
            "write_seal_irreversible": self.write_seal_irreversible,
            "write_seal_order": self.write_seal_order,
        }


def _validate_current_storage_boundary_receipt_v2(
    receipt: object,
) -> StorageBoundaryReceiptV2:
    if type(receipt) is not StorageBoundaryReceiptV2:
        _fail("storage receipt type differs")
    exact = receipt
    _translate_current_state_error(
        "storage receipt",
        lambda: StorageBoundaryReceiptV2.__post_init__(exact),
    )
    return exact


@dataclass(frozen=True, slots=True)
class StorageCleanupReconciliationV1:
    """Deletion-only cleanup or explicit residual-state uncertainty."""

    campaign_id: str
    case_ordinal: int
    candidate_id: str
    candidate_family: str
    qualification_case_id: str
    runtime_intent: ArtifactIdentityV1
    outcome: Literal[
        "committed_receipt_cleaned",
        "failed_before_receipt_cleaned",
        "receipt_commit_uncertain_cleaned",
        "committed_receipt_cleanup_failed",
        "failed_before_receipt_cleanup_failed",
        "receipt_commit_uncertain_cleanup_failed",
    ]
    receipt: ArtifactIdentityV1 | None
    attempted_receipt: ArtifactIdentityV1 | None
    failure_frontier: ArtifactIdentityV1 | None
    namespace_cleanup_receipt: ArtifactIdentityV1 | None
    cleanup_failure_frontier: ArtifactIdentityV1 | None
    cleanup_producer: PinnedStorageComponentIdentityV1
    cleanup_complete: bool
    residual_storage_state: Literal["absent", "unknown"]
    tmpfs_unmounted_before_namespace_release: bool | None
    aggregate_mount_id_absent_before_namespace_release: bool | None
    underlying_aggregate_path_read_only: bool | None
    namespace_process_count: int | None
    retained_namespace_fd_count_after_release: int | None
    schema_version: str = STORAGE_CLEANUP_RECONCILIATION_SCHEMA_VERSION
    status: str = STORAGE_CLEANUP_STATUS
    deletion_only: bool = True
    retained_writable_path_count: int | None = 0
    receipt_committed: bool = False
    receipt_commit_uncertain: bool = False
    consumed: bool = True
    retry_allowed: bool = False
    synthesized_temporary_peak_bytes: int | None = None
    synthesized_disk_peak_bytes: int | None = None

    def __post_init__(self) -> None:
        if self.schema_version != STORAGE_CLEANUP_RECONCILIATION_SCHEMA_VERSION:
            _fail("storage cleanup schema differs")
        if self.status != STORAGE_CLEANUP_STATUS:
            _fail("storage cleanup status differs")
        _require_identifier(self.campaign_id, "cleanup campaign ID")
        _candidate_projection(
            self.case_ordinal,
            self.candidate_id,
            self.candidate_family,
            self.qualification_case_id,
        )
        _require_artifact_identity(
            self.runtime_intent,
            "cleanup runtime intent",
            expected_schema=STORAGE_BOUNDARY_RUNTIME_INTENT_SCHEMA_VERSION,
        )
        for artifact, schema, label in (
            (self.receipt, STORAGE_BOUNDARY_RECEIPT_V2_SCHEMA_VERSION, "cleanup receipt"),
            (
                self.attempted_receipt,
                STORAGE_BOUNDARY_RECEIPT_V2_SCHEMA_VERSION,
                "cleanup attempted receipt",
            ),
            (
                self.failure_frontier,
                STORAGE_OPERATIONAL_FAILURE_SCHEMA_VERSION,
                "cleanup failure frontier",
            ),
            (
                self.namespace_cleanup_receipt,
                STORAGE_NAMESPACE_CLEANUP_RECEIPT_SCHEMA_VERSION,
                "namespace cleanup receipt",
            ),
            (
                self.cleanup_failure_frontier,
                STORAGE_CLEANUP_FAILURE_FRONTIER_SCHEMA_VERSION,
                "cleanup failure frontier",
            ),
        ):
            if artifact is not None:
                _require_artifact_identity(artifact, label, expected_schema=schema)
        cleanup_artifacts = (self.runtime_intent,) + tuple(
            artifact
            for artifact in (
                self.receipt,
                self.attempted_receipt,
                self.failure_frontier,
                self.namespace_cleanup_receipt,
                self.cleanup_failure_frontier,
            )
            if artifact is not None
        )
        _require_distinct_artifact_digests(cleanup_artifacts, "cleanup artifact")
        _require_component_role(
            self.cleanup_producer,
            "namespace_cleanup_producer",
            "storage cleanup reconciliation",
        )
        _require_identifier(self.outcome, "cleanup outcome")
        committed_outcomes = {
            "committed_receipt_cleaned",
            "committed_receipt_cleanup_failed",
        }
        failed_outcomes = {
            "failed_before_receipt_cleaned",
            "failed_before_receipt_cleanup_failed",
        }
        uncertain_outcomes = {
            "receipt_commit_uncertain_cleaned",
            "receipt_commit_uncertain_cleanup_failed",
        }
        if self.outcome in committed_outcomes:
            if (
                self.receipt is None
                or self.attempted_receipt is not None
                or self.failure_frontier is not None
                or self.receipt_committed is not True
                or self.receipt_commit_uncertain is not False
            ):
                _fail("committed cleanup does not bind exactly one committed receipt")
        elif self.outcome in failed_outcomes:
            if (
                self.receipt is not None
                or self.attempted_receipt is not None
                or self.failure_frontier is None
                or self.receipt_committed is not False
                or self.receipt_commit_uncertain is not False
            ):
                _fail("failed cleanup state differs")
        elif self.outcome in uncertain_outcomes:
            if (
                self.receipt is not None
                or self.failure_frontier is None
                or self.attempted_receipt is None
                or self.receipt_committed is not False
                or self.receipt_commit_uncertain is not True
            ):
                _fail("uncertain cleanup state differs")
        else:
            _fail("cleanup outcome differs")
        if not all(
            (self.deletion_only is True, self.consumed is True, self.retry_allowed is False)
        ):
            _fail("cleanup is not deletion-only, consumed, and nonretryable")
        cleaned = self.outcome.endswith("_cleaned")
        if cleaned:
            if (
                self.namespace_cleanup_receipt is None
                or self.cleanup_failure_frontier is not None
                or self.cleanup_complete is not True
                or self.residual_storage_state != "absent"
                or self.tmpfs_unmounted_before_namespace_release is not True
                or self.aggregate_mount_id_absent_before_namespace_release is not True
                or self.underlying_aggregate_path_read_only is not True
            ):
                _fail("cleaned reconciliation lacks exact namespace teardown proof")
            _require_int(self.namespace_process_count, "namespace process count", maximum=0)
            _require_int(
                self.retained_namespace_fd_count_after_release,
                "retained namespace FD count after release",
                maximum=0,
            )
            _require_int(
                self.retained_writable_path_count,
                "retained writable path count",
                maximum=0,
            )
        else:
            if (
                self.namespace_cleanup_receipt is not None
                or self.cleanup_failure_frontier is None
                or self.cleanup_complete is not False
                or self.residual_storage_state != "unknown"
                or self.tmpfs_unmounted_before_namespace_release is not None
                or self.aggregate_mount_id_absent_before_namespace_release is not None
                or self.underlying_aggregate_path_read_only is not None
                or self.namespace_process_count is not None
                or self.retained_namespace_fd_count_after_release is not None
                or self.retained_writable_path_count is not None
            ):
                _fail("failed cleanup must preserve residual-state uncertainty")
        if (
            self.synthesized_temporary_peak_bytes is not None
            or self.synthesized_disk_peak_bytes is not None
        ):
            _fail("cleanup must never synthesize storage values")
        _assert_plain_unaliased_json(self.to_body_dict())

    def to_body_dict(self) -> dict[str, Any]:
        return {
            "aggregate_mount_id_absent_before_namespace_release": (
                self.aggregate_mount_id_absent_before_namespace_release
            ),
            "attempted_receipt": (
                None if self.attempted_receipt is None else self.attempted_receipt.to_dict()
            ),
            "campaign_id": self.campaign_id,
            "candidate_family": self.candidate_family,
            "candidate_id": self.candidate_id,
            "case_ordinal": self.case_ordinal,
            "cleanup_complete": self.cleanup_complete,
            "cleanup_failure_frontier": (
                None
                if self.cleanup_failure_frontier is None
                else self.cleanup_failure_frontier.to_dict()
            ),
            "cleanup_producer": self.cleanup_producer.to_dict(),
            "consumed": self.consumed,
            "deletion_only": self.deletion_only,
            "failure_frontier": (
                None if self.failure_frontier is None else self.failure_frontier.to_dict()
            ),
            "outcome": self.outcome,
            "namespace_cleanup_receipt": (
                None
                if self.namespace_cleanup_receipt is None
                else self.namespace_cleanup_receipt.to_dict()
            ),
            "namespace_process_count": self.namespace_process_count,
            "qualification_case_id": self.qualification_case_id,
            "receipt": None if self.receipt is None else self.receipt.to_dict(),
            "receipt_commit_uncertain": self.receipt_commit_uncertain,
            "receipt_committed": self.receipt_committed,
            "residual_storage_state": self.residual_storage_state,
            "retained_namespace_fd_count_after_release": (
                self.retained_namespace_fd_count_after_release
            ),
            "retained_writable_path_count": self.retained_writable_path_count,
            "retry_allowed": self.retry_allowed,
            "runtime_intent": self.runtime_intent.to_dict(),
            "schema_version": self.schema_version,
            "status": self.status,
            "synthesized_disk_peak_bytes": self.synthesized_disk_peak_bytes,
            "synthesized_temporary_peak_bytes": self.synthesized_temporary_peak_bytes,
            "tmpfs_unmounted_before_namespace_release": (
                self.tmpfs_unmounted_before_namespace_release
            ),
            "underlying_aggregate_path_read_only": self.underlying_aggregate_path_read_only,
        }


def _validate_current_storage_cleanup_reconciliation_v1(
    cleanup: object,
) -> StorageCleanupReconciliationV1:
    if type(cleanup) is not StorageCleanupReconciliationV1:
        _fail("storage cleanup type differs")
    exact = cleanup
    _translate_current_state_error(
        "storage cleanup reconciliation",
        lambda: StorageCleanupReconciliationV1.__post_init__(exact),
    )
    return exact


@dataclass(frozen=True, slots=True)
class StorageRuntimeIntentExternalBindingsV1:
    """Independent phase-6 provider identities required before host GO."""

    campaign_id: str
    case_ordinal: int
    candidate_id: str
    candidate_family: str
    qualification_case_id: str
    image_id: str
    container_name: str
    container_id_commitment_sha256: str
    outer_cgroup_identity_sha256: str
    qualification_plan: ArtifactIdentityV1
    policy: ArtifactIdentityV1
    runtime_qualification_receipt: ArtifactIdentityV1
    host_provisioning_v3_validated_pre_go_prefix: ArtifactIdentityV1
    mount_namespace_identity: ArtifactIdentityV1
    rootfs_mount_identity: ArtifactIdentityV1
    tmpfs_mount_identity: ArtifactIdentityV1
    tmpfs_backing_identity: ArtifactIdentityV1
    mount_inventory: ArtifactIdentityV1
    path_inventory: ArtifactIdentityV1
    storage_root_inventory: ArtifactIdentityV1
    field_inventory: ArtifactIdentityV1
    raw_schema_inventory: ArtifactIdentityV1
    outer_cgroup_memory_swap_max_pre_go: ArtifactIdentityV1
    outer_cgroup_swap_counters_initial: ArtifactIdentityV1
    outer_cgroup_memory_zswap_writeback_pre_go: ArtifactIdentityV1
    docker_implicit_mount_inventory: ArtifactIdentityV1
    docker_create_inspect: ArtifactIdentityV1
    final_oci_spec: ArtifactIdentityV1
    console_stdio_inventory: ArtifactIdentityV1
    rootfs_upperdir_pre_go_baseline: ArtifactIdentityV1
    docker_volume_inventory_pre_go_baseline: ArtifactIdentityV1
    max_temporary_peak_bytes: int
    aggregate_root_case_exclusive: bool
    components: tuple[PinnedStorageComponentIdentityV1, ...]

    def __post_init__(self) -> None:
        _require_identifier(self.campaign_id, "external runtime campaign ID")
        _candidate_projection(
            self.case_ordinal,
            self.candidate_id,
            self.candidate_family,
            self.qualification_case_id,
        )
        _require_image_id(self.image_id, "external runtime image ID")
        _require_identifier(self.container_name, "external runtime container name")
        _require_sha256(
            self.container_id_commitment_sha256,
            "external runtime container commitment",
        )
        _require_sha256(
            self.outer_cgroup_identity_sha256,
            "external runtime outer-cgroup identity",
        )
        if hmac.compare_digest(
            self.container_id_commitment_sha256,
            self.outer_cgroup_identity_sha256,
        ):
            _fail("external runtime container and outer-cgroup identities alias")
        _require_int(
            self.max_temporary_peak_bytes,
            "external runtime temporary hard ceiling",
            minimum=1,
        )
        if self.aggregate_root_case_exclusive is not True:
            _fail("external runtime aggregate root is not case-exclusive")
        _validate_components(self.components)
        external_artifacts = (
            (self.qualification_plan, QUALIFICATION_PLAN_V3_SCHEMA_VERSION),
            (self.policy, STORAGE_BACKEND_POLICY_SCHEMA_VERSION),
            (
                self.runtime_qualification_receipt,
                RUNTIME_QUALIFICATION_RECEIPT_SCHEMA_VERSION,
            ),
            (
                self.host_provisioning_v3_validated_pre_go_prefix,
                HOST_PROVISIONING_VALIDATED_PRE_GO_PREFIX_V3_SCHEMA_VERSION,
            ),
            (self.mount_namespace_identity, MOUNT_NAMESPACE_IDENTITY_SCHEMA_VERSION),
            (self.rootfs_mount_identity, ROOTFS_MOUNT_IDENTITY_SCHEMA_VERSION),
            (self.tmpfs_mount_identity, TMPFS_MOUNT_IDENTITY_SCHEMA_VERSION),
            (self.tmpfs_backing_identity, TMPFS_BACKING_IDENTITY_SCHEMA_VERSION),
            (self.mount_inventory, MOUNT_INVENTORY_SCHEMA_VERSION),
            (self.path_inventory, PATH_INVENTORY_SCHEMA_VERSION),
            (self.storage_root_inventory, STORAGE_ROOT_INVENTORY_SCHEMA_VERSION),
            (self.field_inventory, STORAGE_FIELD_INVENTORY_SCHEMA_VERSION),
            (self.raw_schema_inventory, RAW_SCHEMA_INVENTORY_IDENTITY_SCHEMA_VERSION),
            (
                self.outer_cgroup_memory_swap_max_pre_go,
                RAW_ARTIFACT_SCHEMA_INVENTORY[8],
            ),
            (
                self.outer_cgroup_swap_counters_initial,
                RAW_ARTIFACT_SCHEMA_INVENTORY[9],
            ),
            (
                self.outer_cgroup_memory_zswap_writeback_pre_go,
                RAW_ARTIFACT_SCHEMA_INVENTORY[11],
            ),
            (self.docker_implicit_mount_inventory, RAW_ARTIFACT_SCHEMA_INVENTORY[12]),
            (self.docker_create_inspect, RAW_ARTIFACT_SCHEMA_INVENTORY[13]),
            (self.final_oci_spec, RAW_ARTIFACT_SCHEMA_INVENTORY[14]),
            (self.console_stdio_inventory, RAW_ARTIFACT_SCHEMA_INVENTORY[15]),
            (self.rootfs_upperdir_pre_go_baseline, RAW_ARTIFACT_SCHEMA_INVENTORY[17]),
            (
                self.docker_volume_inventory_pre_go_baseline,
                DOCKER_VOLUME_INVENTORY_PRE_GO_BASELINE_SCHEMA_VERSION,
            ),
        )
        for artifact, schema in external_artifacts:
            _require_artifact_identity(
                artifact,
                "external runtime-intent artifact",
                expected_schema=schema,
            )
        _require_distinct_artifact_digests(
            tuple(artifact for artifact, _ in external_artifacts),
            "external runtime-intent artifact",
        )


def _validate_current_storage_runtime_intent_external_bindings_v1(
    bindings: object,
) -> StorageRuntimeIntentExternalBindingsV1:
    if type(bindings) is not StorageRuntimeIntentExternalBindingsV1:
        _fail("external runtime-intent binding type differs")
    exact = bindings
    _translate_current_state_error(
        "external runtime-intent binding",
        lambda: StorageRuntimeIntentExternalBindingsV1.__post_init__(exact),
    )
    return exact


@dataclass(frozen=True, slots=True)
class StorageReceiptExternalBindingsV1:
    """Independently parsed identities required to accept one storage receipt."""

    host_handshake: HostV3HandshakeProjectionV1
    host_go_storage_intent_binding: HostGoStorageIntentBindingV1
    publication_wrapper: ArtifactIdentityV1
    publication_reload_validation: ArtifactIdentityV1
    terminal_relay_preseal_attestation: ArtifactIdentityV1
    nonstorage_channel_preseal_attestation: ArtifactIdentityV1
    irreversible_write_seal: ArtifactIdentityV1
    raw_artifacts: tuple[ArtifactIdentityV1, ...]

    def __post_init__(self) -> None:
        _validate_current_host_v3_handshake_projection_v1(self.host_handshake)
        _validate_current_host_go_storage_intent_binding_v1(
            self.host_go_storage_intent_binding
        )
        if self.host_go_storage_intent_binding.host_go != self.host_handshake.go:
            _fail("external host-GO binding crosswires the host handshake")
        first_class_artifacts = (
            (
                self.host_handshake.request,
                HOST_CASE_REQUEST_V3_SCHEMA_VERSION,
                "host request",
            ),
            (
                self.host_handshake.intent,
                HOST_CASE_INTENT_V3_SCHEMA_VERSION,
                "host intent",
            ),
            (self.host_handshake.ready, HOST_READY_V3_SCHEMA_VERSION, "host READY"),
            (
                self.host_handshake.observer_anchor,
                HOST_OBSERVER_ANCHOR_V3_SCHEMA_VERSION,
                "host observer anchor",
            ),
            (self.host_handshake.go, HOST_GO_V3_SCHEMA_VERSION, "host GO"),
            (
                self.publication_wrapper,
                NORMALIZED_PUBLICATION_WRAPPER_SCHEMA_VERSION,
                "publication wrapper",
            ),
            (
                self.publication_reload_validation,
                PUBLICATION_RELOAD_VALIDATION_SCHEMA_VERSION,
                "publication reload validation",
            ),
            (
                self.terminal_relay_preseal_attestation,
                TERMINAL_RELAY_PRESEAL_ATTESTATION_SCHEMA_VERSION,
                "terminal relay preseal attestation",
            ),
            (
                self.nonstorage_channel_preseal_attestation,
                NONSTORAGE_CHANNEL_PRESEAL_ATTESTATION_SCHEMA_VERSION,
                "nonstorage channel preseal attestation",
            ),
            (
                self.irreversible_write_seal,
                IRREVERSIBLE_WRITE_SEAL_SCHEMA_VERSION,
                "irreversible write seal",
            ),
        )
        for artifact, schema, label in first_class_artifacts:
            _require_artifact_identity(artifact, f"external {label}", expected_schema=schema)
        _require_distinct_artifact_digests(
            tuple(artifact for artifact, _, _ in first_class_artifacts),
            "external first-class receipt artifact",
        )
        if type(self.raw_artifacts) is not tuple:
            _fail("external raw artifacts must be one exact ArtifactIdentity tuple")
        if len(self.raw_artifacts) != len(RAW_ARTIFACT_SCHEMA_INVENTORY):
            _fail("external raw artifact schema inventory differs")
        for index, artifact in enumerate(self.raw_artifacts):
            _require_artifact_identity(artifact, f"external raw artifact {index}")
        if tuple(artifact.schema_version for artifact in self.raw_artifacts) != (
            RAW_ARTIFACT_SCHEMA_INVENTORY
        ):
            _fail("external raw artifact schema inventory differs")
        _require_distinct_artifact_digests(self.raw_artifacts, "external raw artifact")
        all_artifacts = tuple(artifact for artifact, _, _ in first_class_artifacts) + (
            self.raw_artifacts
        )
        _require_distinct_artifact_digests(
            all_artifacts,
            "external first-class and raw receipt artifact",
        )


def _validate_current_storage_receipt_external_bindings_v1(
    bindings: object,
) -> StorageReceiptExternalBindingsV1:
    if type(bindings) is not StorageReceiptExternalBindingsV1:
        _fail("external receipt binding type differs")
    exact = bindings
    _translate_current_state_error(
        "external receipt binding",
        lambda: StorageReceiptExternalBindingsV1.__post_init__(exact),
    )
    return exact


@dataclass(frozen=True, slots=True)
class StorageCleanupExternalBindingsV1:
    """Independent operational identities required for cleanup reconciliation."""

    attempted_receipt: ArtifactIdentityV1 | None
    operational_failure_frontier: ArtifactIdentityV1 | None
    namespace_cleanup_receipt: ArtifactIdentityV1 | None
    cleanup_failure_frontier: ArtifactIdentityV1 | None
    cleanup_producer: PinnedStorageComponentIdentityV1

    def __post_init__(self) -> None:
        optional_artifacts = (
            (
                self.attempted_receipt,
                STORAGE_BOUNDARY_RECEIPT_V2_SCHEMA_VERSION,
                "attempted receipt",
            ),
            (
                self.operational_failure_frontier,
                STORAGE_OPERATIONAL_FAILURE_SCHEMA_VERSION,
                "operational failure frontier",
            ),
            (
                self.namespace_cleanup_receipt,
                STORAGE_NAMESPACE_CLEANUP_RECEIPT_SCHEMA_VERSION,
                "namespace cleanup receipt",
            ),
            (
                self.cleanup_failure_frontier,
                STORAGE_CLEANUP_FAILURE_FRONTIER_SCHEMA_VERSION,
                "cleanup failure frontier",
            ),
        )
        present: list[ArtifactIdentityV1] = []
        for artifact, schema, label in optional_artifacts:
            if artifact is not None:
                present.append(
                    _require_artifact_identity(
                        artifact,
                        f"external cleanup {label}",
                        expected_schema=schema,
                    )
                )
        _require_distinct_artifact_digests(tuple(present), "external cleanup artifact")
        if (self.namespace_cleanup_receipt is None) == (self.cleanup_failure_frontier is None):
            _fail("external cleanup must bind exactly one cleanup terminal artifact")
        if self.attempted_receipt is not None and self.operational_failure_frontier is None:
            _fail("external attempted receipt requires an operational failure frontier")
        _require_component_role(
            self.cleanup_producer,
            "namespace_cleanup_producer",
            "external cleanup binding",
        )


def _validate_current_storage_cleanup_external_bindings_v1(
    bindings: object,
) -> StorageCleanupExternalBindingsV1:
    if type(bindings) is not StorageCleanupExternalBindingsV1:
        _fail("external cleanup binding type differs")
    exact = bindings
    _translate_current_state_error(
        "external cleanup binding",
        lambda: StorageCleanupExternalBindingsV1.__post_init__(exact),
    )
    return exact


@dataclass(frozen=True, slots=True)
class StorageBackendContractDescriptorV2:
    """Self-description only; every operational capability remains false."""

    schema_version: str = STORAGE_BACKEND_CONTRACT_DESCRIPTOR_SCHEMA_VERSION
    status: str = STORAGE_DESCRIPTOR_STATUS
    artifact_schemas: tuple[str, ...] = (
        STORAGE_BACKEND_POLICY_SCHEMA_VERSION,
        STORAGE_BOUNDARY_RUNTIME_INTENT_SCHEMA_VERSION,
        STORAGE_BOUNDARY_RECEIPT_V2_SCHEMA_VERSION,
        STORAGE_CLEANUP_RECONCILIATION_SCHEMA_VERSION,
    )
    component_roles: tuple[str, ...] = STORAGE_COMPONENT_ROLES
    component_descriptor_schemas: tuple[str, ...] = STORAGE_COMPONENT_DESCRIPTOR_SCHEMA_INVENTORY
    phase_split: tuple[str, ...] = STORAGE_PHASE_SPLIT
    chronology: tuple[str, ...] = STORAGE_CHRONOLOGY
    resource_fields: tuple[str, ...] = RESOURCE_FIELDS
    storage_field_positions: tuple[int, int] = (FIELD24_POSITION, FIELD25_POSITION)
    raw_artifact_schemas: tuple[str, ...] = RAW_ARTIFACT_SCHEMA_INVENTORY
    required_external_artifact_schemas: tuple[str, ...] = REQUIRED_EXTERNAL_ARTIFACT_SCHEMAS
    capabilities: Mapping[str, bool] = SOURCE_ONLY_CAPABILITIES
    readiness: Mapping[str, bool] = SOURCE_ONLY_READINESS
    authority: Mapping[str, bool] = SOURCE_ONLY_AUTHORITY
    claims: Mapping[str, bool] = SOURCE_ONLY_CLAIMS
    operational_apis: tuple[()] = ()
    descriptor_self_pin_sha256: str = DESCRIPTOR_SERIALIZED_SELF_PIN_SENTINEL
    source_file_sha256_pin: None = None

    def __post_init__(self) -> None:
        if self.schema_version != STORAGE_BACKEND_CONTRACT_DESCRIPTOR_SCHEMA_VERSION:
            _fail("storage descriptor schema differs")
        if self.status != STORAGE_DESCRIPTOR_STATUS:
            _fail("storage descriptor status differs")
        if self.artifact_schemas != (
            STORAGE_BACKEND_POLICY_SCHEMA_VERSION,
            STORAGE_BOUNDARY_RUNTIME_INTENT_SCHEMA_VERSION,
            STORAGE_BOUNDARY_RECEIPT_V2_SCHEMA_VERSION,
            STORAGE_CLEANUP_RECONCILIATION_SCHEMA_VERSION,
        ):
            _fail("descriptor artifact schemas differ")
        if self.component_roles != STORAGE_COMPONENT_ROLES:
            _fail("descriptor component roles differ")
        if self.component_descriptor_schemas != STORAGE_COMPONENT_DESCRIPTOR_SCHEMA_INVENTORY:
            _fail("descriptor component descriptor schemas differ")
        if self.phase_split != STORAGE_PHASE_SPLIT or self.chronology != STORAGE_CHRONOLOGY:
            _fail("descriptor phase split or chronology differs")
        if self.resource_fields != RESOURCE_FIELDS:
            _fail("descriptor 28-field inventory differs")
        if (
            type(self.storage_field_positions) is not tuple
            or len(self.storage_field_positions) != 2
        ):
            _fail("descriptor storage field positions type differs")
        _require_int(
            self.storage_field_positions[0],
            "descriptor temporary field position",
            minimum=FIELD24_POSITION,
            maximum=FIELD24_POSITION,
        )
        _require_int(
            self.storage_field_positions[1],
            "descriptor disk field position",
            minimum=FIELD25_POSITION,
            maximum=FIELD25_POSITION,
        )
        if self.storage_field_positions != (FIELD24_POSITION, FIELD25_POSITION):
            _fail("descriptor storage field positions differ")
        if self.raw_artifact_schemas != RAW_ARTIFACT_SCHEMA_INVENTORY:
            _fail("descriptor raw artifact schemas differ")
        if self.required_external_artifact_schemas != REQUIRED_EXTERNAL_ARTIFACT_SCHEMAS:
            _fail("descriptor required external artifact schemas differ")
        for mapping, expected, label in (
            (self.capabilities, SOURCE_ONLY_CAPABILITIES, "capabilities"),
            (self.readiness, SOURCE_ONLY_READINESS, "readiness"),
            (self.authority, SOURCE_ONLY_AUTHORITY, "authority"),
            (self.claims, SOURCE_ONLY_CLAIMS, "claims"),
        ):
            if type(mapping) not in {dict, MappingProxyType} or dict(mapping) != dict(expected):
                _fail(f"descriptor {label} differ")
            if any(value is not False for value in mapping.values()):
                _fail(f"descriptor {label} must all be false")
            object.__setattr__(self, label, MappingProxyType(dict(mapping)))
        if self.operational_apis != ():
            _fail("descriptor operational APIs must be empty")
        _require_sha256(self.descriptor_self_pin_sha256, "descriptor self-pin", permit_zero=True)
        if (
            self.descriptor_self_pin_sha256 != DESCRIPTOR_SERIALIZED_SELF_PIN_SENTINEL
            or self.source_file_sha256_pin is not None
        ):
            _fail("serialized descriptor self-pin must remain zero and source self-pin absent")
        _assert_plain_unaliased_json(self.to_body_dict())

    def to_body_dict(self) -> dict[str, Any]:
        return {
            "artifact_schemas": list(self.artifact_schemas),
            "authority": dict(self.authority),
            "capabilities": dict(self.capabilities),
            "chronology": list(self.chronology),
            "claims": dict(self.claims),
            "component_descriptor_schemas": list(self.component_descriptor_schemas),
            "component_roles": list(self.component_roles),
            "descriptor_self_pin_sha256": self.descriptor_self_pin_sha256,
            "operational_apis": list(self.operational_apis),
            "phase_split": list(self.phase_split),
            "raw_artifact_schemas": list(self.raw_artifact_schemas),
            "readiness": dict(self.readiness),
            "required_external_artifact_schemas": list(self.required_external_artifact_schemas),
            "resource_fields": list(self.resource_fields),
            "schema_version": self.schema_version,
            "source_file_sha256_pin": self.source_file_sha256_pin,
            "status": self.status,
            "storage_field_positions": list(self.storage_field_positions),
        }


def _validate_current_storage_backend_contract_descriptor_v2(
    descriptor: object,
) -> StorageBackendContractDescriptorV2:
    if type(descriptor) is not StorageBackendContractDescriptorV2:
        _fail("storage descriptor type differs")
    exact = descriptor
    _translate_current_state_error(
        "storage descriptor",
        lambda: StorageBackendContractDescriptorV2.__post_init__(exact),
    )
    return exact


def _artifact_from_dict(value: object, label: str) -> ArtifactIdentityV1:
    data = _require_exact_keys(
        value,
        frozenset({"body_sha256", "file_sha256", "schema_version"}),
        label,
    )
    return ArtifactIdentityV1(
        schema_version=data["schema_version"],
        file_sha256=data["file_sha256"],
        body_sha256=data["body_sha256"],
    )


def _component_from_dict(value: object, label: str) -> PinnedStorageComponentIdentityV1:
    data = _require_exact_keys(
        value,
        frozenset(
            {
                "descriptor_body_sha256",
                "descriptor_file_sha256",
                "descriptor_schema_version",
                "role",
                "source_sha256",
            }
        ),
        label,
    )
    return PinnedStorageComponentIdentityV1(
        role=data["role"],
        descriptor_schema_version=data["descriptor_schema_version"],
        descriptor_file_sha256=data["descriptor_file_sha256"],
        descriptor_body_sha256=data["descriptor_body_sha256"],
        source_sha256=data["source_sha256"],
    )


def _components_from_list(value: object) -> tuple[PinnedStorageComponentIdentityV1, ...]:
    if type(value) is not list:
        _fail("components must be one JSON array")
    return tuple(
        _component_from_dict(item, f"component {index}")
        for index, item in enumerate(cast(list[object], value))
    )


def _raw_projection_from_dict(value: object, label: str) -> RawArtifactProjectionV1:
    data = _require_exact_keys(
        value,
        frozenset(RawArtifactProjectionV1.__dataclass_fields__),
        label,
    )
    predecessors = data["predecessors"]
    if type(predecessors) is not list:
        _fail(f"{label} predecessors must be one JSON array")
    return RawArtifactProjectionV1(
        kind=data["kind"],
        first_class_artifact=_artifact_from_dict(
            data["first_class_artifact"],
            f"{label} first-class artifact",
        ),
        raw_artifact=_artifact_from_dict(data["raw_artifact"], f"{label} raw artifact"),
        predecessors=tuple(
            _artifact_from_dict(item, f"{label} predecessor {index}")
            for index, item in enumerate(cast(list[object], predecessors))
        ),
        producer=_component_from_dict(data["producer"], f"{label} producer"),
        exact_projection=data["exact_projection"],
        predecessor_identity_bound=data["predecessor_identity_bound"],
    )


def _host_go_storage_intent_binding_from_dict(
    value: object,
) -> HostGoStorageIntentBindingV1:
    label = "host-GO storage-intent binding"
    data = _require_exact_keys(
        value,
        frozenset(HostGoStorageIntentBindingV1.__dataclass_fields__),
        label,
    )
    return HostGoStorageIntentBindingV1(
        host_go=_artifact_from_dict(data["host_go"], f"{label} host GO"),
        runtime_intent=_artifact_from_dict(
            data["runtime_intent"],
            f"{label} runtime intent",
        ),
        verifier=_component_from_dict(data["verifier"], f"{label} verifier"),
        exact_bidirectional_projection=data["exact_bidirectional_projection"],
        intent_committed_before_go=data["intent_committed_before_go"],
    )


def canonical_storage_backend_policy_v1_body_bytes(policy: StorageBackendPolicyV1) -> bytes:
    exact = _validate_current_storage_backend_policy_v1(policy)
    return canonical_storage_backend_json_bytes(exact.to_body_dict(), final_lf=False)


def canonical_storage_backend_policy_v1_file_bytes(policy: StorageBackendPolicyV1) -> bytes:
    exact = _validate_current_storage_backend_policy_v1(policy)
    return canonical_storage_backend_json_bytes(
        _file_dict(exact.to_body_dict(), _POLICY_BODY_FIELD)
    )


def canonical_storage_boundary_runtime_intent_v1_body_bytes(
    intent: StorageBoundaryRuntimeIntentV1,
) -> bytes:
    exact = _validate_current_storage_boundary_runtime_intent_v1(intent)
    return canonical_storage_backend_json_bytes(exact.to_body_dict(), final_lf=False)


def canonical_storage_boundary_runtime_intent_v1_file_bytes(
    intent: StorageBoundaryRuntimeIntentV1,
) -> bytes:
    exact = _validate_current_storage_boundary_runtime_intent_v1(intent)
    return canonical_storage_backend_json_bytes(
        _file_dict(exact.to_body_dict(), _INTENT_BODY_FIELD)
    )


def canonical_storage_boundary_receipt_v2_body_bytes(receipt: StorageBoundaryReceiptV2) -> bytes:
    exact = _validate_current_storage_boundary_receipt_v2(receipt)
    return canonical_storage_backend_json_bytes(exact.to_body_dict(), final_lf=False)


def canonical_storage_boundary_receipt_v2_file_bytes(receipt: StorageBoundaryReceiptV2) -> bytes:
    exact = _validate_current_storage_boundary_receipt_v2(receipt)
    return canonical_storage_backend_json_bytes(
        _file_dict(exact.to_body_dict(), _RECEIPT_BODY_FIELD)
    )


def canonical_storage_cleanup_reconciliation_v1_body_bytes(
    cleanup: StorageCleanupReconciliationV1,
) -> bytes:
    exact = _validate_current_storage_cleanup_reconciliation_v1(cleanup)
    return canonical_storage_backend_json_bytes(exact.to_body_dict(), final_lf=False)


def canonical_storage_cleanup_reconciliation_v1_file_bytes(
    cleanup: StorageCleanupReconciliationV1,
) -> bytes:
    exact = _validate_current_storage_cleanup_reconciliation_v1(cleanup)
    return canonical_storage_backend_json_bytes(
        _file_dict(exact.to_body_dict(), _CLEANUP_BODY_FIELD)
    )


def canonical_storage_backend_contract_descriptor_v2_body_bytes(
    descriptor: StorageBackendContractDescriptorV2,
) -> bytes:
    exact = _validate_current_storage_backend_contract_descriptor_v2(descriptor)
    return canonical_storage_backend_json_bytes(exact.to_body_dict(), final_lf=False)


def canonical_storage_backend_contract_descriptor_v2_file_bytes(
    descriptor: StorageBackendContractDescriptorV2,
) -> bytes:
    exact = _validate_current_storage_backend_contract_descriptor_v2(descriptor)
    return canonical_storage_backend_json_bytes(
        _file_dict(exact.to_body_dict(), _DESCRIPTOR_BODY_FIELD)
    )


def _identity(schema: str, body: bytes, file: bytes) -> ArtifactIdentityV1:
    return ArtifactIdentityV1(
        schema_version=schema, file_sha256=_sha256(file), body_sha256=_sha256(body)
    )


def storage_backend_policy_identity_v1(policy: StorageBackendPolicyV1) -> ArtifactIdentityV1:
    return _identity(
        STORAGE_BACKEND_POLICY_SCHEMA_VERSION,
        canonical_storage_backend_policy_v1_body_bytes(policy),
        canonical_storage_backend_policy_v1_file_bytes(policy),
    )


def storage_boundary_runtime_intent_identity_v1(
    intent: StorageBoundaryRuntimeIntentV1,
) -> ArtifactIdentityV1:
    return _identity(
        STORAGE_BOUNDARY_RUNTIME_INTENT_SCHEMA_VERSION,
        canonical_storage_boundary_runtime_intent_v1_body_bytes(intent),
        canonical_storage_boundary_runtime_intent_v1_file_bytes(intent),
    )


def storage_boundary_receipt_v2_identity(receipt: StorageBoundaryReceiptV2) -> ArtifactIdentityV1:
    return _identity(
        STORAGE_BOUNDARY_RECEIPT_V2_SCHEMA_VERSION,
        canonical_storage_boundary_receipt_v2_body_bytes(receipt),
        canonical_storage_boundary_receipt_v2_file_bytes(receipt),
    )


def storage_cleanup_reconciliation_identity_v1(
    cleanup: StorageCleanupReconciliationV1,
) -> ArtifactIdentityV1:
    return _identity(
        STORAGE_CLEANUP_RECONCILIATION_SCHEMA_VERSION,
        canonical_storage_cleanup_reconciliation_v1_body_bytes(cleanup),
        canonical_storage_cleanup_reconciliation_v1_file_bytes(cleanup),
    )


def _policy_from_body(body: dict[str, Any]) -> StorageBackendPolicyV1:
    _reject_keys(body, _FORBIDDEN_PHASE4_KEYS, "phase-4 contamination")
    expected = frozenset(StorageBackendPolicyV1.__dataclass_fields__)
    data = _require_exact_keys(body, expected, "storage policy")
    return StorageBackendPolicyV1(
        qualification_plan=_artifact_from_dict(data["qualification_plan"], "qualification plan"),
        max_temporary_peak_bytes=data["max_temporary_peak_bytes"],
        components=_components_from_list(data["components"]),
        schema_version=data["schema_version"],
        status=data["status"],
        resource_field_order_sha256=data["resource_field_order_sha256"],
        candidate_order_sha256=data["candidate_order_sha256"],
        temporary_field_position=data["temporary_field_position"],
        disk_field_position=data["disk_field_position"],
        aggregate_root_path=data["aggregate_root_path"],
        aggregate_root_count=data["aggregate_root_count"],
        aggregate_root_case_exclusive=data["aggregate_root_case_exclusive"],
        aggregate_root_application_writable=data["aggregate_root_application_writable"],
        aggregate_root_allocatable=data["aggregate_root_allocatable"],
        tmpfs_hard_size_limit_bytes=data["tmpfs_hard_size_limit_bytes"],
        tmpfs_exact_size_readback_required=data["tmpfs_exact_size_readback_required"],
        tmpfs_noswap_required=data["tmpfs_noswap_required"],
        tmpfs_mount_mutation_closure_required=data["tmpfs_mount_mutation_closure_required"],
        disk_published_value_bytes=data["disk_published_value_bytes"],
        disk_scope=data["disk_scope"],
        persistent_storage_scope=data["persistent_storage_scope"],
        persistent_storage_measurement_interval=data["persistent_storage_measurement_interval"],
        trusted_runtime_bookkeeping_exclusions=_string_tuple_from_json(
            data["trusted_runtime_bookkeeping_exclusions"],
            "trusted runtime bookkeeping exclusions",
        ),
        rootfs_read_only=data["rootfs_read_only"],
        rootfs_copy_up_enabled=data["rootfs_copy_up_enabled"],
        container_log_driver=data["container_log_driver"],
        application_bind_mount_count=data["application_bind_mount_count"],
        volume_count=data["volume_count"],
        added_device_count=data["added_device_count"],
        network_enabled=data["network_enabled"],
        inherited_allocatable_storage_fd_count=data["inherited_allocatable_storage_fd_count"],
        alternate_writable_path_count=data["alternate_writable_path_count"],
        application_writable_tmpfs_count=data["application_writable_tmpfs_count"],
        docker_implicit_config_bind_paths=_string_tuple_from_json(
            data["docker_implicit_config_bind_paths"],
            "Docker implicit configuration bind paths",
        ),
        implicit_readonly_etc_bind_count=data["implicit_readonly_etc_bind_count"],
        writable_persistent_mount_count=data["writable_persistent_mount_count"],
        writable_persistent_fd_count=data["writable_persistent_fd_count"],
        candidate_stdio_transport_count=data["candidate_stdio_transport_count"],
        docker_implicit_mount_inventory_required=data["docker_implicit_mount_inventory_required"],
        docker_implicit_mounts_all_read_only=data["docker_implicit_mounts_all_read_only"],
        docker_implicit_mounts_all_nonallocatable=data["docker_implicit_mounts_all_nonallocatable"],
        docker_implicit_mounts_have_application_writable_path=data[
            "docker_implicit_mounts_have_application_writable_path"
        ],
        docker_ipc_mode=data["docker_ipc_mode"],
        docker_shm_mount_present=data["docker_shm_mount_present"],
        docker_tty_enabled=data["docker_tty_enabled"],
        docker_stdin_open=data["docker_stdin_open"],
        image_declared_volume_count=data["image_declared_volume_count"],
        docker_exec_permitted=data["docker_exec_permitted"],
        docker_archive_api_permitted=data["docker_archive_api_permitted"],
        docker_api_candidate_accessible=data["docker_api_candidate_accessible"],
        container_console_or_fifo_candidate_accessible=data[
            "container_console_or_fifo_candidate_accessible"
        ],
        default_device_inventory_required=data["default_device_inventory_required"],
        default_devices_can_allocate_storage=data["default_devices_can_allocate_storage"],
        device_open_ioctl_confinement_required=data["device_open_ioctl_confinement_required"],
        memfd_posix_or_sysv_shm_permitted=data["memfd_posix_or_sysv_shm_permitted"],
        post_go_mount_mutation_permitted=data["post_go_mount_mutation_permitted"],
        candidate_cgroup_mutation_permitted=data["candidate_cgroup_mutation_permitted"],
        rootfs_upperdir_candidate_writable=data["rootfs_upperdir_candidate_writable"],
        daemon_runtime_storage_candidate_accessible=data[
            "daemon_runtime_storage_candidate_accessible"
        ],
        host_archival_candidate_accessible=data["host_archival_candidate_accessible"],
        outer_cgroup_memory_swap_max_bytes=data["outer_cgroup_memory_swap_max_bytes"],
        outer_cgroup_memory_zswap_writeback_enabled=data[
            "outer_cgroup_memory_zswap_writeback_enabled"
        ],
        architecture=data["architecture"],
        raw_artifact_schema_inventory=_string_tuple_from_json(
            data["raw_artifact_schema_inventory"], "raw artifact schema inventory"
        ),
        measurement_failure_policy=data["measurement_failure_policy"],
        cleanup_policy=data["cleanup_policy"],
        phase_number=data["phase_number"],
        runtime_observations_present=data["runtime_observations_present"],
    )


def parse_storage_backend_policy_v1(
    raw: bytes,
    *,
    expected_file_sha256: str,
    expected_body_sha256: str,
) -> StorageBackendPolicyV1:
    return _policy_from_body(
        _validate_file(
            raw,
            expected_file_sha256,
            expected_body_sha256,
            _POLICY_BODY_FIELD,
            "policy",
        )
    )


def _intent_from_body(body: dict[str, Any]) -> StorageBoundaryRuntimeIntentV1:
    data = _require_exact_keys(
        body,
        frozenset(StorageBoundaryRuntimeIntentV1.__dataclass_fields__),
        "runtime intent",
    )
    return StorageBoundaryRuntimeIntentV1(
        campaign_id=data["campaign_id"],
        case_ordinal=data["case_ordinal"],
        candidate_id=data["candidate_id"],
        candidate_family=data["candidate_family"],
        qualification_case_id=data["qualification_case_id"],
        qualification_plan=_artifact_from_dict(data["qualification_plan"], "qualification plan"),
        policy=_artifact_from_dict(data["policy"], "policy"),
        image_id=data["image_id"],
        runtime_qualification_receipt=_artifact_from_dict(
            data["runtime_qualification_receipt"], "runtime qualification receipt"
        ),
        host_provisioning_v3_validated_pre_go_prefix=_artifact_from_dict(
            data["host_provisioning_v3_validated_pre_go_prefix"], "host provisioning prefix"
        ),
        container_name=data["container_name"],
        container_id_commitment_sha256=data["container_id_commitment_sha256"],
        outer_cgroup_identity_sha256=data["outer_cgroup_identity_sha256"],
        mount_namespace_identity=_artifact_from_dict(
            data["mount_namespace_identity"], "mount namespace"
        ),
        rootfs_mount_identity=_artifact_from_dict(data["rootfs_mount_identity"], "rootfs mount"),
        tmpfs_mount_identity=_artifact_from_dict(data["tmpfs_mount_identity"], "tmpfs mount"),
        tmpfs_backing_identity=_artifact_from_dict(data["tmpfs_backing_identity"], "tmpfs backing"),
        mount_inventory=_artifact_from_dict(data["mount_inventory"], "mount inventory"),
        path_inventory=_artifact_from_dict(data["path_inventory"], "path inventory"),
        storage_root_inventory=_artifact_from_dict(
            data["storage_root_inventory"], "storage root inventory"
        ),
        field_inventory=_artifact_from_dict(data["field_inventory"], "field inventory"),
        raw_schema_inventory=_artifact_from_dict(
            data["raw_schema_inventory"], "raw schema inventory"
        ),
        outer_cgroup_memory_swap_max_pre_go=_artifact_from_dict(
            data["outer_cgroup_memory_swap_max_pre_go"],
            "outer-cgroup memory.swap.max pre-GO",
        ),
        outer_cgroup_swap_counters_initial=_artifact_from_dict(
            data["outer_cgroup_swap_counters_initial"],
            "outer-cgroup initial swap counters",
        ),
        outer_cgroup_memory_zswap_writeback_pre_go=_artifact_from_dict(
            data["outer_cgroup_memory_zswap_writeback_pre_go"],
            "outer-cgroup memory.zswap.writeback pre-GO",
        ),
        docker_implicit_mount_inventory=_artifact_from_dict(
            data["docker_implicit_mount_inventory"], "Docker implicit mount inventory"
        ),
        docker_create_inspect=_artifact_from_dict(
            data["docker_create_inspect"],
            "Docker create/inspect storage projection",
        ),
        final_oci_spec=_artifact_from_dict(data["final_oci_spec"], "final OCI spec"),
        console_stdio_inventory=_artifact_from_dict(
            data["console_stdio_inventory"],
            "console/stdin/stdout inventory",
        ),
        rootfs_upperdir_pre_go_baseline=_artifact_from_dict(
            data["rootfs_upperdir_pre_go_baseline"],
            "rootfs upperdir pre-GO baseline",
        ),
        docker_volume_inventory_pre_go_baseline=_artifact_from_dict(
            data["docker_volume_inventory_pre_go_baseline"],
            "Docker volume inventory pre-GO baseline",
        ),
        max_temporary_peak_bytes=data["max_temporary_peak_bytes"],
        components=_components_from_list(data["components"]),
        schema_version=data["schema_version"],
        status=data["status"],
        resource_field_order_sha256=data["resource_field_order_sha256"],
        candidate_order_sha256=data["candidate_order_sha256"],
        aggregate_root_path=data["aggregate_root_path"],
        aggregate_root_count=data["aggregate_root_count"],
        aggregate_root_case_exclusive=data["aggregate_root_case_exclusive"],
        aggregate_root_application_writable=data["aggregate_root_application_writable"],
        aggregate_root_allocatable=data["aggregate_root_allocatable"],
        tmpfs_hard_size_limit_bytes=data["tmpfs_hard_size_limit_bytes"],
        tmpfs_exact_size_readback_matches_ceiling_pre_go=data[
            "tmpfs_exact_size_readback_matches_ceiling_pre_go"
        ],
        tmpfs_noswap_active_pre_go=data["tmpfs_noswap_active_pre_go"],
        mount_mutation_disabled_before_go=data["mount_mutation_disabled_before_go"],
        disk_published_value_bytes=data["disk_published_value_bytes"],
        disk_scope=data["disk_scope"],
        persistent_storage_scope=data["persistent_storage_scope"],
        persistent_storage_measurement_interval=data["persistent_storage_measurement_interval"],
        trusted_runtime_bookkeeping_exclusions=_string_tuple_from_json(
            data["trusted_runtime_bookkeeping_exclusions"],
            "runtime trusted bookkeeping exclusions",
        ),
        rootfs_read_only=data["rootfs_read_only"],
        rootfs_copy_up_enabled=data["rootfs_copy_up_enabled"],
        container_log_driver=data["container_log_driver"],
        application_bind_mount_count=data["application_bind_mount_count"],
        volume_count=data["volume_count"],
        added_device_count=data["added_device_count"],
        network_enabled=data["network_enabled"],
        inherited_allocatable_storage_fd_count=data["inherited_allocatable_storage_fd_count"],
        alternate_writable_path_count=data["alternate_writable_path_count"],
        application_writable_tmpfs_count=data["application_writable_tmpfs_count"],
        docker_implicit_config_bind_paths=_string_tuple_from_json(
            data["docker_implicit_config_bind_paths"],
            "runtime Docker implicit configuration bind paths",
        ),
        implicit_readonly_etc_bind_count=data["implicit_readonly_etc_bind_count"],
        docker_implicit_mount_inventory_required=data["docker_implicit_mount_inventory_required"],
        writable_persistent_mount_count=data["writable_persistent_mount_count"],
        writable_persistent_fd_count=data["writable_persistent_fd_count"],
        candidate_stdio_transport_count=data["candidate_stdio_transport_count"],
        docker_implicit_mounts_all_read_only=data["docker_implicit_mounts_all_read_only"],
        docker_implicit_mounts_all_nonallocatable=data["docker_implicit_mounts_all_nonallocatable"],
        docker_implicit_mounts_have_application_writable_path=data[
            "docker_implicit_mounts_have_application_writable_path"
        ],
        docker_ipc_mode=data["docker_ipc_mode"],
        docker_shm_mount_present=data["docker_shm_mount_present"],
        docker_tty_enabled=data["docker_tty_enabled"],
        docker_stdin_open=data["docker_stdin_open"],
        image_declared_volume_count=data["image_declared_volume_count"],
        docker_exec_permitted=data["docker_exec_permitted"],
        docker_archive_api_permitted=data["docker_archive_api_permitted"],
        docker_api_candidate_accessible=data["docker_api_candidate_accessible"],
        container_console_or_fifo_candidate_accessible=data[
            "container_console_or_fifo_candidate_accessible"
        ],
        default_device_inventory_exact=data["default_device_inventory_exact"],
        default_devices_can_allocate_storage=data["default_devices_can_allocate_storage"],
        device_open_ioctl_confinement_active=data["device_open_ioctl_confinement_active"],
        memfd_posix_or_sysv_shm_permitted=data["memfd_posix_or_sysv_shm_permitted"],
        post_go_mount_mutation_permitted=data["post_go_mount_mutation_permitted"],
        candidate_cgroup_mutation_permitted=data["candidate_cgroup_mutation_permitted"],
        rootfs_upperdir_candidate_writable=data["rootfs_upperdir_candidate_writable"],
        daemon_runtime_storage_candidate_accessible=data[
            "daemon_runtime_storage_candidate_accessible"
        ],
        host_archival_candidate_accessible=data["host_archival_candidate_accessible"],
        outer_cgroup_memory_swap_max_bytes_pre_go=data["outer_cgroup_memory_swap_max_bytes_pre_go"],
        outer_cgroup_memory_zswap_writeback_enabled_pre_go=data[
            "outer_cgroup_memory_zswap_writeback_enabled_pre_go"
        ],
        phase_number=data["phase_number"],
        committed_before_go=data["committed_before_go"],
        future_receipt_bound=data["future_receipt_bound"],
        future_seal_bound=data["future_seal_bound"],
        future_terminal_bound=data["future_terminal_bound"],
    )


def parse_storage_boundary_runtime_intent_v1(
    raw: bytes,
    *,
    expected_file_sha256: str,
    expected_body_sha256: str,
) -> StorageBoundaryRuntimeIntentV1:
    return _intent_from_body(
        _validate_file(
            raw,
            expected_file_sha256,
            expected_body_sha256,
            _INTENT_BODY_FIELD,
            "intent",
        )
    )


def _handshake_from_dict(value: object) -> HostV3HandshakeProjectionV1:
    data = _require_exact_keys(
        value,
        frozenset(HostV3HandshakeProjectionV1.__dataclass_fields__),
        "host handshake",
    )
    return HostV3HandshakeProjectionV1(
        request=_artifact_from_dict(data["request"], "host request"),
        intent=_artifact_from_dict(data["intent"], "host intent"),
        ready=_artifact_from_dict(data["ready"], "host READY"),
        observer_anchor=_artifact_from_dict(data["observer_anchor"], "host anchor"),
        go=_artifact_from_dict(data["go"], "host GO"),
        campaign_id=data["campaign_id"],
        case_ordinal=data["case_ordinal"],
        candidate_id=data["candidate_id"],
        qualification_case_id=data["qualification_case_id"],
        image_id=data["image_id"],
        container_name=data["container_name"],
        container_id_commitment_sha256=data["container_id_commitment_sha256"],
        exact_projection_validated=data["exact_projection_validated"],
        order=_string_tuple_from_json(data["order"], "host handshake order"),
    )


def _tmpfs_from_dict(value: object) -> TmpfsConservativeBoundEvidenceV1:
    data = _require_exact_keys(
        value,
        frozenset(TmpfsConservativeBoundEvidenceV1.__dataclass_fields__),
        "tmpfs conservative bound",
    )
    return TmpfsConservativeBoundEvidenceV1(
        aggregate_root_path=data["aggregate_root_path"],
        hard_limit_bytes=data["hard_limit_bytes"],
        initial_observed_used_bytes=data["initial_observed_used_bytes"],
        terminal_observed_used_bytes=data["terminal_observed_used_bytes"],
        measurement_interval_start=_artifact_from_dict(
            data["measurement_interval_start"],
            "tmpfs measurement interval start",
        ),
        measurement_interval_end=_artifact_from_dict(
            data["measurement_interval_end"],
            "tmpfs measurement interval end",
        ),
        raw_mountinfo=_artifact_from_dict(data["raw_mountinfo"], "tmpfs mountinfo"),
        raw_statfs_diagnostic_samples=_artifact_from_dict(
            data["raw_statfs_diagnostic_samples"],
            "tmpfs statfs diagnostic samples",
        ),
        raw_hard_limit_mount_mutation_closure=_artifact_from_dict(
            data["raw_hard_limit_mount_mutation_closure"],
            "tmpfs hard-limit and mount-mutation closure",
        ),
        measurement_producer=_component_from_dict(
            data["measurement_producer"],
            "tmpfs measurement producer",
        ),
        field_position=data["field_position"],
        accounting_mode=data["accounting_mode"],
        publication_semantics=data["publication_semantics"],
        aggregate_root_count=data["aggregate_root_count"],
        aggregate_root_case_exclusive=data["aggregate_root_case_exclusive"],
        exact_size_readback_equals_ceiling_pre_go=data["exact_size_readback_equals_ceiling_pre_go"],
        hard_limit_unchanged_through_write_seal=data["hard_limit_unchanged_through_write_seal"],
        noswap_active_pre_go=data["noswap_active_pre_go"],
        noswap_unchanged_through_write_seal=data["noswap_unchanged_through_write_seal"],
        mount_mutation_disabled_before_go=data["mount_mutation_disabled_before_go"],
        mount_mutation_disabled_through_write_seal=data[
            "mount_mutation_disabled_through_write_seal"
        ],
        aggregate_root_non_bypassable=data["aggregate_root_non_bypassable"],
        statfs_samples_are_diagnostic_only=data["statfs_samples_are_diagnostic_only"],
    )


def _disk_from_dict(value: object) -> DiskStructuralAbsenceEvidenceV1:
    data = _require_exact_keys(
        value,
        frozenset(DiskStructuralAbsenceEvidenceV1.__dataclass_fields__),
        "disk absence",
    )
    return DiskStructuralAbsenceEvidenceV1(
        measurement_interval_start=_artifact_from_dict(
            data["measurement_interval_start"],
            "disk measurement interval start",
        ),
        measurement_interval_end=_artifact_from_dict(
            data["measurement_interval_end"],
            "disk measurement interval end",
        ),
        raw_mountinfo=_artifact_from_dict(data["raw_mountinfo"], "disk mountinfo"),
        raw_absence_scan=_artifact_from_dict(data["raw_absence_scan"], "disk scan"),
        raw_writable_fd_inventory=_artifact_from_dict(
            data["raw_writable_fd_inventory"], "writable FD inventory"
        ),
        measurement_producer=_component_from_dict(
            data["measurement_producer"],
            "disk measurement producer",
        ),
        field_position=data["field_position"],
        published_value_bytes=data["published_value_bytes"],
        absence_kind=data["absence_kind"],
        persistent_storage_scope=data["persistent_storage_scope"],
        measurement_interval=data["measurement_interval"],
        trusted_runtime_bookkeeping_exclusions=_string_tuple_from_json(
            data["trusted_runtime_bookkeeping_exclusions"],
            "disk trusted runtime bookkeeping exclusions",
        ),
        measurement_mode=data["measurement_mode"],
        rootfs_read_only=data["rootfs_read_only"],
        copy_up_disabled=data["copy_up_disabled"],
        no_bind_volume_device_network_log_paths=data["no_bind_volume_device_network_log_paths"],
        no_inherited_or_alternate_writable_paths=data["no_inherited_or_alternate_writable_paths"],
        writable_persistent_mount_count=data["writable_persistent_mount_count"],
        writable_persistent_fd_count=data["writable_persistent_fd_count"],
        transient_persistent_open_structurally_impossible=data[
            "transient_persistent_open_structurally_impossible"
        ],
        structural_nonaddressability_complete=data["structural_nonaddressability_complete"],
    )


def _swap_and_implicit_mount_from_dict(
    value: object,
) -> SwapAndImplicitMountClosureEvidenceV1:
    data = _require_exact_keys(
        value,
        frozenset(SwapAndImplicitMountClosureEvidenceV1.__dataclass_fields__),
        "swap and implicit-mount closure",
    )
    return SwapAndImplicitMountClosureEvidenceV1(
        outer_cgroup_identity_sha256=data["outer_cgroup_identity_sha256"],
        measurement_interval_start=_artifact_from_dict(
            data["measurement_interval_start"],
            "swap/implicit-mount measurement interval start",
        ),
        measurement_interval_end=_artifact_from_dict(
            data["measurement_interval_end"],
            "swap/implicit-mount measurement interval end",
        ),
        docker_volume_inventory_pre_go_baseline=_artifact_from_dict(
            data["docker_volume_inventory_pre_go_baseline"],
            "Docker volume inventory pre-GO baseline",
        ),
        raw_memory_swap_max_pre_go=_artifact_from_dict(
            data["raw_memory_swap_max_pre_go"], "raw memory.swap.max pre-GO"
        ),
        raw_swap_counters_initial=_artifact_from_dict(
            data["raw_swap_counters_initial"], "raw initial swap counters"
        ),
        raw_swap_counters_terminal=_artifact_from_dict(
            data["raw_swap_counters_terminal"], "raw terminal swap counters"
        ),
        raw_memory_zswap_writeback_pre_go=_artifact_from_dict(
            data["raw_memory_zswap_writeback_pre_go"],
            "raw memory.zswap.writeback pre-GO",
        ),
        raw_docker_implicit_mount_inventory=_artifact_from_dict(
            data["raw_docker_implicit_mount_inventory"],
            "raw Docker implicit mount inventory",
        ),
        raw_docker_create_inspect=_artifact_from_dict(
            data["raw_docker_create_inspect"],
            "raw Docker create/inspect projection",
        ),
        raw_final_oci_spec=_artifact_from_dict(
            data["raw_final_oci_spec"],
            "raw final OCI spec",
        ),
        raw_console_stdio_inventory=_artifact_from_dict(
            data["raw_console_stdio_inventory"],
            "raw console/stdin/stdout inventory",
        ),
        raw_docker_api_operation_journal=_artifact_from_dict(
            data["raw_docker_api_operation_journal"],
            "raw Docker API operation journal",
        ),
        raw_rootfs_upperdir_pre_go_baseline=_artifact_from_dict(
            data["raw_rootfs_upperdir_pre_go_baseline"],
            "raw rootfs upperdir pre-GO baseline",
        ),
        raw_rootfs_upperdir_interval_delta=_artifact_from_dict(
            data["raw_rootfs_upperdir_interval_delta"],
            "raw rootfs upperdir interval delta",
        ),
        raw_docker_volume_inventory_delta=_artifact_from_dict(
            data["raw_docker_volume_inventory_delta"],
            "raw Docker volume inventory delta",
        ),
        measurement_producer=_component_from_dict(
            data["measurement_producer"],
            "swap/implicit-mount measurement producer",
        ),
        runtime_storage_escape_gate=_component_from_dict(
            data["runtime_storage_escape_gate"],
            "runtime storage escape gate",
        ),
        initial_observation_monotonic_ns=data["initial_observation_monotonic_ns"],
        go_commit_monotonic_ns=data["go_commit_monotonic_ns"],
        worker_exit_monotonic_ns=data["worker_exit_monotonic_ns"],
        publication_wrapper_monotonic_ns=data["publication_wrapper_monotonic_ns"],
        reload_validation_monotonic_ns=data["reload_validation_monotonic_ns"],
        relay_preseal_monotonic_ns=data["relay_preseal_monotonic_ns"],
        channel_preseal_monotonic_ns=data["channel_preseal_monotonic_ns"],
        write_seal_monotonic_ns=data["write_seal_monotonic_ns"],
        terminal_observation_monotonic_ns=data["terminal_observation_monotonic_ns"],
        receipt_precommit_monotonic_ns=data["receipt_precommit_monotonic_ns"],
        memory_swap_max_bytes_pre_go=data["memory_swap_max_bytes_pre_go"],
        memory_zswap_writeback_enabled_pre_go=data["memory_zswap_writeback_enabled_pre_go"],
        memory_swap_current_initial_bytes=data["memory_swap_current_initial_bytes"],
        memory_swap_current_terminal_bytes=data["memory_swap_current_terminal_bytes"],
        memory_swap_peak_initial_bytes=data["memory_swap_peak_initial_bytes"],
        memory_swap_peak_terminal_bytes=data["memory_swap_peak_terminal_bytes"],
        memory_zswap_current_initial_bytes=data["memory_zswap_current_initial_bytes"],
        memory_zswap_current_terminal_bytes=data["memory_zswap_current_terminal_bytes"],
        retained_counter_endpoints_same_outer_cgroup=data[
            "retained_counter_endpoints_same_outer_cgroup"
        ],
        counters_retained_from_pre_go_through_terminal=data[
            "counters_retained_from_pre_go_through_terminal"
        ],
        application_writable_tmpfs_count=data["application_writable_tmpfs_count"],
        application_writable_tmpfs_is_aggregate_root=data[
            "application_writable_tmpfs_is_aggregate_root"
        ],
        application_writable_tmpfs_allocatable=data["application_writable_tmpfs_allocatable"],
        docker_implicit_mount_inventory_complete=data["docker_implicit_mount_inventory_complete"],
        docker_implicit_mounts_all_read_only=data["docker_implicit_mounts_all_read_only"],
        docker_implicit_mounts_all_nonallocatable=data["docker_implicit_mounts_all_nonallocatable"],
        docker_implicit_mounts_have_application_writable_path=data[
            "docker_implicit_mounts_have_application_writable_path"
        ],
        persistent_storage_scope=data["persistent_storage_scope"],
        persistent_storage_measurement_interval=data["persistent_storage_measurement_interval"],
        trusted_runtime_bookkeeping_exclusions=_string_tuple_from_json(
            data["trusted_runtime_bookkeeping_exclusions"],
            "closure trusted runtime bookkeeping exclusions",
        ),
        docker_implicit_config_bind_paths=_string_tuple_from_json(
            data["docker_implicit_config_bind_paths"],
            "closure Docker implicit configuration bind paths",
        ),
        implicit_readonly_etc_bind_count=data["implicit_readonly_etc_bind_count"],
        user_bind_mount_count=data["user_bind_mount_count"],
        user_volume_mount_count=data["user_volume_mount_count"],
        image_declared_volume_count=data["image_declared_volume_count"],
        added_device_count=data["added_device_count"],
        writable_persistent_mount_count=data["writable_persistent_mount_count"],
        writable_persistent_fd_count=data["writable_persistent_fd_count"],
        candidate_stdio_transport_count=data["candidate_stdio_transport_count"],
        forbidden_docker_api_operation_count=data["forbidden_docker_api_operation_count"],
        rootfs_upperdir_interval_delta_bytes=data["rootfs_upperdir_interval_delta_bytes"],
        docker_volume_inventory_delta=data["docker_volume_inventory_delta"],
        docker_ipc_mode=data["docker_ipc_mode"],
        docker_shm_mount_present=data["docker_shm_mount_present"],
        docker_tty_enabled=data["docker_tty_enabled"],
        docker_stdin_open=data["docker_stdin_open"],
        docker_exec_permitted=data["docker_exec_permitted"],
        docker_archive_api_permitted=data["docker_archive_api_permitted"],
        docker_api_candidate_accessible=data["docker_api_candidate_accessible"],
        container_console_or_fifo_candidate_accessible=data[
            "container_console_or_fifo_candidate_accessible"
        ],
        default_device_inventory_exact=data["default_device_inventory_exact"],
        default_devices_can_allocate_storage=data["default_devices_can_allocate_storage"],
        memfd_posix_or_sysv_shm_permitted=data["memfd_posix_or_sysv_shm_permitted"],
        post_go_mount_mutation_permitted=data["post_go_mount_mutation_permitted"],
        candidate_cgroup_mutation_permitted=data["candidate_cgroup_mutation_permitted"],
        daemon_runtime_storage_candidate_accessible=data[
            "daemon_runtime_storage_candidate_accessible"
        ],
        host_archival_candidate_accessible=data["host_archival_candidate_accessible"],
        final_oci_spec_exact=data["final_oci_spec_exact"],
        custom_runtime_neutralized_stock_writable_implicit_mounts=data[
            "custom_runtime_neutralized_stock_writable_implicit_mounts"
        ],
        structural_nonaddressability_complete=data["structural_nonaddressability_complete"],
        docker_api_allowlist_complete_and_lossless=data[
            "docker_api_allowlist_complete_and_lossless"
        ],
        device_and_ipc_confinement_active=data["device_and_ipc_confinement_active"],
        rootfs_upperdir_candidate_inaccessible=data["rootfs_upperdir_candidate_inaccessible"],
    )


def _receipt_from_body(body: dict[str, Any]) -> StorageBoundaryReceiptV2:
    data = _require_exact_keys(
        body,
        frozenset(StorageBoundaryReceiptV2.__dataclass_fields__),
        "storage receipt",
    )
    raw = data["raw_artifacts"]
    if type(raw) is not list:
        _fail("raw artifacts must be one JSON array")
    return StorageBoundaryReceiptV2(
        campaign_id=data["campaign_id"],
        case_ordinal=data["case_ordinal"],
        candidate_id=data["candidate_id"],
        candidate_family=data["candidate_family"],
        qualification_case_id=data["qualification_case_id"],
        image_id=data["image_id"],
        handshake=_handshake_from_dict(data["handshake"]),
        policy=_artifact_from_dict(data["policy"], "policy"),
        runtime_intent=_artifact_from_dict(data["runtime_intent"], "runtime intent"),
        host_go_storage_intent_binding=_host_go_storage_intent_binding_from_dict(
            data["host_go_storage_intent_binding"]
        ),
        publication_wrapper=_artifact_from_dict(data["publication_wrapper"], "wrapper"),
        publication_reload_validation=_artifact_from_dict(
            data["publication_reload_validation"], "reload"
        ),
        terminal_relay_preseal_attestation=_artifact_from_dict(
            data["terminal_relay_preseal_attestation"], "relay preseal"
        ),
        nonstorage_channel_preseal_attestation=_artifact_from_dict(
            data["nonstorage_channel_preseal_attestation"], "channel preseal"
        ),
        irreversible_write_seal=_artifact_from_dict(data["irreversible_write_seal"], "write seal"),
        tmpfs_conservative_bound_evidence=_tmpfs_from_dict(
            data["tmpfs_conservative_bound_evidence"]
        ),
        disk_absence_evidence=_disk_from_dict(data["disk_absence_evidence"]),
        swap_and_implicit_mount_closure_evidence=_swap_and_implicit_mount_from_dict(
            data["swap_and_implicit_mount_closure_evidence"]
        ),
        raw_publication_reload=_artifact_from_dict(
            data["raw_publication_reload"],
            "raw publication reload",
        ),
        raw_write_seal=_artifact_from_dict(data["raw_write_seal"], "raw write seal"),
        publication_reload_projection=_raw_projection_from_dict(
            data["publication_reload_projection"],
            "publication reload projection",
        ),
        write_seal_projection=_raw_projection_from_dict(
            data["write_seal_projection"],
            "write seal projection",
        ),
        terminal_relay_preseal_producer=_component_from_dict(
            data["terminal_relay_preseal_producer"],
            "terminal relay preseal producer",
        ),
        nonstorage_channel_preseal_producer=_component_from_dict(
            data["nonstorage_channel_preseal_producer"],
            "nonstorage channel preseal producer",
        ),
        raw_artifacts=tuple(
            _artifact_from_dict(item, f"raw artifact {index}")
            for index, item in enumerate(cast(list[object], raw))
        ),
        max_temporary_peak_bytes=data["max_temporary_peak_bytes"],
        max_disk_peak_bytes=data["max_disk_peak_bytes"],
        components=_components_from_list(data["components"]),
        schema_version=data["schema_version"],
        status=data["status"],
        publication_wrapper_order=data["publication_wrapper_order"],
        reload_validation_order=data["reload_validation_order"],
        relay_preseal_order=data["relay_preseal_order"],
        channel_preseal_order=data["channel_preseal_order"],
        write_seal_order=data["write_seal_order"],
        receipt_commit_order=data["receipt_commit_order"],
        reload_read_only=data["reload_read_only"],
        publication_projection_exact=data["publication_projection_exact"],
        write_seal_irreversible=data["write_seal_irreversible"],
        later_measured_writes_possible=data["later_measured_writes_possible"],
        terminal_bound=data["terminal_bound"],
        lifecycle_bound=data["lifecycle_bound"],
        merger_bound=data["merger_bound"],
    )


def parse_storage_boundary_receipt_v2(
    raw: bytes,
    *,
    expected_file_sha256: str,
    expected_body_sha256: str,
) -> StorageBoundaryReceiptV2:
    return _receipt_from_body(
        _validate_file(
            raw,
            expected_file_sha256,
            expected_body_sha256,
            _RECEIPT_BODY_FIELD,
            "receipt",
        )
    )


def _cleanup_from_body(body: dict[str, Any]) -> StorageCleanupReconciliationV1:
    data = _require_exact_keys(
        body,
        frozenset(StorageCleanupReconciliationV1.__dataclass_fields__),
        "storage cleanup",
    )
    return StorageCleanupReconciliationV1(
        campaign_id=data["campaign_id"],
        case_ordinal=data["case_ordinal"],
        candidate_id=data["candidate_id"],
        candidate_family=data["candidate_family"],
        qualification_case_id=data["qualification_case_id"],
        runtime_intent=_artifact_from_dict(data["runtime_intent"], "runtime intent"),
        outcome=data["outcome"],
        receipt=None
        if data["receipt"] is None
        else _artifact_from_dict(data["receipt"], "receipt"),
        attempted_receipt=(
            None
            if data["attempted_receipt"] is None
            else _artifact_from_dict(data["attempted_receipt"], "attempted receipt")
        ),
        failure_frontier=(
            None
            if data["failure_frontier"] is None
            else _artifact_from_dict(data["failure_frontier"], "failure frontier")
        ),
        namespace_cleanup_receipt=(
            None
            if data["namespace_cleanup_receipt"] is None
            else _artifact_from_dict(
                data["namespace_cleanup_receipt"],
                "namespace cleanup receipt",
            )
        ),
        cleanup_failure_frontier=(
            None
            if data["cleanup_failure_frontier"] is None
            else _artifact_from_dict(
                data["cleanup_failure_frontier"],
                "cleanup failure frontier",
            )
        ),
        cleanup_producer=_component_from_dict(
            data["cleanup_producer"],
            "namespace cleanup producer",
        ),
        cleanup_complete=data["cleanup_complete"],
        residual_storage_state=data["residual_storage_state"],
        tmpfs_unmounted_before_namespace_release=data["tmpfs_unmounted_before_namespace_release"],
        aggregate_mount_id_absent_before_namespace_release=data[
            "aggregate_mount_id_absent_before_namespace_release"
        ],
        underlying_aggregate_path_read_only=data["underlying_aggregate_path_read_only"],
        namespace_process_count=data["namespace_process_count"],
        retained_namespace_fd_count_after_release=data["retained_namespace_fd_count_after_release"],
        schema_version=data["schema_version"],
        status=data["status"],
        deletion_only=data["deletion_only"],
        retained_writable_path_count=data["retained_writable_path_count"],
        receipt_committed=data["receipt_committed"],
        receipt_commit_uncertain=data["receipt_commit_uncertain"],
        consumed=data["consumed"],
        retry_allowed=data["retry_allowed"],
        synthesized_temporary_peak_bytes=data["synthesized_temporary_peak_bytes"],
        synthesized_disk_peak_bytes=data["synthesized_disk_peak_bytes"],
    )


def parse_storage_cleanup_reconciliation_v1(
    raw: bytes,
    *,
    expected_file_sha256: str,
    expected_body_sha256: str,
) -> StorageCleanupReconciliationV1:
    return _cleanup_from_body(
        _validate_file(
            raw,
            expected_file_sha256,
            expected_body_sha256,
            _CLEANUP_BODY_FIELD,
            "cleanup",
        )
    )


def _descriptor_from_body(body: dict[str, Any]) -> StorageBackendContractDescriptorV2:
    data = _require_exact_keys(
        body,
        frozenset(StorageBackendContractDescriptorV2.__dataclass_fields__),
        "storage descriptor",
    )
    return StorageBackendContractDescriptorV2(
        schema_version=data["schema_version"],
        status=data["status"],
        artifact_schemas=_string_tuple_from_json(data["artifact_schemas"], "artifact schemas"),
        component_descriptor_schemas=_string_tuple_from_json(
            data["component_descriptor_schemas"],
            "component descriptor schemas",
        ),
        component_roles=_string_tuple_from_json(data["component_roles"], "component roles"),
        phase_split=_string_tuple_from_json(data["phase_split"], "phase split"),
        chronology=_string_tuple_from_json(data["chronology"], "chronology"),
        resource_fields=_string_tuple_from_json(data["resource_fields"], "resource fields"),
        storage_field_positions=cast(
            tuple[int, int],
            _int_tuple_from_json(data["storage_field_positions"], "storage field positions"),
        ),
        raw_artifact_schemas=_string_tuple_from_json(
            data["raw_artifact_schemas"], "raw artifact schemas"
        ),
        required_external_artifact_schemas=_string_tuple_from_json(
            data["required_external_artifact_schemas"],
            "required external artifact schemas",
        ),
        capabilities=data["capabilities"],
        readiness=data["readiness"],
        authority=data["authority"],
        claims=data["claims"],
        operational_apis=cast(
            tuple[()],
            _string_tuple_from_json(data["operational_apis"], "operational APIs"),
        ),
        descriptor_self_pin_sha256=data["descriptor_self_pin_sha256"],
        source_file_sha256_pin=data["source_file_sha256_pin"],
    )


def _guard_storage_backend_v2_descriptor_pin() -> tuple[str, str]:
    descriptor = StorageBackendContractDescriptorV2()
    body = canonical_storage_backend_contract_descriptor_v2_body_bytes(descriptor)
    file = canonical_storage_backend_contract_descriptor_v2_file_bytes(descriptor)
    body_pin = _require_sha256(
        PINNED_STORAGE_BACKEND_V2_DESCRIPTOR_BODY_SHA256,
        "storage descriptor repository BODY pin",
    )
    file_pin = _require_sha256(
        PINNED_STORAGE_BACKEND_V2_DESCRIPTOR_FILE_SHA256,
        "storage descriptor repository FILE pin",
    )
    if not hmac.compare_digest(body_pin, _sha256(body)):
        _fail("storage descriptor BODY drifted from its repository literal")
    if not hmac.compare_digest(file_pin, _sha256(file)):
        _fail("storage descriptor FILE drifted from its repository literal")
    return body_pin, file_pin


def parse_storage_backend_contract_descriptor_v2(
    raw: bytes,
    *,
    expected_file_sha256: str,
    expected_body_sha256: str,
) -> StorageBackendContractDescriptorV2:
    pinned_body, pinned_file = _guard_storage_backend_v2_descriptor_pin()
    caller_file = _require_sha256(expected_file_sha256, "descriptor caller FILE pin")
    caller_body = _require_sha256(expected_body_sha256, "descriptor caller BODY pin")
    if not hmac.compare_digest(caller_file, pinned_file) or not hmac.compare_digest(
        caller_body,
        pinned_body,
    ):
        _fail("descriptor caller identity differs from the repository literals")
    body = _validate_file(
        raw,
        pinned_file,
        pinned_body,
        _DESCRIPTOR_BODY_FIELD,
        "descriptor",
    )
    descriptor = _descriptor_from_body(body)
    expected = StorageBackendContractDescriptorV2()
    if descriptor != expected:
        _fail("storage descriptor content differs from the frozen descriptor")
    if raw != canonical_storage_backend_contract_descriptor_v2_file_bytes(expected):
        _fail("storage descriptor canonical replay differs")
    return expected


def storage_backend_v2_descriptor_sha256() -> str:
    """Return the finalized descriptor FILE identity after both pin guards pass."""

    return _guard_storage_backend_v2_descriptor_pin()[1]


def validate_storage_policy_runtime_intent_v1(
    policy: StorageBackendPolicyV1,
    intent: StorageBoundaryRuntimeIntentV1,
) -> None:
    policy = _validate_current_storage_backend_policy_v1(policy)
    intent = _validate_current_storage_boundary_runtime_intent_v1(intent)
    if intent.qualification_plan != policy.qualification_plan:
        _fail("runtime intent qualification plan crosswires phase 4")
    if intent.policy != storage_backend_policy_identity_v1(policy):
        _fail("runtime intent storage policy identity differs")
    if intent.max_temporary_peak_bytes != policy.max_temporary_peak_bytes:
        _fail("runtime field-24 hard ceiling differs from policy")
    if intent.components != policy.components:
        _fail("runtime storage components differ from policy")
    for name in (
        "aggregate_root_case_exclusive",
        "aggregate_root_application_writable",
        "aggregate_root_allocatable",
        "application_writable_tmpfs_count",
        "docker_implicit_mounts_all_read_only",
        "docker_implicit_mounts_all_nonallocatable",
        "docker_implicit_mounts_have_application_writable_path",
        "persistent_storage_scope",
        "persistent_storage_measurement_interval",
        "trusted_runtime_bookkeeping_exclusions",
        "rootfs_read_only",
        "rootfs_copy_up_enabled",
        "container_log_driver",
        "application_bind_mount_count",
        "volume_count",
        "added_device_count",
        "network_enabled",
        "inherited_allocatable_storage_fd_count",
        "alternate_writable_path_count",
        "docker_implicit_config_bind_paths",
        "implicit_readonly_etc_bind_count",
        "docker_implicit_mount_inventory_required",
        "writable_persistent_mount_count",
        "writable_persistent_fd_count",
        "candidate_stdio_transport_count",
        "docker_ipc_mode",
        "docker_shm_mount_present",
        "docker_tty_enabled",
        "docker_stdin_open",
        "image_declared_volume_count",
        "docker_exec_permitted",
        "docker_archive_api_permitted",
        "docker_api_candidate_accessible",
        "container_console_or_fifo_candidate_accessible",
        "default_devices_can_allocate_storage",
        "memfd_posix_or_sysv_shm_permitted",
        "post_go_mount_mutation_permitted",
        "candidate_cgroup_mutation_permitted",
        "rootfs_upperdir_candidate_writable",
        "daemon_runtime_storage_candidate_accessible",
        "host_archival_candidate_accessible",
    ):
        if getattr(intent, name) != getattr(policy, name):
            _fail(f"runtime {name} differs from policy")
    if (
        intent.tmpfs_exact_size_readback_matches_ceiling_pre_go
        != policy.tmpfs_exact_size_readback_required
        or intent.tmpfs_noswap_active_pre_go != policy.tmpfs_noswap_required
        or intent.mount_mutation_disabled_before_go != policy.tmpfs_mount_mutation_closure_required
    ):
        _fail("runtime tmpfs hard-limit, noswap, or mount-mutation proof differs from policy")
    if (
        intent.default_device_inventory_exact != policy.default_device_inventory_required
        or intent.device_open_ioctl_confinement_active
        != policy.device_open_ioctl_confinement_required
    ):
        _fail("runtime default-device or ioctl confinement differs from policy")
    if (
        intent.outer_cgroup_memory_swap_max_bytes_pre_go
        != policy.outer_cgroup_memory_swap_max_bytes
    ):
        _fail("runtime memory.swap.max differs from policy")
    if (
        intent.outer_cgroup_memory_zswap_writeback_enabled_pre_go
        != policy.outer_cgroup_memory_zswap_writeback_enabled
    ):
        _fail("runtime memory.zswap.writeback differs from policy")


def validate_storage_runtime_intent_artifact_bindings_v1(
    intent: StorageBoundaryRuntimeIntentV1,
    bindings: StorageRuntimeIntentExternalBindingsV1,
) -> None:
    """Cross-check every phase-6 provider input supplied independently of intent bytes."""

    intent = _validate_current_storage_boundary_runtime_intent_v1(intent)
    bindings = _validate_current_storage_runtime_intent_external_bindings_v1(bindings)
    for name in StorageRuntimeIntentExternalBindingsV1.__dataclass_fields__:
        if getattr(intent, name) != getattr(bindings, name):
            _fail(f"runtime intent external {name} differs")


def validate_storage_boundary_receipt_v2(
    policy: StorageBackendPolicyV1,
    intent: StorageBoundaryRuntimeIntentV1,
    receipt: StorageBoundaryReceiptV2,
) -> None:
    validate_storage_policy_runtime_intent_v1(policy, intent)
    receipt = _validate_current_storage_boundary_receipt_v2(receipt)
    if receipt.policy != storage_backend_policy_identity_v1(policy):
        _fail("receipt storage policy identity differs")
    if receipt.runtime_intent != storage_boundary_runtime_intent_identity_v1(intent):
        _fail("receipt runtime intent identity differs")
    for name in (
        "campaign_id",
        "case_ordinal",
        "candidate_id",
        "candidate_family",
        "qualification_case_id",
        "image_id",
    ):
        if getattr(receipt, name) != getattr(intent, name):
            _fail(f"receipt {name} crosswires runtime intent")
    handshake = receipt.handshake
    for name in (
        "campaign_id",
        "case_ordinal",
        "candidate_id",
        "qualification_case_id",
        "image_id",
        "container_name",
        "container_id_commitment_sha256",
    ):
        if getattr(handshake, name) != getattr(intent, name):
            _fail(f"host handshake {name} crosswires runtime intent")
    if receipt.max_temporary_peak_bytes != policy.max_temporary_peak_bytes:
        _fail("receipt field-24 conservative limit differs from policy")
    if receipt.tmpfs_conservative_bound_evidence.aggregate_root_path != policy.aggregate_root_path:
        _fail("receipt tmpfs aggregate root differs from policy")
    if (
        receipt.tmpfs_conservative_bound_evidence.aggregate_root_case_exclusive
        != intent.aggregate_root_case_exclusive
        or receipt.tmpfs_conservative_bound_evidence.aggregate_root_case_exclusive
        != policy.aggregate_root_case_exclusive
    ):
        _fail("receipt tmpfs aggregate-root case exclusivity differs")
    if receipt.components != policy.components:
        _fail("receipt storage components differ from policy")
    raw_by_schema = {artifact.schema_version: artifact for artifact in receipt.raw_artifacts}
    for artifact in receipt.raw_artifacts:
        if raw_by_schema.get(artifact.schema_version) != artifact:
            _fail("typed storage evidence crosswires its first-class raw artifact")
    if receipt.tmpfs_conservative_bound_evidence.raw_mountinfo != intent.tmpfs_mount_identity:
        _fail("receipt tmpfs mountinfo crosswires the pre-GO runtime boundary")
    if receipt.disk_absence_evidence.raw_mountinfo != intent.mount_inventory:
        _fail("receipt disk mount inventory crosswires the pre-GO runtime boundary")
    if receipt.disk_absence_evidence.raw_absence_scan != intent.path_inventory:
        _fail("receipt disk path inventory crosswires the pre-GO runtime boundary")
    closure = receipt.swap_and_implicit_mount_closure_evidence
    if closure.outer_cgroup_identity_sha256 != intent.outer_cgroup_identity_sha256:
        _fail("receipt retained swap counters crosswire the pre-GO outer cgroup")
    if closure.raw_memory_swap_max_pre_go != intent.outer_cgroup_memory_swap_max_pre_go:
        _fail("receipt memory.swap.max pre-GO raw identity differs from runtime intent")
    if closure.raw_swap_counters_initial != intent.outer_cgroup_swap_counters_initial:
        _fail("receipt initial swap counters differ from runtime intent")
    if (
        closure.raw_memory_zswap_writeback_pre_go
        != intent.outer_cgroup_memory_zswap_writeback_pre_go
    ):
        _fail("receipt zswap writeback pre-GO raw identity differs from runtime intent")
    if closure.raw_docker_implicit_mount_inventory != intent.docker_implicit_mount_inventory:
        _fail("receipt Docker implicit mount inventory differs from runtime intent")
    for observed, expected, label in (
        (closure.raw_docker_create_inspect, intent.docker_create_inspect, "Docker create/inspect"),
        (closure.raw_final_oci_spec, intent.final_oci_spec, "final OCI spec"),
        (
            closure.raw_console_stdio_inventory,
            intent.console_stdio_inventory,
            "console/stdin/stdout inventory",
        ),
        (
            closure.raw_rootfs_upperdir_pre_go_baseline,
            intent.rootfs_upperdir_pre_go_baseline,
            "rootfs upperdir pre-GO baseline",
        ),
        (
            closure.docker_volume_inventory_pre_go_baseline,
            intent.docker_volume_inventory_pre_go_baseline,
            "Docker volume inventory pre-GO baseline",
        ),
    ):
        if observed != expected:
            _fail(f"receipt {label} differs from runtime intent")
    for name in (
        "application_writable_tmpfs_count",
        "docker_implicit_mounts_all_read_only",
        "docker_implicit_mounts_all_nonallocatable",
        "docker_implicit_mounts_have_application_writable_path",
        "persistent_storage_scope",
        "persistent_storage_measurement_interval",
        "trusted_runtime_bookkeeping_exclusions",
        "docker_implicit_config_bind_paths",
        "implicit_readonly_etc_bind_count",
        "added_device_count",
        "writable_persistent_mount_count",
        "writable_persistent_fd_count",
        "candidate_stdio_transport_count",
        "image_declared_volume_count",
        "docker_ipc_mode",
        "docker_shm_mount_present",
        "docker_tty_enabled",
        "docker_stdin_open",
        "docker_exec_permitted",
        "docker_archive_api_permitted",
        "docker_api_candidate_accessible",
        "container_console_or_fifo_candidate_accessible",
        "default_device_inventory_exact",
        "default_devices_can_allocate_storage",
        "memfd_posix_or_sysv_shm_permitted",
        "post_go_mount_mutation_permitted",
        "candidate_cgroup_mutation_permitted",
        "daemon_runtime_storage_candidate_accessible",
        "host_archival_candidate_accessible",
    ):
        if getattr(closure, name) != getattr(intent, name):
            _fail(f"receipt closure {name} differs from runtime intent")
    if (
        closure.user_bind_mount_count != intent.application_bind_mount_count
        or closure.user_volume_mount_count != intent.volume_count
        or closure.docker_implicit_mount_inventory_complete
        != intent.docker_implicit_mount_inventory_required
        or closure.application_writable_tmpfs_allocatable != intent.aggregate_root_allocatable
        or closure.memory_swap_max_bytes_pre_go != intent.outer_cgroup_memory_swap_max_bytes_pre_go
        or closure.memory_zswap_writeback_enabled_pre_go
        != intent.outer_cgroup_memory_zswap_writeback_enabled_pre_go
        or closure.rootfs_upperdir_candidate_inaccessible
        == intent.rootfs_upperdir_candidate_writable
    ):
        _fail("receipt closure topology or counter commitment differs from runtime intent")


def validate_storage_receipt_artifact_bindings_v2(
    receipt: StorageBoundaryReceiptV2,
    bindings: StorageReceiptExternalBindingsV1,
) -> None:
    """Bind every external receipt input to its independently supplied identity."""

    receipt = _validate_current_storage_boundary_receipt_v2(receipt)
    bindings = _validate_current_storage_receipt_external_bindings_v1(bindings)
    expected = (
        (
            receipt.handshake,
            bindings.host_handshake,
            "full host handshake projection",
        ),
        (
            receipt.host_go_storage_intent_binding,
            bindings.host_go_storage_intent_binding,
            "full host-GO storage-intent binding projection",
        ),
        (receipt.publication_wrapper, bindings.publication_wrapper, "publication wrapper"),
        (
            receipt.publication_reload_validation,
            bindings.publication_reload_validation,
            "publication reload validation",
        ),
        (
            receipt.terminal_relay_preseal_attestation,
            bindings.terminal_relay_preseal_attestation,
            "terminal relay preseal attestation",
        ),
        (
            receipt.nonstorage_channel_preseal_attestation,
            bindings.nonstorage_channel_preseal_attestation,
            "nonstorage channel preseal attestation",
        ),
        (
            receipt.irreversible_write_seal,
            bindings.irreversible_write_seal,
            "irreversible write seal",
        ),
    )
    for observed, supplied, label in expected:
        if observed != supplied:
            _fail(f"receipt {label} external identity differs")
    if receipt.raw_artifacts != bindings.raw_artifacts:
        _fail("receipt external raw artifact inventory differs")


def validate_storage_cleanup_reconciliation_v1(
    intent: StorageBoundaryRuntimeIntentV1,
    cleanup: StorageCleanupReconciliationV1,
    *,
    receipt: StorageBoundaryReceiptV2 | None,
    external_bindings: StorageCleanupExternalBindingsV1,
) -> None:
    intent = _validate_current_storage_boundary_runtime_intent_v1(intent)
    cleanup = _validate_current_storage_cleanup_reconciliation_v1(cleanup)
    external_bindings = _validate_current_storage_cleanup_external_bindings_v1(
        external_bindings
    )
    if receipt is not None:
        receipt = _validate_current_storage_boundary_receipt_v2(receipt)
    if cleanup.runtime_intent != storage_boundary_runtime_intent_identity_v1(intent):
        _fail("cleanup runtime intent identity differs")
    if cleanup.cleanup_producer != intent.components[5]:
        _fail("cleanup namespace producer differs from the pre-GO runtime intent")
    for observed, supplied, label in (
        (
            cleanup.attempted_receipt,
            external_bindings.attempted_receipt,
            "attempted receipt",
        ),
        (
            cleanup.failure_frontier,
            external_bindings.operational_failure_frontier,
            "operational failure frontier",
        ),
        (
            cleanup.namespace_cleanup_receipt,
            external_bindings.namespace_cleanup_receipt,
            "namespace cleanup receipt",
        ),
        (
            cleanup.cleanup_failure_frontier,
            external_bindings.cleanup_failure_frontier,
            "cleanup failure frontier",
        ),
        (
            cleanup.cleanup_producer,
            external_bindings.cleanup_producer,
            "cleanup producer",
        ),
    ):
        if observed != supplied:
            _fail(f"cleanup external {label} differs")
    for name in (
        "campaign_id",
        "case_ordinal",
        "candidate_id",
        "candidate_family",
        "qualification_case_id",
    ):
        if getattr(cleanup, name) != getattr(intent, name):
            _fail(f"cleanup {name} crosswires runtime intent")
    if cleanup.receipt_committed:
        if type(receipt) is not StorageBoundaryReceiptV2:
            _fail("committed cleanup requires the exact receipt")
        if cleanup.receipt != storage_boundary_receipt_v2_identity(receipt):
            _fail("cleanup committed receipt identity differs")
    elif receipt is not None:
        _fail("failed or uncertain cleanup must not receive a committed receipt")


def validate_storage_backend_chain_v2(
    policy: StorageBackendPolicyV1,
    intent: StorageBoundaryRuntimeIntentV1,
    receipt: StorageBoundaryReceiptV2 | None,
    cleanup: StorageCleanupReconciliationV1,
    *,
    intent_bindings: StorageRuntimeIntentExternalBindingsV1,
    receipt_bindings: StorageReceiptExternalBindingsV1 | None,
    cleanup_bindings: StorageCleanupExternalBindingsV1,
) -> None:
    validate_storage_policy_runtime_intent_v1(policy, intent)
    validate_storage_runtime_intent_artifact_bindings_v1(intent, intent_bindings)
    if receipt is not None:
        validate_storage_boundary_receipt_v2(policy, intent, receipt)
        if type(receipt_bindings) is not StorageReceiptExternalBindingsV1:
            _fail("committed receipt requires exact external receipt bindings")
        validate_storage_receipt_artifact_bindings_v2(receipt, receipt_bindings)
    elif receipt_bindings is not None:
        _fail("failed or uncertain chain must not receive receipt bindings")
    validate_storage_cleanup_reconciliation_v1(
        intent,
        cleanup,
        receipt=receipt,
        external_bindings=cleanup_bindings,
    )


__all__ = (
    "STORAGE_BACKEND_CONTRACT_DESCRIPTOR_SCHEMA_VERSION",
    "STORAGE_BACKEND_POLICY_SCHEMA_VERSION",
    "STORAGE_BOUNDARY_RUNTIME_INTENT_SCHEMA_VERSION",
    "STORAGE_BOUNDARY_RECEIPT_V2_SCHEMA_VERSION",
    "STORAGE_CLEANUP_RECONCILIATION_SCHEMA_VERSION",
    "QUALIFICATION_PLAN_V3_SCHEMA_VERSION",
    "RUNTIME_QUALIFICATION_RECEIPT_SCHEMA_VERSION",
    "HOST_PROVISIONING_VALIDATED_PRE_GO_PREFIX_V3_SCHEMA_VERSION",
    "HOST_CASE_REQUEST_V3_SCHEMA_VERSION",
    "HOST_CASE_INTENT_V3_SCHEMA_VERSION",
    "HOST_READY_V3_SCHEMA_VERSION",
    "HOST_OBSERVER_ANCHOR_V3_SCHEMA_VERSION",
    "HOST_GO_V3_SCHEMA_VERSION",
    "NORMALIZED_PUBLICATION_WRAPPER_SCHEMA_VERSION",
    "PUBLICATION_RELOAD_VALIDATION_SCHEMA_VERSION",
    "TERMINAL_RELAY_PRESEAL_ATTESTATION_SCHEMA_VERSION",
    "NONSTORAGE_CHANNEL_PRESEAL_ATTESTATION_SCHEMA_VERSION",
    "IRREVERSIBLE_WRITE_SEAL_SCHEMA_VERSION",
    "STORAGE_OPERATIONAL_FAILURE_SCHEMA_VERSION",
    "STORAGE_NAMESPACE_CLEANUP_RECEIPT_SCHEMA_VERSION",
    "STORAGE_CLEANUP_FAILURE_FRONTIER_SCHEMA_VERSION",
    "MOUNT_NAMESPACE_IDENTITY_SCHEMA_VERSION",
    "ROOTFS_MOUNT_IDENTITY_SCHEMA_VERSION",
    "TMPFS_MOUNT_IDENTITY_SCHEMA_VERSION",
    "TMPFS_BACKING_IDENTITY_SCHEMA_VERSION",
    "MOUNT_INVENTORY_SCHEMA_VERSION",
    "PATH_INVENTORY_SCHEMA_VERSION",
    "STORAGE_ROOT_INVENTORY_SCHEMA_VERSION",
    "STORAGE_FIELD_INVENTORY_SCHEMA_VERSION",
    "RAW_SCHEMA_INVENTORY_IDENTITY_SCHEMA_VERSION",
    "DOCKER_CREATE_INSPECT_SCHEMA_VERSION",
    "FINAL_OCI_SPEC_SCHEMA_VERSION",
    "CONSOLE_STDIO_INVENTORY_SCHEMA_VERSION",
    "DOCKER_API_OPERATION_JOURNAL_SCHEMA_VERSION",
    "ROOTFS_UPPERDIR_ACCESSIBILITY_SCHEMA_VERSION",
    "ROOTFS_UPPERDIR_INTERVAL_DELTA_SCHEMA_VERSION",
    "DOCKER_VOLUME_INVENTORY_DELTA_SCHEMA_VERSION",
    "DOCKER_VOLUME_INVENTORY_PRE_GO_BASELINE_SCHEMA_VERSION",
    "MEASUREMENT_PRODUCER_DESCRIPTOR_SCHEMA_VERSION",
    "TERMINAL_RELAY_DESCRIPTOR_SCHEMA_VERSION",
    "NONSTORAGE_CHANNEL_DESCRIPTOR_SCHEMA_VERSION",
    "WRITE_SEAL_PRODUCER_DESCRIPTOR_SCHEMA_VERSION",
    "RUNTIME_STORAGE_ESCAPE_GATE_DESCRIPTOR_SCHEMA_VERSION",
    "NAMESPACE_CLEANUP_PRODUCER_DESCRIPTOR_SCHEMA_VERSION",
    "RESOURCE_FIELDS",
    "RESOURCE_FIELD_ORDER_SHA256",
    "CANDIDATE_ORDER_SHA256",
    "MATCHED_V3_CANDIDATE_IDS",
    "STORAGE_COMPONENT_ROLES",
    "STORAGE_COMPONENT_DESCRIPTOR_SCHEMAS",
    "STORAGE_COMPONENT_DESCRIPTOR_SCHEMA_INVENTORY",
    "RAW_ARTIFACT_SCHEMA_INVENTORY",
    "REQUIRED_EXTERNAL_ARTIFACT_SCHEMAS",
    "TMPFS_AGGREGATE_ROOT",
    "PERSISTENT_STORAGE_SCOPE",
    "PERSISTENT_STORAGE_MEASUREMENT_INTERVAL",
    "TRUSTED_RUNTIME_BOOKKEEPING_EXCLUSIONS",
    "DOCKER_IMPLICIT_CONFIG_BIND_PATHS",
    "FIELD24_POSITION",
    "FIELD25_POSITION",
    "DESCRIPTOR_SERIALIZED_SELF_PIN_SENTINEL",
    "PINNED_STORAGE_BACKEND_V2_DESCRIPTOR_FILE_SHA256",
    "PINNED_STORAGE_BACKEND_V2_DESCRIPTOR_BODY_SHA256",
    "SOURCE_ONLY_CAPABILITIES",
    "SOURCE_ONLY_READINESS",
    "SOURCE_ONLY_AUTHORITY",
    "SOURCE_ONLY_CLAIMS",
    "ForagerMatchedV3QualificationStorageBackendV2Error",
    "ArtifactIdentityV1",
    "PinnedStorageComponentIdentityV1",
    "StorageBackendPolicyV1",
    "StorageBoundaryRuntimeIntentV1",
    "HostV3HandshakeProjectionV1",
    "RawArtifactProjectionV1",
    "HostGoStorageIntentBindingV1",
    "TmpfsConservativeBoundEvidenceV1",
    "DiskStructuralAbsenceEvidenceV1",
    "SwapAndImplicitMountClosureEvidenceV1",
    "StorageBoundaryReceiptV2",
    "StorageCleanupReconciliationV1",
    "StorageRuntimeIntentExternalBindingsV1",
    "StorageReceiptExternalBindingsV1",
    "StorageCleanupExternalBindingsV1",
    "StorageBackendContractDescriptorV2",
    "canonical_storage_backend_json_bytes",
    "decode_canonical_storage_backend_json",
    "canonical_storage_backend_policy_v1_body_bytes",
    "canonical_storage_backend_policy_v1_file_bytes",
    "canonical_storage_boundary_runtime_intent_v1_body_bytes",
    "canonical_storage_boundary_runtime_intent_v1_file_bytes",
    "canonical_storage_boundary_receipt_v2_body_bytes",
    "canonical_storage_boundary_receipt_v2_file_bytes",
    "canonical_storage_cleanup_reconciliation_v1_body_bytes",
    "canonical_storage_cleanup_reconciliation_v1_file_bytes",
    "canonical_storage_backend_contract_descriptor_v2_body_bytes",
    "canonical_storage_backend_contract_descriptor_v2_file_bytes",
    "storage_backend_policy_identity_v1",
    "storage_boundary_runtime_intent_identity_v1",
    "storage_boundary_receipt_v2_identity",
    "storage_cleanup_reconciliation_identity_v1",
    "parse_storage_backend_policy_v1",
    "parse_storage_boundary_runtime_intent_v1",
    "parse_storage_boundary_receipt_v2",
    "parse_storage_cleanup_reconciliation_v1",
    "parse_storage_backend_contract_descriptor_v2",
    "storage_backend_v2_descriptor_sha256",
    "validate_storage_policy_runtime_intent_v1",
    "validate_storage_runtime_intent_artifact_bindings_v1",
    "validate_storage_boundary_receipt_v2",
    "validate_storage_receipt_artifact_bindings_v2",
    "validate_storage_cleanup_reconciliation_v1",
    "validate_storage_backend_chain_v2",
)
