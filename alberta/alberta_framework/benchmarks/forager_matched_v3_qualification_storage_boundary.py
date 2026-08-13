"""Source-only storage-boundary contract for matched Forager v3.

This module constructs and parses canonical metadata only.  It never inspects
storage, starts a container, samples a filesystem, installs a quota, reads a
clock, or grants execution or qualification authority.

The dependency order is deliberately acyclic.  A storage intent is committed
before host GO.  After native case publication, a normalized wrapper commits
the expected reload observation and the shared publication reload-validation
artifact proves an exact read-only reload.  An irreversible preterminal write
seal then closes every measured writable path: later container or descendant
writes, allocations, and copy-up are impossible, terminal transport is outside
the measured boundary, and teardown is deletion-only.  The storage receipt is
then committed and may be bound by terminal-v2.  Consequently neither artifact
defined here may bind terminal, lifecycle, host-success, or merger identities.
"""

from __future__ import annotations

import copy
import hashlib
import hmac
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final, Literal, NoReturn, cast

QUALIFICATION_STORAGE_BOUNDARY_CONTRACT_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.qualification_storage_boundary_contract_descriptor.v1"
)
QUALIFICATION_STORAGE_BOUNDARY_PRODUCER_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.qualification_storage_boundary_producer_descriptor.v1"
)
QUALIFICATION_STORAGE_BOUNDARY_INTENT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.qualification_storage_boundary_intent.v1"
)
QUALIFICATION_STORAGE_BOUNDARY_RECEIPT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.qualification_storage_boundary_receipt.v1"
)
QUALIFICATION_STORAGE_WRITE_SEAL_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.qualification_storage_write_quiescence_seal.v1"
)
QUALIFICATION_PUBLICATION_RELOAD_VALIDATION_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.qualification_publication_reload_validation.v1"
)
QUALIFICATION_STORAGE_TERMINAL_RELAY_PRESEAL_ATTESTATION_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.qualification_storage_terminal_relay_preseal_attestation.v1"
)
QUALIFICATION_STORAGE_NONSTORAGE_CHANNEL_READINESS_ATTESTATION_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.qualification_storage_nonstorage_channel_readiness_attestation.v1"
)
QUALIFICATION_STORAGE_TERMINAL_RELAY_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.qualification_storage_terminal_relay_descriptor.v1"
)
QUALIFICATION_STORAGE_NONSTORAGE_CHANNEL_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.qualification_storage_nonstorage_channel_descriptor.v1"
)
QUALIFICATION_STORAGE_WRITE_SEAL_PRODUCER_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.qualification_storage_write_seal_producer_descriptor.v1"
)

HOST_PROVISIONING_RECEIPT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.host_provisioning_receipt.v2"
)
RUNTIME_CANDIDATE_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.runtime_observation_candidate.v2"
)
RUNTIME_QUALIFICATION_RECEIPT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.runtime_qualification_receipt.v1"
)
HOST_CASE_REQUEST_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.host_qualification_case_request.v2"
)
HOST_CASE_INTENT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.host_qualification_case_intent.v2"
)
HOST_READY_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.in_container_qualification_driver_ready.v2"
)
HOST_OBSERVER_ANCHOR_SCHEMA_VERSION: Final = "alberta.forager_matched_v3.host_observer_anchor.v2"
HOST_GO_SCHEMA_VERSION: Final = "alberta.forager_matched_v3.host_qualification_go_commitment.v2"
NORMALIZED_PUBLICATION_COMMITMENT_WRAPPER_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.qualification_publication_commitment_wrapper.v1"
)

QUALIFICATION_STORAGE_BOUNDARY_CONTRACT_STATUS: Final = (
    "implemented_source_only_contract_uninvoked_no_production_receipt"
)
QUALIFICATION_STORAGE_BOUNDARY_CONTRACT_CLASSIFICATION: Final = (
    "score_blind_metadata_only_storage_boundary_contract_non_authorizing"
)
QUALIFICATION_STORAGE_BOUNDARY_INTENT_STATUS: Final = (
    "pre_go_storage_boundary_intent_content_only_non_authorizing"
)
QUALIFICATION_STORAGE_BOUNDARY_RECEIPT_STATUS: Final = (
    "post_case_preterminal_irreversibly_sealed_storage_receipt_non_authorizing"
)

EVENT_COMPLETE_ACCOUNTING_MODE: Final = "event_complete_kernel_filesystem_accounting"
HARD_QUOTA_ENFORCEMENT_MODE: Final = "immutable_non_bypass_pre_go_hard_quota_enforcement"
EXACT_OBSERVATION: Final = "exact_observation"
CONSERVATIVE_ENFORCED_UPPER_BOUND: Final = "conservative_enforced_upper_bound"
STORAGE_LIFETIME_SCOPE: Final = (
    "fresh_case_storage_boundary_through_irreversible_preterminal_write_seal"
)
NOT_ABSENT: Final = "not_absent"
TEMPORARY_STORAGE_STRUCTURALLY_ABSENT: Final = "temporary_storage_scope_structurally_absent"
DISK_STORAGE_STRUCTURALLY_ABSENT: Final = "disk_storage_scope_structurally_absent"

STORAGE_ENFORCEMENT_KINDS: Final = (
    "kernel_project_hard_quota",
    "dedicated_filesystem_capacity_limit",
    "tmpfs_hard_size_limit",
)

MATCHED_V3_LOCAL_CANDIDATE_IDS: Final = (
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
)
MATCHED_V3_EXTERNAL_CANDIDATE_IDS: Final = (
    "external_dqn_plain",
    "external_dqn_crelu",
    "external_dqn_redo",
    "external_dqn_reward_trace",
    "external_dqn_l2_init",
    "external_pt_dqn_xfinal",
    "external_drqn_xfinal",
    "isolated_ppo_generic",
    "isolated_rtu_paper_scale",
    "random_policy",
    "search_nearest",
    "search_oracle",
)
MATCHED_V3_ADAPTER_CANDIDATE_IDS: Final = (
    "adapted_full_rainbow",
    "adapted_ppo_gru",
)
MATCHED_V3_STORAGE_CANDIDATE_IDS: Final = (
    MATCHED_V3_LOCAL_CANDIDATE_IDS
    + MATCHED_V3_EXTERNAL_CANDIDATE_IDS[:9]
    + MATCHED_V3_ADAPTER_CANDIDATE_IDS
    + MATCHED_V3_EXTERNAL_CANDIDATE_IDS[9:]
)
MATCHED_V3_STORAGE_CANDIDATE_ORDER_SHA256: Final = (
    "d93aaf66053aaf9a7b1c6d268a47740078dd2c1007f7287bd80908707e40b858"
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
STORAGE_RESOURCE_FIELDS: Final = (
    "max_temporary_peak_bytes",
    "max_disk_peak_bytes",
)
STORAGE_FIELD_POSITIONS: Final[Mapping[str, int]] = MappingProxyType(
    {
        "max_temporary_peak_bytes": 24,
        "max_disk_peak_bytes": 25,
    }
)
STORAGE_FIELD_ZERO_ABSENCE: Final[Mapping[str, str]] = MappingProxyType(
    {
        "max_temporary_peak_bytes": TEMPORARY_STORAGE_STRUCTURALLY_ABSENT,
        "max_disk_peak_bytes": DISK_STORAGE_STRUCTURALLY_ABSENT,
    }
)

_MAX_ARTIFACT_BYTES: Final = 4 * 1024 * 1024
_MAX_JSON_DEPTH: Final = 64
_MAX_JSON_NODES: Final = 40_000
_MAX_TEXT_LENGTH: Final = 16_384
_MAX_INTEGER: Final = 2**63 - 1
_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_IMAGE_ID_RE: Final = re.compile(r"sha256:[0-9a-f]{64}\Z")
_IDENTIFIER_RE: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}\Z")

_FORBIDDEN_REVERSE_BINDING_KEYS: Final = frozenset(
    {
        "evaluation",
        "evaluation_receipt",
        "evaluator_identity",
        "evaluator_receipt",
        "full_resource_merger",
        "full_resource_merger_receipt",
        "host_execution_receipt",
        "host_success",
        "host_success_v2",
        "host_success_v2_identity",
        "host_success_v2_receipt",
        "host_success_receipt",
        "issuance",
        "issuance_receipt",
        "issuer_identity",
        "issuer_receipt",
        "lifecycle",
        "lifecycle_record",
        "lifecycle_v2",
        "lifecycle_v2_identity",
        "lifecycle_v2_record",
        "merger",
        "observation_handoff",
        "qualification_evaluator",
        "qualification_evaluator_receipt",
        "qualification_issuer",
        "qualification_issuer_receipt",
        "resource_merger",
        "terminal",
        "terminal_metadata",
        "terminal_receipt",
        "terminal_v2",
        "terminal_v2_identity",
        "terminal_v2_metadata",
        "terminal_v2_receipt",
    }
)


class ForagerMatchedV3QualificationStorageBoundaryError(ValueError):
    """A source-only storage-boundary artifact failed closed."""


def _fail(message: str) -> NoReturn:
    raise ForagerMatchedV3QualificationStorageBoundaryError(message)


def _require_sha256(value: object, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None or value == "0" * 64:
        _fail(f"{label} must be one nonzero lowercase SHA-256")
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


def _require_image_id(value: object, label: str) -> str:
    if (
        type(value) is not str
        or _IMAGE_ID_RE.fullmatch(value) is None
        or value == "sha256:" + "0" * 64
    ):
        _fail(f"{label} must be one immutable sha256 image ID")
    return value


def _require_absolute_path(value: object, label: str) -> str:
    path = _require_text(value, label)
    if not path.startswith("/") or path.startswith("//"):
        _fail(f"{label} must be one canonical absolute POSIX path")
    if path != "/" and path.endswith("/"):
        _fail(f"{label} must not have one trailing slash")
    parts = path.split("/")[1:]
    if any(part in {"", ".", ".."} for part in parts):
        _fail(f"{label} contains an empty, dot, or parent component")
    return path


def _require_exact_keys(
    value: object,
    expected: frozenset[str],
    label: str,
) -> dict[str, Any]:
    if type(value) is not dict or frozenset(value) != expected:
        _fail(f"{label} keys differ")
    return cast(dict[str, Any], value)


def _reject_constant(value: str) -> NoReturn:
    _fail(f"storage-boundary JSON contains forbidden constant {value!r}")


def _reject_float(value: str) -> NoReturn:
    _fail(f"storage-boundary JSON contains forbidden float {value!r}")


def _parse_int(value: str) -> int:
    if len(value.lstrip("-")) > 19:
        _fail("storage-boundary JSON integer exceeds its lexical bound")
    parsed = int(value, 10)
    if not -_MAX_INTEGER <= parsed <= _MAX_INTEGER:
        _fail("storage-boundary JSON integer exceeds its value bound")
    return parsed


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if type(key) is not str or key in result:
            _fail("storage-boundary JSON contains a duplicate or non-text key")
        result[key] = value
    return result


def _assert_plain_unaliased_json(value: object) -> None:
    seen: set[int] = set()
    nodes = 0

    def visit(item: object, depth: int) -> None:
        nonlocal nodes
        nodes += 1
        if nodes > _MAX_JSON_NODES or depth > _MAX_JSON_DEPTH:
            _fail("storage-boundary JSON structure exceeds its bound")
        if item is None or type(item) is bool:
            return
        if type(item) is int:
            _require_int(item, "storage-boundary JSON integer", minimum=-_MAX_INTEGER)
            return
        if type(item) is str:
            if len(item) > _MAX_TEXT_LENGTH or any(
                ord(character) < 0x20 or ord(character) > 0x7E for character in item
            ):
                _fail("storage-boundary JSON strings must be bounded printable ASCII")
            return
        if type(item) not in {dict, list}:
            _fail("storage-boundary JSON contains a non-plain value")
        identity = id(item)
        if identity in seen:
            _fail("storage-boundary JSON contains an alias or cycle")
        seen.add(identity)
        if type(item) is list:
            for child in cast(list[object], item):
                visit(child, depth + 1)
        else:
            for key, child in cast(dict[object, object], item).items():
                if type(key) is not str:
                    _fail("storage-boundary JSON keys must be exact strings")
                visit(key, depth + 1)
                visit(child, depth + 1)

    visit(value, 0)


def _reject_reverse_binding_keys(value: object) -> None:
    if type(value) is list:
        for child in cast(list[object], value):
            _reject_reverse_binding_keys(child)
        return
    if type(value) is not dict:
        return
    for key, child in cast(dict[str, object], value).items():
        if key in _FORBIDDEN_REVERSE_BINDING_KEYS:
            _fail(f"reverse or cyclic storage binding key {key!r} is forbidden")
        _reject_reverse_binding_keys(child)


def _exact_json_equal(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        left_map = cast(dict[str, object], left)
        right_map = cast(dict[str, object], right)
        return frozenset(left_map) == frozenset(right_map) and all(
            _exact_json_equal(left_map[key], right_map[key]) for key in left_map
        )
    if type(left) is list:
        left_items = cast(list[object], left)
        right_items = cast(list[object], right)
        return len(left_items) == len(right_items) and all(
            _exact_json_equal(a, b) for a, b in zip(left_items, right_items, strict=True)
        )
    return bool(left == right)


def _canonical_json(value: object, *, newline: bool = True) -> bytes:
    _assert_plain_unaliased_json(value)
    try:
        raw = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise ForagerMatchedV3QualificationStorageBoundaryError(
            "storage-boundary value is not canonical JSON"
        ) from exc
    if newline:
        raw += b"\n"
    if not 1 <= len(raw) <= _MAX_ARTIFACT_BYTES:
        _fail("storage-boundary canonical JSON exceeds its byte bound")
    return raw


def _strict_json_load(raw: bytes) -> dict[str, Any]:
    if type(raw) is not bytes or not 1 <= len(raw) <= _MAX_ARTIFACT_BYTES:
        _fail("storage-boundary bytes violate their bound")
    try:
        value = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_object_without_duplicates,
            parse_constant=_reject_constant,
            parse_float=_reject_float,
            parse_int=_parse_int,
        )
    except ForagerMatchedV3QualificationStorageBoundaryError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ForagerMatchedV3QualificationStorageBoundaryError(
            "storage-boundary bytes are not strict ASCII JSON"
        ) from exc
    if type(value) is not dict:
        _fail("storage-boundary JSON root must be one object")
    exact = cast(dict[str, Any], value)
    _assert_plain_unaliased_json(exact)
    _reject_reverse_binding_keys(exact)
    if not hmac.compare_digest(_canonical_json(exact), raw):
        _fail("storage-boundary bytes are not canonical")
    return exact


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _body_sha256(value: Mapping[str, Any]) -> str:
    return _sha256(_canonical_json(dict(value), newline=False))


def _with_body_sha256(body: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = dict(body)
    result[field] = _body_sha256(body)
    return result


def _validate_body_sha256(value: Mapping[str, Any], field: str, label: str) -> str:
    supplied = _require_sha256(value.get(field), f"{label} body")
    body = dict(value)
    body.pop(field, None)
    expected = _body_sha256(body)
    if not hmac.compare_digest(supplied, expected):
        _fail(f"{label} body SHA-256 differs")
    return supplied


def _validate_caller_file_pin(raw: bytes, expected_file_sha256: object, label: str) -> str:
    expected = _require_sha256(expected_file_sha256, f"{label} caller file pin")
    observed = _sha256(raw)
    if not hmac.compare_digest(observed, expected):
        _fail(f"{label} full-file SHA-256 differs from its caller pin")
    return observed


def _capabilities() -> dict[str, bool]:
    return {
        "artifact_authentication": False,
        "clock": False,
        "container_control": False,
        "default_inputs": False,
        "evaluator": False,
        "executor": False,
        "filesystem": False,
        "issuer": False,
        "network": False,
        "process": False,
        "producer_execution": False,
        "quota_control": False,
        "storage_measurement": False,
    }


def _readiness() -> dict[str, bool]:
    return {
        "evaluation_ready": False,
        "execution_ready": False,
        "issuance_ready": False,
        "measurement_producer_available": False,
        "production_receipt_available": False,
        "qualification_ready": False,
        "storage_boundary_ready": False,
    }


def _authority() -> dict[str, bool]:
    return {
        "evaluation_performed": False,
        "execution_authorized": False,
        "issuance_performed": False,
        "publication_authority_granted": False,
        "qualification_granted": False,
        "storage_boundary_authorized": False,
    }


def _claims() -> dict[str, bool]:
    return {
        "performance_claim_allowed": False,
        "promotion_allowed": False,
        "resource_matched": False,
        "runtime_qualified": False,
        "scientific_evidence_created": False,
        "source_qualified": False,
        "storage_peak_established_by_this_module": False,
        "universal_sota_claim_allowed": False,
    }


def _limitations() -> list[str]:
    return [
        "This contract validates canonical caller-pinned structural metadata only.",
        "No filesystem, container, quota, process, network, or clock API is available here.",
        "No storage measurement producer is implemented or invoked here.",
        "Polling, periodic sampling, du snapshots, and container layer sizes are insufficient.",
        "A missing value can never be interpreted as zero.",
        "A zero requires the exact typed structural-absence proof for its storage field.",
        "A receipt requires an irreversible preterminal seal proving no later peak increase.",
        "Terminal transport must remain outside measured storage and teardown is deletion-only.",
        "Relay delivery and terminal emission occur after this receipt and are not observed here.",
        "Downstream terminal-v2 and HostSuccess-v2 must prove delivery and emission.",
        "Terminal-v2 may bind the receipt one-way only after the receipt exists.",
        "Terminal, lifecycle, host-success, and merger reverse identities are forbidden.",
        "No ceiling comparison, qualification, issuance, evidence, or claim is produced.",
    ]


def _family_for_candidate(candidate_id: object) -> Literal["local", "external", "adapter"]:
    exact_candidate_id = _require_identifier(candidate_id, "candidate ID")
    if exact_candidate_id in MATCHED_V3_LOCAL_CANDIDATE_IDS:
        return "local"
    if exact_candidate_id in MATCHED_V3_EXTERNAL_CANDIDATE_IDS:
        return "external"
    if exact_candidate_id in MATCHED_V3_ADAPTER_CANDIDATE_IDS:
        return "adapter"
    _fail("candidate ID is outside the frozen matched-v3 universe")


def _require_case_projection(
    *,
    case_ordinal: object,
    candidate_id: object,
    candidate_family: object,
    qualification_case_id: object,
) -> tuple[int, str, Literal["local", "external", "adapter"], str]:
    ordinal = _require_int(
        case_ordinal,
        "case ordinal",
        maximum=len(MATCHED_V3_STORAGE_CANDIDATE_IDS) - 1,
    )
    expected_candidate = MATCHED_V3_STORAGE_CANDIDATE_IDS[ordinal]
    exact_candidate_id = _require_identifier(candidate_id, "candidate ID")
    if exact_candidate_id != expected_candidate:
        _fail("case ordinal and candidate ID differ from the frozen order")
    family = _family_for_candidate(exact_candidate_id)
    exact_family = _require_identifier(candidate_family, "candidate family")
    if exact_family != family:
        _fail("candidate family differs from the frozen case projection")
    case_id = f"qualification_{ordinal:02d}_{expected_candidate}"
    exact_case_id = _require_identifier(qualification_case_id, "qualification case ID")
    if exact_case_id != case_id:
        _fail("qualification case ID differs from the frozen case projection")
    return ordinal, expected_candidate, family, case_id


@dataclass(frozen=True, slots=True)
class ArtifactIdentityV1:
    """One caller-carried canonical artifact identity; bytes are never loaded."""

    schema_version: str
    file_sha256: str
    body_sha256: str

    def __post_init__(self) -> None:
        _require_identifier(self.schema_version, "artifact schema")
        _require_sha256(self.file_sha256, "artifact file")
        _require_sha256(self.body_sha256, "artifact body")

    def to_dict(self) -> dict[str, str]:
        return {
            "schema_version": self.schema_version,
            "file_sha256": self.file_sha256,
            "body_sha256": self.body_sha256,
        }


def _require_artifact_schema(
    artifact: object,
    schema_version: str,
    label: str,
) -> ArtifactIdentityV1:
    if type(artifact) is not ArtifactIdentityV1:
        _fail(f"{label} identity type differs")
    exact = artifact
    if exact.schema_version != schema_version:
        _fail(f"{label} schema differs")
    return exact


@dataclass(frozen=True, slots=True)
class ProducerIdentityV1:
    """One independently pinned storage-measurement producer identity."""

    descriptor_schema_version: str
    descriptor_sha256: str
    source_sha256: str

    def __post_init__(self) -> None:
        descriptor_schema = _require_identifier(
            self.descriptor_schema_version,
            "storage producer descriptor schema",
        )
        if descriptor_schema != QUALIFICATION_STORAGE_BOUNDARY_PRODUCER_DESCRIPTOR_SCHEMA_VERSION:
            _fail("storage producer descriptor schema differs")
        _require_sha256(self.descriptor_sha256, "storage producer descriptor")
        _require_sha256(self.source_sha256, "storage producer source")

    def to_dict(self) -> dict[str, str]:
        return {
            "descriptor_schema_version": self.descriptor_schema_version,
            "descriptor_sha256": self.descriptor_sha256,
            "source_sha256": self.source_sha256,
        }


@dataclass(frozen=True, slots=True)
class ComponentIdentityV1:
    """One independently pinned relay, channel, or write-seal component."""

    descriptor_schema_version: str
    descriptor_sha256: str
    source_sha256: str

    def __post_init__(self) -> None:
        descriptor_schema = _require_identifier(
            self.descriptor_schema_version,
            "storage seal component descriptor schema",
        )
        if descriptor_schema not in {
            QUALIFICATION_STORAGE_TERMINAL_RELAY_DESCRIPTOR_SCHEMA_VERSION,
            QUALIFICATION_STORAGE_NONSTORAGE_CHANNEL_DESCRIPTOR_SCHEMA_VERSION,
            QUALIFICATION_STORAGE_WRITE_SEAL_PRODUCER_DESCRIPTOR_SCHEMA_VERSION,
        }:
            _fail("storage seal component descriptor schema differs")
        _require_sha256(self.descriptor_sha256, "storage seal component descriptor")
        _require_sha256(self.source_sha256, "storage seal component source")

    def to_dict(self) -> dict[str, str]:
        return {
            "descriptor_schema_version": self.descriptor_schema_version,
            "descriptor_sha256": self.descriptor_sha256,
            "source_sha256": self.source_sha256,
        }


@dataclass(frozen=True, slots=True)
class StorageSealArchitectureV1:
    """Pre-GO worker/relay isolation and nonstorage terminal transport policy."""

    architecture_kind: Literal[
        "worker_exit_then_isolated_terminal_relay",
        "irreversible_seccomp_fd_closure_then_isolated_terminal_relay",
    ]
    terminal_relay: ComponentIdentityV1
    nonstorage_control_channel: ComponentIdentityV1
    write_seal_producer: ComponentIdentityV1
    terminal_relay_binary_sha256: str
    nonstorage_channel_commitment_sha256: str
    write_seal_policy_body_sha256: str
    container_log_driver: str
    architecture_committed_before_go: bool
    worker_and_terminal_relay_separated_before_go: bool
    terminal_relay_has_measured_writable_namespace: bool
    terminal_relay_has_measured_writable_fd: bool
    terminal_transport_uses_nonstorage_channel: bool
    container_logging_can_write_measured_storage: bool
    late_remount_used_or_permitted: bool

    def __post_init__(self) -> None:
        architecture_kind = _require_identifier(
            self.architecture_kind,
            "storage write-seal architecture",
        )
        if architecture_kind not in {
            "worker_exit_then_isolated_terminal_relay",
            "irreversible_seccomp_fd_closure_then_isolated_terminal_relay",
        }:
            _fail("storage write-seal architecture differs")
        expected_components = (
            (
                self.terminal_relay,
                QUALIFICATION_STORAGE_TERMINAL_RELAY_DESCRIPTOR_SCHEMA_VERSION,
                "terminal relay",
            ),
            (
                self.nonstorage_control_channel,
                QUALIFICATION_STORAGE_NONSTORAGE_CHANNEL_DESCRIPTOR_SCHEMA_VERSION,
                "nonstorage control channel",
            ),
            (
                self.write_seal_producer,
                QUALIFICATION_STORAGE_WRITE_SEAL_PRODUCER_DESCRIPTOR_SCHEMA_VERSION,
                "write-seal producer",
            ),
        )
        for component, schema, label in expected_components:
            if type(component) is not ComponentIdentityV1:
                _fail(f"{label} component identity type differs")
            if component.descriptor_schema_version != schema:
                _fail(f"{label} descriptor schema differs")
        if len({component for component, _, _ in expected_components}) != 3:
            _fail("relay, channel, and write-seal components cannot alias")
        for value, label in (
            (self.terminal_relay_binary_sha256, "terminal relay binary"),
            (self.nonstorage_channel_commitment_sha256, "nonstorage channel commitment"),
            (self.write_seal_policy_body_sha256, "write-seal policy body"),
        ):
            _require_sha256(value, label)
        container_log_driver = _require_identifier(
            self.container_log_driver,
            "container log driver",
        )
        if container_log_driver != "none":
            _fail("container log driver must be exact none")
        for flag, label in (
            (self.architecture_committed_before_go, "seal architecture committed before GO"),
            (
                self.worker_and_terminal_relay_separated_before_go,
                "worker and terminal relay separated before GO",
            ),
            (
                self.terminal_transport_uses_nonstorage_channel,
                "terminal transport uses nonstorage channel",
            ),
        ):
            if _require_bool(flag, label) is not True:
                _fail(f"{label} must be exact true")
        for flag, label in (
            (
                self.terminal_relay_has_measured_writable_namespace,
                "terminal relay has measured writable namespace",
            ),
            (
                self.terminal_relay_has_measured_writable_fd,
                "terminal relay has measured writable FD",
            ),
            (
                self.container_logging_can_write_measured_storage,
                "container logging can write measured storage",
            ),
            (self.late_remount_used_or_permitted, "late remount used or permitted"),
        ):
            if _require_bool(flag, label) is not False:
                _fail(f"{label} must be exact false")

    def to_dict(self) -> dict[str, Any]:
        return {
            "architecture_kind": self.architecture_kind,
            "terminal_relay": self.terminal_relay.to_dict(),
            "nonstorage_control_channel": self.nonstorage_control_channel.to_dict(),
            "write_seal_producer": self.write_seal_producer.to_dict(),
            "terminal_relay_binary_sha256": self.terminal_relay_binary_sha256,
            "nonstorage_channel_commitment_sha256": (self.nonstorage_channel_commitment_sha256),
            "write_seal_policy_body_sha256": self.write_seal_policy_body_sha256,
            "container_log_driver": self.container_log_driver,
            "architecture_committed_before_go": self.architecture_committed_before_go,
            "worker_and_terminal_relay_separated_before_go": (
                self.worker_and_terminal_relay_separated_before_go
            ),
            "terminal_relay_has_measured_writable_namespace": (
                self.terminal_relay_has_measured_writable_namespace
            ),
            "terminal_relay_has_measured_writable_fd": (
                self.terminal_relay_has_measured_writable_fd
            ),
            "terminal_transport_uses_nonstorage_channel": (
                self.terminal_transport_uses_nonstorage_channel
            ),
            "container_logging_can_write_measured_storage": (
                self.container_logging_can_write_measured_storage
            ),
            "late_remount_used_or_permitted": self.late_remount_used_or_permitted,
        }

    @property
    def body_sha256(self) -> str:
        return _body_sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class RuntimeStorageIdentityV1:
    """Immutable runtime artifacts and executable/configuration identity."""

    runtime_candidate: ArtifactIdentityV1
    runtime_qualification_receipt: ArtifactIdentityV1
    runtime_name: str
    runtime_binary_sha256: str
    runtime_configuration_body_sha256: str

    def __post_init__(self) -> None:
        _require_artifact_schema(
            self.runtime_candidate,
            RUNTIME_CANDIDATE_SCHEMA_VERSION,
            "runtime candidate",
        )
        _require_artifact_schema(
            self.runtime_qualification_receipt,
            RUNTIME_QUALIFICATION_RECEIPT_SCHEMA_VERSION,
            "runtime qualification receipt",
        )
        _require_identifier(self.runtime_name, "runtime name")
        _require_sha256(self.runtime_binary_sha256, "runtime binary")
        _require_sha256(
            self.runtime_configuration_body_sha256,
            "runtime configuration body",
        )
        if self.runtime_candidate == self.runtime_qualification_receipt:
            _fail("runtime candidate and qualification receipt cannot alias")

    def to_dict(self) -> dict[str, Any]:
        return {
            "runtime_candidate": self.runtime_candidate.to_dict(),
            "runtime_qualification_receipt": self.runtime_qualification_receipt.to_dict(),
            "runtime_name": self.runtime_name,
            "runtime_binary_sha256": self.runtime_binary_sha256,
            "runtime_configuration_body_sha256": (self.runtime_configuration_body_sha256),
        }


@dataclass(frozen=True, slots=True)
class ContainerStorageIdentityV1:
    """Precommitted container and mount-namespace storage identity."""

    container_name: str
    container_identity_commitment_sha256: str
    mount_namespace_identity_sha256: str
    rootfs_mount_identity_sha256: str
    image_layers_read_only: bool
    container_identity_precommitted: bool
    mount_namespace_mutation_disabled_after_go: bool
    rootfs_copy_up_policy: Literal["bound_writable_root", "copy_up_disabled"]

    def __post_init__(self) -> None:
        _require_identifier(self.container_name, "container name")
        for value, label in (
            (self.container_identity_commitment_sha256, "container identity commitment"),
            (self.mount_namespace_identity_sha256, "mount namespace identity"),
            (self.rootfs_mount_identity_sha256, "rootfs mount identity"),
        ):
            _require_sha256(value, label)
        if _require_bool(self.image_layers_read_only, "image layers read-only") is not True:
            _fail("image layers must be immutable and read-only")
        if (
            _require_bool(
                self.container_identity_precommitted,
                "container identity precommitted",
            )
            is not True
        ):
            _fail("container identity must be precommitted")
        if (
            _require_bool(
                self.mount_namespace_mutation_disabled_after_go,
                "mount namespace mutation disabled after GO",
            )
            is not True
        ):
            _fail("mount namespace mutation must be disabled after GO")
        rootfs_copy_up_policy = _require_identifier(
            self.rootfs_copy_up_policy,
            "rootfs copy-up policy",
        )
        if rootfs_copy_up_policy not in {"bound_writable_root", "copy_up_disabled"}:
            _fail("rootfs copy-up policy differs")

    def to_dict(self) -> dict[str, Any]:
        return {
            "container_name": self.container_name,
            "container_identity_commitment_sha256": (self.container_identity_commitment_sha256),
            "mount_namespace_identity_sha256": self.mount_namespace_identity_sha256,
            "rootfs_mount_identity_sha256": self.rootfs_mount_identity_sha256,
            "image_layers_read_only": self.image_layers_read_only,
            "container_identity_precommitted": self.container_identity_precommitted,
            "mount_namespace_mutation_disabled_after_go": (
                self.mount_namespace_mutation_disabled_after_go
            ),
            "rootfs_copy_up_policy": self.rootfs_copy_up_policy,
        }


@dataclass(frozen=True, slots=True)
class StorageRootBindingV1:
    """One fresh, exclusive writable root assigned to exactly one storage field."""

    root_id: str
    field_name: Literal["max_temporary_peak_bytes", "max_disk_peak_bytes"]
    absolute_path: str
    mount_identity_sha256: str
    backing_store_identity_sha256: str
    filesystem_type: str
    created_exclusively_for_case: bool
    empty_at_fresh_boundary: bool
    writable: bool
    includes_overlay_copy_up: bool

    def __post_init__(self) -> None:
        _require_identifier(self.root_id, "storage root ID")
        field_name = _require_identifier(self.field_name, "storage root field")
        if field_name not in STORAGE_RESOURCE_FIELDS:
            _fail("storage root field differs")
        _require_absolute_path(self.absolute_path, "storage root path")
        _require_sha256(self.mount_identity_sha256, "storage root mount identity")
        _require_sha256(
            self.backing_store_identity_sha256,
            "storage root backing-store identity",
        )
        _require_identifier(self.filesystem_type, "storage root filesystem type")
        if (
            _require_bool(
                self.created_exclusively_for_case,
                "storage root case exclusivity",
            )
            is not True
        ):
            _fail("storage root must be created exclusively for one case")
        if _require_bool(self.empty_at_fresh_boundary, "storage root fresh emptiness") is not True:
            _fail("storage root must be empty at its fresh boundary")
        if _require_bool(self.writable, "storage root writable") is not True:
            _fail("bound storage root must be writable")
        _require_bool(self.includes_overlay_copy_up, "storage root overlay copy-up")
        if self.includes_overlay_copy_up and field_name != "max_disk_peak_bytes":
            _fail("overlay copy-up can only enter the disk storage field")

    def to_dict(self) -> dict[str, Any]:
        return {
            "root_id": self.root_id,
            "field_name": self.field_name,
            "absolute_path": self.absolute_path,
            "mount_identity_sha256": self.mount_identity_sha256,
            "backing_store_identity_sha256": self.backing_store_identity_sha256,
            "filesystem_type": self.filesystem_type,
            "created_exclusively_for_case": self.created_exclusively_for_case,
            "empty_at_fresh_boundary": self.empty_at_fresh_boundary,
            "writable": self.writable,
            "includes_overlay_copy_up": self.includes_overlay_copy_up,
        }


def _validate_storage_roots(
    value: object,
    container: ContainerStorageIdentityV1,
) -> tuple[StorageRootBindingV1, ...]:
    if type(container) is not ContainerStorageIdentityV1:
        _fail("storage-root container identity type differs")
    if type(value) is not tuple or any(type(item) is not StorageRootBindingV1 for item in value):
        _fail("storage roots must use one exact immutable tuple")
    roots = cast(tuple[StorageRootBindingV1, ...], value)
    if tuple(item.root_id for item in roots) != tuple(sorted(item.root_id for item in roots)):
        _fail("storage roots must use lexicographic root-ID order")
    if len({item.root_id for item in roots}) != len(roots):
        _fail("storage root IDs must be unique")
    if len({item.absolute_path for item in roots}) != len(roots):
        _fail("storage root paths must be unique")
    for index, left in enumerate(roots):
        left_prefix = left.absolute_path.rstrip("/") + "/"
        for right in roots[index + 1 :]:
            right_prefix = right.absolute_path.rstrip("/") + "/"
            if right.absolute_path.startswith(left_prefix) or left.absolute_path.startswith(
                right_prefix
            ):
                _fail("storage roots cannot overlap by ancestry")
    copy_up_roots = tuple(item for item in roots if item.includes_overlay_copy_up)
    if container.rootfs_copy_up_policy == "bound_writable_root":
        if len(copy_up_roots) != 1:
            _fail("writable rootfs copy-up requires one exact bound disk root")
    elif copy_up_roots:
        _fail("copy-up-disabled rootfs cannot name a copy-up storage root")
    return roots


def storage_root_inventory_sha256(
    roots: tuple[StorageRootBindingV1, ...],
    container: ContainerStorageIdentityV1,
) -> str:
    """Return a detached digest of one exact ordered storage-root inventory."""

    exact = _validate_storage_roots(roots, container)
    return _sha256(
        _canonical_json(
            {"storage_roots": [item.to_dict() for item in exact]},
            newline=False,
        )
    )


def storage_field_root_inventory_sha256(
    field_name: str,
    root_ids: tuple[str, ...],
    roots: tuple[StorageRootBindingV1, ...],
) -> str:
    """Return the exact field-scoped root-union digest used by peak evidence."""

    exact_field_name = _require_identifier(field_name, "field-scoped storage-root field")
    if exact_field_name not in STORAGE_RESOURCE_FIELDS:
        _fail("field-scoped storage-root inventory field differs")
    if type(roots) is not tuple or any(type(item) is not StorageRootBindingV1 for item in roots):
        _fail("field-scoped storage roots must use one exact immutable tuple")
    if type(root_ids) is not tuple or any(type(item) is not str for item in root_ids):
        _fail("field-scoped storage-root IDs must use one exact tuple")
    for root_id in root_ids:
        _require_identifier(root_id, "field-scoped storage-root ID")
    expected_roots = tuple(item for item in roots if item.field_name == exact_field_name)
    if root_ids != tuple(item.root_id for item in expected_roots):
        _fail("field-scoped storage-root IDs differ from the exact root inventory")
    return _sha256(
        _canonical_json(
            {
                "field_name": exact_field_name,
                "root_ids": list(root_ids),
                "roots": [item.to_dict() for item in expected_roots],
            },
            newline=False,
        )
    )


@dataclass(frozen=True, slots=True)
class StorageFieldPolicyV1:
    """Pre-GO mode, roots, semantics, and anti-substitution policy for one field."""

    field_name: Literal["max_temporary_peak_bytes", "max_disk_peak_bytes"]
    field_position: int
    measurement_mode: Literal[
        "event_complete_kernel_filesystem_accounting",
        "immutable_non_bypass_pre_go_hard_quota_enforcement",
    ]
    value_semantics: Literal[
        "exact_observation",
        "conservative_enforced_upper_bound",
    ]
    lifetime_scope: str
    root_ids: tuple[str, ...]
    measurement_policy_body_sha256: str
    event_accounting_policy_body_sha256: str | None
    quota_enforcement_policy_body_sha256: str | None
    hard_limit_bytes: int | None
    committed_before_go: bool
    polling_or_sampling_sufficient: bool
    du_snapshot_sufficient: bool
    container_layer_size_sufficient: bool
    missing_value_defaults_to_zero: bool

    def __post_init__(self) -> None:
        field_name = _require_identifier(self.field_name, "storage field policy name")
        if field_name not in STORAGE_RESOURCE_FIELDS:
            _fail("storage field policy name differs")
        expected_position = STORAGE_FIELD_POSITIONS[field_name]
        _require_int(
            self.field_position,
            "storage field policy position",
            minimum=expected_position,
            maximum=expected_position,
        )
        lifetime_scope = _require_identifier(
            self.lifetime_scope,
            "storage field policy lifetime scope",
        )
        if lifetime_scope != STORAGE_LIFETIME_SCOPE:
            _fail("storage field policy lifetime scope differs")
        if type(self.root_ids) is not tuple or any(type(item) is not str for item in self.root_ids):
            _fail("storage field root IDs must use one exact tuple")
        for item in self.root_ids:
            _require_identifier(item, "storage field root ID")
        if self.root_ids != tuple(sorted(self.root_ids)) or len(set(self.root_ids)) != len(
            self.root_ids
        ):
            _fail("storage field root IDs must be unique and lexicographically ordered")
        _require_sha256(self.measurement_policy_body_sha256, "measurement policy body")
        measurement_mode = _require_identifier(
            self.measurement_mode,
            "storage field policy measurement mode",
        )
        value_semantics = _require_identifier(
            self.value_semantics,
            "storage field policy value semantics",
        )
        if measurement_mode == EVENT_COMPLETE_ACCOUNTING_MODE:
            if value_semantics != EXACT_OBSERVATION:
                _fail("event-complete mode must report exact observations")
            _require_sha256(
                self.event_accounting_policy_body_sha256,
                "event accounting policy body",
            )
            if self.quota_enforcement_policy_body_sha256 is not None:
                _fail("event-complete policy cannot bind a quota policy")
            if self.hard_limit_bytes is not None:
                _fail("event-complete policy cannot declare a hard quota value")
        elif measurement_mode == HARD_QUOTA_ENFORCEMENT_MODE:
            if value_semantics != CONSERVATIVE_ENFORCED_UPPER_BOUND:
                _fail("hard-quota mode must report conservative enforced upper bounds")
            if not self.root_ids:
                _fail("hard-quota mode requires at least one bound storage root")
            _require_sha256(
                self.quota_enforcement_policy_body_sha256,
                "quota enforcement policy body",
            )
            if self.event_accounting_policy_body_sha256 is not None:
                _fail("hard-quota policy cannot bind an event-accounting policy")
            _require_int(
                self.hard_limit_bytes,
                "storage hard limit",
                minimum=1,
            )
        else:
            _fail("storage field policy measurement mode differs")
        if _require_bool(self.committed_before_go, "policy committed before GO") is not True:
            _fail("storage field policy must be committed before GO")
        for value, label in (
            (self.polling_or_sampling_sufficient, "polling or sampling sufficient"),
            (self.du_snapshot_sufficient, "du snapshot sufficient"),
            (self.container_layer_size_sufficient, "container layer size sufficient"),
            (self.missing_value_defaults_to_zero, "missing value defaults to zero"),
        ):
            if _require_bool(value, label) is not False:
                _fail(f"{label} must remain false")

    def to_dict(self) -> dict[str, Any]:
        return {
            "field_name": self.field_name,
            "field_position": self.field_position,
            "measurement_mode": self.measurement_mode,
            "value_semantics": self.value_semantics,
            "lifetime_scope": self.lifetime_scope,
            "root_ids": list(self.root_ids),
            "measurement_policy_body_sha256": self.measurement_policy_body_sha256,
            "event_accounting_policy_body_sha256": (self.event_accounting_policy_body_sha256),
            "quota_enforcement_policy_body_sha256": (self.quota_enforcement_policy_body_sha256),
            "hard_limit_bytes": self.hard_limit_bytes,
            "committed_before_go": self.committed_before_go,
            "polling_or_sampling_sufficient": self.polling_or_sampling_sufficient,
            "du_snapshot_sufficient": self.du_snapshot_sufficient,
            "container_layer_size_sufficient": self.container_layer_size_sufficient,
            "missing_value_defaults_to_zero": self.missing_value_defaults_to_zero,
        }


def _validate_field_policies(
    value: object,
    roots: tuple[StorageRootBindingV1, ...],
) -> tuple[StorageFieldPolicyV1, ...]:
    if type(roots) is not tuple or any(type(item) is not StorageRootBindingV1 for item in roots):
        _fail("storage policy roots must use one exact immutable tuple")
    if type(value) is not tuple or any(type(item) is not StorageFieldPolicyV1 for item in value):
        _fail("storage field policies must use one exact immutable tuple")
    policies = cast(tuple[StorageFieldPolicyV1, ...], value)
    if tuple(item.field_name for item in policies) != STORAGE_RESOURCE_FIELDS:
        _fail("storage field policies must use exact fields 24 and 25 in order")
    for policy in policies:
        expected_ids = tuple(item.root_id for item in roots if item.field_name == policy.field_name)
        if policy.root_ids != expected_ids:
            _fail(f"storage root projection differs for {policy.field_name}")
    return policies


def storage_field_policy_inventory_sha256(
    policies: tuple[StorageFieldPolicyV1, ...],
    roots: tuple[StorageRootBindingV1, ...],
) -> str:
    """Return a detached digest of the exact two-field pre-GO policy."""

    exact = _validate_field_policies(policies, roots)
    return _sha256(
        _canonical_json(
            {"field_policy": [item.to_dict() for item in exact]},
            newline=False,
        )
    )


@dataclass(frozen=True, slots=True)
class CollectionBoundaryChainV1:
    """Exact request-to-intent-to-READY-to-anchor-to-GO identity projections."""

    host_case_request: ArtifactIdentityV1
    host_case_intent: ArtifactIdentityV1
    host_ready: ArtifactIdentityV1
    host_observer_anchor: ArtifactIdentityV1
    host_go: ArtifactIdentityV1
    handshake_chain_body_sha256: str
    host_intent_request_file_sha256: str
    host_intent_request_body_sha256: str
    ready_host_intent_file_sha256: str
    ready_host_intent_body_sha256: str
    observer_anchor_ready_file_sha256: str
    observer_anchor_ready_body_sha256: str
    request_storage_intent_file_sha256: str
    request_storage_intent_body_sha256: str
    ready_storage_intent_file_sha256: str
    ready_storage_intent_body_sha256: str
    go_ready_file_sha256: str
    go_ready_body_sha256: str
    go_observer_anchor_file_sha256: str
    go_observer_anchor_body_sha256: str

    def __post_init__(self) -> None:
        artifacts = (
            _require_artifact_schema(
                self.host_case_request,
                HOST_CASE_REQUEST_SCHEMA_VERSION,
                "host case request",
            ),
            _require_artifact_schema(
                self.host_case_intent,
                HOST_CASE_INTENT_SCHEMA_VERSION,
                "host case intent",
            ),
            _require_artifact_schema(self.host_ready, HOST_READY_SCHEMA_VERSION, "host READY"),
            _require_artifact_schema(
                self.host_observer_anchor,
                HOST_OBSERVER_ANCHOR_SCHEMA_VERSION,
                "host observer anchor",
            ),
            _require_artifact_schema(self.host_go, HOST_GO_SCHEMA_VERSION, "host GO"),
        )
        if len(set(artifacts)) != len(artifacts):
            _fail("host handshake artifact identities cannot alias")
        _require_sha256(self.handshake_chain_body_sha256, "host handshake chain body")
        for value, label in (
            (self.host_intent_request_file_sha256, "host-intent request file projection"),
            (self.host_intent_request_body_sha256, "host-intent request body projection"),
            (self.ready_host_intent_file_sha256, "READY host-intent file projection"),
            (self.ready_host_intent_body_sha256, "READY host-intent body projection"),
            (
                self.observer_anchor_ready_file_sha256,
                "observer-anchor READY file projection",
            ),
            (
                self.observer_anchor_ready_body_sha256,
                "observer-anchor READY body projection",
            ),
            (
                self.request_storage_intent_file_sha256,
                "request storage-intent file projection",
            ),
            (
                self.request_storage_intent_body_sha256,
                "request storage-intent body projection",
            ),
            (
                self.ready_storage_intent_file_sha256,
                "READY storage-intent file projection",
            ),
            (
                self.ready_storage_intent_body_sha256,
                "READY storage-intent body projection",
            ),
            (self.go_ready_file_sha256, "GO READY file projection"),
            (self.go_ready_body_sha256, "GO READY body projection"),
            (
                self.go_observer_anchor_file_sha256,
                "GO observer-anchor file projection",
            ),
            (
                self.go_observer_anchor_body_sha256,
                "GO observer-anchor body projection",
            ),
        ):
            _require_sha256(value, label)
        if (
            self.request_storage_intent_file_sha256 != self.ready_storage_intent_file_sha256
            or self.request_storage_intent_body_sha256 != self.ready_storage_intent_body_sha256
        ):
            _fail("request and READY storage-intent identities differ")
        if (
            self.host_intent_request_file_sha256 != self.host_case_request.file_sha256
            or self.host_intent_request_body_sha256 != self.host_case_request.body_sha256
            or self.ready_host_intent_file_sha256 != self.host_case_intent.file_sha256
            or self.ready_host_intent_body_sha256 != self.host_case_intent.body_sha256
            or self.observer_anchor_ready_file_sha256 != self.host_ready.file_sha256
            or self.observer_anchor_ready_body_sha256 != self.host_ready.body_sha256
            or self.go_ready_file_sha256 != self.host_ready.file_sha256
            or self.go_ready_body_sha256 != self.host_ready.body_sha256
            or self.go_observer_anchor_file_sha256 != self.host_observer_anchor.file_sha256
            or self.go_observer_anchor_body_sha256 != self.host_observer_anchor.body_sha256
        ):
            _fail("host request-to-intent-to-READY-to-anchor-to-GO projections differ")
        expected_chain_body = _body_sha256(self._projection_dict())
        if not hmac.compare_digest(self.handshake_chain_body_sha256, expected_chain_body):
            _fail("host handshake chain body digest does not replay canonically")

    def _projection_dict(self) -> dict[str, Any]:
        return {
            "host_case_request": self.host_case_request.to_dict(),
            "host_case_intent": self.host_case_intent.to_dict(),
            "host_ready": self.host_ready.to_dict(),
            "host_observer_anchor": self.host_observer_anchor.to_dict(),
            "host_go": self.host_go.to_dict(),
            "host_intent_request_file_sha256": self.host_intent_request_file_sha256,
            "host_intent_request_body_sha256": self.host_intent_request_body_sha256,
            "ready_host_intent_file_sha256": self.ready_host_intent_file_sha256,
            "ready_host_intent_body_sha256": self.ready_host_intent_body_sha256,
            "observer_anchor_ready_file_sha256": (self.observer_anchor_ready_file_sha256),
            "observer_anchor_ready_body_sha256": (self.observer_anchor_ready_body_sha256),
            "request_storage_intent_file_sha256": (self.request_storage_intent_file_sha256),
            "request_storage_intent_body_sha256": (self.request_storage_intent_body_sha256),
            "ready_storage_intent_file_sha256": (self.ready_storage_intent_file_sha256),
            "ready_storage_intent_body_sha256": (self.ready_storage_intent_body_sha256),
            "go_ready_file_sha256": self.go_ready_file_sha256,
            "go_ready_body_sha256": self.go_ready_body_sha256,
            "go_observer_anchor_file_sha256": self.go_observer_anchor_file_sha256,
            "go_observer_anchor_body_sha256": self.go_observer_anchor_body_sha256,
        }

    def to_dict(self) -> dict[str, Any]:
        result = self._projection_dict()
        result["handshake_chain_body_sha256"] = self.handshake_chain_body_sha256
        return result


def _validate_collection_boundary_projection(
    chain: object,
    measurement_intent: ArtifactIdentityV1,
) -> CollectionBoundaryChainV1:
    if type(chain) is not CollectionBoundaryChainV1:
        _fail("storage collection-boundary chain type differs")
    exact = chain
    exact_measurement_intent = _require_artifact_schema(
        measurement_intent,
        QUALIFICATION_STORAGE_BOUNDARY_INTENT_SCHEMA_VERSION,
        "storage collection-boundary measurement intent",
    )
    if (
        exact.request_storage_intent_file_sha256 != exact_measurement_intent.file_sha256
        or exact.request_storage_intent_body_sha256 != exact_measurement_intent.body_sha256
        or exact.ready_storage_intent_file_sha256 != exact_measurement_intent.file_sha256
        or exact.ready_storage_intent_body_sha256 != exact_measurement_intent.body_sha256
    ):
        _fail("request and READY must bind the exact storage measurement intent")
    return exact


@dataclass(frozen=True, slots=True)
class WritableSurfaceEvidenceV1:
    """Complete no-alternate-writable-mount/path proof at the seal boundary."""

    writable_mount_inventory_body_sha256: str
    writable_path_inventory_body_sha256: str
    bound_root_inventory_sha256: str
    proof_body_sha256: str
    all_writable_mounts_bound: bool
    all_writable_paths_beneath_bound_roots: bool
    unbound_writable_mount_count: int
    unbound_writable_path_count: int
    alternate_writable_mounts_possible: bool
    alternate_writable_paths_possible: bool
    rootfs_copy_up_bound_or_impossible: bool
    deleted_open_files_accounted_or_impossible: bool
    anonymous_files_accounted_or_impossible: bool
    memory_backed_files_accounted_or_impossible: bool
    mount_namespace_mutation_disabled: bool
    descendant_mount_creation_disabled: bool
    host_path_write_escape_disabled: bool
    device_write_escape_disabled: bool
    network_storage_write_escape_disabled: bool
    inherited_writable_fd_escape_disabled: bool

    def __post_init__(self) -> None:
        for value, label in (
            (self.writable_mount_inventory_body_sha256, "writable mount inventory body"),
            (self.writable_path_inventory_body_sha256, "writable path inventory body"),
            (self.bound_root_inventory_sha256, "bound root inventory"),
            (self.proof_body_sha256, "writable surface proof body"),
        ):
            _require_sha256(value, label)
        for flag, label in (
            (self.all_writable_mounts_bound, "all writable mounts bound"),
            (
                self.all_writable_paths_beneath_bound_roots,
                "all writable paths beneath bound roots",
            ),
            (self.rootfs_copy_up_bound_or_impossible, "rootfs copy-up bound or impossible"),
            (
                self.deleted_open_files_accounted_or_impossible,
                "deleted open files accounted or impossible",
            ),
            (
                self.anonymous_files_accounted_or_impossible,
                "anonymous files accounted or impossible",
            ),
            (
                self.memory_backed_files_accounted_or_impossible,
                "memory-backed files accounted or impossible",
            ),
            (self.mount_namespace_mutation_disabled, "mount namespace mutation disabled"),
            (self.descendant_mount_creation_disabled, "descendant mount creation disabled"),
            (self.host_path_write_escape_disabled, "host-path write escape disabled"),
            (self.device_write_escape_disabled, "device write escape disabled"),
            (self.network_storage_write_escape_disabled, "network storage escape disabled"),
            (
                self.inherited_writable_fd_escape_disabled,
                "inherited writable-FD escape disabled",
            ),
        ):
            if _require_bool(flag, label) is not True:
                _fail(f"{label} must be exact true")
        for flag, label in (
            (self.alternate_writable_mounts_possible, "alternate writable mounts possible"),
            (self.alternate_writable_paths_possible, "alternate writable paths possible"),
        ):
            if _require_bool(flag, label) is not False:
                _fail(f"{label} must be exact false")
        _require_int(
            self.unbound_writable_mount_count,
            "unbound writable mount count",
            maximum=0,
        )
        _require_int(
            self.unbound_writable_path_count,
            "unbound writable path count",
            maximum=0,
        )

    def to_dict(self) -> dict[str, Any]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}


def _publication_reload_validation_identity_hashes(
    *,
    publication_commitment_wrapper_file_sha256: str,
    publication_commitment_wrapper_body_sha256: str,
    expected_reload_observation_sha256: str,
    actual_reload_observation_sha256: str,
    reload_performed: bool,
    reload_read_only: bool,
) -> tuple[str, str]:
    body = {
        "schema_version": QUALIFICATION_PUBLICATION_RELOAD_VALIDATION_SCHEMA_VERSION,
        "publication_commitment_wrapper_file_sha256": (publication_commitment_wrapper_file_sha256),
        "publication_commitment_wrapper_body_sha256": (publication_commitment_wrapper_body_sha256),
        "expected_reload_observation_sha256": expected_reload_observation_sha256,
        "actual_reload_observation_sha256": actual_reload_observation_sha256,
        "reload_performed": reload_performed,
        "reload_read_only": reload_read_only,
    }
    body_sha256 = _body_sha256(body)
    full_file = dict(body)
    full_file["reload_validation_body_sha256"] = body_sha256
    return body_sha256, _sha256(_canonical_json(full_file))


@dataclass(frozen=True, slots=True)
class WriteQuiescenceSealProofV1:
    """Irreversible preterminal seal making the recorded peak whole-lifetime final."""

    publication_commitment: ArtifactIdentityV1
    write_quiescence_seal: ArtifactIdentityV1
    terminal_relay_preseal_attestation: ArtifactIdentityV1
    nonstorage_channel_readiness_attestation: ArtifactIdentityV1
    publication_reload_validation: ArtifactIdentityV1
    seal_architecture_body_sha256: str
    publication_commitment_wrapper_file_sha256: str
    publication_commitment_wrapper_body_sha256: str
    expected_reload_observation_sha256: str
    actual_reload_observation_sha256: str
    sealed_mount_inventory_body_sha256: str
    sealed_path_inventory_body_sha256: str
    write_quiescence_seal_body_sha256: str
    publication_committed_before_seal: bool
    reload_validated_before_seal: bool
    reload_performed: bool
    reload_read_only: bool
    write_quiescence_irreversible: bool
    container_writes_disabled: bool
    descendant_writes_disabled: bool
    later_allocation_possible: bool
    later_copy_up_possible: bool
    later_peak_increase_possible: bool
    terminal_transport_outside_measured_storage: bool
    terminal_transport_can_allocate_measured_storage: bool
    worker_exit_observed: bool
    irreversible_seccomp_fd_closure_observed: bool
    worker_has_measured_writable_namespace: bool
    worker_has_measured_writable_fd: bool
    only_trusted_terminal_relay_retains_terminal_transport_capability: bool
    terminal_relay_input_policy_restricted_to_receipt_identity: bool
    nonstorage_channel_ready_for_post_receipt_terminal: bool
    container_log_driver_none: bool
    container_logging_write_possible: bool
    no_later_writer_exists: bool
    teardown_deletion_only: bool
    teardown_can_increase_measured_usage: bool
    peak_is_whole_fresh_case_lifetime_peak: bool

    def __post_init__(self) -> None:
        _require_artifact_schema(
            self.publication_commitment,
            NORMALIZED_PUBLICATION_COMMITMENT_WRAPPER_SCHEMA_VERSION,
            "publication commitment",
        )
        _require_artifact_schema(
            self.write_quiescence_seal,
            QUALIFICATION_STORAGE_WRITE_SEAL_SCHEMA_VERSION,
            "write-quiescence seal",
        )
        _require_artifact_schema(
            self.terminal_relay_preseal_attestation,
            QUALIFICATION_STORAGE_TERMINAL_RELAY_PRESEAL_ATTESTATION_SCHEMA_VERSION,
            "terminal relay preseal attestation",
        )
        _require_artifact_schema(
            self.nonstorage_channel_readiness_attestation,
            QUALIFICATION_STORAGE_NONSTORAGE_CHANNEL_READINESS_ATTESTATION_SCHEMA_VERSION,
            "nonstorage channel readiness attestation",
        )
        _require_artifact_schema(
            self.publication_reload_validation,
            QUALIFICATION_PUBLICATION_RELOAD_VALIDATION_SCHEMA_VERSION,
            "publication reload validation",
        )
        artifacts = (
            self.publication_commitment,
            self.write_quiescence_seal,
            self.terminal_relay_preseal_attestation,
            self.nonstorage_channel_readiness_attestation,
            self.publication_reload_validation,
        )
        if len(set(artifacts)) != len(artifacts):
            _fail("publication, write-seal, relay, and channel artifacts cannot alias")
        for value, label in (
            (self.seal_architecture_body_sha256, "seal architecture body"),
            (
                self.publication_commitment_wrapper_file_sha256,
                "reload-validation wrapper file projection",
            ),
            (
                self.publication_commitment_wrapper_body_sha256,
                "reload-validation wrapper body projection",
            ),
            (
                self.expected_reload_observation_sha256,
                "reload-validation expected observation",
            ),
            (
                self.actual_reload_observation_sha256,
                "reload-validation actual observation",
            ),
            (self.sealed_mount_inventory_body_sha256, "sealed mount inventory body"),
            (self.sealed_path_inventory_body_sha256, "sealed path inventory body"),
            (self.write_quiescence_seal_body_sha256, "write-quiescence seal body"),
        ):
            _require_sha256(value, label)
        if self.write_quiescence_seal_body_sha256 != self.write_quiescence_seal.body_sha256:
            _fail("write-seal proof body must equal the write-seal artifact body identity")
        if (
            self.publication_commitment_wrapper_file_sha256
            != self.publication_commitment.file_sha256
            or self.publication_commitment_wrapper_body_sha256
            != self.publication_commitment.body_sha256
        ):
            _fail("reload validation must bind the exact publication wrapper identity")
        if self.actual_reload_observation_sha256 != self.expected_reload_observation_sha256:
            _fail("actual reload observation differs from the wrapper commitment")
        for flag, label in (
            (self.publication_committed_before_seal, "publication committed before seal"),
            (self.reload_validated_before_seal, "publication reload validated before seal"),
            (self.reload_performed, "publication reload performed"),
            (self.reload_read_only, "publication reload read-only"),
            (self.write_quiescence_irreversible, "write quiescence irreversible"),
            (self.container_writes_disabled, "container writes disabled"),
            (self.descendant_writes_disabled, "descendant writes disabled"),
            (
                self.terminal_transport_outside_measured_storage,
                "terminal transport outside measured storage",
            ),
            (
                self.only_trusted_terminal_relay_retains_terminal_transport_capability,
                "only trusted terminal relay retains terminal transport capability",
            ),
            (
                self.terminal_relay_input_policy_restricted_to_receipt_identity,
                "terminal relay input policy restricted to receipt identity",
            ),
            (
                self.nonstorage_channel_ready_for_post_receipt_terminal,
                "nonstorage channel ready for post-receipt terminal",
            ),
            (self.container_log_driver_none, "container log driver none"),
            (self.no_later_writer_exists, "no later writer exists"),
            (self.teardown_deletion_only, "teardown deletion-only"),
            (
                self.peak_is_whole_fresh_case_lifetime_peak,
                "peak is whole fresh-case lifetime peak",
            ),
        ):
            if _require_bool(flag, label) is not True:
                _fail(f"{label} must be exact true")
        for flag, label in (
            (self.later_allocation_possible, "later allocation possible"),
            (self.later_copy_up_possible, "later copy-up possible"),
            (self.later_peak_increase_possible, "later peak increase possible"),
            (
                self.terminal_transport_can_allocate_measured_storage,
                "terminal transport can allocate measured storage",
            ),
            (
                self.worker_has_measured_writable_namespace,
                "worker has measured writable namespace",
            ),
            (self.worker_has_measured_writable_fd, "worker has measured writable FD"),
            (self.container_logging_write_possible, "container logging write possible"),
            (
                self.teardown_can_increase_measured_usage,
                "teardown can increase measured usage",
            ),
        ):
            if _require_bool(flag, label) is not False:
                _fail(f"{label} must be exact false")
        worker_exit = _require_bool(self.worker_exit_observed, "worker exit observed")
        seccomp_closure = _require_bool(
            self.irreversible_seccomp_fd_closure_observed,
            "irreversible seccomp and FD closure observed",
        )
        if worker_exit is seccomp_closure:
            _fail("write seal requires exactly one worker-quiescence mechanism")
        expected_reload_body, expected_reload_file = _publication_reload_validation_identity_hashes(
            publication_commitment_wrapper_file_sha256=(
                self.publication_commitment_wrapper_file_sha256
            ),
            publication_commitment_wrapper_body_sha256=(
                self.publication_commitment_wrapper_body_sha256
            ),
            expected_reload_observation_sha256=(self.expected_reload_observation_sha256),
            actual_reload_observation_sha256=self.actual_reload_observation_sha256,
            reload_performed=self.reload_performed,
            reload_read_only=self.reload_read_only,
        )
        if not hmac.compare_digest(
            self.publication_reload_validation.body_sha256,
            expected_reload_body,
        ) or not hmac.compare_digest(
            self.publication_reload_validation.file_sha256,
            expected_reload_file,
        ):
            _fail("publication reload-validation artifact file or body does not replay")

    def to_dict(self) -> dict[str, Any]:
        return {
            "publication_commitment": self.publication_commitment.to_dict(),
            "write_quiescence_seal": self.write_quiescence_seal.to_dict(),
            "terminal_relay_preseal_attestation": (
                self.terminal_relay_preseal_attestation.to_dict()
            ),
            "nonstorage_channel_readiness_attestation": (
                self.nonstorage_channel_readiness_attestation.to_dict()
            ),
            "publication_reload_validation": self.publication_reload_validation.to_dict(),
            "seal_architecture_body_sha256": self.seal_architecture_body_sha256,
            "publication_commitment_wrapper_file_sha256": (
                self.publication_commitment_wrapper_file_sha256
            ),
            "publication_commitment_wrapper_body_sha256": (
                self.publication_commitment_wrapper_body_sha256
            ),
            "expected_reload_observation_sha256": self.expected_reload_observation_sha256,
            "actual_reload_observation_sha256": self.actual_reload_observation_sha256,
            "sealed_mount_inventory_body_sha256": (self.sealed_mount_inventory_body_sha256),
            "sealed_path_inventory_body_sha256": self.sealed_path_inventory_body_sha256,
            "write_quiescence_seal_body_sha256": (self.write_quiescence_seal_body_sha256),
            "publication_committed_before_seal": self.publication_committed_before_seal,
            "reload_validated_before_seal": self.reload_validated_before_seal,
            "reload_performed": self.reload_performed,
            "reload_read_only": self.reload_read_only,
            "write_quiescence_irreversible": self.write_quiescence_irreversible,
            "container_writes_disabled": self.container_writes_disabled,
            "descendant_writes_disabled": self.descendant_writes_disabled,
            "later_allocation_possible": self.later_allocation_possible,
            "later_copy_up_possible": self.later_copy_up_possible,
            "later_peak_increase_possible": self.later_peak_increase_possible,
            "terminal_transport_outside_measured_storage": (
                self.terminal_transport_outside_measured_storage
            ),
            "terminal_transport_can_allocate_measured_storage": (
                self.terminal_transport_can_allocate_measured_storage
            ),
            "worker_exit_observed": self.worker_exit_observed,
            "irreversible_seccomp_fd_closure_observed": (
                self.irreversible_seccomp_fd_closure_observed
            ),
            "worker_has_measured_writable_namespace": (self.worker_has_measured_writable_namespace),
            "worker_has_measured_writable_fd": self.worker_has_measured_writable_fd,
            "only_trusted_terminal_relay_retains_terminal_transport_capability": (
                self.only_trusted_terminal_relay_retains_terminal_transport_capability
            ),
            "terminal_relay_input_policy_restricted_to_receipt_identity": (
                self.terminal_relay_input_policy_restricted_to_receipt_identity
            ),
            "nonstorage_channel_ready_for_post_receipt_terminal": (
                self.nonstorage_channel_ready_for_post_receipt_terminal
            ),
            "container_log_driver_none": self.container_log_driver_none,
            "container_logging_write_possible": self.container_logging_write_possible,
            "no_later_writer_exists": self.no_later_writer_exists,
            "teardown_deletion_only": self.teardown_deletion_only,
            "teardown_can_increase_measured_usage": (self.teardown_can_increase_measured_usage),
            "peak_is_whole_fresh_case_lifetime_peak": (self.peak_is_whole_fresh_case_lifetime_peak),
        }


@dataclass(frozen=True, slots=True)
class EventCompleteAccountingEvidenceV1:
    """Lossless event accounting from the fresh boundary through the write seal."""

    event_stream_body_sha256: str
    field_name: Literal["max_temporary_peak_bytes", "max_disk_peak_bytes"]
    field_root_ids: tuple[str, ...]
    field_root_inventory_sha256: str
    replayed_peak_bytes: int
    replayed_peak_is_simultaneous_aggregate_union_high_water: bool
    root_event_inventory_body_sha256: str
    writable_mount_inventory_body_sha256: str
    writable_path_inventory_body_sha256: str
    write_quiescence_seal_file_sha256: str
    write_quiescence_seal_body_sha256: str
    fresh_boundary_sequence: int
    seal_boundary_sequence: int
    accounting_started_before_go: bool
    accounting_closed_at_irreversible_seal: bool
    all_bound_roots_covered: bool
    allocation_and_deallocation_events_complete: bool
    filesystem_copy_up_events_complete: bool
    deleted_open_file_events_complete: bool
    event_loss_count: int
    event_overflow_count: int
    polling_or_sampling_used: bool
    du_snapshot_used: bool
    container_layer_size_used: bool
    exact_high_water_replayed: bool

    def __post_init__(self) -> None:
        field_name = _require_identifier(self.field_name, "event evidence storage field")
        if field_name not in STORAGE_RESOURCE_FIELDS:
            _fail("event evidence storage field differs")
        if type(self.field_root_ids) is not tuple or any(
            type(item) is not str for item in self.field_root_ids
        ):
            _fail("event evidence field-root IDs must use one exact tuple")
        for root_id in self.field_root_ids:
            _require_identifier(root_id, "event evidence field-root ID")
        if self.field_root_ids != tuple(sorted(self.field_root_ids)) or len(
            set(self.field_root_ids)
        ) != len(self.field_root_ids):
            _fail("event evidence field-root IDs must be unique and ordered")
        _require_int(self.replayed_peak_bytes, "event replayed peak bytes")
        if (
            _require_bool(
                self.replayed_peak_is_simultaneous_aggregate_union_high_water,
                "event replayed peak is simultaneous aggregate union high water",
            )
            is not True
        ):
            _fail("event replay must be the simultaneous aggregate field-root high water")
        for value, label in (
            (self.event_stream_body_sha256, "storage event stream body"),
            (self.field_root_inventory_sha256, "event field-root inventory"),
            (self.root_event_inventory_body_sha256, "root event inventory body"),
            (self.writable_mount_inventory_body_sha256, "event writable mount inventory"),
            (self.writable_path_inventory_body_sha256, "event writable path inventory"),
            (self.write_quiescence_seal_file_sha256, "event write-seal file"),
            (self.write_quiescence_seal_body_sha256, "event write-seal body"),
        ):
            _require_sha256(value, label)
        _require_int(
            self.fresh_boundary_sequence,
            "event accounting fresh-boundary sequence",
            maximum=0,
        )
        _require_int(
            self.seal_boundary_sequence,
            "event accounting seal sequence",
            minimum=1,
        )
        for flag, label in (
            (self.accounting_started_before_go, "event accounting started before GO"),
            (
                self.accounting_closed_at_irreversible_seal,
                "event accounting closed at irreversible seal",
            ),
            (self.all_bound_roots_covered, "event accounting covers all bound roots"),
            (
                self.allocation_and_deallocation_events_complete,
                "allocation and deallocation events complete",
            ),
            (self.filesystem_copy_up_events_complete, "filesystem copy-up events complete"),
            (self.deleted_open_file_events_complete, "deleted-open-file events complete"),
            (self.exact_high_water_replayed, "exact high water replayed"),
        ):
            if _require_bool(flag, label) is not True:
                _fail(f"{label} must be exact true")
        _require_int(self.event_loss_count, "event loss count", maximum=0)
        _require_int(self.event_overflow_count, "event overflow count", maximum=0)
        for flag, label in (
            (self.polling_or_sampling_used, "polling or sampling used"),
            (self.du_snapshot_used, "du snapshot used"),
            (self.container_layer_size_used, "container layer size used"),
        ):
            if _require_bool(flag, label) is not False:
                _fail(f"{label} must be exact false")

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            field: getattr(self, field) for field in self.__dataclass_fields__
        }
        result["field_root_ids"] = list(self.field_root_ids)
        return result


@dataclass(frozen=True, slots=True)
class QuotaEnforcementEvidenceV1:
    """Immutable, non-bypass hard-bound proof with final breach status."""

    enforcement_kind: Literal[
        "kernel_project_hard_quota",
        "dedicated_filesystem_capacity_limit",
        "tmpfs_hard_size_limit",
    ]
    hard_limit_bytes: int
    field_name: Literal["max_temporary_peak_bytes", "max_disk_peak_bytes"]
    field_root_ids: tuple[str, ...]
    field_root_inventory_sha256: str
    enforcement_receipt_body_sha256: str
    enforcement_boundary_body_sha256: str
    writable_mount_inventory_body_sha256: str
    writable_path_inventory_body_sha256: str
    storage_root_inventory_sha256: str
    write_quiescence_seal_file_sha256: str
    write_quiescence_seal_body_sha256: str
    installed_before_go: bool
    immutable_through_container_removal: bool
    non_bypass_through_container_removal: bool
    all_bound_roots_covered: bool
    hard_limit_applies_to_aggregate_union: bool
    alternate_writable_mount_count: int
    alternate_writable_path_count: int
    overlay_copy_up_outside_boundary_possible: bool
    descendant_bypass_possible: bool
    quota_breached: bool
    breach_count: int
    breach_status_final_at_seal: bool
    polling_or_sampling_used: bool
    du_snapshot_used: bool
    container_layer_size_used: bool

    def __post_init__(self) -> None:
        enforcement_kind = _require_identifier(
            self.enforcement_kind,
            "storage quota enforcement kind",
        )
        if enforcement_kind not in STORAGE_ENFORCEMENT_KINDS:
            _fail("storage quota enforcement kind differs")
        _require_int(self.hard_limit_bytes, "quota hard limit", minimum=1)
        field_name = _require_identifier(self.field_name, "quota evidence storage field")
        if field_name not in STORAGE_RESOURCE_FIELDS:
            _fail("quota evidence storage field differs")
        if type(self.field_root_ids) is not tuple or any(
            type(item) is not str for item in self.field_root_ids
        ):
            _fail("quota evidence field-root IDs must use one exact tuple")
        for root_id in self.field_root_ids:
            _require_identifier(root_id, "quota evidence field-root ID")
        if (
            not self.field_root_ids
            or self.field_root_ids != tuple(sorted(self.field_root_ids))
            or len(set(self.field_root_ids)) != len(self.field_root_ids)
        ):
            _fail("quota evidence field-root IDs must be nonempty, unique, and ordered")
        for value, label in (
            (self.field_root_inventory_sha256, "quota field-root inventory"),
            (self.enforcement_receipt_body_sha256, "quota enforcement receipt body"),
            (self.enforcement_boundary_body_sha256, "quota enforcement boundary body"),
            (self.writable_mount_inventory_body_sha256, "quota writable mount inventory"),
            (self.writable_path_inventory_body_sha256, "quota writable path inventory"),
            (self.storage_root_inventory_sha256, "quota storage-root inventory"),
            (self.write_quiescence_seal_file_sha256, "quota write-seal file"),
            (self.write_quiescence_seal_body_sha256, "quota write-seal body"),
        ):
            _require_sha256(value, label)
        for flag, label in (
            (self.installed_before_go, "quota installed before GO"),
            (
                self.immutable_through_container_removal,
                "quota immutable through container removal",
            ),
            (
                self.non_bypass_through_container_removal,
                "quota non-bypass through container removal",
            ),
            (self.all_bound_roots_covered, "quota covers all bound roots"),
            (
                self.hard_limit_applies_to_aggregate_union,
                "quota hard limit applies to aggregate root union",
            ),
            (self.breach_status_final_at_seal, "quota breach status final at seal"),
        ):
            if _require_bool(flag, label) is not True:
                _fail(f"{label} must be exact true")
        _require_int(
            self.alternate_writable_mount_count,
            "quota alternate writable mount count",
            maximum=0,
        )
        _require_int(
            self.alternate_writable_path_count,
            "quota alternate writable path count",
            maximum=0,
        )
        for flag, label in (
            (
                self.overlay_copy_up_outside_boundary_possible,
                "overlay copy-up outside quota possible",
            ),
            (self.descendant_bypass_possible, "descendant quota bypass possible"),
            (self.quota_breached, "quota breached"),
            (self.polling_or_sampling_used, "quota polling or sampling used"),
            (self.du_snapshot_used, "quota du snapshot used"),
            (self.container_layer_size_used, "quota container layer size used"),
        ):
            if _require_bool(flag, label) is not False:
                _fail(f"{label} must be exact false")
        _require_int(self.breach_count, "quota breach count", maximum=0)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            field: getattr(self, field) for field in self.__dataclass_fields__
        }
        result["field_root_ids"] = list(self.field_root_ids)
        return result


@dataclass(frozen=True, slots=True)
class StructuralAbsenceEvidenceV1:
    """Exact typed proof that one storage namespace is structurally absent."""

    field_name: Literal["max_temporary_peak_bytes", "max_disk_peak_bytes"]
    absence_kind: str
    namespace_inventory_body_sha256: str
    absence_proof_body_sha256: str
    no_bound_storage_root_for_field: bool
    no_writable_mount_for_field: bool
    no_writable_path_for_field: bool
    no_overlay_copy_up_target_for_field: bool

    def __post_init__(self) -> None:
        field_name = _require_identifier(
            self.field_name,
            "structural-absence storage field",
        )
        if field_name not in STORAGE_RESOURCE_FIELDS:
            _fail("structural-absence storage field differs")
        absence_kind = _require_identifier(
            self.absence_kind,
            "structural-absence kind",
        )
        if absence_kind != STORAGE_FIELD_ZERO_ABSENCE[field_name]:
            _fail("structural-absence kind differs for its storage field")
        _require_sha256(
            self.namespace_inventory_body_sha256,
            "structural-absence namespace inventory",
        )
        _require_sha256(self.absence_proof_body_sha256, "structural-absence proof body")
        for value, label in (
            (
                self.no_bound_storage_root_for_field,
                "no bound storage root for absent field",
            ),
            (self.no_writable_mount_for_field, "no writable mount for absent field"),
            (self.no_writable_path_for_field, "no writable path for absent field"),
            (
                self.no_overlay_copy_up_target_for_field,
                "no overlay copy-up target for absent field",
            ),
        ):
            if _require_bool(value, label) is not True:
                _fail(f"{label} must be exact true")

    def to_dict(self) -> dict[str, Any]:
        return {
            "field_name": self.field_name,
            "absence_kind": self.absence_kind,
            "namespace_inventory_body_sha256": self.namespace_inventory_body_sha256,
            "absence_proof_body_sha256": self.absence_proof_body_sha256,
            "no_bound_storage_root_for_field": self.no_bound_storage_root_for_field,
            "no_writable_mount_for_field": self.no_writable_mount_for_field,
            "no_writable_path_for_field": self.no_writable_path_for_field,
            "no_overlay_copy_up_target_for_field": (self.no_overlay_copy_up_target_for_field),
        }


@dataclass(frozen=True, slots=True)
class StoragePeakMeasurementV1:
    """One exact field-24/25 result with mode-specific proof metadata."""

    field_name: Literal["max_temporary_peak_bytes", "max_disk_peak_bytes"]
    field_position: int
    observed_value: int
    value_semantics: Literal[
        "exact_observation",
        "conservative_enforced_upper_bound",
    ]
    measurement_mode: Literal[
        "event_complete_kernel_filesystem_accounting",
        "immutable_non_bypass_pre_go_hard_quota_enforcement",
    ]
    lifetime_scope: str
    measurement_basis_body_sha256: str
    structural_absence_kind: str
    event_complete_evidence: EventCompleteAccountingEvidenceV1 | None
    quota_enforcement_evidence: QuotaEnforcementEvidenceV1 | None
    structural_absence_evidence: StructuralAbsenceEvidenceV1 | None

    def __post_init__(self) -> None:
        field_name = _require_identifier(
            self.field_name,
            "storage peak measurement field",
        )
        if field_name not in STORAGE_RESOURCE_FIELDS:
            _fail("storage peak measurement field differs")
        expected_position = STORAGE_FIELD_POSITIONS[field_name]
        _require_int(
            self.field_position,
            "storage peak field position",
            minimum=expected_position,
            maximum=expected_position,
        )
        value = _require_int(self.observed_value, f"storage measurement {field_name}")
        lifetime_scope = _require_identifier(
            self.lifetime_scope,
            "storage measurement lifetime scope",
        )
        if lifetime_scope != STORAGE_LIFETIME_SCOPE:
            _fail("storage measurement lifetime scope differs")
        _require_sha256(self.measurement_basis_body_sha256, "storage measurement basis body")
        measurement_mode = _require_identifier(
            self.measurement_mode,
            "storage peak measurement mode",
        )
        value_semantics = _require_identifier(
            self.value_semantics,
            "storage peak measurement value semantics",
        )
        absence_kind = _require_identifier(
            self.structural_absence_kind,
            "storage peak structural-absence kind",
        )
        if measurement_mode == EVENT_COMPLETE_ACCOUNTING_MODE:
            if value_semantics != EXACT_OBSERVATION:
                _fail("event-complete measurement semantics differ")
            if type(self.event_complete_evidence) is not EventCompleteAccountingEvidenceV1:
                _fail("event-complete measurement requires exact event evidence")
            event_evidence = self.event_complete_evidence
            if (
                event_evidence.field_name != field_name
                or event_evidence.replayed_peak_bytes != value
            ):
                _fail("event evidence field or replayed peak differs from its measurement")
            if self.quota_enforcement_evidence is not None:
                _fail("event-complete measurement cannot carry quota evidence")
        elif measurement_mode == HARD_QUOTA_ENFORCEMENT_MODE:
            if value_semantics != CONSERVATIVE_ENFORCED_UPPER_BOUND:
                _fail("hard-quota measurement semantics differ")
            if type(self.quota_enforcement_evidence) is not QuotaEnforcementEvidenceV1:
                _fail("hard-quota measurement requires exact quota evidence")
            quota_evidence = self.quota_enforcement_evidence
            if self.event_complete_evidence is not None:
                _fail("hard-quota measurement cannot carry event evidence")
            if quota_evidence.field_name != field_name:
                _fail("quota evidence field differs from its measurement")
            if value != quota_evidence.hard_limit_bytes:
                _fail("quota upper-bound value must equal its enforced hard limit")
        else:
            _fail("storage peak measurement mode differs")
        if value == 0:
            if measurement_mode != EVENT_COMPLETE_ACCOUNTING_MODE:
                _fail("zero cannot be represented by hard-quota upper-bound mode")
            if absence_kind != STORAGE_FIELD_ZERO_ABSENCE[field_name]:
                _fail("zero storage value lacks its exact typed absence kind")
            if type(self.structural_absence_evidence) is not StructuralAbsenceEvidenceV1:
                _fail("zero storage value lacks typed structural-absence evidence")
            if (
                self.structural_absence_evidence.field_name != field_name
                or self.structural_absence_evidence.absence_kind != absence_kind
            ):
                _fail("zero structural-absence evidence projection differs")
        elif absence_kind != NOT_ABSENT:
            _fail("positive storage value must use not_absent")
        elif self.structural_absence_evidence is not None:
            _fail("positive storage value cannot carry structural-absence evidence")

    def to_dict(self) -> dict[str, Any]:
        return {
            "field_name": self.field_name,
            "field_position": self.field_position,
            "observed_value": self.observed_value,
            "value_semantics": self.value_semantics,
            "measurement_mode": self.measurement_mode,
            "lifetime_scope": self.lifetime_scope,
            "measurement_basis_body_sha256": self.measurement_basis_body_sha256,
            "structural_absence_kind": self.structural_absence_kind,
            "event_complete_evidence": (
                None
                if self.event_complete_evidence is None
                else self.event_complete_evidence.to_dict()
            ),
            "quota_enforcement_evidence": (
                None
                if self.quota_enforcement_evidence is None
                else self.quota_enforcement_evidence.to_dict()
            ),
            "structural_absence_evidence": (
                None
                if self.structural_absence_evidence is None
                else self.structural_absence_evidence.to_dict()
            ),
        }


def _validate_measurements(
    value: object,
    policies: tuple[StorageFieldPolicyV1, ...],
    roots: tuple[StorageRootBindingV1, ...],
    root_inventory_sha256: str,
    writable_surface: WritableSurfaceEvidenceV1,
    write_seal: WriteQuiescenceSealProofV1,
) -> tuple[StoragePeakMeasurementV1, ...]:
    exact_root_inventory_sha256 = _require_sha256(
        root_inventory_sha256,
        "storage measurement root inventory",
    )
    if type(policies) is not tuple or any(
        type(item) is not StorageFieldPolicyV1 for item in policies
    ):
        _fail("storage measurement policies must use one exact immutable tuple")
    if type(roots) is not tuple or any(type(item) is not StorageRootBindingV1 for item in roots):
        _fail("storage measurement roots must use one exact immutable tuple")
    if type(writable_surface) is not WritableSurfaceEvidenceV1:
        _fail("storage measurement writable-surface evidence type differs")
    if type(write_seal) is not WriteQuiescenceSealProofV1:
        _fail("storage measurement write-seal proof type differs")
    exact_policies = _validate_field_policies(policies, roots)
    if type(value) is not tuple or any(
        type(item) is not StoragePeakMeasurementV1 for item in value
    ):
        _fail("storage measurements must use one exact immutable tuple")
    fields = cast(tuple[StoragePeakMeasurementV1, ...], value)
    if tuple(item.field_name for item in fields) != STORAGE_RESOURCE_FIELDS:
        _fail("storage receipt must report exactly fields 24 and 25 in order")
    if writable_surface.bound_root_inventory_sha256 != exact_root_inventory_sha256:
        _fail("writable-surface root inventory differs from the receipt")
    if (
        write_seal.sealed_mount_inventory_body_sha256
        != writable_surface.writable_mount_inventory_body_sha256
        or write_seal.sealed_path_inventory_body_sha256
        != writable_surface.writable_path_inventory_body_sha256
    ):
        _fail("write-seal and writable-surface inventories differ")
    by_policy = {item.field_name: item for item in exact_policies}
    for measurement in fields:
        policy = by_policy[measurement.field_name]
        if (
            measurement.field_position != policy.field_position
            or measurement.measurement_mode != policy.measurement_mode
            or measurement.value_semantics != policy.value_semantics
            or measurement.lifetime_scope != policy.lifetime_scope
        ):
            _fail(f"measurement policy projection differs for {measurement.field_name}")
        field_roots = tuple(item for item in roots if item.field_name == measurement.field_name)
        if measurement.observed_value == 0 and field_roots:
            _fail("zero structural absence cannot coexist with one bound field root")
        if measurement.observed_value > 0 and not field_roots:
            _fail("positive storage value requires at least one bound field root")
        expected_field_root_ids = tuple(item.root_id for item in field_roots)
        expected_field_root_inventory = storage_field_root_inventory_sha256(
            measurement.field_name,
            expected_field_root_ids,
            roots,
        )
        if measurement.event_complete_evidence is not None:
            event = measurement.event_complete_evidence
            if (
                event.field_name != measurement.field_name
                or event.field_root_ids != policy.root_ids
                or event.field_root_ids != expected_field_root_ids
                or event.field_root_inventory_sha256 != expected_field_root_inventory
                or event.replayed_peak_bytes != measurement.observed_value
                or event.root_event_inventory_body_sha256 != exact_root_inventory_sha256
                or event.writable_mount_inventory_body_sha256
                != writable_surface.writable_mount_inventory_body_sha256
                or event.writable_path_inventory_body_sha256
                != writable_surface.writable_path_inventory_body_sha256
                or event.write_quiescence_seal_file_sha256
                != write_seal.write_quiescence_seal.file_sha256
                or event.write_quiescence_seal_body_sha256
                != write_seal.write_quiescence_seal.body_sha256
            ):
                _fail("event evidence field, value, roots, or write-seal identity differ")
        if measurement.quota_enforcement_evidence is not None:
            quota = measurement.quota_enforcement_evidence
            if (
                quota.field_name != measurement.field_name
                or quota.field_root_ids != policy.root_ids
                or quota.field_root_ids != expected_field_root_ids
                or quota.field_root_inventory_sha256 != expected_field_root_inventory
                or quota.hard_limit_bytes != policy.hard_limit_bytes
                or quota.writable_mount_inventory_body_sha256
                != writable_surface.writable_mount_inventory_body_sha256
                or quota.writable_path_inventory_body_sha256
                != writable_surface.writable_path_inventory_body_sha256
                or quota.storage_root_inventory_sha256 != exact_root_inventory_sha256
                or quota.write_quiescence_seal_file_sha256
                != write_seal.write_quiescence_seal.file_sha256
                or quota.write_quiescence_seal_body_sha256
                != write_seal.write_quiescence_seal.body_sha256
            ):
                _fail("quota evidence differs from its policy, roots, or write seal")
        if measurement.structural_absence_evidence is not None:
            absence = measurement.structural_absence_evidence
            if (
                absence.namespace_inventory_body_sha256
                != writable_surface.writable_path_inventory_body_sha256
            ):
                _fail("structural absence namespace inventory differs")
    return fields


def storage_measurement_inventory_sha256(
    fields: tuple[StoragePeakMeasurementV1, ...],
    policies: tuple[StorageFieldPolicyV1, ...],
    roots: tuple[StorageRootBindingV1, ...],
    root_inventory_sha256: str,
    writable_surface: WritableSurfaceEvidenceV1,
    write_seal: WriteQuiescenceSealProofV1,
) -> str:
    """Return a detached digest of exact fields 24 and 25 after full validation."""

    exact = _validate_measurements(
        fields,
        policies,
        roots,
        root_inventory_sha256,
        writable_surface,
        write_seal,
    )
    return _sha256(
        _canonical_json(
            {"fields": [item.to_dict() for item in exact]},
            newline=False,
        )
    )


def _validate_common_identity(
    *,
    schema_version: object,
    expected_schema_version: str,
    campaign_spine_sha256: object,
    case_spine_sha256: object,
    case_ordinal: object,
    candidate_id: object,
    candidate_family: object,
    qualification_case_id: object,
    resource_requirement_body_sha256: object,
    resource_field_order_sha256: object,
    resource_fields: object,
    image_id: object,
    runtime_identity: object,
    container_identity: object,
    seal_architecture: object,
    host_provisioning_receipt: object,
    measurement_producer: object,
    storage_root_inventory_sha256_value: object,
    storage_roots: object,
    field_policy_inventory_sha256: object,
    field_policy: object,
) -> tuple[tuple[StorageRootBindingV1, ...], tuple[StorageFieldPolicyV1, ...]]:
    exact_schema_version = _require_identifier(schema_version, "storage artifact schema")
    if exact_schema_version != expected_schema_version:
        _fail("storage artifact schema differs")
    _require_sha256(campaign_spine_sha256, "storage campaign spine")
    _require_sha256(case_spine_sha256, "storage case spine")
    _require_case_projection(
        case_ordinal=case_ordinal,
        candidate_id=candidate_id,
        candidate_family=candidate_family,
        qualification_case_id=qualification_case_id,
    )
    _require_sha256(resource_requirement_body_sha256, "resource requirement body")
    exact_resource_field_order_sha256 = _require_sha256(
        resource_field_order_sha256,
        "resource field-order identity",
    )
    if exact_resource_field_order_sha256 != RESOURCE_FIELD_ORDER_SHA256:
        _fail("resource field-order identity differs")
    if type(resource_fields) is not tuple or any(type(item) is not str for item in resource_fields):
        _fail("resource fields must use one exact tuple of exact strings")
    exact_resource_fields = cast(tuple[str, ...], resource_fields)
    for item in exact_resource_fields:
        _require_identifier(item, "resource field")
    if exact_resource_fields != RESOURCE_FIELDS:
        _fail("resource fields must use the exact 28-field identity and order")
    _require_image_id(image_id, "storage-boundary image")
    if type(runtime_identity) is not RuntimeStorageIdentityV1:
        _fail("runtime storage identity type differs")
    if type(container_identity) is not ContainerStorageIdentityV1:
        _fail("container storage identity type differs")
    if type(seal_architecture) is not StorageSealArchitectureV1:
        _fail("storage seal architecture type differs")
    _require_artifact_schema(
        host_provisioning_receipt,
        HOST_PROVISIONING_RECEIPT_SCHEMA_VERSION,
        "host provisioning receipt",
    )
    if type(measurement_producer) is not ProducerIdentityV1:
        _fail("storage measurement producer identity type differs")
    exact_root_inventory_sha256 = _require_sha256(
        storage_root_inventory_sha256_value,
        "storage root inventory",
    )
    exact_roots = _validate_storage_roots(storage_roots, container_identity)
    expected_root_digest = storage_root_inventory_sha256(exact_roots, container_identity)
    if exact_root_inventory_sha256 != expected_root_digest:
        _fail("storage root inventory digest does not replay")
    exact_field_policy_inventory_sha256 = _require_sha256(
        field_policy_inventory_sha256,
        "storage field-policy inventory",
    )
    exact_policy = _validate_field_policies(field_policy, exact_roots)
    if exact_field_policy_inventory_sha256 != storage_field_policy_inventory_sha256(
        exact_policy,
        exact_roots,
    ):
        _fail("storage field-policy inventory digest does not replay")
    return exact_roots, exact_policy


@dataclass(frozen=True, slots=True)
class QualificationStorageBoundaryIntentV1:
    """Pre-GO exact storage identity and measurement-mode commitment."""

    schema_version: str
    campaign_spine_sha256: str
    case_spine_sha256: str
    case_ordinal: int
    candidate_id: str
    candidate_family: Literal["local", "external", "adapter"]
    qualification_case_id: str
    resource_requirement_body_sha256: str
    resource_field_order_sha256: str
    resource_fields: tuple[str, ...]
    image_id: str
    runtime_identity: RuntimeStorageIdentityV1
    container_identity: ContainerStorageIdentityV1
    seal_architecture: StorageSealArchitectureV1
    host_provisioning_receipt: ArtifactIdentityV1
    measurement_producer: ProducerIdentityV1
    writable_mount_policy_body_sha256: str
    writable_path_policy_body_sha256: str
    storage_root_inventory_sha256: str
    storage_roots: tuple[StorageRootBindingV1, ...]
    field_policy_inventory_sha256: str
    field_policy: tuple[StorageFieldPolicyV1, ...]
    intent_committed_before_go: bool
    go_identity_bound_in_intent: bool

    def __post_init__(self) -> None:
        _validate_common_identity(
            schema_version=self.schema_version,
            expected_schema_version=QUALIFICATION_STORAGE_BOUNDARY_INTENT_SCHEMA_VERSION,
            campaign_spine_sha256=self.campaign_spine_sha256,
            case_spine_sha256=self.case_spine_sha256,
            case_ordinal=self.case_ordinal,
            candidate_id=self.candidate_id,
            candidate_family=self.candidate_family,
            qualification_case_id=self.qualification_case_id,
            resource_requirement_body_sha256=self.resource_requirement_body_sha256,
            resource_field_order_sha256=self.resource_field_order_sha256,
            resource_fields=self.resource_fields,
            image_id=self.image_id,
            runtime_identity=self.runtime_identity,
            container_identity=self.container_identity,
            seal_architecture=self.seal_architecture,
            host_provisioning_receipt=self.host_provisioning_receipt,
            measurement_producer=self.measurement_producer,
            storage_root_inventory_sha256_value=self.storage_root_inventory_sha256,
            storage_roots=self.storage_roots,
            field_policy_inventory_sha256=self.field_policy_inventory_sha256,
            field_policy=self.field_policy,
        )
        _require_sha256(
            self.writable_mount_policy_body_sha256,
            "pre-GO writable mount policy body",
        )
        _require_sha256(
            self.writable_path_policy_body_sha256,
            "pre-GO writable path policy body",
        )
        if _require_bool(self.intent_committed_before_go, "intent committed before GO") is not True:
            _fail("storage intent must be committed before host GO")
        if (
            _require_bool(self.go_identity_bound_in_intent, "GO identity bound in intent")
            is not False
        ):
            _fail("pre-GO storage intent cannot reverse-bind a future GO identity")

    def to_body_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "status": QUALIFICATION_STORAGE_BOUNDARY_INTENT_STATUS,
            "classification": QUALIFICATION_STORAGE_BOUNDARY_CONTRACT_CLASSIFICATION,
            "campaign_spine_sha256": self.campaign_spine_sha256,
            "case_spine_sha256": self.case_spine_sha256,
            "case_ordinal": self.case_ordinal,
            "candidate_id": self.candidate_id,
            "candidate_family": self.candidate_family,
            "qualification_case_id": self.qualification_case_id,
            "resource_requirement_body_sha256": self.resource_requirement_body_sha256,
            "resource_field_order_sha256": self.resource_field_order_sha256,
            "resource_fields": list(self.resource_fields),
            "image_id": self.image_id,
            "runtime_identity": self.runtime_identity.to_dict(),
            "container_identity": self.container_identity.to_dict(),
            "seal_architecture": self.seal_architecture.to_dict(),
            "host_provisioning_receipt": self.host_provisioning_receipt.to_dict(),
            "measurement_producer": self.measurement_producer.to_dict(),
            "writable_mount_policy_body_sha256": (self.writable_mount_policy_body_sha256),
            "writable_path_policy_body_sha256": self.writable_path_policy_body_sha256,
            "storage_root_inventory_sha256": self.storage_root_inventory_sha256,
            "storage_roots": [item.to_dict() for item in self.storage_roots],
            "field_policy_inventory_sha256": self.field_policy_inventory_sha256,
            "field_policy": [item.to_dict() for item in self.field_policy],
            "intent_committed_before_go": self.intent_committed_before_go,
            "go_identity_bound_in_intent": self.go_identity_bound_in_intent,
            "capabilities": _capabilities(),
            "readiness": _readiness(),
            "authority": _authority(),
            "claims": _claims(),
            "limitations": _limitations(),
        }

    def to_dict(self) -> dict[str, Any]:
        return _with_body_sha256(self.to_body_dict(), "intent_body_sha256")

    @property
    def body_sha256(self) -> str:
        return _body_sha256(self.to_body_dict())


@dataclass(frozen=True, slots=True)
class QualificationStorageBoundaryReceiptV1:
    """Post-case, preterminal storage peak receipt closed by an irreversible seal."""

    schema_version: str
    campaign_spine_sha256: str
    case_spine_sha256: str
    case_ordinal: int
    candidate_id: str
    candidate_family: Literal["local", "external", "adapter"]
    qualification_case_id: str
    resource_requirement_body_sha256: str
    resource_field_order_sha256: str
    resource_fields: tuple[str, ...]
    image_id: str
    runtime_identity: RuntimeStorageIdentityV1
    container_identity: ContainerStorageIdentityV1
    seal_architecture: StorageSealArchitectureV1
    host_provisioning_receipt: ArtifactIdentityV1
    measurement_producer: ProducerIdentityV1
    measurement_intent: ArtifactIdentityV1
    collection_boundary_chain: CollectionBoundaryChainV1
    writable_mount_policy_body_sha256: str
    writable_path_policy_body_sha256: str
    storage_root_inventory_sha256: str
    storage_roots: tuple[StorageRootBindingV1, ...]
    field_policy_inventory_sha256: str
    field_policy: tuple[StorageFieldPolicyV1, ...]
    writable_surface_evidence: WritableSurfaceEvidenceV1
    write_quiescence_seal: WriteQuiescenceSealProofV1
    field_inventory_sha256: str
    fields: tuple[StoragePeakMeasurementV1, ...]

    def __post_init__(self) -> None:
        roots, policies = _validate_common_identity(
            schema_version=self.schema_version,
            expected_schema_version=QUALIFICATION_STORAGE_BOUNDARY_RECEIPT_SCHEMA_VERSION,
            campaign_spine_sha256=self.campaign_spine_sha256,
            case_spine_sha256=self.case_spine_sha256,
            case_ordinal=self.case_ordinal,
            candidate_id=self.candidate_id,
            candidate_family=self.candidate_family,
            qualification_case_id=self.qualification_case_id,
            resource_requirement_body_sha256=self.resource_requirement_body_sha256,
            resource_field_order_sha256=self.resource_field_order_sha256,
            resource_fields=self.resource_fields,
            image_id=self.image_id,
            runtime_identity=self.runtime_identity,
            container_identity=self.container_identity,
            seal_architecture=self.seal_architecture,
            host_provisioning_receipt=self.host_provisioning_receipt,
            measurement_producer=self.measurement_producer,
            storage_root_inventory_sha256_value=self.storage_root_inventory_sha256,
            storage_roots=self.storage_roots,
            field_policy_inventory_sha256=self.field_policy_inventory_sha256,
            field_policy=self.field_policy,
        )
        _require_artifact_schema(
            self.measurement_intent,
            QUALIFICATION_STORAGE_BOUNDARY_INTENT_SCHEMA_VERSION,
            "storage measurement intent",
        )
        _validate_collection_boundary_projection(
            self.collection_boundary_chain,
            self.measurement_intent,
        )
        for value, label in (
            (self.writable_mount_policy_body_sha256, "writable mount policy body"),
            (self.writable_path_policy_body_sha256, "writable path policy body"),
        ):
            _require_sha256(value, label)
        if type(self.writable_surface_evidence) is not WritableSurfaceEvidenceV1:
            _fail("writable-surface evidence type differs")
        if type(self.write_quiescence_seal) is not WriteQuiescenceSealProofV1:
            _fail("write-quiescence seal proof type differs")
        if (
            self.write_quiescence_seal.seal_architecture_body_sha256
            != self.seal_architecture.body_sha256
        ):
            _fail("write-seal proof architecture identity differs")
        worker_exit_expected = (
            self.seal_architecture.architecture_kind == "worker_exit_then_isolated_terminal_relay"
        )
        if (
            self.write_quiescence_seal.worker_exit_observed is not worker_exit_expected
            or self.write_quiescence_seal.irreversible_seccomp_fd_closure_observed
            is worker_exit_expected
        ):
            _fail("write-seal mechanism differs from its pre-GO architecture")
        fields = _validate_measurements(
            self.fields,
            policies,
            roots,
            self.storage_root_inventory_sha256,
            self.writable_surface_evidence,
            self.write_quiescence_seal,
        )
        exact_field_inventory_sha256 = _require_sha256(
            self.field_inventory_sha256,
            "storage field inventory",
        )
        expected_field_digest = storage_measurement_inventory_sha256(
            fields,
            policies,
            roots,
            self.storage_root_inventory_sha256,
            self.writable_surface_evidence,
            self.write_quiescence_seal,
        )
        if exact_field_inventory_sha256 != expected_field_digest:
            _fail("storage field inventory digest does not replay")

    def to_body_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "status": QUALIFICATION_STORAGE_BOUNDARY_RECEIPT_STATUS,
            "classification": QUALIFICATION_STORAGE_BOUNDARY_CONTRACT_CLASSIFICATION,
            "campaign_spine_sha256": self.campaign_spine_sha256,
            "case_spine_sha256": self.case_spine_sha256,
            "case_ordinal": self.case_ordinal,
            "candidate_id": self.candidate_id,
            "candidate_family": self.candidate_family,
            "qualification_case_id": self.qualification_case_id,
            "resource_requirement_body_sha256": self.resource_requirement_body_sha256,
            "resource_field_order_sha256": self.resource_field_order_sha256,
            "resource_fields": list(self.resource_fields),
            "image_id": self.image_id,
            "runtime_identity": self.runtime_identity.to_dict(),
            "container_identity": self.container_identity.to_dict(),
            "seal_architecture": self.seal_architecture.to_dict(),
            "host_provisioning_receipt": self.host_provisioning_receipt.to_dict(),
            "measurement_producer": self.measurement_producer.to_dict(),
            "measurement_intent": self.measurement_intent.to_dict(),
            "collection_boundary_chain": self.collection_boundary_chain.to_dict(),
            "writable_mount_policy_body_sha256": (self.writable_mount_policy_body_sha256),
            "writable_path_policy_body_sha256": self.writable_path_policy_body_sha256,
            "storage_root_inventory_sha256": self.storage_root_inventory_sha256,
            "storage_roots": [item.to_dict() for item in self.storage_roots],
            "field_policy_inventory_sha256": self.field_policy_inventory_sha256,
            "field_policy": [item.to_dict() for item in self.field_policy],
            "writable_surface_evidence": self.writable_surface_evidence.to_dict(),
            "write_quiescence_seal": self.write_quiescence_seal.to_dict(),
            "field_inventory_sha256": self.field_inventory_sha256,
            "fields": [item.to_dict() for item in self.fields],
            "capabilities": _capabilities(),
            "readiness": _readiness(),
            "authority": _authority(),
            "claims": _claims(),
            "limitations": _limitations(),
        }

    def to_dict(self) -> dict[str, Any]:
        return _with_body_sha256(self.to_body_dict(), "receipt_body_sha256")

    @property
    def body_sha256(self) -> str:
        return _body_sha256(self.to_body_dict())


def canonical_matched_v3_qualification_storage_boundary_intent_bytes(
    intent: QualificationStorageBoundaryIntentV1,
) -> bytes:
    """Serialize one pre-GO intent as canonical ASCII JSON with one trailing LF."""

    if type(intent) is not QualificationStorageBoundaryIntentV1:
        raise TypeError("storage boundary intent must use the exact intent type")
    return _canonical_json(intent.to_dict())


def canonical_matched_v3_qualification_storage_boundary_receipt_bytes(
    receipt: QualificationStorageBoundaryReceiptV1,
) -> bytes:
    """Serialize one sealed receipt as canonical ASCII JSON with one trailing LF."""

    if type(receipt) is not QualificationStorageBoundaryReceiptV1:
        raise TypeError("storage boundary receipt must use the exact receipt type")
    return _canonical_json(receipt.to_dict())


def _artifact_identity_from_dict(value: object, label: str) -> ArtifactIdentityV1:
    item = _require_exact_keys(
        value,
        frozenset(ArtifactIdentityV1.__dataclass_fields__),
        label,
    )
    return ArtifactIdentityV1(**item)


def _producer_identity_from_dict(value: object) -> ProducerIdentityV1:
    item = _require_exact_keys(
        value,
        frozenset(ProducerIdentityV1.__dataclass_fields__),
        "storage measurement producer identity",
    )
    return ProducerIdentityV1(**item)


def _component_identity_from_dict(value: object, label: str) -> ComponentIdentityV1:
    item = _require_exact_keys(
        value,
        frozenset(ComponentIdentityV1.__dataclass_fields__),
        label,
    )
    return ComponentIdentityV1(**item)


def _runtime_identity_from_dict(value: object) -> RuntimeStorageIdentityV1:
    item = dict(
        _require_exact_keys(
            value,
            frozenset(RuntimeStorageIdentityV1.__dataclass_fields__),
            "runtime storage identity",
        )
    )
    runtime_candidate = _artifact_identity_from_dict(
        item.pop("runtime_candidate"),
        "runtime candidate identity",
    )
    runtime_receipt = _artifact_identity_from_dict(
        item.pop("runtime_qualification_receipt"),
        "runtime qualification-receipt identity",
    )
    return RuntimeStorageIdentityV1(
        **item,
        runtime_candidate=runtime_candidate,
        runtime_qualification_receipt=runtime_receipt,
    )


def _container_identity_from_dict(value: object) -> ContainerStorageIdentityV1:
    item = _require_exact_keys(
        value,
        frozenset(ContainerStorageIdentityV1.__dataclass_fields__),
        "container storage identity",
    )
    return ContainerStorageIdentityV1(**item)


def _seal_architecture_from_dict(value: object) -> StorageSealArchitectureV1:
    item = dict(
        _require_exact_keys(
            value,
            frozenset(StorageSealArchitectureV1.__dataclass_fields__),
            "storage seal architecture",
        )
    )
    terminal_relay = _component_identity_from_dict(
        item.pop("terminal_relay"),
        "terminal relay component identity",
    )
    control_channel = _component_identity_from_dict(
        item.pop("nonstorage_control_channel"),
        "nonstorage channel component identity",
    )
    seal_producer = _component_identity_from_dict(
        item.pop("write_seal_producer"),
        "write-seal producer component identity",
    )
    return StorageSealArchitectureV1(
        **item,
        terminal_relay=terminal_relay,
        nonstorage_control_channel=control_channel,
        write_seal_producer=seal_producer,
    )


def _storage_root_from_dict(value: object) -> StorageRootBindingV1:
    item = _require_exact_keys(
        value,
        frozenset(StorageRootBindingV1.__dataclass_fields__),
        "storage root binding",
    )
    return StorageRootBindingV1(**item)


def _field_policy_from_dict(value: object) -> StorageFieldPolicyV1:
    item = dict(
        _require_exact_keys(
            value,
            frozenset(StorageFieldPolicyV1.__dataclass_fields__),
            "storage field policy",
        )
    )
    root_ids = item.pop("root_ids")
    if type(root_ids) is not list:
        _fail("storage field policy root IDs must be one list")
    return StorageFieldPolicyV1(
        **item,
        root_ids=tuple(root_ids),
    )


def _collection_boundary_from_dict(value: object) -> CollectionBoundaryChainV1:
    item = dict(
        _require_exact_keys(
            value,
            frozenset(CollectionBoundaryChainV1.__dataclass_fields__),
            "storage collection-boundary chain",
        )
    )
    request = _artifact_identity_from_dict(
        item.pop("host_case_request"),
        "host case-request identity",
    )
    intent = _artifact_identity_from_dict(
        item.pop("host_case_intent"),
        "host case-intent identity",
    )
    ready = _artifact_identity_from_dict(
        item.pop("host_ready"),
        "host READY identity",
    )
    observer_anchor = _artifact_identity_from_dict(
        item.pop("host_observer_anchor"),
        "host observer-anchor identity",
    )
    go = _artifact_identity_from_dict(
        item.pop("host_go"),
        "host GO identity",
    )
    return CollectionBoundaryChainV1(
        **item,
        host_case_request=request,
        host_case_intent=intent,
        host_ready=ready,
        host_observer_anchor=observer_anchor,
        host_go=go,
    )


def _writable_surface_from_dict(value: object) -> WritableSurfaceEvidenceV1:
    item = _require_exact_keys(
        value,
        frozenset(WritableSurfaceEvidenceV1.__dataclass_fields__),
        "writable-surface evidence",
    )
    return WritableSurfaceEvidenceV1(**item)


def _write_seal_from_dict(value: object) -> WriteQuiescenceSealProofV1:
    item = dict(
        _require_exact_keys(
            value,
            frozenset(WriteQuiescenceSealProofV1.__dataclass_fields__),
            "write-quiescence seal proof",
        )
    )
    publication = _artifact_identity_from_dict(
        item.pop("publication_commitment"),
        "publication commitment identity",
    )
    seal = _artifact_identity_from_dict(
        item.pop("write_quiescence_seal"),
        "write-quiescence seal identity",
    )
    relay = _artifact_identity_from_dict(
        item.pop("terminal_relay_preseal_attestation"),
        "terminal relay preseal-attestation identity",
    )
    channel = _artifact_identity_from_dict(
        item.pop("nonstorage_channel_readiness_attestation"),
        "nonstorage channel readiness-attestation identity",
    )
    reload_validation = _artifact_identity_from_dict(
        item.pop("publication_reload_validation"),
        "publication reload-validation identity",
    )
    return WriteQuiescenceSealProofV1(
        **item,
        publication_commitment=publication,
        write_quiescence_seal=seal,
        terminal_relay_preseal_attestation=relay,
        nonstorage_channel_readiness_attestation=channel,
        publication_reload_validation=reload_validation,
    )


def _event_evidence_from_dict(value: object) -> EventCompleteAccountingEvidenceV1:
    item = dict(
        _require_exact_keys(
            value,
            frozenset(EventCompleteAccountingEvidenceV1.__dataclass_fields__),
            "event-complete accounting evidence",
        )
    )
    root_ids = item.pop("field_root_ids")
    if type(root_ids) is not list:
        _fail("event evidence field-root IDs must be one list")
    return EventCompleteAccountingEvidenceV1(
        **item,
        field_root_ids=tuple(root_ids),
    )


def _quota_evidence_from_dict(value: object) -> QuotaEnforcementEvidenceV1:
    item = dict(
        _require_exact_keys(
            value,
            frozenset(QuotaEnforcementEvidenceV1.__dataclass_fields__),
            "quota-enforcement evidence",
        )
    )
    root_ids = item.pop("field_root_ids")
    if type(root_ids) is not list:
        _fail("quota evidence field-root IDs must be one list")
    return QuotaEnforcementEvidenceV1(
        **item,
        field_root_ids=tuple(root_ids),
    )


def _absence_evidence_from_dict(value: object) -> StructuralAbsenceEvidenceV1:
    item = _require_exact_keys(
        value,
        frozenset(StructuralAbsenceEvidenceV1.__dataclass_fields__),
        "storage structural-absence evidence",
    )
    return StructuralAbsenceEvidenceV1(**item)


def _measurement_from_dict(value: object) -> StoragePeakMeasurementV1:
    item = dict(
        _require_exact_keys(
            value,
            frozenset(StoragePeakMeasurementV1.__dataclass_fields__),
            "storage peak measurement",
        )
    )
    event_raw = item.pop("event_complete_evidence")
    quota_raw = item.pop("quota_enforcement_evidence")
    absence_raw = item.pop("structural_absence_evidence")
    return StoragePeakMeasurementV1(
        **item,
        event_complete_evidence=(
            None if event_raw is None else _event_evidence_from_dict(event_raw)
        ),
        quota_enforcement_evidence=(
            None if quota_raw is None else _quota_evidence_from_dict(quota_raw)
        ),
        structural_absence_evidence=(
            None if absence_raw is None else _absence_evidence_from_dict(absence_raw)
        ),
    )


_COMMON_ENVELOPE_KEYS: Final = frozenset(
    {
        "status",
        "classification",
        "capabilities",
        "readiness",
        "authority",
        "claims",
        "limitations",
    }
)


def _validate_envelope(item: dict[str, Any], *, status: str, label: str) -> None:
    if not _exact_json_equal(item.pop("status"), status):
        _fail(f"{label} status differs")
    if not _exact_json_equal(
        item.pop("classification"),
        QUALIFICATION_STORAGE_BOUNDARY_CONTRACT_CLASSIFICATION,
    ):
        _fail(f"{label} classification differs")
    if not _exact_json_equal(item.pop("capabilities"), _capabilities()):
        _fail(f"{label} capabilities differ")
    if not _exact_json_equal(item.pop("readiness"), _readiness()):
        _fail(f"{label} readiness differs")
    if not _exact_json_equal(item.pop("authority"), _authority()):
        _fail(f"{label} authority differs")
    if not _exact_json_equal(item.pop("claims"), _claims()):
        _fail(f"{label} claims differ")
    if not _exact_json_equal(item.pop("limitations"), _limitations()):
        _fail(f"{label} limitations differ")


def parse_matched_v3_qualification_storage_boundary_intent(
    raw: bytes,
    *,
    expected_file_sha256: str,
) -> QualificationStorageBoundaryIntentV1:
    """Parse one exact pre-GO intent after checking its caller full-file pin."""

    _validate_caller_file_pin(raw, expected_file_sha256, "storage boundary intent")
    value = _strict_json_load(raw)
    _validate_body_sha256(value, "intent_body_sha256", "storage boundary intent")
    expected_keys = frozenset(
        {
            *QualificationStorageBoundaryIntentV1.__dataclass_fields__,
            *_COMMON_ENVELOPE_KEYS,
            "intent_body_sha256",
        }
    )
    item = dict(_require_exact_keys(value, expected_keys, "storage boundary intent"))
    item.pop("intent_body_sha256")
    _validate_envelope(
        item,
        status=QUALIFICATION_STORAGE_BOUNDARY_INTENT_STATUS,
        label="storage boundary intent",
    )
    resource_fields = item.pop("resource_fields")
    roots = item.pop("storage_roots")
    policies = item.pop("field_policy")
    if type(resource_fields) is not list:
        _fail("storage intent resource fields must be one list")
    if type(roots) is not list:
        _fail("storage intent roots must be one list")
    if type(policies) is not list:
        _fail("storage intent policies must be one list")
    runtime_identity = _runtime_identity_from_dict(item.pop("runtime_identity"))
    container_identity = _container_identity_from_dict(item.pop("container_identity"))
    seal_architecture = _seal_architecture_from_dict(item.pop("seal_architecture"))
    provisioning = _artifact_identity_from_dict(
        item.pop("host_provisioning_receipt"),
        "host provisioning-receipt identity",
    )
    producer = _producer_identity_from_dict(item.pop("measurement_producer"))
    intent = QualificationStorageBoundaryIntentV1(
        **item,
        resource_fields=tuple(resource_fields),
        runtime_identity=runtime_identity,
        container_identity=container_identity,
        seal_architecture=seal_architecture,
        host_provisioning_receipt=provisioning,
        measurement_producer=producer,
        storage_roots=tuple(_storage_root_from_dict(child) for child in roots),
        field_policy=tuple(_field_policy_from_dict(child) for child in policies),
    )
    if raw != canonical_matched_v3_qualification_storage_boundary_intent_bytes(intent):
        _fail("storage boundary intent canonical replay differs")
    return intent


def parse_matched_v3_qualification_storage_boundary_receipt(
    raw: bytes,
    *,
    expected_file_sha256: str,
) -> QualificationStorageBoundaryReceiptV1:
    """Parse one exact sealed receipt after checking its caller full-file pin."""

    _validate_caller_file_pin(raw, expected_file_sha256, "storage boundary receipt")
    value = _strict_json_load(raw)
    _validate_body_sha256(value, "receipt_body_sha256", "storage boundary receipt")
    expected_keys = frozenset(
        {
            *QualificationStorageBoundaryReceiptV1.__dataclass_fields__,
            *_COMMON_ENVELOPE_KEYS,
            "receipt_body_sha256",
        }
    )
    item = dict(_require_exact_keys(value, expected_keys, "storage boundary receipt"))
    item.pop("receipt_body_sha256")
    _validate_envelope(
        item,
        status=QUALIFICATION_STORAGE_BOUNDARY_RECEIPT_STATUS,
        label="storage boundary receipt",
    )
    resource_fields = item.pop("resource_fields")
    roots = item.pop("storage_roots")
    policies = item.pop("field_policy")
    fields = item.pop("fields")
    if type(resource_fields) is not list:
        _fail("storage receipt resource fields must be one list")
    if type(roots) is not list:
        _fail("storage receipt roots must be one list")
    if type(policies) is not list:
        _fail("storage receipt policies must be one list")
    if type(fields) is not list:
        _fail("storage receipt fields must be one list")
    runtime_identity = _runtime_identity_from_dict(item.pop("runtime_identity"))
    container_identity = _container_identity_from_dict(item.pop("container_identity"))
    seal_architecture = _seal_architecture_from_dict(item.pop("seal_architecture"))
    provisioning = _artifact_identity_from_dict(
        item.pop("host_provisioning_receipt"),
        "host provisioning-receipt identity",
    )
    producer = _producer_identity_from_dict(item.pop("measurement_producer"))
    measurement_intent = _artifact_identity_from_dict(
        item.pop("measurement_intent"),
        "storage measurement-intent identity",
    )
    boundary_chain = _collection_boundary_from_dict(item.pop("collection_boundary_chain"))
    writable_surface = _writable_surface_from_dict(item.pop("writable_surface_evidence"))
    write_seal = _write_seal_from_dict(item.pop("write_quiescence_seal"))
    receipt = QualificationStorageBoundaryReceiptV1(
        **item,
        resource_fields=tuple(resource_fields),
        runtime_identity=runtime_identity,
        container_identity=container_identity,
        seal_architecture=seal_architecture,
        host_provisioning_receipt=provisioning,
        measurement_producer=producer,
        measurement_intent=measurement_intent,
        collection_boundary_chain=boundary_chain,
        storage_roots=tuple(_storage_root_from_dict(child) for child in roots),
        field_policy=tuple(_field_policy_from_dict(child) for child in policies),
        writable_surface_evidence=writable_surface,
        write_quiescence_seal=write_seal,
        fields=tuple(_measurement_from_dict(child) for child in fields),
    )
    if raw != canonical_matched_v3_qualification_storage_boundary_receipt_bytes(receipt):
        _fail("storage boundary receipt canonical replay differs")
    return receipt


def validate_matched_v3_qualification_storage_boundary_chain(
    intent: QualificationStorageBoundaryIntentV1,
    receipt: QualificationStorageBoundaryReceiptV1,
) -> None:
    """Validate the acyclic intent-to-sealed-receipt projection without I/O."""

    if type(intent) is not QualificationStorageBoundaryIntentV1:
        raise TypeError("storage boundary intent must use the exact intent type")
    if type(receipt) is not QualificationStorageBoundaryReceiptV1:
        raise TypeError("storage boundary receipt must use the exact receipt type")
    intent_raw = canonical_matched_v3_qualification_storage_boundary_intent_bytes(intent)
    expected_intent = ArtifactIdentityV1(
        schema_version=QUALIFICATION_STORAGE_BOUNDARY_INTENT_SCHEMA_VERSION,
        file_sha256=_sha256(intent_raw),
        body_sha256=intent.body_sha256,
    )
    if receipt.measurement_intent != expected_intent:
        _fail("storage receipt measurement-intent identity differs")
    for field in (
        "campaign_spine_sha256",
        "case_spine_sha256",
        "case_ordinal",
        "candidate_id",
        "candidate_family",
        "qualification_case_id",
        "resource_requirement_body_sha256",
        "resource_field_order_sha256",
        "resource_fields",
        "image_id",
        "runtime_identity",
        "container_identity",
        "seal_architecture",
        "host_provisioning_receipt",
        "measurement_producer",
        "writable_mount_policy_body_sha256",
        "writable_path_policy_body_sha256",
        "storage_root_inventory_sha256",
        "storage_roots",
        "field_policy_inventory_sha256",
        "field_policy",
    ):
        if getattr(receipt, field) != getattr(intent, field):
            _fail(f"storage receipt intent projection differs for {field}")


def _contract_descriptor() -> dict[str, Any]:
    body: dict[str, Any] = {
        "schema_version": (QUALIFICATION_STORAGE_BOUNDARY_CONTRACT_DESCRIPTOR_SCHEMA_VERSION),
        "status": QUALIFICATION_STORAGE_BOUNDARY_CONTRACT_STATUS,
        "classification": QUALIFICATION_STORAGE_BOUNDARY_CONTRACT_CLASSIFICATION,
        "canonical_encoding": "ascii_sorted_keys_compact_one_trailing_newline",
        "full_file_caller_pin_required": True,
        "intent_schema_version": QUALIFICATION_STORAGE_BOUNDARY_INTENT_SCHEMA_VERSION,
        "receipt_schema_version": QUALIFICATION_STORAGE_BOUNDARY_RECEIPT_SCHEMA_VERSION,
        "producer_descriptor_schema_version": (
            QUALIFICATION_STORAGE_BOUNDARY_PRODUCER_DESCRIPTOR_SCHEMA_VERSION
        ),
        "write_seal_schema_version": QUALIFICATION_STORAGE_WRITE_SEAL_SCHEMA_VERSION,
        "publication_reload_validation_schema_version": (
            QUALIFICATION_PUBLICATION_RELOAD_VALIDATION_SCHEMA_VERSION
        ),
        "terminal_relay_preseal_attestation_schema_version": (
            QUALIFICATION_STORAGE_TERMINAL_RELAY_PRESEAL_ATTESTATION_SCHEMA_VERSION
        ),
        "nonstorage_channel_readiness_attestation_schema_version": (
            QUALIFICATION_STORAGE_NONSTORAGE_CHANNEL_READINESS_ATTESTATION_SCHEMA_VERSION
        ),
        "host_observer_anchor_schema_version": HOST_OBSERVER_ANCHOR_SCHEMA_VERSION,
        "seal_component_descriptor_schemas": {
            "terminal_relay": (QUALIFICATION_STORAGE_TERMINAL_RELAY_DESCRIPTOR_SCHEMA_VERSION),
            "nonstorage_control_channel": (
                QUALIFICATION_STORAGE_NONSTORAGE_CHANNEL_DESCRIPTOR_SCHEMA_VERSION
            ),
            "write_seal_producer": (
                QUALIFICATION_STORAGE_WRITE_SEAL_PRODUCER_DESCRIPTOR_SCHEMA_VERSION
            ),
        },
        "candidate_order": list(MATCHED_V3_STORAGE_CANDIDATE_IDS),
        "candidate_order_sha256": MATCHED_V3_STORAGE_CANDIDATE_ORDER_SHA256,
        "resource_fields": list(RESOURCE_FIELDS),
        "resource_field_order_sha256": RESOURCE_FIELD_ORDER_SHA256,
        "storage_fields": list(STORAGE_RESOURCE_FIELDS),
        "storage_field_positions": dict(STORAGE_FIELD_POSITIONS),
        "storage_field_zero_absence": dict(STORAGE_FIELD_ZERO_ABSENCE),
        "allowed_measurement_modes": [
            EVENT_COMPLETE_ACCOUNTING_MODE,
            HARD_QUOTA_ENFORCEMENT_MODE,
        ],
        "allowed_value_semantics": [
            EXACT_OBSERVATION,
            CONSERVATIVE_ENFORCED_UPPER_BOUND,
        ],
        "storage_lifetime_scope": STORAGE_LIFETIME_SCOPE,
        "event_complete_contract": {
            "fresh_boundary_through_irreversible_seal": True,
            "kernel_and_filesystem_events_complete": True,
            "allocation_deallocation_copy_up_and_deleted_open_files_covered": True,
            "loss_or_overflow_permitted": False,
            "exact_high_water_replay_required": True,
            "reported_peak_is_simultaneous_aggregate_over_exact_field_root_union": True,
            "maximum_of_per_root_peaks_is_sufficient": False,
        },
        "hard_quota_contract": {
            "allowed_enforcement_kinds": list(STORAGE_ENFORCEMENT_KINDS),
            "installed_before_go": True,
            "immutable_and_non_bypass_through_container_removal": True,
            "all_bound_roots_covered": True,
            "alternate_writable_mounts_or_paths_permitted": False,
            "breach_status_required": True,
            "successful_receipt_requires_no_breach": True,
            "reported_value_equals_hard_limit": True,
            "hard_limit_applies_once_to_aggregate_exact_field_root_union": True,
            "independent_per_root_limits_are_sufficient": False,
        },
        "anti_substitution_contract": {
            "polling_or_periodic_sampling_sufficient": False,
            "du_snapshots_sufficient": False,
            "container_layer_size_sufficient": False,
            "missing_value_defaults_to_zero": False,
            "zero_requires_exact_typed_structural_absence": True,
        },
        "writable_surface_contract": {
            "exact_mount_and_path_inventories_required": True,
            "all_writable_mounts_and_paths_bound": True,
            "unbound_or_alternate_writable_surfaces_permitted": False,
            "overlay_copy_up_bound_or_disabled": True,
            "deleted_open_anonymous_and_memory_backed_files_covered": True,
            "host_device_network_and_inherited_fd_write_escape_disabled": True,
            "post_go_mount_namespace_mutation_disabled": True,
            "late_remount_seal_permitted": False,
        },
        "preterminal_write_seal_contract": {
            "allowed_architectures": [
                "worker_exit_then_isolated_terminal_relay",
                "irreversible_seccomp_fd_closure_then_isolated_terminal_relay",
            ],
            "architecture_committed_before_go": True,
            "publication_committed_before_seal": True,
            "wrapper_precommits_expected_reload_observation": True,
            "actual_reload_observation_must_match_wrapper_commitment": True,
            "publication_reload_validation_body_replayed": True,
            "publication_reload_validation_file_replayed": True,
            "reload_performed": True,
            "reload_read_only": True,
            "reload_validated_before_seal": True,
            "worker_loses_all_measured_writable_namespace_and_fds": True,
            "only_trusted_terminal_relay_retains_terminal_transport_capability": True,
            "inert_seccomp_closed_worker_may_remain": True,
            "terminal_relay_input_policy_restricted_to_receipt_identity": True,
            "nonstorage_channel_ready_for_post_receipt_terminal": True,
            "receipt_delivery_observed_in_storage_receipt": False,
            "terminal_emission_observed_in_storage_receipt": False,
            "terminal_v2_proves_post_receipt_delivery_and_emission": True,
            "host_success_v2_projects_downstream_delivery_and_emission_proof": True,
            "container_log_driver": "none",
            "container_logging_can_write_measured_storage": False,
            "later_container_or_descendant_write_allocation_or_copy_up_possible": False,
            "terminal_transport_inside_measured_storage": False,
            "teardown_is_deletion_only": True,
            "teardown_can_increase_peak": False,
            "missing_seal_or_no_later_writer_proof_fails_closed": True,
        },
        "artifact_chain": [
            "pre_go_storage_boundary_intent",
            "host_request_v2_binds_storage_intent",
            "host_intent_v2_binds_request",
            "host_ready_v2_binds_storage_intent",
            "host_ready_v2_binds_host_intent_v2",
            "host_observer_anchor_v2_binds_ready_v2",
            "host_go_v2",
            "fresh_case_storage_boundary",
            "case_execution_and_native_publication",
            "normalized_wrapper_expected_reload_commitment",
            "actual_publication_reload_validation",
            "irreversible_preterminal_write_quiescence_seal",
            "storage_boundary_receipt",
            "terminal_v2_binds_storage_receipt",
            "lifecycle_and_host_success_v2",
            "full_resource_merger",
        ],
        "receipt_may_bind_preexisting_handshake_and_collection_boundaries": True,
        "collection_boundary_exact_projection_chain": (
            "request_to_host_intent_to_ready_to_observer_anchor_to_go"
        ),
        "request_and_ready_bind_exact_storage_intent_file_and_body": True,
        "go_binds_exact_ready_and_observer_anchor_file_and_body": True,
        "handshake_chain_body_digest_replayed_from_exact_projections": True,
        "measurement_evidence_binds_exact_write_seal_file_and_body": True,
        "quota_evidence_binds_exact_storage_root_inventory": True,
        "reverse_receipt_pins_forbidden": [
            "terminal_v2",
            "lifecycle_v2",
            "host_success_v2",
            "full_resource_merger",
        ],
        "intent_precedes_go": True,
        "receipt_precedes_terminal_v2": True,
        "terminal_v2_binds_receipt_one_way": True,
        "storage_operations_performed": False,
        "candidate_values_supplied_or_inferred": False,
        "ceiling_comparison_performed": False,
        "capabilities": _capabilities(),
        "readiness": _readiness(),
        "authority": _authority(),
        "claims": _claims(),
        "limitations": _limitations(),
    }
    return _with_body_sha256(body, "descriptor_body_sha256")


_DESCRIPTOR: Final = _contract_descriptor()
_DESCRIPTOR_BYTES: Final = _canonical_json(_DESCRIPTOR)

# Independently replayed from the canonical one-LF descriptor file after schema audit.
PINNED_QUALIFICATION_STORAGE_BOUNDARY_CONTRACT_DESCRIPTOR_SHA256: Final = (
    "d294de196f3b96192e3810571ddbe5b39fdf4615efec9d4460cf4e4d5f6c6a4c"
)


def matched_v3_qualification_storage_boundary_contract_descriptor() -> dict[str, Any]:
    """Return the source-only, non-authorizing contract descriptor."""

    return copy.deepcopy(_DESCRIPTOR)


def canonical_matched_v3_qualification_storage_boundary_contract_descriptor_bytes() -> bytes:
    """Return canonical descriptor bytes without bypassing the literal audit pin."""

    return bytes(_DESCRIPTOR_BYTES)


def matched_v3_qualification_storage_boundary_contract_descriptor_sha256() -> str:
    """Return the descriptor identity only after its literal pin is audited."""

    observed = _sha256(_DESCRIPTOR_BYTES)
    if not hmac.compare_digest(
        observed,
        PINNED_QUALIFICATION_STORAGE_BOUNDARY_CONTRACT_DESCRIPTOR_SHA256,
    ):
        _fail("storage-boundary contract descriptor drifted from its literal pin")
    return observed


def parse_matched_v3_qualification_storage_boundary_contract_descriptor(
    raw: bytes,
) -> dict[str, Any]:
    """Parse the exact descriptor only after an independent pin audit."""

    value = _strict_json_load(raw)
    _validate_body_sha256(value, "descriptor_body_sha256", "contract descriptor")
    if not _exact_json_equal(value, _DESCRIPTOR):
        _fail("storage-boundary contract descriptor content differs")
    if raw != _DESCRIPTOR_BYTES:
        _fail("storage-boundary contract descriptor canonical replay differs")
    matched_v3_qualification_storage_boundary_contract_descriptor_sha256()
    return copy.deepcopy(value)


__all__ = [
    "CONSERVATIVE_ENFORCED_UPPER_BOUND",
    "ComponentIdentityV1",
    "ContainerStorageIdentityV1",
    "CollectionBoundaryChainV1",
    "DISK_STORAGE_STRUCTURALLY_ABSENT",
    "EVENT_COMPLETE_ACCOUNTING_MODE",
    "EXACT_OBSERVATION",
    "EventCompleteAccountingEvidenceV1",
    "ForagerMatchedV3QualificationStorageBoundaryError",
    "HARD_QUOTA_ENFORCEMENT_MODE",
    "HOST_CASE_INTENT_SCHEMA_VERSION",
    "HOST_CASE_REQUEST_SCHEMA_VERSION",
    "HOST_GO_SCHEMA_VERSION",
    "HOST_OBSERVER_ANCHOR_SCHEMA_VERSION",
    "HOST_PROVISIONING_RECEIPT_SCHEMA_VERSION",
    "HOST_READY_SCHEMA_VERSION",
    "MATCHED_V3_ADAPTER_CANDIDATE_IDS",
    "MATCHED_V3_EXTERNAL_CANDIDATE_IDS",
    "MATCHED_V3_LOCAL_CANDIDATE_IDS",
    "MATCHED_V3_STORAGE_CANDIDATE_IDS",
    "MATCHED_V3_STORAGE_CANDIDATE_ORDER_SHA256",
    "NORMALIZED_PUBLICATION_COMMITMENT_WRAPPER_SCHEMA_VERSION",
    "NOT_ABSENT",
    "PINNED_QUALIFICATION_STORAGE_BOUNDARY_CONTRACT_DESCRIPTOR_SHA256",
    "ProducerIdentityV1",
    "QUALIFICATION_STORAGE_BOUNDARY_CONTRACT_CLASSIFICATION",
    "QUALIFICATION_STORAGE_BOUNDARY_CONTRACT_DESCRIPTOR_SCHEMA_VERSION",
    "QUALIFICATION_STORAGE_BOUNDARY_CONTRACT_STATUS",
    "QUALIFICATION_STORAGE_BOUNDARY_INTENT_SCHEMA_VERSION",
    "QUALIFICATION_STORAGE_BOUNDARY_INTENT_STATUS",
    "QUALIFICATION_STORAGE_BOUNDARY_PRODUCER_DESCRIPTOR_SCHEMA_VERSION",
    "QUALIFICATION_STORAGE_BOUNDARY_RECEIPT_SCHEMA_VERSION",
    "QUALIFICATION_STORAGE_BOUNDARY_RECEIPT_STATUS",
    "QUALIFICATION_STORAGE_NONSTORAGE_CHANNEL_READINESS_ATTESTATION_SCHEMA_VERSION",
    "QUALIFICATION_STORAGE_NONSTORAGE_CHANNEL_DESCRIPTOR_SCHEMA_VERSION",
    "QUALIFICATION_PUBLICATION_RELOAD_VALIDATION_SCHEMA_VERSION",
    "QUALIFICATION_STORAGE_TERMINAL_RELAY_PRESEAL_ATTESTATION_SCHEMA_VERSION",
    "QUALIFICATION_STORAGE_TERMINAL_RELAY_DESCRIPTOR_SCHEMA_VERSION",
    "QUALIFICATION_STORAGE_WRITE_SEAL_PRODUCER_DESCRIPTOR_SCHEMA_VERSION",
    "QUALIFICATION_STORAGE_WRITE_SEAL_SCHEMA_VERSION",
    "QualificationStorageBoundaryIntentV1",
    "QualificationStorageBoundaryReceiptV1",
    "QuotaEnforcementEvidenceV1",
    "RESOURCE_FIELDS",
    "RESOURCE_FIELD_ORDER_SHA256",
    "RUNTIME_CANDIDATE_SCHEMA_VERSION",
    "RUNTIME_QUALIFICATION_RECEIPT_SCHEMA_VERSION",
    "RuntimeStorageIdentityV1",
    "STORAGE_ENFORCEMENT_KINDS",
    "STORAGE_FIELD_POSITIONS",
    "STORAGE_FIELD_ZERO_ABSENCE",
    "STORAGE_LIFETIME_SCOPE",
    "STORAGE_RESOURCE_FIELDS",
    "StorageFieldPolicyV1",
    "StoragePeakMeasurementV1",
    "StorageRootBindingV1",
    "StorageSealArchitectureV1",
    "StructuralAbsenceEvidenceV1",
    "TEMPORARY_STORAGE_STRUCTURALLY_ABSENT",
    "WritableSurfaceEvidenceV1",
    "WriteQuiescenceSealProofV1",
    "ArtifactIdentityV1",
    "canonical_matched_v3_qualification_storage_boundary_contract_descriptor_bytes",
    "canonical_matched_v3_qualification_storage_boundary_intent_bytes",
    "canonical_matched_v3_qualification_storage_boundary_receipt_bytes",
    "matched_v3_qualification_storage_boundary_contract_descriptor",
    "matched_v3_qualification_storage_boundary_contract_descriptor_sha256",
    "parse_matched_v3_qualification_storage_boundary_contract_descriptor",
    "parse_matched_v3_qualification_storage_boundary_intent",
    "parse_matched_v3_qualification_storage_boundary_receipt",
    "storage_field_policy_inventory_sha256",
    "storage_field_root_inventory_sha256",
    "storage_measurement_inventory_sha256",
    "storage_root_inventory_sha256",
    "validate_matched_v3_qualification_storage_boundary_chain",
]
