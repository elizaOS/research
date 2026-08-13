"""Source-only phase-1 host-qualification v2 contract for matched Forager v3.

This module defines canonical metadata and validation boundaries for a future
single-case CPU OCI host executor.  It deliberately contains no subprocess,
Docker, OCI, cgroup, process, filesystem-mutation, issuer, evaluator, resource
merger, or workload implementation.  Importing it cannot execute a case.

Operational progress and recovery are separate records.  An operational
failure freezes the first uncertain boundary, while a conditional recovery DAG
can still reach terminal metadata, a nonretryable failure receipt, and a
canonical observation handoff without pretending skipped operational phases
committed.  All artifact links carry schema, full-file, and BODY identities.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final, Literal, NoReturn, Protocol, cast

HOST_EXECUTOR_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.host_qualification_executor_descriptor.v2"
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
HOST_OBSERVER_ANCHOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.host_observer_anchor.v2"
)
HOST_GO_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.host_qualification_go_commitment.v2"
)
HOST_OPERATIONAL_FRONTIER_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.host_qualification_operational_frontier.v2"
)
HOST_INITIAL_CGROUP_SAMPLE_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.host_initial_cgroup_sample.v2"
)
HOST_PRECLEANUP_CGROUP_SAMPLE_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.host_precleanup_cgroup_sample.v2"
)
HOST_CGROUP_KILL_RECEIPT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.host_cgroup_kill_receipt.v2"
)
HOST_CGROUP_EMPTY_OBSERVATION_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.host_cgroup_empty_observation.v2"
)
HOST_CONTAINER_ABSENCE_OBSERVATION_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.host_container_absence_observation.v2"
)
HOST_POST_CONTAINER_REMOVE_CGROUP_SAMPLE_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.host_post_container_remove_cgroup_sample.v2"
)
HOST_CGROUP_COUNTER_FDS_CLOSED_RECEIPT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.host_cgroup_counter_fds_closed_receipt.v2"
)
HOST_OUTER_CGROUP_ABSENCE_OBSERVATION_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.host_outer_cgroup_absence_observation.v2"
)
HOST_CGROUP_PROOF_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.host_cgroup_v2_boundary_proof.v1"
)
HOST_CLEANUP_RECONCILIATION_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.host_cleanup_reconciliation.v2"
)
HOST_TERMINAL_METADATA_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.host_terminal_metadata.v2"
)
IN_CONTAINER_DRIVER_TERMINAL_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.in_container_qualification_driver_terminal.v2"
)
HOST_LIFECYCLE_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.host_qualification_lifecycle_record.v2"
)
HOST_SUCCESS_RECEIPT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.host_qualification_case_execution_receipt.v2"
)
HOST_FAILURE_RECEIPT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.host_qualification_failure_receipt.v2"
)
HOST_OBSERVATION_HANDOFF_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.host_qualification_observation_handoff.v2"
)

QUALIFICATION_PLAN_V3_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.qualification_plan.v3"
)
QUALIFICATION_OBSERVATION_REGISTRY_V2_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.qualification_observation_registry_descriptor.v2"
)
PLAN_ISSUANCE_RECEIPT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.qualification_plan_issuance_receipt.v1"
)
CASE_EXECUTION_TICKET_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.qualification_case_execution_ticket.v1"
)
QUALIFICATION_CASE_MANIFEST_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.qualification_case_manifest.v2"
)
JOINT_SOURCE_CLOSURE_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.qualification_joint_source_closure_candidate.v1"
)
SEALED_STAGING_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.qualification_sealed_staging_candidate.v1"
)
FRESH_BUILD_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.fresh_cpu_oci_build_candidate.v2"
)
RUNTIME_QUALIFICATION_RECEIPT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.runtime_qualification_receipt.v1"
)
HOST_PROVISIONING_RECEIPT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.host_provisioning_receipt.v2"
)
ALGORITHMIC_MEASUREMENT_INTENT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.algorithmic_resource_measurement_intent.v1"
)
STORAGE_BOUNDARY_INTENT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.qualification_storage_boundary_intent.v1"
)
ALGORITHMIC_CONTRACT_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.algorithmic_resource_contract_descriptor.v1"
)
PUBLICATION_CONTRACT_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.qualification_publication_commitment_contract_descriptor.v1"
)
STORAGE_CONTRACT_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.qualification_storage_boundary_contract_descriptor.v1"
)
LOCAL_ALGORITHMIC_RESOURCE_RECEIPT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.local_algorithmic_resource_receipt.v1"
)
EXTERNAL_ALGORITHMIC_RESOURCE_RECEIPT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.external_algorithmic_resource_receipt.v1"
)
ADAPTER_ALGORITHMIC_RESOURCE_RECEIPT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.adapter_algorithmic_resource_receipt.v1"
)
PUBLICATION_COMMITMENT_WRAPPER_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.qualification_publication_commitment_wrapper.v1"
)
PUBLICATION_RELOAD_VALIDATION_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.qualification_publication_reload_validation.v1"
)
STORAGE_WRITE_SEAL_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.qualification_storage_write_quiescence_seal.v1"
)
STORAGE_BOUNDARY_RECEIPT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.qualification_storage_boundary_receipt.v1"
)
CPU_OCI_BUILD_CONTEXT_RECEIPT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.cpu_oci_build_context_receipt.v1"
)
CPU_OCI_BUILD_EXECUTION_RECEIPT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.cpu_oci_build_execution_receipt.v1"
)
CPU_OCI_BUILD_PUBLICATION_RECEIPT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.cpu_oci_build_publication.v1"
)
FULL_RESOURCE_MERGER_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.full_resource_merger_descriptor.v1"
)
LOCAL_PUBLICATION_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.local_reward_publication_descriptor.v1"
)
EXTERNAL_PUBLICATION_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.external_reward_publication_descriptor.v1"
)
STRICT_ADAPTER_PUBLICATION_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.adapter_qualification_publication_descriptor.v1"
)
ATOMIC_PUBLICATION_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.atomic_publication_descriptor.v1"
)
STRICT_ADAPTER_ATOMIC_PUBLICATION_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.adapter_qualification_atomic_publication_descriptor.v1"
)
LOCAL_RUNNER_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.local_runner_descriptor.v1"
)
EXTERNAL_EXECUTION_RUNNER_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.external_execution_runner_descriptor.v1"
)
FULL_RAINBOW_RUNNER_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.full_rainbow_runner.v1"
)
PPO_GRU_RUNNER_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.ppo_gru_runner.v1"
)
EXTERNAL_ATOMIC_PUBLICATION_RECEIPT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.external_atomic_publication_receipt.v1"
)
STRICT_ADAPTER_ATOMIC_PUBLICATION_RECEIPT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.adapter_atomic_publication_receipt.v1"
)
PUBLICATION_RECONCILIATION_REFERENCE_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.qualification_publication_reconciliation_reference.v1"
)
FAILURE_PUBLICATION_PROJECTION_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.qualification_failure_publication_projection.v2"
)
HOST_CGROUP_MEMBERSHIP_EVENT_LOG_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.host_cgroup_membership_event_log.v2"
)

HOST_EXECUTION_ACKNOWLEDGEMENT: Final = (
    "AUTHORIZE ONE MATCHED-V3 HOST OCI QUALIFICATION CASE EXECUTION"
)
MATCHED_V3_HORIZON: Final = 499_712

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
MATCHED_V3_CANDIDATE_IDS: Final = (
    MATCHED_V3_LOCAL_CANDIDATE_IDS
    + MATCHED_V3_EXTERNAL_CANDIDATE_IDS[:9]
    + MATCHED_V3_ADAPTER_CANDIDATE_IDS
    + MATCHED_V3_EXTERNAL_CANDIDATE_IDS[9:]
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

OPERATIONAL_PHASES: Final = (
    "request_validated",
    "authorization_validated",
    "intent_committed",
    "fresh_cgroup_created",
    "retained_counter_fds_opened",
    "initial_cgroup_sample_committed",
    "container_created",
    "container_started",
    "driver_ready",
    "observer_anchored",
    "go_committed",
    "workload_started",
    "workload_exited",
    "algorithmic_resource_receipt_committed",
    "native_publication_committed",
    "publication_commitment_wrapper_committed",
    "publication_reload_validated",
    "storage_write_seal_committed",
    "storage_boundary_receipt_committed",
)

CGROUP_COUNTER_ENDPOINTS: Final = (
    "cpu.stat",
    "memory.current",
    "memory.peak",
    "memory.events",
    "pids.current",
    "pids.peak",
    "pids.events",
    "cgroup.events",
    "cgroup.stat",
    "cgroup.kill",
)
RECOVERY_NODE_NAMES: Final = (
    "precleanup_cgroup_sample",
    "cgroup_kill",
    "cgroup_empty",
    "container_absence",
    "post_container_remove_cgroup_sample",
    "cgroup_counter_fds_closed",
    "outer_cgroup_absence",
    "final_cgroup_proof",
)
RECOVERY_NODE_DEPENDENCIES: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType({
    "precleanup_cgroup_sample": (),
    "cgroup_kill": ("precleanup_cgroup_sample",),
    # These are conditional branches, so a kill-write uncertainty does not
    # suppress direct emptiness observation or container reconciliation.
    "cgroup_empty": ("precleanup_cgroup_sample",),
    "container_absence": (),
    "post_container_remove_cgroup_sample": ("container_absence",),
    "cgroup_counter_fds_closed": ("post_container_remove_cgroup_sample",),
    "outer_cgroup_absence": ("cgroup_counter_fds_closed",),
    "final_cgroup_proof": (
        "precleanup_cgroup_sample",
        "cgroup_kill",
        "cgroup_empty",
        "container_absence",
        "post_container_remove_cgroup_sample",
        "cgroup_counter_fds_closed",
        "outer_cgroup_absence",
    ),
})
RECOVERY_NODE_SCHEMAS: Final[Mapping[str, str]] = MappingProxyType({
    "precleanup_cgroup_sample": HOST_PRECLEANUP_CGROUP_SAMPLE_SCHEMA_VERSION,
    "cgroup_kill": HOST_CGROUP_KILL_RECEIPT_SCHEMA_VERSION,
    "cgroup_empty": HOST_CGROUP_EMPTY_OBSERVATION_SCHEMA_VERSION,
    "container_absence": HOST_CONTAINER_ABSENCE_OBSERVATION_SCHEMA_VERSION,
    "post_container_remove_cgroup_sample": (
        HOST_POST_CONTAINER_REMOVE_CGROUP_SAMPLE_SCHEMA_VERSION
    ),
    "cgroup_counter_fds_closed": HOST_CGROUP_COUNTER_FDS_CLOSED_RECEIPT_SCHEMA_VERSION,
    "outer_cgroup_absence": HOST_OUTER_CGROUP_ABSENCE_OBSERVATION_SCHEMA_VERSION,
    "final_cgroup_proof": HOST_CGROUP_PROOF_SCHEMA_VERSION,
})

FINAL_ALGORITHMIC_CONTRACT_DESCRIPTOR_SHA256: Final = (
    "9eb50aa96169dc9cb38745d729e0b429b01781b32435c86a54cee99b6590321d"
)
FINAL_ALGORITHMIC_CONTRACT_SOURCE_SHA256: Final = (
    "c0df02b504d3d5695782f0b68b1518ae4b549a5e13074c7a5ce6dd39313abef3"
)
FINAL_PUBLICATION_CONTRACT_DESCRIPTOR_SHA256: Final = (
    "e2b2c556bba5ee4eb168a1d990eb73b6b273a6685c7e86818ed5bee142191420"
)
FINAL_PUBLICATION_CONTRACT_SOURCE_SHA256: Final = (
    "7737ff1b12dab2fc569cda241821a37fee47c6038dcadf1c3578f79fccf82c80"
)
FINAL_STORAGE_CONTRACT_DESCRIPTOR_SHA256: Final = (
    "d294de196f3b96192e3810571ddbe5b39fdf4615efec9d4460cf4e4d5f6c6a4c"
)
FINAL_STORAGE_CONTRACT_SOURCE_SHA256: Final = (
    "9ae173c4ddbecac1ea64777d6227db6f07b78db97c8485175e7cf4954b645dcf"
)

INCOMPATIBLE_HOST_V1_DESCRIPTOR_SHA256: Final = (
    "da7692691aee585b774a2d4a31ba7243d2f5ce005b9b31fe8ceb4a1993653bb8"
)
INCOMPATIBLE_HOST_V1_SOURCE_SHA256: Final = (
    "d8bbc666a49e252662807f256c7f212c9a7c8c3be279b928a6a93ed77532a2e1"
)
HISTORICAL_IMAGE_IDS: Final = (
    "sha256:a1f491fc786a788b2629e0670ee52ad84138057e58dd795703a830ea2e42c269",
    "sha256:5ecaabefce6439a8731c19e7a55fedb666788242baf035e6ffca86eb31299768",
)
HISTORICAL_BUILD_LINEAGE_SHA256S: Final = frozenset(
    {
        "ccacc85f9adf6d81368050be37c67cbd38bb2423cc147deea580a152acf2b330",
        "38cab52b6d247bf045405bd9de9d63b36f00d4e2f79bbb7a154d663ee24b8e9d",
        "28892dd3be5c29df122a94a4feb35045fd17f95475e5e7237c0a04b4b15cbd88",
    }
)
INCOMPATIBLE_ADAPTER_IDENTITY_SHA256S: Final = frozenset(
    {
        "5ca0f236a7b6ac58a67578282ca2091f1a443a72502c81fe08b2ecf850ec7905",
        "679ea0f6b5d572ec7777d45f4bc115c8d6bcf7df3f3155bd3a784fa59c48dfc6",
        "546009c19454a7839876df6e758b984db931db5eb234ac23833a232c387aa3bc",
        "e9cfa6785ef48783224f548fa17db0f8291ee1a47ef29f098692c31beb5f00b2",
        "3d95ed7f550cdbd946934e02f452f072bf2a0397a39dfb712be9782d2d6e2565",
        "cc9e2ad605496682ff2870bb6db312f56ad4926f4805a4a90fbacac4f648cf08",
        "a7827fd32b526c1ad3f9d22549a66fa054c3785c75891560356db82791a3f500",
        "8c2c42aad0db0a8eeb45ad2d33f3d76046121fe1f74160e8d1a10231dbe545b5",
        "bae29ef65246c7beabe34a134a755c18e10a1467dd9914b65be1f05a760bb6f2",
        "5546b8cd6b394857ad96d4e2bdcaf6e3427cdb16057dd8f67e79654dd617146c",
        "afffdbaf46b9af2cfffe131c8a3bb88dee6de257a8b21296068f22ad5aa93d47",
        "08dc9c8d36fb98661ec4a8922973dc25df78d881807651f873843e7ddf64a27f",
        "e50466c185d66334f629915944407d72cb4aff4aa611dffbbe20de8aa8146f6e",
        "42ea4bbf5f01818b1f1f44c9410eeaa0a1fe51326a29399c175e1e859e6b8a71",
    }
)

_MAX_ARTIFACT_BYTES: Final = 4 * 1024 * 1024
_MAX_JSON_DEPTH: Final = 64
_MAX_JSON_NODES: Final = 100_000
_MAX_TEXT_LENGTH: Final = 16_384
_MAX_INTEGER: Final = 2**63 - 1
_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_IMAGE_ID_RE: Final = re.compile(r"sha256:[0-9a-f]{64}\Z")
_IDENTIFIER_RE: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}\Z")
_CONTAINER_ID_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_CONTAINER_NAME_RE: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_CGROUP_PATH_RE: Final = re.compile(r"/sys/fs/cgroup/[a-z0-9_.\-/]{1,4000}\Z")
_FORBIDDEN_KEY_PARTS: Final = ("score", "reward")

CGROUP_DELEGATE_ROOT_PATH: Final = "/sys/fs/cgroup/alberta-qualified-host"
CGROUP_DELEGATE_PARENT_ARGUMENT: Final = "/alberta-qualified-host"


class ForagerMatchedV3HostQualificationExecutorV2Error(ValueError):
    """A host-v2 canonical contract or chain failed closed."""


class _BodyArtifact(Protocol):
    def to_body_dict(self) -> dict[str, Any]: ...


def _fail(message: str) -> NoReturn:
    raise ForagerMatchedV3HostQualificationExecutorV2Error(message)


def _require_bool(value: object, label: str) -> bool:
    if type(value) is not bool:
        _fail(f"{label} must be one exact bool")
    return value


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


def _require_text(value: object, label: str) -> str:
    if (
        type(value) is not str
        or not value
        or len(value) > _MAX_TEXT_LENGTH
        or any(ord(character) < 0x20 or ord(character) > 0x7E for character in value)
    ):
        _fail(f"{label} must be bounded printable ASCII text")
    return value


def _require_identifier(value: object, label: str) -> str:
    exact = _require_text(value, label)
    if _IDENTIFIER_RE.fullmatch(exact) is None:
        _fail(f"{label} must be one portable identifier")
    return exact


def _require_sha256(value: object, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None or value == "0" * 64:
        _fail(f"{label} must be one nonzero lowercase SHA-256")
    return value


def _require_optional_sha256(value: object, label: str) -> str | None:
    if value is None:
        return None
    return _require_sha256(value, label)


def _require_image_id(value: object, label: str) -> str:
    if type(value) is not str or _IMAGE_ID_RE.fullmatch(value) is None:
        _fail(f"{label} must be one exact sha256 OCI image identity")
    exact = value
    if exact == "sha256:" + "0" * 64:
        _fail(f"{label} cannot use the all-zero OCI image sentinel")
    if exact in HISTORICAL_IMAGE_IDS:
        _fail(f"{label} is a permanently excluded historical image")
    return exact


def _reject_constant(value: str) -> NoReturn:
    _fail(f"canonical JSON contains forbidden constant {value!r}")


def _reject_float(value: str) -> NoReturn:
    _fail(f"canonical JSON contains forbidden float {value!r}")


def _parse_int(value: str) -> int:
    if len(value.lstrip("-")) > 19:
        _fail("canonical JSON integer exceeds its lexical bound")
    parsed = int(value)
    if abs(parsed) > _MAX_INTEGER:
        _fail("canonical JSON integer exceeds its value bound")
    return parsed


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if type(key) is not str or key in result:
            _fail("canonical JSON contains a duplicate or non-text key")
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
            _fail("canonical JSON structure exceeds its bound")
        if item is None or type(item) is bool:
            continue
        if type(item) is int:
            _require_int(item, "canonical JSON integer", minimum=-_MAX_INTEGER)
            continue
        if type(item) is str:
            _require_text(item, "canonical JSON text")
            continue
        if type(item) not in {list, dict}:
            _fail("canonical JSON contains a non-plain value")
        identity = id(item)
        if identity in containers:
            _fail("canonical JSON contains an alias or cycle")
        containers.add(identity)
        if type(item) is list:
            pending.extend((child, depth + 1) for child in cast(list[object], item))
        else:
            mapping = cast(dict[object, object], item)
            for key, child in mapping.items():
                if type(key) is not str:
                    _fail("canonical JSON key is not one exact string")
                _require_text(key, "canonical JSON key")
                lowered = key.lower()
                if any(part in lowered for part in _FORBIDDEN_KEY_PARTS):
                    _fail(f"canonical host metadata contains forbidden key {key!r}")
                pending.append((child, depth + 1))


def _canonical_json(value: object, *, newline: bool) -> bytes:
    _assert_plain_unaliased_json(value)
    try:
        raw = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise ForagerMatchedV3HostQualificationExecutorV2Error(
            "value is not canonical ASCII JSON"
        ) from exc
    if newline:
        raw += b"\n"
    if len(raw) > _MAX_ARTIFACT_BYTES:
        _fail("canonical JSON exceeds its byte bound")
    return raw


def _strict_json(raw: bytes) -> dict[str, Any]:
    if type(raw) is not bytes or not raw or len(raw) > _MAX_ARTIFACT_BYTES:
        _fail("artifact bytes violate their bound")
    try:
        value = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_object_without_duplicates,
            parse_float=_reject_float,
            parse_constant=_reject_constant,
            parse_int=_parse_int,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ForagerMatchedV3HostQualificationExecutorV2Error(
            "artifact bytes are not strict ASCII JSON"
        ) from exc
    if type(value) is not dict:
        _fail("artifact JSON root must be one object")
    exact = cast(dict[str, Any], value)
    _assert_plain_unaliased_json(exact)
    if raw != _canonical_json(exact, newline=True):
        _fail("artifact bytes are not canonical one-LF JSON")
    return exact


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _body_sha256(body: Mapping[str, Any]) -> str:
    return _sha256(_canonical_json(dict(body), newline=False))


def _file_dict(body: Mapping[str, Any], body_field: str) -> dict[str, Any]:
    return {**dict(body), body_field: _body_sha256(body)}


def _require_exact_keys(value: object, keys: frozenset[str], label: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != keys:
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
    expected = _require_sha256(expected_file_sha256, f"{label} caller file pin")
    if not hmac.compare_digest(_sha256(raw), expected):
        _fail(f"{label} full-file SHA-256 differs")
    item = _strict_json(raw)
    _require_exact_keys(item, body_keys | {body_field}, label)
    supplied = _require_sha256(item[body_field], f"{label} BODY digest")
    body = {key: value for key, value in item.items() if key != body_field}
    if not hmac.compare_digest(supplied, _body_sha256(body)):
        _fail(f"{label} BODY digest differs")
    return body


def _canonical_artifact_body_bytes(value: object, expected_type: type[Any]) -> bytes:
    if type(value) is not expected_type:
        raise TypeError(f"artifact must use exact type {expected_type.__name__}")
    return _canonical_json(cast(_BodyArtifact, value).to_body_dict(), newline=False)


def _canonical_artifact_file_bytes(
    value: object,
    expected_type: type[Any],
    body_field: str,
) -> bytes:
    if type(value) is not expected_type:
        raise TypeError(f"artifact must use exact type {expected_type.__name__}")
    body = cast(_BodyArtifact, value).to_body_dict()
    return _canonical_json(_file_dict(body, body_field), newline=True)


def _family_for_candidate(candidate_id: str) -> Literal["local", "external", "adapter"]:
    if candidate_id in MATCHED_V3_LOCAL_CANDIDATE_IDS:
        return "local"
    if candidate_id in MATCHED_V3_EXTERNAL_CANDIDATE_IDS:
        return "external"
    if candidate_id in MATCHED_V3_ADAPTER_CANDIDATE_IDS:
        return "adapter"
    _fail("candidate is outside the exact matched-v3 order")


def expected_container_name_v2(case_spine_sha256: object) -> str:
    spine = _require_sha256(case_spine_sha256, "container-name case spine")
    name = "alberta-mv3-" + spine
    if _CONTAINER_NAME_RE.fullmatch(name) is None:
        _fail("derived container name is not Docker-valid")
    return name


def container_lookup_identity_sha256_v2(
    case_spine_sha256: object,
    container_name: object,
) -> str:
    spine = _require_sha256(case_spine_sha256, "container lookup case spine")
    name = _require_text(container_name, "container lookup name")
    if name != expected_container_name_v2(spine):
        _fail("container lookup name differs from the full case spine")
    return _body_sha256(
        {
            "case_spine_sha256": spine,
            "container_name": name,
            "lookup_semantics": "exact_name_lookup_without_default_selection",
        }
    )


def container_runtime_identity_sha256_v2(
    case_spine_sha256: object,
    container_name: object,
    container_id: object,
) -> str:
    spine = _require_sha256(case_spine_sha256, "runtime-container case spine")
    name = _require_text(container_name, "runtime-container name")
    if name != expected_container_name_v2(spine):
        _fail("runtime-container name differs from the full case spine")
    if (
        type(container_id) is not str
        or _CONTAINER_ID_RE.fullmatch(container_id) is None
        or container_id == "0" * 64
    ):
        _fail("runtime-container ID must be one nonzero exact 64-hex identity")
    return _body_sha256(
        {
            "case_spine_sha256": spine,
            "container_name": name,
            "container_id": container_id,
        }
    )


def _expected_publisher_schema(family: str) -> str:
    exact = _require_text(family, "publisher family")
    if exact == "local":
        return LOCAL_PUBLICATION_DESCRIPTOR_SCHEMA_VERSION
    if exact == "external":
        return EXTERNAL_PUBLICATION_DESCRIPTOR_SCHEMA_VERSION
    if exact == "adapter":
        return STRICT_ADAPTER_PUBLICATION_DESCRIPTOR_SCHEMA_VERSION
    _fail("publisher family differs")


def _expected_atomic_schema(family: str) -> str:
    exact = _require_text(family, "atomic producer family")
    if exact in {"local", "external"}:
        return ATOMIC_PUBLICATION_DESCRIPTOR_SCHEMA_VERSION
    if exact == "adapter":
        return STRICT_ADAPTER_ATOMIC_PUBLICATION_DESCRIPTOR_SCHEMA_VERSION
    _fail("atomic producer family differs")


def _expected_driver_schema(candidate_id: str) -> str:
    candidate = _require_identifier(candidate_id, "driver candidate ID")
    family = _family_for_candidate(candidate)
    if family == "local":
        return LOCAL_RUNNER_DESCRIPTOR_SCHEMA_VERSION
    if family == "external":
        return EXTERNAL_EXECUTION_RUNNER_DESCRIPTOR_SCHEMA_VERSION
    if candidate == "adapted_full_rainbow":
        return FULL_RAINBOW_RUNNER_DESCRIPTOR_SCHEMA_VERSION
    if candidate == "adapted_ppo_gru":
        return PPO_GRU_RUNNER_DESCRIPTOR_SCHEMA_VERSION
    _fail("adapter candidate has no exact driver schema")


def _require_case(
    ordinal: object,
    candidate_id: object,
    candidate_family: object,
    qualification_case_id: object,
) -> tuple[int, str, Literal["local", "external", "adapter"], str]:
    exact_ordinal = _require_int(
        ordinal,
        "case ordinal",
        maximum=len(MATCHED_V3_CANDIDATE_IDS) - 1,
    )
    exact_candidate = _require_identifier(candidate_id, "candidate ID")
    if exact_candidate != MATCHED_V3_CANDIDATE_IDS[exact_ordinal]:
        _fail("case ordinal and candidate differ from the exact order")
    family = _family_for_candidate(exact_candidate)
    if _require_text(candidate_family, "candidate family") != family:
        _fail("candidate family differs")
    exact_case_id = _require_identifier(qualification_case_id, "qualification case ID")
    if exact_case_id != f"qualification_{exact_ordinal:02d}_{exact_candidate}":
        _fail("qualification case ID differs")
    return exact_ordinal, exact_candidate, family, exact_case_id


def _claims() -> dict[str, bool]:
    return {
        "execution_authorized": False,
        "execution_performed": False,
        "qualification_granted": False,
        "resource_matched": False,
        "scientific_evidence_created": False,
    }


def _authority() -> dict[str, bool]:
    return {
        "issuer_available": False,
        "evaluator_available": False,
        "merger_available": False,
        "production_backend_available": False,
    }


@dataclass(frozen=True, slots=True)
class ArtifactIdentityV2:
    schema_version: str
    file_sha256: str
    body_sha256: str

    def __post_init__(self) -> None:
        _require_identifier(self.schema_version, "artifact schema")
        _require_sha256(self.file_sha256, "artifact file")
        _require_sha256(self.body_sha256, "artifact BODY")

    def to_dict(self) -> dict[str, str]:
        return {
            "schema_version": self.schema_version,
            "file_sha256": self.file_sha256,
            "body_sha256": self.body_sha256,
        }


@dataclass(frozen=True, slots=True)
class ProducerIdentityV2:
    descriptor_schema_version: str
    descriptor_sha256: str
    source_sha256: str

    def __post_init__(self) -> None:
        _require_identifier(self.descriptor_schema_version, "producer descriptor schema")
        _require_sha256(self.descriptor_sha256, "producer descriptor")
        _require_sha256(self.source_sha256, "producer source")
        if self.descriptor_sha256 == self.source_sha256:
            _fail("producer descriptor and source identities cannot alias")

    def to_dict(self) -> dict[str, str]:
        return {
            "descriptor_schema_version": self.descriptor_schema_version,
            "descriptor_sha256": self.descriptor_sha256,
            "source_sha256": self.source_sha256,
        }


def _artifact_identity(value: object, label: str) -> ArtifactIdentityV2:
    item = _require_exact_keys(
        value,
        frozenset({"schema_version", "file_sha256", "body_sha256"}),
        label,
    )
    return ArtifactIdentityV2(**item)


def _optional_artifact_identity(value: object, label: str) -> ArtifactIdentityV2 | None:
    if value is None:
        return None
    return _artifact_identity(value, label)


def _producer_identity(value: object, label: str) -> ProducerIdentityV2:
    item = _require_exact_keys(
        value,
        frozenset({"descriptor_schema_version", "descriptor_sha256", "source_sha256"}),
        label,
    )
    return ProducerIdentityV2(**item)


def _require_artifact_schema(
    value: object,
    schema_version: str,
    label: str,
) -> ArtifactIdentityV2:
    if type(value) is not ArtifactIdentityV2 or value.schema_version != schema_version:
        _fail(f"{label} artifact schema differs")
    return value


def _require_producer(
    value: object,
    schema_version: str,
    label: str,
) -> ProducerIdentityV2:
    if type(value) is not ProducerIdentityV2 or value.descriptor_schema_version != schema_version:
        _fail(f"{label} producer schema differs")
    return value


def _validate_resource_ceilings(value: object) -> tuple[tuple[str, int], ...]:
    if type(value) is not tuple or any(
        type(item) is not tuple or len(item) != 2 for item in cast(tuple[object, ...], value)
    ):
        _fail("resource ceilings must use one exact tuple of pairs")
    exact = cast(tuple[tuple[str, int], ...], value)
    if (
        any(type(name) is not str for name, _ in exact)
        or tuple(name for name, _ in exact) != RESOURCE_FIELDS
    ):
        _fail("resource ceilings differ from the exact 28-field order")
    for name, ceiling in exact:
        _require_int(ceiling, f"resource ceiling {name}")
    by_name = dict(exact)
    if (
        by_name["max_environment_interactions"] != MATCHED_V3_HORIZON
        or by_name["max_attempt_count"] != 1
        or by_name["max_failure_count"] != 0
        or by_name["max_thread_count"] < 1
    ):
        _fail("resource horizon, thread, attempt, or failure ceiling differs")
    return exact


def _reject_historical_lineage(artifacts: tuple[ArtifactIdentityV2, ...]) -> None:
    for artifact in artifacts:
        if (
            artifact.file_sha256 in HISTORICAL_BUILD_LINEAGE_SHA256S
            or artifact.body_sha256 in HISTORICAL_BUILD_LINEAGE_SHA256S
        ):
            _fail("historical build-lineage identity is permanently excluded")


def _reject_adapter_substitution(
    family: str,
    producers: tuple[ProducerIdentityV2, ...],
) -> None:
    if family != "adapter":
        return
    for producer in producers:
        if (
            producer.descriptor_sha256 in INCOMPATIBLE_ADAPTER_IDENTITY_SHA256S
            or producer.source_sha256 in INCOMPATIBLE_ADAPTER_IDENTITY_SHA256S
        ):
            _fail("historical or unqualified adapter identity cannot fill a v2 slot")


@dataclass(frozen=True, slots=True)
class HostQualificationCaseRequestV2:
    case_spine_sha256: str
    case_ordinal: int
    candidate_id: str
    candidate_family: Literal["local", "external", "adapter"]
    qualification_case_id: str
    container_name: str
    qualification_plan: ArtifactIdentityV2
    plan_issuance_receipt: ArtifactIdentityV2
    case_execution_ticket: ArtifactIdentityV2
    qualification_case_manifest: ArtifactIdentityV2
    observation_registry: ProducerIdentityV2
    joint_source_closure: ArtifactIdentityV2
    sealed_staging: ArtifactIdentityV2
    fresh_build: ArtifactIdentityV2
    build_context_receipt: ArtifactIdentityV2
    build_execution_receipt: ArtifactIdentityV2
    build_publication_receipt: ArtifactIdentityV2
    image_id: str
    runtime_qualification_receipt: ArtifactIdentityV2
    host_provisioning_receipt: ArtifactIdentityV2
    algorithmic_contract: ProducerIdentityV2
    algorithmic_measurement_intent: ArtifactIdentityV2
    publication_contract: ProducerIdentityV2
    storage_contract: ProducerIdentityV2
    storage_boundary_intent: ArtifactIdentityV2
    host_executor: ProducerIdentityV2
    full_resource_merger: ProducerIdentityV2
    publisher: ProducerIdentityV2
    native_atomic_producer: ProducerIdentityV2
    in_container_driver: ProducerIdentityV2
    resource_requirement_body_sha256: str
    declared_ceilings: tuple[tuple[str, int], ...]
    horizon: int
    attempt_ordinal: int
    exact_acknowledgement: str

    def __post_init__(self) -> None:
        _, _, family, _ = _require_case(
            self.case_ordinal,
            self.candidate_id,
            self.candidate_family,
            self.qualification_case_id,
        )
        _require_sha256(self.case_spine_sha256, "request case spine")
        name = _require_text(self.container_name, "request container name")
        if (
            _CONTAINER_NAME_RE.fullmatch(name) is None
            or name != expected_container_name_v2(self.case_spine_sha256)
        ):
            _fail("request container name differs from its complete case spine")
        _require_artifact_schema(
            self.qualification_plan,
            QUALIFICATION_PLAN_V3_SCHEMA_VERSION,
            "qualification plan",
        )
        _require_artifact_schema(
            self.plan_issuance_receipt,
            PLAN_ISSUANCE_RECEIPT_SCHEMA_VERSION,
            "plan issuance receipt",
        )
        _require_artifact_schema(
            self.case_execution_ticket,
            CASE_EXECUTION_TICKET_SCHEMA_VERSION,
            "case execution ticket",
        )
        _require_artifact_schema(
            self.qualification_case_manifest,
            QUALIFICATION_CASE_MANIFEST_SCHEMA_VERSION,
            "qualification case manifest",
        )
        _require_producer(
            self.observation_registry,
            QUALIFICATION_OBSERVATION_REGISTRY_V2_SCHEMA_VERSION,
            "observation registry",
        )
        _require_artifact_schema(
            self.joint_source_closure,
            JOINT_SOURCE_CLOSURE_SCHEMA_VERSION,
            "joint source closure",
        )
        _require_artifact_schema(self.sealed_staging, SEALED_STAGING_SCHEMA_VERSION, "staging")
        _require_artifact_schema(self.fresh_build, FRESH_BUILD_SCHEMA_VERSION, "fresh build")
        _require_artifact_schema(
            self.runtime_qualification_receipt,
            RUNTIME_QUALIFICATION_RECEIPT_SCHEMA_VERSION,
            "runtime qualification receipt",
        )
        _require_artifact_schema(
            self.host_provisioning_receipt,
            HOST_PROVISIONING_RECEIPT_SCHEMA_VERSION,
            "host provisioning receipt",
        )
        _require_artifact_schema(
            self.algorithmic_measurement_intent,
            ALGORITHMIC_MEASUREMENT_INTENT_SCHEMA_VERSION,
            "algorithmic measurement intent",
        )
        _require_artifact_schema(
            self.storage_boundary_intent,
            STORAGE_BOUNDARY_INTENT_SCHEMA_VERSION,
            "storage boundary intent",
        )
        expected_contracts = (
            (
                self.algorithmic_contract,
                ALGORITHMIC_CONTRACT_DESCRIPTOR_SCHEMA_VERSION,
                FINAL_ALGORITHMIC_CONTRACT_DESCRIPTOR_SHA256,
                FINAL_ALGORITHMIC_CONTRACT_SOURCE_SHA256,
                "algorithmic contract",
            ),
            (
                self.publication_contract,
                PUBLICATION_CONTRACT_DESCRIPTOR_SCHEMA_VERSION,
                FINAL_PUBLICATION_CONTRACT_DESCRIPTOR_SHA256,
                FINAL_PUBLICATION_CONTRACT_SOURCE_SHA256,
                "publication contract",
            ),
            (
                self.storage_contract,
                STORAGE_CONTRACT_DESCRIPTOR_SCHEMA_VERSION,
                FINAL_STORAGE_CONTRACT_DESCRIPTOR_SHA256,
                FINAL_STORAGE_CONTRACT_SOURCE_SHA256,
                "storage contract",
            ),
        )
        for producer, schema, descriptor, source, label in expected_contracts:
            _require_producer(producer, schema, label)
            if producer.descriptor_sha256 != descriptor or producer.source_sha256 != source:
                _fail(f"{label} differs from its final audited dependency")
        _require_producer(
            self.host_executor,
            HOST_EXECUTOR_DESCRIPTOR_SCHEMA_VERSION,
            "host executor",
        )
        incompatible_host_v1 = {
            INCOMPATIBLE_HOST_V1_DESCRIPTOR_SHA256,
            INCOMPATIBLE_HOST_V1_SOURCE_SHA256,
        }
        if any(
            digest in incompatible_host_v1
            for digest in (
                self.host_executor.descriptor_sha256,
                self.host_executor.source_sha256,
            )
        ):
            _fail("request host executor is not an additive v2 producer")
        audited_self_descriptor = _guard_host_executor_descriptor_pin()
        if self.host_executor.descriptor_sha256 != audited_self_descriptor:
            _fail("request host executor differs from the finalized self descriptor")
        for producer, schema, label in (
            (
                self.full_resource_merger,
                FULL_RESOURCE_MERGER_DESCRIPTOR_SCHEMA_VERSION,
                "full resource merger",
            ),
            (self.publisher, _expected_publisher_schema(family), "publisher"),
            (
                self.native_atomic_producer,
                _expected_atomic_schema(family),
                "native atomic producer",
            ),
            (
                self.in_container_driver,
                _expected_driver_schema(self.candidate_id),
                "in-container driver",
            ),
        ):
            _require_producer(producer, schema, label)
        _reject_adapter_substitution(
            family,
            (self.publisher, self.native_atomic_producer, self.in_container_driver),
        )
        if any(
            type(artifact) is not ArtifactIdentityV2
            for artifact in (
                self.build_context_receipt,
                self.build_execution_receipt,
                self.build_publication_receipt,
            )
        ):
            _fail("build receipt identity types differ")
        for artifact, schema, label in (
            (
                self.build_context_receipt,
                CPU_OCI_BUILD_CONTEXT_RECEIPT_SCHEMA_VERSION,
                "build context receipt",
            ),
            (
                self.build_execution_receipt,
                CPU_OCI_BUILD_EXECUTION_RECEIPT_SCHEMA_VERSION,
                "build execution receipt",
            ),
            (
                self.build_publication_receipt,
                CPU_OCI_BUILD_PUBLICATION_RECEIPT_SCHEMA_VERSION,
                "build publication receipt",
            ),
        ):
            _require_artifact_schema(artifact, schema, label)
        _reject_historical_lineage(
            (
                self.build_context_receipt,
                self.build_execution_receipt,
                self.build_publication_receipt,
            )
        )
        if len(
            {
                (artifact.file_sha256, artifact.body_sha256)
                for artifact in (
                    self.build_context_receipt,
                    self.build_execution_receipt,
                    self.build_publication_receipt,
                )
            }
        ) != 3:
            _fail("build context, execution, and publication receipts cannot alias")
        _require_image_id(self.image_id, "request image")
        _require_sha256(self.resource_requirement_body_sha256, "resource requirement BODY")
        _validate_resource_ceilings(self.declared_ceilings)
        if (
            _require_int(self.horizon, "request horizon") != MATCHED_V3_HORIZON
            or _require_int(self.attempt_ordinal, "request attempt ordinal") != 0
        ):
            _fail("request horizon or attempt ordinal differs")
        if (
            type(self.exact_acknowledgement) is not str
            or not hmac.compare_digest(
                self.exact_acknowledgement,
                HOST_EXECUTION_ACKNOWLEDGEMENT,
            )
        ):
            _fail("exact host execution acknowledgement differs")

    def to_body_dict(self) -> dict[str, Any]:
        return {
            "schema_version": HOST_CASE_REQUEST_SCHEMA_VERSION,
            "status": "source_only_request_validated_non_authorizing",
            "case_spine_sha256": self.case_spine_sha256,
            "case_ordinal": self.case_ordinal,
            "candidate_id": self.candidate_id,
            "candidate_family": self.candidate_family,
            "qualification_case_id": self.qualification_case_id,
            "container_name": self.container_name,
            "qualification_plan": self.qualification_plan.to_dict(),
            "plan_issuance_receipt": self.plan_issuance_receipt.to_dict(),
            "case_execution_ticket": self.case_execution_ticket.to_dict(),
            "qualification_case_manifest": self.qualification_case_manifest.to_dict(),
            "observation_registry": self.observation_registry.to_dict(),
            "joint_source_closure": self.joint_source_closure.to_dict(),
            "sealed_staging": self.sealed_staging.to_dict(),
            "fresh_build": self.fresh_build.to_dict(),
            "build_context_receipt": self.build_context_receipt.to_dict(),
            "build_execution_receipt": self.build_execution_receipt.to_dict(),
            "build_publication_receipt": self.build_publication_receipt.to_dict(),
            "image_id": self.image_id,
            "runtime_qualification_receipt": self.runtime_qualification_receipt.to_dict(),
            "host_provisioning_receipt": self.host_provisioning_receipt.to_dict(),
            "algorithmic_contract": self.algorithmic_contract.to_dict(),
            "algorithmic_measurement_intent": self.algorithmic_measurement_intent.to_dict(),
            "publication_contract": self.publication_contract.to_dict(),
            "storage_contract": self.storage_contract.to_dict(),
            "storage_boundary_intent": self.storage_boundary_intent.to_dict(),
            "host_executor": self.host_executor.to_dict(),
            "full_resource_merger": self.full_resource_merger.to_dict(),
            "publisher": self.publisher.to_dict(),
            "native_atomic_producer": self.native_atomic_producer.to_dict(),
            "in_container_driver": self.in_container_driver.to_dict(),
            "resource_requirement_body_sha256": self.resource_requirement_body_sha256,
            "declared_ceilings": [
                {"field_name": name, "declared_ceiling": ceiling}
                for name, ceiling in self.declared_ceilings
            ],
            "horizon": self.horizon,
            "attempt_ordinal": self.attempt_ordinal,
            "exact_acknowledgement": self.exact_acknowledgement,
            "acknowledgement_is_execution_authority": False,
            "case_execution_ticket_is_execution_authority": False,
            "authority": _authority(),
            "claims": _claims(),
        }


REQUEST_BODY_SHA256_FIELD: Final = "request_body_sha256"


def canonical_host_case_request_v2_body_bytes(request: HostQualificationCaseRequestV2) -> bytes:
    return _canonical_artifact_body_bytes(request, HostQualificationCaseRequestV2)


def canonical_host_case_request_v2_file_bytes(request: HostQualificationCaseRequestV2) -> bytes:
    return _canonical_artifact_file_bytes(
        request,
        HostQualificationCaseRequestV2,
        REQUEST_BODY_SHA256_FIELD,
    )


def parse_host_case_request_v2(
    raw: bytes,
    *,
    expected_file_sha256: str,
) -> HostQualificationCaseRequestV2:
    body_keys = frozenset(HostQualificationCaseRequestV2.__dataclass_fields__) | {
        "schema_version",
        "status",
        "acknowledgement_is_execution_authority",
        "case_execution_ticket_is_execution_authority",
        "authority",
        "claims",
    }
    body = _parse_artifact_file(
        raw,
        expected_file_sha256=expected_file_sha256,
        body_field=REQUEST_BODY_SHA256_FIELD,
        body_keys=body_keys,
        label="host case request",
    )
    item = dict(body)
    if (
        item.pop("schema_version") != HOST_CASE_REQUEST_SCHEMA_VERSION
        or item.pop("status") != "source_only_request_validated_non_authorizing"
        or item.pop("acknowledgement_is_execution_authority") is not False
        or item.pop("case_execution_ticket_is_execution_authority") is not False
        or item.pop("authority") != _authority()
        or item.pop("claims") != _claims()
    ):
        _fail("host case request envelope differs")
    for field in (
        "qualification_plan",
        "plan_issuance_receipt",
        "case_execution_ticket",
        "qualification_case_manifest",
        "joint_source_closure",
        "sealed_staging",
        "fresh_build",
        "build_context_receipt",
        "build_execution_receipt",
        "build_publication_receipt",
        "runtime_qualification_receipt",
        "host_provisioning_receipt",
        "algorithmic_measurement_intent",
        "storage_boundary_intent",
    ):
        item[field] = _artifact_identity(item[field], f"request {field}")
    for field in (
        "observation_registry",
        "algorithmic_contract",
        "publication_contract",
        "storage_contract",
        "host_executor",
        "full_resource_merger",
        "publisher",
        "native_atomic_producer",
        "in_container_driver",
    ):
        item[field] = _producer_identity(item[field], f"request {field}")
    ceilings = item.pop("declared_ceilings")
    if type(ceilings) is not list:
        _fail("request resource ceilings must be one list")
    pairs: list[tuple[str, int]] = []
    for child in ceilings:
        record = _require_exact_keys(
            child,
            frozenset({"field_name", "declared_ceiling"}),
            "request resource ceiling",
        )
        pairs.append((record["field_name"], record["declared_ceiling"]))
    result = HostQualificationCaseRequestV2(**item, declared_ceilings=tuple(pairs))
    if raw != canonical_host_case_request_v2_file_bytes(result):
        _fail("host case request canonical replay differs")
    return result


@dataclass(frozen=True, slots=True)
class HostQualificationCaseIntentV2:
    case_spine_sha256: str
    case_ordinal: int
    candidate_id: str
    candidate_family: Literal["local", "external", "adapter"]
    qualification_case_id: str
    request: ArtifactIdentityV2
    case_execution_ticket: ArtifactIdentityV2
    image_id: str
    algorithmic_measurement_intent: ArtifactIdentityV2
    storage_boundary_intent: ArtifactIdentityV2
    container_name: str
    retained_fd_policy_body_sha256: str
    cleanup_policy_body_sha256: str
    exact_acknowledgement_sha256: str
    intent_committed: bool
    same_case_retry_permitted: bool

    def __post_init__(self) -> None:
        _require_case(
            self.case_ordinal,
            self.candidate_id,
            self.candidate_family,
            self.qualification_case_id,
        )
        _require_sha256(self.case_spine_sha256, "intent case spine")
        _require_artifact_schema(self.request, HOST_CASE_REQUEST_SCHEMA_VERSION, "intent request")
        _require_artifact_schema(
            self.case_execution_ticket,
            CASE_EXECUTION_TICKET_SCHEMA_VERSION,
            "intent case ticket",
        )
        _require_image_id(self.image_id, "intent image")
        _require_artifact_schema(
            self.algorithmic_measurement_intent,
            ALGORITHMIC_MEASUREMENT_INTENT_SCHEMA_VERSION,
            "intent algorithmic measurement intent",
        )
        _require_artifact_schema(
            self.storage_boundary_intent,
            STORAGE_BOUNDARY_INTENT_SCHEMA_VERSION,
            "intent storage boundary intent",
        )
        name = _require_text(self.container_name, "intent container name")
        if (
            _CONTAINER_NAME_RE.fullmatch(name) is None
            or name != expected_container_name_v2(self.case_spine_sha256)
        ):
            _fail("intent container name differs from its full case spine")
        _require_sha256(self.retained_fd_policy_body_sha256, "retained-FD policy BODY")
        _require_sha256(self.cleanup_policy_body_sha256, "cleanup policy BODY")
        expected_ack = _sha256(HOST_EXECUTION_ACKNOWLEDGEMENT.encode("ascii"))
        if (
            _require_sha256(
                self.exact_acknowledgement_sha256,
                "intent acknowledgement digest",
            )
            != expected_ack
        ):
            _fail("intent acknowledgement digest differs")
        if _require_bool(self.intent_committed, "intent committed") is not True:
            _fail("case intent must represent one committed route")
        if _require_bool(self.same_case_retry_permitted, "intent retry") is not False:
            _fail("case intent cannot permit same-case retry")

    def to_body_dict(self) -> dict[str, Any]:
        return {
            "schema_version": HOST_CASE_INTENT_SCHEMA_VERSION,
            "status": "single_use_intent_committed_non_authorizing",
            **{
                "case_spine_sha256": self.case_spine_sha256,
                "case_ordinal": self.case_ordinal,
                "candidate_id": self.candidate_id,
                "candidate_family": self.candidate_family,
                "qualification_case_id": self.qualification_case_id,
                "request": self.request.to_dict(),
                "case_execution_ticket": self.case_execution_ticket.to_dict(),
                "image_id": self.image_id,
                "algorithmic_measurement_intent": (
                    self.algorithmic_measurement_intent.to_dict()
                ),
                "storage_boundary_intent": self.storage_boundary_intent.to_dict(),
                "container_name": self.container_name,
                "retained_fd_policy_body_sha256": self.retained_fd_policy_body_sha256,
                "cleanup_policy_body_sha256": self.cleanup_policy_body_sha256,
                "exact_acknowledgement_sha256": self.exact_acknowledgement_sha256,
                "intent_committed": self.intent_committed,
                "same_case_retry_permitted": self.same_case_retry_permitted,
            },
            "case_execution_ticket_is_execution_authority": False,
            "authority": _authority(),
            "claims": _claims(),
        }


INTENT_BODY_SHA256_FIELD: Final = "intent_body_sha256"


def canonical_host_case_intent_v2_body_bytes(intent: HostQualificationCaseIntentV2) -> bytes:
    return _canonical_artifact_body_bytes(intent, HostQualificationCaseIntentV2)


def canonical_host_case_intent_v2_file_bytes(intent: HostQualificationCaseIntentV2) -> bytes:
    return _canonical_artifact_file_bytes(
        intent,
        HostQualificationCaseIntentV2,
        INTENT_BODY_SHA256_FIELD,
    )


def parse_host_case_intent_v2(
    raw: bytes,
    *,
    expected_file_sha256: str,
) -> HostQualificationCaseIntentV2:
    body = _parse_artifact_file(
        raw,
        expected_file_sha256=expected_file_sha256,
        body_field=INTENT_BODY_SHA256_FIELD,
        body_keys=frozenset(HostQualificationCaseIntentV2.__dataclass_fields__)
        | {
            "schema_version",
            "status",
            "case_execution_ticket_is_execution_authority",
            "authority",
            "claims",
        },
        label="host case intent",
    )
    item = dict(body)
    if (
        item.pop("schema_version") != HOST_CASE_INTENT_SCHEMA_VERSION
        or item.pop("status") != "single_use_intent_committed_non_authorizing"
        or item.pop("case_execution_ticket_is_execution_authority") is not False
        or item.pop("authority") != _authority()
        or item.pop("claims") != _claims()
    ):
        _fail("host case intent envelope differs")
    for field in (
        "request",
        "case_execution_ticket",
        "algorithmic_measurement_intent",
        "storage_boundary_intent",
    ):
        item[field] = _artifact_identity(item[field], f"intent {field}")
    result = HostQualificationCaseIntentV2(**item)
    if raw != canonical_host_case_intent_v2_file_bytes(result):
        _fail("host case intent canonical replay differs")
    return result


def validate_host_request_intent_v2_chain(
    request: HostQualificationCaseRequestV2,
    intent: HostQualificationCaseIntentV2,
) -> None:
    if type(request) is not HostQualificationCaseRequestV2:
        raise TypeError("request must use the exact request-v2 type")
    if type(intent) is not HostQualificationCaseIntentV2:
        raise TypeError("intent must use the exact intent-v2 type")
    expected_request = ArtifactIdentityV2(
        HOST_CASE_REQUEST_SCHEMA_VERSION,
        _sha256(canonical_host_case_request_v2_file_bytes(request)),
        _sha256(canonical_host_case_request_v2_body_bytes(request)),
    )
    if intent.request != expected_request:
        _fail("intent does not bind the exact request FILE and BODY")
    expected = {
        "case_spine_sha256": request.case_spine_sha256,
        "case_ordinal": request.case_ordinal,
        "candidate_id": request.candidate_id,
        "candidate_family": request.candidate_family,
        "qualification_case_id": request.qualification_case_id,
        "container_name": request.container_name,
        "case_execution_ticket": request.case_execution_ticket,
        "image_id": request.image_id,
        "algorithmic_measurement_intent": request.algorithmic_measurement_intent,
        "storage_boundary_intent": request.storage_boundary_intent,
    }
    for field, value in expected.items():
        if getattr(intent, field) != value:
            _fail(f"request-to-intent projection differs for {field}")


@dataclass(frozen=True, slots=True)
class RetainedCgroupCounterFdV2:
    endpoint_name: str
    endpoint_device: int
    endpoint_inode: int
    open_monotonic_ns: int
    open_flags: tuple[str, ...]
    reset_performed: bool
    reopened: bool
    retained_through_post_container_remove_sample: bool

    def __post_init__(self) -> None:
        if _require_text(self.endpoint_name, "retained cgroup endpoint") not in (
            CGROUP_COUNTER_ENDPOINTS
        ):
            _fail("retained cgroup endpoint differs from the exact inventory")
        _require_int(self.endpoint_device, "retained FD device", minimum=1)
        _require_int(self.endpoint_inode, "retained FD inode", minimum=1)
        _require_int(self.open_monotonic_ns, "retained FD open time")
        expected_flags = (
            "O_CLOEXEC",
            "O_NOFOLLOW",
            "O_WRONLY" if self.endpoint_name == "cgroup.kill" else "O_RDONLY",
        )
        if (
            type(self.open_flags) is not tuple
            or any(type(item) is not str for item in self.open_flags)
            or self.open_flags != expected_flags
        ):
            _fail("retained cgroup FD flags differ")
        if _require_bool(self.reset_performed, "retained FD reset") is not False:
            _fail("fresh cgroup counters cannot be reset")
        if _require_bool(self.reopened, "retained FD reopen") is not False:
            _fail("retained cgroup counter FDs cannot be reopened")
        if (
            _require_bool(
                self.retained_through_post_container_remove_sample,
                "retained FD lifetime",
            )
            is not True
        ):
            _fail("counter FD must remain open through the post-container sample")

    def to_dict(self) -> dict[str, Any]:
        return {
            "endpoint_name": self.endpoint_name,
            "endpoint_device": self.endpoint_device,
            "endpoint_inode": self.endpoint_inode,
            "open_monotonic_ns": self.open_monotonic_ns,
            "open_flags": list(self.open_flags),
            "reset_performed": self.reset_performed,
            "reopened": self.reopened,
            "retained_through_post_container_remove_sample": (
                self.retained_through_post_container_remove_sample
            ),
        }


def _retained_fd_from_dict(value: object) -> RetainedCgroupCounterFdV2:
    item = _require_exact_keys(
        value,
        frozenset(RetainedCgroupCounterFdV2.__dataclass_fields__),
        "retained cgroup FD",
    )
    flags = item.pop("open_flags")
    if type(flags) is not list:
        _fail("retained cgroup FD flags must be one list")
    return RetainedCgroupCounterFdV2(**item, open_flags=tuple(flags))


def validate_retained_cgroup_fd_inventory_v2(
    value: object,
) -> tuple[RetainedCgroupCounterFdV2, ...]:
    if type(value) is not tuple or any(
        type(item) is not RetainedCgroupCounterFdV2 for item in cast(tuple[object, ...], value)
    ):
        _fail("retained cgroup FDs must use one exact tuple")
    exact = cast(tuple[RetainedCgroupCounterFdV2, ...], value)
    if tuple(item.endpoint_name for item in exact) != CGROUP_COUNTER_ENDPOINTS:
        _fail("retained cgroup FD inventory order or coverage differs")
    if len({(item.endpoint_device, item.endpoint_inode) for item in exact}) != len(exact):
        _fail("retained cgroup FD endpoint identities alias")
    return exact


def retained_cgroup_fd_inventory_sha256_v2(
    value: tuple[RetainedCgroupCounterFdV2, ...],
) -> str:
    exact = validate_retained_cgroup_fd_inventory_v2(value)
    return _body_sha256({"counter_fds": [item.to_dict() for item in exact]})


@dataclass(frozen=True, slots=True)
class HostCgroupCaseIdentityV2:
    """Strong directory identity for the supported cgroupfs-qualified-host route."""

    case_spine_sha256: str
    route_kind: Literal["cgroupfs_qualified_host"]
    delegate_root_path: str
    delegate_cgroup_device: int
    delegate_cgroup_inode: int
    case_cgroup_path: str
    case_cgroup_device: int
    case_cgroup_inode: int
    docker_cgroup_parent: str
    enabled_controllers: tuple[str, ...]
    subtree_control: tuple[str, ...]
    max_depth: int
    max_descendants: int

    def __post_init__(self) -> None:
        spine = _require_sha256(self.case_spine_sha256, "cgroup-case spine")
        if _require_text(self.route_kind, "cgroup-case route") != "cgroupfs_qualified_host":
            _fail("only the cgroupfs-qualified-host route is supported")
        if self.delegate_root_path != CGROUP_DELEGATE_ROOT_PATH:
            _fail("cgroup delegate root path differs from the qualified-host route")
        expected_case_path = f"{CGROUP_DELEGATE_ROOT_PATH}/case-{spine}"
        if (
            type(self.case_cgroup_path) is not str
            or _CGROUP_PATH_RE.fullmatch(self.case_cgroup_path) is None
            or self.case_cgroup_path != expected_case_path
        ):
            _fail("case cgroup path differs from the complete case spine")
        expected_parent = f"{CGROUP_DELEGATE_PARENT_ARGUMENT}/case-{spine}"
        if self.docker_cgroup_parent != expected_parent:
            _fail("Docker cgroup-parent differs from the exact case cgroup path")
        delegate_device = _require_int(
            self.delegate_cgroup_device,
            "delegate cgroup device",
            minimum=1,
        )
        case_device = _require_int(
            self.case_cgroup_device,
            "case cgroup device",
            minimum=1,
        )
        delegate_inode = _require_int(
            self.delegate_cgroup_inode,
            "delegate cgroup inode",
            minimum=1,
        )
        case_inode = _require_int(
            self.case_cgroup_inode,
            "case cgroup inode",
            minimum=1,
        )
        if delegate_device != case_device or delegate_inode == case_inode:
            _fail("delegate and case cgroups must be distinct directories on one device")
        expected_controllers = ("cpu", "memory", "pids")
        if (
            type(self.enabled_controllers) is not tuple
            or self.enabled_controllers != expected_controllers
            or type(self.subtree_control) is not tuple
            or self.subtree_control != expected_controllers
        ):
            _fail("cgroup controller delegation differs from cpu/memory/pids")
        if (
            _require_int(self.max_depth, "case cgroup max depth", maximum=1) != 1
            or _require_int(
                self.max_descendants,
                "case cgroup max descendants",
                maximum=1,
            )
            != 1
        ):
            _fail("case cgroup must permit exactly one direct container child")

    def to_dict(self) -> dict[str, Any]:
        return {
            field: (
                list(getattr(self, field))
                if field in {"enabled_controllers", "subtree_control"}
                else getattr(self, field)
            )
            for field in self.__dataclass_fields__
        }


def _cgroup_case_identity_from_dict(value: object) -> HostCgroupCaseIdentityV2:
    item = _require_exact_keys(
        value,
        frozenset(HostCgroupCaseIdentityV2.__dataclass_fields__),
        "cgroup case identity",
    )
    controllers = item.pop("enabled_controllers")
    subtree = item.pop("subtree_control")
    if type(controllers) is not list or type(subtree) is not list:
        _fail("cgroup controller identities must be exact lists")
    return HostCgroupCaseIdentityV2(
        **item,
        enabled_controllers=tuple(controllers),
        subtree_control=tuple(subtree),
    )


def cgroup_case_identity_sha256_v2(value: HostCgroupCaseIdentityV2) -> str:
    if type(value) is not HostCgroupCaseIdentityV2:
        raise TypeError("cgroup case identity must use the exact v2 type")
    return _body_sha256(value.to_dict())


@dataclass(frozen=True, slots=True)
class CgroupSampleFactsV2:
    monotonic_ns: int
    cgroup_identity_sha256: str
    retained_fd_set_sha256: str
    cpu_usage_usec: int
    memory_current_bytes: int
    memory_peak_bytes: int
    memory_oom_kill_count: int
    pids_current: int
    pids_peak: int
    pids_max_event_count: int
    populated: bool
    nr_descendants: int
    nr_dying_descendants: int

    def __post_init__(self) -> None:
        _require_int(self.monotonic_ns, "cgroup sample monotonic time")
        _require_sha256(self.cgroup_identity_sha256, "cgroup sample identity")
        _require_sha256(self.retained_fd_set_sha256, "sample retained-FD set")
        for field in (
            "cpu_usage_usec",
            "memory_current_bytes",
            "memory_peak_bytes",
            "memory_oom_kill_count",
            "pids_current",
            "pids_peak",
            "pids_max_event_count",
            "nr_descendants",
            "nr_dying_descendants",
        ):
            _require_int(getattr(self, field), f"cgroup sample {field}")
        _require_bool(self.populated, "cgroup populated")
        if self.memory_peak_bytes < self.memory_current_bytes:
            _fail("cgroup memory peak cannot be below memory current")
        if self.pids_peak < self.pids_current:
            _fail("cgroup pids peak cannot be below pids current")

    def to_dict(self) -> dict[str, Any]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}


_MONOTONIC_CGROUP_COUNTER_FIELDS: Final = (
    "cpu_usage_usec",
    "memory_peak_bytes",
    "memory_oom_kill_count",
    "pids_peak",
    "pids_max_event_count",
)


def _validate_cgroup_sample_progression_v2(
    earlier: CgroupSampleFactsV2,
    later: CgroupSampleFactsV2,
    label: str,
) -> None:
    if type(earlier) is not CgroupSampleFactsV2 or type(later) is not CgroupSampleFactsV2:
        raise TypeError("cgroup sample progression requires exact v2 fact types")
    if (
        later.retained_fd_set_sha256 != earlier.retained_fd_set_sha256
        or later.cgroup_identity_sha256 != earlier.cgroup_identity_sha256
        or later.monotonic_ns <= earlier.monotonic_ns
        or any(
            getattr(later, field) < getattr(earlier, field)
            for field in _MONOTONIC_CGROUP_COUNTER_FIELDS
        )
    ):
        _fail(f"{label} changes retained identity, chronology, or monotonic counters")


def _sample_facts_from_dict(value: object) -> CgroupSampleFactsV2:
    item = _require_exact_keys(
        value,
        frozenset(CgroupSampleFactsV2.__dataclass_fields__),
        "cgroup sample facts",
    )
    return CgroupSampleFactsV2(**item)


@dataclass(frozen=True, slots=True)
class HostInitialCgroupSampleV2:
    case_spine_sha256: str
    intent: ArtifactIdentityV2
    cgroup_case_identity: HostCgroupCaseIdentityV2
    counter_fds: tuple[RetainedCgroupCounterFdV2, ...]
    facts: CgroupSampleFactsV2

    def __post_init__(self) -> None:
        _require_sha256(self.case_spine_sha256, "initial sample case spine")
        _require_artifact_schema(self.intent, HOST_CASE_INTENT_SCHEMA_VERSION, "initial intent")
        if type(self.cgroup_case_identity) is not HostCgroupCaseIdentityV2:
            _fail("initial cgroup case identity type differs")
        if self.cgroup_case_identity.case_spine_sha256 != self.case_spine_sha256:
            _fail("initial cgroup case identity crosses case spines")
        if type(self.facts) is not CgroupSampleFactsV2:
            _fail("initial sample facts type differs")
        fds = validate_retained_cgroup_fd_inventory_v2(self.counter_fds)
        if self.facts.retained_fd_set_sha256 != retained_cgroup_fd_inventory_sha256_v2(fds):
            _fail("initial sample retained-FD inventory digest differs")
        if self.facts.cgroup_identity_sha256 != cgroup_case_identity_sha256_v2(
            self.cgroup_case_identity
        ):
            _fail("initial sample cgroup identity is not derived from its directory evidence")
        forbidden_inodes = {
            self.cgroup_case_identity.delegate_cgroup_inode,
            self.cgroup_case_identity.case_cgroup_inode,
        }
        if any(
            item.endpoint_device != self.cgroup_case_identity.case_cgroup_device
            or item.endpoint_inode in forbidden_inodes
            for item in fds
        ):
            _fail("retained endpoint identity is outside the exact case cgroup directory")
        if any(item.open_monotonic_ns >= self.facts.monotonic_ns for item in fds):
            _fail("all retained FDs must open before the initial sample")
        if (
            self.facts.populated
            or self.facts.cpu_usage_usec
            or self.facts.memory_current_bytes
            or self.facts.memory_peak_bytes
            or self.facts.memory_oom_kill_count
            or self.facts.pids_current
            or self.facts.pids_peak
            or self.facts.pids_max_event_count
            or self.facts.nr_descendants
            or self.facts.nr_dying_descendants
        ):
            _fail("initial sample is not an exact newly created empty cgroup")

    def to_body_dict(self) -> dict[str, Any]:
        return {
            "schema_version": HOST_INITIAL_CGROUP_SAMPLE_SCHEMA_VERSION,
            "case_spine_sha256": self.case_spine_sha256,
            "intent": self.intent.to_dict(),
            "cgroup_case_identity": self.cgroup_case_identity.to_dict(),
            "counter_fds": [item.to_dict() for item in self.counter_fds],
            "facts": self.facts.to_dict(),
        }


INITIAL_SAMPLE_BODY_SHA256_FIELD: Final = "initial_sample_body_sha256"


def canonical_host_initial_cgroup_sample_v2_body_bytes(
    sample: HostInitialCgroupSampleV2,
) -> bytes:
    return _canonical_artifact_body_bytes(sample, HostInitialCgroupSampleV2)


def canonical_host_initial_cgroup_sample_v2_file_bytes(
    sample: HostInitialCgroupSampleV2,
) -> bytes:
    return _canonical_artifact_file_bytes(
        sample,
        HostInitialCgroupSampleV2,
        INITIAL_SAMPLE_BODY_SHA256_FIELD,
    )


def parse_host_initial_cgroup_sample_v2(
    raw: bytes,
    *,
    expected_file_sha256: str,
) -> HostInitialCgroupSampleV2:
    body = _parse_artifact_file(
        raw,
        expected_file_sha256=expected_file_sha256,
        body_field=INITIAL_SAMPLE_BODY_SHA256_FIELD,
        body_keys=frozenset({"schema_version", *HostInitialCgroupSampleV2.__dataclass_fields__}),
        label="initial cgroup sample",
    )
    item = dict(body)
    if item.pop("schema_version") != HOST_INITIAL_CGROUP_SAMPLE_SCHEMA_VERSION:
        _fail("initial cgroup sample schema differs")
    item["intent"] = _artifact_identity(item["intent"], "initial sample intent")
    item["cgroup_case_identity"] = _cgroup_case_identity_from_dict(
        item["cgroup_case_identity"]
    )
    fds = item.pop("counter_fds")
    if type(fds) is not list:
        _fail("initial cgroup counter FDs must be one list")
    item["counter_fds"] = tuple(_retained_fd_from_dict(child) for child in fds)
    item["facts"] = _sample_facts_from_dict(item["facts"])
    result = HostInitialCgroupSampleV2(**item)
    if raw != canonical_host_initial_cgroup_sample_v2_file_bytes(result):
        _fail("initial cgroup sample canonical replay differs")
    return result


@dataclass(frozen=True, slots=True)
class HostReadyV2:
    case_spine_sha256: str
    case_ordinal: int
    candidate_id: str
    candidate_family: Literal["local", "external", "adapter"]
    qualification_case_id: str
    intent: ArtifactIdentityV2
    algorithmic_measurement_intent: ArtifactIdentityV2
    storage_boundary_intent: ArtifactIdentityV2
    initial_cgroup_sample: ArtifactIdentityV2
    retained_fd_set_sha256: str
    cgroup_identity_sha256: str
    container_identity_sha256: str
    container_id: str
    container_name: str
    container_cgroup_path: str
    container_cgroup_device: int
    container_cgroup_inode: int
    host_pid: int
    host_process_start_time_ticks: int
    inner_pid: int
    ready_monotonic_ns: int
    candidate_code_loaded: bool
    go_committed: bool

    def __post_init__(self) -> None:
        _require_case(
            self.case_ordinal,
            self.candidate_id,
            self.candidate_family,
            self.qualification_case_id,
        )
        _require_sha256(self.case_spine_sha256, "READY case spine")
        _require_artifact_schema(self.intent, HOST_CASE_INTENT_SCHEMA_VERSION, "READY intent")
        _require_artifact_schema(
            self.algorithmic_measurement_intent,
            ALGORITHMIC_MEASUREMENT_INTENT_SCHEMA_VERSION,
            "READY algorithmic intent",
        )
        _require_artifact_schema(
            self.storage_boundary_intent,
            STORAGE_BOUNDARY_INTENT_SCHEMA_VERSION,
            "READY storage intent",
        )
        _require_artifact_schema(
            self.initial_cgroup_sample,
            HOST_INITIAL_CGROUP_SAMPLE_SCHEMA_VERSION,
            "READY initial cgroup sample",
        )
        for value, label in (
            (self.retained_fd_set_sha256, "READY retained-FD set"),
            (self.cgroup_identity_sha256, "READY cgroup identity"),
            (self.container_identity_sha256, "READY container identity"),
        ):
            _require_sha256(value, label)
        if (
            type(self.container_id) is not str
            or _CONTAINER_ID_RE.fullmatch(self.container_id) is None
            or self.container_id == "0" * 64
        ):
            _fail("READY container ID differs")
        name = _require_text(self.container_name, "READY container name")
        if (
            _CONTAINER_NAME_RE.fullmatch(name) is None
            or name != expected_container_name_v2(self.case_spine_sha256)
        ):
            _fail("READY container name differs from the complete case spine")
        if self.container_identity_sha256 != container_runtime_identity_sha256_v2(
            self.case_spine_sha256,
            name,
            self.container_id,
        ):
            _fail("READY runtime-container identity is not derived from name and ID")
        if (
            type(self.container_cgroup_path) is not str
            or _CGROUP_PATH_RE.fullmatch(self.container_cgroup_path) is None
        ):
            _fail("READY container cgroup path differs")
        _require_int(self.container_cgroup_device, "READY container cgroup device", minimum=1)
        _require_int(self.container_cgroup_inode, "READY container cgroup inode", minimum=1)
        _require_int(self.host_pid, "READY host PID", minimum=1)
        _require_int(self.host_process_start_time_ticks, "READY process start", minimum=1)
        if _require_int(self.inner_pid, "READY inner PID", minimum=1, maximum=1) != 1:
            _fail("READY inner PID must be exact namespace init PID 1")
        _require_int(self.ready_monotonic_ns, "READY monotonic time")
        if _require_bool(self.candidate_code_loaded, "READY candidate loaded") is not False:
            _fail("READY cannot claim candidate code loaded before GO")
        if _require_bool(self.go_committed, "READY GO committed") is not False:
            _fail("READY must precede GO")

    def to_body_dict(self) -> dict[str, Any]:
        return {
            "schema_version": HOST_READY_SCHEMA_VERSION,
            "status": "driver_ready_waiting_for_one_way_go_non_authorizing",
            **{
                field: (
                    getattr(self, field).to_dict()
                    if type(getattr(self, field)) is ArtifactIdentityV2
                    else getattr(self, field)
                )
                for field in self.__dataclass_fields__
            },
            "authority": _authority(),
            "claims": _claims(),
        }


READY_BODY_SHA256_FIELD: Final = "ready_body_sha256"


def canonical_host_ready_v2_body_bytes(ready: HostReadyV2) -> bytes:
    return _canonical_artifact_body_bytes(ready, HostReadyV2)


def canonical_host_ready_v2_file_bytes(ready: HostReadyV2) -> bytes:
    return _canonical_artifact_file_bytes(ready, HostReadyV2, READY_BODY_SHA256_FIELD)


def parse_host_ready_v2(raw: bytes, *, expected_file_sha256: str) -> HostReadyV2:
    body = _parse_artifact_file(
        raw,
        expected_file_sha256=expected_file_sha256,
        body_field=READY_BODY_SHA256_FIELD,
        body_keys=frozenset(HostReadyV2.__dataclass_fields__)
        | {"schema_version", "status", "authority", "claims"},
        label="host READY",
    )
    item = dict(body)
    if (
        item.pop("schema_version") != HOST_READY_SCHEMA_VERSION
        or item.pop("status") != "driver_ready_waiting_for_one_way_go_non_authorizing"
        or item.pop("authority") != _authority()
        or item.pop("claims") != _claims()
    ):
        _fail("host READY envelope differs")
    for field in (
        "intent",
        "algorithmic_measurement_intent",
        "storage_boundary_intent",
        "initial_cgroup_sample",
    ):
        item[field] = _artifact_identity(item[field], f"READY {field}")
    result = HostReadyV2(**item)
    if raw != canonical_host_ready_v2_file_bytes(result):
        _fail("host READY canonical replay differs")
    return result


@dataclass(frozen=True, slots=True)
class HostObserverAnchorV2:
    case_spine_sha256: str
    ready: ArtifactIdentityV2
    initial_cgroup_sample: ArtifactIdentityV2
    retained_fd_set_sha256: str
    cgroup_identity_sha256: str
    observer_descriptor_sha256: str
    observer_source_sha256: str
    observer_started_monotonic_ns: int
    observation_loss_detected: bool

    def __post_init__(self) -> None:
        _require_sha256(self.case_spine_sha256, "observer-anchor case spine")
        _require_artifact_schema(self.ready, HOST_READY_SCHEMA_VERSION, "observer-anchor READY")
        _require_artifact_schema(
            self.initial_cgroup_sample,
            HOST_INITIAL_CGROUP_SAMPLE_SCHEMA_VERSION,
            "observer-anchor initial sample",
        )
        for value, label in (
            (self.retained_fd_set_sha256, "observer-anchor retained-FD set"),
            (self.cgroup_identity_sha256, "observer-anchor cgroup identity"),
            (self.observer_descriptor_sha256, "observer descriptor"),
            (self.observer_source_sha256, "observer source"),
        ):
            _require_sha256(value, label)
        _require_int(self.observer_started_monotonic_ns, "observer start time")
        if _require_bool(self.observation_loss_detected, "observation loss") is not False:
            _fail("observer anchor cannot report observation loss")

    def to_body_dict(self) -> dict[str, Any]:
        return {
            "schema_version": HOST_OBSERVER_ANCHOR_SCHEMA_VERSION,
            **{
                "case_spine_sha256": self.case_spine_sha256,
                "ready": self.ready.to_dict(),
                "initial_cgroup_sample": self.initial_cgroup_sample.to_dict(),
                "retained_fd_set_sha256": self.retained_fd_set_sha256,
                "cgroup_identity_sha256": self.cgroup_identity_sha256,
                "observer_descriptor_sha256": self.observer_descriptor_sha256,
                "observer_source_sha256": self.observer_source_sha256,
                "observer_started_monotonic_ns": self.observer_started_monotonic_ns,
                "observation_loss_detected": self.observation_loss_detected,
            },
        }


OBSERVER_ANCHOR_BODY_SHA256_FIELD: Final = "observer_anchor_body_sha256"


def canonical_host_observer_anchor_v2_body_bytes(anchor: HostObserverAnchorV2) -> bytes:
    return _canonical_artifact_body_bytes(anchor, HostObserverAnchorV2)


def canonical_host_observer_anchor_v2_file_bytes(anchor: HostObserverAnchorV2) -> bytes:
    return _canonical_artifact_file_bytes(
        anchor,
        HostObserverAnchorV2,
        OBSERVER_ANCHOR_BODY_SHA256_FIELD,
    )


def parse_host_observer_anchor_v2(
    raw: bytes,
    *,
    expected_file_sha256: str,
) -> HostObserverAnchorV2:
    body = _parse_artifact_file(
        raw,
        expected_file_sha256=expected_file_sha256,
        body_field=OBSERVER_ANCHOR_BODY_SHA256_FIELD,
        body_keys=frozenset({"schema_version", *HostObserverAnchorV2.__dataclass_fields__}),
        label="host observer anchor",
    )
    item = dict(body)
    if item.pop("schema_version") != HOST_OBSERVER_ANCHOR_SCHEMA_VERSION:
        _fail("host observer-anchor schema differs")
    item["ready"] = _artifact_identity(item["ready"], "observer-anchor READY")
    item["initial_cgroup_sample"] = _artifact_identity(
        item["initial_cgroup_sample"],
        "observer-anchor initial sample",
    )
    result = HostObserverAnchorV2(**item)
    if raw != canonical_host_observer_anchor_v2_file_bytes(result):
        _fail("host observer-anchor canonical replay differs")
    return result


@dataclass(frozen=True, slots=True)
class HostGoCommitmentV2:
    case_spine_sha256: str
    ready: ArtifactIdentityV2
    observer_anchor: ArtifactIdentityV2
    retained_fd_set_sha256: str
    cgroup_identity_sha256: str
    go_payload_sha256: str
    go_committed_monotonic_ns: int
    go_commit_count: int
    one_way: bool
    same_case_retry_permitted: bool

    def __post_init__(self) -> None:
        _require_sha256(self.case_spine_sha256, "GO case spine")
        _require_artifact_schema(self.ready, HOST_READY_SCHEMA_VERSION, "GO READY")
        _require_artifact_schema(
            self.observer_anchor,
            HOST_OBSERVER_ANCHOR_SCHEMA_VERSION,
            "GO observer anchor",
        )
        for value, label in (
            (self.retained_fd_set_sha256, "GO retained-FD set"),
            (self.cgroup_identity_sha256, "GO cgroup identity"),
            (self.go_payload_sha256, "GO payload"),
        ):
            _require_sha256(value, label)
        _require_int(self.go_committed_monotonic_ns, "GO monotonic time")
        if _require_int(self.go_commit_count, "GO commitment count", maximum=1) != 1:
            _fail("GO commitment count must be exact one")
        if _require_bool(self.one_way, "GO one-way") is not True:
            _fail("GO must be one-way")
        if _require_bool(self.same_case_retry_permitted, "GO retry") is not False:
            _fail("GO cannot permit same-case retry")

    def to_body_dict(self) -> dict[str, Any]:
        return {
            "schema_version": HOST_GO_SCHEMA_VERSION,
            "status": "one_way_go_committed_non_authorizing",
            "case_spine_sha256": self.case_spine_sha256,
            "ready": self.ready.to_dict(),
            "observer_anchor": self.observer_anchor.to_dict(),
            "retained_fd_set_sha256": self.retained_fd_set_sha256,
            "cgroup_identity_sha256": self.cgroup_identity_sha256,
            "go_payload_sha256": self.go_payload_sha256,
            "go_committed_monotonic_ns": self.go_committed_monotonic_ns,
            "go_commit_count": self.go_commit_count,
            "one_way": self.one_way,
            "same_case_retry_permitted": self.same_case_retry_permitted,
            "authority": _authority(),
            "claims": _claims(),
        }


GO_BODY_SHA256_FIELD: Final = "go_body_sha256"


def canonical_host_go_v2_body_bytes(go: HostGoCommitmentV2) -> bytes:
    return _canonical_artifact_body_bytes(go, HostGoCommitmentV2)


def canonical_host_go_v2_file_bytes(go: HostGoCommitmentV2) -> bytes:
    return _canonical_artifact_file_bytes(go, HostGoCommitmentV2, GO_BODY_SHA256_FIELD)


def parse_host_go_v2(raw: bytes, *, expected_file_sha256: str) -> HostGoCommitmentV2:
    body = _parse_artifact_file(
        raw,
        expected_file_sha256=expected_file_sha256,
        body_field=GO_BODY_SHA256_FIELD,
        body_keys=frozenset(HostGoCommitmentV2.__dataclass_fields__)
        | {"schema_version", "status", "authority", "claims"},
        label="host GO commitment",
    )
    item = dict(body)
    if (
        item.pop("schema_version") != HOST_GO_SCHEMA_VERSION
        or item.pop("status") != "one_way_go_committed_non_authorizing"
        or item.pop("authority") != _authority()
        or item.pop("claims") != _claims()
    ):
        _fail("host GO envelope differs")
    item["ready"] = _artifact_identity(item["ready"], "GO READY")
    item["observer_anchor"] = _artifact_identity(item["observer_anchor"], "GO anchor")
    result = HostGoCommitmentV2(**item)
    if raw != canonical_host_go_v2_file_bytes(result):
        _fail("host GO canonical replay differs")
    return result


def validate_host_ready_anchor_go_v2_chain(
    intent: HostQualificationCaseIntentV2,
    initial_sample: HostInitialCgroupSampleV2,
    ready: HostReadyV2,
    anchor: HostObserverAnchorV2,
    go: HostGoCommitmentV2,
) -> None:
    expected_types = (
        (intent, HostQualificationCaseIntentV2),
        (initial_sample, HostInitialCgroupSampleV2),
        (ready, HostReadyV2),
        (anchor, HostObserverAnchorV2),
        (go, HostGoCommitmentV2),
    )
    if any(type(value) is not expected for value, expected in expected_types):
        raise TypeError("READY/anchor/GO chain requires exact v2 types")
    intent_identity = ArtifactIdentityV2(
        HOST_CASE_INTENT_SCHEMA_VERSION,
        _sha256(canonical_host_case_intent_v2_file_bytes(intent)),
        _sha256(canonical_host_case_intent_v2_body_bytes(intent)),
    )
    initial_identity = ArtifactIdentityV2(
        HOST_INITIAL_CGROUP_SAMPLE_SCHEMA_VERSION,
        _sha256(canonical_host_initial_cgroup_sample_v2_file_bytes(initial_sample)),
        _sha256(canonical_host_initial_cgroup_sample_v2_body_bytes(initial_sample)),
    )
    ready_identity = ArtifactIdentityV2(
        HOST_READY_SCHEMA_VERSION,
        _sha256(canonical_host_ready_v2_file_bytes(ready)),
        _sha256(canonical_host_ready_v2_body_bytes(ready)),
    )
    anchor_identity = ArtifactIdentityV2(
        HOST_OBSERVER_ANCHOR_SCHEMA_VERSION,
        _sha256(canonical_host_observer_anchor_v2_file_bytes(anchor)),
        _sha256(canonical_host_observer_anchor_v2_body_bytes(anchor)),
    )
    if initial_sample.intent != intent_identity or ready.intent != intent_identity:
        _fail("initial sample or READY is cross-wired from intent")
    if any(
        value.case_spine_sha256 != intent.case_spine_sha256
        for value in (initial_sample, ready, anchor, go)
    ):
        _fail("READY/anchor/GO prefix crosses the committed case spine")
    for field in (
        "case_ordinal",
        "candidate_id",
        "candidate_family",
        "qualification_case_id",
    ):
        if getattr(ready, field) != getattr(intent, field):
            _fail(f"READY case projection differs for {field}")
    if (
        ready.algorithmic_measurement_intent != intent.algorithmic_measurement_intent
        or ready.storage_boundary_intent != intent.storage_boundary_intent
        or ready.container_name != intent.container_name
        or ready.container_name != expected_container_name_v2(intent.case_spine_sha256)
    ):
        _fail("READY intent or deterministic container-name projection differs")
    if (
        ready.initial_cgroup_sample != initial_identity
        or anchor.initial_cgroup_sample != initial_identity
    ):
        _fail("READY or observer anchor is cross-wired from the initial sample")
    if anchor.ready != ready_identity or go.ready != ready_identity:
        _fail("observer anchor or GO is cross-wired from READY")
    if go.observer_anchor != anchor_identity:
        _fail("GO is cross-wired from the observer anchor")
    if (
        len(
            {
                ready.retained_fd_set_sha256,
                anchor.retained_fd_set_sha256,
                go.retained_fd_set_sha256,
            }
        )
        != 1
        or len(
            {
                ready.cgroup_identity_sha256,
                anchor.cgroup_identity_sha256,
                go.cgroup_identity_sha256,
            }
        )
        != 1
        or ready.retained_fd_set_sha256 != initial_sample.facts.retained_fd_set_sha256
        or ready.cgroup_identity_sha256 != initial_sample.facts.cgroup_identity_sha256
    ):
        _fail("READY/anchor/GO retained-FD or cgroup identity drifted")
    case_identity = initial_sample.cgroup_case_identity
    if (
        ready.container_cgroup_path
        != f"{case_identity.case_cgroup_path}/{ready.container_id}"
        or ready.container_cgroup_device != case_identity.case_cgroup_device
        or ready.container_cgroup_inode
        in {
            case_identity.delegate_cgroup_inode,
            case_identity.case_cgroup_inode,
            *(item.endpoint_inode for item in initial_sample.counter_fds),
        }
    ):
        _fail("READY container is not the one direct child of the case cgroup")
    if not (
        initial_sample.facts.monotonic_ns
        < ready.ready_monotonic_ns
        < anchor.observer_started_monotonic_ns
        < go.go_committed_monotonic_ns
    ):
        _fail("READY/anchor/GO monotonic chronology differs")


type BoundaryStateV2 = Literal["not_started", "commit_uncertain", "committed"]
type BoundaryCountStateV2 = Literal["exact", "uncertain"]


def _derived_boundary_state(
    completed: tuple[str, ...],
    failure_phase: str | None,
    failure_effect_state: str | None,
    phase: str,
) -> BoundaryStateV2:
    if phase in completed:
        return "committed"
    if failure_phase == phase and failure_effect_state == "commit_uncertain":
        return "commit_uncertain"
    return "not_started"


def _derived_count(
    completed: tuple[str, ...],
    failure_phase: str | None,
    failure_effect_state: str | None,
    phase: str,
) -> tuple[BoundaryCountStateV2, int | None]:
    state = _derived_boundary_state(completed, failure_phase, failure_effect_state, phase)
    if state == "committed":
        return "exact", 1
    if state == "commit_uncertain":
        return "uncertain", None
    return "exact", 0


@dataclass(frozen=True, slots=True)
class HostOperationalFrontierV2:
    case_spine_sha256: str
    completed_phases: tuple[str, ...]
    failure_phase: str | None
    failure_effect_state: Literal["failed_before_commit", "commit_uncertain"] | None
    container_create_state: BoundaryStateV2
    container_start_state: BoundaryStateV2
    workload_start_state: BoundaryStateV2
    workload_exit_state: BoundaryStateV2
    container_create_count_state: BoundaryCountStateV2
    container_create_count: int | None
    container_start_count_state: BoundaryCountStateV2
    container_start_count: int | None
    workload_start_count_state: BoundaryCountStateV2
    workload_start_count: int | None
    workload_exit_count_state: BoundaryCountStateV2
    workload_exit_count: int | None
    attempt_count_state: BoundaryCountStateV2
    attempt_count: int | None
    failure_count: int
    case_consumed: bool
    same_case_retry_permitted: bool

    def __post_init__(self) -> None:
        _require_sha256(self.case_spine_sha256, "operational frontier case spine")
        if (
            type(self.completed_phases) is not tuple
            or any(type(item) is not str for item in self.completed_phases)
            or self.completed_phases != OPERATIONAL_PHASES[: len(self.completed_phases)]
        ):
            _fail("operational frontier must be one exact phase prefix")
        if self.failure_phase is None:
            if self.failure_effect_state is not None:
                _fail("success frontier cannot carry a failure effect")
            if self.completed_phases != OPERATIONAL_PHASES:
                _fail("nonfailure operational frontier must be complete")
        elif (
            len(self.completed_phases) >= len(OPERATIONAL_PHASES)
            or type(self.failure_phase) is not str
            or self.failure_phase != OPERATIONAL_PHASES[len(self.completed_phases)]
            or type(self.failure_effect_state) is not str
            or self.failure_effect_state not in {"failed_before_commit", "commit_uncertain"}
        ):
            _fail("operational failure must be the exact next phase and effect")
        for field in (
            "container_create_state",
            "container_start_state",
            "workload_start_state",
            "workload_exit_state",
            "container_create_count_state",
            "container_start_count_state",
            "workload_start_count_state",
            "workload_exit_count_state",
            "attempt_count_state",
        ):
            _require_text(getattr(self, field), f"operational frontier {field}")
        expected_states = {
            "container_create_state": _derived_boundary_state(
                self.completed_phases,
                self.failure_phase,
                self.failure_effect_state,
                "container_created",
            ),
            "container_start_state": _derived_boundary_state(
                self.completed_phases,
                self.failure_phase,
                self.failure_effect_state,
                "container_started",
            ),
            "workload_start_state": _derived_boundary_state(
                self.completed_phases,
                self.failure_phase,
                self.failure_effect_state,
                "workload_started",
            ),
            "workload_exit_state": _derived_boundary_state(
                self.completed_phases,
                self.failure_phase,
                self.failure_effect_state,
                "workload_exited",
            ),
        }
        for field, expected in expected_states.items():
            if getattr(self, field) != expected:
                _fail(f"operational frontier derived {field} differs")
        expected_counts = {
            "container_create": _derived_count(
                self.completed_phases,
                self.failure_phase,
                self.failure_effect_state,
                "container_created",
            ),
            "container_start": _derived_count(
                self.completed_phases,
                self.failure_phase,
                self.failure_effect_state,
                "container_started",
            ),
            "workload_start": _derived_count(
                self.completed_phases,
                self.failure_phase,
                self.failure_effect_state,
                "workload_started",
            ),
            "workload_exit": _derived_count(
                self.completed_phases,
                self.failure_phase,
                self.failure_effect_state,
                "workload_exited",
            ),
        }
        for prefix, (state, count) in expected_counts.items():
            actual_count = getattr(self, f"{prefix}_count")
            if actual_count is not None:
                _require_int(actual_count, f"operational frontier {prefix} count", maximum=1)
            if (
                getattr(self, f"{prefix}_count_state") != state
                or actual_count != count
            ):
                _fail(f"operational frontier {prefix} count differs")
        if (
            self.attempt_count_state != self.workload_start_count_state
            or self.attempt_count != self.workload_start_count
        ):
            _fail("attempt count must equal the exact workload-start count state")
        expected_failure_count = 0 if self.failure_phase is None else 1
        if (
            _require_int(
                self.failure_count,
                "operational frontier failure count",
                maximum=1,
            )
            != expected_failure_count
        ):
            _fail("operational failure count differs")
        if _require_bool(self.case_consumed, "operational case consumed") is not True:
            _fail("every committed frontier consumes the single-use case")
        if _require_bool(self.same_case_retry_permitted, "operational retry") is not False:
            _fail("operational frontier cannot permit same-case retry")

    @property
    def succeeded(self) -> bool:
        return self.failure_phase is None

    def to_body_dict(self) -> dict[str, Any]:
        return {
            "schema_version": HOST_OPERATIONAL_FRONTIER_SCHEMA_VERSION,
            "status": "operational_frontier_recorded_nonretryable_non_authorizing",
            **{
                field: (
                    list(getattr(self, field))
                    if field == "completed_phases"
                    else getattr(self, field)
                )
                for field in self.__dataclass_fields__
            },
            "claims": _claims(),
        }


OPERATIONAL_FRONTIER_BODY_SHA256_FIELD: Final = "operational_frontier_body_sha256"


def canonical_host_operational_frontier_v2_body_bytes(
    frontier: HostOperationalFrontierV2,
) -> bytes:
    return _canonical_artifact_body_bytes(frontier, HostOperationalFrontierV2)


def canonical_host_operational_frontier_v2_file_bytes(
    frontier: HostOperationalFrontierV2,
) -> bytes:
    return _canonical_artifact_file_bytes(
        frontier,
        HostOperationalFrontierV2,
        OPERATIONAL_FRONTIER_BODY_SHA256_FIELD,
    )


def parse_host_operational_frontier_v2(
    raw: bytes,
    *,
    expected_file_sha256: str,
) -> HostOperationalFrontierV2:
    body = _parse_artifact_file(
        raw,
        expected_file_sha256=expected_file_sha256,
        body_field=OPERATIONAL_FRONTIER_BODY_SHA256_FIELD,
        body_keys=frozenset(HostOperationalFrontierV2.__dataclass_fields__)
        | {"schema_version", "status", "claims"},
        label="host operational frontier",
    )
    item = dict(body)
    if (
        item.pop("schema_version") != HOST_OPERATIONAL_FRONTIER_SCHEMA_VERSION
        or item.pop("status")
        != "operational_frontier_recorded_nonretryable_non_authorizing"
        or item.pop("claims") != _claims()
    ):
        _fail("host operational frontier envelope differs")
    phases = item.pop("completed_phases")
    if type(phases) is not list:
        _fail("operational completed phases must be one list")
    result = HostOperationalFrontierV2(**item, completed_phases=tuple(phases))
    if raw != canonical_host_operational_frontier_v2_file_bytes(result):
        _fail("host operational frontier canonical replay differs")
    return result


def validate_host_committed_prefix_v2_chain(
    request: HostQualificationCaseRequestV2,
    intent: HostQualificationCaseIntentV2 | None,
    initial_sample: HostInitialCgroupSampleV2 | None,
    ready: HostReadyV2 | None,
    anchor: HostObserverAnchorV2 | None,
    go: HostGoCommitmentV2 | None,
    frontier: HostOperationalFrontierV2,
) -> None:
    """Validate every available operational artifact through the frozen frontier."""

    if type(request) is not HostQualificationCaseRequestV2:
        raise TypeError("request must use the exact request-v2 type")
    if type(frontier) is not HostOperationalFrontierV2:
        raise TypeError("frontier must use the exact operational-frontier-v2 type")
    exact_types: tuple[tuple[object | None, type[Any]], ...] = (
        (intent, HostQualificationCaseIntentV2),
        (initial_sample, HostInitialCgroupSampleV2),
        (ready, HostReadyV2),
        (anchor, HostObserverAnchorV2),
        (go, HostGoCommitmentV2),
    )
    if any(value is not None and type(value) is not expected for value, expected in exact_types):
        raise TypeError("committed-prefix artifacts must use their exact v2 types")
    if frontier.case_spine_sha256 != request.case_spine_sha256:
        _fail("operational frontier crosses its request case spine")
    phase_artifacts = (
        ("intent_committed", intent),
        ("initial_cgroup_sample_committed", initial_sample),
        ("driver_ready", ready),
        ("observer_anchored", anchor),
        ("go_committed", go),
    )
    for phase, artifact in phase_artifacts:
        if (phase in frontier.completed_phases) is not (artifact is not None):
            _fail(f"committed-prefix artifact presence differs at {phase}")
    if intent is None:
        return
    validate_host_request_intent_v2_chain(request, intent)
    if initial_sample is None:
        return
    intent_id = _identity_from_canonical_bytes(
        HOST_CASE_INTENT_SCHEMA_VERSION,
        canonical_host_case_intent_v2_body_bytes(intent),
        canonical_host_case_intent_v2_file_bytes(intent),
    )
    if (
        initial_sample.case_spine_sha256 != intent.case_spine_sha256
        or initial_sample.intent != intent_id
    ):
        _fail("initial sample is cross-wired from the committed intent")
    if ready is None:
        return
    initial_id = _identity_from_canonical_bytes(
        HOST_INITIAL_CGROUP_SAMPLE_SCHEMA_VERSION,
        canonical_host_initial_cgroup_sample_v2_body_bytes(initial_sample),
        canonical_host_initial_cgroup_sample_v2_file_bytes(initial_sample),
    )
    if (
        ready.intent != intent_id
        or ready.initial_cgroup_sample != initial_id
        or ready.case_spine_sha256 != intent.case_spine_sha256
        or ready.container_name != request.container_name
        or ready.algorithmic_measurement_intent != request.algorithmic_measurement_intent
        or ready.storage_boundary_intent != request.storage_boundary_intent
        or any(
            getattr(ready, field) != getattr(request, field)
            for field in (
                "case_ordinal",
                "candidate_id",
                "candidate_family",
                "qualification_case_id",
            )
        )
    ):
        _fail("READY is cross-wired from its request, intent, or initial sample")
    case_identity = initial_sample.cgroup_case_identity
    if (
        ready.cgroup_identity_sha256 != cgroup_case_identity_sha256_v2(case_identity)
        or ready.retained_fd_set_sha256 != initial_sample.facts.retained_fd_set_sha256
        or ready.container_cgroup_path
        != f"{case_identity.case_cgroup_path}/{ready.container_id}"
        or ready.container_cgroup_device != case_identity.case_cgroup_device
        or ready.container_cgroup_inode
        in {
            case_identity.delegate_cgroup_inode,
            case_identity.case_cgroup_inode,
            *(item.endpoint_inode for item in initial_sample.counter_fds),
        }
        or ready.ready_monotonic_ns <= initial_sample.facts.monotonic_ns
    ):
        _fail("READY cgroup identity differs from the retained-FD initial sample")
    if anchor is None:
        return
    ready_id = _identity_from_canonical_bytes(
        HOST_READY_SCHEMA_VERSION,
        canonical_host_ready_v2_body_bytes(ready),
        canonical_host_ready_v2_file_bytes(ready),
    )
    if (
        anchor.case_spine_sha256 != request.case_spine_sha256
        or anchor.ready != ready_id
        or anchor.initial_cgroup_sample != initial_id
        or anchor.retained_fd_set_sha256 != ready.retained_fd_set_sha256
        or anchor.cgroup_identity_sha256 != ready.cgroup_identity_sha256
        or anchor.observer_started_monotonic_ns <= ready.ready_monotonic_ns
    ):
        _fail("observer anchor is cross-wired from READY")
    if go is None:
        return
    validate_host_ready_anchor_go_v2_chain(intent, initial_sample, ready, anchor, go)


@dataclass(frozen=True, slots=True)
class HostPrecleanupCgroupSampleV2:
    case_spine_sha256: str
    operational_frontier: ArtifactIdentityV2
    facts: CgroupSampleFactsV2

    def __post_init__(self) -> None:
        _require_sha256(self.case_spine_sha256, "precleanup case spine")
        _require_artifact_schema(
            self.operational_frontier,
            HOST_OPERATIONAL_FRONTIER_SCHEMA_VERSION,
            "precleanup operational frontier",
        )
        if type(self.facts) is not CgroupSampleFactsV2:
            _fail("precleanup sample facts type differs")

    def to_body_dict(self) -> dict[str, Any]:
        return {
            "schema_version": HOST_PRECLEANUP_CGROUP_SAMPLE_SCHEMA_VERSION,
            "case_spine_sha256": self.case_spine_sha256,
            "operational_frontier": self.operational_frontier.to_dict(),
            "facts": self.facts.to_dict(),
        }


PRECLEANUP_SAMPLE_BODY_SHA256_FIELD: Final = "precleanup_sample_body_sha256"


def canonical_host_precleanup_cgroup_sample_v2_body_bytes(
    sample: HostPrecleanupCgroupSampleV2,
) -> bytes:
    return _canonical_artifact_body_bytes(sample, HostPrecleanupCgroupSampleV2)


def canonical_host_precleanup_cgroup_sample_v2_file_bytes(
    sample: HostPrecleanupCgroupSampleV2,
) -> bytes:
    return _canonical_artifact_file_bytes(
        sample,
        HostPrecleanupCgroupSampleV2,
        PRECLEANUP_SAMPLE_BODY_SHA256_FIELD,
    )


def parse_host_precleanup_cgroup_sample_v2(
    raw: bytes,
    *,
    expected_file_sha256: str,
) -> HostPrecleanupCgroupSampleV2:
    body = _parse_artifact_file(
        raw,
        expected_file_sha256=expected_file_sha256,
        body_field=PRECLEANUP_SAMPLE_BODY_SHA256_FIELD,
        body_keys=frozenset(
            {"schema_version", *HostPrecleanupCgroupSampleV2.__dataclass_fields__}
        ),
        label="host precleanup cgroup sample",
    )
    item = dict(body)
    if item.pop("schema_version") != HOST_PRECLEANUP_CGROUP_SAMPLE_SCHEMA_VERSION:
        _fail("precleanup cgroup sample schema differs")
    item["operational_frontier"] = _artifact_identity(
        item["operational_frontier"],
        "precleanup operational frontier",
    )
    item["facts"] = _sample_facts_from_dict(item["facts"])
    result = HostPrecleanupCgroupSampleV2(**item)
    if raw != canonical_host_precleanup_cgroup_sample_v2_file_bytes(result):
        _fail("precleanup cgroup sample canonical replay differs")
    return result


@dataclass(frozen=True, slots=True)
class HostCgroupKillReceiptV2:
    case_spine_sha256: str
    precleanup_sample: ArtifactIdentityV2
    retained_fd_set_sha256: str
    cgroup_identity_sha256: str
    kill_monotonic_ns: int
    cgroup_kill_value: int
    entire_subtree_targeted: bool

    def __post_init__(self) -> None:
        _require_sha256(self.case_spine_sha256, "cgroup.kill case spine")
        _require_artifact_schema(
            self.precleanup_sample,
            HOST_PRECLEANUP_CGROUP_SAMPLE_SCHEMA_VERSION,
            "cgroup.kill precleanup sample",
        )
        _require_sha256(self.retained_fd_set_sha256, "cgroup.kill retained-FD set")
        _require_sha256(self.cgroup_identity_sha256, "cgroup.kill cgroup identity")
        _require_int(self.kill_monotonic_ns, "cgroup.kill monotonic time")
        if _require_int(self.cgroup_kill_value, "cgroup.kill write value", maximum=1) != 1:
            _fail("cgroup.kill receipt must record exact write value one")
        if _require_bool(self.entire_subtree_targeted, "cgroup.kill subtree") is not True:
            _fail("cgroup.kill must target the entire fresh subtree")

    def to_body_dict(self) -> dict[str, Any]:
        return {
            "schema_version": HOST_CGROUP_KILL_RECEIPT_SCHEMA_VERSION,
            "case_spine_sha256": self.case_spine_sha256,
            "precleanup_sample": self.precleanup_sample.to_dict(),
            "retained_fd_set_sha256": self.retained_fd_set_sha256,
            "cgroup_identity_sha256": self.cgroup_identity_sha256,
            "kill_monotonic_ns": self.kill_monotonic_ns,
            "cgroup_kill_value": self.cgroup_kill_value,
            "entire_subtree_targeted": self.entire_subtree_targeted,
        }


CGROUP_KILL_BODY_SHA256_FIELD: Final = "cgroup_kill_body_sha256"


def canonical_host_cgroup_kill_receipt_v2_body_bytes(
    receipt: HostCgroupKillReceiptV2,
) -> bytes:
    return _canonical_artifact_body_bytes(receipt, HostCgroupKillReceiptV2)


def canonical_host_cgroup_kill_receipt_v2_file_bytes(
    receipt: HostCgroupKillReceiptV2,
) -> bytes:
    return _canonical_artifact_file_bytes(
        receipt,
        HostCgroupKillReceiptV2,
        CGROUP_KILL_BODY_SHA256_FIELD,
    )


def parse_host_cgroup_kill_receipt_v2(
    raw: bytes,
    *,
    expected_file_sha256: str,
) -> HostCgroupKillReceiptV2:
    body = _parse_artifact_file(
        raw,
        expected_file_sha256=expected_file_sha256,
        body_field=CGROUP_KILL_BODY_SHA256_FIELD,
        body_keys=frozenset({"schema_version", *HostCgroupKillReceiptV2.__dataclass_fields__}),
        label="host cgroup.kill receipt",
    )
    item = dict(body)
    if item.pop("schema_version") != HOST_CGROUP_KILL_RECEIPT_SCHEMA_VERSION:
        _fail("host cgroup.kill receipt schema differs")
    item["precleanup_sample"] = _artifact_identity(
        item["precleanup_sample"],
        "cgroup.kill precleanup sample",
    )
    result = HostCgroupKillReceiptV2(**item)
    if raw != canonical_host_cgroup_kill_receipt_v2_file_bytes(result):
        _fail("host cgroup.kill receipt canonical replay differs")
    return result


@dataclass(frozen=True, slots=True)
class HostCgroupEmptyObservationV2:
    case_spine_sha256: str
    precleanup_sample: ArtifactIdentityV2
    cgroup_kill_receipt: ArtifactIdentityV2 | None
    retained_fd_set_sha256: str
    cgroup_identity_sha256: str
    observed_monotonic_ns: int
    populated: bool
    pids_current: int
    recursive_process_count: int

    def __post_init__(self) -> None:
        _require_sha256(self.case_spine_sha256, "empty observation case spine")
        _require_artifact_schema(
            self.precleanup_sample,
            HOST_PRECLEANUP_CGROUP_SAMPLE_SCHEMA_VERSION,
            "empty observation precleanup sample",
        )
        if self.cgroup_kill_receipt is not None:
            _require_artifact_schema(
                self.cgroup_kill_receipt,
                HOST_CGROUP_KILL_RECEIPT_SCHEMA_VERSION,
                "empty observation optional kill receipt",
            )
        _require_sha256(self.retained_fd_set_sha256, "empty observation retained-FD set")
        _require_sha256(self.cgroup_identity_sha256, "empty observation cgroup identity")
        _require_int(self.observed_monotonic_ns, "empty observation time")
        if _require_bool(self.populated, "empty observation populated") is not False:
            _fail("empty cgroup observation must be unpopulated")
        if (
            _require_int(self.pids_current, "empty observation pids current") != 0
            or _require_int(
                self.recursive_process_count,
                "empty observation recursive process count",
            )
            != 0
        ):
            _fail("empty cgroup observation must contain zero live tasks")

    def to_body_dict(self) -> dict[str, Any]:
        return {
            "schema_version": HOST_CGROUP_EMPTY_OBSERVATION_SCHEMA_VERSION,
            "case_spine_sha256": self.case_spine_sha256,
            "precleanup_sample": self.precleanup_sample.to_dict(),
            "cgroup_kill_receipt": (
                None if self.cgroup_kill_receipt is None else self.cgroup_kill_receipt.to_dict()
            ),
            "retained_fd_set_sha256": self.retained_fd_set_sha256,
            "cgroup_identity_sha256": self.cgroup_identity_sha256,
            "observed_monotonic_ns": self.observed_monotonic_ns,
            "populated": self.populated,
            "pids_current": self.pids_current,
            "recursive_process_count": self.recursive_process_count,
        }


CGROUP_EMPTY_BODY_SHA256_FIELD: Final = "cgroup_empty_body_sha256"


def canonical_host_cgroup_empty_observation_v2_body_bytes(
    observation: HostCgroupEmptyObservationV2,
) -> bytes:
    return _canonical_artifact_body_bytes(observation, HostCgroupEmptyObservationV2)


def canonical_host_cgroup_empty_observation_v2_file_bytes(
    observation: HostCgroupEmptyObservationV2,
) -> bytes:
    return _canonical_artifact_file_bytes(
        observation,
        HostCgroupEmptyObservationV2,
        CGROUP_EMPTY_BODY_SHA256_FIELD,
    )


def parse_host_cgroup_empty_observation_v2(
    raw: bytes,
    *,
    expected_file_sha256: str,
) -> HostCgroupEmptyObservationV2:
    body = _parse_artifact_file(
        raw,
        expected_file_sha256=expected_file_sha256,
        body_field=CGROUP_EMPTY_BODY_SHA256_FIELD,
        body_keys=frozenset(
            {"schema_version", *HostCgroupEmptyObservationV2.__dataclass_fields__}
        ),
        label="host cgroup-empty observation",
    )
    item = dict(body)
    if item.pop("schema_version") != HOST_CGROUP_EMPTY_OBSERVATION_SCHEMA_VERSION:
        _fail("host cgroup-empty observation schema differs")
    item["precleanup_sample"] = _artifact_identity(
        item["precleanup_sample"],
        "empty observation precleanup sample",
    )
    item["cgroup_kill_receipt"] = _optional_artifact_identity(
        item["cgroup_kill_receipt"],
        "empty observation optional kill receipt",
    )
    result = HostCgroupEmptyObservationV2(**item)
    if raw != canonical_host_cgroup_empty_observation_v2_file_bytes(result):
        _fail("host cgroup-empty observation canonical replay differs")
    return result


type ContainerResolutionStateV2 = Literal[
    "never_created",
    "create_uncertain_resolved_absent",
    "create_uncertain_found_removed",
    "created_removed",
]


@dataclass(frozen=True, slots=True)
class HostContainerAbsenceObservationV2:
    case_spine_sha256: str
    operational_frontier: ArtifactIdentityV2
    cgroup_empty_observation: ArtifactIdentityV2 | None
    container_name: str
    container_lookup_identity_sha256: str
    resolution_state: ContainerResolutionStateV2
    actual_runtime_container_identity_sha256: str | None
    actual_container_id: str | None
    removal_monotonic_ns: int
    container_remove_count: int
    container_absent: bool

    def __post_init__(self) -> None:
        _require_sha256(self.case_spine_sha256, "container absence case spine")
        _require_artifact_schema(
            self.operational_frontier,
            HOST_OPERATIONAL_FRONTIER_SCHEMA_VERSION,
            "container absence operational frontier",
        )
        if self.cgroup_empty_observation is not None:
            _require_artifact_schema(
                self.cgroup_empty_observation,
                HOST_CGROUP_EMPTY_OBSERVATION_SCHEMA_VERSION,
                "container absence optional empty observation",
            )
        name = _require_text(self.container_name, "container absence lookup name")
        if (
            _CONTAINER_NAME_RE.fullmatch(name) is None
            or name != expected_container_name_v2(self.case_spine_sha256)
            or self.container_lookup_identity_sha256
            != container_lookup_identity_sha256_v2(self.case_spine_sha256, name)
        ):
            _fail("container absence lookup identity differs from its complete case spine")
        state = _require_text(self.resolution_state, "container absence resolution state")
        if state not in {
            "never_created",
            "create_uncertain_resolved_absent",
            "create_uncertain_found_removed",
            "created_removed",
        }:
            _fail("container absence resolution state differs")
        actual_identity = _require_optional_sha256(
            self.actual_runtime_container_identity_sha256,
            "container absence optional runtime identity",
        )
        actual_id = self.actual_container_id
        if actual_id is not None:
            container_runtime_identity_sha256_v2(
                self.case_spine_sha256,
                name,
                actual_id,
            )
        if (actual_identity is None) is not (actual_id is None):
            _fail("container absence runtime identity and ID presence differ")
        if actual_id is not None and actual_identity != container_runtime_identity_sha256_v2(
            self.case_spine_sha256,
            name,
            actual_id,
        ):
            _fail("container absence runtime identity is not derived from its actual ID")
        _require_int(self.removal_monotonic_ns, "container removal time")
        count = _require_int(self.container_remove_count, "container removal count", maximum=1)
        expected_actual = state in {"create_uncertain_found_removed", "created_removed"}
        expected_count = 1 if expected_actual else 0
        if (actual_id is not None) is not expected_actual or count != expected_count:
            _fail("container absence runtime identity/count differs from resolution state")
        if _require_bool(self.container_absent, "container absence") is not True:
            _fail("container absence observation must prove absence")

    def to_body_dict(self) -> dict[str, Any]:
        return {
            "schema_version": HOST_CONTAINER_ABSENCE_OBSERVATION_SCHEMA_VERSION,
            "case_spine_sha256": self.case_spine_sha256,
            "operational_frontier": self.operational_frontier.to_dict(),
            "cgroup_empty_observation": (
                None
                if self.cgroup_empty_observation is None
                else self.cgroup_empty_observation.to_dict()
            ),
            "container_name": self.container_name,
            "container_lookup_identity_sha256": self.container_lookup_identity_sha256,
            "resolution_state": self.resolution_state,
            "actual_runtime_container_identity_sha256": (
                self.actual_runtime_container_identity_sha256
            ),
            "actual_container_id": self.actual_container_id,
            "removal_monotonic_ns": self.removal_monotonic_ns,
            "container_remove_count": self.container_remove_count,
            "container_absent": self.container_absent,
        }


CONTAINER_ABSENCE_BODY_SHA256_FIELD: Final = "container_absence_body_sha256"


def canonical_host_container_absence_observation_v2_body_bytes(
    observation: HostContainerAbsenceObservationV2,
) -> bytes:
    return _canonical_artifact_body_bytes(observation, HostContainerAbsenceObservationV2)


def canonical_host_container_absence_observation_v2_file_bytes(
    observation: HostContainerAbsenceObservationV2,
) -> bytes:
    return _canonical_artifact_file_bytes(
        observation,
        HostContainerAbsenceObservationV2,
        CONTAINER_ABSENCE_BODY_SHA256_FIELD,
    )


def parse_host_container_absence_observation_v2(
    raw: bytes,
    *,
    expected_file_sha256: str,
) -> HostContainerAbsenceObservationV2:
    body = _parse_artifact_file(
        raw,
        expected_file_sha256=expected_file_sha256,
        body_field=CONTAINER_ABSENCE_BODY_SHA256_FIELD,
        body_keys=frozenset(
            {"schema_version", *HostContainerAbsenceObservationV2.__dataclass_fields__}
        ),
        label="host container-absence observation",
    )
    item = dict(body)
    if item.pop("schema_version") != HOST_CONTAINER_ABSENCE_OBSERVATION_SCHEMA_VERSION:
        _fail("host container-absence schema differs")
    item["operational_frontier"] = _artifact_identity(
        item["operational_frontier"],
        "container absence operational frontier",
    )
    item["cgroup_empty_observation"] = _optional_artifact_identity(
        item["cgroup_empty_observation"],
        "container absence optional empty observation",
    )
    result = HostContainerAbsenceObservationV2(**item)
    if raw != canonical_host_container_absence_observation_v2_file_bytes(result):
        _fail("host container-absence canonical replay differs")
    return result


def _validate_container_absence_against_frontier(
    frontier: HostOperationalFrontierV2,
    observation: HostContainerAbsenceObservationV2,
) -> None:
    if observation.operational_frontier != _frontier_identity(frontier):
        _fail("container absence crosses its operational frontier")
    allowed: Mapping[BoundaryStateV2, frozenset[str]] = {
        "not_started": frozenset({"never_created"}),
        "commit_uncertain": frozenset(
            {
                "create_uncertain_resolved_absent",
                "create_uncertain_found_removed",
            }
        ),
        "committed": frozenset({"created_removed"}),
    }
    if observation.resolution_state not in allowed[frontier.container_create_state]:
        _fail("container absence resolution differs from the container-create frontier")


@dataclass(frozen=True, slots=True)
class HostPostContainerRemoveCgroupSampleV2:
    case_spine_sha256: str
    container_absence_observation: ArtifactIdentityV2
    container_name: str
    container_lookup_identity_sha256: str
    actual_runtime_container_identity_sha256: str | None
    actual_container_id: str | None
    facts: CgroupSampleFactsV2
    retained_fds_still_open: bool

    def __post_init__(self) -> None:
        _require_sha256(self.case_spine_sha256, "post-remove sample case spine")
        _require_artifact_schema(
            self.container_absence_observation,
            HOST_CONTAINER_ABSENCE_OBSERVATION_SCHEMA_VERSION,
            "post-remove container absence",
        )
        name = _require_text(self.container_name, "post-remove container name")
        if (
            name != expected_container_name_v2(self.case_spine_sha256)
            or self.container_lookup_identity_sha256
            != container_lookup_identity_sha256_v2(self.case_spine_sha256, name)
        ):
            _fail("post-remove container lookup identity differs")
        if (self.actual_runtime_container_identity_sha256 is None) is not (
            self.actual_container_id is None
        ):
            _fail("post-remove actual runtime identity and ID presence differ")
        if self.actual_container_id is not None:
            expected_actual = container_runtime_identity_sha256_v2(
                self.case_spine_sha256,
                name,
                self.actual_container_id,
            )
            if self.actual_runtime_container_identity_sha256 != expected_actual:
                _fail("post-remove actual runtime identity differs")
        if type(self.facts) is not CgroupSampleFactsV2:
            _fail("post-container-remove sample facts type differs")
        if (
            self.facts.populated
            or self.facts.memory_current_bytes
            or self.facts.pids_current
            or self.facts.nr_descendants
            or self.facts.nr_dying_descendants
        ):
            _fail("post-container-remove cgroup sample is not empty without descendants")
        if _require_bool(self.retained_fds_still_open, "post-remove retained FDs") is not True:
            _fail("post-container-remove sample must precede retained-FD closure")

    def to_body_dict(self) -> dict[str, Any]:
        return {
            "schema_version": HOST_POST_CONTAINER_REMOVE_CGROUP_SAMPLE_SCHEMA_VERSION,
            "case_spine_sha256": self.case_spine_sha256,
            "container_absence_observation": self.container_absence_observation.to_dict(),
            "container_name": self.container_name,
            "container_lookup_identity_sha256": self.container_lookup_identity_sha256,
            "actual_runtime_container_identity_sha256": (
                self.actual_runtime_container_identity_sha256
            ),
            "actual_container_id": self.actual_container_id,
            "facts": self.facts.to_dict(),
            "retained_fds_still_open": self.retained_fds_still_open,
        }


POST_REMOVE_SAMPLE_BODY_SHA256_FIELD: Final = "post_remove_sample_body_sha256"


def canonical_host_post_container_remove_cgroup_sample_v2_body_bytes(
    sample: HostPostContainerRemoveCgroupSampleV2,
) -> bytes:
    return _canonical_artifact_body_bytes(sample, HostPostContainerRemoveCgroupSampleV2)


def canonical_host_post_container_remove_cgroup_sample_v2_file_bytes(
    sample: HostPostContainerRemoveCgroupSampleV2,
) -> bytes:
    return _canonical_artifact_file_bytes(
        sample,
        HostPostContainerRemoveCgroupSampleV2,
        POST_REMOVE_SAMPLE_BODY_SHA256_FIELD,
    )


def parse_host_post_container_remove_cgroup_sample_v2(
    raw: bytes,
    *,
    expected_file_sha256: str,
) -> HostPostContainerRemoveCgroupSampleV2:
    body = _parse_artifact_file(
        raw,
        expected_file_sha256=expected_file_sha256,
        body_field=POST_REMOVE_SAMPLE_BODY_SHA256_FIELD,
        body_keys=frozenset(
            {"schema_version", *HostPostContainerRemoveCgroupSampleV2.__dataclass_fields__}
        ),
        label="host post-container-remove cgroup sample",
    )
    item = dict(body)
    if item.pop("schema_version") != HOST_POST_CONTAINER_REMOVE_CGROUP_SAMPLE_SCHEMA_VERSION:
        _fail("host post-container-remove sample schema differs")
    item["container_absence_observation"] = _artifact_identity(
        item["container_absence_observation"],
        "post-remove container absence",
    )
    item["facts"] = _sample_facts_from_dict(item["facts"])
    result = HostPostContainerRemoveCgroupSampleV2(**item)
    if raw != canonical_host_post_container_remove_cgroup_sample_v2_file_bytes(result):
        _fail("host post-container-remove sample canonical replay differs")
    return result


@dataclass(frozen=True, slots=True)
class HostCgroupCounterFdsClosedReceiptV2:
    case_spine_sha256: str
    post_container_remove_sample: ArtifactIdentityV2
    retained_fd_set_sha256: str
    cgroup_identity_sha256: str
    container_name: str
    container_lookup_identity_sha256: str
    actual_runtime_container_identity_sha256: str | None
    actual_container_id: str | None
    closed_endpoint_names: tuple[str, ...]
    close_monotonic_ns: int
    all_fds_closed: bool
    reopen_permitted: bool

    def __post_init__(self) -> None:
        _require_sha256(self.case_spine_sha256, "FD-close case spine")
        _require_artifact_schema(
            self.post_container_remove_sample,
            HOST_POST_CONTAINER_REMOVE_CGROUP_SAMPLE_SCHEMA_VERSION,
            "FD-close post-remove sample",
        )
        for value, label in (
            (self.retained_fd_set_sha256, "FD-close retained-FD set"),
            (self.cgroup_identity_sha256, "FD-close cgroup identity"),
            (self.container_lookup_identity_sha256, "FD-close container lookup identity"),
        ):
            _require_sha256(value, label)
        name = _require_text(self.container_name, "FD-close container name")
        if (
            name != expected_container_name_v2(self.case_spine_sha256)
            or self.container_lookup_identity_sha256
            != container_lookup_identity_sha256_v2(self.case_spine_sha256, name)
        ):
            _fail("FD-close container lookup identity differs")
        if (self.actual_runtime_container_identity_sha256 is None) is not (
            self.actual_container_id is None
        ):
            _fail("FD-close actual runtime identity and ID presence differ")
        if self.actual_container_id is not None:
            expected_actual = container_runtime_identity_sha256_v2(
                self.case_spine_sha256,
                name,
                self.actual_container_id,
            )
            if self.actual_runtime_container_identity_sha256 != expected_actual:
                _fail("FD-close actual runtime identity differs")
        if (
            type(self.closed_endpoint_names) is not tuple
            or any(type(item) is not str for item in self.closed_endpoint_names)
            or self.closed_endpoint_names != CGROUP_COUNTER_ENDPOINTS
        ):
            _fail("FD-close endpoint coverage or order differs")
        _require_int(self.close_monotonic_ns, "FD-close monotonic time")
        if _require_bool(self.all_fds_closed, "all retained FDs closed") is not True:
            _fail("FD-close receipt must account for every retained FD")
        if _require_bool(self.reopen_permitted, "FD reopen permitted") is not False:
            _fail("counter FD reopen cannot be permitted")

    def to_body_dict(self) -> dict[str, Any]:
        return {
            "schema_version": HOST_CGROUP_COUNTER_FDS_CLOSED_RECEIPT_SCHEMA_VERSION,
            "case_spine_sha256": self.case_spine_sha256,
            "post_container_remove_sample": self.post_container_remove_sample.to_dict(),
            "retained_fd_set_sha256": self.retained_fd_set_sha256,
            "cgroup_identity_sha256": self.cgroup_identity_sha256,
            "container_name": self.container_name,
            "container_lookup_identity_sha256": self.container_lookup_identity_sha256,
            "actual_runtime_container_identity_sha256": (
                self.actual_runtime_container_identity_sha256
            ),
            "actual_container_id": self.actual_container_id,
            "closed_endpoint_names": list(self.closed_endpoint_names),
            "close_monotonic_ns": self.close_monotonic_ns,
            "all_fds_closed": self.all_fds_closed,
            "reopen_permitted": self.reopen_permitted,
        }


FD_CLOSE_BODY_SHA256_FIELD: Final = "fd_close_body_sha256"


def canonical_host_cgroup_counter_fds_closed_receipt_v2_body_bytes(
    receipt: HostCgroupCounterFdsClosedReceiptV2,
) -> bytes:
    return _canonical_artifact_body_bytes(receipt, HostCgroupCounterFdsClosedReceiptV2)


def canonical_host_cgroup_counter_fds_closed_receipt_v2_file_bytes(
    receipt: HostCgroupCounterFdsClosedReceiptV2,
) -> bytes:
    return _canonical_artifact_file_bytes(
        receipt,
        HostCgroupCounterFdsClosedReceiptV2,
        FD_CLOSE_BODY_SHA256_FIELD,
    )


def parse_host_cgroup_counter_fds_closed_receipt_v2(
    raw: bytes,
    *,
    expected_file_sha256: str,
) -> HostCgroupCounterFdsClosedReceiptV2:
    body = _parse_artifact_file(
        raw,
        expected_file_sha256=expected_file_sha256,
        body_field=FD_CLOSE_BODY_SHA256_FIELD,
        body_keys=frozenset(
            {"schema_version", *HostCgroupCounterFdsClosedReceiptV2.__dataclass_fields__}
        ),
        label="host retained-FD close receipt",
    )
    item = dict(body)
    if item.pop("schema_version") != HOST_CGROUP_COUNTER_FDS_CLOSED_RECEIPT_SCHEMA_VERSION:
        _fail("host retained-FD close receipt schema differs")
    item["post_container_remove_sample"] = _artifact_identity(
        item["post_container_remove_sample"],
        "FD-close post-remove sample",
    )
    names = item.pop("closed_endpoint_names")
    if type(names) is not list:
        _fail("closed endpoint names must be one list")
    result = HostCgroupCounterFdsClosedReceiptV2(
        **item,
        closed_endpoint_names=tuple(names),
    )
    if raw != canonical_host_cgroup_counter_fds_closed_receipt_v2_file_bytes(result):
        _fail("host retained-FD close receipt canonical replay differs")
    return result


@dataclass(frozen=True, slots=True)
class HostOuterCgroupAbsenceObservationV2:
    case_spine_sha256: str
    cgroup_counter_fds_closed_receipt: ArtifactIdentityV2
    cgroup_case_identity: HostCgroupCaseIdentityV2
    cgroup_identity_sha256: str
    case_cgroup_path: str
    removal_monotonic_ns: int
    outer_cgroup_remove_count: int
    outer_cgroup_absent: bool

    def __post_init__(self) -> None:
        _require_sha256(self.case_spine_sha256, "outer absence case spine")
        _require_artifact_schema(
            self.cgroup_counter_fds_closed_receipt,
            HOST_CGROUP_COUNTER_FDS_CLOSED_RECEIPT_SCHEMA_VERSION,
            "outer absence FD-close receipt",
        )
        if type(self.cgroup_case_identity) is not HostCgroupCaseIdentityV2:
            _fail("outer absence cgroup case identity type differs")
        expected_identity = cgroup_case_identity_sha256_v2(self.cgroup_case_identity)
        if (
            self.cgroup_case_identity.case_spine_sha256 != self.case_spine_sha256
            or _require_sha256(
                self.cgroup_identity_sha256,
                "outer absence cgroup identity",
            )
            != expected_identity
            or self.case_cgroup_path != self.cgroup_case_identity.case_cgroup_path
        ):
            _fail("outer absence does not bind the exact case cgroup directory evidence")
        _require_int(self.removal_monotonic_ns, "outer absence time")
        if (
            _require_int(
                self.outer_cgroup_remove_count,
                "outer cgroup removal count",
                maximum=1,
            )
            != 1
        ):
            _fail("outer cgroup removal count must be exact one")
        if _require_bool(self.outer_cgroup_absent, "outer cgroup absent") is not True:
            _fail("outer-cgroup absence observation must prove absence")

    def to_body_dict(self) -> dict[str, Any]:
        return {
            "schema_version": HOST_OUTER_CGROUP_ABSENCE_OBSERVATION_SCHEMA_VERSION,
            "case_spine_sha256": self.case_spine_sha256,
            "cgroup_counter_fds_closed_receipt": (
                self.cgroup_counter_fds_closed_receipt.to_dict()
            ),
            "cgroup_case_identity": self.cgroup_case_identity.to_dict(),
            "cgroup_identity_sha256": self.cgroup_identity_sha256,
            "case_cgroup_path": self.case_cgroup_path,
            "removal_monotonic_ns": self.removal_monotonic_ns,
            "outer_cgroup_remove_count": self.outer_cgroup_remove_count,
            "outer_cgroup_absent": self.outer_cgroup_absent,
        }


OUTER_ABSENCE_BODY_SHA256_FIELD: Final = "outer_absence_body_sha256"


def canonical_host_outer_cgroup_absence_observation_v2_body_bytes(
    observation: HostOuterCgroupAbsenceObservationV2,
) -> bytes:
    return _canonical_artifact_body_bytes(observation, HostOuterCgroupAbsenceObservationV2)


def canonical_host_outer_cgroup_absence_observation_v2_file_bytes(
    observation: HostOuterCgroupAbsenceObservationV2,
) -> bytes:
    return _canonical_artifact_file_bytes(
        observation,
        HostOuterCgroupAbsenceObservationV2,
        OUTER_ABSENCE_BODY_SHA256_FIELD,
    )


def parse_host_outer_cgroup_absence_observation_v2(
    raw: bytes,
    *,
    expected_file_sha256: str,
) -> HostOuterCgroupAbsenceObservationV2:
    body = _parse_artifact_file(
        raw,
        expected_file_sha256=expected_file_sha256,
        body_field=OUTER_ABSENCE_BODY_SHA256_FIELD,
        body_keys=frozenset(
            {"schema_version", *HostOuterCgroupAbsenceObservationV2.__dataclass_fields__}
        ),
        label="host outer-cgroup absence observation",
    )
    item = dict(body)
    if item.pop("schema_version") != HOST_OUTER_CGROUP_ABSENCE_OBSERVATION_SCHEMA_VERSION:
        _fail("host outer-cgroup absence schema differs")
    item["cgroup_counter_fds_closed_receipt"] = _artifact_identity(
        item["cgroup_counter_fds_closed_receipt"],
        "outer absence FD-close receipt",
    )
    item["cgroup_case_identity"] = _cgroup_case_identity_from_dict(
        item["cgroup_case_identity"]
    )
    result = HostOuterCgroupAbsenceObservationV2(**item)
    if raw != canonical_host_outer_cgroup_absence_observation_v2_file_bytes(result):
        _fail("host outer-cgroup absence canonical replay differs")
    return result


@dataclass(frozen=True, slots=True)
class HostRawResourceMeasurementsV2:
    memory_peak_bytes: int
    memory_peak_semantics: Literal["conservative_observed_upper_bound"]
    memory_oom_kill_count: int
    memory_oom_kill_count_semantics: Literal["exact_observation"]
    initial_cpu_usage_usec: int
    post_remove_cpu_usage_usec: int
    cpu_delta_usec: int
    cpu_time_ns: int
    cpu_time_semantics: Literal["exact_observation"]
    initial_monotonic_ns: int
    post_remove_monotonic_ns: int
    wall_time_ns: int
    wall_time_semantics: Literal["exact_observation"]
    pids_peak: int
    pids_peak_semantics: Literal["conservative_observed_upper_bound"]
    pids_max_event_count: int
    pids_max_event_count_semantics: Literal["exact_observation"]
    attempt_count: int
    attempt_count_semantics: Literal["exact_observation"]
    failure_count: int
    failure_count_semantics: Literal["exact_observation"]
    structural_measurements_only: bool
    production_qualified: bool

    def __post_init__(self) -> None:
        for field in (
            "memory_peak_bytes",
            "memory_oom_kill_count",
            "initial_cpu_usage_usec",
            "post_remove_cpu_usage_usec",
            "cpu_delta_usec",
            "cpu_time_ns",
            "initial_monotonic_ns",
            "post_remove_monotonic_ns",
            "wall_time_ns",
            "pids_peak",
            "pids_max_event_count",
            "attempt_count",
            "failure_count",
        ):
            _require_int(getattr(self, field), f"host resource {field}")
        if (
            _require_text(self.memory_peak_semantics, "memory.peak semantics")
            != "conservative_observed_upper_bound"
        ):
            _fail("memory.peak value semantics differ")
        if (
            _require_text(self.pids_peak_semantics, "pids.peak semantics")
            != "conservative_observed_upper_bound"
        ):
            _fail("pids.peak value semantics differ")
        if any(
            _require_text(semantics, label) != "exact_observation"
            for semantics, label in (
                (self.cpu_time_semantics, "CPU-time semantics"),
                (self.wall_time_semantics, "wall-time semantics"),
                (
                    self.memory_oom_kill_count_semantics,
                    "memory OOM-kill-count semantics",
                ),
                (
                    self.pids_max_event_count_semantics,
                    "pids max-event-count semantics",
                ),
                (self.attempt_count_semantics, "attempt-count semantics"),
                (self.failure_count_semantics, "failure-count semantics"),
            )
        ):
            _fail("exact counter or time semantics differ")
        if self.post_remove_cpu_usage_usec < self.initial_cpu_usage_usec:
            _fail("fresh-cgroup CPU counter rolled back")
        expected_cpu_delta = self.post_remove_cpu_usage_usec - self.initial_cpu_usage_usec
        if self.cpu_delta_usec != expected_cpu_delta:
            _fail("CPU usec delta differs")
        if (
            self.cpu_delta_usec > _MAX_INTEGER // 1000
            or self.cpu_time_ns != self.cpu_delta_usec * 1000
        ):
            _fail("CPU usec-to-ns conversion differs or overflows")
        if self.post_remove_monotonic_ns <= self.initial_monotonic_ns:
            _fail("wall-clock sample chronology differs")
        if self.wall_time_ns != self.post_remove_monotonic_ns - self.initial_monotonic_ns:
            _fail("wall-time monotonic delta differs")
        if self.attempt_count > 1 or self.failure_count > 1:
            _fail("host resource attempt or failure count exceeds one")
        if (
            _require_bool(
                self.structural_measurements_only,
                "host resource structural-only qualification",
            )
            is not True
            or _require_bool(
                self.production_qualified,
                "host resource production qualification",
            )
            is not False
        ):
            _fail("host resource values must remain structural and production-unqualified")

    def to_dict(self) -> dict[str, Any]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}


def _raw_resources_from_dict(value: object) -> HostRawResourceMeasurementsV2:
    item = _require_exact_keys(
        value,
        frozenset(HostRawResourceMeasurementsV2.__dataclass_fields__),
        "host raw resource measurements",
    )
    return HostRawResourceMeasurementsV2(**item)


@dataclass(frozen=True, slots=True)
class HostCgroupMembershipEventV2:
    sequence_ordinal: int
    event_kind: Literal[
        "initial_empty_boundary",
        "anchored_container_membership",
        "post_remove_empty_boundary",
    ]
    monotonic_ns: int
    live_member_count: int
    migration_event_count: int
    observation_gap_count: int

    def __post_init__(self) -> None:
        ordinal = _require_int(self.sequence_ordinal, "membership event ordinal", maximum=2)
        kinds = (
            "initial_empty_boundary",
            "anchored_container_membership",
            "post_remove_empty_boundary",
        )
        if _require_text(self.event_kind, "membership event kind") != kinds[ordinal]:
            _fail("membership event kind differs from its exact sequence position")
        _require_int(self.monotonic_ns, "membership event time")
        expected_members = (0, 1, 0)[ordinal]
        if (
            _require_int(self.live_member_count, "membership event live-member count")
            != expected_members
            or _require_int(self.migration_event_count, "membership event migration count") != 0
            or _require_int(self.observation_gap_count, "membership event gap count") != 0
        ):
            _fail("membership event contains an unexpected member, migration, or gap")

    def to_dict(self) -> dict[str, Any]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}


def _membership_event_from_dict(value: object) -> HostCgroupMembershipEventV2:
    item = _require_exact_keys(
        value,
        frozenset(HostCgroupMembershipEventV2.__dataclass_fields__),
        "cgroup membership event",
    )
    return HostCgroupMembershipEventV2(**item)


def _membership_event_inventory_sha256_v2(
    events: tuple[HostCgroupMembershipEventV2, ...],
) -> str:
    if (
        type(events) is not tuple
        or any(type(event) is not HostCgroupMembershipEventV2 for event in events)
        or tuple(event.sequence_ordinal for event in events) != (0, 1, 2)
    ):
        _fail("membership event inventory differs from the exact three-event sequence")
    return _body_sha256({"events": [event.to_dict() for event in events]})


@dataclass(frozen=True, slots=True)
class HostCgroupMembershipEventLogV2:
    case_spine_sha256: str
    cgroup_case_identity_sha256: str
    container_name: str
    container_lookup_identity_sha256: str
    actual_runtime_container_identity_sha256: str
    actual_container_id: str
    container_cgroup_path: str
    container_cgroup_device: int
    container_cgroup_inode: int
    host_pid: int
    host_process_start_time_ticks: int
    observer_descriptor_sha256: str
    observer_source_sha256: str
    host_provisioning_receipt: ArtifactIdentityV2
    observer_anchor: ArtifactIdentityV2
    initial_sample: ArtifactIdentityV2
    post_container_remove_sample: ArtifactIdentityV2
    events: tuple[HostCgroupMembershipEventV2, ...]
    event_inventory_sha256: str
    continuous_all_descendant_membership_proven: bool
    provisioning_receipt_semantics_validated: bool
    provisioning_receipt_producer_authenticated: bool
    production_containment_eligible: bool

    def __post_init__(self) -> None:
        spine = _require_sha256(self.case_spine_sha256, "membership log case spine")
        _require_sha256(self.cgroup_case_identity_sha256, "membership log cgroup identity")
        name = _require_text(self.container_name, "membership log container name")
        if (
            name != expected_container_name_v2(spine)
            or self.container_lookup_identity_sha256
            != container_lookup_identity_sha256_v2(spine, name)
            or self.actual_runtime_container_identity_sha256
            != container_runtime_identity_sha256_v2(spine, name, self.actual_container_id)
        ):
            _fail("membership log container identity differs from the complete case spine")
        if (
            type(self.container_cgroup_path) is not str
            or _CGROUP_PATH_RE.fullmatch(self.container_cgroup_path) is None
        ):
            _fail("membership log container cgroup path differs")
        _require_int(
            self.container_cgroup_device,
            "membership log container cgroup device",
            minimum=1,
        )
        _require_int(
            self.container_cgroup_inode,
            "membership log container cgroup inode",
            minimum=1,
        )
        _require_int(self.host_pid, "membership log host PID", minimum=1)
        _require_int(
            self.host_process_start_time_ticks,
            "membership log process start time",
            minimum=1,
        )
        _require_sha256(self.observer_descriptor_sha256, "membership log observer descriptor")
        _require_sha256(self.observer_source_sha256, "membership log observer source")
        for artifact, schema, label in (
            (
                self.host_provisioning_receipt,
                HOST_PROVISIONING_RECEIPT_SCHEMA_VERSION,
                "host provisioning receipt",
            ),
            (self.observer_anchor, HOST_OBSERVER_ANCHOR_SCHEMA_VERSION, "observer anchor"),
            (self.initial_sample, HOST_INITIAL_CGROUP_SAMPLE_SCHEMA_VERSION, "initial sample"),
            (
                self.post_container_remove_sample,
                HOST_POST_CONTAINER_REMOVE_CGROUP_SAMPLE_SCHEMA_VERSION,
                "post-remove sample",
            ),
        ):
            _require_artifact_schema(artifact, schema, f"membership log {label}")
        if self.event_inventory_sha256 != _membership_event_inventory_sha256_v2(self.events):
            _fail("membership event inventory digest differs")
        if not (
            self.events[0].monotonic_ns
            < self.events[1].monotonic_ns
            < self.events[2].monotonic_ns
        ):
            _fail("membership event chronology differs")
        if any(
            _require_bool(value, label) is not False
            for value, label in (
                (
                    self.continuous_all_descendant_membership_proven,
                    "membership log continuous all-descendant membership",
                ),
                (
                    self.provisioning_receipt_semantics_validated,
                    "membership log provisioning semantics validation",
                ),
                (
                    self.provisioning_receipt_producer_authenticated,
                    "membership log provisioning producer authentication",
                ),
                (
                    self.production_containment_eligible,
                    "membership log production containment eligibility",
                ),
            )
        ):
            _fail("canonical membership snapshots cannot claim authenticated containment")

    def to_body_dict(self) -> dict[str, Any]:
        return {
            "schema_version": HOST_CGROUP_MEMBERSHIP_EVENT_LOG_SCHEMA_VERSION,
            "case_spine_sha256": self.case_spine_sha256,
            "cgroup_case_identity_sha256": self.cgroup_case_identity_sha256,
            "container_name": self.container_name,
            "container_lookup_identity_sha256": self.container_lookup_identity_sha256,
            "actual_runtime_container_identity_sha256": (
                self.actual_runtime_container_identity_sha256
            ),
            "actual_container_id": self.actual_container_id,
            "container_cgroup_path": self.container_cgroup_path,
            "container_cgroup_device": self.container_cgroup_device,
            "container_cgroup_inode": self.container_cgroup_inode,
            "host_pid": self.host_pid,
            "host_process_start_time_ticks": self.host_process_start_time_ticks,
            "observer_descriptor_sha256": self.observer_descriptor_sha256,
            "observer_source_sha256": self.observer_source_sha256,
            "host_provisioning_receipt": self.host_provisioning_receipt.to_dict(),
            "observer_anchor": self.observer_anchor.to_dict(),
            "initial_sample": self.initial_sample.to_dict(),
            "post_container_remove_sample": self.post_container_remove_sample.to_dict(),
            "events": [event.to_dict() for event in self.events],
            "event_inventory_sha256": self.event_inventory_sha256,
            "continuous_all_descendant_membership_proven": (
                self.continuous_all_descendant_membership_proven
            ),
            "provisioning_receipt_semantics_validated": (
                self.provisioning_receipt_semantics_validated
            ),
            "provisioning_receipt_producer_authenticated": (
                self.provisioning_receipt_producer_authenticated
            ),
            "production_containment_eligible": self.production_containment_eligible,
        }


MEMBERSHIP_EVENT_LOG_BODY_SHA256_FIELD: Final = "membership_event_log_body_sha256"


def canonical_host_cgroup_membership_event_log_v2_body_bytes(
    event_log: HostCgroupMembershipEventLogV2,
) -> bytes:
    return _canonical_artifact_body_bytes(event_log, HostCgroupMembershipEventLogV2)


def canonical_host_cgroup_membership_event_log_v2_file_bytes(
    event_log: HostCgroupMembershipEventLogV2,
) -> bytes:
    return _canonical_artifact_file_bytes(
        event_log,
        HostCgroupMembershipEventLogV2,
        MEMBERSHIP_EVENT_LOG_BODY_SHA256_FIELD,
    )


def parse_host_cgroup_membership_event_log_v2(
    raw: bytes,
    *,
    expected_file_sha256: str,
) -> HostCgroupMembershipEventLogV2:
    body = _parse_artifact_file(
        raw,
        expected_file_sha256=expected_file_sha256,
        body_field=MEMBERSHIP_EVENT_LOG_BODY_SHA256_FIELD,
        body_keys=frozenset(
            {"schema_version", *HostCgroupMembershipEventLogV2.__dataclass_fields__}
        ),
        label="host cgroup membership event log",
    )
    item = dict(body)
    if item.pop("schema_version") != HOST_CGROUP_MEMBERSHIP_EVENT_LOG_SCHEMA_VERSION:
        _fail("host cgroup membership event-log schema differs")
    for field in (
        "host_provisioning_receipt",
        "observer_anchor",
        "initial_sample",
        "post_container_remove_sample",
    ):
        item[field] = _artifact_identity(item[field], f"membership log {field}")
    events = item.pop("events")
    if type(events) is not list:
        _fail("membership event-log inventory must be one list")
    result = HostCgroupMembershipEventLogV2(
        **item,
        events=tuple(_membership_event_from_dict(event) for event in events),
    )
    if raw != canonical_host_cgroup_membership_event_log_v2_file_bytes(result):
        _fail("host cgroup membership event-log canonical replay differs")
    return result


@dataclass(frozen=True, slots=True)
class HostObserverTerminalEvidenceV2:
    """Canonical evidence identity spanning the first through final retained-FD sample."""

    case_spine_sha256: str
    observer_anchor: ArtifactIdentityV2
    host_provisioning_receipt: ArtifactIdentityV2
    initial_sample: ArtifactIdentityV2
    post_container_remove_sample: ArtifactIdentityV2
    membership_event_log: ArtifactIdentityV2
    membership_event_inventory_sha256: str
    continuous_all_descendant_membership_proven: bool
    provisioning_validated: bool
    provisioning_producer_authenticated: bool
    production_containment_eligible: bool

    def __post_init__(self) -> None:
        _require_sha256(self.case_spine_sha256, "observer-terminal case spine")
        for artifact, schema, label in (
            (self.observer_anchor, HOST_OBSERVER_ANCHOR_SCHEMA_VERSION, "observer anchor"),
            (
                self.host_provisioning_receipt,
                HOST_PROVISIONING_RECEIPT_SCHEMA_VERSION,
                "host provisioning receipt",
            ),
            (self.initial_sample, HOST_INITIAL_CGROUP_SAMPLE_SCHEMA_VERSION, "initial sample"),
            (
                self.post_container_remove_sample,
                HOST_POST_CONTAINER_REMOVE_CGROUP_SAMPLE_SCHEMA_VERSION,
                "post-remove sample",
            ),
            (
                self.membership_event_log,
                HOST_CGROUP_MEMBERSHIP_EVENT_LOG_SCHEMA_VERSION,
                "membership event log",
            ),
        ):
            _require_artifact_schema(artifact, schema, f"observer-terminal {label}")
        _require_sha256(
            self.membership_event_inventory_sha256,
            "observer-terminal membership-event inventory",
        )
        if (
            _require_bool(
                self.continuous_all_descendant_membership_proven,
                "continuous all-descendant membership proof",
            )
            is not False
            or _require_bool(
                self.provisioning_validated,
                "host provisioning validation",
            )
            is not False
            or _require_bool(
                self.provisioning_producer_authenticated,
                "host provisioning producer authentication",
            )
            is not False
            or _require_bool(
                self.production_containment_eligible,
                "production containment eligibility",
            )
            is not False
        ):
            _fail("source-only observer evidence cannot claim production containment")

    def to_dict(self) -> dict[str, Any]:
        return {
            field: (
                getattr(self, field).to_dict()
                if type(getattr(self, field)) is ArtifactIdentityV2
                else getattr(self, field)
            )
            for field in self.__dataclass_fields__
        }


def _observer_terminal_evidence_from_dict(value: object) -> HostObserverTerminalEvidenceV2:
    item = _require_exact_keys(
        value,
        frozenset(HostObserverTerminalEvidenceV2.__dataclass_fields__),
        "observer terminal evidence",
    )
    for field in (
        "observer_anchor",
        "host_provisioning_receipt",
        "initial_sample",
        "post_container_remove_sample",
        "membership_event_log",
    ):
        item[field] = _artifact_identity(item[field], f"observer-terminal {field}")
    return HostObserverTerminalEvidenceV2(**item)


def observer_terminal_evidence_sha256_v2(value: HostObserverTerminalEvidenceV2) -> str:
    if type(value) is not HostObserverTerminalEvidenceV2:
        raise TypeError("observer terminal evidence must use the exact v2 type")
    return _body_sha256(value.to_dict())


@dataclass(frozen=True, slots=True)
class HostCgroupBoundaryProofV2:
    case_spine_sha256: str
    operational_frontier: ArtifactIdentityV2
    initial_sample: ArtifactIdentityV2
    precleanup_sample: ArtifactIdentityV2
    cgroup_kill_receipt: ArtifactIdentityV2
    cgroup_empty_observation: ArtifactIdentityV2
    container_absence_observation: ArtifactIdentityV2
    post_container_remove_sample: ArtifactIdentityV2
    cgroup_counter_fds_closed_receipt: ArtifactIdentityV2
    outer_cgroup_absence_observation: ArtifactIdentityV2
    cgroup_case_identity: HostCgroupCaseIdentityV2
    retained_fd_set_sha256: str
    cgroup_identity_sha256: str
    container_name: str
    container_lookup_identity_sha256: str
    actual_runtime_container_identity_sha256: str | None
    actual_container_id: str | None
    observer_terminal_evidence: HostObserverTerminalEvidenceV2
    observer_terminal_evidence_sha256: str
    resources: HostRawResourceMeasurementsV2
    continuous_all_descendant_membership_proven: bool
    provisioning_validated: bool
    provisioning_producer_authenticated: bool
    production_containment_eligible: bool

    def __post_init__(self) -> None:
        _require_sha256(self.case_spine_sha256, "cgroup proof case spine")
        expected_schemas = (
            (self.operational_frontier, HOST_OPERATIONAL_FRONTIER_SCHEMA_VERSION),
            (self.initial_sample, HOST_INITIAL_CGROUP_SAMPLE_SCHEMA_VERSION),
            (self.precleanup_sample, HOST_PRECLEANUP_CGROUP_SAMPLE_SCHEMA_VERSION),
            (self.cgroup_kill_receipt, HOST_CGROUP_KILL_RECEIPT_SCHEMA_VERSION),
            (self.cgroup_empty_observation, HOST_CGROUP_EMPTY_OBSERVATION_SCHEMA_VERSION),
            (
                self.container_absence_observation,
                HOST_CONTAINER_ABSENCE_OBSERVATION_SCHEMA_VERSION,
            ),
            (
                self.post_container_remove_sample,
                HOST_POST_CONTAINER_REMOVE_CGROUP_SAMPLE_SCHEMA_VERSION,
            ),
            (
                self.cgroup_counter_fds_closed_receipt,
                HOST_CGROUP_COUNTER_FDS_CLOSED_RECEIPT_SCHEMA_VERSION,
            ),
            (
                self.outer_cgroup_absence_observation,
                HOST_OUTER_CGROUP_ABSENCE_OBSERVATION_SCHEMA_VERSION,
            ),
        )
        for artifact, schema in expected_schemas:
            _require_artifact_schema(artifact, schema, "cgroup proof input")
        for value, label in (
            (self.retained_fd_set_sha256, "cgroup proof retained-FD set"),
            (self.cgroup_identity_sha256, "cgroup proof cgroup identity"),
            (self.container_lookup_identity_sha256, "cgroup proof container lookup"),
            (self.observer_terminal_evidence_sha256, "observer-terminal evidence"),
        ):
            _require_sha256(value, label)
        if type(self.cgroup_case_identity) is not HostCgroupCaseIdentityV2:
            _fail("cgroup proof case identity type differs")
        if (
            self.cgroup_case_identity.case_spine_sha256 != self.case_spine_sha256
            or self.cgroup_identity_sha256
            != cgroup_case_identity_sha256_v2(self.cgroup_case_identity)
        ):
            _fail("cgroup proof directory identity differs")
        name = _require_text(self.container_name, "cgroup proof container name")
        if (
            name != expected_container_name_v2(self.case_spine_sha256)
            or self.container_lookup_identity_sha256
            != container_lookup_identity_sha256_v2(self.case_spine_sha256, name)
        ):
            _fail("cgroup proof container lookup identity differs")
        if (self.actual_runtime_container_identity_sha256 is None) is not (
            self.actual_container_id is None
        ):
            _fail("cgroup proof actual runtime identity and ID presence differ")
        if self.actual_container_id is not None:
            expected_actual = container_runtime_identity_sha256_v2(
                self.case_spine_sha256,
                name,
                self.actual_container_id,
            )
            if self.actual_runtime_container_identity_sha256 != expected_actual:
                _fail("cgroup proof actual runtime identity differs")
        if type(self.observer_terminal_evidence) is not HostObserverTerminalEvidenceV2:
            _fail("cgroup proof observer-terminal evidence type differs")
        if (
            self.observer_terminal_evidence.case_spine_sha256 != self.case_spine_sha256
            or self.observer_terminal_evidence_sha256
            != observer_terminal_evidence_sha256_v2(self.observer_terminal_evidence)
        ):
            _fail("cgroup proof observer-terminal evidence identity differs")
        if type(self.resources) is not HostRawResourceMeasurementsV2:
            _fail("cgroup proof resource-measurement type differs")
        if (
            _require_bool(
                self.continuous_all_descendant_membership_proven,
                "cgroup proof continuous all-descendant membership",
            )
            is not False
            or _require_bool(
                self.provisioning_validated,
                "cgroup proof provisioning validation",
            )
            is not False
            or _require_bool(
                self.provisioning_producer_authenticated,
                "cgroup proof provisioning producer authentication",
            )
            is not False
            or _require_bool(
                self.production_containment_eligible,
                "cgroup proof production containment eligibility",
            )
            is not False
        ):
            _fail("source-only cgroup proof cannot claim production containment")

    def to_body_dict(self) -> dict[str, Any]:
        return {
            "schema_version": HOST_CGROUP_PROOF_SCHEMA_VERSION,
            **{
                "case_spine_sha256": self.case_spine_sha256,
                "operational_frontier": self.operational_frontier.to_dict(),
                "initial_sample": self.initial_sample.to_dict(),
                "precleanup_sample": self.precleanup_sample.to_dict(),
                "cgroup_kill_receipt": self.cgroup_kill_receipt.to_dict(),
                "cgroup_empty_observation": self.cgroup_empty_observation.to_dict(),
                "container_absence_observation": self.container_absence_observation.to_dict(),
                "post_container_remove_sample": self.post_container_remove_sample.to_dict(),
                "cgroup_counter_fds_closed_receipt": (
                    self.cgroup_counter_fds_closed_receipt.to_dict()
                ),
                "outer_cgroup_absence_observation": (
                    self.outer_cgroup_absence_observation.to_dict()
                ),
                "cgroup_case_identity": self.cgroup_case_identity.to_dict(),
                "retained_fd_set_sha256": self.retained_fd_set_sha256,
                "cgroup_identity_sha256": self.cgroup_identity_sha256,
                "container_name": self.container_name,
                "container_lookup_identity_sha256": self.container_lookup_identity_sha256,
                "actual_runtime_container_identity_sha256": (
                    self.actual_runtime_container_identity_sha256
                ),
                "actual_container_id": self.actual_container_id,
                "observer_terminal_evidence": self.observer_terminal_evidence.to_dict(),
                "observer_terminal_evidence_sha256": (
                    self.observer_terminal_evidence_sha256
                ),
                "resources": self.resources.to_dict(),
                "continuous_all_descendant_membership_proven": (
                    self.continuous_all_descendant_membership_proven
                ),
                "provisioning_validated": self.provisioning_validated,
                "provisioning_producer_authenticated": (
                    self.provisioning_producer_authenticated
                ),
                "production_containment_eligible": self.production_containment_eligible,
            },
            "claims": _claims(),
        }


CGROUP_PROOF_BODY_SHA256_FIELD: Final = "cgroup_proof_body_sha256"


def canonical_host_cgroup_boundary_proof_v2_body_bytes(
    proof: HostCgroupBoundaryProofV2,
) -> bytes:
    return _canonical_artifact_body_bytes(proof, HostCgroupBoundaryProofV2)


def canonical_host_cgroup_boundary_proof_v2_file_bytes(
    proof: HostCgroupBoundaryProofV2,
) -> bytes:
    return _canonical_artifact_file_bytes(
        proof,
        HostCgroupBoundaryProofV2,
        CGROUP_PROOF_BODY_SHA256_FIELD,
    )


def parse_host_cgroup_boundary_proof_v2(
    raw: bytes,
    *,
    expected_file_sha256: str,
) -> HostCgroupBoundaryProofV2:
    body = _parse_artifact_file(
        raw,
        expected_file_sha256=expected_file_sha256,
        body_field=CGROUP_PROOF_BODY_SHA256_FIELD,
        body_keys=frozenset(
            {"schema_version", *HostCgroupBoundaryProofV2.__dataclass_fields__, "claims"}
        ),
        label="host cgroup boundary proof",
    )
    item = dict(body)
    if (
        item.pop("schema_version") != HOST_CGROUP_PROOF_SCHEMA_VERSION
        or item.pop("claims") != _claims()
    ):
        _fail("host cgroup boundary proof envelope differs")
    for field in (
        "operational_frontier",
        "initial_sample",
        "precleanup_sample",
        "cgroup_kill_receipt",
        "cgroup_empty_observation",
        "container_absence_observation",
        "post_container_remove_sample",
        "cgroup_counter_fds_closed_receipt",
        "outer_cgroup_absence_observation",
    ):
        item[field] = _artifact_identity(item[field], f"cgroup proof {field}")
    item["cgroup_case_identity"] = _cgroup_case_identity_from_dict(
        item["cgroup_case_identity"]
    )
    item["observer_terminal_evidence"] = _observer_terminal_evidence_from_dict(
        item["observer_terminal_evidence"]
    )
    item["resources"] = _raw_resources_from_dict(item["resources"])
    result = HostCgroupBoundaryProofV2(**item)
    if raw != canonical_host_cgroup_boundary_proof_v2_file_bytes(result):
        _fail("host cgroup boundary proof canonical replay differs")
    return result


def _identity_from_canonical_bytes(
    schema_version: str,
    body_bytes: bytes,
    file_bytes: bytes,
) -> ArtifactIdentityV2:
    return ArtifactIdentityV2(
        schema_version=schema_version,
        file_sha256=_sha256(file_bytes),
        body_sha256=_sha256(body_bytes),
    )


def _require_identity_match(
    supplied: ArtifactIdentityV2,
    *,
    schema_version: str,
    body_bytes: bytes,
    file_bytes: bytes,
    label: str,
) -> None:
    expected = _identity_from_canonical_bytes(schema_version, body_bytes, file_bytes)
    if supplied != expected:
        _fail(f"{label} does not bind the exact canonical FILE and BODY")


def _frontier_identity(frontier: HostOperationalFrontierV2) -> ArtifactIdentityV2:
    return _identity_from_canonical_bytes(
        HOST_OPERATIONAL_FRONTIER_SCHEMA_VERSION,
        canonical_host_operational_frontier_v2_body_bytes(frontier),
        canonical_host_operational_frontier_v2_file_bytes(frontier),
    )


def validate_host_cgroup_boundary_proof_v2_chain(
    frontier: HostOperationalFrontierV2,
    initial_sample: HostInitialCgroupSampleV2,
    precleanup_sample: HostPrecleanupCgroupSampleV2,
    cgroup_kill_receipt: HostCgroupKillReceiptV2,
    cgroup_empty_observation: HostCgroupEmptyObservationV2,
    container_absence_observation: HostContainerAbsenceObservationV2,
    post_container_remove_sample: HostPostContainerRemoveCgroupSampleV2,
    cgroup_counter_fds_closed_receipt: HostCgroupCounterFdsClosedReceiptV2,
    outer_cgroup_absence_observation: HostOuterCgroupAbsenceObservationV2,
    proof: HostCgroupBoundaryProofV2,
    *,
    request: HostQualificationCaseRequestV2,
    intent: HostQualificationCaseIntentV2,
    ready: HostReadyV2,
    observer_anchor: HostObserverAnchorV2,
    go: HostGoCommitmentV2,
    membership_event_log: HostCgroupMembershipEventLogV2,
) -> None:
    """Validate the strict retained-FD proof path without performing host operations."""

    expected_types = (
        (frontier, HostOperationalFrontierV2),
        (initial_sample, HostInitialCgroupSampleV2),
        (precleanup_sample, HostPrecleanupCgroupSampleV2),
        (cgroup_kill_receipt, HostCgroupKillReceiptV2),
        (cgroup_empty_observation, HostCgroupEmptyObservationV2),
        (container_absence_observation, HostContainerAbsenceObservationV2),
        (post_container_remove_sample, HostPostContainerRemoveCgroupSampleV2),
        (cgroup_counter_fds_closed_receipt, HostCgroupCounterFdsClosedReceiptV2),
        (outer_cgroup_absence_observation, HostOuterCgroupAbsenceObservationV2),
        (proof, HostCgroupBoundaryProofV2),
        (request, HostQualificationCaseRequestV2),
        (intent, HostQualificationCaseIntentV2),
        (ready, HostReadyV2),
        (observer_anchor, HostObserverAnchorV2),
        (go, HostGoCommitmentV2),
        (membership_event_log, HostCgroupMembershipEventLogV2),
    )
    if any(type(value) is not expected for value, expected in expected_types):
        raise TypeError("cgroup proof chain values must use their exact v2 types")
    spine = frontier.case_spine_sha256
    if any(value.case_spine_sha256 != spine for value, _ in expected_types[1:]):
        _fail("cgroup proof chain crosses case spines")
    validate_host_request_intent_v2_chain(request, intent)
    validate_host_ready_anchor_go_v2_chain(
        intent,
        initial_sample,
        ready,
        observer_anchor,
        go,
    )

    frontier_id = _frontier_identity(frontier)
    initial_id = _identity_from_canonical_bytes(
        HOST_INITIAL_CGROUP_SAMPLE_SCHEMA_VERSION,
        canonical_host_initial_cgroup_sample_v2_body_bytes(initial_sample),
        canonical_host_initial_cgroup_sample_v2_file_bytes(initial_sample),
    )
    precleanup_id = _identity_from_canonical_bytes(
        HOST_PRECLEANUP_CGROUP_SAMPLE_SCHEMA_VERSION,
        canonical_host_precleanup_cgroup_sample_v2_body_bytes(precleanup_sample),
        canonical_host_precleanup_cgroup_sample_v2_file_bytes(precleanup_sample),
    )
    kill_id = _identity_from_canonical_bytes(
        HOST_CGROUP_KILL_RECEIPT_SCHEMA_VERSION,
        canonical_host_cgroup_kill_receipt_v2_body_bytes(cgroup_kill_receipt),
        canonical_host_cgroup_kill_receipt_v2_file_bytes(cgroup_kill_receipt),
    )
    empty_id = _identity_from_canonical_bytes(
        HOST_CGROUP_EMPTY_OBSERVATION_SCHEMA_VERSION,
        canonical_host_cgroup_empty_observation_v2_body_bytes(cgroup_empty_observation),
        canonical_host_cgroup_empty_observation_v2_file_bytes(cgroup_empty_observation),
    )
    container_id = _identity_from_canonical_bytes(
        HOST_CONTAINER_ABSENCE_OBSERVATION_SCHEMA_VERSION,
        canonical_host_container_absence_observation_v2_body_bytes(
            container_absence_observation
        ),
        canonical_host_container_absence_observation_v2_file_bytes(
            container_absence_observation
        ),
    )
    post_id = _identity_from_canonical_bytes(
        HOST_POST_CONTAINER_REMOVE_CGROUP_SAMPLE_SCHEMA_VERSION,
        canonical_host_post_container_remove_cgroup_sample_v2_body_bytes(
            post_container_remove_sample
        ),
        canonical_host_post_container_remove_cgroup_sample_v2_file_bytes(
            post_container_remove_sample
        ),
    )
    close_id = _identity_from_canonical_bytes(
        HOST_CGROUP_COUNTER_FDS_CLOSED_RECEIPT_SCHEMA_VERSION,
        canonical_host_cgroup_counter_fds_closed_receipt_v2_body_bytes(
            cgroup_counter_fds_closed_receipt
        ),
        canonical_host_cgroup_counter_fds_closed_receipt_v2_file_bytes(
            cgroup_counter_fds_closed_receipt
        ),
    )
    outer_id = _identity_from_canonical_bytes(
        HOST_OUTER_CGROUP_ABSENCE_OBSERVATION_SCHEMA_VERSION,
        canonical_host_outer_cgroup_absence_observation_v2_body_bytes(
            outer_cgroup_absence_observation
        ),
        canonical_host_outer_cgroup_absence_observation_v2_file_bytes(
            outer_cgroup_absence_observation
        ),
    )
    anchor_id = _identity_from_canonical_bytes(
        HOST_OBSERVER_ANCHOR_SCHEMA_VERSION,
        canonical_host_observer_anchor_v2_body_bytes(observer_anchor),
        canonical_host_observer_anchor_v2_file_bytes(observer_anchor),
    )
    if (
        precleanup_sample.operational_frontier != frontier_id
        or cgroup_kill_receipt.precleanup_sample != precleanup_id
        or cgroup_empty_observation.precleanup_sample != precleanup_id
        or cgroup_empty_observation.cgroup_kill_receipt != kill_id
        or container_absence_observation.operational_frontier != frontier_id
        or container_absence_observation.cgroup_empty_observation != empty_id
        or post_container_remove_sample.container_absence_observation != container_id
        or cgroup_counter_fds_closed_receipt.post_container_remove_sample != post_id
        or outer_cgroup_absence_observation.cgroup_counter_fds_closed_receipt != close_id
    ):
        _fail("cgroup proof retained-FD chain is cross-wired")
    proof_links = (
        (proof.operational_frontier, frontier_id),
        (proof.initial_sample, initial_id),
        (proof.precleanup_sample, precleanup_id),
        (proof.cgroup_kill_receipt, kill_id),
        (proof.cgroup_empty_observation, empty_id),
        (proof.container_absence_observation, container_id),
        (proof.post_container_remove_sample, post_id),
        (proof.cgroup_counter_fds_closed_receipt, close_id),
        (proof.outer_cgroup_absence_observation, outer_id),
    )
    if any(actual != expected for actual, expected in proof_links):
        _fail("cgroup proof does not bind every exact recovery artifact")
    _validate_container_absence_against_frontier(frontier, container_absence_observation)
    _validate_cgroup_sample_progression_v2(
        initial_sample.facts,
        precleanup_sample.facts,
        "initial-to-precleanup retained-FD sample",
    )
    _validate_cgroup_sample_progression_v2(
        precleanup_sample.facts,
        post_container_remove_sample.facts,
        "precleanup-to-post-remove retained-FD sample",
    )
    case_identity = initial_sample.cgroup_case_identity
    if (
        proof.cgroup_case_identity != case_identity
        or outer_cgroup_absence_observation.cgroup_case_identity != case_identity
        or outer_cgroup_absence_observation.case_cgroup_path != case_identity.case_cgroup_path
    ):
        _fail("cgroup proof directory/device/inode identity changes")
    retained_fd_set = initial_sample.facts.retained_fd_set_sha256
    cgroup_identity = initial_sample.facts.cgroup_identity_sha256
    if any(
        value != retained_fd_set
        for value in (
            precleanup_sample.facts.retained_fd_set_sha256,
            cgroup_kill_receipt.retained_fd_set_sha256,
            cgroup_empty_observation.retained_fd_set_sha256,
            post_container_remove_sample.facts.retained_fd_set_sha256,
            cgroup_counter_fds_closed_receipt.retained_fd_set_sha256,
            proof.retained_fd_set_sha256,
        )
    ) or any(
        value != cgroup_identity
        for value in (
            precleanup_sample.facts.cgroup_identity_sha256,
            cgroup_kill_receipt.cgroup_identity_sha256,
            cgroup_empty_observation.cgroup_identity_sha256,
            post_container_remove_sample.facts.cgroup_identity_sha256,
            cgroup_counter_fds_closed_receipt.cgroup_identity_sha256,
            outer_cgroup_absence_observation.cgroup_identity_sha256,
            proof.cgroup_identity_sha256,
        )
    ):
        _fail("cgroup proof retained-FD set or cgroup identity changes")
    container_projection = (
        container_absence_observation.container_name,
        container_absence_observation.container_lookup_identity_sha256,
        container_absence_observation.actual_runtime_container_identity_sha256,
        container_absence_observation.actual_container_id,
    )
    ready_projection = (
        ready.container_name,
        container_lookup_identity_sha256_v2(spine, ready.container_name),
        ready.container_identity_sha256,
        ready.container_id,
    )
    if container_projection != ready_projection:
        _fail("container absence does not bind the exact committed READY identity")
    if any(
        (
            value.container_name,
            value.container_lookup_identity_sha256,
            value.actual_runtime_container_identity_sha256,
            value.actual_container_id,
        )
        != container_projection
        for value in (post_container_remove_sample, cgroup_counter_fds_closed_receipt, proof)
    ):
        _fail("container lookup or optional actual runtime identity changes")
    if not (
        go.go_committed_monotonic_ns
        < precleanup_sample.facts.monotonic_ns
        < cgroup_kill_receipt.kill_monotonic_ns
        < cgroup_empty_observation.observed_monotonic_ns
    ):
        _fail("cgroup proof GO/precleanup/kill/empty chronology differs")
    if not (
        initial_sample.facts.monotonic_ns
        < cgroup_empty_observation.observed_monotonic_ns
        < container_absence_observation.removal_monotonic_ns
        < post_container_remove_sample.facts.monotonic_ns
        < cgroup_counter_fds_closed_receipt.close_monotonic_ns
        < outer_cgroup_absence_observation.removal_monotonic_ns
    ):
        _fail("retained-FD post-remove/close/outer-absence chronology differs")
    resources = proof.resources
    post_facts = post_container_remove_sample.facts
    if (
        resources.memory_peak_bytes != post_facts.memory_peak_bytes
        or resources.memory_oom_kill_count != post_facts.memory_oom_kill_count
        or resources.initial_cpu_usage_usec != initial_sample.facts.cpu_usage_usec
        or resources.post_remove_cpu_usage_usec != post_facts.cpu_usage_usec
        or resources.initial_monotonic_ns != initial_sample.facts.monotonic_ns
        or resources.post_remove_monotonic_ns != post_facts.monotonic_ns
        or resources.pids_peak != post_facts.pids_peak
        or resources.pids_max_event_count != post_facts.pids_max_event_count
        or frontier.attempt_count_state != "exact"
        or resources.attempt_count != frontier.attempt_count
        or resources.failure_count != frontier.failure_count
    ):
        _fail("cgroup proof resource values differ from retained-FD observations")
    if resources.memory_oom_kill_count or resources.pids_max_event_count:
        _fail("cgroup proof cannot claim structural success after a kernel quota breach")
    evidence = proof.observer_terminal_evidence
    event_log_id = _identity_from_canonical_bytes(
        HOST_CGROUP_MEMBERSHIP_EVENT_LOG_SCHEMA_VERSION,
        canonical_host_cgroup_membership_event_log_v2_body_bytes(membership_event_log),
        canonical_host_cgroup_membership_event_log_v2_file_bytes(membership_event_log),
    )
    if (
        evidence.observer_anchor != anchor_id
        or evidence.host_provisioning_receipt != request.host_provisioning_receipt
        or evidence.initial_sample != initial_id
        or evidence.post_container_remove_sample != post_id
        or evidence.membership_event_log != event_log_id
        or evidence.membership_event_inventory_sha256
        != membership_event_log.event_inventory_sha256
        or observer_anchor.initial_cgroup_sample != initial_id
        or observer_anchor.retained_fd_set_sha256 != proof.retained_fd_set_sha256
        or observer_anchor.cgroup_identity_sha256 != proof.cgroup_identity_sha256
    ):
        _fail("observer-terminal event evidence is cross-wired or does not span cleanup")
    if (
        membership_event_log.case_spine_sha256 != spine
        or membership_event_log.cgroup_case_identity_sha256
        != cgroup_case_identity_sha256_v2(case_identity)
        or membership_event_log.container_name != ready.container_name
        or membership_event_log.container_lookup_identity_sha256
        != container_lookup_identity_sha256_v2(spine, ready.container_name)
        or membership_event_log.actual_runtime_container_identity_sha256
        != ready.container_identity_sha256
        or membership_event_log.actual_container_id != ready.container_id
        or membership_event_log.container_cgroup_path != ready.container_cgroup_path
        or membership_event_log.container_cgroup_device != ready.container_cgroup_device
        or membership_event_log.container_cgroup_inode != ready.container_cgroup_inode
        or membership_event_log.host_pid != ready.host_pid
        or membership_event_log.host_process_start_time_ticks
        != ready.host_process_start_time_ticks
        or membership_event_log.observer_descriptor_sha256
        != observer_anchor.observer_descriptor_sha256
        or membership_event_log.observer_source_sha256
        != observer_anchor.observer_source_sha256
        or membership_event_log.host_provisioning_receipt
        != request.host_provisioning_receipt
        or membership_event_log.observer_anchor != anchor_id
        or membership_event_log.initial_sample != initial_id
        or membership_event_log.post_container_remove_sample != post_id
        or membership_event_log.events[0].monotonic_ns
        != initial_sample.facts.monotonic_ns
        or membership_event_log.events[1].monotonic_ns
        != observer_anchor.observer_started_monotonic_ns
        or membership_event_log.events[2].monotonic_ns
        != post_container_remove_sample.facts.monotonic_ns
        or observer_anchor.observer_descriptor_sha256
        != request.observation_registry.descriptor_sha256
        or observer_anchor.observer_source_sha256
        != request.observation_registry.source_sha256
    ):
        _fail("membership event log does not bind the exact observed process/cgroup chain")


type RecoveryStateV2 = Literal[
    "not_applicable", "committed", "commit_uncertain", "failed_before_commit"
]


@dataclass(frozen=True, slots=True)
class RecoveryNodeV2:
    node_name: str
    state: RecoveryStateV2
    artifact: ArtifactIdentityV2 | None
    dependencies: tuple[str, ...]
    uncertainty_detail_sha256: str | None

    def __post_init__(self) -> None:
        name = _require_text(self.node_name, "recovery node name")
        state = _require_text(self.state, "recovery node state")
        if name not in RECOVERY_NODE_NAMES:
            _fail("recovery node name differs")
        if state not in {
            "not_applicable",
            "committed",
            "commit_uncertain",
            "failed_before_commit",
        }:
            _fail("recovery node state differs")
        if (
            type(self.dependencies) is not tuple
            or any(type(item) is not str for item in self.dependencies)
            or self.dependencies != RECOVERY_NODE_DEPENDENCIES[name]
        ):
            _fail("recovery node dependencies differ from the conditional DAG")
        if state == "committed":
            if self.artifact is None:
                _fail("committed recovery node lacks its artifact")
            _require_artifact_schema(
                self.artifact,
                RECOVERY_NODE_SCHEMAS[name],
                f"recovery node {name}",
            )
            if self.uncertainty_detail_sha256 is not None:
                _fail("committed recovery node cannot carry uncertainty detail")
        else:
            if self.artifact is not None:
                _fail("uncommitted recovery node cannot carry a committed artifact")
            if state in {"commit_uncertain", "failed_before_commit"}:
                _require_sha256(
                    self.uncertainty_detail_sha256,
                    f"recovery node {name} uncertainty detail",
                )
            elif self.uncertainty_detail_sha256 is not None:
                _fail("not-applicable recovery node cannot carry uncertainty detail")

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_name": self.node_name,
            "state": self.state,
            "artifact": None if self.artifact is None else self.artifact.to_dict(),
            "dependencies": list(self.dependencies),
            "uncertainty_detail_sha256": self.uncertainty_detail_sha256,
        }


def _recovery_node_from_dict(value: object) -> RecoveryNodeV2:
    item = _require_exact_keys(
        value,
        frozenset(RecoveryNodeV2.__dataclass_fields__),
        "recovery node",
    )
    item["artifact"] = _optional_artifact_identity(item["artifact"], "recovery node artifact")
    dependencies = item.pop("dependencies")
    if type(dependencies) is not list:
        _fail("recovery node dependencies must be one list")
    return RecoveryNodeV2(**item, dependencies=tuple(dependencies))


@dataclass(frozen=True, slots=True)
class HostCleanupReconciliationV2:
    case_spine_sha256: str
    operational_frontier: ArtifactIdentityV2
    cgroup_may_exist: bool
    recovery_nodes: tuple[RecoveryNodeV2, ...]
    cleanup_proven: bool
    unresolved_recovery_nodes: tuple[str, ...]
    recovery_complete: bool
    terminalization_permitted: bool
    workload_resume_permitted: bool
    same_case_retry_permitted: bool

    def __post_init__(self) -> None:
        _require_sha256(self.case_spine_sha256, "cleanup reconciliation case spine")
        _require_artifact_schema(
            self.operational_frontier,
            HOST_OPERATIONAL_FRONTIER_SCHEMA_VERSION,
            "cleanup operational frontier",
        )
        cgroup_may_exist = _require_bool(self.cgroup_may_exist, "cleanup cgroup may exist")
        if (
            type(self.recovery_nodes) is not tuple
            or any(type(item) is not RecoveryNodeV2 for item in self.recovery_nodes)
            or tuple(item.node_name for item in self.recovery_nodes) != RECOVERY_NODE_NAMES
        ):
            _fail("cleanup recovery-node inventory or order differs")
        states = {item.node_name: item.state for item in self.recovery_nodes}
        if cgroup_may_exist:
            if any(state == "not_applicable" for state in states.values()):
                _fail("possibly-created cgroup requires an outcome for every recovery node")
        elif (
            states["container_absence"] == "not_applicable"
            or any(
                states[name] != "not_applicable"
                for name in RECOVERY_NODE_NAMES
                if name != "container_absence"
            )
        ):
            _fail("a never-created cgroup still requires exact-name container reconciliation")
        for node in self.recovery_nodes:
            if node.state in {"committed", "commit_uncertain"} and any(
                states[dependency] != "committed" for dependency in node.dependencies
            ):
                _fail("attempted recovery node lacks all committed prerequisites")
        applicable = (
            RECOVERY_NODE_NAMES if cgroup_may_exist else ("container_absence",)
        )
        expected_cleanup_proven = all(states[name] == "committed" for name in applicable)
        if _require_bool(self.cleanup_proven, "cleanup proven") is not expected_cleanup_proven:
            _fail("cleanup proof state differs from recovery outcomes")
        expected_unresolved = tuple(
            name
            for name in RECOVERY_NODE_NAMES
            if states[name] in {"commit_uncertain", "failed_before_commit"}
        )
        if (
            type(self.unresolved_recovery_nodes) is not tuple
            or any(type(item) is not str for item in self.unresolved_recovery_nodes)
            or self.unresolved_recovery_nodes != expected_unresolved
        ):
            _fail("cleanup unresolved-node projection differs")
        if _require_bool(self.recovery_complete, "recovery complete") is not True:
            _fail("cleanup reconciliation must record a terminal outcome for every branch")
        if _require_bool(self.terminalization_permitted, "terminalization permitted") is not True:
            _fail("cleanup uncertainty cannot suppress terminal recording")
        if _require_bool(self.workload_resume_permitted, "workload resume") is not False:
            _fail("cleanup reconciliation cannot resume a consumed workload")
        if _require_bool(self.same_case_retry_permitted, "cleanup retry") is not False:
            _fail("cleanup reconciliation cannot permit same-case retry")

    def to_body_dict(self) -> dict[str, Any]:
        return {
            "schema_version": HOST_CLEANUP_RECONCILIATION_SCHEMA_VERSION,
            "status": "conditional_recovery_dag_reconciled_non_authorizing",
            "case_spine_sha256": self.case_spine_sha256,
            "operational_frontier": self.operational_frontier.to_dict(),
            "cgroup_may_exist": self.cgroup_may_exist,
            "recovery_nodes": [item.to_dict() for item in self.recovery_nodes],
            "cleanup_proven": self.cleanup_proven,
            "unresolved_recovery_nodes": list(self.unresolved_recovery_nodes),
            "recovery_complete": self.recovery_complete,
            "terminalization_permitted": self.terminalization_permitted,
            "workload_resume_permitted": self.workload_resume_permitted,
            "same_case_retry_permitted": self.same_case_retry_permitted,
            "claims": _claims(),
        }


CLEANUP_RECONCILIATION_BODY_SHA256_FIELD: Final = "cleanup_reconciliation_body_sha256"


def canonical_host_cleanup_reconciliation_v2_body_bytes(
    reconciliation: HostCleanupReconciliationV2,
) -> bytes:
    return _canonical_artifact_body_bytes(reconciliation, HostCleanupReconciliationV2)


def canonical_host_cleanup_reconciliation_v2_file_bytes(
    reconciliation: HostCleanupReconciliationV2,
) -> bytes:
    return _canonical_artifact_file_bytes(
        reconciliation,
        HostCleanupReconciliationV2,
        CLEANUP_RECONCILIATION_BODY_SHA256_FIELD,
    )


def parse_host_cleanup_reconciliation_v2(
    raw: bytes,
    *,
    expected_file_sha256: str,
) -> HostCleanupReconciliationV2:
    body = _parse_artifact_file(
        raw,
        expected_file_sha256=expected_file_sha256,
        body_field=CLEANUP_RECONCILIATION_BODY_SHA256_FIELD,
        body_keys=frozenset(HostCleanupReconciliationV2.__dataclass_fields__)
        | {"schema_version", "status", "claims"},
        label="host cleanup reconciliation",
    )
    item = dict(body)
    if (
        item.pop("schema_version") != HOST_CLEANUP_RECONCILIATION_SCHEMA_VERSION
        or item.pop("status") != "conditional_recovery_dag_reconciled_non_authorizing"
        or item.pop("claims") != _claims()
    ):
        _fail("host cleanup reconciliation envelope differs")
    item["operational_frontier"] = _artifact_identity(
        item["operational_frontier"],
        "cleanup operational frontier",
    )
    raw_nodes = item.pop("recovery_nodes")
    raw_unresolved = item.pop("unresolved_recovery_nodes")
    if type(raw_nodes) is not list or type(raw_unresolved) is not list:
        _fail("cleanup node inventories must be exact lists")
    result = HostCleanupReconciliationV2(
        **item,
        recovery_nodes=tuple(_recovery_node_from_dict(child) for child in raw_nodes),
        unresolved_recovery_nodes=tuple(raw_unresolved),
    )
    if raw != canonical_host_cleanup_reconciliation_v2_file_bytes(result):
        _fail("host cleanup reconciliation canonical replay differs")
    return result


def validate_host_cleanup_reconciliation_v2_chain(
    frontier: HostOperationalFrontierV2,
    reconciliation: HostCleanupReconciliationV2,
    *,
    request: HostQualificationCaseRequestV2 | None = None,
    intent: HostQualificationCaseIntentV2 | None = None,
    ready: HostReadyV2 | None = None,
    observer_anchor: HostObserverAnchorV2 | None = None,
    go: HostGoCommitmentV2 | None = None,
    membership_event_log: HostCgroupMembershipEventLogV2 | None = None,
    initial_sample: HostInitialCgroupSampleV2 | None = None,
    precleanup_sample: HostPrecleanupCgroupSampleV2 | None = None,
    cgroup_kill_receipt: HostCgroupKillReceiptV2 | None = None,
    cgroup_empty_observation: HostCgroupEmptyObservationV2 | None = None,
    container_absence_observation: HostContainerAbsenceObservationV2 | None = None,
    post_container_remove_sample: HostPostContainerRemoveCgroupSampleV2 | None = None,
    cgroup_counter_fds_closed_receipt: HostCgroupCounterFdsClosedReceiptV2 | None = None,
    outer_cgroup_absence_observation: HostOuterCgroupAbsenceObservationV2 | None = None,
    cgroup_proof: HostCgroupBoundaryProofV2 | None = None,
) -> None:
    """Validate committed DAG nodes while allowing unrelated branches to continue."""

    if type(frontier) is not HostOperationalFrontierV2:
        raise TypeError("frontier must use the exact operational-frontier-v2 type")
    if type(reconciliation) is not HostCleanupReconciliationV2:
        raise TypeError("reconciliation must use the exact cleanup-reconciliation-v2 type")
    auxiliary_types: tuple[tuple[object | None, type[Any]], ...] = (
        (request, HostQualificationCaseRequestV2),
        (intent, HostQualificationCaseIntentV2),
        (ready, HostReadyV2),
        (observer_anchor, HostObserverAnchorV2),
        (go, HostGoCommitmentV2),
        (membership_event_log, HostCgroupMembershipEventLogV2),
        (initial_sample, HostInitialCgroupSampleV2),
        (precleanup_sample, HostPrecleanupCgroupSampleV2),
        (cgroup_kill_receipt, HostCgroupKillReceiptV2),
        (cgroup_empty_observation, HostCgroupEmptyObservationV2),
        (container_absence_observation, HostContainerAbsenceObservationV2),
        (post_container_remove_sample, HostPostContainerRemoveCgroupSampleV2),
        (cgroup_counter_fds_closed_receipt, HostCgroupCounterFdsClosedReceiptV2),
        (outer_cgroup_absence_observation, HostOuterCgroupAbsenceObservationV2),
        (cgroup_proof, HostCgroupBoundaryProofV2),
    )
    if any(
        value is not None and type(value) is not expected
        for value, expected in auxiliary_types
    ):
        raise TypeError("cleanup auxiliary artifacts must use their exact v2 types")
    frontier_id = _frontier_identity(frontier)
    if (
        reconciliation.case_spine_sha256 != frontier.case_spine_sha256
        or reconciliation.operational_frontier != frontier_id
    ):
        _fail("cleanup reconciliation crosses its operational frontier")
    expected_cgroup_may_exist = (
        "fresh_cgroup_created" in frontier.completed_phases
        or (
            frontier.failure_phase == "fresh_cgroup_created"
            and frontier.failure_effect_state == "commit_uncertain"
        )
    )
    if reconciliation.cgroup_may_exist is not expected_cgroup_may_exist:
        _fail("cleanup cgroup-existence projection differs from the operational frontier")

    actuals: dict[str, object | None] = {
        "precleanup_cgroup_sample": precleanup_sample,
        "cgroup_kill": cgroup_kill_receipt,
        "cgroup_empty": cgroup_empty_observation,
        "container_absence": container_absence_observation,
        "post_container_remove_cgroup_sample": post_container_remove_sample,
        "cgroup_counter_fds_closed": cgroup_counter_fds_closed_receipt,
        "outer_cgroup_absence": outer_cgroup_absence_observation,
        "final_cgroup_proof": cgroup_proof,
    }
    expected_types: Mapping[str, type[Any]] = {
        "precleanup_cgroup_sample": HostPrecleanupCgroupSampleV2,
        "cgroup_kill": HostCgroupKillReceiptV2,
        "cgroup_empty": HostCgroupEmptyObservationV2,
        "container_absence": HostContainerAbsenceObservationV2,
        "post_container_remove_cgroup_sample": HostPostContainerRemoveCgroupSampleV2,
        "cgroup_counter_fds_closed": HostCgroupCounterFdsClosedReceiptV2,
        "outer_cgroup_absence": HostOuterCgroupAbsenceObservationV2,
        "final_cgroup_proof": HostCgroupBoundaryProofV2,
    }
    canonical_bytes: dict[
        str,
        tuple[Callable[[Any], bytes], Callable[[Any], bytes]],
    ] = {
        "precleanup_cgroup_sample": (
            canonical_host_precleanup_cgroup_sample_v2_body_bytes,
            canonical_host_precleanup_cgroup_sample_v2_file_bytes,
        ),
        "cgroup_kill": (
            canonical_host_cgroup_kill_receipt_v2_body_bytes,
            canonical_host_cgroup_kill_receipt_v2_file_bytes,
        ),
        "cgroup_empty": (
            canonical_host_cgroup_empty_observation_v2_body_bytes,
            canonical_host_cgroup_empty_observation_v2_file_bytes,
        ),
        "container_absence": (
            canonical_host_container_absence_observation_v2_body_bytes,
            canonical_host_container_absence_observation_v2_file_bytes,
        ),
        "post_container_remove_cgroup_sample": (
            canonical_host_post_container_remove_cgroup_sample_v2_body_bytes,
            canonical_host_post_container_remove_cgroup_sample_v2_file_bytes,
        ),
        "cgroup_counter_fds_closed": (
            canonical_host_cgroup_counter_fds_closed_receipt_v2_body_bytes,
            canonical_host_cgroup_counter_fds_closed_receipt_v2_file_bytes,
        ),
        "outer_cgroup_absence": (
            canonical_host_outer_cgroup_absence_observation_v2_body_bytes,
            canonical_host_outer_cgroup_absence_observation_v2_file_bytes,
        ),
        "final_cgroup_proof": (
            canonical_host_cgroup_boundary_proof_v2_body_bytes,
            canonical_host_cgroup_boundary_proof_v2_file_bytes,
        ),
    }
    node_by_name = {node.node_name: node for node in reconciliation.recovery_nodes}
    for name in RECOVERY_NODE_NAMES:
        node = node_by_name[name]
        actual = actuals[name]
        if node.state != "committed":
            if actual is not None:
                _fail(f"uncommitted recovery node {name} received an artifact value")
            continue
        if actual is None or type(actual) is not expected_types[name]:
            _fail(f"committed recovery node {name} lacks its exact artifact value")
        if cast(Any, actual).case_spine_sha256 != frontier.case_spine_sha256:
            _fail(f"recovery node {name} crosses case spines")
        body_builder, file_builder = canonical_bytes[name]
        _require_identity_match(
            cast(ArtifactIdentityV2, node.artifact),
            schema_version=RECOVERY_NODE_SCHEMAS[name],
            body_bytes=body_builder(cast(Any, actual)),
            file_bytes=file_builder(cast(Any, actual)),
            label=f"recovery node {name}",
        )

    if precleanup_sample is not None and precleanup_sample.operational_frontier != frontier_id:
        _fail("precleanup sample does not bind the operational frontier")
    if initial_sample is not None and precleanup_sample is not None:
        _validate_cgroup_sample_progression_v2(
            initial_sample.facts,
            precleanup_sample.facts,
            "initial-to-precleanup retained-FD sample",
        )
    if precleanup_sample is not None:
        committed_prefix_times = tuple(
            value
            for value in (
                None if initial_sample is None else initial_sample.facts.monotonic_ns,
                None if ready is None else ready.ready_monotonic_ns,
                (
                    None
                    if observer_anchor is None
                    else observer_anchor.observer_started_monotonic_ns
                ),
                None if go is None else go.go_committed_monotonic_ns,
            )
            if value is not None
        )
        if (
            committed_prefix_times
            and precleanup_sample.facts.monotonic_ns <= max(committed_prefix_times)
        ):
            _fail("precleanup sample does not follow the latest committed prefix artifact")
    if cgroup_kill_receipt is not None:
        if precleanup_sample is None:
            _fail("cgroup.kill receipt lacks its committed precleanup sample")
        expected_pre = _identity_from_canonical_bytes(
            HOST_PRECLEANUP_CGROUP_SAMPLE_SCHEMA_VERSION,
            canonical_host_precleanup_cgroup_sample_v2_body_bytes(precleanup_sample),
            canonical_host_precleanup_cgroup_sample_v2_file_bytes(precleanup_sample),
        )
        if cgroup_kill_receipt.precleanup_sample != expected_pre:
            _fail("cgroup.kill receipt is cross-wired")
        if (
            cgroup_kill_receipt.retained_fd_set_sha256
            != precleanup_sample.facts.retained_fd_set_sha256
            or cgroup_kill_receipt.cgroup_identity_sha256
            != precleanup_sample.facts.cgroup_identity_sha256
            or cgroup_kill_receipt.kill_monotonic_ns
            <= precleanup_sample.facts.monotonic_ns
        ):
            _fail("cgroup.kill receipt changes retained-FD identity or chronology")
    if cgroup_empty_observation is not None:
        if precleanup_sample is None:
            _fail("cgroup-empty observation lacks its committed precleanup sample")
        expected_pre = _identity_from_canonical_bytes(
            HOST_PRECLEANUP_CGROUP_SAMPLE_SCHEMA_VERSION,
            canonical_host_precleanup_cgroup_sample_v2_body_bytes(precleanup_sample),
            canonical_host_precleanup_cgroup_sample_v2_file_bytes(precleanup_sample),
        )
        if cgroup_empty_observation.precleanup_sample != expected_pre:
            _fail("cgroup-empty observation is cross-wired from precleanup")
        if cgroup_empty_observation.cgroup_kill_receipt is not None:
            if cgroup_kill_receipt is None:
                _fail("cgroup-empty observation claims an unavailable kill receipt")
            expected_kill = _identity_from_canonical_bytes(
                HOST_CGROUP_KILL_RECEIPT_SCHEMA_VERSION,
                canonical_host_cgroup_kill_receipt_v2_body_bytes(cgroup_kill_receipt),
                canonical_host_cgroup_kill_receipt_v2_file_bytes(cgroup_kill_receipt),
            )
            if cgroup_empty_observation.cgroup_kill_receipt != expected_kill:
                _fail("cgroup-empty optional kill link is cross-wired")
        if (
            cgroup_empty_observation.retained_fd_set_sha256
            != precleanup_sample.facts.retained_fd_set_sha256
            or cgroup_empty_observation.cgroup_identity_sha256
            != precleanup_sample.facts.cgroup_identity_sha256
            or cgroup_empty_observation.observed_monotonic_ns
            <= precleanup_sample.facts.monotonic_ns
            or (
                cgroup_kill_receipt is not None
                and cgroup_empty_observation.observed_monotonic_ns
                <= cgroup_kill_receipt.kill_monotonic_ns
            )
        ):
            _fail("cgroup-empty observation changes retained-FD identity or chronology")
    if container_absence_observation is not None:
        _validate_container_absence_against_frontier(frontier, container_absence_observation)
        if request is not None and container_absence_observation.container_name != (
            request.container_name
        ):
            _fail("container-absence name differs from the request lookup identity")
        if ready is not None and (
            container_absence_observation.actual_runtime_container_identity_sha256
            != ready.container_identity_sha256
            or container_absence_observation.actual_container_id != ready.container_id
            or container_absence_observation.container_name != ready.container_name
        ):
            _fail("container-absence actual identity differs from committed READY")
        if container_absence_observation.cgroup_empty_observation is not None:
            if cgroup_empty_observation is None:
                _fail("container absence claims an unavailable empty observation")
            expected_empty = _identity_from_canonical_bytes(
                HOST_CGROUP_EMPTY_OBSERVATION_SCHEMA_VERSION,
                canonical_host_cgroup_empty_observation_v2_body_bytes(
                    cgroup_empty_observation
                ),
                canonical_host_cgroup_empty_observation_v2_file_bytes(
                    cgroup_empty_observation
                ),
            )
            if container_absence_observation.cgroup_empty_observation != expected_empty:
                _fail("container absence optional empty-observation link is cross-wired")
            if (
                container_absence_observation.removal_monotonic_ns
                <= cgroup_empty_observation.observed_monotonic_ns
            ):
                _fail("container absence must follow its linked cgroup-empty observation")
    if post_container_remove_sample is not None:
        if container_absence_observation is None:
            _fail("post-container sample lacks a committed container-absence observation")
        expected_container = _identity_from_canonical_bytes(
            HOST_CONTAINER_ABSENCE_OBSERVATION_SCHEMA_VERSION,
            canonical_host_container_absence_observation_v2_body_bytes(
                container_absence_observation
            ),
            canonical_host_container_absence_observation_v2_file_bytes(
                container_absence_observation
            ),
        )
        if post_container_remove_sample.container_absence_observation != expected_container:
            _fail("post-container retained-FD sample is cross-wired")
        expected_projection = (
            container_absence_observation.container_name,
            container_absence_observation.container_lookup_identity_sha256,
            container_absence_observation.actual_runtime_container_identity_sha256,
            container_absence_observation.actual_container_id,
        )
        actual_projection = (
            post_container_remove_sample.container_name,
            post_container_remove_sample.container_lookup_identity_sha256,
            post_container_remove_sample.actual_runtime_container_identity_sha256,
            post_container_remove_sample.actual_container_id,
        )
        if actual_projection != expected_projection:
            _fail("post-container sample changes lookup or optional runtime identity")
        if post_container_remove_sample.facts.monotonic_ns <= (
            container_absence_observation.removal_monotonic_ns
        ):
            _fail("post-container retained-FD sample must follow container reconciliation")
        if precleanup_sample is not None:
            _validate_cgroup_sample_progression_v2(
                precleanup_sample.facts,
                post_container_remove_sample.facts,
                "precleanup-to-post-remove retained-FD sample",
            )
        elif initial_sample is not None:
            _validate_cgroup_sample_progression_v2(
                initial_sample.facts,
                post_container_remove_sample.facts,
                "initial-to-post-remove retained-FD sample",
            )
    if cgroup_counter_fds_closed_receipt is not None:
        if post_container_remove_sample is None:
            _fail("retained-FD close lacks the post-container sample")
        expected_post = _identity_from_canonical_bytes(
            HOST_POST_CONTAINER_REMOVE_CGROUP_SAMPLE_SCHEMA_VERSION,
            canonical_host_post_container_remove_cgroup_sample_v2_body_bytes(
                post_container_remove_sample
            ),
            canonical_host_post_container_remove_cgroup_sample_v2_file_bytes(
                post_container_remove_sample
            ),
        )
        if cgroup_counter_fds_closed_receipt.post_container_remove_sample != expected_post:
            _fail("retained-FD close does not immediately follow the post-container sample")
        close_projection = (
            cgroup_counter_fds_closed_receipt.container_name,
            cgroup_counter_fds_closed_receipt.container_lookup_identity_sha256,
            cgroup_counter_fds_closed_receipt.actual_runtime_container_identity_sha256,
            cgroup_counter_fds_closed_receipt.actual_container_id,
        )
        post_projection = (
            post_container_remove_sample.container_name,
            post_container_remove_sample.container_lookup_identity_sha256,
            post_container_remove_sample.actual_runtime_container_identity_sha256,
            post_container_remove_sample.actual_container_id,
        )
        if close_projection != post_projection:
            _fail("retained-FD close changes lookup or optional runtime identity")
        if (
            cgroup_counter_fds_closed_receipt.retained_fd_set_sha256
            != post_container_remove_sample.facts.retained_fd_set_sha256
            or cgroup_counter_fds_closed_receipt.cgroup_identity_sha256
            != post_container_remove_sample.facts.cgroup_identity_sha256
            or cgroup_counter_fds_closed_receipt.close_monotonic_ns
            <= post_container_remove_sample.facts.monotonic_ns
        ):
            _fail("retained-FD close changes identity or precedes its final sample")
    if outer_cgroup_absence_observation is not None:
        if cgroup_counter_fds_closed_receipt is None:
            _fail("outer-cgroup absence lacks the retained-FD close receipt")
        expected_close = _identity_from_canonical_bytes(
            HOST_CGROUP_COUNTER_FDS_CLOSED_RECEIPT_SCHEMA_VERSION,
            canonical_host_cgroup_counter_fds_closed_receipt_v2_body_bytes(
                cgroup_counter_fds_closed_receipt
            ),
            canonical_host_cgroup_counter_fds_closed_receipt_v2_file_bytes(
                cgroup_counter_fds_closed_receipt
            ),
        )
        if outer_cgroup_absence_observation.cgroup_counter_fds_closed_receipt != expected_close:
            _fail("outer-cgroup absence does not follow retained-FD closure")
        if (
            outer_cgroup_absence_observation.cgroup_identity_sha256
            != cgroup_counter_fds_closed_receipt.cgroup_identity_sha256
            or outer_cgroup_absence_observation.removal_monotonic_ns
            <= cgroup_counter_fds_closed_receipt.close_monotonic_ns
            or (
                initial_sample is not None
                and outer_cgroup_absence_observation.cgroup_case_identity
                != initial_sample.cgroup_case_identity
            )
        ):
            _fail("outer-cgroup absence changes directory identity or chronology")
    if cgroup_proof is not None:
        if any(
            value is None
            for value in (
                request,
                intent,
                ready,
                observer_anchor,
                go,
                membership_event_log,
                initial_sample,
                precleanup_sample,
                cgroup_kill_receipt,
                cgroup_empty_observation,
                container_absence_observation,
                post_container_remove_sample,
                cgroup_counter_fds_closed_receipt,
                outer_cgroup_absence_observation,
            )
        ):
            _fail("committed cgroup proof lacks a complete retained-FD chain")
        validate_host_cgroup_boundary_proof_v2_chain(
            frontier,
            cast(HostInitialCgroupSampleV2, initial_sample),
            cast(HostPrecleanupCgroupSampleV2, precleanup_sample),
            cast(HostCgroupKillReceiptV2, cgroup_kill_receipt),
            cast(HostCgroupEmptyObservationV2, cgroup_empty_observation),
            cast(HostContainerAbsenceObservationV2, container_absence_observation),
            cast(HostPostContainerRemoveCgroupSampleV2, post_container_remove_sample),
            cast(HostCgroupCounterFdsClosedReceiptV2, cgroup_counter_fds_closed_receipt),
            cast(HostOuterCgroupAbsenceObservationV2, outer_cgroup_absence_observation),
            cgroup_proof,
            request=cast(HostQualificationCaseRequestV2, request),
            intent=cast(HostQualificationCaseIntentV2, intent),
            ready=cast(HostReadyV2, ready),
            observer_anchor=cast(HostObserverAnchorV2, observer_anchor),
            go=cast(HostGoCommitmentV2, go),
            membership_event_log=cast(
                HostCgroupMembershipEventLogV2,
                membership_event_log,
            ),
        )


def _algorithmic_receipt_schema(candidate_family: str) -> str:
    family = _require_text(candidate_family, "algorithmic receipt candidate family")
    if family == "local":
        return LOCAL_ALGORITHMIC_RESOURCE_RECEIPT_SCHEMA_VERSION
    if family == "external":
        return EXTERNAL_ALGORITHMIC_RESOURCE_RECEIPT_SCHEMA_VERSION
    if family == "adapter":
        return ADAPTER_ALGORITHMIC_RESOURCE_RECEIPT_SCHEMA_VERSION
    _fail("algorithmic receipt candidate family differs")


@dataclass(frozen=True, slots=True)
class HostTerminalMetadataV2:
    case_spine_sha256: str
    case_ordinal: int
    candidate_id: str
    candidate_family: Literal["local", "external", "adapter"]
    qualification_case_id: str
    record_kind: Literal["success", "terminal_failure"]
    operational_frontier: ArtifactIdentityV2
    cleanup_reconciliation: ArtifactIdentityV2
    driver_terminal: ArtifactIdentityV2 | None
    algorithmic_resource_receipt: ArtifactIdentityV2 | None
    publication_commitment_wrapper: ArtifactIdentityV2 | None
    publication_reload_validation: ArtifactIdentityV2 | None
    storage_write_seal: ArtifactIdentityV2 | None
    storage_boundary_receipt: ArtifactIdentityV2 | None
    returncode: int | None
    timed_out: bool
    error_message_sha256: str | None
    cleanup_proven: bool
    case_consumed: bool
    same_case_retry_permitted: bool

    def __post_init__(self) -> None:
        _require_case(
            self.case_ordinal,
            self.candidate_id,
            self.candidate_family,
            self.qualification_case_id,
        )
        _require_sha256(self.case_spine_sha256, "terminal metadata case spine")
        record_kind = _require_text(self.record_kind, "terminal metadata record kind")
        if record_kind not in {"success", "terminal_failure"}:
            _fail("terminal metadata record kind differs")
        _require_artifact_schema(
            self.operational_frontier,
            HOST_OPERATIONAL_FRONTIER_SCHEMA_VERSION,
            "terminal operational frontier",
        )
        _require_artifact_schema(
            self.cleanup_reconciliation,
            HOST_CLEANUP_RECONCILIATION_SCHEMA_VERSION,
            "terminal cleanup reconciliation",
        )
        optional_artifacts = (
            (self.driver_terminal, IN_CONTAINER_DRIVER_TERMINAL_SCHEMA_VERSION, "driver terminal"),
            (
                self.algorithmic_resource_receipt,
                _algorithmic_receipt_schema(self.candidate_family),
                "algorithmic resource receipt",
            ),
            (
                self.publication_commitment_wrapper,
                PUBLICATION_COMMITMENT_WRAPPER_SCHEMA_VERSION,
                "publication commitment wrapper",
            ),
            (
                self.publication_reload_validation,
                PUBLICATION_RELOAD_VALIDATION_SCHEMA_VERSION,
                "publication reload validation",
            ),
            (self.storage_write_seal, STORAGE_WRITE_SEAL_SCHEMA_VERSION, "storage write seal"),
            (
                self.storage_boundary_receipt,
                STORAGE_BOUNDARY_RECEIPT_SCHEMA_VERSION,
                "storage boundary receipt",
            ),
        )
        for artifact, schema, label in optional_artifacts:
            if artifact is not None:
                _require_artifact_schema(artifact, schema, f"terminal {label}")
        if self.returncode is not None:
            _require_int(
                self.returncode,
                "terminal returncode",
                minimum=-_MAX_INTEGER,
            )
        timed_out = _require_bool(self.timed_out, "terminal timeout")
        cleanup_proven = _require_bool(self.cleanup_proven, "terminal cleanup proven")
        if record_kind == "success":
            if any(artifact is None for artifact, _, _ in optional_artifacts):
                _fail("success terminal metadata lacks a committed output artifact")
            if self.returncode != 0 or timed_out or self.error_message_sha256 is not None:
                _fail("success terminal metadata return state differs")
            if not cleanup_proven:
                _fail("success terminal metadata requires proven cleanup")
        else:
            _require_sha256(self.error_message_sha256, "terminal failure error message")
        if _require_bool(self.case_consumed, "terminal case consumed") is not True:
            _fail("terminal metadata must consume the case")
        if _require_bool(self.same_case_retry_permitted, "terminal retry") is not False:
            _fail("terminal metadata cannot permit same-case retry")

    def to_body_dict(self) -> dict[str, Any]:
        return {
            "schema_version": HOST_TERMINAL_METADATA_SCHEMA_VERSION,
            "case_spine_sha256": self.case_spine_sha256,
            "case_ordinal": self.case_ordinal,
            "candidate_id": self.candidate_id,
            "candidate_family": self.candidate_family,
            "qualification_case_id": self.qualification_case_id,
            "record_kind": self.record_kind,
            "operational_frontier": self.operational_frontier.to_dict(),
            "cleanup_reconciliation": self.cleanup_reconciliation.to_dict(),
            "driver_terminal": (
                None if self.driver_terminal is None else self.driver_terminal.to_dict()
            ),
            "algorithmic_resource_receipt": (
                None
                if self.algorithmic_resource_receipt is None
                else self.algorithmic_resource_receipt.to_dict()
            ),
            "publication_commitment_wrapper": (
                None
                if self.publication_commitment_wrapper is None
                else self.publication_commitment_wrapper.to_dict()
            ),
            "publication_reload_validation": (
                None
                if self.publication_reload_validation is None
                else self.publication_reload_validation.to_dict()
            ),
            "storage_write_seal": (
                None if self.storage_write_seal is None else self.storage_write_seal.to_dict()
            ),
            "storage_boundary_receipt": (
                None
                if self.storage_boundary_receipt is None
                else self.storage_boundary_receipt.to_dict()
            ),
            "returncode": self.returncode,
            "timed_out": self.timed_out,
            "error_message_sha256": self.error_message_sha256,
            "cleanup_proven": self.cleanup_proven,
            "case_consumed": self.case_consumed,
            "same_case_retry_permitted": self.same_case_retry_permitted,
            "authority": _authority(),
            "claims": _claims(),
        }


TERMINAL_METADATA_BODY_SHA256_FIELD: Final = "terminal_metadata_body_sha256"


def canonical_host_terminal_metadata_v2_body_bytes(
    terminal: HostTerminalMetadataV2,
) -> bytes:
    return _canonical_artifact_body_bytes(terminal, HostTerminalMetadataV2)


def canonical_host_terminal_metadata_v2_file_bytes(
    terminal: HostTerminalMetadataV2,
) -> bytes:
    return _canonical_artifact_file_bytes(
        terminal,
        HostTerminalMetadataV2,
        TERMINAL_METADATA_BODY_SHA256_FIELD,
    )


def parse_host_terminal_metadata_v2(
    raw: bytes,
    *,
    expected_file_sha256: str,
) -> HostTerminalMetadataV2:
    body = _parse_artifact_file(
        raw,
        expected_file_sha256=expected_file_sha256,
        body_field=TERMINAL_METADATA_BODY_SHA256_FIELD,
        body_keys=frozenset(HostTerminalMetadataV2.__dataclass_fields__)
        | {"schema_version", "authority", "claims"},
        label="host terminal metadata",
    )
    item = dict(body)
    if (
        item.pop("schema_version") != HOST_TERMINAL_METADATA_SCHEMA_VERSION
        or item.pop("authority") != _authority()
        or item.pop("claims") != _claims()
    ):
        _fail("host terminal metadata envelope differs")
    for field in ("operational_frontier", "cleanup_reconciliation"):
        item[field] = _artifact_identity(item[field], f"terminal {field}")
    for field in (
        "driver_terminal",
        "algorithmic_resource_receipt",
        "publication_commitment_wrapper",
        "publication_reload_validation",
        "storage_write_seal",
        "storage_boundary_receipt",
    ):
        item[field] = _optional_artifact_identity(item[field], f"terminal {field}")
    result = HostTerminalMetadataV2(**item)
    if raw != canonical_host_terminal_metadata_v2_file_bytes(result):
        _fail("host terminal metadata canonical replay differs")
    return result


def validate_host_terminal_metadata_v2_chain(
    frontier: HostOperationalFrontierV2,
    reconciliation: HostCleanupReconciliationV2,
    terminal: HostTerminalMetadataV2,
    *,
    request: HostQualificationCaseRequestV2,
) -> None:
    if type(frontier) is not HostOperationalFrontierV2:
        raise TypeError("frontier must use the exact operational-frontier-v2 type")
    if type(reconciliation) is not HostCleanupReconciliationV2:
        raise TypeError("reconciliation must use the exact cleanup-reconciliation-v2 type")
    if type(terminal) is not HostTerminalMetadataV2:
        raise TypeError("terminal must use the exact terminal-metadata-v2 type")
    if type(request) is not HostQualificationCaseRequestV2:
        raise TypeError("request must use the exact request-v2 type")
    cleanup_id = _identity_from_canonical_bytes(
        HOST_CLEANUP_RECONCILIATION_SCHEMA_VERSION,
        canonical_host_cleanup_reconciliation_v2_body_bytes(reconciliation),
        canonical_host_cleanup_reconciliation_v2_file_bytes(reconciliation),
    )
    if (
        terminal.case_spine_sha256 != frontier.case_spine_sha256
        or terminal.case_spine_sha256 != request.case_spine_sha256
        or reconciliation.case_spine_sha256 != frontier.case_spine_sha256
        or terminal.operational_frontier != _frontier_identity(frontier)
        or terminal.cleanup_reconciliation != cleanup_id
        or terminal.cleanup_proven is not reconciliation.cleanup_proven
    ):
        _fail("terminal metadata does not bind its exact frontier and cleanup record")
    case_fields = ("case_ordinal", "candidate_id", "candidate_family", "qualification_case_id")
    if any(getattr(terminal, field) != getattr(request, field) for field in case_fields):
        _fail("terminal metadata crosses its request case identity")
    expected_kind = (
        "success" if frontier.succeeded and reconciliation.cleanup_proven else "terminal_failure"
    )
    if terminal.record_kind != expected_kind:
        _fail("terminal record kind differs from operational and recovery outcomes")
    output_boundaries = (
        (terminal.driver_terminal, "workload_exited"),
        (
            terminal.algorithmic_resource_receipt,
            "algorithmic_resource_receipt_committed",
        ),
        (
            terminal.publication_commitment_wrapper,
            "publication_commitment_wrapper_committed",
        ),
        (terminal.publication_reload_validation, "publication_reload_validated"),
        (terminal.storage_write_seal, "storage_write_seal_committed"),
        (terminal.storage_boundary_receipt, "storage_boundary_receipt_committed"),
    )
    for artifact, phase in output_boundaries:
        if (phase in frontier.completed_phases) is not (artifact is not None):
            _fail(f"terminal metadata output presence differs at {phase}")


def publication_reconciliation_key_sha256_v2(
    case_spine_sha256: object,
    expected_publication_address_sha256: object,
    producer: ProducerIdentityV2,
) -> str:
    spine = _require_sha256(case_spine_sha256, "publication reconciliation case spine")
    address = _require_sha256(
        expected_publication_address_sha256,
        "publication reconciliation address",
    )
    if type(producer) is not ProducerIdentityV2:
        raise TypeError("publication producer must use the exact v2 type")
    return _body_sha256(
        {
            "case_spine_sha256": spine,
            "expected_publication_address_sha256": address,
            "native_atomic_producer": producer.to_dict(),
        }
    )


@dataclass(frozen=True, slots=True)
class HostNativePublicationProjectionV2:
    case_spine_sha256: str
    native_publication_state: BoundaryStateV2
    native_atomic_producer: ProducerIdentityV2
    native_publication_receipt: ArtifactIdentityV2 | None
    expected_publication_address_sha256: str | None
    publication_reconciliation_key_sha256: str | None
    publication_reconciliation_reference: ArtifactIdentityV2 | None
    failure_publication_projection: ArtifactIdentityV2 | None
    native_publication_commit_count_state: BoundaryCountStateV2
    native_publication_commit_count: int | None

    def __post_init__(self) -> None:
        _require_sha256(self.case_spine_sha256, "native publication case spine")
        state = _require_text(self.native_publication_state, "native publication state")
        if state not in {"not_started", "commit_uncertain", "committed"}:
            _fail("native publication state differs")
        if type(self.native_atomic_producer) is not ProducerIdentityV2:
            _fail("native atomic producer type differs")
        if self.native_publication_receipt is not None and type(
            self.native_publication_receipt
        ) is not ArtifactIdentityV2:
            _fail("native publication receipt type differs")
        address = _require_optional_sha256(
            self.expected_publication_address_sha256,
            "native publication expected address",
        )
        key = _require_optional_sha256(
            self.publication_reconciliation_key_sha256,
            "native publication reconciliation key",
        )
        reference = self.publication_reconciliation_reference
        if reference is not None:
            _require_artifact_schema(
                reference,
                PUBLICATION_RECONCILIATION_REFERENCE_SCHEMA_VERSION,
                "native publication reconciliation reference",
            )
        if state == "not_started":
            if any(
                value is not None
                for value in (self.native_publication_receipt, address, key, reference)
            ):
                _fail("unstarted native publication cannot carry reconciliation identities")
            expected_count: tuple[BoundaryCountStateV2, int | None] = ("exact", 0)
        else:
            if address is None or key is None or reference is None:
                _fail("attempted native publication lacks its reconciliation identities")
            if key != publication_reconciliation_key_sha256_v2(
                self.case_spine_sha256,
                address,
                self.native_atomic_producer,
            ):
                _fail("native publication reconciliation key differs")
            expected_count = ("uncertain", None) if state == "commit_uncertain" else ("exact", 1)
            if state == "commit_uncertain" and self.native_publication_receipt is not None:
                _fail("commit-uncertain native publication cannot claim a committed receipt")
        if self.native_publication_commit_count is not None:
            _require_int(
                self.native_publication_commit_count,
                "native publication commit count",
                maximum=1,
            )
        if (
            self.native_publication_commit_count_state,
            self.native_publication_commit_count,
        ) != expected_count:
            _fail("native publication count is not separate and derived from its state")
        if self.failure_publication_projection is not None:
            _require_artifact_schema(
                self.failure_publication_projection,
                FAILURE_PUBLICATION_PROJECTION_SCHEMA_VERSION,
                "failure publication projection",
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_spine_sha256": self.case_spine_sha256,
            "native_publication_state": self.native_publication_state,
            "native_atomic_producer": self.native_atomic_producer.to_dict(),
            "native_publication_receipt": (
                None
                if self.native_publication_receipt is None
                else self.native_publication_receipt.to_dict()
            ),
            "expected_publication_address_sha256": (
                self.expected_publication_address_sha256
            ),
            "publication_reconciliation_key_sha256": (
                self.publication_reconciliation_key_sha256
            ),
            "publication_reconciliation_reference": (
                None
                if self.publication_reconciliation_reference is None
                else self.publication_reconciliation_reference.to_dict()
            ),
            "failure_publication_projection": (
                None
                if self.failure_publication_projection is None
                else self.failure_publication_projection.to_dict()
            ),
            "native_publication_commit_count_state": (
                self.native_publication_commit_count_state
            ),
            "native_publication_commit_count": self.native_publication_commit_count,
        }


def _native_publication_projection_from_dict(
    value: object,
) -> HostNativePublicationProjectionV2:
    item = _require_exact_keys(
        value,
        frozenset(HostNativePublicationProjectionV2.__dataclass_fields__),
        "native publication projection",
    )
    item["native_atomic_producer"] = _producer_identity(
        item["native_atomic_producer"],
        "native publication producer",
    )
    for field in (
        "native_publication_receipt",
        "publication_reconciliation_reference",
        "failure_publication_projection",
    ):
        item[field] = _optional_artifact_identity(item[field], f"native publication {field}")
    return HostNativePublicationProjectionV2(**item)


def _validate_native_publication_projection_v2_chain(
    request: HostQualificationCaseRequestV2,
    frontier: HostOperationalFrontierV2,
    projection: HostNativePublicationProjectionV2,
    *,
    record_kind: Literal["success", "terminal_failure"],
) -> None:
    expected_state = _derived_boundary_state(
        frontier.completed_phases,
        frontier.failure_phase,
        frontier.failure_effect_state,
        "native_publication_committed",
    )
    if (
        projection.case_spine_sha256 != request.case_spine_sha256
        or projection.case_spine_sha256 != frontier.case_spine_sha256
        or projection.native_publication_state != expected_state
        or projection.native_atomic_producer != request.native_atomic_producer
    ):
        _fail("native publication projection crosses request or frontier state")
    family = request.candidate_family
    if projection.native_atomic_producer.descriptor_schema_version != _expected_atomic_schema(
        family
    ):
        _fail("native publication producer schema differs from candidate family")
    if expected_state == "committed":
        if family == "local":
            if projection.native_publication_receipt is not None:
                _fail("local native publication cannot carry a receipt")
        else:
            schema = (
                EXTERNAL_ATOMIC_PUBLICATION_RECEIPT_SCHEMA_VERSION
                if family == "external"
                else STRICT_ADAPTER_ATOMIC_PUBLICATION_RECEIPT_SCHEMA_VERSION
            )
            if projection.native_publication_receipt is None:
                _fail("committed nonlocal native publication lacks its receipt")
            _require_artifact_schema(
                projection.native_publication_receipt,
                schema,
                "native publication receipt",
            )
    wrapper_committed = "publication_commitment_wrapper_committed" in (
        frontier.completed_phases
    )
    failure_projection_required = record_kind == "terminal_failure" and wrapper_committed
    if failure_projection_required is not (projection.failure_publication_projection is not None):
        _fail("failure publication projection presence differs from wrapper commitment")


@dataclass(frozen=True, slots=True)
class HostLifecycleRollupV2:
    case_spine_sha256: str
    case_ordinal: int
    candidate_id: str
    candidate_family: Literal["local", "external", "adapter"]
    qualification_case_id: str
    record_kind: Literal["success", "terminal_failure"]
    operational_frontier: ArtifactIdentityV2
    cleanup_reconciliation: ArtifactIdentityV2
    terminal_metadata: ArtifactIdentityV2
    native_publication: HostNativePublicationProjectionV2
    operational_success: bool
    cleanup_proven: bool
    terminal_metadata_validated: bool
    container_create_state: BoundaryStateV2
    container_start_state: BoundaryStateV2
    workload_start_state: BoundaryStateV2
    workload_exit_state: BoundaryStateV2
    attempt_count_state: BoundaryCountStateV2
    attempt_count: int | None
    operational_failure_count: int
    recovery_failure_count: int
    recovery_uncertainty_count: int
    terminal_failure_count: int
    structural_success_shape_only: bool
    production_execution_success: bool
    production_acceptance_eligible: bool
    evidence_eligible: bool
    case_consumed: bool
    same_case_retry_permitted: bool

    def __post_init__(self) -> None:
        _require_case(
            self.case_ordinal,
            self.candidate_id,
            self.candidate_family,
            self.qualification_case_id,
        )
        _require_sha256(self.case_spine_sha256, "lifecycle case spine")
        if _require_text(self.record_kind, "lifecycle record kind") not in {
            "success",
            "terminal_failure",
        }:
            _fail("lifecycle record kind differs")
        for artifact, schema, label in (
            (
                self.operational_frontier,
                HOST_OPERATIONAL_FRONTIER_SCHEMA_VERSION,
                "operational frontier",
            ),
            (
                self.cleanup_reconciliation,
                HOST_CLEANUP_RECONCILIATION_SCHEMA_VERSION,
                "cleanup reconciliation",
            ),
            (self.terminal_metadata, HOST_TERMINAL_METADATA_SCHEMA_VERSION, "terminal metadata"),
        ):
            _require_artifact_schema(artifact, schema, f"lifecycle {label}")
        if type(self.native_publication) is not HostNativePublicationProjectionV2:
            _fail("lifecycle native publication projection type differs")
        _require_bool(self.operational_success, "lifecycle operational success")
        _require_bool(self.cleanup_proven, "lifecycle cleanup proven")
        if (
            _require_bool(
                self.terminal_metadata_validated,
                "lifecycle terminal metadata validated",
            )
            is not True
        ):
            _fail("lifecycle requires validated common terminal metadata")
        for field in (
            "container_create_state",
            "container_start_state",
            "workload_start_state",
            "workload_exit_state",
            "attempt_count_state",
        ):
            _require_text(getattr(self, field), f"lifecycle {field}")
        if self.attempt_count is not None:
            _require_int(self.attempt_count, "lifecycle attempt count", maximum=1)
        _require_int(
            self.operational_failure_count,
            "lifecycle operational failure count",
            maximum=1,
        )
        _require_int(
            self.recovery_failure_count,
            "lifecycle recovery failure count",
            maximum=len(RECOVERY_NODE_NAMES),
        )
        _require_int(
            self.recovery_uncertainty_count,
            "lifecycle recovery uncertainty count",
            maximum=len(RECOVERY_NODE_NAMES),
        )
        _require_int(
            self.terminal_failure_count,
            "lifecycle terminal failure count",
            maximum=1,
        )
        expected_terminal_failures = 0 if self.record_kind == "success" else 1
        if self.terminal_failure_count != expected_terminal_failures:
            _fail("lifecycle terminal failure count differs")
        expected_structural_success = self.record_kind == "success"
        if (
            _require_bool(
                self.structural_success_shape_only,
                "lifecycle structural success shape",
            )
            is not expected_structural_success
            or _require_bool(
                self.production_execution_success,
                "lifecycle production execution success",
            )
            is not False
            or _require_bool(
                self.production_acceptance_eligible,
                "lifecycle production acceptance eligibility",
            )
            is not False
            or _require_bool(self.evidence_eligible, "lifecycle evidence eligibility")
            is not False
        ):
            _fail("source-only lifecycle cannot claim production or evidence success")
        if _require_bool(self.case_consumed, "lifecycle case consumed") is not True:
            _fail("lifecycle must consume the case")
        if _require_bool(self.same_case_retry_permitted, "lifecycle retry") is not False:
            _fail("lifecycle cannot permit same-case retry")

    def to_body_dict(self) -> dict[str, Any]:
        return {
            "schema_version": HOST_LIFECYCLE_SCHEMA_VERSION,
            "status": "acyclic_terminal_lifecycle_rollup_non_authorizing",
            **{
                field: (
                    getattr(self, field).to_dict()
                    if type(getattr(self, field))
                    in {ArtifactIdentityV2, HostNativePublicationProjectionV2}
                    else getattr(self, field)
                )
                for field in self.__dataclass_fields__
            },
            "claims": _claims(),
        }


LIFECYCLE_BODY_SHA256_FIELD: Final = "lifecycle_body_sha256"


def canonical_host_lifecycle_v2_body_bytes(lifecycle: HostLifecycleRollupV2) -> bytes:
    return _canonical_artifact_body_bytes(lifecycle, HostLifecycleRollupV2)


def canonical_host_lifecycle_v2_file_bytes(lifecycle: HostLifecycleRollupV2) -> bytes:
    return _canonical_artifact_file_bytes(
        lifecycle,
        HostLifecycleRollupV2,
        LIFECYCLE_BODY_SHA256_FIELD,
    )


def parse_host_lifecycle_v2(
    raw: bytes,
    *,
    expected_file_sha256: str,
) -> HostLifecycleRollupV2:
    body = _parse_artifact_file(
        raw,
        expected_file_sha256=expected_file_sha256,
        body_field=LIFECYCLE_BODY_SHA256_FIELD,
        body_keys=frozenset(HostLifecycleRollupV2.__dataclass_fields__)
        | {"schema_version", "status", "claims"},
        label="host lifecycle",
    )
    item = dict(body)
    if (
        item.pop("schema_version") != HOST_LIFECYCLE_SCHEMA_VERSION
        or item.pop("status") != "acyclic_terminal_lifecycle_rollup_non_authorizing"
        or item.pop("claims") != _claims()
    ):
        _fail("host lifecycle envelope differs")
    for field in ("operational_frontier", "cleanup_reconciliation", "terminal_metadata"):
        item[field] = _artifact_identity(item[field], f"lifecycle {field}")
    item["native_publication"] = _native_publication_projection_from_dict(
        item["native_publication"]
    )
    result = HostLifecycleRollupV2(**item)
    if raw != canonical_host_lifecycle_v2_file_bytes(result):
        _fail("host lifecycle canonical replay differs")
    return result


def validate_host_lifecycle_v2_chain(
    frontier: HostOperationalFrontierV2,
    reconciliation: HostCleanupReconciliationV2,
    terminal: HostTerminalMetadataV2,
    lifecycle: HostLifecycleRollupV2,
    *,
    request: HostQualificationCaseRequestV2,
) -> None:
    if type(request) is not HostQualificationCaseRequestV2:
        raise TypeError("request must use the exact request-v2 type")
    validate_host_terminal_metadata_v2_chain(
        frontier,
        reconciliation,
        terminal,
        request=request,
    )
    if type(lifecycle) is not HostLifecycleRollupV2:
        raise TypeError("lifecycle must use the exact lifecycle-v2 type")
    cleanup_id = _identity_from_canonical_bytes(
        HOST_CLEANUP_RECONCILIATION_SCHEMA_VERSION,
        canonical_host_cleanup_reconciliation_v2_body_bytes(reconciliation),
        canonical_host_cleanup_reconciliation_v2_file_bytes(reconciliation),
    )
    terminal_id = _identity_from_canonical_bytes(
        HOST_TERMINAL_METADATA_SCHEMA_VERSION,
        canonical_host_terminal_metadata_v2_body_bytes(terminal),
        canonical_host_terminal_metadata_v2_file_bytes(terminal),
    )
    expected = {
        "case_spine_sha256": frontier.case_spine_sha256,
        "record_kind": terminal.record_kind,
        "operational_frontier": _frontier_identity(frontier),
        "cleanup_reconciliation": cleanup_id,
        "terminal_metadata": terminal_id,
        "operational_success": frontier.succeeded,
        "cleanup_proven": reconciliation.cleanup_proven,
        "container_create_state": frontier.container_create_state,
        "container_start_state": frontier.container_start_state,
        "workload_start_state": frontier.workload_start_state,
        "workload_exit_state": frontier.workload_exit_state,
        "attempt_count_state": frontier.attempt_count_state,
        "attempt_count": frontier.attempt_count,
        "operational_failure_count": frontier.failure_count,
        "recovery_failure_count": sum(
            node.state == "failed_before_commit" for node in reconciliation.recovery_nodes
        ),
        "recovery_uncertainty_count": sum(
            node.state == "commit_uncertain" for node in reconciliation.recovery_nodes
        ),
        "terminal_failure_count": 0 if terminal.record_kind == "success" else 1,
        "structural_success_shape_only": terminal.record_kind == "success",
        "production_execution_success": False,
        "production_acceptance_eligible": False,
        "evidence_eligible": False,
    }
    if any(getattr(lifecycle, field) != value for field, value in expected.items()):
        _fail("lifecycle rollup differs from frontier, cleanup, or terminal metadata")
    _validate_native_publication_projection_v2_chain(
        request,
        frontier,
        lifecycle.native_publication,
        record_kind=terminal.record_kind,
    )
    case_fields = ("case_ordinal", "candidate_id", "candidate_family", "qualification_case_id")
    if any(
        getattr(source, field) != getattr(request, field)
        for source in (terminal, lifecycle)
        for field in case_fields
    ):
        _fail("terminal or lifecycle case projection differs from its request")


@dataclass(frozen=True, slots=True)
class HostSuccessReceiptV2:
    case_spine_sha256: str
    case_ordinal: int
    candidate_id: str
    candidate_family: Literal["local", "external", "adapter"]
    qualification_case_id: str
    request: ArtifactIdentityV2
    intent: ArtifactIdentityV2
    ready: ArtifactIdentityV2
    observer_anchor: ArtifactIdentityV2
    go_commitment: ArtifactIdentityV2
    operational_frontier: ArtifactIdentityV2
    cleanup_reconciliation: ArtifactIdentityV2
    cgroup_proof: ArtifactIdentityV2
    terminal_metadata: ArtifactIdentityV2
    lifecycle: ArtifactIdentityV2
    container_create_count: int
    container_start_count: int
    go_commit_count: int
    workload_start_count: int
    workload_exit_count: int
    attempt_count: int
    failure_count: int
    returncode: int
    timed_out: bool
    cleanup_proven: bool
    structural_success_shape_only: bool
    production_execution_success: bool
    production_acceptance_eligible: bool
    evidence_eligible: bool
    case_consumed: bool
    same_case_retry_permitted: bool

    def __post_init__(self) -> None:
        _require_case(
            self.case_ordinal,
            self.candidate_id,
            self.candidate_family,
            self.qualification_case_id,
        )
        _require_sha256(self.case_spine_sha256, "success receipt case spine")
        for artifact, schema, label in (
            (self.request, HOST_CASE_REQUEST_SCHEMA_VERSION, "request"),
            (self.intent, HOST_CASE_INTENT_SCHEMA_VERSION, "intent"),
            (self.ready, HOST_READY_SCHEMA_VERSION, "READY"),
            (self.observer_anchor, HOST_OBSERVER_ANCHOR_SCHEMA_VERSION, "observer anchor"),
            (self.go_commitment, HOST_GO_SCHEMA_VERSION, "GO commitment"),
            (
                self.operational_frontier,
                HOST_OPERATIONAL_FRONTIER_SCHEMA_VERSION,
                "operational frontier",
            ),
            (
                self.cleanup_reconciliation,
                HOST_CLEANUP_RECONCILIATION_SCHEMA_VERSION,
                "cleanup reconciliation",
            ),
            (self.cgroup_proof, HOST_CGROUP_PROOF_SCHEMA_VERSION, "cgroup proof"),
            (self.terminal_metadata, HOST_TERMINAL_METADATA_SCHEMA_VERSION, "terminal metadata"),
            (self.lifecycle, HOST_LIFECYCLE_SCHEMA_VERSION, "lifecycle"),
        ):
            _require_artifact_schema(artifact, schema, f"success receipt {label}")
        for field in (
            "container_create_count",
            "container_start_count",
            "go_commit_count",
            "workload_start_count",
            "workload_exit_count",
            "attempt_count",
        ):
            if _require_int(getattr(self, field), f"success receipt {field}") != 1:
                _fail("success receipt execution counts must all be exact one")
        if (
            _require_int(self.failure_count, "success failure count", maximum=0) != 0
            or _require_int(
                self.returncode,
                "success returncode",
                maximum=0,
            )
            != 0
        ):
            _fail("success receipt failure count or returncode differs")
        if _require_bool(self.timed_out, "success timeout") is not False:
            _fail("success receipt cannot be timed out")
        if _require_bool(self.cleanup_proven, "success cleanup") is not True:
            _fail("success receipt requires proven cleanup")
        if (
            _require_bool(
                self.structural_success_shape_only,
                "success receipt structural success shape",
            )
            is not True
            or _require_bool(
                self.production_execution_success,
                "success receipt production execution success",
            )
            is not False
            or _require_bool(
                self.production_acceptance_eligible,
                "success receipt production acceptance eligibility",
            )
            is not False
            or _require_bool(self.evidence_eligible, "success receipt evidence eligibility")
            is not False
        ):
            _fail("source-only success receipt cannot claim production or evidence success")
        if _require_bool(self.case_consumed, "success case consumed") is not True:
            _fail("success receipt must consume the case")
        if _require_bool(self.same_case_retry_permitted, "success retry") is not False:
            _fail("success receipt cannot permit same-case retry")

    def to_body_dict(self) -> dict[str, Any]:
        return {
            "schema_version": HOST_SUCCESS_RECEIPT_SCHEMA_VERSION,
            "record_kind": "success",
            **{
                field: (
                    getattr(self, field).to_dict()
                    if type(getattr(self, field)) is ArtifactIdentityV2
                    else getattr(self, field)
                )
                for field in self.__dataclass_fields__
            },
            "authority": _authority(),
            "claims": _claims(),
        }


SUCCESS_RECEIPT_BODY_SHA256_FIELD: Final = "success_receipt_body_sha256"


def canonical_host_success_receipt_v2_body_bytes(receipt: HostSuccessReceiptV2) -> bytes:
    return _canonical_artifact_body_bytes(receipt, HostSuccessReceiptV2)


def canonical_host_success_receipt_v2_file_bytes(receipt: HostSuccessReceiptV2) -> bytes:
    return _canonical_artifact_file_bytes(
        receipt,
        HostSuccessReceiptV2,
        SUCCESS_RECEIPT_BODY_SHA256_FIELD,
    )


def parse_host_success_receipt_v2(
    raw: bytes,
    *,
    expected_file_sha256: str,
) -> HostSuccessReceiptV2:
    body = _parse_artifact_file(
        raw,
        expected_file_sha256=expected_file_sha256,
        body_field=SUCCESS_RECEIPT_BODY_SHA256_FIELD,
        body_keys=frozenset(HostSuccessReceiptV2.__dataclass_fields__)
        | {"schema_version", "record_kind", "authority", "claims"},
        label="host success receipt",
    )
    item = dict(body)
    if (
        item.pop("schema_version") != HOST_SUCCESS_RECEIPT_SCHEMA_VERSION
        or item.pop("record_kind") != "success"
        or item.pop("authority") != _authority()
        or item.pop("claims") != _claims()
    ):
        _fail("host success receipt envelope differs")
    for field in (
        "request",
        "intent",
        "ready",
        "observer_anchor",
        "go_commitment",
        "operational_frontier",
        "cleanup_reconciliation",
        "cgroup_proof",
        "terminal_metadata",
        "lifecycle",
    ):
        item[field] = _artifact_identity(item[field], f"success receipt {field}")
    result = HostSuccessReceiptV2(**item)
    if raw != canonical_host_success_receipt_v2_file_bytes(result):
        _fail("host success receipt canonical replay differs")
    return result


@dataclass(frozen=True, slots=True)
class HostFailureReceiptV2:
    case_spine_sha256: str
    case_ordinal: int
    candidate_id: str
    candidate_family: Literal["local", "external", "adapter"]
    qualification_case_id: str
    request: ArtifactIdentityV2
    intent: ArtifactIdentityV2 | None
    initial_sample: ArtifactIdentityV2 | None
    ready: ArtifactIdentityV2 | None
    observer_anchor: ArtifactIdentityV2 | None
    go_commitment: ArtifactIdentityV2 | None
    operational_frontier: ArtifactIdentityV2
    cleanup_reconciliation: ArtifactIdentityV2
    terminal_metadata: ArtifactIdentityV2
    lifecycle: ArtifactIdentityV2
    native_publication: HostNativePublicationProjectionV2
    failure_receipt_state: Literal["committed"]
    classification: str
    exception_type: str
    error_message_sha256: str
    operational_failure_phase: str | None
    operational_failure_effect_state: Literal[
        "failed_before_commit", "commit_uncertain"
    ] | None
    unresolved_recovery_nodes: tuple[str, ...]
    container_create_count_state: BoundaryCountStateV2
    container_create_count: int | None
    container_start_count_state: BoundaryCountStateV2
    container_start_count: int | None
    workload_start_count_state: BoundaryCountStateV2
    workload_start_count: int | None
    workload_exit_count_state: BoundaryCountStateV2
    workload_exit_count: int | None
    attempt_count_state: BoundaryCountStateV2
    attempt_count: int | None
    operational_failure_count: int
    recovery_failure_count: int
    recovery_uncertainty_count: int
    terminal_failure_count: int
    cleanup_proven: bool
    ticket_quarantined: bool
    reconciliation_only: bool
    case_consumed: bool
    same_case_retry_permitted: bool
    clean_rejection_recorded: bool

    def __post_init__(self) -> None:
        _require_case(
            self.case_ordinal,
            self.candidate_id,
            self.candidate_family,
            self.qualification_case_id,
        )
        _require_sha256(self.case_spine_sha256, "failure receipt case spine")
        for artifact, schema, label in (
            (self.request, HOST_CASE_REQUEST_SCHEMA_VERSION, "request"),
            (
                self.operational_frontier,
                HOST_OPERATIONAL_FRONTIER_SCHEMA_VERSION,
                "operational frontier",
            ),
            (
                self.cleanup_reconciliation,
                HOST_CLEANUP_RECONCILIATION_SCHEMA_VERSION,
                "cleanup reconciliation",
            ),
            (self.terminal_metadata, HOST_TERMINAL_METADATA_SCHEMA_VERSION, "terminal metadata"),
            (self.lifecycle, HOST_LIFECYCLE_SCHEMA_VERSION, "lifecycle"),
        ):
            _require_artifact_schema(artifact, schema, f"failure receipt {label}")
        if self.intent is not None:
            _require_artifact_schema(
                self.intent,
                HOST_CASE_INTENT_SCHEMA_VERSION,
                "failure receipt intent",
            )
        for optional_artifact, schema, label in (
            (
                self.initial_sample,
                HOST_INITIAL_CGROUP_SAMPLE_SCHEMA_VERSION,
                "initial sample",
            ),
            (self.ready, HOST_READY_SCHEMA_VERSION, "READY"),
            (
                self.observer_anchor,
                HOST_OBSERVER_ANCHOR_SCHEMA_VERSION,
                "observer anchor",
            ),
            (self.go_commitment, HOST_GO_SCHEMA_VERSION, "GO commitment"),
        ):
            if optional_artifact is not None:
                _require_artifact_schema(
                    optional_artifact,
                    schema,
                    f"failure receipt {label}",
                )
        if type(self.native_publication) is not HostNativePublicationProjectionV2:
            _fail("failure receipt native publication projection type differs")
        if _require_text(self.failure_receipt_state, "failure receipt state") != "committed":
            _fail("a canonical failure receipt must record its known commit")
        _require_identifier(self.classification, "failure receipt classification")
        _require_identifier(self.exception_type, "failure receipt exception type")
        _require_sha256(self.error_message_sha256, "failure receipt error message")
        if self.operational_failure_phase is None:
            if self.operational_failure_effect_state is not None:
                _fail("absent operational failure cannot carry an effect")
        else:
            if _require_text(
                self.operational_failure_phase,
                "failure receipt operational phase",
            ) not in OPERATIONAL_PHASES:
                _fail("failure receipt operational phase differs")
            if _require_text(
                self.operational_failure_effect_state,
                "failure receipt operational effect",
            ) not in {"failed_before_commit", "commit_uncertain"}:
                _fail("failure receipt operational effect differs")
        if (
            type(self.unresolved_recovery_nodes) is not tuple
            or any(type(item) is not str for item in self.unresolved_recovery_nodes)
            or any(item not in RECOVERY_NODE_NAMES for item in self.unresolved_recovery_nodes)
            or len(set(self.unresolved_recovery_nodes)) != len(self.unresolved_recovery_nodes)
        ):
            _fail("failure receipt unresolved recovery-node projection differs")
        if self.operational_failure_phase is None and not self.unresolved_recovery_nodes:
            _fail("failure receipt lacks an operational or recovery failure")
        for field in (
            "container_create_count_state",
            "container_start_count_state",
            "workload_start_count_state",
            "workload_exit_count_state",
            "attempt_count_state",
        ):
            if _require_text(getattr(self, field), f"failure receipt {field}") not in {
                "exact",
                "uncertain",
            }:
                _fail("failure receipt count state differs")
        for field in (
            "container_create_count",
            "container_start_count",
            "workload_start_count",
            "workload_exit_count",
            "attempt_count",
        ):
            value = getattr(self, field)
            if value is not None:
                _require_int(value, f"failure receipt {field}", maximum=1)
        _require_int(
            self.operational_failure_count,
            "failure receipt operational failure count",
            maximum=1,
        )
        _require_int(
            self.recovery_failure_count,
            "failure receipt recovery failure count",
            maximum=len(RECOVERY_NODE_NAMES),
        )
        _require_int(
            self.recovery_uncertainty_count,
            "failure receipt recovery uncertainty count",
            maximum=len(RECOVERY_NODE_NAMES),
        )
        if (
            _require_int(
                self.terminal_failure_count,
                "failure receipt terminal failure count",
                minimum=1,
                maximum=1,
            )
            != 1
        ):
            _fail("failure receipt terminal failure count must be exact one")
        _require_bool(self.cleanup_proven, "failure receipt cleanup proven")
        if _require_bool(self.ticket_quarantined, "failure ticket quarantine") is not True:
            _fail("terminal failure must quarantine its ticket")
        if _require_bool(self.reconciliation_only, "failure reconciliation-only") is not True:
            _fail("terminal failure may only continue through reconciliation")
        if _require_bool(self.case_consumed, "failure case consumed") is not True:
            _fail("terminal failure must consume the case")
        if _require_bool(self.same_case_retry_permitted, "failure retry") is not False:
            _fail("terminal failure cannot permit same-case retry")
        if _require_bool(self.clean_rejection_recorded, "failure clean rejection") is not False:
            _fail("operational/recovery failure is not a clean scientific rejection")

    def to_body_dict(self) -> dict[str, Any]:
        return {
            "schema_version": HOST_FAILURE_RECEIPT_SCHEMA_VERSION,
            "record_kind": "terminal_failure",
            **{
                field: (
                    getattr(self, field).to_dict()
                    if type(getattr(self, field))
                    in {ArtifactIdentityV2, HostNativePublicationProjectionV2}
                    else list(getattr(self, field))
                    if field == "unresolved_recovery_nodes"
                    else getattr(self, field)
                )
                for field in self.__dataclass_fields__
            },
            "authority": _authority(),
            "claims": _claims(),
        }


FAILURE_RECEIPT_BODY_SHA256_FIELD: Final = "failure_receipt_body_sha256"


def canonical_host_failure_receipt_v2_body_bytes(receipt: HostFailureReceiptV2) -> bytes:
    return _canonical_artifact_body_bytes(receipt, HostFailureReceiptV2)


def canonical_host_failure_receipt_v2_file_bytes(receipt: HostFailureReceiptV2) -> bytes:
    return _canonical_artifact_file_bytes(
        receipt,
        HostFailureReceiptV2,
        FAILURE_RECEIPT_BODY_SHA256_FIELD,
    )


def parse_host_failure_receipt_v2(
    raw: bytes,
    *,
    expected_file_sha256: str,
) -> HostFailureReceiptV2:
    body = _parse_artifact_file(
        raw,
        expected_file_sha256=expected_file_sha256,
        body_field=FAILURE_RECEIPT_BODY_SHA256_FIELD,
        body_keys=frozenset(HostFailureReceiptV2.__dataclass_fields__)
        | {"schema_version", "record_kind", "authority", "claims"},
        label="host failure receipt",
    )
    item = dict(body)
    if (
        item.pop("schema_version") != HOST_FAILURE_RECEIPT_SCHEMA_VERSION
        or item.pop("record_kind") != "terminal_failure"
        or item.pop("authority") != _authority()
        or item.pop("claims") != _claims()
    ):
        _fail("host failure receipt envelope differs")
    for field in (
        "request",
        "operational_frontier",
        "cleanup_reconciliation",
        "terminal_metadata",
        "lifecycle",
    ):
        item[field] = _artifact_identity(item[field], f"failure receipt {field}")
    for field in (
        "intent",
        "initial_sample",
        "ready",
        "observer_anchor",
        "go_commitment",
    ):
        item[field] = _optional_artifact_identity(
            item[field],
            f"failure receipt {field}",
        )
    item["native_publication"] = _native_publication_projection_from_dict(
        item["native_publication"]
    )
    unresolved = item.pop("unresolved_recovery_nodes")
    if type(unresolved) is not list:
        _fail("failure receipt unresolved nodes must be one list")
    result = HostFailureReceiptV2(**item, unresolved_recovery_nodes=tuple(unresolved))
    if raw != canonical_host_failure_receipt_v2_file_bytes(result):
        _fail("host failure receipt canonical replay differs")
    return result


def _canonical_identity(
    value: Any,
    schema_version: str,
    body_builder: Any,
    file_builder: Any,
) -> ArtifactIdentityV2:
    return _identity_from_canonical_bytes(
        schema_version,
        body_builder(value),
        file_builder(value),
    )


def validate_host_success_receipt_v2_chain(
    request: HostQualificationCaseRequestV2,
    intent: HostQualificationCaseIntentV2,
    ready: HostReadyV2,
    anchor: HostObserverAnchorV2,
    go: HostGoCommitmentV2,
    frontier: HostOperationalFrontierV2,
    reconciliation: HostCleanupReconciliationV2,
    proof: HostCgroupBoundaryProofV2,
    terminal: HostTerminalMetadataV2,
    lifecycle: HostLifecycleRollupV2,
    receipt: HostSuccessReceiptV2,
    *,
    initial_sample: HostInitialCgroupSampleV2,
    precleanup_sample: HostPrecleanupCgroupSampleV2,
    cgroup_kill_receipt: HostCgroupKillReceiptV2,
    cgroup_empty_observation: HostCgroupEmptyObservationV2,
    container_absence_observation: HostContainerAbsenceObservationV2,
    post_container_remove_sample: HostPostContainerRemoveCgroupSampleV2,
    cgroup_counter_fds_closed_receipt: HostCgroupCounterFdsClosedReceiptV2,
    outer_cgroup_absence_observation: HostOuterCgroupAbsenceObservationV2,
    membership_event_log: HostCgroupMembershipEventLogV2,
) -> None:
    if type(receipt) is not HostSuccessReceiptV2:
        raise TypeError("receipt must use the exact success-receipt-v2 type")
    validate_host_committed_prefix_v2_chain(
        request,
        intent,
        initial_sample,
        ready,
        anchor,
        go,
        frontier,
    )
    validate_host_cleanup_reconciliation_v2_chain(
        frontier,
        reconciliation,
        request=request,
        intent=intent,
        ready=ready,
        observer_anchor=anchor,
        go=go,
        membership_event_log=membership_event_log,
        initial_sample=initial_sample,
        precleanup_sample=precleanup_sample,
        cgroup_kill_receipt=cgroup_kill_receipt,
        cgroup_empty_observation=cgroup_empty_observation,
        container_absence_observation=container_absence_observation,
        post_container_remove_sample=post_container_remove_sample,
        cgroup_counter_fds_closed_receipt=cgroup_counter_fds_closed_receipt,
        outer_cgroup_absence_observation=outer_cgroup_absence_observation,
        cgroup_proof=proof,
    )
    validate_host_lifecycle_v2_chain(
        frontier,
        reconciliation,
        terminal,
        lifecycle,
        request=request,
    )
    if not frontier.succeeded or not reconciliation.cleanup_proven:
        _fail("success receipt cannot represent an operational or recovery failure")
    expected_links = {
        "request": _canonical_identity(
            request,
            HOST_CASE_REQUEST_SCHEMA_VERSION,
            canonical_host_case_request_v2_body_bytes,
            canonical_host_case_request_v2_file_bytes,
        ),
        "intent": _canonical_identity(
            intent,
            HOST_CASE_INTENT_SCHEMA_VERSION,
            canonical_host_case_intent_v2_body_bytes,
            canonical_host_case_intent_v2_file_bytes,
        ),
        "ready": _canonical_identity(
            ready,
            HOST_READY_SCHEMA_VERSION,
            canonical_host_ready_v2_body_bytes,
            canonical_host_ready_v2_file_bytes,
        ),
        "observer_anchor": _canonical_identity(
            anchor,
            HOST_OBSERVER_ANCHOR_SCHEMA_VERSION,
            canonical_host_observer_anchor_v2_body_bytes,
            canonical_host_observer_anchor_v2_file_bytes,
        ),
        "go_commitment": _canonical_identity(
            go,
            HOST_GO_SCHEMA_VERSION,
            canonical_host_go_v2_body_bytes,
            canonical_host_go_v2_file_bytes,
        ),
        "operational_frontier": _frontier_identity(frontier),
        "cleanup_reconciliation": _canonical_identity(
            reconciliation,
            HOST_CLEANUP_RECONCILIATION_SCHEMA_VERSION,
            canonical_host_cleanup_reconciliation_v2_body_bytes,
            canonical_host_cleanup_reconciliation_v2_file_bytes,
        ),
        "cgroup_proof": _canonical_identity(
            proof,
            HOST_CGROUP_PROOF_SCHEMA_VERSION,
            canonical_host_cgroup_boundary_proof_v2_body_bytes,
            canonical_host_cgroup_boundary_proof_v2_file_bytes,
        ),
        "terminal_metadata": _canonical_identity(
            terminal,
            HOST_TERMINAL_METADATA_SCHEMA_VERSION,
            canonical_host_terminal_metadata_v2_body_bytes,
            canonical_host_terminal_metadata_v2_file_bytes,
        ),
        "lifecycle": _canonical_identity(
            lifecycle,
            HOST_LIFECYCLE_SCHEMA_VERSION,
            canonical_host_lifecycle_v2_body_bytes,
            canonical_host_lifecycle_v2_file_bytes,
        ),
    }
    if any(getattr(receipt, field) != value for field, value in expected_links.items()):
        _fail("success receipt artifact chain is cross-wired")
    final_proof_node = reconciliation.recovery_nodes[-1]
    if (
        final_proof_node.node_name != "final_cgroup_proof"
        or final_proof_node.state != "committed"
        or final_proof_node.artifact != expected_links["cgroup_proof"]
    ):
        _fail("success receipt cgroup proof differs from the cleanup DAG terminal node")
    for source in (request, intent, ready, terminal, lifecycle, receipt):
        if source.case_spine_sha256 != request.case_spine_sha256:
            _fail("success receipt chain crosses case spines")
    case_fields = ("case_ordinal", "candidate_id", "candidate_family", "qualification_case_id")
    if any(
        getattr(source, field) != getattr(request, field)
        for source in (intent, ready, terminal, lifecycle, receipt)
        for field in case_fields
    ):
        _fail("success receipt chain crosses case identities")
    if (
        receipt.container_create_count != frontier.container_create_count
        or receipt.container_start_count != frontier.container_start_count
        or receipt.workload_start_count != frontier.workload_start_count
        or receipt.workload_exit_count != frontier.workload_exit_count
        or receipt.attempt_count != frontier.attempt_count
        or receipt.failure_count != frontier.failure_count
        or receipt.go_commit_count != go.go_commit_count
    ):
        _fail("success receipt counts differ from the exact operational chain")


def _failure_classification(
    frontier: HostOperationalFrontierV2,
    reconciliation: HostCleanupReconciliationV2,
) -> str:
    if frontier.failure_phase is not None:
        if frontier.failure_effect_state == "commit_uncertain":
            return "operational_commit_uncertain_ticket_quarantined_nonretryable"
        return "operational_failed_before_commit_ticket_quarantined_nonretryable"
    return "recovery_failure_after_complete_operational_frontier_nonretryable"


def validate_host_failure_receipt_v2_chain(
    request: HostQualificationCaseRequestV2,
    intent: HostQualificationCaseIntentV2 | None,
    frontier: HostOperationalFrontierV2,
    reconciliation: HostCleanupReconciliationV2,
    terminal: HostTerminalMetadataV2,
    lifecycle: HostLifecycleRollupV2,
    receipt: HostFailureReceiptV2,
    *,
    initial_sample: HostInitialCgroupSampleV2 | None = None,
    ready: HostReadyV2 | None = None,
    anchor: HostObserverAnchorV2 | None = None,
    go: HostGoCommitmentV2 | None = None,
    precleanup_sample: HostPrecleanupCgroupSampleV2 | None = None,
    cgroup_kill_receipt: HostCgroupKillReceiptV2 | None = None,
    cgroup_empty_observation: HostCgroupEmptyObservationV2 | None = None,
    container_absence_observation: HostContainerAbsenceObservationV2 | None = None,
    post_container_remove_sample: HostPostContainerRemoveCgroupSampleV2 | None = None,
    cgroup_counter_fds_closed_receipt: HostCgroupCounterFdsClosedReceiptV2 | None = None,
    outer_cgroup_absence_observation: HostOuterCgroupAbsenceObservationV2 | None = None,
    membership_event_log: HostCgroupMembershipEventLogV2 | None = None,
    cgroup_proof: HostCgroupBoundaryProofV2 | None = None,
) -> None:
    if type(receipt) is not HostFailureReceiptV2:
        raise TypeError("receipt must use the exact failure-receipt-v2 type")
    validate_host_committed_prefix_v2_chain(
        request,
        intent,
        initial_sample,
        ready,
        anchor,
        go,
        frontier,
    )
    validate_host_cleanup_reconciliation_v2_chain(
        frontier,
        reconciliation,
        request=request,
        intent=intent,
        ready=ready,
        observer_anchor=anchor,
        go=go,
        membership_event_log=membership_event_log,
        initial_sample=initial_sample,
        precleanup_sample=precleanup_sample,
        cgroup_kill_receipt=cgroup_kill_receipt,
        cgroup_empty_observation=cgroup_empty_observation,
        container_absence_observation=container_absence_observation,
        post_container_remove_sample=post_container_remove_sample,
        cgroup_counter_fds_closed_receipt=cgroup_counter_fds_closed_receipt,
        outer_cgroup_absence_observation=outer_cgroup_absence_observation,
        cgroup_proof=cgroup_proof,
    )
    validate_host_lifecycle_v2_chain(
        frontier,
        reconciliation,
        terminal,
        lifecycle,
        request=request,
    )
    if terminal.record_kind != "terminal_failure":
        _fail("failure receipt requires common terminal-failure metadata")
    request_id = _canonical_identity(
        request,
        HOST_CASE_REQUEST_SCHEMA_VERSION,
        canonical_host_case_request_v2_body_bytes,
        canonical_host_case_request_v2_file_bytes,
    )
    if intent is None:
        intent_id = None
    else:
        validate_host_request_intent_v2_chain(request, intent)
        intent_id = _canonical_identity(
            intent,
            HOST_CASE_INTENT_SCHEMA_VERSION,
            canonical_host_case_intent_v2_body_bytes,
            canonical_host_case_intent_v2_file_bytes,
        )
    intent_committed = "intent_committed" in frontier.completed_phases
    if intent_committed is not (intent is not None):
        _fail("failure intent artifact differs from its operational commit boundary")
    prefix_values: tuple[
        tuple[
            str,
            object | None,
            str,
            str,
            Callable[[Any], bytes],
            Callable[[Any], bytes],
        ],
        ...,
    ] = (
        (
            "initial_sample",
            initial_sample,
            "initial_cgroup_sample_committed",
            HOST_INITIAL_CGROUP_SAMPLE_SCHEMA_VERSION,
            canonical_host_initial_cgroup_sample_v2_body_bytes,
            canonical_host_initial_cgroup_sample_v2_file_bytes,
        ),
        (
            "ready",
            ready,
            "driver_ready",
            HOST_READY_SCHEMA_VERSION,
            canonical_host_ready_v2_body_bytes,
            canonical_host_ready_v2_file_bytes,
        ),
        (
            "observer_anchor",
            anchor,
            "observer_anchored",
            HOST_OBSERVER_ANCHOR_SCHEMA_VERSION,
            canonical_host_observer_anchor_v2_body_bytes,
            canonical_host_observer_anchor_v2_file_bytes,
        ),
        (
            "go_commitment",
            go,
            "go_committed",
            HOST_GO_SCHEMA_VERSION,
            canonical_host_go_v2_body_bytes,
            canonical_host_go_v2_file_bytes,
        ),
    )
    prefix_links: dict[str, ArtifactIdentityV2 | None] = {}
    for field, value, phase, schema, body_builder, file_builder in prefix_values:
        committed = phase in frontier.completed_phases
        if committed is not (value is not None):
            _fail(f"failure {field} artifact differs from its operational commit boundary")
        prefix_links[field] = (
            None
            if value is None
            else _canonical_identity(
                value,
                schema,
                body_builder,
                file_builder,
            )
        )
    expected_links = {
        "request": request_id,
        "intent": intent_id,
        **prefix_links,
        "operational_frontier": _frontier_identity(frontier),
        "cleanup_reconciliation": _canonical_identity(
            reconciliation,
            HOST_CLEANUP_RECONCILIATION_SCHEMA_VERSION,
            canonical_host_cleanup_reconciliation_v2_body_bytes,
            canonical_host_cleanup_reconciliation_v2_file_bytes,
        ),
        "terminal_metadata": _canonical_identity(
            terminal,
            HOST_TERMINAL_METADATA_SCHEMA_VERSION,
            canonical_host_terminal_metadata_v2_body_bytes,
            canonical_host_terminal_metadata_v2_file_bytes,
        ),
        "lifecycle": _canonical_identity(
            lifecycle,
            HOST_LIFECYCLE_SCHEMA_VERSION,
            canonical_host_lifecycle_v2_body_bytes,
            canonical_host_lifecycle_v2_file_bytes,
        ),
    }
    if any(getattr(receipt, field) != value for field, value in expected_links.items()):
        _fail("failure receipt artifact chain is cross-wired")
    if (
        receipt.case_spine_sha256 != request.case_spine_sha256
        or receipt.case_spine_sha256 != frontier.case_spine_sha256
        or receipt.case_spine_sha256 != terminal.case_spine_sha256
        or receipt.operational_failure_phase != frontier.failure_phase
        or receipt.operational_failure_effect_state != frontier.failure_effect_state
        or receipt.unresolved_recovery_nodes != reconciliation.unresolved_recovery_nodes
        or receipt.classification != _failure_classification(frontier, reconciliation)
        or receipt.error_message_sha256 != terminal.error_message_sha256
        or receipt.cleanup_proven is not reconciliation.cleanup_proven
        or receipt.native_publication != lifecycle.native_publication
    ):
        _fail("failure receipt projections differ from frontier, recovery, or terminal state")
    count_fields = (
        "container_create_count_state",
        "container_create_count",
        "container_start_count_state",
        "container_start_count",
        "workload_start_count_state",
        "workload_start_count",
        "workload_exit_count_state",
        "workload_exit_count",
        "attempt_count_state",
        "attempt_count",
    )
    if any(getattr(receipt, field) != getattr(frontier, field) for field in count_fields):
        _fail("failure receipt count projections differ from the operational frontier")
    if receipt.operational_failure_count != frontier.failure_count:
        _fail("failure receipt operational failure count differs")
    if (
        receipt.recovery_failure_count
        != sum(node.state == "failed_before_commit" for node in reconciliation.recovery_nodes)
        or receipt.recovery_uncertainty_count
        != sum(node.state == "commit_uncertain" for node in reconciliation.recovery_nodes)
    ):
        _fail("failure receipt recovery failure/uncertainty counts differ")
    case_fields = ("case_ordinal", "candidate_id", "candidate_family", "qualification_case_id")
    if any(getattr(receipt, field) != getattr(request, field) for field in case_fields):
        _fail("failure receipt case projection differs from its request")


@dataclass(frozen=True, slots=True)
class HostObservationHandoffV2:
    case_spine_sha256: str
    case_ordinal: int
    candidate_id: str
    qualification_case_id: str
    record_kind: Literal["success", "terminal_failure"]
    terminal_receipt_file_sha256: str
    terminal_receipt_body_sha256: str
    terminal_metadata_file_sha256: str
    terminal_metadata_body_sha256: str
    structural_success_shape_only: bool
    production_execution_success: bool
    production_acceptance_eligible: bool
    evidence_eligible: bool

    def __post_init__(self) -> None:
        ordinal = _require_int(
            self.case_ordinal,
            "handoff case ordinal",
            maximum=len(MATCHED_V3_CANDIDATE_IDS) - 1,
        )
        candidate = _require_identifier(self.candidate_id, "handoff candidate ID")
        case_id = _require_identifier(self.qualification_case_id, "handoff case ID")
        if (
            candidate != MATCHED_V3_CANDIDATE_IDS[ordinal]
            or case_id != f"qualification_{ordinal:02d}_{candidate}"
        ):
            _fail("handoff case projection differs from the matched-v3 order")
        _require_sha256(self.case_spine_sha256, "handoff case spine")
        if _require_text(self.record_kind, "handoff record kind") not in {
            "success",
            "terminal_failure",
        }:
            _fail("handoff record kind differs")
        for value, label in (
            (self.terminal_receipt_file_sha256, "handoff terminal receipt FILE"),
            (self.terminal_receipt_body_sha256, "handoff terminal receipt BODY"),
            (self.terminal_metadata_file_sha256, "handoff terminal metadata FILE"),
            (self.terminal_metadata_body_sha256, "handoff terminal metadata BODY"),
        ):
            _require_sha256(value, label)
        if (
            self.terminal_receipt_file_sha256 == self.terminal_metadata_file_sha256
            or self.terminal_receipt_body_sha256 == self.terminal_metadata_body_sha256
        ):
            _fail("handoff receipt and common terminal metadata must remain distinct")
        if (
            _require_bool(
                self.structural_success_shape_only,
                "handoff structural success shape",
            )
            is not (self.record_kind == "success")
            or _require_bool(
                self.production_execution_success,
                "handoff production execution success",
            )
            is not False
            or _require_bool(
                self.production_acceptance_eligible,
                "handoff production acceptance eligibility",
            )
            is not False
            or _require_bool(self.evidence_eligible, "handoff evidence eligibility")
            is not False
        ):
            _fail("source-only handoff cannot claim production or evidence success")

    def to_body_dict(self) -> dict[str, Any]:
        return {
            "schema_version": HOST_OBSERVATION_HANDOFF_SCHEMA_VERSION,
            **{field: getattr(self, field) for field in self.__dataclass_fields__},
        }


HANDOFF_BODY_SHA256_FIELD: Final = "handoff_body_sha256"


def canonical_host_observation_handoff_v2_body_bytes(
    handoff: HostObservationHandoffV2,
) -> bytes:
    return _canonical_artifact_body_bytes(handoff, HostObservationHandoffV2)


def canonical_host_observation_handoff_v2_file_bytes(
    handoff: HostObservationHandoffV2,
) -> bytes:
    return _canonical_artifact_file_bytes(
        handoff,
        HostObservationHandoffV2,
        HANDOFF_BODY_SHA256_FIELD,
    )


def parse_host_observation_handoff_v2(
    raw: bytes,
    *,
    expected_file_sha256: str,
) -> HostObservationHandoffV2:
    body = _parse_artifact_file(
        raw,
        expected_file_sha256=expected_file_sha256,
        body_field=HANDOFF_BODY_SHA256_FIELD,
        body_keys=frozenset(HostObservationHandoffV2.__dataclass_fields__)
        | {"schema_version"},
        label="host observation handoff",
    )
    item = dict(body)
    if item.pop("schema_version") != HOST_OBSERVATION_HANDOFF_SCHEMA_VERSION:
        _fail("host observation handoff schema differs")
    result = HostObservationHandoffV2(**item)
    if raw != canonical_host_observation_handoff_v2_file_bytes(result):
        _fail("host observation handoff canonical replay differs")
    return result


def validate_host_observation_handoff_v2_chain(
    receipt: HostSuccessReceiptV2 | HostFailureReceiptV2,
    terminal: HostTerminalMetadataV2,
    handoff: HostObservationHandoffV2,
) -> None:
    if type(terminal) is not HostTerminalMetadataV2:
        raise TypeError("terminal must use the exact terminal-metadata-v2 type")
    if type(handoff) is not HostObservationHandoffV2:
        raise TypeError("handoff must use the exact observation-handoff-v2 type")
    if type(receipt) is HostSuccessReceiptV2:
        record_kind = "success"
        receipt_body = canonical_host_success_receipt_v2_body_bytes(receipt)
        receipt_file = canonical_host_success_receipt_v2_file_bytes(receipt)
    elif type(receipt) is HostFailureReceiptV2:
        if receipt.failure_receipt_state != "committed":
            _fail("failure handoff requires a known committed failure receipt")
        record_kind = "terminal_failure"
        receipt_body = canonical_host_failure_receipt_v2_body_bytes(receipt)
        receipt_file = canonical_host_failure_receipt_v2_file_bytes(receipt)
    else:
        raise TypeError("receipt must use an exact host-v2 terminal receipt type")
    terminal_body = canonical_host_terminal_metadata_v2_body_bytes(terminal)
    terminal_file = canonical_host_terminal_metadata_v2_file_bytes(terminal)
    expected = {
        "case_spine_sha256": receipt.case_spine_sha256,
        "case_ordinal": receipt.case_ordinal,
        "candidate_id": receipt.candidate_id,
        "qualification_case_id": receipt.qualification_case_id,
        "record_kind": record_kind,
        "terminal_receipt_file_sha256": _sha256(receipt_file),
        "terminal_receipt_body_sha256": _sha256(receipt_body),
        "terminal_metadata_file_sha256": _sha256(terminal_file),
        "terminal_metadata_body_sha256": _sha256(terminal_body),
        "structural_success_shape_only": record_kind == "success",
        "production_execution_success": False,
        "production_acceptance_eligible": False,
        "evidence_eligible": False,
    }
    if any(getattr(handoff, field) != value for field, value in expected.items()):
        _fail("observation handoff does not bind the exact terminal receipt and metadata")
    if (
        terminal.record_kind != record_kind
        or terminal.case_spine_sha256 != receipt.case_spine_sha256
        or terminal.case_ordinal != receipt.case_ordinal
        or terminal.candidate_id != receipt.candidate_id
        or terminal.candidate_family != receipt.candidate_family
        or terminal.qualification_case_id != receipt.qualification_case_id
        or receipt.terminal_metadata.file_sha256 != _sha256(terminal_file)
        or receipt.terminal_metadata.body_sha256 != _sha256(terminal_body)
    ):
        _fail("observation handoff eligibility chain differs")


HOST_EXECUTOR_PUBLIC_CANONICAL_BUILDERS: Final = (
    "canonical_host_executor_descriptor_v2_body_bytes",
    "canonical_host_executor_descriptor_v2_file_bytes",
    "canonical_host_case_request_v2_body_bytes",
    "canonical_host_case_request_v2_file_bytes",
    "canonical_host_case_intent_v2_body_bytes",
    "canonical_host_case_intent_v2_file_bytes",
    "canonical_host_ready_v2_body_bytes",
    "canonical_host_ready_v2_file_bytes",
    "canonical_host_observer_anchor_v2_body_bytes",
    "canonical_host_observer_anchor_v2_file_bytes",
    "canonical_host_go_v2_body_bytes",
    "canonical_host_go_v2_file_bytes",
    "canonical_host_operational_frontier_v2_body_bytes",
    "canonical_host_operational_frontier_v2_file_bytes",
    "canonical_host_cgroup_membership_event_log_v2_body_bytes",
    "canonical_host_cgroup_membership_event_log_v2_file_bytes",
    "canonical_host_initial_cgroup_sample_v2_body_bytes",
    "canonical_host_initial_cgroup_sample_v2_file_bytes",
    "canonical_host_precleanup_cgroup_sample_v2_body_bytes",
    "canonical_host_precleanup_cgroup_sample_v2_file_bytes",
    "canonical_host_cgroup_kill_receipt_v2_body_bytes",
    "canonical_host_cgroup_kill_receipt_v2_file_bytes",
    "canonical_host_cgroup_empty_observation_v2_body_bytes",
    "canonical_host_cgroup_empty_observation_v2_file_bytes",
    "canonical_host_container_absence_observation_v2_body_bytes",
    "canonical_host_container_absence_observation_v2_file_bytes",
    "canonical_host_post_container_remove_cgroup_sample_v2_body_bytes",
    "canonical_host_post_container_remove_cgroup_sample_v2_file_bytes",
    "canonical_host_cgroup_counter_fds_closed_receipt_v2_body_bytes",
    "canonical_host_cgroup_counter_fds_closed_receipt_v2_file_bytes",
    "canonical_host_outer_cgroup_absence_observation_v2_body_bytes",
    "canonical_host_outer_cgroup_absence_observation_v2_file_bytes",
    "canonical_host_cgroup_boundary_proof_v2_body_bytes",
    "canonical_host_cgroup_boundary_proof_v2_file_bytes",
    "canonical_host_cleanup_reconciliation_v2_body_bytes",
    "canonical_host_cleanup_reconciliation_v2_file_bytes",
    "canonical_host_terminal_metadata_v2_body_bytes",
    "canonical_host_terminal_metadata_v2_file_bytes",
    "canonical_host_lifecycle_v2_body_bytes",
    "canonical_host_lifecycle_v2_file_bytes",
    "canonical_host_success_receipt_v2_body_bytes",
    "canonical_host_success_receipt_v2_file_bytes",
    "canonical_host_failure_receipt_v2_body_bytes",
    "canonical_host_failure_receipt_v2_file_bytes",
    "canonical_host_observation_handoff_v2_body_bytes",
    "canonical_host_observation_handoff_v2_file_bytes",
)
HOST_EXECUTOR_PUBLIC_PARSERS: Final = (
    "parse_host_executor_descriptor_v2",
    "parse_host_case_request_v2",
    "parse_host_case_intent_v2",
    "parse_host_ready_v2",
    "parse_host_observer_anchor_v2",
    "parse_host_go_v2",
    "parse_host_operational_frontier_v2",
    "parse_host_cgroup_membership_event_log_v2",
    "parse_host_initial_cgroup_sample_v2",
    "parse_host_precleanup_cgroup_sample_v2",
    "parse_host_cgroup_kill_receipt_v2",
    "parse_host_cgroup_empty_observation_v2",
    "parse_host_container_absence_observation_v2",
    "parse_host_post_container_remove_cgroup_sample_v2",
    "parse_host_cgroup_counter_fds_closed_receipt_v2",
    "parse_host_outer_cgroup_absence_observation_v2",
    "parse_host_cgroup_boundary_proof_v2",
    "parse_host_cleanup_reconciliation_v2",
    "parse_host_terminal_metadata_v2",
    "parse_host_lifecycle_v2",
    "parse_host_success_receipt_v2",
    "parse_host_failure_receipt_v2",
    "parse_host_observation_handoff_v2",
)
HOST_EXECUTOR_PUBLIC_VALIDATORS: Final = (
    "validate_host_request_intent_v2_chain",
    "validate_host_committed_prefix_v2_chain",
    "validate_host_ready_anchor_go_v2_chain",
    "validate_retained_cgroup_fd_inventory_v2",
    "validate_host_cgroup_boundary_proof_v2_chain",
    "validate_host_cleanup_reconciliation_v2_chain",
    "validate_host_terminal_metadata_v2_chain",
    "validate_host_lifecycle_v2_chain",
    "validate_host_success_receipt_v2_chain",
    "validate_host_failure_receipt_v2_chain",
    "validate_host_observation_handoff_v2_chain",
)
HOST_EXECUTOR_OPERATIONAL_APIS: Final[tuple[str, ...]] = ()

PINNED_HOST_EXECUTOR_V2_DESCRIPTOR_FILE_SHA256: Final = (
    "24c205cc4e3d189b4580512c281a84764d81dce51f43e5d959b8650a516343a4"
)


@dataclass(frozen=True, slots=True)
class HostExecutorDescriptorV2:
    """The immutable, zero-capability phase-1 descriptor value."""

    def to_body_dict(self) -> dict[str, Any]:
        raw_resource_bindings = (
            {
                "resource_field": "max_peak_rss_bytes",
                "raw_observation": "fresh_cgroup_memory_peak_bytes",
                "value_semantics": "conservative_observed_upper_bound",
            },
            {
                "resource_field": "max_cpu_time_ns",
                "raw_observation": "fresh_cgroup_cpu_usage_usec_delta_times_1000",
                "value_semantics": "exact_observation",
            },
            {
                "resource_field": "max_wall_time_ns",
                "raw_observation": "initial_to_post_remove_monotonic_ns_delta",
                "value_semantics": "exact_observation",
            },
            {
                "resource_field": "max_thread_count",
                "raw_observation": "fresh_cgroup_pids_peak",
                "value_semantics": "conservative_observed_upper_bound",
            },
            {
                "resource_field": "max_attempt_count",
                "raw_observation": "workload_start_count",
                "value_semantics": "exact_observation",
            },
            {
                "resource_field": "max_failure_count",
                "raw_observation": "operational_failure_count",
                "value_semantics": "exact_observation",
            },
        )
        kernel_quota_breach_bindings = (
            {
                "raw_observation": "fresh_cgroup_memory_oom_kill_count_delta",
                "required_structural_success_value": 0,
                "value_semantics": "exact_observation",
            },
            {
                "raw_observation": "fresh_cgroup_pids_max_event_count_delta",
                "required_structural_success_value": 0,
                "value_semantics": "exact_observation",
            },
        )
        schemas = {
            "request": HOST_CASE_REQUEST_SCHEMA_VERSION,
            "intent": HOST_CASE_INTENT_SCHEMA_VERSION,
            "ready": HOST_READY_SCHEMA_VERSION,
            "observer_anchor": HOST_OBSERVER_ANCHOR_SCHEMA_VERSION,
            "go_commitment": HOST_GO_SCHEMA_VERSION,
            "operational_frontier": HOST_OPERATIONAL_FRONTIER_SCHEMA_VERSION,
            "initial_cgroup_sample": HOST_INITIAL_CGROUP_SAMPLE_SCHEMA_VERSION,
            "precleanup_cgroup_sample": HOST_PRECLEANUP_CGROUP_SAMPLE_SCHEMA_VERSION,
            "cgroup_kill_receipt": HOST_CGROUP_KILL_RECEIPT_SCHEMA_VERSION,
            "cgroup_empty_observation": HOST_CGROUP_EMPTY_OBSERVATION_SCHEMA_VERSION,
            "container_absence_observation": HOST_CONTAINER_ABSENCE_OBSERVATION_SCHEMA_VERSION,
            "post_container_remove_cgroup_sample": (
                HOST_POST_CONTAINER_REMOVE_CGROUP_SAMPLE_SCHEMA_VERSION
            ),
            "cgroup_counter_fds_closed_receipt": (
                HOST_CGROUP_COUNTER_FDS_CLOSED_RECEIPT_SCHEMA_VERSION
            ),
            "outer_cgroup_absence_observation": (
                HOST_OUTER_CGROUP_ABSENCE_OBSERVATION_SCHEMA_VERSION
            ),
            "cgroup_boundary_proof": HOST_CGROUP_PROOF_SCHEMA_VERSION,
            "cgroup_membership_event_log": (
                HOST_CGROUP_MEMBERSHIP_EVENT_LOG_SCHEMA_VERSION
            ),
            "cleanup_reconciliation": HOST_CLEANUP_RECONCILIATION_SCHEMA_VERSION,
            "terminal_metadata": HOST_TERMINAL_METADATA_SCHEMA_VERSION,
            "lifecycle": HOST_LIFECYCLE_SCHEMA_VERSION,
            "success_receipt": HOST_SUCCESS_RECEIPT_SCHEMA_VERSION,
            "failure_receipt": HOST_FAILURE_RECEIPT_SCHEMA_VERSION,
            "observation_handoff": HOST_OBSERVATION_HANDOFF_SCHEMA_VERSION,
        }
        return {
            "schema_version": HOST_EXECUTOR_DESCRIPTOR_SCHEMA_VERSION,
            "status": "implemented_source_only_uninvoked_non_authorizing",
            "scope": "canonical_metadata_contract_only",
            "candidate_order": list(MATCHED_V3_CANDIDATE_IDS),
            "resource_fields": list(RESOURCE_FIELDS),
            "raw_resource_bindings": list(raw_resource_bindings),
            "kernel_quota_breach_bindings": list(kernel_quota_breach_bindings),
            "operational_phases": list(OPERATIONAL_PHASES),
            "boundary_state_semantics": {
                "not_started": "known_not_committed",
                "failed_before_commit": "known_noncommit",
                "commit_uncertain": "side_effect_commit_unknown",
                "committed": "canonical_artifact_or_boundary_committed",
                "uncertain_count_value": None,
            },
            "recovery_dag": [
                {
                    "node_name": name,
                    "dependencies": list(RECOVERY_NODE_DEPENDENCIES[name]),
                    "artifact_schema_version": RECOVERY_NODE_SCHEMAS[name],
                }
                for name in RECOVERY_NODE_NAMES
            ],
            "recovery_semantics": {
                "conditional_branches_continue_after_unrelated_uncertainty": True,
                "terminalization_requires_cleanup_reconciliation_record": True,
                "terminalization_requires_cleanup_proof": False,
                "retained_fd_order": [
                    "post_container_remove_cgroup_sample",
                    "cgroup_counter_fds_closed",
                    "outer_cgroup_absence",
                    "final_cgroup_proof",
                ],
                "workload_resume_permitted": False,
                "same_case_retry_permitted": False,
            },
            "containment_contract": {
                "supported_route": "cgroupfs_qualified_host",
                "delegate_root_path": CGROUP_DELEGATE_ROOT_PATH,
                "docker_cgroup_parent": CGROUP_DELEGATE_PARENT_ARGUMENT,
                "enabled_controllers": ["cpu", "memory", "pids"],
                "maximum_container_depth": 1,
                "maximum_container_descendants": 1,
                "container_name_derived_from_complete_case_spine": True,
                "container_cgroup_child_path_is_derived": True,
                "actual_container_cgroup_path_observed": False,
                "actual_container_cgroup_identity_authenticated": False,
                "production_v3_requires_observed_container_cgroup_path": True,
                "membership_requires_caller_pinned_canonical_event_log": True,
                "membership_log_is_three_structural_snapshots_not_continuous_stream": True,
                "opaque_identity_or_summary_boolean_is_proof": False,
                "production_containment_claim_available": False,
                "provisioning_receipt_semantics_validated": False,
                "provisioning_receipt_producer_authenticated": False,
                "production_acceptance_eligible": False,
                "future_production_backend_requires_additive_new_schema": True,
            },
            "terminal_chain": [
                "operational_frontier",
                "cleanup_reconciliation",
                "terminal_metadata",
                "lifecycle",
                "terminal_receipt",
                "observation_handoff",
            ],
            "artifact_schemas": schemas,
            "final_dependency_pins": {
                "algorithmic_contract_descriptor_sha256": (
                    FINAL_ALGORITHMIC_CONTRACT_DESCRIPTOR_SHA256
                ),
                "algorithmic_contract_source_sha256": (
                    FINAL_ALGORITHMIC_CONTRACT_SOURCE_SHA256
                ),
                "publication_contract_descriptor_sha256": (
                    FINAL_PUBLICATION_CONTRACT_DESCRIPTOR_SHA256
                ),
                "publication_contract_source_sha256": FINAL_PUBLICATION_CONTRACT_SOURCE_SHA256,
                "storage_contract_descriptor_sha256": (
                    FINAL_STORAGE_CONTRACT_DESCRIPTOR_SHA256
                ),
                "storage_contract_source_sha256": FINAL_STORAGE_CONTRACT_SOURCE_SHA256,
            },
            "permanent_exclusions": {
                "host_v1_descriptor_sha256": INCOMPATIBLE_HOST_V1_DESCRIPTOR_SHA256,
                "host_v1_source_sha256": INCOMPATIBLE_HOST_V1_SOURCE_SHA256,
                "historical_image_ids": list(HISTORICAL_IMAGE_IDS),
                "historical_build_lineage_sha256s": sorted(
                    HISTORICAL_BUILD_LINEAGE_SHA256S
                ),
                "incompatible_adapter_identity_sha256s": sorted(
                    INCOMPATIBLE_ADAPTER_IDENTITY_SHA256S
                ),
            },
            "execution_acknowledgement": HOST_EXECUTION_ACKNOWLEDGEMENT,
            "acknowledgement_is_execution_authority": False,
            "public_api": {
                "canonical_builders": list(HOST_EXECUTOR_PUBLIC_CANONICAL_BUILDERS),
                "strict_parsers": list(HOST_EXECUTOR_PUBLIC_PARSERS),
                "chain_validators": list(HOST_EXECUTOR_PUBLIC_VALIDATORS),
                "operational_apis": list(HOST_EXECUTOR_OPERATIONAL_APIS),
            },
            "capabilities": {
                "executes_subprocesses": False,
                "invokes_container_runtime": False,
                "mutates_cgroups": False,
                "creates_processes": False,
                "writes_artifacts": False,
                "issues_authority": False,
                "evaluates_outputs": False,
                "merges_resources": False,
                "production_backend_available": False,
            },
            "limitations": [
                "no_operational_backend",
                "no_docker_or_oci_api",
                "no_cgroup_or_process_api",
                "does_not_parse_or_pin_raw_host_provisioning_receipts",
                "cannot_alone_establish_containment_or_production_success",
                "structural_success_is_not_production_acceptance_or_scientific_evidence",
                "cannot_fill_matched_v3_execution_gap",
                "cannot_promote_or_create_scientific_evidence",
                "raw_container_id_child_path_is_nonportable_source_only",
                "production_v3_must_observe_actual_path_device_inode_from_retained_pid",
                "descriptor_file_pin_requires_separate_audit_before_use",
            ],
            "authority": _authority(),
            "claims": _claims(),
        }


HOST_EXECUTOR_DESCRIPTOR_BODY_SHA256_FIELD: Final = "descriptor_body_sha256"


def canonical_host_executor_descriptor_v2_body_bytes(
    descriptor: HostExecutorDescriptorV2,
) -> bytes:
    return _canonical_artifact_body_bytes(descriptor, HostExecutorDescriptorV2)


def canonical_host_executor_descriptor_v2_file_bytes(
    descriptor: HostExecutorDescriptorV2,
) -> bytes:
    return _canonical_artifact_file_bytes(
        descriptor,
        HostExecutorDescriptorV2,
        HOST_EXECUTOR_DESCRIPTOR_BODY_SHA256_FIELD,
    )


def _guard_host_executor_descriptor_pin() -> str:
    descriptor = HostExecutorDescriptorV2()
    body = canonical_host_executor_descriptor_v2_body_bytes(descriptor)
    file = canonical_host_executor_descriptor_v2_file_bytes(descriptor)
    embedded = _strict_json(file).get(HOST_EXECUTOR_DESCRIPTOR_BODY_SHA256_FIELD)
    observed_body = _sha256(body)
    if type(embedded) is not str or not hmac.compare_digest(embedded, observed_body):
        _fail("host executor descriptor embedded BODY identity differs")
    pin = PINNED_HOST_EXECUTOR_V2_DESCRIPTOR_FILE_SHA256
    if type(pin) is not str or _SHA256_RE.fullmatch(pin) is None or pin == "0" * 64:
        _fail("host executor descriptor FILE pin is not independently finalized")
    observed_file = _sha256(file)
    if not hmac.compare_digest(pin, observed_file):
        _fail("host executor descriptor FILE drifted from its repository literal")
    return pin


def parse_host_executor_descriptor_v2(
    raw: bytes,
    *,
    expected_file_sha256: str,
) -> HostExecutorDescriptorV2:
    pinned = _guard_host_executor_descriptor_pin()
    caller_pin = _require_sha256(expected_file_sha256, "host executor descriptor caller pin")
    if not hmac.compare_digest(caller_pin, pinned):
        _fail("host executor descriptor caller pin differs from the repository literal")
    expected = HostExecutorDescriptorV2()
    expected_body = expected.to_body_dict()
    body = _parse_artifact_file(
        raw,
        expected_file_sha256=pinned,
        body_field=HOST_EXECUTOR_DESCRIPTOR_BODY_SHA256_FIELD,
        body_keys=frozenset(expected_body),
        label="host executor descriptor",
    )
    if body != expected_body:
        _fail("host executor descriptor content differs")
    if raw != canonical_host_executor_descriptor_v2_file_bytes(expected):
        _fail("host executor descriptor canonical replay differs")
    return expected


def host_executor_v2_descriptor_sha256() -> str:
    """Return the descriptor FILE identity only after its repository pin is finalized."""

    return _guard_host_executor_descriptor_pin()


__all__ = [
    "ArtifactIdentityV2",
    "CgroupSampleFactsV2",
    "ForagerMatchedV3HostQualificationExecutorV2Error",
    "HostCgroupBoundaryProofV2",
    "HostCgroupCaseIdentityV2",
    "HostCgroupMembershipEventLogV2",
    "HostCgroupMembershipEventV2",
    "HostCgroupCounterFdsClosedReceiptV2",
    "HostCgroupEmptyObservationV2",
    "HostCgroupKillReceiptV2",
    "HostCleanupReconciliationV2",
    "HostContainerAbsenceObservationV2",
    "HostExecutorDescriptorV2",
    "HostFailureReceiptV2",
    "HostGoCommitmentV2",
    "HostInitialCgroupSampleV2",
    "HostLifecycleRollupV2",
    "HostNativePublicationProjectionV2",
    "HostObservationHandoffV2",
    "HostObserverAnchorV2",
    "HostObserverTerminalEvidenceV2",
    "HostOperationalFrontierV2",
    "HostOuterCgroupAbsenceObservationV2",
    "HostPostContainerRemoveCgroupSampleV2",
    "HostPrecleanupCgroupSampleV2",
    "HostQualificationCaseIntentV2",
    "HostQualificationCaseRequestV2",
    "HostRawResourceMeasurementsV2",
    "HostReadyV2",
    "HostSuccessReceiptV2",
    "HostTerminalMetadataV2",
    "ProducerIdentityV2",
    "RecoveryNodeV2",
    "RetainedCgroupCounterFdV2",
    "HOST_EXECUTION_ACKNOWLEDGEMENT",
    "HOST_EXECUTOR_DESCRIPTOR_SCHEMA_VERSION",
    "HOST_EXECUTOR_OPERATIONAL_APIS",
    "HOST_OBSERVATION_HANDOFF_SCHEMA_VERSION",
    "MATCHED_V3_CANDIDATE_IDS",
    "OPERATIONAL_PHASES",
    "PINNED_HOST_EXECUTOR_V2_DESCRIPTOR_FILE_SHA256",
    "RECOVERY_NODE_DEPENDENCIES",
    "RECOVERY_NODE_NAMES",
    "RESOURCE_FIELDS",
    "cgroup_case_identity_sha256_v2",
    "container_lookup_identity_sha256_v2",
    "container_runtime_identity_sha256_v2",
    "expected_container_name_v2",
    "observer_terminal_evidence_sha256_v2",
    "publication_reconciliation_key_sha256_v2",
    "retained_cgroup_fd_inventory_sha256_v2",
    *HOST_EXECUTOR_PUBLIC_CANONICAL_BUILDERS,
    *HOST_EXECUTOR_PUBLIC_PARSERS,
    *HOST_EXECUTOR_PUBLIC_VALIDATORS,
    "host_executor_v2_descriptor_sha256",
]
