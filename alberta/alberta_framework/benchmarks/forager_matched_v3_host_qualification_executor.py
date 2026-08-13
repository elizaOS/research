"""Pure, nonexecuting host-qualification contract for matched Forager v3.

This module deliberately exposes no Docker, OCI, subprocess, signal, cgroup
mutation, execution-capability, or workload API.  It freezes the metadata-only
schemas that a future separately authorized host executor would have to emit,
including a fresh-cgroup/retained-counter-FD proof, READY-before-GO handshake,
nonretryable lifecycle, and observation handoff.  Structural validity is never
execution readiness, qualification, evidence, publication authority, or
promotion authority.

The currently available qualification-plan descriptor v2 and observation
registry v1 are explicitly incompatible: the former is content-only and
incomplete, and the latter has structural validators but no observation issuer.
"""

from __future__ import annotations

import copy
import hashlib
import hmac
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Final, NoReturn, cast

HOST_QUALIFICATION_EXECUTOR_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.host_qualification_executor_descriptor.v1"
)
HOST_QUALIFICATION_CASE_REQUEST_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.host_qualification_case_request.v1"
)
HOST_QUALIFICATION_CASE_INTENT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.host_qualification_case_intent.v1"
)
HOST_CONTAINER_READY_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.in_container_qualification_driver_ready.v1"
)
HOST_QUALIFICATION_GO_COMMITMENT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.host_qualification_go_commitment.v1"
)
HOST_CGROUP_V2_BOUNDARY_PROOF_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.host_cgroup_v2_boundary_proof.v1"
)
HOST_CONTAINER_TERMINAL_METADATA_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.in_container_qualification_driver_terminal.v1"
)
HOST_QUALIFICATION_LIFECYCLE_RECORD_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.host_qualification_lifecycle_record.v1"
)
HOST_QUALIFICATION_CASE_RECEIPT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.host_qualification_case_execution_receipt.v1"
)
HOST_QUALIFICATION_CASE_FAILURE_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.host_qualification_case_failure.v1"
)
HOST_QUALIFICATION_OBSERVATION_HANDOFF_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.host_qualification_observation_handoff.v1"
)

QUALIFICATION_PLAN_V2_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.qualification_plan_descriptor.v2"
)
QUALIFICATION_OBSERVATION_REGISTRY_V1_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.qualification_observation_registry_descriptor.v1"
)
QUALIFICATION_RESOURCE_OBSERVATION_REQUEST_V1_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.qualification_resource_observation_request.v1"
)
QUALIFICATION_RESOURCE_OBSERVATION_RECEIPT_V1_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.qualification_resource_observation_receipt.v1"
)

HOST_QUALIFICATION_EXECUTION_ACKNOWLEDGEMENT: Final = (
    "AUTHORIZE ONE MATCHED-V3 HOST OCI QUALIFICATION CASE EXECUTION"
)
MATCHED_V3_HORIZON: Final = 499_712

MATCHED_V3_HOST_QUALIFICATION_CANDIDATE_IDS: Final = (
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

RESOURCE_CEILING_FIELDS: Final = (
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

PINNED_ENDPOINT_RESOURCE_OBSERVER_DESCRIPTOR_SHA256: Final = (
    "e424201576200d05f5da31822cb59a5a61ef06ee29ec267cb20727e8e2e6bfb7"
)
PINNED_ENDPOINT_RESOURCE_OBSERVER_SOURCE_SHA256: Final = (
    "4d34951ccb4b265caa29794457cdd8a5dd837ecf4b73b7a44e4f849bf8c8106e"
)

STALE_IMAGE_IDS: Final = (
    "sha256:a1f491fc786a788b2629e0670ee52ad84138057e58dd795703a830ea2e42c269",
    "sha256:5ecaabefce6439a8731c19e7a55fedb666788242baf035e6ffca86eb31299768",
)
STALE_BUILD_LINEAGES: Final = (
    {
        "classification": "pre_v3_source_closure_drift",
        "context_receipt_sha256": (
            "ccacc85f9adf6d81368050be37c67cbd38bb2423cc147deea580a152acf2b330"
        ),
        "execution_receipt_sha256": (
            "38cab52b6d247bf045405bd9de9d63b36f00d4e2f79bbb7a154d663ee24b8e9d"
        ),
        "publication_receipt_sha256": (
            "28892dd3be5c29df122a94a4feb35045fd17f95475e5e7237c0a04b4b15cbd88"
        ),
        "image_id": STALE_IMAGE_IDS[0],
    },
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
CGROUP_COUNTER_SEMANTICS: Final[Mapping[str, str]] = {
    "cpu.stat": "fresh_cgroup_cpu_usage_usec_cumulative_retained_fd",
    "memory.current": "fresh_cgroup_hierarchical_current_bytes_retained_fd",
    "memory.peak": "fresh_cgroup_since_creation_retained_fd_no_reopen",
    "memory.events": "fresh_cgroup_hierarchical_oom_event_counters_retained_fd",
    "pids.current": "fresh_cgroup_hierarchical_current_task_count_retained_fd",
    "pids.peak": "fresh_cgroup_required_read_only_never_resettable",
    "pids.events": "fresh_cgroup_hierarchical_pid_limit_events_retained_fd",
    "cgroup.events": "recursive_population_state_retained_fd",
    "cgroup.stat": "descendant_and_dying_descendant_counts_retained_fd",
    "cgroup.kill": "write_one_kills_entire_subtree_concurrent_fork_safe",
}
CGROUP_SAMPLE_PHASES: Final = (
    "initial_empty",
    "driver_ready",
    "pre_cleanup",
    "post_kill_empty",
    "post_container_remove",
)

HOST_QUALIFICATION_LIFECYCLE_PHASES: Final = (
    "request_validated",
    "intent_committed",
    "fresh_cgroup_created",
    "container_created",
    "driver_ready",
    "observer_anchored",
    "go_committed",
    "case_started",
    "workload_exited",
    "publication_committed",
    "terminal_metadata_validated",
    "cgroup_empty",
    "container_absent",
    "postflight_revalidated",
    "receipt_committed",
    "handoff_committed",
)
HOST_QUALIFICATION_PRE_RECEIPT_PHASES: Final = HOST_QUALIFICATION_LIFECYCLE_PHASES[
    : HOST_QUALIFICATION_LIFECYCLE_PHASES.index("receipt_committed")
]
HOST_QUALIFICATION_PRE_HANDOFF_PHASES: Final = HOST_QUALIFICATION_LIFECYCLE_PHASES[
    : HOST_QUALIFICATION_LIFECYCLE_PHASES.index("handoff_committed")
]
HOST_QUALIFICATION_UNCERTAINTY_KINDS: Final = (
    "cleanup_state",
    "container_state",
    "observation_state",
    "publication_state",
    "receipt_state",
)

FORBIDDEN_METADATA_KEYS: Final = frozenset(
    {
        "acceptance",
        "accepted",
        "candidate_rank",
        "candidate_ranking",
        "cumulative_reward",
        "database_bytes",
        "mean_reward",
        "npz_bytes",
        "performance_score",
        "rank",
        "ranking",
        "raw_result",
        "raw_results",
        "raw_reward",
        "result_bytes",
        "reward_magnitude",
        "reward_sum",
        "reward_total",
        "reward_trace",
        "score",
        "scores",
        "total_reward",
        "trace_bytes",
        "video_bytes",
    }
)

_BASE_READINESS_BLOCKERS: Final = (
    "no_authorizing_plan_schema_registered",
    "no_compatible_observation_issuer_registered",
    "no_current_fresh_cpu_oci_build_publication",
    "historical_image_lineages_forbidden",
    "in_container_outcome_consumer_driver_not_production_bound",
    "resource_observer_v1_cannot_complete_28_field_observation",
    "fresh_cgroup_provisioning_not_implemented",
    "retained_counter_fd_observation_not_implemented",
    "docker_or_oci_mutation_not_implemented",
    "external_privileged_cgroup_migration_not_portably_excluded",
    "daemon_owned_container_init_not_directly_reaped",
    "host_stability_sampling_policy_not_frozen",
    "separate_observation_issuer_and_evaluator_not_implemented",
    "source_closure_drift_requires_new_build",
)

_AUTHORITY_FIELDS: Final = (
    "execution_authorized",
    "observation_issuance_authorized",
    "publication_authority_granted",
    "qualification_granted",
    "promotion_allowed",
    "scientific_evidence_created",
)
_CLAIM_FIELDS: Final = (
    "build_qualified",
    "executed_bytecode_attested",
    "performance_claim_allowed",
    "production_plan_issued",
    "publisher_registry_complete",
    "resource_matched",
    "runtime_qualified",
    "source_qualified",
    "universal_sota_claim_allowed",
)

_MAX_ARTIFACT_BYTES: Final = 8 * 1024 * 1024
_MAX_JSON_DEPTH: Final = 64
_MAX_JSON_NODES: Final = 150_000
_MAX_TEXT_LENGTH: Final = 16_384
_MAX_INTEGER: Final = 2**63 - 1
_MAX_PUBLICATION_FILES: Final = 128
_MAX_PUBLICATION_AGGREGATE_BYTES: Final = 1024 * 1024 * 1024
_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_IMAGE_ID_RE: Final = re.compile(r"sha256:[0-9a-f]{64}\Z")
_CONTAINER_ID_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_CONTAINER_NAME_RE: Final = re.compile(r"alberta-matched-v3-q-[0-9a-f]{32}\Z")
_IDENTIFIER_RE: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}\Z")
_SAFE_TYPE_RE: Final = re.compile(r"[A-Za-z_][A-Za-z0-9_.]{0,255}\Z")
_RELATIVE_PATH_RE: Final = re.compile(r"[A-Za-z0-9_.+-]+(?:/[A-Za-z0-9_.+-]+)*\Z")


class ForagerMatchedV3HostQualificationExecutorError(ValueError):
    """A pure host-qualification contract artifact failed closed."""


def _fail(message: str) -> NoReturn:
    raise ForagerMatchedV3HostQualificationExecutorError(message)


def _require_int(
    value: object,
    label: str,
    *,
    minimum: int = 0,
    maximum: int = _MAX_INTEGER,
) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        _fail(f"{label} must be one exact integer in range")
    return value


def _require_bool(value: object, label: str) -> bool:
    if type(value) is not bool:
        _fail(f"{label} must be one exact boolean")
    return value


def _require_text(value: object, label: str, *, maximum: int = _MAX_TEXT_LENGTH) -> str:
    if type(value) is not str or not value or len(value) > maximum:
        _fail(f"{label} must be one bounded nonempty string")
    try:
        value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ForagerMatchedV3HostQualificationExecutorError(
            f"{label} must be ASCII"
        ) from exc
    if "\x00" in value:
        _fail(f"{label} contains NUL")
    return value


def _require_identifier(value: object, label: str) -> str:
    exact = _require_text(value, label, maximum=256)
    if _IDENTIFIER_RE.fullmatch(exact) is None:
        _fail(f"{label} is not one portable identifier")
    return exact


def _require_sha256(value: object, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        _fail(f"{label} must be one lowercase SHA-256")
    return value


def _require_optional_sha256(value: object, label: str) -> str | None:
    if value is None:
        return None
    return _require_sha256(value, label)


def _require_image_id(value: object, label: str) -> str:
    if type(value) is not str or _IMAGE_ID_RE.fullmatch(value) is None:
        _fail(f"{label} must be one exact sha256 image ID")
    return value


def _require_relative_path(value: object, label: str) -> str:
    exact = _require_text(value, label, maximum=4096)
    path = PurePosixPath(exact)
    if (
        _RELATIVE_PATH_RE.fullmatch(exact) is None
        or path.is_absolute()
        or not path.parts
        or any(component in {"", ".", ".."} for component in path.parts)
        or path.as_posix() != exact
    ):
        _fail(f"{label} is not one canonical relative path")
    return exact


def _require_cgroup_path(value: object, label: str) -> str:
    exact = _require_text(value, label, maximum=4096)
    path = PurePosixPath(exact)
    if (
        not path.is_absolute()
        or len(path.parts) < 5
        or path.parts[:3] != ("/", "sys", "fs")
        or path.parts[3] != "cgroup"
        or any(component in {"", ".", ".."} for component in path.parts[1:])
        or path.as_posix() != exact
    ):
        _fail(f"{label} must be one canonical child below /sys/fs/cgroup")
    return exact


def _require_proc_cgroup_path(value: object, label: str) -> str:
    exact = _require_text(value, label, maximum=4096)
    path = PurePosixPath(exact)
    if (
        not path.is_absolute()
        or len(path.parts) < 2
        or path.parts[0] != "/"
        or any(component in {"", ".", ".."} for component in path.parts[1:])
        or path.as_posix() != exact
    ):
        _fail(f"{label} must be one canonical absolute cgroup-membership path")
    return exact


def _require_candidate(candidate_id: object, ordinal: object) -> tuple[str, int]:
    exact_ordinal = _require_int(
        ordinal,
        "case ordinal",
        maximum=len(MATCHED_V3_HOST_QUALIFICATION_CANDIDATE_IDS) - 1,
    )
    exact_candidate = _require_identifier(candidate_id, "candidate ID")
    if exact_candidate != MATCHED_V3_HOST_QUALIFICATION_CANDIDATE_IDS[exact_ordinal]:
        _fail("candidate ID is cross-wired from the literal 28-case order")
    return exact_candidate, exact_ordinal


def _authority() -> dict[str, bool]:
    return {field: False for field in _AUTHORITY_FIELDS}


def _claims() -> dict[str, bool]:
    return {field: False for field in _CLAIM_FIELDS}


def _reject_constant(value: str) -> NoReturn:
    _fail(f"host-qualification JSON contains non-finite constant {value!r}")


def _reject_float(value: str) -> NoReturn:
    _fail(f"host-qualification JSON contains forbidden float {value!r}")


def _parse_int(value: str) -> int:
    if len(value.lstrip("-")) > 19:
        _fail("host-qualification JSON integer exceeds its lexical bound")
    return int(value)


def _without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"host-qualification JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _assert_plain_unaliased_json(value: object) -> None:
    seen: set[int] = set()
    nodes = 0

    def visit(item: object, depth: int) -> None:
        nonlocal nodes
        nodes += 1
        if nodes > _MAX_JSON_NODES:
            _fail("host-qualification JSON exceeds its node bound")
        if depth > _MAX_JSON_DEPTH:
            _fail("host-qualification JSON exceeds its depth bound")
        if item is None or type(item) in {bool, int}:
            if type(item) is int:
                _require_int(item, "host-qualification JSON integer", minimum=-_MAX_INTEGER)
            return
        if type(item) is str:
            _require_text(item, "host-qualification JSON string")
            return
        if type(item) not in {dict, list}:
            _fail("host-qualification JSON contains an inexact or non-JSON value")
        identity = id(item)
        if identity in seen:
            _fail("host-qualification JSON containers must be unaliased")
        seen.add(identity)
        if type(item) is list:
            for child in cast(list[object], item):
                visit(child, depth + 1)
            return
        for key, child in cast(dict[object, object], item).items():
            if type(key) is not str:
                _fail("host-qualification JSON object key must be an exact string")
            _require_text(key, "host-qualification JSON object key")
            visit(child, depth + 1)

    visit(value, 0)


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
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise ForagerMatchedV3HostQualificationExecutorError(
            "host-qualification JSON cannot be canonically encoded"
        ) from exc
    if newline:
        raw += b"\n"
    if not 1 <= len(raw) <= _MAX_ARTIFACT_BYTES:
        _fail("host-qualification artifact exceeds its byte bound")
    return raw


def _strict_json(raw: bytes) -> dict[str, Any]:
    if type(raw) is not bytes or not 1 <= len(raw) <= _MAX_ARTIFACT_BYTES:
        _fail("host-qualification artifact bytes are absent or oversized")
    try:
        value = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_without_duplicates,
            parse_constant=_reject_constant,
            parse_float=_reject_float,
            parse_int=_parse_int,
        )
    except ForagerMatchedV3HostQualificationExecutorError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ForagerMatchedV3HostQualificationExecutorError(
            "host-qualification artifact is not strict ASCII JSON"
        ) from exc
    if type(value) is not dict:
        _fail("host-qualification artifact root must be one object")
    result = cast(dict[str, Any], value)
    _assert_plain_unaliased_json(result)
    if raw != _canonical_json(result):
        _fail("host-qualification artifact bytes are not exact canonical JSON")
    return result


def _exact(value: object, fields: frozenset[str], label: str) -> dict[str, Any]:
    if type(value) is not dict or frozenset(cast(dict[str, Any], value)) != fields:
        _fail(f"{label} fields are not exact")
    return dict(cast(dict[str, Any], value))


def _exact_json_equal(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        left_map = cast(dict[str, object], left)
        right_map = cast(dict[str, object], right)
        return set(left_map) == set(right_map) and all(
            _exact_json_equal(left_map[key], right_map[key]) for key in left_map
        )
    if type(left) is list:
        left_items = cast(list[object], left)
        right_items = cast(list[object], right)
        return len(left_items) == len(right_items) and all(
            _exact_json_equal(a, b) for a, b in zip(left_items, right_items, strict=True)
        )
    return bool(left == right)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _with_body_sha256(body: Mapping[str, Any], field: str) -> dict[str, Any]:
    exact = copy.deepcopy(dict(body))
    exact[field] = _sha256(_canonical_json(body, newline=False))
    return exact


def _validate_body_sha256(value: Mapping[str, Any], field: str, label: str) -> None:
    body = copy.deepcopy(dict(value))
    supplied = _require_sha256(body.pop(field, None), f"{label} body")
    expected = _sha256(_canonical_json(body, newline=False))
    if not hmac.compare_digest(supplied, expected):
        _fail(f"{label} body SHA-256 differs")


def validate_matched_v3_host_metadata_only_mapping(value: Mapping[str, Any]) -> None:
    """Reject performance content keys recursively without decoding any values."""

    if type(value) is not dict:
        _fail("metadata-only payload must be one exact mapping")
    _assert_plain_unaliased_json(value)
    pending: list[object] = [value]
    while pending:
        current = pending.pop()
        if type(current) is list:
            pending.extend(cast(list[object], current))
            continue
        if type(current) is not dict:
            continue
        for key, child in cast(dict[str, Any], current).items():
            if key in FORBIDDEN_METADATA_KEYS:
                _fail(f"metadata-only payload contains forbidden field {key!r}")
            pending.append(child)


def _validate_resource_ceilings(
    value: object,
) -> tuple[tuple[str, int], ...]:
    if (
        type(value) is not tuple
        or any(
            type(item) is not tuple or len(item) != 2
            for item in cast(tuple[object, ...], value)
        )
    ):
        _fail("resource ceilings must be one exact tuple of pairs")
    exact = cast(tuple[tuple[str, int], ...], value)
    if tuple(name for name, _number in exact) != RESOURCE_CEILING_FIELDS:
        _fail("resource ceilings must use the exact 28-field order")
    values: dict[str, int] = {}
    for name, number in exact:
        values[name] = _require_int(number, f"resource ceiling {name}")
    if values["max_environment_interactions"] != MATCHED_V3_HORIZON:
        _fail("environment-interaction ceiling must equal the exact matched-v3 horizon")
    if values["max_thread_count"] < 1:
        _fail("thread ceiling must be positive")
    if values["max_attempt_count"] != 1 or values["max_failure_count"] != 0:
        _fail("resource ceilings must encode one attempt and zero retryable failures")
    return exact


def _validate_image_lineage(
    *,
    image_id: str,
    context_receipt_sha256: str | None = None,
    execution_receipt_sha256: str | None = None,
    publication_receipt_sha256: str | None = None,
) -> None:
    exact_image = _require_image_id(image_id, "CPU OCI image")
    if exact_image in STALE_IMAGE_IDS:
        _fail("historical or stale CPU OCI image is permanently forbidden")
    supplied = {
        "context_receipt_sha256": context_receipt_sha256,
        "execution_receipt_sha256": execution_receipt_sha256,
        "publication_receipt_sha256": publication_receipt_sha256,
    }
    for lineage in STALE_BUILD_LINEAGES:
        if any(
            supplied[field] is not None and supplied[field] == lineage[field]
            for field in supplied
        ):
            _fail("stale pre-v3 source-closure build lineage is permanently forbidden")


def matched_v3_host_qualification_readiness(
    request: MatchedV3HostQualificationCaseRequest | None = None,
) -> dict[str, Any]:
    """Return blockers only; this contract can never return execution readiness."""

    blockers: list[str] = list(_BASE_READINESS_BLOCKERS)
    if (
        request is None
        or request.qualification_plan_schema_version
        == QUALIFICATION_PLAN_V2_SCHEMA_VERSION
    ):
        blockers.insert(0, "qualification_plan_v2_is_content_only_and_incomplete")
    if (
        request is None
        or request.observation_registry_schema_version
        == QUALIFICATION_OBSERVATION_REGISTRY_V1_SCHEMA_VERSION
    ):
        blockers.insert(1, "qualification_observation_registry_v1_has_no_issuer")
    if request is None or request.plan_issuance_receipt_sha256 is None:
        blockers.append("no_plan_issuance_receipt")
    if request is None or request.case_execution_ticket_sha256 is None:
        blockers.append("no_case_execution_ticket")
    if request is None or request.runtime_qualification_receipt_sha256 is None:
        blockers.append("no_runtime_qualification_receipt")
    if request is None or request.host_provisioning_receipt_sha256 is None:
        blockers.append("no_host_provisioning_receipt")
    if request is None or request.full_resource_merger_descriptor_sha256 is None:
        blockers.append("no_full_28_field_resource_merger")
    if request is None or not request.publisher_registry_complete:
        blockers.append("publisher_registry_28_of_28_not_bound")
    return {
        "execution_ready": False,
        "production_mutation_permitted": False,
        "qualification_evaluated": False,
        "same_case_retry_authorized": False,
        "blockers": blockers,
    }


@dataclass(frozen=True, slots=True)
class MatchedV3HostQualificationCaseRequest:
    """Caller-carried content pins; never an execution request accepted for mutation."""

    qualification_plan_schema_version: str
    qualification_plan_sha256: str
    qualification_plan_body_sha256: str
    plan_issuance_receipt_sha256: str | None
    observation_registry_schema_version: str
    observation_registry_descriptor_sha256: str
    case_execution_ticket_sha256: str | None
    case_ordinal: int
    candidate_id: str
    qualification_case_id: str
    qualification_case_manifest_sha256: str
    resource_requirement_body_sha256: str
    declared_ceilings: tuple[tuple[str, int], ...]
    horizon: int
    attempt_ordinal: int
    external_source_tree_sha256: str
    local_source_tree_sha256: str
    build_context_receipt_sha256: str
    build_execution_receipt_sha256: str
    build_publication_receipt_sha256: str
    image_id: str
    runtime_identity_sha256: str
    runtime_profile_sha256: str
    runtime_qualification_receipt_sha256: str | None
    resource_observer_descriptor_sha256: str
    resource_observer_source_sha256: str
    full_resource_merger_descriptor_sha256: str | None
    publisher_registry_sha256: str
    publisher_registry_complete: bool
    publisher_descriptor_sha256: str
    publisher_source_sha256: str
    in_container_driver_descriptor_sha256: str
    in_container_driver_source_sha256: str
    host_provisioning_receipt_sha256: str | None
    exact_acknowledgement: str

    def __post_init__(self) -> None:
        _require_text(self.qualification_plan_schema_version, "qualification plan schema")
        _require_sha256(self.qualification_plan_sha256, "qualification plan")
        _require_sha256(self.qualification_plan_body_sha256, "qualification plan body")
        _require_optional_sha256(self.plan_issuance_receipt_sha256, "plan issuance receipt")
        _require_text(self.observation_registry_schema_version, "observation registry schema")
        _require_sha256(
            self.observation_registry_descriptor_sha256,
            "observation registry descriptor",
        )
        _require_optional_sha256(self.case_execution_ticket_sha256, "case execution ticket")
        _require_candidate(self.candidate_id, self.case_ordinal)
        _require_identifier(self.qualification_case_id, "qualification case ID")
        _require_sha256(self.qualification_case_manifest_sha256, "qualification case manifest")
        _require_sha256(self.resource_requirement_body_sha256, "resource requirement body")
        _validate_resource_ceilings(self.declared_ceilings)
        if self.horizon != MATCHED_V3_HORIZON:
            _fail("host qualification horizon must be exact 499712")
        if self.attempt_ordinal != 0:
            _fail("host qualification contract permits only attempt ordinal zero")
        for value, label in (
            (self.external_source_tree_sha256, "external source tree"),
            (self.local_source_tree_sha256, "local source tree"),
            (self.build_context_receipt_sha256, "build context receipt"),
            (self.build_execution_receipt_sha256, "build execution receipt"),
            (self.build_publication_receipt_sha256, "build publication receipt"),
            (self.runtime_identity_sha256, "runtime identity"),
            (self.runtime_profile_sha256, "runtime profile"),
            (self.publisher_registry_sha256, "publisher registry"),
            (self.publisher_descriptor_sha256, "publisher descriptor"),
            (self.publisher_source_sha256, "publisher source"),
            (self.in_container_driver_descriptor_sha256, "in-container driver descriptor"),
            (self.in_container_driver_source_sha256, "in-container driver source"),
        ):
            _require_sha256(value, label)
        if self.external_source_tree_sha256 == self.local_source_tree_sha256:
            _fail("external and local source trees must remain distinct")
        if len(
            {
                self.build_context_receipt_sha256,
                self.build_execution_receipt_sha256,
                self.build_publication_receipt_sha256,
            }
        ) != 3:
            _fail("build context, execution, and publication receipts must be distinct")
        _validate_image_lineage(
            image_id=self.image_id,
            context_receipt_sha256=self.build_context_receipt_sha256,
            execution_receipt_sha256=self.build_execution_receipt_sha256,
            publication_receipt_sha256=self.build_publication_receipt_sha256,
        )
        _require_optional_sha256(
            self.runtime_qualification_receipt_sha256,
            "runtime qualification receipt",
        )
        if (
            self.resource_observer_descriptor_sha256
            != PINNED_ENDPOINT_RESOURCE_OBSERVER_DESCRIPTOR_SHA256
            or self.resource_observer_source_sha256
            != PINNED_ENDPOINT_RESOURCE_OBSERVER_SOURCE_SHA256
        ):
            _fail("endpoint resource observer descriptor or source identity differs")
        _require_optional_sha256(
            self.full_resource_merger_descriptor_sha256,
            "full resource merger descriptor",
        )
        _require_bool(self.publisher_registry_complete, "publisher registry completeness")
        _require_optional_sha256(
            self.host_provisioning_receipt_sha256,
            "host provisioning receipt",
        )
        if (
            type(self.exact_acknowledgement) is not str
            or not hmac.compare_digest(
                self.exact_acknowledgement,
                HOST_QUALIFICATION_EXECUTION_ACKNOWLEDGEMENT,
            )
        ):
            _fail("exact host-qualification execution acknowledgement differs")

    def to_dict(self) -> dict[str, Any]:
        body = {
            "schema_version": HOST_QUALIFICATION_CASE_REQUEST_SCHEMA_VERSION,
            "status": "content_request_validated_nonexecuting",
            **{
                field: getattr(self, field)
                for field in (
                    "qualification_plan_schema_version",
                    "qualification_plan_sha256",
                    "qualification_plan_body_sha256",
                    "plan_issuance_receipt_sha256",
                    "observation_registry_schema_version",
                    "observation_registry_descriptor_sha256",
                    "case_execution_ticket_sha256",
                    "case_ordinal",
                    "candidate_id",
                    "qualification_case_id",
                    "qualification_case_manifest_sha256",
                    "resource_requirement_body_sha256",
                    "horizon",
                    "attempt_ordinal",
                    "external_source_tree_sha256",
                    "local_source_tree_sha256",
                    "build_context_receipt_sha256",
                    "build_execution_receipt_sha256",
                    "build_publication_receipt_sha256",
                    "image_id",
                    "runtime_identity_sha256",
                    "runtime_profile_sha256",
                    "runtime_qualification_receipt_sha256",
                    "resource_observer_descriptor_sha256",
                    "resource_observer_source_sha256",
                    "full_resource_merger_descriptor_sha256",
                    "publisher_registry_sha256",
                    "publisher_registry_complete",
                    "publisher_descriptor_sha256",
                    "publisher_source_sha256",
                    "in_container_driver_descriptor_sha256",
                    "in_container_driver_source_sha256",
                    "host_provisioning_receipt_sha256",
                    "exact_acknowledgement",
                )
            },
            "declared_ceilings": dict(self.declared_ceilings),
            "readiness": matched_v3_host_qualification_readiness(self),
            "authority": _authority(),
            "claims": _claims(),
        }
        return _with_body_sha256(body, "request_body_sha256")


def canonical_matched_v3_host_qualification_case_request_bytes(
    request: MatchedV3HostQualificationCaseRequest,
) -> bytes:
    if type(request) is not MatchedV3HostQualificationCaseRequest:
        raise TypeError("request must use the exact host-qualification request type")
    return _canonical_json(request.to_dict())


def _request_from_dict(value: Mapping[str, Any]) -> MatchedV3HostQualificationCaseRequest:
    fields = set(MatchedV3HostQualificationCaseRequest.__dataclass_fields__)
    item = _exact(
        value,
        frozenset(
            {
                "schema_version",
                "status",
                "declared_ceilings",
                "readiness",
                "authority",
                "claims",
                "request_body_sha256",
                *fields,
            }
        ),
        "host qualification request",
    )
    if item.pop("schema_version") != HOST_QUALIFICATION_CASE_REQUEST_SCHEMA_VERSION:
        _fail("host qualification request schema differs")
    if item.pop("status") != "content_request_validated_nonexecuting":
        _fail("host qualification request status differs")
    readiness = item.pop("readiness")
    authority = item.pop("authority")
    claims = item.pop("claims")
    item.pop("request_body_sha256")
    ceilings = _exact(
        item.pop("declared_ceilings"),
        frozenset(RESOURCE_CEILING_FIELDS),
        "request resource ceilings",
    )
    request = MatchedV3HostQualificationCaseRequest(
        **item,
        declared_ceilings=tuple((field, ceilings[field]) for field in RESOURCE_CEILING_FIELDS),
    )
    if not _exact_json_equal(readiness, matched_v3_host_qualification_readiness(request)):
        _fail("host qualification request readiness projection differs")
    if not _exact_json_equal(authority, _authority()) or not _exact_json_equal(claims, _claims()):
        _fail("host qualification request authority or claims differ")
    return request


def parse_matched_v3_host_qualification_case_request(
    raw: bytes,
) -> MatchedV3HostQualificationCaseRequest:
    value = _strict_json(raw)
    _validate_body_sha256(value, "request_body_sha256", "host qualification request")
    request = _request_from_dict(value)
    if raw != canonical_matched_v3_host_qualification_case_request_bytes(request):
        _fail("host qualification request replay differs")
    return request


def replay_matched_v3_host_qualification_case_request(
    raw: bytes,
    *,
    expected_sha256: str,
) -> MatchedV3HostQualificationCaseRequest:
    expected = _require_sha256(expected_sha256, "expected request")
    if not hmac.compare_digest(_sha256(raw), expected):
        _fail("host qualification request file SHA-256 differs")
    return parse_matched_v3_host_qualification_case_request(raw)


@dataclass(frozen=True, slots=True)
class MatchedV3HostQualificationCaseIntent:
    """Deterministic intent content; no route is committed by this module."""

    request_sha256: str
    qualification_plan_sha256: str
    case_ordinal: int
    candidate_id: str
    qualification_case_id: str
    qualification_case_manifest_sha256: str
    build_context_receipt_sha256: str
    build_execution_receipt_sha256: str
    build_publication_receipt_sha256: str
    image_id: str
    runtime_identity_sha256: str
    resource_requirement_body_sha256: str
    publisher_descriptor_sha256: str
    in_container_driver_descriptor_sha256: str
    horizon: int
    attempt_ordinal: int
    readiness_blockers: tuple[str, ...]

    def __post_init__(self) -> None:
        for value, label in (
            (self.request_sha256, "intent request"),
            (self.qualification_plan_sha256, "intent plan"),
            (self.qualification_case_manifest_sha256, "intent case manifest"),
            (self.build_context_receipt_sha256, "intent build context"),
            (self.build_execution_receipt_sha256, "intent build execution"),
            (self.build_publication_receipt_sha256, "intent build publication"),
            (self.runtime_identity_sha256, "intent runtime identity"),
            (self.resource_requirement_body_sha256, "intent resource requirement"),
            (self.publisher_descriptor_sha256, "intent publisher descriptor"),
            (self.in_container_driver_descriptor_sha256, "intent driver descriptor"),
        ):
            _require_sha256(value, label)
        _require_candidate(self.candidate_id, self.case_ordinal)
        _require_identifier(self.qualification_case_id, "intent qualification case")
        _validate_image_lineage(
            image_id=self.image_id,
            context_receipt_sha256=self.build_context_receipt_sha256,
            execution_receipt_sha256=self.build_execution_receipt_sha256,
            publication_receipt_sha256=self.build_publication_receipt_sha256,
        )
        if self.horizon != MATCHED_V3_HORIZON or self.attempt_ordinal != 0:
            _fail("intent horizon or attempt ordinal differs")
        if (
            type(self.readiness_blockers) is not tuple
            or not self.readiness_blockers
            or any(type(item) is not str or not item for item in self.readiness_blockers)
            or len(set(self.readiness_blockers)) != len(self.readiness_blockers)
        ):
            _fail("intent readiness blockers differ")

    def to_dict(self) -> dict[str, Any]:
        body = {
            "schema_version": HOST_QUALIFICATION_CASE_INTENT_SCHEMA_VERSION,
            "status": "deterministic_intent_content_only_not_committed",
            **{
                field: getattr(self, field)
                for field in self.__dataclass_fields__
                if field != "readiness_blockers"
            },
            "readiness_blockers": list(self.readiness_blockers),
            "policy": {
                "intent_route_committed": False,
                "production_mutation_permitted": False,
                "same_case_retry_permitted": False,
            },
            "authority": _authority(),
            "claims": _claims(),
        }
        return _with_body_sha256(body, "intent_body_sha256")


def build_matched_v3_host_qualification_case_intent(
    request: MatchedV3HostQualificationCaseRequest,
) -> MatchedV3HostQualificationCaseIntent:
    if type(request) is not MatchedV3HostQualificationCaseRequest:
        raise TypeError("request must use the exact host-qualification request type")
    readiness = matched_v3_host_qualification_readiness(request)
    return MatchedV3HostQualificationCaseIntent(
        request_sha256=_sha256(canonical_matched_v3_host_qualification_case_request_bytes(request)),
        qualification_plan_sha256=request.qualification_plan_sha256,
        case_ordinal=request.case_ordinal,
        candidate_id=request.candidate_id,
        qualification_case_id=request.qualification_case_id,
        qualification_case_manifest_sha256=request.qualification_case_manifest_sha256,
        build_context_receipt_sha256=request.build_context_receipt_sha256,
        build_execution_receipt_sha256=request.build_execution_receipt_sha256,
        build_publication_receipt_sha256=request.build_publication_receipt_sha256,
        image_id=request.image_id,
        runtime_identity_sha256=request.runtime_identity_sha256,
        resource_requirement_body_sha256=request.resource_requirement_body_sha256,
        publisher_descriptor_sha256=request.publisher_descriptor_sha256,
        in_container_driver_descriptor_sha256=request.in_container_driver_descriptor_sha256,
        horizon=request.horizon,
        attempt_ordinal=request.attempt_ordinal,
        readiness_blockers=tuple(cast(list[str], readiness["blockers"])),
    )


def canonical_matched_v3_host_qualification_case_intent_bytes(
    intent: MatchedV3HostQualificationCaseIntent,
) -> bytes:
    if type(intent) is not MatchedV3HostQualificationCaseIntent:
        raise TypeError("intent must use the exact host-qualification intent type")
    return _canonical_json(intent.to_dict())


def parse_matched_v3_host_qualification_case_intent(
    raw: bytes,
) -> MatchedV3HostQualificationCaseIntent:
    value = _strict_json(raw)
    validate_matched_v3_host_metadata_only_mapping(value)
    _validate_body_sha256(value, "intent_body_sha256", "host qualification intent")
    fields = set(MatchedV3HostQualificationCaseIntent.__dataclass_fields__)
    item = _exact(
        value,
        frozenset(
            {
                "schema_version",
                "status",
                "policy",
                "authority",
                "claims",
                "intent_body_sha256",
                *fields,
            }
        ),
        "host qualification intent",
    )
    if (
        item.pop("schema_version") != HOST_QUALIFICATION_CASE_INTENT_SCHEMA_VERSION
        or item.pop("status") != "deterministic_intent_content_only_not_committed"
    ):
        _fail("host qualification intent identity differs")
    if item.pop("policy") != {
        "intent_route_committed": False,
        "production_mutation_permitted": False,
        "same_case_retry_permitted": False,
    }:
        _fail("host qualification intent policy differs")
    if item.pop("authority") != _authority() or item.pop("claims") != _claims():
        _fail("host qualification intent authority or claims differ")
    item.pop("intent_body_sha256")
    blockers = item.pop("readiness_blockers")
    if type(blockers) is not list:
        _fail("intent readiness blockers must be one list")
    intent = MatchedV3HostQualificationCaseIntent(
        **item,
        readiness_blockers=tuple(blockers),
    )
    if raw != canonical_matched_v3_host_qualification_case_intent_bytes(intent):
        _fail("host qualification intent replay differs")
    return intent


@dataclass(frozen=True, slots=True)
class MatchedV3HostContainerReadyMetadata:
    """Pre-GO driver metadata binding PID/start-time/cgroup and content identities."""

    qualification_plan_sha256: str
    intent_sha256: str
    case_ordinal: int
    candidate_id: str
    qualification_case_id: str
    qualification_case_manifest_sha256: str
    image_id: str
    container_id: str
    container_name: str
    host_pid: int
    host_process_start_time_ticks: int
    inner_pid: int
    proc_cgroup_path: str
    container_cgroup_device: int
    container_cgroup_inode: int
    driver_descriptor_sha256: str
    driver_source_sha256: str
    runtime_identity_sha256: str
    runtime_profile_sha256: str
    sandbox_observation_sha256: str
    ready_cgroup_sample_sha256: str
    expected_go_payload_sha256: str
    ready_monotonic_ns: int
    stdout_frame_ordinal: int
    candidate_code_loaded: bool
    outcome_capability_issued: bool
    go_committed: bool

    def __post_init__(self) -> None:
        for value, label in (
            (self.qualification_plan_sha256, "READY qualification plan"),
            (self.intent_sha256, "READY intent"),
            (self.qualification_case_manifest_sha256, "READY case manifest"),
            (self.driver_descriptor_sha256, "READY driver descriptor"),
            (self.driver_source_sha256, "READY driver source"),
            (self.runtime_identity_sha256, "READY runtime identity"),
            (self.runtime_profile_sha256, "READY runtime profile"),
            (self.sandbox_observation_sha256, "READY sandbox observation"),
            (self.ready_cgroup_sample_sha256, "READY cgroup sample"),
            (self.expected_go_payload_sha256, "READY expected GO payload"),
        ):
            _require_sha256(value, label)
        _require_candidate(self.candidate_id, self.case_ordinal)
        _require_identifier(self.qualification_case_id, "READY qualification case")
        _validate_image_lineage(image_id=self.image_id)
        if _CONTAINER_ID_RE.fullmatch(self.container_id) is None:
            _fail("READY container ID differs")
        if _CONTAINER_NAME_RE.fullmatch(self.container_name) is None:
            _fail("READY container name differs")
        _require_int(self.host_pid, "READY host PID", minimum=1)
        _require_int(
            self.host_process_start_time_ticks,
            "READY host process start time",
            minimum=1,
        )
        if self.inner_pid != 1:
            _fail("READY inner PID must be exact namespace init PID 1")
        _require_proc_cgroup_path(self.proc_cgroup_path, "READY proc cgroup path")
        _require_int(self.container_cgroup_device, "READY cgroup device")
        _require_int(self.container_cgroup_inode, "READY cgroup inode", minimum=1)
        _require_int(self.ready_monotonic_ns, "READY monotonic time")
        if self.stdout_frame_ordinal != 0:
            _fail("READY must be stdout metadata frame zero")
        for flag, flag_label in (
            (self.candidate_code_loaded, "READY candidate code loaded"),
            (self.outcome_capability_issued, "READY outcome capability issued"),
            (self.go_committed, "READY GO committed"),
        ):
            if _require_bool(flag, flag_label) is not False:
                _fail(f"{flag_label} must be false before GO")

    def to_dict(self) -> dict[str, Any]:
        body = {
            "schema_version": HOST_CONTAINER_READY_SCHEMA_VERSION,
            "status": "driver_ready_waiting_for_one_way_go_non_authorizing",
            **{field: getattr(self, field) for field in self.__dataclass_fields__},
            "metadata_policy": {
                "raw_content_transported": False,
                "score_or_reward_decoded": False,
            },
            "authority": _authority(),
            "claims": _claims(),
        }
        return _with_body_sha256(body, "ready_metadata_body_sha256")


def canonical_matched_v3_host_container_ready_metadata_bytes(
    ready: MatchedV3HostContainerReadyMetadata,
) -> bytes:
    if type(ready) is not MatchedV3HostContainerReadyMetadata:
        raise TypeError("READY metadata must use the exact READY type")
    return _canonical_json(ready.to_dict())


def parse_matched_v3_host_container_ready_metadata(
    raw: bytes,
) -> MatchedV3HostContainerReadyMetadata:
    value = _strict_json(raw)
    validate_matched_v3_host_metadata_only_mapping(value)
    _validate_body_sha256(value, "ready_metadata_body_sha256", "READY metadata")
    item = _exact(
        value,
        frozenset(
            {
                "schema_version",
                "status",
                "metadata_policy",
                "authority",
                "claims",
                "ready_metadata_body_sha256",
                *MatchedV3HostContainerReadyMetadata.__dataclass_fields__,
            }
        ),
        "READY metadata",
    )
    if (
        item.pop("schema_version") != HOST_CONTAINER_READY_SCHEMA_VERSION
        or item.pop("status") != "driver_ready_waiting_for_one_way_go_non_authorizing"
    ):
        _fail("READY metadata identity differs")
    if item.pop("metadata_policy") != {
        "raw_content_transported": False,
        "score_or_reward_decoded": False,
    }:
        _fail("READY metadata content policy differs")
    if item.pop("authority") != _authority() or item.pop("claims") != _claims():
        _fail("READY metadata authority or claims differ")
    item.pop("ready_metadata_body_sha256")
    ready = MatchedV3HostContainerReadyMetadata(**item)
    if raw != canonical_matched_v3_host_container_ready_metadata_bytes(ready):
        _fail("READY metadata replay differs")
    return ready


@dataclass(frozen=True, slots=True)
class MatchedV3HostQualificationGoCommitment:
    """Structural record of the one-way GO boundary; this module never commits it."""

    qualification_plan_sha256: str
    intent_sha256: str
    ready_metadata_sha256: str
    go_payload_sha256: str
    case_ordinal: int
    candidate_id: str
    qualification_case_id: str
    qualification_case_manifest_sha256: str
    host_pid: int
    host_process_start_time_ticks: int
    container_cgroup_device: int
    container_cgroup_inode: int
    go_commitment_monotonic_ns: int

    def __post_init__(self) -> None:
        for value, label in (
            (self.qualification_plan_sha256, "GO qualification plan"),
            (self.intent_sha256, "GO intent"),
            (self.ready_metadata_sha256, "GO READY metadata"),
            (self.go_payload_sha256, "GO payload"),
            (self.qualification_case_manifest_sha256, "GO case manifest"),
        ):
            _require_sha256(value, label)
        _require_candidate(self.candidate_id, self.case_ordinal)
        _require_identifier(self.qualification_case_id, "GO qualification case")
        _require_int(self.host_pid, "GO host PID", minimum=1)
        _require_int(self.host_process_start_time_ticks, "GO process start time", minimum=1)
        _require_int(self.container_cgroup_device, "GO cgroup device")
        _require_int(self.container_cgroup_inode, "GO cgroup inode", minimum=1)
        _require_int(self.go_commitment_monotonic_ns, "GO commitment time")

    def to_dict(self) -> dict[str, Any]:
        body = {
            "schema_version": HOST_QUALIFICATION_GO_COMMITMENT_SCHEMA_VERSION,
            "status": "one_way_go_boundary_record_non_authorizing",
            **{field: getattr(self, field) for field in self.__dataclass_fields__},
            "policy": {
                "one_way_commitment": True,
                "same_case_retry_permitted": False,
                "publication_state_failure_after_go_is_uncertain": True,
            },
            "authority": _authority(),
            "claims": _claims(),
        }
        return _with_body_sha256(body, "go_commitment_body_sha256")


def build_matched_v3_host_qualification_go_commitment(
    ready: MatchedV3HostContainerReadyMetadata,
    *,
    go_commitment_monotonic_ns: int,
) -> MatchedV3HostQualificationGoCommitment:
    if type(ready) is not MatchedV3HostContainerReadyMetadata:
        raise TypeError("READY metadata must use the exact READY type")
    exact_time = _require_int(go_commitment_monotonic_ns, "GO commitment time")
    if exact_time <= ready.ready_monotonic_ns:
        _fail("GO commitment must occur strictly after READY")
    return MatchedV3HostQualificationGoCommitment(
        qualification_plan_sha256=ready.qualification_plan_sha256,
        intent_sha256=ready.intent_sha256,
        ready_metadata_sha256=_sha256(
            canonical_matched_v3_host_container_ready_metadata_bytes(ready)
        ),
        go_payload_sha256=ready.expected_go_payload_sha256,
        case_ordinal=ready.case_ordinal,
        candidate_id=ready.candidate_id,
        qualification_case_id=ready.qualification_case_id,
        qualification_case_manifest_sha256=ready.qualification_case_manifest_sha256,
        host_pid=ready.host_pid,
        host_process_start_time_ticks=ready.host_process_start_time_ticks,
        container_cgroup_device=ready.container_cgroup_device,
        container_cgroup_inode=ready.container_cgroup_inode,
        go_commitment_monotonic_ns=exact_time,
    )


def canonical_matched_v3_host_qualification_go_commitment_bytes(
    commitment: MatchedV3HostQualificationGoCommitment,
) -> bytes:
    if type(commitment) is not MatchedV3HostQualificationGoCommitment:
        raise TypeError("GO commitment must use the exact GO type")
    return _canonical_json(commitment.to_dict())


def parse_matched_v3_host_qualification_go_commitment(
    raw: bytes,
) -> MatchedV3HostQualificationGoCommitment:
    value = _strict_json(raw)
    validate_matched_v3_host_metadata_only_mapping(value)
    _validate_body_sha256(value, "go_commitment_body_sha256", "GO commitment")
    item = _exact(
        value,
        frozenset(
            {
                "schema_version",
                "status",
                "policy",
                "authority",
                "claims",
                "go_commitment_body_sha256",
                *MatchedV3HostQualificationGoCommitment.__dataclass_fields__,
            }
        ),
        "GO commitment",
    )
    if (
        item.pop("schema_version") != HOST_QUALIFICATION_GO_COMMITMENT_SCHEMA_VERSION
        or item.pop("status") != "one_way_go_boundary_record_non_authorizing"
    ):
        _fail("GO commitment identity differs")
    if item.pop("policy") != {
        "one_way_commitment": True,
        "same_case_retry_permitted": False,
        "publication_state_failure_after_go_is_uncertain": True,
    }:
        _fail("GO commitment policy differs")
    if item.pop("authority") != _authority() or item.pop("claims") != _claims():
        _fail("GO commitment authority or claims differ")
    item.pop("go_commitment_body_sha256")
    commitment = MatchedV3HostQualificationGoCommitment(**item)
    if raw != canonical_matched_v3_host_qualification_go_commitment_bytes(commitment):
        _fail("GO commitment replay differs")
    return commitment


@dataclass(frozen=True, slots=True)
class HostCgroupV2CounterFdIdentity:
    """Identity and lifecycle of one retained cgroup-v2 control-file descriptor."""

    endpoint_name: str
    endpoint_device: int
    endpoint_inode: int
    open_monotonic_ns: int
    open_flags: tuple[str, ...]
    counter_semantics: str
    reset_performed: bool
    retained_through_final_sample: bool
    reopened: bool

    def __post_init__(self) -> None:
        if self.endpoint_name not in CGROUP_COUNTER_ENDPOINTS:
            _fail("cgroup counter endpoint is outside the frozen allowlist")
        _require_int(self.endpoint_device, "cgroup counter device")
        _require_int(self.endpoint_inode, "cgroup counter inode", minimum=1)
        _require_int(self.open_monotonic_ns, "cgroup counter open time")
        expected_flags = (
            "O_CLOEXEC",
            "O_NOFOLLOW",
            "O_WRONLY" if self.endpoint_name == "cgroup.kill" else "O_RDONLY",
        )
        if self.open_flags != expected_flags:
            _fail("cgroup counter open flags differ")
        if self.counter_semantics != CGROUP_COUNTER_SEMANTICS[self.endpoint_name]:
            _fail("cgroup counter semantics differ")
        if _require_bool(self.reset_performed, "cgroup counter reset") is not False:
            _fail("fresh-cgroup proof forbids post-creation counter reset claims")
        if (
            _require_bool(
                self.retained_through_final_sample,
                "cgroup counter retained-through-final-sample",
            )
            is not True
        ):
            _fail("cgroup counter descriptor must be retained through the final sample")
        if _require_bool(self.reopened, "cgroup counter reopened") is not False:
            _fail("cgroup counter descriptor cannot be reopened")

    def to_dict(self) -> dict[str, Any]:
        return {
            "endpoint_name": self.endpoint_name,
            "endpoint_device": self.endpoint_device,
            "endpoint_inode": self.endpoint_inode,
            "open_monotonic_ns": self.open_monotonic_ns,
            "open_flags": list(self.open_flags),
            "counter_semantics": self.counter_semantics,
            "reset_performed": self.reset_performed,
            "retained_through_final_sample": self.retained_through_final_sample,
            "reopened": self.reopened,
        }


@dataclass(frozen=True, slots=True)
class HostCgroupV2DescendantIdentity:
    """One exact descendant cgroup under the fresh outer case cgroup."""

    relative_path: str
    device: int
    inode: int

    def __post_init__(self) -> None:
        _require_relative_path(self.relative_path, "descendant cgroup path")
        _require_int(self.device, "descendant cgroup device")
        _require_int(self.inode, "descendant cgroup inode", minimum=1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "relative_path": self.relative_path,
            "device": self.device,
            "inode": self.inode,
        }


@dataclass(frozen=True, slots=True)
class HostCgroupV2ProcessIdentity:
    """One PID/start-time identity observed recursively inside the case subtree."""

    pid: int
    start_time_ticks: int
    cgroup_device: int
    cgroup_inode: int

    def __post_init__(self) -> None:
        _require_int(self.pid, "cgroup process PID", minimum=1)
        _require_int(self.start_time_ticks, "cgroup process start time", minimum=1)
        _require_int(self.cgroup_device, "process cgroup device")
        _require_int(self.cgroup_inode, "process cgroup inode", minimum=1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pid": self.pid,
            "start_time_ticks": self.start_time_ticks,
            "cgroup_device": self.cgroup_device,
            "cgroup_inode": self.cgroup_inode,
        }


def _validate_descendants(
    descendants: object,
) -> tuple[HostCgroupV2DescendantIdentity, ...]:
    if (
        type(descendants) is not tuple
        or any(type(item) is not HostCgroupV2DescendantIdentity for item in descendants)
    ):
        _fail("cgroup descendants must use one exact tuple of identity records")
    exact = cast(tuple[HostCgroupV2DescendantIdentity, ...], descendants)
    paths = tuple(item.relative_path for item in exact)
    if paths != tuple(sorted(set(paths), key=str.encode)):
        _fail("cgroup descendant paths must be sorted and unique")
    return exact


def _validate_processes(
    processes: object,
) -> tuple[HostCgroupV2ProcessIdentity, ...]:
    if (
        type(processes) is not tuple
        or any(type(item) is not HostCgroupV2ProcessIdentity for item in processes)
    ):
        _fail("cgroup processes must use one exact tuple of identity records")
    exact = cast(tuple[HostCgroupV2ProcessIdentity, ...], processes)
    pids = tuple(item.pid for item in exact)
    if pids != tuple(sorted(set(pids))):
        _fail("cgroup process PIDs must be sorted and unique")
    return exact


@dataclass(frozen=True, slots=True)
class HostCgroupV2Sample:
    """One non-atomic but exact ordered cgroup/proc metadata sample."""

    phase: str
    monotonic_ns: int
    cgroup_device: int
    cgroup_inode: int
    cpu_usage_usec: int
    memory_current_bytes: int
    memory_peak_bytes: int
    memory_oom_kill_count: int
    pids_current: int
    pids_peak: int
    pids_max_event_count: int
    populated: bool
    frozen: bool
    nr_descendants: int
    nr_dying_descendants: int
    descendant_cgroups: tuple[HostCgroupV2DescendantIdentity, ...]
    recursive_processes: tuple[HostCgroupV2ProcessIdentity, ...]
    recursive_thread_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.phase not in CGROUP_SAMPLE_PHASES:
            _fail("cgroup sample phase differs")
        for value, label in (
            (self.monotonic_ns, "cgroup sample monotonic time"),
            (self.cgroup_device, "cgroup sample device"),
            (self.cpu_usage_usec, "cgroup CPU usage"),
            (self.memory_current_bytes, "cgroup memory current"),
            (self.memory_peak_bytes, "cgroup memory peak"),
            (self.memory_oom_kill_count, "cgroup OOM-kill count"),
            (self.pids_current, "cgroup pids current"),
            (self.pids_peak, "cgroup pids peak"),
            (self.pids_max_event_count, "cgroup pids max event count"),
            (self.nr_descendants, "cgroup descendant count"),
            (self.nr_dying_descendants, "cgroup dying descendant count"),
        ):
            _require_int(value, label)
        _require_int(self.cgroup_inode, "cgroup sample inode", minimum=1)
        _require_bool(self.populated, "cgroup sample populated")
        _require_bool(self.frozen, "cgroup sample frozen")
        descendants = _validate_descendants(self.descendant_cgroups)
        processes = _validate_processes(self.recursive_processes)
        if (
            type(self.recursive_thread_ids) is not tuple
            or any(type(item) is not int for item in self.recursive_thread_ids)
            or self.recursive_thread_ids != tuple(sorted(set(self.recursive_thread_ids)))
            or any(item < 1 for item in self.recursive_thread_ids)
        ):
            _fail("cgroup recursive thread IDs must be one sorted unique tuple")
        if self.nr_descendants != len(descendants):
            _fail("cgroup sample descendant count differs from its inventory")
        if self.memory_peak_bytes < self.memory_current_bytes:
            _fail("cgroup memory peak cannot be below current memory")
        if self.pids_peak < self.pids_current:
            _fail("cgroup pids peak cannot be below current pids")
        if self.pids_current != len(self.recursive_thread_ids):
            _fail("cgroup current pids differs from its recursive thread inventory")
        if not self.populated and (processes or self.recursive_thread_ids or self.pids_current):
            _fail("unpopulated cgroup sample cannot contain live tasks")
        if self.populated and not processes:
            _fail("populated cgroup sample must inventory at least one process")
        descendant_identities = {(item.device, item.inode) for item in descendants}
        if any(
            (process.cgroup_device, process.cgroup_inode) not in descendant_identities
            for process in processes
        ):
            _fail("cgroup process inventory escapes its descendant inventory")
        if any(process.pid not in self.recursive_thread_ids for process in processes):
            _fail("cgroup process leaders must appear in the recursive thread inventory")

    def to_dict(self) -> dict[str, Any]:
        return {
            **{
                field: getattr(self, field)
                for field in self.__dataclass_fields__
                if field
                not in {"descendant_cgroups", "recursive_processes", "recursive_thread_ids"}
            },
            "descendant_cgroups": [item.to_dict() for item in self.descendant_cgroups],
            "recursive_processes": [item.to_dict() for item in self.recursive_processes],
            "recursive_thread_ids": list(self.recursive_thread_ids),
        }


def matched_v3_host_cgroup_sample_sha256(sample: HostCgroupV2Sample) -> str:
    """Return the exact standalone identity used by READY for one cgroup sample."""

    if type(sample) is not HostCgroupV2Sample:
        raise TypeError("cgroup sample must use the exact sample type")
    return _sha256(_canonical_json(sample.to_dict()))


def _validate_cgroup_sample_sequence(samples: object) -> tuple[HostCgroupV2Sample, ...]:
    if (
        type(samples) is not tuple
        or any(type(item) is not HostCgroupV2Sample for item in samples)
    ):
        _fail("cgroup proof samples must use one exact tuple")
    exact = cast(tuple[HostCgroupV2Sample, ...], samples)
    if tuple(sample.phase for sample in exact) != CGROUP_SAMPLE_PHASES:
        _fail("cgroup proof sample coverage or order differs")
    if any(
        current.monotonic_ns >= following.monotonic_ns
        for current, following in zip(exact, exact[1:], strict=False)
    ):
        _fail("cgroup sample monotonic times must strictly increase")
    if any(sample.frozen for sample in exact):
        _fail("cgroup samples cannot claim a freezer operation absent from this contract")
    if any(
        sample.nr_descendants + sample.nr_dying_descendants > 1 for sample in exact
    ):
        _fail("cgroup sample exceeds cgroup.max.descendants=1")
    identity = (exact[0].cgroup_device, exact[0].cgroup_inode)
    if any((sample.cgroup_device, sample.cgroup_inode) != identity for sample in exact):
        _fail("case cgroup directory identity drifted between samples")
    for current, following in zip(exact, exact[1:], strict=False):
        if (
            following.cpu_usage_usec < current.cpu_usage_usec
            or following.memory_peak_bytes < current.memory_peak_bytes
            or following.memory_oom_kill_count < current.memory_oom_kill_count
            or following.pids_peak < current.pids_peak
            or following.pids_max_event_count < current.pids_max_event_count
        ):
            _fail("fresh cgroup cumulative counter rolled back")
    initial = exact[0]
    if (
        initial.populated
        or initial.frozen
        or initial.descendant_cgroups
        or initial.recursive_processes
        or initial.recursive_thread_ids
        or any(
            value != 0
            for value in (
                initial.cpu_usage_usec,
                initial.memory_current_bytes,
                initial.memory_peak_bytes,
                initial.memory_oom_kill_count,
                initial.pids_current,
                initial.pids_peak,
                initial.pids_max_event_count,
                initial.nr_descendants,
                initial.nr_dying_descendants,
            )
        )
    ):
        _fail("initial cgroup sample is not a newly created exact empty boundary")
    ready = exact[1]
    if (
        not ready.populated
        or ready.frozen
        or ready.nr_descendants != 1
        or ready.nr_dying_descendants != 0
        or len(ready.descendant_cgroups) != 1
        or len(ready.recursive_processes) != 1
    ):
        _fail("READY cgroup sample is not one exact singleton container boundary")
    if exact[2].descendant_cgroups != ready.descendant_cgroups:
        _fail("pre-cleanup container cgroup identity differs from READY")
    post_kill = exact[3]
    if (
        post_kill.populated
        or post_kill.recursive_processes
        or post_kill.recursive_thread_ids
        or post_kill.pids_current != 0
        or post_kill.nr_descendants != 1
        or post_kill.nr_dying_descendants != 0
    ):
        _fail("post-kill cgroup sample is not recursively empty")
    if post_kill.descendant_cgroups != ready.descendant_cgroups:
        _fail("post-kill container cgroup identity differs from READY")
    final = exact[4]
    if (
        final.populated
        or final.recursive_processes
        or final.recursive_thread_ids
        or final.pids_current != 0
        or final.descendant_cgroups
        or final.nr_descendants != 0
        or final.nr_dying_descendants != 0
    ):
        _fail("post-container-removal cgroup sample is not exact empty-without-descendants")
    return exact


@dataclass(frozen=True, slots=True)
class MatchedV3HostCgroupV2BoundaryProof:
    """Structural fresh-cgroup proof; no cgroup endpoints are touched here."""

    qualification_plan_sha256: str
    intent_sha256: str
    case_ordinal: int
    candidate_id: str
    qualification_case_id: str
    qualification_case_manifest_sha256: str
    delegate_root_path: str
    delegate_root_device: int
    delegate_root_inode: int
    case_cgroup_path: str
    case_cgroup_device: int
    case_cgroup_inode: int
    docker_cgroup_parent_argument: str
    container_id: str
    container_name: str
    container_cgroup_relative_path: str
    container_cgroup_device: int
    container_cgroup_inode: int
    target_pid: int
    target_process_start_time_ticks: int
    pidfd_opened: bool
    proc_cgroup_path: str
    controllers: tuple[str, ...]
    subtree_control: tuple[str, ...]
    cgroup_max_depth: int
    cgroup_max_descendants: int
    counter_fds: tuple[HostCgroupV2CounterFdIdentity, ...]
    samples: tuple[HostCgroupV2Sample, ...]
    cgroup_namespace_private: bool
    pid_namespace_private: bool
    writable_cgroup_mount_observed: bool
    setsid_changes_cgroup: bool
    fork_clone_inherit_cgroup: bool
    cgroup_kill_supported: bool
    cgroup_kill_written: bool
    direct_cli_child_waited: bool
    direct_cli_child_reaped: bool
    daemon_owned_container_init_directly_reaped: bool
    container_absent_after_cleanup: bool
    case_cgroup_path_absent_after_cleanup: bool
    continuous_membership_proven: bool
    external_privileged_migration_excluded: bool

    def __post_init__(self) -> None:
        for value, label in (
            (self.qualification_plan_sha256, "cgroup proof plan"),
            (self.intent_sha256, "cgroup proof intent"),
            (self.qualification_case_manifest_sha256, "cgroup proof case manifest"),
        ):
            _require_sha256(value, label)
        _require_candidate(self.candidate_id, self.case_ordinal)
        _require_identifier(self.qualification_case_id, "cgroup proof qualification case")
        delegate = _require_cgroup_path(self.delegate_root_path, "delegated cgroup root")
        case = _require_cgroup_path(self.case_cgroup_path, "case cgroup path")
        if not case.startswith(delegate + "/"):
            _fail("case cgroup path is not a strict child of its delegated root")
        _require_int(self.delegate_root_device, "delegated cgroup device")
        _require_int(self.delegate_root_inode, "delegated cgroup inode", minimum=1)
        _require_int(self.case_cgroup_device, "case cgroup device")
        _require_int(self.case_cgroup_inode, "case cgroup inode", minimum=1)
        cgroup_parent = _require_relative_path(
            self.docker_cgroup_parent_argument,
            "Docker cgroup-parent argument",
        )
        if cgroup_parent != case.removeprefix("/sys/fs/cgroup/"):
            _fail("Docker cgroup-parent argument differs from the fresh case cgroup")
        if _CONTAINER_ID_RE.fullmatch(self.container_id) is None:
            _fail("cgroup proof container ID differs")
        if _CONTAINER_NAME_RE.fullmatch(self.container_name) is None:
            _fail("cgroup proof container name differs")
        container_relative = _require_relative_path(
            self.container_cgroup_relative_path,
            "container cgroup relative path",
        )
        if "/" in container_relative:
            _fail("container cgroup must be one direct child under cgroup.max.depth=1")
        _require_int(self.container_cgroup_device, "container cgroup device")
        _require_int(self.container_cgroup_inode, "container cgroup inode", minimum=1)
        if not (
            self.delegate_root_device
            == self.case_cgroup_device
            == self.container_cgroup_device
        ):
            _fail("delegate, case, and container cgroups must share one cgroupfs device")
        if len(
            {
                self.delegate_root_inode,
                self.case_cgroup_inode,
                self.container_cgroup_inode,
            }
        ) != 3:
            _fail("delegate, case, and container cgroup directory inodes must be distinct")
        _require_int(self.target_pid, "cgroup proof target PID", minimum=1)
        _require_int(
            self.target_process_start_time_ticks,
            "cgroup proof target process start time",
            minimum=1,
        )
        if not _require_bool(self.pidfd_opened, "cgroup proof pidfd opened"):
            _fail("cgroup proof requires a retained pidfd identity observation")
        proc_path = _require_proc_cgroup_path(
            self.proc_cgroup_path,
            "cgroup proof proc path",
        )
        expected_proc_path = (
            "/"
            + case.removeprefix("/sys/fs/cgroup/")
            + "/"
            + self.container_cgroup_relative_path
        )
        if proc_path != expected_proc_path:
            _fail("target proc cgroup path is not the exact fresh-case container descendant")
        if self.controllers != ("cpu", "memory", "pids"):
            _fail("fresh case cgroup controllers differ")
        if self.subtree_control != self.controllers:
            _fail("fresh case cgroup subtree control differs")
        if self.cgroup_max_depth != 1 or self.cgroup_max_descendants != 1:
            _fail("fresh case cgroup depth/descendant caps must both be exact one")
        if (
            type(self.counter_fds) is not tuple
            or any(type(item) is not HostCgroupV2CounterFdIdentity for item in self.counter_fds)
            or tuple(item.endpoint_name for item in self.counter_fds) != CGROUP_COUNTER_ENDPOINTS
        ):
            _fail("retained cgroup counter-FD coverage or order differs")
        if (
            any(item.endpoint_device != self.case_cgroup_device for item in self.counter_fds)
            or len({item.endpoint_inode for item in self.counter_fds}) != len(self.counter_fds)
            or any(
                item.endpoint_inode
                in {
                    self.delegate_root_inode,
                    self.case_cgroup_inode,
                    self.container_cgroup_inode,
                }
                for item in self.counter_fds
            )
        ):
            _fail("retained cgroup counter descriptors do not bind the fresh case cgroup")
        samples = _validate_cgroup_sample_sequence(self.samples)
        if any(
            item.open_monotonic_ns >= samples[0].monotonic_ns
            for item in self.counter_fds
        ):
            _fail("cgroup counter descriptors must be retained before the initial sample")
        if (samples[0].cgroup_device, samples[0].cgroup_inode) != (
            self.case_cgroup_device,
            self.case_cgroup_inode,
        ):
            _fail("case cgroup identity differs from its retained samples")
        ready = samples[1]
        descendant = ready.descendant_cgroups[0]
        process = ready.recursive_processes[0]
        if (
            descendant.relative_path != self.container_cgroup_relative_path
            or (descendant.device, descendant.inode)
            != (self.container_cgroup_device, self.container_cgroup_inode)
        ):
            _fail("READY descendant cgroup differs from the exact container boundary")
        if (
            process.pid != self.target_pid
            or process.start_time_ticks != self.target_process_start_time_ticks
            or (process.cgroup_device, process.cgroup_inode)
            != (self.container_cgroup_device, self.container_cgroup_inode)
        ):
            _fail("READY target PID/start-time/cgroup identity differs")
        expected_true = (
            (self.cgroup_namespace_private, "private cgroup namespace"),
            (self.pid_namespace_private, "private PID namespace"),
            (self.fork_clone_inherit_cgroup, "fork/clone cgroup inheritance"),
            (self.cgroup_kill_supported, "cgroup.kill support"),
            (self.cgroup_kill_written, "cgroup.kill write"),
            (self.direct_cli_child_waited, "direct CLI child wait"),
            (self.direct_cli_child_reaped, "direct CLI child reap"),
            (self.container_absent_after_cleanup, "container absence"),
            (self.case_cgroup_path_absent_after_cleanup, "case cgroup path absence"),
        )
        for flag, flag_label in expected_true:
            if _require_bool(flag, flag_label) is not True:
                _fail(f"{flag_label} must be exact true")
        expected_false = (
            (self.writable_cgroup_mount_observed, "writable cgroup mount observed"),
            (self.setsid_changes_cgroup, "setsid changes cgroup"),
            (
                self.daemon_owned_container_init_directly_reaped,
                "daemon-owned init directly reaped",
            ),
            (self.continuous_membership_proven, "continuous membership proven"),
            (
                self.external_privileged_migration_excluded,
                "external privileged migration excluded",
            ),
        )
        for flag, flag_label in expected_false:
            if _require_bool(flag, flag_label) is not False:
                _fail(f"{flag_label} is a nonportable claim and must remain false")

    def to_dict(self) -> dict[str, Any]:
        body = {
            "schema_version": HOST_CGROUP_V2_BOUNDARY_PROOF_SCHEMA_VERSION,
            "status": "structural_fresh_cgroup_boundary_record_non_authorizing",
            **{
                field: getattr(self, field)
                for field in self.__dataclass_fields__
                if field not in {"counter_fds", "samples", "controllers", "subtree_control"}
            },
            "controllers": list(self.controllers),
            "subtree_control": list(self.subtree_control),
            "counter_fds": [item.to_dict() for item in self.counter_fds],
            "samples": [item.to_dict() for item in self.samples],
            "portable_limitations": {
                "control_file_sample_atomic": False,
                "continuous_membership_proven": False,
                "daemon_owned_init_directly_reaped": False,
                "external_privileged_migration_excluded": False,
            },
            "authority": _authority(),
            "claims": _claims(),
        }
        return _with_body_sha256(body, "cgroup_proof_body_sha256")


def _counter_fd_from_dict(value: object) -> HostCgroupV2CounterFdIdentity:
    item = _exact(
        value,
        frozenset(HostCgroupV2CounterFdIdentity.__dataclass_fields__),
        "cgroup counter-FD identity",
    )
    flags = item.pop("open_flags")
    if type(flags) is not list:
        _fail("cgroup counter open flags must be one list")
    return HostCgroupV2CounterFdIdentity(**item, open_flags=tuple(flags))


def _descendant_from_dict(value: object) -> HostCgroupV2DescendantIdentity:
    item = _exact(
        value,
        frozenset(HostCgroupV2DescendantIdentity.__dataclass_fields__),
        "cgroup descendant identity",
    )
    return HostCgroupV2DescendantIdentity(**item)


def _process_from_dict(value: object) -> HostCgroupV2ProcessIdentity:
    item = _exact(
        value,
        frozenset(HostCgroupV2ProcessIdentity.__dataclass_fields__),
        "cgroup process identity",
    )
    return HostCgroupV2ProcessIdentity(**item)


def _sample_from_dict(value: object) -> HostCgroupV2Sample:
    item = _exact(
        value,
        frozenset(HostCgroupV2Sample.__dataclass_fields__),
        "cgroup sample",
    )
    descendants = item.pop("descendant_cgroups")
    processes = item.pop("recursive_processes")
    threads = item.pop("recursive_thread_ids")
    if type(descendants) is not list or type(processes) is not list or type(threads) is not list:
        _fail("cgroup sample inventories must be exact lists")
    return HostCgroupV2Sample(
        **item,
        descendant_cgroups=tuple(_descendant_from_dict(child) for child in descendants),
        recursive_processes=tuple(_process_from_dict(child) for child in processes),
        recursive_thread_ids=tuple(threads),
    )


def canonical_matched_v3_host_cgroup_v2_boundary_proof_bytes(
    proof: MatchedV3HostCgroupV2BoundaryProof,
) -> bytes:
    if type(proof) is not MatchedV3HostCgroupV2BoundaryProof:
        raise TypeError("cgroup proof must use the exact boundary-proof type")
    return _canonical_json(proof.to_dict())


def parse_matched_v3_host_cgroup_v2_boundary_proof(
    raw: bytes,
) -> MatchedV3HostCgroupV2BoundaryProof:
    value = _strict_json(raw)
    validate_matched_v3_host_metadata_only_mapping(value)
    _validate_body_sha256(value, "cgroup_proof_body_sha256", "cgroup boundary proof")
    item = _exact(
        value,
        frozenset(
            {
                "schema_version",
                "status",
                "portable_limitations",
                "authority",
                "claims",
                "cgroup_proof_body_sha256",
                *MatchedV3HostCgroupV2BoundaryProof.__dataclass_fields__,
            }
        ),
        "cgroup boundary proof",
    )
    if (
        item.pop("schema_version") != HOST_CGROUP_V2_BOUNDARY_PROOF_SCHEMA_VERSION
        or item.pop("status")
        != "structural_fresh_cgroup_boundary_record_non_authorizing"
    ):
        _fail("cgroup boundary proof identity differs")
    if item.pop("portable_limitations") != {
        "control_file_sample_atomic": False,
        "continuous_membership_proven": False,
        "daemon_owned_init_directly_reaped": False,
        "external_privileged_migration_excluded": False,
    }:
        _fail("cgroup boundary portable limitations differ")
    if item.pop("authority") != _authority() or item.pop("claims") != _claims():
        _fail("cgroup boundary proof authority or claims differ")
    item.pop("cgroup_proof_body_sha256")
    controllers = item.pop("controllers")
    subtree_control = item.pop("subtree_control")
    counters = item.pop("counter_fds")
    samples = item.pop("samples")
    if (
        type(controllers) is not list
        or type(subtree_control) is not list
        or type(counters) is not list
        or type(samples) is not list
    ):
        _fail("cgroup boundary list fields differ")
    proof = MatchedV3HostCgroupV2BoundaryProof(
        **item,
        controllers=tuple(controllers),
        subtree_control=tuple(subtree_control),
        counter_fds=tuple(_counter_fd_from_dict(child) for child in counters),
        samples=tuple(_sample_from_dict(child) for child in samples),
    )
    if raw != canonical_matched_v3_host_cgroup_v2_boundary_proof_bytes(proof):
        _fail("cgroup boundary proof replay differs")
    return proof


@dataclass(frozen=True, slots=True)
class MatchedV3HostPublishedFileMetadata:
    """One name/size/digest record; file content never enters this contract."""

    role: str
    name: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        _require_identifier(self.role, "published file role")
        _require_relative_path(self.name, "published file name")
        _require_int(self.size_bytes, "published file size")
        _require_sha256(self.sha256, "published file")

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "name": self.name,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


def _validate_file_records(
    files: object,
) -> tuple[MatchedV3HostPublishedFileMetadata, ...]:
    if (
        type(files) is not tuple
        or not files
        or len(files) > _MAX_PUBLICATION_FILES
        or any(type(item) is not MatchedV3HostPublishedFileMetadata for item in files)
    ):
        _fail("published file metadata must use one bounded exact nonempty tuple")
    exact = cast(tuple[MatchedV3HostPublishedFileMetadata, ...], files)
    names = tuple(item.name for item in exact)
    roles = tuple(item.role for item in exact)
    if len(set(names)) != len(names) or len(set(roles)) != len(roles):
        _fail("published file names and roles must each be unique")
    return exact


def matched_v3_host_published_file_inventory_sha256(
    files: tuple[MatchedV3HostPublishedFileMetadata, ...],
) -> str:
    exact = _validate_file_records(files)
    return _sha256(
        _canonical_json(
            {"files": [item.to_dict() for item in exact]},
            newline=False,
        )
    )


@dataclass(frozen=True, slots=True)
class MatchedV3HostContainerTerminalMetadata:
    """Common metadata-only terminal record after atomic publish and strict reload."""

    operation: str
    candidate_id: str
    case_ordinal: int
    qualification_case_id: str
    qualification_case_manifest_sha256: str
    qualification_plan_sha256: str
    interaction_horizon: int
    image_id: str
    driver_descriptor_sha256: str
    driver_source_sha256: str
    execution_receipt_sha256: str
    resource_metadata_sha256: str
    publisher_descriptor_sha256: str
    publisher_source_sha256: str
    publication_address_sha256: str
    publication_manifest_sha256: str
    publication_receipt_sha256: str
    published_bundle_sha256: str
    reload_observation_sha256: str
    file_inventory_sha256: str
    file_count: int
    total_size_bytes: int
    files: tuple[MatchedV3HostPublishedFileMetadata, ...]
    family_metadata_sha256: str
    publication_committed: bool
    raw_content_transported: bool
    score_or_reward_decoded: bool

    def __post_init__(self) -> None:
        if self.operation != "publish_and_strict_reload_metadata_only":
            _fail("terminal metadata operation differs")
        _require_candidate(self.candidate_id, self.case_ordinal)
        _require_identifier(self.qualification_case_id, "terminal qualification case")
        for value, label in (
            (self.qualification_case_manifest_sha256, "terminal case manifest"),
            (self.qualification_plan_sha256, "terminal qualification plan"),
            (self.driver_descriptor_sha256, "terminal driver descriptor"),
            (self.driver_source_sha256, "terminal driver source"),
            (self.execution_receipt_sha256, "terminal execution receipt"),
            (self.resource_metadata_sha256, "terminal resource metadata"),
            (self.publisher_descriptor_sha256, "terminal publisher descriptor"),
            (self.publisher_source_sha256, "terminal publisher source"),
            (self.publication_address_sha256, "terminal publication address"),
            (self.publication_manifest_sha256, "terminal publication manifest"),
            (self.publication_receipt_sha256, "terminal publication receipt"),
            (self.published_bundle_sha256, "terminal published bundle"),
            (self.reload_observation_sha256, "terminal reload observation"),
            (self.file_inventory_sha256, "terminal file inventory"),
            (self.family_metadata_sha256, "terminal family metadata"),
        ):
            _require_sha256(value, label)
        if self.interaction_horizon != MATCHED_V3_HORIZON:
            _fail("terminal interaction horizon must be exact 499712")
        _validate_image_lineage(image_id=self.image_id)
        files = _validate_file_records(self.files)
        _require_int(
            self.file_count,
            "terminal file count",
            minimum=1,
            maximum=_MAX_PUBLICATION_FILES,
        )
        if self.file_count != len(files):
            _fail("terminal file count differs from the exact metadata inventory")
        _require_int(
            self.total_size_bytes,
            "terminal aggregate size",
            maximum=_MAX_PUBLICATION_AGGREGATE_BYTES,
        )
        if self.total_size_bytes != sum(item.size_bytes for item in files):
            _fail("terminal total size differs from the exact metadata inventory")
        if self.file_inventory_sha256 != matched_v3_host_published_file_inventory_sha256(files):
            _fail("terminal file inventory SHA-256 differs")
        if _require_bool(self.publication_committed, "terminal publication committed") is not True:
            _fail("terminal metadata requires one committed publication")
        if (
            _require_bool(
                self.raw_content_transported,
                "terminal raw content transported",
            )
            is not False
        ):
            _fail("terminal metadata cannot transport raw content")
        if (
            _require_bool(
                self.score_or_reward_decoded,
                "terminal score/reward decoded",
            )
            is not False
        ):
            _fail("terminal metadata cannot decode score or reward")

    def to_dict(self) -> dict[str, Any]:
        body = {
            "schema_version": HOST_CONTAINER_TERMINAL_METADATA_SCHEMA_VERSION,
            "status": "publication_committed_strictly_reloaded_metadata_only",
            **{
                field: getattr(self, field)
                for field in self.__dataclass_fields__
                if field != "files"
            },
            "files": [item.to_dict() for item in self.files],
            "authority": _authority(),
            "claims": _claims(),
        }
        return _with_body_sha256(body, "terminal_metadata_body_sha256")


def _published_file_from_dict(value: object) -> MatchedV3HostPublishedFileMetadata:
    item = _exact(
        value,
        frozenset(MatchedV3HostPublishedFileMetadata.__dataclass_fields__),
        "published file metadata",
    )
    return MatchedV3HostPublishedFileMetadata(**item)


def canonical_matched_v3_host_container_terminal_metadata_bytes(
    terminal: MatchedV3HostContainerTerminalMetadata,
) -> bytes:
    if type(terminal) is not MatchedV3HostContainerTerminalMetadata:
        raise TypeError("terminal metadata must use the exact terminal type")
    return _canonical_json(terminal.to_dict())


def parse_matched_v3_host_container_terminal_metadata(
    raw: bytes,
) -> MatchedV3HostContainerTerminalMetadata:
    value = _strict_json(raw)
    validate_matched_v3_host_metadata_only_mapping(value)
    _validate_body_sha256(value, "terminal_metadata_body_sha256", "terminal metadata")
    item = _exact(
        value,
        frozenset(
            {
                "schema_version",
                "status",
                "authority",
                "claims",
                "terminal_metadata_body_sha256",
                *MatchedV3HostContainerTerminalMetadata.__dataclass_fields__,
            }
        ),
        "terminal metadata",
    )
    if (
        item.pop("schema_version") != HOST_CONTAINER_TERMINAL_METADATA_SCHEMA_VERSION
        or item.pop("status") != "publication_committed_strictly_reloaded_metadata_only"
    ):
        _fail("terminal metadata identity differs")
    if item.pop("authority") != _authority() or item.pop("claims") != _claims():
        _fail("terminal metadata authority or claims differ")
    item.pop("terminal_metadata_body_sha256")
    files = item.pop("files")
    if type(files) is not list:
        _fail("terminal file inventory must be one list")
    terminal = MatchedV3HostContainerTerminalMetadata(
        **item,
        files=tuple(_published_file_from_dict(child) for child in files),
    )
    if raw != canonical_matched_v3_host_container_terminal_metadata_bytes(terminal):
        _fail("terminal metadata replay differs")
    return terminal


@dataclass(frozen=True, slots=True)
class MatchedV3HostQualificationLifecycleRecord:
    """Pure phase-prefix classification with conservative uncertainty semantics."""

    completed_phases: tuple[str, ...]
    failure_phase: str | None
    uncertainty_kind: str | None
    terminal_state: str
    intent_committed: bool
    intent_commit_may_have_occurred: bool
    case_may_have_started: bool
    publication_may_be_visible: bool
    cleanup_proven: bool
    handoff_committed: bool
    same_case_retry_permitted: bool
    qualification_evaluated: bool

    def __post_init__(self) -> None:
        if (
            type(self.completed_phases) is not tuple
            or self.completed_phases
            != HOST_QUALIFICATION_LIFECYCLE_PHASES[: len(self.completed_phases)]
        ):
            _fail("lifecycle completed phases must be one exact ordered prefix")
        if self.failure_phase is not None:
            if (
                type(self.failure_phase) is not str
                or self.failure_phase not in HOST_QUALIFICATION_LIFECYCLE_PHASES
                or len(self.completed_phases) >= len(HOST_QUALIFICATION_LIFECYCLE_PHASES)
                or self.failure_phase
                != HOST_QUALIFICATION_LIFECYCLE_PHASES[len(self.completed_phases)]
            ):
                _fail("lifecycle failure phase must be the exact next uncompleted phase")
        if self.uncertainty_kind is not None and (
            type(self.uncertainty_kind) is not str
            or self.uncertainty_kind not in HOST_QUALIFICATION_UNCERTAINTY_KINDS
        ):
            _fail("lifecycle uncertainty kind differs")
        if self.failure_phase is None and self.uncertainty_kind is not None:
            _fail("lifecycle uncertainty requires one failure phase")
        if self.failure_phase is not None:
            failure_ordinal = HOST_QUALIFICATION_LIFECYCLE_PHASES.index(self.failure_phase)
            intent_ordinal = HOST_QUALIFICATION_LIFECYCLE_PHASES.index("intent_committed")
            if failure_ordinal >= intent_ordinal and self.uncertainty_kind is None:
                _fail("failure at or after intent commitment requires explicit uncertainty")
        expected = _lifecycle_projection(
            self.completed_phases,
            failure_phase=self.failure_phase,
            uncertainty_kind=self.uncertainty_kind,
        )
        for field, expected_value in expected.items():
            if getattr(self, field) != expected_value:
                _fail(f"lifecycle derived field {field!r} differs")

    def to_dict(self) -> dict[str, Any]:
        body = {
            "schema_version": HOST_QUALIFICATION_LIFECYCLE_RECORD_SCHEMA_VERSION,
            "status": "lifecycle_classified_without_execution_authority",
            **{
                field: getattr(self, field)
                for field in self.__dataclass_fields__
                if field != "completed_phases"
            },
            "completed_phases": list(self.completed_phases),
            "policy": {
                "intent_collision_requires_read_only_reconciliation": True,
                "post_go_observation_failure_is_not_clean_rejection": True,
                "post_go_publication_failure_is_uncertain": True,
                "same_case_retry_permitted": False,
            },
            "authority": _authority(),
            "claims": _claims(),
        }
        return _with_body_sha256(body, "lifecycle_body_sha256")


def _lifecycle_projection(
    completed_phases: tuple[str, ...],
    *,
    failure_phase: str | None,
    uncertainty_kind: str | None,
) -> dict[str, Any]:
    completed = set(completed_phases)
    intent_committed = "intent_committed" in completed
    failure_ordinal = (
        HOST_QUALIFICATION_LIFECYCLE_PHASES.index(failure_phase)
        if failure_phase is not None
        else None
    )
    intent_ordinal = HOST_QUALIFICATION_LIFECYCLE_PHASES.index("intent_committed")
    case_start_ordinal = HOST_QUALIFICATION_LIFECYCLE_PHASES.index("case_started")
    intent_commit_may_have_occurred = intent_committed or (
        failure_ordinal is not None and failure_ordinal >= intent_ordinal
    )
    case_started = "case_started" in completed or (
        failure_ordinal is not None and failure_ordinal >= case_start_ordinal
    )
    publication_visible = case_started or "publication_committed" in completed
    cleanup_proven = {"cgroup_empty", "container_absent"}.issubset(completed)
    handoff_committed = "handoff_committed" in completed
    if len(completed_phases) == len(HOST_QUALIFICATION_LIFECYCLE_PHASES):
        terminal_state = "metadata_handoff_recorded_non_authorizing"
    elif (
        failure_phase is None
        and completed_phases
        == HOST_QUALIFICATION_LIFECYCLE_PHASES[:
            HOST_QUALIFICATION_LIFECYCLE_PHASES.index("receipt_committed")
        ]
    ):
        terminal_state = "postflight_complete_ready_for_receipt_non_authorizing"
    elif (
        failure_phase is None
        and completed_phases
        == HOST_QUALIFICATION_LIFECYCLE_PHASES[:
            HOST_QUALIFICATION_LIFECYCLE_PHASES.index("handoff_committed")
        ]
    ):
        terminal_state = "receipt_committed_ready_for_handoff_non_authorizing"
    elif failure_ordinal == intent_ordinal and not intent_committed:
        terminal_state = "intent_commit_state_uncertain_non_retriable"
    elif failure_phase is not None and not intent_commit_may_have_occurred:
        terminal_state = "pre_intent_failure_no_execution_authority"
    elif failure_phase is not None and case_started:
        terminal_state = "case_state_uncertain_non_retriable"
    elif intent_committed:
        terminal_state = (
            "intent_consumed_no_workload_started_non_retriable"
            if failure_phase is not None
            else "in_progress_consumed_case_non_authorizing_non_retriable"
        )
    else:
        terminal_state = "incomplete_structural_record_non_authorizing"
    return {
        "terminal_state": terminal_state,
        "intent_committed": intent_committed,
        "intent_commit_may_have_occurred": intent_commit_may_have_occurred,
        "case_may_have_started": case_started,
        "publication_may_be_visible": publication_visible,
        "cleanup_proven": cleanup_proven,
        "handoff_committed": handoff_committed,
        "same_case_retry_permitted": False,
        "qualification_evaluated": False,
    }


def classify_matched_v3_host_qualification_lifecycle(
    *,
    completed_phases: Sequence[str],
    failure_phase: str | None,
    uncertainty_kind: str | None,
) -> MatchedV3HostQualificationLifecycleRecord:
    if type(completed_phases) not in {tuple, list} or any(
        type(item) is not str for item in completed_phases
    ):
        _fail("lifecycle completed phases must be one exact sequence of strings")
    phases = tuple(completed_phases)
    if phases != HOST_QUALIFICATION_LIFECYCLE_PHASES[: len(phases)]:
        _fail("lifecycle phases are skipped, reordered, or duplicated")
    if failure_phase is not None and (
        len(phases) >= len(HOST_QUALIFICATION_LIFECYCLE_PHASES)
        or failure_phase != HOST_QUALIFICATION_LIFECYCLE_PHASES[len(phases)]
    ):
        _fail("lifecycle failure phase must be the next exact phase")
    if failure_phase is None and uncertainty_kind is not None:
        _fail("lifecycle uncertainty requires one failure phase")
    if failure_phase is not None and (
        HOST_QUALIFICATION_LIFECYCLE_PHASES.index(failure_phase)
        >= HOST_QUALIFICATION_LIFECYCLE_PHASES.index("intent_committed")
        and uncertainty_kind is None
    ):
        _fail("failure at or after intent commitment requires explicit uncertainty")
    projection = _lifecycle_projection(
        phases,
        failure_phase=failure_phase,
        uncertainty_kind=uncertainty_kind,
    )
    return MatchedV3HostQualificationLifecycleRecord(
        completed_phases=phases,
        failure_phase=failure_phase,
        uncertainty_kind=uncertainty_kind,
        **projection,
    )


def canonical_matched_v3_host_qualification_lifecycle_record_bytes(
    record: MatchedV3HostQualificationLifecycleRecord,
) -> bytes:
    if type(record) is not MatchedV3HostQualificationLifecycleRecord:
        raise TypeError("lifecycle record must use the exact lifecycle type")
    return _canonical_json(record.to_dict())


def parse_matched_v3_host_qualification_lifecycle_record(
    raw: bytes,
) -> MatchedV3HostQualificationLifecycleRecord:
    value = _strict_json(raw)
    validate_matched_v3_host_metadata_only_mapping(value)
    _validate_body_sha256(value, "lifecycle_body_sha256", "lifecycle record")
    item = _exact(
        value,
        frozenset(
            {
                "schema_version",
                "status",
                "policy",
                "authority",
                "claims",
                "lifecycle_body_sha256",
                *MatchedV3HostQualificationLifecycleRecord.__dataclass_fields__,
            }
        ),
        "lifecycle record",
    )
    if (
        item.pop("schema_version") != HOST_QUALIFICATION_LIFECYCLE_RECORD_SCHEMA_VERSION
        or item.pop("status") != "lifecycle_classified_without_execution_authority"
    ):
        _fail("lifecycle record identity differs")
    if item.pop("policy") != {
        "intent_collision_requires_read_only_reconciliation": True,
        "post_go_observation_failure_is_not_clean_rejection": True,
        "post_go_publication_failure_is_uncertain": True,
        "same_case_retry_permitted": False,
    }:
        _fail("lifecycle policy differs")
    if item.pop("authority") != _authority() or item.pop("claims") != _claims():
        _fail("lifecycle authority or claims differ")
    item.pop("lifecycle_body_sha256")
    phases = item.pop("completed_phases")
    if type(phases) is not list:
        _fail("lifecycle completed phases must be one list")
    record = MatchedV3HostQualificationLifecycleRecord(
        **item,
        completed_phases=tuple(phases),
    )
    if raw != canonical_matched_v3_host_qualification_lifecycle_record_bytes(record):
        _fail("lifecycle record replay differs")
    return record


def _require_request_intent_chain(
    request: MatchedV3HostQualificationCaseRequest,
    intent: MatchedV3HostQualificationCaseIntent,
) -> tuple[str, str]:
    if type(request) is not MatchedV3HostQualificationCaseRequest:
        raise TypeError("request must use the exact host-qualification request type")
    if type(intent) is not MatchedV3HostQualificationCaseIntent:
        raise TypeError("intent must use the exact host-qualification intent type")
    expected_intent = build_matched_v3_host_qualification_case_intent(request)
    if intent != expected_intent:
        _fail("intent is not the exact deterministic projection of its request")
    return (
        _sha256(canonical_matched_v3_host_qualification_case_request_bytes(request)),
        _sha256(canonical_matched_v3_host_qualification_case_intent_bytes(intent)),
    )


def _require_ready_chain(
    request: MatchedV3HostQualificationCaseRequest,
    intent_sha256: str,
    ready: MatchedV3HostContainerReadyMetadata,
) -> str:
    if type(ready) is not MatchedV3HostContainerReadyMetadata:
        raise TypeError("READY metadata must use the exact READY type")
    expected = {
        "qualification_plan_sha256": request.qualification_plan_sha256,
        "intent_sha256": intent_sha256,
        "case_ordinal": request.case_ordinal,
        "candidate_id": request.candidate_id,
        "qualification_case_id": request.qualification_case_id,
        "qualification_case_manifest_sha256": request.qualification_case_manifest_sha256,
        "image_id": request.image_id,
        "runtime_identity_sha256": request.runtime_identity_sha256,
        "driver_descriptor_sha256": request.in_container_driver_descriptor_sha256,
        "driver_source_sha256": request.in_container_driver_source_sha256,
        "runtime_profile_sha256": request.runtime_profile_sha256,
    }
    if any(getattr(ready, field) != value for field, value in expected.items()):
        _fail("READY metadata is cross-wired from its request or intent")
    return _sha256(canonical_matched_v3_host_container_ready_metadata_bytes(ready))


def _require_go_chain(
    ready: MatchedV3HostContainerReadyMetadata,
    commitment: MatchedV3HostQualificationGoCommitment,
) -> str:
    if type(commitment) is not MatchedV3HostQualificationGoCommitment:
        raise TypeError("GO commitment must use the exact GO type")
    expected = build_matched_v3_host_qualification_go_commitment(
        ready,
        go_commitment_monotonic_ns=commitment.go_commitment_monotonic_ns,
    )
    if commitment != expected:
        _fail("GO commitment is not the exact one-way projection of READY metadata")
    return _sha256(canonical_matched_v3_host_qualification_go_commitment_bytes(commitment))


def _require_cgroup_chain(
    request: MatchedV3HostQualificationCaseRequest,
    intent_sha256: str,
    ready: MatchedV3HostContainerReadyMetadata,
    commitment: MatchedV3HostQualificationGoCommitment,
    proof: MatchedV3HostCgroupV2BoundaryProof,
) -> str:
    if type(proof) is not MatchedV3HostCgroupV2BoundaryProof:
        raise TypeError("cgroup proof must use the exact boundary-proof type")
    expected = {
        "qualification_plan_sha256": request.qualification_plan_sha256,
        "intent_sha256": intent_sha256,
        "case_ordinal": request.case_ordinal,
        "candidate_id": request.candidate_id,
        "qualification_case_id": request.qualification_case_id,
        "qualification_case_manifest_sha256": request.qualification_case_manifest_sha256,
        "container_id": ready.container_id,
        "container_name": ready.container_name,
        "target_pid": ready.host_pid,
        "target_process_start_time_ticks": ready.host_process_start_time_ticks,
        "proc_cgroup_path": ready.proc_cgroup_path,
        "container_cgroup_device": ready.container_cgroup_device,
        "container_cgroup_inode": ready.container_cgroup_inode,
    }
    if any(getattr(proof, field) != value for field, value in expected.items()):
        _fail("cgroup boundary proof is cross-wired from its request, intent, or READY")
    if proof.samples[1].monotonic_ns != ready.ready_monotonic_ns:
        _fail("READY time differs from the retained-FD driver-ready cgroup sample")
    if not (
        ready.ready_monotonic_ns
        < commitment.go_commitment_monotonic_ns
        < proof.samples[2].monotonic_ns
    ):
        _fail("GO commitment must precede the retained-FD pre-cleanup sample")
    if (
        ready.ready_cgroup_sample_sha256
        != matched_v3_host_cgroup_sample_sha256(proof.samples[1])
    ):
        _fail("READY cgroup sample identity differs from the retained-FD proof")
    return _sha256(canonical_matched_v3_host_cgroup_v2_boundary_proof_bytes(proof))


def _require_terminal_chain(
    request: MatchedV3HostQualificationCaseRequest,
    terminal: MatchedV3HostContainerTerminalMetadata,
) -> str:
    if type(terminal) is not MatchedV3HostContainerTerminalMetadata:
        raise TypeError("terminal metadata must use the exact terminal type")
    expected = {
        "qualification_plan_sha256": request.qualification_plan_sha256,
        "case_ordinal": request.case_ordinal,
        "candidate_id": request.candidate_id,
        "qualification_case_id": request.qualification_case_id,
        "qualification_case_manifest_sha256": request.qualification_case_manifest_sha256,
        "interaction_horizon": request.horizon,
        "image_id": request.image_id,
        "driver_descriptor_sha256": request.in_container_driver_descriptor_sha256,
        "driver_source_sha256": request.in_container_driver_source_sha256,
        "publisher_descriptor_sha256": request.publisher_descriptor_sha256,
        "publisher_source_sha256": request.publisher_source_sha256,
    }
    if any(getattr(terminal, field) != value for field, value in expected.items()):
        _fail("terminal metadata is cross-wired from its request")
    return _sha256(canonical_matched_v3_host_container_terminal_metadata_bytes(terminal))


def _require_lifecycle_checkpoint(
    record: MatchedV3HostQualificationLifecycleRecord,
    expected_phases: tuple[str, ...],
    label: str,
) -> str:
    if type(record) is not MatchedV3HostQualificationLifecycleRecord:
        raise TypeError(f"{label} must use the exact lifecycle type")
    if (
        record.completed_phases != expected_phases
        or record.failure_phase is not None
        or record.uncertainty_kind is not None
    ):
        _fail(f"{label} is not the exact successful chronology checkpoint")
    return _sha256(canonical_matched_v3_host_qualification_lifecycle_record_bytes(record))


@dataclass(frozen=True, slots=True)
class MatchedV3HostQualificationCaseExecutionReceipt:
    """Typed metadata linkage for a fully observed case, still nonauthorizing."""

    qualification_plan_sha256: str
    request_sha256: str
    intent_sha256: str
    ready_metadata_sha256: str
    go_commitment_sha256: str
    lifecycle_record_sha256: str
    cgroup_boundary_proof_sha256: str
    terminal_metadata_sha256: str
    case_ordinal: int
    candidate_id: str
    qualification_case_id: str
    qualification_case_manifest_sha256: str
    image_id: str
    resource_observation_request_sha256: str
    resource_observation_receipt_sha256: str
    publication_address_sha256: str
    publication_receipt_sha256: str
    returncode: int
    timed_out: bool
    execution_state: str
    publication_state: str
    cleanup_state: str
    case_consumed: bool
    same_case_retry_permitted: bool

    def __post_init__(self) -> None:
        for value, label in (
            (self.qualification_plan_sha256, "execution receipt plan"),
            (self.request_sha256, "execution receipt request"),
            (self.intent_sha256, "execution receipt intent"),
            (self.ready_metadata_sha256, "execution receipt READY metadata"),
            (self.go_commitment_sha256, "execution receipt GO commitment"),
            (self.lifecycle_record_sha256, "execution receipt lifecycle"),
            (self.cgroup_boundary_proof_sha256, "execution receipt cgroup proof"),
            (self.terminal_metadata_sha256, "execution receipt terminal metadata"),
            (self.qualification_case_manifest_sha256, "execution receipt case manifest"),
            (self.resource_observation_request_sha256, "resource observation request"),
            (self.resource_observation_receipt_sha256, "resource observation receipt"),
            (self.publication_address_sha256, "execution receipt publication address"),
            (self.publication_receipt_sha256, "execution receipt publication receipt"),
        ):
            _require_sha256(value, label)
        _require_candidate(self.candidate_id, self.case_ordinal)
        _require_identifier(self.qualification_case_id, "execution receipt qualification case")
        _validate_image_lineage(image_id=self.image_id)
        _require_int(self.returncode, "execution receipt return code", minimum=-_MAX_INTEGER)
        if _require_bool(self.timed_out, "execution receipt timeout") is not False:
            _fail("complete execution metadata cannot be a timed-out receipt")
        if self.returncode != 0:
            _fail("complete execution metadata requires return code zero")
        if self.execution_state != "metadata_complete_non_authorizing":
            _fail("execution receipt state differs")
        if self.publication_state != "committed" or self.cleanup_state != "proven_empty":
            _fail("complete execution receipt publication or cleanup state differs")
        if _require_bool(self.case_consumed, "execution receipt case consumed") is not True:
            _fail("complete execution receipt must preserve consumed case state")
        if (
            _require_bool(
                self.same_case_retry_permitted,
                "execution receipt same-case retry",
            )
            is not False
        ):
            _fail("execution receipt can never permit same-case retry")

    def to_dict(self) -> dict[str, Any]:
        body = {
            "schema_version": HOST_QUALIFICATION_CASE_RECEIPT_SCHEMA_VERSION,
            "status": "case_metadata_complete_non_authorizing_not_evaluated",
            **{field: getattr(self, field) for field in self.__dataclass_fields__},
            "readiness": {
                "execution_ready": False,
                "qualification_evaluated": False,
                "observation_issuer_available": False,
            },
            "authority": _authority(),
            "claims": _claims(),
        }
        return _with_body_sha256(body, "execution_receipt_body_sha256")


def canonical_matched_v3_host_qualification_case_execution_receipt_bytes(
    receipt: MatchedV3HostQualificationCaseExecutionReceipt,
) -> bytes:
    if type(receipt) is not MatchedV3HostQualificationCaseExecutionReceipt:
        raise TypeError("execution receipt must use the exact receipt type")
    return _canonical_json(receipt.to_dict())


def build_matched_v3_host_qualification_case_execution_receipt(
    *,
    request: MatchedV3HostQualificationCaseRequest,
    intent: MatchedV3HostQualificationCaseIntent,
    ready: MatchedV3HostContainerReadyMetadata,
    commitment: MatchedV3HostQualificationGoCommitment,
    cgroup_proof: MatchedV3HostCgroupV2BoundaryProof,
    pre_receipt_lifecycle: MatchedV3HostQualificationLifecycleRecord,
    terminal: MatchedV3HostContainerTerminalMetadata,
    resource_observation_request_sha256: str,
    resource_observation_receipt_sha256: str,
) -> MatchedV3HostQualificationCaseExecutionReceipt:
    """Build one fully cross-linked, pre-receipt-checkpointed content receipt."""

    request_sha256, intent_sha256 = _require_request_intent_chain(request, intent)
    ready_sha256 = _require_ready_chain(request, intent_sha256, ready)
    commitment_sha256 = _require_go_chain(ready, commitment)
    cgroup_sha256 = _require_cgroup_chain(
        request,
        intent_sha256,
        ready,
        commitment,
        cgroup_proof,
    )
    lifecycle_sha256 = _require_lifecycle_checkpoint(
        pre_receipt_lifecycle,
        HOST_QUALIFICATION_PRE_RECEIPT_PHASES,
        "pre-receipt lifecycle",
    )
    terminal_sha256 = _require_terminal_chain(request, terminal)
    resource_request = _require_sha256(
        resource_observation_request_sha256,
        "resource observation request",
    )
    resource_receipt = _require_sha256(
        resource_observation_receipt_sha256,
        "resource observation receipt",
    )
    return MatchedV3HostQualificationCaseExecutionReceipt(
        qualification_plan_sha256=request.qualification_plan_sha256,
        request_sha256=request_sha256,
        intent_sha256=intent_sha256,
        ready_metadata_sha256=ready_sha256,
        go_commitment_sha256=commitment_sha256,
        lifecycle_record_sha256=lifecycle_sha256,
        cgroup_boundary_proof_sha256=cgroup_sha256,
        terminal_metadata_sha256=terminal_sha256,
        case_ordinal=request.case_ordinal,
        candidate_id=request.candidate_id,
        qualification_case_id=request.qualification_case_id,
        qualification_case_manifest_sha256=request.qualification_case_manifest_sha256,
        image_id=request.image_id,
        resource_observation_request_sha256=resource_request,
        resource_observation_receipt_sha256=resource_receipt,
        publication_address_sha256=terminal.publication_address_sha256,
        publication_receipt_sha256=terminal.publication_receipt_sha256,
        returncode=0,
        timed_out=False,
        execution_state="metadata_complete_non_authorizing",
        publication_state="committed",
        cleanup_state="proven_empty",
        case_consumed=True,
        same_case_retry_permitted=False,
    )


def validate_matched_v3_host_qualification_case_execution_receipt_chain(
    receipt: MatchedV3HostQualificationCaseExecutionReceipt,
    *,
    request: MatchedV3HostQualificationCaseRequest,
    intent: MatchedV3HostQualificationCaseIntent,
    ready: MatchedV3HostContainerReadyMetadata,
    commitment: MatchedV3HostQualificationGoCommitment,
    cgroup_proof: MatchedV3HostCgroupV2BoundaryProof,
    pre_receipt_lifecycle: MatchedV3HostQualificationLifecycleRecord,
    terminal: MatchedV3HostContainerTerminalMetadata,
    expected_resource_observation_request_sha256: str,
    expected_resource_observation_receipt_sha256: str,
) -> None:
    """Reject a structurally typed receipt unless every available link is exact."""

    if type(receipt) is not MatchedV3HostQualificationCaseExecutionReceipt:
        raise TypeError("execution receipt must use the exact receipt type")
    expected = build_matched_v3_host_qualification_case_execution_receipt(
        request=request,
        intent=intent,
        ready=ready,
        commitment=commitment,
        cgroup_proof=cgroup_proof,
        pre_receipt_lifecycle=pre_receipt_lifecycle,
        terminal=terminal,
        resource_observation_request_sha256=(
            expected_resource_observation_request_sha256
        ),
        resource_observation_receipt_sha256=(
            expected_resource_observation_receipt_sha256
        ),
    )
    if receipt != expected:
        _fail("execution receipt is structurally valid but cross-linked content differs")


def parse_matched_v3_host_qualification_case_execution_receipt(
    raw: bytes,
) -> MatchedV3HostQualificationCaseExecutionReceipt:
    value = _strict_json(raw)
    validate_matched_v3_host_metadata_only_mapping(value)
    _validate_body_sha256(value, "execution_receipt_body_sha256", "execution receipt")
    item = _exact(
        value,
        frozenset(
            {
                "schema_version",
                "status",
                "readiness",
                "authority",
                "claims",
                "execution_receipt_body_sha256",
                *MatchedV3HostQualificationCaseExecutionReceipt.__dataclass_fields__,
            }
        ),
        "execution receipt",
    )
    if (
        item.pop("schema_version") != HOST_QUALIFICATION_CASE_RECEIPT_SCHEMA_VERSION
        or item.pop("status") != "case_metadata_complete_non_authorizing_not_evaluated"
    ):
        _fail("execution receipt identity differs")
    if item.pop("readiness") != {
        "execution_ready": False,
        "qualification_evaluated": False,
        "observation_issuer_available": False,
    }:
        _fail("execution receipt readiness differs")
    if item.pop("authority") != _authority() or item.pop("claims") != _claims():
        _fail("execution receipt authority or claims differ")
    item.pop("execution_receipt_body_sha256")
    receipt = MatchedV3HostQualificationCaseExecutionReceipt(**item)
    if raw != canonical_matched_v3_host_qualification_case_execution_receipt_bytes(receipt):
        _fail("execution receipt replay differs")
    return receipt


@dataclass(frozen=True, slots=True)
class MatchedV3HostQualificationCaseFailureReceipt:
    """Typed nonretryable failure or uncertainty metadata; never a clean acceptance decision."""

    qualification_plan_sha256: str
    request_sha256: str
    intent_sha256: str
    lifecycle_record_sha256: str
    ready_metadata_sha256: str | None
    go_commitment_sha256: str | None
    host_execution_receipt_sha256: str | None
    host_execution_receipt_state: str
    case_ordinal: int
    candidate_id: str
    qualification_case_id: str
    qualification_case_manifest_sha256: str
    image_id: str
    failure_phase: str
    exception_type: str
    error_message_sha256: str
    case_start_state: str
    container_state: str
    publication_state: str
    cgroup_cleanup_state: str
    cgroup_boundary_proof_sha256: str | None
    terminal_metadata_sha256: str | None
    case_consumed: bool
    same_case_retry_permitted: bool

    def __post_init__(self) -> None:
        for value, label in (
            (self.qualification_plan_sha256, "failure plan"),
            (self.request_sha256, "failure request"),
            (self.intent_sha256, "failure intent"),
            (self.lifecycle_record_sha256, "failure lifecycle"),
            (self.qualification_case_manifest_sha256, "failure case manifest"),
            (self.error_message_sha256, "failure error message"),
        ):
            _require_sha256(value, label)
        _require_candidate(self.candidate_id, self.case_ordinal)
        _require_identifier(self.qualification_case_id, "failure qualification case")
        _validate_image_lineage(image_id=self.image_id)
        if self.failure_phase not in HOST_QUALIFICATION_LIFECYCLE_PHASES:
            _fail("failure phase differs")
        if _SAFE_TYPE_RE.fullmatch(self.exception_type) is None:
            _fail("failure exception type differs")
        if self.case_start_state not in {"not_started", "started", "uncertain"}:
            _fail("failure case-start state differs")
        if self.container_state not in {"known_absent", "known_present", "uncertain"}:
            _fail("failure container state differs")
        if self.publication_state not in {"not_started", "committed", "uncertain"}:
            _fail("failure publication state differs")
        if self.cgroup_cleanup_state not in {
            "not_created",
            "proven_empty",
            "not_empty",
            "uncertain",
        }:
            _fail("failure cgroup-cleanup state differs")
        _require_optional_sha256(self.cgroup_boundary_proof_sha256, "failure cgroup proof")
        _require_optional_sha256(self.terminal_metadata_sha256, "failure terminal metadata")
        _require_optional_sha256(self.ready_metadata_sha256, "failure READY metadata")
        _require_optional_sha256(self.go_commitment_sha256, "failure GO commitment")
        _require_optional_sha256(
            self.host_execution_receipt_sha256,
            "failure host execution receipt",
        )
        if self.host_execution_receipt_state not in {
            "not_committed",
            "commit_uncertain",
            "committed",
        }:
            _fail("failure host execution-receipt state differs")
        if (
            self.host_execution_receipt_state == "committed"
        ) != (self.host_execution_receipt_sha256 is not None):
            _fail("failure host execution-receipt state and identity differ")
        if _require_bool(self.case_consumed, "failure case consumed") is not True:
            _fail("intent-bound failure receipt must preserve consumed case state")
        if (
            _require_bool(self.same_case_retry_permitted, "failure same-case retry")
            is not False
        ):
            _fail("failure receipt can never permit same-case retry")
        if self.case_start_state != "not_started" and self.publication_state == "not_started":
            _fail("post-start failure must conservatively mark publication committed or uncertain")

    @property
    def classification(self) -> str:
        if self.case_start_state in {"started", "uncertain"}:
            return "case_state_uncertain_non_retriable"
        return "intent_consumed_no_workload_started_non_retriable"

    def to_dict(self) -> dict[str, Any]:
        body = {
            "schema_version": HOST_QUALIFICATION_CASE_FAILURE_SCHEMA_VERSION,
            "status": "case_failure_or_uncertainty_recorded_non_authorizing",
            "classification": self.classification,
            **{field: getattr(self, field) for field in self.__dataclass_fields__},
            "policy": {
                "clean_rejection_recorded": False,
                "read_only_reconciliation_only": True,
                "same_case_retry_permitted": False,
            },
            "authority": _authority(),
            "claims": _claims(),
        }
        return _with_body_sha256(body, "failure_receipt_body_sha256")


def canonical_matched_v3_host_qualification_case_failure_receipt_bytes(
    receipt: MatchedV3HostQualificationCaseFailureReceipt,
) -> bytes:
    if type(receipt) is not MatchedV3HostQualificationCaseFailureReceipt:
        raise TypeError("failure receipt must use the exact failure-receipt type")
    return _canonical_json(receipt.to_dict())


def build_matched_v3_host_qualification_case_failure_receipt(
    *,
    request: MatchedV3HostQualificationCaseRequest,
    intent: MatchedV3HostQualificationCaseIntent,
    lifecycle: MatchedV3HostQualificationLifecycleRecord,
    exception_type: str,
    error_message_sha256: str,
    ready: MatchedV3HostContainerReadyMetadata | None = None,
    commitment: MatchedV3HostQualificationGoCommitment | None = None,
    cgroup_proof: MatchedV3HostCgroupV2BoundaryProof | None = None,
    terminal: MatchedV3HostContainerTerminalMetadata | None = None,
    execution_receipt: MatchedV3HostQualificationCaseExecutionReceipt | None = None,
    pre_receipt_lifecycle: MatchedV3HostQualificationLifecycleRecord | None = None,
    resource_observation_request_sha256: str | None = None,
    resource_observation_receipt_sha256: str | None = None,
) -> MatchedV3HostQualificationCaseFailureReceipt:
    """Derive one consumed, nonretryable failure receipt from exact phase state."""

    request_sha256, intent_sha256 = _require_request_intent_chain(request, intent)
    if type(lifecycle) is not MatchedV3HostQualificationLifecycleRecord:
        raise TypeError("failure lifecycle must use the exact lifecycle type")
    failure_phase = lifecycle.failure_phase
    if failure_phase is None or not lifecycle.intent_committed:
        _fail("intent-bound failure receipt requires a definite committed intent failure")
    phases = set(lifecycle.completed_phases)
    failure_ordinal = HOST_QUALIFICATION_LIFECYCLE_PHASES.index(failure_phase)
    ready_completed = "driver_ready" in phases
    go_completed = "go_committed" in phases
    postflight_completed = "postflight_revalidated" in phases
    terminal_completed = "terminal_metadata_validated" in phases
    if ready_completed != (ready is not None):
        _fail("failure READY linkage differs from the completed lifecycle prefix")
    if go_completed != (commitment is not None):
        _fail("failure GO linkage differs from the completed lifecycle prefix")
    if postflight_completed != (cgroup_proof is not None):
        _fail("failure cgroup-proof linkage differs from the completed lifecycle prefix")
    if terminal_completed != (terminal is not None):
        _fail("failure terminal linkage differs from the completed lifecycle prefix")
    ready_sha256: str | None = None
    commitment_sha256: str | None = None
    cgroup_sha256: str | None = None
    terminal_sha256: str | None = None
    host_execution_receipt_sha256: str | None = None
    if ready is not None:
        ready_sha256 = _require_ready_chain(request, intent_sha256, ready)
    if commitment is not None:
        if ready is None:
            _fail("failure GO linkage requires exact READY metadata")
        commitment_sha256 = _require_go_chain(ready, commitment)
    if cgroup_proof is not None:
        if ready is None or commitment is None:
            _fail("failure cgroup proof requires exact READY and GO metadata")
        cgroup_sha256 = _require_cgroup_chain(
            request,
            intent_sha256,
            ready,
            commitment,
            cgroup_proof,
        )
    if terminal is not None:
        terminal_sha256 = _require_terminal_chain(request, terminal)
    receipt_committed = "receipt_committed" in phases
    receipt_commit_uncertain = failure_phase == "receipt_committed"
    receipt_link_values = (
        execution_receipt,
        pre_receipt_lifecycle,
        resource_observation_request_sha256,
        resource_observation_receipt_sha256,
    )
    if receipt_committed and any(value is None for value in receipt_link_values):
        _fail("failure after receipt commit requires the exact host receipt chain")
    if not receipt_committed and any(value is not None for value in receipt_link_values):
        _fail("failure before a definite receipt commit cannot claim a host receipt")
    if receipt_committed:
        if (
            execution_receipt is None
            or pre_receipt_lifecycle is None
            or resource_observation_request_sha256 is None
            or resource_observation_receipt_sha256 is None
            or ready is None
            or commitment is None
            or cgroup_proof is None
            or terminal is None
        ):
            _fail("failure host receipt chain is incomplete")
        validate_matched_v3_host_qualification_case_execution_receipt_chain(
            execution_receipt,
            request=request,
            intent=intent,
            ready=ready,
            commitment=commitment,
            cgroup_proof=cgroup_proof,
            pre_receipt_lifecycle=pre_receipt_lifecycle,
            terminal=terminal,
            expected_resource_observation_request_sha256=(
                resource_observation_request_sha256
            ),
            expected_resource_observation_receipt_sha256=(
                resource_observation_receipt_sha256
            ),
        )
        host_execution_receipt_sha256 = _sha256(
            canonical_matched_v3_host_qualification_case_execution_receipt_bytes(
                execution_receipt
            )
        )
    host_execution_receipt_state = (
        "committed"
        if receipt_committed
        else "commit_uncertain"
        if receipt_commit_uncertain
        else "not_committed"
    )
    case_start_ordinal = HOST_QUALIFICATION_LIFECYCLE_PHASES.index("case_started")
    container_ordinal = HOST_QUALIFICATION_LIFECYCLE_PHASES.index("container_created")
    fresh_cgroup_ordinal = HOST_QUALIFICATION_LIFECYCLE_PHASES.index(
        "fresh_cgroup_created"
    )
    if "case_started" in phases:
        case_start_state = "started"
    elif failure_ordinal == case_start_ordinal:
        case_start_state = "uncertain"
    else:
        case_start_state = "not_started"
    if "publication_committed" in phases:
        publication_state = "committed"
    elif lifecycle.publication_may_be_visible:
        publication_state = "uncertain"
    else:
        publication_state = "not_started"
    if "container_absent" in phases or failure_ordinal < container_ordinal:
        container_state = "known_absent"
    else:
        container_state = "uncertain"
    if "cgroup_empty" in phases:
        cleanup_state = "proven_empty"
    elif failure_ordinal < fresh_cgroup_ordinal:
        cleanup_state = "not_created"
    else:
        cleanup_state = "uncertain"
    return MatchedV3HostQualificationCaseFailureReceipt(
        qualification_plan_sha256=request.qualification_plan_sha256,
        request_sha256=request_sha256,
        intent_sha256=intent_sha256,
        lifecycle_record_sha256=_sha256(
            canonical_matched_v3_host_qualification_lifecycle_record_bytes(lifecycle)
        ),
        ready_metadata_sha256=ready_sha256,
        go_commitment_sha256=commitment_sha256,
        host_execution_receipt_sha256=host_execution_receipt_sha256,
        host_execution_receipt_state=host_execution_receipt_state,
        case_ordinal=request.case_ordinal,
        candidate_id=request.candidate_id,
        qualification_case_id=request.qualification_case_id,
        qualification_case_manifest_sha256=request.qualification_case_manifest_sha256,
        image_id=request.image_id,
        failure_phase=failure_phase,
        exception_type=exception_type,
        error_message_sha256=error_message_sha256,
        case_start_state=case_start_state,
        container_state=container_state,
        publication_state=publication_state,
        cgroup_cleanup_state=cleanup_state,
        cgroup_boundary_proof_sha256=cgroup_sha256,
        terminal_metadata_sha256=terminal_sha256,
        case_consumed=True,
        same_case_retry_permitted=False,
    )


def validate_matched_v3_host_qualification_case_failure_receipt_chain(
    receipt: MatchedV3HostQualificationCaseFailureReceipt,
    *,
    request: MatchedV3HostQualificationCaseRequest,
    intent: MatchedV3HostQualificationCaseIntent,
    lifecycle: MatchedV3HostQualificationLifecycleRecord,
    expected_exception_type: str,
    expected_error_message_sha256: str,
    ready: MatchedV3HostContainerReadyMetadata | None = None,
    commitment: MatchedV3HostQualificationGoCommitment | None = None,
    cgroup_proof: MatchedV3HostCgroupV2BoundaryProof | None = None,
    terminal: MatchedV3HostContainerTerminalMetadata | None = None,
    execution_receipt: MatchedV3HostQualificationCaseExecutionReceipt | None = None,
    pre_receipt_lifecycle: MatchedV3HostQualificationLifecycleRecord | None = None,
    resource_observation_request_sha256: str | None = None,
    resource_observation_receipt_sha256: str | None = None,
) -> None:
    """Reject a typed failure receipt unless its phase-derived chain is exact."""

    if type(receipt) is not MatchedV3HostQualificationCaseFailureReceipt:
        raise TypeError("failure receipt must use the exact failure-receipt type")
    expected = build_matched_v3_host_qualification_case_failure_receipt(
        request=request,
        intent=intent,
        lifecycle=lifecycle,
        exception_type=expected_exception_type,
        error_message_sha256=expected_error_message_sha256,
        ready=ready,
        commitment=commitment,
        cgroup_proof=cgroup_proof,
        terminal=terminal,
        execution_receipt=execution_receipt,
        pre_receipt_lifecycle=pre_receipt_lifecycle,
        resource_observation_request_sha256=resource_observation_request_sha256,
        resource_observation_receipt_sha256=resource_observation_receipt_sha256,
    )
    if receipt != expected:
        _fail("failure receipt is structurally valid but its phase-derived chain differs")


def parse_matched_v3_host_qualification_case_failure_receipt(
    raw: bytes,
) -> MatchedV3HostQualificationCaseFailureReceipt:
    value = _strict_json(raw)
    validate_matched_v3_host_metadata_only_mapping(value)
    _validate_body_sha256(value, "failure_receipt_body_sha256", "failure receipt")
    item = _exact(
        value,
        frozenset(
            {
                "schema_version",
                "status",
                "classification",
                "policy",
                "authority",
                "claims",
                "failure_receipt_body_sha256",
                *MatchedV3HostQualificationCaseFailureReceipt.__dataclass_fields__,
            }
        ),
        "failure receipt",
    )
    if (
        item.pop("schema_version") != HOST_QUALIFICATION_CASE_FAILURE_SCHEMA_VERSION
        or item.pop("status") != "case_failure_or_uncertainty_recorded_non_authorizing"
    ):
        _fail("failure receipt identity differs")
    classification = item.pop("classification")
    if item.pop("policy") != {
        "clean_rejection_recorded": False,
        "read_only_reconciliation_only": True,
        "same_case_retry_permitted": False,
    }:
        _fail("failure receipt policy differs")
    if item.pop("authority") != _authority() or item.pop("claims") != _claims():
        _fail("failure receipt authority or claims differ")
    item.pop("failure_receipt_body_sha256")
    receipt = MatchedV3HostQualificationCaseFailureReceipt(**item)
    if classification != receipt.classification:
        _fail("failure receipt classification differs")
    if raw != canonical_matched_v3_host_qualification_case_failure_receipt_bytes(receipt):
        _fail("failure receipt replay differs")
    return receipt


@dataclass(frozen=True, slots=True)
class MatchedV3HostQualificationObservationHandoff:
    """Metadata-only input references for a future separate issuer/evaluator."""

    qualification_plan_schema_version: str
    qualification_plan_sha256: str
    observation_registry_schema_version: str
    observation_registry_descriptor_sha256: str
    case_ordinal: int
    candidate_id: str
    qualification_case_id: str
    qualification_case_manifest_sha256: str
    intent_sha256: str
    lifecycle_record_sha256: str
    host_execution_receipt_sha256: str
    cgroup_boundary_proof_sha256: str
    resource_observation_request_sha256: str
    resource_observation_receipt_sha256: str
    terminal_metadata_sha256: str
    runtime_observation_candidate_sha256: str
    candidate_observation_candidate_sha256: str
    fresh_replay_observation_candidate_sha256: str
    publication_address_sha256: str
    publication_manifest_sha256: str
    publication_receipt_sha256: str
    published_bundle_sha256: str
    reload_observation_sha256: str

    def __post_init__(self) -> None:
        if self.qualification_plan_schema_version != QUALIFICATION_PLAN_V2_SCHEMA_VERSION:
            _fail("handoff must name the exact currently incompatible plan-v2 schema")
        if (
            self.observation_registry_schema_version
            != QUALIFICATION_OBSERVATION_REGISTRY_V1_SCHEMA_VERSION
        ):
            _fail("handoff must name the exact currently incompatible observation registry v1")
        _require_candidate(self.candidate_id, self.case_ordinal)
        _require_identifier(self.qualification_case_id, "handoff qualification case")
        for value, label in (
            (self.qualification_plan_sha256, "handoff qualification plan"),
            (self.observation_registry_descriptor_sha256, "handoff observation registry"),
            (self.qualification_case_manifest_sha256, "handoff case manifest"),
            (self.intent_sha256, "handoff intent"),
            (self.lifecycle_record_sha256, "handoff lifecycle"),
            (self.host_execution_receipt_sha256, "handoff host execution receipt"),
            (self.cgroup_boundary_proof_sha256, "handoff cgroup proof"),
            (self.resource_observation_request_sha256, "handoff resource request"),
            (self.resource_observation_receipt_sha256, "handoff resource receipt"),
            (self.terminal_metadata_sha256, "handoff terminal metadata"),
            (self.runtime_observation_candidate_sha256, "runtime observation candidate"),
            (self.candidate_observation_candidate_sha256, "candidate observation candidate"),
            (
                self.fresh_replay_observation_candidate_sha256,
                "fresh-replay observation candidate",
            ),
            (self.publication_address_sha256, "handoff publication address"),
            (self.publication_manifest_sha256, "handoff publication manifest"),
            (self.publication_receipt_sha256, "handoff publication receipt"),
            (self.published_bundle_sha256, "handoff published bundle"),
            (self.reload_observation_sha256, "handoff reload observation"),
        ):
            _require_sha256(value, label)

    def to_dict(self) -> dict[str, Any]:
        body = {
            "schema_version": HOST_QUALIFICATION_OBSERVATION_HANDOFF_SCHEMA_VERSION,
            "status": "metadata_only_handoff_waiting_for_future_separate_issuer",
            **{field: getattr(self, field) for field in self.__dataclass_fields__},
            "compatibility": {
                "observation_registry_v1_compatible": False,
                "qualification_plan_v2_compatible": False,
            },
            "metadata_policy": {
                "raw_content_transported": False,
                "score_or_reward_decoded": False,
                "host_may_branch_retry_rank_or_select": False,
            },
            "readiness": {
                "observation_issuer_available": False,
                "qualification_evaluated": False,
                "execution_ready": False,
            },
            "authority": _authority(),
            "claims": _claims(),
        }
        return _with_body_sha256(body, "handoff_body_sha256")


def canonical_matched_v3_host_qualification_observation_handoff_bytes(
    handoff: MatchedV3HostQualificationObservationHandoff,
) -> bytes:
    if type(handoff) is not MatchedV3HostQualificationObservationHandoff:
        raise TypeError("handoff must use the exact observation-handoff type")
    return _canonical_json(handoff.to_dict())


def build_matched_v3_host_qualification_observation_handoff(
    *,
    request: MatchedV3HostQualificationCaseRequest,
    intent: MatchedV3HostQualificationCaseIntent,
    ready: MatchedV3HostContainerReadyMetadata,
    commitment: MatchedV3HostQualificationGoCommitment,
    cgroup_proof: MatchedV3HostCgroupV2BoundaryProof,
    pre_receipt_lifecycle: MatchedV3HostQualificationLifecycleRecord,
    execution_receipt: MatchedV3HostQualificationCaseExecutionReceipt,
    pre_handoff_lifecycle: MatchedV3HostQualificationLifecycleRecord,
    terminal: MatchedV3HostContainerTerminalMetadata,
    resource_observation_request_sha256: str,
    resource_observation_receipt_sha256: str,
    runtime_observation_candidate_sha256: str,
    candidate_observation_candidate_sha256: str,
    fresh_replay_observation_candidate_sha256: str,
) -> MatchedV3HostQualificationObservationHandoff:
    """Build a cross-linked handoff after receipt commit and before handoff commit."""

    validate_matched_v3_host_qualification_case_execution_receipt_chain(
        execution_receipt,
        request=request,
        intent=intent,
        ready=ready,
        commitment=commitment,
        cgroup_proof=cgroup_proof,
        pre_receipt_lifecycle=pre_receipt_lifecycle,
        terminal=terminal,
        expected_resource_observation_request_sha256=(
            resource_observation_request_sha256
        ),
        expected_resource_observation_receipt_sha256=(
            resource_observation_receipt_sha256
        ),
    )
    intent_sha256 = _sha256(
        canonical_matched_v3_host_qualification_case_intent_bytes(intent)
    )
    cgroup_sha256 = _sha256(
        canonical_matched_v3_host_cgroup_v2_boundary_proof_bytes(cgroup_proof)
    )
    terminal_sha256 = _sha256(
        canonical_matched_v3_host_container_terminal_metadata_bytes(terminal)
    )
    lifecycle_sha256 = _require_lifecycle_checkpoint(
        pre_handoff_lifecycle,
        HOST_QUALIFICATION_PRE_HANDOFF_PHASES,
        "pre-handoff lifecycle",
    )
    runtime_candidate = _require_sha256(
        runtime_observation_candidate_sha256,
        "runtime observation candidate",
    )
    candidate_candidate = _require_sha256(
        candidate_observation_candidate_sha256,
        "candidate observation candidate",
    )
    replay_candidate = _require_sha256(
        fresh_replay_observation_candidate_sha256,
        "fresh-replay observation candidate",
    )
    return MatchedV3HostQualificationObservationHandoff(
        qualification_plan_schema_version=request.qualification_plan_schema_version,
        qualification_plan_sha256=request.qualification_plan_sha256,
        observation_registry_schema_version=request.observation_registry_schema_version,
        observation_registry_descriptor_sha256=(
            request.observation_registry_descriptor_sha256
        ),
        case_ordinal=request.case_ordinal,
        candidate_id=request.candidate_id,
        qualification_case_id=request.qualification_case_id,
        qualification_case_manifest_sha256=request.qualification_case_manifest_sha256,
        intent_sha256=intent_sha256,
        lifecycle_record_sha256=lifecycle_sha256,
        host_execution_receipt_sha256=_sha256(
            canonical_matched_v3_host_qualification_case_execution_receipt_bytes(
                execution_receipt
            )
        ),
        cgroup_boundary_proof_sha256=cgroup_sha256,
        resource_observation_request_sha256=(
            execution_receipt.resource_observation_request_sha256
        ),
        resource_observation_receipt_sha256=(
            execution_receipt.resource_observation_receipt_sha256
        ),
        terminal_metadata_sha256=terminal_sha256,
        runtime_observation_candidate_sha256=runtime_candidate,
        candidate_observation_candidate_sha256=candidate_candidate,
        fresh_replay_observation_candidate_sha256=replay_candidate,
        publication_address_sha256=terminal.publication_address_sha256,
        publication_manifest_sha256=terminal.publication_manifest_sha256,
        publication_receipt_sha256=terminal.publication_receipt_sha256,
        published_bundle_sha256=terminal.published_bundle_sha256,
        reload_observation_sha256=terminal.reload_observation_sha256,
    )


def validate_matched_v3_host_qualification_observation_handoff_chain(
    handoff: MatchedV3HostQualificationObservationHandoff,
    *,
    request: MatchedV3HostQualificationCaseRequest,
    intent: MatchedV3HostQualificationCaseIntent,
    ready: MatchedV3HostContainerReadyMetadata,
    commitment: MatchedV3HostQualificationGoCommitment,
    cgroup_proof: MatchedV3HostCgroupV2BoundaryProof,
    pre_receipt_lifecycle: MatchedV3HostQualificationLifecycleRecord,
    execution_receipt: MatchedV3HostQualificationCaseExecutionReceipt,
    pre_handoff_lifecycle: MatchedV3HostQualificationLifecycleRecord,
    terminal: MatchedV3HostContainerTerminalMetadata,
    expected_resource_observation_request_sha256: str,
    expected_resource_observation_receipt_sha256: str,
    expected_runtime_observation_candidate_sha256: str,
    expected_candidate_observation_candidate_sha256: str,
    expected_fresh_replay_observation_candidate_sha256: str,
) -> None:
    """Reject a typed handoff unless its pre-handoff chain is exact."""

    if type(handoff) is not MatchedV3HostQualificationObservationHandoff:
        raise TypeError("handoff must use the exact observation-handoff type")
    expected = build_matched_v3_host_qualification_observation_handoff(
        request=request,
        intent=intent,
        ready=ready,
        commitment=commitment,
        cgroup_proof=cgroup_proof,
        pre_receipt_lifecycle=pre_receipt_lifecycle,
        execution_receipt=execution_receipt,
        pre_handoff_lifecycle=pre_handoff_lifecycle,
        terminal=terminal,
        resource_observation_request_sha256=(
            expected_resource_observation_request_sha256
        ),
        resource_observation_receipt_sha256=(
            expected_resource_observation_receipt_sha256
        ),
        runtime_observation_candidate_sha256=(
            expected_runtime_observation_candidate_sha256
        ),
        candidate_observation_candidate_sha256=(
            expected_candidate_observation_candidate_sha256
        ),
        fresh_replay_observation_candidate_sha256=(
            expected_fresh_replay_observation_candidate_sha256
        ),
    )
    if handoff != expected:
        _fail("handoff is structurally valid but its pre-handoff chain differs")


def parse_matched_v3_host_qualification_observation_handoff(
    raw: bytes,
) -> MatchedV3HostQualificationObservationHandoff:
    value = _strict_json(raw)
    validate_matched_v3_host_metadata_only_mapping(value)
    _validate_body_sha256(value, "handoff_body_sha256", "observation handoff")
    item = _exact(
        value,
        frozenset(
            {
                "schema_version",
                "status",
                "compatibility",
                "metadata_policy",
                "readiness",
                "authority",
                "claims",
                "handoff_body_sha256",
                *MatchedV3HostQualificationObservationHandoff.__dataclass_fields__,
            }
        ),
        "observation handoff",
    )
    if (
        item.pop("schema_version") != HOST_QUALIFICATION_OBSERVATION_HANDOFF_SCHEMA_VERSION
        or item.pop("status")
        != "metadata_only_handoff_waiting_for_future_separate_issuer"
    ):
        _fail("observation handoff identity differs")
    if item.pop("compatibility") != {
        "observation_registry_v1_compatible": False,
        "qualification_plan_v2_compatible": False,
    }:
        _fail("observation handoff compatibility differs")
    if item.pop("metadata_policy") != {
        "raw_content_transported": False,
        "score_or_reward_decoded": False,
        "host_may_branch_retry_rank_or_select": False,
    }:
        _fail("observation handoff metadata policy differs")
    if item.pop("readiness") != {
        "observation_issuer_available": False,
        "qualification_evaluated": False,
        "execution_ready": False,
    }:
        _fail("observation handoff readiness differs")
    if item.pop("authority") != _authority() or item.pop("claims") != _claims():
        _fail("observation handoff authority or claims differ")
    item.pop("handoff_body_sha256")
    handoff = MatchedV3HostQualificationObservationHandoff(**item)
    if raw != canonical_matched_v3_host_qualification_observation_handoff_bytes(handoff):
        _fail("observation handoff replay differs")
    return handoff


def _descriptor() -> dict[str, Any]:
    body: dict[str, Any] = {
        "schema_version": HOST_QUALIFICATION_EXECUTOR_DESCRIPTOR_SCHEMA_VERSION,
        "status": "contract_implemented_uninvoked_nonexecuting",
        "classification": "metadata_only_structural_host_executor_contract_non_authorizing",
        "horizon": MATCHED_V3_HORIZON,
        "candidate_count": len(MATCHED_V3_HOST_QUALIFICATION_CANDIDATE_IDS),
        "candidate_order": list(MATCHED_V3_HOST_QUALIFICATION_CANDIDATE_IDS),
        "resource_ceiling_fields": list(RESOURCE_CEILING_FIELDS),
        "schemas": {
            "case_execution_receipt": HOST_QUALIFICATION_CASE_RECEIPT_SCHEMA_VERSION,
            "case_failure_receipt": HOST_QUALIFICATION_CASE_FAILURE_SCHEMA_VERSION,
            "case_intent": HOST_QUALIFICATION_CASE_INTENT_SCHEMA_VERSION,
            "case_request": HOST_QUALIFICATION_CASE_REQUEST_SCHEMA_VERSION,
            "cgroup_boundary_proof": HOST_CGROUP_V2_BOUNDARY_PROOF_SCHEMA_VERSION,
            "container_ready": HOST_CONTAINER_READY_SCHEMA_VERSION,
            "container_terminal_metadata": HOST_CONTAINER_TERMINAL_METADATA_SCHEMA_VERSION,
            "go_commitment": HOST_QUALIFICATION_GO_COMMITMENT_SCHEMA_VERSION,
            "lifecycle_record": HOST_QUALIFICATION_LIFECYCLE_RECORD_SCHEMA_VERSION,
            "observation_handoff": HOST_QUALIFICATION_OBSERVATION_HANDOFF_SCHEMA_VERSION,
        },
        "compatibility": {
            QUALIFICATION_PLAN_V2_SCHEMA_VERSION: False,
            QUALIFICATION_OBSERVATION_REGISTRY_V1_SCHEMA_VERSION: False,
        },
        "dependency_pins": {
            "endpoint_resource_observer": {
                "descriptor_sha256": PINNED_ENDPOINT_RESOURCE_OBSERVER_DESCRIPTOR_SHA256,
                "source_sha256": PINNED_ENDPOINT_RESOURCE_OBSERVER_SOURCE_SHA256,
                "role": "corroborating_endpoint_observer_incomplete_for_28_fields",
                "request_schema_version": (
                    QUALIFICATION_RESOURCE_OBSERVATION_REQUEST_V1_SCHEMA_VERSION
                ),
                "receipt_schema_version": (
                    QUALIFICATION_RESOURCE_OBSERVATION_RECEIPT_V1_SCHEMA_VERSION
                ),
                "horizon": MATCHED_V3_HORIZON,
                "complete_28_field_observation": False,
            },
        },
        "historical_exclusions": {
            "image_ids": list(STALE_IMAGE_IDS),
            "build_lineages": [dict(item) for item in STALE_BUILD_LINEAGES],
            "partial_lineage_component_reuse_allowed": False,
        },
        "cgroup_contract": {
            "version": 2,
            "outer_case_cgroup_fresh_required": True,
            "initial_empty_required": True,
            "final_empty_and_removed_required": True,
            "controllers": ["cpu", "memory", "pids"],
            "max_depth": 1,
            "max_descendants": 1,
            "counter_endpoints": list(CGROUP_COUNTER_ENDPOINTS),
            "counter_semantics": dict(CGROUP_COUNTER_SEMANTICS),
            "counter_fds_retained_without_reopen": True,
            "intermediate_container_descendant_identity_stable": True,
            "memory_peak_reset_model": "fresh_cgroup_no_reset_retained_fd",
            "pids_peak_reset_model": "read_only_no_reset_fresh_cgroup_mandatory",
            "setsid_changes_cgroup": False,
            "fork_clone_inherit_cgroup": True,
            "cgroup_kill_required": True,
            "continuous_membership_proven": False,
            "external_privileged_migration_excluded": False,
            "daemon_owned_init_directly_reaped": False,
            "control_file_sample_atomic": False,
        },
        "handshake": {
            "ready_before_go_required": True,
            "ready_runtime_identity_bound_from_request": True,
            "ready_candidate_code_loaded": False,
            "ready_outcome_capability_issued": False,
            "ready_go_committed": False,
            "go_is_one_way": True,
            "go_precedes_pre_cleanup_sample": True,
            "post_go_publication_or_observation_failure": (
                "case_state_uncertain_non_retriable"
            ),
        },
        "metadata_policy": {
            "forbidden_keys": sorted(FORBIDDEN_METADATA_KEYS),
            "raw_content_transport_allowed": False,
            "score_or_reward_decode_allowed": False,
            "host_branch_retry_rank_or_select_allowed": False,
            "publication_digests_counts_and_sizes_visible": True,
            "publication_file_count_maximum": _MAX_PUBLICATION_FILES,
            "publication_aggregate_bytes_maximum": _MAX_PUBLICATION_AGGREGATE_BYTES,
            "zero_byte_file_metadata_allowed": True,
        },
        "lifecycle": {
            "phases": list(HOST_QUALIFICATION_LIFECYCLE_PHASES),
            "pre_receipt_checkpoint": list(HOST_QUALIFICATION_PRE_RECEIPT_PHASES),
            "pre_handoff_checkpoint": list(HOST_QUALIFICATION_PRE_HANDOFF_PHASES),
            "uncertainty_kinds": list(HOST_QUALIFICATION_UNCERTAINTY_KINDS),
            "same_case_retry_permitted": False,
            "failure_after_completed_receipt_commit_binds_host_receipt": True,
            "failure_during_receipt_commit_has_no_receipt_claim": True,
            "read_only_reconciliation_only_after_uncertainty": True,
            "clean_rejection_issued_here": False,
        },
        "public_api": {
            "structural_parser_cross_link_validation": False,
            "cross_linked_builders_are_content_only": True,
            "cross_link_validators_require_external_expectations": True,
            "projection_builders": [
                "build_matched_v3_host_qualification_case_intent",
                "build_matched_v3_host_qualification_go_commitment",
                "classify_matched_v3_host_qualification_lifecycle",
            ],
            "cross_linked_content_builders": [
                "build_matched_v3_host_qualification_case_execution_receipt",
                "build_matched_v3_host_qualification_case_failure_receipt",
                "build_matched_v3_host_qualification_observation_handoff",
            ],
            "cross_link_validators": [
                "validate_matched_v3_host_qualification_case_execution_receipt_chain",
                "validate_matched_v3_host_qualification_case_failure_receipt_chain",
                "validate_matched_v3_host_qualification_observation_handoff_chain",
            ],
            "execution_capability_issued": False,
            "mutation_apis": [],
            "structural_parsers": [
                "parse_matched_v3_host_qualification_case_request",
                "parse_matched_v3_host_qualification_case_intent",
                "parse_matched_v3_host_container_ready_metadata",
                "parse_matched_v3_host_qualification_go_commitment",
                "parse_matched_v3_host_cgroup_v2_boundary_proof",
                "parse_matched_v3_host_container_terminal_metadata",
                "parse_matched_v3_host_qualification_lifecycle_record",
                "parse_matched_v3_host_qualification_case_execution_receipt",
                "parse_matched_v3_host_qualification_case_failure_receipt",
                "parse_matched_v3_host_qualification_observation_handoff",
            ],
        },
        "runtime_surface": {
            "cgroup_writes_implemented": False,
            "docker_or_oci_invocation_implemented": False,
            "production_process_runner_implemented": False,
            "production_provisioner_implemented": False,
            "subprocess_imported": False,
        },
        "readiness": matched_v3_host_qualification_readiness(),
        "authority": _authority(),
        "claims": _claims(),
        "limitations": [
            "This module validates content only and exposes no production mutation path.",
            "Qualification-plan v2 is content-only, incomplete, and not execution authority.",
            "Observation-registry v1 has no issuer and cannot consume this handoff.",
            "The pinned endpoint observer cannot complete the exact 28-field resource record.",
            "A future version must bind a fresh full resource merger and separate issuer.",
            (
                "Structurally valid READY, GO, cgroup, receipt, failure, or "
                "handoff bytes attest nothing."
            ),
        ],
    }
    return _with_body_sha256(body, "descriptor_body_sha256")


_DESCRIPTOR: Final = _descriptor()
_DESCRIPTOR_BYTES: Final = _canonical_json(_DESCRIPTOR)

# Replaced only when the canonical descriptor changes deliberately with tests.
PINNED_HOST_QUALIFICATION_EXECUTOR_DESCRIPTOR_SHA256: Final = (
    "da7692691aee585b774a2d4a31ba7243d2f5ce005b9b31fe8ceb4a1993653bb8"
)


def matched_v3_host_qualification_executor_descriptor() -> dict[str, Any]:
    """Return the frozen, explicitly nonexecuting descriptor."""

    return copy.deepcopy(_DESCRIPTOR)


def canonical_matched_v3_host_qualification_executor_descriptor_bytes() -> bytes:
    """Return exact canonical descriptor bytes."""

    return bytes(_DESCRIPTOR_BYTES)


def matched_v3_host_qualification_executor_descriptor_sha256() -> str:
    """Return the computed descriptor identity after checking its literal pin."""

    observed = _sha256(_DESCRIPTOR_BYTES)
    if not hmac.compare_digest(observed, PINNED_HOST_QUALIFICATION_EXECUTOR_DESCRIPTOR_SHA256):
        _fail("host-qualification executor descriptor drifted from its literal pin")
    return observed


def parse_matched_v3_host_qualification_executor_descriptor(
    raw: bytes,
) -> dict[str, Any]:
    value = _strict_json(raw)
    validate_matched_v3_host_metadata_only_mapping(value)
    _validate_body_sha256(value, "descriptor_body_sha256", "executor descriptor")
    if not _exact_json_equal(value, _DESCRIPTOR):
        _fail("host-qualification executor descriptor content differs")
    if raw != _DESCRIPTOR_BYTES:
        _fail("host-qualification executor descriptor replay differs")
    matched_v3_host_qualification_executor_descriptor_sha256()
    return copy.deepcopy(value)


__all__ = [
    "CGROUP_COUNTER_ENDPOINTS",
    "CGROUP_COUNTER_SEMANTICS",
    "CGROUP_SAMPLE_PHASES",
    "FORBIDDEN_METADATA_KEYS",
    "ForagerMatchedV3HostQualificationExecutorError",
    "HOST_CGROUP_V2_BOUNDARY_PROOF_SCHEMA_VERSION",
    "HOST_CONTAINER_READY_SCHEMA_VERSION",
    "HOST_CONTAINER_TERMINAL_METADATA_SCHEMA_VERSION",
    "HOST_QUALIFICATION_CASE_FAILURE_SCHEMA_VERSION",
    "HOST_QUALIFICATION_CASE_INTENT_SCHEMA_VERSION",
    "HOST_QUALIFICATION_CASE_RECEIPT_SCHEMA_VERSION",
    "HOST_QUALIFICATION_CASE_REQUEST_SCHEMA_VERSION",
    "HOST_QUALIFICATION_EXECUTION_ACKNOWLEDGEMENT",
    "HOST_QUALIFICATION_EXECUTOR_DESCRIPTOR_SCHEMA_VERSION",
    "HOST_QUALIFICATION_GO_COMMITMENT_SCHEMA_VERSION",
    "HOST_QUALIFICATION_LIFECYCLE_PHASES",
    "HOST_QUALIFICATION_LIFECYCLE_RECORD_SCHEMA_VERSION",
    "HOST_QUALIFICATION_OBSERVATION_HANDOFF_SCHEMA_VERSION",
    "HOST_QUALIFICATION_PRE_HANDOFF_PHASES",
    "HOST_QUALIFICATION_PRE_RECEIPT_PHASES",
    "HOST_QUALIFICATION_UNCERTAINTY_KINDS",
    "HostCgroupV2CounterFdIdentity",
    "HostCgroupV2DescendantIdentity",
    "HostCgroupV2ProcessIdentity",
    "HostCgroupV2Sample",
    "MATCHED_V3_HORIZON",
    "MATCHED_V3_HOST_QUALIFICATION_CANDIDATE_IDS",
    "MatchedV3HostCgroupV2BoundaryProof",
    "MatchedV3HostContainerReadyMetadata",
    "MatchedV3HostContainerTerminalMetadata",
    "MatchedV3HostPublishedFileMetadata",
    "MatchedV3HostQualificationCaseExecutionReceipt",
    "MatchedV3HostQualificationCaseFailureReceipt",
    "MatchedV3HostQualificationCaseIntent",
    "MatchedV3HostQualificationCaseRequest",
    "MatchedV3HostQualificationGoCommitment",
    "MatchedV3HostQualificationLifecycleRecord",
    "MatchedV3HostQualificationObservationHandoff",
    "PINNED_ENDPOINT_RESOURCE_OBSERVER_DESCRIPTOR_SHA256",
    "PINNED_ENDPOINT_RESOURCE_OBSERVER_SOURCE_SHA256",
    "PINNED_HOST_QUALIFICATION_EXECUTOR_DESCRIPTOR_SHA256",
    "QUALIFICATION_OBSERVATION_REGISTRY_V1_SCHEMA_VERSION",
    "QUALIFICATION_PLAN_V2_SCHEMA_VERSION",
    "RESOURCE_CEILING_FIELDS",
    "STALE_BUILD_LINEAGES",
    "STALE_IMAGE_IDS",
    "build_matched_v3_host_qualification_case_intent",
    "build_matched_v3_host_qualification_case_execution_receipt",
    "build_matched_v3_host_qualification_case_failure_receipt",
    "build_matched_v3_host_qualification_go_commitment",
    "build_matched_v3_host_qualification_observation_handoff",
    "canonical_matched_v3_host_cgroup_v2_boundary_proof_bytes",
    "canonical_matched_v3_host_container_ready_metadata_bytes",
    "canonical_matched_v3_host_container_terminal_metadata_bytes",
    "canonical_matched_v3_host_qualification_case_execution_receipt_bytes",
    "canonical_matched_v3_host_qualification_case_failure_receipt_bytes",
    "canonical_matched_v3_host_qualification_case_intent_bytes",
    "canonical_matched_v3_host_qualification_case_request_bytes",
    "canonical_matched_v3_host_qualification_executor_descriptor_bytes",
    "canonical_matched_v3_host_qualification_go_commitment_bytes",
    "canonical_matched_v3_host_qualification_lifecycle_record_bytes",
    "canonical_matched_v3_host_qualification_observation_handoff_bytes",
    "classify_matched_v3_host_qualification_lifecycle",
    "matched_v3_host_cgroup_sample_sha256",
    "matched_v3_host_published_file_inventory_sha256",
    "matched_v3_host_qualification_executor_descriptor",
    "matched_v3_host_qualification_executor_descriptor_sha256",
    "matched_v3_host_qualification_readiness",
    "parse_matched_v3_host_cgroup_v2_boundary_proof",
    "parse_matched_v3_host_container_ready_metadata",
    "parse_matched_v3_host_container_terminal_metadata",
    "parse_matched_v3_host_qualification_case_execution_receipt",
    "parse_matched_v3_host_qualification_case_failure_receipt",
    "parse_matched_v3_host_qualification_case_intent",
    "parse_matched_v3_host_qualification_case_request",
    "parse_matched_v3_host_qualification_executor_descriptor",
    "parse_matched_v3_host_qualification_go_commitment",
    "parse_matched_v3_host_qualification_lifecycle_record",
    "parse_matched_v3_host_qualification_observation_handoff",
    "replay_matched_v3_host_qualification_case_request",
    "validate_matched_v3_host_metadata_only_mapping",
    "validate_matched_v3_host_qualification_case_execution_receipt_chain",
    "validate_matched_v3_host_qualification_case_failure_receipt_chain",
    "validate_matched_v3_host_qualification_observation_handoff_chain",
]
