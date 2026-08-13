"""Bounded, score-blind system-resource observer for matched-v3 qualification.

The observer takes two endpoint samples around an already-running process.  It
does not launch, stop, signal, wait, reap, or authorize that process.  The only
production reads are a fixed cgroup-v2/procfs allowlist; tests can inject a
reader with the same exact sample contract.  No output content, reward, score,
or ranking enters this module.

CPU and wall time are interval deltas.  A read-only endpoint observer cannot
prove that ``memory.peak``/``pids.peak`` was reset, that the cgroup was freshly
created, or that observation began at process launch.  Receipts therefore keep
those raw values and explicit blockers but never emit a complete 28-field
qualification resource observation or enforce the peak-memory ceiling.
"""

from __future__ import annotations

import copy
import hashlib
import hmac
import json
import os
import re
import stat
import threading
import time
import weakref
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Final, NoReturn, Protocol, cast

QUALIFICATION_RESOURCE_OBSERVER_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.qualification_resource_observer_descriptor.v1"
)
QUALIFICATION_RESOURCE_OBSERVATION_REQUEST_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.qualification_resource_observation_request.v1"
)
QUALIFICATION_RESOURCE_FINISH_REPORT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.qualification_resource_finish_report.v1"
)
QUALIFICATION_RESOURCE_OBSERVATION_RECEIPT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.qualification_resource_observation_receipt.v1"
)
QUALIFICATION_RESOURCE_OBSERVER_STATUS: Final = "implemented_uninvoked_non_authorizing"
QUALIFICATION_RESOURCE_OBSERVER_CLASSIFICATION: Final = (
    "score_blind_endpoint_system_resource_observer_non_authorizing"
)
PINNED_QUALIFICATION_RESOURCE_OBSERVER_DESCRIPTOR_SHA256: Final = (
    "e424201576200d05f5da31822cb59a5a61ef06ee29ec267cb20727e8e2e6bfb7"
)
MONOTONIC_CLOCK_ID: Final = "python_time_monotonic_ns_host_v1"

LINUX_CGROUP_V2_READER_KIND: Final = "linux_cgroup_v2_procfs_read_only_v1"
INJECTED_TEST_READER_KIND: Final = "injected_test_reader_v1"

_MAX_INTEGER: Final = 2**63 - 1
_MAX_ARTIFACT_BYTES: Final = 4 * 1024 * 1024
_MAX_TEXT_BYTES: Final = 16_384
_MAX_JSON_DEPTH: Final = 64
_MAX_JSON_NODES: Final = 100_000
_MAX_CONTROL_FILE_BYTES: Final = 1024 * 1024
_CGROUP_ROOT: Final = "/sys/fs/cgroup"
_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_PORTABLE_ID_RE: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}\Z")
_CGROUP_COMPONENT_RE: Final = re.compile(r"[A-Za-z0-9_.:@+-]{1,255}\Z")

_QUALIFICATION_PLAN_V2_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.qualification_plan_descriptor.v2"
)
_MATCHED_V3_HORIZON: Final = 499_712
_CANDIDATE_IDS: Final = (
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
_RESOURCE_FIELDS: Final = (
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
_OUTPUT_CEILING_FIELDS: Final = (
    "maximum_result_bytes",
    "maximum_stdout_bytes",
    "maximum_stderr_bytes",
)
_OPTIONAL_FIELDS: Final = (
    "memory.current",
    "memory.peak",
    "pids.current",
    "pids.peak",
    "memory.events:oom_kill",
)
_ALLOWLISTED_FILES: Final = (
    "cgroup.events",
    "cgroup.procs",
    "cpu.stat",
    "memory.current",
    "memory.events",
    "memory.peak",
    "pids.current",
    "pids.peak",
    "/proc/<exact-pid>/cgroup",
    "/proc/<exact-pid>/stat",
)
_QUALIFICATION_RESOURCE_OBSERVATION_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.resource_observation.v1"
)


class ForagerMatchedV3QualificationResourceObserverError(RuntimeError):
    """A resource request, sample, capability, counter, or receipt failed closed."""


def _fail(message: str) -> NoReturn:
    raise ForagerMatchedV3QualificationResourceObserverError(message)


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


def _require_optional_int(value: object, label: str) -> int | None:
    if value is None:
        return None
    return _require_int(value, label)


def _require_optional_bool(value: object, label: str) -> bool | None:
    if value is None:
        return None
    return _require_bool(value, label)


def _require_sha256(value: object, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        _fail(f"{label} must be one lowercase SHA-256")
    return value


def _require_identifier(value: object, label: str) -> str:
    if type(value) is not str or _PORTABLE_ID_RE.fullmatch(value) is None:
        _fail(f"{label} must be one bounded portable identifier")
    return value


def _require_phase(value: object) -> str:
    if type(value) is not str or value not in {"start", "finish"}:
        _fail("sample phase must be exactly start or finish")
    return value


def _require_exact_keys(value: object, expected: frozenset[str], label: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != expected:
        _fail(f"{label} keys differ")
    return cast(dict[str, Any], value)


def _reject_constant(value: str) -> NoReturn:
    _fail(f"JSON contains forbidden constant {value!r}")


def _reject_float(value: str) -> NoReturn:
    _fail(f"JSON contains forbidden float {value!r}")


def _parse_int(value: str) -> int:
    try:
        result = int(value, 10)
    except ValueError:
        _fail("JSON integer is invalid")
    return _require_int(result, "JSON integer", minimum=-_MAX_INTEGER)


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
        if nodes > _MAX_JSON_NODES or depth > _MAX_JSON_DEPTH:
            _fail("JSON structure exceeds its bound")
        if item is None or type(item) is bool:
            return
        if type(item) is int:
            _require_int(item, "JSON integer", minimum=-_MAX_INTEGER)
            return
        if type(item) is str:
            if len(item.encode("utf-8")) > _MAX_TEXT_BYTES:
                _fail("JSON string exceeds its byte bound")
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
                    _fail("JSON object key must be exact text")
                visit(key, depth + 1)
                visit(child, depth + 1)

    visit(value, 0)


def _exact_json_equal(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        left_map = cast(dict[object, object], left)
        right_map = cast(dict[object, object], right)
        return set(left_map) == set(right_map) and all(
            _exact_json_equal(left_map[key], right_map[key]) for key in left_map
        )
    if type(left) is list:
        left_list = cast(list[object], left)
        right_list = cast(list[object], right)
        return len(left_list) == len(right_list) and all(
            _exact_json_equal(a, b) for a, b in zip(left_list, right_list, strict=True)
        )
    return bool(left == right)


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    try:
        detached = copy.deepcopy(dict(value))
    except RecursionError as exc:
        raise ForagerMatchedV3QualificationResourceObserverError(
            "value exceeds the canonicalization recursion bound"
        ) from exc
    _assert_plain_unaliased_json(detached)
    try:
        raw = json.dumps(
            detached,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii") + b"\n"
    except (TypeError, ValueError, UnicodeError, RecursionError) as exc:
        raise ForagerMatchedV3QualificationResourceObserverError(
            "value is not canonicalizable JSON"
        ) from exc
    if len(raw) > _MAX_ARTIFACT_BYTES:
        _fail("canonical artifact exceeds its byte bound")
    return raw


def _strict_json_load(raw: bytes) -> dict[str, Any]:
    if (
        type(raw) is not bytes
        or not 1 <= len(raw) <= _MAX_ARTIFACT_BYTES
        or not raw.endswith(b"\n")
    ):
        _fail("artifact must be bounded exact bytes")
    try:
        text = raw.decode("ascii")
        value = json.loads(
            text,
            object_pairs_hook=_object_without_duplicates,
            parse_constant=_reject_constant,
            parse_float=_reject_float,
            parse_int=_parse_int,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ForagerMatchedV3QualificationResourceObserverError(
            "artifact is not strict ASCII JSON"
        ) from exc
    if type(value) is not dict:
        _fail("artifact root must be one object")
    _assert_plain_unaliased_json(value)
    result = cast(dict[str, Any], value)
    if not hmac.compare_digest(raw, _canonical_json(result)):
        _fail("artifact bytes are not canonical")
    return result


def _validate_pairs(
    value: object,
    *,
    fields: tuple[str, ...],
    label: str,
    positive: bool = False,
) -> tuple[tuple[str, int], ...]:
    if (
        type(value) is not tuple
        or any(type(item) is not tuple or len(item) != 2 for item in value)
        or tuple(item[0] for item in value) != fields
    ):
        _fail(f"{label} must use the exact frozen field order")
    result = cast(tuple[tuple[str, int], ...], value)
    for name, number in result:
        _require_int(number, f"{label} {name}", minimum=1 if positive else 0)
    return result


@dataclass(frozen=True, slots=True)
class MatchedV3QualificationResourceObservationRequest:
    """All predeclared identities and ceilings for one endpoint observation."""

    candidate_id: str
    qualification_case_id: str
    qualification_case_manifest_sha256: str
    qualification_plan_sha256: str
    resource_requirement_body_sha256: str
    executor_identity_sha256: str
    observation_nonce_sha256: str
    target_pid: int
    target_process_start_time_ticks: int
    cgroup_v2_path: str
    expected_cgroup_device: int
    expected_cgroup_inode: int
    reader_kind: str
    monotonic_clock_id: str
    declared_ceilings: tuple[tuple[str, int], ...]
    output_byte_ceilings: tuple[tuple[str, int], ...]
    attempt_ordinal: int

    def __post_init__(self) -> None:
        if self.candidate_id not in _CANDIDATE_IDS:
            _fail("request candidate is unknown")
        _require_identifier(self.qualification_case_id, "qualification case ID")
        for label, value in (
            ("qualification case manifest", self.qualification_case_manifest_sha256),
            ("qualification plan", self.qualification_plan_sha256),
            ("resource requirement body", self.resource_requirement_body_sha256),
            ("executor identity", self.executor_identity_sha256),
            ("observation nonce", self.observation_nonce_sha256),
        ):
            _require_sha256(value, label)
        _require_int(self.target_pid, "target PID", minimum=1)
        _require_int(self.target_process_start_time_ticks, "target process start time", minimum=1)
        _validate_cgroup_path(self.cgroup_v2_path)
        _require_int(self.expected_cgroup_device, "expected cgroup device")
        _require_int(self.expected_cgroup_inode, "expected cgroup inode", minimum=1)
        if self.reader_kind not in {LINUX_CGROUP_V2_READER_KIND, INJECTED_TEST_READER_KIND}:
            _fail("request reader kind is unsupported")
        if self.monotonic_clock_id != MONOTONIC_CLOCK_ID:
            _fail("request monotonic clock identity differs")
        ceilings = _validate_pairs(
            self.declared_ceilings,
            fields=_RESOURCE_FIELDS,
            label="declared resource ceilings",
        )
        outputs = _validate_pairs(
            self.output_byte_ceilings,
            fields=_OUTPUT_CEILING_FIELDS,
            label="output byte ceilings",
            positive=True,
        )
        ceiling_map = dict(ceilings)
        if ceiling_map["max_environment_interactions"] < _MATCHED_V3_HORIZON:
            _fail("resource ceilings cannot cover the matched-v3 horizon")
        if ceiling_map["max_thread_count"] < 1 or ceiling_map["max_attempt_count"] < 1:
            _fail("thread and attempt ceilings must be positive")
        if ceiling_map["max_failure_count"] >= ceiling_map["max_attempt_count"]:
            _fail("failure ceiling must be below attempt ceiling")
        ordinal = _require_int(self.attempt_ordinal, "attempt ordinal")
        if ordinal >= ceiling_map["max_attempt_count"]:
            _fail("attempt ordinal exceeds the predeclared attempt ceiling")
        if dict(outputs).keys() != dict(self.output_byte_ceilings).keys():
            _fail("output ceiling mapping aliased")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": QUALIFICATION_RESOURCE_OBSERVATION_REQUEST_SCHEMA_VERSION,
            "descriptor_sha256": matched_v3_qualification_resource_observer_descriptor_sha256(),
            "candidate_id": self.candidate_id,
            "qualification_case_id": self.qualification_case_id,
            "qualification_case_manifest_sha256": self.qualification_case_manifest_sha256,
            "qualification_plan_sha256": self.qualification_plan_sha256,
            "resource_requirement_body_sha256": self.resource_requirement_body_sha256,
            "executor_identity_sha256": self.executor_identity_sha256,
            "observation_nonce_sha256": self.observation_nonce_sha256,
            "target_pid": self.target_pid,
            "target_process_start_time_ticks": self.target_process_start_time_ticks,
            "cgroup_v2_path": self.cgroup_v2_path,
            "expected_cgroup_device": self.expected_cgroup_device,
            "expected_cgroup_inode": self.expected_cgroup_inode,
            "reader_kind": self.reader_kind,
            "monotonic_clock_id": self.monotonic_clock_id,
            "declared_ceilings": dict(self.declared_ceilings),
            "output_byte_ceilings": dict(self.output_byte_ceilings),
            "attempt_ordinal": self.attempt_ordinal,
            "authority": {
                "execution_authorized": False,
                "qualification_granted": False,
            },
        }


@dataclass(frozen=True, slots=True)
class MatchedV3QualificationResourceFinishReport:
    """Bounded executor-reported process state and output byte counts, never content."""

    request_sha256: str
    executor_finish_receipt_sha256: str
    candidate_id: str
    qualification_case_id: str
    target_pid: int
    returncode: int
    timed_out: bool
    process_group_termination_requested: bool
    process_group_termination_succeeded: bool
    direct_child_waited: bool
    direct_child_reaped: bool
    descendant_cleanup_reported_complete: bool
    cleanup_deadline_expired: bool
    output_counts_complete: bool
    stdout_size_bytes: int
    stderr_size_bytes: int
    result_size_bytes: int

    def __post_init__(self) -> None:
        _require_sha256(self.request_sha256, "finish report request")
        _require_sha256(self.executor_finish_receipt_sha256, "executor finish receipt")
        if self.candidate_id not in _CANDIDATE_IDS:
            _fail("finish report candidate is unknown")
        _require_identifier(self.qualification_case_id, "finish report qualification case")
        _require_int(self.target_pid, "finish report target PID", minimum=1)
        _require_int(self.returncode, "finish report return code", minimum=-_MAX_INTEGER)
        for label, value in (
            ("timed out", self.timed_out),
            ("termination requested", self.process_group_termination_requested),
            ("termination succeeded", self.process_group_termination_succeeded),
            ("direct child waited", self.direct_child_waited),
            ("direct child reaped", self.direct_child_reaped),
            ("descendant cleanup", self.descendant_cleanup_reported_complete),
            ("cleanup deadline", self.cleanup_deadline_expired),
            ("output counts complete", self.output_counts_complete),
        ):
            _require_bool(value, f"finish report {label}")
        for label, integer_value in (
            ("stdout size", self.stdout_size_bytes),
            ("stderr size", self.stderr_size_bytes),
            ("result size", self.result_size_bytes),
        ):
            _require_int(integer_value, f"finish report {label}")
        if (
            self.process_group_termination_succeeded
            and not self.process_group_termination_requested
        ):
            _fail("successful termination requires a request")
        if self.direct_child_reaped and not self.direct_child_waited:
            _fail("reaped direct child must have been waited")
        if self.timed_out and not self.process_group_termination_requested:
            _fail("timed-out process must have a termination request")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": QUALIFICATION_RESOURCE_FINISH_REPORT_SCHEMA_VERSION,
            **{
                name: getattr(self, name)
                for name in self.__dataclass_fields__
            },
        }


@dataclass(frozen=True, slots=True)
class ResourceReaderSample:
    """One exact cgroup/proc endpoint sample with explicit unsupported fields."""

    phase: str
    reader_kind: str
    target_pid: int
    target_proc_present: bool
    target_process_start_time_ticks: int | None
    target_proc_cgroup_exact: bool | None
    cgroup_v2_path: str
    cgroup_device: int
    cgroup_inode: int
    cgroup_cpu_usage_usec: int
    memory_current_bytes: int | None
    memory_peak_bytes: int | None
    pids_current: int | None
    pids_peak: int | None
    oom_kill_count: int | None
    cgroup_populated: bool
    cgroup_member_pids: tuple[int, ...]
    unsupported_fields: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_phase(self.phase)
        if self.reader_kind not in {LINUX_CGROUP_V2_READER_KIND, INJECTED_TEST_READER_KIND}:
            _fail("sample reader kind is unsupported")
        _require_int(self.target_pid, "sample target PID", minimum=1)
        _require_bool(self.target_proc_present, "sample target process presence")
        _require_optional_int(self.target_process_start_time_ticks, "sample process start time")
        _require_optional_bool(self.target_proc_cgroup_exact, "sample process cgroup membership")
        if self.target_proc_present:
            if (
                self.target_process_start_time_ticks is None
                or self.target_proc_cgroup_exact is None
            ):
                _fail("present target process requires exact proc identity fields")
        elif (
            self.target_process_start_time_ticks is not None
            or self.target_proc_cgroup_exact is not None
        ):
            _fail("absent target process cannot carry proc identity fields")
        _validate_cgroup_path(self.cgroup_v2_path)
        _require_int(self.cgroup_device, "sample cgroup device")
        _require_int(self.cgroup_inode, "sample cgroup inode", minimum=1)
        _require_int(self.cgroup_cpu_usage_usec, "sample cgroup CPU counter")
        optional_values = (
            self.memory_current_bytes,
            self.memory_peak_bytes,
            self.pids_current,
            self.pids_peak,
            self.oom_kill_count,
        )
        for label, value in zip(_OPTIONAL_FIELDS, optional_values, strict=True):
            _require_optional_int(value, f"sample {label}")
        _require_bool(self.cgroup_populated, "sample cgroup populated")
        if (
            type(self.cgroup_member_pids) is not tuple
            or any(type(pid) is not int for pid in self.cgroup_member_pids)
        ):
            _fail("sample cgroup member PIDs must be one exact tuple")
        for pid in self.cgroup_member_pids:
            _require_int(pid, "sample cgroup member PID", minimum=1)
        if tuple(sorted(set(self.cgroup_member_pids))) != self.cgroup_member_pids:
            _fail("sample cgroup member PIDs must be sorted and unique")
        if not self.cgroup_populated and self.cgroup_member_pids:
            _fail("unpopulated cgroup cannot have direct member PIDs")
        if (
            type(self.unsupported_fields) is not tuple
            or any(type(item) is not str for item in self.unsupported_fields)
            or tuple(item for item in _OPTIONAL_FIELDS if item in self.unsupported_fields)
            != self.unsupported_fields
            or len(set(self.unsupported_fields)) != len(self.unsupported_fields)
        ):
            _fail("sample unsupported fields must be an ordered exact subset")
        for label, value in zip(_OPTIONAL_FIELDS, optional_values, strict=True):
            if (value is None) is not (label in self.unsupported_fields):
                _fail("sample optional value and unsupported-field state differ")

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "reader_kind": self.reader_kind,
            "target_pid": self.target_pid,
            "target_proc_present": self.target_proc_present,
            "target_process_start_time_ticks": self.target_process_start_time_ticks,
            "target_proc_cgroup_exact": self.target_proc_cgroup_exact,
            "cgroup_v2_path": self.cgroup_v2_path,
            "cgroup_device": self.cgroup_device,
            "cgroup_inode": self.cgroup_inode,
            "cgroup_cpu_usage_usec": self.cgroup_cpu_usage_usec,
            "memory_current_bytes": self.memory_current_bytes,
            "memory_peak_bytes": self.memory_peak_bytes,
            "pids_current": self.pids_current,
            "pids_peak": self.pids_peak,
            "oom_kill_count": self.oom_kill_count,
            "cgroup_populated": self.cgroup_populated,
            "cgroup_member_pids": list(self.cgroup_member_pids),
            "unsupported_fields": list(self.unsupported_fields),
        }


class MatchedV3QualificationResourceReader(Protocol):
    """Reader protocol; implementations return content-free integer samples only."""

    reader_kind: str

    def sample(
        self,
        *,
        phase: str,
        target_pid: int,
        expected_start_time_ticks: int,
        cgroup_v2_path: str,
    ) -> ResourceReaderSample: ...


def _validate_cgroup_path(value: object) -> str:
    if type(value) is not str or not value.startswith(f"{_CGROUP_ROOT}/"):
        _fail("cgroup path must be an absolute child beneath /sys/fs/cgroup")
    if len(value.encode("utf-8")) > 4096:
        _fail("cgroup path exceeds its byte bound")
    components = value.removeprefix(f"{_CGROUP_ROOT}/").split("/")
    if not components or any(_CGROUP_COMPONENT_RE.fullmatch(item) is None for item in components):
        _fail("cgroup path has a forbidden or aliased component")
    return value


def _read_fd_bounded(descriptor: int, *, maximum_bytes: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        try:
            chunk = os.read(descriptor, min(65_536, maximum_bytes + 1 - total))
        except OSError as exc:
            raise ForagerMatchedV3QualificationResourceObserverError(
                "allowlisted control file could not be read"
            ) from exc
        if not chunk:
            return b"".join(chunks)
        total += len(chunk)
        if total > maximum_bytes:
            _fail("allowlisted control file exceeds its byte bound")
        chunks.append(chunk)


def _read_at(
    directory_fd: int,
    name: str,
    *,
    optional: bool = False,
) -> bytes | None:
    flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=directory_fd)
    except FileNotFoundError:
        if optional:
            return None
        _fail(f"required allowlisted control file is missing: {name}")
    except OSError as exc:
        raise ForagerMatchedV3QualificationResourceObserverError(
            f"allowlisted control file could not be opened: {name}"
        ) from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            _fail(f"allowlisted control endpoint is not regular: {name}")
        return _read_fd_bounded(descriptor, maximum_bytes=_MAX_CONTROL_FILE_BYTES)
    finally:
        os.close(descriptor)


def _ascii_lines(raw: bytes, label: str) -> list[str]:
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ForagerMatchedV3QualificationResourceObserverError(
            f"{label} is not ASCII"
        ) from exc
    if text and not text.endswith("\n"):
        _fail(f"{label} lacks a final newline")
    return text.splitlines()


def _parse_single_integer(raw: bytes, label: str) -> int:
    lines = _ascii_lines(raw, label)
    if len(lines) != 1 or re.fullmatch(r"[0-9]+", lines[0]) is None:
        _fail(f"{label} is not one exact unsigned integer")
    return _require_int(int(lines[0]), label)


def _parse_keyed_integers(raw: bytes, label: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for line in _ascii_lines(raw, label):
        fields = line.split(" ")
        if len(fields) != 2 or not fields[0] or re.fullmatch(r"[0-9]+", fields[1]) is None:
            _fail(f"{label} contains a malformed counter")
        if fields[0] in result:
            _fail(f"{label} contains a duplicate counter")
        result[fields[0]] = _require_int(int(fields[1]), f"{label} {fields[0]}")
    return result


def _parse_pid_lines(raw: bytes) -> tuple[int, ...]:
    values: list[int] = []
    for line in _ascii_lines(raw, "cgroup.procs"):
        if re.fullmatch(r"[0-9]+", line) is None:
            _fail("cgroup.procs contains a malformed PID")
        values.append(_require_int(int(line), "cgroup member PID", minimum=1))
    if len(values) != len(set(values)):
        _fail("cgroup.procs contains a duplicate PID")
    return tuple(sorted(values))


def _parse_proc_start_time(raw: bytes, target_pid: int) -> int:
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ForagerMatchedV3QualificationResourceObserverError(
            "proc stat is not ASCII"
        ) from exc
    if not text.endswith("\n") or not text.startswith(f"{target_pid} ("):
        _fail("proc stat target PID framing differs")
    close = text.rfind(")")
    if close < len(str(target_pid)) + 2:
        _fail("proc stat command framing differs")
    fields = text[close + 2 :].removesuffix("\n").split(" ")
    if len(fields) < 20 or any(not field for field in fields):
        _fail("proc stat field count differs")
    if re.fullmatch(r"[0-9]+", fields[19]) is None:
        _fail("proc stat start-time field is invalid")
    return _require_int(int(fields[19]), "proc stat start time", minimum=1)


def _proc_cgroup_exact(raw: bytes, cgroup_v2_path: str) -> bool:
    expected = cgroup_v2_path.removeprefix(_CGROUP_ROOT)
    lines = _ascii_lines(raw, "proc cgroup")
    return lines == [f"0::{expected}"]


class LinuxCgroupV2ResourceReader:
    """Read-only production reader for one exact cgroup-v2 and proc PID boundary."""

    __slots__ = ()

    reader_kind = LINUX_CGROUP_V2_READER_KIND

    def sample(
        self,
        *,
        phase: str,
        target_pid: int,
        expected_start_time_ticks: int,
        cgroup_v2_path: str,
    ) -> ResourceReaderSample:
        exact_phase = _require_phase(phase)
        exact_pid = _require_int(target_pid, "reader target PID", minimum=1)
        _require_int(expected_start_time_ticks, "reader expected process start time", minimum=1)
        exact_path = _validate_cgroup_path(cgroup_v2_path)
        if os.path.realpath(exact_path) != exact_path:
            _fail("cgroup path resolves through an alias or symlink")
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
        try:
            cgroup_fd = os.open(exact_path, flags)
        except OSError as exc:
            raise ForagerMatchedV3QualificationResourceObserverError(
                "exact cgroup directory could not be opened"
            ) from exc
        try:
            metadata = os.fstat(cgroup_fd)
            if not stat.S_ISDIR(metadata.st_mode):
                _fail("exact cgroup endpoint is not a directory")
            events = _parse_keyed_integers(
                cast(bytes, _read_at(cgroup_fd, "cgroup.events")), "cgroup.events"
            )
            if "populated" not in events or events["populated"] not in {0, 1}:
                _fail("cgroup.events populated field differs")
            members = _parse_pid_lines(cast(bytes, _read_at(cgroup_fd, "cgroup.procs")))
            cpu = _parse_keyed_integers(cast(bytes, _read_at(cgroup_fd, "cpu.stat")), "cpu.stat")
            if "usage_usec" not in cpu:
                _fail("cpu.stat lacks usage_usec")

            optional_values: dict[str, int | None] = {}
            unsupported: list[str] = []
            for filename, label in (
                ("memory.current", "memory.current"),
                ("memory.peak", "memory.peak"),
                ("pids.current", "pids.current"),
                ("pids.peak", "pids.peak"),
            ):
                raw = _read_at(cgroup_fd, filename, optional=True)
                if raw is None:
                    optional_values[label] = None
                    unsupported.append(label)
                else:
                    optional_values[label] = _parse_single_integer(raw, label)
            memory_events_raw = _read_at(cgroup_fd, "memory.events", optional=True)
            if memory_events_raw is None:
                optional_values["memory.events:oom_kill"] = None
                unsupported.append("memory.events:oom_kill")
            else:
                memory_events = _parse_keyed_integers(memory_events_raw, "memory.events")
                if "oom_kill" not in memory_events:
                    optional_values["memory.events:oom_kill"] = None
                    unsupported.append("memory.events:oom_kill")
                else:
                    optional_values["memory.events:oom_kill"] = memory_events["oom_kill"]
        finally:
            os.close(cgroup_fd)

        proc_present = False
        proc_start: int | None = None
        proc_cgroup: bool | None = None
        proc_path = f"/proc/{exact_pid}"
        try:
            proc_fd = os.open(proc_path, flags)
        except (FileNotFoundError, ProcessLookupError):
            proc_fd = -1
        except OSError as exc:
            raise ForagerMatchedV3QualificationResourceObserverError(
                "exact proc directory could not be opened"
            ) from exc
        if proc_fd >= 0:
            try:
                proc_start = _parse_proc_start_time(
                    cast(bytes, _read_at(proc_fd, "stat")), exact_pid
                )
                proc_cgroup = _proc_cgroup_exact(
                    cast(bytes, _read_at(proc_fd, "cgroup")), exact_path
                )
                proc_present = True
            finally:
                os.close(proc_fd)
        return ResourceReaderSample(
            phase=exact_phase,
            reader_kind=self.reader_kind,
            target_pid=exact_pid,
            target_proc_present=proc_present,
            target_process_start_time_ticks=proc_start,
            target_proc_cgroup_exact=proc_cgroup,
            cgroup_v2_path=exact_path,
            cgroup_device=metadata.st_dev,
            cgroup_inode=metadata.st_ino,
            cgroup_cpu_usage_usec=cpu["usage_usec"],
            memory_current_bytes=optional_values["memory.current"],
            memory_peak_bytes=optional_values["memory.peak"],
            pids_current=optional_values["pids.current"],
            pids_peak=optional_values["pids.peak"],
            oom_kill_count=optional_values["memory.events:oom_kill"],
            cgroup_populated=events["populated"] == 1,
            cgroup_member_pids=members,
            unsupported_fields=tuple(item for item in _OPTIONAL_FIELDS if item in unsupported),
        )


def _descriptor() -> dict[str, Any]:
    return {
        "schema_version": QUALIFICATION_RESOURCE_OBSERVER_DESCRIPTOR_SCHEMA_VERSION,
        "status": QUALIFICATION_RESOURCE_OBSERVER_STATUS,
        "classification": QUALIFICATION_RESOURCE_OBSERVER_CLASSIFICATION,
        "schemas": {
            "request": QUALIFICATION_RESOURCE_OBSERVATION_REQUEST_SCHEMA_VERSION,
            "finish_report": QUALIFICATION_RESOURCE_FINISH_REPORT_SCHEMA_VERSION,
            "receipt": QUALIFICATION_RESOURCE_OBSERVATION_RECEIPT_SCHEMA_VERSION,
        },
        "qualification_binding": {
            "plan_schema_version": _QUALIFICATION_PLAN_V2_SCHEMA_VERSION,
            "resource_observation_schema_version": (
                _QUALIFICATION_RESOURCE_OBSERVATION_SCHEMA_VERSION
            ),
            "candidate_order": list(_CANDIDATE_IDS),
            "resource_ceiling_fields": list(_RESOURCE_FIELDS),
            "output_byte_ceiling_fields": list(_OUTPUT_CEILING_FIELDS),
            "complete_resource_observation_produced": False,
        },
        "production_reader": {
            "reader_kind": LINUX_CGROUP_V2_READER_KIND,
            "filesystem_roots": [_CGROUP_ROOT, "/proc/<exact-pid>"],
            "allowlisted_files": list(_ALLOWLISTED_FILES),
            "writes_allowed": False,
            "cgroup_v1_fallback": False,
            "rusage_fallback": False,
            "endpoint_directory_identity_checked": True,
            "continuous_path_immutability_attested": False,
        },
        "measurement_semantics": {
            "cpu_time": "cgroup_v2_cpu_stat_usage_usec_endpoint_delta_to_ns",
            "wall_time": "caller_pinned_monotonic_ns_endpoint_delta",
            "memory_current": "cgroup_v2_memory_current_endpoint_values",
            "memory_peak": (
                "raw_cgroup_v2_memory_peak_not_enforced_without_reset_or_freshness_proof"
            ),
            "task_peak": "raw_cgroup_v2_pids_peak_not_enforced_without_reset_or_freshness_proof",
            "outputs": "executor_reported_integer_byte_counts_only",
            "process_state": "cgroup_population_observed_executor_exit_cleanup_state_reported",
            "whole_process_boundary_proven": False,
            "current_and_peak_distinguished": True,
            "continuous_cgroup_membership_attested": False,
            "control_file_snapshot_atomic": False,
        },
        "blockers": [
            "read_only_memory_peak_freshness_or_reset_not_proven",
            "whole_process_launch_boundary_not_observed",
            "continuous_cgroup_membership_between_endpoints_not_attested",
            "nested_child_cgroup_exclusivity_not_attested",
            "multi_file_endpoint_sample_not_atomic",
            "endpoint_samples_do_not_prove_continuous_thread_or_storage_peaks",
            "executor_reported_output_and_process_state_not_independently_observed",
            "remaining_qualification_resource_fields_not_observed",
        ],
        "prohibitions": {
            "performance_payload_input": True,
            "output_content_read": True,
            "result_content_enumeration": True,
            "post_observation_ceiling_retuning": True,
            "acceptance_inference": True,
            "qualification_inference": True,
        },
        "capabilities": {
            "clock_default": False,
            "default_inputs": False,
            "filesystem_write": False,
            "network": False,
            "process_launcher": False,
            "process_signaler": False,
            "process_waiter": False,
            "reader_injection": True,
        },
        "claims": {
            "acceptance_evaluated": False,
            "execution_authorized": False,
            "full_28_field_resource_accounting_complete": False,
            "peak_memory_ceiling_enforced": False,
            "qualification_granted": False,
            "runtime_qualified": False,
        },
    }


_FROZEN_DESCRIPTOR: Final = _descriptor()


def matched_v3_qualification_resource_observer_descriptor() -> dict[str, Any]:
    """Return detached, nonauthorizing observer contract content."""

    return copy.deepcopy(_FROZEN_DESCRIPTOR)


def canonical_matched_v3_qualification_resource_observer_descriptor_bytes() -> bytes:
    """Return the exact canonical descriptor bytes."""

    return _canonical_json(_FROZEN_DESCRIPTOR)


def matched_v3_qualification_resource_observer_descriptor_sha256() -> str:
    """Return the exact descriptor file digest."""

    return hashlib.sha256(
        canonical_matched_v3_qualification_resource_observer_descriptor_bytes()
    ).hexdigest()


if (
    matched_v3_qualification_resource_observer_descriptor_sha256()
    != PINNED_QUALIFICATION_RESOURCE_OBSERVER_DESCRIPTOR_SHA256
):
    raise AssertionError("matched-v3 qualification resource observer descriptor drifted")


def parse_matched_v3_qualification_resource_observer_descriptor(raw: bytes) -> dict[str, Any]:
    """Strictly replay the one frozen descriptor."""

    value = _strict_json_load(raw)
    if not _exact_json_equal(value, _FROZEN_DESCRIPTOR):
        _fail("resource observer descriptor content differs")
    return copy.deepcopy(value)


def _request_from_dict(value: object) -> MatchedV3QualificationResourceObservationRequest:
    keys = frozenset(
        {
            "schema_version",
            "descriptor_sha256",
            "candidate_id",
            "qualification_case_id",
            "qualification_case_manifest_sha256",
            "qualification_plan_sha256",
            "resource_requirement_body_sha256",
            "executor_identity_sha256",
            "observation_nonce_sha256",
            "target_pid",
            "target_process_start_time_ticks",
            "cgroup_v2_path",
            "expected_cgroup_device",
            "expected_cgroup_inode",
            "reader_kind",
            "monotonic_clock_id",
            "declared_ceilings",
            "output_byte_ceilings",
            "attempt_ordinal",
            "authority",
        }
    )
    item = dict(_require_exact_keys(value, keys, "resource observation request"))
    if item.pop("schema_version") != QUALIFICATION_RESOURCE_OBSERVATION_REQUEST_SCHEMA_VERSION:
        _fail("resource observation request schema differs")
    if (
        item.pop("descriptor_sha256")
        != matched_v3_qualification_resource_observer_descriptor_sha256()
    ):
        _fail("resource observation request descriptor binding differs")
    authority = _require_exact_keys(
        item.pop("authority"),
        frozenset({"execution_authorized", "qualification_granted"}),
        "resource observation request authority",
    )
    if not _exact_json_equal(
        authority,
        {"execution_authorized": False, "qualification_granted": False},
    ):
        _fail("resource observation request claims authority")
    declared = _require_exact_keys(
        item.pop("declared_ceilings"), frozenset(_RESOURCE_FIELDS), "declared ceilings"
    )
    outputs = _require_exact_keys(
        item.pop("output_byte_ceilings"),
        frozenset(_OUTPUT_CEILING_FIELDS),
        "output ceilings",
    )
    result = MatchedV3QualificationResourceObservationRequest(
        **item,
        declared_ceilings=tuple((name, declared[name]) for name in _RESOURCE_FIELDS),
        output_byte_ceilings=tuple((name, outputs[name]) for name in _OUTPUT_CEILING_FIELDS),
    )
    if not _exact_json_equal(result.to_dict(), value):
        _fail("resource observation request semantic replay differs")
    return result


def canonical_matched_v3_qualification_resource_observation_request_bytes(
    request: MatchedV3QualificationResourceObservationRequest,
) -> bytes:
    """Canonicalize one exact caller-supplied request."""

    if type(request) is not MatchedV3QualificationResourceObservationRequest:
        _fail("resource observation request type differs")
    return _canonical_json(request.to_dict())


def matched_v3_qualification_resource_observation_request_sha256(
    request: MatchedV3QualificationResourceObservationRequest,
) -> str:
    """Digest one exact canonical request."""

    return hashlib.sha256(
        canonical_matched_v3_qualification_resource_observation_request_bytes(request)
    ).hexdigest()


def parse_matched_v3_qualification_resource_observation_request(
    raw: bytes,
) -> MatchedV3QualificationResourceObservationRequest:
    """Parse one strict canonical request without issuing a capability."""

    return _request_from_dict(_strict_json_load(raw))


def replay_matched_v3_qualification_resource_observation_request(
    raw: bytes,
    *,
    expected_request_sha256: str,
) -> MatchedV3QualificationResourceObservationRequest:
    """Replay a request against one caller-supplied full-file digest."""

    expected = _require_sha256(expected_request_sha256, "expected request")
    if not hmac.compare_digest(hashlib.sha256(raw).hexdigest(), expected):
        _fail("resource observation request digest differs")
    return parse_matched_v3_qualification_resource_observation_request(raw)


def _finish_report_from_dict(value: object) -> MatchedV3QualificationResourceFinishReport:
    keys = frozenset(
        {"schema_version", *MatchedV3QualificationResourceFinishReport.__dataclass_fields__}
    )
    item = dict(_require_exact_keys(value, keys, "resource finish report"))
    if item.pop("schema_version") != QUALIFICATION_RESOURCE_FINISH_REPORT_SCHEMA_VERSION:
        _fail("resource finish report schema differs")
    result = MatchedV3QualificationResourceFinishReport(**item)
    if not _exact_json_equal(result.to_dict(), value):
        _fail("resource finish report semantic replay differs")
    return result


def canonical_matched_v3_qualification_resource_finish_report_bytes(
    report: MatchedV3QualificationResourceFinishReport,
) -> bytes:
    """Canonicalize one exact content-free executor finish report."""

    if type(report) is not MatchedV3QualificationResourceFinishReport:
        _fail("resource finish report type differs")
    return _canonical_json(report.to_dict())


def parse_matched_v3_qualification_resource_finish_report(
    raw: bytes,
) -> MatchedV3QualificationResourceFinishReport:
    """Parse one strict finish report."""

    return _finish_report_from_dict(_strict_json_load(raw))


def _sample_from_dict(value: object) -> ResourceReaderSample:
    keys = frozenset(ResourceReaderSample.__dataclass_fields__)
    item = dict(_require_exact_keys(value, keys, "resource reader sample"))
    members = item.pop("cgroup_member_pids")
    unsupported = item.pop("unsupported_fields")
    if type(members) is not list or type(unsupported) is not list:
        _fail("resource sample sequence fields differ")
    result = ResourceReaderSample(
        **item,
        cgroup_member_pids=tuple(members),
        unsupported_fields=tuple(unsupported),
    )
    if not _exact_json_equal(result.to_dict(), value):
        _fail("resource reader sample semantic replay differs")
    return result


def _validate_sample_against_request(
    sample: ResourceReaderSample,
    *,
    request: MatchedV3QualificationResourceObservationRequest,
    phase: str,
) -> None:
    if type(sample) is not ResourceReaderSample:
        _fail("reader returned a nonexact sample type")
    if sample.phase != phase:
        _fail("sample phase differs")
    if sample.reader_kind != request.reader_kind:
        _fail("sample reader kind differs")
    if sample.target_pid != request.target_pid:
        _fail("sample target PID differs")
    if sample.cgroup_v2_path != request.cgroup_v2_path:
        _fail("sample cgroup path differs")
    if (
        sample.cgroup_device != request.expected_cgroup_device
        or sample.cgroup_inode != request.expected_cgroup_inode
    ):
        _fail("sample cgroup identity drifted")
    if sample.target_proc_present:
        if sample.target_process_start_time_ticks != request.target_process_start_time_ticks:
            if phase == "finish":
                _fail("target PID was reused before finish observation")
            _fail("target process start identity differs")
        if sample.target_proc_cgroup_exact is not True:
            _fail("target process cgroup membership differs")
    elif phase == "start":
        _fail("target process is absent at the start boundary")
    if phase == "start":
        if not sample.cgroup_populated or sample.cgroup_member_pids != (request.target_pid,):
            _fail("start boundary requires a dedicated singleton cgroup")


def _validate_finish_report_against_request(
    finish_report: MatchedV3QualificationResourceFinishReport,
    request: MatchedV3QualificationResourceObservationRequest,
) -> None:
    request_sha = matched_v3_qualification_resource_observation_request_sha256(request)
    if finish_report.request_sha256 != request_sha:
        _fail("finish report request digest differs")
    if finish_report.candidate_id != request.candidate_id:
        _fail("finish report candidate differs")
    if finish_report.qualification_case_id != request.qualification_case_id:
        _fail("finish report qualification case differs")
    if finish_report.target_pid != request.target_pid:
        _fail("finish report target PID differs")


@dataclass(slots=True)
class _CapabilityState:
    issuer_pid: int
    request: MatchedV3QualificationResourceObservationRequest
    reader: MatchedV3QualificationResourceReader
    monotonic_ns: Callable[[], int]
    start_monotonic_ns: int
    start_sample: ResourceReaderSample


class _ResourceObservationCapability:
    __slots__ = ("__weakref__",)

    def __repr__(self) -> str:
        return "<matched-v3 qualification resource observation capability>"

    def __copy__(self) -> NoReturn:
        raise TypeError("resource observation capabilities cannot be copied")

    def __deepcopy__(self, _memo: object) -> NoReturn:
        raise TypeError("resource observation capabilities cannot be deep-copied")

    def __reduce__(self) -> NoReturn:
        raise TypeError("resource observation capabilities cannot be serialized")


_CAPABILITY_LOCK: Final = threading.Lock()
_CAPABILITY_STATES: Final[
    weakref.WeakKeyDictionary[_ResourceObservationCapability, _CapabilityState]
] = weakref.WeakKeyDictionary()


def _current_pid() -> int:
    return os.getpid()


def _clock_read(clock: Callable[[], int], label: str) -> int:
    try:
        value = clock()
    except BaseException as exc:
        raise ForagerMatchedV3QualificationResourceObserverError(
            f"{label} monotonic clock failed"
        ) from exc
    return _require_int(value, f"{label} monotonic clock")


def issue_matched_v3_qualification_resource_observation_capability(
    *,
    request: MatchedV3QualificationResourceObservationRequest,
    reader: MatchedV3QualificationResourceReader,
    monotonic_ns: Callable[[], int],
) -> object:
    """Sample the start endpoint and issue one process-bound single-use capability."""

    if type(request) is not MatchedV3QualificationResourceObservationRequest:
        _fail("resource observation request type differs")
    if not callable(monotonic_ns):
        _fail("monotonic clock must be one explicit callable")
    try:
        reader_kind = reader.reader_kind
    except (AttributeError, TypeError) as exc:
        raise ForagerMatchedV3QualificationResourceObserverError(
            "resource reader lacks an exact kind"
        ) from exc
    if type(reader_kind) is not str or reader_kind != request.reader_kind:
        _fail("reader kind differs from the predeclared request")
    if request.reader_kind == LINUX_CGROUP_V2_READER_KIND:
        if type(reader) is not LinuxCgroupV2ResourceReader:
            _fail("production reader kind requires the exact production reader type")
        if monotonic_ns is not time.monotonic_ns:
            _fail("production observation requires exact time.monotonic_ns")
    elif type(reader) is LinuxCgroupV2ResourceReader:
        _fail("production reader cannot claim the injected-test reader kind")
    start_clock = _clock_read(monotonic_ns, "start")
    try:
        start = reader.sample(
            phase="start",
            target_pid=request.target_pid,
            expected_start_time_ticks=request.target_process_start_time_ticks,
            cgroup_v2_path=request.cgroup_v2_path,
        )
    except ForagerMatchedV3QualificationResourceObserverError:
        raise
    except BaseException as exc:
        raise ForagerMatchedV3QualificationResourceObserverError(
            "resource reader start sample failed"
        ) from exc
    _validate_sample_against_request(start, request=request, phase="start")
    capability = _ResourceObservationCapability()
    state = _CapabilityState(
        issuer_pid=_current_pid(),
        request=request,
        reader=reader,
        monotonic_ns=monotonic_ns,
        start_monotonic_ns=start_clock,
        start_sample=start,
    )
    with _CAPABILITY_LOCK:
        _CAPABILITY_STATES[capability] = state
    return capability


def _check_optional_support_and_counters(
    start: ResourceReaderSample,
    finish: ResourceReaderSample,
) -> None:
    if start.unsupported_fields != finish.unsupported_fields:
        _fail("optional resource support changed between endpoint samples")
    for label, start_value, finish_value, message in (
        (
            "memory.peak",
            start.memory_peak_bytes,
            finish.memory_peak_bytes,
            "memory peak counter rolled back",
        ),
        ("pids.peak", start.pids_peak, finish.pids_peak, "task peak counter rolled back"),
        (
            "memory.events:oom_kill",
            start.oom_kill_count,
            finish.oom_kill_count,
            "OOM-kill counter rolled back",
        ),
    ):
        if (start_value is None) is not (finish_value is None):
            _fail("optional resource support changed between endpoint samples")
        if start_value is not None and finish_value is not None and finish_value < start_value:
            _fail(message)
        if (start_value is None) is not (label in start.unsupported_fields):
            _fail("optional resource support declaration differs")


def _ceiling_record(*, ceiling: int, observed: int) -> dict[str, Any]:
    return {
        "ceiling": ceiling,
        "observed": observed,
        "supported": True,
        "within_ceiling": observed <= ceiling,
    }


def _unsupported_ceiling_record(
    *,
    ceiling: int,
    raw_observed: int | None,
    reason: str,
) -> dict[str, Any]:
    return {
        "ceiling": ceiling,
        "raw_observed": raw_observed,
        "reason": reason,
        "supported": False,
    }


def _receipt_body(
    *,
    request: MatchedV3QualificationResourceObservationRequest,
    finish_report: MatchedV3QualificationResourceFinishReport,
    start: ResourceReaderSample,
    finish: ResourceReaderSample,
    start_monotonic_ns: int,
    finish_monotonic_ns: int,
) -> dict[str, Any]:
    _validate_finish_report_against_request(finish_report, request)
    _validate_sample_against_request(start, request=request, phase="start")
    _validate_sample_against_request(finish, request=request, phase="finish")
    cpu_usec_delta = finish.cgroup_cpu_usage_usec - start.cgroup_cpu_usage_usec
    if cpu_usec_delta < 0:
        _fail("cgroup CPU counter rolled back")
    if cpu_usec_delta > _MAX_INTEGER // 1000:
        _fail("CPU nanosecond conversion overflowed")
    cpu_ns = cpu_usec_delta * 1000
    wall_ns = finish_monotonic_ns - start_monotonic_ns
    if wall_ns < 0:
        _fail("monotonic clock rolled back")
    _check_optional_support_and_counters(start, finish)

    ceilings = dict(request.declared_ceilings)
    output_ceilings = dict(request.output_byte_ceilings)
    output_counts = {
        "result": {
            "ceiling": output_ceilings["maximum_result_bytes"],
            "observed_size_bytes": finish_report.result_size_bytes,
            "provenance": "executor_finish_report_integer_only",
            "within_ceiling": (
                finish_report.result_size_bytes <= output_ceilings["maximum_result_bytes"]
            ),
        },
        "stderr": {
            "ceiling": output_ceilings["maximum_stderr_bytes"],
            "observed_size_bytes": finish_report.stderr_size_bytes,
            "provenance": "executor_finish_report_integer_only",
            "within_ceiling": (
                finish_report.stderr_size_bytes <= output_ceilings["maximum_stderr_bytes"]
            ),
        },
        "stdout": {
            "ceiling": output_ceilings["maximum_stdout_bytes"],
            "observed_size_bytes": finish_report.stdout_size_bytes,
            "provenance": "executor_finish_report_integer_only",
            "within_ceiling": (
                finish_report.stdout_size_bytes <= output_ceilings["maximum_stdout_bytes"]
            ),
        },
    }
    cgroup_cleanup = not finish.cgroup_populated and not finish.cgroup_member_pids
    cleanup_exact = bool(
        cgroup_cleanup
        and finish_report.direct_child_waited
        and finish_report.direct_child_reaped
        and finish_report.descendant_cleanup_reported_complete
        and not finish_report.cleanup_deadline_expired
        and finish_report.output_counts_complete
        and not finish_report.timed_out
    )
    raw_memory_peak = finish.memory_peak_bytes
    raw_task_peak = finish.pids_peak
    resource_checks = {
        "max_cpu_time_ns": _ceiling_record(
            ceiling=ceilings["max_cpu_time_ns"], observed=cpu_ns
        ),
        "max_disk_peak_bytes": _unsupported_ceiling_record(
            ceiling=ceilings["max_disk_peak_bytes"],
            raw_observed=None,
            reason="no_content_free_continuous_disk_peak_source",
        ),
        "max_peak_rss_bytes": _unsupported_ceiling_record(
            ceiling=ceilings["max_peak_rss_bytes"],
            raw_observed=raw_memory_peak,
            reason="cgroup_memory_peak_reset_or_freshness_not_proven",
        ),
        "max_temporary_peak_bytes": _unsupported_ceiling_record(
            ceiling=ceilings["max_temporary_peak_bytes"],
            raw_observed=None,
            reason="no_content_free_continuous_temporary_peak_source",
        ),
        "max_thread_count": _unsupported_ceiling_record(
            ceiling=ceilings["max_thread_count"],
            raw_observed=raw_task_peak,
            reason="cgroup_pids_peak_reset_or_freshness_not_proven",
        ),
        "max_wall_time_ns": _ceiling_record(
            ceiling=ceilings["max_wall_time_ns"], observed=wall_ns
        ),
    }
    blockers = [
        "read_only_memory_peak_freshness_or_reset_not_proven",
        "whole_process_launch_boundary_not_observed",
        "continuous_cgroup_membership_between_endpoints_not_attested",
        "nested_child_cgroup_exclusivity_not_attested",
        "multi_file_endpoint_sample_not_atomic",
        "endpoint_samples_do_not_prove_continuous_thread_or_storage_peaks",
        "executor_reported_output_and_process_state_not_independently_observed",
        "remaining_qualification_resource_fields_not_observed",
    ]
    if start.unsupported_fields:
        blockers.append("optional_cgroup_measurements_unsupported")
    if not cgroup_cleanup:
        blockers.append("cgroup_still_populated_after_finish")
    if finish_report.timed_out:
        blockers.append("execution_timed_out")
    if finish_report.cleanup_deadline_expired:
        blockers.append("cleanup_deadline_expired")
    if not finish_report.output_counts_complete:
        blockers.append("executor_output_counts_incomplete")
    if not finish_report.direct_child_reaped:
        blockers.append("direct_child_not_reported_reaped")
    if not finish_report.descendant_cleanup_reported_complete:
        blockers.append("descendant_cleanup_not_reported_complete")
    if any(not cast(bool, item["within_ceiling"]) for item in output_counts.values()):
        blockers.append("output_byte_ceiling_exceeded")
    if not cast(bool, resource_checks["max_cpu_time_ns"]["within_ceiling"]):
        blockers.append("interval_cpu_ceiling_exceeded")
    if not cast(bool, resource_checks["max_wall_time_ns"]["within_ceiling"]):
        blockers.append("interval_wall_ceiling_exceeded")
    if (
        start.oom_kill_count is not None
        and finish.oom_kill_count is not None
        and finish.oom_kill_count > start.oom_kill_count
    ):
        blockers.append("cgroup_oom_kill_observed")
    return {
        "schema_version": QUALIFICATION_RESOURCE_OBSERVATION_RECEIPT_SCHEMA_VERSION,
        "status": "endpoint_resource_observation_recorded_non_authorizing",
        "classification": QUALIFICATION_RESOURCE_OBSERVER_CLASSIFICATION,
        "descriptor_sha256": matched_v3_qualification_resource_observer_descriptor_sha256(),
        "request_sha256": matched_v3_qualification_resource_observation_request_sha256(request),
        "finish_report_sha256": hashlib.sha256(
            canonical_matched_v3_qualification_resource_finish_report_bytes(finish_report)
        ).hexdigest(),
        "request": request.to_dict(),
        "finish_report": finish_report.to_dict(),
        "reader_provenance": {
            "reader_kind": request.reader_kind,
            "cgroup_v2_path": request.cgroup_v2_path,
            "cgroup_device": request.expected_cgroup_device,
            "cgroup_inode": request.expected_cgroup_inode,
            "monotonic_clock_id": request.monotonic_clock_id,
            "endpoint_samples_only": True,
            "production_reader_exact": (
                request.reader_kind == LINUX_CGROUP_V2_READER_KIND
            ),
            "production_monotonic_clock_exact": (
                request.reader_kind == LINUX_CGROUP_V2_READER_KIND
            ),
            "production_allowlisted_files": (
                list(_ALLOWLISTED_FILES)
                if request.reader_kind == LINUX_CGROUP_V2_READER_KIND
                else []
            ),
        },
        "interval_counters": {
            "start_monotonic_ns": start_monotonic_ns,
            "finish_monotonic_ns": finish_monotonic_ns,
            "cpu_time_ns": cpu_ns,
            "wall_time_ns": wall_ns,
        },
        "samples": {"start": start.to_dict(), "finish": finish.to_dict()},
        "output_counts": output_counts,
        "process_state": {
            "returncode": finish_report.returncode,
            "timed_out": finish_report.timed_out,
            "process_group_termination_requested": (
                finish_report.process_group_termination_requested
            ),
            "process_group_termination_succeeded": (
                finish_report.process_group_termination_succeeded
            ),
            "direct_child_waited": finish_report.direct_child_waited,
            "direct_child_reaped": finish_report.direct_child_reaped,
            "descendant_cleanup_reported_complete": (
                finish_report.descendant_cleanup_reported_complete
            ),
            "cleanup_deadline_expired": finish_report.cleanup_deadline_expired,
            "cgroup_cleanup_observed_complete": cgroup_cleanup,
            "cleanup_exact": cleanup_exact,
            "executor_finish_receipt_sha256": (
                finish_report.executor_finish_receipt_sha256
            ),
        },
        "resource_ceiling_checks": resource_checks,
        "qualification_projection": {
            "horizon_accounting_exact": False,
            "reward_membership_structural_only": False,
            "all_resource_observations_within_predeclared_integer_ceilings": False,
            "full_28_field_resource_observation_emitted": False,
        },
        "blockers": blockers,
        "claims": {
            "acceptance_evaluated": False,
            "execution_authorized": False,
            "peak_memory_ceiling_enforced": False,
            "qualification_granted": False,
            "runtime_qualified": False,
        },
    }


@dataclass(frozen=True, slots=True)
class MatchedV3QualificationResourceObservation:
    """Immutable exact receipt bytes plus their full-file digest."""

    canonical_receipt_bytes: bytes
    receipt_sha256: str

    def __post_init__(self) -> None:
        _require_sha256(self.receipt_sha256, "resource observation receipt")
        parse_matched_v3_qualification_resource_observation_receipt(
            self.canonical_receipt_bytes,
            expected_receipt_sha256=self.receipt_sha256,
        )

    def receipt(self) -> dict[str, Any]:
        """Return detached, strictly replayed receipt content."""

        return parse_matched_v3_qualification_resource_observation_receipt(
            self.canonical_receipt_bytes,
            expected_receipt_sha256=self.receipt_sha256,
        )


def finish_matched_v3_qualification_resource_observation(
    *,
    capability: object,
    finish_report: MatchedV3QualificationResourceFinishReport,
) -> MatchedV3QualificationResourceObservation:
    """Consume one capability, take the finish endpoint, and return a denial-safe receipt."""

    if type(capability) is not _ResourceObservationCapability:
        _fail("resource observation capability type differs")
    exact_capability = capability
    with _CAPABILITY_LOCK:
        state = _CAPABILITY_STATES.get(exact_capability)
        if state is None:
            _fail("resource observation capability is unknown or already consumed")
        if state.issuer_pid != _current_pid():
            _fail("resource observation capability was used in a different observer process")
        del _CAPABILITY_STATES[exact_capability]
    if type(finish_report) is not MatchedV3QualificationResourceFinishReport:
        _fail("resource finish report type differs")
    request = state.request
    _validate_finish_report_against_request(finish_report, request)
    try:
        finish = state.reader.sample(
            phase="finish",
            target_pid=request.target_pid,
            expected_start_time_ticks=request.target_process_start_time_ticks,
            cgroup_v2_path=request.cgroup_v2_path,
        )
    except ForagerMatchedV3QualificationResourceObserverError:
        raise
    except BaseException as exc:
        raise ForagerMatchedV3QualificationResourceObserverError(
            "resource reader finish sample failed"
        ) from exc
    _validate_sample_against_request(finish, request=request, phase="finish")
    finish_clock = _clock_read(state.monotonic_ns, "finish")
    body = _receipt_body(
        request=request,
        finish_report=finish_report,
        start=state.start_sample,
        finish=finish,
        start_monotonic_ns=state.start_monotonic_ns,
        finish_monotonic_ns=finish_clock,
    )
    raw = _canonical_json(body)
    return MatchedV3QualificationResourceObservation(
        canonical_receipt_bytes=raw,
        receipt_sha256=hashlib.sha256(raw).hexdigest(),
    )


def _validate_receipt(value: object) -> dict[str, Any]:
    keys = frozenset(
        {
            "schema_version",
            "status",
            "classification",
            "descriptor_sha256",
            "request_sha256",
            "finish_report_sha256",
            "request",
            "finish_report",
            "reader_provenance",
            "interval_counters",
            "samples",
            "output_counts",
            "process_state",
            "resource_ceiling_checks",
            "qualification_projection",
            "blockers",
            "claims",
        }
    )
    item = _require_exact_keys(value, keys, "resource observation receipt")
    request = _request_from_dict(item["request"])
    finish_report = _finish_report_from_dict(item["finish_report"])
    samples = _require_exact_keys(
        item["samples"], frozenset({"start", "finish"}), "receipt samples"
    )
    start = _sample_from_dict(samples["start"])
    finish = _sample_from_dict(samples["finish"])
    counters = _require_exact_keys(
        item["interval_counters"],
        frozenset(
            {"start_monotonic_ns", "finish_monotonic_ns", "cpu_time_ns", "wall_time_ns"}
        ),
        "receipt interval counters",
    )
    start_clock = _require_int(counters["start_monotonic_ns"], "receipt start clock")
    finish_clock = _require_int(counters["finish_monotonic_ns"], "receipt finish clock")
    expected = _receipt_body(
        request=request,
        finish_report=finish_report,
        start=start,
        finish=finish,
        start_monotonic_ns=start_clock,
        finish_monotonic_ns=finish_clock,
    )
    if not _exact_json_equal(item, expected):
        _fail("resource observation receipt semantic replay differs")
    return copy.deepcopy(item)


def parse_matched_v3_qualification_resource_observation_receipt(
    raw: bytes,
    *,
    expected_receipt_sha256: str,
) -> dict[str, Any]:
    """Strictly replay one receipt against its caller-supplied full-file digest."""

    expected = _require_sha256(expected_receipt_sha256, "expected resource receipt")
    if not hmac.compare_digest(hashlib.sha256(raw).hexdigest(), expected):
        _fail("resource observation receipt digest differs")
    return _validate_receipt(_strict_json_load(raw))


__all__ = [
    "ForagerMatchedV3QualificationResourceObserverError",
    "INJECTED_TEST_READER_KIND",
    "LINUX_CGROUP_V2_READER_KIND",
    "LinuxCgroupV2ResourceReader",
    "MONOTONIC_CLOCK_ID",
    "MatchedV3QualificationResourceFinishReport",
    "MatchedV3QualificationResourceObservation",
    "MatchedV3QualificationResourceObservationRequest",
    "MatchedV3QualificationResourceReader",
    "PINNED_QUALIFICATION_RESOURCE_OBSERVER_DESCRIPTOR_SHA256",
    "QUALIFICATION_RESOURCE_FINISH_REPORT_SCHEMA_VERSION",
    "QUALIFICATION_RESOURCE_OBSERVATION_RECEIPT_SCHEMA_VERSION",
    "QUALIFICATION_RESOURCE_OBSERVATION_REQUEST_SCHEMA_VERSION",
    "QUALIFICATION_RESOURCE_OBSERVER_CLASSIFICATION",
    "QUALIFICATION_RESOURCE_OBSERVER_DESCRIPTOR_SCHEMA_VERSION",
    "QUALIFICATION_RESOURCE_OBSERVER_STATUS",
    "ResourceReaderSample",
    "canonical_matched_v3_qualification_resource_finish_report_bytes",
    "canonical_matched_v3_qualification_resource_observation_request_bytes",
    "canonical_matched_v3_qualification_resource_observer_descriptor_bytes",
    "finish_matched_v3_qualification_resource_observation",
    "issue_matched_v3_qualification_resource_observation_capability",
    "matched_v3_qualification_resource_observation_request_sha256",
    "matched_v3_qualification_resource_observer_descriptor",
    "matched_v3_qualification_resource_observer_descriptor_sha256",
    "parse_matched_v3_qualification_resource_finish_report",
    "parse_matched_v3_qualification_resource_observation_receipt",
    "parse_matched_v3_qualification_resource_observation_request",
    "parse_matched_v3_qualification_resource_observer_descriptor",
    "replay_matched_v3_qualification_resource_observation_request",
]
