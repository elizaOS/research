"""Durable, explicitly nonqualifying smoke test for one matched-v3 CPU OCI image.

The smoke is deliberately separate from both the one-shot image build and any
qualification protocol.  It fresh-validates a retained build publication,
commits a content-addressed intent before creating a container, and then runs
exactly two engineering probes in two separately created containers.  Every
container is inspected before start, actively bounded while attached, and
force-removed or proven absent by exact name before the call can succeed.

Passing this smoke says only that the local image can start its pinned Python
runtime and its in-image runtime verifier.  It creates no benchmark, evidence,
qualification, promotion, or campaign-execution authority.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import hmac
import json
import os
import re
import secrets
import stat
import sys
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Literal, NoReturn, Protocol, cast

from alberta_framework.benchmarks import (
    forager_matched_v3_cpu_oci_build_publication as build_publication,
)

CPU_OCI_ENGINEERING_SMOKE_INTENT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.cpu_oci_engineering_smoke_intent.v1"
)
CPU_OCI_ENGINEERING_SMOKE_SUCCESS_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.cpu_oci_engineering_smoke.v1"
)
CPU_OCI_ENGINEERING_SMOKE_FAILURE_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.cpu_oci_engineering_smoke_failure.v1"
)
CPU_OCI_ENGINEERING_SMOKE_STATUS: Final = (
    "local_cpu_oci_engineering_smoke_passed_unqualified_non_authorizing"
)
ENGINEERING_SMOKE_ACKNOWLEDGEMENT: Final = (
    "AUTHORIZE ONE NONQUALIFYING MATCHED-V3 CPU OCI ENGINEERING SMOKE OF THIS "
    "FRESHLY VALIDATED LOCAL IMAGE"
)

_INTENT_STATUS: Final = "engineering_smoke_intent_committed_before_container_creation"
_FAILURE_STATUS: Final = "local_cpu_oci_engineering_smoke_failed_non_authorizing"
_INTENT_FILENAME: Final = "engineering-smoke-intent.v1.json"
_SUCCESS_FILENAME: Final = "engineering-smoke-receipt.v1.json"
_FAILURE_FILENAME: Final = "engineering-smoke-failure.v1.json"
_BUILD_EXECUTION_FILENAME: Final = "oci-build-execution-receipt.v1.json"

_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_IMAGE_ID_RE: Final = re.compile(r"sha256:[0-9a-f]{64}\Z")
_CONTAINER_ID_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_CONTAINER_NAME_RE: Final = re.compile(
    r"alberta-matched-v3-smoke-(python-version|runtime-verifier)-[0-9a-f]{32}\Z"
)
_SAFE_TYPE_RE: Final = re.compile(r"[A-Za-z_][A-Za-z0-9_.]{0,255}\Z")

_DOCKER_HOST: Final = "unix:///var/run/docker.sock"
_MAX_JSON_BYTES: Final = 2 * 1024 * 1024
_MAX_JSON_DEPTH: Final = 64
_MAX_JSON_NODES: Final = 100_000
_MAX_JSON_TEXT_BYTES: Final = 1024 * 1024
_MAX_JSON_INTEGER: Final = 2**63 - 1
_MAX_ERROR_MESSAGE_BYTES: Final = 8192
_MAX_PROCESS_OUTPUT_BYTES: Final = 64 * 1024
_MAX_CONTROL_OUTPUT_BYTES: Final = 512 * 1024
_MIN_TIMEOUT_SECONDS: Final = 10
_MAX_TIMEOUT_SECONDS: Final = 900
_CLEANUP_TIMEOUT_SECONDS: Final = 60
_MEMORY_BYTES: Final = 4 * 1024 * 1024 * 1024
_NANO_CPUS: Final = 2_000_000_000
_PIDS_LIMIT: Final = 256
_TMPFS_SPEC: Final = (
    "/run/alberta:rw,noexec,nosuid,nodev,size=1g,uid=65532,gid=65532,mode=0700"
)
_FAILURE_PHASES: Final[frozenset[str]] = frozenset(
    {
        "final_build_publication_revalidation",
        "probe_python_version",
        "probe_runtime_verifier",
        "runtime_binding",
        "runtime_postflight",
        "success_publication",
    }
)

_CONTAINER_ENVIRONMENT: Final[tuple[str, ...]] = (
    "ALL_PROXY=",
    "HOME=/run/alberta",
    "HTTP_PROXY=",
    "HTTPS_PROXY=",
    "JAX_ENABLE_COMPILATION_CACHE=false",
    "JAX_PLATFORM_NAME=cpu",
    "JAX_PLATFORMS=cpu",
    "JAX_SKIP_CUDA_CONSTRAINTS_CHECK=1",
    "LANG=C.UTF-8",
    "LC_ALL=C.UTF-8",
    "LD_LIBRARY_PATH=",
    "LD_PRELOAD=",
    "NVIDIA_VISIBLE_DEVICES=void",
    "NO_PROXY=",
    "PYTHONHASHSEED=0",
    "PYTHONNOUSERSITE=1",
    "PYTHONPATH=",
    "PYTHONDONTWRITEBYTECODE=1",
    "TMPDIR=/run/alberta",
    "TZ=UTC",
    "XDG_CACHE_HOME=/run/alberta",
    "all_proxy=",
    "http_proxy=",
    "https_proxy=",
    "no_proxy=",
)
_REQUIRED_IMAGE_ENVIRONMENT: Final[Mapping[str, str]] = {
    "PATH": "/usr/local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    "PIP_NO_INDEX": "1",
    "PYTHON_VERSION": "3.12.3",
    "XLA_FLAGS": "--xla_force_host_platform_device_count=1",
}


@dataclass(frozen=True, slots=True)
class _Probe:
    probe_id: Literal["python_version", "runtime_verifier"]
    name_component: Literal["python-version", "runtime-verifier"]
    argv: tuple[str, ...]
    expected_stdout: bytes
    expected_stderr: bytes = b""


_PROBES: Final[tuple[_Probe, ...]] = (
    _Probe(
        "python_version",
        "python-version",
        ("/usr/local/bin/python", "--version"),
        b"Python 3.12.3\n",
    ),
    _Probe(
        "runtime_verifier",
        "runtime-verifier",
        (
            "/usr/local/bin/python",
            "-I",
            "-B",
            "/opt/elizaos/build/verify-runtime.py",
        ),
        b"",
    ),
)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )


def _reject_json_constant(value: str) -> NoReturn:
    raise ForagerMatchedV3CpuOciEngineeringSmokeError(
        f"engineering-smoke JSON contains a non-finite constant: {value}"
    )


def _reject_json_float(value: str) -> NoReturn:
    raise ForagerMatchedV3CpuOciEngineeringSmokeError(
        f"engineering-smoke JSON contains a floating-point value: {value}"
    )


def _parse_json_integer(value: str) -> int:
    digits = value.removeprefix("-")
    if not digits or len(digits) > 19:
        _fail("engineering-smoke JSON integer is outside its bound")
    parsed = int(value)
    if not -_MAX_JSON_INTEGER <= parsed <= _MAX_JSON_INTEGER:
        _fail("engineering-smoke JSON integer is outside its bound")
    return parsed


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail("engineering-smoke JSON object contains a duplicate key")
        result[key] = value
    return result


def _assert_bounded_plain_json(value: Any) -> None:
    stack: list[tuple[Any, int]] = [(value, 1)]
    nodes = 0
    while stack:
        item, depth = stack.pop()
        nodes += 1
        if nodes > _MAX_JSON_NODES or depth > _MAX_JSON_DEPTH:
            _fail("engineering-smoke JSON exceeds its structural bound")
        if item is None or type(item) in {bool, int}:
            continue
        if type(item) is str:
            if len(item.encode("utf-8")) > _MAX_JSON_TEXT_BYTES:
                _fail("engineering-smoke JSON text exceeds its bound")
            continue
        if type(item) is list:
            stack.extend((child, depth + 1) for child in item)
            continue
        if type(item) is dict:
            for key, child in item.items():
                if type(key) is not str:
                    _fail("engineering-smoke JSON object key is not text")
                stack.append((key, depth + 1))
                stack.append((child, depth + 1))
            continue
        _fail("engineering-smoke JSON contains a non-plain value")


def _exact_json_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        left_mapping = cast(Mapping[str, Any], left)
        right_mapping = cast(Mapping[str, Any], right)
        return set(left_mapping) == set(right_mapping) and all(
            _exact_json_equal(left_mapping[key], right_mapping[key])
            for key in left_mapping
        )
    if type(left) is list:
        left_items = left
        right_items = right
        return len(left_items) == len(right_items) and all(
            _exact_json_equal(left_item, right_item)
            for left_item, right_item in zip(left_items, right_items, strict=True)
        )
    return bool(left == right)


def _bounded_error_message(error: BaseException, *, fallback: str) -> str:
    try:
        rendered = str(error)
    except BaseException:
        rendered = fallback
    if type(rendered) is not str:
        rendered = fallback
    try:
        encoded = rendered.encode("utf-8", errors="replace")
    except BaseException:
        encoded = fallback.encode("utf-8", errors="replace")
    message = encoded[:_MAX_ERROR_MESSAGE_BYTES].decode("utf-8", errors="ignore")
    return message or fallback


def _safe_error_type(error: BaseException) -> str:
    try:
        error_type = type(error).__name__
    except BaseException:
        return "UnclassifiedError"
    if type(error_type) is not str or _SAFE_TYPE_RE.fullmatch(error_type) is None:
        return "UnclassifiedError"
    return error_type


def _safe_exception_summary(error: BaseException) -> str:
    error_type = _safe_error_type(error)
    message = _bounded_error_message(error, fallback=error_type)
    return f"{error_type}: {message}"


_MISSING_ERROR_ATTRIBUTE: Final = object()


def _safe_error_attribute(error: BaseException, name: str) -> Any:
    try:
        return object.__getattribute__(error, name)
    except BaseException:
        return _MISSING_ERROR_ATTRIBUTE


def _safe_error_bool(error: BaseException, name: str, *, default: bool = False) -> bool:
    value = _safe_error_attribute(error, name)
    return value if type(value) is bool else default


def _safe_optional_error_bool(
    error: BaseException,
    name: str,
    *,
    default: bool | None,
) -> bool | None:
    value = _safe_error_attribute(error, name)
    return value if value is None or type(value) is bool else default


def _safe_error_sha256(error: BaseException, name: str) -> str | None:
    value = _safe_error_attribute(error, name)
    if type(value) is str and _SHA256_RE.fullmatch(value) is not None:
        return value
    return None


def _safe_set_error_attribute(error: BaseException, name: str, value: Any) -> None:
    try:
        object.__setattr__(error, name, value)
    except BaseException:
        pass


def _safe_add_note(error: BaseException, note: str) -> None:
    try:
        BaseException.add_note(error, note)
    except BaseException:
        pass


def _safe_mark_container_state_uncertain(error: BaseException) -> None:
    _safe_set_error_attribute(error, "container_state_uncertain", True)


def _fail(message: str, *, container_state_uncertain: bool = False) -> NoReturn:
    raise ForagerMatchedV3CpuOciEngineeringSmokeError(
        message,
        container_state_uncertain=container_state_uncertain,
    )


def _require_sha256(value: Any, *, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None or value == "0" * 64:
        _fail(f"{label} must be one nonzero lowercase SHA-256")
    return value


def _require_image_id(value: Any, *, label: str) -> str:
    if (
        type(value) is not str
        or _IMAGE_ID_RE.fullmatch(value) is None
        or value == "sha256:" + "0" * 64
    ):
        _fail(f"{label} must be one content-addressed image ID")
    return value


def _require_absolute_path(value: Any, *, label: str) -> Path:
    if (
        type(value) is not type(Path())
        or not value.is_absolute()
        or value == Path("/")
        or any(part in {".", ".."} for part in value.parts)
    ):
        _fail(f"{label} must be one exact absolute pathlib.Path")
    return value


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _require_pairwise_nonoverlapping_roots(*roots: Path) -> None:
    if any(
        _paths_overlap(left, right)
        for index, left in enumerate(roots)
        for right in roots[index + 1 :]
    ):
        _fail("engineering-smoke roots must be pairwise nonoverlapping")


def _classification() -> dict[str, bool]:
    return {
        "benchmark_execution_authorized": False,
        "engineering_only": True,
        "evidence": False,
        "promotion_authorized": False,
        "qualification": False,
    }


def _claims() -> dict[str, bool]:
    return {
        "benchmark_scores_observed": False,
        "campaign_execution_authorized": False,
        "image_runtime_engineering_smoke_only": True,
        "image_publication_claim_created": False,
        "performance_claim_created": False,
        "scientific_evidence_created": False,
    }


def _stream_record(raw: bytes) -> dict[str, Any]:
    return {"sha256": _sha256(raw), "size_bytes": len(raw)}


def _probe_contract(probe: _Probe) -> dict[str, Any]:
    return {
        "argv": list(probe.argv),
        "expected_returncode": 0,
        "expected_stderr": _stream_record(probe.expected_stderr),
        "expected_stdout": _stream_record(probe.expected_stdout),
        "probe_id": probe.probe_id,
    }


def _sandbox_contract() -> dict[str, Any]:
    return {
        "auto_remove": False,
        "cap_drop": ["ALL"],
        "cgroup_namespace": "private",
        "container_count": 2,
        "container_environment": list(_CONTAINER_ENVIRONMENT),
        "cpu_count": 2,
        "ipc_namespace": "private",
        "memory_bytes": _MEMORY_BYTES,
        "memory_swap_bytes": _MEMORY_BYTES,
        "network": "none",
        "no_new_privileges": True,
        "operating_system": "linux",
        "pids_limit": _PIDS_LIMIT,
        "platform": "linux/amd64",
        "pull": "never",
        "read_only_root": True,
        "required_image_environment": dict(_REQUIRED_IMAGE_ENVIRONMENT),
        "tmpfs": _TMPFS_SPEC,
        "user": "65532:65532",
        "workdir": "/work",
    }


class ForagerMatchedV3CpuOciEngineeringSmokeError(RuntimeError):
    """The engineering smoke failed closed."""

    container_state_uncertain: bool
    failure_committed: bool | None
    failure_full_lineage_validated: bool | None
    failure_publication_state_uncertain: bool
    failure_receipt_sha256: str | None

    def __init__(self, message: str, *, container_state_uncertain: bool = False) -> None:
        super().__init__(message)
        object.__setattr__(self, "container_state_uncertain", container_state_uncertain)
        object.__setattr__(self, "failure_committed", False)
        object.__setattr__(self, "failure_full_lineage_validated", None)
        object.__setattr__(self, "failure_publication_state_uncertain", False)
        object.__setattr__(self, "failure_receipt_sha256", None)


class MatchedV3CpuOciEngineeringSmokeIntentExistsError(
    ForagerMatchedV3CpuOciEngineeringSmokeError
):
    """The same deterministic smoke already consumed its publication route."""

    def __init__(self, message: str, *, intent_sha256: str) -> None:
        self.intent_sha256 = _require_sha256(intent_sha256, label="existing smoke intent")
        self.intent_committed = True
        super().__init__(message, container_state_uncertain=False)


class MatchedV3CpuOciEngineeringSmokeIntentPublicationUncertainError(
    ForagerMatchedV3CpuOciEngineeringSmokeError
):
    """Intent publication visibility escaped exact classification."""

    def __init__(
        self,
        message: str,
        *,
        intent_sha256: str,
        intent_committed: bool | None = True,
    ) -> None:
        self.intent_sha256 = _require_sha256(
            intent_sha256,
            label="uncertain smoke intent",
        )
        if intent_committed is not None and type(intent_committed) is not bool:
            _fail("uncertain smoke intent commit state differs")
        self.intent_committed = intent_committed
        super().__init__(message, container_state_uncertain=False)


class MatchedV3CpuOciEngineeringSmokeSuccessPublicationUncertainError(
    ForagerMatchedV3CpuOciEngineeringSmokeError
):
    """A success address may be visible and must be replayed before classification."""

    intent_committed: bool | None
    intent_sha256: str | None
    receipt_sha256: str
    success_committed: bool | None
    success_publication_state_uncertain: bool

    def __init__(
        self,
        message: str,
        *,
        receipt_sha256: str,
        success_committed: bool | None,
    ) -> None:
        object.__setattr__(
            self,
            "receipt_sha256",
            _require_sha256(
                receipt_sha256,
                label="uncertain smoke success receipt",
            ),
        )
        if success_committed is not None and type(success_committed) is not bool:
            _fail("uncertain smoke success commit state differs")
        object.__setattr__(self, "success_committed", success_committed)
        object.__setattr__(self, "success_publication_state_uncertain", True)
        object.__setattr__(self, "intent_sha256", None)
        object.__setattr__(self, "intent_committed", None)
        super().__init__(message, container_state_uncertain=False)


class MatchedV3CpuOciEngineeringSmokeFailurePublicationUncertainError(
    ForagerMatchedV3CpuOciEngineeringSmokeError
):
    """A failure address may be visible and requires an explicit fresh replay."""

    def __init__(
        self,
        message: str,
        *,
        failure_receipt_sha256: str,
        failure_committed: bool | None,
    ) -> None:
        if failure_committed is not None and type(failure_committed) is not bool:
            _fail("uncertain smoke failure commit state differs")
        super().__init__(message, container_state_uncertain=False)
        object.__setattr__(
            self,
            "failure_receipt_sha256",
            _require_sha256(
                failure_receipt_sha256,
                label="uncertain smoke failure receipt",
            ),
        )
        object.__setattr__(self, "failure_committed", failure_committed)
        object.__setattr__(self, "failure_publication_state_uncertain", True)


@dataclass(frozen=True, slots=True)
class MatchedV3CpuOciEngineeringSmokeRequest:
    """Exact caller-carried pins and authorization for one engineering smoke."""

    artifact_root: Path
    build_publication_root: Path
    publication_root: Path
    expected_build_context_receipt_sha256: str
    expected_build_execution_receipt_sha256: str
    exact_acknowledgement: str
    timeout_seconds: int = 300

    def __post_init__(self) -> None:
        _require_absolute_path(self.artifact_root, label="artifact root")
        _require_absolute_path(
            self.build_publication_root,
            label="build publication root",
        )
        _require_absolute_path(self.publication_root, label="smoke publication root")
        roots = (self.artifact_root, self.build_publication_root, self.publication_root)
        _require_pairwise_nonoverlapping_roots(*roots)
        _require_sha256(
            self.expected_build_context_receipt_sha256,
            label="expected build context receipt",
        )
        _require_sha256(
            self.expected_build_execution_receipt_sha256,
            label="expected build execution receipt",
        )
        if (
            type(self.exact_acknowledgement) is not str
            or not hmac.compare_digest(
                self.exact_acknowledgement,
                ENGINEERING_SMOKE_ACKNOWLEDGEMENT,
            )
        ):
            _fail("exact engineering-smoke acknowledgement differs")
        if (
            type(self.timeout_seconds) is not int
            or isinstance(self.timeout_seconds, bool)
            or not _MIN_TIMEOUT_SECONDS <= self.timeout_seconds <= _MAX_TIMEOUT_SECONDS
        ):
            _fail("engineering-smoke timeout is outside its bounded range")


@dataclass(frozen=True, slots=True)
class PublishedMatchedV3CpuOciEngineeringSmoke:
    """One retained engineering-smoke success receipt."""

    intent_directory: Path
    success_directory: Path
    intent_sha256: str
    receipt_sha256: str
    build_publication_receipt_sha256: str
    image_id: str

    def __post_init__(self) -> None:
        _require_absolute_path(self.intent_directory, label="published smoke intent directory")
        _require_absolute_path(
            self.success_directory,
            label="published smoke success directory",
        )
        _require_sha256(self.intent_sha256, label="published smoke intent")
        _require_sha256(self.receipt_sha256, label="published smoke receipt")
        _require_sha256(
            self.build_publication_receipt_sha256,
            label="published build publication receipt",
        )
        _require_image_id(self.image_id, label="published smoke image ID")
        if (
            self.intent_directory.name != self.intent_sha256
            or self.success_directory.name != self.receipt_sha256
        ):
            _fail("published engineering-smoke directory address differs")


@dataclass(frozen=True, slots=True)
class PublishedMatchedV3CpuOciEngineeringSmokeFailure:
    """One retained, bounded engineering-smoke failure receipt."""

    directory: Path
    receipt_sha256: str
    intent_sha256: str
    phase: str
    container_state_uncertain: bool

    def __post_init__(self) -> None:
        _require_absolute_path(self.directory, label="published smoke failure directory")
        _require_sha256(self.receipt_sha256, label="published smoke failure")
        _require_sha256(self.intent_sha256, label="published failed smoke intent")
        if type(self.phase) is not str or self.phase not in _FAILURE_PHASES:
            _fail("published smoke failure phase differs")
        if type(self.container_state_uncertain) is not bool:
            _fail("published smoke failure uncertainty differs")
        if self.directory.name != self.receipt_sha256:
            _fail("published engineering-smoke failure directory address differs")


@dataclass(frozen=True, slots=True)
class _ProcessResult:
    returncode: int
    stdout: bytes
    stderr: bytes


class _Invoke(Protocol):
    def __call__(
        self,
        argv: tuple[str, ...],
        *,
        timeout_seconds: int,
        stdout_limit_bytes: int,
        stderr_limit_bytes: int,
        container_state_uncertain: bool,
    ) -> _ProcessResult: ...


@dataclass(frozen=True, slots=True)
class _RuntimeBinding:
    docker_path: str
    cli_working_directory: str
    toolchain_record: Mapping[str, Any]
    invoke: _Invoke


@dataclass(slots=True)
class _CleanupProgress:
    state: str = "not_attempted"
    resolved_container_id: str | None = None
    exact_name_absent: bool = False
    proven_absent_ids: set[str] = field(default_factory=set)


@dataclass(slots=True)
class _SmokeAttemptState:
    daemon_projection: dict[str, Any] | None = None
    lifecycle: dict[str, Any] | None = None
    container_state_uncertain_latched: bool = False
    create_invoked: bool = False
    cleanup_progress: _CleanupProgress | None = None
    current_probe: _Probe | None = None
    container_name: str | None = None
    authoritative_container_id: str | None = None
    candidate_container_ids: set[str] = field(default_factory=set)


def _lifecycle_record(
    *,
    probe: _Probe,
    container_name: str,
    authoritative_container_id: str | None,
    candidate_container_ids: set[str],
    create_invoked: bool,
    uncertainty_latched: bool,
    cleanup: _CleanupProgress,
) -> dict[str, Any]:
    return {
        "authoritative_container_id": authoritative_container_id,
        "candidate_container_ids": sorted(candidate_container_ids),
        "cleanup": {
            "all_candidate_ids_absent": candidate_container_ids
            <= cleanup.proven_absent_ids,
            "exact_name_absent": cleanup.exact_name_absent,
            "proven_absent_ids": sorted(cleanup.proven_absent_ids),
            "resolved_container_id": cleanup.resolved_container_id,
            "state": cleanup.state,
        },
        "container_name": container_name,
        "create_invoked": create_invoked,
        "probe_id": probe.probe_id,
        "uncertainty_latched": uncertainty_latched,
    }


def _attempt_container_state_uncertain(attempt_state: _SmokeAttemptState) -> bool:
    if attempt_state.container_state_uncertain_latched:
        return True
    if not attempt_state.create_invoked:
        return False
    cleanup = attempt_state.cleanup_progress
    if cleanup is None:
        return True
    return not (
        cleanup.state
        in {
            "already_absent_with_all_proofs",
            "force_removed_by_id_with_all_absence_proofs",
        }
        and cleanup.exact_name_absent
        and (
            cleanup.resolved_container_id is None
            or cleanup.resolved_container_id in cleanup.proven_absent_ids
        )
    )


def _rebuild_attempt_lifecycle(
    attempt_state: _SmokeAttemptState,
) -> dict[str, Any] | None:
    if (
        attempt_state.current_probe is None
        or attempt_state.container_name is None
        or attempt_state.cleanup_progress is None
    ):
        return attempt_state.lifecycle
    return _lifecycle_record(
        probe=attempt_state.current_probe,
        container_name=attempt_state.container_name,
        authoritative_container_id=attempt_state.authoritative_container_id,
        candidate_container_ids=attempt_state.candidate_container_ids,
        create_invoked=attempt_state.create_invoked,
        uncertainty_latched=attempt_state.container_state_uncertain_latched,
        cleanup=attempt_state.cleanup_progress,
    )


@dataclass(frozen=True, slots=True)
class _ProbeObservation:
    probe_id: str
    container_name: str
    container_id: str
    postrun_inspect_object_sha256: str
    prestart_inspect_object_sha256: str
    returncode: int
    stdout: bytes
    stderr: bytes
    cleanup_state: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "cleanup": {
                "exact_id_absent": True,
                "exact_name_absent": True,
                "state": self.cleanup_state,
            },
            "container": {
                "id": self.container_id,
                "name": self.container_name,
                "postrun_inspect_object_sha256": self.postrun_inspect_object_sha256,
                "prestart_inspect_object_sha256": self.prestart_inspect_object_sha256,
            },
            "observed": {
                "returncode": self.returncode,
                "stderr": _stream_record(self.stderr),
                "stdout": _stream_record(self.stdout),
            },
            "probe_id": self.probe_id,
            "state_projection": {
                "postrun_dead": False,
                "postrun_error": "",
                "postrun_exit_code": 0,
                "postrun_oom_killed": False,
                "postrun_status": "exited",
                "prestart_status": "created",
            },
        }


def _load_executor_contract() -> Any:
    from alberta_framework.benchmarks import (
        forager_matched_v3_cpu_oci_build_executor as executor_contract,
    )

    return executor_contract


def _decode_object(raw: bytes, *, label: str) -> dict[str, Any]:
    if type(raw) is not bytes or not 1 <= len(raw) <= _MAX_JSON_BYTES:
        _fail(f"{label} bytes are absent or oversized")
    try:
        text = raw.decode("ascii")
        value = json.loads(
            text,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_json_constant,
            parse_float=_reject_json_float,
            parse_int=_parse_json_integer,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ForagerMatchedV3CpuOciEngineeringSmokeError(
            f"{label} is not strict ASCII JSON"
        ) from exc
    if type(value) is not dict:
        _fail(f"{label} must be one JSON object")
    _assert_bounded_plain_json(value)
    return cast(dict[str, Any], value)


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], *, label: str) -> None:
    if type(value) is not dict or set(value) != expected:
        _fail(f"{label} key set differs")


def _read_build_execution_receipt(
    published_build: build_publication.PublishedMatchedV3CpuOciBuild,
    build_publication_root: Path,
) -> dict[str, Any]:
    try:
        with build_publication._open_root(  # noqa: SLF001
            build_publication_root,
            label="engineering-smoke build publication root",
            mutable=True,
        ) as root:
            with build_publication._retain_addressed_directory(  # noqa: SLF001
                root,
                category="successes",
                address=published_build.execution_receipt_sha256,
            ) as directory:
                raw = build_publication._read_unpinned_file_at(  # noqa: SLF001
                    directory,
                    _BUILD_EXECUTION_FILENAME,
                    maximum_size_bytes=_MAX_JSON_BYTES,
                )
    except build_publication.ForagerMatchedV3CpuOciBuildPublicationError as exc:
        raise ForagerMatchedV3CpuOciEngineeringSmokeError(
            "cannot retain the validated build execution receipt"
        ) from exc
    if _sha256(raw) != published_build.execution_receipt_sha256:
        _fail("retained build execution receipt address differs")
    return _decode_object(raw, label="build execution receipt")


def _bound_toolchain_record(
    published_build: build_publication.PublishedMatchedV3CpuOciBuild,
    build_publication_root: Path,
) -> dict[str, Any]:
    execution = _read_build_execution_receipt(published_build, build_publication_root)
    toolchain = execution.get("execution_toolchain")
    if type(toolchain) is not dict:
        _fail("build execution toolchain record is absent")
    _require_exact_keys(
        toolchain,
        {"contract", "contract_sha256", "postflight", "preflight"},
        label="build execution toolchain",
    )
    contract = toolchain["contract"]
    preflight = toolchain["preflight"]
    postflight = toolchain["postflight"]
    if type(contract) is not dict or type(preflight) is not dict or type(postflight) is not dict:
        _fail("build execution toolchain components must be objects")
    contract_sha = _require_sha256(
        toolchain["contract_sha256"],
        label="build execution toolchain contract",
    )
    if _sha256(_canonical_json({"execution_toolchain": contract})) != contract_sha:
        _fail("build execution toolchain contract hash differs")
    retained_keys = {"buildx_plugin", "docker_cli", "docker_dynamic_runtime"}
    expected_observation = {
        key: copy.deepcopy(value)
        for key, value in contract.items()
        if key in retained_keys
    }
    if not _exact_json_equal(preflight, expected_observation) or not _exact_json_equal(
        postflight,
        expected_observation,
    ):
        _fail("build execution toolchain observations differ from the contract")
    environment = contract.get("environment")
    if type(environment) is not dict:
        _fail("build execution toolchain environment is absent")
    fixed = environment.get("fixed")
    if not _exact_json_equal(
        fixed,
        {
            "DOCKER_HOST": _DOCKER_HOST,
            "LC_ALL": "C.UTF-8",
            "PATH": "/usr/bin:/bin",
        },
    ):
        _fail("build execution Docker routing environment differs")
    docker_cli = contract.get("docker_cli")
    if type(docker_cli) is not dict or docker_cli.get("path") != "/usr/bin/docker":
        _fail("build execution Docker CLI binding differs")
    return {
        "contract": copy.deepcopy(contract),
        "contract_sha256": contract_sha,
        "docker_cli": copy.deepcopy(docker_cli),
    }


@contextmanager
def _retain_runtime_binding(
    expected_toolchain: Mapping[str, Any],
) -> Iterator[_RuntimeBinding]:
    executor = _load_executor_contract()
    expected_contract = expected_toolchain.get("contract")
    if type(expected_contract) is not dict:
        _fail("expected retained toolchain contract is absent")
    try:
        current_contract = executor._execution_toolchain_contract()  # noqa: SLF001
    except BaseException as exc:
        raise ForagerMatchedV3CpuOciEngineeringSmokeError(
            "current Docker toolchain contract cannot be measured"
        ) from exc
    if not _exact_json_equal(current_contract, expected_contract):
        _fail("current Docker toolchain differs from the validated build toolchain")

    with executor._retain_execution_toolchain() as toolchain:  # noqa: SLF001
        with executor._exclusive_cli_state() as cli_state:  # noqa: SLF001
            try:
                observed = toolchain.reverify(image_state_uncertain=False)
                cli_state.reverify(image_state_uncertain=False)
            except BaseException as exc:
                raise ForagerMatchedV3CpuOciEngineeringSmokeError(
                    "retained Docker toolchain preflight failed"
                ) from exc
            expected_observed = {
                key: copy.deepcopy(value)
                for key, value in expected_contract.items()
                if key in {"buildx_plugin", "docker_cli", "docker_dynamic_runtime"}
            }
            if not _exact_json_equal(observed, expected_observed):
                _fail("retained Docker toolchain preflight differs")
            docker_path = cast(
                str,
                cast(Mapping[str, Any], expected_contract["docker_cli"])["path"],
            )

            def invoke(
                argv: tuple[str, ...],
                *,
                timeout_seconds: int,
                stdout_limit_bytes: int,
                stderr_limit_bytes: int,
                container_state_uncertain: bool,
            ) -> _ProcessResult:
                if not argv or argv[0] != docker_path:
                    _fail(
                        "engineering-smoke command does not select the pinned Docker CLI",
                        container_state_uncertain=container_state_uncertain,
                    )
                try:
                    toolchain.reverify(image_state_uncertain=container_state_uncertain)
                    cli_state.reverify(image_state_uncertain=container_state_uncertain)
                    result = executor._default_process_runner(  # noqa: SLF001
                        argv,
                        environment=cli_state.environment,
                        executable_descriptor=toolchain.docker_cli.descriptor,
                        inherited_descriptors=(cli_state.directory_descriptor,),
                        working_directory=cli_state.working_directory,
                        stdin_descriptor=None,
                        timeout_seconds=timeout_seconds,
                        stdout_limit_bytes=stdout_limit_bytes,
                        stderr_limit_bytes=stderr_limit_bytes,
                    )
                    toolchain.reverify(image_state_uncertain=container_state_uncertain)
                    cli_state.reverify(image_state_uncertain=container_state_uncertain)
                except ForagerMatchedV3CpuOciEngineeringSmokeError:
                    raise
                except BaseException as exc:
                    raise ForagerMatchedV3CpuOciEngineeringSmokeError(
                        "bounded pinned Docker invocation failed",
                        container_state_uncertain=container_state_uncertain,
                    ) from exc
                if (
                    type(result.returncode) is not int
                    or type(result.stdout) is not bytes
                    or type(result.stderr) is not bytes
                    or type(result.timed_out) is not bool
                    or type(result.output_limit_exceeded) is not bool
                ):
                    _fail(
                        "bounded pinned Docker invocation returned a noncanonical result",
                        container_state_uncertain=container_state_uncertain,
                    )
                if result.timed_out:
                    _fail(
                        "bounded pinned Docker invocation timed out",
                        container_state_uncertain=container_state_uncertain,
                    )
                if result.output_limit_exceeded:
                    _fail(
                        "bounded pinned Docker invocation exceeded its output limit",
                        container_state_uncertain=container_state_uncertain,
                    )
                if (
                    len(result.stdout) > stdout_limit_bytes
                    or len(result.stderr) > stderr_limit_bytes
                ):
                    _fail(
                        "bounded pinned Docker invocation escaped its output limit",
                        container_state_uncertain=container_state_uncertain,
                    )
                return _ProcessResult(result.returncode, result.stdout, result.stderr)

            yield _RuntimeBinding(
                docker_path=docker_path,
                cli_working_directory=cli_state.working_directory,
                toolchain_record=copy.deepcopy(expected_toolchain),
                invoke=invoke,
            )


def _docker_command(binding: _RuntimeBinding, *arguments: str) -> tuple[str, ...]:
    return (binding.docker_path, f"--host={_DOCKER_HOST}", *arguments)


def _name_absence_command(binding: _RuntimeBinding, container_name: str) -> tuple[str, ...]:
    return _docker_command(
        binding,
        "container",
        "ls",
        "--all",
        "--quiet",
        "--no-trunc",
        f"--filter=name=^/{container_name}$",
    )


def _id_absence_command(binding: _RuntimeBinding, container_id: str) -> tuple[str, ...]:
    if _CONTAINER_ID_RE.fullmatch(container_id) is None:
        _fail("engineering-smoke absence-query container ID differs")
    return _docker_command(
        binding,
        "container",
        "ls",
        "--all",
        "--quiet",
        "--no-trunc",
        f"--filter=id={container_id}",
    )


def _prove_query_absent(
    binding: _RuntimeBinding,
    command: tuple[str, ...],
    *,
    label: str,
    container_state_uncertain: bool,
) -> None:
    result = binding.invoke(
        command,
        timeout_seconds=_CLEANUP_TIMEOUT_SECONDS,
        stdout_limit_bytes=_MAX_CONTROL_OUTPUT_BYTES,
        stderr_limit_bytes=_MAX_CONTROL_OUTPUT_BYTES,
        container_state_uncertain=container_state_uncertain,
    )
    if result.returncode != 0 or result.stdout or result.stderr:
        _fail(
            f"cannot prove the exact engineering-smoke container {label} absent",
            container_state_uncertain=container_state_uncertain,
        )


def _prove_name_absent(
    binding: _RuntimeBinding,
    container_name: str,
    *,
    container_state_uncertain: bool,
) -> None:
    _prove_query_absent(
        binding,
        _name_absence_command(binding, container_name),
        label="name",
        container_state_uncertain=container_state_uncertain,
    )


def _capture_daemon_projection(
    binding: _RuntimeBinding,
    *,
    container_state_uncertain: bool,
) -> dict[str, Any]:
    result = binding.invoke(
        _docker_command(binding, "info", "--format={{json .}}"),
        timeout_seconds=_CLEANUP_TIMEOUT_SECONDS,
        stdout_limit_bytes=_MAX_CONTROL_OUTPUT_BYTES,
        stderr_limit_bytes=_MAX_CONTROL_OUTPUT_BYTES,
        container_state_uncertain=container_state_uncertain,
    )
    if result.returncode != 0 or result.stderr:
        _fail(
            "Docker daemon identity projection failed",
            container_state_uncertain=container_state_uncertain,
        )
    value = _decode_object(result.stdout, label="Docker daemon information")
    string_fields = {
        "Architecture": "architecture",
        "CgroupDriver": "cgroup_driver",
        "CgroupVersion": "cgroup_version",
        "DockerRootDir": "docker_root_directory",
        "Driver": "storage_driver",
        "ID": "daemon_id",
        "KernelVersion": "kernel_version",
        "Name": "name",
        "OSType": "operating_system_type",
        "OperatingSystem": "operating_system",
        "ServerVersion": "server_version",
    }
    projection: dict[str, Any] = {}
    for source, target in string_fields.items():
        item = value.get(source)
        if type(item) is not str or not item or len(item.encode("utf-8")) > 1024:
            _fail(
                f"Docker daemon {source} projection differs",
                container_state_uncertain=container_state_uncertain,
            )
        projection[target] = item
    for source, target in (("NCPU", "cpu_count"), ("MemTotal", "memory_bytes")):
        item = value.get(source)
        if type(item) is not int or item <= 0:
            _fail(
                f"Docker daemon {source} projection differs",
                container_state_uncertain=container_state_uncertain,
            )
        projection[target] = item
    security_options = value.get("SecurityOptions")
    if (
        type(security_options) is not list
        or not security_options
        or any(type(item) is not str or not item for item in security_options)
    ):
        _fail(
            "Docker daemon security-options projection differs",
            container_state_uncertain=container_state_uncertain,
        )
    projection["security_options"] = list(security_options)
    if projection["operating_system_type"] != "linux":
        _fail(
            "Docker daemon operating-system type is not linux",
            container_state_uncertain=container_state_uncertain,
        )
    return _validate_daemon_projection_record(projection)


def _validate_daemon_projection_record(value: Any) -> dict[str, Any]:
    if type(value) is not dict:
        _fail("engineering-smoke daemon projection must be an object")
    string_keys = {
        "architecture",
        "cgroup_driver",
        "cgroup_version",
        "daemon_id",
        "docker_root_directory",
        "kernel_version",
        "name",
        "operating_system",
        "operating_system_type",
        "server_version",
        "storage_driver",
    }
    _require_exact_keys(
        value,
        string_keys | {"cpu_count", "memory_bytes", "security_options"},
        label="engineering-smoke daemon projection",
    )
    if any(
        type(value[key]) is not str
        or not value[key]
        or len(value[key].encode("utf-8")) > 1024
        for key in string_keys
    ):
        _fail("engineering-smoke daemon string projection differs")
    if value["operating_system_type"] != "linux":
        _fail("engineering-smoke daemon is not linux")
    if (
        type(value["cpu_count"]) is not int
        or value["cpu_count"] <= 0
        or type(value["memory_bytes"]) is not int
        or value["memory_bytes"] <= 0
    ):
        _fail("engineering-smoke daemon resource projection differs")
    options = value["security_options"]
    if (
        type(options) is not list
        or not options
        or any(type(item) is not str or not item for item in options)
    ):
        _fail("engineering-smoke daemon security projection differs")
    return copy.deepcopy(value)


def _require_daemon_continuity(
    binding: _RuntimeBinding,
    expected: Mapping[str, Any],
    *,
    container_state_uncertain: bool,
) -> None:
    observed = _capture_daemon_projection(
        binding,
        container_state_uncertain=container_state_uncertain,
    )
    if not _exact_json_equal(observed, expected):
        _fail(
            "Docker daemon identity changed during the engineering smoke",
            container_state_uncertain=container_state_uncertain,
        )


def _require_probe_daemon_continuity(
    binding: _RuntimeBinding,
    expected: Mapping[str, Any],
    *,
    attempt_state: _SmokeAttemptState,
    create_invoked: bool,
    container_state_uncertain: bool,
) -> None:
    try:
        _require_daemon_continuity(
            binding,
            expected,
            container_state_uncertain=container_state_uncertain,
        )
    except BaseException:
        if create_invoked:
            attempt_state.container_state_uncertain_latched = True
        raise


def _cleanup_container(
    binding: _RuntimeBinding,
    *,
    image_id: str,
    probe: _Probe,
    container_name: str,
    authoritative_container_id: str | None,
    candidate_container_ids: set[str],
    expected_daemon: Mapping[str, Any],
    progress: _CleanupProgress,
) -> None:
    progress.state = "daemon_precleanup_revalidation"
    _require_daemon_continuity(
        binding,
        expected_daemon,
        container_state_uncertain=True,
    )
    for candidate in candidate_container_ids:
        if _CONTAINER_ID_RE.fullmatch(candidate) is None:
            _fail("cleanup candidate container ID differs", container_state_uncertain=True)
    container_id = authoritative_container_id
    if container_id is not None:
        if (
            _CONTAINER_ID_RE.fullmatch(container_id) is None
            or container_id not in candidate_container_ids
        ):
            _fail("authoritative cleanup container ID differs", container_state_uncertain=True)
    else:
        progress.state = "resolving_exact_name"
        discovery = binding.invoke(
            _name_absence_command(binding, container_name),
            timeout_seconds=_CLEANUP_TIMEOUT_SECONDS,
            stdout_limit_bytes=_MAX_CONTROL_OUTPUT_BYTES,
            stderr_limit_bytes=_MAX_CONTROL_OUTPUT_BYTES,
            container_state_uncertain=True,
        )
        if discovery.returncode != 0 or discovery.stderr:
            _fail(
                "cannot resolve the engineering-smoke cleanup target by exact name",
                container_state_uncertain=True,
            )
        if discovery.stdout:
            try:
                discovered = discovery.stdout.decode("ascii").removesuffix("\n")
            except UnicodeDecodeError as exc:
                raise ForagerMatchedV3CpuOciEngineeringSmokeError(
                    "engineering-smoke cleanup discovery is not ASCII",
                    container_state_uncertain=True,
                ) from exc
            if (
                _CONTAINER_ID_RE.fullmatch(discovered) is None
                or discovery.stdout != f"{discovered}\n".encode("ascii")
            ):
                _fail(
                    "engineering-smoke cleanup discovery is not one exact container ID",
                    container_state_uncertain=True,
                )
            corroboration = binding.invoke(
                _docker_command(
                    binding,
                    "container",
                    "inspect",
                    "--format={{json .}}",
                    discovered,
                ),
                timeout_seconds=_CLEANUP_TIMEOUT_SECONDS,
                stdout_limit_bytes=_MAX_CONTROL_OUTPUT_BYTES,
                stderr_limit_bytes=_MAX_CONTROL_OUTPUT_BYTES,
                container_state_uncertain=True,
            )
            if corroboration.returncode != 0 or corroboration.stderr:
                _fail(
                    "cannot corroborate the exact-name cleanup target",
                    container_state_uncertain=True,
                )
            _validate_container_inspection(
                corroboration.stdout,
                image_id=image_id,
                probe=probe,
                container_name=container_name,
                container_id=discovered,
                expected_state="created",
            )
            _require_daemon_continuity(
                binding,
                expected_daemon,
                container_state_uncertain=True,
            )
            container_id = discovered
        else:
            progress.state = "proving_absence_without_resolved_id"
            for candidate in sorted(candidate_container_ids):
                _prove_query_absent(
                    binding,
                    _id_absence_command(binding, candidate),
                    label=f"candidate ID {candidate}",
                    container_state_uncertain=True,
                )
                progress.proven_absent_ids.add(candidate)
            _prove_name_absent(binding, container_name, container_state_uncertain=True)
            progress.exact_name_absent = True
            _require_daemon_continuity(
                binding,
                expected_daemon,
                container_state_uncertain=True,
            )
            progress.state = "already_absent_with_all_proofs"
            return
    if container_id is None or _CONTAINER_ID_RE.fullmatch(container_id) is None:
        _fail("cleanup could not resolve one exact container ID", container_state_uncertain=True)
    progress.resolved_container_id = container_id
    progress.state = "force_removing_resolved_id"
    removal = binding.invoke(
        _docker_command(binding, "container", "rm", "--force", container_id),
        timeout_seconds=_CLEANUP_TIMEOUT_SECONDS,
        stdout_limit_bytes=_MAX_CONTROL_OUTPUT_BYTES,
        stderr_limit_bytes=_MAX_CONTROL_OUTPUT_BYTES,
        container_state_uncertain=True,
    )
    progress.state = "proving_all_absence_routes"
    for candidate in sorted({container_id, *candidate_container_ids}):
        _prove_query_absent(
            binding,
            _id_absence_command(binding, candidate),
            label=f"ID {candidate}",
            container_state_uncertain=True,
        )
        progress.proven_absent_ids.add(candidate)
    _prove_name_absent(binding, container_name, container_state_uncertain=True)
    progress.exact_name_absent = True
    _require_daemon_continuity(
        binding,
        expected_daemon,
        container_state_uncertain=True,
    )
    if removal.returncode == 0:
        progress.state = "force_removed_by_id_with_all_absence_proofs"
    else:
        progress.state = "already_absent_with_all_proofs"


def _read_cidfile(path: Path) -> str:
    try:
        before = path.lstat()
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_ISLNK(before.st_mode)
            or before.st_uid != os.geteuid()
            or before.st_nlink != 1
            or not 64 <= before.st_size <= 65
        ):
            _fail("engineering-smoke cidfile metadata differs")
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
        )
        try:
            opened = os.fstat(descriptor)
            raw = os.read(descriptor, 66)
            trailing = os.read(descriptor, 1)
        finally:
            os.close(descriptor)
        after = path.lstat()
    except ForagerMatchedV3CpuOciEngineeringSmokeError:
        raise
    except OSError as exc:
        raise ForagerMatchedV3CpuOciEngineeringSmokeError(
            "engineering-smoke cidfile cannot be read"
        ) from exc
    if (
        trailing
        or (before.st_dev, before.st_ino, before.st_size)
        != (opened.st_dev, opened.st_ino, opened.st_size)
        or (before.st_dev, before.st_ino, before.st_size)
        != (after.st_dev, after.st_ino, after.st_size)
    ):
        _fail("engineering-smoke cidfile changed while read")
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ForagerMatchedV3CpuOciEngineeringSmokeError(
            "engineering-smoke cidfile is not ASCII"
        ) from exc
    if text.endswith("\n"):
        text = text[:-1]
    if _CONTAINER_ID_RE.fullmatch(text) is None:
        _fail("engineering-smoke cidfile does not contain one exact container ID")
    return text


def _create_command(
    binding: _RuntimeBinding,
    *,
    image_id: str,
    probe: _Probe,
    container_name: str,
    cidfile: Path,
) -> tuple[str, ...]:
    if _CONTAINER_NAME_RE.fullmatch(container_name) is None:
        _fail("engineering-smoke container name differs")
    return _docker_command(
        binding,
        "container",
        "create",
        f"--cidfile={cidfile.as_posix()}",
        f"--name={container_name}",
        "--attach=stdout",
        "--attach=stderr",
        "--pull=never",
        "--platform=linux/amd64",
        "--network=none",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--user=65532:65532",
        "--cpus=2.0",
        "--memory=4g",
        "--memory-swap=4g",
        "--pids-limit=256",
        "--cgroupns=private",
        "--ipc=private",
        "--no-healthcheck",
        "--restart=no",
        f"--tmpfs={_TMPFS_SPEC}",
        *(f"--env={item}" for item in _CONTAINER_ENVIRONMENT),
        "--workdir=/work",
        image_id,
        *probe.argv,
    )


def _environment_mapping(value: Any) -> dict[str, str]:
    if type(value) is not list or any(type(item) is not str or "=" not in item for item in value):
        _fail("container inspection environment differs")
    result: dict[str, str] = {}
    for item in value:
        key, content = item.split("=", 1)
        if not key or key in result or "\x00" in item:
            _fail("container inspection environment is duplicated or invalid")
        result[key] = content
    return result


def _validate_container_inspection(
    raw: bytes,
    *,
    image_id: str,
    probe: _Probe,
    container_name: str,
    container_id: str,
    expected_state: Literal["created", "exited"],
) -> str:
    value = _decode_object(raw, label="engineering-smoke container inspection")
    config = value.get("Config")
    host = value.get("HostConfig")
    state = value.get("State")
    mounts = value.get("Mounts")
    if type(config) is not dict or type(host) is not dict or type(state) is not dict:
        _fail("container inspection omits its configuration")
    if (
        value.get("Id") != container_id
        or value.get("Image") != image_id
        or value.get("Name") != f"/{container_name}"
        or value.get("Path") != probe.argv[0]
        or not _exact_json_equal(value.get("Args"), list(probe.argv[1:]))
        or type(value.get("RestartCount")) is not int
        or value.get("RestartCount") != 0
        or not _exact_json_equal(mounts, [])
    ):
        _fail("container inspection identity or mounts differ")
    if (
        config.get("Image") != image_id
        or config.get("Cmd") != list(probe.argv)
        or "Entrypoint" not in config
        or config["Entrypoint"] is not None
        or config.get("User") != "65532:65532"
        or config.get("WorkingDir") != "/work"
        or config.get("AttachStdin") is not False
        or config.get("AttachStdout") is not True
        or config.get("AttachStderr") is not True
        or config.get("OpenStdin") is not False
        or config.get("Tty") is not False
    ):
        _fail("container inspection process configuration differs")
    healthcheck = config.get("Healthcheck")
    if type(healthcheck) is not dict or not _exact_json_equal(
        healthcheck.get("Test"),
        ["NONE"],
    ):
        _fail("container inspection healthcheck disablement differs")
    environment = _environment_mapping(config.get("Env"))
    for key, expected in _REQUIRED_IMAGE_ENVIRONMENT.items():
        if environment.get(key) != expected:
            _fail("container inspection inherited image environment differs")
    for item in _CONTAINER_ENVIRONMENT:
        key, expected = item.split("=", 1)
        if environment.get(key) != expected:
            _fail("container inspection environment override differs")
    tmpfs = host.get("Tmpfs")
    if (
        host.get("AutoRemove") is not False
        or host.get("ReadonlyRootfs") is not True
        or host.get("Privileged") is not False
        or host.get("NetworkMode") != "none"
        or host.get("CapDrop") != ["ALL"]
        or host.get("SecurityOpt") != ["no-new-privileges"]
        or not _exact_json_equal(
            host.get("RestartPolicy"),
            {"MaximumRetryCount": 0, "Name": "no"},
        )
        or type(host.get("NanoCpus")) is not int
        or host.get("NanoCpus") != _NANO_CPUS
        or type(host.get("Memory")) is not int
        or host.get("Memory") != _MEMORY_BYTES
        or type(host.get("MemorySwap")) is not int
        or host.get("MemorySwap") != _MEMORY_BYTES
        or type(host.get("PidsLimit")) is not int
        or host.get("PidsLimit") != _PIDS_LIMIT
        or host.get("CgroupnsMode") != "private"
        or host.get("IpcMode") != "private"
        or host.get("PidMode") != ""
        or host.get("Binds") not in (None, [])
        or host.get("Devices") not in (None, [])
        or host.get("DeviceRequests") not in (None, [])
        or host.get("PublishAllPorts") is not False
        or host.get("PortBindings") not in (None, {})
        or not _exact_json_equal(
            tmpfs,
            {"/run/alberta": _TMPFS_SPEC.split(":", 1)[1]},
        )
    ):
        _fail("container inspection sandbox differs")
    common_stopped = (
        state.get("Running") is False
        and state.get("Paused") is False
        and state.get("Restarting") is False
        and state.get("Dead") is False
    )
    if expected_state == "created":
        if state.get("Status") != "created" or not common_stopped:
            _fail("container was not retained in the created state before start")
    elif (
        state.get("Status") != "exited"
        or not common_stopped
        or state.get("OOMKilled") is not False
        or type(state.get("ExitCode")) is not int
        or state.get("ExitCode") != 0
        or state.get("Error") != ""
    ):
        _fail("container was not retained in a clean exited state after start")
    return _sha256(_canonical_json(value))


def _run_probe_inside_temporary_boundary(
    binding: _RuntimeBinding,
    *,
    image_id: str,
    probe: _Probe,
    timeout_seconds: int,
    expected_daemon: Mapping[str, Any],
    attempt_state: _SmokeAttemptState,
) -> _ProbeObservation:
    attempt_state.lifecycle = None
    attempt_state.create_invoked = False
    attempt_state.cleanup_progress = None
    attempt_state.current_probe = probe
    attempt_state.container_name = None
    attempt_state.authoritative_container_id = None
    attempt_state.candidate_container_ids = set()
    container_name = f"alberta-matched-v3-smoke-{probe.name_component}-{secrets.token_hex(16)}"
    if _CONTAINER_NAME_RE.fullmatch(container_name) is None:
        _fail("generated engineering-smoke container name differs")
    attempt_state.container_name = container_name
    container_id: str | None = None
    candidate_container_ids = attempt_state.candidate_container_ids
    create_invoked = False
    observation: tuple[int, bytes, bytes, str, str] | None = None
    primary: BaseException | None = None
    cleanup = _CleanupProgress()
    attempt_state.cleanup_progress = cleanup
    attempt_state.lifecycle = _lifecycle_record(
        probe=probe,
        container_name=container_name,
        authoritative_container_id=None,
        candidate_container_ids=candidate_container_ids,
        create_invoked=False,
        uncertainty_latched=attempt_state.container_state_uncertain_latched,
        cleanup=cleanup,
    )
    with tempfile.TemporaryDirectory(
        prefix=f"{probe.name_component}-",
        dir=binding.cli_working_directory,
    ) as temporary:
        cidfile = Path(temporary) / "container.cid"
        try:
            _require_probe_daemon_continuity(
                binding,
                expected_daemon,
                attempt_state=attempt_state,
                create_invoked=create_invoked,
                container_state_uncertain=False,
            )
            _prove_name_absent(binding, container_name, container_state_uncertain=False)
            create_invoked = True
            attempt_state.create_invoked = True
            try:
                created = binding.invoke(
                    _create_command(
                        binding,
                        image_id=image_id,
                        probe=probe,
                        container_name=container_name,
                        cidfile=cidfile,
                    ),
                    timeout_seconds=timeout_seconds,
                    stdout_limit_bytes=_MAX_CONTROL_OUTPUT_BYTES,
                    stderr_limit_bytes=_MAX_CONTROL_OUTPUT_BYTES,
                    container_state_uncertain=True,
                )
            except BaseException:
                attempt_state.container_state_uncertain_latched = True
                raise
            _require_probe_daemon_continuity(
                binding,
                expected_daemon,
                attempt_state=attempt_state,
                create_invoked=create_invoked,
                container_state_uncertain=True,
            )
            if created.returncode != 0 or created.stderr:
                attempt_state.container_state_uncertain_latched = True
                _fail("engineering-smoke container creation failed", container_state_uncertain=True)
            try:
                create_stdout_id = created.stdout.decode("ascii").removesuffix("\n")
            except UnicodeDecodeError as exc:
                raise ForagerMatchedV3CpuOciEngineeringSmokeError(
                    "engineering-smoke create output is not ASCII",
                    container_state_uncertain=True,
                ) from exc
            if (
                _CONTAINER_ID_RE.fullmatch(create_stdout_id) is None
                or created.stdout != f"{create_stdout_id}\n".encode("ascii")
            ):
                _fail("engineering-smoke create output differs", container_state_uncertain=True)
            candidate_container_ids.add(create_stdout_id)
            cidfile_id = _read_cidfile(cidfile)
            candidate_container_ids.add(cidfile_id)
            if cidfile_id != create_stdout_id:
                _fail(
                    "engineering-smoke cidfile and create output differ",
                    container_state_uncertain=True,
                )
            inspected = binding.invoke(
                _docker_command(
                    binding,
                    "container",
                    "inspect",
                    "--format={{json .}}",
                    create_stdout_id,
                ),
                timeout_seconds=timeout_seconds,
                stdout_limit_bytes=_MAX_CONTROL_OUTPUT_BYTES,
                stderr_limit_bytes=_MAX_CONTROL_OUTPUT_BYTES,
                container_state_uncertain=True,
            )
            if inspected.returncode != 0 or inspected.stderr:
                _fail(
                    "engineering-smoke container inspection failed",
                    container_state_uncertain=True,
                )
            prestart_inspect_sha = _validate_container_inspection(
                inspected.stdout,
                image_id=image_id,
                probe=probe,
                container_name=container_name,
                container_id=create_stdout_id,
                expected_state="created",
            )
            container_id = create_stdout_id
            attempt_state.authoritative_container_id = container_id
            _require_probe_daemon_continuity(
                binding,
                expected_daemon,
                attempt_state=attempt_state,
                create_invoked=create_invoked,
                container_state_uncertain=True,
            )
            started = binding.invoke(
                _docker_command(
                    binding,
                    "container",
                    "start",
                    "--attach",
                    container_id,
                ),
                timeout_seconds=timeout_seconds,
                stdout_limit_bytes=_MAX_PROCESS_OUTPUT_BYTES,
                stderr_limit_bytes=_MAX_PROCESS_OUTPUT_BYTES,
                container_state_uncertain=True,
            )
            _require_probe_daemon_continuity(
                binding,
                expected_daemon,
                attempt_state=attempt_state,
                create_invoked=create_invoked,
                container_state_uncertain=True,
            )
            if (
                started.returncode != 0
                or started.stdout != probe.expected_stdout
                or started.stderr != probe.expected_stderr
            ):
                _fail("engineering-smoke probe result differs", container_state_uncertain=True)
            postrun = binding.invoke(
                _docker_command(
                    binding,
                    "container",
                    "inspect",
                    "--format={{json .}}",
                    container_id,
                ),
                timeout_seconds=timeout_seconds,
                stdout_limit_bytes=_MAX_CONTROL_OUTPUT_BYTES,
                stderr_limit_bytes=_MAX_CONTROL_OUTPUT_BYTES,
                container_state_uncertain=True,
            )
            if postrun.returncode != 0 or postrun.stderr:
                _fail(
                    "engineering-smoke post-run container inspection failed",
                    container_state_uncertain=True,
                )
            postrun_inspect_sha = _validate_container_inspection(
                postrun.stdout,
                image_id=image_id,
                probe=probe,
                container_name=container_name,
                container_id=container_id,
                expected_state="exited",
            )
            observation = (
                started.returncode,
                started.stdout,
                started.stderr,
                prestart_inspect_sha,
                postrun_inspect_sha,
            )
        except BaseException as exc:
            primary = exc
        finally:
            if create_invoked:
                try:
                    _cleanup_container(
                        binding,
                        image_id=image_id,
                        probe=probe,
                        container_name=container_name,
                        authoritative_container_id=container_id,
                        candidate_container_ids=candidate_container_ids,
                        expected_daemon=expected_daemon,
                        progress=cleanup,
                    )
                    if isinstance(
                        primary,
                        ForagerMatchedV3CpuOciEngineeringSmokeError,
                    ):
                        _safe_set_error_attribute(
                            primary,
                            "container_state_uncertain",
                            attempt_state.container_state_uncertain_latched,
                        )
                except BaseException as cleanup_error:
                    if primary is not None:
                        _safe_add_note(
                            primary,
                            "engineering-smoke cleanup also failed: "
                            f"{_safe_exception_summary(cleanup_error)}",
                        )
                        _safe_mark_container_state_uncertain(primary)
                    else:
                        primary = cleanup_error
            attempt_state.lifecycle = _lifecycle_record(
                probe=probe,
                container_name=container_name,
                authoritative_container_id=container_id,
                candidate_container_ids=candidate_container_ids,
                create_invoked=create_invoked,
                uncertainty_latched=(
                    attempt_state.container_state_uncertain_latched
                ),
                cleanup=cleanup,
            )
        if primary is not None:
            if isinstance(primary, ForagerMatchedV3CpuOciEngineeringSmokeError):
                raise primary
            if not isinstance(primary, Exception):
                if _attempt_container_state_uncertain(attempt_state):
                    _safe_mark_container_state_uncertain(primary)
                raise primary
            raise ForagerMatchedV3CpuOciEngineeringSmokeError(
                "engineering-smoke probe escaped its lifecycle boundary",
                container_state_uncertain=_attempt_container_state_uncertain(
                    attempt_state
                ),
            ) from primary
    if (
        container_id is None
        or observation is None
        or not cleanup.exact_name_absent
        or container_id not in cleanup.proven_absent_ids
    ):
        _fail("engineering-smoke probe observation is incomplete", container_state_uncertain=True)
    returncode, stdout, stderr, prestart_inspect_sha, postrun_inspect_sha = observation
    return _ProbeObservation(
        probe_id=probe.probe_id,
        container_name=container_name,
        container_id=container_id,
        postrun_inspect_object_sha256=postrun_inspect_sha,
        prestart_inspect_object_sha256=prestart_inspect_sha,
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
        cleanup_state=cleanup.state,
    )


def _run_probe(
    binding: _RuntimeBinding,
    *,
    image_id: str,
    probe: _Probe,
    timeout_seconds: int,
    expected_daemon: Mapping[str, Any],
    attempt_state: _SmokeAttemptState,
) -> _ProbeObservation:
    if attempt_state.container_state_uncertain_latched:
        _fail("engineering-smoke entered a new probe with latched uncertainty")
    try:
        return _run_probe_inside_temporary_boundary(
            binding,
            image_id=image_id,
            probe=probe,
            timeout_seconds=timeout_seconds,
            expected_daemon=expected_daemon,
            attempt_state=attempt_state,
        )
    except BaseException as exc:
        uncertain = _attempt_container_state_uncertain(attempt_state)
        if isinstance(exc, ForagerMatchedV3CpuOciEngineeringSmokeError):
            if uncertain:
                _safe_mark_container_state_uncertain(exc)
            raise
        if not isinstance(exc, Exception):
            if uncertain:
                _safe_mark_container_state_uncertain(exc)
            raise
        raise ForagerMatchedV3CpuOciEngineeringSmokeError(
            "engineering-smoke probe escaped its outer lifecycle boundary",
            container_state_uncertain=uncertain,
        ) from exc


def _build_record(
    published_build: build_publication.PublishedMatchedV3CpuOciBuild,
) -> dict[str, str]:
    return {
        "context_receipt_sha256": published_build.context_receipt_sha256,
        "execution_receipt_sha256": published_build.execution_receipt_sha256,
        "image_id": published_build.image_id,
        "publication_receipt_sha256": published_build.publication_receipt_sha256,
    }


def _intent_payload(
    *,
    build_record: Mapping[str, Any],
    toolchain_record: Mapping[str, Any],
    timeout_seconds: int,
) -> dict[str, Any]:
    return {
        "acknowledgement_sha256": _sha256(ENGINEERING_SMOKE_ACKNOWLEDGEMENT.encode("ascii")),
        "build_publication": copy.deepcopy(dict(build_record)),
        "claims": _claims(),
        "classification": _classification(),
        "probes": [_probe_contract(probe) for probe in _PROBES],
        "sandbox": _sandbox_contract(),
        "schema_version": CPU_OCI_ENGINEERING_SMOKE_INTENT_SCHEMA_VERSION,
        "status": _INTENT_STATUS,
        "timeout_seconds_per_probe": timeout_seconds,
        "toolchain": {
            "contract_sha256": toolchain_record["contract_sha256"],
            "docker_cli": copy.deepcopy(toolchain_record["docker_cli"]),
        },
    }


def _publish_files(
    publication_root: Path,
    *,
    category: str,
    address: str,
    files: Mapping[str, bytes],
    intent: bool = False,
    intent_commit_state: build_publication._IntentCommitState | None = None,  # noqa: SLF001
) -> Path:
    if intent:
        commit_state = (
            build_publication._IntentCommitState()  # noqa: SLF001
            if intent_commit_state is None
            else intent_commit_state
        )
    else:
        if intent_commit_state is not None:
            _fail("engineering-smoke intent commit state used for a non-intent")
        commit_state = None
    publication_committed = False
    try:
        with build_publication._open_root(  # noqa: SLF001
            publication_root,
            label="engineering-smoke publication root",
            mutable=True,
        ) as root:
            build_publication._prepare_layout(root)  # noqa: SLF001
            published_directory = build_publication._publish_files(  # noqa: SLF001
                root,
                category=category,
                address=address,
                files=files,
                intent=intent,
                commit_state=commit_state,
            )
            publication_committed = True
            return published_directory
    except build_publication.MatchedV3CpuOciBuildIntentExistsError as exc:
        raise MatchedV3CpuOciEngineeringSmokeIntentExistsError(
            "durable engineering-smoke intent already exists; refusing automatic retry",
            intent_sha256=address,
        ) from exc
    except build_publication.MatchedV3CpuOciBuildPublicationStateUncertainError as exc:
        if category == "successes":
            raise MatchedV3CpuOciEngineeringSmokeSuccessPublicationUncertainError(
                "engineering-smoke success publication state is uncertain",
                receipt_sha256=address,
                success_committed=True if publication_committed else None,
            ) from exc
        if category == "failures":
            raise MatchedV3CpuOciEngineeringSmokeFailurePublicationUncertainError(
                "engineering-smoke failure publication state is uncertain",
                failure_receipt_sha256=address,
                failure_committed=True if publication_committed else None,
            ) from exc
        if intent:
            raise MatchedV3CpuOciEngineeringSmokeIntentPublicationUncertainError(
                "engineering-smoke intent publication state is uncertain",
                intent_sha256=address,
                intent_committed=(
                    True if commit_state is not None and commit_state.committed else None
                ),
            ) from exc
        raise ForagerMatchedV3CpuOciEngineeringSmokeError(
            f"cannot publish engineering-smoke {category}"
        ) from exc
    except build_publication.ForagerMatchedV3CpuOciBuildPublicationError as exc:
        if category == "successes":
            raise MatchedV3CpuOciEngineeringSmokeSuccessPublicationUncertainError(
                "engineering-smoke success publication escaped exact classification",
                receipt_sha256=address,
                success_committed=True if publication_committed else None,
            ) from exc
        if category == "failures" and publication_committed:
            raise MatchedV3CpuOciEngineeringSmokeFailurePublicationUncertainError(
                "engineering-smoke failure committed but final publication replay failed",
                failure_receipt_sha256=address,
                failure_committed=True,
            ) from exc
        if intent and commit_state is not None and commit_state.committed:
            raise MatchedV3CpuOciEngineeringSmokeIntentPublicationUncertainError(
                "engineering-smoke intent committed but final publication replay failed",
                intent_sha256=address,
            ) from exc
        raise ForagerMatchedV3CpuOciEngineeringSmokeError(
            f"cannot publish engineering-smoke {category}"
        ) from exc
    except BaseException as exc:
        if category == "successes":
            raise MatchedV3CpuOciEngineeringSmokeSuccessPublicationUncertainError(
                "engineering-smoke success publication escaped exact classification",
                receipt_sha256=address,
                success_committed=True if publication_committed else None,
            ) from exc
        if category == "failures":
            raise MatchedV3CpuOciEngineeringSmokeFailurePublicationUncertainError(
                "engineering-smoke failure publication escaped exact classification",
                failure_receipt_sha256=address,
                failure_committed=True if publication_committed else None,
            ) from exc
        if intent and commit_state is not None and commit_state.committed:
            raise MatchedV3CpuOciEngineeringSmokeIntentPublicationUncertainError(
                "engineering-smoke intent committed but final publication replay failed",
                intent_sha256=address,
            ) from exc
        raise ForagerMatchedV3CpuOciEngineeringSmokeError(
            f"cannot publish engineering-smoke {category}"
        ) from exc


def _failure_payload(
    *,
    intent_sha256: str,
    build_record: Mapping[str, Any],
    phase: str,
    error: BaseException,
    daemon_projection: Mapping[str, Any] | None,
    lifecycle: Mapping[str, Any] | None,
    container_state_uncertain_override: bool | None = None,
) -> dict[str, Any]:
    error_type = _safe_error_type(error)
    message = _bounded_error_message(error, fallback=error_type)
    if (
        container_state_uncertain_override is not None
        and type(container_state_uncertain_override) is not bool
    ):
        _fail("failure uncertainty override differs")
    container_state_uncertain = (
        _safe_error_bool(error, "container_state_uncertain")
        if container_state_uncertain_override is None
        else container_state_uncertain_override
    )
    return {
        "build_publication": copy.deepcopy(dict(build_record)),
        "claims": _claims(),
        "classification": _classification(),
        "container_state_uncertain": container_state_uncertain,
        "daemon_projection": (
            None if daemon_projection is None else copy.deepcopy(dict(daemon_projection))
        ),
        "error": {"message": message, "type": error_type},
        "intent_sha256": intent_sha256,
        "lifecycle": None if lifecycle is None else copy.deepcopy(dict(lifecycle)),
        "phase": phase,
        "retry_authorized": False,
        "schema_version": CPU_OCI_ENGINEERING_SMOKE_FAILURE_SCHEMA_VERSION,
        "status": _FAILURE_STATUS,
    }


def _success_payload(
    *,
    intent_sha256: str,
    build_record: Mapping[str, Any],
    toolchain_record: Mapping[str, Any],
    daemon_projection: Mapping[str, Any],
    observations: Sequence[_ProbeObservation],
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "build_publication": copy.deepcopy(dict(build_record)),
        "claims": _claims(),
        "classification": _classification(),
        "container_count_created": 2,
        "container_count_started": 2,
        "daemon_projection": copy.deepcopy(dict(daemon_projection)),
        "intent_sha256": intent_sha256,
        "limitations": [
            "engineering smoke only",
            "not benchmark qualification",
            "not scientific evidence",
            "does not authorize a campaign",
            "local Docker daemon observation only",
            "daemon registry egress absence is not attested",
            "host kernel and Docker daemon remain external to the image",
            "no candidate workload or resource profiling was executed",
            "container absence was observed only at receipt creation time",
        ],
        "observations": [observation.to_dict() for observation in observations],
        "schema_version": CPU_OCI_ENGINEERING_SMOKE_SUCCESS_SCHEMA_VERSION,
        "status": CPU_OCI_ENGINEERING_SMOKE_STATUS,
        "toolchain": {
            "contract_sha256": toolchain_record["contract_sha256"],
            "docker_cli": copy.deepcopy(toolchain_record["docker_cli"]),
        },
    }
    body["receipt_body_sha256"] = _sha256(_canonical_json(body))
    return body


def _publish_failure_best_effort(
    request: MatchedV3CpuOciEngineeringSmokeRequest,
    *,
    intent_sha256: str,
    build_record: Mapping[str, Any],
    phase: str,
    error: BaseException,
    attempt_state: _SmokeAttemptState,
) -> None:
    try:
        container_state_uncertain = _attempt_container_state_uncertain(
            attempt_state
        ) or _safe_error_bool(error, "container_state_uncertain")
        if container_state_uncertain:
            _safe_mark_container_state_uncertain(error)
        lifecycle = _rebuild_attempt_lifecycle(attempt_state)
        attempt_state.lifecycle = copy.deepcopy(lifecycle)
        payload = _failure_payload(
            intent_sha256=intent_sha256,
            build_record=build_record,
            phase=phase,
            error=error,
            daemon_projection=attempt_state.daemon_projection,
            lifecycle=lifecycle,
            container_state_uncertain_override=container_state_uncertain,
        )
        raw = _canonical_json(payload)
        address = _sha256(raw)
        _validate_failure_receipt(raw, expected_sha256=address)
        _publish_files(
            request.publication_root,
            category="failures",
            address=address,
            files={_FAILURE_FILENAME: raw},
        )
        _safe_set_error_attribute(error, "failure_receipt_sha256", address)
        _safe_set_error_attribute(error, "failure_committed", True)
        _safe_set_error_attribute(
            error,
            "failure_publication_state_uncertain",
            False,
        )
        _safe_add_note(
            error,
            "committed engineering-smoke failure receipt address "
            f"(full lineage replay pending): sha256:{address}",
        )
        published_raw = _read_published_file(
            request.publication_root,
            category="failures",
            address=address,
            filename=_FAILURE_FILENAME,
        )
        replayed_payload = _validate_failure_receipt(
            published_raw,
            expected_sha256=address,
        )
        if not hmac.compare_digest(raw, published_raw) or not _exact_json_equal(
            replayed_payload,
            payload,
        ):
            _fail("published engineering-smoke failure receipt readback differs")
        _safe_add_note(
            error,
            f"durable engineering-smoke failure receipt: sha256:{address}",
        )
        try:
            replayed = validate_published_matched_v3_cpu_oci_engineering_smoke_failure(
                request.publication_root,
                build_publication_root=request.build_publication_root,
                artifact_root=request.artifact_root,
                expected_failure_receipt_sha256=address,
            )
            if (
                replayed.receipt_sha256 != address
                or replayed.intent_sha256 != intent_sha256
                or replayed.phase != phase
                or replayed.container_state_uncertain is not container_state_uncertain
            ):
                _fail("published engineering-smoke failure lineage replay differs")
            _safe_set_error_attribute(error, "failure_full_lineage_validated", True)
        except BaseException as replay_error:
            _safe_set_error_attribute(error, "failure_full_lineage_validated", False)
            _safe_add_note(
                error,
                "durable engineering-smoke failure receipt full-lineage replay also "
                f"failed: {_safe_exception_summary(replay_error)}",
            )
    except MatchedV3CpuOciEngineeringSmokeFailurePublicationUncertainError as publication_error:
        _safe_set_error_attribute(
            error,
            "failure_receipt_sha256",
            publication_error.failure_receipt_sha256,
        )
        _safe_set_error_attribute(
            error,
            "failure_committed",
            publication_error.failure_committed,
        )
        _safe_set_error_attribute(error, "failure_publication_state_uncertain", True)
        _safe_add_note(
            error,
            "engineering-smoke failure receipt publication state is uncertain; "
            "fresh-validate this exact address before classification: "
            f"sha256:{publication_error.failure_receipt_sha256}",
        )
    except BaseException as publication_error:
        _safe_add_note(
            error,
            "engineering-smoke failure-receipt publication also failed: "
            f"{_safe_exception_summary(publication_error)}",
        )


def execute_and_publish_matched_v3_cpu_oci_engineering_smoke(
    request: MatchedV3CpuOciEngineeringSmokeRequest,
) -> PublishedMatchedV3CpuOciEngineeringSmoke:
    """Consume one exact acknowledgement and publish one nonqualifying smoke."""

    if type(request) is not MatchedV3CpuOciEngineeringSmokeRequest:
        raise TypeError("engineering-smoke request must use the exact request type")
    published_build = build_publication.validate_published_matched_v3_cpu_oci_build(
        request.build_publication_root,
        artifact_root=request.artifact_root,
        expected_context_receipt_sha256=request.expected_build_context_receipt_sha256,
        expected_execution_receipt_sha256=request.expected_build_execution_receipt_sha256,
    )
    build_record = _build_record(published_build)
    toolchain_record = _bound_toolchain_record(
        published_build,
        request.build_publication_root,
    )
    intent_payload = _intent_payload(
        build_record=build_record,
        toolchain_record=toolchain_record,
        timeout_seconds=request.timeout_seconds,
    )
    intent_raw = _canonical_json(intent_payload)
    intent_sha = _sha256(intent_raw)
    attempt_state = _SmokeAttemptState()
    intent_commit_state = build_publication._IntentCommitState()  # noqa: SLF001
    phase = "runtime_binding"
    try:
        intent_directory = _publish_files(
            request.publication_root,
            category="intents",
            address=intent_sha,
            files={_INTENT_FILENAME: intent_raw},
            intent=True,
            intent_commit_state=intent_commit_state,
        )
        observations: list[_ProbeObservation] = []
        with _retain_runtime_binding(toolchain_record) as binding:
            daemon_projection = _capture_daemon_projection(
                binding,
                container_state_uncertain=False,
            )
            attempt_state.daemon_projection = copy.deepcopy(daemon_projection)
            for probe in _PROBES:
                if attempt_state.container_state_uncertain_latched:
                    _fail(
                        "engineering-smoke retained uncertainty before a new probe",
                        container_state_uncertain=True,
                    )
                phase = f"probe_{probe.probe_id}"
                observation = _run_probe(
                    binding,
                    image_id=published_build.image_id,
                    probe=probe,
                    timeout_seconds=request.timeout_seconds,
                    expected_daemon=daemon_projection,
                    attempt_state=attempt_state,
                )
                if attempt_state.container_state_uncertain_latched:
                    _fail(
                        "engineering-smoke probe returned with uncertainty latched",
                        container_state_uncertain=True,
                    )
                observations.append(observation)
            phase = "runtime_postflight"
            _require_daemon_continuity(
                binding,
                daemon_projection,
                container_state_uncertain=False,
            )
        if tuple(observation.probe_id for observation in observations) != tuple(
            probe.probe_id for probe in _PROBES
        ):
            _fail("engineering-smoke observation order differs")
        if attempt_state.container_state_uncertain_latched:
            _fail(
                "engineering-smoke cannot publish success after latched uncertainty",
                container_state_uncertain=True,
            )
        phase = "final_build_publication_revalidation"
        replayed_build = build_publication.validate_published_matched_v3_cpu_oci_build(
            request.build_publication_root,
            artifact_root=request.artifact_root,
            expected_context_receipt_sha256=(
                request.expected_build_context_receipt_sha256
            ),
            expected_execution_receipt_sha256=(
                request.expected_build_execution_receipt_sha256
            ),
        )
        if not _exact_json_equal(_build_record(replayed_build), build_record):
            _fail("post-smoke build publication replay differs")
        replayed_toolchain = _bound_toolchain_record(
            replayed_build,
            request.build_publication_root,
        )
        if not _exact_json_equal(replayed_toolchain, toolchain_record):
            _fail("post-smoke build toolchain replay differs")
        if attempt_state.daemon_projection is None:
            _fail("engineering-smoke daemon projection is absent")
        phase = "success_publication"
        success_payload = _success_payload(
            intent_sha256=intent_sha,
            build_record=build_record,
            toolchain_record=toolchain_record,
            daemon_projection=attempt_state.daemon_projection,
            observations=observations,
        )
        success_raw = _canonical_json(success_payload)
        success_sha = _sha256(success_raw)
        try:
            _publish_files(
                request.publication_root,
                category="successes",
                address=success_sha,
                files={_SUCCESS_FILENAME: success_raw},
            )
            replayed_smoke = validate_published_matched_v3_cpu_oci_engineering_smoke(
                request.publication_root,
                build_publication_root=request.build_publication_root,
                artifact_root=request.artifact_root,
                expected_receipt_sha256=success_sha,
            )
            if (
                replayed_smoke.intent_sha256 != intent_sha
                or replayed_smoke.receipt_sha256 != success_sha
                or replayed_smoke.image_id != published_build.image_id
                or replayed_smoke.build_publication_receipt_sha256
                != published_build.publication_receipt_sha256
            ):
                _fail("published engineering-smoke readback differs")
            if replayed_smoke.intent_directory != intent_directory:
                _fail("published engineering-smoke intent route differs after readback")
        except MatchedV3CpuOciEngineeringSmokeSuccessPublicationUncertainError:
            raise
        except BaseException as exc:
            raise MatchedV3CpuOciEngineeringSmokeSuccessPublicationUncertainError(
                "engineering-smoke success committed but final readback failed",
                receipt_sha256=success_sha,
                success_committed=True,
            ) from exc
    except (
        MatchedV3CpuOciEngineeringSmokeIntentExistsError,
        MatchedV3CpuOciEngineeringSmokeIntentPublicationUncertainError,
    ):
        raise
    except MatchedV3CpuOciEngineeringSmokeSuccessPublicationUncertainError as exc:
        _safe_set_error_attribute(exc, "intent_sha256", intent_sha)
        _safe_set_error_attribute(exc, "intent_committed", True)
        raise
    except BaseException as exc:
        if not intent_commit_state.committed:
            raise
        _safe_set_error_attribute(exc, "intent_sha256", intent_sha)
        _safe_set_error_attribute(exc, "intent_committed", True)
        if _attempt_container_state_uncertain(attempt_state):
            _safe_mark_container_state_uncertain(exc)
        _publish_failure_best_effort(
            request,
            intent_sha256=intent_sha,
            build_record=build_record,
            phase=phase,
            error=exc,
            attempt_state=attempt_state,
        )
        raise
    return replayed_smoke


def _read_published_file(
    publication_root: Path,
    *,
    category: str,
    address: str,
    filename: str,
) -> bytes:
    try:
        with build_publication._open_root(  # noqa: SLF001
            publication_root,
            label="engineering-smoke publication root replay",
            mutable=True,
        ) as root:
            with build_publication._retain_addressed_directory(  # noqa: SLF001
                root,
                category=category,
                address=address,
            ) as directory:
                return build_publication._read_unpinned_file_at(  # noqa: SLF001
                    directory,
                    filename,
                    maximum_size_bytes=_MAX_JSON_BYTES,
                )
    except build_publication.ForagerMatchedV3CpuOciBuildPublicationError as exc:
        raise ForagerMatchedV3CpuOciEngineeringSmokeError(
            f"cannot replay engineering-smoke {category} publication"
        ) from exc


def _validate_build_record(value: Any) -> dict[str, str]:
    if type(value) is not dict:
        _fail("engineering-smoke build record must be an object")
    _require_exact_keys(
        value,
        {
            "context_receipt_sha256",
            "execution_receipt_sha256",
            "image_id",
            "publication_receipt_sha256",
        },
        label="engineering-smoke build record",
    )
    return {
        "context_receipt_sha256": _require_sha256(
            value["context_receipt_sha256"],
            label="smoke build context receipt",
        ),
        "execution_receipt_sha256": _require_sha256(
            value["execution_receipt_sha256"],
            label="smoke build execution receipt",
        ),
        "image_id": _require_image_id(value["image_id"], label="smoke build image ID"),
        "publication_receipt_sha256": _require_sha256(
            value["publication_receipt_sha256"],
            label="smoke build publication receipt",
        ),
    }


def _validate_success_receipt(raw: bytes, *, expected_sha256: str) -> dict[str, Any]:
    expected = _require_sha256(expected_sha256, label="expected engineering-smoke receipt")
    if _sha256(raw) != expected:
        _fail("engineering-smoke success receipt address differs")
    value = _decode_object(raw, label="engineering-smoke success receipt")
    if _canonical_json(value) != raw:
        _fail("engineering-smoke success receipt is not canonical JSON")
    _require_exact_keys(
        value,
        {
            "build_publication",
            "claims",
            "classification",
            "container_count_created",
            "container_count_started",
            "daemon_projection",
            "intent_sha256",
            "limitations",
            "observations",
            "receipt_body_sha256",
            "schema_version",
            "status",
            "toolchain",
        },
        label="engineering-smoke success receipt",
    )
    body = dict(value)
    body_sha = _require_sha256(
        body.pop("receipt_body_sha256"),
        label="engineering-smoke receipt body",
    )
    if _sha256(_canonical_json(body)) != body_sha:
        _fail("engineering-smoke receipt body hash differs")
    if (
        value["schema_version"] != CPU_OCI_ENGINEERING_SMOKE_SUCCESS_SCHEMA_VERSION
        or value["status"] != CPU_OCI_ENGINEERING_SMOKE_STATUS
        or not _exact_json_equal(value["classification"], _classification())
        or not _exact_json_equal(value["claims"], _claims())
        or type(value["container_count_created"]) is not int
        or value["container_count_created"] != 2
        or type(value["container_count_started"]) is not int
        or value["container_count_started"] != 2
        or not _exact_json_equal(
            value["limitations"],
            [
                "engineering smoke only",
                "not benchmark qualification",
                "not scientific evidence",
                "does not authorize a campaign",
                "local Docker daemon observation only",
                "daemon registry egress absence is not attested",
                "host kernel and Docker daemon remain external to the image",
                "no candidate workload or resource profiling was executed",
                "container absence was observed only at receipt creation time",
            ],
        )
    ):
        _fail("engineering-smoke success classification differs")
    _require_sha256(value["intent_sha256"], label="engineering-smoke linked intent")
    _validate_build_record(value["build_publication"])
    _validate_daemon_projection_record(value["daemon_projection"])
    toolchain = value["toolchain"]
    if type(toolchain) is not dict:
        _fail("engineering-smoke success toolchain must be an object")
    _require_exact_keys(
        toolchain,
        {"contract_sha256", "docker_cli"},
        label="engineering-smoke success toolchain",
    )
    _require_sha256(toolchain["contract_sha256"], label="smoke toolchain contract")
    observations = value["observations"]
    if type(observations) is not list or len(observations) != len(_PROBES):
        _fail("engineering-smoke observation count differs")
    names: set[str] = set()
    ids: set[str] = set()
    for probe, observation in zip(_PROBES, observations, strict=True):
        if type(observation) is not dict:
            _fail("engineering-smoke observation must be an object")
        _require_exact_keys(
            observation,
            {"cleanup", "container", "observed", "probe_id", "state_projection"},
            label="engineering-smoke observation",
        )
        cleanup = observation["cleanup"]
        container = observation["container"]
        observed = observation["observed"]
        state_projection = observation["state_projection"]
        if (
            type(cleanup) is not dict
            or type(container) is not dict
            or type(observed) is not dict
            or type(state_projection) is not dict
        ):
            _fail("engineering-smoke observation components must be objects")
        _require_exact_keys(
            cleanup,
            {"exact_id_absent", "exact_name_absent", "state"},
            label="engineering-smoke cleanup",
        )
        _require_exact_keys(
            container,
            {
                "id",
                "name",
                "postrun_inspect_object_sha256",
                "prestart_inspect_object_sha256",
            },
            label="engineering-smoke container",
        )
        _require_exact_keys(
            observed,
            {"returncode", "stderr", "stdout"},
            label="engineering-smoke observed process",
        )
        _require_exact_keys(
            state_projection,
            {
                "postrun_dead",
                "postrun_error",
                "postrun_exit_code",
                "postrun_oom_killed",
                "postrun_status",
                "prestart_status",
            },
            label="engineering-smoke state projection",
        )
        name = container["name"]
        container_id = container["id"]
        if (
            observation["probe_id"] != probe.probe_id
            or type(name) is not str
            or _CONTAINER_NAME_RE.fullmatch(name) is None
            or not name.startswith(f"alberta-matched-v3-smoke-{probe.name_component}-")
            or type(container_id) is not str
            or _CONTAINER_ID_RE.fullmatch(container_id) is None
            or cleanup["exact_name_absent"] is not True
            or cleanup["exact_id_absent"] is not True
            or cleanup["state"]
            not in {
                "already_absent_with_all_proofs",
                "force_removed_by_id_with_all_absence_proofs",
            }
            or type(observed["returncode"]) is not int
            or observed["returncode"] != 0
            or not _exact_json_equal(
                observed["stdout"],
                _stream_record(probe.expected_stdout),
            )
            or not _exact_json_equal(
                observed["stderr"],
                _stream_record(probe.expected_stderr),
            )
            or not _exact_json_equal(
                state_projection,
                {
                    "postrun_dead": False,
                    "postrun_error": "",
                    "postrun_exit_code": 0,
                    "postrun_oom_killed": False,
                    "postrun_status": "exited",
                    "prestart_status": "created",
                },
            )
        ):
            _fail("engineering-smoke observation differs from its frozen probe")
        for key in (
            "postrun_inspect_object_sha256",
            "prestart_inspect_object_sha256",
        ):
            _require_sha256(
                container[key],
                label=f"engineering-smoke {key}",
            )
        names.add(name)
        ids.add(container_id)
    if len(names) != 2 or len(ids) != 2:
        _fail("engineering-smoke containers are not distinct")
    return value


def _validate_lifecycle_record(value: Any) -> dict[str, Any]:
    if type(value) is not dict:
        _fail("engineering-smoke lifecycle must be an object")
    _require_exact_keys(
        value,
        {
            "authoritative_container_id",
            "candidate_container_ids",
            "cleanup",
            "container_name",
            "create_invoked",
            "probe_id",
            "uncertainty_latched",
        },
        label="engineering-smoke lifecycle",
    )
    cleanup = value["cleanup"]
    if type(cleanup) is not dict:
        _fail("engineering-smoke lifecycle cleanup must be an object")
    _require_exact_keys(
        cleanup,
        {
            "all_candidate_ids_absent",
            "exact_name_absent",
            "proven_absent_ids",
            "resolved_container_id",
            "state",
        },
        label="engineering-smoke lifecycle cleanup",
    )
    probe_id = value["probe_id"]
    name = value["container_name"]
    candidates = value["candidate_container_ids"]
    authoritative = value["authoritative_container_id"]
    resolved = cleanup["resolved_container_id"]
    proven = cleanup["proven_absent_ids"]
    create_invoked = value["create_invoked"]
    uncertainty_latched = value["uncertainty_latched"]
    allowed_states = {
        "already_absent_with_all_proofs",
        "daemon_precleanup_revalidation",
        "force_removed_by_id_with_all_absence_proofs",
        "force_removing_resolved_id",
        "not_attempted",
        "proving_absence_without_resolved_id",
        "proving_all_absence_routes",
        "resolving_exact_name",
    }
    if (
        probe_id not in {probe.probe_id for probe in _PROBES}
        or type(name) is not str
        or _CONTAINER_NAME_RE.fullmatch(name) is None
        or not any(
            probe_id == probe.probe_id
            and name.startswith(
                f"alberta-matched-v3-smoke-{probe.name_component}-"
            )
            for probe in _PROBES
        )
        or type(create_invoked) is not bool
        or type(uncertainty_latched) is not bool
        or (uncertainty_latched and not create_invoked)
        or type(candidates) is not list
        or len(candidates) > 2
        or any(
            type(item) is not str
            or _CONTAINER_ID_RE.fullmatch(item) is None
            for item in candidates
        )
        or candidates != sorted(set(candidates))
        or (authoritative is not None and authoritative not in candidates)
        or (
            resolved is not None
            and (
                type(resolved) is not str
                or _CONTAINER_ID_RE.fullmatch(resolved) is None
            )
        )
        or type(proven) is not list
        or len(proven) > 3
        or any(
            type(item) is not str or _CONTAINER_ID_RE.fullmatch(item) is None
            for item in proven
        )
        or proven != sorted(set(proven))
        or not set(proven)
        <= set(candidates) | ({resolved} if resolved is not None else set())
        or type(cleanup["all_candidate_ids_absent"]) is not bool
        or cleanup["all_candidate_ids_absent"] != (set(candidates) <= set(proven))
        or type(cleanup["exact_name_absent"]) is not bool
        or type(cleanup["state"]) is not str
        or cleanup["state"] not in allowed_states
    ):
        _fail("engineering-smoke lifecycle contract differs")
    if create_invoked is False and (
        authoritative is not None
        or candidates
        or resolved is not None
        or proven
        or cleanup["all_candidate_ids_absent"] is not True
        or cleanup["exact_name_absent"] is not False
        or cleanup["state"] != "not_attempted"
        or uncertainty_latched is not False
    ):
        _fail("engineering-smoke no-create lifecycle differs")
    if cleanup["state"] in {
        "already_absent_with_all_proofs",
        "force_removed_by_id_with_all_absence_proofs",
    } and (
        cleanup["exact_name_absent"] is not True
        or cleanup["all_candidate_ids_absent"] is not True
        or (resolved is not None and resolved not in proven)
    ):
        _fail("engineering-smoke terminal lifecycle lacks complete absence proofs")
    if cleanup["state"] == "force_removed_by_id_with_all_absence_proofs" and (
        resolved is None or resolved not in proven
    ):
        _fail("engineering-smoke removed lifecycle lacks its resolved ID proof")
    return copy.deepcopy(value)


def _validate_failure_receipt(raw: bytes, *, expected_sha256: str) -> dict[str, Any]:
    expected = _require_sha256(
        expected_sha256,
        label="expected engineering-smoke failure receipt",
    )
    if _sha256(raw) != expected:
        _fail("engineering-smoke failure receipt address differs")
    value = _decode_object(raw, label="engineering-smoke failure receipt")
    if _canonical_json(value) != raw:
        _fail("engineering-smoke failure receipt is not canonical JSON")
    _require_exact_keys(
        value,
        {
            "build_publication",
            "claims",
            "classification",
            "container_state_uncertain",
            "daemon_projection",
            "error",
            "intent_sha256",
            "lifecycle",
            "phase",
            "retry_authorized",
            "schema_version",
            "status",
        },
        label="engineering-smoke failure receipt",
    )
    error = value["error"]
    if type(error) is not dict:
        _fail("engineering-smoke failure error must be an object")
    _require_exact_keys(
        error,
        {"message", "type"},
        label="engineering-smoke failure error",
    )
    if (
        value["schema_version"] != CPU_OCI_ENGINEERING_SMOKE_FAILURE_SCHEMA_VERSION
        or value["status"] != _FAILURE_STATUS
        or not _exact_json_equal(value["classification"], _classification())
        or not _exact_json_equal(value["claims"], _claims())
        or type(value["container_state_uncertain"]) is not bool
        or type(value["phase"]) is not str
        or value["phase"] not in _FAILURE_PHASES
        or value["retry_authorized"] is not False
        or type(error["message"]) is not str
        or not 1 <= len(error["message"].encode("utf-8")) <= 8192
        or type(error["type"]) is not str
        or _SAFE_TYPE_RE.fullmatch(error["type"]) is None
    ):
        _fail("engineering-smoke failure classification differs")
    _require_sha256(value["intent_sha256"], label="engineering-smoke failed intent")
    _validate_build_record(value["build_publication"])
    daemon_projection = value["daemon_projection"]
    lifecycle_value = value["lifecycle"]
    phase = value["phase"]
    uncertain = value["container_state_uncertain"]
    if daemon_projection is not None:
        _validate_daemon_projection_record(value["daemon_projection"])
    if phase != "runtime_binding" and daemon_projection is None:
        _fail("engineering-smoke failure phase lacks its daemon projection")
    lifecycle: dict[str, Any] | None = None
    if lifecycle_value is not None:
        lifecycle = _validate_lifecycle_record(lifecycle_value)
        if lifecycle["uncertainty_latched"] is True and uncertain is not True:
            _fail("latched lifecycle uncertainty was downgraded")
        cleanup = cast(Mapping[str, Any], lifecycle["cleanup"])
        cleanup_complete = (
            cleanup["exact_name_absent"] is True
            and cleanup["all_candidate_ids_absent"] is True
            and cleanup["state"]
            in {
                "already_absent_with_all_proofs",
                "force_removed_by_id_with_all_absence_proofs",
            }
        )
        expected_uncertainty = lifecycle["uncertainty_latched"] is True or (
            lifecycle["create_invoked"] is True and not cleanup_complete
        )
        if uncertain is not expected_uncertainty:
            _fail("failure uncertainty differs from its lifecycle proofs")
        if uncertain is False:
            if lifecycle["create_invoked"] is True and (
                cleanup["exact_name_absent"] is not True
                or cleanup["all_candidate_ids_absent"] is not True
                or cleanup["state"]
                not in {
                    "already_absent_with_all_proofs",
                    "force_removed_by_id_with_all_absence_proofs",
                }
            ):
                _fail("known cleanup failure receipt lacks complete absence proofs")
        elif lifecycle["create_invoked"] is False:
            _fail("uncertain failure cannot claim that create was never invoked")
    probe_for_phase = {
        "probe_python_version": "python_version",
        "probe_runtime_verifier": "runtime_verifier",
    }.get(phase)
    if probe_for_phase is not None:
        if lifecycle is None:
            if uncertain:
                _fail("uncertain probe failure lacks its current lifecycle")
        elif lifecycle["probe_id"] != probe_for_phase:
            _fail("engineering-smoke failure phase and lifecycle probe differ")
    if phase in {
        "final_build_publication_revalidation",
        "runtime_postflight",
        "success_publication",
    }:
        if lifecycle is None or lifecycle["probe_id"] != "runtime_verifier":
            _fail("post-probe failure lacks the terminal verifier lifecycle")
        cleanup = cast(Mapping[str, Any], lifecycle["cleanup"])
        if (
            lifecycle["create_invoked"] is not True
            or cleanup["exact_name_absent"] is not True
            or cleanup["all_candidate_ids_absent"] is not True
            or cleanup["state"]
            not in {
                "already_absent_with_all_proofs",
                "force_removed_by_id_with_all_absence_proofs",
            }
        ):
            _fail("post-probe failure lacks complete verifier cleanup proofs")
    return value


def _replay_linked_intent(
    publication_root: Path,
    *,
    intent_sha256: str,
    build_record: Mapping[str, Any],
    toolchain_record: Mapping[str, Any],
) -> Path:
    intent_sha = _require_sha256(intent_sha256, label="engineering-smoke linked intent")
    intent_raw = _read_published_file(
        publication_root,
        category="intents",
        address=intent_sha,
        filename=_INTENT_FILENAME,
    )
    if _sha256(intent_raw) != intent_sha:
        _fail("engineering-smoke intent address differs")
    intent = _decode_object(intent_raw, label="engineering-smoke intent")
    if _canonical_json(intent) != intent_raw:
        _fail("engineering-smoke intent is not canonical JSON")
    timeout = intent.get("timeout_seconds_per_probe")
    if (
        type(timeout) is not int
        or isinstance(timeout, bool)
        or not _MIN_TIMEOUT_SECONDS <= timeout <= _MAX_TIMEOUT_SECONDS
    ):
        _fail("engineering-smoke intent timeout differs")
    expected_intent = _intent_payload(
        build_record=build_record,
        toolchain_record=toolchain_record,
        timeout_seconds=timeout,
    )
    if not _exact_json_equal(intent, expected_intent):
        _fail("engineering-smoke intent differs from replayed exact inputs")
    return publication_root / "intents" / "sha256" / intent_sha


def validate_published_matched_v3_cpu_oci_engineering_smoke(
    publication_root: Path,
    *,
    build_publication_root: Path,
    artifact_root: Path,
    expected_receipt_sha256: str,
) -> PublishedMatchedV3CpuOciEngineeringSmoke:
    """Fresh-process replay of one smoke receipt and its exact build lineage."""

    _require_absolute_path(publication_root, label="smoke publication root")
    _require_absolute_path(build_publication_root, label="build publication root")
    _require_absolute_path(artifact_root, label="artifact root")
    _require_pairwise_nonoverlapping_roots(
        artifact_root,
        build_publication_root,
        publication_root,
    )
    receipt_sha = _require_sha256(
        expected_receipt_sha256,
        label="expected engineering-smoke receipt",
    )
    success_raw = _read_published_file(
        publication_root,
        category="successes",
        address=receipt_sha,
        filename=_SUCCESS_FILENAME,
    )
    success = _validate_success_receipt(success_raw, expected_sha256=receipt_sha)
    build_record = _validate_build_record(success["build_publication"])
    published_build = build_publication.validate_published_matched_v3_cpu_oci_build(
        build_publication_root,
        artifact_root=artifact_root,
        expected_context_receipt_sha256=build_record["context_receipt_sha256"],
        expected_execution_receipt_sha256=build_record["execution_receipt_sha256"],
    )
    if _build_record(published_build) != build_record:
        _fail("fresh build publication replay differs from the smoke receipt")
    toolchain_record = _bound_toolchain_record(published_build, build_publication_root)
    expected_toolchain = {
        "contract_sha256": toolchain_record["contract_sha256"],
        "docker_cli": toolchain_record["docker_cli"],
    }
    if not _exact_json_equal(success["toolchain"], expected_toolchain):
        _fail("fresh build toolchain replay differs from the smoke receipt")
    intent_sha = cast(str, success["intent_sha256"])
    intent_directory = _replay_linked_intent(
        publication_root,
        intent_sha256=intent_sha,
        build_record=build_record,
        toolchain_record=toolchain_record,
    )
    final_success_raw = _read_published_file(
        publication_root,
        category="successes",
        address=receipt_sha,
        filename=_SUCCESS_FILENAME,
    )
    final_intent_raw = _read_published_file(
        publication_root,
        category="intents",
        address=intent_sha,
        filename=_INTENT_FILENAME,
    )
    if not hmac.compare_digest(success_raw, final_success_raw) or not hmac.compare_digest(
        _sha256(final_intent_raw),
        intent_sha,
    ):
        _fail("engineering-smoke receipt or intent changed during final replay")
    return PublishedMatchedV3CpuOciEngineeringSmoke(
        intent_directory=intent_directory,
        success_directory=publication_root / "successes" / "sha256" / receipt_sha,
        intent_sha256=intent_sha,
        receipt_sha256=receipt_sha,
        build_publication_receipt_sha256=published_build.publication_receipt_sha256,
        image_id=published_build.image_id,
    )


def validate_published_matched_v3_cpu_oci_engineering_smoke_failure(
    publication_root: Path,
    *,
    build_publication_root: Path,
    artifact_root: Path,
    expected_failure_receipt_sha256: str,
) -> PublishedMatchedV3CpuOciEngineeringSmokeFailure:
    """Fresh-process replay of one bounded smoke failure and its build lineage."""

    _require_absolute_path(publication_root, label="smoke publication root")
    _require_absolute_path(build_publication_root, label="build publication root")
    _require_absolute_path(artifact_root, label="artifact root")
    _require_pairwise_nonoverlapping_roots(
        artifact_root,
        build_publication_root,
        publication_root,
    )
    receipt_sha = _require_sha256(
        expected_failure_receipt_sha256,
        label="expected engineering-smoke failure receipt",
    )
    raw = _read_published_file(
        publication_root,
        category="failures",
        address=receipt_sha,
        filename=_FAILURE_FILENAME,
    )
    failure = _validate_failure_receipt(raw, expected_sha256=receipt_sha)
    build_record = _validate_build_record(failure["build_publication"])
    published_build = build_publication.validate_published_matched_v3_cpu_oci_build(
        build_publication_root,
        artifact_root=artifact_root,
        expected_context_receipt_sha256=build_record["context_receipt_sha256"],
        expected_execution_receipt_sha256=build_record["execution_receipt_sha256"],
    )
    if _build_record(published_build) != build_record:
        _fail("fresh build publication replay differs from the smoke failure")
    toolchain_record = _bound_toolchain_record(published_build, build_publication_root)
    intent_sha = cast(str, failure["intent_sha256"])
    _replay_linked_intent(
        publication_root,
        intent_sha256=intent_sha,
        build_record=build_record,
        toolchain_record=toolchain_record,
    )
    final_failure_raw = _read_published_file(
        publication_root,
        category="failures",
        address=receipt_sha,
        filename=_FAILURE_FILENAME,
    )
    final_intent_raw = _read_published_file(
        publication_root,
        category="intents",
        address=intent_sha,
        filename=_INTENT_FILENAME,
    )
    if not hmac.compare_digest(raw, final_failure_raw) or not hmac.compare_digest(
        _sha256(final_intent_raw),
        intent_sha,
    ):
        _fail("engineering-smoke failure receipt or intent changed during final replay")
    return PublishedMatchedV3CpuOciEngineeringSmokeFailure(
        directory=publication_root / "failures" / "sha256" / receipt_sha,
        receipt_sha256=receipt_sha,
        intent_sha256=intent_sha,
        phase=cast(str, failure["phase"]),
        container_state_uncertain=cast(bool, failure["container_state_uncertain"]),
    )


def _cli_path(value: str) -> Path:
    path = Path(value)
    if (
        not path.is_absolute()
        or path == Path("/")
        or any(part in {".", ".."} for part in path.parts)
    ):
        raise argparse.ArgumentTypeError(
            "path must be absolute, non-root, and contain no dot segments"
        )
    return path


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="alberta-forager-matched-v3-cpu-oci-engineering-smoke",
        description=(
            "Run or replay an explicitly nonqualifying engineering smoke of one "
            "freshly validated matched-v3 CPU OCI build."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    execute = subparsers.add_parser("execute")
    execute.add_argument("--artifact-root", required=True, type=_cli_path)
    execute.add_argument("--build-publication-root", required=True, type=_cli_path)
    execute.add_argument("--publication-root", required=True, type=_cli_path)
    execute.add_argument("--build-context-receipt-sha256", required=True)
    execute.add_argument("--build-execution-receipt-sha256", required=True)
    execute.add_argument("--exact-acknowledgement", required=True)
    execute.add_argument("--timeout-seconds", type=int, default=300)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--artifact-root", required=True, type=_cli_path)
    validate.add_argument("--build-publication-root", required=True, type=_cli_path)
    validate.add_argument("--publication-root", required=True, type=_cli_path)
    validate.add_argument("--smoke-receipt-sha256", required=True)
    return parser


def _cli_error(error: BaseException) -> dict[str, Any]:
    error_type = _safe_error_type(error)
    return {
        "container_state_uncertain": _safe_error_bool(
            error,
            "container_state_uncertain",
        ),
        "error": {
            "message": _bounded_error_message(error, fallback=error_type),
            "type": error_type,
        },
        "failure_committed": _safe_optional_error_bool(
            error,
            "failure_committed",
            default=False,
        ),
        "failure_full_lineage_validated": _safe_optional_error_bool(
            error,
            "failure_full_lineage_validated",
            default=None,
        ),
        "failure_publication_state_uncertain": _safe_error_bool(
            error,
            "failure_publication_state_uncertain",
        ),
        "failure_receipt_sha256": _safe_error_sha256(
            error,
            "failure_receipt_sha256",
        ),
        "intent_committed": _safe_optional_error_bool(
            error,
            "intent_committed",
            default=False,
        ),
        "intent_sha256": _safe_error_sha256(error, "intent_sha256"),
        "receipt_sha256": _safe_error_sha256(error, "receipt_sha256"),
        "retry_authorized": False,
        "schema_version": "alberta.forager_matched_v3.cpu_oci_engineering_smoke_cli_error.v1",
        "status": "cpu_oci_engineering_smoke_command_failed_non_authorizing",
        "success_committed": _safe_optional_error_bool(
            error,
            "success_committed",
            default=False,
        ),
        "success_publication_state_uncertain": _safe_error_bool(
            error,
            "success_publication_state_uncertain",
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run or fresh-validate one explicitly nonqualifying engineering smoke."""

    arguments = _argument_parser().parse_args(argv)
    try:
        if arguments.command == "execute":
            published = execute_and_publish_matched_v3_cpu_oci_engineering_smoke(
                MatchedV3CpuOciEngineeringSmokeRequest(
                    artifact_root=arguments.artifact_root,
                    build_publication_root=arguments.build_publication_root,
                    publication_root=arguments.publication_root,
                    expected_build_context_receipt_sha256=(
                        arguments.build_context_receipt_sha256
                    ),
                    expected_build_execution_receipt_sha256=(
                        arguments.build_execution_receipt_sha256
                    ),
                    exact_acknowledgement=arguments.exact_acknowledgement,
                    timeout_seconds=arguments.timeout_seconds,
                )
            )
        else:
            published = validate_published_matched_v3_cpu_oci_engineering_smoke(
                arguments.publication_root,
                build_publication_root=arguments.build_publication_root,
                artifact_root=arguments.artifact_root,
                expected_receipt_sha256=arguments.smoke_receipt_sha256,
            )
    except Exception as exc:
        sys.stderr.buffer.write(_canonical_json(_cli_error(exc)))
        return 2
    sys.stdout.buffer.write(
        _canonical_json(
            {
                "build_publication_receipt_sha256": (
                    published.build_publication_receipt_sha256
                ),
                "image_id": published.image_id,
                "intent_sha256": published.intent_sha256,
                "receipt_sha256": published.receipt_sha256,
                "schema_version": CPU_OCI_ENGINEERING_SMOKE_SUCCESS_SCHEMA_VERSION,
                "status": CPU_OCI_ENGINEERING_SMOKE_STATUS,
            }
        )
    )
    return 0


__all__ = [
    "CPU_OCI_ENGINEERING_SMOKE_FAILURE_SCHEMA_VERSION",
    "CPU_OCI_ENGINEERING_SMOKE_INTENT_SCHEMA_VERSION",
    "CPU_OCI_ENGINEERING_SMOKE_STATUS",
    "CPU_OCI_ENGINEERING_SMOKE_SUCCESS_SCHEMA_VERSION",
    "ENGINEERING_SMOKE_ACKNOWLEDGEMENT",
    "ForagerMatchedV3CpuOciEngineeringSmokeError",
    "MatchedV3CpuOciEngineeringSmokeFailurePublicationUncertainError",
    "MatchedV3CpuOciEngineeringSmokeIntentExistsError",
    "MatchedV3CpuOciEngineeringSmokeIntentPublicationUncertainError",
    "MatchedV3CpuOciEngineeringSmokeSuccessPublicationUncertainError",
    "MatchedV3CpuOciEngineeringSmokeRequest",
    "PublishedMatchedV3CpuOciEngineeringSmoke",
    "PublishedMatchedV3CpuOciEngineeringSmokeFailure",
    "execute_and_publish_matched_v3_cpu_oci_engineering_smoke",
    "main",
    "validate_published_matched_v3_cpu_oci_engineering_smoke",
    "validate_published_matched_v3_cpu_oci_engineering_smoke_failure",
]


if __name__ == "__main__":
    raise SystemExit(main())
