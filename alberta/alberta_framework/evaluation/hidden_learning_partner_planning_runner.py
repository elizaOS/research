# mypy: disable-error-code="call-arg"
"""In-memory matched runner for the hidden learning-partner development plan.

The canonical scan plan intentionally carries no execution or evidence
authority.  This module adds an explicitly invoked orchestration layer while
preserving that boundary: importing it, reconstructing evaluator streams, and
building a run schedule never execute a learner life.  The public suite runner
requires a full-source/runtime-bound request, the exact acknowledgement that
all four declared roots become consumed nonpromoting development seeds, and a
single-use process-local permit issued under a strict live host check.

Every arm calls the existing one-life kernel from
``hidden_learning_partner_planning_development`` with a fresh root key.  The
runner does not reproduce bridge semantics.  Before any arm is run it derives
one evaluator-owned, action-independent cue/channel stream per seed directly
from the world's named keys.  Each completed arm is then checked against that
stream, the other arms' named key traces, and the canonical plan.

Results are fixed-shape, seed-major/condition-major in-memory records.  Loop
order is deliberately absent from record identity and output ordering.  There
is no CLI, file writer, threshold, aggregate verdict, evidence claim, or
promotion path.
"""

from __future__ import annotations

import dataclasses
import hashlib
import hmac
import json
import os
import secrets
import socket
import threading
import time
from importlib.metadata import version
from pathlib import Path
from typing import Final, Literal, Protocol, cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array

import alberta_framework.core.behavior_model as behavior_module
import alberta_framework.core.grounded_joint_world_model as grounded_module
import alberta_framework.core.signaling_bandit as signaling_module
import alberta_framework.evaluation.hidden_learning_partner_planning_development as bridge_module
import alberta_framework.evaluation.hidden_learning_partner_planning_scan_plan as plan_module
import alberta_framework.streams.learning_partner as world_module
from alberta_framework.evaluation.hidden_learning_partner_planning_development import (
    CONSTANT_ONE_DELIVERY,
    CONSTANT_ZERO_DELIVERY,
    JOINT_ADAPTIVE,
    SHUFFLED_DELIVERY,
    HiddenLearningPartnerPhaseDiagnostics,
    HiddenLearningPartnerPlanningConfig,
    HiddenLearningPartnerPlanningMetrics,
    HiddenLearningPartnerPlanningResourceBudget,
    HiddenLearningPartnerPlanningRun,
    HiddenLearningPartnerPlanningState,
    HiddenLearningPartnerPlanningTrace,
    HiddenPlanningCondition,
    condition_spec,
)
from alberta_framework.evaluation.hidden_learning_partner_planning_scan_plan import (
    CANONICAL_CONDITION_ORDER,
    PAIRED_DEVELOPMENT_SEEDS,
    HiddenLearningPartnerPlanningScanPlan,
    HiddenPlanningArm,
    HiddenPlanningExactChildClock,
    HiddenPlanningSeedBinding,
)
from alberta_framework.streams.learning_partner import (
    SHUFFLED_CHANNEL,
    LearningPartnerWorld,
    LearningPartnerWorldConfig,
    LearningPartnerWorldKeys,
    LearningPartnerWorldState,
)

HIDDEN_LEARNING_PARTNER_PLANNING_RUNNER_SCHEMA: Final = (
    "alberta.hidden-learning-partner-planning.matched-runner.development.v1"
)
HIDDEN_LEARNING_PARTNER_PLANNING_RUNNER_STATUS: Final = "DEVELOPMENT_RAW_MATCHED_SUITE_NOT_ASSESSED"
ASSESSMENT_STATUS: Final = "not_assessed"
DEVELOPMENT_SEED_ROLE: Final = "development_consumed_nonpromoting"
DEVELOPMENT_EXECUTION_ACKNOWLEDGEMENT: Final = (
    "consume_all_four_development_seeds_run_all_eleven_conditions_not_assessed"
)

DEVELOPMENT_ONLY = True
ARTIFACT_WRITES_AUTHORIZED = False
EVIDENCE_AUTHORIZED = False
SCIENTIFIC_PROMOTION_ALLOWED = False

type HiddenPlanningExecutionMode = Literal["jit", "eager"]
type HiddenPlanningPermitPhase = Literal["issued", "run_consumed", "replay_consumed"]

_SOURCE_RUNTIME_MANIFEST_SCHEMA: Final = (
    "alberta.hidden-learning-partner-planning.source-runtime-manifest.v1"
)
_EXECUTION_REQUEST_SCHEMA: Final = (
    "alberta.hidden-learning-partner-planning.execution-request.development.v1"
)
_EXECUTION_PERMIT_SCHEMA: Final = (
    "alberta.hidden-learning-partner-planning.execution-permit.development.v1"
)
_AUTHENTICATED_REPLAY_SCHEMA: Final = (
    "alberta.hidden-learning-partner-planning.authenticated-replay.development.v1"
)
_HOST_QUIESCENCE_POLICY_SCHEMA: Final = (
    "alberta.hidden-learning-partner-planning.host-quiescence-policy.v1"
)
_MAX_LOAD_1: Final = 8.0
_MAX_LOAD_5: Final = 8.0
_MAX_RUNNABLE_PROCESSES: Final = 4
_MAX_LOAD_PER_LOGICAL_CPU: Final = 0.25
_PERMIT_LIFETIME_NS: Final = 30_000_000_000
_PERMIT_SIGNING_KEY: Final = secrets.token_bytes(32)
_PERMIT_REGISTRY_LOCK: Final = threading.Lock()

_PROJECT_ROOT: Final = Path(__file__).resolve().parents[2]
_SOURCE_MODULES: Final = (
    ("runner", __file__),
    ("development_kernel", bridge_module.__file__),
    ("scan_plan", plan_module.__file__),
    ("world", world_module.__file__),
    ("signaling_learner", signaling_module.__file__),
    ("behavior_model", behavior_module.__file__),
    ("grounded_world_model", grounded_module.__file__),
)

_STATE_SHARED_INITIAL_FIELDS: Final = (
    "world",
    "learner",
    "behavior",
    "grounded",
    "planner_key",
    "intervention_key",
    "valid",
    "step_count",
)
_TRACE_NAMED_KEY_FIELDS: Final = (
    "helper_key_before",
    "helper_key_after",
    "beneficiary_key_before",
    "beneficiary_key_after",
    "planner_key_before",
    "planner_key_after",
    "intervention_key_before",
    "intervention_key_after",
)
_FINAL_NAMED_KEY_PATHS: Final = (
    "world.cue_key",
    "world.channel_key",
    "learner.helper.key",
    "learner.beneficiary.key",
    "planner_key",
    "intervention_key",
)


class HiddenLearningPartnerPlanningRunnerError(RuntimeError):
    """Raised when runner preflight or a completed record fails closed."""


@dataclasses.dataclass(frozen=True, slots=True)
class HiddenPlanningSourceFileHash:
    """Full-byte hash of one load-bearing implementation source."""

    role: str
    repository_path: str
    nbytes: int
    sha256: str


@dataclasses.dataclass(frozen=True, slots=True)
class HiddenPlanningSourceRuntimeManifest:
    """Exact source, JAX, PRNG, backend, and execution-mode identity."""

    schema: str
    source_files: tuple[HiddenPlanningSourceFileHash, ...]
    jax_version: str
    jaxlib_version: str
    backend: str
    device_platforms: tuple[str, ...]
    prng_impl: str
    prng_key_dtype: str
    prng_key_data_shape: tuple[int, ...]
    prng_key_data_dtype: str
    execution_mode: HiddenPlanningExecutionMode
    manifest_sha256: str


@dataclasses.dataclass(frozen=True, slots=True)
class HiddenPlanningHostQuiescenceSnapshot:
    """One live Linux host-load observation under the strict runner policy."""

    schema: str
    captured_time_ns: int
    hostname: str
    boot_id: str
    logical_cpu_count: int
    load_1: float
    load_5: float
    load_15: float
    load_1_per_logical_cpu: float
    runnable_processes: int
    max_load_1: float
    max_load_5: float
    max_load_per_logical_cpu: float
    max_runnable_processes: int
    quiescent: bool
    rejection_reasons: tuple[str, ...]
    snapshot_sha256: str


@dataclasses.dataclass(frozen=True, slots=True)
class HiddenPlanningExecutionRequest:
    """Inert exact request for the canonical development campaign."""

    schema: str
    development_only: bool
    assessment_status: str
    artifact_writes_authorized: bool
    evidence_authorized: bool
    scientific_promotion_allowed: bool
    plan_sha256: str
    config_sha256: str
    source_runtime_manifest: HiddenPlanningSourceRuntimeManifest
    seeds: tuple[int, ...]
    conditions: tuple[str, ...]
    planned_run_count: int
    life_steps: int
    execution_mode: HiddenPlanningExecutionMode
    consumption_acknowledgement: str
    host_quiescence_policy_sha256: str
    request_sha256: str


@dataclasses.dataclass(frozen=True, slots=True)
class HiddenPlanningExecutionPermit:
    """One-campaign/one-bound-replay permit issued after a live quiescence check."""

    schema: str
    development_only: bool
    artifact_writes_authorized: bool
    evidence_authorized: bool
    scientific_promotion_allowed: bool
    request_sha256: str
    source_manifest_sha256: str
    issued_time_ns: int
    expires_time_ns: int
    nonce: str
    host_snapshot: HiddenPlanningHostQuiescenceSnapshot
    permit_hmac_sha256: str


@dataclasses.dataclass(frozen=True, slots=True)
class _HiddenPlanningPermitRegistryEntry:
    """Process-local one-run/one-replay state for an authenticated permit."""

    signature: str
    request_sha256: str
    phase: HiddenPlanningPermitPhase
    suite_binding_sha256: str | None


_ISSUED_PERMITS: dict[str, _HiddenPlanningPermitRegistryEntry] = {}
_REQUEST_PERMITS: dict[str, str] = {}


@dataclasses.dataclass(frozen=True, slots=True)
class HiddenPlanningAuthenticatedReplayValidation:
    """Exact replay result; descriptive and never an acceptance verdict."""

    schema: str
    assessment_status: str
    development_only: bool
    evidence_authorized: bool
    scientific_promotion_allowed: bool
    authenticated_replay_verified: bool
    rerun_count: int
    source_runtime_manifest: HiddenPlanningSourceRuntimeManifest | None
    errors: tuple[str, ...]


@dataclasses.dataclass(frozen=True, slots=True)
class HiddenPlanningRunRequest:
    """One execution-order request with a canonical result identity."""

    execution_index: int
    seed_index: int
    seed: int
    canonical_arm_index: int
    condition: str


@dataclasses.dataclass(frozen=True, slots=True)
class HiddenPlanningEvaluatorOwnedStream:
    """Action-independent exogenous stream reconstructed before arm execution."""

    seed_index: int
    seed: int
    num_steps: int
    helper_cue: Array
    next_helper_cue: Array
    oracle_phase_index: Array
    oracle_context: Array
    oracle_target: Array
    shuffled_channel_output: Array
    cue_key_before: Array
    cue_key_after: Array
    channel_key_before: Array
    channel_key_after: Array
    initial_cue_key_data: tuple[int, int]
    initial_channel_key_data: tuple[int, int]
    final_cue_key_data: tuple[int, int]
    final_channel_key_data: tuple[int, int]


@dataclasses.dataclass(frozen=True, slots=True)
class HiddenPlanningProposalWriteAccounting:
    """Raw update opportunities, explicit markers, and committed writes."""

    num_steps: int
    active_transition_proposal_opportunities: int
    accepted_transitions: int
    rejected_active_transition_proposals: int
    helper_update_proposal_opportunities: int
    helper_committed_writes: int
    beneficiary_update_proposal_opportunities: int
    beneficiary_committed_writes: int
    behavior_update_proposal_opportunities: int
    behavior_applied_proposal_markers: int
    behavior_committed_writes: int
    grounded_update_proposal_opportunities: int
    grounded_applied_proposal_markers: int
    grounded_committed_writes: int
    planner_proposal_opportunities: int
    planner_consumptions: int


@dataclasses.dataclass(frozen=True, slots=True)
class HiddenPlanningMatchedRunRecord:
    """One canonical raw record; no threshold or aggregate assessment."""

    record_index: int
    seed_index: int
    seed: int
    seed_role: str
    canonical_arm_index: int
    condition: str
    assessment_status: str
    run: HiddenLearningPartnerPlanningRun
    phase_diagnostics: HiddenLearningPartnerPhaseDiagnostics
    proposal_write_accounting: HiddenPlanningProposalWriteAccounting
    strict_run_validation_errors: tuple[str, ...]
    environment_stream_errors: tuple[str, ...]


@dataclasses.dataclass(frozen=True, slots=True)
class HiddenPlanningCommonRandomNumberAudit:
    """Threshold-free structural audit of the complete matched panel."""

    paired_seed_count: int
    arm_count: int
    record_count: int
    evaluator_stream_reconstruction_passed: bool
    action_independent_environment_parity_passed: bool
    shared_initial_state_parity_passed: bool
    cross_arm_trace_key_parity_passed: bool
    final_named_key_parity_passed: bool
    shuffled_channel_output_binding_passed: bool
    canonical_record_order_passed: bool
    errors: tuple[str, ...]


@dataclasses.dataclass(frozen=True, slots=True)
class HiddenLearningPartnerPlanningMatchedSuite:
    """Complete in-memory development panel returned after explicit execution."""

    schema: str
    status: str
    assessment_status: str
    development_only: bool
    seed_role: str
    consumed_development_seeds: tuple[int, ...]
    held_out_seeds_used: bool
    evidence_authorized: bool
    scientific_promotion_allowed: bool
    artifact_writes_authorized: bool
    execution_acknowledgement: str
    source_plan_sha256: str
    source_runtime_manifest: HiddenPlanningSourceRuntimeManifest
    execution_request_sha256: str
    execution_permit_hmac_sha256: str
    suite_binding_sha256: str
    authenticated_replay_verified: bool
    canonical_condition_order: tuple[str, ...]
    canonical_record_order: bool
    raw_records_present: bool
    evaluator_streams: tuple[HiddenPlanningEvaluatorOwnedStream, ...]
    records: tuple[HiddenPlanningMatchedRunRecord, ...]
    common_random_number_audit: HiddenPlanningCommonRandomNumberAudit
    aggregate_statistics: None
    thresholds: None
    artifact_output_path: None


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


class _HashWriter(Protocol):
    def update(self, data: bytes, /) -> object:
        """Add bytes to an incremental digest."""


def _hash_part(digest: _HashWriter, payload: bytes) -> None:
    """Length-frame one exact-object hash component."""

    digest.update(len(payload).to_bytes(8, byteorder="big", signed=False))
    digest.update(payload)


def _update_exact_object_hash(digest: _HashWriter, value: object) -> None:
    """Hash supported result values with concrete types, shapes, and bits."""

    type_name = f"{type(value).__module__}.{type(value).__qualname__}".encode()
    _hash_part(digest, type_name)
    if isinstance(value, jax.Array):
        is_key = bool(jnp.issubdtype(value.dtype, jax.dtypes.prng_key))
        _hash_part(digest, b"typed_prng_key" if is_key else b"jax_array")
        if is_key:
            _hash_part(digest, str(jr.key_impl(value)).encode("ascii"))
            array = np.asarray(jax.device_get(jr.key_data(value)))
        else:
            array = np.asarray(jax.device_get(value))
        _hash_part(digest, array.dtype.str.encode("ascii"))
        _hash_part(digest, _canonical_json_bytes(tuple(int(size) for size in array.shape)))
        _hash_part(digest, array.tobytes(order="C"))
        return
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        for field in dataclasses.fields(value):
            _hash_part(digest, field.name.encode("utf-8"))
            _update_exact_object_hash(digest, getattr(value, field.name))
        return
    if type(value) is tuple:
        items = cast(tuple[object, ...], value)
        _hash_part(digest, len(items).to_bytes(8, byteorder="big", signed=False))
        for item in items:
            _update_exact_object_hash(digest, item)
        return
    if value is None:
        _hash_part(digest, b"none")
        return
    if type(value) is bool:
        _hash_part(digest, b"1" if value else b"0")
        return
    if type(value) is int:
        _hash_part(digest, str(value).encode("ascii"))
        return
    if type(value) is float:
        _hash_part(
            digest,
            np.asarray(value, dtype=np.float64).view(np.uint8).tobytes(order="C"),
        )
        return
    if type(value) is str:
        _hash_part(digest, value.encode("utf-8"))
        return
    raise HiddenLearningPartnerPlanningRunnerError(
        f"unsupported exact suite-binding type: {type(value).__qualname__}"
    )


def _suite_binding_sha256(suite: HiddenLearningPartnerPlanningMatchedSuite) -> str:
    """Bind every suite scalar and every nested array/key bit without recursion cycles."""

    digest = hashlib.sha256()
    _update_exact_object_hash(
        digest,
        dataclasses.replace(suite, suite_binding_sha256=""),
    )
    return digest.hexdigest()


def _dataclass_payload_without(value: object, field_name: str) -> dict[str, object]:
    if not dataclasses.is_dataclass(value) or isinstance(value, type):
        raise HiddenLearningPartnerPlanningRunnerError(
            "canonical hashing requires one concrete dataclass instance"
        )
    payload = cast(dict[str, object], dataclasses.asdict(value))
    payload.pop(field_name, None)
    return payload


def _normalize_execution_mode(mode: object) -> HiddenPlanningExecutionMode:
    if mode == "jit":
        return "jit"
    if mode == "eager":
        return "eager"
    raise HiddenLearningPartnerPlanningRunnerError("execution_mode must be 'jit' or 'eager'")


def _source_file_hash(role: str, file_name: object) -> HiddenPlanningSourceFileHash:
    if type(file_name) is not str:
        raise HiddenLearningPartnerPlanningRunnerError(
            f"load-bearing source {role!r} has no concrete file path"
        )
    path = Path(file_name).resolve(strict=True)
    try:
        repository_path = path.relative_to(_PROJECT_ROOT).as_posix()
    except ValueError as exc:
        raise HiddenLearningPartnerPlanningRunnerError(
            f"load-bearing source {role!r} is outside the repository"
        ) from exc
    payload = path.read_bytes()
    return HiddenPlanningSourceFileHash(
        role=role,
        repository_path=repository_path,
        nbytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def build_hidden_learning_partner_source_runtime_manifest(
    *,
    execution_mode: object,
) -> HiddenPlanningSourceRuntimeManifest:
    """Hash every load-bearing source byte plus the exact JAX/PRNG runtime."""

    mode = _normalize_execution_mode(execution_mode)
    source_files = tuple(_source_file_hash(role, file_name) for role, file_name in _SOURCE_MODULES)
    key = jr.key(0, impl="threefry2x32")
    key_data = jr.key_data(key)
    devices = tuple(sorted({device.platform for device in jax.devices()}))
    provisional = HiddenPlanningSourceRuntimeManifest(
        schema=_SOURCE_RUNTIME_MANIFEST_SCHEMA,
        source_files=source_files,
        jax_version=str(jax.__version__),
        jaxlib_version=version("jaxlib"),
        backend=str(jax.default_backend()),
        device_platforms=devices,
        prng_impl=str(jr.key_impl(key)),
        prng_key_dtype=str(key.dtype),
        prng_key_data_shape=tuple(int(size) for size in key_data.shape),
        prng_key_data_dtype=str(key_data.dtype),
        execution_mode=mode,
        manifest_sha256="",
    )
    digest = _sha256_json(_dataclass_payload_without(provisional, "manifest_sha256"))
    return dataclasses.replace(provisional, manifest_sha256=digest)


def _host_policy_sha256() -> str:
    return _sha256_json(
        {
            "schema": _HOST_QUIESCENCE_POLICY_SCHEMA,
            "max_load_1": _MAX_LOAD_1,
            "max_load_5": _MAX_LOAD_5,
            "max_load_per_logical_cpu": _MAX_LOAD_PER_LOGICAL_CPU,
            "max_runnable_processes": _MAX_RUNNABLE_PROCESSES,
            "permit_lifetime_ns": _PERMIT_LIFETIME_NS,
        }
    )


def _host_rejection_reasons(
    *,
    load_1: float,
    load_5: float,
    load_15: float,
    load_per_cpu: float,
    runnable_processes: int,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if not np.isfinite((load_1, load_5, load_15, load_per_cpu)).all():
        reasons.append("load values are non-finite")
    if load_1 > _MAX_LOAD_1:
        reasons.append(f"load_1 {load_1:.3f} exceeds {_MAX_LOAD_1:.3f}")
    if load_5 > _MAX_LOAD_5:
        reasons.append(f"load_5 {load_5:.3f} exceeds {_MAX_LOAD_5:.3f}")
    if load_per_cpu > _MAX_LOAD_PER_LOGICAL_CPU:
        reasons.append(
            f"load_1_per_logical_cpu {load_per_cpu:.6f} exceeds {_MAX_LOAD_PER_LOGICAL_CPU:.6f}"
        )
    if runnable_processes > _MAX_RUNNABLE_PROCESSES:
        reasons.append(f"runnable_processes {runnable_processes} exceeds {_MAX_RUNNABLE_PROCESSES}")
    return tuple(reasons)


def _capture_host_quiescence() -> HiddenPlanningHostQuiescenceSnapshot:
    """Capture live Linux load; failure to observe any field rejects closed."""

    try:
        load_text = Path("/proc/loadavg").read_text(encoding="ascii").strip().split()
        if len(load_text) < 4:
            raise ValueError("/proc/loadavg has fewer than four fields")
        load_1, load_5, load_15 = (float(load_text[index]) for index in range(3))
        runnable_text = load_text[3].split("/", maxsplit=1)[0]
        runnable_processes = int(runnable_text)
        logical_cpu_count = os.cpu_count()
        if type(logical_cpu_count) is not int or logical_cpu_count < 1:
            raise ValueError("logical CPU count is unavailable")
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
        if not boot_id:
            raise ValueError("boot id is empty")
    except (OSError, TypeError, ValueError) as exc:
        raise HiddenLearningPartnerPlanningRunnerError(
            f"live host-quiescence observation failed closed: {exc}"
        ) from exc
    load_per_cpu = load_1 / float(logical_cpu_count)
    reasons = _host_rejection_reasons(
        load_1=load_1,
        load_5=load_5,
        load_15=load_15,
        load_per_cpu=load_per_cpu,
        runnable_processes=runnable_processes,
    )
    provisional = HiddenPlanningHostQuiescenceSnapshot(
        schema=_HOST_QUIESCENCE_POLICY_SCHEMA,
        captured_time_ns=time.time_ns(),
        hostname=socket.gethostname(),
        boot_id=boot_id,
        logical_cpu_count=logical_cpu_count,
        load_1=load_1,
        load_5=load_5,
        load_15=load_15,
        load_1_per_logical_cpu=load_per_cpu,
        runnable_processes=runnable_processes,
        max_load_1=_MAX_LOAD_1,
        max_load_5=_MAX_LOAD_5,
        max_load_per_logical_cpu=_MAX_LOAD_PER_LOGICAL_CPU,
        max_runnable_processes=_MAX_RUNNABLE_PROCESSES,
        quiescent=not reasons,
        rejection_reasons=reasons,
        snapshot_sha256="",
    )
    digest = _sha256_json(_dataclass_payload_without(provisional, "snapshot_sha256"))
    return dataclasses.replace(provisional, snapshot_sha256=digest)


def _host_snapshot_errors(snapshot: object) -> tuple[str, ...]:
    """Validate the complete nested snapshot and independently derive its verdict."""

    if type(snapshot) is not HiddenPlanningHostQuiescenceSnapshot:
        return ("host snapshot must have the exact concrete type",)
    checked = snapshot
    errors: list[str] = []
    expected_types: tuple[tuple[str, type[object]], ...] = (
        ("schema", str),
        ("captured_time_ns", int),
        ("hostname", str),
        ("boot_id", str),
        ("logical_cpu_count", int),
        ("load_1", float),
        ("load_5", float),
        ("load_15", float),
        ("load_1_per_logical_cpu", float),
        ("runnable_processes", int),
        ("max_load_1", float),
        ("max_load_5", float),
        ("max_load_per_logical_cpu", float),
        ("max_runnable_processes", int),
        ("quiescent", bool),
        ("rejection_reasons", tuple),
        ("snapshot_sha256", str),
    )
    for name, expected_type in expected_types:
        if type(getattr(checked, name)) is not expected_type:
            errors.append(f"host snapshot {name} has the wrong concrete type")
    if errors:
        return tuple(errors)
    if checked.schema != _HOST_QUIESCENCE_POLICY_SCHEMA:
        errors.append("host snapshot policy schema differs")
    if checked.captured_time_ns < 1 or not checked.hostname or not checked.boot_id:
        errors.append("host snapshot identity or capture time is invalid")
    if checked.logical_cpu_count < 1 or checked.runnable_processes < 0:
        errors.append("host snapshot CPU/process counts are invalid")
    if (
        checked.max_load_1 != _MAX_LOAD_1
        or checked.max_load_5 != _MAX_LOAD_5
        or checked.max_load_per_logical_cpu != _MAX_LOAD_PER_LOGICAL_CPU
        or checked.max_runnable_processes != _MAX_RUNNABLE_PROCESSES
    ):
        errors.append("host snapshot thresholds differ from the bound policy")
    if checked.load_1_per_logical_cpu != (checked.load_1 / float(checked.logical_cpu_count)):
        errors.append("host snapshot per-CPU load is not reconstructed exactly")
    expected_reasons = _host_rejection_reasons(
        load_1=checked.load_1,
        load_5=checked.load_5,
        load_15=checked.load_15,
        load_per_cpu=checked.load_1_per_logical_cpu,
        runnable_processes=checked.runnable_processes,
    )
    if type(checked.rejection_reasons) is tuple and any(
        type(reason) is not str for reason in checked.rejection_reasons
    ):
        errors.append("host snapshot rejection reasons must be exact strings")
    if checked.rejection_reasons != expected_reasons:
        errors.append("host snapshot rejection reasons differ from reconstruction")
    if checked.quiescent is not (not expected_reasons):
        errors.append("host snapshot quiescence verdict differs from reconstruction")
    expected_digest = _sha256_json(_dataclass_payload_without(checked, "snapshot_sha256"))
    if checked.snapshot_sha256 != expected_digest:
        errors.append("host snapshot digest does not bind its complete contents")
    return tuple(dict.fromkeys(errors))


def build_hidden_learning_partner_execution_request(
    plan: object,
    *,
    execution_mode: object,
    consumption_acknowledgement: str,
) -> HiddenPlanningExecutionRequest:
    """Build an inert request; this neither checks load nor issues a permit."""

    checked = _canonical_plan(plan)
    mode = _normalize_execution_mode(execution_mode)
    if type(consumption_acknowledgement) is not str or (
        consumption_acknowledgement != DEVELOPMENT_EXECUTION_ACKNOWLEDGEMENT
    ):
        raise HiddenLearningPartnerPlanningRunnerError(
            "exact consumed-development execution acknowledgement is required"
        )
    manifest = build_hidden_learning_partner_source_runtime_manifest(execution_mode=mode)
    provisional = HiddenPlanningExecutionRequest(
        schema=_EXECUTION_REQUEST_SCHEMA,
        development_only=True,
        assessment_status=ASSESSMENT_STATUS,
        artifact_writes_authorized=False,
        evidence_authorized=False,
        scientific_promotion_allowed=False,
        plan_sha256=checked.plan_sha256,
        config_sha256=checked.config_sha256,
        source_runtime_manifest=manifest,
        seeds=tuple(binding.seed for binding in checked.seed_contract.bindings),
        conditions=CANONICAL_CONDITION_ORDER,
        planned_run_count=checked.counts.planned_run_count,
        life_steps=checked.life_steps,
        execution_mode=mode,
        consumption_acknowledgement=consumption_acknowledgement,
        host_quiescence_policy_sha256=_host_policy_sha256(),
        request_sha256="",
    )
    digest = _sha256_json(_dataclass_payload_without(provisional, "request_sha256"))
    return dataclasses.replace(provisional, request_sha256=digest)


def validate_hidden_learning_partner_execution_request(
    request: object,
    *,
    plan: object,
) -> tuple[str, ...]:
    """Rebuild every request binding and return deterministic errors."""

    if type(request) is not HiddenPlanningExecutionRequest:
        return ("request must be an exact HiddenPlanningExecutionRequest",)
    try:
        checked = _canonical_plan(plan)
        expected = build_hidden_learning_partner_execution_request(
            checked,
            execution_mode=request.execution_mode,
            consumption_acknowledgement=request.consumption_acknowledgement,
        )
    except (AttributeError, OSError, TypeError, ValueError, RuntimeError) as exc:
        return (f"execution request reconstruction failed closed: {exc}",)
    errors: list[str] = []
    if request != expected:
        errors.append("execution request differs from exact canonical reconstruction")
    try:
        actual_digest = _sha256_json(_dataclass_payload_without(request, "request_sha256"))
    except (TypeError, ValueError, RuntimeError) as exc:
        errors.append(f"execution request hashing failed closed: {exc}")
    else:
        if request.request_sha256 != actual_digest:
            errors.append("execution request digest does not bind its complete contents")
    if not _is_sha256_hex(request.request_sha256):
        errors.append("execution request digest is not exact lowercase SHA-256 hex")
    if request.plan_sha256 != checked.plan_sha256:
        errors.append("execution request plan digest differs")
    if any(
        value is not False
        for value in (
            request.artifact_writes_authorized,
            request.evidence_authorized,
            request.scientific_promotion_allowed,
        )
    ):
        errors.append("execution request carries forbidden authority")
    return tuple(dict.fromkeys(errors))


def _permit_payload(permit: HiddenPlanningExecutionPermit) -> dict[str, object]:
    return _dataclass_payload_without(permit, "permit_hmac_sha256")


def _is_sha256_hex(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _permit_contract_errors(permit: HiddenPlanningExecutionPermit) -> tuple[str, ...]:
    """Validate exact permit scalar/nested types before authentication arithmetic."""

    errors: list[str] = []
    for name in (
        "schema",
        "request_sha256",
        "source_manifest_sha256",
        "nonce",
        "permit_hmac_sha256",
    ):
        if type(getattr(permit, name)) is not str:
            errors.append(f"permit {name} has the wrong concrete type")
    for name in ("issued_time_ns", "expires_time_ns"):
        if type(getattr(permit, name)) is not int:
            errors.append(f"permit {name} has the wrong concrete type")
    for name in (
        "development_only",
        "artifact_writes_authorized",
        "evidence_authorized",
        "scientific_promotion_allowed",
    ):
        if type(getattr(permit, name)) is not bool:
            errors.append(f"permit {name} has the wrong concrete type")
    if type(permit.host_snapshot) is not HiddenPlanningHostQuiescenceSnapshot:
        errors.append("permit host snapshot has the wrong concrete type")
    if errors:
        return tuple(errors)
    if not _is_sha256_hex(permit.request_sha256):
        errors.append("permit request digest is not exact lowercase SHA-256 hex")
    if not _is_sha256_hex(permit.source_manifest_sha256):
        errors.append("permit source manifest digest is not exact lowercase SHA-256 hex")
    if not _is_sha256_hex(permit.permit_hmac_sha256):
        errors.append("permit HMAC is not exact lowercase SHA-256 hex")
    if len(permit.nonce) != 64 or any(
        character not in "0123456789abcdef" for character in permit.nonce
    ):
        errors.append("permit nonce is not exact lowercase 256-bit hex")
    if (
        permit.issued_time_ns < 1
        or permit.expires_time_ns - permit.issued_time_ns != _PERMIT_LIFETIME_NS
    ):
        errors.append("permit validity interval differs from the strict policy")
    if permit.development_only is not True:
        errors.append("permit is not development-only")
    errors.extend(_host_snapshot_errors(permit.host_snapshot))
    return tuple(dict.fromkeys(errors))


def issue_hidden_learning_partner_execution_permit(
    request: object,
    *,
    plan: object,
) -> HiddenPlanningExecutionPermit:
    """Issue a short-lived in-process permit only on a live quiescent host."""

    request_errors = validate_hidden_learning_partner_execution_request(
        request,
        plan=plan,
    )
    if request_errors:
        raise HiddenLearningPartnerPlanningRunnerError(
            "execution request is invalid: " + "; ".join(request_errors)
        )
    checked_request = cast(HiddenPlanningExecutionRequest, request)
    snapshot = _capture_host_quiescence()
    snapshot_errors = _host_snapshot_errors(snapshot)
    if snapshot_errors:
        raise HiddenLearningPartnerPlanningRunnerError(
            "host-quiescence snapshot is invalid: " + "; ".join(snapshot_errors)
        )
    if not snapshot.quiescent:
        raise HiddenLearningPartnerPlanningRunnerError(
            "host is not quiescent: " + "; ".join(snapshot.rejection_reasons)
        )
    issued = time.time_ns()
    provisional = HiddenPlanningExecutionPermit(
        schema=_EXECUTION_PERMIT_SCHEMA,
        development_only=True,
        artifact_writes_authorized=False,
        evidence_authorized=False,
        scientific_promotion_allowed=False,
        request_sha256=checked_request.request_sha256,
        source_manifest_sha256=(checked_request.source_runtime_manifest.manifest_sha256),
        issued_time_ns=issued,
        expires_time_ns=issued + _PERMIT_LIFETIME_NS,
        nonce=secrets.token_hex(32),
        host_snapshot=snapshot,
        permit_hmac_sha256="",
    )
    signature = hmac.new(
        _PERMIT_SIGNING_KEY,
        _canonical_json_bytes(_permit_payload(provisional)),
        hashlib.sha256,
    ).hexdigest()
    permit = dataclasses.replace(provisional, permit_hmac_sha256=signature)
    with _PERMIT_REGISTRY_LOCK:
        prior_nonce = _REQUEST_PERMITS.get(checked_request.request_sha256)
        if prior_nonce is not None:
            raise HiddenLearningPartnerPlanningRunnerError(
                "this exact execution request already has a process-local permit"
            )
        _ISSUED_PERMITS[permit.nonce] = _HiddenPlanningPermitRegistryEntry(
            signature=signature,
            request_sha256=checked_request.request_sha256,
            phase="issued",
            suite_binding_sha256=None,
        )
        _REQUEST_PERMITS[checked_request.request_sha256] = permit.nonce
    return permit


def _require_live_execution_permit(
    request: object,
    permit: object,
    *,
    plan: object,
    purpose: Literal["run", "replay"],
) -> tuple[HiddenPlanningExecutionRequest, HiddenPlanningExecutionPermit]:
    """Reject missing, forged, source-drifted, wrong-phase, or high-load permits.

    The short issuance interval gates campaign start.  Its one suite-bound
    replay may occur later, but still requires unchanged source/runtime bytes
    and a fresh strict live host check.
    """

    if type(request) is not HiddenPlanningExecutionRequest:
        raise HiddenLearningPartnerPlanningRunnerError(
            "an exact execution request is required before any execution work"
        )
    if type(permit) is not HiddenPlanningExecutionPermit:
        raise HiddenLearningPartnerPlanningRunnerError(
            "an exact execution permit is required before any execution work"
        )
    if type(request.source_runtime_manifest) is not HiddenPlanningSourceRuntimeManifest:
        raise HiddenLearningPartnerPlanningRunnerError(
            "execution request source/runtime manifest has the wrong nested type"
        )
    permit_contract_errors = _permit_contract_errors(permit)
    if permit_contract_errors:
        raise HiddenLearningPartnerPlanningRunnerError(
            "execution permit contract is invalid: " + "; ".join(permit_contract_errors)
        )
    request_errors = validate_hidden_learning_partner_execution_request(request, plan=plan)
    if request_errors:
        raise HiddenLearningPartnerPlanningRunnerError(
            "execution request failed revalidation: " + "; ".join(request_errors)
        )
    expected_signature = hmac.new(
        _PERMIT_SIGNING_KEY,
        _canonical_json_bytes(_permit_payload(permit)),
        hashlib.sha256,
    ).hexdigest()
    with _PERMIT_REGISTRY_LOCK:
        registered = _ISSUED_PERMITS.get(permit.nonce)
    now = time.time_ns()
    errors: list[str] = []
    if permit.schema != _EXECUTION_PERMIT_SCHEMA:
        errors.append("permit schema differs")
    if not hmac.compare_digest(permit.permit_hmac_sha256, expected_signature):
        errors.append("permit HMAC authentication failed")
    if registered is None or not hmac.compare_digest(
        permit.permit_hmac_sha256,
        registered.signature if registered is not None else "",
    ):
        errors.append("permit was not issued by this live runner process")
    elif registered.request_sha256 != request.request_sha256:
        errors.append("permit registry is bound to a different request")
    elif purpose == "run" and registered.phase != "issued":
        errors.append("permit campaign execution has already been consumed")
    elif purpose == "replay" and (
        registered.phase != "run_consumed" or registered.suite_binding_sha256 is None
    ):
        errors.append("permit replay requires one completed bound campaign run")
    if permit.request_sha256 != request.request_sha256:
        errors.append("permit is bound to a different request")
    if permit.source_manifest_sha256 != request.source_runtime_manifest.manifest_sha256:
        errors.append("permit is bound to a different source/runtime manifest")
    if now < permit.issued_time_ns or (purpose == "run" and now > permit.expires_time_ns):
        errors.append("permit is not within its short live validity interval")
    if any(
        value is not False
        for value in (
            permit.artifact_writes_authorized,
            permit.evidence_authorized,
            permit.scientific_promotion_allowed,
        )
    ):
        errors.append("permit carries forbidden authority")
    if not permit.host_snapshot.quiescent:
        errors.append("permit issuance snapshot was not quiescent")
    if errors:
        raise HiddenLearningPartnerPlanningRunnerError(
            "execution permit is invalid: " + "; ".join(errors)
        )
    live_snapshot = _capture_host_quiescence()
    live_snapshot_errors = _host_snapshot_errors(live_snapshot)
    if live_snapshot_errors:
        raise HiddenLearningPartnerPlanningRunnerError(
            "fresh live host snapshot is invalid: " + "; ".join(live_snapshot_errors)
        )
    if not live_snapshot.quiescent:
        raise HiddenLearningPartnerPlanningRunnerError(
            "host is no longer quiescent: " + "; ".join(live_snapshot.rejection_reasons)
        )
    if (
        live_snapshot.hostname != permit.host_snapshot.hostname
        or live_snapshot.boot_id != permit.host_snapshot.boot_id
        or live_snapshot.logical_cpu_count != permit.host_snapshot.logical_cpu_count
    ):
        raise HiddenLearningPartnerPlanningRunnerError(
            "live host identity differs from the permit issuance host"
        )
    return request, permit


def _consume_permit_for_run(permit: HiddenPlanningExecutionPermit) -> None:
    """Atomically spend the campaign phase before stream or learner work."""

    with _PERMIT_REGISTRY_LOCK:
        registered = _ISSUED_PERMITS.get(permit.nonce)
        if registered is None or not hmac.compare_digest(
            registered.signature,
            permit.permit_hmac_sha256,
        ):
            raise HiddenLearningPartnerPlanningRunnerError(
                "execution permit disappeared before campaign consumption"
            )
        if registered.phase != "issued":
            raise HiddenLearningPartnerPlanningRunnerError(
                "execution permit campaign phase was already consumed"
            )
        _ISSUED_PERMITS[permit.nonce] = dataclasses.replace(
            registered,
            phase="run_consumed",
        )


def _bind_completed_suite_to_permit(
    permit: HiddenPlanningExecutionPermit,
    *,
    suite_binding_sha256: str,
) -> None:
    """Bind the one completed raw suite that may use the replay phase."""

    with _PERMIT_REGISTRY_LOCK:
        registered = _ISSUED_PERMITS.get(permit.nonce)
        if registered is None or registered.phase != "run_consumed":
            raise HiddenLearningPartnerPlanningRunnerError(
                "execution permit is not in the consumed campaign phase"
            )
        if registered.suite_binding_sha256 is not None:
            raise HiddenLearningPartnerPlanningRunnerError(
                "execution permit already has a completed suite binding"
            )
        _ISSUED_PERMITS[permit.nonce] = dataclasses.replace(
            registered,
            suite_binding_sha256=suite_binding_sha256,
        )


def _consume_permit_for_replay(
    permit: HiddenPlanningExecutionPermit,
    *,
    suite_binding_sha256: str,
) -> None:
    """Atomically spend the sole replay phase before any replayed learner life."""

    with _PERMIT_REGISTRY_LOCK:
        registered = _ISSUED_PERMITS.get(permit.nonce)
        if registered is None or not hmac.compare_digest(
            registered.signature,
            permit.permit_hmac_sha256,
        ):
            raise HiddenLearningPartnerPlanningRunnerError(
                "execution permit disappeared before replay consumption"
            )
        if registered.phase != "run_consumed":
            raise HiddenLearningPartnerPlanningRunnerError(
                "execution permit replay phase is unavailable or already consumed"
            )
        if registered.suite_binding_sha256 != suite_binding_sha256:
            raise HiddenLearningPartnerPlanningRunnerError(
                "execution permit replay is bound to a different completed suite"
            )
        _ISSUED_PERMITS[permit.nonce] = dataclasses.replace(
            registered,
            phase="replay_consumed",
        )


def _canonical_plan(plan: object) -> HiddenLearningPartnerPlanningScanPlan:
    """Reconstruct and require the exact canonical plan, wrapping failures."""

    try:
        checked = plan_module.require_valid_hidden_learning_partner_planning_scan_plan(plan)
        canonical = plan_module.build_hidden_learning_partner_planning_scan_plan()
    except (TypeError, ValueError, RuntimeError) as exc:
        raise HiddenLearningPartnerPlanningRunnerError(
            f"canonical scan-plan validation failed: {exc}"
        ) from exc
    if checked != canonical:
        raise HiddenLearningPartnerPlanningRunnerError(
            "validated scan plan differs from a fresh canonical reconstruction"
        )
    live_schema = getattr(bridge_module, "HIDDEN_LEARNING_PARTNER_PLANNING_SCHEMA", None)
    if (
        checked.bridge_schema != "alberta.hidden-learning-partner-planning.development.v1"
        or live_schema != checked.bridge_schema
    ):
        raise HiddenLearningPartnerPlanningRunnerError(
            "runner requires the exact development.v1 source kernel schema"
        )
    budget = checked.resource_budget
    resource_values = (
        budget.signaling_state_nbytes,
        budget.behavior_state_nbytes,
        budget.grounded_state_nbytes,
        budget.world_state_nbytes,
        budget.total_state_nbytes,
    )
    if resource_values != (80, 48, 108, 32, 321) or budget.exact_tree_match is not True:
        raise HiddenLearningPartnerPlanningRunnerError(
            "runner requires the exact 80/48/108/32/321-byte source resource contract"
        )
    crn = checked.common_random_numbers
    required_true = (
        crn.same_seed_set_every_arm,
        crn.same_root_key_for_seed_every_arm,
        crn.fresh_state_per_seed_condition,
        crn.world_cue_stream_reconstruction_required,
        crn.world_channel_stream_reconstruction_required,
        crn.branch_invariant_persistent_key_advancement_required,
        crn.shuffled_channel_output_binding_required,
    )
    required_false = (
        crn.condition_is_key_derivation_input,
        crn.arm_order_is_key_derivation_input,
        crn.cross_arm_state_reuse_allowed,
    )
    if not all(value is True for value in required_true) or not all(
        value is False for value in required_false
    ):
        raise HiddenLearningPartnerPlanningRunnerError(
            "canonical common-random-number obligations are not runner-compatible"
        )
    if crn.allowed_initial_state_difference_fields != ("config_token",):
        raise HiddenLearningPartnerPlanningRunnerError(
            "runner only supports config_token as an initial cross-arm difference"
        )
    if crn.required_cross_arm_trace_key_fields != _TRACE_NAMED_KEY_FIELDS:
        raise HiddenLearningPartnerPlanningRunnerError(
            "runner cross-arm trace-key audit differs from the canonical plan"
        )
    if tuple(binding.seed for binding in checked.seed_contract.bindings) != (
        PAIRED_DEVELOPMENT_SEEDS
    ):
        raise HiddenLearningPartnerPlanningRunnerError(
            "runner seed schedule differs from the exact four development roots"
        )
    return checked


def _normalize_arm_order(arm_order: object) -> tuple[str, ...]:
    if type(arm_order) is not tuple:
        raise HiddenLearningPartnerPlanningRunnerError("arm_order must be an exact tuple")
    checked = cast(tuple[object, ...], arm_order)
    if any(type(condition) is not str for condition in checked):
        raise HiddenLearningPartnerPlanningRunnerError(
            "every arm_order entry must be an exact string"
        )
    normalized = cast(tuple[str, ...], checked)
    if len(normalized) != len(CANONICAL_CONDITION_ORDER):
        raise HiddenLearningPartnerPlanningRunnerError(
            "arm_order must contain all eleven declared conditions exactly once"
        )
    if len(set(normalized)) != len(normalized):
        raise HiddenLearningPartnerPlanningRunnerError("arm_order contains a duplicate condition")
    if set(normalized) != set(CANONICAL_CONDITION_ORDER):
        raise HiddenLearningPartnerPlanningRunnerError(
            "arm_order is missing or adds a declared condition"
        )
    return normalized


def build_hidden_learning_partner_planning_run_schedule(
    plan: object,
    *,
    arm_order: object = CANONICAL_CONDITION_ORDER,
) -> tuple[HiddenPlanningRunRequest, ...]:
    """Build an inert full-panel schedule without initializing any learner."""

    checked = _canonical_plan(plan)
    execution_order = _normalize_arm_order(arm_order)
    canonical_indices = {
        condition: index for index, condition in enumerate(CANONICAL_CONDITION_ORDER)
    }
    requests: list[HiddenPlanningRunRequest] = []
    for binding in checked.seed_contract.bindings:
        for condition in execution_order:
            requests.append(
                HiddenPlanningRunRequest(
                    execution_index=len(requests),
                    seed_index=binding.seed_index,
                    seed=binding.seed,
                    canonical_arm_index=canonical_indices[condition],
                    condition=condition,
                )
            )
    if len(requests) != checked.counts.planned_run_count:
        raise HiddenLearningPartnerPlanningRunnerError(
            "run schedule differs from the canonical planned-run count"
        )
    return tuple(requests)


def canonical_hidden_learning_partner_planning_record_keys(
    plan: object,
) -> tuple[tuple[int, str], ...]:
    """Return the result identities, independent of evaluator loop order."""

    checked = _canonical_plan(plan)
    return tuple(
        (binding.seed, condition)
        for binding in checked.seed_contract.bindings
        for condition in CANONICAL_CONDITION_ORDER
    )


def canonicalize_hidden_learning_partner_planning_records(
    records: object,
    *,
    seed_order: tuple[int, ...],
) -> tuple[HiddenPlanningMatchedRunRecord, ...]:
    """Canonicalize a complete all-condition panel independent of loop order."""

    if type(seed_order) is not tuple or not seed_order:
        raise HiddenLearningPartnerPlanningRunnerError("seed_order must be a nonempty exact tuple")
    if any(type(seed) is not int or seed < 0 for seed in seed_order):
        raise HiddenLearningPartnerPlanningRunnerError(
            "seed_order must contain non-negative built-in integers"
        )
    if len(set(seed_order)) != len(seed_order):
        raise HiddenLearningPartnerPlanningRunnerError("seed_order contains duplicates")
    if type(records) is not tuple:
        raise HiddenLearningPartnerPlanningRunnerError("records must be an exact tuple")
    checked = cast(tuple[object, ...], records)
    expected_count = len(seed_order) * len(CANONICAL_CONDITION_ORDER)
    if len(checked) != expected_count:
        raise HiddenLearningPartnerPlanningRunnerError(
            "records do not cover every seed/condition identity"
        )
    by_position: dict[tuple[int, int], HiddenPlanningMatchedRunRecord] = {}
    seed_indices = {seed: index for index, seed in enumerate(seed_order)}
    arm_indices = {condition: index for index, condition in enumerate(CANONICAL_CONDITION_ORDER)}
    for item in checked:
        if type(item) is not HiddenPlanningMatchedRunRecord:
            raise HiddenLearningPartnerPlanningRunnerError(
                "every record must have the exact matched-run record type"
            )
        record = item
        expected_seed_index = seed_indices.get(record.seed)
        expected_arm_index = arm_indices.get(record.condition)
        if expected_seed_index is None or expected_arm_index is None:
            raise HiddenLearningPartnerPlanningRunnerError(
                "record contains an undeclared seed or condition"
            )
        expected_record_index = (
            expected_seed_index * len(CANONICAL_CONDITION_ORDER) + expected_arm_index
        )
        if (
            record.seed_index != expected_seed_index
            or record.canonical_arm_index != expected_arm_index
            or record.record_index != expected_record_index
        ):
            raise HiddenLearningPartnerPlanningRunnerError(
                "record identity metadata differs from its canonical position"
            )
        position = (expected_seed_index, expected_arm_index)
        if position in by_position:
            raise HiddenLearningPartnerPlanningRunnerError(
                "records contain a duplicate seed/condition identity"
            )
        by_position[position] = record
    expected_positions = tuple(
        (seed_index, arm_index)
        for seed_index in range(len(seed_order))
        for arm_index in range(len(CANONICAL_CONDITION_ORDER))
    )
    if set(by_position) != set(expected_positions):
        raise HiddenLearningPartnerPlanningRunnerError(
            "records are missing a canonical seed/condition identity"
        )
    return tuple(by_position[position] for position in expected_positions)


def _key_data_tuple(key: Array) -> tuple[int, int]:
    data = np.asarray(jr.key_data(key), dtype=np.uint32)
    if data.shape != (2,) or str(jr.key_impl(key)) != "threefry2x32":
        raise HiddenLearningPartnerPlanningRunnerError(
            "evaluator stream contains a noncanonical PRNG key"
        )
    return int(data[0]), int(data[1])


def _binding_key(binding: HiddenPlanningSeedBinding, name: str, *, impl: str) -> Array:
    named = dict(binding.named_key_data)
    if tuple(named) != (
        "world.cue",
        "world.channel",
        "learner.helper",
        "learner.beneficiary",
        "behavior.initialization",
        "grounded.initialization",
        "planner",
        "intervention",
    ):
        raise HiddenLearningPartnerPlanningRunnerError(
            "seed binding named-key order differs from the canonical contract"
        )
    try:
        words = named[name]
    except KeyError as exc:  # pragma: no cover - canonical plan validation binds this.
        raise HiddenLearningPartnerPlanningRunnerError(
            f"seed binding is missing named key {name!r}"
        ) from exc
    return cast(
        Array,
        jr.wrap_key_data(jnp.asarray(words, dtype=jnp.uint32), impl=impl),
    )


def reconstruct_hidden_learning_partner_evaluator_stream(
    binding: HiddenPlanningSeedBinding,
    *,
    config: HiddenLearningPartnerPlanningConfig,
    prng_impl: str = "threefry2x32",
) -> HiddenPlanningEvaluatorOwnedStream:
    """Precompute the action-independent world stream from one named-key binding."""

    if type(binding) is not HiddenPlanningSeedBinding:
        raise HiddenLearningPartnerPlanningRunnerError(
            "binding must be an exact HiddenPlanningSeedBinding"
        )
    if type(config) is not HiddenLearningPartnerPlanningConfig:
        raise HiddenLearningPartnerPlanningRunnerError(
            "config must be an exact HiddenLearningPartnerPlanningConfig"
        )
    if prng_impl != "threefry2x32":
        raise HiddenLearningPartnerPlanningRunnerError(
            "evaluator stream requires the canonical threefry2x32 implementation"
        )
    world = LearningPartnerWorld(LearningPartnerWorldConfig(config.phase_length))
    initial = world.init(
        LearningPartnerWorldKeys(
            cue=_binding_key(binding, "world.cue", impl=prng_impl),
            channel=_binding_key(binding, "world.channel", impl=prng_impl),
        )
    )

    def scan_step(
        state: LearningPartnerWorldState,
        _: None,
    ) -> tuple[LearningPartnerWorldState, tuple[Array, ...]]:
        shuffled = world.deliver(
            state,
            jnp.asarray(0, dtype=jnp.int32),
            SHUFFLED_CHANNEL,
        )
        transition, next_state = world.step_with_delivery(
            state,
            jnp.asarray(0, dtype=jnp.int32),
            shuffled,
            jnp.asarray(0, dtype=jnp.int32),
        )
        row = (
            transition.observation.helper_cue,
            transition.next_observation.helper_cue,
            transition.oracle.phase_index,
            transition.oracle.context,
            transition.oracle.target,
            shuffled,
            jr.key_data(state.cue_key),
            jr.key_data(next_state.cue_key),
            jr.key_data(state.channel_key),
            jr.key_data(next_state.channel_key),
        )
        return next_state, row

    final_object, rows = jax.lax.scan(
        scan_step,
        initial,
        xs=None,
        length=config.num_steps,
    )
    final = final_object
    (
        helper_cue,
        next_helper_cue,
        phase_index,
        context,
        target,
        shuffled_channel_output,
        cue_key_before,
        cue_key_after,
        channel_key_before,
        channel_key_after,
    ) = rows
    return HiddenPlanningEvaluatorOwnedStream(
        seed_index=binding.seed_index,
        seed=binding.seed,
        num_steps=config.num_steps,
        helper_cue=helper_cue,
        next_helper_cue=next_helper_cue,
        oracle_phase_index=phase_index,
        oracle_context=context,
        oracle_target=target,
        shuffled_channel_output=shuffled_channel_output,
        cue_key_before=cue_key_before,
        cue_key_after=cue_key_after,
        channel_key_before=channel_key_before,
        channel_key_after=channel_key_after,
        initial_cue_key_data=_key_data_tuple(initial.cue_key),
        initial_channel_key_data=_key_data_tuple(initial.channel_key),
        final_cue_key_data=_key_data_tuple(final.cue_key),
        final_channel_key_data=_key_data_tuple(final.channel_key),
    )


def _array_equal(left: object, right: object) -> bool:
    try:
        left_array = np.asarray(jax.device_get(left))
        right_array = np.asarray(jax.device_get(right))
    except (TypeError, ValueError):
        return False
    return (
        left_array.shape == right_array.shape
        and left_array.dtype == right_array.dtype
        and left_array.tobytes(order="C") == right_array.tobytes(order="C")
    )


def _exact_value_errors(
    expected: object,
    observed: object,
    *,
    path: str,
    compare_array_bits: bool = True,
) -> tuple[str, ...]:
    """Recursively compare exact dataclass/PyTree structure and concrete bits."""

    errors: list[str] = []
    if isinstance(expected, jax.Array):
        if not isinstance(observed, jax.Array) or isinstance(observed, jax.core.Tracer):
            return (f"{path} must be a concrete JAX array",)
        if expected.shape != observed.shape or expected.dtype != observed.dtype:
            return (f"{path} array shape or dtype differs",)
        expected_is_key = bool(jnp.issubdtype(expected.dtype, jax.dtypes.prng_key))
        observed_is_key = bool(jnp.issubdtype(observed.dtype, jax.dtypes.prng_key))
        if expected_is_key != observed_is_key:
            return (f"{path} typed-key status differs",)
        if expected_is_key:
            if str(jr.key_impl(expected)) != str(jr.key_impl(observed)):
                errors.append(f"{path} PRNG implementation differs")
            expected_array = jr.key_data(expected)
            observed_array = jr.key_data(observed)
        else:
            expected_array = expected
            observed_array = observed
        if compare_array_bits and not _array_equal(expected_array, observed_array):
            errors.append(f"{path} array bits differ")
        return tuple(errors)
    if dataclasses.is_dataclass(expected) and not isinstance(expected, type):
        if type(observed) is not type(expected):
            return (f"{path} dataclass concrete type differs",)
        for field in dataclasses.fields(expected):
            child_path = f"{path}.{field.name}" if path else field.name
            errors.extend(
                _exact_value_errors(
                    getattr(expected, field.name),
                    getattr(observed, field.name),
                    path=child_path,
                    compare_array_bits=compare_array_bits,
                )
            )
        return tuple(errors)
    if type(expected) is tuple:
        if type(observed) is not tuple:
            return (f"{path} must be an exact tuple",)
        expected_tuple = cast(tuple[object, ...], expected)
        observed_tuple = cast(tuple[object, ...], observed)
        if len(expected_tuple) != len(observed_tuple):
            return (f"{path} tuple length differs",)
        for index, (expected_item, observed_item) in enumerate(
            zip(expected_tuple, observed_tuple, strict=True)
        ):
            errors.extend(
                _exact_value_errors(
                    expected_item,
                    observed_item,
                    path=f"{path}[{index}]",
                    compare_array_bits=compare_array_bits,
                )
            )
        return tuple(errors)
    if type(observed) is not type(expected):
        return (f"{path} scalar concrete type differs",)
    if type(expected) is float:
        expected_bits = np.asarray(expected, dtype=np.float64).view(np.uint64)
        observed_bits = np.asarray(observed, dtype=np.float64).view(np.uint64)
        if bool(expected_bits != observed_bits):
            errors.append(f"{path} float bits differ")
    elif expected != observed:
        errors.append(f"{path} value differs")
    return tuple(errors)


def _tree_bit_equal(left: object, right: object) -> bool:
    return not _exact_value_errors(left, right, path="tree", compare_array_bits=True)


def _path_value(root: object, path: str) -> object:
    value = root
    for component in path.split("."):
        value = getattr(value, component)
    return value


def _plan_exact_child_clock_errors(
    run: HiddenLearningPartnerPlanningRun,
    arm: HiddenPlanningArm,
) -> tuple[str, ...]:
    """Independently bind one run's child words to its hash-bound plan arm."""

    if type(run) is not HiddenLearningPartnerPlanningRun:
        return ("run must be an exact HiddenLearningPartnerPlanningRun",)
    if type(arm) is not HiddenPlanningArm:
        return ("plan arm must be an exact HiddenPlanningArm",)
    errors: list[str] = []
    if run.condition != arm.condition:
        errors.append("run condition differs from the plan child-clock arm")
    clocks = arm.exact_child_clocks
    if (
        type(clocks) is not tuple
        or any(type(clock) is not HiddenPlanningExactChildClock for clock in clocks)
        or tuple(clock.name for clock in clocks) != ("behavior", "grounded")
    ):
        return (*errors, "plan arm exact child clocks are not the canonical ordered pair")
    for clock in clocks:
        if type(clock) is not HiddenPlanningExactChildClock:
            errors.append("plan arm child clock has the wrong concrete type")
            continue
        if clock.words_dtype != "uint32" or clock.words_shape != (2,):
            errors.append(f"plan arm {clock.name} exact word schema differs")
            continue
        for state_name, expected_words, expected_telemetry in (
            ("initial_state", clock.initial_words, clock.initial_telemetry),
            ("final_state", clock.final_words, clock.final_telemetry),
        ):
            state = getattr(run, state_name)
            words_path = f"run.{state_name}.{clock.words_state_path}"
            telemetry_path = f"run.{state_name}.{clock.telemetry_state_path}"
            try:
                raw_words = _path_value(state, clock.words_state_path)
                raw_telemetry = _path_value(state, clock.telemetry_state_path)
            except (AttributeError, TypeError) as exc:
                errors.append(f"{clock.name} child-clock state path failed closed: {exc}")
                continue
            if not isinstance(raw_words, jax.Array) or isinstance(raw_words, jax.core.Tracer):
                errors.append(f"{words_path} must be one concrete JAX array")
            else:
                words = np.asarray(jax.device_get(raw_words))
                if words.shape != clock.words_shape or words.dtype != np.uint32:
                    errors.append(f"{words_path} differs from plan exact child-clock schema")
                elif not np.array_equal(
                    words,
                    np.asarray(expected_words, dtype=np.uint32),
                ):
                    errors.append(f"{words_path} differs from plan exact child-clock words")
            if not isinstance(raw_telemetry, jax.Array) or isinstance(
                raw_telemetry,
                jax.core.Tracer,
            ):
                errors.append(f"{telemetry_path} must be one concrete JAX array")
            else:
                telemetry = np.asarray(jax.device_get(raw_telemetry))
                if telemetry.shape != () or telemetry.dtype != np.int32:
                    errors.append(f"{telemetry_path} differs from plan child-clock schema")
                elif int(telemetry) != expected_telemetry:
                    errors.append(f"{telemetry_path} differs from plan child-clock telemetry")
    return tuple(dict.fromkeys(errors))


def _nested_run_contract_errors(
    run: object,
) -> tuple[str, ...]:
    """Validate exact nested types and array schemas without trusting the run validator."""

    if type(run) is not HiddenLearningPartnerPlanningRun:
        return ("run must be an exact HiddenLearningPartnerPlanningRun",)
    checked = run
    errors: list[str] = []
    if type(checked.seed) is not int or checked.seed < 0:
        errors.append("run.seed must be a non-negative built-in integer")
    if type(checked.condition) is not str or checked.condition not in CANONICAL_CONDITION_ORDER:
        errors.append("run.condition must be one exact declared condition string")
    if type(checked.config) is not HiddenLearningPartnerPlanningConfig:
        return ("run.config has the wrong concrete type",)
    try:
        reconstructed_config = HiddenLearningPartnerPlanningConfig(
            **dataclasses.asdict(checked.config)
        )
    except (TypeError, ValueError) as exc:
        errors.append(f"run.config validation failed closed: {exc}")
    else:
        if checked.config != reconstructed_config:
            errors.append("run.config differs from exact reconstruction")
    if type(checked.initial_state) is not HiddenLearningPartnerPlanningState:
        errors.append("run.initial_state has the wrong concrete type")
    if type(checked.final_state) is not HiddenLearningPartnerPlanningState:
        errors.append("run.final_state has the wrong concrete type")
    if type(checked.trace) is not HiddenLearningPartnerPlanningTrace:
        errors.append("run.trace has the wrong concrete type")
    if type(checked.metrics) is not HiddenLearningPartnerPlanningMetrics:
        errors.append("run.metrics has the wrong concrete type")
    elif type(checked.metrics.phase_diagnostics) is not HiddenLearningPartnerPhaseDiagnostics:
        errors.append("run.metrics.phase_diagnostics has the wrong concrete type")
    if type(checked.resource) is not HiddenLearningPartnerPlanningResourceBudget:
        errors.append("run.resource has the wrong concrete type")
    if errors:
        return tuple(errors)
    resource_int_fields = tuple(
        field.name
        for field in dataclasses.fields(HiddenLearningPartnerPlanningResourceBudget)
        if field.name != "exact_tree_match"
    )
    for name in resource_int_fields:
        value = getattr(checked.resource, name)
        if type(value) is not int or value < 0:
            errors.append(f"run.resource.{name} must be a non-negative built-in integer")
    if type(checked.resource.exact_tree_match) is not bool:
        errors.append("run.resource.exact_tree_match must be a built-in bool")
    metrics = checked.metrics
    metric_int_fields = (
        "num_steps",
        "eligible_steps",
        "treated_eligible_steps",
        "control_eligible_steps",
    )
    metric_float_fields = (
        "mean_reward",
        "behavior_mean_nll",
        "behavior_mean_brier",
        "grounded_reward_mse",
        "grounded_next_observation_mse",
        "planner_consumption_rate",
        "action_change_rate",
        "randomized_effect",
        "potential_effect",
    )
    for name in metric_int_fields:
        if type(getattr(metrics, name)) is not int:
            errors.append(f"run.metrics.{name} must be a built-in integer")
    for name in metric_float_fields:
        if type(getattr(metrics, name)) is not float:
            errors.append(f"run.metrics.{name} must be a built-in float")
    for name in ("randomized_effect_valid", "potential_effect_valid"):
        if type(getattr(metrics, name)) is not bool:
            errors.append(f"run.metrics.{name} must be a built-in bool")
    phase = metrics.phase_diagnostics
    if type(phase.n_phases) is not int or type(phase.window_steps) is not int:
        errors.append("run.metrics.phase_diagnostics scalar counts must be built-in integers")
    else:
        phase_types: tuple[tuple[str, type[object]], ...] = (
            ("phase_index", int),
            ("hidden_context", int),
            ("phase_counts", int),
            ("phase_valid", bool),
            ("mean_reward", float),
            ("leading_reward", float),
            ("leading_counts", int),
            ("trailing_reward", float),
            ("trailing_counts", int),
            ("behavior_mean_nll", float),
            ("grounded_reward_mse", float),
            ("switch_cost", float),
            ("switch_cost_valid", bool),
            ("switch_cost_counts", int),
            ("recurrence_reference_phase", int),
            ("recurrence_savings", float),
            ("recurrence_savings_valid", bool),
            ("recurrence_counts", int),
        )
        for name, item_type in phase_types:
            value = getattr(phase, name)
            if (
                type(value) is not tuple
                or len(value) != phase.n_phases
                or any(type(item) is not item_type for item in value)
            ):
                errors.append(f"run.metrics.phase_diagnostics.{name} has the wrong tuple schema")
    if errors:
        return tuple(errors)
    try:
        bridge = bridge_module.HiddenLearningPartnerPlanningBridge(
            checked.config,
            checked.condition,
        )
        expected_initial = bridge.initialize(jr.key(checked.seed, impl="threefry2x32"))
        neutral_trace = cast(
            HiddenLearningPartnerPlanningTrace,
            getattr(bridge, "_neutral_trace")(expected_initial),
        )
    except (AttributeError, IndexError, TypeError, ValueError, RuntimeError) as exc:
        return (f"nested run schema reconstruction failed closed: {exc}",)
    errors.extend(
        _exact_value_errors(
            expected_initial,
            checked.initial_state,
            path="run.initial_state",
            compare_array_bits=False,
        )
    )
    errors.extend(
        _exact_value_errors(
            expected_initial,
            checked.final_state,
            path="run.final_state",
            compare_array_bits=False,
        )
    )
    num_steps = checked.config.num_steps
    for field in dataclasses.fields(neutral_trace):  # type: ignore[arg-type]
        expected_leaf = getattr(neutral_trace, field.name)
        observed_leaf = getattr(checked.trace, field.name)
        path = f"run.trace.{field.name}"
        if not isinstance(observed_leaf, jax.Array) or isinstance(
            observed_leaf,
            jax.core.Tracer,
        ):
            errors.append(f"{path} must be a concrete JAX array")
            continue
        expected_shape = (num_steps, *expected_leaf.shape)
        if observed_leaf.shape != expected_shape or observed_leaf.dtype != expected_leaf.dtype:
            errors.append(f"{path} shape or dtype differs from the exact trace schema")
    return tuple(errors)


def _validate_stream_contract(
    stream: HiddenPlanningEvaluatorOwnedStream,
    *,
    config: HiddenLearningPartnerPlanningConfig,
) -> tuple[str, ...]:
    errors: list[str] = []
    if type(stream.seed_index) is not int or stream.seed_index < 0:
        errors.append("stream.seed_index must be a non-negative built-in integer")
    if type(stream.seed) is not int or stream.seed < 0:
        errors.append("stream.seed must be a non-negative built-in integer")
    if type(stream.num_steps) is not int or stream.num_steps < 1:
        errors.append("stream.num_steps must be a positive built-in integer")
    scalar_fields = (
        "helper_cue",
        "next_helper_cue",
        "oracle_phase_index",
        "oracle_context",
        "oracle_target",
        "shuffled_channel_output",
    )
    key_fields = (
        "cue_key_before",
        "cue_key_after",
        "channel_key_before",
        "channel_key_after",
    )
    for name in scalar_fields:
        raw_value = getattr(stream, name)
        if not isinstance(raw_value, jax.Array) or isinstance(raw_value, jax.core.Tracer):
            errors.append(f"stream.{name} must be a concrete JAX array")
            continue
        value = np.asarray(jax.device_get(raw_value))
        if value.shape != (config.num_steps,) or value.dtype != np.int32:
            errors.append(f"stream.{name} must be exact int32[{config.num_steps}]")
    for name in key_fields:
        raw_value = getattr(stream, name)
        if not isinstance(raw_value, jax.Array) or isinstance(raw_value, jax.core.Tracer):
            errors.append(f"stream.{name} must be a concrete JAX array")
            continue
        value = np.asarray(jax.device_get(raw_value))
        if value.shape != (config.num_steps, 2) or value.dtype != np.uint32:
            errors.append(f"stream.{name} must be exact uint32[{config.num_steps},2]")
    for name in (
        "initial_cue_key_data",
        "initial_channel_key_data",
        "final_cue_key_data",
        "final_channel_key_data",
    ):
        words = getattr(stream, name)
        if (
            type(words) is not tuple
            or len(words) != 2
            or any(type(word) is not int for word in words)
        ):
            errors.append(f"stream.{name} must be an exact two-word integer tuple")
        elif any(word < 0 or word > np.iinfo(np.uint32).max for word in words):
            errors.append(f"stream.{name} words must be in the uint32 range")
    if stream.num_steps != config.num_steps:
        errors.append("stream.num_steps differs from the configured life")
    return tuple(errors)


def audit_hidden_learning_partner_planning_environment(
    run: HiddenLearningPartnerPlanningRun,
    stream: HiddenPlanningEvaluatorOwnedStream,
) -> tuple[str, ...]:
    """Bind one raw learner run to an evaluator-owned exogenous stream."""

    if type(run) is not HiddenLearningPartnerPlanningRun:
        return ("run must be an exact HiddenLearningPartnerPlanningRun",)
    if type(stream) is not HiddenPlanningEvaluatorOwnedStream:
        return ("stream must be an exact HiddenPlanningEvaluatorOwnedStream",)
    nested_errors = _nested_run_contract_errors(run)
    if nested_errors:
        return tuple(f"nested contract: {error}" for error in nested_errors)
    errors = list(_validate_stream_contract(stream, config=run.config))
    if run.seed != stream.seed:
        errors.append("run seed differs from evaluator stream seed")
    trace_bindings = (
        ("helper_cue", stream.helper_cue),
        ("next_helper_cue", stream.next_helper_cue),
        ("oracle_phase_index", stream.oracle_phase_index),
        ("oracle_context", stream.oracle_context),
        ("oracle_target", stream.oracle_target),
    )
    for name, expected in trace_bindings:
        if not _array_equal(getattr(run.trace, name), expected):
            errors.append(f"trace.{name} differs from the evaluator-owned stream")
    initial_world = run.initial_state.world
    final_world = run.final_state.world
    try:
        initial_cue_words = _key_data_tuple(initial_world.cue_key)
        initial_channel_words = _key_data_tuple(initial_world.channel_key)
        final_cue_words = _key_data_tuple(final_world.cue_key)
        final_channel_words = _key_data_tuple(final_world.channel_key)
    except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
        errors.append(f"world key reconstruction failed closed: {exc}")
    else:
        if initial_cue_words != stream.initial_cue_key_data:
            errors.append("initial world cue key differs from evaluator reconstruction")
        if initial_channel_words != stream.initial_channel_key_data:
            errors.append("initial world channel key differs from evaluator reconstruction")
        if final_cue_words != stream.final_cue_key_data:
            errors.append("final world cue key differs from evaluator reconstruction")
        if final_channel_words != stream.final_channel_key_data:
            errors.append("final world channel key differs from evaluator reconstruction")
        try:
            key_links = (
                (initial_world.cue_key, stream.cue_key_before[0], "initial cue"),
                (
                    initial_world.channel_key,
                    stream.channel_key_before[0],
                    "initial channel",
                ),
                (final_world.cue_key, stream.cue_key_after[-1], "final cue"),
                (
                    final_world.channel_key,
                    stream.channel_key_after[-1],
                    "final channel",
                ),
            )
        except (IndexError, TypeError) as exc:
            errors.append(f"world key stream indexing failed closed: {exc}")
        else:
            for key, expected, label in key_links:
                if not _array_equal(jr.key_data(key), expected):
                    errors.append(f"{label} key is not bound to the evaluator key stream")
    try:
        spec = condition_spec(run.condition)
    except (TypeError, ValueError) as exc:
        errors.append(f"run condition is invalid: {exc}")
        return tuple(dict.fromkeys(errors))
    if run.condition == SHUFFLED_DELIVERY:
        expected_delivery = stream.shuffled_channel_output
    elif run.condition == CONSTANT_ZERO_DELIVERY:
        expected_delivery = jnp.zeros((run.config.num_steps,), dtype=jnp.int32)
    elif run.condition == CONSTANT_ONE_DELIVERY:
        expected_delivery = jnp.ones((run.config.num_steps,), dtype=jnp.int32)
    else:
        expected_delivery = run.trace.helper_message
    if not _array_equal(run.trace.delivered_message, expected_delivery):
        errors.append("delivered-message trace differs from its declared channel intervention")
    if spec.channel == SHUFFLED_CHANNEL and not _array_equal(
        run.trace.delivered_message,
        stream.shuffled_channel_output,
    ):
        errors.append("shuffled delivery is not bound to the evaluator channel stream")
    return tuple(dict.fromkeys(errors))


def summarize_hidden_learning_partner_proposal_writes(
    run: HiddenLearningPartnerPlanningRun,
) -> HiddenPlanningProposalWriteAccounting:
    """Count raw update opportunities separately from markers and committed writes."""

    if type(run) is not HiddenLearningPartnerPlanningRun:
        raise HiddenLearningPartnerPlanningRunnerError(
            "run must be an exact HiddenLearningPartnerPlanningRun"
        )
    nested_errors = _nested_run_contract_errors(run)
    if nested_errors:
        raise HiddenLearningPartnerPlanningRunnerError(
            "run nested contract is invalid: " + "; ".join(nested_errors)
        )
    active = np.asarray(run.trace.active, dtype=np.bool_)
    accepted = np.asarray(run.trace.accepted, dtype=np.bool_)
    active_count = int(np.count_nonzero(active))
    accepted_count = int(np.count_nonzero(accepted))
    num_steps = run.config.num_steps
    return HiddenPlanningProposalWriteAccounting(
        num_steps=num_steps,
        active_transition_proposal_opportunities=active_count,
        accepted_transitions=accepted_count,
        rejected_active_transition_proposals=active_count - accepted_count,
        helper_update_proposal_opportunities=active_count,
        helper_committed_writes=int(np.count_nonzero(np.asarray(run.trace.helper_write))),
        beneficiary_update_proposal_opportunities=active_count,
        beneficiary_committed_writes=int(np.count_nonzero(np.asarray(run.trace.beneficiary_write))),
        behavior_update_proposal_opportunities=active_count,
        behavior_applied_proposal_markers=int(
            np.count_nonzero(np.asarray(run.trace.behavior_proposal_applied))
        ),
        behavior_committed_writes=int(
            np.count_nonzero(np.asarray(run.trace.behavior_committed_write))
        ),
        grounded_update_proposal_opportunities=active_count,
        grounded_applied_proposal_markers=int(
            np.count_nonzero(np.asarray(run.trace.grounded_proposal_applied))
        ),
        grounded_committed_writes=int(
            np.count_nonzero(np.asarray(run.trace.grounded_committed_write))
        ),
        planner_proposal_opportunities=active_count,
        planner_consumptions=int(np.count_nonzero(np.asarray(run.trace.planner_consumed))),
    )


def _accounting_errors(
    accounting: HiddenPlanningProposalWriteAccounting,
    *,
    condition: str,
    num_steps: int,
) -> tuple[str, ...]:
    errors: list[str] = []
    spec = condition_spec(condition)
    exact_full_counts = (
        "active_transition_proposal_opportunities",
        "accepted_transitions",
        "helper_update_proposal_opportunities",
        "beneficiary_update_proposal_opportunities",
        "behavior_update_proposal_opportunities",
        "behavior_applied_proposal_markers",
        "grounded_update_proposal_opportunities",
        "grounded_applied_proposal_markers",
        "planner_proposal_opportunities",
    )
    if accounting.num_steps != num_steps:
        errors.append("proposal accounting life length differs")
    for name in exact_full_counts:
        if getattr(accounting, name) != num_steps:
            errors.append(f"proposal accounting {name} differs from the full life")
    if accounting.rejected_active_transition_proposals != 0:
        errors.append("proposal accounting contains rejected transitions")
    expected_writes = {
        "helper_committed_writes": num_steps if spec.helper_write else 0,
        "beneficiary_committed_writes": num_steps if spec.beneficiary_write else 0,
        "behavior_committed_writes": num_steps if spec.behavior_write else 0,
        "grounded_committed_writes": num_steps if spec.grounded_write else 0,
    }
    for name, expected in expected_writes.items():
        if getattr(accounting, name) != expected:
            errors.append(f"proposal accounting {name} differs from the static write mask")
    if not 0 <= accounting.planner_consumptions <= num_steps:
        errors.append("proposal accounting planner consumption count is out of range")
    return tuple(errors)


def _plan_accounting_errors(
    plan: HiddenLearningPartnerPlanningScanPlan,
    accounting: HiddenPlanningProposalWriteAccounting,
    *,
    condition: str,
) -> tuple[str, ...]:
    """Bind observed proposal/write counts to the plan's named operations."""

    arms = {arm.condition: arm for arm in plan.arms}
    arm = arms.get(condition)
    if arm is None:
        return ("proposal accounting condition has no canonical plan arm",)
    operations = {
        operation.name: operation.per_run_total for operation in arm.named_operation_totals
    }
    field_to_operation = {
        "active_transition_proposal_opportunities": "bridge_step_calls",
        "accepted_transitions": "world_step_with_delivery_calls",
        "helper_update_proposal_opportunities": ("helper_value_update_proposal_opportunities"),
        "helper_committed_writes": ("helper_value_committed_writes_on_required_valid_trace"),
        "beneficiary_update_proposal_opportunities": (
            "beneficiary_value_update_proposal_opportunities"
        ),
        "beneficiary_committed_writes": (
            "beneficiary_value_committed_writes_on_required_valid_trace"
        ),
        "behavior_update_proposal_opportunities": ("behavior_update_proposal_opportunities"),
        "behavior_committed_writes": ("behavior_model_committed_writes_on_required_valid_trace"),
        "grounded_update_proposal_opportunities": ("grounded_update_proposal_opportunities"),
        "grounded_committed_writes": ("grounded_model_committed_writes_on_required_valid_trace"),
        "planner_proposal_opportunities": "planner_tie_random_draws",
    }
    errors: list[str] = []
    for field, operation in field_to_operation.items():
        expected = operations.get(operation)
        if expected is None:
            errors.append(f"plan is missing named operation {operation}")
        elif getattr(accounting, field) != expected:
            errors.append(f"proposal accounting {field} differs from plan operation {operation}")
    return tuple(errors)


def audit_hidden_learning_partner_planning_matched_records(
    *,
    config: HiddenLearningPartnerPlanningConfig,
    bindings: tuple[HiddenPlanningSeedBinding, ...],
    streams: tuple[HiddenPlanningEvaluatorOwnedStream, ...],
    records: tuple[HiddenPlanningMatchedRunRecord, ...],
    prng_impl: str = "threefry2x32",
) -> HiddenPlanningCommonRandomNumberAudit:
    """Audit an all-eleven-condition panel at any development life length.

    The canonical runner supplies all four plan bindings and the default
    configuration.  Accepting a smaller binding tuple here permits cheap
    mechanism tests of the same audit without executing the default campaign;
    such a subpanel has no suite, outcome, or evidence status.
    """

    if type(config) is not HiddenLearningPartnerPlanningConfig:
        raise HiddenLearningPartnerPlanningRunnerError(
            "audit config must be an exact HiddenLearningPartnerPlanningConfig"
        )
    if (
        type(bindings) is not tuple
        or not bindings
        or any(type(binding) is not HiddenPlanningSeedBinding for binding in bindings)
    ):
        raise HiddenLearningPartnerPlanningRunnerError(
            "audit bindings must be a nonempty exact tuple of seed bindings"
        )
    if any(
        type(binding.seed_index) is not int
        or type(binding.seed) is not int
        or binding.seed_index < 0
        or binding.seed < 0
        for binding in bindings
    ):
        raise HiddenLearningPartnerPlanningRunnerError(
            "audit binding identities must be non-negative built-in integers"
        )
    if len({binding.seed for binding in bindings}) != len(bindings):
        raise HiddenLearningPartnerPlanningRunnerError("audit bindings contain duplicate seeds")
    if tuple(binding.seed_index for binding in bindings) != tuple(range(len(bindings))):
        raise HiddenLearningPartnerPlanningRunnerError(
            "audit binding indices must be contiguous from zero"
        )
    if type(streams) is not tuple or any(
        type(stream) is not HiddenPlanningEvaluatorOwnedStream for stream in streams
    ):
        raise HiddenLearningPartnerPlanningRunnerError(
            "audit streams must be an exact tuple of evaluator-owned streams"
        )
    if type(records) is not tuple or any(
        type(record) is not HiddenPlanningMatchedRunRecord for record in records
    ):
        raise HiddenLearningPartnerPlanningRunnerError(
            "audit records must be an exact tuple of matched-run records"
        )
    if prng_impl != "threefry2x32":
        raise HiddenLearningPartnerPlanningRunnerError(
            "audit requires the threefry2x32 implementation"
        )
    stream_errors: list[str] = []
    environment_errors: list[str] = []
    initial_errors: list[str] = []
    trace_key_errors: list[str] = []
    final_key_errors: list[str] = []
    shuffled_errors: list[str] = []
    order_errors: list[str] = []
    expected_stream_identities = tuple((binding.seed_index, binding.seed) for binding in bindings)
    stream_contracts = tuple(_validate_stream_contract(stream, config=config) for stream in streams)
    for index, contract_errors in enumerate(stream_contracts):
        stream_errors.extend(f"stream[{index}].{error}" for error in contract_errors)
    stream_identities_are_safe = all(
        type(stream.seed_index) is int and type(stream.seed) is int for stream in streams
    )
    observed_stream_identities = (
        tuple((stream.seed_index, stream.seed) for stream in streams)
        if stream_identities_are_safe
        else ()
    )
    if len(streams) != len(bindings):
        stream_errors.append("evaluator stream count differs from seed-binding count")
    if not stream_identities_are_safe:
        stream_errors.append("evaluator stream identities have wrong concrete types")
    elif len(set(observed_stream_identities)) != len(observed_stream_identities):
        stream_errors.append("evaluator stream identities are not unique")
    if observed_stream_identities != expected_stream_identities:
        stream_errors.append("evaluator stream order/identities differ from bindings")
    for index, binding in enumerate(bindings):
        if index >= len(streams):
            break
        try:
            expected_stream = reconstruct_hidden_learning_partner_evaluator_stream(
                binding,
                config=config,
                prng_impl=prng_impl,
            )
        except (AttributeError, IndexError, OSError, TypeError, ValueError, RuntimeError) as exc:
            stream_errors.append(
                f"seed[{binding.seed_index}] stream reconstruction failed closed: {exc}"
            )
        else:
            if not _stream_equal(streams[index], expected_stream):
                stream_errors.append(
                    f"seed[{binding.seed_index}] supplied evaluator stream differs "
                    "from independent reconstruction"
                )
    streams_by_seed = {
        stream.seed: stream
        for stream, contract_errors in zip(streams, stream_contracts, strict=True)
        if not contract_errors
    }
    records_by_seed: dict[int, list[HiddenPlanningMatchedRunRecord]] = {}
    for index, record in enumerate(records):
        if (
            type(record.seed_index) is not int
            or type(record.seed) is not int
            or type(record.record_index) is not int
            or type(record.canonical_arm_index) is not int
            or type(record.condition) is not str
        ):
            environment_errors.append(f"record[{index}] identity fields have wrong concrete types")
            continue
        nested_errors = _nested_run_contract_errors(record.run)
        if nested_errors:
            environment_errors.extend(
                f"record[{index}] nested contract: {error}" for error in nested_errors
            )
            continue
        records_by_seed.setdefault(record.seed, []).append(record)

    expected_record_keys = tuple(
        (binding.seed, condition) for binding in bindings for condition in CANONICAL_CONDITION_ORDER
    )
    observed_record_keys = tuple((record.seed, record.condition) for record in records)
    if observed_record_keys != expected_record_keys:
        order_errors.append("records are not in canonical seed-major/condition-major order")

    for binding in bindings:
        stream = streams_by_seed.get(binding.seed)
        if stream is None:
            stream_errors.append(f"seed[{binding.seed_index}] evaluator stream is missing")
            continue
        seed_records = records_by_seed.get(binding.seed, [])
        if len(seed_records) != len(CANONICAL_CONDITION_ORDER):
            environment_errors.append(
                f"seed[{binding.seed_index}] does not contain all eleven condition records"
            )
            continue
        by_condition = {record.condition: record for record in seed_records}
        reference = by_condition.get(JOINT_ADAPTIVE)
        if reference is None:
            environment_errors.append(f"seed[{binding.seed_index}] has no joint-adaptive reference")
            continue
        for condition in CANONICAL_CONDITION_ORDER:
            condition_record = by_condition.get(condition)
            if condition_record is None:
                environment_errors.append(
                    f"seed[{binding.seed_index}] is missing condition {condition}"
                )
                continue
            prefix = f"seed[{binding.seed_index}].condition[{condition}]"
            per_run_environment = audit_hidden_learning_partner_planning_environment(
                condition_record.run,
                stream,
            )
            environment_errors.extend(f"{prefix}.{error}" for error in per_run_environment)
            if condition == SHUFFLED_DELIVERY and not _array_equal(
                condition_record.run.trace.delivered_message,
                stream.shuffled_channel_output,
            ):
                shuffled_errors.append(f"{prefix}.shuffled channel output binding mismatch")
            expected_initial = bridge_module.HiddenLearningPartnerPlanningBridge(
                config,
                cast(HiddenPlanningCondition, condition),
            ).initialize(jr.key(binding.seed, impl=prng_impl))
            if not _tree_bit_equal(condition_record.run.initial_state, expected_initial):
                initial_errors.append(f"{prefix}.initial state was not freshly reconstructed")
            for field in _STATE_SHARED_INITIAL_FIELDS:
                if not _tree_bit_equal(
                    getattr(condition_record.run.initial_state, field),
                    getattr(reference.run.initial_state, field),
                ):
                    initial_errors.append(f"{prefix}.initial shared field differs: {field}")
            for field in _TRACE_NAMED_KEY_FIELDS:
                if not _array_equal(
                    getattr(condition_record.run.trace, field),
                    getattr(reference.run.trace, field),
                ):
                    trace_key_errors.append(f"{prefix}.cross-arm trace key differs: {field}")
            for path in _FINAL_NAMED_KEY_PATHS:
                if not _tree_bit_equal(
                    _path_value(condition_record.run.final_state, path),
                    _path_value(reference.run.final_state, path),
                ):
                    final_key_errors.append(f"{prefix}.final named key differs: {path}")

    errors = tuple(
        dict.fromkeys(
            (
                *stream_errors,
                *environment_errors,
                *initial_errors,
                *trace_key_errors,
                *final_key_errors,
                *shuffled_errors,
                *order_errors,
            )
        )
    )
    return HiddenPlanningCommonRandomNumberAudit(
        paired_seed_count=len(bindings),
        arm_count=len(CANONICAL_CONDITION_ORDER),
        record_count=len(records),
        evaluator_stream_reconstruction_passed=not stream_errors,
        action_independent_environment_parity_passed=not environment_errors,
        shared_initial_state_parity_passed=not initial_errors,
        cross_arm_trace_key_parity_passed=not trace_key_errors,
        final_named_key_parity_passed=not final_key_errors,
        shuffled_channel_output_binding_passed=not shuffled_errors,
        canonical_record_order_passed=not order_errors,
        errors=errors,
    )


def _stream_equal(
    left: HiddenPlanningEvaluatorOwnedStream,
    right: HiddenPlanningEvaluatorOwnedStream,
) -> bool:
    if (
        type(left) is not HiddenPlanningEvaluatorOwnedStream
        or type(right) is not HiddenPlanningEvaluatorOwnedStream
    ):
        return False
    scalar_names = (
        "seed_index",
        "seed",
        "num_steps",
        "initial_cue_key_data",
        "initial_channel_key_data",
        "final_cue_key_data",
        "final_channel_key_data",
    )
    array_names = (
        "helper_cue",
        "next_helper_cue",
        "oracle_phase_index",
        "oracle_context",
        "oracle_target",
        "shuffled_channel_output",
        "cue_key_before",
        "cue_key_after",
        "channel_key_before",
        "channel_key_after",
    )
    for name in scalar_names:
        left_value = getattr(left, name)
        right_value = getattr(right, name)
        if type(left_value) is not type(right_value) or left_value != right_value:
            return False
    return all(_array_equal(getattr(left, name), getattr(right, name)) for name in array_names)


def validate_hidden_learning_partner_planning_matched_suite_structural_unauthenticated(
    suite: object,
) -> tuple[str, ...]:
    """Validate structure/current source only; this does not replay runs."""

    if type(suite) is not HiddenLearningPartnerPlanningMatchedSuite:
        return ("suite must be an exact HiddenLearningPartnerPlanningMatchedSuite",)
    checked = suite
    try:
        plan = _canonical_plan(plan_module.build_hidden_learning_partner_planning_scan_plan())
    except HiddenLearningPartnerPlanningRunnerError as exc:
        return (f"canonical plan reconstruction failed: {exc}",)
    errors: list[str] = []
    if checked.schema != HIDDEN_LEARNING_PARTNER_PLANNING_RUNNER_SCHEMA:
        errors.append("suite schema differs from the development runner schema")
    if checked.status != HIDDEN_LEARNING_PARTNER_PLANNING_RUNNER_STATUS:
        errors.append("suite status differs from the raw development status")
    if checked.assessment_status != ASSESSMENT_STATUS:
        errors.append("suite assessment status must remain not_assessed")
    if checked.development_only is not True or checked.seed_role != DEVELOPMENT_SEED_ROLE:
        errors.append("suite seed role is not consumed nonpromoting development")
    if checked.consumed_development_seeds != PAIRED_DEVELOPMENT_SEEDS:
        errors.append("suite does not bind the exact four consumed development seeds")
    if checked.held_out_seeds_used is not False:
        errors.append("suite falsely labels development roots as held out")
    if any(
        value is not False
        for value in (
            checked.evidence_authorized,
            checked.scientific_promotion_allowed,
            checked.artifact_writes_authorized,
        )
    ):
        errors.append("suite carries artifact, evidence, or promotion authority")
    if checked.execution_acknowledgement != DEVELOPMENT_EXECUTION_ACKNOWLEDGEMENT:
        errors.append("suite execution acknowledgement differs")
    if checked.source_plan_sha256 != plan.plan_sha256:
        errors.append("suite source plan digest differs from the canonical plan")
    if type(checked.source_runtime_manifest) is not HiddenPlanningSourceRuntimeManifest:
        errors.append("suite source/runtime manifest has the wrong concrete type")
    else:
        try:
            expected_manifest = build_hidden_learning_partner_source_runtime_manifest(
                execution_mode=checked.source_runtime_manifest.execution_mode
            )
        except (AttributeError, OSError, TypeError, ValueError, RuntimeError) as exc:
            errors.append(f"suite source/runtime reconstruction failed closed: {exc}")
        else:
            if checked.source_runtime_manifest != expected_manifest:
                errors.append("suite source/runtime manifest differs from current full bytes")
    if not _is_sha256_hex(checked.execution_request_sha256):
        errors.append("suite execution request digest is not one exact SHA-256")
    if not _is_sha256_hex(checked.execution_permit_hmac_sha256):
        errors.append("suite execution permit HMAC is not one exact SHA-256")
    if not _is_sha256_hex(checked.suite_binding_sha256):
        errors.append("suite binding digest is not one exact SHA-256")
    if checked.authenticated_replay_verified is not False:
        errors.append("raw suite must not self-assert authenticated replay")
    if checked.canonical_condition_order != CANONICAL_CONDITION_ORDER:
        errors.append("suite condition order differs from all eleven declared conditions")
    if checked.canonical_record_order is not True or checked.raw_records_present is not True:
        errors.append("suite does not declare canonical raw-record ordering")
    if checked.aggregate_statistics is not None or checked.thresholds is not None:
        errors.append("aggregate statistics and thresholds are forbidden")
    if checked.artifact_output_path is not None:
        errors.append("artifact output paths are forbidden")
    if errors:
        return tuple(dict.fromkeys(errors))

    records_are_exact_tuple = type(checked.records) is tuple
    if not records_are_exact_tuple or len(checked.records) != plan.counts.planned_run_count:
        return ("suite records do not contain exactly 44 fixed-shape runs",)
    wrong_record_indices = tuple(
        index
        for index, record in enumerate(checked.records)
        if type(record) is not HiddenPlanningMatchedRunRecord
    )
    if wrong_record_indices:
        return tuple(
            f"suite record {index} has the wrong concrete type" for index in wrong_record_indices
        )
    if type(checked.evaluator_streams) is not tuple or len(checked.evaluator_streams) != len(
        plan.seed_contract.bindings
    ):
        return ("suite evaluator streams do not cover exactly four paired seeds",)

    expected_streams = tuple(
        reconstruct_hidden_learning_partner_evaluator_stream(
            binding,
            config=plan.config,
            prng_impl=plan.seed_contract.prng_impl,
        )
        for binding in plan.seed_contract.bindings
    )
    for index, (observed, expected) in enumerate(
        zip(checked.evaluator_streams, expected_streams, strict=True)
    ):
        if type(observed) is not HiddenPlanningEvaluatorOwnedStream or not _stream_equal(
            observed,
            expected,
        ):
            errors.append(f"suite evaluator stream differs at seed index {index}")

    expected_keys = canonical_hidden_learning_partner_planning_record_keys(plan)
    for index, (record, expected_key) in enumerate(
        zip(checked.records, expected_keys, strict=True)
    ):
        scalar_type_errors = tuple(
            name
            for name, value, expected_type in (
                ("record_index", record.record_index, int),
                ("seed_index", record.seed_index, int),
                ("seed", record.seed, int),
                ("seed_role", record.seed_role, str),
                ("canonical_arm_index", record.canonical_arm_index, int),
                ("condition", record.condition, str),
                ("assessment_status", record.assessment_status, str),
            )
            if type(value) is not expected_type
        )
        if scalar_type_errors:
            errors.append(
                f"suite record {index} scalar fields have wrong concrete types: "
                + ", ".join(scalar_type_errors)
            )
            continue
        expected_seed, expected_condition = expected_key
        expected_seed_index = index // len(CANONICAL_CONDITION_ORDER)
        expected_arm_index = index % len(CANONICAL_CONDITION_ORDER)
        if (
            record.record_index != index
            or record.seed_index != expected_seed_index
            or record.seed != expected_seed
            or record.canonical_arm_index != expected_arm_index
            or record.condition != expected_condition
        ):
            errors.append(f"suite record {index} identity or canonical position differs")
        if record.seed_role != DEVELOPMENT_SEED_ROLE:
            errors.append(f"suite record {index} seed role differs")
        if record.assessment_status != ASSESSMENT_STATUS:
            errors.append(f"suite record {index} is not not_assessed")
        nested_record_errors: list[str] = []
        if type(record.phase_diagnostics) is not HiddenLearningPartnerPhaseDiagnostics:
            nested_record_errors.append("phase_diagnostics has the wrong concrete type")
        if type(record.proposal_write_accounting) is not HiddenPlanningProposalWriteAccounting:
            nested_record_errors.append("proposal_write_accounting has the wrong concrete type")
        if type(record.strict_run_validation_errors) is not tuple or any(
            type(error) is not str for error in record.strict_run_validation_errors
        ):
            nested_record_errors.append(
                "strict_run_validation_errors must be an exact string tuple"
            )
        if type(record.environment_stream_errors) is not tuple or any(
            type(error) is not str for error in record.environment_stream_errors
        ):
            nested_record_errors.append("environment_stream_errors must be an exact string tuple")
        if nested_record_errors:
            errors.extend(
                f"suite record {index} nested record: {error}" for error in nested_record_errors
            )
            continue
        if type(record.run) is not HiddenLearningPartnerPlanningRun:
            errors.append(f"suite record {index} run has the wrong concrete type")
            continue
        nested_errors = _nested_run_contract_errors(record.run)
        if nested_errors:
            errors.extend(
                f"suite record {index} nested contract: {error}" for error in nested_errors
            )
            continue
        if (
            record.run.seed != expected_seed
            or record.run.condition != expected_condition
            or record.run.config != plan.config
            or record.run.resource != plan.resource_budget
        ):
            errors.append(f"suite record {index} run binding differs from the plan")
        child_clock_errors = _plan_exact_child_clock_errors(
            record.run,
            plan.arms[expected_arm_index],
        )
        errors.extend(
            f"suite record {index} plan child clock: {error}"
            for error in child_clock_errors
        )
        try:
            strict_errors = tuple(
                bridge_module.validate_hidden_learning_partner_planning_run(record.run)
            )
        except (AttributeError, IndexError, TypeError, ValueError, RuntimeError) as exc:
            strict_errors = (f"strict per-run validator failed closed: {exc}",)
        if record.strict_run_validation_errors != strict_errors:
            errors.append(f"suite record {index} cached strict validation errors differ")
        if strict_errors:
            errors.append(f"suite record {index} failed strict per-run validation")
        expected_accounting = summarize_hidden_learning_partner_proposal_writes(record.run)
        if record.proposal_write_accounting != expected_accounting:
            errors.append(f"suite record {index} proposal/write accounting differs")
            continue
        for error in _accounting_errors(
            record.proposal_write_accounting,
            condition=record.condition,
            num_steps=plan.life_steps,
        ):
            errors.append(f"suite record {index}: {error}")
        for error in _plan_accounting_errors(
            plan,
            record.proposal_write_accounting,
            condition=record.condition,
        ):
            errors.append(f"suite record {index}: {error}")
        expected_stream = expected_streams[expected_seed_index]
        environment_errors = audit_hidden_learning_partner_planning_environment(
            record.run,
            expected_stream,
        )
        if record.environment_stream_errors != environment_errors:
            errors.append(f"suite record {index} cached environment errors differ")
        if environment_errors:
            errors.append(f"suite record {index} failed evaluator-stream binding")
        if record.phase_diagnostics != record.run.metrics.phase_diagnostics:
            errors.append(f"suite record {index} raw phase diagnostics differ")

    if errors:
        return tuple(dict.fromkeys(errors))
    try:
        expected_suite_binding = _suite_binding_sha256(checked)
    except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
        errors.append(f"suite binding reconstruction failed closed: {exc}")
    else:
        if checked.suite_binding_sha256 != expected_suite_binding:
            errors.append("suite binding digest differs from the complete raw suite")
    if errors:
        return tuple(dict.fromkeys(errors))
    expected_audit = audit_hidden_learning_partner_planning_matched_records(
        config=plan.config,
        bindings=plan.seed_contract.bindings,
        streams=expected_streams,
        records=checked.records,
        prng_impl=plan.seed_contract.prng_impl,
    )
    if checked.common_random_number_audit != expected_audit:
        errors.append("suite common-random-number audit differs from reconstruction")
    if expected_audit.errors:
        errors.append("suite common-random-number audit failed")
    return tuple(dict.fromkeys(errors))


def validate_hidden_learning_partner_planning_matched_suite(
    suite: object,
) -> tuple[str, ...]:
    """Compatibility alias for the explicitly unauthenticated structural validator."""

    return validate_hidden_learning_partner_planning_matched_suite_structural_unauthenticated(suite)


def _authenticate_records_by_exact_replay(
    *,
    config: HiddenLearningPartnerPlanningConfig,
    bindings: tuple[HiddenPlanningSeedBinding, ...],
    records: tuple[HiddenPlanningMatchedRunRecord, ...],
    expected_manifest: HiddenPlanningSourceRuntimeManifest,
    allow_canonical_panel: bool,
) -> HiddenPlanningAuthenticatedReplayValidation:
    errors: list[str] = []
    if type(expected_manifest) is not HiddenPlanningSourceRuntimeManifest:
        errors.append("expected manifest has the wrong concrete type")
        mode: HiddenPlanningExecutionMode = "eager"
    else:
        mode = expected_manifest.execution_mode
    try:
        current_manifest = build_hidden_learning_partner_source_runtime_manifest(
            execution_mode=mode
        )
    except (AttributeError, OSError, TypeError, ValueError, RuntimeError) as exc:
        current_manifest = None
        errors.append(f"source/runtime reconstruction failed closed: {exc}")
    if type(config) is not HiddenLearningPartnerPlanningConfig:
        errors.append("replay config has the wrong concrete type")
    if type(bindings) is not tuple or any(
        type(binding) is not HiddenPlanningSeedBinding for binding in bindings
    ):
        errors.append("replay bindings have the wrong nested type")
    if type(records) is not tuple or any(
        type(record) is not HiddenPlanningMatchedRunRecord for record in records
    ):
        errors.append("replay records have the wrong nested type")
    if type(expected_manifest) is HiddenPlanningSourceRuntimeManifest and (
        expected_manifest != current_manifest
    ):
        errors.append("expected manifest differs from current full source/runtime bytes")
    safe_bindings = (
        bindings
        if type(bindings) is tuple
        and all(type(binding) is HiddenPlanningSeedBinding for binding in bindings)
        else ()
    )
    safe_records = (
        records
        if type(records) is tuple
        and all(type(record) is HiddenPlanningMatchedRunRecord for record in records)
        else ()
    )
    if not safe_bindings:
        errors.append("replay requires a nonempty explicitly supplied seed panel")
    if not safe_records:
        errors.append("replay requires a nonempty explicitly supplied record panel")
    if safe_bindings and len(safe_records) != (len(safe_bindings) * len(CANONICAL_CONDITION_ORDER)):
        errors.append("replay panel must contain all eleven conditions for every seed")
    canonical_panel = type(config) is HiddenLearningPartnerPlanningConfig and (
        config == HiddenLearningPartnerPlanningConfig()
        and tuple(binding.seed for binding in safe_bindings) == PAIRED_DEVELOPMENT_SEEDS
        and len(safe_records) == len(PAIRED_DEVELOPMENT_SEEDS) * len(CANONICAL_CONDITION_ORDER)
    )
    if canonical_panel and not allow_canonical_panel:
        errors.append("canonical 44-run replay requires a live execution permit")
    try:
        canonical_records = canonicalize_hidden_learning_partner_planning_records(
            safe_records,
            seed_order=tuple(binding.seed for binding in safe_bindings),
        )
    except (AttributeError, IndexError, TypeError, ValueError, RuntimeError) as exc:
        canonical_records = ()
        errors.append(f"record canonicalization failed closed: {exc}")
    if errors:
        return HiddenPlanningAuthenticatedReplayValidation(
            schema=_AUTHENTICATED_REPLAY_SCHEMA,
            assessment_status=ASSESSMENT_STATUS,
            development_only=True,
            evidence_authorized=False,
            scientific_promotion_allowed=False,
            authenticated_replay_verified=False,
            rerun_count=0,
            source_runtime_manifest=current_manifest,
            errors=tuple(dict.fromkeys(errors)),
        )
    rerun_count = 0
    for index, record in enumerate(canonical_records):
        nested_errors = _nested_run_contract_errors(record.run)
        if nested_errors:
            errors.extend(f"record[{index}] nested contract: {error}" for error in nested_errors)
            continue
        try:
            rerun = bridge_module.run_hidden_learning_partner_planning(
                cast(HiddenPlanningCondition, record.condition),
                seed=record.seed,
                config=config,
                jit_compile=mode == "jit",
            )
        except (AttributeError, IndexError, TypeError, ValueError, RuntimeError) as exc:
            errors.append(f"record[{index}] replay execution failed closed: {exc}")
            continue
        rerun_count += 1
        replay_errors = _exact_value_errors(
            record.run,
            rerun,
            path=f"record[{index}].run",
            compare_array_bits=True,
        )
        errors.extend(replay_errors)
    verified = not errors and rerun_count == len(canonical_records)
    return HiddenPlanningAuthenticatedReplayValidation(
        schema=_AUTHENTICATED_REPLAY_SCHEMA,
        assessment_status=ASSESSMENT_STATUS,
        development_only=True,
        evidence_authorized=False,
        scientific_promotion_allowed=False,
        authenticated_replay_verified=verified,
        rerun_count=rerun_count,
        source_runtime_manifest=current_manifest,
        errors=tuple(dict.fromkeys(errors)),
    )


def authenticate_hidden_learning_partner_planning_development_subpanel(
    *,
    config: HiddenLearningPartnerPlanningConfig,
    bindings: tuple[HiddenPlanningSeedBinding, ...],
    records: tuple[HiddenPlanningMatchedRunRecord, ...],
    expected_manifest: HiddenPlanningSourceRuntimeManifest,
) -> HiddenPlanningAuthenticatedReplayValidation:
    """Replay a noncanonical tiny development panel for mechanism verification."""

    return _authenticate_records_by_exact_replay(
        config=config,
        bindings=bindings,
        records=records,
        expected_manifest=expected_manifest,
        allow_canonical_panel=False,
    )


def authenticate_hidden_learning_partner_planning_matched_suite(
    suite: object,
    *,
    plan: object,
    request: object,
    permit: object,
) -> HiddenPlanningAuthenticatedReplayValidation:
    """Live-permit-gated exact replay of every canonical seed/condition root."""

    if type(suite) is not HiddenLearningPartnerPlanningMatchedSuite:
        manifest = build_hidden_learning_partner_source_runtime_manifest(execution_mode="eager")
        return HiddenPlanningAuthenticatedReplayValidation(
            schema=_AUTHENTICATED_REPLAY_SCHEMA,
            assessment_status=ASSESSMENT_STATUS,
            development_only=True,
            evidence_authorized=False,
            scientific_promotion_allowed=False,
            authenticated_replay_verified=False,
            rerun_count=0,
            source_runtime_manifest=manifest,
            errors=("suite must be an exact HiddenLearningPartnerPlanningMatchedSuite",),
        )
    checked_plan = _canonical_plan(plan)
    checked_request, checked_permit = _require_live_execution_permit(
        request,
        permit,
        plan=checked_plan,
        purpose="replay",
    )
    errors = list(
        validate_hidden_learning_partner_planning_matched_suite_structural_unauthenticated(suite)
    )
    if suite.execution_request_sha256 != checked_request.request_sha256:
        errors.append("suite is bound to a different execution request")
    if suite.execution_permit_hmac_sha256 != checked_permit.permit_hmac_sha256:
        errors.append("suite is bound to a different execution permit")
    if suite.source_runtime_manifest != checked_request.source_runtime_manifest:
        errors.append("suite is bound to a different source/runtime manifest")
    try:
        suite_binding = _suite_binding_sha256(suite)
    except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
        suite_binding = ""
        errors.append(f"suite binding reconstruction failed closed: {exc}")
    if suite.suite_binding_sha256 != suite_binding:
        errors.append("suite is not bound to its complete raw contents")
    if errors:
        return HiddenPlanningAuthenticatedReplayValidation(
            schema=_AUTHENTICATED_REPLAY_SCHEMA,
            assessment_status=ASSESSMENT_STATUS,
            development_only=True,
            evidence_authorized=False,
            scientific_promotion_allowed=False,
            authenticated_replay_verified=False,
            rerun_count=0,
            source_runtime_manifest=checked_request.source_runtime_manifest,
            errors=tuple(dict.fromkeys(errors)),
        )
    _consume_permit_for_replay(
        checked_permit,
        suite_binding_sha256=suite_binding,
    )
    return _authenticate_records_by_exact_replay(
        config=checked_plan.config,
        bindings=checked_plan.seed_contract.bindings,
        records=suite.records,
        expected_manifest=checked_request.source_runtime_manifest,
        allow_canonical_panel=True,
    )


def run_hidden_learning_partner_planning_matched_suite(
    plan: object,
    *,
    request: object,
    permit: object,
    arm_order: object = CANONICAL_CONDITION_ORDER,
) -> HiddenLearningPartnerPlanningMatchedSuite:
    """Execute all 44 canonical development lives and return raw records in memory.

    Calling this function consumes the plan's four development roots.  The
    exact request and short-lived permit have no defaults.  They grant no
    artifact, evidence, threshold, or promotion authority.  Permit validation,
    source reconstruction, and a fresh live host check happen before evaluator
    stream reconstruction or any one-life call.
    """

    checked = _canonical_plan(plan)
    checked_request, checked_permit = _require_live_execution_permit(
        request,
        permit,
        plan=checked,
        purpose="run",
    )
    jit_compile = checked_request.execution_mode == "jit"
    schedule = build_hidden_learning_partner_planning_run_schedule(
        checked,
        arm_order=arm_order,
    )
    _consume_permit_for_run(checked_permit)
    streams = tuple(
        reconstruct_hidden_learning_partner_evaluator_stream(
            binding,
            config=checked.config,
            prng_impl=checked.seed_contract.prng_impl,
        )
        for binding in checked.seed_contract.bindings
    )
    streams_by_seed = {stream.seed: stream for stream in streams}
    execution_records: list[HiddenPlanningMatchedRunRecord] = []
    for run_request in schedule:
        condition = cast(HiddenPlanningCondition, run_request.condition)
        run = bridge_module.run_hidden_learning_partner_planning(
            condition,
            seed=run_request.seed,
            config=checked.config,
            jit_compile=jit_compile,
        )
        child_clock_errors = _plan_exact_child_clock_errors(
            run,
            checked.arms[run_request.canonical_arm_index],
        )
        if child_clock_errors:
            raise HiddenLearningPartnerPlanningRunnerError(
                f"seed {run_request.seed} condition {run_request.condition} failed plan child "
                f"clock binding: {'; '.join(child_clock_errors)}"
            )
        strict_errors = tuple(bridge_module.validate_hidden_learning_partner_planning_run(run))
        if strict_errors:
            raise HiddenLearningPartnerPlanningRunnerError(
                f"seed {run_request.seed} condition {run_request.condition} failed strict run "
                f"validation: {'; '.join(strict_errors)}"
            )
        accounting = summarize_hidden_learning_partner_proposal_writes(run)
        accounting_errors = _accounting_errors(
            accounting,
            condition=run_request.condition,
            num_steps=checked.life_steps,
        )
        accounting_errors = (
            *accounting_errors,
            *_plan_accounting_errors(
                checked,
                accounting,
                condition=run_request.condition,
            ),
        )
        if accounting_errors:
            raise HiddenLearningPartnerPlanningRunnerError(
                f"seed {run_request.seed} condition {run_request.condition} failed proposal/write "
                f"accounting: {'; '.join(accounting_errors)}"
            )
        environment_errors = audit_hidden_learning_partner_planning_environment(
            run,
            streams_by_seed[run_request.seed],
        )
        if environment_errors:
            raise HiddenLearningPartnerPlanningRunnerError(
                f"seed {run_request.seed} condition {run_request.condition} failed "
                "evaluator-stream "
                f"binding: {'; '.join(environment_errors)}"
            )
        record_index = (
            run_request.seed_index * len(CANONICAL_CONDITION_ORDER)
            + run_request.canonical_arm_index
        )
        execution_records.append(
            HiddenPlanningMatchedRunRecord(
                record_index=record_index,
                seed_index=run_request.seed_index,
                seed=run_request.seed,
                seed_role=DEVELOPMENT_SEED_ROLE,
                canonical_arm_index=run_request.canonical_arm_index,
                condition=run_request.condition,
                assessment_status=ASSESSMENT_STATUS,
                run=run,
                phase_diagnostics=run.metrics.phase_diagnostics,
                proposal_write_accounting=accounting,
                strict_run_validation_errors=strict_errors,
                environment_stream_errors=environment_errors,
            )
        )
    records = canonicalize_hidden_learning_partner_planning_records(
        tuple(execution_records),
        seed_order=tuple(binding.seed for binding in checked.seed_contract.bindings),
    )
    audit = audit_hidden_learning_partner_planning_matched_records(
        config=checked.config,
        bindings=checked.seed_contract.bindings,
        streams=streams,
        records=records,
        prng_impl=checked.seed_contract.prng_impl,
    )
    if audit.errors:
        raise HiddenLearningPartnerPlanningRunnerError(
            "cross-arm common-random-number audit failed: " + "; ".join(audit.errors)
        )
    suite = HiddenLearningPartnerPlanningMatchedSuite(
        schema=HIDDEN_LEARNING_PARTNER_PLANNING_RUNNER_SCHEMA,
        status=HIDDEN_LEARNING_PARTNER_PLANNING_RUNNER_STATUS,
        assessment_status=ASSESSMENT_STATUS,
        development_only=True,
        seed_role=DEVELOPMENT_SEED_ROLE,
        consumed_development_seeds=PAIRED_DEVELOPMENT_SEEDS,
        held_out_seeds_used=False,
        evidence_authorized=False,
        scientific_promotion_allowed=False,
        artifact_writes_authorized=False,
        execution_acknowledgement=checked_request.consumption_acknowledgement,
        source_plan_sha256=checked.plan_sha256,
        source_runtime_manifest=checked_request.source_runtime_manifest,
        execution_request_sha256=checked_request.request_sha256,
        execution_permit_hmac_sha256=checked_permit.permit_hmac_sha256,
        suite_binding_sha256="",
        authenticated_replay_verified=False,
        canonical_condition_order=CANONICAL_CONDITION_ORDER,
        canonical_record_order=True,
        raw_records_present=True,
        evaluator_streams=streams,
        records=records,
        common_random_number_audit=audit,
        aggregate_statistics=None,
        thresholds=None,
        artifact_output_path=None,
    )
    suite = dataclasses.replace(
        suite,
        suite_binding_sha256=_suite_binding_sha256(suite),
    )
    suite_errors = (
        validate_hidden_learning_partner_planning_matched_suite_structural_unauthenticated(suite)
    )
    if suite_errors:
        raise HiddenLearningPartnerPlanningRunnerError(
            "completed suite failed canonical reconstruction: " + "; ".join(suite_errors)
        )
    _bind_completed_suite_to_permit(
        checked_permit,
        suite_binding_sha256=suite.suite_binding_sha256,
    )
    return suite


__all__ = [
    "ARTIFACT_WRITES_AUTHORIZED",
    "ASSESSMENT_STATUS",
    "DEVELOPMENT_EXECUTION_ACKNOWLEDGEMENT",
    "DEVELOPMENT_ONLY",
    "DEVELOPMENT_SEED_ROLE",
    "EVIDENCE_AUTHORIZED",
    "HIDDEN_LEARNING_PARTNER_PLANNING_RUNNER_SCHEMA",
    "HIDDEN_LEARNING_PARTNER_PLANNING_RUNNER_STATUS",
    "HiddenLearningPartnerPlanningMatchedSuite",
    "HiddenLearningPartnerPlanningRunnerError",
    "HiddenPlanningAuthenticatedReplayValidation",
    "HiddenPlanningCommonRandomNumberAudit",
    "HiddenPlanningEvaluatorOwnedStream",
    "HiddenPlanningExecutionPermit",
    "HiddenPlanningExecutionRequest",
    "HiddenPlanningHostQuiescenceSnapshot",
    "HiddenPlanningMatchedRunRecord",
    "HiddenPlanningProposalWriteAccounting",
    "HiddenPlanningRunRequest",
    "HiddenPlanningSourceFileHash",
    "HiddenPlanningSourceRuntimeManifest",
    "SCIENTIFIC_PROMOTION_ALLOWED",
    "audit_hidden_learning_partner_planning_environment",
    "audit_hidden_learning_partner_planning_matched_records",
    "authenticate_hidden_learning_partner_planning_development_subpanel",
    "authenticate_hidden_learning_partner_planning_matched_suite",
    "build_hidden_learning_partner_execution_request",
    "build_hidden_learning_partner_planning_run_schedule",
    "build_hidden_learning_partner_source_runtime_manifest",
    "canonical_hidden_learning_partner_planning_record_keys",
    "canonicalize_hidden_learning_partner_planning_records",
    "reconstruct_hidden_learning_partner_evaluator_stream",
    "issue_hidden_learning_partner_execution_permit",
    "run_hidden_learning_partner_planning_matched_suite",
    "summarize_hidden_learning_partner_proposal_writes",
    "validate_hidden_learning_partner_execution_request",
    "validate_hidden_learning_partner_planning_matched_suite",
    "validate_hidden_learning_partner_planning_matched_suite_structural_unauthenticated",
]
