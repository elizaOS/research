"""Fail-closed execution and aggregation for hidden-regime calibration.

This module consumes the frozen, development-only factorial protocol.  It does
not execute anything on import and it does not contain a protected-evaluation
path.  A case can run only through the separately authorized, content-addressed
readiness ZIP.  The 16,528-row primitive trace is audited while ephemeral; the
persisted shard contains exact hexadecimal outcome values, complete compact
metric sources, and audit digests, but never the raw trace.

Calibration shards and aggregates are nonpromoting development records.  Once
an immutable aggregate has been exactly recomputed, this module can ask the
same certified ZIP to run the separately pure threshold engine and publish its
one development-only freeze-or-valid-rejection receipt.  No path here accepts
a claim or supports scientific promotion.
"""

from __future__ import annotations

import base64
import concurrent.futures
import dataclasses
import errno
import fcntl
import hashlib
import json
import math
import os
import stat
import sys
import zipfile
import zipimport
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn, cast

import numpy as np
from scipy import __version__ as scipy_version
from scipy.stats import t as student_t

from alberta_framework.core.slot_signaling_agent import SlotSignalingConfig
from alberta_framework.evaluation.hidden_regime_calibration_readiness import (
    BoundCalibrationRuntimeBatch,
    ReadinessError,
    ValidatedReadinessBundle,
    bound_calibration_runtime_batch,
    build_runtime_execution_identity,
    execute_bound_calibration_worker,
    load_validated_published_readiness_bundle,
    require_current_full_runtime_identity,
    runtime_execution_identity_from_receipt,
)
from alberta_framework.evaluation.hidden_regime_execution_governance import (
    CALIBRATION_EXECUTION_INVENTORY_SCHEMA,
    CALIBRATION_EXECUTION_OUTCOME_DIGEST_SCHEMA,
    CALIBRATION_EXECUTION_PRIMITIVE_TRACE_DIGEST_SCHEMA,
    CALIBRATION_EXECUTION_RESOURCE_DIGEST_SCHEMA,
    CALIBRATION_EXECUTION_SUMMARY_DIGEST_SCHEMA,
    CALIBRATION_ZIP_PROVENANCE_SCHEMA,
    EXECUTION_AUTHORIZATION_ACKNOWLEDGEMENT,
    READINESS_EXECUTION_GOVERNANCE_FIELD,
    ZIP_PROVENANCE_POLICY,
    ZIP_PROVENANCE_SOURCE_ARCHIVE_LOCATOR,
    HiddenRegimeExecutionGovernanceError,
    PublishedCalibrationExecutionLedger,
    atomic_install_new_immutable,
    attest_calibration_zip_provenance,
    build_calibration_execution_genesis,
    calibration_case_attempt_binding,
    calibration_execution_configuration_sha256,
    calibration_execution_primitive_trace_sha256,
    calibration_execution_resource_sha256,
    calibration_execution_summary_sha256,
    finalize_calibration_case_shard,
    initialize_calibration_execution_ledger,
    issue_calibration_execution_authorization,
    load_finalized_calibration_case_shard,
    require_valid_calibration_execution_inventory,
    snapshot_calibration_execution_inventory,
    validate_completed_calibration_ledger_snapshot,
)
from alberta_framework.evaluation.hidden_regime_factorial_protected_plan import (
    PROTECTED_PLAN_SCHEMA,
    ProtectedPlanError,
    build_hidden_regime_factorial_protected_plan,
    validate_hidden_regime_factorial_protected_plan,
)
from alberta_framework.evaluation.hidden_regime_factorial_protocol import (
    BOUND_DEVELOPMENT_SUMMARY_SCHEMA,
    BOUND_PRIMITIVE_TRACE_SCHEMA,
    CALIBRATION_DESIGN_PAYLOAD_SHA256,
    CALIBRATION_MANIFEST_ORDER,
    CANONICAL_CONDITION_ORDER,
    CONSUMED_CALIBRATION_NAMESPACE,
    N_MATCHED_CASES,
    N_SEED_PAIRS,
    SEED_SNAPSHOT_SHA256,
    THRESHOLD_FREEZE_RECEIPT_SCHEMA,
    EstimandContract,
    HiddenRegimeFactorialCalibrationDesign,
    MatchedCalibrationCase,
    MetricContract,
    build_hidden_regime_factorial_calibration_design,
    canonical_json_bytes,
    canonical_sha256,
)
from alberta_framework.evaluation.hidden_regime_factorial_thresholds import (
    MANDATORY_STATISTICAL_ENDPOINT_COUNT,
    MANDATORY_STATISTICAL_ENDPOINT_IDENTITIES_SHA256,
    MANDATORY_STATISTICAL_ENDPOINT_IDS_SHA256,
    THRESHOLD_FREEZE_DECISION_FROZEN,
    THRESHOLD_FREEZE_DECISION_REJECTION,
    ThresholdFreezeError,
    materialize_hidden_regime_factorial_threshold_freeze_receipt,
    validate_hidden_regime_factorial_threshold_freeze_receipt,
)
from alberta_framework.evaluation.hidden_regime_lineage_oracle import (
    HIDDEN_REGIME_LINEAGE_ORACLE_SCHEMA,
    validate_hidden_regime_lineage_summary,
)
from alberta_framework.evaluation.hidden_regime_signaling_development import (
    HIDDEN_REGIME_DEVELOPMENT_SCHEMA,
    HIDDEN_REGIME_TRACE_SCHEMA,
    CommitGenerationLineage,
    DormantGenerationProbe,
    HiddenRegimeDevelopmentConfig,
    HiddenRegimePrimitiveTrace,
    HiddenRegimeResourceReport,
    HiddenRegimeRunResult,
    HiddenRegimeRunSummary,
    HiddenRegimeSeedPair,
    RecurrenceLineageProbe,
    RecurrenceRetentionRecord,
    RegimeRecurrenceSummary,
    RetentionAggregateSummary,
    SegmentRewardSummary,
    run_hidden_regime_condition,
)
from alberta_framework.evaluation.hidden_regime_summary_oracle import (
    HIDDEN_REGIME_SUMMARY_ORACLE_SCHEMA,
)
from alberta_framework.evaluation.hidden_regime_trace_audit import (
    EVIDENCE_BOUNDARY,
    HIDDEN_REGIME_TRACE_AUDIT_REPORT_SCHEMA,
    HiddenRegimeTraceAuditReport,
    audit_hidden_regime_run_result,
)
from alberta_framework.streams.hidden_regime_signaling import (
    CALIBRATION_ONLY_PARTITION,
    PROTECTED_CANDIDATE_PARTITION,
    hidden_regime_calibration_manifest,
)

CALIBRATION_CASE_REQUEST_SCHEMA = "alberta.hidden-regime-factorial.case-request.v1"
CALIBRATION_CASE_REQUEST_BINDING_SCHEMA = (
    "alberta.hidden-regime-factorial.case-request-binding.v1"
)
CALIBRATION_PREFLIGHT_REQUEST_SCHEMA = "alberta.hidden-regime-factorial.preflight-request.v1"
CALIBRATION_PREFLIGHT_REPORT_SCHEMA = "alberta.hidden-regime-factorial.preflight-report.v1"
CALIBRATION_CASE_SHARD_SCHEMA = "alberta.hidden-regime-factorial.case-shard.v3"
CALIBRATION_ATTEMPT_SCHEMA = "alberta.hidden-regime-factorial.case-attempt.v1"
CALIBRATION_LEDGER_SCHEMA = "alberta.hidden-regime-factorial.case-ledger.v1"
CALIBRATION_AGGREGATE_SCHEMA = "alberta.hidden-regime-factorial.calibration-aggregate.v4"
CALIBRATION_MANDATORY_AUDIT_SUMMARY_SCHEMA = (
    "alberta.hidden-regime-factorial.mandatory-audit-summary.v1"
)
THRESHOLD_FREEZE_WORKER_RESULT_SCHEMA = (
    "alberta.hidden-regime-factorial.threshold-freeze-worker-result.v1"
)
THRESHOLD_FREEZE_EXACT_INPUT_BINDING_SCHEMA = (
    "alberta.hidden-regime-factorial.threshold-freeze-exact-input-binding.v1"
)
PROTECTED_PLAN_WORKER_RESULT_SCHEMA = (
    "alberta.hidden-regime-factorial.protected-plan-worker-result.v1"
)
PRIMITIVE_TRACE_DIGEST_SCHEMA = "alberta.hidden-regime-factorial.trace-digest.v1"

EXECUTION_ACKNOWLEDGEMENT = EXECUTION_AUTHORIZATION_ACKNOWLEDGEMENT
PREFLIGHT_ACKNOWLEDGEMENT = (
    "authorize a non-consuming ZIP preflight and optional process-local authorization issuance"
)
WORKER_RESULT_PREFIX = b"ALBERTA_HIDDEN_REGIME_CALIBRATION_SHARD_V1:"
PREFLIGHT_RESULT_PREFIX = b"ALBERTA_HIDDEN_REGIME_CALIBRATION_PREFLIGHT_V1:"
AGGREGATE_RESULT_PREFIX = b"ALBERTA_HIDDEN_REGIME_CALIBRATION_AGGREGATE_V1:"
THRESHOLD_FREEZE_RESULT_PREFIX = b"ALBERTA_HIDDEN_REGIME_THRESHOLD_FREEZE_V1:"
PROTECTED_PLAN_RESULT_PREFIX = b"ALBERTA_HIDDEN_REGIME_PROTECTED_PLAN_V1:"

EXPECTED_STEPS = 16_528
EXPECTED_CONDITIONS = 8
EXPECTED_SEED_PAIRS = 30
EXPECTED_CASES = 240
EXPECTED_MANIFEST_CASES = 80
EXPECTED_MANIFEST_SEED_PAIRS = 10

READINESS_EQUIVALENCE_CERTIFICATION_ID = (
    "checkpoint_resume_and_decentralized_role_bit_exact_equivalence"
)

_SHA256_LENGTH = 64
_MAX_RECEIPT_BYTES = 4 * 1024 * 1024
_MAX_SOURCE_ZIP_BYTES = 32 * 1024 * 1024
_MAX_WORKER_OUTPUT_BYTES = 8 * 1024 * 1024
_MAX_SHARD_BYTES = 8 * 1024 * 1024
_MAX_AGGREGATE_BYTES = 64 * 1024 * 1024
_MAX_PROTECTED_PLAN_BYTES = 32 * 1024 * 1024
_MAX_CALIBRATION_WORKERS = 16

type RecurrenceIdentity = tuple[int, int, int, int]
type JSONScalar = str | int | bool | None
type JSONValue = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]


class CalibrationError(RuntimeError):
    """Raised whenever execution or aggregation cannot be proven exact."""


def _fail(message: str) -> NoReturn:
    raise CalibrationError(message)


def _require(condition: bool, message: str) -> None:
    if not condition:
        _fail(message)


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == _SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _plain_dict(value: object, label: str) -> dict[str, object]:
    if type(value) is not dict:
        _fail(f"{label} must be a plain dict")
    return cast(dict[str, object], value)


def _plain_list(value: object, label: str) -> list[object]:
    if type(value) is not list:
        _fail(f"{label} must be a plain list")
    return cast(list[object], value)


def _exact_keys(value: Mapping[str, object], keys: set[str], label: str) -> None:
    if set(value) != keys:
        _fail(f"{label} keys are not exact")


def _strict_int(value: object, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        _fail(f"{label} must be a strict integer >= {minimum}")
    return value


def _strict_bool(value: object, label: str) -> bool:
    if type(value) is not bool:
        _fail(f"{label} must be a strict boolean")
    return value


def _float_hex(value: object, label: str) -> str:
    if not isinstance(value, (float, np.floating)) or isinstance(value, bool):
        _fail(f"{label} must be a floating-point value")
    result = float(value)
    if not math.isfinite(result):
        _fail(f"{label} must be finite")
    return result.hex()


def _parse_float_hex(value: object, label: str) -> float:
    if type(value) is not str:
        _fail(f"{label} must be canonical float.hex text")
    try:
        result = float.fromhex(value)
    except ValueError as exc:
        raise CalibrationError(f"{label} is not float.hex text") from exc
    if not math.isfinite(result) or result.hex() != value:
        _fail(f"{label} is not canonical finite float.hex text")
    return result


def _encode_exact(value: object, *, label: str = "value") -> JSONValue:
    """Convert dataclass output to integer/string-only canonical JSON values."""

    if value is None or type(value) in (str, int, bool):
        return cast(JSONScalar, value)
    if isinstance(value, (float, np.floating)):
        return _float_hex(value, label)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _encode_exact(dataclasses.asdict(value), label=label)
    if isinstance(value, tuple | list):
        return [_encode_exact(item, label=f"{label}[{index}]") for index, item in enumerate(value)]
    if type(value) is dict:
        output: dict[str, JSONValue] = {}
        for key, item in cast(dict[object, object], value).items():
            if type(key) is not str:
                _fail(f"{label} contains a non-string key")
            output[key] = _encode_exact(item, label=f"{label}.{key}")
        return output
    _fail(f"{label} contains unsupported type {type(value).__name__}")


def _payload_with_digest(body: Mapping[str, object]) -> dict[str, object]:
    normalized = _plain_dict(_encode_exact(dict(body)), "canonical body")
    return {**normalized, "payload_sha256": canonical_sha256(normalized)}


def _validate_payload_digest(payload: Mapping[str, object], label: str) -> dict[str, object]:
    normalized = dict(payload)
    digest = normalized.pop("payload_sha256", None)
    if not _is_sha256(digest):
        _fail(f"{label}.payload_sha256 is not lowercase SHA-256")
    if canonical_sha256(normalized) != digest:
        _fail(f"{label} payload digest mismatch")
    return normalized


def _open_directory_without_symlinks(path: Path, *, label: str) -> tuple[int, Path]:
    absolute = Path(os.path.abspath(os.fspath(path)))
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open("/", flags)
    try:
        for component in absolute.parts[1:]:
            try:
                next_descriptor = os.open(component, flags, dir_fd=descriptor)
            except OSError as exc:
                if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
                    raise CalibrationError(
                        f"{label} has a symlinked directory component: {absolute}"
                    ) from exc
                raise
            os.close(descriptor)
            descriptor = next_descriptor
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor, absolute


def _read_regular_file(path: Path, *, max_bytes: int, label: str) -> bytes:
    parent_fd, absolute_parent = _open_directory_without_symlinks(path.parent, label=label)
    parent_status = os.fstat(parent_fd)
    absolute = absolute_parent / path.name
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path.name, flags, dir_fd=parent_fd)
    except OSError as exc:
        os.close(parent_fd)
        if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise CalibrationError(f"{label} is symlinked: {absolute}") from exc
        raise
    try:
        before = os.fstat(descriptor)
        _require(stat.S_ISREG(before.st_mode), f"{label} is not a regular file")
        _require(stat.S_IMODE(before.st_mode) == 0o444, f"{label} mode is not immutable 0444")
        _require(before.st_nlink == 1, f"{label} has multiple hard links")
        _require(before.st_size <= max_bytes, f"{label} exceeds maximum size")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, max_bytes + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            _require(total <= max_bytes, f"{label} exceeds maximum size")
        after = os.fstat(descriptor)
        locator = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)

        def identity(item: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
            return (
                item.st_dev,
                item.st_ino,
                item.st_mode,
                item.st_nlink,
                item.st_size,
                item.st_mtime_ns,
                item.st_ctime_ns,
            )

        _require(
            identity(before) == identity(after) == identity(locator),
            f"{label} changed while reading",
        )
        reopened_parent_fd, reopened_parent = _open_directory_without_symlinks(
            absolute_parent,
            label=label,
        )
        try:
            reopened_status = os.fstat(reopened_parent_fd)
            _require(
                reopened_parent == absolute_parent
                and (reopened_status.st_dev, reopened_status.st_ino)
                == (parent_status.st_dev, parent_status.st_ino),
                f"{label} parent changed while reading",
            )
        finally:
            os.close(reopened_parent_fd)
        return b"".join(chunks)
    finally:
        os.close(descriptor)
        os.close(parent_fd)


def _strict_json(raw: bytes, label: str) -> dict[str, object]:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CalibrationError(f"{label} is not canonical JSON") from exc
    payload = _plain_dict(value, label)
    if canonical_json_bytes(payload) != raw:
        _fail(f"{label} is not byte-canonical JSON")
    return payload


def _design() -> HiddenRegimeFactorialCalibrationDesign:
    design = build_hidden_regime_factorial_calibration_design()
    _require(N_MATCHED_CASES == EXPECTED_CASES, "protocol case count changed")
    _require(N_SEED_PAIRS == EXPECTED_SEED_PAIRS, "protocol seed count changed")
    _require(
        len(CANONICAL_CONDITION_ORDER) == EXPECTED_CONDITIONS,
        "protocol condition count changed",
    )
    _require(
        canonical_sha256(design.to_payload()) == CALIBRATION_DESIGN_PAYLOAD_SHA256,
        "protocol payload differs from frozen digest",
    )
    return design


def _case(
    design: HiddenRegimeFactorialCalibrationDesign, case_index: int
) -> MatchedCalibrationCase:
    if type(case_index) is not int or not 0 <= case_index < EXPECTED_CASES:
        _fail("case_index must be a strict integer in [0, 240)")
    case = design.cases[case_index]
    _require(case.case_index == case_index, "protocol case ledger is not index-addressable")
    return case


def _runtime_binding(
    design: HiddenRegimeFactorialCalibrationDesign,
    condition: str,
) -> object:
    matches = tuple(
        binding for binding in design.condition_runtime_bindings if binding.condition == condition
    )
    _require(len(matches) == 1, "condition has no unique frozen runtime binding")
    return matches[0]


def _manifest_binding(
    design: HiddenRegimeFactorialCalibrationDesign,
    manifest_name: str,
) -> object:
    matches = tuple(
        binding for binding in design.manifest_bindings if binding.name == manifest_name
    )
    _require(len(matches) == 1, "case has no unique frozen manifest binding")
    return matches[0]


def _recurrence_binding(
    design: HiddenRegimeFactorialCalibrationDesign,
    manifest_name: str,
) -> object:
    matches = tuple(
        binding for binding in design.recurrence_bindings if binding.manifest_name == manifest_name
    )
    _require(len(matches) == 1, "case has no unique frozen recurrence binding")
    return matches[0]


def _build_case_config(
    design: HiddenRegimeFactorialCalibrationDesign,
    case: MatchedCalibrationCase,
) -> HiddenRegimeDevelopmentConfig:
    base = design.base_config_binding
    manifest = hidden_regime_calibration_manifest(case.manifest_name)
    _require(
        manifest.use_partition == CALIBRATION_ONLY_PARTITION, "manifest is not calibration-only"
    )
    learner = SlotSignalingConfig(
        learning_rate=float(base.learning_rate_decimal),
        epsilon=float(base.epsilon_decimal),
        relevance_rate=float(base.relevance_rate_decimal),
        lease_length=base.lease_length,
        confirmation_steps=base.confirmation_steps,
        durable_retrieval_threshold=float(base.durable_retrieval_threshold_decimal),
        candidate_confirmation_threshold=float(base.candidate_confirmation_threshold_decimal),
        candidate_confirmation_leases=base.candidate_confirmation_leases,
        scratch_training_leases_before_retest=base.scratch_training_leases_before_retest,
        writable_lru_ablation=base.writable_lru_ablation,
        durable_write_policy=base.requested_durable_write_policy,
        replacement_target_policy=base.requested_replacement_target_policy,
    )
    config = HiddenRegimeDevelopmentConfig(
        world=manifest.to_world_config(repeat_schedule=False),
        learner=learner,
        metric_window=base.metric_window,
    )
    _require(config.num_steps == EXPECTED_STEPS, "case is not exactly one 16,528-step schedule")
    _require(config.world.repeat_schedule is False, "case schedule unexpectedly repeats")
    _require(
        config.learner.effective_durable_write_policy == base.effective_base_durable_write_policy,
        "base durable-write policy drift",
    )
    _require(
        config.learner.effective_replacement_target_policy
        == base.effective_base_replacement_target_policy,
        "base replacement-target policy drift",
    )
    return config


def _seed_pair(case: MatchedCalibrationCase) -> HiddenRegimeSeedPair:
    return HiddenRegimeSeedPair(
        namespace=CONSUMED_CALIBRATION_NAMESPACE,
        index=case.seed_index,
        world_seed=case.world_seed,
        learner_seed=case.learner_seed,
    )


def _trace_sha256(trace: HiddenRegimePrimitiveTrace) -> str:
    """Hash schema, field names, dtypes, shapes, and exact C-order bytes."""

    hasher = hashlib.sha256()
    hasher.update(PRIMITIVE_TRACE_DIGEST_SCHEMA.encode("ascii"))
    hasher.update(b"\0")
    hasher.update(HIDDEN_REGIME_TRACE_SCHEMA.encode("ascii"))
    for field in dataclasses.fields(trace):
        array = np.ascontiguousarray(np.asarray(getattr(trace, field.name)))
        header = {
            "field": field.name,
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "nbytes": int(array.nbytes),
        }
        hasher.update(canonical_json_bytes(header))
        hasher.update(b"\0")
        hasher.update(array.tobytes(order="C"))
    digest = hasher.hexdigest()
    _require(
        digest == calibration_execution_primitive_trace_sha256(trace),
        "primitive trace digest differs from managed completion encoding",
    )
    return digest


def _zip_worker_provenance(
    archive_path: Path,
    bundle: ValidatedReadinessBundle,
) -> dict[str, object]:
    """Prove every loaded project module came from the bound ZIP in an empty cwd."""

    absolute_archive = archive_path.resolve(strict=True)
    _require(
        hashlib.sha256(
            _read_regular_file(
                absolute_archive,
                max_bytes=_MAX_SOURCE_ZIP_BYTES,
                label="worker source ZIP",
            )
        ).hexdigest()
        == bundle.source_archive_sha256,
        "worker source ZIP digest differs from readiness",
    )
    _require(not any(Path.cwd().iterdir()), "worker cwd was not empty before execution")
    _require(sys.flags.no_site == 1, "worker did not disable automatic site initialization")
    _require(
        all(
            name not in sys.modules
            for name in ("_virtualenv", "site", "sitecustomize", "usercustomize")
        ),
        "worker loaded a site or pre-bootstrap path-hook module",
    )
    readiness_body = _plain_dict(bundle.payload.get("body"), "readiness body")
    runtime_identity = _plain_dict(
        readiness_body.get("runtime_identity"),
        "readiness runtime identity",
    )
    runtime_python = _plain_dict(runtime_identity.get("python"), "readiness runtime python")
    runtime_paths: dict[str, str] = {}
    for field in ("prefix", "exec_prefix", "purelib", "platlib"):
        value = runtime_python.get(field)
        _require(type(value) is str and os.path.isabs(value), f"runtime {field} is invalid")
        runtime_paths[field] = cast(str, value)
    no_site_stdlib_search_paths = tuple(
        _plain_list(
            runtime_python.get("no_site_stdlib_search_paths"),
            "readiness no-site stdlib search paths",
        )
    )
    _require(
        bool(no_site_stdlib_search_paths)
        and all(
            type(entry) is str and os.path.isabs(entry)
            for entry in no_site_stdlib_search_paths
        )
        and len(no_site_stdlib_search_paths) == len(set(no_site_stdlib_search_paths)),
        "readiness no-site stdlib search paths are invalid",
    )
    _require(
        sys.prefix == runtime_paths["prefix"]
        and sys.exec_prefix == runtime_paths["exec_prefix"],
        "worker runtime prefix differs from readiness",
    )
    expected_site_paths = tuple(
        dict.fromkeys((runtime_paths["purelib"], runtime_paths["platlib"]))
    )
    _require(
        all(type(entry) is str for entry in sys.path),
        "worker import search path contains a non-string entry",
    )
    expected_import_search_path = (
        absolute_archive.as_posix(),
        *cast(tuple[str, ...], no_site_stdlib_search_paths),
        *expected_site_paths,
    )
    _require(
        len(expected_import_search_path) == len(set(expected_import_search_path)),
        "readiness-bound import search path contains overlapping entries",
    )
    _require(
        tuple(sys.path) == expected_import_search_path,
        "worker import search path differs from exact readiness-bound construction",
    )
    _require(sys.dont_write_bytecode is True, "worker bytecode writes are not disabled")
    pycache_value = sys.pycache_prefix
    _require(
        type(pycache_value) is str and bool(pycache_value),
        "worker command-line pycache prefix is missing",
    )
    xoptions = getattr(sys, "_xoptions", {})
    _require(
        type(xoptions) is dict and xoptions.get("pycache_prefix") == pycache_value,
        "worker pycache prefix was not supplied on the command line",
    )
    pycache_prefix = Path(cast(str, pycache_value)).absolute()
    pycache_fd, _ = _open_directory_without_symlinks(
        pycache_prefix,
        label="worker pycache prefix",
    )
    try:
        _require(not os.listdir(pycache_fd), "worker pycache prefix is not fresh and empty")
    finally:
        os.close(pycache_fd)

    def overlaps(first: Path, second: Path) -> bool:
        common = Path(os.path.commonpath((first.as_posix(), second.as_posix())))
        return common == first or common == second

    _require(
        not overlaps(pycache_prefix, Path.cwd().absolute())
        and not overlaps(pycache_prefix, absolute_archive.parent)
        and not overlaps(pycache_prefix, absolute_archive),
        "worker pycache prefix overlaps a bound source or working path",
    )
    _require(Path(sys.path[0]).resolve(strict=True) == absolute_archive, "source ZIP is not first")
    module_rows: list[dict[str, object]] = []
    archive_prefix = f"{absolute_archive.as_posix()}/"
    for name, module in sorted(sys.modules.items()):
        if name != "alberta_framework" and not name.startswith("alberta_framework."):
            continue
        loader = getattr(module, "__loader__", None)
        origin = getattr(module, "__file__", None)
        _require(
            isinstance(loader, zipimport.zipimporter), f"project module is not zipimport: {name}"
        )
        if type(origin) is not str or not origin.startswith(archive_prefix):
            _fail(f"project module origin is outside source ZIP: {name}")
        module_rows.append(
            {
                "module": name,
                "archive_member": origin[len(archive_prefix) :],
            }
        )
    _require(bool(module_rows), "worker loaded no project modules")
    with zipfile.ZipFile(absolute_archive, "r") as source_zip:
        names = set(source_zip.namelist())
    _require(
        all(cast(str, row["archive_member"]) in names for row in module_rows),
        "worker module origin is not a ZIP member",
    )
    mutable_project_paths = [
        entry for entry in sys.path[1:] if entry and (Path(entry) / "alberta_framework").is_dir()
    ]
    _require(not mutable_project_paths, "mutable project checkout remains on worker sys.path")
    return {
        "source_archive_sha256": bundle.source_archive_sha256,
        "source_manifest_sha256": bundle.source_manifest_sha256,
        "loader": "zipimport.zipimporter",
        "source_zip_first": True,
        "mutable_project_path_count": 0,
        "fresh_empty_working_directory": True,
        "no_site_startup": True,
        "prebootstrap_pth_hook_absent": True,
        "receipt_bound_runtime_prefix": True,
        "exact_receipt_bound_site_search_paths": True,
        "dont_write_bytecode": True,
        "command_line_pycache_prefix": True,
        "pycache_prefix_fresh_empty_nonsymlink": True,
        "pycache_prefix_outside_bound_paths": True,
        "project_module_count": len(module_rows),
        "project_modules_sha256": canonical_sha256(module_rows),
    }


def _decode_exact(value: object) -> object:
    """Decode only canonical hexadecimal floating-point leaves."""

    if type(value) is str and (value.startswith("0x") or value.startswith("-0x")):
        try:
            parsed = float.fromhex(value)
        except ValueError:
            return value
        if math.isfinite(parsed) and parsed.hex() == value:
            return parsed
    if type(value) is list:
        return [_decode_exact(item) for item in cast(list[object], value)]
    if type(value) is dict:
        return {
            cast(str, key): _decode_exact(item)
            for key, item in cast(dict[object, object], value).items()
        }
    return value


def _number(value: object, label: str) -> float:
    if not isinstance(value, (int, float, np.integer, np.floating)) or isinstance(value, bool):
        _fail(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        _fail(f"{label} must be finite")
    return result


def _optional_number(value: object, label: str) -> float | None:
    return None if value is None else _number(value, label)


def _identity(record: Mapping[str, object]) -> RecurrenceIdentity:
    return (
        _strict_int(record.get("segment_index"), "recurrence.segment_index"),
        _strict_int(record.get("regime_id"), "recurrence.regime_id"),
        _strict_int(record.get("occurrence_index"), "recurrence.occurrence_index"),
        _strict_int(
            record.get("raw_segment_occurrence_index"),
            "recurrence.raw_segment_occurrence_index",
        ),
    )


def _identity_payload(identity: RecurrenceIdentity) -> list[int]:
    return list(identity)


def _parse_identity(value: object, label: str) -> RecurrenceIdentity:
    values = _plain_list(value, label)
    if len(values) != 4:
        _fail(f"{label} must contain four strict integers")
    return cast(
        RecurrenceIdentity,
        tuple(_strict_int(item, f"{label}[{index}]") for index, item in enumerate(values)),
    )


def _canonical_identities(values: Iterable[RecurrenceIdentity]) -> tuple[RecurrenceIdentity, ...]:
    return tuple(sorted(values))


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64), dtype=np.float64))


def _ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0.0:
        return None
    return numerator / denominator


_QUALIFIED_METRICS = frozenset(
    {
        "qualified_first_entry_window_error_rate",
        "latest_prior_qualified_lineage_survival_rate",
        "selected_lineage_joint_bit_exact_preservation_rate",
        "selected_lineage_exact_generation_relock_rate",
        "selected_lineage_retrieval_before_scratch_rate",
        "recurrence_minus_latest_qualified_acquisition_error_rate",
        "any_qualified_lineage_survival_rate",
    }
)
_SELECTED_METRICS = frozenset(
    {
        "selected_lineage_entry_composed_accuracy",
        "selected_lineage_commit_to_entry_accuracy_change",
        "selected_lineage_dormant_at_entry_rate",
        "selected_lineage_active_at_entry_rate",
    }
)
_DORMANT_METRICS = frozenset(
    {
        "all_dormant_probe_composed_accuracy",
        "all_dormant_probe_composed_minus_zero_helper_accuracy",
        "all_dormant_probe_composed_minus_zero_beneficiary_accuracy",
        "all_dormant_probe_composed_minus_role_swapped_accuracy",
    }
)
_BEST_DORMANT_METRICS = frozenset(
    {
        "best_dormant_composed_accuracy",
        "best_dormant_zero_helper_accuracy",
        "best_dormant_zero_beneficiary_accuracy",
        "best_dormant_role_swapped_accuracy",
    }
)


def _metric_value(
    metric_id: str,
    records: Sequence[Mapping[str, object]],
    summary: Mapping[str, object],
) -> tuple[float | None, tuple[RecurrenceIdentity, ...], tuple[RecurrenceIdentity, ...]]:
    """Recompute one predeclared run metric from its persisted direct sources."""

    all_ids = _canonical_identities(_identity(record) for record in records)
    qualified = tuple(
        record for record in records if record.get("lineage_retention_applicable") is True
    )
    selected = tuple(
        record for record in records if record.get("selected_lineage_available") is True
    )

    if metric_id == "mean_prequential_reward":
        reward_value = _number(
            summary.get("mean_prequential_reward"),
            "summary.mean_prequential_reward",
        )
        return reward_value, (), ()
    if metric_id == "acquisition_qualified_recurrence_coverage_rate":
        coverage_value = _ratio(float(len(qualified)), float(len(records)))
        return coverage_value, all_ids, all_ids
    if metric_id in {
        "qualified_first_entry_window_error_rate",
        "all_recurrence_first_entry_window_error_rate",
    }:
        population = qualified if metric_id.startswith("qualified_") else tuple(records)
        observed = tuple(
            record
            for record in population
            if record.get("first_world_window_complete") is True
            and record.get("first_world_window_errors") is not None
            and record.get("first_world_window_length") is not None
        )
        numerator = math.fsum(
            _number(record["first_world_window_errors"], "first_world_window_errors")
            for record in observed
        )
        denominator = math.fsum(
            _number(record["first_world_window_length"], "first_world_window_length")
            for record in observed
        )
        return (
            _ratio(numerator, denominator),
            _canonical_identities(_identity(record) for record in population),
            _canonical_identities(_identity(record) for record in observed),
        )
    if metric_id == "latest_prior_qualified_lineage_survival_rate":
        latest_values = tuple(
            record.get("latest_prior_qualified_survived") is True for record in qualified
        )
        return (
            _ratio(float(sum(latest_values)), float(len(latest_values))),
            _canonical_identities(_identity(record) for record in qualified),
            _canonical_identities(_identity(record) for record in qualified),
        )
    if metric_id == "any_qualified_lineage_survival_rate":
        any_values = tuple(
            record.get("any_prior_qualified_survived") is True for record in qualified
        )
        return (
            _ratio(float(sum(any_values)), float(len(any_values))),
            _canonical_identities(_identity(record) for record in qualified),
            _canonical_identities(_identity(record) for record in qualified),
        )
    if metric_id == "selected_lineage_joint_bit_exact_preservation_rate":
        bit_values = tuple(
            record.get("selected_lineage_available") is True
            and record.get("selected_lineage_joint_bit_exact_preserved") is True
            for record in qualified
        )
        return (
            _ratio(float(sum(bit_values)), float(len(bit_values))),
            _canonical_identities(_identity(record) for record in qualified),
            _canonical_identities(_identity(record) for record in qualified),
        )
    if metric_id == "selected_lineage_exact_generation_relock_rate":
        relock_values = tuple(
            record.get("selected_lineage_available") is True
            and record.get("selected_exact_generation_relock_observed") is True
            for record in qualified
        )
        return (
            _ratio(float(sum(relock_values)), float(len(relock_values))),
            _canonical_identities(_identity(record) for record in qualified),
            _canonical_identities(_identity(record) for record in qualified),
        )
    if metric_id == "selected_lineage_retrieval_before_scratch_rate":
        retrieval_values = tuple(
            record.get("selected_lineage_available") is True
            and record.get("selected_durable_retrieval_before_scratch") is True
            for record in qualified
        )
        return (
            _ratio(float(sum(retrieval_values)), float(len(retrieval_values))),
            _canonical_identities(_identity(record) for record in qualified),
            _canonical_identities(_identity(record) for record in qualified),
        )
    if metric_id in {
        "selected_lineage_entry_composed_accuracy",
        "selected_lineage_commit_to_entry_accuracy_change",
    }:
        field = (
            "selected_lineage_entry_composed_greedy_accuracy"
            if metric_id.endswith("composed_accuracy")
            else "selected_lineage_entry_minus_commit_accuracy"
        )
        observed = tuple(record for record in selected if record.get(field) is not None)
        selected_values = tuple(_number(record[field], field) for record in observed)
        return (
            _mean(selected_values),
            _canonical_identities(_identity(record) for record in selected),
            _canonical_identities(_identity(record) for record in observed),
        )
    if metric_id == "recurrence_minus_latest_qualified_acquisition_error_rate":
        observed = tuple(
            record
            for record in qualified
            if record.get("latest_qualified_acquisition_comparison_available") is True
            and record.get("recurrence_minus_latest_qualified_acquisition_error_rate_delta")
            is not None
        )
        comparison_values = tuple(
            _number(
                record["recurrence_minus_latest_qualified_acquisition_error_rate_delta"],
                "recurrence_minus_latest_qualified_acquisition_error_rate_delta",
            )
            for record in observed
        )
        return (
            _mean(comparison_values),
            _canonical_identities(_identity(record) for record in qualified),
            _canonical_identities(_identity(record) for record in observed),
        )
    if metric_id == "all_surviving_qualified_lineage_entry_composed_accuracy":
        lineage_values: list[float] = []
        lineage_observed_ids: list[RecurrenceIdentity] = []
        for record in records:
            probes = _plain_list(record.get("prior_same_regime_lineages"), "lineage probes")
            found = False
            for raw_probe in probes:
                probe = _plain_dict(raw_probe, "lineage probe")
                if (
                    probe.get("acquisition_qualified") is True
                    and probe.get("synchronized_generation_survives") is True
                    and probe.get("entry_composed_greedy_accuracy") is not None
                ):
                    lineage_values.append(
                        _number(
                            probe["entry_composed_greedy_accuracy"],
                            "entry_composed_greedy_accuracy",
                        )
                    )
                    found = True
            if found:
                lineage_observed_ids.append(_identity(record))
        lineage_observed = _canonical_identities(lineage_observed_ids)
        return _mean(lineage_values), lineage_observed, lineage_observed
    if metric_id in _DORMANT_METRICS:
        dormant_values: list[float] = []
        dormant_observed_ids: list[RecurrenceIdentity] = []
        for record in records:
            probes = _plain_list(record.get("eligible_dormant_generations"), "dormant probes")
            if probes:
                dormant_observed_ids.append(_identity(record))
            for raw_probe in probes:
                probe = _plain_dict(raw_probe, "dormant probe")
                composed = _number(probe.get("composed_greedy_accuracy"), "composed accuracy")
                if metric_id == "all_dormant_probe_composed_accuracy":
                    value = composed
                elif metric_id.endswith("zero_helper_accuracy"):
                    value = composed - _number(probe.get("zero_helper_accuracy"), "zero helper")
                elif metric_id.endswith("zero_beneficiary_accuracy"):
                    value = composed - _number(
                        probe.get("zero_beneficiary_accuracy"), "zero beneficiary"
                    )
                else:
                    value = composed - _number(probe.get("role_swapped_accuracy"), "role swapped")
                dormant_values.append(value)
        dormant_observed = _canonical_identities(dormant_observed_ids)
        return _mean(dormant_values), dormant_observed, dormant_observed
    if metric_id in _BEST_DORMANT_METRICS:
        field = {
            "best_dormant_composed_accuracy": "best_dormant_composed_greedy_accuracy",
            "best_dormant_zero_helper_accuracy": "best_dormant_zero_helper_accuracy",
            "best_dormant_zero_beneficiary_accuracy": "best_dormant_zero_beneficiary_accuracy",
            "best_dormant_role_swapped_accuracy": "best_dormant_role_swapped_accuracy",
        }[metric_id]
        observed_records = tuple(record for record in records if record.get(field) is not None)
        best_values = tuple(_number(record[field], field) for record in observed_records)
        best_observed = _canonical_identities(_identity(record) for record in observed_records)
        return _mean(best_values), best_observed, best_observed
    if metric_id in {
        "selected_lineage_dormant_at_entry_rate",
        "selected_lineage_active_at_entry_rate",
    }:
        expected = "dormant" if "dormant" in metric_id else "active"
        observed_records = tuple(
            record
            for record in selected
            if record.get("selected_lineage_entry_activity_status") is not None
        )
        activity_values = tuple(
            record.get("selected_lineage_entry_activity_status") == expected
            for record in observed_records
        )
        return (
            _ratio(float(sum(activity_values)), float(len(activity_values))),
            _canonical_identities(_identity(record) for record in selected),
            _canonical_identities(_identity(record) for record in observed_records),
        )
    _fail(f"no frozen metric extractor exists for {metric_id!r}")


def _metric_observation(
    contract: MetricContract,
    records: Sequence[Mapping[str, object]],
    summary: Mapping[str, object],
) -> dict[str, object]:
    value, eligible, observed = _metric_value(contract.metric_id, records, summary)
    observed_set = set(observed)
    structural_missing = tuple(identity for identity in eligible if identity not in observed_set)
    return {
        "metric_id": contract.metric_id,
        "orientation": contract.orientation,
        "gate_mode": contract.gate_mode,
        "value_hex": None if value is None else _float_hex(value, contract.metric_id),
        "eligible_recurrence_identities": [_identity_payload(item) for item in eligible],
        "observed_recurrence_identities": [_identity_payload(item) for item in observed],
        "structural_missing_recurrence_identities": [
            _identity_payload(item) for item in structural_missing
        ],
        "eligible_n": len(eligible),
        "observed_n": len(observed),
        "structural_missing_n": len(structural_missing),
    }


def _metric_observations(
    design: HiddenRegimeFactorialCalibrationDesign,
    summary: Mapping[str, object],
) -> list[dict[str, object]]:
    records = tuple(
        _plain_dict(item, "summary.recurrence_retention[]")
        for item in _plain_list(
            summary.get("recurrence_retention"),
            "summary.recurrence_retention",
        )
    )
    return [_metric_observation(contract, records, summary) for contract in design.metrics]


def _validate_recurrence_population(
    design: HiddenRegimeFactorialCalibrationDesign,
    case: MatchedCalibrationCase,
    summary: Mapping[str, object],
) -> tuple[RecurrenceIdentity, ...]:
    records = tuple(
        _plain_dict(item, "summary.recurrence_retention[]")
        for item in _plain_list(summary.get("recurrence_retention"), "recurrence retention")
    )
    identities = tuple(_identity(record) for record in records)
    _require(len(set(identities)) == len(identities), "recurrence identities are not unique")
    _require(identities == tuple(sorted(identities)), "recurrence records are not canonical")
    binding = cast(Any, _recurrence_binding(design, case.manifest_name))
    expected_triples = tuple(binding.eligible_recurrence_identities)
    actual_triples = tuple(identity[:3] for identity in identities)
    _require(
        actual_triples == expected_triples, "recurrence population differs from frozen manifest"
    )
    _require(
        all(identity[3] >= identity[2] for identity in identities),
        "raw segment occurrence precedes coalesced occurrence",
    )
    return identities


def _same_optional_float(actual: object, expected: float | None, label: str) -> None:
    if actual is None or expected is None:
        _require(actual is None and expected is None, f"{label} nullability mismatch")
        return
    _require(
        _float_hex(_number(actual, label), label) == _float_hex(expected, label),
        f"{label} differs from direct metric reconstruction",
    )


def _dataclass_field_names(dataclass_type: Any) -> set[str]:
    return {field.name for field in dataclasses.fields(dataclass_type)}


def _validate_full_summary_shape(summary: Mapping[str, object]) -> None:
    _exact_keys(summary, _dataclass_field_names(HiddenRegimeRunSummary), "run summary")
    retention = _plain_dict(summary.get("retention"), "summary.retention")
    _exact_keys(
        retention,
        _dataclass_field_names(RetentionAggregateSummary),
        "summary.retention",
    )
    for raw in _plain_list(summary.get("segment_rewards"), "segment rewards"):
        _exact_keys(
            _plain_dict(raw, "segment reward"),
            _dataclass_field_names(SegmentRewardSummary),
            "segment reward",
        )
    for raw in _plain_list(summary.get("recurrence_by_regime"), "recurrence by regime"):
        _exact_keys(
            _plain_dict(raw, "regime recurrence"),
            _dataclass_field_names(RegimeRecurrenceSummary),
            "regime recurrence",
        )
    for raw in _plain_list(summary.get("commit_generation_lineages"), "commit lineages"):
        _exact_keys(
            _plain_dict(raw, "commit lineage"),
            _dataclass_field_names(CommitGenerationLineage),
            "commit lineage",
        )
    for raw in _plain_list(summary.get("recurrence_retention"), "recurrence retention"):
        recurrence = _plain_dict(raw, "recurrence retention record")
        _exact_keys(
            recurrence,
            _dataclass_field_names(RecurrenceRetentionRecord),
            "recurrence retention record",
        )
        for probe_raw in _plain_list(
            recurrence.get("prior_same_regime_lineages"),
            "recurrence lineage probes",
        ):
            _exact_keys(
                _plain_dict(probe_raw, "recurrence lineage probe"),
                _dataclass_field_names(RecurrenceLineageProbe),
                "recurrence lineage probe",
            )
        for probe_raw in _plain_list(
            recurrence.get("eligible_dormant_generations"),
            "dormant generation probes",
        ):
            _exact_keys(
                _plain_dict(probe_raw, "dormant generation probe"),
                _dataclass_field_names(DormantGenerationProbe),
                "dormant generation probe",
            )


def _validate_summary_metric_crosschecks(
    design: HiddenRegimeFactorialCalibrationDesign,
    summary: Mapping[str, object],
    observations: Sequence[Mapping[str, object]],
) -> None:
    """Cross-check predeclared values against independently persisted sources."""

    by_id = {cast(str, item["metric_id"]): item for item in observations}
    _require(
        tuple(by_id) == tuple(metric.metric_id for metric in design.metrics), "metric order drift"
    )
    retention = _plain_dict(summary.get("retention"), "summary.retention")
    checks = {
        "acquisition_qualified_recurrence_coverage_rate": "qualification_coverage_fraction",
        "latest_prior_qualified_lineage_survival_rate": (
            "latest_qualified_version_survival_fraction"
        ),
        "selected_lineage_joint_bit_exact_preservation_rate": (
            "selected_joint_bit_exact_preservation_fraction_all_qualified"
        ),
        "selected_lineage_entry_composed_accuracy": (
            "selected_entry_composed_greedy_accuracy_mean"
        ),
        "selected_lineage_commit_to_entry_accuracy_change": (
            "selected_entry_minus_commit_accuracy_mean"
        ),
        "selected_lineage_exact_generation_relock_rate": (
            "selected_exact_generation_relock_fraction_all_qualified"
        ),
        "selected_lineage_retrieval_before_scratch_rate": (
            "selected_durable_retrieval_before_scratch_fraction_all_qualified"
        ),
        "recurrence_minus_latest_qualified_acquisition_error_rate": (
            "recurrence_minus_latest_qualified_acquisition_error_rate_delta_mean"
        ),
        "all_recurrence_first_entry_window_error_rate": "first_world_window_error_rate_mean",
        "any_qualified_lineage_survival_rate": "any_qualified_knowledge_survival_fraction",
        "best_dormant_composed_accuracy": "dormant_composed_greedy_accuracy_mean",
        "best_dormant_zero_helper_accuracy": "dormant_zero_helper_accuracy_mean",
        "best_dormant_zero_beneficiary_accuracy": "dormant_zero_beneficiary_accuracy_mean",
        "best_dormant_role_swapped_accuracy": "dormant_role_swapped_accuracy_mean",
    }
    for metric_id, field in checks.items():
        value_hex = by_id[metric_id]["value_hex"]
        expected = None if value_hex is None else _parse_float_hex(value_hex, metric_id)
        _same_optional_float(retention.get(field), expected, f"summary.retention.{field}")

    records = tuple(
        _plain_dict(item, "recurrence")
        for item in _plain_list(summary.get("recurrence_retention"), "recurrence records")
    )
    qualified = tuple(
        record for record in records if record.get("lineage_retention_applicable") is True
    )
    selected = tuple(
        record for record in qualified if record.get("selected_lineage_available") is True
    )
    _require(
        retention.get("recurrence_count") == len(records), "retention recurrence count mismatch"
    )
    _require(
        retention.get("qualification_coverage_denominator") == len(records),
        "retention qualification denominator mismatch",
    )
    _require(
        retention.get("lineage_retention_applicable_count") == len(qualified),
        "retention qualified count mismatch",
    )
    _require(
        retention.get("acquisition_coverage_failure_count") == len(records) - len(qualified),
        "retention acquisition failure count mismatch",
    )
    _require(
        retention.get("selected_lineage_probe_available_count") == len(selected),
        "retention selected count mismatch",
    )
    _require(
        retention.get("selected_lineage_probe_denominator") == len(qualified),
        "retention selected denominator mismatch",
    )
    _require(
        retention.get("selected_lineage_survival_failure_count") == len(qualified) - len(selected),
        "retention selected survival-failure count mismatch",
    )


def _dependency_lock_bindings(bundle: ValidatedReadinessBundle) -> list[dict[str, object]]:
    body = _plain_dict(bundle.payload.get("body"), "readiness.body")
    source = _plain_dict(body.get("source_snapshot"), "readiness.source_snapshot")
    manifest = _plain_dict(source.get("manifest"), "readiness.source_snapshot.manifest")
    support = _plain_list(manifest.get("support_files"), "source support files")
    bindings = [
        _plain_dict(item, "source support file")
        for item in support
        if _plain_dict(item, "source support file").get("role") == "dependency_lock"
    ]
    bindings.sort(key=lambda item: cast(str, item.get("locator")))
    _require(
        tuple(item.get("locator") for item in bindings) == ("pyproject.toml", "uv.lock"),
        "readiness receipt does not bind both dependency locks",
    )
    for item in bindings:
        _require(_is_sha256(item.get("sha256")), "dependency lock digest is invalid")
    return bindings


def _readiness_binding(bundle: ValidatedReadinessBundle) -> dict[str, object]:
    body = _plain_dict(bundle.payload.get("body"), "readiness.body")
    protocol = _plain_dict(body.get("protocol_binding"), "readiness.protocol_binding")
    runtime = _plain_dict(body.get("runtime_identity"), "readiness.runtime_identity")
    dependencies = _plain_dict(runtime.get("dependencies"), "readiness dependencies")
    versions = _plain_dict(dependencies.get("key_versions"), "readiness key versions")
    governance = _plain_dict(
        body.get(READINESS_EXECUTION_GOVERNANCE_FIELD),
        "readiness execution governance",
    )
    _require(versions.get("scipy") == scipy_version, "readiness does not bind current SciPy")
    _require(
        protocol.get("protocol_payload_sha256") == CALIBRATION_DESIGN_PAYLOAD_SHA256,
        "readiness protocol digest mismatch",
    )
    _require(
        protocol.get("seed_snapshot_sha256") == SEED_SNAPSHOT_SHA256,
        "readiness seed snapshot mismatch",
    )
    _require(
        protocol.get("development_summary_schema") == BOUND_DEVELOPMENT_SUMMARY_SCHEMA,
        "readiness development schema mismatch",
    )
    _require(
        protocol.get("primitive_trace_schema") == BOUND_PRIMITIVE_TRACE_SCHEMA,
        "readiness trace schema mismatch",
    )
    _require(
        bundle.calibration_runner_module == __name__,
        "readiness source closure binds another calibration runner",
    )
    for field, expected in (
        ("protocol_payload_sha256", CALIBRATION_DESIGN_PAYLOAD_SHA256),
        ("seed_snapshot_sha256", SEED_SNAPSHOT_SHA256),
        ("source_archive_sha256", bundle.source_archive_sha256),
        ("source_manifest_sha256", bundle.source_manifest_sha256),
        ("runtime_identity_sha256", bundle.runtime_identity_sha256),
    ):
        _require(governance.get(field) == expected, f"execution governance {field} differs")
    _require(
        governance.get("protected_execution_permitted") is False,
        "execution governance permits an external evaluation partition",
    )
    _require(_is_sha256(governance.get("genesis_sha256")), "governance genesis is invalid")
    _require(
        governance["genesis_sha256"] == bundle.execution_genesis_sha256,
        "validated readiness genesis identity differs",
    )
    return {
        "readiness_receipt_sha256": bundle.receipt_sha256,
        "source_archive_sha256": bundle.source_archive_sha256,
        "source_manifest_sha256": bundle.source_manifest_sha256,
        "runtime_identity_sha256": bundle.runtime_identity_sha256,
        "dependency_locks": _dependency_lock_bindings(bundle),
        "scipy_version": scipy_version,
        "execution_governance": governance,
    }


def _aggregation_readiness_certification_binding(
    bundle: ValidatedReadinessBundle,
) -> dict[str, object]:
    """Bind aggregate-only audit claims to exact passed readiness certifications."""

    body = _plain_dict(bundle.payload.get("body"), "readiness.body")
    contract = _plain_dict(
        body.get("certification_contract"),
        "readiness certification contract",
    )
    specifications = _plain_list(
        contract.get("specifications"),
        "readiness certification specifications",
    )
    records = _plain_list(contract.get("records"), "readiness certification records")
    specification_ids = [
        _plain_dict(item, "readiness certification specification").get("certification_id")
        for item in specifications
    ]
    record_ids = [
        _plain_dict(item, "readiness certification record").get("certification_id")
        for item in records
    ]
    _require(
        bool(specification_ids)
        and all(type(item) is str and bool(item) for item in specification_ids)
        and len(specification_ids) == len(set(specification_ids)),
        "readiness certification identifiers are invalid",
    )
    _require(record_ids == specification_ids, "readiness certification record order differs")
    _require(
        all(
            _plain_dict(item, "readiness certification record").get("status") == "passed"
            and _plain_dict(item, "readiness certification record").get("exit_code") == 0
            for item in records
        ),
        "readiness certification record is not passed",
    )
    _require(
        READINESS_EQUIVALENCE_CERTIFICATION_ID in specification_ids,
        "readiness equivalence certification is absent",
    )
    _require(
        contract.get("all_required_certifications_passed") is True,
        "readiness certifications are incomplete",
    )
    _require(
        contract.get("specifications_sha256") == canonical_sha256(specifications),
        "readiness certification specification digest differs",
    )
    _require(
        contract.get("records_sha256") == canonical_sha256(records),
        "readiness certification record digest differs",
    )
    return {
        "readiness_receipt_sha256": bundle.receipt_sha256,
        "certification_ids": specification_ids,
        "certification_specifications_sha256": contract["specifications_sha256"],
        "certification_records_sha256": contract["records_sha256"],
        "all_required_certifications_passed": True,
    }


def _validate_aggregation_readiness_certification_binding(
    binding: Mapping[str, object],
) -> dict[str, object]:
    normalized = _plain_dict(binding, "aggregation readiness certification binding")
    _exact_keys(
        normalized,
        {
            "readiness_receipt_sha256",
            "certification_ids",
            "certification_specifications_sha256",
            "certification_records_sha256",
            "all_required_certifications_passed",
        },
        "aggregation readiness certification binding",
    )
    for field in (
        "readiness_receipt_sha256",
        "certification_specifications_sha256",
        "certification_records_sha256",
    ):
        _require(_is_sha256(normalized[field]), f"aggregation certification {field} invalid")
    certification_ids = _plain_list(
        normalized["certification_ids"],
        "aggregation certification identifiers",
    )
    _require(
        bool(certification_ids)
        and all(type(item) is str and bool(item) for item in certification_ids)
        and len(certification_ids) == len(set(certification_ids)),
        "aggregation certification identifiers are invalid",
    )
    _require(
        READINESS_EQUIVALENCE_CERTIFICATION_ID in certification_ids,
        "aggregation equivalence certification is absent",
    )
    _require(
        normalized["all_required_certifications_passed"] is True,
        "aggregation readiness certifications are incomplete",
    )
    return normalized


def load_validated_readiness_bundle(
    directory: Path,
    *,
    recheck_current: bool = False,
    recheck_runtime: bool = True,
) -> ValidatedReadinessBundle:
    """Load one coherent published receipt/ZIP byte pair exactly once."""

    try:
        bundle = load_validated_published_readiness_bundle(
            directory,
            recheck_current=recheck_current,
            recheck_runtime=recheck_runtime,
        )
    except ReadinessError as exc:
        raise CalibrationError(str(exc)) from exc
    _readiness_binding(bundle)
    return bundle


def _execution_genesis(bundle: ValidatedReadinessBundle) -> dict[str, object]:
    genesis = build_calibration_execution_genesis(
        source_archive_sha256=bundle.source_archive_sha256,
        source_manifest_sha256=bundle.source_manifest_sha256,
        runtime_identity_sha256=bundle.runtime_identity_sha256,
    )
    binding = _readiness_binding(bundle)
    _require(
        _plain_dict(binding["execution_governance"], "execution governance").get("genesis_sha256")
        == genesis["genesis_sha256"],
        "reconstructed execution genesis differs from readiness",
    )
    return genesis


def initialize_calibration_ledger_from_readiness(
    readiness_directory: Path,
    publication_root: Path,
    *,
    authorize_initialization: bool,
) -> PublishedCalibrationExecutionLedger:
    """Publish one pristine content-addressed 240-case execution ledger."""

    _require(authorize_initialization is True, "ledger initialization requires authorization")
    bundle = load_validated_readiness_bundle(
        readiness_directory,
        recheck_current=False,
        recheck_runtime=True,
    )
    return initialize_calibration_execution_ledger(
        publication_root,
        _execution_genesis(bundle),
        authorize_initialization=True,
    )


@dataclass(frozen=True, slots=True)
class CalibrationCaseRequest:
    """Exact authorization and provenance binding for one immutable case attempt."""

    case_index: int
    case_binding: dict[str, object]
    readiness_binding: dict[str, object]
    managed_ledger_genesis_sha256: str
    allow_exact_replay: bool
    explicit_acknowledgement: str
    schema: str = CALIBRATION_CASE_REQUEST_SCHEMA

    def to_payload(self) -> dict[str, object]:
        return _payload_with_digest(
            {
                "schema": self.schema,
                "development_only": True,
                "scientific_promotion_allowed": False,
                "thresholds_frozen": False,
                "case_index": self.case_index,
                "case_binding": self.case_binding,
                "protocol_payload_sha256": CALIBRATION_DESIGN_PAYLOAD_SHA256,
                "seed_snapshot_sha256": SEED_SNAPSHOT_SHA256,
                "readiness_binding": self.readiness_binding,
                "managed_ledger_genesis_sha256": self.managed_ledger_genesis_sha256,
                "allow_exact_replay": self.allow_exact_replay,
                "explicit_acknowledgement": self.explicit_acknowledgement,
            }
        )


def calibration_case_request_binding_sha256(request: CalibrationCaseRequest) -> str:
    """Hash the immutable request identity without attempt-local replay consent."""

    _require(
        type(request) is CalibrationCaseRequest,
        "case request binding requires the exact request type",
    )
    body = {
        "schema": CALIBRATION_CASE_REQUEST_BINDING_SCHEMA,
        "case_request_schema": request.schema,
        "development_only": True,
        "scientific_promotion_allowed": False,
        "thresholds_frozen": False,
        "case_index": request.case_index,
        "case_binding": request.case_binding,
        "protocol_payload_sha256": CALIBRATION_DESIGN_PAYLOAD_SHA256,
        "seed_snapshot_sha256": SEED_SNAPSHOT_SHA256,
        "readiness_binding": request.readiness_binding,
        "managed_ledger_genesis_sha256": request.managed_ledger_genesis_sha256,
        "explicit_acknowledgement": request.explicit_acknowledgement,
    }
    return canonical_sha256(body)


def build_calibration_case_request(
    case_index: int,
    bundle: ValidatedReadinessBundle,
    *,
    managed_ledger_directory: Path,
    explicit_acknowledgement: str,
    allow_exact_replay: bool = False,
) -> CalibrationCaseRequest:
    """Bind one frozen case; the acknowledgement has no default by design."""

    _require(
        explicit_acknowledgement == EXECUTION_ACKNOWLEDGEMENT,
        "exact explicit calibration authorization acknowledgement is required",
    )
    design = _design()
    case = _case(design, case_index)
    _require(type(allow_exact_replay) is bool, "allow_exact_replay must be a strict boolean")
    inventory = snapshot_calibration_execution_inventory(managed_ledger_directory)
    readiness_binding = _readiness_binding(bundle)
    governance = _plain_dict(
        readiness_binding["execution_governance"],
        "execution governance",
    )
    _require(
        inventory["genesis_sha256"] == governance["genesis_sha256"],
        "managed ledger genesis differs from readiness",
    )
    return CalibrationCaseRequest(
        case_index=case_index,
        case_binding=case.to_payload(),
        readiness_binding=readiness_binding,
        managed_ledger_genesis_sha256=cast(str, inventory["genesis_sha256"]),
        allow_exact_replay=allow_exact_replay,
        explicit_acknowledgement=explicit_acknowledgement,
    )


def validate_calibration_case_request(
    payload: Mapping[str, object],
    bundle: ValidatedReadinessBundle,
) -> CalibrationCaseRequest:
    body = _validate_payload_digest(payload, "case request")
    _exact_keys(
        body,
        {
            "schema",
            "development_only",
            "scientific_promotion_allowed",
            "thresholds_frozen",
            "case_index",
            "case_binding",
            "protocol_payload_sha256",
            "seed_snapshot_sha256",
            "readiness_binding",
            "managed_ledger_genesis_sha256",
            "allow_exact_replay",
            "explicit_acknowledgement",
        },
        "case request",
    )
    _require(body["schema"] == CALIBRATION_CASE_REQUEST_SCHEMA, "case request schema differs")
    _require(body["development_only"] is True, "case request is not development-only")
    _require(body["scientific_promotion_allowed"] is False, "case request allows promotion")
    _require(body["thresholds_frozen"] is False, "case request claims frozen thresholds")
    _require(
        body["protocol_payload_sha256"] == CALIBRATION_DESIGN_PAYLOAD_SHA256,
        "case request protocol digest differs",
    )
    _require(
        body["seed_snapshot_sha256"] == SEED_SNAPSHOT_SHA256,
        "case request seed snapshot differs",
    )
    _require(
        body["explicit_acknowledgement"] == EXECUTION_ACKNOWLEDGEMENT,
        "case request lacks explicit authorization",
    )
    case_index = _strict_int(body["case_index"], "case request case_index")
    design = _design()
    case = _case(design, case_index)
    case_binding = _plain_dict(body["case_binding"], "case request case binding")
    _require(case_binding == case.to_payload(), "case request differs from frozen case")
    readiness_binding = _plain_dict(body["readiness_binding"], "request readiness binding")
    _require(readiness_binding == _readiness_binding(bundle), "request readiness binding differs")
    genesis_sha256 = body["managed_ledger_genesis_sha256"]
    _require(_is_sha256(genesis_sha256), "request ledger genesis digest is invalid")
    governance = _plain_dict(readiness_binding["execution_governance"], "execution governance")
    _require(genesis_sha256 == governance["genesis_sha256"], "request ledger genesis differs")
    allow_exact_replay = _strict_bool(body["allow_exact_replay"], "allow_exact_replay")
    return CalibrationCaseRequest(
        case_index=case_index,
        case_binding=case_binding,
        readiness_binding=readiness_binding,
        managed_ledger_genesis_sha256=cast(str, genesis_sha256),
        allow_exact_replay=allow_exact_replay,
        explicit_acknowledgement=EXECUTION_ACKNOWLEDGEMENT,
    )


def _require_pristine_execution_inventory(
    inventory: Mapping[str, object],
    *,
    expected_genesis_sha256: str,
) -> dict[str, object]:
    normalized = _plain_dict(inventory, "preflight execution inventory")
    _require(
        normalized.get("schema") == CALIBRATION_EXECUTION_INVENTORY_SCHEMA,
        "preflight inventory schema differs",
    )
    _require(
        normalized.get("genesis_sha256") == expected_genesis_sha256,
        "preflight inventory genesis differs",
    )
    _require(normalized.get("expected_case_count") == EXPECTED_CASES, "case count differs")
    for field in (
        "started_case_indices",
        "completed_case_indices",
        "finalized_case_indices",
        "learner_interrupted_case_indices",
        "post_audit_unfinalized_case_indices",
        "started_records",
        "completed_records",
        "finalized_records",
        "attempt_records",
    ):
        _require(normalized.get(field) == [], f"preflight inventory {field} is not empty")
    for field in (
        "started_record_count",
        "completed_record_count",
        "finalized_record_count",
        "managed_execution_attempt_count",
        "protected_started_record_count",
        "protected_completed_record_count",
    ):
        _require(normalized.get(field) == 0, f"preflight inventory {field} is not zero")
    _require(normalized.get("pristine") is True, "preflight ledger is not pristine")
    _require(_is_sha256(normalized.get("inventory_sha256")), "inventory digest is invalid")
    return normalized


def build_calibration_preflight_request(
    bundle: ValidatedReadinessBundle,
    *,
    managed_ledger_directory: Path,
    issue_process_local_authorizations: bool = False,
    explicit_acknowledgement: str | None = None,
) -> dict[str, object]:
    """Bind a full 240-case non-consuming ZIP preflight request."""

    _require(
        explicit_acknowledgement == PREFLIGHT_ACKNOWLEDGEMENT,
        "exact explicit calibration preflight acknowledgement is required",
    )
    _require(
        type(issue_process_local_authorizations) is bool,
        "issue_process_local_authorizations must be a strict boolean",
    )
    readiness = _readiness_binding(bundle)
    genesis = cast(
        str,
        _plain_dict(readiness["execution_governance"], "execution governance")["genesis_sha256"],
    )
    inventory = snapshot_calibration_execution_inventory(managed_ledger_directory)
    try:
        validated_inventory = require_valid_calibration_execution_inventory(
            inventory,
            managed_ledger_directory,
        )
    except RuntimeError as exc:
        raise CalibrationError(str(exc)) from exc
    _require_pristine_execution_inventory(
        validated_inventory,
        expected_genesis_sha256=genesis,
    )
    design = _design()
    case_bindings = [case.to_payload() for case in design.cases]
    return _payload_with_digest(
        {
            "schema": CALIBRATION_PREFLIGHT_REQUEST_SCHEMA,
            "development_only": True,
            "scientific_promotion_allowed": False,
            "learner_execution_authorized": False,
            "case_indices": list(range(EXPECTED_CASES)),
            "case_bindings_sha256": canonical_sha256(case_bindings),
            "protocol_payload_sha256": CALIBRATION_DESIGN_PAYLOAD_SHA256,
            "seed_snapshot_sha256": SEED_SNAPSHOT_SHA256,
            "readiness_binding": readiness,
            "managed_ledger_genesis_sha256": genesis,
            "pristine_inventory_sha256": validated_inventory["inventory_sha256"],
            "issue_process_local_authorizations": issue_process_local_authorizations,
            "explicit_acknowledgement": PREFLIGHT_ACKNOWLEDGEMENT,
        }
    )


def validate_calibration_preflight_request(
    payload: Mapping[str, object],
    bundle: ValidatedReadinessBundle,
) -> dict[str, object]:
    body = _validate_payload_digest(payload, "preflight request")
    _exact_keys(
        body,
        {
            "schema",
            "development_only",
            "scientific_promotion_allowed",
            "learner_execution_authorized",
            "case_indices",
            "case_bindings_sha256",
            "protocol_payload_sha256",
            "seed_snapshot_sha256",
            "readiness_binding",
            "managed_ledger_genesis_sha256",
            "pristine_inventory_sha256",
            "issue_process_local_authorizations",
            "explicit_acknowledgement",
        },
        "preflight request",
    )
    _require(body["schema"] == CALIBRATION_PREFLIGHT_REQUEST_SCHEMA, "preflight schema differs")
    _require(body["development_only"] is True, "preflight is not development-only")
    _require(body["scientific_promotion_allowed"] is False, "preflight permits promotion")
    _require(body["learner_execution_authorized"] is False, "preflight authorizes execution")
    _require(body["case_indices"] == list(range(EXPECTED_CASES)), "preflight cases differ")
    design = _design()
    _require(
        body["case_bindings_sha256"]
        == canonical_sha256([case.to_payload() for case in design.cases]),
        "preflight case bindings differ",
    )
    _require(
        body["protocol_payload_sha256"] == CALIBRATION_DESIGN_PAYLOAD_SHA256,
        "preflight protocol differs",
    )
    _require(body["seed_snapshot_sha256"] == SEED_SNAPSHOT_SHA256, "preflight seeds differ")
    readiness = _plain_dict(body["readiness_binding"], "preflight readiness")
    _require(readiness == _readiness_binding(bundle), "preflight readiness differs")
    genesis = _plain_dict(readiness["execution_governance"], "execution governance")[
        "genesis_sha256"
    ]
    _require(body["managed_ledger_genesis_sha256"] == genesis, "preflight genesis differs")
    _require(_is_sha256(body["pristine_inventory_sha256"]), "preflight inventory digest invalid")
    _strict_bool(
        body["issue_process_local_authorizations"],
        "issue_process_local_authorizations",
    )
    _require(
        body["explicit_acknowledgement"] == PREFLIGHT_ACKNOWLEDGEMENT,
        "preflight acknowledgement differs",
    )
    return dict(payload)


def _compact_audit(
    report: HiddenRegimeTraceAuditReport,
    *,
    lineage_valid: bool,
    lineage_mismatches: tuple[str, ...],
    lineage_commit_count: int,
    lineage_recurrence_count: int,
    lineage_aggregate_count: int,
) -> dict[str, object]:
    _require(report.schema == HIDDEN_REGIME_TRACE_AUDIT_REPORT_SCHEMA, "audit schema differs")
    _require(report.valid, "primitive trace audit failed")
    _require(not report.mismatches, "primitive trace audit has mismatches")
    _require(not report.unobserved_transition_fields, "primitive audit leaves fields unobserved")
    _require(report.expected_steps == EXPECTED_STEPS, "audit expected-step count differs")
    for name, value in (
        ("rows_checked", report.rows_checked),
        ("helper_transitions_checked", report.helper_transitions_checked),
        ("beneficiary_transitions_checked", report.beneficiary_transitions_checked),
        ("world_transitions_checked", report.world_transitions_checked),
    ):
        _require(value == EXPECTED_STEPS, f"audit {name} is incomplete")
    _require(lineage_valid and not lineage_mismatches, "independent lineage audit failed")
    full_report = report.to_dict()
    return {
        "trace_audit_schema": report.schema,
        "trace_audit_report_sha256": canonical_sha256(_encode_exact(full_report)),
        "valid": True,
        "expected_steps": report.expected_steps,
        "rows_checked": report.rows_checked,
        "helper_transitions_checked": report.helper_transitions_checked,
        "beneficiary_transitions_checked": report.beneficiary_transitions_checked,
        "world_transitions_checked": report.world_transitions_checked,
        "commit_lineages_checked": report.commit_lineages_checked,
        "recurrence_records_checked": report.recurrence_records_checked,
        "retention_aggregate_fields_checked": report.retention_aggregate_fields_checked,
        "summary_fields_checked": report.summary_fields_checked,
        "resource_fields_checked": report.resource_fields_checked,
        "mismatch_count": 0,
        "mismatches_sha256": canonical_sha256([]),
        "accepted_float32_contraction_count": len(report.accepted_float32_contractions),
        "accepted_float32_contractions_sha256": canonical_sha256(
            list(report.accepted_float32_contractions)
        ),
        "unobserved_transition_fields": [],
        "evidence_boundary_sha256": hashlib.sha256(
            report.evidence_boundary.encode("utf-8")
        ).hexdigest(),
        "lineage_oracle_schema": HIDDEN_REGIME_LINEAGE_ORACLE_SCHEMA,
        "lineage_oracle_valid": True,
        "lineage_oracle_mismatches_sha256": canonical_sha256([]),
        "lineage_commit_lineages_checked": lineage_commit_count,
        "lineage_recurrence_records_checked": lineage_recurrence_count,
        "lineage_aggregate_fields_checked": lineage_aggregate_count,
    }


def _validate_compact_lineage_sources(summary: Mapping[str, object]) -> None:
    commits = tuple(
        _plain_dict(item, "commit lineage")
        for item in _plain_list(summary.get("commit_generation_lineages"), "commit lineages")
    )
    indices = tuple(_strict_int(item.get("lineage_index"), "lineage_index") for item in commits)
    _require(indices == tuple(range(len(commits))), "commit lineage indices are not canonical")
    _require(
        summary.get("synchronized_commit_lineage_count") == len(commits),
        "synchronized commit lineage count mismatch",
    )
    qualified_count = sum(item.get("acquisition_qualified") is True for item in commits)
    for item in commits:
        accuracy = _number(
            item.get("committed_composed_greedy_accuracy"),
            "committed composed greedy accuracy",
        )
        expected_qualified = (
            accuracy == 1.0 and item.get("committed_composed_greedy_tie_free") is True
        )
        _require(
            item.get("acquisition_qualified") is expected_qualified,
            "commit acquisition qualification is not exact-and-tie-free",
        )
    _require(
        summary.get("acquisition_qualified_commit_lineage_count") == qualified_count,
        "qualified commit lineage count mismatch",
    )
    _require(
        summary.get("acquisition_unqualified_commit_lineage_count")
        == len(commits) - qualified_count,
        "unqualified commit lineage count mismatch",
    )
    ordered = tuple(
        sorted(
            commits,
            key=lambda item: (
                _strict_int(item.get("commit_step"), "commit_step"),
                _strict_int(item.get("slot"), "slot"),
                _strict_int(item.get("generation"), "generation"),
                _strict_int(item.get("lineage_index"), "lineage_index"),
            ),
        )
    )
    _require(commits == ordered, "commit lineage ledger order differs")
    by_index = {cast(int, item["lineage_index"]): item for item in commits}

    recurrences = tuple(
        _plain_dict(item, "recurrence")
        for item in _plain_list(summary.get("recurrence_retention"), "recurrences")
    )
    for recurrence in recurrences:
        regime = _strict_int(recurrence.get("regime_id"), "recurrence regime")
        entry_segment = _strict_int(recurrence.get("segment_index"), "recurrence segment")
        probes = tuple(
            _plain_dict(item, "recurrence lineage probe")
            for item in _plain_list(
                recurrence.get("prior_same_regime_lineages"),
                "prior same-regime lineages",
            )
        )
        expected_commits = tuple(
            item
            for item in commits
            if item.get("regime_id") == regime
            and _strict_int(item.get("commit_segment_index"), "commit segment") < entry_segment
        )
        probe_indices = tuple(
            _strict_int(item.get("lineage_index"), "probe lineage index") for item in probes
        )
        expected_indices = tuple(cast(int, item["lineage_index"]) for item in expected_commits)
        _require(probe_indices == expected_indices, "recurrence omits or reorders a prior lineage")
        for probe in probes:
            commit = by_index[_strict_int(probe["lineage_index"], "probe lineage index")]
            for probe_field, commit_field in (
                ("commit_step", "commit_step"),
                ("commit_segment_index", "commit_segment_index"),
                ("slot", "slot"),
                ("generation", "generation"),
                ("acquisition_qualified", "acquisition_qualified"),
            ):
                _require(
                    probe.get(probe_field) == commit.get(commit_field),
                    f"recurrence probe {probe_field} does not join its commit",
                )
        qualified = tuple(item for item in probes if item.get("acquisition_qualified") is True)
        surviving = tuple(
            item for item in qualified if item.get("synchronized_generation_survives") is True
        )
        latest = None if not qualified else qualified[-1]
        selected = None if not surviving else surviving[-1]
        _require(
            recurrence.get("prior_same_regime_lineage_count") == len(probes),
            "recurrence prior lineage count mismatch",
        )
        _require(
            recurrence.get("prior_qualified_lineage_count") == len(qualified),
            "recurrence qualified lineage count mismatch",
        )
        _require(
            recurrence.get("prior_unqualified_lineage_count") == len(probes) - len(qualified),
            "recurrence unqualified lineage count mismatch",
        )
        _require(
            recurrence.get("lineage_retention_applicable") is bool(qualified),
            "recurrence qualification applicability mismatch",
        )
        _require(
            recurrence.get("acquisition_coverage_failure") is (not bool(qualified)),
            "recurrence acquisition coverage failure mismatch",
        )
        _require(
            recurrence.get("latest_prior_qualified_lineage_index")
            == (None if latest is None else latest["lineage_index"]),
            "latest prior qualified lineage mismatch",
        )
        _require(
            recurrence.get("latest_prior_qualified_commit_step")
            == (None if latest is None else latest["commit_step"]),
            "latest prior qualified commit step mismatch",
        )
        _require(
            recurrence.get("latest_prior_qualified_survived")
            == (None if latest is None else latest.get("synchronized_generation_survives") is True),
            "latest prior qualified survival mismatch",
        )
        _require(
            recurrence.get("any_prior_qualified_survived")
            == (None if not qualified else bool(surviving)),
            "any-qualified survival mismatch",
        )
        _require(
            recurrence.get("surviving_qualified_lineage_count") == len(surviving),
            "surviving qualified lineage count mismatch",
        )
        _require(
            recurrence.get("selected_lineage_available") is bool(selected),
            "selected-lineage availability mismatch",
        )
        _require(
            recurrence.get("selected_lineage_index")
            == (None if selected is None else selected["lineage_index"]),
            "selected lineage is not the latest surviving qualified lineage",
        )


def extract_calibration_case_shard(
    run: HiddenRegimeRunResult,
    request: CalibrationCaseRequest,
    audit_report: HiddenRegimeTraceAuditReport,
    *,
    worker_provenance: Mapping[str, object],
    execution_record_binding: Mapping[str, object],
) -> dict[str, object]:
    """Audit and normalize one completed run before its primitive trace is discarded."""

    design = _design()
    case = _case(design, request.case_index)
    _require(request.case_binding == case.to_payload(), "request case binding differs")
    _require(run.condition == case.condition, "run condition differs from frozen case")
    _require(run.seed_pair == _seed_pair(case), "run seed pair differs from frozen case")
    expected_config = _build_case_config(design, case)
    _require(run.config == expected_config, "run configuration differs from frozen case")
    _require(run.summary.num_steps == EXPECTED_STEPS, "run summary step count differs")
    _require(run.resource.resource_constant, "learner resource changed within case")
    _require(run.resource.resource_matched, "learner resource differs from frozen budget")
    _require(run.resource.final_state_bytes == 552, "dyad resource byte count differs")

    lineage = validate_hidden_regime_lineage_summary(run.trace, run.config, run.summary)
    compact_audit = _compact_audit(
        audit_report,
        lineage_valid=lineage.valid,
        lineage_mismatches=lineage.mismatches,
        lineage_commit_count=lineage.commit_lineages_checked,
        lineage_recurrence_count=lineage.recurrence_records_checked,
        lineage_aggregate_count=lineage.aggregate_fields_checked,
    )
    summary = _plain_dict(_decode_exact(_encode_exact(run.summary.to_dict())), "summary")
    _validate_full_summary_shape(summary)
    compact_audit["audited_summary_sha256"] = canonical_sha256(summary)
    _validate_recurrence_population(design, case, summary)
    _validate_compact_lineage_sources(summary)
    observations = _metric_observations(design, summary)
    _validate_summary_metric_crosschecks(design, summary, observations)
    config_payload = _plain_dict(_encode_exact(run.config.to_dict()), "config")
    resource_payload = _plain_dict(_encode_exact(run.resource.to_dict()), "resource")
    summary_sha256 = canonical_sha256(summary)
    resource_sha256 = canonical_sha256(resource_payload)
    _require(
        canonical_sha256(config_payload) == calibration_execution_configuration_sha256(run.config),
        "configuration digest differs from managed start encoding",
    )
    _require(
        summary_sha256 == calibration_execution_summary_sha256(run.summary),
        "summary digest differs from managed completion encoding",
    )
    _require(
        resource_sha256 == calibration_execution_resource_sha256(run.resource),
        "resource digest differs from managed completion encoding",
    )
    trace_digest = _trace_sha256(run.trace)
    runtime_binding = cast(Any, _runtime_binding(design, case.condition)).to_payload()
    manifest_binding = cast(Any, _manifest_binding(design, case.manifest_name)).to_payload()
    recurrence_binding = cast(Any, _recurrence_binding(design, case.manifest_name)).to_payload()
    case_request_binding_sha256 = calibration_case_request_binding_sha256(request)
    provenance = _plain_dict(_encode_exact(dict(worker_provenance)), "worker provenance")
    _require(
        provenance.get("source_archive_sha256")
        == request.readiness_binding["source_archive_sha256"],
        "worker provenance source ZIP differs from request",
    )
    _require(
        provenance.get("source_manifest_sha256")
        == request.readiness_binding["source_manifest_sha256"],
        "worker provenance source manifest differs from request",
    )
    _require(provenance.get("loader") == "zipimport.zipimporter", "worker loader differs")
    _require(provenance.get("source_zip_first") is True, "worker source ZIP was not first")
    _require(
        provenance.get("mutable_project_path_count") == 0,
        "worker retained a mutable project path",
    )
    _require(
        provenance.get("fresh_empty_working_directory") is True,
        "worker cwd was not fresh and empty",
    )
    _require(_is_sha256(provenance.get("project_modules_sha256")), "module digest invalid")
    execution_record = _plain_dict(
        _encode_exact(dict(execution_record_binding)),
        "execution record binding",
    )
    _exact_keys(
        execution_record,
        {
            "case_index",
            "genesis_sha256",
            "started_record_sha256",
            "completed_record_sha256",
            "summary_sha256",
            "resource_sha256",
            "primitive_trace_sha256",
            "final_state_sha256",
            "outcome_sha256",
            "managed_execution_attempt_count",
            "attempt_records_sha256",
            "zip_provenance_binding_sha256",
            "zip_provenance_attestation_sha256",
        },
        "execution record binding",
    )
    _require(execution_record["case_index"] == case.case_index, "execution case index differs")
    governance = _plain_dict(
        request.readiness_binding["execution_governance"],
        "execution governance",
    )
    _require(
        execution_record["genesis_sha256"] == governance["genesis_sha256"],
        "execution genesis differs from readiness",
    )
    for field in (
        "genesis_sha256",
        "started_record_sha256",
        "completed_record_sha256",
        "summary_sha256",
        "resource_sha256",
        "primitive_trace_sha256",
        "outcome_sha256",
        "attempt_records_sha256",
        "zip_provenance_binding_sha256",
        "zip_provenance_attestation_sha256",
    ):
        _require(_is_sha256(execution_record[field]), f"execution {field} is invalid")
    _strict_int(
        execution_record["managed_execution_attempt_count"],
        "managed execution attempt count",
        minimum=1,
    )
    _require(
        execution_record["summary_sha256"] == summary_sha256,
        "summary differs from managed completion",
    )
    _require(
        execution_record["resource_sha256"] == resource_sha256,
        "resource differs from managed completion",
    )
    _require(
        execution_record["primitive_trace_sha256"] == trace_digest,
        "primitive trace differs from managed completion",
    )
    body: dict[str, object] = {
        "schema": CALIBRATION_CASE_SHARD_SCHEMA,
        "development_only": True,
        "scientific_promotion_allowed": False,
        "claim_accepted": False,
        "thresholds_frozen": False,
        "promotion_artifact": False,
        "case": case.to_payload(),
        "condition_runtime_binding": runtime_binding,
        "manifest_binding": manifest_binding,
        "recurrence_binding": recurrence_binding,
        "protocol_payload_sha256": CALIBRATION_DESIGN_PAYLOAD_SHA256,
        "seed_snapshot_sha256": SEED_SNAPSHOT_SHA256,
        "case_request_binding_sha256": case_request_binding_sha256,
        "readiness_binding": request.readiness_binding,
        "worker_provenance": provenance,
        "execution_record_binding": execution_record,
        "runtime_schemas": {
            "development": HIDDEN_REGIME_DEVELOPMENT_SCHEMA,
            "primitive_trace": HIDDEN_REGIME_TRACE_SCHEMA,
            "trace_audit": HIDDEN_REGIME_TRACE_AUDIT_REPORT_SCHEMA,
            "lineage_oracle": HIDDEN_REGIME_LINEAGE_ORACLE_SCHEMA,
            "summary_oracle": HIDDEN_REGIME_SUMMARY_ORACLE_SCHEMA,
        },
        "execution_digest_schemas": {
            "summary": CALIBRATION_EXECUTION_SUMMARY_DIGEST_SCHEMA,
            "resource": CALIBRATION_EXECUTION_RESOURCE_DIGEST_SCHEMA,
            "primitive_trace": CALIBRATION_EXECUTION_PRIMITIVE_TRACE_DIGEST_SCHEMA,
            "component_outcome": CALIBRATION_EXECUTION_OUTCOME_DIGEST_SCHEMA,
        },
        "executed_steps": EXPECTED_STEPS,
        "managed_execution_attempt_count": execution_record["managed_execution_attempt_count"],
        "unique_completed_outcome_count": 1,
        "configuration": config_payload,
        "configuration_sha256": canonical_sha256(config_payload),
        "resource": resource_payload,
        "resource_sha256": resource_sha256,
        "summary": summary,
        "summary_sha256": summary_sha256,
        "metric_observations": observations,
        "metric_observations_sha256": canonical_sha256(observations),
        "primitive_trace": {
            "schema": HIDDEN_REGIME_TRACE_SCHEMA,
            "digest_schema": PRIMITIVE_TRACE_DIGEST_SCHEMA,
            "sha256": trace_digest,
            "rows": EXPECTED_STEPS,
            "persisted": False,
            "discard_required_after_audit": True,
        },
        "audit": compact_audit,
    }
    return _payload_with_digest(body)


def _validate_audit_payload(audit: Mapping[str, object], summary: Mapping[str, object]) -> None:
    expected_keys = {
        "trace_audit_schema",
        "trace_audit_report_sha256",
        "valid",
        "expected_steps",
        "rows_checked",
        "helper_transitions_checked",
        "beneficiary_transitions_checked",
        "world_transitions_checked",
        "commit_lineages_checked",
        "recurrence_records_checked",
        "retention_aggregate_fields_checked",
        "summary_fields_checked",
        "resource_fields_checked",
        "mismatch_count",
        "mismatches_sha256",
        "accepted_float32_contraction_count",
        "accepted_float32_contractions_sha256",
        "unobserved_transition_fields",
        "evidence_boundary_sha256",
        "lineage_oracle_schema",
        "lineage_oracle_valid",
        "lineage_oracle_mismatches_sha256",
        "lineage_commit_lineages_checked",
        "lineage_recurrence_records_checked",
        "lineage_aggregate_fields_checked",
        "audited_summary_sha256",
    }
    _exact_keys(audit, expected_keys, "case audit")
    _require(audit["trace_audit_schema"] == HIDDEN_REGIME_TRACE_AUDIT_REPORT_SCHEMA, "audit schema")
    _require(audit["valid"] is True, "case audit is invalid")
    for field in (
        "expected_steps",
        "rows_checked",
        "helper_transitions_checked",
        "beneficiary_transitions_checked",
        "world_transitions_checked",
    ):
        _require(audit[field] == EXPECTED_STEPS, f"case audit {field} is incomplete")
    _require(audit["mismatch_count"] == 0, "case audit has mismatches")
    _require(audit["mismatches_sha256"] == canonical_sha256([]), "audit mismatch digest differs")
    _require(audit["unobserved_transition_fields"] == [], "case audit leaves fields unobserved")
    _strict_int(
        audit["accepted_float32_contraction_count"],
        "accepted float32 contraction count",
    )
    for field in (
        "trace_audit_report_sha256",
        "accepted_float32_contractions_sha256",
        "evidence_boundary_sha256",
        "lineage_oracle_mismatches_sha256",
    ):
        _require(_is_sha256(audit[field]), f"case audit {field} is invalid")
    _require(
        audit["evidence_boundary_sha256"]
        == hashlib.sha256(EVIDENCE_BOUNDARY.encode("utf-8")).hexdigest(),
        "case audit evidence boundary differs",
    )
    _require(
        audit["lineage_oracle_schema"] == HIDDEN_REGIME_LINEAGE_ORACLE_SCHEMA, "lineage schema"
    )
    _require(audit["lineage_oracle_valid"] is True, "lineage audit invalid")
    _require(
        audit["lineage_oracle_mismatches_sha256"] == canonical_sha256([]),
        "lineage mismatch digest differs",
    )
    commits = _plain_list(summary.get("commit_generation_lineages"), "commit lineages")
    recurrences = _plain_list(summary.get("recurrence_retention"), "recurrences")
    aggregate_fields = len(dataclasses.fields(RetentionAggregateSummary))
    for prefix in ("", "lineage_"):
        _require(
            audit[f"{prefix}commit_lineages_checked"] == len(commits),
            f"{prefix}commit audit count mismatch",
        )
        _require(
            audit[f"{prefix}recurrence_records_checked"] == len(recurrences),
            f"{prefix}recurrence audit count mismatch",
        )
    _require(
        audit["retention_aggregate_fields_checked"] == aggregate_fields,
        "trace audit aggregate-field count mismatch",
    )
    _require(
        audit["lineage_aggregate_fields_checked"] == aggregate_fields,
        "lineage audit aggregate-field count mismatch",
    )
    _require(
        audit["summary_fields_checked"] == len(dataclasses.fields(HiddenRegimeRunSummary)),
        "summary audit field count mismatch",
    )
    _require(
        audit["resource_fields_checked"] == len(dataclasses.fields(HiddenRegimeResourceReport)),
        "resource audit field count mismatch",
    )
    _require(
        audit["audited_summary_sha256"] == canonical_sha256(_encode_exact(dict(summary))),
        "audited summary digest mismatch",
    )


def validate_calibration_case_shard(
    payload: Mapping[str, object],
    *,
    expected_readiness_binding: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Strictly validate a compact shard and every reconstructible source metric."""

    body = _validate_payload_digest(payload, "case shard")
    expected_keys = {
        "schema",
        "development_only",
        "scientific_promotion_allowed",
        "claim_accepted",
        "thresholds_frozen",
        "promotion_artifact",
        "case",
        "condition_runtime_binding",
        "manifest_binding",
        "recurrence_binding",
        "protocol_payload_sha256",
        "seed_snapshot_sha256",
        "case_request_binding_sha256",
        "readiness_binding",
        "worker_provenance",
        "execution_record_binding",
        "runtime_schemas",
        "execution_digest_schemas",
        "executed_steps",
        "managed_execution_attempt_count",
        "unique_completed_outcome_count",
        "configuration",
        "configuration_sha256",
        "resource",
        "resource_sha256",
        "summary",
        "summary_sha256",
        "metric_observations",
        "metric_observations_sha256",
        "primitive_trace",
        "audit",
    }
    _exact_keys(body, expected_keys, "case shard")
    _require(body["schema"] == CALIBRATION_CASE_SHARD_SCHEMA, "case shard schema differs")
    _require(body["development_only"] is True, "case shard is not development-only")
    for field in (
        "scientific_promotion_allowed",
        "claim_accepted",
        "thresholds_frozen",
        "promotion_artifact",
    ):
        _require(body[field] is False, f"case shard {field} must be false")
    _require(
        body["protocol_payload_sha256"] == CALIBRATION_DESIGN_PAYLOAD_SHA256, "protocol digest"
    )
    _require(body["seed_snapshot_sha256"] == SEED_SNAPSHOT_SHA256, "seed digest")
    _require(
        _is_sha256(body["case_request_binding_sha256"]),
        "case request binding digest invalid",
    )
    readiness = _plain_dict(body["readiness_binding"], "shard readiness binding")
    _exact_keys(
        readiness,
        {
            "readiness_receipt_sha256",
            "source_archive_sha256",
            "source_manifest_sha256",
            "runtime_identity_sha256",
            "dependency_locks",
            "scipy_version",
            "execution_governance",
        },
        "shard readiness binding",
    )
    if expected_readiness_binding is not None:
        _require(readiness == dict(expected_readiness_binding), "mixed readiness provenance")
    for field in (
        "readiness_receipt_sha256",
        "source_archive_sha256",
        "source_manifest_sha256",
        "runtime_identity_sha256",
    ):
        _require(_is_sha256(readiness.get(field)), f"readiness {field} invalid")
    _require(readiness.get("scipy_version") == scipy_version, "SciPy version drift")
    locks = tuple(
        _plain_dict(item, "dependency lock")
        for item in _plain_list(readiness.get("dependency_locks"), "dependency locks")
    )
    _require(
        tuple(item.get("locator") for item in locks) == ("pyproject.toml", "uv.lock"),
        "dependency lock binding differs",
    )
    _require(
        all(
            item.get("role") == "dependency_lock" and _is_sha256(item.get("sha256"))
            for item in locks
        ),
        "dependency lock provenance is invalid",
    )
    governance = _plain_dict(readiness.get("execution_governance"), "execution governance")
    for field, expected in (
        ("protocol_payload_sha256", CALIBRATION_DESIGN_PAYLOAD_SHA256),
        ("seed_snapshot_sha256", SEED_SNAPSHOT_SHA256),
        ("source_archive_sha256", readiness["source_archive_sha256"]),
        ("source_manifest_sha256", readiness["source_manifest_sha256"]),
        ("runtime_identity_sha256", readiness["runtime_identity_sha256"]),
    ):
        _require(governance.get(field) == expected, f"execution governance {field} mismatch")
    _require(_is_sha256(governance.get("genesis_sha256")), "governance genesis invalid")
    _require(
        governance.get("protected_execution_permitted") is False,
        "execution governance permits external evaluation",
    )
    provenance = _plain_dict(body["worker_provenance"], "worker provenance")
    _exact_keys(
        provenance,
        {
            "source_archive_sha256",
            "source_manifest_sha256",
            "loader",
            "source_zip_first",
            "mutable_project_path_count",
            "fresh_empty_working_directory",
            "no_site_startup",
            "prebootstrap_pth_hook_absent",
            "receipt_bound_runtime_prefix",
            "exact_receipt_bound_site_search_paths",
            "dont_write_bytecode",
            "command_line_pycache_prefix",
            "pycache_prefix_fresh_empty_nonsymlink",
            "pycache_prefix_outside_bound_paths",
            "project_module_count",
            "project_modules_sha256",
        },
        "worker provenance",
    )
    _require(
        provenance.get("source_archive_sha256") == readiness.get("source_archive_sha256"),
        "worker/source ZIP provenance mismatch",
    )
    _require(
        provenance.get("source_manifest_sha256") == readiness.get("source_manifest_sha256"),
        "worker/source manifest provenance mismatch",
    )
    _require(provenance.get("loader") == "zipimport.zipimporter", "worker loader mismatch")
    _require(provenance.get("source_zip_first") is True, "worker ZIP was not first")
    _require(provenance.get("mutable_project_path_count") == 0, "mutable source path present")
    _require(
        provenance.get("fresh_empty_working_directory") is True,
        "worker cwd provenance mismatch",
    )
    for field in (
        "no_site_startup",
        "prebootstrap_pth_hook_absent",
        "receipt_bound_runtime_prefix",
        "exact_receipt_bound_site_search_paths",
        "dont_write_bytecode",
        "command_line_pycache_prefix",
        "pycache_prefix_fresh_empty_nonsymlink",
        "pycache_prefix_outside_bound_paths",
    ):
        _require(provenance.get(field) is True, f"worker provenance {field} differs")
    _require(
        _strict_int(provenance.get("project_module_count"), "project module count", minimum=1) > 0,
        "worker loaded no project modules",
    )
    _require(_is_sha256(provenance.get("project_modules_sha256")), "module digest invalid")
    execution_record = _plain_dict(
        body["execution_record_binding"],
        "execution record binding",
    )
    _exact_keys(
        execution_record,
        {
            "case_index",
            "genesis_sha256",
            "started_record_sha256",
            "completed_record_sha256",
            "summary_sha256",
            "resource_sha256",
            "primitive_trace_sha256",
            "final_state_sha256",
            "outcome_sha256",
            "managed_execution_attempt_count",
            "attempt_records_sha256",
            "zip_provenance_binding_sha256",
            "zip_provenance_attestation_sha256",
        },
        "execution record binding",
    )
    design = _design()
    case_payload = _plain_dict(body["case"], "case binding")
    case_index = _strict_int(case_payload.get("case_index"), "case index")
    case = _case(design, case_index)
    _require(case_payload == case.to_payload(), "shard case differs from frozen ledger")
    expected_request_binding = calibration_case_request_binding_sha256(
        CalibrationCaseRequest(
            case_index=case_index,
            case_binding=case.to_payload(),
            readiness_binding=readiness,
            managed_ledger_genesis_sha256=cast(str, governance["genesis_sha256"]),
            allow_exact_replay=False,
            explicit_acknowledgement=EXECUTION_ACKNOWLEDGEMENT,
        )
    )
    _require(
        body["case_request_binding_sha256"] == expected_request_binding,
        "case request binding differs from replay-invariant request projection",
    )
    _require(execution_record["case_index"] == case_index, "execution record case differs")
    _require(
        execution_record["genesis_sha256"] == governance["genesis_sha256"],
        "execution record genesis differs",
    )
    for field in (
        "genesis_sha256",
        "started_record_sha256",
        "completed_record_sha256",
        "summary_sha256",
        "resource_sha256",
        "primitive_trace_sha256",
        "outcome_sha256",
        "attempt_records_sha256",
        "zip_provenance_binding_sha256",
        "zip_provenance_attestation_sha256",
    ):
        _require(_is_sha256(execution_record[field]), f"execution record {field} invalid")
    attempt_count = _strict_int(
        execution_record["managed_execution_attempt_count"],
        "managed execution attempt count",
        minimum=1,
    )
    _require(
        body["condition_runtime_binding"]
        == cast(Any, _runtime_binding(design, case.condition)).to_payload(),
        "condition runtime binding differs",
    )
    _require(
        body["manifest_binding"]
        == cast(Any, _manifest_binding(design, case.manifest_name)).to_payload(),
        "manifest binding differs",
    )
    _require(
        body["recurrence_binding"]
        == cast(Any, _recurrence_binding(design, case.manifest_name)).to_payload(),
        "recurrence binding differs",
    )
    schemas = _plain_dict(body["runtime_schemas"], "runtime schemas")
    _require(
        schemas
        == {
            "development": HIDDEN_REGIME_DEVELOPMENT_SCHEMA,
            "primitive_trace": HIDDEN_REGIME_TRACE_SCHEMA,
            "trace_audit": HIDDEN_REGIME_TRACE_AUDIT_REPORT_SCHEMA,
            "lineage_oracle": HIDDEN_REGIME_LINEAGE_ORACLE_SCHEMA,
            "summary_oracle": HIDDEN_REGIME_SUMMARY_ORACLE_SCHEMA,
        },
        "runtime schema binding differs",
    )
    digest_schemas = _plain_dict(body["execution_digest_schemas"], "execution digests")
    _require(
        digest_schemas
        == {
            "summary": CALIBRATION_EXECUTION_SUMMARY_DIGEST_SCHEMA,
            "resource": CALIBRATION_EXECUTION_RESOURCE_DIGEST_SCHEMA,
            "primitive_trace": CALIBRATION_EXECUTION_PRIMITIVE_TRACE_DIGEST_SCHEMA,
            "component_outcome": CALIBRATION_EXECUTION_OUTCOME_DIGEST_SCHEMA,
        },
        "execution digest schema binding differs",
    )
    _require(body["executed_steps"] == EXPECTED_STEPS, "case step count differs")
    _require(
        body["managed_execution_attempt_count"] == attempt_count,
        "case execution attempt count differs",
    )
    _require(body["unique_completed_outcome_count"] == 1, "completed outcome count differs")
    config = _plain_dict(body["configuration"], "configuration")
    _require(canonical_sha256(config) == body["configuration_sha256"], "config digest mismatch")
    expected_config = _plain_dict(
        _encode_exact(_build_case_config(design, case).to_dict()), "config"
    )
    _require(config == expected_config, "case configuration differs from frozen config")
    _require(
        body["configuration_sha256"]
        == calibration_execution_configuration_sha256(_build_case_config(design, case)),
        "configuration digest differs from managed start encoding",
    )
    resource = _plain_dict(body["resource"], "resource")
    _exact_keys(resource, _dataclass_field_names(HiddenRegimeResourceReport), "resource")
    _require(resource.get("resource_constant") is True, "resource is not constant")
    _require(resource.get("resource_matched") is True, "resource is not matched")
    _require(resource.get("final_state_bytes") == 552, "resource bytes differ")
    _require(canonical_sha256(resource) == body["resource_sha256"], "resource digest mismatch")
    _require(
        execution_record["resource_sha256"] == body["resource_sha256"],
        "resource differs from managed completion binding",
    )
    summary_encoded = _plain_dict(body["summary"], "summary")
    _require(canonical_sha256(summary_encoded) == body["summary_sha256"], "summary digest mismatch")
    _require(
        execution_record["summary_sha256"] == body["summary_sha256"],
        "summary differs from managed completion binding",
    )
    summary = _plain_dict(_decode_exact(summary_encoded), "decoded summary")
    _validate_full_summary_shape(summary)
    _require(summary.get("num_steps") == EXPECTED_STEPS, "summary step count differs")
    _validate_recurrence_population(design, case, summary)
    _validate_compact_lineage_sources(summary)
    expected_observations = _metric_observations(design, summary)
    observations = _plain_list(body["metric_observations"], "metric observations")
    _require(
        observations == expected_observations, "metric observations differ from direct sources"
    )
    _require(
        canonical_sha256(observations) == body["metric_observations_sha256"],
        "metric observation digest mismatch",
    )
    _validate_summary_metric_crosschecks(
        design,
        summary,
        tuple(_plain_dict(item, "metric observation") for item in observations),
    )
    trace = _plain_dict(body["primitive_trace"], "primitive trace binding")
    _require(
        trace
        == {
            "schema": HIDDEN_REGIME_TRACE_SCHEMA,
            "digest_schema": PRIMITIVE_TRACE_DIGEST_SCHEMA,
            "sha256": trace.get("sha256"),
            "rows": EXPECTED_STEPS,
            "persisted": False,
            "discard_required_after_audit": True,
        },
        "primitive trace discard binding differs",
    )
    _require(_is_sha256(trace.get("sha256")), "primitive trace digest invalid")
    _require(
        trace["digest_schema"] == CALIBRATION_EXECUTION_PRIMITIVE_TRACE_DIGEST_SCHEMA,
        "primitive trace digest schema differs from managed completion",
    )
    _require(
        execution_record["primitive_trace_sha256"] == trace["sha256"],
        "primitive trace differs from managed completion binding",
    )
    _validate_audit_payload(_plain_dict(body["audit"], "audit"), summary)
    return dict(payload)


def _write_new_immutable(
    parent: Path,
    name: str,
    raw: bytes,
    *,
    max_bytes: int,
    label: str,
) -> None:
    """Publish through an invisible anonymous inode and one atomic final link."""

    directory_fd, _ = _open_directory_without_symlinks(parent, label="publication parent")
    try:
        atomic_install_new_immutable(
            directory_fd,
            name,
            raw,
            max_bytes=max_bytes,
            label=label,
        )
    except HiddenRegimeExecutionGovernanceError as exc:
        raise CalibrationError(str(exc)) from exc
    finally:
        os.close(directory_fd)


def _shard_directory(publication_root: Path, readiness_receipt_sha256: str) -> Path:
    _require(_is_sha256(readiness_receipt_sha256), "readiness receipt digest is invalid")
    root_fd, root = _open_directory_without_symlinks(
        publication_root,
        label="shard publication root",
    )
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0)
    directory_flags |= getattr(os, "O_NOFOLLOW", 0)
    directory_fd: int | None = None
    try:
        try:
            os.mkdir(readiness_receipt_sha256, 0o700, dir_fd=root_fd)
            os.fsync(root_fd)
        except FileExistsError:
            pass
        try:
            directory_fd = os.open(
                readiness_receipt_sha256,
                directory_flags,
                dir_fd=root_fd,
            )
        except OSError as exc:
            if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
                raise CalibrationError("shard content-address directory is symlinked") from exc
            raise
        status = os.fstat(directory_fd)
        _require(
            stat.S_ISDIR(status.st_mode),
            "shard content-address directory is not a directory",
        )
    finally:
        if directory_fd is not None:
            os.close(directory_fd)
        os.close(root_fd)
    return root / readiness_receipt_sha256


def _directory_member_names(path: Path, *, label: str) -> tuple[str, ...]:
    directory_fd, _ = _open_directory_without_symlinks(path, label=label)
    try:
        return tuple(sorted(os.listdir(directory_fd)))
    finally:
        os.close(directory_fd)


def calibration_case_shard_path(
    publication_root: Path,
    *,
    readiness_receipt_sha256: str,
    case_index: int,
) -> Path:
    if type(case_index) is not int or not 0 <= case_index < EXPECTED_CASES:
        _fail("case_index must be a strict integer in [0, 240)")
    _require(_is_sha256(readiness_receipt_sha256), "readiness receipt digest is invalid")
    return publication_root.absolute() / readiness_receipt_sha256 / f"case-{case_index:03d}.json"


def validate_finalized_calibration_case_shard(
    payload: Mapping[str, object],
    *,
    expected_readiness_binding: Mapping[str, object],
    managed_ledger_directory: Path,
) -> dict[str, object]:
    """Require structural validity and an exact immutable post-audit finalization."""

    validated = validate_calibration_case_shard(
        payload,
        expected_readiness_binding=expected_readiness_binding,
    )
    body = _validate_payload_digest(validated, "finalized case shard")
    case_index = _strict_int(_plain_dict(body["case"], "case")["case_index"], "case index")
    try:
        recovered = load_finalized_calibration_case_shard(
            managed_ledger_directory,
            case_index,
        )
    except RuntimeError as exc:
        raise CalibrationError(str(exc)) from exc
    _require(
        canonical_json_bytes(recovered) == canonical_json_bytes(validated),
        "case shard differs from immutable post-audit finalization",
    )
    return validated


def publish_calibration_case_shard_new_only(
    publication_root: Path,
    shard: Mapping[str, object],
    *,
    expected_readiness_binding: Mapping[str, object],
    managed_ledger_directory: Path,
) -> Path:
    """Publish one canonical shard once; an existing duplicate must be byte-identical."""

    validated = validate_finalized_calibration_case_shard(
        shard,
        expected_readiness_binding=expected_readiness_binding,
        managed_ledger_directory=managed_ledger_directory,
    )
    body = _validate_payload_digest(validated, "case shard")
    case_index = _strict_int(_plain_dict(body["case"], "case")["case_index"], "case index")
    readiness = _plain_dict(body["readiness_binding"], "readiness binding")
    receipt_digest = cast(str, readiness["readiness_receipt_sha256"])
    path = calibration_case_shard_path(
        publication_root,
        readiness_receipt_sha256=receipt_digest,
        case_index=case_index,
    )
    _shard_directory(publication_root, receipt_digest)
    raw = canonical_json_bytes(validated)
    try:
        _write_new_immutable(
            path.parent,
            path.name,
            raw,
            max_bytes=_MAX_SHARD_BYTES,
            label="case shard",
        )
    except FileExistsError:
        existing_raw = _read_regular_file(path, max_bytes=_MAX_SHARD_BYTES, label="case shard")
        existing = _strict_json(existing_raw, "case shard")
        validate_finalized_calibration_case_shard(
            existing,
            expected_readiness_binding=expected_readiness_binding,
            managed_ledger_directory=managed_ledger_directory,
        )
        _require(existing_raw == raw, "duplicate case shard is not byte-identical")
    return path


def load_complete_calibration_case_shards(
    publication_root: Path,
    *,
    expected_readiness_binding: Mapping[str, object],
    managed_ledger_directory: Path,
) -> tuple[dict[str, object], ...]:
    """Load the exact 240 unique canonical shards and reject every extra member."""

    receipt_digest = expected_readiness_binding.get("readiness_receipt_sha256")
    _require(_is_sha256(receipt_digest), "expected readiness digest is invalid")
    directory = publication_root.absolute() / cast(str, receipt_digest)
    expected_names = tuple(f"case-{index:03d}.json" for index in range(EXPECTED_CASES))
    actual_names = _directory_member_names(directory, label="shard directory")
    _require(actual_names == expected_names, "shard directory is incomplete or has extra files")
    shards: list[dict[str, object]] = []
    for index, name in enumerate(expected_names):
        raw = _read_regular_file(
            directory / name,
            max_bytes=_MAX_SHARD_BYTES,
            label=f"case shard {index}",
        )
        payload = _strict_json(raw, f"case shard {index}")
        validated = validate_finalized_calibration_case_shard(
            payload,
            expected_readiness_binding=expected_readiness_binding,
            managed_ledger_directory=managed_ledger_directory,
        )
        body = _validate_payload_digest(validated, f"case shard {index}")
        _require(
            _plain_dict(body["case"], "case")["case_index"] == index,
            "case shard is stored under the wrong name",
        )
        shards.append(validated)
    return tuple(shards)


def _execution_record_binding(
    inventory: Mapping[str, object],
    case_index: int,
) -> dict[str, object]:
    started_matches = tuple(
        _plain_dict(item, "started inventory record")
        for item in _plain_list(inventory.get("started_records"), "started inventory records")
        if _plain_dict(item, "started inventory record").get("case_index") == case_index
    )
    completed_matches = tuple(
        _plain_dict(item, "completed inventory record")
        for item in _plain_list(
            inventory.get("completed_records"),
            "completed inventory records",
        )
        if _plain_dict(item, "completed inventory record").get("case_index") == case_index
    )
    _require(len(started_matches) == 1, "case has no unique managed start record")
    _require(len(completed_matches) == 1, "case has no unique managed completion record")
    started = started_matches[0]
    completed = completed_matches[0]
    _require(
        completed.get("started_record_sha256") == started.get("started_record_sha256"),
        "case completion does not join its start",
    )
    try:
        attempt_binding = calibration_case_attempt_binding(inventory, case_index)
    except RuntimeError as exc:
        raise CalibrationError(str(exc)) from exc
    zip_binding_sha256 = started.get("zip_provenance_binding_sha256")
    zip_attestation_sha256 = started.get("zip_provenance_attestation_sha256")
    _require(_is_sha256(zip_binding_sha256), "started ZIP provenance binding is invalid")
    _require(_is_sha256(zip_attestation_sha256), "started ZIP provenance attestation is invalid")
    return {
        "case_index": case_index,
        "genesis_sha256": inventory["genesis_sha256"],
        "started_record_sha256": started["started_record_sha256"],
        "completed_record_sha256": completed["completed_record_sha256"],
        "summary_sha256": completed["summary_sha256"],
        "resource_sha256": completed["resource_sha256"],
        "primitive_trace_sha256": completed["primitive_trace_sha256"],
        "final_state_sha256": completed["final_state_sha256"],
        "outcome_sha256": completed["outcome_sha256"],
        "managed_execution_attempt_count": attempt_binding["managed_execution_attempt_count"],
        "attempt_records_sha256": attempt_binding["attempt_records_sha256"],
        "zip_provenance_binding_sha256": zip_binding_sha256,
        "zip_provenance_attestation_sha256": zip_attestation_sha256,
    }


def _decode_worker_request(value: str) -> dict[str, object]:
    try:
        raw = base64.b64decode(value.encode("ascii"), validate=True)
    except (UnicodeEncodeError, ValueError) as exc:
        raise CalibrationError("worker request is not strict base64") from exc
    _require(len(raw) <= _MAX_RECEIPT_BYTES, "worker request exceeds maximum size")
    return _strict_json(raw, "worker request")


def _preflight_case_binding_rows(
    request_payload: Mapping[str, object],
    bundle: ValidatedReadinessBundle,
) -> tuple[dict[str, object], ...]:
    request_body = _validate_payload_digest(request_payload, "preflight request")
    design = _design()
    readiness = _plain_dict(request_body["readiness_binding"], "preflight readiness")
    genesis = cast(str, request_body["managed_ledger_genesis_sha256"])
    rows: list[dict[str, object]] = []
    for case in design.cases:
        config = _build_case_config(design, case)
        seed_pair = _seed_pair(case)
        case_request = CalibrationCaseRequest(
            case_index=case.case_index,
            case_binding=case.to_payload(),
            readiness_binding=readiness,
            managed_ledger_genesis_sha256=genesis,
            allow_exact_replay=False,
            explicit_acknowledgement=EXECUTION_ACKNOWLEDGEMENT,
        )
        case_request_payload = case_request.to_payload()
        validate_calibration_case_request(case_request_payload, bundle)
        configuration_payload = _plain_dict(_encode_exact(config.to_dict()), "configuration")
        configuration_sha256 = canonical_sha256(configuration_payload)
        _require(
            configuration_sha256 == calibration_execution_configuration_sha256(config),
            "preflight configuration digest differs from managed execution",
        )
        rows.append(
            {
                "case_index": case.case_index,
                "case_binding_sha256": canonical_sha256(case.to_payload()),
                "condition_runtime_binding_sha256": canonical_sha256(
                    cast(Any, _runtime_binding(design, case.condition)).to_payload()
                ),
                "manifest_binding_sha256": canonical_sha256(
                    cast(Any, _manifest_binding(design, case.manifest_name)).to_payload()
                ),
                "recurrence_binding_sha256": canonical_sha256(
                    cast(Any, _recurrence_binding(design, case.manifest_name)).to_payload()
                ),
                "configuration_sha256": configuration_sha256,
                "seed_pair_binding_sha256": canonical_sha256(seed_pair.to_dict()),
                "case_request_binding_sha256": calibration_case_request_binding_sha256(
                    case_request
                ),
                "case_request_payload_sha256": case_request_payload["payload_sha256"],
            }
        )
    return tuple(rows)


def _validate_preflight_worker_provenance(
    provenance: Mapping[str, object],
    bundle: ValidatedReadinessBundle,
) -> dict[str, object]:
    normalized = _plain_dict(provenance, "preflight worker provenance")
    _exact_keys(
        normalized,
        {
            "source_archive_sha256",
            "source_manifest_sha256",
            "loader",
            "source_zip_first",
            "mutable_project_path_count",
            "fresh_empty_working_directory",
            "no_site_startup",
            "prebootstrap_pth_hook_absent",
            "receipt_bound_runtime_prefix",
            "exact_receipt_bound_site_search_paths",
            "dont_write_bytecode",
            "command_line_pycache_prefix",
            "pycache_prefix_fresh_empty_nonsymlink",
            "pycache_prefix_outside_bound_paths",
            "project_module_count",
            "project_modules_sha256",
        },
        "preflight worker provenance",
    )
    _require(
        normalized["source_archive_sha256"] == bundle.source_archive_sha256,
        "preflight source ZIP differs",
    )
    _require(
        normalized["source_manifest_sha256"] == bundle.source_manifest_sha256,
        "preflight source manifest differs",
    )
    _require(normalized["loader"] == "zipimport.zipimporter", "preflight loader differs")
    _require(normalized["source_zip_first"] is True, "preflight source ZIP was not first")
    _require(normalized["mutable_project_path_count"] == 0, "preflight mutable source present")
    _require(
        normalized["fresh_empty_working_directory"] is True,
        "preflight working directory was not empty",
    )
    for field in (
        "no_site_startup",
        "prebootstrap_pth_hook_absent",
        "receipt_bound_runtime_prefix",
        "exact_receipt_bound_site_search_paths",
        "dont_write_bytecode",
        "command_line_pycache_prefix",
        "pycache_prefix_fresh_empty_nonsymlink",
        "pycache_prefix_outside_bound_paths",
    ):
        _require(normalized[field] is True, f"preflight worker {field} differs")
    _strict_int(normalized["project_module_count"], "project module count", minimum=1)
    _require(_is_sha256(normalized["project_modules_sha256"]), "module digest invalid")
    return normalized


def _validate_aggregate_provenance_bindings(
    aggregate_body: Mapping[str, object],
    bundle: ValidatedReadinessBundle,
) -> None:
    """Bind persisted aggregation provenance to the exact successful ZIP worker."""

    certification_binding = _validate_aggregation_readiness_certification_binding(
        _plain_dict(
            aggregate_body.get("aggregation_readiness_certification_binding"),
            "aggregation readiness certification binding",
        )
    )
    _require(
        certification_binding == _aggregation_readiness_certification_binding(bundle),
        "aggregation readiness certification binding differs from receipt",
    )
    _require(
        aggregate_body.get("aggregation_readiness_certification_binding_sha256")
        == canonical_sha256(certification_binding),
        "aggregation readiness certification digest differs",
    )

    worker = _validate_preflight_worker_provenance(
        _plain_dict(
            aggregate_body.get("aggregation_worker_provenance"),
            "aggregation worker provenance",
        ),
        bundle,
    )
    _require(
        aggregate_body.get("aggregation_worker_provenance_sha256")
        == canonical_sha256(worker),
        "aggregation worker provenance digest differs",
    )
    attestation = _plain_dict(
        aggregate_body.get("aggregation_zip_provenance_attestation"),
        "aggregation ZIP provenance attestation",
    )
    _require(
        aggregate_body.get("aggregation_zip_provenance_attestation_sha256")
        == canonical_sha256(attestation),
        "aggregation ZIP provenance attestation binding differs",
    )
    attestation_body = dict(attestation)
    attestation_sha256 = attestation_body.pop("zip_provenance_attestation_sha256", None)
    _require(
        _is_sha256(attestation_sha256)
        and canonical_sha256(attestation_body) == attestation_sha256,
        "aggregation ZIP provenance attestation digest differs",
    )
    _require(
        set(attestation_body)
        == {"schema", "binding", "environment", "zip_provenance_policy"},
        "aggregation ZIP provenance attestation fields differ",
    )
    _require(
        attestation_body["schema"] == CALIBRATION_ZIP_PROVENANCE_SCHEMA,
        "aggregation ZIP provenance schema differs",
    )
    _require(
        attestation_body["zip_provenance_policy"] == ZIP_PROVENANCE_POLICY,
        "aggregation ZIP provenance policy differs",
    )
    binding = _plain_dict(attestation_body["binding"], "aggregation ZIP provenance binding")
    binding_body = dict(binding)
    binding_sha256 = binding_body.pop("zip_provenance_binding_sha256", None)
    _require(
        _is_sha256(binding_sha256) and canonical_sha256(binding_body) == binding_sha256,
        "aggregation ZIP provenance binding digest differs",
    )
    _require(
        (
            binding_body.get("readiness_receipt_sha256"),
            binding_body.get("source_archive_sha256"),
            binding_body.get("source_manifest_sha256"),
            binding_body.get("runtime_identity_sha256"),
        )
        == (
            bundle.receipt_sha256,
            bundle.source_archive_sha256,
            bundle.source_manifest_sha256,
            bundle.runtime_identity_sha256,
        ),
        "aggregation ZIP provenance binding differs from readiness",
    )
    environment = _plain_dict(
        attestation_body["environment"],
        "aggregation ZIP provenance environment",
    )
    _require(
        environment.get("source_archive_locator") == ZIP_PROVENANCE_SOURCE_ARCHIVE_LOCATOR,
        "aggregation ZIP provenance source archive locator differs",
    )
    _require(
        "source_archive_path" not in environment,
        "aggregation ZIP provenance leaks a physical source archive path",
    )
    _require(
        _is_sha256(environment.get("canonical_runtime_search_paths_sha256")),
        "aggregation ZIP canonical runtime search-path digest differs",
    )
    _require(
        environment.get("project_modules_sha256") == worker["project_modules_sha256"],
        "aggregation ZIP provenance module inventory differs",
    )


def _worker_preflight(
    *,
    readiness_directory: Path,
    ledger_directory: Path,
    request_payload: Mapping[str, object],
) -> dict[str, object]:
    """Validate all 240 frozen bindings without beginning learner execution."""

    bundle = load_validated_readiness_bundle(
        readiness_directory,
        recheck_current=False,
        recheck_runtime=False,
    )
    readiness_body = _plain_dict(bundle.payload.get("body"), "readiness body")
    expected_runtime_execution_identity = runtime_execution_identity_from_receipt(
        readiness_body.get("runtime_identity")
    )
    _require(
        build_runtime_execution_identity() == expected_runtime_execution_identity,
        "worker process runtime execution identity differs before preflight",
    )
    validated_request = validate_calibration_preflight_request(request_payload, bundle)
    request_body = _validate_payload_digest(validated_request, "preflight request")
    genesis = cast(str, request_body["managed_ledger_genesis_sha256"])
    inventory_before = snapshot_calibration_execution_inventory(ledger_directory)
    try:
        inventory_before = require_valid_calibration_execution_inventory(
            inventory_before,
            ledger_directory,
        )
    except RuntimeError as exc:
        raise CalibrationError(str(exc)) from exc
    _require_pristine_execution_inventory(
        inventory_before,
        expected_genesis_sha256=genesis,
    )
    _require(
        inventory_before["inventory_sha256"] == request_body["pristine_inventory_sha256"],
        "preflight inventory changed after request creation",
    )
    archive_path = readiness_directory.absolute() / "source.zip"
    provenance = _zip_worker_provenance(archive_path, bundle)
    source_archive = _read_regular_file(
        archive_path,
        max_bytes=_MAX_SOURCE_ZIP_BYTES,
        label="preflight source ZIP",
    )
    zip_provenance_capability = attest_calibration_zip_provenance(
        readiness_bundle=bundle,
        readiness_source_archive=source_archive,
        source_archive_path=archive_path,
    )
    rows = _preflight_case_binding_rows(validated_request, bundle)
    issue_authorizations = cast(bool, request_body["issue_process_local_authorizations"])
    if issue_authorizations:
        design = _design()
        for case, row in zip(design.cases, rows, strict=True):
            authorization = issue_calibration_execution_authorization(
                ledger_directory=ledger_directory,
                readiness_bundle=bundle,
                readiness_source_archive=source_archive,
                zip_provenance_capability=zip_provenance_capability,
                case_index=case.case_index,
                condition=case.condition,
                seed_pair=_seed_pair(case),
                config=_build_case_config(design, case),
                case_request_binding_sha256=cast(
                    str,
                    row["case_request_binding_sha256"],
                ),
                attempt_request_payload_sha256=cast(
                    str,
                    row["case_request_payload_sha256"],
                ),
                explicit_acknowledgement=EXECUTION_AUTHORIZATION_ACKNOWLEDGEMENT,
                allow_exact_replay=False,
            )
            del authorization
    inventory_after = snapshot_calibration_execution_inventory(ledger_directory)
    try:
        inventory_after = require_valid_calibration_execution_inventory(
            inventory_after,
            ledger_directory,
        )
    except RuntimeError as exc:
        raise CalibrationError(str(exc)) from exc
    _require(inventory_after == inventory_before, "preflight mutated the managed ledger")
    _require(
        build_runtime_execution_identity() == expected_runtime_execution_identity,
        "worker process runtime execution identity drifted during preflight",
    )
    rows_payload = list(rows)
    return _payload_with_digest(
        {
            "schema": CALIBRATION_PREFLIGHT_REPORT_SCHEMA,
            "development_only": True,
            "scientific_promotion_allowed": False,
            "report_is_execution_authorization": False,
            "outcome_observed": False,
            "learner_execution_called": False,
            "managed_start_published": False,
            "managed_completion_published": False,
            "raw_trace_observed_or_persisted": False,
            "request_payload_sha256": validated_request["payload_sha256"],
            "readiness_receipt_sha256": bundle.receipt_sha256,
            "managed_ledger_genesis_sha256": genesis,
            "case_count": EXPECTED_CASES,
            "case_indices": list(range(EXPECTED_CASES)),
            "per_case_binding_digests": rows_payload,
            "per_case_binding_digests_sha256": canonical_sha256(rows_payload),
            "process_local_authorizations_issued_and_discarded": issue_authorizations,
            "process_local_authorization_count": EXPECTED_CASES if issue_authorizations else 0,
            "authorization_material_serialized": False,
            "inventory_before_sha256": inventory_before["inventory_sha256"],
            "inventory_after_sha256": inventory_after["inventory_sha256"],
            "inventory_byte_equal_before_after": True,
            "worker_provenance": provenance,
        }
    )


def validate_calibration_preflight_report(
    payload: Mapping[str, object],
    request_payload: Mapping[str, object],
    bundle: ValidatedReadinessBundle,
) -> dict[str, object]:
    """Recompute every non-outcome binding in a ZIP preflight report."""

    validated_request = validate_calibration_preflight_request(request_payload, bundle)
    request_body = _validate_payload_digest(validated_request, "preflight request")
    body = _validate_payload_digest(payload, "preflight report")
    _exact_keys(
        body,
        {
            "schema",
            "development_only",
            "scientific_promotion_allowed",
            "report_is_execution_authorization",
            "outcome_observed",
            "learner_execution_called",
            "managed_start_published",
            "managed_completion_published",
            "raw_trace_observed_or_persisted",
            "request_payload_sha256",
            "readiness_receipt_sha256",
            "managed_ledger_genesis_sha256",
            "case_count",
            "case_indices",
            "per_case_binding_digests",
            "per_case_binding_digests_sha256",
            "process_local_authorizations_issued_and_discarded",
            "process_local_authorization_count",
            "authorization_material_serialized",
            "inventory_before_sha256",
            "inventory_after_sha256",
            "inventory_byte_equal_before_after",
            "worker_provenance",
        },
        "preflight report",
    )
    _require(body["schema"] == CALIBRATION_PREFLIGHT_REPORT_SCHEMA, "report schema differs")
    _require(body["development_only"] is True, "report is not development-only")
    for field in (
        "scientific_promotion_allowed",
        "report_is_execution_authorization",
        "outcome_observed",
        "learner_execution_called",
        "managed_start_published",
        "managed_completion_published",
        "raw_trace_observed_or_persisted",
        "authorization_material_serialized",
    ):
        _require(body[field] is False, f"preflight report {field} must be false")
    _require(
        body["request_payload_sha256"] == validated_request["payload_sha256"],
        "report request digest differs",
    )
    _require(body["readiness_receipt_sha256"] == bundle.receipt_sha256, "report receipt differs")
    _require(
        body["managed_ledger_genesis_sha256"] == request_body["managed_ledger_genesis_sha256"],
        "report genesis differs",
    )
    _require(body["case_count"] == EXPECTED_CASES, "report case count differs")
    _require(body["case_indices"] == list(range(EXPECTED_CASES)), "report cases differ")
    expected_rows = list(_preflight_case_binding_rows(validated_request, bundle))
    _require(body["per_case_binding_digests"] == expected_rows, "report case bindings differ")
    _require(
        body["per_case_binding_digests_sha256"] == canonical_sha256(expected_rows),
        "report case-binding aggregate digest differs",
    )
    issued = cast(bool, request_body["issue_process_local_authorizations"])
    _require(
        body["process_local_authorizations_issued_and_discarded"] is issued,
        "report authorization issuance differs",
    )
    _require(
        body["process_local_authorization_count"] == (EXPECTED_CASES if issued else 0),
        "report authorization count differs",
    )
    _require(
        body["inventory_before_sha256"] == request_body["pristine_inventory_sha256"],
        "report initial inventory differs",
    )
    _require(
        body["inventory_after_sha256"] == body["inventory_before_sha256"],
        "report inventory changed",
    )
    _require(body["inventory_byte_equal_before_after"] is True, "report inventory not equal")
    _validate_preflight_worker_provenance(
        _plain_dict(body["worker_provenance"], "worker provenance"),
        bundle,
    )
    return dict(payload)


def _worker_case(
    *,
    readiness_directory: Path,
    ledger_directory: Path,
    request_payload: Mapping[str, object],
) -> dict[str, object]:
    """Execute inside the certified ZIP process; never call this from the checkout."""

    bundle = load_validated_readiness_bundle(
        readiness_directory,
        recheck_current=False,
        recheck_runtime=False,
    )
    readiness_body = _plain_dict(bundle.payload.get("body"), "readiness body")
    expected_runtime_execution_identity = runtime_execution_identity_from_receipt(
        readiness_body.get("runtime_identity")
    )
    _require(
        build_runtime_execution_identity() == expected_runtime_execution_identity,
        "worker process runtime execution identity differs before the calibration case",
    )
    request = validate_calibration_case_request(request_payload, bundle)
    inventory_before = snapshot_calibration_execution_inventory(ledger_directory)
    _require(
        inventory_before["genesis_sha256"] == request.managed_ledger_genesis_sha256,
        "worker managed ledger genesis differs from request",
    )
    archive_path = readiness_directory.absolute() / "source.zip"
    provenance = _zip_worker_provenance(archive_path, bundle)
    source_archive = _read_regular_file(
        archive_path,
        max_bytes=_MAX_SOURCE_ZIP_BYTES,
        label="worker source ZIP",
    )
    zip_provenance_capability = attest_calibration_zip_provenance(
        readiness_bundle=bundle,
        readiness_source_archive=source_archive,
        source_archive_path=archive_path,
    )
    design = _design()
    case = _case(design, request.case_index)
    seed_pair = _seed_pair(case)
    config = _build_case_config(design, case)
    authorization = issue_calibration_execution_authorization(
        ledger_directory=ledger_directory,
        readiness_bundle=bundle,
        readiness_source_archive=source_archive,
        zip_provenance_capability=zip_provenance_capability,
        case_index=case.case_index,
        condition=case.condition,
        seed_pair=seed_pair,
        config=config,
        case_request_binding_sha256=calibration_case_request_binding_sha256(request),
        attempt_request_payload_sha256=cast(str, request.to_payload()["payload_sha256"]),
        explicit_acknowledgement=EXECUTION_AUTHORIZATION_ACKNOWLEDGEMENT,
        allow_exact_replay=request.allow_exact_replay,
    )
    run = run_hidden_regime_condition(
        cast(Any, case.condition),
        seed_pair=seed_pair,
        config=config,
        execution_authorization=authorization,
    )
    inventory_after = snapshot_calibration_execution_inventory(ledger_directory)
    execution_record = _execution_record_binding(inventory_after, case.case_index)
    audit = audit_hidden_regime_run_result(run)
    shard = extract_calibration_case_shard(
        run,
        request,
        audit,
        worker_provenance=provenance,
        execution_record_binding=execution_record,
    )
    validated_shard = validate_calibration_case_shard(
        shard,
        expected_readiness_binding=request.readiness_binding,
    )
    _require(
        build_runtime_execution_identity() == expected_runtime_execution_identity,
        "worker process runtime execution identity drifted during the calibration case",
    )
    require_current_full_runtime_identity(readiness_body.get("runtime_identity"))
    finalize_calibration_case_shard(
        authorization,
        ledger_directory=ledger_directory,
        shard_payload=validated_shard,
        run_result=run,
    )
    recovered = load_finalized_calibration_case_shard(
        ledger_directory,
        case.case_index,
    )
    _require(
        canonical_json_bytes(recovered) == canonical_json_bytes(validated_shard),
        "finalized shard recovery differs before worker return",
    )
    return validated_shard


def _worker_aggregate(
    *,
    readiness_directory: Path,
    ledger_directory: Path,
    shard_publication_root: Path,
) -> dict[str, object]:
    """Compute and validate the completed aggregate inside the certified source ZIP."""

    bundle = load_validated_readiness_bundle(
        readiness_directory,
        recheck_current=False,
        recheck_runtime=False,
    )
    readiness_body = _plain_dict(bundle.payload.get("body"), "readiness body")
    expected_runtime_execution_identity = runtime_execution_identity_from_receipt(
        readiness_body.get("runtime_identity")
    )
    _require(
        build_runtime_execution_identity() == expected_runtime_execution_identity,
        "worker process runtime execution identity differs before aggregation",
    )
    readiness = _readiness_binding(bundle)
    readiness_certification_binding = _aggregation_readiness_certification_binding(bundle)
    archive_path = readiness_directory.absolute() / "source.zip"
    worker_provenance = _zip_worker_provenance(archive_path, bundle)
    source_archive = _read_regular_file(
        archive_path,
        max_bytes=_MAX_SOURCE_ZIP_BYTES,
        label="aggregate worker source ZIP",
    )
    zip_provenance_capability = attest_calibration_zip_provenance(
        readiness_bundle=bundle,
        readiness_source_archive=source_archive,
        source_archive_path=archive_path,
    )
    shards = load_complete_calibration_case_shards(
        shard_publication_root,
        expected_readiness_binding=readiness,
        managed_ledger_directory=ledger_directory,
    )
    inventory = snapshot_calibration_execution_inventory(ledger_directory)
    aggregate = aggregate_hidden_regime_factorial_calibration(
        shards,
        managed_ledger_snapshot=inventory,
        managed_ledger_directory=ledger_directory,
        aggregation_worker_provenance=worker_provenance,
        aggregation_zip_provenance_attestation=zip_provenance_capability.payload,
        aggregation_readiness_certification_binding=readiness_certification_binding,
    )
    validated = validate_calibration_aggregate(
        aggregate,
        shards,
        managed_ledger_snapshot=inventory,
        managed_ledger_directory=ledger_directory,
        aggregation_worker_provenance=worker_provenance,
        aggregation_zip_provenance_attestation=zip_provenance_capability.payload,
        aggregation_readiness_certification_binding=readiness_certification_binding,
    )
    _validate_aggregate_provenance_bindings(
        _validate_payload_digest(validated, "aggregate worker result"),
        bundle,
    )
    _require(
        build_runtime_execution_identity() == expected_runtime_execution_identity,
        "worker process runtime execution identity drifted during aggregation",
    )
    require_current_full_runtime_identity(readiness_body.get("runtime_identity"))
    return validated


def _threshold_freeze_receipt_body(
    payload: Mapping[str, object],
    *,
    label: str,
) -> dict[str, object]:
    """Validate the self-address and fail-closed decision shape of one receipt."""

    try:
        normalized = dict(payload)
        canonical_json_bytes(normalized)
    except (TypeError, ValueError) as exc:
        raise CalibrationError(f"{label} is not canonical JSON data") from exc
    digest = normalized.pop("receipt_payload_sha256", None)
    _require(_is_sha256(digest), f"{label} receipt payload digest is invalid")
    _require(canonical_sha256(normalized) == digest, f"{label} receipt payload digest differs")
    _require(
        normalized.get("receipt_schema") == THRESHOLD_FREEZE_RECEIPT_SCHEMA,
        f"{label} receipt schema differs",
    )
    _require(normalized.get("development_only") is True, f"{label} is not development-only")
    _require(normalized.get("claim_accepted") is False, f"{label} accepts a claim")
    _require(
        normalized.get("scientific_promotion_allowed") is False,
        f"{label} permits scientific promotion",
    )
    _require(normalized.get("amendments_allowed") is False, f"{label} permits amendments")
    _require(
        _is_sha256(normalized.get("calibration_outcomes_payload_sha256")),
        f"{label} aggregate binding is invalid",
    )
    _require(
        _is_sha256(normalized.get("readiness_receipt_sha256")),
        f"{label} readiness binding is invalid",
    )
    decision = normalized.get("decision_status")
    _require(
        decision in {THRESHOLD_FREEZE_DECISION_FROZEN, THRESHOLD_FREEZE_DECISION_REJECTION},
        f"{label} decision status differs",
    )
    frozen = _plain_list(normalized.get("frozen_thresholds"), f"{label} frozen thresholds")
    reasons = _plain_list(normalized.get("rejection_reasons"), f"{label} rejection reasons")
    if decision == THRESHOLD_FREEZE_DECISION_FROZEN:
        _require(normalized.get("thresholds_frozen") is True, f"{label} does not freeze")
        _require(bool(frozen), f"{label} has no frozen thresholds")
        _require(not reasons, f"{label} success contains rejection reasons")
    else:
        _require(normalized.get("thresholds_frozen") is False, f"{label} rejection freezes")
        _require(not frozen, f"{label} rejection partially freezes thresholds")
        _require(bool(reasons), f"{label} rejection has no reason")
    return normalized


def _successful_threshold_freeze_receipt_body(
    payload: Mapping[str, object],
    *,
    label: str,
) -> dict[str, object]:
    """Require the structural success capability needed for protected planning."""

    body = _threshold_freeze_receipt_body(payload, label=label)
    _require(
        body.get("decision_status") == THRESHOLD_FREEZE_DECISION_FROZEN
        and body.get("thresholds_frozen") is True,
        f"{label} is not a successful threshold freeze",
    )
    _require(body.get("rejection_reasons") == [], f"{label} contains rejection reasons")
    frozen_thresholds = _plain_list(
        body.get("frozen_thresholds"),
        f"{label} frozen thresholds",
    )
    _require(
        body.get("mandatory_statistical_endpoint_count")
        == MANDATORY_STATISTICAL_ENDPOINT_COUNT
        and len(frozen_thresholds) == MANDATORY_STATISTICAL_ENDPOINT_COUNT,
        f"{label} does not bind exactly 35 endpoints",
    )
    _require(
        body.get("mandatory_statistical_endpoint_identities_sha256")
        == MANDATORY_STATISTICAL_ENDPOINT_IDENTITIES_SHA256,
        f"{label} endpoint identity digest differs",
    )
    _require(
        body.get("mandatory_statistical_endpoint_ids_sha256")
        == MANDATORY_STATISTICAL_ENDPOINT_IDS_SHA256,
        f"{label} endpoint ID digest differs",
    )
    return body


def _threshold_freeze_exact_input_binding(
    calibration_aggregate: Mapping[str, object],
    shards: Sequence[Mapping[str, object]],
    managed_ledger_snapshot: Mapping[str, object],
) -> dict[str, object]:
    """Join the live immutable shards and ledger to the aggregate's exact identities."""

    aggregate_body = _validate_payload_digest(calibration_aggregate, "calibration aggregate")
    aggregate_payload_sha256 = calibration_aggregate.get("payload_sha256")
    _require(_is_sha256(aggregate_payload_sha256), "aggregate payload digest is invalid")
    embedded_inventory = _plain_dict(
        aggregate_body.get("managed_ledger_snapshot"),
        "aggregate managed ledger snapshot",
    )
    current_inventory = _plain_dict(managed_ledger_snapshot, "current managed ledger snapshot")
    _require(
        canonical_sha256(embedded_inventory)
        == aggregate_body.get("managed_ledger_snapshot_sha256"),
        "aggregate managed ledger snapshot digest differs",
    )
    _require(
        canonical_json_bytes(current_inventory) == canonical_json_bytes(embedded_inventory),
        "managed ledger changed after threshold worker recomputation",
    )
    inventory_sha256 = current_inventory.get("inventory_sha256")
    _require(_is_sha256(inventory_sha256), "managed ledger inventory digest is invalid")

    case_ledger = _plain_list(aggregate_body.get("case_ledger"), "aggregate case ledger")
    _require(
        canonical_sha256(case_ledger) == aggregate_body.get("case_ledger_sha256"),
        "aggregate case ledger digest differs",
    )
    expected_rows = [
        {
            "case_index": _strict_int(
                _plain_dict(item, "aggregate case ledger row").get("case_index"),
                "aggregate case index",
            ),
            "case_shard_payload_sha256": _plain_dict(
                item,
                "aggregate case ledger row",
            ).get("case_shard_payload_sha256"),
        }
        for item in case_ledger
    ]
    _require(
        len(expected_rows) == EXPECTED_CASES
        and [row["case_index"] for row in expected_rows] == list(range(EXPECTED_CASES)),
        "aggregate case ledger indices differ",
    )
    _require(
        all(_is_sha256(row["case_shard_payload_sha256"]) for row in expected_rows),
        "aggregate case shard digest is invalid",
    )
    actual_rows: list[dict[str, object]] = []
    for expected_index, shard in enumerate(shards):
        shard_body = _validate_payload_digest(shard, f"threshold input shard {expected_index}")
        case_index = _strict_int(
            _plain_dict(shard_body.get("case"), "threshold input shard case").get(
                "case_index"
            ),
            "threshold input shard case index",
        )
        actual_rows.append(
            {
                "case_index": case_index,
                "case_shard_payload_sha256": shard.get("payload_sha256"),
            }
        )
    _require(actual_rows == expected_rows, "case shards changed after threshold recomputation")
    body = {
        "schema": THRESHOLD_FREEZE_EXACT_INPUT_BINDING_SCHEMA,
        "calibration_aggregate_payload_sha256": aggregate_payload_sha256,
        "managed_ledger_inventory_sha256": inventory_sha256,
        "managed_ledger_snapshot_sha256": canonical_sha256(current_inventory),
        "case_ledger_sha256": aggregate_body["case_ledger_sha256"],
        "case_shard_payloads_sha256": canonical_sha256(actual_rows),
        "case_count": len(actual_rows),
    }
    return body


def _protected_plan_body(
    payload: Mapping[str, object],
    *,
    label: str,
) -> dict[str, object]:
    """Structurally validate certified plan bytes without re-deriving protected seeds."""

    body = _validate_payload_digest(payload, label)
    _exact_keys(
        body,
        {
            "schema",
            "status_time_scope",
            "plan_status",
            "use_partition",
            "scientific_evidence_eligible_if_validated",
            "scientific_promotion_allowed",
            "automatic_promotion_allowed",
            "amendments_allowed",
            "thresholds_frozen",
            "threshold_adjustment_permitted",
            "protected_namespace_derived",
            "protected_outcomes_observed",
            "learner_outcomes_executed",
            "learner_execution_authorized",
            "protected_execution_permitted",
            "execution_issuer_available",
            "protected_readiness_receipt_sha256",
            "protected_execution_ledger_genesis_sha256",
            "calibration_binding",
            "threshold_freeze_receipt_binding",
            "frozen_thresholds",
            "frozen_thresholds_sha256",
            "protected_seed_snapshot",
            "protected_seed_snapshot_sha256",
            "seed_disjointness_proof",
            "seed_disjointness_proof_sha256",
            "structural_manifest_order",
            "structural_manifest_order_sha256",
            "manifest_bindings",
            "manifest_bindings_sha256",
            "recurrence_eligibility_bindings",
            "recurrence_eligibility_bindings_sha256",
            "assignment_rule",
            "assignments",
            "assignments_sha256",
            "condition_order",
            "condition_order_sha256",
            "cases",
            "cases_sha256",
            "seed_pair_count",
            "condition_count",
            "matched_case_count",
            "manifest_seed_pair_counts",
            "manifest_case_counts",
            "condition_case_counts",
            "evaluation_contract",
            "evaluation_contract_sha256",
            "protected_decision_rule",
            "protected_decision_rule_sha256",
            "claim_scope",
            "limitations",
        },
        label,
    )
    _require(body["schema"] == PROTECTED_PLAN_SCHEMA, f"{label} schema differs")
    _require(body["plan_status"] == "preregistered_unexecuted", f"{label} status differs")
    _require(body["use_partition"] == PROTECTED_CANDIDATE_PARTITION, f"{label} partition differs")
    _require(
        body["scientific_evidence_eligible_if_validated"] is True,
        f"{label} evidence eligibility differs",
    )
    _require(body["thresholds_frozen"] is True, f"{label} thresholds are not frozen")
    _require(body["protected_namespace_derived"] is True, f"{label} namespace is not derived")
    for field in (
        "scientific_promotion_allowed",
        "automatic_promotion_allowed",
        "amendments_allowed",
        "threshold_adjustment_permitted",
        "protected_outcomes_observed",
        "learner_outcomes_executed",
        "learner_execution_authorized",
        "protected_execution_permitted",
        "execution_issuer_available",
    ):
        _require(body[field] is False, f"{label} {field} must be false")
    for field in (
        "protected_readiness_receipt_sha256",
        "protected_execution_ledger_genesis_sha256",
    ):
        _require(body[field] is None, f"{label} {field} must be absent")
    for value_field, digest_field in (
        ("frozen_thresholds", "frozen_thresholds_sha256"),
        ("protected_seed_snapshot", "protected_seed_snapshot_sha256"),
        ("seed_disjointness_proof", "seed_disjointness_proof_sha256"),
        ("structural_manifest_order", "structural_manifest_order_sha256"),
        ("manifest_bindings", "manifest_bindings_sha256"),
        (
            "recurrence_eligibility_bindings",
            "recurrence_eligibility_bindings_sha256",
        ),
        ("assignments", "assignments_sha256"),
        ("condition_order", "condition_order_sha256"),
        ("cases", "cases_sha256"),
        ("evaluation_contract", "evaluation_contract_sha256"),
        ("protected_decision_rule", "protected_decision_rule_sha256"),
    ):
        _require(_is_sha256(body[digest_field]), f"{label} {digest_field} is invalid")
        _require(
            canonical_sha256(body[value_field]) == body[digest_field],
            f"{label} {digest_field} differs",
        )
    _require(
        body["seed_pair_count"] == EXPECTED_SEED_PAIRS
        and body["condition_count"] == EXPECTED_CONDITIONS
        and body["matched_case_count"] == EXPECTED_CASES,
        f"{label} factorial counts differ",
    )
    assignments = _plain_list(body["assignments"], f"{label} assignments")
    cases = _plain_list(body["cases"], f"{label} cases")
    _require(len(assignments) == EXPECTED_SEED_PAIRS, f"{label} assignment count differs")
    _require(len(cases) == EXPECTED_CASES, f"{label} case count differs")
    _require(
        [
            _plain_dict(item, f"{label} assignment").get("seed_index")
            for item in assignments
        ]
        == list(range(EXPECTED_SEED_PAIRS)),
        f"{label} assignment order differs",
    )
    _require(
        [_plain_dict(item, f"{label} case").get("case_index") for item in cases]
        == list(range(EXPECTED_CASES)),
        f"{label} case order differs",
    )
    return body


def _validate_protected_plan_bindings(
    plan_body: Mapping[str, object],
    *,
    protected_plan: Mapping[str, object],
    threshold_receipt: Mapping[str, object],
    calibration_aggregate: Mapping[str, object],
) -> None:
    """Bind certified plan bytes to the exact receipt and aggregate without recomputation."""

    aggregate_digest = calibration_aggregate.get("payload_sha256")
    receipt_digest = threshold_receipt.get("receipt_payload_sha256")
    plan_digest = protected_plan.get("payload_sha256")
    for value, label in (
        (aggregate_digest, "plan aggregate digest"),
        (receipt_digest, "plan threshold receipt digest"),
        (plan_digest, "protected plan digest"),
    ):
        _require(_is_sha256(value), f"{label} is invalid")
    calibration_binding = _plain_dict(
        plan_body.get("calibration_binding"),
        "protected plan calibration binding",
    )
    receipt_binding = _plain_dict(
        plan_body.get("threshold_freeze_receipt_binding"),
        "protected plan threshold receipt binding",
    )
    _exact_keys(
        calibration_binding,
        {
            "protocol_payload_sha256",
            "calibration_seed_snapshot_sha256",
            "calibration_aggregate_schema",
            "calibration_aggregate_payload_sha256",
            "calibration_readiness_receipt_sha256",
            "calibration_gate_matrix_sha256",
            "calibration_source_closure_sha256",
            "calibration_source_archive_sha256",
            "calibration_environment_identity_sha256",
            "calibration_managed_ledger_snapshot_sha256",
            "calibration_managed_ledger_content_address",
            "calibration_execution_governance_genesis_sha256",
            "calibration_case_ledger_sha256",
            "aggregation_readiness_certification_binding_sha256",
            "mandatory_audit_summary_sha256",
        },
        "protected plan calibration binding",
    )
    _exact_keys(
        receipt_binding,
        {
            "receipt_schema",
            "receipt_payload_sha256",
            "decision_status",
            "mandatory_statistical_endpoint_count",
            "mandatory_statistical_endpoint_identities_sha256",
            "mandatory_statistical_endpoint_ids_sha256",
        },
        "protected plan threshold receipt binding",
    )
    _require(
        calibration_binding.get("calibration_aggregate_payload_sha256")
        == aggregate_digest,
        "protected plan binds another aggregate",
    )
    _require(
        receipt_binding.get("receipt_payload_sha256") == receipt_digest,
        "protected plan binds another threshold receipt",
    )
    _require(
        receipt_binding.get("receipt_schema") == THRESHOLD_FREEZE_RECEIPT_SCHEMA
        and receipt_binding.get("decision_status") == THRESHOLD_FREEZE_DECISION_FROZEN,
        "protected plan threshold receipt capability differs",
    )
    _require(
        receipt_binding.get("mandatory_statistical_endpoint_count")
        == MANDATORY_STATISTICAL_ENDPOINT_COUNT
        and receipt_binding.get("mandatory_statistical_endpoint_identities_sha256")
        == MANDATORY_STATISTICAL_ENDPOINT_IDENTITIES_SHA256
        and receipt_binding.get("mandatory_statistical_endpoint_ids_sha256")
        == MANDATORY_STATISTICAL_ENDPOINT_IDS_SHA256,
        "protected plan endpoint identity binding differs",
    )
    receipt_body = _successful_threshold_freeze_receipt_body(
        threshold_receipt,
        label="protected plan threshold-freeze receipt",
    )
    _require(
        receipt_body.get("calibration_outcomes_payload_sha256") == aggregate_digest,
        "threshold receipt binds another aggregate",
    )
    _require(
        plan_body.get("frozen_thresholds") == receipt_body.get("frozen_thresholds"),
        "protected plan frozen thresholds differ from threshold receipt",
    )
    for plan_field, receipt_field in (
        ("protocol_payload_sha256", "protocol_payload_sha256"),
        ("calibration_seed_snapshot_sha256", "seed_snapshot_sha256"),
        ("calibration_readiness_receipt_sha256", "readiness_receipt_sha256"),
        ("calibration_gate_matrix_sha256", "gate_matrix_sha256"),
        ("calibration_source_closure_sha256", "source_closure_sha256"),
        ("calibration_source_archive_sha256", "source_archive_sha256"),
        ("calibration_environment_identity_sha256", "environment_identity_sha256"),
        ("calibration_managed_ledger_snapshot_sha256", "managed_ledger_snapshot_sha256"),
        ("calibration_managed_ledger_content_address", "managed_ledger_content_address"),
        (
            "calibration_execution_governance_genesis_sha256",
            "execution_governance_genesis_sha256",
        ),
        ("calibration_case_ledger_sha256", "case_ledger_sha256"),
        (
            "aggregation_readiness_certification_binding_sha256",
            "aggregation_readiness_certification_binding_sha256",
        ),
        ("mandatory_audit_summary_sha256", "mandatory_audit_summary_sha256"),
    ):
        _require(
            calibration_binding.get(plan_field) == receipt_body.get(receipt_field),
            f"protected plan {plan_field} differs from threshold receipt",
        )
    _require(
        calibration_binding.get("calibration_aggregate_schema")
        == CALIBRATION_AGGREGATE_SCHEMA,
        "protected plan aggregate schema binding differs",
    )


def _threshold_freeze_worker_result(
    *,
    calibration_aggregate: Mapping[str, object],
    threshold_freeze_receipt: Mapping[str, object],
    readiness_receipt_sha256: str,
    threshold_exact_input_binding: Mapping[str, object],
    threshold_worker_readiness_certification_binding: Mapping[str, object],
    threshold_worker_provenance: Mapping[str, object],
    threshold_zip_provenance_attestation: Mapping[str, object],
) -> dict[str, object]:
    """Bind a threshold decision to the certified worker that produced it."""

    aggregate_body = _validate_payload_digest(calibration_aggregate, "calibration aggregate")
    aggregate_payload_sha256 = calibration_aggregate.get("payload_sha256")
    _require(_is_sha256(aggregate_payload_sha256), "aggregate payload digest is invalid")
    _require(_is_sha256(readiness_receipt_sha256), "readiness receipt digest is invalid")
    receipt = dict(threshold_freeze_receipt)
    receipt_body = _threshold_freeze_receipt_body(receipt, label="threshold-freeze receipt")
    _require(
        receipt_body["calibration_outcomes_payload_sha256"] == aggregate_payload_sha256,
        "threshold-freeze receipt binds another aggregate",
    )
    _require(
        receipt_body["readiness_receipt_sha256"] == readiness_receipt_sha256,
        "threshold-freeze receipt binds another readiness receipt",
    )
    readiness = _plain_dict(
        aggregate_body.get("readiness_binding"),
        "aggregate readiness binding",
    )
    _require(
        readiness.get("readiness_receipt_sha256") == readiness_receipt_sha256,
        "aggregate and threshold worker readiness bindings differ",
    )
    certification = _plain_dict(
        threshold_worker_readiness_certification_binding,
        "threshold worker readiness certification binding",
    )
    exact_input_binding = _plain_dict(
        threshold_exact_input_binding,
        "threshold exact input binding",
    )
    _require(
        exact_input_binding.get("schema") == THRESHOLD_FREEZE_EXACT_INPUT_BINDING_SCHEMA,
        "threshold exact input binding schema differs",
    )
    _require(
        exact_input_binding.get("calibration_aggregate_payload_sha256")
        == aggregate_payload_sha256,
        "threshold exact input binding references another aggregate",
    )
    worker = _plain_dict(threshold_worker_provenance, "threshold worker provenance")
    attestation = _plain_dict(
        threshold_zip_provenance_attestation,
        "threshold ZIP provenance attestation",
    )
    body = {
        "schema": THRESHOLD_FREEZE_WORKER_RESULT_SCHEMA,
        "development_only": True,
        "scientific_promotion_allowed": False,
        "claim_accepted": False,
        "promotion_artifact": False,
        "calibration_aggregate_payload_sha256": aggregate_payload_sha256,
        "readiness_receipt_sha256": readiness_receipt_sha256,
        "threshold_exact_input_binding": exact_input_binding,
        "threshold_exact_input_binding_sha256": canonical_sha256(exact_input_binding),
        "threshold_worker_readiness_certification_binding": certification,
        "threshold_worker_readiness_certification_binding_sha256": canonical_sha256(
            certification
        ),
        "threshold_worker_provenance": worker,
        "threshold_worker_provenance_sha256": canonical_sha256(worker),
        "threshold_zip_provenance_attestation": attestation,
        "threshold_zip_provenance_attestation_sha256": canonical_sha256(attestation),
        "threshold_freeze_receipt": receipt,
        "threshold_freeze_receipt_sha256": receipt["receipt_payload_sha256"],
    }
    return _payload_with_digest(body)


def _protected_plan_worker_result(
    *,
    calibration_aggregate: Mapping[str, object],
    threshold_freeze_receipt: Mapping[str, object],
    protected_plan: Mapping[str, object],
    readiness_receipt_sha256: str,
    exact_input_binding: Mapping[str, object],
    worker_readiness_certification_binding: Mapping[str, object],
    worker_provenance: Mapping[str, object],
    zip_provenance_attestation: Mapping[str, object],
) -> dict[str, object]:
    """Bind certified nonauthorizing plan bytes to every authoritative input."""

    aggregate_body = _validate_payload_digest(calibration_aggregate, "calibration aggregate")
    aggregate_digest = calibration_aggregate.get("payload_sha256")
    receipt = dict(threshold_freeze_receipt)
    receipt_body = _successful_threshold_freeze_receipt_body(
        receipt,
        label="protected-plan worker threshold receipt",
    )
    plan = dict(protected_plan)
    plan_body = _protected_plan_body(plan, label="protected-plan worker plan")
    _validate_protected_plan_bindings(
        plan_body,
        protected_plan=plan,
        threshold_receipt=receipt,
        calibration_aggregate=calibration_aggregate,
    )
    _require(_is_sha256(aggregate_digest), "protected-plan aggregate digest is invalid")
    receipt_digest = receipt.get("receipt_payload_sha256")
    plan_digest = plan.get("payload_sha256")
    _require(_is_sha256(receipt_digest), "protected-plan receipt digest is invalid")
    _require(_is_sha256(plan_digest), "protected-plan payload digest is invalid")
    _require(_is_sha256(readiness_receipt_sha256), "protected-plan readiness digest is invalid")
    _require(
        receipt_body["calibration_outcomes_payload_sha256"] == aggregate_digest,
        "protected-plan receipt aggregate binding differs",
    )
    readiness = _plain_dict(
        aggregate_body.get("readiness_binding"),
        "protected-plan aggregate readiness binding",
    )
    _require(
        readiness.get("readiness_receipt_sha256") == readiness_receipt_sha256,
        "protected-plan worker readiness binding differs",
    )
    input_binding = _plain_dict(exact_input_binding, "protected-plan exact input binding")
    _require(
        input_binding.get("schema") == THRESHOLD_FREEZE_EXACT_INPUT_BINDING_SCHEMA
        and input_binding.get("calibration_aggregate_payload_sha256") == aggregate_digest,
        "protected-plan exact input binding differs",
    )
    certification = _plain_dict(
        worker_readiness_certification_binding,
        "protected-plan worker readiness certification binding",
    )
    provenance = _plain_dict(worker_provenance, "protected-plan worker provenance")
    attestation = _plain_dict(
        zip_provenance_attestation,
        "protected-plan ZIP provenance attestation",
    )
    body = {
        "schema": PROTECTED_PLAN_WORKER_RESULT_SCHEMA,
        "nonauthorizing": True,
        "scientific_promotion_allowed": False,
        "automatic_promotion_allowed": False,
        "protected_outcomes_observed": False,
        "learner_outcomes_executed": False,
        "learner_execution_authorized": False,
        "protected_execution_permitted": False,
        "execution_issuer_available": False,
        "protected_readiness_created": False,
        "protected_execution_ledger_created": False,
        "calibration_aggregate_payload_sha256": aggregate_digest,
        "threshold_freeze_receipt_payload_sha256": receipt_digest,
        "readiness_receipt_sha256": readiness_receipt_sha256,
        "exact_input_binding": input_binding,
        "exact_input_binding_sha256": canonical_sha256(input_binding),
        "worker_readiness_certification_binding": certification,
        "worker_readiness_certification_binding_sha256": canonical_sha256(certification),
        "worker_provenance": provenance,
        "worker_provenance_sha256": canonical_sha256(provenance),
        "zip_provenance_attestation": attestation,
        "zip_provenance_attestation_sha256": canonical_sha256(attestation),
        "protected_plan": plan,
        "protected_plan_payload_sha256": plan_digest,
    }
    return _payload_with_digest(body)


def _worker_threshold_freeze(
    *,
    readiness_directory: Path,
    ledger_directory: Path,
    shard_publication_root: Path,
    aggregate_publication_root: Path,
    aggregate_payload_sha256: str,
) -> dict[str, object]:
    """Recompute an immutable aggregate and freeze thresholds inside its source ZIP."""

    bundle = load_validated_readiness_bundle(
        readiness_directory,
        recheck_current=False,
        recheck_runtime=False,
    )
    readiness_body = _plain_dict(bundle.payload.get("body"), "readiness body")
    expected_runtime_execution_identity = runtime_execution_identity_from_receipt(
        readiness_body.get("runtime_identity")
    )
    _require(
        build_runtime_execution_identity() == expected_runtime_execution_identity,
        "worker process runtime execution identity differs before threshold freezing",
    )
    readiness = _readiness_binding(bundle)
    certification = _aggregation_readiness_certification_binding(bundle)
    archive_path = readiness_directory.absolute() / "source.zip"
    worker_provenance = _zip_worker_provenance(archive_path, bundle)
    source_archive = _read_regular_file(
        archive_path,
        max_bytes=_MAX_SOURCE_ZIP_BYTES,
        label="threshold worker source ZIP",
    )
    zip_provenance_capability = attest_calibration_zip_provenance(
        readiness_bundle=bundle,
        readiness_source_archive=source_archive,
        source_archive_path=archive_path,
    )
    aggregate = _load_content_addressed_calibration_aggregate(
        aggregate_publication_root,
        aggregate_payload_sha256,
    )
    aggregate_body = _validate_payload_digest(aggregate, "calibration aggregate")
    _validate_aggregate_provenance_bindings(aggregate_body, bundle)
    _require(
        _plain_dict(aggregate_body.get("readiness_binding"), "aggregate readiness binding")
        == readiness,
        "threshold worker aggregate readiness binding differs",
    )
    with _threshold_input_publication_guard(
        shard_publication_root=shard_publication_root,
        readiness_receipt_sha256=bundle.receipt_sha256,
        managed_ledger_directory=ledger_directory,
    ):
        shards = load_complete_calibration_case_shards(
            shard_publication_root,
            expected_readiness_binding=readiness,
            managed_ledger_directory=ledger_directory,
        )
        inventory = snapshot_calibration_execution_inventory(ledger_directory)
        validated_aggregate = validate_calibration_aggregate(
            aggregate,
            shards,
            managed_ledger_snapshot=inventory,
            managed_ledger_directory=ledger_directory,
            aggregation_worker_provenance=worker_provenance,
            aggregation_zip_provenance_attestation=zip_provenance_capability.payload,
            aggregation_readiness_certification_binding=certification,
        )
        exact_input_binding = _threshold_freeze_exact_input_binding(
            validated_aggregate,
            shards,
            inventory,
        )
        try:
            receipt = materialize_hidden_regime_factorial_threshold_freeze_receipt(
                validated_aggregate
            )
            validated_receipt = validate_hidden_regime_factorial_threshold_freeze_receipt(
                receipt,
                calibration_aggregate=validated_aggregate,
            )
        except ThresholdFreezeError as exc:
            raise CalibrationError(str(exc)) from exc
        _require(
            build_runtime_execution_identity() == expected_runtime_execution_identity,
            "worker process runtime execution identity drifted during threshold freezing",
        )
        require_current_full_runtime_identity(readiness_body.get("runtime_identity"))
        return _threshold_freeze_worker_result(
            calibration_aggregate=validated_aggregate,
            threshold_freeze_receipt=validated_receipt,
            readiness_receipt_sha256=bundle.receipt_sha256,
            threshold_exact_input_binding=exact_input_binding,
            threshold_worker_readiness_certification_binding=certification,
            threshold_worker_provenance=worker_provenance,
            threshold_zip_provenance_attestation=zip_provenance_capability.payload,
        )


def _worker_protected_plan(
    *,
    readiness_directory: Path,
    ledger_directory: Path,
    shard_publication_root: Path,
    aggregate_publication_root: Path,
    aggregate_payload_sha256: str,
    threshold_receipt_publication_root: Path,
    threshold_receipt_payload_sha256: str,
) -> dict[str, object]:
    """Derive one unexecuted, nonauthorizing protected plan inside the certified ZIP."""

    bundle = load_validated_readiness_bundle(
        readiness_directory,
        recheck_current=False,
        recheck_runtime=False,
    )
    readiness_body = _plain_dict(bundle.payload.get("body"), "readiness body")
    expected_runtime_execution_identity = runtime_execution_identity_from_receipt(
        readiness_body.get("runtime_identity")
    )
    _require(
        build_runtime_execution_identity() == expected_runtime_execution_identity,
        "worker process runtime execution identity differs before protected planning",
    )
    readiness = _readiness_binding(bundle)
    certification = _aggregation_readiness_certification_binding(bundle)
    archive_path = readiness_directory.absolute() / "source.zip"
    worker_provenance = _zip_worker_provenance(archive_path, bundle)
    source_archive = _read_regular_file(
        archive_path,
        max_bytes=_MAX_SOURCE_ZIP_BYTES,
        label="protected-plan worker source ZIP",
    )
    zip_provenance_capability = attest_calibration_zip_provenance(
        readiness_bundle=bundle,
        readiness_source_archive=source_archive,
        source_archive_path=archive_path,
    )
    aggregate = _load_content_addressed_calibration_aggregate(
        aggregate_publication_root,
        aggregate_payload_sha256,
    )
    aggregate_body = _validate_payload_digest(aggregate, "protected-plan calibration aggregate")
    _validate_aggregate_provenance_bindings(aggregate_body, bundle)
    _require(
        _plain_dict(aggregate_body.get("readiness_binding"), "aggregate readiness binding")
        == readiness,
        "protected-plan worker aggregate readiness binding differs",
    )
    receipt = _load_content_addressed_threshold_freeze_receipt(
        threshold_receipt_publication_root,
        threshold_receipt_payload_sha256,
    )
    with _threshold_input_publication_guard(
        shard_publication_root=shard_publication_root,
        readiness_receipt_sha256=bundle.receipt_sha256,
        managed_ledger_directory=ledger_directory,
    ):
        shards = load_complete_calibration_case_shards(
            shard_publication_root,
            expected_readiness_binding=readiness,
            managed_ledger_directory=ledger_directory,
        )
        inventory = snapshot_calibration_execution_inventory(ledger_directory)
        validated_aggregate = validate_calibration_aggregate(
            aggregate,
            shards,
            managed_ledger_snapshot=inventory,
            managed_ledger_directory=ledger_directory,
            aggregation_worker_provenance=worker_provenance,
            aggregation_zip_provenance_attestation=zip_provenance_capability.payload,
            aggregation_readiness_certification_binding=certification,
        )
        exact_input_binding = _threshold_freeze_exact_input_binding(
            validated_aggregate,
            shards,
            inventory,
        )
        try:
            validated_receipt = validate_hidden_regime_factorial_threshold_freeze_receipt(
                receipt,
                calibration_aggregate=validated_aggregate,
            )
        except ThresholdFreezeError as exc:
            raise CalibrationError("protected-plan threshold receipt validation failed") from exc
        _successful_threshold_freeze_receipt_body(
            validated_receipt,
            label="protected-plan validated threshold receipt",
        )
        try:
            plan = build_hidden_regime_factorial_protected_plan(
                validated_receipt,
                calibration_aggregate=validated_aggregate,
            )
            validated_plan = validate_hidden_regime_factorial_protected_plan(
                plan,
                threshold_receipt=validated_receipt,
                calibration_aggregate=validated_aggregate,
            )
        except ProtectedPlanError as exc:
            raise CalibrationError(str(exc)) from exc
        _require(
            build_runtime_execution_identity() == expected_runtime_execution_identity,
            "worker process runtime execution identity drifted during protected planning",
        )
        require_current_full_runtime_identity(readiness_body.get("runtime_identity"))
        return _protected_plan_worker_result(
            calibration_aggregate=validated_aggregate,
            threshold_freeze_receipt=validated_receipt,
            protected_plan=validated_plan,
            readiness_receipt_sha256=bundle.receipt_sha256,
            exact_input_binding=exact_input_binding,
            worker_readiness_certification_binding=certification,
            worker_provenance=worker_provenance,
            zip_provenance_attestation=zip_provenance_capability.payload,
        )


def main(arguments: Sequence[str] | None = None) -> int:
    """ZIP-only worker entry point bound by the readiness receipt."""

    argv = tuple(sys.argv[1:] if arguments is None else arguments)
    _require(bool(argv), "worker arguments are not exact")
    if argv[0] == "--worker-protected-plan-v1":
        _require(len(argv) == 8, "protected-plan worker arguments are not exact")
        result = _worker_protected_plan(
            readiness_directory=Path(argv[1]).absolute(),
            ledger_directory=Path(argv[2]).absolute(),
            shard_publication_root=Path(argv[3]).absolute(),
            aggregate_publication_root=Path(argv[4]).absolute(),
            aggregate_payload_sha256=argv[5],
            threshold_receipt_publication_root=Path(argv[6]).absolute(),
            threshold_receipt_payload_sha256=argv[7],
        )
        raw = canonical_json_bytes(result)
        _require(
            len(raw) <= _MAX_PROTECTED_PLAN_BYTES,
            "protected-plan worker result exceeds output limit",
        )
        sys.stdout.buffer.write(PROTECTED_PLAN_RESULT_PREFIX + base64.b64encode(raw))
        sys.stdout.buffer.flush()
        return 0
    if argv[0] == "--worker-threshold-freeze-v1":
        _require(len(argv) == 6, "threshold-freeze worker arguments are not exact")
        result = _worker_threshold_freeze(
            readiness_directory=Path(argv[1]).absolute(),
            ledger_directory=Path(argv[2]).absolute(),
            shard_publication_root=Path(argv[3]).absolute(),
            aggregate_publication_root=Path(argv[4]).absolute(),
            aggregate_payload_sha256=argv[5],
        )
        raw = canonical_json_bytes(result)
        _require(
            len(raw) <= _MAX_WORKER_OUTPUT_BYTES,
            "threshold-freeze worker result exceeds output limit",
        )
        sys.stdout.buffer.write(THRESHOLD_FREEZE_RESULT_PREFIX + base64.b64encode(raw))
        sys.stdout.buffer.flush()
        return 0
    if argv[0] == "--worker-aggregate-v1":
        _require(len(argv) == 4, "aggregate worker arguments are not exact")
        result = _worker_aggregate(
            readiness_directory=Path(argv[1]).absolute(),
            ledger_directory=Path(argv[2]).absolute(),
            shard_publication_root=Path(argv[3]).absolute(),
        )
        prefix = AGGREGATE_RESULT_PREFIX
        raw = canonical_json_bytes(result)
        _require(len(raw) <= _MAX_AGGREGATE_BYTES, "aggregate worker result exceeds output limit")
        sys.stdout.buffer.write(prefix + base64.b64encode(raw))
        sys.stdout.buffer.flush()
        return 0
    _require(len(argv) == 4, "case/preflight worker arguments are not exact")
    readiness_directory = Path(argv[1]).absolute()
    ledger_directory = Path(argv[2]).absolute()
    request_payload = _decode_worker_request(argv[3])
    if argv[0] == "--worker-case-v1":
        result = _worker_case(
            readiness_directory=readiness_directory,
            ledger_directory=ledger_directory,
            request_payload=request_payload,
        )
        prefix = WORKER_RESULT_PREFIX
    elif argv[0] == "--worker-preflight-v1":
        result = _worker_preflight(
            readiness_directory=readiness_directory,
            ledger_directory=ledger_directory,
            request_payload=request_payload,
        )
        prefix = PREFLIGHT_RESULT_PREFIX
    else:
        _fail("unsupported calibration worker mode")
    raw = canonical_json_bytes(result)
    _require(len(raw) <= _MAX_WORKER_OUTPUT_BYTES, "worker result exceeds output limit")
    sys.stdout.buffer.write(prefix + base64.b64encode(raw))
    sys.stdout.buffer.flush()
    return 0


def _parse_worker_result(stdout: bytes) -> dict[str, object]:
    _require(len(stdout) <= _MAX_WORKER_OUTPUT_BYTES * 2, "worker output exceeds limit")
    _require(stdout.startswith(WORKER_RESULT_PREFIX), "worker output prefix differs")
    encoded = stdout[len(WORKER_RESULT_PREFIX) :]
    _require(
        bool(encoded) and b"\n" not in encoded and b"\r" not in encoded, "worker output differs"
    )
    try:
        raw = base64.b64decode(encoded, validate=True)
    except ValueError as exc:
        raise CalibrationError("worker result is not strict base64") from exc
    _require(len(raw) <= _MAX_WORKER_OUTPUT_BYTES, "decoded worker shard exceeds limit")
    return _strict_json(raw, "worker case shard")


def _parse_preflight_result(stdout: bytes) -> dict[str, object]:
    _require(len(stdout) <= _MAX_WORKER_OUTPUT_BYTES * 2, "preflight output exceeds limit")
    _require(stdout.startswith(PREFLIGHT_RESULT_PREFIX), "preflight output prefix differs")
    encoded = stdout[len(PREFLIGHT_RESULT_PREFIX) :]
    _require(
        bool(encoded) and b"\n" not in encoded and b"\r" not in encoded,
        "preflight output differs",
    )
    try:
        raw = base64.b64decode(encoded, validate=True)
    except ValueError as exc:
        raise CalibrationError("preflight result is not strict base64") from exc
    _require(len(raw) <= _MAX_WORKER_OUTPUT_BYTES, "decoded preflight result exceeds limit")
    return _strict_json(raw, "worker preflight report")


def _parse_aggregate_worker_result(stdout: bytes) -> dict[str, object]:
    _require(len(stdout) <= _MAX_AGGREGATE_BYTES * 2, "aggregate worker output exceeds limit")
    _require(stdout.startswith(AGGREGATE_RESULT_PREFIX), "aggregate worker output prefix differs")
    encoded = stdout[len(AGGREGATE_RESULT_PREFIX) :]
    _require(
        bool(encoded) and b"\n" not in encoded and b"\r" not in encoded,
        "aggregate worker output differs",
    )
    try:
        raw = base64.b64decode(encoded, validate=True)
    except ValueError as exc:
        raise CalibrationError("aggregate worker result is not strict base64") from exc
    _require(len(raw) <= _MAX_AGGREGATE_BYTES, "decoded aggregate result exceeds limit")
    payload = _strict_json(raw, "aggregate worker result")
    _require(raw == canonical_json_bytes(payload), "aggregate worker result is not canonical")
    body = _validate_payload_digest(payload, "aggregate worker result")
    _require(body.get("schema") == CALIBRATION_AGGREGATE_SCHEMA, "aggregate schema differs")
    _require(body.get("development_only") is True, "aggregate is not development-only")
    _require(
        body.get("scientific_promotion_allowed") is False,
        "aggregate permits scientific promotion",
    )
    _require(body.get("claim_accepted") is False, "aggregate accepts a claim")
    _require(body.get("thresholds_frozen") is False, "aggregate freezes thresholds")
    _require(body.get("promotion_artifact") is False, "aggregate is a promotion artifact")
    _require(body.get("case_count") == EXPECTED_CASES, "aggregate case count differs")
    audit_summary = _plain_dict(
        body.get("mandatory_audit_summary"),
        "aggregate mandatory audit summary",
    )
    _require(
        audit_summary.get("schema") == CALIBRATION_MANDATORY_AUDIT_SUMMARY_SCHEMA,
        "aggregate mandatory audit schema differs",
    )
    _require(
        body.get("mandatory_audit_summary_sha256") == canonical_sha256(audit_summary),
        "aggregate mandatory audit digest differs",
    )
    audit_decision = audit_summary.get("decision")
    _require(
        audit_decision in {"passed_nonstatistical", "invalid_calibration"}
        and body.get("mandatory_audit_decision") == audit_decision,
        "aggregate mandatory audit decision differs",
    )
    expected_gate_status = (
        "mandatory_audits_passed_statistical_thresholds_unset"
        if audit_decision == "passed_nonstatistical"
        else "invalid_calibration_mandatory_audit_failure"
    )
    _require(
        body.get("gate_decision_status") == expected_gate_status,
        "aggregate gate decision status differs",
    )
    _require(
        _is_sha256(payload.get("payload_sha256")),
        "aggregate worker payload digest is invalid",
    )
    return payload


def _validate_threshold_freeze_worker_result_payload(
    payload: Mapping[str, object],
) -> dict[str, object]:
    body = _validate_payload_digest(payload, "threshold-freeze worker result")
    _exact_keys(
        body,
        {
            "schema",
            "development_only",
            "scientific_promotion_allowed",
            "claim_accepted",
            "promotion_artifact",
            "calibration_aggregate_payload_sha256",
            "readiness_receipt_sha256",
            "threshold_exact_input_binding",
            "threshold_exact_input_binding_sha256",
            "threshold_worker_readiness_certification_binding",
            "threshold_worker_readiness_certification_binding_sha256",
            "threshold_worker_provenance",
            "threshold_worker_provenance_sha256",
            "threshold_zip_provenance_attestation",
            "threshold_zip_provenance_attestation_sha256",
            "threshold_freeze_receipt",
            "threshold_freeze_receipt_sha256",
        },
        "threshold-freeze worker result",
    )
    _require(
        body["schema"] == THRESHOLD_FREEZE_WORKER_RESULT_SCHEMA,
        "threshold-freeze worker result schema differs",
    )
    _require(body["development_only"] is True, "threshold worker result is not development-only")
    for field in (
        "scientific_promotion_allowed",
        "claim_accepted",
        "promotion_artifact",
    ):
        _require(body[field] is False, f"threshold worker result {field} must be false")
    for field in (
        "calibration_aggregate_payload_sha256",
        "readiness_receipt_sha256",
        "threshold_exact_input_binding_sha256",
        "threshold_worker_readiness_certification_binding_sha256",
        "threshold_worker_provenance_sha256",
        "threshold_zip_provenance_attestation_sha256",
        "threshold_freeze_receipt_sha256",
    ):
        _require(_is_sha256(body[field]), f"threshold worker result {field} is invalid")
    for value_field, digest_field in (
        ("threshold_exact_input_binding", "threshold_exact_input_binding_sha256"),
        (
            "threshold_worker_readiness_certification_binding",
            "threshold_worker_readiness_certification_binding_sha256",
        ),
        ("threshold_worker_provenance", "threshold_worker_provenance_sha256"),
        (
            "threshold_zip_provenance_attestation",
            "threshold_zip_provenance_attestation_sha256",
        ),
    ):
        value = _plain_dict(body[value_field], f"threshold worker result {value_field}")
        _require(
            canonical_sha256(value) == body[digest_field],
            f"threshold worker result {digest_field} differs",
        )
    exact_input_binding = _plain_dict(
        body["threshold_exact_input_binding"],
        "threshold exact input binding",
    )
    _exact_keys(
        exact_input_binding,
        {
            "schema",
            "calibration_aggregate_payload_sha256",
            "managed_ledger_inventory_sha256",
            "managed_ledger_snapshot_sha256",
            "case_ledger_sha256",
            "case_shard_payloads_sha256",
            "case_count",
        },
        "threshold exact input binding",
    )
    _require(
        exact_input_binding["schema"] == THRESHOLD_FREEZE_EXACT_INPUT_BINDING_SCHEMA,
        "threshold exact input binding schema differs",
    )
    for field in (
        "calibration_aggregate_payload_sha256",
        "managed_ledger_inventory_sha256",
        "managed_ledger_snapshot_sha256",
        "case_ledger_sha256",
        "case_shard_payloads_sha256",
    ):
        _require(_is_sha256(exact_input_binding[field]), f"threshold input {field} is invalid")
    _require(
        exact_input_binding["calibration_aggregate_payload_sha256"]
        == body["calibration_aggregate_payload_sha256"],
        "threshold exact input aggregate binding differs",
    )
    _require(
        exact_input_binding["case_count"] == EXPECTED_CASES,
        "threshold exact input case count differs",
    )
    receipt = _plain_dict(body["threshold_freeze_receipt"], "threshold-freeze receipt")
    receipt_body = _threshold_freeze_receipt_body(receipt, label="threshold-freeze receipt")
    _require(
        receipt["receipt_payload_sha256"] == body["threshold_freeze_receipt_sha256"],
        "threshold worker receipt digest binding differs",
    )
    _require(
        receipt_body["calibration_outcomes_payload_sha256"]
        == body["calibration_aggregate_payload_sha256"],
        "threshold worker receipt aggregate binding differs",
    )
    _require(
        receipt_body["readiness_receipt_sha256"] == body["readiness_receipt_sha256"],
        "threshold worker receipt readiness binding differs",
    )
    return dict(payload)


def _parse_threshold_freeze_worker_result(stdout: bytes) -> dict[str, object]:
    _require(
        len(stdout) <= _MAX_WORKER_OUTPUT_BYTES * 2,
        "threshold-freeze worker output exceeds limit",
    )
    _require(
        stdout.startswith(THRESHOLD_FREEZE_RESULT_PREFIX),
        "threshold-freeze worker output prefix differs",
    )
    encoded = stdout[len(THRESHOLD_FREEZE_RESULT_PREFIX) :]
    _require(
        bool(encoded) and b"\n" not in encoded and b"\r" not in encoded,
        "threshold-freeze worker output differs",
    )
    try:
        raw = base64.b64decode(encoded, validate=True)
    except ValueError as exc:
        raise CalibrationError("threshold-freeze worker result is not strict base64") from exc
    _require(
        len(raw) <= _MAX_WORKER_OUTPUT_BYTES,
        "decoded threshold-freeze worker result exceeds limit",
    )
    payload = _strict_json(raw, "threshold-freeze worker result")
    _require(
        raw == canonical_json_bytes(payload),
        "threshold-freeze worker result is not canonical",
    )
    return _validate_threshold_freeze_worker_result_payload(payload)


def _validate_protected_plan_worker_result_payload(
    payload: Mapping[str, object],
) -> dict[str, object]:
    body = _validate_payload_digest(payload, "protected-plan worker result")
    _exact_keys(
        body,
        {
            "schema",
            "nonauthorizing",
            "scientific_promotion_allowed",
            "automatic_promotion_allowed",
            "protected_outcomes_observed",
            "learner_outcomes_executed",
            "learner_execution_authorized",
            "protected_execution_permitted",
            "execution_issuer_available",
            "protected_readiness_created",
            "protected_execution_ledger_created",
            "calibration_aggregate_payload_sha256",
            "threshold_freeze_receipt_payload_sha256",
            "readiness_receipt_sha256",
            "exact_input_binding",
            "exact_input_binding_sha256",
            "worker_readiness_certification_binding",
            "worker_readiness_certification_binding_sha256",
            "worker_provenance",
            "worker_provenance_sha256",
            "zip_provenance_attestation",
            "zip_provenance_attestation_sha256",
            "protected_plan",
            "protected_plan_payload_sha256",
        },
        "protected-plan worker result",
    )
    _require(
        body["schema"] == PROTECTED_PLAN_WORKER_RESULT_SCHEMA,
        "protected-plan worker result schema differs",
    )
    _require(body["nonauthorizing"] is True, "protected-plan worker result authorizes")
    for field in (
        "scientific_promotion_allowed",
        "automatic_promotion_allowed",
        "protected_outcomes_observed",
        "learner_outcomes_executed",
        "learner_execution_authorized",
        "protected_execution_permitted",
        "execution_issuer_available",
        "protected_readiness_created",
        "protected_execution_ledger_created",
    ):
        _require(body[field] is False, f"protected-plan worker result {field} must be false")
    for field in (
        "calibration_aggregate_payload_sha256",
        "threshold_freeze_receipt_payload_sha256",
        "readiness_receipt_sha256",
        "exact_input_binding_sha256",
        "worker_readiness_certification_binding_sha256",
        "worker_provenance_sha256",
        "zip_provenance_attestation_sha256",
        "protected_plan_payload_sha256",
    ):
        _require(_is_sha256(body[field]), f"protected-plan result {field} is invalid")
    for value_field, digest_field in (
        ("exact_input_binding", "exact_input_binding_sha256"),
        (
            "worker_readiness_certification_binding",
            "worker_readiness_certification_binding_sha256",
        ),
        ("worker_provenance", "worker_provenance_sha256"),
        ("zip_provenance_attestation", "zip_provenance_attestation_sha256"),
    ):
        value = _plain_dict(body[value_field], f"protected-plan result {value_field}")
        _require(
            canonical_sha256(value) == body[digest_field],
            f"protected-plan result {digest_field} differs",
        )
    exact_input_binding = _plain_dict(
        body["exact_input_binding"],
        "protected-plan exact input binding",
    )
    _exact_keys(
        exact_input_binding,
        {
            "schema",
            "calibration_aggregate_payload_sha256",
            "managed_ledger_inventory_sha256",
            "managed_ledger_snapshot_sha256",
            "case_ledger_sha256",
            "case_shard_payloads_sha256",
            "case_count",
        },
        "protected-plan exact input binding",
    )
    _require(
        exact_input_binding.get("schema") == THRESHOLD_FREEZE_EXACT_INPUT_BINDING_SCHEMA
        and exact_input_binding.get("calibration_aggregate_payload_sha256")
        == body["calibration_aggregate_payload_sha256"]
        and exact_input_binding.get("case_count") == EXPECTED_CASES,
        "protected-plan exact input binding differs",
    )
    for field in (
        "calibration_aggregate_payload_sha256",
        "managed_ledger_inventory_sha256",
        "managed_ledger_snapshot_sha256",
        "case_ledger_sha256",
        "case_shard_payloads_sha256",
    ):
        _require(
            _is_sha256(exact_input_binding[field]),
            f"protected-plan exact input {field} is invalid",
        )
    plan = _plain_dict(body["protected_plan"], "protected-plan worker plan")
    plan_body = _protected_plan_body(plan, label="protected-plan worker plan")
    _require(
        plan.get("payload_sha256") == body["protected_plan_payload_sha256"],
        "protected-plan worker plan digest binding differs",
    )
    calibration_binding = _plain_dict(
        plan_body.get("calibration_binding"),
        "protected-plan calibration binding",
    )
    receipt_binding = _plain_dict(
        plan_body.get("threshold_freeze_receipt_binding"),
        "protected-plan threshold receipt binding",
    )
    _require(
        calibration_binding.get("calibration_aggregate_payload_sha256")
        == body["calibration_aggregate_payload_sha256"],
        "protected-plan result aggregate binding differs",
    )
    _require(
        receipt_binding.get("receipt_payload_sha256")
        == body["threshold_freeze_receipt_payload_sha256"],
        "protected-plan result threshold receipt binding differs",
    )
    return dict(payload)


def _parse_protected_plan_worker_result(stdout: bytes) -> dict[str, object]:
    _require(
        len(stdout) <= _MAX_PROTECTED_PLAN_BYTES * 2,
        "protected-plan worker output exceeds limit",
    )
    _require(
        stdout.startswith(PROTECTED_PLAN_RESULT_PREFIX),
        "protected-plan worker output prefix differs",
    )
    encoded = stdout[len(PROTECTED_PLAN_RESULT_PREFIX) :]
    _require(
        bool(encoded) and b"\n" not in encoded and b"\r" not in encoded,
        "protected-plan worker output differs",
    )
    try:
        raw = base64.b64decode(encoded, validate=True)
    except ValueError as exc:
        raise CalibrationError("protected-plan worker result is not strict base64") from exc
    _require(
        len(raw) <= _MAX_PROTECTED_PLAN_BYTES,
        "decoded protected-plan worker result exceeds limit",
    )
    payload = _strict_json(raw, "protected-plan worker result")
    _require(
        raw == canonical_json_bytes(payload),
        "protected-plan worker result is not canonical",
    )
    return _validate_protected_plan_worker_result_payload(payload)


def run_calibration_preflight_subprocess(
    *,
    readiness_directory: Path,
    managed_ledger_directory: Path,
    issue_process_local_authorizations: bool = False,
    explicit_acknowledgement: str | None = None,
    authorize_calibration_preflight: bool = False,
    timeout_seconds: int | None = None,
) -> dict[str, object]:
    """Validate all case bindings once in the ZIP without consuming any case."""

    _require(
        authorize_calibration_preflight is True,
        "calibration preflight requires explicit authorization",
    )
    _require(
        explicit_acknowledgement == PREFLIGHT_ACKNOWLEDGEMENT,
        "exact explicit calibration preflight acknowledgement is required",
    )
    _require(
        type(issue_process_local_authorizations) is bool,
        "issue_process_local_authorizations must be a strict boolean",
    )
    if timeout_seconds is not None:
        _strict_int(timeout_seconds, "timeout_seconds", minimum=1)
    bundle = load_validated_readiness_bundle(
        readiness_directory,
        recheck_current=False,
        recheck_runtime=False,
    )
    request = build_calibration_preflight_request(
        bundle,
        managed_ledger_directory=managed_ledger_directory,
        issue_process_local_authorizations=issue_process_local_authorizations,
        explicit_acknowledgement=explicit_acknowledgement,
    )
    encoded_request = base64.b64encode(canonical_json_bytes(request)).decode("ascii")
    with bound_calibration_runtime_batch(
        readiness_directory,
        authorize_batch_execution=True,
    ) as runtime_batch_guard:
        completed = execute_bound_calibration_worker(
            readiness_directory,
            (
                "--worker-preflight-v1",
                readiness_directory.absolute().as_posix(),
                managed_ledger_directory.absolute().as_posix(),
                encoded_request,
            ),
            authorize_calibration_execution=True,
            timeout_seconds=timeout_seconds,
            runtime_batch_guard=runtime_batch_guard,
        )
    if completed.returncode != 0:
        stderr_digest = hashlib.sha256(completed.stderr).hexdigest()
        raise CalibrationError(
            "isolated calibration preflight failed without learner execution; "
            f"returncode={completed.returncode},stderr_bytes={len(completed.stderr)},"
            f"stderr_sha256={stderr_digest}"
        )
    report = _parse_preflight_result(completed.stdout)
    validated = validate_calibration_preflight_report(report, request, bundle)
    inventory_after = snapshot_calibration_execution_inventory(managed_ledger_directory)
    try:
        inventory_after = require_valid_calibration_execution_inventory(
            inventory_after,
            managed_ledger_directory,
        )
    except RuntimeError as exc:
        raise CalibrationError(str(exc)) from exc
    request_body = _validate_payload_digest(request, "preflight request")
    _require_pristine_execution_inventory(
        inventory_after,
        expected_genesis_sha256=cast(str, request_body["managed_ledger_genesis_sha256"]),
    )
    report_body = _validate_payload_digest(validated, "preflight report")
    _require(
        inventory_after["inventory_sha256"] == report_body["inventory_after_sha256"],
        "managed ledger changed after preflight report",
    )
    return validated


def run_calibration_case_subprocess(
    *,
    case_index: int,
    readiness_directory: Path,
    managed_ledger_directory: Path,
    shard_publication_root: Path,
    explicit_acknowledgement: str,
    authorize_calibration_execution: bool,
    allow_exact_replay: bool = False,
    timeout_seconds: int | None = None,
    _runtime_batch_guard: BoundCalibrationRuntimeBatch | None = None,
) -> dict[str, object]:
    """Run or resume one exact case through the isolated content-addressed ZIP."""

    _require(
        authorize_calibration_execution is True,
        "calibration subprocess execution requires explicit authorization",
    )
    _case(_design(), case_index)
    _require(
        explicit_acknowledgement == EXECUTION_ACKNOWLEDGEMENT,
        "exact explicit calibration authorization acknowledgement is required",
    )
    _require(type(allow_exact_replay) is bool, "allow_exact_replay must be a strict boolean")
    if timeout_seconds is not None:
        _strict_int(timeout_seconds, "timeout_seconds", minimum=1)
    bundle = load_validated_readiness_bundle(
        readiness_directory,
        recheck_current=False,
        recheck_runtime=False,
    )
    readiness_binding = _readiness_binding(bundle)
    existing_path = calibration_case_shard_path(
        shard_publication_root,
        readiness_receipt_sha256=cast(
            str,
            readiness_binding["readiness_receipt_sha256"],
        ),
        case_index=case_index,
    )
    try:
        existing_raw = _read_regular_file(
            existing_path,
            max_bytes=_MAX_SHARD_BYTES,
            label="existing case shard",
        )
    except FileNotFoundError:
        existing_raw = None
    if existing_raw is not None:
        existing = _strict_json(existing_raw, "existing case shard")
        validated_existing = validate_finalized_calibration_case_shard(
            existing,
            expected_readiness_binding=readiness_binding,
            managed_ledger_directory=managed_ledger_directory,
        )
        existing_body = _validate_payload_digest(validated_existing, "existing case shard")
        _require(
            _plain_dict(existing_body["case"], "case")["case_index"] == case_index,
            "existing shard case index differs",
        )
        inventory = snapshot_calibration_execution_inventory(managed_ledger_directory)
        _require(
            existing_body["execution_record_binding"]
            == _execution_record_binding(inventory, case_index),
            "existing shard differs from managed completion",
        )
        return validated_existing
    inventory_before = snapshot_calibration_execution_inventory(managed_ledger_directory)
    try:
        inventory_before = require_valid_calibration_execution_inventory(
            inventory_before,
            managed_ledger_directory,
        )
    except RuntimeError as exc:
        raise CalibrationError(str(exc)) from exc
    finalized_indices = _plain_list(
        inventory_before.get("finalized_case_indices"),
        "finalized case indices",
    )
    if case_index in finalized_indices:
        try:
            recovered = load_finalized_calibration_case_shard(
                managed_ledger_directory,
                case_index,
            )
        except RuntimeError as exc:
            raise CalibrationError(str(exc)) from exc
        validated_recovered = validate_finalized_calibration_case_shard(
            recovered,
            expected_readiness_binding=readiness_binding,
            managed_ledger_directory=managed_ledger_directory,
        )
        publish_calibration_case_shard_new_only(
            shard_publication_root,
            validated_recovered,
            expected_readiness_binding=readiness_binding,
            managed_ledger_directory=managed_ledger_directory,
        )
        return validated_recovered
    request = build_calibration_case_request(
        case_index,
        bundle,
        managed_ledger_directory=managed_ledger_directory,
        explicit_acknowledgement=explicit_acknowledgement,
        allow_exact_replay=allow_exact_replay,
    )
    request_payload = request.to_payload()
    encoded_request = base64.b64encode(canonical_json_bytes(request_payload)).decode("ascii")
    completed = execute_bound_calibration_worker(
        readiness_directory,
        (
            "--worker-case-v1",
            readiness_directory.absolute().as_posix(),
            managed_ledger_directory.absolute().as_posix(),
            encoded_request,
        ),
        authorize_calibration_execution=True,
        timeout_seconds=timeout_seconds,
        runtime_batch_guard=_runtime_batch_guard,
    )
    if completed.returncode != 0:
        stderr_digest = hashlib.sha256(completed.stderr).hexdigest()
        raise CalibrationError(
            "isolated calibration worker failed; "
            f"returncode={completed.returncode},stderr_bytes={len(completed.stderr)},"
            f"stderr_sha256={stderr_digest}; the case may now be consumed and only exact "
            "explicit replay is permitted"
        )
    shard = _parse_worker_result(completed.stdout)
    validated = validate_finalized_calibration_case_shard(
        shard,
        expected_readiness_binding=readiness_binding,
        managed_ledger_directory=managed_ledger_directory,
    )
    inventory = snapshot_calibration_execution_inventory(managed_ledger_directory)
    expected_record = _execution_record_binding(inventory, case_index)
    shard_body = _validate_payload_digest(validated, "case shard")
    _require(
        shard_body["execution_record_binding"] == expected_record,
        "worker shard differs from managed completion",
    )
    publish_calibration_case_shard_new_only(
        shard_publication_root,
        validated,
        expected_readiness_binding=readiness_binding,
        managed_ledger_directory=managed_ledger_directory,
    )
    return validated


def run_calibration_cases_subprocess(
    *,
    case_indices: tuple[int, ...] | range,
    max_workers: int,
    readiness_directory: Path,
    managed_ledger_directory: Path,
    shard_publication_root: Path,
    explicit_acknowledgement: str | None = None,
    authorize_calibration_execution: bool = False,
    allow_exact_replay: bool = False,
    timeout_seconds: int | None = None,
) -> tuple[dict[str, object], ...]:
    """Run or resume an explicit bounded batch and return shards in caller order.

    Cases run in fixed batches of at most ``max_workers``.  A failed batch is
    allowed to finish, its successful immutable shards remain resumable, and no
    later batch starts.  The coordinator never retries, substitutes, or selects
    a replacement seed.
    """

    _require(
        authorize_calibration_execution is True,
        "calibration batch execution requires explicit authorization",
    )
    _require(
        explicit_acknowledgement == EXECUTION_ACKNOWLEDGEMENT,
        "exact explicit calibration authorization acknowledgement is required",
    )
    _require(
        type(case_indices) in {tuple, range},
        "case_indices must be an explicit tuple or range",
    )
    indices = tuple(case_indices)
    _require(bool(indices), "case_indices must not be empty")
    _require(len(indices) == len(set(indices)), "case_indices must be unique")
    for case_index in indices:
        _case(_design(), case_index)
    _require(
        type(max_workers) is int and 1 <= max_workers <= _MAX_CALIBRATION_WORKERS,
        f"max_workers must be a strict integer in [1, {_MAX_CALIBRATION_WORKERS}]",
    )
    _require(type(allow_exact_replay) is bool, "allow_exact_replay must be a strict boolean")
    if timeout_seconds is not None:
        _strict_int(timeout_seconds, "timeout_seconds", minimum=1)

    ordered: dict[int, dict[str, object]] = {}
    with bound_calibration_runtime_batch(
        readiness_directory,
        authorize_batch_execution=True,
    ) as runtime_batch_guard:
        for start in range(0, len(indices), max_workers):
            batch = indices[start : start + max_workers]
            failures: list[dict[str, object]] = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    case_index: executor.submit(
                        run_calibration_case_subprocess,
                        case_index=case_index,
                        readiness_directory=readiness_directory,
                        managed_ledger_directory=managed_ledger_directory,
                        shard_publication_root=shard_publication_root,
                        explicit_acknowledgement=cast(str, explicit_acknowledgement),
                        authorize_calibration_execution=True,
                        allow_exact_replay=allow_exact_replay,
                        timeout_seconds=timeout_seconds,
                        _runtime_batch_guard=runtime_batch_guard,
                    )
                    for case_index in batch
                }
                for case_index in batch:
                    try:
                        ordered[case_index] = futures[case_index].result()
                    except Exception as exc:
                        failure_text = f"{type(exc).__module__}.{type(exc).__qualname__}:{exc}"
                        failures.append(
                            {
                                "case_index": case_index,
                                "failure_sha256": hashlib.sha256(
                                    failure_text.encode("utf-8")
                                ).hexdigest(),
                            }
                        )
            if failures:
                failure_digest = canonical_sha256(failures)
                failed_indices = [item["case_index"] for item in failures]
                raise CalibrationError(
                    "calibration batch failed without retry or substitution; "
                    f"case_indices={failed_indices},failure_manifest_sha256={failure_digest}"
                )
    return tuple(ordered[case_index] for case_index in indices)


def _observation_map(shard: Mapping[str, object]) -> dict[str, dict[str, object]]:
    observations = _plain_list(shard.get("metric_observations"), "metric observations")
    result: dict[str, dict[str, object]] = {}
    for raw in observations:
        observation = _plain_dict(raw, "metric observation")
        metric_id = observation.get("metric_id")
        if type(metric_id) is not str or metric_id in result:
            _fail("metric observations have invalid or duplicate identifiers")
        result[metric_id] = observation
    return result


def _summary_records(shard: Mapping[str, object]) -> tuple[dict[str, object], ...]:
    summary = _plain_dict(_decode_exact(_plain_dict(shard.get("summary"), "summary")), "summary")
    return tuple(
        _plain_dict(item, "recurrence")
        for item in _plain_list(summary.get("recurrence_retention"), "recurrences")
    )


def _records_by_identity(
    shard: Mapping[str, object],
) -> dict[RecurrenceIdentity, dict[str, object]]:
    records = _summary_records(shard)
    result = {_identity(record): record for record in records}
    _require(len(result) == len(records), "shard recurrence identity collision")
    return result


def _metric_population_kind(metric_id: str) -> str:
    if metric_id in _SELECTED_METRICS:
        return "selected_surviving_qualified_intersection"
    if metric_id == "all_surviving_qualified_lineage_entry_composed_accuracy":
        return "surviving_qualified_probe_intersection"
    if metric_id in _DORMANT_METRICS or metric_id in _BEST_DORMANT_METRICS:
        return "dormant_probe_intersection"
    if metric_id in _QUALIFIED_METRICS:
        return "acquisition_qualified_intersection"
    return "full_matched_population"


def _record_observes_population(record: Mapping[str, object], kind: str) -> bool:
    if kind == "acquisition_qualified_intersection":
        return record.get("lineage_retention_applicable") is True
    if kind == "selected_surviving_qualified_intersection":
        return record.get("selected_lineage_available") is True
    if kind == "surviving_qualified_probe_intersection":
        probes = _plain_list(record.get("prior_same_regime_lineages"), "lineage probes")
        return any(
            _plain_dict(item, "lineage probe").get("acquisition_qualified") is True
            and _plain_dict(item, "lineage probe").get("synchronized_generation_survives") is True
            and _plain_dict(item, "lineage probe").get("entry_composed_greedy_accuracy") is not None
            for item in probes
        )
    if kind == "dormant_probe_intersection":
        return bool(_plain_list(record.get("eligible_dormant_generations"), "dormant probes"))
    if kind == "full_matched_population":
        return True
    _fail(f"unknown paired population kind {kind!r}")


def _paired_seed_metric(
    contract: MetricContract,
    estimand: EstimandContract,
    condition_shards: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    terms = tuple(estimand.condition_terms)
    records_by_condition = {
        condition: _records_by_identity(condition_shards[condition]) for condition, _ in terms
    }
    identity_sets = tuple(set(records) for records in records_by_condition.values())
    _require(bool(identity_sets), "estimand has no condition terms")
    all_identities = set.intersection(*identity_sets)
    _require(
        all(identity_set == all_identities for identity_set in identity_sets),
        "matched condition recurrence populations differ",
    )
    kind = _metric_population_kind(contract.metric_id)
    if kind == "full_matched_population":
        included = _canonical_identities(all_identities)
    else:
        included = _canonical_identities(
            identity
            for identity in all_identities
            if all(
                _record_observes_population(records_by_condition[condition][identity], kind)
                for condition, _ in terms
            )
        )
    included_set = set(included)
    excluded = _canonical_identities(
        identity for identity in all_identities if identity not in included_set
    )
    term_values: list[tuple[str, int, float | None, tuple[RecurrenceIdentity, ...]]] = []
    structural_missing: set[RecurrenceIdentity] = set()
    for condition, coefficient in terms:
        shard = condition_shards[condition]
        decoded_summary = _plain_dict(
            _decode_exact(_plain_dict(shard.get("summary"), "summary")),
            "summary",
        )
        selected_records = tuple(records_by_condition[condition][identity] for identity in included)
        value, eligible, observed = _metric_value(
            contract.metric_id,
            selected_records,
            decoded_summary,
        )
        observed_set = set(observed)
        structural_missing.update(identity for identity in eligible if identity not in observed_set)
        term_values.append((condition, coefficient, value, observed))

    delta: float | None
    if any(value is None for _, _, value, _ in term_values) or structural_missing:
        delta = None
    else:
        sign = 1.0 if contract.orientation == "higher" else -1.0
        delta = math.fsum(
            coefficient * sign * cast(float, value) for _, coefficient, value, _ in term_values
        )
    return {
        "population_kind": kind,
        "included_recurrence_identities": [_identity_payload(item) for item in included],
        "excluded_recurrence_identities": [_identity_payload(item) for item in excluded],
        "structural_missing_recurrence_identities": [
            _identity_payload(item) for item in sorted(structural_missing)
        ],
        "included_n": len(included),
        "excluded_n": len(excluded),
        "structural_missing_n": len(structural_missing),
        "term_values": [
            {
                "condition": condition,
                "coefficient": coefficient,
                "value_hex": None if value is None else _float_hex(value, "term value"),
                "observed_recurrence_identities": [_identity_payload(item) for item in observed],
            }
            for condition, coefficient, value, observed in term_values
        ],
        "oriented_delta_hex": None if delta is None else _float_hex(delta, "paired delta"),
    }


def _sample_statistics(values: Sequence[float], *, null: float | None) -> dict[str, object]:
    """Compute exact pre-rounding summaries under the frozen one-sided t plan."""

    array = np.asarray(values, dtype=np.float64)
    _require(array.ndim == 1, "statistical values must be one-dimensional")
    _require(bool(np.all(np.isfinite(array))), "statistical values must be finite")
    n = int(array.size)
    mean = None if n == 0 else float(np.mean(array, dtype=np.float64))
    sample_sd = None if n < 2 else float(np.std(array, ddof=1, dtype=np.float64))
    standard_error = None if sample_sd is None else float(sample_sd / math.sqrt(n))
    critical = None if n < 2 else float(student_t.ppf(0.95, n - 1))
    bound = (
        None
        if mean is None or standard_error is None or critical is None
        else float(mean - critical * standard_error)
    )
    wins = None if null is None else sum(value > null for value in values)
    ties = None if null is None else sum(value == null for value in values)
    losses = None if null is None else sum(value < null for value in values)
    return {
        "observed_n": n,
        "mean_hex": None if mean is None else _float_hex(mean, "mean"),
        "sample_standard_deviation_hex": (
            None if sample_sd is None else _float_hex(sample_sd, "sample SD")
        ),
        "standard_error_hex": (
            None if standard_error is None else _float_hex(standard_error, "standard error")
        ),
        "student_t_0_95_critical_hex": (
            None if critical is None else _float_hex(critical, "t critical")
        ),
        "one_sided_95_percent_lower_confidence_bound_hex": (
            None if bound is None else _float_hex(bound, "one-sided bound")
        ),
        "wins": wins,
        "ties": ties,
        "losses": losses,
    }


def _stratified_statistics(
    rows: Sequence[tuple[int, str, float | None, bool, bool]],
    *,
    null: float | None,
) -> dict[str, object]:
    """Summarize seed rows, separating conditional absence from structural loss."""

    def one(part: Sequence[tuple[int, str, float | None, bool, bool]]) -> dict[str, object]:
        values = [value for _, _, value, _, _ in part if value is not None]
        conditional = [
            seed
            for seed, _, value, conditional_absent, _ in part
            if value is None and conditional_absent
        ]
        structural = [
            seed
            for seed, _, value, _, structural_missing in part
            if value is None and structural_missing
        ]
        unclassified = [
            seed
            for seed, _, value, conditional_absent, structural_missing in part
            if value is None and not conditional_absent and not structural_missing
        ]
        _require(not unclassified, "a missing statistical value has no declared category")
        return {
            "eligible_n": len(part),
            "conditional_unobserved_n": len(conditional),
            "conditional_unobserved_seed_indices": conditional,
            "structural_missing_n": len(structural),
            "structural_missing_seed_indices": structural,
            **_sample_statistics(values, null=null),
        }

    pooled = one(rows)
    manifests: list[dict[str, object]] = []
    for manifest in CALIBRATION_MANIFEST_ORDER:
        subset = tuple(row for row in rows if row[1] == manifest)
        _require(len(subset) == EXPECTED_MANIFEST_SEED_PAIRS, "manifest stratum is not n=10")
        manifests.append({"manifest_name": manifest, **one(subset)})
    return {"pooled": pooled, "by_manifest": manifests}


def _level_rows(
    shards_by_case: Mapping[int, Mapping[str, object]],
    contract: MetricContract,
    condition: str,
) -> list[tuple[int, str, float | None, bool, bool]]:
    sign = 1.0 if contract.orientation == "higher" else -1.0
    rows: list[tuple[int, str, float | None, bool, bool]] = []
    for seed_index in range(EXPECTED_SEED_PAIRS):
        case_index = seed_index * EXPECTED_CONDITIONS + CANONICAL_CONDITION_ORDER.index(condition)
        shard = shards_by_case[case_index]
        case = _plain_dict(shard.get("case"), "case")
        observation = _observation_map(shard)[contract.metric_id]
        raw_value = observation.get("value_hex")
        value = (
            None if raw_value is None else sign * _parse_float_hex(raw_value, contract.metric_id)
        )
        eligible_n = _strict_int(observation.get("eligible_n"), "eligible_n")
        observed_n = _strict_int(observation.get("observed_n"), "observed_n")
        structural_n = _strict_int(observation.get("structural_missing_n"), "structural_missing_n")
        conditional_absent = value is None and eligible_n == 0 and structural_n == 0
        structural_missing = structural_n > 0 or (value is None and observed_n > 0)
        rows.append(
            (
                seed_index,
                cast(str, case["manifest_name"]),
                value,
                conditional_absent,
                structural_missing,
            )
        )
    return rows


def _paired_rows(
    shards_by_case: Mapping[int, Mapping[str, object]],
    contract: MetricContract,
    estimand: EstimandContract,
) -> tuple[list[tuple[int, str, float | None, bool, bool]], list[dict[str, object]]]:
    rows: list[tuple[int, str, float | None, bool, bool]] = []
    population_records: list[dict[str, object]] = []
    for seed_index in range(EXPECTED_SEED_PAIRS):
        condition_shards = {
            condition: shards_by_case[
                seed_index * EXPECTED_CONDITIONS + CANONICAL_CONDITION_ORDER.index(condition)
            ]
            for condition, _ in estimand.condition_terms
        }
        population = _paired_seed_metric(contract, estimand, condition_shards)
        raw_delta = population["oriented_delta_hex"]
        delta = None if raw_delta is None else _parse_float_hex(raw_delta, "oriented delta")
        structural_missing = cast(int, population["structural_missing_n"]) > 0
        conditional_absent = (
            delta is None and cast(int, population["included_n"]) == 0 and not structural_missing
        )
        first_shard = next(iter(condition_shards.values()))
        manifest = cast(str, _plain_dict(first_shard["case"], "case")["manifest_name"])
        population_records.append(
            {
                "seed_index": seed_index,
                "manifest_name": manifest,
                **population,
            }
        )
        rows.append((seed_index, manifest, delta, conditional_absent, structural_missing))
    return rows, population_records


def _paired_support_rows(
    shards_by_case: Mapping[int, Mapping[str, object]],
    conditions: tuple[str, ...],
    *,
    require_selected: bool,
) -> tuple[
    list[tuple[int, str, float | None, bool, bool]],
    list[dict[str, object]],
]:
    rows: list[tuple[int, str, float | None, bool, bool]] = []
    population_records: list[dict[str, object]] = []
    for seed_index in range(EXPECTED_SEED_PAIRS):
        condition_records = tuple(
            _records_by_identity(
                shards_by_case[
                    seed_index * EXPECTED_CONDITIONS + CANONICAL_CONDITION_ORDER.index(condition)
                ]
            )
            for condition in conditions
        )
        identities = set(condition_records[0])
        _require(
            all(set(records) == identities for records in condition_records), "support mismatch"
        )
        included = _canonical_identities(
            identity
            for identity in identities
            if all(
                records[identity].get(
                    "selected_lineage_available"
                    if require_selected
                    else "lineage_retention_applicable"
                )
                is True
                for records in condition_records
            )
        )
        included_set = set(included)
        excluded = _canonical_identities(
            identity for identity in identities if identity not in included_set
        )
        value = float(len(included) / len(identities))
        case = _plain_dict(
            shards_by_case[seed_index * EXPECTED_CONDITIONS].get("case"),
            "case",
        )
        manifest = cast(str, case["manifest_name"])
        rows.append((seed_index, manifest, value, False, False))
        population_records.append(
            {
                "seed_index": seed_index,
                "manifest_name": manifest,
                "included_recurrence_identities": [_identity_payload(item) for item in included],
                "excluded_recurrence_identities": [_identity_payload(item) for item in excluded],
                "included_n": len(included),
                "excluded_n": len(excluded),
                "denominator_n": len(identities),
                "coverage_hex": _float_hex(value, "paired support coverage"),
            }
        )
    return rows, population_records


def _with_worst_manifest(summary: dict[str, object]) -> dict[str, object]:
    strata = tuple(
        _plain_dict(item, "manifest summary")
        for item in _plain_list(summary.get("by_manifest"), "manifest summaries")
    )

    def floats(field: str) -> list[float]:
        return [
            _parse_float_hex(item[field], field) for item in strata if item.get(field) is not None
        ]

    means = floats("mean_hex")
    bounds = floats("one_sided_95_percent_lower_confidence_bound_hex")
    wins = [cast(int, item["wins"]) for item in strata if item.get("wins") is not None]
    worst = {
        "minimum_manifest_mean_hex": (
            None if len(means) != len(strata) else _float_hex(min(means), "worst mean")
        ),
        "minimum_manifest_one_sided_95_percent_lower_confidence_bound_hex": (
            None if len(bounds) != len(strata) else _float_hex(min(bounds), "worst bound")
        ),
        "minimum_manifest_wins": None if len(wins) != len(strata) else min(wins),
        "maximum_manifest_structural_missing_n": max(
            cast(int, item["structural_missing_n"]) for item in strata
        ),
        "maximum_manifest_conditional_unobserved_n": max(
            cast(int, item["conditional_unobserved_n"]) for item in strata
        ),
    }
    return {**summary, "worst_manifest": worst}


def _aggregate_levels(
    design: HiddenRegimeFactorialCalibrationDesign,
    shards_by_case: Mapping[int, Mapping[str, object]],
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for condition in CANONICAL_CONDITION_ORDER:
        for contract in design.metrics:
            null = (
                None
                if contract.null_value_decimal is None
                else float(contract.null_value_decimal)
                * (1.0 if contract.orientation == "higher" else -1.0)
            )
            rows = _level_rows(shards_by_case, contract, condition)
            statistics = _with_worst_manifest(_stratified_statistics(rows, null=null))
            results.append(
                {
                    "condition": condition,
                    "metric_id": contract.metric_id,
                    "metric_role": contract.role,
                    "gate_mode": contract.gate_mode,
                    "orientation": contract.orientation,
                    "orientation_applied": True,
                    "oriented_null_hex": None if null is None else _float_hex(null, "level null"),
                    "statistics": statistics,
                }
            )
    return results


def _aggregate_estimands(
    design: HiddenRegimeFactorialCalibrationDesign,
    shards_by_case: Mapping[int, Mapping[str, object]],
) -> list[dict[str, object]]:
    contracts = {item.metric_id: item for item in design.metrics}
    results: list[dict[str, object]] = []
    for estimand in design.factorial_estimands + design.control_estimands:
        metric_results: list[dict[str, object]] = []
        for metric_id in estimand.metrics:
            contract = contracts[metric_id]
            rows, populations = _paired_rows(shards_by_case, contract, estimand)
            statistics = _with_worst_manifest(_stratified_statistics(rows, null=0.0))
            metric_results.append(
                {
                    "metric_id": metric_id,
                    "orientation": contract.orientation,
                    "population_rule": estimand.population_rule,
                    "seed_population_records": populations,
                    "seed_population_records_sha256": canonical_sha256(populations),
                    "statistics": statistics,
                }
            )
        results.append(
            {
                "estimand_id": estimand.estimand_id,
                "role": estimand.role,
                "condition_terms": [
                    {"condition": condition, "coefficient": coefficient}
                    for condition, coefficient in estimand.condition_terms
                ],
                "oriented_null_hex": _float_hex(0.0, "contrast null"),
                "metrics": metric_results,
            }
        )
    return results


def _aggregate_support_metrics(
    design: HiddenRegimeFactorialCalibrationDesign,
    shards_by_case: Mapping[int, Mapping[str, object]],
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for support in design.paired_population_support_metrics:
        rows, populations = _paired_support_rows(
            shards_by_case,
            support.conditions,
            require_selected="selected_survival" in support.metric_id,
        )
        results.append(
            {
                "metric_id": support.metric_id,
                "estimand_id": support.estimand_id,
                "conditions": list(support.conditions),
                "orientation": support.orientation,
                "oriented_null_hex": _float_hex(float(support.null_value_decimal), "support null"),
                "seed_population_records": populations,
                "seed_population_records_sha256": canonical_sha256(populations),
                "statistics": _with_worst_manifest(_stratified_statistics(rows, null=0.0)),
            }
        )
    return results


def _mandatory_case_audit_reference(
    design: HiddenRegimeFactorialCalibrationDesign,
    shard: Mapping[str, object],
) -> dict[str, object]:
    """Project one validated shard into deterministic mandatory audit predicates."""

    case = _plain_dict(shard.get("case"), "mandatory audit case")
    case_index = _strict_int(case.get("case_index"), "mandatory audit case index")
    expected_case = _case(design, case_index)
    _require(case == expected_case.to_payload(), "mandatory audit case binding differs")
    condition = expected_case.condition
    summary_encoded = _plain_dict(shard.get("summary"), "mandatory audit summary")
    summary = _plain_dict(_decode_exact(summary_encoded), "decoded mandatory audit summary")
    audit = _plain_dict(shard.get("audit"), "mandatory case audit")
    resource = _plain_dict(shard.get("resource"), "mandatory case resource")
    trace = _plain_dict(shard.get("primitive_trace"), "mandatory primitive trace binding")

    for field in (
        "payload_sha256",
        "summary_sha256",
        "resource_sha256",
        "configuration_sha256",
        "case_request_binding_sha256",
    ):
        _require(_is_sha256(shard.get(field)), f"mandatory audit shard {field} is invalid")
    _require(
        canonical_sha256(summary_encoded) == shard["summary_sha256"],
        "mandatory audit summary digest differs",
    )
    _require(
        canonical_sha256(resource) == shard["resource_sha256"],
        "mandatory audit resource digest differs",
    )
    _require(_is_sha256(trace.get("sha256")), "mandatory audit trace digest is invalid")
    _require(
        audit.get("audited_summary_sha256") == canonical_sha256(_encode_exact(summary)),
        "mandatory audit summary was not the audited summary",
    )

    helper_writes = _strict_int(
        summary.get("helper_value_write_count"),
        "helper value-write count",
    )
    beneficiary_writes = _strict_int(
        summary.get("beneficiary_value_write_count"),
        "beneficiary value-write count",
    )
    helper_learning = _strict_int(
        summary.get("helper_effective_learning_update_count"),
        "helper effective-learning count",
    )
    beneficiary_learning = _strict_int(
        summary.get("beneficiary_effective_learning_update_count"),
        "beneficiary effective-learning count",
    )
    both_roles_learned = _strict_bool(
        summary.get("both_roles_learned"),
        "both_roles_learned",
    )
    both_roles_learning = (
        helper_writes > 0
        and beneficiary_writes > 0
        and helper_learning > 0
        and beneficiary_learning > 0
        and both_roles_learned
    )

    replacement_count = _strict_int(
        summary.get("c_old_to_c_new_replacement_count"),
        "C-old to C-new replacement count",
    )
    replacement_slots = tuple(
        _strict_int(item, "C-old to C-new target slot")
        for item in _plain_list(
            summary.get("c_old_to_c_new_target_slots"),
            "C-old to C-new target slots",
        )
    )
    generation_pairs: list[tuple[int, int]] = []
    for raw_pair in _plain_list(
        summary.get("c_old_to_c_new_generation_pairs"),
        "C-old to C-new generation pairs",
    ):
        pair = _plain_list(raw_pair, "C-old to C-new generation pair")
        _require(len(pair) == 2, "C-old to C-new generation pair is not exact")
        generation_pairs.append(
            (
                _strict_int(pair[0], "retired C-old generation"),
                _strict_int(pair[1], "committed C-new generation"),
            )
        )
    exactly_one_target = _strict_bool(
        summary.get("c_old_to_c_new_exactly_one_target"),
        "C-old to C-new exactly-one-target flag",
    )
    lifecycle_synchronized = _strict_bool(
        summary.get("lifecycle_synchronized_every_step"),
        "lifecycle synchronized-every-step flag",
    )
    atomic_replacement = (
        exactly_one_target
        and lifecycle_synchronized
        and replacement_count == 1
        and len(replacement_slots) == 1
        and replacement_slots[0] in (1, 2, 3)
        and len(generation_pairs) == 1
        and 0 < generation_pairs[0][0] < generation_pairs[0][1]
    )

    d_short_checked = _strict_bool(summary.get("d_short_checked"), "D-short checked flag")
    d_short_non_displacement = _strict_bool(
        summary.get("d_short_non_displacement"),
        "D-short non-displacement flag",
    )
    d_short_lifecycle_passed = (
        d_short_checked and d_short_non_displacement and lifecycle_synchronized
    )

    immutability_applicable = condition not in {"writable_evidence", "writable_lru"}
    recorded_immutability_applicable = _strict_bool(
        summary.get("selective_immutability_applicable"),
        "selective immutability applicability",
    )
    helper_mutations = _strict_int(
        summary.get("helper_selective_mutation_violations"),
        "helper selective mutation violations",
    )
    beneficiary_mutations = _strict_int(
        summary.get("beneficiary_selective_mutation_violations"),
        "beneficiary selective mutation violations",
    )
    selective_immutable = _strict_bool(
        summary.get("selective_durable_bit_immutable_until_atomic_replacement"),
        "selective durable immutability flag",
    )
    immutability_applicability_exact = recorded_immutability_applicable is immutability_applicable
    selective_immutability = (
        immutability_applicability_exact
        and helper_mutations == 0
        and beneficiary_mutations == 0
        and (selective_immutable if immutability_applicable else not selective_immutable)
    )
    d_short_passed = d_short_lifecycle_passed and selective_immutability

    resource_constant = (
        _strict_bool(resource.get("resource_constant"), "resource constant flag")
        and _strict_bool(resource.get("resource_matched"), "resource matched flag")
        and _strict_int(resource.get("initial_state_scalars"), "initial state scalars") == 138
        and _strict_int(resource.get("final_state_scalars"), "final state scalars") == 138
        and _strict_int(resource.get("initial_state_bytes"), "initial state bytes") == 552
        and _strict_int(resource.get("final_state_bytes"), "final state bytes") == 552
        and _strict_int(resource.get("expected_state_bytes"), "expected state bytes") == 552
    )

    trace_valid = _strict_bool(audit.get("valid"), "case trace audit valid flag")
    transition_counts_complete = all(
        _strict_int(audit.get(field), f"case audit {field}") == EXPECTED_STEPS
        for field in (
            "expected_steps",
            "rows_checked",
            "helper_transitions_checked",
            "beneficiary_transitions_checked",
            "world_transitions_checked",
        )
    )
    lineage_valid = _strict_bool(
        audit.get("lineage_oracle_valid"),
        "case lineage audit valid flag",
    ) and audit.get("lineage_oracle_mismatches_sha256") == canonical_sha256([])
    no_trace_mismatches = (
        _strict_int(audit.get("mismatch_count"), "case audit mismatch count") == 0
        and audit.get("mismatches_sha256") == canonical_sha256([])
        and audit.get("unobserved_transition_fields") == []
    )
    complete_trace_audit = trace_valid and transition_counts_complete and no_trace_mismatches
    complete_role_lifecycle = (
        complete_trace_audit and lifecycle_synchronized and selective_immutability
    )
    complete_world = complete_trace_audit

    frozen_role_control = True
    if condition == "helper_frozen":
        frozen_role_control = (
            helper_writes == 0
            and helper_learning == 0
            and _strict_int(summary.get("helper_commit_count"), "helper commit count") == 0
            and _strict_int(
                summary.get("helper_replacement_count"),
                "helper replacement count",
            )
            == 0
            and resource_constant
        )
    elif condition == "beneficiary_frozen":
        frozen_role_control = (
            beneficiary_writes == 0
            and beneficiary_learning == 0
            and _strict_int(
                summary.get("beneficiary_commit_count"),
                "beneficiary commit count",
            )
            == 0
            and _strict_int(
                summary.get("beneficiary_replacement_count"),
                "beneficiary replacement count",
            )
            == 0
            and resource_constant
        )

    source_bound_trace = (
        trace.get("schema") == HIDDEN_REGIME_TRACE_SCHEMA
        and trace.get("rows") == EXPECTED_STEPS
        and trace.get("persisted") is False
        and trace.get("discard_required_after_audit") is True
    )
    channel_control = complete_world and resource_constant
    predicate_results = {
        "lineage_serialization": lineage_valid and complete_trace_audit,
        "both_roles_learning": both_roles_learning,
        "atomic_c_old_to_c_new_replacement": atomic_replacement,
        "d_short_non_displacement": d_short_passed,
        "constant_resource": resource_constant,
        "complete_role_lifecycle_oracle": complete_role_lifecycle,
        "complete_world_oracle": complete_world,
        "source_bound_trace_contract": source_bound_trace and complete_trace_audit,
        "frozen_role_causal_controls": frozen_role_control,
        "channel_causal_controls": channel_control,
        "selective_immutability_where_applicable": selective_immutability,
    }
    _require(
        all(type(value) is bool for value in predicate_results.values()),
        "mandatory case predicate result is not boolean",
    )
    reference_body = {
        "case_index": case_index,
        "seed_index": expected_case.seed_index,
        "condition": condition,
        "manifest_name": expected_case.manifest_name,
        "case_shard_payload_sha256": shard["payload_sha256"],
        "case_request_binding_sha256": shard["case_request_binding_sha256"],
        "configuration_sha256": shard["configuration_sha256"],
        "summary_sha256": shard["summary_sha256"],
        "resource_sha256": shard["resource_sha256"],
        "audit_sha256": canonical_sha256(audit),
        "primitive_trace_sha256": trace["sha256"],
        "predicate_results": predicate_results,
    }
    return {
        **reference_body,
        "case_audit_reference_sha256": canonical_sha256(reference_body),
    }


def _case_predicate_reference(
    case_reference: Mapping[str, object],
    predicate_id: str,
    *,
    descriptive_only: bool,
) -> dict[str, object]:
    predicates = _plain_dict(case_reference.get("predicate_results"), "case predicates")
    observed = _strict_bool(
        predicates.get(predicate_id),
        f"case predicate {predicate_id}",
    )
    return {
        "kind": "descriptive_case_audit" if descriptive_only else "required_case_audit",
        "case_index": case_reference["case_index"],
        "seed_index": case_reference["seed_index"],
        "condition": case_reference["condition"],
        "case_audit_reference_sha256": case_reference["case_audit_reference_sha256"],
        "predicate_observed": observed,
        "decision_effect": "descriptive_only" if descriptive_only else "mandatory",
    }


def _case_requirement_result(
    requirement: object,
    case_references: Sequence[Mapping[str, object]],
    *,
    predicate_id: str,
    required_conditions: tuple[str, ...],
    descriptive_conditions: tuple[str, ...] = (),
) -> dict[str, object]:
    requirement_payload = cast(Any, requirement).to_payload()
    required = [
        _case_predicate_reference(item, predicate_id, descriptive_only=False)
        for item in case_references
        if item.get("condition") in required_conditions
    ]
    descriptive = [
        _case_predicate_reference(item, predicate_id, descriptive_only=True)
        for item in case_references
        if item.get("condition") in descriptive_conditions
    ]
    _require(bool(required), f"mandatory audit {predicate_id} has no required references")
    failed = [
        _strict_int(item["case_index"], "failed mandatory audit case")
        for item in required
        if item["predicate_observed"] is False
    ]
    result_body = {
        **requirement_payload,
        "evaluation_mode": "case_outcome_or_validated_case_invariant",
        "threshold_independent": True,
        "thresholds_consulted": False,
        "decision": "passed_nonstatistical" if not failed else "invalid_calibration",
        "required_reference_count": len(required),
        "required_references": required,
        "required_references_sha256": canonical_sha256(required),
        "descriptive_reference_count": len(descriptive),
        "descriptive_references": descriptive,
        "descriptive_references_sha256": canonical_sha256(descriptive),
        "failed_case_indices": failed,
    }
    return {
        **result_body,
        "requirement_result_sha256": canonical_sha256(result_body),
    }


def _readiness_requirement_result(
    requirement: object,
    certification_binding: Mapping[str, object],
) -> dict[str, object]:
    requirement_payload = cast(Any, requirement).to_payload()
    certification_binding_sha256 = canonical_sha256(certification_binding)
    reference = {
        "kind": "readiness_certification",
        "certification_id": READINESS_EQUIVALENCE_CERTIFICATION_ID,
        "readiness_receipt_sha256": certification_binding["readiness_receipt_sha256"],
        "certification_binding_sha256": certification_binding_sha256,
        "predicate_observed": True,
        "decision_effect": "mandatory",
    }
    result_body = {
        **requirement_payload,
        "evaluation_mode": "readiness_certification_not_per_case_execution",
        "threshold_independent": True,
        "thresholds_consulted": False,
        "decision": "passed_nonstatistical",
        "required_reference_count": 1,
        "required_references": [reference],
        "required_references_sha256": canonical_sha256([reference]),
        "descriptive_reference_count": 0,
        "descriptive_references": [],
        "descriptive_references_sha256": canonical_sha256([]),
        "failed_case_indices": [],
    }
    return {
        **result_body,
        "requirement_result_sha256": canonical_sha256(result_body),
    }


def _build_mandatory_audit_summary(
    design: HiddenRegimeFactorialCalibrationDesign,
    shards_by_case: Mapping[int, Mapping[str, object]],
    aggregation_readiness_certification_binding: Mapping[str, object],
) -> dict[str, object]:
    """Evaluate all frozen non-statistical requirements without thresholds."""

    _require(
        set(shards_by_case) == set(range(EXPECTED_CASES)),
        "mandatory audit requires the exact 240-case ledger",
    )
    certification_binding = _validate_aggregation_readiness_certification_binding(
        aggregation_readiness_certification_binding
    )
    case_references = [
        _mandatory_case_audit_reference(design, shards_by_case[index])
        for index in range(EXPECTED_CASES)
    ]
    requirements = {item.requirement_id: item for item in design.audits}
    _require(
        len(requirements) == len(design.audits),
        "mandatory audit requirement identifiers are not unique",
    )
    factor_conditions = (
        "selective_full",
        "writable_evidence",
        "selective_lru",
        "writable_lru",
    )
    all_conditions = tuple(CANONICAL_CONDITION_ORDER)
    results_by_id = {
        "lineage_serialization": _case_requirement_result(
            requirements["lineage_serialization"],
            case_references,
            predicate_id="lineage_serialization",
            required_conditions=all_conditions,
        ),
        "both_roles_learning": _case_requirement_result(
            requirements["both_roles_learning"],
            case_references,
            predicate_id="both_roles_learning",
            required_conditions=factor_conditions,
        ),
        "atomic_c_old_to_c_new_replacement": _case_requirement_result(
            requirements["atomic_c_old_to_c_new_replacement"],
            case_references,
            predicate_id="atomic_c_old_to_c_new_replacement",
            required_conditions=("selective_full",),
            descriptive_conditions=factor_conditions[1:],
        ),
        "d_short_non_displacement": _case_requirement_result(
            requirements["d_short_non_displacement"],
            case_references,
            predicate_id="d_short_non_displacement",
            required_conditions=("selective_full",),
            descriptive_conditions=factor_conditions[1:],
        ),
        "constant_resource": _case_requirement_result(
            requirements["constant_resource"],
            case_references,
            predicate_id="constant_resource",
            required_conditions=all_conditions,
        ),
        "complete_role_lifecycle_oracle": _case_requirement_result(
            requirements["complete_role_lifecycle_oracle"],
            case_references,
            predicate_id="complete_role_lifecycle_oracle",
            required_conditions=all_conditions,
        ),
        "complete_world_oracle": _case_requirement_result(
            requirements["complete_world_oracle"],
            case_references,
            predicate_id="complete_world_oracle",
            required_conditions=all_conditions,
        ),
        "source_bound_trace_contract": _case_requirement_result(
            requirements["source_bound_trace_contract"],
            case_references,
            predicate_id="source_bound_trace_contract",
            required_conditions=all_conditions,
        ),
        "decentralized_role_equivalence": _readiness_requirement_result(
            requirements["decentralized_role_equivalence"],
            certification_binding,
        ),
        "checkpoint_resume_equivalence": _readiness_requirement_result(
            requirements["checkpoint_resume_equivalence"],
            certification_binding,
        ),
        "frozen_role_causal_controls": _case_requirement_result(
            requirements["frozen_role_causal_controls"],
            case_references,
            predicate_id="frozen_role_causal_controls",
            required_conditions=("helper_frozen", "beneficiary_frozen"),
        ),
        "channel_causal_controls": _case_requirement_result(
            requirements["channel_causal_controls"],
            case_references,
            predicate_id="channel_causal_controls",
            required_conditions=("constant_channel_0", "shuffled_channel"),
        ),
    }
    requirement_results = [results_by_id[item.requirement_id] for item in design.audits]
    applicable_immutability = [
        _case_predicate_reference(
            item,
            "selective_immutability_where_applicable",
            descriptive_only=False,
        )
        for item in case_references
        if item["condition"] not in {"writable_evidence", "writable_lru"}
    ]
    immutability_failures = [
        _strict_int(item["case_index"], "selective immutability failure case")
        for item in applicable_immutability
        if item["predicate_observed"] is False
    ]
    selective_immutability_result = {
        "subpredicate_id": "selective_immutability_where_applicable",
        "scope": "every non-writable selective-policy case",
        "threshold_independent": True,
        "thresholds_consulted": False,
        "decision": (
            "passed_nonstatistical" if not immutability_failures else "invalid_calibration"
        ),
        "required_reference_count": len(applicable_immutability),
        "required_references": applicable_immutability,
        "required_references_sha256": canonical_sha256(applicable_immutability),
        "failed_case_indices": immutability_failures,
    }
    failed_requirement_ids = [
        cast(str, item["requirement_id"])
        for item in requirement_results
        if item["decision"] == "invalid_calibration"
    ]
    if immutability_failures and "complete_role_lifecycle_oracle" not in failed_requirement_ids:
        _fail("selective immutability failure escaped the lifecycle requirement")
    decision = "passed_nonstatistical" if not failed_requirement_ids else "invalid_calibration"
    return {
        "schema": CALIBRATION_MANDATORY_AUDIT_SUMMARY_SCHEMA,
        "threshold_independent": True,
        "thresholds_consulted": False,
        "integrity_status": "passed_before_mechanism_decision",
        "decision": decision,
        "case_audit_reference_count": len(case_references),
        "case_audit_references": case_references,
        "case_audit_references_sha256": canonical_sha256(case_references),
        "selective_immutability_result": selective_immutability_result,
        "selective_immutability_result_sha256": canonical_sha256(
            selective_immutability_result
        ),
        "requirement_result_count": len(requirement_results),
        "requirement_results": requirement_results,
        "requirement_results_sha256": canonical_sha256(requirement_results),
        "failed_requirement_ids": failed_requirement_ids,
        "readiness_certification_binding_sha256": canonical_sha256(certification_binding),
    }


def _gate_result_matrix(
    design: HiddenRegimeFactorialCalibrationDesign,
    levels: Sequence[Mapping[str, object]],
    estimands: Sequence[Mapping[str, object]],
    supports: Sequence[Mapping[str, object]],
    mandatory_audit_summary: Mapping[str, object],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    audit_summary = _plain_dict(mandatory_audit_summary, "mandatory audit summary")
    _require(
        audit_summary.get("schema") == CALIBRATION_MANDATORY_AUDIT_SUMMARY_SCHEMA,
        "mandatory audit summary schema differs",
    )
    audit_decision = audit_summary.get("decision")
    _require(
        audit_decision in {"passed_nonstatistical", "invalid_calibration"},
        "mandatory audit decision differs",
    )
    _require(
        audit_summary.get("threshold_independent") is True
        and audit_summary.get("thresholds_consulted") is False,
        "mandatory audit decision is not threshold-independent",
    )
    audit_requirement_results = [
        _plain_dict(item, "mandatory audit requirement result")
        for item in _plain_list(
            audit_summary.get("requirement_results"),
            "mandatory audit requirement results",
        )
    ]
    _require(
        [item.get("requirement_id") for item in audit_requirement_results]
        == [item.requirement_id for item in design.audits],
        "mandatory audit requirement result order differs",
    )
    _require(
        audit_summary.get("requirement_results_sha256")
        == canonical_sha256(audit_requirement_results),
        "mandatory audit requirement result digest differs",
    )
    for item in audit_requirement_results:
        result_body = dict(item)
        result_sha256 = result_body.pop("requirement_result_sha256", None)
        _require(
            _is_sha256(result_sha256) and canonical_sha256(result_body) == result_sha256,
            "mandatory audit requirement content digest differs",
        )
        _require(
            item.get("decision") in {"passed_nonstatistical", "invalid_calibration"},
            "mandatory audit requirement decision differs",
        )
        requirement_references = _plain_list(
            item.get("required_references"),
            "audit references",
        )
        _require(
            item.get("required_references_sha256")
            == canonical_sha256(requirement_references),
            "mandatory audit required-reference digest differs",
        )
    expected_audit_decision = (
        "invalid_calibration"
        if any(item["decision"] == "invalid_calibration" for item in audit_requirement_results)
        else "passed_nonstatistical"
    )
    _require(
        audit_decision == expected_audit_decision,
        "mandatory audit family decision differs from requirement results",
    )
    audit_gate_references = [
        {
            "kind": "threshold_independent_audit_requirement",
            "requirement_id": item["requirement_id"],
            "decision": item["decision"],
            "requirement_result_sha256": item["requirement_result_sha256"],
            "required_references_sha256": item["required_references_sha256"],
        }
        for item in audit_requirement_results
    ]
    _require(
        bool(audit_gate_references)
        and all(
            _strict_int(item.get("required_reference_count"), "audit reference count", minimum=1)
            > 0
            and bool(_plain_list(item.get("required_references"), "audit references"))
            for item in audit_requirement_results
        ),
        "mandatory audit gate has an empty requirement reference",
    )
    level_index = {(item["condition"], item["metric_id"]): item for item in levels}
    estimand_index: dict[tuple[object, object], Mapping[str, object]] = {}
    for estimand in estimands:
        for raw_metric in _plain_list(estimand.get("metrics"), "estimand metrics"):
            metric = _plain_dict(raw_metric, "estimand metric")
            estimand_index[(estimand["estimand_id"], metric["metric_id"])] = metric
    support_index = {item["metric_id"]: item for item in supports}
    mandatory: list[dict[str, object]] = []
    descriptive: list[dict[str, object]] = []
    for family in design.gate_families:
        if family.gate_family_id == "mandatory_trace_and_lifecycle_audits":
            mandatory.append(
                {
                    "gate_family_id": family.gate_family_id,
                    "mandatory": True,
                    "threshold_status": "not_applicable_nonstatistical",
                    "decision": audit_decision,
                    "references": audit_gate_references,
                }
            )
            continue
        references: list[dict[str, object]] = []
        if family.estimand_ids:
            for estimand_id in family.estimand_ids:
                for metric_id in family.metric_ids:
                    if metric_id in support_index:
                        support_source = support_index[metric_id]
                        _require(
                            support_source["estimand_id"] == estimand_id,
                            "support estimand mismatch",
                        )
                        references.append(
                            {
                                "kind": "paired_population_support_level",
                                "metric_id": metric_id,
                                "estimand_id": estimand_id,
                                "statistics_sha256": canonical_sha256(support_source["statistics"]),
                            }
                        )
                    else:
                        estimand_source = estimand_index.get((estimand_id, metric_id))
                        _require(
                            estimand_source is not None,
                            "gate matrix has no estimand result",
                        )
                        assert estimand_source is not None
                        references.append(
                            {
                                "kind": "paired_contrast",
                                "metric_id": metric_id,
                                "estimand_id": estimand_id,
                                "statistics_sha256": canonical_sha256(
                                    estimand_source["statistics"]
                                ),
                            }
                        )
        else:
            for condition in family.conditions:
                for metric_id in family.metric_ids:
                    level_source = level_index.get((condition, metric_id))
                    _require(level_source is not None, "gate matrix has no level result")
                    assert level_source is not None
                    references.append(
                        {
                            "kind": "absolute_level",
                            "metric_id": metric_id,
                            "condition": condition,
                            "statistics_sha256": canonical_sha256(level_source["statistics"]),
                        }
                    )
        result = {
            "gate_family_id": family.gate_family_id,
            "mandatory": family.mandatory,
            "threshold_status": "unset_pending_consumed_calibration_outcomes",
            "decision": "not_evaluated_no_thresholds",
            "references": references,
        }
        (mandatory if family.mandatory else descriptive).append(result)
    return mandatory, descriptive


def _validated_shard_bodies(
    shards: Sequence[Mapping[str, object]],
) -> tuple[dict[int, dict[str, object]], dict[str, object]]:
    _require(len(shards) == EXPECTED_CASES, "calibration requires exactly 240 shard inputs")
    by_case: dict[int, dict[str, object]] = {}
    common_readiness: dict[str, object] | None = None
    seen_payload_digests: set[str] = set()
    for raw in shards:
        validated = validate_calibration_case_shard(
            raw,
            expected_readiness_binding=common_readiness,
        )
        digest = validated.get("payload_sha256")
        _require(_is_sha256(digest), "validated shard lacks payload digest")
        _require(cast(str, digest) not in seen_payload_digests, "duplicate shard payload supplied")
        seen_payload_digests.add(cast(str, digest))
        body = _validate_payload_digest(validated, "case shard")
        readiness = _plain_dict(body["readiness_binding"], "readiness binding")
        if common_readiness is None:
            common_readiness = readiness
        else:
            _require(readiness == common_readiness, "shards mix readiness provenance")
        case_index = _strict_int(_plain_dict(body["case"], "case")["case_index"], "case index")
        _require(case_index not in by_case, "duplicate case index supplied")
        by_case[case_index] = {**body, "payload_sha256": digest}
    _require(set(by_case) == set(range(EXPECTED_CASES)), "case ledger is incomplete")
    _require(common_readiness is not None, "calibration has no readiness binding")
    counts = Counter(
        cast(str, _plain_dict(item["case"], "case")["manifest_name"]) for item in by_case.values()
    )
    _require(
        counts == Counter({name: EXPECTED_MANIFEST_CASES for name in CALIBRATION_MANIFEST_ORDER}),
        "manifest case counts differ from 80 each",
    )
    assert common_readiness is not None
    return by_case, common_readiness


def _validate_managed_ledger_snapshot(
    snapshot: Mapping[str, object],
    directory: Path,
    shards_by_case: Mapping[int, Mapping[str, object]],
) -> dict[str, object]:
    """Require a fresh complete inventory and exact shard/outcome joins."""

    try:
        validated = require_valid_calibration_execution_inventory(snapshot, directory)
        joined = validate_completed_calibration_ledger_snapshot(validated, shards_by_case)
    except RuntimeError as exc:
        raise CalibrationError(str(exc)) from exc
    _require(
        validated.get("schema") == CALIBRATION_EXECUTION_INVENTORY_SCHEMA,
        "execution inventory schema differs",
    )
    _require(joined == validated, "completed ledger join normalization differs")
    return joined


def aggregate_hidden_regime_factorial_calibration(
    shards: Sequence[Mapping[str, object]],
    *,
    managed_ledger_snapshot: Mapping[str, object],
    managed_ledger_directory: Path,
    aggregation_worker_provenance: Mapping[str, object],
    aggregation_zip_provenance_attestation: Mapping[str, object],
    aggregation_readiness_certification_binding: Mapping[str, object],
) -> dict[str, object]:
    """Aggregate the exact 240-case ledger without setting or evaluating thresholds."""

    design = _design()
    shards_by_case, readiness = _validated_shard_bodies(shards)
    ledger = _validate_managed_ledger_snapshot(
        managed_ledger_snapshot,
        managed_ledger_directory,
        shards_by_case,
    )
    levels = _aggregate_levels(design, shards_by_case)
    estimands = _aggregate_estimands(design, shards_by_case)
    supports = _aggregate_support_metrics(design, shards_by_case)
    certification_binding = _validate_aggregation_readiness_certification_binding(
        aggregation_readiness_certification_binding
    )
    mandatory_audit_summary = _build_mandatory_audit_summary(
        design,
        shards_by_case,
        certification_binding,
    )
    mandatory, descriptive = _gate_result_matrix(
        design,
        levels,
        estimands,
        supports,
        mandatory_audit_summary,
    )
    worker_provenance = _plain_dict(
        aggregation_worker_provenance,
        "aggregation worker provenance",
    )
    zip_provenance_attestation = _plain_dict(
        aggregation_zip_provenance_attestation,
        "aggregation ZIP provenance attestation",
    )
    finalized_by_case = {
        _strict_int(item.get("case_index"), "finalized case index"): item
        for item in (
            _plain_dict(raw, "finalized inventory record")
            for raw in _plain_list(ledger.get("finalized_records"), "finalized records")
        )
    }
    _require(set(finalized_by_case) == set(shards_by_case), "finalized case ledger differs")
    case_ledger = [
        {
            "case_index": case_index,
            "seed_index": _plain_dict(shard["case"], "case")["seed_index"],
            "condition": _plain_dict(shard["case"], "case")["condition"],
            "manifest_name": _plain_dict(shard["case"], "case")["manifest_name"],
            "case_shard_payload_sha256": shard["payload_sha256"],
            "case_request_binding_sha256": shard["case_request_binding_sha256"],
            "summary_sha256": shard["summary_sha256"],
            "resource_sha256": shard["resource_sha256"],
            "worker_provenance_sha256": canonical_sha256(shard["worker_provenance"]),
            "execution_record_binding": shard["execution_record_binding"],
            "finalized_record_sha256": finalized_by_case[case_index]["finalized_record_sha256"],
            "shard_canonical_sha256": finalized_by_case[case_index]["shard_canonical_sha256"],
            "primitive_trace_sha256": _plain_dict(shard["primitive_trace"], "trace")["sha256"],
        }
        for case_index, shard in sorted(shards_by_case.items())
    ]
    gate_payload = [family.to_payload() for family in design.gate_families]
    body = {
        "schema": CALIBRATION_AGGREGATE_SCHEMA,
        "development_only": True,
        "scientific_promotion_allowed": False,
        "claim_accepted": False,
        "thresholds_frozen": False,
        "threshold_freeze_receipt": None,
        "promotion_artifact": False,
        "protocol_payload_sha256": CALIBRATION_DESIGN_PAYLOAD_SHA256,
        "seed_snapshot_sha256": SEED_SNAPSHOT_SHA256,
        "gate_matrix_sha256": canonical_sha256(gate_payload),
        "readiness_binding": readiness,
        "managed_ledger_snapshot": ledger,
        "managed_ledger_snapshot_sha256": canonical_sha256(ledger),
        "managed_ledger_content_address": ledger["genesis_sha256"],
        "aggregation_readiness_certification_binding": certification_binding,
        "aggregation_readiness_certification_binding_sha256": canonical_sha256(
            certification_binding
        ),
        "aggregation_worker_provenance": worker_provenance,
        "aggregation_worker_provenance_sha256": canonical_sha256(worker_provenance),
        "aggregation_zip_provenance_attestation": zip_provenance_attestation,
        "aggregation_zip_provenance_attestation_sha256": canonical_sha256(
            zip_provenance_attestation
        ),
        "case_count": EXPECTED_CASES,
        "seed_pair_count": EXPECTED_SEED_PAIRS,
        "condition_count": EXPECTED_CONDITIONS,
        "case_ledger": case_ledger,
        "case_ledger_sha256": canonical_sha256(case_ledger),
        "level_summaries": levels,
        "level_summaries_sha256": canonical_sha256(levels),
        "estimand_summaries": estimands,
        "estimand_summaries_sha256": canonical_sha256(estimands),
        "paired_population_support_summaries": supports,
        "paired_population_support_summaries_sha256": canonical_sha256(supports),
        "mandatory_audit_summary": mandatory_audit_summary,
        "mandatory_audit_summary_sha256": canonical_sha256(mandatory_audit_summary),
        "mandatory_audit_decision": mandatory_audit_summary["decision"],
        "mandatory_gate_results": mandatory,
        "descriptive_only_results": descriptive,
        "gate_decision_status": (
            "mandatory_audits_passed_statistical_thresholds_unset"
            if mandatory_audit_summary["decision"] == "passed_nonstatistical"
            else "invalid_calibration_mandatory_audit_failure"
        ),
        "scipy_version_for_student_t_quantiles": scipy_version,
        "float_serialization": "canonical_python_float_hex_exact_ieee754_binary64",
        "comparison_rounding": "none_precomparison_display_rounding_forbidden",
        "claim_scope": (
            "nonpromoting consumed calibration of one finite hidden-regime signaling factorial"
        ),
    }
    return _payload_with_digest(body)


def validate_calibration_aggregate(
    payload: Mapping[str, object],
    shards: Sequence[Mapping[str, object]],
    *,
    managed_ledger_snapshot: Mapping[str, object],
    managed_ledger_directory: Path,
    aggregation_worker_provenance: Mapping[str, object],
    aggregation_zip_provenance_attestation: Mapping[str, object],
    aggregation_readiness_certification_binding: Mapping[str, object],
) -> dict[str, object]:
    """Recompute an aggregate from its exact shards and managed ledger."""

    expected = aggregate_hidden_regime_factorial_calibration(
        shards,
        managed_ledger_snapshot=managed_ledger_snapshot,
        managed_ledger_directory=managed_ledger_directory,
        aggregation_worker_provenance=aggregation_worker_provenance,
        aggregation_zip_provenance_attestation=aggregation_zip_provenance_attestation,
        aggregation_readiness_certification_binding=(
            aggregation_readiness_certification_binding
        ),
    )
    if canonical_json_bytes(dict(payload)) != canonical_json_bytes(expected):
        _fail("calibration aggregate differs from exact recomputation")
    return expected


@dataclass(frozen=True, slots=True)
class PublishedCalibrationAggregate:
    """One immutable aggregate addressed by its exact payload digest."""

    path: Path
    payload_sha256: str
    payload: dict[str, object]


def calibration_aggregate_path(publication_root: Path, payload_sha256: str) -> Path:
    _require(_is_sha256(payload_sha256), "aggregate payload digest is invalid")
    return publication_root.absolute() / f"{payload_sha256}.json"


def _load_content_addressed_calibration_aggregate(
    publication_root: Path,
    payload_sha256: str,
) -> dict[str, object]:
    """Read one immutable aggregate only from its declared content address."""

    path = calibration_aggregate_path(publication_root, payload_sha256)
    raw = _read_regular_file(
        path,
        max_bytes=_MAX_AGGREGATE_BYTES,
        label="content-addressed calibration aggregate",
    )
    payload = _strict_json(raw, "content-addressed calibration aggregate")
    body = _validate_payload_digest(payload, "content-addressed calibration aggregate")
    _require(payload["payload_sha256"] == payload_sha256, "aggregate content address differs")
    _require(body.get("schema") == CALIBRATION_AGGREGATE_SCHEMA, "aggregate schema differs")
    _require(body.get("development_only") is True, "aggregate is not development-only")
    _require(
        body.get("scientific_promotion_allowed") is False,
        "aggregate permits scientific promotion",
    )
    _require(body.get("claim_accepted") is False, "aggregate accepts a claim")
    _require(body.get("thresholds_frozen") is False, "aggregate already freezes thresholds")
    _require(body.get("promotion_artifact") is False, "aggregate is a promotion artifact")
    return payload


def _install_verified_aggregate_worker_output_new_only(
    publication_root: Path,
    payload: dict[str, object],
    raw: bytes,
) -> PublishedCalibrationAggregate:
    """Install exact post-bootstrap worker bytes without semantic checkout re-evaluation."""

    _require(raw == canonical_json_bytes(payload), "verified aggregate worker bytes differ")
    payload_sha256 = cast(str, payload["payload_sha256"])
    path = calibration_aggregate_path(publication_root, payload_sha256)
    _require(len(raw) <= _MAX_AGGREGATE_BYTES, "calibration aggregate exceeds maximum size")
    try:
        _write_new_immutable(
            path.parent,
            path.name,
            raw,
            max_bytes=_MAX_AGGREGATE_BYTES,
            label="calibration aggregate",
        )
    except FileExistsError:
        existing_raw = _read_regular_file(
            path,
            max_bytes=_MAX_AGGREGATE_BYTES,
            label="calibration aggregate",
        )
        _require(existing_raw == raw, "duplicate aggregate is not byte-identical")
    installed_raw = _read_regular_file(
        path,
        max_bytes=_MAX_AGGREGATE_BYTES,
        label="installed calibration aggregate",
    )
    _require(installed_raw == raw, "installed aggregate bytes differ from verified worker output")
    return PublishedCalibrationAggregate(path, payload_sha256, payload)


def _require_disjoint_aggregate_publication_root(
    aggregate_publication_root: Path,
    *,
    readiness_directory: Path,
    shard_publication_root: Path,
    managed_ledger_directory: Path,
) -> None:
    """Keep aggregate installation outside every exact input/inventory tree."""

    descriptor, aggregate_root = _open_directory_without_symlinks(
        aggregate_publication_root,
        label="aggregate publication root",
    )
    os.close(descriptor)
    normalized_aggregate = Path(os.path.realpath(aggregate_root))
    for label, raw_path in (
        ("readiness publication", readiness_directory),
        ("shard publication", shard_publication_root),
        ("managed ledger", managed_ledger_directory),
    ):
        normalized_input = Path(os.path.realpath(raw_path.absolute()))
        overlaps = (
            normalized_aggregate == normalized_input
            or normalized_aggregate.is_relative_to(normalized_input)
            or normalized_input.is_relative_to(normalized_aggregate)
        )
        _require(not overlaps, f"aggregate publication root overlaps the {label} tree")


def aggregate_and_publish_completed_calibration(
    *,
    readiness_directory: Path,
    shard_publication_root: Path,
    managed_ledger_directory: Path,
    aggregate_publication_root: Path,
    authorize_publication: bool = False,
    timeout_seconds: int | None = None,
) -> PublishedCalibrationAggregate:
    """Compute and publish only through the receipt's isolated source ZIP."""

    _require(authorize_publication is True, "aggregate publication requires authorization")
    if timeout_seconds is not None:
        _strict_int(timeout_seconds, "timeout_seconds", minimum=1)
    bundle = load_validated_readiness_bundle(
        readiness_directory,
        recheck_current=False,
        recheck_runtime=False,
    )
    _require_disjoint_aggregate_publication_root(
        aggregate_publication_root,
        readiness_directory=readiness_directory,
        shard_publication_root=shard_publication_root,
        managed_ledger_directory=managed_ledger_directory,
    )
    completed = execute_bound_calibration_worker(
        readiness_directory,
        (
            "--worker-aggregate-v1",
            readiness_directory.absolute().as_posix(),
            managed_ledger_directory.absolute().as_posix(),
            shard_publication_root.absolute().as_posix(),
        ),
        authorize_calibration_execution=True,
        timeout_seconds=timeout_seconds,
    )
    if completed.returncode != 0:
        stderr_digest = hashlib.sha256(completed.stderr).hexdigest()
        raise CalibrationError(
            "isolated calibration aggregation failed; "
            f"returncode={completed.returncode},stderr_bytes={len(completed.stderr)},"
            f"stderr_sha256={stderr_digest}"
        )
    payload = _parse_aggregate_worker_result(completed.stdout)
    body = _validate_payload_digest(payload, "aggregate worker result")
    _validate_aggregate_provenance_bindings(body, bundle)
    readiness = _plain_dict(body["readiness_binding"], "aggregate readiness binding")
    _require(
        readiness == _readiness_binding(bundle),
        "aggregate worker readiness binding differs",
    )
    _require(
        body["managed_ledger_content_address"] == bundle.execution_genesis_sha256,
        "aggregate worker ledger binding differs",
    )
    readiness_body = _plain_dict(bundle.payload.get("body"), "readiness body")
    require_current_full_runtime_identity(readiness_body.get("runtime_identity"))
    _require_disjoint_aggregate_publication_root(
        aggregate_publication_root,
        readiness_directory=readiness_directory,
        shard_publication_root=shard_publication_root,
        managed_ledger_directory=managed_ledger_directory,
    )
    raw = canonical_json_bytes(payload)
    return _install_verified_aggregate_worker_output_new_only(
        aggregate_publication_root,
        payload,
        raw,
    )


@dataclass(frozen=True, slots=True)
class PublishedThresholdFreezeReceipt:
    """One immutable threshold decision addressed by its exact receipt digest."""

    path: Path
    receipt_payload_sha256: str
    payload: dict[str, object]


def threshold_freeze_receipt_path(
    publication_root: Path,
    receipt_payload_sha256: str,
) -> Path:
    _require(_is_sha256(receipt_payload_sha256), "threshold receipt digest is invalid")
    return publication_root.absolute() / f"{receipt_payload_sha256}.json"


def _load_content_addressed_threshold_freeze_receipt(
    publication_root: Path,
    receipt_payload_sha256: str,
) -> dict[str, object]:
    """Read one immutable threshold receipt only from its declared content address."""

    path = threshold_freeze_receipt_path(publication_root, receipt_payload_sha256)
    raw = _read_regular_file(
        path,
        max_bytes=_MAX_RECEIPT_BYTES,
        label="content-addressed threshold-freeze receipt",
    )
    payload = _strict_json(raw, "content-addressed threshold-freeze receipt")
    _threshold_freeze_receipt_body(
        payload,
        label="content-addressed threshold-freeze receipt",
    )
    _require(
        payload.get("receipt_payload_sha256") == receipt_payload_sha256,
        "threshold receipt content address differs",
    )
    return payload


def _install_verified_threshold_freeze_receipt_new_only(
    publication_root: Path,
    payload: Mapping[str, object],
    raw: bytes,
) -> PublishedThresholdFreezeReceipt:
    """Install only canonical bytes already checked against the exact aggregate."""

    normalized = dict(payload)
    _require(raw == canonical_json_bytes(normalized), "verified threshold receipt bytes differ")
    _threshold_freeze_receipt_body(normalized, label="verified threshold-freeze receipt")
    digest = normalized.get("receipt_payload_sha256")
    _require(_is_sha256(digest), "threshold receipt payload digest is invalid")
    path = threshold_freeze_receipt_path(publication_root, cast(str, digest))
    _require(len(raw) <= _MAX_RECEIPT_BYTES, "threshold-freeze receipt exceeds maximum size")
    try:
        _write_new_immutable(
            path.parent,
            path.name,
            raw,
            max_bytes=_MAX_RECEIPT_BYTES,
            label="threshold-freeze receipt",
        )
    except FileExistsError as exc:
        raise FileExistsError(
            f"refusing to overwrite threshold-freeze receipt: {path}"
        ) from exc
    installed_raw = _read_regular_file(
        path,
        max_bytes=_MAX_RECEIPT_BYTES,
        label="installed threshold-freeze receipt",
    )
    _require(installed_raw == raw, "installed threshold receipt bytes differ")
    return PublishedThresholdFreezeReceipt(path, cast(str, digest), normalized)


@dataclass(frozen=True, slots=True)
class PublishedCertifiedProtectedPlan:
    """Certified nonauthorizing plan bytes installed at one immutable content address."""

    path: Path
    payload_sha256: str
    payload: dict[str, object]


def certified_protected_plan_path(
    publication_root: Path,
    payload_sha256: str,
) -> Path:
    _require(_is_sha256(payload_sha256), "protected plan payload digest is invalid")
    return publication_root.absolute() / f"{payload_sha256}.json"


def _install_verified_protected_plan_new_only(
    publication_root: Path,
    payload: Mapping[str, object],
    raw: bytes,
) -> PublishedCertifiedProtectedPlan:
    """Install exact certified worker bytes without deriving a seed in the checkout."""

    normalized = dict(payload)
    _require(raw == canonical_json_bytes(normalized), "verified protected plan bytes differ")
    _protected_plan_body(normalized, label="verified protected plan")
    digest = normalized.get("payload_sha256")
    _require(_is_sha256(digest), "protected plan payload digest is invalid")
    path = certified_protected_plan_path(publication_root, cast(str, digest))
    _require(len(raw) <= _MAX_PROTECTED_PLAN_BYTES, "protected plan exceeds maximum size")
    try:
        _write_new_immutable(
            path.parent,
            path.name,
            raw,
            max_bytes=_MAX_PROTECTED_PLAN_BYTES,
            label="certified protected plan",
        )
    except FileExistsError as exc:
        raise FileExistsError(
            f"refusing to overwrite or reuse a protected plan: {path}"
        ) from exc
    installed_raw = _read_regular_file(
        path,
        max_bytes=_MAX_PROTECTED_PLAN_BYTES,
        label="installed certified protected plan",
    )
    _require(installed_raw == raw, "installed protected plan bytes differ")
    return PublishedCertifiedProtectedPlan(path, cast(str, digest), normalized)


def _require_disjoint_threshold_freeze_publication_root(
    threshold_receipt_publication_root: Path,
    *,
    readiness_directory: Path,
    shard_publication_root: Path,
    managed_ledger_directory: Path,
    aggregate_publication_root: Path,
) -> None:
    """Keep the final receipt outside every immutable input and inventory tree."""

    descriptor, threshold_root = _open_directory_without_symlinks(
        threshold_receipt_publication_root,
        label="threshold receipt publication root",
    )
    os.close(descriptor)
    normalized_threshold = Path(os.path.realpath(threshold_root))
    for label, raw_path in (
        ("readiness publication", readiness_directory),
        ("shard publication", shard_publication_root),
        ("managed ledger", managed_ledger_directory),
        ("aggregate publication", aggregate_publication_root),
    ):
        normalized_input = Path(os.path.realpath(raw_path.absolute()))
        overlaps = (
            normalized_threshold == normalized_input
            or normalized_threshold.is_relative_to(normalized_input)
            or normalized_input.is_relative_to(normalized_threshold)
        )
        _require(not overlaps, f"threshold receipt publication root overlaps the {label} tree")


def _require_disjoint_protected_plan_publication_root(
    protected_plan_publication_root: Path,
    *,
    readiness_directory: Path,
    shard_publication_root: Path,
    managed_ledger_directory: Path,
    aggregate_publication_root: Path,
    threshold_receipt_publication_root: Path,
) -> None:
    """Keep plan publication outside all certified inputs, including the receipt root."""

    plan_descriptor, plan_root = _open_directory_without_symlinks(
        protected_plan_publication_root,
        label="protected plan publication root",
    )
    os.close(plan_descriptor)
    normalized_plan = Path(os.path.realpath(plan_root))
    for label, raw_path in (
        ("readiness publication", readiness_directory),
        ("shard publication", shard_publication_root),
        ("managed ledger", managed_ledger_directory),
        ("aggregate publication", aggregate_publication_root),
        ("threshold receipt publication", threshold_receipt_publication_root),
    ):
        descriptor, normalized_input = _open_directory_without_symlinks(
            raw_path,
            label=label,
        )
        os.close(descriptor)
        normalized_input = Path(os.path.realpath(normalized_input))
        overlaps = (
            normalized_plan == normalized_input
            or normalized_plan.is_relative_to(normalized_input)
            or normalized_input.is_relative_to(normalized_plan)
        )
        _require(not overlaps, f"protected plan publication root overlaps the {label} tree")


@contextmanager
def _threshold_input_publication_guard(
    *,
    shard_publication_root: Path,
    readiness_receipt_sha256: str,
    managed_ledger_directory: Path,
) -> Iterator[None]:
    """Hold cooperative shard/ledger directory locks through parent recheck and install."""

    _require(_is_sha256(readiness_receipt_sha256), "guard readiness digest is invalid")
    directories = [
        (
            shard_publication_root.absolute() / readiness_receipt_sha256,
            "threshold input shard directory",
        ),
        (managed_ledger_directory.absolute(), "threshold input managed ledger"),
        (
            managed_ledger_directory.absolute() / "cases",
            "threshold input managed cases directory",
        ),
        *[
            (
                managed_ledger_directory.absolute()
                / "cases"
                / f"case-{case_index:03d}",
                f"threshold input managed case {case_index}",
            )
            for case_index in range(EXPECTED_CASES)
        ],
    ]
    descriptors: list[int] = []
    try:
        for path, label in directories:
            descriptor, _ = _open_directory_without_symlinks(path, label=label)
            try:
                fcntl.flock(descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                os.close(descriptor)
                raise CalibrationError(
                    f"{label} is locked by an active execution or mutation"
                ) from exc
            except BaseException:
                os.close(descriptor)
                raise
            descriptors.append(descriptor)
        yield
    finally:
        for descriptor in reversed(descriptors):
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)


def _validate_threshold_worker_provenance_bindings(
    result_body: Mapping[str, object],
    bundle: ValidatedReadinessBundle,
) -> None:
    """Validate the threshold worker with the same exact ZIP provenance contract."""

    proxy = {
        "aggregation_readiness_certification_binding": result_body.get(
            "threshold_worker_readiness_certification_binding"
        ),
        "aggregation_readiness_certification_binding_sha256": result_body.get(
            "threshold_worker_readiness_certification_binding_sha256"
        ),
        "aggregation_worker_provenance": result_body.get("threshold_worker_provenance"),
        "aggregation_worker_provenance_sha256": result_body.get(
            "threshold_worker_provenance_sha256"
        ),
        "aggregation_zip_provenance_attestation": result_body.get(
            "threshold_zip_provenance_attestation"
        ),
        "aggregation_zip_provenance_attestation_sha256": result_body.get(
            "threshold_zip_provenance_attestation_sha256"
        ),
    }
    _validate_aggregate_provenance_bindings(proxy, bundle)


def _validate_protected_plan_worker_provenance_bindings(
    result_body: Mapping[str, object],
    bundle: ValidatedReadinessBundle,
) -> None:
    """Validate protected-plan worker identity against the exact current ZIP receipt."""

    proxy = {
        "aggregation_readiness_certification_binding": result_body.get(
            "worker_readiness_certification_binding"
        ),
        "aggregation_readiness_certification_binding_sha256": result_body.get(
            "worker_readiness_certification_binding_sha256"
        ),
        "aggregation_worker_provenance": result_body.get("worker_provenance"),
        "aggregation_worker_provenance_sha256": result_body.get(
            "worker_provenance_sha256"
        ),
        "aggregation_zip_provenance_attestation": result_body.get(
            "zip_provenance_attestation"
        ),
        "aggregation_zip_provenance_attestation_sha256": result_body.get(
            "zip_provenance_attestation_sha256"
        ),
    }
    _validate_aggregate_provenance_bindings(proxy, bundle)


def _verify_and_install_threshold_freeze_worker_result(
    *,
    result: Mapping[str, object],
    bundle: ValidatedReadinessBundle,
    readiness_directory: Path,
    shard_publication_root: Path,
    managed_ledger_directory: Path,
    aggregate_publication_root: Path,
    aggregate_payload_sha256: str,
    threshold_receipt_publication_root: Path,
) -> PublishedThresholdFreezeReceipt:
    """Recheck all live identities under locks before the one allowed install."""

    result_body = _validate_payload_digest(result, "threshold-freeze worker result")
    _require(
        result_body["calibration_aggregate_payload_sha256"] == aggregate_payload_sha256,
        "threshold worker result binds another aggregate",
    )
    _require(
        result_body["readiness_receipt_sha256"] == bundle.receipt_sha256,
        "threshold worker result binds another readiness receipt",
    )
    aggregate = _load_content_addressed_calibration_aggregate(
        aggregate_publication_root,
        aggregate_payload_sha256,
    )
    aggregate_body = _validate_payload_digest(aggregate, "calibration aggregate")
    _validate_aggregate_provenance_bindings(aggregate_body, bundle)
    readiness = _readiness_binding(bundle)
    _require(
        _plain_dict(aggregate_body.get("readiness_binding"), "aggregate readiness binding")
        == readiness,
        "threshold parent aggregate readiness binding differs",
    )
    _require(
        aggregate_body.get("managed_ledger_content_address")
        == bundle.execution_genesis_sha256,
        "threshold parent aggregate ledger binding differs",
    )
    _validate_threshold_worker_provenance_bindings(result_body, bundle)

    current_shards = load_complete_calibration_case_shards(
        shard_publication_root,
        expected_readiness_binding=readiness,
        managed_ledger_directory=managed_ledger_directory,
    )
    current_inventory = snapshot_calibration_execution_inventory(managed_ledger_directory)
    current_input_binding = _threshold_freeze_exact_input_binding(
        aggregate,
        current_shards,
        current_inventory,
    )
    worker_input_binding = _plain_dict(
        result_body["threshold_exact_input_binding"],
        "threshold worker exact input binding",
    )
    _require(
        canonical_json_bytes(current_input_binding) == canonical_json_bytes(worker_input_binding),
        "threshold inputs changed after certified worker recomputation",
    )

    receipt = _plain_dict(result_body["threshold_freeze_receipt"], "threshold-freeze receipt")
    try:
        validated_receipt = validate_hidden_regime_factorial_threshold_freeze_receipt(
            receipt,
            calibration_aggregate=aggregate,
        )
    except ThresholdFreezeError as exc:
        raise CalibrationError(str(exc)) from exc
    _require(
        canonical_json_bytes(validated_receipt) == canonical_json_bytes(receipt),
        "threshold parent receipt validation changed worker bytes",
    )
    readiness_body = _plain_dict(bundle.payload.get("body"), "readiness body")
    require_current_full_runtime_identity(readiness_body.get("runtime_identity"))
    _require_disjoint_aggregate_publication_root(
        aggregate_publication_root,
        readiness_directory=readiness_directory,
        shard_publication_root=shard_publication_root,
        managed_ledger_directory=managed_ledger_directory,
    )
    _require_disjoint_threshold_freeze_publication_root(
        threshold_receipt_publication_root,
        readiness_directory=readiness_directory,
        shard_publication_root=shard_publication_root,
        managed_ledger_directory=managed_ledger_directory,
        aggregate_publication_root=aggregate_publication_root,
    )
    aggregate_after = _load_content_addressed_calibration_aggregate(
        aggregate_publication_root,
        aggregate_payload_sha256,
    )
    current_shards_after = load_complete_calibration_case_shards(
        shard_publication_root,
        expected_readiness_binding=readiness,
        managed_ledger_directory=managed_ledger_directory,
    )
    current_inventory_after = snapshot_calibration_execution_inventory(
        managed_ledger_directory
    )
    current_input_binding_after = _threshold_freeze_exact_input_binding(
        aggregate_after,
        current_shards_after,
        current_inventory_after,
    )
    _require(
        canonical_json_bytes(aggregate_after) == canonical_json_bytes(aggregate),
        "content-addressed aggregate changed before threshold receipt installation",
    )
    _require(
        canonical_json_bytes(current_input_binding_after)
        == canonical_json_bytes(current_input_binding),
        "shard or managed-ledger input changed before threshold receipt installation",
    )
    raw = canonical_json_bytes(validated_receipt)
    return _install_verified_threshold_freeze_receipt_new_only(
        threshold_receipt_publication_root,
        validated_receipt,
        raw,
    )


def freeze_and_publish_completed_calibration_thresholds(
    *,
    readiness_directory: Path,
    shard_publication_root: Path,
    managed_ledger_directory: Path,
    aggregate_publication_root: Path,
    aggregate_payload_sha256: str,
    threshold_receipt_publication_root: Path,
    authorize_publication: bool = False,
    timeout_seconds: int | None = None,
) -> PublishedThresholdFreezeReceipt:
    """Exact-recompute, decide, recheck, and publish one threshold receipt."""

    _require(authorize_publication is True, "threshold receipt publication requires authorization")
    _require(_is_sha256(aggregate_payload_sha256), "aggregate payload digest is invalid")
    if timeout_seconds is not None:
        _strict_int(timeout_seconds, "timeout_seconds", minimum=1)
    bundle = load_validated_readiness_bundle(
        readiness_directory,
        recheck_current=False,
        recheck_runtime=False,
    )
    _require_disjoint_aggregate_publication_root(
        aggregate_publication_root,
        readiness_directory=readiness_directory,
        shard_publication_root=shard_publication_root,
        managed_ledger_directory=managed_ledger_directory,
    )
    _require_disjoint_threshold_freeze_publication_root(
        threshold_receipt_publication_root,
        readiness_directory=readiness_directory,
        shard_publication_root=shard_publication_root,
        managed_ledger_directory=managed_ledger_directory,
        aggregate_publication_root=aggregate_publication_root,
    )
    completed = execute_bound_calibration_worker(
        readiness_directory,
        (
            "--worker-threshold-freeze-v1",
            readiness_directory.absolute().as_posix(),
            managed_ledger_directory.absolute().as_posix(),
            shard_publication_root.absolute().as_posix(),
            aggregate_publication_root.absolute().as_posix(),
            aggregate_payload_sha256,
        ),
        authorize_calibration_execution=True,
        timeout_seconds=timeout_seconds,
    )
    if completed.returncode != 0:
        stderr_digest = hashlib.sha256(completed.stderr).hexdigest()
        raise CalibrationError(
            "isolated calibration threshold freezing failed without publication; "
            f"returncode={completed.returncode},stderr_bytes={len(completed.stderr)},"
            f"stderr_sha256={stderr_digest}"
        )
    result = _parse_threshold_freeze_worker_result(completed.stdout)
    with _threshold_input_publication_guard(
        shard_publication_root=shard_publication_root,
        readiness_receipt_sha256=bundle.receipt_sha256,
        managed_ledger_directory=managed_ledger_directory,
    ):
        return _verify_and_install_threshold_freeze_worker_result(
            result=result,
            bundle=bundle,
            readiness_directory=readiness_directory,
            shard_publication_root=shard_publication_root,
            managed_ledger_directory=managed_ledger_directory,
            aggregate_publication_root=aggregate_publication_root,
            aggregate_payload_sha256=aggregate_payload_sha256,
            threshold_receipt_publication_root=threshold_receipt_publication_root,
        )


def _require_disjoint_protected_plan_workflow_roots(
    *,
    readiness_directory: Path,
    shard_publication_root: Path,
    managed_ledger_directory: Path,
    aggregate_publication_root: Path,
    threshold_receipt_publication_root: Path,
    protected_plan_publication_root: Path,
) -> None:
    """Re-resolve every publication-root separation required by protected planning."""

    _require_disjoint_aggregate_publication_root(
        aggregate_publication_root,
        readiness_directory=readiness_directory,
        shard_publication_root=shard_publication_root,
        managed_ledger_directory=managed_ledger_directory,
    )
    _require_disjoint_threshold_freeze_publication_root(
        threshold_receipt_publication_root,
        readiness_directory=readiness_directory,
        shard_publication_root=shard_publication_root,
        managed_ledger_directory=managed_ledger_directory,
        aggregate_publication_root=aggregate_publication_root,
    )
    _require_disjoint_protected_plan_publication_root(
        protected_plan_publication_root,
        readiness_directory=readiness_directory,
        shard_publication_root=shard_publication_root,
        managed_ledger_directory=managed_ledger_directory,
        aggregate_publication_root=aggregate_publication_root,
        threshold_receipt_publication_root=threshold_receipt_publication_root,
    )


def _load_and_validate_protected_plan_live_inputs(
    *,
    result_body: Mapping[str, object],
    bundle: ValidatedReadinessBundle,
    shard_publication_root: Path,
    managed_ledger_directory: Path,
    aggregate_publication_root: Path,
    aggregate_payload_sha256: str,
    threshold_receipt_publication_root: Path,
    threshold_receipt_payload_sha256: str,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    """Reload and bind one complete live aggregate/receipt/shard/ledger snapshot."""

    aggregate = _load_content_addressed_calibration_aggregate(
        aggregate_publication_root,
        aggregate_payload_sha256,
    )
    aggregate_body = _validate_payload_digest(aggregate, "protected-plan calibration aggregate")
    _validate_aggregate_provenance_bindings(aggregate_body, bundle)
    readiness = _readiness_binding(bundle)
    _require(
        _plain_dict(aggregate_body.get("readiness_binding"), "aggregate readiness binding")
        == readiness,
        "protected-plan parent aggregate readiness binding differs",
    )
    _require(
        aggregate_body.get("managed_ledger_content_address")
        == bundle.execution_genesis_sha256,
        "protected-plan parent aggregate ledger binding differs",
    )
    receipt = _load_content_addressed_threshold_freeze_receipt(
        threshold_receipt_publication_root,
        threshold_receipt_payload_sha256,
    )
    try:
        validated_receipt = validate_hidden_regime_factorial_threshold_freeze_receipt(
            receipt,
            calibration_aggregate=aggregate,
        )
    except (ThresholdFreezeError, TypeError, ValueError) as exc:
        raise CalibrationError("protected-plan threshold receipt validation failed") from exc
    _require(
        canonical_json_bytes(validated_receipt) == canonical_json_bytes(receipt),
        "protected-plan threshold validation changed persisted receipt bytes",
    )
    _successful_threshold_freeze_receipt_body(
        validated_receipt,
        label="protected-plan parent threshold receipt",
    )
    shards = load_complete_calibration_case_shards(
        shard_publication_root,
        expected_readiness_binding=readiness,
        managed_ledger_directory=managed_ledger_directory,
    )
    inventory = snapshot_calibration_execution_inventory(managed_ledger_directory)
    exact_input_binding = _threshold_freeze_exact_input_binding(
        aggregate,
        shards,
        inventory,
    )
    _require(
        exact_input_binding.get("calibration_aggregate_payload_sha256")
        == result_body.get("calibration_aggregate_payload_sha256"),
        "protected-plan live inputs bind another aggregate",
    )
    return aggregate, validated_receipt, exact_input_binding


def _verify_and_install_protected_plan_worker_result(
    *,
    result: Mapping[str, object],
    bundle: ValidatedReadinessBundle,
    readiness_directory: Path,
    shard_publication_root: Path,
    managed_ledger_directory: Path,
    aggregate_publication_root: Path,
    aggregate_payload_sha256: str,
    threshold_receipt_publication_root: Path,
    threshold_receipt_payload_sha256: str,
    protected_plan_publication_root: Path,
) -> PublishedCertifiedProtectedPlan:
    """Twice reload all live inputs, then install only certified plan bytes."""

    result_payload = _validate_protected_plan_worker_result_payload(result)
    result_body = _validate_payload_digest(result_payload, "protected-plan worker result")
    _require(
        result_body["calibration_aggregate_payload_sha256"] == aggregate_payload_sha256,
        "protected-plan worker result binds another aggregate",
    )
    _require(
        result_body["threshold_freeze_receipt_payload_sha256"]
        == threshold_receipt_payload_sha256,
        "protected-plan worker result binds another threshold receipt",
    )
    _require(
        result_body["readiness_receipt_sha256"] == bundle.receipt_sha256,
        "protected-plan worker result binds another readiness receipt",
    )
    _validate_protected_plan_worker_provenance_bindings(result_body, bundle)
    plan = _plain_dict(result_body["protected_plan"], "certified protected plan")
    plan_body = _protected_plan_body(plan, label="certified protected plan")

    aggregate, receipt, exact_input_binding = _load_and_validate_protected_plan_live_inputs(
        result_body=result_body,
        bundle=bundle,
        shard_publication_root=shard_publication_root,
        managed_ledger_directory=managed_ledger_directory,
        aggregate_publication_root=aggregate_publication_root,
        aggregate_payload_sha256=aggregate_payload_sha256,
        threshold_receipt_publication_root=threshold_receipt_publication_root,
        threshold_receipt_payload_sha256=threshold_receipt_payload_sha256,
    )
    worker_input_binding = _plain_dict(
        result_body["exact_input_binding"],
        "protected-plan worker exact input binding",
    )
    _require(
        canonical_json_bytes(exact_input_binding) == canonical_json_bytes(worker_input_binding),
        "protected-plan inputs changed after certified worker recomputation",
    )
    _validate_protected_plan_bindings(
        plan_body,
        protected_plan=plan,
        threshold_receipt=receipt,
        calibration_aggregate=aggregate,
    )
    readiness_body = _plain_dict(bundle.payload.get("body"), "readiness body")
    require_current_full_runtime_identity(readiness_body.get("runtime_identity"))
    _require_disjoint_protected_plan_workflow_roots(
        readiness_directory=readiness_directory,
        shard_publication_root=shard_publication_root,
        managed_ledger_directory=managed_ledger_directory,
        aggregate_publication_root=aggregate_publication_root,
        threshold_receipt_publication_root=threshold_receipt_publication_root,
        protected_plan_publication_root=protected_plan_publication_root,
    )

    bundle_after = load_validated_readiness_bundle(
        readiness_directory,
        recheck_current=False,
        recheck_runtime=False,
    )
    _require(
        canonical_json_bytes(bundle_after.payload) == canonical_json_bytes(bundle.payload)
        and bundle_after.receipt_sha256 == bundle.receipt_sha256
        and bundle_after.source_archive_sha256 == bundle.source_archive_sha256
        and bundle_after.source_manifest_sha256 == bundle.source_manifest_sha256
        and bundle_after.runtime_identity_sha256 == bundle.runtime_identity_sha256
        and bundle_after.execution_genesis_sha256 == bundle.execution_genesis_sha256,
        "protected-plan readiness bundle changed before publication",
    )
    _validate_protected_plan_worker_provenance_bindings(result_body, bundle_after)
    aggregate_after, receipt_after, exact_input_binding_after = (
        _load_and_validate_protected_plan_live_inputs(
            result_body=result_body,
            bundle=bundle_after,
            shard_publication_root=shard_publication_root,
            managed_ledger_directory=managed_ledger_directory,
            aggregate_publication_root=aggregate_publication_root,
            aggregate_payload_sha256=aggregate_payload_sha256,
            threshold_receipt_publication_root=threshold_receipt_publication_root,
            threshold_receipt_payload_sha256=threshold_receipt_payload_sha256,
        )
    )
    _require(
        canonical_json_bytes(aggregate_after) == canonical_json_bytes(aggregate),
        "content-addressed aggregate changed before protected plan installation",
    )
    _require(
        canonical_json_bytes(receipt_after) == canonical_json_bytes(receipt),
        "content-addressed threshold receipt changed before protected plan installation",
    )
    _require(
        canonical_json_bytes(exact_input_binding_after)
        == canonical_json_bytes(exact_input_binding),
        "shard or managed-ledger input changed before protected plan installation",
    )
    _validate_protected_plan_bindings(
        plan_body,
        protected_plan=plan,
        threshold_receipt=receipt_after,
        calibration_aggregate=aggregate_after,
    )
    require_current_full_runtime_identity(
        _plain_dict(bundle_after.payload.get("body"), "readiness body").get(
            "runtime_identity"
        )
    )
    _require_disjoint_protected_plan_workflow_roots(
        readiness_directory=readiness_directory,
        shard_publication_root=shard_publication_root,
        managed_ledger_directory=managed_ledger_directory,
        aggregate_publication_root=aggregate_publication_root,
        threshold_receipt_publication_root=threshold_receipt_publication_root,
        protected_plan_publication_root=protected_plan_publication_root,
    )
    raw = canonical_json_bytes(plan)
    return _install_verified_protected_plan_new_only(
        protected_plan_publication_root,
        plan,
        raw,
    )


def derive_and_publish_completed_calibration_protected_plan(
    *,
    readiness_directory: Path,
    shard_publication_root: Path,
    managed_ledger_directory: Path,
    aggregate_publication_root: Path,
    aggregate_payload_sha256: str,
    threshold_receipt_publication_root: Path,
    threshold_receipt_payload_sha256: str,
    protected_plan_publication_root: Path,
    authorize_publication: bool = False,
    timeout_seconds: int | None = None,
) -> PublishedCertifiedProtectedPlan:
    """Derive in the certified ZIP and publish one strictly nonauthorizing plan."""

    _require(authorize_publication is True, "protected plan publication requires authorization")
    _require(_is_sha256(aggregate_payload_sha256), "aggregate payload digest is invalid")
    _require(
        _is_sha256(threshold_receipt_payload_sha256),
        "threshold receipt payload digest is invalid",
    )
    if timeout_seconds is not None:
        _strict_int(timeout_seconds, "timeout_seconds", minimum=1)
    bundle = load_validated_readiness_bundle(
        readiness_directory,
        recheck_current=False,
        recheck_runtime=False,
    )
    _require_disjoint_protected_plan_workflow_roots(
        readiness_directory=readiness_directory,
        shard_publication_root=shard_publication_root,
        managed_ledger_directory=managed_ledger_directory,
        aggregate_publication_root=aggregate_publication_root,
        threshold_receipt_publication_root=threshold_receipt_publication_root,
        protected_plan_publication_root=protected_plan_publication_root,
    )
    completed = execute_bound_calibration_worker(
        readiness_directory,
        (
            "--worker-protected-plan-v1",
            readiness_directory.absolute().as_posix(),
            managed_ledger_directory.absolute().as_posix(),
            shard_publication_root.absolute().as_posix(),
            aggregate_publication_root.absolute().as_posix(),
            aggregate_payload_sha256,
            threshold_receipt_publication_root.absolute().as_posix(),
            threshold_receipt_payload_sha256,
        ),
        authorize_calibration_execution=True,
        timeout_seconds=timeout_seconds,
    )
    if completed.returncode != 0:
        stderr_digest = hashlib.sha256(completed.stderr).hexdigest()
        raise CalibrationError(
            "isolated protected-plan derivation failed without publication; "
            f"returncode={completed.returncode},stderr_bytes={len(completed.stderr)},"
            f"stderr_sha256={stderr_digest}"
        )
    result = _parse_protected_plan_worker_result(completed.stdout)
    with _threshold_input_publication_guard(
        shard_publication_root=shard_publication_root,
        readiness_receipt_sha256=bundle.receipt_sha256,
        managed_ledger_directory=managed_ledger_directory,
    ):
        return _verify_and_install_protected_plan_worker_result(
            result=result,
            bundle=bundle,
            readiness_directory=readiness_directory,
            shard_publication_root=shard_publication_root,
            managed_ledger_directory=managed_ledger_directory,
            aggregate_publication_root=aggregate_publication_root,
            aggregate_payload_sha256=aggregate_payload_sha256,
            threshold_receipt_publication_root=threshold_receipt_publication_root,
            threshold_receipt_payload_sha256=threshold_receipt_payload_sha256,
            protected_plan_publication_root=protected_plan_publication_root,
        )
