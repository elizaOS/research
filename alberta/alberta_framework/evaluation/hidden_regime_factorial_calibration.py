"""Fail-closed execution and aggregation for hidden-regime calibration.

This module consumes the frozen, development-only factorial protocol.  It does
not execute anything on import and it does not contain a protected-evaluation
path.  A case can run only through the separately authorized, content-addressed
readiness ZIP.  The 16,528-row primitive trace is audited while ephemeral; the
persisted shard contains exact hexadecimal outcome values, complete compact
metric sources, and audit digests, but never the raw trace.

Calibration shards and aggregates are nonpromoting development records.  They
cannot freeze thresholds, accept a claim, or support scientific promotion.
"""

from __future__ import annotations

import base64
import concurrent.futures
import dataclasses
import errno
import hashlib
import json
import math
import os
import stat
import sys
import zipfile
import zipimport
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn, cast

import numpy as np
from scipy import __version__ as scipy_version
from scipy.stats import t as student_t

from alberta_framework.core.slot_signaling_agent import SlotSignalingConfig
from alberta_framework.evaluation.hidden_regime_calibration_readiness import (
    ReadinessError,
    ValidatedReadinessBundle,
    execute_bound_calibration_worker,
    require_validated_readiness_receipt,
    validate_published_readiness_receipt,
)
from alberta_framework.evaluation.hidden_regime_execution_governance import (
    CALIBRATION_EXECUTION_INVENTORY_SCHEMA,
    CALIBRATION_EXECUTION_OUTCOME_DIGEST_SCHEMA,
    CALIBRATION_EXECUTION_PRIMITIVE_TRACE_DIGEST_SCHEMA,
    CALIBRATION_EXECUTION_RESOURCE_DIGEST_SCHEMA,
    CALIBRATION_EXECUTION_SUMMARY_DIGEST_SCHEMA,
    EXECUTION_AUTHORIZATION_ACKNOWLEDGEMENT,
    READINESS_EXECUTION_GOVERNANCE_FIELD,
    PublishedCalibrationExecutionLedger,
    build_calibration_execution_genesis,
    calibration_execution_configuration_sha256,
    calibration_execution_primitive_trace_sha256,
    calibration_execution_resource_sha256,
    calibration_execution_summary_sha256,
    initialize_calibration_execution_ledger,
    issue_calibration_execution_authorization,
    require_valid_calibration_execution_inventory,
    snapshot_calibration_execution_inventory,
    validate_completed_calibration_ledger_snapshot,
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
    EstimandContract,
    HiddenRegimeFactorialCalibrationDesign,
    MatchedCalibrationCase,
    MetricContract,
    build_hidden_regime_factorial_calibration_design,
    canonical_json_bytes,
    canonical_sha256,
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
    HIDDEN_REGIME_TRACE_AUDIT_REPORT_SCHEMA,
    HiddenRegimeTraceAuditReport,
    audit_hidden_regime_run_result,
)
from alberta_framework.streams.hidden_regime_signaling import (
    CALIBRATION_ONLY_PARTITION,
    hidden_regime_calibration_manifest,
)

CALIBRATION_CASE_REQUEST_SCHEMA = "alberta.hidden-regime-factorial.case-request.v1"
CALIBRATION_PREFLIGHT_REQUEST_SCHEMA = "alberta.hidden-regime-factorial.preflight-request.v1"
CALIBRATION_PREFLIGHT_REPORT_SCHEMA = "alberta.hidden-regime-factorial.preflight-report.v1"
CALIBRATION_CASE_SHARD_SCHEMA = "alberta.hidden-regime-factorial.case-shard.v1"
CALIBRATION_ATTEMPT_SCHEMA = "alberta.hidden-regime-factorial.case-attempt.v1"
CALIBRATION_LEDGER_SCHEMA = "alberta.hidden-regime-factorial.case-ledger.v1"
CALIBRATION_AGGREGATE_SCHEMA = "alberta.hidden-regime-factorial.calibration-aggregate.v1"
PRIMITIVE_TRACE_DIGEST_SCHEMA = "alberta.hidden-regime-factorial.trace-digest.v1"

EXECUTION_ACKNOWLEDGEMENT = EXECUTION_AUTHORIZATION_ACKNOWLEDGEMENT
PREFLIGHT_ACKNOWLEDGEMENT = (
    "authorize a non-consuming ZIP preflight and optional process-local authorization issuance"
)
WORKER_RESULT_PREFIX = b"ALBERTA_HIDDEN_REGIME_CALIBRATION_SHARD_V1:"
PREFLIGHT_RESULT_PREFIX = b"ALBERTA_HIDDEN_REGIME_CALIBRATION_PREFLIGHT_V1:"

EXPECTED_STEPS = 16_528
EXPECTED_CONDITIONS = 8
EXPECTED_SEED_PAIRS = 30
EXPECTED_CASES = 240
EXPECTED_MANIFEST_CASES = 80
EXPECTED_MANIFEST_SEED_PAIRS = 10

_SHA256_LENGTH = 64
_MAX_RECEIPT_BYTES = 4 * 1024 * 1024
_MAX_SOURCE_ZIP_BYTES = 32 * 1024 * 1024
_MAX_WORKER_OUTPUT_BYTES = 8 * 1024 * 1024
_MAX_SHARD_BYTES = 8 * 1024 * 1024
_MAX_AGGREGATE_BYTES = 64 * 1024 * 1024
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

        def identity(item: os.stat_result) -> tuple[int, int, int, int]:
            return (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns)

        _require(
            identity(before) == identity(after) == identity(locator),
            f"{label} changed while reading",
        )
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


def load_validated_readiness_bundle(
    directory: Path,
    *,
    recheck_current: bool = False,
    recheck_runtime: bool = True,
) -> ValidatedReadinessBundle:
    """Load the exact published receipt and ZIP, rejecting mutable-path defects."""

    validation = validate_published_readiness_receipt(
        directory,
        recheck_current=recheck_current,
        recheck_runtime=recheck_runtime,
    )
    _require(validation.valid, "; ".join(validation.errors))
    receipt_raw = _read_regular_file(
        directory / "readiness.json",
        max_bytes=_MAX_RECEIPT_BYTES,
        label="readiness receipt",
    )
    archive = _read_regular_file(
        directory / "source.zip",
        max_bytes=_MAX_SOURCE_ZIP_BYTES,
        label="readiness source ZIP",
    )
    receipt = _strict_json(receipt_raw, "readiness receipt")
    try:
        bundle = require_validated_readiness_receipt(
            receipt,
            archive,
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
        "interrupted_case_indices",
        "started_records",
        "completed_records",
    ):
        _require(normalized.get(field) == [], f"preflight inventory {field} is not empty")
    for field in (
        "started_record_count",
        "completed_record_count",
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
    request_payload = request.to_payload()
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
            "outcome_sha256",
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
    ):
        _require(_is_sha256(execution_record[field]), f"execution {field} is invalid")
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
        "request_payload_sha256": request_payload["payload_sha256"],
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
        "execution_count": 1,
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
    for field in (
        "trace_audit_report_sha256",
        "accepted_float32_contractions_sha256",
        "evidence_boundary_sha256",
        "lineage_oracle_mismatches_sha256",
    ):
        _require(_is_sha256(audit[field]), f"case audit {field} is invalid")
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
        "request_payload_sha256",
        "readiness_binding",
        "worker_provenance",
        "execution_record_binding",
        "runtime_schemas",
        "execution_digest_schemas",
        "executed_steps",
        "execution_count",
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
    _require(_is_sha256(body["request_payload_sha256"]), "request digest invalid")
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
            "outcome_sha256",
        },
        "execution record binding",
    )
    design = _design()
    case_payload = _plain_dict(body["case"], "case binding")
    case_index = _strict_int(case_payload.get("case_index"), "case index")
    case = _case(design, case_index)
    _require(case_payload == case.to_payload(), "shard case differs from frozen ledger")
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
    ):
        _require(_is_sha256(execution_record[field]), f"execution record {field} invalid")
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
    _require(body["execution_count"] == 1, "case execution count differs")
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


def _write_new_immutable(parent: Path, name: str, raw: bytes) -> None:
    _require(
        bool(name) and name not in {".", ".."} and "/" not in name,
        "immutable publication member name is invalid",
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    directory_fd, _ = _open_directory_without_symlinks(parent, label="publication parent")
    try:
        descriptor = os.open(name, flags, 0o600, dir_fd=directory_fd)
        try:
            view = memoryview(raw)
            while view:
                written = os.write(descriptor, view)
                _require(written > 0, "short write while publishing case shard")
                view = view[written:]
            os.fchmod(descriptor, 0o444)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.fsync(directory_fd)
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


def publish_calibration_case_shard_new_only(
    publication_root: Path,
    shard: Mapping[str, object],
    *,
    expected_readiness_binding: Mapping[str, object],
) -> Path:
    """Publish one canonical shard once; an existing duplicate must be byte-identical."""

    validated = validate_calibration_case_shard(
        shard,
        expected_readiness_binding=expected_readiness_binding,
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
        _write_new_immutable(path.parent, path.name, raw)
    except FileExistsError:
        existing_raw = _read_regular_file(path, max_bytes=_MAX_SHARD_BYTES, label="case shard")
        existing = _strict_json(existing_raw, "case shard")
        validate_calibration_case_shard(
            existing,
            expected_readiness_binding=expected_readiness_binding,
        )
        _require(existing_raw == raw, "duplicate case shard is not byte-identical")
    return path


def load_complete_calibration_case_shards(
    publication_root: Path,
    *,
    expected_readiness_binding: Mapping[str, object],
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
        validated = validate_calibration_case_shard(
            payload,
            expected_readiness_binding=expected_readiness_binding,
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
    return {
        "case_index": case_index,
        "genesis_sha256": inventory["genesis_sha256"],
        "started_record_sha256": started["started_record_sha256"],
        "completed_record_sha256": completed["completed_record_sha256"],
        "summary_sha256": completed["summary_sha256"],
        "resource_sha256": completed["resource_sha256"],
        "primitive_trace_sha256": completed["primitive_trace_sha256"],
        "outcome_sha256": completed["outcome_sha256"],
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
    _strict_int(normalized["project_module_count"], "project module count", minimum=1)
    _require(_is_sha256(normalized["project_modules_sha256"]), "module digest invalid")
    return normalized


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
        recheck_runtime=True,
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
    rows = _preflight_case_binding_rows(validated_request, bundle)
    issue_authorizations = cast(bool, request_body["issue_process_local_authorizations"])
    if issue_authorizations:
        design = _design()
        for case, row in zip(design.cases, rows, strict=True):
            authorization = issue_calibration_execution_authorization(
                ledger_directory=ledger_directory,
                readiness_bundle=bundle,
                readiness_source_archive=source_archive,
                case_index=case.case_index,
                condition=case.condition,
                seed_pair=_seed_pair(case),
                config=_build_case_config(design, case),
                request_payload_sha256=cast(str, row["case_request_payload_sha256"]),
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
        recheck_runtime=True,
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
    design = _design()
    case = _case(design, request.case_index)
    seed_pair = _seed_pair(case)
    config = _build_case_config(design, case)
    authorization = issue_calibration_execution_authorization(
        ledger_directory=ledger_directory,
        readiness_bundle=bundle,
        readiness_source_archive=source_archive,
        case_index=case.case_index,
        condition=case.condition,
        seed_pair=seed_pair,
        config=config,
        request_payload_sha256=cast(str, request.to_payload()["payload_sha256"]),
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
    return extract_calibration_case_shard(
        run,
        request,
        audit,
        worker_provenance=provenance,
        execution_record_binding=execution_record,
    )


def main(arguments: Sequence[str] | None = None) -> int:
    """ZIP-only worker entry point bound by the readiness receipt."""

    argv = tuple(sys.argv[1:] if arguments is None else arguments)
    _require(len(argv) == 4, "worker arguments are not exact")
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
        recheck_runtime=True,
    )
    request = build_calibration_preflight_request(
        bundle,
        managed_ledger_directory=managed_ledger_directory,
        issue_process_local_authorizations=issue_process_local_authorizations,
        explicit_acknowledgement=explicit_acknowledgement,
    )
    encoded_request = base64.b64encode(canonical_json_bytes(request)).decode("ascii")
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
        recheck_runtime=True,
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
        validated_existing = validate_calibration_case_shard(
            existing,
            expected_readiness_binding=readiness_binding,
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
    validated = validate_calibration_case_shard(
        shard,
        expected_readiness_binding=readiness_binding,
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


def _gate_result_matrix(
    design: HiddenRegimeFactorialCalibrationDesign,
    levels: Sequence[Mapping[str, object]],
    estimands: Sequence[Mapping[str, object]],
    supports: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
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
    mandatory, descriptive = _gate_result_matrix(design, levels, estimands, supports)
    case_ledger = [
        {
            "case_index": case_index,
            "seed_index": _plain_dict(shard["case"], "case")["seed_index"],
            "condition": _plain_dict(shard["case"], "case")["condition"],
            "manifest_name": _plain_dict(shard["case"], "case")["manifest_name"],
            "case_shard_payload_sha256": shard["payload_sha256"],
            "request_payload_sha256": shard["request_payload_sha256"],
            "summary_sha256": shard["summary_sha256"],
            "resource_sha256": shard["resource_sha256"],
            "worker_provenance_sha256": canonical_sha256(shard["worker_provenance"]),
            "execution_record_binding": shard["execution_record_binding"],
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
        "mandatory_gate_results": mandatory,
        "descriptive_only_results": descriptive,
        "gate_decision_status": "not_evaluated_thresholds_unset",
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
) -> dict[str, object]:
    """Recompute an aggregate from its exact shards and managed ledger."""

    expected = aggregate_hidden_regime_factorial_calibration(
        shards,
        managed_ledger_snapshot=managed_ledger_snapshot,
        managed_ledger_directory=managed_ledger_directory,
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


def publish_calibration_aggregate_new_only(
    publication_root: Path,
    payload: Mapping[str, object],
    shards: Sequence[Mapping[str, object]],
    *,
    managed_ledger_snapshot: Mapping[str, object],
    managed_ledger_directory: Path,
    authorize_publication: bool = False,
) -> PublishedCalibrationAggregate:
    """Validate and publish a canonical aggregate once under its payload digest."""

    _require(authorize_publication is True, "aggregate publication requires authorization")
    validated = validate_calibration_aggregate(
        payload,
        shards,
        managed_ledger_snapshot=managed_ledger_snapshot,
        managed_ledger_directory=managed_ledger_directory,
    )
    payload_sha256 = cast(str, validated["payload_sha256"])
    path = calibration_aggregate_path(publication_root, payload_sha256)
    raw = canonical_json_bytes(validated)
    _require(len(raw) <= _MAX_AGGREGATE_BYTES, "calibration aggregate exceeds maximum size")
    try:
        _write_new_immutable(path.parent, path.name, raw)
    except FileExistsError:
        existing_raw = _read_regular_file(
            path,
            max_bytes=_MAX_AGGREGATE_BYTES,
            label="calibration aggregate",
        )
        existing = _strict_json(existing_raw, "calibration aggregate")
        validate_calibration_aggregate(
            existing,
            shards,
            managed_ledger_snapshot=managed_ledger_snapshot,
            managed_ledger_directory=managed_ledger_directory,
        )
        _require(existing_raw == raw, "duplicate aggregate is not byte-identical")
    return PublishedCalibrationAggregate(path, payload_sha256, validated)


def load_published_calibration_aggregate(
    publication_root: Path,
    payload_sha256: str,
    shards: Sequence[Mapping[str, object]],
    *,
    managed_ledger_snapshot: Mapping[str, object],
    managed_ledger_directory: Path,
) -> dict[str, object]:
    """Load one immutable content-addressed aggregate and recompute it exactly."""

    path = calibration_aggregate_path(publication_root, payload_sha256)
    raw = _read_regular_file(
        path,
        max_bytes=_MAX_AGGREGATE_BYTES,
        label="calibration aggregate",
    )
    payload = _strict_json(raw, "calibration aggregate")
    _require(payload.get("payload_sha256") == payload_sha256, "aggregate path digest differs")
    return validate_calibration_aggregate(
        payload,
        shards,
        managed_ledger_snapshot=managed_ledger_snapshot,
        managed_ledger_directory=managed_ledger_directory,
    )


def aggregate_and_publish_completed_calibration(
    *,
    readiness_directory: Path,
    shard_publication_root: Path,
    managed_ledger_directory: Path,
    aggregate_publication_root: Path,
    authorize_publication: bool = False,
) -> PublishedCalibrationAggregate:
    """Load the exact completed ledger, recompute, and publish its aggregate."""

    _require(authorize_publication is True, "aggregate publication requires authorization")
    bundle = load_validated_readiness_bundle(
        readiness_directory,
        recheck_current=False,
        recheck_runtime=True,
    )
    readiness = _readiness_binding(bundle)
    shards = load_complete_calibration_case_shards(
        shard_publication_root,
        expected_readiness_binding=readiness,
    )
    inventory = snapshot_calibration_execution_inventory(managed_ledger_directory)
    aggregate = aggregate_hidden_regime_factorial_calibration(
        shards,
        managed_ledger_snapshot=inventory,
        managed_ledger_directory=managed_ledger_directory,
    )
    return publish_calibration_aggregate_new_only(
        aggregate_publication_root,
        aggregate,
        shards,
        managed_ledger_snapshot=inventory,
        managed_ledger_directory=managed_ledger_directory,
        authorize_publication=True,
    )
