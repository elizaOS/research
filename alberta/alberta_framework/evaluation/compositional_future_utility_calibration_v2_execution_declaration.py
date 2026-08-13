"""Pure-stdlib execution declaration for future-utility calibration v2.

This declaration authorizes one development-only invocation of the public
five-arm runner from a fresh Python process.  The attempt is consumed before
the evaluator is imported and remains consumed after any import, runtime,
validation, formatting, or output failure.  It grants no retry, writer,
winner-selection, tuning, evidence, or promotion authority.
"""

from __future__ import annotations

import hashlib
import json
import struct
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, cast

DECLARATION_SCHEMA: Final = (
    "alberta.compositional-future-utility-calibration-v2-execution-declaration.v1"
)
SUMMARY_SCHEMA: Final = (
    "alberta.compositional-future-utility-calibration-v2-execution-summary.v1"
)
PROTOCOL_SCHEMA: Final = (
    "alberta.compositional-future-utility-calibration-v2-development.protocol.v1"
)
REPORT_SCHEMA: Final = (
    "alberta.compositional-future-utility-calibration-v2-development.report.v1"
)
REPORT_STATUS: Final = "DEVELOPMENT_FUTURE_UTILITY_CALIBRATION_V2_NOT_ASSESSED"
ASSESSMENT_STATUS: Final = "not-assessed"

DEVELOPMENT_ONLY: Final = True
SCIENTIFIC_PROMOTION_ALLOWED: Final = False
EVIDENCE_AUTHORIZED: Final = False
OUTPUT_WRITES_ALLOWED: Final = False
ARTIFACT_BYTES_AUTHORIZED: Final = 0
WINNER_OR_DEFAULT_SELECTION_ALLOWED: Final = False
THRESHOLD_ALLOWED: Final = False
SEARCH_OR_TUNING_ALLOWED: Final = False
RERUN_ALLOWED: Final = False
RECOVERY_ALLOWED: Final = False
FRESH_PROCESS_REQUIRED: Final = True
EXECUTION_ATTEMPTS_AUTHORIZED: Final = 1
ATTEMPTS_CONSUMED_BEFORE_DECLARATION: Final = 0
ATTEMPT_CONSUMED_BEFORE_EVALUATOR_IMPORT: Final = True
FAILURE_AFTER_ATTEMPT_ENTRY_CONSUMES_ATTEMPT: Final = True
ROOT_CONSUMED_ON_SUCCESS_OR_FAILURE: Final = True
CROSS_PROCESS_REPLAY_PREVENTED: Final = False
PERMITTED_CONCLUSION_SCOPE: Final = (
    "descriptive-five-arm-development-outcome-or-fail-closed-execution-error"
)
EXECUTION_ACKNOWLEDGEMENT: Final = (
    "consume future-utility-v2 development root 0x72B0A3F6; no retry"
)
FORBIDDEN_PRELOADED_MODULE_ROOTS: Final = (
    "alberta_framework",
    "jax",
    "jaxlib",
    "numpy",
)

RUNNER_MODULE: Final = (
    "alberta_framework.evaluation."
    "compositional_future_utility_calibration_v2_development"
)
RUNNER_FUNCTION: Final = (
    "run_compositional_future_utility_calibration_v2_development"
)
SUMMARY_OUTPUT_ORDER: Final = (
    "canonical_summary_json",
    "report_sha256",
    "postflight_source_validation",
)

PROTOCOL_NAMESPACE: Final = "alberta.compositional-future-utility-calibration-v2"
PROTOCOL_NAMESPACE_SHA256: Final = (
    "72b0a3f637872fbdaa750423a784367d5940004aad60e3d53bf81b60fa062217"
)
PROTOCOL_CONFIG_SHA256: Final = (
    "9eebf20c8052a280d4b9864c8c307be92c40365ce2c3b7724040bd4a42b30b6a"
)
DEVELOPMENT_ROOT: Final = 1_924_178_934
DEVELOPMENT_ROOT_HEX: Final = "0x72B0A3F6"
PHASE_ORDER: Final = ("A", "B", "A", "D", "A", "C", "A", "B", "C", "A")
PHASE_LENGTHS: Final = (797, 829, 857, 883, 911, 941, 971, 1009, 1031, 769)
PHASE_BOUNDARIES: Final = (0, 797, 1626, 2483, 3366, 4277, 5218, 6189, 7198, 8229, 8998)
CURATION_OPPORTUNITIES_PER_PHASE: Final = (24, 26, 27, 28, 28, 30, 30, 31, 33, 24)
TOTAL_STEPS: Final = 8_998
CURATION_INTERVAL: Final = 32
TOTAL_CURATION_OPPORTUNITIES: Final = 281
STREAM_SHA256: Final = (
    "bb741db073a13026425d2cc98cce93a1af1d1b65f2abf24ebc97e43b61abd39c"
)
STREAM_DIGEST_SCOPE: Final = (
    "observations, phase_indices, exploration_mask, and random_actions; "
    "learner genesis is separately key-bound"
)

KEY_MANIFEST: Final[Mapping[str, tuple[int, int]]] = MappingProxyType(
    {
        "root": (0, 1_924_178_934),
        "observations": (1_189_056_302, 2_383_774_845),
        "exploration": (3_352_410_003, 3_947_271_724),
        "random_actions": (3_382_640_669, 4_117_898_437),
        "learner_genesis": (2_592_838_183, 3_227_537_730),
    }
)
KEY_MANIFEST_SHA256: Final = (
    "13b0ba0abe86fc8d7f7e34f1a7d674e79080e8b5c36f6f94e9841b3867c980df"
)

ARM_NAMES: Final = (
    "current_mix0_decay095_none",
    "future_mix1_decay095_none",
    "calibrated_mix05_decay095_none",
    "normalized_mix1_decay095_uncertainty_age",
    "horizon_mix1_decay883_uncertainty_age",
)
ARM_PARAMETERS: Final = (
    (0.0, 0.95, "none"),
    (1.0, 0.95, "none"),
    (0.5, 0.95, "none"),
    (1.0, 0.95, "uncertainty_age"),
    (1.0, 0.999215304851532, "uncertainty_age"),
)
LONG_TRACE_DECAY_F32_BITS: Final = "3f7fcc93"
PRIMARY_ENDPOINTS: Final = (
    "margin_passes",
    "promotions",
    "candidate_refreshes",
    "cascade_losses",
    "target_admission_loss_end",
    "pre_recurrence_presence",
    "target_occupancy",
    "pre_recurrence_ranks",
)
SECONDARY_ENDPOINTS: Final = ("lifetime_reward", "phase_reward")

# Ordered triples are evaluator-key, repository-relative path, lowercase SHA-256.
SELECTED_SOURCE_MANIFEST: Final = (
    (
        "evaluation_module_sha256",
        "alberta_framework/evaluation/"
        "compositional_future_utility_calibration_v2_development.py",
        "643227c17c944ffdb47ad255a2ae204b2e4be107f207b7ba4e1ce5f93ea9231b",
    ),
    (
        "control_life_v1_sha256",
        "alberta_framework/evaluation/compositional_control_life_development.py",
        "ca55a1577f699bce89bd3a75b1ec60e2522046df1ba236437d641251847ae2c3",
    ),
    (
        "compositional_core_sha256",
        "alberta_framework/core/compositional_features.py",
        "767f054bb3413b2408e664a17bcb8690a9f83018f638d6acfcfde2e9debf5b5a",
    ),
    (
        "future_utility_core_sha256",
        "alberta_framework/core/future_utility.py",
        "2c513fedea7423f7b24b55510582980d3fc21c8d1e9853aa6f9cf34182839e92",
    ),
    (
        "birth_identity_scrub_sha256",
        "alberta_framework/evaluation/generated_birth_identity_scrub_epoch.py",
        "57337a73b47140f149f2afbc382c8bc0f7b6316361c99c8eba37172f69d4150c",
    ),
    (
        "lifecycle_state_size_sha256",
        "alberta_framework/evaluation/generated_class_lifecycle_scrub.py",
        "d389ea4294ab354a47af010d76914c3f4e6e87f231b9ccd4101dd1e40741dba9",
    ),
)
SELECTED_SOURCE_MANIFEST_SHA256: Final = (
    "458e7bdc5a1cf8a522b0c299a36a0494101f25c469007f89b9e0200d770f5e88"
)
SELECTED_SOURCE_PATH_MANIFEST_SHA256: Final = (
    "a5e5c9eed522f725a9e706e82ba2ea32f04128fb513b672ea50e378363a1290b"
)
SOURCE_MANIFEST_SCOPE: Final = "selected-direct-files-not-transitive-closure"
FOCUSED_CONTRACT_TEST_PATH: Final = (
    "tests/test_compositional_future_utility_calibration_v2_development.py"
)
FOCUSED_CONTRACT_TEST_SHA256: Final = (
    "6cbbf6c13455a69a5fb61b8836e29f8ac07f33d74eb443d501ac09c1d74d2e03"
)


def canonical_json(value: object) -> str:
    """Return the evaluator's exact canonical JSON encoding."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def canonical_json_sha256(value: object) -> str:
    """Return SHA-256 of the exact canonical JSON encoding."""

    return hashlib.sha256(canonical_json(value).encode("ascii")).hexdigest()


def selected_source_hashes() -> dict[str, str]:
    """Return the exact mapping constructed by the evaluator before its digest."""

    return {key: digest for key, _path, digest in SELECTED_SOURCE_MANIFEST}


def evaluator_selected_source_manifest() -> dict[str, str]:
    """Return the exact expected evaluator manifest including its JSON digest."""

    files = selected_source_hashes()
    return {**files, "manifest_sha256": canonical_json_sha256(files)}


def selected_source_path_manifest_sha256(
    manifest: Sequence[tuple[str, str, str]] = SELECTED_SOURCE_MANIFEST,
) -> str:
    """Digest ordered UTF-8 ``path + NUL + sha256 + LF`` framing."""

    payload = "".join(f"{path}\0{digest}\n" for _key, path, digest in manifest)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_selected_sources(root: Path) -> tuple[str, ...]:
    """Validate bound source/test bytes without importing Alberta or JAX."""

    errors: list[str] = []
    resolved_root = root.resolve()
    if canonical_json_sha256(selected_source_hashes()) != SELECTED_SOURCE_MANIFEST_SHA256:
        errors.append("selected-source JSON manifest digest is internally inconsistent")
    if selected_source_path_manifest_sha256() != SELECTED_SOURCE_PATH_MANIFEST_SHA256:
        errors.append("selected-source path manifest digest is internally inconsistent")
    expected_files = tuple(
        (path, digest) for _key, path, digest in SELECTED_SOURCE_MANIFEST
    ) + ((FOCUSED_CONTRACT_TEST_PATH, FOCUSED_CONTRACT_TEST_SHA256),)
    for path, expected in expected_files:
        candidate = (resolved_root / path).resolve()
        if not candidate.is_relative_to(resolved_root) or not candidate.is_file():
            errors.append(f"missing declared source: {path}")
            continue
        observed = hashlib.sha256(candidate.read_bytes()).hexdigest()
        if observed != expected:
            errors.append(f"declared source digest mismatch: {path}: {expected} != {observed}")
    return tuple(errors)


def validate_execution_declaration(root: Path) -> tuple[str, ...]:
    """Return fail-closed declaration or source errors without importing JAX."""

    errors = list(validate_selected_sources(root))
    namespace_digest = hashlib.sha256(PROTOCOL_NAMESPACE.encode("ascii")).hexdigest()
    if namespace_digest != PROTOCOL_NAMESPACE_SHA256:
        errors.append("protocol namespace digest is inconsistent")
    if int(PROTOCOL_NAMESPACE_SHA256[:8], 16) != DEVELOPMENT_ROOT:
        errors.append("development root is not derived from the namespace")
    if DEVELOPMENT_ROOT_HEX != f"0x{DEVELOPMENT_ROOT:08X}":
        errors.append("development root hexadecimal form is inconsistent")
    if sum(PHASE_LENGTHS) != TOTAL_STEPS:
        errors.append("phase lengths do not total the declared life")
    boundaries = [0]
    for length in PHASE_LENGTHS:
        boundaries.append(boundaries[-1] + length)
    if tuple(boundaries) != PHASE_BOUNDARIES:
        errors.append("phase boundaries do not reconstruct")
    per_phase = tuple(
        sum(
            1
            for due in range(CURATION_INTERVAL, TOTAL_STEPS + 1, CURATION_INTERVAL)
            if start < due <= stop
        )
        for start, stop in zip(PHASE_BOUNDARIES[:-1], PHASE_BOUNDARIES[1:], strict=True)
    )
    if per_phase != CURATION_OPPORTUNITIES_PER_PHASE:
        errors.append("per-phase curation opportunities do not reconstruct")
    if sum(per_phase) != TOTAL_CURATION_OPPORTUNITIES:
        errors.append("total curation opportunities do not reconstruct")
    key_payload = {name: list(words) for name, words in KEY_MANIFEST.items()}
    if canonical_json_sha256(key_payload) != KEY_MANIFEST_SHA256:
        errors.append("key manifest digest is inconsistent")
    if struct.pack(">f", ARM_PARAMETERS[-1][1]).hex() != LONG_TRACE_DECAY_F32_BITS:
        errors.append("long-horizon decay float32 bits are inconsistent")
    if len(ARM_NAMES) != 5 or len(ARM_PARAMETERS) != len(ARM_NAMES):
        errors.append("the five-arm declaration is incomplete")
    if EXECUTION_ATTEMPTS_AUTHORIZED != 1 or ATTEMPTS_CONSUMED_BEFORE_DECLARATION != 0:
        errors.append("the one-attempt budget is inconsistent")
    if not FRESH_PROCESS_REQUIRED or not ATTEMPT_CONSUMED_BEFORE_EVALUATOR_IMPORT:
        errors.append("fresh-process attempt chronology is not fail closed")
    if not ROOT_CONSUMED_ON_SUCCESS_OR_FAILURE or RERUN_ALLOWED or RECOVERY_ALLOWED:
        errors.append("the root/retry authority is inconsistent")
    if OUTPUT_WRITES_ALLOWED or ARTIFACT_BYTES_AUTHORIZED != 0:
        errors.append("the declaration acquired output authority")
    if SCIENTIFIC_PROMOTION_ALLOWED or EVIDENCE_AUTHORIZED:
        errors.append("the declaration acquired scientific authority")
    if WINNER_OR_DEFAULT_SELECTION_ALLOWED or THRESHOLD_ALLOWED or SEARCH_OR_TUNING_ALLOWED:
        errors.append("the declaration acquired selection or tuning authority")
    if SUMMARY_OUTPUT_ORDER != (
        "canonical_summary_json",
        "report_sha256",
        "postflight_source_validation",
    ):
        errors.append("the summary-first output order drifted")
    return tuple(errors)


def build_clean_preflight(
    root: Path,
    loaded_modules: Iterable[str],
) -> dict[str, object]:
    """Build a nonexecuting fresh-process preflight record.

    Callers must inspect ``valid`` before entering the attempt.  This function
    never imports the evaluator and never consumes the operational attempt.
    """

    names = tuple(loaded_modules)
    preloaded = sorted(
        name
        for name in names
        if any(
            name == module_root or name.startswith(module_root + ".")
            for module_root in FORBIDDEN_PRELOADED_MODULE_ROOTS
        )
    )
    errors = list(validate_execution_declaration(root))
    if preloaded:
        errors.append("forbidden modules were already loaded: " + ", ".join(preloaded))
    return {
        "declaration_schema": DECLARATION_SCHEMA,
        "valid": not errors,
        "errors": errors,
        "fresh_process_required": FRESH_PROCESS_REQUIRED,
        "forbidden_preloaded_module_roots": list(FORBIDDEN_PRELOADED_MODULE_ROOTS),
        "forbidden_preloaded_modules_observed": preloaded,
        "panel_executed": False,
        "attempt_consumed": False,
        "attempts_authorized": EXECUTION_ATTEMPTS_AUTHORIZED,
        "acknowledgement_required": EXECUTION_ACKNOWLEDGEMENT,
        "selected_source_manifest_sha256": SELECTED_SOURCE_MANIFEST_SHA256,
        "protocol_sha256": PROTOCOL_CONFIG_SHA256,
        "key_manifest_sha256": KEY_MANIFEST_SHA256,
        "stream_sha256": STREAM_SHA256,
    }


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return cast(Mapping[str, Any], value)


def _list(value: object, field: str) -> list[Any]:
    if type(value) is not list:
        raise ValueError(f"{field} must be an exact list")
    return value


def _summarize_report_only(report: Mapping[str, object]) -> dict[str, object]:
    """Validate in-report bindings after an independent source postflight."""

    try:
        candidate = cast(dict[str, Any], json.loads(canonical_json(dict(report))))
    except (TypeError, ValueError) as error:
        raise ValueError(f"report is not canonical JSON: {error}") from error
    body = {key: value for key, value in candidate.items() if key != "report_sha256"}
    if candidate.get("report_sha256") != canonical_json_sha256(body):
        raise ValueError("report_sha256 does not reconstruct")
    exact_fields = {
        "schema": REPORT_SCHEMA,
        "status": REPORT_STATUS,
        "assessment_status": ASSESSMENT_STATUS,
        "development_only": True,
        "scientific_promotion_allowed": False,
        "evidence_authorized": False,
        "output_writes_allowed": False,
        "artifact_available": False,
        "artifact_bytes_written": 0,
        "protocol_sha256": PROTOCOL_CONFIG_SHA256,
        "development_root": DEVELOPMENT_ROOT,
        "development_root_hex": DEVELOPMENT_ROOT_HEX,
        "stream_sha256": STREAM_SHA256,
        "source_manifest_scope": SOURCE_MANIFEST_SCOPE,
        "source_manifest_pre_post_import_equal": True,
        "runtime_identity_pre_post_equal": True,
        "winner_or_default_selected": False,
        "threshold_defined_or_applied": False,
        "search_performed": False,
        "rerun_or_tuning_authorized": False,
    }
    mismatches = {
        key: (expected, candidate.get(key))
        for key, expected in exact_fields.items()
        if candidate.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"report execution bindings differ: {mismatches}")
    expected_manifest = evaluator_selected_source_manifest()
    for field in (
        "source_manifest_import_snapshot",
        "source_manifest_live_pre",
        "source_manifest_live_post",
    ):
        if candidate.get(field) != expected_manifest:
            raise ValueError(f"{field} does not match the execution declaration")
    if candidate.get("arm_order") != list(ARM_NAMES):
        raise ValueError("report arm order differs from the declaration")
    if candidate.get("primary_endpoint_order") != list(PRIMARY_ENDPOINTS):
        raise ValueError("report primary endpoint order differs from the declaration")
    if candidate.get("secondary_endpoint_order") != list(SECONDARY_ENDPOINTS):
        raise ValueError("report secondary endpoint order differs from the declaration")

    runs = _list(candidate.get("runs"), "runs")
    if len(runs) != len(ARM_NAMES):
        raise ValueError("report does not contain exactly five arms")
    arm_summaries: list[dict[str, object]] = []
    genesis_hashes: set[str] = set()
    for expected_arm, raw_run in zip(ARM_NAMES, runs, strict=True):
        run = _mapping(raw_run, f"run[{expected_arm}]")
        if run.get("arm") != expected_arm:
            raise ValueError(f"run order/name mismatch for {expected_arm}")
        if run.get("initial_persistent_state_nbytes") != 2_072:
            raise ValueError(f"{expected_arm} initial state is not 2,072 bytes")
        if run.get("final_persistent_state_nbytes") != 2_072:
            raise ValueError(f"{expected_arm} final state is not 2,072 bytes")
        genesis = run.get("initial_state_sha256")
        if type(genesis) is not str:
            raise ValueError(f"{expected_arm} initial state digest is invalid")
        genesis_hashes.add(genesis)
        primary = _mapping(run.get("primary_endpoints"), f"{expected_arm}.primary")
        secondary = _mapping(
            run.get("secondary_reward_endpoints"), f"{expected_arm}.secondary"
        )
        if primary.get("endpoint_order") != list(PRIMARY_ENDPOINTS):
            raise ValueError(f"{expected_arm} primary endpoint order drifted")
        if secondary.get("endpoint_order") != list(SECONDARY_ENDPOINTS):
            raise ValueError(f"{expected_arm} secondary endpoint order drifted")
        occupancy = _mapping(primary.get("target_occupancy"), "target_occupancy")
        lifetime = _mapping(secondary.get("lifetime_reward"), "lifetime_reward")
        ranks = _mapping(primary.get("pre_recurrence_ranks"), "pre_recurrence_ranks")
        arm_summaries.append(
            {
                "arm": expected_arm,
                "primary_endpoints_sha256": canonical_json_sha256(primary),
                "margin_passes": primary.get("margin_passes"),
                "promotions": primary.get("promotions"),
                "candidate_refreshes": primary.get("candidate_refreshes"),
                "cascade_refill_slot_count": primary.get("cascade_refill_slot_count"),
                "cascade_losses": primary.get("cascade_losses"),
                "target_admission_loss_end": primary.get("target_admission_loss_end"),
                "pre_recurrence_presence": primary.get("pre_recurrence_presence"),
                "a_retention": primary.get("a_retention"),
                "maximum_distinct_active_target_count": occupancy.get(
                    "maximum_distinct_active_target_count"
                ),
                "final_active_targets": occupancy.get("final_active_targets"),
                "pre_recurrence_rank_records": ranks.get("records"),
                "lifetime_executed_reward": lifetime.get("executed_reward"),
            }
        )
    if len(genesis_hashes) != 1:
        raise ValueError("the five arms do not share one exact genesis")

    comparison = _mapping(candidate.get("arm_comparison"), "arm_comparison")
    comparison_contract = {
        "shared_base_logical_work_equal": True,
        "stream_shapes_and_update_opportunities_equal": True,
        "intervention_specific_logical_work_equal": False,
        "total_named_logical_work_equivalence_claimed": False,
        "behavior_dependent_branch_work_equivalence_claimed": False,
        "winner_selected": False,
        "threshold_applied": False,
        "rerun_or_tuning_authorized": False,
    }
    for key, expected in comparison_contract.items():
        if comparison.get(key) != expected:
            raise ValueError(f"arm comparison contract drifted at {key}")
    return {
        "schema": SUMMARY_SCHEMA,
        "declaration_schema": DECLARATION_SCHEMA,
        "attempts_authorized": EXECUTION_ATTEMPTS_AUTHORIZED,
        "attempts_consumed": 1,
        "assessment_status": ASSESSMENT_STATUS,
        "development_only": True,
        "scientific_promotion_allowed": False,
        "evidence_authorized": False,
        "output_writes_allowed": False,
        "winner_or_default_selected": False,
        "threshold_applied": False,
        "rerun_or_tuning_authorized": False,
        "report_sha256": candidate["report_sha256"],
        "protocol_sha256": PROTOCOL_CONFIG_SHA256,
        "stream_sha256": STREAM_SHA256,
        "selected_source_manifest_sha256": SELECTED_SOURCE_MANIFEST_SHA256,
        "shared_initial_state_sha256": next(iter(genesis_hashes)),
        "arm_summaries": arm_summaries,
    }


def validate_postrun_binding(
    report: Mapping[str, object],
    root: Path,
) -> tuple[str, ...]:
    """Validate current source bytes and the completed report without rerunning."""

    errors = list(validate_selected_sources(root))
    if errors:
        return tuple(errors)
    try:
        _summarize_report_only(report)
    except ValueError as error:
        errors.append(str(error))
    return tuple(errors)


def summarize_completed_report(
    report: Mapping[str, object],
    root: Path,
) -> dict[str, object]:
    """Return endpoint conclusions only after an independent clean postflight."""

    errors = validate_postrun_binding(report, root)
    if errors:
        raise ValueError("post-run binding failed: " + "; ".join(errors))
    return _summarize_report_only(report)


def canonical_summary_json(report: Mapping[str, object], root: Path) -> str:
    """Return the required first output after a clean completed attempt."""

    return canonical_json(summarize_completed_report(report, root))


__all__ = [
    "ARM_NAMES",
    "ARM_PARAMETERS",
    "ARTIFACT_BYTES_AUTHORIZED",
    "ATTEMPT_CONSUMED_BEFORE_EVALUATOR_IMPORT",
    "CROSS_PROCESS_REPLAY_PREVENTED",
    "CURATION_INTERVAL",
    "DECLARATION_SCHEMA",
    "DEVELOPMENT_ROOT",
    "DEVELOPMENT_ROOT_HEX",
    "EVIDENCE_AUTHORIZED",
    "EXECUTION_ACKNOWLEDGEMENT",
    "EXECUTION_ATTEMPTS_AUTHORIZED",
    "FAILURE_AFTER_ATTEMPT_ENTRY_CONSUMES_ATTEMPT",
    "FOCUSED_CONTRACT_TEST_PATH",
    "FOCUSED_CONTRACT_TEST_SHA256",
    "FRESH_PROCESS_REQUIRED",
    "KEY_MANIFEST",
    "KEY_MANIFEST_SHA256",
    "OUTPUT_WRITES_ALLOWED",
    "PERMITTED_CONCLUSION_SCOPE",
    "PHASE_LENGTHS",
    "PHASE_ORDER",
    "PRIMARY_ENDPOINTS",
    "PROTOCOL_CONFIG_SHA256",
    "PROTOCOL_NAMESPACE",
    "PROTOCOL_NAMESPACE_SHA256",
    "RECOVERY_ALLOWED",
    "RERUN_ALLOWED",
    "ROOT_CONSUMED_ON_SUCCESS_OR_FAILURE",
    "RUNNER_FUNCTION",
    "RUNNER_MODULE",
    "SCIENTIFIC_PROMOTION_ALLOWED",
    "SECONDARY_ENDPOINTS",
    "SELECTED_SOURCE_MANIFEST",
    "SELECTED_SOURCE_MANIFEST_SHA256",
    "SELECTED_SOURCE_PATH_MANIFEST_SHA256",
    "STREAM_SHA256",
    "SUMMARY_OUTPUT_ORDER",
    "SUMMARY_SCHEMA",
    "TOTAL_CURATION_OPPORTUNITIES",
    "TOTAL_STEPS",
    "canonical_json",
    "canonical_json_sha256",
    "canonical_summary_json",
    "build_clean_preflight",
    "evaluator_selected_source_manifest",
    "selected_source_hashes",
    "selected_source_path_manifest_sha256",
    "summarize_completed_report",
    "validate_execution_declaration",
    "validate_postrun_binding",
    "validate_selected_sources",
]
