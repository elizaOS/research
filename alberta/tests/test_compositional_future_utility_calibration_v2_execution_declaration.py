"""Pure-stdlib checks for the one-shot future-utility-v2 declaration."""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
DECLARATION = (
    ROOT
    / "alberta_framework/evaluation/"
    "compositional_future_utility_calibration_v2_execution_declaration.py"
)
EVALUATOR = (
    ROOT
    / "alberta_framework/evaluation/"
    "compositional_future_utility_calibration_v2_development.py"
)

def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_future_v2_execution_declaration", DECLARATION)
    if spec is None or spec.loader is None:
        raise RuntimeError("execution declaration spec unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _literal_assignment(tree: ast.Module, name: str) -> Any:
    node = next(
        item
        for item in tree.body
        if isinstance(item, ast.AnnAssign)
        and isinstance(item.target, ast.Name)
        and item.target.id == name
    )
    if node.value is None:
        raise AssertionError(f"{name} has no assigned value")
    return ast.literal_eval(node.value)


def _synthetic_report(module: ModuleType) -> dict[str, object]:
    runs: list[dict[str, object]] = []
    for index, arm in enumerate(module.ARM_NAMES):
        primary = {
            "endpoint_order": list(module.PRIMARY_ENDPOINTS),
            "margin_passes": {
                "selected_strict_margin_pass_count": index,
                "candidate_destination_strict_margin_pair_count": index + 1,
                "due_curation_event_count": 281,
            },
            "promotions": {"event_count": index},
            "candidate_refreshes": {"total_refreshed_slot_count": index + 2},
            "cascade_refill_slot_count": index + 3,
            "cascade_losses": {"A": {"loss_episode_count": index}},
            "target_admission_loss_end": {"A": {"present_at_end": True}},
            "pre_recurrence_presence": [{"target": "A", "active_present": True}],
            "a_retention": {"pre_recurrence_presence": [True] * 4},
            "target_occupancy": {
                "maximum_distinct_active_target_count": 3,
                "final_active_targets": ["A", "B", "C"],
            },
            "pre_recurrence_ranks": {"records": [{"target": "A"}]},
        }
        runs.append(
            {
                "arm": arm,
                "initial_persistent_state_nbytes": 2_072,
                "final_persistent_state_nbytes": 2_072,
                "initial_state_sha256": "shared-genesis",
                "primary_endpoints": primary,
                "secondary_reward_endpoints": {
                    "endpoint_order": list(module.SECONDARY_ENDPOINTS),
                    "lifetime_reward": {"executed_reward": 0.1 * index},
                    "phase_reward": [],
                },
            }
        )
    manifest = module.evaluator_selected_source_manifest()
    body: dict[str, object] = {
        "schema": "alberta.compositional-future-utility-calibration-v2-development.report.v1",
        "status": "DEVELOPMENT_FUTURE_UTILITY_CALIBRATION_V2_NOT_ASSESSED",
        "assessment_status": "not-assessed",
        "development_only": True,
        "scientific_promotion_allowed": False,
        "evidence_authorized": False,
        "output_writes_allowed": False,
        "artifact_available": False,
        "artifact_bytes_written": 0,
        "protocol_sha256": module.PROTOCOL_CONFIG_SHA256,
        "development_root": module.DEVELOPMENT_ROOT,
        "development_root_hex": module.DEVELOPMENT_ROOT_HEX,
        "stream_sha256": module.STREAM_SHA256,
        "source_manifest_scope": "selected-direct-files-not-transitive-closure",
        "source_manifest_import_snapshot": manifest,
        "source_manifest_live_pre": manifest,
        "source_manifest_live_post": manifest,
        "source_manifest_pre_post_import_equal": True,
        "runtime_identity_pre_post_equal": True,
        "winner_or_default_selected": False,
        "threshold_defined_or_applied": False,
        "search_performed": False,
        "rerun_or_tuning_authorized": False,
        "arm_order": list(module.ARM_NAMES),
        "primary_endpoint_order": list(module.PRIMARY_ENDPOINTS),
        "secondary_endpoint_order": list(module.SECONDARY_ENDPOINTS),
        "runs": runs,
        "arm_comparison": {
            "shared_base_logical_work_equal": True,
            "stream_shapes_and_update_opportunities_equal": True,
            "intervention_specific_logical_work_equal": False,
            "total_named_logical_work_equivalence_claimed": False,
            "behavior_dependent_branch_work_equivalence_claimed": False,
            "winner_selected": False,
            "threshold_applied": False,
            "rerun_or_tuning_authorized": False,
        },
    }
    return {**body, "report_sha256": module.canonical_json_sha256(body)}


def test_historical_declaration_is_pure_stdlib_and_now_source_invalid() -> None:
    module = _load()
    tree = ast.parse(DECLARATION.read_text(encoding="utf-8"))
    roots = {
        alias.name.split(".", 1)[0]
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    roots |= {
        node.module.split(".", 1)[0]
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert roots <= set(sys.stdlib_module_names) | {"__future__"}
    errors = module.validate_execution_declaration(ROOT)
    assert len(errors) == 3
    assert any(
        "compositional_future_utility_calibration_v2_development.py" in error
        for error in errors
    )
    assert any(
        "compositional_control_life_development.py" in error for error in errors
    )
    assert any(
        "test_compositional_future_utility_calibration_v2_development.py" in error
        for error in errors
    )
    assert module.EXECUTION_ATTEMPTS_AUTHORIZED == 1
    assert module.ATTEMPT_CONSUMED_BEFORE_EVALUATOR_IMPORT
    assert module.ROOT_CONSUMED_ON_SUCCESS_OR_FAILURE
    assert not module.RERUN_ALLOWED
    assert not module.RECOVERY_ALLOWED
    assert not module.OUTPUT_WRITES_ALLOWED
    assert not module.SCIENTIFIC_PROMOTION_ALLOWED


def test_protocol_and_source_bindings_match_evaluator_source_without_import() -> None:
    module = _load()
    tree = ast.parse(EVALUATOR.read_text(encoding="utf-8"))
    for name in (
        "PROTOCOL_NAMESPACE",
        "PROTOCOL_NAMESPACE_SHA256",
        "PROTOCOL_SCHEMA",
        "REPORT_SCHEMA",
        "DEVELOPMENT_ROOT",
        "DEVELOPMENT_ROOT_HEX",
        "PHASE_ORDER",
        "PHASE_LENGTHS",
        "TOTAL_STEPS",
        "CURATION_INTERVAL",
        "STREAM_SHA256",
        "ARM_NAMES",
        "LONG_TRACE_DECAY_F32_BITS",
    ):
        assert _literal_assignment(tree, name) == getattr(module, name)
    assert module.evaluator_selected_source_manifest()["manifest_sha256"] == (
        module.SELECTED_SOURCE_MANIFEST_SHA256
    )
    assert module.selected_source_path_manifest_sha256() == (
        module.SELECTED_SOURCE_PATH_MANIFEST_SHA256
    )


def test_private_panel_helpers_require_latch_capability_in_source() -> None:
    tree = ast.parse(EVALUATOR.read_text(encoding="utf-8"))
    build_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_build_report"
    ]
    assert len(build_calls) == 1
    assert len(build_calls[0].args) == 1
    arm_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_run_arm"
    ]
    assert len(arm_calls) == 1
    assert {keyword.arg for keyword in arm_calls[0].keywords} == {
        "_execution_capability"
    }
    source = EVALUATOR.read_text(encoding="utf-8")
    assert "_FULL_REPORT_ATTEMPT.authorizes(_execution_capability)" in source
    assert "lambda capability: _canonical_json(_build_report(capability))" in source


def test_summary_projection_validates_bindings_and_covers_endpoint_families() -> None:
    module = _load()
    report = _synthetic_report(module)
    summary = cast(dict[str, Any], module._summarize_report_only(report))

    assert summary["report_sha256"] == report["report_sha256"]
    assert summary["attempts_authorized"] == summary["attempts_consumed"] == 1
    assert len(summary["arm_summaries"]) == 5
    first = cast(list[dict[str, Any]], summary["arm_summaries"])[0]
    assert first["margin_passes"]["selected_strict_margin_pass_count"] == 0
    assert first["promotions"] == {"event_count": 0}
    assert first["final_active_targets"] == ["A", "B", "C"]
    assert first["pre_recurrence_rank_records"] == [{"target": "A"}]

    corrupted = dict(report)
    corrupted["stream_sha256"] = "0" * 64
    body = {key: value for key, value in corrupted.items() if key != "report_sha256"}
    corrupted["report_sha256"] = module.canonical_json_sha256(body)
    try:
        module._summarize_report_only(corrupted)
    except ValueError as error:
        assert "execution bindings differ" in str(error)
    else:
        raise AssertionError("corrupted execution binding was accepted")

    try:
        module.summarize_completed_report(report, ROOT)
    except ValueError as error:
        assert "post-run binding failed" in str(error)
    else:
        raise AssertionError("consumed historical declaration remained executable")


def test_consumed_preflight_is_nonexecuting_invalid_and_rejects_preloaded_stack() -> None:
    module = _load()
    clean = cast(
        dict[str, Any],
        module.build_clean_preflight(ROOT, ("sys", "json", "hashlib")),
    )
    assert clean["valid"] is False
    assert len(clean["errors"]) == 3
    assert clean["panel_executed"] is False
    assert clean["attempt_consumed"] is False

    dirty = cast(
        dict[str, Any],
        module.build_clean_preflight(
            ROOT,
            ("sys", "numpy", "jax.numpy", "alberta_framework.evaluation"),
        ),
    )
    assert dirty["valid"] is False
    assert dirty["panel_executed"] is False
    assert dirty["attempt_consumed"] is False
    assert dirty["forbidden_preloaded_modules_observed"] == [
        "alberta_framework.evaluation",
        "jax.numpy",
        "numpy",
    ]


def test_source_postflight_failure_suppresses_endpoint_summary() -> None:
    module = _load()
    report = _synthetic_report(module)
    missing_root = ROOT / "intentionally-missing-future-v2-postflight-root"
    errors = module.validate_postrun_binding(report, missing_root)
    assert errors
    assert any("missing declared source" in error for error in errors)
    try:
        module.summarize_completed_report(report, missing_root)
    except ValueError as error:
        assert "post-run binding failed" in str(error)
    else:
        raise AssertionError("endpoint summary survived a failed source postflight")
