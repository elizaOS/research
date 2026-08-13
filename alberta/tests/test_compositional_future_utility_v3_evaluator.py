"""Nonexecuting contract tests for the post-consumption v3 evaluator."""

from __future__ import annotations

import ast
import dataclasses
import inspect
from pathlib import Path
from typing import Any, cast

import pytest

from alberta_framework.evaluation import (
    _compositional_future_utility_v3_evaluator as evaluator,
)
from alberta_framework.evaluation import (
    _compositional_future_utility_v3_report_gate as report_gate,
)
from alberta_framework.evaluation import (
    compositional_future_utility_calibration_v3_protocol as protocol,
)

MODULE_PATH = (
    Path(__file__).parents[1]
    / "alberta_framework/evaluation/_compositional_future_utility_v3_evaluator.py"
)


def _sha(character: str) -> str:
    return character * 64


def _bindings() -> report_gate.ExpectedExecutionBindings:
    return report_gate.ExpectedExecutionBindings(
        execution_source_closure_sha256=_sha("a"),
        bootstrap_sha256=_sha("b"),
        ledger_primitive_sha256=_sha("c"),
        declared_loader_sha256=_sha("d"),
        genesis_sha256=_sha("e"),
        started_sha256=_sha("f"),
    )


def _complete_progress() -> evaluator.V3EvaluatorProgress:
    progress = evaluator.V3EvaluatorProgress()
    progress._enter()
    for arm_name in protocol.ARM_NAMES:
        progress._start_arm(arm_name)
        progress._complete_scan(arm_name)
        progress._complete_arm_record(arm_name)
    progress._complete()
    return progress


def test_entry_surface_has_no_convenience_defaults_or_ledger_path() -> None:
    signature = inspect.signature(evaluator.evaluate_v3_operational_panel)

    assert tuple(signature.parameters) == (
        "attempt_capability",
        "attempt_authorizer",
        "expected_bindings",
        "progress",
    )
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        and parameter.default is inspect.Parameter.empty
        for parameter in signature.parameters.values()
    )
    public = set(evaluator.__all__)
    assert not any(
        token in name.lower()
        for name in public
        if callable(getattr(evaluator, name))
        for token in ("write", "ledger", "issue", "retry", "select", "threshold")
    )


def test_source_uses_public_execution_analysis_and_no_source_replace_workaround() -> None:
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    attributes = {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }
    calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }

    assert "analyze_compositional_control_life_arm_execution" in calls
    assert "validate_bound_v3_source" in calls
    assert not {
        "_curation_decision_audit",
        "_validate_curation_decision_audit",
        "_structural_trajectory",
        "_array_tree_sha256",
    } & attributes
    assert "replace" not in calls
    assert not {"open", "write_text", "write_bytes", "unlink", "mkdir"} & calls


def test_source_keeps_the_learner_genesis_key_unsplit() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")

    assert "bound.source.learner_key" in source
    assert "jr.split" not in source
    assert "jr.fold_in" not in source
    assert "random.split" not in source
    assert "random.fold_in" not in source


def test_attempt_authorizer_receives_opaque_capability_stage_and_bindings() -> None:
    capability = object()
    bindings = _bindings()
    observed: list[tuple[object, str, report_gate.ExpectedExecutionBindings]] = []

    def authorize(
        candidate: object,
        stage: str,
        expected: report_gate.ExpectedExecutionBindings,
    ) -> bool:
        observed.append((candidate, stage, expected))
        return candidate is capability and stage == "fixture-preflight" and expected is bindings

    evaluator._require_active_attempt(
        capability,
        authorize,
        bindings,
        stage="fixture-preflight",
    )

    assert observed == [(capability, "fixture-preflight", bindings)]


@pytest.mark.parametrize("answer", [False, 0, 1, None, "yes"])
def test_attempt_authorizer_fails_closed(answer: object) -> None:
    def authorize(
        _capability: object,
        _stage: str,
        _bindings: report_gate.ExpectedExecutionBindings,
    ) -> Any:
        return answer

    expected_error = PermissionError if answer is False else TypeError
    with pytest.raises(expected_error):
        evaluator._require_active_attempt(
            object(),
            authorize,
            _bindings(),
            stage="fixture-preflight",
        )


def test_attempt_authorizer_rejects_none_capability_before_callback() -> None:
    called = False

    def authorize(
        _capability: object,
        _stage: str,
        _bindings: report_gate.ExpectedExecutionBindings,
    ) -> bool:
        nonlocal called
        called = True
        return True

    with pytest.raises(TypeError, match="non-None"):
        evaluator._require_active_attempt(
            None,
            authorize,
            _bindings(),
            stage="fixture-preflight",
        )
    assert called is False


def test_progress_defines_panel_completion_at_fifth_returned_scan() -> None:
    progress = evaluator.V3EvaluatorProgress()
    progress._enter()
    for arm_index, arm_name in enumerate(protocol.ARM_NAMES):
        progress._start_arm(arm_name)
        progress._complete_scan(arm_name)
        after_scan = progress.snapshot()
        assert after_scan.scans_completed == arm_index + 1
        assert after_scan.panel_completed is (
            arm_index + 1 == len(protocol.ARM_NAMES)
        )
        if arm_index == len(protocol.ARM_NAMES) - 1:
            assert after_scan.arm_records_completed == 4
            assert after_scan.current_arm == arm_name
            progress._record_failure(RuntimeError("fixture fifth-arm postscan failure"))
            failed = progress.snapshot()
            assert failed.failed is True
            assert failed.panel_completed is True
            assert failed.succeeded is False
            assert failed.failure_type == "RuntimeError"
            assert failed.failure_message == "fixture fifth-arm postscan failure"
            break
        progress._complete_arm_record(arm_name)


def test_progress_before_fifth_scan_is_not_panel_completed() -> None:
    progress = evaluator.V3EvaluatorProgress()
    progress._enter()
    for arm_name in protocol.ARM_NAMES[:3]:
        progress._start_arm(arm_name)
        progress._complete_scan(arm_name)
        progress._complete_arm_record(arm_name)

    progress._record_failure(ValueError("fixture early failure"))
    snapshot = progress.snapshot()

    assert snapshot.scans_completed == 3
    assert snapshot.arm_records_completed == 3
    assert snapshot.panel_completed is False
    assert snapshot.failed is True


def test_progress_success_requires_all_five_records_and_is_single_use() -> None:
    progress = _complete_progress()
    snapshot = progress.snapshot()

    assert snapshot.to_config() == {
        "entered": True,
        "stage": "completed",
        "current_arm": None,
        "scans_completed": 5,
        "arm_records_completed": 5,
        "panel_completed": True,
        "succeeded": True,
        "failed": False,
        "failure_type": None,
        "failure_message": None,
    }
    with pytest.raises(RuntimeError, match="single-use"):
        progress._enter()


def test_progress_can_seal_failure_after_completion_gate_before_return() -> None:
    progress = _complete_progress()

    progress._record_failure(RuntimeError("fixture result construction failed"))
    snapshot = progress.snapshot()

    assert snapshot.panel_completed is True
    assert snapshot.succeeded is False
    assert snapshot.failed is True
    assert snapshot.stage == "result-construction-failed"
    assert snapshot.failure_message == "fixture result construction failed"


def test_progress_rejects_out_of_order_arm_and_record_completion() -> None:
    progress = evaluator.V3EvaluatorProgress()
    progress._enter()

    with pytest.raises(RuntimeError, match="order"):
        progress._start_arm(protocol.ARM_NAMES[1])
    progress._start_arm(protocol.ARM_NAMES[0])
    with pytest.raises(RuntimeError, match="record completion"):
        progress._complete_arm_record(protocol.ARM_NAMES[0])


def test_result_is_frozen_authority_free_and_returns_fresh_report_copies() -> None:
    body: dict[str, object] = {"fixture": "non-v3-execution"}
    report = {
        **body,
        "report_sha256": report_gate.canonical_json_sha256(body),
    }
    result = evaluator.V3OperationalEvaluationResult(
        canonical_report_json=report_gate.canonical_json(report),
        report_sha256=cast(str, report["report_sha256"]),
        progress=_complete_progress().snapshot(),
    )

    first = result.report
    second = result.report
    first["fixture"] = "mutated copy"

    assert second["fixture"] == "non-v3-execution"
    assert result.development_only is True
    assert result.output_writes_allowed is False
    assert result.evidence_authorized is False
    assert result.scientific_promotion_allowed is False
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.report_sha256 = _sha("0")  # type: ignore[misc]


def test_result_rejects_noncanonical_text_and_unclosed_hash() -> None:
    progress = _complete_progress().snapshot()
    report = {"report_sha256": _sha("a"), "fixture": True}

    with pytest.raises(ValueError, match="not canonical"):
        evaluator.V3OperationalEvaluationResult(
            canonical_report_json='{"report_sha256": "' + _sha("a") + '", "fixture": true}',
            report_sha256=_sha("a"),
            progress=progress,
        )
    with pytest.raises(ValueError, match="does not reconstruct"):
        evaluator.V3OperationalEvaluationResult(
            canonical_report_json=report_gate.canonical_json(report),
            report_sha256=_sha("a"),
            progress=progress,
        )


def test_module_declares_capability_boundary_is_not_a_sandbox() -> None:
    docstring = ast.get_docstring(ast.parse(MODULE_PATH.read_text(encoding="utf-8")))

    assert docstring is not None
    assert "not a sandbox" in docstring
    assert "cannot issue a root" in docstring
    assert "write output" in docstring
