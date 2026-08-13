"""Pure-stdlib checks for the consumed future-utility-v2 outcome."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
OUTCOME = (
    ROOT
    / "alberta_framework/evaluation/"
    "compositional_future_utility_calibration_v2_execution_outcome.py"
)


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_future_v2_execution_outcome", OUTCOME)
    if spec is None or spec.loader is None:
        raise RuntimeError("execution outcome spec unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_consumed_attempt_failed_before_any_arm_endpoint_or_report() -> None:
    outcome = _load()

    assert outcome.validate_execution_outcome() == ()
    assert outcome.ATTEMPTS_AUTHORIZED == outcome.ATTEMPTS_CONSUMED == 1
    assert outcome.FIRST_ARM_COMPILED_SCAN_COMPLETED
    assert outcome.FAILURE_STAGE == "first-arm-primary-endpoint-extraction"
    assert outcome.FAILED_ARM == "current_mix0_decay095_none"
    assert outcome.ARMS_RETURNED == 0
    assert not outcome.PANEL_COMPLETED
    assert not outcome.REPORT_AVAILABLE
    assert outcome.REPORT_SHA256 is None
    assert not outcome.ENDPOINT_CONCLUSIONS_ALLOWED


def test_failure_is_the_invalid_cadence_invariant_not_a_learner_result() -> None:
    outcome = _load()

    assert outcome.FAILURE_TYPE == "RuntimeError"
    assert outcome.FAILURE_MESSAGE == (
        "a strict-margin pass occurred outside a due curation event"
    )
    assert outcome.FAILURE_CLASSIFICATION == "invalid-evaluator-cadence-invariant"
    assert "every update" in outcome.CAUSE
    assert "new development root" in outcome.REPAIR_SCOPE


def test_outcome_grants_no_retry_artifact_endpoint_or_promotion_authority() -> None:
    outcome = _load()

    assert outcome.POSTRUN_SELECTED_SOURCE_MANIFEST_VALID
    assert outcome.DECLARATION_PRE_POST_SHA256_EQUAL
    assert not outcome.DECLARATION_LITERAL_SELF_HASH_ENFORCED_BY_EXECUTOR
    assert not outcome.RESOLVED_SELECTED_MODULE_PATHS_BOUND_BY_EXECUTOR
    assert not outcome.TRANSITIVE_PACKAGE_IMPORT_CLOSURE_BOUND
    assert not outcome.SUCCESS_SEMANTIC_INTEGRITY_GATES_REACHED
    assert not outcome.RERUN_AUTHORIZED
    assert not outcome.RECOVERY_AUTHORIZED
    assert not outcome.OUTPUT_ARTIFACT_WRITTEN
    assert not outcome.SCIENTIFIC_PROMOTION_ALLOWED
    assert not outcome.EVIDENCE_AUTHORIZED
