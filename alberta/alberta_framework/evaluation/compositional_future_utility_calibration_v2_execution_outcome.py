"""Pure record of the consumed future-utility-v2 execution failure.

The sole authorized attempt completed the first arm's compiled scan and then
failed while extracting its primary endpoints.  The evaluator treated the
all-step ``decision_margin_passed`` diagnostic as if it could be true only on
32-step curation opportunities.  Production computes that diagnostic every
update, so the cadence assertion was invalid.  No arm record, report, artifact,
or endpoint conclusion was produced, and the development root cannot be run
again.
"""

from __future__ import annotations

from typing import Final

SCHEMA: Final = (
    "alberta.compositional-future-utility-calibration-v2-execution-outcome.v1"
)
DECLARATION_SHA256: Final = (
    "1ea4517f0711e2e56fe62cbe408771499e897d1dcc0b2f6f2e53c51e4264ae5c"
)
EVALUATOR_SHA256: Final = (
    "643227c17c944ffdb47ad255a2ae204b2e4be107f207b7ba4e1ce5f93ea9231b"
)
FOCUSED_EVALUATOR_TEST_SHA256: Final = (
    "6cbbf6c13455a69a5fb61b8836e29f8ac07f33d74eb443d501ac09c1d74d2e03"
)
SELECTED_SOURCE_MANIFEST_SHA256: Final = (
    "458e7bdc5a1cf8a522b0c299a36a0494101f25c469007f89b9e0200d770f5e88"
)
PROTOCOL_CONFIG_SHA256: Final = (
    "9eebf20c8052a280d4b9864c8c307be92c40365ce2c3b7724040bd4a42b30b6a"
)
STREAM_SHA256: Final = (
    "bb741db073a13026425d2cc98cce93a1af1d1b65f2abf24ebc97e43b61abd39c"
)
DEVELOPMENT_ROOT_HEX: Final = "0x72B0A3F6"

ATTEMPTS_AUTHORIZED: Final = 1
ATTEMPTS_CONSUMED: Final = 1
PREFLIGHT_VALID: Final = True
ATTEMPT_CONSUMED_BEFORE_EVALUATOR_IMPORT: Final = True
EVALUATOR_IMPORT_BINDINGS_VALID: Final = True
FIRST_ARM_COMPILED_SCAN_COMPLETED: Final = True
FAILURE_STAGE: Final = "first-arm-primary-endpoint-extraction"
FAILURE_TYPE: Final = "RuntimeError"
FAILURE_MESSAGE: Final = "a strict-margin pass occurred outside a due curation event"
FAILURE_CLASSIFICATION: Final = "invalid-evaluator-cadence-invariant"
FAILED_ARM: Final = "current_mix0_decay095_none"
ARMS_RETURNED: Final = 0
PANEL_COMPLETED: Final = False
REPORT_AVAILABLE: Final = False
REPORT_SHA256: Final[str | None] = None
ENDPOINT_CONCLUSIONS_ALLOWED: Final = False
POSTRUN_SELECTED_SOURCE_MANIFEST_VALID: Final = True
DECLARATION_PRE_POST_SHA256_EQUAL: Final = True
DECLARATION_LITERAL_SELF_HASH_ENFORCED_BY_EXECUTOR: Final = False
DECLARATION_SHA256_EXTERNALLY_OBSERVED_BEFORE_ATTEMPT: Final = True
RESOLVED_SELECTED_MODULE_PATHS_BOUND_BY_EXECUTOR: Final = False
TRANSITIVE_PACKAGE_IMPORT_CLOSURE_BOUND: Final = False
SUCCESS_SEMANTIC_INTEGRITY_GATES_REACHED: Final = False
OUTPUT_ARTIFACT_WRITTEN: Final = False
SCIENTIFIC_PROMOTION_ALLOWED: Final = False
EVIDENCE_AUTHORIZED: Final = False
RERUN_AUTHORIZED: Final = False
RECOVERY_AUTHORIZED: Final = False

CAUSE: Final = (
    "the production trace computes decision_margin_passed on every update, while "
    "the failed endpoint extractor required every raw pass to coincide with a "
    "32-step curation opportunity"
)
REPAIR_SCOPE: Final = (
    "separate all-step diagnostic passes from due-opportunity endpoint passes on a "
    "new development root; do not reconstruct or rerun this consumed panel"
)


def validate_execution_outcome() -> tuple[str, ...]:
    """Return fail-closed inconsistencies in the recorded execution facts."""

    errors: list[str] = []
    if ATTEMPTS_AUTHORIZED != 1 or ATTEMPTS_CONSUMED != ATTEMPTS_AUTHORIZED:
        errors.append("the one-attempt budget is inconsistent")
    if not PREFLIGHT_VALID or not ATTEMPT_CONSUMED_BEFORE_EVALUATOR_IMPORT:
        errors.append("the execution chronology is inconsistent")
    if not EVALUATOR_IMPORT_BINDINGS_VALID or not FIRST_ARM_COMPILED_SCAN_COMPLETED:
        errors.append("the recorded failure stage is inconsistent")
    if FAILURE_STAGE != "first-arm-primary-endpoint-extraction":
        errors.append("the failure stage drifted")
    if FAILURE_TYPE != "RuntimeError" or FAILURE_CLASSIFICATION != (
        "invalid-evaluator-cadence-invariant"
    ):
        errors.append("the failure classification drifted")
    if FAILED_ARM != "current_mix0_decay095_none" or ARMS_RETURNED != 0:
        errors.append("the first-arm failure accounting is inconsistent")
    if PANEL_COMPLETED or REPORT_AVAILABLE or REPORT_SHA256 is not None:
        errors.append("the failed attempt acquired a report")
    if ENDPOINT_CONCLUSIONS_ALLOWED:
        errors.append("the failed attempt acquired endpoint authority")
    if not POSTRUN_SELECTED_SOURCE_MANIFEST_VALID:
        errors.append("the recorded independent source postflight is inconsistent")
    if not DECLARATION_PRE_POST_SHA256_EQUAL:
        errors.append("the declaration changed during execution")
    if DECLARATION_LITERAL_SELF_HASH_ENFORCED_BY_EXECUTOR:
        errors.append("the executor's self-hash coverage is overstated")
    if not DECLARATION_SHA256_EXTERNALLY_OBSERVED_BEFORE_ATTEMPT:
        errors.append("the external pre-attempt observation is missing")
    if RESOLVED_SELECTED_MODULE_PATHS_BOUND_BY_EXECUTOR:
        errors.append("the executor's resolved-path coverage is overstated")
    if TRANSITIVE_PACKAGE_IMPORT_CLOSURE_BOUND:
        errors.append("the executor's transitive import coverage is overstated")
    if SUCCESS_SEMANTIC_INTEGRITY_GATES_REACHED:
        errors.append("unreached success-only integrity gates were recorded")
    if OUTPUT_ARTIFACT_WRITTEN:
        errors.append("the failed attempt acquired an artifact")
    if SCIENTIFIC_PROMOTION_ALLOWED or EVIDENCE_AUTHORIZED:
        errors.append("the failed attempt acquired scientific authority")
    if RERUN_AUTHORIZED or RECOVERY_AUTHORIZED:
        errors.append("the consumed root acquired another attempt")
    return tuple(errors)


__all__ = [
    "ARMS_RETURNED",
    "ATTEMPTS_AUTHORIZED",
    "ATTEMPTS_CONSUMED",
    "CAUSE",
    "DECLARATION_LITERAL_SELF_HASH_ENFORCED_BY_EXECUTOR",
    "DECLARATION_PRE_POST_SHA256_EQUAL",
    "DECLARATION_SHA256",
    "ENDPOINT_CONCLUSIONS_ALLOWED",
    "EVALUATOR_SHA256",
    "FAILED_ARM",
    "FAILURE_CLASSIFICATION",
    "FAILURE_MESSAGE",
    "FAILURE_STAGE",
    "FAILURE_TYPE",
    "FIRST_ARM_COMPILED_SCAN_COMPLETED",
    "OUTPUT_ARTIFACT_WRITTEN",
    "PANEL_COMPLETED",
    "POSTRUN_SELECTED_SOURCE_MANIFEST_VALID",
    "RECOVERY_AUTHORIZED",
    "REPAIR_SCOPE",
    "REPORT_AVAILABLE",
    "REPORT_SHA256",
    "RESOLVED_SELECTED_MODULE_PATHS_BOUND_BY_EXECUTOR",
    "RERUN_AUTHORIZED",
    "SCHEMA",
    "SCIENTIFIC_PROMOTION_ALLOWED",
    "SELECTED_SOURCE_MANIFEST_SHA256",
    "SUCCESS_SEMANTIC_INTEGRITY_GATES_REACHED",
    "TRANSITIVE_PACKAGE_IMPORT_CLOSURE_BOUND",
    "validate_execution_outcome",
]
