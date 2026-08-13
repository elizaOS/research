"""Nonpromoting recovery declaration for one lost U0 compatibility report.

The first current-source replay completed both arms and constructed its partial
comparison.  A caller-side compact formatter then requested the nonexistent
top-level metric ``a2_early_mean_agent_reward`` instead of reading the nested
``phase_mean_agent_reward["A2"]["early_mean"]`` field.  The process exited
before printing or retaining the already-computed comparison.

This declaration authorizes exactly one diagnostic recovery replay with the
same source manifest and completely unchanged expected values.  It cannot
promote evidence, select a model, alter a threshold, write an artifact, or
claim that the repeated consumed-root execution is a new scientific result.
"""

from __future__ import annotations

from typing import Final

RECOVERY_SCHEMA: Final = (
    "alberta.hidden-prototype-two-agent-replay-report-recovery.v1"
)
ORIGINAL_DECLARATION_SCHEMA: Final = (
    "alberta.hidden-prototype-two-agent-replay-declaration.v1"
)
SOURCE_MANIFEST_DIGEST: Final = (
    "7cfb95f3a96fcb441f2d8e5e471ccbe3fb3fba260c9e987ff8ffc776eca921da"
)
FIRST_ATTEMPT_STATUS: Final = "life-completed-comparison-constructed-report-not-retained"
FIRST_ATTEMPT_FAILURE: Final = (
    "post-run formatter KeyError: a2_early_mean_agent_reward"
)
FIRST_ATTEMPT_RESULT_OBSERVED: Final = False
FIRST_ATTEMPT_REPORT_RETAINED: Final = False
RECOVERY_REPLAY_ATTEMPTS_AUTHORIZED: Final = 1
EXPECTED_VALUES_CHANGED_AFTER_FIRST_ATTEMPT: Final = False
SOURCE_MANIFEST_CHANGED_AFTER_FIRST_ATTEMPT: Final = False
DEVELOPMENT_ONLY: Final = True
SCIENTIFIC_PROMOTION_ALLOWED: Final = False
CONSUMED_SEED_REPLAY_IS_NEW_EVIDENCE: Final = False
OUTPUT_WRITES_ALLOWED: Final = False
ARTIFACT_BYTES_AUTHORIZED: Final = 0
THRESHOLD_OR_WINNER_AUTHORIZED: Final = False
FULL_REPORT_IDENTITY_CLAIM_ALLOWED: Final = False
PERMITTED_CONCLUSION_SCOPE: Final = "declared-fields-exact-partial-coverage-or-discrepancy"


def validate_recovery_declaration() -> tuple[str, ...]:
    """Return fail-closed declaration errors without importing JAX."""

    errors: list[str] = []
    if RECOVERY_REPLAY_ATTEMPTS_AUTHORIZED != 1:
        errors.append("recovery replay authorization is not exactly one attempt")
    if EXPECTED_VALUES_CHANGED_AFTER_FIRST_ATTEMPT:
        errors.append("expected values changed after the lost first report")
    if SOURCE_MANIFEST_CHANGED_AFTER_FIRST_ATTEMPT:
        errors.append("source manifest changed after the lost first report")
    if FIRST_ATTEMPT_RESULT_OBSERVED or FIRST_ATTEMPT_REPORT_RETAINED:
        errors.append("first-attempt result/report status is inconsistent")
    if SCIENTIFIC_PROMOTION_ALLOWED or CONSUMED_SEED_REPLAY_IS_NEW_EVIDENCE:
        errors.append("recovery replay acquired scientific authority")
    if OUTPUT_WRITES_ALLOWED or ARTIFACT_BYTES_AUTHORIZED != 0:
        errors.append("recovery replay acquired output authority")
    if THRESHOLD_OR_WINNER_AUTHORIZED or FULL_REPORT_IDENTITY_CLAIM_ALLOWED:
        errors.append("recovery replay acquired selection or full-identity authority")
    return tuple(errors)


__all__ = [
    "ARTIFACT_BYTES_AUTHORIZED",
    "CONSUMED_SEED_REPLAY_IS_NEW_EVIDENCE",
    "DEVELOPMENT_ONLY",
    "EXPECTED_VALUES_CHANGED_AFTER_FIRST_ATTEMPT",
    "FIRST_ATTEMPT_FAILURE",
    "FIRST_ATTEMPT_REPORT_RETAINED",
    "FIRST_ATTEMPT_RESULT_OBSERVED",
    "FIRST_ATTEMPT_STATUS",
    "FULL_REPORT_IDENTITY_CLAIM_ALLOWED",
    "ORIGINAL_DECLARATION_SCHEMA",
    "OUTPUT_WRITES_ALLOWED",
    "PERMITTED_CONCLUSION_SCOPE",
    "RECOVERY_REPLAY_ATTEMPTS_AUTHORIZED",
    "RECOVERY_SCHEMA",
    "SCIENTIFIC_PROMOTION_ALLOWED",
    "SOURCE_MANIFEST_CHANGED_AFTER_FIRST_ATTEMPT",
    "SOURCE_MANIFEST_DIGEST",
    "THRESHOLD_OR_WINNER_AUTHORIZED",
    "validate_recovery_declaration",
]
