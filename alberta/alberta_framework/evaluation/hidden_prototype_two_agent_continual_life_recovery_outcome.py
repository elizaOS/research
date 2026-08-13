"""Pure record of the consumed, nonpromoting U0 report-recovery outcome.

The sole authorized recovery reached the comparator after running both arms.
Every predeclared report field matched, but the selected Prototype source
changed after the clean preflight and before comparison.  The fail-closed
outcome is therefore a source-manifest mismatch, not a compatibility result.
No full report or artifact was retained.
"""

from __future__ import annotations

from typing import Final

SCHEMA: Final = "alberta.hidden-prototype-two-agent-replay-recovery-outcome.v1"
RECOVERY_DECLARATION_SHA256: Final = (
    "140b85781fa871a59e2f614543bccf3532f460bb5b4716a5dec3eab00f72e998"
)
REPLAY_DECLARATION_SHA256: Final = (
    "0918be00dcfc1e100a87f3a90ad737a9b5ba63bcfd6cae98dbb07c918cabdc4e"
)
ATTEMPTS_AUTHORIZED: Final = 1
ATTEMPTS_CONSUMED: Final = 1
PREFLIGHT_SOURCE_MANIFEST_VALID: Final = True
POSTRUN_SOURCE_MANIFEST_VALID: Final = False
EXPECTED_PROTOTYPE_SOURCE_SHA256: Final = (
    "1e05b1f8ea935ac454a485b4ebb5dcff1e2676d13574735ed2d277b7a366f25c"
)
OBSERVED_POSTRUN_PROTOTYPE_SOURCE_SHA256: Final = (
    "37fe39e56eca9395a87a7df83f9f07f0e40de5b6ceb91886e2a938b888964571"
)
DECLARED_REPORT_FIELD_MISMATCHES: Final[tuple[str, ...]] = ()
DECLARED_REPORT_SUBSET_MATCHED: Final = True
COMPARATOR_DECLARED_FIELDS_EXACT: Final = False
COMPARATOR_CONCLUSION: Final = "source-manifest-mismatch"
COMPARATOR_COVERAGE: Final = "partial"
FULL_REPORT_IDENTITY_CLAIMED: Final = False
RUNTIME_IDENTITY_BOUND: Final = False
REPORT_RETAINED: Final = False
ARTIFACT_WRITTEN: Final = False
SCIENTIFIC_PROMOTION_ALLOWED: Final = False
RECOVERY_REPLAY_IS_NEW_EVIDENCE: Final = False
FURTHER_RECOVERY_AUTHORIZED: Final = False


def validate_recovery_outcome() -> tuple[str, ...]:
    """Return fail-closed inconsistencies in the immutable outcome facts."""

    errors: list[str] = []
    if ATTEMPTS_AUTHORIZED != 1 or ATTEMPTS_CONSUMED != ATTEMPTS_AUTHORIZED:
        errors.append("the one-attempt recovery budget is inconsistent")
    if not PREFLIGHT_SOURCE_MANIFEST_VALID or POSTRUN_SOURCE_MANIFEST_VALID:
        errors.append("the preflight/post-run source chronology is inconsistent")
    if EXPECTED_PROTOTYPE_SOURCE_SHA256 == OBSERVED_POSTRUN_PROTOTYPE_SOURCE_SHA256:
        errors.append("the recorded source mismatch does not differ")
    if DECLARED_REPORT_FIELD_MISMATCHES or not DECLARED_REPORT_SUBSET_MATCHED:
        errors.append("the declared report-subset comparison is inconsistent")
    if COMPARATOR_DECLARED_FIELDS_EXACT:
        errors.append("the combined comparator incorrectly ignores source validity")
    if COMPARATOR_CONCLUSION != "source-manifest-mismatch":
        errors.append("the fail-closed conclusion drifted")
    if FULL_REPORT_IDENTITY_CLAIMED or RUNTIME_IDENTITY_BOUND:
        errors.append("the partial replay acquired unsupported identity coverage")
    if REPORT_RETAINED or ARTIFACT_WRITTEN:
        errors.append("the in-memory recovery acquired retained output")
    if SCIENTIFIC_PROMOTION_ALLOWED or RECOVERY_REPLAY_IS_NEW_EVIDENCE:
        errors.append("the consumed recovery acquired scientific authority")
    if FURTHER_RECOVERY_AUTHORIZED:
        errors.append("an additional recovery attempt was authorized")
    return tuple(errors)


__all__ = [
    "ARTIFACT_WRITTEN",
    "ATTEMPTS_AUTHORIZED",
    "ATTEMPTS_CONSUMED",
    "COMPARATOR_CONCLUSION",
    "COMPARATOR_COVERAGE",
    "COMPARATOR_DECLARED_FIELDS_EXACT",
    "DECLARED_REPORT_FIELD_MISMATCHES",
    "DECLARED_REPORT_SUBSET_MATCHED",
    "EXPECTED_PROTOTYPE_SOURCE_SHA256",
    "FULL_REPORT_IDENTITY_CLAIMED",
    "FURTHER_RECOVERY_AUTHORIZED",
    "OBSERVED_POSTRUN_PROTOTYPE_SOURCE_SHA256",
    "POSTRUN_SOURCE_MANIFEST_VALID",
    "PREFLIGHT_SOURCE_MANIFEST_VALID",
    "RECOVERY_DECLARATION_SHA256",
    "RECOVERY_REPLAY_IS_NEW_EVIDENCE",
    "REPLAY_DECLARATION_SHA256",
    "REPORT_RETAINED",
    "RUNTIME_IDENTITY_BOUND",
    "SCHEMA",
    "SCIENTIFIC_PROMOTION_ALLOWED",
    "validate_recovery_outcome",
]
