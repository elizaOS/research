"""Pure-stdlib contract for the consumed U0 recovery outcome."""

from alberta_framework.evaluation import (
    hidden_prototype_two_agent_continual_life_recovery_outcome as outcome,
)


def test_recovery_outcome_fails_closed_on_midrun_source_drift() -> None:
    assert outcome.validate_recovery_outcome() == ()
    assert outcome.ATTEMPTS_AUTHORIZED == outcome.ATTEMPTS_CONSUMED == 1
    assert outcome.PREFLIGHT_SOURCE_MANIFEST_VALID
    assert not outcome.POSTRUN_SOURCE_MANIFEST_VALID
    assert outcome.DECLARED_REPORT_FIELD_MISMATCHES == ()
    assert outcome.DECLARED_REPORT_SUBSET_MATCHED
    assert not outcome.COMPARATOR_DECLARED_FIELDS_EXACT
    assert outcome.COMPARATOR_CONCLUSION == "source-manifest-mismatch"
    assert outcome.EXPECTED_PROTOTYPE_SOURCE_SHA256 != (
        outcome.OBSERVED_POSTRUN_PROTOTYPE_SOURCE_SHA256
    )


def test_recovery_outcome_grants_no_retry_output_identity_or_promotion() -> None:
    assert not outcome.FURTHER_RECOVERY_AUTHORIZED
    assert not outcome.REPORT_RETAINED
    assert not outcome.ARTIFACT_WRITTEN
    assert not outcome.FULL_REPORT_IDENTITY_CLAIMED
    assert not outcome.RUNTIME_IDENTITY_BOUND
    assert not outcome.SCIENTIFIC_PROMOTION_ALLOWED
    assert not outcome.RECOVERY_REPLAY_IS_NEW_EVIDENCE
