"""Pure-stdlib contract for the single U0 report-recovery replay."""

from alberta_framework.evaluation\
    .hidden_prototype_two_agent_continual_life_recovery_declaration import (
    ARTIFACT_BYTES_AUTHORIZED,
    CONSUMED_SEED_REPLAY_IS_NEW_EVIDENCE,
    EXPECTED_VALUES_CHANGED_AFTER_FIRST_ATTEMPT,
    FIRST_ATTEMPT_REPORT_RETAINED,
    FIRST_ATTEMPT_RESULT_OBSERVED,
    FULL_REPORT_IDENTITY_CLAIM_ALLOWED,
    OUTPUT_WRITES_ALLOWED,
    RECOVERY_REPLAY_ATTEMPTS_AUTHORIZED,
    SCIENTIFIC_PROMOTION_ALLOWED,
    SOURCE_MANIFEST_CHANGED_AFTER_FIRST_ATTEMPT,
    SOURCE_MANIFEST_DIGEST,
    THRESHOLD_OR_WINNER_AUTHORIZED,
    validate_recovery_declaration,
)


def test_recovery_declaration_is_exactly_one_unchanged_nonpromoting_attempt() -> None:
    assert validate_recovery_declaration() == ()
    assert RECOVERY_REPLAY_ATTEMPTS_AUTHORIZED == 1
    assert not EXPECTED_VALUES_CHANGED_AFTER_FIRST_ATTEMPT
    assert not SOURCE_MANIFEST_CHANGED_AFTER_FIRST_ATTEMPT
    assert not FIRST_ATTEMPT_RESULT_OBSERVED
    assert not FIRST_ATTEMPT_REPORT_RETAINED
    assert not SCIENTIFIC_PROMOTION_ALLOWED
    assert not CONSUMED_SEED_REPLAY_IS_NEW_EVIDENCE
    assert not OUTPUT_WRITES_ALLOWED
    assert ARTIFACT_BYTES_AUTHORIZED == 0
    assert not THRESHOLD_OR_WINNER_AUTHORIZED
    assert not FULL_REPORT_IDENTITY_CLAIM_ALLOWED


def test_recovery_binds_the_already_frozen_source_manifest() -> None:
    assert SOURCE_MANIFEST_DIGEST == (
        "7cfb95f3a96fcb441f2d8e5e471ccbe3fb3fba260c9e987ff8ffc776eca921da"
    )
