# mypy: disable-error-code="attr-defined"
"""Resume and authenticated-replay integration for the nonpromoting WP7.4 lane."""

from __future__ import annotations

import dataclasses

import chex
import pytest

import alberta_framework.evaluation.calibrated_extended_search_development as development_module
from alberta_framework.evaluation.calibrated_extended_search_development import (
    ASSESSMENT_STATUS,
    CALIBRATED_EXTENDED_SEARCH_DEVELOPMENT_CHECKPOINT_SCHEMA,
    MODEL_FREE_VS_OPTION_MODEL,
    PRIMITIVE_VS_COMBINED,
    CalibratedExtendedSearchDevelopmentConfig,
    CalibratedExtendedSearchDevelopmentEvaluator,
    CalibratedSearchAuthenticatedReplayValidation,
    CalibratedSearchDevelopmentState,
    CalibratedSearchDevelopmentSuite,
    authenticate_calibrated_search_development_replay,
    validate_calibrated_search_development_suite,
)

pytestmark = [pytest.mark.integration, pytest.mark.slow]


@dataclasses.dataclass(frozen=True, slots=True)
class _ResumedRun:
    evaluator: CalibratedExtendedSearchDevelopmentEvaluator
    partial: CalibratedSearchDevelopmentState
    checkpoint: dict[str, object]
    restored: CalibratedSearchDevelopmentState
    suite: CalibratedSearchDevelopmentSuite
    validation: CalibratedSearchAuthenticatedReplayValidation


@pytest.fixture(scope="module")
def resumed_run() -> _ResumedRun:
    config = CalibratedExtendedSearchDevelopmentConfig()
    evaluator = CalibratedExtendedSearchDevelopmentEvaluator(config)
    partial = evaluator.advance(evaluator.init(), steps=2)
    checkpoint = evaluator.checkpoint_payload(partial)

    resumed_evaluator = CalibratedExtendedSearchDevelopmentEvaluator(config)
    restored = resumed_evaluator.restore_checkpoint(checkpoint)
    final_state = resumed_evaluator.advance(
        restored, steps=config.num_steps - int(restored.step_index)
    )
    suite = resumed_evaluator.finalize(final_state)
    validation = authenticate_calibrated_search_development_replay(suite)
    return _ResumedRun(
        evaluator=resumed_evaluator,
        partial=partial,
        checkpoint=checkpoint,
        restored=restored,
        suite=suite,
        validation=validation,
    )


def test_checkpoint_resume_is_exact_and_completes_the_fixed_trace(
    resumed_run: _ResumedRun,
) -> None:
    chex.assert_trees_all_equal(resumed_run.partial, resumed_run.restored)
    assert resumed_run.evaluator.validate_state(resumed_run.restored) == ()
    assert resumed_run.checkpoint["schema"] == (
        CALIBRATED_EXTENDED_SEARCH_DEVELOPMENT_CHECKPOINT_SCHEMA
    )
    assert set(resumed_run.checkpoint) == {
        "schema",
        "config",
        "config_sha256",
        "protocol_sha256",
        "source_runtime_manifest",
        "model_snapshot_sha256",
        "evaluator_trace_sha256",
        "state",
        "state_sha256",
        "checkpoint_sha256",
    }
    assert all(
        len(str(resumed_run.checkpoint[name])) == 64
        for name in (
            "config_sha256",
            "protocol_sha256",
            "model_snapshot_sha256",
            "evaluator_trace_sha256",
            "state_sha256",
            "checkpoint_sha256",
        )
    )
    assert validate_calibrated_search_development_suite(resumed_run.suite) == ()


def test_checkpoint_rejects_digest_and_cross_configuration_tamper(
    resumed_run: _ResumedRun,
) -> None:
    digest_tamper = dict(resumed_run.checkpoint)
    digest_tamper["state_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="checkpoint digest differs"):
        resumed_run.evaluator.restore_checkpoint(digest_tamper)

    manifest_tamper = dict(resumed_run.checkpoint)
    manifest_tamper["source_runtime_manifest"] = None
    with pytest.raises(ValueError, match="checkpoint digest differs"):
        resumed_run.evaluator.restore_checkpoint(manifest_tamper)

    different = CalibratedExtendedSearchDevelopmentEvaluator(
        CalibratedExtendedSearchDevelopmentConfig(seed=123)
    )
    with pytest.raises(ValueError, match="checkpoint binding differs"):
        different.restore_checkpoint(resumed_run.checkpoint)


def _rebind_checkpoint(
    checkpoint: dict[str, object], state: CalibratedSearchDevelopmentState
) -> dict[str, object]:
    rebound = dict(checkpoint)
    rebound["state"] = state
    rebound["state_sha256"] = development_module._exact_sha256(state)
    rebound.pop("checkpoint_sha256")
    rebound["checkpoint_sha256"] = development_module._exact_sha256(rebound)
    return rebound


def test_checkpoint_rejects_digest_aware_controller_and_diagnostic_forgery(
    resumed_run: _ResumedRun,
) -> None:
    partial = resumed_run.partial
    changed_controller = partial.controller_states[0].replace(
        q_values=partial.controller_states[0].q_values.at[0, 0].add(9.0)
    )
    forged_controller = partial.replace(
        controller_states=(
            changed_controller,
            partial.controller_states[1],
            partial.controller_states[2],
            partial.controller_states[3],
        )
    )
    with pytest.raises(ValueError, match="canonical prefix.*controller state 0"):
        resumed_run.evaluator.restore_checkpoint(
            _rebind_checkpoint(resumed_run.checkpoint, forged_controller)
        )

    forged_diagnostic = partial.replace(
        priority_sum=partial.priority_sum.at[0, 0].add(123.0)
    )
    with pytest.raises(ValueError, match="canonical prefix.*diagnostic priority_sum"):
        resumed_run.evaluator.restore_checkpoint(
            _rebind_checkpoint(resumed_run.checkpoint, forged_diagnostic)
        )


def test_authenticated_replay_verifies_source_bound_resumed_suite(
    resumed_run: _ResumedRun,
) -> None:
    validation = resumed_run.validation
    assert validation.assessment_status == ASSESSMENT_STATUS
    assert validation.source_runtime_verified
    assert validation.structural_validation_passed
    assert validation.authenticated_replay_verified
    assert validation.replay_suite_binding_sha256 == resumed_run.suite.suite_binding_sha256
    assert validation.errors == ()
    # Authentication is a separate validation result; it grants the raw suite no authority.
    assert resumed_run.suite.authenticated_replay_verified is False
    assert resumed_run.suite.thresholds is None
    assert resumed_run.suite.aggregate_verdict is None
    assert resumed_run.suite.scientific_promotion_allowed is False


def test_authenticated_replay_rejects_digest_aware_structural_tamper(
    resumed_run: _ResumedRun,
) -> None:
    """Even a fully rebound descriptive edit must differ from exact causal replay."""

    evaluator = resumed_run.evaluator
    suite = resumed_run.suite
    first = suite.arm_records[0]
    changed_first = dataclasses.replace(
        first,
        summary=dataclasses.replace(
            first.summary,
            mean_priority=first.summary.mean_priority + 0.125,
        ),
    )
    records = (
        changed_first,
        suite.arm_records[1],
        suite.arm_records[2],
        suite.arm_records[3],
    )
    contrasts = (
        evaluator._contrast(MODEL_FREE_VS_OPTION_MODEL, records[0], records[2]),
        evaluator._contrast(PRIMITIVE_VS_COMBINED, records[1], records[3]),
    )
    provisional = dataclasses.replace(
        suite,
        arm_records=records,
        contrasts=contrasts,
        suite_binding_sha256="",
        replay_authenticator_sha256="",
    )
    rebound = dataclasses.replace(
        provisional,
        suite_binding_sha256=evaluator._suite_binding(provisional),
    )
    rebound = dataclasses.replace(
        rebound,
        replay_authenticator_sha256=evaluator._replay_authenticator(rebound),
    )

    assert validate_calibrated_search_development_suite(rebound) == ()
    validation = authenticate_calibrated_search_development_replay(rebound)
    assert validation.source_runtime_verified
    assert validation.structural_validation_passed
    assert not validation.authenticated_replay_verified
    assert "authenticated replay differs from the supplied raw suite" in validation.errors
