"""Development contracts for the matched nonlinear/SARSA RiverSwim lane."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import pytest

import alberta_framework.evaluation.nonlinear_discrete_control_matched_development as lane
from alberta_framework.evaluation.nonlinear_discrete_control_matched_development import (
    MatchedDiscreteControlDevelopmentConfig,
    load_matched_discrete_control_checkpoint,
    run_matched_discrete_control_development,
    save_matched_discrete_control_checkpoint,
    validate_matched_discrete_control_development_report,
)

pytestmark = pytest.mark.development


def _small_config() -> MatchedDiscreteControlDevelopmentConfig:
    return MatchedDiscreteControlDevelopmentConfig(
        seeds=(101,),
        phase_length=2,
        summary_window=1,
    )


@pytest.fixture(scope="module")
def report() -> dict[str, object]:
    return run_matched_discrete_control_development(_small_config())


def _arms(report: Mapping[str, object]) -> list[Mapping[str, object]]:
    run = cast(list[Mapping[str, object]], report["runs"])[0]
    return cast(list[Mapping[str, object]], run["arms"])


def test_config_is_strict_frozen_and_permanently_nonpromoting() -> None:
    config = _small_config()
    payload = config.to_config()
    assert MatchedDiscreteControlDevelopmentConfig.from_config(payload) == config
    assert payload["development_only"] is True
    assert payload["assessment_status"] == "not_assessed"
    assert payload["scientific_promotion_allowed"] is False
    assert payload["winner_selection_allowed"] is False
    assert payload["performance_thresholds_applied"] is False
    assert payload["regime_schedule"] == ["A", "B", "A"]
    assert payload["environment"] == "uninterrupted-six-state-riverswim"

    promoted = copy.deepcopy(payload)
    promoted["scientific_promotion_allowed"] = True
    with pytest.raises(ValueError, match="nonpromoting"):
        MatchedDiscreteControlDevelopmentConfig.from_config(promoted)

    with pytest.raises(ValueError):
        MatchedDiscreteControlDevelopmentConfig(phase_length=3)
    with pytest.raises(ValueError):
        MatchedDiscreteControlDevelopmentConfig(seeds=(True,))


def test_report_replays_and_retains_full_prequential_arm_traces(
    report: dict[str, object],
) -> None:
    validation = validate_matched_discrete_control_development_report(report)
    assert validation.valid, validation.errors
    assert report["development_only"] is True
    assert report["assessment_status"] == "not_assessed"
    assert report["scientific_promotion_allowed"] is False
    assert report["winner_selection_allowed"] is False
    assert report["performance_thresholds_applied"] is False

    arms = _arms(report)
    assert [arm["arm"] for arm in arms] == ["nonlinear_actor_critic", "differential_sarsa"]
    assert all(len(cast(list[object], arm["prequential_trace"])) == 6 for arm in arms)

    nonlinear, sarsa = arms
    assert (
        cast(Mapping[str, object], nonlinear["policy_semantics"])["successor_policy_timing"]
        == "post-update"
    )
    assert (
        cast(Mapping[str, object], sarsa["policy_semantics"])["successor_policy_timing"]
        == "pre-update"
    )
    assert (
        cast(Mapping[str, object], nonlinear["policy_semantics"])["behavior"]
        == "epsilon-uniform-mixture-of-current-softmax-target"
    )
    assert (
        cast(Mapping[str, object], sarsa["policy_semantics"])["behavior"]
        == "epsilon-uniform-mixture-of-lowest-index-greedy-target"
    )

    nonlinear_trace = cast(list[Mapping[str, object]], nonlinear["prequential_trace"])
    sarsa_trace = cast(list[Mapping[str, object]], sarsa["prequential_trace"])
    assert [row["environment_rng_words"] for row in nonlinear_trace] == [
        row["environment_rng_words"] for row in sarsa_trace
    ]
    assert [row["action_rng_words"] for row in nonlinear_trace] == [
        row["action_rng_words"] for row in sarsa_trace
    ]
    assert all(row["policy_bound_before_outcome"] is True for row in nonlinear_trace)
    assert all(row["policy_bound_before_outcome"] is True for row in sarsa_trace)
    assert all(row["update_applied"] is True for row in nonlinear_trace)
    assert all(row["update_applied"] is True for row in sarsa_trace)


def test_diagnostics_and_work_accounting_do_not_invent_a_matched_winner(
    report: dict[str, object],
) -> None:
    run = cast(list[Mapping[str, object]], report["runs"])[0]
    comparison = cast(Mapping[str, object], run["comparison"])
    assert comparison["winner"] is None
    assert comparison["verdict"] == "not_assessed"
    assert comparison["performance_thresholds_applied"] is False
    work = cast(Mapping[str, object], comparison["work_matching"])
    assert work["environment_transition_opportunities_match"] is True
    assert work["learner_update_opportunities_match"] is True
    assert work["categorical_action_draws_match"] is True
    assert work["successor_policy_timing_matches"] is False
    assert work["realized_scalar_update_work_matches"] is False
    assert work["exact_realized_work_matched"] is False
    assert cast(list[str], work["reported_mismatches"])

    for arm in _arms(report):
        diagnostics = cast(Mapping[str, object], arm["diagnostics"])
        assert set(diagnostics) == {
            "actor",
            "critic_or_action_value",
            "reward_rate",
            "policy_churn",
            "recovery",
            "return",
        }
        resources = cast(Mapping[str, object], arm["resources"])
        assert cast(int, resources["persistent_state_nbytes"]) > 0
        assert cast(int, resources["logical_trace_nbytes"]) > 0
        operations = cast(Mapping[str, object], arm["logical_operations"])
        assert operations["environment_transitions"] == 6
        assert operations["learner_update_attempts"] == 6
        assert operations["learner_update_commits"] == 6


def test_validator_rejects_trace_work_and_claim_tampering(
    report: dict[str, object],
) -> None:
    tampered = copy.deepcopy(report)
    run = cast(dict[str, object], cast(list[object], tampered["runs"])[0])
    arm = cast(dict[str, object], cast(list[object], run["arms"])[0])
    row = cast(dict[str, object], cast(list[object], arm["prequential_trace"])[0])
    row["reward"] = cast(float, row["reward"]) + 1.0
    lane._redigest_report(tampered)
    validation = validate_matched_discrete_control_development_report(tampered)
    assert not validation.valid
    assert any("replay" in error for error in validation.errors)

    promoted = copy.deepcopy(report)
    promoted["scientific_promotion_allowed"] = True
    lane._redigest_report(promoted)
    validation = validate_matched_discrete_control_development_report(promoted)
    assert not validation.valid
    assert any("nonpromoting" in error for error in validation.errors)


def test_checkpoint_roundtrip_is_source_config_seed_and_state_bound(tmp_path: Path) -> None:
    config = _small_config()
    state = lane.initialize_matched_discrete_control_run(config, seed=101)
    advanced, _ = lane.advance_matched_discrete_control_run(
        config,
        state,
        stop_step=config.checkpoint_step,
    )
    checkpoint = tmp_path / "matched-control"
    save_matched_discrete_control_checkpoint(advanced, checkpoint, config=config)
    restored = load_matched_discrete_control_checkpoint(
        advanced,
        checkpoint,
        config=config,
    )
    assert lane.matched_discrete_control_run_state_sha256(restored) == (
        lane.matched_discrete_control_run_state_sha256(advanced)
    )

    resumed, suffix = lane.advance_matched_discrete_control_run(
        config,
        restored,
        stop_step=config.total_steps,
    )
    uninterrupted, trace = lane.advance_matched_discrete_control_run(
        config,
        lane.initialize_matched_discrete_control_run(config, seed=101),
        stop_step=config.total_steps,
    )
    assert len(suffix.nonlinear) == config.total_steps - config.checkpoint_step
    assert len(trace.nonlinear) == config.total_steps
    assert lane.matched_discrete_control_run_state_sha256(resumed) == (
        lane.matched_discrete_control_run_state_sha256(uninterrupted)
    )

    wrong_config = MatchedDiscreteControlDevelopmentConfig(
        seeds=(101,), phase_length=4, summary_window=1
    )
    with pytest.raises(ValueError, match="configuration"):
        load_matched_discrete_control_checkpoint(
            advanced,
            checkpoint,
            config=wrong_config,
        )
