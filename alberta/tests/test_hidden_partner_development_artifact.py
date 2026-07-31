"""Fail-closed tests for the nonpromoting hidden-partner artifact."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from alberta_framework.evaluation.hidden_partner_development import (
    HIDDEN_PARTNER_CONDITIONS,
    TUNING_SEED_NAMESPACE,
    HiddenPartnerDevelopmentProtocol,
    HiddenPartnerFeatureSummary,
    HiddenPartnerRunSummary,
    HiddenPartnerSegmentSummary,
    derive_hidden_partner_seed_pairs,
    hidden_partner_run_summary_from_dict,
)
from alberta_framework.evaluation.hidden_partner_development_artifact import (
    build_hidden_partner_development_artifact,
    hidden_partner_development_artifact_json,
    load_hidden_partner_development_artifact,
    validate_hidden_partner_development_artifact,
)
from alberta_framework.evaluation.hidden_partner_development_cli import main
from alberta_framework.streams.hidden_partner_mapping import (
    DEFAULT_BASE_SEGMENT_LENGTHS,
    DEFAULT_REGIME_SCHEDULE,
    REGIME_NAMES,
)

pytestmark = pytest.mark.integration


def _canonical_sha256(value: object) -> str:
    serialized = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(serialized).hexdigest()


def _summary(
    condition: str,
    seed_index: int,
    *,
    reward: float,
    d_at_life_end: bool = False,
) -> HiddenPartnerRunSummary:
    prior_by_regime: dict[int, int] = {}
    segments: list[HiddenPartnerSegmentSummary] = []
    for index, (regime, length) in enumerate(
        zip(DEFAULT_REGIME_SCHEDULE, DEFAULT_BASE_SEGMENT_LENGTHS, strict=True)
    ):
        prior = prior_by_regime.get(regime)
        segments.append(
            HiddenPartnerSegmentSummary(
                segment_index=index,
                regime_id=regime,
                regime_name=REGIME_NAMES[regime],
                length=length,
                mean_reward=reward,
                early_reward=reward,
                late_reward=reward,
                mean_behavior_nll=0.1,
                late_behavior_nll=0.1,
                mean_behavior_brier=0.1,
                intended_prediction_accuracy=0.9,
                late_intended_prediction_accuracy=0.9,
                mean_realized_regret=0.1,
                mean_expected_greedy_regret=0.1,
                recovery_steps=128,
                prior_same_regime_segment=prior,
                recurrent_early_to_prior_late_ratio=(None if prior is None else 1.0),
                recurrence_retained=None if prior is None else True,
            )
        )
        prior_by_regime[regime] = index

    lifecycle_disabled = condition in {"lifecycle_frozen", "random_curation"}
    cycle_steps = sum(DEFAULT_BASE_SEGMENT_LENGTHS)
    epsilon = next(
        item.config.epsilon for item in HIDDEN_PARTNER_CONDITIONS if item.name == condition
    )
    perfect_policy_reward = (1.0 - epsilon) * 0.95 + epsilon * 0.5
    return HiddenPartnerRunSummary(
        condition=condition,  # type: ignore[arg-type]
        seed_pair=derive_hidden_partner_seed_pairs(
            TUNING_SEED_NAMESPACE,
            seed_index + 1,
        )[seed_index],
        cycle_steps=cycle_steps,
        segment_lengths=DEFAULT_BASE_SEGMENT_LENGTHS,
        mean_reward=reward,
        normalized_control_score=(reward - 0.5) / (perfect_policy_reward - 0.5),
        mean_behavior_nll=0.1,
        mean_behavior_brier=0.1,
        behavior_actual_accuracy=0.85,
        behavior_intended_accuracy=0.9,
        planner_intended_accuracy=0.9,
        executed_intended_accuracy=0.9,
        mean_realized_counterfactual_regret=0.1,
        mean_expected_greedy_regret=0.1,
        model_intervention_rate=0.1,
        helpful_model_intervention_rate=0.05,
        harmful_model_intervention_rate=0.02,
        mean_world_reward_absolute_error=0.1,
        mean_world_outcome_squared_error=0.1,
        descriptor_transaction_count=20,
        counter_contract_valid=True,
        causal_contract_valid=True,
        all_finite=True,
        initial_state_nbytes=4_096,
        final_state_nbytes=4_096,
        resource_shape_matched=True,
        compilation_wall_seconds=0.0,
        execution_wall_seconds=1.0,
        mean_execution_microseconds_per_step=1e6 / cycle_steps,
        segments=tuple(segments),
        features=HiddenPartnerFeatureSummary(
            c_first_active_step=None if lifecycle_disabled else 8_000,
            d_first_active_step=None if lifecycle_disabled else 5_000,
            c_active_evictions=0,
            d_active_evictions=0,
            c_active_late_first_c=not lifecycle_disabled,
            c_active_at_recurrent_c_entry=not lifecycle_disabled,
            c_survived_first_to_recurrent_c=not lifecycle_disabled,
            d_active_at_end_of_d=not lifecycle_disabled,
            d_active_at_life_end=d_at_life_end,
            c_candidate_fraction=1.0,
            d_candidate_fraction=1.0,
        ),
    )


def _summaries(*, d_at_life_end: bool = False) -> tuple[HiddenPartnerRunSummary, ...]:
    rewards = {
        "full": 0.90,
        "lifecycle_frozen": 0.70,
        "uniform_partner": 0.80,
        "random_curation": 0.65,
    }
    return tuple(
        _summary(
            condition.name,
            seed_index,
            reward=rewards.get(condition.name, 0.85),
            d_at_life_end=(d_at_life_end and condition.name == "full"),
        )
        for seed_index in range(2)
        for condition in HIDDEN_PARTNER_CONDITIONS
    )


def _operational_metadata() -> dict[str, object]:
    return {
        "argv": ["--role", "tuning-replay"],
        "generated_at_utc": "2026-07-30T12:00:00+00:00",
        "jax_backend": "cpu",
        "jax_device_count": 1,
        "jax_version": "test",
        "platform": "test",
        "python_version": "3.12",
        "wall_seconds": 1.0,
    }


def test_run_summary_round_trip_rejects_nonfinite_and_inconsistent_metrics() -> None:
    summary = _summary("full", 0, reward=0.9)
    assert hidden_partner_run_summary_from_dict(summary.to_dict()) == summary

    changed_mean = copy.deepcopy(summary.to_dict())
    changed_mean["mean_reward"] = 0.8
    with pytest.raises(ValueError, match="does not reconstruct"):
        hidden_partner_run_summary_from_dict(changed_mean)

    nonfinite = copy.deepcopy(summary.to_dict())
    nonfinite["execution_wall_seconds"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        hidden_partner_run_summary_from_dict(nonfinite)

    promoted = copy.deepcopy(summary.to_dict())
    promoted["scientific_promotion_allowed"] = True
    with pytest.raises(ValueError, match="forbid"):
        hidden_partner_run_summary_from_dict(promoted)


def test_artifact_round_trip_reconstructs_aggregate_and_separates_check_failure(
    tmp_path: Path,
) -> None:
    protocol = HiddenPartnerDevelopmentProtocol()
    artifact = build_hidden_partner_development_artifact(
        protocol,
        "tuning-replay",
        _summaries(),
        operational_metadata=_operational_metadata(),
    )
    validation = validate_hidden_partner_development_artifact(artifact)
    assert validation.valid
    assert validation.development_checks_passed

    path = tmp_path / "artifact.json"
    path.write_text(hidden_partner_development_artifact_json(artifact), encoding="utf-8")
    loaded = load_hidden_partner_development_artifact(path)
    assert hidden_partner_development_artifact_json(loaded) == (
        hidden_partner_development_artifact_json(artifact)
    )
    assert validate_hidden_partner_development_artifact(loaded).valid

    failed_checks = build_hidden_partner_development_artifact(
        protocol,
        "tuning-replay",
        _summaries(d_at_life_end=True),
        operational_metadata=_operational_metadata(),
    )
    failed_validation = validate_hidden_partner_development_artifact(failed_checks)
    assert failed_validation.valid
    assert not failed_validation.development_checks_passed


def test_validator_rejects_digest_source_aggregate_and_run_drift() -> None:
    artifact = build_hidden_partner_development_artifact(
        HiddenPartnerDevelopmentProtocol(),
        "tuning-replay",
        _summaries(),
        operational_metadata=_operational_metadata(),
    )

    changed_run = copy.deepcopy(artifact)
    changed_run["scientific_payload"]["runs"][0]["mean_reward"] = 0.1
    validation = validate_hidden_partner_development_artifact(changed_run)
    assert not validation.valid
    assert any("digest mismatch" in error for error in validation.errors)
    assert any("runs[0]" in error for error in validation.errors)

    changed_aggregate = copy.deepcopy(artifact)
    changed_aggregate["scientific_payload"]["aggregate"]["conditions"]["full"]["mean_reward"] = 0.0
    changed_aggregate["scientific_digest"]["sha256"] = _canonical_sha256(
        changed_aggregate["scientific_payload"]
    )
    validation = validate_hidden_partner_development_artifact(changed_aggregate)
    assert not validation.valid
    assert any("does not exactly reconstruct" in error for error in validation.errors)

    changed_source = copy.deepcopy(artifact)
    first_source = next(iter(changed_source["scientific_payload"]["source_sha256"]))
    changed_source["scientific_payload"]["source_sha256"][first_source] = "0" * 64
    changed_source["scientific_digest"]["sha256"] = _canonical_sha256(
        changed_source["scientific_payload"]
    )
    validation = validate_hidden_partner_development_artifact(changed_source)
    assert not validation.valid
    assert any("source hashes" in error for error in validation.errors)


def test_cli_injected_run_writes_once_and_verifies(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "development.json"
    argv = [
        "--role",
        "tuning-replay",
        "--count",
        "2",
        "--output",
        str(output),
    ]
    assert main(argv, summaries=_summaries(), wall_seconds=1.0) == 0
    assert output.exists()
    first_output = json.loads(capsys.readouterr().out)
    assert first_output["valid"] is True
    assert first_output["development_checks_passed"] is True

    assert main(["--verify", str(output)]) == 0
    verification_output = json.loads(capsys.readouterr().out)
    assert verification_output["valid"] is True

    assert main(argv, summaries=_summaries(), wall_seconds=1.0) == 2
    overwrite_output = json.loads(capsys.readouterr().out)
    assert "refusing to overwrite" in overwrite_output["errors"][0]
