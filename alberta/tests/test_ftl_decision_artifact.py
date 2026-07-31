"""Fail-closed artifact tests for the frozen FTL decision-fidelity probe."""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

import alberta_framework.evaluation.ftl_decision_artifact as ftl_artifact_module
from alberta_framework.evaluation.ftl_decision_artifact import (
    FROZEN_BOOTSTRAP_RESAMPLES,
    PROTOCOL_VERSION,
    SCHEMA_VERSION,
    artifact_json,
    build_ftl_decision_artifact,
    load_ftl_decision_artifact,
    scientific_payload_sha256,
    validate_ftl_decision_artifact,
)
from alberta_framework.evaluation.ftl_decision_cli import (
    main as ftl_decision_cli_main,
)
from alberta_framework.evaluation.ftl_decision_fidelity import (
    CONDITION_NAMES,
    DEVELOPMENT_SEEDS,
    EVIDENCE_SEEDS,
    DecisionFidelityConfig,
    DecisionFidelityReport,
    DecisionMetrics,
    SeedDecisionResult,
)

pytestmark = pytest.mark.scientific


def _as_dict(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    return value


def _as_list(value: object) -> list[object]:
    assert isinstance(value, list)
    return value


def _scientific(artifact: dict[str, object]) -> dict[str, object]:
    return _as_dict(artifact["scientific_payload"])


def _rehash(artifact: dict[str, object]) -> None:
    digest = _as_dict(artifact["scientific_digest"])
    digest["sha256"] = scientific_payload_sha256(_scientific(artifact))


def _metrics(
    condition: str,
    normalized_regret: float,
    domain_a: float,
    domain_b: float,
    oracle_picks: int,
    reward_mae: float,
    return_mae: float,
    normalized_return_mae: float,
) -> DecisionMetrics:
    assert normalized_regret == pytest.approx(0.5 * (domain_a + domain_b))
    return DecisionMetrics(
        condition=condition,
        normalized_regret=normalized_regret,
        domain_a_normalized_regret=domain_a,
        domain_b_normalized_regret=domain_b,
        oracle_pick_rate=oracle_picks / 24.0,
        reward_mae=reward_mae,
        return_mae=return_mae,
        normalized_return_mae=normalized_return_mae,
    )


def _synthetic_report(
    *,
    sparse_after_b_regret: float = 0.001,
) -> DecisionFidelityReport:
    """Create cheap primitive evidence satisfying the frozen gate by default.

    The artifact builder deliberately ignores cached aggregates and rebuilds
    every derived statistic from these seed records.
    """

    sparse_after_b_domain = sparse_after_b_regret
    per_seed = (
        _metrics("sparse_untrained", 0.30, 0.30, 0.30, 12, 1.0, 2.0, 0.50),
        _metrics(
            "sparse_after_a1",
            0.10025,
            0.0005,
            0.20,
            18,
            0.10,
            0.50,
            0.05,
        ),
        _metrics(
            "sparse_after_b",
            sparse_after_b_regret,
            sparse_after_b_domain,
            sparse_after_b_domain,
            22,
            0.05,
            0.25,
            0.02,
        ),
        _metrics(
            "sparse_after_a2",
            0.00075,
            0.0005,
            0.001,
            23,
            0.045,
            0.22,
            0.018,
        ),
        _metrics(
            "linear_after_a1",
            0.0775,
            0.005,
            0.15,
            18,
            0.60,
            2.80,
            0.25,
        ),
        _metrics(
            "linear_after_b",
            0.04,
            0.04,
            0.04,
            19,
            0.50,
            2.50,
            0.20,
        ),
        _metrics(
            "linear_after_a2",
            0.03,
            0.03,
            0.03,
            20,
            0.45,
            2.20,
            0.18,
        ),
    )
    assert tuple(metric.condition for metric in per_seed) == CONDITION_NAMES
    return DecisionFidelityReport(
        config=DecisionFidelityConfig(),
        seeds=EVIDENCE_SEEDS,
        seed_results=tuple(
            SeedDecisionResult(seed=seed, metrics=per_seed) for seed in EVIDENCE_SEEDS
        ),
        aggregates=(),
        comparisons=(),
    )


@pytest.fixture(scope="module")
def accepted_report() -> DecisionFidelityReport:
    return _synthetic_report()


@pytest.fixture(scope="module")
def accepted_artifact(
    accepted_report: DecisionFidelityReport,
) -> dict[str, object]:
    return build_ftl_decision_artifact(
        accepted_report,
        evaluation_wall_seconds=7.5,
        generated_at=datetime(2026, 7, 30, 12, 0, tzinfo=UTC),
    )


def test_artifact_is_valid_accepted_versioned_and_narrow(
    accepted_artifact: dict[str, object],
) -> None:
    validation = validate_ftl_decision_artifact(accepted_artifact)

    assert validation.valid
    assert validation.accepted
    assert validation.errors == ()
    assert accepted_artifact["schema_version"] == SCHEMA_VERSION
    scientific = _scientific(accepted_artifact)
    protocol = _as_dict(scientific["protocol"])
    assert protocol["protocol_version"] == PROTOCOL_VERSION
    seed_roles = _as_dict(protocol["seed_roles"])
    assert seed_roles["development_and_threshold_calibration"] == list(DEVELOPMENT_SEEDS)
    assert seed_roles["promoted_held_out_evidence"] == list(EVIDENCE_SEEDS)
    assert set(DEVELOPMENT_SEEDS).isdisjoint(EVIDENCE_SEEDS)
    excluded = _as_list(protocol["excluded_claims"])
    assert "closed-loop acting or model-predictive control" in excluded
    assert "reward-model learning" in excluded
    assert "indefinite-lifetime retention or scaling theorem" in excluded
    assert "completion of the Alberta Plan" in excluded
    assert "deterministic, fully observed, one-dimensional" in str(protocol["environment"])
    assert "known reward" in str(protocol["decision_probe"])
    assert "hand-designed horizon-six open-loop" in str(protocol["decision_probe"])
    assert "not an official" in str(protocol["external_protocol_relationship"])
    assert "not compute-" in str(protocol["baseline_scope"])
    provenance = _as_dict(scientific["source_provenance"])
    assert set(provenance) == {
        "repository_subtree",
        "git_head",
        "source_sha256",
        "interpretation",
    }
    assert len(str(provenance["git_head"])) == 40
    assert set(_as_dict(provenance["source_sha256"])) == {
        "alberta_framework/core/ftl_world_model.py",
        "alberta_framework/evaluation/ftl_decision_fidelity.py",
        "alberta_framework/evaluation/ftl_decision_artifact.py",
        "alberta_framework/evaluation/ftl_decision_cli.py",
    }
    assert "not signatures" in str(provenance["interpretation"])
    assert "authenticity" in str(provenance["interpretation"])
    operational = _as_dict(accepted_artifact["operational_metadata"])
    assert set(_as_dict(operational["runtime"])) == {
        "python",
        "platform",
        "packages",
        "jax",
    }
    assert set(_as_dict(operational["git_worktree"])) == {"head", "dirty"}


def test_artifact_remains_valid_after_checkout_head_advances_without_source_changes(
    accepted_artifact: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Committing the artifact must not make its pinned source evidence stale."""

    generation_head = str(_as_dict(_scientific(accepted_artifact)["source_provenance"])["git_head"])
    later_head = "1" * 40 if generation_head != "1" * 40 else "2" * 40
    monkeypatch.setattr(ftl_artifact_module, "_git_head", lambda: later_head)

    validation = validate_ftl_decision_artifact(accepted_artifact)

    assert validation.valid
    assert validation.accepted


def test_artifact_preserves_exact_frozen_configuration_and_five_thousand_bootstraps(
    accepted_artifact: dict[str, object],
) -> None:
    scientific = _scientific(accepted_artifact)
    configuration = _as_dict(scientific["configuration"])
    bootstrap = _as_dict(scientific["bootstrap"])

    assert configuration["horizon"] == 6
    assert configuration["probes_per_domain"] == 12
    assert configuration["bootstrap_resamples"] == 5_000
    assert bootstrap["resamples"] == FROZEN_BOOTSTRAP_RESAMPLES == 5_000
    assert "no post-held-out 10,000-resample replacement" in str(bootstrap["freeze_note"])
    summaries = _as_list(scientific["seed_summaries"])
    assert [_as_dict(summary)["seed"] for summary in summaries] == list(EVIDENCE_SEEDS)


def test_aggregate_and_paired_effects_are_reconstructed_from_primitive_seeds(
    accepted_artifact: dict[str, object],
) -> None:
    scientific = _scientific(accepted_artifact)
    aggregate = _as_dict(scientific["aggregate"])
    conditions = _as_dict(aggregate["conditions"])
    learned = _as_dict(_as_dict(conditions["sparse_after_b"])["normalized_regret"])
    assert set(learned) == {
        "estimate",
        "lower",
        "upper",
        "confidence_level",
        "resamples",
        "sample_size",
        "method",
        "pairing_unit",
    }
    assert learned["estimate"] == pytest.approx(0.001)
    assert learned["lower"] == pytest.approx(0.001)
    assert learned["upper"] == pytest.approx(0.001)
    assert learned["confidence_level"] == 0.95
    assert learned["resamples"] == 5_000
    assert learned["sample_size"] == 30
    assert learned["method"] == "percentile-bootstrap"
    assert learned["pairing_unit"] == "seed"

    comparisons = _as_dict(scientific["paired_comparisons"])
    versus_untrained = _as_dict(
        _as_dict(comparisons["sparse_after_b_vs_untrained_regret_reduction"])["interval"]
    )
    assert versus_untrained["estimate"] == pytest.approx(0.299)
    assert versus_untrained["lower"] == pytest.approx(0.299)
    assert versus_untrained["upper"] == pytest.approx(0.299)
    assert versus_untrained["method"] == "paired-percentile-bootstrap"
    assert versus_untrained["resamples"] == 5_000
    assert versus_untrained["sample_size"] == 30

    versus_linear = _as_dict(
        _as_dict(comparisons["sparse_after_b_vs_linear_regret_reduction"])["interval"]
    )
    assert versus_linear["estimate"] == pytest.approx(0.039)
    interference = _as_dict(
        _as_dict(comparisons["sparse_domain_a_regret_change_after_b"])["interval"]
    )
    assert interference["upper"] == pytest.approx(0.0005)
    recovery = _as_dict(_as_dict(comparisons["sparse_domain_a_recovery_after_a2"])["interval"])
    assert recovery["estimate"] == pytest.approx(0.0005)


def test_acceptance_is_bound_to_reconstructed_values_and_frozen_thresholds(
    accepted_artifact: dict[str, object],
) -> None:
    scientific = _scientific(accepted_artifact)
    acceptance = _as_dict(scientific["acceptance"])
    checks = {_as_dict(check)["name"]: _as_dict(check) for check in _as_list(acceptance["checks"])}

    assert acceptance["passed"] is True
    assert checks["untrained_normalized_regret"]["comparator"] == ">"
    assert checks["untrained_normalized_regret"]["threshold"] == 0.25
    assert checks["sparse_after_b_normalized_regret"]["comparator"] == "<"
    assert checks["sparse_after_b_normalized_regret"]["threshold"] == 0.01
    assert checks["sparse_after_b_reward_mae_ratio_to_linear"]["actual"] == pytest.approx(0.10)
    assert checks["sparse_after_b_normalized_return_mae_ratio_to_linear"][
        "actual"
    ] == pytest.approx(0.10)
    assert checks["sparse_after_b_vs_linear_regret_reduction"]["threshold"] == 0.03


def test_digest_is_deterministic_and_excludes_host_timing_and_timestamp(
    accepted_report: DecisionFidelityReport,
    accepted_artifact: dict[str, object],
) -> None:
    later = build_ftl_decision_artifact(
        accepted_report,
        evaluation_wall_seconds=999.0,
        generated_at=datetime(2026, 7, 31, 12, 0, tzinfo=UTC),
    )

    assert _scientific(later) == _scientific(accepted_artifact)
    assert later["scientific_digest"] == accepted_artifact["scientific_digest"]
    assert later["operational_metadata"] != accepted_artifact["operational_metadata"]
    assert validate_ftl_decision_artifact(later).accepted
    digest = _as_dict(accepted_artifact["scientific_digest"])
    assert digest["sha256"] == scientific_payload_sha256(_scientific(accepted_artifact))


def test_strict_json_round_trip_rejects_nonstandard_numbers_and_duplicate_keys(
    accepted_artifact: dict[str, object],
    tmp_path: Path,
) -> None:
    path = tmp_path / "evidence.json"
    path.write_text(artifact_json(accepted_artifact), encoding="utf-8")
    loaded = load_ftl_decision_artifact(path)
    assert loaded == accepted_artifact
    assert validate_ftl_decision_artifact(loaded).accepted

    nonstandard = tmp_path / "nan.json"
    nonstandard.write_text('{"value": NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="non-standard JSON"):
        load_ftl_decision_artifact(nonstandard)

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"value": 1, "value": 2}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON object key"):
        load_ftl_decision_artifact(duplicate)


def test_unrehashed_scientific_tampering_fails_digest(
    accepted_artifact: dict[str, object],
) -> None:
    tampered = copy.deepcopy(accepted_artifact)
    aggregate = _as_dict(_scientific(tampered)["aggregate"])
    _as_dict(aggregate["conditions"])["unexpected"] = {}

    validation = validate_ftl_decision_artifact(tampered)
    assert not validation.valid
    assert not validation.accepted
    assert "scientific_digest.sha256 does not match scientific payload" in validation.errors


def test_rehashed_primitive_tampering_fails_aggregate_reconstruction(
    accepted_artifact: dict[str, object],
) -> None:
    tampered = copy.deepcopy(accepted_artifact)
    first = _as_dict(_as_list(_scientific(tampered)["seed_summaries"])[0])
    conditions = _as_dict(first["conditions"])
    sparse = _as_dict(conditions["sparse_after_b"])
    sparse["normalized_regret"] = 0.02
    sparse["domain_a_normalized_regret"] = 0.02
    sparse["domain_b_normalized_regret"] = 0.02
    _rehash(tampered)

    validation = validate_ftl_decision_artifact(tampered)
    assert not validation.valid
    assert not validation.accepted
    assert any(
        "scientific_payload.aggregate" in error and "reconstructed evidence" in error
        for error in validation.errors
    )


def test_rehashed_aggregate_and_acceptance_tampering_fail_reconstruction(
    accepted_artifact: dict[str, object],
) -> None:
    tampered = copy.deepcopy(accepted_artifact)
    scientific = _scientific(tampered)
    aggregate = _as_dict(scientific["aggregate"])
    conditions = _as_dict(aggregate["conditions"])
    interval = _as_dict(_as_dict(conditions["sparse_after_b"])["normalized_regret"])
    interval["upper"] = 0.0
    acceptance = _as_dict(scientific["acceptance"])
    acceptance["passed"] = False
    _rehash(tampered)

    validation = validate_ftl_decision_artifact(tampered)
    assert not validation.valid
    assert not validation.accepted
    assert any("scientific_payload.aggregate" in error for error in validation.errors)
    assert any("scientific_payload.acceptance" in error for error in validation.errors)


def test_rehashed_schema_threshold_and_provenance_tampering_fail_closed(
    accepted_artifact: dict[str, object],
) -> None:
    tampered = copy.deepcopy(accepted_artifact)
    scientific = _scientific(tampered)
    _as_dict(scientific["protocol"])["unexpected"] = True
    _as_dict(scientific["thresholds"])["minimum_sparse_vs_linear_regret_reduction_lower"] = 0.0
    _as_dict(scientific["source_provenance"])["git_head"] = "0" * 40
    _rehash(tampered)

    validation = validate_ftl_decision_artifact(tampered)
    assert not validation.valid
    assert not validation.accepted
    assert any("protocol keys do not match" in error for error in validation.errors)
    assert any("thresholds" in error for error in validation.errors)
    assert any("source_provenance" in error for error in validation.errors)


def test_rehashed_duplicate_missing_unknown_and_impossible_seed_metrics_fail_closed(
    accepted_artifact: dict[str, object],
) -> None:
    duplicate_seed = copy.deepcopy(accepted_artifact)
    summaries = _as_list(_scientific(duplicate_seed)["seed_summaries"])
    _as_dict(summaries[1])["seed"] = _as_dict(summaries[0])["seed"]
    _rehash(duplicate_seed)
    duplicate_validation = validate_ftl_decision_artifact(duplicate_seed)
    assert not duplicate_validation.valid
    assert any("30-59 exactly" in error for error in duplicate_validation.errors)

    unknown_metric = copy.deepcopy(accepted_artifact)
    first = _as_dict(_as_list(_scientific(unknown_metric)["seed_summaries"])[0])
    conditions = _as_dict(first["conditions"])
    _as_dict(conditions["sparse_after_b"])["invented_score"] = 1.0
    _rehash(unknown_metric)
    unknown_validation = validate_ftl_decision_artifact(unknown_metric)
    assert not unknown_validation.valid
    assert any("metric schema" in error for error in unknown_validation.errors)

    missing_metric = copy.deepcopy(accepted_artifact)
    first = _as_dict(_as_list(_scientific(missing_metric)["seed_summaries"])[0])
    conditions = _as_dict(first["conditions"])
    del _as_dict(conditions["sparse_after_b"])["return_mae"]
    _rehash(missing_metric)
    missing_validation = validate_ftl_decision_artifact(missing_metric)
    assert not missing_validation.valid
    assert any("metric schema" in error for error in missing_validation.errors)
    assert any("return_mae must be finite" in error for error in missing_validation.errors)

    impossible = copy.deepcopy(accepted_artifact)
    first = _as_dict(_as_list(_scientific(impossible)["seed_summaries"])[0])
    conditions = _as_dict(first["conditions"])
    _as_dict(conditions["sparse_after_b"])["oracle_pick_rate"] = 0.90
    _rehash(impossible)
    impossible_validation = validate_ftl_decision_artifact(impossible)
    assert not impossible_validation.valid
    assert any("multiple of 1/24" in error for error in impossible_validation.errors)

    unknown_top_level = copy.deepcopy(accepted_artifact)
    unknown_top_level["unexpected"] = True
    top_level_validation = validate_ftl_decision_artifact(unknown_top_level)
    assert not top_level_validation.valid
    assert any("top-level keys" in error for error in top_level_validation.errors)


def test_wrong_configuration_or_seed_schedule_cannot_be_built(
    accepted_report: DecisionFidelityReport,
) -> None:
    changed_config = replace(
        accepted_report,
        config=replace(accepted_report.config, bootstrap_resamples=10_000),
    )
    with pytest.raises(ValueError, match="canonical frozen configuration"):
        build_ftl_decision_artifact(
            changed_config,
            evaluation_wall_seconds=0.0,
        )

    development = replace(
        accepted_report,
        seeds=DEVELOPMENT_SEEDS,
        seed_results=tuple(
            replace(seed_result, seed=seed)
            for seed_result, seed in zip(
                accepted_report.seed_results,
                DEVELOPMENT_SEEDS,
                strict=True,
            )
        ),
    )
    with pytest.raises(ValueError, match="held-out seeds 30-59"):
        build_ftl_decision_artifact(
            development,
            evaluation_wall_seconds=0.0,
        )


def test_valid_scientific_rejection_remains_parseable_and_is_not_promoted(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rejected_report = _synthetic_report(sparse_after_b_regret=0.20)
    artifact = build_ftl_decision_artifact(
        rejected_report,
        evaluation_wall_seconds=1.0,
        generated_at=datetime(2026, 7, 30, tzinfo=UTC),
    )
    validation = validate_ftl_decision_artifact(artifact)
    assert validation.valid
    assert not validation.accepted
    checks = _as_list(_as_dict(_scientific(artifact)["acceptance"])["checks"])
    assert any(_as_dict(check)["passed"] is False for check in checks)

    path = tmp_path / "rejected.json"
    status = ftl_decision_cli_main(
        ["--output", str(path)],
        report=rejected_report,
        evaluation_wall_seconds=1.0,
        generated_at=datetime(2026, 7, 30, tzinfo=UTC),
    )
    emitted = json.loads(capsys.readouterr().out)
    assert status == 1
    assert emitted["valid"] is True
    assert emitted["accepted"] is False
    assert path.exists()


def test_operational_metadata_is_outside_digest_but_still_validated(
    accepted_artifact: dict[str, object],
) -> None:
    changed = copy.deepcopy(accepted_artifact)
    operational = _as_dict(changed["operational_metadata"])
    operational["evaluation_wall_seconds"] = 123.0
    operational["generated_at_utc"] = (
        datetime(2026, 7, 30, tzinfo=UTC) + timedelta(days=1)
    ).isoformat()
    assert changed["scientific_digest"] == accepted_artifact["scientific_digest"]
    assert validate_ftl_decision_artifact(changed).accepted

    operational["evaluation_wall_seconds"] = -1.0
    validation = validate_ftl_decision_artifact(changed)
    assert not validation.valid
    assert not validation.accepted


def test_cli_writes_verifies_and_returns_two_for_invalid_artifact(
    accepted_report: DecisionFidelityReport,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "accepted.json"
    status = ftl_decision_cli_main(
        ["--output", str(path)],
        report=accepted_report,
        evaluation_wall_seconds=7.5,
        generated_at=datetime(2026, 7, 30, tzinfo=UTC),
    )
    emitted = json.loads(capsys.readouterr().out)
    assert status == 0
    assert emitted["valid"] is True
    assert emitted["accepted"] is True
    assert emitted["seed_count"] == 30

    verify_status = ftl_decision_cli_main(["--verify", str(path)])
    verified = json.loads(capsys.readouterr().out)
    assert verify_status == 0
    assert verified["valid"] is True
    assert verified["accepted"] is True

    tampered = load_ftl_decision_artifact(path)
    _as_dict(_scientific(tampered)["aggregate"])["seed_count"] = 1
    path.write_text(artifact_json(tampered), encoding="utf-8")
    tampered_status = ftl_decision_cli_main(["--verify", str(path)])
    rejected = json.loads(capsys.readouterr().out)
    assert tampered_status == 2
    assert rejected["valid"] is False
    assert rejected["accepted"] is False


def test_cli_exposes_no_seed_bootstrap_or_threshold_tuning_options(
    accepted_report: DecisionFidelityReport,
) -> None:
    with pytest.raises(SystemExit):
        ftl_decision_cli_main(["--seed-start", "0"], report=accepted_report)
    with pytest.raises(SystemExit):
        ftl_decision_cli_main(
            ["--bootstrap-resamples", "10000"],
            report=accepted_report,
        )
    with pytest.raises(SystemExit):
        ftl_decision_cli_main(
            ["--maximum-sparse-after-b-normalized-regret", "1"],
            report=accepted_report,
        )
