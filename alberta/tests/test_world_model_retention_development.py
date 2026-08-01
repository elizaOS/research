from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from alberta_framework.benchmarks.world_model_retention_development import (
    ASSESSMENT_STATUS,
    DEVELOPMENT_ONLY,
    LEARNER_ORDER,
    PHASE_LENGTH,
    REGIME_SCHEDULE,
    SCIENTIFIC_PROMOTION_ALLOWED,
    WORLD_MODEL_RETENTION_DEVELOPMENT_SCHEMA,
    WorldModelRetentionDevelopmentConfig,
    atomic_save_world_model_retention_development_report,
    run_world_model_retention_development,
    validate_world_model_retention_development_report,
    world_model_retention_development_report_json,
)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _refresh_report_digest(report: dict[str, object]) -> None:
    report.pop("report_digest", None)
    report["report_digest"] = {
        "algorithm": "sha256",
        "scope": "$ excluding $.report_digest",
        "sha256": _canonical_sha256(report),
    }


@pytest.fixture(scope="module")
def report() -> dict[str, object]:
    return run_world_model_retention_development()


def test_protocol_is_fixed_development_only_and_has_no_assessment_gate() -> None:
    config = WorldModelRetentionDevelopmentConfig()
    payload = config.to_config()
    assert payload["schema_version"] == WORLD_MODEL_RETENTION_DEVELOPMENT_SCHEMA
    assert DEVELOPMENT_ONLY is True
    assert SCIENTIFIC_PROMOTION_ALLOWED is False
    assert ASSESSMENT_STATUS == "not_assessed"
    assert payload["thresholds"] is None
    assert payload["regime_schedule"] == ["A", "B", "A"]
    assert payload["total_steps"] == PHASE_LENGTH * len(REGIME_SCHEDULE)
    assert WorldModelRetentionDevelopmentConfig.from_config(payload) == config

    changed = copy.deepcopy(payload)
    changed["phase_length"] = PHASE_LENGTH + 1
    with pytest.raises(ValueError, match="noncanonical"):
        WorldModelRetentionDevelopmentConfig.from_config(changed)


def test_report_self_validates_and_is_deterministic(report: dict[str, object]) -> None:
    validation = validate_world_model_retention_development_report(report)
    assert validation.valid, validation.errors
    assert run_world_model_retention_development() == report
    assert report["assessment_status"] == "not_assessed"
    assert report["scientific_promotion_allowed"] is False
    promotion = cast(Mapping[str, object], report["promotion"])
    assert promotion == {
        "allowed": False,
        "evidence_level": None,
        "thresholds": None,
        "decision": None,
    }


def test_all_learners_receive_one_common_raw_stream_without_regime_labels(
    report: dict[str, object],
) -> None:
    protocol = cast(Mapping[str, object], report["protocol"])
    assert protocol["learner_input_fields"] == [
        "observation",
        "action",
        "next_observation",
        "reward",
        "continuation",
    ]
    assert "regime_id" in cast(list[str], protocol["evaluator_only_fields"])
    assert "noisy_tv" in cast(list[str], protocol["evaluator_only_fields"])
    runs = cast(list[Mapping[str, object]], report["runs"])
    streams = cast(list[Mapping[str, object]], report["matched_streams"])
    for stream_record in streams:
        stream = cast(Mapping[str, object], stream_record["stream"])
        assert stream["labels_visible_to_learners"] is False
        assert stream["regime_id"] == ["A"] * PHASE_LENGTH + ["B"] * PHASE_LENGTH + [
            "A"
        ] * PHASE_LENGTH
        seed_runs = [run for run in runs if run["seed"] == stream["seed"]]
        assert [run["learner"] for run in seed_runs] == list(LEARNER_ORDER)
        assert {run["common_stream_sha256"] for run in seed_runs} == {
            stream_record["stream_sha256"]
        }
        for run in seed_runs:
            trace = cast(Mapping[str, object], run["trace"])
            assert "regime_id" not in trace
            assert "phase_index" not in trace
            assert "noisy_tv" not in trace
            assert run["common_grounded_targets"] is True
            assert run["prediction_order"] == "predict_before_update"


def test_raw_channel_errors_and_aggregate_loss_reconstruct_from_common_targets(
    report: dict[str, object],
) -> None:
    streams = {
        cast(Mapping[str, object], item["stream"])["seed"]: cast(
            Mapping[str, object], item["stream"]
        )
        for item in cast(list[Mapping[str, object]], report["matched_streams"])
    }
    for run in cast(list[Mapping[str, object]], report["runs"]):
        stream = streams[run["seed"]]
        trace = cast(Mapping[str, object], run["trace"])
        predicted_next = np.asarray(trace["predicted_next_observation"], dtype=np.float32)
        target_next = np.asarray(stream["next_observation"], dtype=np.float32)
        next_error = predicted_next - target_next
        reward_error = np.asarray(trace["predicted_reward"], dtype=np.float32) - np.asarray(
            stream["reward"], dtype=np.float32
        )
        continuation_error = np.asarray(
            trace["predicted_continuation"], dtype=np.float32
        ) - np.asarray(stream["continuation"], dtype=np.float32)
        expected_loss = np.mean(
            np.concatenate(
                (
                    np.square(next_error),
                    np.square(reward_error)[:, None],
                    np.square(continuation_error)[:, None],
                ),
                axis=1,
            ),
            axis=1,
        )
        np.testing.assert_array_equal(
            np.asarray(trace["next_observation_error"], dtype=np.float32), next_error
        )
        np.testing.assert_array_equal(
            np.asarray(trace["reward_error"], dtype=np.float32), reward_error
        )
        np.testing.assert_array_equal(
            np.asarray(trace["continuation_error"], dtype=np.float32), continuation_error
        )
        np.testing.assert_allclose(
            np.asarray(trace["aggregate_prequential_loss"], dtype=np.float32),
            expected_loss,
            rtol=0.0,
            atol=2.0e-7,
        )

    shallow = next(
        run
        for run in cast(list[Mapping[str, object]], report["runs"])
        if run["learner"] == "shallow_ridge_world_model"
    )
    shallow_trace = cast(Mapping[str, object], shallow["trace"])
    assert shallow_trace["predicted_next_observation"][0] == [0.0, 0.0]  # type: ignore[index]
    assert shallow_trace["predicted_reward"][0] == 0.0  # type: ignore[index]


def test_retention_calibration_noisy_tv_and_unavailable_fields_are_explicit(
    report: dict[str, object],
) -> None:
    for run in cast(list[Mapping[str, object]], report["runs"]):
        metrics = cast(Mapping[str, object], run["metrics"])
        assert set(metrics) == {
            "raw_grounded_error_summary",
            "aggregate_prequential_loss",
            "phase_metrics",
            "post_change_adaptation_auc",
            "recurrence_and_recovery",
            "best_to_final_forgetting",
        }
        assert len(cast(list[object], metrics["post_change_adaptation_auc"])) == 2
        calibration = cast(Mapping[str, object], run["ensemble_disagreement_calibration"])
        noisy = cast(Mapping[str, object], run["noisy_tv_diagnostic"])
        replay = cast(Mapping[str, object], noisy["replay_prioritization_and_composition"])
        if run["learner"] == "shallow_ridge_world_model":
            assert calibration["available"] is False
            assert calibration["bins"] is None
            assert calibration["reason"]
        else:
            assert calibration["available"] is True
            bins = cast(list[Mapping[str, object]], calibration["bins"])
            assert sum(cast(int, item["count"]) for item in bins) == PHASE_LENGTH * 3
            for item in bins:
                if item["count"] == 0:
                    assert item["mean_disagreement"] is None
                    assert item["mean_squared_error"] is None
        if run["learner"] == "model_replay_rehearsal":
            assert replay["available"] is True
            composition = cast(Mapping[str, object], replay["sampled_composition"])
            assert cast(Mapping[str, int], composition["short_term"])["valid"] > 0
            final_memory = cast(Mapping[str, object], replay["final_memory_composition"])
            long_term = cast(Mapping[str, object], final_memory["long_term"])
            assert cast(Mapping[str, int], long_term["counts"])["active_count"] > 0
        else:
            assert replay["available"] is False
            assert replay["sampled_composition"] is None
            assert replay["final_memory_composition"] is None


def test_exact_update_counts_and_nonparity_disclosure(report: dict[str, object]) -> None:
    total_steps = PHASE_LENGTH * len(REGIME_SCHEDULE)
    for run in cast(list[Mapping[str, object]], report["runs"]):
        operations = cast(Mapping[str, object], run["operations"])
        assert operations["real_prediction_events"] == total_steps
        assert operations["real_update_attempts"] == total_steps
        assert operations["real_update_commits"] == total_steps
        if run["learner"] == "shallow_ridge_world_model":
            assert operations["real_member_update_candidates"] == total_steps
            assert operations["replay_quota_positions"] is None
        elif run["learner"] == "plain_world_model_ensemble":
            assert operations["real_member_update_candidates"] == 2 * total_steps
            assert operations["replay_quota_positions"] is None
        else:
            assert operations["real_member_update_candidates"] == 2 * total_steps
            assert operations["replay_quota_positions"] == 2 * total_steps
            assert cast(int, operations["replay_available_positions"]) + cast(
                int, operations["replay_padding_positions"]
            ) == 2 * total_steps
    comparability = cast(Mapping[str, object], report["comparability"])
    assert comparability["common_raw_grounded_targets"] is True
    assert comparability["realized_resource_parity"] is False
    assert comparability["resource_parity_claim"] is None
    differences = cast(list[str], comparability["unavoidable_algorithm_and_state_differences"])
    assert any("eigvalsh" in item for item in differences)
    for comparison in cast(list[Mapping[str, object]], report["comparisons"]):
        assert comparison["plain_and_rehearsal_initial_ensemble_state_matched"] is True
        assert comparison["plain_and_rehearsal_real_bootstrap_masks_matched"] is True
        assert comparison["winner"] is None
        assert comparison["assessment"] == "not_assessed"


def test_validator_rejects_tampering_even_after_attacker_rehashes(
    report: dict[str, object],
) -> None:
    changed = copy.deepcopy(report)
    run = cast(list[dict[str, object]], changed["runs"])[0]
    trace = cast(dict[str, object], run["trace"])
    rewards = cast(list[float], trace["predicted_reward"])
    rewards[3] += 0.125
    run["trace_sha256"] = _canonical_sha256(trace)
    _refresh_report_digest(changed)
    validation = validate_world_model_retention_development_report(changed)
    assert not validation.valid
    assert any("reconstruct" in error for error in validation.errors)

    noncanonical = copy.deepcopy(report)
    cast(dict[str, object], noncanonical["protocol"])["seeds"] = tuple(
        cast(list[int], cast(Mapping[str, object], report["protocol"])["seeds"])
    )
    validation = validate_world_model_retention_development_report(noncanonical)
    assert not validation.valid
    assert any("noncanonical" in error for error in validation.errors)

    nonfinite = copy.deepcopy(report)
    cast(dict[str, object], nonfinite["comparability"])["resource_parity_claim"] = float(
        "nan"
    )
    validation = validate_world_model_retention_development_report(nonfinite)
    assert not validation.valid
    assert any("noncanonical" in error for error in validation.errors)

    foreign: dict[str, object] = {
        "schema_version": WORLD_MODEL_RETENTION_DEVELOPMENT_SCHEMA,
        "report_digest": {
            "algorithm": "sha256",
            "scope": "$ excluding $.report_digest",
            "sha256": "0" * 64,
        },
        "foreign": object(),
    }
    validation = validate_world_model_retention_development_report(foreign)
    assert not validation.valid


def test_atomic_save_validates_before_replace_and_writes_canonical_json(
    report: dict[str, object],
    tmp_path: Path,
) -> None:
    destination = tmp_path / "retention.json"
    destination.write_text("sentinel", encoding="utf-8")
    changed = copy.deepcopy(report)
    cast(dict[str, object], changed["promotion"])["allowed"] = True
    _refresh_report_digest(changed)
    with pytest.raises(ValueError, match="invalid"):
        atomic_save_world_model_retention_development_report(changed, destination)
    assert destination.read_text(encoding="utf-8") == "sentinel"

    atomic_save_world_model_retention_development_report(report, destination)
    assert destination.read_text(encoding="utf-8") == (
        world_model_retention_development_report_json(report)
    )
    loaded = json.loads(destination.read_text(encoding="utf-8"))
    assert validate_world_model_retention_development_report(loaded).valid
    assert not list(tmp_path.glob(".retention.json.*.tmp"))
