"""Contracts for the bounded WP1 Prototype continual-control report."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from typing import cast

import pytest

from alberta_framework.evaluation.prototype_continual_control_development import (
    ASSESSMENT_STATUS,
    DEVELOPMENT_SEEDS,
    PROTOTYPE_CONTROL_DEVELOPMENT_REPORT_SCHEMA,
    PrototypeContinualControlDevelopmentConfig,
    build_prototype_continual_control_development_report,
    validate_prototype_continual_control_development_report,
)


def _mapping(value: object) -> Mapping[str, object]:
    return cast(Mapping[str, object], value)


def _digest(value: object) -> str:
    raw = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture(scope="module")
def report() -> dict[str, object]:
    return build_prototype_continual_control_development_report()


@pytest.mark.unit
def test_report_is_fixed_development_only_and_replay_bound(
    report: dict[str, object],
) -> None:
    assert report["schema_version"] == PROTOTYPE_CONTROL_DEVELOPMENT_REPORT_SCHEMA
    assert report["assessment_status"] == ASSESSMENT_STATUS == "not_assessed"
    assert report["development_seeds"] == list(DEVELOPMENT_SEEDS)
    assert report["thresholds"] == []
    assert report["winner"] is None
    assert report["efficacy_claimed"] is False
    assert report["scientific_promotion_allowed"] is False
    assert report["output_written"] is False
    assert report["exact_causal_replay_required"] is True
    assert report["source_runtime_bound"] is True


@pytest.mark.unit
def test_each_seed_has_prototype_flat_and_frozen_with_exact_opportunities(
    report: dict[str, object],
) -> None:
    runs = cast(list[Mapping[str, object]], report["runs"])
    assert len(runs) == len(DEVELOPMENT_SEEDS)
    for run in runs:
        control = _mapping(run["control_report"])
        conditions = cast(list[Mapping[str, object]], control["conditions"])
        assert [condition["name"] for condition in conditions] == [
            "prototype_agent",
            "flat_running_reward_mean",
            "frozen_action_zero",
        ]
        opportunities = _mapping(run["opportunity_accounting"])
        assert opportunities["declared_transitions_per_condition"] == 6
        assert opportunities["identical_exogenous_opportunities"] is True
        assert opportunities["realized_transition_counts"] == {
            "prototype_agent": 6,
            "flat_running_reward_mean": 6,
            "frozen_action_zero": 6,
        }
        ownership = cast(list[Mapping[str, object]], run["ownership_traces"])
        assert len(ownership) == 3
        for condition in ownership:
            records = cast(list[Mapping[str, object]], condition["records"])
            assert len(records) == 6
            decision_ids = [tuple(cast(list[int], record["decision_id"])) for record in records]
            assert len(set(decision_ids)) == len(decision_ids)
            assert condition["consumed_decision_ids"] == [list(value) for value in decision_ids]
        for condition in conditions:
            trace = _mapping(condition["trace"])
            assert trace["evaluator_regime_ids_in_learner_trace"] is False
            assert condition["predict_action_before_environment_outcome"] is True


@pytest.mark.unit
def test_all_wp1_diagnostic_fields_have_explicit_three_state_applicability(
    report: dict[str, object],
) -> None:
    expected_plasticity = {
        "dormant_units",
        "activation_entropy",
        "effective_rank",
        "stable_rank",
        "parameter_norm",
        "gradient_norm",
        "sampled_ntk_rank",
        "policy_churn",
        "value_churn",
    }
    expected_retention = {
        "feature_survival_curve",
        "prediction_survival_curve",
        "option_survival_curve",
        "model_survival_curve",
    }
    run = cast(list[Mapping[str, object]], report["runs"])[0]
    diagnostics = _mapping(run["diagnostics"])
    conditions = _mapping(diagnostics["conditions"])
    prototype = _mapping(conditions["prototype_agent"])
    plasticity = _mapping(prototype["plasticity"])
    retention = _mapping(prototype["component_retention"])
    assert set(plasticity) == expected_plasticity
    assert set(retention) == expected_retention
    for record in (*plasticity.values(), *retention.values()):
        item = _mapping(record)
        assert (item["applicable"], item["available"]) in {
            (True, True),
            (True, False),
            (False, False),
        }
        assert item["status"] in {"available", "unavailable", "inapplicable"}
    assert _mapping(plasticity["parameter_norm"])["status"] == "available"
    assert _mapping(plasticity["policy_churn"])["status"] == "available"
    assert _mapping(plasticity["value_churn"])["status"] == "available"
    assert _mapping(plasticity["gradient_norm"])["status"] == "unavailable"
    assert _mapping(plasticity["dormant_units"])["status"] == "inapplicable"

    coverage = _mapping(report["wp1_field_coverage"])
    world_model = _mapping(coverage["world_model"])
    assert all(_mapping(value)["status"] == "inapplicable" for value in world_model.values())
    energy = _mapping(coverage["energy_proxy"])
    assert energy["status"] == "unavailable"
    fresh = _mapping(_mapping(report["references"])["fresh_per_regime"])
    assert fresh["status"] == "unavailable"
    assert fresh["ordinary_lane_executed"] is False
    assert "PrivilegedContinualControlReferenceSuite" in cast(str, fresh["available_via"])


@pytest.mark.unit
def test_report_validation_rejects_raw_ownership_and_digest_tamper(
    report: dict[str, object],
) -> None:
    tampered = copy.deepcopy(report)
    tampered["runs"][0]["ownership_traces"][0]["records"][0]["action"] ^= 1
    body = {name: tampered[name] for name in tampered if name != "report_sha256"}
    tampered["report_sha256"] = _digest(body)
    with pytest.raises(ValueError, match="ownership"):
        validate_prototype_continual_control_development_report(tampered)

    tampered = copy.deepcopy(report)
    tampered["report_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="digest"):
        validate_prototype_continual_control_development_report(tampered)


@pytest.mark.unit
def test_config_rejects_seed_or_horizon_drift() -> None:
    config = PrototypeContinualControlDevelopmentConfig()
    payload = config.to_config()
    payload["development_seeds"] = [1, 2]
    with pytest.raises(ValueError, match="development_seeds"):
        PrototypeContinualControlDevelopmentConfig.from_config(payload)
    payload = config.to_config()
    payload["horizon"] = 7
    with pytest.raises(ValueError, match="horizon"):
        PrototypeContinualControlDevelopmentConfig.from_config(payload)
