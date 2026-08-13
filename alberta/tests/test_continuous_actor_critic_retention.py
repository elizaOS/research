# mypy: disable-error-code="arg-type,attr-defined,call-arg,call-overload,no-any-return,operator"
"""Strict development contracts for continuous actor/critic recurrence diagnostics."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.continuous_average_reward_actor_critic import (
    ContinuousAverageRewardActorCriticAgent,
    ContinuousAverageRewardActorCriticConfig,
    ContinuousAverageRewardActorCriticState,
)
from alberta_framework.evaluation.continuous_actor_critic_retention import (
    CONTINUOUS_ACTOR_CRITIC_RETENTION_CHECKPOINT_SCHEMA,
    CONTINUOUS_ACTOR_CRITIC_RETENTION_CONFIG_SCHEMA,
    CONTINUOUS_ACTOR_CRITIC_RETENTION_PROTOCOL_SCHEMA,
    CONTINUOUS_ACTOR_CRITIC_RETENTION_REPORT_SCHEMA,
    ContinuousActorCriticRetentionConfig,
    ContinuousActorCriticRetentionEvaluator,
    ContinuousActorCriticRetentionProtocol,
    build_continuous_actor_critic_retention_report,
    canonical_continuous_actor_critic_retention_protocol,
    canonical_continuous_actor_critic_retention_report_bytes,
    continuous_actor_critic_retention_source_snapshot,
    frozen_continuous_actor_critic_state_sha256,
    load_continuous_actor_critic_retention_report,
    load_continuous_actor_critic_retention_snapshot_checkpoint,
    reconstruct_continuous_actor_critic_retention_summary,
    save_continuous_actor_critic_retention_report,
    save_continuous_actor_critic_retention_snapshot_checkpoint,
    validate_continuous_actor_critic_retention_report,
)

pytestmark = pytest.mark.unit


def _agent() -> ContinuousAverageRewardActorCriticAgent:
    return ContinuousAverageRewardActorCriticAgent(
        ContinuousAverageRewardActorCriticConfig(
            action_dim=1,
            action_low=-1.0,
            action_high=1.0,
            actor_step_size=0.015,
            critic_step_size=0.04,
            average_reward_step_size=0.01,
            actor_trace_lambda=0.3,
            critic_trace_lambda=0.4,
            target_log_std_init=-0.4,
            behavior_std_scale=1.4,
            max_updates=100,
        )
    )


def _snapshot(
    *, key: int = 0, target_latent_mean: float = 0.0
) -> tuple[
    ContinuousAverageRewardActorCriticAgent,
    ContinuousAverageRewardActorCriticState,
]:
    agent = _agent()
    state = agent.init(2, jr.key(key))
    state = state.replace(
        actor_params=state.actor_params.replace(
            mean_bias=jnp.asarray([target_latent_mean], dtype=jnp.float32)
        )
    )
    first = canonical_continuous_actor_critic_retention_protocol().events[0]
    started = agent.start(state, jnp.asarray(first.observation, dtype=jnp.float32))
    assert bool(started.accepted)
    return agent, started.state


def _config(**overrides: Any) -> ContinuousActorCriticRetentionConfig:
    values: dict[str, Any] = {
        "execution_mode": "jit",
        "recovery_window": 2,
        "max_phases": 3,
        "max_events": 12,
        "max_initial_snapshot_bytes": 2_000_000,
        "max_final_state_bytes": 2_000_000,
        "max_report_bytes": 4_000_000,
        "max_trace_scalar_values": 10_000,
        "activity_epsilon": 1.0e-8,
    }
    values.update(overrides)
    return ContinuousActorCriticRetentionConfig(**values)


def _event_trace(report: dict[str, object]) -> list[dict[str, object]]:
    payload = report["payload"]
    assert isinstance(payload, dict)
    trace = payload["event_trace"]
    assert isinstance(trace, list)
    return cast(list[dict[str, object]], trace)


def test_config_and_canonical_protocol_are_strict_development_only() -> None:
    config = _config()
    payload = config.to_config()
    assert payload["schema"] == CONTINUOUS_ACTOR_CRITIC_RETENTION_CONFIG_SCHEMA
    assert payload["development_status"] == "development-only-not-assessed"
    assert payload["retention_claimed"] is False
    assert payload["transfer_claimed"] is False
    assert payload["efficacy_claimed"] is False
    assert payload["calibration_claimed"] is False
    assert payload["sota_claimed"] is False
    assert payload["performance_thresholds_applied"] is False
    assert payload["candidate_update_safety_audit_performed"] is False
    assert payload["paper_defined_delight_computed"] is False
    assert payload["kondo_sparse_actor_backward_executed"] is False
    assert "kondo_sparks_joy_backward_selection_used" not in payload
    assert CONTINUOUS_ACTOR_CRITIC_RETENTION_CONFIG_SCHEMA == (
        "alberta.continuous-actor-critic-retention.config.v2"
    )
    assert ContinuousActorCriticRetentionConfig.from_config(payload) == config
    legacy_payload = dict(payload)
    legacy_payload["schema"] = "alberta.continuous-actor-critic-retention.config.v1"
    with pytest.raises(ValueError, match="schema"):
        ContinuousActorCriticRetentionConfig.from_config(legacy_payload)
    with pytest.raises(ValueError, match="fields"):
        ContinuousActorCriticRetentionConfig.from_config({**payload, "extra": 1})
    with pytest.raises(ValueError, match="absolute hard limit"):
        _config(max_report_bytes=16_000_001)
    with pytest.raises(ValueError, match="absolute hard limit"):
        _config(max_trace_scalar_values=20_001)

    protocol = canonical_continuous_actor_critic_retention_protocol()
    protocol_payload = protocol.to_config()
    assert protocol_payload["schema"] == CONTINUOUS_ACTOR_CRITIC_RETENTION_PROTOCOL_SCHEMA
    assert CONTINUOUS_ACTOR_CRITIC_RETENTION_PROTOCOL_SCHEMA == (
        "alberta.continuous-actor-critic-retention.protocol.v2"
    )
    assert protocol.learner_visible_fields == (
        "observation",
        "exact_cached_action",
        "realized_scalar_reward_after_action",
        "next_observation",
    )
    assert "phase_id" in protocol.evaluator_only_fields
    assert "preferred_action_center" in protocol.evaluator_only_fields
    assert "reward_function" in protocol.evaluator_only_fields
    assert "reference_value_target" in protocol.evaluator_only_fields
    assert len(protocol.phases) == 3
    assert len(protocol.events) == 12
    assert protocol.phases[2].recurrence_of_phase_id == "first-a"
    assert ContinuousActorCriticRetentionProtocol.from_config(protocol_payload) == protocol


def test_report_is_source_snapshot_final_and_component_hash_bound() -> None:
    agent, state = _snapshot(key=1)
    before = frozen_continuous_actor_critic_state_sha256(state)
    report = build_continuous_actor_critic_retention_report(agent, state, _config())
    assert frozen_continuous_actor_critic_state_sha256(state) == before
    assert report["schema"] == CONTINUOUS_ACTOR_CRITIC_RETENTION_REPORT_SCHEMA
    payload = report["payload"]
    assert isinstance(payload, dict)
    assert payload["development_only"] is True
    assert payload["assessment_status"] == "not-assessed"
    assert payload["retention_claimed"] is False
    assert payload["transfer_claimed"] is False
    assert payload["off_policy_state_distribution_correction_claimed"] is False
    assert payload["off_policy_convergence_claimed"] is False
    assert payload["candidate_update_safety_audit_performed"] is False
    assert payload["paper_defined_delight_computed"] is False
    assert payload["kondo_sparse_actor_backward_executed"] is False
    assert "kondo_sparks_joy_backward_selection_used" not in payload
    assert CONTINUOUS_ACTOR_CRITIC_RETENTION_REPORT_SCHEMA == (
        "alberta.continuous-actor-critic-retention.report.v2"
    )
    hashes = payload["hashes"]
    assert isinstance(hashes, dict)
    assert set(hashes) == {
        "config_sha256",
        "protocol_sha256",
        "source_manifest_sha256",
        "initial_snapshot_sha256",
        "event_trace_sha256",
        "final_isolated_state_sha256",
        "summary_sha256",
        "resource_accounting_sha256",
    }
    validation = validate_continuous_actor_critic_retention_report(report)
    assert validation.valid
    assert validation.assessment_status == "not-assessed"
    assert not validation.errors


def test_raw_trace_reconstructs_actions_densities_ratio_and_causal_boundary() -> None:
    agent, state = _snapshot(key=2)
    report = ContinuousActorCriticRetentionEvaluator(_config()).evaluate(agent, state)
    protocol = canonical_continuous_actor_critic_retention_protocol()
    trace = _event_trace(report)
    assert len(trace) == 12
    for record, event in zip(trace, protocol.events, strict=True):
        assert record["phase_id_learner_visible"] is False
        assert record["targets_learner_visible"] is False
        assert record["reward_function_learner_visible"] is False
        assert record["realized_scalar_reward_learner_visible_after_action"] is True
        assert record["observation"] == list(event.observation)
        assert record["next_observation"] == list(event.next_observation)
        assert record["preferred_action_center"] == event.preferred_action_center
        assert record["reference_value_target_raw"] == event.reference_value_target
        assert record["direct_transform_reconstruction_abs_error"] == 0.0
        assert int(record["target_log_density_reconstruction_ulp_distance"]) <= 8
        assert int(record["behavior_log_density_reconstruction_ulp_distance"]) <= 8
        assert record["rho_reconstruction_abs_error"] == 0.0
        assert record["decision_target_latent_mean"] == record["decision_behavior_latent_mean"]
        assert float(record["decision_behavior_std"]) >= float(record["decision_target_std"])
        assert float(record["decision_target_behavior_ratio"]) >= 0.0
        action = float(record["cached_action"])
        expected_reward = np.float32(1.0 - (action - event.preferred_action_center) ** 2)
        assert float(record["realized_reward"]) == float(expected_reward)
        assert record["update_accepted"] is True
        assert record["all_recorded_finite"] is True


def test_same_state_four_case_critic_centering_removes_the_differential_value_gauge() -> None:
    agent, state = _snapshot(key=3)
    report = build_continuous_actor_critic_retention_report(agent, state, _config())
    payload = report["payload"]
    assert isinstance(payload, dict)
    protocol = canonical_continuous_actor_critic_retention_protocol()
    trace = _event_trace(report)
    for index, record in enumerate(trace):
        raw = [float(value) for value in record["critic_same_state_case_predictions_raw"]]
        centered = [float(value) for value in record["critic_same_state_case_predictions_centered"]]
        target_raw = [float(value) for value in record["reference_value_targets_phase_raw"]]
        target_centered = [
            float(value) for value in record["reference_value_targets_phase_centered"]
        ]
        assert len(raw) == len(centered) == len(target_raw) == len(target_centered) == 4
        assert float(record["critic_same_state_case_prediction_mean"]) == pytest.approx(
            sum(raw) / 4
        )
        assert centered == pytest.approx([value - sum(raw) / 4 for value in raw], abs=2e-7)
        assert sum(centered) == pytest.approx(0.0, abs=2e-7)
        assert sum(target_centered) == pytest.approx(0.0, abs=2e-7)
        selected = index % 4
        assert float(record["critic_prediction_raw"]) == raw[selected]
        assert float(record["critic_prediction_same_state_centered"]) == centered[selected]
        expected = centered[selected] - target_centered[selected]
        assert float(record["critic_same_state_centered_error"]) == pytest.approx(expected)
    summary = payload["summary"]
    assert isinstance(summary, dict)
    reconstructed = reconstruct_continuous_actor_critic_retention_summary(
        trace,
        protocol,
        recovery_window=2,
        initial_snapshot=payload["initial_snapshot"],
        final_isolated_state=payload["final_isolated_state"],
    )
    assert reconstructed == summary

    shifted = state.replace(
        critic_params=state.critic_params.replace(
            bias=state.critic_params.bias + jnp.asarray(7.0, dtype=jnp.float32)
        )
    )
    shifted_report = build_continuous_actor_critic_retention_report(agent, shifted, _config())
    shifted_first = _event_trace(shifted_report)[0]
    first = trace[0]
    first_raw = [float(value) for value in first["critic_same_state_case_predictions_raw"]]
    shifted_raw = [
        float(value) for value in shifted_first["critic_same_state_case_predictions_raw"]
    ]
    assert shifted_raw == pytest.approx([value + 7.0 for value in first_raw])
    assert shifted_first["critic_same_state_case_predictions_centered"] == pytest.approx(
        first["critic_same_state_case_predictions_centered"]
    )
    assert shifted_first["critic_same_state_centered_error"] == pytest.approx(
        first["critic_same_state_centered_error"]
    )


def test_target_latent_mean_is_not_mislabeled_as_bounded_median_action() -> None:
    agent, state = _snapshot(key=30, target_latent_mean=0.7)
    report = build_continuous_actor_critic_retention_report(agent, state, _config())
    record = _event_trace(report)[0]
    latent_mean = float(record["decision_target_latent_mean"])
    median_action = float(record["target_median_action"])
    expected_median = float(np.tanh(np.float32(0.7)))
    assert latent_mean == pytest.approx(0.7, rel=1e-6)
    assert median_action == pytest.approx(expected_median, rel=1e-6)
    assert median_action != pytest.approx(latent_mean)
    protocol = canonical_continuous_actor_critic_retention_protocol()
    expected_error = median_action - protocol.events[0].preferred_action_center
    assert float(record["target_median_action_error"]) == pytest.approx(expected_error)


def test_trace_retains_churn_postupdate_plasticity_saturation_and_activity() -> None:
    agent, state = _snapshot(key=4)
    report = build_continuous_actor_critic_retention_report(agent, state, _config())
    trace = _event_trace(report)
    required = {
        "target_median_action_error",
        "sampled_action_error",
        "sampled_action_squared_error",
        "realized_reward",
        "target_latent_mean_churn_available",
        "target_latent_mean_churn_abs",
        "target_std_churn_available",
        "target_std_churn_abs",
        "postupdate_target_latent_mean_change_abs",
        "postupdate_target_median_action_change_abs",
        "postupdate_target_std_change_abs",
        "td_error",
        "average_reward_before_update",
        "average_reward_after_update",
        "actor_parameter_update_l2",
        "critic_parameter_update_l2",
        "actor_trace_update_l2",
        "critic_trace_update_l2",
        "actor_parameter_delta_from_initial_l2",
        "critic_parameter_delta_from_initial_l2",
        "action_boundary_saturated",
        "log_std_boundary_saturated_after_update",
        "sampled_action_active",
        "target_latent_mean_active",
        "target_median_action_active",
        "actor_update_active",
        "critic_update_active",
        "next_cached_action",
    }
    assert required <= set(trace[0])
    assert any(bool(record["target_latent_mean_churn_available"]) for record in trace[4:])
    assert any(bool(record["actor_update_active"]) for record in trace)
    assert any(bool(record["critic_update_active"]) for record in trace)
    assert all(np.isfinite(float(record["td_error"])) for record in trace)


def test_validator_reconstructs_and_rejects_trace_or_hash_tampering() -> None:
    agent, state = _snapshot(key=5)
    report = build_continuous_actor_critic_retention_report(agent, state, _config())
    live = validate_continuous_actor_critic_retention_report(report, agent=agent, state=state)
    assert live.valid

    tampered = json.loads(json.dumps(report))
    tampered["payload"]["event_trace"][0]["cached_action"] += 0.01
    invalid = validate_continuous_actor_critic_retention_report(tampered)
    assert not invalid.valid

    rehashed = json.loads(json.dumps(report))
    rehashed["payload"]["summary"]["total_realized_return"] += 0.25
    encoded = json.dumps(
        rehashed["payload"], allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    rehashed["payload_sha256"] = hashlib.sha256(encoded).hexdigest()
    assert not validate_continuous_actor_critic_retention_report(rehashed).valid


def test_report_canonical_json_no_overwrite_and_duplicate_rejection(tmp_path: Path) -> None:
    agent, state = _snapshot(key=6)
    report = build_continuous_actor_critic_retention_report(agent, state, _config())
    path = tmp_path / "continuous-retention.json"
    save_continuous_actor_critic_retention_report(report, path)
    assert path.read_bytes() == canonical_continuous_actor_critic_retention_report_bytes(report)
    assert load_continuous_actor_critic_retention_report(path) == report
    with pytest.raises(FileExistsError, match="overwrite"):
        save_continuous_actor_critic_retention_report(report, path)

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema":1,"schema":2}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_continuous_actor_critic_retention_report(duplicate)


def test_snapshot_checkpoint_roundtrip_is_strict_replayable_and_no_overwrite(
    tmp_path: Path,
) -> None:
    agent, state = _snapshot(key=7)
    before = frozen_continuous_actor_critic_state_sha256(state)
    path = tmp_path / "continuous-snapshot.json"
    save_continuous_actor_critic_retention_snapshot_checkpoint(agent, state, path)
    restored_agent, restored_state = load_continuous_actor_critic_retention_snapshot_checkpoint(
        path
    )
    assert restored_agent.config == agent.config
    assert frozen_continuous_actor_critic_state_sha256(restored_state) == before
    assert path.read_bytes().startswith(b'{"payload":')
    report = build_continuous_actor_critic_retention_report(
        restored_agent, restored_state, _config()
    )
    assert validate_continuous_actor_critic_retention_report(report, agent=agent, state=state).valid
    with pytest.raises(FileExistsError, match="overwrite"):
        save_continuous_actor_critic_retention_snapshot_checkpoint(agent, state, path)
    assert CONTINUOUS_ACTOR_CRITIC_RETENTION_CHECKPOINT_SCHEMA.encode() in path.read_bytes()
    assert CONTINUOUS_ACTOR_CRITIC_RETENTION_CHECKPOINT_SCHEMA == (
        "alberta.continuous-actor-critic-retention.snapshot.v2"
    )

    tampered_document = json.loads(path.read_text(encoding="utf-8"))
    tampered_payload = tampered_document["payload"]
    core_checkpoint = tampered_payload["core_checkpoint"]
    core_checkpoint["state"]["last_sample"]["target_behavior_ratio"] += 0.25
    core_encoded = json.dumps(
        core_checkpoint, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    tampered_payload["core_checkpoint_sha256"] = hashlib.sha256(core_encoded).hexdigest()
    payload_encoded = json.dumps(
        tampered_payload, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    tampered_document["payload_sha256"] = hashlib.sha256(payload_encoded).hexdigest()
    tampered_path = tmp_path / "tampered-continuous-snapshot.json"
    tampered_path.write_text(
        json.dumps(
            tampered_document,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="core checkpoint"):
        load_continuous_actor_critic_retention_snapshot_checkpoint(tampered_path)


def test_eager_and_jit_execution_match_raw_diagnostics_within_float32_roundoff() -> None:
    agent, state = _snapshot(key=8)
    eager = build_continuous_actor_critic_retention_report(
        agent, state, _config(execution_mode="eager")
    )
    compiled = build_continuous_actor_critic_retention_report(
        agent, state, _config(execution_mode="jit")
    )
    eager_trace = _event_trace(eager)
    compiled_trace = _event_trace(compiled)
    for left, right in zip(eager_trace, compiled_trace, strict=True):
        assert left["event_id"] == right["event_id"]
        for field in (
            "cached_pre_tanh_action",
            "cached_action",
            "decision_target_log_density",
            "decision_behavior_log_density",
            "decision_target_behavior_ratio",
            "critic_prediction_raw",
            "td_error",
            "average_reward_after_update",
            "next_cached_action",
        ):
            np.testing.assert_allclose(
                float(left[field]), float(right[field]), rtol=2e-5, atol=2e-6
            )


def test_invalid_snapshot_and_hard_resource_limits_fail_before_report() -> None:
    agent = _agent()
    unstarted = agent.init(2, jr.key(9))
    with pytest.raises(ValueError, match="started"):
        build_continuous_actor_critic_retention_report(agent, unstarted, _config())
    agent, state = _snapshot(key=9)
    wrong_observation = state.replace(
        last_sample=state.last_sample.replace(
            observation=jnp.asarray([0.0, 1.0], dtype=jnp.float32)
        )
    )
    with pytest.raises(ValueError, match="snapshot"):
        build_continuous_actor_critic_retention_report(agent, wrong_observation, _config())
    with pytest.raises(ValueError, match="snapshot byte"):
        build_continuous_actor_critic_retention_report(
            agent, state, _config(max_initial_snapshot_bytes=1)
        )
    with pytest.raises(ValueError, match="report byte"):
        build_continuous_actor_critic_retention_report(agent, state, _config(max_report_bytes=1))


def test_source_closure_contains_new_core_and_evaluator() -> None:
    sources = continuous_actor_critic_retention_source_snapshot()
    assert "alberta_framework/core/continuous_average_reward_actor_critic.py" in sources
    assert "alberta_framework/evaluation/continuous_actor_critic_retention.py" in sources
    assert all(len(digest) == 64 for digest in sources.values())
