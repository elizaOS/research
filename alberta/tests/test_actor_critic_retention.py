# mypy: disable-error-code="attr-defined,call-arg,index"
"""Strict contracts for the development-only actor/critic retention probe."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.average_reward import (
    AverageRewardHordeActorCriticAgent,
    AverageRewardHordeActorCriticConfig,
    AverageRewardHordeActorCriticState,
)
from alberta_framework.core.checkpoints import load_checkpoint_metadata
from alberta_framework.evaluation.actor_critic_retention import (
    ACTOR_CRITIC_RETENTION_CHECKPOINT_SCHEMA,
    ACTOR_CRITIC_RETENTION_CONFIG_SCHEMA,
    ACTOR_CRITIC_RETENTION_PROTOCOL_SCHEMA,
    ACTOR_CRITIC_RETENTION_REPORT_SCHEMA,
    ActorCriticRetentionConfig,
    ActorCriticRetentionEvaluator,
    build_actor_critic_retention_report,
    canonical_actor_critic_retention_protocol,
    canonical_actor_critic_retention_report_bytes,
    frozen_actor_critic_state_sha256,
    load_actor_critic_retention_report,
    load_actor_critic_retention_snapshot_checkpoint,
    reconstruct_actor_critic_retention_summary,
    save_actor_critic_retention_report,
    save_actor_critic_retention_snapshot_checkpoint,
    validate_actor_critic_retention_report,
)

pytestmark = pytest.mark.development


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _agent(*, frozen: bool = False) -> AverageRewardHordeActorCriticAgent:
    return AverageRewardHordeActorCriticAgent(
        AverageRewardHordeActorCriticConfig(
            n_actions=2,
            hidden_sizes=(3,),
            critic_step_size=0.0 if frozen else 0.025,
            average_reward_step_size=0.0 if frozen else 0.01,
            epsilon=1.0 if frozen else 0.15,
            actor_update_clip=0.2,
        )
    )


def _config(*, execution_mode: str = "jit") -> ActorCriticRetentionConfig:
    return ActorCriticRetentionConfig(
        action_seed=37,
        recovery_window=2,
        max_phases=3,
        max_events=12,
        max_initial_snapshot_bytes=1_000_000,
        max_report_bytes=2_000_000,
        execution_mode=execution_mode,
    )


@pytest.fixture(scope="module")
def fixture() -> tuple[
    AverageRewardHordeActorCriticAgent,
    AverageRewardHordeActorCriticState,
    ActorCriticRetentionConfig,
    dict[str, object],
]:
    agent = _agent()
    state = agent.init(2, jr.key(11))
    config = _config()
    report = build_actor_critic_retention_report(agent, state, config)
    return agent, state, config, report


def test_fixed_protocol_is_continuing_recurring_and_evaluator_owned() -> None:
    protocol = canonical_actor_critic_retention_protocol()
    assert protocol.phase_ids == ("first-a", "interference-b", "return-a")
    assert protocol.phases[-1].recurrence_of_phase_id == "first-a"
    assert len(protocol.events) == 12
    assert all(
        event.next_observation == protocol.events[(index + 1) % len(protocol.events)].observation
        for index, event in enumerate(protocol.events)
    )
    first = protocol.events[:4]
    returned = protocol.events[-4:]
    assert [event.case_id for event in returned] == [event.case_id for event in first]
    assert [event.observation for event in returned] == [event.observation for event in first]
    assert [event.action_rewards for event in returned] == [
        event.action_rewards for event in first
    ]
    assert protocol.learner_visible_fields == (
        "observation",
        "cached_sampled_action",
        "realized_scalar_reward_after_action",
        "next_observation",
    )
    assert protocol.evaluator_only_fields == (
        "phase_id",
        "case_id",
        "preferred_action",
        "reference_value_target",
        "action_rewards",
    )
    protocol_event = cast(list[Mapping[str, object]], protocol.to_config()["events"])[0]
    assert protocol_event["reward_table_learner_visible"] is False
    assert protocol_event["realized_scalar_reward_learner_visible_after_action"] is True
    assert type(protocol).from_config(protocol.to_config()) == protocol
    assert ActorCriticRetentionConfig.from_config(_config().to_config()) == _config()
    assert ACTOR_CRITIC_RETENTION_CONFIG_SCHEMA == (
        "alberta.actor-critic-retention.config.v2"
    )
    assert ACTOR_CRITIC_RETENTION_PROTOCOL_SCHEMA == (
        "alberta.actor-critic-retention.protocol.v2"
    )
    legacy_config = _config().to_config()
    legacy_config["schema"] = "alberta.actor-critic-retention.config.v1"
    with pytest.raises(ValueError, match="schema"):
        ActorCriticRetentionConfig.from_config(legacy_config)


def test_report_is_preupdate_ordinary_pg_and_does_not_mutate_snapshot(
    fixture: tuple[
        AverageRewardHordeActorCriticAgent,
        AverageRewardHordeActorCriticState,
        ActorCriticRetentionConfig,
        dict[str, object],
    ],
) -> None:
    agent, state, config, report = fixture
    before = frozen_actor_critic_state_sha256(state)
    payload = cast(Mapping[str, object], report["payload"])
    trace = cast(list[Mapping[str, object]], payload["event_trace"])
    first_event = canonical_actor_critic_retention_protocol().events[0]

    isolated = jax.tree.map(lambda leaf: leaf.copy() if hasattr(leaf, "copy") else leaf, state)
    isolated = isolated.replace(rng_key=jr.key(config.action_seed))
    isolated, action = agent.start(
        isolated,
        jnp.asarray(first_event.observation, dtype=jnp.float32),
    )
    expected_critic = agent.critic.predict(
        isolated.critic_state,
        jnp.asarray(first_event.observation, dtype=jnp.float32),
    )[0]
    first = trace[0]

    assert report["schema"] == ACTOR_CRITIC_RETENTION_REPORT_SCHEMA
    assert ACTOR_CRITIC_RETENTION_REPORT_SCHEMA == (
        "alberta.actor-critic-retention.report.v2"
    )
    assert payload["development_only"] is True
    assert payload["assessment_status"] == "not-assessed"
    assert payload["policy_gradient_mode"] == "ordinary"
    assert payload["target_policy_semantics"] == "softmax-target-before-epsilon-exploration"
    assert payload["behavior_policy_semantics"] == (
        "fixed-epsilon-mixture-used-for-action-sampling"
    )
    assert payload["actor_score_chain_rule_semantics"] == (
        "exact-epsilon-mixture-behavior-score-chain-rule-scale"
    )
    assert payload["off_policy_target_policy_correction_claimed"] is False
    assert payload["paper_specific_dg_delight_used"] is False
    assert payload["candidate_update_safety_audit_performed"] is False
    assert "literal_gradient_joy_audited" not in payload
    assert payload["retention_claimed"] is False
    assert payload["efficacy_claimed"] is False
    assert payload["scientific_promotion_allowed"] is False
    assert payload["performance_thresholds_applied"] is False
    assert first["event_order"] == [
        "cached-decision-consumed-before-update",
        "critic-predicted-before-update",
        "reward-realized-after-action",
        "atomic-update-committed-before-next-decision",
        "next-decision-sampled-from-committed-parameters",
    ]
    assert first["action"] == int(action)
    assert first["critic_prediction"] == float(expected_critic)
    np.testing.assert_array_equal(
        first["decision_target_policy"],
        isolated.last_policy_sample.target_policy,
    )
    assert first["realized_reward"] == first_event.action_rewards[int(action)]
    assert first["phase_id_learner_visible"] is False
    assert first["targets_learner_visible"] is False
    assert first["reward_table_learner_visible"] is False
    assert first["realized_scalar_reward_learner_visible_after_action"] is True
    for event in trace:
        np.testing.assert_array_equal(
            event["decision_target_policy"], event["preupdate_current_target_policy"]
        )
        np.testing.assert_array_equal(
            event["decision_behavior_policy"],
            event["preupdate_current_behavior_policy"],
        )
        action_index = cast(int, event["action"])
        assert event["decision_target_action_probability"] == cast(
            list[float], event["decision_target_policy"]
        )[action_index]
        assert event["decision_behavior_action_probability"] == cast(
            list[float], event["decision_behavior_policy"]
        )[action_index]
        score = np.asarray(event["actor_score_scale"], dtype=np.float32)
        expected_score = np.asarray(
            event["actor_score_scale_expected_from_probabilities"], dtype=np.float32
        )
        assert score.tobytes() == expected_score.tobytes()
        target_probability = np.asarray(
            event["decision_target_action_probability"], dtype=np.float32
        )
        behavior_probability = np.asarray(
            event["decision_behavior_action_probability"], dtype=np.float32
        )
        ratio = target_probability / np.maximum(
            behavior_probability,
            np.asarray(1.0e-8, dtype=np.float32),
        )
        logged_ratio = np.asarray(
            event["target_behavior_action_probability_ratio"], dtype=np.float32
        )
        assert ratio.tobytes() == logged_ratio.tobytes()
    assert frozen_actor_critic_state_sha256(state) == before


def test_phase_recurrence_plasticity_and_activity_metrics_reconstruct(
    fixture: tuple[
        AverageRewardHordeActorCriticAgent,
        AverageRewardHordeActorCriticState,
        ActorCriticRetentionConfig,
        dict[str, object],
    ],
) -> None:
    _, _, _, report = fixture
    payload = cast(Mapping[str, object], report["payload"])
    summary = cast(Mapping[str, object], payload["summary"])
    assert reconstruct_actor_critic_retention_summary(
        cast(list[Mapping[str, object]], payload["event_trace"]),
        canonical_actor_critic_retention_protocol(),
        recovery_window=2,
        initial_snapshot=cast(Mapping[str, object], payload["initial_snapshot"]),
        final_isolated_state=cast(Mapping[str, object], payload["final_isolated_state"]),
    ) == summary

    phases = cast(list[Mapping[str, object]], summary["phase_metrics"])
    assert [phase["phase_id"] for phase in phases] == [
        "first-a",
        "interference-b",
        "return-a",
    ]
    assert all(phase["event_count"] == 4 for phase in phases)
    assert all("mean_critic_value_squared_error" in phase for phase in phases)
    assert all("mean_actor_action_probability_margin" in phase for phase in phases)
    assert all("total_realized_return" in phase for phase in phases)
    assert all("within_phase_realized_return_recovery_delta" in phase for phase in phases)

    recurrence = cast(list[Mapping[str, object]], summary["recurrence_metrics"])
    assert len(recurrence) == 1
    assert recurrence[0]["phase_id"] == "return-a"
    assert recurrence[0]["reference_phase_id"] == "first-a"
    assert recurrence[0]["exact_case_reuse"] is True
    assert recurrence[0]["case_count"] == 4
    assert "mean_reference_phase_policy_l1" in recurrence[0]

    plasticity = cast(Mapping[str, object], summary["plasticity_diagnostics"])
    assert set(plasticity) == {
        "actor_parameter_delta_l2",
        "critic_parameter_delta_l2",
        "actor_parameter_update_l2_sum",
        "critic_parameter_update_l2_sum",
        "actor_parameter_update_nonzero_event_count",
        "critic_parameter_update_nonzero_event_count",
        "policy_update_l1_sum",
        "policy_update_nonzero_event_count",
        "absolute_td_error_sum",
        "nonzero_td_error_event_count",
    }
    activity = cast(Mapping[str, object], summary["action_activity_diagnostics"])
    assert sum(cast(list[int], activity["action_counts"])) == 12
    assert 1 <= cast(int, activity["unique_action_count"]) <= 2
    assert 0 <= cast(int, activity["action_switch_count"]) <= 11
    assert summary["claims"] == {
        "retention_established": False,
        "efficacy_established": False,
        "calibration_established": False,
        "scientific_promotion": False,
        "alberta_plan_completion": False,
    }


def test_frozen_constant_policy_loophole_is_explicitly_disclosed() -> None:
    agent = _agent(frozen=True)
    state = agent.init(2, jr.key(13))
    report = build_actor_critic_retention_report(agent, state, _config())
    payload = cast(Mapping[str, object], report["payload"])
    summary = cast(Mapping[str, object], payload["summary"])
    plasticity = cast(Mapping[str, object], summary["plasticity_diagnostics"])
    assert plasticity["actor_parameter_delta_l2"] == 0.0
    assert plasticity["critic_parameter_delta_l2"] == 0.0
    assert plasticity["actor_parameter_update_nonzero_event_count"] == 0
    assert plasticity["critic_parameter_update_nonzero_event_count"] == 0
    assert plasticity["policy_update_l1_sum"] == 0.0
    assert summary["claims"]["retention_established"] is False  # type: ignore[index]
    assert validate_actor_critic_retention_report(report).valid


def test_eager_and_jit_traces_are_deterministic_and_validator_replays(
    fixture: tuple[
        AverageRewardHordeActorCriticAgent,
        AverageRewardHordeActorCriticState,
        ActorCriticRetentionConfig,
        dict[str, object],
    ],
) -> None:
    agent, state, _, jit_report = fixture
    eager_report = build_actor_critic_retention_report(
        agent,
        state,
        _config(execution_mode="eager"),
    )
    jit_payload = cast(Mapping[str, object], jit_report["payload"])
    eager_payload = cast(Mapping[str, object], eager_report["payload"])

    def assert_numerical_parity(left: object, right: object) -> None:
        if isinstance(left, Mapping) and isinstance(right, Mapping):
            assert set(left) == set(right)
            for key in left:
                assert_numerical_parity(left[key], right[key])
        elif isinstance(left, list) and isinstance(right, list):
            assert len(left) == len(right)
            for left_item, right_item in zip(left, right, strict=True):
                assert_numerical_parity(left_item, right_item)
        elif type(left) is float and type(right) is float:
            assert left == pytest.approx(right, rel=0.0, abs=2.0e-7)
        else:
            assert left == right

    assert_numerical_parity(eager_payload["event_trace"], jit_payload["event_trace"])
    assert_numerical_parity(eager_payload["summary"], jit_payload["summary"])

    validation = validate_actor_critic_retention_report(
        jit_report,
        agent=agent,
        state=state,
    )
    assert validation.valid, validation.errors

    changed = copy.deepcopy(jit_report)
    changed_payload = cast(dict[str, object], changed["payload"])
    trace = cast(list[dict[str, object]], changed_payload["event_trace"])
    trace[0]["realized_reward"] = cast(float, trace[0]["realized_reward"]) + 0.25
    hashes = cast(dict[str, object], changed_payload["hashes"])
    hashes["event_trace_sha256"] = _digest(trace)
    changed["payload_sha256"] = _digest(changed_payload)
    invalid = validate_actor_critic_retention_report(changed)
    assert not invalid.valid
    assert any("trace" in error or "summary" in error for error in invalid.errors)

    legacy = copy.deepcopy(jit_report)
    legacy["schema"] = "alberta.actor-critic-retention.report.v1"
    assert not validate_actor_critic_retention_report(legacy).valid


def test_canonical_report_resources_and_snapshot_checkpoint_round_trip(
    tmp_path: Path,
    fixture: tuple[
        AverageRewardHordeActorCriticAgent,
        AverageRewardHordeActorCriticState,
        ActorCriticRetentionConfig,
        dict[str, object],
    ],
) -> None:
    agent, state, config, report = fixture
    payload = cast(Mapping[str, object], report["payload"])
    resources = cast(Mapping[str, object], payload["resource_accounting"])
    assert resources["phase_count"] == 3
    assert resources["event_count"] == resources["updates_executed"] == 12
    assert resources["decisions_sampled"] == 13
    assert resources["categorical_action_draw_count"] == 13
    assert resources["learner_visible_phase_identifiers"] == 0
    assert resources["learner_visible_target_fields"] == 0
    assert resources["external_snapshot_mutations"] == 0
    assert resources["candidate_update_safety_audits"] == 0
    assert "literal_gradient_joy_audits" not in resources
    assert resources["initial_snapshot_state_bytes"] <= config.max_initial_snapshot_bytes
    assert resources["canonical_report_bytes"] == len(
        canonical_actor_critic_retention_report_bytes(report)
    )

    report_path = tmp_path / "actor-critic-retention.json"
    save_actor_critic_retention_report(report, report_path)
    assert load_actor_critic_retention_report(report_path) == report
    with pytest.raises(FileExistsError, match="overwrite"):
        save_actor_critic_retention_report(report, report_path)

    checkpoint = tmp_path / "actor-critic-retention.ckpt"
    save_actor_critic_retention_snapshot_checkpoint(agent, state, checkpoint)
    assert load_checkpoint_metadata(checkpoint)["schema"] == (
        ACTOR_CRITIC_RETENTION_CHECKPOINT_SCHEMA
    )
    assert ACTOR_CRITIC_RETENTION_CHECKPOINT_SCHEMA == (
        "alberta.actor-critic-retention.snapshot.v2"
    )
    restored_agent, restored_state = load_actor_critic_retention_snapshot_checkpoint(checkpoint)
    assert restored_agent.to_config() == agent.to_config()
    assert frozen_actor_critic_state_sha256(restored_state) == (
        frozen_actor_critic_state_sha256(state)
    )
    with pytest.raises(FileExistsError, match="overwrite"):
        save_actor_critic_retention_snapshot_checkpoint(agent, state, checkpoint)

    evaluator = ActorCriticRetentionEvaluator(config)
    assert evaluator.evaluate(agent, state) == report


def test_invalid_bounds_and_nonfinite_snapshot_fail_closed() -> None:
    with pytest.raises(ValueError, match="max_events"):
        dataclasses.replace(_config(), max_events=11)
    with pytest.raises(ValueError, match="action_seed"):
        dataclasses.replace(_config(), action_seed=-1)
    with pytest.raises(ValueError, match="execution_mode"):
        dataclasses.replace(_config(), execution_mode="vectorized")

    agent = _agent()
    state = agent.init(2, jr.key(3)).replace(
        actor_bias=jnp.asarray([jnp.nan, 0.0], dtype=jnp.float32)
    )
    with pytest.raises(ValueError, match="finite"):
        build_actor_critic_retention_report(agent, state, _config())

    bounded_state = agent.init(2, jr.key(5))
    with pytest.raises(ValueError, match="snapshot byte bound"):
        build_actor_critic_retention_report(
            agent,
            bounded_state,
            dataclasses.replace(_config(), max_initial_snapshot_bytes=1),
        )
