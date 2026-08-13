# mypy: disable-error-code="index,no-any-return"
"""Short slow smoke for the strict two-learning-agent recurrence report."""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, cast

import pytest

from alberta_framework.evaluation.prototype_two_learning_agent_recurrence_development import (
    CLAIM_ASSESSMENTS,
    PROTOTYPE_TWO_LEARNING_AGENT_RECURRENCE_REPORT_SCHEMA,
    PrototypeTwoLearningAgentRecurrenceProtocol,
    _comparison_contract,
    _metrics_from_trace,
    _work_from_trace,
    prototype_two_learning_agent_recurrence_report_json,
    run_prototype_two_learning_agent_recurrence_development,
    validate_prototype_two_learning_agent_recurrence_report,
)

pytestmark = [pytest.mark.integration, pytest.mark.slow]


@pytest.fixture(scope="module")
def short_report() -> dict[str, object]:
    protocol = PrototypeTwoLearningAgentRecurrenceProtocol(
        segment_length=1,
        active_pair_slots=2,
        memory_capacity=2,
        replacement_interval=1,
        metric_window=1,
        arm_names=("joint_full", "memory_readout_blocked"),
    )
    return run_prototype_two_learning_agent_recurrence_development(protocol, seed=11)


def _runs(report: dict[str, object]) -> list[dict[str, Any]]:
    return cast(list[dict[str, Any]], report["runs"])


def _payload_digest(value: object) -> str:
    canonical = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _refresh_derived_payloads(report: dict[str, object]) -> None:
    protocol = PrototypeTwoLearningAgentRecurrenceProtocol.from_config(
        cast(dict[str, object], report["protocol"])
    )
    for run in _runs(report):
        run["trace_sha256"] = _payload_digest(run["trace"])
        run["metrics"] = _metrics_from_trace(run["trace"], protocol)
        run["work"] = _work_from_trace(run["trace"], run["resources"])
    report["comparison_contract"] = _comparison_contract(_runs(report))
    report["report_sha256"] = _payload_digest(
        {key: value for key, value in report.items() if key != "report_sha256"}
    )


def test_short_report_is_strict_canonical_and_explicitly_nonpromoting(
    short_report: dict[str, object],
) -> None:
    validation = validate_prototype_two_learning_agent_recurrence_report(short_report)

    assert validation.valid, validation.errors
    assert short_report["schema_version"] == (
        PROTOTYPE_TWO_LEARNING_AGENT_RECURRENCE_REPORT_SCHEMA
    )
    assert short_report["development_only"] is True
    assert short_report["scientific_promotion_allowed"] is False
    assert short_report["accepted_scientific_evidence"] is False
    assert short_report["acceptance_status"] == "not-assessed"
    assert cast(dict[str, object], short_report["comparison_contract"])[
        "persistent_state_shape_matched"
    ] is True
    assert short_report["claim_assessments"] == CLAIM_ASSESSMENTS
    assert all(
        claim["status"] == "not-assessed"
        for claim in cast(dict[str, dict[str, str]], short_report["claim_assessments"]).values()
    )
    encoded = prototype_two_learning_agent_recurrence_report_json(short_report)
    assert json.loads(encoded) == short_report


def test_every_event_uses_four_joint_proposals_and_two_atomic_learners(
    short_report: dict[str, object],
) -> None:
    for run in _runs(short_report):
        assert len(run["trace"]) == 3
        for index, event in enumerate(run["trace"]):
            assert event["environment_pre_words"] == [0, index]
            assert event["environment_post_words"] == [0, index + 1]
            assert event["environment_source_state_preserved"] is True
            assert event["joint_transaction_committed"] is True
            joint = event["joint_dispatch"]
            assert set(joint["primitive_actions"]) == {
                "actual_actual",
                "base0_actual1",
                "actual0_base1",
                "base_base",
            }
            rewards = joint["rewards"]
            effects = joint["effects"]
            assert effects["agent0_unilateral"] == (
                rewards["actual_actual"] - rewards["base0_actual1"]
            )
            assert effects["agent1_unilateral"] == (
                rewards["actual_actual"] - rewards["actual0_base1"]
            )
            assert effects["joint"] == (
                rewards["actual_actual"] - rewards["base_base"]
            )
            assert len(event["agents"]) == 2
            for agent_index, agent in enumerate(event["agents"]):
                assert agent["agent_index"] == agent_index
                assert agent["prototype_pre_step_words"] == [0, index]
                assert agent["prototype_post_step_words"] == [0, index + 1]
                assert agent["source_state_preserved"] is True
                assert agent["preview_state_discarded"] is True
                assert agent["candidate_accepted"] is True
                assert agent["world_model"]["partner_action_observed"] is False
                assert agent["world_model"]["discount_target"] == 1.0
                assert agent["prototype_reported_horde_td_error"] == pytest.approx(
                    [
                        cumulant - prediction
                        for cumulant, prediction in zip(
                            agent["horde_cumulant"],
                            agent["horde_prediction"],
                            strict=True,
                        )
                    ]
                )


def test_work_resources_stale_replay_and_checkpoint_identity_are_exact(
    short_report: dict[str, object],
) -> None:
    for run in _runs(short_report):
        work = run["work"]
        assert work["environment_proposal_calls"] == 12
        assert work["counterfactual_environment_proposal_calls"] == 9
        assert work["discarded_preview_update_calls"] == 6
        assert work["committed_candidate_update_calls"] == 6
        assert work["regular_prototype_update_calls"] == 12
        assert work["stale_identity_probe_update_calls"] == 2
        assert work["world_model_update_calls"] == 12
        assert work["world_model_carried_updates"] == 6
        assert work["explicit_horde_prediction_calls"] == 6
        assert work["managed_horde_update_calls"] == 12
        assert work["regular_memory_sidecars_supplied"] == 6
        assert work["stale_probe_memory_sidecars_supplied"] == 0
        assert work["total_memory_sidecars_supplied"] == 6
        assert work["checkpoint_shadow_object_save_calls"] == 12
        assert work["checkpoint_shadow_object_load_calls"] == 12

        resources = run["resources"]
        assert len(resources["per_agent"]) == 2
        assert resources["combined"]["initial_total_nbytes"] == (
            resources["combined"]["final_total_nbytes"]
        )
        assert len(resources["combined"]["phase_boundary_total_nbytes"]) == 4
        for entry in resources["per_agent"]:
            world_model = entry["declaration"]["stable_base_world_model"]
            assert world_model["coordinates"] == "stable_base_only"
            assert world_model["observation_dim"] == 8

        stale = run["stale_replay_audit"]
        assert stale["environment_unchanged"] is True
        assert stale["probe_isolated_to_decision_id"] is True
        assert stale["replay_update_calls"] == 2
        assert stale["memory_sidecars_supplied"] == 0
        assert all(result["stale_decision_rejected"] for result in stale["agent_results"])
        assert all(not result["decision_id_matches"] for result in stale["agent_results"])
        assert all(
            not result["update_decision_id_matches"]
            for result in stale["agent_results"]
        )
        assert all(
            result["update_reported_rejected"] for result in stale["agent_results"]
        )
        assert all(result["observation_matches"] for result in stale["agent_results"])
        assert all(result["action_matches"] for result in stale["agent_results"])
        assert all(
            result["nonidentity_prechecks_valid"] for result in stale["agent_results"]
        )
        assert all(result["state_bit_exact"] for result in stale["agent_results"])

        audits = run["checkpoint_shadow_audits"]
        assert [audit["label"] for audit in audits] == ["initial", "A1", "B", "A2"]
        assert [audit["event_count"] for audit in audits] == [0, 1, 2, 3]
        assert all(audit["environment_round_trip_bit_exact"] for audit in audits)
        assert all(audit["agent_state_round_trip_bit_exact"] == [True, True] for audit in audits)
        assert all(
            audit["environment_state_witness"]
            == audit["restored_environment_state_witness"]
            for audit in audits
        )
        assert all(
            audit["agent_state_witnesses"] == audit["restored_agent_state_witnesses"]
            for audit in audits
        )
        assert all(len(audit["agent_current_actions"]) == 2 for audit in audits)
        assert all(audit["checkpoint_state_carried"] is False for audit in audits)
        assert run["temporary_checkpoint_storage_retained"] is False
        assert run["work"]["total_memory_sidecars_supplied"] == 6


def test_metrics_cover_each_agent_and_joint_effects_without_transfer_claims(
    short_report: dict[str, object],
) -> None:
    for run in _runs(short_report):
        metrics = run["metrics"]
        assert len(metrics["per_agent"]) == 2
        assert set(metrics["phase_joint_reward_effects"]) == {"A1", "B", "A2"}
        assert metrics["standard_forward_transfer_assessed"] is False
        assert metrics["partner_learning_uplift_assessed"] is False
        for agent in metrics["per_agent"]:
            assert set(agent["phase_reward"]) == {"A1", "B", "A2"}
            assert set(agent["phase_horde_mse"]) == {"A1", "B", "A2"}
            assert set(agent["phase_world_prediction_error"]) == {"A1", "B", "A2"}
            assert set(agent["phase_features"]) == {"A1", "B", "A2"}
            assert set(agent["phase_memory"]) == {"A1", "B", "A2"}
            assert "a2_entry_minus_a1_tail_reward" in agent["recurrence"]


def test_readout_blocked_arm_preserves_both_preview_actions(
    short_report: dict[str, object],
) -> None:
    blocked = next(
        run for run in _runs(short_report) if run["arm"] == "memory_readout_blocked"
    )

    for event in blocked["trace"]:
        for agent in event["agents"]:
            assert agent["memory_action_changed"] is False
            assert agent["next_committed_action"] == agent["next_preview_action"]
            assert agent["memory_wrote"] is True


@pytest.mark.parametrize(
    "mutation",
    [
        "promotion",
        "reward_effect",
        "agent_clock",
        "checkpoint_identity",
        "work",
    ],
)
def test_validator_rejects_claim_transaction_checkpoint_and_work_tampering(
    short_report: dict[str, object],
    mutation: str,
) -> None:
    corrupted = copy.deepcopy(short_report)
    first = _runs(corrupted)[0]
    if mutation == "promotion":
        corrupted["scientific_promotion_allowed"] = True
    elif mutation == "reward_effect":
        first["trace"][0]["joint_dispatch"]["effects"]["joint"] += 0.25
    elif mutation == "agent_clock":
        first["trace"][1]["agents"][0]["prototype_post_step_words"] = [0, 99]
    elif mutation == "checkpoint_identity":
        first["checkpoint_shadow_audits"][1]["harness_base_actions"][0] ^= 1
    else:
        first["work"]["environment_proposal_calls"] = 0

    validation = validate_prototype_two_learning_agent_recurrence_report(corrupted)
    assert not validation.valid
    with pytest.raises(ValueError, match="invalid two-learner recurrence report"):
        prototype_two_learning_agent_recurrence_report_json(corrupted)


@pytest.mark.parametrize(
    "mutation",
    [
        "fabricated_feature_count",
        "fabricated_counterfactual_reward",
        "fabricated_horde_cumulant",
        "checkpoint_boolean_integer_alias",
        "stale_identity_relabel",
    ],
)
def test_validator_rejects_self_consistent_fabrication_and_type_aliases(
    short_report: dict[str, object],
    mutation: str,
) -> None:
    corrupted = copy.deepcopy(short_report)
    first = _runs(corrupted)[0]
    if mutation == "fabricated_feature_count":
        first["trace"][0]["agents"][0]["a_critical_pair_count"] = 999
    elif mutation == "fabricated_counterfactual_reward":
        joint = first["trace"][0]["joint_dispatch"]
        joint["rewards"]["base_base"] += 0.25
        joint["effects"]["joint"] = (
            joint["rewards"]["actual_actual"] - joint["rewards"]["base_base"]
        )
        joint["effects"]["interaction"] = (
            joint["rewards"]["actual_actual"]
            - joint["rewards"]["base0_actual1"]
            - joint["rewards"]["actual0_base1"]
            + joint["rewards"]["base_base"]
        )
    elif mutation == "fabricated_horde_cumulant":
        agent = first["trace"][0]["agents"][0]
        agent["horde_cumulant"][0] += 0.25
        td_error = agent["horde_cumulant"][0] - agent["horde_prediction"][0]
        agent["prototype_reported_horde_td_error"][0] = td_error
        agent["horde_squared_error"][0] = td_error**2
    elif mutation == "checkpoint_boolean_integer_alias":
        audit = first["checkpoint_shadow_audits"][0]
        audit["agent_config_round_trip_exact"] = [1, 1]
        audit["agent_state_round_trip_bit_exact"] = [1, 1]
    else:
        first["stale_replay_audit"]["agent_results"][0][
            "decision_id_matches"
        ] = True
    _refresh_derived_payloads(corrupted)

    validation = validate_prototype_two_learning_agent_recurrence_report(corrupted)
    assert not validation.valid
