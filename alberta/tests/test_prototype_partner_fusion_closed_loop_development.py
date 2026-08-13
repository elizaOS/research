"""Adversarial contract for the causal closed-loop Prototype partner lane."""

from __future__ import annotations

import copy
import dataclasses
import json
import math
from typing import Any

import pytest

import alberta_framework.core.multi_head_learner as multi_head_learner
from alberta_framework.core.prototype_agent import PrototypeAgent
from alberta_framework.evaluation import (
    prototype_partner_fusion_closed_loop_development as lane,
)

pytestmark = [pytest.mark.development, pytest.mark.slow]


def _arm(state: lane.ClosedLoopRunState, condition: lane.Condition) -> lane.ClosedLoopArmState:
    return next(arm for arm in state.arms if arm.condition == condition)


def test_protocol_is_frozen_consumed_l0_and_has_no_promotion_surface() -> None:
    assert lane.ASSESSMENT == "not_assessed"
    assert lane.EVIDENCE_LEVEL.startswith("L0_")
    assert lane.DEVELOPMENT_ONLY
    assert lane.DEVELOPMENT_PROTOCOL_CONSUMED
    assert not lane.SCIENTIFIC_PROMOTION_ALLOWED
    assert not lane.OUTPUT_WRITES_ALLOWED
    assert not lane.THRESHOLDS_DEFINED
    assert not lane.WINNER_DECLARED
    assert lane.CONDITIONS == (
        lane.LEARNED_FUSION,
        lane.OUTCOME_BLIND_FUSION,
        lane.BASE_ONLY,
    )
    assert lane.CONFIG.horizon == 12
    assert lane.CONFIG.reversal_event == 6
    assert len(lane.EXOGENOUS_SCHEDULE) == lane.CONFIG.horizon + 1
    assert all("write" not in name for name in lane.__all__)
    assert all("promote" not in name for name in lane.__all__)


def test_host_timing_metadata_cannot_change_exact_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class AdvancingClock:
        def __init__(self) -> None:
            self.value = 1_700_000_000.0

        def time(self) -> float:
            self.value += 256.0
            return self.value

    monkeypatch.setattr(multi_head_learner, "time", AdvancingClock())
    evaluator = lane.PrototypePartnerFusionClosedLoopDevelopmentEvaluator()
    state = evaluator.initial_state()
    assert evaluator.validate_state(state)
    state = evaluator.step(state)
    assert evaluator.validate_state(state)
    for arm in state.arms:
        learner = arm.prototype_state.oak_state.stomp_state.base_learner_state
        assert float(learner.birth_timestamp) == 0.0
        assert float(learner.uptime_s) == 0.0

    learned = _arm(state, lane.LEARNED_FUSION)
    oak_state = learned.prototype_state.oak_state
    stomp_state = oak_state.stomp_state
    learner_state = stomp_state.base_learner_state.replace(birth_timestamp=1.0)
    prototype_state = learned.prototype_state.replace(
        oak_state=oak_state.replace(
            stomp_state=stomp_state.replace(base_learner_state=learner_state)
        )
    )
    tampered_arm = lane._seal_arm(
        dataclasses.replace(learned, prototype_state=prototype_state, state_seal="")
    )
    tampered = lane._seal_run(
        dataclasses.replace(
            state,
            arms=(tampered_arm, *state.arms[1:]),
            run_seal="",
        )
    )
    assert not evaluator.validate_state(tampered, reconstruct=False)


@pytest.mark.parametrize(
    "change",
    [
        {"horizon": 11},
        {"reversal_event": 5},
        {"declared_confidence": 0.8},
        {"base_partner_costs": (0.1, 0.1)},
        {"cost_spike": 0.7},
    ],
)
def test_consumed_protocol_rejects_retuning(change: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="frozen"):
        lane.ClosedLoopPartnerFusionConfig(**change)


def test_schedule_pairs_only_exogenous_values_and_exercises_failures() -> None:
    schedule = lane.build_closed_loop_exogenous_schedule()
    assert schedule == lane.EXOGENOUS_SCHEDULE
    execution = schedule[: lane.CONFIG.horizon]
    assert any(not event.partner_available[0] for event in execution)
    assert any(not event.partner_available[1] for event in execution)
    assert sum(not any(event.partner_available) for event in execution) == 2
    assert any(
        max(event.communication_cost) == pytest.approx(lane.CONFIG.cost_spike)
        for event in execution
    )
    assert any(not all(event.hard_action_mask) for event in execution)
    assert {event.after_reversal for event in execution} == {False, True}
    for context_bit in (0, 1):
        before = next(
            event
            for event in execution
            if event.context_bit == context_bit and not event.after_reversal
        )
        after = next(
            event
            for event in execution
            if event.context_bit == context_bit and event.after_reversal
        )
        assert before.context_bit == after.context_bit


def test_real_prototype_states_and_owner_receipts_are_independent() -> None:
    evaluator = lane.PrototypePartnerFusionClosedLoopDevelopmentEvaluator()
    assert isinstance(evaluator.agent, PrototypeAgent)
    assert evaluator.agent.partner_policy_fusion is not None
    state = evaluator.initial_state()
    assert evaluator.validate_state(state)
    assert len({id(arm.prototype_state) for arm in state.arms}) == len(lane.CONDITIONS)
    assert len({arm.evaluator_owner_digest for arm in state.arms}) == len(lane.CONDITIONS)
    assert len({arm.prototype_owner_digest for arm in state.arms}) == len(lane.CONDITIONS)
    assert len({arm.fusion_owner_digest for arm in state.arms}) == len(lane.CONDITIONS)
    assert len({arm.environment.owner_digest for arm in state.arms}) == len(lane.CONDITIONS)
    for arm in state.arms:
        receipt = arm.pending_decision
        assert receipt.condition == arm.condition
        assert receipt.prototype_decision_id == tuple(
            int(value) for value in arm.prototype_state.current_decision_id
        )
        assert receipt.effective_action == int(arm.prototype_state.current_action)
        assert receipt.hard_mask_receipt.owner_digest not in {
            arm.evaluator_owner_digest,
            arm.prototype_owner_digest,
            arm.fusion_owner_digest,
            arm.environment.owner_digest,
        }


def test_receipts_reject_stale_and_cross_owner_use_atomically() -> None:
    evaluator = lane.PrototypePartnerFusionClosedLoopDevelopmentEvaluator()
    initial = evaluator.initial_state()
    advanced = evaluator.step(initial)
    original_seal = advanced.run_seal
    stale = _arm(initial, lane.LEARNED_FUSION).pending_decision
    with pytest.raises(ValueError, match="exact pending receipt"):
        evaluator.step(
            advanced,
            receipt_overrides={lane.LEARNED_FUSION: stale},
        )
    cross_owner = _arm(advanced, lane.OUTCOME_BLIND_FUSION).pending_decision
    with pytest.raises(ValueError, match="exact pending receipt"):
        evaluator.step(
            advanced,
            receipt_overrides={lane.LEARNED_FUSION: cross_owner},
        )
    assert advanced.run_seal == original_seal
    assert evaluator.validate_state(advanced)


def test_numeric_fail_stop_and_hash_only_reseal_cannot_forge_causal_state() -> None:
    evaluator = lane.PrototypePartnerFusionClosedLoopDevelopmentEvaluator()
    state = evaluator.reconstruct(2)
    learned = _arm(state, lane.LEARNED_FUSION)

    nonfinite_environment = dataclasses.replace(learned.environment, last_net_reward=float("nan"))
    nonfinite_arm = dataclasses.replace(learned, environment=nonfinite_environment)
    nonfinite_state = dataclasses.replace(
        state,
        arms=(nonfinite_arm, *state.arms[1:]),
    )
    assert not evaluator.validate_state(nonfinite_state)
    with pytest.raises(ValueError, match="invalid or noncausal"):
        evaluator.step(nonfinite_state)
    assert math.isfinite(learned.environment.last_net_reward)

    forged_environment = dataclasses.replace(
        learned.environment,
        last_net_reward=learned.environment.last_net_reward + 0.125,
    )
    forged_pending = dataclasses.replace(
        learned.pending_decision,
        decision_environment_digest=lane._digest(forged_environment.to_dict()),
        receipt_digest="",
    )
    forged_pending = lane._seal_pending(forged_pending)
    forged_arm = lane._seal_arm(
        dataclasses.replace(
            learned,
            environment=forged_environment,
            pending_decision=forged_pending,
            state_seal="",
        )
    )
    forged_state = lane._seal_run(
        dataclasses.replace(
            state,
            arms=(forged_arm, *state.arms[1:]),
            run_seal="",
        )
    )
    assert evaluator.validate_state(forged_state, reconstruct=False)
    assert not evaluator.validate_state(forged_state)


@pytest.fixture(scope="module")
def report() -> dict[str, object]:
    return lane.run_prototype_partner_fusion_closed_loop_development()


@pytest.mark.integration
def test_closed_loop_traces_are_causal_isolated_and_feedback_is_outcome_bound(
    report: dict[str, object],
) -> None:
    traces = report["traces"]
    assert isinstance(traces, dict)
    learned = traces[lane.LEARNED_FUSION]
    blinded = traces[lane.OUTCOME_BLIND_FUSION]
    base = traces[lane.BASE_ONLY]
    assert isinstance(learned, list)
    assert isinstance(blinded, list)
    assert isinstance(base, list)
    expected_events = [event.to_dict() for event in lane.EXOGENOUS_SCHEDULE[: lane.CONFIG.horizon]]
    for condition in lane.CONDITIONS:
        trace = traces[condition]
        assert isinstance(trace, list)
        assert [record["exogenous_event"] for record in trace] == expected_events
        assert len(trace) == lane.CONFIG.horizon
        for clock, record in enumerate(trace):
            assert record["condition"] == condition
            assert record["execution_clock"] == clock
            assert record["environment_before"]["owner_digest"] != ""
            assert record["pending_receipt"]["condition"] == condition
            assert record["transition_valid"]
            assert record["next_decision_applied"]
            assert record["next_action_allowed_by_caller"]

    assert [record["executed_action"] for record in learned] != [
        record["executed_action"] for record in base
    ]
    assert any(
        learned[index]["environment_after"] != base[index]["environment_after"]
        for index in range(lane.CONFIG.horizon)
    )
    assert any(
        learned[index]["next_message_suggestions"] != blinded[index]["next_message_suggestions"]
        for index in range(lane.CONFIG.horizon)
    )

    for record in learned:
        pending = record["pending_receipt"]
        if pending["feedback_armed"]:
            assert record["feedback_applied"]
            assert record["feedback_target_kind"] == "own_realized_assistance"
            assert record["feedback_target_supplied"] == pytest.approx(
                record["realized_assistance_value"]
            )
    for record in blinded:
        pending = record["pending_receipt"]
        if pending["feedback_armed"]:
            assert record["feedback_applied"]
            assert record["feedback_target_kind"] == "fixed_zero_outcome_blind"
            assert record["feedback_target_supplied"] == 0.0
    assert all(not record["feedback_applied"] for record in base)
    assert all(record["next_message_available"] == (False, False) for record in base)


def test_reversal_disconnect_query_cost_and_hard_mask_boundaries_are_raw(
    report: dict[str, object],
) -> None:
    traces = report["traces"]
    summaries = report["summaries"]
    owners = report["arm_owner_receipts"]
    assert isinstance(traces, dict)
    assert isinstance(summaries, dict)
    assert isinstance(owners, dict)
    learned = traces[lane.LEARNED_FUSION]
    assert isinstance(learned, list)
    for context_bit in (0, 1):
        before = next(
            record
            for record in learned
            if record["exogenous_event"]["context_bit"] == context_bit
            and not record["exogenous_event"]["after_reversal"]
        )
        after = next(
            record
            for record in learned
            if record["exogenous_event"]["context_bit"] == context_bit
            and record["exogenous_event"]["after_reversal"]
        )
        assert before["hidden_reliable_partner"] != after["hidden_reliable_partner"]
    assert sum(not any(record["exogenous_event"]["partner_available"]) for record in learned) == 2
    assert any(
        max(record["exogenous_event"]["communication_cost"])
        == pytest.approx(lane.CONFIG.cost_spike)
        for record in learned
    )
    assert summaries[lane.LEARNED_FUSION]["query_route_count"] > 0
    assert any(record["charged_communication_cost"] > 0.0 for record in learned)
    assert all(record["charged_communication_cost"] == 0.0 for record in traces[lane.BASE_ONLY])
    assert any(
        not all(record["next_hard_mask_receipt"]["exogenous_candidate_mask"]) for record in learned
    )
    assert any(not all(record["next_hard_mask_receipt"]["mask"]) for record in learned)
    for condition in lane.CONDITIONS:
        condition_owners = owners[condition]
        assert condition_owners["hard_mask_owner_digest"] not in {
            condition_owners["fusion_owner_digest"],
            condition_owners["prototype_owner_digest"],
            condition_owners["environment_owner_digest"],
        }
        for record in traces[condition]:
            receipt = record["next_hard_mask_receipt"]
            action = record["next_effective_action"]
            base_action = record["next_counterfactual_base_action"]
            assert receipt["mask"][action]
            assert receipt["mask"][base_action]


def test_raw_hash_chain_report_replay_and_resealed_tamper_rejection(
    report: dict[str, object],
) -> None:
    assert lane.validate_prototype_partner_fusion_closed_loop_report(report) == ()
    encoded = json.dumps(report, allow_nan=False, sort_keys=True)
    assert "not_assessed" in encoded
    payload_digest = report["deterministic_payload_digest"]
    assert isinstance(payload_digest, str)
    assert payload_digest in encoded

    forged = copy.deepcopy(report)
    traces = forged["traces"]
    assert isinstance(traces, dict)
    learned = traces[lane.LEARNED_FUSION]
    assert isinstance(learned, list)
    learned[0]["net_reward"] = float(learned[0]["net_reward"]) + 0.25
    unsigned = dict(forged)
    unsigned.pop("deterministic_payload_digest")
    forged["deterministic_payload_digest"] = lane._digest(unsigned)
    errors = lane.validate_prototype_partner_fusion_closed_loop_report(forged)
    assert "report differs from exact causal replay" in errors

    promoted = dict(report)
    promoted["assessment"] = "accepted"
    errors = lane.validate_prototype_partner_fusion_closed_loop_report(promoted)
    assert "assessment must remain not_assessed" in errors


def test_matched_logical_budgets_and_real_kernel_parity_are_explicit(
    report: dict[str, object],
) -> None:
    assert report["prototype_eager_jit_parity"] is True
    resource = report["resource_report"]
    assert isinstance(resource, dict)
    assert resource["matched_logical_budgets"] is True
    assert resource["paired_randomness_scope"] == "frozen_exogenous_values_only"
    assert resource["learner_rng_states_are_independent"] is True
    per_condition = resource["per_condition"]
    assert isinstance(per_condition, dict)
    budgets = list(per_condition.values())
    assert all(budget == budgets[0] for budget in budgets[1:])
    for budget in budgets:
        assert budget["committed_prototype_transitions"] == lane.CONFIG.horizon
        assert budget["discarded_base_action_previews"] == lane.CONFIG.horizon
        assert budget["prototype_update_calls"] == 2 * lane.CONFIG.horizon
        assert budget["fusion_decision_calls"] == 2 * lane.CONFIG.horizon
        assert budget["fusion_feedback_calls"] == 2 * lane.CONFIG.horizon
        assert budget["fixed_message_slots_per_call"] == lane.MAX_PARTNERS
        assert budget["shared_mutable_agent_state"] is False
        assert budget["shared_mutable_environment_state"] is False


@pytest.mark.integration
def test_in_memory_checkpoint_resume_is_exact_and_tamper_fails_closed(
    report: dict[str, object],
) -> None:
    checkpoint = lane.make_prototype_partner_fusion_closed_loop_checkpoint(5)
    assert lane._validate_checkpoint(checkpoint) == ()
    resumed = lane.resume_prototype_partner_fusion_closed_loop_checkpoint(checkpoint)
    assert resumed == report

    stale_clock = dataclasses.replace(
        checkpoint,
        next_event=6,
        checkpoint_digest="",
    )
    stale_clock = dataclasses.replace(
        stale_clock,
        checkpoint_digest=lane._digest(stale_clock.metadata()),
    )
    errors = lane._validate_checkpoint(stale_clock)
    assert "checkpoint state clock mismatch" in errors

    learned = _arm(checkpoint.state, lane.LEARNED_FUSION)
    blinded = _arm(checkpoint.state, lane.OUTCOME_BLIND_FUSION)
    crossed_arm = dataclasses.replace(
        learned,
        pending_decision=blinded.pending_decision,
    )
    crossed_state = dataclasses.replace(
        checkpoint.state,
        arms=(crossed_arm, *checkpoint.state.arms[1:]),
    )
    crossed = dataclasses.replace(
        checkpoint,
        state=crossed_state,
        state_digest=lane._digest(
            {**lane._run_body(crossed_state), "run_seal": crossed_state.run_seal}
        ),
        checkpoint_digest="",
    )
    crossed = dataclasses.replace(
        crossed,
        checkpoint_digest=lane._digest(crossed.metadata()),
    )
    with pytest.raises(ValueError, match="exact causal prefix"):
        lane.resume_prototype_partner_fusion_closed_loop_checkpoint(crossed)
