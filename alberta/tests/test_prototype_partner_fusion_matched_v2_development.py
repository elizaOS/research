"""Adversarial contract for the matched-initialization partner-fusion v2 lane."""

from __future__ import annotations

import copy
import dataclasses
import json
import math
from typing import Any

import numpy as np
import pytest

from alberta_framework.core.prototype_agent import PrototypeAgent, PrototypeInteractionState
from alberta_framework.evaluation import (
    prototype_partner_fusion_matched_v2_development as lane,
)

pytestmark = [pytest.mark.development, pytest.mark.slow]


def _arm(state: lane.MatchedV2RunState, condition: lane.Condition) -> lane.MatchedV2ArmState:
    return next(arm for arm in state.arms if arm.condition == condition)


def test_protocol_is_permanently_nonpromoting_l0_and_separate_from_v1() -> None:
    assert lane.SCHEMA.endswith(".v2")
    assert "matched-initialization" in lane.SCHEMA
    assert "matched-initialization" in lane.PROTOCOL_NAMESPACE
    assert lane.ASSESSMENT == "not_assessed"
    assert lane.EVIDENCE_LEVEL.startswith("L0_")
    assert lane.DEVELOPMENT_ONLY
    assert lane.DEVELOPMENT_PROTOCOL_CONSUMED
    assert not lane.SCIENTIFIC_PROMOTION_ALLOWED
    assert not lane.OUTPUT_WRITES_ALLOWED
    assert not lane.THRESHOLDS_DEFINED
    assert not lane.WINNER_DECLARED
    assert lane.CONDITIONS == (
        lane.LEARNED_FEEDBACK,
        lane.FIXED_ZERO_FEEDBACK,
        lane.EMPTY_MESSAGE_BASE_ONLY,
    )
    assert all("write" not in name for name in lane.__all__)
    assert all("promote" not in name for name in lane.__all__)


@pytest.mark.parametrize(
    "change",
    [
        {"horizon": 11},
        {"reversal_event": 5},
        {"initialization_seed": 24_603},
        {"declared_confidence": 0.8},
        {"base_partner_costs": (0.1, 0.1)},
        {"cost_spike": 0.7},
    ],
)
def test_consumed_v2_protocol_rejects_retuning(change: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="frozen"):
        lane.MatchedV2Config(**change)


def test_initialization_is_bit_identical_but_ownership_is_independent() -> None:
    evaluator = lane.PrototypePartnerFusionMatchedV2DevelopmentEvaluator()
    assert isinstance(evaluator.agent, PrototypeAgent)
    state = evaluator.initial_state()
    assert evaluator.validate_state(state)

    prototype_digests = [lane._tree_digest(arm.prototype_state) for arm in state.arms]
    environment_digests = [lane._digest(arm.environment.to_dict()) for arm in state.arms]
    fusion_digests = []
    for arm in state.arms:
        wrapper = arm.prototype_state.ia_state
        assert isinstance(wrapper, PrototypeInteractionState)
        fusion_digests.append(lane._tree_digest(wrapper.partner_policy_fusion_state))
    assert len(set(prototype_digests)) == 1
    assert len(set(fusion_digests)) == 1
    assert len(set(environment_digests)) == 1
    assert len({arm.initialization_key_digest for arm in state.arms}) == 1

    assert len({id(arm.prototype_state) for arm in state.arms}) == len(lane.CONDITIONS)
    assert len({arm.evaluator_owner_digest for arm in state.arms}) == len(lane.CONDITIONS)
    assert len({arm.prototype_owner_digest for arm in state.arms}) == len(lane.CONDITIONS)
    assert len({arm.fusion_owner_digest for arm in state.arms}) == len(lane.CONDITIONS)
    assert len({arm.environment_owner_digest for arm in state.arms}) == len(lane.CONDITIONS)
    assert len({arm.pending_decision.hard_mask_receipt.owner_digest for arm in state.arms}) == len(
        lane.CONDITIONS
    )

    receipt = evaluator.initialization_receipt(state)
    assert receipt["typed_rng_keys_bit_identical"] is True
    assert receipt["prototype_states_bit_identical"] is True
    assert receipt["fusion_states_bit_identical"] is True
    assert receipt["environment_states_bit_identical"] is True
    typed_key = receipt["typed_rng_key"]
    assert isinstance(typed_key, dict)
    assert str(typed_key["dtype"]).startswith("key<")
    assert typed_key["shape"] == []


def test_schedule_pairs_exogenous_values_and_declares_post_divergence_scope() -> None:
    schedule = lane.build_matched_v2_exogenous_schedule()
    assert schedule == lane.EXOGENOUS_SCHEDULE
    execution = schedule[: lane.CONFIG.horizon]
    assert len(schedule) == lane.CONFIG.horizon + 1
    assert any(not event.partner_available[0] for event in execution)
    assert any(not event.partner_available[1] for event in execution)
    assert any(
        max(event.communication_cost) == pytest.approx(lane.CONFIG.cost_spike)
        for event in execution
    )
    assert {event.after_reversal for event in execution} == {False, True}
    assert all(event.hard_action_mask == (True, True, True) for event in execution)
    assert lane.PAIRING_SCOPE == (
        "bit-identical initialization plus paired exogenous context/noise/drift/"
        "availability/cost/hard-mask schedule"
    )
    assert lane.IDENTICAL_POST_DIVERGENCE_INPUTS is False


def test_source_and_runtime_manifests_bind_the_executed_local_kernel() -> None:
    source_manifest = lane._source_manifest()
    assert {
        "alberta_framework/core/initializers.py",
        "alberta_framework/core/multi_head_learner.py",
        "alberta_framework/core/normalizers.py",
        "alberta_framework/core/oak.py",
        "alberta_framework/core/optimizers.py",
        "alberta_framework/core/options.py",
        "alberta_framework/core/partner_policy_fusion.py",
        "alberta_framework/core/prototype_agent.py",
        "alberta_framework/core/state_builder.py",
        "alberta_framework/core/types.py",
        "alberta_framework/evaluation/prototype_partner_fusion_closed_loop_development.py",
        "alberta_framework/evaluation/prototype_partner_fusion_matched_v2_development.py",
    }.issubset(source_manifest)
    runtime_manifest = lane._runtime_manifest()
    assert runtime_manifest["numpy"] == str(lane.np.__version__)


def test_receipts_reject_stale_cross_owner_and_mask_tamper_atomically() -> None:
    evaluator = lane.PrototypePartnerFusionMatchedV2DevelopmentEvaluator()
    initial = evaluator.initial_state()
    advanced = evaluator.step(initial)
    original_seal = advanced.run_seal

    stale = _arm(initial, lane.LEARNED_FEEDBACK).pending_decision
    with pytest.raises(ValueError, match="exact pending receipt"):
        evaluator.step(advanced, receipt_overrides={lane.LEARNED_FEEDBACK: stale})
    crossed = _arm(advanced, lane.FIXED_ZERO_FEEDBACK).pending_decision
    with pytest.raises(ValueError, match="exact pending receipt"):
        evaluator.step(advanced, receipt_overrides={lane.LEARNED_FEEDBACK: crossed})

    learned = _arm(advanced, lane.LEARNED_FEEDBACK)
    receipt = learned.pending_decision.hard_mask_receipt
    forged_mask = dataclasses.replace(
        receipt,
        mask=(True, False, True),
        receipt_digest="",
    )
    forged_mask = dataclasses.replace(
        forged_mask,
        receipt_digest=lane._digest(forged_mask.body()),
    )
    forged_pending = dataclasses.replace(
        learned.pending_decision,
        hard_mask_receipt=forged_mask,
        receipt_digest="",
    )
    forged_pending = lane._seal_pending(forged_pending)
    forged_arm = lane._seal_arm(
        dataclasses.replace(learned, pending_decision=forged_pending, state_seal="")
    )
    forged = lane._seal_run(
        dataclasses.replace(advanced, arms=(forged_arm, *advanced.arms[1:]), run_seal="")
    )
    assert not evaluator.validate_state(forged, reconstruct=False)
    assert advanced.run_seal == original_seal
    assert evaluator.validate_state(advanced)


def test_nonfinite_and_hash_only_resealed_state_fail_closed() -> None:
    evaluator = lane.PrototypePartnerFusionMatchedV2DevelopmentEvaluator()
    state = evaluator.reconstruct(2)
    learned = _arm(state, lane.LEARNED_FEEDBACK)

    bad_environment = dataclasses.replace(learned.environment, last_net_reward=float("nan"))
    bad_arm = dataclasses.replace(learned, environment=bad_environment)
    bad_state = dataclasses.replace(state, arms=(bad_arm, *state.arms[1:]))
    assert not evaluator.validate_state(bad_state)
    with pytest.raises(ValueError, match="invalid or noncausal"):
        evaluator.step(bad_state)
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
    forged = lane._seal_run(
        dataclasses.replace(state, arms=(forged_arm, *state.arms[1:]), run_seal="")
    )
    assert evaluator.validate_state(forged, reconstruct=False)
    assert not evaluator.validate_state(forged)


@pytest.fixture(scope="module")
def report() -> dict[str, object]:
    return lane.run_prototype_partner_fusion_matched_v2_development()


@pytest.mark.integration
def test_traces_pair_exogenous_inputs_then_diverge_only_causally(
    report: dict[str, object],
) -> None:
    traces = report["traces"]
    assert isinstance(traces, dict)
    learned = traces[lane.LEARNED_FEEDBACK]
    fixed_zero = traces[lane.FIXED_ZERO_FEEDBACK]
    base = traces[lane.EMPTY_MESSAGE_BASE_ONLY]
    assert isinstance(learned, list)
    assert isinstance(fixed_zero, list)
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
            assert record["pending_receipt"]["condition"] == condition
            assert record["transition_valid"]
            assert record["next_decision_applied"]
            assert record["next_action_allowed_by_caller"]
            assert record["next_hard_mask_receipt"]["mask"] == (True, True, True)

    # The two message-consuming arms receive byte-identical first messages;
    # later messages may differ only because their executed histories differ.
    assert learned[0]["next_message_batch_digest"] == fixed_zero[0]["next_message_batch_digest"]
    assert learned[0]["next_message_provenance"] == fixed_zero[0]["next_message_provenance"]
    assert all(record["next_message_available"] == (False, False) for record in base)
    assert any(
        learned[index]["environment_after"] != base[index]["environment_after"]
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
    for record in fixed_zero:
        pending = record["pending_receipt"]
        if pending["feedback_armed"]:
            assert record["feedback_applied"]
            assert record["feedback_target_kind"] == "fixed_zero_outcome_blind"
            assert record["feedback_target_supplied"] == 0.0
    assert any(
        record["pending_receipt"]["feedback_armed"]
        and abs(float(record["realized_assistance_value"])) > 0.0
        for record in learned
    )
    assert all(not record["feedback_applied"] for record in base)


def test_descriptive_summary_reports_action_changes_return_and_cost_without_a_winner(
    report: dict[str, object],
) -> None:
    assert report["assessment"] == "not_assessed"
    assert report["identical_post_divergence_inputs"] is False
    claim_scope = report["claim_scope"]
    assert isinstance(claim_scope, str)
    assert "threshold" not in claim_scope
    summaries = report["summaries"]
    traces = report["traces"]
    assert isinstance(summaries, dict)
    assert isinstance(traces, dict)
    assert summaries[lane.LEARNED_FEEDBACK]["action_changing_assistance_count"] > 0
    assert summaries[lane.FIXED_ZERO_FEEDBACK]["action_changing_assistance_count"] > 0
    assert summaries[lane.EMPTY_MESSAGE_BASE_ONLY]["action_changing_assistance_count"] == 0
    for condition in lane.CONDITIONS:
        summary = summaries[condition]
        trace = traces[condition]
        assert summary["action_changing_assistance_count"] == sum(
            bool(record["action_changed_by_assistance"]) for record in trace
        )
        assert summary["realized_task_return"] == pytest.approx(
            sum(float(record["task_reward"]) for record in trace)
        )
        assert summary["realized_net_return"] == pytest.approx(
            sum(float(record["net_reward"]) for record in trace)
        )
        assert summary["realized_communication_cost"] == pytest.approx(
            sum(float(record["charged_communication_cost"]) for record in trace)
        )
    assert report["thresholds_defined"] is False
    assert report["winner_declared"] is False


def test_raw_hash_chain_exact_replay_and_resealed_report_tamper_rejection(
    report: dict[str, object],
) -> None:
    assert lane.validate_prototype_partner_fusion_matched_v2_report(report) == ()
    encoded = json.dumps(report, allow_nan=False, sort_keys=True)
    assert "not_assessed" in encoded
    traces = report["traces"]
    summaries = report["summaries"]
    assert isinstance(traces, dict)
    assert isinstance(summaries, dict)
    for condition in lane.CONDITIONS:
        trace = traces[condition]
        assert isinstance(trace, list)
        previous: str | None = None
        for record in trace:
            assert isinstance(record, dict)
            if previous is not None:
                assert record["previous_record_hash"] == previous
            body = dict(record)
            supplied_previous = body.pop("previous_record_hash")
            supplied_hash = body.pop("record_hash")
            assert supplied_hash == lane._digest(
                {"previous_record_hash": supplied_previous, "record": body}
            )
            previous = str(supplied_hash)
        assert previous == summaries[condition]["trace_head"]

    forged = copy.deepcopy(report)
    forged_traces = forged["traces"]
    assert isinstance(forged_traces, dict)
    learned = forged_traces[lane.LEARNED_FEEDBACK]
    assert isinstance(learned, list)
    learned[0]["net_reward"] = float(learned[0]["net_reward"]) + 0.25
    unsigned = dict(forged)
    unsigned.pop("deterministic_payload_digest")
    forged["deterministic_payload_digest"] = lane._digest(unsigned)
    errors = lane.validate_prototype_partner_fusion_matched_v2_report(forged)
    assert "report differs from exact causal replay" in errors

    promoted = dict(report)
    promoted["assessment"] = "accepted"
    errors = lane.validate_prototype_partner_fusion_matched_v2_report(promoted)
    assert "assessment must remain not_assessed" in errors


@pytest.mark.parametrize(
    ("binding", "expected_error"),
    [
        ("source", "report source manifest changed"),
        ("runtime", "report runtime manifest changed"),
    ],
)
def test_report_checks_fresh_external_bindings_after_replay_cache_is_warm(
    report: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
    binding: str,
    expected_error: str,
) -> None:
    assert lane.validate_prototype_partner_fusion_matched_v2_report(report) == ()
    if binding == "source":
        monkeypatch.setattr(lane, "_source_manifest", lambda: {"changed.py": "0" * 64})
    else:
        changed_runtime = dict(lane._runtime_manifest())
        changed_runtime["backend"] = "changed-after-cache-warmup"
        monkeypatch.setattr(lane, "_runtime_manifest", lambda: changed_runtime)

    stale_but_resealed = copy.deepcopy(report)
    unsigned = dict(stale_but_resealed)
    unsigned.pop("deterministic_payload_digest")
    stale_but_resealed["deterministic_payload_digest"] = lane._digest(unsigned)
    errors = lane.validate_prototype_partner_fusion_matched_v2_report(stale_but_resealed)
    assert expected_error in errors


def test_report_runtime_binding_tracks_numpy_after_replay_cache_is_warm(
    report: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert lane.validate_prototype_partner_fusion_matched_v2_report(report) == ()
    monkeypatch.setattr(lane.np, "__version__", "changed-after-cache-warmup")
    assert "report runtime manifest changed" in (
        lane.validate_prototype_partner_fusion_matched_v2_report(report)
    )


@pytest.mark.parametrize(
    ("binding", "forged_value", "expected_error"),
    [
        ("config", {"forged": True}, "report config binding changed"),
        ("config_digest", "0" * 64, "report config digest binding changed"),
        ("agent_config", {"forged": True}, "report agent config binding changed"),
        ("protocol_digest", "0" * 64, "report protocol digest binding changed"),
        ("schedule_digest", "0" * 64, "report schedule digest binding changed"),
    ],
)
def test_report_rejects_resealed_internal_binding_drift_before_cached_replay(
    report: dict[str, object],
    binding: str,
    forged_value: object,
    expected_error: str,
) -> None:
    forged = copy.deepcopy(report)
    forged[binding] = forged_value
    unsigned = dict(forged)
    unsigned.pop("deterministic_payload_digest")
    forged["deterministic_payload_digest"] = lane._digest(unsigned)
    errors = lane.validate_prototype_partner_fusion_matched_v2_report(forged)
    assert expected_error in errors


def test_report_validator_contains_replay_failure_at_public_boundary(
    report: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable_replay() -> dict[str, object]:
        raise RuntimeError("replay authority unavailable")

    monkeypatch.setattr(lane, "_replay_expected_report_condition_by_condition", unavailable_replay)
    assert lane.validate_prototype_partner_fusion_matched_v2_report(report) == (
        "exact causal replay failed: RuntimeError: replay authority unavailable",
    )


@pytest.mark.parametrize(
    "field",
    ["schema", "assessment", "pairing_scope", "deterministic_payload_digest"],
)
def test_report_validation_rejects_malformed_scalar_leaves_without_raising(
    report: dict[str, object],
    field: str,
) -> None:
    malformed = dict(report)
    malformed[field] = np.asarray([1, 2], dtype=np.int32)
    errors = lane.validate_prototype_partner_fusion_matched_v2_report(malformed)
    assert errors


def test_matched_logical_work_and_real_kernel_parity_are_explicit(
    report: dict[str, object],
) -> None:
    assert report["prototype_eager_jit_parity"] is True
    resource = report["resource_report"]
    assert isinstance(resource, dict)
    assert resource["matched_logical_budgets"] is True
    assert resource["initial_typed_rng_keys_bit_identical"] is True
    assert resource["initial_learner_states_bit_identical"] is True
    assert resource["runtime_evaluator_rng_draws"] == 0
    per_condition = resource["per_condition"]
    assert isinstance(per_condition, dict)
    budgets = list(per_condition.values())
    assert all(budget == budgets[0] for budget in budgets[1:])
    for budget in budgets:
        assert budget["committed_prototype_transitions"] == lane.CONFIG.horizon
        assert budget["discarded_base_action_previews"] == lane.CONFIG.horizon
        assert budget["prototype_update_calls"] == 2 * lane.CONFIG.horizon
        assert budget["fusion_decision_calls"] == 2 * lane.CONFIG.horizon
        assert budget["fusion_feedback_opportunities"] == 2 * lane.CONFIG.horizon
        assert budget["fixed_message_slots_per_call"] == lane.MAX_PARTNERS
        assert budget["shared_mutable_agent_state"] is False
        assert budget["shared_mutable_environment_state"] is False


@pytest.mark.integration
def test_in_memory_checkpoint_resume_is_exact_and_tamper_fails_closed(
    report: dict[str, object],
) -> None:
    checkpoint = lane.make_prototype_partner_fusion_matched_v2_checkpoint(5)
    assert lane._validate_checkpoint(checkpoint) == ()
    resumed = lane.resume_prototype_partner_fusion_matched_v2_checkpoint(checkpoint)
    assert resumed == report

    stale_clock = dataclasses.replace(checkpoint, next_event=6, checkpoint_digest="")
    stale_clock = dataclasses.replace(
        stale_clock,
        checkpoint_digest=lane._digest(stale_clock.metadata()),
    )
    assert "checkpoint state clock mismatch" in lane._validate_checkpoint(stale_clock)

    changed_source = dict(checkpoint.source_manifest)
    source_name = next(iter(changed_source))
    changed_source[source_name] = "0" * 64
    bad_source = dataclasses.replace(
        checkpoint,
        source_manifest=changed_source,
        checkpoint_digest="",
    )
    bad_source = dataclasses.replace(
        bad_source,
        checkpoint_digest=lane._digest(bad_source.metadata()),
    )
    assert "checkpoint source manifest changed" in lane._validate_checkpoint(bad_source)

    changed_runtime = dict(checkpoint.runtime_manifest)
    changed_runtime["backend"] = "forged"
    bad_runtime = dataclasses.replace(
        checkpoint,
        runtime_manifest=changed_runtime,
        checkpoint_digest="",
    )
    bad_runtime = dataclasses.replace(
        bad_runtime,
        checkpoint_digest=lane._digest(bad_runtime.metadata()),
    )
    assert "checkpoint runtime manifest changed" in lane._validate_checkpoint(bad_runtime)

    bad_config = dataclasses.replace(
        checkpoint,
        config_digest="0" * 64,
        checkpoint_digest="",
    )
    bad_config = dataclasses.replace(
        bad_config,
        checkpoint_digest=lane._digest(bad_config.metadata()),
    )
    assert "checkpoint config digest changed" in lane._validate_checkpoint(bad_config)

    crossed_state = checkpoint.state
    learned = _arm(crossed_state, lane.LEARNED_FEEDBACK)
    fixed = _arm(crossed_state, lane.FIXED_ZERO_FEEDBACK)
    crossed_arm = dataclasses.replace(learned, pending_decision=fixed.pending_decision)
    crossed_state = dataclasses.replace(
        crossed_state,
        arms=(crossed_arm, *crossed_state.arms[1:]),
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
        lane.resume_prototype_partner_fusion_matched_v2_checkpoint(crossed)


def test_checkpoint_validation_rejects_malformed_fields_without_raising() -> None:
    checkpoint = lane.make_prototype_partner_fusion_matched_v2_checkpoint(1)

    wrong_state = dataclasses.replace(checkpoint, state=object())
    assert "checkpoint state has the wrong type" in lane._validate_checkpoint(wrong_state)
    with pytest.raises(ValueError, match="checkpoint state has the wrong type"):
        lane.resume_prototype_partner_fusion_matched_v2_checkpoint(wrong_state)

    malformed_source = dataclasses.replace(
        checkpoint,
        source_manifest={"source.py": object()},
    )
    assert any(
        error.startswith("checkpoint source manifest cannot be compared:")
        for error in lane._validate_checkpoint(malformed_source)
    )

    malformed_runtime = dataclasses.replace(
        checkpoint,
        runtime_manifest={"backend": object()},
    )
    assert any(
        error.startswith("checkpoint runtime manifest cannot be compared:")
        for error in lane._validate_checkpoint(malformed_runtime)
    )
