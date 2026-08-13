"""Fast host-side contracts for canonical HCCL causal-core endpoint metrics."""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest

from alberta_framework.evaluation.hccl_causal_core_endpoints import (
    HCCL_CAUSAL_CORE_COMPLETE_TRACE_SCHEMA,
    HCCL_CAUSAL_CORE_ENDPOINT_ENTRY_WINDOW,
    HCCL_CAUSAL_CORE_ENDPOINT_REPORT_SCHEMA,
    HCCL_CAUSAL_CORE_ENDPOINT_STATUS,
    HCCL_CAUSAL_CORE_ENDPOINT_TAIL_WINDOW,
    HCCL_CAUSAL_CORE_ENDPOINT_TOTAL_STEPS,
    HCCLCausalCoreCompleteTrace,
    HCCLCausalCoreEndpointConfig,
    evaluate_hccl_causal_core_endpoints,
    validate_hccl_causal_core_complete_trace,
    validate_hccl_causal_core_endpoint_report,
)
from alberta_framework.streams.hccl_causal_core import (
    HCCL_CAUSAL_CORE_REGIME_NAMES,
    HCCL_CAUSAL_CORE_SCHEDULE,
)

pytestmark = [pytest.mark.unit, pytest.mark.development]


def _synthetic_trace() -> HCCLCausalCoreCompleteTrace:
    n = HCCL_CAUSAL_CORE_ENDPOINT_TOTAL_STEPS
    regime_ids = np.empty((n,), dtype=np.int32)
    score_matrix = np.empty((n, 4), dtype=np.float32)
    selected_phase_scores = {
        0: 10.0,
        1: 10.0,
        2: 8.0,
        3: 4.0,
        4: 11.0,
        5: 7.0,
        6: 9.0,
        8: 6.0,
        9: 13.0,
    }
    for phase_index, (regime_name, start, end) in enumerate(HCCL_CAUSAL_CORE_SCHEDULE):
        regime_id = HCCL_CAUSAL_CORE_REGIME_NAMES.index(regime_name)
        regime_ids[start:end] = regime_id
        score_matrix[start:end] = np.asarray(
            (phase_index, 100 + phase_index, 200 + phase_index, 300 + phase_index),
            dtype=np.float32,
        )
        if phase_index == 7:
            score_matrix[start : start + 64, regime_id] = np.float32(8.0)
            score_matrix[start + 64 : end, regime_id] = np.float32(12.0)
        else:
            score_matrix[start:end, regime_id] = np.float32(selected_phase_scores[phase_index])

    # Make A's phase-8 counterfactual endpoint the longitudinal peak, so its
    # peak-to-latest forgetting formula has a nonzero, exactly checkable value.
    _name, phase_eight_start, phase_eight_end = HCCL_CAUSAL_CORE_SCHEDULE[8]
    score_matrix[phase_eight_start:phase_eight_end, 0] = np.float32(20.0)

    task_scores = score_matrix[np.arange(n), regime_ids].copy()
    net_rewards = np.broadcast_to(task_scores[:, None], (n, 2)).copy()
    pre = np.zeros((n, 2), dtype=np.uint32)
    post = np.zeros((n, 2), dtype=np.uint32)
    pre[:, 1] = np.arange(n, dtype=np.uint32)
    post[:, 1] = np.arange(1, n + 1, dtype=np.uint32)
    return HCCLCausalCoreCompleteTrace(
        regime_ids=regime_ids,
        transaction_committed=np.ones((n,), dtype=np.bool_),
        pre_step_words=pre,
        post_step_words=post,
        task_scores=task_scores,
        net_rewards=net_rewards,
        all_regime_score_matrix=score_matrix,
    )


def test_fixed_config_json_roundtrip_and_nonclaims_are_explicit() -> None:
    config = HCCLCausalCoreEndpointConfig()
    payload = config.to_config()
    decoded = json.loads(json.dumps(payload, allow_nan=False, sort_keys=True))

    assert config.total_steps == 8_998
    assert config.entry_window_steps == config.tail_window_steps == 64
    assert payload["schedule"] == [
        {"regime": name, "start": start, "end": end}
        for name, start, end in HCCL_CAUSAL_CORE_SCHEDULE
    ]
    assert payload["regime_id_order"] == ["A", "B", "C", "D"]
    assert payload["evaluator_labels_exposed_to_learner"] is False
    assert payload["counterfactual_score_columns_exposed_to_learner"] is False
    assert payload["acceptance_thresholds_defined"] is False
    for field in (
        "benchmark_execution_authorized",
        "output_writes_authorized",
        "artifact_authorized",
        "evidence_authorized",
        "seed_reservation_or_consumption_authorized",
        "promotion_authorized",
    ):
        assert payload[field] is False
    assert "seed" not in payload
    assert HCCLCausalCoreEndpointConfig.from_config(decoded) == config

    decoded["entry_window_steps"] = 64.0
    with pytest.raises(ValueError, match="differs"):
        HCCLCausalCoreEndpointConfig.from_config(decoded)


def test_complete_trace_and_occurrence_endpoint_formulas() -> None:
    trace = _synthetic_trace()
    report = evaluate_hccl_causal_core_endpoints(trace)

    assert trace.schema == HCCL_CAUSAL_CORE_COMPLETE_TRACE_SCHEMA
    assert validate_hccl_causal_core_complete_trace(trace) is trace
    assert report.schema == HCCL_CAUSAL_CORE_ENDPOINT_REPORT_SCHEMA
    assert report.status == HCCL_CAUSAL_CORE_ENDPOINT_STATUS == "not_assessed"
    assert len(report.occurrences) == len(HCCL_CAUSAL_CORE_SCHEDULE) == 10
    assert len(report.phase_performance_matrix) == 10
    assert all(len(row) == 4 for row in report.phase_performance_matrix)
    assert len(report.recurrences) == 6

    second_b = report.occurrences[7]
    assert second_b.regime_name == "B"
    assert second_b.occurrence_index == 2
    assert second_b.entry_task_mean == pytest.approx(8.0)
    assert second_b.tail_task_mean == pytest.approx(12.0)
    assert second_b.entry_net_reward_means == pytest.approx((8.0, 8.0))
    assert second_b.tail_net_reward_means == pytest.approx((12.0, 12.0))
    assert report.phase_performance_matrix[7][1] == pytest.approx(12.0)
    assert report.phase_performance_matrix[8][0] == pytest.approx(20.0)

    payload = report.to_config()
    assert payload["transactions_committed"] == 8_998
    assert payload["reset_callback_count"] == payload["boundary_callback_count"] == 0
    assert payload["evaluator_labels_exposed_to_learner"] is False
    assert payload["counterfactual_score_columns_exposed_to_learner"] is False
    assert json.loads(json.dumps(payload, allow_nan=False)) == payload


def test_recurrence_gap_recovery_area_and_slope_formulas() -> None:
    report = evaluate_hccl_causal_core_endpoints(_synthetic_trace())
    second_b = next(record for record in report.recurrences if record.regime_name == "B")

    assert second_b.prior_tail_reference == pytest.approx(10.0)
    assert second_b.current_entry_mean == pytest.approx(8.0)
    assert second_b.current_tail_mean == pytest.approx(12.0)
    assert second_b.entry_gap == pytest.approx(2.0)
    assert second_b.tail_gap == pytest.approx(-2.0)
    assert second_b.tail_backward_transfer == pytest.approx(2.0)
    assert second_b.trailing64_recovered is True
    assert second_b.trailing64_recovery_steps_after_entry == 32
    assert second_b.trailing64_recovery_endpoint_step_exclusive == 5_987 + 64 + 32
    # Gaps are 2, 31/16, ..., 1/16, 0 over the eligible endpoints.
    assert second_b.positive_gap_recovery_area == pytest.approx(33.0)

    second_a = next(
        record
        for record in report.recurrences
        if record.regime_name == "A" and record.current_occurrence_index == 2
    )
    assert second_a.trailing64_recovered is False
    assert second_a.trailing64_recovery_steps_after_entry is None
    assert second_a.trailing64_recovery_endpoint_step_exclusive is None
    # Phase A2 has 829 transitions and therefore 829 - 64 + 1 rolling endpoints.
    assert second_a.positive_gap_recovery_area == pytest.approx((829 - 64 + 1) * 2.0)

    summaries = {record.regime_name: record for record in report.regime_summaries}
    assert summaries["A"].recurrence_slope == pytest.approx(0.7)
    assert summaries["B"].recurrence_slope == pytest.approx(2.0)
    assert summaries["C"].recurrence_slope == pytest.approx(-1.0)
    assert report.worst_recurring_entry_gap.value == pytest.approx(2.0)
    assert report.worst_recurring_entry_gap.regime_name == "A"
    assert report.worst_recurring_entry_gap.current_phase_index == 2
    assert report.worst_recurring_tail_gap.value == pytest.approx(2.0)


def test_per_regime_forgetting_transfer_and_d_nonrecurrence_are_explicit() -> None:
    report = evaluate_hccl_causal_core_endpoints(_synthetic_trace())
    summaries = {record.regime_name: record for record in report.regime_summaries}

    a = summaries["A"]
    assert a.scheduled_occurrence_count == 5
    assert a.recurring is a.recurrence_available is True
    assert a.first_exposure_phase_index == 0
    assert a.peak_phase_index == 8
    assert a.first_exposure_tail_performance == pytest.approx(10.0)
    assert a.peak_tail_performance == pytest.approx(20.0)
    assert a.latest_tail_performance == pytest.approx(13.0)
    assert a.peak_to_latest_forgetting == pytest.approx(7.0)
    assert a.backward_transfer == pytest.approx(3.0)

    d = summaries["D"]
    assert d.scheduled_occurrence_count == 1
    assert d.recurring is False
    assert d.recurrence_available is False
    assert d.recurrence_slope is None
    assert d.recurrence_unavailable_reason == (
        "D has one scheduled occurrence; recurrence metrics are unavailable"
    )
    assert d.learner_inferred_known_obsolete is False
    assert all(record.regime_name != "D" for record in report.recurrences)
    assert report.to_config()["d_known_obsolete_inferred_by_learner"] is False


def test_report_records_are_frozen_deterministic_and_strictly_recomputed() -> None:
    trace = _synthetic_trace()
    first = evaluate_hccl_causal_core_endpoints(trace)
    second = evaluate_hccl_causal_core_endpoints(trace)

    assert first == second
    assert first.to_config() == second.to_config()
    assert validate_hccl_causal_core_endpoint_report(first, trace) is first
    with pytest.raises(dataclasses.FrozenInstanceError):
        first.status = "accepted"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        first.occurrences[0].tail_task_mean = 0.0  # type: ignore[misc]

    tampered = dataclasses.replace(first, status="accepted")
    with pytest.raises(ValueError, match="recomputation"):
        validate_hccl_causal_core_endpoint_report(tampered, trace)


def test_trace_arrays_are_defensively_copied_and_read_only() -> None:
    source = _synthetic_trace()
    regime_copy = source.regime_ids.copy()
    reconstructed = dataclasses.replace(source, regime_ids=regime_copy)
    regime_copy[0] = 1

    assert reconstructed.regime_ids[0] == 0
    with pytest.raises(ValueError, match="read-only"):
        reconstructed.regime_ids[0] = 1

    # NumPy permits an owning array's writeability flag to be re-enabled.  The
    # public validator must notice that loss of the frozen-trace contract even
    # when no element has yet changed.
    reconstructed.regime_ids.flags.writeable = True
    with pytest.raises(ValueError, match="read-only C-contiguous"):
        validate_hccl_causal_core_complete_trace(reconstructed)


def test_trace_fails_closed_on_regime_clock_commit_task_column_and_nan_tamper() -> None:
    trace = _synthetic_trace()

    regimes = trace.regime_ids.copy()
    regimes[0] = 1
    with pytest.raises(ValueError, match="regime_ids"):
        dataclasses.replace(trace, regime_ids=regimes)

    clocks = trace.post_step_words.copy()
    clocks[100, 0] = np.uint32(1)
    with pytest.raises(ValueError, match="committed canonical clocks"):
        dataclasses.replace(trace, post_step_words=clocks)

    committed = trace.transaction_committed.copy()
    committed[123] = False
    with pytest.raises(ValueError, match="must be committed"):
        dataclasses.replace(trace, transaction_committed=committed)

    scores = trace.all_regime_score_matrix.copy()
    scores[0, 0] += np.float32(1.0)
    with pytest.raises(ValueError, match="matrix column"):
        dataclasses.replace(trace, all_regime_score_matrix=scores)

    net_rewards = trace.net_rewards.copy()
    net_rewards[5, 1] = np.float32(np.nan)
    with pytest.raises(ValueError, match="entirely finite"):
        dataclasses.replace(trace, net_rewards=net_rewards)


def test_trace_rejects_finite_net_reward_values_that_causal_core_cannot_emit() -> None:
    trace = _synthetic_trace()
    net_rewards = trace.net_rewards.copy()
    net_rewards[5, 1] = np.float32(net_rewards[5, 1] + 0.25)

    with pytest.raises(ValueError, match="net rewards must exactly equal task score"):
        dataclasses.replace(trace, net_rewards=net_rewards)


def test_trace_fails_closed_on_callbacks_or_evaluator_input_exposure() -> None:
    trace = _synthetic_trace()
    with pytest.raises(ValueError, match="reset_callback_count"):
        dataclasses.replace(trace, reset_callback_count=1)
    with pytest.raises(ValueError, match="boundary_callback_count"):
        dataclasses.replace(trace, boundary_callback_count=1)
    with pytest.raises(ValueError, match="learner_received_evaluator_regime_ids"):
        dataclasses.replace(trace, learner_received_evaluator_regime_ids=True)
    with pytest.raises(ValueError, match="learner_received_counterfactual_scores"):
        dataclasses.replace(trace, learner_received_counterfactual_scores=True)


def test_window_and_schedule_constants_are_exactly_canonical() -> None:
    assert HCCL_CAUSAL_CORE_ENDPOINT_ENTRY_WINDOW == 64
    assert HCCL_CAUSAL_CORE_ENDPOINT_TAIL_WINDOW == 64
    assert HCCL_CAUSAL_CORE_ENDPOINT_TOTAL_STEPS == 8_998
    assert HCCL_CAUSAL_CORE_SCHEDULE == (
        ("A", 0, 769),
        ("B", 769, 1566),
        ("A", 1566, 2395),
        ("D", 2395, 3252),
        ("A", 3252, 4135),
        ("C", 4135, 5046),
        ("A", 5046, 5987),
        ("B", 5987, 6958),
        ("C", 6958, 7967),
        ("A", 7967, 8998),
    )
