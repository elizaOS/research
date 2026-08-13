"""Cheap host contracts for development-only HCCL Core-L2/L3 longevity metrics."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Iterator

import numpy as np
import pytest

from alberta_framework.evaluation.hccl_longevity_endpoints import (
    HCCL_LONGEVITY_COMPLETE_TRACE_SCHEMA,
    HCCL_LONGEVITY_ENDPOINT_REPORT_SCHEMA,
    HCCL_LONGEVITY_ENDPOINT_STATUS,
    HCCL_LONGEVITY_ENDPOINT_WINDOW_STEPS,
    HCCLLongevityCompleteTrace,
    HCCLLongevityEndpointConfig,
    HCCLLongevityEndpointReport,
    evaluate_hccl_longevity_endpoints,
    validate_hccl_longevity_complete_trace,
    validate_hccl_longevity_endpoint_report,
)
from alberta_framework.streams.hccl_causal_core import (
    HCCL_CAUSAL_CORE_L2_PROFILE,
    HCCL_CAUSAL_CORE_L3_PROFILE,
    HCCL_CAUSAL_CORE_REGIME_NAMES,
    hccl_causal_core_cycle_count_for_profile,
    hccl_causal_core_lifetime_for_profile,
    hccl_causal_core_schedule_for_profile,
)

pytestmark = [pytest.mark.unit, pytest.mark.development]


def _synthetic_l2_trace() -> HCCLLongevityCompleteTrace:
    """Build one small (about 3 MiB) exact L2 trace; never allocate an L3 trace."""

    profile = HCCL_CAUSAL_CORE_L2_PROFILE
    schedule = hccl_causal_core_schedule_for_profile(profile)
    n = hccl_causal_core_lifetime_for_profile(profile)
    regime_ids = np.empty((n,), dtype=np.int32)
    matrix = np.empty((n, 4), dtype=np.float32)
    initial_tail = {"A": 10.0, "B": 20.0, "C": 30.0, "D": 40.0}

    for segment_index, (regime_name, start, end) in enumerate(schedule):
        cycle_index, _canonical_segment_index = divmod(segment_index, 10)
        regime_id = HCCL_CAUSAL_CORE_REGIME_NAMES.index(regime_name)
        matrix[start:end] = np.asarray(
            (
                100.0 + cycle_index,
                200.0 + 2.0 * cycle_index,
                300.0 - cycle_index,
                400.0 - 3.0 * cycle_index,
            ),
            dtype=np.float32,
        )
        regime_ids[start:end] = regime_id

    runs: list[tuple[str, int, int]] = []
    for regime_name, start, end in schedule:
        if runs and runs[-1][0] == regime_name and runs[-1][2] == start:
            runs[-1] = (regime_name, runs[-1][1], end)
        else:
            runs.append((regime_name, start, end))
    regime_occurrence = {name: 0 for name in HCCL_CAUSAL_CORE_REGIME_NAMES}
    for regime_name, start, end in runs:
        regime_id = HCCL_CAUSAL_CORE_REGIME_NAMES.index(regime_name)
        occurrence_index = regime_occurrence[regime_name]
        regime_occurrence[regime_name] += 1
        if occurrence_index == 0:
            matrix[start:end, regime_id] = np.float32(initial_tail[regime_name])
        else:
            prior_tail = initial_tail[regime_name] + 2.0 * (occurrence_index - 1)
            matrix[start : start + HCCL_LONGEVITY_ENDPOINT_WINDOW_STEPS, regime_id] = (
                np.float32(prior_tail - 2.0)
            )
            matrix[start + HCCL_LONGEVITY_ENDPOINT_WINDOW_STEPS : end, regime_id] = (
                np.float32(prior_tail + 2.0)
            )

    task_scores = matrix[np.arange(n), regime_ids].astype(np.float32, copy=True)
    net_rewards = np.repeat(task_scores[:, None], 2, axis=1).astype(np.float32)
    pre = np.zeros((n, 2), dtype=np.uint32)
    post = np.zeros((n, 2), dtype=np.uint32)
    pre[:, 1] = np.arange(n, dtype=np.uint32)
    post[:, 1] = np.arange(1, n + 1, dtype=np.uint32)
    return HCCLLongevityCompleteTrace(
        schedule_profile=profile,
        regime_ids=regime_ids,
        transaction_committed=np.ones((n,), dtype=np.bool_),
        pre_step_words=pre,
        post_step_words=post,
        task_scores=task_scores,
        net_rewards=net_rewards,
        all_regime_score_matrix=matrix,
    )


@pytest.fixture(scope="module")
def l2_trace() -> Iterator[HCCLLongevityCompleteTrace]:
    trace = _synthetic_l2_trace()
    yield trace


@pytest.fixture(scope="module")
def l2_report(l2_trace: HCCLLongevityCompleteTrace) -> HCCLLongevityEndpointReport:
    return evaluate_hccl_longevity_endpoints(l2_trace)


@pytest.mark.parametrize(
    (
        "profile",
        "cycles",
        "steps",
        "segments",
        "maximal_occurrences",
        "genuine_recurrences",
    ),
    (
        (HCCL_CAUSAL_CORE_L2_PROFILE, 8, 71_984, 80, 59, 55),
        (HCCL_CAUSAL_CORE_L3_PROFILE, 112, 1_007_776, 1_120, 787, 783),
    ),
)
def test_fixed_profiles_reuse_world_geometry_without_allocating_l3(
    profile: str,
    cycles: int,
    steps: int,
    segments: int,
    maximal_occurrences: int,
    genuine_recurrences: int,
) -> None:
    config = HCCLLongevityEndpointConfig(schedule_profile=profile)
    payload = config.to_config()

    assert config.cycle_count == cycles == hccl_causal_core_cycle_count_for_profile(profile)
    assert config.total_steps == steps == hccl_causal_core_lifetime_for_profile(profile)
    assert len(config.schedule) == segments == len(
        hccl_causal_core_schedule_for_profile(profile)
    )
    assert config.maximal_regime_occurrence_count == maximal_occurrences
    assert config.genuine_recurrence_comparison_count == genuine_recurrences
    assert payload["schedule_profile"] == profile
    assert payload["schedule_segment_count"] == segments
    assert payload["maximal_regime_occurrence_count"] == maximal_occurrences
    assert payload["genuine_recurrence_comparison_count"] == genuine_recurrences
    assert payload["window_steps"] == HCCL_LONGEVITY_ENDPOINT_WINDOW_STEPS == 64
    assert payload["complete_trace_required"] is True
    assert payload["learner_visible_regime_labels"] is False
    assert payload["learner_visible_schedule_boundaries"] is False
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
    assert HCCLLongevityEndpointConfig.from_config(
        json.loads(json.dumps(payload, allow_nan=False, sort_keys=True))
    ) == config


def test_config_rejects_nonlongevity_profiles_and_payload_mutation() -> None:
    with pytest.raises(ValueError, match="Core-L2 or Core-L3"):
        HCCLLongevityEndpointConfig(schedule_profile="canonical-8998-v1")
    with pytest.raises(TypeError, match="exact string"):
        HCCLLongevityEndpointConfig(schedule_profile=1)  # type: ignore[arg-type]

    payload = HCCLLongevityEndpointConfig().to_config()
    payload["promotion_authorized"] = True
    with pytest.raises(ValueError, match="fixed configuration"):
        HCCLLongevityEndpointConfig.from_config(payload)


def test_complete_l2_trace_is_strict_frozen_and_profile_bound(
    l2_trace: HCCLLongevityCompleteTrace,
) -> None:
    assert l2_trace.schema == HCCL_LONGEVITY_COMPLETE_TRACE_SCHEMA
    assert validate_hccl_longevity_complete_trace(l2_trace) is l2_trace
    for name in (
        "regime_ids",
        "transaction_committed",
        "pre_step_words",
        "post_step_words",
        "task_scores",
        "net_rewards",
        "all_regime_score_matrix",
    ):
        array = getattr(l2_trace, name)
        assert array.flags.c_contiguous
        assert not array.flags.writeable
    selected = l2_trace.all_regime_score_matrix[
        np.arange(l2_trace.task_scores.size), l2_trace.regime_ids
    ]
    np.testing.assert_array_equal(selected, l2_trace.task_scores)


def test_strict_adapter_accepts_an_existing_exact_runner_life(
    l2_trace: HCCLLongevityCompleteTrace,
) -> None:
    from alberta_framework.core.hccl_continual_dyad_runner import (
        HCCLContinualDyadLifeTrace,
    )

    life = HCCLContinualDyadLifeTrace(
        schedule_profile=l2_trace.schedule_profile,
        regime_ids=l2_trace.regime_ids,
        transaction_committed=l2_trace.transaction_committed,
        pre_step_words=l2_trace.pre_step_words,
        post_step_words=l2_trace.post_step_words,
        task_scores=l2_trace.task_scores,
        net_rewards=l2_trace.net_rewards,
        all_regime_score_matrix=l2_trace.all_regime_score_matrix,
    )
    converted = HCCLLongevityCompleteTrace.from_continual_dyad_life_trace(life)

    assert converted.schedule_profile == HCCL_CAUSAL_CORE_L2_PROFILE
    assert validate_hccl_longevity_complete_trace(converted) is converted
    np.testing.assert_array_equal(converted.task_scores, life.task_scores)
    with pytest.raises(TypeError, match="exact HCCLContinualDyadLifeTrace"):
        HCCLLongevityCompleteTrace.from_continual_dyad_life_trace(l2_trace)


def test_all_occurrences_cycles_and_recurrence_formulas_are_explicit(
    l2_report: HCCLLongevityEndpointReport,
) -> None:
    assert l2_report.schema == HCCL_LONGEVITY_ENDPOINT_REPORT_SCHEMA
    assert l2_report.status == HCCL_LONGEVITY_ENDPOINT_STATUS == "not_assessed"
    assert len(l2_report.segments) == 80
    assert len(l2_report.occurrences) == 59
    assert len(l2_report.recurrences) == 55
    assert len(l2_report.cycles) == 8
    assert len(l2_report.segment_performance_matrix) == 80
    assert len(l2_report.occurrence_performance_matrix) == 59
    assert all(
        left.end_step_exclusive == right.start_step
        and left.regime_name != right.regime_name
        for left, right in zip(l2_report.occurrences, l2_report.occurrences[1:])
    )
    assert any(
        record.regime_name == "A"
        and record.first_segment_index == 9
        and record.last_segment_index_exclusive == 11
        and record.start_cycle_index == 0
        and record.end_cycle_index_inclusive == 1
        for record in l2_report.occurrences
    )
    assert any(
        record.regime_name == "A"
        and record.first_segment_index == 12
        and record.last_segment_index_exclusive == 15
        for record in l2_report.occurrences
    )
    assert all(record.intervening_segment_count >= 1 for record in l2_report.recurrences)
    assert all(
        record.current_first_segment_index
        > record.previous_last_segment_index_exclusive
        for record in l2_report.recurrences
    )

    second_b = next(
        record
        for record in l2_report.recurrences
        if record.regime_name == "B" and record.current_regime_occurrence_index == 1
    )
    assert second_b.previous_regime_occurrence_index == 0
    assert second_b.prior_tail_reference == pytest.approx(20.0)
    assert second_b.current_entry_mean == pytest.approx(18.0)
    assert second_b.current_tail_mean == pytest.approx(22.0)
    assert second_b.entry_retention_delta == pytest.approx(-2.0)
    assert second_b.entry_forgetting_gap == pytest.approx(2.0)
    assert second_b.tail_backward_transfer == pytest.approx(2.0)
    assert second_b.tail_forgetting_gap == pytest.approx(-2.0)
    assert second_b.trailing64_recovered is True
    assert second_b.trailing64_recovery_steps_after_entry == 32
    assert second_b.trailing64_recovery_endpoint_step_exclusive == 5_987 + 64 + 32
    assert second_b.positive_gap_recovery_area == pytest.approx(33.0)

    assert [record.cycle_index for record in l2_report.cycles] == list(range(8))
    for cycle in l2_report.cycles:
        assert cycle.start_step == cycle.cycle_index * 8_998
        assert cycle.end_step_exclusive == (cycle.cycle_index + 1) * 8_998
        assert cycle.segment_count == 10
        assert all(value is not None for value in cycle.regime_segment_tail_task_means[:3])
        assert (cycle.regime_segment_tail_task_means[3] is not None) is (
            cycle.cycle_index == 0
        )
    assert sum(record.maximal_occurrence_start_count for record in l2_report.cycles) == 59
    assert sum(record.maximal_occurrence_end_count for record in l2_report.cycles) == 59
    assert l2_report.cycles[0].task_mean_change_from_previous is None
    assert all(
        record.task_mean_change_from_previous is not None
        for record in l2_report.cycles[1:]
    )


def test_regime_aggregates_cover_every_comparison_and_d_is_unavailable(
    l2_report: HCCLLongevityEndpointReport,
) -> None:
    summaries = {record.regime_name: record for record in l2_report.regime_summaries}
    expected_counts = {"A": (47, 26), "B": (16, 16), "C": (16, 16)}
    for name, (segment_count, occurrence_count) in expected_counts.items():
        summary = summaries[name]
        assert summary.scheduled_segment_count == segment_count
        assert summary.maximal_contiguous_occurrence_count == occurrence_count
        assert summary.recurrence_available is True
        assert summary.recurrence_comparison_count == occurrence_count - 1
        assert summary.mean_entry_retention_delta == pytest.approx(-2.0)
        assert summary.worst_entry_retention_delta == pytest.approx(-2.0)
        assert summary.mean_tail_backward_transfer == pytest.approx(2.0)
        assert summary.worst_tail_backward_transfer == pytest.approx(2.0)
        assert summary.occurrence_tail_slope == pytest.approx(2.0)
        assert summary.recovered_comparison_count == occurrence_count - 1
        assert summary.recovery_fraction == pytest.approx(1.0)
        assert summary.mean_recovery_steps_after_entry_for_recovered == pytest.approx(32.0)
        assert summary.mean_positive_gap_recovery_area == pytest.approx(33.0)

    d = summaries["D"]
    assert d.scheduled_segment_count == 1
    assert d.maximal_contiguous_occurrence_count == 1
    assert d.recurrence_available is False
    assert d.recurrence_unavailable_reason == (
        "D has one maximal contiguous occurrence; recurrence-dependent metrics are "
        "unavailable"
    )
    for field in (
        "recurrence_comparison_count",
        "mean_entry_retention_delta",
        "worst_entry_retention_delta",
        "mean_tail_backward_transfer",
        "worst_tail_backward_transfer",
        "occurrence_tail_slope",
        "recovered_comparison_count",
        "recovery_fraction",
        "mean_recovery_steps_after_entry_for_recovered",
        "mean_positive_gap_recovery_area",
    ):
        assert getattr(d, field) is None
    assert d.first_exposure_tail_counterfactual is not None
    assert d.latest_life_tail_counterfactual is not None
    assert d.learner_inferred_known_obsolete is False
    assert all(record.regime_name != "D" for record in l2_report.recurrences)


def test_lifetime_trend_is_recomputed_from_every_cycle_and_occurrence(
    l2_report: HCCLLongevityEndpointReport,
) -> None:
    trend = l2_report.lifetime_trend
    cycle_task_means = np.asarray([record.task_mean for record in l2_report.cycles])
    cycle_tail_means = np.asarray(
        [record.segment_tail_task_mean for record in l2_report.cycles]
    )
    segment_tail_means = np.asarray(
        [record.tail_task_mean for record in l2_report.segments]
    )
    occurrence_tail_means = np.asarray(
        [record.tail_task_mean for record in l2_report.occurrences]
    )

    def slope(values: np.ndarray) -> float:
        x = np.arange(values.size, dtype=np.float64)
        return float(
            np.sum((x - x.mean()) * (values - values.mean()))
            / np.sum((x - x.mean()) ** 2)
        )

    assert trend.cycle_count == len(l2_report.cycles) == 8
    assert trend.segment_count == len(l2_report.segments) == 80
    assert trend.occurrence_count == len(l2_report.occurrences) == 59
    assert trend.first_cycle_task_mean == pytest.approx(cycle_task_means[0])
    assert trend.latest_cycle_task_mean == pytest.approx(cycle_task_means[-1])
    assert trend.first_to_latest_cycle_task_change == pytest.approx(
        cycle_task_means[-1] - cycle_task_means[0]
    )
    assert trend.cycle_task_mean_slope == pytest.approx(slope(cycle_task_means))
    assert trend.cycle_segment_tail_mean_slope == pytest.approx(slope(cycle_tail_means))
    assert trend.global_segment_tail_mean_slope == pytest.approx(slope(segment_tail_means))
    assert trend.global_occurrence_tail_mean_slope == pytest.approx(
        slope(occurrence_tail_means)
    )


def test_report_is_deterministic_strict_and_permanently_nonpromoting(
    l2_trace: HCCLLongevityCompleteTrace,
    l2_report: HCCLLongevityEndpointReport,
) -> None:
    assert validate_hccl_longevity_endpoint_report(l2_report, l2_trace) is l2_report
    assert evaluate_hccl_longevity_endpoints(l2_trace) == l2_report
    payload = l2_report.to_config()
    assert payload["transactions_committed"] == 71_984
    assert payload["schedule_segment_count"] == 80
    assert payload["maximal_regime_occurrence_count"] == 59
    assert payload["d_maximal_regime_occurrence_count"] == 1
    assert payload["d_recurrence_available"] is False
    assert payload["d_recurrence_dependent_metrics"] is None
    assert payload["learner_visible_regime_labels"] is False
    assert payload["learner_visible_schedule_boundaries"] is False
    assert payload["counterfactual_score_columns_exposed_to_learner"] is False
    assert payload["acceptance_thresholds_defined"] is False
    assert payload["promotion_authorized"] is False
    assert json.loads(json.dumps(payload, allow_nan=False, sort_keys=True)) == payload

    tampered = dataclasses.replace(l2_report, status="accepted")
    with pytest.raises(ValueError, match="deterministic recomputation"):
        validate_hccl_longevity_endpoint_report(tampered, l2_trace)


def test_trace_fails_closed_on_profile_exposure_clock_commit_score_and_finite_tamper() -> None:
    trace = _synthetic_l2_trace()
    with pytest.raises(ValueError, match="does not match trace"):
        evaluate_hccl_longevity_endpoints(
            trace,
            HCCLLongevityEndpointConfig(schedule_profile=HCCL_CAUSAL_CORE_L3_PROFILE),
        )
    with pytest.raises(ValueError, match="evaluator_regime_ids"):
        dataclasses.replace(trace, learner_received_evaluator_regime_ids=True)
    with pytest.raises(ValueError, match="schedule_boundaries"):
        dataclasses.replace(trace, learner_received_evaluator_schedule_boundaries=True)
    with pytest.raises(ValueError, match="counterfactual_scores"):
        dataclasses.replace(trace, learner_received_counterfactual_scores=True)

    bad_clock = trace.post_step_words.copy()
    bad_clock[100, 1] += np.uint32(1)
    with pytest.raises(ValueError, match="monotone committed clocks"):
        dataclasses.replace(trace, post_step_words=bad_clock)

    bad_commit = trace.transaction_committed.copy()
    bad_commit[100] = False
    with pytest.raises(ValueError, match="must be committed"):
        dataclasses.replace(trace, transaction_committed=bad_commit)

    bad_regime = trace.regime_ids.copy()
    bad_regime[0] = 1
    with pytest.raises(ValueError, match="world-owned schedule"):
        dataclasses.replace(trace, regime_ids=bad_regime)

    bad_matrix = trace.all_regime_score_matrix.copy()
    bad_matrix[0, 0] += np.float32(1.0)
    with pytest.raises(ValueError, match="selected evaluator column"):
        dataclasses.replace(trace, all_regime_score_matrix=bad_matrix)

    bad_rewards = trace.net_rewards.copy()
    bad_rewards[5, 1] = np.float32(np.nan)
    with pytest.raises(ValueError, match="entirely finite"):
        dataclasses.replace(trace, net_rewards=bad_rewards)

    wrong_finite_rewards = trace.net_rewards.copy()
    wrong_finite_rewards[5, 1] += np.float32(1.0)
    with pytest.raises(ValueError, match="exactly equal task score"):
        dataclasses.replace(trace, net_rewards=wrong_finite_rewards)


def test_trace_revalidator_detects_thaw_after_construction() -> None:
    trace = _synthetic_l2_trace()
    trace.regime_ids.flags.writeable = True
    with pytest.raises(ValueError, match="read-only C-contiguous"):
        validate_hccl_longevity_complete_trace(trace)
