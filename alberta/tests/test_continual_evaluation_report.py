"""Invariant and toy integration tests for the strict continual report."""

from __future__ import annotations

import json
import math
import sys
import tracemalloc
from copy import deepcopy
from dataclasses import fields, replace
from time import perf_counter_ns

import pytest

from alberta_framework.evaluation.continual_evaluation_report import (
    ACCEPTANCE_STATUS,
    SCHEMA_VERSION,
    ConditionEvaluationInput,
    ContinualEvaluationProtocol,
    EnergyMeasurement,
    EvaluatorMatrix,
    LatencyMeasurements,
    LatencySummary,
    MatchedBudget,
    MemoryHighWaterMarks,
    OperationalMeasurements,
    OperationCounts,
    PredictBeforeUpdateTrace,
    SafetyMeasurements,
    build_continual_evaluation_report,
    continual_evaluation_report_json,
    validate_continual_evaluation_report,
)

pytestmark = pytest.mark.unit


def _protocol() -> ContinualEvaluationProtocol:
    return ContinualEvaluationProtocol(
        protocol_id="hand-computed-aba.v1",
        higher_is_better=True,
        regime_schedule=("A", "A", "A", "B", "B", "B", "A", "A", "A"),
        evaluator_regime_ids=("A", "B"),
        checkpoint_steps=(3, 6, 9),
        first_exposure_checkpoint={"A": 0, "B": 1},
        forward_transfer_reference={"A": 0.5, "B": 0.5},
        recovery_thresholds={"A": 0.8, "B": 0.8},
        stability_references={"A": 0.8, "B": 0.8},
        recovery_window=2,
        worst_window_size=2,
        dropped_observation_score=0.0,
    )


def _budget() -> MatchedBudget:
    return MatchedBudget(
        observation_limit=9,
        forward_call_limit=30,
        update_call_limit=9,
        backward_call_limit=9,
        parameter_count_limit=16,
        persistent_state_bytes_limit=4_096,
    )


def _latencies(*, measured: bool = True) -> LatencySummary:
    if not measured:
        return LatencySummary(None, None, None)
    return LatencySummary(0.1, 0.2, 0.3)


def _condition(
    name: str,
    *,
    scores: tuple[float | None, ...],
    processed: tuple[bool, ...],
    matrix: tuple[tuple[float, ...], ...],
    candidate_diagnostics: bool = False,
    budget: MatchedBudget | None = None,
) -> ConditionEvaluationInput:
    processed_count = sum(processed)
    return ConditionEvaluationInput(
        name=name,
        trace=PredictBeforeUpdateTrace(scores=scores, processed=processed),
        evaluator_matrix=EvaluatorMatrix(matrix),
        budget=_budget() if budget is None else budget,
        operations=OperationalMeasurements(
            counts=OperationCounts(
                processed_observations=processed_count,
                delayed_observations=0,
                dropped_observations=len(processed) - processed_count,
                forward_calls=processed_count + 6,
                update_calls=processed_count,
                backward_calls=processed_count,
            ),
            latency=LatencyMeasurements(
                forward=_latencies(),
                update=_latencies(),
                backward=_latencies(),
            ),
            memory=MemoryHighWaterMarks(
                host_high_water_bytes=3_072,
                accelerator_high_water_bytes=None,
                persistent_state_bytes=1_024,
                parameter_count=8,
                measurement_method="test fixture: explicitly supplied measurements",
            ),
            energy=(
                EnergyMeasurement(
                    value=1.5,
                    unit="joule",
                    measurement_method="test fixture",
                )
                if candidate_diagnostics
                else EnergyMeasurement(None, None, None)
            ),
        ),
        safety=SafetyMeasurements(
            checks=len(processed),
            violations=1 if candidate_diagnostics else 0,
            interventions=1 if candidate_diagnostics else 0,
            near_misses=1 if candidate_diagnostics else 0,
            cumulative_cost=0.2 if candidate_diagnostics else 0.0,
            cumulative_near_miss_cost=(
                0.1 if candidate_diagnostics else 0.0
            ),
            maximum_step_cost=0.2 if candidate_diagnostics else 0.0,
        ),
        applicable_components=("toy_component",) if candidate_diagnostics else (),
        component_diagnostics=(
            {"toy_component": {"active_units": 2, "bounded": True}}
            if candidate_diagnostics
            else None
        ),
        plasticity_applicable=candidate_diagnostics,
        plasticity_diagnostics=(
            {
                "initial_segment_auc": 0.6,
                "recurrent_segment_auc": 0.75,
            }
            if candidate_diagnostics
            else None
        ),
    )


def _hand_computed_report() -> dict[str, object]:
    candidate = _condition(
        "candidate",
        scores=(0.2, 0.6, 1.0, 0.1, None, 0.9, 0.4, 0.8, 1.0),
        processed=(True, True, True, True, False, True, True, True, True),
        matrix=((0.9, 0.4), (0.5, 0.85), (0.6, 0.7)),
        candidate_diagnostics=True,
    )
    frozen = _condition(
        "frozen_baseline",
        scores=(0.5,) * 9,
        processed=(True,) * 9,
        matrix=((0.5, 0.5),) * 3,
    )
    fresh = _condition(
        "fresh_baseline",
        scores=(0.7,) * 9,
        processed=(True,) * 9,
        matrix=((0.7, 0.7),) * 3,
    )
    return build_continual_evaluation_report(
        protocol=_protocol(),
        candidate=candidate,
        baselines=(frozen, fresh),
    )


def _condition_record(
    report: dict[str, object],
    name: str,
) -> dict[str, object]:
    conditions = report["conditions"]
    assert isinstance(conditions, list)
    return next(
        condition
        for condition in conditions
        if isinstance(condition, dict) and condition["name"] == name
    )


@pytest.mark.unit
def test_hand_computed_metrics_and_raw_invariants() -> None:
    report = _hand_computed_report()
    candidate = _condition_record(report, "candidate")
    metrics = candidate["metrics"]
    operations = candidate["operations"]
    assert isinstance(metrics, dict)
    assert isinstance(operations, dict)

    assert report["schema_version"] == SCHEMA_VERSION
    assert report["acceptance_status"] == ACCEPTANCE_STATUS
    assert metrics["prequential_score"] == pytest.approx(0.625)
    assert metrics["lifetime_score"] == pytest.approx(5.0 / 9.0)

    adaptation = metrics["adaptation_auc"]
    assert isinstance(adaptation, dict)
    assert adaptation["mean_normalized_auc"] == pytest.approx((0.6 + 0.25 + 0.75) / 3.0)

    recovery = metrics["recovery"]
    assert isinstance(recovery, dict)
    assert recovery["event_count"] == 2
    assert recovery["recovered_count"] == 1
    assert recovery["recovery_rate"] == pytest.approx(0.5)
    assert recovery["mean_recovery_steps"] == pytest.approx(3.0)

    assert metrics["per_regime_final_performance"] == {"A": 0.6, "B": 0.7}
    forgetting = metrics["forgetting"]
    backward = metrics["backward_transfer"]
    forward = metrics["forward_transfer"]
    stability = metrics["stability"]
    worst = metrics["worst_window"]
    assert isinstance(forgetting, dict)
    assert isinstance(backward, dict)
    assert isinstance(forward, dict)
    assert isinstance(stability, dict)
    assert isinstance(worst, dict)
    assert forgetting["per_regime"] == pytest.approx({"A": 0.3, "B": 0.15})
    assert forgetting["mean"] == pytest.approx(0.225)
    assert backward["per_regime"] == pytest.approx({"A": -0.3, "B": -0.15})
    assert backward["mean"] == pytest.approx(-0.225)
    assert forward["per_regime"] == {"A": None, "B": pytest.approx(-0.1)}
    assert forward["mean_over_available"] == pytest.approx(-0.1)
    assert stability["mean_gap"] == pytest.approx(0.3)
    assert stability["maximum_gap"] == pytest.approx(0.8)
    assert worst == {
        "window_size": 2,
        "score": pytest.approx(0.05),
        "start_step": 3,
        "end_step_exclusive": 5,
    }

    counts = operations["counts"]
    assert isinstance(counts, dict)
    assert counts["processed_observations"] == 8
    assert counts["delayed_observations"] == 0
    assert counts["dropped_observations"] == 1
    assert counts["forward_calls"] == 14
    assert counts["update_calls"] == 8
    assert counts["backward_calls"] == 8
    latency = operations["latency"]
    assert isinstance(latency, dict)
    assert latency["update"] == {
        "p50_ms": 0.1,
        "p95_ms": 0.2,
        "p99_ms": 0.3,
    }
    safety = candidate["safety"]
    assert isinstance(safety, dict)
    assert safety["near_misses"] == 1
    assert safety["cumulative_near_miss_cost"] == pytest.approx(0.1)
    assert operations["energy"] == {
        "value": 1.5,
        "unit": "joule",
        "measurement_method": "test fixture",
    }
    assert validate_continual_evaluation_report(report).valid
    assert json.loads(continual_evaluation_report_json(report)) == report


@pytest.mark.unit
def test_learner_trace_api_has_no_regime_identifier() -> None:
    assert {field.name for field in fields(PredictBeforeUpdateTrace)} == {
        "scores",
        "processed",
    }
    assert all("regime" not in field.name for field in fields(PredictBeforeUpdateTrace))
    assert "regime_schedule" in {field.name for field in fields(ContinualEvaluationProtocol)}


@pytest.mark.unit
def test_strict_validator_rejects_missing_extra_and_modified_fields() -> None:
    report = _hand_computed_report()

    missing = deepcopy(report)
    condition = _condition_record(missing, "candidate")
    operations = condition["operations"]
    assert isinstance(operations, dict)
    del operations["memory"]
    validation = validate_continual_evaluation_report(missing)
    assert not validation.valid
    assert "missing=['memory']" in validation.errors[0]

    extra = deepcopy(report)
    extra["claimed_pass"] = True
    validation = validate_continual_evaluation_report(extra)
    assert not validation.valid
    assert "extra=['claimed_pass']" in validation.errors[0]

    modified = deepcopy(report)
    condition = _condition_record(modified, "candidate")
    metrics = condition["metrics"]
    assert isinstance(metrics, dict)
    metrics["lifetime_score"] = 1.0
    validation = validate_continual_evaluation_report(modified)
    assert not validation.valid
    assert "do not reconstruct" in validation.errors[0]


@pytest.mark.unit
def test_builder_requires_two_unique_exactly_matched_baselines() -> None:
    candidate = _condition(
        "candidate",
        scores=(0.5,) * 9,
        processed=(True,) * 9,
        matrix=((0.5, 0.5),) * 3,
    )
    baseline = replace(candidate, name="baseline")

    with pytest.raises(ValueError, match="at least two"):
        build_continual_evaluation_report(
            protocol=_protocol(),
            candidate=candidate,
            baselines=(baseline,),
        )

    with pytest.raises(ValueError, match="names must be unique"):
        build_continual_evaluation_report(
            protocol=_protocol(),
            candidate=candidate,
            baselines=(baseline, baseline),
        )

    mismatched = replace(
        baseline,
        name="mismatched",
        budget=replace(_budget(), parameter_count_limit=17),
    )
    with pytest.raises(ValueError, match="exactly match"):
        build_continual_evaluation_report(
            protocol=_protocol(),
            candidate=candidate,
            baselines=(baseline, mismatched),
        )


def _measured_latency(values_ns: list[int]) -> LatencySummary:
    if not values_ns:
        return LatencySummary(None, None, None)
    values_ms = sorted(value / 1_000_000.0 for value in values_ns)
    p50_index = math.ceil(0.50 * len(values_ms)) - 1
    p95_index = math.ceil(0.95 * len(values_ms)) - 1
    p99_index = math.ceil(0.99 * len(values_ms)) - 1
    return LatencySummary(
        p50_ms=values_ms[p50_index],
        p95_ms=values_ms[p95_index],
        p99_ms=values_ms[p99_index],
    )


def _run_toy_condition(
    name: str,
    *,
    mode: str,
    protocol: ContinualEvaluationProtocol,
    budget: MatchedBudget,
) -> ConditionEvaluationInput:
    targets = {"A": 1.0, "B": -1.0}
    state = {"weight": 0.0, "updates": 0}
    scores: list[float] = []
    matrix: list[tuple[float, ...]] = []
    forward_latencies: list[int] = []
    update_latencies: list[int] = []

    def score(target: float) -> float:
        start = perf_counter_ns()
        result = max(0.0, 1.0 - abs(state["weight"] - target) / 2.0)
        forward_latencies.append(perf_counter_ns() - start)
        return result

    tracemalloc.start()
    for step, regime_id in enumerate(protocol.regime_schedule, start=1):
        target = targets[regime_id]
        scores.append(score(target))

        update_start = perf_counter_ns()
        state["updates"] += 1
        if mode == "overwrite":
            state["weight"] = target
        elif mode == "running_mean":
            state["weight"] += (target - state["weight"]) / state["updates"]
        elif mode != "frozen":
            raise ValueError(f"unknown toy mode {mode!r}")
        update_latencies.append(perf_counter_ns() - update_start)

        if step in protocol.checkpoint_steps:
            matrix.append(tuple(score(targets[regime]) for regime in protocol.evaluator_regime_ids))
    _, host_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    persistent_bytes = sys.getsizeof(state) + sum(sys.getsizeof(value) for value in state.values())
    host_peak = max(host_peak, persistent_bytes)

    component_diagnostics = {
        "toy_scalar_learner": {
            "final_weight": state["weight"],
            "updates": state["updates"],
        }
    }
    plasticity_applicable = mode != "frozen"
    return ConditionEvaluationInput(
        name=name,
        trace=PredictBeforeUpdateTrace(
            scores=tuple(scores),
            processed=(True,) * len(scores),
        ),
        evaluator_matrix=EvaluatorMatrix(tuple(matrix)),
        budget=budget,
        operations=OperationalMeasurements(
            counts=OperationCounts(
                processed_observations=len(scores),
                delayed_observations=0,
                dropped_observations=0,
                forward_calls=len(forward_latencies),
                update_calls=len(update_latencies),
                backward_calls=0,
            ),
            latency=LatencyMeasurements(
                forward=_measured_latency(forward_latencies),
                update=_measured_latency(update_latencies),
                backward=_measured_latency([]),
            ),
            memory=MemoryHighWaterMarks(
                host_high_water_bytes=host_peak,
                accelerator_high_water_bytes=None,
                persistent_state_bytes=persistent_bytes,
                parameter_count=1,
                measurement_method=("tracemalloc peak; persistent state via sys.getsizeof"),
            ),
            energy=EnergyMeasurement(None, None, None),
        ),
        safety=SafetyMeasurements(
            checks=len(scores),
            violations=0,
            interventions=0,
            near_misses=0,
            cumulative_cost=0.0,
            cumulative_near_miss_cost=0.0,
            maximum_step_cost=0.0,
        ),
        applicable_components=("toy_scalar_learner",),
        component_diagnostics=component_diagnostics,
        plasticity_applicable=plasticity_applicable,
        plasticity_diagnostics=(
            {
                "first_segment_score": sum(scores[:3]) / 3.0,
                "last_segment_score": sum(scores[-3:]) / 3.0,
            }
            if plasticity_applicable
            else None
        ),
    )


@pytest.mark.integration
def test_end_to_end_deliberately_forgetting_toy_trace() -> None:
    protocol = _protocol()
    budget = MatchedBudget(
        observation_limit=9,
        forward_call_limit=30,
        update_call_limit=9,
        backward_call_limit=0,
        parameter_count_limit=2,
        persistent_state_bytes_limit=1_000_000,
    )
    candidate = _run_toy_condition(
        "overwrite_last_target",
        mode="overwrite",
        protocol=protocol,
        budget=budget,
    )
    frozen = _run_toy_condition(
        "frozen_zero",
        mode="frozen",
        protocol=protocol,
        budget=budget,
    )
    running = _run_toy_condition(
        "running_mean",
        mode="running_mean",
        protocol=protocol,
        budget=budget,
    )

    report = build_continual_evaluation_report(
        protocol=protocol,
        candidate=candidate,
        baselines=(frozen, running),
    )
    candidate_record = _condition_record(report, "overwrite_last_target")
    metrics = candidate_record["metrics"]
    assert isinstance(metrics, dict)
    forgetting = metrics["forgetting"]
    backward = metrics["backward_transfer"]
    assert isinstance(forgetting, dict)
    assert isinstance(backward, dict)
    assert forgetting["per_regime"] == {"A": 0.0, "B": 1.0}
    assert backward["per_regime"] == {"A": 0.0, "B": -1.0}
    assert report["acceptance_status"] == "not-assessed"
    assert validate_continual_evaluation_report(report).valid
