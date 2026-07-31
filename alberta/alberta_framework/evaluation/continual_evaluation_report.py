"""Strict v1 report builder for continual-learning comparisons.

The report is constructed from two deliberately separate evidence surfaces:

* :class:`PredictBeforeUpdateTrace` contains only learner-observable outcomes
  and processing status.  It has no task or regime identifier field.
* :class:`ContinualEvaluationProtocol` and :class:`EvaluatorMatrix` contain
  evaluator-only regime metadata and read-only probe results.

This separation keeps evaluator regime IDs out of the learner-input API.  The
builder requires one candidate and at least two named, exactly budget-matched
baselines.  It summarizes evidence; it does not decide scientific acceptance
and does not certify PrototypeAgent or completion of the Alberta Plan.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, NoReturn, TypeGuard, cast

SCHEMA_VERSION = "alberta.continual_evaluation_report.v1"
TRACE_ORDERING_VERSION = "alberta.predict_before_update_trace.v1"
ACCEPTANCE_STATUS = "not-assessed"
REPORT_INTERPRETATION = (
    "Strict descriptive continual-evaluation report; construction is not "
    "scientific acceptance, does not establish that PrototypeAgent passed, "
    "and is not an Alberta Plan completion certificate."
)
METRIC_DEFINITIONS: Mapping[str, str] = {
    "prequential_score": ("mean finite score emitted before update over processed observations"),
    "lifetime_score": (
        "mean score over every scheduled observation, replacing each explicit "
        "drop with protocol.dropped_observation_score"
    ),
    "adaptation_auc": (
        "normalized trapezoid AUC within each contiguous regime segment over lifetime scores"
    ),
    "recovery": (
        "first sustained threshold window after each regime change, bounded by the next change"
    ),
    "forgetting": ("direction-normalized peak-to-final degradation after first exposure"),
    "backward_transfer": ("direction-normalized final minus first-post-exposure performance"),
    "forward_transfer": ("direction-normalized last pre-exposure probe minus frozen reference"),
    "stability": "non-negative deficit from the per-regime reference",
    "worst_window": ("worst rolling lifetime-score mean in the configured metric direction"),
}

type DiagnosticScalar = bool | int | float | str
type ConditionRole = Literal["candidate", "baseline"]


def _is_finite_number(value: object) -> TypeGuard[int | float]:
    return (
        not isinstance(value, bool)
        and isinstance(value, int | float)
        and math.isfinite(float(value))
    )


def _require_nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _require_positive_int(value: object, *, name: str) -> int:
    resolved = _require_nonnegative_int(value, name=name)
    if resolved == 0:
        raise ValueError(f"{name} must be positive")
    return resolved


def _require_nonnegative_float(value: object, *, name: str) -> float:
    if not _is_finite_number(value) or float(value) < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return float(value)


def _require_finite_float(value: object, *, name: str) -> float:
    if not _is_finite_number(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _validate_named_float_mapping(
    value: Mapping[str, float],
    *,
    expected_names: tuple[str, ...],
    name: str,
) -> None:
    if set(value) != set(expected_names):
        raise ValueError(f"{name} keys must exactly match evaluator_regime_ids")
    for key, item in value.items():
        _require_finite_float(item, name=f"{name}.{key}")


def _validate_diagnostic_value(
    value: object,
    *,
    location: str,
) -> DiagnosticScalar:
    if isinstance(value, bool | int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{location} must be finite")
        return value
    if isinstance(value, str) and value:
        return value
    raise ValueError(f"{location} must be a non-empty string, boolean, integer, or finite float")


def _validate_diagnostic_mapping(
    value: Mapping[str, DiagnosticScalar],
    *,
    location: str,
) -> None:
    if not value:
        raise ValueError(f"{location} must be non-empty")
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{location} keys must be non-empty strings")
        _validate_diagnostic_value(item, location=f"{location}.{key}")


@dataclass(frozen=True)
class ContinualEvaluationProtocol:
    """Evaluator-only schedule and frozen metric policy.

    Regime identifiers belong here, never in
    :class:`PredictBeforeUpdateTrace`.
    """

    protocol_id: str
    higher_is_better: bool
    regime_schedule: tuple[str, ...]
    evaluator_regime_ids: tuple[str, ...]
    checkpoint_steps: tuple[int, ...]
    first_exposure_checkpoint: Mapping[str, int]
    forward_transfer_reference: Mapping[str, float]
    recovery_thresholds: Mapping[str, float]
    stability_references: Mapping[str, float]
    recovery_window: int
    worst_window_size: int
    dropped_observation_score: float

    def __post_init__(self) -> None:
        if not isinstance(self.protocol_id, str) or ".v" not in self.protocol_id:
            raise ValueError("protocol_id must be a non-empty versioned identifier")
        if not isinstance(self.higher_is_better, bool):
            raise ValueError("higher_is_better must be boolean")
        if len(self.regime_schedule) < 2:
            raise ValueError("regime_schedule must contain at least two observations")
        if any(not isinstance(item, str) or not item for item in self.regime_schedule):
            raise ValueError("regime_schedule IDs must be non-empty strings")
        if (
            len(self.evaluator_regime_ids) < 2
            or len(set(self.evaluator_regime_ids)) != len(self.evaluator_regime_ids)
            or any(not isinstance(item, str) or not item for item in self.evaluator_regime_ids)
        ):
            raise ValueError("evaluator_regime_ids must contain at least two unique names")
        if set(self.regime_schedule) != set(self.evaluator_regime_ids):
            raise ValueError("regime_schedule IDs must exactly match evaluator_regime_ids")
        if all(
            left == right
            for left, right in zip(
                self.regime_schedule,
                self.regime_schedule[1:],
                strict=False,
            )
        ):
            raise ValueError("regime_schedule must contain at least one change")
        if len(self.checkpoint_steps) < 2:
            raise ValueError("checkpoint_steps must contain at least two checkpoints")
        if any(
            isinstance(step, bool) or not isinstance(step, int) or step <= 0
            for step in self.checkpoint_steps
        ):
            raise ValueError("checkpoint_steps must contain positive integers")
        if any(
            later <= earlier
            for earlier, later in zip(
                self.checkpoint_steps,
                self.checkpoint_steps[1:],
                strict=False,
            )
        ):
            raise ValueError("checkpoint_steps must be strictly increasing")
        if self.checkpoint_steps[-1] != len(self.regime_schedule):
            raise ValueError("the final checkpoint must equal the trace length")
        if set(self.first_exposure_checkpoint) != set(self.evaluator_regime_ids):
            raise ValueError(
                "first_exposure_checkpoint keys must exactly match evaluator_regime_ids"
            )
        exposure_rows: list[int] = []
        for regime_id, row in self.first_exposure_checkpoint.items():
            if (
                isinstance(row, bool)
                or not isinstance(row, int)
                or not 0 <= row < len(self.checkpoint_steps)
            ):
                raise ValueError(f"first_exposure_checkpoint.{regime_id} must index checkpoints")
            exposure_rows.append(row)
        if not any(row > 0 for row in exposure_rows):
            raise ValueError("at least one regime must have a pre-exposure checkpoint for FWT")
        _validate_named_float_mapping(
            self.forward_transfer_reference,
            expected_names=self.evaluator_regime_ids,
            name="forward_transfer_reference",
        )
        _validate_named_float_mapping(
            self.recovery_thresholds,
            expected_names=self.evaluator_regime_ids,
            name="recovery_thresholds",
        )
        _validate_named_float_mapping(
            self.stability_references,
            expected_names=self.evaluator_regime_ids,
            name="stability_references",
        )
        _require_positive_int(self.recovery_window, name="recovery_window")
        _require_positive_int(self.worst_window_size, name="worst_window_size")
        if self.worst_window_size > len(self.regime_schedule):
            raise ValueError("worst_window_size cannot exceed the trace length")
        _require_finite_float(
            self.dropped_observation_score,
            name="dropped_observation_score",
        )


@dataclass(frozen=True)
class PredictBeforeUpdateTrace:
    """Learner-facing trace with no evaluator regime IDs."""

    scores: tuple[float | None, ...]
    processed: tuple[bool, ...]

    def __post_init__(self) -> None:
        if not self.scores or len(self.scores) != len(self.processed):
            raise ValueError("scores and processed must be non-empty and aligned")
        for index, (score, was_processed) in enumerate(
            zip(self.scores, self.processed, strict=True)
        ):
            if not isinstance(was_processed, bool):
                raise ValueError(f"processed[{index}] must be boolean")
            if was_processed:
                if not _is_finite_number(score):
                    raise ValueError(f"processed trace score {index} must be a finite number")
            elif score is not None:
                raise ValueError(f"dropped trace score {index} must be None, not a hidden metric")
        if not any(self.processed):
            raise ValueError("trace must contain at least one processed observation")


@dataclass(frozen=True)
class EvaluatorMatrix:
    """Read-only checkpoint-by-regime performance matrix."""

    values: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        if not self.values or any(not row for row in self.values):
            raise ValueError("evaluator matrix must be non-empty and rectangular")
        width = len(self.values[0])
        if any(len(row) != width for row in self.values):
            raise ValueError("evaluator matrix must be rectangular")
        for row_index, row in enumerate(self.values):
            for column_index, value in enumerate(row):
                _require_finite_float(
                    value,
                    name=f"evaluator_matrix[{row_index}][{column_index}]",
                )


@dataclass(frozen=True)
class MatchedBudget:
    """Frozen comparison budget shared exactly by candidate and baselines."""

    observation_limit: int
    forward_call_limit: int
    update_call_limit: int
    backward_call_limit: int
    parameter_count_limit: int
    persistent_state_bytes_limit: int

    def __post_init__(self) -> None:
        _require_positive_int(self.observation_limit, name="observation_limit")
        _require_positive_int(self.forward_call_limit, name="forward_call_limit")
        _require_nonnegative_int(self.update_call_limit, name="update_call_limit")
        _require_nonnegative_int(self.backward_call_limit, name="backward_call_limit")
        _require_positive_int(self.parameter_count_limit, name="parameter_count_limit")
        _require_positive_int(
            self.persistent_state_bytes_limit,
            name="persistent_state_bytes_limit",
        )


@dataclass(frozen=True)
class OperationCounts:
    """Observed counts; the builder never synthesizes missing resource data.

    ``delayed_observations`` is the subset of processed observations whose
    decision or update missed the protocol's declared latency deadline.
    """

    processed_observations: int
    delayed_observations: int
    dropped_observations: int
    forward_calls: int
    update_calls: int
    backward_calls: int

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            _require_nonnegative_int(value, name=name)
        if self.delayed_observations > self.processed_observations:
            raise ValueError(
                "delayed_observations must be a subset of processed_observations"
            )


@dataclass(frozen=True)
class LatencySummary:
    """Measured latency quantiles for one operation, or an explicit absence."""

    p50_ms: float | None
    p95_ms: float | None
    p99_ms: float | None

    def __post_init__(self) -> None:
        values = (self.p50_ms, self.p95_ms, self.p99_ms)
        if all(value is None for value in values):
            return
        if any(value is None for value in values):
            raise ValueError("latency fields must be all measured or all None")
        resolved = tuple(_require_nonnegative_float(value, name="latency") for value in values)
        if not resolved[0] <= resolved[1] <= resolved[2]:
            raise ValueError("latency must satisfy p50 <= p95 <= p99")

    @property
    def measured(self) -> bool:
        return self.p50_ms is not None


@dataclass(frozen=True)
class LatencyMeasurements:
    forward: LatencySummary
    update: LatencySummary
    backward: LatencySummary


@dataclass(frozen=True)
class MemoryHighWaterMarks:
    """Explicit measured memory high-water marks and persistent state size."""

    host_high_water_bytes: int
    accelerator_high_water_bytes: int | None
    persistent_state_bytes: int
    parameter_count: int
    measurement_method: str

    def __post_init__(self) -> None:
        _require_positive_int(
            self.host_high_water_bytes,
            name="host_high_water_bytes",
        )
        if self.accelerator_high_water_bytes is not None:
            _require_nonnegative_int(
                self.accelerator_high_water_bytes,
                name="accelerator_high_water_bytes",
            )
        _require_nonnegative_int(
            self.persistent_state_bytes,
            name="persistent_state_bytes",
        )
        _require_nonnegative_int(
            self.parameter_count,
            name="parameter_count",
        )
        if not isinstance(self.measurement_method, str) or not self.measurement_method:
            raise ValueError("measurement_method must be a non-empty string")


@dataclass(frozen=True)
class EnergyMeasurement:
    """Measured energy proxy, or an explicit unavailable triplet.

    The report does not equate unlike units. A measured value therefore
    carries its unit and measurement method verbatim.
    """

    value: float | None
    unit: str | None
    measurement_method: str | None

    def __post_init__(self) -> None:
        values = (self.value, self.unit, self.measurement_method)
        if all(value is None for value in values):
            return
        if any(value is None for value in values):
            raise ValueError("energy fields must be all measured or all None")
        _require_nonnegative_float(self.value, name="energy.value")
        if not isinstance(self.unit, str) or not self.unit:
            raise ValueError("energy.unit must be a non-empty string")
        if (
            not isinstance(self.measurement_method, str)
            or not self.measurement_method
        ):
            raise ValueError(
                "energy.measurement_method must be a non-empty string"
            )

    @property
    def measured(self) -> bool:
        return self.value is not None


@dataclass(frozen=True)
class OperationalMeasurements:
    counts: OperationCounts
    latency: LatencyMeasurements
    memory: MemoryHighWaterMarks
    energy: EnergyMeasurement


@dataclass(frozen=True)
class SafetyMeasurements:
    """Observed hard violations, interventions, and graded near misses."""

    checks: int
    violations: int
    interventions: int
    near_misses: int
    cumulative_cost: float
    cumulative_near_miss_cost: float
    maximum_step_cost: float

    def __post_init__(self) -> None:
        checks = _require_nonnegative_int(self.checks, name="safety.checks")
        violations = _require_nonnegative_int(
            self.violations,
            name="safety.violations",
        )
        interventions = _require_nonnegative_int(
            self.interventions,
            name="safety.interventions",
        )
        near_misses = _require_nonnegative_int(
            self.near_misses,
            name="safety.near_misses",
        )
        if violations > checks or interventions > checks or near_misses > checks:
            raise ValueError(
                "safety violations/interventions/near_misses cannot exceed checks"
            )
        cumulative = _require_nonnegative_float(
            self.cumulative_cost,
            name="safety.cumulative_cost",
        )
        near_miss_cost = _require_nonnegative_float(
            self.cumulative_near_miss_cost,
            name="safety.cumulative_near_miss_cost",
        )
        maximum = _require_nonnegative_float(
            self.maximum_step_cost,
            name="safety.maximum_step_cost",
        )
        if maximum > cumulative and checks > 0:
            raise ValueError("maximum_step_cost cannot exceed cumulative_cost")
        if near_miss_cost > cumulative:
            raise ValueError("cumulative_near_miss_cost cannot exceed cumulative_cost")


@dataclass(frozen=True)
class ConditionEvaluationInput:
    """Complete raw and operational evidence for one comparison condition."""

    name: str
    trace: PredictBeforeUpdateTrace
    evaluator_matrix: EvaluatorMatrix
    budget: MatchedBudget
    operations: OperationalMeasurements
    safety: SafetyMeasurements
    applicable_components: tuple[str, ...]
    component_diagnostics: Mapping[str, Mapping[str, DiagnosticScalar]] | None
    plasticity_applicable: bool
    plasticity_diagnostics: Mapping[str, DiagnosticScalar] | None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("condition name must be a non-empty string")
        if len(set(self.applicable_components)) != len(self.applicable_components) or any(
            not isinstance(component, str) or not component
            for component in self.applicable_components
        ):
            raise ValueError("applicable_components must contain unique names")
        if self.applicable_components:
            if self.component_diagnostics is None:
                raise ValueError("component diagnostics are required for applicable components")
            if set(self.component_diagnostics) != set(self.applicable_components):
                raise ValueError(
                    "component diagnostic keys must exactly match applicable_components"
                )
            for component, diagnostics in self.component_diagnostics.items():
                _validate_diagnostic_mapping(
                    diagnostics,
                    location=f"component_diagnostics.{component}",
                )
        elif self.component_diagnostics is not None:
            raise ValueError("component_diagnostics must be None when no components apply")
        if not isinstance(self.plasticity_applicable, bool):
            raise ValueError("plasticity_applicable must be boolean")
        if self.plasticity_applicable:
            if self.plasticity_diagnostics is None:
                raise ValueError("plasticity diagnostics are required when plasticity applies")
            _validate_diagnostic_mapping(
                self.plasticity_diagnostics,
                location="plasticity_diagnostics",
            )
        elif self.plasticity_diagnostics is not None:
            raise ValueError("plasticity_diagnostics must be None when plasticity does not apply")


@dataclass(frozen=True)
class ReportValidation:
    valid: bool
    errors: tuple[str, ...]


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot summarize an empty metric vector")
    return float(sum(values) / len(values))


def _effective_scores(
    trace: PredictBeforeUpdateTrace,
    *,
    dropped_score: float,
) -> list[float]:
    values: list[float] = []
    for index, (score, processed) in enumerate(zip(trace.scores, trace.processed, strict=True)):
        values.append(
            _require_finite_float(score, name=f"trace.scores[{index}]")
            if processed
            else dropped_score
        )
    return values


def _contiguous_segments(
    schedule: tuple[str, ...],
) -> list[tuple[str, int, int]]:
    segments: list[tuple[str, int, int]] = []
    start = 0
    for index in range(1, len(schedule)):
        if schedule[index] != schedule[start]:
            segments.append((schedule[start], start, index))
            start = index
    segments.append((schedule[start], start, len(schedule)))
    return segments


def _normalized_trapezoid_auc(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("adaptation segment must be non-empty")
    if len(values) == 1:
        return float(values[0])
    area = sum(0.5 * (left + right) for left, right in zip(values, values[1:], strict=False))
    return float(area / (len(values) - 1))


def _adaptation_metrics(
    scores: Sequence[float],
    protocol: ContinualEvaluationProtocol,
) -> dict[str, object]:
    records: list[dict[str, object]] = []
    auc_values: list[float] = []
    for segment_index, (regime_id, start, stop) in enumerate(
        _contiguous_segments(protocol.regime_schedule)
    ):
        auc = _normalized_trapezoid_auc(scores[start:stop])
        auc_values.append(auc)
        records.append(
            {
                "segment_index": segment_index,
                "regime_id": regime_id,
                "start_step": start,
                "end_step_exclusive": stop,
                "normalized_auc": auc,
            }
        )
    return {
        "mean_normalized_auc": _mean(auc_values),
        "segments": records,
    }


def _recovery_metrics(
    scores: Sequence[float],
    protocol: ContinualEvaluationProtocol,
) -> dict[str, object]:
    events: list[dict[str, object]] = []
    recovered_lengths: list[int] = []
    segments = _contiguous_segments(protocol.regime_schedule)
    for segment_index, (regime_id, start, stop) in enumerate(segments[1:], start=1):
        threshold = float(protocol.recovery_thresholds[regime_id])
        recovery_steps: int | None = None
        last_window_start = stop - protocol.recovery_window
        for window_start in range(start, last_window_start + 1):
            window = scores[window_start : window_start + protocol.recovery_window]
            meets = (
                all(value >= threshold for value in window)
                if protocol.higher_is_better
                else all(value <= threshold for value in window)
            )
            if meets:
                recovery_steps = window_start + protocol.recovery_window - start
                recovered_lengths.append(recovery_steps)
                break
        events.append(
            {
                "segment_index": segment_index,
                "regime_id": regime_id,
                "start_step": start,
                "end_step_exclusive": stop,
                "threshold": threshold,
                "window_size": protocol.recovery_window,
                "recovered": recovery_steps is not None,
                "recovery_steps": recovery_steps,
            }
        )
    return {
        "event_count": len(events),
        "recovered_count": len(recovered_lengths),
        "recovery_rate": len(recovered_lengths) / len(events),
        "mean_recovery_steps": (
            _mean([float(value) for value in recovered_lengths]) if recovered_lengths else None
        ),
        "maximum_recovery_steps": (max(recovered_lengths) if recovered_lengths else None),
        "events": events,
    }


def _signed_improvement(
    later: float,
    earlier: float,
    *,
    higher_is_better: bool,
) -> float:
    return later - earlier if higher_is_better else earlier - later


def _evaluator_metrics(
    matrix: EvaluatorMatrix,
    protocol: ContinualEvaluationProtocol,
) -> dict[str, object]:
    per_regime_final: dict[str, float] = {}
    per_regime_forgetting: dict[str, float] = {}
    per_regime_backward: dict[str, float] = {}
    per_regime_forward: dict[str, float | None] = {}
    forward_values: list[float] = []

    for column, regime_id in enumerate(protocol.evaluator_regime_ids):
        first_row = int(protocol.first_exposure_checkpoint[regime_id])
        trajectory = [float(row[column]) for row in matrix.values[first_row:]]
        final = trajectory[-1]
        best = max(trajectory) if protocol.higher_is_better else min(trajectory)
        degradation = best - final if protocol.higher_is_better else final - best
        per_regime_final[regime_id] = final
        per_regime_forgetting[regime_id] = max(0.0, degradation)
        per_regime_backward[regime_id] = _signed_improvement(
            final,
            trajectory[0],
            higher_is_better=protocol.higher_is_better,
        )

        if first_row == 0:
            per_regime_forward[regime_id] = None
        else:
            pre_exposure = float(matrix.values[first_row - 1][column])
            forward = _signed_improvement(
                pre_exposure,
                float(protocol.forward_transfer_reference[regime_id]),
                higher_is_better=protocol.higher_is_better,
            )
            per_regime_forward[regime_id] = forward
            forward_values.append(forward)

    forgetting_values = list(per_regime_forgetting.values())
    backward_values = list(per_regime_backward.values())
    return {
        "per_regime_final_performance": per_regime_final,
        "mean_final_performance": _mean(list(per_regime_final.values())),
        "forgetting": {
            "mean": _mean(forgetting_values),
            "maximum": max(forgetting_values),
            "per_regime": per_regime_forgetting,
        },
        "backward_transfer": {
            "mean": _mean(backward_values),
            "per_regime": per_regime_backward,
        },
        "forward_transfer": {
            "mean_over_available": (_mean(forward_values) if forward_values else None),
            "available_regime_count": len(forward_values),
            "per_regime": per_regime_forward,
        },
    }


def _stability_metrics(
    scores: Sequence[float],
    protocol: ContinualEvaluationProtocol,
) -> dict[str, float]:
    gaps = [
        max(
            0.0,
            (
                float(protocol.stability_references[regime_id]) - score
                if protocol.higher_is_better
                else score - float(protocol.stability_references[regime_id])
            ),
        )
        for score, regime_id in zip(
            scores,
            protocol.regime_schedule,
            strict=True,
        )
    ]
    return {
        "mean_gap": _mean(gaps),
        "maximum_gap": max(gaps),
    }


def _worst_window_metrics(
    scores: Sequence[float],
    protocol: ContinualEvaluationProtocol,
) -> dict[str, int | float]:
    window = protocol.worst_window_size
    means = [_mean(scores[start : start + window]) for start in range(len(scores) - window + 1)]
    if protocol.higher_is_better:
        worst_score = min(means)
    else:
        worst_score = max(means)
    start = means.index(worst_score)
    return {
        "window_size": window,
        "score": worst_score,
        "start_step": start,
        "end_step_exclusive": start + window,
    }


def _validate_condition_against_protocol(
    condition: ConditionEvaluationInput,
    protocol: ContinualEvaluationProtocol,
) -> None:
    trace_length = len(protocol.regime_schedule)
    if len(condition.trace.scores) != trace_length:
        raise ValueError(f"{condition.name}: trace length must match regime_schedule")
    if condition.budget.observation_limit != trace_length:
        raise ValueError(f"{condition.name}: observation budget must equal trace length")
    if len(condition.evaluator_matrix.values) != len(protocol.checkpoint_steps):
        raise ValueError(f"{condition.name}: evaluator rows must match checkpoint_steps")
    if any(
        len(row) != len(protocol.evaluator_regime_ids) for row in condition.evaluator_matrix.values
    ):
        raise ValueError(f"{condition.name}: evaluator columns must match evaluator_regime_ids")

    counts = condition.operations.counts
    observed_processed = sum(condition.trace.processed)
    observed_dropped = trace_length - observed_processed
    if counts.processed_observations != observed_processed:
        raise ValueError(f"{condition.name}: processed observation count does not match trace")
    if counts.dropped_observations != observed_dropped:
        raise ValueError(f"{condition.name}: dropped observation count does not match trace")
    if counts.forward_calls < observed_processed:
        raise ValueError(f"{condition.name}: forward_calls cannot be below processed observations")
    budget_checks = (
        (
            counts.forward_calls,
            condition.budget.forward_call_limit,
            "forward_calls",
        ),
        (
            counts.update_calls,
            condition.budget.update_call_limit,
            "update_calls",
        ),
        (
            counts.backward_calls,
            condition.budget.backward_call_limit,
            "backward_calls",
        ),
    )
    for actual, limit, name in budget_checks:
        if actual > limit:
            raise ValueError(f"{condition.name}: {name} exceeds the matched budget")

    latency_checks = (
        (counts.forward_calls, condition.operations.latency.forward, "forward"),
        (counts.update_calls, condition.operations.latency.update, "update"),
        (
            counts.backward_calls,
            condition.operations.latency.backward,
            "backward",
        ),
    )
    for count, latency, operation in latency_checks:
        if (count > 0) != latency.measured:
            raise ValueError(f"{condition.name}: {operation} latency measurement/count mismatch")

    memory = condition.operations.memory
    if memory.persistent_state_bytes > condition.budget.persistent_state_bytes_limit:
        raise ValueError(f"{condition.name}: persistent state exceeds the matched budget")
    if memory.parameter_count > condition.budget.parameter_count_limit:
        raise ValueError(f"{condition.name}: parameter count exceeds the matched budget")
    if memory.persistent_state_bytes > memory.host_high_water_bytes:
        raise ValueError(f"{condition.name}: persistent state exceeds host memory high-water mark")


def _json_clone(value: object) -> object:
    return json.loads(
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
    )


def _protocol_payload(
    protocol: ContinualEvaluationProtocol,
) -> dict[str, object]:
    return {
        "protocol_id": protocol.protocol_id,
        "higher_is_better": protocol.higher_is_better,
        "regime_schedule": list(protocol.regime_schedule),
        "evaluator_regime_ids": list(protocol.evaluator_regime_ids),
        "checkpoint_steps": list(protocol.checkpoint_steps),
        "first_exposure_checkpoint": dict(protocol.first_exposure_checkpoint),
        "forward_transfer_reference": dict(protocol.forward_transfer_reference),
        "recovery_thresholds": dict(protocol.recovery_thresholds),
        "stability_references": dict(protocol.stability_references),
        "recovery_window": protocol.recovery_window,
        "worst_window_size": protocol.worst_window_size,
        "dropped_observation_score": protocol.dropped_observation_score,
    }


def _latency_payload(latency: LatencySummary) -> dict[str, float | None]:
    return {
        "p50_ms": latency.p50_ms,
        "p95_ms": latency.p95_ms,
        "p99_ms": latency.p99_ms,
    }


def _condition_payload(
    condition: ConditionEvaluationInput,
    *,
    role: ConditionRole,
    protocol: ContinualEvaluationProtocol,
) -> dict[str, object]:
    effective_scores = _effective_scores(
        condition.trace,
        dropped_score=protocol.dropped_observation_score,
    )
    processed_scores = [
        float(score)
        for score, processed in zip(
            condition.trace.scores,
            condition.trace.processed,
            strict=True,
        )
        if processed and score is not None
    ]
    evaluator = _evaluator_metrics(condition.evaluator_matrix, protocol)
    metrics: dict[str, object] = {
        "prequential_score": _mean(processed_scores),
        "lifetime_score": _mean(effective_scores),
        "adaptation_auc": _adaptation_metrics(effective_scores, protocol),
        "recovery": _recovery_metrics(effective_scores, protocol),
        "per_regime_final_performance": evaluator["per_regime_final_performance"],
        "mean_final_performance": evaluator["mean_final_performance"],
        "forgetting": evaluator["forgetting"],
        "backward_transfer": evaluator["backward_transfer"],
        "forward_transfer": evaluator["forward_transfer"],
        "stability": _stability_metrics(effective_scores, protocol),
        "worst_window": _worst_window_metrics(effective_scores, protocol),
    }
    operations = condition.operations
    return {
        "name": condition.name,
        "role": role,
        "metrics": metrics,
        "operations": {
            "counts": asdict(operations.counts),
            "latency": {
                "forward": _latency_payload(operations.latency.forward),
                "update": _latency_payload(operations.latency.update),
                "backward": _latency_payload(operations.latency.backward),
            },
            "memory": asdict(operations.memory),
            "energy": asdict(operations.energy),
        },
        "safety": asdict(condition.safety),
        "diagnostics": {
            "applicable_components": list(condition.applicable_components),
            "component_diagnostics": (
                _json_clone(condition.component_diagnostics)
                if condition.component_diagnostics is not None
                else None
            ),
            "plasticity_applicable": condition.plasticity_applicable,
            "plasticity_diagnostics": (
                _json_clone(condition.plasticity_diagnostics)
                if condition.plasticity_diagnostics is not None
                else None
            ),
        },
        "evidence": {
            "predict_before_update_trace": {
                "ordering": TRACE_ORDERING_VERSION,
                "scores": list(condition.trace.scores),
                "processed": list(condition.trace.processed),
            },
            "per_regime_evaluator": {
                "regime_ids": list(protocol.evaluator_regime_ids),
                "checkpoint_steps": list(protocol.checkpoint_steps),
                "values": [list(row) for row in condition.evaluator_matrix.values],
            },
        },
    }


def build_continual_evaluation_report(
    *,
    protocol: ContinualEvaluationProtocol,
    candidate: ConditionEvaluationInput,
    baselines: Sequence[ConditionEvaluationInput],
) -> dict[str, object]:
    """Build one strict descriptive report without inventing measurements."""

    baseline_tuple = tuple(baselines)
    if len(baseline_tuple) < 2:
        raise ValueError("at least two named matched-budget baselines are required")
    all_conditions = (candidate, *baseline_tuple)
    names = tuple(condition.name for condition in all_conditions)
    if len(set(names)) != len(names):
        raise ValueError("candidate and baseline names must be unique")
    if any(condition.budget != candidate.budget for condition in baseline_tuple):
        raise ValueError("all baselines must exactly match the candidate budget")
    for condition in all_conditions:
        _validate_condition_against_protocol(condition, protocol)

    condition_payloads = [
        _condition_payload(candidate, role="candidate", protocol=protocol),
        *[
            _condition_payload(
                baseline,
                role="baseline",
                protocol=protocol,
            )
            for baseline in baseline_tuple
        ],
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "trace_ordering_version": TRACE_ORDERING_VERSION,
        "acceptance_status": ACCEPTANCE_STATUS,
        "interpretation": REPORT_INTERPRETATION,
        "metric_definitions": dict(METRIC_DEFINITIONS),
        "protocol": _protocol_payload(protocol),
        "comparison_contract": {
            "candidate": candidate.name,
            "baselines": [baseline.name for baseline in baseline_tuple],
            "baseline_count": len(baseline_tuple),
            "matched_budget": asdict(candidate.budget),
            "budget_match_verified": True,
        },
        "conditions": condition_payloads,
    }


def _expect_mapping(value: object, *, location: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{location} must be an object with string keys")
    return value


def _expect_list(value: object, *, location: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{location} must be an array")
    return value


def _exact_keys(
    value: Mapping[str, object],
    expected: set[str],
    *,
    location: str,
) -> None:
    observed = set(value)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(f"{location} keys do not match schema: missing={missing}, extra={extra}")


def _expect_string(value: object, *, location: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{location} must be a non-empty string")
    return value


def _expect_bool(value: object, *, location: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{location} must be boolean")
    return value


def _expect_int(value: object, *, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{location} must be an integer")
    return value


def _expect_float(value: object, *, location: str) -> float:
    return _require_finite_float(value, name=location)


def _expect_optional_float(value: object, *, location: str) -> float | None:
    if value is None:
        return None
    return _expect_float(value, location=location)


def _parse_float_mapping(value: object, *, location: str) -> dict[str, float]:
    mapping = _expect_mapping(value, location=location)
    return {key: _expect_float(item, location=f"{location}.{key}") for key, item in mapping.items()}


def _parse_int_mapping(value: object, *, location: str) -> dict[str, int]:
    mapping = _expect_mapping(value, location=location)
    return {key: _expect_int(item, location=f"{location}.{key}") for key, item in mapping.items()}


def _parse_protocol(value: object) -> ContinualEvaluationProtocol:
    location = "protocol"
    mapping = _expect_mapping(value, location=location)
    _exact_keys(
        mapping,
        {
            "protocol_id",
            "higher_is_better",
            "regime_schedule",
            "evaluator_regime_ids",
            "checkpoint_steps",
            "first_exposure_checkpoint",
            "forward_transfer_reference",
            "recovery_thresholds",
            "stability_references",
            "recovery_window",
            "worst_window_size",
            "dropped_observation_score",
        },
        location=location,
    )
    regime_schedule = tuple(
        _expect_string(item, location=f"{location}.regime_schedule")
        for item in _expect_list(
            mapping["regime_schedule"],
            location=f"{location}.regime_schedule",
        )
    )
    evaluator_regime_ids = tuple(
        _expect_string(item, location=f"{location}.evaluator_regime_ids")
        for item in _expect_list(
            mapping["evaluator_regime_ids"],
            location=f"{location}.evaluator_regime_ids",
        )
    )
    checkpoint_steps = tuple(
        _expect_int(item, location=f"{location}.checkpoint_steps")
        for item in _expect_list(
            mapping["checkpoint_steps"],
            location=f"{location}.checkpoint_steps",
        )
    )
    return ContinualEvaluationProtocol(
        protocol_id=_expect_string(
            mapping["protocol_id"],
            location=f"{location}.protocol_id",
        ),
        higher_is_better=_expect_bool(
            mapping["higher_is_better"],
            location=f"{location}.higher_is_better",
        ),
        regime_schedule=regime_schedule,
        evaluator_regime_ids=evaluator_regime_ids,
        checkpoint_steps=checkpoint_steps,
        first_exposure_checkpoint=_parse_int_mapping(
            mapping["first_exposure_checkpoint"],
            location=f"{location}.first_exposure_checkpoint",
        ),
        forward_transfer_reference=_parse_float_mapping(
            mapping["forward_transfer_reference"],
            location=f"{location}.forward_transfer_reference",
        ),
        recovery_thresholds=_parse_float_mapping(
            mapping["recovery_thresholds"],
            location=f"{location}.recovery_thresholds",
        ),
        stability_references=_parse_float_mapping(
            mapping["stability_references"],
            location=f"{location}.stability_references",
        ),
        recovery_window=_expect_int(
            mapping["recovery_window"],
            location=f"{location}.recovery_window",
        ),
        worst_window_size=_expect_int(
            mapping["worst_window_size"],
            location=f"{location}.worst_window_size",
        ),
        dropped_observation_score=_expect_float(
            mapping["dropped_observation_score"],
            location=f"{location}.dropped_observation_score",
        ),
    )


def _parse_budget(value: object) -> MatchedBudget:
    location = "comparison_contract.matched_budget"
    mapping = _expect_mapping(value, location=location)
    fields = {
        "observation_limit",
        "forward_call_limit",
        "update_call_limit",
        "backward_call_limit",
        "parameter_count_limit",
        "persistent_state_bytes_limit",
    }
    _exact_keys(mapping, fields, location=location)
    return MatchedBudget(
        observation_limit=_expect_int(
            mapping["observation_limit"],
            location=f"{location}.observation_limit",
        ),
        forward_call_limit=_expect_int(
            mapping["forward_call_limit"],
            location=f"{location}.forward_call_limit",
        ),
        update_call_limit=_expect_int(
            mapping["update_call_limit"],
            location=f"{location}.update_call_limit",
        ),
        backward_call_limit=_expect_int(
            mapping["backward_call_limit"],
            location=f"{location}.backward_call_limit",
        ),
        parameter_count_limit=_expect_int(
            mapping["parameter_count_limit"],
            location=f"{location}.parameter_count_limit",
        ),
        persistent_state_bytes_limit=_expect_int(
            mapping["persistent_state_bytes_limit"],
            location=f"{location}.persistent_state_bytes_limit",
        ),
    )


def _parse_latency(value: object, *, location: str) -> LatencySummary:
    mapping = _expect_mapping(value, location=location)
    _exact_keys(
        mapping,
        {"p50_ms", "p95_ms", "p99_ms"},
        location=location,
    )
    return LatencySummary(
        p50_ms=_expect_optional_float(
            mapping["p50_ms"],
            location=f"{location}.p50_ms",
        ),
        p95_ms=_expect_optional_float(
            mapping["p95_ms"],
            location=f"{location}.p95_ms",
        ),
        p99_ms=_expect_optional_float(
            mapping["p99_ms"],
            location=f"{location}.p99_ms",
        ),
    )


def _parse_operations(value: object, *, location: str) -> OperationalMeasurements:
    mapping = _expect_mapping(value, location=location)
    _exact_keys(
        mapping,
        {"counts", "latency", "memory", "energy"},
        location=location,
    )

    counts_value = _expect_mapping(
        mapping["counts"],
        location=f"{location}.counts",
    )
    count_fields = {
        "processed_observations",
        "delayed_observations",
        "dropped_observations",
        "forward_calls",
        "update_calls",
        "backward_calls",
    }
    _exact_keys(counts_value, count_fields, location=f"{location}.counts")
    counts = OperationCounts(
        processed_observations=_expect_int(
            counts_value["processed_observations"],
            location=f"{location}.counts.processed_observations",
        ),
        delayed_observations=_expect_int(
            counts_value["delayed_observations"],
            location=f"{location}.counts.delayed_observations",
        ),
        dropped_observations=_expect_int(
            counts_value["dropped_observations"],
            location=f"{location}.counts.dropped_observations",
        ),
        forward_calls=_expect_int(
            counts_value["forward_calls"],
            location=f"{location}.counts.forward_calls",
        ),
        update_calls=_expect_int(
            counts_value["update_calls"],
            location=f"{location}.counts.update_calls",
        ),
        backward_calls=_expect_int(
            counts_value["backward_calls"],
            location=f"{location}.counts.backward_calls",
        ),
    )

    latency_value = _expect_mapping(
        mapping["latency"],
        location=f"{location}.latency",
    )
    _exact_keys(
        latency_value,
        {"forward", "update", "backward"},
        location=f"{location}.latency",
    )
    latency = LatencyMeasurements(
        forward=_parse_latency(
            latency_value["forward"],
            location=f"{location}.latency.forward",
        ),
        update=_parse_latency(
            latency_value["update"],
            location=f"{location}.latency.update",
        ),
        backward=_parse_latency(
            latency_value["backward"],
            location=f"{location}.latency.backward",
        ),
    )

    memory_value = _expect_mapping(
        mapping["memory"],
        location=f"{location}.memory",
    )
    _exact_keys(
        memory_value,
        {
            "host_high_water_bytes",
            "accelerator_high_water_bytes",
            "persistent_state_bytes",
            "parameter_count",
            "measurement_method",
        },
        location=f"{location}.memory",
    )
    accelerator = memory_value["accelerator_high_water_bytes"]
    if accelerator is not None:
        accelerator = _expect_int(
            accelerator,
            location=f"{location}.memory.accelerator_high_water_bytes",
        )
    memory = MemoryHighWaterMarks(
        host_high_water_bytes=_expect_int(
            memory_value["host_high_water_bytes"],
            location=f"{location}.memory.host_high_water_bytes",
        ),
        accelerator_high_water_bytes=accelerator,
        persistent_state_bytes=_expect_int(
            memory_value["persistent_state_bytes"],
            location=f"{location}.memory.persistent_state_bytes",
        ),
        parameter_count=_expect_int(
            memory_value["parameter_count"],
            location=f"{location}.memory.parameter_count",
        ),
        measurement_method=_expect_string(
            memory_value["measurement_method"],
            location=f"{location}.memory.measurement_method",
        ),
    )
    energy_value = _expect_mapping(
        mapping["energy"],
        location=f"{location}.energy",
    )
    _exact_keys(
        energy_value,
        {"value", "unit", "measurement_method"},
        location=f"{location}.energy",
    )
    energy_unit = energy_value["unit"]
    if energy_unit is not None:
        energy_unit = _expect_string(
            energy_unit,
            location=f"{location}.energy.unit",
        )
    energy_method = energy_value["measurement_method"]
    if energy_method is not None:
        energy_method = _expect_string(
            energy_method,
            location=f"{location}.energy.measurement_method",
        )
    energy = EnergyMeasurement(
        value=_expect_optional_float(
            energy_value["value"],
            location=f"{location}.energy.value",
        ),
        unit=energy_unit,
        measurement_method=energy_method,
    )
    return OperationalMeasurements(
        counts=counts,
        latency=latency,
        memory=memory,
        energy=energy,
    )


def _parse_safety(value: object, *, location: str) -> SafetyMeasurements:
    mapping = _expect_mapping(value, location=location)
    _exact_keys(
        mapping,
        {
            "checks",
            "violations",
            "interventions",
            "near_misses",
            "cumulative_cost",
            "cumulative_near_miss_cost",
            "maximum_step_cost",
        },
        location=location,
    )
    return SafetyMeasurements(
        checks=_expect_int(mapping["checks"], location=f"{location}.checks"),
        violations=_expect_int(
            mapping["violations"],
            location=f"{location}.violations",
        ),
        interventions=_expect_int(
            mapping["interventions"],
            location=f"{location}.interventions",
        ),
        near_misses=_expect_int(
            mapping["near_misses"],
            location=f"{location}.near_misses",
        ),
        cumulative_cost=_expect_float(
            mapping["cumulative_cost"],
            location=f"{location}.cumulative_cost",
        ),
        cumulative_near_miss_cost=_expect_float(
            mapping["cumulative_near_miss_cost"],
            location=f"{location}.cumulative_near_miss_cost",
        ),
        maximum_step_cost=_expect_float(
            mapping["maximum_step_cost"],
            location=f"{location}.maximum_step_cost",
        ),
    )


def _parse_diagnostic_scalars(
    value: object,
    *,
    location: str,
) -> dict[str, DiagnosticScalar]:
    mapping = _expect_mapping(value, location=location)
    parsed: dict[str, DiagnosticScalar] = {}
    for key, item in mapping.items():
        parsed[key] = _validate_diagnostic_value(
            item,
            location=f"{location}.{key}",
        )
    return parsed


def _validate_metric_shape(value: object, *, location: str) -> None:
    mapping = _expect_mapping(value, location=location)
    _exact_keys(
        mapping,
        {
            "prequential_score",
            "lifetime_score",
            "adaptation_auc",
            "recovery",
            "per_regime_final_performance",
            "mean_final_performance",
            "forgetting",
            "backward_transfer",
            "forward_transfer",
            "stability",
            "worst_window",
        },
        location=location,
    )
    nested_keys = {
        "adaptation_auc": {"mean_normalized_auc", "segments"},
        "recovery": {
            "event_count",
            "recovered_count",
            "recovery_rate",
            "mean_recovery_steps",
            "maximum_recovery_steps",
            "events",
        },
        "forgetting": {"mean", "maximum", "per_regime"},
        "backward_transfer": {"mean", "per_regime"},
        "forward_transfer": {
            "mean_over_available",
            "available_regime_count",
            "per_regime",
        },
        "stability": {"mean_gap", "maximum_gap"},
        "worst_window": {
            "window_size",
            "score",
            "start_step",
            "end_step_exclusive",
        },
    }
    for key, expected in nested_keys.items():
        nested = _expect_mapping(
            mapping[key],
            location=f"{location}.{key}",
        )
        _exact_keys(nested, expected, location=f"{location}.{key}")

    adaptation = _expect_mapping(
        mapping["adaptation_auc"],
        location=f"{location}.adaptation_auc",
    )
    for index, record in enumerate(
        _expect_list(
            adaptation["segments"],
            location=f"{location}.adaptation_auc.segments",
        )
    ):
        segment = _expect_mapping(
            record,
            location=f"{location}.adaptation_auc.segments[{index}]",
        )
        _exact_keys(
            segment,
            {
                "segment_index",
                "regime_id",
                "start_step",
                "end_step_exclusive",
                "normalized_auc",
            },
            location=f"{location}.adaptation_auc.segments[{index}]",
        )

    recovery = _expect_mapping(
        mapping["recovery"],
        location=f"{location}.recovery",
    )
    for index, record in enumerate(
        _expect_list(
            recovery["events"],
            location=f"{location}.recovery.events",
        )
    ):
        event = _expect_mapping(
            record,
            location=f"{location}.recovery.events[{index}]",
        )
        _exact_keys(
            event,
            {
                "segment_index",
                "regime_id",
                "start_step",
                "end_step_exclusive",
                "threshold",
                "window_size",
                "recovered",
                "recovery_steps",
            },
            location=f"{location}.recovery.events[{index}]",
        )


def _parse_condition(
    value: object,
    *,
    index: int,
    protocol: ContinualEvaluationProtocol,
    budget: MatchedBudget,
) -> tuple[ConditionEvaluationInput, ConditionRole]:
    location = f"conditions[{index}]"
    mapping = _expect_mapping(value, location=location)
    _exact_keys(
        mapping,
        {
            "name",
            "role",
            "metrics",
            "operations",
            "safety",
            "diagnostics",
            "evidence",
        },
        location=location,
    )
    name = _expect_string(mapping["name"], location=f"{location}.name")
    role_value = _expect_string(mapping["role"], location=f"{location}.role")
    if role_value not in {"candidate", "baseline"}:
        raise ValueError(f"{location}.role must be candidate or baseline")
    role = cast(ConditionRole, role_value)
    _validate_metric_shape(mapping["metrics"], location=f"{location}.metrics")

    evidence = _expect_mapping(
        mapping["evidence"],
        location=f"{location}.evidence",
    )
    _exact_keys(
        evidence,
        {"predict_before_update_trace", "per_regime_evaluator"},
        location=f"{location}.evidence",
    )
    trace_value = _expect_mapping(
        evidence["predict_before_update_trace"],
        location=f"{location}.evidence.predict_before_update_trace",
    )
    _exact_keys(
        trace_value,
        {"ordering", "scores", "processed"},
        location=f"{location}.evidence.predict_before_update_trace",
    )
    if trace_value["ordering"] != TRACE_ORDERING_VERSION:
        raise ValueError(f"{location}: trace ordering version mismatch")
    scores = tuple(
        _expect_optional_float(
            item,
            location=f"{location}.evidence.predict_before_update_trace.scores",
        )
        for item in _expect_list(
            trace_value["scores"],
            location=f"{location}.evidence.predict_before_update_trace.scores",
        )
    )
    processed = tuple(
        _expect_bool(
            item,
            location=f"{location}.evidence.predict_before_update_trace.processed",
        )
        for item in _expect_list(
            trace_value["processed"],
            location=f"{location}.evidence.predict_before_update_trace.processed",
        )
    )
    trace = PredictBeforeUpdateTrace(scores=scores, processed=processed)

    evaluator_value = _expect_mapping(
        evidence["per_regime_evaluator"],
        location=f"{location}.evidence.per_regime_evaluator",
    )
    _exact_keys(
        evaluator_value,
        {"regime_ids", "checkpoint_steps", "values"},
        location=f"{location}.evidence.per_regime_evaluator",
    )
    observed_regimes = tuple(
        _expect_string(
            item,
            location=f"{location}.evidence.per_regime_evaluator.regime_ids",
        )
        for item in _expect_list(
            evaluator_value["regime_ids"],
            location=f"{location}.evidence.per_regime_evaluator.regime_ids",
        )
    )
    observed_checkpoints = tuple(
        _expect_int(
            item,
            location=f"{location}.evidence.per_regime_evaluator.checkpoint_steps",
        )
        for item in _expect_list(
            evaluator_value["checkpoint_steps"],
            location=f"{location}.evidence.per_regime_evaluator.checkpoint_steps",
        )
    )
    if observed_regimes != protocol.evaluator_regime_ids:
        raise ValueError(f"{location}: evaluator regime IDs do not match protocol")
    if observed_checkpoints != protocol.checkpoint_steps:
        raise ValueError(f"{location}: evaluator checkpoints do not match protocol")
    rows: list[tuple[float, ...]] = []
    for row_index, raw_row in enumerate(
        _expect_list(
            evaluator_value["values"],
            location=f"{location}.evidence.per_regime_evaluator.values",
        )
    ):
        rows.append(
            tuple(
                _expect_float(
                    item,
                    location=(f"{location}.evidence.per_regime_evaluator.values[{row_index}]"),
                )
                for item in _expect_list(
                    raw_row,
                    location=(f"{location}.evidence.per_regime_evaluator.values[{row_index}]"),
                )
            )
        )
    evaluator_matrix = EvaluatorMatrix(values=tuple(rows))

    diagnostics_value = _expect_mapping(
        mapping["diagnostics"],
        location=f"{location}.diagnostics",
    )
    _exact_keys(
        diagnostics_value,
        {
            "applicable_components",
            "component_diagnostics",
            "plasticity_applicable",
            "plasticity_diagnostics",
        },
        location=f"{location}.diagnostics",
    )
    applicable_components = tuple(
        _expect_string(
            item,
            location=f"{location}.diagnostics.applicable_components",
        )
        for item in _expect_list(
            diagnostics_value["applicable_components"],
            location=f"{location}.diagnostics.applicable_components",
        )
    )
    raw_components = diagnostics_value["component_diagnostics"]
    components: dict[str, dict[str, DiagnosticScalar]] | None
    if raw_components is None:
        components = None
    else:
        component_mapping = _expect_mapping(
            raw_components,
            location=f"{location}.diagnostics.component_diagnostics",
        )
        components = {
            component: _parse_diagnostic_scalars(
                diagnostics,
                location=(f"{location}.diagnostics.component_diagnostics.{component}"),
            )
            for component, diagnostics in component_mapping.items()
        }
    raw_plasticity = diagnostics_value["plasticity_diagnostics"]
    plasticity = (
        None
        if raw_plasticity is None
        else _parse_diagnostic_scalars(
            raw_plasticity,
            location=f"{location}.diagnostics.plasticity_diagnostics",
        )
    )

    return (
        ConditionEvaluationInput(
            name=name,
            trace=trace,
            evaluator_matrix=evaluator_matrix,
            budget=budget,
            operations=_parse_operations(
                mapping["operations"],
                location=f"{location}.operations",
            ),
            safety=_parse_safety(
                mapping["safety"],
                location=f"{location}.safety",
            ),
            applicable_components=applicable_components,
            component_diagnostics=components,
            plasticity_applicable=_expect_bool(
                diagnostics_value["plasticity_applicable"],
                location=f"{location}.diagnostics.plasticity_applicable",
            ),
            plasticity_diagnostics=plasticity,
        ),
        role,
    )


def _reconstruct_report(report: Mapping[str, object]) -> dict[str, object]:
    _exact_keys(
        report,
        {
            "schema_version",
            "trace_ordering_version",
            "acceptance_status",
            "interpretation",
            "metric_definitions",
            "protocol",
            "comparison_contract",
            "conditions",
        },
        location="report",
    )
    if report["schema_version"] != SCHEMA_VERSION:
        raise ValueError("schema_version does not match continual report v1")
    if report["trace_ordering_version"] != TRACE_ORDERING_VERSION:
        raise ValueError("trace_ordering_version does not match v1")
    if report["acceptance_status"] != ACCEPTANCE_STATUS:
        raise ValueError("acceptance_status must remain not-assessed")
    if report["interpretation"] != REPORT_INTERPRETATION:
        raise ValueError("report interpretation does not match the v1 contract")
    if report["metric_definitions"] != METRIC_DEFINITIONS:
        raise ValueError("metric_definitions do not match the v1 contract")

    protocol = _parse_protocol(report["protocol"])
    comparison = _expect_mapping(
        report["comparison_contract"],
        location="comparison_contract",
    )
    _exact_keys(
        comparison,
        {
            "candidate",
            "baselines",
            "baseline_count",
            "matched_budget",
            "budget_match_verified",
        },
        location="comparison_contract",
    )
    candidate_name = _expect_string(
        comparison["candidate"],
        location="comparison_contract.candidate",
    )
    baseline_names = tuple(
        _expect_string(
            item,
            location="comparison_contract.baselines",
        )
        for item in _expect_list(
            comparison["baselines"],
            location="comparison_contract.baselines",
        )
    )
    baseline_count = _expect_int(
        comparison["baseline_count"],
        location="comparison_contract.baseline_count",
    )
    if baseline_count != len(baseline_names) or baseline_count < 2:
        raise ValueError("comparison baseline_count must equal at least two names")
    if comparison["budget_match_verified"] is not True:
        raise ValueError("comparison budget_match_verified must be true")
    budget = _parse_budget(comparison["matched_budget"])

    raw_conditions = _expect_list(report["conditions"], location="conditions")
    parsed = [
        _parse_condition(
            condition,
            index=index,
            protocol=protocol,
            budget=budget,
        )
        for index, condition in enumerate(raw_conditions)
    ]
    candidates = [condition for condition, role in parsed if role == "candidate"]
    baselines = [condition for condition, role in parsed if role == "baseline"]
    if len(candidates) != 1:
        raise ValueError("report must contain exactly one candidate condition")
    if candidates[0].name != candidate_name:
        raise ValueError("candidate condition does not match comparison contract")
    if tuple(condition.name for condition in baselines) != baseline_names:
        raise ValueError("baseline conditions do not match comparison contract")
    return build_continual_evaluation_report(
        protocol=protocol,
        candidate=candidates[0],
        baselines=baselines,
    )


def validate_continual_evaluation_report(
    report: Mapping[str, object],
) -> ReportValidation:
    """Strictly reconstruct raw evidence and every derived report field."""

    try:
        normalized = _json_clone(report)
        if not isinstance(normalized, dict):
            raise ValueError("report must serialize to a JSON object")
        reconstructed = _reconstruct_report(normalized)
        if reconstructed != normalized:
            raise ValueError("derived metrics or declared evidence do not reconstruct exactly")
    except (KeyError, TypeError, ValueError) as error:
        return ReportValidation(valid=False, errors=(str(error),))
    return ReportValidation(valid=True, errors=())


def continual_evaluation_report_json(report: Mapping[str, object]) -> str:
    """Serialize only a strict, reconstructable v1 report."""

    validation = validate_continual_evaluation_report(report)
    if not validation.valid:
        raise ValueError("; ".join(validation.errors))
    return (
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )


def _reject_nonstandard_json_constant(value: str) -> NoReturn:
    raise ValueError(f"non-standard JSON numeric constant {value!r}")


def _reject_duplicate_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def load_continual_evaluation_report(path: Path) -> dict[str, object]:
    """Load strict standard JSON and reject incomplete or modified reports."""

    parsed = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=_reject_nonstandard_json_constant,
        object_pairs_hook=_reject_duplicate_keys,
    )
    if not isinstance(parsed, dict):
        raise ValueError("continual evaluation report root must be an object")
    validation = validate_continual_evaluation_report(parsed)
    if not validation.valid:
        raise ValueError("; ".join(validation.errors))
    return parsed


__all__ = [
    "ACCEPTANCE_STATUS",
    "METRIC_DEFINITIONS",
    "REPORT_INTERPRETATION",
    "SCHEMA_VERSION",
    "TRACE_ORDERING_VERSION",
    "ConditionEvaluationInput",
    "ContinualEvaluationProtocol",
    "EnergyMeasurement",
    "EvaluatorMatrix",
    "LatencyMeasurements",
    "LatencySummary",
    "MatchedBudget",
    "MemoryHighWaterMarks",
    "OperationCounts",
    "OperationalMeasurements",
    "PredictBeforeUpdateTrace",
    "ReportValidation",
    "SafetyMeasurements",
    "build_continual_evaluation_report",
    "continual_evaluation_report_json",
    "load_continual_evaluation_report",
    "validate_continual_evaluation_report",
]
