"""Strict development-only endpoint metrics for complete HCCL longevity lives.

This pure host evaluator accepts an already executed, uninterrupted Core-L2 or
Core-L3 life.  It does not execute a world or learner.  The evaluator-only
regime labels, schedule boundaries, and four counterfactual score columns must
never be supplied to a learner.

The fixed schedule retains ten source segments per cycle for geometry.  A
regime occurrence is instead one maximal contiguous same-label run, including
an A run that crosses a cycle boundary or spans the D-to-A replacement slot.
Every segment and maximal occurrence entry and tail uses 64 committed
transitions.  Consecutive nonadjacent same-regime occurrences define retention,
backward transfer, and recovery:
entry retention is ``current_entry - previous_tail``; tail backward transfer is
``current_tail - previous_tail``.  Recovery is the first current-occurrence
trailing-64 task mean at least the previous tail, and its step count is the
number of additional transitions after the entry-window endpoint.  These are
descriptive development diagnostics, not thresholds or efficacy decisions.

Core-L2 and Core-L3 repeat the canonical ten-segment geometry over one clock
and one RNG life.  D occurs only in cycle zero; the corresponding segment is A
afterward.  D's recurrence-dependent fields are therefore explicitly
unavailable.  Post-exposure evaluator-only D counterfactual columns remain
descriptive and are not represented as a recurrence.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any, Final, cast

import numpy as np
from numpy.typing import NDArray

from alberta_framework.streams.hccl_causal_core import (
    HCCL_CAUSAL_CORE_L2_PROFILE,
    HCCL_CAUSAL_CORE_L3_PROFILE,
    HCCL_CAUSAL_CORE_REGIME_NAMES,
    HCCL_CAUSAL_CORE_SCHEDULE,
    hccl_causal_core_cycle_count_for_profile,
    hccl_causal_core_lifetime_for_profile,
    hccl_causal_core_schedule_for_profile,
)

HCCL_LONGEVITY_ENDPOINT_CONFIG_SCHEMA: Final = (
    "alberta.hccl-longevity-endpoints.config.v1"
)
HCCL_LONGEVITY_COMPLETE_TRACE_SCHEMA: Final = (
    "alberta.hccl-longevity-endpoints.complete-trace.v1"
)
HCCL_LONGEVITY_ENDPOINT_REPORT_SCHEMA: Final = (
    "alberta.hccl-longevity-endpoints.report.v1"
)
HCCL_LONGEVITY_ENDPOINT_STATUS: Final = "not_assessed"
HCCL_LONGEVITY_ENDPOINT_EVIDENCE_LEVEL: Final = "L0-development-diagnostic-only"
HCCL_LONGEVITY_ENDPOINT_WINDOW_STEPS: Final = 64

HCCL_LONGEVITY_ENDPOINT_LIMITATIONS: Final = (
    "development-only descriptive Core-L2/Core-L3 endpoint evaluator",
    "complete host-resident exact trace required",
    "no world or learner execution",
    "no acceptance thresholds or efficacy decision",
    "no output writer, artifact, evidence, seed authority, or promotion path",
    "evaluator labels, boundaries, and counterfactual columns are never learner inputs",
    "trace declarations and unkeyed digests do not authenticate provenance or nonexposure",
    "finite Core-L2/Core-L3 lives do not establish indefinite continual operation",
    "D occurs once, so D recurrence-dependent metrics are unavailable",
)

HCCL_LONGEVITY_ENDPOINT_METRIC_DEFINITIONS: Final = (
    (
        "segment_entry_and_tail",
        "first/final-64 means for all ten source schedule segments per cycle",
    ),
    (
        "maximal_regime_occurrence",
        "one maximal contiguous same-label run, merged across adjacent segments and cycles",
    ),
    (
        "entry_retention_delta",
        "current same-regime entry mean minus the preceding same-regime tail",
    ),
    (
        "tail_backward_transfer",
        "current same-regime tail mean minus the preceding same-regime tail",
    ),
    (
        "trailing64_recovery",
        "first current-occurrence trailing-64 task mean at least the preceding tail",
    ),
    (
        "positive_gap_recovery_area",
        "sum of positive preceding-tail minus trailing-64 gaps through recovery or phase end",
    ),
    (
        "occurrence_tail_slope",
        "OLS slope of all same-regime occurrence-tail task means versus occurrence ordinal",
    ),
    (
        "cycle_task_trend",
        "OLS slope and first-to-latest change of all full-cycle task means",
    ),
    (
        "cycle_segment_tail_trend",
        "OLS slope of the mean of all ten fixed segment tails in each cycle",
    ),
    (
        "counterfactual_retention",
        "evaluator-only phase-tail column forgetting and transfer after first exposure",
    ),
)

_PROFILE_CYCLES: Final = {
    HCCL_CAUSAL_CORE_L2_PROFILE: 8,
    HCCL_CAUSAL_CORE_L3_PROFILE: 112,
}
_CANONICAL_PHASES_PER_CYCLE: Final = len(HCCL_CAUSAL_CORE_SCHEDULE)
_CANONICAL_CYCLE_STEPS: Final = HCCL_CAUSAL_CORE_SCHEDULE[-1][2]
_REGIME_IDS: Final = tuple(range(len(HCCL_CAUSAL_CORE_REGIME_NAMES)))
_ARRAY_VALIDATION_CHUNK: Final = 65_536

if HCCL_CAUSAL_CORE_REGIME_NAMES != ("A", "B", "C", "D") or _REGIME_IDS != (
    0,
    1,
    2,
    3,
):
    raise RuntimeError("the imported HCCL evaluator regime identifiers have changed")
if _CANONICAL_PHASES_PER_CYCLE != 10 or _CANONICAL_CYCLE_STEPS != 8_998:
    raise RuntimeError("the imported HCCL canonical cycle geometry has changed")

type Float32Array = NDArray[np.float32]
type Int32Array = NDArray[np.int32]
type UInt32Array = NDArray[np.uint32]
type BoolArray = NDArray[np.bool_]
type Schedule = tuple[tuple[str, int, int], ...]


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _profile_geometry(schedule_profile: str) -> tuple[Schedule, int, int]:
    if type(schedule_profile) is not str:
        raise TypeError("schedule_profile must be an exact string")
    if schedule_profile not in _PROFILE_CYCLES:
        raise ValueError("schedule_profile must select Core-L2 or Core-L3")
    cycle_count = hccl_causal_core_cycle_count_for_profile(schedule_profile)
    expected_cycles = _PROFILE_CYCLES[schedule_profile]
    if cycle_count != expected_cycles:
        raise RuntimeError("the imported versioned longevity cycle count has changed")
    schedule = hccl_causal_core_schedule_for_profile(schedule_profile)
    total_steps = hccl_causal_core_lifetime_for_profile(schedule_profile)
    if total_steps != _CANONICAL_CYCLE_STEPS * cycle_count:
        raise RuntimeError("the imported versioned longevity lifetime has changed")
    if len(schedule) != _CANONICAL_PHASES_PER_CYCLE * cycle_count:
        raise RuntimeError("the imported versioned longevity segment count has changed")

    cursor = 0
    d_count = 0
    for phase_index, (actual_name, actual_start, actual_end) in enumerate(schedule):
        cycle_index, segment_index = divmod(phase_index, _CANONICAL_PHASES_PER_CYCLE)
        canonical_name, canonical_start, canonical_end = HCCL_CAUSAL_CORE_SCHEDULE[
            segment_index
        ]
        expected_name = (
            "A" if cycle_index > 0 and canonical_name == "D" else canonical_name
        )
        expected_end = cursor + canonical_end - canonical_start
        if (actual_name, actual_start, actual_end) != (
            expected_name,
            cursor,
            expected_end,
        ):
            raise RuntimeError("the imported versioned longevity schedule geometry has changed")
        d_count += int(actual_name == "D")
        cursor = expected_end
    if cursor != total_steps or d_count != 1:
        raise RuntimeError("the imported longevity schedule must have exact coverage and one D")
    return schedule, cycle_count, total_steps


def _maximal_schedule_runs(
    schedule: Schedule,
) -> tuple[tuple[str, int, int, int, int], ...]:
    """Return (name, start, end, first segment, last segment exclusive) runs."""

    runs: list[tuple[str, int, int, int, int]] = []
    for segment_index, (name, start, end) in enumerate(schedule):
        if runs and runs[-1][0] == name and runs[-1][2] == start:
            prior_name, prior_start, _prior_end, first_segment, _last_segment = runs[-1]
            runs[-1] = (prior_name, prior_start, end, first_segment, segment_index + 1)
        else:
            runs.append((name, start, end, segment_index, segment_index + 1))
    if not runs or runs[0][1] != 0 or runs[-1][2] != schedule[-1][2]:
        raise AssertionError("maximal schedule runs must exactly cover the fixed life")
    if any(
        left[0] == right[0] or left[2] != right[1]
        for left, right in zip(runs, runs[1:])
    ):
        raise AssertionError("maximal schedule runs must be contiguous and label-maximal")
    return tuple(runs)


def _schedule_sha256(schedule_profile: str) -> str:
    schedule, _cycles, _steps = _profile_geometry(schedule_profile)
    return hashlib.sha256(
        b"alberta.hccl-longevity-endpoints.schedule.v1\0"
        + _canonical_json_bytes(
            {
                "schedule_profile": schedule_profile,
                "schedule": [list(item) for item in schedule],
            }
        )
    ).hexdigest()


def _frozen_array(
    value: object,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: np.dtype[Any],
) -> NDArray[Any]:
    if type(value) is not np.ndarray:
        raise TypeError(f"{name} must be an exact numpy.ndarray")
    array = cast(NDArray[Any], value)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if array.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}")
    result = np.array(array, dtype=dtype, order="C", copy=True)
    result.flags.writeable = False
    return result


def _all_finite(array: NDArray[Any]) -> bool:
    for start in range(0, array.shape[0], _ARRAY_VALIDATION_CHUNK):
        if not bool(np.all(np.isfinite(array[start : start + _ARRAY_VALIDATION_CHUNK]))):
            return False
    return True


def _mean(values: NDArray[np.float32]) -> float:
    return float(np.mean(values, dtype=np.float64))


def _agent_means(values: NDArray[np.float32]) -> tuple[float, float]:
    means = np.mean(values, axis=0, dtype=np.float64)
    return (float(means[0]), float(means[1]))


def _ordinary_slope(values: Sequence[float]) -> float:
    if len(values) < 2:
        raise ValueError("an OLS lifetime slope requires at least two values")
    y = np.asarray(values, dtype=np.float64)
    x = np.arange(y.size, dtype=np.float64)
    centered_x = x - np.mean(x)
    centered_y = y - np.mean(y)
    return float(np.sum(centered_x * centered_y) / np.sum(centered_x * centered_x))


@dataclasses.dataclass(frozen=True, slots=True)
class HCCLLongevityEndpointConfig:
    """One strict Core-L2 or Core-L3 descriptive evaluator configuration."""

    schedule_profile: str = HCCL_CAUSAL_CORE_L2_PROFILE
    schema: str = HCCL_LONGEVITY_ENDPOINT_CONFIG_SCHEMA

    def __post_init__(self) -> None:
        _profile_geometry(self.schedule_profile)
        if type(self.schema) is not str or self.schema != HCCL_LONGEVITY_ENDPOINT_CONFIG_SCHEMA:
            raise ValueError("longevity endpoint config schema is fixed")

    @property
    def schedule(self) -> Schedule:
        return _profile_geometry(self.schedule_profile)[0]

    @property
    def cycle_count(self) -> int:
        return _profile_geometry(self.schedule_profile)[1]

    @property
    def total_steps(self) -> int:
        return _profile_geometry(self.schedule_profile)[2]

    @property
    def maximal_regime_occurrence_count(self) -> int:
        return len(_maximal_schedule_runs(self.schedule))

    @property
    def genuine_recurrence_comparison_count(self) -> int:
        runs = _maximal_schedule_runs(self.schedule)
        return len(runs) - len({run[0] for run in runs})

    def to_config(self) -> dict[str, object]:
        """Return the complete JSON-compatible configuration and nonauthority."""

        return {
            "type": type(self).__name__,
            "schema": self.schema,
            "trace_schema": HCCL_LONGEVITY_COMPLETE_TRACE_SCHEMA,
            "report_schema": HCCL_LONGEVITY_ENDPOINT_REPORT_SCHEMA,
            "status": HCCL_LONGEVITY_ENDPOINT_STATUS,
            "evidence_level": HCCL_LONGEVITY_ENDPOINT_EVIDENCE_LEVEL,
            "development_only": True,
            "schedule_profile": self.schedule_profile,
            "schedule_sha256": _schedule_sha256(self.schedule_profile),
            "schedule_segment_count": len(self.schedule),
            "maximal_regime_occurrence_count": self.maximal_regime_occurrence_count,
            "genuine_recurrence_comparison_count": (
                self.genuine_recurrence_comparison_count
            ),
            "cycle_count": self.cycle_count,
            "total_steps": self.total_steps,
            "window_steps": HCCL_LONGEVITY_ENDPOINT_WINDOW_STEPS,
            "complete_trace_required": True,
            "all_transactions_must_be_committed": True,
            "reset_callbacks_required": 0,
            "boundary_callbacks_required": 0,
            "learner_visible_regime_labels": False,
            "learner_visible_schedule_boundaries": False,
            "counterfactual_score_columns_exposed_to_learner": False,
            "d_maximal_regime_occurrence_count": 1,
            "d_recurrence_available": False,
            "metric_definitions": dict(HCCL_LONGEVITY_ENDPOINT_METRIC_DEFINITIONS),
            "acceptance_thresholds_defined": False,
            "benchmark_execution_authorized": False,
            "output_writes_authorized": False,
            "artifact_authorized": False,
            "evidence_authorized": False,
            "seed_reservation_or_consumption_authorized": False,
            "promotion_authorized": False,
            "limitations": list(HCCL_LONGEVITY_ENDPOINT_LIMITATIONS),
        }

    @classmethod
    def from_config(cls, payload: Mapping[str, object]) -> HCCLLongevityEndpointConfig:
        """Fail closed unless a decoded mapping is one exact fixed configuration."""

        if not isinstance(payload, Mapping) or any(type(key) is not str for key in payload):
            raise TypeError("longevity endpoint payload must be a string-keyed mapping")
        profile = payload.get("schedule_profile")
        if type(profile) is not str:
            raise ValueError("longevity endpoint payload must select an exact profile")
        candidate = cls(schedule_profile=profile)
        if _canonical_json_bytes(dict(payload)) != _canonical_json_bytes(candidate.to_config()):
            raise ValueError("longevity endpoint payload differs from the fixed configuration")
        return candidate


@dataclasses.dataclass(frozen=True, slots=True)
class HCCLLongevityCompleteTrace:
    """One complete, uninterrupted, host-resident Core-L2 or Core-L3 life."""

    schedule_profile: str
    regime_ids: Int32Array
    transaction_committed: BoolArray
    pre_step_words: UInt32Array
    post_step_words: UInt32Array
    task_scores: Float32Array
    net_rewards: Float32Array
    all_regime_score_matrix: Float32Array
    reset_callback_count: int = 0
    boundary_callback_count: int = 0
    learner_received_evaluator_regime_ids: bool = False
    learner_received_evaluator_schedule_boundaries: bool = False
    learner_received_counterfactual_scores: bool = False
    schema: str = HCCL_LONGEVITY_COMPLETE_TRACE_SCHEMA

    def __post_init__(self) -> None:
        _schedule, _cycles, steps = _profile_geometry(self.schedule_profile)
        for name, shape, dtype in (
            ("regime_ids", (steps,), np.dtype(np.int32)),
            ("transaction_committed", (steps,), np.dtype(np.bool_)),
            ("pre_step_words", (steps, 2), np.dtype(np.uint32)),
            ("post_step_words", (steps, 2), np.dtype(np.uint32)),
            ("task_scores", (steps,), np.dtype(np.float32)),
            ("net_rewards", (steps, 2), np.dtype(np.float32)),
            ("all_regime_score_matrix", (steps, 4), np.dtype(np.float32)),
        ):
            object.__setattr__(
                self,
                name,
                _frozen_array(getattr(self, name), name=name, shape=shape, dtype=dtype),
            )
        _validate_complete_trace_fields(self)

    @property
    def total_steps(self) -> int:
        return _profile_geometry(self.schedule_profile)[2]

    @property
    def cycle_count(self) -> int:
        return _profile_geometry(self.schedule_profile)[1]

    @classmethod
    def from_continual_dyad_life_trace(
        cls,
        trace: object,
    ) -> HCCLLongevityCompleteTrace:
        """Defensively copy one exact runner-owned Core-L2/Core-L3 life."""

        from alberta_framework.core.hccl_continual_dyad_runner import (
            HCCLContinualDyadLifeTrace,
            validate_hccl_continual_dyad_life_trace,
        )

        if type(trace) is not HCCLContinualDyadLifeTrace:
            raise TypeError("trace must be exact HCCLContinualDyadLifeTrace")
        life = validate_hccl_continual_dyad_life_trace(trace)
        if not life.longevity_life:
            raise ValueError("runner life must select Core-L2 or Core-L3")
        return cls(
            schedule_profile=life.schedule_profile,
            regime_ids=life.regime_ids,
            transaction_committed=life.transaction_committed,
            pre_step_words=life.pre_step_words,
            post_step_words=life.post_step_words,
            task_scores=life.task_scores,
            net_rewards=life.net_rewards,
            all_regime_score_matrix=life.all_regime_score_matrix,
            reset_callback_count=life.reset_callback_count,
            boundary_callback_count=life.boundary_callback_count,
            learner_received_evaluator_regime_ids=(
                life.learner_received_evaluator_regime_ids
            ),
            learner_received_evaluator_schedule_boundaries=False,
            learner_received_counterfactual_scores=(
                life.learner_received_counterfactual_scores
            ),
        )


def _validate_exact_clocks(trace: HCCLLongevityCompleteTrace) -> None:
    steps = trace.total_steps
    if (
        bool(np.any(trace.pre_step_words[:, 0] != np.uint32(0)))
        or bool(np.any(trace.post_step_words[:, 0] != np.uint32(0)))
        or int(trace.pre_step_words[0, 1]) != 0
        or int(trace.post_step_words[-1, 1]) != steps
        or not np.array_equal(trace.pre_step_words[1:, 1], trace.post_step_words[:-1, 1])
    ):
        raise ValueError("trace clocks are not the exact monotone committed clocks")
    for start in range(0, steps, _ARRAY_VALIDATION_CHUNK):
        end = min(start + _ARRAY_VALIDATION_CHUNK, steps)
        expected_post = trace.pre_step_words[start:end, 1] + np.uint32(1)
        if not np.array_equal(trace.post_step_words[start:end, 1], expected_post):
            raise ValueError("trace clocks are not the exact monotone committed clocks")


def _validate_complete_trace_fields(trace: HCCLLongevityCompleteTrace) -> None:
    schedule, _cycles, steps = _profile_geometry(trace.schedule_profile)
    if type(trace.schema) is not str or trace.schema != HCCL_LONGEVITY_COMPLETE_TRACE_SCHEMA:
        raise ValueError("trace schema differs from the fixed longevity trace schema")
    for name, shape, dtype in (
        ("regime_ids", (steps,), np.dtype(np.int32)),
        ("transaction_committed", (steps,), np.dtype(np.bool_)),
        ("pre_step_words", (steps, 2), np.dtype(np.uint32)),
        ("post_step_words", (steps, 2), np.dtype(np.uint32)),
        ("task_scores", (steps,), np.dtype(np.float32)),
        ("net_rewards", (steps, 2), np.dtype(np.float32)),
        ("all_regime_score_matrix", (steps, 4), np.dtype(np.float32)),
    ):
        value = getattr(trace, name)
        if type(value) is not np.ndarray:
            raise TypeError(f"trace.{name} must remain an exact numpy.ndarray")
        array = cast(NDArray[Any], value)
        if array.shape != shape or array.dtype != dtype:
            raise ValueError(f"trace.{name} no longer has its exact shape and dtype")
        if array.flags.writeable or not array.flags.c_contiguous:
            raise ValueError(f"trace.{name} must remain a read-only C-contiguous array")
    for name in ("reset_callback_count", "boundary_callback_count"):
        value = getattr(trace, name)
        if type(value) is not int or value != 0:
            raise ValueError(f"trace.{name} must be the exact integer zero")
    for name in (
        "learner_received_evaluator_regime_ids",
        "learner_received_evaluator_schedule_boundaries",
        "learner_received_counterfactual_scores",
    ):
        value = getattr(trace, name)
        if type(value) is not bool or value:
            raise ValueError(f"trace.{name} must be exact False")

    for regime_name, start, end in schedule:
        regime_id = HCCL_CAUSAL_CORE_REGIME_NAMES.index(regime_name)
        if not bool(np.all(trace.regime_ids[start:end] == regime_id)):
            raise ValueError("trace regime_ids do not equal the world-owned schedule")
    if not bool(np.all(trace.transaction_committed)):
        raise ValueError("every transaction in a complete longevity life must be committed")
    _validate_exact_clocks(trace)

    for name in ("task_scores", "net_rewards", "all_regime_score_matrix"):
        if not _all_finite(cast(NDArray[Any], getattr(trace, name))):
            raise ValueError(f"trace.{name} must be entirely finite")
    for regime_name, start, end in schedule:
        regime_id = HCCL_CAUSAL_CORE_REGIME_NAMES.index(regime_name)
        if not np.array_equal(
            trace.task_scores[start:end],
            trace.all_regime_score_matrix[start:end, regime_id],
        ):
            raise ValueError("task scores must exactly equal their selected evaluator column")
    for start in range(0, steps, _ARRAY_VALIDATION_CHUNK):
        end = min(start + _ARRAY_VALIDATION_CHUNK, steps)
        task = trace.task_scores[start:end]
        if not (
            np.array_equal(trace.net_rewards[start:end, 0], task)
            and np.array_equal(trace.net_rewards[start:end, 1], task)
        ):
            raise ValueError("causal-core net rewards must exactly equal task score per agent")


def validate_hccl_longevity_complete_trace(
    trace: HCCLLongevityCompleteTrace,
) -> HCCLLongevityCompleteTrace:
    """Revalidate a complete trace, including its post-construction freeze status."""

    if type(trace) is not HCCLLongevityCompleteTrace:
        raise TypeError("trace must be exact HCCLLongevityCompleteTrace")
    _validate_complete_trace_fields(trace)
    return trace


@dataclasses.dataclass(frozen=True, slots=True)
class HCCLLongevitySegmentRecord:
    segment_index: int
    cycle_index: int
    canonical_segment_index: int
    regime_id: int
    regime_name: str
    start_step: int
    end_step_exclusive: int
    entry_task_mean: float
    tail_task_mean: float
    entry_net_reward_means: tuple[float, float]
    tail_net_reward_means: tuple[float, float]

    def to_config(self) -> dict[str, object]:
        payload = dataclasses.asdict(self)
        payload["entry_net_reward_means"] = list(self.entry_net_reward_means)
        payload["tail_net_reward_means"] = list(self.tail_net_reward_means)
        return payload


@dataclasses.dataclass(frozen=True, slots=True)
class HCCLLongevityOccurrenceRecord:
    occurrence_index: int
    regime_id: int
    regime_name: str
    regime_occurrence_index: int
    first_segment_index: int
    last_segment_index_exclusive: int
    start_cycle_index: int
    end_cycle_index_inclusive: int
    start_step: int
    end_step_exclusive: int
    entry_task_mean: float
    tail_task_mean: float
    entry_net_reward_means: tuple[float, float]
    tail_net_reward_means: tuple[float, float]

    def to_config(self) -> dict[str, object]:
        payload = dataclasses.asdict(self)
        payload["entry_net_reward_means"] = list(self.entry_net_reward_means)
        payload["tail_net_reward_means"] = list(self.tail_net_reward_means)
        return payload


@dataclasses.dataclass(frozen=True, slots=True)
class HCCLLongevityRecurrenceRecord:
    regime_id: int
    regime_name: str
    previous_occurrence_index: int
    current_occurrence_index: int
    previous_last_segment_index_exclusive: int
    current_first_segment_index: int
    intervening_segment_count: int
    previous_end_cycle_index_inclusive: int
    current_start_cycle_index: int
    previous_regime_occurrence_index: int
    current_regime_occurrence_index: int
    prior_tail_reference: float
    current_entry_mean: float
    current_tail_mean: float
    entry_retention_delta: float
    entry_forgetting_gap: float
    tail_backward_transfer: float
    tail_forgetting_gap: float
    trailing64_recovered: bool
    trailing64_recovery_steps_after_entry: int | None
    trailing64_recovery_endpoint_step_exclusive: int | None
    positive_gap_recovery_area: float

    def to_config(self) -> dict[str, object]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True, slots=True)
class HCCLLongevityCycleRecord:
    cycle_index: int
    start_step: int
    end_step_exclusive: int
    segment_count: int
    maximal_occurrence_start_count: int
    maximal_occurrence_end_count: int
    task_mean: float
    segment_entry_task_mean: float
    segment_tail_task_mean: float
    regime_segment_tail_task_means: tuple[
        float | None, float | None, float | None, float | None
    ]
    end_tail_all_regime_score_means: tuple[float, float, float, float]
    task_mean_change_from_previous: float | None
    segment_tail_mean_change_from_previous: float | None

    def to_config(self) -> dict[str, object]:
        payload = dataclasses.asdict(self)
        payload["regime_segment_tail_task_means"] = list(
            self.regime_segment_tail_task_means
        )
        payload["end_tail_all_regime_score_means"] = list(
            self.end_tail_all_regime_score_means
        )
        return payload


@dataclasses.dataclass(frozen=True, slots=True)
class HCCLLongevityRegimeSummary:
    regime_id: int
    regime_name: str
    scheduled_segment_count: int
    maximal_contiguous_occurrence_count: int
    recurrence_available: bool
    recurrence_unavailable_reason: str | None
    recurrence_comparison_count: int | None
    mean_entry_retention_delta: float | None
    worst_entry_retention_delta: float | None
    mean_tail_backward_transfer: float | None
    worst_tail_backward_transfer: float | None
    occurrence_tail_slope: float | None
    recovered_comparison_count: int | None
    recovery_fraction: float | None
    mean_recovery_steps_after_entry_for_recovered: float | None
    mean_positive_gap_recovery_area: float | None
    first_exposure_occurrence_index: int
    first_exposure_tail_counterfactual: float
    peak_post_exposure_occurrence_index: int
    peak_post_exposure_tail_counterfactual: float
    latest_life_tail_counterfactual: float
    peak_to_latest_counterfactual_forgetting: float
    first_to_latest_counterfactual_transfer: float
    learner_inferred_known_obsolete: bool

    def to_config(self) -> dict[str, object]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True, slots=True)
class HCCLLongevityTrend:
    cycle_count: int
    segment_count: int
    occurrence_count: int
    first_cycle_task_mean: float
    latest_cycle_task_mean: float
    first_to_latest_cycle_task_change: float
    cycle_task_mean_slope: float
    first_cycle_segment_tail_mean: float
    latest_cycle_segment_tail_mean: float
    first_to_latest_cycle_segment_tail_change: float
    cycle_segment_tail_mean_slope: float
    global_segment_tail_mean_slope: float
    global_occurrence_tail_mean_slope: float

    def to_config(self) -> dict[str, object]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True, slots=True)
class HCCLLongevityEndpointReport:
    """Deterministic in-memory descriptive report for one complete longevity life."""

    schedule_profile: str
    total_steps: int
    cycle_count: int
    config_sha256: str
    trace_sha256: str
    segments: tuple[HCCLLongevitySegmentRecord, ...]
    segment_performance_matrix: tuple[tuple[float, float, float, float], ...]
    occurrences: tuple[HCCLLongevityOccurrenceRecord, ...]
    occurrence_performance_matrix: tuple[tuple[float, float, float, float], ...]
    recurrences: tuple[HCCLLongevityRecurrenceRecord, ...]
    cycles: tuple[HCCLLongevityCycleRecord, ...]
    regime_summaries: tuple[HCCLLongevityRegimeSummary, ...]
    lifetime_trend: HCCLLongevityTrend
    schema: str = HCCL_LONGEVITY_ENDPOINT_REPORT_SCHEMA
    status: str = HCCL_LONGEVITY_ENDPOINT_STATUS
    evidence_level: str = HCCL_LONGEVITY_ENDPOINT_EVIDENCE_LEVEL

    def to_config(self) -> dict[str, object]:
        """Return a JSON-compatible report payload; this method performs no write."""

        return {
            "schema": self.schema,
            "status": self.status,
            "evidence_level": self.evidence_level,
            "development_only": True,
            "schedule_profile": self.schedule_profile,
            "total_steps": self.total_steps,
            "transactions_committed": self.total_steps,
            "cycle_count": self.cycle_count,
            "schedule_segment_count": len(self.segments),
            "maximal_regime_occurrence_count": len(self.occurrences),
            "genuine_recurrence_comparison_count": len(self.recurrences),
            "window_steps": HCCL_LONGEVITY_ENDPOINT_WINDOW_STEPS,
            "config_sha256": self.config_sha256,
            "trace_sha256": self.trace_sha256,
            "reset_callback_count": 0,
            "boundary_callback_count": 0,
            "learner_visible_regime_labels": False,
            "learner_visible_schedule_boundaries": False,
            "counterfactual_score_columns_exposed_to_learner": False,
            "segments": [record.to_config() for record in self.segments],
            "segment_performance_matrix": [
                list(row) for row in self.segment_performance_matrix
            ],
            "occurrences": [record.to_config() for record in self.occurrences],
            "occurrence_performance_matrix": [
                list(row) for row in self.occurrence_performance_matrix
            ],
            "recurrences": [record.to_config() for record in self.recurrences],
            "cycles": [record.to_config() for record in self.cycles],
            "regime_summaries": [record.to_config() for record in self.regime_summaries],
            "lifetime_trend": self.lifetime_trend.to_config(),
            "d_maximal_regime_occurrence_count": 1,
            "d_recurrence_available": False,
            "d_recurrence_dependent_metrics": None,
            "d_known_obsolete_inferred_by_learner": False,
            "acceptance_thresholds_defined": False,
            "benchmark_execution_authorized": False,
            "output_writes_authorized": False,
            "artifact_authorized": False,
            "evidence_authorized": False,
            "seed_reservation_or_consumption_authorized": False,
            "promotion_authorized": False,
            "limitations": list(HCCL_LONGEVITY_ENDPOINT_LIMITATIONS),
        }


def _trace_sha256(trace: HCCLLongevityCompleteTrace) -> str:
    digest = hashlib.sha256()
    digest.update(b"alberta.hccl-longevity-endpoints.trace-digest.v1\0")
    for name in (
        "regime_ids",
        "transaction_committed",
        "pre_step_words",
        "post_step_words",
        "task_scores",
        "net_rewards",
        "all_regime_score_matrix",
    ):
        array = cast(NDArray[Any], getattr(trace, name))
        digest.update(name.encode("ascii") + b"\0")
        digest.update(array.dtype.str.encode("ascii") + b"\0")
        digest.update(_canonical_json_bytes(list(array.shape)))
        digest.update(memoryview(array).cast("B"))
    digest.update(
        _canonical_json_bytes(
            {
                "schema": trace.schema,
                "schedule_profile": trace.schedule_profile,
                "reset_callback_count": trace.reset_callback_count,
                "boundary_callback_count": trace.boundary_callback_count,
                "learner_received_evaluator_regime_ids": (
                    trace.learner_received_evaluator_regime_ids
                ),
                "learner_received_evaluator_schedule_boundaries": (
                    trace.learner_received_evaluator_schedule_boundaries
                ),
                "learner_received_counterfactual_scores": (
                    trace.learner_received_counterfactual_scores
                ),
            }
        )
    )
    return digest.hexdigest()


def _segment_records(
    trace: HCCLLongevityCompleteTrace,
) -> tuple[HCCLLongevitySegmentRecord, ...]:
    schedule, _cycles, _steps = _profile_geometry(trace.schedule_profile)
    records: list[HCCLLongevitySegmentRecord] = []
    for segment_index, (regime_name, start, end) in enumerate(schedule):
        cycle_index, canonical_segment_index = divmod(
            segment_index, _CANONICAL_PHASES_PER_CYCLE
        )
        regime_id = HCCL_CAUSAL_CORE_REGIME_NAMES.index(regime_name)
        entry = slice(start, start + HCCL_LONGEVITY_ENDPOINT_WINDOW_STEPS)
        tail = slice(end - HCCL_LONGEVITY_ENDPOINT_WINDOW_STEPS, end)
        records.append(
            HCCLLongevitySegmentRecord(
                segment_index=segment_index,
                cycle_index=cycle_index,
                canonical_segment_index=canonical_segment_index,
                regime_id=regime_id,
                regime_name=regime_name,
                start_step=start,
                end_step_exclusive=end,
                entry_task_mean=_mean(trace.task_scores[entry]),
                tail_task_mean=_mean(trace.task_scores[tail]),
                entry_net_reward_means=_agent_means(trace.net_rewards[entry]),
                tail_net_reward_means=_agent_means(trace.net_rewards[tail]),
            )
        )
    return tuple(records)


def _occurrence_records(
    trace: HCCLLongevityCompleteTrace,
) -> tuple[HCCLLongevityOccurrenceRecord, ...]:
    schedule, _cycles, _steps = _profile_geometry(trace.schedule_profile)
    counts = {name: 0 for name in HCCL_CAUSAL_CORE_REGIME_NAMES}
    records: list[HCCLLongevityOccurrenceRecord] = []
    for occurrence_index, (
        regime_name,
        start,
        end,
        first_segment,
        last_segment_exclusive,
    ) in enumerate(_maximal_schedule_runs(schedule)):
        regime_id = HCCL_CAUSAL_CORE_REGIME_NAMES.index(regime_name)
        regime_occurrence_index = counts[regime_name]
        counts[regime_name] += 1
        entry = slice(start, start + HCCL_LONGEVITY_ENDPOINT_WINDOW_STEPS)
        tail = slice(end - HCCL_LONGEVITY_ENDPOINT_WINDOW_STEPS, end)
        records.append(
            HCCLLongevityOccurrenceRecord(
                occurrence_index=occurrence_index,
                regime_id=regime_id,
                regime_name=regime_name,
                regime_occurrence_index=regime_occurrence_index,
                first_segment_index=first_segment,
                last_segment_index_exclusive=last_segment_exclusive,
                start_cycle_index=first_segment // _CANONICAL_PHASES_PER_CYCLE,
                end_cycle_index_inclusive=(
                    (last_segment_exclusive - 1) // _CANONICAL_PHASES_PER_CYCLE
                ),
                start_step=start,
                end_step_exclusive=end,
                entry_task_mean=_mean(trace.task_scores[entry]),
                tail_task_mean=_mean(trace.task_scores[tail]),
                entry_net_reward_means=_agent_means(trace.net_rewards[entry]),
                tail_net_reward_means=_agent_means(trace.net_rewards[tail]),
            )
        )
    return tuple(records)


def _tail_performance_matrix(
    trace: HCCLLongevityCompleteTrace,
    end_steps_exclusive: Sequence[int],
) -> tuple[tuple[float, float, float, float], ...]:
    rows: list[tuple[float, float, float, float]] = []
    for end in end_steps_exclusive:
        tail = trace.all_regime_score_matrix[
            end - HCCL_LONGEVITY_ENDPOINT_WINDOW_STEPS : end
        ]
        means = np.mean(tail, axis=0, dtype=np.float64)
        rows.append((float(means[0]), float(means[1]), float(means[2]), float(means[3])))
    return tuple(rows)


def _trailing_means(values: Float32Array) -> NDArray[np.float64]:
    cumulative = np.concatenate(
        (np.zeros((1,), dtype=np.float64), np.cumsum(values, dtype=np.float64))
    )
    window = HCCL_LONGEVITY_ENDPOINT_WINDOW_STEPS
    return (cumulative[window:] - cumulative[:-window]) / float(window)


def _recurrence_records(
    trace: HCCLLongevityCompleteTrace,
    occurrences: tuple[HCCLLongevityOccurrenceRecord, ...],
) -> tuple[HCCLLongevityRecurrenceRecord, ...]:
    previous: dict[str, HCCLLongevityOccurrenceRecord] = {}
    records: list[HCCLLongevityRecurrenceRecord] = []
    for current in occurrences:
        prior = previous.get(current.regime_name)
        previous[current.regime_name] = current
        if prior is None:
            continue
        if current.regime_name == "D":
            raise AssertionError("the fixed longevity schedule cannot recur D")
        intervening_segments = (
            current.first_segment_index - prior.last_segment_index_exclusive
        )
        if intervening_segments < 1 or current.start_step == prior.end_step_exclusive:
            raise AssertionError(
                "recurrence comparisons require an intervening differently labelled segment"
            )

        reference = prior.tail_task_mean
        current_scores = trace.task_scores[current.start_step : current.end_step_exclusive]
        rolling = _trailing_means(current_scores)
        recovered_indices = np.flatnonzero(rolling >= reference)
        recovery_index = int(recovered_indices[0]) if recovered_indices.size else None
        area_stop = rolling.size if recovery_index is None else recovery_index + 1
        recovery_area = float(np.maximum(reference - rolling[:area_stop], 0.0).sum())
        recovery_endpoint = (
            None
            if recovery_index is None
            else current.start_step
            + HCCL_LONGEVITY_ENDPOINT_WINDOW_STEPS
            + recovery_index
        )
        entry_retention = current.entry_task_mean - reference
        tail_transfer = current.tail_task_mean - reference
        records.append(
            HCCLLongevityRecurrenceRecord(
                regime_id=current.regime_id,
                regime_name=current.regime_name,
                previous_occurrence_index=prior.occurrence_index,
                current_occurrence_index=current.occurrence_index,
                previous_last_segment_index_exclusive=(
                    prior.last_segment_index_exclusive
                ),
                current_first_segment_index=current.first_segment_index,
                intervening_segment_count=intervening_segments,
                previous_end_cycle_index_inclusive=prior.end_cycle_index_inclusive,
                current_start_cycle_index=current.start_cycle_index,
                previous_regime_occurrence_index=prior.regime_occurrence_index,
                current_regime_occurrence_index=current.regime_occurrence_index,
                prior_tail_reference=reference,
                current_entry_mean=current.entry_task_mean,
                current_tail_mean=current.tail_task_mean,
                entry_retention_delta=entry_retention,
                entry_forgetting_gap=-entry_retention,
                tail_backward_transfer=tail_transfer,
                tail_forgetting_gap=-tail_transfer,
                trailing64_recovered=recovery_index is not None,
                trailing64_recovery_steps_after_entry=recovery_index,
                trailing64_recovery_endpoint_step_exclusive=recovery_endpoint,
                positive_gap_recovery_area=recovery_area,
            )
        )
    return tuple(records)


def _cycle_records(
    trace: HCCLLongevityCompleteTrace,
    segments: tuple[HCCLLongevitySegmentRecord, ...],
    occurrences: tuple[HCCLLongevityOccurrenceRecord, ...],
) -> tuple[HCCLLongevityCycleRecord, ...]:
    records: list[HCCLLongevityCycleRecord] = []
    previous_task_mean: float | None = None
    previous_tail_mean: float | None = None
    for cycle_index in range(trace.cycle_count):
        cycle_segments = segments[
            cycle_index
            * _CANONICAL_PHASES_PER_CYCLE : (cycle_index + 1)
            * _CANONICAL_PHASES_PER_CYCLE
        ]
        if len(cycle_segments) != _CANONICAL_PHASES_PER_CYCLE:
            raise AssertionError("fixed longevity cycles must contain ten segments")
        start = cycle_segments[0].start_step
        end = cycle_segments[-1].end_step_exclusive
        task_mean = _mean(trace.task_scores[start:end])
        entry_mean = float(
            np.mean(
                np.asarray(
                    [record.entry_task_mean for record in cycle_segments],
                    dtype=np.float64,
                )
            )
        )
        tail_mean = float(
            np.mean(
                np.asarray(
                    [record.tail_task_mean for record in cycle_segments],
                    dtype=np.float64,
                )
            )
        )
        per_regime: list[float | None] = []
        for regime_name in HCCL_CAUSAL_CORE_REGIME_NAMES:
            values = [
                record.tail_task_mean
                for record in cycle_segments
                if record.regime_name == regime_name
            ]
            per_regime.append(
                None
                if not values
                else float(np.mean(np.asarray(values, dtype=np.float64)))
            )
        end_tail = trace.all_regime_score_matrix[
            end - HCCL_LONGEVITY_ENDPOINT_WINDOW_STEPS : end
        ]
        end_tail_means = np.mean(end_tail, axis=0, dtype=np.float64)
        records.append(
            HCCLLongevityCycleRecord(
                cycle_index=cycle_index,
                start_step=start,
                end_step_exclusive=end,
                segment_count=len(cycle_segments),
                maximal_occurrence_start_count=sum(
                    record.start_cycle_index == cycle_index for record in occurrences
                ),
                maximal_occurrence_end_count=sum(
                    record.end_cycle_index_inclusive == cycle_index
                    for record in occurrences
                ),
                task_mean=task_mean,
                segment_entry_task_mean=entry_mean,
                segment_tail_task_mean=tail_mean,
                regime_segment_tail_task_means=(
                    per_regime[0],
                    per_regime[1],
                    per_regime[2],
                    per_regime[3],
                ),
                end_tail_all_regime_score_means=(
                    float(end_tail_means[0]),
                    float(end_tail_means[1]),
                    float(end_tail_means[2]),
                    float(end_tail_means[3]),
                ),
                task_mean_change_from_previous=(
                    None if previous_task_mean is None else task_mean - previous_task_mean
                ),
                segment_tail_mean_change_from_previous=(
                    None if previous_tail_mean is None else tail_mean - previous_tail_mean
                ),
            )
        )
        previous_task_mean = task_mean
        previous_tail_mean = tail_mean
    return tuple(records)


def _regime_summaries(
    segments: tuple[HCCLLongevitySegmentRecord, ...],
    occurrences: tuple[HCCLLongevityOccurrenceRecord, ...],
    recurrences: tuple[HCCLLongevityRecurrenceRecord, ...],
    occurrence_performance: tuple[tuple[float, float, float, float], ...],
) -> tuple[HCCLLongevityRegimeSummary, ...]:
    matrix = np.asarray(occurrence_performance, dtype=np.float64)
    summaries: list[HCCLLongevityRegimeSummary] = []
    for regime_id, regime_name in zip(
        _REGIME_IDS, HCCL_CAUSAL_CORE_REGIME_NAMES, strict=True
    ):
        matching_occurrences = tuple(
            record for record in occurrences if record.regime_id == regime_id
        )
        matching_segments = tuple(
            record for record in segments if record.regime_id == regime_id
        )
        matching_recurrences = tuple(
            record for record in recurrences if record.regime_id == regime_id
        )
        first_occurrence = matching_occurrences[0].occurrence_index
        longitudinal = matrix[first_occurrence:, regime_id]
        peak_offset = int(np.argmax(longitudinal))
        peak_occurrence = first_occurrence + peak_offset
        first_counterfactual = float(matrix[first_occurrence, regime_id])
        peak_counterfactual = float(matrix[peak_occurrence, regime_id])
        latest_counterfactual = float(matrix[-1, regime_id])
        recurrence_available = len(matching_occurrences) >= 2

        if recurrence_available:
            recovered = tuple(
                record for record in matching_recurrences if record.trailing64_recovered
            )
            recovered_steps = tuple(
                cast(int, record.trailing64_recovery_steps_after_entry)
                for record in recovered
            )
            comparison_count: int | None = len(matching_recurrences)
            mean_entry_retention: float | None = float(
                np.mean(
                    np.asarray(
                        [record.entry_retention_delta for record in matching_recurrences],
                        dtype=np.float64,
                    )
                )
            )
            worst_entry_retention: float | None = min(
                record.entry_retention_delta for record in matching_recurrences
            )
            mean_tail_transfer: float | None = float(
                np.mean(
                    np.asarray(
                        [record.tail_backward_transfer for record in matching_recurrences],
                        dtype=np.float64,
                    )
                )
            )
            worst_tail_transfer: float | None = min(
                record.tail_backward_transfer for record in matching_recurrences
            )
            tail_slope: float | None = _ordinary_slope(
                tuple(record.tail_task_mean for record in matching_occurrences)
            )
            recovered_count: int | None = len(recovered)
            recovery_fraction: float | None = len(recovered) / len(matching_recurrences)
            mean_recovery_steps: float | None = (
                None
                if not recovered_steps
                else float(np.mean(np.asarray(recovered_steps, dtype=np.float64)))
            )
            mean_recovery_area: float | None = float(
                np.mean(
                    np.asarray(
                        [record.positive_gap_recovery_area for record in matching_recurrences],
                        dtype=np.float64,
                    )
                )
            )
            unavailable_reason = None
        else:
            comparison_count = None
            mean_entry_retention = None
            worst_entry_retention = None
            mean_tail_transfer = None
            worst_tail_transfer = None
            tail_slope = None
            recovered_count = None
            recovery_fraction = None
            mean_recovery_steps = None
            mean_recovery_area = None
            unavailable_reason = (
                "D has one maximal contiguous occurrence; recurrence-dependent metrics are "
                "unavailable"
            )

        summaries.append(
            HCCLLongevityRegimeSummary(
                regime_id=regime_id,
                regime_name=regime_name,
                scheduled_segment_count=len(matching_segments),
                maximal_contiguous_occurrence_count=len(matching_occurrences),
                recurrence_available=recurrence_available,
                recurrence_unavailable_reason=unavailable_reason,
                recurrence_comparison_count=comparison_count,
                mean_entry_retention_delta=mean_entry_retention,
                worst_entry_retention_delta=worst_entry_retention,
                mean_tail_backward_transfer=mean_tail_transfer,
                worst_tail_backward_transfer=worst_tail_transfer,
                occurrence_tail_slope=tail_slope,
                recovered_comparison_count=recovered_count,
                recovery_fraction=recovery_fraction,
                mean_recovery_steps_after_entry_for_recovered=mean_recovery_steps,
                mean_positive_gap_recovery_area=mean_recovery_area,
                first_exposure_occurrence_index=first_occurrence,
                first_exposure_tail_counterfactual=first_counterfactual,
                peak_post_exposure_occurrence_index=peak_occurrence,
                peak_post_exposure_tail_counterfactual=peak_counterfactual,
                latest_life_tail_counterfactual=latest_counterfactual,
                peak_to_latest_counterfactual_forgetting=(
                    peak_counterfactual - latest_counterfactual
                ),
                first_to_latest_counterfactual_transfer=(
                    latest_counterfactual - first_counterfactual
                ),
                learner_inferred_known_obsolete=False,
            )
        )
    return tuple(summaries)


def _lifetime_trend(
    segments: tuple[HCCLLongevitySegmentRecord, ...],
    occurrences: tuple[HCCLLongevityOccurrenceRecord, ...],
    cycles: tuple[HCCLLongevityCycleRecord, ...],
) -> HCCLLongevityTrend:
    task_means = tuple(record.task_mean for record in cycles)
    cycle_tail_means = tuple(record.segment_tail_task_mean for record in cycles)
    segment_tail_means = tuple(record.tail_task_mean for record in segments)
    occurrence_tail_means = tuple(record.tail_task_mean for record in occurrences)
    return HCCLLongevityTrend(
        cycle_count=len(cycles),
        segment_count=len(segments),
        occurrence_count=len(occurrences),
        first_cycle_task_mean=task_means[0],
        latest_cycle_task_mean=task_means[-1],
        first_to_latest_cycle_task_change=task_means[-1] - task_means[0],
        cycle_task_mean_slope=_ordinary_slope(task_means),
        first_cycle_segment_tail_mean=cycle_tail_means[0],
        latest_cycle_segment_tail_mean=cycle_tail_means[-1],
        first_to_latest_cycle_segment_tail_change=(
            cycle_tail_means[-1] - cycle_tail_means[0]
        ),
        cycle_segment_tail_mean_slope=_ordinary_slope(cycle_tail_means),
        global_segment_tail_mean_slope=_ordinary_slope(segment_tail_means),
        global_occurrence_tail_mean_slope=_ordinary_slope(occurrence_tail_means),
    )


def evaluate_hccl_longevity_endpoints(
    trace: HCCLLongevityCompleteTrace,
    config: HCCLLongevityEndpointConfig | None = None,
) -> HCCLLongevityEndpointReport:
    """Recompute every descriptive endpoint from one exact complete life."""

    validate_hccl_longevity_complete_trace(trace)
    resolved_config = (
        HCCLLongevityEndpointConfig(schedule_profile=trace.schedule_profile)
        if config is None
        else config
    )
    if type(resolved_config) is not HCCLLongevityEndpointConfig:
        raise TypeError("config must be exact HCCLLongevityEndpointConfig")
    if resolved_config.schedule_profile != trace.schedule_profile:
        raise ValueError("endpoint config schedule profile does not match trace")
    segments = _segment_records(trace)
    occurrences = _occurrence_records(trace)
    segment_performance = _tail_performance_matrix(
        trace, tuple(record.end_step_exclusive for record in segments)
    )
    occurrence_performance = _tail_performance_matrix(
        trace, tuple(record.end_step_exclusive for record in occurrences)
    )
    recurrences = _recurrence_records(trace, occurrences)
    cycles = _cycle_records(trace, segments, occurrences)
    summaries = _regime_summaries(
        segments, occurrences, recurrences, occurrence_performance
    )
    return HCCLLongevityEndpointReport(
        schedule_profile=trace.schedule_profile,
        total_steps=trace.total_steps,
        cycle_count=trace.cycle_count,
        config_sha256=hashlib.sha256(
            _canonical_json_bytes(resolved_config.to_config())
        ).hexdigest(),
        trace_sha256=_trace_sha256(trace),
        segments=segments,
        segment_performance_matrix=segment_performance,
        occurrences=occurrences,
        occurrence_performance_matrix=occurrence_performance,
        recurrences=recurrences,
        cycles=cycles,
        regime_summaries=summaries,
        lifetime_trend=_lifetime_trend(segments, occurrences, cycles),
    )


def validate_hccl_longevity_endpoint_report(
    report: HCCLLongevityEndpointReport,
    trace: HCCLLongevityCompleteTrace,
    config: HCCLLongevityEndpointConfig | None = None,
) -> HCCLLongevityEndpointReport:
    """Fail closed unless a report exactly equals deterministic recomputation."""

    if type(report) is not HCCLLongevityEndpointReport:
        raise TypeError("report must be exact HCCLLongevityEndpointReport")
    expected = evaluate_hccl_longevity_endpoints(trace, config)
    if _canonical_json_bytes(report.to_config()) != _canonical_json_bytes(
        expected.to_config()
    ):
        raise ValueError("longevity endpoint report differs from deterministic recomputation")
    return report


__all__ = [
    "HCCL_LONGEVITY_COMPLETE_TRACE_SCHEMA",
    "HCCL_LONGEVITY_ENDPOINT_CONFIG_SCHEMA",
    "HCCL_LONGEVITY_ENDPOINT_EVIDENCE_LEVEL",
    "HCCL_LONGEVITY_ENDPOINT_LIMITATIONS",
    "HCCL_LONGEVITY_ENDPOINT_METRIC_DEFINITIONS",
    "HCCL_LONGEVITY_ENDPOINT_REPORT_SCHEMA",
    "HCCL_LONGEVITY_ENDPOINT_STATUS",
    "HCCL_LONGEVITY_ENDPOINT_WINDOW_STEPS",
    "HCCLLongevityCompleteTrace",
    "HCCLLongevityCycleRecord",
    "HCCLLongevityEndpointConfig",
    "HCCLLongevityEndpointReport",
    "HCCLLongevityOccurrenceRecord",
    "HCCLLongevityRecurrenceRecord",
    "HCCLLongevityRegimeSummary",
    "HCCLLongevitySegmentRecord",
    "HCCLLongevityTrend",
    "evaluate_hccl_longevity_endpoints",
    "validate_hccl_longevity_complete_trace",
    "validate_hccl_longevity_endpoint_report",
]
