"""Pure development-only endpoint metrics for one complete HCCL causal-core life.

This module consumes an already executed, uninterrupted canonical HCCL trace.
It does not run a learner or world, choose actions, write an artifact, reserve a
seed, define an acceptance threshold, or authorize evidence or promotion.  The
regime labels and the four counterfactual score columns belong only to this
evaluator and must never be supplied to a learner.

Endpoint conventions are fixed.  Every phase entry and tail is 64 committed
transitions.  A phase-performance row is the mean of each evaluator-only score
column over that phase's trailing 64 transitions.  For a recurrence, the prior
same-regime tail is the descriptive reference.  Signed entry and tail gaps are
``prior_tail - current_mean`` (positive means degradation), while tail backward
transfer is ``current_tail - prior_tail``.

Recovery uses exact trailing-64 task-score means within the current occurrence.
The first eligible endpoint is its entry-window endpoint.  ``recovery_steps``
counts additional committed transitions after that endpoint (so immediate
recovery is zero), and recovery occurs at the first mean greater than or equal
to the prior-tail reference.  Positive-gap recovery area sums
``max(prior_tail - trailing_mean, 0)`` from the entry endpoint through the first
recovery endpoint, or through the occurrence tail when recovery is absent.
This is a descriptive reference comparison, not an acceptance gate.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any, Final, cast

import numpy as np
from numpy.typing import NDArray

from alberta_framework.streams.hccl_causal_core import (
    HCCL_CAUSAL_CORE_REGIME_NAMES,
    HCCL_CAUSAL_CORE_SCHEDULE,
    HCCL_REGIME_A,
    HCCL_REGIME_B,
    HCCL_REGIME_C,
    HCCL_REGIME_D,
)

HCCL_CAUSAL_CORE_ENDPOINT_CONFIG_SCHEMA: Final = (
    "alberta.hccl-causal-core-endpoints.config.v1"
)
HCCL_CAUSAL_CORE_COMPLETE_TRACE_SCHEMA: Final = (
    "alberta.hccl-causal-core-endpoints.complete-trace.v1"
)
HCCL_CAUSAL_CORE_ENDPOINT_REPORT_SCHEMA: Final = (
    "alberta.hccl-causal-core-endpoints.report.v1"
)
HCCL_CAUSAL_CORE_ENDPOINT_STATUS: Final = "not_assessed"
HCCL_CAUSAL_CORE_ENDPOINT_EVIDENCE_LEVEL: Final = "L0-development-diagnostic-only"
HCCL_CAUSAL_CORE_ENDPOINT_ENTRY_WINDOW: Final = 64
HCCL_CAUSAL_CORE_ENDPOINT_TAIL_WINDOW: Final = 64
HCCL_CAUSAL_CORE_ENDPOINT_TOTAL_STEPS: Final = 8_998

HCCL_CAUSAL_CORE_ENDPOINT_LIMITATIONS: Final = (
    "development-only descriptive endpoint evaluator",
    "no acceptance thresholds or efficacy decision",
    "no benchmark execution, output writer, artifact, or evidence path",
    "no seed reservation, consumption, or held-out-seed authority",
    "no scientific-promotion or Alberta-Plan-completion authority",
    "evaluator regime labels and counterfactual score columns are never learner inputs",
    "one finite 8998-transition life does not establish indefinite continual operation",
)

HCCL_CAUSAL_CORE_ENDPOINT_METRIC_DEFINITIONS: Final = (
    (
        "occurrence_entry",
        "arithmetic means over the first 64 committed transitions of each phase",
    ),
    (
        "occurrence_tail",
        "arithmetic means over the final 64 committed transitions of each phase",
    ),
    (
        "phase_performance_matrix",
        "row=phase, column=A/B/C/D; trailing-64 mean of evaluator-only score columns",
    ),
    (
        "recurrence_gaps",
        "prior same-regime tail minus current entry or tail; positive means degradation",
    ),
    (
        "tail_backward_transfer",
        "current same-regime tail minus the preceding same-regime tail",
    ),
    (
        "trailing64_recovery",
        "first current-phase trailing-64 task mean at least the preceding tail; steps count "
        "additional transitions after the entry endpoint",
    ),
    (
        "positive_gap_recovery_area",
        "sum of positive prior-tail minus trailing-64 gaps through recovery, or phase end",
    ),
    (
        "recurrence_slope",
        "ordinary-least-squares slope of occurrence tail means versus zero-based occurrence "
        "ordinal",
    ),
    (
        "peak_to_latest_forgetting",
        "largest phase-tail counterfactual performance since first exposure minus final-phase "
        "performance",
    ),
    (
        "backward_transfer",
        "final-phase counterfactual performance minus first-exposure phase-tail performance",
    ),
)

_EXPECTED_SCHEDULE: Final = (
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
_REGIME_IDS: Final = (HCCL_REGIME_A, HCCL_REGIME_B, HCCL_REGIME_C, HCCL_REGIME_D)

if HCCL_CAUSAL_CORE_SCHEDULE != _EXPECTED_SCHEDULE:
    raise RuntimeError("the imported HCCL canonical schedule has changed")
if HCCL_CAUSAL_CORE_REGIME_NAMES != ("A", "B", "C", "D") or _REGIME_IDS != (0, 1, 2, 3):
    raise RuntimeError("the imported HCCL evaluator regime identifiers have changed")

type Float32Array = NDArray[np.float32]
type Int32Array = NDArray[np.int32]
type UInt32Array = NDArray[np.uint32]
type BoolArray = NDArray[np.bool_]


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _strict_json_equal(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        left_mapping = cast(dict[object, object], left)
        right_mapping = cast(dict[object, object], right)
        return set(left_mapping) == set(right_mapping) and all(
            _strict_json_equal(left_mapping[key], right_mapping[key]) for key in left_mapping
        )
    if type(left) is list:
        left_list = cast(list[object], left)
        right_list = cast(list[object], right)
        return len(left_list) == len(right_list) and all(
            _strict_json_equal(a, b) for a, b in zip(left_list, right_list, strict=True)
        )
    return left == right


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _exact_int(
    value: object,
    *,
    name: str,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if type(value) is not int or value < minimum or (
        maximum is not None and value > maximum
    ):
        raise ValueError(f"{name} must be an exact bounded integer")
    return value


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


def _expected_regime_ids() -> Int32Array:
    result = np.empty((HCCL_CAUSAL_CORE_ENDPOINT_TOTAL_STEPS,), dtype=np.int32)
    for regime_name, start, end in HCCL_CAUSAL_CORE_SCHEDULE:
        result[start:end] = HCCL_CAUSAL_CORE_REGIME_NAMES.index(regime_name)
    result.flags.writeable = False
    return result


_EXPECTED_REGIME_IDS = _expected_regime_ids()


@dataclasses.dataclass(frozen=True, slots=True)
class HCCLCausalCoreEndpointConfig:
    """The single fixed, JSON-roundtrippable canonical-life metric configuration."""

    schema: str = HCCL_CAUSAL_CORE_ENDPOINT_CONFIG_SCHEMA
    entry_window_steps: int = HCCL_CAUSAL_CORE_ENDPOINT_ENTRY_WINDOW
    tail_window_steps: int = HCCL_CAUSAL_CORE_ENDPOINT_TAIL_WINDOW
    total_steps: int = HCCL_CAUSAL_CORE_ENDPOINT_TOTAL_STEPS

    def __post_init__(self) -> None:
        expected = (
            HCCL_CAUSAL_CORE_ENDPOINT_CONFIG_SCHEMA,
            HCCL_CAUSAL_CORE_ENDPOINT_ENTRY_WINDOW,
            HCCL_CAUSAL_CORE_ENDPOINT_TAIL_WINDOW,
            HCCL_CAUSAL_CORE_ENDPOINT_TOTAL_STEPS,
        )
        actual = (
            self.schema,
            self.entry_window_steps,
            self.tail_window_steps,
            self.total_steps,
        )
        if any(type(value) is not type(reference) for value, reference in zip(actual, expected)):
            raise ValueError("HCCL endpoint config fields must retain their exact types")
        if actual != expected:
            raise ValueError("HCCL endpoint config is a fixed canonical development resolution")
        if HCCL_CAUSAL_CORE_SCHEDULE != _EXPECTED_SCHEDULE:
            raise RuntimeError("the imported HCCL canonical schedule has changed")

    def to_config(self) -> dict[str, object]:
        """Return the complete JSON-compatible fixed configuration and nonclaims."""

        return {
            "type": type(self).__name__,
            "schema": self.schema,
            "trace_schema": HCCL_CAUSAL_CORE_COMPLETE_TRACE_SCHEMA,
            "report_schema": HCCL_CAUSAL_CORE_ENDPOINT_REPORT_SCHEMA,
            "status": HCCL_CAUSAL_CORE_ENDPOINT_STATUS,
            "evidence_level": HCCL_CAUSAL_CORE_ENDPOINT_EVIDENCE_LEVEL,
            "development_only": True,
            "schedule": [
                {"regime": name, "start": start, "end": end}
                for name, start, end in HCCL_CAUSAL_CORE_SCHEDULE
            ],
            "regime_id_order": list(HCCL_CAUSAL_CORE_REGIME_NAMES),
            "total_steps": self.total_steps,
            "entry_window_steps": self.entry_window_steps,
            "tail_window_steps": self.tail_window_steps,
            "complete_trace_required": True,
            "all_transactions_must_be_committed": True,
            "reset_callbacks_required": 0,
            "boundary_callbacks_required": 0,
            "evaluator_labels_exposed_to_learner": False,
            "counterfactual_score_columns_exposed_to_learner": False,
            "metric_definitions": dict(HCCL_CAUSAL_CORE_ENDPOINT_METRIC_DEFINITIONS),
            "acceptance_thresholds_defined": False,
            "benchmark_execution_authorized": False,
            "output_writes_authorized": False,
            "artifact_authorized": False,
            "evidence_authorized": False,
            "seed_reservation_or_consumption_authorized": False,
            "promotion_authorized": False,
            "limitations": list(HCCL_CAUSAL_CORE_ENDPOINT_LIMITATIONS),
        }

    @classmethod
    def from_config(cls, payload: Mapping[str, object]) -> HCCLCausalCoreEndpointConfig:
        """Fail closed unless a JSON-decoded payload is the exact fixed configuration."""

        if not isinstance(payload, Mapping) or any(type(key) is not str for key in payload):
            raise TypeError("endpoint config payload must be a string-keyed mapping")
        config = cls()
        if not _strict_json_equal(dict(payload), config.to_config()):
            raise ValueError("endpoint config payload differs from the fixed configuration")
        return config


@dataclasses.dataclass(frozen=True, slots=True)
class HCCLCausalCoreCompleteTrace:
    """One exact, uninterrupted, host-resident canonical HCCL transaction trace."""

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
    learner_received_counterfactual_scores: bool = False
    schema: str = HCCL_CAUSAL_CORE_COMPLETE_TRACE_SCHEMA

    def __post_init__(self) -> None:
        n = HCCL_CAUSAL_CORE_ENDPOINT_TOTAL_STEPS
        specifications = (
            ("regime_ids", (n,), np.dtype(np.int32)),
            ("transaction_committed", (n,), np.dtype(np.bool_)),
            ("pre_step_words", (n, 2), np.dtype(np.uint32)),
            ("post_step_words", (n, 2), np.dtype(np.uint32)),
            ("task_scores", (n,), np.dtype(np.float32)),
            ("net_rewards", (n, 2), np.dtype(np.float32)),
            ("all_regime_score_matrix", (n, 4), np.dtype(np.float32)),
        )
        for name, shape, dtype in specifications:
            object.__setattr__(
                self,
                name,
                _frozen_array(getattr(self, name), name=name, shape=shape, dtype=dtype),
            )
        _validate_complete_trace_fields(self)


def _validate_complete_trace_fields(trace: HCCLCausalCoreCompleteTrace) -> None:
    n = HCCL_CAUSAL_CORE_ENDPOINT_TOTAL_STEPS
    for name, shape, dtype in (
        ("regime_ids", (n,), np.dtype(np.int32)),
        ("transaction_committed", (n,), np.dtype(np.bool_)),
        ("pre_step_words", (n, 2), np.dtype(np.uint32)),
        ("post_step_words", (n, 2), np.dtype(np.uint32)),
        ("task_scores", (n,), np.dtype(np.float32)),
        ("net_rewards", (n, 2), np.dtype(np.float32)),
        ("all_regime_score_matrix", (n, 4), np.dtype(np.float32)),
    ):
        value = getattr(trace, name)
        if type(value) is not np.ndarray:
            raise TypeError(f"trace.{name} must remain an exact numpy.ndarray")
        array = cast(NDArray[Any], value)
        if array.shape != shape or array.dtype != dtype:
            raise ValueError(f"trace.{name} no longer has its exact shape and dtype")
        if array.flags.writeable or not array.flags.c_contiguous:
            raise ValueError(f"trace.{name} must remain a read-only C-contiguous array")
    if type(trace.schema) is not str or trace.schema != HCCL_CAUSAL_CORE_COMPLETE_TRACE_SCHEMA:
        raise ValueError("trace schema differs from the fixed complete-trace schema")
    for name in ("reset_callback_count", "boundary_callback_count"):
        value = getattr(trace, name)
        if type(value) is not int or value != 0:
            raise ValueError(f"trace.{name} must be the exact integer zero")
    for name in (
        "learner_received_evaluator_regime_ids",
        "learner_received_counterfactual_scores",
    ):
        value = getattr(trace, name)
        if type(value) is not bool or value:
            raise ValueError(f"trace.{name} must be exact False")

    if not np.array_equal(trace.regime_ids, _EXPECTED_REGIME_IDS):
        raise ValueError("trace regime_ids do not equal the canonical HCCL schedule")
    if not bool(np.all(trace.transaction_committed)):
        raise ValueError("every canonical HCCL transaction must be committed")

    expected_pre = np.zeros((HCCL_CAUSAL_CORE_ENDPOINT_TOTAL_STEPS, 2), dtype=np.uint32)
    expected_post = np.zeros_like(expected_pre)
    expected_pre[:, 1] = np.arange(HCCL_CAUSAL_CORE_ENDPOINT_TOTAL_STEPS, dtype=np.uint32)
    expected_post[:, 1] = np.arange(
        1, HCCL_CAUSAL_CORE_ENDPOINT_TOTAL_STEPS + 1, dtype=np.uint32
    )
    if not np.array_equal(trace.pre_step_words, expected_pre):
        raise ValueError("trace pre_step_words are not the exact monotone canonical clocks")
    if not np.array_equal(trace.post_step_words, expected_post):
        raise ValueError("trace post_step_words are not the exact committed canonical clocks")

    for name in ("task_scores", "net_rewards", "all_regime_score_matrix"):
        if not bool(np.all(np.isfinite(getattr(trace, name)))):
            raise ValueError(f"trace.{name} must be entirely finite")
    selected = trace.all_regime_score_matrix[
        np.arange(HCCL_CAUSAL_CORE_ENDPOINT_TOTAL_STEPS), trace.regime_ids
    ]
    if not np.array_equal(selected, trace.task_scores):
        raise ValueError(
            "each task score must exactly equal its evaluator-regime matrix column"
        )
    expected_net_rewards = np.broadcast_to(
        trace.task_scores[:, None],
        (HCCL_CAUSAL_CORE_ENDPOINT_TOTAL_STEPS, 2),
    )
    if not np.array_equal(trace.net_rewards, expected_net_rewards):
        raise ValueError(
            "causal-core net rewards must exactly equal task score for both agents"
        )


def validate_hccl_causal_core_complete_trace(
    trace: HCCLCausalCoreCompleteTrace,
) -> HCCLCausalCoreCompleteTrace:
    """Revalidate and return an exact complete trace, including after possible mutation."""

    if type(trace) is not HCCLCausalCoreCompleteTrace:
        raise TypeError("trace must be exact HCCLCausalCoreCompleteTrace")
    _validate_complete_trace_fields(trace)
    return trace


@dataclasses.dataclass(frozen=True, slots=True)
class HCCLOccurrenceEndpointRecord:
    phase_index: int
    regime_id: int
    regime_name: str
    occurrence_index: int
    start_step: int
    end_step_exclusive: int
    entry_task_mean: float
    tail_task_mean: float
    entry_net_reward_means: tuple[float, float]
    tail_net_reward_means: tuple[float, float]

    def to_config(self) -> dict[str, object]:
        return {
            "phase_index": self.phase_index,
            "regime_id": self.regime_id,
            "regime_name": self.regime_name,
            "occurrence_index": self.occurrence_index,
            "start_step": self.start_step,
            "end_step_exclusive": self.end_step_exclusive,
            "entry_task_mean": self.entry_task_mean,
            "tail_task_mean": self.tail_task_mean,
            "entry_net_reward_means": list(self.entry_net_reward_means),
            "tail_net_reward_means": list(self.tail_net_reward_means),
        }


@dataclasses.dataclass(frozen=True, slots=True)
class HCCLRecurrenceEndpointRecord:
    regime_id: int
    regime_name: str
    previous_occurrence_index: int
    current_occurrence_index: int
    previous_phase_index: int
    current_phase_index: int
    prior_tail_reference: float
    current_entry_mean: float
    current_tail_mean: float
    entry_gap: float
    tail_gap: float
    tail_backward_transfer: float
    trailing64_recovered: bool
    trailing64_recovery_steps_after_entry: int | None
    trailing64_recovery_endpoint_step_exclusive: int | None
    positive_gap_recovery_area: float

    def to_config(self) -> dict[str, object]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True, slots=True)
class HCCLRegimeEndpointSummary:
    regime_id: int
    regime_name: str
    scheduled_occurrence_count: int
    recurring: bool
    recurrence_available: bool
    recurrence_unavailable_reason: str | None
    learner_inferred_known_obsolete: bool
    recurrence_slope: float | None
    first_exposure_phase_index: int
    peak_phase_index: int
    first_exposure_tail_performance: float
    peak_tail_performance: float
    latest_tail_performance: float
    peak_to_latest_forgetting: float
    backward_transfer: float

    def to_config(self) -> dict[str, object]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True, slots=True)
class HCCLWorstRecurringGap:
    metric: str
    value: float
    regime_id: int
    regime_name: str
    previous_phase_index: int
    current_phase_index: int

    def to_config(self) -> dict[str, object]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True, slots=True)
class HCCLCausalCoreEndpointReport:
    """Frozen in-memory descriptive records for one validated canonical life."""

    config_sha256: str
    trace_sha256: str
    occurrences: tuple[HCCLOccurrenceEndpointRecord, ...]
    phase_performance_matrix: tuple[tuple[float, float, float, float], ...]
    recurrences: tuple[HCCLRecurrenceEndpointRecord, ...]
    regime_summaries: tuple[HCCLRegimeEndpointSummary, ...]
    worst_recurring_entry_gap: HCCLWorstRecurringGap
    worst_recurring_tail_gap: HCCLWorstRecurringGap
    schema: str = HCCL_CAUSAL_CORE_ENDPOINT_REPORT_SCHEMA
    status: str = HCCL_CAUSAL_CORE_ENDPOINT_STATUS
    evidence_level: str = HCCL_CAUSAL_CORE_ENDPOINT_EVIDENCE_LEVEL

    def to_config(self) -> dict[str, object]:
        """Return a JSON-compatible in-memory report payload; this is not a writer."""

        return {
            "schema": self.schema,
            "status": self.status,
            "evidence_level": self.evidence_level,
            "development_only": True,
            "config_sha256": self.config_sha256,
            "trace_sha256": self.trace_sha256,
            "total_steps": HCCL_CAUSAL_CORE_ENDPOINT_TOTAL_STEPS,
            "transactions_committed": HCCL_CAUSAL_CORE_ENDPOINT_TOTAL_STEPS,
            "reset_callback_count": 0,
            "boundary_callback_count": 0,
            "entry_window_steps": HCCL_CAUSAL_CORE_ENDPOINT_ENTRY_WINDOW,
            "tail_window_steps": HCCL_CAUSAL_CORE_ENDPOINT_TAIL_WINDOW,
            "regime_id_order": list(HCCL_CAUSAL_CORE_REGIME_NAMES),
            "occurrences": [record.to_config() for record in self.occurrences],
            "phase_performance_matrix": [list(row) for row in self.phase_performance_matrix],
            "recurrences": [record.to_config() for record in self.recurrences],
            "regime_summaries": [record.to_config() for record in self.regime_summaries],
            "worst_recurring_entry_gap": self.worst_recurring_entry_gap.to_config(),
            "worst_recurring_tail_gap": self.worst_recurring_tail_gap.to_config(),
            "d_recurrence_available": False,
            "d_known_obsolete_inferred_by_learner": False,
            "evaluator_labels_exposed_to_learner": False,
            "counterfactual_score_columns_exposed_to_learner": False,
            "acceptance_thresholds_defined": False,
            "benchmark_execution_authorized": False,
            "output_writes_authorized": False,
            "artifact_authorized": False,
            "evidence_authorized": False,
            "seed_reservation_or_consumption_authorized": False,
            "promotion_authorized": False,
            "limitations": list(HCCL_CAUSAL_CORE_ENDPOINT_LIMITATIONS),
        }


def _trace_sha256(trace: HCCLCausalCoreCompleteTrace) -> str:
    digest = hashlib.sha256()
    digest.update(b"alberta.hccl-causal-core-endpoints.trace-digest.v1\0")
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
        digest.update(array.tobytes(order="C"))
    digest.update(
        _canonical_json_bytes(
            {
                "schema": trace.schema,
                "reset_callback_count": trace.reset_callback_count,
                "boundary_callback_count": trace.boundary_callback_count,
                "learner_received_evaluator_regime_ids": (
                    trace.learner_received_evaluator_regime_ids
                ),
                "learner_received_counterfactual_scores": (
                    trace.learner_received_counterfactual_scores
                ),
            }
        )
    )
    return digest.hexdigest()


def _mean(values: NDArray[np.float32]) -> float:
    return float(np.mean(values, dtype=np.float64))


def _agent_means(values: NDArray[np.float32]) -> tuple[float, float]:
    means = np.mean(values, axis=0, dtype=np.float64)
    return (float(means[0]), float(means[1]))


def _occurrence_records(
    trace: HCCLCausalCoreCompleteTrace,
) -> tuple[HCCLOccurrenceEndpointRecord, ...]:
    counts = {name: 0 for name in HCCL_CAUSAL_CORE_REGIME_NAMES}
    records: list[HCCLOccurrenceEndpointRecord] = []
    for phase_index, (regime_name, start, end) in enumerate(HCCL_CAUSAL_CORE_SCHEDULE):
        counts[regime_name] += 1
        regime_id = HCCL_CAUSAL_CORE_REGIME_NAMES.index(regime_name)
        entry = slice(start, start + HCCL_CAUSAL_CORE_ENDPOINT_ENTRY_WINDOW)
        tail = slice(end - HCCL_CAUSAL_CORE_ENDPOINT_TAIL_WINDOW, end)
        records.append(
            HCCLOccurrenceEndpointRecord(
                phase_index=phase_index,
                regime_id=regime_id,
                regime_name=regime_name,
                occurrence_index=counts[regime_name],
                start_step=start,
                end_step_exclusive=end,
                entry_task_mean=_mean(trace.task_scores[entry]),
                tail_task_mean=_mean(trace.task_scores[tail]),
                entry_net_reward_means=_agent_means(trace.net_rewards[entry]),
                tail_net_reward_means=_agent_means(trace.net_rewards[tail]),
            )
        )
    return tuple(records)


def _phase_performance_matrix(
    trace: HCCLCausalCoreCompleteTrace,
) -> tuple[tuple[float, float, float, float], ...]:
    rows: list[tuple[float, float, float, float]] = []
    for _name, _start, end in HCCL_CAUSAL_CORE_SCHEDULE:
        tail = trace.all_regime_score_matrix[
            end - HCCL_CAUSAL_CORE_ENDPOINT_TAIL_WINDOW : end
        ]
        means = np.mean(tail, axis=0, dtype=np.float64)
        rows.append((float(means[0]), float(means[1]), float(means[2]), float(means[3])))
    return tuple(rows)


def _trailing_means(values: Float32Array, window: int) -> NDArray[np.float64]:
    cumulative = np.concatenate(
        (np.zeros((1,), dtype=np.float64), np.cumsum(values, dtype=np.float64))
    )
    return (cumulative[window:] - cumulative[:-window]) / float(window)


def _recurrence_records(
    trace: HCCLCausalCoreCompleteTrace,
    occurrences: tuple[HCCLOccurrenceEndpointRecord, ...],
) -> tuple[HCCLRecurrenceEndpointRecord, ...]:
    previous: dict[str, HCCLOccurrenceEndpointRecord] = {}
    records: list[HCCLRecurrenceEndpointRecord] = []
    for current in occurrences:
        prior = previous.get(current.regime_name)
        previous[current.regime_name] = current
        if prior is None or current.regime_name == "D":
            continue

        reference = prior.tail_task_mean
        current_scores = trace.task_scores[current.start_step : current.end_step_exclusive]
        rolling = _trailing_means(current_scores, HCCL_CAUSAL_CORE_ENDPOINT_ENTRY_WINDOW)
        recovered_indices = np.flatnonzero(rolling >= reference)
        recovered = bool(recovered_indices.size)
        recovery_index = int(recovered_indices[0]) if recovered else None
        area_stop = rolling.size if recovery_index is None else recovery_index + 1
        recovery_area = float(np.maximum(reference - rolling[:area_stop], 0.0).sum())
        endpoint = (
            None
            if recovery_index is None
            else current.start_step
            + HCCL_CAUSAL_CORE_ENDPOINT_ENTRY_WINDOW
            + recovery_index
        )
        entry_gap = reference - current.entry_task_mean
        tail_gap = reference - current.tail_task_mean
        records.append(
            HCCLRecurrenceEndpointRecord(
                regime_id=current.regime_id,
                regime_name=current.regime_name,
                previous_occurrence_index=prior.occurrence_index,
                current_occurrence_index=current.occurrence_index,
                previous_phase_index=prior.phase_index,
                current_phase_index=current.phase_index,
                prior_tail_reference=reference,
                current_entry_mean=current.entry_task_mean,
                current_tail_mean=current.tail_task_mean,
                entry_gap=entry_gap,
                tail_gap=tail_gap,
                tail_backward_transfer=-tail_gap,
                trailing64_recovered=recovered,
                trailing64_recovery_steps_after_entry=recovery_index,
                trailing64_recovery_endpoint_step_exclusive=endpoint,
                positive_gap_recovery_area=recovery_area,
            )
        )
    return tuple(records)


def _ordinary_slope(values: Sequence[float]) -> float:
    y = np.asarray(values, dtype=np.float64)
    x = np.arange(y.size, dtype=np.float64)
    centered_x = x - np.mean(x)
    centered_y = y - np.mean(y)
    return float(np.sum(centered_x * centered_y) / np.sum(centered_x * centered_x))


def _regime_summaries(
    occurrences: tuple[HCCLOccurrenceEndpointRecord, ...],
    phase_performance: tuple[tuple[float, float, float, float], ...],
) -> tuple[HCCLRegimeEndpointSummary, ...]:
    matrix = np.asarray(phase_performance, dtype=np.float64)
    summaries: list[HCCLRegimeEndpointSummary] = []
    for regime_id, regime_name in zip(_REGIME_IDS, HCCL_CAUSAL_CORE_REGIME_NAMES, strict=True):
        matching = tuple(record for record in occurrences if record.regime_id == regime_id)
        first_phase = matching[0].phase_index
        longitudinal = matrix[first_phase:, regime_id]
        peak_offset = int(np.argmax(longitudinal))
        peak_phase = first_phase + peak_offset
        first = float(matrix[first_phase, regime_id])
        peak = float(matrix[peak_phase, regime_id])
        latest = float(matrix[-1, regime_id])
        recurring = len(matching) >= 2
        summaries.append(
            HCCLRegimeEndpointSummary(
                regime_id=regime_id,
                regime_name=regime_name,
                scheduled_occurrence_count=len(matching),
                recurring=recurring,
                recurrence_available=recurring,
                recurrence_unavailable_reason=(
                    None
                    if recurring
                    else "D has one scheduled occurrence; recurrence metrics are unavailable"
                ),
                learner_inferred_known_obsolete=False,
                recurrence_slope=(
                    _ordinary_slope(tuple(record.tail_task_mean for record in matching))
                    if recurring
                    else None
                ),
                first_exposure_phase_index=first_phase,
                peak_phase_index=peak_phase,
                first_exposure_tail_performance=first,
                peak_tail_performance=peak,
                latest_tail_performance=latest,
                peak_to_latest_forgetting=peak - latest,
                backward_transfer=latest - first,
            )
        )
    return tuple(summaries)


def _worst_gap(
    records: tuple[HCCLRecurrenceEndpointRecord, ...], metric: str
) -> HCCLWorstRecurringGap:
    if metric not in {"entry_gap", "tail_gap"}:
        raise ValueError("worst-gap metric must be entry_gap or tail_gap")
    worst = max(records, key=lambda record: cast(float, getattr(record, metric)))
    return HCCLWorstRecurringGap(
        metric=metric,
        value=cast(float, getattr(worst, metric)),
        regime_id=worst.regime_id,
        regime_name=worst.regime_name,
        previous_phase_index=worst.previous_phase_index,
        current_phase_index=worst.current_phase_index,
    )


def evaluate_hccl_causal_core_endpoints(
    trace: HCCLCausalCoreCompleteTrace,
    config: HCCLCausalCoreEndpointConfig | None = None,
) -> HCCLCausalCoreEndpointReport:
    """Recompute all frozen endpoint records from one validated canonical trace."""

    resolved_config = HCCLCausalCoreEndpointConfig() if config is None else config
    if type(resolved_config) is not HCCLCausalCoreEndpointConfig:
        raise TypeError("config must be exact HCCLCausalCoreEndpointConfig")
    validate_hccl_causal_core_complete_trace(trace)
    occurrences = _occurrence_records(trace)
    phase_performance = _phase_performance_matrix(trace)
    recurrences = _recurrence_records(trace, occurrences)
    summaries = _regime_summaries(occurrences, phase_performance)
    return HCCLCausalCoreEndpointReport(
        config_sha256=hashlib.sha256(
            _canonical_json_bytes(resolved_config.to_config())
        ).hexdigest(),
        trace_sha256=_trace_sha256(trace),
        occurrences=occurrences,
        phase_performance_matrix=phase_performance,
        recurrences=recurrences,
        regime_summaries=summaries,
        worst_recurring_entry_gap=_worst_gap(recurrences, "entry_gap"),
        worst_recurring_tail_gap=_worst_gap(recurrences, "tail_gap"),
    )


def validate_hccl_causal_core_endpoint_report(
    report: HCCLCausalCoreEndpointReport,
    trace: HCCLCausalCoreCompleteTrace,
    config: HCCLCausalCoreEndpointConfig | None = None,
) -> HCCLCausalCoreEndpointReport:
    """Fail closed unless ``report`` exactly equals deterministic recomputation."""

    if type(report) is not HCCLCausalCoreEndpointReport:
        raise TypeError("report must be exact HCCLCausalCoreEndpointReport")
    expected = evaluate_hccl_causal_core_endpoints(trace, config)
    if not _strict_json_equal(report.to_config(), expected.to_config()):
        raise ValueError("endpoint report differs from deterministic trace recomputation")
    return report
