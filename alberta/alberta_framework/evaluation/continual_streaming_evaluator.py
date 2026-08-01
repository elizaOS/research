"""Bounded streaming executor for the continual evaluation-report core.

The executor owns task/regime scheduling and held-out probe selection.  A
learner receives only an observation and target, never a task or regime ID.
Every online score is formed from a prediction made before the corresponding
update, and evaluator probes never update learner state.

This is a development evaluation mechanism.  Its reports retain the
``not-assessed`` status of :mod:`continual_evaluation_report`; running it does
not create or promote scientific evidence.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from time import perf_counter_ns
from typing import Any, NoReturn, Protocol, cast, runtime_checkable

from alberta_framework.evaluation.continual_evaluation_report import (
    SCHEMA_VERSION as REPORT_SCHEMA_VERSION,
)
from alberta_framework.evaluation.continual_evaluation_report import (
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
    validate_continual_evaluation_report,
)

STREAMING_EVALUATOR_SCHEMA = "alberta.continual_streaming_evaluator.v2"
STREAMING_EVALUATOR_CHECKPOINT_SCHEMA = "alberta.continual_streaming_evaluator_checkpoint.v2"
STREAMING_EVALUATOR_REPORT_SCHEMA = "alberta.continual_streaming_evaluator_report.v1"

type DiagnosticScalar = bool | int | float | str


@dataclasses.dataclass(frozen=True)
class StreamingEvaluatorReportValidation:
    """Strict validation result for one evaluator-bound scalar report."""

    valid: bool
    errors: tuple[str, ...]


def _finite_float(value: object, *, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _positive_int(value: object, *, name: str) -> int:
    resolved = _nonnegative_int(value, name=name)
    if resolved == 0:
        raise ValueError(f"{name} must be positive")
    return resolved


def _versioned_identifier(value: object, *, name: str) -> str:
    if not isinstance(value, str) or ".v" not in value:
        raise ValueError(f"{name} must be a versioned identifier")
    return value


def _json_clone(value: object) -> object:
    return json.loads(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _atomic_write_json(value: object, path: str | Path) -> None:
    """Write standard JSON through an fsynced sibling and atomic replace."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    value,
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _validate_diagnostics(
    diagnostics: Mapping[str, DiagnosticScalar] | None,
    *,
    name: str,
) -> None:
    if diagnostics is None:
        return
    if not diagnostics:
        raise ValueError(f"{name} must be non-empty when supplied")
    for key, value in diagnostics.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{name} keys must be non-empty strings")
        if isinstance(value, bool | int):
            continue
        if isinstance(value, float) and math.isfinite(value):
            continue
        if isinstance(value, str) and value:
            continue
        raise ValueError(
            f"{name}.{key} must be a non-empty string, boolean, integer, or finite float"
        )


@dataclasses.dataclass(frozen=True)
class StreamingExample:
    """One learner-visible supervised streaming item with no regime identity."""

    observation: tuple[float, ...]
    target: float

    def __post_init__(self) -> None:
        if not isinstance(self.observation, tuple):
            raise ValueError("observation must be an immutable tuple")
        if not self.observation:
            raise ValueError("observation must be non-empty")
        for index, value in enumerate(self.observation):
            _finite_float(value, name=f"observation[{index}]")
        _finite_float(self.target, name="target")

    def to_config(self) -> dict[str, object]:
        return {"observation": list(self.observation), "target": self.target}


@dataclasses.dataclass(frozen=True)
class StreamingPrediction:
    """A learner prediction and its fail-closed runtime validity flag."""

    value: float
    valid: bool

    def __post_init__(self) -> None:
        if not isinstance(self.valid, bool):
            raise ValueError("prediction.valid must be boolean")
        if self.valid:
            _finite_float(self.value, name="prediction.value")


@dataclasses.dataclass(frozen=True)
class StreamingUpdate:
    """One learner update result.

    ``backward_latencies_ns`` contains separately measured backward operations
    when the learner can expose them.  It may be empty for closed-form updates.
    """

    state: Any
    applied: bool
    backward_latencies_ns: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.applied, bool):
            raise ValueError("update.applied must be boolean")
        for index, value in enumerate(self.backward_latencies_ns):
            _nonnegative_int(value, name=f"backward_latencies_ns[{index}]")


@dataclasses.dataclass(frozen=True)
class LearnerResourceUsage:
    """Caller-measured persistent and host-resident learner resources."""

    persistent_state_bytes: int
    parameter_count: int
    host_resident_bytes: int
    measurement_method: str

    def __post_init__(self) -> None:
        persistent = _nonnegative_int(
            self.persistent_state_bytes,
            name="persistent_state_bytes",
        )
        _nonnegative_int(self.parameter_count, name="parameter_count")
        host = _positive_int(self.host_resident_bytes, name="host_resident_bytes")
        if persistent > host:
            raise ValueError("persistent_state_bytes cannot exceed host_resident_bytes")
        if not isinstance(self.measurement_method, str) or not self.measurement_method:
            raise ValueError("measurement_method must be a non-empty string")


@runtime_checkable
class StreamingLearner(Protocol):
    """Learner-neutral state machine; task/regime IDs are intentionally absent."""

    @property
    def name(self) -> str: ...

    @property
    def max_backward_calls_per_update(self) -> int: ...

    @property
    def plasticity_applicable(self) -> bool: ...

    def to_config(self) -> dict[str, object]: ...

    def init(self) -> Any: ...

    def predict(
        self,
        state: Any,
        observation: tuple[float, ...],
    ) -> StreamingPrediction: ...

    def update(
        self,
        state: Any,
        observation: tuple[float, ...],
        target: float,
    ) -> StreamingUpdate: ...

    def state_to_config(self, state: Any) -> object: ...

    def state_from_config(self, payload: object) -> Any: ...

    def resource_usage(self, state: Any) -> LearnerResourceUsage: ...

    def diagnostics(
        self,
        state: Any,
    ) -> Mapping[str, DiagnosticScalar] | None: ...

    def plasticity_diagnostics(
        self,
        state: Any,
    ) -> Mapping[str, DiagnosticScalar] | None: ...


@runtime_checkable
class PredictionScore(Protocol):
    """Versioned scalar score applied outside the learner."""

    @property
    def higher_is_better(self) -> bool: ...

    def score(self, prediction: float, target: float) -> float: ...

    def to_config(self) -> dict[str, object]: ...


@dataclasses.dataclass(frozen=True)
class BoundedAbsoluteScore:
    """Score ``max(0, 1 - |prediction-target| / scale)``."""

    scale: float = 1.0

    def __post_init__(self) -> None:
        scale = _finite_float(self.scale, name="scale")
        if scale <= 0.0:
            raise ValueError("scale must be positive")

    @property
    def higher_is_better(self) -> bool:
        return True

    def score(self, prediction: float, target: float) -> float:
        prediction_value = _finite_float(prediction, name="prediction")
        target_value = _finite_float(target, name="target")
        return max(0.0, 1.0 - abs(prediction_value - target_value) / self.scale)

    def to_config(self) -> dict[str, object]:
        return {
            "type": "BoundedAbsoluteScore",
            "schema_version": "alberta.bounded_absolute_score.v1",
            "scale": self.scale,
        }


@dataclasses.dataclass(frozen=True)
class FrozenScalarState:
    prediction: float


class FrozenScalarBaseline:
    """No-learning scalar predictor baseline."""

    def __init__(self, name: str = "frozen", prediction: float = 0.0):
        if not isinstance(name, str) or not name:
            raise ValueError("name must be a non-empty string")
        self._name = name
        self._prediction = _finite_float(prediction, name="prediction")

    @property
    def name(self) -> str:
        return self._name

    @property
    def max_backward_calls_per_update(self) -> int:
        return 0

    @property
    def plasticity_applicable(self) -> bool:
        return False

    def to_config(self) -> dict[str, object]:
        return {
            "type": "FrozenScalarBaseline",
            "schema_version": "alberta.frozen_scalar_baseline.v1",
            "name": self._name,
            "prediction": self._prediction,
        }

    def init(self) -> FrozenScalarState:
        return FrozenScalarState(prediction=self._prediction)

    def predict(
        self,
        state: Any,
        observation: tuple[float, ...],
    ) -> StreamingPrediction:
        del observation
        resolved = cast(FrozenScalarState, state)
        return StreamingPrediction(value=resolved.prediction, valid=True)

    def update(
        self,
        state: Any,
        observation: tuple[float, ...],
        target: float,
    ) -> StreamingUpdate:
        del observation, target
        return StreamingUpdate(state=state, applied=True)

    def state_to_config(self, state: Any) -> object:
        resolved = cast(FrozenScalarState, state)
        return {"prediction": resolved.prediction}

    def state_from_config(self, payload: object) -> FrozenScalarState:
        if not isinstance(payload, Mapping) or set(payload) != {"prediction"}:
            raise ValueError("frozen baseline state keys are invalid")
        return FrozenScalarState(prediction=_finite_float(payload["prediction"], name="prediction"))

    def resource_usage(self, state: Any) -> LearnerResourceUsage:
        self.state_to_config(state)
        return LearnerResourceUsage(
            persistent_state_bytes=8,
            parameter_count=0,
            host_resident_bytes=8,
            measurement_method="exact logical scalar accounting",
        )

    def diagnostics(self, state: Any) -> Mapping[str, DiagnosticScalar]:
        resolved = cast(FrozenScalarState, state)
        return {"prediction": resolved.prediction, "learning_enabled": False}

    def plasticity_diagnostics(
        self,
        state: Any,
    ) -> Mapping[str, DiagnosticScalar] | None:
        del state
        return None


@dataclasses.dataclass(frozen=True)
class RunningMeanScalarState:
    mean: float
    count: int


class RunningMeanScalarBaseline:
    """Causal running-target-mean baseline with fixed scalar state."""

    def __init__(self, name: str = "running_mean", initial_prediction: float = 0.0):
        if not isinstance(name, str) or not name:
            raise ValueError("name must be a non-empty string")
        self._name = name
        self._initial_prediction = _finite_float(
            initial_prediction,
            name="initial_prediction",
        )

    @property
    def name(self) -> str:
        return self._name

    @property
    def max_backward_calls_per_update(self) -> int:
        return 0

    @property
    def plasticity_applicable(self) -> bool:
        return True

    def to_config(self) -> dict[str, object]:
        return {
            "type": "RunningMeanScalarBaseline",
            "schema_version": "alberta.running_mean_scalar_baseline.v1",
            "name": self._name,
            "initial_prediction": self._initial_prediction,
        }

    def init(self) -> RunningMeanScalarState:
        return RunningMeanScalarState(mean=self._initial_prediction, count=0)

    def predict(
        self,
        state: Any,
        observation: tuple[float, ...],
    ) -> StreamingPrediction:
        del observation
        resolved = cast(RunningMeanScalarState, state)
        return StreamingPrediction(value=resolved.mean, valid=True)

    def update(
        self,
        state: Any,
        observation: tuple[float, ...],
        target: float,
    ) -> StreamingUpdate:
        del observation
        resolved = cast(RunningMeanScalarState, state)
        target_value = _finite_float(target, name="target")
        next_count = resolved.count + 1
        next_mean = resolved.mean + (target_value - resolved.mean) / next_count
        return StreamingUpdate(
            state=RunningMeanScalarState(mean=next_mean, count=next_count),
            applied=True,
        )

    def state_to_config(self, state: Any) -> object:
        resolved = cast(RunningMeanScalarState, state)
        _nonnegative_int(resolved.count, name="count")
        _finite_float(resolved.mean, name="mean")
        return {"mean": resolved.mean, "count": resolved.count}

    def state_from_config(self, payload: object) -> RunningMeanScalarState:
        if not isinstance(payload, Mapping) or set(payload) != {"mean", "count"}:
            raise ValueError("running-mean baseline state keys are invalid")
        return RunningMeanScalarState(
            mean=_finite_float(payload["mean"], name="mean"),
            count=_nonnegative_int(payload["count"], name="count"),
        )

    def resource_usage(self, state: Any) -> LearnerResourceUsage:
        self.state_to_config(state)
        return LearnerResourceUsage(
            persistent_state_bytes=16,
            parameter_count=1,
            host_resident_bytes=16,
            measurement_method="exact logical scalar accounting",
        )

    def diagnostics(self, state: Any) -> Mapping[str, DiagnosticScalar]:
        resolved = cast(RunningMeanScalarState, state)
        return {"mean": resolved.mean, "count": resolved.count}

    def plasticity_diagnostics(
        self,
        state: Any,
    ) -> Mapping[str, DiagnosticScalar]:
        resolved = cast(RunningMeanScalarState, state)
        return {"updates": resolved.count, "online_adaptation": True}


@dataclasses.dataclass(frozen=True)
class StreamingConditionRunState:
    """Bounded accumulator for one learner condition."""

    learner_state: Any
    scores: tuple[float | None, ...]
    processed: tuple[bool, ...]
    delayed: tuple[bool, ...]
    evaluator_matrix: tuple[tuple[float, ...], ...]
    forward_latencies_ns: tuple[int, ...]
    update_latencies_ns: tuple[int, ...]
    backward_latencies_ns: tuple[int, ...]
    host_high_water_bytes: int


@dataclasses.dataclass(frozen=True)
class ContinualStreamingRunState:
    """Checkpointable state for candidate followed by baseline conditions."""

    step: int
    conditions: tuple[StreamingConditionRunState, ...]


class ContinualStreamingEvaluator:
    """Execute one candidate and at least two baselines under one budget."""

    def __init__(
        self,
        *,
        run_id: str,
        protocol: ContinualEvaluationProtocol,
        stream: Sequence[StreamingExample],
        probes: Mapping[str, Sequence[StreamingExample]],
        candidate: StreamingLearner,
        baselines: Sequence[StreamingLearner],
        budget: MatchedBudget,
        score: PredictionScore,
        clock_ns: Callable[[], int] | None = None,
        latency_measurement_method: str = "time.perf_counter_ns wall clock",
    ):
        self._run_id = _versioned_identifier(run_id, name="run_id")
        if not isinstance(protocol, ContinualEvaluationProtocol):
            raise TypeError("protocol must be a ContinualEvaluationProtocol")
        self._protocol = dataclasses.replace(
            protocol,
            regime_schedule=tuple(protocol.regime_schedule),
            evaluator_regime_ids=tuple(protocol.evaluator_regime_ids),
            checkpoint_steps=tuple(protocol.checkpoint_steps),
            first_exposure_checkpoint=dict(protocol.first_exposure_checkpoint),
            forward_transfer_reference=dict(protocol.forward_transfer_reference),
            recovery_thresholds=dict(protocol.recovery_thresholds),
            stability_references=dict(protocol.stability_references),
        )
        self._stream = tuple(stream)
        self._probes = {key: tuple(value) for key, value in probes.items()}
        self._candidate = candidate
        self._baselines = tuple(baselines)
        self._learners = (candidate, *self._baselines)
        self._budget = budget
        self._score = score
        if clock_ns is not None and not callable(clock_ns):
            raise TypeError("clock_ns must be callable")
        self._clock_ns = perf_counter_ns if clock_ns is None else clock_ns
        if not isinstance(latency_measurement_method, str) or not latency_measurement_method:
            raise ValueError("latency_measurement_method must be a non-empty string")
        self._latency_measurement_method = latency_measurement_method
        self._validate_static_contract()

    @property
    def condition_names(self) -> tuple[str, ...]:
        return tuple(learner.name for learner in self._learners)

    def _validate_static_contract(self) -> None:
        if len(self._baselines) < 2:
            raise ValueError("at least two simple baselines are required")
        if any(not isinstance(learner, StreamingLearner) for learner in self._learners):
            raise TypeError("candidate and baselines must implement StreamingLearner")
        names = self.condition_names
        if any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(
            names
        ):
            raise ValueError("candidate and baseline names must be non-empty and unique")
        if not isinstance(self._score, PredictionScore):
            raise TypeError("score must implement PredictionScore")
        if any(not isinstance(example, StreamingExample) for example in self._stream):
            raise TypeError("stream must contain only StreamingExample values")
        if len(self._stream) != len(self._protocol.regime_schedule):
            raise ValueError("stream length must match evaluator regime schedule")
        if self._budget.observation_limit != len(self._stream):
            raise ValueError("observation budget must equal stream length")
        if set(self._probes) != set(self._protocol.evaluator_regime_ids):
            raise ValueError("probe keys must exactly match evaluator_regime_ids")
        if any(not examples for examples in self._probes.values()):
            raise ValueError("every evaluator regime requires a non-empty probe set")
        if any(
            not isinstance(example, StreamingExample)
            for examples in self._probes.values()
            for example in examples
        ):
            raise TypeError("probes must contain only StreamingExample values")
        if self._score.higher_is_better != self._protocol.higher_is_better:
            raise ValueError("score direction must match the report protocol")

        probe_calls_per_checkpoint = sum(len(items) for items in self._probes.values())
        maximum_forward_calls = len(self._stream) + (
            len(self._protocol.checkpoint_steps) * probe_calls_per_checkpoint
        )
        if maximum_forward_calls > self._budget.forward_call_limit:
            raise ValueError("held-out probes exceed the per-condition forward-call budget")
        if len(self._stream) > self._budget.update_call_limit:
            raise ValueError("stream exceeds the per-condition update-call budget")
        learner_configs: list[dict[str, object]] = []
        initial_state_payloads: list[object] = []
        for learner in self._learners:
            maximum_backward = _nonnegative_int(
                learner.max_backward_calls_per_update,
                name=f"{learner.name}.max_backward_calls_per_update",
            )
            if len(self._stream) * maximum_backward > self._budget.backward_call_limit:
                raise ValueError(f"{learner.name}: backward-call budget is too small")
            config = _json_clone(learner.to_config())
            if not isinstance(config, dict):
                raise ValueError(f"{learner.name}: learner config must be a JSON object")
            if not isinstance(config.get("type"), str) or not config["type"]:
                raise ValueError(f"{learner.name}: learner type must be a non-empty string")
            _versioned_identifier(
                config.get("schema_version"),
                name=f"{learner.name}.schema_version",
            )
            learner_configs.append(config)
            first_initial = learner.init()
            first_payload = self._learner_state_payload(learner, first_initial)
            second_initial = learner.init()
            second_payload = self._learner_state_payload(learner, second_initial)
            if first_payload != second_payload:
                raise ValueError(f"{learner.name}: init must be deterministic from learner config")
            if _json_clone(learner.to_config()) != config:
                raise ValueError(f"{learner.name}: init mutated learner config")
            self._validate_resource_usage(learner, first_initial)
            initial_state_payloads.append(first_payload)

        score_config = _json_clone(self._score.to_config())
        if not isinstance(score_config, dict):
            raise ValueError("score config must be a JSON object")
        if not isinstance(score_config.get("type"), str) or not score_config["type"]:
            raise ValueError("score type must be a non-empty string")
        _versioned_identifier(
            score_config.get("schema_version"),
            name="score.schema_version",
        )
        self._learner_configs = tuple(learner_configs)
        self._initial_state_payloads = tuple(initial_state_payloads)
        self._score_config = score_config

    def _learner_state_payload(
        self,
        learner: StreamingLearner,
        learner_state: Any,
    ) -> object:
        payload = _json_clone(learner.state_to_config(learner_state))
        repeated = _json_clone(learner.state_to_config(learner_state))
        if repeated != payload:
            raise ValueError(f"{learner.name}: learner state serialization is not idempotent")
        restored = learner.state_from_config(payload)
        round_trip = _json_clone(learner.state_to_config(restored))
        if round_trip != payload:
            raise ValueError(f"{learner.name}: learner state serialization is not canonical")
        return payload

    def _assert_learner_config_stable(
        self,
        learner: StreamingLearner,
        *,
        index: int,
    ) -> None:
        observed = _json_clone(learner.to_config())
        if observed != self._learner_configs[index]:
            raise ValueError(f"{learner.name}: learner config changed during evaluation")

    def _learner_index(self, learner: StreamingLearner) -> int:
        for index, configured in enumerate(self._learners):
            if configured is learner:
                return index
        raise ValueError("learner is not configured for this evaluator")

    def _assert_score_config_stable(self) -> None:
        if _json_clone(self._score.to_config()) != self._score_config:
            raise ValueError("score config changed during evaluation")

    def _validate_resource_usage(
        self,
        learner: StreamingLearner,
        learner_state: Any,
    ) -> LearnerResourceUsage:
        source_payload = self._learner_state_payload(learner, learner_state)
        usage = learner.resource_usage(learner_state)
        if not isinstance(usage, LearnerResourceUsage):
            raise TypeError(f"{learner.name}: resource_usage must return LearnerResourceUsage")
        if self._learner_state_payload(learner, learner_state) != source_payload:
            raise ValueError(f"{learner.name}: resource_usage mutated learner state")
        if usage.persistent_state_bytes > self._budget.persistent_state_bytes_limit:
            raise ValueError(f"{learner.name}: persistent state exceeds matched budget")
        if usage.parameter_count > self._budget.parameter_count_limit:
            raise ValueError(f"{learner.name}: parameter count exceeds matched budget")
        if usage.host_resident_bytes > self._budget.host_high_water_bytes_limit:
            raise ValueError(f"{learner.name}: host-resident state exceeds matched budget")
        return usage

    def init(self) -> ContinualStreamingRunState:
        conditions = []
        for learner, state_payload in zip(
            self._learners,
            self._initial_state_payloads,
            strict=True,
        ):
            learner_state = learner.state_from_config(_json_clone(state_payload))
            if self._learner_state_payload(learner, learner_state) != state_payload:
                raise ValueError(f"{learner.name}: initial state restore is not canonical")
            usage = self._validate_resource_usage(learner, learner_state)
            conditions.append(
                StreamingConditionRunState(
                    learner_state=learner_state,
                    scores=(),
                    processed=(),
                    delayed=(),
                    evaluator_matrix=(),
                    forward_latencies_ns=(),
                    update_latencies_ns=(),
                    backward_latencies_ns=(),
                    host_high_water_bytes=usage.host_resident_bytes,
                )
            )
        state = ContinualStreamingRunState(step=0, conditions=tuple(conditions))
        self._validate_run_state(state)
        return state

    def _time_call(self, callback: Callable[[], Any], *, name: str) -> tuple[Any, int]:
        start = _nonnegative_int(self._clock_ns(), name=f"{name} clock start")
        result = callback()
        stop = _nonnegative_int(self._clock_ns(), name=f"{name} clock stop")
        duration = stop - start
        if duration < 0:
            raise ValueError(f"{name} clock moved backwards")
        return result, duration

    def _timed_prediction(
        self,
        learner: StreamingLearner,
        learner_state: Any,
        observation: tuple[float, ...],
        *,
        name: str,
    ) -> tuple[StreamingPrediction, int, object]:
        learner_index = self._learner_index(learner)
        self._assert_learner_config_stable(learner, index=learner_index)
        source_payload = self._learner_state_payload(learner, learner_state)
        raw_prediction, latency = self._time_call(
            lambda: learner.predict(learner_state, observation),
            name=name,
        )
        if not isinstance(raw_prediction, StreamingPrediction):
            raise TypeError(f"{learner.name}: predict must return StreamingPrediction")
        if self._learner_state_payload(learner, learner_state) != source_payload:
            raise ValueError(f"{learner.name}: predict mutated learner state")
        self._assert_learner_config_stable(learner, index=learner_index)
        return raw_prediction, latency, source_payload

    def _probe(
        self,
        learner: StreamingLearner,
        learner_state: Any,
    ) -> tuple[tuple[float, ...], tuple[int, ...]]:
        row: list[float] = []
        latencies: list[int] = []
        for regime_id in self._protocol.evaluator_regime_ids:
            probe_scores: list[float] = []
            for example in self._probes[regime_id]:
                prediction, latency, _ = self._timed_prediction(
                    learner,
                    learner_state,
                    example.observation,
                    name=f"{learner.name}.probe_predict",
                )
                if not prediction.valid or not math.isfinite(prediction.value):
                    raise ValueError(f"{learner.name}: held-out probe prediction is unavailable")
                self._assert_score_config_stable()
                score = self._score.score(prediction.value, example.target)
                self._assert_score_config_stable()
                probe_scores.append(_finite_float(score, name="probe score"))
                latencies.append(latency)
            row.append(sum(probe_scores) / len(probe_scores))
        return tuple(row), tuple(latencies)

    def _advance_condition(
        self,
        learner: StreamingLearner,
        condition: StreamingConditionRunState,
        example: StreamingExample,
        *,
        checkpoint: bool,
    ) -> StreamingConditionRunState:
        prediction, forward_latency, original_payload = self._timed_prediction(
            learner,
            condition.learner_state,
            example.observation,
            name=f"{learner.name}.predict",
        )
        next_learner_state = condition.learner_state
        processed = False
        delayed = False
        score: float | None = None
        update_latencies = condition.update_latencies_ns
        backward_latencies = condition.backward_latencies_ns

        if prediction.valid and math.isfinite(prediction.value):
            self._assert_score_config_stable()
            candidate_score = _finite_float(
                self._score.score(prediction.value, example.target),
                name="online score",
            )
            self._assert_score_config_stable()
            raw_update, update_latency = self._time_call(
                lambda: learner.update(
                    condition.learner_state,
                    example.observation,
                    example.target,
                ),
                name=f"{learner.name}.update",
            )
            if not isinstance(raw_update, StreamingUpdate):
                raise TypeError(f"{learner.name}: update must return StreamingUpdate")
            update = raw_update
            if self._learner_state_payload(learner, condition.learner_state) != original_payload:
                raise ValueError(f"{learner.name}: update mutated its source learner state")
            if len(update.backward_latencies_ns) > learner.max_backward_calls_per_update:
                raise ValueError(f"{learner.name}: backward-call bound exceeded")
            update_latencies = (*update_latencies, update_latency)
            backward_latencies = (
                *backward_latencies,
                *update.backward_latencies_ns,
            )
            if update.applied:
                next_learner_state = update.state
                processed = True
                score = candidate_score
                deadline_ns = self._protocol.operation_latency_deadline_ms * 1_000_000.0
                delayed = forward_latency > deadline_ns or update_latency > deadline_ns
            else:
                rejected_payload = self._learner_state_payload(learner, update.state)
                if rejected_payload != original_payload:
                    raise ValueError(f"{learner.name}: rejected update must preserve learner state")

        usage = self._validate_resource_usage(learner, next_learner_state)
        evaluator_matrix = condition.evaluator_matrix
        forward_latencies = (*condition.forward_latencies_ns, forward_latency)
        if checkpoint:
            row, probe_latencies = self._probe(learner, next_learner_state)
            evaluator_matrix = (*evaluator_matrix, row)
            forward_latencies = (*forward_latencies, *probe_latencies)
        return StreamingConditionRunState(
            learner_state=next_learner_state,
            scores=(*condition.scores, score),
            processed=(*condition.processed, processed),
            delayed=(*condition.delayed, delayed),
            evaluator_matrix=evaluator_matrix,
            forward_latencies_ns=forward_latencies,
            update_latencies_ns=update_latencies,
            backward_latencies_ns=backward_latencies,
            host_high_water_bytes=max(
                condition.host_high_water_bytes,
                usage.host_resident_bytes,
            ),
        )

    def advance(
        self,
        state: ContinualStreamingRunState,
        *,
        steps: int = 1,
    ) -> ContinualStreamingRunState:
        """Advance a bounded number of scheduled observations."""
        self._validate_run_state(state)
        requested_steps = _positive_int(steps, name="steps")
        if state.step + requested_steps > len(self._stream):
            raise ValueError("advance would exceed the configured observation stream")
        current = state
        checkpoints = set(self._protocol.checkpoint_steps)
        for _ in range(requested_steps):
            example = self._stream[current.step]
            completed_step = current.step + 1
            conditions = tuple(
                self._advance_condition(
                    learner,
                    condition,
                    example,
                    checkpoint=completed_step in checkpoints,
                )
                for learner, condition in zip(
                    self._learners,
                    current.conditions,
                    strict=True,
                )
            )
            current = ContinualStreamingRunState(
                step=completed_step,
                conditions=conditions,
            )
            self._validate_run_state(current)
        return current

    @staticmethod
    def _latency_summary(values_ns: tuple[int, ...]) -> LatencySummary:
        if not values_ns:
            return LatencySummary(None, None, None)
        ordered = sorted(value / 1_000_000.0 for value in values_ns)

        def quantile(fraction: float) -> float:
            index = math.ceil(fraction * len(ordered)) - 1
            return ordered[index]

        return LatencySummary(
            p50_ms=quantile(0.50),
            p95_ms=quantile(0.95),
            p99_ms=quantile(0.99),
        )

    def _condition_input(
        self,
        learner: StreamingLearner,
        condition: StreamingConditionRunState,
    ) -> ConditionEvaluationInput:
        source_payload = self._learner_state_payload(learner, condition.learner_state)
        usage = self._validate_resource_usage(learner, condition.learner_state)
        diagnostics = learner.diagnostics(condition.learner_state)
        _validate_diagnostics(diagnostics, name=f"{learner.name}.diagnostics")
        plasticity = learner.plasticity_diagnostics(condition.learner_state)
        _validate_diagnostics(
            plasticity,
            name=f"{learner.name}.plasticity_diagnostics",
        )
        if learner.plasticity_applicable != (plasticity is not None):
            raise ValueError(f"{learner.name}: plasticity applicability/diagnostics mismatch")
        if self._learner_state_payload(learner, condition.learner_state) != source_payload:
            raise ValueError(f"{learner.name}: report diagnostics mutated learner state")
        learner_index = self._learner_index(learner)
        self._assert_learner_config_stable(learner, index=learner_index)
        component_name = cast(str, self._learner_configs[learner_index]["type"])
        return ConditionEvaluationInput(
            name=learner.name,
            trace=PredictBeforeUpdateTrace(
                scores=condition.scores,
                processed=condition.processed,
            ),
            evaluator_matrix=EvaluatorMatrix(condition.evaluator_matrix),
            budget=self._budget,
            operations=OperationalMeasurements(
                counts=OperationCounts(
                    processed_observations=sum(condition.processed),
                    delayed_observations=sum(condition.delayed),
                    dropped_observations=(len(condition.processed) - sum(condition.processed)),
                    forward_calls=len(condition.forward_latencies_ns),
                    update_calls=len(condition.update_latencies_ns),
                    backward_calls=len(condition.backward_latencies_ns),
                ),
                latency=LatencyMeasurements(
                    forward=self._latency_summary(condition.forward_latencies_ns),
                    update=self._latency_summary(condition.update_latencies_ns),
                    backward=self._latency_summary(condition.backward_latencies_ns),
                    measurement_method=self._latency_measurement_method,
                ),
                memory=MemoryHighWaterMarks(
                    host_high_water_bytes=condition.host_high_water_bytes,
                    accelerator_high_water_bytes=None,
                    persistent_state_bytes=usage.persistent_state_bytes,
                    parameter_count=usage.parameter_count,
                    measurement_method=usage.measurement_method,
                ),
                energy=EnergyMeasurement(None, None, None),
            ),
            safety=SafetyMeasurements(
                checks=0,
                violations=0,
                interventions=0,
                near_misses=0,
                cumulative_cost=0.0,
                cumulative_near_miss_cost=0.0,
                maximum_step_cost=0.0,
                measurement_method=None,
            ),
            applicable_components=((component_name,) if diagnostics is not None else ()),
            component_diagnostics=(
                {component_name: diagnostics} if diagnostics is not None else None
            ),
            plasticity_applicable=learner.plasticity_applicable,
            plasticity_diagnostics=plasticity,
        )

    def build_report(
        self,
        state: ContinualStreamingRunState,
    ) -> dict[str, object]:
        """Build the strict metric core only after the complete bounded stream.

        Use :meth:`build_bound_report` for the portable artifact.  The bound
        form adds the exact evaluator, learner, stream, probe, score, and
        budget identity that the learner-neutral metric schema intentionally
        does not carry.
        """
        self._validate_run_state(state)
        if state.step != len(self._stream):
            raise ValueError("cannot build a report from an incomplete stream")
        conditions = tuple(
            self._condition_input(learner, condition)
            for learner, condition in zip(
                self._learners,
                state.conditions,
                strict=True,
            )
        )
        return build_continual_evaluation_report(
            protocol=self._protocol,
            candidate=conditions[0],
            baselines=conditions[1:],
        )

    def build_bound_report(
        self,
        state: ContinualStreamingRunState,
    ) -> dict[str, object]:
        """Build a self-checking artifact bound to every evaluator input."""

        core_report = self.build_report(state)
        evaluator_config = self.to_config()
        artifact = {
            "schema_version": STREAMING_EVALUATOR_REPORT_SCHEMA,
            "acceptance_status": "not-assessed",
            "evaluator_config": evaluator_config,
            "evaluator_config_sha256": _digest(evaluator_config),
            "core_report": core_report,
            "core_report_sha256": _digest(core_report),
            "accepted_scientific_evidence": False,
        }
        validation = validate_streaming_evaluator_report(artifact)
        if not validation.valid:
            raise ValueError("; ".join(validation.errors))
        return cast(dict[str, object], _json_clone(artifact))

    def save_bound_report(
        self,
        state: ContinualStreamingRunState,
        path: str | Path,
    ) -> None:
        """Atomically persist a complete evaluator-bound report artifact."""

        _atomic_write_json(self.build_bound_report(state), path)

    def load_bound_report(self, path: str | Path) -> dict[str, object]:
        """Load a strict artifact and require this exact evaluator identity."""

        report = load_streaming_evaluator_report(path)
        if report["evaluator_config"] != self.to_config():
            raise ValueError("streaming evaluator report config does not match")
        return report

    def _completed_checkpoint_count(self, step: int) -> int:
        return sum(checkpoint <= step for checkpoint in self._protocol.checkpoint_steps)

    def _validate_run_state(self, state: ContinualStreamingRunState) -> None:
        if not isinstance(state, ContinualStreamingRunState):
            raise TypeError("state must be a ContinualStreamingRunState")
        step = _nonnegative_int(state.step, name="state.step")
        if step > len(self._stream):
            raise ValueError("state.step exceeds the configured stream")
        if len(state.conditions) != len(self._learners):
            raise ValueError("state condition count does not match evaluator")
        completed_checkpoints = self._completed_checkpoint_count(step)
        probe_calls = sum(len(items) for items in self._probes.values())
        expected_forward_calls = step + completed_checkpoints * probe_calls
        for learner, condition in zip(
            self._learners,
            state.conditions,
            strict=True,
        ):
            if (
                len(condition.scores) != step
                or len(condition.processed) != step
                or len(condition.delayed) != step
            ):
                raise ValueError(f"{learner.name}: trace length does not match state.step")
            for index, (score, processed, delayed) in enumerate(
                zip(condition.scores, condition.processed, condition.delayed, strict=True)
            ):
                if not isinstance(processed, bool):
                    raise ValueError(f"{learner.name}: processed[{index}] must be boolean")
                if not isinstance(delayed, bool):
                    raise ValueError(f"{learner.name}: delayed[{index}] must be boolean")
                if delayed and not processed:
                    raise ValueError(f"{learner.name}: delayed observations must be processed")
                if processed:
                    _finite_float(score, name=f"{learner.name}.scores[{index}]")
                elif score is not None:
                    raise ValueError(f"{learner.name}: dropped scores must be None")
            if len(condition.evaluator_matrix) != completed_checkpoints:
                raise ValueError(f"{learner.name}: evaluator row count is inconsistent")
            for row in condition.evaluator_matrix:
                if len(row) != len(self._protocol.evaluator_regime_ids):
                    raise ValueError(f"{learner.name}: evaluator row width is invalid")
                for value in row:
                    _finite_float(value, name=f"{learner.name}.evaluator_matrix")
            if len(condition.forward_latencies_ns) != expected_forward_calls:
                raise ValueError(f"{learner.name}: forward-call count is inconsistent")
            if len(condition.update_latencies_ns) > step:
                raise ValueError(f"{learner.name}: update-call count exceeds observations")
            if len(condition.update_latencies_ns) < sum(condition.processed):
                raise ValueError(
                    f"{learner.name}: update-call count is below processed observations"
                )
            if (
                len(condition.backward_latencies_ns)
                > len(condition.update_latencies_ns) * learner.max_backward_calls_per_update
            ):
                raise ValueError(f"{learner.name}: backward-call count exceeds bound")
            for values, label in (
                (condition.forward_latencies_ns, "forward"),
                (condition.update_latencies_ns, "update"),
                (condition.backward_latencies_ns, "backward"),
            ):
                for index, value in enumerate(values):
                    _nonnegative_int(
                        value,
                        name=f"{learner.name}.{label}_latencies_ns[{index}]",
                    )
            if len(condition.forward_latencies_ns) > self._budget.forward_call_limit:
                raise ValueError(f"{learner.name}: forward-call budget exceeded")
            if len(condition.update_latencies_ns) > self._budget.update_call_limit:
                raise ValueError(f"{learner.name}: update-call budget exceeded")
            if len(condition.backward_latencies_ns) > self._budget.backward_call_limit:
                raise ValueError(f"{learner.name}: backward-call budget exceeded")
            usage = self._validate_resource_usage(learner, condition.learner_state)
            high_water = _positive_int(
                condition.host_high_water_bytes,
                name=f"{learner.name}.host_high_water_bytes",
            )
            if high_water < usage.host_resident_bytes:
                raise ValueError(f"{learner.name}: host high-water mark is inconsistent")
            if high_water > self._budget.host_high_water_bytes_limit:
                raise ValueError(f"{learner.name}: host high-water budget exceeded")
            self._learner_state_payload(learner, condition.learner_state)
            self._assert_learner_config_stable(
                learner,
                index=self._learner_index(learner),
            )

    def to_config(self) -> dict[str, object]:
        """Return the checkpoint-bound evaluator identity and hard bounds."""
        for index, learner in enumerate(self._learners):
            self._assert_learner_config_stable(learner, index=index)
        self._assert_score_config_stable()
        stream_payload = [example.to_config() for example in self._stream]
        probe_payload = {
            regime_id: [example.to_config() for example in self._probes[regime_id]]
            for regime_id in self._protocol.evaluator_regime_ids
        }
        protocol_payload = _json_clone(dataclasses.asdict(self._protocol))
        return {
            "type": "ContinualStreamingEvaluator",
            "schema_version": STREAMING_EVALUATOR_SCHEMA,
            "report_schema_version": REPORT_SCHEMA_VERSION,
            "run_id": self._run_id,
            "protocol": protocol_payload,
            "matched_budget": dataclasses.asdict(self._budget),
            "score": _json_clone(self._score_config),
            "conditions": [
                {
                    "role": "candidate" if index == 0 else "baseline",
                    "name": learner.name,
                    "learner": _json_clone(self._learner_configs[index]),
                }
                for index, learner in enumerate(self._learners)
            ],
            "stream_sha256": _digest(stream_payload),
            "probe_sha256": _digest(probe_payload),
            "stream_length": len(self._stream),
            "probe_examples_per_checkpoint": sum(len(items) for items in self._probes.values()),
            "latency_measurement_method": self._latency_measurement_method,
            "development_only": True,
            "accepted_scientific_evidence": False,
        }

    def _state_payload(self, state: ContinualStreamingRunState) -> dict[str, object]:
        self._validate_run_state(state)
        conditions = []
        for learner, condition in zip(
            self._learners,
            state.conditions,
            strict=True,
        ):
            conditions.append(
                {
                    "name": learner.name,
                    "learner_state": self._learner_state_payload(
                        learner,
                        condition.learner_state,
                    ),
                    "scores": list(condition.scores),
                    "processed": list(condition.processed),
                    "delayed": list(condition.delayed),
                    "evaluator_matrix": [list(row) for row in condition.evaluator_matrix],
                    "forward_latencies_ns": list(condition.forward_latencies_ns),
                    "update_latencies_ns": list(condition.update_latencies_ns),
                    "backward_latencies_ns": list(condition.backward_latencies_ns),
                    "host_high_water_bytes": condition.host_high_water_bytes,
                }
            )
        return {"step": state.step, "conditions": conditions}

    def save_checkpoint(
        self,
        state: ContinualStreamingRunState,
        path: str | Path,
    ) -> None:
        """Save strict JSON state bound to this evaluator and input stream."""
        state_payload = self._state_payload(state)
        config = self.to_config()
        payload = {
            "schema_version": STREAMING_EVALUATOR_CHECKPOINT_SCHEMA,
            "evaluator_config": config,
            "evaluator_config_sha256": _digest(config),
            "state": state_payload,
            "state_sha256": _digest(state_payload),
        }
        _atomic_write_json(payload, path)

    def load_checkpoint(self, path: str | Path) -> ContinualStreamingRunState:
        """Restore strict JSON state and reject config, stream, or trace drift."""
        payload = json.loads(
            Path(path).read_text(encoding="utf-8"),
            parse_constant=_reject_nonstandard_json_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
        if not isinstance(payload, Mapping) or set(payload) != {
            "schema_version",
            "evaluator_config",
            "evaluator_config_sha256",
            "state",
            "state_sha256",
        }:
            raise ValueError("streaming evaluator checkpoint keys are invalid")
        if payload["schema_version"] != STREAMING_EVALUATOR_CHECKPOINT_SCHEMA:
            raise ValueError("streaming evaluator checkpoint schema is invalid")
        config = payload["evaluator_config"]
        if config != self.to_config() or payload["evaluator_config_sha256"] != _digest(config):
            raise ValueError("streaming evaluator checkpoint config does not match")
        raw_state = payload["state"]
        if payload["state_sha256"] != _digest(raw_state):
            raise ValueError("streaming evaluator checkpoint state digest does not match")
        state_mapping = _mapping(raw_state, name="state")
        if set(state_mapping) != {"step", "conditions"}:
            raise ValueError("checkpoint state keys are invalid")
        step = _nonnegative_int(state_mapping["step"], name="state.step")
        raw_conditions = state_mapping["conditions"]
        if not isinstance(raw_conditions, list) or len(raw_conditions) != len(self._learners):
            raise ValueError("checkpoint conditions do not match evaluator")
        conditions = []
        for index, (learner, raw_condition) in enumerate(
            zip(self._learners, raw_conditions, strict=True)
        ):
            location = f"state.conditions[{index}]"
            condition = _mapping(raw_condition, name=location)
            if set(condition) != {
                "name",
                "learner_state",
                "scores",
                "processed",
                "delayed",
                "evaluator_matrix",
                "forward_latencies_ns",
                "update_latencies_ns",
                "backward_latencies_ns",
                "host_high_water_bytes",
            }:
                raise ValueError(f"{location} keys are invalid")
            if condition["name"] != learner.name:
                raise ValueError(f"{location} learner name does not match")
            scores = tuple(
                None if value is None else _finite_float(value, name=f"{location}.scores")
                for value in _list(condition["scores"], name=f"{location}.scores")
            )
            processed_values = _list(
                condition["processed"],
                name=f"{location}.processed",
            )
            if any(not isinstance(value, bool) for value in processed_values):
                raise ValueError(f"{location}.processed values must be boolean")
            delayed_values = _list(
                condition["delayed"],
                name=f"{location}.delayed",
            )
            if any(not isinstance(value, bool) for value in delayed_values):
                raise ValueError(f"{location}.delayed values must be boolean")
            rows = tuple(
                tuple(
                    _finite_float(value, name=f"{location}.evaluator_matrix")
                    for value in _list(row, name=f"{location}.evaluator_matrix")
                )
                for row in _list(
                    condition["evaluator_matrix"],
                    name=f"{location}.evaluator_matrix",
                )
            )

            def latency_values(field: str) -> tuple[int, ...]:
                return tuple(
                    _nonnegative_int(value, name=f"{location}.{field}")
                    for value in _list(condition[field], name=f"{location}.{field}")
                )

            conditions.append(
                StreamingConditionRunState(
                    learner_state=learner.state_from_config(condition["learner_state"]),
                    scores=scores,
                    processed=cast(tuple[bool, ...], tuple(processed_values)),
                    delayed=cast(tuple[bool, ...], tuple(delayed_values)),
                    evaluator_matrix=rows,
                    forward_latencies_ns=latency_values("forward_latencies_ns"),
                    update_latencies_ns=latency_values("update_latencies_ns"),
                    backward_latencies_ns=latency_values("backward_latencies_ns"),
                    host_high_water_bytes=_positive_int(
                        condition["host_high_water_bytes"],
                        name=f"{location}.host_high_water_bytes",
                    ),
                )
            )
        restored = ContinualStreamingRunState(
            step=step,
            conditions=tuple(conditions),
        )
        self._validate_run_state(restored)
        if self._state_payload(restored) != raw_state:
            raise ValueError("streaming evaluator checkpoint state is not canonical")
        return restored


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be an object with string keys")
    return value


def _list(value: object, *, name: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be an array")
    return value


def _reject_nonstandard_json_constant(value: str) -> NoReturn:
    raise ValueError(f"non-standard JSON numeric constant {value!r}")


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _require_sha256(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _validate_bound_evaluator_config(value: object) -> Mapping[str, object]:
    config = _mapping(value, name="evaluator_config")
    if set(config) != {
        "type",
        "schema_version",
        "report_schema_version",
        "run_id",
        "protocol",
        "matched_budget",
        "score",
        "conditions",
        "stream_sha256",
        "probe_sha256",
        "stream_length",
        "probe_examples_per_checkpoint",
        "latency_measurement_method",
        "development_only",
        "accepted_scientific_evidence",
    }:
        raise ValueError("evaluator_config keys are invalid")
    if config["type"] != "ContinualStreamingEvaluator":
        raise ValueError("evaluator_config type is invalid")
    if config["schema_version"] != STREAMING_EVALUATOR_SCHEMA:
        raise ValueError("evaluator_config schema is invalid")
    if config["report_schema_version"] != REPORT_SCHEMA_VERSION:
        raise ValueError("evaluator_config report schema is invalid")
    _versioned_identifier(config["run_id"], name="evaluator_config.run_id")
    if not isinstance(config["protocol"], Mapping):
        raise ValueError("evaluator_config.protocol must be an object")
    if not isinstance(config["matched_budget"], Mapping):
        raise ValueError("evaluator_config.matched_budget must be an object")
    score = _mapping(config["score"], name="evaluator_config.score")
    if not isinstance(score.get("type"), str) or not score["type"]:
        raise ValueError("evaluator_config.score type is invalid")
    _versioned_identifier(
        score.get("schema_version"),
        name="evaluator_config.score.schema_version",
    )
    _require_sha256(config["stream_sha256"], name="evaluator_config.stream_sha256")
    _require_sha256(config["probe_sha256"], name="evaluator_config.probe_sha256")
    _positive_int(config["stream_length"], name="evaluator_config.stream_length")
    _positive_int(
        config["probe_examples_per_checkpoint"],
        name="evaluator_config.probe_examples_per_checkpoint",
    )
    if (
        not isinstance(config["latency_measurement_method"], str)
        or not config["latency_measurement_method"]
    ):
        raise ValueError("evaluator_config latency method is invalid")
    if config["development_only"] is not True:
        raise ValueError("streaming evaluator reports must remain development-only")
    if config["accepted_scientific_evidence"] is not False:
        raise ValueError("streaming evaluator reports are not scientific evidence")

    raw_conditions = _list(config["conditions"], name="evaluator_config.conditions")
    if len(raw_conditions) < 3:
        raise ValueError("evaluator_config requires one candidate and two baselines")
    condition_names: list[str] = []
    for index, raw_condition in enumerate(raw_conditions):
        location = f"evaluator_config.conditions[{index}]"
        condition = _mapping(raw_condition, name=location)
        if set(condition) != {"role", "name", "learner"}:
            raise ValueError(f"{location} keys are invalid")
        expected_role = "candidate" if index == 0 else "baseline"
        if condition["role"] != expected_role:
            raise ValueError(f"{location}.role is invalid")
        name = condition["name"]
        if not isinstance(name, str) or not name:
            raise ValueError(f"{location}.name must be a non-empty string")
        learner = _mapping(condition["learner"], name=f"{location}.learner")
        if not isinstance(learner.get("type"), str) or not learner["type"]:
            raise ValueError(f"{location}.learner type is invalid")
        _versioned_identifier(
            learner.get("schema_version"),
            name=f"{location}.learner.schema_version",
        )
        condition_names.append(name)
    if len(set(condition_names)) != len(condition_names):
        raise ValueError("evaluator_config condition names must be unique")
    return config


def _validate_streaming_evaluator_report_or_raise(
    report: Mapping[str, object],
) -> dict[str, object]:
    if set(report) != {
        "schema_version",
        "acceptance_status",
        "evaluator_config",
        "evaluator_config_sha256",
        "core_report",
        "core_report_sha256",
        "accepted_scientific_evidence",
    }:
        raise ValueError("streaming evaluator report keys are invalid")
    if report["schema_version"] != STREAMING_EVALUATOR_REPORT_SCHEMA:
        raise ValueError("streaming evaluator report schema is invalid")
    if report["acceptance_status"] != "not-assessed":
        raise ValueError("streaming evaluator report must remain not-assessed")
    if report["accepted_scientific_evidence"] is not False:
        raise ValueError("streaming evaluator report is not scientific evidence")

    config = _validate_bound_evaluator_config(report["evaluator_config"])
    config_digest = _require_sha256(
        report["evaluator_config_sha256"],
        name="evaluator_config_sha256",
    )
    if config_digest != _digest(config):
        raise ValueError("streaming evaluator report config digest does not match")

    core = _mapping(report["core_report"], name="core_report")
    core_digest = _require_sha256(
        report["core_report_sha256"],
        name="core_report_sha256",
    )
    if core_digest != _digest(core):
        raise ValueError("streaming evaluator core-report digest does not match")
    core_validation = validate_continual_evaluation_report(core)
    if not core_validation.valid:
        raise ValueError(
            "streaming evaluator core report is invalid: " + "; ".join(core_validation.errors)
        )
    if core.get("schema_version") != config["report_schema_version"]:
        raise ValueError("core report schema does not match evaluator config")
    if core.get("protocol") != config["protocol"]:
        raise ValueError("core report protocol does not match evaluator config")

    comparison = _mapping(core.get("comparison_contract"), name="core comparison")
    if comparison.get("matched_budget") != config["matched_budget"]:
        raise ValueError("core report budget does not match evaluator config")
    configured_conditions = _list(
        config["conditions"],
        name="evaluator_config.conditions",
    )
    configured_names = [
        cast(str, _mapping(item, name="configured condition")["name"])
        for item in configured_conditions
    ]
    if (
        comparison.get("candidate") != configured_names[0]
        or comparison.get("baselines") != configured_names[1:]
    ):
        raise ValueError("core comparison names do not match evaluator config")

    core_conditions = _list(core.get("conditions"), name="core_report.conditions")
    if len(core_conditions) != len(configured_names):
        raise ValueError("core condition count does not match evaluator config")
    stream_length = cast(int, config["stream_length"])
    latency_method = config["latency_measurement_method"]
    for index, raw_condition in enumerate(core_conditions):
        location = f"core_report.conditions[{index}]"
        condition = _mapping(raw_condition, name=location)
        if condition.get("name") != configured_names[index]:
            raise ValueError(f"{location}.name does not match evaluator config")
        expected_role = "candidate" if index == 0 else "baseline"
        if condition.get("role") != expected_role:
            raise ValueError(f"{location}.role does not match evaluator config")
        evidence = _mapping(condition.get("evidence"), name=f"{location}.evidence")
        trace = _mapping(
            evidence.get("predict_before_update_trace"),
            name=f"{location}.evidence.predict_before_update_trace",
        )
        if len(_list(trace.get("scores"), name=f"{location}.scores")) != stream_length:
            raise ValueError(f"{location} trace length does not match evaluator config")
        operations = _mapping(condition.get("operations"), name=f"{location}.operations")
        latency = _mapping(operations.get("latency"), name=f"{location}.latency")
        if latency.get("measurement_method") != latency_method:
            raise ValueError(f"{location} latency method does not match evaluator config")
    return cast(dict[str, object], _json_clone(report))


def validate_streaming_evaluator_report(
    report: Mapping[str, object],
) -> StreamingEvaluatorReportValidation:
    """Validate hashes, metric reconstruction, and cross-object identity."""

    try:
        normalized = _json_clone(report)
        if not isinstance(normalized, dict):
            raise ValueError("streaming evaluator report must be a JSON object")
        _validate_streaming_evaluator_report_or_raise(normalized)
    except (KeyError, TypeError, ValueError) as error:
        return StreamingEvaluatorReportValidation(valid=False, errors=(str(error),))
    return StreamingEvaluatorReportValidation(valid=True, errors=())


def streaming_evaluator_report_json(report: Mapping[str, object]) -> str:
    """Serialize only a strict evaluator-bound report."""

    validation = validate_streaming_evaluator_report(report)
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


def load_streaming_evaluator_report(path: str | Path) -> dict[str, object]:
    """Load standard JSON and reject modified or internally unbound reports."""

    parsed = json.loads(
        Path(path).read_text(encoding="utf-8"),
        parse_constant=_reject_nonstandard_json_constant,
        object_pairs_hook=_reject_duplicate_keys,
    )
    if not isinstance(parsed, dict):
        raise ValueError("streaming evaluator report root must be an object")
    validation = validate_streaming_evaluator_report(parsed)
    if not validation.valid:
        raise ValueError("; ".join(validation.errors))
    return cast(dict[str, object], _json_clone(parsed))


__all__ = [
    "STREAMING_EVALUATOR_SCHEMA",
    "STREAMING_EVALUATOR_CHECKPOINT_SCHEMA",
    "STREAMING_EVALUATOR_REPORT_SCHEMA",
    "BoundedAbsoluteScore",
    "ContinualStreamingEvaluator",
    "ContinualStreamingRunState",
    "FrozenScalarBaseline",
    "FrozenScalarState",
    "LearnerResourceUsage",
    "PredictionScore",
    "RunningMeanScalarBaseline",
    "RunningMeanScalarState",
    "StreamingConditionRunState",
    "StreamingEvaluatorReportValidation",
    "StreamingExample",
    "StreamingLearner",
    "StreamingPrediction",
    "StreamingUpdate",
    "load_streaming_evaluator_report",
    "streaming_evaluator_report_json",
    "validate_streaming_evaluator_report",
]
