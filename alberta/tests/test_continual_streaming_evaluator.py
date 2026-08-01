"""Contracts for the bounded reusable continual streaming evaluator."""

from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, cast

import pytest

import alberta_framework.evaluation.continual_streaming_evaluator as streaming_module
from alberta_framework.evaluation.continual_evaluation_report import (
    SCHEMA_VERSION,
    ContinualEvaluationProtocol,
    MatchedBudget,
    validate_continual_evaluation_report,
)
from alberta_framework.evaluation.continual_streaming_evaluator import (
    STREAMING_EVALUATOR_CHECKPOINT_SCHEMA,
    STREAMING_EVALUATOR_REPORT_SCHEMA,
    BoundedAbsoluteScore,
    ContinualStreamingEvaluator,
    FrozenScalarBaseline,
    LearnerResourceUsage,
    RunningMeanScalarBaseline,
    StreamingExample,
    StreamingLearner,
    StreamingPrediction,
    StreamingUpdate,
    load_streaming_evaluator_report,
    streaming_evaluator_report_json,
    validate_streaming_evaluator_report,
)


class ConstantDurationClock:
    """Every measured operation takes exactly one microsecond."""

    def __init__(self) -> None:
        self._now = 0

    def __call__(self) -> int:
        value = self._now
        self._now += 1_000
        return value


@dataclass(frozen=True)
class LastTargetState:
    prediction: float
    update_count: int


class LastTargetLearner:
    """Deliberately forgetting learner used to pin evaluation semantics."""

    def __init__(
        self,
        name: str = "last_target",
        *,
        reject_observation: float | None = None,
    ) -> None:
        self._name = name
        self._reject_observation = reject_observation

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
            "type": "LastTargetLearner",
            "schema_version": "tests.last_target_learner.v1",
            "name": self._name,
            "reject_observation": self._reject_observation,
        }

    def init(self) -> LastTargetState:
        return LastTargetState(prediction=0.0, update_count=0)

    def predict(
        self,
        state: Any,
        observation: tuple[float, ...],
    ) -> StreamingPrediction:
        del observation
        resolved = cast(LastTargetState, state)
        return StreamingPrediction(value=resolved.prediction, valid=True)

    def update(
        self,
        state: Any,
        observation: tuple[float, ...],
        target: float,
    ) -> StreamingUpdate:
        resolved = cast(LastTargetState, state)
        if self._reject_observation is not None and observation[0] == self._reject_observation:
            return StreamingUpdate(state=state, applied=False)
        return StreamingUpdate(
            state=LastTargetState(
                prediction=target,
                update_count=resolved.update_count + 1,
            ),
            applied=True,
        )

    def state_to_config(self, state: Any) -> object:
        resolved = cast(LastTargetState, state)
        return {
            "prediction": resolved.prediction,
            "update_count": resolved.update_count,
        }

    def state_from_config(self, payload: object) -> LastTargetState:
        if not isinstance(payload, Mapping) or set(payload) != {
            "prediction",
            "update_count",
        }:
            raise ValueError("last-target state is invalid")
        prediction = payload["prediction"]
        update_count = payload["update_count"]
        if isinstance(prediction, bool) or not isinstance(prediction, int | float):
            raise ValueError("prediction must be numeric")
        if isinstance(update_count, bool) or not isinstance(update_count, int) or update_count < 0:
            raise ValueError("update_count must be non-negative")
        return LastTargetState(float(prediction), update_count)

    def resource_usage(self, state: Any) -> LearnerResourceUsage:
        self.state_to_config(state)
        return LearnerResourceUsage(
            persistent_state_bytes=16,
            parameter_count=1,
            host_resident_bytes=16,
            measurement_method="test exact logical scalar accounting",
        )

    def diagnostics(self, state: Any) -> Mapping[str, bool | int | float | str]:
        resolved = cast(LastTargetState, state)
        return {
            "prediction": resolved.prediction,
            "update_count": resolved.update_count,
        }

    def plasticity_diagnostics(
        self,
        state: Any,
    ) -> Mapping[str, bool | int | float | str]:
        resolved = cast(LastTargetState, state)
        return {
            "online_adaptation": True,
            "updates": resolved.update_count,
        }


class ProbeMutatingLearner(LastTargetLearner):
    """Adversarial learner whose nominally read-only held-out probe mutates state."""

    def predict(
        self,
        state: Any,
        observation: tuple[float, ...],
    ) -> StreamingPrediction:
        resolved = cast(LastTargetState, state)
        if abs(observation[0]) >= 100.0:
            object.__setattr__(resolved, "update_count", resolved.update_count + 1)
        return super().predict(state, observation)


class SourceMutatingUpdateLearner(LastTargetLearner):
    """Adversarial learner that mutates its source before returning a new state."""

    def update(
        self,
        state: Any,
        observation: tuple[float, ...],
        target: float,
    ) -> StreamingUpdate:
        resolved = cast(LastTargetState, state)
        object.__setattr__(resolved, "update_count", resolved.update_count + 1)
        return StreamingUpdate(
            state=LastTargetState(prediction=target, update_count=resolved.update_count),
            applied=True,
        )


class NondeterministicInitLearner(LastTargetLearner):
    def __init__(self) -> None:
        super().__init__(name="nondeterministic_init")
        self._init_calls = 0

    def init(self) -> LastTargetState:
        self._init_calls += 1
        return LastTargetState(prediction=float(self._init_calls), update_count=0)


class NoncanonicalStateLearner(LastTargetLearner):
    def __init__(self) -> None:
        super().__init__(name="noncanonical_state")

    def init(self) -> LastTargetState:
        return LastTargetState(prediction=0.0, update_count=1)

    def state_from_config(self, payload: object) -> LastTargetState:
        restored = super().state_from_config(payload)
        return replace(restored, update_count=0)


class DiagnosticsMutatingLearner(LastTargetLearner):
    def __init__(self) -> None:
        super().__init__(name="diagnostics_mutator")

    def diagnostics(self, state: Any) -> Mapping[str, bool | int | float | str]:
        resolved = cast(LastTargetState, state)
        object.__setattr__(resolved, "update_count", resolved.update_count + 1)
        return super().diagnostics(state)


def _protocol() -> ContinualEvaluationProtocol:
    return ContinualEvaluationProtocol(
        protocol_id="streaming-aba.v1",
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
        operation_latency_deadline_ms=0.001,
    )


def _budget() -> MatchedBudget:
    return MatchedBudget(
        observation_limit=9,
        forward_call_limit=15,
        update_call_limit=9,
        backward_call_limit=0,
        parameter_count_limit=2,
        persistent_state_bytes_limit=32,
        host_high_water_bytes_limit=32,
    )


def _stream() -> tuple[StreamingExample, ...]:
    targets = (1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0)
    return tuple(
        StreamingExample(observation=(float(index),), target=target)
        for index, target in enumerate(targets)
    )


def _evaluator(
    *,
    candidate: StreamingLearner | None = None,
    clock: ConstantDurationClock | None = None,
    protocol: ContinualEvaluationProtocol | None = None,
    budget: MatchedBudget | None = None,
) -> ContinualStreamingEvaluator:
    return ContinualStreamingEvaluator(
        run_id="tests.streaming-aba.v1",
        protocol=_protocol() if protocol is None else protocol,
        stream=_stream(),
        probes={
            "A": (StreamingExample(observation=(100.0,), target=1.0),),
            "B": (StreamingExample(observation=(-100.0,), target=-1.0),),
        },
        candidate=LastTargetLearner() if candidate is None else candidate,
        baselines=(
            FrozenScalarBaseline(name="frozen_zero"),
            RunningMeanScalarBaseline(name="running_mean"),
        ),
        budget=_budget() if budget is None else budget,
        score=BoundedAbsoluteScore(scale=2.0),
        clock_ns=ConstantDurationClock() if clock is None else clock,
        latency_measurement_method="deterministic test clock: 1000 ns per call",
    )


def _condition(report: dict[str, object], name: str) -> dict[str, object]:
    conditions = report["conditions"]
    assert isinstance(conditions, list)
    return next(item for item in conditions if isinstance(item, dict) and item["name"] == name)


@pytest.mark.unit
def test_learner_surface_and_stream_item_exclude_task_and_regime_identity() -> None:
    assert {field.name for field in fields(StreamingExample)} == {
        "observation",
        "target",
    }
    for method_name in ("predict", "update"):
        parameters = inspect.signature(getattr(StreamingLearner, method_name)).parameters
        assert all("task" not in name and "regime" not in name for name in parameters)
    learner = LastTargetLearner()
    assert "task" not in json.dumps(learner.to_config()).lower()
    assert "regime" not in json.dumps(learner.to_config()).lower()


@pytest.mark.integration
def test_streaming_executor_pins_preupdate_trace_matrix_metrics_and_resources() -> None:
    evaluator = _evaluator()
    state = evaluator.advance(evaluator.init(), steps=9)
    report = evaluator.build_report(state)
    assert report["schema_version"] == SCHEMA_VERSION
    assert report["acceptance_status"] == "not-assessed"
    assert validate_continual_evaluation_report(report).valid

    candidate = _condition(report, "last_target")
    evidence = cast(dict[str, object], candidate["evidence"])
    trace = cast(dict[str, object], evidence["predict_before_update_trace"])
    assert trace["scores"] == [0.5, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]
    assert trace["processed"] == [True] * 9
    matrix = cast(dict[str, object], evidence["per_regime_evaluator"])
    assert matrix["values"] == [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]

    metrics = cast(dict[str, object], candidate["metrics"])
    assert metrics["prequential_score"] == pytest.approx(6.5 / 9.0)
    assert metrics["lifetime_score"] == pytest.approx(6.5 / 9.0)
    adaptation = cast(dict[str, object], metrics["adaptation_auc"])
    assert adaptation["mean_normalized_auc"] == pytest.approx(0.75)
    recovery = cast(dict[str, object], metrics["recovery"])
    assert recovery["recovery_rate"] == 1.0
    assert recovery["mean_recovery_steps"] == 3.0
    forgetting = cast(dict[str, object], metrics["forgetting"])
    backward = cast(dict[str, object], metrics["backward_transfer"])
    forward = cast(dict[str, object], metrics["forward_transfer"])
    stability = cast(dict[str, object], metrics["stability"])
    worst = cast(dict[str, object], metrics["worst_window"])
    assert forgetting["per_regime"] == {"A": 0.0, "B": 1.0}
    assert backward["per_regime"] == {"A": 0.0, "B": -1.0}
    assert forward["per_regime"] == {"A": None, "B": -0.5}
    assert stability["mean_gap"] == pytest.approx(0.8)
    assert stability["maximum_gap"] == 0.8
    assert worst == {
        "window_size": 2,
        "score": 0.5,
        "start_step": 2,
        "end_step_exclusive": 4,
    }

    applicability = cast(dict[str, object], metrics["metric_applicability"])
    fwt = cast(dict[str, object], applicability["forward_transfer"])
    assert fwt["applicable"] is True
    assert fwt["available_regimes"] == ["B"]
    assert fwt["unavailable_regimes"] == {"A": "no pre-exposure checkpoint exists for this regime"}

    operations = cast(dict[str, object], candidate["operations"])
    counts = cast(dict[str, object], operations["counts"])
    assert counts == {
        "processed_observations": 9,
        "delayed_observations": 0,
        "dropped_observations": 0,
        "forward_calls": 15,
        "update_calls": 9,
        "backward_calls": 0,
    }
    latency = cast(dict[str, object], operations["latency"])
    assert latency["forward"] == {
        "p50_ms": 0.001,
        "p95_ms": 0.001,
        "p99_ms": 0.001,
    }
    assert latency["update"] == latency["forward"]
    assert latency["backward"] == {
        "p50_ms": None,
        "p95_ms": None,
        "p99_ms": None,
    }
    assert latency["measurement_method"] == ("deterministic test clock: 1000 ns per call")
    assert operations["memory"] == {
        "host_high_water_bytes": 16,
        "accelerator_high_water_bytes": None,
        "persistent_state_bytes": 16,
        "parameter_count": 1,
        "measurement_method": "test exact logical scalar accounting",
    }
    assert operations["energy"] == {
        "value": None,
        "unit": None,
        "measurement_method": None,
    }

    comparison = cast(dict[str, object], report["comparison_contract"])
    assert comparison["baselines"] == ["frozen_zero", "running_mean"]
    assert comparison["budget_match_verified"] is True


@pytest.mark.integration
def test_rejected_update_is_counted_as_dropped_without_leaking_score() -> None:
    evaluator = _evaluator(candidate=LastTargetLearner(reject_observation=4.0))
    state = evaluator.advance(evaluator.init(), steps=9)
    report = evaluator.build_report(state)
    candidate = _condition(report, "last_target")
    evidence = cast(dict[str, object], candidate["evidence"])
    trace = cast(dict[str, object], evidence["predict_before_update_trace"])
    assert cast(list[object], trace["scores"])[4] is None
    assert cast(list[object], trace["processed"])[4] is False
    operations = cast(dict[str, object], candidate["operations"])
    counts = cast(dict[str, object], operations["counts"])
    assert counts["processed_observations"] == 8
    assert counts["dropped_observations"] == 1
    assert counts["update_calls"] == 9
    assert validate_continual_evaluation_report(report).valid


@pytest.mark.integration
def test_checkpoint_resume_is_deterministic_and_bound_to_evaluator(
    tmp_path: Path,
) -> None:
    uninterrupted_evaluator = _evaluator()
    uninterrupted = uninterrupted_evaluator.advance(
        uninterrupted_evaluator.init(),
        steps=9,
    )
    uninterrupted_report = uninterrupted_evaluator.build_report(uninterrupted)

    first_evaluator = _evaluator()
    partial = first_evaluator.advance(first_evaluator.init(), steps=4)
    checkpoint = tmp_path / "streaming-evaluator.json"
    first_evaluator.save_checkpoint(partial, checkpoint)
    raw = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert raw["schema_version"] == STREAMING_EVALUATOR_CHECKPOINT_SCHEMA
    assert raw["state"]["step"] == 4

    tampered = tmp_path / "tampered.json"
    raw["state"]["conditions"][0]["scores"][0] = 1.0
    tampered.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="state digest does not match"):
        _evaluator().load_checkpoint(tampered)

    resumed_evaluator = _evaluator()
    restored = resumed_evaluator.load_checkpoint(checkpoint)
    resumed = resumed_evaluator.advance(restored, steps=5)
    resumed_report = resumed_evaluator.build_report(resumed)
    assert resumed == uninterrupted
    assert resumed_report == uninterrupted_report

    incompatible = ContinualStreamingEvaluator(
        run_id="tests.streaming-aba.v2",
        protocol=_protocol(),
        stream=_stream(),
        probes={
            "A": (StreamingExample((100.0,), 1.0),),
            "B": (StreamingExample((-100.0,), -1.0),),
        },
        candidate=LastTargetLearner(),
        baselines=(FrozenScalarBaseline(), RunningMeanScalarBaseline()),
        budget=_budget(),
        score=BoundedAbsoluteScore(scale=2.0),
        clock_ns=ConstantDurationClock(),
        latency_measurement_method="deterministic test clock: 1000 ns per call",
    )
    with pytest.raises(ValueError, match="config does not match"):
        incompatible.load_checkpoint(checkpoint)


@pytest.mark.integration
def test_bound_report_pins_stream_probes_learners_and_metric_core(
    tmp_path: Path,
) -> None:
    evaluator = _evaluator()
    completed = evaluator.advance(evaluator.init(), steps=9)
    artifact = evaluator.build_bound_report(completed)

    assert artifact["schema_version"] == STREAMING_EVALUATOR_REPORT_SCHEMA
    assert artifact["acceptance_status"] == "not-assessed"
    assert artifact["accepted_scientific_evidence"] is False
    assert validate_streaming_evaluator_report(artifact).valid
    assert json.loads(streaming_evaluator_report_json(artifact)) == artifact

    config = cast(dict[str, object], artifact["evaluator_config"])
    assert config["stream_sha256"] == evaluator.to_config()["stream_sha256"]
    assert config["probe_sha256"] == evaluator.to_config()["probe_sha256"]
    configured_conditions = cast(list[dict[str, object]], config["conditions"])
    assert [condition["name"] for condition in configured_conditions] == [
        "last_target",
        "frozen_zero",
        "running_mean",
    ]
    core = cast(dict[str, object], artifact["core_report"])
    assert validate_continual_evaluation_report(core).valid

    destination = tmp_path / "bound-streaming-report.json"
    evaluator.save_bound_report(completed, destination)
    assert load_streaming_evaluator_report(destination) == artifact
    assert evaluator.load_bound_report(destination) == artifact


@pytest.mark.unit
def test_bound_report_rejects_digest_and_recomputed_cross_identity_tampering() -> None:
    evaluator = _evaluator()
    completed = evaluator.advance(evaluator.init(), steps=9)
    artifact = evaluator.build_bound_report(completed)

    digest_tampered = cast(
        dict[str, object],
        json.loads(json.dumps(artifact)),
    )
    config = cast(dict[str, object], digest_tampered["evaluator_config"])
    config["stream_sha256"] = "0" * 64
    validation = validate_streaming_evaluator_report(digest_tampered)
    assert not validation.valid
    assert "config digest does not match" in validation.errors[0]

    cross_tampered = cast(
        dict[str, object],
        json.loads(json.dumps(artifact)),
    )
    cross_config = cast(dict[str, object], cross_tampered["evaluator_config"])
    conditions = cast(list[dict[str, object]], cross_config["conditions"])
    conditions[0]["name"] = "different_candidate"
    cross_tampered["evaluator_config_sha256"] = hashlib.sha256(
        json.dumps(
            cross_config,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    validation = validate_streaming_evaluator_report(cross_tampered)
    assert not validation.valid
    assert "comparison names do not match" in validation.errors[0]


@pytest.mark.unit
def test_atomic_checkpoint_failure_preserves_previous_file_and_cleans_temp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator = _evaluator()
    partial = evaluator.advance(evaluator.init(), steps=4)
    destination = tmp_path / "streaming-checkpoint.json"
    destination.write_text("previous checkpoint\n", encoding="utf-8")

    def fail_replace(source: object, target: object) -> None:
        del source, target
        raise OSError("injected atomic replace failure")

    monkeypatch.setattr(streaming_module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected atomic replace failure"):
        evaluator.save_checkpoint(partial, destination)

    assert destination.read_text(encoding="utf-8") == "previous checkpoint\n"
    assert list(tmp_path.glob(".streaming-checkpoint.json.*.tmp")) == []


@pytest.mark.unit
def test_executor_rejects_unbounded_baseline_and_forward_probe_configs() -> None:
    with pytest.raises(ValueError, match="at least two"):
        ContinualStreamingEvaluator(
            run_id="tests.invalid.v1",
            protocol=_protocol(),
            stream=_stream(),
            probes={
                "A": (StreamingExample((0.0,), 1.0),),
                "B": (StreamingExample((0.0,), -1.0),),
            },
            candidate=LastTargetLearner(),
            baselines=(FrozenScalarBaseline(),),
            budget=_budget(),
            score=BoundedAbsoluteScore(scale=2.0),
        )

    with pytest.raises(ValueError, match="host-resident state exceeds"):
        _evaluator(
            budget=replace(
                _budget(),
                host_high_water_bytes_limit=15,
            )
        )

    insufficient = replace(_budget(), forward_call_limit=14)
    with pytest.raises(ValueError, match="forward-call budget"):
        ContinualStreamingEvaluator(
            run_id="tests.invalid.v1",
            protocol=_protocol(),
            stream=_stream(),
            probes={
                "A": (StreamingExample((0.0,), 1.0),),
                "B": (StreamingExample((0.0,), -1.0),),
            },
            candidate=LastTargetLearner(),
            baselines=(FrozenScalarBaseline(), RunningMeanScalarBaseline()),
            budget=insufficient,
            score=BoundedAbsoluteScore(scale=2.0),
        )


@pytest.mark.unit
def test_executor_rejects_mutating_predict_update_and_noncanonical_initial_state() -> None:
    probe_mutator = _evaluator(candidate=ProbeMutatingLearner())
    with pytest.raises(ValueError, match="predict mutated learner state"):
        probe_mutator.advance(probe_mutator.init(), steps=3)

    source_mutator = _evaluator(candidate=SourceMutatingUpdateLearner())
    with pytest.raises(ValueError, match="update mutated its source learner state"):
        source_mutator.advance(source_mutator.init())

    with pytest.raises(ValueError, match="init must be deterministic"):
        _evaluator(candidate=NondeterministicInitLearner())

    with pytest.raises(ValueError, match="state serialization is not canonical"):
        _evaluator(candidate=NoncanonicalStateLearner())

    diagnostics_mutator = _evaluator(candidate=DiagnosticsMutatingLearner())
    completed = diagnostics_mutator.advance(diagnostics_mutator.init(), steps=9)
    with pytest.raises(ValueError, match="report diagnostics mutated learner state"):
        diagnostics_mutator.build_report(completed)


@pytest.mark.unit
def test_evaluator_snapshots_mutable_protocol_mappings() -> None:
    protocol = _protocol()
    evaluator = _evaluator(protocol=protocol)
    thresholds = cast(dict[str, float], protocol.recovery_thresholds)
    thresholds["A"] = 123.0
    config = evaluator.to_config()
    frozen_protocol = cast(dict[str, object], config["protocol"])
    assert frozen_protocol["recovery_thresholds"] == {"A": 0.8, "B": 0.8}


@pytest.mark.integration
def test_executor_counts_processed_observations_that_miss_frozen_deadline() -> None:
    evaluator = _evaluator(
        protocol=replace(
            _protocol(),
            operation_latency_deadline_ms=0.0005,
        )
    )
    state = evaluator.advance(evaluator.init(), steps=9)
    report = evaluator.build_report(state)
    for name in evaluator.condition_names:
        condition = _condition(report, name)
        operations = cast(dict[str, object], condition["operations"])
        counts = cast(dict[str, object], operations["counts"])
        assert counts["processed_observations"] == 9
        assert counts["delayed_observations"] == 9
    assert validate_continual_evaluation_report(report).valid
