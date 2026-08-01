"""Adversarial contracts for bounded WP4.5 component-retention probes."""

from __future__ import annotations

import dataclasses
import inspect
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import pytest

from alberta_framework.evaluation.component_retention_probes import (
    COMPONENT_NAMES,
    COMPONENT_RETENTION_CHECKPOINT_SCHEMA,
    COMPONENT_RETENTION_REPORT_SCHEMA,
    ComponentProbeInput,
    ComponentProbeLearner,
    ComponentProbeTargets,
    ComponentRetentionBudget,
    ComponentRetentionEvaluator,
    ComponentRetentionProtocol,
    HeldOutComponentProbe,
    LearnerComponentPredictions,
    ProbeValue,
    validate_component_retention_report,
)


@dataclasses.dataclass(frozen=True)
class SampleLearnerState:
    step: int
    drift: float
    padding: str = ""


class SampleComponentLearner:
    def __init__(
        self,
        *,
        name: str = "test_component_learner",
        missing_critic_at_step: int | None = None,
        malformed_representation_at_step: int | None = None,
    ) -> None:
        self._name = name
        self._missing_critic_at_step = missing_critic_at_step
        self._malformed_representation_at_step = malformed_representation_at_step
        self.prediction_calls = 0

    @property
    def name(self) -> str:
        return self._name

    def to_config(self) -> dict[str, object]:
        return {
            "type": "SampleComponentLearner",
            "schema_version": "tests.component_learner.v1",
            "name": self._name,
            "missing_critic_at_step": self._missing_critic_at_step,
            "malformed_representation_at_step": (self._malformed_representation_at_step),
        }

    def training_step(self, state: Any) -> int:
        return cast(SampleLearnerState, state).step

    def state_to_config(self, state: Any) -> object:
        resolved = cast(SampleLearnerState, state)
        return {
            "step": resolved.step,
            "drift": resolved.drift,
            "padding": resolved.padding,
        }

    def state_from_config(self, payload: object) -> SampleLearnerState:
        if not isinstance(payload, Mapping) or set(payload) != {"step", "drift", "padding"}:
            raise ValueError("test learner state is invalid")
        step = payload["step"]
        drift = payload["drift"]
        padding = payload["padding"]
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError("step is invalid")
        if isinstance(drift, bool) or not isinstance(drift, int | float):
            raise ValueError("drift is invalid")
        if not isinstance(padding, str):
            raise ValueError("padding is invalid")
        return SampleLearnerState(step=step, drift=float(drift), padding=padding)

    def predict_components(
        self,
        state: Any,
        probe_input: ComponentProbeInput,
    ) -> LearnerComponentPredictions:
        self.prediction_calls += 1
        resolved = cast(SampleLearnerState, state)
        x, y = probe_input.observation
        drift = resolved.drift
        representation: tuple[float, ...] = (x + drift, y + drift)
        if resolved.step == self._malformed_representation_at_step:
            representation = (*representation, 99.0)
        critic = (
            ProbeValue.missing("critic_warmup")
            if resolved.step == self._missing_critic_at_step
            else ProbeValue.present((x + y + drift,))
        )
        actor_logits = (2.0, 1.0, 0.0) if drift == 0.0 else (0.5, 1.5, 0.0)
        return LearnerComponentPredictions(
            representation=ProbeValue.present(representation),
            dynamics_observation=ProbeValue.present(
                (x + probe_input.action + drift, y - probe_input.action + drift)
            ),
            reward=ProbeValue.present((x - y + drift,)),
            termination_discount=ProbeValue.present((0.75 + drift / 10.0,)),
            critic_value=critic,
            actor_logits=ProbeValue.present(actor_logits),
        )


def _targets(
    x: float,
    y: float,
    action: int,
    *,
    actor_return_available: bool = True,
) -> ComponentProbeTargets:
    return ComponentProbeTargets(
        representation=ProbeValue.present((x, y)),
        dynamics_observation=ProbeValue.present((x + action, y - action)),
        reward=ProbeValue.present((x - y,)),
        termination_discount=ProbeValue.present((0.75,)),
        critic_value=ProbeValue.present((x + y,)),
        actor_margin=ProbeValue.present((0.0,)),
        actor_return=(
            ProbeValue.present((3.0, 1.0, -2.0))
            if actor_return_available
            else ProbeValue.missing("counterfactual_returns_not_collected")
        ),
    )


def _probe(
    x: float,
    y: float,
    action: int,
    *,
    actor_return_available: bool = True,
) -> HeldOutComponentProbe:
    return HeldOutComponentProbe(
        probe_input=ComponentProbeInput(observation=(x, y), action=action),
        targets=_targets(
            x,
            y,
            action,
            actor_return_available=actor_return_available,
        ),
    )


def _protocol() -> ComponentRetentionProtocol:
    return ComponentRetentionProtocol(
        protocol_id="tests.component-retention.v1",
        evaluator_regime_ids=("old-a", "old-b"),
        checkpoint_steps=(2, 4),
    )


def _budget(**overrides: int) -> ComponentRetentionBudget:
    values = {
        "max_probe_calls": 4,
        "max_stored_records": 28,
        "max_learner_snapshot_bytes": 1_024,
        "max_evaluator_state_bytes": 32_000,
    }
    values.update(overrides)
    return ComponentRetentionBudget(**values)


def _evaluator(
    *,
    learner: ComponentProbeLearner | None = None,
    run_id: str = "tests.component-retention-run.v1",
    budget: ComponentRetentionBudget | None = None,
) -> ComponentRetentionEvaluator:
    return ComponentRetentionEvaluator(
        run_id=run_id,
        protocol=_protocol(),
        held_out_probes={
            "old-a": (_probe(1.0, 2.0, 1),),
            "old-b": (_probe(-2.0, 1.0, 0, actor_return_available=False),),
        },
        learner=SampleComponentLearner() if learner is None else learner,
        budget=_budget() if budget is None else budget,
    )


def _summary(
    report: dict[str, object],
    regime: str,
    component: str,
) -> dict[str, object]:
    summaries = cast(list[dict[str, object]], report["retention_summaries"])
    return next(
        summary
        for summary in summaries
        if summary["evaluator_regime_id"] == regime and summary["component"] == component
    )


def _record(
    report: dict[str, object],
    step: int,
    regime: str,
    component: str,
) -> dict[str, object]:
    records = cast(list[dict[str, object]], report["records"])
    return next(
        record
        for record in records
        if record["checkpoint_step"] == step
        and record["evaluator_regime_id"] == regime
        and record["component"] == component
    )


@pytest.mark.unit
def test_probe_surface_keeps_regime_identity_and_targets_outside_learner() -> None:
    assert {field.name for field in dataclasses.fields(ComponentProbeInput)} == {
        "observation",
        "action",
    }
    parameters = inspect.signature(ComponentProbeLearner.predict_components).parameters
    assert set(parameters) == {"self", "state", "probe_input"}
    assert all("regime" not in name and "target" not in name for name in parameters)
    assert "update" not in ComponentProbeLearner.__dict__


@pytest.mark.integration
def test_all_components_are_separate_bounded_and_retention_is_fail_closed() -> None:
    learner = SampleComponentLearner()
    evaluator = _evaluator(learner=learner)
    first = evaluator.probe_before_update(
        evaluator.init(),
        completed_observations=2,
        learner_state=SampleLearnerState(step=2, drift=0.0),
    )
    completed = evaluator.probe_before_update(
        first,
        completed_observations=4,
        learner_state=SampleLearnerState(step=4, drift=0.1),
    )
    report = evaluator.build_report(completed)

    assert report["schema_version"] == COMPONENT_RETENTION_REPORT_SCHEMA
    assert report["status"] == "not-assessed"
    assert report["accepted_scientific_evidence"] is False
    assert validate_component_retention_report(report).valid
    assert learner.prediction_calls == 4
    assert len(cast(list[object], report["records"])) == 28
    assert set(COMPONENT_NAMES) == {
        "representation",
        "dynamics_observation",
        "reward",
        "termination_discount",
        "critic_value",
        "actor_margin",
        "actor_return",
    }

    assert _record(report, 2, "old-a", "representation")["score"] == 0.0
    representation = _summary(report, "old-a", "representation")
    assert representation["first_score"] == 0.0
    assert representation["current_score"] == pytest.approx(-0.01)
    assert representation["peak_to_current_forgetting"] == pytest.approx(0.01)
    assert _summary(report, "old-a", "dynamics_observation")[
        "peak_to_current_forgetting"
    ] == pytest.approx(0.01)
    assert _summary(report, "old-a", "reward")["peak_to_current_forgetting"] == pytest.approx(0.01)
    assert _summary(report, "old-a", "termination_discount")[
        "peak_to_current_forgetting"
    ] == pytest.approx(0.0001)
    assert _summary(report, "old-a", "critic_value")["peak_to_current_forgetting"] == pytest.approx(
        0.01
    )
    assert _summary(report, "old-a", "actor_margin")["peak_to_current_forgetting"] == 2.0
    assert _summary(report, "old-a", "actor_return")["peak_to_current_forgetting"] == 2.0

    unavailable = _summary(report, "old-b", "actor_return")
    assert unavailable == {
        "evaluator_regime_id": "old-b",
        "component": "actor_return",
        "applicable": False,
        "retention_available": False,
        "unavailable_reason": "component target is unavailable for this evaluator regime",
        "first_score": None,
        "current_score": None,
        "peak_score": None,
        "change_from_first": None,
        "peak_to_current_forgetting": None,
    }
    unavailable_record = _record(report, 4, "old-b", "actor_return")
    assert unavailable_record["scored_case_count"] == 0
    assert unavailable_record["unavailable_reason_counts"] == [
        {"reason": "target_unavailable:counterfactual_returns_not_collected", "count": 1}
    ]

    resources = cast(dict[str, object], report["resources"])
    assert resources["probe_calls"] == 4
    assert resources["probe_call_limit"] == 4
    assert resources["stored_records"] == 28
    assert resources["stored_record_limit"] == 28
    assert resources["raw_per_case_traces_stored"] == 0
    assert cast(int, resources["evaluator_state_bytes"]) <= cast(
        int, resources["evaluator_state_byte_limit"]
    )
    assert "multi-seed" in " ".join(cast(list[str], report["known_limitations"]))


@pytest.mark.integration
@pytest.mark.parametrize(
    ("learner", "component", "expected_reason"),
    [
        (
            SampleComponentLearner(missing_critic_at_step=4),
            "critic_value",
            "learner_prediction_unavailable:critic_warmup",
        ),
        (
            SampleComponentLearner(malformed_representation_at_step=4),
            "representation",
            "prediction_shape_mismatch",
        ),
    ],
)
def test_runtime_unavailability_never_averages_a_partial_retention_trace(
    learner: SampleComponentLearner,
    component: str,
    expected_reason: str,
) -> None:
    evaluator = _evaluator(learner=learner)
    first = evaluator.probe_before_update(
        evaluator.init(),
        completed_observations=2,
        learner_state=SampleLearnerState(step=2, drift=0.0),
    )
    completed = evaluator.probe_before_update(
        first,
        completed_observations=4,
        learner_state=SampleLearnerState(step=4, drift=0.1),
    )
    report = evaluator.build_report(completed)
    record = _record(report, 4, "old-a", component)
    assert record["available"] is False
    assert record["score"] is None
    assert record["unavailable_reason_counts"] == [{"reason": expected_reason, "count": 1}]
    summary = _summary(report, "old-a", component)
    assert summary["applicable"] is True
    assert summary["retention_available"] is False
    assert summary["first_score"] is None
    assert validate_component_retention_report(report).valid


class MutatingProbeLearner:
    prediction_calls = 0

    @property
    def name(self) -> str:
        return "mutating_probe"

    def to_config(self) -> dict[str, object]:
        return {
            "type": "MutatingProbeLearner",
            "schema_version": "tests.mutating_probe.v1",
        }

    def training_step(self, state: Any) -> int:
        return cast(dict[str, int], state)["step"]

    def state_to_config(self, state: Any) -> object:
        return dict(cast(dict[str, int], state))

    def state_from_config(self, payload: object) -> dict[str, int]:
        if not isinstance(payload, Mapping) or set(payload) != {"step"}:
            raise ValueError("mutating state is invalid")
        step = payload["step"]
        if isinstance(step, bool) or not isinstance(step, int):
            raise ValueError("step is invalid")
        return {"step": step}

    def predict_components(
        self,
        state: Any,
        probe_input: ComponentProbeInput,
    ) -> LearnerComponentPredictions:
        del probe_input
        resolved = cast(dict[str, int], state)
        resolved["step"] += 1
        self.prediction_calls += 1
        present = ProbeValue.present((0.0,))
        return LearnerComponentPredictions(
            representation=present,
            dynamics_observation=present,
            reward=present,
            termination_discount=present,
            critic_value=present,
            actor_logits=ProbeValue.present((1.0, 0.0)),
        )


@pytest.mark.unit
def test_probe_uses_reconstructed_snapshot_and_rejects_mutation_atomically() -> None:
    learner = MutatingProbeLearner()
    evaluator = _evaluator(learner=learner)
    live_state = {"step": 2}
    initial = evaluator.init()
    with pytest.raises(ValueError, match="mutated the reconstructed learner state"):
        evaluator.probe_before_update(
            initial,
            completed_observations=2,
            learner_state=live_state,
        )
    assert live_state == {"step": 2}
    assert initial == evaluator.init()
    assert learner.prediction_calls == 1


@pytest.mark.unit
def test_checkpoint_and_training_step_order_are_strict_before_any_prediction() -> None:
    learner = SampleComponentLearner()
    evaluator = _evaluator(learner=learner)
    initial = evaluator.init()
    with pytest.raises(ValueError, match="next checkpoint"):
        evaluator.probe_before_update(
            initial,
            completed_observations=4,
            learner_state=SampleLearnerState(step=4, drift=0.0),
        )
    with pytest.raises(ValueError, match="scheduled before-update checkpoint"):
        evaluator.probe_before_update(
            initial,
            completed_observations=2,
            learner_state=SampleLearnerState(step=3, drift=0.0),
        )
    assert learner.prediction_calls == 0
    with pytest.raises(ValueError, match="cannot report"):
        evaluator.build_report(initial)


@pytest.mark.integration
def test_checkpoint_resume_is_deterministic_and_rejects_drift_and_tampering(
    tmp_path: Path,
) -> None:
    uninterrupted_evaluator = _evaluator()
    uninterrupted_first = uninterrupted_evaluator.probe_before_update(
        uninterrupted_evaluator.init(),
        completed_observations=2,
        learner_state=SampleLearnerState(step=2, drift=0.0),
    )
    uninterrupted = uninterrupted_evaluator.probe_before_update(
        uninterrupted_first,
        completed_observations=4,
        learner_state=SampleLearnerState(step=4, drift=0.1),
    )
    uninterrupted_report = uninterrupted_evaluator.build_report(uninterrupted)

    first_evaluator = _evaluator()
    partial = first_evaluator.probe_before_update(
        first_evaluator.init(),
        completed_observations=2,
        learner_state=SampleLearnerState(step=2, drift=0.0),
    )
    checkpoint = tmp_path / "component-retention.json"
    first_evaluator.save_checkpoint(partial, checkpoint)
    raw = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert raw["schema_version"] == COMPONENT_RETENTION_CHECKPOINT_SCHEMA
    assert raw["state"]["next_checkpoint_index"] == 1

    resumed_evaluator = _evaluator()
    restored = resumed_evaluator.load_checkpoint(checkpoint)
    resumed = resumed_evaluator.probe_before_update(
        restored,
        completed_observations=4,
        learner_state=SampleLearnerState(step=4, drift=0.1),
    )
    assert resumed == uninterrupted
    assert resumed_evaluator.build_report(resumed) == uninterrupted_report

    raw["state"]["probe_calls"] = 99
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="state digest does not match"):
        _evaluator().load_checkpoint(tampered)

    with pytest.raises(ValueError, match="config does not match"):
        _evaluator(run_id="tests.component-retention-run.v2").load_checkpoint(checkpoint)

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_version":"x","schema_version":"y"}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON object key"):
        _evaluator().load_checkpoint(duplicate)


@pytest.mark.unit
def test_static_applicability_and_lifetime_resource_bounds_fail_closed() -> None:
    mixed_probes = {
        "old-a": (
            _probe(1.0, 2.0, 1),
            _probe(2.0, 3.0, 0, actor_return_available=False),
        ),
        "old-b": (_probe(-2.0, 1.0, 0, actor_return_available=False),),
    }
    with pytest.raises(ValueError, match="applicability must be uniform"):
        ComponentRetentionEvaluator(
            run_id="tests.invalid.v1",
            protocol=_protocol(),
            held_out_probes=mixed_probes,
            learner=SampleComponentLearner(),
            budget=ComponentRetentionBudget(6, 28, 1_024, 32_000),
        )
    with pytest.raises(ValueError, match="max_probe_calls"):
        _evaluator(budget=_budget(max_probe_calls=3))
    with pytest.raises(ValueError, match="max_stored_records"):
        _evaluator(budget=_budget(max_stored_records=27))

    snapshot_limited = _evaluator(budget=_budget(max_learner_snapshot_bytes=8))
    with pytest.raises(ValueError, match="snapshot exceeds"):
        snapshot_limited.probe_before_update(
            snapshot_limited.init(),
            completed_observations=2,
            learner_state=SampleLearnerState(step=2, drift=0.0, padding="large"),
        )
    state_limited = _evaluator(budget=_budget(max_evaluator_state_bytes=2))
    with pytest.raises(ValueError, match="state exceeds"):
        state_limited.init()


@pytest.mark.unit
def test_value_target_and_report_validation_reject_malformed_inputs() -> None:
    with pytest.raises(ValueError, match="finite"):
        ProbeValue.present((float("nan"),))
    with pytest.raises(ValueError, match="finite"):
        ProbeValue.present((True,))
    with pytest.raises(ValueError, match="reason"):
        ProbeValue(available=False, values=(), unavailable_reason="")
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        dataclasses.replace(
            _targets(1.0, 2.0, 0),
            termination_discount=ProbeValue.present((1.1,)),
        )
    with pytest.raises(ValueError, match="non-negative integer"):
        dataclasses.replace(_targets(1.0, 2.0, 0), actor_margin=ProbeValue.present((0.5,)))

    evaluator = _evaluator()
    first = evaluator.probe_before_update(
        evaluator.init(),
        completed_observations=2,
        learner_state=SampleLearnerState(step=2, drift=0.0),
    )
    completed = evaluator.probe_before_update(
        first,
        completed_observations=4,
        learner_state=SampleLearnerState(step=4, drift=0.1),
    )
    report = evaluator.build_report(completed)
    tampered = json.loads(json.dumps(report))
    tampered["status"] = "accepted"
    validation = validate_component_retention_report(tampered)
    assert not validation.valid
    assert "not-assessed" in validation.errors[0]

    tampered_score = json.loads(json.dumps(report))
    tampered_score["records"][0]["score"] = -99.0
    validation = validate_component_retention_report(tampered_score)
    assert not validation.valid
    assert "reconstruct" in validation.errors[0]


@pytest.mark.unit
def test_discrete_actor_return_tie_break_is_deterministic_first_argmax() -> None:
    class TiedActorLearner(SampleComponentLearner):
        def predict_components(
            self,
            state: Any,
            probe_input: ComponentProbeInput,
        ) -> LearnerComponentPredictions:
            predictions = super().predict_components(state, probe_input)
            return dataclasses.replace(
                predictions,
                actor_logits=ProbeValue.present((1.0, 1.0, 0.0)),
            )

    evaluator = _evaluator(learner=TiedActorLearner())
    first = evaluator.probe_before_update(
        evaluator.init(),
        completed_observations=2,
        learner_state=SampleLearnerState(step=2, drift=0.0),
    )
    completed = evaluator.probe_before_update(
        first,
        completed_observations=4,
        learner_state=SampleLearnerState(step=4, drift=0.0),
    )
    report = evaluator.build_report(completed)
    assert _record(report, 2, "old-a", "actor_return")["score"] == 3.0
    assert _record(report, 4, "old-a", "actor_return")["score"] == 3.0
