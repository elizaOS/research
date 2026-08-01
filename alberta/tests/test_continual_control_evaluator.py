"""Contracts for learner-neutral continuing-control evaluation."""

from __future__ import annotations

import dataclasses
import inspect
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import pytest

import alberta_framework.evaluation.continual_control_evaluator as control_evaluator_module
from alberta_framework.core.oak import OaKConfig
from alberta_framework.core.options import STOMPConfig, SubtaskSpec
from alberta_framework.core.prototype_agent import PrototypeAgent, PrototypeAgentConfig
from alberta_framework.evaluation.continual_control_evaluator import (
    ACCEPTANCE_STATUS,
    CONTROL_CHECKPOINT_SCHEMA,
    CONTROL_REPORT_SCHEMA,
    ContinualControlEvaluator,
    ContinuingControlBudget,
    ContinuingControlLearner,
    ContinuingControlProtocol,
    ControlDecision,
    ControlEnvironmentUpdate,
    ControlLearnerUpdate,
    ControlProbe,
    ControlResourceUsage,
    ControlTransition,
    FrozenActionControlBaseline,
    PrototypeAgentControlAdapter,
    RunningRewardBanditControlBaseline,
    continual_control_report_json,
    load_continual_control_report,
    validate_continual_control_report,
)


class ConstantDurationClock:
    """Every measured operation consumes exactly one microsecond."""

    def __init__(self) -> None:
        self._now = 0

    def __call__(self) -> int:
        value = self._now
        self._now += 1_000
        return value


def _decision_id(lifecycle: tuple[int, int], generation: int) -> tuple[int, int, int, int]:
    return (
        lifecycle[0],
        lifecycle[1],
        generation >> 32,
        generation & ((1 << 32) - 1),
    )


@dataclass(frozen=True)
class CueState:
    observation: tuple[float, ...]
    generation: int
    terminated_count: int
    truncated_count: int
    last_bootstrap: tuple[float, ...]
    last_reset: tuple[float, ...] | None


class CueControlLearner:
    """Uses an observation cue, not evaluator regime identity, to act."""

    def __init__(
        self,
        *,
        name: str = "cue_candidate",
        lifecycle: tuple[int, int] = (101, 202),
    ) -> None:
        self._name = name
        self._lifecycle = lifecycle

    @property
    def name(self) -> str:
        return self._name

    @property
    def n_actions(self) -> int:
        return 2

    @property
    def max_backward_calls_per_update(self) -> int:
        return 0

    def to_config(self) -> dict[str, object]:
        return {
            "type": type(self).__name__,
            "schema_version": "tests.cue_control_learner.v1",
            "name": self._name,
            "lifecycle": list(self._lifecycle),
        }

    def init(self, initial_observation: tuple[float, ...]) -> CueState:
        return CueState(initial_observation, 0, 0, 0, initial_observation, None)

    def state_valid_for_observation(
        self,
        state: Any,
        observation: tuple[float, ...],
    ) -> bool:
        resolved = cast(CueState, state)
        return resolved.observation == observation and resolved.generation >= 0

    @staticmethod
    def _action(observation: tuple[float, ...]) -> int:
        return int(observation[0] >= 0.5)

    def decide(
        self,
        state: Any,
        observation: tuple[float, ...],
    ) -> ControlDecision:
        resolved = cast(CueState, state)
        if not self.state_valid_for_observation(resolved, observation):
            raise ValueError("cue state does not own observation")
        return ControlDecision(
            observation=observation,
            action=self._action(observation),
            decision_id=_decision_id(self._lifecycle, resolved.generation),
        )

    def update(
        self,
        state: Any,
        transition: ControlTransition,
    ) -> ControlLearnerUpdate:
        resolved = cast(CueState, state)
        expected = self.decide(resolved, resolved.observation)
        if (
            transition.observation != expected.observation
            or transition.action != expected.action
            or transition.decision_id != expected.decision_id
        ):
            return ControlLearnerUpdate(resolved, False, 0)
        return ControlLearnerUpdate(
            CueState(
                observation=transition.next_decision_observation,
                generation=resolved.generation + 1,
                terminated_count=resolved.terminated_count + int(transition.terminated),
                truncated_count=resolved.truncated_count + int(transition.truncated),
                last_bootstrap=transition.bootstrap_observation,
                last_reset=transition.reset_observation,
            ),
            True,
            0,
        )

    def probe_action(self, state: Any, observation: tuple[float, ...]) -> int:
        del state
        return self._action(observation)

    def state_to_config(self, state: Any) -> object:
        resolved = cast(CueState, state)
        return {
            "observation": list(resolved.observation),
            "generation": resolved.generation,
            "terminated_count": resolved.terminated_count,
            "truncated_count": resolved.truncated_count,
            "last_bootstrap": list(resolved.last_bootstrap),
            "last_reset": None if resolved.last_reset is None else list(resolved.last_reset),
        }

    def state_from_config(self, payload: object) -> CueState:
        if not isinstance(payload, Mapping) or set(payload) != {
            "observation",
            "generation",
            "terminated_count",
            "truncated_count",
            "last_bootstrap",
            "last_reset",
        }:
            raise ValueError("cue state is invalid")

        def values(name: str) -> tuple[float, ...]:
            raw = payload[name]
            if not isinstance(raw, list):
                raise ValueError(f"{name} must be a list")
            return tuple(float(value) for value in raw)

        raw_reset = payload["last_reset"]
        if raw_reset is not None and not isinstance(raw_reset, list):
            raise ValueError("last_reset must be a list or None")
        return CueState(
            observation=values("observation"),
            generation=int(payload["generation"]),
            terminated_count=int(payload["terminated_count"]),
            truncated_count=int(payload["truncated_count"]),
            last_bootstrap=values("last_bootstrap"),
            last_reset=(
                None
                if raw_reset is None
                else tuple(float(value) for value in raw_reset)
            ),
        )

    def resource_usage(self, state: Any) -> ControlResourceUsage:
        resolved = cast(CueState, state)
        scalars = 4 + len(resolved.observation) + len(resolved.last_bootstrap)
        if resolved.last_reset is not None:
            scalars += len(resolved.last_reset)
        return ControlResourceUsage(
            persistent_state_bytes=8 * scalars,
            state_scalar_count=scalars,
            trainable_parameter_count=0,
            measurement_method="test exact logical scalar accounting",
        )

    def diagnostics(self, state: Any) -> Mapping[str, bool | int | float | str]:
        resolved = cast(CueState, state)
        return {
            "generation": resolved.generation,
            "terminated_count": resolved.terminated_count,
            "truncated_count": resolved.truncated_count,
        }


class RepeatedDecisionLearner(CueControlLearner):
    def __init__(self) -> None:
        super().__init__(name="repeated_decision")

    def decide(
        self,
        state: Any,
        observation: tuple[float, ...],
    ) -> ControlDecision:
        resolved = cast(CueState, state)
        if not self.state_valid_for_observation(resolved, observation):
            raise ValueError("cue state does not own observation")
        return ControlDecision(
            observation=observation,
            action=self._action(observation),
            decision_id=_decision_id(self._lifecycle, 0),
        )


class RejectingLearner(CueControlLearner):
    def __init__(self) -> None:
        super().__init__(name="rejecting")

    def update(
        self,
        state: Any,
        transition: ControlTransition,
    ) -> ControlLearnerUpdate:
        del transition
        return ControlLearnerUpdate(state, False, 0)


class ProbeMutatingLearner(CueControlLearner):
    def __init__(self) -> None:
        super().__init__(name="probe_mutator")

    def probe_action(self, state: Any, observation: tuple[float, ...]) -> int:
        resolved = cast(CueState, state)
        object.__setattr__(resolved, "generation", resolved.generation + 1)
        return super().probe_action(resolved, observation)


class DeliberatelyForgettingProbeLearner(CueControlLearner):
    """Controls from the current cue but probes with only the last online action."""

    def __init__(self) -> None:
        super().__init__(name="deliberately_forgetting")

    def update(
        self,
        state: Any,
        transition: ControlTransition,
    ) -> ControlLearnerUpdate:
        result = super().update(state, transition)
        if not result.applied:
            return result
        next_state = cast(CueState, result.state)
        return ControlLearnerUpdate(
            dataclasses.replace(
                next_state,
                last_bootstrap=(float(transition.action),),
            ),
            True,
            0,
        )

    def probe_action(self, state: Any, observation: tuple[float, ...]) -> int:
        del observation
        resolved = cast(CueState, state)
        return int(resolved.last_bootstrap[0])


@dataclass(frozen=True)
class CueEnvironmentState:
    step: int
    observation: tuple[float, ...]


class CueEnvironment:
    """Deterministic evaluator-side switched control environment."""

    def __init__(
        self,
        schedule: tuple[str, ...],
        *,
        observation_dim: int = 1,
        ownership_mismatch: bool = False,
        mutate_source: bool = False,
    ) -> None:
        self._schedule = schedule
        self._observation_dim = observation_dim
        self._ownership_mismatch = ownership_mismatch
        self._mutate_source = mutate_source

    @property
    def n_actions(self) -> int:
        return 2

    def to_config(self) -> dict[str, object]:
        return {
            "type": "CueEnvironment",
            "schema_version": "tests.cue_environment.v1",
            "schedule": list(self._schedule),
            "observation_dim": self._observation_dim,
            "ownership_mismatch": self._ownership_mismatch,
            "mutate_source": self._mutate_source,
        }

    def _cue(self, index: int) -> tuple[float, ...]:
        regime = self._schedule[min(index, len(self._schedule) - 1)]
        first = 0.0 if regime == "A" else 1.0
        return (first, *((0.0,) * (self._observation_dim - 1)))

    def init(self) -> CueEnvironmentState:
        return CueEnvironmentState(0, self._cue(0))

    def observation(self, state: Any) -> tuple[float, ...]:
        return cast(CueEnvironmentState, state).observation

    def step(
        self,
        state: Any,
        decision: ControlDecision,
        evaluator_regime_id: str,
    ) -> ControlEnvironmentUpdate:
        resolved = cast(CueEnvironmentState, state)
        if evaluator_regime_id != self._schedule[resolved.step]:
            raise ValueError("evaluator schedule and environment state disagree")
        if decision.observation != resolved.observation:
            raise ValueError("environment received a stale observation")
        source_step = resolved.step
        if self._mutate_source:
            object.__setattr__(resolved, "step", resolved.step + 1)
        completed = source_step + 1
        expected_action = 0 if evaluator_regime_id == "A" else 1
        reward = 1.0 if decision.action == expected_action else -1.0
        truncated = completed == 2
        terminated = completed == 4
        boundary = truncated or terminated
        next_cue = self._cue(completed)
        bootstrap = (
            (9.0, *((0.0,) * (self._observation_dim - 1)))
            if boundary
            else next_cue
        )
        reset = next_cue if boundary else None
        next_state = CueEnvironmentState(completed, next_cue)
        decision_id = decision.decision_id
        if self._ownership_mismatch:
            decision_id = (*decision_id[:3], decision_id[3] + 1)
        wrong = reward < 0.0
        return ControlEnvironmentUpdate(
            state=next_state,
            transition=ControlTransition(
                observation=decision.observation,
                action=decision.action,
                decision_id=decision_id,
                reward=reward,
                discount=0.0 if terminated else 0.9,
                terminated=terminated,
                truncated=truncated,
                bootstrap_observation=bootstrap,
                reset_observation=reset,
                safety_violation=False,
                intervention=False,
                near_miss=wrong,
                safety_cost=float(wrong),
                near_miss_cost=float(wrong),
            ),
        )

    def state_to_config(self, state: Any) -> object:
        resolved = cast(CueEnvironmentState, state)
        return {"step": resolved.step, "observation": list(resolved.observation)}

    def state_from_config(self, payload: object) -> CueEnvironmentState:
        if not isinstance(payload, Mapping) or set(payload) != {"step", "observation"}:
            raise ValueError("environment state is invalid")
        observation = payload["observation"]
        if not isinstance(observation, list):
            raise ValueError("environment observation must be a list")
        return CueEnvironmentState(
            int(payload["step"]),
            tuple(float(value) for value in observation),
        )


SCHEDULE = ("A", "A", "B", "B", "A", "A")


def _protocol() -> ContinuingControlProtocol:
    return ContinuingControlProtocol(
        protocol_id="tests.switched-control.v2",
        higher_is_better=True,
        regime_schedule=SCHEDULE,
        evaluator_regime_ids=("A", "B"),
        checkpoint_steps=(2, 4, 6),
        first_exposure_checkpoint={"A": 0, "B": 1},
        forward_transfer_reference={"A": 0.5, "B": 0.5},
        recovery_thresholds={"A": 0.5, "B": 0.5},
        stability_references={"A": 1.0, "B": 1.0},
        recovery_window=1,
        worst_window_size=2,
        operation_latency_deadline_ms=0.0005,
    )


def _budget(*, transition_count: int = 6, probe_calls: int = 6) -> ContinuingControlBudget:
    return ContinuingControlBudget(
        transition_limit=transition_count,
        decision_call_limit=transition_count,
        environment_call_limit=transition_count,
        update_call_limit=transition_count,
        probe_call_limit=probe_calls,
        backward_call_limit=0,
        persistent_state_bytes_limit=1_000_000,
        state_scalar_count_limit=250_000,
        trainable_parameter_count_limit=None,
        stored_decision_id_limit=transition_count,
    )


def _probes(observation_dim: int = 1) -> dict[str, tuple[ControlProbe, ...]]:
    zeros = (0.0,) * (observation_dim - 1)
    return {
        "A": (ControlProbe((0.0, *zeros), (1.0, 0.0)),),
        "B": (ControlProbe((1.0, *zeros), (0.0, 1.0)),),
    }


def _evaluator(
    *,
    candidate: ContinuingControlLearner | None = None,
    environment: CueEnvironment | None = None,
) -> ContinualControlEvaluator:
    return ContinualControlEvaluator(
        run_id="tests.switched-control-run.v1",
        protocol=_protocol(),
        environment=CueEnvironment(SCHEDULE) if environment is None else environment,
        probes=_probes(),
        candidate=CueControlLearner() if candidate is None else candidate,
        baselines=(
            FrozenActionControlBaseline(n_actions=2, action=0, name="frozen_zero"),
            FrozenActionControlBaseline(n_actions=2, action=1, name="frozen_one"),
            RunningRewardBanditControlBaseline(n_actions=2),
        ),
        budget=_budget(),
        clock_ns=ConstantDurationClock(),
        latency_measurement_method="deterministic test clock: 1000 ns per operation",
    )


def _condition(report: dict[str, object], name: str) -> dict[str, object]:
    conditions = cast(list[dict[str, object]], report["conditions"])
    return next(condition for condition in conditions if condition["name"] == name)


@pytest.mark.unit
def test_control_learner_surface_and_transition_exclude_evaluator_regime_identity() -> None:
    assert "regime" not in {field.name for field in dataclasses.fields(ControlTransition)}
    assert "task" not in {field.name for field in dataclasses.fields(ControlTransition)}
    for method_name in (
        "init",
        "state_valid_for_observation",
        "decide",
        "update",
        "probe_action",
    ):
        parameters = inspect.signature(
            getattr(ContinuingControlLearner, method_name)
        ).parameters
        assert all("regime" not in name and "task" not in name for name in parameters)
    environment_parameters = inspect.signature(CueEnvironment.step).parameters
    assert "evaluator_regime_id" in environment_parameters


@pytest.mark.unit
def test_transition_requires_explicit_and_consistent_boundary_observations() -> None:
    with pytest.raises(ValueError, match="reset_observation"):
        ControlTransition(
            observation=(0.0,),
            action=0,
            decision_id=(1, 2, 3, 4),
            reward=0.0,
            discount=0.9,
            terminated=False,
            truncated=True,
            bootstrap_observation=(9.0,),
            reset_observation=None,
        )
    with pytest.raises(ValueError, match="zero discount"):
        ControlTransition(
            observation=(0.0,),
            action=0,
            decision_id=(1, 2, 3, 4),
            reward=0.0,
            discount=0.0,
            terminated=False,
            truncated=False,
            bootstrap_observation=(1.0,),
            reset_observation=None,
        )
    with pytest.raises(ValueError, match="must be None"):
        ControlTransition(
            observation=(0.0,),
            action=0,
            decision_id=(1, 2, 3, 4),
            reward=0.0,
            discount=0.9,
            terminated=False,
            truncated=False,
            bootstrap_observation=(1.0,),
            reset_observation=(2.0,),
        )
    with pytest.raises(ValueError, match="cannot exceed"):
        ControlTransition(
            observation=(0.0,),
            action=0,
            decision_id=(1, 2, 3, 4),
            reward=0.0,
            discount=0.9,
            terminated=False,
            truncated=False,
            bootstrap_observation=(1.0,),
            reset_observation=None,
            near_miss=True,
            safety_cost=0.25,
            near_miss_cost=0.5,
        )
    with pytest.raises(ValueError, match="requires near_miss"):
        ControlTransition(
            observation=(0.0,),
            action=0,
            decision_id=(1, 2, 3, 4),
            reward=0.0,
            discount=0.9,
            terminated=False,
            truncated=False,
            bootstrap_observation=(1.0,),
            reset_observation=None,
            near_miss=False,
            safety_cost=0.5,
            near_miss_cost=0.5,
        )


@pytest.mark.integration
def test_control_run_pins_ordering_boundaries_probes_resources_and_honest_limits() -> None:
    evaluator = _evaluator()
    state = evaluator.advance(evaluator.init(), steps=6)
    report = evaluator.build_report(state)
    assert report["schema_version"] == CONTROL_REPORT_SCHEMA
    assert report["acceptance_status"] == ACCEPTANCE_STATUS
    assert report["accepted_scientific_evidence"] is False
    protocol = cast(dict[str, object], report["protocol"])
    assert protocol["regime_metadata_is_evaluator_only"] is True

    candidate = _condition(report, "cue_candidate")
    assert candidate["predict_action_before_environment_outcome"] is True
    trace = cast(dict[str, object], candidate["trace"])
    assert trace == {
        "rewards": [1.0] * 6,
        "processed": [True] * 6,
        "evaluator_regime_ids_in_learner_trace": False,
    }
    probes = cast(dict[str, object], candidate["held_out_non_learning_probes"])
    assert probes["values"] == [[1.0, 1.0]] * 3
    assert probes["snapshot_mutation_checks"] is True
    metrics = cast(dict[str, object], candidate["metrics"])
    assert metrics["prequential_return"] == 1.0
    assert metrics["lifetime_return"] == 1.0
    assert metrics["per_regime_online_mean_return"] == {"A": 1.0, "B": 1.0}

    operations = cast(dict[str, object], candidate["operations"])
    counts = cast(dict[str, object], operations["counts"])
    assert counts == {
        "processed_transitions": 6,
        "delayed_transitions": 6,
        "dropped_transitions": 0,
        "decision_calls": 6,
        "environment_calls": 6,
        "update_calls": 6,
        "held_out_probe_calls": 6,
        "backward_calls": 0,
        "backward_calls_available": True,
    }
    latency = cast(dict[str, object], operations["latency_ms"])
    expected_latency = {"p50_ms": 0.001, "p95_ms": 0.001, "p99_ms": 0.001}
    assert latency["decision"] == expected_latency
    assert latency["environment"] == expected_latency
    assert latency["update"] == expected_latency
    assert latency["held_out_probe"] == expected_latency

    diagnostics = cast(dict[str, object], candidate["diagnostics"])
    assert diagnostics["terminated_count"] == 1
    assert diagnostics["truncated_count"] == 1
    safety = cast(dict[str, object], candidate["safety"])
    assert safety["checks"] == 6
    assert safety["near_misses"] == 0

    frozen_zero = _condition(report, "frozen_zero")
    frozen_safety = cast(dict[str, object], frozen_zero["safety"])
    assert frozen_safety["near_misses"] == 2
    comparison = cast(dict[str, object], report["comparison_contract"])
    assert comparison["shared_ceiling_verified"] is True
    assert comparison["realized_compute_memory_parity_verified"] is False
    assert comparison["baselines"] == [
        "frozen_zero",
        "frozen_one",
        "running_reward_bandit",
    ]

    identity = cast(dict[str, object], report["evaluator_identity"])
    assert len(cast(str, identity["evaluator_config_sha256"])) == 64
    assert len(cast(str, identity["environment_config_sha256"])) == 64
    assert identity["probe_sha256"] == cast(
        dict[str, object],
        report["evaluator_config"],
    )["probe_sha256"]
    assert validate_continual_control_report(
        report,
        expected_evaluator_config=evaluator.to_config(),
    ).valid


@pytest.mark.unit
def test_longitudinal_control_metrics_match_hand_calculation_and_exclude_initial_segment() -> None:
    evaluator = _evaluator()
    report = evaluator.build_report(evaluator.advance(evaluator.init(), steps=6))
    frozen = _condition(report, "frozen_zero")
    metrics = cast(dict[str, object], frozen["metrics"])
    assert metrics["prequential_return"] == pytest.approx(1.0 / 3.0)
    assert metrics["lifetime_return"] == pytest.approx(1.0 / 3.0)
    assert metrics["per_regime_online_mean_return"] == {"A": 1.0, "B": -1.0}

    adaptation = cast(dict[str, object], metrics["adaptation_auc"])
    assert adaptation["mean_normalized_auc"] == 0.0
    segments = cast(list[dict[str, object]], adaptation["segments"])
    assert segments == [
        {
            "segment_index": 1,
            "regime_id": "B",
            "start_step": 2,
            "end_step_exclusive": 4,
            "normalized_auc": -1.0,
        },
        {
            "segment_index": 2,
            "regime_id": "A",
            "start_step": 4,
            "end_step_exclusive": 6,
            "normalized_auc": 1.0,
        },
    ]

    recovery = cast(dict[str, object], metrics["recovery"])
    assert recovery["event_count"] == 2
    assert recovery["assessable_event_count"] == 2
    assert recovery["unavailable_event_count"] == 0
    assert recovery["recovered_count"] == 1
    assert recovery["recovery_rate_over_assessable"] == 0.5
    assert recovery["mean_recovery_steps_over_recovered"] == 1.0
    recovery_events = cast(list[dict[str, object]], recovery["events"])
    assert recovery_events[0]["recovered"] is False
    assert recovery_events[0]["recovery_steps"] is None
    assert recovery_events[1]["recovered"] is True
    assert recovery_events[1]["recovery_steps"] == 1

    assert metrics["per_regime_final_performance"] == {"A": 1.0, "B": 0.0}
    assert metrics["mean_final_performance"] == 0.5
    forgetting = cast(dict[str, object], metrics["forgetting"])
    backward = cast(dict[str, object], metrics["backward_transfer"])
    forward = cast(dict[str, object], metrics["forward_transfer"])
    assert forgetting == {
        "mean_over_available": 0.0,
        "maximum_over_available": 0.0,
        "available_regime_count": 2,
        "per_regime": {"A": 0.0, "B": 0.0},
    }
    assert backward == {
        "mean_over_available": 0.0,
        "available_regime_count": 2,
        "per_regime": {"A": 0.0, "B": 0.0},
    }
    assert forward == {
        "mean_over_available": -0.5,
        "available_regime_count": 1,
        "per_regime": {"A": None, "B": -0.5},
    }
    stability = cast(dict[str, object], metrics["stability"])
    assert stability["mean_gap"] == 1.0
    assert stability["maximum_gap"] == 2.0
    stability_events = cast(list[dict[str, object]], stability["events"])
    assert [event["step"] for event in stability_events] == [2, 4]
    assert metrics["worst_window"] == {
        "window_size": 2,
        "mean_return": -1.0,
        "start_step": 2,
        "end_step_exclusive": 4,
    }
    applicability = cast(dict[str, object], metrics["metric_applicability"])
    fwt_applicability = cast(dict[str, object], applicability["forward_transfer"])
    assert fwt_applicability["available_regimes"] == ["B"]
    assert fwt_applicability["unavailable_regimes"] == {
        "A": "no pre-exposure held-out checkpoint exists for this regime"
    }


@pytest.mark.integration
def test_deliberately_forgetting_probe_learner_exposes_forgetting_and_negative_bwt() -> None:
    evaluator = _evaluator(candidate=DeliberatelyForgettingProbeLearner())
    report = evaluator.build_report(evaluator.advance(evaluator.init(), steps=6))
    candidate = _condition(report, "deliberately_forgetting")
    trace = cast(dict[str, object], candidate["trace"])
    assert trace["rewards"] == [1.0] * 6
    probes = cast(dict[str, object], candidate["held_out_non_learning_probes"])
    assert probes["values"] == [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
    metrics = cast(dict[str, object], candidate["metrics"])
    forgetting = cast(dict[str, object], metrics["forgetting"])
    backward = cast(dict[str, object], metrics["backward_transfer"])
    forward = cast(dict[str, object], metrics["forward_transfer"])
    assert forgetting == {
        "mean_over_available": 0.5,
        "maximum_over_available": 1.0,
        "available_regime_count": 2,
        "per_regime": {"A": 0.0, "B": 1.0},
    }
    assert backward == {
        "mean_over_available": -0.5,
        "available_regime_count": 2,
        "per_regime": {"A": 0.0, "B": -1.0},
    }
    assert forward == {
        "mean_over_available": -0.5,
        "available_regime_count": 1,
        "per_regime": {"A": None, "B": -0.5},
    }
    assert validate_continual_control_report(report).valid


@pytest.mark.integration
def test_longitudinal_metric_unavailability_is_explicit_for_sparse_probes() -> None:
    schedule = ("A", "B", "A", "B")
    protocol = ContinuingControlProtocol(
        protocol_id="tests.sparse-control.v2",
        higher_is_better=True,
        regime_schedule=schedule,
        evaluator_regime_ids=("A", "B"),
        checkpoint_steps=(4,),
        first_exposure_checkpoint={"A": 0, "B": 0},
        forward_transfer_reference={"A": 0.5, "B": 0.5},
        recovery_thresholds={"A": 0.5, "B": 0.5},
        stability_references={"A": 1.0, "B": 1.0},
        recovery_window=2,
        worst_window_size=1,
        operation_latency_deadline_ms=1.0,
    )
    evaluator = ContinualControlEvaluator(
        run_id="tests.sparse-control-run.v2",
        protocol=protocol,
        environment=CueEnvironment(schedule),
        probes=_probes(),
        candidate=CueControlLearner(),
        baselines=(
            FrozenActionControlBaseline(n_actions=2, name="sparse_frozen_zero"),
            FrozenActionControlBaseline(
                n_actions=2,
                action=1,
                name="sparse_frozen_one",
            ),
        ),
        budget=_budget(transition_count=4, probe_calls=2),
        clock_ns=ConstantDurationClock(),
        latency_measurement_method="deterministic test clock: 1000 ns per operation",
    )
    report = evaluator.build_report(evaluator.advance(evaluator.init(), steps=4))
    candidate = _condition(report, "cue_candidate")
    metrics = cast(dict[str, object], candidate["metrics"])
    recovery = cast(dict[str, object], metrics["recovery"])
    assert recovery["assessable_event_count"] == 0
    assert recovery["unavailable_event_count"] == 3
    assert recovery["recovery_rate_over_assessable"] is None
    assert all(
        event["recovered"] is None and event["unavailable_reason"] is not None
        for event in cast(list[dict[str, object]], recovery["events"])
    )
    for metric_name in ("forgetting", "backward_transfer", "forward_transfer"):
        metric = cast(dict[str, object], metrics[metric_name])
        assert metric["available_regime_count"] == 0
        assert metric["per_regime"] == {"A": None, "B": None}
    applicability = cast(dict[str, object], metrics["metric_applicability"])
    assert cast(dict[str, object], applicability["recovery"])["applicable"] is False
    for metric_name in ("forgetting", "backward_transfer", "forward_transfer"):
        assert cast(dict[str, object], applicability[metric_name])["applicable"] is False
    assert validate_continual_control_report(report).valid


@pytest.mark.integration
def test_report_loader_strictly_reconstructs_metrics_identity_and_json(tmp_path: Path) -> None:
    evaluator = _evaluator()
    report = evaluator.build_report(evaluator.advance(evaluator.init(), steps=6))
    destination = tmp_path / "report.json"
    destination.write_text(continual_control_report_json(report), encoding="utf-8")
    assert load_continual_control_report(
        destination,
        expected_evaluator_config=evaluator.to_config(),
    ) == report

    tampered_metric = json.loads(destination.read_text(encoding="utf-8"))
    tampered_metric["conditions"][0]["metrics"]["lifetime_return"] = -999.0
    validation = validate_continual_control_report(tampered_metric)
    assert not validation.valid
    assert "do not reconstruct" in validation.errors[0]

    tampered_identity = json.loads(destination.read_text(encoding="utf-8"))
    tampered_identity["evaluator_identity"]["probe_sha256"] = "0" * 64
    validation = validate_continual_control_report(tampered_identity)
    assert not validation.valid
    assert "identity digests" in validation.errors[0]

    duplicate = destination.read_text(encoding="utf-8").replace(
        "{\n",
        '{\n  "schema_version": "alberta.continual_control_report.v2",\n',
        1,
    )
    duplicate_path = tmp_path / "duplicate.json"
    duplicate_path.write_text(duplicate, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON"):
        load_continual_control_report(duplicate_path)


@pytest.mark.integration
def test_checkpoint_resume_is_canonical_deterministic_and_config_bound(tmp_path: Path) -> None:
    uninterrupted_evaluator = _evaluator()
    uninterrupted = uninterrupted_evaluator.advance(
        uninterrupted_evaluator.init(),
        steps=6,
    )
    uninterrupted_report = uninterrupted_evaluator.build_report(uninterrupted)

    first = _evaluator()
    partial = first.advance(first.init(), steps=3)
    checkpoint = tmp_path / "control.json"
    first.save_checkpoint(partial, checkpoint)
    raw = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert raw["schema_version"] == CONTROL_CHECKPOINT_SCHEMA
    assert raw["state"]["step"] == 3

    resumed_evaluator = _evaluator()
    restored = resumed_evaluator.load_checkpoint(checkpoint)
    resumed = resumed_evaluator.advance(restored, steps=3)
    assert resumed_evaluator.build_report(resumed) == uninterrupted_report

    raw["state"]["conditions"][0]["rewards"][0] = -123.0
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="state digest"):
        _evaluator().load_checkpoint(tampered)

    incompatible = ContinualControlEvaluator(
        run_id="tests.switched-control-run.v2",
        protocol=_protocol(),
        environment=CueEnvironment(SCHEDULE),
        probes=_probes(),
        candidate=CueControlLearner(),
        baselines=(
            FrozenActionControlBaseline(n_actions=2),
            FrozenActionControlBaseline(n_actions=2, action=1, name="frozen_one"),
        ),
        budget=_budget(),
        clock_ns=ConstantDurationClock(),
        latency_measurement_method="deterministic test clock: 1000 ns per operation",
    )
    with pytest.raises(ValueError, match="config does not match"):
        incompatible.load_checkpoint(checkpoint)


@pytest.mark.integration
def test_checkpoint_replace_failure_preserves_previous_file_and_cleans_temp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator = _evaluator()
    destination = tmp_path / "atomic-control.json"
    initial = evaluator.init()
    evaluator.save_checkpoint(initial, destination)
    previous = destination.read_bytes()
    advanced = evaluator.advance(initial)

    def fail_replace(_source: object, _destination: object) -> None:
        raise OSError("injected replace failure")

    monkeypatch.setattr(control_evaluator_module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected replace failure"):
        evaluator.save_checkpoint(advanced, destination)
    assert destination.read_bytes() == previous
    assert not list(tmp_path.glob(".atomic-control.json.*.tmp"))


@pytest.mark.unit
def test_fail_closed_on_environment_ownership_replay_rejection_and_mutation() -> None:
    mismatch = _evaluator(environment=CueEnvironment(SCHEDULE, ownership_mismatch=True))
    mismatch_state = mismatch.init()
    with pytest.raises(ValueError, match="does not own"):
        mismatch.advance(mismatch_state)
    assert mismatch_state.step == 0

    repeated = _evaluator(candidate=RepeatedDecisionLearner())
    repeated_state = repeated.init()
    with pytest.raises(ValueError, match="already consumed"):
        repeated.advance(repeated_state)
    assert repeated_state.step == 0

    rejecting = _evaluator(candidate=RejectingLearner())
    rejecting_state = rejecting.init()
    with pytest.raises(ValueError, match="rejected an owned"):
        rejecting.advance(rejecting_state)
    assert rejecting_state.step == 0

    mutating_environment = _evaluator(
        environment=CueEnvironment(SCHEDULE, mutate_source=True)
    )
    mutating_state = mutating_environment.init()
    with pytest.raises(ValueError, match="mutated its source"):
        mutating_environment.advance(mutating_state)
    assert cast(CueEnvironmentState, mutating_state.conditions[0].environment_state).step == 0


@pytest.mark.unit
def test_held_out_probe_uses_isolated_snapshot_and_detects_mutation() -> None:
    evaluator = _evaluator(candidate=ProbeMutatingLearner())
    state = evaluator.init()
    with pytest.raises(ValueError, match="held-out probe mutated snapshot"):
        evaluator.advance(state, steps=2)
    live = cast(CueState, state.conditions[0].learner_state)
    assert live.generation == 0


def _prototype_agent() -> PrototypeAgent:
    stomp = STOMPConfig(
        subtask_specs=(
            SubtaskSpec(feature_index=0, threshold=1.0e6, max_option_steps=4),
        ),
        observation_dim=2,
        n_primitive_actions=2,
        base_step_size=0.05,
        base_avg_reward_step_size=0.01,
        option_gamma=0.9,
        epsilon_base=0.0,
        epsilon_option=0.0,
    )
    return PrototypeAgent(PrototypeAgentConfig(oak=OaKConfig(stomp=stomp)))


@pytest.mark.integration
def test_prototype_adapter_roundtrips_and_runs_owned_control_without_fake_counts() -> None:
    adapter = PrototypeAgentControlAdapter(
        _prototype_agent(),
        seed=7,
        lifecycle_id=(700, 701),
    )
    state = adapter.init((0.0, 0.0))
    payload = adapter.state_to_config(state)
    restored = adapter.state_from_config(payload)
    assert adapter.state_to_config(restored) == payload
    assert adapter.state_valid_for_observation(restored, (0.0, 0.0))
    decision = adapter.decide(restored, (0.0, 0.0))
    assert decision.decision_id[:2] == (700, 701)

    protocol = ContinuingControlProtocol(
        protocol_id="tests.prototype-control.v2",
        higher_is_better=True,
        regime_schedule=("A", "B"),
        evaluator_regime_ids=("A", "B"),
        checkpoint_steps=(1, 2),
        first_exposure_checkpoint={"A": 0, "B": 1},
        forward_transfer_reference={"A": 0.5, "B": 0.5},
        recovery_thresholds={"A": 0.0, "B": 0.0},
        stability_references={"A": 0.0, "B": 0.0},
        recovery_window=1,
        worst_window_size=1,
        operation_latency_deadline_ms=10_000.0,
    )
    evaluator = ContinualControlEvaluator(
        run_id="tests.prototype-control-run.v1",
        protocol=protocol,
        environment=CueEnvironment(("A", "B"), observation_dim=2),
        probes=_probes(observation_dim=2),
        candidate=adapter,
        baselines=(
            FrozenActionControlBaseline(n_actions=2, name="prototype_frozen_zero"),
            FrozenActionControlBaseline(
                n_actions=2,
                action=1,
                name="prototype_frozen_one",
            ),
        ),
        budget=_budget(transition_count=2, probe_calls=4),
        clock_ns=ConstantDurationClock(),
        latency_measurement_method="deterministic test clock: 1000 ns per operation",
    )
    completed = evaluator.advance(evaluator.init(), steps=2)
    report = evaluator.build_report(completed)
    candidate = _condition(report, "prototype_agent")
    counts = cast(
        dict[str, object],
        cast(dict[str, object], candidate["operations"])["counts"],
    )
    assert counts["backward_calls"] is None
    assert counts["backward_calls_available"] is False
    resources = cast(dict[str, object], candidate["resources"])
    assert resources["trainable_parameter_count"] is None
    assert "allocator residency" in cast(str, resources["measurement_method"])
    assert report["accepted_scientific_evidence"] is False


@pytest.mark.unit
def test_prototype_adapter_rejects_malformed_state_payload() -> None:
    adapter = PrototypeAgentControlAdapter(
        _prototype_agent(),
        seed=3,
        lifecycle_id=(300, 301),
    )
    state = adapter.init((0.0, 0.0))
    payload = cast(dict[str, object], adapter.state_to_config(state))
    leaves = cast(list[dict[str, object]], payload["leaves"])
    malformed = json.loads(json.dumps(payload))
    malformed["leaves"][0]["shape"] = [999]
    with pytest.raises(ValueError, match="shape does not match"):
        adapter.state_from_config(malformed)
    assert leaves

    inconsistent = adapter.state_to_config(
        state.replace(started=jnp.asarray(False, dtype=jnp.bool_))
    )
    with pytest.raises(ValueError, match="reconstructed PrototypeAgent state is inconsistent"):
        adapter.state_from_config(inconsistent)

    dishonest = json.loads(json.dumps(payload))
    typed_key = next(
        leaf
        for leaf in dishonest["leaves"]
        if leaf["kind"] == "typed_prng_key"
    )
    typed_key["data"][0] = 0.5
    with pytest.raises(ValueError, match="numeric payload is not canonical"):
        adapter.state_from_config(dishonest)


@pytest.mark.unit
@pytest.mark.parametrize("seed", (True, 1 << 32))
def test_prototype_adapter_rejects_boolean_and_overflow_seeds(seed: object) -> None:
    with pytest.raises(ValueError, match="seed"):
        PrototypeAgentControlAdapter(
            _prototype_agent(),
            seed=cast(int, seed),
            lifecycle_id=(300, 301),
        )


@pytest.mark.unit
def test_production_baselines_have_causal_unique_ids_and_exact_resources() -> None:
    frozen = FrozenActionControlBaseline(n_actions=2)
    bandit = RunningRewardBanditControlBaseline(n_actions=2)
    for learner in (frozen, bandit):
        state = learner.init((0.0,))
        first = learner.decide(state, (0.0,))
        transition = ControlTransition(
            observation=first.observation,
            action=first.action,
            decision_id=first.decision_id,
            reward=1.0,
            discount=0.9,
            terminated=False,
            truncated=False,
            bootstrap_observation=(1.0,),
            reset_observation=None,
        )
        update = learner.update(state, transition)
        assert update.applied
        second = learner.decide(update.state, (1.0,))
        assert second.decision_id != first.decision_id
        usage = learner.resource_usage(update.state)
        assert usage.persistent_state_bytes > 0
        assert usage.trainable_parameter_count is not None
        assert learner.max_backward_calls_per_update == 0
        assert jnp.isfinite(float(usage.persistent_state_bytes))
        assert jax.tree_util.tree_leaves(update.state)
