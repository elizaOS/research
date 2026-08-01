"""Bounded held-out component-retention probes for continual learners.

The evaluator owns regime identity, targets, and scoring.  A learner adapter
receives only :class:`ComponentProbeInput` and a reconstructed snapshot of its
state.  There is deliberately no learning/update method on the adapter
protocol: every measurement is a prediction from a frozen state, and the
evaluator rejects a probe that mutates that reconstructed state.

All component scores use a higher-is-better convention.  Representation,
dynamics/observation, reward, termination/discount, and critic/value scores
are negative mean squared errors.  Actor margin is the preferred held-out
action logit minus the strongest alternative; actor return is the held-out
return assigned to the deterministic argmax action.

This is a development-only localization mechanism.  It does not calibrate the
targets, establish causal attribution, compare learners, or promote a
scientific claim.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, NoReturn, Protocol, TypeGuard, cast, runtime_checkable

COMPONENT_RETENTION_REPORT_SCHEMA = "alberta.component_retention_report.v1"
COMPONENT_RETENTION_CHECKPOINT_SCHEMA = "alberta.component_retention_checkpoint.v1"
COMPONENT_RETENTION_EVALUATOR_SCHEMA = "alberta.component_retention_evaluator.v1"

COMPONENT_RETENTION_INTERPRETATION = (
    "Development-only descriptive held-out component-retention probe; status is "
    "not-assessed, and this report is not scientific acceptance or an Alberta Plan "
    "completion certificate."
)

COMPONENT_RETENTION_LIMITATIONS = (
    "Targets and probe coverage are caller-supplied and are not calibrated by this evaluator.",
    "Negative MSE does not establish semantic representation quality or causal attribution.",
    "Pointwise reward, discount, and value squared error is not distribution calibration or "
    "reward-discrimination analysis, and the dynamics probe is one-step only.",
    "Actor probes support finite discrete-action logits only; actor return uses caller-supplied "
    "per-action held-out returns rather than an environment rollout.",
    "Exact checkpoint ordering is checked against the adapter-reported learner step, but the "
    "evaluator cannot independently audit a dishonest adapter.",
    "State nonmutation is checked through canonical adapter serialization; hidden external side "
    "effects that serialization omits cannot be detected.",
    "The evaluator checkpoint contains its bounded accumulator only; learner state must be "
    "checkpointed separately and restored at the adapter-reported training step.",
    "One deterministic run is descriptive and supplies no multi-seed uncertainty or promotion.",
)

type ComponentName = Literal[
    "representation",
    "dynamics_observation",
    "reward",
    "termination_discount",
    "critic_value",
    "actor_margin",
    "actor_return",
]

COMPONENT_NAMES: tuple[ComponentName, ...] = (
    "representation",
    "dynamics_observation",
    "reward",
    "termination_discount",
    "critic_value",
    "actor_margin",
    "actor_return",
)

COMPONENT_METRIC_DEFINITIONS: Mapping[ComponentName, str] = {
    "representation": "negative mean squared error against the held-out representation target",
    "dynamics_observation": (
        "negative mean squared error of the one-step held-out observation prediction"
    ),
    "reward": "negative squared error of the held-out reward prediction",
    "termination_discount": (
        "negative squared error of the held-out continuation/discount prediction"
    ),
    "critic_value": "negative squared error of the held-out critic/value prediction",
    "actor_margin": (
        "held-out preferred-action logit minus the strongest alternative action logit"
    ),
    "actor_return": (
        "held-out per-action return assigned to the deterministic first-argmax policy action"
    ),
}


def _finite_number(value: object) -> TypeGuard[int | float]:
    return (
        not isinstance(value, bool)
        and isinstance(value, int | float)
        and math.isfinite(float(value))
    )


def _finite_float(value: object, *, name: str) -> float:
    if not _finite_number(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _positive_int(value: object, *, name: str) -> int:
    result = _nonnegative_int(value, name=name)
    if result == 0:
        raise ValueError(f"{name} must be positive")
    return result


def _versioned_identifier(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value or ".v" not in value:
        raise ValueError(f"{name} must be a non-empty versioned identifier")
    return value


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _json_clone(value: object) -> object:
    return json.loads(_canonical_json(value))


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _serialized_size(value: object) -> int:
    return len(_canonical_json(value).encode("utf-8"))


def _component_name(value: object, *, name: str) -> ComponentName:
    if value not in COMPONENT_NAMES:
        raise ValueError(f"{name} is not a recognized component")
    return value


@dataclasses.dataclass(frozen=True)
class ProbeValue:
    """An explicit available vector or an explicit unavailability reason."""

    available: bool
    values: tuple[float, ...]
    unavailable_reason: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.available, bool):
            raise ValueError("available must be boolean")
        if self.available:
            if not self.values:
                raise ValueError("available probe values must be non-empty")
            if self.unavailable_reason is not None:
                raise ValueError("available probe values cannot have an unavailable reason")
            for index, value in enumerate(self.values):
                _finite_float(value, name=f"values[{index}]")
        else:
            if self.values:
                raise ValueError("unavailable probe values must be empty")
            if not isinstance(self.unavailable_reason, str) or not self.unavailable_reason:
                raise ValueError("unavailable probe values require a non-empty reason")

    @classmethod
    def present(cls, values: Sequence[float]) -> ProbeValue:
        return cls(
            available=True,
            values=tuple(
                _finite_float(value, name=f"values[{index}]") for index, value in enumerate(values)
            ),
            unavailable_reason=None,
        )

    @classmethod
    def missing(cls, reason: str) -> ProbeValue:
        return cls(available=False, values=(), unavailable_reason=reason)

    def to_config(self) -> dict[str, object]:
        return {
            "available": self.available,
            "values": list(self.values),
            "unavailable_reason": self.unavailable_reason,
        }


@dataclasses.dataclass(frozen=True)
class ComponentProbeInput:
    """Learner-visible probe input; task and regime identity are absent."""

    observation: tuple[float, ...]
    action: int

    def __post_init__(self) -> None:
        if not self.observation:
            raise ValueError("observation must be non-empty")
        for index, value in enumerate(self.observation):
            _finite_float(value, name=f"observation[{index}]")
        _nonnegative_int(self.action, name="action")

    def to_config(self) -> dict[str, object]:
        return {"observation": list(self.observation), "action": self.action}


@dataclasses.dataclass(frozen=True)
class ComponentProbeTargets:
    """Evaluator-only targets for all WP4.5 component channels.

    ``actor_margin`` contains one integer-valued preferred action encoded as a
    float.  ``actor_return`` contains one held-out return for every discrete
    action.  An unavailable target makes that component inapplicable for the
    corresponding evaluator regime.
    """

    representation: ProbeValue
    dynamics_observation: ProbeValue
    reward: ProbeValue
    termination_discount: ProbeValue
    critic_value: ProbeValue
    actor_margin: ProbeValue
    actor_return: ProbeValue

    def __post_init__(self) -> None:
        for component in COMPONENT_NAMES:
            value = self.for_component(component)
            if not isinstance(value, ProbeValue):
                raise TypeError(f"{component} target must be ProbeValue")
        for component in ("reward", "termination_discount", "critic_value"):
            target = self.for_component(component)
            if target.available and len(target.values) != 1:
                raise ValueError(f"{component} target must contain exactly one scalar")
        discount = self.termination_discount
        if discount.available and not 0.0 <= discount.values[0] <= 1.0:
            raise ValueError("termination_discount target must be in [0, 1]")
        margin = self.actor_margin
        if margin.available:
            if len(margin.values) != 1:
                raise ValueError("actor_margin target must contain one preferred action")
            action = margin.values[0]
            if action < 0.0 or not action.is_integer():
                raise ValueError("actor_margin preferred action must be a non-negative integer")
        returns = self.actor_return
        if returns.available and len(returns.values) < 2:
            raise ValueError("actor_return target requires at least two discrete actions")
        if margin.available and returns.available:
            if int(margin.values[0]) >= len(returns.values):
                raise ValueError("actor_margin preferred action exceeds actor_return actions")

    def for_component(self, component: ComponentName) -> ProbeValue:
        return cast(ProbeValue, getattr(self, component))

    def to_config(self) -> dict[str, object]:
        return {
            component: self.for_component(component).to_config() for component in COMPONENT_NAMES
        }


@dataclasses.dataclass(frozen=True)
class HeldOutComponentProbe:
    """One evaluator-owned held-out case."""

    probe_input: ComponentProbeInput
    targets: ComponentProbeTargets

    def to_config(self) -> dict[str, object]:
        return {
            "probe_input": self.probe_input.to_config(),
            "targets": self.targets.to_config(),
        }


@dataclasses.dataclass(frozen=True)
class LearnerComponentPredictions:
    """Learner predictions from one frozen state and target-free input."""

    representation: ProbeValue
    dynamics_observation: ProbeValue
    reward: ProbeValue
    termination_discount: ProbeValue
    critic_value: ProbeValue
    actor_logits: ProbeValue

    def __post_init__(self) -> None:
        for name in (
            "representation",
            "dynamics_observation",
            "reward",
            "termination_discount",
            "critic_value",
            "actor_logits",
        ):
            if not isinstance(getattr(self, name), ProbeValue):
                raise TypeError(f"{name} prediction must be ProbeValue")


@runtime_checkable
class ComponentProbeLearner(Protocol):
    """Predict-only adapter over a reconstructable continual-learner state."""

    @property
    def name(self) -> str: ...

    def to_config(self) -> dict[str, object]: ...

    def training_step(self, state: Any) -> int: ...

    def state_to_config(self, state: Any) -> object: ...

    def state_from_config(self, payload: object) -> Any: ...

    def predict_components(
        self,
        state: Any,
        probe_input: ComponentProbeInput,
    ) -> LearnerComponentPredictions: ...


@dataclasses.dataclass(frozen=True)
class ComponentRetentionProtocol:
    """Evaluator-only regime identity and before-update checkpoint schedule."""

    protocol_id: str
    evaluator_regime_ids: tuple[str, ...]
    checkpoint_steps: tuple[int, ...]

    def __post_init__(self) -> None:
        _versioned_identifier(self.protocol_id, name="protocol_id")
        if (
            not self.evaluator_regime_ids
            or len(set(self.evaluator_regime_ids)) != len(self.evaluator_regime_ids)
            or any(not isinstance(value, str) or not value for value in self.evaluator_regime_ids)
        ):
            raise ValueError("evaluator_regime_ids must contain unique non-empty names")
        if not self.checkpoint_steps:
            raise ValueError("checkpoint_steps must be non-empty")
        for index, step in enumerate(self.checkpoint_steps):
            _positive_int(step, name=f"checkpoint_steps[{index}]")
        if any(
            later <= earlier
            for earlier, later in zip(
                self.checkpoint_steps,
                self.checkpoint_steps[1:],
                strict=False,
            )
        ):
            raise ValueError("checkpoint_steps must be strictly increasing")


@dataclasses.dataclass(frozen=True)
class ComponentRetentionBudget:
    """Hard lifetime and serialization bounds for the evaluator."""

    max_probe_calls: int
    max_stored_records: int
    max_learner_snapshot_bytes: int
    max_evaluator_state_bytes: int

    def __post_init__(self) -> None:
        _positive_int(self.max_probe_calls, name="max_probe_calls")
        _positive_int(self.max_stored_records, name="max_stored_records")
        _positive_int(self.max_learner_snapshot_bytes, name="max_learner_snapshot_bytes")
        _positive_int(self.max_evaluator_state_bytes, name="max_evaluator_state_bytes")


@dataclasses.dataclass(frozen=True)
class ComponentRetentionRecord:
    """One aggregate checkpoint/regime/component measurement."""

    checkpoint_step: int
    evaluator_regime_id: str
    component: ComponentName
    applicable: bool
    available: bool
    score: float | None
    probe_case_count: int
    scored_case_count: int
    unavailable_reason_counts: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        _positive_int(self.checkpoint_step, name="checkpoint_step")
        if not isinstance(self.evaluator_regime_id, str) or not self.evaluator_regime_id:
            raise ValueError("evaluator_regime_id must be non-empty")
        _component_name(self.component, name="component")
        if not isinstance(self.applicable, bool) or not isinstance(self.available, bool):
            raise ValueError("applicable and available must be boolean")
        case_count = _positive_int(self.probe_case_count, name="probe_case_count")
        scored_count = _nonnegative_int(self.scored_case_count, name="scored_case_count")
        if scored_count > case_count:
            raise ValueError("scored_case_count cannot exceed probe_case_count")
        previous_reason = ""
        reason_total = 0
        for reason, count in self.unavailable_reason_counts:
            if not isinstance(reason, str) or not reason:
                raise ValueError("unavailability reasons must be non-empty strings")
            if reason <= previous_reason:
                raise ValueError("unavailability reasons must be unique and sorted")
            previous_reason = reason
            reason_total += _positive_int(count, name="unavailability reason count")
        if self.available:
            if not self.applicable:
                raise ValueError("an inapplicable component cannot be available")
            _finite_float(self.score, name="score")
            if scored_count != case_count or self.unavailable_reason_counts:
                raise ValueError("available records must score every case without reasons")
        else:
            if self.score is not None:
                raise ValueError("unavailable records cannot have a score")
            if not self.unavailable_reason_counts or reason_total == 0:
                raise ValueError("unavailable records require reason counts")
            if scored_count + reason_total != case_count:
                raise ValueError("record outcomes must account for every probe case")
            if not self.applicable and scored_count != 0:
                raise ValueError("inapplicable records cannot score cases")


@dataclasses.dataclass(frozen=True)
class ComponentRetentionState:
    """Bounded checkpointable evaluator accumulator."""

    next_checkpoint_index: int
    probe_calls: int
    max_learner_snapshot_bytes_observed: int
    records: tuple[ComponentRetentionRecord, ...]


def _score_component(
    component: ComponentName,
    target: ProbeValue,
    predictions: LearnerComponentPredictions,
) -> tuple[float | None, str | None]:
    if component in (
        "representation",
        "dynamics_observation",
        "reward",
        "termination_discount",
        "critic_value",
    ):
        prediction = cast(ProbeValue, getattr(predictions, component))
        if not prediction.available:
            return None, f"learner_prediction_unavailable:{prediction.unavailable_reason}"
        if len(prediction.values) != len(target.values):
            return None, "prediction_shape_mismatch"
        if component == "termination_discount" and not 0.0 <= prediction.values[0] <= 1.0:
            return None, "prediction_out_of_range"
        squared_error = sum(
            (predicted - expected) ** 2
            for predicted, expected in zip(
                prediction.values,
                target.values,
                strict=True,
            )
        ) / len(target.values)
        return -squared_error, None

    logits = predictions.actor_logits
    if not logits.available:
        return None, f"learner_prediction_unavailable:{logits.unavailable_reason}"
    if len(logits.values) < 2:
        return None, "actor_logits_require_at_least_two_actions"
    if component == "actor_margin":
        preferred_action = int(target.values[0])
        if preferred_action >= len(logits.values):
            return None, "preferred_action_out_of_range"
        strongest_alternative = max(
            value for index, value in enumerate(logits.values) if index != preferred_action
        )
        return logits.values[preferred_action] - strongest_alternative, None
    if len(logits.values) != len(target.values):
        return None, "actor_return_action_count_mismatch"
    selected_action = max(range(len(logits.values)), key=logits.values.__getitem__)
    return target.values[selected_action], None


def _reason_counts(reasons: Sequence[str]) -> tuple[tuple[str, int], ...]:
    return tuple(sorted(Counter(reasons).items()))


def _record_payload(record: ComponentRetentionRecord) -> dict[str, object]:
    return {
        "checkpoint_step": record.checkpoint_step,
        "evaluator_regime_id": record.evaluator_regime_id,
        "component": record.component,
        "applicable": record.applicable,
        "available": record.available,
        "score": record.score,
        "probe_case_count": record.probe_case_count,
        "scored_case_count": record.scored_case_count,
        "unavailable_reason_counts": [
            {"reason": reason, "count": count} for reason, count in record.unavailable_reason_counts
        ],
    }


def _record_from_payload(payload: object, *, name: str) -> ComponentRetentionRecord:
    record = _mapping(payload, name=name)
    if set(record) != {
        "checkpoint_step",
        "evaluator_regime_id",
        "component",
        "applicable",
        "available",
        "score",
        "probe_case_count",
        "scored_case_count",
        "unavailable_reason_counts",
    }:
        raise ValueError(f"{name} keys are invalid")
    raw_reason_counts = _list(
        record["unavailable_reason_counts"],
        name=f"{name}.unavailable_reason_counts",
    )
    reason_counts: list[tuple[str, int]] = []
    for reason_index, raw_reason in enumerate(raw_reason_counts):
        reason_name = f"{name}.unavailable_reason_counts[{reason_index}]"
        reason = _mapping(raw_reason, name=reason_name)
        if set(reason) != {"reason", "count"}:
            raise ValueError(f"{reason_name} keys are invalid")
        reason_counts.append(
            (
                _nonempty_string(reason["reason"], name=f"{reason_name}.reason"),
                _positive_int(reason["count"], name=f"{reason_name}.count"),
            )
        )
    applicable = record["applicable"]
    available = record["available"]
    if not isinstance(applicable, bool) or not isinstance(available, bool):
        raise ValueError(f"{name} applicability fields must be boolean")
    raw_score = record["score"]
    return ComponentRetentionRecord(
        checkpoint_step=_positive_int(
            record["checkpoint_step"],
            name=f"{name}.checkpoint_step",
        ),
        evaluator_regime_id=_nonempty_string(
            record["evaluator_regime_id"],
            name=f"{name}.evaluator_regime_id",
        ),
        component=_component_name(record["component"], name=f"{name}.component"),
        applicable=applicable,
        available=available,
        score=(None if raw_score is None else _finite_float(raw_score, name=f"{name}.score")),
        probe_case_count=_positive_int(
            record["probe_case_count"],
            name=f"{name}.probe_case_count",
        ),
        scored_case_count=_nonnegative_int(
            record["scored_case_count"],
            name=f"{name}.scored_case_count",
        ),
        unavailable_reason_counts=tuple(reason_counts),
    )


def _retention_summaries(
    records: Sequence[ComponentRetentionRecord],
    regime_ids: Sequence[str],
) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for regime_id in regime_ids:
        for component in COMPONENT_NAMES:
            selected = tuple(
                record
                for record in records
                if record.evaluator_regime_id == regime_id and record.component == component
            )
            if not selected:
                raise ValueError("component retention records are incomplete")
            applicable = selected[0].applicable
            if any(record.applicable != applicable for record in selected):
                raise ValueError("component applicability changed across checkpoints")
            all_available = applicable and all(record.available for record in selected)
            if all_available:
                scores = [cast(float, record.score) for record in selected]
                peak = max(scores)
                current = scores[-1]
                summaries.append(
                    {
                        "evaluator_regime_id": regime_id,
                        "component": component,
                        "applicable": True,
                        "retention_available": True,
                        "unavailable_reason": None,
                        "first_score": scores[0],
                        "current_score": current,
                        "peak_score": peak,
                        "change_from_first": current - scores[0],
                        "peak_to_current_forgetting": max(0.0, peak - current),
                    }
                )
            else:
                reason = (
                    "component target is unavailable for this evaluator regime"
                    if not applicable
                    else "one or more scheduled checkpoint measurements are unavailable"
                )
                summaries.append(
                    {
                        "evaluator_regime_id": regime_id,
                        "component": component,
                        "applicable": applicable,
                        "retention_available": False,
                        "unavailable_reason": reason,
                        "first_score": None,
                        "current_score": None,
                        "peak_score": None,
                        "change_from_first": None,
                        "peak_to_current_forgetting": None,
                    }
                )
    return summaries


class ComponentRetentionEvaluator:
    """Measure component retention at strictly ordered before-update checkpoints."""

    def __init__(
        self,
        *,
        run_id: str,
        protocol: ComponentRetentionProtocol,
        held_out_probes: Mapping[str, Sequence[HeldOutComponentProbe]],
        learner: ComponentProbeLearner,
        budget: ComponentRetentionBudget,
    ) -> None:
        self._run_id = _versioned_identifier(run_id, name="run_id")
        self._protocol = protocol
        self._held_out_probes = {
            regime_id: tuple(probes) for regime_id, probes in held_out_probes.items()
        }
        self._learner = learner
        self._budget = budget
        if not isinstance(learner, ComponentProbeLearner):
            raise TypeError("learner must implement ComponentProbeLearner")
        learner_name = learner.name
        if not isinstance(learner_name, str) or not learner_name:
            raise ValueError("learner.name must be a non-empty string")
        self._learner_name = learner_name
        raw_learner_config = _json_clone(learner.to_config())
        if not isinstance(raw_learner_config, dict):
            raise ValueError("learner config must be a JSON object")
        _versioned_identifier(
            raw_learner_config.get("schema_version"),
            name="learner.schema_version",
        )
        self._learner_config = raw_learner_config
        self._validate_static_contract()

    @property
    def learner_name(self) -> str:
        return self._learner_name

    def _validate_static_contract(self) -> None:
        if set(self._held_out_probes) != set(self._protocol.evaluator_regime_ids):
            raise ValueError("held_out_probes keys must exactly match evaluator_regime_ids")
        if any(not probes for probes in self._held_out_probes.values()):
            raise ValueError("every evaluator regime requires at least one held-out probe")
        calls_per_checkpoint = sum(len(probes) for probes in self._held_out_probes.values())
        maximum_calls = calls_per_checkpoint * len(self._protocol.checkpoint_steps)
        maximum_records = (
            len(self._protocol.checkpoint_steps)
            * len(self._protocol.evaluator_regime_ids)
            * len(COMPONENT_NAMES)
        )
        if maximum_calls > self._budget.max_probe_calls:
            raise ValueError("held-out probes exceed max_probe_calls")
        if maximum_records > self._budget.max_stored_records:
            raise ValueError("component records exceed max_stored_records")

        for regime_id in self._protocol.evaluator_regime_ids:
            probes = self._held_out_probes[regime_id]
            for index, probe in enumerate(probes):
                if not isinstance(probe, HeldOutComponentProbe):
                    raise TypeError(f"held_out_probes.{regime_id}[{index}] has invalid type")
            for component in COMPONENT_NAMES:
                availability = {
                    probe.targets.for_component(component).available for probe in probes
                }
                if len(availability) != 1:
                    raise ValueError(
                        f"{regime_id}.{component} target applicability must be uniform"
                    )

    def init(self) -> ComponentRetentionState:
        state = ComponentRetentionState(
            next_checkpoint_index=0,
            probe_calls=0,
            max_learner_snapshot_bytes_observed=0,
            records=(),
        )
        self._validate_state(state)
        return state

    def probe_before_update(
        self,
        state: ComponentRetentionState,
        *,
        completed_observations: int,
        learner_state: Any,
    ) -> ComponentRetentionState:
        """Probe the next scheduled frozen snapshot before any subsequent update.

        The adapter-reported training step must equal the scheduled number of
        completed observations.  Targets never cross the adapter boundary, and
        each prediction uses a separately reconstructed state snapshot.
        """
        self._validate_state(state)
        if state.next_checkpoint_index >= len(self._protocol.checkpoint_steps):
            raise ValueError("all component-retention checkpoints are already complete")
        expected_step = self._protocol.checkpoint_steps[state.next_checkpoint_index]
        supplied_step = _positive_int(completed_observations, name="completed_observations")
        if supplied_step != expected_step:
            raise ValueError("completed_observations does not match the next checkpoint")
        learner_step = _nonnegative_int(
            self._learner.training_step(learner_state),
            name="learner.training_step",
        )
        if learner_step != expected_step:
            raise ValueError("learner state is not from the scheduled before-update checkpoint")

        snapshot_payload = _json_clone(self._learner.state_to_config(learner_state))
        snapshot_bytes = _serialized_size(snapshot_payload)
        if snapshot_bytes > self._budget.max_learner_snapshot_bytes:
            raise ValueError("learner snapshot exceeds max_learner_snapshot_bytes")
        if _json_clone(self._learner.state_to_config(learner_state)) != snapshot_payload:
            raise ValueError("learner state serialization is nondeterministic")

        new_records: list[ComponentRetentionRecord] = []
        calls = 0
        for regime_id in self._protocol.evaluator_regime_ids:
            probes = self._held_out_probes[regime_id]
            scores: dict[ComponentName, list[float]] = {
                component: [] for component in COMPONENT_NAMES
            }
            reasons: dict[ComponentName, list[str]] = {
                component: [] for component in COMPONENT_NAMES
            }
            applicable = {
                component: probes[0].targets.for_component(component).available
                for component in COMPONENT_NAMES
            }
            for probe in probes:
                reconstructed = self._learner.state_from_config(_json_clone(snapshot_payload))
                reconstructed_before = _json_clone(self._learner.state_to_config(reconstructed))
                if reconstructed_before != snapshot_payload:
                    raise ValueError("learner snapshot reconstruction is not canonical")
                predictions = self._learner.predict_components(
                    reconstructed,
                    probe.probe_input,
                )
                calls += 1
                if not isinstance(predictions, LearnerComponentPredictions):
                    raise TypeError("predict_components must return LearnerComponentPredictions")
                reconstructed_after = _json_clone(self._learner.state_to_config(reconstructed))
                if reconstructed_after != reconstructed_before:
                    raise ValueError("component probe mutated the reconstructed learner state")
                live_after = _json_clone(self._learner.state_to_config(learner_state))
                if live_after != snapshot_payload:
                    raise ValueError("component probe mutated the live learner state")
                for component in COMPONENT_NAMES:
                    target = probe.targets.for_component(component)
                    if not applicable[component]:
                        unavailable_reason = cast(str, target.unavailable_reason)
                        reasons[component].append(f"target_unavailable:{unavailable_reason}")
                        continue
                    score, score_reason = _score_component(component, target, predictions)
                    if score_reason is None:
                        scores[component].append(cast(float, score))
                    else:
                        reasons[component].append(score_reason)

            for component in COMPONENT_NAMES:
                component_scores = scores[component]
                component_reasons = reasons[component]
                fully_available = (
                    applicable[component]
                    and len(component_scores) == len(probes)
                    and not component_reasons
                )
                new_records.append(
                    ComponentRetentionRecord(
                        checkpoint_step=expected_step,
                        evaluator_regime_id=regime_id,
                        component=component,
                        applicable=applicable[component],
                        available=fully_available,
                        score=(
                            sum(component_scores) / len(component_scores)
                            if fully_available
                            else None
                        ),
                        probe_case_count=len(probes),
                        scored_case_count=len(component_scores),
                        unavailable_reason_counts=_reason_counts(component_reasons),
                    )
                )

        next_state = ComponentRetentionState(
            next_checkpoint_index=state.next_checkpoint_index + 1,
            probe_calls=state.probe_calls + calls,
            max_learner_snapshot_bytes_observed=max(
                state.max_learner_snapshot_bytes_observed,
                snapshot_bytes,
            ),
            records=(*state.records, *new_records),
        )
        self._validate_state(next_state)
        return next_state

    def _expected_record_keys(
        self,
        completed_checkpoints: int,
    ) -> tuple[tuple[int, str, ComponentName], ...]:
        return tuple(
            (step, regime_id, component)
            for step in self._protocol.checkpoint_steps[:completed_checkpoints]
            for regime_id in self._protocol.evaluator_regime_ids
            for component in COMPONENT_NAMES
        )

    def _state_payload(self, state: ComponentRetentionState) -> dict[str, object]:
        return {
            "next_checkpoint_index": state.next_checkpoint_index,
            "probe_calls": state.probe_calls,
            "max_learner_snapshot_bytes_observed": (state.max_learner_snapshot_bytes_observed),
            "records": [_record_payload(record) for record in state.records],
        }

    def _validate_state(self, state: ComponentRetentionState) -> None:
        if not isinstance(state, ComponentRetentionState):
            raise TypeError("state must be ComponentRetentionState")
        checkpoint_index = _nonnegative_int(
            state.next_checkpoint_index,
            name="state.next_checkpoint_index",
        )
        if checkpoint_index > len(self._protocol.checkpoint_steps):
            raise ValueError("state.next_checkpoint_index exceeds the protocol")
        calls_per_checkpoint = sum(len(probes) for probes in self._held_out_probes.values())
        expected_calls = checkpoint_index * calls_per_checkpoint
        if state.probe_calls != expected_calls:
            raise ValueError("state.probe_calls is inconsistent with completed checkpoints")
        if state.probe_calls > self._budget.max_probe_calls:
            raise ValueError("state exceeds max_probe_calls")
        observed_snapshot = _nonnegative_int(
            state.max_learner_snapshot_bytes_observed,
            name="state.max_learner_snapshot_bytes_observed",
        )
        if observed_snapshot > self._budget.max_learner_snapshot_bytes:
            raise ValueError("state snapshot high-water mark exceeds its bound")
        for record in state.records:
            if not isinstance(record, ComponentRetentionRecord):
                raise TypeError("state records must be ComponentRetentionRecord")
        expected_keys = self._expected_record_keys(checkpoint_index)
        actual_keys = tuple(
            (record.checkpoint_step, record.evaluator_regime_id, record.component)
            for record in state.records
        )
        if actual_keys != expected_keys:
            raise ValueError("state records do not match canonical checkpoint order")
        if len(state.records) > self._budget.max_stored_records:
            raise ValueError("state exceeds max_stored_records")
        payload_size = _serialized_size(self._state_payload(state))
        if payload_size > self._budget.max_evaluator_state_bytes:
            raise ValueError("state exceeds max_evaluator_state_bytes")

    def to_config(self) -> dict[str, object]:
        probe_payload = {
            regime_id: [probe.to_config() for probe in self._held_out_probes[regime_id]]
            for regime_id in self._protocol.evaluator_regime_ids
        }
        return {
            "type": "ComponentRetentionEvaluator",
            "schema_version": COMPONENT_RETENTION_EVALUATOR_SCHEMA,
            "report_schema_version": COMPONENT_RETENTION_REPORT_SCHEMA,
            "run_id": self._run_id,
            "protocol": _json_clone(dataclasses.asdict(self._protocol)),
            "budget": dataclasses.asdict(self._budget),
            "learner": {
                "name": self._learner_name,
                "config": _json_clone(self._learner_config),
            },
            "held_out_probe_sha256": _digest(probe_payload),
            "probe_cases_per_checkpoint": sum(
                len(probes) for probes in self._held_out_probes.values()
            ),
            "component_names": list(COMPONENT_NAMES),
            "measurement_order": (
                "reconstruct frozen learner state -> predict all components -> "
                "verify nonmutation -> evaluator score; no probe update"
            ),
            "development_only": True,
            "accepted_scientific_evidence": False,
        }

    def save_checkpoint(self, state: ComponentRetentionState, path: str | Path) -> None:
        """Save strict digested JSON bound to evaluator, learner, and probes."""
        self._validate_state(state)
        config = self.to_config()
        state_payload = self._state_payload(state)
        payload = {
            "schema_version": COMPONENT_RETENTION_CHECKPOINT_SCHEMA,
            "evaluator_config": config,
            "evaluator_config_sha256": _digest(config),
            "state": state_payload,
            "state_sha256": _digest(state_payload),
        }
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(
                payload,
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    def load_checkpoint(self, path: str | Path) -> ComponentRetentionState:
        """Load a strict checkpoint and reject schema, digest, or config drift."""
        payload = json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonstandard_json_constant,
        )
        mapping = _mapping(payload, name="checkpoint")
        if set(mapping) != {
            "schema_version",
            "evaluator_config",
            "evaluator_config_sha256",
            "state",
            "state_sha256",
        }:
            raise ValueError("component-retention checkpoint keys are invalid")
        if mapping["schema_version"] != COMPONENT_RETENTION_CHECKPOINT_SCHEMA:
            raise ValueError("component-retention checkpoint schema is invalid")
        config = mapping["evaluator_config"]
        if config != self.to_config() or mapping["evaluator_config_sha256"] != _digest(config):
            raise ValueError("component-retention checkpoint config does not match")
        raw_state = mapping["state"]
        if mapping["state_sha256"] != _digest(raw_state):
            raise ValueError("component-retention checkpoint state digest does not match")
        restored = self._state_from_payload(raw_state)
        self._validate_state(restored)
        if self._state_payload(restored) != raw_state:
            raise ValueError("component-retention checkpoint state is not canonical")
        return restored

    def _state_from_payload(self, payload: object) -> ComponentRetentionState:
        mapping = _mapping(payload, name="state")
        if set(mapping) != {
            "next_checkpoint_index",
            "probe_calls",
            "max_learner_snapshot_bytes_observed",
            "records",
        }:
            raise ValueError("component-retention state keys are invalid")
        raw_records = _list(mapping["records"], name="state.records")
        records = [
            _record_from_payload(raw_record, name=f"state.records[{index}]")
            for index, raw_record in enumerate(raw_records)
        ]
        return ComponentRetentionState(
            next_checkpoint_index=_nonnegative_int(
                mapping["next_checkpoint_index"],
                name="state.next_checkpoint_index",
            ),
            probe_calls=_nonnegative_int(mapping["probe_calls"], name="state.probe_calls"),
            max_learner_snapshot_bytes_observed=_nonnegative_int(
                mapping["max_learner_snapshot_bytes_observed"],
                name="state.max_learner_snapshot_bytes_observed",
            ),
            records=tuple(records),
        )

    def build_report(self, state: ComponentRetentionState) -> dict[str, object]:
        """Build the descriptive v1 report after every configured checkpoint."""
        self._validate_state(state)
        if state.next_checkpoint_index != len(self._protocol.checkpoint_steps):
            raise ValueError("cannot report before every component-retention checkpoint")
        summaries = _retention_summaries(
            state.records,
            self._protocol.evaluator_regime_ids,
        )

        report: dict[str, object] = {
            "schema_version": COMPONENT_RETENTION_REPORT_SCHEMA,
            "run_id": self._run_id,
            "status": "not-assessed",
            "interpretation": COMPONENT_RETENTION_INTERPRETATION,
            "accepted_scientific_evidence": False,
            "evaluator_config": self.to_config(),
            "evaluator_config_sha256": _digest(self.to_config()),
            "metric_direction": "higher_is_better",
            "metric_definitions": dict(COMPONENT_METRIC_DEFINITIONS),
            "records": self._state_payload(state)["records"],
            "retention_summaries": summaries,
            "resources": {
                "probe_calls": state.probe_calls,
                "probe_call_limit": self._budget.max_probe_calls,
                "stored_records": len(state.records),
                "stored_record_limit": self._budget.max_stored_records,
                "max_learner_snapshot_bytes_observed": (state.max_learner_snapshot_bytes_observed),
                "learner_snapshot_byte_limit": self._budget.max_learner_snapshot_bytes,
                "evaluator_state_bytes": _serialized_size(self._state_payload(state)),
                "evaluator_state_byte_limit": self._budget.max_evaluator_state_bytes,
                "raw_per_case_traces_stored": 0,
            },
            "known_limitations": list(COMPONENT_RETENTION_LIMITATIONS),
        }
        validation = validate_component_retention_report(report)
        if not validation.valid:
            raise ValueError("internal component-retention report validation failed")
        return report


@dataclasses.dataclass(frozen=True)
class ComponentRetentionReportValidation:
    valid: bool
    errors: tuple[str, ...]


def validate_component_retention_report(
    report: object,
) -> ComponentRetentionReportValidation:
    """Strictly validate a v1 report without upgrading its status."""
    errors: list[str] = []
    try:
        payload = _mapping(report, name="report")
        expected_keys = {
            "schema_version",
            "run_id",
            "status",
            "interpretation",
            "accepted_scientific_evidence",
            "evaluator_config",
            "evaluator_config_sha256",
            "metric_direction",
            "metric_definitions",
            "records",
            "retention_summaries",
            "resources",
            "known_limitations",
        }
        if set(payload) != expected_keys:
            raise ValueError("report keys are invalid")
        if payload["schema_version"] != COMPONENT_RETENTION_REPORT_SCHEMA:
            raise ValueError("report schema is invalid")
        _versioned_identifier(payload["run_id"], name="run_id")
        if payload["status"] != "not-assessed":
            raise ValueError("report status must remain not-assessed")
        if payload["interpretation"] != COMPONENT_RETENTION_INTERPRETATION:
            raise ValueError("report interpretation is invalid")
        if payload["accepted_scientific_evidence"] is not False:
            raise ValueError("report cannot assert scientific acceptance")
        config = _mapping(payload["evaluator_config"], name="evaluator_config")
        if payload["evaluator_config_sha256"] != _digest(config):
            raise ValueError("evaluator config digest is invalid")
        if set(config) != {
            "type",
            "schema_version",
            "report_schema_version",
            "run_id",
            "protocol",
            "budget",
            "learner",
            "held_out_probe_sha256",
            "probe_cases_per_checkpoint",
            "component_names",
            "measurement_order",
            "development_only",
            "accepted_scientific_evidence",
        }:
            raise ValueError("evaluator config keys are invalid")
        if (
            config["type"] != "ComponentRetentionEvaluator"
            or config["schema_version"] != COMPONENT_RETENTION_EVALUATOR_SCHEMA
            or config["report_schema_version"] != COMPONENT_RETENTION_REPORT_SCHEMA
            or config["run_id"] != payload["run_id"]
        ):
            raise ValueError("evaluator config identity is invalid")
        if config["development_only"] is not True:
            raise ValueError("evaluator config must remain development-only")
        if config["accepted_scientific_evidence"] is not False:
            raise ValueError("evaluator config cannot assert scientific acceptance")
        if config["component_names"] != list(COMPONENT_NAMES):
            raise ValueError("evaluator component names are invalid")
        if config["measurement_order"] != (
            "reconstruct frozen learner state -> predict all components -> "
            "verify nonmutation -> evaluator score; no probe update"
        ):
            raise ValueError("evaluator measurement order is invalid")
        probe_digest = config["held_out_probe_sha256"]
        if (
            not isinstance(probe_digest, str)
            or len(probe_digest) != 64
            or any(character not in "0123456789abcdef" for character in probe_digest)
        ):
            raise ValueError("held-out probe digest is invalid")
        learner = _mapping(config["learner"], name="evaluator_config.learner")
        if set(learner) != {"name", "config"}:
            raise ValueError("learner config envelope keys are invalid")
        _nonempty_string(learner["name"], name="learner.name")
        learner_config = _mapping(learner["config"], name="learner.config")
        _versioned_identifier(
            learner_config.get("schema_version"),
            name="learner.schema_version",
        )
        if payload["metric_direction"] != "higher_is_better":
            raise ValueError("metric direction is invalid")
        definitions = _mapping(payload["metric_definitions"], name="metric_definitions")
        if definitions != COMPONENT_METRIC_DEFINITIONS:
            raise ValueError("metric definitions are invalid")
        protocol = _mapping(config.get("protocol"), name="evaluator_config.protocol")
        if set(protocol) != {"protocol_id", "evaluator_regime_ids", "checkpoint_steps"}:
            raise ValueError("evaluator protocol keys are invalid")
        _versioned_identifier(protocol["protocol_id"], name="protocol_id")
        regime_ids = _string_list(
            protocol.get("evaluator_regime_ids"),
            name="evaluator_regime_ids",
        )
        if len(set(regime_ids)) != len(regime_ids):
            raise ValueError("evaluator regime IDs must be unique")
        checkpoint_steps = _positive_int_list(
            protocol.get("checkpoint_steps"),
            name="checkpoint_steps",
        )
        if any(
            later <= earlier
            for earlier, later in zip(checkpoint_steps, checkpoint_steps[1:], strict=False)
        ):
            raise ValueError("checkpoint steps must be strictly increasing")
        budget = _mapping(config["budget"], name="evaluator_config.budget")
        if set(budget) != {
            "max_probe_calls",
            "max_stored_records",
            "max_learner_snapshot_bytes",
            "max_evaluator_state_bytes",
        }:
            raise ValueError("evaluator budget keys are invalid")
        budget_values = {
            key: _positive_int(value, name=f"budget.{key}") for key, value in budget.items()
        }
        cases_per_checkpoint = _positive_int(
            config["probe_cases_per_checkpoint"],
            name="probe_cases_per_checkpoint",
        )
        expected_calls = cases_per_checkpoint * len(checkpoint_steps)
        if expected_calls > budget_values["max_probe_calls"]:
            raise ValueError("configured probe calls exceed their budget")
        raw_records = _list(payload["records"], name="records")
        expected_record_count = len(regime_ids) * len(checkpoint_steps) * len(COMPONENT_NAMES)
        if len(raw_records) != expected_record_count:
            raise ValueError("report record count is invalid")
        if expected_record_count > budget_values["max_stored_records"]:
            raise ValueError("configured records exceed their budget")
        records = tuple(
            _record_from_payload(raw_record, name=f"records[{index}]")
            for index, raw_record in enumerate(raw_records)
        )
        if [_record_payload(record) for record in records] != raw_records:
            raise ValueError("report records are not canonical")
        expected_record_keys = tuple(
            (step, regime_id, component)
            for step in checkpoint_steps
            for regime_id in regime_ids
            for component in COMPONENT_NAMES
        )
        actual_record_keys = tuple(
            (record.checkpoint_step, record.evaluator_regime_id, record.component)
            for record in records
        )
        if actual_record_keys != expected_record_keys:
            raise ValueError("report records are not in canonical checkpoint order")
        first_step = checkpoint_steps[0]
        first_component = COMPONENT_NAMES[0]
        regime_case_counts = [
            next(
                record.probe_case_count
                for record in records
                if record.checkpoint_step == first_step
                and record.evaluator_regime_id == regime_id
                and record.component == first_component
            )
            for regime_id in regime_ids
        ]
        if sum(regime_case_counts) != cases_per_checkpoint:
            raise ValueError("record probe-case counts do not match evaluator config")
        for record in records:
            regime_index = regime_ids.index(record.evaluator_regime_id)
            if record.probe_case_count != regime_case_counts[regime_index]:
                raise ValueError("record probe-case count changed across components")
        raw_summaries = _list(
            payload["retention_summaries"],
            name="retention_summaries",
        )
        if raw_summaries != _retention_summaries(records, regime_ids):
            raise ValueError("retention summaries do not reconstruct from records")
        resources = _mapping(payload["resources"], name="resources")
        if set(resources) != {
            "probe_calls",
            "probe_call_limit",
            "stored_records",
            "stored_record_limit",
            "max_learner_snapshot_bytes_observed",
            "learner_snapshot_byte_limit",
            "evaluator_state_bytes",
            "evaluator_state_byte_limit",
            "raw_per_case_traces_stored",
        }:
            raise ValueError("resource keys are invalid")
        for key, value in resources.items():
            _nonnegative_int(value, name=f"resources.{key}")
        if resources["probe_calls"] != expected_calls:
            raise ValueError("reported probe calls do not reconstruct")
        if resources["probe_call_limit"] != budget_values["max_probe_calls"]:
            raise ValueError("reported probe-call limit does not match config")
        if resources["stored_records"] != len(records):
            raise ValueError("reported stored records do not reconstruct")
        if resources["stored_record_limit"] != budget_values["max_stored_records"]:
            raise ValueError("reported stored-record limit does not match config")
        if resources["learner_snapshot_byte_limit"] != budget_values["max_learner_snapshot_bytes"]:
            raise ValueError("reported learner snapshot limit does not match config")
        if (
            cast(int, resources["max_learner_snapshot_bytes_observed"])
            > budget_values["max_learner_snapshot_bytes"]
        ):
            raise ValueError("snapshot resource bound is exceeded")
        if resources["evaluator_state_byte_limit"] != budget_values["max_evaluator_state_bytes"]:
            raise ValueError("reported evaluator-state limit does not match config")
        reconstructed_state_payload = {
            "next_checkpoint_index": len(checkpoint_steps),
            "probe_calls": expected_calls,
            "max_learner_snapshot_bytes_observed": resources["max_learner_snapshot_bytes_observed"],
            "records": [_record_payload(record) for record in records],
        }
        if resources["evaluator_state_bytes"] != _serialized_size(reconstructed_state_payload):
            raise ValueError("reported evaluator-state bytes do not reconstruct")
        if resources["evaluator_state_bytes"] > budget_values["max_evaluator_state_bytes"]:
            raise ValueError("state resource bound is exceeded")
        if resources["raw_per_case_traces_stored"] != 0:
            raise ValueError("raw per-case traces must not be stored")
        limitations = _string_list(payload["known_limitations"], name="known_limitations")
        if tuple(limitations) != COMPONENT_RETENTION_LIMITATIONS:
            raise ValueError("known limitations are invalid")
    except (KeyError, TypeError, ValueError) as error:
        errors.append(str(error))
    return ComponentRetentionReportValidation(valid=not errors, errors=tuple(errors))


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be an object with string keys")
    return value


def _list(value: object, *, name: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be an array")
    return value


def _string_list(value: object, *, name: str) -> list[str]:
    raw = _list(value, name=name)
    if any(not isinstance(item, str) or not item for item in raw):
        raise ValueError(f"{name} must contain non-empty strings")
    return cast(list[str], raw)


def _positive_int_list(value: object, *, name: str) -> list[int]:
    raw = _list(value, name=name)
    return [_positive_int(item, name=f"{name}[{index}]") for index, item in enumerate(raw)]


def _nonempty_string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
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


__all__ = [
    "COMPONENT_METRIC_DEFINITIONS",
    "COMPONENT_NAMES",
    "COMPONENT_RETENTION_CHECKPOINT_SCHEMA",
    "COMPONENT_RETENTION_EVALUATOR_SCHEMA",
    "COMPONENT_RETENTION_INTERPRETATION",
    "COMPONENT_RETENTION_LIMITATIONS",
    "COMPONENT_RETENTION_REPORT_SCHEMA",
    "ComponentName",
    "ComponentProbeInput",
    "ComponentProbeLearner",
    "ComponentProbeTargets",
    "ComponentRetentionBudget",
    "ComponentRetentionEvaluator",
    "ComponentRetentionProtocol",
    "ComponentRetentionRecord",
    "ComponentRetentionReportValidation",
    "ComponentRetentionState",
    "HeldOutComponentProbe",
    "LearnerComponentPredictions",
    "ProbeValue",
    "validate_component_retention_report",
]
