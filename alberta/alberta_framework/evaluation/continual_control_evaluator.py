"""Fail-closed continuing-control evaluator and PrototypeAgent adapter.

The evaluator owns regime scheduling and the environment boundary.  Learners
see only observations and owned transitions; evaluator regime identifiers are
never arguments to a learner method.  Every online action is selected before
the corresponding environment outcome, and every environment transition must
copy the exact observation, action, and four-word decision identifier that was
dispatched.

This module is a bounded development mechanism.  Its reports are always
``not-assessed`` and explicitly distinguish a common resource ceiling from
realized compute or memory parity.  It does not create scientific evidence.
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
from typing import Any, NoReturn, Protocol, TypeGuard, cast, runtime_checkable

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from alberta_framework.core.prototype_agent import (
    PrototypeAgent,
    PrototypeAgentState,
    PrototypeTransition,
)

CONTROL_REPORT_SCHEMA = "alberta.continual_control_report.v2"
CONTROL_CHECKPOINT_SCHEMA = "alberta.continual_control_evaluator_checkpoint.v2"
CONTROL_EVALUATOR_SCHEMA = "alberta.continual_control_evaluator.v2"
CONTROL_PROTOCOL_SCHEMA = "alberta.continual_control_protocol.v2"
ACCEPTANCE_STATUS = "not-assessed"
REPORT_INTERPRETATION = (
    "Development-only continuing-control trace. It is not scientific acceptance, "
    "does not establish matched realized compute or memory, and is not an Alberta "
    "Plan completion certificate."
)
CONTROL_REPORT_LIMITATIONS = (
    "development-only mechanism; no preregistered or untouched held-out seeds",
    "no fresh-per-regime, oracle-data, or stationary multitask control reference",
    "shared ceilings do not establish equal realized compute or memory",
    "energy and accelerator allocator residency are unavailable",
    "longitudinal metrics are descriptive; no stratified bootstrap or paired-seed inference",
    "held-out probes score one action, not a frozen-policy control rollout",
    "plasticity, component survival, and world-model calibration are not emitted",
)
# Metric names follow the continual-learning transfer literature: forgetting is
# the best-to-final degradation measure of Chaudhry et al. (2018, "Riemannian
# Walk for Incremental Learning"), and backward/forward transfer adapt the
# BWT/FWT definitions of Lopez-Paz & Ranzato (2017, "Gradient Episodic Memory
# for Continual Learning") from per-task accuracies to held-out action-score
# probes over evaluator regimes.  Forward transfer is measured against the
# protocol's frozen per-regime reference rather than GEM's random-init
# baseline.  "Direction-normalized" means the sign convention flips with the
# protocol's ``higher_is_better`` flag so that larger is always better.
CONTROL_METRIC_DEFINITIONS: Mapping[str, str] = {
    "prequential_return": (
        "mean environment return from actions selected before their corresponding outcomes"
    ),
    "lifetime_return": (
        "mean return over the complete fail-closed transition schedule; no dropped transitions "
        "are imputed"
    ),
    "per_regime_online_mean_return": (
        "evaluator-stratified online mean return for each scheduled regime"
    ),
    "adaptation_auc": (
        "time-normalized trapezoid return AUC for every post-change segment; the initial "
        "segment is excluded"
    ),
    "recovery": (
        "first sustained post-change window meeting the frozen regime threshold, bounded by "
        "the next change"
    ),
    "per_regime_final_performance": (
        "final held-out non-learning action-score probe for each evaluator regime"
    ),
    "mean_final_performance": "mean final held-out action score across evaluator regimes",
    "forgetting": (
        "direction-normalized best-to-final held-out degradation after first exposure"
    ),
    "backward_transfer": (
        "direction-normalized final minus first-post-exposure held-out performance"
    ),
    "forward_transfer": (
        "direction-normalized last pre-exposure held-out performance minus frozen reference"
    ),
    "stability": "immediate post-change return gap from the frozen regime reference",
    "worst_window": "worst rolling mean return in the configured metric direction",
}

type DiagnosticScalar = bool | int | float | str
type DecisionId = tuple[int, int, int, int]


def _is_finite_number(value: object) -> TypeGuard[int | float]:
    return (
        not isinstance(value, bool)
        and isinstance(value, int | float)
        and math.isfinite(float(value))
    )


def _finite_float(value: object, *, name: str) -> float:
    if not _is_finite_number(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _nonnegative_float(value: object, *, name: str) -> float:
    resolved = _finite_float(value, name=name)
    if resolved < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return resolved


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


def _observation(value: object, *, name: str) -> tuple[float, ...]:
    if not isinstance(value, tuple) or not value:
        raise ValueError(f"{name} must be a non-empty immutable tuple")
    return tuple(_finite_float(item, name=f"{name}[{index}]") for index, item in enumerate(value))


def _decision_id(value: object, *, name: str) -> DecisionId:
    if not isinstance(value, tuple) or len(value) != 4:
        raise ValueError(f"{name} must contain exactly four uint32 words")
    words: list[int] = []
    for index, item in enumerate(value):
        word = _nonnegative_int(item, name=f"{name}[{index}]")
        if word > np.iinfo(np.uint32).max:
            raise ValueError(f"{name}[{index}] exceeds uint32")
        words.append(word)
    return cast(DecisionId, tuple(words))


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


def _serialized_size(value: object) -> int:
    return len(_canonical_json(value).encode("utf-8"))


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
class ControlDecision:
    """One armed action together with its exact transition ownership token."""

    observation: tuple[float, ...]
    action: int
    decision_id: DecisionId
    armed: bool = True

    def __post_init__(self) -> None:
        _observation(self.observation, name="decision.observation")
        _nonnegative_int(self.action, name="decision.action")
        _decision_id(self.decision_id, name="decision.decision_id")
        if not isinstance(self.armed, bool):
            raise ValueError("decision.armed must be boolean")

    def to_config(self) -> dict[str, object]:
        return {
            "observation": list(self.observation),
            "action": self.action,
            "decision_id": list(self.decision_id),
            "armed": self.armed,
        }


@dataclasses.dataclass(frozen=True)
class ControlTransition:
    """Learner-visible continuing transition with explicit boundary observations.

    ``bootstrap_observation`` is the final observation reached by the action.
    ``reset_observation`` is required exactly when ``terminated`` or
    ``truncated`` is true and is the observation used for the next decision.
    Away from a boundary, the next decision observation is exactly the
    bootstrap observation.  Evaluator regime identity is deliberately absent.
    """

    observation: tuple[float, ...]
    action: int
    decision_id: DecisionId
    reward: float
    discount: float
    terminated: bool
    truncated: bool
    bootstrap_observation: tuple[float, ...]
    reset_observation: tuple[float, ...] | None
    safety_violation: bool = False
    intervention: bool = False
    near_miss: bool = False
    safety_cost: float = 0.0
    near_miss_cost: float = 0.0

    def __post_init__(self) -> None:
        source = _observation(self.observation, name="transition.observation")
        bootstrap = _observation(
            self.bootstrap_observation,
            name="transition.bootstrap_observation",
        )
        if len(source) != len(bootstrap):
            raise ValueError("transition observation dimensions must match")
        _nonnegative_int(self.action, name="transition.action")
        _decision_id(self.decision_id, name="transition.decision_id")
        _finite_float(self.reward, name="transition.reward")
        discount = _finite_float(self.discount, name="transition.discount")
        if not 0.0 <= discount <= 1.0:
            raise ValueError("transition.discount must lie in [0, 1]")
        for name in (
            "terminated",
            "truncated",
            "safety_violation",
            "intervention",
            "near_miss",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"transition.{name} must be boolean")
        boundary = self.terminated or self.truncated
        if (discount == 0.0) != self.terminated:
            raise ValueError("zero discount must exactly identify termination")
        if self.truncated and not self.terminated and discount <= 0.0:
            raise ValueError("nonterminal truncation requires a positive bootstrap discount")
        if boundary:
            if self.reset_observation is None:
                raise ValueError("a boundary requires an explicit reset_observation")
            reset = _observation(
                self.reset_observation,
                name="transition.reset_observation",
            )
            if len(reset) != len(source):
                raise ValueError("transition reset observation dimension must match")
        elif self.reset_observation is not None:
            raise ValueError("reset_observation must be None away from a boundary")
        _nonnegative_float(self.safety_cost, name="transition.safety_cost")
        near_miss_cost = _nonnegative_float(
            self.near_miss_cost,
            name="transition.near_miss_cost",
        )
        if near_miss_cost > self.safety_cost:
            raise ValueError("near_miss_cost cannot exceed total safety_cost")
        if near_miss_cost > 0.0 and not self.near_miss:
            raise ValueError("nonzero near_miss_cost requires near_miss=True")

    @property
    def next_decision_observation(self) -> tuple[float, ...]:
        if self.reset_observation is None:
            return self.bootstrap_observation
        return self.reset_observation


@dataclasses.dataclass(frozen=True)
class ControlEnvironmentUpdate:
    """Functional environment result for one dispatched decision."""

    state: Any
    transition: ControlTransition


@dataclasses.dataclass(frozen=True)
class ControlLearnerUpdate:
    """Functional learner result for one owned environment transition.

    ``backward_call_count`` is ``None`` when the adapter cannot instrument the
    learner's internal differentiation calls.  The evaluator reports that
    absence instead of silently converting it to zero.
    """

    state: Any
    applied: bool
    backward_call_count: int | None

    def __post_init__(self) -> None:
        if not isinstance(self.applied, bool):
            raise ValueError("learner update.applied must be boolean")
        if self.backward_call_count is not None:
            _nonnegative_int(self.backward_call_count, name="backward_call_count")


@dataclasses.dataclass(frozen=True)
class ControlProbe:
    """Evaluator-owned held-out observation and score for every discrete action."""

    observation: tuple[float, ...]
    action_scores: tuple[float, ...]

    def __post_init__(self) -> None:
        _observation(self.observation, name="probe.observation")
        if not isinstance(self.action_scores, tuple) or not self.action_scores:
            raise ValueError("probe.action_scores must be a non-empty immutable tuple")
        for index, value in enumerate(self.action_scores):
            _finite_float(value, name=f"probe.action_scores[{index}]")

    def to_config(self) -> dict[str, object]:
        return {
            "observation": list(self.observation),
            "action_scores": list(self.action_scores),
        }


@dataclasses.dataclass(frozen=True)
class ControlResourceUsage:
    """Exact logical learner-state accounting, with optional parameter count."""

    persistent_state_bytes: int
    state_scalar_count: int
    trainable_parameter_count: int | None
    measurement_method: str

    def __post_init__(self) -> None:
        _positive_int(self.persistent_state_bytes, name="persistent_state_bytes")
        _positive_int(self.state_scalar_count, name="state_scalar_count")
        if self.trainable_parameter_count is not None:
            _nonnegative_int(
                self.trainable_parameter_count,
                name="trainable_parameter_count",
            )
        if not isinstance(self.measurement_method, str) or not self.measurement_method:
            raise ValueError("measurement_method must be a non-empty string")


@runtime_checkable
class ContinuingControlLearner(Protocol):
    """Learner-neutral control state machine with no regime-ID arguments."""

    @property
    def name(self) -> str: ...

    @property
    def n_actions(self) -> int: ...

    @property
    def max_backward_calls_per_update(self) -> int | None: ...

    def to_config(self) -> dict[str, object]: ...

    def init(self, initial_observation: tuple[float, ...]) -> Any: ...

    def state_valid_for_observation(
        self,
        state: Any,
        observation: tuple[float, ...],
    ) -> bool: ...

    def decide(
        self,
        state: Any,
        observation: tuple[float, ...],
    ) -> ControlDecision: ...

    def update(
        self,
        state: Any,
        transition: ControlTransition,
    ) -> ControlLearnerUpdate: ...

    def probe_action(
        self,
        state: Any,
        observation: tuple[float, ...],
    ) -> int: ...

    def state_to_config(self, state: Any) -> object: ...

    def state_from_config(self, payload: object) -> Any: ...

    def resource_usage(self, state: Any) -> ControlResourceUsage: ...

    def diagnostics(
        self,
        state: Any,
    ) -> Mapping[str, DiagnosticScalar] | None: ...


@runtime_checkable
class ContinuingControlEnvironment(Protocol):
    """Evaluator-side functional environment; regime identity stays here."""

    @property
    def n_actions(self) -> int: ...

    def to_config(self) -> dict[str, object]: ...

    def init(self) -> Any: ...

    def observation(self, state: Any) -> tuple[float, ...]: ...

    def step(
        self,
        state: Any,
        decision: ControlDecision,
        evaluator_regime_id: str,
    ) -> ControlEnvironmentUpdate: ...

    def state_to_config(self, state: Any) -> object: ...

    def state_from_config(self, payload: object) -> Any: ...


@dataclasses.dataclass(frozen=True)
class ContinuingControlProtocol:
    """Evaluator-only regime schedule and bounded probe checkpoints."""

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
    operation_latency_deadline_ms: float
    schema_version: str = CONTROL_PROTOCOL_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != CONTROL_PROTOCOL_SCHEMA:
            raise ValueError("continuing-control protocol schema_version is invalid")
        _versioned_identifier(self.protocol_id, name="protocol_id")
        if not isinstance(self.higher_is_better, bool):
            raise ValueError("higher_is_better must be boolean")
        if len(self.regime_schedule) < 2:
            raise ValueError("regime_schedule must contain at least two transitions")
        if any(not isinstance(value, str) or not value for value in self.regime_schedule):
            raise ValueError("regime_schedule IDs must be non-empty strings")
        if (
            len(self.evaluator_regime_ids) < 2
            or len(set(self.evaluator_regime_ids)) != len(self.evaluator_regime_ids)
            or any(not isinstance(value, str) or not value for value in self.evaluator_regime_ids)
        ):
            raise ValueError("evaluator_regime_ids must contain at least two unique names")
        if set(self.regime_schedule) != set(self.evaluator_regime_ids):
            raise ValueError("regime schedule IDs must exactly match evaluator_regime_ids")
        if all(
            left == right
            for left, right in zip(self.regime_schedule, self.regime_schedule[1:], strict=False)
        ):
            raise ValueError("regime_schedule must contain at least one change")
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
        if self.checkpoint_steps[-1] != len(self.regime_schedule):
            raise ValueError("the final checkpoint must equal the transition count")
        expected_regimes = set(self.evaluator_regime_ids)
        if set(self.first_exposure_checkpoint) != expected_regimes:
            raise ValueError(
                "first_exposure_checkpoint keys must exactly match evaluator_regime_ids"
            )
        for regime_id, row in self.first_exposure_checkpoint.items():
            if (
                isinstance(row, bool)
                or not isinstance(row, int)
                or not 0 <= row < len(self.checkpoint_steps)
            ):
                raise ValueError(
                    f"first_exposure_checkpoint.{regime_id} must index checkpoints"
                )
            first_transition = self.regime_schedule.index(regime_id) + 1
            expected_row = next(
                index
                for index, checkpoint in enumerate(self.checkpoint_steps)
                if checkpoint >= first_transition
            )
            if row != expected_row:
                raise ValueError(
                    f"first_exposure_checkpoint.{regime_id} must be {expected_row}, "
                    "the first checkpoint at or after first exposure"
                )
        for field_name in (
            "forward_transfer_reference",
            "recovery_thresholds",
            "stability_references",
        ):
            values = getattr(self, field_name)
            if set(values) != expected_regimes:
                raise ValueError(
                    f"{field_name} keys must exactly match evaluator_regime_ids"
                )
            for regime_id, value in values.items():
                _finite_float(value, name=f"{field_name}.{regime_id}")
        _positive_int(self.recovery_window, name="recovery_window")
        window = _positive_int(self.worst_window_size, name="worst_window_size")
        if window > len(self.regime_schedule):
            raise ValueError("worst_window_size cannot exceed transition count")
        deadline = _finite_float(
            self.operation_latency_deadline_ms,
            name="operation_latency_deadline_ms",
        )
        if deadline <= 0.0:
            raise ValueError("operation_latency_deadline_ms must be positive")


@dataclasses.dataclass(frozen=True)
class ContinuingControlBudget:
    """Common hard ceilings; these do not assert equal realized utilization."""

    transition_limit: int
    decision_call_limit: int
    environment_call_limit: int
    update_call_limit: int
    probe_call_limit: int
    backward_call_limit: int
    persistent_state_bytes_limit: int
    state_scalar_count_limit: int
    trainable_parameter_count_limit: int | None
    stored_decision_id_limit: int

    def __post_init__(self) -> None:
        for name in (
            "transition_limit",
            "decision_call_limit",
            "environment_call_limit",
            "update_call_limit",
            "probe_call_limit",
            "persistent_state_bytes_limit",
            "state_scalar_count_limit",
            "stored_decision_id_limit",
        ):
            _positive_int(getattr(self, name), name=name)
        _nonnegative_int(self.backward_call_limit, name="backward_call_limit")
        if self.trainable_parameter_count_limit is not None:
            _nonnegative_int(
                self.trainable_parameter_count_limit,
                name="trainable_parameter_count_limit",
            )


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot summarize an empty metric vector")
    return float(sum(values) / len(values))


def _control_segments(schedule: tuple[str, ...]) -> tuple[tuple[str, int, int], ...]:
    segments: list[tuple[str, int, int]] = []
    start = 0
    for index in range(1, len(schedule)):
        if schedule[index] != schedule[start]:
            segments.append((schedule[start], start, index))
            start = index
    segments.append((schedule[start], start, len(schedule)))
    return tuple(segments)


def _normalized_trapezoid_auc(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("adaptation segment must be non-empty")
    if len(values) == 1:
        return float(values[0])
    area = sum(
        0.5 * (left + right)
        for left, right in zip(values, values[1:], strict=False)
    )
    return float(area / (len(values) - 1))


def _signed_improvement(
    later: float,
    earlier: float,
    *,
    higher_is_better: bool,
) -> float:
    return later - earlier if higher_is_better else earlier - later


def _availability_record(
    *,
    evidence_source: str,
    available: Sequence[str],
    unavailable: Mapping[str, str],
) -> dict[str, object]:
    return {
        "applicable": bool(available),
        "unavailable_reason": (
            None if available else "no evaluator regime has sufficient longitudinal probes"
        ),
        "evidence_source": evidence_source,
        "available_regimes": list(available),
        "unavailable_regimes": dict(unavailable),
    }


def _control_metric_applicability(
    protocol: ContinuingControlProtocol,
) -> dict[str, object]:
    online = {
        "applicable": True,
        "unavailable_reason": None,
        "evidence_source": "decision_before_environment_return_trace",
    }
    held_out = {
        "applicable": True,
        "unavailable_reason": None,
        "evidence_source": "held_out_non_learning_action_score_probes",
    }
    post_change_segments = _control_segments(protocol.regime_schedule)[1:]
    assessable_recovery = [
        str(index)
        for index, (_, start, stop) in enumerate(post_change_segments, start=1)
        if stop - start >= protocol.recovery_window
    ]
    unavailable_recovery = {
        str(index): "segment is shorter than recovery_window"
        for index, (_, start, stop) in enumerate(post_change_segments, start=1)
        if stop - start < protocol.recovery_window
    }

    forgetting_available: list[str] = []
    forgetting_unavailable: dict[str, str] = {}
    backward_available: list[str] = []
    backward_unavailable: dict[str, str] = {}
    forward_available: list[str] = []
    forward_unavailable: dict[str, str] = {}
    final_row = len(protocol.checkpoint_steps) - 1
    for regime_id in protocol.evaluator_regime_ids:
        first_row = int(protocol.first_exposure_checkpoint[regime_id])
        if final_row > first_row:
            forgetting_available.append(regime_id)
            backward_available.append(regime_id)
        else:
            reason = "only one held-out checkpoint exists at or after first exposure"
            forgetting_unavailable[regime_id] = reason
            backward_unavailable[regime_id] = reason
        if first_row > 0:
            forward_available.append(regime_id)
        else:
            forward_unavailable[regime_id] = (
                "no pre-exposure held-out checkpoint exists for this regime"
            )

    return {
        "prequential_return": dict(online),
        "lifetime_return": dict(online),
        "per_regime_online_mean_return": dict(online),
        "adaptation_auc": dict(online),
        "recovery": {
            "applicable": bool(assessable_recovery),
            "unavailable_reason": (
                None
                if assessable_recovery
                else "no post-change segment is at least recovery_window long"
            ),
            "evidence_source": "decision_before_environment_return_trace",
            "available_events": assessable_recovery,
            "unavailable_events": unavailable_recovery,
        },
        "per_regime_final_performance": dict(held_out),
        "mean_final_performance": dict(held_out),
        "forgetting": _availability_record(
            evidence_source="held_out_non_learning_action_score_probes",
            available=forgetting_available,
            unavailable=forgetting_unavailable,
        ),
        "backward_transfer": _availability_record(
            evidence_source="held_out_non_learning_action_score_probes",
            available=backward_available,
            unavailable=backward_unavailable,
        ),
        "forward_transfer": _availability_record(
            evidence_source="held_out_non_learning_action_score_probes",
            available=forward_available,
            unavailable=forward_unavailable,
        ),
        "stability": dict(online),
        "worst_window": dict(online),
    }


def _control_metrics(
    rewards: tuple[float, ...],
    evaluator_matrix: tuple[tuple[float, ...], ...],
    protocol: ContinuingControlProtocol,
) -> dict[str, object]:
    if len(rewards) != len(protocol.regime_schedule):
        raise ValueError("control metric reward length must match protocol")
    if len(evaluator_matrix) != len(protocol.checkpoint_steps):
        raise ValueError("control metric probe rows must match protocol checkpoints")

    segments = _control_segments(protocol.regime_schedule)
    adaptation_records: list[dict[str, object]] = []
    adaptation_values: list[float] = []
    stability_records: list[dict[str, object]] = []
    stability_gaps: list[float] = []
    recovery_events: list[dict[str, object]] = []
    recovered_steps: list[int] = []
    assessable_recovery_count = 0
    recovered_count = 0
    for segment_index, (regime_id, start, stop) in enumerate(segments[1:], start=1):
        auc = _normalized_trapezoid_auc(rewards[start:stop])
        adaptation_values.append(auc)
        adaptation_records.append(
            {
                "segment_index": segment_index,
                "regime_id": regime_id,
                "start_step": start,
                "end_step_exclusive": stop,
                "normalized_auc": auc,
            }
        )
        reference = float(protocol.stability_references[regime_id])
        immediate_return = rewards[start]
        gap = max(
            0.0,
            (
                reference - immediate_return
                if protocol.higher_is_better
                else immediate_return - reference
            ),
        )
        stability_gaps.append(gap)
        stability_records.append(
            {
                "segment_index": segment_index,
                "regime_id": regime_id,
                "step": start,
                "reference": reference,
                "observed_return": immediate_return,
                "gap": gap,
            }
        )

        threshold = float(protocol.recovery_thresholds[regime_id])
        assessable = stop - start >= protocol.recovery_window
        recovery_steps: int | None = None
        recovered: bool | None = None
        unavailable_reason: str | None = None
        if assessable:
            assessable_recovery_count += 1
            recovered = False
            last_window_start = stop - protocol.recovery_window
            for window_start in range(start, last_window_start + 1):
                recovery_values = rewards[
                    window_start : window_start + protocol.recovery_window
                ]
                meets = (
                    all(value >= threshold for value in recovery_values)
                    if protocol.higher_is_better
                    else all(value <= threshold for value in recovery_values)
                )
                if meets:
                    recovered = True
                    recovery_steps = (
                        window_start + protocol.recovery_window - start
                    )
                    recovered_steps.append(recovery_steps)
                    recovered_count += 1
                    break
        else:
            unavailable_reason = "segment is shorter than recovery_window"
        recovery_events.append(
            {
                "segment_index": segment_index,
                "regime_id": regime_id,
                "start_step": start,
                "end_step_exclusive": stop,
                "threshold": threshold,
                "window_size": protocol.recovery_window,
                "assessable": assessable,
                "unavailable_reason": unavailable_reason,
                "recovered": recovered,
                "recovery_steps": recovery_steps,
            }
        )

    per_regime_online = {
        regime_id: _mean(
            [
                reward
                for reward, scheduled in zip(
                    rewards,
                    protocol.regime_schedule,
                    strict=True,
                )
                if scheduled == regime_id
            ]
        )
        for regime_id in protocol.evaluator_regime_ids
    }
    per_regime_final: dict[str, float] = {}
    per_regime_forgetting: dict[str, float | None] = {}
    per_regime_backward: dict[str, float | None] = {}
    per_regime_forward: dict[str, float | None] = {}
    forgetting_values: list[float] = []
    backward_values: list[float] = []
    forward_values: list[float] = []
    final_row = len(evaluator_matrix) - 1
    for column, regime_id in enumerate(protocol.evaluator_regime_ids):
        first_row = int(protocol.first_exposure_checkpoint[regime_id])
        trajectory = [float(row[column]) for row in evaluator_matrix[first_row:]]
        final = trajectory[-1]
        per_regime_final[regime_id] = final
        if final_row > first_row:
            best = max(trajectory) if protocol.higher_is_better else min(trajectory)
            degradation = (
                best - final if protocol.higher_is_better else final - best
            )
            forgetting = max(0.0, degradation)
            backward = _signed_improvement(
                final,
                trajectory[0],
                higher_is_better=protocol.higher_is_better,
            )
            per_regime_forgetting[regime_id] = forgetting
            per_regime_backward[regime_id] = backward
            forgetting_values.append(forgetting)
            backward_values.append(backward)
        else:
            per_regime_forgetting[regime_id] = None
            per_regime_backward[regime_id] = None
        if first_row > 0:
            pre_exposure = float(evaluator_matrix[first_row - 1][column])
            forward = _signed_improvement(
                pre_exposure,
                float(protocol.forward_transfer_reference[regime_id]),
                higher_is_better=protocol.higher_is_better,
            )
            per_regime_forward[regime_id] = forward
            forward_values.append(forward)
        else:
            per_regime_forward[regime_id] = None

    worst_window_size = protocol.worst_window_size
    window_means = tuple(
        _mean(rewards[start : start + worst_window_size])
        for start in range(len(rewards) - worst_window_size + 1)
    )
    worst_return = (
        min(window_means) if protocol.higher_is_better else max(window_means)
    )
    worst_start = window_means.index(worst_return)
    mean_return = _mean(rewards)
    return {
        "prequential_return": mean_return,
        "lifetime_return": mean_return,
        "per_regime_online_mean_return": per_regime_online,
        "adaptation_auc": {
            "mean_normalized_auc": _mean(adaptation_values),
            "segments": adaptation_records,
        },
        "recovery": {
            "event_count": len(recovery_events),
            "assessable_event_count": assessable_recovery_count,
            "unavailable_event_count": (
                len(recovery_events) - assessable_recovery_count
            ),
            "recovered_count": recovered_count,
            "recovery_rate_over_assessable": (
                recovered_count / assessable_recovery_count
                if assessable_recovery_count
                else None
            ),
            "mean_recovery_steps_over_recovered": (
                _mean([float(value) for value in recovered_steps])
                if recovered_steps
                else None
            ),
            "maximum_recovery_steps": (
                max(recovered_steps) if recovered_steps else None
            ),
            "events": recovery_events,
        },
        "per_regime_final_performance": per_regime_final,
        "mean_final_performance": _mean(list(per_regime_final.values())),
        "forgetting": {
            "mean_over_available": (
                _mean(forgetting_values) if forgetting_values else None
            ),
            "maximum_over_available": (
                max(forgetting_values) if forgetting_values else None
            ),
            "available_regime_count": len(forgetting_values),
            "per_regime": per_regime_forgetting,
        },
        "backward_transfer": {
            "mean_over_available": (
                _mean(backward_values) if backward_values else None
            ),
            "available_regime_count": len(backward_values),
            "per_regime": per_regime_backward,
        },
        "forward_transfer": {
            "mean_over_available": (
                _mean(forward_values) if forward_values else None
            ),
            "available_regime_count": len(forward_values),
            "per_regime": per_regime_forward,
        },
        "stability": {
            "mean_gap": _mean(stability_gaps),
            "maximum_gap": max(stability_gaps),
            "events": stability_records,
        },
        "worst_window": {
            "window_size": worst_window_size,
            "mean_return": worst_return,
            "start_step": worst_start,
            "end_step_exclusive": worst_start + worst_window_size,
        },
        "metric_applicability": _control_metric_applicability(protocol),
    }


@dataclasses.dataclass(frozen=True)
class _FrozenActionState:
    observation: tuple[float, ...]
    generation: int


def _generation_decision_id(lifecycle_id: tuple[int, int], generation: int) -> DecisionId:
    """Pack two lifecycle words plus a 64-bit generation counter split across two uint32 words."""
    if generation >= 1 << 64:
        raise ValueError("decision generation is exhausted")
    return (
        lifecycle_id[0],
        lifecycle_id[1],
        (generation >> 32) & np.iinfo(np.uint32).max,
        generation & np.iinfo(np.uint32).max,
    )


def _lifecycle_id(value: tuple[int, int], *, name: str) -> tuple[int, int]:
    resolved = _decision_id((*value, 0, 0), name=name)
    return resolved[:2]


class FrozenActionControlBaseline:
    """Fixed-action non-learning baseline with exact decision ownership."""

    def __init__(
        self,
        *,
        n_actions: int,
        action: int = 0,
        name: str = "frozen_action",
        # Word 0 is ASCII "FROZ": a human-readable tag that identifies this
        # baseline's decision stream in traces and checkpoints.
        lifecycle_id: tuple[int, int] = (0x46524F5A, 1),
    ):
        self._n_actions = _positive_int(n_actions, name="n_actions")
        self._action = _nonnegative_int(action, name="action")
        if self._action >= self._n_actions:
            raise ValueError("action must be below n_actions")
        if not isinstance(name, str) or not name:
            raise ValueError("name must be a non-empty string")
        self._name = name
        self._lifecycle_id = _lifecycle_id(lifecycle_id, name="lifecycle_id")

    @property
    def name(self) -> str:
        return self._name

    @property
    def n_actions(self) -> int:
        return self._n_actions

    @property
    def max_backward_calls_per_update(self) -> int:
        return 0

    def to_config(self) -> dict[str, object]:
        return {
            "type": "FrozenActionControlBaseline",
            "schema_version": "alberta.frozen_action_control_baseline.v1",
            "name": self._name,
            "n_actions": self._n_actions,
            "action": self._action,
            "lifecycle_id": list(self._lifecycle_id),
        }

    def init(self, initial_observation: tuple[float, ...]) -> _FrozenActionState:
        return _FrozenActionState(_observation(initial_observation, name="initial_observation"), 0)

    def state_valid_for_observation(
        self,
        state: Any,
        observation: tuple[float, ...],
    ) -> bool:
        resolved = cast(_FrozenActionState, state)
        return resolved.observation == observation and 0 <= resolved.generation < 1 << 64

    def decide(
        self,
        state: Any,
        observation: tuple[float, ...],
    ) -> ControlDecision:
        resolved = cast(_FrozenActionState, state)
        if not self.state_valid_for_observation(resolved, observation):
            raise ValueError("observation does not match frozen baseline state")
        return ControlDecision(
            observation=observation,
            action=self._action,
            decision_id=_generation_decision_id(self._lifecycle_id, resolved.generation),
        )

    def update(
        self,
        state: Any,
        transition: ControlTransition,
    ) -> ControlLearnerUpdate:
        resolved = cast(_FrozenActionState, state)
        expected = self.decide(resolved, resolved.observation)
        valid = (
            transition.observation == expected.observation
            and transition.action == expected.action
            and transition.decision_id == expected.decision_id
            and resolved.generation + 1 < 1 << 64
        )
        if not valid:
            return ControlLearnerUpdate(resolved, False, 0)
        return ControlLearnerUpdate(
            _FrozenActionState(
                transition.next_decision_observation,
                resolved.generation + 1,
            ),
            True,
            0,
        )

    def probe_action(self, state: Any, observation: tuple[float, ...]) -> int:
        del state
        _observation(observation, name="probe observation")
        return self._action

    def state_to_config(self, state: Any) -> object:
        resolved = cast(_FrozenActionState, state)
        return {
            "observation": list(_observation(resolved.observation, name="state.observation")),
            "generation": _nonnegative_int(resolved.generation, name="state.generation"),
        }

    def state_from_config(self, payload: object) -> _FrozenActionState:
        mapping = _mapping(payload, name="frozen baseline state")
        if set(mapping) != {"observation", "generation"}:
            raise ValueError("frozen baseline state keys are invalid")
        return _FrozenActionState(
            _observation_from_json(mapping["observation"], name="state.observation"),
            _nonnegative_int(mapping["generation"], name="state.generation"),
        )

    def resource_usage(self, state: Any) -> ControlResourceUsage:
        resolved = cast(_FrozenActionState, state)
        self.state_to_config(resolved)
        scalar_count = len(resolved.observation) + 1
        return ControlResourceUsage(
            persistent_state_bytes=8 * scalar_count,
            state_scalar_count=scalar_count,
            trainable_parameter_count=0,
            measurement_method="exact logical float64/uint64 state accounting",
        )

    def diagnostics(self, state: Any) -> Mapping[str, DiagnosticScalar]:
        resolved = cast(_FrozenActionState, state)
        return {
            "fixed_action": self._action,
            "generation": resolved.generation,
            "learning_enabled": False,
        }


@dataclasses.dataclass(frozen=True)
class _RewardBanditState:
    observation: tuple[float, ...]
    generation: int
    reward_sums: tuple[float, ...]
    action_counts: tuple[int, ...]


class RunningRewardBanditControlBaseline:
    """Observation-agnostic causal sample-mean action-value baseline."""

    def __init__(
        self,
        *,
        n_actions: int,
        name: str = "running_reward_bandit",
        # Word 0 is ASCII "BAND": a human-readable tag that identifies this
        # baseline's decision stream in traces and checkpoints.
        lifecycle_id: tuple[int, int] = (0x42414E44, 1),
    ):
        self._n_actions = _positive_int(n_actions, name="n_actions")
        if not isinstance(name, str) or not name:
            raise ValueError("name must be a non-empty string")
        self._name = name
        self._lifecycle_id = _lifecycle_id(lifecycle_id, name="lifecycle_id")

    @property
    def name(self) -> str:
        return self._name

    @property
    def n_actions(self) -> int:
        return self._n_actions

    @property
    def max_backward_calls_per_update(self) -> int:
        return 0

    def to_config(self) -> dict[str, object]:
        return {
            "type": "RunningRewardBanditControlBaseline",
            "schema_version": "alberta.running_reward_bandit_control_baseline.v1",
            "name": self._name,
            "n_actions": self._n_actions,
            "lifecycle_id": list(self._lifecycle_id),
        }

    def init(self, initial_observation: tuple[float, ...]) -> _RewardBanditState:
        return _RewardBanditState(
            observation=_observation(initial_observation, name="initial_observation"),
            generation=0,
            reward_sums=(0.0,) * self._n_actions,
            action_counts=(0,) * self._n_actions,
        )

    def _action(self, state: _RewardBanditState) -> int:
        values = tuple(
            total / count if count else 0.0
            for total, count in zip(state.reward_sums, state.action_counts, strict=True)
        )
        return max(range(self._n_actions), key=lambda index: (values[index], -index))

    def state_valid_for_observation(
        self,
        state: Any,
        observation: tuple[float, ...],
    ) -> bool:
        resolved = cast(_RewardBanditState, state)
        return (
            resolved.observation == observation
            and 0 <= resolved.generation < 1 << 64
            and len(resolved.reward_sums) == self._n_actions
            and len(resolved.action_counts) == self._n_actions
            and all(math.isfinite(value) for value in resolved.reward_sums)
            and all(value >= 0 for value in resolved.action_counts)
        )

    def decide(
        self,
        state: Any,
        observation: tuple[float, ...],
    ) -> ControlDecision:
        resolved = cast(_RewardBanditState, state)
        if not self.state_valid_for_observation(resolved, observation):
            raise ValueError("observation does not match reward-bandit state")
        return ControlDecision(
            observation=observation,
            action=self._action(resolved),
            decision_id=_generation_decision_id(self._lifecycle_id, resolved.generation),
        )

    def update(
        self,
        state: Any,
        transition: ControlTransition,
    ) -> ControlLearnerUpdate:
        resolved = cast(_RewardBanditState, state)
        expected = self.decide(resolved, resolved.observation)
        valid = (
            transition.observation == expected.observation
            and transition.action == expected.action
            and transition.decision_id == expected.decision_id
            and resolved.generation + 1 < 1 << 64
        )
        if not valid:
            return ControlLearnerUpdate(resolved, False, 0)
        sums = list(resolved.reward_sums)
        counts = list(resolved.action_counts)
        sums[transition.action] += transition.reward
        counts[transition.action] += 1
        if not math.isfinite(sums[transition.action]):
            return ControlLearnerUpdate(resolved, False, 0)
        return ControlLearnerUpdate(
            _RewardBanditState(
                observation=transition.next_decision_observation,
                generation=resolved.generation + 1,
                reward_sums=tuple(sums),
                action_counts=tuple(counts),
            ),
            True,
            0,
        )

    def probe_action(self, state: Any, observation: tuple[float, ...]) -> int:
        _observation(observation, name="probe observation")
        return self._action(cast(_RewardBanditState, state))

    def state_to_config(self, state: Any) -> object:
        resolved = cast(_RewardBanditState, state)
        if not self.state_valid_for_observation(resolved, resolved.observation):
            raise ValueError("reward-bandit state is invalid")
        return {
            "observation": list(resolved.observation),
            "generation": resolved.generation,
            "reward_sums": list(resolved.reward_sums),
            "action_counts": list(resolved.action_counts),
        }

    def state_from_config(self, payload: object) -> _RewardBanditState:
        mapping = _mapping(payload, name="reward-bandit state")
        if set(mapping) != {"observation", "generation", "reward_sums", "action_counts"}:
            raise ValueError("reward-bandit state keys are invalid")
        reward_values = _list(mapping["reward_sums"], name="reward_sums")
        count_values = _list(mapping["action_counts"], name="action_counts")
        if len(reward_values) != self._n_actions or len(count_values) != self._n_actions:
            raise ValueError("reward-bandit action-state width is invalid")
        state = _RewardBanditState(
            observation=_observation_from_json(mapping["observation"], name="observation"),
            generation=_nonnegative_int(mapping["generation"], name="generation"),
            reward_sums=tuple(
                _finite_float(value, name=f"reward_sums[{index}]")
                for index, value in enumerate(reward_values)
            ),
            action_counts=tuple(
                _nonnegative_int(value, name=f"action_counts[{index}]")
                for index, value in enumerate(count_values)
            ),
        )
        if not self.state_valid_for_observation(state, state.observation):
            raise ValueError("reward-bandit state is invalid")
        return state

    def resource_usage(self, state: Any) -> ControlResourceUsage:
        resolved = cast(_RewardBanditState, state)
        self.state_to_config(resolved)
        scalar_count = len(resolved.observation) + 1 + 2 * self._n_actions
        return ControlResourceUsage(
            persistent_state_bytes=8 * scalar_count,
            state_scalar_count=scalar_count,
            trainable_parameter_count=self._n_actions,
            measurement_method="exact logical float64/int64 state accounting",
        )

    def diagnostics(self, state: Any) -> Mapping[str, DiagnosticScalar]:
        resolved = cast(_RewardBanditState, state)
        return {
            "generation": resolved.generation,
            "total_updates": sum(resolved.action_counts),
            "learning_enabled": True,
        }


class PrototypeAgentControlAdapter:
    """Opt-in learner-neutral adapter over PrototypeAgent's owned transition API.

    JAX array payload bytes are exact, but allocator residency and the number
    of trainable parameters are deliberately reported unavailable.  The
    current Prototype stack does not expose separately instrumented backward
    calls, so ``backward_call_count`` is also unavailable rather than zero.
    """

    def __init__(
        self,
        agent: PrototypeAgent,
        *,
        seed: int,
        lifecycle_id: tuple[int, int],
        name: str = "prototype_agent",
    ):
        if not isinstance(agent, PrototypeAgent):
            raise TypeError("agent must be a PrototypeAgent")
        self._agent = agent
        self._seed = _nonnegative_int(seed, name="seed")
        if self._seed > np.iinfo(np.uint32).max:
            raise ValueError("seed must fit in uint32")
        self._lifecycle_id = _lifecycle_id(lifecycle_id, name="lifecycle_id")
        if not isinstance(name, str) or not name:
            raise ValueError("name must be a non-empty string")
        self._name = name
        unstarted = self._agent.init(
            jr.key(self._seed),
            lifecycle_id=jnp.asarray(self._lifecycle_id, dtype=jnp.uint32),
        )
        self._observation_dim = int(unstarted.current_raw_observation.shape[0])
        self._template = self._agent.start(
            unstarted,
            jnp.zeros((self._observation_dim,), dtype=jnp.float32),
        )
        if not bool(self._template.started):
            raise ValueError("PrototypeAgent template could not be started")
        self._treedef = jax.tree_util.tree_structure(self._template)
        self._config = _json_clone(self._agent.to_config())

    @property
    def name(self) -> str:
        return self._name

    @property
    def n_actions(self) -> int:
        return self._agent.config.oak.n_primitive_actions

    @property
    def max_backward_calls_per_update(self) -> None:
        return None

    def to_config(self) -> dict[str, object]:
        if _json_clone(self._agent.to_config()) != self._config:
            raise ValueError("underlying PrototypeAgent config changed after adapter creation")
        return {
            "type": "PrototypeAgentControlAdapter",
            "schema_version": "alberta.prototype_agent_control_adapter.v1",
            "name": self._name,
            "seed": self._seed,
            "lifecycle_id": list(self._lifecycle_id),
            "agent_config": _json_clone(self._config),
            "backward_call_count_available": False,
            "trainable_parameter_count_available": False,
        }

    def _jax_observation(self, observation: tuple[float, ...]) -> jax.Array:
        resolved = _observation(observation, name="observation")
        if len(resolved) != self._observation_dim:
            raise ValueError("observation dimension does not match PrototypeAgent")
        return jnp.asarray(resolved, dtype=jnp.float32)

    def init(self, initial_observation: tuple[float, ...]) -> PrototypeAgentState:
        unstarted = self._agent.init(
            jr.key(self._seed),
            lifecycle_id=jnp.asarray(self._lifecycle_id, dtype=jnp.uint32),
        )
        state = self._agent.start(unstarted, self._jax_observation(initial_observation))
        if not self.state_valid_for_observation(state, initial_observation):
            raise ValueError("PrototypeAgent initialization produced an invalid decision state")
        return state

    def state_valid_for_observation(
        self,
        state: Any,
        observation: tuple[float, ...],
    ) -> bool:
        if not isinstance(state, PrototypeAgentState):
            return False
        try:
            expected = self._jax_observation(observation)
            # Use PrototypeAgent's public structural + numeric-leaf boundary;
            # "valid" here is exactly the invariant the agent enforces on its
            # own checkpoints.
            return bool(
                self._agent.validate_state(state)
                & jnp.array_equal(state.current_raw_observation, expected)
            )
        except (TypeError, ValueError):
            return False

    def decide(
        self,
        state: Any,
        observation: tuple[float, ...],
    ) -> ControlDecision:
        resolved = cast(PrototypeAgentState, state)
        if not self.state_valid_for_observation(resolved, observation):
            raise ValueError("observation does not match PrototypeAgent decision state")
        decision = self._agent.decision(resolved)
        if not bool(decision.armed):
            raise ValueError("PrototypeAgent decision is unarmed")
        return ControlDecision(
            observation=tuple(float(value) for value in np.asarray(decision.observation)),
            action=int(decision.action),
            decision_id=cast(
                DecisionId,
                tuple(int(value) for value in np.asarray(decision.decision_id)),
            ),
            armed=True,
        )

    def update(
        self,
        state: Any,
        transition: ControlTransition,
    ) -> ControlLearnerUpdate:
        resolved = cast(PrototypeAgentState, state)
        prototype_transition_type = cast(Any, PrototypeTransition)
        result = self._agent.update_transition(
            resolved,
            prototype_transition_type(
                observation=self._jax_observation(transition.observation),
                action=jnp.asarray(transition.action, dtype=jnp.int32),
                decision_id=jnp.asarray(transition.decision_id, dtype=jnp.uint32),
                reward=jnp.asarray(transition.reward, dtype=jnp.float32),
                discount=jnp.asarray(transition.discount, dtype=jnp.float32),
                terminated=jnp.asarray(transition.terminated, dtype=jnp.bool_),
                truncated=jnp.asarray(transition.truncated, dtype=jnp.bool_),
                next_observation=self._jax_observation(
                    transition.bootstrap_observation
                ),
                next_decision_observation=self._jax_observation(
                    transition.next_decision_observation
                ),
            ),
        )
        applied = bool(result.transition_diagnostics.valid)
        return ControlLearnerUpdate(
            state=result.state if applied else resolved,
            applied=applied,
            backward_call_count=None,
        )

    def probe_action(self, state: Any, observation: tuple[float, ...]) -> int:
        return int(
            self._agent.greedy_action(
                cast(PrototypeAgentState, state),
                self._jax_observation(observation),
            )
        )

    def _leaf_payload(self, leaf: object) -> dict[str, object]:
        if not hasattr(leaf, "dtype") or not hasattr(leaf, "shape"):
            raise ValueError("PrototypeAgent state contains a non-array leaf")
        array = cast(jax.Array, leaf)
        if jnp.issubdtype(array.dtype, jax.dtypes.prng_key):
            raw = np.asarray(jax.device_get(jr.key_data(array)))
            return {
                "kind": "typed_prng_key",
                "dtype": str(array.dtype),
                "shape": list(array.shape),
                "raw_shape": list(raw.shape),
                "data": raw.tolist(),
            }
        materialized = np.asarray(jax.device_get(array))
        return {
            "kind": "array",
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "data": materialized.tolist(),
        }

    def state_to_config(self, state: Any) -> object:
        if not isinstance(state, PrototypeAgentState):
            raise TypeError("state must be a PrototypeAgentState")
        leaves, treedef = jax.tree_util.tree_flatten(state)
        if treedef != self._treedef:  # type: ignore[operator]
            raise ValueError("PrototypeAgent state PyTree does not match adapter config")
        return {
            "schema_version": "alberta.prototype_agent_control_state.v1",
            "leaves": [self._leaf_payload(leaf) for leaf in leaves],
        }

    def state_from_config(self, payload: object) -> PrototypeAgentState:
        mapping = _mapping(payload, name="PrototypeAgent state")
        if set(mapping) != {"schema_version", "leaves"}:
            raise ValueError("PrototypeAgent state keys are invalid")
        if mapping["schema_version"] != "alberta.prototype_agent_control_state.v1":
            raise ValueError("PrototypeAgent control-state schema is invalid")
        raw_leaves = _list(mapping["leaves"], name="PrototypeAgent state leaves")
        template_leaves = jax.tree_util.tree_leaves(self._template)
        if len(raw_leaves) != len(template_leaves):
            raise ValueError("PrototypeAgent state leaf count is invalid")
        leaves: list[jax.Array] = []
        for index, (raw_leaf, template) in enumerate(
            zip(raw_leaves, template_leaves, strict=True)
        ):
            location = f"PrototypeAgent state leaves[{index}]"
            leaf = _mapping(raw_leaf, name=location)
            expected_kind = (
                "typed_prng_key"
                if jnp.issubdtype(template.dtype, jax.dtypes.prng_key)
                else "array"
            )
            expected_keys = (
                {"kind", "dtype", "shape", "raw_shape", "data"}
                if expected_kind == "typed_prng_key"
                else {"kind", "dtype", "shape", "data"}
            )
            if set(leaf) != expected_keys or leaf["kind"] != expected_kind:
                raise ValueError(f"{location} keys or kind are invalid")
            if leaf["dtype"] != str(template.dtype):
                raise ValueError(f"{location} dtype does not match template")
            shape = _shape_from_json(leaf["shape"], name=f"{location}.shape")
            if shape != tuple(template.shape):
                raise ValueError(f"{location} shape does not match template")
            if expected_kind == "typed_prng_key":
                template_raw = jr.key_data(template)
                raw_shape = _shape_from_json(
                    leaf["raw_shape"],
                    name=f"{location}.raw_shape",
                )
                if raw_shape != tuple(template_raw.shape):
                    raise ValueError(f"{location} raw key shape does not match template")
                raw_array = jnp.asarray(leaf["data"], dtype=template_raw.dtype)
                if tuple(raw_array.shape) != raw_shape:
                    raise ValueError(f"{location} key data shape is invalid")
                leaves.append(jr.wrap_key_data(raw_array, impl=jr.key_impl(template)))
            else:
                array = jnp.asarray(leaf["data"], dtype=template.dtype)
                if tuple(array.shape) != shape:
                    raise ValueError(f"{location} data shape is invalid")
                leaves.append(array)
        restored = cast(
            PrototypeAgentState,
            jax.tree_util.tree_unflatten(self._treedef, leaves),
        )
        if _canonical_json(self.state_to_config(restored)) != _canonical_json(mapping):
            raise ValueError("PrototypeAgent control state numeric payload is not canonical")
        if not bool(self._agent.validate_state(restored)):
            raise ValueError("reconstructed PrototypeAgent state is inconsistent")
        return restored

    def resource_usage(self, state: Any) -> ControlResourceUsage:
        resolved = cast(PrototypeAgentState, state)
        leaves = jax.tree_util.tree_leaves(resolved)
        byte_count = 0
        scalar_count = 0
        for leaf in leaves:
            array = jr.key_data(leaf) if jnp.issubdtype(
                leaf.dtype,
                jax.dtypes.prng_key,
            ) else leaf
            byte_count += int(array.size * array.dtype.itemsize)
            scalar_count += int(array.size)
        return ControlResourceUsage(
            persistent_state_bytes=byte_count,
            state_scalar_count=scalar_count,
            trainable_parameter_count=None,
            measurement_method=(
                "exact logical JAX array payload; allocator residency and trainable-parameter "
                "partition unavailable"
            ),
        )

    def diagnostics(self, state: Any) -> Mapping[str, DiagnosticScalar]:
        resolved = cast(PrototypeAgentState, state)
        return {
            "step_count": int(resolved.step_count),
            "observation_event_count": int(resolved.observation_event_count),
            "current_action": int(resolved.current_action),
            "decision_armed": bool(self._agent.decision(resolved).armed),
        }


@dataclasses.dataclass(frozen=True)
class _ControlConditionState:
    learner_state: Any
    environment_state: Any
    current_observation: tuple[float, ...]
    pending_decision: ControlDecision | None
    used_decision_ids: tuple[DecisionId, ...]
    rewards: tuple[float, ...]
    delayed: tuple[bool, ...]
    evaluator_matrix: tuple[tuple[float, ...], ...]
    decision_latencies_ns: tuple[int, ...]
    environment_latencies_ns: tuple[int, ...]
    update_latencies_ns: tuple[int, ...]
    probe_latencies_ns: tuple[int, ...]
    backward_call_counts: tuple[int | None, ...]
    persistent_state_bytes_high_water: int
    state_scalar_count_high_water: int
    safety_violations: int
    interventions: int
    near_misses: int
    cumulative_safety_cost: float
    cumulative_near_miss_cost: float
    maximum_step_safety_cost: float


@dataclasses.dataclass(frozen=True)
class ContinualControlRunState:
    """Checkpointable state after only complete environment transitions."""

    step: int
    conditions: tuple[_ControlConditionState, ...]


@dataclasses.dataclass(frozen=True)
class ControlReportValidation:
    """Strict report validation result; an invalid artifact has explicit errors."""

    valid: bool
    errors: tuple[str, ...]


class ContinualControlEvaluator:
    """Run one candidate and honest simple baselines on independent env copies."""

    def __init__(
        self,
        *,
        run_id: str,
        protocol: ContinuingControlProtocol,
        environment: ContinuingControlEnvironment,
        probes: Mapping[str, Sequence[ControlProbe]],
        candidate: ContinuingControlLearner,
        baselines: Sequence[ContinuingControlLearner],
        budget: ContinuingControlBudget,
        clock_ns: Callable[[], int] | None = None,
        latency_measurement_method: str = "time.perf_counter_ns wall clock",
    ):
        self._run_id = _versioned_identifier(run_id, name="run_id")
        if not isinstance(protocol, ContinuingControlProtocol):
            raise TypeError("protocol must be a ContinuingControlProtocol")
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
        if not isinstance(environment, ContinuingControlEnvironment):
            raise TypeError("environment must implement ContinuingControlEnvironment")
        self._environment = environment
        self._candidate = candidate
        self._baselines = tuple(baselines)
        self._learners = (candidate, *self._baselines)
        self._probes = {key: tuple(values) for key, values in probes.items()}
        self._budget = budget
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

    def _learner_index(self, learner: ContinuingControlLearner) -> int:
        for index, configured in enumerate(self._learners):
            if configured is learner:
                return index
        raise ValueError("learner is not configured for this evaluator")

    def _validate_static_contract(self) -> None:
        if len(self._baselines) < 2:
            raise ValueError("at least two explicit control baselines are required")
        if any(not isinstance(learner, ContinuingControlLearner) for learner in self._learners):
            raise TypeError("candidate and baselines must implement ContinuingControlLearner")
        names = self.condition_names
        if (
            any(not isinstance(name, str) or not name for name in names)
            or len(set(names)) != len(names)
        ):
            raise ValueError("condition names must be unique non-empty strings")
        n_actions = _positive_int(self._environment.n_actions, name="environment.n_actions")
        if any(learner.n_actions != n_actions for learner in self._learners):
            raise ValueError("every learner action count must match the environment")
        if set(self._probes) != set(self._protocol.evaluator_regime_ids):
            raise ValueError("probe keys must exactly match evaluator_regime_ids")
        if any(not values for values in self._probes.values()):
            raise ValueError("every evaluator regime requires held-out control probes")
        for regime_id, probes in self._probes.items():
            for index, probe in enumerate(probes):
                if not isinstance(probe, ControlProbe):
                    raise TypeError(f"probes.{regime_id}[{index}] must be a ControlProbe")
                if len(probe.action_scores) != n_actions:
                    raise ValueError("probe action-score width must match environment actions")
        transition_count = len(self._protocol.regime_schedule)
        if self._budget.transition_limit != transition_count:
            raise ValueError("transition_limit must exactly match the scheduled transition count")
        if self._budget.decision_call_limit < transition_count:
            raise ValueError("decision-call budget is too small")
        if self._budget.environment_call_limit < transition_count:
            raise ValueError("environment-call budget is too small")
        if self._budget.update_call_limit < transition_count:
            raise ValueError("update-call budget is too small")
        probe_count = sum(len(values) for values in self._probes.values())
        maximum_probe_calls = len(self._protocol.checkpoint_steps) * probe_count
        if self._budget.probe_call_limit < maximum_probe_calls:
            raise ValueError("probe-call budget is too small")
        if self._budget.stored_decision_id_limit < transition_count:
            raise ValueError("stored-decision-ID budget is too small")

        environment_config = _json_clone(self._environment.to_config())
        if not isinstance(environment_config, dict):
            raise ValueError("environment config must be a JSON object")
        _versioned_identifier(
            environment_config.get("schema_version"),
            name="environment.schema_version",
        )
        initial_environment = self._environment.init()
        initial_environment_payload = self._environment_state_payload(initial_environment)
        if self._environment_state_payload(self._environment.init()) != initial_environment_payload:
            raise ValueError("environment init must be deterministic from config")
        if _json_clone(self._environment.to_config()) != environment_config:
            raise ValueError("environment init mutated environment config")
        initial_observation = self._environment_observation(initial_environment)

        learner_configs: list[dict[str, object]] = []
        initial_learner_payloads: list[object] = []
        for learner in self._learners:
            config = _json_clone(learner.to_config())
            if not isinstance(config, dict):
                raise ValueError(f"{learner.name}: learner config must be a JSON object")
            _versioned_identifier(
                config.get("schema_version"),
                name=f"{learner.name}.schema_version",
            )
            first = learner.init(initial_observation)
            first_payload = self._learner_state_payload(learner, first)
            second_payload = self._learner_state_payload(
                learner,
                learner.init(initial_observation),
            )
            if first_payload != second_payload:
                raise ValueError(f"{learner.name}: init must be deterministic from config")
            if _json_clone(learner.to_config()) != config:
                raise ValueError(f"{learner.name}: init mutated learner config")
            if not self._state_valid_for_observation(learner, first, initial_observation):
                raise ValueError(f"{learner.name}: initial state does not own initial observation")
            usage = self._validate_resource_usage(learner, first)
            maximum_backward = learner.max_backward_calls_per_update
            if maximum_backward is not None:
                per_update = _nonnegative_int(
                    maximum_backward,
                    name=f"{learner.name}.max_backward_calls_per_update",
                )
                if transition_count * per_update > self._budget.backward_call_limit:
                    raise ValueError(f"{learner.name}: backward-call budget is too small")
            if usage.trainable_parameter_count is None:
                if self._budget.trainable_parameter_count_limit is not None:
                    raise ValueError(
                        f"{learner.name}: parameter count unavailable under a finite "
                        "parameter budget"
                    )
            learner_configs.append(config)
            initial_learner_payloads.append(first_payload)

        self._environment_config = environment_config
        self._initial_environment_payload = initial_environment_payload
        self._initial_observation = initial_observation
        self._learner_configs = tuple(learner_configs)
        self._initial_learner_payloads = tuple(initial_learner_payloads)

    def _assert_environment_config_stable(self) -> None:
        if _json_clone(self._environment.to_config()) != self._environment_config:
            raise ValueError("environment config changed during control evaluation")

    def _assert_learner_config_stable(self, learner: ContinuingControlLearner) -> None:
        index = self._learner_index(learner)
        if _json_clone(learner.to_config()) != self._learner_configs[index]:
            raise ValueError(f"{learner.name}: config changed during control evaluation")

    def _environment_state_payload(self, state: Any) -> object:
        """Serialize environment state; require idempotent and canonical round-trips."""
        payload = _json_clone(self._environment.state_to_config(state))
        if _json_clone(self._environment.state_to_config(state)) != payload:
            raise ValueError("environment state serialization is not idempotent")
        restored = self._environment.state_from_config(_json_clone(payload))
        if _json_clone(self._environment.state_to_config(restored)) != payload:
            raise ValueError("environment state serialization is not canonical")
        return payload

    @staticmethod
    def _learner_state_payload(learner: ContinuingControlLearner, state: Any) -> object:
        """Serialize learner state; require idempotent and canonical round-trips."""
        payload = _json_clone(learner.state_to_config(state))
        if _json_clone(learner.state_to_config(state)) != payload:
            raise ValueError(f"{learner.name}: state serialization is not idempotent")
        restored = learner.state_from_config(_json_clone(payload))
        if _json_clone(learner.state_to_config(restored)) != payload:
            raise ValueError(f"{learner.name}: state serialization is not canonical")
        return payload

    def _environment_observation(self, state: Any) -> tuple[float, ...]:
        config_bound = hasattr(self, "_environment_config")
        if config_bound:
            self._assert_environment_config_stable()
        payload = self._environment_state_payload(state)
        working = self._environment.state_from_config(_json_clone(payload))
        observation = _observation(
            self._environment.observation(working),
            name="environment observation",
        )
        if self._environment_state_payload(working) != payload:
            raise ValueError("environment observation query mutated environment state")
        if self._environment_state_payload(state) != payload:
            raise ValueError("environment observation query mutated live environment state")
        if config_bound:
            self._assert_environment_config_stable()
        return observation

    def _state_valid_for_observation(
        self,
        learner: ContinuingControlLearner,
        state: Any,
        observation: tuple[float, ...],
    ) -> bool:
        payload = self._learner_state_payload(learner, state)
        working = learner.state_from_config(_json_clone(payload))
        valid = learner.state_valid_for_observation(working, observation)
        if not isinstance(valid, bool):
            raise TypeError(f"{learner.name}: state validity must be boolean")
        if self._learner_state_payload(learner, working) != payload:
            raise ValueError(f"{learner.name}: state validity query mutated learner state")
        if self._learner_state_payload(learner, state) != payload:
            raise ValueError(f"{learner.name}: state validity query mutated live learner state")
        return valid

    def _validate_resource_usage(
        self,
        learner: ContinuingControlLearner,
        state: Any,
    ) -> ControlResourceUsage:
        config_bound = hasattr(self, "_learner_configs")
        if config_bound:
            self._assert_learner_config_stable(learner)
        payload = self._learner_state_payload(learner, state)
        working = learner.state_from_config(_json_clone(payload))
        usage = learner.resource_usage(working)
        if not isinstance(usage, ControlResourceUsage):
            raise TypeError(f"{learner.name}: resource_usage must return ControlResourceUsage")
        if self._learner_state_payload(learner, working) != payload:
            raise ValueError(f"{learner.name}: resource accounting mutated learner state")
        if usage.persistent_state_bytes > self._budget.persistent_state_bytes_limit:
            raise ValueError(f"{learner.name}: persistent state exceeds common ceiling")
        if usage.state_scalar_count > self._budget.state_scalar_count_limit:
            raise ValueError(f"{learner.name}: state scalar count exceeds common ceiling")
        parameter_limit = self._budget.trainable_parameter_count_limit
        if parameter_limit is not None:
            parameter_count = cast(int, usage.trainable_parameter_count)
            if parameter_count > parameter_limit:
                raise ValueError(f"{learner.name}: parameter count exceeds common ceiling")
        if config_bound:
            self._assert_learner_config_stable(learner)
        return usage

    def _time_call(self, callback: Callable[[], Any], *, name: str) -> tuple[Any, int]:
        start = _nonnegative_int(self._clock_ns(), name=f"{name} clock start")
        result = callback()
        stop = _nonnegative_int(self._clock_ns(), name=f"{name} clock stop")
        duration = stop - start
        if duration < 0:
            raise ValueError(f"{name} clock moved backwards")
        return result, duration

    def _decision(
        self,
        learner: ContinuingControlLearner,
        state: Any,
        observation: tuple[float, ...],
        used_ids: tuple[DecisionId, ...],
    ) -> tuple[ControlDecision, int]:
        # Fail-closed call pattern used for every learner/environment call in
        # this class: serialize the live state, run the callee on a
        # deserialized clone, then re-serialize both clone and live state to
        # prove the call had no side effects.
        self._assert_learner_config_stable(learner)
        payload = self._learner_state_payload(learner, state)
        working = learner.state_from_config(_json_clone(payload))
        raw, latency = self._time_call(
            lambda: learner.decide(working, observation),
            name=f"{learner.name}.decide",
        )
        if not isinstance(raw, ControlDecision):
            raise TypeError(f"{learner.name}: decide must return ControlDecision")
        if self._learner_state_payload(learner, working) != payload:
            raise ValueError(f"{learner.name}: decide mutated learner state")
        if self._learner_state_payload(learner, state) != payload:
            raise ValueError(f"{learner.name}: decide mutated live learner state")
        self._assert_learner_config_stable(learner)
        if not raw.armed:
            raise ValueError(f"{learner.name}: decision is not armed")
        if raw.observation != observation:
            raise ValueError(f"{learner.name}: decision observation ownership mismatch")
        if raw.action >= self._environment.n_actions:
            raise ValueError(f"{learner.name}: decision action is out of range")
        if raw.decision_id in used_ids:
            raise ValueError(f"{learner.name}: decision ID was already consumed")
        return raw, latency

    def init(self) -> ContinualControlRunState:
        self._assert_environment_config_stable()
        for learner in self._learners:
            self._assert_learner_config_stable(learner)
        conditions: list[_ControlConditionState] = []
        for learner, learner_payload in zip(
            self._learners,
            self._initial_learner_payloads,
            strict=True,
        ):
            learner_state = learner.state_from_config(_json_clone(learner_payload))
            environment_state = self._environment.state_from_config(
                _json_clone(self._initial_environment_payload)
            )
            decision, latency = self._decision(
                learner,
                learner_state,
                self._initial_observation,
                (),
            )
            usage = self._validate_resource_usage(learner, learner_state)
            conditions.append(
                _ControlConditionState(
                    learner_state=learner_state,
                    environment_state=environment_state,
                    current_observation=self._initial_observation,
                    pending_decision=decision,
                    used_decision_ids=(),
                    rewards=(),
                    delayed=(),
                    evaluator_matrix=(),
                    decision_latencies_ns=(latency,),
                    environment_latencies_ns=(),
                    update_latencies_ns=(),
                    probe_latencies_ns=(),
                    backward_call_counts=(),
                    persistent_state_bytes_high_water=usage.persistent_state_bytes,
                    state_scalar_count_high_water=usage.state_scalar_count,
                    safety_violations=0,
                    interventions=0,
                    near_misses=0,
                    cumulative_safety_cost=0.0,
                    cumulative_near_miss_cost=0.0,
                    maximum_step_safety_cost=0.0,
                )
            )
        state = ContinualControlRunState(step=0, conditions=tuple(conditions))
        self._validate_run_state(state)
        return state

    def _environment_step(
        self,
        condition: _ControlConditionState,
        decision: ControlDecision,
        regime_id: str,
    ) -> tuple[Any, ControlTransition, int]:
        self._assert_environment_config_stable()
        payload = self._environment_state_payload(condition.environment_state)
        working = self._environment.state_from_config(_json_clone(payload))
        raw, latency = self._time_call(
            lambda: self._environment.step(working, decision, regime_id),
            name="environment.step",
        )
        if not isinstance(raw, ControlEnvironmentUpdate):
            raise TypeError("environment.step must return ControlEnvironmentUpdate")
        if self._environment_state_payload(working) != payload:
            raise ValueError("environment.step mutated its source state")
        if self._environment_state_payload(condition.environment_state) != payload:
            raise ValueError("environment.step mutated live environment state")
        self._assert_environment_config_stable()
        transition = raw.transition
        if not isinstance(transition, ControlTransition):
            raise TypeError("environment update transition must be a ControlTransition")
        if (
            transition.observation != decision.observation
            or transition.action != decision.action
            or transition.decision_id != decision.decision_id
        ):
            raise ValueError("environment transition does not own the dispatched decision")
        next_payload = self._environment_state_payload(raw.state)
        next_state = self._environment.state_from_config(_json_clone(next_payload))
        next_observation = self._environment_observation(next_state)
        if next_observation != transition.next_decision_observation:
            raise ValueError("environment state does not match transition reset/bootstrap result")
        return next_state, transition, latency

    def _learner_update(
        self,
        learner: ContinuingControlLearner,
        state: Any,
        transition: ControlTransition,
    ) -> tuple[Any, int | None, int]:
        self._assert_learner_config_stable(learner)
        payload = self._learner_state_payload(learner, state)
        working = learner.state_from_config(_json_clone(payload))
        raw, latency = self._time_call(
            lambda: learner.update(working, transition),
            name=f"{learner.name}.update",
        )
        if not isinstance(raw, ControlLearnerUpdate):
            raise TypeError(f"{learner.name}: update must return ControlLearnerUpdate")
        if self._learner_state_payload(learner, working) != payload:
            raise ValueError(f"{learner.name}: update mutated its source state")
        if self._learner_state_payload(learner, state) != payload:
            raise ValueError(f"{learner.name}: update mutated live learner state")
        self._assert_learner_config_stable(learner)
        if not raw.applied:
            if self._learner_state_payload(learner, raw.state) != payload:
                raise ValueError(f"{learner.name}: rejected update changed learner state")
            raise ValueError(f"{learner.name}: rejected an owned environment transition")
        maximum_backward = learner.max_backward_calls_per_update
        if maximum_backward is None:
            if raw.backward_call_count is not None:
                raise ValueError(
                    f"{learner.name}: update reported backward calls without a declared bound"
                )
        else:
            if raw.backward_call_count is None:
                raise ValueError(f"{learner.name}: backward call count is unexpectedly unavailable")
            if raw.backward_call_count > maximum_backward:
                raise ValueError(f"{learner.name}: backward-call bound exceeded")
        next_payload = self._learner_state_payload(learner, raw.state)
        next_state = learner.state_from_config(_json_clone(next_payload))
        if not self._state_valid_for_observation(
            learner,
            next_state,
            transition.next_decision_observation,
        ):
            raise ValueError(f"{learner.name}: updated state does not own next observation")
        return next_state, raw.backward_call_count, latency

    def _probe(
        self,
        learner: ContinuingControlLearner,
        state: Any,
    ) -> tuple[tuple[float, ...], tuple[int, ...]]:
        self._assert_learner_config_stable(learner)
        live_payload = self._learner_state_payload(learner, state)
        row: list[float] = []
        latencies: list[int] = []
        for regime_id in self._protocol.evaluator_regime_ids:
            scores: list[float] = []
            for probe in self._probes[regime_id]:
                working = learner.state_from_config(_json_clone(live_payload))
                before = self._learner_state_payload(learner, working)
                raw_action, latency = self._time_call(
                    lambda: learner.probe_action(working, probe.observation),
                    name=f"{learner.name}.held_out_probe",
                )
                action = _nonnegative_int(raw_action, name="held-out probe action")
                if action >= self._environment.n_actions:
                    raise ValueError(f"{learner.name}: held-out probe action is out of range")
                if self._learner_state_payload(learner, working) != before:
                    raise ValueError(f"{learner.name}: held-out probe mutated snapshot state")
                if self._learner_state_payload(learner, state) != live_payload:
                    raise ValueError(f"{learner.name}: held-out probe mutated live state")
                self._assert_learner_config_stable(learner)
                scores.append(probe.action_scores[action])
                latencies.append(latency)
            row.append(sum(scores) / len(scores))
        return tuple(row), tuple(latencies)

    def _advance_condition(
        self,
        learner: ContinuingControlLearner,
        condition: _ControlConditionState,
        *,
        regime_id: str,
        completed_step: int,
    ) -> _ControlConditionState:
        """Run one full transaction for a single condition.

        The order is fixed: dispatch the pre-armed decision to the
        environment, feed the owned transition to the learner, re-check the
        resource ceilings, probe at checkpoint steps, then arm the next
        decision.  Arming ahead of the next environment step is what makes
        the return trace prequential — every action is committed before its
        outcome exists.
        """
        decision = condition.pending_decision
        if decision is None:
            raise ValueError(f"{learner.name}: no pending decision is available")
        environment_state, transition, environment_latency = self._environment_step(
            condition,
            decision,
            regime_id,
        )
        learner_state, backward_calls, update_latency = self._learner_update(
            learner,
            condition.learner_state,
            transition,
        )
        usage = self._validate_resource_usage(learner, learner_state)
        matrix = condition.evaluator_matrix
        probe_latencies = condition.probe_latencies_ns
        if completed_step in self._protocol.checkpoint_steps:
            row, new_probe_latencies = self._probe(learner, learner_state)
            matrix = (*matrix, row)
            probe_latencies = (*probe_latencies, *new_probe_latencies)

        used_ids = (*condition.used_decision_ids, decision.decision_id)
        pending: ControlDecision | None = None
        decision_latencies = condition.decision_latencies_ns
        if completed_step < len(self._protocol.regime_schedule):
            pending, decision_latency = self._decision(
                learner,
                learner_state,
                transition.next_decision_observation,
                used_ids,
            )
            decision_latencies = (*decision_latencies, decision_latency)

        deadline_ns = self._protocol.operation_latency_deadline_ms * 1_000_000.0
        dispatched_decision_latency = condition.decision_latencies_ns[-1]
        delayed = (
            dispatched_decision_latency > deadline_ns
            or environment_latency > deadline_ns
            or update_latency > deadline_ns
        )
        return _ControlConditionState(
            learner_state=learner_state,
            environment_state=environment_state,
            current_observation=transition.next_decision_observation,
            pending_decision=pending,
            used_decision_ids=used_ids,
            rewards=(*condition.rewards, transition.reward),
            delayed=(*condition.delayed, delayed),
            evaluator_matrix=matrix,
            decision_latencies_ns=decision_latencies,
            environment_latencies_ns=(*condition.environment_latencies_ns, environment_latency),
            update_latencies_ns=(*condition.update_latencies_ns, update_latency),
            probe_latencies_ns=probe_latencies,
            backward_call_counts=(*condition.backward_call_counts, backward_calls),
            persistent_state_bytes_high_water=max(
                condition.persistent_state_bytes_high_water,
                usage.persistent_state_bytes,
            ),
            state_scalar_count_high_water=max(
                condition.state_scalar_count_high_water,
                usage.state_scalar_count,
            ),
            safety_violations=condition.safety_violations + int(transition.safety_violation),
            interventions=condition.interventions + int(transition.intervention),
            near_misses=condition.near_misses + int(transition.near_miss),
            cumulative_safety_cost=condition.cumulative_safety_cost + transition.safety_cost,
            cumulative_near_miss_cost=(
                condition.cumulative_near_miss_cost + transition.near_miss_cost
            ),
            maximum_step_safety_cost=max(
                condition.maximum_step_safety_cost,
                transition.safety_cost,
            ),
        )

    def advance(
        self,
        state: ContinualControlRunState,
        *,
        steps: int = 1,
    ) -> ContinualControlRunState:
        """Advance only complete decision/environment/update transactions.

        Run-state invariants are re-validated after every completed step, so
        a contract violation raises before the partial state can be observed
        or checkpointed.
        """
        self._validate_run_state(state)
        requested = _positive_int(steps, name="steps")
        if state.step + requested > len(self._protocol.regime_schedule):
            raise ValueError("advance would exceed the bounded regime schedule")
        current = state
        for _ in range(requested):
            completed_step = current.step + 1
            regime_id = self._protocol.regime_schedule[current.step]
            conditions = tuple(
                self._advance_condition(
                    learner,
                    condition,
                    regime_id=regime_id,
                    completed_step=completed_step,
                )
                for learner, condition in zip(
                    self._learners,
                    current.conditions,
                    strict=True,
                )
            )
            current = ContinualControlRunState(completed_step, conditions)
            self._validate_run_state(current)
        return current

    @staticmethod
    def _latency_summary(values: tuple[int, ...]) -> dict[str, float | None]:
        if not values:
            return {"p50_ms": None, "p95_ms": None, "p99_ms": None}
        ordered = sorted(value / 1_000_000.0 for value in values)

        # Nearest-rank (ceiling) quantiles: every reported percentile is an
        # observed sample, never an interpolation between samples.
        def quantile(fraction: float) -> float:
            return ordered[math.ceil(fraction * len(ordered)) - 1]

        return {
            "p50_ms": quantile(0.50),
            "p95_ms": quantile(0.95),
            "p99_ms": quantile(0.99),
        }

    def _condition_report(
        self,
        learner: ContinuingControlLearner,
        condition: _ControlConditionState,
        *,
        role: str,
    ) -> dict[str, object]:
        self._assert_learner_config_stable(learner)
        learner_payload = self._learner_state_payload(learner, condition.learner_state)
        working = learner.state_from_config(_json_clone(learner_payload))
        diagnostics = learner.diagnostics(working)
        _validate_diagnostics(diagnostics, name=f"{learner.name}.diagnostics")
        if self._learner_state_payload(learner, working) != learner_payload:
            raise ValueError(f"{learner.name}: diagnostics mutated snapshot state")
        if self._learner_state_payload(learner, condition.learner_state) != learner_payload:
            raise ValueError(f"{learner.name}: diagnostics mutated live state")
        usage = self._validate_resource_usage(learner, condition.learner_state)
        rewards = condition.rewards
        backward_available = all(value is not None for value in condition.backward_call_counts)
        backward_total = (
            sum(cast(int, value) for value in condition.backward_call_counts)
            if backward_available
            else None
        )
        return {
            "name": learner.name,
            "role": role,
            "learner_type": cast(
                str,
                self._learner_configs[self._learner_index(learner)]["type"],
            ),
            "predict_action_before_environment_outcome": True,
            "trace": {
                "rewards": list(rewards),
                "processed": [True] * len(rewards),
                "evaluator_regime_ids_in_learner_trace": False,
            },
            "held_out_non_learning_probes": {
                "regime_order": list(self._protocol.evaluator_regime_ids),
                "checkpoint_steps": list(self._protocol.checkpoint_steps),
                "values": [list(row) for row in condition.evaluator_matrix],
                "snapshot_mutation_checks": True,
            },
            "metrics": _control_metrics(
                rewards,
                condition.evaluator_matrix,
                self._protocol,
            ),
            "operations": {
                "counts": {
                    "processed_transitions": len(rewards),
                    "delayed_transitions": sum(condition.delayed),
                    "dropped_transitions": 0,
                    "decision_calls": len(condition.decision_latencies_ns),
                    "environment_calls": len(condition.environment_latencies_ns),
                    "update_calls": len(condition.update_latencies_ns),
                    "held_out_probe_calls": len(condition.probe_latencies_ns),
                    "backward_calls": backward_total,
                    "backward_calls_available": backward_available,
                },
                "latency_ms": {
                    "decision": self._latency_summary(condition.decision_latencies_ns),
                    "environment": self._latency_summary(condition.environment_latencies_ns),
                    "update": self._latency_summary(condition.update_latencies_ns),
                    "held_out_probe": self._latency_summary(condition.probe_latencies_ns),
                    "measurement_method": self._latency_measurement_method,
                },
            },
            "resources": {
                "current_persistent_state_bytes": usage.persistent_state_bytes,
                "persistent_state_bytes_high_water": condition.persistent_state_bytes_high_water,
                "current_state_scalar_count": usage.state_scalar_count,
                "state_scalar_count_high_water": condition.state_scalar_count_high_water,
                "trainable_parameter_count": usage.trainable_parameter_count,
                "measurement_method": usage.measurement_method,
            },
            "safety": {
                "checks": len(rewards),
                "violations": condition.safety_violations,
                "interventions": condition.interventions,
                "near_misses": condition.near_misses,
                "cumulative_cost": condition.cumulative_safety_cost,
                "cumulative_near_miss_cost": condition.cumulative_near_miss_cost,
                "maximum_step_cost": condition.maximum_step_safety_cost,
                "measurement_method": "environment-supplied per-transition control outcomes",
            },
            "diagnostics": None if diagnostics is None else dict(diagnostics),
        }

    def build_report(self, state: ContinualControlRunState) -> dict[str, object]:
        """Build a versioned descriptive report after the complete schedule.

        The freshly built report is immediately round-tripped through the
        same strict reconstruction used for externally loaded reports, so the
        evaluator cannot emit an artifact its own validator would reject.
        """
        self._validate_run_state(state)
        if state.step != len(self._protocol.regime_schedule):
            raise ValueError("cannot build report from an incomplete control run")
        evaluator_config = self.to_config()
        condition_config_digests = {
            learner.name: _digest(config)
            for learner, config in zip(
                self._learners,
                self._learner_configs,
                strict=True,
            )
        }
        report = {
            "schema_version": CONTROL_REPORT_SCHEMA,
            "acceptance_status": ACCEPTANCE_STATUS,
            "interpretation": REPORT_INTERPRETATION,
            "run_id": self._run_id,
            "metric_definitions": dict(CONTROL_METRIC_DEFINITIONS),
            "evaluator_config": evaluator_config,
            "evaluator_identity": {
                "evaluator_config_sha256": _digest(evaluator_config),
                "environment_config_sha256": _digest(self._environment_config),
                "condition_config_sha256": condition_config_digests,
                "probe_sha256": evaluator_config["probe_sha256"],
            },
            "protocol": {
                **cast(
                    dict[str, object],
                    _json_clone(dataclasses.asdict(self._protocol)),
                ),
                "regime_metadata_is_evaluator_only": True,
            },
            "comparison_contract": {
                "candidate": self._candidate.name,
                "baselines": [learner.name for learner in self._baselines],
                "shared_ceiling_budget": dataclasses.asdict(self._budget),
                "shared_ceiling_verified": True,
                "realized_compute_memory_parity_verified": False,
                "limitation": (
                    "Conditions share hard ceilings and report realized usage, but this runner "
                    "does not pad or otherwise establish equal realized compute or memory."
                ),
            },
            "conditions": [
                self._condition_report(
                    learner,
                    condition,
                    role="candidate" if index == 0 else "baseline",
                )
                for index, (learner, condition) in enumerate(
                    zip(self._learners, state.conditions, strict=True)
                )
            ],
            "limitations": list(CONTROL_REPORT_LIMITATIONS),
            "accepted_scientific_evidence": False,
        }
        canonical = cast(dict[str, object], _json_clone(report))
        validation = validate_continual_control_report(
            canonical,
            expected_evaluator_config=evaluator_config,
        )
        if not validation.valid:
            raise RuntimeError(
                "internal continual-control report construction failed validation: "
                + "; ".join(validation.errors)
            )
        return canonical

    def _validate_run_state(self, state: ContinualControlRunState) -> None:
        """Fail-closed invariant sweep over the complete run state.

        Trace lengths, budget ceilings, decision-ID uniqueness, cached
        observations, and resource high-water marks are all re-derived from
        the protocol and cross-checked; any mismatch raises rather than being
        repaired.
        """
        self._assert_environment_config_stable()
        if not isinstance(state, ContinualControlRunState):
            raise TypeError("state must be a ContinualControlRunState")
        step = _nonnegative_int(state.step, name="state.step")
        transition_count = len(self._protocol.regime_schedule)
        if step > transition_count:
            raise ValueError("state.step exceeds transition schedule")
        if len(state.conditions) != len(self._learners):
            raise ValueError("state condition count does not match evaluator")
        completed_checkpoints = sum(value <= step for value in self._protocol.checkpoint_steps)
        probes_per_checkpoint = sum(len(values) for values in self._probes.values())
        # Decisions run one ahead of transitions (prequential arming): the
        # first is armed at construction and each transaction arms the next,
        # so `step` transitions imply ``step + 1`` decision calls — except
        # after the final transition, when no further decision is armed.
        expected_decisions = min(step + 1, transition_count)
        expected_probes = completed_checkpoints * probes_per_checkpoint
        for learner, condition in zip(self._learners, state.conditions, strict=True):
            self._assert_learner_config_stable(learner)
            if len(condition.rewards) != step or len(condition.delayed) != step:
                raise ValueError(f"{learner.name}: online trace length does not match step")
            for index, reward in enumerate(condition.rewards):
                _finite_float(reward, name=f"{learner.name}.rewards[{index}]")
            if any(not isinstance(value, bool) for value in condition.delayed):
                raise ValueError(f"{learner.name}: delayed flags must be boolean")
            if len(condition.used_decision_ids) != step:
                raise ValueError(f"{learner.name}: consumed decision count is inconsistent")
            if len(set(condition.used_decision_ids)) != len(condition.used_decision_ids):
                raise ValueError(f"{learner.name}: consumed decision IDs must be unique")
            if len(condition.used_decision_ids) > self._budget.stored_decision_id_limit:
                raise ValueError(f"{learner.name}: stored decision-ID budget exceeded")
            pending_expected = step < transition_count
            if (condition.pending_decision is not None) != pending_expected:
                raise ValueError(f"{learner.name}: pending-decision state is inconsistent")
            if condition.pending_decision is not None:
                pending = condition.pending_decision
                if (
                    not pending.armed
                    or pending.observation != condition.current_observation
                    or pending.action >= self._environment.n_actions
                    or pending.decision_id in condition.used_decision_ids
                ):
                    raise ValueError(f"{learner.name}: pending decision is invalid or stale")
            if len(condition.evaluator_matrix) != completed_checkpoints:
                raise ValueError(f"{learner.name}: held-out evaluator row count is inconsistent")
            for row in condition.evaluator_matrix:
                if len(row) != len(self._protocol.evaluator_regime_ids):
                    raise ValueError(f"{learner.name}: held-out evaluator row width is invalid")
                for value in row:
                    _finite_float(value, name=f"{learner.name}.held-out score")
            expected_lengths = (
                (condition.decision_latencies_ns, expected_decisions, "decision"),
                (condition.environment_latencies_ns, step, "environment"),
                (condition.update_latencies_ns, step, "update"),
                (condition.probe_latencies_ns, expected_probes, "probe"),
                (condition.backward_call_counts, step, "backward"),
            )
            for values, expected, label in expected_lengths:
                if len(values) != expected:
                    raise ValueError(f"{learner.name}: {label} operation count is inconsistent")
            for values in (
                condition.decision_latencies_ns,
                condition.environment_latencies_ns,
                condition.update_latencies_ns,
                condition.probe_latencies_ns,
            ):
                for value in values:
                    _nonnegative_int(value, name=f"{learner.name}.latency")
            known_backward = [
                value for value in condition.backward_call_counts if value is not None
            ]
            if sum(known_backward) > self._budget.backward_call_limit:
                raise ValueError(f"{learner.name}: backward-call budget exceeded")
            if len(condition.decision_latencies_ns) > self._budget.decision_call_limit:
                raise ValueError(f"{learner.name}: decision-call budget exceeded")
            if len(condition.environment_latencies_ns) > self._budget.environment_call_limit:
                raise ValueError(f"{learner.name}: environment-call budget exceeded")
            if len(condition.update_latencies_ns) > self._budget.update_call_limit:
                raise ValueError(f"{learner.name}: update-call budget exceeded")
            if len(condition.probe_latencies_ns) > self._budget.probe_call_limit:
                raise ValueError(f"{learner.name}: probe-call budget exceeded")
            observation = _observation(
                condition.current_observation,
                name=f"{learner.name}.current_observation",
            )
            if self._environment_observation(condition.environment_state) != observation:
                raise ValueError(f"{learner.name}: environment observation cache is inconsistent")
            if not self._state_valid_for_observation(
                learner,
                condition.learner_state,
                observation,
            ):
                raise ValueError(f"{learner.name}: learner observation cache is inconsistent")
            usage = self._validate_resource_usage(learner, condition.learner_state)
            if condition.persistent_state_bytes_high_water < usage.persistent_state_bytes:
                raise ValueError(f"{learner.name}: persistent-state high-water is inconsistent")
            if condition.state_scalar_count_high_water < usage.state_scalar_count:
                raise ValueError(f"{learner.name}: scalar-count high-water is inconsistent")
            if (
                condition.persistent_state_bytes_high_water
                > self._budget.persistent_state_bytes_limit
                or condition.state_scalar_count_high_water > self._budget.state_scalar_count_limit
            ):
                raise ValueError(f"{learner.name}: resource high-water exceeds common ceiling")
            for name in ("safety_violations", "interventions", "near_misses"):
                count = _nonnegative_int(getattr(condition, name), name=f"{learner.name}.{name}")
                if count > step:
                    raise ValueError(f"{learner.name}: {name} exceeds processed transitions")
            _nonnegative_float(
                condition.cumulative_safety_cost,
                name=f"{learner.name}.cumulative_safety_cost",
            )
            _nonnegative_float(
                condition.cumulative_near_miss_cost,
                name=f"{learner.name}.cumulative_near_miss_cost",
            )
            _nonnegative_float(
                condition.maximum_step_safety_cost,
                name=f"{learner.name}.maximum_step_safety_cost",
            )

    def to_config(self) -> dict[str, object]:
        """Return checkpoint-bound evaluator identity without probe targets in learners."""
        self._assert_environment_config_stable()
        for learner in self._learners:
            self._assert_learner_config_stable(learner)
        protocol = dataclasses.asdict(self._protocol)
        probes = {
            regime_id: [probe.to_config() for probe in self._probes[regime_id]]
            for regime_id in self._protocol.evaluator_regime_ids
        }
        return {
            "type": "ContinualControlEvaluator",
            "schema_version": CONTROL_EVALUATOR_SCHEMA,
            "report_schema_version": CONTROL_REPORT_SCHEMA,
            "run_id": self._run_id,
            "protocol": _json_clone(protocol),
            "environment": _json_clone(self._environment_config),
            "conditions": [
                {
                    "role": "candidate" if index == 0 else "baseline",
                    "learner": _json_clone(config),
                }
                for index, config in enumerate(self._learner_configs)
            ],
            "budget": dataclasses.asdict(self._budget),
            "probe_sha256": _digest(probes),
            "probe_examples_per_checkpoint": sum(
                len(values) for values in self._probes.values()
            ),
            "latency_measurement_method": self._latency_measurement_method,
            "development_only": True,
            "accepted_scientific_evidence": False,
        }

    def _state_payload(self, state: ContinualControlRunState) -> dict[str, object]:
        self._validate_run_state(state)
        conditions: list[dict[str, object]] = []
        for learner, condition in zip(self._learners, state.conditions, strict=True):
            conditions.append(
                {
                    "name": learner.name,
                    "learner_state": self._learner_state_payload(
                        learner,
                        condition.learner_state,
                    ),
                    "environment_state": self._environment_state_payload(
                        condition.environment_state
                    ),
                    "current_observation": list(condition.current_observation),
                    "pending_decision": (
                        None
                        if condition.pending_decision is None
                        else condition.pending_decision.to_config()
                    ),
                    "used_decision_ids": [list(value) for value in condition.used_decision_ids],
                    "rewards": list(condition.rewards),
                    "delayed": list(condition.delayed),
                    "evaluator_matrix": [list(row) for row in condition.evaluator_matrix],
                    "decision_latencies_ns": list(condition.decision_latencies_ns),
                    "environment_latencies_ns": list(condition.environment_latencies_ns),
                    "update_latencies_ns": list(condition.update_latencies_ns),
                    "probe_latencies_ns": list(condition.probe_latencies_ns),
                    "backward_call_counts": list(condition.backward_call_counts),
                    "persistent_state_bytes_high_water": (
                        condition.persistent_state_bytes_high_water
                    ),
                    "state_scalar_count_high_water": condition.state_scalar_count_high_water,
                    "safety_violations": condition.safety_violations,
                    "interventions": condition.interventions,
                    "near_misses": condition.near_misses,
                    "cumulative_safety_cost": condition.cumulative_safety_cost,
                    "cumulative_near_miss_cost": condition.cumulative_near_miss_cost,
                    "maximum_step_safety_cost": condition.maximum_step_safety_cost,
                }
            )
        return {"step": state.step, "conditions": conditions}

    def save_checkpoint(self, state: ContinualControlRunState, path: str | Path) -> None:
        """Save canonical JSON only between complete environment transactions.

        The write is atomic: the payload goes to a same-directory temporary
        file, is fsynced, then renamed over the destination, so a crash can
        never leave a truncated checkpoint behind.
        """
        state_payload = self._state_payload(state)
        config = self.to_config()
        payload = {
            "schema_version": CONTROL_CHECKPOINT_SCHEMA,
            "evaluator_config": config,
            "evaluator_config_sha256": _digest(config),
            "state": state_payload,
            "state_sha256": _digest(state_payload),
        }
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
                    json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
                    + "\n"
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    def load_checkpoint(self, path: str | Path) -> ContinualControlRunState:
        """Restore strict canonical state bound to config, schedule, and probes."""
        payload = json.loads(
            Path(path).read_text(encoding="utf-8"),
            parse_constant=_reject_nonstandard_json_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
        mapping = _mapping(payload, name="checkpoint")
        if set(mapping) != {
            "schema_version",
            "evaluator_config",
            "evaluator_config_sha256",
            "state",
            "state_sha256",
        }:
            raise ValueError("control checkpoint keys are invalid")
        if mapping["schema_version"] != CONTROL_CHECKPOINT_SCHEMA:
            raise ValueError("control checkpoint schema is invalid")
        config = mapping["evaluator_config"]
        if config != self.to_config() or mapping["evaluator_config_sha256"] != _digest(config):
            raise ValueError("control checkpoint config does not match evaluator")
        raw_state = mapping["state"]
        if mapping["state_sha256"] != _digest(raw_state):
            raise ValueError("control checkpoint state digest does not match")
        state_mapping = _mapping(raw_state, name="state")
        if set(state_mapping) != {"step", "conditions"}:
            raise ValueError("control checkpoint state keys are invalid")
        step = _nonnegative_int(state_mapping["step"], name="state.step")
        raw_conditions = _list(state_mapping["conditions"], name="state.conditions")
        if len(raw_conditions) != len(self._learners):
            raise ValueError("control checkpoint conditions do not match evaluator")
        conditions: list[_ControlConditionState] = []
        for index, (learner, raw_condition) in enumerate(
            zip(self._learners, raw_conditions, strict=True)
        ):
            location = f"state.conditions[{index}]"
            condition = _mapping(raw_condition, name=location)
            expected_keys = {
                "name",
                "learner_state",
                "environment_state",
                "current_observation",
                "pending_decision",
                "used_decision_ids",
                "rewards",
                "delayed",
                "evaluator_matrix",
                "decision_latencies_ns",
                "environment_latencies_ns",
                "update_latencies_ns",
                "probe_latencies_ns",
                "backward_call_counts",
                "persistent_state_bytes_high_water",
                "state_scalar_count_high_water",
                "safety_violations",
                "interventions",
                "near_misses",
                "cumulative_safety_cost",
                "cumulative_near_miss_cost",
                "maximum_step_safety_cost",
            }
            if set(condition) != expected_keys:
                raise ValueError(f"{location} keys are invalid")
            if condition["name"] != learner.name:
                raise ValueError(f"{location} learner name does not match")
            pending_raw = condition["pending_decision"]
            pending = None if pending_raw is None else _decision_from_config(
                pending_raw,
                name=f"{location}.pending_decision",
            )
            delayed_raw = _list(condition["delayed"], name=f"{location}.delayed")
            if any(not isinstance(value, bool) for value in delayed_raw):
                raise ValueError(f"{location}.delayed values must be boolean")

            def int_values(field: str) -> tuple[int, ...]:
                return tuple(
                    _nonnegative_int(value, name=f"{location}.{field}")
                    for value in _list(condition[field], name=f"{location}.{field}")
                )

            backward_values: list[int | None] = []
            for value in _list(
                condition["backward_call_counts"],
                name=f"{location}.backward_call_counts",
            ):
                backward_values.append(
                    None
                    if value is None
                    else _nonnegative_int(value, name=f"{location}.backward_call_counts")
                )
            conditions.append(
                _ControlConditionState(
                    learner_state=learner.state_from_config(
                        _json_clone(condition["learner_state"])
                    ),
                    environment_state=self._environment.state_from_config(
                        _json_clone(condition["environment_state"])
                    ),
                    current_observation=_observation_from_json(
                        condition["current_observation"],
                        name=f"{location}.current_observation",
                    ),
                    pending_decision=pending,
                    used_decision_ids=tuple(
                        _decision_id_from_json(value, name=f"{location}.used_decision_ids")
                        for value in _list(
                            condition["used_decision_ids"],
                            name=f"{location}.used_decision_ids",
                        )
                    ),
                    rewards=tuple(
                        _finite_float(value, name=f"{location}.rewards")
                        for value in _list(condition["rewards"], name=f"{location}.rewards")
                    ),
                    delayed=cast(tuple[bool, ...], tuple(delayed_raw)),
                    evaluator_matrix=tuple(
                        tuple(
                            _finite_float(value, name=f"{location}.evaluator_matrix")
                            for value in _list(row, name=f"{location}.evaluator_matrix")
                        )
                        for row in _list(
                            condition["evaluator_matrix"],
                            name=f"{location}.evaluator_matrix",
                        )
                    ),
                    decision_latencies_ns=int_values("decision_latencies_ns"),
                    environment_latencies_ns=int_values("environment_latencies_ns"),
                    update_latencies_ns=int_values("update_latencies_ns"),
                    probe_latencies_ns=int_values("probe_latencies_ns"),
                    backward_call_counts=tuple(backward_values),
                    persistent_state_bytes_high_water=_positive_int(
                        condition["persistent_state_bytes_high_water"],
                        name=f"{location}.persistent_state_bytes_high_water",
                    ),
                    state_scalar_count_high_water=_positive_int(
                        condition["state_scalar_count_high_water"],
                        name=f"{location}.state_scalar_count_high_water",
                    ),
                    safety_violations=_nonnegative_int(
                        condition["safety_violations"],
                        name=f"{location}.safety_violations",
                    ),
                    interventions=_nonnegative_int(
                        condition["interventions"],
                        name=f"{location}.interventions",
                    ),
                    near_misses=_nonnegative_int(
                        condition["near_misses"],
                        name=f"{location}.near_misses",
                    ),
                    cumulative_safety_cost=_nonnegative_float(
                        condition["cumulative_safety_cost"],
                        name=f"{location}.cumulative_safety_cost",
                    ),
                    cumulative_near_miss_cost=_nonnegative_float(
                        condition["cumulative_near_miss_cost"],
                        name=f"{location}.cumulative_near_miss_cost",
                    ),
                    maximum_step_safety_cost=_nonnegative_float(
                        condition["maximum_step_safety_cost"],
                        name=f"{location}.maximum_step_safety_cost",
                    ),
                )
            )
        state = ContinualControlRunState(step=step, conditions=tuple(conditions))
        self._validate_run_state(state)
        # Round-trip: re-serializing the restored state must reproduce the
        # stored payload exactly, or the checkpoint is rejected.
        if self._state_payload(state) != raw_state:
            raise ValueError("control checkpoint state is not canonical")
        return state


def _exact_keys(
    mapping: Mapping[str, object],
    expected: set[str],
    *,
    name: str,
) -> None:
    if set(mapping) != expected:
        raise ValueError(f"{name} keys are invalid")


def _string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _boolean(value: object, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be boolean")
    return value


def _sha256(value: object, *, name: str) -> str:
    resolved = _string(value, name=name)
    if len(resolved) != 64 or any(character not in "0123456789abcdef" for character in resolved):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return resolved


def _string_list(value: object, *, name: str) -> list[str]:
    items = _list(value, name=name)
    return [_string(item, name=f"{name}[{index}]") for index, item in enumerate(items)]


def _parse_protocol_config(value: object) -> tuple[ContinuingControlProtocol, dict[str, object]]:
    mapping = _mapping(value, name="evaluator_config.protocol")
    _exact_keys(
        mapping,
        {field.name for field in dataclasses.fields(ContinuingControlProtocol)},
        name="evaluator_config.protocol",
    )
    checkpoint_values = _list(
        mapping["checkpoint_steps"],
        name="evaluator_config.protocol.checkpoint_steps",
    )
    regime_ids = tuple(
        _string_list(
            mapping["evaluator_regime_ids"],
            name="evaluator_config.protocol.evaluator_regime_ids",
        )
    )

    def named_float_mapping(field: str) -> dict[str, float]:
        values = _mapping(mapping[field], name=f"evaluator_config.protocol.{field}")
        if set(values) != set(regime_ids):
            raise ValueError(
                f"evaluator_config.protocol.{field} keys do not match evaluator regimes"
            )
        return {
            regime_id: _finite_float(
                values[regime_id],
                name=f"evaluator_config.protocol.{field}.{regime_id}",
            )
            for regime_id in regime_ids
        }

    raw_first_exposure = _mapping(
        mapping["first_exposure_checkpoint"],
        name="evaluator_config.protocol.first_exposure_checkpoint",
    )
    if set(raw_first_exposure) != set(regime_ids):
        raise ValueError(
            "evaluator_config.protocol.first_exposure_checkpoint keys do not match "
            "evaluator regimes"
        )
    protocol = ContinuingControlProtocol(
        schema_version=_string(
            mapping["schema_version"],
            name="evaluator_config.protocol.schema_version",
        ),
        protocol_id=_string(
            mapping["protocol_id"],
            name="evaluator_config.protocol.protocol_id",
        ),
        higher_is_better=_boolean(
            mapping["higher_is_better"],
            name="evaluator_config.protocol.higher_is_better",
        ),
        regime_schedule=tuple(
            _string_list(
                mapping["regime_schedule"],
                name="evaluator_config.protocol.regime_schedule",
            )
        ),
        evaluator_regime_ids=regime_ids,
        checkpoint_steps=tuple(
            _positive_int(
                item,
                name=f"evaluator_config.protocol.checkpoint_steps[{index}]",
            )
            for index, item in enumerate(checkpoint_values)
        ),
        first_exposure_checkpoint={
            regime_id: _nonnegative_int(
                raw_first_exposure[regime_id],
                name=(
                    "evaluator_config.protocol.first_exposure_checkpoint."
                    f"{regime_id}"
                ),
            )
            for regime_id in regime_ids
        },
        forward_transfer_reference=named_float_mapping(
            "forward_transfer_reference"
        ),
        recovery_thresholds=named_float_mapping("recovery_thresholds"),
        stability_references=named_float_mapping("stability_references"),
        recovery_window=_positive_int(
            mapping["recovery_window"],
            name="evaluator_config.protocol.recovery_window",
        ),
        worst_window_size=_positive_int(
            mapping["worst_window_size"],
            name="evaluator_config.protocol.worst_window_size",
        ),
        operation_latency_deadline_ms=_finite_float(
            mapping["operation_latency_deadline_ms"],
            name="evaluator_config.protocol.operation_latency_deadline_ms",
        ),
    )
    canonical = cast(dict[str, object], _json_clone(dataclasses.asdict(protocol)))
    if canonical != mapping:
        raise ValueError("evaluator_config.protocol is not canonical")
    return protocol, canonical


def _parse_budget_config(value: object) -> tuple[ContinuingControlBudget, dict[str, object]]:
    mapping = _mapping(value, name="evaluator_config.budget")
    expected = {field.name for field in dataclasses.fields(ContinuingControlBudget)}
    _exact_keys(mapping, expected, name="evaluator_config.budget")
    raw_parameter_limit = mapping["trainable_parameter_count_limit"]
    parameter_limit = (
        None
        if raw_parameter_limit is None
        else _nonnegative_int(
            raw_parameter_limit,
            name="evaluator_config.budget.trainable_parameter_count_limit",
        )
    )
    budget = ContinuingControlBudget(
        transition_limit=_positive_int(
            mapping["transition_limit"],
            name="evaluator_config.budget.transition_limit",
        ),
        decision_call_limit=_positive_int(
            mapping["decision_call_limit"],
            name="evaluator_config.budget.decision_call_limit",
        ),
        environment_call_limit=_positive_int(
            mapping["environment_call_limit"],
            name="evaluator_config.budget.environment_call_limit",
        ),
        update_call_limit=_positive_int(
            mapping["update_call_limit"],
            name="evaluator_config.budget.update_call_limit",
        ),
        probe_call_limit=_positive_int(
            mapping["probe_call_limit"],
            name="evaluator_config.budget.probe_call_limit",
        ),
        backward_call_limit=_nonnegative_int(
            mapping["backward_call_limit"],
            name="evaluator_config.budget.backward_call_limit",
        ),
        persistent_state_bytes_limit=_positive_int(
            mapping["persistent_state_bytes_limit"],
            name="evaluator_config.budget.persistent_state_bytes_limit",
        ),
        state_scalar_count_limit=_positive_int(
            mapping["state_scalar_count_limit"],
            name="evaluator_config.budget.state_scalar_count_limit",
        ),
        trainable_parameter_count_limit=parameter_limit,
        stored_decision_id_limit=_positive_int(
            mapping["stored_decision_id_limit"],
            name="evaluator_config.budget.stored_decision_id_limit",
        ),
    )
    canonical = cast(dict[str, object], _json_clone(dataclasses.asdict(budget)))
    if canonical != mapping:
        raise ValueError("evaluator_config.budget is not canonical")
    return budget, canonical


def _parse_evaluator_config(
    value: object,
) -> tuple[
    dict[str, object],
    ContinuingControlProtocol,
    ContinuingControlBudget,
    tuple[tuple[str, str, str, dict[str, object]], ...],
]:
    mapping = _mapping(value, name="evaluator_config")
    _exact_keys(
        mapping,
        {
            "type",
            "schema_version",
            "report_schema_version",
            "run_id",
            "protocol",
            "environment",
            "conditions",
            "budget",
            "probe_sha256",
            "probe_examples_per_checkpoint",
            "latency_measurement_method",
            "development_only",
            "accepted_scientific_evidence",
        },
        name="evaluator_config",
    )
    if mapping["type"] != "ContinualControlEvaluator":
        raise ValueError("evaluator_config.type is invalid")
    if mapping["schema_version"] != CONTROL_EVALUATOR_SCHEMA:
        raise ValueError("evaluator_config.schema_version is invalid")
    if mapping["report_schema_version"] != CONTROL_REPORT_SCHEMA:
        raise ValueError("evaluator_config.report_schema_version is invalid")
    _versioned_identifier(mapping["run_id"], name="evaluator_config.run_id")
    protocol, _ = _parse_protocol_config(mapping["protocol"])
    budget, _ = _parse_budget_config(mapping["budget"])
    if budget.transition_limit != len(protocol.regime_schedule):
        raise ValueError("evaluator config transition limit does not match protocol")
    environment = _mapping(mapping["environment"], name="evaluator_config.environment")
    _string(environment.get("type"), name="evaluator_config.environment.type")
    _versioned_identifier(
        environment.get("schema_version"),
        name="evaluator_config.environment.schema_version",
    )
    raw_conditions = _list(mapping["conditions"], name="evaluator_config.conditions")
    if len(raw_conditions) < 3:
        raise ValueError("evaluator config requires one candidate and at least two baselines")
    condition_specs: list[tuple[str, str, str, dict[str, object]]] = []
    names: set[str] = set()
    for index, raw_condition in enumerate(raw_conditions):
        location = f"evaluator_config.conditions[{index}]"
        condition = _mapping(raw_condition, name=location)
        _exact_keys(condition, {"role", "learner"}, name=location)
        role = _string(condition["role"], name=f"{location}.role")
        expected_role = "candidate" if index == 0 else "baseline"
        if role != expected_role:
            raise ValueError(f"{location}.role must be {expected_role}")
        learner = _mapping(condition["learner"], name=f"{location}.learner")
        learner_type = _string(learner.get("type"), name=f"{location}.learner.type")
        _versioned_identifier(
            learner.get("schema_version"),
            name=f"{location}.learner.schema_version",
        )
        learner_name = _string(learner.get("name"), name=f"{location}.learner.name")
        if learner_name in names:
            raise ValueError("evaluator config learner names must be unique")
        names.add(learner_name)
        learner_config = cast(dict[str, object], _json_clone(learner))
        condition_specs.append((role, learner_name, learner_type, learner_config))
    _sha256(mapping["probe_sha256"], name="evaluator_config.probe_sha256")
    probe_examples = _positive_int(
        mapping["probe_examples_per_checkpoint"],
        name="evaluator_config.probe_examples_per_checkpoint",
    )
    if probe_examples * len(protocol.checkpoint_steps) > budget.probe_call_limit:
        raise ValueError("evaluator config probe calls exceed budget")
    _string(
        mapping["latency_measurement_method"],
        name="evaluator_config.latency_measurement_method",
    )
    if _boolean(mapping["development_only"], name="evaluator_config.development_only") is not True:
        raise ValueError("evaluator config must remain development-only")
    if (
        _boolean(
            mapping["accepted_scientific_evidence"],
            name="evaluator_config.accepted_scientific_evidence",
        )
        is not False
    ):
        raise ValueError("evaluator config cannot claim accepted scientific evidence")
    canonical = cast(dict[str, object], _json_clone(mapping))
    return canonical, protocol, budget, tuple(condition_specs)


def _parse_latency_summary(value: object, *, name: str) -> dict[str, float | None]:
    mapping = _mapping(value, name=name)
    _exact_keys(mapping, {"p50_ms", "p95_ms", "p99_ms"}, name=name)
    values: list[float | None] = []
    for field in ("p50_ms", "p95_ms", "p99_ms"):
        raw = mapping[field]
        values.append(
            None if raw is None else _nonnegative_float(raw, name=f"{name}.{field}")
        )
    if any(item is None for item in values) and not all(item is None for item in values):
        raise ValueError(f"{name} latency quantiles must be all measured or all unavailable")
    if all(item is not None for item in values):
        resolved = cast(list[float], values)
        if not resolved[0] <= resolved[1] <= resolved[2]:
            raise ValueError(f"{name} must satisfy p50 <= p95 <= p99")
    return dict(zip(("p50_ms", "p95_ms", "p99_ms"), values, strict=True))


def _parse_control_condition_report(
    value: object,
    *,
    index: int,
    spec: tuple[str, str, str, dict[str, object]],
    protocol: ContinuingControlProtocol,
    budget: ContinuingControlBudget,
    probe_examples_per_checkpoint: int,
    latency_method: str,
) -> dict[str, object]:
    role, expected_name, learner_type, _ = spec
    location = f"conditions[{index}]"
    mapping = _mapping(value, name=location)
    _exact_keys(
        mapping,
        {
            "name",
            "role",
            "learner_type",
            "predict_action_before_environment_outcome",
            "trace",
            "held_out_non_learning_probes",
            "metrics",
            "operations",
            "resources",
            "safety",
            "diagnostics",
        },
        name=location,
    )
    if mapping["name"] != expected_name or mapping["role"] != role:
        raise ValueError(f"{location} name or role does not match evaluator config")
    if mapping["learner_type"] != learner_type:
        raise ValueError(f"{location}.learner_type does not match evaluator config")
    if (
        _boolean(
            mapping["predict_action_before_environment_outcome"],
            name=f"{location}.predict_action_before_environment_outcome",
        )
        is not True
    ):
        raise ValueError(f"{location} must attest predict-before-outcome ordering")

    trace = _mapping(mapping["trace"], name=f"{location}.trace")
    _exact_keys(
        trace,
        {"rewards", "processed", "evaluator_regime_ids_in_learner_trace"},
        name=f"{location}.trace",
    )
    rewards = tuple(
        _finite_float(item, name=f"{location}.trace.rewards[{item_index}]")
        for item_index, item in enumerate(
            _list(trace["rewards"], name=f"{location}.trace.rewards")
        )
    )
    if len(rewards) != len(protocol.regime_schedule):
        raise ValueError(f"{location} reward trace length does not match protocol")
    processed = _list(trace["processed"], name=f"{location}.trace.processed")
    if processed != [True] * len(rewards):
        raise ValueError(f"{location} processed trace must contain only true values")
    if (
        _boolean(
            trace["evaluator_regime_ids_in_learner_trace"],
            name=f"{location}.trace.evaluator_regime_ids_in_learner_trace",
        )
        is not False
    ):
        raise ValueError(f"{location} learner trace cannot contain evaluator regime IDs")

    probes = _mapping(
        mapping["held_out_non_learning_probes"],
        name=f"{location}.held_out_non_learning_probes",
    )
    _exact_keys(
        probes,
        {
            "regime_order",
            "checkpoint_steps",
            "values",
            "snapshot_mutation_checks",
        },
        name=f"{location}.held_out_non_learning_probes",
    )
    if _string_list(
        probes["regime_order"],
        name=f"{location}.held_out_non_learning_probes.regime_order",
    ) != list(protocol.evaluator_regime_ids):
        raise ValueError(f"{location} held-out regime order does not match protocol")
    checkpoint_values = _list(
        probes["checkpoint_steps"],
        name=f"{location}.held_out_non_learning_probes.checkpoint_steps",
    )
    if checkpoint_values != list(protocol.checkpoint_steps):
        raise ValueError(f"{location} held-out checkpoints do not match protocol")
    if (
        _boolean(
            probes["snapshot_mutation_checks"],
            name=f"{location}.held_out_non_learning_probes.snapshot_mutation_checks",
        )
        is not True
    ):
        raise ValueError(f"{location} held-out probes must attest mutation checks")
    matrix = tuple(
        tuple(
            _finite_float(item, name=f"{location}.held_out.values")
            for item in _list(row, name=f"{location}.held_out.values")
        )
        for row in _list(probes["values"], name=f"{location}.held_out.values")
    )
    if len(matrix) != len(protocol.checkpoint_steps) or any(
        len(row) != len(protocol.evaluator_regime_ids) for row in matrix
    ):
        raise ValueError(f"{location} held-out matrix shape is invalid")
    expected_metrics = _control_metrics(rewards, matrix, protocol)
    if _json_clone(mapping["metrics"]) != _json_clone(expected_metrics):
        raise ValueError(f"{location}.metrics do not reconstruct from raw traces")

    operations = _mapping(mapping["operations"], name=f"{location}.operations")
    _exact_keys(operations, {"counts", "latency_ms"}, name=f"{location}.operations")
    counts = _mapping(operations["counts"], name=f"{location}.operations.counts")
    count_keys = {
        "processed_transitions",
        "delayed_transitions",
        "dropped_transitions",
        "decision_calls",
        "environment_calls",
        "update_calls",
        "held_out_probe_calls",
        "backward_calls",
        "backward_calls_available",
    }
    _exact_keys(counts, count_keys, name=f"{location}.operations.counts")
    transition_count = len(rewards)
    expected_probe_calls = len(protocol.checkpoint_steps) * probe_examples_per_checkpoint
    exact_counts = {
        "processed_transitions": transition_count,
        "dropped_transitions": 0,
        "decision_calls": transition_count,
        "environment_calls": transition_count,
        "update_calls": transition_count,
        "held_out_probe_calls": expected_probe_calls,
    }
    for field, expected in exact_counts.items():
        if _nonnegative_int(counts[field], name=f"{location}.counts.{field}") != expected:
            raise ValueError(f"{location}.counts.{field} is inconsistent")
    delayed = _nonnegative_int(
        counts["delayed_transitions"],
        name=f"{location}.counts.delayed_transitions",
    )
    if delayed > transition_count:
        raise ValueError(f"{location}.counts.delayed_transitions exceeds processed count")
    backward_available = _boolean(
        counts["backward_calls_available"],
        name=f"{location}.counts.backward_calls_available",
    )
    raw_backward = counts["backward_calls"]
    if backward_available:
        resolved_backward_calls = _nonnegative_int(
            raw_backward,
            name=f"{location}.counts.backward_calls",
        )
        if resolved_backward_calls > budget.backward_call_limit:
            raise ValueError(f"{location}.counts.backward_calls exceeds budget")
        backward_calls: int | None = resolved_backward_calls
    else:
        if raw_backward is not None:
            raise ValueError(f"{location}.counts.backward_calls must be None when unavailable")
        backward_calls = None

    latency = _mapping(operations["latency_ms"], name=f"{location}.operations.latency_ms")
    _exact_keys(
        latency,
        {"decision", "environment", "update", "held_out_probe", "measurement_method"},
        name=f"{location}.operations.latency_ms",
    )
    parsed_latency = {
        name: _parse_latency_summary(
            latency[name],
            name=f"{location}.operations.latency_ms.{name}",
        )
        for name in ("decision", "environment", "update", "held_out_probe")
    }
    if latency["measurement_method"] != latency_method:
        raise ValueError(f"{location} latency method does not match evaluator config")

    resources = _mapping(mapping["resources"], name=f"{location}.resources")
    resource_keys = {
        "current_persistent_state_bytes",
        "persistent_state_bytes_high_water",
        "current_state_scalar_count",
        "state_scalar_count_high_water",
        "trainable_parameter_count",
        "measurement_method",
    }
    _exact_keys(resources, resource_keys, name=f"{location}.resources")
    current_bytes = _positive_int(
        resources["current_persistent_state_bytes"],
        name=f"{location}.resources.current_persistent_state_bytes",
    )
    high_bytes = _positive_int(
        resources["persistent_state_bytes_high_water"],
        name=f"{location}.resources.persistent_state_bytes_high_water",
    )
    current_scalars = _positive_int(
        resources["current_state_scalar_count"],
        name=f"{location}.resources.current_state_scalar_count",
    )
    high_scalars = _positive_int(
        resources["state_scalar_count_high_water"],
        name=f"{location}.resources.state_scalar_count_high_water",
    )
    if current_bytes > high_bytes or high_bytes > budget.persistent_state_bytes_limit:
        raise ValueError(f"{location} persistent-state accounting is inconsistent")
    if current_scalars > high_scalars or high_scalars > budget.state_scalar_count_limit:
        raise ValueError(f"{location} scalar-state accounting is inconsistent")
    raw_parameter_count = resources["trainable_parameter_count"]
    parameter_count = (
        None
        if raw_parameter_count is None
        else _nonnegative_int(
            raw_parameter_count,
            name=f"{location}.resources.trainable_parameter_count",
        )
    )
    if budget.trainable_parameter_count_limit is not None:
        if parameter_count is None or parameter_count > budget.trainable_parameter_count_limit:
            raise ValueError(f"{location} parameter accounting does not satisfy budget")
    measurement_method = _string(
        resources["measurement_method"],
        name=f"{location}.resources.measurement_method",
    )

    safety = _mapping(mapping["safety"], name=f"{location}.safety")
    safety_keys = {
        "checks",
        "violations",
        "interventions",
        "near_misses",
        "cumulative_cost",
        "cumulative_near_miss_cost",
        "maximum_step_cost",
        "measurement_method",
    }
    _exact_keys(safety, safety_keys, name=f"{location}.safety")
    checks = _nonnegative_int(safety["checks"], name=f"{location}.safety.checks")
    if checks != transition_count:
        raise ValueError(f"{location}.safety.checks must match processed transitions")
    safety_counts = {
        field: _nonnegative_int(safety[field], name=f"{location}.safety.{field}")
        for field in ("violations", "interventions", "near_misses")
    }
    if any(value > checks for value in safety_counts.values()):
        raise ValueError(f"{location} safety event count exceeds checks")
    cumulative_cost = _nonnegative_float(
        safety["cumulative_cost"],
        name=f"{location}.safety.cumulative_cost",
    )
    near_cost = _nonnegative_float(
        safety["cumulative_near_miss_cost"],
        name=f"{location}.safety.cumulative_near_miss_cost",
    )
    maximum_cost = _nonnegative_float(
        safety["maximum_step_cost"],
        name=f"{location}.safety.maximum_step_cost",
    )
    if near_cost > cumulative_cost or maximum_cost > cumulative_cost:
        raise ValueError(f"{location} cumulative safety costs are inconsistent")
    if near_cost > 0.0 and safety_counts["near_misses"] == 0:
        raise ValueError(f"{location} near-miss cost requires a near-miss event")
    safety_method = _string(
        safety["measurement_method"],
        name=f"{location}.safety.measurement_method",
    )

    raw_diagnostics = mapping["diagnostics"]
    diagnostics: dict[str, DiagnosticScalar] | None = None
    if raw_diagnostics is not None:
        diagnostic_mapping = _mapping(raw_diagnostics, name=f"{location}.diagnostics")
        diagnostics = cast(dict[str, DiagnosticScalar], dict(diagnostic_mapping))
        _validate_diagnostics(diagnostics, name=f"{location}.diagnostics")

    return {
        "name": expected_name,
        "role": role,
        "learner_type": learner_type,
        "predict_action_before_environment_outcome": True,
        "trace": {
            "rewards": list(rewards),
            "processed": [True] * transition_count,
            "evaluator_regime_ids_in_learner_trace": False,
        },
        "held_out_non_learning_probes": {
            "regime_order": list(protocol.evaluator_regime_ids),
            "checkpoint_steps": list(protocol.checkpoint_steps),
            "values": [list(row) for row in matrix],
            "snapshot_mutation_checks": True,
        },
        "metrics": expected_metrics,
        "operations": {
            "counts": {
                **exact_counts,
                "delayed_transitions": delayed,
                "backward_calls": backward_calls,
                "backward_calls_available": backward_available,
            },
            "latency_ms": {
                **parsed_latency,
                "measurement_method": latency_method,
            },
        },
        "resources": {
            "current_persistent_state_bytes": current_bytes,
            "persistent_state_bytes_high_water": high_bytes,
            "current_state_scalar_count": current_scalars,
            "state_scalar_count_high_water": high_scalars,
            "trainable_parameter_count": parameter_count,
            "measurement_method": measurement_method,
        },
        "safety": {
            "checks": checks,
            **safety_counts,
            "cumulative_cost": cumulative_cost,
            "cumulative_near_miss_cost": near_cost,
            "maximum_step_cost": maximum_cost,
            "measurement_method": safety_method,
        },
        "diagnostics": diagnostics,
    }


def _reconstruct_control_report(
    report: Mapping[str, object],
    *,
    expected_evaluator_config: Mapping[str, object] | None,
) -> dict[str, object]:
    _exact_keys(
        report,
        {
            "schema_version",
            "acceptance_status",
            "interpretation",
            "run_id",
            "metric_definitions",
            "evaluator_config",
            "evaluator_identity",
            "protocol",
            "comparison_contract",
            "conditions",
            "limitations",
            "accepted_scientific_evidence",
        },
        name="report",
    )
    if report["schema_version"] != CONTROL_REPORT_SCHEMA:
        raise ValueError("report schema_version is invalid")
    if report["acceptance_status"] != ACCEPTANCE_STATUS:
        raise ValueError("report acceptance_status must remain not-assessed")
    if report["interpretation"] != REPORT_INTERPRETATION:
        raise ValueError("report interpretation is invalid")
    metric_definitions = _mapping(
        report["metric_definitions"],
        name="metric_definitions",
    )
    if metric_definitions != CONTROL_METRIC_DEFINITIONS:
        raise ValueError("report metric_definitions are invalid")
    evaluator_config, protocol, budget, specs = _parse_evaluator_config(
        report["evaluator_config"]
    )
    if expected_evaluator_config is not None and evaluator_config != _json_clone(
        expected_evaluator_config
    ):
        raise ValueError("report evaluator_config does not match expected evaluator")
    run_id = _versioned_identifier(report["run_id"], name="report.run_id")
    if run_id != evaluator_config["run_id"]:
        raise ValueError("report run_id does not match evaluator config")

    identity = _mapping(report["evaluator_identity"], name="evaluator_identity")
    _exact_keys(
        identity,
        {
            "evaluator_config_sha256",
            "environment_config_sha256",
            "condition_config_sha256",
            "probe_sha256",
        },
        name="evaluator_identity",
    )
    expected_condition_digests = {name: _digest(config) for _, name, _, config in specs}
    raw_condition_digests = _mapping(
        identity["condition_config_sha256"],
        name="evaluator_identity.condition_config_sha256",
    )
    parsed_condition_digests = {
        key: _sha256(value, name=f"evaluator_identity.condition_config_sha256.{key}")
        for key, value in raw_condition_digests.items()
    }
    expected_identity = {
        "evaluator_config_sha256": _digest(evaluator_config),
        "environment_config_sha256": _digest(evaluator_config["environment"]),
        "condition_config_sha256": expected_condition_digests,
        "probe_sha256": evaluator_config["probe_sha256"],
    }
    _sha256(identity["evaluator_config_sha256"], name="evaluator_config_sha256")
    _sha256(identity["environment_config_sha256"], name="environment_config_sha256")
    _sha256(identity["probe_sha256"], name="probe_sha256")
    observed_identity = {
        "evaluator_config_sha256": identity["evaluator_config_sha256"],
        "environment_config_sha256": identity["environment_config_sha256"],
        "condition_config_sha256": parsed_condition_digests,
        "probe_sha256": identity["probe_sha256"],
    }
    if observed_identity != expected_identity:
        raise ValueError("report evaluator identity digests do not match embedded config")

    protocol_report = _mapping(report["protocol"], name="protocol")
    expected_protocol_report = {
        **cast(
            dict[str, object],
            _json_clone(dataclasses.asdict(protocol)),
        ),
        "regime_metadata_is_evaluator_only": True,
    }
    if protocol_report != expected_protocol_report:
        raise ValueError("report protocol does not match evaluator config")

    comparison = _mapping(report["comparison_contract"], name="comparison_contract")
    expected_comparison = {
        "candidate": specs[0][1],
        "baselines": [spec[1] for spec in specs[1:]],
        "shared_ceiling_budget": cast(dict[str, object], _json_clone(dataclasses.asdict(budget))),
        "shared_ceiling_verified": True,
        "realized_compute_memory_parity_verified": False,
        "limitation": (
            "Conditions share hard ceilings and report realized usage, but this runner "
            "does not pad or otherwise establish equal realized compute or memory."
        ),
    }
    if comparison != expected_comparison:
        raise ValueError("report comparison contract does not match evaluator config")

    raw_conditions = _list(report["conditions"], name="conditions")
    if len(raw_conditions) != len(specs):
        raise ValueError("report condition count does not match evaluator config")
    probe_examples = _positive_int(
        evaluator_config["probe_examples_per_checkpoint"],
        name="probe_examples_per_checkpoint",
    )
    latency_method = _string(
        evaluator_config["latency_measurement_method"],
        name="latency_measurement_method",
    )
    conditions = [
        _parse_control_condition_report(
            value,
            index=index,
            spec=spec,
            protocol=protocol,
            budget=budget,
            probe_examples_per_checkpoint=probe_examples,
            latency_method=latency_method,
        )
        for index, (value, spec) in enumerate(zip(raw_conditions, specs, strict=True))
    ]
    limitations = _string_list(report["limitations"], name="limitations")
    if limitations != list(CONTROL_REPORT_LIMITATIONS):
        raise ValueError("report limitations are invalid or incomplete")
    if (
        _boolean(
            report["accepted_scientific_evidence"],
            name="accepted_scientific_evidence",
        )
        is not False
    ):
        raise ValueError("control report cannot claim accepted scientific evidence")
    reconstructed = {
        "schema_version": CONTROL_REPORT_SCHEMA,
        "acceptance_status": ACCEPTANCE_STATUS,
        "interpretation": REPORT_INTERPRETATION,
        "run_id": run_id,
        "metric_definitions": dict(CONTROL_METRIC_DEFINITIONS),
        "evaluator_config": evaluator_config,
        "evaluator_identity": expected_identity,
        "protocol": expected_protocol_report,
        "comparison_contract": expected_comparison,
        "conditions": conditions,
        "limitations": limitations,
        "accepted_scientific_evidence": False,
    }
    if _json_clone(reconstructed) != _json_clone(report):
        raise ValueError("control report is not canonical")
    return cast(dict[str, object], _json_clone(reconstructed))


def validate_continual_control_report(
    report: Mapping[str, object],
    *,
    expected_evaluator_config: Mapping[str, object] | None = None,
) -> ControlReportValidation:
    """Strictly reconstruct a development report and verify every identity digest."""
    try:
        _reconstruct_control_report(
            report,
            expected_evaluator_config=expected_evaluator_config,
        )
    except (TypeError, ValueError) as error:
        return ControlReportValidation(False, (str(error),))
    return ControlReportValidation(True, ())


def continual_control_report_json(report: Mapping[str, object]) -> str:
    """Return canonical JSON only for a strictly valid control report."""
    validation = validate_continual_control_report(report)
    if not validation.valid:
        raise ValueError("invalid continual-control report: " + "; ".join(validation.errors))
    return json.dumps(
        report,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"


def load_continual_control_report(
    path: str | Path,
    *,
    expected_evaluator_config: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Load strict JSON, reject duplicate/nonfinite content, and reconstruct it."""
    raw = json.loads(
        Path(path).read_text(encoding="utf-8"),
        parse_constant=_reject_nonstandard_json_constant,
        object_pairs_hook=_reject_duplicate_keys,
    )
    mapping = _mapping(raw, name="report")
    return _reconstruct_control_report(
        mapping,
        expected_evaluator_config=expected_evaluator_config,
    )


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be an object with string keys")
    return value


def _list(value: object, *, name: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be an array")
    return value


def _observation_from_json(value: object, *, name: str) -> tuple[float, ...]:
    return _observation(
        tuple(_list(value, name=name)),
        name=name,
    )


def _decision_id_from_json(value: object, *, name: str) -> DecisionId:
    return _decision_id(tuple(_list(value, name=name)), name=name)


def _shape_from_json(value: object, *, name: str) -> tuple[int, ...]:
    return tuple(
        _nonnegative_int(item, name=f"{name}[{index}]")
        for index, item in enumerate(_list(value, name=name))
    )


def _decision_from_config(value: object, *, name: str) -> ControlDecision:
    mapping = _mapping(value, name=name)
    if set(mapping) != {"observation", "action", "decision_id", "armed"}:
        raise ValueError(f"{name} keys are invalid")
    if not isinstance(mapping["armed"], bool):
        raise ValueError(f"{name}.armed must be boolean")
    return ControlDecision(
        observation=_observation_from_json(mapping["observation"], name=f"{name}.observation"),
        action=_nonnegative_int(mapping["action"], name=f"{name}.action"),
        decision_id=_decision_id_from_json(
            mapping["decision_id"],
            name=f"{name}.decision_id",
        ),
        armed=mapping["armed"],
    )


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
    "ACCEPTANCE_STATUS",
    "CONTROL_CHECKPOINT_SCHEMA",
    "CONTROL_EVALUATOR_SCHEMA",
    "CONTROL_METRIC_DEFINITIONS",
    "CONTROL_PROTOCOL_SCHEMA",
    "CONTROL_REPORT_SCHEMA",
    "CONTROL_REPORT_LIMITATIONS",
    "ControlReportValidation",
    "ControlDecision",
    "ControlEnvironmentUpdate",
    "ControlLearnerUpdate",
    "ControlProbe",
    "ControlResourceUsage",
    "ControlTransition",
    "ContinualControlEvaluator",
    "ContinualControlRunState",
    "ContinuingControlBudget",
    "ContinuingControlEnvironment",
    "ContinuingControlLearner",
    "ContinuingControlProtocol",
    "FrozenActionControlBaseline",
    "PrototypeAgentControlAdapter",
    "REPORT_INTERPRETATION",
    "RunningRewardBanditControlBaseline",
    "continual_control_report_json",
    "load_continual_control_report",
    "validate_continual_control_report",
]
