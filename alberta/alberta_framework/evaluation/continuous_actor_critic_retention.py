# mypy: disable-error-code="attr-defined,call-arg,no-any-return,no-untyped-call"
"""Strict development-only continuous actor/critic recurrence diagnostics.

The evaluator owns a fixed bounded one-action-dimensional continuing A/B/A
schedule, preferred action centers, reward function, reference differential
value fixtures, phase labels, and recurrence labels.  The learner-visible
boundary is only its observation, exact cached action, the scalar reward
realized from that action, and the next observation.

Reports retain raw decision, density, correction, value, action-error,
plasticity, trace, saturation, and activity diagnostics.  Differential values
have an arbitrary additive gauge.  At every event the evaluator evaluates the
pre-update critic on all four canonical cases under that single parameter
state, centers that vector, and selects the current case.  Raw vectors and
their means remain present.

This is a source-bound development diagnostic with assessment status
``not-assessed``.  It applies no threshold and makes no transfer, retention,
efficacy, calibration, SOTA, off-policy convergence, state-distribution
correction, candidate-update safety-audit, paper-defined delight, or
KondoSparseActor backward-execution claim.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import numbers
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal, cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array

from alberta_framework.core.continuous_average_reward_actor_critic import (
    TRANSFORMED_LOG_DENSITY_MAX_ULPS,
    ContinuousAverageRewardActorCriticAgent,
    ContinuousAverageRewardActorCriticState,
    diagonal_gaussian_target_behavior_ratio,
    float32_ulp_distance,
    transformed_diagonal_gaussian_log_density,
)

CONTINUOUS_ACTOR_CRITIC_RETENTION_CONFIG_SCHEMA = (
    "alberta.continuous-actor-critic-retention.config.v2"
)
CONTINUOUS_ACTOR_CRITIC_RETENTION_PROTOCOL_SCHEMA = (
    "alberta.continuous-actor-critic-retention.protocol.v2"
)
CONTINUOUS_ACTOR_CRITIC_RETENTION_REPORT_SCHEMA = (
    "alberta.continuous-actor-critic-retention.report.v2"
)
CONTINUOUS_ACTOR_CRITIC_RETENTION_CHECKPOINT_SCHEMA = (
    "alberta.continuous-actor-critic-retention.snapshot.v2"
)
DEVELOPMENT_STATUS = "development-only-not-assessed"
REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATHS = (
    Path("alberta_framework/core/continuous_average_reward_actor_critic.py"),
    Path("alberta_framework/core/optimizers.py"),
    Path("alberta_framework/core/types.py"),
    Path("alberta_framework/evaluation/continuous_actor_critic_retention.py"),
)

_PHASE_COUNT = 3
_EVENTS_PER_PHASE = 4
_EVENT_COUNT = _PHASE_COUNT * _EVENTS_PER_PHASE
_INT32_MAX = 2**31 - 1
_ABSOLUTE_STATE_BYTE_LIMIT = 4_000_000
_ABSOLUTE_REPORT_BYTE_LIMIT = 16_000_000
_ABSOLUTE_TRACE_SCALAR_LIMIT = 20_000
_ABSOLUTE_CHECKPOINT_BYTE_LIMIT = 16_000_000
_IDENTIFIER = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}\Z")
_EVENT_ORDER = (
    "cached-decision-consumed-before-update",
    "critic-predicted-before-reward-and-update",
    "scalar-reward-realized-from-cached-action",
    "atomic-update-committed-before-next-decision",
    "successor-sampled-on-committed-parameters",
)
_LIMITATIONS = (
    "development diagnostics only; assessment status is not-assessed",
    "one fixed bounded A/B/A trace does not establish transfer or retention",
    "preferred action centers and rewards are evaluator fixtures, not learner inputs",
    "same-state four-case centering removes only the additive differential-value gauge",
    "sampled return is one undiscounted continuing trace and is not an efficacy estimate",
    "exact action-likelihood correction does not correct state-distribution mismatch",
    "no off-policy convergence or external-validity claim is made",
    "plasticity, activity, churn, and saturation are diagnostics rather than success gates",
    "structural validation needs the separately bound snapshot for exact live replay",
    "no threshold, calibration, scientific promotion, SOTA, or completion claim follows",
    "the separate candidate-update safety audit is not performed",
    "paper-defined delight (advantage times action surprisal) is not computed",
    "KondoSparseActor backward execution is not performed",
)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _strict_json_equal(actual: object, expected: object) -> bool:
    if expected is None:
        return actual is None
    if type(expected) in {bool, int, float, str}:
        return type(actual) is type(expected) and actual == expected
    if isinstance(expected, list):
        return (
            type(actual) is list
            and len(actual) == len(expected)
            and all(
                _strict_json_equal(left, right)
                for left, right in zip(actual, expected, strict=True)
            )
        )
    if isinstance(expected, Mapping):
        return (
            isinstance(actual, Mapping)
            and set(actual) == set(expected)
            and all(_strict_json_equal(actual[key], expected[key]) for key in expected)
        )
    return False


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise ValueError(f"{name} must be an object with string keys")
    return cast(Mapping[str, object], value)


def _list(value: object, *, name: str) -> list[object]:
    if type(value) is not list:
        raise ValueError(f"{name} must be a canonical JSON array")
    return cast(list[object], value)


def _identifier(value: object, *, name: str) -> str:
    if type(value) is not str or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase canonical identifier")
    return value


def _finite_float(value: object, *, name: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{name} must be a canonical finite JSON float")
    return value


def _finite_config_float(value: object, *, name: str, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum}")
    return result


def _nonnegative_int(value: object, *, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a canonical non-negative integer")
    return value


def _positive_int(value: object, *, name: str) -> int:
    result = _nonnegative_int(value, name=name)
    if result == 0:
        raise ValueError(f"{name} must be positive")
    return result


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def continuous_actor_critic_retention_source_snapshot(
    root: Path = REPO_ROOT,
) -> dict[str, str]:
    """Hash the complete local source closure used by this evaluator."""
    return {relative.as_posix(): _file_sha256(root / relative) for relative in SOURCE_PATHS}


@dataclasses.dataclass(frozen=True)
class ContinuousActorCriticRetentionConfig:
    """Execution choice and hard fixed-protocol resource bounds."""

    recovery_window: int = 2
    max_phases: int = _PHASE_COUNT
    max_events: int = _EVENT_COUNT
    max_initial_snapshot_bytes: int = 2_000_000
    max_final_state_bytes: int = 2_000_000
    max_report_bytes: int = 4_000_000
    max_trace_scalar_values: int = 10_000
    activity_epsilon: float = 1.0e-8
    execution_mode: Literal["eager", "jit"] = "jit"

    def __post_init__(self) -> None:
        _positive_int(self.recovery_window, name="recovery_window")
        if self.recovery_window > _EVENTS_PER_PHASE:
            raise ValueError("recovery_window cannot exceed one fixed phase")
        _positive_int(self.max_phases, name="max_phases")
        if self.max_phases != _PHASE_COUNT:
            raise ValueError("max_phases must equal the fixed protocol phase count")
        _positive_int(self.max_events, name="max_events")
        if self.max_events != _EVENT_COUNT:
            raise ValueError("max_events must equal the fixed protocol event count")
        _positive_int(self.max_initial_snapshot_bytes, name="max_initial_snapshot_bytes")
        _positive_int(self.max_final_state_bytes, name="max_final_state_bytes")
        _positive_int(self.max_report_bytes, name="max_report_bytes")
        _positive_int(self.max_trace_scalar_values, name="max_trace_scalar_values")
        if self.max_initial_snapshot_bytes > _ABSOLUTE_STATE_BYTE_LIMIT:
            raise ValueError("max_initial_snapshot_bytes exceeds the absolute hard limit")
        if self.max_final_state_bytes > _ABSOLUTE_STATE_BYTE_LIMIT:
            raise ValueError("max_final_state_bytes exceeds the absolute hard limit")
        if self.max_report_bytes > _ABSOLUTE_REPORT_BYTE_LIMIT:
            raise ValueError("max_report_bytes exceeds the absolute hard limit")
        if self.max_trace_scalar_values > _ABSOLUTE_TRACE_SCALAR_LIMIT:
            raise ValueError("max_trace_scalar_values exceeds the absolute hard limit")
        epsilon = _finite_config_float(
            self.activity_epsilon,
            name="activity_epsilon",
            minimum=float(np.finfo(np.float32).tiny),
        )
        if epsilon > 1.0:
            raise ValueError("activity_epsilon must be <= 1")
        if self.execution_mode not in {"eager", "jit"}:
            raise ValueError("execution_mode must be 'eager' or 'jit'")
        object.__setattr__(self, "activity_epsilon", epsilon)

    def to_config(self) -> dict[str, object]:
        return {
            "schema": CONTINUOUS_ACTOR_CRITIC_RETENTION_CONFIG_SCHEMA,
            "type": type(self).__name__,
            "development_status": DEVELOPMENT_STATUS,
            "assessment_status": "not-assessed",
            "development_only": True,
            "scientific_promotion_allowed": False,
            "performance_thresholds_applied": False,
            "retention_claimed": False,
            "transfer_claimed": False,
            "efficacy_claimed": False,
            "calibration_claimed": False,
            "sota_claimed": False,
            "off_policy_state_distribution_correction_claimed": False,
            "off_policy_convergence_claimed": False,
            "candidate_update_safety_audit_performed": False,
            "paper_defined_delight_computed": False,
            "kondo_sparse_actor_backward_executed": False,
            "recovery_window": self.recovery_window,
            "max_phases": self.max_phases,
            "max_events": self.max_events,
            "max_initial_snapshot_bytes": self.max_initial_snapshot_bytes,
            "max_final_state_bytes": self.max_final_state_bytes,
            "max_report_bytes": self.max_report_bytes,
            "max_trace_scalar_values": self.max_trace_scalar_values,
            "activity_epsilon": self.activity_epsilon,
            "execution_mode": self.execution_mode,
        }

    @classmethod
    def from_config(cls, payload: Mapping[str, object]) -> ContinuousActorCriticRetentionConfig:
        expected_fields = set(cls().to_config())
        if set(payload) != expected_fields:
            raise ValueError("continuous actor/critic retention config fields do not match v2")
        fixed = cls().to_config()
        variable = {
            "recovery_window",
            "max_phases",
            "max_events",
            "max_initial_snapshot_bytes",
            "max_final_state_bytes",
            "max_report_bytes",
            "max_trace_scalar_values",
            "activity_epsilon",
            "execution_mode",
        }
        for name, expected in fixed.items():
            if name not in variable and not _strict_json_equal(payload.get(name), expected):
                raise ValueError(f"continuous actor/critic retention config {name} is invalid")
        mode = payload.get("execution_mode")
        if mode not in {"eager", "jit"}:
            raise ValueError("continuous actor/critic retention execution_mode is invalid")
        result = cls(
            recovery_window=_positive_int(payload.get("recovery_window"), name="recovery_window"),
            max_phases=_positive_int(payload.get("max_phases"), name="max_phases"),
            max_events=_positive_int(payload.get("max_events"), name="max_events"),
            max_initial_snapshot_bytes=_positive_int(
                payload.get("max_initial_snapshot_bytes"), name="max_initial_snapshot_bytes"
            ),
            max_final_state_bytes=_positive_int(
                payload.get("max_final_state_bytes"), name="max_final_state_bytes"
            ),
            max_report_bytes=_positive_int(
                payload.get("max_report_bytes"), name="max_report_bytes"
            ),
            max_trace_scalar_values=_positive_int(
                payload.get("max_trace_scalar_values"), name="max_trace_scalar_values"
            ),
            activity_epsilon=_finite_float(
                payload.get("activity_epsilon"), name="activity_epsilon"
            ),
            execution_mode=mode,
        )
        if not _strict_json_equal(dict(payload), result.to_config()):
            raise ValueError("continuous actor/critic retention config is noncanonical")
        return result


@dataclasses.dataclass(frozen=True)
class ContinuousActorCriticRetentionPhase:
    """Evaluator-only contiguous phase annotation."""

    phase_id: str
    event_count: int
    recurrence_of_phase_id: str | None = None

    def __post_init__(self) -> None:
        _identifier(self.phase_id, name="phase_id")
        _positive_int(self.event_count, name="event_count")
        if self.recurrence_of_phase_id is not None:
            _identifier(self.recurrence_of_phase_id, name="recurrence_of_phase_id")
            if self.recurrence_of_phase_id == self.phase_id:
                raise ValueError("a phase cannot recur from itself")

    def to_config(self) -> dict[str, object]:
        return {
            "phase_id": self.phase_id,
            "event_count": self.event_count,
            "recurrence_of_phase_id": self.recurrence_of_phase_id,
            "learner_visible": False,
        }


@dataclasses.dataclass(frozen=True)
class ContinuousActorCriticRetentionEvent:
    """Evaluator fixture for one cached-action continuing transition."""

    event_id: str
    phase_id: str
    case_id: str
    observation: tuple[float, ...]
    next_observation: tuple[float, ...]
    preferred_action_center: float
    reference_value_target: float

    def __post_init__(self) -> None:
        _identifier(self.event_id, name="event_id")
        _identifier(self.phase_id, name="phase_id")
        _identifier(self.case_id, name="case_id")
        if len(self.observation) != 2 or len(self.next_observation) != 2:
            raise ValueError("fixed protocol observations must have dimension two")
        for name, values in (
            ("observation", self.observation),
            ("next_observation", self.next_observation),
        ):
            if any(type(item) is not float or not math.isfinite(item) for item in values):
                raise ValueError(f"{name} must contain finite floats")
        if type(self.preferred_action_center) is not float or not math.isfinite(
            self.preferred_action_center
        ):
            raise ValueError("preferred_action_center must be a finite float")
        if not -1.0 < self.preferred_action_center < 1.0:
            raise ValueError("preferred_action_center must be strictly inside [-1, 1]")
        if type(self.reference_value_target) is not float or not math.isfinite(
            self.reference_value_target
        ):
            raise ValueError("reference_value_target must be a finite float")

    def realized_reward(self, action: float) -> float:
        """Evaluate the hidden quadratic reward and round to learner float32."""
        if not math.isfinite(action) or not -1.0 <= action <= 1.0:
            raise ValueError("cached action must be finite and bounded")
        return float(np.float32(1.0 - (action - self.preferred_action_center) ** 2))

    def to_config(self) -> dict[str, object]:
        return {
            "event_id": self.event_id,
            "phase_id": self.phase_id,
            "case_id": self.case_id,
            "observation": list(self.observation),
            "next_observation": list(self.next_observation),
            "preferred_action_center": self.preferred_action_center,
            "reference_value_target": self.reference_value_target,
            "reward_function": "float32(1-(cached_action-preferred_action_center)^2)",
            "phase_id_learner_visible": False,
            "targets_learner_visible": False,
            "reward_function_learner_visible": False,
            "realized_scalar_reward_learner_visible_after_action": True,
        }


@dataclasses.dataclass(frozen=True)
class ContinuousActorCriticRetentionProtocol:
    """The sole accepted fixed continuous continuing A/B/A protocol."""

    protocol_id: str
    phases: tuple[ContinuousActorCriticRetentionPhase, ...]
    events: tuple[ContinuousActorCriticRetentionEvent, ...]
    learner_visible_fields: tuple[str, ...] = (
        "observation",
        "exact_cached_action",
        "realized_scalar_reward_after_action",
        "next_observation",
    )
    evaluator_only_fields: tuple[str, ...] = (
        "event_id",
        "phase_id",
        "case_id",
        "preferred_action_center",
        "reward_function",
        "reference_value_target",
    )

    def __post_init__(self) -> None:
        _identifier(self.protocol_id, name="protocol_id")
        if len(self.phases) != _PHASE_COUNT or len(self.events) != _EVENT_COUNT:
            raise ValueError("continuous actor/critic protocol must retain fixed 3x4 shape")
        if self.learner_visible_fields != (
            "observation",
            "exact_cached_action",
            "realized_scalar_reward_after_action",
            "next_observation",
        ):
            raise ValueError("learner-visible fields changed")
        if self.evaluator_only_fields != (
            "event_id",
            "phase_id",
            "case_id",
            "preferred_action_center",
            "reward_function",
            "reference_value_target",
        ):
            raise ValueError("evaluator-only fields changed")
        cursor = 0
        by_phase: dict[str, tuple[ContinuousActorCriticRetentionEvent, ...]] = {}
        seen: set[str] = set()
        for phase in self.phases:
            if phase.phase_id in seen:
                raise ValueError("phase identifiers must be unique")
            if (
                phase.recurrence_of_phase_id is not None
                and phase.recurrence_of_phase_id not in seen
            ):
                raise ValueError("recurrence must reference an earlier phase")
            chunk = self.events[cursor : cursor + phase.event_count]
            if len(chunk) != phase.event_count or any(
                event.phase_id != phase.phase_id for event in chunk
            ):
                raise ValueError("events must be contiguous in phase order")
            by_phase[phase.phase_id] = chunk
            seen.add(phase.phase_id)
            cursor += phase.event_count
        if cursor != len(self.events):
            raise ValueError("phase counts must cover every event")
        if len({event.event_id for event in self.events}) != len(self.events):
            raise ValueError("event identifiers must be unique")
        for index, event in enumerate(self.events):
            following = self.events[(index + 1) % len(self.events)]
            if event.next_observation != following.observation:
                raise ValueError("events must form one continuing observation cycle")
        for phase in self.phases:
            reference_id = phase.recurrence_of_phase_id
            if reference_id is None:
                continue
            old = by_phase[reference_id]
            new = by_phase[phase.phase_id]
            if any(
                (
                    left.case_id,
                    left.observation,
                    left.next_observation,
                    left.preferred_action_center,
                    left.reference_value_target,
                )
                != (
                    right.case_id,
                    right.observation,
                    right.next_observation,
                    right.preferred_action_center,
                    right.reference_value_target,
                )
                for left, right in zip(old, new, strict=True)
            ):
                raise ValueError("recurrence must repeat the exact ordered evaluator fixtures")

    def to_config(self) -> dict[str, object]:
        return {
            "schema": CONTINUOUS_ACTOR_CRITIC_RETENTION_PROTOCOL_SCHEMA,
            "type": type(self).__name__,
            "protocol_id": self.protocol_id,
            "development_status": DEVELOPMENT_STATUS,
            "assessment_status": "not-assessed",
            "action_dim": 1,
            "action_low": -1.0,
            "action_high": 1.0,
            "continuing": True,
            "fixed_schedule": True,
            "phase_labels_learner_visible": False,
            "targets_learner_visible": False,
            "reward_function_learner_visible": False,
            "learner_visible_fields": list(self.learner_visible_fields),
            "evaluator_only_fields": list(self.evaluator_only_fields),
            "phases": [phase.to_config() for phase in self.phases],
            "events": [event.to_config() for event in self.events],
        }

    @classmethod
    def from_config(cls, payload: Mapping[str, object]) -> ContinuousActorCriticRetentionProtocol:
        expected = canonical_continuous_actor_critic_retention_protocol()
        if not _strict_json_equal(dict(payload), expected.to_config()):
            raise ValueError("only the exact canonical continuous actor/critic protocol is valid")
        return expected


def canonical_continuous_actor_critic_retention_protocol() -> (
    ContinuousActorCriticRetentionProtocol
):
    """Construct the evaluator-owned bounded A/B/A continuing schedule."""
    observations = (
        (1.0, 0.0),
        (0.0, 1.0),
        (-1.0, 0.0),
        (0.0, -1.0),
    )
    centers_a = (-0.75, -0.25, 0.25, 0.75)
    centers_b = (0.75, 0.25, -0.25, -0.75)
    values_a = (0.3, -0.1, 0.1, -0.3)
    values_b = (-0.3, 0.1, -0.1, 0.3)

    def events_for(
        phase_id: str,
        prefix: str,
        centers: tuple[float, ...],
        targets: tuple[float, ...],
    ) -> tuple[ContinuousActorCriticRetentionEvent, ...]:
        return tuple(
            ContinuousActorCriticRetentionEvent(
                event_id=f"{prefix}-{index}",
                phase_id=phase_id,
                case_id=f"cycle-{index}",
                observation=observation,
                next_observation=observations[(index + 1) % len(observations)],
                preferred_action_center=center,
                reference_value_target=target,
            )
            for index, (observation, center, target) in enumerate(
                zip(observations, centers, targets, strict=True)
            )
        )

    return ContinuousActorCriticRetentionProtocol(
        protocol_id="continuous-average-reward-actor-critic-aba-v1",
        phases=(
            ContinuousActorCriticRetentionPhase("first-a", 4),
            ContinuousActorCriticRetentionPhase("interference-b", 4),
            ContinuousActorCriticRetentionPhase("return-a", 4, "first-a"),
        ),
        events=(
            *events_for("first-a", "a", centers_a, values_a),
            *events_for("interference-b", "b", centers_b, values_b),
            *events_for("return-a", "a-return", centers_a, values_a),
        ),
    )


def _numpy_leaf(leaf: object) -> np.ndarray:
    dtype = getattr(leaf, "dtype", None)
    if dtype is not None and jnp.issubdtype(dtype, jax.dtypes.prng_key):
        array = np.asarray(jr.key_data(cast(Array, leaf)))
    else:
        array = np.asarray(leaf)
    if array.dtype.hasobject:
        raise ValueError("continuous actor/critic state contains an object leaf")
    return np.ascontiguousarray(array)


def _leaf_descriptor(leaf: object) -> dict[str, object]:
    array = _numpy_leaf(leaf)
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "nbytes": int(array.nbytes),
        "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }


def _state_manifest(
    state: ContinuousAverageRewardActorCriticState,
) -> list[dict[str, object]]:
    return [_leaf_descriptor(leaf) for leaf in jax.tree.leaves(state)]


def frozen_continuous_actor_critic_state_sha256(
    state: ContinuousAverageRewardActorCriticState,
) -> str:
    """Hash every persistent state leaf, including the typed RNG key."""
    if not isinstance(state, ContinuousAverageRewardActorCriticState):
        raise TypeError("state must be ContinuousAverageRewardActorCriticState")
    return _canonical_sha256(_state_manifest(state))


def _state_bytes(state: ContinuousAverageRewardActorCriticState) -> int:
    return sum(
        _nonnegative_int(item["nbytes"], name="state leaf nbytes")
        for item in _state_manifest(state)
    )


def _parameter_manifest(parameters: Sequence[Array]) -> list[dict[str, object]]:
    return [_leaf_descriptor(parameter) for parameter in parameters]


def _actor_parameters(state: ContinuousAverageRewardActorCriticState) -> tuple[Array, ...]:
    return (
        state.actor_params.mean_weights,
        state.actor_params.mean_bias,
        state.actor_params.log_std,
    )


def _critic_parameters(state: ContinuousAverageRewardActorCriticState) -> tuple[Array, ...]:
    return (state.critic_params.weights, state.critic_params.bias)


def _actor_traces(state: ContinuousAverageRewardActorCriticState) -> tuple[Array, ...]:
    return (
        state.actor_trace.mean_weights,
        state.actor_trace.mean_bias,
        state.actor_trace.log_std,
    )


def _critic_traces(state: ContinuousAverageRewardActorCriticState) -> tuple[Array, ...]:
    return (state.critic_trace.weights, state.critic_trace.bias)


def _parameter_delta_l2(before: Sequence[Array], after: Sequence[Array]) -> float:
    if len(before) != len(after):
        raise ValueError("parameter structures do not match")
    squared = 0.0
    for left, right in zip(before, after, strict=True):
        left_array = np.asarray(left, dtype=np.float64)
        right_array = np.asarray(right, dtype=np.float64)
        if left_array.shape != right_array.shape:
            raise ValueError("parameter shapes do not match")
        difference = right_array - left_array
        squared += float(np.sum(difference * difference, dtype=np.float64))
    result = math.sqrt(squared)
    if not math.isfinite(result):
        raise ValueError("parameter delta must remain finite")
    return result


def _parameter_l2(parameters: Sequence[Array]) -> float:
    zeros = tuple(jnp.zeros_like(parameter) for parameter in parameters)
    return _parameter_delta_l2(zeros, parameters)


def _state_descriptor(
    agent: ContinuousAverageRewardActorCriticAgent,
    state: ContinuousAverageRewardActorCriticState,
) -> dict[str, object]:
    agent_config = agent.to_config()
    actor_manifest = _parameter_manifest(_actor_parameters(state))
    critic_manifest = _parameter_manifest(_critic_parameters(state))
    actor_trace_manifest = _parameter_manifest(_actor_traces(state))
    critic_trace_manifest = _parameter_manifest(_critic_traces(state))
    return {
        "agent_config": agent_config,
        "agent_config_sha256": _canonical_sha256(agent_config),
        "state_sha256": frozen_continuous_actor_critic_state_sha256(state),
        "state_bytes": _state_bytes(state),
        "observation_dim": int(state.actor_params.mean_weights.shape[1]),
        "action_dim": agent.config.action_dim,
        "action_low": list(cast(tuple[float, ...], agent.config.action_low)),
        "action_high": list(cast(tuple[float, ...], agent.config.action_high)),
        "decision_count": int(state.decision_count),
        "update_count": int(state.update_count),
        "average_reward": float(state.average_reward),
        "actor_parameters_sha256": _canonical_sha256(actor_manifest),
        "critic_parameters_sha256": _canonical_sha256(critic_manifest),
        "actor_traces_sha256": _canonical_sha256(actor_trace_manifest),
        "critic_traces_sha256": _canonical_sha256(critic_trace_manifest),
    }


def _validate_agent_state(
    agent: ContinuousAverageRewardActorCriticAgent,
    state: ContinuousAverageRewardActorCriticState,
    *,
    require_first_event: bool = True,
) -> None:
    if not isinstance(agent, ContinuousAverageRewardActorCriticAgent):
        raise TypeError("agent must be ContinuousAverageRewardActorCriticAgent")
    if not isinstance(state, ContinuousAverageRewardActorCriticState):
        raise TypeError("state must be ContinuousAverageRewardActorCriticState")
    if agent.config.action_dim != 1:
        raise ValueError("fixed continuous protocol requires action_dim=1")
    if agent.config.action_low != (-1.0,) or agent.config.action_high != (1.0,):
        raise ValueError("fixed continuous protocol requires exact action bounds [-1, 1]")
    if state.actor_params.mean_weights.shape != (1, 2):
        raise ValueError("fixed continuous protocol requires observation_dim=2")
    if not bool(np.asarray(state.last_sample.valid)):
        raise ValueError("continuous actor/critic snapshot must be started")
    if int(state.update_count) > agent.config.max_updates - _EVENT_COUNT:
        raise ValueError("continuous actor/critic snapshot lacks fixed update capacity")
    if int(state.decision_count) > _INT32_MAX - _EVENT_COUNT:
        raise ValueError("continuous actor/critic snapshot lacks decision counter capacity")
    try:
        json.loads(json.dumps(agent.checkpoint_payload(state), allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"continuous actor/critic snapshot core validation failed: {error}"
        ) from error
    if require_first_event:
        first = canonical_continuous_actor_critic_retention_protocol().events[0]
        expected = np.asarray(first.observation, dtype=np.float32)
        actual = np.asarray(state.last_sample.observation, dtype=np.float32)
        if actual.shape != expected.shape or actual.tobytes() != expected.tobytes():
            raise ValueError("continuous actor/critic snapshot does not own the first observation")


def _isolated_copy(
    agent: ContinuousAverageRewardActorCriticAgent,
    state: ContinuousAverageRewardActorCriticState,
) -> tuple[ContinuousAverageRewardActorCriticAgent, ContinuousAverageRewardActorCriticState]:
    payload = json.loads(json.dumps(agent.checkpoint_payload(state), allow_nan=False))
    return ContinuousAverageRewardActorCriticAgent.from_checkpoint_payload(payload)


def _require_bit_exact_float32(left: object, right: object, *, name: str) -> None:
    lhs = np.ascontiguousarray(np.asarray(left, dtype=np.float32))
    rhs = np.ascontiguousarray(np.asarray(right, dtype=np.float32))
    if lhs.shape != rhs.shape or lhs.tobytes() != rhs.tobytes():
        raise RuntimeError(f"{name} is not bit-exact at the cached decision boundary")


def _latent_ratio(
    latent: Array,
    mean: Array,
    target_std: Array,
    behavior_std: Array,
) -> Array:
    return diagonal_gaussian_target_behavior_ratio(
        latent,
        mean,
        target_std,
        behavior_std,
    )


def _all_float_fields_finite(record: Mapping[str, object]) -> bool:
    for value in record.values():
        if type(value) is float and not math.isfinite(value):
            return False
        if isinstance(value, list) and any(
            type(item) is float and not math.isfinite(item) for item in value
        ):
            return False
    return True


def _execute_trace(
    agent: ContinuousAverageRewardActorCriticAgent,
    state: ContinuousAverageRewardActorCriticState,
    config: ContinuousActorCriticRetentionConfig,
    protocol: ContinuousActorCriticRetentionProtocol,
) -> tuple[list[dict[str, object]], ContinuousAverageRewardActorCriticState]:
    isolated_agent, isolated = _isolated_copy(agent, state)
    initial_actor = _actor_parameters(isolated)
    initial_critic = _critic_parameters(isolated)
    previous_latent_mean_by_case: dict[str, float] = {}
    previous_std_by_case: dict[str, float] = {}
    trace: list[dict[str, object]] = []
    epsilon = config.activity_epsilon

    with jax.disable_jit(config.execution_mode == "eager"):
        for index, event in enumerate(protocol.events):
            observation = jnp.asarray(event.observation, dtype=jnp.float32)
            next_observation = jnp.asarray(event.next_observation, dtype=jnp.float32)
            _require_bit_exact_float32(
                isolated.last_sample.observation,
                observation,
                name="cached and scheduled observation",
            )
            decision = isolated.last_sample
            if not bool(decision.valid):
                raise RuntimeError("continuous actor/critic lost its cached decision")
            cached_pre_tanh = float(decision.pre_tanh_action[0])
            cached_action = float(decision.action[0])
            target_latent_mean = float(decision.target_mean[0])
            target_std = float(decision.target_std[0])
            behavior_latent_mean = target_latent_mean
            behavior_std = float(decision.behavior_std[0])
            target_log_density = float(decision.target_log_density)
            behavior_log_density = float(decision.behavior_log_density)
            exact_ratio = float(decision.target_behavior_ratio)

            fresh_mean, fresh_std = isolated_agent.target_policy_params(isolated, observation)
            _require_bit_exact_float32(decision.target_mean, fresh_mean, name="target mean")
            _require_bit_exact_float32(decision.target_std, fresh_std, name="target std")
            expected_behavior_std = fresh_std * jnp.asarray(
                isolated_agent.config.behavior_std_scale, dtype=jnp.float32
            )
            _require_bit_exact_float32(
                decision.behavior_std, expected_behavior_std, name="behavior std"
            )
            reconstructed_action_array = isolated_agent.squash_pre_tanh_action(
                decision.pre_tanh_action
            )
            _require_bit_exact_float32(
                reconstructed_action_array, decision.action, name="direct affine-tanh action"
            )
            reconstructed_target_log = transformed_diagonal_gaussian_log_density(
                decision.pre_tanh_action,
                decision.target_mean,
                decision.target_std,
                jnp.asarray([-1.0], dtype=jnp.float32),
                jnp.asarray([1.0], dtype=jnp.float32),
            )
            reconstructed_behavior_log = transformed_diagonal_gaussian_log_density(
                decision.pre_tanh_action,
                decision.target_mean,
                decision.behavior_std,
                jnp.asarray([-1.0], dtype=jnp.float32),
                jnp.asarray([1.0], dtype=jnp.float32),
            )
            target_density_ulp_distance = int(
                float32_ulp_distance(reconstructed_target_log, decision.target_log_density)
            )
            behavior_density_ulp_distance = int(
                float32_ulp_distance(reconstructed_behavior_log, decision.behavior_log_density)
            )
            if target_density_ulp_distance > TRANSFORMED_LOG_DENSITY_MAX_ULPS:
                raise RuntimeError("target transformed log density exceeds ULP contract")
            if behavior_density_ulp_distance > TRANSFORMED_LOG_DENSITY_MAX_ULPS:
                raise RuntimeError("behavior transformed log density exceeds ULP contract")
            reconstructed_ratio_array = _latent_ratio(
                decision.pre_tanh_action,
                decision.target_mean,
                decision.target_std,
                decision.behavior_std,
            )
            reconstructed_ratio = float(reconstructed_ratio_array)
            _require_bit_exact_float32(
                reconstructed_ratio_array,
                decision.target_behavior_ratio,
                name="exact latent target/behavior ratio",
            )

            phase_events = tuple(
                candidate for candidate in protocol.events if candidate.phase_id == event.phase_id
            )
            case_index = next(
                position
                for position, candidate in enumerate(phase_events)
                if candidate.event_id == event.event_id
            )
            critic_same_state_case_predictions_raw = [
                float(
                    isolated_agent.value(
                        isolated,
                        jnp.asarray(candidate.observation, dtype=jnp.float32),
                    )
                )
                for candidate in phase_events
            ]
            critic_same_state_case_prediction_mean = _mean(critic_same_state_case_predictions_raw)
            critic_same_state_case_predictions_centered = [
                prediction - critic_same_state_case_prediction_mean
                for prediction in critic_same_state_case_predictions_raw
            ]
            reference_value_targets_phase_raw = [
                candidate.reference_value_target for candidate in phase_events
            ]
            reference_value_target_phase_mean = _mean(reference_value_targets_phase_raw)
            reference_value_targets_phase_centered = [
                target - reference_value_target_phase_mean
                for target in reference_value_targets_phase_raw
            ]
            critic_prediction = critic_same_state_case_predictions_raw[case_index]
            critic_prediction_centered = critic_same_state_case_predictions_centered[case_index]
            reference_value_target_centered = reference_value_targets_phase_centered[case_index]
            critic_centered_error = critic_prediction_centered - reference_value_target_centered
            target_median_action = float(
                isolated_agent.squash_pre_tanh_action(decision.target_mean)[0]
            )
            realized_reward = event.realized_reward(cached_action)
            previous_latent_mean = previous_latent_mean_by_case.get(event.case_id)
            previous_std = previous_std_by_case.get(event.case_id)
            previous_latent_mean_by_case[event.case_id] = target_latent_mean
            previous_std_by_case[event.case_id] = target_std
            actor_before = _actor_parameters(isolated)
            critic_before = _critic_parameters(isolated)
            actor_trace_before = _actor_traces(isolated)
            critic_trace_before = _critic_traces(isolated)
            average_reward_before = float(isolated.average_reward)
            update_count_before = int(isolated.update_count)
            decision_count_before = int(isolated.decision_count)

            result = isolated_agent.update(
                isolated,
                jnp.asarray(realized_reward, dtype=jnp.float32),
                next_observation,
            )
            if not bool(result.accepted):
                raise RuntimeError("continuous actor/critic rejected a canonical transition")
            if float(result.value) != critic_prediction:
                raise RuntimeError("public update value disagrees with preupdate critic prediction")
            isolated = result.state
            post_mean_array, post_std_array = isolated_agent.target_policy_params(
                isolated, observation
            )
            post_latent_mean = float(post_mean_array[0])
            post_std = float(post_std_array[0])
            post_target_median_action = float(
                isolated_agent.squash_pre_tanh_action(post_mean_array)[0]
            )
            actor_update_l2 = _parameter_delta_l2(actor_before, _actor_parameters(isolated))
            critic_update_l2 = _parameter_delta_l2(critic_before, _critic_parameters(isolated))
            actor_trace_update_l2 = _parameter_delta_l2(actor_trace_before, _actor_traces(isolated))
            critic_trace_update_l2 = _parameter_delta_l2(
                critic_trace_before, _critic_traces(isolated)
            )
            actor_from_initial = _parameter_delta_l2(initial_actor, _actor_parameters(isolated))
            critic_from_initial = _parameter_delta_l2(initial_critic, _critic_parameters(isolated))
            action_error = cached_action - event.preferred_action_center
            target_median_action_error = target_median_action - event.preferred_action_center
            action_boundary_saturated = cached_action in {-1.0, 1.0}
            log_std = float(isolated.actor_params.log_std[0])
            log_std_saturated = log_std in {
                isolated_agent.config.target_log_std_min,
                isolated_agent.config.target_log_std_max,
            }
            next_decision = isolated.last_sample
            record: dict[str, object] = {
                "event_index": index,
                "event_id": event.event_id,
                "phase_id": event.phase_id,
                "case_id": event.case_id,
                "phase_id_learner_visible": False,
                "targets_learner_visible": False,
                "reward_function_learner_visible": False,
                "realized_scalar_reward_learner_visible_after_action": True,
                "event_order": list(_EVENT_ORDER),
                "observation": list(event.observation),
                "next_observation": list(event.next_observation),
                "preferred_action_center": event.preferred_action_center,
                "reward_function": "float32(1-(cached_action-preferred_action_center)^2)",
                "reference_value_target_raw": event.reference_value_target,
                "cached_pre_tanh_action": cached_pre_tanh,
                "cached_action": cached_action,
                "decision_target_latent_mean": target_latent_mean,
                "decision_behavior_latent_mean": behavior_latent_mean,
                "decision_target_std": target_std,
                "decision_behavior_std": behavior_std,
                "decision_target_log_density": target_log_density,
                "decision_behavior_log_density": behavior_log_density,
                "decision_target_behavior_ratio": exact_ratio,
                "direct_transform_reconstructed_action": float(reconstructed_action_array[0]),
                "direct_transform_reconstruction_abs_error": abs(
                    float(reconstructed_action_array[0]) - cached_action
                ),
                "target_log_density_reconstructed": float(reconstructed_target_log),
                "target_log_density_reconstruction_abs_error": abs(
                    float(reconstructed_target_log) - target_log_density
                ),
                "target_log_density_reconstruction_ulp_distance": (target_density_ulp_distance),
                "behavior_log_density_reconstructed": float(reconstructed_behavior_log),
                "behavior_log_density_reconstruction_abs_error": abs(
                    float(reconstructed_behavior_log) - behavior_log_density
                ),
                "behavior_log_density_reconstruction_ulp_distance": (behavior_density_ulp_distance),
                "rho_reconstructed_from_latent_gaussians": reconstructed_ratio,
                "rho_reconstruction_abs_error": abs(reconstructed_ratio - exact_ratio),
                "critic_same_state_case_predictions_raw": (critic_same_state_case_predictions_raw),
                "critic_same_state_case_prediction_mean": (critic_same_state_case_prediction_mean),
                "critic_same_state_case_predictions_centered": (
                    critic_same_state_case_predictions_centered
                ),
                "critic_prediction_raw": critic_prediction,
                "critic_prediction_same_state_centered": critic_prediction_centered,
                "reference_value_targets_phase_raw": reference_value_targets_phase_raw,
                "reference_value_target_phase_mean": reference_value_target_phase_mean,
                "reference_value_targets_phase_centered": (reference_value_targets_phase_centered),
                "reference_value_target_centered": reference_value_target_centered,
                "critic_same_state_centered_error": critic_centered_error,
                "critic_same_state_centered_squared_error": (
                    critic_centered_error * critic_centered_error
                ),
                "target_median_action": target_median_action,
                "target_median_action_error": target_median_action_error,
                "target_median_action_abs_error": abs(target_median_action_error),
                "sampled_action_error": action_error,
                "sampled_action_abs_error": abs(action_error),
                "sampled_action_squared_error": action_error * action_error,
                "realized_reward": realized_reward,
                "sampled_return_contribution": realized_reward,
                "target_latent_mean_churn_available": previous_latent_mean is not None,
                "target_latent_mean_churn_abs": (
                    0.0
                    if previous_latent_mean is None
                    else abs(target_latent_mean - previous_latent_mean)
                ),
                "target_std_churn_available": previous_std is not None,
                "target_std_churn_abs": (
                    0.0 if previous_std is None else abs(target_std - previous_std)
                ),
                "postupdate_target_latent_mean": post_latent_mean,
                "postupdate_target_std": post_std,
                "postupdate_target_median_action": post_target_median_action,
                "postupdate_target_latent_mean_change_abs": abs(
                    post_latent_mean - target_latent_mean
                ),
                "postupdate_target_std_change_abs": abs(post_std - target_std),
                "postupdate_target_median_action_change_abs": abs(
                    post_target_median_action - target_median_action
                ),
                "td_error": float(result.td_error),
                "average_reward_before_update": average_reward_before,
                "average_reward_after_update": float(result.average_reward),
                "average_reward_change_abs": abs(
                    float(result.average_reward) - average_reward_before
                ),
                "actor_parameter_update_l2": actor_update_l2,
                "critic_parameter_update_l2": critic_update_l2,
                "actor_trace_update_l2": actor_trace_update_l2,
                "critic_trace_update_l2": critic_trace_update_l2,
                "actor_parameter_delta_from_initial_l2": actor_from_initial,
                "critic_parameter_delta_from_initial_l2": critic_from_initial,
                "actor_trace_l2_after_update": _parameter_l2(_actor_traces(isolated)),
                "critic_trace_l2_after_update": _parameter_l2(_critic_traces(isolated)),
                "action_boundary_saturated": action_boundary_saturated,
                "log_std_boundary_saturated_after_update": log_std_saturated,
                "sampled_action_active": abs(cached_action) > epsilon,
                "target_latent_mean_active": abs(target_latent_mean) > epsilon,
                "target_median_action_active": abs(target_median_action) > epsilon,
                "actor_update_active": actor_update_l2 > epsilon,
                "critic_update_active": critic_update_l2 > epsilon,
                "update_accepted": True,
                "all_recorded_finite": True,
                "update_count_before": update_count_before,
                "update_count_after": int(isolated.update_count),
                "decision_count_before": decision_count_before,
                "decision_count_after": int(isolated.decision_count),
                "next_cached_pre_tanh_action": float(next_decision.pre_tanh_action[0]),
                "next_cached_action": float(next_decision.action[0]),
                "next_decision_target_latent_mean": float(next_decision.target_mean[0]),
                "next_target_median_action": float(
                    isolated_agent.squash_pre_tanh_action(next_decision.target_mean)[0]
                ),
                "next_decision_target_std": float(next_decision.target_std[0]),
                "next_decision_behavior_std": float(next_decision.behavior_std[0]),
                "next_decision_target_log_density": float(next_decision.target_log_density),
                "next_decision_behavior_log_density": float(next_decision.behavior_log_density),
                "next_decision_target_behavior_ratio": float(next_decision.target_behavior_ratio),
            }
            record["all_recorded_finite"] = _all_float_fields_finite(record)
            if not bool(record["all_recorded_finite"]):
                raise RuntimeError("continuous actor/critic produced non-finite diagnostics")
            if (
                _nonnegative_int(record["update_count_after"], name="record update_count_after")
                != update_count_before + 1
            ):
                raise RuntimeError("continuous actor/critic update counter ordering changed")
            if (
                _nonnegative_int(record["decision_count_after"], name="record decision_count_after")
                != decision_count_before + 1
            ):
                raise RuntimeError("continuous actor/critic decision counter ordering changed")
            following = protocol.events[(index + 1) % len(protocol.events)]
            _require_bit_exact_float32(
                isolated.last_sample.observation,
                np.asarray(following.observation, dtype=np.float32),
                name="successor observation",
            )
            trace.append(record)

    return trace, isolated


def _trace_float(record: Mapping[str, object], name: str) -> float:
    return _finite_float(record.get(name), name=name)


def _trace_bool(record: Mapping[str, object], name: str) -> bool:
    value = record.get(name)
    if type(value) is not bool:
        raise ValueError(f"{name} must be a canonical JSON boolean")
    return value


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty diagnostic sequence")
    result = float(sum(values) / len(values))
    if not math.isfinite(result):
        raise ValueError("diagnostic mean must remain finite")
    return result


def _phase_summary(
    records: Sequence[Mapping[str, object]],
    phase: ContinuousActorCriticRetentionPhase,
) -> dict[str, object]:
    if len(records) != phase.event_count:
        raise ValueError("phase records do not match the canonical phase count")
    centered_squared = [
        _trace_float(record, "critic_same_state_centered_squared_error") for record in records
    ]
    return {
        "phase_id": phase.phase_id,
        "recurrence_of_phase_id": phase.recurrence_of_phase_id,
        "event_count": len(records),
        "realized_return": float(
            sum(_trace_float(record, "realized_reward") for record in records)
        ),
        "mean_realized_reward": _mean(
            [_trace_float(record, "realized_reward") for record in records]
        ),
        "mean_sampled_action_abs_error": _mean(
            [_trace_float(record, "sampled_action_abs_error") for record in records]
        ),
        "mean_target_median_action_abs_error": _mean(
            [_trace_float(record, "target_median_action_abs_error") for record in records]
        ),
        "critic_same_state_centered_rmse": math.sqrt(_mean(centered_squared)),
        "critic_prediction_raw_mean": _mean(
            [_trace_float(record, "critic_prediction_raw") for record in records]
        ),
        "reference_value_target_raw_mean": _mean(
            [_trace_float(record, "reference_value_target_raw") for record in records]
        ),
        "mean_abs_td_error": _mean([abs(_trace_float(record, "td_error")) for record in records]),
        "mean_exact_target_behavior_ratio": _mean(
            [_trace_float(record, "decision_target_behavior_ratio") for record in records]
        ),
        "target_latent_mean_churn_available_count": sum(
            int(_trace_bool(record, "target_latent_mean_churn_available")) for record in records
        ),
        "mean_available_target_latent_mean_churn_abs": _mean(
            [
                _trace_float(record, "target_latent_mean_churn_abs")
                for record in records
                if _trace_bool(record, "target_latent_mean_churn_available")
            ]
            or [0.0]
        ),
        "target_std_churn_available_count": sum(
            int(_trace_bool(record, "target_std_churn_available")) for record in records
        ),
        "mean_available_target_std_churn_abs": _mean(
            [
                _trace_float(record, "target_std_churn_abs")
                for record in records
                if _trace_bool(record, "target_std_churn_available")
            ]
            or [0.0]
        ),
        "mean_actor_parameter_update_l2": _mean(
            [_trace_float(record, "actor_parameter_update_l2") for record in records]
        ),
        "mean_critic_parameter_update_l2": _mean(
            [_trace_float(record, "critic_parameter_update_l2") for record in records]
        ),
        "action_boundary_saturation_count": sum(
            int(_trace_bool(record, "action_boundary_saturated")) for record in records
        ),
        "log_std_boundary_saturation_count": sum(
            int(_trace_bool(record, "log_std_boundary_saturated_after_update"))
            for record in records
        ),
        "sampled_action_activity_count": sum(
            int(_trace_bool(record, "sampled_action_active")) for record in records
        ),
        "target_latent_mean_activity_count": sum(
            int(_trace_bool(record, "target_latent_mean_active")) for record in records
        ),
        "target_median_action_activity_count": sum(
            int(_trace_bool(record, "target_median_action_active")) for record in records
        ),
        "actor_update_activity_count": sum(
            int(_trace_bool(record, "actor_update_active")) for record in records
        ),
        "critic_update_activity_count": sum(
            int(_trace_bool(record, "critic_update_active")) for record in records
        ),
    }


def reconstruct_continuous_actor_critic_retention_summary(
    event_trace: Sequence[Mapping[str, object]],
    protocol: ContinuousActorCriticRetentionProtocol,
    *,
    recovery_window: int,
    initial_snapshot: Mapping[str, object],
    final_isolated_state: Mapping[str, object],
) -> dict[str, object]:
    """Rebuild every report summary solely from raw trace and bound descriptors."""
    if len(event_trace) != len(protocol.events):
        raise ValueError("event trace length does not match the canonical protocol")
    _positive_int(recovery_window, name="recovery_window")
    if recovery_window > _EVENTS_PER_PHASE:
        raise ValueError("recovery_window exceeds a fixed phase")
    phase_summaries: list[dict[str, object]] = []
    records_by_phase: dict[str, list[Mapping[str, object]]] = {}
    for phase in protocol.phases:
        records = [record for record in event_trace if record.get("phase_id") == phase.phase_id]
        records_by_phase[phase.phase_id] = records
        phase_summaries.append(_phase_summary(records, phase))

    first_by_case = {cast(str, record["case_id"]): record for record in records_by_phase["first-a"]}
    return_by_case = {
        cast(str, record["case_id"]): record for record in records_by_phase["return-a"]
    }
    recurrence_cases: list[dict[str, object]] = []
    for case_id in sorted(first_by_case):
        first = first_by_case[case_id]
        returned = return_by_case[case_id]
        recurrence_cases.append(
            {
                "case_id": case_id,
                "target_latent_mean_abs_churn_first_to_return": abs(
                    _trace_float(returned, "decision_target_latent_mean")
                    - _trace_float(first, "decision_target_latent_mean")
                ),
                "target_std_abs_churn_first_to_return": abs(
                    _trace_float(returned, "decision_target_std")
                    - _trace_float(first, "decision_target_std")
                ),
                "sampled_action_abs_error_change_return_minus_first": (
                    _trace_float(returned, "sampled_action_abs_error")
                    - _trace_float(first, "sampled_action_abs_error")
                ),
                "target_median_action_abs_error_change_return_minus_first": (
                    _trace_float(returned, "target_median_action_abs_error")
                    - _trace_float(first, "target_median_action_abs_error")
                ),
                "critic_centered_abs_error_change_return_minus_first": (
                    abs(_trace_float(returned, "critic_same_state_centered_error"))
                    - abs(_trace_float(first, "critic_same_state_centered_error"))
                ),
                "realized_reward_change_return_minus_first": (
                    _trace_float(returned, "realized_reward")
                    - _trace_float(first, "realized_reward")
                ),
            }
        )

    first_window = records_by_phase["first-a"][:recovery_window]
    return_window = records_by_phase["return-a"][:recovery_window]
    recovery = {
        "window": recovery_window,
        "first_a_mean_sampled_action_abs_error": _mean(
            [_trace_float(record, "sampled_action_abs_error") for record in first_window]
        ),
        "return_a_mean_sampled_action_abs_error": _mean(
            [_trace_float(record, "sampled_action_abs_error") for record in return_window]
        ),
        "first_a_mean_target_median_action_abs_error": _mean(
            [_trace_float(record, "target_median_action_abs_error") for record in first_window]
        ),
        "return_a_mean_target_median_action_abs_error": _mean(
            [_trace_float(record, "target_median_action_abs_error") for record in return_window]
        ),
    }
    recovery["sampled_action_abs_error_change_return_minus_first"] = float(
        recovery["return_a_mean_sampled_action_abs_error"]
        - recovery["first_a_mean_sampled_action_abs_error"]
    )
    recovery["target_median_action_abs_error_change_return_minus_first"] = float(
        recovery["return_a_mean_target_median_action_abs_error"]
        - recovery["first_a_mean_target_median_action_abs_error"]
    )

    return {
        "assessment_status": "not-assessed",
        "development_only": True,
        "performance_thresholds_applied": False,
        "event_count": len(event_trace),
        "phase_summaries": phase_summaries,
        "recurrence_case_diagnostics": recurrence_cases,
        "recovery_window_diagnostics": recovery,
        "total_realized_return": float(
            sum(_trace_float(record, "realized_reward") for record in event_trace)
        ),
        "mean_sampled_action_abs_error": _mean(
            [_trace_float(record, "sampled_action_abs_error") for record in event_trace]
        ),
        "mean_target_median_action_abs_error": _mean(
            [_trace_float(record, "target_median_action_abs_error") for record in event_trace]
        ),
        "mean_critic_same_state_centered_squared_error": _mean(
            [
                _trace_float(record, "critic_same_state_centered_squared_error")
                for record in event_trace
            ]
        ),
        "mean_exact_target_behavior_ratio": _mean(
            [_trace_float(record, "decision_target_behavior_ratio") for record in event_trace]
        ),
        "action_boundary_saturation_count": sum(
            int(_trace_bool(record, "action_boundary_saturated")) for record in event_trace
        ),
        "log_std_boundary_saturation_count": sum(
            int(_trace_bool(record, "log_std_boundary_saturated_after_update"))
            for record in event_trace
        ),
        "sampled_action_activity_count": sum(
            int(_trace_bool(record, "sampled_action_active")) for record in event_trace
        ),
        "target_latent_mean_activity_count": sum(
            int(_trace_bool(record, "target_latent_mean_active")) for record in event_trace
        ),
        "target_median_action_activity_count": sum(
            int(_trace_bool(record, "target_median_action_active")) for record in event_trace
        ),
        "actor_update_activity_count": sum(
            int(_trace_bool(record, "actor_update_active")) for record in event_trace
        ),
        "critic_update_activity_count": sum(
            int(_trace_bool(record, "critic_update_active")) for record in event_trace
        ),
        "actor_parameter_delta_from_initial_l2": _trace_float(
            event_trace[-1], "actor_parameter_delta_from_initial_l2"
        ),
        "critic_parameter_delta_from_initial_l2": _trace_float(
            event_trace[-1], "critic_parameter_delta_from_initial_l2"
        ),
        "initial_update_count": _nonnegative_int(
            initial_snapshot.get("update_count"), name="initial update_count"
        ),
        "final_update_count": _nonnegative_int(
            final_isolated_state.get("update_count"), name="final update_count"
        ),
        "initial_decision_count": _nonnegative_int(
            initial_snapshot.get("decision_count"), name="initial decision_count"
        ),
        "final_decision_count": _nonnegative_int(
            final_isolated_state.get("decision_count"), name="final decision_count"
        ),
        "interpretation": "raw-development-diagnostics-only-no-claim",
    }


def _metric_definitions() -> dict[str, str]:
    return {
        "critic_same_state_case_predictions_raw": (
            "four preupdate differential values evaluated under one critic parameter state"
        ),
        "critic_same_state_centered_error": (
            "selected same-state centered prediction minus the selected centered phase fixture"
        ),
        "decision_target_latent_mean": (
            "target Gaussian mean in unbounded pre-tanh latent coordinates"
        ),
        "target_median_action": (
            "direct affine-tanh transform of the target latent mean; the bounded policy "
            "median, not its expectation"
        ),
        "target_median_action_error": (
            "bounded target median action minus preferred bounded-action center"
        ),
        "sampled_action_error": "cached transformed action minus preferred bounded-action center",
        "realized_reward": "float32 one minus squared sampled-action error",
        "decision_target_behavior_ratio": "exact target/behavior latent action-likelihood ratio",
        "log_density_reconstruction_ulp_distance": (
            "symmetric float32 ULP distance under the core transformed-density "
            "backend-reproducibility bound"
        ),
        "target_latent_mean_churn_abs": (
            "absolute same-case pre-tanh target-latent-mean change from its previous occurrence"
        ),
        "target_std_churn_abs": "absolute same-case target-std change from its previous occurrence",
        "parameter_update_l2": "L2 change across the named parameter owner on one update",
        "trace_update_l2": "L2 change across the named eligibility-trace owner on one update",
        "activity": "absolute magnitude or L2 change above configured activity_epsilon",
        "saturation": "exact finite-precision equality to an action or log-std boundary",
    }


def _count_scalar_values(value: object) -> int:
    if value is None or type(value) in {bool, int, float, str}:
        return int(type(value) in {bool, int, float})
    if isinstance(value, list):
        return sum(_count_scalar_values(item) for item in value)
    if isinstance(value, Mapping):
        return sum(_count_scalar_values(item) for item in value.values())
    raise ValueError("resource accounting encountered a noncanonical value")


def _resource_accounting(
    *,
    config: ContinuousActorCriticRetentionConfig,
    sources: Mapping[str, str],
    initial_snapshot: Mapping[str, object],
    final_snapshot: Mapping[str, object],
    event_trace: Sequence[Mapping[str, object]],
    canonical_report_bytes: int,
    root: Path,
) -> dict[str, object]:
    source_bytes = sum((root / relative).stat().st_size for relative in SOURCE_PATHS)
    return {
        "fixed_phase_count": _PHASE_COUNT,
        "fixed_event_count": _EVENT_COUNT,
        "configured_max_phases": config.max_phases,
        "configured_max_events": config.max_events,
        "initial_snapshot_bytes": _positive_int(
            initial_snapshot.get("state_bytes"), name="initial snapshot bytes"
        ),
        "final_state_bytes": _positive_int(
            final_snapshot.get("state_bytes"), name="final state bytes"
        ),
        "configured_max_initial_snapshot_bytes": config.max_initial_snapshot_bytes,
        "configured_max_final_state_bytes": config.max_final_state_bytes,
        "trace_scalar_values": _count_scalar_values(list(event_trace)),
        "configured_max_trace_scalar_values": config.max_trace_scalar_values,
        "canonical_report_bytes": canonical_report_bytes,
        "configured_max_report_bytes": config.max_report_bytes,
        "source_file_count": len(sources),
        "source_bytes": int(source_bytes),
        "replay_capacity": 0,
        "external_routes_per_transition": 0,
    }


def _assemble_report(
    *,
    config: ContinuousActorCriticRetentionConfig,
    protocol: ContinuousActorCriticRetentionProtocol,
    sources: Mapping[str, str],
    initial_snapshot: Mapping[str, object],
    final_snapshot: Mapping[str, object],
    event_trace: list[dict[str, object]],
    summary: Mapping[str, object],
    root: Path,
) -> dict[str, object]:
    config_value = config.to_config()
    protocol_value = protocol.to_config()
    report_size = 0
    report: dict[str, object] = {}
    for _ in range(12):
        resources = _resource_accounting(
            config=config,
            sources=sources,
            initial_snapshot=initial_snapshot,
            final_snapshot=final_snapshot,
            event_trace=event_trace,
            canonical_report_bytes=report_size,
            root=root,
        )
        hash_inputs: dict[str, object] = {
            "config_sha256": config_value,
            "protocol_sha256": protocol_value,
            "source_manifest_sha256": sources,
            "initial_snapshot_sha256": initial_snapshot,
            "event_trace_sha256": event_trace,
            "final_isolated_state_sha256": final_snapshot,
            "summary_sha256": summary,
            "resource_accounting_sha256": resources,
        }
        hashes = {name: _canonical_sha256(value) for name, value in hash_inputs.items()}
        payload: dict[str, object] = {
            "development_only": True,
            "development_status": DEVELOPMENT_STATUS,
            "assessment_status": "not-assessed",
            "continuing_control_semantics": "differential-average-reward",
            "target_policy_semantics": "bounded-affine-tanh-diagonal-gaussian",
            "behavior_policy_semantics": "same-mean-broader-std-bounded-affine-tanh-gaussian",
            "exact_action_likelihood_correction_used": True,
            "off_policy_state_distribution_correction_claimed": False,
            "off_policy_convergence_claimed": False,
            "retention_claimed": False,
            "transfer_claimed": False,
            "efficacy_claimed": False,
            "calibration_claimed": False,
            "sota_claimed": False,
            "candidate_update_safety_audit_performed": False,
            "paper_defined_delight_computed": False,
            "kondo_sparse_actor_backward_executed": False,
            "scientific_promotion_allowed": False,
            "performance_thresholds_applied": False,
            "config": config_value,
            "protocol": protocol_value,
            "source_sha256": dict(sources),
            "initial_snapshot": dict(initial_snapshot),
            "metric_definitions": _metric_definitions(),
            "event_trace": event_trace,
            "final_isolated_state": dict(final_snapshot),
            "summary": dict(summary),
            "resource_accounting": resources,
            "hashes": hashes,
            "limitations": list(_LIMITATIONS),
        }
        report = {
            "schema": CONTINUOUS_ACTOR_CRITIC_RETENTION_REPORT_SCHEMA,
            "payload": payload,
            "payload_sha256": _canonical_sha256(payload),
        }
        measured = len(_canonical_json_bytes(report)) + 1
        if measured == report_size:
            return report
        report_size = measured
    raise RuntimeError("canonical report byte accounting did not converge")


def build_continuous_actor_critic_retention_report(
    agent: ContinuousAverageRewardActorCriticAgent,
    state: ContinuousAverageRewardActorCriticState,
    config: ContinuousActorCriticRetentionConfig,
    *,
    root: Path = REPO_ROOT,
) -> dict[str, object]:
    """Run an isolated fixed A/B/A trace from one immutable supplied snapshot."""
    if not isinstance(config, ContinuousActorCriticRetentionConfig):
        raise TypeError("config must be ContinuousActorCriticRetentionConfig")
    protocol = canonical_continuous_actor_critic_retention_protocol()
    _validate_agent_state(agent, state)
    before_hash = frozen_continuous_actor_critic_state_sha256(state)
    initial_snapshot = _state_descriptor(agent, state)
    initial_bytes = _positive_int(
        initial_snapshot["state_bytes"], name="initial snapshot state_bytes"
    )
    if initial_bytes > config.max_initial_snapshot_bytes:
        raise ValueError("initial snapshot byte bound exceeded")
    event_trace, final_state = _execute_trace(agent, state, config, protocol)
    if frozen_continuous_actor_critic_state_sha256(state) != before_hash:
        raise RuntimeError("evaluator mutated the supplied continuous actor/critic snapshot")
    final_snapshot = _state_descriptor(agent, final_state)
    final_bytes = _positive_int(final_snapshot["state_bytes"], name="final state_bytes")
    if final_bytes > config.max_final_state_bytes:
        raise ValueError("final state byte bound exceeded")
    if len(event_trace) > config.max_events:
        raise ValueError("event trace exceeds configured event bound")
    scalar_count = _count_scalar_values(event_trace)
    if scalar_count > config.max_trace_scalar_values:
        raise ValueError("event trace scalar-value bound exceeded")
    summary = reconstruct_continuous_actor_critic_retention_summary(
        event_trace,
        protocol,
        recovery_window=config.recovery_window,
        initial_snapshot=initial_snapshot,
        final_isolated_state=final_snapshot,
    )
    sources = continuous_actor_critic_retention_source_snapshot(root)
    report = _assemble_report(
        config=config,
        protocol=protocol,
        sources=sources,
        initial_snapshot=initial_snapshot,
        final_snapshot=final_snapshot,
        event_trace=event_trace,
        summary=summary,
        root=root,
    )
    measured = len(_canonical_json_bytes(report)) + 1
    if measured > config.max_report_bytes:
        raise ValueError("canonical report byte bound exceeded")
    validation = validate_continuous_actor_critic_retention_report(report, root=root)
    if not validation.valid:
        raise RuntimeError("constructed report failed validation: " + "; ".join(validation.errors))
    return report


@dataclasses.dataclass(frozen=True)
class ContinuousActorCriticRetentionValidation:
    """Fail-closed structural result with no scientific assessment verdict."""

    valid: bool
    assessment_status: str
    errors: tuple[str, ...]


_TRACE_FIELDS = {
    "event_index",
    "event_id",
    "phase_id",
    "case_id",
    "phase_id_learner_visible",
    "targets_learner_visible",
    "reward_function_learner_visible",
    "realized_scalar_reward_learner_visible_after_action",
    "event_order",
    "observation",
    "next_observation",
    "preferred_action_center",
    "reward_function",
    "reference_value_target_raw",
    "cached_pre_tanh_action",
    "cached_action",
    "decision_target_latent_mean",
    "decision_behavior_latent_mean",
    "decision_target_std",
    "decision_behavior_std",
    "decision_target_log_density",
    "decision_behavior_log_density",
    "decision_target_behavior_ratio",
    "direct_transform_reconstructed_action",
    "direct_transform_reconstruction_abs_error",
    "target_log_density_reconstructed",
    "target_log_density_reconstruction_abs_error",
    "target_log_density_reconstruction_ulp_distance",
    "behavior_log_density_reconstructed",
    "behavior_log_density_reconstruction_abs_error",
    "behavior_log_density_reconstruction_ulp_distance",
    "rho_reconstructed_from_latent_gaussians",
    "rho_reconstruction_abs_error",
    "critic_same_state_case_predictions_raw",
    "critic_same_state_case_prediction_mean",
    "critic_same_state_case_predictions_centered",
    "critic_prediction_raw",
    "critic_prediction_same_state_centered",
    "reference_value_targets_phase_raw",
    "reference_value_target_phase_mean",
    "reference_value_targets_phase_centered",
    "reference_value_target_centered",
    "critic_same_state_centered_error",
    "critic_same_state_centered_squared_error",
    "target_median_action",
    "target_median_action_error",
    "target_median_action_abs_error",
    "sampled_action_error",
    "sampled_action_abs_error",
    "sampled_action_squared_error",
    "realized_reward",
    "sampled_return_contribution",
    "target_latent_mean_churn_available",
    "target_latent_mean_churn_abs",
    "target_std_churn_available",
    "target_std_churn_abs",
    "postupdate_target_latent_mean",
    "postupdate_target_std",
    "postupdate_target_median_action",
    "postupdate_target_latent_mean_change_abs",
    "postupdate_target_std_change_abs",
    "postupdate_target_median_action_change_abs",
    "td_error",
    "average_reward_before_update",
    "average_reward_after_update",
    "average_reward_change_abs",
    "actor_parameter_update_l2",
    "critic_parameter_update_l2",
    "actor_trace_update_l2",
    "critic_trace_update_l2",
    "actor_parameter_delta_from_initial_l2",
    "critic_parameter_delta_from_initial_l2",
    "actor_trace_l2_after_update",
    "critic_trace_l2_after_update",
    "action_boundary_saturated",
    "log_std_boundary_saturated_after_update",
    "sampled_action_active",
    "target_latent_mean_active",
    "target_median_action_active",
    "actor_update_active",
    "critic_update_active",
    "update_accepted",
    "all_recorded_finite",
    "update_count_before",
    "update_count_after",
    "decision_count_before",
    "decision_count_after",
    "next_cached_pre_tanh_action",
    "next_cached_action",
    "next_decision_target_latent_mean",
    "next_target_median_action",
    "next_decision_target_std",
    "next_decision_behavior_std",
    "next_decision_target_log_density",
    "next_decision_behavior_log_density",
    "next_decision_target_behavior_ratio",
}


def _validate_digest(value: object, *, name: str) -> str:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _float_vector(value: object, *, length: int, name: str) -> list[float]:
    raw = _list(value, name=name)
    if len(raw) != length:
        raise ValueError(f"{name} must contain exactly {length} finite floats")
    return [_finite_float(item, name=f"{name}[{index}]") for index, item in enumerate(raw)]


def _validate_state_descriptor(
    value: object,
    *,
    name: str,
) -> Mapping[str, object]:
    descriptor = _mapping(value, name=name)
    expected_fields = {
        "agent_config",
        "agent_config_sha256",
        "state_sha256",
        "state_bytes",
        "observation_dim",
        "action_dim",
        "action_low",
        "action_high",
        "decision_count",
        "update_count",
        "average_reward",
        "actor_parameters_sha256",
        "critic_parameters_sha256",
        "actor_traces_sha256",
        "critic_traces_sha256",
    }
    if set(descriptor) != expected_fields:
        raise ValueError(f"{name} fields do not match v2")
    agent_config = _mapping(descriptor.get("agent_config"), name=f"{name} agent_config")
    try:
        agent = ContinuousAverageRewardActorCriticAgent.from_config(dict(agent_config))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{name} agent config is invalid: {error}") from error
    if not _strict_json_equal(dict(agent_config), agent.to_config()):
        raise ValueError(f"{name} agent config is noncanonical")
    if descriptor.get("agent_config_sha256") != _canonical_sha256(agent_config):
        raise ValueError(f"{name} agent config digest does not match")
    for digest_name in (
        "state_sha256",
        "actor_parameters_sha256",
        "critic_parameters_sha256",
        "actor_traces_sha256",
        "critic_traces_sha256",
    ):
        _validate_digest(descriptor.get(digest_name), name=f"{name} {digest_name}")
    _positive_int(descriptor.get("state_bytes"), name=f"{name} state_bytes")
    if _positive_int(descriptor.get("observation_dim"), name=f"{name} observation_dim") != 2:
        raise ValueError(f"{name} observation_dim must be two")
    if _positive_int(descriptor.get("action_dim"), name=f"{name} action_dim") != 1:
        raise ValueError(f"{name} action_dim must be one")
    if not _strict_json_equal(descriptor.get("action_low"), [-1.0]):
        raise ValueError(f"{name} action_low must be [-1.0]")
    if not _strict_json_equal(descriptor.get("action_high"), [1.0]):
        raise ValueError(f"{name} action_high must be [1.0]")
    _nonnegative_int(descriptor.get("decision_count"), name=f"{name} decision_count")
    _nonnegative_int(descriptor.get("update_count"), name=f"{name} update_count")
    _finite_float(descriptor.get("average_reward"), name=f"{name} average_reward")
    return descriptor


def _expect_exact(record: Mapping[str, object], field: str, expected: object, index: int) -> None:
    if not _strict_json_equal(record.get(field), expected):
        raise ValueError(f"event trace {index} {field} does not reconstruct")


def _validate_trace(
    value: object,
    protocol: ContinuousActorCriticRetentionProtocol,
    initial_snapshot: Mapping[str, object],
    config: ContinuousActorCriticRetentionConfig,
) -> list[Mapping[str, object]]:
    values = _list(value, name="event_trace")
    if len(values) != len(protocol.events):
        raise ValueError("event trace length does not match the fixed protocol")
    initial_update_count = _nonnegative_int(
        initial_snapshot.get("update_count"), name="initial update_count"
    )
    initial_decision_count = _nonnegative_int(
        initial_snapshot.get("decision_count"), name="initial decision_count"
    )
    agent_config = _mapping(initial_snapshot.get("agent_config"), name="initial agent_config")
    agent = ContinuousAverageRewardActorCriticAgent.from_config(dict(agent_config))
    previous_latent_mean_by_case: dict[str, float] = {}
    previous_std_by_case: dict[str, float] = {}
    trace: list[Mapping[str, object]] = []
    boolean_fields = (
        "phase_id_learner_visible",
        "targets_learner_visible",
        "reward_function_learner_visible",
        "realized_scalar_reward_learner_visible_after_action",
        "target_latent_mean_churn_available",
        "target_std_churn_available",
        "action_boundary_saturated",
        "log_std_boundary_saturated_after_update",
        "sampled_action_active",
        "target_latent_mean_active",
        "target_median_action_active",
        "actor_update_active",
        "critic_update_active",
        "update_accepted",
        "all_recorded_finite",
    )
    vector_fields = {
        "critic_same_state_case_predictions_raw",
        "critic_same_state_case_predictions_centered",
        "reference_value_targets_phase_raw",
        "reference_value_targets_phase_centered",
    }
    nonnegative_fields = (
        "direct_transform_reconstruction_abs_error",
        "target_log_density_reconstruction_abs_error",
        "behavior_log_density_reconstruction_abs_error",
        "rho_reconstruction_abs_error",
        "critic_same_state_centered_squared_error",
        "target_median_action_abs_error",
        "sampled_action_abs_error",
        "sampled_action_squared_error",
        "target_latent_mean_churn_abs",
        "target_std_churn_abs",
        "postupdate_target_latent_mean_change_abs",
        "postupdate_target_std_change_abs",
        "postupdate_target_median_action_change_abs",
        "average_reward_change_abs",
        "actor_parameter_update_l2",
        "critic_parameter_update_l2",
        "actor_trace_update_l2",
        "critic_trace_update_l2",
        "actor_parameter_delta_from_initial_l2",
        "critic_parameter_delta_from_initial_l2",
        "actor_trace_l2_after_update",
        "critic_trace_l2_after_update",
    )
    non_float_fields = {
        "event_index",
        "event_id",
        "phase_id",
        "case_id",
        "event_order",
        "observation",
        "next_observation",
        "reward_function",
        "update_count_before",
        "update_count_after",
        "decision_count_before",
        "decision_count_after",
        "target_log_density_reconstruction_ulp_distance",
        "behavior_log_density_reconstruction_ulp_distance",
        *boolean_fields,
        *vector_fields,
    }
    for index, (raw_record, event) in enumerate(zip(values, protocol.events, strict=True)):
        record = _mapping(raw_record, name=f"event_trace[{index}]")
        if set(record) != _TRACE_FIELDS:
            raise ValueError(f"event trace {index} fields do not match v2")
        expected_annotations: dict[str, object] = {
            "event_index": index,
            "event_id": event.event_id,
            "phase_id": event.phase_id,
            "case_id": event.case_id,
            "phase_id_learner_visible": False,
            "targets_learner_visible": False,
            "reward_function_learner_visible": False,
            "realized_scalar_reward_learner_visible_after_action": True,
            "event_order": list(_EVENT_ORDER),
            "observation": list(event.observation),
            "next_observation": list(event.next_observation),
            "preferred_action_center": event.preferred_action_center,
            "reward_function": "float32(1-(cached_action-preferred_action_center)^2)",
            "reference_value_target_raw": event.reference_value_target,
        }
        for field, expected in expected_annotations.items():
            _expect_exact(record, field, expected, index)
        for field in boolean_fields:
            if type(record.get(field)) is not bool:
                raise ValueError(f"event trace {index} {field} must be boolean")
        for field in _TRACE_FIELDS - non_float_fields:
            _finite_float(record.get(field), name=f"event trace {index} {field}")
        for field in nonnegative_fields:
            if _finite_float(record.get(field), name=f"event trace {index} {field}") < 0.0:
                raise ValueError(f"event trace {index} {field} must be non-negative")
        for field in (
            "target_log_density_reconstruction_ulp_distance",
            "behavior_log_density_reconstruction_ulp_distance",
        ):
            ulps = _nonnegative_int(record.get(field), name=f"event trace {index} {field}")
            if ulps > TRANSFORMED_LOG_DENSITY_MAX_ULPS:
                raise ValueError(f"event trace {index} {field} exceeds core ULP bound")

        cached_latent = _trace_float(record, "cached_pre_tanh_action")
        cached_action = _trace_float(record, "cached_action")
        target_latent_mean = _trace_float(record, "decision_target_latent_mean")
        behavior_latent_mean = _trace_float(record, "decision_behavior_latent_mean")
        target_std = _trace_float(record, "decision_target_std")
        behavior_std = _trace_float(record, "decision_behavior_std")
        if behavior_latent_mean != target_latent_mean:
            raise ValueError(f"event trace {index} behavior latent mean changed")
        if target_std <= 0.0 or behavior_std <= 0.0:
            raise ValueError(f"event trace {index} policy standard deviations must be positive")
        expected_behavior_std = jnp.asarray(target_std, dtype=jnp.float32) * jnp.asarray(
            agent.config.behavior_std_scale, dtype=jnp.float32
        )
        _require_bit_exact_float32(
            behavior_std,
            expected_behavior_std,
            name=f"event trace {index} behavior standard deviation",
        )
        latent_array = jnp.asarray([cached_latent], dtype=jnp.float32)
        mean_array = jnp.asarray([target_latent_mean], dtype=jnp.float32)
        target_std_array = jnp.asarray([target_std], dtype=jnp.float32)
        behavior_std_array = jnp.asarray([behavior_std], dtype=jnp.float32)
        reconstructed_action = float(agent.squash_pre_tanh_action(latent_array)[0])
        _require_bit_exact_float32(
            cached_action,
            reconstructed_action,
            name=f"event trace {index} cached action",
        )
        _expect_exact(
            record,
            "direct_transform_reconstructed_action",
            reconstructed_action,
            index,
        )
        _expect_exact(
            record,
            "direct_transform_reconstruction_abs_error",
            abs(reconstructed_action - cached_action),
            index,
        )
        target_log_density = float(
            transformed_diagonal_gaussian_log_density(
                latent_array,
                mean_array,
                target_std_array,
                jnp.asarray([-1.0], dtype=jnp.float32),
                jnp.asarray([1.0], dtype=jnp.float32),
            )
        )
        behavior_log_density = float(
            transformed_diagonal_gaussian_log_density(
                latent_array,
                mean_array,
                behavior_std_array,
                jnp.asarray([-1.0], dtype=jnp.float32),
                jnp.asarray([1.0], dtype=jnp.float32),
            )
        )
        for field, expected in (
            ("target_log_density_reconstructed", target_log_density),
            ("behavior_log_density_reconstructed", behavior_log_density),
        ):
            _require_bit_exact_float32(
                record.get(field), expected, name=f"event trace {index} {field}"
            )
        target_density_ulps = int(
            float32_ulp_distance(
                jnp.asarray(
                    _trace_float(record, "decision_target_log_density"),
                    dtype=jnp.float32,
                ),
                jnp.asarray(target_log_density, dtype=jnp.float32),
            )
        )
        behavior_density_ulps = int(
            float32_ulp_distance(
                jnp.asarray(
                    _trace_float(record, "decision_behavior_log_density"),
                    dtype=jnp.float32,
                ),
                jnp.asarray(behavior_log_density, dtype=jnp.float32),
            )
        )
        _expect_exact(
            record,
            "target_log_density_reconstruction_ulp_distance",
            target_density_ulps,
            index,
        )
        _expect_exact(
            record,
            "behavior_log_density_reconstruction_ulp_distance",
            behavior_density_ulps,
            index,
        )
        _expect_exact(
            record,
            "target_log_density_reconstruction_abs_error",
            abs(target_log_density - _trace_float(record, "decision_target_log_density")),
            index,
        )
        _expect_exact(
            record,
            "behavior_log_density_reconstruction_abs_error",
            abs(behavior_log_density - _trace_float(record, "decision_behavior_log_density")),
            index,
        )
        reconstructed_ratio = float(
            _latent_ratio(
                latent_array,
                mean_array,
                target_std_array,
                behavior_std_array,
            )
        )
        _require_bit_exact_float32(
            record.get("rho_reconstructed_from_latent_gaussians"),
            reconstructed_ratio,
            name=f"event trace {index} reconstructed rho",
        )
        exact_ratio = _trace_float(record, "decision_target_behavior_ratio")
        if exact_ratio < 0.0:
            raise ValueError(f"event trace {index} exact rho must be non-negative")
        _require_bit_exact_float32(
            exact_ratio,
            reconstructed_ratio,
            name=f"event trace {index} exact cached rho",
        )
        _expect_exact(
            record,
            "rho_reconstruction_abs_error",
            abs(reconstructed_ratio - exact_ratio),
            index,
        )
        if _trace_float(record, "rho_reconstruction_abs_error") != 0.0:
            raise ValueError(f"event trace {index} exact rho does not reconstruct")

        target_median_action = float(agent.squash_pre_tanh_action(mean_array)[0])
        _require_bit_exact_float32(
            record.get("target_median_action"),
            target_median_action,
            name=f"event trace {index} bounded target median action",
        )
        median_error = target_median_action - event.preferred_action_center
        sampled_error = cached_action - event.preferred_action_center
        for field, expected in (
            ("target_median_action_error", median_error),
            ("target_median_action_abs_error", abs(median_error)),
            ("sampled_action_error", sampled_error),
            ("sampled_action_abs_error", abs(sampled_error)),
            ("sampled_action_squared_error", sampled_error * sampled_error),
        ):
            _expect_exact(record, field, expected, index)
        realized_reward = event.realized_reward(cached_action)
        _expect_exact(record, "realized_reward", realized_reward, index)
        _expect_exact(record, "sampled_return_contribution", realized_reward, index)

        phase_events = tuple(
            candidate for candidate in protocol.events if candidate.phase_id == event.phase_id
        )
        selected = next(
            position
            for position, candidate in enumerate(phase_events)
            if candidate.event_id == event.event_id
        )
        critic_raw = _float_vector(
            record.get("critic_same_state_case_predictions_raw"),
            length=_EVENTS_PER_PHASE,
            name=f"event trace {index} critic same-state raw vector",
        )
        critic_mean = _mean(critic_raw)
        critic_centered = [prediction - critic_mean for prediction in critic_raw]
        _expect_exact(record, "critic_same_state_case_prediction_mean", critic_mean, index)
        _expect_exact(record, "critic_same_state_case_predictions_centered", critic_centered, index)
        target_raw = [candidate.reference_value_target for candidate in phase_events]
        target_mean = _mean(target_raw)
        target_centered = [target - target_mean for target in target_raw]
        _expect_exact(record, "reference_value_targets_phase_raw", target_raw, index)
        _expect_exact(record, "reference_value_target_phase_mean", target_mean, index)
        _expect_exact(record, "reference_value_targets_phase_centered", target_centered, index)
        _expect_exact(record, "critic_prediction_raw", critic_raw[selected], index)
        _expect_exact(
            record,
            "critic_prediction_same_state_centered",
            critic_centered[selected],
            index,
        )
        _expect_exact(record, "reference_value_target_centered", target_centered[selected], index)
        centered_error = critic_centered[selected] - target_centered[selected]
        _expect_exact(record, "critic_same_state_centered_error", centered_error, index)
        _expect_exact(
            record,
            "critic_same_state_centered_squared_error",
            centered_error * centered_error,
            index,
        )

        previous_latent_mean = previous_latent_mean_by_case.get(event.case_id)
        previous_std = previous_std_by_case.get(event.case_id)
        _expect_exact(
            record,
            "target_latent_mean_churn_available",
            previous_latent_mean is not None,
            index,
        )
        _expect_exact(
            record,
            "target_latent_mean_churn_abs",
            0.0 if previous_latent_mean is None else abs(target_latent_mean - previous_latent_mean),
            index,
        )
        _expect_exact(
            record,
            "target_std_churn_available",
            previous_std is not None,
            index,
        )
        _expect_exact(
            record,
            "target_std_churn_abs",
            0.0 if previous_std is None else abs(target_std - previous_std),
            index,
        )
        previous_latent_mean_by_case[event.case_id] = target_latent_mean
        previous_std_by_case[event.case_id] = target_std

        post_latent_mean = _trace_float(record, "postupdate_target_latent_mean")
        post_std = _trace_float(record, "postupdate_target_std")
        if post_std <= 0.0:
            raise ValueError(f"event trace {index} postupdate target std must be positive")
        post_median = float(
            agent.squash_pre_tanh_action(jnp.asarray([post_latent_mean], dtype=jnp.float32))[0]
        )
        _require_bit_exact_float32(
            record.get("postupdate_target_median_action"),
            post_median,
            name=f"event trace {index} postupdate target median action",
        )
        for field, expected in (
            (
                "postupdate_target_latent_mean_change_abs",
                abs(post_latent_mean - target_latent_mean),
            ),
            ("postupdate_target_std_change_abs", abs(post_std - target_std)),
            (
                "postupdate_target_median_action_change_abs",
                abs(post_median - target_median_action),
            ),
            (
                "average_reward_change_abs",
                abs(
                    _trace_float(record, "average_reward_after_update")
                    - _trace_float(record, "average_reward_before_update")
                ),
            ),
        ):
            _expect_exact(record, field, expected, index)
        expected_boundary_saturation = cached_action in {-1.0, 1.0}
        _expect_exact(record, "action_boundary_saturated", expected_boundary_saturation, index)
        min_std = float(jnp.exp(jnp.asarray(agent.config.target_log_std_min, dtype=jnp.float32)))
        max_std = float(jnp.exp(jnp.asarray(agent.config.target_log_std_max, dtype=jnp.float32)))
        _expect_exact(
            record,
            "log_std_boundary_saturated_after_update",
            post_std in {min_std, max_std},
            index,
        )
        activity = config.activity_epsilon
        activity_values = {
            "sampled_action_active": abs(cached_action) > activity,
            "target_latent_mean_active": abs(target_latent_mean) > activity,
            "target_median_action_active": abs(target_median_action) > activity,
            "actor_update_active": _trace_float(record, "actor_parameter_update_l2") > activity,
            "critic_update_active": _trace_float(record, "critic_parameter_update_l2") > activity,
            "update_accepted": True,
            "all_recorded_finite": True,
        }
        for field, expected in activity_values.items():
            _expect_exact(record, field, expected, index)
        counters = {
            "update_count_before": initial_update_count + index,
            "update_count_after": initial_update_count + index + 1,
            "decision_count_before": initial_decision_count + index,
            "decision_count_after": initial_decision_count + index + 1,
        }
        for field, expected in counters.items():
            _expect_exact(record, field, expected, index)
        next_latent_mean = _trace_float(record, "next_decision_target_latent_mean")
        next_median = float(
            agent.squash_pre_tanh_action(jnp.asarray([next_latent_mean], dtype=jnp.float32))[0]
        )
        _require_bit_exact_float32(
            record.get("next_target_median_action"),
            next_median,
            name=f"event trace {index} next target median action",
        )
        if not -1.0 <= cached_action <= 1.0 or not -1.0 <= target_median_action <= 1.0:
            raise ValueError(f"event trace {index} bounded action escaped [-1, 1]")
        trace.append(record)

    successor_fields = (
        ("next_cached_pre_tanh_action", "cached_pre_tanh_action"),
        ("next_cached_action", "cached_action"),
        ("next_decision_target_latent_mean", "decision_target_latent_mean"),
        ("next_target_median_action", "target_median_action"),
        ("next_decision_target_std", "decision_target_std"),
        ("next_decision_behavior_std", "decision_behavior_std"),
        ("next_decision_target_log_density", "decision_target_log_density"),
        ("next_decision_behavior_log_density", "decision_behavior_log_density"),
        ("next_decision_target_behavior_ratio", "decision_target_behavior_ratio"),
    )
    for index, (left, right) in enumerate(zip(trace, trace[1:], strict=False)):
        for previous_field, next_field in successor_fields:
            if not _strict_json_equal(left.get(previous_field), right.get(next_field)):
                raise ValueError(
                    f"event trace {index} successor cache field {previous_field} changed"
                )
    return trace


def _validate_report_or_raise(
    report: Mapping[str, object],
    *,
    root: Path,
) -> ContinuousActorCriticRetentionConfig:
    if set(report) != {"schema", "payload", "payload_sha256"}:
        raise ValueError("continuous actor/critic retention report fields do not match v2")
    if report.get("schema") != CONTINUOUS_ACTOR_CRITIC_RETENTION_REPORT_SCHEMA:
        raise ValueError("continuous actor/critic retention report schema is invalid")
    payload = _mapping(report.get("payload"), name="payload")
    expected_payload_fields = {
        "development_only",
        "development_status",
        "assessment_status",
        "continuing_control_semantics",
        "target_policy_semantics",
        "behavior_policy_semantics",
        "exact_action_likelihood_correction_used",
        "off_policy_state_distribution_correction_claimed",
        "off_policy_convergence_claimed",
        "retention_claimed",
        "transfer_claimed",
        "efficacy_claimed",
        "calibration_claimed",
        "sota_claimed",
        "candidate_update_safety_audit_performed",
        "paper_defined_delight_computed",
        "kondo_sparse_actor_backward_executed",
        "scientific_promotion_allowed",
        "performance_thresholds_applied",
        "config",
        "protocol",
        "source_sha256",
        "initial_snapshot",
        "metric_definitions",
        "event_trace",
        "final_isolated_state",
        "summary",
        "resource_accounting",
        "hashes",
        "limitations",
    }
    if set(payload) != expected_payload_fields:
        raise ValueError("continuous actor/critic retention payload fields do not match v2")
    fixed: dict[str, object] = {
        "development_only": True,
        "development_status": DEVELOPMENT_STATUS,
        "assessment_status": "not-assessed",
        "continuing_control_semantics": "differential-average-reward",
        "target_policy_semantics": "bounded-affine-tanh-diagonal-gaussian",
        "behavior_policy_semantics": ("same-mean-broader-std-bounded-affine-tanh-gaussian"),
        "exact_action_likelihood_correction_used": True,
        "off_policy_state_distribution_correction_claimed": False,
        "off_policy_convergence_claimed": False,
        "retention_claimed": False,
        "transfer_claimed": False,
        "efficacy_claimed": False,
        "calibration_claimed": False,
        "sota_claimed": False,
        "candidate_update_safety_audit_performed": False,
        "paper_defined_delight_computed": False,
        "kondo_sparse_actor_backward_executed": False,
        "scientific_promotion_allowed": False,
        "performance_thresholds_applied": False,
    }
    for field, expected in fixed.items():
        if not _strict_json_equal(payload.get(field), expected):
            raise ValueError(f"continuous actor/critic retention payload {field} is invalid")
    if report.get("payload_sha256") != _canonical_sha256(payload):
        raise ValueError("continuous actor/critic retention payload digest does not match")

    config_value = _mapping(payload.get("config"), name="config")
    config = ContinuousActorCriticRetentionConfig.from_config(config_value)
    protocol_value = _mapping(payload.get("protocol"), name="protocol")
    protocol = ContinuousActorCriticRetentionProtocol.from_config(protocol_value)
    sources = _mapping(payload.get("source_sha256"), name="source_sha256")
    expected_source_paths = {path.as_posix() for path in SOURCE_PATHS}
    if set(sources) != expected_source_paths:
        raise ValueError("continuous actor/critic retention source paths changed")
    for path, digest in sources.items():
        _validate_digest(digest, name=f"source digest {path}")
    current_sources = continuous_actor_critic_retention_source_snapshot(root)
    if not _strict_json_equal(dict(sources), current_sources):
        raise ValueError(
            "continuous actor/critic retention source hashes do not match current sources"
        )

    initial_snapshot = _validate_state_descriptor(
        payload.get("initial_snapshot"), name="initial_snapshot"
    )
    final_snapshot = _validate_state_descriptor(
        payload.get("final_isolated_state"), name="final_isolated_state"
    )
    if not _strict_json_equal(
        initial_snapshot.get("agent_config"), final_snapshot.get("agent_config")
    ):
        raise ValueError("final isolated state changed the agent construction")
    initial_updates = _nonnegative_int(
        initial_snapshot.get("update_count"), name="initial update_count"
    )
    initial_decisions = _nonnegative_int(
        initial_snapshot.get("decision_count"), name="initial decision_count"
    )
    if final_snapshot.get("update_count") != initial_updates + _EVENT_COUNT:
        raise ValueError("final isolated update count breaks exact event ordering")
    if final_snapshot.get("decision_count") != initial_decisions + _EVENT_COUNT:
        raise ValueError("final isolated decision count breaks exact event ordering")
    initial_bytes = _positive_int(initial_snapshot.get("state_bytes"), name="initial state_bytes")
    final_bytes = _positive_int(final_snapshot.get("state_bytes"), name="final state bytes")
    if initial_bytes > config.max_initial_snapshot_bytes:
        raise ValueError("initial snapshot exceeds configured snapshot byte bound")
    if final_bytes > config.max_final_state_bytes:
        raise ValueError("final state exceeds configured state byte bound")

    if not _strict_json_equal(payload.get("metric_definitions"), _metric_definitions()):
        raise ValueError("continuous actor/critic retention metric definitions changed")
    with jax.disable_jit(config.execution_mode == "eager"):
        trace = _validate_trace(payload.get("event_trace"), protocol, initial_snapshot, config)
    if len(trace) > config.max_events:
        raise ValueError("event trace exceeds configured event bound")
    if _count_scalar_values(list(trace)) > config.max_trace_scalar_values:
        raise ValueError("event trace exceeds configured scalar-value bound")
    summary = _mapping(payload.get("summary"), name="summary")
    expected_summary = reconstruct_continuous_actor_critic_retention_summary(
        trace,
        protocol,
        recovery_window=config.recovery_window,
        initial_snapshot=initial_snapshot,
        final_isolated_state=final_snapshot,
    )
    if not _strict_json_equal(dict(summary), expected_summary):
        raise ValueError("continuous actor/critic retention summary does not reconstruct")

    resources = _mapping(payload.get("resource_accounting"), name="resource_accounting")
    canonical_size = len(_canonical_json_bytes(report)) + 1
    expected_resources = _resource_accounting(
        config=config,
        sources=cast(Mapping[str, str], sources),
        initial_snapshot=initial_snapshot,
        final_snapshot=final_snapshot,
        event_trace=trace,
        canonical_report_bytes=canonical_size,
        root=root,
    )
    if not _strict_json_equal(dict(resources), expected_resources):
        raise ValueError(
            "continuous actor/critic retention resource accounting does not reconstruct"
        )
    if canonical_size > config.max_report_bytes:
        raise ValueError("continuous actor/critic retention report exceeds report byte bound")
    if canonical_size > _ABSOLUTE_REPORT_BYTE_LIMIT:
        raise ValueError("continuous actor/critic retention report exceeds hard byte limit")
    if not _strict_json_equal(payload.get("limitations"), list(_LIMITATIONS)):
        raise ValueError("continuous actor/critic retention limitations changed")
    hashes = _mapping(payload.get("hashes"), name="hashes")
    hash_inputs: dict[str, object] = {
        "config_sha256": config_value,
        "protocol_sha256": protocol_value,
        "source_manifest_sha256": sources,
        "initial_snapshot_sha256": initial_snapshot,
        "event_trace_sha256": trace,
        "final_isolated_state_sha256": final_snapshot,
        "summary_sha256": summary,
        "resource_accounting_sha256": resources,
    }
    expected_hashes = {
        field: _canonical_sha256(component) for field, component in hash_inputs.items()
    }
    if not _strict_json_equal(dict(hashes), expected_hashes):
        raise ValueError("continuous actor/critic retention component hashes do not reconstruct")
    return config


def validate_continuous_actor_critic_retention_report(
    report: Mapping[str, object],
    *,
    root: Path = REPO_ROOT,
    agent: ContinuousAverageRewardActorCriticAgent | None = None,
    state: ContinuousAverageRewardActorCriticState | None = None,
) -> ContinuousActorCriticRetentionValidation:
    """Validate structure/source binding and optionally replay the exact snapshot."""
    errors: list[str] = []
    config: ContinuousActorCriticRetentionConfig | None = None
    try:
        config = _validate_report_or_raise(report, root=root)
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        errors.append(str(error))
    if (agent is None) != (state is None):
        errors.append("live replay requires both agent and state")
    elif agent is not None and state is not None and config is not None and not errors:
        try:
            payload = _mapping(report.get("payload"), name="payload")
            initial = _mapping(payload.get("initial_snapshot"), name="initial_snapshot")
            _validate_agent_state(agent, state)
            if not _strict_json_equal(_state_descriptor(agent, state), initial):
                raise ValueError("live replay snapshot does not match report")
            before = frozen_continuous_actor_critic_state_sha256(state)
            replay = build_continuous_actor_critic_retention_report(agent, state, config, root=root)
            if frozen_continuous_actor_critic_state_sha256(state) != before:
                raise RuntimeError("live replay mutated the supplied snapshot")
            if not _strict_json_equal(replay, dict(report)):
                raise ValueError("live replay does not reproduce the exact report")
        except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
            errors.append(str(error))
    return ContinuousActorCriticRetentionValidation(
        valid=not errors,
        assessment_status="not-assessed",
        errors=tuple(errors),
    )


def canonical_continuous_actor_critic_retention_report_bytes(
    report: Mapping[str, object],
) -> bytes:
    """Return the sole accepted canonical UTF-8 JSON report encoding."""
    return _canonical_json_bytes(report) + b"\n"


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def _load_canonical_json_document(
    path: str | Path,
    *,
    name: str,
    byte_limit: int,
) -> dict[str, object]:
    source = Path(path)
    size = source.stat().st_size
    if size > byte_limit:
        raise ValueError(f"{name} exceeds the absolute hard byte limit")
    data = source.read_bytes()
    try:
        decoded = json.loads(
            data,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is not valid JSON: {error}") from error
    document = dict(_mapping(decoded, name=name))
    try:
        canonical = _canonical_json_bytes(document) + b"\n"
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} is not canonical JSON: {error}") from error
    if data != canonical:
        raise ValueError(f"{name} is not exact canonical JSON")
    return document


def load_continuous_actor_critic_retention_report(
    path: str | Path,
    *,
    root: Path = REPO_ROOT,
) -> dict[str, object]:
    """Load exact canonical JSON and require full current-source validation."""
    report = _load_canonical_json_document(
        path,
        name="continuous actor/critic retention report",
        byte_limit=_ABSOLUTE_REPORT_BYTE_LIMIT,
    )
    validation = validate_continuous_actor_critic_retention_report(report, root=root)
    if not validation.valid:
        raise ValueError(
            "continuous actor/critic retention report is invalid: " + "; ".join(validation.errors)
        )
    return report


def _write_new_canonical_file(
    value: Mapping[str, object],
    path: str | Path,
    *,
    name: str,
) -> None:
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    resolved = destination.resolve(strict=False)
    if os.path.lexists(destination) or os.path.lexists(resolved):
        raise FileExistsError(f"refusing to overwrite {name}: {destination}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical_json_bytes(value) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination, follow_symlinks=False)
        except FileExistsError as error:
            raise FileExistsError(
                f"refusing to overwrite concurrently created {name}: {destination}"
            ) from error
    finally:
        temporary.unlink(missing_ok=True)


def save_continuous_actor_critic_retention_report(
    report: Mapping[str, object],
    path: str | Path,
    *,
    root: Path = REPO_ROOT,
) -> None:
    """Atomically create one validated canonical report without overwrite."""
    validation = validate_continuous_actor_critic_retention_report(report, root=root)
    if not validation.valid:
        raise ValueError(
            "refusing to save invalid continuous actor/critic retention report: "
            + "; ".join(validation.errors)
        )
    _write_new_canonical_file(report, path, name="continuous actor/critic retention report")


def _snapshot_checkpoint_document(
    agent: ContinuousAverageRewardActorCriticAgent,
    state: ContinuousAverageRewardActorCriticState,
    *,
    root: Path,
) -> dict[str, object]:
    _validate_agent_state(agent, state)
    snapshot = _state_descriptor(agent, state)
    sources = continuous_actor_critic_retention_source_snapshot(root)
    core_checkpoint = json.loads(json.dumps(agent.checkpoint_payload(state), allow_nan=False))
    payload: dict[str, object] = {
        "development_only": True,
        "development_status": DEVELOPMENT_STATUS,
        "assessment_status": "not-assessed",
        "scientific_promotion_allowed": False,
        "performance_thresholds_applied": False,
        "retention_claimed": False,
        "transfer_claimed": False,
        "efficacy_claimed": False,
        "calibration_claimed": False,
        "sota_claimed": False,
        "off_policy_state_distribution_correction_claimed": False,
        "off_policy_convergence_claimed": False,
        "candidate_update_safety_audit_performed": False,
        "paper_defined_delight_computed": False,
        "kondo_sparse_actor_backward_executed": False,
        "source_sha256": sources,
        "source_manifest_sha256": _canonical_sha256(sources),
        "snapshot": snapshot,
        "snapshot_sha256": _canonical_sha256(snapshot),
        "core_checkpoint": core_checkpoint,
        "core_checkpoint_sha256": _canonical_sha256(core_checkpoint),
    }
    return {
        "schema": CONTINUOUS_ACTOR_CRITIC_RETENTION_CHECKPOINT_SCHEMA,
        "payload": payload,
        "payload_sha256": _canonical_sha256(payload),
    }


def _restore_snapshot_checkpoint_document(
    document: Mapping[str, object],
    *,
    root: Path,
) -> tuple[ContinuousAverageRewardActorCriticAgent, ContinuousAverageRewardActorCriticState]:
    if set(document) != {"schema", "payload", "payload_sha256"}:
        raise ValueError("continuous actor/critic snapshot fields do not match v2")
    if document.get("schema") != CONTINUOUS_ACTOR_CRITIC_RETENTION_CHECKPOINT_SCHEMA:
        raise ValueError("continuous actor/critic snapshot schema is invalid")
    payload = _mapping(document.get("payload"), name="snapshot payload")
    expected_fields = {
        "development_only",
        "development_status",
        "assessment_status",
        "scientific_promotion_allowed",
        "performance_thresholds_applied",
        "retention_claimed",
        "transfer_claimed",
        "efficacy_claimed",
        "calibration_claimed",
        "sota_claimed",
        "off_policy_state_distribution_correction_claimed",
        "off_policy_convergence_claimed",
        "candidate_update_safety_audit_performed",
        "paper_defined_delight_computed",
        "kondo_sparse_actor_backward_executed",
        "source_sha256",
        "source_manifest_sha256",
        "snapshot",
        "snapshot_sha256",
        "core_checkpoint",
        "core_checkpoint_sha256",
    }
    if set(payload) != expected_fields:
        raise ValueError("continuous actor/critic snapshot payload fields do not match v2")
    fixed: dict[str, object] = {
        "development_only": True,
        "development_status": DEVELOPMENT_STATUS,
        "assessment_status": "not-assessed",
        "scientific_promotion_allowed": False,
        "performance_thresholds_applied": False,
        "retention_claimed": False,
        "transfer_claimed": False,
        "efficacy_claimed": False,
        "calibration_claimed": False,
        "sota_claimed": False,
        "off_policy_state_distribution_correction_claimed": False,
        "off_policy_convergence_claimed": False,
        "candidate_update_safety_audit_performed": False,
        "paper_defined_delight_computed": False,
        "kondo_sparse_actor_backward_executed": False,
    }
    for field, expected in fixed.items():
        if not _strict_json_equal(payload.get(field), expected):
            raise ValueError(f"continuous actor/critic snapshot {field} is invalid")
    if document.get("payload_sha256") != _canonical_sha256(payload):
        raise ValueError("continuous actor/critic snapshot payload digest does not match")
    sources = _mapping(payload.get("source_sha256"), name="snapshot source_sha256")
    for path, digest in sources.items():
        if path not in {source.as_posix() for source in SOURCE_PATHS}:
            raise ValueError("continuous actor/critic snapshot source path changed")
        _validate_digest(digest, name=f"snapshot source digest {path}")
    current_sources = continuous_actor_critic_retention_source_snapshot(root)
    if not _strict_json_equal(dict(sources), current_sources):
        raise ValueError("continuous actor/critic snapshot source hashes do not match")
    if payload.get("source_manifest_sha256") != _canonical_sha256(sources):
        raise ValueError("continuous actor/critic snapshot source digest does not match")
    snapshot = _validate_state_descriptor(payload.get("snapshot"), name="snapshot")
    if payload.get("snapshot_sha256") != _canonical_sha256(snapshot):
        raise ValueError("continuous actor/critic snapshot descriptor digest does not match")
    core_checkpoint = _mapping(payload.get("core_checkpoint"), name="snapshot core_checkpoint")
    if payload.get("core_checkpoint_sha256") != _canonical_sha256(core_checkpoint):
        raise ValueError("continuous actor/critic core checkpoint digest does not match")
    try:
        agent, state = ContinuousAverageRewardActorCriticAgent.from_checkpoint_payload(
            core_checkpoint
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"continuous actor/critic core checkpoint is invalid: {error}") from error
    _validate_agent_state(agent, state)
    restored_core = json.loads(json.dumps(agent.checkpoint_payload(state), allow_nan=False))
    if not _strict_json_equal(restored_core, dict(core_checkpoint)):
        raise ValueError("continuous actor/critic core checkpoint is noncanonical")
    if not _strict_json_equal(_state_descriptor(agent, state), snapshot):
        raise ValueError("restored continuous actor/critic snapshot descriptor changed")
    return agent, state


def save_continuous_actor_critic_retention_snapshot_checkpoint(
    agent: ContinuousAverageRewardActorCriticAgent,
    state: ContinuousAverageRewardActorCriticState,
    path: str | Path,
    *,
    root: Path = REPO_ROOT,
) -> None:
    """Create a strict source-, construction-, and state-bound JSON snapshot."""
    document = _snapshot_checkpoint_document(agent, state, root=root)
    _restore_snapshot_checkpoint_document(document, root=root)
    encoded_size = len(_canonical_json_bytes(document)) + 1
    if encoded_size > _ABSOLUTE_CHECKPOINT_BYTE_LIMIT:
        raise ValueError("continuous actor/critic snapshot exceeds hard byte limit")
    _write_new_canonical_file(document, path, name="continuous actor/critic retention snapshot")


def load_continuous_actor_critic_retention_snapshot_checkpoint(
    path: str | Path,
    *,
    root: Path = REPO_ROOT,
) -> tuple[ContinuousAverageRewardActorCriticAgent, ContinuousAverageRewardActorCriticState]:
    """Restore only an exact current-source canonical JSON snapshot."""
    document = _load_canonical_json_document(
        path,
        name="continuous actor/critic retention snapshot",
        byte_limit=_ABSOLUTE_CHECKPOINT_BYTE_LIMIT,
    )
    return _restore_snapshot_checkpoint_document(document, root=root)


class ContinuousActorCriticRetentionEvaluator:
    """Immutable adapter binding the fixed protocol and one diagnostic config."""

    def __init__(self, config: ContinuousActorCriticRetentionConfig) -> None:
        if not isinstance(config, ContinuousActorCriticRetentionConfig):
            raise TypeError("config must be ContinuousActorCriticRetentionConfig")
        self._config = config

    @property
    def config(self) -> ContinuousActorCriticRetentionConfig:
        return self._config

    @property
    def protocol(self) -> ContinuousActorCriticRetentionProtocol:
        return canonical_continuous_actor_critic_retention_protocol()

    def evaluate(
        self,
        agent: ContinuousAverageRewardActorCriticAgent,
        state: ContinuousAverageRewardActorCriticState,
        *,
        root: Path = REPO_ROOT,
    ) -> dict[str, object]:
        """Run the bound evaluator without mutating the supplied snapshot."""
        return build_continuous_actor_critic_retention_report(agent, state, self._config, root=root)


__all__ = [
    "CONTINUOUS_ACTOR_CRITIC_RETENTION_CHECKPOINT_SCHEMA",
    "CONTINUOUS_ACTOR_CRITIC_RETENTION_CONFIG_SCHEMA",
    "CONTINUOUS_ACTOR_CRITIC_RETENTION_PROTOCOL_SCHEMA",
    "CONTINUOUS_ACTOR_CRITIC_RETENTION_REPORT_SCHEMA",
    "ContinuousActorCriticRetentionConfig",
    "ContinuousActorCriticRetentionEvaluator",
    "ContinuousActorCriticRetentionEvent",
    "ContinuousActorCriticRetentionPhase",
    "ContinuousActorCriticRetentionProtocol",
    "ContinuousActorCriticRetentionValidation",
    "build_continuous_actor_critic_retention_report",
    "canonical_continuous_actor_critic_retention_protocol",
    "canonical_continuous_actor_critic_retention_report_bytes",
    "continuous_actor_critic_retention_source_snapshot",
    "frozen_continuous_actor_critic_state_sha256",
    "load_continuous_actor_critic_retention_report",
    "load_continuous_actor_critic_retention_snapshot_checkpoint",
    "reconstruct_continuous_actor_critic_retention_summary",
    "save_continuous_actor_critic_retention_report",
    "save_continuous_actor_critic_retention_snapshot_checkpoint",
    "validate_continuous_actor_critic_retention_report",
]
