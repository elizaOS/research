# mypy: disable-error-code="attr-defined,call-arg"
"""Strict development-only actor/critic retention diagnostics.

The evaluator owns one immutable continuing A/B/A event schedule, all reward
tables, preferred actions, reference-value targets, phase annotations, and the
action-sampling seed.  The learner receives observations, its previously
sampled action, the reward realized for that action, and the next observation;
no phase, case, target, or recurrence identifier crosses the learner boundary.

Every event records the exact decision-time policy and a critic prediction
before the single transition update.  The report retains enough raw data to
reconstruct phase, recurrence, plasticity, action-activity, policy-churn, and
realized-return summaries.  This is an ordinary average-reward policy-gradient
lane.  It does not use or silently relabel the paper-specific DG ``delight``
quantity, and it does not run the separate candidate-update safety audit.
The actor objective is the epsilon-mixture behavior policy: the retained
``(1-epsilon) * pi(a) / b(a)`` value is its exact score-function chain-rule
scale, not an importance-sampling correction for a target-policy objective.

The fixed stream and one source-bound run are development diagnostics only.
No threshold is applied and no retention, efficacy, calibration, promotion,
SOTA, or Alberta Plan completion claim follows from a valid report.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
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

from alberta_framework.core.average_reward import (
    AverageRewardHordeActorCriticAgent,
    AverageRewardHordeActorCriticState,
)
from alberta_framework.core.checkpoints import (
    load_checkpoint,
    load_checkpoint_metadata,
    save_checkpoint,
)

ACTOR_CRITIC_RETENTION_CONFIG_SCHEMA = "alberta.actor-critic-retention.config.v2"
ACTOR_CRITIC_RETENTION_PROTOCOL_SCHEMA = "alberta.actor-critic-retention.protocol.v2"
ACTOR_CRITIC_RETENTION_REPORT_SCHEMA = "alberta.actor-critic-retention.report.v2"
ACTOR_CRITIC_RETENTION_CHECKPOINT_SCHEMA = "alberta.actor-critic-retention.snapshot.v2"
DEVELOPMENT_STATUS = "development-only-not-assessed"
REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATHS = (
    Path("alberta_framework/core/average_reward.py"),
    Path("alberta_framework/core/checkpoints.py"),
    Path("alberta_framework/core/initializers.py"),
    Path("alberta_framework/core/multi_head_learner.py"),
    Path("alberta_framework/core/normalizers.py"),
    Path("alberta_framework/core/optimizers.py"),
    Path("alberta_framework/core/types.py"),
    Path("alberta_framework/evaluation/actor_critic_retention.py"),
)

_PHASE_COUNT = 3
_EVENT_COUNT = 12
_N_ACTIONS = 2
_UINT32_MAX = 2**32 - 1
_INT32_MAX = 2**31 - 1
_IDENTIFIER = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}\Z")
_EVENT_ORDER = (
    "cached-decision-consumed-before-update",
    "critic-predicted-before-update",
    "reward-realized-after-action",
    "atomic-update-committed-before-next-decision",
    "next-decision-sampled-from-committed-parameters",
)
_LIMITATIONS = (
    "development diagnostics only; assessment status is not-assessed",
    "one fixed A/B/A stream does not establish external validity or retention",
    "reference-value targets are evaluator fixtures, not calibrated value certificates",
    "realized return is the undiscounted reward sum on this one continuing sampled trace",
    "policy probability margin and L1 churn do not establish control efficacy",
    "the raw target/behavior ratio is diagnostic, not an off-policy correction claim",
    "nonzero plasticity and action activity expose trivial policies but are not success gates",
    "structural report validation needs the separately bound snapshot for exact live replay",
    "one source-bound seed supplies no matched multi-seed comparison or scientific promotion",
    "the probe is not integrated into PrototypeAgent and makes no SOTA or completion claim",
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


def actor_critic_retention_source_snapshot(root: Path = REPO_ROOT) -> dict[str, str]:
    """Hash the exact local source closure used by this diagnostic."""
    return {relative.as_posix(): _file_sha256(root / relative) for relative in SOURCE_PATHS}


@dataclasses.dataclass(frozen=True)
class ActorCriticRetentionConfig:
    """Fixed seed, reconstruction window, execution mode, and hard bounds."""

    action_seed: int = 37
    recovery_window: int = 2
    max_phases: int = _PHASE_COUNT
    max_events: int = _EVENT_COUNT
    max_initial_snapshot_bytes: int = 8_000_000
    max_report_bytes: int = 8_000_000
    execution_mode: Literal["eager", "jit"] = "jit"

    def __post_init__(self) -> None:
        _nonnegative_int(self.action_seed, name="action_seed")
        if self.action_seed > _UINT32_MAX:
            raise ValueError("action_seed must fit an unsigned 32-bit seed")
        _positive_int(self.recovery_window, name="recovery_window")
        if self.recovery_window > 4:
            raise ValueError("recovery_window cannot exceed the fixed phase length")
        _positive_int(self.max_phases, name="max_phases")
        if self.max_phases < _PHASE_COUNT:
            raise ValueError("max_phases is smaller than the fixed protocol")
        _positive_int(self.max_events, name="max_events")
        if self.max_events < _EVENT_COUNT:
            raise ValueError("max_events is smaller than the fixed protocol")
        _positive_int(self.max_initial_snapshot_bytes, name="max_initial_snapshot_bytes")
        _positive_int(self.max_report_bytes, name="max_report_bytes")
        if self.execution_mode not in {"eager", "jit"}:
            raise ValueError("execution_mode must be 'eager' or 'jit'")

    def to_config(self) -> dict[str, object]:
        return {
            "schema": ACTOR_CRITIC_RETENTION_CONFIG_SCHEMA,
            "type": type(self).__name__,
            "development_status": DEVELOPMENT_STATUS,
            "assessment_status": "not-assessed",
            "scientific_promotion_allowed": False,
            "retention_claimed": False,
            "efficacy_claimed": False,
            "calibration_claimed": False,
            "performance_thresholds_applied": False,
            "policy_gradient_mode": "ordinary",
            "behavior_policy_objective": "fixed-epsilon-mixture",
            "off_policy_target_policy_correction_claimed": False,
            "paper_specific_dg_delight_used": False,
            "action_seed": self.action_seed,
            "recovery_window": self.recovery_window,
            "max_phases": self.max_phases,
            "max_events": self.max_events,
            "max_initial_snapshot_bytes": self.max_initial_snapshot_bytes,
            "max_report_bytes": self.max_report_bytes,
            "execution_mode": self.execution_mode,
        }

    @classmethod
    def from_config(cls, payload: Mapping[str, object]) -> ActorCriticRetentionConfig:
        expected_fields = set(cls().to_config())
        if set(payload) != expected_fields:
            raise ValueError("actor/critic retention config fields do not match v2")
        fixed: dict[str, object] = {
            "schema": ACTOR_CRITIC_RETENTION_CONFIG_SCHEMA,
            "type": cls.__name__,
            "development_status": DEVELOPMENT_STATUS,
            "assessment_status": "not-assessed",
            "scientific_promotion_allowed": False,
            "retention_claimed": False,
            "efficacy_claimed": False,
            "calibration_claimed": False,
            "performance_thresholds_applied": False,
            "policy_gradient_mode": "ordinary",
            "behavior_policy_objective": "fixed-epsilon-mixture",
            "off_policy_target_policy_correction_claimed": False,
            "paper_specific_dg_delight_used": False,
        }
        for name, expected in fixed.items():
            if not _strict_json_equal(payload.get(name), expected):
                raise ValueError(f"actor/critic retention config {name} is invalid")
        mode = payload.get("execution_mode")
        if mode not in {"eager", "jit"}:
            raise ValueError("actor/critic retention config execution_mode is invalid")
        result = cls(
            action_seed=_nonnegative_int(payload.get("action_seed"), name="action_seed"),
            recovery_window=_positive_int(
                payload.get("recovery_window"), name="recovery_window"
            ),
            max_phases=_positive_int(payload.get("max_phases"), name="max_phases"),
            max_events=_positive_int(payload.get("max_events"), name="max_events"),
            max_initial_snapshot_bytes=_positive_int(
                payload.get("max_initial_snapshot_bytes"),
                name="max_initial_snapshot_bytes",
            ),
            max_report_bytes=_positive_int(
                payload.get("max_report_bytes"), name="max_report_bytes"
            ),
            execution_mode=mode,
        )
        if not _strict_json_equal(dict(payload), result.to_config()):
            raise ValueError("actor/critic retention config is noncanonical")
        return result


@dataclasses.dataclass(frozen=True)
class ActorCriticRetentionPhase:
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
class ActorCriticRetentionEvent:
    """One event with learner-visible observations and evaluator-only targets."""

    event_id: str
    phase_id: str
    case_id: str
    observation: tuple[float, ...]
    next_observation: tuple[float, ...]
    preferred_action: int
    reference_value_target: float
    action_rewards: tuple[float, ...]

    def __post_init__(self) -> None:
        _identifier(self.event_id, name="event_id")
        _identifier(self.phase_id, name="phase_id")
        _identifier(self.case_id, name="case_id")
        if not self.observation or len(self.next_observation) != len(self.observation):
            raise ValueError("event observations must be non-empty and shape matched")
        for name, values in (
            ("observation", self.observation),
            ("next_observation", self.next_observation),
            ("action_rewards", self.action_rewards),
        ):
            for index, value in enumerate(values):
                if not isinstance(value, float) or not math.isfinite(value):
                    raise ValueError(f"{name}[{index}] must be a finite float")
        if len(self.action_rewards) != _N_ACTIONS:
            raise ValueError("action_rewards must match the fixed two-action protocol")
        _nonnegative_int(self.preferred_action, name="preferred_action")
        if self.preferred_action >= len(self.action_rewards):
            raise ValueError("preferred_action is outside action_rewards")
        if not isinstance(self.reference_value_target, float) or not math.isfinite(
            self.reference_value_target
        ):
            raise ValueError("reference_value_target must be a finite float")
        if self.action_rewards[self.preferred_action] != max(self.action_rewards):
            raise ValueError("preferred_action must select the unique largest action reward")
        if self.action_rewards.count(max(self.action_rewards)) != 1:
            raise ValueError("event action rewards require one unique preferred action")

    def to_config(self) -> dict[str, object]:
        return {
            "event_id": self.event_id,
            "phase_id": self.phase_id,
            "case_id": self.case_id,
            "observation": list(self.observation),
            "next_observation": list(self.next_observation),
            "preferred_action": self.preferred_action,
            "reference_value_target": self.reference_value_target,
            "action_rewards": list(self.action_rewards),
            "phase_id_learner_visible": False,
            "targets_learner_visible": False,
            "reward_table_learner_visible": False,
            "realized_scalar_reward_learner_visible_after_action": True,
        }


@dataclasses.dataclass(frozen=True)
class ActorCriticRetentionProtocol:
    """The only accepted immutable continuing actor/critic diagnostic stream."""

    protocol_id: str
    phases: tuple[ActorCriticRetentionPhase, ...]
    events: tuple[ActorCriticRetentionEvent, ...]
    learner_visible_fields: tuple[str, ...] = (
        "observation",
        "cached_sampled_action",
        "realized_scalar_reward_after_action",
        "next_observation",
    )
    evaluator_only_fields: tuple[str, ...] = (
        "phase_id",
        "case_id",
        "preferred_action",
        "reference_value_target",
        "action_rewards",
    )

    def __post_init__(self) -> None:
        _identifier(self.protocol_id, name="protocol_id")
        if len(self.phases) != _PHASE_COUNT or len(self.events) != _EVENT_COUNT:
            raise ValueError("actor/critic retention protocol must retain the fixed 3x4 shape")
        if self.learner_visible_fields != (
            "observation",
            "cached_sampled_action",
            "realized_scalar_reward_after_action",
            "next_observation",
        ):
            raise ValueError("learner_visible_fields do not match the fixed causal boundary")
        if self.evaluator_only_fields != (
            "phase_id",
            "case_id",
            "preferred_action",
            "reference_value_target",
            "action_rewards",
        ):
            raise ValueError("evaluator_only_fields do not match the fixed isolation contract")
        phase_ids = [phase.phase_id for phase in self.phases]
        if len(set(phase_ids)) != len(phase_ids):
            raise ValueError("phase_id values must be unique")
        if sum(phase.event_count for phase in self.phases) != len(self.events):
            raise ValueError("phase counts must cover every event exactly once")
        event_ids = [event.event_id for event in self.events]
        if len(set(event_ids)) != len(event_ids):
            raise ValueError("event_id values must be unique")
        cursor = 0
        seen_phases: set[str] = set()
        by_phase: dict[str, tuple[ActorCriticRetentionEvent, ...]] = {}
        for phase in self.phases:
            if phase.recurrence_of_phase_id is not None:
                if phase.recurrence_of_phase_id not in seen_phases:
                    raise ValueError("recurrence phase must reference an earlier phase")
            chunk = self.events[cursor : cursor + phase.event_count]
            if any(event.phase_id != phase.phase_id for event in chunk):
                raise ValueError("events must be contiguous in declared phase order")
            by_phase[phase.phase_id] = chunk
            cursor += phase.event_count
            seen_phases.add(phase.phase_id)
        for index, event in enumerate(self.events):
            following = self.events[(index + 1) % len(self.events)]
            if event.next_observation != following.observation:
                raise ValueError("event observations must form one exact continuing cycle")
        for phase in self.phases:
            reference_id = phase.recurrence_of_phase_id
            if reference_id is None:
                continue
            reference = by_phase[reference_id]
            current = by_phase[phase.phase_id]
            if len(reference) != len(current) or any(
                (
                    old.case_id,
                    old.observation,
                    old.next_observation,
                    old.preferred_action,
                    old.reference_value_target,
                    old.action_rewards,
                )
                != (
                    new.case_id,
                    new.observation,
                    new.next_observation,
                    new.preferred_action,
                    new.reference_value_target,
                    new.action_rewards,
                )
                for old, new in zip(reference, current, strict=True)
            ):
                raise ValueError("recurrence must reuse exact ordered evaluator cases")

    @property
    def phase_ids(self) -> tuple[str, ...]:
        return tuple(phase.phase_id for phase in self.phases)

    def to_config(self) -> dict[str, object]:
        return {
            "schema": ACTOR_CRITIC_RETENTION_PROTOCOL_SCHEMA,
            "type": type(self).__name__,
            "protocol_id": self.protocol_id,
            "continuing": True,
            "fixed_schedule": True,
            "phase_labels_learner_visible": False,
            "targets_learner_visible": False,
            "seed_owner": "evaluator",
            "learner_visible_fields": list(self.learner_visible_fields),
            "evaluator_only_fields": list(self.evaluator_only_fields),
            "phases": [phase.to_config() for phase in self.phases],
            "events": [event.to_config() for event in self.events],
        }

    @classmethod
    def from_config(cls, payload: Mapping[str, object]) -> ActorCriticRetentionProtocol:
        expected = canonical_actor_critic_retention_protocol()
        if not _strict_json_equal(dict(payload), expected.to_config()):
            raise ValueError("only the exact canonical actor/critic retention protocol is valid")
        return expected


def canonical_actor_critic_retention_protocol() -> ActorCriticRetentionProtocol:
    """Construct the fixed evaluator-owned A/B/A continuing schedule."""
    observations = (
        (1.0, 0.0),
        (0.0, 1.0),
        (-1.0, 0.0),
        (0.0, -1.0),
    )
    value_targets = (0.25, -0.25, 0.25, -0.25)
    preferred_a = (0, 1, 0, 1)
    preferred_b = (1, 0, 1, 0)

    def events_for(
        phase_id: str,
        event_prefix: str,
        preferred: tuple[int, ...],
    ) -> tuple[ActorCriticRetentionEvent, ...]:
        events: list[ActorCriticRetentionEvent] = []
        for index, (observation, action, target) in enumerate(
            zip(observations, preferred, value_targets, strict=True)
        ):
            rewards = (1.0, -1.0) if action == 0 else (-1.0, 1.0)
            events.append(
                ActorCriticRetentionEvent(
                    event_id=f"{event_prefix}-{index}",
                    phase_id=phase_id,
                    case_id=f"cycle-{index}",
                    observation=observation,
                    next_observation=observations[(index + 1) % len(observations)],
                    preferred_action=action,
                    reference_value_target=target,
                    action_rewards=rewards,
                )
            )
        return tuple(events)

    return ActorCriticRetentionProtocol(
        protocol_id="average-reward-actor-critic-aba-v1",
        phases=(
            ActorCriticRetentionPhase("first-a", 4),
            ActorCriticRetentionPhase("interference-b", 4),
            ActorCriticRetentionPhase("return-a", 4, "first-a"),
        ),
        events=(
            *events_for("first-a", "a", preferred_a),
            *events_for("interference-b", "b", preferred_b),
            *events_for("return-a", "a-return", preferred_a),
        ),
    )


def _numpy_leaf(leaf: object) -> np.ndarray:
    dtype = getattr(leaf, "dtype", None)
    if dtype is not None and jnp.issubdtype(dtype, jax.dtypes.prng_key):
        array = np.asarray(jr.key_data(cast(Array, leaf)))
    else:
        array = np.asarray(leaf)
    if array.dtype.hasobject:
        raise ValueError("actor/critic state contains a noncanonical object leaf")
    return np.ascontiguousarray(array)


def _leaf_descriptor(leaf: object) -> dict[str, object]:
    array = _numpy_leaf(leaf)
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "nbytes": int(array.nbytes),
        "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }


def _state_manifest(state: AverageRewardHordeActorCriticState) -> list[dict[str, object]]:
    return [_leaf_descriptor(leaf) for leaf in jax.tree.leaves(state)]


def frozen_actor_critic_state_sha256(state: AverageRewardHordeActorCriticState) -> str:
    """Hash every leaf of one immutable actor/critic state."""
    if not isinstance(state, AverageRewardHordeActorCriticState):
        raise TypeError("state must be AverageRewardHordeActorCriticState")
    return _canonical_sha256(_state_manifest(state))


def _state_bytes(state: AverageRewardHordeActorCriticState) -> int:
    return sum(
        _nonnegative_int(item["nbytes"], name="state leaf nbytes")
        for item in _state_manifest(state)
    )


def _parameter_manifest(parameters: Sequence[Array]) -> list[dict[str, object]]:
    return [_leaf_descriptor(parameter) for parameter in parameters]


def _actor_parameters(state: AverageRewardHordeActorCriticState) -> tuple[Array, ...]:
    return (state.actor_weights, state.actor_bias)


def _critic_parameters(state: AverageRewardHordeActorCriticState) -> tuple[Array, ...]:
    learner = state.critic_state.learner_state
    return (
        *learner.trunk_params.weights,
        *learner.trunk_params.biases,
        *learner.head_params.weights,
        *learner.head_params.biases,
    )


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


def _state_descriptor(
    agent: AverageRewardHordeActorCriticAgent,
    state: AverageRewardHordeActorCriticState,
) -> dict[str, object]:
    agent_config = agent.to_config()
    observation_dim = int(np.asarray(state.last_observation).shape[0])
    actor_manifest = _parameter_manifest(_actor_parameters(state))
    critic_manifest = _parameter_manifest(_critic_parameters(state))
    return {
        "agent_config": agent_config,
        "agent_config_sha256": _canonical_sha256(agent_config),
        "state_sha256": frozen_actor_critic_state_sha256(state),
        "state_bytes": _state_bytes(state),
        "observation_dim": observation_dim,
        "n_actions": agent.config.n_actions,
        "actor_step_count": int(state.step_count),
        "critic_step_count": int(state.critic_state.step_count),
        "actor_parameters_sha256": _canonical_sha256(actor_manifest),
        "critic_parameters_sha256": _canonical_sha256(critic_manifest),
    }


def _validate_agent_state(
    agent: AverageRewardHordeActorCriticAgent,
    state: AverageRewardHordeActorCriticState,
) -> None:
    if not isinstance(agent, AverageRewardHordeActorCriticAgent):
        raise TypeError("agent must be AverageRewardHordeActorCriticAgent")
    if not isinstance(state, AverageRewardHordeActorCriticState):
        raise TypeError("state must be AverageRewardHordeActorCriticState")
    if agent.config.n_actions != _N_ACTIONS:
        raise ValueError("fixed actor/critic retention protocol requires exactly two actions")
    if state.actor_weights.ndim != 2 or state.actor_weights.shape[0] != _N_ACTIONS:
        raise ValueError("actor weight shape does not match the fixed action space")
    if state.actor_bias.shape != (_N_ACTIONS,):
        raise ValueError("actor bias shape does not match the fixed action space")
    if state.last_observation.shape != (2,):
        raise ValueError("fixed actor/critic retention protocol requires observation_dim=2")
    actor_step = int(state.step_count)
    critic_step = int(state.critic_state.step_count)
    if actor_step < 0 or critic_step < 0:
        raise ValueError("actor/critic step counters must be non-negative")
    if actor_step > _INT32_MAX - _EVENT_COUNT or critic_step > _INT32_MAX - _EVENT_COUNT:
        raise ValueError("actor/critic snapshot lacks fixed update capacity")
    last_action = int(state.last_action)
    sample = state.last_policy_sample
    if last_action == -1:
        if int(sample.action) != -1:
            raise ValueError("unstarted actor/critic state has inconsistent sampled action")
        if not (
            np.isneginf(float(sample.target_log_probability))
            and np.isneginf(float(sample.behavior_log_probability))
        ):
            raise ValueError("unstarted actor/critic state has invalid log-probability sentinels")
    else:
        if last_action < 0 or last_action >= _N_ACTIONS or int(sample.action) != last_action:
            raise ValueError("active actor/critic state has inconsistent sampled action")
        if not (
            math.isfinite(float(sample.target_log_probability))
            and math.isfinite(float(sample.behavior_log_probability))
        ):
            raise ValueError("active actor/critic state has non-finite log probabilities")
    sanitized = state.replace(
        last_policy_sample=sample.replace(
            target_log_probability=jnp.asarray(0.0, dtype=jnp.float32),
            behavior_log_probability=jnp.asarray(0.0, dtype=jnp.float32),
        )
    )
    for index, leaf in enumerate(jax.tree.leaves(sanitized)):
        array = _numpy_leaf(leaf)
        if not np.issubdtype(array.dtype, np.number):
            raise ValueError(f"actor/critic state leaf {index} must be numeric")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"actor/critic state leaf {index} must be finite")


def _copy_state(
    state: AverageRewardHordeActorCriticState,
) -> AverageRewardHordeActorCriticState:
    def copy_leaf(leaf: object) -> object:
        copy_method = getattr(leaf, "copy", None)
        return copy_method() if callable(copy_method) else leaf

    return cast(AverageRewardHordeActorCriticState, jax.tree.map(copy_leaf, state))


def _float_list(value: Array) -> list[float]:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        raise ValueError("policy vector must be one-dimensional and finite")
    return [float(item) for item in array]


def _require_bit_exact_float32(left: object, right: object, *, name: str) -> None:
    left_array = np.ascontiguousarray(np.asarray(left, dtype=np.float32))
    right_array = np.ascontiguousarray(np.asarray(right, dtype=np.float32))
    if left_array.shape != right_array.shape or left_array.tobytes() != right_array.tobytes():
        raise RuntimeError(f"{name} is not bit-exact at the decision boundary")


def _l1(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("policy vectors must have identical lengths")
    return float(sum(abs(a - b) for a, b in zip(left, right, strict=True)))


def _probability_margin(policy: Sequence[float], preferred_action: int) -> float:
    preferred = policy[preferred_action]
    strongest_alternative = max(
        probability for index, probability in enumerate(policy) if index != preferred_action
    )
    return float(preferred - strongest_alternative)


def _execute_trace(
    agent: AverageRewardHordeActorCriticAgent,
    state: AverageRewardHordeActorCriticState,
    config: ActorCriticRetentionConfig,
    protocol: ActorCriticRetentionProtocol,
) -> tuple[list[dict[str, object]], AverageRewardHordeActorCriticState]:
    isolated = _copy_state(state).replace(rng_key=jr.key(config.action_seed))
    initial_actor_parameters = _actor_parameters(isolated)
    initial_critic_parameters = _critic_parameters(isolated)
    first_observation = jnp.asarray(protocol.events[0].observation, dtype=jnp.float32)
    previous_policy_by_case: dict[str, list[float]] = {}
    trace: list[dict[str, object]] = []

    with jax.disable_jit(config.execution_mode == "eager"):
        isolated, _ = agent.start(isolated, first_observation)
        for index, event in enumerate(protocol.events):
            observation = jnp.asarray(event.observation, dtype=jnp.float32)
            next_observation = jnp.asarray(event.next_observation, dtype=jnp.float32)
            if not np.array_equal(np.asarray(isolated.last_observation), np.asarray(observation)):
                raise RuntimeError("actor/critic decision stream lost observation ownership")

            decision = isolated.last_policy_sample
            action = int(isolated.last_action)
            if action < 0 or action >= agent.config.n_actions:
                raise RuntimeError("actor/critic produced an action outside its fixed action space")
            decision_target = _float_list(decision.target_policy)
            decision_behavior = _float_list(decision.behavior_policy)
            fresh_target_array = agent.policy(isolated, observation)
            fresh_behavior_array = agent.behavior_policy(isolated, observation)
            _require_bit_exact_float32(
                decision.target_policy,
                fresh_target_array,
                name="cached and fresh target policy",
            )
            _require_bit_exact_float32(
                decision.behavior_policy,
                fresh_behavior_array,
                name="cached and fresh behavior policy",
            )
            _require_bit_exact_float32(
                decision.target_probability,
                fresh_target_array[action],
                name="cached target action probability",
            )
            _require_bit_exact_float32(
                decision.behavior_probability,
                fresh_behavior_array[action],
                name="cached behavior action probability",
            )
            preupdate_target = _float_list(fresh_target_array)
            preupdate_behavior = _float_list(fresh_behavior_array)
            critic_prediction_array = agent.critic.predict(isolated.critic_state, observation)[0]
            critic_prediction = float(critic_prediction_array)
            if not math.isfinite(critic_prediction):
                raise RuntimeError("actor/critic produced a non-finite critic prediction")
            realized_reward = event.action_rewards[action]
            previous_policy = previous_policy_by_case.get(event.case_id)
            churn = None if previous_policy is None else _l1(decision_target, previous_policy)
            previous_policy_by_case[event.case_id] = decision_target
            actor_before = _actor_parameters(isolated)
            critic_before = _critic_parameters(isolated)
            actor_step_before = int(isolated.step_count)
            critic_step_before = int(isolated.critic_state.step_count)

            result = agent.update(
                isolated,
                jnp.asarray(realized_reward, dtype=jnp.float32),
                next_observation,
            )
            if float(result.critic_prediction) != critic_prediction:
                raise RuntimeError("public critic prediction disagrees with transition update log")
            expected_score_scale = (
                jnp.asarray(1.0 - agent.config.epsilon, dtype=jnp.float32)
                * decision.target_probability
                / jnp.maximum(
                    decision.behavior_probability,
                    jnp.asarray(1.0e-8, dtype=jnp.float32),
                )
            )
            _require_bit_exact_float32(
                result.actor_score_scale,
                expected_score_scale,
                name="epsilon-mixture behavior-score chain-rule scale",
            )
            isolated = result.state
            postupdate_target = _float_list(agent.policy(isolated, observation))
            actor_update_l2 = _parameter_delta_l2(
                actor_before,
                _actor_parameters(isolated),
            )
            critic_update_l2 = _parameter_delta_l2(
                critic_before,
                _critic_parameters(isolated),
            )
            actor_from_initial_l2 = _parameter_delta_l2(
                initial_actor_parameters,
                _actor_parameters(isolated),
            )
            critic_from_initial_l2 = _parameter_delta_l2(
                initial_critic_parameters,
                _critic_parameters(isolated),
            )
            policy_update_l1 = _l1(preupdate_target, postupdate_target)
            critic_error = critic_prediction - event.reference_value_target

            trace.append(
                {
                    "event_index": index,
                    "event_id": event.event_id,
                    "phase_id": event.phase_id,
                    "case_id": event.case_id,
                    "phase_id_learner_visible": False,
                    "targets_learner_visible": False,
                    "reward_table_learner_visible": False,
                    "realized_scalar_reward_learner_visible_after_action": True,
                    "seed_owner": "evaluator",
                    "event_order": list(_EVENT_ORDER),
                    "observation": list(event.observation),
                    "next_observation": list(event.next_observation),
                    "preferred_action": event.preferred_action,
                    "reference_value_target": event.reference_value_target,
                    "action_rewards": list(event.action_rewards),
                    "action": action,
                    "decision_target_policy": decision_target,
                    "decision_behavior_policy": decision_behavior,
                    "decision_target_action_probability": float(
                        decision.target_probability
                    ),
                    "decision_behavior_action_probability": float(
                        decision.behavior_probability
                    ),
                    "target_behavior_action_probability_ratio": float(
                        decision.target_probability
                        / jnp.maximum(
                            decision.behavior_probability,
                            jnp.asarray(1.0e-8, dtype=jnp.float32),
                        )
                    ),
                    "actor_score_scale": float(result.actor_score_scale),
                    "actor_score_scale_expected_from_probabilities": float(
                        expected_score_scale
                    ),
                    "critic_prediction": critic_prediction,
                    "critic_value_squared_error": float(critic_error * critic_error),
                    "actor_action_probability_margin": _probability_margin(
                        decision_target,
                        event.preferred_action,
                    ),
                    "policy_churn_l1_available": churn is not None,
                    "policy_churn_l1": churn,
                    "preupdate_current_target_policy": preupdate_target,
                    "preupdate_current_behavior_policy": preupdate_behavior,
                    "postupdate_current_target_policy": postupdate_target,
                    "policy_update_l1": policy_update_l1,
                    "realized_reward": realized_reward,
                    "actor_parameter_update_l2": actor_update_l2,
                    "critic_parameter_update_l2": critic_update_l2,
                    "actor_parameter_delta_from_initial_l2": actor_from_initial_l2,
                    "critic_parameter_delta_from_initial_l2": critic_from_initial_l2,
                    "td_error": float(result.td_error),
                    "average_reward_after_update": float(result.average_reward),
                    "actor_step_before": actor_step_before,
                    "actor_step_after": int(isolated.step_count),
                    "critic_step_before": critic_step_before,
                    "critic_step_after": int(isolated.critic_state.step_count),
                    "next_action": int(result.action),
                }
            )
    return trace, isolated


def _trace_float(event: Mapping[str, object], name: str) -> float:
    return _finite_float(event.get(name), name=f"trace {name}")


def _trace_int(event: Mapping[str, object], name: str) -> int:
    return _nonnegative_int(event.get(name), name=f"trace {name}")


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot summarize an empty metric sequence")
    return float(sum(values) / len(values))


def _phase_summary(
    phase: ActorCriticRetentionPhase,
    events: Sequence[Mapping[str, object]],
    *,
    recovery_window: int,
) -> dict[str, object]:
    critic_errors = [_trace_float(event, "critic_value_squared_error") for event in events]
    actor_margins = [
        _trace_float(event, "actor_action_probability_margin") for event in events
    ]
    rewards = [_trace_float(event, "realized_reward") for event in events]
    churn = [
        _trace_float(event, "policy_churn_l1")
        for event in events
        if event.get("policy_churn_l1_available") is True
    ]
    policy_updates = [_trace_float(event, "policy_update_l1") for event in events]
    actor_updates = [
        _trace_float(event, "actor_parameter_update_l2") for event in events
    ]
    critic_updates = [
        _trace_float(event, "critic_parameter_update_l2") for event in events
    ]
    actions = [_trace_int(event, "action") for event in events]
    first_critic = _mean(critic_errors[:recovery_window])
    last_critic = _mean(critic_errors[-recovery_window:])
    first_margin = _mean(actor_margins[:recovery_window])
    last_margin = _mean(actor_margins[-recovery_window:])
    first_return = _mean(rewards[:recovery_window])
    last_return = _mean(rewards[-recovery_window:])
    return {
        "phase_id": phase.phase_id,
        "recurrence_of_phase_id": phase.recurrence_of_phase_id,
        "event_count": len(events),
        "recovery_window": recovery_window,
        "mean_critic_value_squared_error": _mean(critic_errors),
        "mean_actor_action_probability_margin": _mean(actor_margins),
        "mean_policy_churn_l1_available_count": len(churn),
        "mean_policy_churn_l1": None if not churn else _mean(churn),
        "mean_policy_update_l1": _mean(policy_updates),
        "total_realized_return": float(sum(rewards)),
        "mean_realized_reward": _mean(rewards),
        "first_window_mean_critic_value_squared_error": first_critic,
        "last_window_mean_critic_value_squared_error": last_critic,
        "within_phase_critic_error_change": last_critic - first_critic,
        "first_window_mean_actor_action_probability_margin": first_margin,
        "last_window_mean_actor_action_probability_margin": last_margin,
        "within_phase_actor_margin_change": last_margin - first_margin,
        "first_window_mean_realized_reward": first_return,
        "last_window_mean_realized_reward": last_return,
        "within_phase_realized_return_recovery_delta": last_return - first_return,
        "actor_parameter_update_l2_sum": float(sum(actor_updates)),
        "critic_parameter_update_l2_sum": float(sum(critic_updates)),
        "action_counts": [actions.count(action) for action in range(_N_ACTIONS)],
        "unique_action_count": len(set(actions)),
        "action_switch_count": sum(
            left != right for left, right in zip(actions, actions[1:], strict=False)
        ),
    }


def reconstruct_actor_critic_retention_summary(
    event_trace: Sequence[Mapping[str, object]],
    protocol: ActorCriticRetentionProtocol,
    *,
    recovery_window: int,
    initial_snapshot: Mapping[str, object],
    final_isolated_state: Mapping[str, object],
) -> dict[str, object]:
    """Reconstruct every descriptive summary solely from retained raw records."""
    if len(event_trace) != len(protocol.events):
        raise ValueError("event trace length does not match the fixed protocol")
    _mapping(initial_snapshot, name="initial_snapshot")
    _mapping(final_isolated_state, name="final_isolated_state")
    _positive_int(recovery_window, name="recovery_window")

    phase_metrics: list[dict[str, object]] = []
    phase_events: dict[str, list[Mapping[str, object]]] = {}
    cursor = 0
    for phase in protocol.phases:
        chunk = list(event_trace[cursor : cursor + phase.event_count])
        phase_events[phase.phase_id] = chunk
        phase_metrics.append(
            _phase_summary(phase, chunk, recovery_window=recovery_window)
        )
        cursor += phase.event_count
    phase_summary_by_id = {
        cast(str, summary["phase_id"]): summary for summary in phase_metrics
    }

    recurrence_metrics: list[dict[str, object]] = []
    for phase in protocol.phases:
        reference_id = phase.recurrence_of_phase_id
        if reference_id is None:
            continue
        reference_events = phase_events[reference_id]
        current_events = phase_events[phase.phase_id]
        reference_by_case = {
            cast(str, event["case_id"]): event for event in reference_events
        }
        current_by_case = {cast(str, event["case_id"]): event for event in current_events}
        case_ids = [cast(str, event["case_id"]) for event in reference_events]
        exact_case_reuse = set(reference_by_case) == set(current_by_case) and len(
            reference_by_case
        ) == len(reference_events)
        policy_l1 = [
            _l1(
                cast(Sequence[float], reference_by_case[case_id]["decision_target_policy"]),
                cast(Sequence[float], current_by_case[case_id]["decision_target_policy"]),
            )
            for case_id in case_ids
        ]
        same_action = [
            _trace_int(reference_by_case[case_id], "action")
            == _trace_int(current_by_case[case_id], "action")
            for case_id in case_ids
        ]
        reference_summary = phase_summary_by_id[reference_id]
        current_summary = phase_summary_by_id[phase.phase_id]
        recurrence_metrics.append(
            {
                "phase_id": phase.phase_id,
                "reference_phase_id": reference_id,
                "case_count": len(case_ids),
                "exact_case_reuse": exact_case_reuse,
                "mean_reference_phase_policy_l1": _mean(policy_l1),
                "same_sampled_action_fraction": float(sum(same_action) / len(same_action)),
                "mean_critic_value_squared_error_delta": (
                    cast(float, current_summary["mean_critic_value_squared_error"])
                    - cast(float, reference_summary["mean_critic_value_squared_error"])
                ),
                "mean_actor_action_probability_margin_delta": (
                    cast(float, current_summary["mean_actor_action_probability_margin"])
                    - cast(float, reference_summary["mean_actor_action_probability_margin"])
                ),
                "mean_realized_reward_delta": (
                    cast(float, current_summary["mean_realized_reward"])
                    - cast(float, reference_summary["mean_realized_reward"])
                ),
                "current_phase_realized_return_recovery_delta": current_summary[
                    "within_phase_realized_return_recovery_delta"
                ],
            }
        )

    actions = [_trace_int(event, "action") for event in event_trace]
    policy_entropies: list[float] = []
    for event in event_trace:
        behavior = cast(Sequence[float], event["decision_behavior_policy"])
        policy_entropies.append(
            float(
                -sum(
                    probability * math.log(max(probability, 1.0e-30))
                    for probability in behavior
                )
            )
        )
    final_event = event_trace[-1]
    plasticity = {
        "actor_parameter_delta_l2": _trace_float(
            final_event, "actor_parameter_delta_from_initial_l2"
        ),
        "critic_parameter_delta_l2": _trace_float(
            final_event, "critic_parameter_delta_from_initial_l2"
        ),
        "actor_parameter_update_l2_sum": float(
            sum(_trace_float(event, "actor_parameter_update_l2") for event in event_trace)
        ),
        "critic_parameter_update_l2_sum": float(
            sum(_trace_float(event, "critic_parameter_update_l2") for event in event_trace)
        ),
        "actor_parameter_update_nonzero_event_count": sum(
            _trace_float(event, "actor_parameter_update_l2") > 0.0 for event in event_trace
        ),
        "critic_parameter_update_nonzero_event_count": sum(
            _trace_float(event, "critic_parameter_update_l2") > 0.0 for event in event_trace
        ),
        "policy_update_l1_sum": float(
            sum(_trace_float(event, "policy_update_l1") for event in event_trace)
        ),
        "policy_update_nonzero_event_count": sum(
            _trace_float(event, "policy_update_l1") > 0.0 for event in event_trace
        ),
        "absolute_td_error_sum": float(
            sum(abs(_trace_float(event, "td_error")) for event in event_trace)
        ),
        "nonzero_td_error_event_count": sum(
            _trace_float(event, "td_error") != 0.0 for event in event_trace
        ),
    }
    action_activity = {
        "action_counts": [actions.count(action) for action in range(_N_ACTIONS)],
        "unique_action_count": len(set(actions)),
        "action_switch_count": sum(
            left != right for left, right in zip(actions, actions[1:], strict=False)
        ),
        "per_action_sampled": [action in actions for action in range(_N_ACTIONS)],
        "mean_decision_behavior_entropy": _mean(policy_entropies),
    }
    return {
        "phase_metrics": phase_metrics,
        "recurrence_metrics": recurrence_metrics,
        "plasticity_diagnostics": plasticity,
        "action_activity_diagnostics": action_activity,
        "claims": {
            "retention_established": False,
            "efficacy_established": False,
            "calibration_established": False,
            "scientific_promotion": False,
            "alberta_plan_completion": False,
        },
    }


def _metric_definitions() -> dict[str, str]:
    return {
        "decision_target_policy": "softmax target policy before fixed epsilon exploration",
        "decision_behavior_policy": (
            "actual action-sampling policy: (1-epsilon) times target plus epsilon-uniform"
        ),
        "actor_score_scale": (
            "exact multiplier (1-epsilon)*target_action_probability/"
            "behavior_action_probability converting grad log target into grad log behavior; "
            "this is not an off-policy importance correction"
        ),
        "critic_value_squared_error": (
            "squared pre-update critic error against the evaluator-only reference value"
        ),
        "actor_action_probability_margin": (
            "decision-time preferred-action target-policy probability minus the strongest "
            "alternative probability"
        ),
        "policy_churn_l1": (
            "L1 distance from the previous decision-time target policy for the same hidden case"
        ),
        "policy_update_l1": (
            "L1 distance between current-policy predictions at the same observation immediately "
            "before and after one update"
        ),
        "realized_return": (
            "undiscounted sum of rewards realized by sampled actions in the continuing phase"
        ),
        "recovery_delta": "last fixed window mean minus first fixed window mean within a phase",
        "parameter_delta_l2": "Euclidean distance over the named stored parameter arrays",
    }


def _resource_accounting(
    *,
    config: ActorCriticRetentionConfig,
    sources: Mapping[str, str],
    initial_snapshot: Mapping[str, object],
    canonical_report_bytes: int,
    root: Path = REPO_ROOT,
) -> dict[str, object]:
    return {
        "phase_count": _PHASE_COUNT,
        "phase_count_limit": config.max_phases,
        "event_count": _EVENT_COUNT,
        "event_count_limit": config.max_events,
        "updates_executed": _EVENT_COUNT,
        "decisions_sampled": _EVENT_COUNT + 1,
        "evaluator_owned_action_seed_count": 1,
        "categorical_action_draw_count": _EVENT_COUNT + 1,
        "learner_visible_phase_identifiers": 0,
        "learner_visible_target_fields": 0,
        "external_snapshot_mutations": 0,
        "source_file_count": len(sources),
        "source_bytes": sum((root / path).stat().st_size for path in sources),
        "initial_snapshot_state_bytes": initial_snapshot["state_bytes"],
        "initial_snapshot_state_byte_limit": config.max_initial_snapshot_bytes,
        "canonical_report_bytes": canonical_report_bytes,
        "canonical_report_byte_limit": config.max_report_bytes,
        "execution_mode": config.execution_mode,
        "ordinary_policy_gradient_lanes": 1,
        "paper_specific_dg_delight_lanes": 0,
        "candidate_update_safety_audits": 0,
    }


def _assemble_report(
    *,
    config: ActorCriticRetentionConfig,
    protocol: ActorCriticRetentionProtocol,
    sources: Mapping[str, str],
    initial_snapshot: Mapping[str, object],
    trace: Sequence[Mapping[str, object]],
    final_state: Mapping[str, object],
    summary: Mapping[str, object],
    resources: Mapping[str, object],
) -> dict[str, object]:
    config_payload = config.to_config()
    protocol_payload = protocol.to_config()
    trace_payload = [dict(event) for event in trace]
    hashes = {
        "config_sha256": _canonical_sha256(config_payload),
        "protocol_sha256": _canonical_sha256(protocol_payload),
        "source_manifest_sha256": _canonical_sha256(sources),
        "initial_snapshot_sha256": _canonical_sha256(initial_snapshot),
        "event_trace_sha256": _canonical_sha256(trace_payload),
        "final_isolated_state_sha256": _canonical_sha256(final_state),
        "summary_sha256": _canonical_sha256(summary),
        "resource_accounting_sha256": _canonical_sha256(resources),
    }
    payload: dict[str, object] = {
        "development_only": True,
        "development_status": DEVELOPMENT_STATUS,
        "assessment_status": "not-assessed",
        "policy_gradient_mode": "ordinary",
        "target_policy_semantics": "softmax-target-before-epsilon-exploration",
        "behavior_policy_semantics": "fixed-epsilon-mixture-used-for-action-sampling",
        "actor_score_chain_rule_semantics": (
            "exact-epsilon-mixture-behavior-score-chain-rule-scale"
        ),
        "off_policy_target_policy_correction_claimed": False,
        "paper_specific_dg_delight_used": False,
        "candidate_update_safety_audit_performed": False,
        "retention_claimed": False,
        "efficacy_claimed": False,
        "calibration_claimed": False,
        "scientific_promotion_allowed": False,
        "performance_thresholds_applied": False,
        "config": config_payload,
        "protocol": protocol_payload,
        "source_sha256": dict(sources),
        "initial_snapshot": dict(initial_snapshot),
        "metric_definitions": _metric_definitions(),
        "event_trace": trace_payload,
        "final_isolated_state": dict(final_state),
        "summary": dict(summary),
        "resource_accounting": dict(resources),
        "hashes": hashes,
        "limitations": list(_LIMITATIONS),
    }
    return {
        "schema": ACTOR_CRITIC_RETENTION_REPORT_SCHEMA,
        "payload": payload,
        "payload_sha256": _canonical_sha256(payload),
    }


def build_actor_critic_retention_report(
    agent: AverageRewardHordeActorCriticAgent,
    state: AverageRewardHordeActorCriticState,
    config: ActorCriticRetentionConfig,
    *,
    root: Path = REPO_ROOT,
) -> dict[str, object]:
    """Run the fixed prequential probe against an isolated snapshot copy."""
    if not isinstance(config, ActorCriticRetentionConfig):
        raise TypeError("config must be ActorCriticRetentionConfig")
    _validate_agent_state(agent, state)
    protocol = canonical_actor_critic_retention_protocol()
    initial_snapshot = _state_descriptor(agent, state)
    snapshot_hash = frozen_actor_critic_state_sha256(state)
    if cast(int, initial_snapshot["state_bytes"]) > config.max_initial_snapshot_bytes:
        raise ValueError("initial actor/critic snapshot exceeds snapshot byte bound")

    trace, final = _execute_trace(agent, state, config, protocol)
    if frozen_actor_critic_state_sha256(state) != snapshot_hash:
        raise RuntimeError("actor/critic retention probe mutated the supplied snapshot")
    final_snapshot = _state_descriptor(agent, final)
    summary = reconstruct_actor_critic_retention_summary(
        trace,
        protocol,
        recovery_window=config.recovery_window,
        initial_snapshot=initial_snapshot,
        final_isolated_state=final_snapshot,
    )
    sources = actor_critic_retention_source_snapshot(root)

    report_size = 0
    report: dict[str, object] | None = None
    for _ in range(16):
        resources = _resource_accounting(
            config=config,
            sources=sources,
            initial_snapshot=initial_snapshot,
            canonical_report_bytes=report_size,
            root=root,
        )
        report = _assemble_report(
            config=config,
            protocol=protocol,
            sources=sources,
            initial_snapshot=initial_snapshot,
            trace=trace,
            final_state=final_snapshot,
            summary=summary,
            resources=resources,
        )
        measured = len(_canonical_json_bytes(report)) + 1
        if measured == report_size:
            break
        report_size = measured
    else:
        raise RuntimeError("canonical actor/critic retention report size did not converge")
    assert report is not None
    if report_size > config.max_report_bytes:
        raise ValueError("canonical actor/critic retention report exceeds report byte bound")
    validation = validate_actor_critic_retention_report(report, root=root)
    if not validation.valid:
        raise ValueError(
            "internal actor/critic retention validation failed: "
            + "; ".join(validation.errors)
        )
    return report


@dataclasses.dataclass(frozen=True)
class ActorCriticRetentionValidation:
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
    "reward_table_learner_visible",
    "realized_scalar_reward_learner_visible_after_action",
    "seed_owner",
    "event_order",
    "observation",
    "next_observation",
    "preferred_action",
    "reference_value_target",
    "action_rewards",
    "action",
    "decision_target_policy",
    "decision_behavior_policy",
    "decision_target_action_probability",
    "decision_behavior_action_probability",
    "target_behavior_action_probability_ratio",
    "actor_score_scale",
    "actor_score_scale_expected_from_probabilities",
    "critic_prediction",
    "critic_value_squared_error",
    "actor_action_probability_margin",
    "policy_churn_l1_available",
    "policy_churn_l1",
    "preupdate_current_target_policy",
    "preupdate_current_behavior_policy",
    "postupdate_current_target_policy",
    "policy_update_l1",
    "realized_reward",
    "actor_parameter_update_l2",
    "critic_parameter_update_l2",
    "actor_parameter_delta_from_initial_l2",
    "critic_parameter_delta_from_initial_l2",
    "td_error",
    "average_reward_after_update",
    "actor_step_before",
    "actor_step_after",
    "critic_step_before",
    "critic_step_after",
    "next_action",
}


def _validate_digest(value: object, *, name: str) -> str:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _validate_probability_vector(value: object, *, name: str) -> list[float]:
    raw = _list(value, name=name)
    if len(raw) != _N_ACTIONS:
        raise ValueError(f"{name} must contain exactly {_N_ACTIONS} probabilities")
    probabilities = [
        _finite_float(item, name=f"{name}[{index}]") for index, item in enumerate(raw)
    ]
    if any(probability < 0.0 or probability > 1.0 for probability in probabilities):
        raise ValueError(f"{name} probabilities must be in [0, 1]")
    if not math.isclose(sum(probabilities), 1.0, rel_tol=0.0, abs_tol=1.0e-6):
        raise ValueError(f"{name} probabilities must sum to one")
    return probabilities


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
        "n_actions",
        "actor_step_count",
        "critic_step_count",
        "actor_parameters_sha256",
        "critic_parameters_sha256",
    }
    if set(descriptor) != expected_fields:
        raise ValueError(f"{name} fields do not match v2")
    agent_config = _mapping(descriptor.get("agent_config"), name=f"{name} agent_config")
    try:
        agent = AverageRewardHordeActorCriticAgent.from_config(dict(agent_config))
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
    ):
        _validate_digest(descriptor.get(digest_name), name=f"{name} {digest_name}")
    _positive_int(descriptor.get("state_bytes"), name=f"{name} state_bytes")
    if _positive_int(descriptor.get("observation_dim"), name=f"{name} observation_dim") != 2:
        raise ValueError(f"{name} observation_dim must match the fixed protocol")
    if _positive_int(descriptor.get("n_actions"), name=f"{name} n_actions") != _N_ACTIONS:
        raise ValueError(f"{name} n_actions must match the fixed protocol")
    _nonnegative_int(descriptor.get("actor_step_count"), name=f"{name} actor_step_count")
    _nonnegative_int(descriptor.get("critic_step_count"), name=f"{name} critic_step_count")
    return descriptor


def _validate_trace(
    value: object,
    protocol: ActorCriticRetentionProtocol,
    initial_snapshot: Mapping[str, object],
) -> list[Mapping[str, object]]:
    values = _list(value, name="event_trace")
    if len(values) != len(protocol.events):
        raise ValueError("event trace length does not match the fixed protocol")
    trace: list[Mapping[str, object]] = []
    previous_policy_by_case: dict[str, list[float]] = {}
    initial_actor_step = _nonnegative_int(
        initial_snapshot.get("actor_step_count"), name="initial actor_step_count"
    )
    initial_critic_step = _nonnegative_int(
        initial_snapshot.get("critic_step_count"), name="initial critic_step_count"
    )
    agent_config = _mapping(
        initial_snapshot.get("agent_config"), name="initial actor/critic agent_config"
    )
    replay_agent = AverageRewardHordeActorCriticAgent.from_config(dict(agent_config))
    epsilon = replay_agent.config.epsilon
    for index, (raw, event) in enumerate(zip(values, protocol.events, strict=True)):
        record = _mapping(raw, name=f"event_trace[{index}]")
        if set(record) != _TRACE_FIELDS:
            raise ValueError(f"event trace {index} fields do not match v2")
        expected_annotations: dict[str, object] = {
            "event_index": index,
            "event_id": event.event_id,
            "phase_id": event.phase_id,
            "case_id": event.case_id,
            "phase_id_learner_visible": False,
            "targets_learner_visible": False,
            "reward_table_learner_visible": False,
            "realized_scalar_reward_learner_visible_after_action": True,
            "seed_owner": "evaluator",
            "event_order": list(_EVENT_ORDER),
            "observation": list(event.observation),
            "next_observation": list(event.next_observation),
            "preferred_action": event.preferred_action,
            "reference_value_target": event.reference_value_target,
            "action_rewards": list(event.action_rewards),
        }
        for field, expected in expected_annotations.items():
            if not _strict_json_equal(record.get(field), expected):
                raise ValueError(f"event trace {index} evaluator annotation {field} changed")
        action = _nonnegative_int(record.get("action"), name=f"event trace {index} action")
        if action >= _N_ACTIONS:
            raise ValueError(f"event trace {index} action is outside the fixed action space")
        target_policy = _validate_probability_vector(
            record.get("decision_target_policy"),
            name=f"event trace {index} decision_target_policy",
        )
        behavior_policy = _validate_probability_vector(
            record.get("decision_behavior_policy"),
            name=f"event trace {index} decision_behavior_policy",
        )
        target_probability = _finite_float(
            record.get("decision_target_action_probability"),
            name=f"event trace {index} decision_target_action_probability",
        )
        behavior_probability = _finite_float(
            record.get("decision_behavior_action_probability"),
            name=f"event trace {index} decision_behavior_action_probability",
        )
        if target_probability != target_policy[action]:
            raise ValueError(f"event trace {index} target action probability changed")
        if behavior_probability != behavior_policy[action]:
            raise ValueError(f"event trace {index} behavior action probability changed")
        ratio = _finite_float(
            record.get("target_behavior_action_probability_ratio"),
            name=f"event trace {index} target_behavior_action_probability_ratio",
        )
        expected_ratio = (
            jnp.asarray(target_probability, dtype=jnp.float32)
            / jnp.maximum(
                jnp.asarray(behavior_probability, dtype=jnp.float32),
                jnp.asarray(1.0e-8, dtype=jnp.float32),
            )
        )
        _require_bit_exact_float32(
            ratio,
            expected_ratio,
            name=f"event trace {index} target/behavior probability ratio",
        )
        actor_score_scale = _finite_float(
            record.get("actor_score_scale"),
            name=f"event trace {index} actor_score_scale",
        )
        expected_score_scale = (
            jnp.asarray(1.0 - epsilon, dtype=jnp.float32)
            * jnp.asarray(target_probability, dtype=jnp.float32)
            / jnp.maximum(
                jnp.asarray(behavior_probability, dtype=jnp.float32),
                jnp.asarray(1.0e-8, dtype=jnp.float32),
            )
        )
        _require_bit_exact_float32(
            actor_score_scale,
            expected_score_scale,
            name=f"event trace {index} actor score scale",
        )
        logged_expected_score_scale = _finite_float(
            record.get("actor_score_scale_expected_from_probabilities"),
            name=f"event trace {index} actor_score_scale_expected_from_probabilities",
        )
        _require_bit_exact_float32(
            logged_expected_score_scale,
            expected_score_scale,
            name=f"event trace {index} logged actor score expectation",
        )
        critic_prediction = _finite_float(
            record.get("critic_prediction"), name=f"event trace {index} critic_prediction"
        )
        critic_squared_error = _finite_float(
            record.get("critic_value_squared_error"),
            name=f"event trace {index} critic_value_squared_error",
        )
        expected_error = (critic_prediction - event.reference_value_target) ** 2
        if critic_squared_error != expected_error:
            raise ValueError(f"event trace {index} critic squared error does not reconstruct")
        margin = _finite_float(
            record.get("actor_action_probability_margin"),
            name=f"event trace {index} actor_action_probability_margin",
        )
        if margin != _probability_margin(target_policy, event.preferred_action):
            raise ValueError(f"event trace {index} actor margin does not reconstruct")
        if type(record.get("policy_churn_l1_available")) is not bool:
            raise ValueError(f"event trace {index} policy churn availability must be boolean")
        previous = previous_policy_by_case.get(event.case_id)
        expected_churn = None if previous is None else _l1(target_policy, previous)
        if record.get("policy_churn_l1_available") is not (expected_churn is not None):
            raise ValueError(f"event trace {index} policy churn availability changed")
        if expected_churn is None:
            if record.get("policy_churn_l1") is not None:
                raise ValueError(f"event trace {index} unavailable policy churn must be null")
        elif _finite_float(
            record.get("policy_churn_l1"), name=f"event trace {index} policy_churn_l1"
        ) != expected_churn:
            raise ValueError(f"event trace {index} policy churn does not reconstruct")
        previous_policy_by_case[event.case_id] = target_policy
        preupdate = _validate_probability_vector(
            record.get("preupdate_current_target_policy"),
            name=f"event trace {index} preupdate_current_target_policy",
        )
        preupdate_behavior = _validate_probability_vector(
            record.get("preupdate_current_behavior_policy"),
            name=f"event trace {index} preupdate_current_behavior_policy",
        )
        _require_bit_exact_float32(
            target_policy,
            preupdate,
            name=f"event trace {index} cached/fresh target policy",
        )
        _require_bit_exact_float32(
            behavior_policy,
            preupdate_behavior,
            name=f"event trace {index} cached/fresh behavior policy",
        )
        postupdate = _validate_probability_vector(
            record.get("postupdate_current_target_policy"),
            name=f"event trace {index} postupdate_current_target_policy",
        )
        policy_update = _finite_float(
            record.get("policy_update_l1"), name=f"event trace {index} policy_update_l1"
        )
        if policy_update != _l1(preupdate, postupdate):
            raise ValueError(f"event trace {index} policy update does not reconstruct")
        realized_reward = _finite_float(
            record.get("realized_reward"), name=f"event trace {index} realized_reward"
        )
        if realized_reward != event.action_rewards[action]:
            raise ValueError(f"event trace {index} realized reward does not match sampled action")
        for metric in (
            "actor_parameter_update_l2",
            "critic_parameter_update_l2",
            "actor_parameter_delta_from_initial_l2",
            "critic_parameter_delta_from_initial_l2",
        ):
            if _finite_float(record.get(metric), name=f"event trace {index} {metric}") < 0.0:
                raise ValueError(f"event trace {index} {metric} must be non-negative")
        _finite_float(record.get("td_error"), name=f"event trace {index} td_error")
        _finite_float(
            record.get("average_reward_after_update"),
            name=f"event trace {index} average_reward_after_update",
        )
        expected_actor_before = initial_actor_step + index
        expected_critic_before = initial_critic_step + index
        counters = {
            "actor_step_before": expected_actor_before,
            "actor_step_after": expected_actor_before + 1,
            "critic_step_before": expected_critic_before,
            "critic_step_after": expected_critic_before + 1,
        }
        for field, expected in counters.items():
            if record.get(field) != expected or type(record.get(field)) is not int:
                raise ValueError(f"event trace {index} {field} breaks exact update ordering")
        next_action = _nonnegative_int(
            record.get("next_action"), name=f"event trace {index} next_action"
        )
        if next_action >= _N_ACTIONS:
            raise ValueError(f"event trace {index} next_action is outside the action space")
        if index + 1 < len(values):
            next_record = _mapping(values[index + 1], name=f"event_trace[{index + 1}]")
            if next_record.get("action") != next_action:
                raise ValueError(f"event trace {index} next decision ownership changed")
        trace.append(record)
    return trace


def _validate_report_or_raise(
    report: Mapping[str, object],
    *,
    root: Path,
) -> ActorCriticRetentionConfig:
    if set(report) != {"schema", "payload", "payload_sha256"}:
        raise ValueError("actor/critic retention report fields do not match v2")
    if report.get("schema") != ACTOR_CRITIC_RETENTION_REPORT_SCHEMA:
        raise ValueError("actor/critic retention report schema is invalid")
    payload = _mapping(report.get("payload"), name="payload")
    expected_payload_fields = {
        "development_only",
        "development_status",
        "assessment_status",
        "policy_gradient_mode",
        "target_policy_semantics",
        "behavior_policy_semantics",
        "actor_score_chain_rule_semantics",
        "off_policy_target_policy_correction_claimed",
        "paper_specific_dg_delight_used",
        "candidate_update_safety_audit_performed",
        "retention_claimed",
        "efficacy_claimed",
        "calibration_claimed",
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
        raise ValueError("actor/critic retention payload fields do not match v2")
    fixed: dict[str, object] = {
        "development_only": True,
        "development_status": DEVELOPMENT_STATUS,
        "assessment_status": "not-assessed",
        "policy_gradient_mode": "ordinary",
        "target_policy_semantics": "softmax-target-before-epsilon-exploration",
        "behavior_policy_semantics": "fixed-epsilon-mixture-used-for-action-sampling",
        "actor_score_chain_rule_semantics": (
            "exact-epsilon-mixture-behavior-score-chain-rule-scale"
        ),
        "off_policy_target_policy_correction_claimed": False,
        "paper_specific_dg_delight_used": False,
        "candidate_update_safety_audit_performed": False,
        "retention_claimed": False,
        "efficacy_claimed": False,
        "calibration_claimed": False,
        "scientific_promotion_allowed": False,
        "performance_thresholds_applied": False,
    }
    for name, expected in fixed.items():
        if not _strict_json_equal(payload.get(name), expected):
            raise ValueError(f"actor/critic retention payload {name} is invalid")
    if report.get("payload_sha256") != _canonical_sha256(payload):
        raise ValueError("actor/critic retention payload digest does not match")

    config_value = _mapping(payload.get("config"), name="config")
    config = ActorCriticRetentionConfig.from_config(config_value)
    protocol_value = _mapping(payload.get("protocol"), name="protocol")
    protocol = ActorCriticRetentionProtocol.from_config(protocol_value)
    sources = _mapping(payload.get("source_sha256"), name="source_sha256")
    current_sources = actor_critic_retention_source_snapshot(root)
    if not _strict_json_equal(dict(sources), current_sources):
        raise ValueError("actor/critic retention source hashes do not match current sources")
    for path, digest in sources.items():
        if path not in {source.as_posix() for source in SOURCE_PATHS}:
            raise ValueError("actor/critic retention source manifest has an unknown path")
        _validate_digest(digest, name=f"source digest {path}")

    initial_snapshot = _validate_state_descriptor(
        payload.get("initial_snapshot"), name="initial_snapshot"
    )
    final_snapshot = _validate_state_descriptor(
        payload.get("final_isolated_state"), name="final_isolated_state"
    )
    if not _strict_json_equal(
        initial_snapshot.get("agent_config"), final_snapshot.get("agent_config")
    ):
        raise ValueError("final isolated state changed the actor/critic construction")
    expected_final_actor_step = (
        _nonnegative_int(
            initial_snapshot.get("actor_step_count"), name="initial actor step count"
        )
        + _EVENT_COUNT
    )
    expected_final_critic_step = (
        _nonnegative_int(
            initial_snapshot.get("critic_step_count"), name="initial critic step count"
        )
        + _EVENT_COUNT
    )
    if final_snapshot.get("actor_step_count") != expected_final_actor_step:
        raise ValueError("final isolated actor step count breaks exact event ordering")
    if final_snapshot.get("critic_step_count") != expected_final_critic_step:
        raise ValueError("final isolated critic step count breaks exact event ordering")
    if (
        _positive_int(initial_snapshot.get("state_bytes"), name="initial state_bytes")
        > config.max_initial_snapshot_bytes
    ):
        raise ValueError("initial snapshot exceeds configured snapshot byte bound")

    if not _strict_json_equal(payload.get("metric_definitions"), _metric_definitions()):
        raise ValueError("actor/critic retention metric definitions changed")
    trace = _validate_trace(payload.get("event_trace"), protocol, initial_snapshot)
    summary = _mapping(payload.get("summary"), name="summary")
    expected_summary = reconstruct_actor_critic_retention_summary(
        trace,
        protocol,
        recovery_window=config.recovery_window,
        initial_snapshot=initial_snapshot,
        final_isolated_state=final_snapshot,
    )
    if not _strict_json_equal(dict(summary), expected_summary):
        raise ValueError("actor/critic retention summary does not reconstruct from raw trace")

    resources = _mapping(payload.get("resource_accounting"), name="resource_accounting")
    canonical_size = len(_canonical_json_bytes(report)) + 1
    expected_resources = _resource_accounting(
        config=config,
        sources=cast(Mapping[str, str], sources),
        initial_snapshot=initial_snapshot,
        canonical_report_bytes=canonical_size,
        root=root,
    )
    if not _strict_json_equal(dict(resources), expected_resources):
        raise ValueError("actor/critic retention resource accounting does not reconstruct")
    if canonical_size > config.max_report_bytes:
        raise ValueError("actor/critic retention report exceeds configured report byte bound")

    limitations = payload.get("limitations")
    if not _strict_json_equal(limitations, list(_LIMITATIONS)):
        raise ValueError("actor/critic retention limitations changed")
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
    expected_hashes = {name: _canonical_sha256(value) for name, value in hash_inputs.items()}
    if not _strict_json_equal(dict(hashes), expected_hashes):
        raise ValueError("actor/critic retention component hashes do not reconstruct")
    return config


def validate_actor_critic_retention_report(
    report: Mapping[str, object],
    *,
    root: Path = REPO_ROOT,
    agent: AverageRewardHordeActorCriticAgent | None = None,
    state: AverageRewardHordeActorCriticState | None = None,
) -> ActorCriticRetentionValidation:
    """Validate structure/source binding and optionally replay the exact snapshot."""
    errors: list[str] = []
    config: ActorCriticRetentionConfig | None = None
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
            if not _strict_json_equal(agent.to_config(), initial.get("agent_config")):
                raise ValueError("live replay agent construction does not match report")
            if frozen_actor_critic_state_sha256(state) != initial.get("state_sha256"):
                raise ValueError("live replay snapshot does not match report")
            before = frozen_actor_critic_state_sha256(state)
            replay = build_actor_critic_retention_report(agent, state, config, root=root)
            if frozen_actor_critic_state_sha256(state) != before:
                raise RuntimeError("live replay mutated the supplied snapshot")
            if not _strict_json_equal(replay, dict(report)):
                raise ValueError("live replay does not reproduce the exact report")
        except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
            errors.append(str(error))
    return ActorCriticRetentionValidation(
        valid=not errors,
        assessment_status="not-assessed",
        errors=tuple(errors),
    )


def canonical_actor_critic_retention_report_bytes(
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


def load_actor_critic_retention_report(
    path: str | Path,
    *,
    root: Path = REPO_ROOT,
) -> dict[str, object]:
    """Load exact canonical JSON and require full current-source validation."""
    data = Path(path).read_bytes()
    try:
        decoded = json.loads(data, object_pairs_hook=_reject_duplicate_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"actor/critic retention report is not valid JSON: {error}") from error
    report = dict(_mapping(decoded, name="report"))
    if data != canonical_actor_critic_retention_report_bytes(report):
        raise ValueError("actor/critic retention report is not exact canonical JSON")
    validation = validate_actor_critic_retention_report(report, root=root)
    if not validation.valid:
        raise ValueError(
            "actor/critic retention report is invalid: " + "; ".join(validation.errors)
        )
    return report


def save_actor_critic_retention_report(
    report: Mapping[str, object],
    path: str | Path,
    *,
    root: Path = REPO_ROOT,
) -> None:
    """Atomically create one validated canonical report without overwrite."""
    validation = validate_actor_critic_retention_report(report, root=root)
    if not validation.valid:
        raise ValueError(
            "refusing to save invalid actor/critic retention report: "
            + "; ".join(validation.errors)
        )
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(destination):
        raise FileExistsError(f"refusing to overwrite actor/critic retention report: {destination}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_actor_critic_retention_report_bytes(report))
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError as error:
            raise FileExistsError(
                f"refusing to overwrite concurrently created actor/critic retention report: "
                f"{destination}"
            ) from error
    finally:
        temporary.unlink(missing_ok=True)


def save_actor_critic_retention_snapshot_checkpoint(
    agent: AverageRewardHordeActorCriticAgent,
    state: AverageRewardHordeActorCriticState,
    path: str | Path,
    *,
    root: Path = REPO_ROOT,
) -> None:
    """Create a source-, construction-, and state-bound snapshot checkpoint."""
    _validate_agent_state(agent, state)
    destination = Path(path).expanduser()
    if os.path.lexists(destination) or os.path.lexists(destination.resolve()):
        raise FileExistsError(
            f"refusing to overwrite actor/critic retention snapshot: {destination}"
        )
    snapshot = _state_descriptor(agent, state)
    sources = actor_critic_retention_source_snapshot(root)
    save_checkpoint(
        state,
        destination,
        metadata={
            "schema": ACTOR_CRITIC_RETENTION_CHECKPOINT_SCHEMA,
            "development_status": DEVELOPMENT_STATUS,
            "assessment_status": "not-assessed",
            "scientific_promotion_allowed": False,
            "retention_claimed": False,
            "efficacy_claimed": False,
            "calibration_claimed": False,
            "policy_gradient_mode": "ordinary",
            "behavior_policy_objective": "fixed-epsilon-mixture",
            "off_policy_target_policy_correction_claimed": False,
            "paper_specific_dg_delight_used": False,
            "snapshot": snapshot,
            "snapshot_sha256": _canonical_sha256(snapshot),
            "source_sha256": sources,
            "source_manifest_sha256": _canonical_sha256(sources),
        },
    )


def load_actor_critic_retention_snapshot_checkpoint(
    path: str | Path,
    *,
    template_key: Array | None = None,
    root: Path = REPO_ROOT,
) -> tuple[AverageRewardHordeActorCriticAgent, AverageRewardHordeActorCriticState]:
    """Restore only an exact current-source actor/critic diagnostic snapshot."""
    metadata = load_checkpoint_metadata(path)
    expected_fields = {
        "schema",
        "development_status",
        "assessment_status",
        "scientific_promotion_allowed",
        "retention_claimed",
        "efficacy_claimed",
        "calibration_claimed",
        "policy_gradient_mode",
        "behavior_policy_objective",
        "off_policy_target_policy_correction_claimed",
        "paper_specific_dg_delight_used",
        "snapshot",
        "snapshot_sha256",
        "source_sha256",
        "source_manifest_sha256",
    }
    if set(metadata) != expected_fields:
        raise ValueError("actor/critic retention snapshot metadata fields do not match v2")
    fixed: dict[str, object] = {
        "schema": ACTOR_CRITIC_RETENTION_CHECKPOINT_SCHEMA,
        "development_status": DEVELOPMENT_STATUS,
        "assessment_status": "not-assessed",
        "scientific_promotion_allowed": False,
        "retention_claimed": False,
        "efficacy_claimed": False,
        "calibration_claimed": False,
        "policy_gradient_mode": "ordinary",
        "behavior_policy_objective": "fixed-epsilon-mixture",
        "off_policy_target_policy_correction_claimed": False,
        "paper_specific_dg_delight_used": False,
    }
    for name, expected in fixed.items():
        if not _strict_json_equal(metadata.get(name), expected):
            raise ValueError(f"actor/critic retention snapshot {name} is invalid")
    sources = actor_critic_retention_source_snapshot(root)
    if not _strict_json_equal(metadata.get("source_sha256"), sources):
        raise ValueError("actor/critic retention snapshot source hashes do not match")
    if metadata.get("source_manifest_sha256") != _canonical_sha256(sources):
        raise ValueError("actor/critic retention snapshot source manifest digest does not match")
    snapshot = _validate_state_descriptor(metadata.get("snapshot"), name="snapshot")
    if metadata.get("snapshot_sha256") != _canonical_sha256(snapshot):
        raise ValueError("actor/critic retention snapshot descriptor digest does not match")
    agent_config = _mapping(snapshot.get("agent_config"), name="snapshot agent_config")
    agent = AverageRewardHordeActorCriticAgent.from_config(dict(agent_config))
    key = jr.key(0) if template_key is None else template_key
    observation_dim = _positive_int(
        snapshot.get("observation_dim"), name="snapshot observation_dim"
    )
    template = agent.init(observation_dim, key)
    restored_value, restored_metadata = load_checkpoint(template, path)
    if not isinstance(restored_value, AverageRewardHordeActorCriticState):
        raise ValueError("actor/critic retention checkpoint state type is invalid")
    restored = restored_value
    if restored_metadata != metadata:
        raise ValueError("actor/critic retention snapshot metadata changed between reads")
    _validate_agent_state(agent, restored)
    if not _strict_json_equal(_state_descriptor(agent, restored), snapshot):
        raise ValueError("restored actor/critic retention state does not match descriptor")
    return agent, restored


class ActorCriticRetentionEvaluator:
    """Immutable adapter binding the fixed protocol and one diagnostic config."""

    def __init__(self, config: ActorCriticRetentionConfig) -> None:
        if not isinstance(config, ActorCriticRetentionConfig):
            raise TypeError("config must be ActorCriticRetentionConfig")
        self._config = config

    @property
    def config(self) -> ActorCriticRetentionConfig:
        return self._config

    @property
    def protocol(self) -> ActorCriticRetentionProtocol:
        return canonical_actor_critic_retention_protocol()

    def evaluate(
        self,
        agent: AverageRewardHordeActorCriticAgent,
        state: AverageRewardHordeActorCriticState,
        *,
        root: Path = REPO_ROOT,
    ) -> dict[str, object]:
        """Run the bound evaluator without mutating the supplied snapshot."""
        return build_actor_critic_retention_report(
            agent,
            state,
            self._config,
            root=root,
        )


__all__ = [
    "ACTOR_CRITIC_RETENTION_CHECKPOINT_SCHEMA",
    "ACTOR_CRITIC_RETENTION_CONFIG_SCHEMA",
    "ACTOR_CRITIC_RETENTION_PROTOCOL_SCHEMA",
    "ACTOR_CRITIC_RETENTION_REPORT_SCHEMA",
    "ActorCriticRetentionConfig",
    "ActorCriticRetentionEvaluator",
    "ActorCriticRetentionEvent",
    "ActorCriticRetentionPhase",
    "ActorCriticRetentionProtocol",
    "ActorCriticRetentionValidation",
    "actor_critic_retention_source_snapshot",
    "build_actor_critic_retention_report",
    "canonical_actor_critic_retention_protocol",
    "canonical_actor_critic_retention_report_bytes",
    "frozen_actor_critic_state_sha256",
    "load_actor_critic_retention_report",
    "load_actor_critic_retention_snapshot_checkpoint",
    "reconstruct_actor_critic_retention_summary",
    "save_actor_critic_retention_report",
    "save_actor_critic_retention_snapshot_checkpoint",
    "validate_actor_critic_retention_report",
]
