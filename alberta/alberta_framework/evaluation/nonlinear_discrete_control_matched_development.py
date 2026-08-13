# mypy: disable-error-code="arg-type,attr-defined,call-arg,no-any-return,no-untyped-call"
"""Matched nonlinear actor-critic and differential-SARSA control diagnostics.

This is a permanently nonpromoting WP6 development lane.  It compares the
repository's bounded nonlinear discrete differential actor-critic with the
repository's linear differential SARSA reference on one uninterrupted
six-state RiverSwim-style A/B/A life.  The transition kernel never resets;
regime B only swaps the valuable boundary.

Both arms receive the same decision-indexed categorical sample keys and the
same transition keys.  That is common exogenous randomness, not a claim that
realized trajectories remain paired after actions or states diverge.  The
nonlinear arm samples the epsilon-uniform mixture of its current softmax
target through its owned receipt API.  SARSA uses its public external-action
surface with an evaluator-owned epsilon-uniform mixture of a lowest-index
greedy target.  SARSA chooses the successor from pre-update Q values, while
the nonlinear transaction chooses its successor from the committed
post-update actor.  The report preserves this unavoidable timing mismatch.

Every outcome is prequential: the action and complete target/behavior
policies are bound before the environment outcome.  Reports retain the full
per-arm trace, actor/critic-or-action-value/reward-rate/churn/recovery
diagnostics, exact persistent array bytes, and fixed logical mutation
opportunities.  They also perform a source/config/seed-bound temporary
checkpoint round trip before continuing.  Validation replays the entire life.

There is no winner rule, performance threshold, held-out seed, output writer,
evidence promotion, SOTA claim, or Alberta Plan completion claim.  Logical
work is not FLOPs, wall clock, energy, allocator peak, or device residency.
The arms have unequal parameterizations and successor-policy timing, so exact
realized work matching is explicitly false even when both commit every update.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array
from jaxtyping import Float, Int, UInt

from alberta_framework.core.average_reward import (
    DifferentialSARSAAgent,
    DifferentialSARSAConfig,
    DifferentialSARSAState,
    measure_differential_sarsa_state_nbytes,
)
from alberta_framework.core.checkpoints import (
    load_checkpoint,
    load_checkpoint_metadata,
    save_checkpoint,
)
from alberta_framework.core.nonlinear_average_reward_actor_critic import (
    NonlinearAverageRewardActionRecord,
    NonlinearAverageRewardActorCritic,
    NonlinearAverageRewardActorCriticConfig,
    NonlinearAverageRewardActorCriticState,
)
from alberta_framework.streams.closed_loop import (
    LEFT_ACTION,
    RIGHT_ACTION,
    RiverSwimConfig,
    RiverSwimMDP,
    RiverSwimState,
)

MATCHED_DISCRETE_CONTROL_CONFIG_SCHEMA = (
    "alberta.nonlinear-discrete-control-matched-development.config.v1"
)
MATCHED_DISCRETE_CONTROL_REPORT_SCHEMA = (
    "alberta.nonlinear-discrete-control-matched-development.report.v1"
)
MATCHED_DISCRETE_CONTROL_CHECKPOINT_SCHEMA = (
    "alberta.nonlinear-discrete-control-matched-development.checkpoint.v1"
)
ASSESSMENT_STATUS = "not_assessed"
DEVELOPMENT_ONLY = True
SCIENTIFIC_PROMOTION_ALLOWED = False
WINNER_SELECTION_ALLOWED = False
PERFORMANCE_THRESHOLDS_APPLIED = False
OUTPUT_WRITES = False
REGIME_SCHEDULE: tuple[str, str, str] = ("A", "B", "A")
ARM_ORDER: tuple[str, str] = ("nonlinear_actor_critic", "differential_sarsa")

_UINT32_MAX = 2**32 - 1
_INT32_MAX = 2**31 - 1
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_PATHS = (
    Path("alberta_framework/core/average_reward.py"),
    Path("alberta_framework/core/checkpoints.py"),
    Path("alberta_framework/core/nonlinear_average_reward_actor_critic.py"),
    Path("alberta_framework/streams/closed_loop.py"),
    Path("alberta_framework/evaluation/nonlinear_discrete_control_matched_development.py"),
)
_BEHAVIOR_OWNER_DIGEST = (
    0x4D415443,
    0x4845442D,
    0x434F4E54,
    0x524F4C2D,
    0x44455645,
    0x4C4F504D,
    0x454E542D,
    0x56310000,
)
_ACTION_ROOT_TAG = 31_001
_ENVIRONMENT_ROOT_TAG = 31_002
_SARSA_INIT_TAG = 31_003
_RIVER_CONFIG = RiverSwimConfig(
    n_states=6,
    p_right_up=0.35,
    p_right_down=0.05,
    reward_left=0.005,
    reward_right=1.0,
    initial_state=0,
)
_LIMITATIONS = (
    "declared development seeds become consumed only when executed; no promotion or "
    "held-out inference",
    "one synthetic six-state RiverSwim-style A/B/A life has limited external validity",
    "common random keys do not keep realized trajectories paired after causal divergence",
    "nonlinear successors use post-update policy; SARSA successors use pre-update Q values",
    "nonlinear state and differential SARSA have unequal parameterization and scalar work",
    "logical mutation opportunities are not FLOPs, wall clock, energy, or allocator peaks",
    "recovery values are descriptive event latencies and deltas, never acceptance gates",
    "source hashes and SHA-256 provide integrity binding, not authenticity",
)

ArmName = Literal["nonlinear_actor_critic", "differential_sarsa"]


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _strict_json_equal(actual: object, expected: object) -> bool:
    if expected is None:
        return actual is None
    if type(expected) in {bool, int, float, str}:
        return type(actual) is type(expected) and actual == expected
    if type(expected) is list:
        return (
            type(actual) is list
            and len(cast(list[object], actual)) == len(cast(list[object], expected))
            and all(
                _strict_json_equal(left, right)
                for left, right in zip(
                    cast(list[object], actual),
                    cast(list[object], expected),
                    strict=True,
                )
            )
        )
    if isinstance(expected, Mapping):
        return (
            isinstance(actual, Mapping)
            and set(actual) == set(expected)
            and all(_strict_json_equal(actual[key], expected[key]) for key in expected)
        )
    return False


def _exact_int(
    value: object,
    *,
    name: str,
    minimum: int = 0,
    maximum: int = _INT32_MAX,
) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an exact integer in {minimum}..{maximum}")
    return value


def _finite_float(
    value: object,
    *,
    name: str,
    minimum: float = 0.0,
    maximum: float | None = None,
    strict_minimum: bool = False,
) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{name} must be an exact finite float")
    if value <= minimum if strict_minimum else value < minimum:
        raise ValueError(f"{name} is below its lower bound")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} exceeds its upper bound")
    narrowed = float(np.float32(value))
    if not math.isfinite(narrowed) or (value != 0.0 and narrowed == 0.0):
        raise ValueError(f"{name} must remain finite and nonzero in float32")
    return value


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def matched_discrete_control_source_manifest(
    root: Path = _REPO_ROOT,
) -> dict[str, str]:
    """Hash the complete declared source closure of this development lane."""

    return {path.as_posix(): _file_sha256(root / path) for path in _SOURCE_PATHS}


@dataclasses.dataclass(frozen=True)
class MatchedDiscreteControlDevelopmentConfig:
    """Frozen deterministic A/B/A development protocol without thresholds."""

    seeds: tuple[int, ...] = (31_101, 31_102)
    phase_length: int = 64
    summary_window: int = 16
    epsilon: float = 0.1
    nonlinear_hidden_size: int = 8
    nonlinear_actor_step_size: float = 0.01
    nonlinear_critic_step_size: float = 0.04
    nonlinear_average_reward_step_size: float = 0.01
    nonlinear_actor_trace_decay: float = 0.0
    nonlinear_critic_trace_decay: float = 0.5
    nonlinear_initialization_scale: float = 0.05
    sarsa_q_step_size: float = 0.04
    sarsa_average_reward_step_size: float = 0.01
    sarsa_trace_decay: float = 0.5
    max_report_bytes: int = 32 * 1024 * 1024

    def __post_init__(self) -> None:
        if (
            type(self.seeds) is not tuple
            or not self.seeds
            or len(set(self.seeds)) != len(self.seeds)
        ):
            raise ValueError("seeds must be a nonempty tuple of unique uint32 integers")
        for seed in self.seeds:
            _exact_int(seed, name="seed", maximum=_UINT32_MAX)
        _exact_int(self.phase_length, name="phase_length", minimum=2)
        if self.phase_length % 2 != 0:
            raise ValueError("phase_length must be even for the frozen checkpoint split")
        _exact_int(self.summary_window, name="summary_window", minimum=1)
        if self.summary_window > self.phase_length:
            raise ValueError("summary_window must not exceed phase_length")
        _exact_int(self.nonlinear_hidden_size, name="nonlinear_hidden_size", minimum=1)
        _exact_int(self.max_report_bytes, name="max_report_bytes", minimum=1)
        _finite_float(self.epsilon, name="epsilon", maximum=1.0)
        for name in (
            "nonlinear_actor_step_size",
            "nonlinear_critic_step_size",
            "nonlinear_average_reward_step_size",
            "sarsa_q_step_size",
            "sarsa_average_reward_step_size",
        ):
            _finite_float(getattr(self, name), name=name)
        for name in (
            "nonlinear_actor_trace_decay",
            "nonlinear_critic_trace_decay",
            "sarsa_trace_decay",
        ):
            _finite_float(getattr(self, name), name=name, maximum=1.0)
        _finite_float(
            self.nonlinear_initialization_scale,
            name="nonlinear_initialization_scale",
            strict_minimum=True,
        )
        if self.total_steps > _INT32_MAX:
            raise ValueError("fixed total steps exceed the int32 environment clock")

    @property
    def total_steps(self) -> int:
        return len(REGIME_SCHEDULE) * self.phase_length

    @property
    def checkpoint_step(self) -> int:
        return self.phase_length + self.phase_length // 2

    def nonlinear_config(self) -> NonlinearAverageRewardActorCriticConfig:
        return NonlinearAverageRewardActorCriticConfig(
            n_actions=2,
            behavior_owner_digest=_BEHAVIOR_OWNER_DIGEST,
            hidden_size=self.nonlinear_hidden_size,
            objective_mode="ordinary_behavior",
            ordinary_behavior_epsilon=self.epsilon,
            actor_head_step_size=self.nonlinear_actor_step_size,
            actor_trunk_step_size=self.nonlinear_actor_step_size,
            critic_head_step_size=self.nonlinear_critic_step_size,
            critic_trunk_step_size=self.nonlinear_critic_step_size,
            average_reward_step_size=self.nonlinear_average_reward_step_size,
            actor_trace_decay=self.nonlinear_actor_trace_decay,
            critic_trace_decay=self.nonlinear_critic_trace_decay,
            initialization_scale=self.nonlinear_initialization_scale,
            momentum=0.0,
        )

    def sarsa_config(self) -> DifferentialSARSAConfig:
        return DifferentialSARSAConfig(
            n_actions=2,
            q_step_size=self.sarsa_q_step_size,
            average_reward_step_size=self.sarsa_average_reward_step_size,
            trace_decay=self.sarsa_trace_decay,
            epsilon_start=self.epsilon,
            epsilon_end=self.epsilon,
            epsilon_decay_steps=0,
            use_bias=True,
        )

    def to_config(self) -> dict[str, object]:
        return {
            "schema": MATCHED_DISCRETE_CONTROL_CONFIG_SCHEMA,
            "type": type(self).__name__,
            "development_only": DEVELOPMENT_ONLY,
            "assessment_status": ASSESSMENT_STATUS,
            "scientific_promotion_allowed": SCIENTIFIC_PROMOTION_ALLOWED,
            "winner_selection_allowed": WINNER_SELECTION_ALLOWED,
            "performance_thresholds_applied": PERFORMANCE_THRESHOLDS_APPLIED,
            "output_writes": OUTPUT_WRITES,
            "environment": "uninterrupted-six-state-riverswim",
            "regime_schedule": list(REGIME_SCHEDULE),
            "arm_order": list(ARM_ORDER),
            "seeds": list(self.seeds),
            "phase_length": self.phase_length,
            "summary_window": self.summary_window,
            "total_steps": self.total_steps,
            "checkpoint_step": self.checkpoint_step,
            "epsilon": self.epsilon,
            "nonlinear_hidden_size": self.nonlinear_hidden_size,
            "nonlinear_actor_step_size": self.nonlinear_actor_step_size,
            "nonlinear_critic_step_size": self.nonlinear_critic_step_size,
            "nonlinear_average_reward_step_size": (self.nonlinear_average_reward_step_size),
            "nonlinear_actor_trace_decay": self.nonlinear_actor_trace_decay,
            "nonlinear_critic_trace_decay": self.nonlinear_critic_trace_decay,
            "nonlinear_initialization_scale": self.nonlinear_initialization_scale,
            "sarsa_q_step_size": self.sarsa_q_step_size,
            "sarsa_average_reward_step_size": self.sarsa_average_reward_step_size,
            "sarsa_trace_decay": self.sarsa_trace_decay,
            "max_report_bytes": self.max_report_bytes,
        }

    @classmethod
    def from_config(
        cls,
        payload: Mapping[str, object],
    ) -> MatchedDiscreteControlDevelopmentConfig:
        constructor_names = {field.name for field in dataclasses.fields(cls)}
        fields: dict[str, object] = {}
        for name in constructor_names:
            if name not in payload:
                raise ValueError("matched-control config fields differ")
            fields[name] = payload[name]
        seeds = fields["seeds"]
        if type(seeds) is not list:
            raise ValueError("serialized seeds must be a canonical list")
        fields["seeds"] = tuple(cast(list[object], seeds))
        try:
            candidate = cls(**fields)
        except (TypeError, ValueError) as error:
            raise ValueError("matched-control config is invalid") from error
        if not _strict_json_equal(dict(payload), candidate.to_config()):
            raise ValueError("matched-control nonpromoting protocol or fields changed")
        return candidate


@chex.dataclass(frozen=True)
class MatchedDiscreteControlRunState:
    """Checkpointable state for both matched arms at one decision boundary."""

    seed: UInt[Array, ""]
    step: Int[Array, ""]
    nonlinear_state: NonlinearAverageRewardActorCriticState
    nonlinear_record: NonlinearAverageRewardActionRecord
    nonlinear_environment_state: RiverSwimState
    sarsa_state: DifferentialSARSAState
    sarsa_environment_state: RiverSwimState
    sarsa_target_policy: Float[Array, " action"]
    sarsa_behavior_policy: Float[Array, " action"]
    current_action_rng_words: UInt[Array, " 2"]


@dataclasses.dataclass(frozen=True)
class MatchedDiscreteControlTrace:
    """Host-owned complete trace fragments returned by one bounded advance."""

    nonlinear: tuple[dict[str, object], ...]
    sarsa: tuple[dict[str, object], ...]


@dataclasses.dataclass(frozen=True)
class MatchedDiscreteControlValidation:
    valid: bool
    errors: tuple[str, ...]


def _river_environment() -> RiverSwimMDP:
    return RiverSwimMDP(_RIVER_CONFIG)


def _root_key(seed: int, tag: int) -> Array:
    return jr.fold_in(jr.key(seed), tag)


def _key_words(key: Array) -> list[int]:
    return [int(word) for word in np.asarray(jax.device_get(jr.key_data(key)))]


def _scalar(value: object) -> float:
    return float(np.asarray(jax.device_get(value)))


def _integer(value: object) -> int:
    return int(np.asarray(jax.device_get(value)))


def _boolean(value: object) -> bool:
    return bool(np.asarray(jax.device_get(value)))


def _vector(value: object) -> list[float]:
    return [float(item) for item in np.asarray(jax.device_get(value), dtype=np.float32)]


def _one_hot_observations() -> Array:
    return jnp.eye(_RIVER_CONFIG.n_states, dtype=jnp.float32)


def _sarsa_policies(
    agent: DifferentialSARSAAgent,
    state: DifferentialSARSAState,
    observation: Array,
) -> tuple[Array, Array]:
    q_values = agent.q_values(state, observation)
    greedy_action = jnp.argmax(q_values).astype(jnp.int32)
    target = jax.nn.one_hot(greedy_action, 2, dtype=jnp.float32)
    epsilon = state.epsilon
    behavior = (jnp.float32(1.0) - epsilon) * target + epsilon * jnp.float32(0.5)
    return target, behavior


def _nonlinear_policy_table(
    agent: NonlinearAverageRewardActorCritic,
    state: NonlinearAverageRewardActorCriticState,
) -> Array:
    return jnp.stack(
        tuple(agent.target_policy(state, observation) for observation in _one_hot_observations())
    )


def _sarsa_policy_table(
    agent: DifferentialSARSAAgent,
    state: DifferentialSARSAState,
) -> tuple[Array, Array]:
    policies = tuple(_sarsa_policies(agent, state, obs) for obs in _one_hot_observations())
    return jnp.stack(tuple(item[0] for item in policies)), jnp.stack(
        tuple(item[1] for item in policies)
    )


def _ordinary_behavior(target: Array, epsilon: float) -> Array:
    return (jnp.float32(1.0) - jnp.float32(epsilon)) * target + jnp.float32(epsilon / 2.0)


def _array_group_l2_delta(
    before: Sequence[Array],
    after: Sequence[Array],
) -> float:
    total = 0.0
    for left, right in zip(before, after, strict=True):
        delta = np.asarray(jax.device_get(right - left), dtype=np.float64)
        total += float(np.sum(np.square(delta), dtype=np.float64))
    return float(math.sqrt(total))


def _nonlinear_actor_arrays(state: NonlinearAverageRewardActorCriticState) -> tuple[Array, ...]:
    return (
        state.actor_trunk_w,
        state.actor_trunk_b,
        state.actor_head_w,
        state.actor_head_b,
    )


def _nonlinear_critic_arrays(
    state: NonlinearAverageRewardActorCriticState,
) -> tuple[Array, ...]:
    return (
        state.critic_trunk_w,
        state.critic_trunk_b,
        state.critic_head_w,
        state.critic_head_b,
    )


def _sarsa_control_arrays(state: DifferentialSARSAState) -> tuple[Array, ...]:
    return (state.q_weights, state.q_bias)


def _reward(regime: str, state_index: int, action: int) -> float:
    if regime == "A":
        if state_index == 0 and action == LEFT_ACTION:
            return 0.005
        if state_index == _RIVER_CONFIG.n_states - 1 and action == RIGHT_ACTION:
            return 1.0
        return 0.0
    if state_index == 0 and action == LEFT_ACTION:
        return 1.0
    if state_index == _RIVER_CONFIG.n_states - 1 and action == RIGHT_ACTION:
        return 0.005
    return 0.0


def initialize_matched_discrete_control_run(
    config: MatchedDiscreteControlDevelopmentConfig,
    *,
    seed: int,
) -> MatchedDiscreteControlRunState:
    """Initialize and prequentially bind both arms' first action."""

    if type(config) is not MatchedDiscreteControlDevelopmentConfig:
        raise TypeError("config must be an exact matched-control config")
    _exact_int(seed, name="seed", maximum=_UINT32_MAX)
    if seed not in config.seeds:
        raise ValueError("seed is not declared by this development config")
    environment = _river_environment()
    observation = environment.observe(environment.init(jr.key(0)))

    nonlinear_agent = NonlinearAverageRewardActorCritic(config.nonlinear_config())
    nonlinear_initial = nonlinear_agent.init(
        environment.feature_dim,
        _root_key(seed, _ACTION_ROOT_TAG),
    )
    _, first_sample_key = jr.split(nonlinear_initial.rng_key)
    first_receipt = nonlinear_agent.ordinary_behavior_receipt(
        nonlinear_initial,
        observation,
        jnp.zeros((2,), dtype=jnp.uint32),
    )
    nonlinear_started = nonlinear_agent.start(
        nonlinear_initial,
        observation,
        first_receipt,
    )
    if not _boolean(nonlinear_started.start_applied):
        raise RuntimeError("nonlinear matched-control start failed closed")

    sarsa_agent = DifferentialSARSAAgent(config.sarsa_config())
    sarsa_initial = sarsa_agent.init(
        environment.feature_dim,
        _root_key(seed, _SARSA_INIT_TAG),
    ).replace(birth_timestamp=0.0, uptime_s=0.0)
    sarsa_target, sarsa_behavior = _sarsa_policies(
        sarsa_agent,
        sarsa_initial,
        observation,
    )
    sarsa_action = jr.categorical(first_sample_key, jnp.log(sarsa_behavior)).astype(jnp.int32)
    sarsa_started, _ = sarsa_agent.start_with_action(
        sarsa_initial,
        observation,
        sarsa_action,
    )
    environment_state = environment.init(jr.key(0))
    return MatchedDiscreteControlRunState(
        seed=jnp.asarray(seed, dtype=jnp.uint32),
        step=jnp.asarray(0, dtype=jnp.int32),
        nonlinear_state=nonlinear_started.state,
        nonlinear_record=nonlinear_started.record,
        nonlinear_environment_state=environment_state,
        sarsa_state=sarsa_started,
        sarsa_environment_state=environment_state,
        sarsa_target_policy=sarsa_target,
        sarsa_behavior_policy=sarsa_behavior,
        current_action_rng_words=jr.key_data(first_sample_key),
    )


def _trace_row(
    *,
    step: int,
    phase: int,
    regime: str,
    state_index: int,
    next_state_index: int,
    action: int,
    target_policy: Array,
    behavior_policy: Array,
    reward: float,
    value: float,
    next_value: float,
    td_error: float,
    average_reward_before: float,
    average_reward_after: float,
    target_policy_churn: float,
    behavior_policy_churn: float,
    actor_delta: float | None,
    critic_or_action_value_delta: float,
    action_rng_words: Array,
    environment_key: Array,
    update_count_after: int,
    update_applied: bool,
) -> dict[str, object]:
    behavior_probability = float(np.asarray(jax.device_get(behavior_policy[action])))
    return {
        "step": step,
        "phase_index": phase,
        "regime": regime,
        "state_index": state_index,
        "next_state_index": next_state_index,
        "action": action,
        "target_policy": _vector(target_policy),
        "behavior_policy": _vector(behavior_policy),
        "behavior_probability": behavior_probability,
        "behavior_log_probability": float(np.float32(math.log(behavior_probability))),
        "reward": float(np.float32(reward)),
        "prequential_value": float(np.float32(value)),
        "prequential_next_value": float(np.float32(next_value)),
        "td_error": float(np.float32(td_error)),
        "average_reward_before": float(np.float32(average_reward_before)),
        "average_reward_after": float(np.float32(average_reward_after)),
        "target_policy_churn_l1": float(np.float32(target_policy_churn)),
        "behavior_policy_churn_l1": float(np.float32(behavior_policy_churn)),
        "actor_parameter_delta_l2": (
            None if actor_delta is None else float(np.float32(actor_delta))
        ),
        "critic_or_action_value_parameter_delta_l2": float(
            np.float32(critic_or_action_value_delta)
        ),
        "policy_bound_before_outcome": True,
        "update_applied": update_applied,
        "update_count_after": update_count_after,
        "action_rng_words": [int(word) for word in np.asarray(action_rng_words)],
        "environment_rng_words": _key_words(environment_key),
    }


def _advance_one(
    config: MatchedDiscreteControlDevelopmentConfig,
    state: MatchedDiscreteControlRunState,
) -> tuple[MatchedDiscreteControlRunState, dict[str, object], dict[str, object]]:
    step = _integer(state.step)
    if step >= config.total_steps:
        raise ValueError("matched-control life is already complete")
    seed = _integer(state.seed)
    phase = step // config.phase_length
    regime = REGIME_SCHEDULE[phase]
    environment = _river_environment()
    environment_key = jr.fold_in(_root_key(seed, _ENVIRONMENT_ROOT_TAG), step)
    nonlinear_agent = NonlinearAverageRewardActorCritic(config.nonlinear_config())
    sarsa_agent = DifferentialSARSAAgent(config.sarsa_config())

    nonlinear_state = state.nonlinear_state
    nonlinear_record = state.nonlinear_record
    nonlinear_action = _integer(nonlinear_record.action)
    nonlinear_state_index = _integer(state.nonlinear_environment_state.state_index)
    nonlinear_actor_before = _nonlinear_actor_arrays(nonlinear_state)
    nonlinear_critic_before = _nonlinear_critic_arrays(nonlinear_state)
    nonlinear_target_table_before = _nonlinear_policy_table(nonlinear_agent, nonlinear_state)
    nonlinear_behavior_table_before = _ordinary_behavior(
        nonlinear_target_table_before,
        config.epsilon,
    )
    nonlinear_next_observation, _, nonlinear_environment_after = environment.step(
        state.nonlinear_environment_state,
        nonlinear_record.action,
        environment_key,
    )
    nonlinear_reward = _reward(regime, nonlinear_state_index, nonlinear_action)
    nonlinear_proposal = nonlinear_agent.propose_update(
        nonlinear_state,
        nonlinear_record,
        jnp.asarray(nonlinear_reward, dtype=jnp.float32),
        nonlinear_next_observation,
    )
    nonlinear_receipt = nonlinear_agent.ordinary_successor_behavior_receipt(nonlinear_proposal)
    _, successor_sample_key = jr.split(nonlinear_state.rng_key)
    nonlinear_result = nonlinear_agent.commit_update(
        nonlinear_state,
        nonlinear_record,
        jnp.asarray(nonlinear_reward, dtype=jnp.float32),
        nonlinear_next_observation,
        nonlinear_proposal,
        nonlinear_receipt,
    )
    if not _boolean(nonlinear_result.update_applied):
        raise RuntimeError("nonlinear matched-control transaction failed closed")
    nonlinear_target_table_after = _nonlinear_policy_table(
        nonlinear_agent,
        nonlinear_result.state,
    )
    nonlinear_behavior_table_after = _ordinary_behavior(
        nonlinear_target_table_after,
        config.epsilon,
    )
    nonlinear_target_churn = _scalar(
        jnp.mean(
            jnp.sum(
                jnp.abs(nonlinear_target_table_after - nonlinear_target_table_before),
                axis=1,
            )
        )
    )
    nonlinear_behavior_churn = _scalar(
        jnp.mean(
            jnp.sum(
                jnp.abs(nonlinear_behavior_table_after - nonlinear_behavior_table_before),
                axis=1,
            )
        )
    )
    nonlinear_actor_delta = _array_group_l2_delta(
        nonlinear_actor_before,
        _nonlinear_actor_arrays(nonlinear_result.state),
    )
    nonlinear_critic_delta = _array_group_l2_delta(
        nonlinear_critic_before,
        _nonlinear_critic_arrays(nonlinear_result.state),
    )

    sarsa_state = state.sarsa_state
    sarsa_action = _integer(sarsa_state.last_action)
    sarsa_state_index = _integer(state.sarsa_environment_state.state_index)
    sarsa_control_before = _sarsa_control_arrays(sarsa_state)
    sarsa_target_table_before, sarsa_behavior_table_before = _sarsa_policy_table(
        sarsa_agent,
        sarsa_state,
    )
    sarsa_observation = environment.observe(state.sarsa_environment_state)
    sarsa_q_current = sarsa_agent.q_values(sarsa_state, sarsa_observation)[sarsa_action]
    sarsa_next_observation, _, sarsa_environment_after = environment.step(
        state.sarsa_environment_state,
        sarsa_state.last_action,
        environment_key,
    )
    sarsa_reward = _reward(regime, sarsa_state_index, sarsa_action)
    next_sarsa_target, next_sarsa_behavior = _sarsa_policies(
        sarsa_agent,
        sarsa_state,
        sarsa_next_observation,
    )
    next_sarsa_action = jr.categorical(
        successor_sample_key,
        jnp.log(next_sarsa_behavior),
    ).astype(jnp.int32)
    sarsa_q_next = sarsa_agent.q_values(sarsa_state, sarsa_next_observation)[next_sarsa_action]
    sarsa_result = sarsa_agent.update(
        sarsa_state,
        jnp.asarray(sarsa_reward, dtype=jnp.float32),
        sarsa_next_observation,
        next_action=next_sarsa_action,
    )
    if not _boolean(sarsa_result.update_applied):
        raise RuntimeError("differential SARSA matched-control update failed closed")
    sarsa_target_table_after, sarsa_behavior_table_after = _sarsa_policy_table(
        sarsa_agent,
        sarsa_result.state,
    )
    sarsa_target_churn = _scalar(
        jnp.mean(jnp.sum(jnp.abs(sarsa_target_table_after - sarsa_target_table_before), axis=1))
    )
    sarsa_behavior_churn = _scalar(
        jnp.mean(
            jnp.sum(
                jnp.abs(sarsa_behavior_table_after - sarsa_behavior_table_before),
                axis=1,
            )
        )
    )
    sarsa_control_delta = _array_group_l2_delta(
        sarsa_control_before,
        _sarsa_control_arrays(sarsa_result.state),
    )

    nonlinear_row = _trace_row(
        step=step,
        phase=phase,
        regime=regime,
        state_index=nonlinear_state_index,
        next_state_index=_integer(nonlinear_environment_after.state_index),
        action=nonlinear_action,
        target_policy=nonlinear_record.target_policy,
        behavior_policy=nonlinear_record.behavior_policy,
        reward=nonlinear_reward,
        value=_scalar(nonlinear_result.value),
        next_value=_scalar(nonlinear_result.next_value),
        td_error=_scalar(nonlinear_result.td_error),
        average_reward_before=_scalar(nonlinear_result.pre_average_reward),
        average_reward_after=_scalar(nonlinear_result.post_average_reward),
        target_policy_churn=nonlinear_target_churn,
        behavior_policy_churn=nonlinear_behavior_churn,
        actor_delta=nonlinear_actor_delta,
        critic_or_action_value_delta=nonlinear_critic_delta,
        action_rng_words=state.current_action_rng_words,
        environment_key=environment_key,
        update_count_after=_integer(nonlinear_result.state.update_words[1]),
        update_applied=True,
    )
    sarsa_row = _trace_row(
        step=step,
        phase=phase,
        regime=regime,
        state_index=sarsa_state_index,
        next_state_index=_integer(sarsa_environment_after.state_index),
        action=sarsa_action,
        target_policy=state.sarsa_target_policy,
        behavior_policy=state.sarsa_behavior_policy,
        reward=sarsa_reward,
        value=_scalar(sarsa_q_current),
        next_value=_scalar(sarsa_q_next),
        td_error=_scalar(sarsa_result.td_error),
        average_reward_before=_scalar(sarsa_state.average_reward),
        average_reward_after=_scalar(sarsa_result.average_reward),
        target_policy_churn=sarsa_target_churn,
        behavior_policy_churn=sarsa_behavior_churn,
        actor_delta=None,
        critic_or_action_value_delta=sarsa_control_delta,
        action_rng_words=state.current_action_rng_words,
        environment_key=environment_key,
        update_count_after=_integer(sarsa_result.state.step_words[1]),
        update_applied=True,
    )
    next_state = MatchedDiscreteControlRunState(
        seed=state.seed,
        step=state.step + jnp.asarray(1, dtype=jnp.int32),
        nonlinear_state=nonlinear_result.state,
        nonlinear_record=nonlinear_result.successor_record,
        nonlinear_environment_state=nonlinear_environment_after,
        sarsa_state=sarsa_result.state,
        sarsa_environment_state=sarsa_environment_after,
        sarsa_target_policy=next_sarsa_target,
        sarsa_behavior_policy=next_sarsa_behavior,
        current_action_rng_words=jr.key_data(successor_sample_key),
    )
    return next_state, nonlinear_row, sarsa_row


def advance_matched_discrete_control_run(
    config: MatchedDiscreteControlDevelopmentConfig,
    state: MatchedDiscreteControlRunState,
    *,
    stop_step: int,
) -> tuple[MatchedDiscreteControlRunState, MatchedDiscreteControlTrace]:
    """Advance both arms to one exact exclusive stop step."""

    _exact_int(stop_step, name="stop_step", maximum=config.total_steps)
    current_step = _integer(state.step)
    if stop_step < current_step:
        raise ValueError("stop_step precedes the checkpoint step")
    nonlinear: list[dict[str, object]] = []
    sarsa: list[dict[str, object]] = []
    current = state
    while _integer(current.step) < stop_step:
        current, nonlinear_row, sarsa_row = _advance_one(config, current)
        nonlinear.append(nonlinear_row)
        sarsa.append(sarsa_row)
    return current, MatchedDiscreteControlTrace(tuple(nonlinear), tuple(sarsa))


def _pytree_sha256(value: object) -> str:
    leaves, structure = jax.tree.flatten(value)
    digest = hashlib.sha256(str(structure).encode("utf-8"))
    for leaf in leaves:
        if str(getattr(leaf, "dtype", "")).startswith("key<"):
            array = np.asarray(jax.device_get(jr.key_data(leaf)))
        elif hasattr(leaf, "dtype") and hasattr(leaf, "shape"):
            array = np.asarray(jax.device_get(leaf))
        elif type(leaf) in {bool, int, float, str}:
            digest.update(type(leaf).__name__.encode("ascii"))
            digest.update(repr(leaf).encode("utf-8"))
            continue
        else:
            raise TypeError("matched-control state contains a noncanonical leaf")
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(_canonical_json_bytes(list(contiguous.shape)))
        digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def matched_discrete_control_run_state_sha256(
    state: MatchedDiscreteControlRunState,
) -> str:
    """Hash every ordered state leaf, including dtype, shape, and key words."""

    return _pytree_sha256(state)


def _checkpoint_metadata(
    state: MatchedDiscreteControlRunState,
    config: MatchedDiscreteControlDevelopmentConfig,
) -> dict[str, object]:
    return {
        "schema": MATCHED_DISCRETE_CONTROL_CHECKPOINT_SCHEMA,
        "config": config.to_config(),
        "seed": _integer(state.seed),
        "step": _integer(state.step),
        "state_sha256": matched_discrete_control_run_state_sha256(state),
        "source_manifest": matched_discrete_control_source_manifest(),
    }


def save_matched_discrete_control_checkpoint(
    state: MatchedDiscreteControlRunState,
    path: str | Path,
    *,
    config: MatchedDiscreteControlDevelopmentConfig,
) -> None:
    """Persist one exact source/config/seed-bound matched run state."""

    step = _integer(state.step)
    if not 0 <= step <= config.total_steps:
        raise ValueError("checkpoint step lies outside the configured life")
    if _integer(state.seed) not in config.seeds:
        raise ValueError("checkpoint seed is not declared by the configuration")
    save_checkpoint(state, path, metadata=_checkpoint_metadata(state, config))


def load_matched_discrete_control_checkpoint(
    template: MatchedDiscreteControlRunState,
    path: str | Path,
    *,
    config: MatchedDiscreteControlDevelopmentConfig,
) -> MatchedDiscreteControlRunState:
    """Restore only an exact current-source and configuration-compatible state."""

    metadata = load_checkpoint_metadata(path)
    expected_keys = {"schema", "config", "seed", "step", "state_sha256", "source_manifest"}
    if set(metadata) != expected_keys:
        raise ValueError("matched-control checkpoint metadata fields differ")
    if metadata["schema"] != MATCHED_DISCRETE_CONTROL_CHECKPOINT_SCHEMA:
        raise ValueError("matched-control checkpoint schema is unsupported")
    if not _strict_json_equal(metadata["config"], config.to_config()):
        raise ValueError("matched-control checkpoint configuration is incompatible")
    if not _strict_json_equal(
        metadata["source_manifest"],
        matched_discrete_control_source_manifest(),
    ):
        raise ValueError("matched-control checkpoint source manifest is stale")
    loaded_raw, restored_metadata = load_checkpoint(template, path)
    if not _strict_json_equal(restored_metadata, metadata):
        raise ValueError("matched-control checkpoint metadata changed during restore")
    loaded = cast(MatchedDiscreteControlRunState, loaded_raw)
    if _integer(loaded.seed) != metadata["seed"] or _integer(loaded.step) != metadata["step"]:
        raise ValueError("matched-control checkpoint clock or seed is inconsistent")
    if matched_discrete_control_run_state_sha256(loaded) != metadata["state_sha256"]:
        raise ValueError("matched-control checkpoint state digest is inconsistent")
    return loaded


def _trace_logical_nbytes(rows: Sequence[Mapping[str, object]]) -> int:
    """Bytes in the declared packed numeric trace representation."""

    if not rows:
        return 0
    scalar_int = (
        "step",
        "phase_index",
        "state_index",
        "next_state_index",
        "action",
        "update_count_after",
    )
    scalar_float = (
        "behavior_probability",
        "behavior_log_probability",
        "reward",
        "prequential_value",
        "prequential_next_value",
        "td_error",
        "average_reward_before",
        "average_reward_after",
        "target_policy_churn_l1",
        "behavior_policy_churn_l1",
        "critic_or_action_value_parameter_delta_l2",
    )
    total = 0
    for name in scalar_int:
        total += np.asarray([row[name] for row in rows], dtype=np.int32).nbytes
    for name in scalar_float:
        total += np.asarray([row[name] for row in rows], dtype=np.float32).nbytes
    if rows[0]["actor_parameter_delta_l2"] is not None:
        total += np.asarray(
            [row["actor_parameter_delta_l2"] for row in rows], dtype=np.float32
        ).nbytes
    for name in ("target_policy", "behavior_policy"):
        total += np.asarray([row[name] for row in rows], dtype=np.float32).nbytes
    for name in ("policy_bound_before_outcome", "update_applied"):
        total += np.asarray([row[name] for row in rows], dtype=np.bool_).nbytes
    for name in ("action_rng_words", "environment_rng_words"):
        total += np.asarray([row[name] for row in rows], dtype=np.uint32).nbytes
    return int(total)


def _first_offset(mask: Sequence[bool]) -> int | None:
    for index, value in enumerate(mask):
        if value:
            return index
    return None


def _diagnostics(
    rows: Sequence[Mapping[str, object]],
    *,
    arm: ArmName,
    config: MatchedDiscreteControlDevelopmentConfig,
) -> dict[str, object]:
    rewards = np.asarray([row["reward"] for row in rows], dtype=np.float64)
    actor_delta = np.asarray(
        [
            0.0 if row["actor_parameter_delta_l2"] is None else row["actor_parameter_delta_l2"]
            for row in rows
        ],
        dtype=np.float64,
    )
    critic_delta = np.asarray(
        [row["critic_or_action_value_parameter_delta_l2"] for row in rows],
        dtype=np.float64,
    )
    td_error = np.asarray([row["td_error"] for row in rows], dtype=np.float64)
    target_churn = np.asarray([row["target_policy_churn_l1"] for row in rows], dtype=np.float64)
    behavior_churn = np.asarray([row["behavior_policy_churn_l1"] for row in rows], dtype=np.float64)
    reward_rate_before = np.asarray(
        [row["average_reward_before"] for row in rows], dtype=np.float64
    )
    reward_rate_after = np.asarray([row["average_reward_after"] for row in rows], dtype=np.float64)
    phase_records: list[dict[str, object]] = []
    for phase, regime in enumerate(REGIME_SCHEDULE):
        start = phase * config.phase_length
        end = start + config.phase_length
        phase_rows = rows[start:end]
        phase_rewards = rewards[start:end]
        valuable_state = _RIVER_CONFIG.n_states - 1 if regime == "A" else 0
        valuable_action = RIGHT_ACTION if regime == "A" else LEFT_ACTION
        state_mask = [cast(int, row["state_index"]) == valuable_state for row in phase_rows]
        reward_mask = [
            cast(int, row["state_index"]) == valuable_state
            and cast(int, row["action"]) == valuable_action
            for row in phase_rows
        ]
        phase_records.append(
            {
                "phase_index": phase,
                "regime": regime,
                "mean_reward": float(np.mean(phase_rewards)),
                "early_window_mean_reward": float(np.mean(phase_rewards[: config.summary_window])),
                "tail_window_mean_reward": float(np.mean(phase_rewards[-config.summary_window :])),
                "first_valuable_boundary_visit_offset": _first_offset(state_mask),
                "first_valuable_boundary_action_offset": _first_offset(reward_mask),
                "maximum_state_index": max(cast(int, row["state_index"]) for row in phase_rows),
            }
        )
    return {
        "actor": (
            {
                "status": "available",
                "component": "separate-softmax-actor",
                "mean_parameter_delta_l2": float(np.mean(actor_delta)),
                "max_parameter_delta_l2": float(np.max(actor_delta)),
            }
            if arm == "nonlinear_actor_critic"
            else {
                "status": "inapplicable",
                "component": None,
                "reason": "differential SARSA has no separate actor parameters",
            }
        ),
        "critic_or_action_value": {
            "status": "available",
            "component": (
                "nonlinear-differential-state-value-critic"
                if arm == "nonlinear_actor_critic"
                else "linear-differential-action-value"
            ),
            "mean_absolute_td_error": float(np.mean(np.abs(td_error))),
            "mean_parameter_delta_l2": float(np.mean(critic_delta)),
            "max_parameter_delta_l2": float(np.max(critic_delta)),
        },
        "reward_rate": {
            "status": "available",
            "initial_pre_update": float(reward_rate_before[0]),
            "final_post_update": float(reward_rate_after[-1]),
            "mean_absolute_update": float(np.mean(np.abs(reward_rate_after - reward_rate_before))),
        },
        "policy_churn": {
            "status": "available",
            "probe_states": list(range(_RIVER_CONFIG.n_states)),
            "mean_target_policy_l1": float(np.mean(target_churn)),
            "max_target_policy_l1": float(np.max(target_churn)),
            "mean_behavior_policy_l1": float(np.mean(behavior_churn)),
            "max_behavior_policy_l1": float(np.max(behavior_churn)),
        },
        "recovery": {
            "status": "descriptive_not_gated",
            "phase_records": phase_records,
            "recurring_A_minus_initial_A_early_reward": float(
                cast(float, phase_records[2]["early_window_mean_reward"])
                - cast(float, phase_records[0]["early_window_mean_reward"])
            ),
            "recurring_A_minus_initial_A_tail_reward": float(
                cast(float, phase_records[2]["tail_window_mean_reward"])
                - cast(float, phase_records[0]["tail_window_mean_reward"])
            ),
            "acceptance_threshold": None,
            "pass": None,
        },
        "return": {
            "status": "descriptive_not_gated",
            "lifetime_return": float(np.sum(rewards)),
            "mean_prequential_reward": float(np.mean(rewards)),
            "phase_mean_rewards": [cast(float, record["mean_reward"]) for record in phase_records],
        },
    }


def _nonlinear_scalar_work(
    state: NonlinearAverageRewardActorCriticState,
) -> dict[str, int]:
    actor = sum(int(value.size) for value in _nonlinear_actor_arrays(state))
    critic = sum(int(value.size) for value in _nonlinear_critic_arrays(state))
    return {
        "actor_parameter_mutation_opportunities_per_transition": actor,
        "critic_parameter_mutation_opportunities_per_transition": critic,
        "eligibility_trace_mutation_opportunities_per_transition": actor + critic,
        "momentum_buffer_mutation_opportunities_per_transition": actor + critic,
        "utility_scalar_mutation_opportunities_per_transition": 4,
        "reward_rate_scalar_mutation_opportunities_per_transition": 1,
        "total_scalar_mutation_opportunities_per_transition": 3 * (actor + critic) + 5,
    }


def _sarsa_scalar_work(state: DifferentialSARSAState) -> dict[str, int]:
    control = sum(int(value.size) for value in _sarsa_control_arrays(state))
    return {
        "actor_parameter_mutation_opportunities_per_transition": 0,
        "action_value_parameter_mutation_opportunities_per_transition": control,
        "eligibility_trace_mutation_opportunities_per_transition": control,
        "epsilon_scalar_mutation_opportunities_per_transition": 1,
        "reward_rate_scalar_mutation_opportunities_per_transition": 1,
        "total_scalar_mutation_opportunities_per_transition": 2 * control + 2,
    }


def _policy_semantics(arm: ArmName) -> dict[str, object]:
    if arm == "nonlinear_actor_critic":
        return {
            "target": "current-temperature-one-softmax-actor",
            "behavior": "epsilon-uniform-mixture-of-current-softmax-target",
            "objective": "ordinary-behavior-policy-gradient-chain-rule-score",
            "critic": "on-behavior-differential-state-value",
            "reward_rate": "on-behavior-unweighted-differential-TD-error",
            "successor_policy_timing": "post-update",
            "action_binding": "core-owned-exact-target-behavior-receipt",
        }
    return {
        "target": "lowest-index-greedy-action-value-target",
        "behavior": "epsilon-uniform-mixture-of-lowest-index-greedy-target",
        "objective": "on-behavior-differential-semi-gradient-SARSA",
        "critic": "not-separate-action-value-control",
        "reward_rate": "on-behavior-unweighted-differential-TD-error",
        "successor_policy_timing": "pre-update",
        "action_binding": "public-external-action-start-and-update-surface",
    }


def _arm_report(
    *,
    arm: ArmName,
    rows: tuple[dict[str, object], ...],
    final_state: MatchedDiscreteControlRunState,
    config: MatchedDiscreteControlDevelopmentConfig,
) -> dict[str, object]:
    nonlinear_agent = NonlinearAverageRewardActorCritic(config.nonlinear_config())
    if arm == "nonlinear_actor_critic":
        learner_nbytes = nonlinear_agent.resource_budget(
            final_state.nonlinear_state
        ).total_state_nbytes
        evaluator_cache_nbytes = 8
        work = _nonlinear_scalar_work(final_state.nonlinear_state)
        final_learner_hash = _pytree_sha256(final_state.nonlinear_state)
    else:
        learner_nbytes = measure_differential_sarsa_state_nbytes(final_state.sarsa_state)
        evaluator_cache_nbytes = 24
        work = _sarsa_scalar_work(final_state.sarsa_state)
        final_learner_hash = _pytree_sha256(final_state.sarsa_state)
    trace_nbytes = _trace_logical_nbytes(rows)
    commits = sum(bool(row["update_applied"]) for row in rows)
    operations = {
        "scope": "fixed logical algorithm events; not FLOPs, wall clock, energy, or peak memory",
        "environment_transitions": len(rows),
        "environment_rng_draws": len(rows),
        "categorical_action_draws": len(rows) + 1,
        "learner_update_attempts": len(rows),
        "learner_update_commits": commits,
        "learner_update_rejections": len(rows) - commits,
        "scalar_mutation_opportunities": {
            **work,
            "total_over_life": work["total_scalar_mutation_opportunities_per_transition"]
            * len(rows),
        },
    }
    return {
        "arm": arm,
        "assessment_status": ASSESSMENT_STATUS,
        "policy_semantics": _policy_semantics(arm),
        "prequential_trace": list(rows),
        "diagnostics": _diagnostics(rows, arm=arm, config=config),
        "logical_operations": operations,
        "resources": {
            "scope": (
                "persistent learner arrays + environment arrays + evaluator action binding "
                "cache + packed retained numeric trace; excludes host objects, compiler/runtime "
                "workspaces, allocator residency, and transient intermediates"
            ),
            "persistent_state_nbytes": learner_nbytes,
            "environment_state_nbytes": 8,
            "evaluator_binding_cache_nbytes": evaluator_cache_nbytes,
            "logical_trace_nbytes": trace_nbytes,
            "peak_owned_logical_nbytes": (
                learner_nbytes + 8 + evaluator_cache_nbytes + trace_nbytes
            ),
        },
        "final_learner_state_sha256": final_learner_hash,
    }


def _comparison(
    nonlinear: Mapping[str, object],
    sarsa: Mapping[str, object],
) -> dict[str, object]:
    nonlinear_return = cast(
        float,
        cast(Mapping[str, object], cast(Mapping[str, object], nonlinear["diagnostics"])["return"])[
            "lifetime_return"
        ],
    )
    sarsa_return = cast(
        float,
        cast(Mapping[str, object], cast(Mapping[str, object], sarsa["diagnostics"])["return"])[
            "lifetime_return"
        ],
    )
    nonlinear_ops = cast(Mapping[str, object], nonlinear["logical_operations"])
    sarsa_ops = cast(Mapping[str, object], sarsa["logical_operations"])
    nonlinear_work = cast(Mapping[str, object], nonlinear_ops["scalar_mutation_opportunities"])
    sarsa_work = cast(Mapping[str, object], sarsa_ops["scalar_mutation_opportunities"])
    nonlinear_resources = cast(Mapping[str, object], nonlinear["resources"])
    sarsa_resources = cast(Mapping[str, object], sarsa["resources"])
    return {
        "assessment_status": ASSESSMENT_STATUS,
        "verdict": "not_assessed",
        "winner": None,
        "performance_thresholds_applied": False,
        "descriptive_lifetime_return_difference_nonlinear_minus_sarsa": (
            nonlinear_return - sarsa_return
        ),
        "work_matching": {
            "environment_transition_opportunities_match": (
                nonlinear_ops["environment_transitions"] == sarsa_ops["environment_transitions"]
            ),
            "learner_update_opportunities_match": (
                nonlinear_ops["learner_update_attempts"] == sarsa_ops["learner_update_attempts"]
            ),
            "categorical_action_draws_match": (
                nonlinear_ops["categorical_action_draws"] == sarsa_ops["categorical_action_draws"]
            ),
            "successor_policy_timing_matches": False,
            "realized_scalar_update_work_matches": (
                nonlinear_work["total_over_life"] == sarsa_work["total_over_life"]
            ),
            "persistent_state_bytes_match": (
                nonlinear_resources["persistent_state_nbytes"]
                == sarsa_resources["persistent_state_nbytes"]
            ),
            "exact_realized_work_matched": False,
            "reported_mismatches": [
                "nonlinear successor actions use a post-update policy; SARSA uses pre-update Q",
                "the nonlinear actor/critic and linear action-value learner have unequal scalars",
                "persistent state bytes differ; neither logical trace bytes nor bytes are FLOPs",
            ],
        },
    }


def _run_seed(
    config: MatchedDiscreteControlDevelopmentConfig,
    seed: int,
) -> dict[str, object]:
    initial = initialize_matched_discrete_control_run(config, seed=seed)
    initial_sha = matched_discrete_control_run_state_sha256(initial)
    checkpoint_state, prefix = advance_matched_discrete_control_run(
        config,
        initial,
        stop_step=config.checkpoint_step,
    )
    checkpoint_sha = matched_discrete_control_run_state_sha256(checkpoint_state)
    with tempfile.TemporaryDirectory(prefix="alberta-matched-control-") as directory:
        checkpoint_path = Path(directory) / "checkpoint"
        save_matched_discrete_control_checkpoint(
            checkpoint_state,
            checkpoint_path,
            config=config,
        )
        restored = load_matched_discrete_control_checkpoint(
            checkpoint_state,
            checkpoint_path,
            config=config,
        )
    restored_sha = matched_discrete_control_run_state_sha256(restored)
    if restored_sha != checkpoint_sha:
        raise RuntimeError("matched-control checkpoint round trip changed state")
    final, suffix = advance_matched_discrete_control_run(
        config,
        restored,
        stop_step=config.total_steps,
    )
    nonlinear_rows = prefix.nonlinear + suffix.nonlinear
    sarsa_rows = prefix.sarsa + suffix.sarsa
    nonlinear_report = _arm_report(
        arm="nonlinear_actor_critic",
        rows=nonlinear_rows,
        final_state=final,
        config=config,
    )
    sarsa_report = _arm_report(
        arm="differential_sarsa",
        rows=sarsa_rows,
        final_state=final,
        config=config,
    )
    return {
        "seed": seed,
        "initial_joint_state_sha256": initial_sha,
        "checkpoint_validation": {
            "schema": MATCHED_DISCRETE_CONTROL_CHECKPOINT_SCHEMA,
            "split_step": config.checkpoint_step,
            "prefix_transition_count_per_arm": len(prefix.nonlinear),
            "suffix_transition_count_per_arm": len(suffix.nonlinear),
            "snapshot_state_sha256": checkpoint_sha,
            "restored_state_sha256": restored_sha,
            "roundtrip_exact": checkpoint_sha == restored_sha,
            "temporary_checkpoint_removed": True,
        },
        "arms": [nonlinear_report, sarsa_report],
        "comparison": _comparison(nonlinear_report, sarsa_report),
        "final_joint_state_sha256": matched_discrete_control_run_state_sha256(final),
    }


def _protocol(config: MatchedDiscreteControlDevelopmentConfig) -> dict[str, object]:
    return {
        "protocol_id": "six-state-riverswim-aba-matched-control-development-v1",
        "environment": {
            "n_states": _RIVER_CONFIG.n_states,
            "n_actions": 2,
            "p_right_up": _RIVER_CONFIG.p_right_up,
            "p_right_down": _RIVER_CONFIG.p_right_down,
            "initial_state": _RIVER_CONFIG.initial_state,
            "regime_A_rewards": {
                "left_boundary_left_action": 0.005,
                "right_boundary_right_action": 1.0,
            },
            "regime_B_rewards": {
                "left_boundary_left_action": 1.0,
                "right_boundary_right_action": 0.005,
            },
            "resets_between_phases": False,
        },
        "regime_schedule": list(REGIME_SCHEDULE),
        "phase_length": config.phase_length,
        "total_steps": config.total_steps,
        "common_randomness": {
            "action": (
                "the exact nonlinear categorical sample key is reused by SARSA's public "
                "external-action path at every decision"
            ),
            "environment": "one identical decision-indexed transition key per arm",
            "causal_scope": (
                "shared exogenous keys only; distributions, actions, states, rewards, and "
                "realized trajectories may diverge"
            ),
        },
        "prequential_order": [
            "bind-current-target-and-behavior-policy",
            "bind-current-action-from-declared-categorical-key",
            "realize-environment-transition-and-reward",
            "apply-one-differential-update",
            "bind-successor-action-for-next-transition",
        ],
        "arm_policy_semantics": {arm: _policy_semantics(cast(ArmName, arm)) for arm in ARM_ORDER},
        "checkpoint_step": config.checkpoint_step,
        "no_threshold_or_winner_rule": True,
    }


def _build_report(
    config: MatchedDiscreteControlDevelopmentConfig,
) -> dict[str, object]:
    report: dict[str, object] = {
        "schema": MATCHED_DISCRETE_CONTROL_REPORT_SCHEMA,
        "development_only": DEVELOPMENT_ONLY,
        "assessment_status": ASSESSMENT_STATUS,
        "scientific_promotion_allowed": SCIENTIFIC_PROMOTION_ALLOWED,
        "winner_selection_allowed": WINNER_SELECTION_ALLOWED,
        "performance_thresholds_applied": PERFORMANCE_THRESHOLDS_APPLIED,
        "output_writes": OUTPUT_WRITES,
        "config": config.to_config(),
        "protocol": _protocol(config),
        "runs": [_run_seed(config, seed) for seed in config.seeds],
        "limitations": list(_LIMITATIONS),
        "source_manifest": matched_discrete_control_source_manifest(),
    }
    report["report_sha256"] = _digest(report)
    if len(_canonical_json_bytes(report)) > config.max_report_bytes:
        raise ValueError("matched-control report exceeds its configured byte bound")
    return report


def _redigest_report(report: dict[str, object]) -> None:
    """Internal test helper: recompute integrity after deliberate tampering."""

    payload = dict(report)
    payload.pop("report_sha256", None)
    report["report_sha256"] = _digest(payload)


def validate_matched_discrete_control_development_report(
    report: Mapping[str, object],
    *,
    replay: bool = True,
) -> MatchedDiscreteControlValidation:
    """Fail closed on claims, digest, sources, structure, or deterministic replay."""

    errors: list[str] = []
    expected_fields = {
        "schema",
        "development_only",
        "assessment_status",
        "scientific_promotion_allowed",
        "winner_selection_allowed",
        "performance_thresholds_applied",
        "output_writes",
        "config",
        "protocol",
        "runs",
        "limitations",
        "source_manifest",
        "report_sha256",
    }
    if set(report) != expected_fields:
        errors.append("report fields differ from the v1 schema")
    fixed = {
        "schema": MATCHED_DISCRETE_CONTROL_REPORT_SCHEMA,
        "development_only": True,
        "assessment_status": ASSESSMENT_STATUS,
        "scientific_promotion_allowed": False,
        "winner_selection_allowed": False,
        "performance_thresholds_applied": False,
        "output_writes": False,
        "limitations": list(_LIMITATIONS),
    }
    for name, expected in fixed.items():
        if not _strict_json_equal(report.get(name), expected):
            errors.append(f"nonpromoting report declaration {name} changed")
    digest = report.get("report_sha256")
    payload = dict(report)
    payload.pop("report_sha256", None)
    if type(digest) is not str or digest != _digest(payload):
        errors.append("report digest does not match canonical content")
    if not _strict_json_equal(
        report.get("source_manifest"),
        matched_discrete_control_source_manifest(),
    ):
        errors.append("report source manifest does not match current sources")
    config: MatchedDiscreteControlDevelopmentConfig | None = None
    try:
        raw_config = report.get("config")
        if not isinstance(raw_config, Mapping):
            raise ValueError("config is not a mapping")
        config = MatchedDiscreteControlDevelopmentConfig.from_config(raw_config)
    except (TypeError, ValueError) as error:
        errors.append(f"report config is invalid: {error}")
    if config is not None and not _strict_json_equal(report.get("protocol"), _protocol(config)):
        errors.append("report protocol does not reconstruct from config")
    if replay and config is not None and not errors:
        try:
            expected_report = _build_report(config)
        except Exception as error:  # pragma: no cover - defensive fail-closed boundary
            errors.append(f"deterministic replay raised {type(error).__name__}: {error}")
        else:
            if not _strict_json_equal(dict(report), expected_report):
                errors.append("deterministic replay does not match the retained full report")
    return MatchedDiscreteControlValidation(valid=not errors, errors=tuple(errors))


def run_matched_discrete_control_development(
    config: MatchedDiscreteControlDevelopmentConfig | None = None,
) -> dict[str, object]:
    """Build and strictly replay-validate the in-memory development report."""

    config = MatchedDiscreteControlDevelopmentConfig() if config is None else config
    if type(config) is not MatchedDiscreteControlDevelopmentConfig:
        raise TypeError("config must be an exact matched-control development config")
    report = _build_report(config)
    validation = validate_matched_discrete_control_development_report(report)
    if not validation.valid:
        raise RuntimeError(
            "internally generated matched-control report failed validation: "
            + "; ".join(validation.errors)
        )
    return report


def matched_discrete_control_development_report_json(
    report: Mapping[str, object],
) -> str:
    """Return canonical JSON only after strict deterministic replay validation."""

    validation = validate_matched_discrete_control_development_report(report)
    if not validation.valid:
        raise ValueError("invalid matched-control report: " + "; ".join(validation.errors))
    return _canonical_json_bytes(dict(report)).decode("utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the non-writing development CLI and print one canonical JSON report."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="31101,31102")
    parser.add_argument("--phase-length", type=int, default=64)
    parser.add_argument("--summary-window", type=int, default=16)
    args = parser.parse_args(argv)
    try:
        seeds = tuple(int(token) for token in args.seeds.split(",") if token)
        config = MatchedDiscreteControlDevelopmentConfig(
            seeds=seeds,
            phase_length=args.phase_length,
            summary_window=args.summary_window,
        )
        report = run_matched_discrete_control_development(config)
    except (RuntimeError, TypeError, ValueError) as error:
        parser.error(str(error))
    sys.stdout.write(_canonical_json_bytes(report).decode("utf-8") + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the module CLI
    raise SystemExit(main())


__all__ = [
    "ARM_ORDER",
    "ASSESSMENT_STATUS",
    "DEVELOPMENT_ONLY",
    "MATCHED_DISCRETE_CONTROL_CHECKPOINT_SCHEMA",
    "MATCHED_DISCRETE_CONTROL_CONFIG_SCHEMA",
    "MATCHED_DISCRETE_CONTROL_REPORT_SCHEMA",
    "MatchedDiscreteControlDevelopmentConfig",
    "MatchedDiscreteControlRunState",
    "MatchedDiscreteControlTrace",
    "MatchedDiscreteControlValidation",
    "OUTPUT_WRITES",
    "PERFORMANCE_THRESHOLDS_APPLIED",
    "REGIME_SCHEDULE",
    "SCIENTIFIC_PROMOTION_ALLOWED",
    "WINNER_SELECTION_ALLOWED",
    "advance_matched_discrete_control_run",
    "initialize_matched_discrete_control_run",
    "load_matched_discrete_control_checkpoint",
    "main",
    "matched_discrete_control_development_report_json",
    "matched_discrete_control_run_state_sha256",
    "matched_discrete_control_source_manifest",
    "run_matched_discrete_control_development",
    "save_matched_discrete_control_checkpoint",
    "validate_matched_discrete_control_development_report",
]
