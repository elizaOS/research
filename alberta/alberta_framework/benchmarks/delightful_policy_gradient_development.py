# mypy: disable-error-code="attr-defined,call-arg"
"""Development-only matched benchmarks for Delightful Policy Gradient.

This module evaluates the paper-specific actor-sample gate, never the separate
Prototype ``sparks_joy`` candidate-update audit.  It reports Kondo selection
accounting: a deterministic post-hoc counterfactual that replays the paper's
delight-price selection rule over each fully executed trace and accounts, per
learning-value channel, the backward work a real Kondo gate would have run
versus skipped.  Actual Kondo compute gating remains unimplemented here.
Every actor, critic, and average-reward update in every life is executed in
full and no compiled backward work is skipped, so ``KONDO_IMPLEMENTED`` stays
``False`` — the implemented-Kondo bar (see
:mod:`alberta_framework.core.kondo_gate`) requires demonstrably skipped
backward work, which masking or post-hoc accounting cannot provide — and
"Kondo compute savings" stays an excluded claim.  There is no promotion path.
Ordinary and paper-specific DG policy gradients
use matched categorical actor-critic configurations that differ only in gate mode,
the same initialization and sampler, paired action/environment random-number
schedules, and the same logical resource budget.  Their realized actions and
transitions may diverge once their policies diverge.

The fixed benchmark order follows the continual-control plan: a contextual
heteroskedastic gambling stream first, then a six-state switching RiverSwim
control.  Both are uninterrupted A/B/A lives.  Every trace is prequential: the
logged policy and action exist before the corresponding outcome and update.

The paper-specific DG delight/gate equations are those of
:func:`alberta_framework.core.delight.discrete_delightful_policy_gradient`,
implementing the bounded discrete-action experiment specified in WP5 of
``CONTINUAL_AGENT_IMPLEMENTATION_PLAN.md``.

Reports are deterministic, versioned, JSON-compatible development records.
Their strict validator checks schema, digest, common-random-number pairing,
environment reconstruction, action sampling, exact paper-specific DG
delight/gate equations, metrics, strata, Kondo selection accounting, and
logical accounting.  No result or threshold in this file can promote a
scientific claim.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import itertools
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any, Literal, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array
from jaxtyping import Bool, Float, Int, UInt32

from alberta_framework.core.delight import (
    DelightfulPolicyGradientConfig,
    PolicyGradientMode,
    discrete_delightful_policy_gradient,
    stratify_delight_outcomes,
)
from alberta_framework.core.delightful_actor_critic import (
    DelightfulActorCriticAgent,
    DelightfulActorCriticConfig,
    DelightfulActorCriticState,
    DelightfulChannelAvailability,
)
from alberta_framework.streams.closed_loop import (
    LEFT_ACTION,
    RIGHT_ACTION,
    RiverSwimConfig,
    RiverSwimMDP,
)

DELIGHTFUL_POLICY_GRADIENT_DEVELOPMENT_SCHEMA = "alberta.delightful-policy-gradient.development.v2"
DEVELOPMENT_ONLY = True
SCIENTIFIC_PROMOTION_ALLOWED = False
# Actual Kondo compute gating (skipping compiled backward work) is not
# implemented in this harness; only counterfactual selection accounting is.
KONDO_IMPLEMENTED = False
KONDO_SELECTION_ACCOUNTING = True

BenchmarkName = Literal[
    "contextual_heteroskedastic_gambling",
    "six_state_riverswim",
]

BENCHMARK_ORDER: tuple[BenchmarkName, ...] = (
    "contextual_heteroskedastic_gambling",
    "six_state_riverswim",
)
POLICY_GRADIENT_MODES: tuple[PolicyGradientMode, ...] = (
    "ordinary_pg",
    "delightful_pg",
)
REGIME_SCHEDULE = (0, 1, 0)

_CLAIM_SCOPE = (
    "development-only matched mechanism diagnostics for discrete on-policy "
    "Delightful Policy Gradient"
)
_EXCLUDED_CLAIMS = (
    "scientific evidence promotion",
    "Delightful Policy Gradient performance improvement",
    "Kondo compute savings",
    "continuous-control validity",
    "Prototype sparks_joy audit validity",
)
_INTERPRETATION = (
    "calculation and matched development diagnostics only; no thresholds or "
    "results in this report can promote a claim"
)
# Frozen parameters of the counterfactual Kondo selection rule.  The price is
# the paper's Bernoulli-gate location parameter; selection is its deterministic
# maximum-probability decision, `delight > price`.
_KONDO_SELECTION_PRICE = 0.0
_KONDO_ACCOUNTING_SEMANTICS = (
    "measured post-hoc counterfactual over one fully executed life; every "
    "actor, critic, and average-reward update ran in full, no backward work "
    "was skipped, and no compute saving is claimed"
)
_UINT32_MAX = 2**32 - 1
_INT32_MIN = -(2**31)
_INT32_MAX = 2**31 - 1
_FLOAT32_TINY = float(np.finfo(np.float32).tiny)

# Regime x context x action expected rewards.  Action 1 is deliberately
# heteroskedastic.  In regime A it is a suboptimal gamble in context 0 and the
# better action in context 1; regime B reverses the preferred action by context.
_CONTEXT_MEANS = (
    ((1.0, 0.4), (0.3, 0.8)),
    ((0.2, 0.9), (1.0, 0.4)),
)
_CONTEXT_STANDARD_DEVIATIONS = (0.1, 2.0)

# RiverSwim-style stochastic chain (Strehl & Littman 2008); values match the
# :class:`RiverSwimConfig` defaults documented in
# ``alberta_framework.streams.closed_loop``.  The regime tables below only swap
# which boundary pays the large reward, so regime B reverses the optimal policy
# without changing the transition kernel.
_RIVER_CONFIG = RiverSwimConfig(
    n_states=6,
    p_right_up=0.35,
    p_right_down=0.05,
    reward_left=0.005,
    reward_right=1.0,
    initial_state=0,
)
_RIVER_LEFT_REWARD_BY_REGIME = (0.005, 1.0)
_RIVER_RIGHT_REWARD_BY_REGIME = (1.0, 0.005)

# Arbitrary-but-frozen fold-in tags (see ``_scenario_keys``) that split each
# seed's root key into independent agent and environment streams per benchmark.
# The values carry no meaning, but changing any of them changes every trace.
_CONTEXT_AGENT_SEED_TAG = 1_001
_CONTEXT_ENVIRONMENT_SEED_TAG = 1_002
_RIVER_AGENT_SEED_TAG = 2_001
_RIVER_ENVIRONMENT_SEED_TAG = 2_002


def _positive_int(value: object, *, name: str, maximum: int = _INT32_MAX) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be a non-boolean integer in 1..{maximum}")
    return value


def _finite_float32(
    value: object,
    *,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
    allow_zero: bool = True,
) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a real scalar")
    parsed = float(value)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        narrowed = float(np.float32(parsed))
    if not math.isfinite(parsed) or not math.isfinite(narrowed):
        raise ValueError(f"{name} must be finite in float32")
    if parsed != 0.0 and abs(narrowed) < _FLOAT32_TINY:
        raise ValueError(f"{name} must not become a float32 subnormal")
    if not allow_zero and narrowed == 0.0:
        raise ValueError(f"{name} must be nonzero")
    if minimum is not None and narrowed < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if maximum is not None and narrowed > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return narrowed


def _strict_json_equal(actual: object, expected: object) -> bool:
    """Compare JSON trees without treating booleans, integers, and floats as interchangeable."""
    if isinstance(expected, bool):
        return isinstance(actual, bool) and actual is expected
    if expected is None:
        return actual is None
    if isinstance(expected, int):
        return isinstance(actual, int) and not isinstance(actual, bool) and actual == expected
    if isinstance(expected, float):
        return isinstance(actual, float) and math.isfinite(actual) and actual == expected
    if isinstance(expected, str):
        return isinstance(actual, str) and actual == expected
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(
                _strict_json_equal(actual_item, expected_item)
                for actual_item, expected_item in zip(actual, expected, strict=True)
            )
        )
    if isinstance(expected, Mapping):
        return (
            isinstance(actual, Mapping)
            and set(actual) == set(expected)
            and all(_strict_json_equal(actual[key], expected[key]) for key in expected)
        )
    return False


@dataclasses.dataclass(frozen=True)
class DelightfulPolicyGradientDevelopmentConfig:
    """Fixed A/B/A development protocol with no acceptance thresholds."""

    seeds: tuple[int, ...] = (0, 1, 2)
    phase_length: int = 128
    adaptation_window: int = 32
    recovery_window: int = 8
    final_window: int = 32
    worst_window: int = 32
    diagnostic_recovery_fraction: float = 0.8
    rare_probability_cutoff: float = 0.25
    contextual_safety_reward_floor: float = -2.0
    actor_step_size: float = 0.02
    critic_step_size: float = 0.05
    average_reward_step_size: float = 0.01
    critic_trace_lambda: float = 0.5
    policy_temperature: float = 1.0
    delight_temperature: float = 1.0
    diagnostics_epsilon: float = 1.0e-8
    max_input_magnitude: float = 1.0e6
    max_parameter_magnitude: float = 1.0e6
    max_update_component_magnitude: float = 10.0

    def __post_init__(self) -> None:
        if (
            not isinstance(self.seeds, tuple)
            or not self.seeds
            or len(set(self.seeds)) != len(self.seeds)
        ):
            raise ValueError("seeds must be a nonempty tuple of unique uint32 integers")
        for seed in self.seeds:
            if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= _UINT32_MAX:
                raise ValueError("seeds must be a nonempty tuple of unique uint32 integers")
        _positive_int(
            self.phase_length,
            name="phase_length",
            maximum=_INT32_MAX // len(REGIME_SCHEDULE),
        )
        for name in (
            "adaptation_window",
            "recovery_window",
            "final_window",
            "worst_window",
        ):
            value = _positive_int(getattr(self, name), name=name)
            if value > self.phase_length:
                raise ValueError(f"{name} must not exceed phase_length")
        controls = {
            "diagnostic_recovery_fraction": (0.0, 1.0, False),
            "rare_probability_cutoff": (0.0, 1.0, False),
            "contextual_safety_reward_floor": (None, None, True),
            "actor_step_size": (0.0, None, True),
            "critic_step_size": (0.0, None, True),
            "average_reward_step_size": (0.0, None, True),
            "critic_trace_lambda": (0.0, 1.0, True),
            "policy_temperature": (_FLOAT32_TINY, None, False),
            "delight_temperature": (_FLOAT32_TINY, None, False),
            "diagnostics_epsilon": (_FLOAT32_TINY, None, False),
            "max_input_magnitude": (_FLOAT32_TINY, None, False),
            "max_parameter_magnitude": (_FLOAT32_TINY, None, False),
            "max_update_component_magnitude": (_FLOAT32_TINY, None, False),
        }
        for name, (minimum, maximum, allow_zero) in controls.items():
            object.__setattr__(
                self,
                name,
                _finite_float32(
                    getattr(self, name),
                    name=name,
                    minimum=minimum,
                    maximum=maximum,
                    allow_zero=allow_zero,
                ),
            )

    @property
    def total_steps(self) -> int:
        """Transitions in one uninterrupted A/B/A life."""
        return len(REGIME_SCHEDULE) * self.phase_length

    def agent_config(
        self,
        observation_dim: int,
        mode: PolicyGradientMode,
    ) -> DelightfulActorCriticConfig:
        """Construct one matched actor-critic config; only ``mode`` may differ."""
        return DelightfulActorCriticConfig(
            observation_dim=observation_dim,
            n_actions=2,
            mode=mode,
            actor_step_size=self.actor_step_size,
            critic_step_size=self.critic_step_size,
            average_reward_step_size=self.average_reward_step_size,
            actor_trace_lambda=0.0,
            critic_trace_lambda=self.critic_trace_lambda,
            policy_temperature=self.policy_temperature,
            delight_temperature=self.delight_temperature,
            diagnostics_epsilon=self.diagnostics_epsilon,
            max_input_magnitude=self.max_input_magnitude,
            max_parameter_magnitude=self.max_parameter_magnitude,
            max_update_component_magnitude=self.max_update_component_magnitude,
            max_updates=self.total_steps,
        )

    def to_config(self) -> dict[str, object]:
        """Return the complete, nonpromoting v2 protocol record."""
        return {
            "schema_version": DELIGHTFUL_POLICY_GRADIENT_DEVELOPMENT_SCHEMA,
            "development_only": DEVELOPMENT_ONLY,
            "scientific_promotion_allowed": SCIENTIFIC_PROMOTION_ALLOWED,
            "kondo_implemented": KONDO_IMPLEMENTED,
            "kondo_selection_accounting": KONDO_SELECTION_ACCOUNTING,
            "claim_scope": _CLAIM_SCOPE,
            "excluded_claims": list(_EXCLUDED_CLAIMS),
            "benchmark_order": list(BENCHMARK_ORDER),
            "policy_gradient_modes": list(POLICY_GRADIENT_MODES),
            "regime_schedule": list(REGIME_SCHEDULE),
            "seeds": list(self.seeds),
            "phase_length": self.phase_length,
            "adaptation_window": self.adaptation_window,
            "recovery_window": self.recovery_window,
            "final_window": self.final_window,
            "worst_window": self.worst_window,
            "diagnostic_recovery_fraction": self.diagnostic_recovery_fraction,
            "rare_probability_cutoff": self.rare_probability_cutoff,
            "contextual_safety_reward_floor": self.contextual_safety_reward_floor,
            "contextual_environment": {
                "means_by_regime_context_action": [
                    [[float(value) for value in action] for action in regime]
                    for regime in _CONTEXT_MEANS
                ],
                "standard_deviation_by_action": list(_CONTEXT_STANDARD_DEVIATIONS),
                "context_schedule": "alternating_0_1",
                "risky_action": 1,
            },
            "riverswim_environment": {
                "config": {
                    "n_states": _RIVER_CONFIG.n_states,
                    "p_right_up": _RIVER_CONFIG.p_right_up,
                    "p_right_down": _RIVER_CONFIG.p_right_down,
                    "reward_left": _RIVER_CONFIG.reward_left,
                    "reward_right": _RIVER_CONFIG.reward_right,
                    "initial_state": _RIVER_CONFIG.initial_state,
                },
                "left_reward_by_regime": list(_RIVER_LEFT_REWARD_BY_REGIME),
                "right_reward_by_regime": list(_RIVER_RIGHT_REWARD_BY_REGIME),
                "safety_cost": "right_action_at_right_boundary",
            },
            "actor_step_size": self.actor_step_size,
            "critic_step_size": self.critic_step_size,
            "average_reward_step_size": self.average_reward_step_size,
            "actor_trace_lambda": 0.0,
            "critic_trace_lambda": self.critic_trace_lambda,
            "policy_temperature": self.policy_temperature,
            "delight_temperature": self.delight_temperature,
            "diagnostics_epsilon": self.diagnostics_epsilon,
            "max_input_magnitude": self.max_input_magnitude,
            "max_parameter_magnitude": self.max_parameter_magnitude,
            "max_update_component_magnitude": self.max_update_component_magnitude,
            "total_steps": self.total_steps,
        }

    @classmethod
    def from_config(
        cls,
        payload: Mapping[str, object],
    ) -> DelightfulPolicyGradientDevelopmentConfig:
        """Reconstruct only an exact nonpromoting v2 protocol record."""
        expected = set(cls().to_config())
        if set(payload) != expected:
            raise ValueError("development protocol fields do not match the v2 schema")
        seeds = payload.get("seeds")
        if not isinstance(seeds, list):
            raise ValueError("development protocol seeds must be a JSON array")
        config = cls(
            seeds=tuple(cast(list[int], seeds)),
            phase_length=cast(int, payload["phase_length"]),
            adaptation_window=cast(int, payload["adaptation_window"]),
            recovery_window=cast(int, payload["recovery_window"]),
            final_window=cast(int, payload["final_window"]),
            worst_window=cast(int, payload["worst_window"]),
            diagnostic_recovery_fraction=cast(float, payload["diagnostic_recovery_fraction"]),
            rare_probability_cutoff=cast(float, payload["rare_probability_cutoff"]),
            contextual_safety_reward_floor=cast(float, payload["contextual_safety_reward_floor"]),
            actor_step_size=cast(float, payload["actor_step_size"]),
            critic_step_size=cast(float, payload["critic_step_size"]),
            average_reward_step_size=cast(float, payload["average_reward_step_size"]),
            critic_trace_lambda=cast(float, payload["critic_trace_lambda"]),
            policy_temperature=cast(float, payload["policy_temperature"]),
            delight_temperature=cast(float, payload["delight_temperature"]),
            diagnostics_epsilon=cast(float, payload["diagnostics_epsilon"]),
            max_input_magnitude=cast(float, payload["max_input_magnitude"]),
            max_parameter_magnitude=cast(float, payload["max_parameter_magnitude"]),
            max_update_component_magnitude=cast(float, payload["max_update_component_magnitude"]),
        )
        if not _strict_json_equal(dict(payload), config.to_config()):
            raise ValueError("development protocol contains changed claims or environments")
        return config


@chex.dataclass(frozen=True)
class _DevelopmentTraceArrays:
    """Fixed-shape prequential arrays emitted by one uninterrupted life."""

    step: Int[Array, " steps"]
    phase_index: Int[Array, " steps"]
    regime_id: Int[Array, " steps"]
    state_index: Int[Array, " steps"]
    next_state_index: Int[Array, " steps"]
    action: Int[Array, " steps"]
    behavior_policy: Float[Array, "steps 2"]
    target_policy: Float[Array, "steps 2"]
    action_probability: Float[Array, " steps"]
    behavior_log_probability: Float[Array, " steps"]
    reward: Float[Array, " steps"]
    expected_reward: Float[Array, " steps"]
    oracle_reward: Float[Array, " steps"]
    advantage: Float[Array, " steps"]
    delight: Float[Array, " steps"]
    gate_weight: Float[Array, " steps"]
    safety_cost: Float[Array, " steps"]
    safety_violation: Bool[Array, " steps"]
    applied: Bool[Array, " steps"]
    transition_count_after: Int[Array, " steps"]
    action_rng_words: UInt32[Array, "steps 2"]
    environment_rng_words: UInt32[Array, "steps 2"]


def _availability() -> DelightfulChannelAvailability:
    true = jnp.asarray(True, dtype=jnp.bool_)
    return DelightfulChannelAvailability(
        safety=true,
        model=true,
        representation=true,
    )


def _phase_index(step: Array, phase_length: int) -> Array:
    return jnp.floor_divide(step, phase_length).astype(jnp.int32)


def _regime_id(phase_index: Array) -> Array:
    return jnp.where(phase_index == 1, 1, 0).astype(jnp.int32)


def _next_action_sample_key(
    state: DelightfulActorCriticState,
    current_sample_key: Array,
    applied: Array,
) -> Array:
    _, candidate_sample_key = jr.split(state.rng_key)
    return cast(
        Array,
        jax.lax.cond(
            applied,
            lambda _: candidate_sample_key,
            lambda _: current_sample_key,
            operand=None,
        ),
    )


@functools.partial(jax.jit, static_argnums=(0, 1))
def _run_contextual_arrays(
    config: DelightfulPolicyGradientDevelopmentConfig,
    mode: PolicyGradientMode,
    agent_key: Array,
    environment_key: Array,
) -> tuple[DelightfulActorCriticState, DelightfulActorCriticState, _DevelopmentTraceArrays]:
    agent = DelightfulActorCriticAgent(config.agent_config(2, mode))
    initial = agent.init(agent_key)
    observation = jax.nn.one_hot(0, 2, dtype=jnp.float32)
    started = agent.start(initial, observation)
    _, first_sample_key = jr.split(agent_key)
    means = jnp.asarray(_CONTEXT_MEANS, dtype=jnp.float32)
    deviations = jnp.asarray(_CONTEXT_STANDARD_DEVIATIONS, dtype=jnp.float32)
    safety_floor = jnp.asarray(config.contextual_safety_reward_floor, dtype=jnp.float32)

    def scan_step(
        carry: tuple[DelightfulActorCriticState, Array],
        step: Array,
    ) -> tuple[tuple[DelightfulActorCriticState, Array], _DevelopmentTraceArrays]:
        state, sample_key = carry
        phase = _phase_index(step, config.phase_length)
        regime = _regime_id(phase)
        context = jnp.mod(step, 2).astype(jnp.int32)
        next_context = jnp.mod(step + 1, 2).astype(jnp.int32)
        action = state.last_sample.action
        outcome_key = jr.fold_in(environment_key, step)
        draw = jr.normal(outcome_key, (), dtype=jnp.float32)
        expected_reward = means[regime, context, action]
        reward = expected_reward + deviations[action] * draw
        oracle_reward = jnp.max(means[regime, context])
        next_observation = jax.nn.one_hot(next_context, 2, dtype=jnp.float32)
        update = agent.update(state, reward, next_observation, _availability())
        next_sample_key = _next_action_sample_key(
            state,
            sample_key,
            update.diagnostics.applied,
        )
        safety_cost = jnp.maximum(safety_floor - reward, 0.0)
        trace = _DevelopmentTraceArrays(
            step=step,
            phase_index=phase,
            regime_id=regime,
            state_index=context,
            next_state_index=next_context,
            action=action,
            behavior_policy=state.last_sample.behavior_policy,
            target_policy=state.last_sample.target_policy,
            action_probability=state.last_sample.behavior_probability,
            behavior_log_probability=state.last_sample.behavior_log_probability,
            reward=reward,
            expected_reward=expected_reward,
            oracle_reward=oracle_reward,
            advantage=update.diagnostics.advantage,
            delight=update.diagnostics.delight,
            gate_weight=update.diagnostics.gate_weight,
            safety_cost=safety_cost,
            safety_violation=reward < safety_floor,
            applied=update.diagnostics.applied,
            transition_count_after=update.state.transition_count,
            action_rng_words=jr.key_data(sample_key),
            environment_rng_words=jr.key_data(outcome_key),
        )
        return (update.state, next_sample_key), trace

    (final_state, _), trace = jax.lax.scan(
        scan_step,
        (started.state, first_sample_key),
        jnp.arange(config.total_steps, dtype=jnp.int32),
    )
    return started.state, final_state, trace


def _stationary_average_reward(
    transition: np.ndarray,
    step_rewards: np.ndarray,
) -> float:
    n_states = transition.shape[0]
    constraints = np.vstack(
        (transition.T - np.eye(n_states, dtype=np.float64), np.ones((1, n_states)))
    )
    targets = np.zeros((n_states + 1,), dtype=np.float64)
    targets[-1] = 1.0
    distribution, *_ = np.linalg.lstsq(constraints, targets, rcond=None)
    distribution = np.clip(distribution, 0.0, None)
    distribution /= np.sum(distribution)
    return float(distribution @ step_rewards)


def _river_reward_table(regime: int) -> np.ndarray:
    rewards = np.zeros((_RIVER_CONFIG.n_states, 2), dtype=np.float32)
    rewards[0, LEFT_ACTION] = _RIVER_LEFT_REWARD_BY_REGIME[regime]
    rewards[-1, RIGHT_ACTION] = _RIVER_RIGHT_REWARD_BY_REGIME[regime]
    return rewards


@functools.lru_cache(maxsize=1)
def _river_oracle_rewards() -> tuple[float, float]:
    environment = RiverSwimMDP(_RIVER_CONFIG)
    transition = environment.transition_tensor.astype(np.float64)
    results: list[float] = []
    for regime in (0, 1):
        rewards = _river_reward_table(regime).astype(np.float64)
        best = -math.inf
        for policy in itertools.product((LEFT_ACTION, RIGHT_ACTION), repeat=_RIVER_CONFIG.n_states):
            actions = np.asarray(policy, dtype=np.int64)
            kernel = transition[actions, np.arange(_RIVER_CONFIG.n_states)]
            per_state = rewards[np.arange(_RIVER_CONFIG.n_states), actions]
            best = max(best, _stationary_average_reward(kernel, per_state))
        results.append(best)
    return results[0], results[1]


@functools.partial(jax.jit, static_argnums=(0, 1))
def _run_riverswim_arrays(
    config: DelightfulPolicyGradientDevelopmentConfig,
    mode: PolicyGradientMode,
    agent_key: Array,
    environment_key: Array,
) -> tuple[DelightfulActorCriticState, DelightfulActorCriticState, _DevelopmentTraceArrays]:
    environment = RiverSwimMDP(_RIVER_CONFIG)
    agent = DelightfulActorCriticAgent(config.agent_config(environment.feature_dim, mode))
    environment_state = environment.init(environment_key)
    initial = agent.init(agent_key)
    started = agent.start(initial, environment.observe(environment_state))
    _, first_sample_key = jr.split(agent_key)
    reward_tables = jnp.stack(
        (
            jnp.asarray(_river_reward_table(0)),
            jnp.asarray(_river_reward_table(1)),
        )
    )
    oracle_rewards = jnp.asarray(_river_oracle_rewards(), dtype=jnp.float32)

    def scan_step(
        carry: tuple[DelightfulActorCriticState, Any, Array],
        step: Array,
    ) -> tuple[tuple[DelightfulActorCriticState, Any, Array], _DevelopmentTraceArrays]:
        state, river_state, sample_key = carry
        phase = _phase_index(step, config.phase_length)
        regime = _regime_id(phase)
        action = state.last_sample.action
        outcome_key = jr.fold_in(environment_key, step)
        next_observation, _, next_river_state = environment.step(
            river_state,
            action,
            outcome_key,
        )
        reward = reward_tables[regime, river_state.state_index, action]
        update = agent.update(state, reward, next_observation, _availability())
        next_sample_key = _next_action_sample_key(
            state,
            sample_key,
            update.diagnostics.applied,
        )
        boundary_pressure = (river_state.state_index == environment.n_states - 1) & (
            action == RIGHT_ACTION
        )
        trace = _DevelopmentTraceArrays(
            step=step,
            phase_index=phase,
            regime_id=regime,
            state_index=river_state.state_index,
            next_state_index=next_river_state.state_index,
            action=action,
            behavior_policy=state.last_sample.behavior_policy,
            target_policy=state.last_sample.target_policy,
            action_probability=state.last_sample.behavior_probability,
            behavior_log_probability=state.last_sample.behavior_log_probability,
            reward=reward,
            expected_reward=reward,
            oracle_reward=oracle_rewards[regime],
            advantage=update.diagnostics.advantage,
            delight=update.diagnostics.delight,
            gate_weight=update.diagnostics.gate_weight,
            safety_cost=boundary_pressure.astype(jnp.float32),
            safety_violation=boundary_pressure,
            applied=update.diagnostics.applied,
            transition_count_after=update.state.transition_count,
            action_rng_words=jr.key_data(sample_key),
            environment_rng_words=jr.key_data(outcome_key),
        )
        return (update.state, next_river_state, next_sample_key), trace

    (final_state, _, _), trace = jax.lax.scan(
        scan_step,
        (started.state, environment_state, first_sample_key),
        jnp.arange(config.total_steps, dtype=jnp.int32),
    )
    return started.state, final_state, trace


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _scenario_keys(seed: int, benchmark: BenchmarkName) -> tuple[Array, Array]:
    root = jr.key(seed)
    if benchmark == "contextual_heteroskedastic_gambling":
        return (
            jr.fold_in(root, _CONTEXT_AGENT_SEED_TAG),
            jr.fold_in(root, _CONTEXT_ENVIRONMENT_SEED_TAG),
        )
    return (
        jr.fold_in(root, _RIVER_AGENT_SEED_TAG),
        jr.fold_in(root, _RIVER_ENVIRONMENT_SEED_TAG),
    )


def _trace_to_dict(trace: _DevelopmentTraceArrays) -> dict[str, object]:
    def values(array: Array) -> object:
        return np.asarray(jax.device_get(array)).tolist()

    return {
        "step": values(trace.step),
        "phase_index": values(trace.phase_index),
        "regime_id": values(trace.regime_id),
        "state_index": values(trace.state_index),
        "next_state_index": values(trace.next_state_index),
        "action": values(trace.action),
        "behavior_policy": values(trace.behavior_policy),
        "target_policy": values(trace.target_policy),
        "action_probability": values(trace.action_probability),
        "behavior_log_probability": values(trace.behavior_log_probability),
        "reward": values(trace.reward),
        "expected_reward": values(trace.expected_reward),
        "oracle_reward": values(trace.oracle_reward),
        "advantage": values(trace.advantage),
        "delight": values(trace.delight),
        "gate_weight": values(trace.gate_weight),
        "safety_cost": values(trace.safety_cost),
        "safety_violation": values(trace.safety_violation),
        "applied": values(trace.applied),
        "transition_count_after": values(trace.transition_count_after),
        "action_rng_words": values(trace.action_rng_words),
        "environment_rng_words": values(trace.environment_rng_words),
    }


def _trace_arrays_from_dict(trace: Mapping[str, object]) -> dict[str, np.ndarray]:
    return {
        "step": np.asarray(trace["step"], dtype=np.int32),
        "phase_index": np.asarray(trace["phase_index"], dtype=np.int32),
        "regime_id": np.asarray(trace["regime_id"], dtype=np.int32),
        "state_index": np.asarray(trace["state_index"], dtype=np.int32),
        "next_state_index": np.asarray(trace["next_state_index"], dtype=np.int32),
        "action": np.asarray(trace["action"], dtype=np.int32),
        "behavior_policy": np.asarray(trace["behavior_policy"], dtype=np.float32),
        "target_policy": np.asarray(trace["target_policy"], dtype=np.float32),
        "action_probability": np.asarray(trace["action_probability"], dtype=np.float32),
        "behavior_log_probability": np.asarray(trace["behavior_log_probability"], dtype=np.float32),
        "reward": np.asarray(trace["reward"], dtype=np.float32),
        "expected_reward": np.asarray(trace["expected_reward"], dtype=np.float32),
        "oracle_reward": np.asarray(trace["oracle_reward"], dtype=np.float32),
        "advantage": np.asarray(trace["advantage"], dtype=np.float32),
        "delight": np.asarray(trace["delight"], dtype=np.float32),
        "gate_weight": np.asarray(trace["gate_weight"], dtype=np.float32),
        "safety_cost": np.asarray(trace["safety_cost"], dtype=np.float32),
        "safety_violation": np.asarray(trace["safety_violation"], dtype=np.bool_),
        "applied": np.asarray(trace["applied"], dtype=np.bool_),
        "transition_count_after": np.asarray(trace["transition_count_after"], dtype=np.int32),
        "action_rng_words": np.asarray(trace["action_rng_words"], dtype=np.uint32),
        "environment_rng_words": np.asarray(trace["environment_rng_words"], dtype=np.uint32),
    }


def _rolling_worst_mean(values: np.ndarray, window: int) -> float:
    if values.size < window:
        raise ValueError("rolling window exceeds the trace length")
    sums = np.convolve(values.astype(np.float64), np.ones(window), mode="valid")
    return float(np.min(sums / window))


def _first_recovery_step(
    rewards: np.ndarray,
    target: float,
    window: int,
) -> int | None:
    if rewards.size < window:
        return None
    for offset in range(rewards.size - window + 1):
        if float(np.mean(rewards[offset : offset + window], dtype=np.float64)) >= target:
            return offset + window
    return None


def _performance_metrics(
    trace: Mapping[str, object],
    config: DelightfulPolicyGradientDevelopmentConfig,
) -> dict[str, object]:
    arrays = _trace_arrays_from_dict(trace)
    reward = arrays["reward"].astype(np.float64)
    oracle = arrays["oracle_reward"].astype(np.float64)
    safety_cost = arrays["safety_cost"].astype(np.float64)
    safety_violation = arrays["safety_violation"]
    phase_means: list[float] = []
    phase_finals: list[float] = []
    adaptation_auc: list[float] = []
    normalized_adaptation_auc: list[float] = []
    recovery_steps: list[int | None] = []
    recovery_targets: list[float] = []
    stability_gaps: list[float] = []
    for phase in range(len(REGIME_SCHEDULE)):
        start = phase * config.phase_length
        end = start + config.phase_length
        phase_reward = reward[start:end]
        phase_oracle = oracle[start:end]
        phase_means.append(float(np.mean(phase_reward)))
        phase_finals.append(float(np.mean(phase_reward[-config.final_window :])))
        if phase == 0:
            continue
        adaptation_reward = phase_reward[: config.adaptation_window]
        adaptation_oracle = phase_oracle[: config.adaptation_window]
        raw_auc = float(np.sum(adaptation_reward))
        oracle_auc = float(np.sum(adaptation_oracle))
        adaptation_auc.append(raw_auc)
        normalized_adaptation_auc.append(raw_auc / max(oracle_auc, _FLOAT32_TINY))
        target = config.diagnostic_recovery_fraction * float(np.mean(phase_oracle))
        recovery_targets.append(target)
        recovery_steps.append(_first_recovery_step(phase_reward, target, config.recovery_window))
        stability_gaps.append(float(np.mean(adaptation_oracle) - np.mean(adaptation_reward)))
    initial_final = phase_finals[0]
    recurring_final = phase_finals[2]
    retention_ratio: float | None = None
    if abs(initial_final) > _FLOAT32_TINY:
        retention_ratio = recurring_final / initial_final
    return {
        "lifetime_return": float(np.sum(reward)),
        "mean_prequential_reward": float(np.mean(reward)),
        "phase_mean_rewards": phase_means,
        "phase_final_window_rewards": phase_finals,
        "adaptation_auc": adaptation_auc,
        "normalized_adaptation_auc": normalized_adaptation_auc,
        "recovery_steps": recovery_steps,
        "recovery_targets": recovery_targets,
        "stability_gaps": stability_gaps,
        "recurrence_retention_delta": recurring_final - initial_final,
        "recurrence_retention_ratio": retention_ratio,
        "worst_window_mean_reward": _rolling_worst_mean(reward, config.worst_window),
        "safety_cost_sum": float(np.sum(safety_cost)),
        "safety_cost_rate": float(np.mean(safety_cost)),
        "safety_violation_count": int(np.sum(safety_violation)),
        "safety_violation_rate": float(np.mean(safety_violation)),
        "mean_action_probability": float(np.mean(arrays["action_probability"])),
    }


def _stratum_to_dict(value: Any) -> dict[str, object]:
    return {
        "count": int(value.count),
        "fraction": float(value.fraction),
        "mean_advantage": float(value.mean_advantage),
        "mean_action_surprisal": float(value.mean_action_surprisal),
        "mean_delight": float(value.mean_delight),
        "mean_gate_rate": float(value.mean_gate_rate),
        "mean_gated_advantage": float(value.mean_gated_advantage),
    }


def _dg_diagnostics(
    trace: Mapping[str, object],
    mode: PolicyGradientMode,
    config: DelightfulPolicyGradientDevelopmentConfig,
) -> dict[str, object]:
    arrays = _trace_arrays_from_dict(trace)
    result = discrete_delightful_policy_gradient(
        jnp.asarray(arrays["behavior_log_probability"]),
        jnp.asarray(arrays["advantage"]),
        DelightfulPolicyGradientConfig(
            mode=mode,
            temperature=config.delight_temperature,
            actor_trace_lambda=0.0,
            diagnostics_epsilon=config.diagnostics_epsilon,
        ),
    )
    rare = arrays["action_probability"] < config.rare_probability_cutoff
    strata = stratify_delight_outcomes(result, jnp.asarray(rare))
    delight = np.asarray(result.delight)
    weights = np.asarray(result.sample_weights)
    positive = delight > 0.0
    negative = delight < 0.0

    def masked_mean(mask: np.ndarray) -> float:
        return 0.0 if not np.any(mask) else float(np.mean(weights[mask]))

    return {
        "effective_sample_size": float(result.diagnostics.effective_sample_size),
        "effective_sample_fraction": float(result.diagnostics.effective_sample_fraction),
        "mean_gate_weight": float(result.diagnostics.mean_gate_rate),
        "positive_delight_rate": float(np.mean(positive)),
        "negative_delight_rate": float(np.mean(negative)),
        "positive_delight_gate_rate": masked_mean(positive),
        "negative_delight_gate_rate": masked_mean(negative),
        "rare_probability_cutoff": config.rare_probability_cutoff,
        "strata": {
            "common_success": _stratum_to_dict(strata.common_success),
            "common_failure": _stratum_to_dict(strata.common_failure),
            "rare_success": _stratum_to_dict(strata.rare_success),
            "rare_failure": _stratum_to_dict(strata.rare_failure),
        },
        "common_advantage_mean": float(strata.common_advantage_mean),
        "rare_advantage_mean": float(strata.rare_advantage_mean),
        "common_gated_advantage_mean": float(strata.common_gated_advantage_mean),
        "rare_gated_advantage_mean": float(strata.rare_gated_advantage_mean),
        "common_advantage_variance": float(strata.common_advantage_variance),
        "rare_advantage_variance": float(strata.rare_advantage_variance),
    }


def _gambling_diagnostics(
    trace: Mapping[str, object],
    benchmark: BenchmarkName,
) -> dict[str, object]:
    if benchmark != "contextual_heteroskedastic_gambling":
        return {
            "applicable": False,
            "risky_action_count": None,
            "safe_action_count": None,
            "lucky_risky_count": None,
            "failing_risky_count": None,
            "risky_reward_mean": None,
            "safe_reward_mean": None,
            "risky_reward_variance": None,
            "safe_reward_variance": None,
        }
    arrays = _trace_arrays_from_dict(trace)
    risky = arrays["action"] == 1
    safe = ~risky
    lucky = risky & (arrays["advantage"] > 0.0)
    failing = risky & (arrays["advantage"] < 0.0)

    def conditional(moment: str, mask: np.ndarray) -> float:
        if not np.any(mask):
            return 0.0
        values = arrays["reward"][mask].astype(np.float64)
        return float(np.mean(values) if moment == "mean" else np.var(values))

    return {
        "applicable": True,
        "risky_action_count": int(np.sum(risky)),
        "safe_action_count": int(np.sum(safe)),
        "lucky_risky_count": int(np.sum(lucky)),
        "failing_risky_count": int(np.sum(failing)),
        "risky_reward_mean": conditional("mean", risky),
        "safe_reward_mean": conditional("mean", safe),
        "risky_reward_variance": conditional("variance", risky),
        "safe_reward_variance": conditional("variance", safe),
    }


def _operation_accounting(trace: Mapping[str, object]) -> dict[str, object]:
    arrays = _trace_arrays_from_dict(trace)
    steps = int(arrays["step"].size)
    committed = int(np.sum(arrays["applied"]))
    return {
        "accounting_scope": (
            "logical algorithm events; excludes JAX validation/compiler operations, "
            "FLOPs, energy, and wall clock"
        ),
        "wall_clock_measured": False,
        "environment_transitions": steps,
        "environment_rng_draws": steps,
        "executed_action_samples": steps,
        "pending_action_samples": 1,
        "total_action_samples": steps + 1,
        "policy_forward_samples": steps + 1,
        "critic_predictions": 2 * steps,
        "actor_gate_evaluations": steps,
        "actor_update_attempts": steps,
        "actor_update_commits": committed,
        "critic_update_attempts": steps,
        "critic_update_commits": committed,
        "average_reward_update_attempts": steps,
        "average_reward_update_commits": committed,
        "rejected_transactions": steps - committed,
        "external_route_declarations": 3 * steps,
        "final_transition_count": int(arrays["transition_count_after"][-1]),
    }


def _resource_accounting(
    agent: DelightfulActorCriticAgent,
    trace: _DevelopmentTraceArrays,
    benchmark: BenchmarkName,
) -> dict[str, object]:
    budget = agent.resource_budget
    trace_nbytes = sum(int(leaf.nbytes) for leaf in jax.tree_util.tree_leaves(trace))
    environment_state_nbytes = 4 if benchmark == "contextual_heteroskedastic_gambling" else 8
    return {
        "persistent_state_nbytes": budget.state_nbytes,
        "trainable_float32_scalars": budget.trainable_float32_scalars,
        "actor_scalar_updates_per_transition": budget.actor_scalar_updates_per_transition,
        "critic_scalar_updates_per_transition": budget.critic_scalar_updates_per_transition,
        "average_reward_scalar_updates_per_transition": (
            budget.average_reward_scalar_updates_per_transition
        ),
        "max_update_component_magnitude": budget.max_update_component_magnitude,
        "max_actor_update_l2_norm": budget.max_actor_update_l2_norm,
        "max_critic_update_l2_norm": budget.max_critic_update_l2_norm,
        "replay_capacity": budget.replay_capacity,
        "environment_state_nbytes": environment_state_nbytes,
        "logical_trace_nbytes": trace_nbytes,
        "peak_owned_logical_nbytes": (
            budget.state_nbytes + environment_state_nbytes + trace_nbytes
        ),
        "accounting_scope": (
            "persistent learner state + environment state + retained trace arrays; "
            "excludes inputs, Python objects, alignment, compiler workspace, and device runtime"
        ),
    }


def _kondo_selection_accounting(
    trace: Mapping[str, object],
    config: DelightfulPolicyGradientDevelopmentConfig,
    benchmark: BenchmarkName,
    mode: PolicyGradientMode,
) -> dict[str, object]:
    """Account the paper's Kondo selection as a measured counterfactual.

    The selection rule is the deterministic maximum-probability decision of
    the paper's ``Bernoulli(sigmoid((delight - price) / temperature))`` gate:
    a step is selected exactly when ``delight > price``.  Each decision uses
    only that step's forward-pass quantities, so a real gate could have made
    it before the backward pass.  Only the actor channel is Kondo-gateable —
    paper-defined delight is a policy-gradient sample statistic, and critic
    and average-reward losses stay outside the policy-gradient helper — so
    their compute is never counted as skippable.  In this development harness
    no backward work was actually skipped: every logged update ran in full,
    and the skipped columns are counterfactual bookkeeping, not savings.
    """
    arrays = _trace_arrays_from_dict(trace)
    delight = arrays["delight"].astype(np.float64)
    steps = int(delight.size)
    price = _KONDO_SELECTION_PRICE
    temperature = float(config.delight_temperature)
    selected = delight > price
    scaled = (delight - price) / temperature
    with np.errstate(over="ignore", invalid="ignore"):
        gate_probability = np.where(
            scaled >= 0.0,
            1.0 / (1.0 + np.exp(-scaled)),
            np.exp(scaled) / (1.0 + np.exp(scaled)),
        )
    selected_steps = int(np.sum(selected))
    selection_rate_by_phase = [
        float(np.mean(selected[phase * config.phase_length : (phase + 1) * config.phase_length]))
        for phase in range(len(REGIME_SCHEDULE))
    ]
    dimension = 2 if benchmark == "contextual_heteroskedastic_gambling" else _RIVER_CONFIG.n_states
    budget = DelightfulActorCriticAgent(config.agent_config(dimension, mode)).resource_budget
    scalar_updates_per_transition = {
        "actor": int(budget.actor_scalar_updates_per_transition),
        "critic": int(budget.critic_scalar_updates_per_transition),
        "average_reward": int(budget.average_reward_scalar_updates_per_transition),
    }
    gateable = {"actor": True, "critic": False, "average_reward": False}
    rationales = {
        "actor": (
            "paper-defined delight is a per-sample policy-gradient statistic; "
            "the Kondo gate decides whether the actor backward pass runs"
        ),
        "critic": (
            "critic prediction losses remain outside the policy-gradient "
            "helper and are never Kondo-gated"
        ),
        "average_reward": (
            "average-reward tracking is not a policy-gradient loss and is "
            "never Kondo-gated"
        ),
    }
    channels: list[dict[str, object]] = []
    for name in ("actor", "critic", "average_reward"):
        executed = steps * scalar_updates_per_transition[name]
        counterfactual_selected = (
            selected_steps * scalar_updates_per_transition[name] if gateable[name] else executed
        )
        channels.append(
            {
                "channel": name,
                "kondo_gateable": gateable[name],
                "gate_rationale": rationales[name],
                "scalar_updates_per_transition": scalar_updates_per_transition[name],
                "executed_scalar_updates": executed,
                "counterfactual_selected_scalar_updates": counterfactual_selected,
                "counterfactual_skipped_scalar_updates": executed - counterfactual_selected,
            }
        )
    executed_total = sum(cast(int, channel["executed_scalar_updates"]) for channel in channels)
    selected_total = sum(
        cast(int, channel["counterfactual_selected_scalar_updates"]) for channel in channels
    )
    skipped_total = executed_total - selected_total
    return {
        "actual_compute_gating": False,
        "accounting_semantics": _KONDO_ACCOUNTING_SEMANTICS,
        "selection_policy": {
            "signal": "paper-defined delight = advantage * action_surprisal",
            "rule": "deterministic_price_threshold",
            "price": price,
            "temperature": temperature,
            "selected_iff": "delight > price",
            "bernoulli_gate_probability": "sigmoid((delight - price) / temperature)",
            "decision_point": (
                "after the forward pass of each step, before the backward "
                "pass a real Kondo gate would skip"
            ),
        },
        "total_steps": steps,
        "selected_steps": selected_steps,
        "selection_rate": selected_steps / steps,
        "expected_bernoulli_selection_rate": float(np.mean(gate_probability)),
        "selection_rate_by_phase": selection_rate_by_phase,
        "channels": channels,
        "executed_scalar_updates_total": executed_total,
        "counterfactual_selected_scalar_updates_total": selected_total,
        "counterfactual_skipped_scalar_updates_total": skipped_total,
        "counterfactual_skipped_fraction": skipped_total / executed_total,
    }


def _state_fingerprint(
    agent: DelightfulActorCriticAgent,
    state: DelightfulActorCriticState,
) -> str:
    checkpoint = agent.checkpoint_payload(state)
    return _canonical_sha256(checkpoint["state"])


@functools.lru_cache(maxsize=128)
def _expected_initial_fingerprint_and_trace(
    config: DelightfulPolicyGradientDevelopmentConfig,
    benchmark: BenchmarkName,
    seed: int,
    mode: PolicyGradientMode,
) -> tuple[str, dict[str, object]]:
    """Replay one declared life so validation binds learner dynamics, not just summaries."""
    agent_key, environment_key = _scenario_keys(seed, benchmark)
    if benchmark == "contextual_heteroskedastic_gambling":
        initial, _, trace = _run_contextual_arrays(config, mode, agent_key, environment_key)
        observation_dim = 2
    else:
        initial, _, trace = _run_riverswim_arrays(config, mode, agent_key, environment_key)
        observation_dim = _RIVER_CONFIG.n_states
    agent = DelightfulActorCriticAgent(config.agent_config(observation_dim, mode))
    return _state_fingerprint(agent, initial), _trace_to_dict(trace)


def _run_record(
    config: DelightfulPolicyGradientDevelopmentConfig,
    benchmark: BenchmarkName,
    seed: int,
    mode: PolicyGradientMode,
) -> dict[str, object]:
    agent_key, environment_key = _scenario_keys(seed, benchmark)
    if benchmark == "contextual_heteroskedastic_gambling":
        initial, final, trace = _run_contextual_arrays(config, mode, agent_key, environment_key)
        observation_dim = 2
    else:
        initial, final, trace = _run_riverswim_arrays(config, mode, agent_key, environment_key)
        observation_dim = _RIVER_CONFIG.n_states
    np.asarray(jax.device_get(final.transition_count))
    agent = DelightfulActorCriticAgent(config.agent_config(observation_dim, mode))
    trace_payload = _trace_to_dict(trace)
    return {
        "benchmark": benchmark,
        "seed": seed,
        "mode": mode,
        "agent_config": agent.config.to_config(),
        "initial_state_sha256": _state_fingerprint(agent, initial),
        "trace": trace_payload,
        "performance": _performance_metrics(trace_payload, config),
        "dg_diagnostics": _dg_diagnostics(trace_payload, mode, config),
        "heteroskedastic_gambling": _gambling_diagnostics(trace_payload, benchmark),
        "kondo_selection_accounting": _kondo_selection_accounting(
            trace_payload, config, benchmark, mode
        ),
        "operations": _operation_accounting(trace_payload),
        "resources": _resource_accounting(agent, trace, benchmark),
    }


_PAIRED_METRICS = (
    "mean_prequential_reward",
    "recurrence_retention_delta",
    "safety_violation_rate",
)
_PAIRED_DG_METRICS = (
    "effective_sample_fraction",
    "mean_gate_weight",
)


def _paired_comparisons(
    runs: Sequence[Mapping[str, object]],
    config: DelightfulPolicyGradientDevelopmentConfig,
) -> list[dict[str, object]]:
    by_key = {(run["benchmark"], run["seed"], run["mode"]): run for run in runs}
    comparisons: list[dict[str, object]] = []
    for benchmark in BENCHMARK_ORDER:
        seed_records: list[dict[str, object]] = []
        for seed in config.seeds:
            ordinary = by_key[(benchmark, seed, "ordinary_pg")]
            delightful = by_key[(benchmark, seed, "delightful_pg")]
            ordinary_trace = cast(Mapping[str, object], ordinary["trace"])
            delightful_trace = cast(Mapping[str, object], delightful["trace"])
            ordinary_performance = cast(Mapping[str, object], ordinary["performance"])
            delightful_performance = cast(Mapping[str, object], delightful["performance"])
            ordinary_dg = cast(Mapping[str, object], ordinary["dg_diagnostics"])
            delightful_dg = cast(Mapping[str, object], delightful["dg_diagnostics"])
            differences: dict[str, float] = {}
            for metric in _PAIRED_METRICS:
                differences[metric] = float(cast(float, delightful_performance[metric])) - float(
                    cast(float, ordinary_performance[metric])
                )
            for metric in _PAIRED_DG_METRICS:
                differences[metric] = float(cast(float, delightful_dg[metric])) - float(
                    cast(float, ordinary_dg[metric])
                )
            seed_records.append(
                {
                    "seed": seed,
                    "initialization_matched": (
                        ordinary["initial_state_sha256"] == delightful["initial_state_sha256"]
                    ),
                    "initial_action_matched": (
                        cast(list[int], ordinary_trace["action"])[0]
                        == cast(list[int], delightful_trace["action"])[0]
                    ),
                    "action_rng_common": (
                        ordinary_trace["action_rng_words"] == delightful_trace["action_rng_words"]
                    ),
                    "environment_rng_common": (
                        ordinary_trace["environment_rng_words"]
                        == delightful_trace["environment_rng_words"]
                    ),
                    "logical_operations_matched": (
                        ordinary["operations"] == delightful["operations"]
                    ),
                    "persistent_resources_matched": (
                        cast(Mapping[str, object], ordinary["resources"])["persistent_state_nbytes"]
                        == cast(Mapping[str, object], delightful["resources"])[
                            "persistent_state_nbytes"
                        ]
                    ),
                    "delightful_minus_ordinary": differences,
                }
            )
        mean_differences = {
            metric: float(
                np.mean(
                    [
                        cast(Mapping[str, float], record["delightful_minus_ordinary"])[metric]
                        for record in seed_records
                    ]
                )
            )
            for metric in (*_PAIRED_METRICS, *_PAIRED_DG_METRICS)
        }
        comparisons.append(
            {
                "benchmark": benchmark,
                "seed_count": len(config.seeds),
                "common_random_numbers_required": True,
                "seed_pairs": seed_records,
                "mean_delightful_minus_ordinary": mean_differences,
                "verdict": None,
            }
        )
    return comparisons


def run_delightful_policy_gradient_development(
    config: DelightfulPolicyGradientDevelopmentConfig | None = None,
) -> dict[str, object]:
    """Run both matched modes in plan order and return a deterministic report."""
    protocol = config or DelightfulPolicyGradientDevelopmentConfig()
    runs: list[dict[str, object]] = []
    for benchmark in BENCHMARK_ORDER:
        for seed in protocol.seeds:
            for mode in POLICY_GRADIENT_MODES:
                runs.append(_run_record(protocol, benchmark, seed, mode))
    report_without_digest: dict[str, object] = {
        "schema_version": DELIGHTFUL_POLICY_GRADIENT_DEVELOPMENT_SCHEMA,
        "development_only": DEVELOPMENT_ONLY,
        "scientific_promotion_allowed": SCIENTIFIC_PROMOTION_ALLOWED,
        "kondo_implemented": KONDO_IMPLEMENTED,
        "kondo_selection_accounting": KONDO_SELECTION_ACCOUNTING,
        "claim_scope": _CLAIM_SCOPE,
        "excluded_claims": list(_EXCLUDED_CLAIMS),
        "interpretation": _INTERPRETATION,
        "benchmark_order": list(BENCHMARK_ORDER),
        "protocol": protocol.to_config(),
        "runs": runs,
        "paired_comparisons": _paired_comparisons(runs, protocol),
    }
    report_without_digest["report_digest"] = {
        "algorithm": "sha256",
        "scope": "$ excluding $.report_digest",
        "sha256": _canonical_sha256(report_without_digest),
    }
    validation = validate_delightful_policy_gradient_development_report(report_without_digest)
    if not validation.valid:
        raise RuntimeError(
            "generated Delightful Policy Gradient development report is invalid: "
            + "; ".join(validation.errors)
        )
    return report_without_digest


def delightful_policy_gradient_development_report_json(
    report: Mapping[str, object],
) -> str:
    """Serialize deterministic pretty JSON while rejecting non-finite values."""
    return (
        json.dumps(
            report,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


@dataclasses.dataclass(frozen=True)
class DelightfulPolicyGradientDevelopmentValidation:
    """Fail-closed structural and calculation validation result."""

    valid: bool
    errors: tuple[str, ...]


_TRACE_FIELDS = {
    "step",
    "phase_index",
    "regime_id",
    "state_index",
    "next_state_index",
    "action",
    "behavior_policy",
    "target_policy",
    "action_probability",
    "behavior_log_probability",
    "reward",
    "expected_reward",
    "oracle_reward",
    "advantage",
    "delight",
    "gate_weight",
    "safety_cost",
    "safety_violation",
    "applied",
    "transition_count_after",
    "action_rng_words",
    "environment_rng_words",
}
_RUN_FIELDS = {
    "benchmark",
    "seed",
    "mode",
    "agent_config",
    "initial_state_sha256",
    "trace",
    "performance",
    "dg_diagnostics",
    "heteroskedastic_gambling",
    "kondo_selection_accounting",
    "operations",
    "resources",
}


def _json_tree_finite(value: object) -> bool:
    if value is None or isinstance(value, bool | str | int):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list | tuple):
        return all(_json_tree_finite(item) for item in value)
    if isinstance(value, Mapping):
        return all(isinstance(key, str) and _json_tree_finite(item) for key, item in value.items())
    return False


def _mapping_close(left: object, right: object) -> bool:
    if isinstance(right, bool):
        return isinstance(left, bool) and left is right
    if right is None:
        return left is None
    if isinstance(right, int):
        return isinstance(left, int) and not isinstance(left, bool) and left == right
    if isinstance(right, float):
        return (
            isinstance(left, float)
            and math.isfinite(left)
            and math.isclose(left, right, rel_tol=1.0e-6, abs_tol=1.0e-7)
        )
    if isinstance(right, str):
        return isinstance(left, str) and left == right
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(
            _mapping_close(left[key], right[key]) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _mapping_close(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def _strict_trace_arrays(
    trace: Mapping[str, object],
    steps: int,
) -> tuple[dict[str, np.ndarray] | None, list[str]]:
    errors: list[str] = []
    if set(trace) != _TRACE_FIELDS:
        errors.append("trace fields do not match the v2 schema")
        return None, errors
    shapes = {
        **{
            name: (steps,)
            for name in _TRACE_FIELDS
            if name
            not in {
                "behavior_policy",
                "target_policy",
                "action_rng_words",
                "environment_rng_words",
            }
        },
        "behavior_policy": (steps, 2),
        "target_policy": (steps, 2),
        "action_rng_words": (steps, 2),
        "environment_rng_words": (steps, 2),
    }
    integer_fields = {
        "step",
        "phase_index",
        "regime_id",
        "state_index",
        "next_state_index",
        "action",
        "transition_count_after",
        "action_rng_words",
        "environment_rng_words",
    }
    boolean_fields = {"safety_violation", "applied"}
    for name, shape in shapes.items():
        try:
            raw = np.asarray(trace[name], dtype=object)
        except (TypeError, ValueError):
            errors.append(f"trace.{name} is not a rectangular JSON array")
            continue
        if raw.shape != shape:
            errors.append(f"trace.{name} must have shape {shape}")
            continue
        if name in boolean_fields:
            valid_scalars = all(isinstance(item, bool) for item in raw.flat)
        elif name in integer_fields:
            valid_scalars = all(
                isinstance(item, int) and not isinstance(item, bool) for item in raw.flat
            )
        else:
            valid_scalars = all(
                isinstance(item, float) and math.isfinite(item) for item in raw.flat
            )
        if not valid_scalars:
            errors.append(f"trace.{name} contains invalid JSON scalar types")
        elif name in {"action_rng_words", "environment_rng_words"} and any(
            not 0 <= cast(int, item) <= _UINT32_MAX for item in raw.flat
        ):
            errors.append(f"trace.{name} contains words outside uint32 range")
        elif name in integer_fields - {"action_rng_words", "environment_rng_words"} and any(
            not _INT32_MIN <= cast(int, item) <= _INT32_MAX for item in raw.flat
        ):
            errors.append(f"trace.{name} contains values outside int32 range")
    if errors:
        return None, errors
    try:
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            arrays = _trace_arrays_from_dict(trace)
    except (TypeError, ValueError, KeyError, OverflowError):
        errors.append("trace arrays cannot be reconstructed")
        return None, errors
    if any(
        np.issubdtype(array.dtype, np.floating) and not np.all(np.isfinite(array))
        for array in arrays.values()
    ):
        errors.append("trace contains non-finite numeric values")
        return None, errors
    return arrays, errors


def _expected_environment_keys(seed: int, benchmark: BenchmarkName, steps: int) -> np.ndarray:
    _, environment_key = _scenario_keys(seed, benchmark)
    keys = jax.vmap(lambda step: jr.fold_in(environment_key, step))(
        jnp.arange(steps, dtype=jnp.int32)
    )
    return np.asarray(jr.key_data(keys), dtype=np.uint32)


def _validate_policy_and_dg_semantics(
    arrays: Mapping[str, np.ndarray],
    mode: PolicyGradientMode,
    config: DelightfulPolicyGradientDevelopmentConfig,
    errors: list[str],
) -> None:
    behavior = arrays["behavior_policy"]
    target = arrays["target_policy"]
    actions = arrays["action"]
    if not np.array_equal(behavior, target):
        errors.append("behavior and target policies are not exactly identical")
    if (
        np.any(behavior <= 0.0)
        or not np.allclose(np.sum(behavior, axis=1), 1.0, atol=1.0e-6, rtol=1.0e-6)
        or np.any((actions < 0) | (actions >= 2))
    ):
        errors.append("categorical policy/action contract is invalid")
        return
    selected = behavior[np.arange(actions.size), actions]
    if not np.allclose(arrays["action_probability"], selected, atol=1.0e-7, rtol=1.0e-6):
        errors.append("selected action probability is not prequential policy probability")
    if not np.allclose(
        arrays["behavior_log_probability"],
        np.log(selected),
        atol=1.0e-6,
        rtol=1.0e-6,
    ):
        errors.append("selected behavior log-probability is inconsistent")
    try:
        action_keys = jr.wrap_key_data(
            jnp.asarray(arrays["action_rng_words"], dtype=jnp.uint32),
            impl="threefry2x32",
        )
        sampled = jax.vmap(
            lambda key, policy: jr.categorical(key, jnp.log(policy)).astype(jnp.int32)
        )(action_keys, jnp.asarray(behavior))
        if not np.array_equal(np.asarray(sampled), actions):
            errors.append("action is not reproduced by its policy and common RNG key")
    except (TypeError, ValueError):
        errors.append("action RNG words cannot reconstruct typed Threefry keys")
    expected_pg = discrete_delightful_policy_gradient(
        jnp.asarray(arrays["behavior_log_probability"]),
        jnp.asarray(arrays["advantage"]),
        DelightfulPolicyGradientConfig(
            mode=mode,
            temperature=config.delight_temperature,
            actor_trace_lambda=0.0,
            diagnostics_epsilon=config.diagnostics_epsilon,
        ),
    )
    if not np.allclose(
        arrays["delight"], np.asarray(expected_pg.delight), atol=1.0e-6, rtol=1.0e-6
    ):
        errors.append(
            "trace paper-specific DG delight does not match advantage times action surprisal"
        )
    if not np.allclose(
        arrays["gate_weight"],
        np.asarray(expected_pg.sample_weights),
        atol=1.0e-6,
        rtol=1.0e-6,
    ):
        errors.append("trace gate weights do not match the declared policy-gradient mode")


def _validate_contextual_environment(
    arrays: Mapping[str, np.ndarray],
    config: DelightfulPolicyGradientDevelopmentConfig,
    errors: list[str],
) -> None:
    steps = config.total_steps
    expected_state = np.arange(steps, dtype=np.int32) % 2
    expected_next = (np.arange(steps, dtype=np.int32) + 1) % 2
    if not np.array_equal(arrays["state_index"], expected_state) or not np.array_equal(
        arrays["next_state_index"], expected_next
    ):
        errors.append("contextual state schedule is not the fixed alternating stream")
    regimes = arrays["regime_id"]
    actions = arrays["action"]
    means = np.asarray(_CONTEXT_MEANS, dtype=np.float32)
    deviations = np.asarray(_CONTEXT_STANDARD_DEVIATIONS, dtype=np.float32)
    expected_means = means[regimes, expected_state, actions]
    expected_oracle = np.max(means[regimes, expected_state], axis=1)
    keys = jr.wrap_key_data(
        jnp.asarray(arrays["environment_rng_words"], dtype=jnp.uint32),
        impl="threefry2x32",
    )
    draws = np.asarray(jax.vmap(lambda key: jr.normal(key, (), dtype=jnp.float32))(keys))
    expected_rewards = expected_means + deviations[actions] * draws
    floor = config.contextual_safety_reward_floor
    expected_cost = np.maximum(floor - expected_rewards, 0.0)
    expected_violation = expected_rewards < floor
    for name, actual, expected in (
        ("expected_reward", arrays["expected_reward"], expected_means),
        ("reward", arrays["reward"], expected_rewards),
        ("oracle_reward", arrays["oracle_reward"], expected_oracle),
        ("safety_cost", arrays["safety_cost"], expected_cost),
    ):
        if not np.allclose(actual, expected, atol=1.0e-6, rtol=1.0e-6):
            errors.append(f"contextual {name} does not reconstruct from the CRN stream")
    if not np.array_equal(arrays["safety_violation"], expected_violation):
        errors.append("contextual safety violations do not reconstruct")


def _validate_riverswim_environment(
    arrays: Mapping[str, np.ndarray],
    errors: list[str],
) -> None:
    states = arrays["state_index"]
    next_states = arrays["next_state_index"]
    actions = arrays["action"]
    if states[0] != _RIVER_CONFIG.initial_state or not np.array_equal(states[1:], next_states[:-1]):
        errors.append("RiverSwim trace is not one uninterrupted continuing state chain")
    environment = RiverSwimMDP(_RIVER_CONFIG)
    transition = environment.transition_tensor
    logits = np.where(
        transition > 0.0,
        np.log(np.clip(transition, 1.0e-30, 1.0)),
        -np.inf,
    ).astype(np.float32)
    selected_logits = logits[actions, states]
    keys = jr.wrap_key_data(
        jnp.asarray(arrays["environment_rng_words"], dtype=jnp.uint32),
        impl="threefry2x32",
    )
    reproduced_next = np.asarray(
        jax.vmap(lambda key, row: jr.categorical(key, row).astype(jnp.int32))(
            keys, jnp.asarray(selected_logits)
        )
    )
    if not np.array_equal(next_states, reproduced_next):
        errors.append("RiverSwim transitions do not reconstruct from action and CRN keys")
    regimes = arrays["regime_id"]
    reward_tables = np.stack((_river_reward_table(0), _river_reward_table(1)))
    expected_reward = reward_tables[regimes, states, actions]
    oracle = np.asarray(_river_oracle_rewards(), dtype=np.float32)[regimes]
    pressure = (states == _RIVER_CONFIG.n_states - 1) & (actions == RIGHT_ACTION)
    for name, actual, expected in (
        ("reward", arrays["reward"], expected_reward),
        ("expected_reward", arrays["expected_reward"], expected_reward),
        ("oracle_reward", arrays["oracle_reward"], oracle),
        ("safety_cost", arrays["safety_cost"], pressure.astype(np.float32)),
    ):
        if not np.allclose(actual, expected, atol=1.0e-7, rtol=1.0e-7):
            errors.append(f"RiverSwim {name} does not reconstruct")
    if not np.array_equal(arrays["safety_violation"], pressure):
        errors.append("RiverSwim safety boundary-pressure events do not reconstruct")


def _expected_resources_from_arrays(
    config: DelightfulPolicyGradientDevelopmentConfig,
    benchmark: BenchmarkName,
    mode: PolicyGradientMode,
    arrays: Mapping[str, np.ndarray],
) -> dict[str, object]:
    dimension = 2 if benchmark == "contextual_heteroskedastic_gambling" else 6
    budget = DelightfulActorCriticAgent(config.agent_config(dimension, mode)).resource_budget
    trace_nbytes = sum(int(array.nbytes) for array in arrays.values())
    environment_state_nbytes = 4 if benchmark == "contextual_heteroskedastic_gambling" else 8
    return {
        "persistent_state_nbytes": budget.state_nbytes,
        "trainable_float32_scalars": budget.trainable_float32_scalars,
        "actor_scalar_updates_per_transition": budget.actor_scalar_updates_per_transition,
        "critic_scalar_updates_per_transition": budget.critic_scalar_updates_per_transition,
        "average_reward_scalar_updates_per_transition": (
            budget.average_reward_scalar_updates_per_transition
        ),
        "max_update_component_magnitude": budget.max_update_component_magnitude,
        "max_actor_update_l2_norm": budget.max_actor_update_l2_norm,
        "max_critic_update_l2_norm": budget.max_critic_update_l2_norm,
        "replay_capacity": budget.replay_capacity,
        "environment_state_nbytes": environment_state_nbytes,
        "logical_trace_nbytes": trace_nbytes,
        "peak_owned_logical_nbytes": (
            budget.state_nbytes + environment_state_nbytes + trace_nbytes
        ),
        "accounting_scope": (
            "persistent learner state + environment state + retained trace arrays; "
            "excludes inputs, Python objects, alignment, compiler workspace, and device runtime"
        ),
    }


def _validate_run(
    run: Mapping[str, object],
    config: DelightfulPolicyGradientDevelopmentConfig,
    expected_benchmark: BenchmarkName,
    expected_seed: int,
    expected_mode: PolicyGradientMode,
) -> list[str]:
    prefix = f"{expected_benchmark}/seed={expected_seed}/{expected_mode}"
    errors: list[str] = []
    if set(run) != _RUN_FIELDS:
        errors.append(f"{prefix}: run fields do not match the v2 schema")
        return errors
    if (
        run.get("benchmark") != expected_benchmark
        or run.get("seed") != expected_seed
        or run.get("mode") != expected_mode
    ):
        errors.append(f"{prefix}: run identity/order is invalid")
    agent_payload = run.get("agent_config")
    if not isinstance(agent_payload, Mapping):
        errors.append(f"{prefix}: agent_config must be an object")
    else:
        try:
            restored = DelightfulActorCriticConfig.from_config(agent_payload)
            dimension = 2 if expected_benchmark == BENCHMARK_ORDER[0] else 6
            expected_agent_config = config.agent_config(dimension, expected_mode)
            if restored != expected_agent_config or not _strict_json_equal(
                agent_payload,
                expected_agent_config.to_config(),
            ):
                errors.append(f"{prefix}: agent config is not the exact matched config")
        except (TypeError, ValueError) as exc:
            errors.append(f"{prefix}: invalid agent config: {exc}")
    fingerprint = run.get("initial_state_sha256")
    if (
        not isinstance(fingerprint, str)
        or len(fingerprint) != 64
        or any(character not in "0123456789abcdef" for character in fingerprint)
    ):
        errors.append(f"{prefix}: initial state digest is invalid")
    trace = run.get("trace")
    if not isinstance(trace, Mapping):
        errors.append(f"{prefix}: trace must be an object")
        return errors
    arrays, trace_errors = _strict_trace_arrays(trace, config.total_steps)
    errors.extend(f"{prefix}: {error}" for error in trace_errors)
    if arrays is None:
        return errors
    expected_fingerprint, expected_trace = _expected_initial_fingerprint_and_trace(
        config,
        expected_benchmark,
        expected_seed,
        expected_mode,
    )
    if fingerprint != expected_fingerprint:
        errors.append(f"{prefix}: initial state does not reconstruct from the declared seed")
    if not _strict_json_equal(dict(trace), expected_trace):
        errors.append(f"{prefix}: trace does not reconstruct from the declared actor-critic life")
    expected_action_keys = np.asarray(expected_trace["action_rng_words"], dtype=np.uint32)
    if not np.array_equal(arrays["action_rng_words"], expected_action_keys):
        errors.append(f"{prefix}: action RNG schedule is not canonical")
    expected_steps = np.arange(config.total_steps, dtype=np.int32)
    expected_phases = expected_steps // config.phase_length
    expected_regimes = np.where(expected_phases == 1, 1, 0).astype(np.int32)
    if not np.array_equal(arrays["step"], expected_steps):
        errors.append(f"{prefix}: step indices are not contiguous")
    if not np.array_equal(arrays["phase_index"], expected_phases):
        errors.append(f"{prefix}: phase indices do not match the A/B/A schedule")
    if not np.array_equal(arrays["regime_id"], expected_regimes):
        errors.append(f"{prefix}: regime identities do not match A/B/A")
    if not np.array_equal(arrays["transition_count_after"], expected_steps + 1):
        errors.append(f"{prefix}: update counters violate action-before-outcome order")
    if not np.all(arrays["applied"]):
        errors.append(f"{prefix}: matched development life contains rejected updates")
    expected_environment_keys = _expected_environment_keys(
        expected_seed, expected_benchmark, config.total_steps
    )
    if not np.array_equal(arrays["environment_rng_words"], expected_environment_keys):
        errors.append(f"{prefix}: environment RNG schedule is not canonical")
    _validate_policy_and_dg_semantics(arrays, expected_mode, config, errors)
    action_range_valid = bool(np.all((arrays["action"] >= 0) & (arrays["action"] < 2)))
    if expected_benchmark == "contextual_heteroskedastic_gambling":
        state_range_valid = bool(
            np.all((arrays["state_index"] >= 0) & (arrays["state_index"] < 2))
            and np.all((arrays["next_state_index"] >= 0) & (arrays["next_state_index"] < 2))
        )
        if action_range_valid and state_range_valid:
            _validate_contextual_environment(arrays, config, errors)
        else:
            errors.append(f"{prefix}: contextual action/state index is out of range")
    else:
        state_range_valid = bool(
            np.all((arrays["state_index"] >= 0) & (arrays["state_index"] < 6))
            and np.all((arrays["next_state_index"] >= 0) & (arrays["next_state_index"] < 6))
        )
        if action_range_valid and state_range_valid:
            _validate_riverswim_environment(arrays, errors)
        else:
            errors.append(f"{prefix}: RiverSwim action/state index is out of range")
    expected_performance = _performance_metrics(trace, config)
    if not _mapping_close(run.get("performance"), expected_performance):
        errors.append(f"{prefix}: performance metrics do not reconstruct from trace")
    expected_dg = _dg_diagnostics(trace, expected_mode, config)
    if not _mapping_close(run.get("dg_diagnostics"), expected_dg):
        errors.append(f"{prefix}: DG diagnostics/strata do not reconstruct from trace")
    expected_gambling = _gambling_diagnostics(trace, expected_benchmark)
    if not _mapping_close(run.get("heteroskedastic_gambling"), expected_gambling):
        errors.append(f"{prefix}: gambling diagnostics do not reconstruct from trace")
    expected_operations = _operation_accounting(trace)
    if not _strict_json_equal(run.get("operations"), expected_operations):
        errors.append(f"{prefix}: operation accounting does not reconstruct")
    expected_resources = _expected_resources_from_arrays(
        config,
        expected_benchmark,
        expected_mode,
        arrays,
    )
    if not _mapping_close(run.get("resources"), expected_resources):
        errors.append(f"{prefix}: logical resource accounting does not reconstruct")
    kondo_payload = run.get("kondo_selection_accounting")
    if not isinstance(kondo_payload, Mapping):
        errors.append(f"{prefix}: kondo_selection_accounting must be an object")
    else:
        if kondo_payload.get("actual_compute_gating") is not False:
            errors.append(
                f"{prefix}: Kondo selection accounting must not claim actual compute gating"
            )
        expected_kondo = _kondo_selection_accounting(
            trace,
            config,
            expected_benchmark,
            expected_mode,
        )
        if not _mapping_close(kondo_payload, expected_kondo):
            errors.append(
                f"{prefix}: Kondo selection accounting does not reconstruct from the trace"
            )
    return errors


def validate_delightful_policy_gradient_development_report(
    report: Mapping[str, object],
) -> DelightfulPolicyGradientDevelopmentValidation:
    """Strictly validate one deterministic, nonpromoting development report."""
    errors: list[str] = []
    expected_top = {
        "schema_version",
        "development_only",
        "scientific_promotion_allowed",
        "kondo_implemented",
        "kondo_selection_accounting",
        "claim_scope",
        "excluded_claims",
        "interpretation",
        "benchmark_order",
        "protocol",
        "runs",
        "paired_comparisons",
        "report_digest",
    }
    if set(report) != expected_top:
        errors.append("top-level report fields do not match the v2 schema")
    if report.get("schema_version") != DELIGHTFUL_POLICY_GRADIENT_DEVELOPMENT_SCHEMA:
        errors.append("development report schema is unsupported")
    if report.get("development_only") is not True:
        errors.append("report must remain development-only")
    if report.get("scientific_promotion_allowed") is not False:
        errors.append("report must forbid scientific promotion")
    if report.get("kondo_implemented") is not False:
        errors.append(
            "Kondo compute gating is outside this development harness; only "
            "counterfactual selection accounting is reported"
        )
    if report.get("kondo_selection_accounting") is not True:
        errors.append("Kondo selection accounting declaration was changed")
    if report.get("claim_scope") != _CLAIM_SCOPE:
        errors.append("claim scope is not the fixed Delightful Policy Gradient scope")
    if report.get("excluded_claims") != list(_EXCLUDED_CLAIMS):
        errors.append("excluded claims were changed")
    if report.get("interpretation") != _INTERPRETATION:
        errors.append("nonpromoting interpretation was changed")
    if report.get("benchmark_order") != list(BENCHMARK_ORDER):
        errors.append("benchmark order does not match the continual-control plan")
    if not _json_tree_finite(report):
        errors.append("report contains a non-JSON or non-finite value")

    protocol_payload = report.get("protocol")
    config: DelightfulPolicyGradientDevelopmentConfig | None = None
    if not isinstance(protocol_payload, Mapping):
        errors.append("protocol must be an object")
    else:
        try:
            config = DelightfulPolicyGradientDevelopmentConfig.from_config(protocol_payload)
        except (TypeError, ValueError) as exc:
            errors.append(f"protocol is invalid: {exc}")

    runs_payload = report.get("runs")
    runs: list[Mapping[str, object]] = []
    if not isinstance(runs_payload, list) or not all(
        isinstance(run, Mapping) for run in runs_payload
    ):
        errors.append("runs must be a JSON array of objects")
    else:
        runs = cast(list[Mapping[str, object]], runs_payload)
    if config is not None:
        expected_identities = [
            (benchmark, seed, mode)
            for benchmark in BENCHMARK_ORDER
            for seed in config.seeds
            for mode in POLICY_GRADIENT_MODES
        ]
        if len(runs) != len(expected_identities):
            errors.append("run count does not cover the exact matched benchmark matrix")
        else:
            run_error_start = len(errors)
            for run, (benchmark, seed, mode) in zip(runs, expected_identities, strict=True):
                errors.extend(_validate_run(run, config, benchmark, seed, mode))

            if len(errors) == run_error_start:
                for benchmark in BENCHMARK_ORDER:
                    for seed in config.seeds:
                        ordinary = next(
                            run
                            for run in runs
                            if run.get("benchmark") == benchmark
                            and run.get("seed") == seed
                            and run.get("mode") == "ordinary_pg"
                        )
                        delightful = next(
                            run
                            for run in runs
                            if run.get("benchmark") == benchmark
                            and run.get("seed") == seed
                            and run.get("mode") == "delightful_pg"
                        )
                        pair_prefix = f"{benchmark}/seed={seed}"
                        if ordinary.get("initial_state_sha256") != delightful.get(
                            "initial_state_sha256"
                        ):
                            errors.append(f"{pair_prefix}: initialization is not matched")
                        ordinary_trace = cast(Mapping[str, object], ordinary["trace"])
                        delightful_trace = cast(Mapping[str, object], delightful["trace"])
                        if ordinary_trace.get("action_rng_words") != delightful_trace.get(
                            "action_rng_words"
                        ):
                            errors.append(f"{pair_prefix}: action CRN schedule diverged")
                        if ordinary_trace.get("environment_rng_words") != delightful_trace.get(
                            "environment_rng_words"
                        ):
                            errors.append(f"{pair_prefix}: environment CRN schedule diverged")
                        if (
                            cast(list[object], ordinary_trace["action"])[0]
                            != cast(list[object], delightful_trace["action"])[0]
                        ):
                            errors.append(
                                f"{pair_prefix}: identical initial policy sampled differently"
                            )

                expected_comparisons = _paired_comparisons(runs, config)
                if not _mapping_close(report.get("paired_comparisons"), expected_comparisons):
                    errors.append("paired comparisons do not reconstruct from matched runs")

    digest = report.get("report_digest")
    if not isinstance(digest, Mapping) or set(digest) != {
        "algorithm",
        "scope",
        "sha256",
    }:
        errors.append("report_digest fields do not match the v2 schema")
    else:
        without_digest = dict(report)
        without_digest.pop("report_digest", None)
        expected_sha = _canonical_sha256(without_digest)
        if (
            digest.get("algorithm") != "sha256"
            or digest.get("scope") != "$ excluding $.report_digest"
            or digest.get("sha256") != expected_sha
        ):
            errors.append("report digest does not bind the complete development record")
    return DelightfulPolicyGradientDevelopmentValidation(
        valid=not errors,
        errors=tuple(errors),
    )


__all__ = [
    "BENCHMARK_ORDER",
    "DEVELOPMENT_ONLY",
    "DELIGHTFUL_POLICY_GRADIENT_DEVELOPMENT_SCHEMA",
    "KONDO_IMPLEMENTED",
    "KONDO_SELECTION_ACCOUNTING",
    "POLICY_GRADIENT_MODES",
    "SCIENTIFIC_PROMOTION_ALLOWED",
    "DelightfulPolicyGradientDevelopmentConfig",
    "DelightfulPolicyGradientDevelopmentValidation",
    "delightful_policy_gradient_development_report_json",
    "run_delightful_policy_gradient_development",
    "validate_delightful_policy_gradient_development_report",
]
