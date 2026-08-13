# mypy: disable-error-code="attr-defined,call-arg"
"""Fail-closed discrete actor-critic integration for Delightful Policy Gradient.

This module is the stateful boundary for the Delightful Policy Gradient (DG)
experiment — §5.4 of ``CONTINUAL_AGENT_IMPLEMENTATION_PLAN.md``, after
"Delightful Policy Gradient" (arXiv:2603.14608).  It deliberately supports
only a continuing, on-policy categorical actor.  The ordinary and delightful
modes use the same state, critic, reward-rate baseline, random stream, and action
sampler; the only difference is the detached coefficient returned by
``discrete_delightful_policy_gradient``.

The actor has no eligibility trace: a current-sample delight gate must not
multiply a trace containing historical score gradients.  The critic may keep an
ordinary differential TD trace, and neither that trace, the critic update, nor
the average-reward update is multiplied by the paper-defined delight gate.
Safety, model, and representation learners are external to this isolated core.
On an accepted transition, their explicit
routes are reported with unit weight whenever the caller declares them
available; this module never silently pretends to update those learners.

Every transition is an atomic immutable transaction.  Static shape/dtype errors
raise before tracing.  Dynamic non-finite values, corrupt state, exhausted update
capacity, or invalid candidates return the original state byte-for-byte and no
usable actor/critic signal.  Passing tests establish mechanism contracts only, not
policy quality or evidence for a research claim.
"""

from __future__ import annotations

import dataclasses
import functools
import math
from collections.abc import Mapping
from typing import Any, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import numpy.typing as npt
from jax import Array
from jaxtyping import Bool, Float, Int

from alberta_framework.core.delight import (
    DelightfulPolicyGradientConfig,
    PolicyGradientMode,
    discrete_delightful_policy_gradient,
)

_CHECKPOINT_SCHEMA = "alberta.delightful_actor_critic.v1"
_INT32_MAX = 2**31 - 1
_FLOAT32_TINY = float(np.finfo(np.float32).tiny)


def _positive_int(value: Any, *, name: str, maximum: int = _INT32_MAX) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0 or value > maximum:
        raise ValueError(f"{name} must be an integer in 1..{maximum}")


def _finite_float32(
    value: Any,
    *,
    name: str,
    allow_zero: bool,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    """Validate a scalar that must survive as a normal finite float32 value."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a real scalar")
    parsed = float(value)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        narrowed = float(np.float32(parsed))
    if not math.isfinite(parsed) or not math.isfinite(narrowed):
        raise ValueError(f"{name} must be finite in float32")
    if parsed != 0.0 and abs(narrowed) < _FLOAT32_TINY:
        raise ValueError(f"{name} must not underflow to a float32 subnormal")
    if not allow_zero and narrowed == 0.0:
        raise ValueError(f"{name} must be nonzero")
    if minimum is not None and narrowed < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if maximum is not None and narrowed > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return narrowed


def _strict_config_payload(
    config: Mapping[str, Any],
    cls: type[Any],
) -> dict[str, Any]:
    payload = dict(config)
    expected = {field.name for field in dataclasses.fields(cls)} | {"type"}
    if set(payload) != expected:
        raise ValueError("config fields do not match the serialized schema")
    type_name = payload.pop("type")
    if type_name != cls.__name__:
        raise ValueError(f"unexpected config type: {type_name!r}")
    return payload


def _array_contract(
    value: Any,
    *,
    name: str,
    shape: tuple[int, ...] | None,
    dtype: Any,
) -> Array:
    source_dtype = getattr(value, "dtype", None)
    if source_dtype is None:
        try:
            source_dtype = np.asarray(value).dtype
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be an array-like numeric value") from exc
    try:
        normalized_source_dtype = np.dtype(source_dtype)
    except TypeError as exc:
        raise ValueError(f"{name} has an unsupported source dtype") from exc
    expected_dtype = np.dtype(dtype)
    if normalized_source_dtype != expected_dtype:
        raise ValueError(f"{name} must have dtype {expected_dtype}")
    array = jnp.asarray(value)
    if array.dtype != jnp.dtype(dtype):
        raise ValueError(f"{name} must have dtype {expected_dtype}")
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape} and dtype {expected_dtype}")
    return array


def _prng_key_contract(value: Any, *, name: str) -> Array:
    array = jnp.asarray(value)
    if array.shape != () or not jnp.issubdtype(array.dtype, jax.dtypes.prng_key):
        raise ValueError(f"{name} must be one typed Threefry JAX PRNG key")
    key_data = jr.key_data(array)
    if key_data.shape != (2,) or key_data.dtype != jnp.uint32:
        raise ValueError(f"{name} must be one typed Threefry JAX PRNG key")
    return array


def _tree_select(condition: Array, yes: Any, no: Any) -> Any:
    return jax.tree_util.tree_map(lambda x, y: jnp.where(condition, x, y), yes, no)


def _safe_masked_mean(values: Array, mask: Array) -> Array:
    mask_f = mask.astype(jnp.float32)
    count = jnp.sum(mask_f)
    return jnp.sum(jnp.where(mask, values, 0.0)) / jnp.maximum(count, 1.0)


@dataclasses.dataclass(frozen=True)
class DelightfulActorCriticConfig:
    """Static contract for the isolated discrete actor-critic experiment."""

    observation_dim: int
    n_actions: int
    mode: PolicyGradientMode = "delightful_pg"
    actor_step_size: float = 0.01
    critic_step_size: float = 0.05
    average_reward_step_size: float = 0.01
    actor_trace_lambda: float = 0.0
    critic_trace_lambda: float = 0.0
    policy_temperature: float = 1.0
    delight_temperature: float = 1.0
    diagnostics_epsilon: float = 1.0e-8
    max_input_magnitude: float = 1_000.0
    max_parameter_magnitude: float = 1_000.0
    max_update_component_magnitude: float = 1.0
    max_updates: int = _INT32_MAX

    def __post_init__(self) -> None:
        """Reject ambiguous modes and controls that are unsafe after narrowing."""
        _positive_int(self.observation_dim, name="observation_dim")
        _positive_int(self.n_actions, name="n_actions")
        _positive_int(self.max_updates, name="max_updates")
        if self.mode not in ("ordinary_pg", "delightful_pg"):
            raise ValueError("mode must be 'ordinary_pg' or 'delightful_pg'")
        object.__setattr__(
            self,
            "actor_step_size",
            _finite_float32(
                self.actor_step_size,
                name="actor_step_size",
                allow_zero=True,
                minimum=0.0,
            ),
        )
        object.__setattr__(
            self,
            "critic_step_size",
            _finite_float32(
                self.critic_step_size,
                name="critic_step_size",
                allow_zero=True,
                minimum=0.0,
            ),
        )
        object.__setattr__(
            self,
            "average_reward_step_size",
            _finite_float32(
                self.average_reward_step_size,
                name="average_reward_step_size",
                allow_zero=True,
                minimum=0.0,
            ),
        )
        object.__setattr__(
            self,
            "actor_trace_lambda",
            _finite_float32(
                self.actor_trace_lambda,
                name="actor_trace_lambda",
                allow_zero=True,
                minimum=0.0,
                maximum=0.0,
            ),
        )
        if self.actor_trace_lambda != 0.0:
            raise ValueError("actor_trace_lambda must be exactly zero")
        object.__setattr__(
            self,
            "critic_trace_lambda",
            _finite_float32(
                self.critic_trace_lambda,
                name="critic_trace_lambda",
                allow_zero=True,
                minimum=0.0,
                maximum=1.0,
            ),
        )
        object.__setattr__(
            self,
            "policy_temperature",
            _finite_float32(
                self.policy_temperature,
                name="policy_temperature",
                allow_zero=False,
                minimum=_FLOAT32_TINY,
            ),
        )
        object.__setattr__(
            self,
            "delight_temperature",
            _finite_float32(
                self.delight_temperature,
                name="delight_temperature",
                allow_zero=False,
                minimum=_FLOAT32_TINY,
            ),
        )
        object.__setattr__(
            self,
            "diagnostics_epsilon",
            _finite_float32(
                self.diagnostics_epsilon,
                name="diagnostics_epsilon",
                allow_zero=False,
                minimum=_FLOAT32_TINY,
            ),
        )
        object.__setattr__(
            self,
            "max_input_magnitude",
            _finite_float32(
                self.max_input_magnitude,
                name="max_input_magnitude",
                allow_zero=False,
                minimum=_FLOAT32_TINY,
            ),
        )
        object.__setattr__(
            self,
            "max_parameter_magnitude",
            _finite_float32(
                self.max_parameter_magnitude,
                name="max_parameter_magnitude",
                allow_zero=False,
                minimum=_FLOAT32_TINY,
            ),
        )
        object.__setattr__(
            self,
            "max_update_component_magnitude",
            _finite_float32(
                self.max_update_component_magnitude,
                name="max_update_component_magnitude",
                allow_zero=False,
                minimum=_FLOAT32_TINY,
            ),
        )

    def to_config(self) -> dict[str, Any]:
        """Return a strict JSON-compatible construction record."""
        payload = dataclasses.asdict(self)
        payload["type"] = type(self).__name__
        return payload

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> DelightfulActorCriticConfig:
        """Strictly reconstruct from :meth:`to_config`."""
        return cls(**_strict_config_payload(config, cls))


@chex.dataclass(frozen=True)
class DelightfulChannelAvailability:
    """External channels this core must never gate with paper-defined delight."""

    safety: Bool[Array, ""]
    model: Bool[Array, ""]
    representation: Bool[Array, ""]


@chex.dataclass(frozen=True)
class DelightfulPolicySample:
    """Exact on-policy behavior/target record for one categorical action."""

    available: Bool[Array, ""]
    action: Int[Array, ""]
    target_policy: Float[Array, " n_actions"]
    behavior_policy: Float[Array, " n_actions"]
    target_probability: Float[Array, ""]
    behavior_probability: Float[Array, ""]
    target_log_probability: Float[Array, ""]
    behavior_log_probability: Float[Array, ""]


@chex.dataclass(frozen=True)
class DelightfulActorCriticState:
    """Fixed-shape actor, ungated baseline state, sample, RNG, and counters."""

    actor_weights: Float[Array, "n_actions observation_dim"]
    actor_bias: Float[Array, " n_actions"]
    critic_weights: Float[Array, " observation_dim"]
    critic_bias: Float[Array, ""]
    critic_trace_weights: Float[Array, " observation_dim"]
    critic_trace_bias: Float[Array, ""]
    average_reward: Float[Array, ""]
    last_observation: Float[Array, " observation_dim"]
    last_sample: DelightfulPolicySample
    rng_key: Array
    transition_count: Int[Array, ""]
    actor_update_count: Int[Array, ""]
    critic_update_count: Int[Array, ""]
    average_reward_update_count: Int[Array, ""]


@chex.dataclass(frozen=True)
class DelightfulChannelRouting:
    """Detached per-channel update weights and explicit availability."""

    actor_available: Bool[Array, ""]
    critic_available: Bool[Array, ""]
    average_reward_available: Bool[Array, ""]
    safety_available: Bool[Array, ""]
    model_available: Bool[Array, ""]
    representation_available: Bool[Array, ""]
    actor_weight: Float[Array, ""]
    critic_weight: Float[Array, ""]
    average_reward_weight: Float[Array, ""]
    safety_weight: Float[Array, ""]
    model_weight: Float[Array, ""]
    representation_weight: Float[Array, ""]


@chex.dataclass(frozen=True)
class DelightfulActorCriticDiagnostics:
    """Finite actor signals, gate summaries, and fail-closed verdicts.

    ``delight``-named fields are paper-defined actor-sample quantities and
    strata, never candidate-update safety-audit verdicts.
    """

    state_valid: Bool[Array, ""]
    input_valid: Bool[Array, ""]
    capacity_available: Bool[Array, ""]
    sample_available: Bool[Array, ""]
    signals_finite: Bool[Array, ""]
    candidate_state_valid: Bool[Array, ""]
    applied: Bool[Array, ""]
    rejected: Bool[Array, ""]
    selected_log_probability: Float[Array, ""]
    action_surprisal: Float[Array, ""]
    advantage: Float[Array, ""]
    delight: Float[Array, ""]
    gate_weight: Float[Array, ""]
    effective_sample_size: Float[Array, ""]
    positive_delight_rate: Float[Array, ""]
    negative_delight_rate: Float[Array, ""]
    positive_delight_gate_rate: Float[Array, ""]
    negative_delight_gate_rate: Float[Array, ""]
    actor_update_norm: Float[Array, ""]
    critic_update_norm: Float[Array, ""]
    routing: DelightfulChannelRouting


@chex.dataclass(frozen=True)
class DelightfulActorCriticStartResult:
    """Atomic first on-policy sample transaction."""

    state: DelightfulActorCriticState
    action: Int[Array, ""]
    target_policy: Float[Array, " n_actions"]
    behavior_policy: Float[Array, " n_actions"]
    target_log_probability: Float[Array, ""]
    behavior_log_probability: Float[Array, ""]
    state_valid: Bool[Array, ""]
    input_valid: Bool[Array, ""]
    applied: Bool[Array, ""]


@chex.dataclass(frozen=True)
class DelightfulActorCriticUpdateResult:
    """One predict-before-update transition and next on-policy action."""

    state: DelightfulActorCriticState
    action: Int[Array, ""]
    target_policy: Float[Array, " n_actions"]
    behavior_policy: Float[Array, " n_actions"]
    value: Float[Array, ""]
    next_value: Float[Array, ""]
    average_reward: Float[Array, ""]
    diagnostics: DelightfulActorCriticDiagnostics


@chex.dataclass(frozen=True)
class DelightfulActorCriticBatchDiagnostics:
    """Finite aggregate diagnostics over accepted scan transitions.

    ``delight``-named fields summarize exact paper-defined DG actor samples.
    """

    attempted_count: Int[Array, ""]
    accepted_count: Int[Array, ""]
    effective_sample_size: Float[Array, ""]
    effective_sample_fraction: Float[Array, ""]
    positive_delight_rate: Float[Array, ""]
    negative_delight_rate: Float[Array, ""]
    positive_delight_gate_rate: Float[Array, ""]
    negative_delight_gate_rate: Float[Array, ""]
    mean_gate_rate: Float[Array, ""]
    mean_selected_log_probability: Float[Array, ""]
    mean_action_surprisal: Float[Array, ""]
    mean_advantage: Float[Array, ""]
    mean_delight: Float[Array, ""]


@chex.dataclass(frozen=True)
class DelightfulActorCriticArrayResult:
    """Final state, bounded per-step outputs, and aggregate scan diagnostics.

    ``delights`` contains exact paper-defined DG actor-sample values, not
    literal candidate-gradient verdicts.
    """

    state: DelightfulActorCriticState
    actions: Int[Array, " num_steps"]
    target_policies: Float[Array, "num_steps n_actions"]
    behavior_policies: Float[Array, "num_steps n_actions"]
    selected_log_probabilities: Float[Array, " num_steps"]
    action_surprisals: Float[Array, " num_steps"]
    advantages: Float[Array, " num_steps"]
    delights: Float[Array, " num_steps"]
    gate_weights: Float[Array, " num_steps"]
    applied: Bool[Array, " num_steps"]
    diagnostics: DelightfulActorCriticBatchDiagnostics


@chex.dataclass(frozen=True)
class _DelightfulActorCriticScanOutput:
    """Only the per-transition arrays retained by the public scan result."""

    action: Int[Array, ""]
    target_policy: Float[Array, " n_actions"]
    behavior_policy: Float[Array, " n_actions"]
    selected_log_probability: Float[Array, ""]
    action_surprisal: Float[Array, ""]
    advantage: Float[Array, ""]
    delight: Float[Array, ""]
    gate_weight: Float[Array, ""]
    applied: Bool[Array, ""]


@dataclasses.dataclass(frozen=True)
class DelightfulActorCriticResourceBudget:
    """Exact persistent state and bounded logical update accounting."""

    observation_dim: int
    n_actions: int
    trainable_float32_scalars: int
    persistent_float32_scalars: int
    persistent_int32_scalars: int
    persistent_uint32_scalars: int
    persistent_bool_scalars: int
    state_nbytes: int
    max_transitions: int
    max_actor_updates_per_transition: int
    max_critic_updates_per_transition: int
    max_average_reward_updates_per_transition: int
    actor_scalar_updates_per_transition: int
    critic_scalar_updates_per_transition: int
    average_reward_scalar_updates_per_transition: int
    max_update_component_magnitude: float
    max_actor_update_l2_norm: float
    max_critic_update_l2_norm: float
    max_average_reward_update_abs: float
    max_external_routes_per_transition: int
    scan_output_nbytes_per_transition: int
    batch_diagnostics_nbytes: int
    replay_capacity: int

    def to_dict(self) -> dict[str, int | float]:
        """Return a JSON-compatible exact resource record."""
        return dataclasses.asdict(self)

    def scan_result_nbytes(self, num_steps: int) -> int:
        """Return logical public result bytes, excluding inputs and compiler workspace."""
        _positive_int(num_steps, name="num_steps")
        return (
            self.state_nbytes
            + self.batch_diagnostics_nbytes
            + num_steps * self.scan_output_nbytes_per_transition
        )


class DelightfulActorCriticAgent:
    """Linear actor-critic with an optional detached paper-defined delight gate."""

    def __init__(self, config: DelightfulActorCriticConfig):
        self._config = config
        self._policy_gradient_config = DelightfulPolicyGradientConfig(
            mode=config.mode,
            temperature=config.delight_temperature,
            actor_trace_lambda=config.actor_trace_lambda,
            diagnostics_epsilon=config.diagnostics_epsilon,
        )

    @property
    def config(self) -> DelightfulActorCriticConfig:
        """Static construction contract."""
        return self._config

    @property
    def resource_budget(self) -> DelightfulActorCriticResourceBudget:
        """Return exact state bytes and per-transition update bounds."""
        cfg = self._config
        trainable = cfg.n_actions * cfg.observation_dim + cfg.n_actions + cfg.observation_dim + 1
        # Actor/critic parameters; critic traces; reward rate; last observation;
        # two policies and four selected probability/log-probability scalars.
        persistent_f32 = (
            trainable + cfg.observation_dim + 1 + 1 + cfg.observation_dim + 2 * cfg.n_actions + 4
        )
        persistent_i32 = 5  # sampled action plus four accepted-update counters
        persistent_u32 = 2  # one Threefry PRNG key
        persistent_bool = 1  # sample availability
        nbytes = 4 * (persistent_f32 + persistent_i32 + persistent_u32) + persistent_bool
        actor_scalars = cfg.n_actions * (cfg.observation_dim + 1)
        critic_scalars = cfg.observation_dim + 1
        component_bound = cfg.max_update_component_magnitude
        return DelightfulActorCriticResourceBudget(
            observation_dim=cfg.observation_dim,
            n_actions=cfg.n_actions,
            trainable_float32_scalars=trainable,
            persistent_float32_scalars=persistent_f32,
            persistent_int32_scalars=persistent_i32,
            persistent_uint32_scalars=persistent_u32,
            persistent_bool_scalars=persistent_bool,
            state_nbytes=nbytes,
            max_transitions=cfg.max_updates,
            max_actor_updates_per_transition=1,
            max_critic_updates_per_transition=1,
            max_average_reward_updates_per_transition=1,
            actor_scalar_updates_per_transition=actor_scalars,
            critic_scalar_updates_per_transition=critic_scalars,
            average_reward_scalar_updates_per_transition=1,
            max_update_component_magnitude=component_bound,
            max_actor_update_l2_norm=component_bound * math.sqrt(actor_scalars),
            max_critic_update_l2_norm=component_bound * math.sqrt(critic_scalars),
            max_average_reward_update_abs=component_bound,
            max_external_routes_per_transition=3,
            scan_output_nbytes_per_transition=8 * cfg.n_actions + 25,
            batch_diagnostics_nbytes=52,
            replay_capacity=0,
        )

    def to_config(self) -> dict[str, Any]:
        """Serialize the complete agent construction."""
        return {"type": type(self).__name__, "config": self._config.to_config()}

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> DelightfulActorCriticAgent:
        """Strictly reconstruct an agent from :meth:`to_config`."""
        payload = dict(config)
        if set(payload) != {"type", "config"}:
            raise ValueError("agent config fields do not match the serialized schema")
        if payload["type"] != cls.__name__:
            raise ValueError(f"unexpected agent type: {payload['type']!r}")
        nested = payload["config"]
        if not isinstance(nested, Mapping):
            raise ValueError("agent config must contain a config mapping")
        return cls(DelightfulActorCriticConfig.from_config(nested))

    def _initial_sample(self) -> DelightfulPolicySample:
        uniform = jnp.full(
            (self._config.n_actions,),
            1.0 / self._config.n_actions,
            dtype=jnp.float32,
        )
        zero = jnp.asarray(0.0, dtype=jnp.float32)
        return DelightfulPolicySample(
            available=jnp.asarray(False, dtype=jnp.bool_),
            action=jnp.asarray(-1, dtype=jnp.int32),
            target_policy=uniform,
            behavior_policy=uniform,
            target_probability=zero,
            behavior_probability=zero,
            target_log_probability=zero,
            behavior_log_probability=zero,
        )

    def init(self, key: Array) -> DelightfulActorCriticState:
        """Initialize deterministic zero parameters from an explicit PRNG key."""
        checked_key = _prng_key_contract(key, name="key")
        cfg = self._config
        zeros_obs = jnp.zeros((cfg.observation_dim,), dtype=jnp.float32)
        zero_i = jnp.asarray(0, dtype=jnp.int32)
        return DelightfulActorCriticState(
            actor_weights=jnp.zeros((cfg.n_actions, cfg.observation_dim), dtype=jnp.float32),
            actor_bias=jnp.zeros((cfg.n_actions,), dtype=jnp.float32),
            critic_weights=zeros_obs,
            critic_bias=jnp.asarray(0.0, dtype=jnp.float32),
            critic_trace_weights=zeros_obs,
            critic_trace_bias=jnp.asarray(0.0, dtype=jnp.float32),
            average_reward=jnp.asarray(0.0, dtype=jnp.float32),
            last_observation=zeros_obs,
            last_sample=self._initial_sample(),
            rng_key=checked_key,
            transition_count=zero_i,
            actor_update_count=zero_i,
            critic_update_count=zero_i,
            average_reward_update_count=zero_i,
        )

    def _validate_sample_static_contract(self, sample: DelightfulPolicySample) -> None:
        if not isinstance(sample, DelightfulPolicySample):
            raise TypeError("state.last_sample must be a DelightfulPolicySample")
        cfg = self._config
        fields = {
            "available": (sample.available, (), jnp.bool_),
            "action": (sample.action, (), jnp.int32),
            "target_policy": (sample.target_policy, (cfg.n_actions,), jnp.float32),
            "behavior_policy": (sample.behavior_policy, (cfg.n_actions,), jnp.float32),
            "target_probability": (sample.target_probability, (), jnp.float32),
            "behavior_probability": (sample.behavior_probability, (), jnp.float32),
            "target_log_probability": (sample.target_log_probability, (), jnp.float32),
            "behavior_log_probability": (sample.behavior_log_probability, (), jnp.float32),
        }
        for name, (value, shape, dtype) in fields.items():
            _array_contract(value, name=f"state.last_sample.{name}", shape=shape, dtype=dtype)

    def _validate_state_static_contract(self, state: DelightfulActorCriticState) -> None:
        if not isinstance(state, DelightfulActorCriticState):
            raise TypeError("state must be a DelightfulActorCriticState")
        cfg = self._config
        fields = {
            "actor_weights": (
                state.actor_weights,
                (cfg.n_actions, cfg.observation_dim),
                jnp.float32,
            ),
            "actor_bias": (state.actor_bias, (cfg.n_actions,), jnp.float32),
            "critic_weights": (state.critic_weights, (cfg.observation_dim,), jnp.float32),
            "critic_bias": (state.critic_bias, (), jnp.float32),
            "critic_trace_weights": (
                state.critic_trace_weights,
                (cfg.observation_dim,),
                jnp.float32,
            ),
            "critic_trace_bias": (state.critic_trace_bias, (), jnp.float32),
            "average_reward": (state.average_reward, (), jnp.float32),
            "last_observation": (
                state.last_observation,
                (cfg.observation_dim,),
                jnp.float32,
            ),
            "transition_count": (state.transition_count, (), jnp.int32),
            "actor_update_count": (state.actor_update_count, (), jnp.int32),
            "critic_update_count": (state.critic_update_count, (), jnp.int32),
            "average_reward_update_count": (
                state.average_reward_update_count,
                (),
                jnp.int32,
            ),
        }
        for name, (value, shape, dtype) in fields.items():
            _array_contract(value, name=f"state.{name}", shape=shape, dtype=dtype)
        self._validate_sample_static_contract(state.last_sample)
        _prng_key_contract(state.rng_key, name="state.rng_key")

    def _policy_unchecked(
        self,
        state: DelightfulActorCriticState,
        observation: Array,
    ) -> tuple[Array, Array]:
        logits = state.actor_weights @ observation + state.actor_bias
        scaled_logits = logits / jnp.asarray(self._config.policy_temperature, dtype=jnp.float32)
        log_policy = jax.nn.log_softmax(scaled_logits)
        return jnp.exp(log_policy), log_policy

    def _sample_unchecked(
        self,
        state: DelightfulActorCriticState,
        observation: Array,
    ) -> tuple[DelightfulPolicySample, Array]:
        next_key, sample_key = jr.split(state.rng_key)
        target, log_target = self._policy_unchecked(state, observation)
        action = jr.categorical(sample_key, log_target).astype(jnp.int32)
        probability = target[action]
        log_probability = log_target[action]
        sample = DelightfulPolicySample(
            available=jnp.asarray(True, dtype=jnp.bool_),
            action=action,
            target_policy=target,
            behavior_policy=target,
            target_probability=probability,
            behavior_probability=probability,
            target_log_probability=log_probability,
            behavior_log_probability=log_probability,
        )
        return sample, next_key

    def _sample_valid(
        self,
        state: DelightfulActorCriticState,
        sample: DelightfulPolicySample,
    ) -> Array:
        cfg = self._config
        target, log_target = self._policy_unchecked(state, state.last_observation)
        action = sample.action
        in_range = (action >= 0) & (action < cfg.n_actions)
        safe_action = jnp.clip(action, 0, cfg.n_actions - 1)
        selected_probability = target[safe_action]
        selected_log_probability = log_target[safe_action]
        available_valid = (
            in_range
            & jnp.all(jnp.isfinite(sample.target_policy))
            & jnp.all(sample.target_policy > 0.0)
            & jnp.array_equal(sample.target_policy, target)
            & jnp.array_equal(sample.behavior_policy, sample.target_policy)
            & jnp.isfinite(sample.target_probability)
            & (sample.target_probability > 0.0)
            & jnp.isfinite(sample.target_log_probability)
            & jnp.array_equal(sample.behavior_probability, sample.target_probability)
            & jnp.array_equal(
                sample.behavior_log_probability,
                sample.target_log_probability,
            )
            & jnp.array_equal(sample.target_probability, selected_probability)
            & jnp.array_equal(
                sample.target_log_probability,
                selected_log_probability,
            )
        )
        initial = self._initial_sample()
        unavailable_valid = (
            (sample.action == -1)
            & jnp.array_equal(sample.target_policy, initial.target_policy)
            & jnp.array_equal(sample.behavior_policy, initial.behavior_policy)
            & (sample.target_probability == 0.0)
            & (sample.behavior_probability == 0.0)
            & (sample.target_log_probability == 0.0)
            & (sample.behavior_log_probability == 0.0)
            & (state.transition_count == 0)
        )
        return jnp.where(sample.available, available_valid, unavailable_valid)

    def _state_valid(self, state: DelightfulActorCriticState) -> Array:
        cfg = self._config
        parameter_bound = jnp.asarray(cfg.max_parameter_magnitude, dtype=jnp.float32)
        input_bound = jnp.asarray(cfg.max_input_magnitude, dtype=jnp.float32)
        counts = jnp.stack(
            (
                state.transition_count,
                state.actor_update_count,
                state.critic_update_count,
                state.average_reward_update_count,
            )
        )
        return (
            jnp.all(jnp.isfinite(state.actor_weights))
            & jnp.all(jnp.abs(state.actor_weights) <= parameter_bound)
            & jnp.all(jnp.isfinite(state.actor_bias))
            & jnp.all(jnp.abs(state.actor_bias) <= parameter_bound)
            & jnp.all(jnp.isfinite(state.critic_weights))
            & jnp.all(jnp.abs(state.critic_weights) <= parameter_bound)
            & jnp.isfinite(state.critic_bias)
            & (jnp.abs(state.critic_bias) <= parameter_bound)
            & jnp.all(jnp.isfinite(state.critic_trace_weights))
            & jnp.all(jnp.abs(state.critic_trace_weights) <= parameter_bound)
            & jnp.isfinite(state.critic_trace_bias)
            & (jnp.abs(state.critic_trace_bias) <= parameter_bound)
            & jnp.isfinite(state.average_reward)
            & (jnp.abs(state.average_reward) <= parameter_bound)
            & jnp.all(jnp.isfinite(state.last_observation))
            & jnp.all(jnp.abs(state.last_observation) <= input_bound)
            & jnp.all(counts >= 0)
            & jnp.all(counts <= cfg.max_updates)
            & jnp.all(counts == state.transition_count)
            & self._sample_valid(state, state.last_sample)
        )

    def state_valid(self, state: DelightfulActorCriticState) -> Array:
        """Return the dynamic state-validity verdict after static validation."""
        self._validate_state_static_contract(state)
        return self._state_valid(state)

    def policy(
        self,
        state: DelightfulActorCriticState,
        observation: Array,
    ) -> Float[Array, " n_actions"]:
        """Return a finite target policy, or uniform when inputs are invalid."""
        self._validate_state_static_contract(state)
        obs = _array_contract(
            observation,
            name="observation",
            shape=(self._config.observation_dim,),
            dtype=jnp.float32,
        )
        return cast(Array, self._policy_compiled(state, obs))

    @functools.partial(jax.jit, static_argnums=(0,))
    def _policy_compiled(
        self,
        state: DelightfulActorCriticState,
        observation: Array,
    ) -> Array:
        obs = observation
        valid = (
            self._state_valid(state)
            & jnp.all(jnp.isfinite(obs))
            & jnp.all(jnp.abs(obs) <= self._config.max_input_magnitude)
        )
        policy, log_policy = self._policy_unchecked(state, obs)
        policy_valid = (
            jnp.all(jnp.isfinite(policy))
            & jnp.all(policy > 0.0)
            & jnp.all(jnp.isfinite(log_policy))
        )
        valid = valid & policy_valid
        uniform = jnp.full_like(policy, 1.0 / self._config.n_actions)
        return jnp.where(valid, policy, uniform)

    def value(
        self,
        state: DelightfulActorCriticState,
        observation: Array,
    ) -> Float[Array, ""]:
        """Return the scalar differential critic prediction."""
        self._validate_state_static_contract(state)
        obs = _array_contract(
            observation,
            name="observation",
            shape=(self._config.observation_dim,),
            dtype=jnp.float32,
        )
        return cast(Array, self._value_compiled(state, obs))

    @functools.partial(jax.jit, static_argnums=(0,))
    def _value_compiled(
        self,
        state: DelightfulActorCriticState,
        observation: Array,
    ) -> Array:
        obs = observation
        prediction = jnp.dot(state.critic_weights, obs) + state.critic_bias
        valid = (
            self._state_valid(state)
            & jnp.all(jnp.isfinite(obs))
            & jnp.all(jnp.abs(obs) <= self._config.max_input_magnitude)
            & jnp.isfinite(prediction)
        )
        return jnp.where(valid, prediction, 0.0)

    def _zero_start_result(
        self,
        state: DelightfulActorCriticState,
        *,
        state_valid: Array,
        input_valid: Array,
    ) -> DelightfulActorCriticStartResult:
        initial = self._initial_sample()
        return DelightfulActorCriticStartResult(
            state=state,
            action=jnp.asarray(-1, dtype=jnp.int32),
            target_policy=initial.target_policy,
            behavior_policy=initial.behavior_policy,
            target_log_probability=jnp.asarray(0.0, dtype=jnp.float32),
            behavior_log_probability=jnp.asarray(0.0, dtype=jnp.float32),
            state_valid=state_valid,
            input_valid=input_valid,
            applied=jnp.asarray(False, dtype=jnp.bool_),
        )

    def start(
        self,
        state: DelightfulActorCriticState,
        observation: Array,
    ) -> DelightfulActorCriticStartResult:
        """Atomically sample and store the first explicitly on-policy action."""
        self._validate_state_static_contract(state)
        obs = _array_contract(
            observation,
            name="observation",
            shape=(self._config.observation_dim,),
            dtype=jnp.float32,
        )
        return cast(DelightfulActorCriticStartResult, self._start_compiled(state, obs))

    @functools.partial(jax.jit, static_argnums=(0,))
    def _start_compiled(
        self,
        state: DelightfulActorCriticState,
        observation: Array,
    ) -> DelightfulActorCriticStartResult:
        obs = observation
        state_valid = self._state_valid(state)
        input_valid = jnp.all(jnp.isfinite(obs)) & jnp.all(
            jnp.abs(obs) <= self._config.max_input_magnitude
        )
        pristine = ~state.last_sample.available & (state.transition_count == 0)
        can_start = state_valid & input_valid & pristine

        def do_start(_: None) -> DelightfulActorCriticStartResult:
            sample, next_key = self._sample_unchecked(state, obs)
            candidate = state.replace(
                last_observation=obs,
                last_sample=sample,
                rng_key=next_key,
            )
            candidate_valid = self._state_valid(candidate)
            next_state = cast(
                DelightfulActorCriticState,
                _tree_select(candidate_valid, candidate, state),
            )
            initial = self._initial_sample()
            return DelightfulActorCriticStartResult(
                state=next_state,
                action=jnp.where(candidate_valid, sample.action, -1).astype(jnp.int32),
                target_policy=jnp.where(
                    candidate_valid, sample.target_policy, initial.target_policy
                ),
                behavior_policy=jnp.where(
                    candidate_valid, sample.behavior_policy, initial.behavior_policy
                ),
                target_log_probability=jnp.where(
                    candidate_valid, sample.target_log_probability, 0.0
                ),
                behavior_log_probability=jnp.where(
                    candidate_valid, sample.behavior_log_probability, 0.0
                ),
                state_valid=state_valid,
                input_valid=input_valid,
                applied=candidate_valid,
            )

        return cast(
            DelightfulActorCriticStartResult,
            jax.lax.cond(
                can_start,
                do_start,
                lambda _: self._zero_start_result(
                    state,
                    state_valid=state_valid,
                    input_valid=input_valid,
                ),
                operand=None,
            ),
        )

    def _validate_channel_availability(
        self,
        availability: DelightfulChannelAvailability,
        *,
        shape: tuple[int, ...],
    ) -> DelightfulChannelAvailability:
        if not isinstance(availability, DelightfulChannelAvailability):
            raise TypeError("availability must be a DelightfulChannelAvailability")
        for name in ("safety", "model", "representation"):
            _array_contract(
                getattr(availability, name),
                name=f"availability.{name}",
                shape=shape,
                dtype=jnp.bool_,
            )
        return availability

    @staticmethod
    def _routing(
        *,
        internal_available: Array,
        actor_weight: Array,
        availability: DelightfulChannelAvailability,
    ) -> DelightfulChannelRouting:
        one = jnp.asarray(1.0, dtype=jnp.float32)
        zero = jnp.asarray(0.0, dtype=jnp.float32)
        actor_gate = jnp.where(internal_available, actor_weight, zero)
        return DelightfulChannelRouting(
            actor_available=internal_available,
            critic_available=internal_available,
            average_reward_available=internal_available,
            safety_available=availability.safety,
            model_available=availability.model,
            representation_available=availability.representation,
            actor_weight=jax.lax.stop_gradient(actor_gate),
            critic_weight=jnp.where(internal_available, one, zero),
            average_reward_weight=jnp.where(internal_available, one, zero),
            safety_weight=jnp.where(internal_available & availability.safety, one, zero),
            model_weight=jnp.where(internal_available & availability.model, one, zero),
            representation_weight=jnp.where(
                internal_available & availability.representation, one, zero
            ),
        )

    def _rejected_update(
        self,
        state: DelightfulActorCriticState,
        availability: DelightfulChannelAvailability,
        *,
        state_valid: Array,
        input_valid: Array,
        capacity_available: Array,
        sample_available: Array,
    ) -> DelightfulActorCriticUpdateResult:
        zero = jnp.asarray(0.0, dtype=jnp.float32)
        false = jnp.asarray(False, dtype=jnp.bool_)
        diagnostics = DelightfulActorCriticDiagnostics(
            state_valid=state_valid,
            input_valid=input_valid,
            capacity_available=capacity_available,
            sample_available=sample_available,
            signals_finite=false,
            candidate_state_valid=false,
            applied=false,
            rejected=jnp.asarray(True, dtype=jnp.bool_),
            selected_log_probability=zero,
            action_surprisal=zero,
            advantage=zero,
            delight=zero,
            gate_weight=zero,
            effective_sample_size=zero,
            positive_delight_rate=zero,
            negative_delight_rate=zero,
            positive_delight_gate_rate=zero,
            negative_delight_gate_rate=zero,
            actor_update_norm=zero,
            critic_update_norm=zero,
            routing=self._routing(
                internal_available=false,
                actor_weight=zero,
                availability=availability,
            ),
        )
        initial = self._initial_sample()
        return DelightfulActorCriticUpdateResult(
            state=state,
            action=initial.action,
            target_policy=initial.target_policy,
            behavior_policy=initial.behavior_policy,
            value=zero,
            next_value=zero,
            average_reward=zero,
            diagnostics=diagnostics,
        )

    def update(
        self,
        state: DelightfulActorCriticState,
        reward: Array,
        next_observation: Array,
        availability: DelightfulChannelAvailability,
    ) -> DelightfulActorCriticUpdateResult:
        """Apply one atomic differential actor-critic transition update."""
        self._validate_state_static_contract(state)
        rew = _array_contract(reward, name="reward", shape=(), dtype=jnp.float32)
        next_obs = _array_contract(
            next_observation,
            name="next_observation",
            shape=(self._config.observation_dim,),
            dtype=jnp.float32,
        )
        availability = self._validate_channel_availability(availability, shape=())
        return cast(
            DelightfulActorCriticUpdateResult,
            self._update_compiled(state, rew, next_obs, availability),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def _update_compiled(
        self,
        state: DelightfulActorCriticState,
        reward: Array,
        next_observation: Array,
        availability: DelightfulChannelAvailability,
    ) -> DelightfulActorCriticUpdateResult:
        rew = reward
        next_obs = next_observation
        cfg = self._config
        state_valid = self._state_valid(state)
        input_valid = (
            jnp.isfinite(rew)
            & (jnp.abs(rew) <= cfg.max_input_magnitude)
            & jnp.all(jnp.isfinite(next_obs))
            & jnp.all(jnp.abs(next_obs) <= cfg.max_input_magnitude)
        )
        capacity_available = state.transition_count < cfg.max_updates
        sample_available = state.last_sample.available
        can_attempt = state_valid & input_valid & capacity_available & sample_available

        def do_update(_: None) -> DelightfulActorCriticUpdateResult:
            old_obs = state.last_observation
            sample = state.last_sample
            value = jnp.dot(state.critic_weights, old_obs) + state.critic_bias
            next_value = jnp.dot(state.critic_weights, next_obs) + state.critic_bias
            advantage = rew - state.average_reward + next_value - value
            pg = discrete_delightful_policy_gradient(
                sample.behavior_log_probability,
                advantage,
                self._policy_gradient_config,
            )

            action_mask = jax.nn.one_hot(sample.action, cfg.n_actions, dtype=jnp.float32)
            score_bias = (action_mask - sample.target_policy) / jnp.asarray(
                cfg.policy_temperature, dtype=jnp.float32
            )
            score_weights = score_bias[:, None] * old_obs[None, :]
            actor_coefficient = jnp.asarray(pg.actor_coefficients, dtype=jnp.float32)
            raw_actor_weight_step = (
                jnp.asarray(cfg.actor_step_size, dtype=jnp.float32)
                * actor_coefficient
                * score_weights
            )
            raw_actor_bias_step = (
                jnp.asarray(cfg.actor_step_size, dtype=jnp.float32) * actor_coefficient * score_bias
            )

            critic_trace_weights = (
                jnp.asarray(cfg.critic_trace_lambda, dtype=jnp.float32) * state.critic_trace_weights
                + old_obs
            )
            critic_trace_bias = (
                jnp.asarray(cfg.critic_trace_lambda, dtype=jnp.float32) * state.critic_trace_bias
                + 1.0
            )
            raw_critic_weight_step = (
                jnp.asarray(cfg.critic_step_size, dtype=jnp.float32)
                * advantage
                * critic_trace_weights
            )
            raw_critic_bias_step = (
                jnp.asarray(cfg.critic_step_size, dtype=jnp.float32) * advantage * critic_trace_bias
            )
            raw_average_reward_step = (
                jnp.asarray(cfg.average_reward_step_size, dtype=jnp.float32) * advantage
            )
            raw_steps_finite = (
                jnp.all(jnp.isfinite(raw_actor_weight_step))
                & jnp.all(jnp.isfinite(raw_actor_bias_step))
                & jnp.all(jnp.isfinite(raw_critic_weight_step))
                & jnp.isfinite(raw_critic_bias_step)
                & jnp.isfinite(raw_average_reward_step)
            )
            update_bound = jnp.asarray(cfg.max_update_component_magnitude, dtype=jnp.float32)
            actor_weight_step = jnp.clip(raw_actor_weight_step, -update_bound, update_bound)
            actor_bias_step = jnp.clip(raw_actor_bias_step, -update_bound, update_bound)
            critic_weight_step = jnp.clip(raw_critic_weight_step, -update_bound, update_bound)
            critic_bias_step = jnp.clip(raw_critic_bias_step, -update_bound, update_bound)
            average_reward_step = jnp.clip(raw_average_reward_step, -update_bound, update_bound)

            count = state.transition_count + jnp.asarray(1, dtype=jnp.int32)
            candidate_without_sample = state.replace(
                actor_weights=state.actor_weights + actor_weight_step,
                actor_bias=state.actor_bias + actor_bias_step,
                critic_weights=state.critic_weights + critic_weight_step,
                critic_bias=state.critic_bias + critic_bias_step,
                critic_trace_weights=critic_trace_weights,
                critic_trace_bias=critic_trace_bias,
                average_reward=state.average_reward + average_reward_step,
                transition_count=count,
                actor_update_count=count,
                critic_update_count=count,
                average_reward_update_count=count,
            )
            next_sample, next_key = self._sample_unchecked(candidate_without_sample, next_obs)
            candidate = candidate_without_sample.replace(
                last_observation=next_obs,
                last_sample=next_sample,
                rng_key=next_key,
            )
            actor_norm = jnp.sqrt(
                jnp.sum(jnp.square(actor_weight_step)) + jnp.sum(jnp.square(actor_bias_step))
            )
            critic_norm = jnp.sqrt(
                jnp.sum(jnp.square(critic_weight_step)) + jnp.square(critic_bias_step)
            )
            signals_finite = (
                jnp.isfinite(value)
                & jnp.isfinite(next_value)
                & jnp.isfinite(advantage)
                & jnp.isfinite(pg.action_surprisal)
                & jnp.isfinite(pg.delight)
                & jnp.isfinite(pg.sample_weights)
                & jnp.isfinite(pg.actor_coefficients)
                & raw_steps_finite
                & jnp.isfinite(actor_norm)
                & jnp.isfinite(critic_norm)
            )
            candidate_valid = self._state_valid(candidate)
            applied = signals_finite & candidate_valid
            next_state = cast(
                DelightfulActorCriticState,
                _tree_select(applied, candidate, state),
            )
            zero = jnp.asarray(0.0, dtype=jnp.float32)
            delight = jnp.asarray(pg.delight, dtype=jnp.float32)
            gate = jnp.asarray(pg.sample_weights, dtype=jnp.float32)
            positive = delight > 0.0
            negative = delight < 0.0
            diagnostics = DelightfulActorCriticDiagnostics(
                state_valid=state_valid,
                input_valid=input_valid,
                capacity_available=capacity_available,
                sample_available=sample_available,
                signals_finite=signals_finite,
                candidate_state_valid=candidate_valid,
                applied=applied,
                rejected=~applied,
                selected_log_probability=jnp.where(applied, sample.behavior_log_probability, zero),
                action_surprisal=jnp.where(applied, pg.action_surprisal, zero),
                advantage=jnp.where(applied, advantage, zero),
                delight=jnp.where(applied, delight, zero),
                gate_weight=jnp.where(applied, gate, zero),
                effective_sample_size=jnp.where(
                    applied, pg.diagnostics.effective_sample_size, zero
                ),
                positive_delight_rate=jnp.where(applied, positive.astype(jnp.float32), zero),
                negative_delight_rate=jnp.where(applied, negative.astype(jnp.float32), zero),
                positive_delight_gate_rate=jnp.where(applied & positive, gate, zero),
                negative_delight_gate_rate=jnp.where(applied & negative, gate, zero),
                actor_update_norm=jnp.where(applied, actor_norm, zero),
                critic_update_norm=jnp.where(applied, critic_norm, zero),
                routing=self._routing(
                    internal_available=applied,
                    actor_weight=gate,
                    availability=availability,
                ),
            )
            initial = self._initial_sample()
            reported_sample = cast(
                DelightfulPolicySample,
                _tree_select(applied, next_sample, initial),
            )
            return DelightfulActorCriticUpdateResult(
                state=next_state,
                action=reported_sample.action,
                target_policy=reported_sample.target_policy,
                behavior_policy=reported_sample.behavior_policy,
                value=jnp.where(applied, value, zero),
                next_value=jnp.where(applied, next_value, zero),
                average_reward=jnp.where(applied, candidate.average_reward, zero),
                diagnostics=diagnostics,
            )

        return cast(
            DelightfulActorCriticUpdateResult,
            jax.lax.cond(
                can_attempt,
                do_update,
                lambda _: self._rejected_update(
                    state,
                    availability,
                    state_valid=state_valid,
                    input_valid=input_valid,
                    capacity_available=capacity_available,
                    sample_available=sample_available,
                ),
                operand=None,
            ),
        )

    def checkpoint_payload(self, state: DelightfulActorCriticState) -> dict[str, Any]:
        """Return a strict versioned JSON-compatible agent/state checkpoint."""
        self._validate_state_static_contract(state)
        if not bool(jax.device_get(self._state_valid(state))):
            raise ValueError("cannot checkpoint an invalid actor-critic state")

        def floats(value: Array) -> Any:
            return np.asarray(jax.device_get(value), dtype=np.float32).tolist()

        sample = state.last_sample
        return {
            "schema": _CHECKPOINT_SCHEMA,
            "agent": self.to_config(),
            "state": {
                "actor_weights": floats(state.actor_weights),
                "actor_bias": floats(state.actor_bias),
                "critic_weights": floats(state.critic_weights),
                "critic_bias": floats(state.critic_bias),
                "critic_trace_weights": floats(state.critic_trace_weights),
                "critic_trace_bias": floats(state.critic_trace_bias),
                "average_reward": floats(state.average_reward),
                "last_observation": floats(state.last_observation),
                "last_sample": {
                    "available": bool(sample.available),
                    "action": int(sample.action),
                    "target_policy": floats(sample.target_policy),
                    "behavior_policy": floats(sample.behavior_policy),
                    "target_probability": floats(sample.target_probability),
                    "behavior_probability": floats(sample.behavior_probability),
                    "target_log_probability": floats(sample.target_log_probability),
                    "behavior_log_probability": floats(sample.behavior_log_probability),
                },
                "rng_key": np.asarray(
                    jax.device_get(jr.key_data(state.rng_key)), dtype=np.uint32
                ).tolist(),
                "transition_count": int(state.transition_count),
                "actor_update_count": int(state.actor_update_count),
                "critic_update_count": int(state.critic_update_count),
                "average_reward_update_count": int(state.average_reward_update_count),
            },
        }

    @staticmethod
    def _checkpoint_float_array(
        value: Any,
        *,
        name: str,
        shape: tuple[int, ...],
    ) -> Array:
        untyped = np.asarray(value, dtype=object)
        if untyped.shape != shape:
            raise ValueError(f"{name} must have shape {shape}, got {untyped.shape}")
        if any(
            isinstance(item, bool) or not isinstance(item, int | float) for item in untyped.flat
        ):
            raise ValueError(f"{name} must contain only JSON real numbers")
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            array: npt.NDArray[np.float32] = np.asarray(value, dtype=np.float32)
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must be finite")
        return jnp.asarray(array, dtype=jnp.float32)

    @staticmethod
    def _checkpoint_count(value: Any, *, name: str) -> Array:
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= _INT32_MAX:
            raise ValueError(f"{name} must be an int32-range integer")
        return jnp.asarray(value, dtype=jnp.int32)

    @classmethod
    def from_checkpoint_payload(
        cls,
        checkpoint: Mapping[str, Any],
    ) -> tuple[DelightfulActorCriticAgent, DelightfulActorCriticState]:
        """Strictly reconstruct an agent/state pair from a v1 checkpoint."""
        if set(checkpoint) != {"schema", "agent", "state"}:
            raise ValueError("checkpoint fields do not match the v1 schema")
        if checkpoint.get("schema") != _CHECKPOINT_SCHEMA:
            raise ValueError("unexpected delightful actor-critic checkpoint schema")
        agent_payload = checkpoint.get("agent")
        state_payload = checkpoint.get("state")
        if not isinstance(agent_payload, Mapping) or not isinstance(state_payload, Mapping):
            raise ValueError("checkpoint must contain agent and state mappings")
        expected_state = {
            "actor_weights",
            "actor_bias",
            "critic_weights",
            "critic_bias",
            "critic_trace_weights",
            "critic_trace_bias",
            "average_reward",
            "last_observation",
            "last_sample",
            "rng_key",
            "transition_count",
            "actor_update_count",
            "critic_update_count",
            "average_reward_update_count",
        }
        if set(state_payload) != expected_state:
            raise ValueError("checkpoint state fields do not match the v1 schema")
        agent = cls.from_config(agent_payload)
        cfg = agent.config
        sample_payload = state_payload["last_sample"]
        if not isinstance(sample_payload, Mapping):
            raise ValueError("checkpoint last_sample must be a mapping")
        expected_sample = {
            "available",
            "action",
            "target_policy",
            "behavior_policy",
            "target_probability",
            "behavior_probability",
            "target_log_probability",
            "behavior_log_probability",
        }
        if set(sample_payload) != expected_sample:
            raise ValueError("checkpoint sample fields do not match the v1 schema")
        available = sample_payload["available"]
        if not isinstance(available, bool):
            raise ValueError("checkpoint sample available must be boolean")
        action = sample_payload["action"]
        if isinstance(action, bool) or not isinstance(action, int):
            raise ValueError("checkpoint sample action must be an integer")
        key_data = np.asarray(state_payload["rng_key"])
        if key_data.shape != (2,) or not np.issubdtype(key_data.dtype, np.integer):
            raise ValueError("checkpoint rng_key must contain two uint32 words")
        if np.any(key_data < 0) or np.any(key_data > np.iinfo(np.uint32).max):
            raise ValueError("checkpoint rng_key words must be in uint32 range")
        sample = DelightfulPolicySample(
            available=jnp.asarray(available, dtype=jnp.bool_),
            action=jnp.asarray(action, dtype=jnp.int32),
            target_policy=agent._checkpoint_float_array(
                sample_payload["target_policy"],
                name="state.last_sample.target_policy",
                shape=(cfg.n_actions,),
            ),
            behavior_policy=agent._checkpoint_float_array(
                sample_payload["behavior_policy"],
                name="state.last_sample.behavior_policy",
                shape=(cfg.n_actions,),
            ),
            target_probability=agent._checkpoint_float_array(
                sample_payload["target_probability"],
                name="state.last_sample.target_probability",
                shape=(),
            ),
            behavior_probability=agent._checkpoint_float_array(
                sample_payload["behavior_probability"],
                name="state.last_sample.behavior_probability",
                shape=(),
            ),
            target_log_probability=agent._checkpoint_float_array(
                sample_payload["target_log_probability"],
                name="state.last_sample.target_log_probability",
                shape=(),
            ),
            behavior_log_probability=agent._checkpoint_float_array(
                sample_payload["behavior_log_probability"],
                name="state.last_sample.behavior_log_probability",
                shape=(),
            ),
        )
        state = DelightfulActorCriticState(
            actor_weights=agent._checkpoint_float_array(
                state_payload["actor_weights"],
                name="state.actor_weights",
                shape=(cfg.n_actions, cfg.observation_dim),
            ),
            actor_bias=agent._checkpoint_float_array(
                state_payload["actor_bias"],
                name="state.actor_bias",
                shape=(cfg.n_actions,),
            ),
            critic_weights=agent._checkpoint_float_array(
                state_payload["critic_weights"],
                name="state.critic_weights",
                shape=(cfg.observation_dim,),
            ),
            critic_bias=agent._checkpoint_float_array(
                state_payload["critic_bias"], name="state.critic_bias", shape=()
            ),
            critic_trace_weights=agent._checkpoint_float_array(
                state_payload["critic_trace_weights"],
                name="state.critic_trace_weights",
                shape=(cfg.observation_dim,),
            ),
            critic_trace_bias=agent._checkpoint_float_array(
                state_payload["critic_trace_bias"],
                name="state.critic_trace_bias",
                shape=(),
            ),
            average_reward=agent._checkpoint_float_array(
                state_payload["average_reward"], name="state.average_reward", shape=()
            ),
            last_observation=agent._checkpoint_float_array(
                state_payload["last_observation"],
                name="state.last_observation",
                shape=(cfg.observation_dim,),
            ),
            last_sample=sample,
            rng_key=jr.wrap_key_data(
                jnp.asarray(key_data, dtype=jnp.uint32),
                impl="threefry2x32",
            ),
            transition_count=agent._checkpoint_count(
                state_payload["transition_count"], name="state.transition_count"
            ),
            actor_update_count=agent._checkpoint_count(
                state_payload["actor_update_count"], name="state.actor_update_count"
            ),
            critic_update_count=agent._checkpoint_count(
                state_payload["critic_update_count"], name="state.critic_update_count"
            ),
            average_reward_update_count=agent._checkpoint_count(
                state_payload["average_reward_update_count"],
                name="state.average_reward_update_count",
            ),
        )
        agent._validate_state_static_contract(state)
        if not bool(jax.device_get(agent._state_valid(state))):
            raise ValueError("checkpoint contains an invalid actor-critic state")
        return agent, state


def _batch_diagnostics(
    selected_log_probabilities: Array,
    action_surprisals: Array,
    advantages: Array,
    delights: Array,
    gate_weights: Array,
    applied: Array,
    diagnostics_epsilon: float,
) -> DelightfulActorCriticBatchDiagnostics:
    mask = applied.astype(jnp.bool_)
    mask_f = mask.astype(jnp.float32)
    count = jnp.sum(mask_f)
    weights = jnp.where(mask, gate_weights, 0.0)
    weight_sum = jnp.sum(weights)
    weight_square_sum = jnp.sum(jnp.square(weights))
    effective = jnp.where(
        count > 0.0,
        weight_sum**2
        / jnp.maximum(
            weight_square_sum,
            jnp.asarray(diagnostics_epsilon, dtype=jnp.float32),
        ),
        0.0,
    )
    positive = mask & (delights > 0.0)
    negative = mask & (delights < 0.0)
    return DelightfulActorCriticBatchDiagnostics(
        attempted_count=jnp.asarray(applied.size, dtype=jnp.int32),
        accepted_count=jnp.sum(mask.astype(jnp.int32)),
        effective_sample_size=effective,
        effective_sample_fraction=effective / jnp.maximum(count, 1.0),
        positive_delight_rate=jnp.sum(positive.astype(jnp.float32)) / jnp.maximum(count, 1.0),
        negative_delight_rate=jnp.sum(negative.astype(jnp.float32)) / jnp.maximum(count, 1.0),
        positive_delight_gate_rate=_safe_masked_mean(gate_weights, positive),
        negative_delight_gate_rate=_safe_masked_mean(gate_weights, negative),
        mean_gate_rate=_safe_masked_mean(gate_weights, mask),
        mean_selected_log_probability=_safe_masked_mean(selected_log_probabilities, mask),
        mean_action_surprisal=_safe_masked_mean(action_surprisals, mask),
        mean_advantage=_safe_masked_mean(advantages, mask),
        mean_delight=_safe_masked_mean(delights, mask),
    )


def run_delightful_actor_critic_from_arrays(
    agent: DelightfulActorCriticAgent,
    state: DelightfulActorCriticState,
    rewards: Array,
    next_observations: Array,
    availability: DelightfulChannelAvailability,
) -> DelightfulActorCriticArrayResult:
    """Run the exact single-transition API with ``jax.lax.scan``."""
    agent._validate_state_static_contract(state)
    reward_arr = _array_contract(
        rewards,
        name="rewards",
        shape=None,
        dtype=jnp.float32,
    )
    if reward_arr.ndim != 1:
        raise ValueError("rewards must be a rank-one float32 array")
    n_steps = reward_arr.shape[0]
    _positive_int(n_steps, name="num_steps")
    observation_arr = _array_contract(
        next_observations,
        name="next_observations",
        shape=(n_steps, agent.config.observation_dim),
        dtype=jnp.float32,
    )
    availability = agent._validate_channel_availability(availability, shape=(n_steps,))
    return cast(
        DelightfulActorCriticArrayResult,
        _run_delightful_actor_critic_from_arrays_compiled(
            agent,
            state,
            reward_arr,
            observation_arr,
            availability,
        ),
    )


@functools.partial(jax.jit, static_argnums=(0,))
def _run_delightful_actor_critic_from_arrays_compiled(
    agent: DelightfulActorCriticAgent,
    state: DelightfulActorCriticState,
    rewards: Array,
    next_observations: Array,
    availability: DelightfulChannelAvailability,
) -> DelightfulActorCriticArrayResult:
    reward_arr = rewards
    observation_arr = next_observations

    def scan_step(
        carry: DelightfulActorCriticState,
        inputs: tuple[Array, Array, Array, Array, Array],
    ) -> tuple[DelightfulActorCriticState, _DelightfulActorCriticScanOutput]:
        reward, observation, safety, model, representation = inputs
        result = agent._update_compiled(
            carry,
            reward,
            observation,
            DelightfulChannelAvailability(
                safety=safety,
                model=model,
                representation=representation,
            ),
        )
        diagnostics = result.diagnostics
        output = _DelightfulActorCriticScanOutput(
            action=result.action,
            target_policy=result.target_policy,
            behavior_policy=result.behavior_policy,
            selected_log_probability=diagnostics.selected_log_probability,
            action_surprisal=diagnostics.action_surprisal,
            advantage=diagnostics.advantage,
            delight=diagnostics.delight,
            gate_weight=diagnostics.gate_weight,
            applied=diagnostics.applied,
        )
        return result.state, output

    final_state, results = jax.lax.scan(
        scan_step,
        state,
        (
            reward_arr,
            observation_arr,
            availability.safety,
            availability.model,
            availability.representation,
        ),
    )
    batch = _batch_diagnostics(
        results.selected_log_probability,
        results.action_surprisal,
        results.advantage,
        results.delight,
        results.gate_weight,
        results.applied,
        agent.config.diagnostics_epsilon,
    )
    return DelightfulActorCriticArrayResult(
        state=final_state,
        actions=results.action,
        target_policies=results.target_policy,
        behavior_policies=results.behavior_policy,
        selected_log_probabilities=results.selected_log_probability,
        action_surprisals=results.action_surprisal,
        advantages=results.advantage,
        delights=results.delight,
        gate_weights=results.gate_weight,
        applied=results.applied,
        diagnostics=batch,
    )


__all__ = [
    "DelightfulActorCriticAgent",
    "DelightfulActorCriticArrayResult",
    "DelightfulActorCriticBatchDiagnostics",
    "DelightfulActorCriticConfig",
    "DelightfulActorCriticDiagnostics",
    "DelightfulActorCriticResourceBudget",
    "DelightfulActorCriticStartResult",
    "DelightfulActorCriticState",
    "DelightfulActorCriticUpdateResult",
    "DelightfulChannelAvailability",
    "DelightfulChannelRouting",
    "DelightfulPolicySample",
    "run_delightful_actor_critic_from_arrays",
]
