"""Actor-critic control with discrete and continuous policies.

This module provides the Step 4b control cores for daemon-style use:
``ActorCriticAgent`` for discrete (softmax) actions and
``ContinuousActorCriticAgent`` for continuous (diagonal-Gaussian) actions.
Both share the same linear-critic AC(lambda) semantics, separate eligibility
traces, and pure single-step APIs compatible with ``jax.jit`` and
``jax.lax.scan``.

The Horde-backed critic integration point is the scalar ``value``/TD-error
path in ``update``: replace the linear critic estimate and critic trace update
with a GVF value adapter that preserves the actor's advantage signal. That
adapter lives in :mod:`alberta_framework.core.horde_actor_critic`
(``HordeActorCriticAgent`` and ``QHordeActorCriticAgent``); it is kept out of
this core slice so the linear AC(lambda) semantics here remain explicit and
covered by focused tests.
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
from jax import Array
from jaxtyping import Bool, Float, Int, UInt

from alberta_framework.core.optimizers import Bounder, bounder_from_config

ACTOR_CRITIC_CONFIG_SCHEMA = "alberta.actor-critic.config.v2"
ACTOR_CRITIC_STATE_SCHEMA = "alberta.actor-critic.state.v2"
CONTINUOUS_ACTOR_CRITIC_CONFIG_SCHEMA = "alberta.continuous-actor-critic.config.v2"
CONTINUOUS_ACTOR_CRITIC_STATE_SCHEMA = "alberta.continuous-actor-critic.state.v2"
ACTOR_CRITIC_EXACT_UPDATE_IDENTITY_NBYTES = 8
CONTINUOUS_ACTOR_CRITIC_EXACT_UPDATE_IDENTITY_NBYTES = 8
ACTOR_CRITIC_LIFETIME_COUNTER_NBYTES = 12
CONTINUOUS_ACTOR_CRITIC_LIFETIME_COUNTER_NBYTES = 12

_INT32_MAX = 2**31 - 1
_UINT32_MAX = 2**32 - 1


def _positive_int(value: Any, *, name: str) -> None:
    if type(value) is not int or value <= 0 or value > _INT32_MAX:
        raise ValueError(f"{name} must be a strict integer in [1, {_INT32_MAX}]")


def _finite_config_float(
    value: Any,
    *,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
    minimum_inclusive: bool = True,
) -> None:
    """Require one real scalar that remains finite after float32 narrowing."""

    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a real scalar")
    parsed = float(value)
    with np.errstate(over="ignore", invalid="ignore"):
        narrowed = float(np.float32(parsed))
    if not math.isfinite(parsed) or not math.isfinite(narrowed):
        raise ValueError(f"{name} must be finite in float32")
    if minimum is not None:
        if minimum_inclusive and narrowed < minimum:
            raise ValueError(f"{name} must be at least {minimum}")
        if not minimum_inclusive and narrowed <= minimum:
            raise ValueError(f"{name} must be greater than {minimum}")
    if maximum is not None and narrowed > maximum:
        raise ValueError(f"{name} must be at most {maximum}")


def _strict_config_payload(
    config: Mapping[str, Any],
    cls: type[Any],
    *,
    schema: str,
) -> dict[str, Any]:
    if not isinstance(config, Mapping):
        raise TypeError("actor-critic config must be a mapping")
    payload = dict(config)
    expected = {field.name for field in dataclasses.fields(cls)} | {
        "schema",
        "type",
    }
    if set(payload) != expected:
        if "schema" not in payload:
            raise ValueError("legacy actor-critic config requires explicit migration")
        missing = sorted(expected - set(payload))
        extra = sorted(set(payload) - expected)
        raise ValueError(
            f"actor-critic config fields do not match v2; missing={missing}, extra={extra}"
        )
    if payload.pop("schema") != schema:
        raise ValueError("actor-critic config schema is unsupported")
    if payload.pop("type") != cls.__name__:
        raise ValueError("actor-critic config type is unsupported")
    return payload


def _require_array_contract(
    value: Any,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: Any,
) -> Array:
    """Require one exact persistent array shape and dtype."""

    array = jnp.asarray(value)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    expected_dtype = jnp.dtype(dtype)
    if array.dtype != expected_dtype:
        raise TypeError(f"{name} must have dtype {expected_dtype}, got {array.dtype}")
    return array


def _require_numeric_source(
    value: Any,
    *,
    name: str,
    shape: tuple[int, ...],
) -> Float[Array, ...]:
    """Require a real numeric source and canonicalize it to float32."""

    array = jnp.asarray(value)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not (jnp.issubdtype(array.dtype, jnp.floating) or jnp.issubdtype(array.dtype, jnp.integer)):
        raise TypeError(f"{name} must have a real numeric dtype")
    return array.astype(jnp.float32)


def _require_terminal_source(value: Any, *, name: str) -> Array:
    array = jnp.asarray(value)
    if array.shape != ():
        raise ValueError(f"{name} must be scalar, got {array.shape}")
    if not (
        jnp.issubdtype(array.dtype, jnp.bool_)
        or jnp.issubdtype(array.dtype, jnp.floating)
        or jnp.issubdtype(array.dtype, jnp.integer)
    ):
        raise TypeError(f"{name} must have a boolean or real numeric dtype")
    return array.astype(jnp.float32)


def _require_terminal_array(
    value: Any,
    *,
    name: str,
    shape: tuple[int, ...],
) -> Array:
    array = jnp.asarray(value)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not (
        jnp.issubdtype(array.dtype, jnp.bool_)
        or jnp.issubdtype(array.dtype, jnp.floating)
        or jnp.issubdtype(array.dtype, jnp.integer)
    ):
        raise TypeError(f"{name} must have a boolean or real numeric dtype")
    return array.astype(jnp.float32)


def _prng_key_contract(value: Any, *, name: str) -> Array:
    """Accept exactly one typed Threefry key or its legacy two-word form."""

    array = jnp.asarray(value)
    if array.shape == () and jnp.issubdtype(array.dtype, jax.dtypes.prng_key):
        data = jr.key_data(array)
        if data.shape == (2,) and data.dtype == jnp.uint32:
            return array
    if array.shape == (2,) and array.dtype == jnp.uint32:
        return array
    raise ValueError(f"{name} must be one two-word Threefry JAX PRNG key")


def _checked_update_words_increment(
    words: Array,
) -> tuple[UInt[Array, " 2"], Bool[Array, ""]]:
    """Propose an exact uint64-word successor without wrapping all-ones."""

    _require_array_contract(
        words,
        name="step_words",
        shape=(2,),
        dtype=jnp.uint32,
    )
    maximum = jnp.asarray(_UINT32_MAX, dtype=jnp.uint32)
    capacity_available = ~jnp.all(words == maximum)
    one = jnp.asarray(1, dtype=jnp.uint32)
    low = words[1] + one
    carry = (low == jnp.asarray(0, dtype=jnp.uint32)).astype(jnp.uint32)
    proposed = jnp.stack((words[0] + carry, low)).astype(jnp.uint32)
    return jnp.where(capacity_available, proposed, words), capacity_available


def _words_to_saturating_int32(words: Array) -> Int[Array, ""]:
    _require_array_contract(
        words,
        name="step_words",
        shape=(2,),
        dtype=jnp.uint32,
    )
    below_saturation = (words[0] == jnp.asarray(0, dtype=jnp.uint32)) & (
        words[1] < jnp.asarray(_INT32_MAX, dtype=jnp.uint32)
    )
    return jnp.where(
        below_saturation,
        words[1].astype(jnp.int32),
        jnp.asarray(_INT32_MAX, dtype=jnp.int32),
    )


def _lifetime_counter_valid(
    words: Array,
    telemetry: Array,
) -> Bool[Array, ""]:
    _require_array_contract(
        telemetry,
        name="step_count",
        shape=(),
        dtype=jnp.int32,
    )
    return (telemetry >= 0) & (telemetry == _words_to_saturating_int32(words))


def _floating_tree_is_finite(tree: object) -> Bool[Array, ""]:
    valid = jnp.asarray(True, dtype=jnp.bool_)
    for leaf in jax.tree.leaves(tree):
        dtype = getattr(leaf, "dtype", None)
        if dtype is not None and jnp.issubdtype(dtype, jnp.inexact):
            valid = valid & jnp.all(jnp.isfinite(leaf))
    return valid


def _measure_array_tree_nbytes(tree: Any) -> int:
    return sum(
        int(leaf.size) * int(leaf.dtype.itemsize)
        for leaf in jax.tree.leaves(tree)
        if isinstance(leaf, Array)
    )


def _legacy_fields(value: Any, *, name: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {field.name: getattr(value, field.name) for field in dataclasses.fields(value)}
    raise TypeError(f"legacy {name} must be a mapping or dataclass")


@dataclasses.dataclass(frozen=True)
class ActorCriticConfig:
    """Configuration for a linear softmax actor-critic agent.

    Attributes:
        n_actions: Number of discrete actions.
        gamma: Discount factor.
        actor_step_size: Step-size for policy parameters.
        critic_step_size: Step-size for value parameters.
        actor_lamda: Eligibility trace decay for the actor.
        critic_lamda: Eligibility trace decay for the critic.
        temperature: Softmax temperature. Values below 1 sharpen the policy.
    """

    n_actions: int
    gamma: float = 0.99
    actor_step_size: float = 0.01
    critic_step_size: float = 0.05
    actor_lamda: float = 0.9
    critic_lamda: float = 0.9
    temperature: float = 1.0

    def __post_init__(self) -> None:
        _positive_int(self.n_actions, name="n_actions")
        _finite_config_float(self.gamma, name="gamma", minimum=0.0, maximum=1.0)
        _finite_config_float(
            self.actor_step_size,
            name="actor_step_size",
            minimum=0.0,
        )
        _finite_config_float(
            self.critic_step_size,
            name="critic_step_size",
            minimum=0.0,
        )
        _finite_config_float(
            self.actor_lamda,
            name="actor_lamda",
            minimum=0.0,
            maximum=1.0,
        )
        _finite_config_float(
            self.critic_lamda,
            name="critic_lamda",
            minimum=0.0,
            maximum=1.0,
        )
        _finite_config_float(
            self.temperature,
            name="temperature",
            minimum=0.0,
            minimum_inclusive=False,
        )

    def to_config(self) -> dict[str, Any]:
        """Serialize this configuration to a dictionary."""
        return {
            "schema": ACTOR_CRITIC_CONFIG_SCHEMA,
            "type": type(self).__name__,
            "n_actions": self.n_actions,
            "gamma": self.gamma,
            "actor_step_size": self.actor_step_size,
            "critic_step_size": self.critic_step_size,
            "actor_lamda": self.actor_lamda,
            "critic_lamda": self.critic_lamda,
            "temperature": self.temperature,
        }

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> ActorCriticConfig:
        """Reconstruct an ``ActorCriticConfig`` from a dictionary."""
        return cls(
            **_strict_config_payload(
                config,
                cls,
                schema=ACTOR_CRITIC_CONFIG_SCHEMA,
            )
        )


def migrate_legacy_actor_critic_config(
    legacy_config: Mapping[str, Any],
) -> ActorCriticConfig:
    """Explicitly migrate the exact schema-less pre-v2 configuration."""

    if not isinstance(legacy_config, Mapping):
        raise TypeError("legacy actor-critic config must be a mapping")
    payload = dict(legacy_config)
    expected = {field.name for field in dataclasses.fields(ActorCriticConfig)}
    if set(payload) != expected:
        missing = sorted(expected - set(payload))
        extra = sorted(set(payload) - expected)
        raise ValueError(
            f"legacy actor-critic config fields are not exact; missing={missing}, extra={extra}"
        )
    return ActorCriticConfig(**payload)


@chex.dataclass(frozen=True)
class ActorCriticState:
    """Immutable state for a linear actor-critic agent.

    Attributes:
        actor_weights: Policy weight matrix, shape ``(n_actions, feature_dim)``.
        actor_bias: Policy bias vector, shape ``(n_actions,)``.
        critic_weights: Value weight vector, shape ``(feature_dim,)``.
        critic_bias: Scalar value bias.
        actor_trace_weights: Eligibility trace for actor weights.
        actor_trace_bias: Eligibility trace for actor bias.
        critic_trace_weights: Eligibility trace for critic weights.
        critic_trace_bias: Eligibility trace for critic bias.
        last_observation: Previous observation ``s_t``.
        last_action: Previous action ``a_t``.
        rng_key: Random key used for action sampling.
        step_count: Saturating int32 compatibility telemetry.
        step_words: Exact big-endian uint32 update identity.
    """

    actor_weights: Float[Array, "n_actions feature_dim"]
    actor_bias: Float[Array, " n_actions"]
    critic_weights: Float[Array, " feature_dim"]
    critic_bias: Float[Array, ""]
    actor_trace_weights: Float[Array, "n_actions feature_dim"]
    actor_trace_bias: Float[Array, " n_actions"]
    critic_trace_weights: Float[Array, " feature_dim"]
    critic_trace_bias: Float[Array, ""]
    last_observation: Float[Array, " feature_dim"]
    last_action: Int[Array, ""]
    rng_key: Array
    step_count: Int[Array, ""]
    step_words: UInt[Array, " 2"]


@dataclasses.dataclass(frozen=True)
class ActorCriticResourceBudget:
    """Exact persistent-state and per-update resource accounting."""

    feature_dim: int
    n_actions: int
    trainable_float32_scalars: int
    trace_float32_scalars: int
    transition_float32_scalars: int
    administrative_int32_scalars: int
    exact_update_identity_uint32_scalars: int
    rng_uint32_scalars: int
    state_nbytes: int
    lifetime_identity_bits: int
    telemetry_saturation: int
    max_rng_draws_per_update: int
    learned_float32_scalars_touched_per_update: int
    replay_capacity: int

    def to_dict(self) -> dict[str, int]:
        """Return a JSON-compatible exact resource record."""

        return dataclasses.asdict(self)


@chex.dataclass(frozen=True)
class ActorCriticUpdateResult:
    """Result from one actor-critic transition update.

    Attributes:
        state: Updated agent state.
        action: Next action selected for the new observation.
        policy: Policy probabilities at the new observation.
        value: Value estimate at the previous observation.
        next_value: Value estimate at the new observation.
        td_error: One-step TD error.
        bound_metric: Mean bounder metric, or 1.0 when no bounder is used.
    """

    state: ActorCriticState
    action: Int[Array, ""]
    policy: Float[Array, " n_actions"]
    value: Float[Array, ""]
    next_value: Float[Array, ""]
    td_error: Float[Array, ""]
    bound_metric: Float[Array, ""]
    pre_step_words: UInt[Array, " 2"]
    proposed_step_words: UInt[Array, " 2"]
    post_step_words: UInt[Array, " 2"]
    lifetime_counter_valid: Bool[Array, ""]
    lifetime_capacity_available: Bool[Array, ""]
    source_state_finite: Bool[Array, ""]
    state_valid: Bool[Array, ""]
    input_valid: Bool[Array, ""]
    source_valid: Bool[Array, ""]
    candidate_state_finite: Bool[Array, ""]
    candidate_state_valid: Bool[Array, ""]
    update_applied: Bool[Array, ""]


@chex.dataclass(frozen=True)
class ActorCriticArrayResult:
    """Result from scan-based actor-critic learning on arrays.

    Attributes:
        state: Final agent state.
        actions: Per-step actions, shape ``(num_steps,)``.
        policies: Per-step policy probabilities, shape ``(num_steps, n_actions)``.
        values: Per-step previous-state value estimates, shape ``(num_steps,)``.
        td_errors: Per-step TD errors, shape ``(num_steps,)``.
    """

    state: ActorCriticState
    actions: Int[Array, " num_steps"]
    policies: Float[Array, "num_steps n_actions"]
    values: Float[Array, " num_steps"]
    td_errors: Float[Array, " num_steps"]
    pre_step_words: UInt[Array, "num_steps 2"]
    post_step_words: UInt[Array, "num_steps 2"]
    lifetime_counter_valid: Bool[Array, " num_steps"]
    lifetime_capacity_available: Bool[Array, " num_steps"]
    state_valid: Bool[Array, " num_steps"]
    input_valid: Bool[Array, " num_steps"]
    candidate_state_finite: Bool[Array, " num_steps"]
    candidate_state_valid: Bool[Array, " num_steps"]
    update_applied: Bool[Array, " num_steps"]


class ActorCriticAgent:
    """Linear actor-critic agent with a discrete softmax policy.

    The actor is a softmax over linear logits and the critic is a scalar
    linear value function. Both components maintain accumulating eligibility
    traces and update at every time step from the same TD error.

    The implemented objective is the continuing or episodic AC(lambda)
    semi-gradient update. For transition ``S_t, A_t, R_{t+1}, S_{t+1}``, the
    critic forms ``delta_t = R_{t+1} + gamma_t V(S_{t+1}) - V(S_t)`` and
    updates value parameters along accumulating traces
    ``e^v_t = gamma_t lambda_v e^v_{t-1} + grad V(S_t)``. The actor updates
    linear softmax logits in the policy-gradient direction
    ``delta_t e^pi_t``, with
    ``e^pi_t = gamma_t lambda_pi e^pi_{t-1} + grad log pi(A_t | S_t)``.
    Because logits are divided by ``temperature`` before the softmax,
    ``grad log pi`` includes the corresponding ``1 / temperature`` factor.
    """

    def __init__(
        self,
        config: ActorCriticConfig,
        bounder: Bounder | None = None,
    ) -> None:
        """Initialize the actor-critic agent.

        Args:
            config: Actor-critic hyperparameters.
            bounder: Optional update bounder compatible with the framework
                ``Bounder`` ABC. When present, actor and critic proposed steps
                are bounded independently using the TD error.
        """
        if not isinstance(config, ActorCriticConfig):
            raise TypeError("config must be an ActorCriticConfig")
        self._config = config
        self._bounder = bounder

    @property
    def config(self) -> ActorCriticConfig:
        """Actor-critic configuration."""
        return self._config

    @property
    def bounder(self) -> Bounder | None:
        """Optional update bounder."""
        return self._bounder

    def to_config(self) -> dict[str, Any]:
        """Serialize this agent to a dictionary."""
        return {
            "type": "ActorCriticAgent",
            "state_schema": ACTOR_CRITIC_STATE_SCHEMA,
            "config": self._config.to_config(),
            "bounder": self._bounder.to_config() if self._bounder is not None else None,
        }

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> ActorCriticAgent:
        """Reconstruct an ``ActorCriticAgent`` from a dictionary."""
        if not isinstance(config, Mapping):
            raise TypeError("actor-critic agent config must be a mapping")
        payload = dict(config)
        expected = {"type", "state_schema", "config", "bounder"}
        if set(payload) != expected:
            if "state_schema" not in payload:
                raise ValueError("legacy actor-critic agent config requires explicit migration")
            raise ValueError("actor-critic agent config fields do not match v2")
        if payload.pop("type") != "ActorCriticAgent":
            raise ValueError("unexpected actor-critic agent type")
        if payload.pop("state_schema") != ACTOR_CRITIC_STATE_SCHEMA:
            raise ValueError("actor-critic state schema is unsupported")
        nested = payload.pop("config")
        if not isinstance(nested, Mapping):
            raise TypeError("actor-critic agent config must contain a mapping")
        ac_config = ActorCriticConfig.from_config(nested)
        bounder_config = payload.pop("bounder")
        if bounder_config is not None and not isinstance(bounder_config, dict):
            raise TypeError("actor-critic bounder config must be a dictionary or null")
        bounder = bounder_from_config(bounder_config) if bounder_config else None
        return cls(config=ac_config, bounder=bounder)

    def _require_state_contract(self, state: ActorCriticState) -> int:
        """Reject malformed persistent structure before traced arithmetic."""

        if not isinstance(state, ActorCriticState):
            raise TypeError("state must be an ActorCriticState")
        observation = jnp.asarray(state.last_observation)
        if observation.ndim != 1 or observation.shape[0] <= 0:
            raise ValueError("state.last_observation must be a nonempty vector")
        feature_dim = observation.shape[0]
        cfg = self._config
        fields = {
            "actor_weights": (
                state.actor_weights,
                (cfg.n_actions, feature_dim),
                jnp.float32,
            ),
            "actor_bias": (state.actor_bias, (cfg.n_actions,), jnp.float32),
            "critic_weights": (state.critic_weights, (feature_dim,), jnp.float32),
            "critic_bias": (state.critic_bias, (), jnp.float32),
            "actor_trace_weights": (
                state.actor_trace_weights,
                (cfg.n_actions, feature_dim),
                jnp.float32,
            ),
            "actor_trace_bias": (
                state.actor_trace_bias,
                (cfg.n_actions,),
                jnp.float32,
            ),
            "critic_trace_weights": (
                state.critic_trace_weights,
                (feature_dim,),
                jnp.float32,
            ),
            "critic_trace_bias": (state.critic_trace_bias, (), jnp.float32),
            "last_observation": (state.last_observation, (feature_dim,), jnp.float32),
            "last_action": (state.last_action, (), jnp.int32),
            "step_count": (state.step_count, (), jnp.int32),
            "step_words": (state.step_words, (2,), jnp.uint32),
        }
        for name, (value, shape, dtype) in fields.items():
            _require_array_contract(
                value,
                name=f"state.{name}",
                shape=shape,
                dtype=dtype,
            )
        _prng_key_contract(state.rng_key, name="state.rng_key")
        return feature_dim

    def state_is_valid(
        self,
        state: ActorCriticState,
        *,
        require_started: bool = False,
    ) -> Bool[Array, ""]:
        """Return dynamic state validity after enforcing its static contract."""

        self._require_state_contract(state)
        lower_action = 0 if require_started else -1
        return (
            _floating_tree_is_finite(state)
            & _lifetime_counter_valid(state.step_words, state.step_count)
            & (state.last_action >= lower_action)
            & (state.last_action < self._config.n_actions)
        )

    def resource_budget(self, feature_dim: int) -> ActorCriticResourceBudget:
        """Return exact fixed-state accounting for one feature width."""

        _positive_int(feature_dim, name="feature_dim")
        n_actions = self._config.n_actions
        trainable = n_actions * feature_dim + n_actions + feature_dim + 1
        traces = trainable
        transition_float32 = feature_dim
        administrative_int32 = 2
        exact_words = 2
        rng_words = 2
        state_nbytes = 4 * (
            trainable + traces + transition_float32 + administrative_int32 + exact_words + rng_words
        )
        return ActorCriticResourceBudget(
            feature_dim=feature_dim,
            n_actions=n_actions,
            trainable_float32_scalars=trainable,
            trace_float32_scalars=traces,
            transition_float32_scalars=transition_float32,
            administrative_int32_scalars=administrative_int32,
            exact_update_identity_uint32_scalars=exact_words,
            rng_uint32_scalars=rng_words,
            state_nbytes=state_nbytes,
            lifetime_identity_bits=64,
            telemetry_saturation=_INT32_MAX,
            max_rng_draws_per_update=1,
            learned_float32_scalars_touched_per_update=(trainable + traces + transition_float32),
            replay_capacity=0,
        )

    def init(self, feature_dim: int, key: Array) -> ActorCriticState:
        """Initialize actor and critic state.

        Args:
            feature_dim: Input feature dimension.
            key: JAX random key.

        Returns:
            Initial immutable actor-critic state.
        """
        _positive_int(feature_dim, name="feature_dim")
        checked_key = _prng_key_contract(key, name="key")
        zeros_actor = jnp.zeros((self._config.n_actions, feature_dim), dtype=jnp.float32)
        zeros_policy_bias = jnp.zeros((self._config.n_actions,), dtype=jnp.float32)
        zeros_critic = jnp.zeros((feature_dim,), dtype=jnp.float32)
        return ActorCriticState(  # type: ignore[call-arg]
            actor_weights=zeros_actor,
            actor_bias=zeros_policy_bias,
            critic_weights=zeros_critic,
            critic_bias=jnp.array(0.0, dtype=jnp.float32),
            actor_trace_weights=zeros_actor,
            actor_trace_bias=zeros_policy_bias,
            critic_trace_weights=zeros_critic,
            critic_trace_bias=jnp.array(0.0, dtype=jnp.float32),
            last_observation=jnp.zeros((feature_dim,), dtype=jnp.float32),
            last_action=jnp.array(-1, dtype=jnp.int32),
            rng_key=checked_key,
            step_count=jnp.array(0, dtype=jnp.int32),
            step_words=jnp.zeros((2,), dtype=jnp.uint32),
        )

    def _policy_unchecked(
        self,
        state: ActorCriticState,
        observation: Array,
    ) -> Array:
        logits = state.actor_weights @ observation + state.actor_bias
        return jax.nn.softmax(logits / self._config.temperature)

    def _value_unchecked(
        self,
        state: ActorCriticState,
        observation: Array,
    ) -> Array:
        return jnp.dot(state.critic_weights, observation) + state.critic_bias

    def _select_action_unchecked(
        self,
        state: ActorCriticState,
        observation: Array,
    ) -> tuple[Array, Array, Array]:
        key, sample_key = jr.split(state.rng_key)
        probs = self._policy_unchecked(state, observation)
        action = jr.categorical(
            sample_key,
            jnp.log(jnp.maximum(probs, 1e-8)),
        ).astype(jnp.int32)
        return action, key, probs

    @functools.partial(jax.jit, static_argnums=(0,))
    def policy(
        self,
        state: ActorCriticState,
        observation: Array,
    ) -> Float[Array, " n_actions"]:
        """Compute softmax action probabilities for one observation."""
        feature_dim = self._require_state_contract(state)
        obs = _require_numeric_source(
            observation,
            name="observation",
            shape=(feature_dim,),
        )
        probs = self._policy_unchecked(state, obs)
        valid = (
            self.state_is_valid(state) & jnp.all(jnp.isfinite(obs)) & jnp.all(jnp.isfinite(probs))
        )
        uniform = jnp.full_like(probs, 1.0 / self._config.n_actions)
        return jnp.where(valid, probs, uniform)

    @functools.partial(jax.jit, static_argnums=(0,))
    def value(self, state: ActorCriticState, observation: Array) -> Float[Array, ""]:
        """Compute the critic value estimate for one observation."""
        feature_dim = self._require_state_contract(state)
        obs = _require_numeric_source(
            observation,
            name="observation",
            shape=(feature_dim,),
        )
        prediction = self._value_unchecked(state, obs)
        valid = self.state_is_valid(state) & jnp.all(jnp.isfinite(obs)) & jnp.isfinite(prediction)
        return jnp.where(valid, prediction, jnp.asarray(0.0, dtype=jnp.float32))

    @functools.partial(jax.jit, static_argnums=(0,))
    def select_action(
        self,
        state: ActorCriticState,
        observation: Array,
    ) -> tuple[Int[Array, ""], Array, Float[Array, " n_actions"]]:
        """Sample one action from the current softmax policy.

        Args:
            state: Current agent state.
            observation: Input feature vector.

        Returns:
            Tuple ``(action, new_rng_key, probabilities)``.
        """
        feature_dim = self._require_state_contract(state)
        obs = _require_numeric_source(
            observation,
            name="observation",
            shape=(feature_dim,),
        )
        action, key, probs = self._select_action_unchecked(state, obs)
        _proposed, capacity = _checked_update_words_increment(state.step_words)
        valid = (
            self.state_is_valid(state)
            & capacity
            & jnp.all(jnp.isfinite(obs))
            & jnp.all(jnp.isfinite(probs))
            & (action >= 0)
            & (action < self._config.n_actions)
        )
        uniform = jnp.full_like(probs, 1.0 / self._config.n_actions)
        return (
            jnp.where(valid, action, jnp.asarray(-1, dtype=jnp.int32)),
            jax.lax.cond(valid, lambda _: key, lambda _: state.rng_key, operand=None),
            jnp.where(valid, probs, uniform),
        )

    def _start_transaction(
        self,
        state: ActorCriticState,
        observation: Array,
    ) -> tuple[ActorCriticState, Array, Array, Bool[Array, ""]]:
        """Stage one start draw and return its internal commit verdict."""

        action, key, probs = self._select_action_unchecked(state, observation)
        _proposed, capacity = _checked_update_words_increment(state.step_words)
        candidate = state.replace(  # type: ignore[attr-defined]
            last_observation=observation,
            last_action=action,
            rng_key=key,
        )
        candidate_valid = (
            _floating_tree_is_finite(candidate)
            & self.state_is_valid(candidate, require_started=True)
            & jnp.all(jnp.isfinite(probs))
        )
        applied = (
            self.state_is_valid(state)
            & capacity
            & jnp.all(jnp.isfinite(observation))
            & candidate_valid
        )
        new_state = jax.lax.cond(
            applied,
            lambda _: candidate,
            lambda _: state,
            operand=None,
        )
        uniform = jnp.full_like(probs, 1.0 / self._config.n_actions)
        return (
            new_state,
            jnp.where(applied, action, jnp.asarray(-1, dtype=jnp.int32)),
            jnp.where(applied, probs, uniform),
            applied,
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def start(
        self,
        state: ActorCriticState,
        observation: Array,
    ) -> tuple[ActorCriticState, Int[Array, ""], Float[Array, " n_actions"]]:
        """Select and store the first action for a new stream or episode."""
        feature_dim = self._require_state_contract(state)
        obs = _require_numeric_source(
            observation,
            name="observation",
            shape=(feature_dim,),
        )
        new_state, action, probs, _applied = self._start_transaction(state, obs)
        return new_state, action, probs

    @functools.partial(jax.jit, static_argnums=(0,))
    def update(
        self,
        state: ActorCriticState,
        reward: Array,
        observation: Array,
        terminated: Array | None = None,
        discount: Array | None = None,
    ) -> ActorCriticUpdateResult:
        """Update actor and critic from one transition.

        The transition is ``(state.last_observation, state.last_action,
        reward, observation)`` plus either a scalar transition ``discount`` or
        the legacy ``terminated`` flag. A next action is sampled and stored in
        the returned state for the following update.

        Args:
            state: Current agent state with a valid previous observation/action.
            reward: Scalar reward.
            observation: Next observation.
            terminated: Backward-compatible scalar terminal flag. Non-zero
                maps to transition discount ``0``; false maps to
                ``config.gamma``. Ignored when ``discount`` is provided.
            discount: Optional scalar per-transition discount ``gamma_t``.
                Use this for continuing logs, variable discounts, time-limit
                truncation semantics, and pre-collected trajectories.

        Returns:
            ``ActorCriticUpdateResult`` containing the updated state and metrics.
        """
        cfg = self._config
        feature_dim = self._require_state_contract(state)
        reward_value = _require_numeric_source(reward, name="reward", shape=())
        next_observation = _require_numeric_source(
            observation,
            name="observation",
            shape=(feature_dim,),
        )
        terminal_valid = jnp.asarray(True, dtype=jnp.bool_)
        if discount is None:
            if terminated is None:
                discount_value = jnp.asarray(cfg.gamma, dtype=jnp.float32)
            else:
                terminal_value = _require_terminal_source(
                    terminated,
                    name="terminated",
                )
                terminal_valid = jnp.isfinite(terminal_value) & (
                    (terminal_value == 0.0) | (terminal_value == 1.0)
                )
                discount_value = jnp.where(terminal_value == 1.0, 0.0, cfg.gamma)
        else:
            discount_value = _require_numeric_source(
                discount,
                name="discount",
                shape=(),
            )
        proposed_step_words, lifetime_capacity_available = _checked_update_words_increment(
            state.step_words
        )
        lifetime_counter_valid = _lifetime_counter_valid(
            state.step_words,
            state.step_count,
        )
        source_state_finite = _floating_tree_is_finite(state)
        state_valid = self.state_is_valid(state, require_started=True)
        input_valid = (
            jnp.isfinite(reward_value)
            & jnp.all(jnp.isfinite(next_observation))
            & jnp.isfinite(discount_value)
            & (discount_value >= 0.0)
            & (discount_value <= 1.0)
            & terminal_valid
        )
        source_valid = state_valid & input_valid
        prev_obs = state.last_observation
        action = jnp.clip(state.last_action, 0, cfg.n_actions - 1)

        old_policy = self._policy_unchecked(state, prev_obs)
        value = self._value_unchecked(state, prev_obs)
        next_value = self._value_unchecked(state, next_observation)
        bootstrap = discount_value * next_value
        td_error = reward_value + bootstrap - value

        one_hot = jax.nn.one_hot(action, cfg.n_actions, dtype=jnp.float32)
        actor_grad_bias = (one_hot - old_policy) / cfg.temperature
        actor_grad_weights = actor_grad_bias[:, None] * prev_obs[None, :]

        actor_decay = discount_value * cfg.actor_lamda
        critic_decay = discount_value * cfg.critic_lamda
        actor_trace_weights = actor_decay * state.actor_trace_weights + actor_grad_weights
        actor_trace_bias = actor_decay * state.actor_trace_bias + actor_grad_bias
        critic_trace_weights = critic_decay * state.critic_trace_weights + prev_obs
        critic_trace_bias = critic_decay * state.critic_trace_bias + 1.0

        actor_steps: tuple[Array, ...] = (
            cfg.actor_step_size * actor_trace_weights,
            cfg.actor_step_size * actor_trace_bias,
        )
        critic_steps: tuple[Array, ...] = (
            cfg.critic_step_size * critic_trace_weights,
            cfg.critic_step_size * critic_trace_bias,
        )
        actor_metric = jnp.array(1.0, dtype=jnp.float32)
        critic_metric = jnp.array(1.0, dtype=jnp.float32)
        if self._bounder is not None:
            actor_steps, actor_metric = self._bounder.bound(
                actor_steps,
                td_error,
                (state.actor_weights, state.actor_bias),
            )
            critic_steps, critic_metric = self._bounder.bound(
                critic_steps,
                td_error,
                (state.critic_weights, state.critic_bias),
            )
        actor_steps = tuple(td_error * step for step in actor_steps)
        critic_steps = tuple(td_error * step for step in critic_steps)

        carry_traces = discount_value != 0.0
        stored_actor_trace_weights = jnp.where(
            carry_traces, actor_trace_weights, jnp.zeros_like(actor_trace_weights)
        )
        stored_actor_trace_bias = jnp.where(
            carry_traces, actor_trace_bias, jnp.zeros_like(actor_trace_bias)
        )
        stored_critic_trace_weights = jnp.where(
            carry_traces, critic_trace_weights, jnp.zeros_like(critic_trace_weights)
        )
        stored_critic_trace_bias = jnp.where(
            carry_traces, critic_trace_bias, jnp.zeros_like(critic_trace_bias)
        )
        updated = state.replace(  # type: ignore[attr-defined]
            actor_weights=state.actor_weights + actor_steps[0],
            actor_bias=state.actor_bias + actor_steps[1],
            critic_weights=state.critic_weights + critic_steps[0],
            critic_bias=state.critic_bias + critic_steps[1],
            actor_trace_weights=stored_actor_trace_weights,
            actor_trace_bias=stored_actor_trace_bias,
            critic_trace_weights=stored_critic_trace_weights,
            critic_trace_bias=stored_critic_trace_bias,
            step_count=_words_to_saturating_int32(proposed_step_words),
            step_words=proposed_step_words,
        )
        next_action, key, next_policy = self._select_action_unchecked(
            updated,
            next_observation,
        )
        candidate_state = updated.replace(
            last_observation=next_observation,
            last_action=next_action,
            rng_key=key,
        )

        bound_metric = (actor_metric + critic_metric) / 2.0
        reports_finite = (
            jnp.all(jnp.isfinite(next_policy))
            & jnp.isfinite(value)
            & jnp.isfinite(next_value)
            & jnp.isfinite(td_error)
            & jnp.isfinite(bound_metric)
            & (next_action >= 0)
            & (next_action < cfg.n_actions)
        )
        candidate_state_finite = _floating_tree_is_finite(candidate_state)
        candidate_state_valid = (
            candidate_state_finite
            & self.state_is_valid(candidate_state, require_started=True)
            & jnp.all(candidate_state.step_words == proposed_step_words)
            & reports_finite
        )
        update_applied = (
            lifetime_counter_valid
            & lifetime_capacity_available
            & source_valid
            & candidate_state_valid
        )
        new_state = jax.lax.cond(
            update_applied,
            lambda _: candidate_state,
            lambda _: state,
            operand=None,
        )
        zero = jnp.asarray(0.0, dtype=jnp.float32)
        uniform = jnp.full((cfg.n_actions,), 1.0 / cfg.n_actions, dtype=jnp.float32)

        return ActorCriticUpdateResult(  # type: ignore[call-arg]
            state=new_state,
            action=jnp.where(update_applied, next_action, -1).astype(jnp.int32),
            policy=jnp.where(update_applied, next_policy, uniform),
            value=jnp.where(update_applied, value, zero),
            next_value=jnp.where(update_applied, next_value, zero),
            td_error=jnp.where(update_applied, td_error, zero),
            bound_metric=jnp.where(update_applied, bound_metric, zero),
            pre_step_words=state.step_words,
            proposed_step_words=proposed_step_words,
            post_step_words=new_state.step_words,
            lifetime_counter_valid=lifetime_counter_valid,
            lifetime_capacity_available=lifetime_capacity_available,
            source_state_finite=source_state_finite,
            state_valid=state_valid,
            input_valid=input_valid,
            source_valid=source_valid,
            candidate_state_finite=candidate_state_finite,
            candidate_state_valid=candidate_state_valid,
            update_applied=update_applied,
        )


def run_actor_critic_from_arrays(
    agent: ActorCriticAgent,
    state: ActorCriticState,
    observations: Float[Array, "num_steps feature_dim"],
    rewards: Float[Array, " num_steps"],
    terminated: Float[Array, " num_steps"] | None,
    next_observations: Float[Array, "num_steps feature_dim"],
    actions: Int[Array, " num_steps"] | None = None,
    discounts: Float[Array, " num_steps"] | None = None,
) -> ActorCriticArrayResult:
    """Run actor-critic updates over arrays with ``jax.lax.scan``.

    By default the scan is on-policy with respect to the current actor. At each
    row it starts from ``observations[t]``, samples/stores an action, and
    applies the transition ending at ``next_observations[t]``. When ``actions``
    is provided, those fixed behavior actions are used instead, which is the
    path intended for pre-collected logs. When ``discounts`` is provided it is
    used as the per-transition discount; otherwise ``terminated`` is mapped to
    ``0`` or ``agent.config.gamma`` for backward compatibility.

    Args:
        agent: Actor-critic agent.
        state: Initial actor-critic state.
        observations: Current observations, shape ``(num_steps, feature_dim)``.
        rewards: Rewards, shape ``(num_steps,)``.
        terminated: Terminal flags, shape ``(num_steps,)``. Required unless
            ``discounts`` is provided.
        next_observations: Next observations, shape ``(num_steps, feature_dim)``.
        actions: Optional fixed current actions, shape ``(num_steps,)``.
        discounts: Optional transition discounts, shape ``(num_steps,)``.

    Returns:
        ``ActorCriticArrayResult`` with final state and per-step metrics.
    """
    if not isinstance(agent, ActorCriticAgent):
        raise TypeError("agent must be an ActorCriticAgent")
    feature_dim = agent._require_state_contract(state)
    reward_array = jnp.asarray(rewards)
    if reward_array.ndim != 1:
        raise ValueError("rewards must be a vector")
    num_steps = reward_array.shape[0]
    reward_values = _require_numeric_source(
        rewards,
        name="rewards",
        shape=(num_steps,),
    )
    observation_values = _require_numeric_source(
        observations,
        name="observations",
        shape=(num_steps, feature_dim),
    )
    next_observation_values = _require_numeric_source(
        next_observations,
        name="next_observations",
        shape=(num_steps, feature_dim),
    )
    if terminated is None and discounts is None:
        raise ValueError("terminated or discounts must be provided")
    if terminated is None:
        terminal_values = jnp.zeros((num_steps,), dtype=jnp.float32)
    else:
        terminal_values = _require_terminal_array(
            terminated,
            name="terminated",
            shape=(num_steps,),
        )
    use_explicit_discounts = discounts is not None
    if discounts is None:
        discount_values = jnp.zeros((num_steps,), dtype=jnp.float32)
    else:
        discount_values = _require_numeric_source(
            discounts,
            name="discounts",
            shape=(num_steps,),
        )
    if actions is None:
        action_values = jnp.full((num_steps,), -1, dtype=jnp.int32)
        use_fixed_actions = False
    else:
        action_values = _require_array_contract(
            actions,
            name="actions",
            shape=(num_steps,),
            dtype=jnp.int32,
        )
        use_fixed_actions = True

    def _scan_fn(
        carry: ActorCriticState,
        inputs: tuple[Array, Array, Array, Array, Array, Array],
    ) -> tuple[ActorCriticState, tuple[Array, ...]]:
        obs, reward, terminal, term_discount, next_obs, fixed_action = inputs
        if use_fixed_actions:
            started_state = carry.replace(  # type: ignore[attr-defined]
                last_observation=obs,
                last_action=fixed_action,
            )
            current_action = fixed_action
            behavior_action_valid = (fixed_action >= 0) & (fixed_action < agent.config.n_actions)
            row_preflight_valid = jnp.all(jnp.isfinite(obs)) & behavior_action_valid
        else:
            (
                started_state,
                current_action,
                _policy,
                row_preflight_valid,
            ) = agent._start_transaction(carry, obs)
        if use_explicit_discounts:
            result = agent.update(
                started_state,
                reward,
                next_obs,
                discount=term_discount,
            )
        else:
            result = agent.update(
                started_state,
                reward,
                next_obs,
                terminated=terminal,
            )
        accepted = result.update_applied & row_preflight_valid
        committed_state = jax.lax.cond(
            accepted,
            lambda _: result.state,
            lambda _: carry,
            operand=None,
        )
        reported_action = jnp.where(accepted, current_action, -1).astype(jnp.int32)
        row_input_valid = result.input_valid & row_preflight_valid
        uniform = jnp.full_like(result.policy, 1.0 / agent.config.n_actions)
        zero = jnp.asarray(0.0, dtype=jnp.float32)
        return committed_state, (
            reported_action,
            jnp.where(accepted, result.policy, uniform),
            jnp.where(accepted, result.value, zero),
            jnp.where(accepted, result.td_error, zero),
            result.pre_step_words,
            committed_state.step_words,
            result.lifetime_counter_valid,
            result.lifetime_capacity_available,
            result.state_valid,
            row_input_valid,
            result.candidate_state_finite & row_preflight_valid,
            result.candidate_state_valid & row_preflight_valid,
            accepted,
        )

    final_state, outputs = jax.lax.scan(
        _scan_fn,
        state,
        (
            observation_values,
            reward_values,
            terminal_values,
            discount_values,
            next_observation_values,
            action_values,
        ),
    )
    (
        actions_out,
        policies,
        values,
        td_errors,
        pre_step_words,
        post_step_words,
        lifetime_counter_valid,
        lifetime_capacity_available,
        state_valid,
        input_valid,
        candidate_state_finite,
        candidate_state_valid,
        update_applied,
    ) = outputs
    return ActorCriticArrayResult(  # type: ignore[call-arg]
        state=final_state,
        actions=actions_out,
        policies=policies,
        values=values,
        td_errors=td_errors,
        pre_step_words=pre_step_words,
        post_step_words=post_step_words,
        lifetime_counter_valid=lifetime_counter_valid,
        lifetime_capacity_available=lifetime_capacity_available,
        state_valid=state_valid,
        input_valid=input_valid,
        candidate_state_finite=candidate_state_finite,
        candidate_state_valid=candidate_state_valid,
        update_applied=update_applied,
    )


# ---------------------------------------------------------------------------
# Continuous-action actor-critic (Step 4 preview)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ContinuousActorCriticConfig:
    """Configuration for a continuous-action linear actor-critic.

    The actor models a diagonal-Gaussian policy ``a ~ N(mu(s), sigma^2)`` with
    a linear mean ``mu(s) = W_mu s + b_mu`` and a per-dimension log-standard-
    deviation parameter ``log_sigma`` (state-independent). Action samples are
    optionally clipped to ``[action_low, action_high]`` after sampling. The
    critic is a scalar linear value function ``V(s) = w_v . s + b_v``. Both
    actor and critic carry their own accumulating eligibility traces and share
    the same TD error.

    Attributes:
        action_dim: Dimensionality of the continuous action vector.
        gamma: Discount factor.
        actor_step_size: Step-size for the actor mean and log-sigma parameters.
        critic_step_size: Step-size for the critic value parameters.
        actor_lamda: Eligibility trace decay for the actor.
        critic_lamda: Eligibility trace decay for the critic.
        log_sigma_init: Initial value for ``log_sigma`` per action dimension.
        log_sigma_min: Lower bound clamp on ``log_sigma`` after each update.
        log_sigma_max: Upper bound clamp on ``log_sigma`` after each update.
        action_low: Lower bound for action clipping. ``None`` disables clipping.
        action_high: Upper bound for action clipping. ``None`` disables clipping.
    """

    action_dim: int
    gamma: float = 0.99
    actor_step_size: float = 0.001
    critic_step_size: float = 0.05
    actor_lamda: float = 0.9
    critic_lamda: float = 0.9
    log_sigma_init: float = -0.5
    log_sigma_min: float = -5.0
    log_sigma_max: float = 2.0
    action_low: float | None = None
    action_high: float | None = None

    def __post_init__(self) -> None:
        _positive_int(self.action_dim, name="action_dim")
        _finite_config_float(self.gamma, name="gamma", minimum=0.0, maximum=1.0)
        _finite_config_float(
            self.actor_step_size,
            name="actor_step_size",
            minimum=0.0,
        )
        _finite_config_float(
            self.critic_step_size,
            name="critic_step_size",
            minimum=0.0,
        )
        _finite_config_float(
            self.actor_lamda,
            name="actor_lamda",
            minimum=0.0,
            maximum=1.0,
        )
        _finite_config_float(
            self.critic_lamda,
            name="critic_lamda",
            minimum=0.0,
            maximum=1.0,
        )
        _finite_config_float(self.log_sigma_init, name="log_sigma_init")
        _finite_config_float(self.log_sigma_min, name="log_sigma_min")
        _finite_config_float(self.log_sigma_max, name="log_sigma_max")
        if self.log_sigma_min > self.log_sigma_max:
            raise ValueError("log_sigma_min must be <= log_sigma_max")
        if not self.log_sigma_min <= self.log_sigma_init <= self.log_sigma_max:
            raise ValueError("log_sigma_init must lie within its configured clamps")
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            sigma_min = float(np.exp(np.float32(self.log_sigma_min)))
            sigma_max = float(np.exp(np.float32(self.log_sigma_max)))
        if sigma_min <= 0.0 or not math.isfinite(sigma_max):
            raise ValueError("log_sigma clamps must produce positive finite float32 sigma")
        if self.action_low is not None:
            _finite_config_float(self.action_low, name="action_low")
        if self.action_high is not None:
            _finite_config_float(self.action_high, name="action_high")
        if (
            self.action_low is not None
            and self.action_high is not None
            and self.action_low > self.action_high
        ):
            raise ValueError("action_low must be <= action_high")

    def to_config(self) -> dict[str, Any]:
        """Serialize this configuration to a dictionary."""
        return {
            "schema": CONTINUOUS_ACTOR_CRITIC_CONFIG_SCHEMA,
            "type": type(self).__name__,
            "action_dim": self.action_dim,
            "gamma": self.gamma,
            "actor_step_size": self.actor_step_size,
            "critic_step_size": self.critic_step_size,
            "actor_lamda": self.actor_lamda,
            "critic_lamda": self.critic_lamda,
            "log_sigma_init": self.log_sigma_init,
            "log_sigma_min": self.log_sigma_min,
            "log_sigma_max": self.log_sigma_max,
            "action_low": self.action_low,
            "action_high": self.action_high,
        }

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
    ) -> ContinuousActorCriticConfig:
        """Reconstruct a ``ContinuousActorCriticConfig`` from a dictionary."""
        return cls(
            **_strict_config_payload(
                config,
                cls,
                schema=CONTINUOUS_ACTOR_CRITIC_CONFIG_SCHEMA,
            )
        )


def migrate_legacy_continuous_actor_critic_config(
    legacy_config: Mapping[str, Any],
) -> ContinuousActorCriticConfig:
    """Explicitly migrate the exact schema-less pre-v2 configuration."""

    if not isinstance(legacy_config, Mapping):
        raise TypeError("legacy continuous actor-critic config must be a mapping")
    payload = dict(legacy_config)
    expected = {field.name for field in dataclasses.fields(ContinuousActorCriticConfig)}
    if set(payload) != expected:
        missing = sorted(expected - set(payload))
        extra = sorted(set(payload) - expected)
        raise ValueError(
            "legacy continuous actor-critic config fields are not exact; "
            f"missing={missing}, extra={extra}"
        )
    return ContinuousActorCriticConfig(**payload)


@chex.dataclass(frozen=True)
class ContinuousActorCriticState:
    """Immutable state for a continuous-action linear actor-critic.

    Attributes:
        mean_weights: Mean head weights, shape ``(action_dim, feature_dim)``.
        mean_bias: Mean head bias, shape ``(action_dim,)``.
        log_sigma: Per-dimension log-standard-deviation, shape ``(action_dim,)``.
        critic_weights: Value weight vector, shape ``(feature_dim,)``.
        critic_bias: Scalar value bias.
        mean_trace_weights: Trace for mean weights.
        mean_trace_bias: Trace for mean bias.
        log_sigma_trace: Trace for ``log_sigma``.
        critic_trace_weights: Trace for critic weights.
        critic_trace_bias: Trace for critic bias.
        last_observation: Previous observation ``s_t``.
        last_action: Previous (continuous) action vector ``a_t``.
        rng_key: Random key used for action sampling.
        step_count: Saturating int32 compatibility telemetry.
        step_words: Exact big-endian uint32 update identity.
    """

    mean_weights: Float[Array, "action_dim feature_dim"]
    mean_bias: Float[Array, " action_dim"]
    log_sigma: Float[Array, " action_dim"]
    critic_weights: Float[Array, " feature_dim"]
    critic_bias: Float[Array, ""]
    mean_trace_weights: Float[Array, "action_dim feature_dim"]
    mean_trace_bias: Float[Array, " action_dim"]
    log_sigma_trace: Float[Array, " action_dim"]
    critic_trace_weights: Float[Array, " feature_dim"]
    critic_trace_bias: Float[Array, ""]
    last_observation: Float[Array, " feature_dim"]
    last_action: Float[Array, " action_dim"]
    rng_key: Array
    step_count: Int[Array, ""]
    step_words: UInt[Array, " 2"]


@dataclasses.dataclass(frozen=True)
class ContinuousActorCriticResourceBudget:
    """Exact persistent-state and per-update resource accounting."""

    feature_dim: int
    action_dim: int
    trainable_float32_scalars: int
    trace_float32_scalars: int
    transition_float32_scalars: int
    administrative_int32_scalars: int
    exact_update_identity_uint32_scalars: int
    rng_uint32_scalars: int
    state_nbytes: int
    lifetime_identity_bits: int
    telemetry_saturation: int
    max_rng_draws_per_update: int
    learned_float32_scalars_touched_per_update: int
    replay_capacity: int

    def to_dict(self) -> dict[str, int]:
        """Return a JSON-compatible exact resource record."""

        return dataclasses.asdict(self)


@chex.dataclass(frozen=True)
class ContinuousActorCriticUpdateResult:
    """Result from one continuous actor-critic transition update.

    Attributes:
        state: Updated agent state.
        action: Next action vector sampled at the new observation.
        mean: Mean of the policy at the new observation.
        sigma: Standard deviation of the policy.
        value: Value estimate at the previous observation.
        next_value: Value estimate at the new observation.
        td_error: One-step TD error.
        bound_metric: Mean bounder metric, or 1.0 when no bounder is used.
    """

    state: ContinuousActorCriticState
    action: Float[Array, " action_dim"]
    mean: Float[Array, " action_dim"]
    sigma: Float[Array, " action_dim"]
    value: Float[Array, ""]
    next_value: Float[Array, ""]
    td_error: Float[Array, ""]
    bound_metric: Float[Array, ""]
    pre_step_words: UInt[Array, " 2"]
    proposed_step_words: UInt[Array, " 2"]
    post_step_words: UInt[Array, " 2"]
    lifetime_counter_valid: Bool[Array, ""]
    lifetime_capacity_available: Bool[Array, ""]
    source_state_finite: Bool[Array, ""]
    state_valid: Bool[Array, ""]
    input_valid: Bool[Array, ""]
    source_valid: Bool[Array, ""]
    candidate_state_finite: Bool[Array, ""]
    candidate_state_valid: Bool[Array, ""]
    update_applied: Bool[Array, ""]


@chex.dataclass(frozen=True)
class ContinuousActorCriticArrayResult:
    """Result from scan-based continuous actor-critic learning on arrays.

    Attributes:
        state: Final agent state.
        actions: Per-step actions, shape ``(num_steps, action_dim)``.
        means: Per-step policy means, shape ``(num_steps, action_dim)``.
        sigmas: Per-step policy standard deviations, shape ``(num_steps, action_dim)``.
        values: Per-step previous-state value estimates, shape ``(num_steps,)``.
        td_errors: Per-step TD errors, shape ``(num_steps,)``.
    """

    state: ContinuousActorCriticState
    actions: Float[Array, "num_steps action_dim"]
    means: Float[Array, "num_steps action_dim"]
    sigmas: Float[Array, "num_steps action_dim"]
    values: Float[Array, " num_steps"]
    td_errors: Float[Array, " num_steps"]
    pre_step_words: UInt[Array, "num_steps 2"]
    post_step_words: UInt[Array, "num_steps 2"]
    lifetime_counter_valid: Bool[Array, " num_steps"]
    lifetime_capacity_available: Bool[Array, " num_steps"]
    state_valid: Bool[Array, " num_steps"]
    input_valid: Bool[Array, " num_steps"]
    candidate_state_finite: Bool[Array, " num_steps"]
    candidate_state_valid: Bool[Array, " num_steps"]
    update_applied: Bool[Array, " num_steps"]


class ContinuousActorCriticAgent:
    """Linear continuous-action actor-critic with a diagonal-Gaussian policy.

    The actor parameterises a diagonal Gaussian
    ``pi(a | s) = N(mu(s), diag(sigma^2))`` with linear mean
    ``mu(s) = W_mu s + b_mu`` and a state-independent log-sigma vector. The
    critic is a scalar linear value function. Both components carry their own
    accumulating eligibility traces and update at every time step from the
    same TD error, mirroring the discrete ``ActorCriticAgent``.

    Policy gradient. With a Gaussian policy, the score function is

    ``grad_{mu_i} log pi(a | s) = (a_i - mu_i) / sigma_i^2``,

    ``grad_{log_sigma_i} log pi(a | s) = (a_i - mu_i)^2 / sigma_i^2 - 1``.

    These gradients enter the actor traces and are scaled by the TD error
    when applied. ``log_sigma`` is optionally clamped after each update for
    numerical stability and to prevent collapse.
    """

    def __init__(
        self,
        config: ContinuousActorCriticConfig,
        bounder: Bounder | None = None,
    ) -> None:
        """Initialize the continuous actor-critic agent.

        Args:
            config: Continuous actor-critic hyperparameters.
            bounder: Optional update bounder compatible with the framework
                ``Bounder`` ABC. When present, actor and critic proposed steps
                are bounded independently using the TD error.
        """
        if not isinstance(config, ContinuousActorCriticConfig):
            raise TypeError("config must be a ContinuousActorCriticConfig")
        self._config = config
        self._bounder = bounder

    @property
    def config(self) -> ContinuousActorCriticConfig:
        """Continuous actor-critic configuration."""
        return self._config

    @property
    def bounder(self) -> Bounder | None:
        """Optional update bounder."""
        return self._bounder

    def to_config(self) -> dict[str, Any]:
        """Serialize this agent to a dictionary."""
        return {
            "type": "ContinuousActorCriticAgent",
            "state_schema": CONTINUOUS_ACTOR_CRITIC_STATE_SCHEMA,
            "config": self._config.to_config(),
            "bounder": self._bounder.to_config() if self._bounder is not None else None,
        }

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
    ) -> ContinuousActorCriticAgent:
        """Reconstruct a ``ContinuousActorCriticAgent`` from a dictionary."""
        if not isinstance(config, Mapping):
            raise TypeError("continuous actor-critic agent config must be a mapping")
        payload = dict(config)
        expected = {"type", "state_schema", "config", "bounder"}
        if set(payload) != expected:
            if "state_schema" not in payload:
                raise ValueError(
                    "legacy continuous actor-critic agent config requires explicit migration"
                )
            raise ValueError("continuous actor-critic agent config fields do not match v2")
        if payload.pop("type") != "ContinuousActorCriticAgent":
            raise ValueError("unexpected continuous actor-critic agent type")
        if payload.pop("state_schema") != CONTINUOUS_ACTOR_CRITIC_STATE_SCHEMA:
            raise ValueError("continuous actor-critic state schema is unsupported")
        nested = payload.pop("config")
        if not isinstance(nested, Mapping):
            raise TypeError("continuous actor-critic agent config must contain a mapping")
        ac_config = ContinuousActorCriticConfig.from_config(nested)
        bounder_config = payload.pop("bounder")
        if bounder_config is not None and not isinstance(bounder_config, dict):
            raise TypeError("continuous actor-critic bounder config must be a dictionary or null")
        bounder = bounder_from_config(bounder_config) if bounder_config else None
        return cls(config=ac_config, bounder=bounder)

    def _require_state_contract(self, state: ContinuousActorCriticState) -> int:
        """Reject malformed persistent structure before traced arithmetic."""

        if not isinstance(state, ContinuousActorCriticState):
            raise TypeError("state must be a ContinuousActorCriticState")
        observation = jnp.asarray(state.last_observation)
        if observation.ndim != 1 or observation.shape[0] <= 0:
            raise ValueError("state.last_observation must be a nonempty vector")
        feature_dim = observation.shape[0]
        action_dim = self._config.action_dim
        fields = {
            "mean_weights": (
                state.mean_weights,
                (action_dim, feature_dim),
                jnp.float32,
            ),
            "mean_bias": (state.mean_bias, (action_dim,), jnp.float32),
            "log_sigma": (state.log_sigma, (action_dim,), jnp.float32),
            "critic_weights": (state.critic_weights, (feature_dim,), jnp.float32),
            "critic_bias": (state.critic_bias, (), jnp.float32),
            "mean_trace_weights": (
                state.mean_trace_weights,
                (action_dim, feature_dim),
                jnp.float32,
            ),
            "mean_trace_bias": (
                state.mean_trace_bias,
                (action_dim,),
                jnp.float32,
            ),
            "log_sigma_trace": (
                state.log_sigma_trace,
                (action_dim,),
                jnp.float32,
            ),
            "critic_trace_weights": (
                state.critic_trace_weights,
                (feature_dim,),
                jnp.float32,
            ),
            "critic_trace_bias": (state.critic_trace_bias, (), jnp.float32),
            "last_observation": (state.last_observation, (feature_dim,), jnp.float32),
            "last_action": (state.last_action, (action_dim,), jnp.float32),
            "step_count": (state.step_count, (), jnp.int32),
            "step_words": (state.step_words, (2,), jnp.uint32),
        }
        for name, (value, shape, dtype) in fields.items():
            _require_array_contract(
                value,
                name=f"state.{name}",
                shape=shape,
                dtype=dtype,
            )
        _prng_key_contract(state.rng_key, name="state.rng_key")
        return feature_dim

    def _action_within_bounds(self, action: Array) -> Bool[Array, ""]:
        valid = jnp.asarray(True, dtype=jnp.bool_)
        if self._config.action_low is not None:
            valid = valid & jnp.all(action >= self._config.action_low)
        if self._config.action_high is not None:
            valid = valid & jnp.all(action <= self._config.action_high)
        return valid

    def state_is_valid(
        self,
        state: ContinuousActorCriticState,
    ) -> Bool[Array, ""]:
        """Return dynamic state validity after enforcing its static contract."""

        self._require_state_contract(state)
        return (
            _floating_tree_is_finite(state)
            & _lifetime_counter_valid(state.step_words, state.step_count)
            & jnp.all(state.log_sigma >= self._config.log_sigma_min)
            & jnp.all(state.log_sigma <= self._config.log_sigma_max)
            & self._action_within_bounds(state.last_action)
        )

    def resource_budget(
        self,
        feature_dim: int,
    ) -> ContinuousActorCriticResourceBudget:
        """Return exact fixed-state accounting for one feature width."""

        _positive_int(feature_dim, name="feature_dim")
        action_dim = self._config.action_dim
        trainable = action_dim * feature_dim + 2 * action_dim + feature_dim + 1
        traces = trainable
        transition_float32 = feature_dim + action_dim
        administrative_int32 = 1
        exact_words = 2
        rng_words = 2
        state_nbytes = 4 * (
            trainable + traces + transition_float32 + administrative_int32 + exact_words + rng_words
        )
        return ContinuousActorCriticResourceBudget(
            feature_dim=feature_dim,
            action_dim=action_dim,
            trainable_float32_scalars=trainable,
            trace_float32_scalars=traces,
            transition_float32_scalars=transition_float32,
            administrative_int32_scalars=administrative_int32,
            exact_update_identity_uint32_scalars=exact_words,
            rng_uint32_scalars=rng_words,
            state_nbytes=state_nbytes,
            lifetime_identity_bits=64,
            telemetry_saturation=_INT32_MAX,
            max_rng_draws_per_update=1,
            learned_float32_scalars_touched_per_update=(trainable + traces + transition_float32),
            replay_capacity=0,
        )

    def init(self, feature_dim: int, key: Array) -> ContinuousActorCriticState:
        """Initialize actor and critic state.

        Args:
            feature_dim: Input feature dimension.
            key: JAX random key.

        Returns:
            Initial immutable continuous actor-critic state.
        """
        _positive_int(feature_dim, name="feature_dim")
        checked_key = _prng_key_contract(key, name="key")
        cfg = self._config
        zeros_mean = jnp.zeros((cfg.action_dim, feature_dim), dtype=jnp.float32)
        zeros_mean_bias = jnp.zeros((cfg.action_dim,), dtype=jnp.float32)
        log_sigma = jnp.full(
            (cfg.action_dim,),
            cfg.log_sigma_init,
            dtype=jnp.float32,
        )
        zeros_critic = jnp.zeros((feature_dim,), dtype=jnp.float32)
        return ContinuousActorCriticState(  # type: ignore[call-arg]
            mean_weights=zeros_mean,
            mean_bias=zeros_mean_bias,
            log_sigma=log_sigma,
            critic_weights=zeros_critic,
            critic_bias=jnp.array(0.0, dtype=jnp.float32),
            mean_trace_weights=zeros_mean,
            mean_trace_bias=zeros_mean_bias,
            log_sigma_trace=jnp.zeros_like(log_sigma),
            critic_trace_weights=zeros_critic,
            critic_trace_bias=jnp.array(0.0, dtype=jnp.float32),
            last_observation=jnp.zeros((feature_dim,), dtype=jnp.float32),
            last_action=self._maybe_clip_action(jnp.zeros((cfg.action_dim,), dtype=jnp.float32)),
            rng_key=checked_key,
            step_count=jnp.array(0, dtype=jnp.int32),
            step_words=jnp.zeros((2,), dtype=jnp.uint32),
        )

    def _policy_params_unchecked(
        self,
        state: ContinuousActorCriticState,
        observation: Array,
    ) -> tuple[Array, Array]:
        mean = state.mean_weights @ observation + state.mean_bias
        sigma = jnp.exp(state.log_sigma)
        return mean, sigma

    def _value_unchecked(
        self,
        state: ContinuousActorCriticState,
        observation: Array,
    ) -> Array:
        return jnp.dot(state.critic_weights, observation) + state.critic_bias

    def _select_action_unchecked(
        self,
        state: ContinuousActorCriticState,
        observation: Array,
    ) -> tuple[Array, Array, Array, Array]:
        key, sample_key = jr.split(state.rng_key)
        mean, sigma = self._policy_params_unchecked(state, observation)
        noise = jr.normal(sample_key, shape=mean.shape, dtype=jnp.float32)
        raw_action = mean + sigma * noise
        action = self._maybe_clip_action(raw_action)
        return action, key, mean, sigma

    @functools.partial(jax.jit, static_argnums=(0,))
    def policy_params(
        self,
        state: ContinuousActorCriticState,
        observation: Array,
    ) -> tuple[Float[Array, " action_dim"], Float[Array, " action_dim"]]:
        """Compute Gaussian policy mean and standard deviation for one observation."""
        feature_dim = self._require_state_contract(state)
        obs = _require_numeric_source(
            observation,
            name="observation",
            shape=(feature_dim,),
        )
        mean, sigma = self._policy_params_unchecked(state, obs)
        valid = (
            self.state_is_valid(state)
            & jnp.all(jnp.isfinite(obs))
            & jnp.all(jnp.isfinite(mean))
            & jnp.all(jnp.isfinite(sigma))
            & jnp.all(sigma > 0.0)
        )
        return (
            jnp.where(valid, mean, jnp.zeros_like(mean)),
            jnp.where(valid, sigma, jnp.ones_like(sigma)),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def value(
        self,
        state: ContinuousActorCriticState,
        observation: Array,
    ) -> Float[Array, ""]:
        """Compute the critic value estimate for one observation."""
        feature_dim = self._require_state_contract(state)
        obs = _require_numeric_source(
            observation,
            name="observation",
            shape=(feature_dim,),
        )
        prediction = self._value_unchecked(state, obs)
        valid = self.state_is_valid(state) & jnp.all(jnp.isfinite(obs)) & jnp.isfinite(prediction)
        return jnp.where(valid, prediction, jnp.asarray(0.0, dtype=jnp.float32))

    def _maybe_clip_action(self, action: Array) -> Array:
        cfg = self._config
        if cfg.action_low is None and cfg.action_high is None:
            return action
        low = -jnp.inf if cfg.action_low is None else cfg.action_low
        high = jnp.inf if cfg.action_high is None else cfg.action_high
        return jnp.clip(action, low, high)

    @functools.partial(jax.jit, static_argnums=(0,))
    def select_action(
        self,
        state: ContinuousActorCriticState,
        observation: Array,
    ) -> tuple[
        Float[Array, " action_dim"],
        Array,
        Float[Array, " action_dim"],
        Float[Array, " action_dim"],
    ]:
        """Sample one action from the current Gaussian policy.

        Args:
            state: Current agent state.
            observation: Input feature vector.

        Returns:
            Tuple ``(action, new_rng_key, mean, sigma)`` where ``action`` is
            optionally clipped to the configured action bounds.
        """
        feature_dim = self._require_state_contract(state)
        obs = _require_numeric_source(
            observation,
            name="observation",
            shape=(feature_dim,),
        )
        action, key, mean, sigma = self._select_action_unchecked(state, obs)
        _proposed, capacity = _checked_update_words_increment(state.step_words)
        valid = (
            self.state_is_valid(state)
            & capacity
            & jnp.all(jnp.isfinite(obs))
            & jnp.all(jnp.isfinite(action))
            & jnp.all(jnp.isfinite(mean))
            & jnp.all(jnp.isfinite(sigma))
            & jnp.all(sigma > 0.0)
            & self._action_within_bounds(action)
        )
        return (
            jnp.where(valid, action, jnp.zeros_like(action)),
            jax.lax.cond(valid, lambda _: key, lambda _: state.rng_key, operand=None),
            jnp.where(valid, mean, jnp.zeros_like(mean)),
            jnp.where(valid, sigma, jnp.ones_like(sigma)),
        )

    def _start_transaction(
        self,
        state: ContinuousActorCriticState,
        observation: Array,
    ) -> tuple[
        ContinuousActorCriticState,
        Array,
        Array,
        Array,
        Bool[Array, ""],
    ]:
        """Stage one start draw and return its internal commit verdict."""

        action, key, mean, sigma = self._select_action_unchecked(state, observation)
        _proposed, capacity = _checked_update_words_increment(state.step_words)
        candidate = state.replace(  # type: ignore[attr-defined]
            last_observation=observation,
            last_action=action,
            rng_key=key,
        )
        candidate_valid = (
            _floating_tree_is_finite(candidate)
            & self.state_is_valid(candidate)
            & jnp.all(jnp.isfinite(mean))
            & jnp.all(jnp.isfinite(sigma))
            & jnp.all(sigma > 0.0)
        )
        applied = (
            self.state_is_valid(state)
            & capacity
            & jnp.all(jnp.isfinite(observation))
            & candidate_valid
        )
        new_state = jax.lax.cond(
            applied,
            lambda _: candidate,
            lambda _: state,
            operand=None,
        )
        return (
            new_state,
            jnp.where(applied, action, jnp.zeros_like(action)),
            jnp.where(applied, mean, jnp.zeros_like(mean)),
            jnp.where(applied, sigma, jnp.ones_like(sigma)),
            applied,
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def start(
        self,
        state: ContinuousActorCriticState,
        observation: Array,
    ) -> tuple[
        ContinuousActorCriticState,
        Float[Array, " action_dim"],
        Float[Array, " action_dim"],
        Float[Array, " action_dim"],
    ]:
        """Select and store the first action for a new stream or episode."""
        feature_dim = self._require_state_contract(state)
        obs = _require_numeric_source(
            observation,
            name="observation",
            shape=(feature_dim,),
        )
        new_state, action, mean, sigma, _applied = self._start_transaction(
            state,
            obs,
        )
        return new_state, action, mean, sigma

    @functools.partial(jax.jit, static_argnums=(0,))
    def update(
        self,
        state: ContinuousActorCriticState,
        reward: Array,
        observation: Array,
        terminated: Array | None = None,
        discount: Array | None = None,
    ) -> ContinuousActorCriticUpdateResult:
        """Update actor and critic from one transition.

        The transition is ``(state.last_observation, state.last_action,
        reward, observation)`` plus either a scalar transition ``discount`` or
        the legacy ``terminated`` flag. A next action is sampled and stored in
        the returned state for the following update.

        Args:
            state: Current agent state with a valid previous observation/action.
            reward: Scalar reward.
            observation: Next observation.
            terminated: Backward-compatible scalar terminal flag. Non-zero
                maps to transition discount ``0``; false maps to
                ``config.gamma``. Ignored when ``discount`` is provided.
            discount: Optional scalar per-transition discount ``gamma_t``.

        Returns:
            ``ContinuousActorCriticUpdateResult`` containing the updated state.
        """
        cfg = self._config
        feature_dim = self._require_state_contract(state)
        reward_value = _require_numeric_source(reward, name="reward", shape=())
        next_observation = _require_numeric_source(
            observation,
            name="observation",
            shape=(feature_dim,),
        )
        terminal_valid = jnp.asarray(True, dtype=jnp.bool_)
        if discount is None:
            if terminated is None:
                discount_value = jnp.asarray(cfg.gamma, dtype=jnp.float32)
            else:
                terminal_value = _require_terminal_source(
                    terminated,
                    name="terminated",
                )
                terminal_valid = jnp.isfinite(terminal_value) & (
                    (terminal_value == 0.0) | (terminal_value == 1.0)
                )
                discount_value = jnp.where(terminal_value == 1.0, 0.0, cfg.gamma)
        else:
            discount_value = _require_numeric_source(
                discount,
                name="discount",
                shape=(),
            )
        proposed_step_words, lifetime_capacity_available = _checked_update_words_increment(
            state.step_words
        )
        lifetime_counter_valid = _lifetime_counter_valid(
            state.step_words,
            state.step_count,
        )
        source_state_finite = _floating_tree_is_finite(state)
        state_valid = self.state_is_valid(state)
        input_valid = (
            jnp.isfinite(reward_value)
            & jnp.all(jnp.isfinite(next_observation))
            & jnp.isfinite(discount_value)
            & (discount_value >= 0.0)
            & (discount_value <= 1.0)
            & terminal_valid
        )
        source_valid = state_valid & input_valid
        prev_obs = state.last_observation
        action = state.last_action

        prev_mean, prev_sigma = self._policy_params_unchecked(state, prev_obs)
        value = self._value_unchecked(state, prev_obs)
        next_value = self._value_unchecked(state, next_observation)
        bootstrap = discount_value * next_value
        td_error = reward_value + bootstrap - value

        sigma_sq = prev_sigma * prev_sigma + 1e-8
        diff = action - prev_mean
        # Gaussian score function (per-dimension):
        #   grad log pi w.r.t. mean   = diff / sigma^2
        #   grad log pi w.r.t. log_sigma = diff^2 / sigma^2 - 1
        mean_grad_bias = diff / sigma_sq
        mean_grad_weights = mean_grad_bias[:, None] * prev_obs[None, :]
        log_sigma_grad = (diff * diff) / sigma_sq - 1.0

        actor_decay = discount_value * cfg.actor_lamda
        critic_decay = discount_value * cfg.critic_lamda
        mean_trace_weights = actor_decay * state.mean_trace_weights + mean_grad_weights
        mean_trace_bias = actor_decay * state.mean_trace_bias + mean_grad_bias
        log_sigma_trace = actor_decay * state.log_sigma_trace + log_sigma_grad
        critic_trace_weights = critic_decay * state.critic_trace_weights + prev_obs
        critic_trace_bias = critic_decay * state.critic_trace_bias + 1.0

        actor_steps: tuple[Array, ...] = (
            cfg.actor_step_size * mean_trace_weights,
            cfg.actor_step_size * mean_trace_bias,
            cfg.actor_step_size * log_sigma_trace,
        )
        critic_steps: tuple[Array, ...] = (
            cfg.critic_step_size * critic_trace_weights,
            cfg.critic_step_size * critic_trace_bias,
        )
        actor_metric = jnp.array(1.0, dtype=jnp.float32)
        critic_metric = jnp.array(1.0, dtype=jnp.float32)
        if self._bounder is not None:
            actor_steps, actor_metric = self._bounder.bound(
                actor_steps,
                td_error,
                (state.mean_weights, state.mean_bias, state.log_sigma),
            )
            critic_steps, critic_metric = self._bounder.bound(
                critic_steps,
                td_error,
                (state.critic_weights, state.critic_bias),
            )
        actor_steps = tuple(td_error * step for step in actor_steps)
        critic_steps = tuple(td_error * step for step in critic_steps)

        carry_traces = discount_value != 0.0
        stored_mean_trace_weights = jnp.where(
            carry_traces, mean_trace_weights, jnp.zeros_like(mean_trace_weights)
        )
        stored_mean_trace_bias = jnp.where(
            carry_traces, mean_trace_bias, jnp.zeros_like(mean_trace_bias)
        )
        stored_log_sigma_trace = jnp.where(
            carry_traces, log_sigma_trace, jnp.zeros_like(log_sigma_trace)
        )
        stored_critic_trace_weights = jnp.where(
            carry_traces, critic_trace_weights, jnp.zeros_like(critic_trace_weights)
        )
        stored_critic_trace_bias = jnp.where(
            carry_traces, critic_trace_bias, jnp.zeros_like(critic_trace_bias)
        )
        new_log_sigma = jnp.clip(
            state.log_sigma + actor_steps[2],
            cfg.log_sigma_min,
            cfg.log_sigma_max,
        )
        updated = state.replace(  # type: ignore[attr-defined]
            mean_weights=state.mean_weights + actor_steps[0],
            mean_bias=state.mean_bias + actor_steps[1],
            log_sigma=new_log_sigma,
            critic_weights=state.critic_weights + critic_steps[0],
            critic_bias=state.critic_bias + critic_steps[1],
            mean_trace_weights=stored_mean_trace_weights,
            mean_trace_bias=stored_mean_trace_bias,
            log_sigma_trace=stored_log_sigma_trace,
            critic_trace_weights=stored_critic_trace_weights,
            critic_trace_bias=stored_critic_trace_bias,
            step_count=_words_to_saturating_int32(proposed_step_words),
            step_words=proposed_step_words,
        )
        next_action, key, next_mean, next_sigma = self._select_action_unchecked(
            updated,
            next_observation,
        )
        candidate_state = updated.replace(
            last_observation=next_observation,
            last_action=next_action,
            rng_key=key,
        )

        bound_metric = (actor_metric + critic_metric) / 2.0
        reports_finite = (
            jnp.all(jnp.isfinite(next_action))
            & jnp.all(jnp.isfinite(next_mean))
            & jnp.all(jnp.isfinite(next_sigma))
            & jnp.all(next_sigma > 0.0)
            & jnp.isfinite(value)
            & jnp.isfinite(next_value)
            & jnp.isfinite(td_error)
            & jnp.isfinite(bound_metric)
        )
        candidate_state_finite = _floating_tree_is_finite(candidate_state)
        candidate_state_valid = (
            candidate_state_finite
            & self.state_is_valid(candidate_state)
            & jnp.all(candidate_state.step_words == proposed_step_words)
            & reports_finite
        )
        update_applied = (
            lifetime_counter_valid
            & lifetime_capacity_available
            & source_valid
            & candidate_state_valid
        )
        new_state = jax.lax.cond(
            update_applied,
            lambda _: candidate_state,
            lambda _: state,
            operand=None,
        )
        zero = jnp.asarray(0.0, dtype=jnp.float32)
        zero_action = jnp.zeros((cfg.action_dim,), dtype=jnp.float32)
        unit_sigma = jnp.ones((cfg.action_dim,), dtype=jnp.float32)

        return ContinuousActorCriticUpdateResult(  # type: ignore[call-arg]
            state=new_state,
            action=jnp.where(update_applied, next_action, zero_action),
            mean=jnp.where(update_applied, next_mean, zero_action),
            sigma=jnp.where(update_applied, next_sigma, unit_sigma),
            value=jnp.where(update_applied, value, zero),
            next_value=jnp.where(update_applied, next_value, zero),
            td_error=jnp.where(update_applied, td_error, zero),
            bound_metric=jnp.where(update_applied, bound_metric, zero),
            pre_step_words=state.step_words,
            proposed_step_words=proposed_step_words,
            post_step_words=new_state.step_words,
            lifetime_counter_valid=lifetime_counter_valid,
            lifetime_capacity_available=lifetime_capacity_available,
            source_state_finite=source_state_finite,
            state_valid=state_valid,
            input_valid=input_valid,
            source_valid=source_valid,
            candidate_state_finite=candidate_state_finite,
            candidate_state_valid=candidate_state_valid,
            update_applied=update_applied,
        )


def run_continuous_actor_critic_from_arrays(
    agent: ContinuousActorCriticAgent,
    state: ContinuousActorCriticState,
    observations: Float[Array, "num_steps feature_dim"],
    rewards: Float[Array, " num_steps"],
    terminated: Float[Array, " num_steps"] | None,
    next_observations: Float[Array, "num_steps feature_dim"],
    actions: Float[Array, "num_steps action_dim"] | None = None,
    discounts: Float[Array, " num_steps"] | None = None,
) -> ContinuousActorCriticArrayResult:
    """Run continuous actor-critic updates over arrays with ``jax.lax.scan``.

    Mirrors :func:`run_actor_critic_from_arrays` for the continuous-action
    variant. By default the scan is on-policy with respect to the current
    actor; pass ``actions`` to use fixed behavior actions.

    Args:
        agent: Continuous actor-critic agent.
        state: Initial agent state.
        observations: Current observations, shape ``(num_steps, feature_dim)``.
        rewards: Rewards, shape ``(num_steps,)``.
        terminated: Terminal flags, shape ``(num_steps,)``. Required unless
            ``discounts`` is provided.
        next_observations: Next observations, shape ``(num_steps, feature_dim)``.
        actions: Optional fixed current actions, shape ``(num_steps, action_dim)``.
        discounts: Optional transition discounts, shape ``(num_steps,)``.

    Returns:
        ``ContinuousActorCriticArrayResult`` with final state and per-step metrics.
    """
    if not isinstance(agent, ContinuousActorCriticAgent):
        raise TypeError("agent must be a ContinuousActorCriticAgent")
    feature_dim = agent._require_state_contract(state)
    reward_array = jnp.asarray(rewards)
    if reward_array.ndim != 1:
        raise ValueError("rewards must be a vector")
    num_steps = reward_array.shape[0]
    action_dim = agent.config.action_dim
    reward_values = _require_numeric_source(
        rewards,
        name="rewards",
        shape=(num_steps,),
    )
    observation_values = _require_numeric_source(
        observations,
        name="observations",
        shape=(num_steps, feature_dim),
    )
    next_observation_values = _require_numeric_source(
        next_observations,
        name="next_observations",
        shape=(num_steps, feature_dim),
    )
    if terminated is None and discounts is None:
        raise ValueError("terminated or discounts must be provided")
    if terminated is None:
        terminal_values = jnp.zeros((num_steps,), dtype=jnp.float32)
    else:
        terminal_values = _require_terminal_array(
            terminated,
            name="terminated",
            shape=(num_steps,),
        )
    use_explicit_discounts = discounts is not None
    if discounts is None:
        discount_values = jnp.zeros((num_steps,), dtype=jnp.float32)
    else:
        discount_values = _require_numeric_source(
            discounts,
            name="discounts",
            shape=(num_steps,),
        )
    if actions is None:
        action_values = jnp.zeros((num_steps, action_dim), dtype=jnp.float32)
        use_fixed_actions = False
    else:
        action_values = _require_numeric_source(
            actions,
            name="actions",
            shape=(num_steps, action_dim),
        )
        use_fixed_actions = True

    def _scan_fn(
        carry: ContinuousActorCriticState,
        inputs: tuple[Array, Array, Array, Array, Array, Array],
    ) -> tuple[ContinuousActorCriticState, tuple[Array, ...]]:
        obs, reward, terminal, term_discount, next_obs, fixed_action = inputs
        if use_fixed_actions:
            started_state = carry.replace(  # type: ignore[attr-defined]
                last_observation=obs,
                last_action=fixed_action,
            )
            current_action = fixed_action
            current_mean, current_sigma = agent._policy_params_unchecked(
                started_state,
                obs,
            )
            behavior_action_valid = jnp.all(
                jnp.isfinite(fixed_action)
            ) & agent._action_within_bounds(fixed_action)
            row_preflight_valid = jnp.all(jnp.isfinite(obs)) & behavior_action_valid
        else:
            (
                started_state,
                current_action,
                current_mean,
                current_sigma,
                row_preflight_valid,
            ) = agent._start_transaction(carry, obs)
        if use_explicit_discounts:
            result = agent.update(
                started_state,
                reward,
                next_obs,
                discount=term_discount,
            )
        else:
            result = agent.update(
                started_state,
                reward,
                next_obs,
                terminated=terminal,
            )
        accepted = result.update_applied & row_preflight_valid
        committed_state = jax.lax.cond(
            accepted,
            lambda _: result.state,
            lambda _: carry,
            operand=None,
        )
        row_input_valid = result.input_valid & row_preflight_valid
        zero = jnp.asarray(0.0, dtype=jnp.float32)
        return committed_state, (
            jnp.where(accepted, current_action, jnp.zeros_like(current_action)),
            jnp.where(accepted, current_mean, jnp.zeros_like(current_mean)),
            jnp.where(accepted, current_sigma, jnp.ones_like(current_sigma)),
            jnp.where(accepted, result.value, zero),
            jnp.where(accepted, result.td_error, zero),
            result.pre_step_words,
            committed_state.step_words,
            result.lifetime_counter_valid,
            result.lifetime_capacity_available,
            result.state_valid,
            row_input_valid,
            result.candidate_state_finite & row_preflight_valid,
            result.candidate_state_valid & row_preflight_valid,
            accepted,
        )

    final_state, outputs = jax.lax.scan(
        _scan_fn,
        state,
        (
            observation_values,
            reward_values,
            terminal_values,
            discount_values,
            next_observation_values,
            action_values,
        ),
    )
    (
        actions_out,
        means_out,
        sigmas_out,
        values,
        td_errors,
        pre_step_words,
        post_step_words,
        lifetime_counter_valid,
        lifetime_capacity_available,
        state_valid,
        input_valid,
        candidate_state_finite,
        candidate_state_valid,
        update_applied,
    ) = outputs
    return ContinuousActorCriticArrayResult(  # type: ignore[call-arg]
        state=final_state,
        actions=actions_out,
        means=means_out,
        sigmas=sigmas_out,
        values=values,
        td_errors=td_errors,
        pre_step_words=pre_step_words,
        post_step_words=post_step_words,
        lifetime_counter_valid=lifetime_counter_valid,
        lifetime_capacity_available=lifetime_capacity_available,
        state_valid=state_valid,
        input_valid=input_valid,
        candidate_state_finite=candidate_state_finite,
        candidate_state_valid=candidate_state_valid,
        update_applied=update_applied,
    )


def actor_critic_lifetime_counter_nbytes() -> int:
    """Return bytes occupied by discrete telemetry plus exact identity."""

    return ACTOR_CRITIC_LIFETIME_COUNTER_NBYTES


def continuous_actor_critic_lifetime_counter_nbytes() -> int:
    """Return bytes occupied by continuous telemetry plus exact identity."""

    return CONTINUOUS_ACTOR_CRITIC_LIFETIME_COUNTER_NBYTES


def measure_actor_critic_state_nbytes(state: ActorCriticState) -> int:
    """Measure every persistent JAX-array byte in one discrete state."""

    if not isinstance(state, ActorCriticState):
        raise TypeError("state must be an ActorCriticState")
    return _measure_array_tree_nbytes(state)


def measure_continuous_actor_critic_state_nbytes(
    state: ContinuousActorCriticState,
) -> int:
    """Measure every persistent JAX-array byte in one continuous state."""

    if not isinstance(state, ContinuousActorCriticState):
        raise TypeError("state must be a ContinuousActorCriticState")
    return _measure_array_tree_nbytes(state)


def migrate_legacy_actor_critic_agent_config(
    legacy_config: Mapping[str, Any],
) -> ActorCriticAgent:
    """Explicitly migrate the exact pre-v2 discrete agent payload."""

    if not isinstance(legacy_config, Mapping):
        raise TypeError("legacy actor-critic agent config must be a mapping")
    payload = dict(legacy_config)
    expected = {"type", "config", "bounder"}
    if set(payload) != expected:
        missing = sorted(expected - set(payload))
        extra = sorted(set(payload) - expected)
        raise ValueError(
            "legacy actor-critic agent config fields are not exact; "
            f"missing={missing}, extra={extra}"
        )
    if payload["type"] != "ActorCriticAgent":
        raise ValueError("legacy actor-critic agent type is unsupported")
    nested = payload["config"]
    if not isinstance(nested, Mapping):
        raise TypeError("legacy actor-critic config must be a mapping")
    bounder_config = payload["bounder"]
    if bounder_config is not None and not isinstance(bounder_config, dict):
        raise TypeError("legacy actor-critic bounder must be a dictionary or null")
    return ActorCriticAgent(
        migrate_legacy_actor_critic_config(nested),
        bounder=bounder_from_config(bounder_config) if bounder_config else None,
    )


def migrate_legacy_continuous_actor_critic_agent_config(
    legacy_config: Mapping[str, Any],
) -> ContinuousActorCriticAgent:
    """Explicitly migrate the exact pre-v2 continuous agent payload."""

    if not isinstance(legacy_config, Mapping):
        raise TypeError("legacy continuous actor-critic agent config must be a mapping")
    payload = dict(legacy_config)
    expected = {"type", "config", "bounder"}
    if set(payload) != expected:
        missing = sorted(expected - set(payload))
        extra = sorted(set(payload) - expected)
        raise ValueError(
            "legacy continuous actor-critic agent config fields are not exact; "
            f"missing={missing}, extra={extra}"
        )
    if payload["type"] != "ContinuousActorCriticAgent":
        raise ValueError("legacy continuous actor-critic agent type is unsupported")
    nested = payload["config"]
    if not isinstance(nested, Mapping):
        raise TypeError("legacy continuous actor-critic config must be a mapping")
    bounder_config = payload["bounder"]
    if bounder_config is not None and not isinstance(bounder_config, dict):
        raise TypeError("legacy continuous actor-critic bounder must be a dictionary or null")
    return ContinuousActorCriticAgent(
        migrate_legacy_continuous_actor_critic_config(nested),
        bounder=bounder_from_config(bounder_config) if bounder_config else None,
    )


def _legacy_state_with_step_words(
    legacy_state: Any,
    *,
    state_type: type[Any],
    name: str,
) -> dict[str, Any]:
    fields = _legacy_fields(legacy_state, name=name)
    current_names = {field.name for field in dataclasses.fields(cast(Any, state_type))}
    legacy_names = current_names - {"step_words"}
    if set(fields) != legacy_names:
        missing = sorted(legacy_names - set(fields))
        extra = sorted(set(fields) - legacy_names)
        raise ValueError(f"legacy {name} fields are not exact; missing={missing}, extra={extra}")
    step_count = jnp.asarray(fields["step_count"])
    if step_count.shape != () or step_count.dtype != jnp.dtype(jnp.int32):
        raise TypeError(f"legacy {name} step_count must be scalar int32")
    step = int(jax.device_get(step_count))
    if step < 0:
        raise ValueError(f"negative legacy {name} step_count indicates wrap")
    if step >= _INT32_MAX:
        raise ValueError(f"saturated legacy {name} step_count is ambiguous")
    fields["step_words"] = jnp.asarray((0, step), dtype=jnp.uint32)
    return fields


def migrate_legacy_actor_critic_state(
    agent: ActorCriticAgent,
    legacy_state: Any,
) -> ActorCriticState:
    """Migrate only an exact, unsaturated pre-v2 discrete state."""

    if not isinstance(agent, ActorCriticAgent):
        raise TypeError("agent must be an ActorCriticAgent")
    fields = _legacy_state_with_step_words(
        legacy_state,
        state_type=ActorCriticState,
        name="actor-critic state",
    )
    state = ActorCriticState(**fields)
    agent._require_state_contract(state)
    if not bool(jax.device_get(agent.state_is_valid(state))):
        raise ValueError("legacy actor-critic state is dynamically invalid")
    return state


def migrate_legacy_continuous_actor_critic_state(
    agent: ContinuousActorCriticAgent,
    legacy_state: Any,
) -> ContinuousActorCriticState:
    """Migrate only an exact, unsaturated pre-v2 continuous state."""

    if not isinstance(agent, ContinuousActorCriticAgent):
        raise TypeError("agent must be a ContinuousActorCriticAgent")
    fields = _legacy_state_with_step_words(
        legacy_state,
        state_type=ContinuousActorCriticState,
        name="continuous actor-critic state",
    )
    state = ContinuousActorCriticState(**fields)
    agent._require_state_contract(state)
    if not bool(jax.device_get(agent.state_is_valid(state))):
        raise ValueError("legacy continuous actor-critic state is dynamically invalid")
    return state


__all__ = [
    "ACTOR_CRITIC_CONFIG_SCHEMA",
    "ACTOR_CRITIC_EXACT_UPDATE_IDENTITY_NBYTES",
    "ACTOR_CRITIC_LIFETIME_COUNTER_NBYTES",
    "ACTOR_CRITIC_STATE_SCHEMA",
    "CONTINUOUS_ACTOR_CRITIC_CONFIG_SCHEMA",
    "CONTINUOUS_ACTOR_CRITIC_EXACT_UPDATE_IDENTITY_NBYTES",
    "CONTINUOUS_ACTOR_CRITIC_LIFETIME_COUNTER_NBYTES",
    "CONTINUOUS_ACTOR_CRITIC_STATE_SCHEMA",
    "ActorCriticAgent",
    "ActorCriticArrayResult",
    "ActorCriticConfig",
    "ActorCriticResourceBudget",
    "ActorCriticState",
    "ActorCriticUpdateResult",
    "ContinuousActorCriticAgent",
    "ContinuousActorCriticArrayResult",
    "ContinuousActorCriticConfig",
    "ContinuousActorCriticResourceBudget",
    "ContinuousActorCriticState",
    "ContinuousActorCriticUpdateResult",
    "actor_critic_lifetime_counter_nbytes",
    "continuous_actor_critic_lifetime_counter_nbytes",
    "measure_actor_critic_state_nbytes",
    "measure_continuous_actor_critic_state_nbytes",
    "migrate_legacy_actor_critic_agent_config",
    "migrate_legacy_actor_critic_config",
    "migrate_legacy_actor_critic_state",
    "migrate_legacy_continuous_actor_critic_agent_config",
    "migrate_legacy_continuous_actor_critic_config",
    "migrate_legacy_continuous_actor_critic_state",
    "run_actor_critic_from_arrays",
    "run_continuous_actor_critic_from_arrays",
]
