# mypy: disable-error-code="call-arg,name-defined"
r"""Causal state-construction contracts and small reference implementations.

The existing Alberta components expose several useful forms of history:
``HistoryFeatureExtractor`` and ``WorkingMemoryFeaturizer`` provide fixed
traces, while ``PrototypeAgent`` optionally uses a fixed-weight echo-state
GRU.  This module gives those design points a narrow common contract and adds
one genuinely trainable recurrent reference.

The trainable reference is deliberately modest.  It is a diagonal bank of
write/hold units,

.. math::

    g_t &= \sigma(W_g u_t + b_g) \\
    c_t &= \tanh(W_c u_t + b_c) \\
    h_t &= (1-g_t) h_{t-1} + g_t c_t ,

where ``u_t`` contains the current observation and the preceding transition's
action, reward, and discount.  It carries an RTRL-style online eligibility
matrix.  With parameters held fixed, that matrix is the exact unrolled
``dh_t / dtheta``; after an online parameter update, carrying it forward is the
usual changing-parameter eligibility approximation.  A downstream prediction
or control head can therefore pass the gradient of its loss with respect to
the emitted state to :meth:`OnlineGatedStateBuilder.learn` without replay or a
backward sweep through stored experience.

This is a learnable recurrent baseline, not a general state-discovery result:

* units do not interact recurrently with one another;
* the caller must supply useful online auxiliary/control gradients;
* carried sensitivities are a fixed, potentially substantial memory cost; and
* there is no generate-and-test or feature-recycling mechanism here.

All builders are pure PyTree transformations, have fixed output/state budgets,
serialize their configuration, and round-trip through Alberta's generic Orbax
checkpoint utilities.
"""

from __future__ import annotations

import functools
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, TypeVar, cast, runtime_checkable

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import Array
from jaxtyping import Float

from alberta_framework.core.checkpoints import (
    load_checkpoint,
    load_checkpoint_metadata,
    save_checkpoint,
)
from alberta_framework.core.working_memory import (
    WorkingMemoryConfig,
    WorkingMemoryFeaturizer,
    WorkingMemoryState,
)

_INT32_MAX = 2**31 - 1


def _saturating_int32_increment(value: Array) -> Array:
    maximum = jnp.asarray(_INT32_MAX, dtype=jnp.int32)
    counter = jnp.asarray(value, dtype=jnp.int32)
    return jnp.minimum(jnp.maximum(counter, 0), maximum - 1) + 1

StateT = TypeVar("StateT")

STATE_BUILDER_CHECKPOINT_SCHEMA = "alberta.state_builder.v1"


@dataclass(frozen=True)
class StateBuilderBudget:
    """Exact history-independent resource counts for a state builder.

    ``state_scalars`` counts every scalar carried in the builder state,
    including parameters and integer counters.  ``state_bytes`` assumes the
    implementations in this module: float32 and int32 arrays throughout.
    It excludes transient compiler buffers and a downstream learning head.
    """

    output_scalars: int
    trainable_scalars: int
    state_scalars: int
    state_bytes: int

    def to_config(self) -> dict[str, int]:
        """Return a JSON-compatible budget description."""
        return asdict(self)


@chex.dataclass(frozen=True)
class StateBuilderLearningDiagnostics:
    """Diagnostics from one representation-learning update."""

    gradient_norm: Float[Array, ""]
    clipped_gradient_norm: Float[Array, ""]
    parameter_update_norm: Float[Array, ""]


@runtime_checkable
class StateBuilder(Protocol[StateT]):
    """Minimal causal state-construction contract.

    ``start`` consumes the first observation and optional preceding-transition
    values.  Thereafter, ``update`` consumes ``(current_observation,
    previous_action, previous_reward, previous_discount)`` and advances state
    exactly once.  The action and outcomes must be from the transition that
    produced the current observation; passing a current action before selecting
    it would leak future information.  Both methods return the representation
    associated with the supplied current observation.

    ``encode`` is pure: it pairs a supplied raw observation with the recurrent
    memory already present in ``state`` and never advances history.  For
    history-dependent builders it is valid for the most recently consumed
    observation (or as an explicitly counterfactual raw-observation query);
    callers must not mistake it for a second recurrent transition.

    ``learn`` accepts ``d(loss) / d(representation)`` from any downstream head.
    Builders without trainable representation parameters implement it as a
    no-op.  Separating ``update`` and ``learn`` prevents target information
    from entering the emitted state before its prediction is scored.
    """

    def init(self, key: Array) -> StateT:
        """Return a fresh builder state."""
        ...

    def start(
        self,
        state: StateT,
        raw_observation: Array,
        last_action: Array | int = -1,
        last_reward: Array | float = 0.0,
        last_discount: Array | float = 1.0,
    ) -> tuple[StateT, Array]:
        """Consume the first observation and emit its representation."""
        ...

    def encode(self, state: StateT, raw_observation: Array) -> Array:
        """Emit a representation without advancing state."""
        ...

    def update(
        self,
        state: StateT,
        raw_observation: Array,
        previous_action: Array | int,
        previous_reward: Array | float,
        previous_discount: Array | float,
    ) -> tuple[StateT, Array]:
        """Consume one continuing transition and emit the new representation."""
        ...

    def learn(
        self,
        state: StateT,
        representation_gradient: Array,
    ) -> tuple[StateT, StateBuilderLearningDiagnostics]:
        """Apply an online representation gradient."""
        ...

    def feature_dim(self) -> int:
        """Return the fixed representation dimension."""
        ...

    def resource_budget(self) -> StateBuilderBudget:
        """Return exact persistent-state and trainable-parameter counts."""
        ...

    def to_config(self) -> dict[str, Any]:
        """Return a JSON-compatible builder configuration."""
        ...


def _zero_learning_diagnostics() -> StateBuilderLearningDiagnostics:
    zero = jnp.asarray(0.0, dtype=jnp.float32)
    return StateBuilderLearningDiagnostics(
        gradient_norm=zero,
        clipped_gradient_norm=zero,
        parameter_update_norm=zero,
    )


def _validate_observation_dim(observation_dim: int) -> None:
    if observation_dim < 1:
        raise ValueError("observation_dim must be positive")


def _action_features(action: Array | int, n_actions: int) -> Array:
    if n_actions == 0:
        return jnp.zeros((0,), dtype=jnp.float32)
    action_id = jnp.asarray(action, dtype=jnp.int32)
    return jax.nn.one_hot(action_id, n_actions, dtype=jnp.float32)


@dataclass(frozen=True)
class IdentityStateBuilderConfig:
    """Configuration for the observation-only state baseline."""

    observation_dim: int

    def __post_init__(self) -> None:
        _validate_observation_dim(self.observation_dim)

    def to_config(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "type": "IdentityStateBuilder",
            "observation_dim": self.observation_dim,
        }

    @classmethod
    def from_config(cls, payload: dict[str, Any]) -> IdentityStateBuilderConfig:
        """Reconstruct from :meth:`to_config` output."""
        data = dict(payload)
        data.pop("type", None)
        return cls(observation_dim=int(data["observation_dim"]))


@chex.dataclass(frozen=True)
class IdentityStateBuilderState:
    """The observation-only baseline carries only a continuing step counter."""

    step_count: Array


class IdentityStateBuilder:
    """Observation-only state builder and lower memory control."""

    def __init__(self, config: IdentityStateBuilderConfig):
        self._config = config

    @property
    def config(self) -> IdentityStateBuilderConfig:
        """Return the immutable configuration."""
        return self._config

    def init(self, key: Array) -> IdentityStateBuilderState:
        """Return a fresh state; ``key`` is accepted for protocol parity."""
        del key
        return IdentityStateBuilderState(step_count=jnp.asarray(0, dtype=jnp.int32))

    def feature_dim(self) -> int:
        """Return the raw observation dimension."""
        return self._config.observation_dim

    def resource_budget(self) -> StateBuilderBudget:
        """Return the one-counter persistent budget."""
        return StateBuilderBudget(
            output_scalars=self.feature_dim(),
            trainable_scalars=0,
            state_scalars=1,
            state_bytes=4,
        )

    def to_config(self) -> dict[str, Any]:
        """Serialize the builder configuration."""
        return self._config.to_config()

    @functools.partial(jax.jit, static_argnums=(0,))
    def encode(
        self,
        state: IdentityStateBuilderState,
        raw_observation: Array,
    ) -> Float[Array, " observation_dim"]:
        """Return the raw observation without touching ``state``."""
        del state
        return jnp.asarray(raw_observation, dtype=jnp.float32).reshape(
            (self._config.observation_dim,)
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def update(
        self,
        state: IdentityStateBuilderState,
        raw_observation: Array,
        previous_action: Array | int,
        previous_reward: Array | float,
        previous_discount: Array | float,
    ) -> tuple[IdentityStateBuilderState, Float[Array, " observation_dim"]]:
        """Advance the counter and emit the raw observation."""
        del previous_action, previous_reward, previous_discount
        features = self.encode(state, raw_observation)
        next_state = IdentityStateBuilderState(
            step_count=_saturating_int32_increment(state.step_count)
        )
        return next_state, features

    def start(
        self,
        state: IdentityStateBuilderState,
        raw_observation: Array,
        last_action: Array | int = -1,
        last_reward: Array | float = 0.0,
        last_discount: Array | float = 1.0,
    ) -> tuple[IdentityStateBuilderState, Array]:
        """Consume the first observation."""
        return cast(
            tuple[IdentityStateBuilderState, Array],
            self.update(
                state,
                raw_observation,
                last_action,
                last_reward,
                last_discount,
            ),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def learn(
        self,
        state: IdentityStateBuilderState,
        representation_gradient: Array,
    ) -> tuple[IdentityStateBuilderState, StateBuilderLearningDiagnostics]:
        """Ignore the gradient because the representation is fixed."""
        jnp.asarray(representation_gradient, dtype=jnp.float32).reshape((self.feature_dim(),))
        return state, _zero_learning_diagnostics()


@dataclass(frozen=True)
class FixedTraceStateBuilderConfig:
    """Configuration for a fixed observation/action/outcome trace bank."""

    observation_dim: int
    n_actions: int = 0
    observation_decay_rates: tuple[float, ...] = (0.5, 0.9, 0.99)
    action_decay_rates: tuple[float, ...] = (0.5, 0.9)
    outcome_decay_rates: tuple[float, ...] = (0.5, 0.9)
    include_raw_observation: bool = True

    def __post_init__(self) -> None:
        _validate_observation_dim(self.observation_dim)
        if self.n_actions < 0:
            raise ValueError("n_actions must be non-negative")
        for name, rates in (
            ("observation_decay_rates", self.observation_decay_rates),
            ("action_decay_rates", self.action_decay_rates),
            ("outcome_decay_rates", self.outcome_decay_rates),
        ):
            if any(not math.isfinite(rate) or rate < 0.0 or rate >= 1.0 for rate in rates):
                raise ValueError(f"{name} must contain finite values in [0, 1)")
        if (
            not self.include_raw_observation
            and not self.observation_decay_rates
            and not self.action_decay_rates
            and not self.outcome_decay_rates
        ):
            raise ValueError("configuration must emit at least one feature")

    def to_config(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "type": "FixedTraceStateBuilder",
            "observation_dim": self.observation_dim,
            "n_actions": self.n_actions,
            "observation_decay_rates": list(self.observation_decay_rates),
            "action_decay_rates": list(self.action_decay_rates),
            "outcome_decay_rates": list(self.outcome_decay_rates),
            "include_raw_observation": self.include_raw_observation,
        }

    @classmethod
    def from_config(cls, payload: dict[str, Any]) -> FixedTraceStateBuilderConfig:
        """Reconstruct from :meth:`to_config` output."""
        data = dict(payload)
        data.pop("type", None)
        return cls(
            observation_dim=int(data["observation_dim"]),
            n_actions=int(data.get("n_actions", 0)),
            observation_decay_rates=tuple(data.get("observation_decay_rates", ())),
            action_decay_rates=tuple(data.get("action_decay_rates", ())),
            outcome_decay_rates=tuple(data.get("outcome_decay_rates", ())),
            include_raw_observation=bool(data.get("include_raw_observation", True)),
        )


class FixedTraceStateBuilder:
    """Fixed multi-timescale trace baseline using ``WorkingMemoryFeaturizer``.

    Reward and discount are treated as a two-channel outcome vector.  Returned
    traces are post-update, so :meth:`encode` reproduces the current recurrent
    state without applying a transition twice.
    """

    def __init__(self, config: FixedTraceStateBuilderConfig):
        self._config = config
        self._memory = WorkingMemoryFeaturizer(
            WorkingMemoryConfig(
                observation_dim=config.observation_dim,
                action_dim=config.n_actions,
                reward_dim=2,
                observation_decay_rates=config.observation_decay_rates,
                action_decay_rates=config.action_decay_rates,
                reward_decay_rates=config.outcome_decay_rates,
                include_current_observation=config.include_raw_observation,
                include_current_action=False,
                include_current_reward=False,
                include_traces=True,
                include_innovations=False,
            )
        )

    @property
    def config(self) -> FixedTraceStateBuilderConfig:
        """Return the immutable configuration."""
        return self._config

    def init(self, key: Array) -> WorkingMemoryState:
        """Return an all-zero trace state; ``key`` is unused."""
        del key
        return self._memory.init()

    def feature_dim(self) -> int:
        """Return the fixed trace representation dimension."""
        return int(self._memory.feature_dim())

    def resource_budget(self) -> StateBuilderBudget:
        """Return exact trace-bank and counter storage."""
        cfg = self._config
        trace_scalars = (
            cfg.observation_dim * len(cfg.observation_decay_rates)
            + cfg.n_actions * len(cfg.action_decay_rates)
            + 2 * len(cfg.outcome_decay_rates)
        )
        state_scalars = trace_scalars + 4  # step_count and three last gates
        return StateBuilderBudget(
            output_scalars=self.feature_dim(),
            trainable_scalars=0,
            state_scalars=state_scalars,
            state_bytes=4 * state_scalars,
        )

    def to_config(self) -> dict[str, Any]:
        """Serialize the builder configuration."""
        return self._config.to_config()

    @functools.partial(jax.jit, static_argnums=(0,))
    def encode(
        self,
        state: WorkingMemoryState,
        raw_observation: Array,
    ) -> Float[Array, " feature_dim"]:
        """Combine a raw observation with the already-current trace state."""
        observation = jnp.asarray(raw_observation, dtype=jnp.float32).reshape(
            (self._config.observation_dim,)
        )
        return cast(
            Float[Array, " feature_dim"],
            self._memory.features(
                state,
                observation,
                self._memory.zero_action(),
                self._memory.zero_reward(),
            ),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def update(
        self,
        state: WorkingMemoryState,
        raw_observation: Array,
        previous_action: Array | int,
        previous_reward: Array | float,
        previous_discount: Array | float,
    ) -> tuple[WorkingMemoryState, Float[Array, " feature_dim"]]:
        """Advance all trace banks and emit the post-update memory state."""
        observation = jnp.asarray(raw_observation, dtype=jnp.float32).reshape(
            (self._config.observation_dim,)
        )
        action_vector = _action_features(previous_action, self._config.n_actions)
        outcomes = jnp.stack(
            [
                jnp.asarray(previous_reward, dtype=jnp.float32),
                jnp.asarray(previous_discount, dtype=jnp.float32),
            ]
        )
        next_state = self._memory.update(
            state,
            observation,
            action_vector,
            outcomes,
        )
        return next_state, self.encode(next_state, observation)

    def start(
        self,
        state: WorkingMemoryState,
        raw_observation: Array,
        last_action: Array | int = -1,
        last_reward: Array | float = 0.0,
        last_discount: Array | float = 1.0,
    ) -> tuple[WorkingMemoryState, Array]:
        """Consume the first observation and seed the trace bank."""
        return cast(
            tuple[WorkingMemoryState, Array],
            self.update(
                state,
                raw_observation,
                last_action,
                last_reward,
                last_discount,
            ),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def learn(
        self,
        state: WorkingMemoryState,
        representation_gradient: Array,
    ) -> tuple[WorkingMemoryState, StateBuilderLearningDiagnostics]:
        """Ignore the gradient because trace dynamics are fixed."""
        jnp.asarray(representation_gradient, dtype=jnp.float32).reshape((self.feature_dim(),))
        return state, _zero_learning_diagnostics()


@dataclass(frozen=True)
class OnlineGatedStateBuilderConfig:
    """Configuration for the online learnable write/hold state builder."""

    observation_dim: int
    n_actions: int = 0
    hidden_dim: int = 8
    step_size: float = 0.01
    gradient_clip: float = 10.0
    initial_gate_bias: float = -2.0
    initialization_scale: float = 0.2
    include_raw_observation: bool = True

    def __post_init__(self) -> None:
        _validate_observation_dim(self.observation_dim)
        if self.n_actions < 0:
            raise ValueError("n_actions must be non-negative")
        if self.hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        if not math.isfinite(self.step_size) or self.step_size <= 0.0:
            raise ValueError("step_size must be finite and positive")
        if not math.isfinite(self.gradient_clip) or self.gradient_clip <= 0.0:
            raise ValueError("gradient_clip must be finite and positive")
        if not math.isfinite(self.initial_gate_bias):
            raise ValueError("initial_gate_bias must be finite")
        if not math.isfinite(self.initialization_scale) or self.initialization_scale <= 0.0:
            raise ValueError("initialization_scale must be finite and positive")

    def event_dim(self) -> int:
        """Return observation + one-hot action + reward + discount width."""
        return self.observation_dim + self.n_actions + 2

    def parameter_count(self) -> int:
        """Return write/candidate weights and biases."""
        return 2 * self.hidden_dim * (self.event_dim() + 1)

    def feature_dim(self) -> int:
        """Return raw-observation plus hidden-state width."""
        return self.hidden_dim + (self.observation_dim if self.include_raw_observation else 0)

    def to_config(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        payload = asdict(self)
        payload["type"] = "OnlineGatedStateBuilder"
        return payload

    @classmethod
    def from_config(cls, payload: dict[str, Any]) -> OnlineGatedStateBuilderConfig:
        """Reconstruct from :meth:`to_config` output."""
        data = dict(payload)
        data.pop("type", None)
        return cls(**data)


@chex.dataclass(frozen=True)
class OnlineGatedStateBuilderState:
    """Parameters, recurrent state, and RTRL-style eligibility sensitivities."""

    parameters: Float[Array, " parameter_count"]
    hidden: Float[Array, " hidden_dim"]
    parameter_sensitivity: Float[Array, "hidden_dim parameter_count"]
    step_count: Array
    update_count: Array
    last_gradient_norm: Float[Array, ""]


class OnlineGatedStateBuilder:
    """Learnable gated recurrent state with an online sensitivity trace.

    The recurrence is advanced by :meth:`start`/:meth:`update`.  Learning is a
    separate call because a prediction must be scored before its target-derived
    gradient is allowed to modify the representation.  ``learn`` updates only
    recurrent parameters; a downstream head owns and checkpoints its own
    parameters. Sensitivities are exact for an unroll with fixed parameters.
    Carrying them across :meth:`learn` calls is an online eligibility
    approximation; it is not the derivative of the stored hidden state with
    respect to the newly updated parameter vector.
    """

    def __init__(self, config: OnlineGatedStateBuilderConfig):
        self._config = config

    @property
    def config(self) -> OnlineGatedStateBuilderConfig:
        """Return the immutable configuration."""
        return self._config

    def feature_dim(self) -> int:
        """Return the fixed emitted representation width."""
        return self._config.feature_dim()

    def resource_budget(self) -> StateBuilderBudget:
        """Return exact persistent state, including recurrent sensitivities."""
        parameter_count = self._config.parameter_count()
        state_scalars = (
            parameter_count
            + self._config.hidden_dim
            + self._config.hidden_dim * parameter_count
            + 3
        )
        return StateBuilderBudget(
            output_scalars=self.feature_dim(),
            trainable_scalars=parameter_count,
            state_scalars=state_scalars,
            state_bytes=4 * state_scalars,
        )

    def to_config(self) -> dict[str, Any]:
        """Serialize the builder configuration."""
        return self._config.to_config()

    def init(self, key: Array) -> OnlineGatedStateBuilderState:
        """Initialize trainable parameters, hidden state, and sensitivities."""
        gate_key, candidate_key = jr.split(key)
        cfg = self._config
        event_dim = cfg.event_dim()
        hidden_dim = cfg.hidden_dim
        scale = jnp.asarray(cfg.initialization_scale, dtype=jnp.float32)
        gate_weights = scale * jr.normal(
            gate_key,
            (hidden_dim, event_dim),
            dtype=jnp.float32,
        )
        gate_bias = jnp.full(
            (hidden_dim,),
            cfg.initial_gate_bias,
            dtype=jnp.float32,
        )
        candidate_weights = scale * jr.normal(
            candidate_key,
            (hidden_dim, event_dim),
            dtype=jnp.float32,
        )
        candidate_bias = jnp.zeros((hidden_dim,), dtype=jnp.float32)
        parameters = jnp.concatenate(
            [
                gate_weights.reshape(-1),
                gate_bias,
                candidate_weights.reshape(-1),
                candidate_bias,
            ]
        )
        return OnlineGatedStateBuilderState(
            parameters=parameters,
            hidden=jnp.zeros((hidden_dim,), dtype=jnp.float32),
            parameter_sensitivity=jnp.zeros(
                (hidden_dim, cfg.parameter_count()),
                dtype=jnp.float32,
            ),
            step_count=jnp.asarray(0, dtype=jnp.int32),
            update_count=jnp.asarray(0, dtype=jnp.int32),
            last_gradient_norm=jnp.asarray(0.0, dtype=jnp.float32),
        )

    def _unpack_parameters(
        self,
        parameters: Array,
    ) -> tuple[Array, Array, Array, Array]:
        cfg = self._config
        matrix_size = cfg.hidden_dim * cfg.event_dim()
        offset = 0
        gate_weights = parameters[offset : offset + matrix_size].reshape(
            (cfg.hidden_dim, cfg.event_dim())
        )
        offset += matrix_size
        gate_bias = parameters[offset : offset + cfg.hidden_dim]
        offset += cfg.hidden_dim
        candidate_weights = parameters[offset : offset + matrix_size].reshape(
            (cfg.hidden_dim, cfg.event_dim())
        )
        offset += matrix_size
        candidate_bias = parameters[offset : offset + cfg.hidden_dim]
        return gate_weights, gate_bias, candidate_weights, candidate_bias

    def _transition(self, parameters: Array, hidden: Array, event: Array) -> Array:
        gate_weights, gate_bias, candidate_weights, candidate_bias = self._unpack_parameters(
            parameters
        )
        gate = jax.nn.sigmoid(gate_weights @ event + gate_bias)
        candidate = jnp.tanh(candidate_weights @ event + candidate_bias)
        return hidden + gate * (candidate - hidden)

    def _event(
        self,
        raw_observation: Array,
        previous_action: Array | int,
        previous_reward: Array | float,
        previous_discount: Array | float,
    ) -> Array:
        observation = jnp.asarray(raw_observation, dtype=jnp.float32).reshape(
            (self._config.observation_dim,)
        )
        return jnp.concatenate(
            [
                observation,
                _action_features(previous_action, self._config.n_actions),
                jnp.atleast_1d(jnp.asarray(previous_reward, dtype=jnp.float32)),
                jnp.atleast_1d(jnp.asarray(previous_discount, dtype=jnp.float32)),
            ]
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def encode(
        self,
        state: OnlineGatedStateBuilderState,
        raw_observation: Array,
    ) -> Float[Array, " feature_dim"]:
        """Pair raw input with the current hidden state without advancing it."""
        observation = jnp.asarray(raw_observation, dtype=jnp.float32).reshape(
            (self._config.observation_dim,)
        )
        if self._config.include_raw_observation:
            return jnp.concatenate([observation, state.hidden])
        return state.hidden

    @functools.partial(jax.jit, static_argnums=(0,))
    def update(
        self,
        state: OnlineGatedStateBuilderState,
        raw_observation: Array,
        previous_action: Array | int,
        previous_reward: Array | float,
        previous_discount: Array | float,
    ) -> tuple[OnlineGatedStateBuilderState, Float[Array, " feature_dim"]]:
        """Advance recurrence and its RTRL-style eligibility sensitivity."""
        event = self._event(
            raw_observation,
            previous_action,
            previous_reward,
            previous_discount,
        )
        new_hidden = self._transition(state.parameters, state.hidden, event)
        direct_sensitivity = jax.jacfwd(self._transition, argnums=0)(
            state.parameters,
            state.hidden,
            event,
        )

        gate_weights, gate_bias, _, _ = self._unpack_parameters(state.parameters)
        gate = jax.nn.sigmoid(gate_weights @ event + gate_bias)
        new_sensitivity = direct_sensitivity + ((1.0 - gate)[:, None] * state.parameter_sensitivity)
        next_state = OnlineGatedStateBuilderState(
            parameters=state.parameters,
            hidden=new_hidden,
            parameter_sensitivity=new_sensitivity,
            step_count=_saturating_int32_increment(state.step_count),
            update_count=state.update_count,
            last_gradient_norm=state.last_gradient_norm,
        )
        return next_state, self.encode(next_state, raw_observation)

    def start(
        self,
        state: OnlineGatedStateBuilderState,
        raw_observation: Array,
        last_action: Array | int = -1,
        last_reward: Array | float = 0.0,
        last_discount: Array | float = 1.0,
    ) -> tuple[OnlineGatedStateBuilderState, Array]:
        """Consume the initial observation."""
        return cast(
            tuple[OnlineGatedStateBuilderState, Array],
            self.update(
                state,
                raw_observation,
                last_action,
                last_reward,
                last_discount,
            ),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def learn(
        self,
        state: OnlineGatedStateBuilderState,
        representation_gradient: Array,
    ) -> tuple[OnlineGatedStateBuilderState, StateBuilderLearningDiagnostics]:
        """Apply one clipped online recurrent-gradient update.

        The supplied gradient must correspond to the representation emitted by
        the most recent ``start`` or ``update`` call.  Only its hidden-state
        block can affect recurrent parameters. The carried sensitivity is
        exact along a fixed-parameter unroll and becomes a changing-parameter
        eligibility approximation after this update.
        """
        gradient = jnp.asarray(representation_gradient, dtype=jnp.float32).reshape(
            (self.feature_dim(),)
        )
        hidden_gradient = gradient[-self._config.hidden_dim :]
        parameter_gradient = state.parameter_sensitivity.T @ hidden_gradient
        gradient_norm = jnp.linalg.norm(parameter_gradient)
        clip = jnp.asarray(self._config.gradient_clip, dtype=jnp.float32)
        clip_scale = jnp.minimum(1.0, clip / (gradient_norm + 1e-8))
        clipped_gradient = clip_scale * parameter_gradient
        parameter_update = (
            -jnp.asarray(
                self._config.step_size,
                dtype=jnp.float32,
            )
            * clipped_gradient
        )
        next_state = OnlineGatedStateBuilderState(
            parameters=state.parameters + parameter_update,
            hidden=state.hidden,
            parameter_sensitivity=state.parameter_sensitivity,
            step_count=state.step_count,
            update_count=_saturating_int32_increment(state.update_count),
            last_gradient_norm=gradient_norm,
        )
        diagnostics = StateBuilderLearningDiagnostics(
            gradient_norm=gradient_norm,
            clipped_gradient_norm=jnp.linalg.norm(clipped_gradient),
            parameter_update_norm=jnp.linalg.norm(parameter_update),
        )
        return next_state, diagnostics


def state_builder_from_config(payload: dict[str, Any]) -> StateBuilder[Any]:
    """Construct a known state builder from its serialized configuration."""
    builder_type = payload.get("type")
    if builder_type == "IdentityStateBuilder":
        return IdentityStateBuilder(IdentityStateBuilderConfig.from_config(payload))
    if builder_type == "FixedTraceStateBuilder":
        return FixedTraceStateBuilder(FixedTraceStateBuilderConfig.from_config(payload))
    if builder_type == "OnlineGatedStateBuilder":
        return OnlineGatedStateBuilder(OnlineGatedStateBuilderConfig.from_config(payload))
    raise ValueError(f"unknown state builder type: {builder_type!r}")


def save_state_builder_checkpoint(
    builder: StateBuilder[Any],
    state: Any,
    path: str | Path,
) -> None:
    """Save a builder's full configuration and PyTree state."""
    save_checkpoint(
        state,
        path,
        metadata={
            "schema": STATE_BUILDER_CHECKPOINT_SCHEMA,
            "builder_config": builder.to_config(),
            "resource_budget": builder.resource_budget().to_config(),
        },
    )


def load_state_builder_checkpoint(
    path: str | Path,
    *,
    template_key: Array | None = None,
) -> tuple[StateBuilder[Any], Any]:
    """Restore a state builder without requiring a caller-constructed template."""
    metadata = load_checkpoint_metadata(path)
    if metadata.get("schema") != STATE_BUILDER_CHECKPOINT_SCHEMA:
        raise ValueError("checkpoint is not an Alberta state-builder v1 checkpoint")
    config = metadata.get("builder_config")
    if not isinstance(config, dict):
        raise ValueError("state-builder checkpoint is missing builder_config")
    builder = state_builder_from_config(config)
    key = jr.key(0) if template_key is None else template_key
    template = builder.init(key)
    state, restored_metadata = load_checkpoint(template, path)
    if restored_metadata != metadata:
        raise ValueError("state-builder checkpoint metadata changed between reads")
    if restored_metadata.get("resource_budget") != builder.resource_budget().to_config():
        raise ValueError("state-builder checkpoint resource budget does not match config")
    return builder, state


__all__ = [
    "STATE_BUILDER_CHECKPOINT_SCHEMA",
    "FixedTraceStateBuilder",
    "FixedTraceStateBuilderConfig",
    "IdentityStateBuilder",
    "IdentityStateBuilderConfig",
    "IdentityStateBuilderState",
    "OnlineGatedStateBuilder",
    "OnlineGatedStateBuilderConfig",
    "OnlineGatedStateBuilderState",
    "StateBuilder",
    "StateBuilderBudget",
    "StateBuilderLearningDiagnostics",
    "load_state_builder_checkpoint",
    "save_state_builder_checkpoint",
    "state_builder_from_config",
]
