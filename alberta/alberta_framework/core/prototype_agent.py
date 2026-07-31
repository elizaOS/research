# mypy: disable-error-code="attr-defined,call-arg,no-any-return,union-attr"
"""Prototype agent assembling mechanisms across the Alberta Plan.

The :class:`PrototypeAgent` is an experimental integration of mechanisms
mapped to the Alberta Plan's retreat-and-return strategy:

- **OaK** (steps 5/6/10/11): Differential average-reward control with temporal
  abstraction, option utility tracking, and curation.
- **Horde** (step 3): Parallel GVF prediction demons sharing a learned trunk.
- **World Model** (step 8): One-step action-conditioned environment model.
- **Guarded Dreaming** (step 9): Model-generated Dyna transitions accepted only
  when the world model's error EMA is below a configurable gate.
- **Intelligence Amplification** (step 12, optional): Exo-cerebellum +
  exo-cortex companion that augments a partner agent's decisions.

The integrated control and prediction paths are online, batch-size-one
learners. Step 1/2 adaptive optimizers and bounded updates are available
elsewhere in the framework, but the default STOMP base learner here uses LMS;
this integration alone is therefore not evidence that every Step 1/2 variant
is active.

References:
    Sutton, Bowling, & Pilarski (2022). "The Alberta Plan for AI Research."
    Barreto et al. (2019). "The Option Keyboard: Combining Skills in RL."
    Sutton et al. (2011). "Horde: A Scalable Real-time Architecture."
    Elsayed & Sutton (2024). "Streaming Backpropagation through Time."
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
from pathlib import Path
from typing import Any, Literal, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import Array
from jaxtyping import Float, Int

from alberta_framework.core.checkpoints import (
    load_checkpoint,
    load_checkpoint_metadata,
    save_checkpoint,
)
from alberta_framework.core.dreaming import (
    DreamingConfig,
    GuardedDreamer,
    RecentObservationBuffer,
)
from alberta_framework.core.horde import HordeLearner
from alberta_framework.core.intelligence_amplification import (
    IAAgent,
    IAConfig,
)
from alberta_framework.core.oak import OaKAgent, OaKConfig, OaKState
from alberta_framework.core.options import STOMPConfig, SubtaskSpec
from alberta_framework.core.types import HordeSpec
from alberta_framework.core.world_model import (
    ActionConditionedWorldModel,
    ActionConditionedWorldModelConfig,
)

PROTOTYPE_CHECKPOINT_SCHEMA = "alberta.prototype_agent.v1"
_DREAM_NEXT_OBSERVATION_STREAM_TAG = 0x44524D4F

# ---------------------------------------------------------------------------
# Standalone utility
# ---------------------------------------------------------------------------


def feature_to_subtask_specs(
    oak_state: OaKState,
    *,
    n_subtasks: int = 4,
    threshold: float = 0.5,
    pseudo_reward_scale: float = 1.0,
    max_option_steps: int = 20,
) -> tuple[SubtaskSpec, ...]:
    """Extract top-k subtask specs from OaK Q-weight feature importance.

    Ranks observation dimensions by the maximum absolute Q-weight across all
    base and option policies.  The top-k most important features become subtask
    targets for the next curation cycle.

    Args:
        oak_state: Current OaK state.
        n_subtasks: Number of subtask specs to return.
        threshold: Pseudo-reward threshold for subtask completion.
        pseudo_reward_scale: Pseudo-reward multiplier for generated specs.
        max_option_steps: Hard cap on option duration.

    Returns:
        Tuple of up to ``n_subtasks`` :class:`SubtaskSpec` instances, ordered
        by descending feature importance.
    """
    bls = oak_state.stomp_state.base_learner_state
    trunk_ws = bls.trunk_params.weights
    if len(trunk_ws) == 0:
        # Linear base Q: head_params.weights[i] has shape (1, obs_dim)
        base_q_mat = jnp.stack([w[0] for w in bls.head_params.weights])
    else:
        # Nonlinear: use first trunk layer as feature-importance proxy
        base_q_mat = trunk_ws[0]  # (hidden_size, obs_dim)
    feature_importance = jnp.max(jnp.abs(base_q_mat), axis=0)  # (obs_dim,)

    opt_q = oak_state.stomp_state.option_policies.q_weights      # (n_opts, n_prim, obs_dim)
    opt_q_abs = jnp.abs(opt_q)
    obs_dim = int(opt_q.shape[-1])
    opt_importance = jnp.max(opt_q_abs.reshape(-1, obs_dim), axis=0)  # (obs_dim,)

    combined = feature_importance + opt_importance
    n = min(n_subtasks, obs_dim)
    ranking = sorted(range(obs_dim), key=lambda i: float(combined[i]), reverse=True)[:n]

    return tuple(
        SubtaskSpec(
            feature_index=int(idx),
            threshold=threshold,
            pseudo_reward_scale=pseudo_reward_scale,
            max_option_steps=max_option_steps,
        )
        for idx in ranking
    )


# ---------------------------------------------------------------------------
# GRU Perception (Step 8 sub-component a — recursive state update)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class GRUPerceptionConfig:
    """Configuration for the fixed-weight GRU perception layer.

    A minimal echo-state GRU that provides the recursive state-update
    (perception) sub-component required by Alberta Plan Step 8.  Weights are
    sampled once at :meth:`PrototypeAgent.init` and remain fixed; the hidden
    state is updated at every step.  The downstream Q-function (OaK) learns to
    use the temporal context encoded in ``hidden``.

    Args:
        observation_dim: Raw observation dimensionality (GRU input).
        hidden_dim: GRU hidden-state dimensionality (GRU output).

    Note:
        When this config is present, the effective observation dimensionality
        seen by OaK, the Horde, the world model, and IA is
        ``observation_dim + hidden_dim``.  Set ``oak.observation_dim``
        (and ``world_model.observation_dim`` when applicable) accordingly.
    """

    observation_dim: int
    hidden_dim: int = 32

    def augmented_dim(self) -> int:
        """Return ``observation_dim + hidden_dim``."""
        return self.observation_dim + self.hidden_dim

    def to_config(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "type": "GRUPerceptionConfig",
            "observation_dim": self.observation_dim,
            "hidden_dim": self.hidden_dim,
        }

    @classmethod
    def from_config(cls, payload: dict[str, Any]) -> GRUPerceptionConfig:
        """Reconstruct from :meth:`to_config` output."""
        d = dict(payload)
        d.pop("type", None)
        return cls(**d)


@chex.dataclass(frozen=True)
class GRUPerceptionState:
    """State for the fixed-weight GRU perception layer.

    Weight matrices are initialised once and never updated; ``hidden`` is the
    only mutable component and is replaced at every step.

    Attributes:
        W_z, U_z, b_z: Update-gate input/recurrent weights and bias.
        W_r, U_r, b_r: Reset-gate input/recurrent weights and bias.
        W_h, U_h, b_h: Candidate-hidden input/recurrent weights and bias.
        hidden: Running GRU hidden state ``h_t``.
    """

    W_z: Float[Array, "hidden_dim obs_dim"]
    U_z: Float[Array, "hidden_dim hidden_dim"]
    b_z: Float[Array, " hidden_dim"]
    W_r: Float[Array, "hidden_dim obs_dim"]
    U_r: Float[Array, "hidden_dim hidden_dim"]
    b_r: Float[Array, " hidden_dim"]
    W_h: Float[Array, "hidden_dim obs_dim"]
    U_h: Float[Array, "hidden_dim hidden_dim"]
    b_h: Float[Array, " hidden_dim"]
    hidden: Float[Array, " hidden_dim"]


def _glorot_uniform(key: Array, shape: tuple[int, int]) -> Array:
    fan_in, fan_out = shape[-1], shape[0]
    limit = jnp.sqrt(6.0 / (fan_in + fan_out))
    return jr.uniform(key, shape, dtype=jnp.float32, minval=-limit, maxval=limit)


def _init_gru_state(cfg: GRUPerceptionConfig, key: Array) -> GRUPerceptionState:
    """Glorot-uniform weight init + zero hidden state."""
    keys = jr.split(key, 6)
    d_obs, d_h = cfg.observation_dim, cfg.hidden_dim
    return GRUPerceptionState(
        W_z=_glorot_uniform(keys[0], (d_h, d_obs)),
        U_z=_glorot_uniform(keys[1], (d_h, d_h)),
        b_z=jnp.zeros((d_h,), dtype=jnp.float32),
        W_r=_glorot_uniform(keys[2], (d_h, d_obs)),
        U_r=_glorot_uniform(keys[3], (d_h, d_h)),
        b_r=jnp.zeros((d_h,), dtype=jnp.float32),
        W_h=_glorot_uniform(keys[4], (d_h, d_obs)),
        U_h=_glorot_uniform(keys[5], (d_h, d_h)),
        b_h=jnp.zeros((d_h,), dtype=jnp.float32),
        hidden=jnp.zeros((d_h,), dtype=jnp.float32),
    )


def _gru_step(
    gru: GRUPerceptionState,
    obs: Float[Array, " obs_dim"],
) -> tuple[GRUPerceptionState, Float[Array, " augmented_dim"]]:
    """One GRU step: update hidden state and return augmented observation.

    Returns the *new* GRU state and the concatenation
    ``[obs, new_hidden]`` as the augmented observation.
    """
    h = gru.hidden
    z = jax.nn.sigmoid(gru.W_z @ obs + gru.U_z @ h + gru.b_z)
    r = jax.nn.sigmoid(gru.W_r @ obs + gru.U_r @ h + gru.b_r)
    h_tilde = jnp.tanh(gru.W_h @ obs + gru.U_h @ (r * h) + gru.b_h)
    new_h = (1.0 - z) * h + z * h_tilde
    new_gru = cast(GRUPerceptionState, gru.replace(hidden=new_h))
    return new_gru, jnp.concatenate([obs, new_h], axis=0)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def _default_oak_config() -> OaKConfig:
    return OaKConfig(
        stomp=STOMPConfig(
            subtask_specs=(SubtaskSpec(feature_index=0),),
            observation_dim=4,
            n_primitive_actions=2,
        )
    )


def _sample_one_hot_dream_observation(
    prediction: Array,
    key: Array,
) -> tuple[Array, Array]:
    """Sample a categorical one-hot state from an expectation-valued prediction.

    Finite coordinates are first projected into ``[0, 1]``. Categorical
    sampling normalizes the remaining mass implicitly and preserves exact zero
    support. Non-finite or zero-mass predictions return a safe dummy one-hot
    value together with ``False`` so the caller can reject the backup.
    """

    values = jnp.asarray(prediction, dtype=jnp.float32)
    if values.ndim != 1 or values.shape[0] == 0:
        raise ValueError(
            "sampled dream observation prediction must be a non-empty vector"
        )
    finite = jnp.all(jnp.isfinite(values))
    weights = jnp.clip(
        jnp.where(jnp.isfinite(values), values, jnp.zeros_like(values)),
        0.0,
        1.0,
    )
    valid = finite & (jnp.sum(weights) > 0.0)
    fallback = jax.nn.one_hot(0, values.shape[0], dtype=values.dtype)
    safe_weights = jnp.where(valid, weights, fallback)
    logits = jnp.where(safe_weights > 0.0, jnp.log(safe_weights), -jnp.inf)
    index = jr.categorical(key, logits).astype(jnp.int32)
    sampled = jax.nn.one_hot(index, values.shape[0], dtype=values.dtype)
    return sampled, valid


@dataclasses.dataclass(frozen=True)
class PrototypeAgentConfig:
    """Configuration for the experimental PrototypeAgent composition.

    All components are designed for *continuing* average-reward settings —
    no episode resets, no offline training phases.

    Args:
        oak: OaK agent configuration (steps 5/6/10/11).  Required.
        world_model: World model configuration (step 8).  When ``None``,
            dreaming is disabled regardless of ``n_dreams_per_step``.
        dreaming: Dreaming guard configuration (step 9).  Only honoured when
            ``world_model`` is not ``None``.
        buffer_capacity: Number of real observations retained as dream anchors.
        n_dreams_per_step: Dyna-style imagined transitions per real step.
            Zero disables dreaming even when a world model is configured.
        dream_next_observation_mode: How a model-predicted next observation is
            exposed to the base-Q learner. ``"model_prediction"`` preserves
            the regression output. ``"sample_one_hot"`` clips its coordinates
            to ``[0, 1]`` and samples one categorical one-hot state; this mode
            is only valid when the complete control observation is one-hot.
        horde_spec: GVF Horde specification (step 3).  When ``None``, the
            prediction-demon pathway is disabled.
        horde_hidden_sizes: Trunk layer widths for the Horde MLP.
        horde_step_size: Base step-size for the Horde learner.
        ia: IA agent configuration (step 12).  When ``None``, the
            intelligence-amplification companion is disabled.
    """

    oak: OaKConfig = dataclasses.field(default_factory=_default_oak_config)
    world_model: ActionConditionedWorldModelConfig | None = None
    dreaming: DreamingConfig | None = None
    buffer_capacity: int = 200
    n_dreams_per_step: int = 0
    dream_next_observation_mode: Literal[
        "model_prediction",
        "sample_one_hot",
    ] = "model_prediction"
    horde_spec: HordeSpec | None = None
    horde_hidden_sizes: tuple[int, ...] = (64, 64)
    horde_step_size: float = 0.1
    ia: IAConfig | None = None
    gru_perception: GRUPerceptionConfig | None = None
    auto_curate_every: int = 0

    def __post_init__(self) -> None:
        if self.buffer_capacity <= 0:
            raise ValueError("buffer_capacity must be positive")
        if self.n_dreams_per_step < 0:
            raise ValueError("n_dreams_per_step must be non-negative")
        if self.dream_next_observation_mode not in {
            "model_prediction",
            "sample_one_hot",
        }:
            raise ValueError(
                "dream_next_observation_mode must be "
                "'model_prediction' or 'sample_one_hot'"
            )
        if self.horde_step_size <= 0.0:
            raise ValueError("horde_step_size must be positive")
        if self.auto_curate_every < 0:
            raise ValueError("auto_curate_every must be non-negative")
        if self.world_model is None and self.n_dreams_per_step > 0:
            raise ValueError(
                "n_dreams_per_step > 0 requires world_model to be configured"
            )
        if self.world_model is not None:
            if self.world_model.observation_dim != self.oak.observation_dim:
                raise ValueError(
                    "world_model.observation_dim must match oak.observation_dim, "
                    f"got {self.world_model.observation_dim} and "
                    f"{self.oak.observation_dim}"
                )
            if self.world_model.n_actions != self.oak.n_primitive_actions:
                raise ValueError(
                    "world_model.n_actions must match oak.n_primitive_actions, "
                    f"got {self.world_model.n_actions} and "
                    f"{self.oak.n_primitive_actions}"
                )
        if self.gru_perception is not None:
            if self.dream_next_observation_mode == "sample_one_hot":
                raise ValueError(
                    "dream_next_observation_mode='sample_one_hot' is incompatible "
                    "with continuous GRU-augmented observations"
                )
            aug = self.gru_perception.augmented_dim()
            if self.oak.observation_dim != aug:
                raise ValueError(
                    f"When gru_perception is set, oak.observation_dim must equal "
                    f"gru_perception.observation_dim + gru_perception.hidden_dim "
                    f"= {aug}, got {self.oak.observation_dim}"
                )
        if self.ia is not None:
            ia_obs = self.ia.cortex.observation_dim
            oak_obs = self.oak.observation_dim
            if ia_obs != oak_obs:
                raise ValueError(
                    f"ia.cortex.observation_dim ({ia_obs}) must match "
                    f"oak.observation_dim ({oak_obs})"
                )

    def to_config(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        payload: dict[str, Any] = {
            "type": "PrototypeAgentConfig",
            "oak": self.oak.to_config(),
            "buffer_capacity": self.buffer_capacity,
            "n_dreams_per_step": self.n_dreams_per_step,
            "horde_hidden_sizes": list(self.horde_hidden_sizes),
            "horde_step_size": self.horde_step_size,
            "auto_curate_every": self.auto_curate_every,
        }
        if self.dream_next_observation_mode != "model_prediction":
            payload["dream_next_observation_mode"] = self.dream_next_observation_mode
        if self.world_model is not None:
            payload["world_model"] = self.world_model.to_config()
        if self.dreaming is not None:
            payload["dreaming"] = self.dreaming.to_config()
        if self.horde_spec is not None:
            payload["horde_spec"] = self.horde_spec.to_config()
        if self.ia is not None:
            payload["ia"] = self.ia.to_config()
        if self.gru_perception is not None:
            payload["gru_perception"] = self.gru_perception.to_config()
        return payload

    @classmethod
    def from_config(cls, payload: dict[str, Any]) -> PrototypeAgentConfig:
        """Reconstruct from :meth:`to_config` output."""
        from alberta_framework.core.types import HordeSpec as _HordeSpec

        data = dict(payload)
        data.pop("type", None)
        oak = OaKConfig.from_config(cast(dict[str, Any], data.pop("oak")))

        wm_raw = data.pop("world_model", None)
        world_model = (
            ActionConditionedWorldModelConfig.from_config(wm_raw) if wm_raw is not None else None
        )
        dream_raw = data.pop("dreaming", None)
        dreaming = DreamingConfig.from_config(dream_raw) if dream_raw is not None else None
        horde_raw = data.pop("horde_spec", None)
        horde_spec = _HordeSpec.from_config(horde_raw) if horde_raw is not None else None
        ia_raw = data.pop("ia", None)
        ia = IAConfig.from_config(ia_raw) if ia_raw is not None else None
        gru_raw = data.pop("gru_perception", None)
        gru_perception = (
            GRUPerceptionConfig.from_config(gru_raw) if gru_raw is not None else None
        )

        hidden = tuple(int(x) for x in data.pop("horde_hidden_sizes", [64, 64]))
        return cls(
            oak=oak,
            world_model=world_model,
            dreaming=dreaming,
            buffer_capacity=int(data.pop("buffer_capacity", 200)),
            n_dreams_per_step=int(data.pop("n_dreams_per_step", 0)),
            dream_next_observation_mode=data.pop(
                "dream_next_observation_mode",
                "model_prediction",
            ),
            horde_spec=horde_spec,
            horde_hidden_sizes=hidden,
            horde_step_size=float(data.pop("horde_step_size", 0.1)),
            ia=ia,
            gru_perception=gru_perception,
            auto_curate_every=int(data.pop("auto_curate_every", 0)),
        )


# ---------------------------------------------------------------------------
# State and result types
# ---------------------------------------------------------------------------


@chex.dataclass(frozen=True)
class PrototypeTransition:
    """Explicit real-transition contract for :class:`PrototypeAgent`.

    ``discount`` is the effective scalar continuation multiplier for control
    and world-model learning.  It must be finite and in ``[0, 1]``; use zero
    for an environmental terminal transition.  Unlike the legacy
    :meth:`PrototypeAgent.update` surface, it is never synthesized from an
    agent configuration.

    Horde questions retain their own horizons.  ``horde_discounts`` may
    provide one effective discount per GVF.  When it is omitted, the adapter
    uses each demon's configured gamma on nonterminal transitions and zeros
    every demon only when the global ``discount`` is zero.  Consequently,
    callers that need fractional environment continuation to alter GVF
    horizons must supply ``horde_discounts`` explicitly.

    Attributes:
        reward: Scalar reward emitted by the environment.
        next_observation: Raw next observation (before optional GRU
            augmentation).
        discount: Effective scalar continuation multiplier.
        horde_cumulants: Optional vector of one cumulant per GVF demon.
            ``NaN`` keeps the framework's existing inactive-demon semantics.
        horde_discounts: Optional vector of effective per-GVF discounts.
    """

    reward: Float[Array, ""]
    next_observation: Float[Array, " observation_dim"]
    discount: Float[Array, ""]
    horde_cumulants: Any = None
    horde_discounts: Any = None


@chex.dataclass(frozen=True)
class PrototypeAgentState:
    """Full prototype agent state.

    Optional sub-states (``world_model_state``, ``buffer_state``,
    ``horde_state``, ``ia_state``) are ``None`` when the corresponding
    component is disabled in the configuration.  The PyTree structure is
    fixed for a given :class:`PrototypeAgent` instance — never switch between
    ``None`` and a real state after initialisation.
    """

    oak_state: OaKState
    world_model_state: Any  # ActionConditionedWorldModelState | None
    buffer_state: Any  # RecentObservationBufferState | None
    horde_state: Any  # MultiHeadMLPState | None
    ia_state: Any  # IAState | None
    gru_state: Any  # GRUPerceptionState | None
    step_count: Int[Array, ""]


@chex.dataclass(frozen=True)
class PrototypeUpdateResult:
    """Result of one real-time prototype agent transition."""

    state: PrototypeAgentState
    action: Int[Array, ""]
    oak_td_error: Float[Array, ""]
    oak_average_reward: Float[Array, ""]
    world_model_error: Any  # Float[Array, ""] | None
    dream_td_errors: Any  # Float[Array, " n_dreams"] | None
    horde_td_errors: Any  # Float[Array, " n_demons"] | None
    ia_augmented_obs: Any  # Float[Array, " augmented_dim"] | None
    ia_recommendation: Any  # Int[Array, ""] | None


@chex.dataclass(frozen=True)
class PrototypeArrayResult:
    """Result from :meth:`PrototypeAgent.scan` over a batch of transitions."""

    state: PrototypeAgentState
    actions: Int[Array, " num_steps"]
    oak_td_errors: Float[Array, " num_steps"]
    oak_average_rewards: Float[Array, " num_steps"]


def _contains_tracer(value: Any) -> bool:
    """Return whether a PyTree contains a JAX tracing value."""
    return any(
        isinstance(leaf, jax.core.Tracer)
        for leaf in jax.tree_util.tree_leaves(value)
    )


def _checked_finite_array(
    value: Any,
    shape: tuple[int, ...],
    *,
    name: str,
    allow_nan: bool = False,
) -> Array:
    """Validate shape/finiteness eagerly and poison invalid traced values."""
    array = jnp.asarray(value, dtype=jnp.float32).reshape(shape)
    element_valid = ~jnp.isinf(array) if allow_nan else jnp.isfinite(array)
    valid = jnp.all(element_valid)
    if not _contains_tracer(array) and not bool(valid):
        requirement = "must not contain infinity" if allow_nan else "must be finite"
        raise ValueError(f"{name} {requirement}")
    return jnp.where(valid, array, jnp.full_like(array, jnp.nan))


def _checked_unit_discount(value: Any, shape: tuple[int, ...], *, name: str) -> Array:
    """Validate finite ``[0, 1]`` discounts at eager and traced boundaries."""
    array = jnp.asarray(value, dtype=jnp.float32).reshape(shape)
    valid = jnp.all(
        jnp.isfinite(array) & (array >= 0.0) & (array <= 1.0)
    )
    if not _contains_tracer(array) and not bool(valid):
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return jnp.where(valid, array, jnp.full_like(array, jnp.nan))


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class PrototypeAgent:
    """Prototype combining online components from across the Alberta Plan.

    Operates in **continuing average-reward mode** — no episode resets, no
    offline batch phases. Designed for use in sim-to-real transfer:
    :meth:`update_transition` is the authoritative transition boundary in
    simulation and on real hardware. :meth:`update` remains a compatibility
    wrapper for callers that predate explicit discounts.

    Single-step daemon usage::

        state = agent.start(agent.init(jr.key(0)), initial_obs)
        action = int(agent.act(state, initial_obs))
        while True:
            reward, next_obs, discount = env.step(action)
            result = agent.update_transition(
                state,
                PrototypeTransition(
                    reward=reward,
                    next_observation=next_obs,
                    discount=discount,
                ),
            )
            state, action = result.state, int(result.action)

    Periodic curation (Python-level, outside JAX)::

        if step % curation_interval == 0:
            agent, state = agent.curate(state, key)
    """

    def __init__(self, config: PrototypeAgentConfig) -> None:
        self._config = config
        self._oak = OaKAgent(config.oak)

        self._world_model: ActionConditionedWorldModel | None = None
        self._buffer: RecentObservationBuffer | None = None
        self._dreamer: GuardedDreamer | None = None
        if config.world_model is not None:
            self._world_model = ActionConditionedWorldModel(config.world_model)
            self._buffer = RecentObservationBuffer(
                config.buffer_capacity, config.oak.observation_dim
            )
            self._dreamer = GuardedDreamer(config.dreaming or DreamingConfig())

        self._horde: HordeLearner | None = None
        if config.horde_spec is not None:
            self._horde = HordeLearner(
                config.horde_spec,
                hidden_sizes=config.horde_hidden_sizes,
                step_size=config.horde_step_size,
            )

        self._ia: IAAgent | None = None
        if config.ia is not None:
            self._ia = IAAgent(config.ia)

    # -- Properties -----------------------------------------------------------

    @property
    def config(self) -> PrototypeAgentConfig:
        """Agent configuration."""
        return self._config

    @property
    def oak_agent(self) -> OaKAgent:
        """Underlying OaK control agent."""
        return self._oak

    # -- Serialization --------------------------------------------------------

    def to_config(self) -> dict[str, Any]:
        """Serialize agent configuration."""
        return self._config.to_config()

    @classmethod
    def from_config(cls, payload: dict[str, Any]) -> PrototypeAgent:
        """Reconstruct from :meth:`to_config` output."""
        return cls(PrototypeAgentConfig.from_config(payload))

    # -- Lifecycle ------------------------------------------------------------

    def init(self, key: Array) -> PrototypeAgentState:
        """Initialise all sub-states.

        Args:
            key: JAX PRNG key.

        Returns:
            Fresh :class:`PrototypeAgentState`.
        """
        key, oak_key, wm_key, horde_key, ia_key, gru_key = jr.split(key, 6)
        oak_state = self._oak.init(oak_key)

        wm_state: Any = None
        buf_state: Any = None
        if self._world_model is not None and self._buffer is not None:
            wm_state = self._world_model.init(wm_key)
            buf_state = self._buffer.init()

        horde_state: Any = None
        if self._horde is not None:
            horde_state = self._horde.init(self._config.oak.observation_dim, horde_key)

        ia_state: Any = None
        if self._ia is not None:
            ia_state = self._ia.init(ia_key)

        gru_state: Any = None
        if self._config.gru_perception is not None:
            gru_state = _init_gru_state(self._config.gru_perception, gru_key)

        return PrototypeAgentState(
            oak_state=oak_state,
            world_model_state=wm_state,
            buffer_state=buf_state,
            horde_state=horde_state,
            ia_state=ia_state,
            gru_state=gru_state,
            step_count=jnp.array(0, dtype=jnp.int32),
        )

    def start(
        self,
        state: PrototypeAgentState,
        initial_observation: Array,
    ) -> PrototypeAgentState:
        """Prime the agent with an initial observation.

        Must be called once before :meth:`update`.

        Args:
            state: Uninitialised state from :meth:`init`.
            initial_observation: First environment observation.

        Returns:
            State with OaK (and optionally IA) primed.
        """
        raw_obs = jnp.asarray(initial_observation, dtype=jnp.float32)
        new_gru_state = state.gru_state
        obs_for_oak = raw_obs
        if state.gru_state is not None:
            new_gru_state, obs_for_oak = _gru_step(state.gru_state, raw_obs)
        new_oak = self._oak.start(state.oak_state, obs_for_oak)
        new_ia = state.ia_state
        if self._ia is not None and state.ia_state is not None:
            new_ia = self._ia.start(state.ia_state, obs_for_oak)
        return cast(
            PrototypeAgentState,
            state.replace(oak_state=new_oak, ia_state=new_ia, gru_state=new_gru_state),
        )

    def act(
        self,
        state: PrototypeAgentState,
        observation: Array,
    ) -> Int[Array, ""]:
        """Return a primitive action without updating state.

        When GRU perception is configured, the raw observation is routed
        through the same augmentation as :meth:`update`; the transient GRU
        hidden update is discarded (``state`` is never mutated).

        Immediately after :meth:`start`, this returns the action already
        sampled and recorded by STOMP, including an option's first
        intra-option action. On later diagnostic queries it returns the
        greedy primitive action for ``observation``.
        """
        obs = jnp.asarray(observation, dtype=jnp.float32)
        if state.gru_state is not None:
            _, obs = _gru_step(state.gru_state, obs)
        n_prim = self._config.oak.n_primitive_actions
        all_q = self._oak.base_q_values(state.oak_state, obs)
        greedy_primitive = jnp.argmax(all_q[:n_prim]).astype(jnp.int32)
        initial_action = (
            state.oak_state.stomp_state.last_primitive_action
        )
        return jnp.where(
            state.step_count == 0,
            initial_action,
            greedy_primitive,
        )

    # -- Dreaming scan (JIT-compiled as a method so the closure is stable) ----

    @functools.partial(jax.jit, static_argnums=(0,))
    def _run_dreams(
        self,
        oak_state: OaKState,
        wm_state: Any,
        buf_state: Any,
        rng_key: Array,
    ) -> tuple[OaKState, Float[Array, " n_dreams"]]:
        n_prim = self._config.oak.n_primitive_actions
        sample_one_hot = (
            self._config.dream_next_observation_mode == "sample_one_hot"
        )
        dream_observation_root = (
            jr.fold_in(rng_key, _DREAM_NEXT_OBSERVATION_STREAM_TAG)
            if sample_one_hot
            else rng_key
        )

        def _dream_step(
            carry: tuple[OaKState, Array], dream_index: Array
        ) -> tuple[tuple[OaKState, Array], Float[Array, ""]]:
            oak_s, k = carry
            # Keep this legacy split unchanged so raw and sampled-one-hot
            # ablations share identical anchor/action key streams.
            k, sample_key, action_key = jr.split(k, 3)
            anchor_obs, _ = self._buffer.sample(buf_state, sample_key)
            action = jr.randint(action_key, (), 0, n_prim, dtype=jnp.int32)
            proposal = self._dreamer.propose(self._world_model, wm_state, anchor_obs, action)
            dream_next_observation = proposal.transition.next_observation
            dream_accepted = proposal.accepted
            if sample_one_hot:
                observation_key = jr.fold_in(
                    dream_observation_root,
                    dream_index,
                )
                dream_next_observation, projection_valid = (
                    _sample_one_hot_dream_observation(
                        dream_next_observation,
                        observation_key,
                    )
                )
                dream_accepted = dream_accepted & projection_valid

            # A dream is a primitive base-Q backup from a sampled real anchor.
            # Force the synthetic state idle so it cannot advance/terminate a
            # real option, then carry back *only* the base learner state. The
            # real option trajectory, reward-rate estimate, action/RNG,
            # utility/curation statistics, and real-step counters are all
            # invariants of imagined computation.
            temp_stomp = oak_s.stomp_state.replace(
                base_last_obs=anchor_obs,
                base_last_action=action,
                last_primitive_action=action,
                executing_option=jnp.array(-1, dtype=jnp.int32),
            )
            temp_oak = cast(OaKState, oak_s.replace(stomp_state=temp_stomp))
            dream_result = self._oak.update(
                temp_oak,
                proposal.transition.reward,
                dream_next_observation,
                proposal.transition.discount,
                enable_option_planning=False,
            )

            candidate_learner = dream_result.state.stomp_state.base_learner_state
            accepted_learner = jax.tree_util.tree_map(
                lambda candidate, real: jnp.where(
                    dream_accepted, candidate, real
                ),
                candidate_learner,
                oak_s.stomp_state.base_learner_state,
            )
            new_oak_s = cast(
                OaKState,
                oak_s.replace(
                    stomp_state=oak_s.stomp_state.replace(
                        base_learner_state=accepted_learner
                    )
                ),
            )
            td_err = jnp.where(
                dream_accepted,
                dream_result.td_error,
                jnp.array(0.0, dtype=jnp.float32),
            )
            return (new_oak_s, k), td_err

        (new_oak_state, _), dream_td_errors = jax.lax.scan(
            _dream_step,
            (oak_state, rng_key),
            jnp.arange(self._config.n_dreams_per_step),
        )
        return new_oak_state, dream_td_errors

    # -- Core update ----------------------------------------------------------

    def update(
        self,
        state: PrototypeAgentState,
        reward: Array,
        next_observation: Array,
        horde_cumulants: Array | None = None,
    ) -> PrototypeUpdateResult:
        """Compatibility wrapper for the historical transition API.

        New integrations should call :meth:`update_transition` and supply an
        explicit discount.  This wrapper preserves the former behavior:
        primitive control bootstraps with one, option returns use
        ``STOMPConfig.option_gamma``, and the world model (when enabled)
        receives its configured gamma as the target discount.
        """
        legacy_model_discount = (
            self._config.world_model.gamma
            if self._config.world_model is not None
            else 1.0
        )
        transition = self._normalize_transition(
            PrototypeTransition(
                reward=reward,
                next_observation=next_observation,
                discount=jnp.asarray(legacy_model_discount, dtype=jnp.float32),
                horde_cumulants=horde_cumulants,
            )
        )
        return self._update_transition_impl(
            state,
            transition,
            control_discount=None,
        )

    def update_transition(
        self,
        state: PrototypeAgentState,
        transition: PrototypeTransition,
    ) -> PrototypeUpdateResult:
        """Process one explicit real-time continuing transition.

        Execution order:

        1. Update world model from the real transition (if configured).
        2. Add ``next_observation`` to the dream anchor buffer.
        3. Update OaK from the real transition.
        4. Run ``n_dreams_per_step`` guarded Dyna updates (if configured).
        5. Update the Horde from the real transition (if configured).
        6. Update the IA companion from the real transition (if configured).

        Args:
            state: Current agent state.  Must have been primed with
                :meth:`start`.
            transition: Reward, raw next observation, continuation discount,
                and optional Horde signals. The world model and control
                learner consume exactly this scalar discount.

        Returns:
            :class:`PrototypeUpdateResult` with updated state, selected action,
            and per-component diagnostics.
        """
        normalized = self._normalize_transition(transition)
        return self._update_transition_impl(
            state,
            normalized,
            control_discount=normalized.discount,
        )

    def _normalize_transition(
        self,
        transition: PrototypeTransition,
    ) -> PrototypeTransition:
        """Validate/coerce a transition without host conversion under JIT."""
        raw_observation_dim = (
            self._config.gru_perception.observation_dim
            if self._config.gru_perception is not None
            else self._config.oak.observation_dim
        )
        reward = _checked_finite_array(
            transition.reward,
            (),
            name="transition.reward",
        )
        next_observation = _checked_finite_array(
            transition.next_observation,
            (raw_observation_dim,),
            name="transition.next_observation",
        )
        discount = _checked_unit_discount(
            transition.discount,
            (),
            name="transition.discount",
        )

        horde_cumulants: Any = None
        horde_discounts: Any = None
        if self._horde is None:
            if (
                transition.horde_cumulants is not None
                or transition.horde_discounts is not None
            ):
                raise ValueError(
                    "Horde transition fields require horde_spec to be configured"
                )
        else:
            n_demons = self._horde.n_demons
            if transition.horde_cumulants is not None:
                horde_cumulants = _checked_finite_array(
                    transition.horde_cumulants,
                    (n_demons,),
                    name="transition.horde_cumulants",
                    allow_nan=True,
                )
            if transition.horde_discounts is not None:
                horde_discounts = _checked_unit_discount(
                    transition.horde_discounts,
                    (n_demons,),
                    name="transition.horde_discounts",
                )

        return PrototypeTransition(
            reward=reward,
            next_observation=next_observation,
            discount=discount,
            horde_cumulants=horde_cumulants,
            horde_discounts=horde_discounts,
        )

    def _update_transition_impl(
        self,
        state: PrototypeAgentState,
        transition: PrototypeTransition,
        *,
        control_discount: Array | None,
    ) -> PrototypeUpdateResult:
        """Apply an already-normalized transition to every enabled component."""
        raw_obs = transition.next_observation
        rew = transition.reward

        # -- Step 8a: GRU recursive state update (perception) -----------------
        new_gru_state = state.gru_state
        obs = raw_obs
        if state.gru_state is not None:
            new_gru_state, obs = _gru_step(state.gru_state, raw_obs)

        # Snapshot the real last observation/action before OaK update. The
        # base action may be an extended option index, but the world model is
        # action-conditioned on the primitive command sent to the environment.
        last_obs = state.oak_state.stomp_state.base_last_obs
        last_action = state.oak_state.stomp_state.last_primitive_action

        # -- Step 8b: world model update (real transition) --------------------
        new_wm_state = state.world_model_state
        new_buf_state = state.buffer_state
        wm_error: Any = None

        if self._world_model is not None and self._buffer is not None:
            wm_result = self._world_model.update(
                state.world_model_state,
                last_obs,
                last_action,
                rew,
                transition.discount,
                obs,
            )
            new_wm_state = wm_result.state
            new_buf_state = self._buffer.add(state.buffer_state, obs)
            wm_error = wm_result.prediction_error

        # -- Steps 5/6/10/11: OaK update (real transition) -------------------
        oak_result = self._oak.update(
            state.oak_state,
            rew,
            obs,
            control_discount,
        )
        new_oak_state = oak_result.state

        # -- Step 9: guarded Dyna dreaming ------------------------------------
        dream_td_errors: Any = None

        if (
            self._world_model is not None
            and self._buffer is not None
            and self._dreamer is not None
            and self._config.n_dreams_per_step > 0
        ):
            rng_key = new_oak_state.stomp_state.rng_key
            new_oak_state, dream_td_errors = self._run_dreams(
                new_oak_state, new_wm_state, new_buf_state, rng_key
            )

        # -- Step 3: Horde GVF update -----------------------------------------
        new_horde_state = state.horde_state
        horde_tderrs: Any = None

        if self._horde is not None:
            horde_cumulants = transition.horde_cumulants
            if horde_cumulants is None:
                horde_cumulants = jnp.full(
                    (self._horde.n_demons,), rew, dtype=jnp.float32
                )
            horde_discounts = transition.horde_discounts
            if horde_discounts is None:
                # Default terminal/global rule: preserve each GVF's declared
                # horizon on every continuing transition. Only a true global
                # terminal (discount == 0) zeros all bootstraps.
                horde_spec = cast(HordeSpec, self._config.horde_spec)
                horde_discounts = jnp.where(
                    transition.discount > 0.0,
                    horde_spec.gammas,
                    jnp.zeros_like(horde_spec.gammas),
                )
            horde_result = self._horde.update_with_discounts(
                state.horde_state,
                last_obs,
                horde_cumulants,
                obs,
                horde_discounts,
            )
            new_horde_state = horde_result.state
            horde_tderrs = horde_result.td_errors

        # -- Step 12: IA update -----------------------------------------------
        new_ia_state = state.ia_state
        ia_augmented: Any = None
        ia_recommendation: Any = None

        if self._ia is not None and state.ia_state is not None:
            ia_result = self._ia.update(
                state.ia_state,
                last_obs,
                rew,
                obs,
                partner_action=last_action,
                discount=control_discount,
            )
            new_ia_state = ia_result.state
            ia_augmented = ia_result.augmented_obs
            ia_recommendation = ia_result.recommendation

        new_state = PrototypeAgentState(
            oak_state=new_oak_state,
            world_model_state=new_wm_state,
            buffer_state=new_buf_state,
            horde_state=new_horde_state,
            ia_state=new_ia_state,
            gru_state=new_gru_state,
            step_count=state.step_count + 1,
        )

        return PrototypeUpdateResult(
            state=new_state,
            action=oak_result.primitive_action,
            oak_td_error=oak_result.td_error,
            oak_average_reward=oak_result.average_reward,
            world_model_error=wm_error,
            dream_td_errors=dream_td_errors,
            horde_td_errors=horde_tderrs,
            ia_augmented_obs=ia_augmented,
            ia_recommendation=ia_recommendation,
        )

    # -- Scan-based loop ------------------------------------------------------

    def scan(
        self,
        state: PrototypeAgentState,
        rewards: Float[Array, " num_steps"],
        next_observations: Float[Array, "num_steps obs_dim"],
        horde_cumulants: Float[Array, "num_steps n_demons"] | None = None,
        discounts: Float[Array, " num_steps"] | None = None,
        horde_discounts: Float[Array, "num_steps n_demons"] | None = None,
    ) -> PrototypeArrayResult:
        """Run the agent over pre-collected transition arrays via scan.

        Suitable for simulator pre-training or offline replay.  The world
        model, Horde, and IA companion are all updated at every step.  When a
        Horde is configured but ``horde_cumulants`` is ``None``, the per-step
        reward is broadcast to all demons. Supplying ``discounts`` selects the
        explicit :class:`PrototypeTransition` path. Omitting it preserves the
        historical compatibility behavior of :meth:`update`.

        Args:
            state: Current agent state (primed with :meth:`start`).
            rewards: Scalar rewards, shape ``(num_steps,)``.
            next_observations: Next observations, shape ``(num_steps, obs_dim)``.
            horde_cumulants: Optional per-demon cumulants,
                shape ``(num_steps, n_demons)``.
            discounts: Optional explicit scalar continuation multipliers,
                shape ``(num_steps,)``.
            horde_discounts: Optional effective per-GVF discounts,
                shape ``(num_steps, n_demons)``. Requires ``discounts``.

        Returns:
            :class:`PrototypeArrayResult` with final state and per-step arrays.
        """
        if horde_discounts is not None and discounts is None:
            raise ValueError("horde_discounts requires explicit discounts")
        if self._horde is None and (
            horde_cumulants is not None or horde_discounts is not None
        ):
            raise ValueError("Horde arrays require horde_spec to be configured")

        use_explicit_discounts = discounts is not None
        use_horde_cumulants = horde_cumulants is not None
        use_horde_discounts = horde_discounts is not None
        n = int(rewards.shape[0])

        if use_explicit_discounts:
            rewards = _checked_finite_array(
                rewards,
                (n,),
                name="rewards",
            )
            raw_observation_dim = (
                self._config.gru_perception.observation_dim
                if self._config.gru_perception is not None
                else self._config.oak.observation_dim
            )
            next_observations = _checked_finite_array(
                next_observations,
                (n, raw_observation_dim),
                name="next_observations",
            )
            discounts = _checked_unit_discount(
                discounts,
                (n,),
                name="discounts",
            )
        if self._horde is not None and horde_cumulants is not None:
            horde_cumulants = _checked_finite_array(
                horde_cumulants,
                (n, self._horde.n_demons),
                name="horde_cumulants",
                allow_nan=True,
            )
        if self._horde is not None and horde_discounts is not None:
            horde_discounts = _checked_unit_discount(
                horde_discounts,
                (n, self._horde.n_demons),
                name="horde_discounts",
            )

        def step_fn(
            carry: PrototypeAgentState,
            inputs: tuple[Array, Array, Array, Array, Array],
        ) -> tuple[PrototypeAgentState, tuple[Array, Array, Array]]:
            rew, next_obs, transition_discount, hc, hd = inputs
            if use_explicit_discounts:
                result = self.update_transition(
                    carry,
                    PrototypeTransition(
                        reward=rew,
                        next_observation=next_obs,
                        discount=transition_discount,
                        horde_cumulants=hc if use_horde_cumulants else None,
                        horde_discounts=hd if use_horde_discounts else None,
                    ),
                )
            else:
                result = self.update(
                    carry,
                    rew,
                    next_obs,
                    hc if use_horde_cumulants else None,
                )
            return result.state, (
                result.action,
                result.oak_td_error,
                result.oak_average_reward,
            )

        scan_discounts = (
            discounts
            if discounts is not None
            else jnp.ones((n,), dtype=jnp.float32)
        )
        n_horde = self._horde.n_demons if self._horde is not None else 1
        scan_cumulants = (
            horde_cumulants
            if horde_cumulants is not None
            else jnp.zeros((n, n_horde), dtype=jnp.float32)
        )
        scan_horde_discounts = (
            horde_discounts
            if horde_discounts is not None
            else jnp.zeros((n, n_horde), dtype=jnp.float32)
        )
        xs = (
            rewards,
            next_observations,
            scan_discounts,
            scan_cumulants,
            scan_horde_discounts,
        )

        final_state, (actions, oak_td_errors, oak_avg_rewards) = jax.lax.scan(
            step_fn, state, xs
        )

        return PrototypeArrayResult(
            state=final_state,
            actions=actions,
            oak_td_errors=oak_td_errors,
            oak_average_rewards=oak_avg_rewards,
        )

    # -- Curation (Python-level) ----------------------------------------------

    def curate(
        self,
        state: PrototypeAgentState,
        key: Array,
        available_feature_indices: list[int] | None = None,
    ) -> tuple[PrototypeAgent, PrototypeAgentState]:
        """Replace the lowest-utility OaK option with a new subtask.

        This is a **Python-level operation** — it runs outside
        ``jax.lax.scan`` / JIT and materialises JAX array values.  Call it
        periodically in the outer Python loop.

        Args:
            state: Current agent state.
            key: JAX PRNG key for sampling the replacement feature.
            available_feature_indices: Pool of candidate feature indices.
                Defaults to all indices not currently used by any option.

        Returns:
            ``(new_agent, new_state)`` where ``new_agent`` has updated subtask
            specs and ``new_state`` has the replaced option's arrays zeroed.
        """
        new_oak, new_oak_state = self._oak.curate(
            state.oak_state, key, available_feature_indices
        )
        if new_oak is self._oak and new_oak_state is state.oak_state:
            return self, state
        new_config = PrototypeAgentConfig(
            oak=new_oak.config,
            world_model=self._config.world_model,
            dreaming=self._config.dreaming,
            buffer_capacity=self._config.buffer_capacity,
            n_dreams_per_step=self._config.n_dreams_per_step,
            dream_next_observation_mode=self._config.dream_next_observation_mode,
            horde_spec=self._config.horde_spec,
            horde_hidden_sizes=self._config.horde_hidden_sizes,
            horde_step_size=self._config.horde_step_size,
            ia=self._config.ia,
            gru_perception=self._config.gru_perception,
            auto_curate_every=self._config.auto_curate_every,
        )
        new_agent = PrototypeAgent(new_config)
        # Transfer all non-OaK sub-states unchanged
        new_state = cast(
            PrototypeAgentState,
            state.replace(oak_state=new_oak_state),
        )
        return new_agent, new_state

    def maybe_curate(
        self,
        state: PrototypeAgentState,
        key: Array,
        available_feature_indices: list[int] | None = None,
    ) -> tuple[PrototypeAgent, PrototypeAgentState]:
        """Curate if ``auto_curate_every`` steps have elapsed.

        Intended for use in the outer Python loop alongside :meth:`update`::

            for obs, reward in stream:
                state, result = agent.update(state, obs, reward, key)
                agent, state = agent.maybe_curate(state, key)

        When ``auto_curate_every == 0`` (default), this returns ``(self, state)``.

        Args:
            state: Current agent state.
            key: JAX PRNG key passed to :meth:`curate` when curation fires.
            available_feature_indices: Pool of candidate features; forwarded
                to :meth:`curate` unchanged.

        Returns:
            ``(agent, state)`` — either the updated pair from :meth:`curate`
            or ``(self, state)`` unchanged.
        """
        n = self._config.auto_curate_every
        if n <= 0 or int(state.step_count) % n != 0:
            return self, state
        return self.curate(state, key, available_feature_indices)

    def auto_subtask_specs(
        self,
        state: PrototypeAgentState,
        *,
        n_subtasks: int = 4,
    ) -> tuple[SubtaskSpec, ...]:
        """Return candidate subtask specs ranked by current Q-weight importance.

        Delegates to :func:`feature_to_subtask_specs` using the threshold,
        scale, and max-step settings from the first existing subtask spec.

        Args:
            state: Current agent state.
            n_subtasks: Number of subtask specs to return.

        Returns:
            Tuple of :class:`SubtaskSpec` instances ranked by importance.
        """
        template = self._config.oak.stomp.subtask_specs[0]
        return feature_to_subtask_specs(
            state.oak_state,
            n_subtasks=n_subtasks,
            threshold=template.threshold,
            pseudo_reward_scale=template.pseudo_reward_scale,
            max_option_steps=template.max_option_steps,
        )


def _prototype_config_digest(config: dict[str, Any]) -> str:
    """Return a canonical SHA-256 digest for a serialized agent config."""

    payload = json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def save_prototype_checkpoint(
    agent: PrototypeAgent,
    state: PrototypeAgentState,
    path: str | Path,
) -> None:
    """Persist the complete prototype config, PyTree state, and every RNG.

    The generic checkpoint layer stores the state exactly as supplied.  This
    wrapper adds the configuration needed to reconstruct a matching template
    and a digest that makes accidental config/metadata edits fail closed.
    """

    config = agent.to_config()
    save_checkpoint(
        state,
        path,
        metadata={
            "schema": PROTOTYPE_CHECKPOINT_SCHEMA,
            "agent_config": config,
            "config_sha256": _prototype_config_digest(config),
        },
    )


def load_prototype_checkpoint(
    path: str | Path,
    *,
    template_key: Array | None = None,
) -> tuple[PrototypeAgent, PrototypeAgentState]:
    """Restore a complete prototype and reject unknown/tampered metadata."""

    metadata = load_checkpoint_metadata(path)
    if metadata.get("schema") != PROTOTYPE_CHECKPOINT_SCHEMA:
        raise ValueError(
            "checkpoint is not an Alberta PrototypeAgent v1 checkpoint"
        )
    config = metadata.get("agent_config")
    if not isinstance(config, dict):
        raise ValueError("prototype checkpoint is missing agent_config")
    expected_digest = metadata.get("config_sha256")
    if not isinstance(expected_digest, str) or expected_digest != (
        _prototype_config_digest(config)
    ):
        raise ValueError("prototype checkpoint config digest does not match")

    agent = PrototypeAgent.from_config(config)
    key = jr.key(0) if template_key is None else template_key
    template = agent.init(key)
    restored, restored_metadata = load_checkpoint(template, path)
    if restored_metadata != metadata:
        raise ValueError("prototype checkpoint metadata changed between reads")
    return agent, cast(PrototypeAgentState, restored)
