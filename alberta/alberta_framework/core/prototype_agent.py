# mypy: disable-error-code="attr-defined,call-arg,no-any-return,union-attr"
"""Prototype agent assembling mechanisms across the Alberta Plan.

The :class:`PrototypeAgent` is an experimental integration of mechanisms
mapped to the Alberta Plan's retreat-and-return strategy:

- **OaK** (steps 5/6/10/11): Differential average-reward control with temporal
  abstraction, option utility tracking, and curation.
- **Horde** (step 3): Parallel GVF prediction demons sharing a learned trunk.
- **World Model** (step 8): One-step action-conditioned environment model.
- **Model-only rehearsal** (step 8): Optional bounded dual replay that updates
  only world-model ensemble members from stored real transitions.
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
import operator
from pathlib import Path
from typing import Any, Literal, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import Array
from jaxtyping import Bool, Float, Int, UInt

from alberta_framework.core.checkpoints import (
    load_checkpoint,
    load_checkpoint_metadata,
    save_checkpoint,
)
from alberta_framework.core.delight import (
    GradientJoyApplicationResult,
    GradientJoyConfig,
    GradientJoyEvidence,
    LearningValue,
    LearningValueAvailability,
    apply_gradient_joy_update,
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
from alberta_framework.core.learning_signals import (
    LearningSignalAvailability,
    TypedLearningSignals,
)
from alberta_framework.core.model_replay_rehearsal import (
    ModelReplayRehearsal,
    ModelReplayRehearsalConfig,
    RealModelReplayEvent,
)
from alberta_framework.core.oak import OaKAgent, OaKConfig, OaKState
from alberta_framework.core.options import (
    STOMPConfig,
    SubtaskSpec,
    check_option_terminated,
    compute_pseudo_reward,
)
from alberta_framework.core.representation_gradient_mixer import (
    RepresentationGradientMixDiagnostics,
    RepresentationGradientMixResult,
    RepresentationGradientMixerConfig,
    mix_representation_gradients,
)
from alberta_framework.core.state_builder import (
    IdentityStateBuilderConfig,
    OnlineGatedStateBuilderConfig,
    StateBuilder,
    StateBuilderConfig,
    StateBuilderLearningDiagnostics,
    replace_state_builder_learning_proposal_update,
    state_builder_from_config,
)
from alberta_framework.core.types import HordeSpec
from alberta_framework.core.world_model import (
    ActionConditionedWorldModel,
    ActionConditionedWorldModelConfig,
)
from alberta_framework.core.world_model_ensemble import (
    WorldModelEnsemble,
    WorldModelEnsembleConfig,
    WorldModelEnsembleDiagnostics,
    WorldModelEnsembleState,
)

PROTOTYPE_CHECKPOINT_SCHEMA = "alberta.prototype_agent.v3"
_PROTOTYPE_CHECKPOINT_SCHEMA_V2 = "alberta.prototype_agent.v2"
_PROTOTYPE_CHECKPOINT_SCHEMA_V1 = "alberta.prototype_agent.v1"
_DREAM_NEXT_OBSERVATION_STREAM_TAG = 0x44524D4F
_PROTOTYPE_V2_REPLAY_MIGRATION_TAG = 0x50525632
_UINT32_MAX = 2**32 - 1
_INT32_MAX = 2**31 - 1

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
        world_model_ensemble: Mutually exclusive bounded bootstrap ensemble.
            It emits causal typed signals and a pre-update representation
            gradient. Dreaming remains disabled on this development lane until
            uncertainty and rollout-validity gates are calibrated.
        model_replay_rehearsal: Mutually exclusive composition of the bounded
            bootstrap ensemble and fixed-capacity dual replay. Rehearsal updates
            only ensemble member models; its replay samples never update the
            actor, critic, state builder, or causal learning-signal calibrator.
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
        state_builder: Canonical causal representation builder. Its output
            width must equal ``oak.observation_dim``. Mutually exclusive with
            the deprecated fixed-weight ``gru_perception`` path.
        learn_state_builder_from_world_model: Apply the configured ensemble
            lane's causal, real pre-update representation gradient to an
            online-gated builder. Model replay gradients are never routed here.
            The proposal is formed from the source state that emitted the
            modeled representation and committed into the already-advanced
            destination state, preserving its recurrent history.
        representation_gradient_mixer: Optional successor path that explicitly
            mixes the current transition's causal control TD-loss gradient with
            the real grounded world-model gradient. Its presence enables online
            builder learning without changing the legacy flag. ``behavior_only``
            and ``discard`` can be used without an ensemble; modes with an active
            grounded-world contribution require an ensemble or model-rehearsal
            lane. Replay gradients are never eligible inputs.
        gradient_joy: Optional literal ``sparks_joy`` audit for each proposed
            builder update. Missing or incomplete probe evidence vetoes only
            the representation update; control and model learning continue.
    """

    oak: OaKConfig = dataclasses.field(default_factory=_default_oak_config)
    world_model: ActionConditionedWorldModelConfig | None = None
    world_model_ensemble: WorldModelEnsembleConfig | None = None
    model_replay_rehearsal: ModelReplayRehearsalConfig | None = None
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
    state_builder: StateBuilderConfig | None = None
    learn_state_builder_from_world_model: bool = False
    representation_gradient_mixer: RepresentationGradientMixerConfig | None = None
    gradient_joy: GradientJoyConfig | None = None
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
        if not isinstance(self.learn_state_builder_from_world_model, bool):
            raise ValueError("learn_state_builder_from_world_model must be boolean")
        mixer = self.representation_gradient_mixer
        if mixer is not None and not isinstance(
            mixer,
            RepresentationGradientMixerConfig,
        ):
            raise ValueError(
                "representation_gradient_mixer must be a "
                "RepresentationGradientMixerConfig"
            )
        configured_model_lanes = sum(
            model is not None
            for model in (
                self.world_model,
                self.world_model_ensemble,
                self.model_replay_rehearsal,
            )
        )
        if configured_model_lanes > 1:
            raise ValueError(
                "world_model, world_model_ensemble, and model_replay_rehearsal "
                "are mutually exclusive"
            )
        if self.world_model is None and self.n_dreams_per_step > 0:
            raise ValueError(
                "n_dreams_per_step > 0 requires the legacy world_model to be configured"
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
        if self.world_model_ensemble is not None:
            ensemble_model = self.world_model_ensemble.model
            if ensemble_model.observation_dim != self.oak.observation_dim:
                raise ValueError(
                    "world_model_ensemble.model.observation_dim must match "
                    "oak.observation_dim, got "
                    f"{ensemble_model.observation_dim} and "
                    f"{self.oak.observation_dim}"
                )
            if ensemble_model.n_actions != self.oak.n_primitive_actions:
                raise ValueError(
                    "world_model_ensemble.model.n_actions must match "
                    "oak.n_primitive_actions, got "
                    f"{ensemble_model.n_actions} and "
                    f"{self.oak.n_primitive_actions}"
                )
            if self.n_dreams_per_step != 0:
                raise ValueError(
                    "dreaming is disabled for world_model_ensemble until its "
                    "uncertainty and rollout-validity gates are calibrated"
                )
        if self.model_replay_rehearsal is not None:
            replay_model = self.model_replay_rehearsal.ensemble.model
            if replay_model.observation_dim != self.oak.observation_dim:
                raise ValueError(
                    "model_replay_rehearsal.ensemble.model.observation_dim must "
                    "match oak.observation_dim, got "
                    f"{replay_model.observation_dim} and {self.oak.observation_dim}"
                )
            if replay_model.n_actions != self.oak.n_primitive_actions:
                raise ValueError(
                    "model_replay_rehearsal.ensemble.model.n_actions must match "
                    "oak.n_primitive_actions, got "
                    f"{replay_model.n_actions} and {self.oak.n_primitive_actions}"
                )
            if self.n_dreams_per_step != 0:
                raise ValueError(
                    "dreaming is disabled for model_replay_rehearsal; replay is "
                    "model-only until rollout-validity gates are calibrated"
                )
        if self.gru_perception is not None and self.state_builder is not None:
            raise ValueError("state_builder and gru_perception are mutually exclusive")
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
        if self.state_builder is not None:
            builder = state_builder_from_config(self.state_builder.to_config())
            if builder.feature_dim() != self.oak.observation_dim:
                raise ValueError(
                    "state_builder feature_dim must match oak.observation_dim, "
                    f"got {builder.feature_dim()} and {self.oak.observation_dim}"
                )
            builder_actions = int(getattr(self.state_builder, "n_actions", 0))
            if builder_actions not in (0, self.oak.n_primitive_actions):
                raise ValueError(
                    "state_builder n_actions must be zero or match "
                    f"oak.n_primitive_actions ({self.oak.n_primitive_actions}), "
                    f"got {builder_actions}"
                )
            if (
                self.dream_next_observation_mode == "sample_one_hot"
                and not isinstance(self.state_builder, IdentityStateBuilderConfig)
            ):
                raise ValueError(
                    "dream_next_observation_mode='sample_one_hot' requires "
                    "the raw legacy path or an identity state_builder"
                )
        if self.learn_state_builder_from_world_model and mixer is None:
            if self.world_model_ensemble is None and self.model_replay_rehearsal is None:
                raise ValueError(
                    "learn_state_builder_from_world_model requires "
                    "world_model_ensemble or model_replay_rehearsal"
                )
        state_builder_learning_enabled = (
            self.learn_state_builder_from_world_model or mixer is not None
        )
        if state_builder_learning_enabled:
            if not isinstance(self.state_builder, OnlineGatedStateBuilderConfig):
                raise ValueError(
                    "online representation learning requires an "
                    "OnlineGatedStateBuilderConfig"
                )
        if mixer is not None:
            if mixer.representation_dim != self.oak.observation_dim:
                raise ValueError(
                    "representation_gradient_mixer.representation_dim must match "
                    f"oak.observation_dim ({self.oak.observation_dim}), got "
                    f"{mixer.representation_dim}"
                )
            grounded_world_active = (
                mixer.mode in {"full", "world_only"}
                and mixer.grounded_world_weight > 0.0
            )
            if grounded_world_active and (
                self.world_model_ensemble is None
                and self.model_replay_rehearsal is None
            ):
                raise ValueError(
                    "an active grounded-world mixer source requires "
                    "world_model_ensemble or model_replay_rehearsal"
                )
        if self.gradient_joy is not None:
            if not state_builder_learning_enabled:
                raise ValueError(
                    "gradient_joy requires online representation learning"
                )
            if (
                self.world_model_ensemble is None
                and self.model_replay_rehearsal is None
            ):
                raise ValueError(
                    "gradient_joy requires causal ensemble learning signals"
                )
            if self.gradient_joy.candidate_semantics != "update":
                raise ValueError(
                    "PrototypeAgent gradient_joy must use candidate_semantics='update'"
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
        if self.world_model_ensemble is not None:
            payload["world_model_ensemble"] = self.world_model_ensemble.to_config()
        if self.model_replay_rehearsal is not None:
            payload["model_replay_rehearsal"] = (
                self.model_replay_rehearsal.to_config()
            )
        if self.dreaming is not None:
            payload["dreaming"] = self.dreaming.to_config()
        if self.horde_spec is not None:
            payload["horde_spec"] = self.horde_spec.to_config()
        if self.ia is not None:
            payload["ia"] = self.ia.to_config()
        if self.gru_perception is not None:
            payload["gru_perception"] = self.gru_perception.to_config()
        if self.state_builder is not None:
            payload["state_builder"] = self.state_builder.to_config()
        if self.learn_state_builder_from_world_model:
            payload["learn_state_builder_from_world_model"] = True
        if self.representation_gradient_mixer is not None:
            payload["representation_gradient_mixer"] = (
                self.representation_gradient_mixer.to_config()
            )
        if self.gradient_joy is not None:
            payload["gradient_joy"] = self.gradient_joy.to_config()
        return payload

    @classmethod
    def from_config(cls, payload: dict[str, Any]) -> PrototypeAgentConfig:
        """Reconstruct from :meth:`to_config` output."""
        from alberta_framework.core.types import HordeSpec as _HordeSpec

        data = dict(payload)
        config_type = data.pop("type", None)
        if config_type != "PrototypeAgentConfig":
            raise ValueError(
                "PrototypeAgentConfig payload type must be 'PrototypeAgentConfig'"
            )
        oak = OaKConfig.from_config(cast(dict[str, Any], data.pop("oak")))

        wm_raw = data.pop("world_model", None)
        world_model = (
            ActionConditionedWorldModelConfig.from_config(wm_raw) if wm_raw is not None else None
        )
        ensemble_raw = data.pop("world_model_ensemble", None)
        world_model_ensemble = (
            WorldModelEnsembleConfig.from_config(ensemble_raw)
            if ensemble_raw is not None
            else None
        )
        model_replay_raw = data.pop("model_replay_rehearsal", None)
        model_replay_rehearsal = (
            ModelReplayRehearsalConfig.from_config(model_replay_raw)
            if model_replay_raw is not None
            else None
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
        state_builder_raw = data.pop("state_builder", None)
        state_builder: StateBuilderConfig | None = None
        if state_builder_raw is not None:
            from alberta_framework.core.state_builder import (
                state_builder_config_from_config,
            )

            state_builder = state_builder_config_from_config(state_builder_raw)

        learn_state_builder_from_world_model = data.pop(
            "learn_state_builder_from_world_model",
            False,
        )
        if not isinstance(learn_state_builder_from_world_model, bool):
            raise ValueError(
                "learn_state_builder_from_world_model must be boolean"
            )
        mixer_raw = data.pop("representation_gradient_mixer", None)
        if mixer_raw is not None and not isinstance(mixer_raw, dict):
            raise ValueError(
                "representation_gradient_mixer must be a configuration object"
            )
        representation_gradient_mixer = (
            RepresentationGradientMixerConfig.from_config(mixer_raw)
            if mixer_raw is not None
            else None
        )
        gradient_joy_raw = data.pop("gradient_joy", None)
        if gradient_joy_raw is not None and not isinstance(gradient_joy_raw, dict):
            raise ValueError("gradient_joy must be a configuration object")
        if (
            gradient_joy_raw is not None
            and gradient_joy_raw.get("type") != "GradientJoyConfig"
        ):
            raise ValueError("gradient_joy type must be 'GradientJoyConfig'")
        gradient_joy = (
            GradientJoyConfig.from_config(gradient_joy_raw)
            if gradient_joy_raw is not None
            else None
        )

        hidden = tuple(int(x) for x in data.pop("horde_hidden_sizes", [64, 64]))
        buffer_capacity = int(data.pop("buffer_capacity", 200))
        n_dreams_per_step = int(data.pop("n_dreams_per_step", 0))
        dream_next_observation_mode = data.pop(
            "dream_next_observation_mode",
            "model_prediction",
        )
        horde_step_size = float(data.pop("horde_step_size", 0.1))
        auto_curate_every = int(data.pop("auto_curate_every", 0))
        if data:
            unknown = ", ".join(sorted(data))
            raise ValueError(f"PrototypeAgentConfig payload has unknown fields: {unknown}")
        return cls(
            oak=oak,
            world_model=world_model,
            world_model_ensemble=world_model_ensemble,
            model_replay_rehearsal=model_replay_rehearsal,
            dreaming=dreaming,
            buffer_capacity=buffer_capacity,
            n_dreams_per_step=n_dreams_per_step,
            dream_next_observation_mode=dream_next_observation_mode,
            horde_spec=horde_spec,
            horde_hidden_sizes=hidden,
            horde_step_size=horde_step_size,
            ia=ia,
            gru_perception=gru_perception,
            state_builder=state_builder,
            learn_state_builder_from_world_model=(
                learn_state_builder_from_world_model
            ),
            representation_gradient_mixer=representation_gradient_mixer,
            gradient_joy=gradient_joy,
            auto_curate_every=auto_curate_every,
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
        observation: Raw observation on which ``action`` was selected.
        action: Primitive command actually dispatched to the environment.
        decision_id: Four-word session/generation token issued with that exact decision.
            This prevents a stale transition from being accepted when a raw
            observation/action pair later recurs (the ABA problem).
        reward: Scalar reward emitted by the environment.
        discount: Effective scalar continuation multiplier.
        terminated: Environment-defined terminal-state flag. A true value
            requires an explicit zero discount.
        truncated: Exogenous/time-limit boundary flag. Without simultaneous
            termination it requires a positive explicit bootstrap discount.
        next_observation: Raw final/bootstrap observation reached by the
            transition. Learning targets consume this observation.
        next_decision_observation: Raw observation on which the next command
            will be selected. It must equal ``next_observation`` off a
            boundary; an autoreset transition supplies the reset observation.
        horde_cumulants: Optional vector of one cumulant per GVF demon.
            ``NaN`` keeps the framework's existing inactive-demon semantics.
        horde_discounts: Optional vector of effective per-GVF discounts.
    """

    observation: Float[Array, " raw_observation_dim"]
    action: Int[Array, ""]
    decision_id: UInt[Array, " 4"]
    reward: Float[Array, ""]
    discount: Float[Array, ""]
    terminated: Bool[Array, ""]
    truncated: Bool[Array, ""]
    next_observation: Float[Array, " raw_observation_dim"]
    next_decision_observation: Float[Array, " raw_observation_dim"]
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
    state_builder_state: Any
    current_raw_observation: Float[Array, " raw_observation_dim"]
    current_representation: Float[Array, " observation_dim"]
    current_action: Int[Array, ""]
    current_decision_id: UInt[Array, " 4"]
    started: Bool[Array, ""]
    observation_event_count: Int[Array, ""]
    step_count: Int[Array, ""]


@chex.dataclass(frozen=True)
class _WorldModelEnsembleStateV1:
    """Exact pre-rehearsal ensemble subtree used only for v2 migration."""

    member_states: tuple[Any, ...]
    residual_variances: Any
    signal_state: Any
    bootstrap_key: Array
    last_bootstrap_mask: Any
    member_update_counts: Any
    event_count: Any


@chex.dataclass(frozen=True)
class _PrototypeAgentStateV1:
    """Exact pre-cache checkpoint template used only for v1 migration."""

    oak_state: OaKState
    world_model_state: Any
    buffer_state: Any
    horde_state: Any
    ia_state: Any
    gru_state: Any
    step_count: Int[Array, ""]


@chex.dataclass(frozen=True)
class PrototypeDecision:
    """Dispatch record that must be returned with the resulting transition."""

    observation: Float[Array, " raw_observation_dim"]
    action: Int[Array, ""]
    decision_id: UInt[Array, " 4"]
    armed: Bool[Array, ""]


@chex.dataclass(frozen=True)
class PrototypeTransitionDiagnostics:
    """Fail-closed ownership and boundary checks for one real transition."""

    started: Bool[Array, ""]
    inputs_finite: Bool[Array, ""]
    action_in_range: Bool[Array, ""]
    observation_matches: Bool[Array, ""]
    action_matches: Bool[Array, ""]
    decision_id_matches: Bool[Array, ""]
    next_generation_available: Bool[Array, ""]
    next_counter_capacity_available: Bool[Array, ""]
    discount_valid: Bool[Array, ""]
    boundary_semantics_valid: Bool[Array, ""]
    state_consistent: Bool[Array, ""]
    post_update_checked: Bool[Array, ""]
    post_update_finite: Bool[Array, ""]
    post_update_consistent: Bool[Array, ""]
    valid: Bool[Array, ""]
    rejected: Bool[Array, ""]


@chex.dataclass(frozen=True)
class PrototypeGradientJoyEvidence:
    """Decision-bound probe evidence for one representation update.

    The three parameter-gradient probes must be measured independently of the
    world-model gradient that forms the candidate update. ``advantage`` and
    ``action_surprisal`` come from the behavior actor, while ``safety_cost``
    comes from the separately trained safety learner. The remaining four
    learning-value channels are supplied causally by the configured world
    model ensemble. The paper-specific delight value is derived internally as
    ``advantage * action_surprisal`` and is never accepted from a caller.

    Every availability flag is explicit. Missing evidence, a stale decision
    ID, a false independence attestation, or any unavailable/non-finite input
    causes the literal joy audit to fail closed without invalidating the real
    control transition.
    """

    decision_id: UInt[Array, " 4"]
    objective_probe_gradient: Float[Array, " parameter_count"]
    retention_probe_gradient: Float[Array, " parameter_count"]
    safety_cost_gradient: Float[Array, " parameter_count"]
    objective_probe_available: Bool[Array, ""]
    retention_probe_available: Bool[Array, ""]
    safety_probe_available: Bool[Array, ""]
    probe_independence_attested: Bool[Array, ""]
    advantage: Float[Array, ""]
    action_surprisal: Float[Array, ""]
    safety_cost: Float[Array, ""]
    advantage_available: Bool[Array, ""]
    action_surprisal_available: Bool[Array, ""]
    safety_cost_available: Bool[Array, ""]


@chex.dataclass(frozen=True)
class PrototypeUpdateResult:
    """Result of one real-time prototype agent transition.

    The ``model_replay_*`` fields describe only work committed by the complete
    all-or-nothing model/replay transaction. They remain false/zero for the
    plain ensemble and legacy model lanes and for any rolled-back rehearsal.
    """

    state: PrototypeAgentState
    action: Int[Array, ""]
    oak_td_error: Float[Array, ""]
    oak_average_reward: Float[Array, ""]
    world_model_error: Any  # Float[Array, ""] | None
    learning_signals: TypedLearningSignals
    world_model_representation_gradient: Float[Array, " observation_dim"]
    world_model_representation_gradient_valid: Bool[Array, ""]
    world_model_ensemble_diagnostics: WorldModelEnsembleDiagnostics
    model_replay_transaction_applied: Bool[Array, ""]
    model_replay_recorded: Bool[Array, ""]
    model_replay_sampled: Bool[Array, ""]
    model_replay_updates_applied: Int[Array, ""]
    model_replay_padding_count: Int[Array, ""]
    state_builder_learning_diagnostics: StateBuilderLearningDiagnostics
    gradient_joy_application: Any  # GradientJoyApplicationResult | None
    gradient_joy_evidence_supplied: Bool[Array, ""]
    gradient_joy_decision_id_matches: Bool[Array, ""]
    dream_td_errors: Any  # Float[Array, " n_dreams"] | None
    horde_td_errors: Any  # Float[Array, " n_demons"] | None
    ia_augmented_obs: Any  # Float[Array, " augmented_dim"] | None
    ia_recommendation: Any  # Int[Array, ""] | None
    transition_diagnostics: PrototypeTransitionDiagnostics

    @property
    def sparks_joy(self) -> Bool[Array, ""]:
        """Answer the literal candidate-gradient question for this event."""
        application = self.gradient_joy_application
        if application is None:
            return jnp.asarray(False, dtype=jnp.bool_)
        return cast(GradientJoyApplicationResult, application).assessment.sparks_joy

    @property
    def joyful_gradient_applied(self) -> Bool[Array, ""]:
        """Report whether the audited finite-precision update was committed."""
        application = self.gradient_joy_application
        if application is None:
            return jnp.asarray(False, dtype=jnp.bool_)
        return (
            cast(GradientJoyApplicationResult, application).applied
            & self.state_builder_learning_diagnostics.applied
        )


@chex.dataclass(frozen=True)
class PrototypeArrayResult:
    """Result from :meth:`PrototypeAgent.scan` over a batch of transitions."""

    state: PrototypeAgentState
    actions: Int[Array, " num_steps"]
    oak_td_errors: Float[Array, " num_steps"]
    oak_average_rewards: Float[Array, " num_steps"]
    transition_valid: Bool[Array, " num_steps"]
    state_builder_learning_applied: Bool[Array, " num_steps"]
    gradient_sparks_joy: Bool[Array, " num_steps"]
    joyful_gradient_applied: Bool[Array, " num_steps"]


def _contains_tracer(value: Any) -> bool:
    """Return whether a PyTree contains a JAX tracing value."""
    return any(
        isinstance(leaf, jax.core.Tracer)
        for leaf in jax.tree_util.tree_leaves(value)
    )


def _floating_tree_is_finite(value: Any) -> Bool[Array, ""]:
    """Return whether every inexact array leaf in a PyTree is finite."""
    predicates: list[Array] = []
    for leaf in jax.tree_util.tree_leaves(value):
        dtype = getattr(leaf, "dtype", None)
        if dtype is not None and jnp.issubdtype(dtype, jnp.inexact):
            predicates.append(jnp.all(jnp.isfinite(leaf)))
    if not predicates:
        return jnp.asarray(True)
    return jnp.all(jnp.stack(predicates))


def _unavailable_learning_signals() -> TypedLearningSignals:
    """Return finite zeros whose typed channels are explicitly unavailable."""
    zero = jnp.asarray(0.0, dtype=jnp.float32)
    unavailable = jnp.asarray(False, dtype=jnp.bool_)
    return TypedLearningSignals(
        epistemic_disagreement=zero,
        epistemic_surprise=zero,
        aleatoric_uncertainty=zero,
        normalized_residual=zero,
        learning_progress=zero,
        calibrated_residual_z=zero,
        instantaneous_change_probability=zero,
        change_probability=zero,
        availability=LearningSignalAvailability(
            input_valid=unavailable,
            epistemic=unavailable,
            aleatoric=unavailable,
            normalized_residual=unavailable,
            learning_progress=unavailable,
            change_probability=unavailable,
        ),
    )


def _unavailable_ensemble_diagnostics() -> WorldModelEnsembleDiagnostics:
    """Return a fixed-shape diagnostic record for a disabled/rejected ensemble."""
    unavailable = jnp.asarray(False, dtype=jnp.bool_)
    return WorldModelEnsembleDiagnostics(
        state_valid=unavailable,
        input_valid=unavailable,
        capacity_available=unavailable,
        predictions_valid=unavailable,
        representation_gradient_valid=unavailable,
        signals_valid=unavailable,
        residual_update_valid=unavailable,
        member_updates_valid=unavailable,
        candidate_state_valid=unavailable,
        applied=unavailable,
        rejected=jnp.asarray(True, dtype=jnp.bool_),
    )


def _unavailable_state_builder_learning_diagnostics() -> StateBuilderLearningDiagnostics:
    """Return a fixed-shape rejected record when online builder learning is off."""
    zero = jnp.asarray(0.0, dtype=jnp.float32)
    unavailable = jnp.asarray(False, dtype=jnp.bool_)
    return StateBuilderLearningDiagnostics(
        gradient_norm=zero,
        clipped_gradient_norm=zero,
        parameter_update_norm=zero,
        proposal_valid=unavailable,
        source_matches=unavailable,
        capacity_available=unavailable,
        candidate_parameters_valid=unavailable,
        applied=unavailable,
        fixed_noop=unavailable,
        valid=unavailable,
        rejected=jnp.asarray(True, dtype=jnp.bool_),
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


def _strict_float_array(value: Any, shape: tuple[int, ...], *, name: str) -> Array:
    """Coerce a real numeric value while rejecting static shape/type drift."""
    array = jnp.asarray(value)
    if not (
        jnp.issubdtype(array.dtype, jnp.floating)
        or jnp.issubdtype(array.dtype, jnp.integer)
    ) or jnp.issubdtype(array.dtype, jnp.bool_):
        raise ValueError(f"{name} must have a real numeric dtype")
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    return jnp.asarray(array, dtype=jnp.float32)


def _strict_float32_array(value: Any, shape: tuple[int, ...], *, name: str) -> Array:
    """Require an exact float32 array at an optimizer-control boundary."""
    array = jnp.asarray(value)
    if array.dtype != jnp.float32 or array.shape != shape:
        raise ValueError(f"{name} must have shape {shape} and dtype float32")
    return jnp.asarray(array, dtype=jnp.float32)


def _strict_bool_scalar(value: Any, *, name: str) -> Array:
    """Require an actual scalar boolean rather than truthy numeric input."""
    array = jnp.asarray(value)
    if array.dtype != jnp.bool_ or array.shape != ():
        raise ValueError(f"{name} must be a scalar boolean")
    return jnp.asarray(array, dtype=jnp.bool_)


def _strict_action_scalar(value: Any, *, name: str) -> Array:
    """Require one scalar integer action that round-trips through int32."""
    array = jnp.asarray(value)
    if not jnp.issubdtype(array.dtype, jnp.integer) or array.shape != ():
        raise ValueError(f"{name} must be a scalar integer")
    if not _contains_tracer(value):
        try:
            exact_value = operator.index(value)
        except TypeError:
            exact_value = int(value)
        if exact_value < -(2**31) or exact_value > 2**31 - 1:
            raise ValueError(f"{name} must be losslessly representable as int32")
    converted = jnp.asarray(array, dtype=jnp.int32)
    source_unsigned = jnp.issubdtype(array.dtype, jnp.unsignedinteger)
    source_bytes = int(array.dtype.itemsize)
    if (source_unsigned and source_bytes <= 2) or (
        not source_unsigned and source_bytes <= 4
    ):
        representable = jnp.asarray(True)
    elif source_unsigned:
        representable = array <= jnp.asarray(2**31 - 1, dtype=array.dtype)
    else:
        representable = (
            (array >= jnp.asarray(-(2**31), dtype=array.dtype))
            & (array <= jnp.asarray(2**31 - 1, dtype=array.dtype))
        )
    return jnp.where(
        representable,
        converted,
        jnp.asarray(-1, dtype=jnp.int32),
    )


def _strict_uint32_words(
    value: Any,
    shape: tuple[int, ...],
    *,
    name: str,
) -> Array:
    """Require an unsigned word array without lossy source coercion."""
    array = jnp.asarray(value)
    source_dtype = getattr(value, "dtype", None)
    if source_dtype is not None and source_dtype != jnp.dtype(jnp.uint32):
        raise ValueError(f"{name} must have shape {shape} and dtype uint32")
    if array.dtype != jnp.uint32 or array.shape != shape:
        raise ValueError(f"{name} must have shape {shape} and dtype uint32")
    return jnp.asarray(array, dtype=jnp.uint32)


def _strict_decision_id(value: Any, *, name: str) -> Array:
    """Require the exact four-word lifecycle/generation token format."""
    return _strict_uint32_words(value, (4,), name=name)


def _decision_generation_available(decision_id: Array) -> Array:
    """Return whether a fresh 64-bit generation remains available."""
    maximum = jnp.asarray(_UINT32_MAX, dtype=jnp.uint32)
    return ~jnp.all(decision_id[2:] == maximum)


def _step_counter_can_process(step_count: Array) -> Bool[Array, ""]:
    """Return whether one already-dispatched transition can still be counted."""
    maximum = jnp.asarray(_INT32_MAX, dtype=jnp.int32)
    return (step_count >= 0) & (step_count < maximum)


def _next_counter_capacity_available(
    state: PrototypeAgentState,
    *,
    execution_boundary: Array,
) -> Bool[Array, ""]:
    """Reserve enough signed-counter capacity for the next dispatched action."""
    maximum = jnp.asarray(_INT32_MAX, dtype=jnp.int32)
    next_step_count = _saturating_int32_increment(state.step_count)
    next_observation_count = jnp.where(
        execution_boundary,
        _saturating_int32_increment(
            _saturating_int32_increment(state.observation_event_count)
        ),
        _saturating_int32_increment(state.observation_event_count),
    )
    # A future transition may itself be an autoreset boundary and consume two
    # observation events, so retain two free observation-event slots.
    return (next_step_count < maximum) & (next_observation_count <= maximum - 2)


def _increment_decision_id(decision_id: Array) -> Array:
    """Increment a big-endian pair of uint32 words without x64 mode."""
    one = jnp.asarray(1, dtype=jnp.uint32)
    low = decision_id[3] + one
    carry = (low == jnp.asarray(0, dtype=jnp.uint32)).astype(jnp.uint32)
    high = decision_id[2] + carry
    return jnp.stack((decision_id[0], decision_id[1], high, low))


def _saturating_int32_increment(value: Array) -> Array:
    """Advance a non-negative lifecycle counter without wraparound."""
    maximum = jnp.asarray(_INT32_MAX, dtype=jnp.int32)
    counter = jnp.asarray(value, dtype=jnp.int32)
    return jnp.minimum(jnp.maximum(counter, 0), maximum - 1) + 1


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
        decision = agent.decision(state)
        while True:
            reward, next_obs, discount, terminated, truncated = env.step(
                int(decision.action)
            )
            result = agent.update_transition(
                state,
                PrototypeTransition(
                    observation=decision.observation,
                    action=decision.action,
                    decision_id=decision.decision_id,
                    reward=reward,
                    discount=discount,
                    terminated=terminated,
                    truncated=truncated,
                    next_observation=next_obs,
                    next_decision_observation=next_obs,
                ),
            )
            state = result.state
            decision = agent.decision(state)

    Periodic curation (Python-level, outside JAX)::

        if step % curation_interval == 0:
            agent, state = agent.curate(state, key)
    """

    def __init__(self, config: PrototypeAgentConfig) -> None:
        self._config = config
        self._oak = OaKAgent(config.oak)
        self._state_builder: StateBuilder[Any] | None = (
            state_builder_from_config(config.state_builder.to_config())
            if config.state_builder is not None
            else None
        )

        self._world_model: ActionConditionedWorldModel | None = None
        self._world_model_ensemble: WorldModelEnsemble | None = None
        self._model_replay_rehearsal: ModelReplayRehearsal | None = None
        self._buffer: RecentObservationBuffer | None = None
        self._dreamer: GuardedDreamer | None = None
        if config.world_model is not None:
            self._world_model = ActionConditionedWorldModel(config.world_model)
            self._buffer = RecentObservationBuffer(
                config.buffer_capacity, config.oak.observation_dim
            )
            self._dreamer = GuardedDreamer(config.dreaming or DreamingConfig())
        elif config.world_model_ensemble is not None:
            self._world_model_ensemble = WorldModelEnsemble(
                config.world_model_ensemble
            )
        elif config.model_replay_rehearsal is not None:
            self._model_replay_rehearsal = ModelReplayRehearsal(
                config.model_replay_rehearsal
            )

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

    @property
    def state_builder(self) -> StateBuilder[Any] | None:
        """Canonical state builder, or ``None`` for a legacy raw/GRU path."""
        return self._state_builder

    @property
    def world_model_ensemble(self) -> WorldModelEnsemble | None:
        """Return the bounded ensemble, or ``None`` on legacy/no-model lanes.

        On the rehearsal lane its matching child state is
        ``state.world_model_state.ensemble_state``; the complete Prototype
        world-model state must still be advanced through
        :attr:`model_replay_rehearsal` so replay remains atomic.
        """
        if self._model_replay_rehearsal is not None:
            return self._model_replay_rehearsal.ensemble
        return self._world_model_ensemble

    @property
    def model_replay_rehearsal(self) -> ModelReplayRehearsal | None:
        """Return the model-only rehearsal composer when configured."""
        return self._model_replay_rehearsal

    def _state_builder_parameter_count(self) -> int:
        """Return the configured learnable builder vector width."""
        builder_config = self._config.state_builder
        if not isinstance(builder_config, OnlineGatedStateBuilderConfig):
            raise RuntimeError("online state-builder learning is not configured")
        return builder_config.parameter_count()

    def _missing_gradient_joy_evidence(
        self,
        decision_id: Array,
    ) -> PrototypeGradientJoyEvidence:
        """Construct explicit unavailable evidence for a fail-closed audit."""
        zeros = jnp.zeros(
            (self._state_builder_parameter_count(),),
            dtype=jnp.float32,
        )
        zero = jnp.asarray(0.0, dtype=jnp.float32)
        unavailable = jnp.asarray(False, dtype=jnp.bool_)
        return PrototypeGradientJoyEvidence(
            decision_id=decision_id,
            objective_probe_gradient=zeros,
            retention_probe_gradient=zeros,
            safety_cost_gradient=zeros,
            objective_probe_available=unavailable,
            retention_probe_available=unavailable,
            safety_probe_available=unavailable,
            probe_independence_attested=unavailable,
            advantage=zero,
            action_surprisal=zero,
            safety_cost=zero,
            advantage_available=unavailable,
            action_surprisal_available=unavailable,
            safety_cost_available=unavailable,
        )

    def _normalize_gradient_joy_evidence(
        self,
        state: PrototypeAgentState,
        evidence: PrototypeGradientJoyEvidence | None,
    ) -> tuple[PrototypeGradientJoyEvidence | None, Array, Array]:
        """Validate one optimizer-control sidecar without owning the transition.

        Static shape or dtype drift raises. Runtime non-finiteness and a stale
        decision binding clear the corresponding availability flags so the joy
        audit rejects, while the authoritative environment transition remains
        eligible for normal control/model learning.
        """
        if self._config.gradient_joy is None:
            if evidence is not None:
                raise ValueError(
                    "gradient_joy_evidence requires gradient_joy to be configured"
                )
            unavailable = jnp.asarray(False, dtype=jnp.bool_)
            return None, unavailable, unavailable

        if evidence is None:
            unavailable = jnp.asarray(False, dtype=jnp.bool_)
            return (
                self._missing_gradient_joy_evidence(state.current_decision_id),
                unavailable,
                unavailable,
            )
        if not isinstance(evidence, PrototypeGradientJoyEvidence):
            raise TypeError(
                "gradient_joy_evidence must be PrototypeGradientJoyEvidence"
            )

        parameter_shape = (self._state_builder_parameter_count(),)
        decision_id = _strict_decision_id(
            evidence.decision_id,
            name="gradient_joy_evidence.decision_id",
        )
        decision_matches = jnp.array_equal(
            decision_id,
            state.current_decision_id,
        )
        gradients = {
            "objective_probe_gradient": _strict_float32_array(
                evidence.objective_probe_gradient,
                parameter_shape,
                name="gradient_joy_evidence.objective_probe_gradient",
            ),
            "retention_probe_gradient": _strict_float32_array(
                evidence.retention_probe_gradient,
                parameter_shape,
                name="gradient_joy_evidence.retention_probe_gradient",
            ),
            "safety_cost_gradient": _strict_float32_array(
                evidence.safety_cost_gradient,
                parameter_shape,
                name="gradient_joy_evidence.safety_cost_gradient",
            ),
        }
        scalars = {
            "advantage": _strict_float32_array(
                evidence.advantage,
                (),
                name="gradient_joy_evidence.advantage",
            ),
            "action_surprisal": _strict_float32_array(
                evidence.action_surprisal,
                (),
                name="gradient_joy_evidence.action_surprisal",
            ),
            "safety_cost": _strict_float32_array(
                evidence.safety_cost,
                (),
                name="gradient_joy_evidence.safety_cost",
            ),
        }
        flags = {
            "objective_probe_available": _strict_bool_scalar(
                evidence.objective_probe_available,
                name="gradient_joy_evidence.objective_probe_available",
            ),
            "retention_probe_available": _strict_bool_scalar(
                evidence.retention_probe_available,
                name="gradient_joy_evidence.retention_probe_available",
            ),
            "safety_probe_available": _strict_bool_scalar(
                evidence.safety_probe_available,
                name="gradient_joy_evidence.safety_probe_available",
            ),
            "probe_independence_attested": _strict_bool_scalar(
                evidence.probe_independence_attested,
                name="gradient_joy_evidence.probe_independence_attested",
            ),
            "advantage_available": _strict_bool_scalar(
                evidence.advantage_available,
                name="gradient_joy_evidence.advantage_available",
            ),
            "action_surprisal_available": _strict_bool_scalar(
                evidence.action_surprisal_available,
                name="gradient_joy_evidence.action_surprisal_available",
            ),
            "safety_cost_available": _strict_bool_scalar(
                evidence.safety_cost_available,
                name="gradient_joy_evidence.safety_cost_available",
            ),
        }
        gradient_finite = {
            name: jnp.all(jnp.isfinite(value))
            for name, value in gradients.items()
        }
        scalar_valid = {
            "advantage": jnp.isfinite(scalars["advantage"]),
            "action_surprisal": (
                jnp.isfinite(scalars["action_surprisal"])
                & (scalars["action_surprisal"] >= 0.0)
            ),
            "safety_cost": jnp.isfinite(scalars["safety_cost"]),
        }
        normalized = PrototypeGradientJoyEvidence(
            decision_id=decision_id,
            objective_probe_gradient=jnp.where(
                gradient_finite["objective_probe_gradient"],
                gradients["objective_probe_gradient"],
                jnp.zeros_like(gradients["objective_probe_gradient"]),
            ),
            retention_probe_gradient=jnp.where(
                gradient_finite["retention_probe_gradient"],
                gradients["retention_probe_gradient"],
                jnp.zeros_like(gradients["retention_probe_gradient"]),
            ),
            safety_cost_gradient=jnp.where(
                gradient_finite["safety_cost_gradient"],
                gradients["safety_cost_gradient"],
                jnp.zeros_like(gradients["safety_cost_gradient"]),
            ),
            objective_probe_available=(
                flags["objective_probe_available"]
                & gradient_finite["objective_probe_gradient"]
                & decision_matches
            ),
            retention_probe_available=(
                flags["retention_probe_available"]
                & gradient_finite["retention_probe_gradient"]
                & decision_matches
            ),
            safety_probe_available=(
                flags["safety_probe_available"]
                & gradient_finite["safety_cost_gradient"]
                & decision_matches
            ),
            probe_independence_attested=(
                flags["probe_independence_attested"] & decision_matches
            ),
            advantage=jnp.where(
                scalar_valid["advantage"],
                scalars["advantage"],
                jnp.asarray(0.0, dtype=jnp.float32),
            ),
            action_surprisal=jnp.where(
                scalar_valid["action_surprisal"],
                scalars["action_surprisal"],
                jnp.asarray(0.0, dtype=jnp.float32),
            ),
            safety_cost=jnp.where(
                scalar_valid["safety_cost"],
                scalars["safety_cost"],
                jnp.asarray(0.0, dtype=jnp.float32),
            ),
            advantage_available=(
                flags["advantage_available"]
                & scalar_valid["advantage"]
                & decision_matches
            ),
            action_surprisal_available=(
                flags["action_surprisal_available"]
                & scalar_valid["action_surprisal"]
                & decision_matches
            ),
            safety_cost_available=(
                flags["safety_cost_available"]
                & scalar_valid["safety_cost"]
                & decision_matches
            ),
        )
        return (
            normalized,
            jnp.asarray(True, dtype=jnp.bool_),
            decision_matches,
        )

    @staticmethod
    def _gradient_joy_evidence_from_signals(
        sidecar: PrototypeGradientJoyEvidence,
        signals: TypedLearningSignals,
        representation_gradient_valid: Array,
    ) -> GradientJoyEvidence:
        """Join decision-bound probes to causal ensemble learning signals."""
        candidate_available = jnp.asarray(
            representation_gradient_valid,
            dtype=jnp.bool_,
        )
        signal_input_available = signals.availability.input_valid
        delight = sidecar.advantage * sidecar.action_surprisal
        delight_finite = jnp.isfinite(delight)
        safe_delight = jnp.where(
            delight_finite,
            delight,
            jnp.asarray(0.0, dtype=jnp.float32),
        )
        advantage_available = sidecar.advantage_available & candidate_available
        surprisal_available = (
            sidecar.action_surprisal_available & candidate_available
        )
        return GradientJoyEvidence(
            objective_probe_gradient=sidecar.objective_probe_gradient,
            retention_probe_gradient=sidecar.retention_probe_gradient,
            safety_cost_gradient=sidecar.safety_cost_gradient,
            objective_probe_available=(
                sidecar.objective_probe_available & candidate_available
            ),
            retention_probe_available=(
                sidecar.retention_probe_available & candidate_available
            ),
            safety_probe_available=(
                sidecar.safety_probe_available & candidate_available
            ),
            probe_independence_attested=(
                sidecar.probe_independence_attested & candidate_available
            ),
            learning_value=LearningValue(
                advantage=sidecar.advantage,
                action_surprisal=sidecar.action_surprisal,
                delight=safe_delight,
                epistemic_surprise=signals.epistemic_surprise,
                aleatoric_uncertainty=signals.aleatoric_uncertainty,
                learning_progress=signals.learning_progress,
                change_probability=signals.change_probability,
                safety_cost=sidecar.safety_cost,
            ),
            learning_value_availability=LearningValueAvailability(
                advantage=advantage_available,
                action_surprisal=surprisal_available,
                delight=(
                    advantage_available
                    & surprisal_available
                    & delight_finite
                ),
                epistemic_surprise=(
                    candidate_available
                    & signal_input_available
                    & signals.availability.epistemic
                ),
                aleatoric_uncertainty=(
                    candidate_available
                    & signal_input_available
                    & signals.availability.aleatoric
                ),
                learning_progress=(
                    candidate_available
                    & signal_input_available
                    & signals.availability.learning_progress
                ),
                change_probability=(
                    candidate_available
                    & signal_input_available
                    & signals.availability.change_probability
                ),
                safety_cost=(
                    sidecar.safety_cost_available & candidate_available
                ),
            ),
        )

    def _apply_state_builder_learning(
        self,
        source_state: Any,
        destination_state: Any,
        representation_gradient: Array,
        representation_gradient_valid: Array,
        signals: TypedLearningSignals,
        joy_sidecar: PrototypeGradientJoyEvidence | None,
    ) -> tuple[
        Any,
        StateBuilderLearningDiagnostics,
        GradientJoyApplicationResult | None,
    ]:
        """Propose from the causal source and commit into advanced recurrence."""
        if not self._config.learn_state_builder_from_world_model:
            return (
                destination_state,
                _unavailable_state_builder_learning_diagnostics(),
                None,
            )
        if self._state_builder is None:
            raise RuntimeError("state builder learning has no configured builder")

        proposal = self._state_builder.propose_learning_update(
            source_state,
            representation_gradient,
        )
        application: GradientJoyApplicationResult | None = None
        approved = jnp.asarray(
            representation_gradient_valid,
            dtype=jnp.bool_,
        )
        candidate_update = proposal.candidate_parameter_update
        if self._config.gradient_joy is not None:
            if joy_sidecar is None:
                raise RuntimeError("configured gradient joy requires a sidecar")
            audit_evidence = self._gradient_joy_evidence_from_signals(
                joy_sidecar,
                signals,
                approved,
            )
            application = apply_gradient_joy_update(
                proposal.source_parameters,
                candidate_update,
                audit_evidence,
                self._config.gradient_joy,
            )
            candidate_update = application.parameters - proposal.source_parameters
            approved = approved & proposal.valid & application.applied

        filtered_proposal = replace_state_builder_learning_proposal_update(
            proposal,
            candidate_update,
            approved,
        )
        learned_state, learning_diagnostics = (
            self._state_builder.commit_learning_update(
                destination_state,
                filtered_proposal,
            )
        )
        return learned_state, learning_diagnostics, application

    def _raw_observation_dim(self) -> int:
        if self._state_builder is not None:
            return self._state_builder.observation_dim()
        if self._config.gru_perception is not None:
            return self._config.gru_perception.observation_dim
        return self._config.oak.observation_dim

    def _state_numeric_valid(self, state: PrototypeAgentState) -> Array:
        """Validate state numerics while preserving world-model init sentinels."""
        sanitized_state = state
        world_model_bounds_valid = jnp.asarray(True)
        state_builder_valid = jnp.asarray(True)
        if self._state_builder is not None:
            state_builder_valid = self._state_builder.state_valid(
                state.state_builder_state
            )
        if (
            self._model_replay_rehearsal is not None
            and state.world_model_state is not None
        ):
            world_model_bounds_valid = self._model_replay_rehearsal.state_valid(
                state.world_model_state
            )
            sanitized_state = cast(
                PrototypeAgentState,
                state.replace(world_model_state=None),
            )
        elif (
            self._world_model_ensemble is not None
            and state.world_model_state is not None
        ):
            world_model_bounds_valid = self._world_model_ensemble.state_valid(
                state.world_model_state
            )
            sanitized_state = cast(
                PrototypeAgentState,
                state.replace(world_model_state=None),
            )
        elif self._world_model is not None and state.world_model_state is not None:
            world_model_state = state.world_model_state
            observation_min = world_model_state.observation_min
            observation_max = world_model_state.observation_max
            reward_min = world_model_state.reward_min
            reward_max = world_model_state.reward_max
            pristine_bounds = (
                jnp.all(jnp.isposinf(observation_min))
                & jnp.all(jnp.isneginf(observation_max))
                & jnp.isposinf(reward_min)
                & jnp.isneginf(reward_max)
            )
            finite_bounds = (
                jnp.all(jnp.isfinite(observation_min))
                & jnp.all(jnp.isfinite(observation_max))
                & jnp.isfinite(reward_min)
                & jnp.isfinite(reward_max)
            )
            world_model_bounds_valid = jnp.where(
                world_model_state.step_count == 0,
                pristine_bounds,
                finite_bounds,
            )
            sanitized_world_model_state = world_model_state.replace(
                observation_min=jnp.zeros_like(observation_min),
                observation_max=jnp.zeros_like(observation_max),
                reward_min=jnp.zeros_like(reward_min),
                reward_max=jnp.zeros_like(reward_max),
            )
            sanitized_state = cast(
                PrototypeAgentState,
                state.replace(world_model_state=sanitized_world_model_state),
            )
        return (
            world_model_bounds_valid
            & state_builder_valid
            & _floating_tree_is_finite(sanitized_state)
        )

    def _representation_cache_consistent(
        self,
        state: PrototypeAgentState,
    ) -> Array:
        """Check that causal representation caches agree with their owners."""
        representation_consistent = jnp.array(True)
        builder_count_consistent = jnp.array(True)
        if self._state_builder is not None:
            rebuilt_representation = self._state_builder.encode(
                state.state_builder_state,
                state.current_raw_observation,
            )
            representation_consistent = jnp.array_equal(
                rebuilt_representation,
                state.current_representation,
            )
            builder_count_consistent = (
                state.state_builder_state.step_count
                == state.observation_event_count
            )
        elif state.gru_state is not None:
            representation_consistent = jnp.array_equal(
                jnp.concatenate(
                    (state.current_raw_observation, state.gru_state.hidden)
                ),
                state.current_representation,
            )
        else:
            representation_consistent = jnp.array_equal(
                state.current_raw_observation,
                state.current_representation,
            )
        return (
            jnp.all(jnp.isfinite(state.current_raw_observation))
            & jnp.all(jnp.isfinite(state.current_representation))
            & (state.observation_event_count >= 1)
            & (state.oak_state.step_count == state.step_count)
            & (state.oak_state.stomp_state.step_count == state.step_count)
            & representation_consistent
            & builder_count_consistent
            & jnp.array_equal(
                state.current_representation,
                state.oak_state.stomp_state.base_last_obs,
            )
        )

    def _state_cache_structurally_consistent(
        self,
        state: PrototypeAgentState,
    ) -> Array:
        """Check the structure of the currently armed representation/action cache."""
        action_in_range = (state.current_action >= 0) & (
            state.current_action < self._config.oak.n_primitive_actions
        )
        observation_maximum = jnp.asarray(_INT32_MAX, dtype=jnp.int32)
        return (
            self._representation_cache_consistent(state)
            & action_in_range
            & (state.current_action == state.oak_state.stomp_state.last_primitive_action)
            & _step_counter_can_process(state.step_count)
            & (state.observation_event_count <= observation_maximum - 2)
        )

    def _state_cache_consistent(self, state: PrototypeAgentState) -> Array:
        """Check the complete finite currently armed decision cache."""
        return (
            self._state_numeric_valid(state)
            & self._state_cache_structurally_consistent(state)
        )

    def _disarmed_state_structurally_consistent(
        self,
        state: PrototypeAgentState,
    ) -> Array:
        """Validate a state disarmed by decision- or counter-capacity exhaustion."""
        counter = state.current_decision_id[2:]
        generation_maximum = jnp.asarray(_UINT32_MAX, dtype=jnp.uint32)
        step_maximum = jnp.asarray(_INT32_MAX, dtype=jnp.int32)
        exhausted = jnp.all(counter == generation_maximum) | (
            state.step_count >= step_maximum - 1
        ) | (state.observation_event_count >= step_maximum - 1)
        return (
            (~state.started)
            & (state.step_count > 0)
            & (state.current_action == -1)
            & exhausted
            & self._representation_cache_consistent(state)
        )

    def _checkpoint_state_structurally_valid(
        self,
        state: PrototypeAgentState,
    ) -> Array:
        """Validate armed, pristine, or capacity-exhausted checkpoint structure."""
        pristine = self._pristine_state_structurally_consistent(state)
        disarmed = self._disarmed_state_structurally_consistent(state)
        armed = state.started & self._state_cache_structurally_consistent(state)
        return pristine | disarmed | armed

    def _checkpoint_state_valid(self, state: PrototypeAgentState) -> Array:
        """Validate checkpoint structure and every floating state leaf."""
        return (
            self._state_numeric_valid(state)
            & self._checkpoint_state_structurally_valid(state)
        )

    def _pristine_state_structurally_consistent(
        self,
        state: PrototypeAgentState,
    ) -> Array:
        """Return whether ``state`` has the structure of an untouched init."""
        recurrent_fresh = jnp.array(True)
        if self._state_builder is not None:
            recurrent_fresh = state.state_builder_state.step_count == 0
        elif state.gru_state is not None:
            recurrent_fresh = jnp.all(state.gru_state.hidden == 0.0)
        return (
            (~state.started)
            & (state.step_count == 0)
            & (state.observation_event_count == 0)
            & (state.current_action == -1)
            & jnp.all(state.current_decision_id[2:] == 0)
            & jnp.all(state.current_raw_observation == 0.0)
            & jnp.all(state.current_representation == 0.0)
            & (state.oak_state.step_count == 0)
            & (state.oak_state.stomp_state.step_count == 0)
            & jnp.all(state.oak_state.stomp_state.base_last_obs == 0.0)
            & recurrent_fresh
        )

    def _pristine_state_consistent(self, state: PrototypeAgentState) -> Array:
        """Return whether ``state`` is a finite untouched result of :meth:`init`."""
        return (
            self._state_numeric_valid(state)
            & self._pristine_state_structurally_consistent(state)
        )

    # -- Serialization --------------------------------------------------------

    def to_config(self) -> dict[str, Any]:
        """Serialize agent configuration."""
        return self._config.to_config()

    @classmethod
    def from_config(cls, payload: dict[str, Any]) -> PrototypeAgent:
        """Reconstruct from :meth:`to_config` output."""
        return cls(PrototypeAgentConfig.from_config(payload))

    # -- Lifecycle ------------------------------------------------------------

    def init(
        self,
        key: Array,
        *,
        lifecycle_id: Array | None = None,
    ) -> PrototypeAgentState:
        """Initialise all sub-states.

        Args:
            key: JAX PRNG key.
            lifecycle_id: Optional caller-owned two-word uint32 session nonce.
                Supplying a persisted runner epoch gives exact cross-session
                transition ownership. When omitted, a deterministic nonce is
                derived from ``key``; callers must then avoid key reuse across
                concurrently valid sessions.

        Returns:
            Fresh :class:`PrototypeAgentState`.
        """
        session_key, oak_key, wm_key, horde_key, ia_key, gru_key, builder_key = jr.split(
            key,
            7,
        )
        if lifecycle_id is None:
            lifecycle_words = jr.key_data(session_key)
        else:
            lifecycle_words = _strict_uint32_words(
                lifecycle_id,
                (2,),
                name="lifecycle_id",
            )
        oak_state = self._oak.init(oak_key)

        wm_state: Any = None
        buf_state: Any = None
        if self._world_model is not None and self._buffer is not None:
            wm_state = self._world_model.init(wm_key)
            buf_state = self._buffer.init()
        elif self._world_model_ensemble is not None:
            wm_state = self._world_model_ensemble.init(wm_key)
        elif self._model_replay_rehearsal is not None:
            wm_state = self._model_replay_rehearsal.init(wm_key)

        horde_state: Any = None
        if self._horde is not None:
            horde_state = self._horde.init(self._config.oak.observation_dim, horde_key)

        ia_state: Any = None
        if self._ia is not None:
            ia_state = self._ia.init(ia_key)

        gru_state: Any = None
        if self._config.gru_perception is not None:
            gru_state = _init_gru_state(self._config.gru_perception, gru_key)

        builder_state: Any = None
        if self._state_builder is not None:
            builder_state = self._state_builder.init(builder_key)

        initial_state = PrototypeAgentState(
            oak_state=oak_state,
            world_model_state=wm_state,
            buffer_state=buf_state,
            horde_state=horde_state,
            ia_state=ia_state,
            gru_state=gru_state,
            state_builder_state=builder_state,
            current_raw_observation=jnp.zeros(
                (self._raw_observation_dim(),),
                dtype=jnp.float32,
            ),
            current_representation=jnp.zeros(
                (self._config.oak.observation_dim,),
                dtype=jnp.float32,
            ),
            current_action=jnp.array(-1, dtype=jnp.int32),
            current_decision_id=jnp.concatenate(
                (lifecycle_words, jnp.zeros((2,), dtype=jnp.uint32))
            ),
            started=jnp.array(False),
            observation_event_count=jnp.array(0, dtype=jnp.int32),
            step_count=jnp.array(0, dtype=jnp.int32),
        )
        return cast(
            PrototypeAgentState,
            jax.tree.map(
                lambda leaf: (
                    jnp.asarray(leaf)
                    if isinstance(leaf, (bool, int, float))
                    else leaf
                ),
                initial_state,
            ),
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
        raw_obs = _strict_float_array(
            initial_observation,
            (self._raw_observation_dim(),),
            name="initial_observation",
        )
        finite = jnp.all(jnp.isfinite(raw_obs))
        pristine = self._pristine_state_consistent(state)
        if not _contains_tracer((raw_obs, state.started, state.step_count)):
            if not bool(pristine):
                raise RuntimeError("start requires a fresh unstarted PrototypeAgentState")
            if not bool(finite):
                raise ValueError("initial_observation must be finite")
        safe_raw_obs = jnp.where(finite, raw_obs, jnp.zeros_like(raw_obs))

        def prime(fresh_state: PrototypeAgentState) -> PrototypeAgentState:
            new_gru_state = fresh_state.gru_state
            new_builder_state = fresh_state.state_builder_state
            obs_for_oak = safe_raw_obs
            if self._state_builder is not None:
                new_builder_state, obs_for_oak = self._state_builder.start(
                    fresh_state.state_builder_state,
                    safe_raw_obs,
                )
            elif fresh_state.gru_state is not None:
                new_gru_state, obs_for_oak = _gru_step(
                    fresh_state.gru_state,
                    safe_raw_obs,
                )
            new_oak = self._oak.start(fresh_state.oak_state, obs_for_oak)
            new_ia = fresh_state.ia_state
            if self._ia is not None and fresh_state.ia_state is not None:
                new_ia = self._ia.start(fresh_state.ia_state, obs_for_oak)
            candidate = cast(
                PrototypeAgentState,
                fresh_state.replace(
                    oak_state=new_oak,
                    ia_state=new_ia,
                    gru_state=new_gru_state,
                    state_builder_state=new_builder_state,
                    current_raw_observation=safe_raw_obs,
                    current_representation=obs_for_oak,
                    current_action=new_oak.stomp_state.last_primitive_action,
                    current_decision_id=fresh_state.current_decision_id.at[2:].set(
                        jnp.zeros((2,), dtype=jnp.uint32)
                    ),
                    started=jnp.array(True),
                    observation_event_count=jnp.array(1, dtype=jnp.int32),
                ),
            )
            return cast(
                PrototypeAgentState,
                jax.lax.cond(
                    self._state_cache_consistent(candidate),
                    lambda _: candidate,
                    lambda _: fresh_state,
                    operand=None,
                ),
            )

        result = cast(
            PrototypeAgentState,
            jax.lax.cond(pristine & finite, prime, lambda unchanged: unchanged, state),
        )
        if not _contains_tracer((raw_obs, state.started, state.step_count)):
            if bool(pristine & finite) and not bool(result.started):
                raise ValueError("initial_observation produced a non-finite agent state")
        return result

    def act(
        self,
        state: PrototypeAgentState,
        observation: Array,
    ) -> Int[Array, ""]:
        """Return the cached primitive command for the current observation.

        This method never advances or re-encodes recurrent state. The retained
        observation argument is an ownership assertion for legacy callers;
        eager mismatches raise and traced mismatches return the unarmed ``-1``
        sentinel. Use :meth:`greedy_action` for a pure counterfactual query
        that was not selected for dispatch.
        """
        obs = _strict_float_array(
            observation,
            (self._raw_observation_dim(),),
            name="observation",
        )
        matches = jnp.all(jnp.isfinite(obs)) & jnp.array_equal(
            obs,
            state.current_raw_observation,
        )
        state_consistent = self._state_cache_consistent(state)
        valid = state.started & state_consistent & matches
        if not _contains_tracer((obs, state.started, state.current_action)):
            if not bool(state.started):
                raise ValueError("agent must be started before act")
            if not bool(state_consistent):
                raise ValueError("agent decision cache is inconsistent")
            if not bool(matches):
                raise ValueError("observation does not match the cached decision")
        return jnp.where(
            valid,
            state.current_action,
            jnp.asarray(-1, dtype=jnp.int32),
        )

    def decision(self, state: PrototypeAgentState) -> PrototypeDecision:
        """Return the complete currently armed decision record.

        Carry this record across the environment boundary and copy its three
        fields into :class:`PrototypeTransition`. The generation closes the
        replay ambiguity left by comparing observation and action alone. A
        well-formed state that exhausted its decision or counter capacity
        returns the explicit unarmed ``-1`` sentinel instead of raising.
        """
        armed_consistent = state.started & self._state_cache_consistent(state)
        disarmed_consistent = (
            self._state_numeric_valid(state)
            & self._disarmed_state_structurally_consistent(state)
        )
        valid = armed_consistent | disarmed_consistent
        if not _contains_tracer((state.started, state.current_action)) and not bool(valid):
            raise ValueError("agent state does not contain a valid decision lifecycle")
        action = jnp.where(
            armed_consistent,
            state.current_action,
            jnp.asarray(-1, dtype=jnp.int32),
        )
        return PrototypeDecision(
            observation=state.current_raw_observation,
            action=action,
            decision_id=state.current_decision_id,
            armed=armed_consistent,
        )

    def greedy_action(
        self,
        state: PrototypeAgentState,
        observation: Array,
    ) -> Int[Array, ""]:
        """Pure counterfactual greedy action without advancing causal state."""
        raw_obs = _strict_float_array(
            observation,
            (self._raw_observation_dim(),),
            name="observation",
        )
        obs = raw_obs
        if self._state_builder is not None:
            obs = self._state_builder.encode(state.state_builder_state, raw_obs)
        elif state.gru_state is not None:
            obs = jnp.concatenate([raw_obs, state.gru_state.hidden], axis=0)
        n_prim = self._config.oak.n_primitive_actions
        all_q = self._oak.base_q_values(state.oak_state, obs)
        return jnp.argmax(all_q[:n_prim]).astype(jnp.int32)

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
        if self._state_builder is not None:
            raise ValueError(
                "legacy update is unavailable with state_builder; use "
                "update_transition with an explicit preceding discount"
            )
        legacy_model_discount = (
            self._config.world_model.gamma
            if self._config.world_model is not None
            else (
                self._config.world_model_ensemble.model.gamma
                if self._config.world_model_ensemble is not None
                else (
                    self._config.model_replay_rehearsal.ensemble.model.gamma
                    if self._config.model_replay_rehearsal is not None
                    else 1.0
                )
            )
        )
        legacy_representation = state.oak_state.stomp_state.base_last_obs
        legacy_state = cast(
            PrototypeAgentState,
            state.replace(
                current_raw_observation=legacy_representation[
                    : self._raw_observation_dim()
                ],
                current_representation=legacy_representation,
                current_action=state.oak_state.stomp_state.last_primitive_action,
            ),
        )
        transition, diagnostics = self._normalize_transition(
            legacy_state,
            PrototypeTransition(
                observation=legacy_state.current_raw_observation,
                action=legacy_state.current_action,
                decision_id=legacy_state.current_decision_id,
                reward=reward,
                discount=jnp.asarray(legacy_model_discount, dtype=jnp.float32),
                terminated=jnp.array(False),
                truncated=jnp.array(False),
                next_observation=next_observation,
                next_decision_observation=next_observation,
                horde_cumulants=horde_cumulants,
            )
        )
        return self._update_transition_impl(
            legacy_state,
            transition,
            diagnostics,
            control_discount=None,
            gradient_joy_evidence=None,
            gradient_joy_evidence_supplied=jnp.asarray(False),
            gradient_joy_decision_id_matches=jnp.asarray(False),
        )

    def update_transition(
        self,
        state: PrototypeAgentState,
        transition: PrototypeTransition,
        gradient_joy_evidence: PrototypeGradientJoyEvidence | None = None,
    ) -> PrototypeUpdateResult:
        """Process one explicit real-time continuing transition.

        Execution order:

        1. Update the legacy world model or bounded ensemble from the real
           transition (if configured).
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
        normalized_joy_evidence, joy_supplied, joy_decision_matches = (
            self._normalize_gradient_joy_evidence(
                state,
                gradient_joy_evidence,
            )
        )
        normalized, diagnostics = self._normalize_transition(state, transition)
        return self._update_transition_impl(
            state,
            normalized,
            diagnostics,
            control_discount=normalized.discount,
            gradient_joy_evidence=normalized_joy_evidence,
            gradient_joy_evidence_supplied=joy_supplied,
            gradient_joy_decision_id_matches=joy_decision_matches,
        )

    def _normalize_transition(
        self,
        state: PrototypeAgentState,
        transition: PrototypeTransition,
    ) -> tuple[PrototypeTransition, PrototypeTransitionDiagnostics]:
        """Validate ownership/semantics and produce safe branch operands."""
        raw_observation_dim = self._raw_observation_dim()
        observation = _strict_float_array(
            transition.observation,
            (raw_observation_dim,),
            name="transition.observation",
        )
        action = _strict_action_scalar(
            transition.action,
            name="transition.action",
        )
        decision_id = _strict_decision_id(
            transition.decision_id,
            name="transition.decision_id",
        )
        reward = _strict_float_array(
            transition.reward,
            (),
            name="transition.reward",
        )
        next_observation = _strict_float_array(
            transition.next_observation,
            (raw_observation_dim,),
            name="transition.next_observation",
        )
        next_decision_observation = _strict_float_array(
            transition.next_decision_observation,
            (raw_observation_dim,),
            name="transition.next_decision_observation",
        )
        discount = _strict_float_array(
            transition.discount,
            (),
            name="transition.discount",
        )
        terminated = _strict_bool_scalar(
            transition.terminated,
            name="transition.terminated",
        )
        truncated = _strict_bool_scalar(
            transition.truncated,
            name="transition.truncated",
        )

        inputs_finite = (
            jnp.all(jnp.isfinite(observation))
            & jnp.isfinite(reward)
            & jnp.isfinite(discount)
            & jnp.all(jnp.isfinite(next_observation))
            & jnp.all(jnp.isfinite(next_decision_observation))
        )
        action_in_range = (action >= 0) & (
            action < self._config.oak.n_primitive_actions
        )
        observation_matches = jnp.array_equal(
            observation,
            state.current_raw_observation,
        )
        action_matches = action == state.current_action
        decision_id_matches = jnp.array_equal(
            decision_id,
            state.current_decision_id,
        )
        next_generation_available = _decision_generation_available(
            state.current_decision_id,
        )
        discount_valid = jnp.isfinite(discount) & (discount >= 0.0) & (discount <= 1.0)
        boundary = terminated | truncated
        next_counter_capacity_available = _next_counter_capacity_available(
            state,
            execution_boundary=boundary,
        )
        observations_match_off_boundary = boundary | jnp.array_equal(
            next_observation,
            next_decision_observation,
        )
        boundary_semantics_valid = (
            ((discount == 0.0) == terminated)
            & ((~(truncated & ~terminated)) | (discount > 0.0))
            & observations_match_off_boundary
        )
        started = state.started
        state_consistent = self._state_cache_consistent(state)

        horde_cumulants: Any = None
        horde_discounts: Any = None
        horde_valid = jnp.array(True)
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
                horde_cumulants = _strict_float_array(
                    transition.horde_cumulants,
                    (n_demons,),
                    name="transition.horde_cumulants",
                )
                horde_valid = horde_valid & jnp.all(~jnp.isinf(horde_cumulants))
            if transition.horde_discounts is not None:
                horde_discounts = _strict_float_array(
                    transition.horde_discounts,
                    (n_demons,),
                    name="transition.horde_discounts",
                )
                horde_valid = horde_valid & jnp.all(
                    jnp.isfinite(horde_discounts)
                    & (horde_discounts >= 0.0)
                    & (horde_discounts <= 1.0)
                )

        valid = (
            started
            & inputs_finite
            & action_in_range
            & observation_matches
            & action_matches
            & decision_id_matches
            & discount_valid
            & boundary_semantics_valid
            & state_consistent
            & horde_valid
        )
        diagnostics = PrototypeTransitionDiagnostics(
            started=started,
            inputs_finite=inputs_finite & horde_valid,
            action_in_range=action_in_range,
            observation_matches=observation_matches,
            action_matches=action_matches,
            decision_id_matches=decision_id_matches,
            next_generation_available=next_generation_available,
            next_counter_capacity_available=next_counter_capacity_available,
            discount_valid=discount_valid,
            boundary_semantics_valid=boundary_semantics_valid,
            state_consistent=state_consistent,
            post_update_checked=jnp.asarray(False),
            post_update_finite=jnp.asarray(False),
            post_update_consistent=jnp.asarray(False),
            valid=valid,
            rejected=~valid,
        )

        safe_observation = jnp.where(inputs_finite, observation, state.current_raw_observation)
        safe_next_observation = jnp.where(
            inputs_finite,
            next_observation,
            state.current_raw_observation,
        )
        safe_next_decision_observation = jnp.where(
            inputs_finite,
            next_decision_observation,
            state.current_raw_observation,
        )
        safe_action = jnp.where(action_in_range, action, state.current_action)
        safe_reward = jnp.where(jnp.isfinite(reward), reward, 0.0)
        safe_discount = jnp.where(discount_valid, discount, 0.0)
        if horde_cumulants is not None:
            horde_cumulants = jnp.where(
                ~jnp.isinf(horde_cumulants),
                horde_cumulants,
                jnp.full_like(horde_cumulants, jnp.nan),
            )
        if horde_discounts is not None:
            horde_discounts = jnp.where(
                jnp.isfinite(horde_discounts)
                & (horde_discounts >= 0.0)
                & (horde_discounts <= 1.0),
                horde_discounts,
                jnp.zeros_like(horde_discounts),
            )

        normalized = PrototypeTransition(
            observation=safe_observation,
            action=safe_action,
            decision_id=state.current_decision_id,
            reward=safe_reward,
            discount=safe_discount,
            terminated=terminated,
            truncated=truncated,
            next_observation=safe_next_observation,
            next_decision_observation=safe_next_decision_observation,
            horde_cumulants=horde_cumulants,
            horde_discounts=horde_discounts,
        )
        return normalized, diagnostics

    def _rejected_update_result(
        self,
        state: PrototypeAgentState,
        diagnostics: PrototypeTransitionDiagnostics,
        *,
        gradient_joy_evidence: PrototypeGradientJoyEvidence | None,
        gradient_joy_evidence_supplied: Array,
        gradient_joy_decision_id_matches: Array,
    ) -> PrototypeUpdateResult:
        """Return neutral diagnostics and preserve ``state`` bit-for-bit."""
        zero = jnp.array(0.0, dtype=jnp.float32)
        world_model_error: Any = (
            zero
            if (
                self._world_model is not None
                or self._world_model_ensemble is not None
                or self._model_replay_rehearsal is not None
            )
            else None
        )
        dream_td_errors: Any = None
        if self._world_model is not None and self._config.n_dreams_per_step > 0:
            dream_td_errors = jnp.zeros(
                (self._config.n_dreams_per_step,),
                dtype=jnp.float32,
            )
        horde_td_errors: Any = None
        if self._horde is not None:
            horde_td_errors = jnp.zeros(
                (self._horde.n_demons,),
                dtype=jnp.float32,
            )
        ia_augmented_obs: Any = None
        ia_recommendation: Any = None
        if self._ia is not None and self._config.ia is not None:
            ia_augmented_obs = jnp.zeros(
                (self._config.ia.augmented_obs_dim,),
                dtype=jnp.float32,
            )
            ia_recommendation = state.current_action
        (
            _,
            builder_learning_diagnostics,
            gradient_joy_application,
        ) = self._apply_state_builder_learning(
            state.state_builder_state,
            state.state_builder_state,
            jnp.zeros(
                (self._config.oak.observation_dim,),
                dtype=jnp.float32,
            ),
            jnp.asarray(False),
            _unavailable_learning_signals(),
            gradient_joy_evidence,
        )
        return PrototypeUpdateResult(
            state=state,
            action=state.current_action,
            oak_td_error=zero,
            oak_average_reward=zero,
            world_model_error=world_model_error,
            learning_signals=_unavailable_learning_signals(),
            world_model_representation_gradient=jnp.zeros(
                (self._config.oak.observation_dim,),
                dtype=jnp.float32,
            ),
            world_model_representation_gradient_valid=jnp.asarray(False),
            world_model_ensemble_diagnostics=_unavailable_ensemble_diagnostics(),
            model_replay_transaction_applied=jnp.asarray(False),
            model_replay_recorded=jnp.asarray(False),
            model_replay_sampled=jnp.asarray(False),
            model_replay_updates_applied=jnp.asarray(0, dtype=jnp.int32),
            model_replay_padding_count=jnp.asarray(0, dtype=jnp.int32),
            state_builder_learning_diagnostics=builder_learning_diagnostics,
            gradient_joy_application=gradient_joy_application,
            gradient_joy_evidence_supplied=gradient_joy_evidence_supplied,
            gradient_joy_decision_id_matches=(
                gradient_joy_decision_id_matches
            ),
            dream_td_errors=dream_td_errors,
            horde_td_errors=horde_td_errors,
            ia_augmented_obs=ia_augmented_obs,
            ia_recommendation=ia_recommendation,
            transition_diagnostics=diagnostics,
        )

    def _update_transition_impl(
        self,
        state: PrototypeAgentState,
        transition: PrototypeTransition,
        diagnostics: PrototypeTransitionDiagnostics,
        *,
        control_discount: Array | None,
        gradient_joy_evidence: PrototypeGradientJoyEvidence | None,
        gradient_joy_evidence_supplied: Array,
        gradient_joy_decision_id_matches: Array,
    ) -> PrototypeUpdateResult:
        """Atomically apply a valid transition or return an exact no-op."""

        def valid_branch(_: None) -> PrototypeUpdateResult:
            result = self._apply_valid_transition_impl(
                state,
                transition,
                diagnostics,
                control_discount=control_discount,
                gradient_joy_evidence=gradient_joy_evidence,
                gradient_joy_evidence_supplied=(
                    gradient_joy_evidence_supplied
                ),
                gradient_joy_decision_id_matches=(
                    gradient_joy_decision_id_matches
                ),
            )
            post_finite = self._state_numeric_valid(result.state)
            post_consistent = self._checkpoint_state_structurally_valid(
                result.state
            )
            post_valid = post_finite & post_consistent
            final_diagnostics = cast(
                PrototypeTransitionDiagnostics,
                diagnostics.replace(
                    post_update_checked=jnp.asarray(True),
                    post_update_finite=post_finite,
                    post_update_consistent=post_consistent,
                    valid=diagnostics.valid & post_valid,
                    rejected=~(diagnostics.valid & post_valid),
                ),
            )
            accepted = cast(
                PrototypeUpdateResult,
                result.replace(transition_diagnostics=final_diagnostics),
            )
            return jax.lax.cond(
                post_valid,
                lambda __: accepted,
                lambda __: self._rejected_update_result(
                    state,
                    final_diagnostics,
                    gradient_joy_evidence=gradient_joy_evidence,
                    gradient_joy_evidence_supplied=(
                        gradient_joy_evidence_supplied
                    ),
                    gradient_joy_decision_id_matches=(
                        gradient_joy_decision_id_matches
                    ),
                ),
                operand=None,
            )

        def rejected_branch(_: None) -> PrototypeUpdateResult:
            return self._rejected_update_result(
                state,
                diagnostics,
                gradient_joy_evidence=gradient_joy_evidence,
                gradient_joy_evidence_supplied=(
                    gradient_joy_evidence_supplied
                ),
                gradient_joy_decision_id_matches=(
                    gradient_joy_decision_id_matches
                ),
            )

        return jax.lax.cond(
            diagnostics.valid,
            valid_branch,
            rejected_branch,
            operand=None,
        )

    def _apply_valid_transition_impl(
        self,
        state: PrototypeAgentState,
        transition: PrototypeTransition,
        diagnostics: PrototypeTransitionDiagnostics,
        *,
        control_discount: Array | None,
        gradient_joy_evidence: PrototypeGradientJoyEvidence | None,
        gradient_joy_evidence_supplied: Array,
        gradient_joy_decision_id_matches: Array,
    ) -> PrototypeUpdateResult:
        """Apply one normalized, ownership-validated real transition."""
        bootstrap_raw_obs = transition.next_observation
        decision_raw_obs = transition.next_decision_observation
        execution_boundary = transition.terminated | transition.truncated
        rew = transition.reward

        # -- Step 8a: consume bootstrap state, then restart once at a boundary -
        new_gru_state = state.gru_state
        new_builder_state = state.state_builder_state
        bootstrap_obs = bootstrap_raw_obs
        decision_obs = decision_raw_obs
        if self._state_builder is not None:
            bootstrap_builder_state, bootstrap_obs = self._state_builder.update(
                state.state_builder_state,
                bootstrap_raw_obs,
                transition.action,
                rew,
                transition.discount,
            )

            def restart_builder(builder_state: Any) -> tuple[Any, Array]:
                reset_state = self._state_builder.reset_episode(builder_state)
                return self._state_builder.start(reset_state, decision_raw_obs)

            new_builder_state, decision_obs = jax.lax.cond(
                execution_boundary,
                restart_builder,
                lambda builder_state: (builder_state, bootstrap_obs),
                bootstrap_builder_state,
            )
        elif state.gru_state is not None:
            bootstrap_gru_state, bootstrap_obs = _gru_step(
                state.gru_state,
                bootstrap_raw_obs,
            )

            def restart_gru(gru_state: GRUPerceptionState) -> tuple[Any, Array]:
                reset_state = cast(
                    GRUPerceptionState,
                    gru_state.replace(hidden=jnp.zeros_like(gru_state.hidden)),
                )
                return _gru_step(reset_state, decision_raw_obs)

            new_gru_state, decision_obs = jax.lax.cond(
                execution_boundary,
                restart_gru,
                lambda gru_state: (gru_state, bootstrap_obs),
                bootstrap_gru_state,
            )

        # Snapshot the real last observation/action before OaK update. The
        # base action may be an extended option index, but the world model is
        # action-conditioned on the primitive command sent to the environment.
        last_obs = state.current_representation
        last_action = transition.action

        # -- Step 8b: world model update (real transition) --------------------
        new_wm_state = state.world_model_state
        new_buf_state = state.buffer_state
        wm_error: Any = None
        learning_signals = _unavailable_learning_signals()
        representation_gradient = jnp.zeros(
            (self._config.oak.observation_dim,),
            dtype=jnp.float32,
        )
        representation_gradient_valid = jnp.asarray(False)
        ensemble_diagnostics = _unavailable_ensemble_diagnostics()
        model_replay_transaction_applied = jnp.asarray(False)
        model_replay_recorded = jnp.asarray(False)
        model_replay_sampled = jnp.asarray(False)
        model_replay_updates_applied = jnp.asarray(0, dtype=jnp.int32)
        model_replay_padding_count = jnp.asarray(0, dtype=jnp.int32)

        if self._world_model is not None and self._buffer is not None:
            wm_result = self._world_model.update(
                state.world_model_state,
                last_obs,
                last_action,
                rew,
                transition.discount,
                bootstrap_obs,
            )
            new_wm_state = wm_result.state
            bootstrap_buffer_state = self._buffer.add(
                state.buffer_state,
                bootstrap_obs,
            )
            new_buf_state = jax.lax.cond(
                execution_boundary,
                lambda buffer_state: self._buffer.add(buffer_state, decision_obs),
                lambda buffer_state: buffer_state,
                bootstrap_buffer_state,
            )
            wm_error = wm_result.prediction_error
        elif self._world_model_ensemble is not None:
            ensemble_result = self._world_model_ensemble.update(
                state.world_model_state,
                last_obs,
                last_action,
                rew,
                transition.discount,
                bootstrap_obs,
            )
            new_wm_state = ensemble_result.state
            wm_error = ensemble_result.observed_loss
            learning_signals = ensemble_result.signals
            representation_gradient = ensemble_result.representation_gradient
            representation_gradient_valid = (
                ensemble_result.representation_gradient_valid
            )
            ensemble_diagnostics = ensemble_result.diagnostics
        elif self._model_replay_rehearsal is not None:
            representation_version = (
                state.state_builder_state.update_count
                if isinstance(
                    self._config.state_builder,
                    OnlineGatedStateBuilderConfig,
                )
                else jnp.asarray(0, dtype=jnp.int32)
            )
            rehearsal_result = self._model_replay_rehearsal.step(
                state.world_model_state,
                RealModelReplayEvent(
                    observation=last_obs,
                    action=last_action,
                    reward=rew,
                    discount=transition.discount,
                    terminated=transition.terminated,
                    truncated=transition.truncated,
                    next_observation=bootstrap_obs,
                    representation_version=representation_version,
                    provenance_id=state.step_count,
                    source_id=jnp.asarray(0, dtype=jnp.int32),
                    safety_cost=jnp.asarray(0.0, dtype=jnp.float32),
                    safety_cost_available=jnp.asarray(False, dtype=jnp.bool_),
                    valid=diagnostics.valid,
                ),
            )
            new_wm_state = rehearsal_result.state
            wm_error = rehearsal_result.real_observed_loss
            learning_signals = rehearsal_result.real_signals
            representation_gradient = (
                rehearsal_result.real_representation_gradient
            )
            representation_gradient_valid = (
                rehearsal_result.real_representation_gradient_valid
            )
            ensemble_diagnostics = cast(
                WorldModelEnsembleDiagnostics,
                rehearsal_result.real_update_diagnostics.replace(
                    applied=rehearsal_result.diagnostics.transaction_applied,
                    rejected=~rehearsal_result.diagnostics.transaction_applied,
                ),
            )
            model_replay_transaction_applied = (
                rehearsal_result.diagnostics.transaction_applied
            )
            model_replay_recorded = (
                rehearsal_result.diagnostics.transaction_applied
                & rehearsal_result.diagnostics.replay_recorded
            )
            model_replay_sampled = (
                rehearsal_result.diagnostics.transaction_applied
                & rehearsal_result.diagnostics.replay_sampled
            )
            model_replay_updates_applied = jnp.where(
                rehearsal_result.diagnostics.transaction_applied,
                jnp.sum(
                    rehearsal_result.trace.model_updates_applied.astype(
                        jnp.int32
                    )
                ),
                jnp.asarray(0, dtype=jnp.int32),
            )
            model_replay_padding_count = jnp.where(
                rehearsal_result.diagnostics.transaction_applied,
                jnp.sum(rehearsal_result.trace.padding.astype(jnp.int32)),
                jnp.asarray(0, dtype=jnp.int32),
            )

        # Learn the representation causally: the proposal is formed from the
        # source state whose sensitivity emitted ``last_obs``. It is committed
        # into the destination state only after that destination has consumed
        # the real transition (and any autoreset observation), so recurrence is
        # neither recomputed under new parameters nor replaced by stale state.
        (
            new_builder_state,
            builder_learning_diagnostics,
            gradient_joy_application,
        ) = self._apply_state_builder_learning(
            state.state_builder_state,
            new_builder_state,
            representation_gradient,
            representation_gradient_valid,
            learning_signals,
            gradient_joy_evidence,
        )

        # -- Steps 5/6/10/11: OaK update (real transition) -------------------
        oak_result = self._oak.update(
            state.oak_state,
            rew,
            bootstrap_obs,
            control_discount,
            decision_observation=decision_obs,
            execution_boundary=execution_boundary,
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
                bootstrap_obs,
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
                bootstrap_obs,
                partner_action=last_action,
                discount=control_discount,
                decision_observation=decision_obs,
                execution_boundary=execution_boundary,
            )
            new_ia_state = ia_result.state
            ia_augmented = ia_result.augmented_obs
            ia_recommendation = ia_result.recommendation

        next_armed = (
            diagnostics.next_generation_available
            & diagnostics.next_counter_capacity_available
        )
        next_action = jnp.where(
            next_armed,
            oak_result.primitive_action,
            jnp.asarray(-1, dtype=jnp.int32),
        )
        next_decision_id = jnp.where(
            next_armed,
            _increment_decision_id(state.current_decision_id),
            state.current_decision_id,
        )
        new_state = PrototypeAgentState(
            oak_state=new_oak_state,
            world_model_state=new_wm_state,
            buffer_state=new_buf_state,
            horde_state=new_horde_state,
            ia_state=new_ia_state,
            gru_state=new_gru_state,
            state_builder_state=new_builder_state,
            current_raw_observation=decision_raw_obs,
            current_representation=decision_obs,
            current_action=next_action,
            current_decision_id=next_decision_id,
            started=next_armed,
            observation_event_count=jnp.where(
                execution_boundary,
                _saturating_int32_increment(
                    _saturating_int32_increment(state.observation_event_count)
                ),
                _saturating_int32_increment(state.observation_event_count),
            ),
            step_count=_saturating_int32_increment(state.step_count),
        )

        return PrototypeUpdateResult(
            state=new_state,
            action=next_action,
            oak_td_error=oak_result.td_error,
            oak_average_reward=oak_result.average_reward,
            world_model_error=wm_error,
            learning_signals=learning_signals,
            world_model_representation_gradient=representation_gradient,
            world_model_representation_gradient_valid=(
                representation_gradient_valid
            ),
            world_model_ensemble_diagnostics=ensemble_diagnostics,
            model_replay_transaction_applied=model_replay_transaction_applied,
            model_replay_recorded=model_replay_recorded,
            model_replay_sampled=model_replay_sampled,
            model_replay_updates_applied=model_replay_updates_applied,
            model_replay_padding_count=model_replay_padding_count,
            state_builder_learning_diagnostics=builder_learning_diagnostics,
            gradient_joy_application=gradient_joy_application,
            gradient_joy_evidence_supplied=gradient_joy_evidence_supplied,
            gradient_joy_decision_id_matches=(
                gradient_joy_decision_id_matches
            ),
            dream_td_errors=dream_td_errors,
            horde_td_errors=horde_tderrs,
            ia_augmented_obs=ia_augmented,
            ia_recommendation=ia_recommendation,
            transition_diagnostics=diagnostics,
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
        if self._state_builder is not None:
            raise ValueError(
                "legacy array scan is unavailable with state_builder; use "
                "scan_transitions with the explicit transition contract"
            )
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
        ) -> tuple[
            PrototypeAgentState,
            tuple[Array, Array, Array, Array, Array, Array, Array],
        ]:
            rew, next_obs, transition_discount, hc, hd = inputs
            if use_explicit_discounts:
                result = self.update_transition(
                    carry,
                    PrototypeTransition(
                        observation=carry.current_raw_observation,
                        action=carry.current_action,
                        decision_id=carry.current_decision_id,
                        reward=rew,
                        discount=transition_discount,
                        terminated=transition_discount == 0.0,
                        truncated=jnp.array(False),
                        next_observation=next_obs,
                        next_decision_observation=next_obs,
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
                result.transition_diagnostics.valid,
                result.state_builder_learning_diagnostics.applied,
                result.sparks_joy,
                result.joyful_gradient_applied,
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

        final_state, outputs = jax.lax.scan(step_fn, state, xs)
        (
            actions,
            oak_td_errors,
            oak_avg_rewards,
            transition_valid,
            builder_learning_applied,
            gradient_sparks_joy,
            joyful_gradient_applied,
        ) = outputs

        return PrototypeArrayResult(
            state=final_state,
            actions=actions,
            oak_td_errors=oak_td_errors,
            oak_average_rewards=oak_avg_rewards,
            transition_valid=transition_valid,
            state_builder_learning_applied=builder_learning_applied,
            gradient_sparks_joy=gradient_sparks_joy,
            joyful_gradient_applied=joyful_gradient_applied,
        )

    def scan_transitions(
        self,
        state: PrototypeAgentState,
        transitions: PrototypeTransition,
        gradient_joy_evidence: PrototypeGradientJoyEvidence | None = None,
    ) -> PrototypeArrayResult:
        """Run authoritative transitions and optional joy sidecars through scan."""

        def transition_only_step(
            carry: PrototypeAgentState,
            transition: PrototypeTransition,
        ) -> tuple[
            PrototypeAgentState,
            tuple[Array, Array, Array, Array, Array, Array, Array],
        ]:
            result = self.update_transition(carry, transition)
            return result.state, (
                result.action,
                result.oak_td_error,
                result.oak_average_reward,
                result.transition_diagnostics.valid,
                result.state_builder_learning_diagnostics.applied,
                result.sparks_joy,
                result.joyful_gradient_applied,
            )

        if gradient_joy_evidence is None:
            final_state, outputs = jax.lax.scan(
                transition_only_step,
                state,
                transitions,
            )
        else:
            def transition_with_joy_step(
                carry: PrototypeAgentState,
                inputs: tuple[PrototypeTransition, PrototypeGradientJoyEvidence],
            ) -> tuple[
                PrototypeAgentState,
                tuple[Array, Array, Array, Array, Array, Array, Array],
            ]:
                transition, sidecar = inputs
                result = self.update_transition(carry, transition, sidecar)
                return result.state, (
                    result.action,
                    result.oak_td_error,
                    result.oak_average_reward,
                    result.transition_diagnostics.valid,
                    result.state_builder_learning_diagnostics.applied,
                    result.sparks_joy,
                    result.joyful_gradient_applied,
                )

            final_state, outputs = jax.lax.scan(
                transition_with_joy_step,
                state,
                (transitions, gradient_joy_evidence),
            )
        (
            actions,
            td_errors,
            average_rewards,
            transition_valid,
            builder_learning_applied,
            gradient_sparks_joy,
            joyful_gradient_applied,
        ) = outputs
        return PrototypeArrayResult(
            state=final_state,
            actions=actions,
            oak_td_errors=td_errors,
            oak_average_rewards=average_rewards,
            transition_valid=transition_valid,
            state_builder_learning_applied=builder_learning_applied,
            gradient_sparks_joy=gradient_sparks_joy,
            joyful_gradient_applied=joyful_gradient_applied,
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
            world_model_ensemble=self._config.world_model_ensemble,
            model_replay_rehearsal=self._config.model_replay_rehearsal,
            dreaming=self._config.dreaming,
            buffer_capacity=self._config.buffer_capacity,
            n_dreams_per_step=self._config.n_dreams_per_step,
            dream_next_observation_mode=self._config.dream_next_observation_mode,
            horde_spec=self._config.horde_spec,
            horde_hidden_sizes=self._config.horde_hidden_sizes,
            horde_step_size=self._config.horde_step_size,
            ia=self._config.ia,
            gru_perception=self._config.gru_perception,
            state_builder=self._config.state_builder,
            learn_state_builder_from_world_model=(
                self._config.learn_state_builder_from_world_model
            ),
            gradient_joy=self._config.gradient_joy,
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

    if not bool(agent._checkpoint_state_valid(state)):
        raise ValueError("cannot save an inconsistent PrototypeAgent state")

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
    trust_v1_started: bool = False,
) -> tuple[PrototypeAgent, PrototypeAgentState]:
    """Restore a complete prototype and reject unknown/tampered metadata.

    Version-2 checkpoints with the bounded ensemble predate isolated
    model-rehearsal state. Their learned members, residual statistics, causal
    signal state, real bootstrap key/mask, and real counters are preserved;
    only the new replay key/mask/counters are deterministically initialized.

    Version-1 checkpoints did not record whether ``start`` had run, so their
    action cache cannot be armed safely by inference. Pass
    ``trust_v1_started=True`` only when external provenance establishes that
    the stored OaK decision was already selected for dispatch.
    """

    metadata = load_checkpoint_metadata(path)
    schema = metadata.get("schema")
    if schema not in {
        PROTOTYPE_CHECKPOINT_SCHEMA,
        _PROTOTYPE_CHECKPOINT_SCHEMA_V2,
        _PROTOTYPE_CHECKPOINT_SCHEMA_V1,
    }:
        raise ValueError(
            "checkpoint is not an Alberta PrototypeAgent v1/v2/v3 checkpoint"
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
    if agent.to_config() != config:
        raise ValueError("prototype checkpoint agent_config is not canonical")
    key = jr.key(0) if template_key is None else template_key
    template = agent.init(key)
    if schema == PROTOTYPE_CHECKPOINT_SCHEMA:
        restored, restored_metadata = load_checkpoint(template, path)
        if restored_metadata != metadata:
            raise ValueError("prototype checkpoint metadata changed between reads")
        restored_state = cast(PrototypeAgentState, restored)
        if not bool(agent._checkpoint_state_valid(restored_state)):
            raise ValueError("prototype checkpoint decision/cache state is inconsistent")
        return agent, restored_state

    if schema == _PROTOTYPE_CHECKPOINT_SCHEMA_V2:
        if agent.config.world_model_ensemble is None:
            restored, restored_metadata = load_checkpoint(template, path)
            if restored_metadata != metadata:
                raise ValueError("prototype checkpoint metadata changed between reads")
            restored_state = cast(PrototypeAgentState, restored)
            if not bool(agent._checkpoint_state_valid(restored_state)):
                raise ValueError(
                    "prototype v2 checkpoint decision/cache state is inconsistent"
                )
            return agent, restored_state

        current_world = cast(WorldModelEnsembleState, template.world_model_state)
        legacy_world_template = _WorldModelEnsembleStateV1(
            member_states=current_world.member_states,
            residual_variances=current_world.residual_variances,
            signal_state=current_world.signal_state,
            bootstrap_key=current_world.bootstrap_key,
            last_bootstrap_mask=current_world.last_bootstrap_mask,
            member_update_counts=current_world.member_update_counts,
            event_count=current_world.event_count,
        )
        legacy_template = template.replace(world_model_state=legacy_world_template)
        restored, restored_metadata = load_checkpoint(legacy_template, path)
        if restored_metadata != metadata:
            raise ValueError("prototype checkpoint metadata changed between reads")
        restored_state = cast(PrototypeAgentState, restored)
        legacy_world = cast(_WorldModelEnsembleStateV1, restored_state.world_model_state)
        migrated_world = WorldModelEnsembleState(
            member_states=legacy_world.member_states,
            residual_variances=legacy_world.residual_variances,
            signal_state=legacy_world.signal_state,
            bootstrap_key=legacy_world.bootstrap_key,
            replay_bootstrap_key=jr.fold_in(
                legacy_world.bootstrap_key,
                _PROTOTYPE_V2_REPLAY_MIGRATION_TAG,
            ),
            last_bootstrap_mask=legacy_world.last_bootstrap_mask,
            last_replay_bootstrap_mask=jnp.zeros_like(
                legacy_world.last_bootstrap_mask,
                dtype=jnp.bool_,
            ),
            member_update_counts=legacy_world.member_update_counts,
            replay_member_update_counts=jnp.zeros_like(
                legacy_world.member_update_counts,
                dtype=jnp.int32,
            ),
            event_count=legacy_world.event_count,
            replay_event_count=jnp.asarray(0, dtype=jnp.int32),
        )
        migrated = restored_state.replace(world_model_state=migrated_world)
        if not bool(agent._checkpoint_state_valid(migrated)):
            raise ValueError("prototype v2 ensemble checkpoint migration is inconsistent")
        return agent, migrated

    if agent.config.state_builder is not None:
        raise ValueError("v1 prototype checkpoints cannot contain state_builder config")
    if not trust_v1_started:
        raise ValueError(
            "v1 prototype checkpoint lifecycle is ambiguous; pass "
            "trust_v1_started=True only for a provenance-verified primed state"
        )
    v1_template = _PrototypeAgentStateV1(
        oak_state=template.oak_state,
        world_model_state=template.world_model_state,
        buffer_state=template.buffer_state,
        horde_state=template.horde_state,
        ia_state=template.ia_state,
        gru_state=template.gru_state,
        step_count=template.step_count,
    )
    restored_v1, restored_metadata = load_checkpoint(v1_template, path)
    if restored_metadata != metadata:
        raise ValueError("prototype checkpoint metadata changed between reads")
    old_state = cast(_PrototypeAgentStateV1, restored_v1)
    representation = old_state.oak_state.stomp_state.base_last_obs
    raw_dim = agent._raw_observation_dim()
    raw_observation = representation[:raw_dim]
    migrated = PrototypeAgentState(
        oak_state=old_state.oak_state,
        world_model_state=old_state.world_model_state,
        buffer_state=old_state.buffer_state,
        horde_state=old_state.horde_state,
        ia_state=old_state.ia_state,
        gru_state=old_state.gru_state,
        state_builder_state=None,
        current_raw_observation=raw_observation,
        current_representation=representation,
        current_action=old_state.oak_state.stomp_state.last_primitive_action,
        current_decision_id=template.current_decision_id,
        started=jnp.array(True),
        observation_event_count=_saturating_int32_increment(
            old_state.step_count,
        ),
        step_count=old_state.step_count,
    )
    if not bool(agent._checkpoint_state_valid(migrated)):
        raise ValueError("trusted v1 prototype checkpoint cache state is inconsistent")
    return agent, migrated
