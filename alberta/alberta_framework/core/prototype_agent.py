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
import math
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
    CandidateUpdateAuditApplicationResult,
    CandidateUpdateAuditConfig,
    CandidateUpdateAuditEvidence,
    LearningValue,
    LearningValueAvailability,
    apply_candidate_update,
)
from alberta_framework.core.dreaming import (
    DreamingConfig,
    GuardedDreamer,
    RecentObservationBuffer,
    RecentObservationBufferState,
)
from alberta_framework.core.experiential_memory import (
    ExperientialMemory,
    ExperientialMemoryConfig,
    ExperientialMemoryEntry,
    ExperientialMemoryState,
    ExperientialMemoryStepResult,
)
from alberta_framework.core.experiential_memory_policy import (
    ExperientialMemoryAdvantageGate,
    ExperientialMemoryAdvantageGateConfig,
    ExperientialMemoryAdvantageGateDiagnostics,
    ExperientialMemoryPolicy,
    ExperientialMemoryPolicyProposal,
)
from alberta_framework.core.horde import HordeLearner
from alberta_framework.core.intelligence_amplification import (
    IAAgent,
    IAConfig,
    IAUpdateResult,
)
from alberta_framework.core.interaction_features import (
    CURATION_ACTIVE_INELIGIBLE_RANK,
    CURATION_CANDIDATE_INELIGIBLE_RANK,
    InteractionCurationPriorityOverride,
)
from alberta_framework.core.learning_signals import (
    LearningSignalAvailability,
    LearningSignalCounterStatus,
    LearningSignalEstimator,
    LearningSignalEstimatorConfig,
    LearningSignalEstimatorState,
    TypedLearningSignals,
)
from alberta_framework.core.learning_value_router import (
    LearningValueRouter,
    LearningValueRouterConfig,
    LearningValueRouterResourceBudget,
    LearningValueRouterResult,
    LearningValueRouterState,
)
from alberta_framework.core.model_replay_rehearsal import (
    ModelReplayRehearsal,
    ModelReplayRehearsalConfig,
    RealModelReplayEvent,
)
from alberta_framework.core.multi_head_learner import MultiHeadMLPState
from alberta_framework.core.oak import (
    OaKAgent,
    OaKConfig,
    OaKKeyboardPolicyProposal,
    OaKState,
)
from alberta_framework.core.option_search_control import (
    OptionSearchControl,
    OptionSearchControlConfig,
    OptionSearchControlDiagnostics,
    OptionSearchControlResourceBudget,
)
from alberta_framework.core.options import (
    DispatchedPrimitiveActionDecision,
    STOMPConfig,
    STOMPUpdateResult,
    SubtaskSpec,
    check_option_terminated,
    compute_pseudo_reward,
    replace_dispatched_primitive_action,
)
from alberta_framework.core.partner_policy_fusion import (
    OptionKeyboardProposal,
    PartnerFusionDecision,
    PartnerFusionFeedbackResult,
    PartnerMessageBatch,
    PartnerPolicyFusion,
    PartnerPolicyFusionConfig,
    PartnerPolicyFusionFeedback,
    PartnerPolicyFusionState,
)
from alberta_framework.core.prototype_feature_lifecycle import (
    PrototypeFeatureConsumerBinding,
    PrototypeFeatureLifecycle,
    PrototypeFeatureLifecycleAdoptionDiagnostics,
    PrototypeFeatureLifecycleConfig,
    PrototypeFeatureLifecycleDiagnostics,
    PrototypeFeatureLifecycleEvent,
    PrototypeFeatureLifecycleHordeResult,
    PrototypeFeatureLifecycleResult,
    PrototypeFeatureLifecycleState,
    PrototypePairGradientPullback,
)
from alberta_framework.core.prototype_feature_memory import (
    PrototypeFeatureMemory,
    PrototypeFeatureMemoryConfig,
    PrototypeFeatureMemoryRebindDiagnostics,
    PrototypeFeatureMemoryRebindResult,
    PrototypeFeatureMemoryResourceBudget,
    PrototypeFeatureMemoryState,
)
from alberta_framework.core.prototype_feature_utility import (
    PrototypeFeatureUtilityAuditor,
    PrototypeFeatureUtilityConfig,
    PrototypeFeatureUtilityDiagnostics,
    PrototypeFeatureUtilityEvent,
    PrototypeFeatureUtilityResourceBudget,
    PrototypeFeatureUtilityState,
)
from alberta_framework.core.prototype_feature_utility_curation import (
    PrototypeFeatureUtilityCurationConfig,
    PrototypeFeatureUtilityCurationDiagnostics,
    PrototypeFeatureUtilityCurationPolicy,
    PrototypeFeatureUtilityCurationResourceBudget,
)
from alberta_framework.core.prototype_routed_linear_world_model import (
    PrototypeRoutedLinearWorldAdoptionDiagnostics,
    PrototypeRoutedLinearWorldConfig,
    PrototypeRoutedLinearWorldModel,
    PrototypeRoutedLinearWorldPrepareResult,
    PrototypeRoutedLinearWorldState,
    PrototypeRoutedLinearWorldTransition,
)
from alberta_framework.core.recurrent_latent_world_model_ensemble import (
    RecurrentLatentDecisionCache,
    RecurrentLatentPredictionAvailability,
    RecurrentLatentStartCache,
    RecurrentLatentTransitionRecord,
    RecurrentLatentWorldModelDiagnostics,
    RecurrentLatentWorldModelEnsemble,
    RecurrentLatentWorldModelEnsembleConfig,
    RecurrentLatentWorldModelEnsembleState,
)
from alberta_framework.core.representation_gradient_mixer import (
    RepresentationGradientMixDiagnostics,
    RepresentationGradientMixerConfig,
    RepresentationGradientMixResult,
    mix_representation_gradients,
)
from alberta_framework.core.rtu_generate_and_test import (
    RTUGenerateAndTest,
    RTUGenerateAndTestProposal,
)
from alberta_framework.core.state_builder import (
    IdentityStateBuilderConfig,
    OnlineGatedStateBuilderConfig,
    RecurrentTraceUnitStateBuilder,
    RecurrentTraceUnitStateBuilderState,
    RecurrentTraceUnitStateBuilderTransitionResult,
    StateBuilder,
    StateBuilderConfig,
    StateBuilderLearningDiagnostics,
    replace_state_builder_learning_proposal_update,
    state_builder_from_config,
)
from alberta_framework.core.stomp_owner_finalization import (
    STOMP_OWNER_STAGE_DYNA,
    STOMP_OWNER_STAGE_FEATURE_ROUTE,
    STOMP_OWNER_STAGE_MEMORY_DISPATCH,
    STOMP_OWNER_STAGE_OPTION_SEARCH,
    STOMP_OWNER_STAGE_PARTNER_DISPATCH,
    STOMPOwnerFinalizationTrace,
    make_stomp_owner_finalization_trace,
    make_stomp_owner_stage_receipt,
    stomp_owner_stage_delta_valid,
    stomp_typed_tree_digest,
)
from alberta_framework.core.types import DemonType, GVFSpec, HordeSpec, LMSState
from alberta_framework.core.world_model import (
    ActionConditionedWorldModel,
    ActionConditionedWorldModelConfig,
    ActionConditionedWorldModelState,
)
from alberta_framework.core.world_model_ensemble import (
    WorldModelEnsemble,
    WorldModelEnsembleConfig,
    WorldModelEnsembleDiagnostics,
    WorldModelEnsembleState,
)

PROTOTYPE_CHECKPOINT_SCHEMA = "alberta.prototype_agent.v13"
PROTOTYPE_FEATURE_UTILITY_CHECKPOINT_SCHEMA = "alberta.prototype_agent.v14"
PROTOTYPE_FEATURE_UTILITY_CURATION_CHECKPOINT_SCHEMA = "alberta.prototype_agent.v15"
PROTOTYPE_FEATURE_MEMORY_CHECKPOINT_SCHEMA = "alberta.prototype_agent.v16"
PROTOTYPE_FEATURE_WORLD_MODEL_CHECKPOINT_SCHEMA = "alberta.prototype_agent.v17"
PROTOTYPE_ATOMIC_FEATURE_WORLD_MEMORY_CHECKPOINT_SCHEMA = "alberta.prototype_agent.v18"
PROTOTYPE_LEARNING_VALUE_ROUTER_CHECKPOINT_SCHEMA = "alberta.prototype_agent.v19"
PROTOTYPE_ATOMIC_FEATURE_WORLD_MEMORY_CONFIG_SCHEMA = (
    "alberta.prototype-atomic-feature-world-memory.config.v1"
)
PROTOTYPE_ATOMIC_FEATURE_WORLD_MEMORY_MECHANISM_STATUS = (
    "l0-mechanism-only-not-assessed"
)
PROTOTYPE_ATOMIC_FEATURE_WORLD_MEMORY_SCIENTIFIC_PROMOTION_ALLOWED = False
PROTOTYPE_FEATURE_WORLD_MODEL_SCHEMA_DIGEST_NBYTES = 32
_PROTOTYPE_CHECKPOINT_SCHEMA_V12 = "alberta.prototype_agent.v12"
_PROTOTYPE_CHECKPOINT_SCHEMA_V11 = "alberta.prototype_agent.v11"
_PROTOTYPE_CHECKPOINT_SCHEMA_V10 = "alberta.prototype_agent.v10"
_PROTOTYPE_CHECKPOINT_SCHEMA_V9 = "alberta.prototype_agent.v9"
_PROTOTYPE_CHECKPOINT_SCHEMA_V8 = "alberta.prototype_agent.v8"
_PROTOTYPE_CHECKPOINT_SCHEMA_V7 = "alberta.prototype_agent.v7"
_PROTOTYPE_CHECKPOINT_SCHEMA_V6 = "alberta.prototype_agent.v6"
_PROTOTYPE_CHECKPOINT_SCHEMA_V5 = "alberta.prototype_agent.v5"
_PROTOTYPE_CHECKPOINT_SCHEMA_V4 = "alberta.prototype_agent.v4"
_PROTOTYPE_CHECKPOINT_SCHEMA_V3 = "alberta.prototype_agent.v3"
_PROTOTYPE_CHECKPOINT_SCHEMA_V2 = "alberta.prototype_agent.v2"
_PROTOTYPE_CHECKPOINT_SCHEMA_V1 = "alberta.prototype_agent.v1"
_DREAM_NEXT_OBSERVATION_STREAM_TAG = 0x44524D4F
_PROTOTYPE_V2_REPLAY_MIGRATION_TAG = 0x50525632
_PROTOTYPE_FEATURE_LIFECYCLE_KEY_TAG = 0x50464C43
_UINT32_MAX = 2**32 - 1
_INT32_MAX = 2**31 - 1

# Two telemetry int32 scalars and two exact two-word clocks.  The v7 outer
# state added only the words, so its persistent delta from the pre-v7 wrapper
# is sixteen bytes.  Nested OaK/STOMP/base-learner clocks report their own
# budgets and are deliberately not double-counted here.
PROTOTYPE_LIFETIME_COUNTER_NBYTES = 24
PROTOTYPE_LIFETIME_COUNTER_DELTA_NBYTES = 16

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
    opt_importance = (
        jnp.max(opt_q_abs.reshape(-1, obs_dim), axis=0)
        if opt_q.shape[0] > 0
        else jnp.zeros((obs_dim,), dtype=jnp.float32)
    )

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


@dataclasses.dataclass(frozen=True, slots=True)
class PrototypeAtomicFeatureWorldMemoryConfig:
    """Opt into one all-consumer pair-bank transaction.

    The authoritative component configurations remain the existing Prototype
    ``oak``, ``prototype_feature_lifecycle``, ``world_model``, Horde, Identity
    builder, and experiential-memory fields.  This enabled-only record owns
    only the routed world's bounded physical-anchor and planning guards, so it
    cannot become a second feature or consumer authority.
    """

    anchor_capacity: int = 8
    planning_enabled: bool = False
    planning_warmup_steps: int = 1
    max_generation_model_error: float = 1_000_000.0
    max_planned_backups: int = _INT32_MAX

    def __post_init__(self) -> None:
        if type(self.anchor_capacity) is not int or not 1 <= self.anchor_capacity <= 4_096:
            raise ValueError("anchor_capacity must be a strict integer in [1, 4096]")
        if type(self.planning_enabled) is not bool:
            raise ValueError("planning_enabled must be an exact bool")
        if (
            type(self.planning_warmup_steps) is not int
            or not 0 <= self.planning_warmup_steps <= _INT32_MAX
        ):
            raise ValueError(
                "planning_warmup_steps must be a strict integer in [0, INT32_MAX]"
            )
        if (
            isinstance(self.max_generation_model_error, bool)
            or not isinstance(self.max_generation_model_error, (int, float))
            or not math.isfinite(float(self.max_generation_model_error))
            or float(self.max_generation_model_error) < 0.0
            or not math.isfinite(float(jnp.float32(self.max_generation_model_error)))
        ):
            raise ValueError(
                "max_generation_model_error must be finite, non-negative, and float32-safe"
            )
        if (
            type(self.max_planned_backups) is not int
            or not 1 <= self.max_planned_backups <= _INT32_MAX
        ):
            raise ValueError(
                "max_planned_backups must be a strict integer in [1, INT32_MAX]"
            )

    def to_config(self) -> dict[str, Any]:
        """Return the exact enabled-only composition record."""

        return {
            "schema": PROTOTYPE_ATOMIC_FEATURE_WORLD_MEMORY_CONFIG_SCHEMA,
            "type": "PrototypeAtomicFeatureWorldMemoryConfig",
            "mechanism_status": (
                PROTOTYPE_ATOMIC_FEATURE_WORLD_MEMORY_MECHANISM_STATUS
            ),
            "scientific_promotion_allowed": False,
            "authority_semantics": (
                "one-lifecycle-router;one-oak-ordered-linear-horde;"
                "one-fixed-physical-routed-world;one-exact-identity-memory"
            ),
            "planning_default": False,
            "anchor_capacity": self.anchor_capacity,
            "planning_enabled": self.planning_enabled,
            "planning_warmup_steps": self.planning_warmup_steps,
            "max_generation_model_error": self.max_generation_model_error,
            "max_planned_backups": self.max_planned_backups,
        }

    @classmethod
    def from_config(
        cls,
        payload: object,
    ) -> PrototypeAtomicFeatureWorldMemoryConfig:
        """Reconstruct only the exact current composition schema."""

        if type(payload) is not dict:
            raise TypeError("atomic feature/world/memory config must be an exact dict")
        raw = cast(dict[str, object], payload)
        expected = {
            "schema",
            "type",
            "mechanism_status",
            "scientific_promotion_allowed",
            "authority_semantics",
            "planning_default",
            "anchor_capacity",
            "planning_enabled",
            "planning_warmup_steps",
            "max_generation_model_error",
            "max_planned_backups",
        }
        if set(raw) != expected:
            raise ValueError("atomic feature/world/memory config fields differ from v1")
        fixed = {
            "schema": PROTOTYPE_ATOMIC_FEATURE_WORLD_MEMORY_CONFIG_SCHEMA,
            "type": "PrototypeAtomicFeatureWorldMemoryConfig",
            "mechanism_status": (
                PROTOTYPE_ATOMIC_FEATURE_WORLD_MEMORY_MECHANISM_STATUS
            ),
            "scientific_promotion_allowed": False,
            "authority_semantics": (
                "one-lifecycle-router;one-oak-ordered-linear-horde;"
                "one-fixed-physical-routed-world;one-exact-identity-memory"
            ),
            "planning_default": False,
        }
        if any(raw[name] != value for name, value in fixed.items()):
            raise ValueError("atomic feature/world/memory fixed semantics differ")
        return cls(
            anchor_capacity=cast(int, raw["anchor_capacity"]),
            planning_enabled=cast(bool, raw["planning_enabled"]),
            planning_warmup_steps=cast(int, raw["planning_warmup_steps"]),
            max_generation_model_error=cast(
                float,
                raw["max_generation_model_error"],
            ),
            max_planned_backups=cast(int, raw["max_planned_backups"]),
        )


@dataclasses.dataclass(frozen=True)
class PrototypeAgentConfig:
    """Configuration for the experimental PrototypeAgent composition.

    All components are designed for *continuing* average-reward settings —
    no episode resets, no offline training phases.

    Args:
        oak: OaK agent configuration (steps 5/6/10/11).  Required.
        option_search_control: Optional stateless support-aware Bellman-residual
            search over learned option models. It applies a fixed number of
            planner-only base-value backups at the next decision
            representation without refreshing the dispatch OaK already
            cached. Value changes become behavior-eligible only at the next
            extended-action selection boundary. The legacy STOMP
            option-planning budget must be zero so the two planners cannot
            silently double the work budget.
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
        recurrent_latent_world_model_ensemble: Mutually exclusive recurrent
            GRU ensemble. Its exact dispatched representation/action prediction
            is cached at decision time and consumed once by the authoritative
            transition. Only its real predict-before-update NLL gradient may
            reach the representation mixer; raw uncertainty is uncalibrated.
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
        partner_policy_fusion: Optional bounded contextual partner-message
            policy fusion. Its action count must match OaK's primitive action
            count and its context width must match OaK's representation width.
            The initial :meth:`start` decision remains OaK-only; the first
            fusion opportunity is the next decision after a real transition.
        experiential_memory: Optional fixed-capacity experiential memory used
            only through its conservative categorical policy boundary. Stored
            observations and keys must match OaK's representation width,
            stored action mass must match the primitive action count, and the
            outcome stores the bootstrap representation plus scalar reward.
            With the pair-feature lifecycle, the base builder must be exact
            identity and every live row is rebound atomically from its stable
            base prefix when the descriptor bank changes.
        state_builder: Canonical causal representation builder. Its output
            width must equal the feature lifecycle's base width when that lane
            is enabled, and ``oak.observation_dim`` otherwise. Mutually
            exclusive with the deprecated fixed-weight ``gru_perception`` path.
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
        gradient_joy: Historical configuration name for the optional
            multi-objective candidate-update safety audit applied to each
            proposed builder update. Missing or incomplete probe evidence
            vetoes only the representation update; control and model learning
            continue.
        learning_value_router: Optional owner-bound causal router for the
            candidate audit's eight typed learning-value channels. It requires
            the candidate-update audit and one causal ensemble signal producer.
            Raw routed evidence, never normalized values or an aggregate score,
            enters the audit. ``None`` preserves the historical config and
            state PyTree exactly.
        prototype_feature_lifecycle: Optional fixed-budget pair-feature bank
            between a supported base state builder and linear OaK.  It always
            owns one scalar, owner-bound control-TD discovery target.  Its
            managed-Horde mode appends ordered linear-Horde TD targets, gives
            control and the aggregate demon group equal proxy-utility mass,
            and atomically routes every OaK/Horde feature axis when a pair
            descriptor changes. Experiential memory is the first additional
            supported consumer through an exact bank-bound re-encoding
            adapter; other unversioned representation consumers are rejected.
        experiential_memory_advantage_gate: Optional stateless conservative
            dispatch-authority gate. It requires action-conditioned local
            reward evidence for both the memory proposal and OaK's
            counterfactual action before memory may replace that action.
            ``None`` preserves the historical experiential-memory policy and
            serialized configuration exactly.
        prototype_feature_utility: Optional diagnostic-only auditor for the
            shared linear OaK/Horde bank. It measures frozen old-consumer
            one-step deletion loss and matched shadow-candidate insertion
            gain, but has no curation authority. Its dimensions and lifetime
            cap must match ``prototype_feature_lifecycle`` exactly.
        prototype_feature_utility_curation: Optional ranking-only adapter that
            supplies mature deletion and insertion ranks to the lifecycle's
            existing within-cohort priority override. The auditor remains
            diagnostic-only; ages, cadence, confirmation, proxy promotion
            gates, and safe routing retain go/no-go authority.
    """

    oak: OaKConfig = dataclasses.field(default_factory=_default_oak_config)
    world_model: ActionConditionedWorldModelConfig | None = None
    world_model_ensemble: WorldModelEnsembleConfig | None = None
    model_replay_rehearsal: ModelReplayRehearsalConfig | None = None
    recurrent_latent_world_model_ensemble: (
        RecurrentLatentWorldModelEnsembleConfig | None
    ) = None
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
    partner_policy_fusion: PartnerPolicyFusionConfig | None = None
    experiential_memory: ExperientialMemoryConfig | None = None
    experiential_memory_advantage_gate: (
        ExperientialMemoryAdvantageGateConfig | None
    ) = None
    gru_perception: GRUPerceptionConfig | None = None
    state_builder: StateBuilderConfig | None = None
    learn_state_builder_from_world_model: bool = False
    representation_gradient_mixer: RepresentationGradientMixerConfig | None = None
    gradient_joy: CandidateUpdateAuditConfig | None = None
    learning_value_router: LearningValueRouterConfig | None = None
    auto_curate_every: int = 0
    option_search_control: OptionSearchControlConfig | None = None
    prototype_feature_lifecycle: PrototypeFeatureLifecycleConfig | None = None
    prototype_feature_utility: PrototypeFeatureUtilityConfig | None = None
    prototype_feature_utility_curation: (
        PrototypeFeatureUtilityCurationConfig | None
    ) = None
    prototype_atomic_feature_world_memory: (
        PrototypeAtomicFeatureWorldMemoryConfig | None
    ) = None

    def __post_init__(self) -> None:
        feature_lifecycle = self.prototype_feature_lifecycle
        atomic_feature_world_memory = self.prototype_atomic_feature_world_memory
        if (
            atomic_feature_world_memory is not None
            and type(atomic_feature_world_memory)
            is not PrototypeAtomicFeatureWorldMemoryConfig
        ):
            raise ValueError(
                "prototype_atomic_feature_world_memory must be an exact "
                "PrototypeAtomicFeatureWorldMemoryConfig"
            )
        if feature_lifecycle is not None and not isinstance(
            feature_lifecycle,
            PrototypeFeatureLifecycleConfig,
        ):
            raise ValueError(
                "prototype_feature_lifecycle must be a "
                "PrototypeFeatureLifecycleConfig"
            )
        if (
            feature_lifecycle is not None
            and self.experiential_memory is not None
            and type(feature_lifecycle) is not PrototypeFeatureLifecycleConfig
        ):
            raise ValueError(
                "prototype feature-memory composition requires an exact "
                "PrototypeFeatureLifecycleConfig"
            )
        if (
            feature_lifecycle is not None
            and self.experiential_memory is not None
            and type(self.experiential_memory) is not ExperientialMemoryConfig
        ):
            raise ValueError(
                "prototype feature-memory composition requires an exact "
                "ExperientialMemoryConfig"
            )
        feature_utility = self.prototype_feature_utility
        if feature_utility is not None and type(feature_utility) is not (
            PrototypeFeatureUtilityConfig
        ):
            raise ValueError(
                "prototype_feature_utility must be a "
                "PrototypeFeatureUtilityConfig"
            )
        if feature_utility is not None:
            if (
                feature_lifecycle is None
                or feature_lifecycle.managed_horde_demons <= 0
            ):
                raise ValueError(
                    "prototype_feature_utility requires the shared managed-Horde "
                    "prototype_feature_lifecycle"
                )
            expected_dimensions = (
                feature_lifecycle.base_feature_dim,
                feature_lifecycle.active_pair_slots,
                feature_lifecycle.candidate_pair_slots,
                feature_lifecycle.managed_horde_demons,
                feature_lifecycle.max_observations,
            )
            actual_dimensions = (
                feature_utility.base_feature_dim,
                feature_utility.active_pair_slots,
                feature_utility.candidate_pair_slots,
                feature_utility.managed_horde_demons,
                feature_utility.max_observations,
            )
            if actual_dimensions != expected_dimensions:
                raise ValueError(
                    "prototype_feature_utility dimensions and max_observations "
                    "must match prototype_feature_lifecycle"
                )
        feature_utility_curation = self.prototype_feature_utility_curation
        if feature_utility_curation is not None:
            if type(feature_utility_curation) is not (
                PrototypeFeatureUtilityCurationConfig
            ):
                raise ValueError(
                    "prototype_feature_utility_curation must be a "
                    "PrototypeFeatureUtilityCurationConfig"
                )
            if feature_utility is None:
                raise ValueError(
                    "prototype_feature_utility_curation requires "
                    "prototype_feature_utility"
                )
            if feature_utility.candidate_pair_slots <= 0:
                raise ValueError(
                    "prototype_feature_utility_curation requires at least one "
                    "candidate pair slot"
                )
            PrototypeFeatureUtilityCurationPolicy(
                feature_utility,
                feature_utility_curation,
            )
        if self.option_search_control is not None:
            if not isinstance(
                self.option_search_control,
                OptionSearchControlConfig,
            ):
                raise ValueError(
                    "option_search_control must be an OptionSearchControlConfig"
                )
            if self.oak.stomp.option_planning_backups_per_step != 0:
                raise ValueError(
                    "option_search_control requires "
                    "oak.stomp.option_planning_backups_per_step == 0"
                )
        if not 0 < self.buffer_capacity <= _INT32_MAX:
            raise ValueError("buffer_capacity must be in [1, INT32_MAX]")
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
        if not 0 <= self.auto_curate_every <= _INT32_MAX:
            raise ValueError(
                "auto_curate_every must be in [0, INT32_MAX] for exact cadence"
            )
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
                self.recurrent_latent_world_model_ensemble,
            )
        )
        if configured_model_lanes > 1:
            raise ValueError(
                "world_model, world_model_ensemble, model_replay_rehearsal, and "
                "recurrent_latent_world_model_ensemble are mutually exclusive"
            )
        if self.world_model is None and self.n_dreams_per_step > 0:
            raise ValueError(
                "n_dreams_per_step > 0 requires the legacy world_model to be configured"
            )
        if self.world_model is not None:
            expected_world_observation_dim = (
                feature_lifecycle.base_feature_dim
                if feature_lifecycle is not None
                else self.oak.observation_dim
            )
            if self.world_model.observation_dim != expected_world_observation_dim:
                if feature_lifecycle is not None:
                    raise ValueError(
                        "world_model.observation_dim must match the stable "
                        "base_feature_dim from prototype_feature_lifecycle, got "
                        f"{self.world_model.observation_dim} and "
                        f"{feature_lifecycle.base_feature_dim}"
                    )
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
        if self.recurrent_latent_world_model_ensemble is not None:
            recurrent_model = self.recurrent_latent_world_model_ensemble
            if recurrent_model.observation_dim != self.oak.observation_dim:
                raise ValueError(
                    "recurrent_latent_world_model_ensemble.observation_dim must "
                    "match oak.observation_dim, got "
                    f"{recurrent_model.observation_dim} and {self.oak.observation_dim}"
                )
            if recurrent_model.n_actions != self.oak.n_primitive_actions:
                raise ValueError(
                    "recurrent_latent_world_model_ensemble.n_actions must match "
                    "oak.n_primitive_actions, got "
                    f"{recurrent_model.n_actions} and {self.oak.n_primitive_actions}"
                )
            if self.n_dreams_per_step != 0:
                raise ValueError(
                    "dreaming is disabled for recurrent_latent_world_model_ensemble; "
                    "its raw uncertainty has no calibrated rollout-validity gate"
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
            expected_builder_width = (
                feature_lifecycle.base_feature_dim
                if feature_lifecycle is not None
                else self.oak.observation_dim
            )
            if builder.feature_dim() != expected_builder_width:
                raise ValueError(
                    "state_builder feature_dim must match the configured base "
                    f"representation width ({expected_builder_width}), got "
                    f"{builder.feature_dim()}"
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
            if (
                self.world_model_ensemble is None
                and self.model_replay_rehearsal is None
                and self.recurrent_latent_world_model_ensemble is None
            ):
                raise ValueError(
                    "learn_state_builder_from_world_model requires "
                    "world_model_ensemble, model_replay_rehearsal, or "
                    "recurrent_latent_world_model_ensemble"
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
                and self.recurrent_latent_world_model_ensemble is None
            ):
                raise ValueError(
                    "an active grounded-world mixer source requires "
                    "world_model_ensemble, model_replay_rehearsal, or "
                    "recurrent_latent_world_model_ensemble"
                )
        if self.gradient_joy is not None:
            if not state_builder_learning_enabled:
                raise ValueError(
                    "gradient_joy requires online representation learning"
                )
            if (
                self.world_model_ensemble is None
                and self.model_replay_rehearsal is None
                and self.recurrent_latent_world_model_ensemble is None
            ):
                raise ValueError(
                    "gradient_joy requires causal ensemble learning signals"
                )
            if self.gradient_joy.candidate_semantics != "update":
                raise ValueError(
                    "PrototypeAgent gradient_joy must use candidate_semantics='update'"
                )
        learning_value_router = self.learning_value_router
        if (
            learning_value_router is not None
            and type(learning_value_router) is not LearningValueRouterConfig
        ):
            raise ValueError(
                "learning_value_router must be an exact LearningValueRouterConfig"
            )
        if learning_value_router is not None:
            if self.gradient_joy is None:
                raise ValueError(
                    "learning_value_router requires the candidate-update audit"
                )
            if (
                self.world_model_ensemble is None
                and self.model_replay_rehearsal is None
                and self.recurrent_latent_world_model_ensemble is None
            ):
                raise ValueError(
                    "learning_value_router requires causal ensemble learning signals"
                )
        if self.ia is not None:
            ia_obs = self.ia.cortex.observation_dim
            oak_obs = self.oak.observation_dim
            if ia_obs != oak_obs:
                raise ValueError(
                    f"ia.cortex.observation_dim ({ia_obs}) must match "
                    f"oak.observation_dim ({oak_obs})"
                )
        if self.partner_policy_fusion is not None:
            fusion = self.partner_policy_fusion
            if not isinstance(fusion, PartnerPolicyFusionConfig):
                raise ValueError(
                    "partner_policy_fusion must be a PartnerPolicyFusionConfig"
                )
            if fusion.n_actions != self.oak.n_primitive_actions:
                raise ValueError(
                    "partner_policy_fusion.n_actions must match "
                    f"oak.n_primitive_actions ({self.oak.n_primitive_actions}), "
                    f"got {fusion.n_actions}"
                )
            if fusion.context_dim != self.oak.observation_dim:
                raise ValueError(
                    "partner_policy_fusion.context_dim must match "
                    f"oak.observation_dim ({self.oak.observation_dim}), "
                    f"got {fusion.context_dim}"
                )
        if self.experiential_memory is not None:
            memory = self.experiential_memory
            if not isinstance(memory, ExperientialMemoryConfig):
                raise ValueError(
                    "experiential_memory must be an ExperientialMemoryConfig"
                )
            # ExperientialMemoryConfig predates dataclass-level validation;
            # constructing the bounded substrate enforces its complete static
            # contract while retaining the caller's exact config object.
            ExperientialMemory(memory)
            representation_dim = self.oak.observation_dim
            if memory.observation_dim != representation_dim:
                raise ValueError(
                    "experiential_memory.observation_dim must match "
                    f"oak.observation_dim ({representation_dim}), got "
                    f"{memory.observation_dim}"
                )
            if memory.key_dim != representation_dim:
                raise ValueError(
                    "experiential_memory.key_dim must match "
                    f"oak.observation_dim ({representation_dim}), got "
                    f"{memory.key_dim}"
                )
            if memory.action_dim != self.oak.n_primitive_actions:
                raise ValueError(
                    "experiential_memory.action_dim must match "
                    "oak.n_primitive_actions "
                    f"({self.oak.n_primitive_actions}), got {memory.action_dim}"
                )
            expected_outcome_dim = representation_dim + 1
            if memory.outcome_dim != expected_outcome_dim:
                raise ValueError(
                    "experiential_memory.outcome_dim must equal "
                    "oak.observation_dim + 1 "
                    f"({expected_outcome_dim}), got {memory.outcome_dim}"
                )
        advantage_gate = self.experiential_memory_advantage_gate
        if advantage_gate is not None:
            if type(advantage_gate) is not ExperientialMemoryAdvantageGateConfig:
                raise ValueError(
                    "experiential_memory_advantage_gate must be an exact "
                    "ExperientialMemoryAdvantageGateConfig"
                )
            if self.experiential_memory is None:
                raise ValueError(
                    "experiential_memory_advantage_gate requires experiential_memory"
                )
            if advantage_gate.min_action_support > self.experiential_memory.top_k:
                raise ValueError(
                    "experiential_memory_advantage_gate.min_action_support must "
                    "not exceed experiential_memory.top_k"
                )

        if feature_lifecycle is not None:
            managed_horde_demons = feature_lifecycle.managed_horde_demons
            if managed_horde_demons == 0 and feature_lifecycle.n_tasks != 1:
                raise ValueError(
                    "prototype_feature_lifecycle.n_tasks must equal 1"
                )
            if managed_horde_demons > 0:
                horde_spec = self.horde_spec
                if type(horde_spec) is not HordeSpec:
                    raise ValueError(
                        "managed prototype feature Horde requires an exact "
                        "HordeSpec"
                    )
                demons = horde_spec.demons
                if (
                    type(demons) is not tuple
                    or len(demons) != managed_horde_demons
                    or not all(type(demon) is GVFSpec for demon in demons)
                ):
                    raise ValueError(
                        "managed prototype feature Horde demon count and "
                        "ordered exact specifications must match"
                    )
                if not all(
                    type(demon.name) is str
                    and type(demon.demon_type) is DemonType
                    and type(demon.gamma) is float
                    and math.isfinite(demon.gamma)
                    and 0.0 <= demon.gamma <= 1.0
                    and type(demon.lamda) is float
                    and math.isfinite(demon.lamda)
                    and 0.0 <= demon.lamda <= 1.0
                    and type(demon.cumulant_index) is int
                    and demon.cumulant_index >= -1
                    and type(demon.terminal_reward) is float
                    and math.isfinite(demon.terminal_reward)
                    for demon in demons
                ):
                    raise ValueError(
                        "managed prototype feature Horde questions must have "
                        "strict finite canonical metadata"
                    )
                expected_horde_tasks = 1 + managed_horde_demons
                if feature_lifecycle.n_tasks != expected_horde_tasks:
                    raise ValueError(
                        "managed prototype feature lifecycle n_tasks must equal "
                        "1 + managed_horde_demons"
                    )
                expected_gammas = jnp.asarray(
                    [demon.gamma for demon in demons],
                    dtype=jnp.float32,
                )
                expected_lamdas = jnp.asarray(
                    [demon.lamda for demon in demons],
                    dtype=jnp.float32,
                )
                gamma_contract_valid = (
                    hasattr(horde_spec.gammas, "shape")
                    and hasattr(horde_spec.gammas, "dtype")
                    and horde_spec.gammas.shape == (managed_horde_demons,)
                    and horde_spec.gammas.dtype == jnp.float32
                )
                lambda_contract_valid = (
                    hasattr(horde_spec.lamdas, "shape")
                    and hasattr(horde_spec.lamdas, "dtype")
                    and horde_spec.lamdas.shape == (managed_horde_demons,)
                    and horde_spec.lamdas.dtype == jnp.float32
                )
                if (
                    not gamma_contract_valid
                    or not lambda_contract_valid
                    or not bool(jnp.array_equal(horde_spec.gammas, expected_gammas))
                    or not bool(jnp.array_equal(horde_spec.lamdas, expected_lamdas))
                ):
                    raise ValueError(
                        "managed prototype feature Horde cached gamma/lambda "
                        "arrays must exactly match the ordered specifications"
                    )
                if type(self.horde_hidden_sizes) is not tuple or self.horde_hidden_sizes != ():
                    raise ValueError(
                        "managed prototype feature Horde requires "
                        "horde_hidden_sizes == ()"
                    )
                if (
                    type(self.horde_step_size) is not float
                    or not math.isfinite(self.horde_step_size)
                    or self.horde_step_size <= 0.0
                    or not math.isfinite(float(jnp.float32(self.horde_step_size)))
                    or float(jnp.float32(self.horde_step_size)) <= 0.0
                ):
                    raise ValueError(
                        "managed prototype feature Horde step size must be a "
                        "positive float32-representable strict float"
                    )
            elif self.horde_spec is not None:
                raise ValueError(
                    "prototype_feature_lifecycle requires "
                    "managed_horde_demons for a Horde consumer"
                )
            if feature_lifecycle.n_options != self.oak.n_options:
                raise ValueError(
                    "prototype_feature_lifecycle.n_options must match "
                    "oak.n_options"
                )
            if (
                feature_lifecycle.n_primitive_actions
                != self.oak.n_primitive_actions
            ):
                raise ValueError(
                    "prototype_feature_lifecycle.n_primitive_actions must "
                    "match oak.n_primitive_actions"
                )
            if feature_lifecycle.total_feature_dim != self.oak.observation_dim:
                raise ValueError(
                    "prototype_feature_lifecycle.total_feature_dim must match "
                    "oak.observation_dim"
                )
            if self.oak.stomp.base_hidden_sizes != ():
                raise ValueError(
                    "prototype_feature_lifecycle requires linear OaK with "
                    "oak.stomp.base_hidden_sizes == ()"
                )
            actual_subtask_indices = tuple(
                spec.feature_index for spec in self.oak.stomp.subtask_specs
            )
            if (
                feature_lifecycle.option_subtask_feature_indices
                != actual_subtask_indices
            ):
                raise ValueError(
                    "prototype_feature_lifecycle option subtask indices must "
                    "exactly match oak.stomp.subtask_specs"
                )
            PrototypeFeatureLifecycle(
                feature_lifecycle
            ).require_compatible_oak_config(self.oak)
            if type(self.state_builder) not in {
                IdentityStateBuilderConfig,
                OnlineGatedStateBuilderConfig,
            }:
                if self.world_model is not None:
                    raise ValueError(
                        "prototype feature/world-model composition requires an "
                        "exact Identity state_builder"
                    )
                raise ValueError(
                    "prototype_feature_lifecycle requires an Identity or "
                    "OnlineGated state_builder"
                )
            if self.gru_perception is not None:
                raise ValueError(
                    "prototype_feature_lifecycle is incompatible with "
                    "gru_perception"
                )
            if (
                self.experiential_memory is not None
                and type(self.state_builder) is not IdentityStateBuilderConfig
            ):
                raise ValueError(
                    "prototype_feature_lifecycle with experiential_memory "
                    "requires an exact Identity state_builder so stored base "
                    "prefixes remain reconstructable across feature-bank changes"
                )
            if self.world_model is not None:
                if type(feature_lifecycle) is not PrototypeFeatureLifecycleConfig:
                    raise ValueError(
                        "prototype feature/world-model composition requires an "
                        "exact PrototypeFeatureLifecycleConfig"
                    )
                if type(self.world_model) is not ActionConditionedWorldModelConfig:
                    raise ValueError(
                        "prototype feature/world-model composition requires an "
                        "exact ActionConditionedWorldModelConfig"
                    )
                if type(self.state_builder) is not IdentityStateBuilderConfig:
                    raise ValueError(
                        "prototype feature/world-model composition requires an "
                        "exact Identity state_builder"
                    )
                if self.dreaming is not None or self.n_dreams_per_step != 0:
                    raise ValueError(
                        "dreaming is disabled for the stable-base prototype "
                        "feature/world-model composition"
                    )
            partner_fusion_with_replaceable_feature_axis = (
                self.partner_policy_fusion is not None
                and (
                    feature_lifecycle.replacement_interval != 0
                    or self.auto_curate_every != 0
                    or feature_utility_curation is not None
                    or atomic_feature_world_memory is not None
                )
            )
            unsupported_consumers = (
                self.world_model_ensemble is not None
                or self.model_replay_rehearsal is not None
                or self.recurrent_latent_world_model_ensemble is not None
                or (self.world_model is None and self.dreaming is not None)
                or self.n_dreams_per_step != 0
                or self.ia is not None
                or partner_fusion_with_replaceable_feature_axis
            )
            if unsupported_consumers:
                raise ValueError(
                    "prototype_feature_lifecycle rejects dreaming, "
                    "ensemble/replay/recurrent world-model, IA, and partner "
                    "fusion over a replaceable feature axis"
                )
            if self.learn_state_builder_from_world_model:
                raise ValueError(
                    "prototype_feature_lifecycle rejects world-model builder learning"
                )
            if self.gradient_joy is not None:
                raise ValueError(
                    "prototype_feature_lifecycle rejects gradient_joy until "
                    "its evidence is representation-versioned"
                )
            if (
                self.representation_gradient_mixer is not None
                and self.representation_gradient_mixer.mode
                not in {"behavior_only", "discard"}
            ):
                raise ValueError(
                    "prototype_feature_lifecycle supports only behavior_only "
                    "or discard representation-gradient mixing"
                )
            if self.auto_curate_every != 0:
                raise ValueError(
                    "prototype_feature_lifecycle requires auto_curate_every == 0"
                )
        if atomic_feature_world_memory is not None:
            if (
                feature_lifecycle is None
                or feature_lifecycle.managed_horde_demons <= 0
            ):
                raise ValueError(
                    "atomic feature/world/memory requires the ordered managed "
                    "linear-Horde feature lifecycle"
                )
            if type(self.world_model) is not ActionConditionedWorldModelConfig:
                raise ValueError(
                    "atomic feature/world/memory requires an exact linear "
                    "ActionConditionedWorldModelConfig"
                )
            if type(self.experiential_memory) is not ExperientialMemoryConfig:
                raise ValueError(
                    "atomic feature/world/memory requires exact experiential memory"
                )
            if type(self.state_builder) is not IdentityStateBuilderConfig:
                raise ValueError(
                    "atomic feature/world/memory requires an exact Identity state_builder"
                )
            if self.prototype_feature_utility is not None or (
                self.prototype_feature_utility_curation is not None
            ):
                raise ValueError(
                    "atomic feature/world/memory excludes utility-auditor curation"
                )
            if self.experiential_memory_advantage_gate is not None:
                raise ValueError(
                    "atomic feature/world/memory excludes the memory advantage gate"
                )
            if self.option_search_control is not None:
                raise ValueError(
                    "atomic feature/world/memory excludes option search control"
                )
            if self.representation_gradient_mixer is not None:
                raise ValueError(
                    "atomic feature/world/memory excludes representation-gradient mixing"
                )
            PrototypeRoutedLinearWorldConfig(
                feature_lifecycle=feature_lifecycle,
                world_model=self.world_model,
                oak=self.oak,
                anchor_capacity=atomic_feature_world_memory.anchor_capacity,
                planning_enabled=atomic_feature_world_memory.planning_enabled,
                planning_warmup_steps=(
                    atomic_feature_world_memory.planning_warmup_steps
                ),
                max_generation_model_error=(
                    atomic_feature_world_memory.max_generation_model_error
                ),
                max_planned_backups=(
                    atomic_feature_world_memory.max_planned_backups
                ),
                carry_survivors=feature_lifecycle.carry_survivors,
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
        if self.option_search_control is not None:
            payload["option_search_control"] = (
                self.option_search_control.to_config()
            )
        if self.prototype_feature_lifecycle is not None:
            payload["prototype_feature_lifecycle"] = (
                self.prototype_feature_lifecycle.to_config()
            )
        if self.prototype_feature_utility is not None:
            payload["prototype_feature_utility"] = (
                self.prototype_feature_utility.to_config()
            )
        if self.prototype_feature_utility_curation is not None:
            payload["prototype_feature_utility_curation"] = (
                self.prototype_feature_utility_curation.to_config()
            )
        if self.world_model is not None:
            payload["world_model"] = self.world_model.to_config()
        if self.world_model_ensemble is not None:
            payload["world_model_ensemble"] = self.world_model_ensemble.to_config()
        if self.model_replay_rehearsal is not None:
            payload["model_replay_rehearsal"] = (
                self.model_replay_rehearsal.to_config()
            )
        if self.recurrent_latent_world_model_ensemble is not None:
            payload["recurrent_latent_world_model_ensemble"] = (
                self.recurrent_latent_world_model_ensemble.to_config()
            )
        if self.dreaming is not None:
            payload["dreaming"] = self.dreaming.to_config()
        if self.horde_spec is not None:
            payload["horde_spec"] = self.horde_spec.to_config()
        if self.ia is not None:
            payload["ia"] = self.ia.to_config()
        if self.partner_policy_fusion is not None:
            payload["partner_policy_fusion"] = (
                self.partner_policy_fusion.to_config()
            )
        if self.experiential_memory is not None:
            payload["experiential_memory"] = self.experiential_memory.to_config()
        if self.experiential_memory_advantage_gate is not None:
            payload["experiential_memory_advantage_gate"] = (
                self.experiential_memory_advantage_gate.to_config()
            )
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
        if self.learning_value_router is not None:
            payload["learning_value_router"] = (
                self.learning_value_router.to_config()
            )
        if self.prototype_atomic_feature_world_memory is not None:
            payload["prototype_atomic_feature_world_memory"] = (
                self.prototype_atomic_feature_world_memory.to_config()
            )
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

        option_search_raw = data.pop("option_search_control", None)
        if option_search_raw is not None and not isinstance(
            option_search_raw,
            dict,
        ):
            raise ValueError("option_search_control must be a configuration object")
        option_search_control = (
            OptionSearchControlConfig.from_config(option_search_raw)
            if option_search_raw is not None
            else None
        )

        feature_lifecycle_raw = data.pop("prototype_feature_lifecycle", None)
        if feature_lifecycle_raw is not None and not isinstance(
            feature_lifecycle_raw,
            dict,
        ):
            raise ValueError(
                "prototype_feature_lifecycle must be a configuration object"
            )
        prototype_feature_lifecycle = (
            PrototypeFeatureLifecycleConfig.from_config(feature_lifecycle_raw)
            if feature_lifecycle_raw is not None
            else None
        )

        feature_utility_raw = data.pop("prototype_feature_utility", None)
        if feature_utility_raw is not None and not isinstance(
            feature_utility_raw,
            dict,
        ):
            raise ValueError(
                "prototype_feature_utility must be a configuration object"
            )
        prototype_feature_utility = (
            PrototypeFeatureUtilityConfig.from_config(feature_utility_raw)
            if feature_utility_raw is not None
            else None
        )

        feature_utility_curation_raw = data.pop(
            "prototype_feature_utility_curation",
            None,
        )
        if feature_utility_curation_raw is not None and not isinstance(
            feature_utility_curation_raw,
            dict,
        ):
            raise ValueError(
                "prototype_feature_utility_curation must be a configuration "
                "object"
            )
        prototype_feature_utility_curation = (
            PrototypeFeatureUtilityCurationConfig.from_config(
                feature_utility_curation_raw
            )
            if feature_utility_curation_raw is not None
            else None
        )

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
        recurrent_latent_raw = data.pop(
            "recurrent_latent_world_model_ensemble",
            None,
        )
        recurrent_latent_world_model_ensemble = (
            RecurrentLatentWorldModelEnsembleConfig.from_config(
                recurrent_latent_raw
            )
            if recurrent_latent_raw is not None
            else None
        )
        dream_raw = data.pop("dreaming", None)
        dreaming = DreamingConfig.from_config(dream_raw) if dream_raw is not None else None
        horde_raw = data.pop("horde_spec", None)
        horde_spec = _HordeSpec.from_config(horde_raw) if horde_raw is not None else None
        ia_raw = data.pop("ia", None)
        ia = IAConfig.from_config(ia_raw) if ia_raw is not None else None
        partner_fusion_raw = data.pop("partner_policy_fusion", None)
        if partner_fusion_raw is not None and not isinstance(
            partner_fusion_raw,
            dict,
        ):
            raise ValueError("partner_policy_fusion must be a configuration object")
        partner_policy_fusion = (
            PartnerPolicyFusionConfig.from_config(partner_fusion_raw)
            if partner_fusion_raw is not None
            else None
        )
        experiential_memory_raw = data.pop("experiential_memory", None)
        if experiential_memory_raw is not None and not isinstance(
            experiential_memory_raw,
            dict,
        ):
            raise ValueError("experiential_memory must be a configuration object")
        experiential_memory = (
            ExperientialMemoryConfig.from_config(experiential_memory_raw)
            if experiential_memory_raw is not None
            else None
        )
        advantage_gate_raw = data.pop(
            "experiential_memory_advantage_gate",
            None,
        )
        if advantage_gate_raw is not None and not isinstance(
            advantage_gate_raw,
            dict,
        ):
            raise ValueError(
                "experiential_memory_advantage_gate must be a configuration object"
            )
        experiential_memory_advantage_gate = (
            ExperientialMemoryAdvantageGateConfig.from_config(advantage_gate_raw)
            if advantage_gate_raw is not None
            else None
        )
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
            CandidateUpdateAuditConfig.from_config(gradient_joy_raw)
            if gradient_joy_raw is not None
            else None
        )
        learning_value_router_raw = data.pop("learning_value_router", None)
        if learning_value_router_raw is not None and not isinstance(
            learning_value_router_raw,
            dict,
        ):
            raise ValueError(
                "learning_value_router must be a configuration object"
            )
        learning_value_router = (
            LearningValueRouterConfig.from_config(learning_value_router_raw)
            if learning_value_router_raw is not None
            else None
        )

        atomic_feature_world_memory_raw = data.pop(
            "prototype_atomic_feature_world_memory",
            None,
        )
        prototype_atomic_feature_world_memory = (
            PrototypeAtomicFeatureWorldMemoryConfig.from_config(
                atomic_feature_world_memory_raw
            )
            if atomic_feature_world_memory_raw is not None
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
            option_search_control=option_search_control,
            world_model=world_model,
            world_model_ensemble=world_model_ensemble,
            model_replay_rehearsal=model_replay_rehearsal,
            recurrent_latent_world_model_ensemble=(
                recurrent_latent_world_model_ensemble
            ),
            dreaming=dreaming,
            buffer_capacity=buffer_capacity,
            n_dreams_per_step=n_dreams_per_step,
            dream_next_observation_mode=dream_next_observation_mode,
            horde_spec=horde_spec,
            horde_hidden_sizes=hidden,
            horde_step_size=horde_step_size,
            ia=ia,
            partner_policy_fusion=partner_policy_fusion,
            experiential_memory=experiential_memory,
            experiential_memory_advantage_gate=(
                experiential_memory_advantage_gate
            ),
            gru_perception=gru_perception,
            state_builder=state_builder,
            learn_state_builder_from_world_model=(
                learn_state_builder_from_world_model
            ),
            representation_gradient_mixer=representation_gradient_mixer,
            gradient_joy=gradient_joy,
            learning_value_router=learning_value_router,
            auto_curate_every=auto_curate_every,
            prototype_feature_lifecycle=prototype_feature_lifecycle,
            prototype_feature_utility=prototype_feature_utility,
            prototype_feature_utility_curation=(
                prototype_feature_utility_curation
            ),
            prototype_atomic_feature_world_memory=(
                prototype_atomic_feature_world_memory
            ),
        )


def _prototype_feature_horde_schema_digest(
    config: PrototypeAgentConfig,
) -> Array:
    """Bind one shared feature bank to its ordered consumer semantics."""

    feature_config = config.prototype_feature_lifecycle
    horde_spec = config.horde_spec
    if (
        feature_config is None
        or feature_config.managed_horde_demons <= 0
        or type(horde_spec) is not HordeSpec
    ):
        raise ValueError("shared feature/Horde schema requires both components")
    payload = {
        "schema": "alberta.prototype_feature_oak_horde.v1",
        "channel_mapping": {
            "control_target": 0,
            "ordered_horde_targets_start": 1,
        },
        "feature_lifecycle": feature_config.to_config(),
        "oak": config.oak.to_config(),
        "horde_spec": horde_spec.to_config(),
        "horde_hidden_sizes": list(config.horde_hidden_sizes),
        "horde_step_size": config.horde_step_size,
    }
    encoded = json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return jnp.asarray(
        tuple(hashlib.sha256(encoded).digest()),
        dtype=jnp.uint8,
    )


def _prototype_feature_world_model_schema_digest(
    config: PrototypeAgentConfig,
) -> Array:
    """Bind a v17 feature/world state to the complete serialized agent config."""

    if (
        config.prototype_feature_lifecycle is None
        or config.world_model is None
    ):
        raise ValueError(
            "feature/world-model schema requires both configured components"
        )
    payload = {
        "schema": "alberta.prototype_feature_world_model.v1",
        "agent_config": config.to_config(),
    }
    encoded = json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return jnp.asarray(
        tuple(hashlib.sha256(encoded).digest()),
        dtype=jnp.uint8,
    )


def _prototype_atomic_feature_world_memory_schema_digest(
    config: PrototypeAgentConfig,
) -> Array:
    """Bind the v18 state shell to the complete enabled Prototype config."""

    if config.prototype_atomic_feature_world_memory is None:
        raise ValueError("atomic feature/world/memory schema requires opt-in config")
    payload = {
        "schema": "alberta.prototype_atomic_feature_world_memory.v1",
        "agent_config": config.to_config(),
    }
    encoded = json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return jnp.asarray(
        tuple(hashlib.sha256(encoded).digest()),
        dtype=jnp.uint8,
    )


def _prototype_feature_utility_schema_digest(
    config: PrototypeAgentConfig,
) -> Array:
    """Bind causal-utility semantics to the exact shared consumer schema."""

    feature_config = config.prototype_feature_lifecycle
    utility_config = config.prototype_feature_utility
    horde_spec = config.horde_spec
    if (
        feature_config is None
        or utility_config is None
        or feature_config.managed_horde_demons <= 0
        or type(horde_spec) is not HordeSpec
    ):
        raise ValueError(
            "feature utility schema requires the shared feature/Horde lane"
        )
    payload = {
        "schema": "alberta.prototype_feature_utility_composition.v1",
        "task_mapping": {
            "control_target": 0,
            "ordered_horde_targets_start": 1,
        },
        "feature_utility": utility_config.to_config(),
        "feature_lifecycle": feature_config.to_config(),
        "oak": config.oak.to_config(),
        "horde_spec": horde_spec.to_config(),
        "horde_hidden_sizes": list(config.horde_hidden_sizes),
        "horde_step_size": config.horde_step_size,
    }
    encoded = json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return jnp.asarray(
        tuple(hashlib.sha256(encoded).digest()),
        dtype=jnp.uint8,
    )


def _prototype_feature_utility_curation_schema_digest(
    config: PrototypeAgentConfig,
) -> Array:
    """Bind v6 ranking semantics around the exact v5 utility composition."""

    feature_config = config.prototype_feature_lifecycle
    utility_config = config.prototype_feature_utility
    curation_config = config.prototype_feature_utility_curation
    horde_spec = config.horde_spec
    if (
        feature_config is None
        or utility_config is None
        or curation_config is None
        or feature_config.managed_horde_demons <= 0
        or type(horde_spec) is not HordeSpec
    ):
        raise ValueError(
            "feature utility curation schema requires the shared v5 utility "
            "composition"
        )
    payload = {
        "schema": "alberta.prototype_feature_utility_curation_composition.v1",
        "rank_semantics": {
            "active_direction": "ascending_deletion_utility",
            "candidate_direction": "descending_insertion_utility",
            "active_ineligible_rank": "float32:+0x1.fffffep+127",
            "candidate_ineligible_rank": "float32:-0x1.fffffep+127",
            "evidence_gate": "all_configured_tasks",
            "missing_task_mass": "fixed_without_renormalization",
            "cross_cohort_audit_comparison": False,
            "legacy_proxy_promotion_gate_retained": True,
        },
        "feature_utility_curation": curation_config.to_config(),
        "feature_utility": utility_config.to_config(),
        "feature_lifecycle": feature_config.to_config(),
        "oak": config.oak.to_config(),
        "horde_spec": horde_spec.to_config(),
        "horde_hidden_sizes": list(config.horde_hidden_sizes),
        "horde_step_size": config.horde_step_size,
    }
    encoded = json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return jnp.asarray(
        tuple(hashlib.sha256(encoded).digest()),
        dtype=jnp.uint8,
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
class PrototypeFeatureRepresentationState:
    """Enabled-only composition inside the historical builder-state slot.

    Disabled configurations retain the exact pre-existing
    ``state_builder_state`` PyTree.  The wrapper exists only when the bounded
    Prototype feature lifecycle is configured.
    """

    builder_state: Any
    feature_lifecycle_state: PrototypeFeatureLifecycleState


@chex.dataclass(frozen=True)
class PrototypeLearningValueRouterState:
    """Enabled-only owner bundle in the historical builder-state slot.

    ``representation_state`` is exactly the PyTree that would occupy
    ``PrototypeAgentState.state_builder_state`` with routing disabled. Keeping
    the router in this opt-in outer shell leaves every historical config and
    checkpoint template unchanged while binding one causal router state to the
    accepted real-transition owner.
    """

    representation_state: Any
    learning_value_router_state: LearningValueRouterState


@chex.dataclass(frozen=True)
class PrototypeFeatureWorldModelState:
    """V17 world-model state bound to the exact full Prototype config."""

    model_state: ActionConditionedWorldModelState
    schema_digest: UInt[Array, " 32"]


@chex.dataclass(frozen=True)
class PrototypeAtomicFeatureWorldMemoryState:
    """V18 routed-world slot bound to the complete atomic composition."""

    world_state: PrototypeRoutedLinearWorldState
    schema_digest: UInt[Array, " 32"]


@chex.dataclass(frozen=True)
class PrototypeFeatureOaKState:
    """Enabled-only OaK subtree coupled to its exact feature-bank identity."""

    oak_state: OaKState
    consumer_binding: PrototypeFeatureConsumerBinding


@chex.dataclass(frozen=True)
class PrototypeFeatureOaKHordeState:
    """Atomic OaK/Horde consumers bound to one ordered feature-bank schema.

    This wrapper exists only for the narrow linear-Horde feature-lifecycle
    composition.  The historical top-level ``horde_state`` slot remains
    ``None`` so the two feature-indexed consumers cannot be swapped or
    restored independently.
    """

    oak_state: OaKState
    horde_state: MultiHeadMLPState
    consumer_binding: PrototypeFeatureConsumerBinding
    schema_digest: UInt[Array, " 32"]


@chex.dataclass(frozen=True)
class PrototypeFeatureOaKHordeUtilityState:
    """Enabled-only atomic consumers plus bound causal-utility audit state."""

    consumer_state: PrototypeFeatureOaKHordeState
    feature_utility_state: PrototypeFeatureUtilityState
    schema_digest: UInt[Array, " 32"]

    @property
    def oak_state(self) -> OaKState:
        """Expose the nested OaK state without duplicating persistent leaves."""

        return self.consumer_state.oak_state

    @property
    def horde_state(self) -> MultiHeadMLPState:
        """Expose the nested Horde state without duplicating persistent leaves."""

        return self.consumer_state.horde_state

    @property
    def consumer_binding(self) -> PrototypeFeatureConsumerBinding:
        """Expose the nested bank identity."""

        return self.consumer_state.consumer_binding


@chex.dataclass(frozen=True)
class PrototypeFeatureOaKHordeUtilityCurationState:
    """V6 shell binding ranking semantics around the unchanged v5 bundle."""

    utility_state: PrototypeFeatureOaKHordeUtilityState
    schema_digest: UInt[Array, " 32"]

    @property
    def oak_state(self) -> OaKState:
        """Expose the nested OaK state without duplicating it."""

        return self.utility_state.oak_state

    @property
    def horde_state(self) -> MultiHeadMLPState:
        """Expose the nested Horde state without duplicating it."""

        return self.utility_state.horde_state

    @property
    def consumer_binding(self) -> PrototypeFeatureConsumerBinding:
        """Expose the nested feature-bank identity."""

        return self.utility_state.consumer_binding

    @property
    def feature_utility_state(self) -> PrototypeFeatureUtilityState:
        """Expose the unchanged v5 auditor state."""

        return self.utility_state.feature_utility_state


@chex.dataclass(frozen=True)
class PrototypeAgentState:
    """Full prototype agent state.

    Optional sub-states (``world_model_state``, ``buffer_state``,
    ``horde_state``, ``ia_state``) are normally ``None`` when the corresponding
    component is disabled.  The shared feature/Horde lane is the deliberate
    exception: its top-level ``horde_state`` is ``None`` because the live
    Horde is atomically stored with OaK in
    :class:`PrototypeFeatureOaKHordeState`, or inside
    :class:`PrototypeFeatureOaKHordeUtilityState` on the diagnostic utility
    lane, or one additional
    :class:`PrototypeFeatureOaKHordeUtilityCurationState` shell on the v6
    ranking-influence lane. The opt-in learning-value router similarly wraps
    the otherwise unchanged ``state_builder_state`` in exactly one
    :class:`PrototypeLearningValueRouterState`. The PyTree structure is fixed
    for a given :class:`PrototypeAgent` instance — never switch between
    ``None`` and a real state after initialisation.

    ``step_words`` and ``observation_event_words`` make the Prototype's outer
    real-transition and observation identities exact. IA, partner fusion, and
    feature lifecycle and feature utility authenticate their own exact word
    identities against these clocks; explicitly smaller observation budgets
    remain deliberate capacity caps. The feature/action-world lane binds its
    model clock exactly to the outer transaction; the legacy direct-world lane
    may lag after a best-effort model refusal. Other recurrent/model lanes
    retain configured update caps or bounded owner counters and are outside
    this outer-clock lifetime claim. Exact identity never implies unbounded
    child capacity or infinite-precision utility estimates.
    """

    oak_state: Any  # OaKState | PrototypeFeatureOaKState
    world_model_state: Any  # Configured model state or exact composition wrapper.
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
    # The int32 scalars are saturating compatibility telemetry.  Scheduling,
    # ownership, and checkpoint identity use the exact uint32 word clocks.
    observation_event_count: Int[Array, ""]
    observation_event_words: UInt[Array, " 2"]
    step_count: Int[Array, ""]
    step_words: UInt[Array, " 2"]


@dataclasses.dataclass(frozen=True)
class PrototypeAgentStateResourceMeasurement:
    """Measured persistent JAX-array resources for one concrete Prototype state.

    Bundle names follow the top-level state ownership boundary.  For example,
    a shared feature/Horde lane is counted once inside ``oak_bundle_nbytes``;
    experiential memory is counted once inside
    ``interaction_memory_bundle_nbytes``. Python objects are excluded. Legacy
    host timing scalars that :meth:`PrototypeAgent.init` materializes as JAX
    arrays are included in the measured bytes, elements, and leaves even
    though they remain outside the learning-state identity semantics.
    """

    total_nbytes: int
    total_array_elements: int
    total_array_leaves: int
    oak_bundle_nbytes: int
    world_model_bundle_nbytes: int
    buffer_nbytes: int
    standalone_horde_nbytes: int
    interaction_memory_bundle_nbytes: int
    gru_nbytes: int
    state_builder_feature_bundle_nbytes: int
    outer_nbytes: int

    def to_config(self) -> dict[str, int]:
        """Return an exact JSON-compatible measurement payload."""

        return dataclasses.asdict(self)


@chex.dataclass(frozen=True)
class PrototypeInteractionState:
    """Opt-in IA-slot composition for partner fusion.

    This wrapper is created only when ``partner_policy_fusion`` is configured.
    Consequently disabled and historical IA-only states retain their exact
    ``None``/``IAState`` PyTree shape. ``ia_state`` may be ``None`` when fusion
    is enabled without the intelligence-amplification companion.
    """

    ia_state: Any
    partner_policy_fusion_state: PartnerPolicyFusionState
    feedback_prototype_decision_id: UInt[Array, " 4"]
    feedback_prototype_decision_id_available: Bool[Array, ""]


@chex.dataclass(frozen=True)
class PrototypeMemoryInteractionState:
    """Opt-in outer wrapper for persistent experiential memory.

    ``interaction_state`` is the exact state that would occupy ``ia_state``
    without memory: ``None``, a bare IA state, or the partner-fusion
    :class:`PrototypeInteractionState`. Consequently every no-memory lane
    retains its historical PyTree exactly.
    """

    interaction_state: Any
    # The enabled-only feature-lifecycle composition stores a
    # PrototypeFeatureMemoryState here; every historical/no-lifecycle memory
    # configuration retains the bare ExperientialMemoryState PyTree.
    experiential_memory_state: Any


@chex.dataclass(frozen=True)
class PrototypeRecurrentLatentWorldModelState:
    """Recurrent lane state stored inside the existing world-model slot.

    ``decision_cache`` owns exactly the currently dispatched Prototype
    representation/action. ``signal_state`` consumes only the same cached
    predict-before-update event as ``model_state``. The wrapper keeps the
    top-level :class:`PrototypeAgentState` shape and every disabled/legacy
    checkpoint shape unchanged.
    """

    model_state: RecurrentLatentWorldModelEnsembleState
    decision_cache: RecurrentLatentDecisionCache
    signal_state: LearningSignalEstimatorState


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
class PrototypeCachedPrimitiveActionReplacement:
    """Atomic replacement of every owner of one cached primitive dispatch.

    ``committed`` is the authority for the outer transaction. The nested
    STOMP diagnostic may describe a valid counterfactual replacement even
    when stale Prototype provenance vetoes the commit; in that case the
    complete Prototype state and returned action are exact no-ops.
    """

    state: PrototypeAgentState
    action: Int[Array, ""]
    dispatch_replacement: DispatchedPrimitiveActionDecision
    decision_id_matches: Bool[Array, ""]
    observation_matches: Bool[Array, ""]
    state_valid_before: Bool[Array, ""]
    state_valid_after: Bool[Array, ""]
    committed: Bool[Array, ""]


@chex.dataclass(frozen=True)
class PrototypeTransitionDiagnostics:
    """Fail-closed ownership and boundary checks for one real transition."""

    pre_step_words: UInt[Array, " 2"]
    proposed_step_words: UInt[Array, " 2"]
    pre_observation_event_words: UInt[Array, " 2"]
    proposed_observation_event_words: UInt[Array, " 2"]
    outer_counter_valid: Bool[Array, ""]
    current_counter_capacity_available: Bool[Array, ""]
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
class PrototypeRTUTransitionPreparation:
    """Read-only RTU recurrence preparation for an external causal learner.

    Preparation owns the exact source state and normalized transition, advances
    the RTU through the real bootstrap plus optional reset/restart observation,
    and performs no control/model learner update or action-selection RNG draw.
    :meth:`PrototypeAgent.finalize_rtu_transition` recomputes this record before
    accepting an externally learned/recycled destination.
    """

    source_state: PrototypeAgentState
    transition: PrototypeTransition
    transition_diagnostics: PrototypeTransitionDiagnostics
    source_builder_state: RecurrentTraceUnitStateBuilderState
    bootstrap_transition: RecurrentTraceUnitStateBuilderTransitionResult
    decision_builder_state: RecurrentTraceUnitStateBuilderState
    decision_representation: Float[Array, " representation_dim"]
    execution_boundary: Bool[Array, ""]
    preparation_valid: Bool[Array, ""]


@chex.dataclass(frozen=True)
class PrototypeRTUFinalizationReceipt:
    """Content-bound RTU proposal and derived destination for one preparation.

    The tag detects mutation but grants no authority by itself.  Finalization
    independently recomputes the proposal through the exact supplied RTU
    mechanism before accepting the destination.  Neither step authenticates
    the external caller's lifecycle source, downstream objective/gradient, or
    source-bound ordinary learning proposal.
    """

    preparation: PrototypeRTUTransitionPreparation
    rtu_proposal: RTUGenerateAndTestProposal
    final_builder_state: RecurrentTraceUnitStateBuilderState
    replaced_unit_mask: Bool[Array, " hidden_dim"]
    content_tag_words: UInt[Array, " 8"]


@chex.dataclass(frozen=True)
class PrototypeRecurrentLatentDiagnostics:
    """Causal recurrent-lane transaction and uncertainty disclosures.

    The two uncertainty values are raw ensemble outputs, not calibrated
    probabilities or confidence intervals. ``prediction_availability`` honors
    the recurrent model's configured warmup. ``transaction_applied`` requires
    both the recurrent update and its causal signal-state update to commit.
    """

    model: RecurrentLatentWorldModelDiagnostics
    prediction_availability: RecurrentLatentPredictionAvailability
    raw_epistemic_disagreement: Float[Array, ""]
    raw_aleatoric_uncertainty: Float[Array, ""]
    raw_uncertainty_calibrated: Bool[Array, ""]
    signals_valid: Bool[Array, ""]
    transaction_applied: Bool[Array, ""]
    next_decision_cached: Bool[Array, ""]


@chex.dataclass(frozen=True)
class PrototypeBehaviorGradientDiagnostics:
    """Causal provenance and numerics for one control-loss input gradient.

    Exactly one source flag is true on an available result. ``idle_base_source``
    denotes the one-step primitive differential base-Q loss. During an option,
    ``intra_option_source`` denotes the current option's one-step TD loss; the
    delayed semi-MDP base update belongs to ``option_start_obs`` and is
    deliberately excluded from the current representation.
    """

    source_available: Bool[Array, ""]
    idle_base_source: Bool[Array, ""]
    intra_option_source: Bool[Array, ""]
    parameters_finite: Bool[Array, ""]
    inputs_finite: Bool[Array, ""]
    source_indices_valid: Bool[Array, ""]
    prediction_finite: Bool[Array, ""]
    target_finite: Bool[Array, ""]
    td_error_finite: Bool[Array, ""]
    loss_finite: Bool[Array, ""]
    gradient_norm_finite: Bool[Array, ""]
    prediction: Float[Array, ""]
    target: Float[Array, ""]
    td_error: Float[Array, ""]
    loss: Float[Array, ""]
    gradient_norm: Float[Array, ""]
    bootstrap_discount: Float[Array, ""]
    option_terminates: Bool[Array, ""]
    gradient_finite: Bool[Array, ""]
    valid: Bool[Array, ""]
    rejected: Bool[Array, ""]


@chex.dataclass(frozen=True)
class PrototypeBehaviorGradientResult:
    """Pre-mix gradient of one frozen-target TD loss.

    ``raw`` here means before source weighting/mixing. A rejected numerical
    computation is represented by an exact finite zero plus ``valid=False``.
    """

    gradient: Float[Array, " observation_dim"]
    valid: Bool[Array, ""]
    diagnostics: PrototypeBehaviorGradientDiagnostics


@chex.dataclass(frozen=True)
class PrototypeFeatureLifecycleIntegrationDiagnostics:
    """Compact audit of one Prototype-owned discovery transaction."""

    available: Bool[Array, ""]
    target: Float[Array, ""]
    target_available: Bool[Array, ""]
    pullback_gradient: Float[Array, " base_feature_dim"]
    pullback_valid: Bool[Array, ""]
    pullback_semantic_generation: Int[Array, ""]
    prediction: Float[Array, ""]
    error: Float[Array, ""]
    task_targets: Float[Array, " n_tasks"]
    task_target_available: Bool[Array, " n_tasks"]
    task_predictions: Float[Array, " n_tasks"]
    task_errors: Float[Array, " n_tasks"]
    metrics: Float[Array, " 7"]
    lifecycle: PrototypeFeatureLifecycleDiagnostics
    outer_transaction_committed: Bool[Array, ""]


@chex.dataclass(frozen=True)
class PrototypeFeatureUtilityIntegrationDiagnostics:
    """Separate old-bank scoring from any destination-generation rebind."""

    observation: PrototypeFeatureUtilityDiagnostics
    rebind: PrototypeFeatureUtilityDiagnostics
    rebind_required: Bool[Array, ""]
    outer_transaction_committed: Bool[Array, ""]


@chex.dataclass(frozen=True)
class PrototypeFeatureUtilityCurationIntegrationDiagnostics:
    """Audit-ranked curation facts without cross-cohort promotion authority."""

    policy: PrototypeFeatureUtilityCurationDiagnostics
    observation_applied: Bool[Array, ""]
    priority_override_supplied: Bool[Array, ""]
    priority_override_consulted: Bool[Array, ""]
    curation_allowed: Bool[Array, ""]
    selected_active_slot: Int[Array, ""]
    selected_candidate_slot: Int[Array, ""]
    selected_active_descriptor: Int[Array, " 2"]
    selected_candidate_descriptor: Int[Array, " 2"]
    lifecycle_curation_proposed: Bool[Array, ""]
    lifecycle_curation_deferred: Bool[Array, ""]
    lifecycle_curation_committed: Bool[Array, ""]
    lifecycle_curation_rolled_back: Bool[Array, ""]
    outer_transaction_committed: Bool[Array, ""]


@chex.dataclass(frozen=True)
class PrototypeCandidateUpdateAuditEvidence:
    """Decision-bound probe evidence for one representation update.

    The three parameter-gradient probes must be measured independently of the
    world-model gradient that forms the candidate update. ``advantage`` and
    ``action_surprisal`` come from the behavior actor, while ``safety_cost``
    comes from the separately trained safety learner. The remaining four
    learning-value channels are supplied causally by the configured world
    model ensemble. The paper-defined delight value is derived internally as
    ``advantage * action_surprisal`` and is never accepted from a caller.

    Every availability flag is explicit. Missing evidence, a stale decision
    ID, a false independence attestation, or any unavailable/non-finite input
    causes the candidate-update audit to fail closed without invalidating the
    real control transition. It is a candidate-update audit sidecar, not the
    paper's Kondo backward-selection decision.
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


# Historical compatibility spelling. Canonical callers use the candidate-audit
# name above so “gradient sparks joy” remains reserved for an executed actor
# backward contribution.
PrototypeGradientJoyEvidence = PrototypeCandidateUpdateAuditEvidence


@chex.dataclass(frozen=True)
class PrototypeExperientialMemoryInput:
    """Learner-visible metadata for one causal memory query/write event.

    Both lifecycle identifiers are full Prototype tokens. The current token
    binds the automatically grounded exemplar to the action that really ran;
    the next token binds the retrieval to the decision it may alter. Opaque
    provenance/source identifiers are not task, regime, evaluator, or target
    labels. Static shape/dtype drift raises; dynamic invalidity is an exact
    memory no-op and ordinary OaK decision.
    """

    available: Bool[Array, ""]
    current_prototype_decision_id: UInt[Array, " 4"]
    next_prototype_decision_id: UInt[Array, " 4"]
    query_representation_version: Int[Array, ""]
    entry_representation_version: Int[Array, ""]
    query_uncertainty: Float[Array, ""]
    query_uncertainty_available: Bool[Array, ""]
    entry_uncertainty: Float[Array, ""]
    entry_uncertainty_available: Bool[Array, ""]
    safety_cost: Float[Array, ""]
    safety_cost_available: Bool[Array, ""]
    reliability: Float[Array, ""]
    utility: Float[Array, ""]
    utility_available: Bool[Array, ""]
    provenance_id: Int[Array, ""]
    source_id: Int[Array, ""]
    next_action_safety_mask: Bool[Array, " actions"]


@chex.dataclass(frozen=True)
class PrototypeExperientialMemoryDiagnostics:
    """Complete causal memory, categorical proposal, and dispatch audit."""

    proposal: ExperientialMemoryPolicyProposal
    advantage_gate: ExperientialMemoryAdvantageGateDiagnostics | None
    dispatch_replacement: DispatchedPrimitiveActionDecision
    input_supplied: Bool[Array, ""]
    input_available: Bool[Array, ""]
    current_prototype_decision_id_matches: Bool[Array, ""]
    next_prototype_decision_id_matches: Bool[Array, ""]
    metadata_valid: Bool[Array, ""]
    transaction_required: Bool[Array, ""]
    retrieval_matches: Bool[Array, ""]
    query_before_write: Bool[Array, ""]
    deterministic_prestate_query_count: Int[Array, ""]
    wrote: Bool[Array, ""]
    slot: Int[Array, ""]
    evicted: Bool[Array, ""]
    evicted_provenance_id: Int[Array, ""]
    transaction_applied: Bool[Array, ""]
    counterfactual_base_action: Int[Array, ""]
    effective_action: Int[Array, ""]


@chex.dataclass(frozen=True)
class PrototypeFeatureMemoryIntegrationDiagnostics:
    """Exact feature-bank migration and memory-version composition audit."""

    rebind: PrototypeFeatureMemoryRebindDiagnostics
    query_source_version_matches: Bool[Array, ""]
    entry_source_version_matches: Bool[Array, ""]
    source_versions_match: Bool[Array, ""]
    current_destination_encoding_valid: Bool[Array, ""]
    bootstrap_destination_encoding_valid: Bool[Array, ""]
    decision_destination_encoding_valid: Bool[Array, ""]
    post_memory_state_valid: Bool[Array, ""]
    outer_transaction_committed: Bool[Array, ""]


@chex.dataclass(frozen=True)
class PrototypeAtomicFeatureWorldMemoryDiagnostics:
    """Audit one all-consumer readiness and adoption decision."""

    available: Bool[Array, ""]
    descriptor_change_requested: Bool[Array, ""]
    lifecycle_destination_ready: Bool[Array, ""]
    world_ordinary_ready: Bool[Array, ""]
    world_destination_ready: Bool[Array, ""]
    memory_destination_ready: Bool[Array, ""]
    all_consumers_ready: Bool[Array, ""]
    destination_adopted: Bool[Array, ""]
    ordinary_updates_retained: Bool[Array, ""]
    external_curation_rolled_back: Bool[Array, ""]
    lifecycle_adoption: PrototypeFeatureLifecycleAdoptionDiagnostics
    world_adoption: PrototypeRoutedLinearWorldAdoptionDiagnostics
    oak_update_evaluations: Int[Array, ""]
    horde_update_evaluations: Int[Array, ""]
    feature_learner_update_evaluations: Int[Array, ""]
    lifecycle_router_evaluations: Int[Array, ""]
    world_learner_update_evaluations: Int[Array, ""]
    world_router_evaluations: Int[Array, ""]
    memory_rebind_evaluations: Int[Array, ""]
    memory_step_evaluations: Int[Array, ""]


@dataclasses.dataclass(frozen=True)
class PrototypeExperientialMemoryResourceDeclaration:
    """Exact persistent allocation and bounded work per required transaction."""

    persistent_state_bytes: int
    categorical_policy_queries: int
    causal_step_queries: int
    total_deterministic_prestate_queries: int
    writes_attempted: int
    random_draws: int
    score_mass_values_interpreted: int
    hard_safety_values_interpreted: int

    def to_config(self) -> dict[str, int]:
        """Return the exact JSON-compatible resource declaration."""

        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class PrototypeAtomicFeatureWorldMemoryResourceBudget:
    """Exact enabled-state ownership and per-transition call maxima."""

    mechanism_status: str
    scientific_promotion_allowed: bool
    persistent_state_nbytes: int
    persistent_capacity_growth: int
    lifecycle_authority_count: int
    router_authority_count: int
    oak_consumer_count: int
    ordered_linear_horde_count: int
    routed_world_count: int
    world_model_buffer_count: int
    experiential_memory_count: int
    mirrored_binding_cache_count: int
    mirrored_binding_cache_nbytes: int
    oak_update_evaluations_per_transition: int
    horde_update_evaluations_per_transition: int
    feature_learner_update_evaluations_per_transition: int
    lifecycle_router_evaluations_per_transition: int
    world_learner_update_evaluations_per_transition: int
    world_router_evaluations_per_transition: int
    memory_rebind_evaluations_per_transition: int
    memory_step_evaluations_per_transition: int
    deterministic_prestate_memory_queries_per_transition: int
    memory_writes_attempted_per_transition: int

    def to_config(self) -> dict[str, str | int | bool]:
        """Return an exact JSON-compatible declaration."""

        return dataclasses.asdict(self)


@chex.dataclass(frozen=True)
class PrototypePartnerPolicyFusionInput:
    """Typed sidecar for the next primitive-action decision.

    The Prototype derives the fusion ``decision_id`` and ``event_id`` from its
    authoritative counters. Messages must bind those exact identifiers. The
    remaining integer references are opaque provenance/context identities, not
    learner-visible task or regime labels. Dynamic invalidity safely disables
    fusion and preserves OaK's counterfactual base primitive; malformed static
    shapes or dtypes raise before traced execution.
    """

    available: Bool[Array, ""]
    prototype_decision_id: UInt[Array, " 4"]
    observation_id: Int[Array, ""]
    context_id: Int[Array, ""]
    context_features: Float[Array, " context_dim"]
    safety_action_mask: Bool[Array, " actions"]
    keyboard_available: Bool[Array, ""]
    keyboard_vector: Float[Array, " options"]
    messages: PartnerMessageBatch


@chex.dataclass(frozen=True)
class PrototypePartnerPolicyFusionFeedback:
    """Realized partner feedback bound to the full Prototype lifecycle ID."""

    prototype_decision_id: UInt[Array, " 4"]
    feedback: PartnerPolicyFusionFeedback


@chex.dataclass(frozen=True)
class PrototypePartnerPolicyFusionDiagnostics:
    """Complete feedback, fusion, and dispatch audit for one transition."""

    feedback: PartnerFusionFeedbackResult
    decision: PartnerFusionDecision
    keyboard_proposal: OaKKeyboardPolicyProposal
    dispatch_replacement: DispatchedPrimitiveActionDecision
    feedback_input_supplied: Bool[Array, ""]
    decision_input_supplied: Bool[Array, ""]
    feedback_prototype_decision_id_matches: Bool[Array, ""]
    decision_prototype_decision_id_matches: Bool[Array, ""]
    transaction_applied: Bool[Array, ""]
    counterfactual_base_action: Int[Array, ""]
    effective_action: Int[Array, ""]


@chex.dataclass(frozen=True)
class PrototypeUpdateResult:
    """Result of one real-time prototype agent transition.

    The ``model_replay_*`` fields describe only work committed by the complete
    all-or-nothing model/replay transaction. They remain false/zero for the
    plain ensemble and legacy model lanes and for any rolled-back rehearsal.
    Candidate audit acceptance and committed application are deliberately
    separate finite-precision facts. The ``sparks_joy`` and
    ``joyful_gradient_applied`` properties are historical compatibility names;
    paper-defined “sparks joy” means the sample gradient actually enters an
    executed ``KondoSparseActor`` backward.

    ``learning_value_router_result`` is ``None`` on historical disabled
    configs. On the opt-in lane it is the one raw/normalized routing receipt
    for an accepted transition, or an explicit no-event receipt when the outer
    transaction rejects. Candidate auditing consumes only its raw evidence
    route.
    """

    state: PrototypeAgentState
    action: Int[Array, ""]
    oak_td_error: Float[Array, ""]
    oak_average_reward: Float[Array, ""]
    oak_stomp_update_result: STOMPUpdateResult
    oak_stomp_update_available: Bool[Array, ""]
    oak_stomp_update_evaluations: Int[Array, ""]
    oak_owner_finalization_trace: STOMPOwnerFinalizationTrace
    oak_real_stomp_update_evaluations: Int[Array, ""]
    oak_imagined_stomp_update_evaluations: Int[Array, ""]
    oak_total_stomp_update_evaluations: Int[Array, ""]
    oak_option_search_learner_updates: Int[Array, ""]
    oak_bootstrap_observation: Float[Array, " observation_dim"]
    oak_decision_observation: Float[Array, " observation_dim"]
    oak_execution_boundary: Bool[Array, ""]
    world_model_error: Any  # Float[Array, ""] | None
    learning_signals: TypedLearningSignals
    learning_value_router_result: LearningValueRouterResult | None
    world_model_representation_gradient: Float[Array, " observation_dim"]
    world_model_representation_gradient_valid: Bool[Array, ""]
    behavior_gradient_result: PrototypeBehaviorGradientResult
    representation_gradient_mix: RepresentationGradientMixResult
    world_model_ensemble_diagnostics: WorldModelEnsembleDiagnostics
    recurrent_latent_world_model_diagnostics: PrototypeRecurrentLatentDiagnostics
    model_replay_transaction_applied: Bool[Array, ""]
    model_replay_recorded: Bool[Array, ""]
    model_replay_sampled: Bool[Array, ""]
    model_replay_updates_applied: Int[Array, ""]
    model_replay_padding_count: Int[Array, ""]
    state_builder_learning_diagnostics: StateBuilderLearningDiagnostics
    gradient_joy_application: Any  # CandidateUpdateAuditApplicationResult | None
    gradient_joy_evidence_supplied: Bool[Array, ""]
    gradient_joy_decision_id_matches: Bool[Array, ""]
    option_search_control_diagnostics: OptionSearchControlDiagnostics | None
    prototype_feature_lifecycle_diagnostics: (
        PrototypeFeatureLifecycleIntegrationDiagnostics | None
    )
    prototype_feature_utility_diagnostics: (
        PrototypeFeatureUtilityIntegrationDiagnostics | None
    )
    prototype_feature_utility_curation_diagnostics: (
        PrototypeFeatureUtilityCurationIntegrationDiagnostics | None
    )
    dream_td_errors: Any  # Float[Array, " n_dreams"] | None
    horde_td_errors: Any  # Float[Array, " n_demons"] | None
    ia_augmented_obs: Any  # Float[Array, " augmented_dim"] | None
    ia_recommendation: Any  # Int[Array, ""] | None
    ia_update_applied: Bool[Array, ""]
    experiential_memory_diagnostics: PrototypeExperientialMemoryDiagnostics | None
    prototype_feature_memory_diagnostics: (
        PrototypeFeatureMemoryIntegrationDiagnostics | None
    )
    partner_policy_fusion_diagnostics: PrototypePartnerPolicyFusionDiagnostics | None
    transition_diagnostics: PrototypeTransitionDiagnostics
    prototype_atomic_feature_world_memory_diagnostics: (
        PrototypeAtomicFeatureWorldMemoryDiagnostics | None
    ) = None

    @property
    def candidate_update_audit_application(
        self,
    ) -> CandidateUpdateAuditApplicationResult | None:
        """Return the historical audit application's canonical view."""
        return cast(
            CandidateUpdateAuditApplicationResult | None,
            self.gradient_joy_application,
        )

    @property
    def candidate_update_audit_evidence_supplied(self) -> Bool[Array, ""]:
        """Return whether a candidate-audit evidence sidecar was supplied."""
        return self.gradient_joy_evidence_supplied

    @property
    def candidate_update_audit_decision_id_matches(self) -> Bool[Array, ""]:
        """Return whether candidate-audit evidence names this decision."""
        return self.gradient_joy_decision_id_matches

    @property
    def candidate_update_audit_passed(self) -> Bool[Array, ""]:
        """Return whether this event's multi-objective candidate audit passed."""
        application = self.candidate_update_audit_application
        if application is None:
            return jnp.asarray(False, dtype=jnp.bool_)
        return application.assessment.candidate_update_audit_passed

    @property
    def sparks_joy(self) -> Bool[Array, ""]:
        """Historical alias; paper-defined “sparks joy” belongs to Kondo."""
        return self.candidate_update_audit_passed

    @property
    def behavior_representation_gradient(self) -> Float[Array, " observation_dim"]:
        """Return the raw causal control-loss input gradient."""
        return self.behavior_gradient_result.gradient

    @property
    def behavior_representation_gradient_valid(self) -> Bool[Array, ""]:
        """Return whether the raw control-loss gradient is usable."""
        return self.behavior_gradient_result.valid

    @property
    def mixed_representation_gradient(self) -> Float[Array, " observation_dim"]:
        """Return the explicitly mixed builder candidate, or exact zero."""
        return self.representation_gradient_mix.gradient

    @property
    def mixed_representation_gradient_valid(self) -> Bool[Array, ""]:
        """Return whether a valid active mixed source produced a candidate."""
        return self.representation_gradient_mix.applied

    @property
    def audited_candidate_update_applied(self) -> Bool[Array, ""]:
        """Report whether the audited finite-precision update was committed."""
        application = self.candidate_update_audit_application
        if application is None:
            return jnp.asarray(False, dtype=jnp.bool_)
        return application.applied & self.state_builder_learning_diagnostics.applied

    @property
    def joyful_gradient_applied(self) -> Bool[Array, ""]:
        """Historical alias; prefer ``audited_candidate_update_applied``."""
        return self.audited_candidate_update_applied


@chex.dataclass(frozen=True)
class PrototypeArrayResult:
    """Result from :meth:`PrototypeAgent.scan` over a batch of transitions.

    ``gradient_sparks_joy`` and ``joyful_gradient_applied`` are serialized
    compatibility fields for the historical candidate-update audit. Prefer
    the canonical vector properties ``candidate_update_audit_passed`` and
    ``audited_candidate_update_applied``. These values do not answer the
    paper's question; actual Kondo actor-backward admission is reported by
    :class:`~alberta_framework.core.kondo_sparse_actor.KondoSparseActorResult`.
    """

    state: PrototypeAgentState
    actions: Int[Array, " num_steps"]
    oak_td_errors: Float[Array, " num_steps"]
    oak_average_rewards: Float[Array, " num_steps"]
    transition_valid: Bool[Array, " num_steps"]
    state_builder_learning_applied: Bool[Array, " num_steps"]
    gradient_sparks_joy: Bool[Array, " num_steps"]
    joyful_gradient_applied: Bool[Array, " num_steps"]

    @property
    def candidate_update_audit_passed(self) -> Bool[Array, " num_steps"]:
        """Return the per-transition candidate-update audit decisions."""
        return self.gradient_sparks_joy

    @property
    def audited_candidate_update_applied(self) -> Bool[Array, " num_steps"]:
        """Return the per-transition audited update application decisions."""
        return self.joyful_gradient_applied


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


def _tree_arrays_equal(left: Any, right: Any) -> Bool[Array, ""]:
    """Compare two fixed PyTrees exactly without leaving traced execution."""
    left_leaves, left_structure = jax.tree_util.tree_flatten(left)
    right_leaves, right_structure = jax.tree_util.tree_flatten(right)
    if (
        cast(Any, left_structure) != right_structure
        or len(left_leaves) != len(right_leaves)
    ):
        return jnp.asarray(False, dtype=jnp.bool_)
    if not left_leaves:
        return jnp.asarray(True, dtype=jnp.bool_)
    return jnp.all(
        jnp.stack(
            [
                jnp.array_equal(jnp.asarray(a), jnp.asarray(b))
                for a, b in zip(left_leaves, right_leaves, strict=True)
            ]
        )
    )


def _prototype_rtu_uint32_words(value: Array) -> UInt[Array, " words"]:
    """Return exact words for one supported RTU finalization leaf."""

    array = jnp.asarray(value)
    if jnp.issubdtype(array.dtype, jax.dtypes.prng_key):
        return jr.key_data(array).reshape((-1,)).astype(jnp.uint32)
    if array.dtype == jnp.dtype(jnp.uint32):
        return array.reshape((-1,))
    if array.dtype in {jnp.dtype(jnp.float32), jnp.dtype(jnp.int32)}:
        return jax.lax.bitcast_convert_type(array, jnp.uint32).reshape((-1,))
    if array.dtype == jnp.dtype(jnp.bool_):
        return array.astype(jnp.uint32).reshape((-1,))
    raise TypeError(f"RTU finalization tag does not support dtype {array.dtype}")


def _prototype_rtu_tree_exact(left: Any, right: Any) -> Bool[Array, ""]:
    """Compare a fixed RTU receipt PyTree by exact typed array content."""

    left_leaves, left_structure = jax.tree.flatten(left)
    right_leaves, right_structure = jax.tree.flatten(right)
    if (
        cast(Any, left_structure) != right_structure
        or len(left_leaves) != len(right_leaves)
    ):
        return jnp.asarray(False, dtype=jnp.bool_)
    predicates: list[Array] = []
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        left_array = jnp.asarray(left_leaf)
        right_array = jnp.asarray(right_leaf)
        if (
            left_array.shape != right_array.shape
            or left_array.dtype != right_array.dtype
        ):
            return jnp.asarray(False, dtype=jnp.bool_)
        predicates.append(
            jnp.array_equal(
                _prototype_rtu_uint32_words(left_array),
                _prototype_rtu_uint32_words(right_array),
            )
        )
    if not predicates:
        return jnp.asarray(True, dtype=jnp.bool_)
    return jnp.all(jnp.stack(predicates))


def _prototype_rtu_destination_tag(
    preparation: PrototypeRTUTransitionPreparation,
    rtu_proposal: RTUGenerateAndTestProposal,
    final_builder_state: RecurrentTraceUnitStateBuilderState,
    replaced_unit_mask: Array,
) -> UInt[Array, " 8"]:
    """Bind the exact prepared event, RTU proposal, destination, and reset mask."""

    lanes = jnp.asarray(
        (
            0x6A09E667,
            0xBB67AE85,
            0x3C6EF372,
            0xA54FF53A,
            0x510E527F,
            0x9B05688C,
            0x1F83D9AB,
            0x5BE0CD19,
        ),
        dtype=jnp.uint32,
    )
    multipliers = jnp.asarray(
        (
            0x9E3779B1,
            0x85EBCA77,
            0xC2B2AE3D,
            0x27D4EB2F,
            0x165667B1,
            0xD3A2646D,
            0xFD7046C5,
            0xB55A4F09,
        ),
        dtype=jnp.uint32,
    )
    bound_values = (
        preparation.source_state.current_decision_id,
        preparation.source_state.step_words,
        preparation.source_state.observation_event_words,
        preparation.transition.observation,
        preparation.transition.action,
        preparation.transition.reward,
        preparation.transition.discount,
        preparation.transition.terminated,
        preparation.transition.truncated,
        preparation.transition.next_observation,
        preparation.transition.next_decision_observation,
        preparation.source_builder_state,
        preparation.bootstrap_transition,
        preparation.decision_builder_state,
        preparation.decision_representation,
        rtu_proposal,
        final_builder_state,
        replaced_unit_mask,
    )
    words = jnp.concatenate(
        tuple(
            _prototype_rtu_uint32_words(leaf)
            for leaf in jax.tree.leaves(bound_values)
        )
    )

    def mix(index: Array, state: Array) -> Array:
        word = words[index]
        lane = index & jnp.asarray(7, dtype=jnp.int32)
        mixed = state.at[lane].set(
            (state[lane] ^ word) * multipliers[lane]
        )
        neighbor = (lane + jnp.asarray(3, dtype=jnp.int32)) & jnp.asarray(
            7,
            dtype=jnp.int32,
        )
        return mixed.at[neighbor].set(
            mixed[neighbor] + jnp.left_shift(word, index & 15)
        )

    return jax.lax.fori_loop(0, words.shape[0], mix, lanes).astype(jnp.uint32)


def _recurrent_nll_estimator_offset(
    config: RecurrentLatentWorldModelEnsembleConfig,
) -> float:
    """Return a fixed affine offset making bounded Gaussian NLL non-negative.

    The offset is constant for a configured model, so loss-window differences
    are unchanged. A margin of one avoids float32 roundoff at the analytic
    minimum; the result diagnostics continue to report the unshifted NLL.
    """
    minimum = 0.5 * (math.log(2.0 * math.pi) + math.log(config.variance_floor))
    return max(1.0, 1.0 - minimum)


def _unavailable_learning_signals() -> TypedLearningSignals:
    """Return finite zeros whose typed channels are explicitly unavailable."""
    zero = jnp.asarray(0.0, dtype=jnp.float32)
    unavailable = jnp.asarray(False, dtype=jnp.bool_)
    zero_words = jnp.zeros((2,), dtype=jnp.uint32)
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
        counter_status=LearningSignalCounterStatus(
            pre_step_words=zero_words,
            post_step_words=zero_words,
            pre_valid_words=zero_words,
            post_valid_words=zero_words,
            pre_invalid_words=zero_words,
            post_invalid_words=zero_words,
            lifetime_counter_valid=unavailable,
            lifetime_capacity_available=unavailable,
            state_valid=unavailable,
            event_recorded=unavailable,
            valid_event_recorded=unavailable,
            invalid_event_recorded=unavailable,
        ),
    )


def _unavailable_stomp_update_result(oak_state: OaKState) -> STOMPUpdateResult:
    """Return a fixed-tree no-evaluation trace for a rejected Prototype step."""

    stomp = oak_state.stomp_state
    zero_float = jnp.asarray(0.0, dtype=jnp.float32)
    zero_int = jnp.asarray(0, dtype=jnp.int32)
    false = jnp.asarray(False, dtype=jnp.bool_)
    return STOMPUpdateResult(
        state=stomp,
        td_error=zero_float,
        average_reward=stomp.base_average_reward,
        primitive_action=stomp.last_primitive_action,
        executing_option=stomp.executing_option,
        option_terminated=false,
        pseudo_reward=zero_float,
        option_importance_ratio=zero_float,
        planning_backups=zero_int,
        planning_td_error=zero_float,
        pre_step_words=stomp.step_words,
        post_step_words=stomp.step_words,
        inputs_valid=false,
        lifetime_counter_valid=false,
        lifetime_capacity_available=false,
        nested_lifetime_counter_valid=false,
        nested_lifetime_capacity_available=false,
        nested_updates_required=zero_int,
        nested_updates_applied=zero_int,
        proposed_state_valid=false,
        update_applied=false,
    )


def _unavailable_stomp_owner_finalization_trace(
    oak_state: OaKState,
) -> STOMPOwnerFinalizationTrace:
    """Return a fixed-tree no-work trace for a rejected Prototype step."""

    stomp = oak_state.stomp_state
    false = jnp.asarray(False, dtype=jnp.bool_)
    true = jnp.asarray(True, dtype=jnp.bool_)
    zero = jnp.asarray(0, dtype=jnp.int32)
    zero_digest = jnp.zeros((8,), dtype=jnp.uint32)
    stages = tuple(
        make_stomp_owner_stage_receipt(
            stomp,
            stomp,
            stage_kind=stage_kind,
            configured=false,
            evaluated=false,
            stomp_update_evaluations=zero,
            learner_updates_applied=zero,
            source_digest=zero_digest,
            destination_digest=zero_digest,
            classified_delta_valid=true,
        )
        for stage_kind in (
            STOMP_OWNER_STAGE_OPTION_SEARCH,
            STOMP_OWNER_STAGE_FEATURE_ROUTE,
            STOMP_OWNER_STAGE_DYNA,
            STOMP_OWNER_STAGE_MEMORY_DISPATCH,
            STOMP_OWNER_STAGE_PARTNER_DISPATCH,
        )
    )
    return make_stomp_owner_finalization_trace(
        stomp,
        cast(Any, stages),
        stomp,
        real_control_stomp_evaluations=zero,
        imagined_stomp_evaluations=zero,
        option_search_learner_updates=zero,
        raw_digest=zero_digest,
        final_digest=zero_digest,
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


def _unavailable_recurrent_latent_diagnostics(
    ensemble_size: int,
) -> PrototypeRecurrentLatentDiagnostics:
    """Return fixed-shape, explicitly unavailable recurrent diagnostics."""
    unavailable = jnp.asarray(False, dtype=jnp.bool_)
    model = RecurrentLatentWorldModelDiagnostics(
        state_valid=unavailable,
        cache_valid=unavailable,
        input_valid=unavailable,
        ownership_valid=unavailable,
        boundary_semantics_valid=unavailable,
        capacity_available=unavailable,
        cached_prediction_exact=unavailable,
        predictions_valid=unavailable,
        losses_valid=unavailable,
        representation_gradient_valid=unavailable,
        member_gradients_valid=jnp.zeros((ensemble_size,), dtype=jnp.bool_),
        candidate_parameters_valid=jnp.zeros((ensemble_size,), dtype=jnp.bool_),
        candidate_state_valid=unavailable,
        recurrent_advanced_once=unavailable,
        recurrent_reset=unavailable,
        applied=unavailable,
        rejected=jnp.asarray(True, dtype=jnp.bool_),
    )
    availability = RecurrentLatentPredictionAvailability(
        prediction=unavailable,
        epistemic=unavailable,
        aleatoric=unavailable,
    )
    zero = jnp.asarray(0.0, dtype=jnp.float32)
    return PrototypeRecurrentLatentDiagnostics(
        model=model,
        prediction_availability=availability,
        raw_epistemic_disagreement=zero,
        raw_aleatoric_uncertainty=zero,
        raw_uncertainty_calibrated=unavailable,
        signals_valid=unavailable,
        transaction_applied=unavailable,
        next_decision_cached=unavailable,
    )


def _gate_recurrent_learning_signals(
    signals: TypedLearningSignals,
    availability: RecurrentLatentPredictionAvailability,
    transaction_applied: Array,
) -> TypedLearningSignals:
    """Honor model warmup while preserving each signal's native availability."""
    false = jnp.asarray(False, dtype=jnp.bool_)
    zero = jnp.asarray(0.0, dtype=jnp.float32)
    uncertainty_available = (
        transaction_applied
        & availability.epistemic
        & availability.aleatoric
        & signals.availability.epistemic
        & signals.availability.aleatoric
        & signals.availability.normalized_residual
    )
    progress_available = (
        transaction_applied & signals.availability.learning_progress
    )
    change_available = (
        uncertainty_available & signals.availability.change_probability
    )

    def gated(value: Array, available: Array) -> Array:
        return jnp.where(available, value, zero)

    return TypedLearningSignals(
        epistemic_disagreement=gated(
            signals.epistemic_disagreement,
            uncertainty_available,
        ),
        epistemic_surprise=gated(signals.epistemic_surprise, uncertainty_available),
        aleatoric_uncertainty=gated(
            signals.aleatoric_uncertainty,
            uncertainty_available,
        ),
        normalized_residual=gated(signals.normalized_residual, uncertainty_available),
        learning_progress=gated(signals.learning_progress, progress_available),
        calibrated_residual_z=gated(signals.calibrated_residual_z, change_available),
        instantaneous_change_probability=gated(
            signals.instantaneous_change_probability,
            change_available,
        ),
        change_probability=gated(signals.change_probability, change_available),
        availability=LearningSignalAvailability(
            input_valid=jnp.where(
                transaction_applied,
                signals.availability.input_valid,
                false,
            ),
            epistemic=uncertainty_available,
            aleatoric=uncertainty_available,
            normalized_residual=uncertainty_available,
            learning_progress=progress_available,
            change_probability=change_available,
        ),
        counter_status=signals.counter_status.replace(
            post_step_words=jnp.where(
                transaction_applied,
                signals.counter_status.post_step_words,
                signals.counter_status.pre_step_words,
            ),
            post_valid_words=jnp.where(
                transaction_applied,
                signals.counter_status.post_valid_words,
                signals.counter_status.pre_valid_words,
            ),
            post_invalid_words=jnp.where(
                transaction_applied,
                signals.counter_status.post_invalid_words,
                signals.counter_status.pre_invalid_words,
            ),
            event_recorded=(
                transaction_applied & signals.counter_status.event_recorded
            ),
            valid_event_recorded=(
                transaction_applied & signals.counter_status.valid_event_recorded
            ),
            invalid_event_recorded=(
                transaction_applied & signals.counter_status.invalid_event_recorded
            ),
        ),
    )


def _unavailable_behavior_gradient(
    observation_dim: int,
) -> PrototypeBehaviorGradientResult:
    """Return an exact-zero control source with explicit unavailability."""
    zero = jnp.asarray(0.0, dtype=jnp.float32)
    unavailable = jnp.asarray(False, dtype=jnp.bool_)
    diagnostics = PrototypeBehaviorGradientDiagnostics(
        source_available=unavailable,
        idle_base_source=unavailable,
        intra_option_source=unavailable,
        parameters_finite=unavailable,
        inputs_finite=unavailable,
        source_indices_valid=unavailable,
        prediction_finite=unavailable,
        target_finite=unavailable,
        td_error_finite=unavailable,
        loss_finite=unavailable,
        gradient_norm_finite=unavailable,
        prediction=zero,
        target=zero,
        td_error=zero,
        loss=zero,
        gradient_norm=zero,
        bootstrap_discount=zero,
        option_terminates=unavailable,
        gradient_finite=unavailable,
        valid=unavailable,
        rejected=jnp.asarray(True, dtype=jnp.bool_),
    )
    return PrototypeBehaviorGradientResult(
        gradient=jnp.zeros((observation_dim,), dtype=jnp.float32),
        valid=unavailable,
        diagnostics=diagnostics,
    )


def _unavailable_representation_gradient_mix(
    observation_dim: int,
) -> RepresentationGradientMixResult:
    """Return an exact-zero, explicitly unavailable two-source mix."""
    zero = jnp.asarray(0.0, dtype=jnp.float32)
    unavailable = jnp.asarray(False, dtype=jnp.bool_)
    diagnostics = RepresentationGradientMixDiagnostics(
        behavior_raw_norm=zero,
        grounded_world_raw_norm=zero,
        behavior_used_norm=zero,
        grounded_world_used_norm=zero,
        behavior_weight=zero,
        grounded_world_weight=zero,
        behavior_effective_weight=zero,
        grounded_world_effective_weight=zero,
        dot_product=zero,
        cosine_similarity=zero,
        conflict=unavailable,
        unclipped_mixed_norm=zero,
        final_mixed_norm=zero,
        behavior_valid=unavailable,
        grounded_world_valid=unavailable,
        behavior_active=unavailable,
        grounded_world_active=unavailable,
        applied=unavailable,
        rejected=jnp.asarray(True, dtype=jnp.bool_),
        zero_output=jnp.asarray(True, dtype=jnp.bool_),
    )
    return RepresentationGradientMixResult(
        gradient=jnp.zeros((observation_dim,), dtype=jnp.float32),
        valid=unavailable,
        applied=unavailable,
        rejected=jnp.asarray(True, dtype=jnp.bool_),
        zero_output=jnp.asarray(True, dtype=jnp.bool_),
        diagnostics=diagnostics,
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


def _strict_bool_array(value: Any, shape: tuple[int, ...], *, name: str) -> Array:
    """Require an exact fixed-shape boolean array."""

    array = jnp.asarray(value)
    if array.dtype != jnp.bool_ or array.shape != shape:
        raise ValueError(f"{name} must have shape {shape} and dtype bool")
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


def _checked_lifetime_words_add(
    words: Array,
    delta: int,
) -> tuple[UInt[Array, " 2"], Bool[Array, ""]]:
    """Advance one uint64 word clock by a small non-negative host delta."""

    if getattr(words, "shape", None) != (2,):
        raise ValueError("lifetime counter words must have shape (2,)")
    if getattr(words, "dtype", None) != jnp.dtype(jnp.uint32):
        raise TypeError("lifetime counter words must have dtype uint32")
    if type(delta) is not int or not 0 <= delta <= 2:
        raise ValueError("lifetime word delta must be one of 0, 1, or 2")
    if delta == 0:
        return jnp.asarray(words, dtype=jnp.uint32), jnp.asarray(True)
    maximum = jnp.asarray(_UINT32_MAX, dtype=jnp.uint32)
    delta_word = jnp.asarray(delta, dtype=jnp.uint32)
    low_capacity = words[1] <= maximum - delta_word
    high_capacity = words[0] < maximum
    capacity_available = low_capacity | high_capacity
    low = words[1] + delta_word
    carry = (low < words[1]).astype(jnp.uint32)
    candidate = jnp.stack((words[0] + carry, low))
    return (
        jnp.where(capacity_available, candidate, words).astype(jnp.uint32),
        capacity_available,
    )


def _lifetime_counter_valid(
    words: Array,
    telemetry: Array,
) -> Bool[Array, ""]:
    """Authenticate one exact uint64 identity against int32 telemetry."""

    if getattr(words, "shape", None) != (2,):
        raise ValueError("lifetime counter words must have shape (2,)")
    if getattr(words, "dtype", None) != jnp.dtype(jnp.uint32):
        raise TypeError("lifetime counter words must have dtype uint32")
    if getattr(telemetry, "shape", None) != ():
        raise ValueError("lifetime counter telemetry must be scalar")
    if getattr(telemetry, "dtype", None) != jnp.dtype(jnp.int32):
        raise TypeError("lifetime counter telemetry must have dtype int32")
    expected = _lifetime_words_to_int32_telemetry(words)
    return (telemetry >= 0) & (telemetry == expected)


def _lifetime_words_to_int32_telemetry(words: Array) -> Int[Array, ""]:
    """Project one exact uint64 identity to saturating scalar telemetry."""

    if getattr(words, "shape", None) != (2,):
        raise ValueError("lifetime counter words must have shape (2,)")
    if getattr(words, "dtype", None) != jnp.dtype(jnp.uint32):
        raise TypeError("lifetime counter words must have dtype uint32")
    maximum_i32 = jnp.asarray(_INT32_MAX, dtype=jnp.int32)
    maximum_u32 = jnp.asarray(_INT32_MAX, dtype=jnp.uint32)
    fits = (words[0] == 0) & (words[1] <= maximum_u32)
    return jnp.where(fits, words[1].astype(jnp.int32), maximum_i32)


def _current_counter_capacity_available(
    state: PrototypeAgentState,
) -> Bool[Array, ""]:
    """Reserve one transition and a possible two-observation boundary."""

    _, step_capacity = _checked_lifetime_words_add(state.step_words, 1)
    _, observation_capacity = _checked_lifetime_words_add(
        state.observation_event_words,
        2,
    )
    return step_capacity & observation_capacity


def _lifetime_words_less_than(left: Array, right: Array) -> Bool[Array, ""]:
    """Lexicographically compare two big-endian uint32 word clocks."""

    return (left[0] < right[0]) | (
        (left[0] == right[0]) & (left[1] < right[1])
    )


def _lifetime_words_less_equal(left: Array, right: Array) -> Bool[Array, ""]:
    """Return ``left <= right`` for exact big-endian word clocks."""

    return _lifetime_words_less_than(left, right) | jnp.all(left == right)


def _prototype_observation_clock_relation_valid(
    step_words: Array,
    observation_words: Array,
) -> Bool[Array, ""]:
    """Authenticate the possible observation history for a transition clock.

    A pristine state owns ``(steps, observations) == (0, 0)``. After start,
    the initial observation plus one or two observations per real transition
    implies ``steps < observations <= 2 * steps + 1``. Once ``steps >= 2**63``
    the mathematical upper bound exceeds uint64, so every representable
    observation clock already satisfies it.
    """

    zero = jnp.zeros((2,), dtype=jnp.uint32)
    pristine = jnp.all(step_words == zero) & jnp.all(observation_words == zero)
    strictly_later = _lifetime_words_less_than(step_words, observation_words)
    high_half = jnp.asarray(2**31, dtype=jnp.uint32)
    upper_unbounded = step_words[0] >= high_half
    doubled_low = step_words[1] + step_words[1]
    carry = (doubled_low < step_words[1]).astype(jnp.uint32)
    doubled = jnp.stack(
        (step_words[0] + step_words[0] + carry, doubled_low)
    )
    upper, upper_capacity = _checked_lifetime_words_add(doubled, 1)
    upper_valid = upper_unbounded | (
        upper_capacity & _lifetime_words_less_equal(observation_words, upper)
    )
    return pristine | (strictly_later & upper_valid)


def _next_counter_capacity_available(
    state: PrototypeAgentState,
    *,
    execution_boundary: Array,
) -> Bool[Array, ""]:
    """Reserve exact capacity for this event and any next boundary event."""

    post_step_words, current_step_capacity = _checked_lifetime_words_add(
        state.step_words,
        1,
    )
    observation_delta = jnp.where(
        execution_boundary,
        jnp.asarray(2, dtype=jnp.int32),
        jnp.asarray(1, dtype=jnp.int32),
    )
    # ``delta`` is static in the helper, so select between both bounded
    # proposals instead of materialising a traced host integer.
    post_observation_one, observation_one_capacity = (
        _checked_lifetime_words_add(state.observation_event_words, 1)
    )
    post_observation_two, observation_two_capacity = (
        _checked_lifetime_words_add(state.observation_event_words, 2)
    )
    post_observation_words = jnp.where(
        observation_delta == 2,
        post_observation_two,
        post_observation_one,
    )
    current_observation_capacity = jnp.where(
        observation_delta == 2,
        observation_two_capacity,
        observation_one_capacity,
    )
    _, next_step_capacity = _checked_lifetime_words_add(post_step_words, 1)
    _, next_observation_capacity = _checked_lifetime_words_add(
        post_observation_words,
        2,
    )
    return (
        current_step_capacity
        & current_observation_capacity
        & next_step_capacity
        & next_observation_capacity
    )


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


def _lifetime_words_modulo(words: Array, interval: int) -> UInt[Array, ""]:
    """Return an exact uint64 clock modulo a positive int without x64 mode."""

    if type(interval) is not int or interval <= 0 or interval > _INT32_MAX:
        raise ValueError("lifetime interval must be in [1, INT32_MAX]")
    if getattr(words, "shape", None) != (2,) or getattr(
        words, "dtype", None
    ) != jnp.dtype(jnp.uint32):
        raise ValueError("lifetime counter words must be uint32 with shape (2,)")
    divisor = jnp.asarray(interval, dtype=jnp.uint32)
    remainder = jnp.asarray(0, dtype=jnp.uint32)
    # ``remainder < interval <= INT32_MAX`` keeps every doubled intermediate
    # below uint32 overflow, including the injected bit.
    for word in (words[0], words[1]):
        for bit_index in range(31, -1, -1):
            bit = (word >> jnp.asarray(bit_index, dtype=jnp.uint32)) & jnp.asarray(
                1,
                dtype=jnp.uint32,
            )
            remainder = jnp.mod(remainder + remainder + bit, divisor)
    return remainder


def _lifetime_words_multiple_of(words: Array, interval: int) -> Bool[Array, ""]:
    """Test an exact uint64 clock modulo a validated positive host interval."""

    return _lifetime_words_modulo(words, interval) == 0


def prototype_lifetime_counter_nbytes() -> int:
    """Return bytes for both Prototype telemetry/exact outer clocks."""

    return PROTOTYPE_LIFETIME_COUNTER_NBYTES


def _prototype_tree_array_resources(tree: object) -> tuple[int, int, int]:
    """Return ``(nbytes, elements, leaves)`` for persistent JAX arrays."""

    arrays = tuple(leaf for leaf in jax.tree.leaves(tree) if isinstance(leaf, Array))
    return (
        sum(int(leaf.size) * int(leaf.dtype.itemsize) for leaf in arrays),
        sum(int(leaf.size) for leaf in arrays),
        len(arrays),
    )


def _prototype_tree_static_contract_matches(value: object, template: object) -> bool:
    """Compare exact PyTree structure plus every persistent array shape/dtype."""

    value_leaves, value_structure = jax.tree.flatten(value)
    template_leaves, template_structure = jax.tree.flatten(template)
    if cast(Any, value_structure) != template_structure or len(value_leaves) != len(
        template_leaves
    ):
        return False
    for value_leaf, template_leaf in zip(
        value_leaves,
        template_leaves,
        strict=True,
    ):
        if isinstance(template_leaf, Array):
            if not isinstance(value_leaf, Array):
                return False
            if (
                value_leaf.shape != template_leaf.shape
                or value_leaf.dtype != template_leaf.dtype
            ):
                return False
        elif type(value_leaf) is not type(template_leaf):
            return False
    return True


def measure_prototype_agent_state_resources(
    state: PrototypeAgentState,
) -> PrototypeAgentStateResourceMeasurement:
    """Measure a concrete Prototype state's persistent JAX-array footprint.

    The result is a top-level ownership partition and therefore sums exactly
    to ``total_nbytes`` without counting nested shared feature/Horde or
    interaction/memory bundles twice.  It measures fixed persistent state,
    not transient compilation buffers, optimizer workspaces, or latency.
    """

    if type(state) is not PrototypeAgentState:
        raise TypeError("state must be an exact PrototypeAgentState")
    bundles = (
        state.oak_state,
        state.world_model_state,
        state.buffer_state,
        state.horde_state,
        state.ia_state,
        state.gru_state,
        state.state_builder_state,
    )
    bundle_resources = tuple(_prototype_tree_array_resources(bundle) for bundle in bundles)
    outer = (
        state.current_raw_observation,
        state.current_representation,
        state.current_action,
        state.current_decision_id,
        state.started,
        state.observation_event_count,
        state.observation_event_words,
        state.step_count,
        state.step_words,
    )
    outer_resources = _prototype_tree_array_resources(outer)
    all_resources = (*bundle_resources, outer_resources)
    nbytes = tuple(resources[0] for resources in all_resources)
    return PrototypeAgentStateResourceMeasurement(
        total_nbytes=sum(nbytes),
        total_array_elements=sum(resources[1] for resources in all_resources),
        total_array_leaves=sum(resources[2] for resources in all_resources),
        oak_bundle_nbytes=nbytes[0],
        world_model_bundle_nbytes=nbytes[1],
        buffer_nbytes=nbytes[2],
        standalone_horde_nbytes=nbytes[3],
        interaction_memory_bundle_nbytes=nbytes[4],
        gru_nbytes=nbytes[5],
        state_builder_feature_bundle_nbytes=nbytes[6],
        outer_nbytes=nbytes[7],
    )


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
        self._option_search_control: OptionSearchControl | None = None
        if config.option_search_control is not None:
            self._option_search_control = OptionSearchControl(
                self._oak.stomp_agent,
                config.option_search_control,
            )
        self._state_builder: StateBuilder[Any] | None = (
            state_builder_from_config(config.state_builder.to_config())
            if config.state_builder is not None
            else None
        )
        self._learning_value_router: LearningValueRouter | None = (
            LearningValueRouter(config.learning_value_router)
            if config.learning_value_router is not None
            else None
        )
        self._prototype_feature_lifecycle: PrototypeFeatureLifecycle | None = None
        if config.prototype_feature_lifecycle is not None:
            self._prototype_feature_lifecycle = PrototypeFeatureLifecycle(
                config.prototype_feature_lifecycle
            )
        self._prototype_feature_horde_digest: Array | None = None
        if (
            config.prototype_feature_lifecycle is not None
            and config.prototype_feature_lifecycle.managed_horde_demons > 0
        ):
            self._prototype_feature_horde_digest = (
                _prototype_feature_horde_schema_digest(config)
            )
        self._prototype_feature_world_model_digest: Array | None = None
        if (
            config.prototype_feature_lifecycle is not None
            and config.world_model is not None
            and config.prototype_atomic_feature_world_memory is None
        ):
            self._prototype_feature_world_model_digest = (
                _prototype_feature_world_model_schema_digest(config)
            )
        self._prototype_atomic_feature_world_memory_digest: Array | None = None
        self._prototype_routed_linear_world_model: (
            PrototypeRoutedLinearWorldModel | None
        ) = None
        if config.prototype_atomic_feature_world_memory is not None:
            feature_config = config.prototype_feature_lifecycle
            world_config = config.world_model
            if feature_config is None or world_config is None:
                raise RuntimeError(
                    "validated atomic feature/world/memory config is incomplete"
                )
            atomic_config = config.prototype_atomic_feature_world_memory
            self._prototype_routed_linear_world_model = (
                PrototypeRoutedLinearWorldModel(
                    PrototypeRoutedLinearWorldConfig(
                        feature_lifecycle=feature_config,
                        world_model=world_config,
                        oak=config.oak,
                        anchor_capacity=atomic_config.anchor_capacity,
                        planning_enabled=atomic_config.planning_enabled,
                        planning_warmup_steps=atomic_config.planning_warmup_steps,
                        max_generation_model_error=(
                            atomic_config.max_generation_model_error
                        ),
                        max_planned_backups=atomic_config.max_planned_backups,
                        carry_survivors=feature_config.carry_survivors,
                    )
                )
            )
            self._prototype_atomic_feature_world_memory_digest = (
                _prototype_atomic_feature_world_memory_schema_digest(config)
            )
        self._prototype_feature_utility: PrototypeFeatureUtilityAuditor | None = None
        self._prototype_feature_utility_digest: Array | None = None
        if config.prototype_feature_utility is not None:
            self._prototype_feature_utility = PrototypeFeatureUtilityAuditor(
                config.prototype_feature_utility
            )
            self._prototype_feature_utility_digest = (
                _prototype_feature_utility_schema_digest(config)
            )
        self._prototype_feature_utility_curation: (
            PrototypeFeatureUtilityCurationPolicy | None
        ) = None
        self._prototype_feature_utility_curation_digest: Array | None = None
        if config.prototype_feature_utility_curation is not None:
            utility_config = config.prototype_feature_utility
            if utility_config is None:
                raise RuntimeError(
                    "feature utility curation requires the utility auditor"
                )
            self._prototype_feature_utility_curation = (
                PrototypeFeatureUtilityCurationPolicy(
                    utility_config,
                    config.prototype_feature_utility_curation,
                )
            )
            self._prototype_feature_utility_curation_digest = (
                _prototype_feature_utility_curation_schema_digest(config)
            )

        self._world_model: ActionConditionedWorldModel | None = None
        self._world_model_ensemble: WorldModelEnsemble | None = None
        self._model_replay_rehearsal: ModelReplayRehearsal | None = None
        self._recurrent_latent_world_model_ensemble: (
            RecurrentLatentWorldModelEnsemble | None
        ) = None
        self._recurrent_signal_estimator: LearningSignalEstimator | None = None
        self._recurrent_signal_nll_offset = 0.0
        self._buffer: RecentObservationBuffer | None = None
        self._dreamer: GuardedDreamer | None = None
        self._world_model_state_contract_template: (
            ActionConditionedWorldModelState | None
        ) = None
        if (
            config.world_model is not None
            and self._prototype_routed_linear_world_model is None
        ):
            self._world_model = ActionConditionedWorldModel(config.world_model)
            self._world_model_state_contract_template = cast(
                ActionConditionedWorldModelState,
                jax.tree.map(
                    lambda leaf: (
                        jnp.asarray(leaf)
                        if isinstance(leaf, (bool, int, float))
                        else leaf
                    ),
                    self._world_model.init(jr.key(0)),
                ),
            )
            self._buffer = RecentObservationBuffer(
                config.buffer_capacity, config.world_model.observation_dim
            )
            if config.n_dreams_per_step > 0:
                self._dreamer = GuardedDreamer(
                    config.dreaming or DreamingConfig()
                )
        elif config.world_model_ensemble is not None:
            self._world_model_ensemble = WorldModelEnsemble(
                config.world_model_ensemble
            )
        elif config.model_replay_rehearsal is not None:
            self._model_replay_rehearsal = ModelReplayRehearsal(
                config.model_replay_rehearsal
            )
        elif config.recurrent_latent_world_model_ensemble is not None:
            recurrent_config = config.recurrent_latent_world_model_ensemble
            self._recurrent_latent_world_model_ensemble = (
                RecurrentLatentWorldModelEnsemble(recurrent_config)
            )
            self._recurrent_signal_nll_offset = _recurrent_nll_estimator_offset(
                recurrent_config
            )
            self._recurrent_signal_estimator = LearningSignalEstimator(
                LearningSignalEstimatorConfig(
                    ensemble_size=recurrent_config.ensemble_size,
                    target_dim=recurrent_config.target_dim,
                    variance_floor=recurrent_config.variance_floor,
                    max_input_magnitude=max(
                        recurrent_config.max_input_magnitude,
                        recurrent_config.max_prediction_magnitude,
                    ),
                    max_predicted_variance=recurrent_config.max_variance,
                    max_observed_loss=(
                        recurrent_config.max_loss_magnitude
                        + self._recurrent_signal_nll_offset
                    ),
                )
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

        self._partner_policy_fusion: PartnerPolicyFusion | None = None
        if config.partner_policy_fusion is not None:
            self._partner_policy_fusion = PartnerPolicyFusion(
                config.partner_policy_fusion
            )

        self._experiential_memory: ExperientialMemory | None = None
        self._experiential_memory_policy: ExperientialMemoryPolicy | None = None
        self._experiential_memory_advantage_gate: (
            ExperientialMemoryAdvantageGate | None
        ) = None
        if config.experiential_memory is not None:
            self._experiential_memory = ExperientialMemory(
                config.experiential_memory
            )
            self._experiential_memory_policy = ExperientialMemoryPolicy(
                self._experiential_memory
            )
        self._prototype_feature_memory: PrototypeFeatureMemory | None = None
        if (
            config.prototype_feature_lifecycle is not None
            and config.experiential_memory is not None
        ):
            if type(config.state_builder) is not IdentityStateBuilderConfig:
                raise RuntimeError(
                    "feature-memory composition requires its validated identity builder"
                )
            self._prototype_feature_memory = PrototypeFeatureMemory(
                PrototypeFeatureMemoryConfig(
                    feature_lifecycle=config.prototype_feature_lifecycle,
                    experiential_memory=config.experiential_memory,
                    base_state_builder=config.state_builder,
                )
            )
            # Keep policy and direct memory operations on the exact substrate
            # authenticated by the feature-memory adapter.
            self._experiential_memory = self._prototype_feature_memory.memory
            self._experiential_memory_policy = ExperientialMemoryPolicy(
                self._experiential_memory
            )
        if config.experiential_memory_advantage_gate is not None:
            if self._experiential_memory is None:
                raise RuntimeError(
                    "experiential-memory advantage gate requires memory"
                )
            self._experiential_memory_advantage_gate = (
                ExperientialMemoryAdvantageGate(
                    self._experiential_memory,
                    config.experiential_memory_advantage_gate,
                )
            )

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
    def option_search_control(self) -> OptionSearchControl | None:
        """Return the opt-in stateless option-model search controller."""

        return self._option_search_control

    @property
    def option_search_control_resource_budget(
        self,
    ) -> OptionSearchControlResourceBudget | None:
        """Return fixed logical work bounds for one option-search call."""

        if self._option_search_control is None:
            return None
        return self._option_search_control.resource_budget

    @property
    def state_builder(self) -> StateBuilder[Any] | None:
        """Canonical state builder, or ``None`` for a legacy raw/GRU path."""
        return self._state_builder

    @property
    def learning_value_router(self) -> LearningValueRouter | None:
        """Return the optional owner-bound typed learning-value router."""

        return self._learning_value_router

    @property
    def learning_value_router_resource_budget(
        self,
    ) -> LearningValueRouterResourceBudget | None:
        """Return the router's exact fixed persistent/work declaration."""

        if self._learning_value_router is None:
            return None
        return self._learning_value_router.resource_budget()

    @property
    def prototype_feature_lifecycle(self) -> PrototypeFeatureLifecycle | None:
        """Return the opt-in bounded pair-feature lifecycle."""

        return self._prototype_feature_lifecycle

    @property
    def prototype_routed_linear_world_model(
        self,
    ) -> PrototypeRoutedLinearWorldModel | None:
        """Return the enabled-only fixed-physical routed world model."""

        return self._prototype_routed_linear_world_model

    @property
    def prototype_feature_utility(
        self,
    ) -> PrototypeFeatureUtilityAuditor | None:
        """Return the opt-in diagnostic actual-consumer utility auditor."""

        return self._prototype_feature_utility

    @property
    def prototype_feature_utility_resource_budget(
        self,
    ) -> PrototypeFeatureUtilityResourceBudget | None:
        """Return the auditor's exact fixed persistent/work declaration."""

        if self._prototype_feature_utility is None:
            return None
        return self._prototype_feature_utility.resource_budget()

    @property
    def prototype_feature_utility_curation(
        self,
    ) -> PrototypeFeatureUtilityCurationPolicy | None:
        """Return the opt-in audit-ranked curation policy."""

        return self._prototype_feature_utility_curation

    @property
    def prototype_feature_utility_curation_resource_budget(
        self,
    ) -> PrototypeFeatureUtilityCurationResourceBudget | None:
        """Return the stateless policy's exact transient-work declaration."""

        if self._prototype_feature_utility_curation is None:
            return None
        return self._prototype_feature_utility_curation.resource_budget()

    def _feature_utility_enabled(self) -> bool:
        """Return whether actual-consumer utility instrumentation is enabled."""

        return self._prototype_feature_utility is not None

    def _feature_utility_curation_enabled(self) -> bool:
        """Return whether v6 audit-ranked curation influence is enabled."""

        return self._prototype_feature_utility_curation is not None

    def _shared_feature_horde_enabled(self) -> bool:
        """Return whether OaK and an exact linear Horde share the bank."""

        feature_config = self._config.prototype_feature_lifecycle
        return (
            feature_config is not None
            and feature_config.managed_horde_demons > 0
        )

    def _representation_component_state(self, slot: Any) -> Any:
        """Unwrap only the opt-in router shell around historical state."""

        if self._learning_value_router is None:
            return slot
        if type(slot) is not PrototypeLearningValueRouterState:
            raise TypeError(
                "state_builder_state must be a "
                "PrototypeLearningValueRouterState when "
                "learning_value_router is configured"
            )
        return slot.representation_state

    def _learning_value_router_component_state(
        self,
        slot: Any,
    ) -> LearningValueRouterState:
        """Return the single router state from its owner-bound shell."""

        if self._learning_value_router is None:
            raise RuntimeError("learning-value router is disabled")
        if type(slot) is not PrototypeLearningValueRouterState:
            raise TypeError(
                "state_builder_state must be a "
                "PrototypeLearningValueRouterState when "
                "learning_value_router is configured"
            )
        return slot.learning_value_router_state

    def _builder_component_state(self, slot: Any) -> Any:
        """Unwrap router and feature shells around the historical builder."""

        slot = self._representation_component_state(slot)
        if self._prototype_feature_lifecycle is None:
            return slot
        if not isinstance(slot, PrototypeFeatureRepresentationState):
            raise TypeError(
                "state_builder_state must be a "
                "PrototypeFeatureRepresentationState when "
                "prototype_feature_lifecycle is configured"
            )
        return slot.builder_state

    def _feature_lifecycle_component_state(
        self,
        slot: Any,
    ) -> PrototypeFeatureLifecycleState:
        """Return persistent feature state from the enabled-only wrapper."""

        if self._prototype_feature_lifecycle is None:
            raise RuntimeError("prototype feature lifecycle is disabled")
        slot = self._representation_component_state(slot)
        if not isinstance(slot, PrototypeFeatureRepresentationState):
            raise TypeError(
                "state_builder_state must be a "
                "PrototypeFeatureRepresentationState when "
                "prototype_feature_lifecycle is configured"
            )
        return slot.feature_lifecycle_state

    def _representation_state_slot(
        self,
        builder_state: Any,
        feature_lifecycle_state: PrototypeFeatureLifecycleState | None,
        learning_value_router_state: LearningValueRouterState | None = None,
    ) -> Any:
        """Compose the builder slot without changing disabled PyTrees."""

        if self._prototype_feature_lifecycle is None:
            representation_state = builder_state
        else:
            if feature_lifecycle_state is None:
                raise RuntimeError(
                    "configured prototype feature lifecycle requires persistent state"
                )
            representation_state = PrototypeFeatureRepresentationState(
                builder_state=builder_state,
                feature_lifecycle_state=feature_lifecycle_state,
            )
        if self._learning_value_router is None:
            if learning_value_router_state is not None:
                raise RuntimeError(
                    "disabled learning-value router cannot own persistent state"
                )
            return representation_state
        if type(learning_value_router_state) is not LearningValueRouterState:
            raise RuntimeError(
                "configured learning-value router requires its exact persistent state"
            )
        return PrototypeLearningValueRouterState(
            representation_state=representation_state,
            learning_value_router_state=learning_value_router_state,
        )

    def _action_world_model_component_state(
        self,
        slot: Any,
    ) -> ActionConditionedWorldModelState:
        """Unwrap the v17 feature/config binding on the stable-base lane."""

        if self._world_model is None:
            raise RuntimeError("legacy action-conditioned world model is disabled")
        if self._prototype_feature_lifecycle is None:
            if type(slot) is not ActionConditionedWorldModelState:
                raise TypeError(
                    "legacy world-model slot must contain an exact model state"
                )
            return slot
        if type(slot) is not PrototypeFeatureWorldModelState:
            raise TypeError(
                "feature/world-model slot must contain its exact v17 wrapper"
            )
        if type(slot.model_state) is not ActionConditionedWorldModelState:
            raise TypeError("v17 wrapper contains the wrong model-state type")
        return slot.model_state

    def _action_world_model_state_slot(
        self,
        model_state: ActionConditionedWorldModelState,
    ) -> ActionConditionedWorldModelState | PrototypeFeatureWorldModelState:
        """Bind stable-base model state to the exact v17 composition config."""

        if type(model_state) is not ActionConditionedWorldModelState:
            raise TypeError("world-model state must be exact")
        if self._prototype_feature_lifecycle is None:
            return model_state
        digest = self._prototype_feature_world_model_digest
        if digest is None:
            raise RuntimeError("feature/world-model composition digest is unavailable")
        return PrototypeFeatureWorldModelState(
            model_state=model_state,
            schema_digest=digest,
        )

    def _atomic_routed_world_component_state(
        self,
        slot: Any,
    ) -> PrototypeRoutedLinearWorldState:
        """Unwrap and authenticate the enabled-only v18 world slot type."""

        if self._prototype_routed_linear_world_model is None:
            raise RuntimeError("atomic routed world composition is disabled")
        if type(slot) is not PrototypeAtomicFeatureWorldMemoryState:
            raise TypeError("v18 world slot has the wrong exact wrapper type")
        if type(slot.world_state) is not PrototypeRoutedLinearWorldState:
            raise TypeError("v18 wrapper contains the wrong routed-world state")
        return slot.world_state

    def _atomic_routed_world_state_slot(
        self,
        world_state: PrototypeRoutedLinearWorldState,
    ) -> PrototypeAtomicFeatureWorldMemoryState:
        """Bind one routed world to the complete v18 Prototype config."""

        if type(world_state) is not PrototypeRoutedLinearWorldState:
            raise TypeError("routed world state must be exact")
        digest = self._prototype_atomic_feature_world_memory_digest
        if digest is None:
            raise RuntimeError("atomic feature/world/memory digest is unavailable")
        return PrototypeAtomicFeatureWorldMemoryState(
            world_state=world_state,
            schema_digest=digest,
        )

    def _oak_component_state(self, slot: Any) -> OaKState:
        """Unwrap OaK only on the feature lane; preserve legacy PyTrees."""

        if self._prototype_feature_lifecycle is None:
            return cast(OaKState, slot)
        consumer: PrototypeFeatureOaKHordeState | PrototypeFeatureOaKState
        if self._shared_feature_horde_enabled():
            consumer = self._shared_feature_horde_bundle(slot)
        else:
            if type(slot) is not PrototypeFeatureOaKState:
                raise TypeError(
                    "oak_state has the wrong enabled feature-consumer wrapper"
                )
            consumer = slot
        if type(consumer.oak_state) is not OaKState:
            raise TypeError("bound feature consumer must contain an exact OaKState")
        return consumer.oak_state

    def _shared_feature_horde_bundle(
        self,
        slot: Any,
    ) -> PrototypeFeatureOaKHordeState:
        """Unwrap exact consumers from the optional v5/v6 shells."""

        if not self._shared_feature_horde_enabled():
            raise RuntimeError("shared feature/Horde composition is disabled")
        consumer = slot
        if self._feature_utility_curation_enabled():
            if type(consumer) is not (
                PrototypeFeatureOaKHordeUtilityCurationState
            ):
                raise TypeError(
                    "feature utility curation requires a "
                    "PrototypeFeatureOaKHordeUtilityCurationState"
                )
            consumer = consumer.utility_state
        if self._feature_utility_enabled():
            if type(consumer) is not PrototypeFeatureOaKHordeUtilityState:
                raise TypeError(
                    "feature utility requires a "
                    "PrototypeFeatureOaKHordeUtilityState"
                )
            consumer = consumer.consumer_state
        if type(consumer) is not PrototypeFeatureOaKHordeState:
            raise TypeError(
                "managed feature Horde requires a PrototypeFeatureOaKHordeState"
            )
        return consumer

    def _feature_utility_component_state(
        self,
        slot: Any,
    ) -> PrototypeFeatureUtilityState:
        """Return the auditor state bound around the shared consumers."""

        if self._prototype_feature_utility is None:
            raise RuntimeError("prototype feature utility is disabled")
        utility_slot = slot
        if self._feature_utility_curation_enabled():
            if type(utility_slot) is not (
                PrototypeFeatureOaKHordeUtilityCurationState
            ):
                raise TypeError(
                    "feature utility curation requires a "
                    "PrototypeFeatureOaKHordeUtilityCurationState"
                )
            utility_slot = utility_slot.utility_state
        if type(utility_slot) is not PrototypeFeatureOaKHordeUtilityState:
            raise TypeError(
                "feature utility requires a PrototypeFeatureOaKHordeUtilityState"
            )
        if type(utility_slot.feature_utility_state) is not (
            PrototypeFeatureUtilityState
        ):
            raise TypeError("feature utility bundle contains the wrong state type")
        return utility_slot.feature_utility_state

    def _feature_consumer_binding(
        self,
        slot: Any,
    ) -> PrototypeFeatureConsumerBinding:
        """Return the identity physically coupled to the enabled OaK subtree."""

        if self._prototype_feature_lifecycle is None:
            raise RuntimeError("prototype feature lifecycle is disabled")
        consumer: PrototypeFeatureOaKHordeState | PrototypeFeatureOaKState
        if self._shared_feature_horde_enabled():
            consumer = self._shared_feature_horde_bundle(slot)
        else:
            if type(slot) is not PrototypeFeatureOaKState:
                raise TypeError(
                    "oak_state has the wrong enabled feature-consumer wrapper"
                )
            consumer = slot
        if type(consumer.consumer_binding) is not PrototypeFeatureConsumerBinding:
            raise TypeError(
                "bound feature consumer must contain an exact consumer binding"
            )
        return consumer.consumer_binding

    def _horde_component_state(self, state: PrototypeAgentState) -> Any:
        """Unwrap Horde from its atomic feature bundle only when managed."""

        if not self._shared_feature_horde_enabled():
            return state.horde_state
        if state.horde_state is not None:
            raise TypeError(
                "managed feature Horde requires the top-level horde_state to be None"
            )
        consumer = self._shared_feature_horde_bundle(state.oak_state)
        if type(consumer.horde_state) is not MultiHeadMLPState:
            raise TypeError("managed feature Horde must contain an exact linear state")
        return consumer.horde_state

    def _feature_horde_bundle_identity_valid(
        self,
        state: PrototypeAgentState,
    ) -> Bool[Array, ""]:
        """Validate atomic placement and the ordered static schema digest."""

        if not self._shared_feature_horde_enabled():
            return jnp.asarray(True, dtype=jnp.bool_)
        slot = state.oak_state
        curation_identity_valid = jnp.asarray(True, dtype=jnp.bool_)
        if self._feature_utility_curation_enabled():
            curation_digest = self._prototype_feature_utility_curation_digest
            if (
                type(slot) is not PrototypeFeatureOaKHordeUtilityCurationState
                or curation_digest is None
                or not hasattr(slot.schema_digest, "shape")
                or slot.schema_digest.shape != (32,)
                or slot.schema_digest.dtype != jnp.uint8
            ):
                return jnp.asarray(False, dtype=jnp.bool_)
            curation_identity_valid = jnp.array_equal(
                slot.schema_digest,
                curation_digest,
            )
            slot = slot.utility_state
        utility_identity_valid = jnp.asarray(True, dtype=jnp.bool_)
        if self._feature_utility_enabled():
            utility_digest = self._prototype_feature_utility_digest
            if (
                type(slot) is not PrototypeFeatureOaKHordeUtilityState
                or utility_digest is None
                or not hasattr(slot.schema_digest, "shape")
                or slot.schema_digest.shape != (32,)
                or slot.schema_digest.dtype != jnp.uint8
            ):
                return jnp.asarray(False, dtype=jnp.bool_)
            utility_identity_valid = jnp.array_equal(
                slot.schema_digest,
                utility_digest,
            )
            slot = slot.consumer_state
        digest = self._prototype_feature_horde_digest
        if (
            type(slot) is not PrototypeFeatureOaKHordeState
            or state.horde_state is not None
            or type(slot.horde_state) is not MultiHeadMLPState
            or digest is None
            or not hasattr(slot.schema_digest, "shape")
            or slot.schema_digest.shape != (32,)
            or slot.schema_digest.dtype != jnp.uint8
        ):
            return jnp.asarray(False, dtype=jnp.bool_)
        return (
            curation_identity_valid
            & utility_identity_valid
            & jnp.array_equal(slot.schema_digest, digest)
        )

    def _managed_horde_optimizer_matches_config(
        self,
        horde_state: Any,
    ) -> Bool[Array, ""]:
        """Bind every linear-head LMS scalar to the serialized step size."""

        feature_config = self._config.prototype_feature_lifecycle
        if (
            feature_config is None
            or type(horde_state) is not MultiHeadMLPState
            or type(horde_state.head_optimizer_states) is not tuple
            or len(horde_state.head_optimizer_states)
            != feature_config.managed_horde_demons
        ):
            return jnp.asarray(False, dtype=jnp.bool_)
        expected = jnp.asarray(
            self._config.horde_step_size,
            dtype=jnp.float32,
        )
        valid = jnp.asarray(True, dtype=jnp.bool_)
        for optimizer_pair in horde_state.head_optimizer_states:
            if (
                type(optimizer_pair) is not tuple
                or len(optimizer_pair) != 2
                or any(type(item) is not LMSState for item in optimizer_pair)
            ):
                return jnp.asarray(False, dtype=jnp.bool_)
            for optimizer_state in optimizer_pair:
                valid = valid & jnp.array_equal(
                    optimizer_state.step_size,
                    expected,
                )
        return valid

    def _action_world_model_optimizer_matches_config(
        self,
        learner_state: Any,
    ) -> Bool[Array, ""]:
        """Bind every action-world-model LMS scalar to its serialized config."""

        world_config = self._config.world_model
        if (
            world_config is None
            or type(learner_state) is not MultiHeadMLPState
            or type(learner_state.trunk_optimizer_states) is not tuple
            or type(learner_state.head_optimizer_states) is not tuple
        ):
            return jnp.asarray(False, dtype=jnp.bool_)
        expected = jnp.asarray(world_config.step_size, dtype=jnp.float32)
        valid = jnp.asarray(True, dtype=jnp.bool_)
        for optimizer_state in learner_state.trunk_optimizer_states:
            if type(optimizer_state) is not LMSState:
                return jnp.asarray(False, dtype=jnp.bool_)
            valid = valid & jnp.array_equal(optimizer_state.step_size, expected)
        for optimizer_pair in learner_state.head_optimizer_states:
            if (
                type(optimizer_pair) is not tuple
                or len(optimizer_pair) != 2
                or any(type(item) is not LMSState for item in optimizer_pair)
            ):
                return jnp.asarray(False, dtype=jnp.bool_)
            for optimizer_state in optimizer_pair:
                valid = valid & jnp.array_equal(
                    optimizer_state.step_size,
                    expected,
                )
        return valid

    def _oak_state_slot(
        self,
        oak_state: OaKState,
        consumer_binding: PrototypeFeatureConsumerBinding | None,
        horde_state: Any = None,
        feature_utility_state: PrototypeFeatureUtilityState | None = None,
    ) -> Any:
        """Compose the enabled-only bound OaK subtree."""

        if self._prototype_feature_lifecycle is None:
            return oak_state
        if type(oak_state) is not OaKState:
            raise TypeError("feature consumer must be an exact OaKState")
        if type(consumer_binding) is not PrototypeFeatureConsumerBinding:
            raise TypeError(
                "configured prototype feature lifecycle requires a consumer binding"
            )
        if self._shared_feature_horde_enabled():
            digest = self._prototype_feature_horde_digest
            if type(horde_state) is not MultiHeadMLPState or digest is None:
                raise TypeError(
                    "managed feature lifecycle requires an exact Horde state and digest"
                )
            consumer_state = PrototypeFeatureOaKHordeState(
                oak_state=oak_state,
                horde_state=horde_state,
                consumer_binding=consumer_binding,
                schema_digest=digest,
            )
            if self._feature_utility_enabled():
                utility_digest = self._prototype_feature_utility_digest
                if (
                    type(feature_utility_state) is not PrototypeFeatureUtilityState
                    or utility_digest is None
                ):
                    raise TypeError(
                        "feature utility requires its exact state and schema digest"
                    )
                utility_state = PrototypeFeatureOaKHordeUtilityState(
                    consumer_state=consumer_state,
                    feature_utility_state=feature_utility_state,
                    schema_digest=utility_digest,
                )
                if self._feature_utility_curation_enabled():
                    curation_digest = (
                        self._prototype_feature_utility_curation_digest
                    )
                    if curation_digest is None:
                        raise TypeError(
                            "feature utility curation requires its schema digest"
                        )
                    return PrototypeFeatureOaKHordeUtilityCurationState(
                        utility_state=utility_state,
                        schema_digest=curation_digest,
                    )
                return utility_state
            if feature_utility_state is not None:
                raise TypeError("disabled feature utility cannot contain audit state")
            return consumer_state
        if horde_state is not None:
            raise TypeError("feature-only OaK wrapper cannot contain Horde state")
        if feature_utility_state is not None:
            raise TypeError("feature-only OaK wrapper cannot contain audit state")
        return PrototypeFeatureOaKState(
            oak_state=oak_state,
            consumer_binding=consumer_binding,
        )

    @property
    def partner_policy_fusion(self) -> PartnerPolicyFusion | None:
        """Return the opt-in bounded partner-message policy fusion mechanism."""

        return self._partner_policy_fusion

    @property
    def experiential_memory(self) -> ExperientialMemory | None:
        """Return the opt-in fixed-capacity experiential memory."""

        return self._experiential_memory

    @property
    def experiential_memory_policy(self) -> ExperientialMemoryPolicy | None:
        """Return the read-only categorical memory proposal boundary."""

        return self._experiential_memory_policy

    @property
    def experiential_memory_advantage_gate(
        self,
    ) -> ExperientialMemoryAdvantageGate | None:
        """Return the optional conservative memory dispatch-authority gate."""

        return self._experiential_memory_advantage_gate

    @property
    def prototype_feature_memory(self) -> PrototypeFeatureMemory | None:
        """Return the exact pair-bank memory adapter when configured."""

        return self._prototype_feature_memory

    @property
    def prototype_feature_memory_resource_budget(
        self,
    ) -> PrototypeFeatureMemoryResourceBudget | None:
        """Return fixed wrapper bytes and worst-case re-encoding work."""

        if self._prototype_feature_memory is None:
            return None
        return self._prototype_feature_memory.resource_budget()

    def prototype_atomic_feature_world_memory_resource_budget(
        self,
        state: PrototypeAgentState,
    ) -> PrototypeAtomicFeatureWorldMemoryResourceBudget:
        """Measure one enabled state and declare exact transition call maxima."""

        if self._prototype_routed_linear_world_model is None:
            raise RuntimeError("atomic feature/world/memory composition is disabled")
        if self._prototype_feature_memory is None or self._horde is None:
            raise RuntimeError("atomic composition components are unavailable")
        if type(state) is not PrototypeAgentState:
            raise TypeError("state must be an exact PrototypeAgentState")
        if not bool(jax.device_get(self._checkpoint_state_valid(state))):
            raise ValueError("atomic resource measurement requires a valid state")
        oak_binding = self._feature_consumer_binding(state.oak_state)
        world_binding = self._atomic_routed_world_component_state(
            state.world_model_state
        ).consumer_binding
        memory_binding = self._feature_memory_component_state(
            state.ia_state
        ).consumer_binding
        mirrored_nbytes = sum(
            _prototype_tree_array_resources(binding)[0]
            for binding in (oak_binding, world_binding, memory_binding)
        )
        memory_declaration = self.experiential_memory_resource_declaration
        if memory_declaration is None:
            raise RuntimeError("atomic composition memory declaration is unavailable")
        return PrototypeAtomicFeatureWorldMemoryResourceBudget(
            mechanism_status=(
                PROTOTYPE_ATOMIC_FEATURE_WORLD_MEMORY_MECHANISM_STATUS
            ),
            scientific_promotion_allowed=(
                PROTOTYPE_ATOMIC_FEATURE_WORLD_MEMORY_SCIENTIFIC_PROMOTION_ALLOWED
            ),
            persistent_state_nbytes=(
                measure_prototype_agent_state_resources(state).total_nbytes
            ),
            persistent_capacity_growth=0,
            lifecycle_authority_count=1,
            router_authority_count=1,
            oak_consumer_count=1,
            ordered_linear_horde_count=1,
            routed_world_count=1,
            world_model_buffer_count=1,
            experiential_memory_count=1,
            mirrored_binding_cache_count=3,
            mirrored_binding_cache_nbytes=mirrored_nbytes,
            oak_update_evaluations_per_transition=1,
            horde_update_evaluations_per_transition=1,
            feature_learner_update_evaluations_per_transition=1,
            lifecycle_router_evaluations_per_transition=2,
            world_learner_update_evaluations_per_transition=1,
            world_router_evaluations_per_transition=1,
            memory_rebind_evaluations_per_transition=1,
            memory_step_evaluations_per_transition=1,
            deterministic_prestate_memory_queries_per_transition=(
                memory_declaration.total_deterministic_prestate_queries
            ),
            memory_writes_attempted_per_transition=(
                memory_declaration.writes_attempted
            ),
        )

    @property
    def experiential_memory_resource_declaration(
        self,
    ) -> PrototypeExperientialMemoryResourceDeclaration | None:
        """Declare memory bytes and both deterministic pre-state queries."""

        policy = self._experiential_memory_policy
        if policy is None:
            return None
        policy_resources = policy.resource_declaration()
        categorical_queries = policy_resources.memory_queries_per_proposal
        causal_queries = 1
        return PrototypeExperientialMemoryResourceDeclaration(
            persistent_state_bytes=(
                policy_resources.external_memory_persistent_state_bytes
            ),
            categorical_policy_queries=categorical_queries,
            causal_step_queries=causal_queries,
            total_deterministic_prestate_queries=(
                categorical_queries + causal_queries
            ),
            writes_attempted=1,
            random_draws=policy_resources.random_draws_per_proposal,
            score_mass_values_interpreted=(
                policy_resources.score_mass_values_interpreted_per_proposal
            ),
            hard_safety_values_interpreted=(
                policy_resources.hard_safety_values_interpreted_per_proposal
            ),
        )

    def _interaction_without_memory(self, slot: Any) -> Any:
        """Unwrap only the opt-in outer memory state composition."""

        if self._experiential_memory is None:
            return slot
        if not isinstance(slot, PrototypeMemoryInteractionState):
            raise TypeError(
                "ia_state must be a PrototypeMemoryInteractionState when "
                "experiential memory is configured"
            )
        return slot.interaction_state

    def _ia_component_state(self, slot: Any) -> Any:
        """Unwrap IA state while preserving historical IA-only slot shapes."""

        slot = self._interaction_without_memory(slot)
        if self._partner_policy_fusion is None:
            return slot
        if not isinstance(slot, PrototypeInteractionState):
            raise TypeError(
                "ia_state must be a PrototypeInteractionState when partner "
                "policy fusion is configured"
            )
        return slot.ia_state

    def _partner_fusion_component_state(
        self,
        slot: Any,
    ) -> PartnerPolicyFusionState:
        """Return the partner state from the opt-in IA-slot wrapper."""

        if self._partner_policy_fusion is None:
            raise RuntimeError("partner policy fusion is disabled")
        slot = self._interaction_without_memory(slot)
        if not isinstance(slot, PrototypeInteractionState):
            raise TypeError(
                "ia_state must be a PrototypeInteractionState when partner "
                "policy fusion is configured"
            )
        return slot.partner_policy_fusion_state

    def _partner_interaction_state(
        self,
        slot: Any,
    ) -> PrototypeInteractionState:
        """Return the complete partner wrapper below optional memory."""

        if self._partner_policy_fusion is None:
            raise RuntimeError("partner policy fusion is disabled")
        inner = self._interaction_without_memory(slot)
        if not isinstance(inner, PrototypeInteractionState):
            raise TypeError(
                "ia_state must contain a PrototypeInteractionState when "
                "partner policy fusion is configured"
            )
        return inner

    def _experiential_memory_component_state(
        self,
        slot: Any,
    ) -> ExperientialMemoryState:
        """Return the generic memory substrate from the opt-in wrapper."""

        memory_slot = self._memory_slot_component_state(slot)
        if self._prototype_feature_memory is not None:
            return cast(PrototypeFeatureMemoryState, memory_slot).memory_state
        return cast(ExperientialMemoryState, memory_slot)

    def _memory_slot_component_state(self, slot: Any) -> Any:
        """Return the raw or exact feature-bound payload stored in memory slot."""

        if self._experiential_memory is None:
            raise RuntimeError("experiential memory is disabled")
        if not isinstance(slot, PrototypeMemoryInteractionState):
            raise TypeError(
                "ia_state must be a PrototypeMemoryInteractionState when "
                "experiential memory is configured"
            )
        payload = slot.experiential_memory_state
        if self._prototype_feature_memory is not None:
            if type(payload) is not PrototypeFeatureMemoryState:
                raise TypeError(
                    "feature-lifecycle memory slot must contain an exact "
                    "PrototypeFeatureMemoryState"
                )
        elif type(payload) is not ExperientialMemoryState:
            raise TypeError(
                "legacy memory slot must contain an exact ExperientialMemoryState"
            )
        return payload

    def _feature_memory_component_state(
        self,
        slot: Any,
    ) -> PrototypeFeatureMemoryState:
        """Return the exact bank-bound memory wrapper."""

        if self._prototype_feature_memory is None:
            raise RuntimeError("prototype feature memory is disabled")
        return cast(
            PrototypeFeatureMemoryState,
            self._memory_slot_component_state(slot),
        )

    def _interaction_slot(
        self,
        ia_state: Any,
        partner_state: Any,
        *,
        experiential_memory_state: Any = None,
        feedback_prototype_decision_id: Array | None = None,
        feedback_prototype_decision_id_available: Array | None = None,
    ) -> Any:
        """Compose only configured interaction lanes into the existing slot."""

        interaction_state = ia_state
        if self._partner_policy_fusion is not None:
            owner = (
                jnp.zeros((4,), dtype=jnp.uint32)
                if feedback_prototype_decision_id is None
                else jnp.asarray(feedback_prototype_decision_id, dtype=jnp.uint32)
            )
            owner_available = (
                jnp.asarray(False, dtype=jnp.bool_)
                if feedback_prototype_decision_id_available is None
                else jnp.asarray(
                    feedback_prototype_decision_id_available,
                    dtype=jnp.bool_,
                )
            )
            interaction_state = PrototypeInteractionState(
                ia_state=ia_state,
                partner_policy_fusion_state=cast(
                    PartnerPolicyFusionState,
                    partner_state,
                ),
                feedback_prototype_decision_id=owner,
                feedback_prototype_decision_id_available=owner_available,
            )
        if self._experiential_memory is None:
            return interaction_state
        if experiential_memory_state is None:
            raise RuntimeError(
                "configured experiential memory requires a persistent state"
            )
        if self._prototype_feature_memory is not None:
            if type(experiential_memory_state) is not PrototypeFeatureMemoryState:
                raise TypeError(
                    "feature-lifecycle memory composition requires its exact wrapper"
                )
        elif type(experiential_memory_state) is not ExperientialMemoryState:
            raise TypeError(
                "memory composition requires an exact ExperientialMemoryState"
            )
        return PrototypeMemoryInteractionState(
            interaction_state=interaction_state,
            experiential_memory_state=experiential_memory_state,
        )

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

    @property
    def recurrent_latent_world_model_ensemble(
        self,
    ) -> RecurrentLatentWorldModelEnsemble | None:
        """Return the recurrent latent ensemble on its opt-in lane."""
        return self._recurrent_latent_world_model_ensemble

    @staticmethod
    def _gate_recurrent_decision_cache(
        cache: RecurrentLatentDecisionCache,
        armed: Array,
    ) -> RecurrentLatentDecisionCache:
        """Disarm a speculative cache without changing its fixed PyTree shape."""
        available = jnp.asarray(armed, dtype=jnp.bool_) & cache.valid
        prediction = cache.prediction
        prediction_availability = RecurrentLatentPredictionAvailability(
            prediction=prediction.availability.prediction & available,
            epistemic=prediction.availability.epistemic & available,
            aleatoric=prediction.availability.aleatoric & available,
        )
        gated_prediction = prediction.replace(
            warmup_ready=prediction.warmup_ready & available,
            availability=prediction_availability,
            valid=prediction.valid & available,
        )
        return cast(
            RecurrentLatentDecisionCache,
            cache.replace(
                prediction=gated_prediction,
                valid=available,
            ),
        )

    def _recurrent_decide_from_start(
        self,
        model_state: RecurrentLatentWorldModelEnsembleState,
        start_cache: RecurrentLatentStartCache,
        action: Array,
        armed: Array,
    ) -> RecurrentLatentDecisionCache:
        """Bind one exact selected primitive action without advancing state."""
        model = self._recurrent_latent_world_model_ensemble
        if model is None:
            raise RuntimeError("recurrent latent world-model lane is disabled")
        safe_action = jnp.where(
            armed,
            action,
            jnp.asarray(0, dtype=jnp.int32),
        )
        return self._gate_recurrent_decision_cache(
            model.decide(model_state, start_cache, safe_action),
            armed,
        )

    def _recurrent_decision_for_observation(
        self,
        model_state: RecurrentLatentWorldModelEnsembleState,
        observation: Array,
        action: Array,
        armed: Array,
    ) -> RecurrentLatentDecisionCache:
        """Capture an observation owner and bind its exact selected action."""
        model = self._recurrent_latent_world_model_ensemble
        if model is None:
            raise RuntimeError("recurrent latent world-model lane is disabled")
        start_cache = model.start(model_state, observation)
        return self._recurrent_decide_from_start(
            model_state,
            start_cache,
            action,
            armed,
        )

    def _recurrent_signal_state_valid(
        self,
        state: LearningSignalEstimatorState,
    ) -> Array:
        """Validate the derived causal signal state and its configured bounds."""
        estimator = self._recurrent_signal_estimator
        if estimator is None:
            raise RuntimeError("recurrent signal estimator is disabled")
        LearningSignalEstimator._validate_state_shapes(state)
        config = estimator.config
        maximum = jnp.asarray(_INT32_MAX, dtype=jnp.int32)
        safe_valid_count = jnp.clip(state.valid_count, 0, maximum)
        safe_invalid_count = jnp.clip(state.invalid_count, 0, maximum)
        counters_sum_without_overflow = safe_valid_count <= maximum - safe_invalid_count
        safe_counter_sum = safe_valid_count + jnp.minimum(
            safe_invalid_count,
            maximum - safe_valid_count,
        )
        return (
            (state.step_count >= 0)
            & (state.valid_count >= 0)
            & (state.invalid_count >= 0)
            & counters_sum_without_overflow
            & (state.step_count == safe_counter_sum)
            & (state.calibration_count >= 0)
            & (state.calibration_count <= config.change_calibration_steps)
            & (state.calibration_count <= state.valid_count)
            & jnp.isfinite(state.calibration_mean)
            & (state.calibration_mean >= 0.0)
            & (state.calibration_mean <= config.max_normalized_residual)
            & jnp.isfinite(state.calibration_m2)
            & (state.calibration_m2 >= 0.0)
            & jnp.isfinite(state.fast_loss_ema)
            & (state.fast_loss_ema >= 0.0)
            & (state.fast_loss_ema <= config.max_observed_loss)
            & jnp.isfinite(state.slow_loss_ema)
            & (state.slow_loss_ema >= 0.0)
            & (state.slow_loss_ema <= config.max_observed_loss)
            & jnp.isfinite(state.sustained_change_probability)
            & (state.sustained_change_probability >= 0.0)
            & (state.sustained_change_probability <= 1.0)
        )

    def _recurrent_wrapper_numeric_valid(
        self,
        wrapper: PrototypeRecurrentLatentWorldModelState,
    ) -> Array:
        """Validate model, cache, estimator, and shared causal counters."""
        model = self._recurrent_latent_world_model_ensemble
        if model is None:
            raise RuntimeError("recurrent latent world-model lane is disabled")
        if not isinstance(wrapper, PrototypeRecurrentLatentWorldModelState):
            raise TypeError(
                "world_model_state must be a PrototypeRecurrentLatentWorldModelState"
            )
        model._validate_decision_static(wrapper.decision_cache)
        cache = wrapper.decision_cache
        prediction = cache.prediction
        cache_valid = (
            _floating_tree_is_finite(cache)
            & (cache.owner_event_count >= 0)
            & (cache.owner_event_count <= model.config.max_updates)
            & jnp.all(jnp.abs(cache.owner_hidden_states) <= 1.0 + 1.0e-6)
            & jnp.all(
                jnp.abs(cache.observation) <= model.config.max_input_magnitude
            )
            & (cache.action >= 0)
            & (cache.action < model.config.n_actions)
            & jnp.where(
                cache.valid,
                prediction.valid
                & prediction.availability.prediction
                & (
                    prediction.availability.epistemic
                    == (prediction.warmup_ready & prediction.valid)
                )
                & (
                    prediction.availability.aleatoric
                    == (prediction.warmup_ready & prediction.valid)
                ),
                (~prediction.valid)
                & (~prediction.availability.prediction)
                & (~prediction.availability.epistemic)
                & (~prediction.availability.aleatoric),
            )
        )
        return (
            model.state_valid(wrapper.model_state)
            & cache_valid
            & self._recurrent_signal_state_valid(wrapper.signal_state)
            & (wrapper.signal_state.step_count == wrapper.model_state.event_count)
        )

    def _recurrent_armed_cache_consistent(
        self,
        state: PrototypeAgentState,
    ) -> Array:
        """Recompute and exactly verify the current decision-owned prediction."""
        wrapper = cast(
            PrototypeRecurrentLatentWorldModelState,
            state.world_model_state,
        )
        expected = self._recurrent_decision_for_observation(
            wrapper.model_state,
            state.current_representation,
            state.current_action,
            jnp.asarray(True, dtype=jnp.bool_),
        )
        return wrapper.decision_cache.valid & _tree_arrays_equal(
            wrapper.decision_cache,
            expected,
        )

    def _state_builder_parameter_count(self) -> int:
        """Return the configured learnable builder vector width."""
        builder_config = self._config.state_builder
        if not isinstance(builder_config, OnlineGatedStateBuilderConfig):
            raise RuntimeError("online state-builder learning is not configured")
        return builder_config.parameter_count()

    def _missing_candidate_update_audit_evidence(
        self,
        decision_id: Array,
    ) -> PrototypeCandidateUpdateAuditEvidence:
        """Construct explicit unavailable evidence for a fail-closed audit."""
        zeros = jnp.zeros(
            (self._state_builder_parameter_count(),),
            dtype=jnp.float32,
        )
        zero = jnp.asarray(0.0, dtype=jnp.float32)
        unavailable = jnp.asarray(False, dtype=jnp.bool_)
        return PrototypeCandidateUpdateAuditEvidence(
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

    def _normalize_candidate_update_audit_evidence(
        self,
        state: PrototypeAgentState,
        evidence: PrototypeCandidateUpdateAuditEvidence | None,
    ) -> tuple[PrototypeCandidateUpdateAuditEvidence | None, Array, Array]:
        """Validate one optimizer-control sidecar without owning the transition.

        Static shape or dtype drift raises. Runtime non-finiteness and a stale
        decision binding clear the corresponding availability flags so the
        candidate-update audit rejects, while the authoritative environment
        transition remains eligible for normal control/model learning.
        """
        if self._config.gradient_joy is None:
            if evidence is not None:
                raise ValueError(
                    "candidate_update_audit_evidence requires the candidate-update "
                    "audit to be configured (legacy config field gradient_joy)"
                )
            unavailable = jnp.asarray(False, dtype=jnp.bool_)
            return None, unavailable, unavailable

        if evidence is None:
            unavailable = jnp.asarray(False, dtype=jnp.bool_)
            return (
                self._missing_candidate_update_audit_evidence(state.current_decision_id),
                unavailable,
                unavailable,
            )
        if not isinstance(evidence, PrototypeCandidateUpdateAuditEvidence):
            raise TypeError(
                "candidate_update_audit_evidence must be "
                "PrototypeCandidateUpdateAuditEvidence"
            )

        parameter_shape = (self._state_builder_parameter_count(),)
        decision_id = _strict_decision_id(
            evidence.decision_id,
            name="candidate_update_audit_evidence.decision_id",
        )
        decision_matches = jnp.array_equal(
            decision_id,
            state.current_decision_id,
        )
        gradients = {
            "objective_probe_gradient": _strict_float32_array(
                evidence.objective_probe_gradient,
                parameter_shape,
                name="candidate_update_audit_evidence.objective_probe_gradient",
            ),
            "retention_probe_gradient": _strict_float32_array(
                evidence.retention_probe_gradient,
                parameter_shape,
                name="candidate_update_audit_evidence.retention_probe_gradient",
            ),
            "safety_cost_gradient": _strict_float32_array(
                evidence.safety_cost_gradient,
                parameter_shape,
                name="candidate_update_audit_evidence.safety_cost_gradient",
            ),
        }
        scalars = {
            "advantage": _strict_float32_array(
                evidence.advantage,
                (),
                name="candidate_update_audit_evidence.advantage",
            ),
            "action_surprisal": _strict_float32_array(
                evidence.action_surprisal,
                (),
                name="candidate_update_audit_evidence.action_surprisal",
            ),
            "safety_cost": _strict_float32_array(
                evidence.safety_cost,
                (),
                name="candidate_update_audit_evidence.safety_cost",
            ),
        }
        flags = {
            "objective_probe_available": _strict_bool_scalar(
                evidence.objective_probe_available,
                name="candidate_update_audit_evidence.objective_probe_available",
            ),
            "retention_probe_available": _strict_bool_scalar(
                evidence.retention_probe_available,
                name="candidate_update_audit_evidence.retention_probe_available",
            ),
            "safety_probe_available": _strict_bool_scalar(
                evidence.safety_probe_available,
                name="candidate_update_audit_evidence.safety_probe_available",
            ),
            "probe_independence_attested": _strict_bool_scalar(
                evidence.probe_independence_attested,
                name="candidate_update_audit_evidence.probe_independence_attested",
            ),
            "advantage_available": _strict_bool_scalar(
                evidence.advantage_available,
                name="candidate_update_audit_evidence.advantage_available",
            ),
            "action_surprisal_available": _strict_bool_scalar(
                evidence.action_surprisal_available,
                name="candidate_update_audit_evidence.action_surprisal_available",
            ),
            "safety_cost_available": _strict_bool_scalar(
                evidence.safety_cost_available,
                name="candidate_update_audit_evidence.safety_cost_available",
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
        normalized = PrototypeCandidateUpdateAuditEvidence(
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

    def _missing_experiential_memory_input(
        self,
    ) -> PrototypeExperientialMemoryInput:
        """Return a fixed-shape unavailable memory transaction sidecar."""

        if self._experiential_memory is None:
            raise RuntimeError("experiential memory is disabled")
        unavailable = jnp.asarray(False, dtype=jnp.bool_)
        missing = jnp.asarray(-1, dtype=jnp.int32)
        return PrototypeExperientialMemoryInput(
            available=unavailable,
            current_prototype_decision_id=jnp.zeros((4,), dtype=jnp.uint32),
            next_prototype_decision_id=jnp.zeros((4,), dtype=jnp.uint32),
            query_representation_version=missing,
            entry_representation_version=missing,
            query_uncertainty=jnp.asarray(0.0, dtype=jnp.float32),
            query_uncertainty_available=unavailable,
            entry_uncertainty=jnp.asarray(0.0, dtype=jnp.float32),
            entry_uncertainty_available=unavailable,
            safety_cost=jnp.asarray(0.0, dtype=jnp.float32),
            safety_cost_available=unavailable,
            reliability=jnp.asarray(0.0, dtype=jnp.float32),
            utility=jnp.asarray(0.0, dtype=jnp.float32),
            utility_available=unavailable,
            provenance_id=missing,
            source_id=missing,
            next_action_safety_mask=jnp.ones(
                (self._config.oak.n_primitive_actions,),
                dtype=jnp.bool_,
            ),
        )

    def _normalize_experiential_memory_input(
        self,
        sidecar: PrototypeExperientialMemoryInput | None,
    ) -> tuple[PrototypeExperientialMemoryInput | None, Array]:
        """Normalize static memory metadata while leaving dynamic gates traced."""

        if self._experiential_memory is None:
            if sidecar is not None:
                raise ValueError(
                    "experiential_memory_input requires experiential memory"
                )
            return None, jnp.asarray(False, dtype=jnp.bool_)
        if sidecar is None:
            return (
                self._missing_experiential_memory_input(),
                jnp.asarray(False, dtype=jnp.bool_),
            )
        if not isinstance(sidecar, PrototypeExperientialMemoryInput):
            raise TypeError(
                "experiential_memory_input must be "
                "PrototypeExperientialMemoryInput"
            )
        mask = jnp.asarray(sidecar.next_action_safety_mask)
        expected_mask_shape = (self._config.oak.n_primitive_actions,)
        if mask.shape != expected_mask_shape or mask.dtype != jnp.bool_:
            raise ValueError(
                "experiential_memory_input.next_action_safety_mask must have "
                f"shape {expected_mask_shape} and dtype bool"
            )
        normalized = PrototypeExperientialMemoryInput(
            available=_strict_bool_scalar(
                sidecar.available,
                name="experiential_memory_input.available",
            ),
            current_prototype_decision_id=_strict_decision_id(
                sidecar.current_prototype_decision_id,
                name=(
                    "experiential_memory_input."
                    "current_prototype_decision_id"
                ),
            ),
            next_prototype_decision_id=_strict_decision_id(
                sidecar.next_prototype_decision_id,
                name="experiential_memory_input.next_prototype_decision_id",
            ),
            query_representation_version=_strict_action_scalar(
                sidecar.query_representation_version,
                name="experiential_memory_input.query_representation_version",
            ),
            entry_representation_version=_strict_action_scalar(
                sidecar.entry_representation_version,
                name="experiential_memory_input.entry_representation_version",
            ),
            query_uncertainty=_strict_float32_array(
                sidecar.query_uncertainty,
                (),
                name="experiential_memory_input.query_uncertainty",
            ),
            query_uncertainty_available=_strict_bool_scalar(
                sidecar.query_uncertainty_available,
                name="experiential_memory_input.query_uncertainty_available",
            ),
            entry_uncertainty=_strict_float32_array(
                sidecar.entry_uncertainty,
                (),
                name="experiential_memory_input.entry_uncertainty",
            ),
            entry_uncertainty_available=_strict_bool_scalar(
                sidecar.entry_uncertainty_available,
                name="experiential_memory_input.entry_uncertainty_available",
            ),
            safety_cost=_strict_float32_array(
                sidecar.safety_cost,
                (),
                name="experiential_memory_input.safety_cost",
            ),
            safety_cost_available=_strict_bool_scalar(
                sidecar.safety_cost_available,
                name="experiential_memory_input.safety_cost_available",
            ),
            reliability=_strict_float32_array(
                sidecar.reliability,
                (),
                name="experiential_memory_input.reliability",
            ),
            utility=_strict_float32_array(
                sidecar.utility,
                (),
                name="experiential_memory_input.utility",
            ),
            utility_available=_strict_bool_scalar(
                sidecar.utility_available,
                name="experiential_memory_input.utility_available",
            ),
            provenance_id=_strict_action_scalar(
                sidecar.provenance_id,
                name="experiential_memory_input.provenance_id",
            ),
            source_id=_strict_action_scalar(
                sidecar.source_id,
                name="experiential_memory_input.source_id",
            ),
            next_action_safety_mask=jnp.asarray(mask, dtype=jnp.bool_),
        )
        return normalized, jnp.asarray(True, dtype=jnp.bool_)

    def _missing_partner_policy_fusion_input(
        self,
    ) -> PrototypePartnerPolicyFusionInput:
        """Return the fixed-shape sidecar used for explicit base fallback."""

        fusion = self._partner_policy_fusion
        if fusion is None:
            raise RuntimeError("partner policy fusion is disabled")
        return PrototypePartnerPolicyFusionInput(
            available=jnp.asarray(False, dtype=jnp.bool_),
            prototype_decision_id=jnp.zeros((4,), dtype=jnp.uint32),
            observation_id=jnp.asarray(0, dtype=jnp.int32),
            context_id=jnp.asarray(0, dtype=jnp.int32),
            context_features=jnp.zeros(
                (fusion.config.context_dim,),
                dtype=jnp.float32,
            ),
            safety_action_mask=jnp.ones(
                (fusion.config.n_actions,),
                dtype=jnp.bool_,
            ),
            keyboard_available=jnp.asarray(False, dtype=jnp.bool_),
            keyboard_vector=jnp.zeros(
                (self._config.oak.n_options,),
                dtype=jnp.float32,
            ),
            messages=fusion.empty_messages(),
        )

    @staticmethod
    def _missing_partner_policy_fusion_feedback(
    ) -> PrototypePartnerPolicyFusionFeedback:
        """Return an explicitly unavailable realized-outcome record."""

        unavailable = jnp.asarray(False, dtype=jnp.bool_)
        missing = jnp.asarray(-1, dtype=jnp.int32)
        return PrototypePartnerPolicyFusionFeedback(
            prototype_decision_id=jnp.zeros((4,), dtype=jnp.uint32),
            feedback=PartnerPolicyFusionFeedback(
                available=unavailable,
                decision_id=missing,
                executed_event_id=missing,
                decision_words=jnp.zeros((2,), dtype=jnp.uint32),
                executed_event_words=jnp.zeros((2,), dtype=jnp.uint32),
                executed_action=missing,
                partner_id=missing,
                assistance_value_available=unavailable,
                realized_assistance_value=jnp.asarray(0.0, dtype=jnp.float32),
                safety_outcome_available=unavailable,
                safety_outcome_ok=unavailable,
            ),
        )

    def _normalize_partner_policy_fusion_input(
        self,
        sidecar: PrototypePartnerPolicyFusionInput | None,
    ) -> tuple[PrototypePartnerPolicyFusionInput | None, Array]:
        """Normalize one fixed next-decision sidecar without advancing state."""

        fusion = self._partner_policy_fusion
        if fusion is None:
            if sidecar is not None:
                raise ValueError(
                    "partner_policy_fusion_input requires partner policy fusion"
                )
            return None, jnp.asarray(False, dtype=jnp.bool_)
        if sidecar is None:
            return (
                self._missing_partner_policy_fusion_input(),
                jnp.asarray(False, dtype=jnp.bool_),
            )
        if not isinstance(sidecar, PrototypePartnerPolicyFusionInput):
            raise TypeError(
                "partner_policy_fusion_input must be "
                "PrototypePartnerPolicyFusionInput"
            )
        if not isinstance(sidecar.messages, PartnerMessageBatch):
            raise TypeError("partner fusion messages must be PartnerMessageBatch")
        mask = jnp.asarray(sidecar.safety_action_mask)
        expected_mask_shape = (fusion.config.n_actions,)
        if mask.dtype != jnp.bool_ or mask.shape != expected_mask_shape:
            raise ValueError(
                "partner_policy_fusion_input.safety_action_mask must have "
                f"shape {expected_mask_shape} and dtype bool"
            )
        normalized = PrototypePartnerPolicyFusionInput(
            available=_strict_bool_scalar(
                sidecar.available,
                name="partner_policy_fusion_input.available",
            ),
            prototype_decision_id=_strict_decision_id(
                sidecar.prototype_decision_id,
                name="partner_policy_fusion_input.prototype_decision_id",
            ),
            observation_id=_strict_action_scalar(
                sidecar.observation_id,
                name="partner_policy_fusion_input.observation_id",
            ),
            context_id=_strict_action_scalar(
                sidecar.context_id,
                name="partner_policy_fusion_input.context_id",
            ),
            context_features=_strict_float32_array(
                sidecar.context_features,
                (fusion.config.context_dim,),
                name="partner_policy_fusion_input.context_features",
            ),
            safety_action_mask=jnp.asarray(mask, dtype=jnp.bool_),
            keyboard_available=_strict_bool_scalar(
                sidecar.keyboard_available,
                name="partner_policy_fusion_input.keyboard_available",
            ),
            keyboard_vector=_strict_float32_array(
                sidecar.keyboard_vector,
                (self._config.oak.n_options,),
                name="partner_policy_fusion_input.keyboard_vector",
            ),
            messages=sidecar.messages,
        )
        # The fusion mechanism performs the complete message static-contract
        # check in the same traced call that uses the batch.
        return normalized, jnp.asarray(True, dtype=jnp.bool_)

    def _normalize_partner_policy_fusion_feedback(
        self,
        feedback: PrototypePartnerPolicyFusionFeedback | None,
    ) -> tuple[PrototypePartnerPolicyFusionFeedback | None, Array]:
        """Normalize realized feedback; dynamic mismatch remains an exact no-op."""

        if self._partner_policy_fusion is None:
            if feedback is not None:
                raise ValueError(
                    "partner_policy_fusion_feedback requires partner policy fusion"
                )
            return None, jnp.asarray(False, dtype=jnp.bool_)
        if feedback is None:
            return (
                self._missing_partner_policy_fusion_feedback(),
                jnp.asarray(False, dtype=jnp.bool_),
            )
        if not isinstance(feedback, PrototypePartnerPolicyFusionFeedback):
            raise TypeError(
                "partner_policy_fusion_feedback must be "
                "PrototypePartnerPolicyFusionFeedback"
            )
        if not isinstance(feedback.feedback, PartnerPolicyFusionFeedback):
            raise TypeError(
                "partner_policy_fusion_feedback.feedback must be "
                "PartnerPolicyFusionFeedback"
            )
        core_feedback = feedback.feedback
        normalized_core = PartnerPolicyFusionFeedback(
            available=_strict_bool_scalar(
                core_feedback.available,
                name="partner_policy_fusion_feedback.available",
            ),
            decision_id=_strict_action_scalar(
                core_feedback.decision_id,
                name="partner_policy_fusion_feedback.decision_id",
            ),
            executed_event_id=_strict_action_scalar(
                core_feedback.executed_event_id,
                name="partner_policy_fusion_feedback.executed_event_id",
            ),
            decision_words=_strict_uint32_words(
                core_feedback.decision_words,
                (2,),
                name="partner_policy_fusion_feedback.decision_words",
            ),
            executed_event_words=_strict_uint32_words(
                core_feedback.executed_event_words,
                (2,),
                name="partner_policy_fusion_feedback.executed_event_words",
            ),
            executed_action=_strict_action_scalar(
                core_feedback.executed_action,
                name="partner_policy_fusion_feedback.executed_action",
            ),
            partner_id=_strict_action_scalar(
                core_feedback.partner_id,
                name="partner_policy_fusion_feedback.partner_id",
            ),
            assistance_value_available=_strict_bool_scalar(
                core_feedback.assistance_value_available,
                name=(
                    "partner_policy_fusion_feedback."
                    "assistance_value_available"
                ),
            ),
            realized_assistance_value=_strict_float32_array(
                core_feedback.realized_assistance_value,
                (),
                name=(
                    "partner_policy_fusion_feedback."
                    "realized_assistance_value"
                ),
            ),
            safety_outcome_available=_strict_bool_scalar(
                core_feedback.safety_outcome_available,
                name="partner_policy_fusion_feedback.safety_outcome_available",
            ),
            safety_outcome_ok=_strict_bool_scalar(
                core_feedback.safety_outcome_ok,
                name="partner_policy_fusion_feedback.safety_outcome_ok",
            ),
        )
        normalized = PrototypePartnerPolicyFusionFeedback(
            prototype_decision_id=_strict_decision_id(
                feedback.prototype_decision_id,
                name="partner_policy_fusion_feedback.prototype_decision_id",
            ),
            feedback=normalized_core,
        )
        return normalized, jnp.asarray(True, dtype=jnp.bool_)

    @staticmethod
    def _learning_value_router_inputs(
        sidecar: PrototypeCandidateUpdateAuditEvidence,
        signals: TypedLearningSignals,
    ) -> tuple[LearningValue, LearningValueAvailability]:
        """Build detached producer channels without candidate-validity gating."""

        delight = jnp.asarray(
            sidecar.advantage * sidecar.action_surprisal,
            dtype=jnp.float32,
        )
        signal_input_available = signals.availability.input_valid
        return (
            LearningValue(
                advantage=sidecar.advantage,
                action_surprisal=sidecar.action_surprisal,
                delight=delight,
                epistemic_surprise=signals.epistemic_surprise,
                aleatoric_uncertainty=signals.aleatoric_uncertainty,
                learning_progress=signals.learning_progress,
                change_probability=signals.change_probability,
                safety_cost=sidecar.safety_cost,
            ),
            LearningValueAvailability(
                advantage=sidecar.advantage_available,
                action_surprisal=sidecar.action_surprisal_available,
                delight=(
                    sidecar.advantage_available
                    & sidecar.action_surprisal_available
                ),
                epistemic_surprise=(
                    signal_input_available & signals.availability.epistemic
                ),
                aleatoric_uncertainty=(
                    signal_input_available & signals.availability.aleatoric
                ),
                learning_progress=(
                    signal_input_available
                    & signals.availability.learning_progress
                ),
                change_probability=(
                    signal_input_available
                    & signals.availability.change_probability
                ),
                safety_cost=sidecar.safety_cost_available,
            ),
        )

    @staticmethod
    def _candidate_update_audit_evidence_from_route(
        sidecar: PrototypeCandidateUpdateAuditEvidence,
        router_result: LearningValueRouterResult,
        representation_gradient_valid: Array,
    ) -> CandidateUpdateAuditEvidence:
        """Join candidate/probe validity to the router's raw evidence route."""

        candidate_available = jnp.asarray(
            representation_gradient_valid,
            dtype=jnp.bool_,
        )
        raw_route = router_result.candidate_update_audit_evidence
        return CandidateUpdateAuditEvidence(
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
            learning_value=raw_route.values,
            learning_value_availability=raw_route.availability,
        )

    @staticmethod
    def _candidate_update_audit_evidence_from_signals(
        sidecar: PrototypeCandidateUpdateAuditEvidence,
        signals: TypedLearningSignals,
        representation_gradient_valid: Array,
    ) -> CandidateUpdateAuditEvidence:
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
        return CandidateUpdateAuditEvidence(
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

    def _state_builder_learning_enabled(self) -> bool:
        """Return whether either legacy or explicit representation learning is on."""
        return (
            self._config.learn_state_builder_from_world_model
            or self._config.representation_gradient_mixer is not None
        )

    def _behavior_representation_gradient(
        self,
        state: PrototypeAgentState,
        reward: Array,
        bootstrap_observation: Array,
        control_discount: Array | None,
    ) -> PrototypeBehaviorGradientResult:
        """Differentiate the current causal pre-update control TD loss.

        Targets and learner parameters are stop-gradient constants. On an idle
        primitive transition this is the base differential-Q loss at
        ``base_last_obs``. While an option executes it is the current
        intra-option loss at the same observation. In particular, an option's
        delayed base semi-MDP update is *not* attributed to the current
        representation because that update belongs to ``option_start_obs``.

        This is a one-step semi-gradient. Eligibility traces, parameter update
        scaling, and option-policy importance weighting remain properties of
        the control learner and are not folded into this representation loss.
        """
        observation_dim = self._config.oak.observation_dim
        stomp_config = self._config.oak.stomp
        stomp_state = self._oak_component_state(state.oak_state).stomp_state
        current = jnp.asarray(state.current_representation, dtype=jnp.float32)
        bootstrap = jnp.asarray(bootstrap_observation, dtype=jnp.float32)
        transition_reward = jnp.asarray(reward, dtype=jnp.float32)
        if control_discount is None:
            transition_discount = jnp.asarray(1.0, dtype=jnp.float32)
            environmental_termination = jnp.asarray(False, dtype=jnp.bool_)
        else:
            transition_discount = jnp.asarray(control_discount, dtype=jnp.float32)
            environmental_termination = transition_discount <= 0.0

        def freeze_leaf(leaf: Any) -> Any:
            dtype = getattr(leaf, "dtype", None)
            return jax.lax.stop_gradient(leaf) if dtype is not None else leaf

        def base_source(_: None) -> tuple[Array, ...]:
            base_state = jax.tree.map(freeze_leaf, stomp_state.base_learner_state)
            action_in_range = (
                (stomp_state.base_last_action >= 0)
                & (stomp_state.base_last_action < stomp_config.n_primitive_actions)
            )
            action = jnp.clip(
                stomp_state.base_last_action,
                0,
                stomp_config.n_primitive_actions - 1,
            )
            bootstrap_predictions = self._oak.stomp_agent.base_learner.predict(
                base_state,
                jax.lax.stop_gradient(bootstrap),
            )
            target = jax.lax.stop_gradient(
                transition_reward
                - jax.lax.stop_gradient(stomp_state.base_average_reward)
                + transition_discount * jnp.max(bootstrap_predictions)
            )

            def loss_fn(representation: Array) -> tuple[Array, Array]:
                prediction = self._oak.stomp_agent.base_learner.predict(
                    base_state,
                    representation,
                )[action]
                error = target - prediction
                return jnp.asarray(0.5, dtype=jnp.float32) * jnp.square(error), prediction

            (loss, prediction), gradient = jax.value_and_grad(
                loss_fn,
                has_aux=True,
            )(current)
            parameters_finite = _floating_tree_is_finite(base_state)
            option_terminates = jnp.asarray(False, dtype=jnp.bool_)
            return (
                gradient,
                prediction,
                target,
                loss,
                transition_discount,
                option_terminates,
                parameters_finite,
                action_in_range,
            )

        def intra_option_source(_: None) -> tuple[Array, ...]:
            option_in_range = (
                (stomp_state.executing_option >= 0)
                & (stomp_state.executing_option < stomp_config.n_options)
            )
            option_idx = jnp.clip(
                stomp_state.executing_option,
                0,
                stomp_config.n_options - 1,
            )
            action_in_range = (
                (stomp_state.option_last_intra_action >= 0)
                & (
                    stomp_state.option_last_intra_action
                    < stomp_config.n_primitive_actions
                )
            )
            action = jnp.clip(
                stomp_state.option_last_intra_action,
                0,
                stomp_config.n_primitive_actions - 1,
            )
            q_weights = jax.lax.stop_gradient(
                stomp_state.option_policies.q_weights[option_idx]
            )
            average_reward = jax.lax.stop_gradient(
                stomp_state.option_policies.average_rewards[option_idx]
            )
            pseudo_reward = compute_pseudo_reward(
                self._oak.stomp_agent.spec_arrays,
                option_idx,
                jax.lax.stop_gradient(bootstrap),
            )
            option_terminates = (
                check_option_terminated(
                    self._oak.stomp_agent.spec_arrays,
                    option_idx,
                    jax.lax.stop_gradient(bootstrap),
                    stomp_state.option_steps + 1,
                )
                | environmental_termination
            )
            bootstrap_discount = jnp.where(
                option_terminates,
                jnp.asarray(0.0, dtype=jnp.float32),
                transition_discount,
            )
            target = jax.lax.stop_gradient(
                pseudo_reward
                - average_reward
                + bootstrap_discount * jnp.max(q_weights @ bootstrap)
            )

            def loss_fn(representation: Array) -> tuple[Array, Array]:
                prediction = q_weights[action] @ representation
                error = target - prediction
                return jnp.asarray(0.5, dtype=jnp.float32) * jnp.square(error), prediction

            (loss, prediction), gradient = jax.value_and_grad(
                loss_fn,
                has_aux=True,
            )(current)
            parameters_finite = (
                jnp.all(jnp.isfinite(q_weights))
                & jnp.isfinite(average_reward)
            )
            return (
                gradient,
                prediction,
                target,
                loss,
                bootstrap_discount,
                option_terminates,
                parameters_finite,
                option_in_range & action_in_range,
            )

        executing = stomp_state.executing_option >= 0
        (
            raw_gradient,
            prediction,
            target,
            loss,
            bootstrap_discount,
            option_terminates,
            parameters_finite,
            source_indices_valid,
        ) = (
            base_source(None)
            if stomp_config.n_options == 0
            else jax.lax.cond(
                executing,
                intra_option_source,
                base_source,
                operand=None,
            )
        )
        inputs_finite = (
            jnp.all(jnp.isfinite(current))
            & jnp.all(jnp.isfinite(bootstrap))
            & jnp.isfinite(transition_reward)
            & jnp.isfinite(transition_discount)
            & (transition_discount >= 0.0)
            & (transition_discount <= 1.0)
        )
        td_error = target - prediction
        gradient_finite = jnp.all(jnp.isfinite(raw_gradient))
        gradient_norm = jnp.linalg.norm(raw_gradient)
        prediction_finite = jnp.isfinite(prediction)
        target_finite = jnp.isfinite(target)
        td_error_finite = jnp.isfinite(td_error)
        loss_finite = jnp.isfinite(loss)
        gradient_norm_finite = jnp.isfinite(gradient_norm)
        numerics_valid = (
            prediction_finite
            & target_finite
            & td_error_finite
            & loss_finite
            & gradient_finite
            & gradient_norm_finite
        )
        valid = (
            inputs_finite
            & parameters_finite
            & source_indices_valid
            & numerics_valid
        )
        safe_gradient = jnp.where(
            valid,
            raw_gradient,
            jnp.zeros((observation_dim,), dtype=jnp.float32),
        )
        zero = jnp.asarray(0.0, dtype=jnp.float32)
        # Rejected optimizer-control sidecars must remain safe to scan, log,
        # checkpoint alongside reports, or serialize with ``allow_nan=False``.
        # Boolean provenance flags retain the rejection reason; floating
        # payloads become one canonical finite zero record.
        safe_prediction = jnp.where(valid, prediction, zero)
        safe_target = jnp.where(valid, target, zero)
        safe_td_error = jnp.where(valid, td_error, zero)
        safe_loss = jnp.where(valid, loss, zero)
        safe_norm = jnp.where(valid, gradient_norm, zero)
        safe_bootstrap_discount = jnp.where(valid, bootstrap_discount, zero)
        diagnostics = PrototypeBehaviorGradientDiagnostics(
            source_available=jnp.asarray(True, dtype=jnp.bool_),
            idle_base_source=~executing,
            intra_option_source=executing,
            parameters_finite=parameters_finite,
            inputs_finite=inputs_finite,
            source_indices_valid=source_indices_valid,
            prediction_finite=prediction_finite,
            target_finite=target_finite,
            td_error_finite=td_error_finite,
            loss_finite=loss_finite,
            gradient_norm_finite=gradient_norm_finite,
            prediction=safe_prediction,
            target=safe_target,
            td_error=safe_td_error,
            loss=safe_loss,
            gradient_norm=safe_norm,
            bootstrap_discount=safe_bootstrap_discount,
            option_terminates=option_terminates,
            gradient_finite=gradient_finite,
            valid=valid,
            rejected=~valid,
        )
        return PrototypeBehaviorGradientResult(
            gradient=safe_gradient,
            valid=valid,
            diagnostics=diagnostics,
        )

    def _behavior_active_pair_weights(
        self,
        state: PrototypeAgentState,
    ) -> Float[Array, " active_pair_slots"]:
        """Return the exact old control owner's active pair coefficients."""

        feature_config = self._config.prototype_feature_lifecycle
        if feature_config is None:
            raise RuntimeError("active pair weights require the feature lifecycle")
        active_slots = feature_config.active_pair_slots
        stomp_state = self._oak_component_state(state.oak_state).stomp_state
        stomp_config = self._config.oak.stomp
        base_action = jnp.clip(
            stomp_state.base_last_action,
            0,
            stomp_config.n_primitive_actions - 1,
        )
        base_weight_rows = jnp.stack(
            tuple(
                jnp.ravel(weights)
                for weights in stomp_state.base_learner_state.head_params.weights
            ),
            axis=0,
        )
        base_weights = base_weight_rows[base_action, -active_slots:]
        if stomp_config.n_options == 0:
            return base_weights
        option_index = jnp.clip(
            stomp_state.executing_option,
            0,
            stomp_config.n_options - 1,
        )
        intra_action = jnp.clip(
            stomp_state.option_last_intra_action,
            0,
            stomp_config.n_primitive_actions - 1,
        )
        option_weights = stomp_state.option_policies.q_weights[
            option_index,
            intra_action,
            -active_slots:,
        ]
        return jnp.where(
            stomp_state.executing_option >= 0,
            option_weights,
            base_weights,
        )

    def _horde_active_pair_weights(
        self,
        state: MultiHeadMLPState,
    ) -> Float[Array, "n_demons active_pair_slots"]:
        """Return ordered old linear-Horde active pair coefficients."""

        feature_config = self._config.prototype_feature_lifecycle
        if feature_config is None or feature_config.managed_horde_demons <= 0:
            raise RuntimeError("Horde pair weights require the shared feature lane")
        active_slots = feature_config.active_pair_slots
        return jnp.stack(
            tuple(
                jnp.ravel(weights)[-active_slots:]
                for weights in state.head_params.weights
            ),
            axis=0,
        )

    def _apply_state_builder_learning(
        self,
        source_state: Any,
        destination_state: Any,
        representation_gradient: Array,
        representation_gradient_valid: Array,
        signals: TypedLearningSignals,
        candidate_audit_sidecar: PrototypeCandidateUpdateAuditEvidence | None,
        learning_value_router_result: LearningValueRouterResult | None,
    ) -> tuple[
        Any,
        StateBuilderLearningDiagnostics,
        CandidateUpdateAuditApplicationResult | None,
    ]:
        """Propose from the causal source and commit into advanced recurrence."""
        if not self._state_builder_learning_enabled():
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
        application: CandidateUpdateAuditApplicationResult | None = None
        approved = jnp.asarray(
            representation_gradient_valid,
            dtype=jnp.bool_,
        )
        candidate_update = proposal.candidate_parameter_update
        if self._config.gradient_joy is not None:
            if candidate_audit_sidecar is None:
                raise RuntimeError(
                    "configured candidate-update audit requires a sidecar"
                )
            if self._learning_value_router is None:
                audit_evidence = self._candidate_update_audit_evidence_from_signals(
                    candidate_audit_sidecar,
                    signals,
                    approved,
                )
            else:
                if learning_value_router_result is None:
                    raise RuntimeError(
                        "configured learning-value router requires one routed result"
                    )
                audit_evidence = self._candidate_update_audit_evidence_from_route(
                    candidate_audit_sidecar,
                    learning_value_router_result,
                    approved,
                )
            application = apply_candidate_update(
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

    def _base_representation_dim(self) -> int:
        """Return the builder width before optional pair augmentation."""

        feature_config = self._config.prototype_feature_lifecycle
        if feature_config is not None:
            return feature_config.base_feature_dim
        return self._config.oak.observation_dim

    def _augment_base_representation(
        self,
        feature_state: PrototypeFeatureLifecycleState | None,
        base_representation: Array,
    ) -> Array:
        """Apply the configured fixed-width pair bank, or return the base."""

        lifecycle = self._prototype_feature_lifecycle
        if lifecycle is None:
            return base_representation
        if feature_state is None:
            raise RuntimeError(
                "configured prototype feature lifecycle requires persistent state"
            )
        return lifecycle.augment(feature_state, base_representation)

    @staticmethod
    def _feature_candidate_descriptors(
        state: PrototypeFeatureLifecycleState,
    ) -> Int[Array, " candidate_pair_slots 2"]:
        """Return candidate identities in their declared slot order."""

        return jnp.stack(
            (
                state.learner_state.candidate_left,
                state.learner_state.candidate_right,
            ),
            axis=1,
        ).astype(jnp.int32)

    def _unavailable_feature_lifecycle_diagnostics(
        self,
        state: PrototypeFeatureLifecycleState,
    ) -> PrototypeFeatureLifecycleIntegrationDiagnostics:
        """Return one finite neutral record without touching learner state."""

        lifecycle = self._prototype_feature_lifecycle
        if lifecycle is None:
            raise RuntimeError("prototype feature lifecycle is disabled")
        zero = jnp.asarray(0.0, dtype=jnp.float32)
        false = jnp.asarray(False, dtype=jnp.bool_)
        generation = state.router_state.generation_count
        return PrototypeFeatureLifecycleIntegrationDiagnostics(
            available=false,
            target=zero,
            target_available=false,
            pullback_gradient=jnp.zeros(
                (self._base_representation_dim(),),
                dtype=jnp.float32,
            ),
            pullback_valid=false,
            pullback_semantic_generation=generation,
            prediction=zero,
            error=zero,
            task_targets=jnp.zeros(
                (lifecycle.config.n_tasks,),
                dtype=jnp.float32,
            ),
            task_target_available=jnp.zeros(
                (lifecycle.config.n_tasks,),
                dtype=jnp.bool_,
            ),
            task_predictions=jnp.zeros(
                (lifecycle.config.n_tasks,),
                dtype=jnp.float32,
            ),
            task_errors=jnp.zeros(
                (lifecycle.config.n_tasks,),
                dtype=jnp.float32,
            ),
            metrics=jnp.zeros((7,), dtype=jnp.float32),
            lifecycle=lifecycle.unavailable_diagnostics(generation),
            outer_transaction_committed=false,
        )

    def _unavailable_feature_utility_diagnostics(
        self,
        state: PrototypeFeatureUtilityState,
    ) -> PrototypeFeatureUtilityDiagnostics:
        """Return a finite neutral utility record without observing twice."""

        auditor = self._prototype_feature_utility
        if auditor is None:
            raise RuntimeError("prototype feature utility is disabled")
        return auditor.unavailable_diagnostics(state)

    def _unavailable_feature_utility_integration_diagnostics(
        self,
        state: PrototypeFeatureUtilityState,
    ) -> PrototypeFeatureUtilityIntegrationDiagnostics:
        """Return two neutral phases without observing or rebinding state."""

        neutral = self._unavailable_feature_utility_diagnostics(state)
        false = jnp.asarray(False, dtype=jnp.bool_)
        return PrototypeFeatureUtilityIntegrationDiagnostics(
            observation=neutral,
            rebind=neutral,
            rebind_required=false,
            outer_transaction_committed=false,
        )

    def _unavailable_feature_memory_integration_diagnostics(
        self,
        state: PrototypeAgentState,
        memory_input: PrototypeExperientialMemoryInput,
    ) -> PrototypeFeatureMemoryIntegrationDiagnostics:
        """Return a bank-authenticated no-op without querying or writing memory."""

        adapter = self._prototype_feature_memory
        if adapter is None:
            raise RuntimeError("prototype feature memory is disabled")
        wrapper = self._feature_memory_component_state(state.ia_state)
        binding = self._feature_consumer_binding(state.oak_state)
        rebind = adapter.rebind(wrapper, binding, binding)
        query_matches = (
            memory_input.query_representation_version
            == binding.semantic_generation
        )
        entry_matches = (
            memory_input.entry_representation_version
            == binding.semantic_generation
        )
        representation_finite = jnp.all(
            jnp.isfinite(state.current_representation)
        )
        return PrototypeFeatureMemoryIntegrationDiagnostics(
            rebind=rebind.diagnostics,
            query_source_version_matches=query_matches,
            entry_source_version_matches=entry_matches,
            source_versions_match=query_matches & entry_matches,
            current_destination_encoding_valid=representation_finite,
            bootstrap_destination_encoding_valid=representation_finite,
            decision_destination_encoding_valid=representation_finite,
            post_memory_state_valid=adapter.state_valid(wrapper, binding),
            outer_transaction_committed=jnp.asarray(False, dtype=jnp.bool_),
        )

    @staticmethod
    def _unavailable_atomic_feature_world_memory_diagnostics(
    ) -> PrototypeAtomicFeatureWorldMemoryDiagnostics:
        """Return one fixed-shape no-attempt atomic-composition audit."""

        false = jnp.asarray(False, dtype=jnp.bool_)
        zero = jnp.asarray(0, dtype=jnp.int32)
        lifecycle = PrototypeFeatureLifecycleAdoptionDiagnostics(
            source_state_matches=false,
            source_oak_state_matches=false,
            source_horde_state_matches=false,
            source_consumer_binding_matches=false,
            receipt_matches_preparation=false,
            preparation_internally_valid=false,
            all_consumers_ready=false,
            destination_adopted=false,
            ordinary_update_retained=false,
            external_curation_rolled_back=false,
            transaction_applied=false,
            rejected=jnp.asarray(True, dtype=jnp.bool_),
            preparation_learner_update_evaluations=zero,
            adoption_learner_update_evaluations=zero,
            total_learner_update_evaluations=zero,
        )
        world = PrototypeRoutedLinearWorldAdoptionDiagnostics(
            source_state_matches=false,
            source_router_state_matches=false,
            receipt_matches_preparation=false,
            ordinary_successor_valid=false,
            destination_candidate_valid=false,
            preparation_internally_valid=false,
            all_consumers_ready=false,
            destination_adopted=false,
            ordinary_update_retained=false,
            external_route_rolled_back=false,
            transaction_applied=false,
            rejected=jnp.asarray(True, dtype=jnp.bool_),
            preparation_learner_update_evaluations=zero,
            adoption_learner_update_evaluations=zero,
            total_learner_update_evaluations=zero,
            preparation_router_evaluations=zero,
            adoption_router_evaluations=zero,
            total_router_evaluations=zero,
        )
        return PrototypeAtomicFeatureWorldMemoryDiagnostics(
            available=false,
            descriptor_change_requested=false,
            lifecycle_destination_ready=false,
            world_ordinary_ready=false,
            world_destination_ready=false,
            memory_destination_ready=false,
            all_consumers_ready=false,
            destination_adopted=false,
            ordinary_updates_retained=false,
            external_curation_rolled_back=false,
            lifecycle_adoption=lifecycle,
            world_adoption=world,
            oak_update_evaluations=zero,
            horde_update_evaluations=zero,
            feature_learner_update_evaluations=zero,
            lifecycle_router_evaluations=zero,
            world_learner_update_evaluations=zero,
            world_router_evaluations=zero,
            memory_rebind_evaluations=zero,
            memory_step_evaluations=zero,
        )

    def _unavailable_feature_utility_curation_integration_diagnostics(
        self,
        feature_state: PrototypeFeatureLifecycleState,
        utility_state: PrototypeFeatureUtilityState,
    ) -> PrototypeFeatureUtilityCurationIntegrationDiagnostics:
        """Return one finite no-attempt v6 ranking record."""

        policy = self._prototype_feature_utility_curation
        if policy is None:
            raise RuntimeError("prototype feature utility curation is disabled")
        ranked = policy.rank(
            utility_state,
            source_semantic_generation=(
                feature_state.router_state.generation_count
            ),
            source_semantic_generation_words=(
                feature_state.router_state.generation_words
            ),
            source_active_descriptors=(
                feature_state.router_state.descriptors
            ),
            source_candidate_descriptors=(
                self._feature_candidate_descriptors(feature_state)
            ),
        )
        false = jnp.asarray(False, dtype=jnp.bool_)
        rejected_index = jnp.asarray(-1, dtype=jnp.int32)
        rejected_descriptor = jnp.full((2,), -1, dtype=jnp.int32)
        diagnostics = ranked.diagnostics
        neutral_policy = cast(
            PrototypeFeatureUtilityCurationDiagnostics,
            diagnostics.replace(
                available=false,
                transaction_valid=false,
                override_enabled=false,
                any_active_rank_ready=false,
                any_candidate_rank_ready=false,
                curation_ready=false,
                active_task_evidence_ready=jnp.zeros_like(
                    diagnostics.active_task_evidence_ready
                ),
                candidate_task_evidence_ready=jnp.zeros_like(
                    diagnostics.candidate_task_evidence_ready
                ),
                active_all_tasks_evidence_ready=jnp.zeros_like(
                    diagnostics.active_all_tasks_evidence_ready
                ),
                candidate_all_tasks_evidence_ready=jnp.zeros_like(
                    diagnostics.candidate_all_tasks_evidence_ready
                ),
                candidate_collision_mask=jnp.zeros_like(
                    diagnostics.candidate_collision_mask
                ),
                candidate_rank_ready_mask=jnp.zeros_like(
                    diagnostics.candidate_rank_ready_mask
                ),
                raw_active_fixed_mass_utilities=jnp.zeros_like(
                    diagnostics.raw_active_fixed_mass_utilities
                ),
                raw_candidate_fixed_mass_utilities=jnp.zeros_like(
                    diagnostics.raw_candidate_fixed_mass_utilities
                ),
                emitted_active_ranks=jnp.zeros_like(
                    diagnostics.emitted_active_ranks
                ),
                emitted_candidate_ranks=jnp.zeros_like(
                    diagnostics.emitted_candidate_ranks
                ),
            ),
        )
        return PrototypeFeatureUtilityCurationIntegrationDiagnostics(
            policy=neutral_policy,
            observation_applied=false,
            priority_override_supplied=false,
            priority_override_consulted=false,
            curation_allowed=false,
            selected_active_slot=rejected_index,
            selected_candidate_slot=rejected_index,
            selected_active_descriptor=rejected_descriptor,
            selected_candidate_descriptor=rejected_descriptor,
            lifecycle_curation_proposed=false,
            lifecycle_curation_deferred=false,
            lifecycle_curation_committed=false,
            lifecycle_curation_rolled_back=false,
            outer_transaction_committed=false,
        )

    def _raw_observation_dim(self) -> int:
        if self._state_builder is not None:
            return self._state_builder.observation_dim()
        if self._config.gru_perception is not None:
            return self._config.gru_perception.observation_dim
        return self._config.oak.observation_dim

    @staticmethod
    def _oak_state_numeric_valid(
        oak_agent: OaKAgent,
        state: OaKState,
    ) -> Array:
        """Authenticate one OaK wrapper and its complete nested STOMP state."""

        counter_ceiling = jnp.where(
            state.step_count < jnp.asarray(_INT32_MAX, dtype=jnp.int32),
            state.step_count + jnp.asarray(1, dtype=jnp.int32),
            state.step_count,
        )
        return (
            _lifetime_counter_valid(state.step_words, state.step_count)
            & _lifetime_counter_valid(
                state.stomp_state.step_words,
                state.stomp_state.step_count,
            )
            & jnp.all(state.step_words == state.stomp_state.step_words)
            & (state.step_count == state.stomp_state.step_count)
            & jnp.all(state.execution_counts >= 0)
            & jnp.all(state.execution_counts <= counter_ceiling)
            & jnp.all(jnp.isfinite(state.cumulative_pseudo_rewards))
            & jnp.all(jnp.isfinite(state.utility_ema))
            & oak_agent.stomp_agent.state_valid(state.stomp_state)
        )

    def _state_numeric_valid(self, state: PrototypeAgentState) -> Array:
        """Validate state numerics while preserving world-model init sentinels."""
        oak_component = self._oak_component_state(state.oak_state)
        outer_lifetime_valid = (
            _lifetime_counter_valid(state.step_words, state.step_count)
            & _lifetime_counter_valid(
                state.observation_event_words,
                state.observation_event_count,
            )
            & _prototype_observation_clock_relation_valid(
                state.step_words,
                state.observation_event_words,
            )
            & jnp.all(state.step_words == oak_component.step_words)
            & jnp.all(state.step_words == oak_component.stomp_state.step_words)
            & (state.step_count == oak_component.step_count)
            & (state.step_count == oak_component.stomp_state.step_count)
            & self._oak_state_numeric_valid(self._oak, oak_component)
        )
        sanitized_state = state
        world_model_bounds_valid = jnp.asarray(True)
        state_builder_valid = jnp.asarray(True)
        learning_value_router_valid = jnp.asarray(True, dtype=jnp.bool_)
        feature_lifecycle_valid = jnp.asarray(True, dtype=jnp.bool_)
        feature_horde_valid = jnp.asarray(True, dtype=jnp.bool_)
        feature_utility_valid = jnp.asarray(True, dtype=jnp.bool_)
        interaction_state_valid = jnp.asarray(True, dtype=jnp.bool_)
        ia_state_valid = jnp.asarray(True, dtype=jnp.bool_)
        horde_lifetime_valid = jnp.asarray(True, dtype=jnp.bool_)
        buffer_state_valid = jnp.asarray(True, dtype=jnp.bool_)
        if self._buffer is not None:
            if type(state.buffer_state) is not RecentObservationBufferState:
                buffer_state_valid = jnp.asarray(False, dtype=jnp.bool_)
            else:
                buffer_state = state.buffer_state
                static_buffer_contract = (
                    buffer_state.observations.shape
                    == (
                        self._config.buffer_capacity,
                        self._buffer.observation_dim,
                    )
                    and buffer_state.observations.dtype == jnp.dtype(jnp.float32)
                    and buffer_state.size.shape == ()
                    and buffer_state.size.dtype == jnp.dtype(jnp.int32)
                    and buffer_state.index.shape == ()
                    and buffer_state.index.dtype == jnp.dtype(jnp.int32)
                )
                capacity = jnp.asarray(
                    self._config.buffer_capacity,
                    dtype=jnp.int32,
                )
                zero_words = jnp.zeros((2,), dtype=jnp.uint32)
                has_observation = jnp.any(state.observation_event_words != 0)
                insertion_low = (
                    state.observation_event_words[1]
                    - jnp.asarray(1, dtype=jnp.uint32)
                )
                insertion_borrow = (
                    state.observation_event_words[1]
                    == jnp.asarray(0, dtype=jnp.uint32)
                ).astype(jnp.uint32)
                insertion_words = jnp.where(
                    has_observation,
                    jnp.stack(
                        (
                            state.observation_event_words[0] - insertion_borrow,
                            insertion_low,
                        )
                    ),
                    zero_words,
                )
                capacity_words = jnp.asarray(
                    (0, self._config.buffer_capacity),
                    dtype=jnp.uint32,
                )
                buffer_not_full = _lifetime_words_less_than(
                    insertion_words,
                    capacity_words,
                )
                expected_buffer_size = jnp.where(
                    buffer_not_full,
                    insertion_words[1].astype(jnp.int32),
                    capacity,
                )
                expected_buffer_index = _lifetime_words_modulo(
                    insertion_words,
                    self._config.buffer_capacity,
                ).astype(jnp.int32)
                unused_rows_are_zero = jnp.all(
                    jnp.where(
                        (
                            jnp.arange(
                                self._config.buffer_capacity,
                                dtype=jnp.int32,
                            )
                            >= expected_buffer_size
                        )[:, None],
                        buffer_state.observations == 0.0,
                        jnp.ones_like(
                            buffer_state.observations,
                            dtype=jnp.bool_,
                        ),
                    )
                )
                buffer_state_valid = (
                    jnp.asarray(static_buffer_contract, dtype=jnp.bool_)
                    & jnp.all(jnp.isfinite(buffer_state.observations))
                    & (buffer_state.size >= 0)
                    & (buffer_state.size <= capacity)
                    & (buffer_state.index >= 0)
                    & (buffer_state.index < capacity)
                    & jnp.where(
                        buffer_state.size < capacity,
                        buffer_state.index == buffer_state.size,
                        jnp.asarray(True, dtype=jnp.bool_),
                    )
                    & (buffer_state.size == expected_buffer_size)
                    & (buffer_state.index == expected_buffer_index)
                    & unused_rows_are_zero
                )
        elif state.buffer_state is not None:
            buffer_state_valid = jnp.asarray(False, dtype=jnp.bool_)
        if self._ia is not None:
            ia_state = self._ia_component_state(state.ia_state)
            ia_cortex = ia_state.cortex_state
            ia_state_valid = (
                self._ia.state_is_valid(ia_state)
                & (ia_state.step_count == state.step_count)
                & jnp.all(ia_state.step_words == state.step_words)
                & (ia_state.cerebellum_state.step_count == state.step_count)
                & jnp.all(
                    ia_state.cerebellum_state.step_words == state.step_words
                )
                & jnp.all(ia_cortex.step_words == state.step_words)
                & jnp.all(
                    ia_cortex.stomp_state.step_words == state.step_words
                )
                & self._oak_state_numeric_valid(
                    self._ia.cortex._oak,
                    ia_cortex,
                )
            )
        if self._horde is not None:
            horde_state = self._horde_component_state(state)
            horde_lifetime_valid = (
                _lifetime_counter_valid(
                    horde_state.step_words,
                    horde_state.step_count,
                )
                & (horde_state.step_count == state.step_count)
                & jnp.all(horde_state.step_words == state.step_words)
            )
        if self._partner_policy_fusion is not None:
            interaction = self._partner_interaction_state(state.ia_state)
            if (
                interaction.feedback_prototype_decision_id.shape != (4,)
                or interaction.feedback_prototype_decision_id.dtype
                != jnp.uint32
                or interaction.feedback_prototype_decision_id_available.shape
                != ()
                or interaction.feedback_prototype_decision_id_available.dtype
                != jnp.bool_
            ):
                raise ValueError(
                    "partner feedback Prototype owner has the wrong shape or dtype"
                )
            partner_state = self._partner_fusion_component_state(state.ia_state)
            owner_generation = interaction.feedback_prototype_decision_id[2:]
            current_generation = state.current_decision_id[2:]
            owner_not_from_future = (
                (owner_generation[0] < current_generation[0])
                | (
                    (owner_generation[0] == current_generation[0])
                    & (owner_generation[1] <= current_generation[1])
                )
            )
            owner_belongs_to_lifecycle = jnp.all(
                interaction.feedback_prototype_decision_id[:2]
                == state.current_decision_id[:2]
            )
            interaction_state_valid = (
                self._partner_policy_fusion._state_valid_predicate(partner_state)
                & (
                    interaction.feedback_prototype_decision_id_available
                    == partner_state.feedback_armed
                )
                & jnp.where(
                    interaction.feedback_prototype_decision_id_available,
                    owner_belongs_to_lifecycle & owner_not_from_future,
                    jnp.all(interaction.feedback_prototype_decision_id == 0),
                )
            )
        if self._experiential_memory_policy is not None:
            memory_state = self._experiential_memory_component_state(
                state.ia_state
            )
            interaction_state_valid = (
                interaction_state_valid
                & self._experiential_memory_policy.state_valid(memory_state)
            )
            if self._prototype_feature_memory is not None:
                feature_memory_state = self._feature_memory_component_state(
                    state.ia_state
                )
                interaction_state_valid = (
                    interaction_state_valid
                    & self._prototype_feature_memory.state_valid(
                        feature_memory_state,
                        self._feature_consumer_binding(state.oak_state),
                    )
                )
        if self._state_builder is not None:
            builder_state = self._builder_component_state(
                state.state_builder_state
            )
            state_builder_valid = self._state_builder.state_valid(
                builder_state
            )
        if self._learning_value_router is not None:
            router_state = self._learning_value_router_component_state(
                state.state_builder_state
            )
            router_owner_step = jnp.minimum(
                state.step_count,
                jnp.asarray(
                    self._learning_value_router.config.max_steps,
                    dtype=jnp.int32,
                ),
            )
            learning_value_router_valid = (
                self._learning_value_router.state_valid(router_state)
                & (router_state.step_count == router_owner_step)
            )
        if self._prototype_feature_lifecycle is not None:
            feature_state = self._feature_lifecycle_component_state(
                state.state_builder_state
            )
            consumer_binding = self._feature_consumer_binding(state.oak_state)
            maximum_observations = (
                self._config.prototype_feature_lifecycle.max_observations
            )
            maximum_observation_words = jnp.asarray(
                (
                    (maximum_observations >> 32) & _UINT32_MAX,
                    maximum_observations & _UINT32_MAX,
                ),
                dtype=jnp.uint32,
            )
            expected_observation_words = jnp.where(
                _lifetime_words_less_equal(
                    state.step_words,
                    maximum_observation_words,
                ),
                state.step_words,
                maximum_observation_words,
            )
            feature_lifecycle_valid = (
                self._prototype_feature_lifecycle.state_valid(feature_state)
                & self._prototype_feature_lifecycle.consumer_binding_valid(
                    feature_state,
                    consumer_binding,
                )
                & jnp.all(
                    feature_state.observe_words
                    == expected_observation_words
                )
                & (
                    feature_state.observe_count
                    == _lifetime_words_to_int32_telemetry(
                        expected_observation_words
                    )
                )
            )
            if self._shared_feature_horde_enabled():
                bundle_identity_valid = self._feature_horde_bundle_identity_valid(
                    state
                )
                try:
                    shared_consumer = self._shared_feature_horde_bundle(
                        state.oak_state
                    )
                except TypeError:
                    shared_consumer = None
                if type(shared_consumer) is PrototypeFeatureOaKHordeState:
                    managed_horde_state = shared_consumer.horde_state
                    feature_horde_valid = (
                        bundle_identity_valid
                        & self._prototype_feature_lifecycle.horde_state_valid(
                            managed_horde_state
                        )
                        & self._managed_horde_optimizer_matches_config(
                            managed_horde_state
                        )
                        & (managed_horde_state.step_count == state.step_count)
                        & jnp.all(managed_horde_state.step_words == state.step_words)
                    )
                else:
                    feature_horde_valid = jnp.asarray(False, dtype=jnp.bool_)
            if self._prototype_feature_utility is not None:
                utility_state = self._feature_utility_component_state(
                    state.oak_state
                )
                expected_candidates = self._feature_candidate_descriptors(
                    feature_state
                )
                feature_utility_valid = (
                    self._prototype_feature_utility.state_valid(utility_state)
                    & jnp.array_equal(
                        utility_state.active_descriptors,
                        feature_state.router_state.descriptors,
                    )
                    & jnp.array_equal(
                        utility_state.candidate_descriptors,
                        expected_candidates,
                    )
                    & (
                        utility_state.semantic_generation
                        == feature_state.router_state.generation_count
                    )
                    & jnp.all(
                        utility_state.semantic_generation_words
                        == feature_state.router_state.generation_words
                    )
                    & (
                        utility_state.observation_count
                        == feature_state.observe_count
                    )
                    & jnp.all(
                        utility_state.observation_words
                        == feature_state.observe_words
                    )
                )
        if (
            self._prototype_routed_linear_world_model is not None
            and state.world_model_state is not None
        ):
            world_slot = state.world_model_state
            expected_digest = self._prototype_atomic_feature_world_memory_digest
            if (
                type(world_slot) is not PrototypeAtomicFeatureWorldMemoryState
                or type(world_slot.world_state)
                is not PrototypeRoutedLinearWorldState
                or expected_digest is None
                or getattr(world_slot.schema_digest, "shape", None) != (32,)
                or getattr(world_slot.schema_digest, "dtype", None)
                != jnp.dtype(jnp.uint8)
            ):
                world_model_bounds_valid = jnp.asarray(False, dtype=jnp.bool_)
            else:
                routed_state = world_slot.world_state
                feature_state = self._feature_lifecycle_component_state(
                    state.state_builder_state
                )
                authoritative_binding = self._feature_consumer_binding(
                    state.oak_state
                )
                world_model_bounds_valid = (
                    jnp.array_equal(world_slot.schema_digest, expected_digest)
                    & self._prototype_routed_linear_world_model.state_valid(
                        routed_state
                    )
                    & _tree_arrays_equal(
                        routed_state.consumer_binding,
                        authoritative_binding,
                    )
                    & self._prototype_routed_linear_world_model._router_matches_binding(
                        feature_state.router_state,
                        routed_state.consumer_binding,
                    )
                    & jnp.all(
                        routed_state.model_state.step_words == state.step_words
                    )
                    & (
                        routed_state.model_state.step_count == state.step_count
                    )
                )
            sanitized_state = cast(
                PrototypeAgentState,
                state.replace(world_model_state=None),
            )
        elif (
            self._recurrent_latent_world_model_ensemble is not None
            and state.world_model_state is not None
        ):
            world_model_bounds_valid = self._recurrent_wrapper_numeric_valid(
                cast(
                    PrototypeRecurrentLatentWorldModelState,
                    state.world_model_state,
                )
            )
            sanitized_state = cast(
                PrototypeAgentState,
                state.replace(world_model_state=None),
            )
        elif (
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
            world_model_slot = state.world_model_state
            template = self._world_model_state_contract_template
            schema_identity_valid = jnp.asarray(True, dtype=jnp.bool_)
            world_model_state: ActionConditionedWorldModelState | None = None
            if self._prototype_feature_lifecycle is None:
                if type(world_model_slot) is ActionConditionedWorldModelState:
                    world_model_state = world_model_slot
            else:
                expected_digest = self._prototype_feature_world_model_digest
                if (
                    type(world_model_slot) is PrototypeFeatureWorldModelState
                    and type(world_model_slot.model_state)
                    is ActionConditionedWorldModelState
                    and expected_digest is not None
                    and getattr(world_model_slot.schema_digest, "shape", None)
                    == (32,)
                    and getattr(world_model_slot.schema_digest, "dtype", None)
                    == jnp.dtype(jnp.uint8)
                ):
                    world_model_state = world_model_slot.model_state
                    schema_identity_valid = jnp.array_equal(
                        world_model_slot.schema_digest,
                        expected_digest,
                    )
            static_contract_valid = (
                world_model_state is not None
                and template is not None
                and _prototype_tree_static_contract_matches(
                    world_model_state,
                    template,
                )
            )
            if not static_contract_valid:
                world_model_bounds_valid = jnp.asarray(False, dtype=jnp.bool_)
                sanitized_state = cast(
                    PrototypeAgentState,
                    state.replace(world_model_state=None),
                )
            else:
                exact_world_state = cast(
                    ActionConditionedWorldModelState,
                    world_model_state,
                )
                observation_min = exact_world_state.observation_min
                observation_max = exact_world_state.observation_max
                reward_min = exact_world_state.reward_min
                reward_max = exact_world_state.reward_max
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
                    & jnp.all(observation_min <= observation_max)
                    & (reward_min <= reward_max)
                )
                learner_status = self._world_model.learner._counter_status(
                    exact_world_state.learner_state
                )
                if self._prototype_feature_lifecycle is None:
                    model_outer_counter_valid = (
                        exact_world_state.step_count <= state.step_count
                    ) & _lifetime_words_less_equal(
                        exact_world_state.step_words,
                        state.step_words,
                    )
                else:
                    model_outer_counter_valid = (
                        exact_world_state.step_count == state.step_count
                    ) & jnp.all(
                        exact_world_state.step_words == state.step_words
                    )
                wrapper_counter_valid = (
                    _lifetime_counter_valid(
                        exact_world_state.step_words,
                        exact_world_state.step_count,
                    )
                    & (
                        exact_world_state.step_count
                        == exact_world_state.learner_state.step_count
                    )
                    & jnp.all(
                        exact_world_state.step_words
                        == exact_world_state.learner_state.step_words
                    )
                    & model_outer_counter_valid
                    & learner_status.lifetime_counter_valid
                    & learner_status.normalizer_counter_aligned
                    & self._action_world_model_optimizer_matches_config(
                        exact_world_state.learner_state
                    )
                )
                world_model_bounds_valid = (
                    schema_identity_valid
                    & wrapper_counter_valid
                    & jnp.where(
                        jnp.all(exact_world_state.step_words == 0),
                        pristine_bounds,
                        finite_bounds,
                    )
                    & jnp.isfinite(exact_world_state.model_error_ema)
                    & (exact_world_state.model_error_ema >= 0.0)
                )
                sanitized_world_model_state = exact_world_state.replace(
                    observation_min=jnp.zeros_like(observation_min),
                    observation_max=jnp.zeros_like(observation_max),
                    reward_min=jnp.zeros_like(reward_min),
                    reward_max=jnp.zeros_like(reward_max),
                )
                sanitized_state = cast(
                    PrototypeAgentState,
                    state.replace(
                        world_model_state=self._action_world_model_state_slot(
                            sanitized_world_model_state
                        )
                    ),
                )
        elif (
            self._prototype_routed_linear_world_model is not None
            or self._world_model is not None
            or self._world_model_ensemble is not None
            or self._model_replay_rehearsal is not None
            or self._recurrent_latent_world_model_ensemble is not None
            or state.world_model_state is not None
        ):
            world_model_bounds_valid = jnp.asarray(False, dtype=jnp.bool_)
            sanitized_state = cast(
                PrototypeAgentState,
                state.replace(world_model_state=None),
            )
        return (
            outer_lifetime_valid
            & world_model_bounds_valid
            & state_builder_valid
            & learning_value_router_valid
            & feature_lifecycle_valid
            & feature_horde_valid
            & feature_utility_valid
            & interaction_state_valid
            & ia_state_valid
            & horde_lifetime_valid
            & buffer_state_valid
            & _floating_tree_is_finite(sanitized_state)
        )

    def _representation_cache_consistent(
        self,
        state: PrototypeAgentState,
    ) -> Array:
        """Check that causal representation caches agree with their owners."""
        oak_state = self._oak_component_state(state.oak_state)
        representation_consistent = jnp.array(True)
        builder_count_consistent = jnp.array(True)
        if self._state_builder is not None:
            builder_state = self._builder_component_state(
                state.state_builder_state
            )
            rebuilt_base_representation = self._state_builder.encode(
                builder_state,
                state.current_raw_observation,
            )
            feature_state = (
                self._feature_lifecycle_component_state(
                    state.state_builder_state
                )
                if self._prototype_feature_lifecycle is not None
                else None
            )
            rebuilt_representation = self._augment_base_representation(
                feature_state,
                rebuilt_base_representation,
            )
            representation_consistent = jnp.array_equal(
                rebuilt_representation,
                state.current_representation,
            )
            builder_count_consistent = (
                builder_state.step_count
                == state.observation_event_count
            )
            if hasattr(builder_state, "step_words"):
                builder_count_consistent = (
                    builder_count_consistent
                    & jnp.all(
                        builder_state.step_words
                        == state.observation_event_words
                    )
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
        horde_count_consistent = jnp.asarray(True, dtype=jnp.bool_)
        if self._shared_feature_horde_enabled():
            horde_count_consistent = (
                self._horde_component_state(state).step_count
                == state.step_count
            ) & jnp.all(
                self._horde_component_state(state).step_words
                == state.step_words
            )
        return (
            jnp.all(jnp.isfinite(state.current_raw_observation))
            & jnp.all(jnp.isfinite(state.current_representation))
            & (state.observation_event_count >= 1)
            & (oak_state.step_count == state.step_count)
            & (oak_state.stomp_state.step_count == state.step_count)
            & jnp.all(oak_state.step_words == state.step_words)
            & jnp.all(oak_state.stomp_state.step_words == state.step_words)
            & horde_count_consistent
            & representation_consistent
            & builder_count_consistent
            & jnp.array_equal(
                state.current_representation,
                oak_state.stomp_state.base_last_obs,
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
        stomp_state = self._oak_component_state(state.oak_state).stomp_state
        executing_option_valid = (
            (stomp_state.executing_option >= -1)
            & (stomp_state.executing_option < self._config.oak.n_options)
        )
        # Bind the dispatched primitive command to the learner whose current
        # TD loss owns its representation credit. An idle transition belongs
        # to the base primitive head; an executing transition belongs to the
        # active option's intra-policy head. Merely checking
        # ``last_primitive_action`` would permit a valid-range owner mismatch.
        control_gradient_owner_matches = jnp.where(
            stomp_state.executing_option >= 0,
            stomp_state.option_last_intra_action == state.current_action,
            stomp_state.base_last_action == state.current_action,
        )
        option_identity_matches = jnp.where(
            stomp_state.executing_option >= 0,
            stomp_state.base_last_action
            == self._config.oak.n_primitive_actions + stomp_state.executing_option,
            jnp.asarray(True, dtype=jnp.bool_),
        )
        recurrent_cache_consistent = jnp.asarray(True, dtype=jnp.bool_)
        if self._recurrent_latent_world_model_ensemble is not None:
            recurrent_cache_consistent = self._recurrent_armed_cache_consistent(
                state
            )
        return (
            self._representation_cache_consistent(state)
            & action_in_range
            & executing_option_valid
            & (state.current_action == stomp_state.last_primitive_action)
            & control_gradient_owner_matches
            & option_identity_matches
            & _current_counter_capacity_available(state)
            & recurrent_cache_consistent
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
        exhausted = jnp.all(counter == generation_maximum) | (
            ~_current_counter_capacity_available(state)
        )
        recurrent_cache_disarmed = jnp.asarray(True, dtype=jnp.bool_)
        if self._recurrent_latent_world_model_ensemble is not None:
            recurrent_wrapper = cast(
                PrototypeRecurrentLatentWorldModelState,
                state.world_model_state,
            )
            recurrent_cache_disarmed = ~recurrent_wrapper.decision_cache.valid
        return (
            (~state.started)
            & (state.step_count > 0)
            & (state.current_action == -1)
            & exhausted
            & self._representation_cache_consistent(state)
            & recurrent_cache_disarmed
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
        oak_state = self._oak_component_state(state.oak_state)
        recurrent_fresh = jnp.array(True)
        if self._state_builder is not None:
            recurrent_fresh = (
                self._builder_component_state(
                    state.state_builder_state
                ).step_count
                == 0
            )
        elif state.gru_state is not None:
            recurrent_fresh = jnp.all(state.gru_state.hidden == 0.0)
        recurrent_world_fresh = jnp.asarray(True, dtype=jnp.bool_)
        if self._prototype_routed_linear_world_model is not None:
            routed_world = self._atomic_routed_world_component_state(
                state.world_model_state
            )
            recurrent_world_fresh = (
                (routed_world.model_state.step_count == 0)
                & jnp.all(routed_world.model_state.step_words == 0)
                & (routed_world.buffer_state.size == 0)
                & (routed_world.buffer_state.index == 0)
                & jnp.all(routed_world.buffer_state.observations == 0.0)
                & (routed_world.generation_update_count == 0)
                & jnp.all(routed_world.generation_update_words == 0)
                & (routed_world.planned_backup_count == 0)
                & jnp.all(routed_world.planned_backup_words == 0)
            )
        if self._recurrent_latent_world_model_ensemble is not None:
            recurrent_wrapper = cast(
                PrototypeRecurrentLatentWorldModelState,
                state.world_model_state,
            )
            recurrent_world_fresh = (
                (~recurrent_wrapper.decision_cache.valid)
                & (recurrent_wrapper.model_state.event_count == 0)
                & (recurrent_wrapper.signal_state.step_count == 0)
            )
        partner_fusion_fresh = jnp.asarray(True, dtype=jnp.bool_)
        if self._partner_policy_fusion is not None:
            partner_fusion_fresh = _tree_arrays_equal(
                self._partner_fusion_component_state(state.ia_state),
                self._partner_policy_fusion.init(),
            )
        experiential_memory_fresh = jnp.asarray(True, dtype=jnp.bool_)
        if self._experiential_memory is not None:
            experiential_memory_fresh = _tree_arrays_equal(
                self._experiential_memory_component_state(state.ia_state),
                self._experiential_memory.init(),
            )
        buffer_fresh = jnp.asarray(True, dtype=jnp.bool_)
        if self._buffer is not None:
            if type(state.buffer_state) is RecentObservationBufferState:
                buffer_fresh = (
                    (state.buffer_state.size == 0)
                    & (state.buffer_state.index == 0)
                    & jnp.all(state.buffer_state.observations == 0.0)
                )
            else:
                buffer_fresh = jnp.asarray(False, dtype=jnp.bool_)
        horde_fresh = jnp.asarray(True, dtype=jnp.bool_)
        if self._shared_feature_horde_enabled():
            horde_fresh = (
                self._horde_component_state(state).step_count
                == jnp.asarray(0, dtype=jnp.int32)
            )
        return (
            (~state.started)
            & (state.step_count == 0)
            & (state.observation_event_count == 0)
            & jnp.all(state.step_words == 0)
            & jnp.all(state.observation_event_words == 0)
            & (state.current_action == -1)
            & jnp.all(state.current_decision_id[2:] == 0)
            & jnp.all(state.current_raw_observation == 0.0)
            & jnp.all(state.current_representation == 0.0)
            & (oak_state.step_count == 0)
            & (oak_state.stomp_state.step_count == 0)
            & jnp.all(oak_state.step_words == 0)
            & jnp.all(oak_state.stomp_state.step_words == 0)
            & horde_fresh
            & jnp.all(oak_state.stomp_state.base_last_obs == 0.0)
            & recurrent_fresh
            & recurrent_world_fresh
            & partner_fusion_fresh
            & experiential_memory_fresh
            & buffer_fresh
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
            wm_state = self._action_world_model_state_slot(
                self._world_model.init(wm_key)
            )
            buf_state = self._buffer.init()
        elif self._world_model_ensemble is not None:
            wm_state = self._world_model_ensemble.init(wm_key)
        elif self._model_replay_rehearsal is not None:
            wm_state = self._model_replay_rehearsal.init(wm_key)
        elif (
            self._recurrent_latent_world_model_ensemble is not None
            and self._recurrent_signal_estimator is not None
        ):
            recurrent_model_state = (
                self._recurrent_latent_world_model_ensemble.init(wm_key)
            )
            recurrent_decision_cache = self._recurrent_decision_for_observation(
                recurrent_model_state,
                jnp.zeros(
                    (self._config.oak.observation_dim,),
                    dtype=jnp.float32,
                ),
                jnp.asarray(0, dtype=jnp.int32),
                jnp.asarray(False, dtype=jnp.bool_),
            )
            wm_state = PrototypeRecurrentLatentWorldModelState(
                model_state=recurrent_model_state,
                decision_cache=recurrent_decision_cache,
                signal_state=self._recurrent_signal_estimator.init(),
            )

        horde_state: Any = None
        if self._horde is not None:
            horde_state = self._horde.init(self._config.oak.observation_dim, horde_key)

        ia_state: Any = None
        if self._ia is not None:
            ia_state = self._ia.init(ia_key)
        partner_fusion_state: Any = None
        if self._partner_policy_fusion is not None:
            partner_fusion_state = self._partner_policy_fusion.init()
        experiential_memory_state: ExperientialMemoryState | None = None
        if self._experiential_memory is not None:
            experiential_memory_state = self._experiential_memory.init()

        gru_state: Any = None
        if self._config.gru_perception is not None:
            gru_state = _init_gru_state(self._config.gru_perception, gru_key)

        builder_state: Any = None
        if self._state_builder is not None:
            builder_state = self._state_builder.init(builder_key)
        feature_lifecycle_state: PrototypeFeatureLifecycleState | None = None
        feature_consumer_binding: PrototypeFeatureConsumerBinding | None = None
        feature_utility_state: PrototypeFeatureUtilityState | None = None
        if self._prototype_feature_lifecycle is not None:
            feature_key = jr.fold_in(
                builder_key,
                _PROTOTYPE_FEATURE_LIFECYCLE_KEY_TAG,
            )
            (
                feature_lifecycle_state,
                feature_consumer_binding,
            ) = self._prototype_feature_lifecycle.init_bound(feature_key)
        if self._prototype_routed_linear_world_model is not None:
            if (
                feature_lifecycle_state is None
                or feature_consumer_binding is None
            ):
                raise RuntimeError(
                    "atomic routed-world initialization requires the live bank"
                )
            wm_state = self._atomic_routed_world_state_slot(
                self._prototype_routed_linear_world_model.init(
                    wm_key,
                    feature_consumer_binding,
                    feature_lifecycle_state.router_state,
                )
            )
            buf_state = None
        memory_slot_state: Any = experiential_memory_state
        if self._prototype_feature_memory is not None:
            if (
                experiential_memory_state is None
                or feature_consumer_binding is None
            ):
                raise RuntimeError(
                    "feature-memory initialization requires memory and bank binding"
                )
            memory_slot_state = self._prototype_feature_memory.init(
                feature_consumer_binding,
                experiential_memory_state,
            )
        interaction_state = self._interaction_slot(
            ia_state,
            partner_fusion_state,
            experiential_memory_state=memory_slot_state,
        )
        if self._prototype_feature_utility is not None:
            if feature_lifecycle_state is None:
                raise RuntimeError(
                    "prototype feature utility requires lifecycle state"
                )
            feature_utility_state = self._prototype_feature_utility.init(
                active_descriptors=(
                    feature_lifecycle_state.router_state.descriptors
                ),
                candidate_descriptors=self._feature_candidate_descriptors(
                    feature_lifecycle_state
                ),
                semantic_generation=(
                    feature_lifecycle_state.router_state.generation_count
                ),
                semantic_generation_words=(
                    feature_lifecycle_state.router_state.generation_words
                ),
            )
        representation_state = self._representation_state_slot(
            builder_state,
            feature_lifecycle_state,
            (
                self._learning_value_router.init()
                if self._learning_value_router is not None
                else None
            ),
        )

        initial_state = PrototypeAgentState(
            oak_state=self._oak_state_slot(
                oak_state,
                feature_consumer_binding,
                horde_state,
                feature_utility_state,
            ),
            world_model_state=wm_state,
            buffer_state=buf_state,
            horde_state=(
                None if self._shared_feature_horde_enabled() else horde_state
            ),
            ia_state=interaction_state,
            gru_state=gru_state,
            state_builder_state=representation_state,
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
            observation_event_words=jnp.zeros((2,), dtype=jnp.uint32),
            step_count=jnp.array(0, dtype=jnp.int32),
            step_words=jnp.zeros((2,), dtype=jnp.uint32),
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
        *,
        extended_action_mask: Array | None = None,
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
        action_mask = (
            None
            if extended_action_mask is None
            else _strict_bool_array(
                extended_action_mask,
                (self._config.oak.stomp.n_total_actions,),
                name="extended_action_mask",
            )
        )
        mask_valid = (
            jnp.asarray(True, dtype=jnp.bool_)
            if action_mask is None
            else jnp.all(
                action_mask[: self._config.oak.n_primitive_actions]
            )
            & jnp.any(action_mask)
        )
        finite = jnp.all(jnp.isfinite(raw_obs))
        pristine = self._pristine_state_consistent(state)
        if not _contains_tracer((raw_obs, state.started, state.step_count)):
            if not bool(pristine):
                raise RuntimeError("start requires a fresh unstarted PrototypeAgentState")
            if not bool(finite):
                raise ValueError("initial_observation must be finite")
            if not bool(mask_valid):
                raise ValueError("extended_action_mask must keep every primitive live")
        safe_raw_obs = jnp.where(finite, raw_obs, jnp.zeros_like(raw_obs))

        def prime(fresh_state: PrototypeAgentState) -> PrototypeAgentState:
            new_gru_state = fresh_state.gru_state
            new_builder_state = self._builder_component_state(
                fresh_state.state_builder_state
            )
            feature_state = (
                self._feature_lifecycle_component_state(
                    fresh_state.state_builder_state
                )
                if self._prototype_feature_lifecycle is not None
                else None
            )
            obs_for_oak = safe_raw_obs
            if self._state_builder is not None:
                new_builder_state, base_obs_for_oak = self._state_builder.start(
                    new_builder_state,
                    safe_raw_obs,
                )
                obs_for_oak = self._augment_base_representation(
                    feature_state,
                    base_obs_for_oak,
                )
            elif fresh_state.gru_state is not None:
                new_gru_state, obs_for_oak = _gru_step(
                    fresh_state.gru_state,
                    safe_raw_obs,
                )
            feature_consumer_binding = (
                self._feature_consumer_binding(fresh_state.oak_state)
                if self._prototype_feature_lifecycle is not None
                else None
            )
            feature_utility_state = (
                self._feature_utility_component_state(fresh_state.oak_state)
                if self._prototype_feature_utility is not None
                else None
            )
            new_oak = self._oak.start(
                self._oak_component_state(fresh_state.oak_state),
                obs_for_oak,
                extended_action_mask=action_mask,
            )
            previous_ia = self._ia_component_state(fresh_state.ia_state)
            new_ia = previous_ia
            if self._ia is not None and previous_ia is not None:
                new_ia = self._ia.start(previous_ia, obs_for_oak)
            partner_state: Any = None
            if self._partner_policy_fusion is not None:
                partner_state = self._partner_fusion_component_state(
                    fresh_state.ia_state
                )
            memory_state: Any = None
            if self._experiential_memory is not None:
                memory_state = self._memory_slot_component_state(
                    fresh_state.ia_state
                )
            new_interaction_state = self._interaction_slot(
                new_ia,
                partner_state,
                experiential_memory_state=memory_state,
            )
            new_world_model_state = fresh_state.world_model_state
            if self._recurrent_latent_world_model_ensemble is not None:
                recurrent_wrapper = cast(
                    PrototypeRecurrentLatentWorldModelState,
                    fresh_state.world_model_state,
                )
                recurrent_decision_cache = (
                    self._recurrent_decision_for_observation(
                        recurrent_wrapper.model_state,
                        obs_for_oak,
                        new_oak.stomp_state.last_primitive_action,
                        jnp.asarray(True, dtype=jnp.bool_),
                    )
                )
                new_world_model_state = recurrent_wrapper.replace(
                    decision_cache=recurrent_decision_cache,
                )
            candidate = cast(
                PrototypeAgentState,
                fresh_state.replace(
                    oak_state=self._oak_state_slot(
                        new_oak,
                        feature_consumer_binding,
                        self._horde_component_state(fresh_state),
                        feature_utility_state,
                    ),
                    world_model_state=new_world_model_state,
                    ia_state=new_interaction_state,
                    gru_state=new_gru_state,
                    state_builder_state=self._representation_state_slot(
                        new_builder_state,
                        feature_state,
                        (
                            self._learning_value_router_component_state(
                                fresh_state.state_builder_state
                            )
                            if self._learning_value_router is not None
                            else None
                        ),
                    ),
                    current_raw_observation=safe_raw_obs,
                    current_representation=obs_for_oak,
                    current_action=new_oak.stomp_state.last_primitive_action,
                    current_decision_id=fresh_state.current_decision_id.at[2:].set(
                        jnp.zeros((2,), dtype=jnp.uint32)
                    ),
                    started=jnp.array(True),
                    observation_event_count=jnp.array(1, dtype=jnp.int32),
                    observation_event_words=jnp.asarray(
                        (0, 1),
                        dtype=jnp.uint32,
                    ),
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
            jax.lax.cond(
                pristine & finite & mask_valid,
                prime,
                lambda unchanged: unchanged,
                state,
            ),
        )
        if not _contains_tracer((raw_obs, state.started, state.step_count)):
            if bool(pristine & finite & mask_valid) and not bool(result.started):
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

    def validate_state(self, state: PrototypeAgentState) -> Bool[Array, ""]:
        """Return whether a pristine, armed, or exhausted state is checkpoint-safe."""

        return self._checkpoint_state_valid(state)

    def replace_cached_primitive_action(
        self,
        state: PrototypeAgentState,
        *,
        decision_id: Array,
        decision_observation: Array,
        proposed_action: Array,
        safety_action_mask: Array,
    ) -> PrototypeCachedPrimitiveActionReplacement:
        """Atomically replace the currently dispatchable primitive command.

        The exact four-word Prototype decision ID and encoded decision
        observation are ownership assertions. On a stale/misattributed ID,
        observation mismatch, invalid STOMP owner, unsafe fallback, or invalid
        proposed action, the complete Prototype state is unchanged. A commit
        synchronizes the base or active-option STOMP credit owner,
        ``current_action``, and the optional recurrent world-model decision
        cache without advancing counters or consuming RNG.
        """

        expected_decision_id = _strict_decision_id(
            decision_id,
            name="decision_id",
        )
        observation = _strict_float_array(
            decision_observation,
            (self._config.oak.observation_dim,),
            name="decision_observation",
        )
        current_oak_state = self._oak_component_state(state.oak_state)
        replacement = replace_dispatched_primitive_action(
            current_oak_state.stomp_state,
            observation,
            proposed_action,
            safety_action_mask=safety_action_mask,
        )
        decision_id_matches = jnp.array_equal(
            expected_decision_id,
            state.current_decision_id,
        )
        observation_matches = (
            replacement.decision.observation_matches
            & jnp.array_equal(observation, state.current_representation)
        )
        state_valid_before = state.started & self._state_cache_consistent(state)
        replacement_accepted = ~replacement.decision.failed_closed
        candidate_action = jnp.where(
            replacement_accepted,
            replacement.decision.effective_action,
            state.current_action,
        ).astype(jnp.int32)
        candidate_oak_state = cast(
            OaKState,
            current_oak_state.replace(stomp_state=replacement.state),
        )
        feature_binding = (
            self._feature_consumer_binding(state.oak_state)
            if self._prototype_feature_lifecycle is not None
            else None
        )
        managed_horde_state = (
            self._horde_component_state(state)
            if self._shared_feature_horde_enabled()
            else None
        )
        feature_utility_state = (
            self._feature_utility_component_state(state.oak_state)
            if self._feature_utility_enabled()
            else None
        )
        candidate_world_model_state = state.world_model_state
        if self._recurrent_latent_world_model_ensemble is not None:
            recurrent_wrapper = cast(
                PrototypeRecurrentLatentWorldModelState,
                state.world_model_state,
            )
            candidate_world_model_state = recurrent_wrapper.replace(
                decision_cache=self._recurrent_decision_for_observation(
                    recurrent_wrapper.model_state,
                    state.current_representation,
                    candidate_action,
                    state.started,
                )
            )
        candidate = cast(
            PrototypeAgentState,
            state.replace(
                oak_state=self._oak_state_slot(
                    candidate_oak_state,
                    feature_binding,
                    managed_horde_state,
                    feature_utility_state,
                ),
                world_model_state=candidate_world_model_state,
                current_action=candidate_action,
            ),
        )
        state_valid_after = self._state_cache_consistent(candidate)
        committed = (
            state_valid_before
            & decision_id_matches
            & observation_matches
            & replacement_accepted
            & state_valid_after
        )
        final_state = cast(
            PrototypeAgentState,
            jax.lax.cond(
                committed,
                lambda _: candidate,
                lambda _: state,
                operand=None,
            ),
        )
        return PrototypeCachedPrimitiveActionReplacement(
            state=final_state,
            action=jnp.where(
                committed,
                candidate_action,
                state.current_action,
            ).astype(jnp.int32),
            dispatch_replacement=replacement.decision,
            decision_id_matches=decision_id_matches,
            observation_matches=observation_matches,
            state_valid_before=state_valid_before,
            state_valid_after=state_valid_after,
            committed=committed,
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
            base_obs = self._state_builder.encode(
                self._builder_component_state(state.state_builder_state),
                raw_obs,
            )
            feature_state = (
                self._feature_lifecycle_component_state(
                    state.state_builder_state
                )
                if self._prototype_feature_lifecycle is not None
                else None
            )
            obs = self._augment_base_representation(
                feature_state,
                base_obs,
            )
        elif state.gru_state is not None:
            obs = jnp.concatenate([raw_obs, state.gru_state.hidden], axis=0)
        n_prim = self._config.oak.n_primitive_actions
        all_q = self._oak.base_q_values(
            self._oak_component_state(state.oak_state),
            obs,
        )
        return jnp.argmax(all_q[:n_prim]).astype(jnp.int32)

    def _oak_counterfactual_dispatch_score(
        self,
        oak_state: OaKState,
        decision_observation: Array,
    ) -> Array:
        """Return the exact current dispatch owner's finite-Q candidate score."""

        stomp = oak_state.stomp_state
        n_primitive = self._config.oak.n_primitive_actions
        n_options = self._config.oak.n_options
        base_values = self._oak.base_q_values(oak_state, decision_observation)
        base_index = jnp.clip(
            stomp.base_last_action,
            0,
            n_primitive + n_options - 1,
        )
        base_score = base_values[base_index]
        if n_options == 0:
            return base_score
        option_index = jnp.clip(stomp.executing_option, 0, n_options - 1)
        primitive_index = jnp.clip(
            stomp.last_primitive_action,
            0,
            n_primitive - 1,
        )
        option_score = jnp.dot(
            stomp.option_policies.q_weights[option_index, primitive_index],
            decision_observation,
        )
        return jnp.where(stomp.executing_option >= 0, option_score, base_score)

    def _apply_experiential_memory(
        self,
        memory_state: ExperientialMemoryState,
        oak_state: OaKState,
        current_representation: Array,
        bootstrap_representation: Array,
        decision_representation: Array,
        executed_action: Array,
        reward: Array,
        *,
        current_prototype_decision_id: Array,
        next_prototype_decision_id: Array,
        next_armed: Array,
        transaction_allowed: Array,
        memory_input: PrototypeExperientialMemoryInput,
        input_supplied: Array,
        source_representation_version: Array | None = None,
        destination_representation_version: Array | None = None,
    ) -> tuple[
        ExperientialMemoryState,
        OaKState,
        Array,
        PrototypeExperientialMemoryDiagnostics,
    ]:
        """Query pre-write memory, ground one real exemplar, then dispatch."""

        memory = self._experiential_memory
        policy = self._experiential_memory_policy
        if memory is None or policy is None:
            raise RuntimeError("experiential memory is disabled")
        if (source_representation_version is None) != (
            destination_representation_version is None
        ):
            raise RuntimeError(
                "feature-owned memory versions require both source and destination"
            )
        feature_owned_versions = source_representation_version is not None
        source_version = (
            jnp.asarray(source_representation_version, dtype=jnp.int32)
            if feature_owned_versions
            else memory_input.query_representation_version
        )
        destination_version = (
            jnp.asarray(destination_representation_version, dtype=jnp.int32)
            if feature_owned_versions
            else memory_input.query_representation_version
        )
        query_source_version_matches = (
            memory_input.query_representation_version == source_version
        )
        entry_source_version_matches = (
            memory_input.entry_representation_version == source_version
        )
        version_metadata_valid = jnp.where(
            jnp.asarray(feature_owned_versions, dtype=jnp.bool_),
            query_source_version_matches & entry_source_version_matches,
            (memory_input.query_representation_version >= 0)
            & (memory_input.entry_representation_version >= 0),
        )
        transaction_gate = jnp.asarray(transaction_allowed, dtype=jnp.bool_)
        current_id_matches = jnp.array_equal(
            memory_input.current_prototype_decision_id,
            current_prototype_decision_id,
        )
        next_id_matches = jnp.array_equal(
            memory_input.next_prototype_decision_id,
            next_prototype_decision_id,
        )
        query_uncertainty_valid = (
            jnp.isfinite(memory_input.query_uncertainty)
            & (memory_input.query_uncertainty >= 0.0)
            & (
                memory_input.query_uncertainty_available
                | (memory_input.query_uncertainty == 0.0)
            )
        )
        entry_uncertainty_valid = (
            jnp.isfinite(memory_input.entry_uncertainty)
            & (memory_input.entry_uncertainty >= 0.0)
            & (
                memory_input.entry_uncertainty_available
                | (memory_input.entry_uncertainty == 0.0)
            )
        )
        safety_cost_valid = (
            jnp.isfinite(memory_input.safety_cost)
            & (memory_input.safety_cost >= 0.0)
            & (
                memory_input.safety_cost_available
                | (memory_input.safety_cost == 0.0)
            )
        )
        utility_valid = (
            jnp.isfinite(memory_input.utility)
            & (memory_input.utility >= 0.0)
            & (
                memory_input.utility_available
                | (memory_input.utility == 0.0)
            )
        )
        metadata_valid = (
            version_metadata_valid
            & query_uncertainty_valid
            & entry_uncertainty_valid
            & safety_cost_valid
            & jnp.isfinite(memory_input.reliability)
            & (memory_input.reliability >= 0.0)
            & (memory_input.reliability <= 1.0)
            & utility_valid
            & (memory_input.provenance_id >= 0)
            & (memory_input.source_id >= 0)
        )
        transaction_required = (
            transaction_gate
            & jnp.asarray(next_armed, dtype=jnp.bool_)
            & memory_input.available
            & current_id_matches
            & next_id_matches
            & metadata_valid
        )
        query_version = jnp.where(
            transaction_required,
            destination_version,
            jnp.asarray(0, dtype=jnp.int32),
        )
        query_uncertainty = jnp.where(
            transaction_required,
            memory_input.query_uncertainty,
            jnp.asarray(0.0, dtype=jnp.float32),
        )
        query_uncertainty_available = (
            transaction_required & memory_input.query_uncertainty_available
        )
        safety_mask = jnp.where(
            transaction_required,
            memory_input.next_action_safety_mask,
            jnp.ones_like(memory_input.next_action_safety_mask),
        )
        proposal = policy.propose(
            memory_state,
            decision_representation,
            query_version,
            query_uncertainty,
            query_uncertainty_available,
            safety_mask,
        )
        entry = ExperientialMemoryEntry(
            observation=current_representation,
            key=current_representation,
            action=jax.nn.one_hot(
                executed_action,
                self._config.oak.n_primitive_actions,
                dtype=jnp.float32,
            ),
            outcome=jnp.concatenate(
                (
                    bootstrap_representation,
                    jnp.asarray(reward, dtype=jnp.float32)[None],
                )
            ),
            reward=jnp.asarray(reward, dtype=jnp.float32),
            uncertainty=memory_input.entry_uncertainty,
            uncertainty_available=memory_input.entry_uncertainty_available,
            safety_cost=memory_input.safety_cost,
            safety_cost_available=memory_input.safety_cost_available,
            reliability=memory_input.reliability,
            utility=memory_input.utility,
            utility_available=memory_input.utility_available,
            representation_version=(
                destination_version
                if feature_owned_versions
                else memory_input.entry_representation_version
            ),
            valid=jnp.asarray(True, dtype=jnp.bool_),
            age=jnp.asarray(0, dtype=jnp.int32),
            provenance_id=memory_input.provenance_id,
            source_id=memory_input.source_id,
        )

        def apply_step(_: None) -> ExperientialMemoryStepResult:
            return memory.step(
                memory_state,
                decision_representation,
                destination_version,
                memory_input.query_uncertainty,
                memory_input.query_uncertainty_available,
                entry,
            )

        def skip_step(_: None) -> ExperientialMemoryStepResult:
            return ExperientialMemoryStepResult(
                state=memory_state,
                retrieval=proposal.retrieval,
                wrote=jnp.asarray(False, dtype=jnp.bool_),
                slot=jnp.asarray(-1, dtype=jnp.int32),
                evicted=jnp.asarray(False, dtype=jnp.bool_),
                evicted_provenance_id=jnp.asarray(-1, dtype=jnp.int32),
            )

        step = cast(
            ExperientialMemoryStepResult,
            jax.lax.cond(
                transaction_required,
                apply_step,
                skip_step,
                operand=None,
            ),
        )
        retrieval_matches = _tree_arrays_equal(
            proposal.retrieval,
            step.retrieval,
        )
        transaction_applied = (
            transaction_required & step.wrote & retrieval_matches
        )
        counterfactual_action = oak_state.stomp_state.last_primitive_action
        advantage_gate_diagnostics = None
        proposal_authorized = proposal.available
        advantage_gate = self._experiential_memory_advantage_gate
        if advantage_gate is not None:
            advantage_gate_diagnostics = advantage_gate.assess(
                memory_state,
                proposal,
                counterfactual_action,
            )
            proposal_authorized = (
                advantage_gate_diagnostics.replacement_allowed
            )
        proposed_action = jnp.where(
            transaction_required & proposal_authorized,
            proposal.action,
            counterfactual_action,
        )
        replacement = replace_dispatched_primitive_action(
            oak_state.stomp_state,
            decision_representation,
            proposed_action,
            safety_action_mask=safety_mask,
        )
        transaction_applied = (
            transaction_applied & (~replacement.decision.failed_closed)
        )
        next_oak_state = cast(
            OaKState,
            oak_state.replace(stomp_state=replacement.state),
        )
        effective_action = jnp.where(
            next_armed,
            replacement.decision.effective_action,
            jnp.asarray(-1, dtype=jnp.int32),
        )
        diagnostics = PrototypeExperientialMemoryDiagnostics(
            proposal=proposal,
            advantage_gate=advantage_gate_diagnostics,
            dispatch_replacement=replacement.decision,
            input_supplied=input_supplied,
            input_available=memory_input.available,
            current_prototype_decision_id_matches=current_id_matches,
            next_prototype_decision_id_matches=next_id_matches,
            metadata_valid=metadata_valid,
            transaction_required=transaction_required,
            retrieval_matches=retrieval_matches,
            query_before_write=transaction_required & retrieval_matches,
            deterministic_prestate_query_count=jnp.where(
                transaction_required,
                jnp.asarray(2, dtype=jnp.int32),
                jnp.asarray(0, dtype=jnp.int32),
            ),
            wrote=step.wrote,
            slot=step.slot,
            evicted=step.evicted,
            evicted_provenance_id=step.evicted_provenance_id,
            transaction_applied=transaction_applied,
            counterfactual_base_action=counterfactual_action,
            effective_action=effective_action,
        )
        return step.state, next_oak_state, effective_action, diagnostics

    def _apply_partner_policy_fusion(
        self,
        interaction_slot: Any,
        ia_component_state: Any,
        oak_state: OaKState,
        decision_observation: Array,
        *,
        derived_decision_id: Array,
        derived_event_id: Array,
        derived_decision_words: Array,
        derived_event_words: Array,
        derived_prototype_decision_id: Array,
        next_armed: Array,
        transaction_allowed: Array,
        experiential_memory_state: ExperientialMemoryState | None,
        upstream_safety_action_mask: Array | None,
        decision_input: PrototypePartnerPolicyFusionInput,
        decision_input_supplied: Array,
        feedback: PrototypePartnerPolicyFusionFeedback,
        feedback_input_supplied: Array,
    ) -> tuple[Any, OaKState, Array, PrototypePartnerPolicyFusionDiagnostics]:
        """Resolve prior feedback, then fuse and bind one next dispatch.

        This method is called only inside the Prototype transition transaction.
        ``transaction_allowed=False`` gates both external surfaces to explicit
        no-ops while still producing the same fixed diagnostics PyTree required
        by ``lax.cond``.
        """

        fusion = self._partner_policy_fusion
        if fusion is None:
            raise RuntimeError("partner policy fusion is disabled")
        interaction = self._partner_interaction_state(interaction_slot)
        partner_state = self._partner_fusion_component_state(interaction_slot)
        transaction_gate = jnp.asarray(transaction_allowed, dtype=jnp.bool_)
        feedback_prototype_id_matches = (
            interaction.feedback_prototype_decision_id_available
            & jnp.array_equal(
                feedback.prototype_decision_id,
                interaction.feedback_prototype_decision_id,
            )
        )
        gated_feedback = cast(
            PartnerPolicyFusionFeedback,
            feedback.feedback.replace(
                available=(
                    feedback.feedback.available
                    & transaction_gate
                    & feedback_prototype_id_matches
                )
            ),
        )
        feedback_result = fusion.apply_feedback(partner_state, gated_feedback)
        pending_owner_available = (
            interaction.feedback_prototype_decision_id_available
            & (~feedback_result.applied)
        )
        pending_owner_id = jnp.where(
            pending_owner_available,
            interaction.feedback_prototype_decision_id,
            jnp.zeros((4,), dtype=jnp.uint32),
        )

        decision_prototype_id_matches = jnp.array_equal(
            decision_input.prototype_decision_id,
            derived_prototype_decision_id,
        )
        decision_gate = (
            transaction_gate
            & jnp.asarray(next_armed, dtype=jnp.bool_)
            & decision_input.available
            & decision_prototype_id_matches
        )
        safe_keyboard_vector = jnp.where(
            decision_gate & decision_input.keyboard_available,
            decision_input.keyboard_vector,
            jnp.zeros_like(decision_input.keyboard_vector),
        )
        keyboard = self._oak.propose_keyboard_policy(
            oak_state,
            decision_observation,
            safe_keyboard_vector,
        )
        option_proposal = OptionKeyboardProposal(
            available=(
                decision_gate
                & decision_input.keyboard_available
                & keyboard.available
            ),
            action=keyboard.action,
            declared_score=keyboard.declared_score,
        )
        # A missing/disabled sidecar deliberately invalidates only contextual
        # fusion numerics. The mechanism then returns the independently safe
        # OaK base action without advancing its decision or reliability state.
        fusion_context = jnp.where(
            decision_gate,
            decision_input.context_features,
            jnp.full_like(decision_input.context_features, jnp.nan),
        )
        combined_safety_mask = decision_input.safety_action_mask
        if upstream_safety_action_mask is not None:
            combined_safety_mask = (
                combined_safety_mask & upstream_safety_action_mask
            )
        fusion_safety_mask = jnp.where(
            decision_gate,
            combined_safety_mask,
            jnp.ones_like(decision_input.safety_action_mask),
        )
        base_action = oak_state.stomp_state.last_primitive_action
        base_score = self._oak_counterfactual_dispatch_score(
            oak_state,
            decision_observation,
        )
        decision_result = fusion.decide(
            feedback_result.state,
            decision_id=jnp.asarray(derived_decision_id, dtype=jnp.int32),
            event_id=jnp.asarray(derived_event_id, dtype=jnp.int32),
            decision_words=jnp.asarray(
                derived_decision_words,
                dtype=jnp.uint32,
            ),
            event_words=jnp.asarray(
                derived_event_words,
                dtype=jnp.uint32,
            ),
            observation_id=jnp.where(
                decision_gate,
                decision_input.observation_id,
                jnp.asarray(0, dtype=jnp.int32),
            ),
            context_id=jnp.where(
                decision_gate,
                decision_input.context_id,
                jnp.asarray(0, dtype=jnp.int32),
            ),
            context_features=fusion_context,
            base_action=base_action,
            base_declared_score=base_score,
            safety_action_mask=fusion_safety_mask,
            option_proposal=option_proposal,
            messages=decision_input.messages,
        )
        replacement = replace_dispatched_primitive_action(
            oak_state.stomp_state,
            decision_observation,
            decision_result.decision.effective_action,
            safety_action_mask=fusion_safety_mask,
        )
        next_oak_state = cast(
            OaKState,
            oak_state.replace(stomp_state=replacement.state),
        )
        next_owner_available = (
            pending_owner_available | decision_result.decision.feedback_armed
        )
        next_owner_id = jnp.where(
            decision_result.decision.feedback_armed,
            derived_prototype_decision_id,
            pending_owner_id,
        )
        next_interaction_state = self._interaction_slot(
            ia_component_state,
            decision_result.state,
            experiential_memory_state=experiential_memory_state,
            feedback_prototype_decision_id=next_owner_id,
            feedback_prototype_decision_id_available=next_owner_available,
        )
        effective_action = jnp.where(
            next_armed,
            replacement.decision.effective_action,
            jnp.asarray(-1, dtype=jnp.int32),
        )
        diagnostics = PrototypePartnerPolicyFusionDiagnostics(
            feedback=feedback_result,
            decision=decision_result.decision,
            keyboard_proposal=keyboard,
            dispatch_replacement=replacement.decision,
            feedback_input_supplied=feedback_input_supplied,
            decision_input_supplied=decision_input_supplied,
            feedback_prototype_decision_id_matches=(
                feedback_prototype_id_matches
            ),
            decision_prototype_decision_id_matches=(
                decision_prototype_id_matches
            ),
            transaction_applied=transaction_gate,
            counterfactual_base_action=base_action,
            effective_action=effective_action,
        )
        return (
            next_interaction_state,
            next_oak_state,
            effective_action,
            diagnostics,
        )

    # -- Dreaming scan (JIT-compiled as a method so the closure is stable) ----

    @functools.partial(jax.jit, static_argnums=(0,))
    def _run_dreams_with_count(
        self,
        oak_state: OaKState,
        wm_state: Any,
        buf_state: Any,
        rng_key: Array,
        extended_action_mask: Array | None = None,
    ) -> tuple[
        OaKState,
        Float[Array, " n_dreams"],
        Int[Array, ""],
    ]:
        n_prim = self._config.oak.n_primitive_actions
        action_mask = (
            jnp.ones(
                (self._config.oak.stomp.n_total_actions,),
                dtype=jnp.bool_,
            )
            if extended_action_mask is None
            else extended_action_mask
        )
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
        ) -> tuple[tuple[OaKState, Array], tuple[Float[Array, ""], Array]]:
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
                extended_action_mask=action_mask,
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
            return (new_oak_s, k), (
                td_err,
                dream_accepted.astype(jnp.int32),
            )

        (new_oak_state, _), (dream_td_errors, applied_updates) = jax.lax.scan(
            _dream_step,
            (oak_state, rng_key),
            jnp.arange(self._config.n_dreams_per_step),
        )
        return (
            new_oak_state,
            dream_td_errors,
            jnp.sum(applied_updates, dtype=jnp.int32),
        )

    def _run_dreams(
        self,
        oak_state: OaKState,
        wm_state: Any,
        buf_state: Any,
        rng_key: Array,
        extended_action_mask: Array | None = None,
    ) -> tuple[OaKState, Float[Array, " n_dreams"]]:
        """Compatibility wrapper that omits the internal applied-work count."""

        next_state, td_errors, _ = self._run_dreams_with_count(
            oak_state,
            wm_state,
            buf_state,
            rng_key,
            extended_action_mask,
        )
        return next_state, td_errors

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
        current_oak_state = self._oak_component_state(state.oak_state)
        feature_consumer_binding = (
            self._feature_consumer_binding(state.oak_state)
            if self._prototype_feature_lifecycle is not None
            else None
        )
        legacy_representation = current_oak_state.stomp_state.base_last_obs
        legacy_stomp_state = current_oak_state.stomp_state
        # The historical wrapper predates the explicit intra-option decision
        # owner cache. Reconstruct that one missing owner from the primitive
        # command STOMP actually dispatched; the authoritative transition API
        # never performs this repair and remains fail-closed on mismatches.
        legacy_stomp_state = legacy_stomp_state.replace(
            option_last_intra_action=jnp.where(
                legacy_stomp_state.executing_option >= 0,
                legacy_stomp_state.last_primitive_action,
                legacy_stomp_state.option_last_intra_action,
            )
        )
        legacy_oak_state = current_oak_state.replace(
            stomp_state=legacy_stomp_state,
        )
        legacy_state = cast(
            PrototypeAgentState,
            state.replace(
                oak_state=self._oak_state_slot(
                    legacy_oak_state,
                    feature_consumer_binding,
                    self._horde_component_state(state),
                    (
                        self._feature_utility_component_state(state.oak_state)
                        if self._prototype_feature_utility is not None
                        else None
                    ),
                ),
                current_raw_observation=legacy_representation[
                    : self._raw_observation_dim()
                ],
                current_representation=legacy_representation,
                current_action=current_oak_state.stomp_state.last_primitive_action,
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
        partner_input, partner_input_supplied = (
            self._normalize_partner_policy_fusion_input(None)
        )
        partner_feedback, partner_feedback_supplied = (
            self._normalize_partner_policy_fusion_feedback(None)
        )
        memory_input, memory_input_supplied = (
            self._normalize_experiential_memory_input(None)
        )
        return self._update_transition_impl(
            legacy_state,
            transition,
            diagnostics,
            control_discount=None,
            gradient_joy_evidence=None,
            gradient_joy_evidence_supplied=jnp.asarray(False),
            gradient_joy_decision_id_matches=jnp.asarray(False),
            experiential_memory_input=memory_input,
            experiential_memory_input_supplied=memory_input_supplied,
            partner_policy_fusion_input=partner_input,
            partner_policy_fusion_input_supplied=partner_input_supplied,
            partner_policy_fusion_feedback=partner_feedback,
            partner_policy_fusion_feedback_supplied=partner_feedback_supplied,
        )

    def update_transition(
        self,
        state: PrototypeAgentState,
        transition: PrototypeTransition,
        candidate_update_audit_evidence: (
            PrototypeCandidateUpdateAuditEvidence | None
        ) = None,
        *,
        gradient_joy_evidence: PrototypeGradientJoyEvidence | None = None,
        experiential_memory_input: (
            PrototypeExperientialMemoryInput | None
        ) = None,
        partner_policy_fusion_input: (
            PrototypePartnerPolicyFusionInput | None
        ) = None,
        partner_policy_fusion_feedback: (
            PrototypePartnerPolicyFusionFeedback | None
        ) = None,
        extended_action_mask: Array | None = None,
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
            candidate_update_audit_evidence: Canonical optional representation-
                update audit sidecar. The historical ``gradient_joy_evidence``
                keyword is accepted only when this argument is absent.

        Returns:
            :class:`PrototypeUpdateResult` with updated state, selected action,
            and per-component diagnostics.
        """
        if (
            candidate_update_audit_evidence is not None
            and gradient_joy_evidence is not None
        ):
            raise ValueError(
                "candidate_update_audit_evidence and gradient_joy_evidence "
                "cannot both be supplied"
            )
        selected_audit_evidence = (
            candidate_update_audit_evidence
            if candidate_update_audit_evidence is not None
            else gradient_joy_evidence
        )
        normalized_audit_evidence, audit_supplied, audit_decision_matches = (
            self._normalize_candidate_update_audit_evidence(
                state,
                selected_audit_evidence,
            )
        )
        normalized_memory_input, memory_input_supplied = (
            self._normalize_experiential_memory_input(
                experiential_memory_input
            )
        )
        normalized_partner_input, partner_input_supplied = (
            self._normalize_partner_policy_fusion_input(
                partner_policy_fusion_input
            )
        )
        normalized_partner_feedback, partner_feedback_supplied = (
            self._normalize_partner_policy_fusion_feedback(
                partner_policy_fusion_feedback
            )
        )
        normalized, diagnostics = self._normalize_transition(state, transition)
        action_mask = (
            None
            if extended_action_mask is None
            else _strict_bool_array(
                extended_action_mask,
                (self._config.oak.stomp.n_total_actions,),
                name="extended_action_mask",
            )
        )
        return self._update_transition_impl(
            state,
            normalized,
            diagnostics,
            control_discount=normalized.discount,
            gradient_joy_evidence=normalized_audit_evidence,
            gradient_joy_evidence_supplied=audit_supplied,
            gradient_joy_decision_id_matches=audit_decision_matches,
            experiential_memory_input=normalized_memory_input,
            experiential_memory_input_supplied=memory_input_supplied,
            partner_policy_fusion_input=normalized_partner_input,
            partner_policy_fusion_input_supplied=partner_input_supplied,
            partner_policy_fusion_feedback=normalized_partner_feedback,
            partner_policy_fusion_feedback_supplied=(
                partner_feedback_supplied
            ),
            extended_action_mask=action_mask,
        )

    def assess_transition(
        self,
        state: PrototypeAgentState,
        transition: PrototypeTransition,
    ) -> PrototypeTransitionDiagnostics:
        """Read-only ownership and boundary assessment for one transition.

        This exposes the same exact pre-transaction validation used by
        :meth:`update_transition` without advancing learners, counters, RNG,
        optional sidecars, or the cached dispatch.
        """

        _, diagnostics = self._normalize_transition(state, transition)
        return diagnostics

    def _prepare_rtu_transition_record(
        self,
        state: PrototypeAgentState,
        transition: PrototypeTransition,
    ) -> PrototypeRTUTransitionPreparation:
        """Build the pure recurrence-only half of the RTU transition seam."""

        if type(self._state_builder) is not RecurrentTraceUnitStateBuilder:
            raise ValueError(
                "RTU transition preparation requires an exact RTU state builder"
            )
        normalized, diagnostics = self._normalize_transition(state, transition)
        source_builder = cast(
            RecurrentTraceUnitStateBuilderState,
            self._builder_component_state(state.state_builder_state),
        )
        bootstrap = self._state_builder.update_with_status(
            source_builder,
            normalized.next_observation,
            normalized.action,
            normalized.reward,
            normalized.discount,
        )
        boundary = normalized.terminated | normalized.truncated
        reset_builder = self._state_builder.reset_episode(bootstrap.state)
        restart = self._state_builder.update_with_status(
            reset_builder,
            normalized.next_decision_observation,
            jnp.asarray(-1, dtype=jnp.int32),
            jnp.asarray(0.0, dtype=jnp.float32),
            jnp.asarray(1.0, dtype=jnp.float32),
        )
        decision_builder = cast(
            RecurrentTraceUnitStateBuilderState,
            jax.lax.cond(
                boundary,
                lambda: restart.state,
                lambda: bootstrap.state,
            ),
        )
        decision_representation = self._state_builder.encode(
            decision_builder,
            normalized.next_decision_observation,
        )
        preparation_valid = (
            diagnostics.valid
            & bootstrap.transition_applied
            & jnp.where(
                boundary,
                restart.transition_applied,
                jnp.asarray(True, dtype=jnp.bool_),
            )
            & self._state_builder.state_valid(decision_builder)
            & jnp.all(jnp.isfinite(decision_representation))
        )
        return PrototypeRTUTransitionPreparation(
            source_state=state,
            transition=normalized,
            transition_diagnostics=diagnostics,
            source_builder_state=source_builder,
            bootstrap_transition=bootstrap,
            decision_builder_state=decision_builder,
            decision_representation=decision_representation,
            execution_boundary=boundary,
            preparation_valid=preparation_valid,
        )

    def prepare_rtu_transition(
        self,
        state: PrototypeAgentState,
        transition: PrototypeTransition,
    ) -> PrototypeRTUTransitionPreparation:
        """Advance only RTU recurrence; do not learn, select, or draw RNG."""

        if type(state) is not PrototypeAgentState:
            raise TypeError("state must be an exact PrototypeAgentState")
        if type(transition) is not PrototypeTransition:
            raise TypeError("transition must be an exact PrototypeTransition")
        return self._prepare_rtu_transition_record(state, transition)

    def bind_rtu_finalization(
        self,
        preparation: PrototypeRTUTransitionPreparation,
        final_builder_state: RecurrentTraceUnitStateBuilderState,
        replaced_unit_mask: Array,
        rtu_proposal: RTUGenerateAndTestProposal,
    ) -> PrototypeRTUFinalizationReceipt:
        """Content-bind one RTU proposal and claimed derived destination.

        Binding is not authorization.  :meth:`finalize_rtu_transition` accepts
        the receipt only after the supplied RTU mechanism independently
        recomputes the proposal and its committed destination.
        """

        if type(preparation) is not PrototypeRTUTransitionPreparation:
            raise TypeError(
                "preparation must be an exact PrototypeRTUTransitionPreparation"
            )
        if type(final_builder_state) is not RecurrentTraceUnitStateBuilderState:
            raise TypeError(
                "final_builder_state must be an exact RTU builder state"
            )
        if type(rtu_proposal) is not RTUGenerateAndTestProposal:
            raise TypeError("rtu_proposal must be an exact RTU proposal")
        if type(self._state_builder) is not RecurrentTraceUnitStateBuilder:
            raise ValueError(
                "RTU finalization binding requires an exact RTU state builder"
            )
        builder = self._state_builder
        builder.state_valid(final_builder_state)
        mask = jnp.asarray(replaced_unit_mask)
        expected_shape = (builder.config.hidden_dim,)
        if mask.shape != expected_shape:
            raise ValueError(
                f"replaced_unit_mask must have shape {expected_shape}, got {mask.shape}"
            )
        if mask.dtype != jnp.bool_:
            raise TypeError(
                f"replaced_unit_mask must have dtype bool, got {mask.dtype}"
            )
        return PrototypeRTUFinalizationReceipt(
            preparation=preparation,
            rtu_proposal=rtu_proposal,
            final_builder_state=final_builder_state,
            replaced_unit_mask=mask,
            content_tag_words=_prototype_rtu_destination_tag(
                preparation,
                rtu_proposal,
                final_builder_state,
                mask,
            ),
        )

    def finalize_rtu_transition(
        self,
        state: PrototypeAgentState,
        transition: PrototypeTransition,
        receipt: PrototypeRTUFinalizationReceipt,
        rtu_generate_and_test: RTUGenerateAndTest,
    ) -> PrototypeUpdateResult:
        """Recompute RTU provenance, then learn/select once from its destination.

        The supplied RTU mechanism and proposal prove deterministic derivation
        from their supplied lifecycle source.  They do not authenticate an
        external caller's authority over that lifecycle source, downstream
        objective/gradient, or source-bound ordinary learning proposal.  The
        comprehensive live adapter closes all three boundaries by constructing
        the gradient and proposal internally and exact-matching its owned
        lifecycle source before accepting the result.
        """

        if type(receipt) is not PrototypeRTUFinalizationReceipt:
            raise TypeError(
                "receipt must be an exact PrototypeRTUFinalizationReceipt"
            )
        if type(rtu_generate_and_test) is not RTUGenerateAndTest:
            raise TypeError(
                "rtu_generate_and_test must be an exact RTUGenerateAndTest"
            )
        if type(self._state_builder) is not RecurrentTraceUnitStateBuilder:
            raise ValueError(
                "RTU transition finalization requires an exact RTU state builder"
            )
        if rtu_generate_and_test.config.builder != self._state_builder.config:
            raise ValueError(
                "RTU finalization mechanism builder must exactly match Prototype"
            )
        expected = self._prepare_rtu_transition_record(state, transition)
        preparation_matches = _prototype_rtu_tree_exact(
            expected,
            receipt.preparation,
        )
        expected_tag = _prototype_rtu_destination_tag(
            receipt.preparation,
            receipt.rtu_proposal,
            receipt.final_builder_state,
            receipt.replaced_unit_mask,
        )
        tag_matches = jnp.array_equal(
            receipt.content_tag_words,
            expected_tag,
        )
        builder = self._state_builder
        expected_advance_receipt = rtu_generate_and_test.make_advance_receipt(
            expected.source_builder_state,
            bootstrap_observation=expected.transition.next_observation,
            previous_action=expected.transition.action,
            previous_reward=expected.transition.reward,
            previous_discount=expected.transition.discount,
            episode_boundary=expected.execution_boundary,
            restart_observation=expected.transition.next_decision_observation,
        )
        proposal_source_builder_matches = _prototype_rtu_tree_exact(
            receipt.rtu_proposal.pre_update_builder_state,
            expected.source_builder_state,
        )
        if receipt.rtu_proposal.advance_receipt is None:
            advance_receipt_matches = jnp.asarray(False, dtype=jnp.bool_)
        else:
            advance_receipt_matches = _prototype_rtu_tree_exact(
                receipt.rtu_proposal.advance_receipt,
                expected_advance_receipt,
            )
        ordinary_learning_present = (
            receipt.rtu_proposal.learning_proposal is not None
            and receipt.rtu_proposal.ordinary_learning_diagnostics is not None
        )
        authorized_rtu_result = rtu_generate_and_test.commit(
            receipt.rtu_proposal.source_state,
            receipt.rtu_proposal.live_builder_state,
            receipt.rtu_proposal,
        )
        authorized_destination_matches = _prototype_rtu_tree_exact(
            receipt.final_builder_state,
            authorized_rtu_result.builder_state,
        )
        authorized_mask_matches = jnp.array_equal(
            receipt.replaced_unit_mask,
            authorized_rtu_result.diagnostics.selected_mask,
        )
        final_builder_valid = builder.state_valid(receipt.final_builder_state)
        step_owner_matches = jnp.array_equal(
            receipt.final_builder_state.step_words,
            expected.decision_builder_state.step_words,
        )
        ordinary_update_words, ordinary_capacity = _checked_lifetime_words_add(
            expected.source_builder_state.update_words,
            1,
        )
        replacement_update_words, replacement_capacity = (
            _checked_lifetime_words_add(ordinary_update_words, 1)
        )
        replacement_event = jnp.any(receipt.replaced_unit_mask)
        expected_update_words = jnp.where(
            replacement_event,
            replacement_update_words,
            ordinary_update_words,
        )
        revision_matches = jnp.array_equal(
            receipt.final_builder_state.update_words,
            expected_update_words,
        )
        source_oak = self._oak_component_state(state.oak_state)
        active_option_safe = (~replacement_event) | (
            source_oak.stomp_state.executing_option < 0
        )
        authorization = (
            expected.preparation_valid
            & preparation_matches
            & tag_matches
            & proposal_source_builder_matches
            & advance_receipt_matches
            & jnp.asarray(ordinary_learning_present, dtype=jnp.bool_)
            & authorized_rtu_result.diagnostics.applied
            & authorized_destination_matches
            & authorized_mask_matches
            & final_builder_valid
            & step_owner_matches
            & revision_matches
            & ordinary_capacity
            & jnp.where(
                replacement_event,
                replacement_capacity,
                jnp.asarray(True, dtype=jnp.bool_),
            )
            & active_option_safe
        )
        diagnostics = cast(
            PrototypeTransitionDiagnostics,
            expected.transition_diagnostics.replace(
                valid=expected.transition_diagnostics.valid & authorization,
                rejected=~(
                    expected.transition_diagnostics.valid & authorization
                ),
            ),
        )
        raw_prefix = (
            builder.config.observation_dim
            if builder.config.include_raw_observation
            else 0
        )
        feature_reset_mask = jnp.concatenate(
            (
                jnp.zeros((raw_prefix,), dtype=jnp.bool_),
                receipt.replaced_unit_mask,
                receipt.replaced_unit_mask,
            )
        )
        return self._update_transition_impl(
            state,
            expected.transition,
            diagnostics,
            control_discount=expected.transition.discount,
            gradient_joy_evidence=None,
            gradient_joy_evidence_supplied=jnp.asarray(False, dtype=jnp.bool_),
            gradient_joy_decision_id_matches=jnp.asarray(False, dtype=jnp.bool_),
            experiential_memory_input=None,
            experiential_memory_input_supplied=jnp.asarray(False, dtype=jnp.bool_),
            partner_policy_fusion_input=None,
            partner_policy_fusion_input_supplied=jnp.asarray(False, dtype=jnp.bool_),
            partner_policy_fusion_feedback=None,
            partner_policy_fusion_feedback_supplied=jnp.asarray(False, dtype=jnp.bool_),
            rtu_preparation=expected,
            external_final_builder_state=receipt.final_builder_state,
            preselection_feature_reset_mask=feature_reset_mask,
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
        proposed_step_words, step_capacity_available = (
            _checked_lifetime_words_add(state.step_words, 1)
        )
        proposed_observation_one, observation_one_capacity = (
            _checked_lifetime_words_add(state.observation_event_words, 1)
        )
        proposed_observation_two, observation_two_capacity = (
            _checked_lifetime_words_add(state.observation_event_words, 2)
        )
        proposed_observation_event_words = jnp.where(
            boundary,
            proposed_observation_two,
            proposed_observation_one,
        )
        current_counter_capacity_available = step_capacity_available & jnp.where(
            boundary,
            observation_two_capacity,
            observation_one_capacity,
        )
        outer_counter_valid = _lifetime_counter_valid(
            state.step_words,
            state.step_count,
        ) & _lifetime_counter_valid(
            state.observation_event_words,
            state.observation_event_count,
        ) & _prototype_observation_clock_relation_valid(
            state.step_words,
            state.observation_event_words,
        )
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
            pre_step_words=state.step_words,
            proposed_step_words=proposed_step_words,
            pre_observation_event_words=state.observation_event_words,
            proposed_observation_event_words=(
                proposed_observation_event_words
            ),
            outer_counter_valid=outer_counter_valid,
            current_counter_capacity_available=(
                current_counter_capacity_available
            ),
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
        gradient_joy_evidence: PrototypeCandidateUpdateAuditEvidence | None,
        gradient_joy_evidence_supplied: Array,
        gradient_joy_decision_id_matches: Array,
        experiential_memory_input: PrototypeExperientialMemoryInput | None,
        experiential_memory_input_supplied: Array,
        partner_policy_fusion_input: PrototypePartnerPolicyFusionInput | None,
        partner_policy_fusion_input_supplied: Array,
        partner_policy_fusion_feedback: (
            PrototypePartnerPolicyFusionFeedback | None
        ),
        partner_policy_fusion_feedback_supplied: Array,
    ) -> PrototypeUpdateResult:
        """Return neutral diagnostics and preserve ``state`` bit-for-bit."""
        zero = jnp.array(0.0, dtype=jnp.float32)
        world_model_error: Any = (
            zero
            if (
                self._prototype_routed_linear_world_model is not None
                or self._world_model is not None
                or self._world_model_ensemble is not None
                or self._model_replay_rehearsal is not None
                or self._recurrent_latent_world_model_ensemble is not None
            )
            else None
        )
        dream_td_errors: Any = None
        if self._world_model is not None and self._config.n_dreams_per_step > 0:
            dream_td_errors = jnp.zeros(
                (self._config.n_dreams_per_step,),
                dtype=jnp.float32,
            )
        option_search_diagnostics: OptionSearchControlDiagnostics | None = None
        if self._option_search_control is not None:
            option_search_diagnostics = (
                self._option_search_control.unavailable_diagnostics(
                    state.current_representation
                )
            )
        feature_lifecycle_diagnostics: (
            PrototypeFeatureLifecycleIntegrationDiagnostics | None
        ) = None
        if self._prototype_feature_lifecycle is not None:
            feature_lifecycle_diagnostics = (
                self._unavailable_feature_lifecycle_diagnostics(
                    self._feature_lifecycle_component_state(
                        state.state_builder_state
                    )
                )
            )
        atomic_feature_world_memory_diagnostics: (
            PrototypeAtomicFeatureWorldMemoryDiagnostics | None
        ) = None
        if self._prototype_routed_linear_world_model is not None:
            atomic_feature_world_memory_diagnostics = (
                self._unavailable_atomic_feature_world_memory_diagnostics()
            )
        feature_utility_diagnostics: (
            PrototypeFeatureUtilityIntegrationDiagnostics | None
        ) = None
        if self._prototype_feature_utility is not None:
            feature_utility_diagnostics = (
                self._unavailable_feature_utility_integration_diagnostics(
                    self._feature_utility_component_state(state.oak_state)
                )
            )
        feature_utility_curation_diagnostics: (
            PrototypeFeatureUtilityCurationIntegrationDiagnostics | None
        ) = None
        if self._prototype_feature_utility_curation is not None:
            feature_utility_curation_diagnostics = (
                self._unavailable_feature_utility_curation_integration_diagnostics(
                    self._feature_lifecycle_component_state(
                        state.state_builder_state
                    ),
                    self._feature_utility_component_state(state.oak_state),
                )
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
        memory_diagnostics: PrototypeExperientialMemoryDiagnostics | None = None
        feature_memory_diagnostics: (
            PrototypeFeatureMemoryIntegrationDiagnostics | None
        ) = None
        current_memory_state: ExperientialMemoryState | None = None
        if self._experiential_memory is not None:
            if experiential_memory_input is None:
                raise RuntimeError(
                    "configured experiential memory requires a fixed sidecar"
                )
            current_memory_state = self._experiential_memory_component_state(
                state.ia_state
            )
            (
                _,
                _,
                _,
                memory_diagnostics,
            ) = self._apply_experiential_memory(
                current_memory_state,
                self._oak_component_state(state.oak_state),
                state.current_representation,
                state.current_representation,
                state.current_representation,
                state.current_action,
                zero,
                current_prototype_decision_id=state.current_decision_id,
                next_prototype_decision_id=_increment_decision_id(
                    state.current_decision_id
                ),
                next_armed=jnp.asarray(False, dtype=jnp.bool_),
                transaction_allowed=jnp.asarray(False, dtype=jnp.bool_),
                memory_input=experiential_memory_input,
                input_supplied=experiential_memory_input_supplied,
            )
            if self._prototype_feature_memory is not None:
                feature_memory_diagnostics = (
                    self._unavailable_feature_memory_integration_diagnostics(
                        state,
                        experiential_memory_input,
                    )
                )
        partner_fusion_diagnostics: (
            PrototypePartnerPolicyFusionDiagnostics | None
        ) = None
        if self._partner_policy_fusion is not None:
            if (
                partner_policy_fusion_input is None
                or partner_policy_fusion_feedback is None
            ):
                raise RuntimeError("configured partner fusion requires fixed sidecars")
            (
                _,
                _,
                _,
                partner_fusion_diagnostics,
            ) = self._apply_partner_policy_fusion(
                state.ia_state,
                self._ia_component_state(state.ia_state),
                self._oak_component_state(state.oak_state),
                state.current_representation,
                derived_decision_id=_lifetime_words_to_int32_telemetry(
                    diagnostics.proposed_step_words
                ),
                derived_event_id=_lifetime_words_to_int32_telemetry(
                    diagnostics.proposed_observation_event_words
                ),
                derived_decision_words=diagnostics.proposed_step_words,
                derived_event_words=(
                    diagnostics.proposed_observation_event_words
                ),
                derived_prototype_decision_id=_increment_decision_id(
                    state.current_decision_id
                ),
                next_armed=jnp.asarray(False, dtype=jnp.bool_),
                transaction_allowed=jnp.asarray(False, dtype=jnp.bool_),
                experiential_memory_state=current_memory_state,
                upstream_safety_action_mask=None,
                decision_input=partner_policy_fusion_input,
                decision_input_supplied=partner_policy_fusion_input_supplied,
                feedback=partner_policy_fusion_feedback,
                feedback_input_supplied=(
                    partner_policy_fusion_feedback_supplied
                ),
            )
        learning_value_router_result: LearningValueRouterResult | None = None
        if self._learning_value_router is not None:
            learning_value_router_result = (
                self._learning_value_router.unavailable_result(
                    self._learning_value_router_component_state(
                        state.state_builder_state
                    )
                )
            )
        (
            _,
            builder_learning_diagnostics,
            gradient_joy_application,
        ) = self._apply_state_builder_learning(
            self._builder_component_state(state.state_builder_state),
            self._builder_component_state(state.state_builder_state),
            jnp.zeros(
                (self._base_representation_dim(),),
                dtype=jnp.float32,
            ),
            jnp.asarray(False),
            _unavailable_learning_signals(),
            gradient_joy_evidence,
            learning_value_router_result,
        )
        return PrototypeUpdateResult(
            state=state,
            action=state.current_action,
            oak_td_error=zero,
            oak_average_reward=zero,
            oak_stomp_update_result=_unavailable_stomp_update_result(
                self._oak_component_state(state.oak_state)
            ),
            oak_stomp_update_available=jnp.asarray(False, dtype=jnp.bool_),
            oak_stomp_update_evaluations=jnp.asarray(0, dtype=jnp.int32),
            oak_owner_finalization_trace=(
                _unavailable_stomp_owner_finalization_trace(
                    self._oak_component_state(state.oak_state)
                )
            ),
            oak_real_stomp_update_evaluations=jnp.asarray(
                0,
                dtype=jnp.int32,
            ),
            oak_imagined_stomp_update_evaluations=jnp.asarray(
                0,
                dtype=jnp.int32,
            ),
            oak_total_stomp_update_evaluations=jnp.asarray(
                0,
                dtype=jnp.int32,
            ),
            oak_option_search_learner_updates=jnp.asarray(
                0,
                dtype=jnp.int32,
            ),
            oak_bootstrap_observation=state.current_representation,
            oak_decision_observation=state.current_representation,
            oak_execution_boundary=jnp.asarray(False, dtype=jnp.bool_),
            world_model_error=world_model_error,
            learning_signals=_unavailable_learning_signals(),
            learning_value_router_result=learning_value_router_result,
            world_model_representation_gradient=jnp.zeros(
                (self._config.oak.observation_dim,),
                dtype=jnp.float32,
            ),
            world_model_representation_gradient_valid=jnp.asarray(False),
            behavior_gradient_result=_unavailable_behavior_gradient(
                self._config.oak.observation_dim
            ),
            representation_gradient_mix=_unavailable_representation_gradient_mix(
                self._config.oak.observation_dim
            ),
            world_model_ensemble_diagnostics=_unavailable_ensemble_diagnostics(),
            recurrent_latent_world_model_diagnostics=(
                _unavailable_recurrent_latent_diagnostics(

                        self._config.recurrent_latent_world_model_ensemble.ensemble_size
                        if self._config.recurrent_latent_world_model_ensemble
                        is not None
                        else 0

                )
            ),
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
            option_search_control_diagnostics=option_search_diagnostics,
            prototype_feature_lifecycle_diagnostics=(
                feature_lifecycle_diagnostics
            ),
            prototype_feature_utility_diagnostics=(
                feature_utility_diagnostics
            ),
            prototype_feature_utility_curation_diagnostics=(
                feature_utility_curation_diagnostics
            ),
            dream_td_errors=dream_td_errors,
            horde_td_errors=horde_td_errors,
            ia_augmented_obs=ia_augmented_obs,
            ia_recommendation=ia_recommendation,
            ia_update_applied=jnp.asarray(False, dtype=jnp.bool_),
            experiential_memory_diagnostics=memory_diagnostics,
            prototype_feature_memory_diagnostics=feature_memory_diagnostics,
            partner_policy_fusion_diagnostics=partner_fusion_diagnostics,
            transition_diagnostics=diagnostics,
            prototype_atomic_feature_world_memory_diagnostics=(
                atomic_feature_world_memory_diagnostics
            ),
        )

    def _update_transition_impl(
        self,
        state: PrototypeAgentState,
        transition: PrototypeTransition,
        diagnostics: PrototypeTransitionDiagnostics,
        *,
        control_discount: Array | None,
        gradient_joy_evidence: PrototypeCandidateUpdateAuditEvidence | None,
        gradient_joy_evidence_supplied: Array,
        gradient_joy_decision_id_matches: Array,
        experiential_memory_input: PrototypeExperientialMemoryInput | None,
        experiential_memory_input_supplied: Array,
        partner_policy_fusion_input: PrototypePartnerPolicyFusionInput | None,
        partner_policy_fusion_input_supplied: Array,
        partner_policy_fusion_feedback: (
            PrototypePartnerPolicyFusionFeedback | None
        ),
        partner_policy_fusion_feedback_supplied: Array,
        rtu_preparation: PrototypeRTUTransitionPreparation | None = None,
        external_final_builder_state: (
            RecurrentTraceUnitStateBuilderState | None
        ) = None,
        preselection_feature_reset_mask: Array | None = None,
        extended_action_mask: Array | None = None,
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
                experiential_memory_input=experiential_memory_input,
                experiential_memory_input_supplied=(
                    experiential_memory_input_supplied
                ),
                partner_policy_fusion_input=partner_policy_fusion_input,
                partner_policy_fusion_input_supplied=(
                    partner_policy_fusion_input_supplied
                ),
                partner_policy_fusion_feedback=partner_policy_fusion_feedback,
                partner_policy_fusion_feedback_supplied=(
                    partner_policy_fusion_feedback_supplied
                ),
                rtu_preparation=rtu_preparation,
                external_final_builder_state=external_final_builder_state,
                preselection_feature_reset_mask=(
                    preselection_feature_reset_mask
                ),
                extended_action_mask=extended_action_mask,
            )
            post_finite = self._state_numeric_valid(result.state)
            post_consistent = self._checkpoint_state_structurally_valid(
                result.state
            )
            recurrent_transaction_valid = jnp.asarray(True, dtype=jnp.bool_)
            if self._recurrent_latent_world_model_ensemble is not None:
                recurrent_transaction_valid = (
                    result.recurrent_latent_world_model_diagnostics.transaction_applied
                )
            memory_transaction_valid = jnp.asarray(True, dtype=jnp.bool_)
            if self._experiential_memory is not None:
                memory_diagnostics = result.experiential_memory_diagnostics
                if memory_diagnostics is None:
                    raise RuntimeError(
                        "configured experiential memory requires diagnostics"
                    )
                memory_transaction_valid = (
                    (~memory_diagnostics.transaction_required)
                    | memory_diagnostics.transaction_applied
                )
            feature_memory_transaction_valid = jnp.asarray(
                True,
                dtype=jnp.bool_,
            )
            if self._prototype_feature_memory is not None:
                feature_memory_diagnostics = (
                    result.prototype_feature_memory_diagnostics
                )
                if feature_memory_diagnostics is None:
                    raise RuntimeError(
                        "configured feature memory requires diagnostics"
                    )
                rebind = feature_memory_diagnostics.rebind
                change_requested = (
                    rebind.destination_descriptors_changed
                    | rebind.destination_generation_changed
                )
                feature_memory_transaction_valid = (
                    jnp.where(
                        change_requested,
                        rebind.transaction_applied,
                        rebind.transaction_noop,
                    )
                    & feature_memory_diagnostics.post_memory_state_valid
                )
            ia_transaction_valid = jnp.asarray(True, dtype=jnp.bool_)
            if self._ia is not None:
                ia_transaction_valid = result.ia_update_applied
            feature_transaction_valid = jnp.asarray(True, dtype=jnp.bool_)
            if self._prototype_feature_lifecycle is not None:
                feature_diagnostics = (
                    result.prototype_feature_lifecycle_diagnostics
                )
                if feature_diagnostics is None:
                    raise RuntimeError(
                        "configured prototype feature lifecycle requires diagnostics"
                    )
                feature_transaction_valid = (
                    feature_diagnostics.lifecycle.transaction_applied
                    | ~feature_diagnostics.lifecycle.update_capacity_available
                )
            feature_utility_curation_transaction_valid = jnp.asarray(
                True,
                dtype=jnp.bool_,
            )
            if self._prototype_feature_utility_curation is not None:
                curation_diagnostics = (
                    result.prototype_feature_utility_curation_diagnostics
                )
                if curation_diagnostics is None:
                    raise RuntimeError(
                        "configured feature utility curation requires diagnostics"
                    )
                feature_utility_curation_transaction_valid = (
                    curation_diagnostics.policy.transaction_valid
                    & (
                        curation_diagnostics.observation_applied
                        | curation_diagnostics.policy.observation_capacity_capped
                    )
                )
            # Recurrent state cannot skip an accepted environment event. If
            # the model/signal transaction rejects, roll back the *entire*
            # Prototype transition so the armed cache and every learner/RNG
            # remain synchronized with the still-unconsumed environment event.
            post_valid = (
                post_finite
                & post_consistent
                & recurrent_transaction_valid
                & memory_transaction_valid
                & feature_memory_transaction_valid
                & ia_transaction_valid
                & feature_transaction_valid
                & feature_utility_curation_transaction_valid
            )
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
            accepted_feature_diagnostics = (
                result.prototype_feature_lifecycle_diagnostics
            )
            if accepted_feature_diagnostics is not None:
                accepted_feature_diagnostics = cast(
                    PrototypeFeatureLifecycleIntegrationDiagnostics,
                    accepted_feature_diagnostics.replace(
                        outer_transaction_committed=jnp.asarray(
                            True,
                            dtype=jnp.bool_,
                        )
                    ),
                )
            accepted_utility_diagnostics = (
                result.prototype_feature_utility_diagnostics
            )
            if accepted_utility_diagnostics is not None:
                accepted_utility_diagnostics = cast(
                    PrototypeFeatureUtilityIntegrationDiagnostics,
                    accepted_utility_diagnostics.replace(
                        outer_transaction_committed=jnp.asarray(
                            True,
                            dtype=jnp.bool_,
                        )
                    ),
                )
            accepted_utility_curation_diagnostics = (
                result.prototype_feature_utility_curation_diagnostics
            )
            if accepted_utility_curation_diagnostics is not None:
                accepted_utility_curation_diagnostics = cast(
                    PrototypeFeatureUtilityCurationIntegrationDiagnostics,
                    accepted_utility_curation_diagnostics.replace(
                        outer_transaction_committed=jnp.asarray(
                            True,
                            dtype=jnp.bool_,
                        )
                    ),
                )
            accepted_feature_memory_diagnostics = (
                result.prototype_feature_memory_diagnostics
            )
            if accepted_feature_memory_diagnostics is not None:
                accepted_feature_memory_diagnostics = cast(
                    PrototypeFeatureMemoryIntegrationDiagnostics,
                    accepted_feature_memory_diagnostics.replace(
                        outer_transaction_committed=jnp.asarray(
                            True,
                            dtype=jnp.bool_,
                        )
                    ),
                )
            accepted = cast(
                PrototypeUpdateResult,
                result.replace(
                    transition_diagnostics=final_diagnostics,
                    prototype_feature_lifecycle_diagnostics=(
                        accepted_feature_diagnostics
                    ),
                    prototype_feature_utility_diagnostics=(
                        accepted_utility_diagnostics
                    ),
                    prototype_feature_utility_curation_diagnostics=(
                        accepted_utility_curation_diagnostics
                    ),
                    prototype_feature_memory_diagnostics=(
                        accepted_feature_memory_diagnostics
                    ),
                ),
            )

            def rollback_rejected_recurrent_or_postcheck(
                __: None,
            ) -> PrototypeUpdateResult:
                rejected = self._rejected_update_result(
                    state,
                    final_diagnostics,
                    gradient_joy_evidence=gradient_joy_evidence,
                    gradient_joy_evidence_supplied=(
                        gradient_joy_evidence_supplied
                    ),
                    gradient_joy_decision_id_matches=(
                        gradient_joy_decision_id_matches
                    ),
                    experiential_memory_input=experiential_memory_input,
                    experiential_memory_input_supplied=(
                        experiential_memory_input_supplied
                    ),
                    partner_policy_fusion_input=partner_policy_fusion_input,
                    partner_policy_fusion_input_supplied=(
                        partner_policy_fusion_input_supplied
                    ),
                    partner_policy_fusion_feedback=(
                        partner_policy_fusion_feedback
                    ),
                    partner_policy_fusion_feedback_supplied=(
                        partner_policy_fusion_feedback_supplied
                    ),
                )
                if self._recurrent_latent_world_model_ensemble is not None:
                    rejected = cast(
                        PrototypeUpdateResult,
                        rejected.replace(
                            recurrent_latent_world_model_diagnostics=(
                                result.recurrent_latent_world_model_diagnostics
                            )
                        ),
                    )
                if self._partner_policy_fusion is not None:
                    attempted_partner = (
                        result.partner_policy_fusion_diagnostics
                    )
                    if attempted_partner is None:
                        raise RuntimeError(
                            "configured partner fusion requires diagnostics"
                        )
                    rejected = cast(
                        PrototypeUpdateResult,
                        rejected.replace(
                            partner_policy_fusion_diagnostics=(
                                attempted_partner.replace(
                                    transaction_applied=jnp.asarray(
                                        False,
                                        dtype=jnp.bool_,
                                    )
                                )
                            )
                        ),
                    )
                if self._experiential_memory is not None:
                    attempted_memory = result.experiential_memory_diagnostics
                    if attempted_memory is None:
                        raise RuntimeError(
                            "configured experiential memory requires diagnostics"
                        )
                    rejected = cast(
                        PrototypeUpdateResult,
                        rejected.replace(
                            experiential_memory_diagnostics=(
                                attempted_memory.replace(
                                    transaction_applied=jnp.asarray(
                                        False,
                                        dtype=jnp.bool_,
                                    )
                                )
                            )
                        ),
                    )
                if self._prototype_feature_lifecycle is not None:
                    attempted_feature = (
                        result.prototype_feature_lifecycle_diagnostics
                    )
                    if attempted_feature is None:
                        raise RuntimeError(
                            "configured prototype feature lifecycle requires "
                            "diagnostics"
                        )
                    rejected = cast(
                        PrototypeUpdateResult,
                        rejected.replace(
                            prototype_feature_lifecycle_diagnostics=(
                                attempted_feature.replace(
                                    outer_transaction_committed=jnp.asarray(
                                        False,
                                        dtype=jnp.bool_,
                                    )
                                )
                            )
                        ),
                    )
                if self._prototype_feature_utility is not None:
                    attempted_utility = (
                        result.prototype_feature_utility_diagnostics
                    )
                    if attempted_utility is None:
                        raise RuntimeError(
                            "configured prototype feature utility requires "
                            "diagnostics"
                        )
                    rejected = cast(
                        PrototypeUpdateResult,
                        rejected.replace(
                            prototype_feature_utility_diagnostics=(
                                attempted_utility.replace(
                                    outer_transaction_committed=jnp.asarray(
                                        False,
                                        dtype=jnp.bool_,
                                    )
                                )
                            )
                        ),
                    )
                if self._prototype_feature_utility_curation is not None:
                    attempted_utility_curation = (
                        result.prototype_feature_utility_curation_diagnostics
                    )
                    if attempted_utility_curation is None:
                        raise RuntimeError(
                            "configured feature utility curation requires "
                            "diagnostics"
                        )
                    rejected = cast(
                        PrototypeUpdateResult,
                        rejected.replace(
                            prototype_feature_utility_curation_diagnostics=(
                                attempted_utility_curation.replace(
                                    outer_transaction_committed=jnp.asarray(
                                        False,
                                        dtype=jnp.bool_,
                                    )
                                )
                            )
                        ),
                    )
                if self._prototype_feature_memory is not None:
                    attempted_feature_memory = (
                        result.prototype_feature_memory_diagnostics
                    )
                    if attempted_feature_memory is None:
                        raise RuntimeError(
                            "configured feature memory requires diagnostics"
                        )
                    rejected = cast(
                        PrototypeUpdateResult,
                        rejected.replace(
                            prototype_feature_memory_diagnostics=(
                                attempted_feature_memory.replace(
                                    outer_transaction_committed=jnp.asarray(
                                        False,
                                        dtype=jnp.bool_,
                                    )
                                )
                            )
                        ),
                    )
                if self._prototype_routed_linear_world_model is not None:
                    attempted_atomic = (
                        result.prototype_atomic_feature_world_memory_diagnostics
                    )
                    if attempted_atomic is None:
                        raise RuntimeError(
                            "configured atomic composition requires diagnostics"
                        )
                    rejected = cast(
                        PrototypeUpdateResult,
                        rejected.replace(
                            prototype_atomic_feature_world_memory_diagnostics=(
                                attempted_atomic.replace(
                                    destination_adopted=jnp.asarray(
                                        False,
                                        dtype=jnp.bool_,
                                    ),
                                    ordinary_updates_retained=jnp.asarray(
                                        False,
                                        dtype=jnp.bool_,
                                    ),
                                )
                            )
                        ),
                    )
                return rejected

            return jax.lax.cond(
                post_valid,
                lambda __: accepted,
                rollback_rejected_recurrent_or_postcheck,
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
                experiential_memory_input=experiential_memory_input,
                experiential_memory_input_supplied=(
                    experiential_memory_input_supplied
                ),
                partner_policy_fusion_input=partner_policy_fusion_input,
                partner_policy_fusion_input_supplied=(
                    partner_policy_fusion_input_supplied
                ),
                partner_policy_fusion_feedback=partner_policy_fusion_feedback,
                partner_policy_fusion_feedback_supplied=(
                    partner_policy_fusion_feedback_supplied
                ),
            )

        return jax.lax.cond(
            diagnostics.valid,
            valid_branch,
            rejected_branch,
            operand=None,
        )

    def _update_horde_for_transition(
        self,
        horde_state: MultiHeadMLPState,
        last_observation: Array,
        bootstrap_observation: Array,
        transition: PrototypeTransition,
    ) -> Any:
        """Update the configured Horde under one unchanged feature bank."""

        if self._horde is None:
            raise RuntimeError("Horde update requested while Horde is disabled")
        horde_cumulants = transition.horde_cumulants
        if horde_cumulants is None:
            horde_cumulants = jnp.full(
                (self._horde.n_demons,),
                transition.reward,
                dtype=jnp.float32,
            )
        horde_discounts = transition.horde_discounts
        if horde_discounts is None:
            # Preserve each question's declared continuing horizon. A true
            # global terminal zeros every bootstrap, matching the historical
            # standalone-Horde ordering exactly.
            horde_spec = cast(HordeSpec, self._config.horde_spec)
            horde_discounts = jnp.where(
                transition.discount > 0.0,
                horde_spec.gammas,
                jnp.zeros_like(horde_spec.gammas),
            )
        return self._horde.update_with_discounts(
            horde_state,
            last_observation,
            horde_cumulants,
            bootstrap_observation,
            horde_discounts,
        )

    def _apply_valid_transition_impl(
        self,
        state: PrototypeAgentState,
        transition: PrototypeTransition,
        diagnostics: PrototypeTransitionDiagnostics,
        *,
        control_discount: Array | None,
        gradient_joy_evidence: PrototypeCandidateUpdateAuditEvidence | None,
        gradient_joy_evidence_supplied: Array,
        gradient_joy_decision_id_matches: Array,
        experiential_memory_input: PrototypeExperientialMemoryInput | None,
        experiential_memory_input_supplied: Array,
        partner_policy_fusion_input: PrototypePartnerPolicyFusionInput | None,
        partner_policy_fusion_input_supplied: Array,
        partner_policy_fusion_feedback: (
            PrototypePartnerPolicyFusionFeedback | None
        ),
        partner_policy_fusion_feedback_supplied: Array,
        rtu_preparation: PrototypeRTUTransitionPreparation | None = None,
        external_final_builder_state: (
            RecurrentTraceUnitStateBuilderState | None
        ) = None,
        preselection_feature_reset_mask: Array | None = None,
        extended_action_mask: Array | None = None,
    ) -> PrototypeUpdateResult:
        """Apply one normalized, ownership-validated real transition."""
        effective_extended_action_mask = (
            jnp.ones(
                (self._config.oak.stomp.n_total_actions,),
                dtype=jnp.bool_,
            )
            if extended_action_mask is None
            else extended_action_mask
        )
        bootstrap_raw_obs = transition.next_observation
        decision_raw_obs = transition.next_decision_observation
        execution_boundary = transition.terminated | transition.truncated
        rew = transition.reward

        # -- Step 8a: consume bootstrap state, then restart once at a boundary -
        new_gru_state = state.gru_state
        old_builder_state = self._builder_component_state(
            state.state_builder_state
        )
        new_builder_state = old_builder_state
        old_feature_state = (
            self._feature_lifecycle_component_state(
                state.state_builder_state
            )
            if self._prototype_feature_lifecycle is not None
            else None
        )
        new_feature_state = old_feature_state
        current_oak_state = self._oak_component_state(state.oak_state)
        old_feature_consumer_binding = (
            self._feature_consumer_binding(state.oak_state)
            if self._prototype_feature_lifecycle is not None
            else None
        )
        new_feature_consumer_binding = old_feature_consumer_binding
        old_feature_utility_state = (
            self._feature_utility_component_state(state.oak_state)
            if self._prototype_feature_utility is not None
            else None
        )
        new_feature_utility_state = old_feature_utility_state
        bootstrap_base_obs = bootstrap_raw_obs
        decision_base_obs = decision_raw_obs
        if rtu_preparation is not None:
            bootstrap_builder_state = (
                rtu_preparation.bootstrap_transition.state
            )
            bootstrap_base_obs = (
                rtu_preparation.bootstrap_transition.representation
            )
            new_builder_state = rtu_preparation.decision_builder_state
            decision_base_obs = rtu_preparation.decision_representation
        elif self._state_builder is not None:
            (
                bootstrap_builder_state,
                bootstrap_base_obs,
            ) = self._state_builder.update(
                old_builder_state,
                bootstrap_raw_obs,
                transition.action,
                rew,
                transition.discount,
            )

            def restart_builder(builder_state: Any) -> tuple[Any, Array]:
                reset_state = self._state_builder.reset_episode(builder_state)
                return self._state_builder.start(reset_state, decision_raw_obs)

            new_builder_state, decision_base_obs = jax.lax.cond(
                execution_boundary,
                restart_builder,
                lambda builder_state: (builder_state, bootstrap_base_obs),
                bootstrap_builder_state,
            )
        elif state.gru_state is not None:
            bootstrap_gru_state, bootstrap_base_obs = _gru_step(
                state.gru_state,
                bootstrap_raw_obs,
            )

            def restart_gru(gru_state: GRUPerceptionState) -> tuple[Any, Array]:
                reset_state = cast(
                    GRUPerceptionState,
                    gru_state.replace(hidden=jnp.zeros_like(gru_state.hidden)),
                )
                return _gru_step(reset_state, decision_raw_obs)

            new_gru_state, decision_base_obs = jax.lax.cond(
                execution_boundary,
                restart_gru,
                lambda gru_state: (gru_state, bootstrap_base_obs),
                bootstrap_gru_state,
            )
        bootstrap_obs = self._augment_base_representation(
            old_feature_state,
            bootstrap_base_obs,
        )
        decision_obs = self._augment_base_representation(
            old_feature_state,
            decision_base_obs,
        )

        # Snapshot the real last observation/action before OaK update. The
        # base action may be an extended option index, but the world model is
        # action-conditioned on the primitive command sent to the environment.
        last_obs = state.current_representation
        last_base_obs = last_obs[: self._base_representation_dim()]
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
        behavior_gradient_result = _unavailable_behavior_gradient(
            self._config.oak.observation_dim
        )
        representation_gradient_mix = _unavailable_representation_gradient_mix(
            self._config.oak.observation_dim
        )
        ensemble_diagnostics = _unavailable_ensemble_diagnostics()
        recurrent_diagnostics = _unavailable_recurrent_latent_diagnostics(

                self._config.recurrent_latent_world_model_ensemble.ensemble_size
                if self._config.recurrent_latent_world_model_ensemble is not None
                else 0

        )
        recurrent_transaction_applied = jnp.asarray(True, dtype=jnp.bool_)
        recurrent_next_start_cache: Any = None
        model_replay_transaction_applied = jnp.asarray(False)
        model_replay_recorded = jnp.asarray(False)
        model_replay_sampled = jnp.asarray(False)
        model_replay_updates_applied = jnp.asarray(0, dtype=jnp.int32)
        model_replay_padding_count = jnp.asarray(0, dtype=jnp.int32)
        atomic_world_prepare: PrototypeRoutedLinearWorldPrepareResult | None = None

        if self._prototype_routed_linear_world_model is not None:
            if old_feature_state is None:
                raise RuntimeError(
                    "atomic routed world requires the source feature authority"
                )
            atomic_world_prepare = (
                self._prototype_routed_linear_world_model.prepare_transition(
                    self._atomic_routed_world_component_state(
                        state.world_model_state
                    ),
                    old_feature_state.router_state,
                    last_base_obs,
                    last_action,
                )
            )
        elif self._world_model is not None and self._buffer is not None:
            stable_base_world_model = self._prototype_feature_lifecycle is not None
            world_last_observation = (
                last_base_obs if stable_base_world_model else last_obs
            )
            world_bootstrap_observation = (
                bootstrap_base_obs if stable_base_world_model else bootstrap_obs
            )
            world_decision_observation = (
                decision_base_obs if stable_base_world_model else decision_obs
            )
            source_world_model_state = self._action_world_model_component_state(
                state.world_model_state
            )
            wm_result = self._world_model.update(
                source_world_model_state,
                world_last_observation,
                last_action,
                rew,
                transition.discount,
                world_bootstrap_observation,
            )
            new_wm_state = self._action_world_model_state_slot(wm_result.state)
            bootstrap_buffer_state = self._buffer.add(
                state.buffer_state,
                world_bootstrap_observation,
            )
            new_buf_state = jax.lax.cond(
                execution_boundary,
                lambda buffer_state: self._buffer.add(
                    buffer_state,
                    world_decision_observation,
                ),
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
                self._builder_component_state(
                    state.state_builder_state
                ).update_count
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
        elif (
            self._recurrent_latent_world_model_ensemble is not None
            and self._recurrent_signal_estimator is not None
        ):
            recurrent_wrapper = cast(
                PrototypeRecurrentLatentWorldModelState,
                state.world_model_state,
            )
            recurrent_result = self._recurrent_latent_world_model_ensemble.update(
                recurrent_wrapper.model_state,
                recurrent_wrapper.decision_cache,
                RecurrentLatentTransitionRecord(
                    observation=last_obs,
                    action=last_action,
                    reward=rew,
                    discount=transition.discount,
                    terminated=transition.terminated,
                    truncated=transition.truncated,
                    bootstrap_observation=bootstrap_obs,
                    next_decision_observation=decision_obs,
                ),
            )
            # Gaussian NLL can be negative. A fixed configuration-derived
            # affine offset (with a one-unit margin) makes the estimator input
            # non-negative without changing learning-progress differences.
            shifted_nll = (
                recurrent_result.mean_negative_log_likelihood
                + jnp.asarray(
                    self._recurrent_signal_nll_offset,
                    dtype=jnp.float32,
                )
            )
            candidate_signal_state, raw_signals = (
                self._recurrent_signal_estimator.observe(
                    recurrent_wrapper.signal_state,
                    recurrent_result.prediction.member_mean_predictions,
                    recurrent_result.prediction.member_aleatoric_variances,
                    recurrent_result.targets,
                    shifted_nll,
                )
            )
            recurrent_signals_valid = raw_signals.availability.input_valid
            recurrent_transaction_applied = (
                recurrent_result.diagnostics.applied
                & recurrent_signals_valid
            )
            committed_model_state = cast(
                RecurrentLatentWorldModelEnsembleState,
                jax.lax.cond(
                    recurrent_transaction_applied,
                    lambda: recurrent_result.state,
                    lambda: recurrent_wrapper.model_state,
                ),
            )
            committed_signal_state = cast(
                LearningSignalEstimatorState,
                jax.lax.cond(
                    recurrent_transaction_applied,
                    lambda: candidate_signal_state,
                    lambda: recurrent_wrapper.signal_state,
                ),
            )
            new_wm_state = recurrent_wrapper.replace(
                model_state=committed_model_state,
                signal_state=committed_signal_state,
            )
            recurrent_next_start_cache = recurrent_result.next_start_cache
            wm_error = jnp.where(
                recurrent_transaction_applied,
                recurrent_result.mean_negative_log_likelihood,
                jnp.asarray(0.0, dtype=jnp.float32),
            )
            learning_signals = _gate_recurrent_learning_signals(
                raw_signals,
                recurrent_result.prediction.availability,
                recurrent_transaction_applied,
            )
            representation_gradient = jnp.where(
                recurrent_transaction_applied,
                recurrent_result.representation_gradient,
                jnp.zeros_like(recurrent_result.representation_gradient),
            )
            representation_gradient_valid = (
                recurrent_transaction_applied
                & recurrent_result.representation_gradient_available
            )
            prediction_availability = RecurrentLatentPredictionAvailability(
                prediction=(
                    recurrent_transaction_applied
                    & recurrent_result.prediction.availability.prediction
                ),
                epistemic=(
                    recurrent_transaction_applied
                    & recurrent_result.prediction.availability.epistemic
                ),
                aleatoric=(
                    recurrent_transaction_applied
                    & recurrent_result.prediction.availability.aleatoric
                ),
            )
            recurrent_diagnostics = PrototypeRecurrentLatentDiagnostics(
                model=recurrent_result.diagnostics,
                prediction_availability=prediction_availability,
                raw_epistemic_disagreement=jnp.where(
                    recurrent_transaction_applied,
                    recurrent_result.prediction.epistemic_disagreement,
                    jnp.asarray(0.0, dtype=jnp.float32),
                ),
                raw_aleatoric_uncertainty=jnp.where(
                    recurrent_transaction_applied,
                    recurrent_result.prediction.aleatoric_uncertainty,
                    jnp.asarray(0.0, dtype=jnp.float32),
                ),
                raw_uncertainty_calibrated=jnp.asarray(False, dtype=jnp.bool_),
                signals_valid=(
                    recurrent_result.diagnostics.applied
                    & recurrent_signals_valid
                ),
                transaction_applied=recurrent_transaction_applied,
                next_decision_cached=jnp.asarray(False, dtype=jnp.bool_),
            )

        learning_value_router_result: LearningValueRouterResult | None = None
        new_learning_value_router_state: LearningValueRouterState | None = None
        if self._learning_value_router is not None:
            if gradient_joy_evidence is None:
                raise RuntimeError(
                    "configured learning-value router requires its fixed sidecar"
                )
            learning_value, learning_value_availability = (
                self._learning_value_router_inputs(
                    gradient_joy_evidence,
                    learning_signals,
                )
            )
            (
                new_learning_value_router_state,
                learning_value_router_result,
            ) = self._learning_value_router.route(
                self._learning_value_router_component_state(
                    state.state_builder_state
                ),
                learning_value,
                learning_value_availability,
            )

        builder_representation_gradient = representation_gradient
        builder_representation_gradient_valid = representation_gradient_valid
        mixer_config = self._config.representation_gradient_mixer
        if (
            mixer_config is not None
            or self._prototype_feature_lifecycle is not None
        ):
            behavior_gradient_result = self._behavior_representation_gradient(
                state,
                rew,
                bootstrap_obs,
                control_discount,
            )
        if mixer_config is not None:
            representation_gradient_mix = mix_representation_gradients(
                mixer_config,
                behavior_gradient_result.gradient,
                representation_gradient,
                behavior_valid=behavior_gradient_result.valid,
                grounded_world_valid=representation_gradient_valid,
            )
            builder_representation_gradient = representation_gradient_mix.gradient
            # ``valid`` also describes a deliberately empty ``discard`` mix.
            # Only ``applied`` means a real active source produced a builder
            # candidate, so zero-source modes never consume update capacity.
            builder_representation_gradient_valid = (
                representation_gradient_mix.applied
            )

        pullback_generation = (
            old_feature_state.router_state.generation_count
            if old_feature_state is not None
            else jnp.asarray(0, dtype=jnp.int32)
        )
        pullback_generation_words = (
            old_feature_state.router_state.generation_words
            if old_feature_state is not None
            else jnp.zeros((2,), dtype=jnp.uint32)
        )
        pair_gradient_pullback = PrototypePairGradientPullback(
            gradient=jnp.zeros(
                (self._base_representation_dim(),),
                dtype=jnp.float32,
            ),
            valid=jnp.asarray(False, dtype=jnp.bool_),
            semantic_generation=pullback_generation,
            semantic_generation_words=pullback_generation_words,
        )
        if self._prototype_feature_lifecycle is not None:
            if old_feature_state is None:
                raise RuntimeError(
                    "configured prototype feature lifecycle requires state"
                )
            candidate_pullback = (
                self._prototype_feature_lifecycle.pullback_pair_gradient(
                    old_feature_state,
                    last_base_obs,
                    builder_representation_gradient,
                    old_feature_state.router_state.generation_count,
                    old_feature_state.router_state.descriptors,
                    expected_generation_words=(
                        old_feature_state.router_state.generation_words
                    ),
                )
            )
            pullback_valid = (
                builder_representation_gradient_valid
                & candidate_pullback.valid
                & (
                    candidate_pullback.semantic_generation
                    == old_feature_state.router_state.generation_count
                )
                & jnp.all(
                    candidate_pullback.semantic_generation_words
                    == old_feature_state.router_state.generation_words
                )
            )
            builder_representation_gradient = jnp.where(
                pullback_valid,
                candidate_pullback.gradient,
                jnp.zeros_like(candidate_pullback.gradient),
            )
            builder_representation_gradient_valid = pullback_valid
            pair_gradient_pullback = cast(
                PrototypePairGradientPullback,
                candidate_pullback.replace(valid=pullback_valid),
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
            old_builder_state,
            new_builder_state,
            builder_representation_gradient,
            builder_representation_gradient_valid,
            learning_signals,
            gradient_joy_evidence,
            learning_value_router_result,
        )

        if external_final_builder_state is not None:
            if type(self._state_builder) is not RecurrentTraceUnitStateBuilder:
                raise RuntimeError(
                    "external RTU finalization requires an exact RTU builder"
                )
            new_builder_state = external_final_builder_state
            decision_base_obs = self._state_builder.encode(
                external_final_builder_state,
                decision_raw_obs,
            )
            decision_obs = self._augment_base_representation(
                old_feature_state,
                decision_base_obs,
            )

        # -- Steps 5/6/10/11: OaK update (real transition) -------------------
        oak_trace = self._oak.update_with_stomp_trace(
            current_oak_state,
            rew,
            bootstrap_obs,
            control_discount,
            decision_observation=decision_obs,
            execution_boundary=execution_boundary,
            extended_action_mask=effective_extended_action_mask,
            preselection_feature_reset_mask=(
                preselection_feature_reset_mask
            ),
        )
        oak_result = oak_trace.update
        new_oak_state = oak_result.state
        raw_stomp_owner = oak_trace.stomp_result.state
        option_search_learner_updates = jnp.asarray(0, dtype=jnp.int32)

        # -- Opt-in bounded option-model search control ----------------------
        # The legacy STOMP planning budget is statically required to be zero
        # when this composition is enabled. Search is anchored to the exact
        # next decision representation (not an autoreset transition's final
        # observation) and commits only the base learner subtree. OaK has
        # already selected this event's cached dispatch, which is deliberately
        # preserved; value changes become behavior-eligible only at the next
        # extended-action selection boundary (possibly after an active option
        # runs for several more primitive decisions).
        option_search_diagnostics: OptionSearchControlDiagnostics | None = None
        if self._option_search_control is not None:
            option_search_result = self._option_search_control.apply(
                new_oak_state.stomp_state,
                decision_obs,
                extended_action_mask=effective_extended_action_mask,
            )
            new_oak_state = cast(
                OaKState,
                new_oak_state.replace(
                    stomp_state=option_search_result.state,
                ),
            )
            option_search_diagnostics = option_search_result.diagnostics
            option_search_learner_updates = (
                option_search_result.diagnostics.applied_count
            )
        option_search_destination_owner = new_oak_state.stomp_state

        next_armed = (
            diagnostics.next_generation_available
            & diagnostics.next_counter_capacity_available
        )
        # Shared consumers must both learn under the old descriptors before
        # either is routed. The standalone-Horde path remains in its historical
        # position below, so configurations outside this opt-in composition
        # retain their exact update order.
        new_horde_state = self._horde_component_state(state)
        old_horde_state = new_horde_state
        horde_tderrs: Any = None
        shared_horde_td_targets: Any = None
        shared_horde_predictions: Any = None
        if self._horde is not None and self._shared_feature_horde_enabled():
            shared_horde_result = self._update_horde_for_transition(
                cast(MultiHeadMLPState, new_horde_state),
                last_obs,
                bootstrap_obs,
                transition,
            )
            new_horde_state = shared_horde_result.state
            horde_tderrs = shared_horde_result.td_errors
            shared_horde_td_targets = shared_horde_result.td_targets
            shared_horde_predictions = shared_horde_result.predictions

        feature_lifecycle_diagnostics: (
            PrototypeFeatureLifecycleIntegrationDiagnostics | None
        ) = None
        feature_utility_diagnostics: (
            PrototypeFeatureUtilityIntegrationDiagnostics | None
        ) = None
        feature_utility_curation_diagnostics: (
            PrototypeFeatureUtilityCurationIntegrationDiagnostics | None
        ) = None
        atomic_feature_world_memory_diagnostics: (
            PrototypeAtomicFeatureWorldMemoryDiagnostics | None
        ) = None
        atomic_memory_rebind_result: PrototypeFeatureMemoryRebindResult | None = None
        atomic_all_consumers_ready = jnp.asarray(False, dtype=jnp.bool_)
        atomic_lifecycle_capacity_capped = jnp.asarray(False, dtype=jnp.bool_)
        if self._prototype_feature_lifecycle is not None:
            if old_feature_state is None:
                raise RuntimeError(
                    "configured prototype feature lifecycle requires state"
                )
            target_available = behavior_gradient_result.valid
            automatic_target = jnp.where(
                target_available,
                behavior_gradient_result.diagnostics.target,
                jnp.asarray(0.0, dtype=jnp.float32),
            )
            control_feature_targets = jnp.where(
                target_available,
                jnp.reshape(automatic_target, (1,)),
                jnp.full((1,), jnp.nan, dtype=jnp.float32),
            )
            lifecycle_event = PrototypeFeatureLifecycleEvent(
                observation=last_base_obs,
                targets=(
                    jnp.concatenate(
                        (
                            control_feature_targets,
                            shared_horde_td_targets,
                        )
                    )
                    if self._shared_feature_horde_enabled()
                    else control_feature_targets
                ),
                next_observation=decision_base_obs,
                allow_curation=jnp.asarray(
                    next_armed,
                    dtype=jnp.bool_,
                ),
            )
            utility_observation_diagnostics: (
                PrototypeFeatureUtilityDiagnostics | None
            ) = None
            utility_curation_policy_diagnostics: (
                PrototypeFeatureUtilityCurationDiagnostics | None
            ) = None
            curation_priority_override: (
                InteractionCurationPriorityOverride | None
            ) = None
            curation_allowed = jnp.asarray(False, dtype=jnp.bool_)
            utility_tail_weights: Array | None = None
            if self._prototype_feature_utility is not None:
                if (
                    old_feature_utility_state is None
                    or shared_horde_predictions is None
                    or shared_horde_td_targets is None
                    or type(old_horde_state) is not MultiHeadMLPState
                ):
                    raise RuntimeError(
                        "feature utility requires old shared-consumer data"
                    )
                ordered_predictions = jnp.concatenate(
                    (
                        jnp.reshape(
                            behavior_gradient_result.diagnostics.prediction,
                            (1,),
                        ),
                        shared_horde_predictions,
                    )
                )
                utility_task_available = (
                    jnp.isfinite(lifecycle_event.targets)
                    & jnp.isfinite(ordered_predictions)
                )
                utility_tail_weights = jnp.concatenate(
                    (
                        self._behavior_active_pair_weights(state)[None, :],
                        self._horde_active_pair_weights(old_horde_state),
                    ),
                    axis=0,
                )
                utility_observation_result = (
                    self._prototype_feature_utility.observe(
                        old_feature_utility_state,
                        PrototypeFeatureUtilityEvent(
                            base_observation=last_base_obs,
                            augmented_observation=last_obs,
                            targets=jnp.where(
                                utility_task_available,
                                lifecycle_event.targets,
                                jnp.zeros_like(lifecycle_event.targets),
                            ),
                            predictions=jnp.where(
                                jnp.isfinite(ordered_predictions),
                                ordered_predictions,
                                jnp.zeros_like(ordered_predictions),
                            ),
                            target_available=utility_task_available,
                            active_consumer_tail_weights=(
                                utility_tail_weights
                            ),
                            semantic_generation=(
                                old_feature_state.router_state.generation_count
                            ),
                            semantic_generation_words=(
                                old_feature_state.router_state.generation_words
                            ),
                            active_descriptors=(
                                old_feature_state.router_state.descriptors
                            ),
                            candidate_descriptors=(
                                self._feature_candidate_descriptors(
                                    old_feature_state
                                )
                            ),
                        ),
                    )
                )
                new_feature_utility_state = utility_observation_result.state
                utility_observation_diagnostics = (
                    utility_observation_result.diagnostics
                )
            if self._prototype_feature_utility_curation is not None:
                if (
                    new_feature_utility_state is None
                    or utility_observation_diagnostics is None
                ):
                    raise RuntimeError(
                        "feature utility curation requires an audit observation"
                    )
                utility_curation_result = (
                    self._prototype_feature_utility_curation.rank(
                        new_feature_utility_state,
                        source_semantic_generation=(
                            old_feature_state.router_state.generation_count
                        ),
                        source_semantic_generation_words=(
                            old_feature_state.router_state.generation_words
                        ),
                        source_active_descriptors=(
                            old_feature_state.router_state.descriptors
                        ),
                        source_candidate_descriptors=(
                            self._feature_candidate_descriptors(
                                old_feature_state
                            )
                        ),
                    )
                )
                utility_curation_policy_diagnostics = (
                    utility_curation_result.diagnostics
                )
                observation_applied = (
                    utility_observation_diagnostics.transaction_applied
                )
                policy_transaction_acceptable = (
                    utility_curation_result.diagnostics.transaction_valid
                    & (
                        observation_applied
                        | utility_curation_result.diagnostics.observation_capacity_capped
                    )
                )
                fail_closed_override = ~policy_transaction_acceptable
                policy_override = utility_curation_result.override
                curation_priority_override = cast(
                    InteractionCurationPriorityOverride,
                    policy_override.replace(
                        enabled=(
                            fail_closed_override
                            | (policy_override.enabled & observation_applied)
                        ),
                        active_ranks=jnp.where(
                            fail_closed_override,
                            jnp.full_like(
                                policy_override.active_ranks,
                                CURATION_ACTIVE_INELIGIBLE_RANK,
                            ),
                            policy_override.active_ranks,
                        ),
                        candidate_ranks=jnp.where(
                            fail_closed_override,
                            jnp.full_like(
                                policy_override.candidate_ranks,
                                CURATION_CANDIDATE_INELIGIBLE_RANK,
                            ),
                            policy_override.candidate_ranks,
                        ),
                    ),
                )
                curation_allowed = (
                    lifecycle_event.allow_curation
                    & observation_applied
                    & utility_curation_result.diagnostics.curation_ready
                )
                lifecycle_event = cast(
                    PrototypeFeatureLifecycleEvent,
                    lifecycle_event.replace(
                        allow_curation=curation_allowed,
                    ),
                )
            feature_result: (
                PrototypeFeatureLifecycleResult
                | PrototypeFeatureLifecycleHordeResult
            )
            if self._shared_feature_horde_enabled():
                if shared_horde_td_targets is None:
                    raise RuntimeError(
                        "managed feature Horde requires causal TD targets"
                    )
                if self._prototype_routed_linear_world_model is not None:
                    if (
                        atomic_world_prepare is None
                        or self._prototype_feature_memory is None
                        or old_feature_consumer_binding is None
                    ):
                        raise RuntimeError(
                            "atomic feature/world/memory preparation is incomplete"
                        )
                    prepared_feature = (
                        self._prototype_feature_lifecycle.prepare_observe_and_route_with_horde(
                            old_feature_state,
                            new_oak_state,
                            cast(MultiHeadMLPState, new_horde_state),
                            old_feature_consumer_binding,
                            lifecycle_event,
                        )
                    )
                    destination_feature = prepared_feature.destination_result
                    capacity_diagnostics = destination_feature.diagnostics
                    maximum_observations = (
                        self._config.prototype_feature_lifecycle.max_observations
                    )
                    maximum_observation_words = jnp.asarray(
                        (
                            (maximum_observations >> 32) & _UINT32_MAX,
                            maximum_observations & _UINT32_MAX,
                        ),
                        dtype=jnp.uint32,
                    )
                    atomic_lifecycle_capacity_capped = (
                        destination_feature.horde_diagnostics.lifecycle_capacity_capped
                        & jnp.all(
                            old_feature_state.observe_words
                            == maximum_observation_words
                        )
                        & ~capacity_diagnostics.update_capacity_available
                        & capacity_diagnostics.post_update_consumer_clock_valid
                        & capacity_diagnostics.learner_update_rejected
                        & ~capacity_diagnostics.transaction_applied
                        & ~capacity_diagnostics.curation_proposed
                        & ~capacity_diagnostics.curation_deferred
                        & ~capacity_diagnostics.routing_attempted
                        & ~capacity_diagnostics.curation_committed
                        & ~capacity_diagnostics.curation_rolled_back
                        & _tree_arrays_equal(
                            destination_feature.state,
                            old_feature_state,
                        )
                        & _tree_arrays_equal(
                            destination_feature.oak_state,
                            new_oak_state,
                        )
                        & _tree_arrays_equal(
                            destination_feature.horde_state,
                            new_horde_state,
                        )
                        & _tree_arrays_equal(
                            destination_feature.consumer_binding,
                            old_feature_consumer_binding,
                        )
                        & jnp.array_equal(
                            destination_feature.next_augmented_observation,
                            self._augment_base_representation(
                                old_feature_state,
                                decision_base_obs,
                            ),
                        )
                    )
                    source_routed_world = self._atomic_routed_world_component_state(
                        state.world_model_state
                    )
                    world_event = PrototypeRoutedLinearWorldTransition(
                        prepared=atomic_world_prepare.prepared,
                        reward=rew,
                        discount=transition.discount,
                        next_base_observation=bootstrap_base_obs,
                        destination_router_state=(
                            destination_feature.state.router_state
                        ),
                        destination_binding=(
                            destination_feature.consumer_binding
                        ),
                    )
                    prepared_world = (
                        self._prototype_routed_linear_world_model.prepare_observe_and_route(
                            source_routed_world,
                            old_feature_state.router_state,
                            world_event,
                        )
                    )
                    source_feature_memory = self._feature_memory_component_state(
                        state.ia_state
                    )
                    atomic_memory_rebind_result = (
                        self._prototype_feature_memory.rebind(
                            source_feature_memory,
                            old_feature_consumer_binding,
                            destination_feature.consumer_binding,
                        )
                    )
                    route_attempted = (
                        destination_feature.diagnostics.routing_attempted
                    )
                    lifecycle_destination_ready = (
                        prepared_feature.internally_valid
                        & jnp.where(
                            route_attempted,
                            destination_feature.diagnostics.curation_committed,
                            jnp.asarray(True, dtype=jnp.bool_),
                        )
                    )
                    memory_rebind = atomic_memory_rebind_result.diagnostics
                    memory_destination_ready = jnp.where(
                        memory_rebind.rebind_required,
                        memory_rebind.transaction_applied,
                        memory_rebind.transaction_noop,
                    )
                    atomic_all_consumers_ready = (
                        lifecycle_destination_ready
                        & prepared_world.destination_valid
                        & memory_destination_ready
                    )
                    feature_receipt = (
                        self._prototype_feature_lifecycle.horde_external_readiness_receipt(
                            prepared_feature,
                            atomic_all_consumers_ready,
                        )
                    )
                    feature_adoption = (
                        self._prototype_feature_lifecycle.adopt_prepared_route_with_horde(
                            old_feature_state,
                            new_oak_state,
                            cast(MultiHeadMLPState, new_horde_state),
                            old_feature_consumer_binding,
                            prepared_feature,
                            feature_receipt,
                        )
                    )
                    world_receipt = (
                        self._prototype_routed_linear_world_model.external_readiness_receipt(
                            prepared_world,
                            atomic_all_consumers_ready,
                        )
                    )
                    world_adoption = (
                        self._prototype_routed_linear_world_model.adopt_prepared_route(
                            source_routed_world,
                            old_feature_state.router_state,
                            prepared_world,
                            world_receipt,
                        )
                    )
                    shared_feature_result = feature_adoption.result
                    new_wm_state = self._atomic_routed_world_state_slot(
                        world_adoption.result.state
                    )
                    wm_error = world_adoption.result.prediction_error
                    atomic_feature_world_memory_diagnostics = (
                        PrototypeAtomicFeatureWorldMemoryDiagnostics(
                            available=jnp.asarray(True, dtype=jnp.bool_),
                            descriptor_change_requested=route_attempted,
                            lifecycle_destination_ready=(
                                lifecycle_destination_ready
                            ),
                            world_ordinary_ready=prepared_world.ordinary_valid,
                            world_destination_ready=(
                                prepared_world.destination_valid
                            ),
                            memory_destination_ready=(
                                memory_destination_ready
                            ),
                            all_consumers_ready=atomic_all_consumers_ready,
                            destination_adopted=(
                                feature_adoption.diagnostics.destination_adopted
                                & world_adoption.diagnostics.destination_adopted
                                & memory_destination_ready
                                & atomic_all_consumers_ready
                            ),
                            ordinary_updates_retained=(
                                (
                                    feature_adoption.diagnostics.ordinary_update_retained
                                    | atomic_lifecycle_capacity_capped
                                )
                                & world_adoption.diagnostics.ordinary_update_retained
                            ),
                            external_curation_rolled_back=(
                                feature_adoption.diagnostics.external_curation_rolled_back
                            ),
                            lifecycle_adoption=feature_adoption.diagnostics,
                            world_adoption=world_adoption.diagnostics,
                            oak_update_evaluations=jnp.int32(1),
                            horde_update_evaluations=jnp.int32(1),
                            feature_learner_update_evaluations=(
                                feature_adoption.diagnostics.total_learner_update_evaluations
                            ),
                            lifecycle_router_evaluations=jnp.int32(2),
                            world_learner_update_evaluations=(
                                world_adoption.diagnostics.total_learner_update_evaluations
                            ),
                            world_router_evaluations=(
                                world_adoption.diagnostics.total_router_evaluations
                            ),
                            memory_rebind_evaluations=jnp.int32(1),
                            memory_step_evaluations=jnp.int32(1),
                        )
                    )
                else:
                    shared_feature_result = (
                        self._prototype_feature_lifecycle.observe_and_route_with_horde(
                            old_feature_state,
                            new_oak_state,
                            cast(MultiHeadMLPState, new_horde_state),
                            old_feature_consumer_binding,
                            lifecycle_event,
                            curation_priority_override=(
                                curation_priority_override
                            ),
                        )
                    )
                feature_result = shared_feature_result
                new_horde_state = shared_feature_result.horde_state
            else:
                feature_result = self._prototype_feature_lifecycle.observe_and_route(
                    old_feature_state,
                    new_oak_state,
                    old_feature_consumer_binding,
                    lifecycle_event,
                    curation_priority_override=curation_priority_override,
                )
            new_feature_state = feature_result.state
            new_oak_state = feature_result.oak_state
            new_feature_consumer_binding = feature_result.consumer_binding
            decision_obs = feature_result.next_augmented_observation
            if self._prototype_feature_utility_curation is not None:
                if (
                    utility_curation_policy_diagnostics is None
                    or utility_observation_diagnostics is None
                ):
                    raise RuntimeError(
                        "feature utility curation diagnostics are unavailable"
                    )
                lifecycle_diagnostics = feature_result.diagnostics
                selected_active_slot = (
                    lifecycle_diagnostics.curation_selected_active_worst_slot
                )
                selected_candidate_slot = (
                    lifecycle_diagnostics.curation_selected_promotion_candidate
                )
                source_active_descriptors = (
                    old_feature_state.router_state.descriptors
                )
                source_candidate_descriptors = (
                    self._feature_candidate_descriptors(old_feature_state)
                )
                active_slot_valid = (
                    (selected_active_slot >= 0)
                    & (
                        selected_active_slot
                        < self._config.prototype_feature_utility.active_pair_slots
                    )
                )
                candidate_slot_valid = (
                    (selected_candidate_slot >= 0)
                    & (
                        selected_candidate_slot
                        < self._config.prototype_feature_utility.candidate_pair_slots
                    )
                )
                safe_active_slot = jnp.clip(
                    selected_active_slot,
                    0,
                    self._config.prototype_feature_utility.active_pair_slots - 1,
                )
                safe_candidate_slot = jnp.clip(
                    selected_candidate_slot,
                    0,
                    self._config.prototype_feature_utility.candidate_pair_slots - 1,
                )
                rejected_descriptor = jnp.full((2,), -1, dtype=jnp.int32)
                selected_active_descriptor = jnp.where(
                    active_slot_valid,
                    source_active_descriptors[safe_active_slot],
                    rejected_descriptor,
                )
                selected_candidate_descriptor = jnp.where(
                    candidate_slot_valid,
                    source_candidate_descriptors[safe_candidate_slot],
                    rejected_descriptor,
                )
                feature_utility_curation_diagnostics = (
                    PrototypeFeatureUtilityCurationIntegrationDiagnostics(
                        policy=utility_curation_policy_diagnostics,
                        observation_applied=(
                            utility_observation_diagnostics.transaction_applied
                        ),
                        priority_override_supplied=jnp.asarray(
                            True,
                            dtype=jnp.bool_,
                        ),
                        priority_override_consulted=(
                            lifecycle_diagnostics.curation_priority_override_applied
                        ),
                        curation_allowed=curation_allowed,
                        selected_active_slot=selected_active_slot,
                        selected_candidate_slot=selected_candidate_slot,
                        selected_active_descriptor=(
                            selected_active_descriptor
                        ),
                        selected_candidate_descriptor=(
                            selected_candidate_descriptor
                        ),
                        lifecycle_curation_proposed=(
                            lifecycle_diagnostics.curation_proposed
                        ),
                        lifecycle_curation_deferred=(
                            lifecycle_diagnostics.curation_deferred
                        ),
                        lifecycle_curation_committed=(
                            lifecycle_diagnostics.curation_committed
                        ),
                        lifecycle_curation_rolled_back=(
                            lifecycle_diagnostics.curation_rolled_back
                        ),
                        outer_transaction_committed=jnp.asarray(
                            False,
                            dtype=jnp.bool_,
                        ),
                    )
                )
            if self._prototype_feature_utility is not None:
                if (
                    new_feature_utility_state is None
                    or utility_observation_diagnostics is None
                ):
                    raise RuntimeError(
                        "feature utility observation result is unavailable"
                    )
                observed_utility_state = new_feature_utility_state
                rebind_required = (
                    ~jnp.all(
                        new_feature_state.router_state.generation_words
                        == observed_utility_state.semantic_generation_words
                    )
                )
                utility_rebind_result = self._prototype_feature_utility.rebind(
                    observed_utility_state,
                    active_descriptors=(
                        new_feature_state.router_state.descriptors
                    ),
                    candidate_descriptors=self._feature_candidate_descriptors(
                        new_feature_state
                    ),
                    semantic_generation=(
                        new_feature_state.router_state.generation_count
                    ),
                    semantic_generation_words=(
                        new_feature_state.router_state.generation_words
                    ),
                )
                neutral_rebind_diagnostics = (
                    self._prototype_feature_utility.unavailable_diagnostics(
                        observed_utility_state
                    )
                )
                new_feature_utility_state = cast(
                    PrototypeFeatureUtilityState,
                    jax.tree.map(
                        lambda rebound, observed: jnp.where(
                            rebind_required,
                            rebound,
                            observed,
                        ),
                        utility_rebind_result.state,
                        observed_utility_state,
                    ),
                )
                selected_rebind_diagnostics = cast(
                    PrototypeFeatureUtilityDiagnostics,
                    jax.tree.map(
                        lambda rebound, neutral: jnp.where(
                            rebind_required,
                            rebound,
                            neutral,
                        ),
                        utility_rebind_result.diagnostics,
                        neutral_rebind_diagnostics,
                    ),
                )
                feature_utility_diagnostics = (
                    PrototypeFeatureUtilityIntegrationDiagnostics(
                        observation=utility_observation_diagnostics,
                        rebind=selected_rebind_diagnostics,
                        rebind_required=jnp.asarray(
                            rebind_required,
                            dtype=jnp.bool_,
                        ),
                        outer_transaction_committed=jnp.asarray(
                            False,
                            dtype=jnp.bool_,
                        ),
                    )
                )
            prediction = feature_result.predictions[0]
            error = feature_result.errors[0]
            feature_lifecycle_diagnostics = (
                PrototypeFeatureLifecycleIntegrationDiagnostics(
                    available=jnp.asarray(True, dtype=jnp.bool_),
                    target=automatic_target,
                    target_available=target_available,
                    pullback_gradient=pair_gradient_pullback.gradient,
                    pullback_valid=pair_gradient_pullback.valid,
                    pullback_semantic_generation=(
                        pair_gradient_pullback.semantic_generation
                    ),
                    prediction=jnp.where(
                        jnp.isfinite(prediction),
                        prediction,
                        jnp.asarray(0.0, dtype=jnp.float32),
                    ),
                    error=jnp.where(
                        jnp.isfinite(error),
                        error,
                        jnp.asarray(0.0, dtype=jnp.float32),
                    ),
                    task_targets=jnp.where(
                        jnp.isfinite(lifecycle_event.targets),
                        lifecycle_event.targets,
                        jnp.zeros_like(lifecycle_event.targets),
                    ),
                    task_target_available=jnp.isfinite(
                        lifecycle_event.targets
                    ),
                    task_predictions=jnp.where(
                        jnp.isfinite(feature_result.predictions),
                        feature_result.predictions,
                        jnp.zeros_like(feature_result.predictions),
                    ),
                    task_errors=jnp.where(
                        jnp.isfinite(feature_result.errors),
                        feature_result.errors,
                        jnp.zeros_like(feature_result.errors),
                    ),
                    metrics=feature_result.metrics,
                    lifecycle=feature_result.diagnostics,
                    outer_transaction_committed=jnp.asarray(
                        False,
                        dtype=jnp.bool_,
                    ),
                )
            )

        feature_route_destination_owner = new_oak_state.stomp_state

        # -- Step 9: guarded Dyna dreaming ------------------------------------
        dream_td_errors: Any = None
        dream_learner_updates = jnp.asarray(0, dtype=jnp.int32)

        if (
            self._world_model is not None
            and self._buffer is not None
            and self._dreamer is not None
            and self._config.n_dreams_per_step > 0
        ):
            rng_key = new_oak_state.stomp_state.rng_key
            (
                new_oak_state,
                dream_td_errors,
                dream_learner_updates,
            ) = self._run_dreams_with_count(
                new_oak_state,
                new_wm_state,
                new_buf_state,
                rng_key,
                effective_extended_action_mask,
            )
        dyna_destination_owner = new_oak_state.stomp_state

        # -- Step 3: Horde GVF update -----------------------------------------
        if self._horde is not None and not self._shared_feature_horde_enabled():
            horde_result = self._update_horde_for_transition(
                cast(MultiHeadMLPState, new_horde_state),
                last_obs,
                bootstrap_obs,
                transition,
            )
            new_horde_state = horde_result.state
            horde_tderrs = horde_result.td_errors

        # -- Step 12: IA update -----------------------------------------------
        current_ia_state = self._ia_component_state(state.ia_state)
        new_ia_component_state = current_ia_state
        ia_augmented: Any = None
        ia_recommendation: Any = None
        ia_update_applied = jnp.asarray(False, dtype=jnp.bool_)

        if self._ia is not None and current_ia_state is not None:
            ia_result: IAUpdateResult = self._ia.update(
                current_ia_state,
                last_obs,
                rew,
                bootstrap_obs,
                partner_action=last_action,
                discount=control_discount,
                decision_observation=decision_obs,
                execution_boundary=execution_boundary,
            )
            new_ia_component_state = ia_result.state
            ia_augmented = ia_result.augmented_obs
            ia_recommendation = ia_result.recommendation
            ia_update_applied = ia_result.update_applied

        next_step_count = _saturating_int32_increment(state.step_count)
        next_step_words, _ = _checked_lifetime_words_add(state.step_words, 1)
        next_observation_event_count = jnp.where(
            execution_boundary,
            _saturating_int32_increment(
                _saturating_int32_increment(state.observation_event_count)
            ),
            _saturating_int32_increment(state.observation_event_count),
        )
        next_observation_event_words_one, _ = _checked_lifetime_words_add(
            state.observation_event_words,
            1,
        )
        next_observation_event_words_two, _ = _checked_lifetime_words_add(
            state.observation_event_words,
            2,
        )
        next_observation_event_words = jnp.where(
            execution_boundary,
            next_observation_event_words_two,
            next_observation_event_words_one,
        )
        counterfactual_next_action = jnp.where(
            next_armed,
            oak_result.primitive_action,
            jnp.asarray(-1, dtype=jnp.int32),
        )
        next_decision_id = jnp.where(
            next_armed,
            _increment_decision_id(state.current_decision_id),
            state.current_decision_id,
        )
        current_memory_state: ExperientialMemoryState | None = None
        new_memory_state: ExperientialMemoryState | None = None
        memory_diagnostics: PrototypeExperientialMemoryDiagnostics | None = None
        feature_memory_diagnostics: (
            PrototypeFeatureMemoryIntegrationDiagnostics | None
        ) = None
        new_memory_slot_state: Any = None
        partner_fusion_diagnostics: (
            PrototypePartnerPolicyFusionDiagnostics | None
        ) = None
        next_action = counterfactual_next_action
        memory_safety_mask = jnp.ones(
            (self._config.oak.n_primitive_actions,),
            dtype=jnp.bool_,
        )
        if self._experiential_memory is not None:
            if experiential_memory_input is None:
                raise RuntimeError(
                    "configured experiential memory requires a fixed sidecar"
                )
            memory_current_representation = last_obs
            memory_bootstrap_representation = bootstrap_obs
            memory_decision_representation = decision_obs
            source_representation_version: Array | None = None
            destination_representation_version: Array | None = None
            rebound_feature_memory_state: PrototypeFeatureMemoryState | None = None
            rebind_diagnostics: PrototypeFeatureMemoryRebindDiagnostics | None = None
            if self._prototype_feature_memory is not None:
                if (
                    old_feature_state is None
                    or new_feature_state is None
                    or old_feature_consumer_binding is None
                    or new_feature_consumer_binding is None
                ):
                    raise RuntimeError(
                        "feature-memory update requires source and destination banks"
                    )
                source_feature_memory_state = self._feature_memory_component_state(
                    state.ia_state
                )
                if self._prototype_routed_linear_world_model is not None:
                    if atomic_memory_rebind_result is None:
                        raise RuntimeError(
                            "atomic memory update requires its readiness candidate"
                        )
                    rebind_candidate = atomic_memory_rebind_result
                    rebound_feature_memory_state = cast(
                        PrototypeFeatureMemoryState,
                        jax.lax.cond(
                            atomic_all_consumers_ready,
                            lambda _: rebind_candidate.state,
                            lambda _: source_feature_memory_state,
                            operand=None,
                        ),
                    )
                    rebind_diagnostics = rebind_candidate.diagnostics.replace(
                        destination_descriptors_changed=(
                            rebind_candidate.diagnostics.destination_descriptors_changed
                            & atomic_all_consumers_ready
                        ),
                        destination_generation_changed=(
                            rebind_candidate.diagnostics.destination_generation_changed
                            & atomic_all_consumers_ready
                        ),
                        transition_consistent=jnp.where(
                            atomic_all_consumers_ready,
                            rebind_candidate.diagnostics.transition_consistent,
                            rebind_candidate.diagnostics.source_state_valid
                            & rebind_candidate.diagnostics.source_binding_valid
                            & rebind_candidate.diagnostics.source_binding_matches,
                        ),
                        rebind_required=(
                            rebind_candidate.diagnostics.rebind_required
                            & atomic_all_consumers_ready
                        ),
                        transaction_applied=(
                            rebind_candidate.diagnostics.transaction_applied
                            & atomic_all_consumers_ready
                        ),
                        transaction_noop=jnp.where(
                            atomic_all_consumers_ready,
                            rebind_candidate.diagnostics.transaction_noop,
                            rebind_candidate.diagnostics.source_state_valid
                            & rebind_candidate.diagnostics.source_binding_valid
                            & rebind_candidate.diagnostics.source_binding_matches,
                        ),
                        valid_rows_reencoded=jnp.where(
                            atomic_all_consumers_ready,
                            rebind_candidate.diagnostics.valid_rows_reencoded,
                            jnp.asarray(0, dtype=jnp.int32),
                        ),
                        requested_generation_words=(
                            new_feature_consumer_binding.semantic_generation_words
                        ),
                        committed_generation_words=(
                            rebound_feature_memory_state.consumer_binding.semantic_generation_words
                        ),
                        memory_step_words_after=(
                            rebound_feature_memory_state.memory_state.step_words
                        ),
                    )
                else:
                    rebind_result = self._prototype_feature_memory.rebind(
                        source_feature_memory_state,
                        old_feature_consumer_binding,
                        new_feature_consumer_binding,
                    )
                    rebound_feature_memory_state = rebind_result.state
                    rebind_diagnostics = rebind_result.diagnostics
                current_memory_state = rebound_feature_memory_state.memory_state
                # OaK/Horde learned under the source bank above. Memory now
                # migrates first, then queries and writes entirely in the
                # destination bank so no row mixes feature meanings.
                memory_current_representation = self._augment_base_representation(
                    new_feature_state,
                    last_base_obs,
                )
                memory_bootstrap_representation = (
                    self._augment_base_representation(
                        new_feature_state,
                        bootstrap_base_obs,
                    )
                )
                memory_decision_representation = (
                    self._augment_base_representation(
                        new_feature_state,
                        decision_base_obs,
                    )
                )
                source_representation_version = (
                    old_feature_consumer_binding.semantic_generation
                )
                destination_representation_version = (
                    new_feature_consumer_binding.semantic_generation
                )
            else:
                current_memory_state = self._experiential_memory_component_state(
                    state.ia_state
                )
            (
                new_memory_state,
                new_oak_state,
                next_action,
                memory_diagnostics,
            ) = self._apply_experiential_memory(
                current_memory_state,
                new_oak_state,
                memory_current_representation,
                memory_bootstrap_representation,
                memory_decision_representation,
                transition.action,
                rew,
                current_prototype_decision_id=state.current_decision_id,
                next_prototype_decision_id=next_decision_id,
                next_armed=next_armed,
                transaction_allowed=(
                    (
                        (
                            atomic_feature_world_memory_diagnostics.lifecycle_adoption.transaction_applied
                            | atomic_lifecycle_capacity_capped
                        )
                        & atomic_feature_world_memory_diagnostics.world_adoption.transaction_applied
                    )
                    if atomic_feature_world_memory_diagnostics is not None
                    else jnp.asarray(True, dtype=jnp.bool_)
                ),
                memory_input=experiential_memory_input,
                input_supplied=experiential_memory_input_supplied,
                source_representation_version=source_representation_version,
                destination_representation_version=(
                    destination_representation_version
                ),
            )
            new_memory_slot_state = new_memory_state
            if self._prototype_feature_memory is not None:
                if (
                    rebound_feature_memory_state is None
                    or rebind_diagnostics is None
                    or new_feature_consumer_binding is None
                    or source_representation_version is None
                ):
                    raise RuntimeError(
                        "feature-memory diagnostics require a completed rebind"
                    )
                rebound_with_event = cast(
                    PrototypeFeatureMemoryState,
                    rebound_feature_memory_state.replace(
                        memory_state=new_memory_state,
                    ),
                )
                post_memory_state_valid = self._prototype_feature_memory.state_valid(
                    rebound_with_event,
                    new_feature_consumer_binding,
                )
                new_memory_slot_state = rebound_with_event
                query_source_version_matches = (
                    experiential_memory_input.query_representation_version
                    == source_representation_version
                )
                entry_source_version_matches = (
                    experiential_memory_input.entry_representation_version
                    == source_representation_version
                )
                feature_memory_diagnostics = (
                    PrototypeFeatureMemoryIntegrationDiagnostics(
                        rebind=rebind_diagnostics,
                        query_source_version_matches=(
                            query_source_version_matches
                        ),
                        entry_source_version_matches=(
                            entry_source_version_matches
                        ),
                        source_versions_match=(
                            query_source_version_matches
                            & entry_source_version_matches
                        ),
                        current_destination_encoding_valid=(
                            jnp.all(jnp.isfinite(memory_current_representation))
                        ),
                        bootstrap_destination_encoding_valid=(
                            jnp.all(jnp.isfinite(memory_bootstrap_representation))
                        ),
                        decision_destination_encoding_valid=(
                            jnp.all(jnp.isfinite(memory_decision_representation))
                            & jnp.array_equal(
                                memory_decision_representation,
                                decision_obs,
                            )
                        ),
                        post_memory_state_valid=post_memory_state_valid,
                        outer_transaction_committed=jnp.asarray(
                            False,
                            dtype=jnp.bool_,
                        ),
                    )
                )
            memory_safety_mask = jnp.where(
                memory_diagnostics.transaction_required,
                experiential_memory_input.next_action_safety_mask,
                memory_safety_mask,
            )
        memory_dispatch_destination_owner = new_oak_state.stomp_state
        new_interaction_state = self._interaction_slot(
            new_ia_component_state,
            (
                self._partner_fusion_component_state(state.ia_state)
                if self._partner_policy_fusion is not None
                else None
            ),
            experiential_memory_state=new_memory_slot_state,
            feedback_prototype_decision_id=(
                self._partner_interaction_state(
                    state.ia_state
                ).feedback_prototype_decision_id
                if self._partner_policy_fusion is not None
                else None
            ),
            feedback_prototype_decision_id_available=(
                self._partner_interaction_state(
                    state.ia_state
                ).feedback_prototype_decision_id_available
                if self._partner_policy_fusion is not None
                else None
            ),
        )
        if self._partner_policy_fusion is not None:
            if (
                partner_policy_fusion_input is None
                or partner_policy_fusion_feedback is None
            ):
                raise RuntimeError("configured partner fusion requires fixed sidecars")
            (
                new_interaction_state,
                new_oak_state,
                next_action,
                partner_fusion_diagnostics,
            ) = self._apply_partner_policy_fusion(
                state.ia_state,
                new_ia_component_state,
                new_oak_state,
                decision_obs,
                # The scalar IDs are authenticated saturating telemetry. The
                # exact outer clocks below remain the ordering and ownership
                # authorities after signed telemetry saturates.
                derived_decision_id=next_step_count,
                derived_event_id=next_observation_event_count,
                derived_decision_words=next_step_words,
                derived_event_words=next_observation_event_words,
                derived_prototype_decision_id=next_decision_id,
                next_armed=next_armed,
                transaction_allowed=jnp.asarray(True, dtype=jnp.bool_),
                experiential_memory_state=new_memory_state,
                upstream_safety_action_mask=memory_safety_mask,
                decision_input=partner_policy_fusion_input,
                decision_input_supplied=partner_policy_fusion_input_supplied,
                feedback=partner_policy_fusion_feedback,
                feedback_input_supplied=(
                    partner_policy_fusion_feedback_supplied
                ),
            )
        partner_dispatch_destination_owner = new_oak_state.stomp_state
        if self._recurrent_latent_world_model_ensemble is not None:
            recurrent_wrapper = cast(
                PrototypeRecurrentLatentWorldModelState,
                new_wm_state,
            )
            next_recurrent_decision = self._recurrent_decide_from_start(
                recurrent_wrapper.model_state,
                cast(RecurrentLatentStartCache, recurrent_next_start_cache),
                next_action,
                next_armed & recurrent_transaction_applied,
            )
            new_wm_state = recurrent_wrapper.replace(
                decision_cache=next_recurrent_decision,
            )
            recurrent_diagnostics = recurrent_diagnostics.replace(
                next_decision_cached=(
                    recurrent_transaction_applied
                    & next_armed
                    & next_recurrent_decision.valid
                ),
            )
        false = jnp.asarray(False, dtype=jnp.bool_)
        true = jnp.asarray(True, dtype=jnp.bool_)
        zero_work = jnp.asarray(0, dtype=jnp.int32)
        option_search_configured = jnp.asarray(
            self._option_search_control is not None,
            dtype=jnp.bool_,
        )
        feature_route_configured = jnp.asarray(
            self._prototype_feature_lifecycle is not None,
            dtype=jnp.bool_,
        )
        dyna_configured = jnp.asarray(
            self._world_model is not None
            and self._buffer is not None
            and self._dreamer is not None
            and self._config.n_dreams_per_step > 0,
            dtype=jnp.bool_,
        )
        memory_dispatch_configured = jnp.asarray(
            self._experiential_memory is not None,
            dtype=jnp.bool_,
        )
        partner_dispatch_configured = jnp.asarray(
            self._partner_policy_fusion is not None,
            dtype=jnp.bool_,
        )
        raw_owner_digest = stomp_typed_tree_digest(raw_stomp_owner)
        if self._option_search_control is None:
            option_search_destination_digest = raw_owner_digest
            option_search_delta_valid = true
        else:
            option_search_destination_digest = stomp_typed_tree_digest(
                option_search_destination_owner
            )
            option_search_delta_valid = stomp_owner_stage_delta_valid(
                raw_stomp_owner,
                option_search_destination_owner,
                stage_kind=STOMP_OWNER_STAGE_OPTION_SEARCH,
            )
        if self._prototype_feature_lifecycle is None:
            feature_route_destination_digest = option_search_destination_digest
            feature_route_delta_valid = true
        else:
            feature_route_destination_digest = stomp_typed_tree_digest(
                feature_route_destination_owner
            )
            feature_route_delta_valid = stomp_owner_stage_delta_valid(
                option_search_destination_owner,
                feature_route_destination_owner,
                stage_kind=STOMP_OWNER_STAGE_FEATURE_ROUTE,
            )
        dyna_enabled = (
            self._world_model is not None
            and self._buffer is not None
            and self._dreamer is not None
            and self._config.n_dreams_per_step > 0
        )
        if not dyna_enabled:
            dyna_destination_digest = feature_route_destination_digest
            dyna_delta_valid = true
        else:
            dyna_destination_digest = stomp_typed_tree_digest(
                dyna_destination_owner
            )
            dyna_delta_valid = stomp_owner_stage_delta_valid(
                feature_route_destination_owner,
                dyna_destination_owner,
                stage_kind=STOMP_OWNER_STAGE_DYNA,
            )
        if self._experiential_memory is None:
            memory_dispatch_destination_digest = dyna_destination_digest
            memory_dispatch_delta_valid = true
        else:
            memory_dispatch_destination_digest = stomp_typed_tree_digest(
                memory_dispatch_destination_owner
            )
            memory_dispatch_delta_valid = stomp_owner_stage_delta_valid(
                dyna_destination_owner,
                memory_dispatch_destination_owner,
                stage_kind=STOMP_OWNER_STAGE_MEMORY_DISPATCH,
            )
        if self._partner_policy_fusion is None:
            partner_dispatch_destination_digest = (
                memory_dispatch_destination_digest
            )
            partner_dispatch_delta_valid = true
        else:
            partner_dispatch_destination_digest = stomp_typed_tree_digest(
                partner_dispatch_destination_owner
            )
            partner_dispatch_delta_valid = stomp_owner_stage_delta_valid(
                memory_dispatch_destination_owner,
                partner_dispatch_destination_owner,
                stage_kind=STOMP_OWNER_STAGE_PARTNER_DISPATCH,
            )
        owner_finalization_stages = (
            make_stomp_owner_stage_receipt(
                raw_stomp_owner,
                option_search_destination_owner,
                stage_kind=STOMP_OWNER_STAGE_OPTION_SEARCH,
                configured=option_search_configured,
                evaluated=jnp.where(option_search_configured, true, false),
                stomp_update_evaluations=zero_work,
                learner_updates_applied=option_search_learner_updates,
                source_digest=raw_owner_digest,
                destination_digest=option_search_destination_digest,
                classified_delta_valid=option_search_delta_valid,
            ),
            make_stomp_owner_stage_receipt(
                option_search_destination_owner,
                feature_route_destination_owner,
                stage_kind=STOMP_OWNER_STAGE_FEATURE_ROUTE,
                configured=feature_route_configured,
                evaluated=jnp.where(feature_route_configured, true, false),
                stomp_update_evaluations=zero_work,
                learner_updates_applied=zero_work,
                source_digest=option_search_destination_digest,
                destination_digest=feature_route_destination_digest,
                classified_delta_valid=feature_route_delta_valid,
            ),
            make_stomp_owner_stage_receipt(
                feature_route_destination_owner,
                dyna_destination_owner,
                stage_kind=STOMP_OWNER_STAGE_DYNA,
                configured=dyna_configured,
                evaluated=jnp.where(dyna_configured, true, false),
                stomp_update_evaluations=jnp.where(
                    dyna_configured,
                    jnp.asarray(
                        self._config.n_dreams_per_step,
                        dtype=jnp.int32,
                    ),
                    zero_work,
                ),
                learner_updates_applied=dream_learner_updates,
                source_digest=feature_route_destination_digest,
                destination_digest=dyna_destination_digest,
                classified_delta_valid=dyna_delta_valid,
            ),
            make_stomp_owner_stage_receipt(
                dyna_destination_owner,
                memory_dispatch_destination_owner,
                stage_kind=STOMP_OWNER_STAGE_MEMORY_DISPATCH,
                configured=memory_dispatch_configured,
                evaluated=jnp.where(memory_dispatch_configured, true, false),
                stomp_update_evaluations=zero_work,
                learner_updates_applied=zero_work,
                source_digest=dyna_destination_digest,
                destination_digest=memory_dispatch_destination_digest,
                classified_delta_valid=memory_dispatch_delta_valid,
            ),
            make_stomp_owner_stage_receipt(
                memory_dispatch_destination_owner,
                partner_dispatch_destination_owner,
                stage_kind=STOMP_OWNER_STAGE_PARTNER_DISPATCH,
                configured=partner_dispatch_configured,
                evaluated=jnp.where(partner_dispatch_configured, true, false),
                stomp_update_evaluations=zero_work,
                learner_updates_applied=zero_work,
                source_digest=memory_dispatch_destination_digest,
                destination_digest=partner_dispatch_destination_digest,
                classified_delta_valid=partner_dispatch_delta_valid,
            ),
        )
        imagined_stomp_evaluations = jnp.where(
            dyna_configured,
            jnp.asarray(self._config.n_dreams_per_step, dtype=jnp.int32),
            zero_work,
        )
        owner_finalization_trace = make_stomp_owner_finalization_trace(
            raw_stomp_owner,
            owner_finalization_stages,
            partner_dispatch_destination_owner,
            real_control_stomp_evaluations=(
                oak_trace.stomp_update_evaluations
            ),
            imagined_stomp_evaluations=imagined_stomp_evaluations,
            option_search_learner_updates=option_search_learner_updates,
            raw_digest=raw_owner_digest,
            final_digest=partner_dispatch_destination_digest,
        )
        new_state = PrototypeAgentState(
            oak_state=self._oak_state_slot(
                new_oak_state,
                new_feature_consumer_binding,
                new_horde_state,
                new_feature_utility_state,
            ),
            world_model_state=new_wm_state,
            buffer_state=new_buf_state,
            horde_state=(
                None
                if self._shared_feature_horde_enabled()
                else new_horde_state
            ),
            ia_state=new_interaction_state,
            gru_state=new_gru_state,
            state_builder_state=self._representation_state_slot(
                new_builder_state,
                new_feature_state,
                new_learning_value_router_state,
            ),
            current_raw_observation=decision_raw_obs,
            current_representation=decision_obs,
            current_action=next_action,
            current_decision_id=next_decision_id,
            started=next_armed,
            observation_event_count=next_observation_event_count,
            observation_event_words=next_observation_event_words,
            step_count=next_step_count,
            step_words=next_step_words,
        )

        return PrototypeUpdateResult(
            state=new_state,
            action=next_action,
            oak_td_error=oak_result.td_error,
            oak_average_reward=oak_result.average_reward,
            oak_stomp_update_result=oak_trace.stomp_result,
            oak_stomp_update_available=(
                oak_result.update_applied
                & (oak_trace.stomp_update_evaluations == 1)
            ),
            oak_stomp_update_evaluations=oak_trace.stomp_update_evaluations,
            oak_owner_finalization_trace=owner_finalization_trace,
            oak_real_stomp_update_evaluations=(
                oak_trace.stomp_update_evaluations
            ),
            oak_imagined_stomp_update_evaluations=(
                imagined_stomp_evaluations
            ),
            oak_total_stomp_update_evaluations=(
                oak_trace.stomp_update_evaluations
                + imagined_stomp_evaluations
            ),
            oak_option_search_learner_updates=(
                option_search_learner_updates
            ),
            oak_bootstrap_observation=bootstrap_obs,
            oak_decision_observation=decision_obs,
            oak_execution_boundary=execution_boundary,
            world_model_error=wm_error,
            learning_signals=learning_signals,
            learning_value_router_result=learning_value_router_result,
            world_model_representation_gradient=representation_gradient,
            world_model_representation_gradient_valid=(
                representation_gradient_valid
            ),
            behavior_gradient_result=behavior_gradient_result,
            representation_gradient_mix=representation_gradient_mix,
            world_model_ensemble_diagnostics=ensemble_diagnostics,
            recurrent_latent_world_model_diagnostics=recurrent_diagnostics,
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
            option_search_control_diagnostics=option_search_diagnostics,
            prototype_feature_lifecycle_diagnostics=(
                feature_lifecycle_diagnostics
            ),
            prototype_feature_utility_diagnostics=(
                feature_utility_diagnostics
            ),
            prototype_feature_utility_curation_diagnostics=(
                feature_utility_curation_diagnostics
            ),
            dream_td_errors=dream_td_errors,
            horde_td_errors=horde_tderrs,
            ia_augmented_obs=ia_augmented,
            ia_recommendation=ia_recommendation,
            ia_update_applied=ia_update_applied,
            experiential_memory_diagnostics=memory_diagnostics,
            prototype_feature_memory_diagnostics=feature_memory_diagnostics,
            partner_policy_fusion_diagnostics=partner_fusion_diagnostics,
            transition_diagnostics=diagnostics,
            prototype_atomic_feature_world_memory_diagnostics=(
                atomic_feature_world_memory_diagnostics
            ),
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
                result.candidate_update_audit_passed,
                result.audited_candidate_update_applied,
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
            candidate_update_audit_passed,
            audited_candidate_update_applied,
        ) = outputs

        return PrototypeArrayResult(
            state=final_state,
            actions=actions,
            oak_td_errors=oak_td_errors,
            oak_average_rewards=oak_avg_rewards,
            transition_valid=transition_valid,
            state_builder_learning_applied=builder_learning_applied,
            gradient_sparks_joy=candidate_update_audit_passed,
            joyful_gradient_applied=audited_candidate_update_applied,
        )

    def scan_transitions(
        self,
        state: PrototypeAgentState,
        transitions: PrototypeTransition,
        candidate_update_audit_evidence: (
            PrototypeCandidateUpdateAuditEvidence | None
        ) = None,
        *,
        gradient_joy_evidence: PrototypeGradientJoyEvidence | None = None,
        experiential_memory_input: (
            PrototypeExperientialMemoryInput | None
        ) = None,
        partner_policy_fusion_input: (
            PrototypePartnerPolicyFusionInput | None
        ) = None,
        partner_policy_fusion_feedback: (
            PrototypePartnerPolicyFusionFeedback | None
        ) = None,
    ) -> PrototypeArrayResult:
        """Run authoritative transitions and fixed optional sidecars through scan."""

        if (
            candidate_update_audit_evidence is not None
            and gradient_joy_evidence is not None
        ):
            raise ValueError(
                "candidate_update_audit_evidence and gradient_joy_evidence "
                "cannot both be supplied"
            )
        gradient_joy_evidence = (
            candidate_update_audit_evidence
            if candidate_update_audit_evidence is not None
            else gradient_joy_evidence
        )

        use_partner_input = partner_policy_fusion_input is not None
        use_partner_feedback = partner_policy_fusion_feedback is not None
        use_memory_input = experiential_memory_input is not None
        if self._experiential_memory is None and use_memory_input:
            raise ValueError(
                "experiential memory scan sidecars require experiential memory"
            )
        if self._partner_policy_fusion is None and (
            use_partner_input or use_partner_feedback
        ):
            raise ValueError(
                "partner fusion scan sidecars require partner policy fusion"
            )

        def outputs_from_result(
            result: PrototypeUpdateResult,
        ) -> tuple[Array, Array, Array, Array, Array, Array, Array]:
            return (
                result.action,
                result.oak_td_error,
                result.oak_average_reward,
                result.transition_diagnostics.valid,
                result.state_builder_learning_diagnostics.applied,
                result.candidate_update_audit_passed,
                result.audited_candidate_update_applied,
            )

        def transition_only_step(
            carry: PrototypeAgentState,
            transition: PrototypeTransition,
        ) -> tuple[
            PrototypeAgentState,
            tuple[Array, Array, Array, Array, Array, Array, Array],
        ]:
            result = self.update_transition(carry, transition)
            return result.state, outputs_from_result(result)

        if (
            self._partner_policy_fusion is None
            and self._experiential_memory is None
            and gradient_joy_evidence is None
        ):
            final_state, outputs = jax.lax.scan(
                transition_only_step,
                state,
                transitions,
            )
        elif (
            self._partner_policy_fusion is None
            and self._experiential_memory is None
        ):
            def transition_with_candidate_audit_step(
                carry: PrototypeAgentState,
                inputs: tuple[
                    PrototypeTransition,
                    PrototypeCandidateUpdateAuditEvidence,
                ],
            ) -> tuple[
                PrototypeAgentState,
                tuple[Array, Array, Array, Array, Array, Array, Array],
            ]:
                transition, sidecar = inputs
                result = self.update_transition(carry, transition, sidecar)
                return result.state, outputs_from_result(result)

            final_state, outputs = jax.lax.scan(
                transition_with_candidate_audit_step,
                state,
                (transitions, gradient_joy_evidence),
            )
        elif self._experiential_memory is None:
            n = int(transitions.reward.shape[0])
            missing_input = self._missing_partner_policy_fusion_input()
            missing_feedback = self._missing_partner_policy_fusion_feedback()
            scan_partner_input = (
                partner_policy_fusion_input
                if partner_policy_fusion_input is not None
                else jax.tree.map(
                    lambda value: jnp.broadcast_to(value, (n,) + value.shape),
                    missing_input,
                )
            )
            scan_partner_feedback = (
                partner_policy_fusion_feedback
                if partner_policy_fusion_feedback is not None
                else jax.tree.map(
                    lambda value: jnp.broadcast_to(value, (n,) + value.shape),
                    missing_feedback,
                )
            )

            if gradient_joy_evidence is None:
                def transition_with_partner_step(
                    carry: PrototypeAgentState,
                    inputs: tuple[
                        PrototypeTransition,
                        PrototypePartnerPolicyFusionInput,
                        PrototypePartnerPolicyFusionFeedback,
                    ],
                ) -> tuple[
                    PrototypeAgentState,
                    tuple[Array, Array, Array, Array, Array, Array, Array],
                ]:
                    transition, fusion_input, fusion_feedback = inputs
                    result = self.update_transition(
                        carry,
                        transition,
                        partner_policy_fusion_input=(
                            fusion_input if use_partner_input else None
                        ),
                        partner_policy_fusion_feedback=(
                            fusion_feedback if use_partner_feedback else None
                        ),
                    )
                    return result.state, outputs_from_result(result)

                final_state, outputs = jax.lax.scan(
                    transition_with_partner_step,
                    state,
                    (
                        transitions,
                        scan_partner_input,
                        scan_partner_feedback,
                    ),
                )
            else:
                def transition_with_all_sidecars_step(
                    carry: PrototypeAgentState,
                    inputs: tuple[
                        PrototypeTransition,
                        PrototypeCandidateUpdateAuditEvidence,
                        PrototypePartnerPolicyFusionInput,
                        PrototypePartnerPolicyFusionFeedback,
                    ],
                ) -> tuple[
                    PrototypeAgentState,
                    tuple[Array, Array, Array, Array, Array, Array, Array],
                ]:
                    transition, audit_sidecar, fusion_input, fusion_feedback = inputs
                    result = self.update_transition(
                        carry,
                        transition,
                        audit_sidecar,
                        partner_policy_fusion_input=(
                            fusion_input if use_partner_input else None
                        ),
                        partner_policy_fusion_feedback=(
                            fusion_feedback if use_partner_feedback else None
                        ),
                    )
                    return result.state, outputs_from_result(result)

                final_state, outputs = jax.lax.scan(
                    transition_with_all_sidecars_step,
                    state,
                    (
                        transitions,
                        gradient_joy_evidence,
                        scan_partner_input,
                        scan_partner_feedback,
                    ),
                )
        else:
            n = int(transitions.reward.shape[0])
            missing_memory_input = self._missing_experiential_memory_input()
            scan_memory_input = (
                experiential_memory_input
                if experiential_memory_input is not None
                else jax.tree.map(
                    lambda value: jnp.broadcast_to(value, (n,) + value.shape),
                    missing_memory_input,
                )
            )
            if self._partner_policy_fusion is None:
                if gradient_joy_evidence is None:
                    def transition_with_memory_step(
                        carry: PrototypeAgentState,
                        inputs: tuple[
                            PrototypeTransition,
                            PrototypeExperientialMemoryInput,
                        ],
                    ) -> tuple[
                        PrototypeAgentState,
                        tuple[Array, Array, Array, Array, Array, Array, Array],
                    ]:
                        transition, memory_input = inputs
                        result = self.update_transition(
                            carry,
                            transition,
                            experiential_memory_input=(
                                memory_input if use_memory_input else None
                            ),
                        )
                        return result.state, outputs_from_result(result)

                    final_state, outputs = jax.lax.scan(
                        transition_with_memory_step,
                        state,
                        (transitions, scan_memory_input),
                    )
                else:
                    def transition_with_memory_and_candidate_audit_step(
                        carry: PrototypeAgentState,
                        inputs: tuple[
                            PrototypeTransition,
                            PrototypeCandidateUpdateAuditEvidence,
                            PrototypeExperientialMemoryInput,
                        ],
                    ) -> tuple[
                        PrototypeAgentState,
                        tuple[Array, Array, Array, Array, Array, Array, Array],
                    ]:
                        transition, audit_sidecar, memory_input = inputs
                        result = self.update_transition(
                            carry,
                            transition,
                            audit_sidecar,
                            experiential_memory_input=(
                                memory_input if use_memory_input else None
                            ),
                        )
                        return result.state, outputs_from_result(result)

                    final_state, outputs = jax.lax.scan(
                        transition_with_memory_and_candidate_audit_step,
                        state,
                        (
                            transitions,
                            gradient_joy_evidence,
                            scan_memory_input,
                        ),
                    )
            else:
                missing_partner_input = self._missing_partner_policy_fusion_input()
                missing_partner_feedback = (
                    self._missing_partner_policy_fusion_feedback()
                )
                scan_partner_input = (
                    partner_policy_fusion_input
                    if partner_policy_fusion_input is not None
                    else jax.tree.map(
                        lambda value: jnp.broadcast_to(
                            value,
                            (n,) + value.shape,
                        ),
                        missing_partner_input,
                    )
                )
                scan_partner_feedback = (
                    partner_policy_fusion_feedback
                    if partner_policy_fusion_feedback is not None
                    else jax.tree.map(
                        lambda value: jnp.broadcast_to(
                            value,
                            (n,) + value.shape,
                        ),
                        missing_partner_feedback,
                    )
                )
                if gradient_joy_evidence is None:
                    def transition_with_memory_and_partner_step(
                        carry: PrototypeAgentState,
                        inputs: tuple[
                            PrototypeTransition,
                            PrototypeExperientialMemoryInput,
                            PrototypePartnerPolicyFusionInput,
                            PrototypePartnerPolicyFusionFeedback,
                        ],
                    ) -> tuple[
                        PrototypeAgentState,
                        tuple[Array, Array, Array, Array, Array, Array, Array],
                    ]:
                        (
                            transition,
                            memory_input,
                            fusion_input,
                            fusion_feedback,
                        ) = inputs
                        result = self.update_transition(
                            carry,
                            transition,
                            experiential_memory_input=(
                                memory_input if use_memory_input else None
                            ),
                            partner_policy_fusion_input=(
                                fusion_input if use_partner_input else None
                            ),
                            partner_policy_fusion_feedback=(
                                fusion_feedback if use_partner_feedback else None
                            ),
                        )
                        return result.state, outputs_from_result(result)

                    final_state, outputs = jax.lax.scan(
                        transition_with_memory_and_partner_step,
                        state,
                        (
                            transitions,
                            scan_memory_input,
                            scan_partner_input,
                            scan_partner_feedback,
                        ),
                    )
                else:
                    def transition_with_every_sidecar_step(
                        carry: PrototypeAgentState,
                        inputs: tuple[
                            PrototypeTransition,
                            PrototypeCandidateUpdateAuditEvidence,
                            PrototypeExperientialMemoryInput,
                            PrototypePartnerPolicyFusionInput,
                            PrototypePartnerPolicyFusionFeedback,
                        ],
                    ) -> tuple[
                        PrototypeAgentState,
                        tuple[Array, Array, Array, Array, Array, Array, Array],
                    ]:
                        (
                            transition,
                            audit_sidecar,
                            memory_input,
                            fusion_input,
                            fusion_feedback,
                        ) = inputs
                        result = self.update_transition(
                            carry,
                            transition,
                            audit_sidecar,
                            experiential_memory_input=(
                                memory_input if use_memory_input else None
                            ),
                            partner_policy_fusion_input=(
                                fusion_input if use_partner_input else None
                            ),
                            partner_policy_fusion_feedback=(
                                fusion_feedback if use_partner_feedback else None
                            ),
                        )
                        return result.state, outputs_from_result(result)

                    final_state, outputs = jax.lax.scan(
                        transition_with_every_sidecar_step,
                        state,
                        (
                            transitions,
                            gradient_joy_evidence,
                            scan_memory_input,
                            scan_partner_input,
                            scan_partner_feedback,
                        ),
                    )
        (
            actions,
            td_errors,
            average_rewards,
            transition_valid,
            builder_learning_applied,
            candidate_update_audit_passed,
            audited_candidate_update_applied,
        ) = outputs
        return PrototypeArrayResult(
            state=final_state,
            actions=actions,
            oak_td_errors=td_errors,
            oak_average_rewards=average_rewards,
            transition_valid=transition_valid,
            state_builder_learning_applied=builder_learning_applied,
            gradient_sparks_joy=candidate_update_audit_passed,
            joyful_gradient_applied=audited_candidate_update_applied,
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
        if self._prototype_feature_lifecycle is not None:
            raise ValueError(
                "PrototypeAgent.curate is unavailable when "
                "prototype_feature_lifecycle owns the fixed feature bank"
            )
        current_oak_state = self._oak_component_state(state.oak_state)
        new_oak, new_oak_state = self._oak.curate(
            current_oak_state,
            key,
            available_feature_indices,
        )
        if new_oak is self._oak and new_oak_state is current_oak_state:
            return self, state
        new_config = PrototypeAgentConfig(
            oak=new_oak.config,
            option_search_control=self._config.option_search_control,
            world_model=self._config.world_model,
            world_model_ensemble=self._config.world_model_ensemble,
            model_replay_rehearsal=self._config.model_replay_rehearsal,
            recurrent_latent_world_model_ensemble=(
                self._config.recurrent_latent_world_model_ensemble
            ),
            dreaming=self._config.dreaming,
            buffer_capacity=self._config.buffer_capacity,
            n_dreams_per_step=self._config.n_dreams_per_step,
            dream_next_observation_mode=self._config.dream_next_observation_mode,
            horde_spec=self._config.horde_spec,
            horde_hidden_sizes=self._config.horde_hidden_sizes,
            horde_step_size=self._config.horde_step_size,
            ia=self._config.ia,
            partner_policy_fusion=self._config.partner_policy_fusion,
            experiential_memory=self._config.experiential_memory,
            experiential_memory_advantage_gate=(
                self._config.experiential_memory_advantage_gate
            ),
            gru_perception=self._config.gru_perception,
            state_builder=self._config.state_builder,
            learn_state_builder_from_world_model=(
                self._config.learn_state_builder_from_world_model
            ),
            representation_gradient_mixer=(
                self._config.representation_gradient_mixer
            ),
            gradient_joy=self._config.gradient_joy,
            learning_value_router=self._config.learning_value_router,
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
        if self._prototype_feature_lifecycle is not None:
            raise ValueError(
                "PrototypeAgent.maybe_curate is unavailable when "
                "prototype_feature_lifecycle owns the fixed feature bank"
            )
        n = self._config.auto_curate_every
        if n <= 0:
            return self, state
        if not bool(self._checkpoint_state_valid(state)):
            return self, state
        if not bool(_lifetime_words_multiple_of(state.step_words, n)):
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
        configured_specs = self._config.oak.stomp.subtask_specs
        template = (
            configured_specs[0]
            if configured_specs
            else SubtaskSpec(feature_index=0)
        )
        return feature_to_subtask_specs(
            self._oak_component_state(state.oak_state),
            n_subtasks=n_subtasks,
            threshold=template.threshold,
            pseudo_reward_scale=template.pseudo_reward_scale,
            max_option_steps=template.max_option_steps,
        )


def _prototype_config_bytes(config: dict[str, Any]) -> bytes:
    """Return the exact canonical JSON encoding of an agent config.

    Comparing this encoding, rather than Python containers, is load-bearing:
    Python considers ``True == 1 == 1.0`` even though those JSON values have
    different configuration semantics.
    """

    return json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _prototype_config_digest(config: dict[str, Any]) -> str:
    """Return a canonical SHA-256 digest for a serialized agent config."""

    return hashlib.sha256(_prototype_config_bytes(config)).hexdigest()


def _unambiguous_legacy_int32_counter_words(value: Any, *, name: str) -> Array:
    """Migrate non-negative, unsaturated int32 telemetry to exact words."""

    counter = jnp.asarray(value)
    if counter.dtype != jnp.dtype(jnp.int32):
        raise TypeError(f"legacy {name} must have dtype int32")
    if not bool(jnp.all(counter >= 0)):
        raise ValueError(f"negative legacy {name} indicates wrap")
    if not bool(jnp.all(counter < _INT32_MAX)):
        raise ValueError(f"saturated legacy {name} is ambiguous")
    low = counter.astype(jnp.uint32)
    return jnp.stack((jnp.zeros_like(low), low), axis=-1)


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
    feature_memory = agent.prototype_feature_memory
    atomic_feature_world_memory = (
        agent.config.prototype_atomic_feature_world_memory is not None
    )
    feature_world_model = (
        agent.config.prototype_feature_lifecycle is not None
        and agent.config.world_model is not None
        and not atomic_feature_world_memory
    )
    if agent.config.learning_value_router is not None:
        schema = PROTOTYPE_LEARNING_VALUE_ROUTER_CHECKPOINT_SCHEMA
    elif atomic_feature_world_memory:
        schema = PROTOTYPE_ATOMIC_FEATURE_WORLD_MEMORY_CHECKPOINT_SCHEMA
    elif feature_world_model:
        schema = PROTOTYPE_FEATURE_WORLD_MODEL_CHECKPOINT_SCHEMA
    elif feature_memory is not None:
        schema = PROTOTYPE_FEATURE_MEMORY_CHECKPOINT_SCHEMA
    elif agent.config.prototype_feature_utility_curation is not None:
        schema = PROTOTYPE_FEATURE_UTILITY_CURATION_CHECKPOINT_SCHEMA
    elif agent.config.prototype_feature_utility is not None:
        schema = PROTOTYPE_FEATURE_UTILITY_CHECKPOINT_SCHEMA
    else:
        schema = PROTOTYPE_CHECKPOINT_SCHEMA
    metadata: dict[str, Any] = {
        "schema": schema,
        "agent_config": config,
        "config_sha256": _prototype_config_digest(config),
    }
    if feature_memory is not None:
        metadata["feature_memory_schema_sha256"] = (
            feature_memory.schema_digest_hex
        )
    if atomic_feature_world_memory:
        atomic_digest = agent._prototype_atomic_feature_world_memory_digest
        if atomic_digest is None:
            raise RuntimeError("atomic checkpoint digest is unavailable")
        metadata["atomic_feature_world_memory_schema_sha256"] = bytes(
            int(value) for value in jax.device_get(atomic_digest).tolist()
        ).hex()
    save_checkpoint(
        state,
        path,
        metadata=metadata,
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
    the stored OaK decision was already selected for dispatch. Exact outer
    clocks make that migration narrower still: only a primed zero-transition
    v1 state is unambiguous, because v1 did not record autoreset observation
    events. Pre-v7 states that do not physically contain the new word clocks
    fail structurally; saturated telemetry is never used to invent history.
    """

    metadata = load_checkpoint_metadata(path)
    schema = metadata.get("schema")
    if schema not in {
        PROTOTYPE_LEARNING_VALUE_ROUTER_CHECKPOINT_SCHEMA,
        PROTOTYPE_ATOMIC_FEATURE_WORLD_MEMORY_CHECKPOINT_SCHEMA,
        PROTOTYPE_FEATURE_WORLD_MODEL_CHECKPOINT_SCHEMA,
        PROTOTYPE_FEATURE_MEMORY_CHECKPOINT_SCHEMA,
        PROTOTYPE_FEATURE_UTILITY_CURATION_CHECKPOINT_SCHEMA,
        PROTOTYPE_FEATURE_UTILITY_CHECKPOINT_SCHEMA,
        PROTOTYPE_CHECKPOINT_SCHEMA,
        _PROTOTYPE_CHECKPOINT_SCHEMA_V12,
        _PROTOTYPE_CHECKPOINT_SCHEMA_V11,
        _PROTOTYPE_CHECKPOINT_SCHEMA_V10,
        _PROTOTYPE_CHECKPOINT_SCHEMA_V9,
        _PROTOTYPE_CHECKPOINT_SCHEMA_V8,
        _PROTOTYPE_CHECKPOINT_SCHEMA_V7,
        _PROTOTYPE_CHECKPOINT_SCHEMA_V6,
        _PROTOTYPE_CHECKPOINT_SCHEMA_V5,
        _PROTOTYPE_CHECKPOINT_SCHEMA_V4,
        _PROTOTYPE_CHECKPOINT_SCHEMA_V3,
        _PROTOTYPE_CHECKPOINT_SCHEMA_V2,
        _PROTOTYPE_CHECKPOINT_SCHEMA_V1,
    }:
        raise ValueError(
            "checkpoint is not an Alberta PrototypeAgent "
            "v1 through v19 checkpoint"
        )
    config = metadata.get("agent_config")
    if not isinstance(config, dict):
        raise ValueError("prototype checkpoint is missing agent_config")
    expected_digest = metadata.get("config_sha256")
    if not isinstance(expected_digest, str) or expected_digest != (
        _prototype_config_digest(config)
    ):
        raise ValueError("prototype checkpoint config digest does not match")
    if schema in {
        _PROTOTYPE_CHECKPOINT_SCHEMA_V12,
        _PROTOTYPE_CHECKPOINT_SCHEMA_V11,
        _PROTOTYPE_CHECKPOINT_SCHEMA_V10,
    }:
        raise ValueError(
            "PrototypeAgent v10/v11/v12 checkpoints predate exact identity and "
            "fixed-trace state-builder clocks; migrate each unambiguous builder "
            "state with its standalone helper and resave under a current schema"
        )

    agent = PrototypeAgent.from_config(config)
    if _prototype_config_bytes(agent.to_config()) != _prototype_config_bytes(config):
        raise ValueError("prototype checkpoint agent_config is not canonical")
    if agent.config.learning_value_router is not None:
        if schema != PROTOTYPE_LEARNING_VALUE_ROUTER_CHECKPOINT_SCHEMA:
            raise ValueError(
                "learning_value_router requires a v19 PrototypeAgent checkpoint"
            )
    elif schema == PROTOTYPE_LEARNING_VALUE_ROUTER_CHECKPOINT_SCHEMA:
        raise ValueError(
            "v19 PrototypeAgent checkpoints require learning_value_router"
        )
    if (
        agent.config.prototype_feature_lifecycle is not None
        and schema in {
            _PROTOTYPE_CHECKPOINT_SCHEMA_V2,
            _PROTOTYPE_CHECKPOINT_SCHEMA_V1,
        }
    ):
        raise ValueError(
            "prototype_feature_lifecycle is unsupported by legacy v1/v2 "
            "PrototypeAgent checkpoints"
        )
    feature_config = agent.config.prototype_feature_lifecycle
    utility_config = agent.config.prototype_feature_utility
    utility_curation_config = (
        agent.config.prototype_feature_utility_curation
    )
    feature_memory = agent.prototype_feature_memory
    feature_memory_digest = metadata.get("feature_memory_schema_sha256")
    atomic_config = agent.config.prototype_atomic_feature_world_memory
    atomic_digest = metadata.get(
        "atomic_feature_world_memory_schema_sha256"
    )
    if atomic_config is not None:
        expected_atomic_digest = (
            agent._prototype_atomic_feature_world_memory_digest
        )
        if schema not in {
            PROTOTYPE_ATOMIC_FEATURE_WORLD_MEMORY_CHECKPOINT_SCHEMA,
            PROTOTYPE_LEARNING_VALUE_ROUTER_CHECKPOINT_SCHEMA,
        }:
            raise ValueError(
                "atomic feature/world/memory composition requires a v18 "
                "PrototypeAgent checkpoint"
            )
        if expected_atomic_digest is None:
            raise RuntimeError("atomic checkpoint digest is unavailable")
        expected_atomic_hex = bytes(
            int(value)
            for value in jax.device_get(expected_atomic_digest).tolist()
        ).hex()
        if not isinstance(atomic_digest, str) or atomic_digest != expected_atomic_hex:
            raise ValueError(
                "atomic feature/world/memory checkpoint schema digest does not match"
            )
    elif schema == PROTOTYPE_ATOMIC_FEATURE_WORLD_MEMORY_CHECKPOINT_SCHEMA:
        raise ValueError(
            "v18 PrototypeAgent checkpoints require the atomic composition"
        )
    elif atomic_digest is not None:
        raise ValueError(
            "PrototypeAgent checkpoint without the atomic composition contains "
            "atomic schema metadata"
        )
    feature_world_model = (
        feature_config is not None
        and agent.config.world_model is not None
        and atomic_config is None
    )
    if feature_world_model:
        if schema not in {
            PROTOTYPE_FEATURE_WORLD_MODEL_CHECKPOINT_SCHEMA,
            PROTOTYPE_LEARNING_VALUE_ROUTER_CHECKPOINT_SCHEMA,
        }:
            raise ValueError(
                "prototype feature/world-model composition requires a v17 "
                "PrototypeAgent checkpoint"
            )
    elif schema == PROTOTYPE_FEATURE_WORLD_MODEL_CHECKPOINT_SCHEMA:
        raise ValueError(
            "v17 PrototypeAgent checkpoints require feature lifecycle and the "
            "stable-base world model to be enabled"
        )
    if feature_memory is not None:
        if schema not in {
            PROTOTYPE_LEARNING_VALUE_ROUTER_CHECKPOINT_SCHEMA,
            PROTOTYPE_ATOMIC_FEATURE_WORLD_MEMORY_CHECKPOINT_SCHEMA,
            PROTOTYPE_FEATURE_WORLD_MODEL_CHECKPOINT_SCHEMA,
            PROTOTYPE_FEATURE_MEMORY_CHECKPOINT_SCHEMA,
        }:
            raise ValueError(
                "prototype feature memory requires a v16, v17, or v18 PrototypeAgent "
                "checkpoint"
            )
        if (
            not isinstance(feature_memory_digest, str)
            or feature_memory_digest != feature_memory.schema_digest_hex
        ):
            raise ValueError(
                "prototype feature-memory checkpoint schema digest does not match"
            )
    elif schema == PROTOTYPE_FEATURE_MEMORY_CHECKPOINT_SCHEMA:
        raise ValueError(
            "v16 PrototypeAgent checkpoints require feature memory to be enabled"
        )
    elif feature_memory_digest is not None:
        raise ValueError(
            "PrototypeAgent checkpoint without feature memory contains "
            "feature-memory metadata"
        )
    if (
        utility_curation_config is not None
        and schema
        not in {
            PROTOTYPE_LEARNING_VALUE_ROUTER_CHECKPOINT_SCHEMA,
            PROTOTYPE_ATOMIC_FEATURE_WORLD_MEMORY_CHECKPOINT_SCHEMA,
            PROTOTYPE_FEATURE_WORLD_MODEL_CHECKPOINT_SCHEMA,
            PROTOTYPE_FEATURE_MEMORY_CHECKPOINT_SCHEMA,
            PROTOTYPE_FEATURE_UTILITY_CURATION_CHECKPOINT_SCHEMA,
        }
    ):
        raise ValueError(
            "prototype_feature_utility_curation requires a v15, v16, or v17 "
            "PrototypeAgent checkpoint"
        )
    if utility_curation_config is None and schema == (
        PROTOTYPE_FEATURE_UTILITY_CURATION_CHECKPOINT_SCHEMA
    ):
        raise ValueError(
            "v15 PrototypeAgent checkpoints require feature utility curation "
            "to be enabled"
        )
    if (
        utility_config is not None
        and utility_curation_config is None
        and schema
        not in {
            PROTOTYPE_LEARNING_VALUE_ROUTER_CHECKPOINT_SCHEMA,
            PROTOTYPE_ATOMIC_FEATURE_WORLD_MEMORY_CHECKPOINT_SCHEMA,
            PROTOTYPE_FEATURE_WORLD_MODEL_CHECKPOINT_SCHEMA,
            PROTOTYPE_FEATURE_MEMORY_CHECKPOINT_SCHEMA,
            PROTOTYPE_FEATURE_UTILITY_CHECKPOINT_SCHEMA,
        }
    ):
        raise ValueError(
            "prototype_feature_utility requires a v14, v16, or v17 "
            "PrototypeAgent checkpoint"
        )
    if utility_config is None and schema in {
        PROTOTYPE_FEATURE_UTILITY_CHECKPOINT_SCHEMA,
        PROTOTYPE_FEATURE_UTILITY_CURATION_CHECKPOINT_SCHEMA,
        _PROTOTYPE_CHECKPOINT_SCHEMA_V8,
        _PROTOTYPE_CHECKPOINT_SCHEMA_V9,
    }:
        raise ValueError(
            "feature utility PrototypeAgent checkpoints require utility to be enabled"
        )
    if (
        feature_config is not None
        and feature_config.managed_horde_demons > 0
        and schema
        not in {
            PROTOTYPE_LEARNING_VALUE_ROUTER_CHECKPOINT_SCHEMA,
            PROTOTYPE_ATOMIC_FEATURE_WORLD_MEMORY_CHECKPOINT_SCHEMA,
            PROTOTYPE_FEATURE_WORLD_MODEL_CHECKPOINT_SCHEMA,
            PROTOTYPE_FEATURE_MEMORY_CHECKPOINT_SCHEMA,
            PROTOTYPE_CHECKPOINT_SCHEMA,
            PROTOTYPE_FEATURE_UTILITY_CHECKPOINT_SCHEMA,
            PROTOTYPE_FEATURE_UTILITY_CURATION_CHECKPOINT_SCHEMA,
        }
    ):
        raise ValueError(
            "managed feature/Horde composition requires a current "
            "PrototypeAgent checkpoint"
        )
    key = jr.key(0) if template_key is None else template_key
    template = agent.init(key)
    if schema in {
        PROTOTYPE_LEARNING_VALUE_ROUTER_CHECKPOINT_SCHEMA,
        PROTOTYPE_ATOMIC_FEATURE_WORLD_MEMORY_CHECKPOINT_SCHEMA,
        PROTOTYPE_FEATURE_WORLD_MODEL_CHECKPOINT_SCHEMA,
        PROTOTYPE_FEATURE_MEMORY_CHECKPOINT_SCHEMA,
        PROTOTYPE_FEATURE_UTILITY_CURATION_CHECKPOINT_SCHEMA,
        PROTOTYPE_FEATURE_UTILITY_CHECKPOINT_SCHEMA,
        PROTOTYPE_CHECKPOINT_SCHEMA,
        _PROTOTYPE_CHECKPOINT_SCHEMA_V9,
        _PROTOTYPE_CHECKPOINT_SCHEMA_V8,
        _PROTOTYPE_CHECKPOINT_SCHEMA_V7,
        _PROTOTYPE_CHECKPOINT_SCHEMA_V6,
        _PROTOTYPE_CHECKPOINT_SCHEMA_V5,
        _PROTOTYPE_CHECKPOINT_SCHEMA_V4,
        _PROTOTYPE_CHECKPOINT_SCHEMA_V3,
    }:
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
            member_update_count_words=(
                _unambiguous_legacy_int32_counter_words(
                    legacy_world.member_update_counts,
                    name="ensemble member_update_counts",
                )
            ),
            replay_member_update_counts=jnp.zeros_like(
                legacy_world.member_update_counts,
                dtype=jnp.int32,
            ),
            replay_member_update_count_words=jnp.zeros(
                (*legacy_world.member_update_counts.shape, 2),
                dtype=jnp.uint32,
            ),
            event_count=legacy_world.event_count,
            event_count_words=_unambiguous_legacy_int32_counter_words(
                legacy_world.event_count,
                name="ensemble event_count",
            ),
            replay_event_count=jnp.asarray(0, dtype=jnp.int32),
            replay_event_count_words=jnp.zeros((2,), dtype=jnp.uint32),
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
    if int(old_state.step_count) != 0:
        raise ValueError(
            "v1 prototype checkpoint observation-event history is ambiguous; "
            "only a provenance-verified primed zero-transition state can migrate"
        )
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
        observation_event_count=jnp.asarray(1, dtype=jnp.int32),
        observation_event_words=jnp.asarray((0, 1), dtype=jnp.uint32),
        step_count=old_state.step_count,
        step_words=jnp.asarray((0, 0), dtype=jnp.uint32),
    )
    if not bool(agent._checkpoint_state_valid(migrated)):
        raise ValueError("trusted v1 prototype checkpoint cache state is inconsistent")
    return agent, migrated


__all__ = [
    "PROTOTYPE_ATOMIC_FEATURE_WORLD_MEMORY_CHECKPOINT_SCHEMA",
    "PROTOTYPE_ATOMIC_FEATURE_WORLD_MEMORY_CONFIG_SCHEMA",
    "PROTOTYPE_ATOMIC_FEATURE_WORLD_MEMORY_MECHANISM_STATUS",
    "PROTOTYPE_ATOMIC_FEATURE_WORLD_MEMORY_SCIENTIFIC_PROMOTION_ALLOWED",
    "PROTOTYPE_CHECKPOINT_SCHEMA",
    "PROTOTYPE_FEATURE_MEMORY_CHECKPOINT_SCHEMA",
    "PROTOTYPE_FEATURE_WORLD_MODEL_CHECKPOINT_SCHEMA",
    "PROTOTYPE_FEATURE_WORLD_MODEL_SCHEMA_DIGEST_NBYTES",
    "PROTOTYPE_FEATURE_UTILITY_CURATION_CHECKPOINT_SCHEMA",
    "PROTOTYPE_FEATURE_UTILITY_CHECKPOINT_SCHEMA",
    "PROTOTYPE_LEARNING_VALUE_ROUTER_CHECKPOINT_SCHEMA",
    "PROTOTYPE_LIFETIME_COUNTER_DELTA_NBYTES",
    "PROTOTYPE_LIFETIME_COUNTER_NBYTES",
    "GRUPerceptionConfig",
    "GRUPerceptionState",
    "PrototypeAgent",
    "PrototypeAgentConfig",
    "PrototypeAgentState",
    "PrototypeAgentStateResourceMeasurement",
    "PrototypeArrayResult",
    "PrototypeAtomicFeatureWorldMemoryConfig",
    "PrototypeAtomicFeatureWorldMemoryDiagnostics",
    "PrototypeAtomicFeatureWorldMemoryResourceBudget",
    "PrototypeAtomicFeatureWorldMemoryState",
    "PrototypeBehaviorGradientDiagnostics",
    "PrototypeBehaviorGradientResult",
    "PrototypeCachedPrimitiveActionReplacement",
    "PrototypeDecision",
    "PrototypeExperientialMemoryDiagnostics",
    "PrototypeExperientialMemoryInput",
    "PrototypeExperientialMemoryResourceDeclaration",
    "PrototypeFeatureLifecycleIntegrationDiagnostics",
    "PrototypeFeatureMemoryIntegrationDiagnostics",
    "PrototypeFeatureOaKHordeState",
    "PrototypeFeatureOaKHordeUtilityCurationState",
    "PrototypeFeatureOaKHordeUtilityState",
    "PrototypeFeatureOaKState",
    "PrototypeFeatureRepresentationState",
    "PrototypeFeatureWorldModelState",
    "PrototypeFeatureUtilityIntegrationDiagnostics",
    "PrototypeFeatureUtilityCurationIntegrationDiagnostics",
    "PrototypeCandidateUpdateAuditEvidence",
    "PrototypeGradientJoyEvidence",
    "PrototypeInteractionState",
    "PrototypeLearningValueRouterState",
    "PrototypeMemoryInteractionState",
    "PrototypePartnerPolicyFusionDiagnostics",
    "PrototypePartnerPolicyFusionFeedback",
    "PrototypePartnerPolicyFusionInput",
    "PrototypeRecurrentLatentDiagnostics",
    "PrototypeRecurrentLatentWorldModelState",
    "PrototypeRTUFinalizationReceipt",
    "PrototypeRTUTransitionPreparation",
    "PrototypeTransition",
    "PrototypeTransitionDiagnostics",
    "PrototypeUpdateResult",
    "feature_to_subtask_specs",
    "load_prototype_checkpoint",
    "measure_prototype_agent_state_resources",
    "prototype_lifetime_counter_nbytes",
    "save_prototype_checkpoint",
]
