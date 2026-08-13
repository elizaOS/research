# mypy: disable-error-code="arg-type,attr-defined,call-arg"
"""Factorized online partner/world planning at a Prototype dispatch boundary.

This defaults-off adapter composes existing mechanisms rather than introducing
another learner:

* :class:`~alberta_framework.core.behavior_model.BehaviorModel` predicts one
  simultaneous partner action distribution from the stable raw observation;
* :class:`~alberta_framework.core.grounded_joint_world_model.GroundedJointWorldModel`
  predicts physical next observation, reward, and discount for every ordered
  ``(own_action, partner_action)`` cell; and
* :meth:`~alberta_framework.core.prototype_agent.PrototypeAgent.replace_cached_primitive_action`
  forms an authenticated counterfactual dispatch candidate.

The same partner belief is reused for every candidate own action.  This is a
simultaneous-action model: conditioning the partner belief on an own action
that the partner cannot yet observe would be a causal error.

Two agents are proposed and committed together.  A completed-transition call
updates both behavior models and both grounded models from the actually
executed joint action, prepares both next decisions from post-Prototype
candidates, and carries every child only if the complete paired proposal is
valid.  The adapter owns no replay, thresholds, checkpoint, or post-init RNG
draw.  The composing outer transaction—not this adapter—must bind each supplied
post-memory Prototype candidate to the exact action/reward/discount transition
that produced it.  The cached pre-replacement base action is a preparation
receipt and corruption guard, not an independently source-authenticated value
after public replacement has overwritten the Prototype action owner.  Planning
is disabled by default.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
import math
from collections.abc import Mapping
from numbers import Real
from typing import Any, Literal, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import Array
from jaxtyping import Bool, Float, Int, UInt

from alberta_framework.core.behavior_model import (
    BehaviorModel,
    BehaviorModelConfig,
    BehaviorModelState,
    BehaviorModelUpdateResult,
)
from alberta_framework.core.grounded_joint_world_model import (
    GroundedJointWorldModel,
    GroundedJointWorldModelConfig,
    GroundedJointWorldModelState,
    GroundedJointWorldUpdateResult,
)
from alberta_framework.core.prototype_agent import PrototypeAgent, PrototypeAgentState

PROTOTYPE_FACTORIZED_PARTNER_PLANNER_SCHEMA = "alberta.prototype-factorized-partner-planner.v1"
DEVELOPMENT_ONLY = True
SCIENTIFIC_PROMOTION_ALLOWED = False
CHECKPOINT_RESUME_CLAIMED = False
THRESHOLD_CALIBRATION_CLAIMED = False
POST_MEMORY_TRANSITION_BINDING_CLAIMED = False
BASE_FALLBACK_SOURCE_BINDING_CLAIMED = False
REPLAY_CAPACITY = 0
N_AGENTS = 2
CONFIG_TOKEN_NBYTES = 32
_INT32_MAX = 2**31 - 1


def _positive_int(value: Any, *, name: str, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}")
    return value


def _finite_positive(value: Any, *, name: str) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise ValueError(f"{name} must be a real non-boolean scalar")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _strict_array(
    value: Any,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: Any,
) -> Array:
    if getattr(value, "shape", None) != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if getattr(value, "dtype", None) != jnp.dtype(dtype):
        raise TypeError(f"{name} must have dtype {jnp.dtype(dtype)}")
    return jnp.asarray(value)


def _probabilities_valid(probabilities: Array) -> Bool[Array, ""]:
    values = jnp.asarray(probabilities, dtype=jnp.float32)
    return (
        jnp.all(jnp.isfinite(values))
        & jnp.all(values >= 0.0)
        & jnp.all(values <= 1.0)
        & jnp.isclose(jnp.sum(values), 1.0, atol=1.0e-5, rtol=0.0)
    )


def _tree_nbytes(tree: object) -> int:
    total = 0
    for leaf in jax.tree.leaves(tree):
        if getattr(leaf, "dtype", None) is not None and jax.dtypes.issubdtype(
            leaf.dtype, jax.dtypes.prng_key
        ):
            leaf = jr.key_data(leaf)
        total += int(getattr(leaf, "nbytes", 0))
    return total


def _behavior_counter_valid(state: BehaviorModelState) -> Bool[Array, ""]:
    words = state.step_words
    count = state.step_count
    below_saturation = (words[0] == jnp.asarray(0, dtype=jnp.uint32)) & (
        words[1] < jnp.asarray(_INT32_MAX, dtype=jnp.uint32)
    )
    expected = words[1].astype(jnp.int32)
    return (count >= 0) & jnp.where(
        below_saturation,
        count == expected,
        count == jnp.asarray(_INT32_MAX, dtype=jnp.int32),
    )


def _behavior_state_valid(
    state: BehaviorModelState,
    *,
    feature_dim: int,
    n_actions: int,
    min_probability: float,
) -> Bool[Array, ""]:
    """Validate every persistent BehaviorModel leaf used or carried by U1.

    Parameter reachability is deliberately not claimed: bounded development
    fixtures may inject finite weights and biases.  Diagnostic genesis and
    positive-step numeric bounds are exact consequences of BehaviorModel's
    update, however, and are enforced here.
    """

    weights = _strict_array(
        state.weights,
        name="behavior.weights",
        shape=(n_actions, feature_dim),
        dtype=jnp.float32,
    )
    bias = _strict_array(
        state.bias,
        name="behavior.bias",
        shape=(n_actions,),
        dtype=jnp.float32,
    )
    step_count = _strict_array(
        state.step_count,
        name="behavior.step_count",
        shape=(),
        dtype=jnp.int32,
    )
    _strict_array(
        state.step_words,
        name="behavior.step_words",
        shape=(2,),
        dtype=jnp.uint32,
    )
    nll_ema = _strict_array(
        state.nll_ema,
        name="behavior.nll_ema",
        shape=(),
        dtype=jnp.float32,
    )
    accuracy_ema = _strict_array(
        state.accuracy_ema,
        name="behavior.accuracy_ema",
        shape=(),
        dtype=jnp.float32,
    )
    confidence_ema = _strict_array(
        state.confidence_ema,
        name="behavior.confidence_ema",
        shape=(),
        dtype=jnp.float32,
    )
    if getattr(state.rng_key, "shape", None) != () or not jax.dtypes.issubdtype(
        getattr(state.rng_key, "dtype", None), jax.dtypes.prng_key
    ):
        raise TypeError("behavior.rng_key must be a scalar typed PRNG key")
    diagnostics = jnp.stack((nll_ema, accuracy_ema, confidence_ema))
    genesis = jnp.all(state.step_words == jnp.asarray(0, dtype=jnp.uint32))
    genesis_diagnostics_valid = jnp.all(diagnostics == jnp.asarray(0.0, dtype=jnp.float32))
    maximum_nll = -jnp.log(jnp.asarray(min_probability, dtype=jnp.float32))
    minimum_confidence = jnp.asarray(1.0 / n_actions, dtype=jnp.float32)
    positive_step_diagnostics_valid = (nll_ema <= maximum_nll) & (
        confidence_ema >= minimum_confidence
    )
    return (
        _behavior_counter_valid(state)
        & (step_count >= 0)
        & jnp.all(jnp.isfinite(weights))
        & jnp.all(jnp.isfinite(bias))
        & jnp.isfinite(nll_ema)
        & (nll_ema >= 0.0)
        & jnp.isfinite(accuracy_ema)
        & (accuracy_ema >= 0.0)
        & (accuracy_ema <= 1.0)
        & jnp.isfinite(confidence_ema)
        & (confidence_ema >= 0.0)
        & (confidence_ema <= 1.0)
        & jnp.where(
            genesis,
            genesis_diagnostics_valid,
            positive_step_diagnostics_valid,
        )
    )


def _increment_decision_id(decision_id: Array) -> UInt[Array, " 4"]:
    """Advance the Prototype's two-word generation while preserving lifecycle."""

    one = jnp.asarray(1, dtype=jnp.uint32)
    low = decision_id[3] + one
    carry = (low == jnp.asarray(0, dtype=jnp.uint32)).astype(jnp.uint32)
    return jnp.stack((decision_id[0], decision_id[1], decision_id[2] + carry, low))


@dataclasses.dataclass(frozen=True, slots=True)
class PrototypeFactorizedPartnerPlannerConfig:
    """Fixed dimensions and inherited online rates for the paired adapter."""

    observation_dim: int
    prototype_representation_dim: int
    n_actions: int = 2
    behavior_step_size: float = 0.05
    grounded_step_size: float = 0.02
    grounded_initialization_scale: float = 0.01
    planning_enabled: bool = False
    uniform_partner_belief: bool = False
    schema_version: str = PROTOTYPE_FACTORIZED_PARTNER_PLANNER_SCHEMA

    def __post_init__(self) -> None:
        _positive_int(self.observation_dim, name="observation_dim")
        _positive_int(
            self.prototype_representation_dim,
            name="prototype_representation_dim",
        )
        _positive_int(self.n_actions, name="n_actions", minimum=2)
        object.__setattr__(
            self,
            "behavior_step_size",
            _finite_positive(self.behavior_step_size, name="behavior_step_size"),
        )
        object.__setattr__(
            self,
            "grounded_step_size",
            _finite_positive(self.grounded_step_size, name="grounded_step_size"),
        )
        object.__setattr__(
            self,
            "grounded_initialization_scale",
            _finite_positive(
                self.grounded_initialization_scale,
                name="grounded_initialization_scale",
            ),
        )
        for name in ("planning_enabled", "uniform_partner_belief"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        if self.schema_version != PROTOTYPE_FACTORIZED_PARTNER_PLANNER_SCHEMA:
            raise ValueError("factorized partner-planner schema is unsupported")

    @property
    def target_dim(self) -> int:
        """Next stable observation plus reward and discount."""

        return self.observation_dim + 2

    def to_config(self) -> dict[str, object]:
        return {
            "type": type(self).__name__,
            "schema_version": self.schema_version,
            "development_only": True,
            "scientific_promotion_allowed": False,
            "checkpoint_resume_claimed": False,
            "threshold_calibration_claimed": False,
            "post_memory_transition_binding_claimed": False,
            "base_fallback_source_binding_claimed": False,
            "replay_capacity": 0,
            "observation_dim": self.observation_dim,
            "prototype_representation_dim": self.prototype_representation_dim,
            "n_actions": self.n_actions,
            "behavior_step_size": self.behavior_step_size,
            "grounded_step_size": self.grounded_step_size,
            "grounded_initialization_scale": self.grounded_initialization_scale,
            "planning_enabled": self.planning_enabled,
            "uniform_partner_belief": self.uniform_partner_belief,
        }

    @classmethod
    def from_config(
        cls,
        payload: Mapping[str, object],
    ) -> PrototypeFactorizedPartnerPlannerConfig:
        values = dict(payload)
        expected = {
            "type",
            "schema_version",
            "development_only",
            "scientific_promotion_allowed",
            "checkpoint_resume_claimed",
            "threshold_calibration_claimed",
            "post_memory_transition_binding_claimed",
            "base_fallback_source_binding_claimed",
            "replay_capacity",
            "observation_dim",
            "prototype_representation_dim",
            "n_actions",
            "behavior_step_size",
            "grounded_step_size",
            "grounded_initialization_scale",
            "planning_enabled",
            "uniform_partner_belief",
        }
        if set(values) != expected:
            raise ValueError("factorized partner-planner config fields do not match")
        if values.pop("type") != cls.__name__:
            raise ValueError("factorized partner-planner config type is unsupported")
        if values.pop("development_only") is not True:
            raise ValueError("factorized partner planner must remain development-only")
        if values.pop("scientific_promotion_allowed") is not False:
            raise ValueError("factorized partner planner cannot permit promotion")
        if values.pop("checkpoint_resume_claimed") is not False:
            raise ValueError("factorized partner planner has no checkpoint claim")
        if values.pop("threshold_calibration_claimed") is not False:
            raise ValueError("factorized partner planner has no threshold claim")
        if values.pop("post_memory_transition_binding_claimed") is not False:
            raise ValueError("post-memory transition binding is owned by the outer transaction")
        if values.pop("base_fallback_source_binding_claimed") is not False:
            raise ValueError("base/fallback source binding is not independently claimed")
        if values.pop("replay_capacity") != 0:
            raise ValueError("factorized partner planner has no replay")
        return cls(**values)


@chex.dataclass(frozen=True)
class FactorizedPartnerDecisionCache:
    """Decision-relevant persistent binding for one dispatch-time factorization.

    The cache is 307 bytes at ``D=8``, Prototype width ``R=12``, and ``A=2``.
    It authenticates every current decision projection against its live model
    and Prototype sources.  Parameters in the null space of the current input
    and the behavior model's unused RNG key are intentionally outside that
    decision identity.  ``base_action`` and ``base_action_guard`` are a
    preparation receipt and a narrow corruption check.  Public replacement
    overwrites the Prototype action owner, so the original base is no longer an
    independently available source: it is decision-irrelevant whenever the
    effective action equals the recomputed proposal and caller-attested on a
    fallback.  ``BASE_FALLBACK_SOURCE_BINDING_CLAIMED`` is therefore false.
    As with every in-memory consistency cache, coordinated replacement of
    sources and a coherently regenerated cache remains a composite-checkpoint
    provenance concern and is not claimed closed here.

    The safety mask is caller-owned, authenticated only during preparation,
    and deliberately not persisted.  A later source check binds the exact
    cached effective dispatch and proves that it is either the recomputed
    proposal or the cached safe base fallback, but cannot replay the original
    mask itself.
    """

    prototype_decision_id: UInt[Array, " 4"]
    prototype_representation: Float[Array, " prototype_representation_dim"]
    world_input: Float[Array, " observation_dim"]
    behavior_step_words: UInt[Array, " 2"]
    grounded_update_words: UInt[Array, " 2"]
    learned_partner_probabilities: Float[Array, " n_actions"]
    behavior_diagnostics: Float[Array, " 3"]
    world_raw_predictions: Float[Array, "n_actions n_actions target_dim"]
    base_action: Int[Array, ""]
    base_action_guard: Int[Array, ""]
    effective_action: Int[Array, ""]
    belief_valid: Bool[Array, ""]
    replacement_candidate_committed: Bool[Array, ""]
    planner_consumed: Bool[Array, ""]


@chex.dataclass(frozen=True)
class FactorizedPartnerPlannerAgentState:
    """One agent's two online models and current authenticated decision cache."""

    behavior: BehaviorModelState
    grounded: GroundedJointWorldModelState
    cache: FactorizedPartnerDecisionCache


@chex.dataclass(frozen=True)
class PrototypeFactorizedPartnerPlannerState:
    """Two sidecars committed on one joint-transition boundary."""

    config_token: UInt[Array, " 32"]
    agent_0: FactorizedPartnerPlannerAgentState
    agent_1: FactorizedPartnerPlannerAgentState


@dataclasses.dataclass(frozen=True, slots=True)
class PrototypeFactorizedPartnerPlannerResourceBudget:
    """Exact persistent JAX-array allocation for the fixed paired sidecar."""

    observation_dim: int
    prototype_representation_dim: int
    n_actions: int
    target_dim: int
    behavior_state_nbytes_per_agent: int
    grounded_state_nbytes_per_agent: int
    cache_nbytes_per_agent: int
    state_nbytes_per_agent: int
    config_token_nbytes: int
    pair_state_nbytes: int
    measured_agent_0_nbytes: int
    measured_agent_1_nbytes: int
    measured_pair_nbytes: int
    exact_tree_match: bool
    replay_capacity: int
    post_init_random_draws_per_event: int

    def to_dict(self) -> dict[str, int | bool]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True, slots=True)
class PrototypeFactorizedPartnerPlannerWorkBudget:
    """Exact adapter calls/products for one public operation.

    A grounded update attempt is one executed-cell prediction equivalent on
    the accepted child path.  A rejected child may skip that lower-level
    arithmetic, while the adapter's invocation schedule remains fixed.
    """

    operation: Literal["standalone_prepare", "completed_transition"]
    n_agents: int
    n_actions: int
    cache_authentication_evaluations: int
    behavior_probability_vector_evaluations: int
    grounded_joint_cell_prediction_equivalents: int
    expected_reward_marginalization_products: int
    prototype_replacement_candidates: int
    behavior_parameter_update_attempts: int
    grounded_parameter_update_attempts: int
    atomic_pair_commit_decisions: int
    environment_transition_proposals: int
    replay_updates: int
    post_init_random_draws: int
    fixed_adapter_invocation_schedule: bool

    def to_dict(self) -> dict[str, int | str | bool]:
        return dataclasses.asdict(self)


@chex.dataclass(frozen=True)
class PrototypeFactorizedPartnerPrepareDiagnostics:
    """Read-only model evaluation and replacement-candidate audit."""

    learned_partner_probabilities: Float[Array, "2 n_actions"]
    applied_partner_probabilities: Float[Array, "2 n_actions"]
    world_raw_predictions: Float[Array, "2 n_actions n_actions target_dim"]
    world_reward_cells: Float[Array, "2 n_actions n_actions"]
    world_cell_valid: Bool[Array, "2 n_actions n_actions"]
    expected_rewards: Float[Array, "2 n_actions"]
    base_actions: Int[Array, " 2"]
    proposed_actions: Int[Array, " 2"]
    effective_actions: Int[Array, " 2"]
    source_valid: Bool[Array, " 2"]
    replacement_candidate_committed: Bool[Array, " 2"]
    config_token_valid: Bool[Array, ""]
    pair_valid: Bool[Array, ""]
    pair_committed: Bool[Array, ""]


@chex.dataclass(frozen=True)
class PrototypeFactorizedPartnerPrepareResult:
    """Atomically prepared sidecars and selected Prototype dispatch states."""

    state: PrototypeFactorizedPartnerPlannerState
    prototype_agent_0: PrototypeAgentState
    prototype_agent_1: PrototypeAgentState
    diagnostics: PrototypeFactorizedPartnerPrepareDiagnostics


@chex.dataclass(frozen=True)
class PrototypeFactorizedPartnerTransitionDiagnostics:
    """Completed-transition updates and next paired preparation audit."""

    source_cache_valid: Bool[Array, " 2"]
    executed_actions: Int[Array, " 2"]
    observed_partner_actions: Int[Array, " 2"]
    behavior_losses: Float[Array, " 2"]
    behavior_update_applied: Bool[Array, " 2"]
    grounded_losses: Float[Array, " 2"]
    grounded_targets: Float[Array, "2 target_dim"]
    grounded_joint_action_indices: Int[Array, " 2"]
    grounded_update_applied: Bool[Array, " 2"]
    prediction_matches_cache: Bool[Array, " 2"]
    candidate_clock_aligned: Bool[Array, " 2"]
    candidate_generation_aligned: Bool[Array, " 2"]
    next_observations_match: Bool[Array, " 2"]
    next_prepare: PrototypeFactorizedPartnerPrepareDiagnostics
    candidate_valid: Bool[Array, ""]
    transaction_committed: Bool[Array, ""]


@chex.dataclass(frozen=True)
class PrototypeFactorizedPartnerTransitionResult:
    """One all-or-none paired sidecar/Prototype successor."""

    state: PrototypeFactorizedPartnerPlannerState
    prototype_agent_0: PrototypeAgentState
    prototype_agent_1: PrototypeAgentState
    diagnostics: PrototypeFactorizedPartnerTransitionDiagnostics


@dataclasses.dataclass(frozen=True)
class _AgentEvaluation:
    learned_probabilities: Array
    applied_probabilities: Array
    raw_predictions: Array
    reward_cells: Array
    cell_valid: Array
    expected_rewards: Array
    base_action: Array
    proposed_action: Array
    effective_action: Array
    source_valid: Array
    replacement_committed: Array
    cache: FactorizedPartnerDecisionCache
    selected_prototype: PrototypeAgentState


class PrototypeFactorizedPartnerPlanner:
    """Defaults-off paired adapter for two simultaneous Prototype agents."""

    def __init__(
        self,
        prototype: PrototypeAgent,
        config: PrototypeFactorizedPartnerPlannerConfig,
    ) -> None:
        if not isinstance(prototype, PrototypeAgent):
            raise TypeError("prototype must be a PrototypeAgent")
        if type(config) is not PrototypeFactorizedPartnerPlannerConfig:
            raise TypeError("config must be an exact planner config")
        if prototype.config.oak.n_primitive_actions != config.n_actions:
            raise ValueError("Prototype and factorized planner action counts must match")
        if prototype.config.oak.observation_dim != config.prototype_representation_dim:
            raise ValueError("Prototype and planner representation widths must match")
        self._prototype = prototype
        self._config = config
        token_payload = json.dumps(
            {
                "planner": config.to_config(),
                "prototype": prototype.to_config(),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        self._config_token = jnp.asarray(
            tuple(hashlib.sha256(token_payload).digest()),
            dtype=jnp.uint8,
        )
        self._behavior = BehaviorModel(
            BehaviorModelConfig(
                n_actions=config.n_actions,
                step_size=config.behavior_step_size,
            )
        )
        self._grounded = GroundedJointWorldModel(
            GroundedJointWorldModelConfig(
                representation_dim=config.observation_dim,
                target_observation_dim=config.observation_dim,
                n_focal_actions=config.n_actions,
                n_partner_actions=config.n_actions,
                step_size=config.grounded_step_size,
                initialization_scale=config.grounded_initialization_scale,
            )
        )

    @property
    def config(self) -> PrototypeFactorizedPartnerPlannerConfig:
        return self._config

    @property
    def behavior_model(self) -> BehaviorModel:
        return self._behavior

    @property
    def grounded_world_model(self) -> GroundedJointWorldModel:
        return self._grounded

    def _neutral_cache(self) -> FactorizedPartnerDecisionCache:
        cfg = self._config
        return FactorizedPartnerDecisionCache(
            prototype_decision_id=jnp.zeros((4,), dtype=jnp.uint32),
            prototype_representation=jnp.zeros(
                (cfg.prototype_representation_dim,), dtype=jnp.float32
            ),
            world_input=jnp.zeros((cfg.observation_dim,), dtype=jnp.float32),
            behavior_step_words=jnp.zeros((2,), dtype=jnp.uint32),
            grounded_update_words=jnp.zeros((2,), dtype=jnp.uint32),
            learned_partner_probabilities=jnp.full(
                (cfg.n_actions,), 1.0 / cfg.n_actions, dtype=jnp.float32
            ),
            behavior_diagnostics=jnp.zeros((3,), dtype=jnp.float32),
            world_raw_predictions=jnp.zeros(
                (cfg.n_actions, cfg.n_actions, cfg.target_dim), dtype=jnp.float32
            ),
            base_action=jnp.asarray(-1, dtype=jnp.int32),
            base_action_guard=jnp.asarray(0, dtype=jnp.int32),
            effective_action=jnp.asarray(-1, dtype=jnp.int32),
            belief_valid=jnp.asarray(False, dtype=jnp.bool_),
            replacement_candidate_committed=jnp.asarray(False, dtype=jnp.bool_),
            planner_consumed=jnp.asarray(False, dtype=jnp.bool_),
        )

    def init(self, key: Array) -> PrototypeFactorizedPartnerPlannerState:
        """Initialize both model pairs; all later adapter calls are RNG-free."""

        if getattr(key, "shape", None) != () or not jax.dtypes.issubdtype(
            getattr(key, "dtype", None), jax.dtypes.prng_key
        ):
            raise TypeError("key must be a scalar typed PRNG key")
        behavior_0_key, grounded_0_key, behavior_1_key, grounded_1_key = jr.split(key, 4)
        neutral_0 = self._neutral_cache()
        neutral_1 = self._neutral_cache()
        return PrototypeFactorizedPartnerPlannerState(
            config_token=self._config_token,
            agent_0=FactorizedPartnerPlannerAgentState(
                behavior=self._behavior.init(self._config.observation_dim, behavior_0_key),
                grounded=self._grounded.init(grounded_0_key),
                cache=neutral_0,
            ),
            agent_1=FactorizedPartnerPlannerAgentState(
                behavior=self._behavior.init(self._config.observation_dim, behavior_1_key),
                grounded=self._grounded.init(grounded_1_key),
                cache=neutral_1,
            ),
        )

    def resource_budget(
        self,
        state: PrototypeFactorizedPartnerPlannerState,
    ) -> PrototypeFactorizedPartnerPlannerResourceBudget:
        """Return formulas and an exact measurement of the supplied pair state."""

        cfg = self._config
        behavior_bytes = 4 * (cfg.n_actions * cfg.observation_dim + cfg.n_actions + 8)
        grounded_bytes = 4 * (cfg.n_actions**2 * cfg.target_dim * (cfg.observation_dim + 1) + 3)
        cache_bytes = (
            59
            + 4 * cfg.observation_dim
            + 4 * cfg.prototype_representation_dim
            + 4 * cfg.n_actions
            + 4 * cfg.n_actions**2 * cfg.target_dim
        )
        per_agent = behavior_bytes + grounded_bytes + cache_bytes
        measured_0 = _tree_nbytes(state.agent_0)
        measured_1 = _tree_nbytes(state.agent_1)
        measured_pair = _tree_nbytes(state)
        pair_bytes = 2 * per_agent + CONFIG_TOKEN_NBYTES
        exact = measured_0 == per_agent and measured_1 == per_agent and measured_pair == pair_bytes
        return PrototypeFactorizedPartnerPlannerResourceBudget(
            observation_dim=cfg.observation_dim,
            prototype_representation_dim=cfg.prototype_representation_dim,
            n_actions=cfg.n_actions,
            target_dim=cfg.target_dim,
            behavior_state_nbytes_per_agent=behavior_bytes,
            grounded_state_nbytes_per_agent=grounded_bytes,
            cache_nbytes_per_agent=cache_bytes,
            state_nbytes_per_agent=per_agent,
            config_token_nbytes=CONFIG_TOKEN_NBYTES,
            pair_state_nbytes=pair_bytes,
            measured_agent_0_nbytes=measured_0,
            measured_agent_1_nbytes=measured_1,
            measured_pair_nbytes=measured_pair,
            exact_tree_match=exact,
            replay_capacity=0,
            post_init_random_draws_per_event=0,
        )

    def standalone_prepare_work_budget(
        self,
    ) -> PrototypeFactorizedPartnerPlannerWorkBudget:
        """Return fixed logical work for one direct :meth:`prepare_pair` call."""

        actions = self._config.n_actions
        return PrototypeFactorizedPartnerPlannerWorkBudget(
            operation="standalone_prepare",
            n_agents=N_AGENTS,
            n_actions=actions,
            cache_authentication_evaluations=0,
            behavior_probability_vector_evaluations=N_AGENTS,
            grounded_joint_cell_prediction_equivalents=N_AGENTS * actions**2,
            expected_reward_marginalization_products=N_AGENTS * actions**2,
            prototype_replacement_candidates=N_AGENTS,
            behavior_parameter_update_attempts=0,
            grounded_parameter_update_attempts=0,
            atomic_pair_commit_decisions=1,
            environment_transition_proposals=0,
            replay_updates=0,
            post_init_random_draws=0,
            fixed_adapter_invocation_schedule=True,
        )

    def completed_transition_work_budget(
        self,
    ) -> PrototypeFactorizedPartnerPlannerWorkBudget:
        """Return fixed work for authentication, learning, and next prepare."""

        actions = self._config.n_actions
        return PrototypeFactorizedPartnerPlannerWorkBudget(
            operation="completed_transition",
            n_agents=N_AGENTS,
            n_actions=actions,
            cache_authentication_evaluations=N_AGENTS,
            behavior_probability_vector_evaluations=4 * N_AGENTS,
            grounded_joint_cell_prediction_equivalents=(2 * N_AGENTS * actions**2 + N_AGENTS),
            expected_reward_marginalization_products=(2 * N_AGENTS * actions**2),
            prototype_replacement_candidates=N_AGENTS,
            behavior_parameter_update_attempts=N_AGENTS,
            grounded_parameter_update_attempts=N_AGENTS,
            atomic_pair_commit_decisions=2,
            environment_transition_proposals=0,
            replay_updates=0,
            post_init_random_draws=0,
            fixed_adapter_invocation_schedule=True,
        )

    def _model_clock_aligned(
        self,
        planner_state: FactorizedPartnerPlannerAgentState,
        prototype_state: PrototypeAgentState,
    ) -> Array:
        return (
            _behavior_state_valid(
                planner_state.behavior,
                feature_dim=self._config.observation_dim,
                n_actions=self._config.n_actions,
                min_probability=self._behavior.config.min_probability,
            )
            & jnp.array_equal(
                planner_state.behavior.step_words,
                planner_state.grounded.update_words,
            )
            & (planner_state.behavior.step_count == planner_state.grounded.update_count)
            & jnp.array_equal(
                prototype_state.step_words,
                planner_state.behavior.step_words,
            )
            & (prototype_state.step_count == planner_state.behavior.step_count)
        )

    def _world_cells(
        self,
        state: GroundedJointWorldModelState,
        world_input: Array,
    ) -> tuple[Array, Array, Array]:
        cfg = self._config
        predictions = tuple(
            self._grounded.predict(
                state,
                world_input,
                jnp.asarray(own_action, dtype=jnp.int32),
                jnp.asarray(partner_action, dtype=jnp.int32),
            )
            for own_action in range(cfg.n_actions)
            for partner_action in range(cfg.n_actions)
        )
        raw = jnp.stack(tuple(item.raw_predictions for item in predictions)).reshape(
            (cfg.n_actions, cfg.n_actions, cfg.target_dim)
        )
        valid = jnp.stack(tuple(item.valid for item in predictions)).reshape(
            (cfg.n_actions, cfg.n_actions)
        )
        rewards = raw[:, :, cfg.observation_dim]
        return raw, rewards, valid

    def _evaluate_agent(
        self,
        planner_state: FactorizedPartnerPlannerAgentState,
        prototype_state: PrototypeAgentState,
        safety_action_mask: Array,
    ) -> _AgentEvaluation:
        cfg = self._config
        raw = _strict_array(
            prototype_state.current_raw_observation,
            name="prototype.current_raw_observation",
            shape=(cfg.observation_dim,),
            dtype=jnp.float32,
        )
        representation = _strict_array(
            prototype_state.current_representation,
            name="prototype.current_representation",
            shape=(cfg.prototype_representation_dim,),
            dtype=jnp.float32,
        )
        mask = _strict_array(
            safety_action_mask,
            name="safety_action_mask",
            shape=(cfg.n_actions,),
            dtype=jnp.bool_,
        )
        learned = self._behavior.predict_probabilities(planner_state.behavior, raw)
        applied = (
            jnp.full((cfg.n_actions,), 1.0 / cfg.n_actions, dtype=jnp.float32)
            if cfg.uniform_partner_belief
            else learned
        )
        raw_predictions, reward_cells, cell_valid = self._world_cells(
            planner_state.grounded,
            raw,
        )
        expected_rewards = reward_cells @ applied
        proposed_action = jnp.argmax(expected_rewards).astype(jnp.int32)
        base_action = prototype_state.current_action
        replacement = self._prototype.replace_cached_primitive_action(
            prototype_state,
            decision_id=prototype_state.current_decision_id,
            decision_observation=representation,
            proposed_action=proposed_action,
            safety_action_mask=mask,
        )
        effective_action = jnp.where(
            jnp.asarray(cfg.planning_enabled, dtype=jnp.bool_),
            replacement.action,
            base_action,
        ).astype(jnp.int32)
        effective_index = jnp.clip(effective_action, 0, cfg.n_actions - 1)
        effective_action_safe = (
            (effective_action >= 0) & (effective_action < cfg.n_actions) & mask[effective_index]
        )
        selected = cast(
            PrototypeAgentState,
            jax.lax.cond(
                jnp.asarray(cfg.planning_enabled, dtype=jnp.bool_),
                lambda _: replacement.state,
                lambda _: prototype_state,
                operand=None,
            ),
        )
        belief_valid = _probabilities_valid(learned) & _probabilities_valid(applied)
        source_valid = (
            self._prototype.validate_state(prototype_state)
            & self._model_clock_aligned(planner_state, prototype_state)
            & jnp.all(jnp.isfinite(raw))
            & jnp.all(jnp.isfinite(representation))
            & belief_valid
            & jnp.all(cell_valid)
            & jnp.all(jnp.isfinite(expected_rewards))
            & (base_action >= 0)
            & (base_action < cfg.n_actions)
            & jnp.any(mask)
            & replacement.committed
            & effective_action_safe
            & self._prototype.validate_state(selected)
            & (selected.current_action == effective_action)
        )
        cache = FactorizedPartnerDecisionCache(
            prototype_decision_id=prototype_state.current_decision_id,
            prototype_representation=representation,
            world_input=raw,
            behavior_step_words=planner_state.behavior.step_words,
            grounded_update_words=planner_state.grounded.update_words,
            learned_partner_probabilities=learned,
            behavior_diagnostics=jnp.stack(
                (
                    planner_state.behavior.nll_ema,
                    planner_state.behavior.accuracy_ema,
                    planner_state.behavior.confidence_ema,
                )
            ),
            world_raw_predictions=raw_predictions,
            base_action=base_action,
            base_action_guard=jnp.bitwise_not(base_action),
            effective_action=effective_action,
            belief_valid=belief_valid,
            replacement_candidate_committed=replacement.committed,
            planner_consumed=jnp.asarray(cfg.planning_enabled, dtype=jnp.bool_),
        )
        return _AgentEvaluation(
            learned_probabilities=learned,
            applied_probabilities=applied,
            raw_predictions=raw_predictions,
            reward_cells=reward_cells,
            cell_valid=cell_valid,
            expected_rewards=expected_rewards,
            base_action=base_action,
            proposed_action=proposed_action,
            effective_action=effective_action,
            source_valid=source_valid,
            replacement_committed=replacement.committed,
            cache=cache,
            selected_prototype=selected,
        )

    def _prepare_pair_impl(
        self,
        state: PrototypeFactorizedPartnerPlannerState,
        prototype_agent_0: PrototypeAgentState,
        prototype_agent_1: PrototypeAgentState,
        safety_action_masks: Array,
    ) -> PrototypeFactorizedPartnerPrepareResult:
        masks = _strict_array(
            safety_action_masks,
            name="safety_action_masks",
            shape=(N_AGENTS, self._config.n_actions),
            dtype=jnp.bool_,
        )
        evaluations = (
            self._evaluate_agent(state.agent_0, prototype_agent_0, masks[0]),
            self._evaluate_agent(state.agent_1, prototype_agent_1, masks[1]),
        )
        source_valid = jnp.stack(tuple(item.source_valid for item in evaluations))
        config_token_valid = jnp.array_equal(state.config_token, self._config_token)
        pair_valid = config_token_valid & jnp.all(source_valid)
        candidate_state = PrototypeFactorizedPartnerPlannerState(
            config_token=state.config_token,
            agent_0=state.agent_0.replace(cache=evaluations[0].cache),
            agent_1=state.agent_1.replace(cache=evaluations[1].cache),
        )
        candidate_prototypes = (
            evaluations[0].selected_prototype,
            evaluations[1].selected_prototype,
        )
        final_state, final_0, final_1 = cast(
            tuple[
                PrototypeFactorizedPartnerPlannerState,
                PrototypeAgentState,
                PrototypeAgentState,
            ],
            jax.lax.cond(
                pair_valid,
                lambda _: (candidate_state, candidate_prototypes[0], candidate_prototypes[1]),
                lambda _: (state, prototype_agent_0, prototype_agent_1),
                operand=None,
            ),
        )
        diagnostics = PrototypeFactorizedPartnerPrepareDiagnostics(
            learned_partner_probabilities=jnp.stack(
                tuple(item.learned_probabilities for item in evaluations)
            ),
            applied_partner_probabilities=jnp.stack(
                tuple(item.applied_probabilities for item in evaluations)
            ),
            world_raw_predictions=jnp.stack(tuple(item.raw_predictions for item in evaluations)),
            world_reward_cells=jnp.stack(tuple(item.reward_cells for item in evaluations)),
            world_cell_valid=jnp.stack(tuple(item.cell_valid for item in evaluations)),
            expected_rewards=jnp.stack(tuple(item.expected_rewards for item in evaluations)),
            base_actions=jnp.stack(tuple(item.base_action for item in evaluations)),
            proposed_actions=jnp.stack(tuple(item.proposed_action for item in evaluations)),
            effective_actions=jnp.stack(tuple(item.effective_action for item in evaluations)),
            source_valid=source_valid,
            replacement_candidate_committed=jnp.stack(
                tuple(item.replacement_committed for item in evaluations)
            ),
            config_token_valid=config_token_valid,
            pair_valid=pair_valid,
            pair_committed=pair_valid,
        )
        return PrototypeFactorizedPartnerPrepareResult(
            state=final_state,
            prototype_agent_0=final_0,
            prototype_agent_1=final_1,
            diagnostics=diagnostics,
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def prepare_pair(
        self,
        state: PrototypeFactorizedPartnerPlannerState,
        prototype_agent_0: PrototypeAgentState,
        prototype_agent_1: PrototypeAgentState,
        safety_action_masks: Array,
    ) -> PrototypeFactorizedPartnerPrepareResult:
        """Read both authenticated Prototype decisions and prepare them together."""

        return self._prepare_pair_impl(
            state,
            prototype_agent_0,
            prototype_agent_1,
            safety_action_masks,
        )

    def _cache_matches_agent(
        self,
        planner_state: FactorizedPartnerPlannerAgentState,
        prototype_state: PrototypeAgentState,
    ) -> Array:
        cfg = self._config
        cache = planner_state.cache
        learned = self._behavior.predict_probabilities(
            planner_state.behavior,
            prototype_state.current_raw_observation,
        )
        applied = (
            jnp.full((cfg.n_actions,), 1.0 / cfg.n_actions, dtype=jnp.float32)
            if cfg.uniform_partner_belief
            else learned
        )
        raw_predictions, reward_cells, cell_valid = self._world_cells(
            planner_state.grounded,
            prototype_state.current_raw_observation,
        )
        expected_rewards = reward_cells @ applied
        proposed_action = jnp.argmax(expected_rewards).astype(jnp.int32)
        # The original base action is overwritten in the selected Prototype
        # state when a proposal is applied.  Persisting it with its complement
        # is therefore a preparation receipt/corruption check, not independent
        # post-replacement source authentication.
        effective_action_allowed = jnp.where(
            jnp.asarray(cfg.planning_enabled, dtype=jnp.bool_),
            (cache.effective_action == proposed_action)
            | (cache.effective_action == cache.base_action),
            cache.effective_action == cache.base_action,
        )
        return (
            self._prototype.validate_state(prototype_state)
            & self._model_clock_aligned(planner_state, prototype_state)
            & jnp.array_equal(cache.prototype_decision_id, prototype_state.current_decision_id)
            & jnp.array_equal(
                cache.prototype_representation,
                prototype_state.current_representation,
            )
            & jnp.array_equal(cache.world_input, prototype_state.current_raw_observation)
            & jnp.array_equal(cache.behavior_step_words, planner_state.behavior.step_words)
            & jnp.array_equal(cache.grounded_update_words, planner_state.grounded.update_words)
            & jnp.array_equal(cache.learned_partner_probabilities, learned)
            & jnp.array_equal(
                cache.behavior_diagnostics,
                jnp.stack(
                    (
                        planner_state.behavior.nll_ema,
                        planner_state.behavior.accuracy_ema,
                        planner_state.behavior.confidence_ema,
                    )
                ),
            )
            & jnp.array_equal(cache.world_raw_predictions, raw_predictions)
            & jnp.array_equal(cache.base_action_guard, jnp.bitwise_not(cache.base_action))
            & effective_action_allowed
            & jnp.array_equal(cache.effective_action, prototype_state.current_action)
            & (cache.base_action >= 0)
            & (cache.base_action < cfg.n_actions)
            & (
                cache.belief_valid
                == (_probabilities_valid(learned) & _probabilities_valid(applied))
            )
            & cache.belief_valid
            & cache.replacement_candidate_committed
            & (cache.planner_consumed == jnp.asarray(cfg.planning_enabled))
            & jnp.all(cell_valid)
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def authenticate_pair(
        self,
        state: PrototypeFactorizedPartnerPlannerState,
        prototype_agent_0: PrototypeAgentState,
        prototype_agent_1: PrototypeAgentState,
    ) -> Bool[Array, " 2"]:
        """Recompute both current caches against their complete live sources."""

        return jnp.stack(
            (
                jnp.array_equal(state.config_token, self._config_token)
                & self._cache_matches_agent(state.agent_0, prototype_agent_0),
                jnp.array_equal(state.config_token, self._config_token)
                & self._cache_matches_agent(state.agent_1, prototype_agent_1),
            )
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def completed_transition(
        self,
        state: PrototypeFactorizedPartnerPlannerState,
        prototype_agent_0: PrototypeAgentState,
        prototype_agent_1: PrototypeAgentState,
        post_memory_agent_0: PrototypeAgentState,
        post_memory_agent_1: PrototypeAgentState,
        executed_actions: Array,
        rewards: Array,
        next_observations: Array,
        discount: Array,
        safety_action_masks: Array,
    ) -> PrototypeFactorizedPartnerTransitionResult:
        """Learn from one completed dyadic event and prepare both next actions.

        ``post_memory_agent_*`` must be the already-validated Prototype
        candidates whose current actions include every in-Prototype dispatch
        authority (including experiential memory) but no factorized-planner
        replacement.  Their actions therefore become the persistent next-cache
        ``base_action`` values.  This adapter checks their next generation,
        clock, and raw observation.  The composing outer transaction owns the
        stronger proof that they were produced from these exact executed
        actions, rewards, and discount; this method does not claim that binding.
        """

        cfg = self._config
        actions = _strict_array(
            executed_actions,
            name="executed_actions",
            shape=(N_AGENTS,),
            dtype=jnp.int32,
        )
        reward_values = _strict_array(
            rewards,
            name="rewards",
            shape=(N_AGENTS,),
            dtype=jnp.float32,
        )
        next_values = _strict_array(
            next_observations,
            name="next_observations",
            shape=(N_AGENTS, cfg.observation_dim),
            dtype=jnp.float32,
        )
        discount_value = _strict_array(
            discount,
            name="discount",
            shape=(),
            dtype=jnp.float32,
        )
        masks = _strict_array(
            safety_action_masks,
            name="safety_action_masks",
            shape=(N_AGENTS, cfg.n_actions),
            dtype=jnp.bool_,
        )
        source_prototypes = (prototype_agent_0, prototype_agent_1)
        post_memory = (post_memory_agent_0, post_memory_agent_1)
        planner_states = (state.agent_0, state.agent_1)
        source_cache_valid = self.authenticate_pair(
            state,
            prototype_agent_0,
            prototype_agent_1,
        )
        action_matches = jnp.stack(
            tuple(
                actions[index] == source_prototypes[index].current_action
                for index in range(N_AGENTS)
            )
        )

        # Agent 0 owns action/reward/observation 0 and observes action 1 as its
        # partner. Agent 1 uses the exact reversed orientation.
        partner_actions = jnp.stack((actions[1], actions[0])).astype(jnp.int32)
        behavior_updates: tuple[BehaviorModelUpdateResult, BehaviorModelUpdateResult] = (
            self._behavior.update(
                planner_states[0].behavior,
                source_prototypes[0].current_raw_observation,
                partner_actions[0],
            ),
            self._behavior.update(
                planner_states[1].behavior,
                source_prototypes[1].current_raw_observation,
                partner_actions[1],
            ),
        )
        grounded_updates: tuple[
            GroundedJointWorldUpdateResult,
            GroundedJointWorldUpdateResult,
        ] = (
            self._grounded.update(
                planner_states[0].grounded,
                source_prototypes[0].current_raw_observation,
                actions[0],
                partner_actions[0],
                next_values[0],
                reward_values[0],
                discount_value,
            ),
            self._grounded.update(
                planner_states[1].grounded,
                source_prototypes[1].current_raw_observation,
                actions[1],
                partner_actions[1],
                next_values[1],
                reward_values[1],
                discount_value,
            ),
        )
        current_learned = jnp.stack(
            tuple(
                self._behavior.predict_probabilities(
                    planner_states[index].behavior,
                    source_prototypes[index].current_raw_observation,
                )
                for index in range(N_AGENTS)
            )
        )
        behavior_prediction_matches = jnp.stack(
            tuple(
                jnp.array_equal(behavior_updates[index].probabilities, current_learned[index])
                for index in range(N_AGENTS)
            )
        )
        grounded_prediction_matches = jnp.stack(
            tuple(
                jnp.array_equal(
                    grounded_updates[index].prediction.raw_predictions,
                    planner_states[index].cache.world_raw_predictions[
                        actions[index], partner_actions[index]
                    ],
                )
                for index in range(N_AGENTS)
            )
        )
        prediction_matches_cache = behavior_prediction_matches & grounded_prediction_matches
        next_observations_match = jnp.stack(
            tuple(
                jnp.array_equal(next_values[index], post_memory[index].current_raw_observation)
                for index in range(N_AGENTS)
            )
        )
        candidate_clock_aligned = jnp.stack(
            tuple(
                jnp.array_equal(
                    behavior_updates[index].post_step_words,
                    grounded_updates[index].post_update_words,
                )
                & jnp.array_equal(
                    behavior_updates[index].post_step_words,
                    post_memory[index].step_words,
                )
                & (
                    behavior_updates[index].state.step_count
                    == grounded_updates[index].state.update_count
                )
                & (behavior_updates[index].state.step_count == post_memory[index].step_count)
                for index in range(N_AGENTS)
            )
        )
        candidate_generation_aligned = jnp.stack(
            tuple(
                jnp.array_equal(
                    post_memory[index].current_decision_id,
                    _increment_decision_id(source_prototypes[index].current_decision_id),
                )
                for index in range(N_AGENTS)
            )
        )
        update_applied = jnp.stack(
            tuple(
                behavior_updates[index].update_applied
                & grounded_updates[index].update_applied
                & grounded_updates[index].diagnostics.applied
                for index in range(N_AGENTS)
            )
        )
        candidate_sidecars = PrototypeFactorizedPartnerPlannerState(
            config_token=state.config_token,
            agent_0=planner_states[0].replace(
                behavior=behavior_updates[0].state,
                grounded=grounded_updates[0].state,
            ),
            agent_1=planner_states[1].replace(
                behavior=behavior_updates[1].state,
                grounded=grounded_updates[1].state,
            ),
        )
        next_prepare = self._prepare_pair_impl(
            candidate_sidecars,
            post_memory_agent_0,
            post_memory_agent_1,
            masks,
        )
        candidate_valid = (
            jnp.all(source_cache_valid)
            & jnp.all(action_matches)
            & jnp.all(actions >= 0)
            & jnp.all(actions < cfg.n_actions)
            & jnp.all(jnp.isfinite(reward_values))
            & jnp.all(jnp.isfinite(next_values))
            & jnp.isfinite(discount_value)
            & (discount_value >= 0.0)
            & (discount_value <= 1.0)
            & jnp.all(update_applied)
            & jnp.all(prediction_matches_cache)
            & jnp.all(next_observations_match)
            & jnp.all(candidate_clock_aligned)
            & jnp.all(candidate_generation_aligned)
            & next_prepare.diagnostics.pair_committed
        )
        final_state, final_0, final_1 = cast(
            tuple[
                PrototypeFactorizedPartnerPlannerState,
                PrototypeAgentState,
                PrototypeAgentState,
            ],
            jax.lax.cond(
                candidate_valid,
                lambda _: (
                    next_prepare.state,
                    next_prepare.prototype_agent_0,
                    next_prepare.prototype_agent_1,
                ),
                lambda _: (state, prototype_agent_0, prototype_agent_1),
                operand=None,
            ),
        )
        diagnostics = PrototypeFactorizedPartnerTransitionDiagnostics(
            source_cache_valid=source_cache_valid,
            executed_actions=actions,
            observed_partner_actions=partner_actions,
            behavior_losses=jnp.stack(tuple(item.loss for item in behavior_updates)),
            behavior_update_applied=jnp.stack(
                tuple(item.update_applied for item in behavior_updates)
            ),
            grounded_losses=jnp.stack(tuple(item.loss for item in grounded_updates)),
            grounded_targets=jnp.stack(tuple(item.targets for item in grounded_updates)),
            grounded_joint_action_indices=jnp.stack(
                tuple(item.prediction.joint_action_index for item in grounded_updates)
            ),
            grounded_update_applied=jnp.stack(
                tuple(item.update_applied for item in grounded_updates)
            ),
            prediction_matches_cache=prediction_matches_cache,
            candidate_clock_aligned=candidate_clock_aligned,
            candidate_generation_aligned=candidate_generation_aligned,
            next_observations_match=next_observations_match,
            next_prepare=next_prepare.diagnostics,
            candidate_valid=candidate_valid,
            transaction_committed=candidate_valid,
        )
        return PrototypeFactorizedPartnerTransitionResult(
            state=final_state,
            prototype_agent_0=final_0,
            prototype_agent_1=final_1,
            diagnostics=diagnostics,
        )


__all__ = [
    "BASE_FALLBACK_SOURCE_BINDING_CLAIMED",
    "CHECKPOINT_RESUME_CLAIMED",
    "CONFIG_TOKEN_NBYTES",
    "DEVELOPMENT_ONLY",
    "FactorizedPartnerDecisionCache",
    "FactorizedPartnerPlannerAgentState",
    "N_AGENTS",
    "POST_MEMORY_TRANSITION_BINDING_CLAIMED",
    "PROTOTYPE_FACTORIZED_PARTNER_PLANNER_SCHEMA",
    "PrototypeFactorizedPartnerPlanner",
    "PrototypeFactorizedPartnerPlannerConfig",
    "PrototypeFactorizedPartnerPlannerResourceBudget",
    "PrototypeFactorizedPartnerPlannerWorkBudget",
    "PrototypeFactorizedPartnerPlannerState",
    "PrototypeFactorizedPartnerPrepareDiagnostics",
    "PrototypeFactorizedPartnerPrepareResult",
    "PrototypeFactorizedPartnerTransitionDiagnostics",
    "PrototypeFactorizedPartnerTransitionResult",
    "REPLAY_CAPACITY",
    "SCIENTIFIC_PROMOTION_ALLOWED",
    "THRESHOLD_CALIBRATION_CLAIMED",
]
