# mypy: disable-error-code="attr-defined,call-arg,no-any-return"
"""Fixed-output linear ensemble over one externally owned changing pair bank.

This L0 component owns model members, residual-proxy statistics, and one causal
learning-signal estimator.  It deliberately owns no feature lifecycle and no
router state.  A caller supplies the authoritative source bank plus one
source-to-destination bank receipt.  Every member first predicts and updates
against the old augmented representation.  One subsequent router evaluation
then migrates the stacked generated-input weight/trace columns for all members:
stable base and action columns are bit-preserved, descriptor survivors move by
identity, and newborn/inactive columns are positive float32 zero.  Any member,
signal, counter, route, or destination failure rolls the entire ensemble back.

Outputs remain the fixed physical coordinates ``[base delta, reward,
continuation]`` regardless of pair-bank contents.  Epistemic variance,
pre-update residual-variance proxies, and typed learning progress are emitted
causally for downstream calibration or learning-value routing.  The residual
variance is not a demonstrated aleatoric likelihood.  This module grants no
planning, dispatch, safety, evidence, or promotion authority and does not alter
the existing single-model or Prototype v18 APIs.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import Array

from alberta_framework.core.checkpoints import (
    load_checkpoint,
    load_checkpoint_metadata,
    save_checkpoint,
)
from alberta_framework.core.feature_bank_router import (
    FeatureBankRouteDiagnostics,
    FeatureBankRouter,
    FeatureBankRouterConfig,
    FeatureBankRouterState,
)
from alberta_framework.core.learning_signals import (
    LearningSignalEstimator,
    LearningSignalEstimatorConfig,
    LearningSignalEstimatorState,
    TypedLearningSignals,
)
from alberta_framework.core.multi_head_learner import (
    MultiHeadMLPLearner,
    MultiHeadMLPState,
)
from alberta_framework.core.normalizers import (
    _checked_lifetime_words_increment,
    _lifetime_counter_valid,
    _saturating_int32_counter_increment,
)
from alberta_framework.core.prototype_feature_lifecycle import (
    PrototypeFeatureConsumerBinding,
)
from alberta_framework.core.types import MLPParams
from alberta_framework.core.world_model import (
    ActionConditionedWorldModel,
    ActionConditionedWorldModelConfig,
)

ROUTED_LINEAR_WORLD_MODEL_ENSEMBLE_CONFIG_SCHEMA = (
    "alberta.routed-linear-world-model-ensemble.config.v1"
)
ROUTED_LINEAR_WORLD_MODEL_ENSEMBLE_STATE_SCHEMA = (
    "alberta.routed-linear-world-model-ensemble.state.v1"
)
ROUTED_LINEAR_WORLD_MODEL_ENSEMBLE_CHECKPOINT_SCHEMA = (
    "alberta.routed-linear-world-model-ensemble.checkpoint.v1"
)
ROUTED_LINEAR_WORLD_MODEL_ENSEMBLE_EVIDENCE_LEVEL = "L0"
ROUTED_LINEAR_WORLD_MODEL_ENSEMBLE_OUTCOME_STATUS = "not_assessed"
ROUTED_LINEAR_WORLD_MODEL_ENSEMBLE_SCIENTIFIC_PROMOTION_ALLOWED = False

_INT32_MAX = 2_147_483_647
_SCHEMA_DIGEST_NBYTES = 32
_FIXED_OUTPUT_SEMANTICS = "[normalized-delta-stable-base,reward,continuation]"
_ROUTE_SEMANTICS = (
    "all-members-source-update-first;one-stacked-authoritative-route;"
    "stable-and-action-exact;survivor-by-descriptor-exact;"
    "newborn-and-inactive-positive-zero"
)


def _require_array(
    value: object,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: Any,
) -> Array:
    if not hasattr(value, "shape") or not hasattr(value, "dtype"):
        raise TypeError(f"{name} must be an array with exact shape and dtype")
    array = cast(Array, value)
    if tuple(array.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}; got {tuple(array.shape)}")
    if jnp.dtype(array.dtype) != jnp.dtype(dtype):
        raise TypeError(f"{name} must have dtype {jnp.dtype(dtype)}; got {array.dtype}")
    return array


def _tree_static_contract_matches(value: object, template: object) -> bool:
    value_leaves, value_tree = jax.tree.flatten(value)
    template_leaves, template_tree = jax.tree.flatten(template)
    if cast(Any, value_tree) != template_tree or len(value_leaves) != len(
        template_leaves
    ):
        return False
    for value_leaf, template_leaf in zip(
        value_leaves,
        template_leaves,
        strict=True,
    ):
        value_array = jnp.asarray(value_leaf)
        template_array = jnp.asarray(template_leaf)
        if (
            value_array.shape != template_array.shape
            or value_array.dtype != template_array.dtype
        ):
            return False
    return True


def _tree_equal(left: object, right: object) -> Array:
    left_leaves, left_tree = jax.tree.flatten(left)
    right_leaves, right_tree = jax.tree.flatten(right)
    if cast(Any, left_tree) != right_tree or len(left_leaves) != len(right_leaves):
        return jnp.asarray(False, dtype=jnp.bool_)
    equal = jnp.asarray(True, dtype=jnp.bool_)
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        left_array = jnp.asarray(left_leaf)
        right_array = jnp.asarray(right_leaf)
        if jax.dtypes.issubdtype(left_array.dtype, jax.dtypes.prng_key):
            equal = equal & jnp.array_equal(
                jr.key_data(left_array),
                jr.key_data(right_array),
            )
        elif left_array.dtype == jnp.dtype(jnp.float32):
            equal = equal & jnp.array_equal(
                jax.lax.bitcast_convert_type(left_array, jnp.uint32),
                jax.lax.bitcast_convert_type(right_array, jnp.uint32),
            )
        else:
            equal = equal & jnp.array_equal(left_array, right_array)
    return equal


def _tree_finite(tree: object) -> Array:
    valid = jnp.asarray(True, dtype=jnp.bool_)
    for leaf in jax.tree.leaves(tree):
        value = jnp.asarray(leaf)
        if jnp.issubdtype(value.dtype, jnp.inexact):
            valid = valid & jnp.all(jnp.isfinite(value))
    return valid


def _tree_size(tree: object) -> tuple[int, int]:
    scalars = 0
    nbytes = 0
    for leaf in jax.tree.leaves(tree):
        value = jnp.asarray(leaf)
        if jax.dtypes.issubdtype(value.dtype, jax.dtypes.prng_key):
            value = jr.key_data(value)
        scalars += int(value.size)
        nbytes += int(value.nbytes)
    return scalars, nbytes


def _words_less_equal(left: Array, right: Array) -> Array:
    return (left[0] < right[0]) | ((left[0] == right[0]) & (left[1] <= right[1]))


def _words_successor(source: Array, destination: Array) -> Array:
    proposed, capacity = _checked_lifetime_words_increment(source)
    return capacity & jnp.array_equal(proposed, destination)


def _checked_words_add(left: Array, right: Array) -> tuple[Array, Array]:
    low = left[1] + right[1]
    carry = (low < left[1]).astype(jnp.uint32)
    high_without_carry = left[0] + right[0]
    high = high_without_carry + carry
    overflow = (high_without_carry < left[0]) | (high < high_without_carry)
    return jnp.stack((high, low)).astype(jnp.uint32), ~overflow


def _words_leq_limit(words: Array, limit: int) -> Array:
    limit_words = jnp.asarray(
        ((limit >> 32) & 0xFFFFFFFF, limit & 0xFFFFFFFF),
        dtype=jnp.uint32,
    )
    return _words_less_equal(words, limit_words)


def _config_digest(config: Mapping[str, object]) -> str:
    encoded = json.dumps(
        dict(config),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclasses.dataclass(frozen=True, slots=True)
class RoutedLinearWorldModelEnsembleConfig:
    """Router geometry, fixed physical heads, ensemble, and signal contract."""

    router: FeatureBankRouterConfig
    world_model: ActionConditionedWorldModelConfig
    signal_estimator: LearningSignalEstimatorConfig
    ensemble_size: int = 3
    residual_variance_decay: float = 0.99
    residual_variance_warmup_steps: int = 1
    residual_variance_floor: float = 1.0e-6
    max_events: int = _INT32_MAX
    carry_survivors: bool = True

    def __post_init__(self) -> None:
        if type(self.router) is not FeatureBankRouterConfig:
            raise TypeError("router must be an exact FeatureBankRouterConfig")
        if type(self.world_model) is not ActionConditionedWorldModelConfig:
            raise TypeError("world_model must be an exact ActionConditionedWorldModelConfig")
        if type(self.signal_estimator) is not LearningSignalEstimatorConfig:
            raise TypeError(
                "signal_estimator must be an exact LearningSignalEstimatorConfig"
            )
        ActionConditionedWorldModel(self.world_model)
        if self.world_model.observation_dim != self.router.base_dim:
            raise ValueError("world_model observation_dim must equal router base_dim")
        if self.world_model.hidden_sizes != ():
            raise ValueError("routed ensemble members must be exact linear models")
        if self.world_model.include_action_interactions:
            raise ValueError("action interactions are unsupported in routed ensemble v1")
        if not self.world_model.predict_delta:
            raise ValueError("routed ensemble physical heads must predict stable-base deltas")
        maximum_pairs = self.router.base_dim * (self.router.base_dim - 1) // 2
        if self.router.active_slots > maximum_pairs:
            raise ValueError("router active_slots exceeds unique base pair capacity")
        if type(self.ensemble_size) is not int or self.ensemble_size < 1:
            raise ValueError("ensemble_size must be an exact integer >= 1")
        if self.signal_estimator.ensemble_size != self.ensemble_size:
            raise ValueError("signal_estimator ensemble_size must match")
        if self.signal_estimator.target_dim != self.target_dim:
            raise ValueError("signal_estimator target_dim must equal base_dim + 2")
        if (
            not math.isfinite(self.residual_variance_decay)
            or not 0.0 <= self.residual_variance_decay < 1.0
        ):
            raise ValueError("residual_variance_decay must be finite and in [0, 1)")
        if (
            type(self.residual_variance_warmup_steps) is not int
            or not 1 <= self.residual_variance_warmup_steps <= _INT32_MAX
        ):
            raise ValueError("residual_variance_warmup_steps must fit positive int32")
        if (
            not math.isfinite(self.residual_variance_floor)
            or self.residual_variance_floor <= 0.0
        ):
            raise ValueError("residual_variance_floor must be positive and finite")
        if self.residual_variance_floor < self.signal_estimator.variance_floor:
            raise ValueError("residual_variance_floor is below signal variance_floor")
        if (
            self.residual_variance_floor
            > self.signal_estimator.max_predicted_variance
        ):
            raise ValueError("residual_variance_floor exceeds signal variance bound")
        if type(self.max_events) is not int or not 1 <= self.max_events <= _INT32_MAX:
            raise ValueError("max_events must be an exact integer in [1, int32 max]")
        if type(self.carry_survivors) is not bool:
            raise ValueError("carry_survivors must be an exact bool")

    @property
    def target_dim(self) -> int:
        return self.router.base_dim + 2

    def to_config(self) -> dict[str, object]:
        return {
            "type": type(self).__name__,
            "schema": ROUTED_LINEAR_WORLD_MODEL_ENSEMBLE_CONFIG_SCHEMA,
            "state_schema": ROUTED_LINEAR_WORLD_MODEL_ENSEMBLE_STATE_SCHEMA,
            "evidence_level": ROUTED_LINEAR_WORLD_MODEL_ENSEMBLE_EVIDENCE_LEVEL,
            "outcome_status": ROUTED_LINEAR_WORLD_MODEL_ENSEMBLE_OUTCOME_STATUS,
            "scientific_promotion_allowed": False,
            "router": self.router.to_config(),
            "world_model": self.world_model.to_config(),
            "signal_estimator": self.signal_estimator.to_config(),
            "ensemble_size": self.ensemble_size,
            "residual_variance_decay": self.residual_variance_decay,
            "residual_variance_warmup_steps": self.residual_variance_warmup_steps,
            "residual_variance_floor": self.residual_variance_floor,
            "max_events": self.max_events,
            "carry_survivors": self.carry_survivors,
            "fixed_output_semantics": _FIXED_OUTPUT_SEMANTICS,
            "route_semantics": _ROUTE_SEMANTICS,
            "feature_lifecycle_state_owned": False,
            "router_state_owned": False,
            "planning_authority": False,
            "safety_authority": False,
        }

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, object],
    ) -> RoutedLinearWorldModelEnsembleConfig:
        expected = {
            "type",
            "schema",
            "state_schema",
            "evidence_level",
            "outcome_status",
            "scientific_promotion_allowed",
            "router",
            "world_model",
            "signal_estimator",
            "ensemble_size",
            "residual_variance_decay",
            "residual_variance_warmup_steps",
            "residual_variance_floor",
            "max_events",
            "carry_survivors",
            "fixed_output_semantics",
            "route_semantics",
            "feature_lifecycle_state_owned",
            "router_state_owned",
            "planning_authority",
            "safety_authority",
        }
        if type(config) is not dict or set(config) != expected:
            raise ValueError("routed ensemble config fields are not exact")
        fixed = {
            "type": cls.__name__,
            "schema": ROUTED_LINEAR_WORLD_MODEL_ENSEMBLE_CONFIG_SCHEMA,
            "state_schema": ROUTED_LINEAR_WORLD_MODEL_ENSEMBLE_STATE_SCHEMA,
            "evidence_level": ROUTED_LINEAR_WORLD_MODEL_ENSEMBLE_EVIDENCE_LEVEL,
            "outcome_status": ROUTED_LINEAR_WORLD_MODEL_ENSEMBLE_OUTCOME_STATUS,
            "scientific_promotion_allowed": False,
            "fixed_output_semantics": _FIXED_OUTPUT_SEMANTICS,
            "route_semantics": _ROUTE_SEMANTICS,
            "feature_lifecycle_state_owned": False,
            "router_state_owned": False,
            "planning_authority": False,
            "safety_authority": False,
        }
        if any(config.get(name) != value for name, value in fixed.items()):
            raise ValueError("routed ensemble fixed semantics differ")
        for name in ("router", "world_model", "signal_estimator"):
            if type(config[name]) is not dict:
                raise ValueError(f"routed ensemble {name} config must be an exact dict")
        restored = cls(
            router=FeatureBankRouterConfig.from_config(
                cast(dict[str, object], config["router"])
            ),
            world_model=ActionConditionedWorldModelConfig.from_config(
                cast(dict[str, Any], config["world_model"])
            ),
            signal_estimator=LearningSignalEstimatorConfig.from_config(
                cast(dict[str, Any], config["signal_estimator"])
            ),
            ensemble_size=cast(int, config["ensemble_size"]),
            residual_variance_decay=cast(float, config["residual_variance_decay"]),
            residual_variance_warmup_steps=cast(
                int, config["residual_variance_warmup_steps"]
            ),
            residual_variance_floor=cast(float, config["residual_variance_floor"]),
            max_events=cast(int, config["max_events"]),
            carry_survivors=cast(bool, config["carry_survivors"]),
        )
        if restored.to_config() != dict(config):
            raise ValueError("routed ensemble config is not canonical")
        return restored


@chex.dataclass(frozen=True)
class RoutedLinearWorldModelMemberState:
    """One independent linear learner; physical bounds remain shared."""

    learner_state: MultiHeadMLPState


@chex.dataclass(frozen=True)
class RoutedLinearWorldModelEnsembleState:
    """Members, causal residual/signals, exact bank binding, and clocks."""

    member_states: tuple[RoutedLinearWorldModelMemberState, ...]
    observation_min: Array
    observation_max: Array
    reward_min: Array
    reward_max: Array
    residual_variances: Array
    signal_state: LearningSignalEstimatorState
    consumer_binding: PrototypeFeatureConsumerBinding
    event_count: Array
    event_count_words: Array
    generation_update_count: Array
    generation_update_words: Array
    generation_birth_event_words: Array
    schema_digest: Array


@chex.dataclass(frozen=True)
class RoutedLinearWorldModelEnsemblePrediction:
    """Causal pre-update physical predictions and uncertainty inputs."""

    member_raw_predictions: Array
    mean_raw_prediction: Array
    member_next_base_observations: Array
    mean_next_base_observation: Array
    member_rewards: Array
    mean_reward: Array
    member_discounts: Array
    mean_discount: Array
    per_head_epistemic_variance: Array
    epistemic_disagreement: Array
    residual_variances: Array
    residual_proxy_ready: Array
    valid: Array


@chex.dataclass(frozen=True)
class RoutedLinearWorldModelEnsemblePreparedTransition:
    """Exact source state/bank/input/action/prediction pre-outcome cache."""

    source_state: RoutedLinearWorldModelEnsembleState
    source_router_state: FeatureBankRouterState
    base_observation: Array
    augmented_observation: Array
    input_features: Array
    primitive_action: Array
    prediction: RoutedLinearWorldModelEnsemblePrediction
    prepared: Array


@chex.dataclass(frozen=True)
class RoutedLinearWorldModelEnsemblePrepareDiagnostics:
    state_valid: Array
    source_router_valid: Array
    source_router_matches_binding: Array
    base_observation_valid: Array
    primitive_action_valid: Array
    prediction_valid: Array
    prepared: Array


@chex.dataclass(frozen=True)
class RoutedLinearWorldModelEnsemblePrepareResult:
    prepared: RoutedLinearWorldModelEnsemblePreparedTransition
    diagnostics: RoutedLinearWorldModelEnsemblePrepareDiagnostics


@chex.dataclass(frozen=True)
class RoutedLinearWorldModelEnsembleTransition:
    """Prepared old-bank prediction, real physical outcome, and new-bank receipt."""

    prepared: RoutedLinearWorldModelEnsemblePreparedTransition
    reward: Array
    discount: Array
    next_base_observation: Array
    destination_router_state: FeatureBankRouterState
    destination_binding: PrototypeFeatureConsumerBinding


@chex.dataclass(frozen=True)
class RoutedLinearWorldModelEnsembleDiagnostics:
    state_valid: Array
    source_router_valid: Array
    source_router_matches_binding: Array
    prepared_source_state_matches: Array
    prepared_source_router_matches: Array
    source_cache_matches: Array
    source_prediction_matches: Array
    destination_binding_valid: Array
    destination_router_valid: Array
    destination_router_matches_binding: Array
    descriptors_changed: Array
    generation_changed: Array
    generation_is_successor: Array
    bank_transition_consistent: Array
    event_values_valid: Array
    event_capacity_available: Array
    all_member_updates_applied: Array
    signal_update_valid: Array
    residual_update_valid: Array
    one_authoritative_route_evaluated: Array
    route_valid: Array
    route_state_matches_destination: Array
    candidate_state_valid: Array
    transaction_applied: Array
    rejected: Array
    router_evaluations: Array
    member_update_evaluations: Array
    physical_output_head_count: Array
    generated_output_head_count: Array
    planning_authority: Array
    safety_authority: Array
    pre_event_words: Array
    post_event_words: Array


@chex.dataclass(frozen=True)
class RoutedLinearWorldModelEnsembleResult:
    state: RoutedLinearWorldModelEnsembleState
    prediction: RoutedLinearWorldModelEnsemblePrediction
    signals: TypedLearningSignals
    targets: Array
    observed_loss: Array
    member_prediction_losses: Array
    member_updates_applied: Array
    route_diagnostics: FeatureBankRouteDiagnostics
    diagnostics: RoutedLinearWorldModelEnsembleDiagnostics


@dataclasses.dataclass(frozen=True, slots=True)
class RoutedLinearWorldModelEnsembleResourceBudget:
    ensemble_size: int
    base_feature_dim: int
    active_pair_slots: int
    model_input_dim: int
    physical_output_heads: int
    generated_output_heads: int
    persistent_state_scalars: int
    persistent_state_bytes: int
    prediction_scalars: int
    prediction_bytes: int
    max_member_updates_per_event: int
    max_router_evaluations_per_event: int
    max_events: int
    feature_lifecycle_state_owned: int
    router_state_owned: int
    planning_authority: int
    safety_authority: int
    scientific_promotion_allowed: bool

    def to_config(self) -> dict[str, object]:
        return dataclasses.asdict(self)


class RoutedLinearWorldModelEnsemble:
    """All-member old-bank update followed by one atomic stacked route."""

    def __init__(self, config: RoutedLinearWorldModelEnsembleConfig) -> None:
        if type(config) is not RoutedLinearWorldModelEnsembleConfig:
            raise TypeError(
                "config must be an exact RoutedLinearWorldModelEnsembleConfig"
            )
        self._config = config
        self._router = FeatureBankRouter(config.router)
        self._signals = LearningSignalEstimator(config.signal_estimator)
        self._base_dim = config.router.base_dim
        self._active_slots = config.router.active_slots
        self._total_dim = config.router.total_feature_dim
        self._n_actions = config.world_model.n_actions
        self._target_dim = config.target_dim
        self._input_dim = self._total_dim + self._n_actions
        world = config.world_model
        self._learner = MultiHeadMLPLearner(
            n_heads=self._target_dim,
            hidden_sizes=(),
            step_size=world.step_size,
            gamma=0.0,
            lamda=0.0,
            sparsity=world.sparsity,
            leaky_relu_slope=world.leaky_relu_slope,
            use_layer_norm=world.use_layer_norm,
            trace_mode=world.trace_mode,
            utility_decay=world.utility_decay,
        )
        learner_template = self._learner.init(self._input_dim, jr.key(0))
        self._learner_template = cast(
            MultiHeadMLPState,
            learner_template.replace(
                birth_timestamp=jnp.asarray(0.0, dtype=jnp.float32),
                uptime_s=jnp.asarray(0.0, dtype=jnp.float32),
            ),
        )
        self._signal_template = self._signals.init()
        digest = hashlib.sha256(
            json.dumps(
                config.to_config(),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).digest()
        self._schema_digest = jnp.asarray(tuple(digest), dtype=jnp.uint8)

    @property
    def config(self) -> RoutedLinearWorldModelEnsembleConfig:
        return self._config

    @property
    def router(self) -> FeatureBankRouter:
        """Expose routing geometry; no router state is owned here."""

        return self._router

    @property
    def learner(self) -> MultiHeadMLPLearner:
        """Expose the shared exact linear member implementation."""

        return self._learner

    @property
    def signal_estimator(self) -> LearningSignalEstimator:
        return self._signals

    def to_config(self) -> dict[str, object]:
        return self._config.to_config()

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, object],
    ) -> RoutedLinearWorldModelEnsemble:
        return cls(RoutedLinearWorldModelEnsembleConfig.from_config(config))

    def _validate_binding_static(
        self,
        binding: object,
        *,
        name: str,
    ) -> None:
        if type(binding) is not PrototypeFeatureConsumerBinding:
            raise TypeError(f"{name} must be an exact PrototypeFeatureConsumerBinding")
        exact = binding
        _require_array(
            exact.semantic_generation,
            name=f"{name}.semantic_generation",
            shape=(),
            dtype=jnp.int32,
        )
        _require_array(
            exact.semantic_generation_words,
            name=f"{name}.semantic_generation_words",
            shape=(2,),
            dtype=jnp.uint32,
        )
        _require_array(
            exact.descriptors,
            name=f"{name}.descriptors",
            shape=(self._active_slots, 2),
            dtype=jnp.int32,
        )

    def _binding_valid(self, binding: PrototypeFeatureConsumerBinding) -> Array:
        validation = self._router.validate_descriptors(binding.descriptors)
        return (
            (binding.semantic_generation >= 0)
            & _lifetime_counter_valid(
                binding.semantic_generation_words,
                binding.semantic_generation,
            )
            & validation.valid
            & jnp.all(validation.live_mask)
        )

    def _validate_router_static(self, state: object, *, name: str) -> None:
        if type(state) is not FeatureBankRouterState:
            raise TypeError(f"{name} must be an exact FeatureBankRouterState")
        exact = state
        _require_array(
            exact.descriptors,
            name=f"{name}.descriptors",
            shape=(self._active_slots, 2),
            dtype=jnp.int32,
        )
        for field_name in ("route_count", "generation_count"):
            _require_array(
                getattr(exact, field_name),
                name=f"{name}.{field_name}",
                shape=(),
                dtype=jnp.int32,
            )
        for field_name in ("route_words", "generation_words"):
            _require_array(
                getattr(exact, field_name),
                name=f"{name}.{field_name}",
                shape=(2,),
                dtype=jnp.uint32,
            )

    def _router_valid(self, state: FeatureBankRouterState) -> Array:
        validation = self._router.validate_descriptors(state.descriptors)
        return (
            validation.valid
            & jnp.all(validation.live_mask)
            & _lifetime_counter_valid(state.route_words, state.route_count)
            & _lifetime_counter_valid(
                state.generation_words,
                state.generation_count,
            )
            & _words_less_equal(state.generation_words, state.route_words)
        )

    def _router_matches_binding(
        self,
        router_state: FeatureBankRouterState,
        binding: PrototypeFeatureConsumerBinding,
    ) -> Array:
        return (
            (router_state.generation_count == binding.semantic_generation)
            & jnp.array_equal(
                router_state.generation_words,
                binding.semantic_generation_words,
            )
            & jnp.array_equal(router_state.descriptors, binding.descriptors)
        )

    def router_state_matches_binding(
        self,
        router_state: FeatureBankRouterState,
        binding: PrototypeFeatureConsumerBinding,
    ) -> Array:
        """Read-only exact source-bank identity check for external owners."""

        self._validate_router_static(router_state, name="router_state")
        self._validate_binding_static(binding, name="binding")
        return (
            self._router_valid(router_state)
            & self._binding_valid(binding)
            & self._router_matches_binding(router_state, binding)
        )

    def _validate_state_static(
        self,
        state: object,
    ) -> None:
        if type(state) is not RoutedLinearWorldModelEnsembleState:
            raise TypeError(
                "state must be an exact RoutedLinearWorldModelEnsembleState"
            )
        exact = state
        if type(exact.member_states) is not tuple or len(exact.member_states) != (
            self._config.ensemble_size
        ):
            raise ValueError("state must contain the configured exact member tuple")
        for index, member in enumerate(exact.member_states):
            if type(member) is not RoutedLinearWorldModelMemberState:
                raise TypeError(f"state member {index} has the wrong exact type")
            if not _tree_static_contract_matches(
                member.learner_state,
                self._learner_template,
            ):
                raise ValueError(f"state member {index} learner contract differs")
        checks = (
            (exact.observation_min, (self._base_dim,), jnp.float32),
            (exact.observation_max, (self._base_dim,), jnp.float32),
            (exact.reward_min, (), jnp.float32),
            (exact.reward_max, (), jnp.float32),
            (
                exact.residual_variances,
                (self._config.ensemble_size, self._target_dim),
                jnp.float32,
            ),
            (exact.event_count, (), jnp.int32),
            (exact.event_count_words, (2,), jnp.uint32),
            (exact.generation_update_count, (), jnp.int32),
            (exact.generation_update_words, (2,), jnp.uint32),
            (exact.generation_birth_event_words, (2,), jnp.uint32),
            (exact.schema_digest, (_SCHEMA_DIGEST_NBYTES,), jnp.uint8),
        )
        for value, shape, dtype in checks:
            _require_array(
                value,
                name="state field",
                shape=shape,
                dtype=dtype,
            )
        if not _tree_static_contract_matches(
            exact.signal_state,
            self._signal_template,
        ):
            raise ValueError("state signal-estimator contract differs")
        self._validate_binding_static(exact.consumer_binding, name="state.consumer_binding")

    def _signal_state_valid(
        self,
        state: LearningSignalEstimatorState,
        event_count: Array,
        event_words: Array,
    ) -> Array:
        status = self._signals.counter_status(state)
        config = self._config.signal_estimator
        return (
            status.lifetime_counter_valid
            & (state.step_count == event_count)
            & (state.valid_count == event_count)
            & (state.invalid_count == 0)
            & jnp.array_equal(state.step_words, event_words)
            & jnp.array_equal(state.valid_words, event_words)
            & jnp.array_equal(
                state.invalid_words,
                jnp.zeros((2,), dtype=jnp.uint32),
            )
            & (state.calibration_count >= 0)
            & (state.calibration_count <= config.change_calibration_steps)
            & (state.calibration_count <= state.valid_count)
            & jnp.isfinite(state.calibration_mean)
            & (state.calibration_mean >= 0.0)
            & jnp.isfinite(state.calibration_m2)
            & (state.calibration_m2 >= 0.0)
            & jnp.isfinite(state.fast_loss_ema)
            & (state.fast_loss_ema >= 0.0)
            & jnp.isfinite(state.slow_loss_ema)
            & (state.slow_loss_ema >= 0.0)
            & jnp.isfinite(state.sustained_change_probability)
            & (state.sustained_change_probability >= 0.0)
            & (state.sustained_change_probability <= 1.0)
        )

    def _state_valid(self, state: RoutedLinearWorldModelEnsembleState) -> Array:
        pristine_bounds = (
            jnp.all(jnp.isposinf(state.observation_min))
            & jnp.all(jnp.isneginf(state.observation_max))
            & jnp.isposinf(state.reward_min)
            & jnp.isneginf(state.reward_max)
        )
        finite_bounds = (
            jnp.all(jnp.isfinite(state.observation_min))
            & jnp.all(jnp.isfinite(state.observation_max))
            & jnp.all(state.observation_min <= state.observation_max)
            & jnp.isfinite(state.reward_min)
            & jnp.isfinite(state.reward_max)
            & (state.reward_min <= state.reward_max)
        )
        bounds_valid = jnp.where(state.event_count == 0, pristine_bounds, finite_bounds)
        member_valid: list[Array] = []
        for member in state.member_states:
            learner_status = self._learner._counter_status(member.learner_state)
            member_valid.append(
                _tree_finite(member.learner_state)
                & learner_status.lifetime_counter_valid
                & learner_status.normalizer_counter_aligned
                & (member.learner_state.normalizer_state is None)
                & (member.learner_state.step_count == state.event_count)
                & jnp.array_equal(
                    member.learner_state.step_words,
                    state.event_count_words,
                )
            )
        generation_sum, generation_sum_valid = _checked_words_add(
            state.generation_birth_event_words,
            state.generation_update_words,
        )
        return (
            _lifetime_counter_valid(state.event_count_words, state.event_count)
            & _words_leq_limit(state.event_count_words, self._config.max_events)
            & jnp.all(jnp.stack(member_valid))
            & bounds_valid
            & jnp.all(jnp.isfinite(state.residual_variances))
            & jnp.all(
                state.residual_variances >= self._config.residual_variance_floor
            )
            & jnp.all(
                state.residual_variances
                <= self._config.signal_estimator.max_predicted_variance
            )
            & self._signal_state_valid(
                state.signal_state,
                state.event_count,
                state.event_count_words,
            )
            & self._binding_valid(state.consumer_binding)
            & _lifetime_counter_valid(
                state.generation_update_words,
                state.generation_update_count,
            )
            & generation_sum_valid
            & jnp.array_equal(generation_sum, state.event_count_words)
            & _words_less_equal(
                state.generation_birth_event_words,
                state.event_count_words,
            )
            & jnp.array_equal(state.schema_digest, self._schema_digest)
        )

    def state_valid(self, state: RoutedLinearWorldModelEnsembleState) -> Array:
        self._validate_state_static(state)
        return self._state_valid(state)

    def _canonical_descriptors(self) -> Array:
        pairs: list[tuple[int, int]] = []
        for left in range(self._base_dim):
            for right in range(left + 1, self._base_dim):
                pairs.append((left, right))
        return jnp.asarray(pairs[: self._active_slots], dtype=jnp.int32)

    def _template_state(self) -> RoutedLinearWorldModelEnsembleState:
        router = self._router.init(self._canonical_descriptors())
        binding = PrototypeFeatureConsumerBinding(
            semantic_generation=router.generation_count,
            semantic_generation_words=router.generation_words,
            descriptors=router.descriptors,
        )
        return self.init(jr.key(0), binding, router)

    def init(
        self,
        key: Array,
        binding: PrototypeFeatureConsumerBinding,
        router_state: FeatureBankRouterState,
    ) -> RoutedLinearWorldModelEnsembleState:
        """Initialize distinct members against one externally owned bank."""

        if not (
            hasattr(key, "shape")
            and hasattr(key, "dtype")
            and key.shape == ()
            and jax.dtypes.issubdtype(key.dtype, jax.dtypes.prng_key)
        ):
            raise TypeError("key must be a scalar typed JAX PRNG key")
        self._validate_binding_static(binding, name="binding")
        self._validate_router_static(router_state, name="router_state")
        keys = jr.split(key, self._config.ensemble_size)
        members: list[RoutedLinearWorldModelMemberState] = []
        for member_key in keys:
            learner = self._learner.init(self._input_dim, member_key)
            learner = cast(
                MultiHeadMLPState,
                learner.replace(
                    birth_timestamp=jnp.asarray(0.0, dtype=jnp.float32),
                    uptime_s=jnp.asarray(0.0, dtype=jnp.float32),
                ),
            )
            members.append(RoutedLinearWorldModelMemberState(learner_state=learner))
        zero = jnp.asarray(0, dtype=jnp.int32)
        zero_words = jnp.zeros((2,), dtype=jnp.uint32)
        state = RoutedLinearWorldModelEnsembleState(
            member_states=tuple(members),
            observation_min=jnp.full((self._base_dim,), jnp.inf, dtype=jnp.float32),
            observation_max=jnp.full((self._base_dim,), -jnp.inf, dtype=jnp.float32),
            reward_min=jnp.asarray(jnp.inf, dtype=jnp.float32),
            reward_max=jnp.asarray(-jnp.inf, dtype=jnp.float32),
            residual_variances=jnp.full(
                (self._config.ensemble_size, self._target_dim),
                self._config.residual_variance_floor,
                dtype=jnp.float32,
            ),
            signal_state=self._signals.init(),
            consumer_binding=binding,
            event_count=zero,
            event_count_words=zero_words,
            generation_update_count=zero,
            generation_update_words=zero_words,
            generation_birth_event_words=zero_words,
            schema_digest=self._schema_digest,
        )
        self._validate_state_static(state)
        valid = (
            self._router_valid(router_state)
            & self._router_matches_binding(router_state, binding)
            & self._state_valid(state)
        )
        if not bool(jax.device_get(valid)):
            raise ValueError("initial routed ensemble composition is invalid")
        return state

    def _augment(
        self,
        binding: PrototypeFeatureConsumerBinding,
        base_observation: Array,
    ) -> Array:
        left = jnp.clip(binding.descriptors[:, 0], 0, self._base_dim - 1)
        right = jnp.clip(binding.descriptors[:, 1], 0, self._base_dim - 1)
        products = base_observation[left] * base_observation[right]
        return jnp.concatenate((base_observation, products), axis=0)

    def _inputs(self, augmented: Array, action: Array) -> Array:
        one_hot = jax.nn.one_hot(action, self._n_actions, dtype=jnp.float32)
        return jnp.concatenate((augmented, one_hot), axis=0)

    def _targets(
        self,
        base_observation: Array,
        reward: Array,
        discount: Array,
        next_base_observation: Array,
    ) -> Array:
        world = self._config.world_model
        scale = jnp.asarray(
            (1.0,) * self._base_dim
            if world.observation_scale is None
            else world.observation_scale,
            dtype=jnp.float32,
        )
        delta = (next_base_observation - base_observation) / jnp.maximum(
            scale,
            jnp.asarray(1.0e-6, dtype=jnp.float32),
        )
        return jnp.concatenate(
            (
                delta,
                jnp.reshape(reward / world.reward_scale, (1,)),
                jnp.reshape(discount, (1,)),
            )
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def targets(
        self,
        base_observation: Array,
        reward: Array,
        discount: Array,
        next_base_observation: Array,
    ) -> Array:
        base = _require_array(
            base_observation,
            name="base_observation",
            shape=(self._base_dim,),
            dtype=jnp.float32,
        )
        next_base = _require_array(
            next_base_observation,
            name="next_base_observation",
            shape=(self._base_dim,),
            dtype=jnp.float32,
        )
        exact_reward = _require_array(
            reward,
            name="reward",
            shape=(),
            dtype=jnp.float32,
        )
        exact_discount = _require_array(
            discount,
            name="discount",
            shape=(),
            dtype=jnp.float32,
        )
        return self._targets(base, exact_reward, exact_discount, next_base)

    def _decode_raw(
        self,
        state: RoutedLinearWorldModelEnsembleState,
        base_observation: Array,
        raw: Array,
    ) -> tuple[Array, Array, Array]:
        world = self._config.world_model
        scale = jnp.asarray(
            (1.0,) * self._base_dim
            if world.observation_scale is None
            else world.observation_scale,
            dtype=jnp.float32,
        )
        delta = jnp.clip(
            raw[:, : self._base_dim],
            -world.max_delta_scale,
            world.max_delta_scale,
        )
        next_base = base_observation[None, :] + delta * scale[None, :]
        low = state.observation_min - world.observation_clip_margin
        high = state.observation_max + world.observation_clip_margin
        next_base = jnp.where(
            state.event_count > 0,
            jnp.clip(next_base, low[None, :], high[None, :]),
            next_base,
        )
        rewards = raw[:, self._base_dim] * world.reward_scale
        rewards = jnp.where(
            state.event_count > 0,
            jnp.clip(
                rewards,
                state.reward_min - world.observation_clip_margin,
                state.reward_max + world.observation_clip_margin,
            ),
            rewards,
        )
        discounts = jnp.clip(raw[:, self._base_dim + 1], 0.0, world.gamma)
        return next_base, rewards, discounts

    def _predict_unchecked(
        self,
        state: RoutedLinearWorldModelEnsembleState,
        base_observation: Array,
        augmented: Array,
        action: Array,
    ) -> RoutedLinearWorldModelEnsemblePrediction:
        inputs = self._inputs(augmented, action)
        raw = jnp.stack(
            tuple(
                self._learner.predict(member.learner_state, inputs)
                for member in state.member_states
            )
        )
        next_base, rewards, discounts = self._decode_raw(
            state,
            base_observation,
            raw,
        )
        epistemic = jnp.var(raw, axis=0)
        finite = (
            jnp.all(jnp.isfinite(raw))
            & jnp.all(jnp.isfinite(next_base))
            & jnp.all(jnp.isfinite(rewards))
            & jnp.all(jnp.isfinite(discounts))
            & jnp.all(
                jnp.abs(raw)
                <= self._config.signal_estimator.max_input_magnitude
            )
        )
        return RoutedLinearWorldModelEnsemblePrediction(
            member_raw_predictions=raw,
            mean_raw_prediction=jnp.mean(raw, axis=0),
            member_next_base_observations=next_base,
            mean_next_base_observation=jnp.mean(next_base, axis=0),
            member_rewards=rewards,
            mean_reward=jnp.mean(rewards),
            member_discounts=discounts,
            mean_discount=jnp.mean(discounts),
            per_head_epistemic_variance=epistemic,
            epistemic_disagreement=jnp.mean(epistemic),
            residual_variances=state.residual_variances,
            residual_proxy_ready=(
                state.event_count
                >= self._config.residual_variance_warmup_steps
            ),
            valid=finite,
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def predict(
        self,
        state: RoutedLinearWorldModelEnsembleState,
        source_router_state: FeatureBankRouterState,
        base_observation: Array,
        primitive_action: Array,
    ) -> RoutedLinearWorldModelEnsemblePrediction:
        self._validate_state_static(state)
        self._validate_router_static(source_router_state, name="source_router_state")
        base = _require_array(
            base_observation,
            name="base_observation",
            shape=(self._base_dim,),
            dtype=jnp.float32,
        )
        action = _require_array(
            primitive_action,
            name="primitive_action",
            shape=(),
            dtype=jnp.int32,
        )
        prediction = self._predict_unchecked(
            state,
            base,
            self._augment(state.consumer_binding, base),
            action,
        )
        valid = (
            self._state_valid(state)
            & self._router_valid(source_router_state)
            & self._router_matches_binding(
                source_router_state,
                state.consumer_binding,
            )
            & jnp.all(jnp.isfinite(base))
            & (action >= 0)
            & (action < self._n_actions)
            & prediction.valid
        )
        return cast(
            RoutedLinearWorldModelEnsemblePrediction,
            prediction.replace(valid=valid),
        )

    def _validate_prediction_static(
        self,
        prediction: object,
        *,
        name: str,
    ) -> None:
        if type(prediction) is not RoutedLinearWorldModelEnsemblePrediction:
            raise TypeError(
                f"{name} must be an exact RoutedLinearWorldModelEnsemblePrediction"
            )
        exact = prediction
        ensemble = self._config.ensemble_size
        checks = (
            (exact.member_raw_predictions, (ensemble, self._target_dim), jnp.float32),
            (exact.mean_raw_prediction, (self._target_dim,), jnp.float32),
            (
                exact.member_next_base_observations,
                (ensemble, self._base_dim),
                jnp.float32,
            ),
            (exact.mean_next_base_observation, (self._base_dim,), jnp.float32),
            (exact.member_rewards, (ensemble,), jnp.float32),
            (exact.mean_reward, (), jnp.float32),
            (exact.member_discounts, (ensemble,), jnp.float32),
            (exact.mean_discount, (), jnp.float32),
            (exact.per_head_epistemic_variance, (self._target_dim,), jnp.float32),
            (exact.epistemic_disagreement, (), jnp.float32),
            (
                exact.residual_variances,
                (ensemble, self._target_dim),
                jnp.float32,
            ),
            (exact.residual_proxy_ready, (), jnp.bool_),
            (exact.valid, (), jnp.bool_),
        )
        for value, shape, dtype in checks:
            _require_array(value, name=name, shape=shape, dtype=dtype)

    def _validate_prepared_static(
        self,
        prepared: object,
    ) -> None:
        if type(prepared) is not RoutedLinearWorldModelEnsemblePreparedTransition:
            raise TypeError(
                "prepared must be an exact RoutedLinearWorldModelEnsemblePreparedTransition"
            )
        exact = prepared
        self._validate_state_static(exact.source_state)
        self._validate_router_static(
            exact.source_router_state,
            name="prepared.source_router_state",
        )
        checks = (
            (exact.base_observation, (self._base_dim,), jnp.float32),
            (exact.augmented_observation, (self._total_dim,), jnp.float32),
            (exact.input_features, (self._input_dim,), jnp.float32),
            (exact.primitive_action, (), jnp.int32),
            (exact.prepared, (), jnp.bool_),
        )
        for value, shape, dtype in checks:
            _require_array(value, name="prepared field", shape=shape, dtype=dtype)
        self._validate_prediction_static(exact.prediction, name="prepared.prediction")

    def prepare_transition(
        self,
        state: RoutedLinearWorldModelEnsembleState,
        source_router_state: FeatureBankRouterState,
        base_observation: Array,
        primitive_action: Array,
    ) -> RoutedLinearWorldModelEnsemblePrepareResult:
        """Cache exact old-bank predictions before the real outcome exists."""

        self._validate_state_static(state)
        self._validate_router_static(source_router_state, name="source_router_state")
        base = _require_array(
            base_observation,
            name="base_observation",
            shape=(self._base_dim,),
            dtype=jnp.float32,
        )
        action = _require_array(
            primitive_action,
            name="primitive_action",
            shape=(),
            dtype=jnp.int32,
        )
        return cast(
            RoutedLinearWorldModelEnsemblePrepareResult,
            self._prepare_transition_jit(state, source_router_state, base, action),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def _prepare_transition_jit(
        self,
        state: RoutedLinearWorldModelEnsembleState,
        source_router_state: FeatureBankRouterState,
        base_observation: Array,
        primitive_action: Array,
    ) -> RoutedLinearWorldModelEnsemblePrepareResult:
        state_valid = self._state_valid(state)
        router_valid = self._router_valid(source_router_state)
        router_matches = self._router_matches_binding(
            source_router_state,
            state.consumer_binding,
        )
        base_valid = jnp.all(jnp.isfinite(base_observation))
        action_valid = (
            (primitive_action >= 0) & (primitive_action < self._n_actions)
        )
        augmented = self._augment(state.consumer_binding, base_observation)
        inputs = self._inputs(augmented, primitive_action)
        prediction = self._predict_unchecked(
            state,
            base_observation,
            augmented,
            primitive_action,
        )
        prepared_valid = (
            state_valid
            & router_valid
            & router_matches
            & base_valid
            & action_valid
            & prediction.valid
        )
        prediction = prediction.replace(valid=prediction.valid & prepared_valid)
        prepared = RoutedLinearWorldModelEnsemblePreparedTransition(
            source_state=state,
            source_router_state=source_router_state,
            base_observation=base_observation,
            augmented_observation=augmented,
            input_features=inputs,
            primitive_action=primitive_action,
            prediction=prediction,
            prepared=prepared_valid,
        )
        return RoutedLinearWorldModelEnsemblePrepareResult(
            prepared=prepared,
            diagnostics=RoutedLinearWorldModelEnsemblePrepareDiagnostics(
                state_valid=state_valid,
                source_router_valid=router_valid,
                source_router_matches_binding=router_matches,
                base_observation_valid=base_valid,
                primitive_action_valid=action_valid,
                prediction_valid=prediction.valid,
                prepared=prepared_valid,
            ),
        )

    def _validate_transition_static(
        self,
        event: object,
    ) -> None:
        if type(event) is not RoutedLinearWorldModelEnsembleTransition:
            raise TypeError(
                "event must be an exact RoutedLinearWorldModelEnsembleTransition"
            )
        exact = event
        self._validate_prepared_static(exact.prepared)
        _require_array(
            exact.reward,
            name="event.reward",
            shape=(),
            dtype=jnp.float32,
        )
        _require_array(
            exact.discount,
            name="event.discount",
            shape=(),
            dtype=jnp.float32,
        )
        _require_array(
            exact.next_base_observation,
            name="event.next_base_observation",
            shape=(self._base_dim,),
            dtype=jnp.float32,
        )
        self._validate_router_static(
            exact.destination_router_state,
            name="event.destination_router_state",
        )
        self._validate_binding_static(
            exact.destination_binding,
            name="event.destination_binding",
        )

    def _route_member_inputs(
        self,
        source_router_state: FeatureBankRouterState,
        destination_descriptors: Array,
        members: tuple[RoutedLinearWorldModelMemberState, ...],
    ) -> tuple[
        tuple[RoutedLinearWorldModelMemberState, ...],
        FeatureBankRouterState,
        FeatureBankRouteDiagnostics,
    ]:
        all_weights = jnp.stack(
            tuple(
                jnp.concatenate(
                    member.learner_state.head_params.weights,
                    axis=0,
                )
                for member in members
            )
        )
        all_weight_traces = jnp.stack(
            tuple(
                jnp.concatenate(
                    tuple(trace[0] for trace in member.learner_state.head_traces),
                    axis=0,
                )
                for member in members
            )
        )
        observation_weights = all_weights[:, :, : self._total_dim]
        action_weights = all_weights[:, :, self._total_dim :]
        observation_traces = all_weight_traces[:, :, : self._total_dim]
        action_traces = all_weight_traces[:, :, self._total_dim :]
        route = self._router.route(
            source_router_state,
            (observation_weights, observation_traces),
            destination_descriptors,
            carry_survivors=self._config.carry_survivors,
        )
        routed_observation_weights, routed_observation_traces = route.consumers
        routed_members: list[RoutedLinearWorldModelMemberState] = []
        for member_index, member in enumerate(members):
            learner = member.learner_state
            routed_weights = jnp.concatenate(
                (
                    routed_observation_weights[member_index],
                    action_weights[member_index],
                ),
                axis=1,
            )
            routed_traces = jnp.concatenate(
                (
                    routed_observation_traces[member_index],
                    action_traces[member_index],
                ),
                axis=1,
            )
            head_weights = tuple(
                routed_weights[index : index + 1]
                for index in range(self._target_dim)
            )
            head_traces = tuple(
                (
                    routed_traces[index : index + 1],
                    learner.head_traces[index][1],
                )
                for index in range(self._target_dim)
            )
            routed_learner = cast(
                MultiHeadMLPState,
                learner.replace(
                    head_params=MLPParams(
                        weights=head_weights,
                        biases=learner.head_params.biases,
                    ),
                    head_traces=head_traces,
                ),
            )
            routed_members.append(
                RoutedLinearWorldModelMemberState(learner_state=routed_learner)
            )
        return tuple(routed_members), route.state, route.diagnostics

    def observe_and_route(
        self,
        state: RoutedLinearWorldModelEnsembleState,
        source_router_state: FeatureBankRouterState,
        event: RoutedLinearWorldModelEnsembleTransition,
    ) -> RoutedLinearWorldModelEnsembleResult:
        """Update every source-bank member, then route them in one transaction."""

        self._validate_state_static(state)
        self._validate_router_static(source_router_state, name="source_router_state")
        self._validate_transition_static(event)
        return cast(
            RoutedLinearWorldModelEnsembleResult,
            self._observe_and_route_jit(state, source_router_state, event),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def _observe_and_route_jit(
        self,
        state: RoutedLinearWorldModelEnsembleState,
        source_router_state: FeatureBankRouterState,
        event: RoutedLinearWorldModelEnsembleTransition,
    ) -> RoutedLinearWorldModelEnsembleResult:
        prepared = event.prepared
        state_valid = self._state_valid(state)
        source_router_valid = self._router_valid(source_router_state)
        source_router_matches = self._router_matches_binding(
            source_router_state,
            state.consumer_binding,
        )
        prepared_state_matches = _tree_equal(state, prepared.source_state)
        prepared_router_matches = _tree_equal(
            source_router_state,
            prepared.source_router_state,
        )
        expected_augmented = self._augment(
            state.consumer_binding,
            prepared.base_observation,
        )
        expected_inputs = self._inputs(
            expected_augmented,
            prepared.primitive_action,
        )
        expected_prediction = self._predict_unchecked(
            state,
            prepared.base_observation,
            expected_augmented,
            prepared.primitive_action,
        ).replace(valid=prepared.prediction.valid)
        source_cache_matches = (
            _tree_equal(expected_augmented, prepared.augmented_observation)
            & _tree_equal(expected_inputs, prepared.input_features)
        )
        source_prediction_matches = _tree_equal(
            expected_prediction,
            prepared.prediction,
        )

        destination_binding_valid = self._binding_valid(event.destination_binding)
        destination_router_valid = self._router_valid(
            event.destination_router_state
        )
        destination_router_matches = self._router_matches_binding(
            event.destination_router_state,
            event.destination_binding,
        )
        descriptors_changed = jnp.any(
            state.consumer_binding.descriptors
            != event.destination_binding.descriptors
        )
        generation_changed = jnp.any(
            state.consumer_binding.semantic_generation_words
            != event.destination_binding.semantic_generation_words
        )
        generation_is_successor = _words_successor(
            state.consumer_binding.semantic_generation_words,
            event.destination_binding.semantic_generation_words,
        )
        bank_transition_consistent = (
            (~descriptors_changed & ~generation_changed)
            | (descriptors_changed & generation_changed & generation_is_successor)
        )
        event_values_valid = (
            prepared.prepared
            & jnp.all(jnp.isfinite(prepared.base_observation))
            & jnp.all(jnp.isfinite(prepared.augmented_observation))
            & jnp.all(jnp.isfinite(prepared.input_features))
            & (prepared.primitive_action >= 0)
            & (prepared.primitive_action < self._n_actions)
            & jnp.isfinite(event.reward)
            & jnp.isfinite(event.discount)
            & (event.discount >= 0.0)
            & (event.discount <= self._config.world_model.gamma)
            & jnp.all(jnp.isfinite(event.next_base_observation))
        )
        next_event_words, event_clock_capacity = (
            _checked_lifetime_words_increment(state.event_count_words)
        )
        event_capacity = event_clock_capacity & _words_leq_limit(
            next_event_words,
            self._config.max_events,
        )
        next_event_count = _saturating_int32_counter_increment(state.event_count)
        next_generation_words, generation_capacity = (
            _checked_lifetime_words_increment(state.generation_update_words)
        )
        next_generation_count = _saturating_int32_counter_increment(
            state.generation_update_count
        )
        targets = self._targets(
            prepared.base_observation,
            event.reward,
            event.discount,
            event.next_base_observation,
        )

        updated_members: list[RoutedLinearWorldModelMemberState] = []
        update_applied: list[Array] = []
        update_predictions: list[Array] = []
        for member in state.member_states:
            update = self._learner.update(
                member.learner_state,
                prepared.input_features,
                targets,
            )
            updated_members.append(
                RoutedLinearWorldModelMemberState(learner_state=update.state)
            )
            update_applied.append(update.update_applied)
            update_predictions.append(update.predictions)
        updated_member_tuple = tuple(updated_members)
        member_updates = jnp.stack(update_applied)
        update_prediction_matrix = jnp.stack(update_predictions)
        all_member_updates = (
            jnp.all(member_updates)
            & _tree_equal(
                update_prediction_matrix,
                prepared.prediction.member_raw_predictions,
            )
        )
        squared_residuals = jnp.square(
            targets[None, :] - prepared.prediction.member_raw_predictions
        )
        member_losses = jnp.mean(squared_residuals, axis=1)
        observed_loss = jnp.mean(member_losses)
        signal_config = self._config.signal_estimator
        prediction_values_valid = (
            prepared.prediction.valid
            & jnp.all(jnp.isfinite(targets))
            & jnp.all(jnp.abs(targets) <= signal_config.max_input_magnitude)
            & jnp.all(jnp.isfinite(squared_residuals))
            & jnp.all(
                squared_residuals <= signal_config.max_predicted_variance
            )
            & jnp.isfinite(observed_loss)
            & (observed_loss >= 0.0)
            & (observed_loss <= signal_config.max_observed_loss)
        )
        candidate_signal_state, raw_signals = self._signals.observe(
            state.signal_state,
            prepared.prediction.member_raw_predictions,
            state.residual_variances,
            targets,
            observed_loss,
        )
        signal_counter = raw_signals.counter_status
        signal_update_valid = (
            raw_signals.availability.input_valid
            & signal_counter.lifetime_counter_valid
            & signal_counter.lifetime_capacity_available
            & signal_counter.state_valid
            & signal_counter.event_recorded
            & signal_counter.valid_event_recorded
            & ~signal_counter.invalid_event_recorded
            & jnp.array_equal(
                signal_counter.pre_step_words,
                state.event_count_words,
            )
            & jnp.array_equal(
                signal_counter.pre_valid_words,
                state.event_count_words,
            )
            & jnp.array_equal(
                signal_counter.post_step_words,
                next_event_words,
            )
            & jnp.array_equal(
                signal_counter.post_valid_words,
                next_event_words,
            )
        )
        floor = jnp.asarray(
            self._config.residual_variance_floor,
            dtype=jnp.float32,
        )
        decay = jnp.asarray(
            self._config.residual_variance_decay,
            dtype=jnp.float32,
        )
        candidate_residuals = jnp.where(
            state.event_count == 0,
            jnp.maximum(squared_residuals, floor),
            jnp.maximum(
                decay * state.residual_variances
                + (1.0 - decay) * squared_residuals,
                floor,
            ),
        )
        residual_update_valid = (
            jnp.all(jnp.isfinite(candidate_residuals))
            & jnp.all(candidate_residuals >= floor)
            & jnp.all(
                candidate_residuals <= signal_config.max_predicted_variance
            )
        )
        routed_members, routed_router_state, route_diagnostics = (
            self._route_member_inputs(
                source_router_state,
                event.destination_binding.descriptors,
                updated_member_tuple,
            )
        )
        selected_members = cast(
            tuple[RoutedLinearWorldModelMemberState, ...],
            jax.lax.cond(
                descriptors_changed,
                lambda _: routed_members,
                lambda _: updated_member_tuple,
                None,
            ),
        )
        route_state_matches = jnp.where(
            descriptors_changed,
            _tree_equal(routed_router_state, event.destination_router_state),
            _tree_equal(source_router_state, event.destination_router_state),
        )
        selected_generation_count = jnp.where(
            descriptors_changed,
            jnp.asarray(0, dtype=jnp.int32),
            next_generation_count,
        )
        selected_generation_words = jnp.where(
            descriptors_changed,
            jnp.zeros((2,), dtype=jnp.uint32),
            next_generation_words,
        )
        selected_birth_words = jnp.where(
            descriptors_changed,
            next_event_words,
            state.generation_birth_event_words,
        )
        candidate = RoutedLinearWorldModelEnsembleState(
            member_states=selected_members,
            observation_min=jnp.minimum(
                state.observation_min,
                jnp.minimum(
                    prepared.base_observation,
                    event.next_base_observation,
                ),
            ),
            observation_max=jnp.maximum(
                state.observation_max,
                jnp.maximum(
                    prepared.base_observation,
                    event.next_base_observation,
                ),
            ),
            reward_min=jnp.minimum(state.reward_min, event.reward),
            reward_max=jnp.maximum(state.reward_max, event.reward),
            residual_variances=candidate_residuals,
            signal_state=candidate_signal_state,
            consumer_binding=event.destination_binding,
            event_count=next_event_count,
            event_count_words=next_event_words,
            generation_update_count=selected_generation_count,
            generation_update_words=selected_generation_words,
            generation_birth_event_words=selected_birth_words,
            schema_digest=state.schema_digest,
        )
        candidate_valid = self._state_valid(candidate)
        prerequisites = (
            state_valid
            & source_router_valid
            & source_router_matches
            & prepared_state_matches
            & prepared_router_matches
            & source_cache_matches
            & source_prediction_matches
            & destination_binding_valid
            & destination_router_valid
            & destination_router_matches
            & bank_transition_consistent
            & event_values_valid
            & event_capacity
            & generation_capacity
            & prediction_values_valid
            & all_member_updates
            & signal_update_valid
            & residual_update_valid
            & route_diagnostics.valid
            & route_state_matches
            & candidate_valid
        )
        selected_state = cast(
            RoutedLinearWorldModelEnsembleState,
            jax.lax.cond(
                prerequisites,
                lambda _: candidate,
                lambda _: state,
                None,
            ),
        )
        zero_signals = cast(
            TypedLearningSignals,
            jax.tree.map(jnp.zeros_like, raw_signals),
        )
        selected_signals = cast(
            TypedLearningSignals,
            jax.lax.cond(
                prerequisites,
                lambda _: raw_signals,
                lambda _: zero_signals,
                None,
            ),
        )
        committed_member_updates = member_updates & prerequisites
        diagnostics = RoutedLinearWorldModelEnsembleDiagnostics(
            state_valid=state_valid,
            source_router_valid=source_router_valid,
            source_router_matches_binding=source_router_matches,
            prepared_source_state_matches=prepared_state_matches,
            prepared_source_router_matches=prepared_router_matches,
            source_cache_matches=source_cache_matches,
            source_prediction_matches=source_prediction_matches,
            destination_binding_valid=destination_binding_valid,
            destination_router_valid=destination_router_valid,
            destination_router_matches_binding=destination_router_matches,
            descriptors_changed=descriptors_changed,
            generation_changed=generation_changed,
            generation_is_successor=generation_is_successor,
            bank_transition_consistent=bank_transition_consistent,
            event_values_valid=event_values_valid,
            event_capacity_available=event_capacity,
            all_member_updates_applied=all_member_updates,
            signal_update_valid=signal_update_valid,
            residual_update_valid=residual_update_valid,
            one_authoritative_route_evaluated=jnp.asarray(True),
            route_valid=route_diagnostics.valid,
            route_state_matches_destination=route_state_matches,
            candidate_state_valid=candidate_valid,
            transaction_applied=prerequisites,
            rejected=~prerequisites,
            router_evaluations=jnp.asarray(1, dtype=jnp.int32),
            member_update_evaluations=jnp.asarray(
                self._config.ensemble_size,
                dtype=jnp.int32,
            ),
            physical_output_head_count=jnp.asarray(
                self._target_dim,
                dtype=jnp.int32,
            ),
            generated_output_head_count=jnp.asarray(0, dtype=jnp.int32),
            planning_authority=jnp.asarray(False),
            safety_authority=jnp.asarray(False),
            pre_event_words=state.event_count_words,
            post_event_words=selected_state.event_count_words,
        )
        return RoutedLinearWorldModelEnsembleResult(
            state=selected_state,
            prediction=prepared.prediction,
            signals=selected_signals,
            targets=jnp.where(
                event_values_valid,
                targets,
                jnp.zeros((self._target_dim,), dtype=jnp.float32),
            ),
            observed_loss=jnp.where(
                prediction_values_valid,
                observed_loss,
                jnp.asarray(0.0, dtype=jnp.float32),
            ),
            member_prediction_losses=jnp.where(
                prediction_values_valid,
                member_losses,
                jnp.zeros((self._config.ensemble_size,), dtype=jnp.float32),
            ),
            member_updates_applied=committed_member_updates,
            route_diagnostics=route_diagnostics,
            diagnostics=diagnostics,
        )

    @property
    def resource_budget(self) -> RoutedLinearWorldModelEnsembleResourceBudget:
        """Return exact persistent/prediction bytes and fixed work ceilings."""

        state = self._template_state()
        base = jnp.zeros((self._base_dim,), dtype=jnp.float32)
        prediction = self._predict_unchecked(
            state,
            base,
            self._augment(state.consumer_binding, base),
            jnp.asarray(0, dtype=jnp.int32),
        )
        state_scalars, state_bytes = _tree_size(state)
        prediction_scalars, prediction_bytes = _tree_size(prediction)
        return RoutedLinearWorldModelEnsembleResourceBudget(
            ensemble_size=self._config.ensemble_size,
            base_feature_dim=self._base_dim,
            active_pair_slots=self._active_slots,
            model_input_dim=self._input_dim,
            physical_output_heads=self._target_dim,
            generated_output_heads=0,
            persistent_state_scalars=state_scalars,
            persistent_state_bytes=state_bytes,
            prediction_scalars=prediction_scalars,
            prediction_bytes=prediction_bytes,
            max_member_updates_per_event=self._config.ensemble_size,
            max_router_evaluations_per_event=1,
            max_events=self._config.max_events,
            feature_lifecycle_state_owned=0,
            router_state_owned=0,
            planning_authority=0,
            safety_authority=0,
            scientific_promotion_allowed=False,
        )


def measure_routed_linear_world_model_ensemble_state_nbytes(
    state: RoutedLinearWorldModelEnsembleState,
) -> int:
    """Measure every persistent array leaf in one ensemble state."""

    if type(state) is not RoutedLinearWorldModelEnsembleState:
        raise TypeError(
            "state must be an exact RoutedLinearWorldModelEnsembleState"
        )
    return _tree_size(state)[1]


def save_routed_linear_world_model_ensemble_checkpoint(
    owner: RoutedLinearWorldModelEnsemble,
    state: RoutedLinearWorldModelEnsembleState,
    path: str | Path,
) -> None:
    """Persist only the ensemble owner state, never lifecycle/router state."""

    if type(owner) is not RoutedLinearWorldModelEnsemble:
        raise TypeError("owner must be an exact RoutedLinearWorldModelEnsemble")
    if not bool(jax.device_get(owner.state_valid(state))):
        raise ValueError("refusing to save an invalid routed ensemble state")
    config = owner.to_config()
    save_checkpoint(
        state,
        path,
        metadata={
            "schema": ROUTED_LINEAR_WORLD_MODEL_ENSEMBLE_CHECKPOINT_SCHEMA,
            "owner_config": config,
            "config_sha256": _config_digest(config),
            "resource_budget": owner.resource_budget.to_config(),
            "evidence_level": ROUTED_LINEAR_WORLD_MODEL_ENSEMBLE_EVIDENCE_LEVEL,
            "outcome_status": ROUTED_LINEAR_WORLD_MODEL_ENSEMBLE_OUTCOME_STATUS,
            "scientific_promotion_allowed": False,
            "feature_lifecycle_state_included": False,
            "router_state_included": False,
            "planning_authority": False,
            "safety_authority": False,
        },
    )


def load_routed_linear_world_model_ensemble_checkpoint(
    path: str | Path,
) -> tuple[RoutedLinearWorldModelEnsemble, RoutedLinearWorldModelEnsembleState]:
    """Strictly restore the sole current routed-ensemble v1 schema."""

    metadata = load_checkpoint_metadata(path)
    expected_fields = {
        "schema",
        "owner_config",
        "config_sha256",
        "resource_budget",
        "evidence_level",
        "outcome_status",
        "scientific_promotion_allowed",
        "feature_lifecycle_state_included",
        "router_state_included",
        "planning_authority",
        "safety_authority",
    }
    if set(metadata) != expected_fields:
        raise ValueError("routed ensemble checkpoint metadata fields are not exact")
    if (
        metadata.get("schema")
        != ROUTED_LINEAR_WORLD_MODEL_ENSEMBLE_CHECKPOINT_SCHEMA
    ):
        raise ValueError("checkpoint is not a routed linear ensemble v1 checkpoint")
    config = metadata.get("owner_config")
    if type(config) is not dict:
        raise ValueError("routed ensemble checkpoint lacks exact owner_config")
    if metadata.get("config_sha256") != _config_digest(config):
        raise ValueError("routed ensemble checkpoint config digest does not match")
    owner = RoutedLinearWorldModelEnsemble.from_config(config)
    if metadata.get("resource_budget") != owner.resource_budget.to_config():
        raise ValueError("routed ensemble checkpoint resource budget does not match")
    if (
        metadata.get("evidence_level")
        != ROUTED_LINEAR_WORLD_MODEL_ENSEMBLE_EVIDENCE_LEVEL
    ):
        raise ValueError("routed ensemble checkpoint must remain L0")
    if (
        metadata.get("outcome_status")
        != ROUTED_LINEAR_WORLD_MODEL_ENSEMBLE_OUTCOME_STATUS
    ):
        raise ValueError("routed ensemble checkpoint must remain not_assessed")
    for name in (
        "scientific_promotion_allowed",
        "feature_lifecycle_state_included",
        "router_state_included",
        "planning_authority",
        "safety_authority",
    ):
        if metadata.get(name) is not False:
            raise ValueError(f"routed ensemble checkpoint {name} must be false")
    template = owner._template_state()
    restored, second_metadata = load_checkpoint(template, path)
    if second_metadata != metadata:
        raise ValueError("routed ensemble checkpoint metadata changed between reads")
    state = cast(RoutedLinearWorldModelEnsembleState, restored)
    if not bool(jax.device_get(owner.state_valid(state))):
        raise ValueError("routed ensemble checkpoint restored an invalid state")
    if measure_routed_linear_world_model_ensemble_state_nbytes(state) != (
        owner.resource_budget.persistent_state_bytes
    ):
        raise ValueError("routed ensemble checkpoint restored a wrong-size state")
    return owner, state


__all__ = [
    "ROUTED_LINEAR_WORLD_MODEL_ENSEMBLE_CHECKPOINT_SCHEMA",
    "ROUTED_LINEAR_WORLD_MODEL_ENSEMBLE_CONFIG_SCHEMA",
    "ROUTED_LINEAR_WORLD_MODEL_ENSEMBLE_EVIDENCE_LEVEL",
    "ROUTED_LINEAR_WORLD_MODEL_ENSEMBLE_OUTCOME_STATUS",
    "ROUTED_LINEAR_WORLD_MODEL_ENSEMBLE_SCIENTIFIC_PROMOTION_ALLOWED",
    "ROUTED_LINEAR_WORLD_MODEL_ENSEMBLE_STATE_SCHEMA",
    "RoutedLinearWorldModelEnsemble",
    "RoutedLinearWorldModelEnsembleConfig",
    "RoutedLinearWorldModelEnsembleDiagnostics",
    "RoutedLinearWorldModelEnsemblePrediction",
    "RoutedLinearWorldModelEnsemblePrepareDiagnostics",
    "RoutedLinearWorldModelEnsemblePrepareResult",
    "RoutedLinearWorldModelEnsemblePreparedTransition",
    "RoutedLinearWorldModelEnsembleResourceBudget",
    "RoutedLinearWorldModelEnsembleResult",
    "RoutedLinearWorldModelEnsembleState",
    "RoutedLinearWorldModelEnsembleTransition",
    "RoutedLinearWorldModelMemberState",
    "load_routed_linear_world_model_ensemble_checkpoint",
    "measure_routed_linear_world_model_ensemble_state_nbytes",
    "save_routed_linear_world_model_ensemble_checkpoint",
]
