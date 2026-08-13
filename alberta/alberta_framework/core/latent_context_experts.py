# mypy: disable-error-code="call-arg,name-defined"
"""Leakage-safe online latent-context regression experts.

This module is a generic integration of the active-only-freeze law already
implemented by :class:`alberta_framework.core.context_inference.ContextInference`.
It is not a conceptually novel context-inference algorithm.  The existing
module demonstrates that an outcome-inferred active slot can learn while
inactive slot parameters remain exact memory.  This sibling supplies the
missing generic regression and predict-before-outcome transaction boundary:

* a fixed bank of linear vector-regression experts exists from birth;
* :meth:`LatentContextExpertLearner.predict` binds the current active owner,
  complete parameters, observation, and every expert prediction before any
  target is accepted;
* :meth:`LatentContextExpertLearner.update` uses the newly observed target to
  select only the *next* owner and the one committed expert update;
* the cached current prediction and its prequential error are never relabeled
  after seeing the target;
* exact loss ties retain the cached owner; and
* every nonselected expert parameter subtree remains bit-identical.

All experts, losses, and analytic candidate gradients are evaluated on every
accepted transaction.  Selective gating changes only which one candidate is
committed.  With ``selective_gating=False`` (the default), the cached owner is
retained, yielding a same-shape/same-work single-expert ablation.  There are no
task identifiers, boundaries, resets, replay samples, allocations, evictions,
future observations, or online random draws.
"""

from __future__ import annotations

import dataclasses
import functools
import math
from collections.abc import Mapping
from numbers import Real
from typing import Any, cast

import chex
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import Bool, Float, Int, UInt

LATENT_CONTEXT_EXPERT_DESIGN_SCHEMA = "alberta.latent-context-expert.design.v1"
LATENT_CONTEXT_EXPERT_CONFIG_SCHEMA = "alberta.latent-context-expert.config.v1"
LATENT_CONTEXT_EXPERT_STATE_SCHEMA = "alberta.latent-context-expert.state.v1"
LATENT_CONTEXT_EXPERT_CACHE_SCHEMA = "alberta.latent-context-expert.cache.v1"
LATENT_CONTEXT_EXPERT_RESULT_SCHEMA = "alberta.latent-context-expert.update-result.v1"
LATENT_CONTEXT_EXPERT_RESOURCE_SCHEMA = "alberta.latent-context-expert.resource-record.v1"

LATENT_CONTEXT_EXPERT_EXACT_LIFETIME_NBYTES = 8
LATENT_CONTEXT_EXPERT_LIFETIME_COUNTER_NBYTES = 12

_INT32_MAX = 2**31 - 1
_UINT32_MAX = 2**32 - 1

_CONFIG_VALUES = {
    "input_dim",
    "output_dim",
    "max_experts",
    "step_size",
    "grad_clip",
    "selective_gating",
}
_CONFIG_FIELDS = _CONFIG_VALUES | {
    "type",
    "schema",
    "design_schema",
    "state_schema",
    "cache_schema",
    "result_schema",
    "resource_schema",
}


@dataclasses.dataclass(frozen=True)
class LatentContextExpertDesignRecord:
    """Machine-readable credit and scope boundary."""

    schema: str
    method_name: str
    conceptual_novelty_claimed: bool
    prior_mechanism: str
    prior_module: str
    integration_scope: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-data record."""

        return dataclasses.asdict(self)


def latent_context_expert_design_record() -> LatentContextExpertDesignRecord:
    """Return the fixed attribution and non-novelty declaration."""

    return LatentContextExpertDesignRecord(
        schema=LATENT_CONTEXT_EXPERT_DESIGN_SCHEMA,
        method_name="Leakage-safe online latent-context regression experts",
        conceptual_novelty_claimed=False,
        prior_mechanism="ContextInference active-only-freeze law",
        prior_module="alberta_framework.core.context_inference",
        integration_scope=(
            "predict-before-outcome ownership cache",
            "generic regression inputs and vector targets",
            "relative current-outcome evidence without reward-scale thresholds",
            "same-state and same-candidate-work no-selection ablation",
        ),
    )


@chex.dataclass(frozen=True)
class LatentContextExpertConfig:
    """Strict fixed-bank construction.

    ``selective_gating`` defaults off.  A positive ``grad_clip`` is a global
    per-expert gradient norm cap; zero disables clipping.
    """

    input_dim: int
    output_dim: int = 1
    max_experts: int = 2
    step_size: float = 5.0e-2
    grad_clip: float = 10.0
    selective_gating: bool = False

    def to_config(self) -> dict[str, Any]:
        """Serialize the exact versioned config."""

        return {
            "type": "LatentContextExpertConfig",
            "schema": LATENT_CONTEXT_EXPERT_CONFIG_SCHEMA,
            "design_schema": LATENT_CONTEXT_EXPERT_DESIGN_SCHEMA,
            "state_schema": LATENT_CONTEXT_EXPERT_STATE_SCHEMA,
            "cache_schema": LATENT_CONTEXT_EXPERT_CACHE_SCHEMA,
            "result_schema": LATENT_CONTEXT_EXPERT_RESULT_SCHEMA,
            "resource_schema": LATENT_CONTEXT_EXPERT_RESOURCE_SCHEMA,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "max_experts": self.max_experts,
            "step_size": self.step_size,
            "grad_clip": self.grad_clip,
            "selective_gating": self.selective_gating,
        }

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> LatentContextExpertConfig:
        """Reconstruct only the exact current config schema."""

        if not isinstance(config, Mapping):
            raise TypeError("latent-context expert config must be a mapping")
        payload = dict(config)
        if set(payload) != _CONFIG_FIELDS:
            missing = sorted(_CONFIG_FIELDS - set(payload))
            extra = sorted(set(payload) - _CONFIG_FIELDS)
            raise ValueError(
                "latent-context expert config fields are invalid; "
                f"missing={missing}, extra={extra}"
            )
        schemas = {
            "type": "LatentContextExpertConfig",
            "schema": LATENT_CONTEXT_EXPERT_CONFIG_SCHEMA,
            "design_schema": LATENT_CONTEXT_EXPERT_DESIGN_SCHEMA,
            "state_schema": LATENT_CONTEXT_EXPERT_STATE_SCHEMA,
            "cache_schema": LATENT_CONTEXT_EXPERT_CACHE_SCHEMA,
            "result_schema": LATENT_CONTEXT_EXPERT_RESULT_SCHEMA,
            "resource_schema": LATENT_CONTEXT_EXPERT_RESOURCE_SCHEMA,
        }
        for name, expected in schemas.items():
            if payload.pop(name) != expected:
                raise ValueError(f"latent-context expert {name} is unsupported")
        restored = cls(**payload)
        _validate_config(restored)
        return restored


@chex.dataclass(frozen=True)
class LatentContextExpertParams:
    """Independent fixed-capacity linear expert subtrees."""

    expert_weights: Float[Array, "max_experts input_dim output_dim"]
    expert_biases: Float[Array, "max_experts output_dim"]


@chex.dataclass(frozen=True)
class LatentContextExpertState:
    """Persistent parameters, next prediction owner, and exact lifetime."""

    params: LatentContextExpertParams
    active_expert: Int[Array, ""]
    step_count: Int[Array, ""]
    step_words: UInt[Array, " 2"]


@chex.dataclass(frozen=True)
class LatentContextExpertPredictionCache:
    """Bound pre-target owner, parameter snapshot, observation, and predictions."""

    owner_params: LatentContextExpertParams
    owner_step_words: UInt[Array, " 2"]
    owner_active_expert: Int[Array, ""]
    observation: Float[Array, " input_dim"]
    expert_predictions: Float[Array, "max_experts output_dim"]
    prediction: Float[Array, " output_dim"]
    valid: Bool[Array, ""]


@chex.dataclass(frozen=True)
class LatentContextExpertUpdateResult:
    """One target-owned transaction and explicit causal routing facts."""

    state: LatentContextExpertState
    prediction: Float[Array, " output_dim"]
    error: Float[Array, " output_dim"]
    expert_predictions: Float[Array, "max_experts output_dim"]
    expert_losses: Float[Array, " max_experts"]
    candidate_gradient_norms: Float[Array, " max_experts"]
    pre_update_owner: Int[Array, ""]
    evidence_best_expert: Int[Array, ""]
    selected_next_expert: Int[Array, ""]
    expert_update_mask: Bool[Array, " max_experts"]
    context_switched: Bool[Array, ""]
    pre_step_words: UInt[Array, " 2"]
    post_step_words: UInt[Array, " 2"]
    lifetime_counter_valid: Bool[Array, ""]
    lifetime_capacity_available: Bool[Array, ""]
    source_state_valid: Bool[Array, ""]
    cache_owner_valid: Bool[Array, ""]
    cache_input_valid: Bool[Array, ""]
    cache_prediction_exact: Bool[Array, ""]
    target_valid: Bool[Array, ""]
    evidence_valid: Bool[Array, ""]
    candidate_gradients_valid: Bool[Array, ""]
    candidate_state_valid: Bool[Array, ""]
    update_applied: Bool[Array, ""]


@chex.dataclass(frozen=True)
class LatentContextExpertLearningResult:
    """Fixed-shape scan trace."""

    state: LatentContextExpertState
    predictions: Float[Array, "steps output_dim"]
    errors: Float[Array, "steps output_dim"]
    expert_predictions: Float[Array, "steps max_experts output_dim"]
    expert_losses: Float[Array, "steps max_experts"]
    candidate_gradient_norms: Float[Array, "steps max_experts"]
    pre_update_owner: Int[Array, " steps"]
    evidence_best_expert: Int[Array, " steps"]
    selected_next_expert: Int[Array, " steps"]
    expert_update_mask: Bool[Array, "steps max_experts"]
    context_switched: Bool[Array, " steps"]
    pre_step_words: UInt[Array, "steps 2"]
    post_step_words: UInt[Array, "steps 2"]
    update_applied: Bool[Array, " steps"]


@dataclasses.dataclass(frozen=True)
class LatentContextExpertResourceRecord:
    """Exact persistent/cache bytes and bounded per-event work."""

    schema: str
    design_schema: str
    config_schema: str
    state_schema: str
    cache_schema: str
    result_schema: str
    input_dim: int
    output_dim: int
    max_experts: int
    parameter_nbytes: int
    exact_lifetime_identity_nbytes: int
    lifetime_counter_nbytes: int
    state_nbytes: int
    prediction_cache_nbytes: int
    maximum_preoutcome_expert_predictions_per_update: int
    maximum_cache_authentication_expert_predictions_per_update: int
    maximum_expert_predictions_per_update: int
    maximum_expert_losses_per_update: int
    maximum_candidate_gradients_per_update: int
    maximum_expert_subtree_commits_per_update: int
    replay_capacity: int
    maximum_stored_examples: int
    persistent_capacity_growth: int
    online_random_draws_per_update: int

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-data resource record."""

        return dataclasses.asdict(self)


def _require_dimension(value: Any, *, name: str, minimum: int = 1) -> None:
    if type(value) is not int or not minimum <= value <= _INT32_MAX:
        raise ValueError(f"{name} must be an exact integer in [{minimum}, {_INT32_MAX}]")


def _require_float(value: Any, *, name: str, minimum: float = 0.0) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real scalar")
    parsed = float(value)
    with np.errstate(over="ignore", invalid="ignore"):
        narrowed = float(np.float32(parsed))
    if not math.isfinite(parsed) or not math.isfinite(narrowed) or narrowed < minimum:
        raise ValueError(f"{name} must be finite in float32 and at least {minimum}")


def _validate_config(config: LatentContextExpertConfig) -> None:
    if not isinstance(config, LatentContextExpertConfig):
        raise TypeError("config must be LatentContextExpertConfig")
    _require_dimension(config.input_dim, name="input_dim")
    _require_dimension(config.output_dim, name="output_dim")
    _require_dimension(config.max_experts, name="max_experts", minimum=2)
    _require_float(config.step_size, name="step_size")
    _require_float(config.grad_clip, name="grad_clip")
    if type(config.selective_gating) is not bool:
        raise ValueError("selective_gating must be an exact bool")


def _require_array(
    value: Any,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: Any,
) -> Array:
    array = jnp.asarray(value)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    expected = jnp.dtype(dtype)
    if array.dtype != expected:
        raise TypeError(f"{name} must have dtype {expected}, got {array.dtype}")
    return array


def _checked_words_increment(words: Array) -> tuple[Array, Array]:
    _require_array(words, name="step_words", shape=(2,), dtype=jnp.uint32)
    maximum = jnp.asarray(_UINT32_MAX, dtype=jnp.uint32)
    capacity = ~jnp.all(words == maximum)
    low = words[1] + jnp.asarray(1, dtype=jnp.uint32)
    carry = (low == jnp.asarray(0, dtype=jnp.uint32)).astype(jnp.uint32)
    proposed = jnp.stack((words[0] + carry, low)).astype(jnp.uint32)
    return jnp.where(capacity, proposed, words), capacity


def _words_to_telemetry(words: Array) -> Array:
    _require_array(words, name="step_words", shape=(2,), dtype=jnp.uint32)
    below = (words[0] == jnp.asarray(0, dtype=jnp.uint32)) & (
        words[1] < jnp.asarray(_INT32_MAX, dtype=jnp.uint32)
    )
    return jnp.where(
        below,
        words[1].astype(jnp.int32),
        jnp.asarray(_INT32_MAX, dtype=jnp.int32),
    )


def _counter_valid(words: Array, telemetry: Array) -> Array:
    _require_array(telemetry, name="step_count", shape=(), dtype=jnp.int32)
    return (telemetry >= 0) & (telemetry == _words_to_telemetry(words))


def _tree_finite(tree: Any) -> Array:
    checks = [
        jnp.all(jnp.isfinite(jnp.asarray(leaf)))
        for leaf in jax.tree_util.tree_leaves(tree)
        if jnp.issubdtype(jnp.asarray(leaf).dtype, jnp.inexact)
    ]
    return jnp.all(jnp.stack(checks)) if checks else jnp.asarray(True)


def _tree_exact(left: Any, right: Any) -> Array:
    left_leaves, left_tree = jax.tree_util.tree_flatten(left)
    right_leaves, right_tree = jax.tree_util.tree_flatten(right)
    if cast(Any, left_tree) != right_tree or len(left_leaves) != len(right_leaves):
        return jnp.asarray(False, dtype=jnp.bool_)
    return jnp.all(
        jnp.stack(
            [
                jnp.array_equal(jnp.asarray(a), jnp.asarray(b))
                for a, b in zip(left_leaves, right_leaves, strict=True)
            ]
        )
    )


def _tree_nbytes(tree: Any) -> int:
    return sum(
        int(array.size) * int(array.dtype.itemsize)
        for leaf in jax.tree_util.tree_leaves(tree)
        if isinstance((array := jnp.asarray(leaf)), Array)
    )


def latent_context_expert_forward(
    params: LatentContextExpertParams,
    observation: Float[Array, " input_dim"],
) -> Float[Array, "max_experts output_dim"]:
    """Return every expert prediction without selecting an owner."""

    return jnp.einsum("i,kio->ko", observation, params.expert_weights) + params.expert_biases


class LatentContextExpertLearner:
    """Fixed-bank causal expert learner with active-only commits."""

    def __init__(self, config: LatentContextExpertConfig):
        _validate_config(config)
        self._config = config

    @property
    def config(self) -> LatentContextExpertConfig:
        """Return the immutable construction."""

        return self._config

    @property
    def design_record(self) -> LatentContextExpertDesignRecord:
        """Return the credit and scope boundary."""

        return latent_context_expert_design_record()

    def to_config(self) -> dict[str, Any]:
        """Serialize a strict learner wrapper."""

        return {
            "type": "LatentContextExpertLearner",
            "schema": LATENT_CONTEXT_EXPERT_CONFIG_SCHEMA,
            "design_schema": LATENT_CONTEXT_EXPERT_DESIGN_SCHEMA,
            "state_schema": LATENT_CONTEXT_EXPERT_STATE_SCHEMA,
            "cache_schema": LATENT_CONTEXT_EXPERT_CACHE_SCHEMA,
            "result_schema": LATENT_CONTEXT_EXPERT_RESULT_SCHEMA,
            "resource_schema": LATENT_CONTEXT_EXPERT_RESOURCE_SCHEMA,
            "config": self._config.to_config(),
        }

    def init(self) -> LatentContextExpertState:
        """Return identical neutral experts with expert zero as the first owner."""

        c = self._config
        return LatentContextExpertState(
            params=LatentContextExpertParams(
                expert_weights=jnp.zeros(
                    (c.max_experts, c.input_dim, c.output_dim),
                    dtype=jnp.float32,
                ),
                expert_biases=jnp.zeros(
                    (c.max_experts, c.output_dim),
                    dtype=jnp.float32,
                ),
            ),
            active_expert=jnp.asarray(0, dtype=jnp.int32),
            step_count=jnp.asarray(0, dtype=jnp.int32),
            step_words=jnp.zeros((2,), dtype=jnp.uint32),
        )

    def _require_params_contract(self, params: LatentContextExpertParams) -> None:
        if not isinstance(params, LatentContextExpertParams):
            raise TypeError("params must be LatentContextExpertParams")
        c = self._config
        _require_array(
            params.expert_weights,
            name="params.expert_weights",
            shape=(c.max_experts, c.input_dim, c.output_dim),
            dtype=jnp.float32,
        )
        _require_array(
            params.expert_biases,
            name="params.expert_biases",
            shape=(c.max_experts, c.output_dim),
            dtype=jnp.float32,
        )

    def _require_state_contract(self, state: LatentContextExpertState) -> None:
        if not isinstance(state, LatentContextExpertState):
            raise TypeError("state must be LatentContextExpertState")
        self._require_params_contract(state.params)
        _require_array(
            state.active_expert,
            name="state.active_expert",
            shape=(),
            dtype=jnp.int32,
        )
        _require_array(state.step_count, name="step_count", shape=(), dtype=jnp.int32)
        _require_array(state.step_words, name="step_words", shape=(2,), dtype=jnp.uint32)

    def _require_cache_contract(self, cache: LatentContextExpertPredictionCache) -> None:
        if not isinstance(cache, LatentContextExpertPredictionCache):
            raise TypeError("cache must be LatentContextExpertPredictionCache")
        self._require_params_contract(cache.owner_params)
        c = self._config
        _require_array(
            cache.owner_step_words,
            name="cache.owner_step_words",
            shape=(2,),
            dtype=jnp.uint32,
        )
        _require_array(
            cache.owner_active_expert,
            name="cache.owner_active_expert",
            shape=(),
            dtype=jnp.int32,
        )
        _require_array(
            cache.observation,
            name="cache.observation",
            shape=(c.input_dim,),
            dtype=jnp.float32,
        )
        _require_array(
            cache.expert_predictions,
            name="cache.expert_predictions",
            shape=(c.max_experts, c.output_dim),
            dtype=jnp.float32,
        )
        _require_array(
            cache.prediction,
            name="cache.prediction",
            shape=(c.output_dim,),
            dtype=jnp.float32,
        )
        _require_array(cache.valid, name="cache.valid", shape=(), dtype=jnp.bool_)

    def state_valid(self, state: LatentContextExpertState) -> Array:
        """Return dynamic validity after enforcing shapes and dtypes."""

        self._require_state_contract(state)
        return (
            _tree_finite(state.params)
            & (state.active_expert >= 0)
            & (state.active_expert < self._config.max_experts)
            & _counter_valid(state.step_words, state.step_count)
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def predict(
        self,
        state: LatentContextExpertState,
        observation: Float[Array, " input_dim"],
    ) -> LatentContextExpertPredictionCache:
        """Bind the current owner and prediction before any target is supplied."""

        self._require_state_contract(state)
        checked = _require_array(
            observation,
            name="observation",
            shape=(self._config.input_dim,),
            dtype=jnp.float32,
        )
        state_valid = self.state_valid(state)
        input_valid = jnp.all(jnp.isfinite(checked))
        safe_observation = jnp.where(input_valid, checked, jnp.zeros_like(checked))
        raw_predictions = latent_context_expert_forward(state.params, safe_observation)
        predictions_valid = jnp.all(jnp.isfinite(raw_predictions))
        valid = state_valid & input_valid & predictions_valid
        safe_active = jnp.clip(state.active_expert, 0, self._config.max_experts - 1)
        expert_predictions = jnp.where(
            valid,
            raw_predictions,
            jnp.zeros_like(raw_predictions),
        )
        prediction = jnp.where(
            valid,
            raw_predictions[safe_active],
            jnp.zeros((self._config.output_dim,), dtype=jnp.float32),
        )
        return LatentContextExpertPredictionCache(
            owner_params=state.params,
            owner_step_words=state.step_words,
            owner_active_expert=state.active_expert,
            observation=safe_observation,
            expert_predictions=expert_predictions,
            prediction=prediction,
            valid=valid,
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def update(
        self,
        state: LatentContextExpertState,
        cache: LatentContextExpertPredictionCache,
        target: Float[Array, " output_dim"],
    ) -> LatentContextExpertUpdateResult:
        """Use target evidence only for the next owner and one expert commit."""

        self._require_state_contract(state)
        self._require_cache_contract(cache)
        c = self._config
        checked_target = _require_array(
            target,
            name="target",
            shape=(c.output_dim,),
            dtype=jnp.float32,
        )
        lifetime_counter_valid = _counter_valid(state.step_words, state.step_count)
        source_state_valid = self.state_valid(state)
        cache_input_valid = jnp.all(jnp.isfinite(cache.observation))
        cache_owner_valid = (
            cache.valid
            & jnp.array_equal(cache.owner_step_words, state.step_words)
            & (cache.owner_active_expert == state.active_expert)
            & _tree_exact(cache.owner_params, state.params)
        )
        safe_active = jnp.clip(state.active_expert, 0, c.max_experts - 1)
        recomputed_predictions = latent_context_expert_forward(
            state.params,
            cache.observation,
        )
        cache_prediction_exact = (
            jnp.array_equal(cache.expert_predictions, recomputed_predictions)
            & jnp.array_equal(cache.prediction, recomputed_predictions[safe_active])
        )
        target_valid = jnp.all(jnp.isfinite(checked_target))
        safe_target = jnp.where(target_valid, checked_target, jnp.zeros_like(checked_target))

        # The current prequential prediction is irrevocably the cached owner's
        # output.  Target evidence is computed for all experts only afterward.
        expert_errors = safe_target[None, :] - cache.expert_predictions
        expert_losses = jnp.mean(jnp.square(expert_errors), axis=1)
        evidence_valid = jnp.all(jnp.isfinite(expert_losses))
        safe_losses = jnp.where(jnp.isfinite(expert_losses), expert_losses, jnp.inf)
        raw_best = jnp.argmin(safe_losses).astype(jnp.int32)
        minimum_loss = safe_losses[raw_best]
        active_tied_for_best = safe_losses[safe_active] == minimum_loss
        evidence_best = jnp.where(active_tied_for_best, safe_active, raw_best).astype(jnp.int32)
        selected = jnp.where(c.selective_gating, evidence_best, safe_active).astype(jnp.int32)

        # Analytic MSE candidates are evaluated for every expert in both arms.
        gradient_factor = jnp.asarray(-2.0 / c.output_dim, dtype=jnp.float32)
        weight_gradients = (
            gradient_factor
            * cache.observation[None, :, None]
            * expert_errors[:, None, :]
        )
        bias_gradients = gradient_factor * expert_errors
        gradient_norms = jnp.sqrt(
            jnp.sum(jnp.square(weight_gradients), axis=(1, 2))
            + jnp.sum(jnp.square(bias_gradients), axis=1)
        )
        gradients_valid = (
            jnp.all(jnp.isfinite(weight_gradients))
            & jnp.all(jnp.isfinite(bias_gradients))
            & jnp.all(jnp.isfinite(gradient_norms))
        )
        if c.grad_clip > 0.0:
            scales = jnp.minimum(
                1.0,
                jnp.asarray(c.grad_clip, dtype=jnp.float32)
                / jnp.maximum(gradient_norms, jnp.asarray(1.0e-12, dtype=jnp.float32)),
            )
        else:
            scales = jnp.ones_like(gradient_norms)
        clipped_weight_gradients = scales[:, None, None] * weight_gradients
        clipped_bias_gradients = scales[:, None] * bias_gradients
        candidate_weights = state.params.expert_weights - jnp.asarray(
            c.step_size,
            dtype=jnp.float32,
        ) * clipped_weight_gradients
        candidate_biases = state.params.expert_biases - jnp.asarray(
            c.step_size,
            dtype=jnp.float32,
        ) * clipped_bias_gradients
        selected_mask = jnp.arange(c.max_experts, dtype=jnp.int32) == selected
        candidate_params = LatentContextExpertParams(
            expert_weights=jnp.where(
                selected_mask[:, None, None],
                candidate_weights,
                state.params.expert_weights,
            ),
            expert_biases=jnp.where(
                selected_mask[:, None],
                candidate_biases,
                state.params.expert_biases,
            ),
        )
        proposed_words, lifetime_capacity_available = _checked_words_increment(
            state.step_words
        )
        candidate_state = LatentContextExpertState(
            params=candidate_params,
            active_expert=selected,
            step_count=_words_to_telemetry(proposed_words),
            step_words=proposed_words,
        )
        candidate_state_valid = self.state_valid(candidate_state)
        update_applied = (
            source_state_valid
            & cache_owner_valid
            & cache_input_valid
            & cache_prediction_exact
            & target_valid
            & evidence_valid
            & gradients_valid
            & candidate_state_valid
            & lifetime_capacity_available
        )
        committed_state = cast(
            LatentContextExpertState,
            jax.tree_util.tree_map(
                lambda proposed, current: jnp.where(update_applied, proposed, current),
                candidate_state,
                state,
            ),
        )
        zero_output = jnp.zeros((c.output_dim,), dtype=jnp.float32)
        zero_expert_outputs = jnp.zeros((c.max_experts, c.output_dim), dtype=jnp.float32)
        zero_expert_scalars = jnp.zeros((c.max_experts,), dtype=jnp.float32)
        no_expert = jnp.asarray(-1, dtype=jnp.int32)
        applied_owner = jnp.where(update_applied, cache.owner_active_expert, no_expert)
        applied_best = jnp.where(update_applied, evidence_best, no_expert)
        applied_selected = jnp.where(update_applied, selected, no_expert)
        applied_mask = update_applied & selected_mask
        return LatentContextExpertUpdateResult(
            state=committed_state,
            prediction=jnp.where(update_applied, cache.prediction, zero_output),
            error=jnp.where(
                update_applied,
                safe_target - cache.prediction,
                zero_output,
            ),
            expert_predictions=jnp.where(
                update_applied,
                cache.expert_predictions,
                zero_expert_outputs,
            ),
            expert_losses=jnp.where(
                update_applied,
                expert_losses,
                zero_expert_scalars,
            ),
            candidate_gradient_norms=jnp.where(
                update_applied,
                gradient_norms,
                zero_expert_scalars,
            ),
            pre_update_owner=applied_owner,
            evidence_best_expert=applied_best,
            selected_next_expert=applied_selected,
            expert_update_mask=applied_mask,
            context_switched=update_applied
            & (cache.owner_active_expert != selected),
            pre_step_words=state.step_words,
            post_step_words=committed_state.step_words,
            lifetime_counter_valid=lifetime_counter_valid,
            lifetime_capacity_available=lifetime_capacity_available,
            source_state_valid=source_state_valid,
            cache_owner_valid=cache_owner_valid,
            cache_input_valid=cache_input_valid,
            cache_prediction_exact=cache_prediction_exact,
            target_valid=target_valid,
            evidence_valid=evidence_valid,
            candidate_gradients_valid=gradients_valid,
            candidate_state_valid=candidate_state_valid,
            update_applied=update_applied,
        )

    def resource_record(
        self,
        state: LatentContextExpertState | None = None,
    ) -> LatentContextExpertResourceRecord:
        """Return exact fixed capacity and maximum transaction work."""

        measured = self.init() if state is None else state
        self._require_state_contract(measured)
        if not bool(jax.device_get(self.state_valid(measured))):
            raise ValueError("cannot account an invalid latent-context expert state")
        parameter_nbytes = _tree_nbytes(measured.params)
        state_nbytes = _tree_nbytes(measured)
        cache = self.predict(
            measured,
            jnp.zeros((self._config.input_dim,), dtype=jnp.float32),
        )
        k = self._config.max_experts
        return LatentContextExpertResourceRecord(
            schema=LATENT_CONTEXT_EXPERT_RESOURCE_SCHEMA,
            design_schema=LATENT_CONTEXT_EXPERT_DESIGN_SCHEMA,
            config_schema=LATENT_CONTEXT_EXPERT_CONFIG_SCHEMA,
            state_schema=LATENT_CONTEXT_EXPERT_STATE_SCHEMA,
            cache_schema=LATENT_CONTEXT_EXPERT_CACHE_SCHEMA,
            result_schema=LATENT_CONTEXT_EXPERT_RESULT_SCHEMA,
            input_dim=self._config.input_dim,
            output_dim=self._config.output_dim,
            max_experts=k,
            parameter_nbytes=parameter_nbytes,
            exact_lifetime_identity_nbytes=LATENT_CONTEXT_EXPERT_EXACT_LIFETIME_NBYTES,
            lifetime_counter_nbytes=LATENT_CONTEXT_EXPERT_LIFETIME_COUNTER_NBYTES,
            state_nbytes=state_nbytes,
            prediction_cache_nbytes=_tree_nbytes(cache),
            maximum_preoutcome_expert_predictions_per_update=k,
            maximum_cache_authentication_expert_predictions_per_update=k,
            maximum_expert_predictions_per_update=2 * k,
            maximum_expert_losses_per_update=k,
            maximum_candidate_gradients_per_update=k,
            maximum_expert_subtree_commits_per_update=1,
            replay_capacity=0,
            maximum_stored_examples=0,
            persistent_capacity_growth=0,
            online_random_draws_per_update=0,
        )


def measure_latent_context_expert_state_nbytes(state: LatentContextExpertState) -> int:
    """Measure every persistent array byte."""

    if not isinstance(state, LatentContextExpertState):
        raise TypeError("state must be LatentContextExpertState")
    return _tree_nbytes(state)


def run_latent_context_expert_arrays(
    learner: LatentContextExpertLearner,
    observations: Float[Array, "steps input_dim"],
    targets: Float[Array, "steps output_dim"],
    *,
    state: LatentContextExpertState | None = None,
) -> LatentContextExpertLearningResult:
    """Run a stream while constructing every prediction cache before its target update."""

    observation_array = jnp.asarray(observations)
    expected_observation_tail = (learner.config.input_dim,)
    if observation_array.ndim != 2 or observation_array.shape[1:] != expected_observation_tail:
        raise ValueError(
            "observations must have shape "
            f"(steps, {learner.config.input_dim}), got {observation_array.shape}"
        )
    if observation_array.dtype != jnp.dtype(jnp.float32):
        raise TypeError(f"observations must have dtype float32, got {observation_array.dtype}")
    target_array = jnp.asarray(targets)
    expected_targets = (observation_array.shape[0], learner.config.output_dim)
    if target_array.shape != expected_targets:
        raise ValueError(f"targets must have shape {expected_targets}, got {target_array.shape}")
    if target_array.dtype != jnp.dtype(jnp.float32):
        raise TypeError(f"targets must have dtype float32, got {target_array.dtype}")
    initial = learner.init() if state is None else state
    learner._require_state_contract(initial)

    def step(
        carry: LatentContextExpertState,
        inputs: tuple[Array, Array],
    ) -> tuple[LatentContextExpertState, tuple[Array, ...]]:
        observation, target = inputs
        cache = learner.predict(carry, observation)
        result = learner.update(carry, cache, target)
        return result.state, (
            result.prediction,
            result.error,
            result.expert_predictions,
            result.expert_losses,
            result.candidate_gradient_norms,
            result.pre_update_owner,
            result.evidence_best_expert,
            result.selected_next_expert,
            result.expert_update_mask,
            result.context_switched,
            result.pre_step_words,
            result.post_step_words,
            result.update_applied,
        )

    final_state, outputs = jax.lax.scan(
        step,
        initial,
        (observation_array, target_array),
    )
    (
        predictions,
        errors,
        expert_predictions,
        expert_losses,
        candidate_gradient_norms,
        pre_update_owner,
        evidence_best_expert,
        selected_next_expert,
        expert_update_mask,
        context_switched,
        pre_step_words,
        post_step_words,
        update_applied,
    ) = outputs
    return LatentContextExpertLearningResult(
        state=final_state,
        predictions=predictions,
        errors=errors,
        expert_predictions=expert_predictions,
        expert_losses=expert_losses,
        candidate_gradient_norms=candidate_gradient_norms,
        pre_update_owner=pre_update_owner,
        evidence_best_expert=evidence_best_expert,
        selected_next_expert=selected_next_expert,
        expert_update_mask=expert_update_mask,
        context_switched=context_switched,
        pre_step_words=pre_step_words,
        post_step_words=post_step_words,
        update_applied=update_applied,
    )


__all__ = [
    "LATENT_CONTEXT_EXPERT_CACHE_SCHEMA",
    "LATENT_CONTEXT_EXPERT_CONFIG_SCHEMA",
    "LATENT_CONTEXT_EXPERT_DESIGN_SCHEMA",
    "LATENT_CONTEXT_EXPERT_EXACT_LIFETIME_NBYTES",
    "LATENT_CONTEXT_EXPERT_LIFETIME_COUNTER_NBYTES",
    "LATENT_CONTEXT_EXPERT_RESOURCE_SCHEMA",
    "LATENT_CONTEXT_EXPERT_RESULT_SCHEMA",
    "LATENT_CONTEXT_EXPERT_STATE_SCHEMA",
    "LatentContextExpertConfig",
    "LatentContextExpertDesignRecord",
    "LatentContextExpertLearner",
    "LatentContextExpertLearningResult",
    "LatentContextExpertParams",
    "LatentContextExpertPredictionCache",
    "LatentContextExpertResourceRecord",
    "LatentContextExpertState",
    "LatentContextExpertUpdateResult",
    "latent_context_expert_design_record",
    "latent_context_expert_forward",
    "measure_latent_context_expert_state_nbytes",
    "run_latent_context_expert_arrays",
]
