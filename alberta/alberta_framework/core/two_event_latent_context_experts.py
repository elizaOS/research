# mypy: disable-error-code="call-arg,name-defined"
"""Two-event pairwise-dominance quarantine for latent regression experts.

This sibling preserves the one-sample latent-context learner unchanged while
testing one prespecified structural intervention.  A unique nonactive
challenger opens a quarantine after the current target is observed.  Opening
advances the exact lifetime but commits no parameter subtree.  The immediately
following accepted outcome supplies the second and final relational
observation.  Confirmation uses the fixed 2PDQ law; rejection falls back to
the opening owner.  An ambiguous set of dormant challengers similarly advances
the clock while abstaining from every parameter commit.

Prediction caches bind the complete source state, including every quarantine
field.  Stale caches and dynamically inconsistent pending state therefore fail
closed.  Both routing arms perform the same prediction, evidence, candidate-
gradient, opening, and pending-transition work.  The defaults-off switch changes
only whether a confirmed quarantine commits the candidate or its owner.
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

from alberta_framework.core.pairwise_dominance_quarantine import (
    TWO_EVENT_PAIRWISE_DOMINANCE_HORIZON,
    pairwise_dominance_observation,
    resolve_two_event_pairwise_dominance,
)

TWO_EVENT_LATENT_CONTEXT_EXPERT_DESIGN_SCHEMA = (
    "alberta.two-event-latent-context-expert.design.v1"
)
TWO_EVENT_LATENT_CONTEXT_EXPERT_CONFIG_SCHEMA = (
    "alberta.two-event-latent-context-expert.config.v1"
)
TWO_EVENT_LATENT_CONTEXT_EXPERT_STATE_SCHEMA = (
    "alberta.two-event-latent-context-expert.state.v1"
)
TWO_EVENT_LATENT_CONTEXT_EXPERT_CACHE_SCHEMA = (
    "alberta.two-event-latent-context-expert.cache.v1"
)
TWO_EVENT_LATENT_CONTEXT_EXPERT_RESULT_SCHEMA = (
    "alberta.two-event-latent-context-expert.update-result.v1"
)
TWO_EVENT_LATENT_CONTEXT_EXPERT_RESOURCE_SCHEMA = (
    "alberta.two-event-latent-context-expert.resource-record.v1"
)

TWO_EVENT_LATENT_CONTEXT_EXPERT_EXACT_LIFETIME_NBYTES = 8
TWO_EVENT_LATENT_CONTEXT_EXPERT_LIFETIME_COUNTER_NBYTES = 12

ZERO_COMMIT_REASON_NONE = 0
ZERO_COMMIT_REASON_QUARANTINE_OPENED = 1
ZERO_COMMIT_REASON_AMBIGUOUS_CHALLENGER = 2
ZERO_COMMIT_REASON_TRANSACTION_REJECTED = 3

_INT32_MAX = 2**31 - 1
_UINT32_MAX = 2**32 - 1

_CONFIG_VALUES = {
    "input_dim",
    "output_dim",
    "max_experts",
    "step_size",
    "grad_clip",
    "confirmation_routing_enabled",
}
_CONFIG_FIELDS = _CONFIG_VALUES | {
    "type",
    "schema",
    "design_schema",
    "state_schema",
    "cache_schema",
    "result_schema",
    "resource_schema",
    "confirmation_horizon",
}


@dataclasses.dataclass(frozen=True)
class TwoEventLatentContextExpertDesignRecord:
    """Machine-readable method and causal-scope boundary."""

    schema: str
    method_name: str
    conceptual_novelty_claimed: bool
    prior_mechanisms: tuple[str, ...]
    integration_scope: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-data record."""

        return dataclasses.asdict(self)


def two_event_latent_context_expert_design_record() -> (
    TwoEventLatentContextExpertDesignRecord
):
    """Return the fixed attribution and scope declaration."""

    return TwoEventLatentContextExpertDesignRecord(
        schema=TWO_EVENT_LATENT_CONTEXT_EXPERT_DESIGN_SCHEMA,
        method_name="Two-Event Pairwise-Dominance Quarantined Latent Experts",
        conceptual_novelty_claimed=False,
        prior_mechanisms=(
            "ContextInference active-only-freeze law",
            "leakage-safe latent-context regression experts",
            "two-event pairwise-dominance relational evidence",
        ),
        integration_scope=(
            "fixed H=2 post-outcome challenger quarantine",
            "zero-parameter-commit opening and ambiguous abstention",
            "complete pending-state prediction-cache authentication",
            "same-work defaults-off confirmation-routing ablation",
        ),
    )


@chex.dataclass(frozen=True)
class TwoEventLatentContextExpertConfig:
    """Strict fixed-bank construction; confirmation routing defaults off."""

    input_dim: int
    output_dim: int = 1
    max_experts: int = 2
    step_size: float = 5.0e-2
    grad_clip: float = 10.0
    confirmation_routing_enabled: bool = False

    def to_config(self) -> dict[str, Any]:
        """Serialize the exact versioned construction."""

        return {
            "type": "TwoEventLatentContextExpertConfig",
            "schema": TWO_EVENT_LATENT_CONTEXT_EXPERT_CONFIG_SCHEMA,
            "design_schema": TWO_EVENT_LATENT_CONTEXT_EXPERT_DESIGN_SCHEMA,
            "state_schema": TWO_EVENT_LATENT_CONTEXT_EXPERT_STATE_SCHEMA,
            "cache_schema": TWO_EVENT_LATENT_CONTEXT_EXPERT_CACHE_SCHEMA,
            "result_schema": TWO_EVENT_LATENT_CONTEXT_EXPERT_RESULT_SCHEMA,
            "resource_schema": TWO_EVENT_LATENT_CONTEXT_EXPERT_RESOURCE_SCHEMA,
            "confirmation_horizon": TWO_EVENT_PAIRWISE_DOMINANCE_HORIZON,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "max_experts": self.max_experts,
            "step_size": self.step_size,
            "grad_clip": self.grad_clip,
            "confirmation_routing_enabled": self.confirmation_routing_enabled,
        }

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
    ) -> TwoEventLatentContextExpertConfig:
        """Reconstruct only the exact current schema and fixed horizon."""

        if not isinstance(config, Mapping):
            raise TypeError("two-event latent-context expert config must be a mapping")
        payload = dict(config)
        if set(payload) != _CONFIG_FIELDS:
            missing = sorted(_CONFIG_FIELDS - set(payload))
            extra = sorted(set(payload) - _CONFIG_FIELDS)
            raise ValueError(
                "two-event latent-context expert config fields are invalid; "
                f"missing={missing}, extra={extra}"
            )
        schemas = {
            "type": "TwoEventLatentContextExpertConfig",
            "schema": TWO_EVENT_LATENT_CONTEXT_EXPERT_CONFIG_SCHEMA,
            "design_schema": TWO_EVENT_LATENT_CONTEXT_EXPERT_DESIGN_SCHEMA,
            "state_schema": TWO_EVENT_LATENT_CONTEXT_EXPERT_STATE_SCHEMA,
            "cache_schema": TWO_EVENT_LATENT_CONTEXT_EXPERT_CACHE_SCHEMA,
            "result_schema": TWO_EVENT_LATENT_CONTEXT_EXPERT_RESULT_SCHEMA,
            "resource_schema": TWO_EVENT_LATENT_CONTEXT_EXPERT_RESOURCE_SCHEMA,
            "confirmation_horizon": TWO_EVENT_PAIRWISE_DOMINANCE_HORIZON,
        }
        for name, expected in schemas.items():
            if payload.pop(name) != expected:
                raise ValueError(f"two-event latent-context expert {name} is unsupported")
        restored = cls(**payload)
        _validate_config(restored)
        return restored


@chex.dataclass(frozen=True)
class TwoEventLatentContextExpertParams:
    """Independent fixed-capacity linear expert subtrees."""

    expert_weights: Float[Array, "max_experts input_dim output_dim"]
    expert_biases: Float[Array, "max_experts output_dim"]


@chex.dataclass(frozen=True)
class TwoEventLatentContextExpertState:
    """Persistent experts, exact lifetime, and one-event quarantine."""

    params: TwoEventLatentContextExpertParams
    active_expert: Int[Array, ""]
    step_count: Int[Array, ""]
    step_words: UInt[Array, " 2"]
    pending_valid: Bool[Array, ""]
    pending_owner: Int[Array, ""]
    pending_candidate: Int[Array, ""]
    pending_birth_words: UInt[Array, " 2"]
    pending_never_worse: Bool[Array, " max_experts"]
    pending_ever_strict: Bool[Array, " max_experts"]


@chex.dataclass(frozen=True)
class TwoEventLatentContextExpertPredictionCache:
    """Complete pre-target source-state and prediction snapshot."""

    owner_state: TwoEventLatentContextExpertState
    observation: Float[Array, " input_dim"]
    expert_predictions: Float[Array, "max_experts output_dim"]
    prediction: Float[Array, " output_dim"]
    valid: Bool[Array, ""]


@chex.dataclass(frozen=True)
class TwoEventLatentContextExpertUpdateResult:
    """One atomic target-owned transaction with explicit quarantine trace."""

    state: TwoEventLatentContextExpertState
    prediction: Float[Array, " output_dim"]
    error: Float[Array, " output_dim"]
    expert_predictions: Float[Array, "max_experts output_dim"]
    expert_losses: Float[Array, " max_experts"]
    candidate_gradient_norms: Float[Array, " max_experts"]
    pre_update_owner: Int[Array, ""]
    evidence_best_expert: Int[Array, ""]
    evidence_candidate_expert: Int[Array, ""]
    selected_next_expert: Int[Array, ""]
    expert_update_mask: Bool[Array, " max_experts"]
    parameter_subtree_commit_count: Int[Array, ""]
    context_switched: Bool[Array, ""]
    quarantine_opened: Bool[Array, ""]
    quarantine_second_evidence: Bool[Array, ""]
    quarantine_confirmed: Bool[Array, ""]
    quarantine_rejected: Bool[Array, ""]
    ambiguous_challenger_abstention: Bool[Array, ""]
    zero_commit_reason: Int[Array, ""]
    quarantine_never_worse: Bool[Array, " max_experts"]
    quarantine_ever_strict: Bool[Array, " max_experts"]
    pending_before_valid: Bool[Array, ""]
    pending_after_valid: Bool[Array, ""]
    pending_before_owner: Int[Array, ""]
    pending_before_candidate: Int[Array, ""]
    pending_before_birth_words: UInt[Array, " 2"]
    pending_after_owner: Int[Array, ""]
    pending_after_candidate: Int[Array, ""]
    pending_after_birth_words: UInt[Array, " 2"]
    current_error_relabelled_after_target: Bool[Array, ""]
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
class TwoEventLatentContextExpertLearningResult:
    """Fixed-shape scan trace."""

    state: TwoEventLatentContextExpertState
    predictions: Float[Array, "steps output_dim"]
    errors: Float[Array, "steps output_dim"]
    expert_predictions: Float[Array, "steps max_experts output_dim"]
    expert_losses: Float[Array, "steps max_experts"]
    candidate_gradient_norms: Float[Array, "steps max_experts"]
    pre_update_owner: Int[Array, " steps"]
    evidence_best_expert: Int[Array, " steps"]
    evidence_candidate_expert: Int[Array, " steps"]
    selected_next_expert: Int[Array, " steps"]
    expert_update_mask: Bool[Array, "steps max_experts"]
    parameter_subtree_commit_count: Int[Array, " steps"]
    context_switched: Bool[Array, " steps"]
    quarantine_opened: Bool[Array, " steps"]
    quarantine_second_evidence: Bool[Array, " steps"]
    quarantine_confirmed: Bool[Array, " steps"]
    quarantine_rejected: Bool[Array, " steps"]
    ambiguous_challenger_abstention: Bool[Array, " steps"]
    zero_commit_reason: Int[Array, " steps"]
    quarantine_never_worse: Bool[Array, "steps max_experts"]
    quarantine_ever_strict: Bool[Array, "steps max_experts"]
    pending_before_valid: Bool[Array, " steps"]
    pending_after_valid: Bool[Array, " steps"]
    pending_before_owner: Int[Array, " steps"]
    pending_before_candidate: Int[Array, " steps"]
    pending_before_birth_words: UInt[Array, "steps 2"]
    pending_after_owner: Int[Array, " steps"]
    pending_after_candidate: Int[Array, " steps"]
    pending_after_birth_words: UInt[Array, "steps 2"]
    pre_step_words: UInt[Array, "steps 2"]
    post_step_words: UInt[Array, "steps 2"]
    update_applied: Bool[Array, " steps"]


@dataclasses.dataclass(frozen=True)
class TwoEventLatentContextExpertResourceRecord:
    """Exact persistent/cache bytes and bounded per-event work."""

    schema: str
    design_schema: str
    config_schema: str
    state_schema: str
    cache_schema: str
    result_schema: str
    confirmation_horizon: int
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


def _validate_config(config: TwoEventLatentContextExpertConfig) -> None:
    if not isinstance(config, TwoEventLatentContextExpertConfig):
        raise TypeError("config must be TwoEventLatentContextExpertConfig")
    _require_dimension(config.input_dim, name="input_dim")
    _require_dimension(config.output_dim, name="output_dim")
    _require_dimension(config.max_experts, name="max_experts", minimum=2)
    _require_float(config.step_size, name="step_size")
    _require_float(config.grad_clip, name="grad_clip")
    if type(config.confirmation_routing_enabled) is not bool:
        raise ValueError("confirmation_routing_enabled must be an exact bool")


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


def two_event_latent_context_expert_forward(
    params: TwoEventLatentContextExpertParams,
    observation: Float[Array, " input_dim"],
) -> Float[Array, "max_experts output_dim"]:
    """Return every expert prediction without selecting an owner."""

    return jnp.einsum("i,kio->ko", observation, params.expert_weights) + params.expert_biases


class TwoEventLatentContextExpertLearner:
    """Fixed-bank causal expert learner with a fixed H=2 quarantine."""

    def __init__(self, config: TwoEventLatentContextExpertConfig):
        _validate_config(config)
        self._config = config

    @property
    def config(self) -> TwoEventLatentContextExpertConfig:
        """Return the immutable construction."""

        return self._config

    @property
    def design_record(self) -> TwoEventLatentContextExpertDesignRecord:
        """Return the method and causal-scope declaration."""

        return two_event_latent_context_expert_design_record()

    def to_config(self) -> dict[str, Any]:
        """Serialize a strict learner wrapper."""

        return {
            "type": "TwoEventLatentContextExpertLearner",
            "schema": TWO_EVENT_LATENT_CONTEXT_EXPERT_CONFIG_SCHEMA,
            "design_schema": TWO_EVENT_LATENT_CONTEXT_EXPERT_DESIGN_SCHEMA,
            "state_schema": TWO_EVENT_LATENT_CONTEXT_EXPERT_STATE_SCHEMA,
            "cache_schema": TWO_EVENT_LATENT_CONTEXT_EXPERT_CACHE_SCHEMA,
            "result_schema": TWO_EVENT_LATENT_CONTEXT_EXPERT_RESULT_SCHEMA,
            "resource_schema": TWO_EVENT_LATENT_CONTEXT_EXPERT_RESOURCE_SCHEMA,
            "confirmation_horizon": TWO_EVENT_PAIRWISE_DOMINANCE_HORIZON,
            "config": self._config.to_config(),
        }

    def init(self) -> TwoEventLatentContextExpertState:
        """Return identical neutral experts and exact empty quarantine state."""

        c = self._config
        return TwoEventLatentContextExpertState(
            params=TwoEventLatentContextExpertParams(
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
            pending_valid=jnp.asarray(False, dtype=jnp.bool_),
            pending_owner=jnp.asarray(0, dtype=jnp.int32),
            pending_candidate=jnp.asarray(0, dtype=jnp.int32),
            pending_birth_words=jnp.zeros((2,), dtype=jnp.uint32),
            pending_never_worse=jnp.zeros((c.max_experts,), dtype=jnp.bool_),
            pending_ever_strict=jnp.zeros((c.max_experts,), dtype=jnp.bool_),
        )

    def _require_params_contract(self, params: TwoEventLatentContextExpertParams) -> None:
        if not isinstance(params, TwoEventLatentContextExpertParams):
            raise TypeError("params must be TwoEventLatentContextExpertParams")
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

    def _require_state_contract(self, state: TwoEventLatentContextExpertState) -> None:
        if not isinstance(state, TwoEventLatentContextExpertState):
            raise TypeError("state must be TwoEventLatentContextExpertState")
        self._require_params_contract(state.params)
        c = self._config
        _require_array(state.active_expert, name="active_expert", shape=(), dtype=jnp.int32)
        _require_array(state.step_count, name="step_count", shape=(), dtype=jnp.int32)
        _require_array(state.step_words, name="step_words", shape=(2,), dtype=jnp.uint32)
        _require_array(state.pending_valid, name="pending_valid", shape=(), dtype=jnp.bool_)
        _require_array(state.pending_owner, name="pending_owner", shape=(), dtype=jnp.int32)
        _require_array(
            state.pending_candidate,
            name="pending_candidate",
            shape=(),
            dtype=jnp.int32,
        )
        _require_array(
            state.pending_birth_words,
            name="pending_birth_words",
            shape=(2,),
            dtype=jnp.uint32,
        )
        _require_array(
            state.pending_never_worse,
            name="pending_never_worse",
            shape=(c.max_experts,),
            dtype=jnp.bool_,
        )
        _require_array(
            state.pending_ever_strict,
            name="pending_ever_strict",
            shape=(c.max_experts,),
            dtype=jnp.bool_,
        )

    def _require_cache_contract(
        self,
        cache: TwoEventLatentContextExpertPredictionCache,
    ) -> None:
        if not isinstance(cache, TwoEventLatentContextExpertPredictionCache):
            raise TypeError("cache must be TwoEventLatentContextExpertPredictionCache")
        self._require_state_contract(cache.owner_state)
        c = self._config
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

    def _pending_valid(self, state: TwoEventLatentContextExpertState) -> Array:
        c = self._config
        zero_words = jnp.zeros((2,), dtype=jnp.uint32)
        zero_masks = jnp.zeros((c.max_experts,), dtype=jnp.bool_)
        empty_payload = (
            (state.pending_owner == 0)
            & (state.pending_candidate == 0)
            & jnp.array_equal(state.pending_birth_words, zero_words)
            & jnp.array_equal(state.pending_never_worse, zero_masks)
            & jnp.array_equal(state.pending_ever_strict, zero_masks)
        )
        owner_valid = (state.pending_owner >= 0) & (state.pending_owner < c.max_experts)
        candidate_valid = (state.pending_candidate >= 0) & (
            state.pending_candidate < c.max_experts
        )
        safe_candidate = jnp.clip(state.pending_candidate, 0, c.max_experts - 1)
        expected_comparators = (
            jnp.arange(c.max_experts, dtype=jnp.int32) != safe_candidate
        )
        live_payload = (
            owner_valid
            & candidate_valid
            & (state.pending_owner != state.pending_candidate)
            & (state.active_expert == state.pending_owner)
            & jnp.array_equal(state.pending_birth_words, state.step_words)
            & jnp.any(state.pending_birth_words != zero_words)
            & jnp.array_equal(state.pending_never_worse, expected_comparators)
            & jnp.all((~state.pending_ever_strict) | expected_comparators)
            & jnp.all((~state.pending_ever_strict) | state.pending_never_worse)
        )
        return jnp.where(state.pending_valid, live_payload, empty_payload)

    def state_valid(self, state: TwoEventLatentContextExpertState) -> Array:
        """Return dynamic validity after enforcing exact shapes and dtypes."""

        self._require_state_contract(state)
        return (
            _tree_finite(state.params)
            & (state.active_expert >= 0)
            & (state.active_expert < self._config.max_experts)
            & _counter_valid(state.step_words, state.step_count)
            & self._pending_valid(state)
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def predict(
        self,
        state: TwoEventLatentContextExpertState,
        observation: Float[Array, " input_dim"],
    ) -> TwoEventLatentContextExpertPredictionCache:
        """Bind the complete current state before any target is supplied."""

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
        raw_predictions = two_event_latent_context_expert_forward(
            state.params,
            safe_observation,
        )
        predictions_valid = jnp.all(jnp.isfinite(raw_predictions))
        valid = state_valid & input_valid & predictions_valid
        safe_active = jnp.clip(state.active_expert, 0, self._config.max_experts - 1)
        expert_predictions = jnp.where(valid, raw_predictions, jnp.zeros_like(raw_predictions))
        prediction = jnp.where(
            valid,
            raw_predictions[safe_active],
            jnp.zeros((self._config.output_dim,), dtype=jnp.float32),
        )
        return TwoEventLatentContextExpertPredictionCache(
            owner_state=state,
            observation=safe_observation,
            expert_predictions=expert_predictions,
            prediction=prediction,
            valid=valid,
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def update(
        self,
        state: TwoEventLatentContextExpertState,
        cache: TwoEventLatentContextExpertPredictionCache,
        target: Float[Array, " output_dim"],
    ) -> TwoEventLatentContextExpertUpdateResult:
        """Resolve one causal target transaction and at most one subtree commit."""

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
        cache_owner_valid = cache.valid & _tree_exact(cache.owner_state, state)
        safe_active = jnp.clip(state.active_expert, 0, c.max_experts - 1)
        recomputed_predictions = two_event_latent_context_expert_forward(
            state.params,
            cache.observation,
        )
        cache_prediction_exact = (
            jnp.array_equal(cache.expert_predictions, recomputed_predictions)
            & jnp.array_equal(cache.prediction, recomputed_predictions[safe_active])
        )
        target_valid = jnp.all(jnp.isfinite(checked_target))
        safe_target = jnp.where(target_valid, checked_target, jnp.zeros_like(checked_target))

        expert_errors = safe_target[None, :] - cache.expert_predictions
        expert_losses = jnp.mean(jnp.square(expert_errors), axis=1)
        evidence_valid = jnp.all(jnp.isfinite(expert_losses))
        safe_losses = jnp.where(jnp.isfinite(expert_losses), expert_losses, jnp.inf)
        raw_best = jnp.argmin(safe_losses).astype(jnp.int32)
        minimum_loss = safe_losses[raw_best]
        best_mask = safe_losses == minimum_loss
        active_tied_for_best = best_mask[safe_active]
        evidence_best = jnp.where(active_tied_for_best, safe_active, raw_best).astype(jnp.int32)
        dormant_mask = jnp.arange(c.max_experts, dtype=jnp.int32) != safe_active
        dormant_losses = jnp.where(dormant_mask, safe_losses, jnp.inf)
        dormant_best = jnp.argmin(dormant_losses).astype(jnp.int32)
        dormant_minimum = dormant_losses[dormant_best]
        dormant_best_mask = dormant_mask & (dormant_losses == dormant_minimum)
        dormant_best_count = jnp.sum(dormant_best_mask.astype(jnp.int32))
        dormant_is_globally_no_worse = dormant_minimum <= safe_losses[safe_active]
        unique_challenger = (dormant_best_count == 1) & dormant_is_globally_no_worse
        ambiguous_challenger = (dormant_best_count > 1) & dormant_is_globally_no_worse

        # Analytic MSE candidates are evaluated for every expert in both arms,
        # including zero-commit opening and abstention transactions.
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
        candidate_gradients_valid = (
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
        candidate_weights = state.params.expert_weights - jnp.asarray(
            c.step_size,
            dtype=jnp.float32,
        ) * (scales[:, None, None] * weight_gradients)
        candidate_biases = state.params.expert_biases - jnp.asarray(
            c.step_size,
            dtype=jnp.float32,
        ) * (scales[:, None] * bias_gradients)
        all_candidate_params_valid = _tree_finite(
            TwoEventLatentContextExpertParams(
                expert_weights=candidate_weights,
                expert_biases=candidate_biases,
            )
        )

        opening_evidence = pairwise_dominance_observation(safe_losses, dormant_best)
        safe_pending_candidate = jnp.clip(
            state.pending_candidate,
            0,
            c.max_experts - 1,
        )
        resolution = resolve_two_event_pairwise_dominance(
            state.pending_never_worse,
            state.pending_never_worse,
            state.pending_ever_strict,
            safe_losses,
            safe_pending_candidate,
        )
        no_pending = ~state.pending_valid
        quarantine_opened = no_pending & unique_challenger
        quarantine_second_evidence = state.pending_valid
        quarantine_confirmed = quarantine_second_evidence & resolution.confirmed
        quarantine_rejected = quarantine_second_evidence & resolution.rejected
        ambiguous_abstention = no_pending & ambiguous_challenger
        normal_active_update = no_pending & (~quarantine_opened) & (~ambiguous_abstention)
        pending_evidence_valid = jnp.where(
            quarantine_second_evidence,
            resolution.valid,
            jnp.where(quarantine_opened, opening_evidence.valid, True),
        )

        routed_resolution = jnp.where(
            quarantine_confirmed & c.confirmation_routing_enabled,
            safe_pending_candidate,
            jnp.clip(state.pending_owner, 0, c.max_experts - 1),
        ).astype(jnp.int32)
        selected = jnp.where(
            quarantine_second_evidence,
            routed_resolution,
            safe_active,
        ).astype(jnp.int32)
        commit_requested = quarantine_second_evidence | normal_active_update
        selected_mask = (
            jnp.arange(c.max_experts, dtype=jnp.int32) == selected
        ) & commit_requested
        candidate_params = TwoEventLatentContextExpertParams(
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
        zero_words = jnp.zeros((2,), dtype=jnp.uint32)
        zero_masks = jnp.zeros((c.max_experts,), dtype=jnp.bool_)
        candidate_state = TwoEventLatentContextExpertState(
            params=candidate_params,
            active_expert=selected,
            step_count=_words_to_telemetry(proposed_words),
            step_words=proposed_words,
            pending_valid=quarantine_opened,
            pending_owner=jnp.where(
                quarantine_opened,
                safe_active,
                jnp.asarray(0, dtype=jnp.int32),
            ),
            pending_candidate=jnp.where(
                quarantine_opened,
                dormant_best,
                jnp.asarray(0, dtype=jnp.int32),
            ),
            pending_birth_words=jnp.where(
                quarantine_opened,
                proposed_words,
                zero_words,
            ),
            pending_never_worse=jnp.where(
                quarantine_opened,
                opening_evidence.never_worse,
                zero_masks,
            ),
            pending_ever_strict=jnp.where(
                quarantine_opened,
                opening_evidence.ever_strict,
                zero_masks,
            ),
        )
        candidate_state_valid = self.state_valid(candidate_state) & all_candidate_params_valid
        update_applied = (
            source_state_valid
            & cache_owner_valid
            & cache_input_valid
            & cache_prediction_exact
            & target_valid
            & evidence_valid
            & pending_evidence_valid
            & candidate_gradients_valid
            & candidate_state_valid
            & lifetime_capacity_available
        )
        committed_state = cast(
            TwoEventLatentContextExpertState,
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
        applied_mask = update_applied & selected_mask
        commit_count = jnp.sum(applied_mask.astype(jnp.int32))
        raw_zero_reason = jnp.where(
            quarantine_opened,
            jnp.asarray(ZERO_COMMIT_REASON_QUARANTINE_OPENED, dtype=jnp.int32),
            jnp.where(
                ambiguous_abstention,
                jnp.asarray(ZERO_COMMIT_REASON_AMBIGUOUS_CHALLENGER, dtype=jnp.int32),
                jnp.asarray(ZERO_COMMIT_REASON_NONE, dtype=jnp.int32),
            ),
        )
        zero_reason = jnp.where(
            update_applied,
            raw_zero_reason,
            jnp.asarray(ZERO_COMMIT_REASON_TRANSACTION_REJECTED, dtype=jnp.int32),
        )
        evidence_candidate = jnp.where(
            quarantine_second_evidence,
            safe_pending_candidate,
            dormant_best,
        )
        relation_never = jnp.where(
            quarantine_second_evidence,
            resolution.never_worse,
            jnp.where(quarantine_opened, opening_evidence.never_worse, zero_masks),
        )
        relation_strict = jnp.where(
            quarantine_second_evidence,
            resolution.ever_strict,
            jnp.where(quarantine_opened, opening_evidence.ever_strict, zero_masks),
        )
        return TwoEventLatentContextExpertUpdateResult(
            state=committed_state,
            prediction=jnp.where(update_applied, cache.prediction, zero_output),
            error=jnp.where(update_applied, safe_target - cache.prediction, zero_output),
            expert_predictions=jnp.where(
                update_applied,
                cache.expert_predictions,
                zero_expert_outputs,
            ),
            expert_losses=jnp.where(update_applied, expert_losses, zero_expert_scalars),
            candidate_gradient_norms=jnp.where(
                update_applied,
                gradient_norms,
                zero_expert_scalars,
            ),
            pre_update_owner=jnp.where(
                update_applied,
                cache.owner_state.active_expert,
                no_expert,
            ),
            evidence_best_expert=jnp.where(update_applied, evidence_best, no_expert),
            evidence_candidate_expert=jnp.where(
                update_applied,
                evidence_candidate,
                no_expert,
            ),
            selected_next_expert=jnp.where(update_applied, selected, no_expert),
            expert_update_mask=applied_mask,
            parameter_subtree_commit_count=commit_count,
            context_switched=update_applied
            & (cache.owner_state.active_expert != selected),
            quarantine_opened=update_applied & quarantine_opened,
            quarantine_second_evidence=update_applied & quarantine_second_evidence,
            quarantine_confirmed=update_applied & quarantine_confirmed,
            quarantine_rejected=update_applied & quarantine_rejected,
            ambiguous_challenger_abstention=update_applied & ambiguous_abstention,
            zero_commit_reason=zero_reason,
            quarantine_never_worse=jnp.where(update_applied, relation_never, zero_masks),
            quarantine_ever_strict=jnp.where(update_applied, relation_strict, zero_masks),
            pending_before_valid=update_applied & state.pending_valid,
            pending_after_valid=update_applied & committed_state.pending_valid,
            pending_before_owner=jnp.where(
                update_applied & state.pending_valid,
                state.pending_owner,
                no_expert,
            ),
            pending_before_candidate=jnp.where(
                update_applied & state.pending_valid,
                state.pending_candidate,
                no_expert,
            ),
            pending_before_birth_words=jnp.where(
                update_applied & state.pending_valid,
                state.pending_birth_words,
                zero_words,
            ),
            pending_after_owner=jnp.where(
                update_applied & committed_state.pending_valid,
                committed_state.pending_owner,
                no_expert,
            ),
            pending_after_candidate=jnp.where(
                update_applied & committed_state.pending_valid,
                committed_state.pending_candidate,
                no_expert,
            ),
            pending_after_birth_words=jnp.where(
                update_applied & committed_state.pending_valid,
                committed_state.pending_birth_words,
                zero_words,
            ),
            current_error_relabelled_after_target=jnp.asarray(False, dtype=jnp.bool_),
            pre_step_words=state.step_words,
            post_step_words=committed_state.step_words,
            lifetime_counter_valid=lifetime_counter_valid,
            lifetime_capacity_available=lifetime_capacity_available,
            source_state_valid=source_state_valid,
            cache_owner_valid=cache_owner_valid,
            cache_input_valid=cache_input_valid,
            cache_prediction_exact=cache_prediction_exact,
            target_valid=target_valid,
            evidence_valid=evidence_valid & pending_evidence_valid,
            candidate_gradients_valid=candidate_gradients_valid,
            candidate_state_valid=candidate_state_valid,
            update_applied=update_applied,
        )

    def resource_record(
        self,
        state: TwoEventLatentContextExpertState | None = None,
    ) -> TwoEventLatentContextExpertResourceRecord:
        """Measure the full state/cache and report bounded transaction work."""

        measured = self.init() if state is None else state
        self._require_state_contract(measured)
        if not bool(jax.device_get(self.state_valid(measured))):
            raise ValueError("cannot account an invalid two-event latent-context state")
        parameter_nbytes = _tree_nbytes(measured.params)
        state_nbytes = _tree_nbytes(measured)
        cache = self.predict(
            measured,
            jnp.zeros((self._config.input_dim,), dtype=jnp.float32),
        )
        k = self._config.max_experts
        return TwoEventLatentContextExpertResourceRecord(
            schema=TWO_EVENT_LATENT_CONTEXT_EXPERT_RESOURCE_SCHEMA,
            design_schema=TWO_EVENT_LATENT_CONTEXT_EXPERT_DESIGN_SCHEMA,
            config_schema=TWO_EVENT_LATENT_CONTEXT_EXPERT_CONFIG_SCHEMA,
            state_schema=TWO_EVENT_LATENT_CONTEXT_EXPERT_STATE_SCHEMA,
            cache_schema=TWO_EVENT_LATENT_CONTEXT_EXPERT_CACHE_SCHEMA,
            result_schema=TWO_EVENT_LATENT_CONTEXT_EXPERT_RESULT_SCHEMA,
            confirmation_horizon=TWO_EVENT_PAIRWISE_DOMINANCE_HORIZON,
            input_dim=self._config.input_dim,
            output_dim=self._config.output_dim,
            max_experts=k,
            parameter_nbytes=parameter_nbytes,
            exact_lifetime_identity_nbytes=(
                TWO_EVENT_LATENT_CONTEXT_EXPERT_EXACT_LIFETIME_NBYTES
            ),
            lifetime_counter_nbytes=(
                TWO_EVENT_LATENT_CONTEXT_EXPERT_LIFETIME_COUNTER_NBYTES
            ),
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


def measure_two_event_latent_context_expert_state_nbytes(
    state: TwoEventLatentContextExpertState,
) -> int:
    """Measure every persistent array byte."""

    if not isinstance(state, TwoEventLatentContextExpertState):
        raise TypeError("state must be TwoEventLatentContextExpertState")
    return _tree_nbytes(state)


def run_two_event_latent_context_expert_arrays(
    learner: TwoEventLatentContextExpertLearner,
    observations: Float[Array, "steps input_dim"],
    targets: Float[Array, "steps output_dim"],
    *,
    state: TwoEventLatentContextExpertState | None = None,
) -> TwoEventLatentContextExpertLearningResult:
    """Run a stream while constructing every cache before its target update."""

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
        carry: TwoEventLatentContextExpertState,
        inputs: tuple[Array, Array],
    ) -> tuple[TwoEventLatentContextExpertState, tuple[Array, ...]]:
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
            result.evidence_candidate_expert,
            result.selected_next_expert,
            result.expert_update_mask,
            result.parameter_subtree_commit_count,
            result.context_switched,
            result.quarantine_opened,
            result.quarantine_second_evidence,
            result.quarantine_confirmed,
            result.quarantine_rejected,
            result.ambiguous_challenger_abstention,
            result.zero_commit_reason,
            result.quarantine_never_worse,
            result.quarantine_ever_strict,
            result.pending_before_valid,
            result.pending_after_valid,
            result.pending_before_owner,
            result.pending_before_candidate,
            result.pending_before_birth_words,
            result.pending_after_owner,
            result.pending_after_candidate,
            result.pending_after_birth_words,
            result.pre_step_words,
            result.post_step_words,
            result.update_applied,
        )

    final_state, outputs = jax.lax.scan(step, initial, (observation_array, target_array))
    (
        predictions,
        errors,
        expert_predictions,
        expert_losses,
        candidate_gradient_norms,
        pre_update_owner,
        evidence_best_expert,
        evidence_candidate_expert,
        selected_next_expert,
        expert_update_mask,
        parameter_subtree_commit_count,
        context_switched,
        quarantine_opened,
        quarantine_second_evidence,
        quarantine_confirmed,
        quarantine_rejected,
        ambiguous_challenger_abstention,
        zero_commit_reason,
        quarantine_never_worse,
        quarantine_ever_strict,
        pending_before_valid,
        pending_after_valid,
        pending_before_owner,
        pending_before_candidate,
        pending_before_birth_words,
        pending_after_owner,
        pending_after_candidate,
        pending_after_birth_words,
        pre_step_words,
        post_step_words,
        update_applied,
    ) = outputs
    return TwoEventLatentContextExpertLearningResult(
        state=final_state,
        predictions=predictions,
        errors=errors,
        expert_predictions=expert_predictions,
        expert_losses=expert_losses,
        candidate_gradient_norms=candidate_gradient_norms,
        pre_update_owner=pre_update_owner,
        evidence_best_expert=evidence_best_expert,
        evidence_candidate_expert=evidence_candidate_expert,
        selected_next_expert=selected_next_expert,
        expert_update_mask=expert_update_mask,
        parameter_subtree_commit_count=parameter_subtree_commit_count,
        context_switched=context_switched,
        quarantine_opened=quarantine_opened,
        quarantine_second_evidence=quarantine_second_evidence,
        quarantine_confirmed=quarantine_confirmed,
        quarantine_rejected=quarantine_rejected,
        ambiguous_challenger_abstention=ambiguous_challenger_abstention,
        zero_commit_reason=zero_commit_reason,
        quarantine_never_worse=quarantine_never_worse,
        quarantine_ever_strict=quarantine_ever_strict,
        pending_before_valid=pending_before_valid,
        pending_after_valid=pending_after_valid,
        pending_before_owner=pending_before_owner,
        pending_before_candidate=pending_before_candidate,
        pending_before_birth_words=pending_before_birth_words,
        pending_after_owner=pending_after_owner,
        pending_after_candidate=pending_after_candidate,
        pending_after_birth_words=pending_after_birth_words,
        pre_step_words=pre_step_words,
        post_step_words=post_step_words,
        update_applied=update_applied,
    )


__all__ = [
    "TWO_EVENT_LATENT_CONTEXT_EXPERT_CACHE_SCHEMA",
    "TWO_EVENT_LATENT_CONTEXT_EXPERT_CONFIG_SCHEMA",
    "TWO_EVENT_LATENT_CONTEXT_EXPERT_DESIGN_SCHEMA",
    "TWO_EVENT_LATENT_CONTEXT_EXPERT_EXACT_LIFETIME_NBYTES",
    "TWO_EVENT_LATENT_CONTEXT_EXPERT_LIFETIME_COUNTER_NBYTES",
    "TWO_EVENT_LATENT_CONTEXT_EXPERT_RESOURCE_SCHEMA",
    "TWO_EVENT_LATENT_CONTEXT_EXPERT_RESULT_SCHEMA",
    "TWO_EVENT_LATENT_CONTEXT_EXPERT_STATE_SCHEMA",
    "ZERO_COMMIT_REASON_AMBIGUOUS_CHALLENGER",
    "ZERO_COMMIT_REASON_NONE",
    "ZERO_COMMIT_REASON_QUARANTINE_OPENED",
    "ZERO_COMMIT_REASON_TRANSACTION_REJECTED",
    "TwoEventLatentContextExpertConfig",
    "TwoEventLatentContextExpertDesignRecord",
    "TwoEventLatentContextExpertLearner",
    "TwoEventLatentContextExpertLearningResult",
    "TwoEventLatentContextExpertParams",
    "TwoEventLatentContextExpertPredictionCache",
    "TwoEventLatentContextExpertResourceRecord",
    "TwoEventLatentContextExpertState",
    "TwoEventLatentContextExpertUpdateResult",
    "measure_two_event_latent_context_expert_state_nbytes",
    "run_two_event_latent_context_expert_arrays",
    "two_event_latent_context_expert_design_record",
    "two_event_latent_context_expert_forward",
]
