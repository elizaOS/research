# mypy: disable-error-code="attr-defined,call-arg,type-var"
"""Causal online scores and a shielded stochastic-trap development world.

This module supplies the mechanism missing from the narrow WP5.6 selector:
expected improvement, ensemble disagreement, information gain, and learning
progress are computed from an online estimator's own observation/transition
history.  No environment value, latent progress target, counterfactual reward,
or evaluator label is an estimator input.

The small continuing world has four primitive actions: stabilize, watch a
high-variance noisy-TV channel, invest in a delayed opportunity, and collect
the opportunity once it is unlocked.  The TV channel is exogenous and
unpredictable; investing has an immediate cost and collection requires several
prior investments.  These are mechanism semantics, not an efficacy claim.

Candidate ranking and execution remain separate.  A caller-owned hard shield
consumes an owned ranking receipt and produces an owned executable-action
receipt.  The environment accepts only that receipt.  Every transition then
updates the estimator at the exact pre-update revision named by the executed
decision.  Stale, aliased, nonfinite, exhausted, or otherwise invalid inputs
are atomic no-ops.

The code is L0 and permanently ``not_assessed``.  It has no physical dispatch,
deployment, evidence, output-writing, or promotion authority.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array
from jaxtyping import Bool, Float, Int, UInt

CAUSAL_EXPLORATION_EVIDENCE_LEVEL = "L0"
CAUSAL_EXPLORATION_ASSESSMENT_STATUS = "not_assessed"
CAUSAL_EXPLORATION_OUTPUT_WRITE_AUTHORITY = False
CAUSAL_EXPLORATION_PHYSICAL_DISPATCH_AUTHORITY = False
CAUSAL_EXPLORATION_DEPLOYMENT_AUTHORITY = False
CAUSAL_EXPLORATION_PROMOTION_AUTHORITY = False
CAUSAL_EXPLORATION_SCIENTIFIC_PROMOTION_ALLOWED = False
CAUSAL_EXPLORATION_LIFETIME_SEMANTICS = "exact-uint64-fail-stop"

STABILIZE_ACTION = 0
NOISY_TV_ACTION = 1
INVEST_ACTION = 2
COLLECT_ACTION = 3
N_EXPLORATION_ACTIONS = 4
EXPLORATION_OBSERVATION_DIM = 5

_DIGEST_WORDS = 8
_UINT32_MAX = 2**32 - 1
_FLOAT32_TINY = float(np.finfo(np.float32).tiny)


def _require_exact_digest(value: object, *, label: str) -> tuple[int, ...]:
    if type(value) is not tuple or len(value) != _DIGEST_WORDS:
        raise TypeError(f"{label} must be an exact {_DIGEST_WORDS}-word tuple")
    words: list[int] = []
    for index, item in enumerate(value):
        if type(item) is not int or not 0 <= item <= _UINT32_MAX:
            raise ValueError(f"{label}[{index}] must be uint32-compatible")
        words.append(item)
    if not any(words):
        raise ValueError(f"{label} must be nonzero")
    return tuple(words)


def _require_float(
    value: object,
    *,
    label: str,
    minimum: float,
    maximum: float | None = None,
    strict_minimum: bool = False,
) -> float:
    if type(value) is not float:
        raise TypeError(f"{label} must be an exact Python float")
    represented = float(np.float32(value))
    below = value <= minimum if strict_minimum else value < minimum
    if (
        not math.isfinite(value)
        or not math.isfinite(represented)
        or below
        or (maximum is not None and value > maximum)
    ):
        raise ValueError(f"{label} is outside its finite float32 range")
    return value


def _require_array(
    value: Any,
    *,
    label: str,
    shape: tuple[int, ...],
    dtype: jnp.dtype,
) -> Array:
    if getattr(value, "shape", None) != shape:
        raise ValueError(f"{label} must have shape {shape}")
    if getattr(value, "dtype", None) != dtype:
        raise TypeError(f"{label} must have dtype {dtype}")
    return jnp.asarray(value)


def _require_key(value: Any, *, label: str) -> None:
    try:
        data = jr.key_data(value)
        implementation = str(jr.key_impl(value))
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{label} must be one typed Threefry key") from exc
    if (
        getattr(value, "shape", None) != ()
        or data.shape != (2,)
        or data.dtype != jnp.dtype(jnp.uint32)
        or implementation != "threefry2x32"
    ):
        raise TypeError(f"{label} must be one typed Threefry key")


def _words_equal(left: Array, right: Array) -> Bool[Array, ""]:
    return jnp.all(left == right)


def _words_greater(left: Array, right: Array) -> Bool[Array, ""]:
    return (left[0] > right[0]) | ((left[0] == right[0]) & (left[1] > right[1]))


def _increment_words(words: Array) -> tuple[UInt[Array, " 2"], Bool[Array, ""]]:
    maximum = jnp.asarray(_UINT32_MAX, dtype=jnp.uint32)
    carry = words[1] == maximum
    capacity = ~(carry & (words[0] == maximum))
    return (
        jnp.stack(
            (words[0] + carry.astype(jnp.uint32), words[1] + jnp.uint32(1)),
            dtype=jnp.uint32,
        ),
        capacity,
    )


def _is_successor(previous: Array, candidate: Array) -> Bool[Array, ""]:
    expected, capacity = _increment_words(previous)
    return capacity & _words_equal(candidate, expected)


def _array_nbytes(value: Array) -> int:
    return int(value.size) * int(value.dtype.itemsize)


def _tree_nbytes(value: object) -> int:
    return sum(_array_nbytes(leaf) for leaf in jax.tree.leaves(value))


def _sum_uint32_as_words(values: Array) -> UInt[Array, " 2"]:
    """Accumulate a short uint32 vector into exact high/low words."""

    high = jnp.asarray(0, dtype=jnp.uint32)
    low = jnp.asarray(0, dtype=jnp.uint32)
    for value in values:
        candidate_low = low + value
        carry = candidate_low < low
        high = high + carry.astype(jnp.uint32)
        low = candidate_low
    return jnp.stack((high, low), dtype=jnp.uint32)


@dataclasses.dataclass(frozen=True, slots=True)
class CausalExplorationEstimatorConfig:
    """Static online TD-ensemble construction and exact ownership."""

    ensemble_size: int
    discount: float
    step_size: float
    prior_scale: float
    fast_error_rate: float
    slow_error_rate: float
    weight_clip: float
    metric_cap: float
    estimator_owner_digest: tuple[int, ...]
    action_owner_digest: tuple[int, ...]
    decision_owner_digest: tuple[int, ...]
    environment_owner_digest: tuple[int, ...]

    def __post_init__(self) -> None:
        if type(self.ensemble_size) is not int or not 2 <= self.ensemble_size <= 16:
            raise ValueError("ensemble_size must be an exact integer in [2, 16]")
        _require_float(self.discount, label="discount", minimum=0.0, maximum=1.0)
        _require_float(
            self.step_size,
            label="step_size",
            minimum=0.0,
            maximum=1.0,
            strict_minimum=True,
        )
        _require_float(
            self.prior_scale,
            label="prior_scale",
            minimum=0.0,
            maximum=1.0,
            strict_minimum=True,
        )
        fast = _require_float(
            self.fast_error_rate,
            label="fast_error_rate",
            minimum=0.0,
            maximum=1.0,
            strict_minimum=True,
        )
        slow = _require_float(
            self.slow_error_rate,
            label="slow_error_rate",
            minimum=0.0,
            maximum=1.0,
            strict_minimum=True,
        )
        if fast <= slow:
            raise ValueError("fast_error_rate must exceed slow_error_rate")
        _require_float(
            self.weight_clip,
            label="weight_clip",
            minimum=0.0,
            strict_minimum=True,
        )
        _require_float(
            self.metric_cap,
            label="metric_cap",
            minimum=0.0,
            strict_minimum=True,
        )
        owners = (
            _require_exact_digest(
                self.estimator_owner_digest,
                label="estimator_owner_digest",
            ),
            _require_exact_digest(self.action_owner_digest, label="action_owner_digest"),
            _require_exact_digest(
                self.decision_owner_digest,
                label="decision_owner_digest",
            ),
            _require_exact_digest(
                self.environment_owner_digest,
                label="environment_owner_digest",
            ),
        )
        if len(set(owners)) != len(owners):
            raise ValueError("estimator, action, decision, and environment owners must differ")

    @property
    def feature_dim(self) -> int:
        return EXPLORATION_OBSERVATION_DIM + 1


@chex.dataclass(frozen=True)
class CausalExplorationEstimatorState:
    """Frozen ensemble, causal receipts, exact revision, and typed RNG."""

    weights: Float[Array, "ensemble action feature"]
    action_counts: UInt[Array, " action"]
    action_precision: Float[Array, " action"]
    fast_absolute_td_error: Float[Array, " action"]
    slow_absolute_td_error: Float[Array, " action"]
    revision_words: UInt[Array, " 2"]
    last_source_event_words: UInt[Array, " 2"]
    last_decision_words: UInt[Array, " 2"]
    rng_key: Array
    estimator_owner_digest: UInt[Array, " digest"]


@chex.dataclass(frozen=True)
class CausalCandidateEstimates:
    """Pre-decision scores derived only from current estimator state and observation."""

    candidate_actions: Int[Array, " action"]
    host_policy: Float[Array, " action"]
    host_action: Int[Array, ""]
    expected_improvement: Float[Array, " action"]
    ensemble_disagreement: Float[Array, " action"]
    information_gain: Float[Array, " action"]
    learning_progress: Float[Array, " action"]
    estimator_revision_words: UInt[Array, " 2"]
    source_event_words: UInt[Array, " 2"]
    estimator_owner_digest: UInt[Array, " digest"]
    causal_online_estimate: Bool[Array, ""]
    oracle_input_used: Bool[Array, ""]


@chex.dataclass(frozen=True)
class ExecutedExplorationTransition:
    """Observed transition and exact pre-update action/decision receipts."""

    observation: Float[Array, " observation"]
    action: Int[Array, ""]
    reward: Float[Array, ""]
    next_observation: Float[Array, " observation"]
    source_event_words: UInt[Array, " 2"]
    decision_words: UInt[Array, " 2"]
    estimator_revision_words: UInt[Array, " 2"]
    estimator_owner_digest: UInt[Array, " digest"]
    action_owner_digest: UInt[Array, " digest"]
    decision_owner_digest: UInt[Array, " digest"]
    environment_owner_digest: UInt[Array, " digest"]


@chex.dataclass(frozen=True)
class CausalEstimatorUpdateResult:
    """Atomic update result and reconstructing TD diagnostics."""

    state: CausalExplorationEstimatorState
    applied: Bool[Array, ""]
    input_valid: Bool[Array, ""]
    ownership_valid: Bool[Array, ""]
    causal_revision_valid: Bool[Array, ""]
    lifetime_capacity_available: Bool[Array, ""]
    mean_td_error: Float[Array, ""]
    mean_absolute_td_error: Float[Array, ""]
    pre_revision_words: UInt[Array, " 2"]
    post_revision_words: UInt[Array, " 2"]


@dataclasses.dataclass(frozen=True, slots=True)
class CausalEstimatorResourceBudget:
    """Exact persistent bytes and fixed logical estimator work."""

    persistent_state_nbytes: int
    initialization_normal_draws: int
    candidate_scores_per_decision: int
    ensemble_predictions_per_decision: int
    observed_updates_per_executed_transition: int
    update_parameter_opportunities_per_transition: int
    temporary_scope: str


class CausalExplorationEstimator:
    """Small online linear TD ensemble with causal novelty/progress scores."""

    def __init__(self, config: CausalExplorationEstimatorConfig) -> None:
        if type(config) is not CausalExplorationEstimatorConfig:
            raise TypeError("config must be an exact CausalExplorationEstimatorConfig")
        self._config = config

    @property
    def config(self) -> CausalExplorationEstimatorConfig:
        return self._config

    def init(self, key: Array) -> CausalExplorationEstimatorState:
        _require_key(key, label="key")
        next_key, prior_key = jr.split(key)
        shape = (
            self._config.ensemble_size,
            N_EXPLORATION_ACTIONS,
            self._config.feature_dim,
        )
        raw_weights = (
            jr.normal(prior_key, shape, dtype=jnp.float32)
            * jnp.float32(self._config.prior_scale)
        )
        weights = jnp.clip(
            raw_weights,
            -jnp.float32(self._config.weight_clip),
            jnp.float32(self._config.weight_clip),
        )
        zero_words = jnp.zeros((2,), dtype=jnp.uint32)
        return CausalExplorationEstimatorState(
            weights=weights,
            action_counts=jnp.zeros((N_EXPLORATION_ACTIONS,), dtype=jnp.uint32),
            action_precision=jnp.ones((N_EXPLORATION_ACTIONS,), dtype=jnp.float32),
            fast_absolute_td_error=jnp.zeros(
                (N_EXPLORATION_ACTIONS,), dtype=jnp.float32
            ),
            slow_absolute_td_error=jnp.zeros(
                (N_EXPLORATION_ACTIONS,), dtype=jnp.float32
            ),
            revision_words=zero_words,
            last_source_event_words=zero_words,
            last_decision_words=zero_words,
            rng_key=next_key,
            estimator_owner_digest=jnp.asarray(
                self._config.estimator_owner_digest,
                dtype=jnp.uint32,
            ),
        )

    def _require_state(self, state: CausalExplorationEstimatorState) -> None:
        if type(state) is not CausalExplorationEstimatorState:
            raise TypeError("state must be an exact CausalExplorationEstimatorState")
        config = self._config
        contracts = (
            (
                state.weights,
                (config.ensemble_size, N_EXPLORATION_ACTIONS, config.feature_dim),
                jnp.float32,
                "weights",
            ),
            (
                state.action_counts,
                (N_EXPLORATION_ACTIONS,),
                jnp.uint32,
                "action_counts",
            ),
            (
                state.action_precision,
                (N_EXPLORATION_ACTIONS,),
                jnp.float32,
                "action_precision",
            ),
            (
                state.fast_absolute_td_error,
                (N_EXPLORATION_ACTIONS,),
                jnp.float32,
                "fast_absolute_td_error",
            ),
            (
                state.slow_absolute_td_error,
                (N_EXPLORATION_ACTIONS,),
                jnp.float32,
                "slow_absolute_td_error",
            ),
            (
                state.estimator_owner_digest,
                (_DIGEST_WORDS,),
                jnp.uint32,
                "estimator_owner_digest",
            ),
        )
        for value, shape, dtype, label in contracts:
            _require_array(value, label=label, shape=shape, dtype=jnp.dtype(dtype))
        for name in ("revision_words", "last_source_event_words", "last_decision_words"):
            _require_array(
                getattr(state, name),
                label=name,
                shape=(2,),
                dtype=jnp.dtype(jnp.uint32),
            )
        _require_key(state.rng_key, label="state.rng_key")

    def _features(self, observation: Array) -> Float[Array, " feature"]:
        _require_array(
            observation,
            label="observation",
            shape=(EXPLORATION_OBSERVATION_DIM,),
            dtype=jnp.dtype(jnp.float32),
        )
        return jnp.concatenate((jnp.ones((1,), dtype=jnp.float32), observation))

    def state_valid(self, state: CausalExplorationEstimatorState) -> Bool[Array, ""]:
        self._require_state(state)
        return (
            jnp.all(jnp.isfinite(state.weights))
            & jnp.all(jnp.abs(state.weights) <= jnp.float32(self._config.weight_clip))
            & jnp.all(jnp.isfinite(state.action_precision))
            & jnp.all(state.action_precision >= 1.0)
            & jnp.all(jnp.isfinite(state.fast_absolute_td_error))
            & jnp.all(state.fast_absolute_td_error >= 0.0)
            & jnp.all(
                state.fast_absolute_td_error <= jnp.float32(self._config.metric_cap)
            )
            & jnp.all(jnp.isfinite(state.slow_absolute_td_error))
            & jnp.all(state.slow_absolute_td_error >= 0.0)
            & jnp.all(
                state.slow_absolute_td_error <= jnp.float32(self._config.metric_cap)
            )
            & _words_equal(state.revision_words, state.last_source_event_words)
            & _words_equal(state.revision_words, state.last_decision_words)
            & _words_equal(
                state.revision_words,
                _sum_uint32_as_words(state.action_counts),
            )
            & jnp.all(
                state.estimator_owner_digest
                == jnp.asarray(self._config.estimator_owner_digest, dtype=jnp.uint32)
            )
        )

    def estimate(
        self,
        state: CausalExplorationEstimatorState,
        observation: Float[Array, " observation"],
        source_event_words: UInt[Array, " 2"],
    ) -> CausalCandidateEstimates:
        """Estimate all candidate metrics without changing estimator state."""

        self._require_state(state)
        phi = self._features(observation)
        _require_array(
            source_event_words,
            label="source_event_words",
            shape=(2,),
            dtype=jnp.dtype(jnp.uint32),
        )
        causal_input_valid = (
            self.state_valid(state)
            & jnp.all(jnp.isfinite(observation))
            & _words_greater(source_event_words, state.last_source_event_words)
        )
        predictions = jnp.einsum("eaf,f->ea", state.weights, phi)
        means = jnp.mean(predictions, axis=0, dtype=jnp.float32)
        disagreement = jnp.std(predictions, axis=0).astype(jnp.float32)
        best = jnp.max(means)
        sigma = jnp.maximum(disagreement, jnp.float32(1e-6))
        z = (means - best) / sigma
        cdf = jnp.float32(0.5) * (
            jnp.float32(1.0) + jax.lax.erf(z / jnp.float32(math.sqrt(2.0)))
        )
        pdf = jnp.exp(jnp.float32(-0.5) * z * z) / jnp.float32(
            math.sqrt(2.0 * math.pi)
        )
        improvement = jnp.maximum(means - best, jnp.float32(0.0)) * cdf + sigma * pdf
        feature_power = jnp.sum(phi * phi, dtype=jnp.float32)
        information_gain = jnp.float32(0.5) * jnp.log1p(
            feature_power / state.action_precision
        )
        learning_progress = jnp.maximum(
            state.slow_absolute_td_error - state.fast_absolute_td_error,
            jnp.float32(0.0),
        )
        cap = jnp.float32(self._config.metric_cap)
        policy = jax.nn.softmax(means).astype(jnp.float32)
        derived_valid = (
            jnp.all(jnp.isfinite(predictions))
            & jnp.all(jnp.isfinite(policy))
            & jnp.all(jnp.isfinite(improvement))
            & jnp.all(jnp.isfinite(disagreement))
            & jnp.all(jnp.isfinite(information_gain))
            & jnp.all(jnp.isfinite(learning_progress))
        )
        valid = causal_input_valid & derived_valid
        zero = jnp.zeros((N_EXPLORATION_ACTIONS,), dtype=jnp.float32)
        return CausalCandidateEstimates(
            candidate_actions=jnp.arange(N_EXPLORATION_ACTIONS, dtype=jnp.int32),
            host_policy=jnp.where(valid, policy, jnp.full_like(policy, 0.25)),
            host_action=jnp.where(valid, jnp.argmax(means).astype(jnp.int32), jnp.int32(0)),
            expected_improvement=jnp.where(
                valid,
                jnp.clip(improvement, 0.0, cap),
                zero,
            ),
            ensemble_disagreement=jnp.where(
                valid,
                jnp.clip(disagreement, 0.0, cap),
                zero,
            ),
            information_gain=jnp.where(
                valid,
                jnp.clip(information_gain, 0.0, cap),
                zero,
            ),
            learning_progress=jnp.where(
                valid,
                jnp.clip(learning_progress, 0.0, cap),
                zero,
            ),
            estimator_revision_words=state.revision_words,
            source_event_words=source_event_words,
            estimator_owner_digest=state.estimator_owner_digest,
            causal_online_estimate=valid,
            oracle_input_used=jnp.asarray(False, dtype=jnp.bool_),
        )

    def update(
        self,
        state: CausalExplorationEstimatorState,
        transition: ExecutedExplorationTransition,
    ) -> CausalEstimatorUpdateResult:
        """Apply one exact-revision TD update from an executed observation."""

        self._require_state(state)
        if type(transition) is not ExecutedExplorationTransition:
            raise TypeError("transition must be an exact ExecutedExplorationTransition")
        phi = self._features(transition.observation)
        next_phi = self._features(transition.next_observation)
        scalar_contracts = (
            (transition.action, jnp.int32, "action"),
            (transition.reward, jnp.float32, "reward"),
        )
        for value, dtype, label in scalar_contracts:
            _require_array(value, label=label, shape=(), dtype=jnp.dtype(dtype))
        for name in (
            "source_event_words",
            "decision_words",
            "estimator_revision_words",
        ):
            _require_array(
                getattr(transition, name),
                label=name,
                shape=(2,),
                dtype=jnp.dtype(jnp.uint32),
            )
        for name in (
            "estimator_owner_digest",
            "action_owner_digest",
            "decision_owner_digest",
            "environment_owner_digest",
        ):
            _require_array(
                getattr(transition, name),
                label=name,
                shape=(_DIGEST_WORDS,),
                dtype=jnp.dtype(jnp.uint32),
            )

        config = self._config
        ownership_valid = (
            jnp.all(
                transition.estimator_owner_digest
                == jnp.asarray(config.estimator_owner_digest, dtype=jnp.uint32)
            )
            & jnp.all(
                transition.action_owner_digest
                == jnp.asarray(config.action_owner_digest, dtype=jnp.uint32)
            )
            & jnp.all(
                transition.decision_owner_digest
                == jnp.asarray(config.decision_owner_digest, dtype=jnp.uint32)
            )
            & jnp.all(
                transition.environment_owner_digest
                == jnp.asarray(config.environment_owner_digest, dtype=jnp.uint32)
            )
        )
        causal_revision_valid = (
            _words_equal(transition.estimator_revision_words, state.revision_words)
            & _is_successor(state.last_source_event_words, transition.source_event_words)
            & _is_successor(state.last_decision_words, transition.decision_words)
            & _words_equal(transition.source_event_words, transition.decision_words)
        )
        input_valid = (
            self.state_valid(state)
            & jnp.all(jnp.isfinite(transition.observation))
            & jnp.isfinite(transition.reward)
            & jnp.all(jnp.isfinite(transition.next_observation))
            & (transition.action >= 0)
            & (transition.action < N_EXPLORATION_ACTIONS)
        )
        next_revision, revision_capacity = _increment_words(state.revision_words)
        safe_action = jnp.clip(transition.action, 0, N_EXPLORATION_ACTIONS - 1)
        count_capacity = state.action_counts[safe_action] < jnp.uint32(_UINT32_MAX)
        lifetime_capacity = revision_capacity & count_capacity
        transaction_preconditions = (
            input_valid & ownership_valid & causal_revision_valid & lifetime_capacity
        )

        current_values = jnp.einsum("eaf,f->ea", state.weights, phi)
        next_values = jnp.einsum("eaf,f->ea", state.weights, next_phi)
        selected_values = current_values[:, safe_action]
        next_best = jnp.max(jnp.mean(next_values, axis=0, dtype=jnp.float32))
        target = transition.reward + jnp.float32(config.discount) * next_best
        td_error = target - selected_values
        normalizer = jnp.maximum(jnp.sum(phi * phi), jnp.float32(1.0))
        increment = (
            jnp.float32(config.step_size) * td_error[:, None] * phi[None, :] / normalizer
        )
        candidate_weights = state.weights.at[:, safe_action, :].add(increment)
        candidate_weights = jnp.clip(
            candidate_weights,
            -jnp.float32(config.weight_clip),
            jnp.float32(config.weight_clip),
        )
        mean_abs = jnp.mean(jnp.abs(td_error), dtype=jnp.float32)
        old_fast = state.fast_absolute_td_error[safe_action]
        old_slow = state.slow_absolute_td_error[safe_action]
        new_fast = jnp.clip(
            old_fast + jnp.float32(config.fast_error_rate) * (mean_abs - old_fast),
            0.0,
            jnp.float32(config.metric_cap),
        )
        new_slow = jnp.clip(
            old_slow + jnp.float32(config.slow_error_rate) * (mean_abs - old_slow),
            0.0,
            jnp.float32(config.metric_cap),
        )
        candidate_state = CausalExplorationEstimatorState(
            weights=candidate_weights,
            action_counts=state.action_counts.at[safe_action].add(jnp.uint32(1)),
            action_precision=state.action_precision.at[safe_action].add(
                jnp.sum(phi * phi, dtype=jnp.float32)
            ),
            fast_absolute_td_error=state.fast_absolute_td_error.at[safe_action].set(new_fast),
            slow_absolute_td_error=state.slow_absolute_td_error.at[safe_action].set(new_slow),
            revision_words=next_revision,
            last_source_event_words=transition.source_event_words,
            last_decision_words=transition.decision_words,
            rng_key=state.rng_key,
            estimator_owner_digest=state.estimator_owner_digest,
        )
        candidate_numerics_valid = (
            jnp.isfinite(target)
            & jnp.all(jnp.isfinite(td_error))
            & jnp.all(jnp.isfinite(increment))
            & jnp.isfinite(mean_abs)
        )
        candidate_valid = self.state_valid(candidate_state)
        applied = transaction_preconditions & candidate_numerics_valid & candidate_valid
        next_state = jax.lax.cond(applied, lambda _: candidate_state, lambda _: state, None)
        return CausalEstimatorUpdateResult(
            state=next_state,
            applied=applied,
            input_valid=input_valid,
            ownership_valid=ownership_valid,
            causal_revision_valid=causal_revision_valid,
            lifetime_capacity_available=lifetime_capacity,
            mean_td_error=jnp.where(
                applied,
                jnp.mean(td_error, dtype=jnp.float32),
                jnp.float32(0.0),
            ),
            mean_absolute_td_error=jnp.where(applied, mean_abs, jnp.float32(0.0)),
            pre_revision_words=state.revision_words,
            post_revision_words=next_state.revision_words,
        )

    def resource_budget(
        self,
        state: CausalExplorationEstimatorState,
    ) -> CausalEstimatorResourceBudget:
        self._require_state(state)
        return CausalEstimatorResourceBudget(
            persistent_state_nbytes=_tree_nbytes(state),
            initialization_normal_draws=(
                self._config.ensemble_size
                * N_EXPLORATION_ACTIONS
                * self._config.feature_dim
            ),
            candidate_scores_per_decision=4 * N_EXPLORATION_ACTIONS,
            ensemble_predictions_per_decision=(
                self._config.ensemble_size * N_EXPLORATION_ACTIONS
            ),
            observed_updates_per_executed_transition=1,
            update_parameter_opportunities_per_transition=(
                self._config.ensemble_size * self._config.feature_dim
            ),
            temporary_scope=(
                "source-level fixed-shape features, predictions, scores, and TD increments; "
                "not a measured allocator or device peak"
            ),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class StochasticTrapEnvironmentConfig:
    """Fixed delayed-benefit world and ownership boundary."""

    delayed_investments_required: int
    stabilize_reward: float
    invest_cost: float
    collect_reward: float
    observation_noise_scale: float
    reward_noise_scale: float
    schedule_owner_digest: tuple[int, ...]
    environment_owner_digest: tuple[int, ...]
    estimator_owner_digest: tuple[int, ...]
    action_owner_digest: tuple[int, ...]
    decision_owner_digest: tuple[int, ...]
    shield_owner_digest: tuple[int, ...]

    def __post_init__(self) -> None:
        if (
            type(self.delayed_investments_required) is not int
            or not 2 <= self.delayed_investments_required <= 16
        ):
            raise ValueError("delayed_investments_required must be in [2, 16]")
        _require_float(
            self.stabilize_reward,
            label="stabilize_reward",
            minimum=0.0,
            strict_minimum=True,
        )
        _require_float(self.invest_cost, label="invest_cost", minimum=-10.0, maximum=0.0)
        _require_float(
            self.collect_reward,
            label="collect_reward",
            minimum=0.0,
            strict_minimum=True,
        )
        _require_float(
            self.observation_noise_scale,
            label="observation_noise_scale",
            minimum=0.0,
            strict_minimum=True,
        )
        _require_float(
            self.reward_noise_scale,
            label="reward_noise_scale",
            minimum=0.0,
            maximum=1.0,
        )
        owners = tuple(
            _require_exact_digest(getattr(self, name), label=name)
            for name in (
                "schedule_owner_digest",
                "environment_owner_digest",
                "estimator_owner_digest",
                "action_owner_digest",
                "decision_owner_digest",
                "shield_owner_digest",
            )
        )
        if len(set(owners)) != len(owners):
            raise ValueError("all stochastic-world owner identities must be distinct")


@chex.dataclass(frozen=True)
class StochasticTrapEnvironmentState:
    """Independent per-arm environment state and exact accepted receipts."""

    delayed_progress: Int[Array, ""]
    previous_reward: Float[Array, ""]
    noisy_tv_channel: Float[Array, ""]
    stable_signal: Float[Array, ""]
    event_words: UInt[Array, " 2"]
    last_decision_words: UInt[Array, " 2"]
    last_estimator_revision_words: UInt[Array, " 2"]
    environment_owner_digest: UInt[Array, " digest"]


@chex.dataclass(frozen=True)
class ExplorationExogenousEvent:
    """Evaluator-owned paired noise; it contains no value or regime answer."""

    source_event_words: UInt[Array, " 2"]
    stable_noise: Float[Array, ""]
    reward_noise: Float[Array, ""]
    noisy_tv_noise: Float[Array, ""]
    schedule_owner_digest: UInt[Array, " digest"]


@chex.dataclass(frozen=True)
class ShieldedExplorationDecision:
    """Caller-owned executable action bound to ranking and estimator revisions."""

    action: Int[Array, ""]
    action_available: Bool[Array, ""]
    executed_action_safety_allowed: Bool[Array, ""]
    source_event_words: UInt[Array, " 2"]
    decision_words: UInt[Array, " 2"]
    estimator_revision_words: UInt[Array, " 2"]
    estimator_owner_digest: UInt[Array, " digest"]
    action_owner_digest: UInt[Array, " digest"]
    decision_owner_digest: UInt[Array, " digest"]
    shield_owner_digest: UInt[Array, " digest"]


@chex.dataclass(frozen=True)
class StochasticTrapStepResult:
    """Atomic environment result and the exact observed transition."""

    state: StochasticTrapEnvironmentState
    observation: Float[Array, " observation"]
    next_observation: Float[Array, " observation"]
    reward: Float[Array, ""]
    transition: ExecutedExplorationTransition
    applied: Bool[Array, ""]
    input_valid: Bool[Array, ""]
    ownership_valid: Bool[Array, ""]
    causal_revision_valid: Bool[Array, ""]
    hard_safety_valid: Bool[Array, ""]
    noisy_tv_observed: Bool[Array, ""]
    delayed_investment_applied: Bool[Array, ""]
    delayed_collection_applied: Bool[Array, ""]


def initial_stochastic_trap_environment(
    config: StochasticTrapEnvironmentConfig,
) -> StochasticTrapEnvironmentState:
    """Return one independent zero-history environment state."""

    if type(config) is not StochasticTrapEnvironmentConfig:
        raise TypeError("config must be an exact StochasticTrapEnvironmentConfig")
    zero = jnp.zeros((2,), dtype=jnp.uint32)
    return StochasticTrapEnvironmentState(
        delayed_progress=jnp.asarray(0, dtype=jnp.int32),
        previous_reward=jnp.asarray(0.0, dtype=jnp.float32),
        noisy_tv_channel=jnp.asarray(0.0, dtype=jnp.float32),
        stable_signal=jnp.asarray(0.0, dtype=jnp.float32),
        event_words=zero,
        last_decision_words=zero,
        last_estimator_revision_words=zero,
        environment_owner_digest=jnp.asarray(
            config.environment_owner_digest,
            dtype=jnp.uint32,
        ),
    )


def _require_stochastic_trap_environment_state(
    state: StochasticTrapEnvironmentState,
) -> None:
    if type(state) is not StochasticTrapEnvironmentState:
        raise TypeError("state must be an exact StochasticTrapEnvironmentState")
    for value, dtype, label in (
        (state.delayed_progress, jnp.int32, "delayed_progress"),
        (state.previous_reward, jnp.float32, "previous_reward"),
        (state.noisy_tv_channel, jnp.float32, "noisy_tv_channel"),
        (state.stable_signal, jnp.float32, "stable_signal"),
    ):
        _require_array(value, label=label, shape=(), dtype=jnp.dtype(dtype))
    for name in ("event_words", "last_decision_words", "last_estimator_revision_words"):
        _require_array(
            getattr(state, name),
            label=name,
            shape=(2,),
            dtype=jnp.dtype(jnp.uint32),
        )
    _require_array(
        state.environment_owner_digest,
        label="environment_owner_digest",
        shape=(_DIGEST_WORDS,),
        dtype=jnp.dtype(jnp.uint32),
    )


def stochastic_trap_environment_state_valid(
    config: StochasticTrapEnvironmentConfig,
    state: StochasticTrapEnvironmentState,
) -> Bool[Array, ""]:
    """Validate bounds, ownership, and the event/decision/estimator partition."""

    if type(config) is not StochasticTrapEnvironmentConfig:
        raise TypeError("config must be an exact StochasticTrapEnvironmentConfig")
    _require_stochastic_trap_environment_state(state)
    zero = jnp.zeros((2,), dtype=jnp.uint32)
    initial_partition = _words_equal(state.event_words, zero) & _words_equal(
        state.last_estimator_revision_words, zero
    )
    live_partition = _is_successor(
        state.last_estimator_revision_words,
        state.event_words,
    )
    return (
        (state.delayed_progress >= 0)
        & (state.delayed_progress <= config.delayed_investments_required)
        & jnp.isfinite(state.previous_reward)
        & jnp.isfinite(state.noisy_tv_channel)
        & jnp.isfinite(state.stable_signal)
        & _words_equal(state.event_words, state.last_decision_words)
        & (initial_partition | live_partition)
        & jnp.all(
            state.environment_owner_digest
            == jnp.asarray(config.environment_owner_digest, dtype=jnp.uint32)
        )
    )


def stochastic_trap_observation(
    config: StochasticTrapEnvironmentConfig,
    state: StochasticTrapEnvironmentState,
) -> Float[Array, " observation"]:
    """Expose learner-visible state; no future value or action label is included."""

    _require_stochastic_trap_environment_state(state)
    return jnp.asarray(
        (
            state.delayed_progress.astype(jnp.float32)
            / jnp.float32(config.delayed_investments_required),
            (state.delayed_progress >= config.delayed_investments_required).astype(jnp.float32),
            jnp.clip(state.previous_reward, -2.0, 2.0),
            jnp.tanh(state.noisy_tv_channel),
            jnp.tanh(state.stable_signal),
        ),
        dtype=jnp.float32,
    )


def stochastic_trap_safety_mask(
    config: StochasticTrapEnvironmentConfig,
    state: StochasticTrapEnvironmentState,
) -> Bool[Array, " action"]:
    """Permit collection only when prior investments visibly unlocked it."""

    _require_stochastic_trap_environment_state(state)
    collect_allowed = state.delayed_progress >= config.delayed_investments_required
    return jnp.asarray((True, True, True, collect_allowed), dtype=jnp.bool_)


def _zero_transition(
    config: StochasticTrapEnvironmentConfig,
    state: StochasticTrapEnvironmentState,
    observation: Array,
) -> ExecutedExplorationTransition:
    return ExecutedExplorationTransition(
        observation=observation,
        action=jnp.asarray(-1, dtype=jnp.int32),
        reward=jnp.asarray(0.0, dtype=jnp.float32),
        next_observation=observation,
        source_event_words=state.event_words,
        decision_words=state.last_decision_words,
        estimator_revision_words=state.last_estimator_revision_words,
        estimator_owner_digest=jnp.asarray(config.estimator_owner_digest, dtype=jnp.uint32),
        action_owner_digest=jnp.asarray(config.action_owner_digest, dtype=jnp.uint32),
        decision_owner_digest=jnp.asarray(config.decision_owner_digest, dtype=jnp.uint32),
        environment_owner_digest=state.environment_owner_digest,
    )


def stochastic_trap_environment_step(
    config: StochasticTrapEnvironmentConfig,
    state: StochasticTrapEnvironmentState,
    event: ExplorationExogenousEvent,
    decision: ShieldedExplorationDecision,
) -> StochasticTrapStepResult:
    """Execute exactly one caller-shielded action and reveal its observation."""

    if type(config) is not StochasticTrapEnvironmentConfig:
        raise TypeError("config must be an exact StochasticTrapEnvironmentConfig")
    if type(state) is not StochasticTrapEnvironmentState:
        raise TypeError("state must be an exact StochasticTrapEnvironmentState")
    if type(event) is not ExplorationExogenousEvent:
        raise TypeError("event must be an exact ExplorationExogenousEvent")
    if type(decision) is not ShieldedExplorationDecision:
        raise TypeError("decision must be an exact ShieldedExplorationDecision")
    _require_stochastic_trap_environment_state(state)
    for value, dtype, label in (
        (event.stable_noise, jnp.float32, "event.stable_noise"),
        (event.reward_noise, jnp.float32, "event.reward_noise"),
        (event.noisy_tv_noise, jnp.float32, "event.noisy_tv_noise"),
        (decision.action, jnp.int32, "decision.action"),
        (decision.action_available, jnp.bool_, "decision.action_available"),
        (
            decision.executed_action_safety_allowed,
            jnp.bool_,
            "decision.executed_action_safety_allowed",
        ),
    ):
        _require_array(value, label=label, shape=(), dtype=jnp.dtype(dtype))
    for value, label in (
        (event.source_event_words, "event.source_event_words"),
        (decision.source_event_words, "decision.source_event_words"),
        (decision.decision_words, "decision.decision_words"),
        (decision.estimator_revision_words, "decision.estimator_revision_words"),
    ):
        _require_array(
            value,
            label=label,
            shape=(2,),
            dtype=jnp.dtype(jnp.uint32),
        )
    for value, label in (
        (event.schedule_owner_digest, "event.schedule_owner_digest"),
        (decision.estimator_owner_digest, "decision.estimator_owner_digest"),
        (decision.action_owner_digest, "decision.action_owner_digest"),
        (decision.decision_owner_digest, "decision.decision_owner_digest"),
        (decision.shield_owner_digest, "decision.shield_owner_digest"),
    ):
        _require_array(
            value,
            label=label,
            shape=(_DIGEST_WORDS,),
            dtype=jnp.dtype(jnp.uint32),
        )
    observation = stochastic_trap_observation(config, state)
    action = jnp.clip(decision.action, 0, N_EXPLORATION_ACTIONS - 1)
    safety_mask = stochastic_trap_safety_mask(config, state)
    hard_safety_valid = decision.action_available & safety_mask[action]
    ownership_valid = (
        jnp.all(
            state.environment_owner_digest
            == jnp.asarray(config.environment_owner_digest, dtype=jnp.uint32)
        )
        & jnp.all(
            event.schedule_owner_digest
            == jnp.asarray(config.schedule_owner_digest, dtype=jnp.uint32)
        )
        & jnp.all(
            decision.estimator_owner_digest
            == jnp.asarray(config.estimator_owner_digest, dtype=jnp.uint32)
        )
        & jnp.all(
            decision.action_owner_digest
            == jnp.asarray(config.action_owner_digest, dtype=jnp.uint32)
        )
        & jnp.all(
            decision.decision_owner_digest
            == jnp.asarray(config.decision_owner_digest, dtype=jnp.uint32)
        )
        & jnp.all(
            decision.shield_owner_digest
            == jnp.asarray(config.shield_owner_digest, dtype=jnp.uint32)
        )
    )
    expected_estimator_revision = jax.lax.cond(
        _words_equal(state.event_words, jnp.zeros((2,), dtype=jnp.uint32)),
        lambda _: jnp.zeros((2,), dtype=jnp.uint32),
        lambda _: _increment_words(state.last_estimator_revision_words)[0],
        None,
    )
    causal_revision_valid = (
        _is_successor(state.event_words, event.source_event_words)
        & _words_equal(decision.source_event_words, event.source_event_words)
        & _is_successor(state.last_decision_words, decision.decision_words)
        & _words_equal(decision.decision_words, event.source_event_words)
        & _words_equal(decision.estimator_revision_words, expected_estimator_revision)
    )
    input_valid = (
        stochastic_trap_environment_state_valid(config, state)
        & jnp.isfinite(event.stable_noise)
        & jnp.isfinite(event.reward_noise)
        & jnp.isfinite(event.noisy_tv_noise)
        & (decision.action >= 0)
        & (decision.action < N_EXPLORATION_ACTIONS)
        & decision.executed_action_safety_allowed
    )
    transaction_preconditions = (
        input_valid & ownership_valid & causal_revision_valid & hard_safety_valid
    )

    is_stabilize = action == STABILIZE_ACTION
    is_tv = action == NOISY_TV_ACTION
    is_invest = action == INVEST_ACTION
    is_collect = action == COLLECT_ACTION
    progress_after = jnp.where(
        is_tv,
        jnp.int32(0),
        jnp.where(
            is_invest,
            jnp.minimum(
                state.delayed_progress + jnp.int32(1),
                jnp.int32(config.delayed_investments_required),
            ),
            jnp.where(is_collect, jnp.int32(0), state.delayed_progress),
        ),
    )
    base_reward = jnp.where(
        is_stabilize,
        jnp.float32(config.stabilize_reward),
        jnp.where(
            is_tv,
            jnp.float32(0.0),
            jnp.where(
                is_invest,
                jnp.float32(config.invest_cost),
                jnp.float32(config.collect_reward),
            ),
        ),
    )
    reward = base_reward + jnp.float32(config.reward_noise_scale) * event.reward_noise
    tv_channel = jnp.where(
        is_tv,
        jnp.float32(config.observation_noise_scale) * event.noisy_tv_noise,
        jnp.float32(0.0),
    )
    stable_signal = jnp.float32(0.8) * state.stable_signal + jnp.float32(
        0.2
    ) * event.stable_noise
    candidate_state = StochasticTrapEnvironmentState(
        delayed_progress=progress_after,
        previous_reward=reward,
        noisy_tv_channel=tv_channel,
        stable_signal=stable_signal,
        event_words=event.source_event_words,
        last_decision_words=decision.decision_words,
        last_estimator_revision_words=decision.estimator_revision_words,
        environment_owner_digest=state.environment_owner_digest,
    )
    candidate_valid = stochastic_trap_environment_state_valid(config, candidate_state)
    applied = transaction_preconditions & candidate_valid
    next_state = jax.lax.cond(applied, lambda _: candidate_state, lambda _: state, None)
    next_observation = stochastic_trap_observation(config, next_state)
    transition = ExecutedExplorationTransition(
        observation=observation,
        action=jnp.where(applied, action, jnp.int32(-1)),
        reward=jnp.where(applied, reward, jnp.float32(0.0)),
        next_observation=next_observation,
        source_event_words=jnp.where(applied, event.source_event_words, state.event_words),
        decision_words=jnp.where(applied, decision.decision_words, state.last_decision_words),
        estimator_revision_words=jnp.where(
            applied,
            decision.estimator_revision_words,
            state.last_estimator_revision_words,
        ),
        estimator_owner_digest=jnp.asarray(config.estimator_owner_digest, dtype=jnp.uint32),
        action_owner_digest=jnp.asarray(config.action_owner_digest, dtype=jnp.uint32),
        decision_owner_digest=jnp.asarray(config.decision_owner_digest, dtype=jnp.uint32),
        environment_owner_digest=state.environment_owner_digest,
    )
    return StochasticTrapStepResult(
        state=next_state,
        observation=observation,
        next_observation=next_observation,
        reward=jnp.where(applied, reward, jnp.float32(0.0)),
        transition=jax.lax.cond(
            applied,
            lambda _: transition,
            lambda _: _zero_transition(config, state, observation),
            None,
        ),
        applied=applied,
        input_valid=input_valid,
        ownership_valid=ownership_valid,
        causal_revision_valid=causal_revision_valid,
        hard_safety_valid=hard_safety_valid,
        noisy_tv_observed=applied & is_tv,
        delayed_investment_applied=applied & is_invest,
        delayed_collection_applied=applied & is_collect,
    )


@dataclasses.dataclass(frozen=True, slots=True)
class CallerOwnedHardShieldConfig:
    """Owner identities for the explicit post-ranking simulation shield."""

    estimator_owner_digest: tuple[int, ...]
    action_owner_digest: tuple[int, ...]
    decision_owner_digest: tuple[int, ...]
    shield_owner_digest: tuple[int, ...]

    def __post_init__(self) -> None:
        owners = tuple(
            _require_exact_digest(getattr(self, name), label=name)
            for name in (
                "estimator_owner_digest",
                "action_owner_digest",
                "decision_owner_digest",
                "shield_owner_digest",
            )
        )
        if len(set(owners)) != len(owners):
            raise ValueError("hard-shield owner identities must be distinct")


@chex.dataclass(frozen=True)
class CallerOwnedHardShieldState:
    """Exact shield revision and accepted ranking receipts."""

    revision_words: UInt[Array, " 2"]
    last_source_event_words: UInt[Array, " 2"]
    last_decision_words: UInt[Array, " 2"]
    last_estimator_revision_words: UInt[Array, " 2"]
    shield_owner_digest: UInt[Array, " digest"]


@chex.dataclass(frozen=True)
class RankedExplorationDecision:
    """Non-dispatching selector output presented to the caller-owned shield."""

    selected_action: Int[Array, ""]
    host_action: Int[Array, ""]
    ranking_applied: Bool[Array, ""]
    source_event_words: UInt[Array, " 2"]
    pre_decision_words: UInt[Array, " 2"]
    post_decision_words: UInt[Array, " 2"]
    estimator_revision_words: UInt[Array, " 2"]
    estimator_owner_digest: UInt[Array, " digest"]
    decision_owner_digest: UInt[Array, " digest"]


@chex.dataclass(frozen=True)
class CallerOwnedHardShieldResult:
    """Shield state, executable action receipt, and fallback diagnostics."""

    state: CallerOwnedHardShieldState
    decision: ShieldedExplorationDecision
    applied: Bool[Array, ""]
    selected_action_allowed: Bool[Array, ""]
    selected_action_executed: Bool[Array, ""]
    fallback_used: Bool[Array, ""]
    ownership_valid: Bool[Array, ""]
    causal_revision_valid: Bool[Array, ""]
    lifetime_capacity_available: Bool[Array, ""]


def initial_caller_owned_hard_shield(
    config: CallerOwnedHardShieldConfig,
) -> CallerOwnedHardShieldState:
    if type(config) is not CallerOwnedHardShieldConfig:
        raise TypeError("config must be an exact CallerOwnedHardShieldConfig")
    zero = jnp.zeros((2,), dtype=jnp.uint32)
    return CallerOwnedHardShieldState(
        revision_words=zero,
        last_source_event_words=zero,
        last_decision_words=zero,
        last_estimator_revision_words=zero,
        shield_owner_digest=jnp.asarray(config.shield_owner_digest, dtype=jnp.uint32),
    )


def _require_caller_owned_hard_shield_state(
    state: CallerOwnedHardShieldState,
) -> None:
    if type(state) is not CallerOwnedHardShieldState:
        raise TypeError("state must be an exact CallerOwnedHardShieldState")
    for name in (
        "revision_words",
        "last_source_event_words",
        "last_decision_words",
        "last_estimator_revision_words",
    ):
        _require_array(
            getattr(state, name),
            label=name,
            shape=(2,),
            dtype=jnp.dtype(jnp.uint32),
        )
    _require_array(
        state.shield_owner_digest,
        label="shield_owner_digest",
        shape=(_DIGEST_WORDS,),
        dtype=jnp.dtype(jnp.uint32),
    )


def caller_owned_hard_shield_state_valid(
    config: CallerOwnedHardShieldConfig,
    state: CallerOwnedHardShieldState,
) -> Bool[Array, ""]:
    """Validate shield ownership and the decision/estimator revision partition."""

    if type(config) is not CallerOwnedHardShieldConfig:
        raise TypeError("config must be an exact CallerOwnedHardShieldConfig")
    _require_caller_owned_hard_shield_state(state)
    zero = jnp.zeros((2,), dtype=jnp.uint32)
    initial_partition = _words_equal(state.revision_words, zero) & _words_equal(
        state.last_estimator_revision_words, zero
    )
    live_partition = _is_successor(
        state.last_estimator_revision_words,
        state.revision_words,
    )
    return (
        _words_equal(state.revision_words, state.last_source_event_words)
        & _words_equal(state.revision_words, state.last_decision_words)
        & (initial_partition | live_partition)
        & jnp.all(
            state.shield_owner_digest
            == jnp.asarray(config.shield_owner_digest, dtype=jnp.uint32)
        )
    )


def apply_caller_owned_hard_shield(
    config: CallerOwnedHardShieldConfig,
    state: CallerOwnedHardShieldState,
    ranked: RankedExplorationDecision,
    safety_mask: Bool[Array, " action"],
) -> CallerOwnedHardShieldResult:
    """Apply a caller-owned safety decision after ranking and before execution."""

    if type(config) is not CallerOwnedHardShieldConfig:
        raise TypeError("config must be an exact CallerOwnedHardShieldConfig")
    if type(state) is not CallerOwnedHardShieldState:
        raise TypeError("state must be an exact CallerOwnedHardShieldState")
    if type(ranked) is not RankedExplorationDecision:
        raise TypeError("ranked must be an exact RankedExplorationDecision")
    _require_caller_owned_hard_shield_state(state)
    _require_array(
        safety_mask,
        label="safety_mask",
        shape=(N_EXPLORATION_ACTIONS,),
        dtype=jnp.dtype(jnp.bool_),
    )
    for value, dtype, label in (
        (ranked.selected_action, jnp.int32, "ranked.selected_action"),
        (ranked.host_action, jnp.int32, "ranked.host_action"),
        (ranked.ranking_applied, jnp.bool_, "ranked.ranking_applied"),
    ):
        _require_array(value, label=label, shape=(), dtype=jnp.dtype(dtype))
    for value, label in (
        (ranked.source_event_words, "ranked.source_event_words"),
        (ranked.pre_decision_words, "ranked.pre_decision_words"),
        (ranked.post_decision_words, "ranked.post_decision_words"),
        (ranked.estimator_revision_words, "ranked.estimator_revision_words"),
    ):
        _require_array(
            value,
            label=label,
            shape=(2,),
            dtype=jnp.dtype(jnp.uint32),
        )
    for value, label in (
        (ranked.estimator_owner_digest, "ranked.estimator_owner_digest"),
        (ranked.decision_owner_digest, "ranked.decision_owner_digest"),
    ):
        _require_array(
            value,
            label=label,
            shape=(_DIGEST_WORDS,),
            dtype=jnp.dtype(jnp.uint32),
        )
    safe_selected = jnp.clip(ranked.selected_action, 0, N_EXPLORATION_ACTIONS - 1)
    selected_allowed = safety_mask[safe_selected]
    fallback_allowed = safety_mask[STABILIZE_ACTION]
    next_revision, capacity = _increment_words(state.revision_words)
    ownership_valid = (
        jnp.all(
            state.shield_owner_digest
            == jnp.asarray(config.shield_owner_digest, dtype=jnp.uint32)
        )
        & jnp.all(
            ranked.estimator_owner_digest
            == jnp.asarray(config.estimator_owner_digest, dtype=jnp.uint32)
        )
        & jnp.all(
            ranked.decision_owner_digest
            == jnp.asarray(config.decision_owner_digest, dtype=jnp.uint32)
        )
    )
    expected_estimator_revision = jax.lax.cond(
        _words_equal(state.revision_words, jnp.zeros((2,), dtype=jnp.uint32)),
        lambda _: jnp.zeros((2,), dtype=jnp.uint32),
        lambda _: _increment_words(state.last_estimator_revision_words)[0],
        None,
    )
    causal_revision_valid = (
        _is_successor(state.last_source_event_words, ranked.source_event_words)
        & _words_equal(ranked.pre_decision_words, state.last_decision_words)
        & _is_successor(ranked.pre_decision_words, ranked.post_decision_words)
        & _words_equal(ranked.post_decision_words, ranked.source_event_words)
        & _words_equal(ranked.estimator_revision_words, expected_estimator_revision)
    )
    input_valid = (
        caller_owned_hard_shield_state_valid(config, state)
        & ranked.ranking_applied
        & (ranked.selected_action >= 0)
        & (ranked.selected_action < N_EXPLORATION_ACTIONS)
        & (ranked.host_action >= 0)
        & (ranked.host_action < N_EXPLORATION_ACTIONS)
    )
    candidate_state = CallerOwnedHardShieldState(
        revision_words=next_revision,
        last_source_event_words=ranked.source_event_words,
        last_decision_words=ranked.post_decision_words,
        last_estimator_revision_words=ranked.estimator_revision_words,
        shield_owner_digest=state.shield_owner_digest,
    )
    candidate_valid = caller_owned_hard_shield_state_valid(config, candidate_state)
    safe_action_available = selected_allowed | fallback_allowed
    applied = (
        input_valid
        & ownership_valid
        & causal_revision_valid
        & capacity
        & safe_action_available
        & candidate_valid
    )
    selected_executed = applied & selected_allowed
    fallback_used = applied & ~selected_allowed & fallback_allowed
    action_available = selected_executed | fallback_used
    executable_action = jnp.where(
        selected_executed,
        ranked.selected_action,
        jnp.where(fallback_used, jnp.int32(STABILIZE_ACTION), jnp.int32(-1)),
    )
    next_state = jax.lax.cond(applied, lambda _: candidate_state, lambda _: state, None)
    zero = jnp.zeros((2,), dtype=jnp.uint32)
    decision = ShieldedExplorationDecision(
        action=jnp.where(action_available, executable_action, jnp.int32(-1)),
        action_available=action_available,
        executed_action_safety_allowed=action_available,
        source_event_words=jnp.where(applied, ranked.source_event_words, zero),
        decision_words=jnp.where(applied, ranked.post_decision_words, zero),
        estimator_revision_words=jnp.where(applied, ranked.estimator_revision_words, zero),
        estimator_owner_digest=jnp.asarray(config.estimator_owner_digest, dtype=jnp.uint32),
        action_owner_digest=jnp.asarray(config.action_owner_digest, dtype=jnp.uint32),
        decision_owner_digest=jnp.asarray(config.decision_owner_digest, dtype=jnp.uint32),
        shield_owner_digest=jnp.asarray(config.shield_owner_digest, dtype=jnp.uint32),
    )
    return CallerOwnedHardShieldResult(
        state=next_state,
        decision=decision,
        applied=applied,
        selected_action_allowed=applied & selected_allowed,
        selected_action_executed=selected_executed,
        fallback_used=fallback_used,
        ownership_valid=ownership_valid,
        causal_revision_valid=causal_revision_valid,
        lifetime_capacity_available=capacity,
    )


@dataclasses.dataclass(frozen=True, slots=True)
class CausalExplorationCoreResources:
    """Exact persistent byte counts for the non-selector core states."""

    estimator_state_nbytes: int
    environment_state_nbytes: int
    hard_shield_state_nbytes: int
    total_state_nbytes: int


def measure_causal_exploration_core_resources(
    estimator: CausalExplorationEstimatorState,
    environment: StochasticTrapEnvironmentState,
    shield: CallerOwnedHardShieldState,
) -> CausalExplorationCoreResources:
    estimator_bytes = _tree_nbytes(estimator)
    environment_bytes = _tree_nbytes(environment)
    shield_bytes = _tree_nbytes(shield)
    return CausalExplorationCoreResources(
        estimator_state_nbytes=estimator_bytes,
        environment_state_nbytes=environment_bytes,
        hard_shield_state_nbytes=shield_bytes,
        total_state_nbytes=estimator_bytes + environment_bytes + shield_bytes,
    )


__all__ = [
    "CAUSAL_EXPLORATION_ASSESSMENT_STATUS",
    "CAUSAL_EXPLORATION_DEPLOYMENT_AUTHORITY",
    "CAUSAL_EXPLORATION_EVIDENCE_LEVEL",
    "CAUSAL_EXPLORATION_LIFETIME_SEMANTICS",
    "CAUSAL_EXPLORATION_OUTPUT_WRITE_AUTHORITY",
    "CAUSAL_EXPLORATION_PHYSICAL_DISPATCH_AUTHORITY",
    "CAUSAL_EXPLORATION_PROMOTION_AUTHORITY",
    "CAUSAL_EXPLORATION_SCIENTIFIC_PROMOTION_ALLOWED",
    "COLLECT_ACTION",
    "EXPLORATION_OBSERVATION_DIM",
    "INVEST_ACTION",
    "NOISY_TV_ACTION",
    "N_EXPLORATION_ACTIONS",
    "STABILIZE_ACTION",
    "CallerOwnedHardShieldConfig",
    "CallerOwnedHardShieldResult",
    "CallerOwnedHardShieldState",
    "CausalCandidateEstimates",
    "CausalEstimatorResourceBudget",
    "CausalEstimatorUpdateResult",
    "CausalExplorationCoreResources",
    "CausalExplorationEstimator",
    "CausalExplorationEstimatorConfig",
    "CausalExplorationEstimatorState",
    "ExecutedExplorationTransition",
    "ExplorationExogenousEvent",
    "RankedExplorationDecision",
    "ShieldedExplorationDecision",
    "StochasticTrapEnvironmentConfig",
    "StochasticTrapEnvironmentState",
    "StochasticTrapStepResult",
    "apply_caller_owned_hard_shield",
    "caller_owned_hard_shield_state_valid",
    "initial_caller_owned_hard_shield",
    "initial_stochastic_trap_environment",
    "measure_causal_exploration_core_resources",
    "stochastic_trap_environment_step",
    "stochastic_trap_environment_state_valid",
    "stochastic_trap_observation",
    "stochastic_trap_safety_mask",
]
