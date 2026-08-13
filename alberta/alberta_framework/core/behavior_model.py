# mypy: disable-error-code="call-arg,name-defined"
"""Online behavior/action prediction for discrete-action agents.

The behavior model is a temporally uniform supervised learner for
``P(A_t | features_t)``.  It is deliberately separate from control: SARSA,
actor-critic, scripted policies, external logs, and future dream rollouts can
all feed the same observed ``(features, action)`` stream into this model.
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
from jax import Array
from jaxtyping import Bool, Float, Int, UInt

BEHAVIOR_MODEL_STATE_SCHEMA = "alberta.behavior-model-state.v2"
BEHAVIOR_MODEL_LIFETIME_COUNTER_NBYTES = 12
BEHAVIOR_MODEL_LIFETIME_COUNTER_DELTA_NBYTES = 8
_INT32_MAX = 2**31 - 1
_UINT32_MAX = 2**32 - 1


def _saturating_int32_increment(value: Array) -> Array:
    maximum = jnp.asarray(_INT32_MAX, dtype=jnp.int32)
    counter = jnp.asarray(value, dtype=jnp.int32)
    return jnp.minimum(jnp.maximum(counter, 0), maximum - 1) + 1


def _checked_lifetime_words_increment(
    words: Array,
) -> tuple[UInt[Array, " 2"], Bool[Array, ""]]:
    """Propose the next exact word identity without wrapping all-ones."""

    if getattr(words, "shape", None) != (2,):
        raise ValueError("behavior lifetime words must have shape (2,)")
    if getattr(words, "dtype", None) != jnp.dtype(jnp.uint32):
        raise TypeError("behavior lifetime words must have dtype uint32")
    maximum = jnp.asarray(_UINT32_MAX, dtype=jnp.uint32)
    capacity_available = ~jnp.all(words == maximum)
    one = jnp.asarray(1, dtype=jnp.uint32)
    low = words[1] + one
    carry = (low == jnp.asarray(0, dtype=jnp.uint32)).astype(jnp.uint32)
    proposed = jnp.stack((words[0] + carry, low))
    return (
        jnp.where(capacity_available, proposed, words).astype(jnp.uint32),
        capacity_available,
    )


def _lifetime_counter_valid(words: Array, telemetry: Array) -> Bool[Array, ""]:
    """Validate exact identity against saturating int32 compatibility telemetry."""

    if getattr(words, "shape", None) != (2,):
        raise ValueError("behavior lifetime words must have shape (2,)")
    if getattr(words, "dtype", None) != jnp.dtype(jnp.uint32):
        raise TypeError("behavior lifetime words must have dtype uint32")
    if getattr(telemetry, "shape", None) != ():
        raise ValueError("behavior step_count must be scalar")
    if getattr(telemetry, "dtype", None) != jnp.dtype(jnp.int32):
        raise TypeError("behavior step_count must have dtype int32")
    maximum_i32 = jnp.asarray(_INT32_MAX, dtype=jnp.int32)
    below_saturation = (words[0] == jnp.asarray(0, dtype=jnp.uint32)) & (
        words[1] < jnp.asarray(_INT32_MAX, dtype=jnp.uint32)
    )
    expected = words[1].astype(jnp.int32)
    return (telemetry >= 0) & jnp.where(
        below_saturation,
        telemetry == expected,
        telemetry == maximum_i32,
    )


def floor_and_renormalize_probabilities(
    probabilities: Array,
    min_probability: float = 1e-6,
) -> Array:
    """Floor probabilities and return a valid simplex along the last axis.

    This helper is for sampling or reporting a proper simplex distribution
    whose entries are at least ``min_probability``. Importance-ratio denominators
    should use :func:`selected_action_probabilities`, which floors only the
    selected action probability and does not change other actions.
    """
    probs = jnp.asarray(probabilities, dtype=jnp.float32)
    n_actions = probabilities.shape[-1]
    if min_probability * n_actions >= 1.0:
        return jnp.ones_like(probs) / n_actions
    clipped = jnp.maximum(probs, 0.0)
    normalizer = jnp.maximum(
        jnp.sum(clipped, axis=-1, keepdims=True),
        jnp.asarray(1e-12, dtype=jnp.float32),
    )
    normalized = clipped / normalizer
    floor_mass = jnp.asarray(min_probability * n_actions, dtype=jnp.float32)
    return jnp.asarray(min_probability, dtype=jnp.float32) + (1.0 - floor_mass) * normalized


def selected_action_probabilities(
    probabilities: Array,
    actions: Array,
    min_probability: float = 1e-6,
) -> Array:
    """Return floor-clipped probabilities for selected discrete actions.

    ``probabilities`` may be a single action distribution with shape
    ``(n_actions,)`` or a batch with actions on the last axis. ``actions`` must
    broadcast to ``probabilities.shape[:-1]``.
    """
    probs = jnp.asarray(probabilities, dtype=jnp.float32)
    action_ids = jnp.asarray(actions, dtype=jnp.int32)
    one_hot = jax.nn.one_hot(action_ids, probs.shape[-1], dtype=jnp.float32)
    selected = jnp.sum(probs * one_hot, axis=-1)
    return jnp.maximum(selected, jnp.asarray(min_probability, dtype=jnp.float32))


def action_log_likelihoods(
    probabilities: Array,
    actions: Array,
    min_probability: float = 1e-6,
) -> Array:
    """Return log-likelihoods for selected actions under a behavior model."""
    return jnp.log(
        selected_action_probabilities(
            probabilities,
            actions,
            min_probability=min_probability,
        )
    )


def clipped_importance_ratios(
    target_probabilities: Array,
    behavior_probabilities: Array,
    actions: Array,
    *,
    clip: float | None = 10.0,
    min_behavior_probability: float = 1e-6,
) -> Array:
    """Compute selected-action target/behavior ratios with safe denominators.

    Args:
        target_probabilities: Target policy probabilities with actions on the
            last axis.
        behavior_probabilities: Behavior model probabilities with actions on
            the last axis.
        actions: Discrete selected actions.
        clip: Optional upper bound on ratios. ``None`` disables clipping.
        min_behavior_probability: Lower bound for behavior denominators.

    Returns:
        Per-sample ratios with shape ``target_probabilities.shape[:-1]``.
    """
    target = selected_action_probabilities(
        target_probabilities,
        actions,
        min_probability=0.0,
    )
    behavior = selected_action_probabilities(
        behavior_probabilities,
        actions,
        min_probability=min_behavior_probability,
    )
    ratios = target / behavior
    if clip is None:
        return ratios
    return jnp.minimum(ratios, jnp.asarray(clip, dtype=jnp.float32))


def epsilon_greedy_probabilities(
    q_values: Array,
    epsilon: Array,
    tie_tolerance: float = 1e-6,
) -> Array:
    """Return the exact epsilon-greedy action distribution for Q-values.

    This mirrors the SARSA/Q-learning policy surface: exploration is uniform
    over all actions and exploitation is uniform over maximal actions.
    """
    q = jnp.asarray(q_values, dtype=jnp.float32)
    n_actions = q.shape[-1]
    eps = jnp.asarray(epsilon, dtype=jnp.float32)
    max_q = jnp.max(q, axis=-1, keepdims=True)
    greedy_mask = jnp.isclose(q, max_q, atol=tie_tolerance, rtol=0.0).astype(jnp.float32)
    n_greedy = jnp.sum(greedy_mask, axis=-1, keepdims=True)
    explore = eps / n_actions
    exploit = (1.0 - eps) * greedy_mask / jnp.maximum(n_greedy, 1.0)
    return exploit + explore


@dataclasses.dataclass(frozen=True)
class BehaviorModelConfig:
    """Configuration for a linear online discrete behavior model.

    Attributes:
        n_actions: Number of discrete actions.
        step_size: Cross-entropy gradient step-size.
        temperature: Softmax temperature for behavior probabilities.
        l2_penalty: Optional L2 shrinkage on weights and biases.
        max_gradient_norm: Optional global gradient-norm clip before applying
            ``step_size``.
        min_probability: Probability floor for likelihood and ratio helpers.
        ratio_clip: Default ratio clip for off-policy helper methods.
        diagnostic_decay: EMA decay used for online reliability diagnostics.
    """

    n_actions: int
    step_size: float = 0.05
    temperature: float = 1.0
    l2_penalty: float = 0.0
    max_gradient_norm: float | None = None
    min_probability: float = 1e-6
    ratio_clip: float = 10.0
    diagnostic_decay: float = 0.99

    def __post_init__(self) -> None:
        """Validate scalar hyperparameters."""
        if (
            isinstance(self.n_actions, bool)
            or not isinstance(self.n_actions, int)
            or self.n_actions <= 0
        ):
            raise ValueError("n_actions must be positive")
        if not math.isfinite(self.step_size) or self.step_size < 0.0:
            raise ValueError("step_size must be finite and non-negative")
        if not math.isfinite(self.temperature) or self.temperature <= 0.0:
            raise ValueError("temperature must be finite and positive")
        if not math.isfinite(self.l2_penalty) or self.l2_penalty < 0.0:
            raise ValueError("l2_penalty must be finite and non-negative")
        if self.max_gradient_norm is not None and (
            not math.isfinite(self.max_gradient_norm) or self.max_gradient_norm <= 0.0
        ):
            raise ValueError("max_gradient_norm must be positive when provided")
        if not math.isfinite(self.min_probability) or not 0.0 < self.min_probability < 1.0:
            raise ValueError("min_probability must be finite and lie in (0, 1)")
        if not math.isfinite(self.ratio_clip) or self.ratio_clip <= 0.0:
            raise ValueError("ratio_clip must be finite and positive")
        if not math.isfinite(self.diagnostic_decay) or not 0.0 <= self.diagnostic_decay < 1.0:
            raise ValueError("diagnostic_decay must be finite and lie in [0, 1)")

    def to_config(self) -> dict[str, Any]:
        """Serialize configuration to a JSON-compatible dictionary."""
        return dataclasses.asdict(self)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> BehaviorModelConfig:
        """Reconstruct from :meth:`to_config` output."""
        return cls(**config)


@chex.dataclass(frozen=True)
class BehaviorModelState:
    """Immutable state for the behavior/action predictor."""

    weights: Float[Array, "n_actions feature_dim"]
    bias: Float[Array, " n_actions"]
    rng_key: Array
    step_count: Int[Array, ""]
    step_words: UInt[Array, " 2"]
    nll_ema: Float[Array, ""]
    accuracy_ema: Float[Array, ""]
    confidence_ema: Float[Array, ""]


@dataclasses.dataclass(frozen=True)
class BehaviorModelResourceBudget:
    """Exact persistent-state accounting for a configured feature width.

    The byte count covers the arrays in :class:`BehaviorModelState` when
    initialized with JAX's default two-word typed PRNG key. It excludes Python
    objects, compiled executables, and transient update buffers. The model has
    no replay storage.
    """

    feature_dim: int
    n_actions: int
    trainable_float32_scalars: int
    diagnostic_float32_scalars: int
    administrative_int32_scalars: int
    lifetime_counter_uint32_scalars: int
    rng_uint32_scalars: int
    state_nbytes: int
    learned_float32_scalars_touched_per_update: int
    replay_capacity: int

    def to_dict(self) -> dict[str, int]:
        """Return a JSON-compatible resource record."""
        return dataclasses.asdict(self)


@chex.dataclass(frozen=True)
class BehaviorModelInputGradient:
    """Pre-update cross-entropy gradient with respect to input features.

    This result is read-only: computing it does not advance diagnostics, RNG,
    or parameters. ``gradient`` is the derivative of the unfloored softmax
    cross entropy. Probability flooring remains confined to reporting and
    importance-ratio safety.
    """

    logits: Float[Array, " n_actions"]
    probabilities: Float[Array, " n_actions"]
    loss: Float[Array, ""]
    gradient: Float[Array, " feature_dim"]
    gradient_norm: Float[Array, ""]


@chex.dataclass(frozen=True)
class BehaviorModelUpdateResult:
    """Result of one online behavior-model update."""

    state: BehaviorModelState
    logits: Float[Array, " n_actions"]
    probabilities: Float[Array, " n_actions"]
    action_probability: Float[Array, ""]
    log_likelihood: Float[Array, ""]
    loss: Float[Array, ""]
    entropy: Float[Array, ""]
    confidence: Float[Array, ""]
    predicted_action: Int[Array, ""]
    correct: Float[Array, ""]
    pre_step_words: UInt[Array, " 2"]
    post_step_words: UInt[Array, " 2"]
    lifetime_counter_valid: Bool[Array, ""]
    lifetime_capacity_available: Bool[Array, ""]
    update_applied: Bool[Array, ""]


@chex.dataclass(frozen=True)
class BehaviorModelSampleResult:
    """Result of sampling an action from the learned behavior model."""

    state: BehaviorModelState
    action: Int[Array, ""]
    probabilities: Float[Array, " n_actions"]
    action_probability: Float[Array, ""]
    log_likelihood: Float[Array, ""]


@chex.dataclass(frozen=True)
class BehaviorModelArrayResult:
    """Result from scan-based behavior-model learning."""

    state: BehaviorModelState
    probabilities: Float[Array, "num_steps n_actions"]
    action_probabilities: Float[Array, " num_steps"]
    log_likelihoods: Float[Array, " num_steps"]
    losses: Float[Array, " num_steps"]
    entropies: Float[Array, " num_steps"]
    confidences: Float[Array, " num_steps"]
    correct: Float[Array, " num_steps"]


class BehaviorModel:
    """Online softmax model of the behavior policy.

    The model learns from the actually executed action at every step using a
    one-step cross-entropy update.  It is suitable for estimating behavior
    denominators in off-policy ratios and for sampling plausible actions during
    short model-based rollouts.
    """

    def __init__(self, config: BehaviorModelConfig):
        """Initialize the model."""
        self._config = config

    @property
    def config(self) -> BehaviorModelConfig:
        """Behavior-model configuration."""
        return self._config

    def to_config(self) -> dict[str, Any]:
        """Serialize model configuration."""
        return {
            "type": "BehaviorModel",
            "state_schema": BEHAVIOR_MODEL_STATE_SCHEMA,
            "config": self._config.to_config(),
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> BehaviorModel:
        """Reconstruct a behavior model from :meth:`to_config` output."""
        config = dict(config)
        config.pop("type", None)
        state_schema = config.pop("state_schema", BEHAVIOR_MODEL_STATE_SCHEMA)
        if state_schema != BEHAVIOR_MODEL_STATE_SCHEMA:
            raise ValueError(f"Unsupported behavior-model state schema: {state_schema!r}")
        return cls(BehaviorModelConfig.from_config(config["config"]))

    def init(self, feature_dim: int, key: Array) -> BehaviorModelState:
        """Initialize parameters and diagnostics."""
        if isinstance(feature_dim, bool) or not isinstance(feature_dim, int) or feature_dim <= 0:
            raise ValueError("feature_dim must be a positive integer")
        return BehaviorModelState(
            weights=jnp.zeros(
                (self._config.n_actions, feature_dim),
                dtype=jnp.float32,
            ),
            bias=jnp.zeros((self._config.n_actions,), dtype=jnp.float32),
            rng_key=key,
            step_count=jnp.array(0, dtype=jnp.int32),
            step_words=jnp.zeros((2,), dtype=jnp.uint32),
            nll_ema=jnp.array(0.0, dtype=jnp.float32),
            accuracy_ema=jnp.array(0.0, dtype=jnp.float32),
            confidence_ema=jnp.array(0.0, dtype=jnp.float32),
        )

    def resource_budget(self, feature_dim: int) -> BehaviorModelResourceBudget:
        """Return exact fixed-state accounting for ``feature_dim``.

        The implementation initializes ``weights`` and ``bias`` as trainable
        float32 arrays, keeps three float32 diagnostics and one int32 counter,
        and stores a default JAX typed key backed by two uint32 words.
        """
        if isinstance(feature_dim, bool) or not isinstance(feature_dim, int) or feature_dim <= 0:
            raise ValueError("feature_dim must be a positive integer")
        trainable = self._config.n_actions * feature_dim + self._config.n_actions
        diagnostics = 3
        administrative = 1
        lifetime_words = 2
        rng_words = 2
        state_nbytes = 4 * (
            trainable + diagnostics + administrative + lifetime_words + rng_words
        )
        return BehaviorModelResourceBudget(
            feature_dim=feature_dim,
            n_actions=self._config.n_actions,
            trainable_float32_scalars=trainable,
            diagnostic_float32_scalars=diagnostics,
            administrative_int32_scalars=administrative,
            lifetime_counter_uint32_scalars=lifetime_words,
            rng_uint32_scalars=rng_words,
            state_nbytes=state_nbytes,
            learned_float32_scalars_touched_per_update=trainable + diagnostics,
            replay_capacity=0,
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def predict_logits(
        self,
        state: BehaviorModelState,
        observation: Array,
    ) -> Float[Array, " n_actions"]:
        """Predict behavior logits for one feature vector."""
        obs = jnp.asarray(observation, dtype=jnp.float32)
        return state.weights @ obs + state.bias

    @functools.partial(jax.jit, static_argnums=(0,))
    def predict_probabilities(
        self,
        state: BehaviorModelState,
        observation: Array,
    ) -> Float[Array, " n_actions"]:
        """Predict behavior action probabilities for one feature vector."""
        logits = self.predict_logits(state, observation)
        return jax.nn.softmax(logits / self._config.temperature)

    @functools.partial(jax.jit, static_argnums=(0,))
    def action_probability(
        self,
        state: BehaviorModelState,
        observation: Array,
        action: Array,
    ) -> Float[Array, ""]:
        """Return the floor-clipped probability of ``action``."""
        probs = self.predict_probabilities(state, observation)
        return selected_action_probabilities(
            probs,
            action,
            min_probability=self._config.min_probability,
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def action_log_likelihood(
        self,
        state: BehaviorModelState,
        observation: Array,
        action: Array,
    ) -> Float[Array, ""]:
        """Return the floor-clipped log-likelihood of ``action``."""
        return jnp.log(self.action_probability(state, observation, action))

    @functools.partial(jax.jit, static_argnums=(0,))
    def input_loss_gradient(
        self,
        state: BehaviorModelState,
        observation: Array,
        action: Array,
    ) -> BehaviorModelInputGradient:
        """Differentiate the pre-update prediction loss through its features.

        This is the supported causal bridge from partner prediction into a
        trainable state builder. Callers must invoke it before
        :meth:`update`, and before advancing a recurrent state builder to the
        next observation, so the gradient refers to the representation that
        produced the scored prediction.
        """
        cfg = self._config
        obs = jnp.asarray(observation, dtype=jnp.float32)
        action_id = jnp.asarray(action, dtype=jnp.int32)
        logits = state.weights @ obs + state.bias
        scaled_logits = logits / cfg.temperature
        probabilities = jax.nn.softmax(scaled_logits)
        one_hot = jax.nn.one_hot(action_id, cfg.n_actions, dtype=jnp.float32)
        loss = -jnp.sum(one_hot * jax.nn.log_softmax(scaled_logits))
        logit_gradient = (probabilities - one_hot) / cfg.temperature
        gradient = state.weights.T @ logit_gradient
        return BehaviorModelInputGradient(
            logits=logits,
            probabilities=probabilities,
            loss=loss,
            gradient=gradient,
            gradient_norm=jnp.linalg.norm(gradient),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def update(
        self,
        state: BehaviorModelState,
        observation: Array,
        action: Array,
    ) -> BehaviorModelUpdateResult:
        """Update the behavior model from one observed action."""
        cfg = self._config
        proposed_step_words, lifetime_capacity_available = (
            _checked_lifetime_words_increment(state.step_words)
        )
        lifetime_counter_valid = _lifetime_counter_valid(
            state.step_words,
            state.step_count,
        )
        update_available = lifetime_counter_valid & lifetime_capacity_available
        obs = jnp.asarray(observation, dtype=jnp.float32)
        action_id = jnp.asarray(action, dtype=jnp.int32)
        logits = state.weights @ obs + state.bias
        probabilities = jax.nn.softmax(logits / cfg.temperature)
        one_hot = jax.nn.one_hot(action_id, cfg.n_actions, dtype=jnp.float32)

        logit_error = (one_hot - probabilities) / cfg.temperature
        weight_gradient = logit_error[:, None] * obs[None, :]
        bias_gradient = logit_error
        if cfg.l2_penalty > 0.0:
            weight_gradient = weight_gradient - cfg.l2_penalty * state.weights
            bias_gradient = bias_gradient - cfg.l2_penalty * state.bias

        if cfg.max_gradient_norm is not None:
            grad_norm = jnp.sqrt(
                jnp.sum(weight_gradient * weight_gradient) + jnp.sum(bias_gradient * bias_gradient)
            )
            grad_scale = jnp.minimum(
                1.0,
                jnp.asarray(cfg.max_gradient_norm, dtype=jnp.float32)
                / jnp.maximum(grad_norm, 1e-12),
            )
            weight_gradient = grad_scale * weight_gradient
            bias_gradient = grad_scale * bias_gradient

        action_prob = selected_action_probabilities(
            probabilities,
            action_id,
            min_probability=cfg.min_probability,
        )
        log_likelihood = jnp.log(action_prob)
        loss = -log_likelihood
        entropy = -jnp.sum(probabilities * jnp.log(jnp.maximum(probabilities, cfg.min_probability)))
        confidence = jnp.max(probabilities)
        predicted_action = jnp.argmax(probabilities).astype(jnp.int32)
        correct = (predicted_action == action_id).astype(jnp.float32)

        decay = jnp.asarray(cfg.diagnostic_decay, dtype=jnp.float32)
        first = jnp.all(state.step_words == jnp.asarray(0, dtype=jnp.uint32))
        nll_ema = jnp.where(
            first,
            loss,
            decay * state.nll_ema + (1.0 - decay) * loss,
        )
        accuracy_ema = jnp.where(
            first,
            correct,
            decay * state.accuracy_ema + (1.0 - decay) * correct,
        )
        confidence_ema = jnp.where(
            first,
            confidence,
            decay * state.confidence_ema + (1.0 - decay) * confidence,
        )

        proposed_state = state.replace(  # type: ignore[attr-defined]
            weights=state.weights + cfg.step_size * weight_gradient,
            bias=state.bias + cfg.step_size * bias_gradient,
            step_count=_saturating_int32_increment(state.step_count),
            step_words=proposed_step_words,
            nll_ema=nll_ema,
            accuracy_ema=accuracy_ema,
            confidence_ema=confidence_ema,
        )
        new_state = jax.lax.cond(
            update_available,
            lambda _: proposed_state,
            lambda _: state,
            operand=None,
        )
        return BehaviorModelUpdateResult(
            state=new_state,
            logits=logits,
            probabilities=probabilities,
            action_probability=action_prob,
            log_likelihood=log_likelihood,
            loss=loss,
            entropy=entropy,
            confidence=confidence,
            predicted_action=predicted_action,
            correct=correct,
            pre_step_words=state.step_words,
            post_step_words=new_state.step_words,
            lifetime_counter_valid=lifetime_counter_valid,
            lifetime_capacity_available=lifetime_capacity_available,
            update_applied=update_available,
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def sample_action(
        self,
        state: BehaviorModelState,
        observation: Array,
    ) -> BehaviorModelSampleResult:
        """Sample one action from the learned behavior distribution."""
        key, sample_key = jr.split(state.rng_key)
        probabilities = floor_and_renormalize_probabilities(
            self.predict_probabilities(state, observation),
            min_probability=self._config.min_probability,
        )
        action = jr.categorical(
            sample_key,
            jnp.log(probabilities),
        ).astype(jnp.int32)
        action_prob = selected_action_probabilities(
            probabilities,
            action,
            min_probability=self._config.min_probability,
        )
        return BehaviorModelSampleResult(
            state=state.replace(rng_key=key),  # type: ignore[attr-defined]
            action=action,
            probabilities=probabilities,
            action_probability=action_prob,
            log_likelihood=jnp.log(action_prob),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def importance_ratio(
        self,
        state: BehaviorModelState,
        observation: Array,
        action: Array,
        target_probabilities: Array,
    ) -> Float[Array, ""]:
        """Compute a clipped target/behavior ratio for one transition."""
        behavior = self.predict_probabilities(state, observation)
        ratio = clipped_importance_ratios(
            target_probabilities,
            behavior,
            action,
            clip=self._config.ratio_clip,
            min_behavior_probability=self._config.min_probability,
        )
        return ratio


def behavior_model_lifetime_counter_nbytes() -> int:
    """Return bytes occupied by compatibility telemetry plus exact identity."""

    return BEHAVIOR_MODEL_LIFETIME_COUNTER_NBYTES


def measure_behavior_model_state_nbytes(state: BehaviorModelState) -> int:
    """Measure persistent JAX-array bytes in one behavior-model state."""

    return sum(
        int(leaf.size) * int(leaf.dtype.itemsize)
        for leaf in jax.tree.leaves(state)
        if isinstance(leaf, Array)
    )


def migrate_legacy_behavior_model_state(legacy_state: Any) -> BehaviorModelState:
    """Migrate an exact pre-v2 state whose int32 clock is still unambiguous."""

    if isinstance(legacy_state, Mapping):
        fields = dict(legacy_state)
    elif dataclasses.is_dataclass(legacy_state) and not isinstance(legacy_state, type):
        fields = {
            field.name: getattr(legacy_state, field.name)
            for field in dataclasses.fields(legacy_state)
        }
    else:
        raise TypeError("legacy behavior-model state must be a mapping or dataclass")
    current_names = {
        field.name
        for field in dataclasses.fields(cast(Any, BehaviorModelState))
    }
    legacy_names = current_names - {"step_words"}
    supplied_names = set(fields)
    if supplied_names != legacy_names:
        missing = sorted(legacy_names - supplied_names)
        extra = sorted(supplied_names - legacy_names)
        raise ValueError(
            "legacy behavior-model field manifest is not exact; "
            f"missing={missing}, extra={extra}"
        )
    step_count = jnp.asarray(fields["step_count"])
    if step_count.shape != () or step_count.dtype != jnp.dtype(jnp.int32):
        raise TypeError("legacy behavior-model step_count must be scalar int32")
    step = int(step_count)
    if step < 0:
        raise ValueError("negative legacy behavior-model step_count indicates wrap")
    if step >= _INT32_MAX:
        raise ValueError("saturated legacy behavior-model step_count is ambiguous")
    fields["step_words"] = jnp.asarray((0, step), dtype=jnp.uint32)
    return BehaviorModelState(**fields)


def run_behavior_model_from_arrays(
    model: BehaviorModel,
    state: BehaviorModelState,
    observations: Float[Array, "num_steps feature_dim"],
    actions: Int[Array, " num_steps"],
) -> BehaviorModelArrayResult:
    """Run online behavior prediction over arrays with ``jax.lax.scan``."""

    def _scan_fn(
        carry: BehaviorModelState,
        inputs: tuple[Array, Array],
    ) -> tuple[BehaviorModelState, tuple[Array, Array, Array, Array, Array, Array, Array]]:
        obs, action = inputs
        result = model.update(carry, obs, action)
        return result.state, (
            result.probabilities,
            result.action_probability,
            result.log_likelihood,
            result.loss,
            result.entropy,
            result.confidence,
            result.correct,
        )

    (
        final_state,
        (
            probabilities,
            action_probabilities,
            log_likelihoods,
            losses,
            entropies,
            confidences,
            correct,
        ),
    ) = jax.lax.scan(_scan_fn, state, (observations, actions))
    return BehaviorModelArrayResult(
        state=final_state,
        probabilities=probabilities,
        action_probabilities=action_probabilities,
        log_likelihoods=log_likelihoods,
        losses=losses,
        entropies=entropies,
        confidences=confidences,
        correct=correct,
    )


__all__ = [
    "BEHAVIOR_MODEL_LIFETIME_COUNTER_DELTA_NBYTES",
    "BEHAVIOR_MODEL_LIFETIME_COUNTER_NBYTES",
    "BEHAVIOR_MODEL_STATE_SCHEMA",
    "BehaviorModel",
    "BehaviorModelArrayResult",
    "BehaviorModelConfig",
    "BehaviorModelInputGradient",
    "BehaviorModelResourceBudget",
    "BehaviorModelSampleResult",
    "BehaviorModelState",
    "BehaviorModelUpdateResult",
    "action_log_likelihoods",
    "behavior_model_lifetime_counter_nbytes",
    "clipped_importance_ratios",
    "epsilon_greedy_probabilities",
    "floor_and_renormalize_probabilities",
    "measure_behavior_model_state_nbytes",
    "migrate_legacy_behavior_model_state",
    "run_behavior_model_from_arrays",
    "selected_action_probabilities",
]
