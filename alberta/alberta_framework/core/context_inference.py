# mypy: disable-error-code="call-arg"
"""Latent context inference from the experience stream.

Every context-keyed memory result in this repository so far observed the task
context directly: an oracle wrapper appended the active-rule one-hot to the
observation, and exclusive gating on that channel produced the certified
memory mechanism (``tests/test_integrated_life.py``).  This module removes the
oracle.  :class:`ContextInference` maintains a bounded bank of ``max_contexts``
slots, each storing a cheap online model of one reward regime, and infers the
active context from ``(observation, action, reward)`` triples alone.

Mechanism
=========

* **Per-slot regime model** — a linear per-action reward predictor
  ``r_hat = w[slot, action] @ observation`` updated by normalized LMS.  For a
  (near) one-hot observation this is exactly a per-slot EMA table of the
  observed reward for each (state, action) pair — the sufficient statistic of
  a tabular reward rule.  Only the *active* slot's model is updated: inactive
  slots receive zero gradient, so a stored regime model is untouchable while
  another regime is in force (the same exclusive-gating design law as the
  gated Q blocks it feeds).
* **Evidence** — every step, all slots predict the observed reward; each
  in-use slot keeps an EMA of its recent absolute prediction error.  Unused
  slots are pinned at ``novelty_prior_error``, the error a fresh neutral
  model would incur, so "allocate" competes on the same scale as "reuse".
* **Change-point** — when the active slot's error EMA exceeds
  ``switch_threshold`` (and the slot has been active for at least
  ``min_dwell`` steps), the module switches to the in-use slot with the
  lowest recent error; if every stored slot is worse than a fresh model
  (its error EMA above ``novelty_prior_error``), a free slot is allocated —
  or, with all ``max_contexts`` slots in use, the least-recently-active
  non-active slot is evicted and reset (bounded memory).
* **Surprise gate** — the active model is only updated on samples whose
  absolute error is at most ``update_error_gate``.  During the few-step
  detection lag after a regime flip, off-regime samples (error near the
  full reward scale) are therefore rejected instead of corrupting the
  stored model of the outgoing regime.

Defaults are calibrated for {0, 1}-scale rewards over (near) one-hot
observations (the :class:`SwitchingTwoStateMDP` family): a fresh model
initialized at ``initial_reward_estimate = 0.5`` has per-step error 0.5
(= ``novelty_prior_error``), a maximally wrong stored model has error 1.0,
and ``update_error_gate = 0.75`` separates the two.  Measured behavior
(``tests/test_context_inference.py``, 2026-07-30): detection lag 3-4 steps
after a rule flip at ``error_decay = 0.8`` / ``switch_threshold = 0.55``;
previously seen rules are re-identified by slot reuse, not fresh allocation.
Other reward scales require recalibrating the four threshold parameters.

The API is pure-functional: frozen ``chex`` dataclass state and
``update(state, observation, action, reward) -> (state, context_onehot)``,
safe under ``jax.jit``, ``jax.vmap``, and ``jax.lax.scan``.
"""

from __future__ import annotations

import dataclasses
import functools
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import chex
import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Bool, Float, Int, UInt

from alberta_framework.core.checkpoints import (
    load_checkpoint,
    load_checkpoint_metadata,
    save_checkpoint,
)

_INT32_MAX = 2**31 - 1
_UINT32_MAX = 2**32 - 1
_UINT64_MAX = 2**64 - 1
_FLOAT32_MAX = 3.4028234663852886e38

CONTEXT_INFERENCE_STATE_SCHEMA = "alberta.context-inference-state.v2"
CONTEXT_INFERENCE_CHECKPOINT_SCHEMA = "alberta.context-inference-checkpoint.v2"
_LEGACY_CONTEXT_INFERENCE_CHECKPOINT_SCHEMA = "alberta.context-inference-checkpoint.v1"

__all__ = [
    "CONTEXT_INFERENCE_CHECKPOINT_SCHEMA",
    "CONTEXT_INFERENCE_STATE_SCHEMA",
    "ContextInference",
    "ContextInferenceConfig",
    "ContextInferenceResourceBudget",
    "ContextInferenceState",
    "ContextInferencePrioritizedUpdateResult",
    "ContextInferenceUpdateResult",
    "context_inference_clock_nbytes",
    "context_inference_exact_clock_delta_nbytes",
    "load_context_inference_checkpoint",
    "measure_context_inference_state_nbytes",
    "migrate_legacy_context_inference_state",
    "save_context_inference_checkpoint",
]


def _require_array_contract(
    value: Any,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: jnp.dtype,
) -> Array:
    """Require a persistent array's trace-time shape and effective dtype."""

    array = jnp.asarray(value)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if array.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {array.dtype}")
    return array


def _checked_words_increment(
    words: Array,
) -> tuple[UInt[Array, " 2"], Bool[Array, ""]]:
    """Propose one exact increment without committing an all-ones wrap."""

    maximum = jnp.asarray(_UINT32_MAX, dtype=jnp.uint32)
    capacity_available = ~jnp.all(words == maximum)
    low = words[1] + jnp.asarray(1, dtype=jnp.uint32)
    carry = (low == jnp.asarray(0, dtype=jnp.uint32)).astype(jnp.uint32)
    proposed = jnp.stack((words[0] + carry, low)).astype(jnp.uint32)
    return jnp.where(capacity_available, proposed, words), capacity_available


def _words_le(left: Array, right: Array) -> Bool[Array, ...]:
    """Compare big-endian uint32-word identities without enabling x64."""

    return (left[..., 0] < right[..., 0]) | (
        (left[..., 0] == right[..., 0]) & (left[..., 1] <= right[..., 1])
    )


def _words_predecessor(words: Array) -> UInt[Array, " 2"]:
    """Return ``max(words - 1, 0)`` in exact two-word arithmetic."""

    zero = jnp.zeros((2,), dtype=jnp.uint32)
    is_zero = jnp.all(words == zero)
    borrow = (words[1] == jnp.asarray(0, dtype=jnp.uint32)).astype(jnp.uint32)
    predecessor = jnp.stack(
        (
            words[0] - borrow,
            words[1] - jnp.asarray(1, dtype=jnp.uint32),
        )
    ).astype(jnp.uint32)
    return jnp.where(is_zero, zero, predecessor)


def _words_to_int32_telemetry(words: Array) -> Int[Array, ...]:
    """Project exact identities onto saturating non-negative int32 telemetry."""

    fits = (words[..., 0] == jnp.asarray(0, dtype=jnp.uint32)) & (
        words[..., 1] <= jnp.asarray(_INT32_MAX, dtype=jnp.uint32)
    )
    return jnp.where(
        fits,
        words[..., 1].astype(jnp.int32),
        jnp.asarray(_INT32_MAX, dtype=jnp.int32),
    )


def _words_at_least_python_int(words: Array, threshold: int) -> Bool[Array, ""]:
    """Compare an exact identity with a validated Python uint64 threshold."""

    threshold_words = jnp.asarray(
        ((threshold >> 32) & _UINT32_MAX, threshold & _UINT32_MAX),
        dtype=jnp.uint32,
    )
    return _words_le(threshold_words, words)


@chex.dataclass(frozen=True)
class ContextInferenceConfig:
    """Configuration for :class:`ContextInference`.

    Attributes:
        n_actions: Number of discrete actions the reward models condition on.
        observation_dim: Dimension of the observation vector.
        max_contexts: Bounded number of context slots (at least 2).
        model_step_size: Normalized-LMS step size for the active slot's
            reward model.
        error_decay: Per-step decay of each slot's recent absolute-error EMA;
            the effective evidence window is ``1 / (1 - error_decay)`` steps.
        switch_threshold: Change-point trigger — the active slot's error EMA
            above this value starts re-inference.
        novelty_prior_error: Virtual recent error of a fresh (unused) slot;
            allocation wins over reuse only when every stored slot's recent
            error exceeds this value.  Set it to the expected per-step error
            of a model pinned at ``initial_reward_estimate``.
        update_error_gate: Surprise gate — the active model is not updated on
            samples whose absolute error exceeds this value.  Must exceed the
            fresh-model error scale (``novelty_prior_error``) or newly
            allocated slots can never learn.
        min_dwell: Minimum steps a slot stays active before another switch is
            allowed (grace period for a just-adopted model).
        initial_reward_estimate: Initial value of every reward-model weight;
            with one-hot observations this is the fresh model's prediction
            for every (state, action) cell.
    """

    n_actions: int
    observation_dim: int
    max_contexts: int = 4
    model_step_size: float = 0.3
    error_decay: float = 0.8
    switch_threshold: float = 0.55
    novelty_prior_error: float = 0.5
    update_error_gate: float = 0.75
    min_dwell: int = 10
    initial_reward_estimate: float = 0.5

    def __post_init__(self) -> None:
        """Validate scalar hyperparameters."""
        if isinstance(self.n_actions, bool) or not isinstance(self.n_actions, int):
            raise ValueError("n_actions must be an integer")
        if self.n_actions < 1:
            raise ValueError("n_actions must be positive")
        if isinstance(self.observation_dim, bool) or not isinstance(self.observation_dim, int):
            raise ValueError("observation_dim must be an integer")
        if self.observation_dim < 1:
            raise ValueError("observation_dim must be positive")
        if isinstance(self.max_contexts, bool) or not isinstance(self.max_contexts, int):
            raise ValueError("max_contexts must be an integer")
        if self.max_contexts < 2:
            raise ValueError("max_contexts must be at least 2")
        for name, value in (
            ("model_step_size", self.model_step_size),
            ("error_decay", self.error_decay),
            ("switch_threshold", self.switch_threshold),
            ("novelty_prior_error", self.novelty_prior_error),
            ("update_error_gate", self.update_error_gate),
            ("initial_reward_estimate", self.initial_reward_estimate),
        ):
            if not math.isfinite(value) or abs(value) > _FLOAT32_MAX:
                raise ValueError(f"{name} must be finite and representable as float32")
        if self.model_step_size <= 0.0:
            raise ValueError("model_step_size must be positive")
        if not 0.0 <= self.error_decay < 1.0:
            raise ValueError("error_decay must be in [0, 1)")
        if self.switch_threshold <= 0.0:
            raise ValueError("switch_threshold must be positive")
        if self.novelty_prior_error <= 0.0:
            raise ValueError("novelty_prior_error must be positive")
        if self.update_error_gate <= self.novelty_prior_error:
            raise ValueError(
                "update_error_gate must exceed novelty_prior_error, or freshly "
                "allocated slots could never learn"
            )
        if (
            isinstance(self.min_dwell, bool)
            or not isinstance(self.min_dwell, int)
            or not 0 <= self.min_dwell < _UINT64_MAX
        ):
            raise ValueError("min_dwell must be below the terminal uint64 identity")

    def to_config(self) -> dict[str, Any]:
        """Return a strict JSON-compatible configuration manifest."""

        return {"type": type(self).__name__, **dataclasses.asdict(cast(Any, self))}

    @classmethod
    def from_config(cls, payload: Mapping[str, Any]) -> ContextInferenceConfig:
        """Reconstruct only an exact :meth:`to_config` field manifest."""

        values = dict(payload)
        expected = {field.name for field in dataclasses.fields(cast(Any, cls))} | {"type"}
        if set(values) != expected:
            raise ValueError("context-inference config fields do not match the schema")
        type_name = values.pop("type")
        if type_name != cls.__name__:
            raise ValueError(f"unexpected context-inference config type: {type_name!r}")
        return cls(**values)


@dataclasses.dataclass(frozen=True)
class ContextInferenceResourceBudget:
    """Exact persistent-array and clock accounting for one fixed slot bank."""

    allocated_float32_scalars: int
    allocated_bool_scalars: int
    allocated_int32_scalars: int
    allocated_uint32_scalars: int
    state_nbytes: int
    clock_nbytes: int
    exact_clock_delta_nbytes: int
    max_contexts: int
    replay_capacity: int = 0

    def to_dict(self) -> dict[str, int]:
        """Return a JSON-compatible resource record."""

        return dataclasses.asdict(self)


@chex.dataclass(frozen=True)
class ContextInferenceState:
    """Immutable state of :class:`ContextInference`.

    Attributes:
        reward_weights: Per-slot per-action linear reward models, shape
            ``(max_contexts, n_actions, observation_dim)``.  Only the active
            slot's rows are ever updated.
        error_ema: Recent absolute prediction error per slot; unused slots
            are pinned at ``novelty_prior_error``.
        in_use: Which slots hold an allocated regime model.
        active_context: Currently inferred context slot.
        last_active_step: Step count at which each slot was last active
            (``-1`` for never); saturating compatibility telemetry only.
        dwell: Saturating compatibility telemetry for steps since the last
            context switch.
        step_count: Saturating compatibility telemetry for committed updates.
        last_active_words: Exact big-endian uint32-word recency identities per
            slot.  Only rows selected by ``in_use`` have semantic meaning.
        dwell_words: Exact big-endian uint32-word dwell authority.
        step_words: Exact big-endian uint32-word lifetime identity.  The
            all-ones value is terminal and can never be incremented.
    """

    reward_weights: Float[Array, "max_contexts n_actions observation_dim"]
    error_ema: Float[Array, " max_contexts"]
    in_use: Bool[Array, " max_contexts"]
    active_context: Int[Array, ""]
    last_active_step: Int[Array, " max_contexts"]
    dwell: Int[Array, ""]
    step_count: Int[Array, ""]
    last_active_words: UInt[Array, "max_contexts 2"]
    dwell_words: UInt[Array, " 2"]
    step_words: UInt[Array, " 2"]


@chex.dataclass(frozen=True)
class ContextInferenceUpdateResult:
    """Transactional update, exact-clock status, and returned context."""

    state: ContextInferenceState
    context_onehot: Float[Array, " max_contexts"]
    pre_step_words: UInt[Array, " 2"]
    post_step_words: UInt[Array, " 2"]
    source_state_valid: Bool[Array, ""]
    input_valid: Bool[Array, ""]
    candidate_state_valid: Bool[Array, ""]
    lifetime_capacity_available: Bool[Array, ""]
    update_applied: Bool[Array, ""]


@chex.dataclass(frozen=True)
class ContextInferencePrioritizedUpdateResult:
    """Explicit defaults-off full-bank eviction-protection result.

    ``eviction_protection`` never changes stored-context reuse or allocation
    into a free slot.  It is consulted only when an otherwise-valid change
    point requests a fresh semantic birth while every slot is occupied.  The
    least-protected eligible slot is selected, with the ordinary exact LRU
    order retained as the deterministic tie-break.
    """

    state: ContextInferenceState
    context_onehot: Float[Array, " max_contexts"]
    eviction_protection: Float[Array, " max_contexts"]
    pre_step_words: UInt[Array, " 2"]
    post_step_words: UInt[Array, " 2"]
    source_state_valid: Bool[Array, ""]
    input_valid: Bool[Array, ""]
    eviction_protection_input_valid: Bool[Array, ""]
    candidate_state_valid: Bool[Array, ""]
    lifetime_capacity_available: Bool[Array, ""]
    allocation_requested: Bool[Array, ""]
    full_bank_eviction_requested: Bool[Array, ""]
    ordinary_lru_slot: Int[Array, ""]
    protected_lru_slot: Int[Array, ""]
    selected_eviction_slot: Int[Array, ""]
    eviction_protection_used: Bool[Array, ""]
    eviction_target_adjusted: Bool[Array, ""]
    update_applied: Bool[Array, ""]


class ContextInference:
    """Bounded-slot latent context inference over an experience stream.

    Each slot stores a cheap online model of one reward regime; the active
    slot is the one whose stored model best explains recent rewards.  See the
    module docstring for the full mechanism and calibration notes.

    Args:
        config: Immutable configuration.
    """

    def __init__(self, config: ContextInferenceConfig):
        """Store the configuration."""
        self._config = config

    @property
    def config(self) -> ContextInferenceConfig:
        """The immutable configuration."""
        return self._config

    @property
    def resource_budget(self) -> ContextInferenceResourceBudget:
        """Return exact static storage accounting for this slot-bank shape."""

        cfg = self._config
        float_scalars = cfg.max_contexts * cfg.n_actions * cfg.observation_dim + cfg.max_contexts
        bool_scalars = cfg.max_contexts
        int_scalars = cfg.max_contexts + 3
        uint_scalars = 2 * cfg.max_contexts + 4
        state_nbytes = (
            4 * float_scalars
            + bool_scalars
            + 4 * int_scalars
            + 4 * uint_scalars
        )
        return ContextInferenceResourceBudget(
            allocated_float32_scalars=float_scalars,
            allocated_bool_scalars=bool_scalars,
            allocated_int32_scalars=int_scalars,
            allocated_uint32_scalars=uint_scalars,
            state_nbytes=state_nbytes,
            clock_nbytes=context_inference_clock_nbytes(cfg.max_contexts),
            exact_clock_delta_nbytes=context_inference_exact_clock_delta_nbytes(
                cfg.max_contexts
            ),
            max_contexts=cfg.max_contexts,
        )

    def to_config(self) -> dict[str, Any]:
        """Serialize the mechanism and exact state schema without learned state."""

        return {
            "type": type(self).__name__,
            "state_schema": CONTEXT_INFERENCE_STATE_SCHEMA,
            "config": self._config.to_config(),
        }

    @classmethod
    def from_config(cls, payload: Mapping[str, Any]) -> ContextInference:
        """Strictly reconstruct one mechanism manifest."""

        values = dict(payload)
        if set(values) != {"type", "state_schema", "config"}:
            raise ValueError("context-inference module manifest is not exact")
        if values["type"] != cls.__name__:
            raise ValueError(f"unexpected context-inference type: {values['type']!r}")
        if values["state_schema"] != CONTEXT_INFERENCE_STATE_SCHEMA:
            raise ValueError("context-inference state schema is unsupported")
        config = values["config"]
        if not isinstance(config, Mapping):
            raise ValueError("context-inference config must be a mapping")
        return cls(ContextInferenceConfig.from_config(config))

    def init(self) -> ContextInferenceState:
        """Create the birth state: slot 0 active and allocated, others free."""
        cfg = self._config
        k = cfg.max_contexts
        return ContextInferenceState(
            reward_weights=jnp.full(
                (k, cfg.n_actions, cfg.observation_dim),
                cfg.initial_reward_estimate,
                dtype=jnp.float32,
            ),
            error_ema=jnp.full((k,), cfg.novelty_prior_error, dtype=jnp.float32).at[0].set(0.0),
            in_use=jnp.zeros((k,), dtype=bool).at[0].set(True),
            active_context=jnp.array(0, dtype=jnp.int32),
            last_active_step=jnp.full((k,), -1, dtype=jnp.int32).at[0].set(0),
            dwell=jnp.array(0, dtype=jnp.int32),
            step_count=jnp.array(0, dtype=jnp.int32),
            last_active_words=jnp.zeros((k, 2), dtype=jnp.uint32),
            dwell_words=jnp.zeros((2,), dtype=jnp.uint32),
            step_words=jnp.zeros((2,), dtype=jnp.uint32),
        )

    def _require_state_contract(self, state: ContextInferenceState) -> None:
        """Require every persistent field's static shape and effective dtype."""

        cfg = self._config
        k = cfg.max_contexts
        _require_array_contract(
            state.reward_weights,
            name="state.reward_weights",
            shape=(k, cfg.n_actions, cfg.observation_dim),
            dtype=jnp.dtype(jnp.float32),
        )
        _require_array_contract(
            state.error_ema,
            name="state.error_ema",
            shape=(k,),
            dtype=jnp.dtype(jnp.float32),
        )
        _require_array_contract(
            state.in_use,
            name="state.in_use",
            shape=(k,),
            dtype=jnp.dtype(jnp.bool_),
        )
        _require_array_contract(
            state.active_context,
            name="state.active_context",
            shape=(),
            dtype=jnp.dtype(jnp.int32),
        )
        _require_array_contract(
            state.last_active_step,
            name="state.last_active_step",
            shape=(k,),
            dtype=jnp.dtype(jnp.int32),
        )
        _require_array_contract(
            state.dwell,
            name="state.dwell",
            shape=(),
            dtype=jnp.dtype(jnp.int32),
        )
        _require_array_contract(
            state.step_count,
            name="state.step_count",
            shape=(),
            dtype=jnp.dtype(jnp.int32),
        )
        _require_array_contract(
            state.last_active_words,
            name="state.last_active_words",
            shape=(k, 2),
            dtype=jnp.dtype(jnp.uint32),
        )
        _require_array_contract(
            state.dwell_words,
            name="state.dwell_words",
            shape=(2,),
            dtype=jnp.dtype(jnp.uint32),
        )
        _require_array_contract(
            state.step_words,
            name="state.step_words",
            shape=(2,),
            dtype=jnp.dtype(jnp.uint32),
        )

    def state_is_valid(self, state: ContextInferenceState) -> Bool[Array, ""]:
        """Authenticate dynamic values and all exact/telemetry relationships."""

        self._require_state_contract(state)
        cfg = self._config
        k = cfg.max_contexts
        safe_active = jnp.clip(state.active_context, 0, k - 1)
        active_in_range = (state.active_context >= 0) & (state.active_context < k)
        expected_last_telemetry = jnp.where(
            state.in_use,
            _words_to_int32_telemetry(state.last_active_words),
            jnp.asarray(-1, dtype=jnp.int32),
        )
        unused_words_zero = jnp.all(
            jnp.where(
                state.in_use[:, None],
                jnp.asarray(True),
                state.last_active_words == jnp.asarray(0, dtype=jnp.uint32),
            )
        )
        active_expected_stamp = _words_predecessor(state.step_words)
        active_stamp_valid = jnp.all(
            state.last_active_words[safe_active] == active_expected_stamp
        )
        unused_errors_pinned = jnp.all(
            jnp.where(
                state.in_use,
                jnp.asarray(True),
                state.error_ema == jnp.float32(cfg.novelty_prior_error),
            )
        )
        return (
            jnp.all(jnp.isfinite(state.reward_weights))
            & jnp.all(jnp.isfinite(state.error_ema))
            & jnp.all(state.error_ema >= 0.0)
            & jnp.any(state.in_use)
            & active_in_range
            & state.in_use[safe_active]
            & (state.step_count == _words_to_int32_telemetry(state.step_words))
            & (state.dwell == _words_to_int32_telemetry(state.dwell_words))
            & jnp.all(state.last_active_step == expected_last_telemetry)
            & _words_le(state.dwell_words, state.step_words)
            & jnp.all(_words_le(state.last_active_words, active_expected_stamp))
            & unused_words_zero
            & active_stamp_valid
            & unused_errors_pinned
        )

    def context_onehot(self, state: ContextInferenceState) -> Float[Array, " max_contexts"]:
        """One-hot channels of the currently inferred context."""
        return jax.nn.one_hot(state.active_context, self._config.max_contexts, dtype=jnp.float32)

    def num_contexts_in_use(self, state: ContextInferenceState) -> Int[Array, ""]:
        """Number of allocated context slots."""
        return jnp.sum(state.in_use).astype(jnp.int32)

    @functools.partial(jax.jit, static_argnums=(0,))
    def update_result(
        self,
        state: ContextInferenceState,
        observation: Array,
        action: Array,
        reward: Array,
    ) -> ContextInferenceUpdateResult:
        """Transactionally infer one context and refine its active model.

        Invalid source state, invalid input, non-finite proposed state, or an
        exhausted all-ones lifetime identity returns the source bit-for-bit.
        """
        self._require_state_contract(state)
        cfg = self._config
        k = cfg.max_contexts
        obs = jnp.asarray(observation, dtype=jnp.float32)
        if obs.shape != (cfg.observation_dim,):
            raise ValueError(
                f"observation must have shape {(cfg.observation_dim,)}, got {obs.shape}"
            )
        action_index = jnp.squeeze(jnp.asarray(action, dtype=jnp.int32))
        if action_index.shape != ():
            raise ValueError("action must be scalar after squeezing")
        reward_s = jnp.squeeze(jnp.asarray(reward, dtype=jnp.float32))
        if reward_s.shape != ():
            raise ValueError("reward must be scalar after squeezing")

        source_state_valid = self.state_is_valid(state)
        input_valid = (
            jnp.all(jnp.isfinite(obs))
            & jnp.isfinite(reward_s)
            & (action_index >= 0)
            & (action_index < cfg.n_actions)
        )
        safe_obs = jnp.where(jnp.isfinite(obs), obs, jnp.zeros_like(obs))
        safe_reward = jnp.where(jnp.isfinite(reward_s), reward_s, jnp.float32(0.0))
        safe_action = jnp.clip(action_index, 0, cfg.n_actions - 1)
        safe_active = jnp.clip(state.active_context, 0, k - 1)
        proposed_step_words, lifetime_capacity_available = _checked_words_increment(
            state.step_words
        )
        proposed_dwell_words, dwell_capacity_available = _checked_words_increment(
            state.dwell_words
        )

        # 1. Evidence: every slot predicts the observed reward; in-use slots
        #    track a recent-|error| EMA, unused slots sit at the fresh-model
        #    prior so reuse and allocation compete on one scale.
        predictions = state.reward_weights[:, safe_action, :] @ safe_obs
        errors = jnp.abs(safe_reward - predictions)
        decay = jnp.float32(cfg.error_decay)
        error_ema = decay * state.error_ema + (1.0 - decay) * errors
        error_ema = jnp.where(state.in_use, error_ema, jnp.float32(cfg.novelty_prior_error))

        # 2. Change-point: leave a misfitting active slot for the best
        #    alternative — a stored regime if one explains recent rewards,
        #    else a fresh slot (evicting the least-recently-active slot when
        #    all are in use).
        active = safe_active
        slot_ids = jnp.arange(k, dtype=jnp.int32)
        active_mask = slot_ids == active
        stored_scores = jnp.where(active_mask | ~state.in_use, jnp.inf, error_ema)
        best_stored = jnp.argmin(stored_scores).astype(jnp.int32)
        best_stored_error = stored_scores[best_stored]
        allocate = best_stored_error > cfg.novelty_prior_error
        free_exists = jnp.any(~state.in_use)
        free_mask = ~state.in_use
        first_free = jnp.argmax(free_mask).astype(jnp.int32)

        # LRU is the lexicographic minimum exact timestamp.  ``argmax`` over
        # the final equality mask makes a timestamp tie deterministic by the
        # lowest slot id, rather than by saturated int32 telemetry.
        eligible_lru = state.in_use & ~active_mask
        maximum_word = jnp.asarray(_UINT32_MAX, dtype=jnp.uint32)
        eligible_high = jnp.where(
            eligible_lru,
            state.last_active_words[:, 0],
            maximum_word,
        )
        minimum_high = jnp.min(eligible_high)
        eligible_low = jnp.where(
            eligible_lru & (state.last_active_words[:, 0] == minimum_high),
            state.last_active_words[:, 1],
            maximum_word,
        )
        minimum_low = jnp.min(eligible_low)
        oldest = (
            eligible_lru
            & (state.last_active_words[:, 0] == minimum_high)
            & (state.last_active_words[:, 1] == minimum_low)
        )
        lru_slot = jnp.argmax(oldest).astype(jnp.int32)
        fresh_slot = jnp.where(free_exists, first_free, lru_slot)
        target = jnp.where(allocate, fresh_slot, best_stored).astype(jnp.int32)
        target_error = jnp.minimum(best_stored_error, jnp.float32(cfg.novelty_prior_error))
        do_switch = (
            (error_ema[active] > cfg.switch_threshold)
            & _words_at_least_python_int(state.dwell_words, cfg.min_dwell)
            & (target_error < error_ema[active])
        )
        did_allocate = do_switch & allocate
        new_active = jnp.where(do_switch, target, active).astype(jnp.int32)

        reward_weights = jnp.where(
            did_allocate,
            state.reward_weights.at[target].set(jnp.float32(cfg.initial_reward_estimate)),
            state.reward_weights,
        )
        in_use = jnp.where(did_allocate, state.in_use.at[target].set(True), state.in_use)
        error_ema = jnp.where(do_switch, error_ema.at[target].set(0.0), error_ema)

        # 3. Surprise-gated normalized-LMS update of the active model only:
        #    inactive slots get exactly zero gradient (they ARE the memory),
        #    and off-regime samples during the detection lag are rejected.
        prediction = reward_weights[new_active, safe_action, :] @ safe_obs
        model_error = safe_reward - prediction
        gate = (jnp.abs(model_error) <= cfg.update_error_gate).astype(jnp.float32)
        norm = jnp.maximum(safe_obs @ safe_obs, 1e-8)
        reward_weights = reward_weights.at[new_active, safe_action, :].add(
            jnp.float32(cfg.model_step_size) * gate * model_error * safe_obs / norm
        )

        next_dwell_words = jnp.where(
            do_switch,
            jnp.zeros((2,), dtype=jnp.uint32),
            proposed_dwell_words,
        )
        next_last_active_words = state.last_active_words.at[new_active].set(
            state.step_words
        )
        candidate_state = ContextInferenceState(
            reward_weights=reward_weights,
            error_ema=error_ema,
            in_use=in_use,
            active_context=new_active,
            last_active_step=state.last_active_step.at[new_active].set(
                _words_to_int32_telemetry(state.step_words)
            ),
            dwell=_words_to_int32_telemetry(next_dwell_words),
            step_count=_words_to_int32_telemetry(proposed_step_words),
            last_active_words=next_last_active_words,
            dwell_words=next_dwell_words,
            step_words=proposed_step_words,
        )
        candidate_state_valid = self.state_is_valid(candidate_state)
        update_applied = (
            source_state_valid
            & input_valid
            & lifetime_capacity_available
            & dwell_capacity_available
            & candidate_state_valid
        )
        new_state = jax.tree_util.tree_map(
            lambda proposed, current: jnp.where(update_applied, proposed, current),
            candidate_state,
            state,
        )
        return ContextInferenceUpdateResult(
            state=new_state,
            context_onehot=jax.nn.one_hot(
                new_state.active_context,
                k,
                dtype=jnp.float32,
            ),
            pre_step_words=state.step_words,
            post_step_words=new_state.step_words,
            source_state_valid=source_state_valid,
            input_valid=input_valid,
            candidate_state_valid=candidate_state_valid,
            lifetime_capacity_available=lifetime_capacity_available,
            update_applied=update_applied,
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def update_result_with_eviction_protection(
        self,
        state: ContextInferenceState,
        observation: Array,
        action: Array,
        reward: Array,
        eviction_protection: Array,
    ) -> ContextInferencePrioritizedUpdateResult:
        """Apply explicit full-bank eviction protection.

        The ordinary :meth:`update_result` remains the defaults-off surface
        and is intentionally unchanged.  This sibling accepts one finite,
        non-negative float32 protection score per slot.  Scores are ignored
        for stored-context reuse and free-slot allocation.  On a full-bank
        fresh allocation, the least-protected non-active slot is chosen; the
        ordinary exact LRU order breaks equal-score ties.  Invalid scores roll
        the complete context update back bit-for-bit.
        """

        self._require_state_contract(state)
        cfg = self._config
        k = cfg.max_contexts
        obs = jnp.asarray(observation, dtype=jnp.float32)
        if obs.shape != (cfg.observation_dim,):
            raise ValueError(
                f"observation must have shape {(cfg.observation_dim,)}, got {obs.shape}"
            )
        action_index = jnp.squeeze(jnp.asarray(action, dtype=jnp.int32))
        if action_index.shape != ():
            raise ValueError("action must be scalar after squeezing")
        reward_s = jnp.squeeze(jnp.asarray(reward, dtype=jnp.float32))
        if reward_s.shape != ():
            raise ValueError("reward must be scalar after squeezing")
        raw_protection = jnp.asarray(eviction_protection)
        if raw_protection.shape != (k,):
            raise ValueError(f"eviction_protection must have shape {(k,)}")
        if raw_protection.dtype != jnp.dtype(jnp.float32):
            raise TypeError("eviction_protection must have dtype float32")

        source_state_valid = self.state_is_valid(state)
        protection_input_valid = jnp.all(jnp.isfinite(raw_protection)) & jnp.all(
            raw_protection >= 0.0
        )
        input_valid = (
            jnp.all(jnp.isfinite(obs))
            & jnp.isfinite(reward_s)
            & (action_index >= 0)
            & (action_index < cfg.n_actions)
            & protection_input_valid
        )
        safe_obs = jnp.where(jnp.isfinite(obs), obs, jnp.zeros_like(obs))
        safe_reward = jnp.where(jnp.isfinite(reward_s), reward_s, jnp.float32(0.0))
        safe_action = jnp.clip(action_index, 0, cfg.n_actions - 1)
        safe_protection = jnp.where(
            jnp.isfinite(raw_protection) & (raw_protection >= 0.0),
            raw_protection,
            jnp.zeros_like(raw_protection),
        )
        safe_active = jnp.clip(state.active_context, 0, k - 1)
        proposed_step_words, lifetime_capacity_available = _checked_words_increment(
            state.step_words
        )
        proposed_dwell_words, dwell_capacity_available = _checked_words_increment(
            state.dwell_words
        )

        predictions = state.reward_weights[:, safe_action, :] @ safe_obs
        errors = jnp.abs(safe_reward - predictions)
        decay = jnp.float32(cfg.error_decay)
        error_ema = decay * state.error_ema + (1.0 - decay) * errors
        error_ema = jnp.where(
            state.in_use,
            error_ema,
            jnp.float32(cfg.novelty_prior_error),
        )

        active = safe_active
        slot_ids = jnp.arange(k, dtype=jnp.int32)
        active_mask = slot_ids == active
        stored_scores = jnp.where(active_mask | ~state.in_use, jnp.inf, error_ema)
        best_stored = jnp.argmin(stored_scores).astype(jnp.int32)
        best_stored_error = stored_scores[best_stored]
        allocate = best_stored_error > cfg.novelty_prior_error
        free_exists = jnp.any(~state.in_use)
        first_free = jnp.argmax(~state.in_use).astype(jnp.int32)

        eligible_lru = state.in_use & ~active_mask
        maximum_word = jnp.asarray(_UINT32_MAX, dtype=jnp.uint32)

        def exact_lru(eligible: Array) -> Array:
            eligible_high = jnp.where(
                eligible,
                state.last_active_words[:, 0],
                maximum_word,
            )
            minimum_high = jnp.min(eligible_high)
            eligible_low = jnp.where(
                eligible & (state.last_active_words[:, 0] == minimum_high),
                state.last_active_words[:, 1],
                maximum_word,
            )
            minimum_low = jnp.min(eligible_low)
            oldest = (
                eligible
                & (state.last_active_words[:, 0] == minimum_high)
                & (state.last_active_words[:, 1] == minimum_low)
            )
            return jnp.argmax(oldest).astype(jnp.int32)

        ordinary_lru_slot = exact_lru(eligible_lru)
        eligible_protection = jnp.where(eligible_lru, safe_protection, jnp.inf)
        minimum_protection = jnp.min(eligible_protection)
        least_protected = eligible_lru & (safe_protection == minimum_protection)
        protected_lru_slot = exact_lru(least_protected)
        fresh_slot = jnp.where(free_exists, first_free, protected_lru_slot)
        target = jnp.where(allocate, fresh_slot, best_stored).astype(jnp.int32)
        target_error = jnp.minimum(
            best_stored_error,
            jnp.float32(cfg.novelty_prior_error),
        )
        do_switch = (
            (error_ema[active] > cfg.switch_threshold)
            & _words_at_least_python_int(state.dwell_words, cfg.min_dwell)
            & (target_error < error_ema[active])
        )
        did_allocate = do_switch & allocate
        full_bank_eviction_requested = did_allocate & ~free_exists
        new_active = jnp.where(do_switch, target, active).astype(jnp.int32)

        reward_weights = jnp.where(
            did_allocate,
            state.reward_weights.at[target].set(jnp.float32(cfg.initial_reward_estimate)),
            state.reward_weights,
        )
        in_use = jnp.where(did_allocate, state.in_use.at[target].set(True), state.in_use)
        error_ema = jnp.where(do_switch, error_ema.at[target].set(0.0), error_ema)

        prediction = reward_weights[new_active, safe_action, :] @ safe_obs
        model_error = safe_reward - prediction
        gate = (jnp.abs(model_error) <= cfg.update_error_gate).astype(jnp.float32)
        norm = jnp.maximum(safe_obs @ safe_obs, 1e-8)
        reward_weights = reward_weights.at[new_active, safe_action, :].add(
            jnp.float32(cfg.model_step_size) * gate * model_error * safe_obs / norm
        )

        next_dwell_words = jnp.where(
            do_switch,
            jnp.zeros((2,), dtype=jnp.uint32),
            proposed_dwell_words,
        )
        next_last_active_words = state.last_active_words.at[new_active].set(
            state.step_words
        )
        candidate_state = ContextInferenceState(
            reward_weights=reward_weights,
            error_ema=error_ema,
            in_use=in_use,
            active_context=new_active,
            last_active_step=state.last_active_step.at[new_active].set(
                _words_to_int32_telemetry(state.step_words)
            ),
            dwell=_words_to_int32_telemetry(next_dwell_words),
            step_count=_words_to_int32_telemetry(proposed_step_words),
            last_active_words=next_last_active_words,
            dwell_words=next_dwell_words,
            step_words=proposed_step_words,
        )
        candidate_state_valid = self.state_is_valid(candidate_state)
        update_applied = (
            source_state_valid
            & input_valid
            & lifetime_capacity_available
            & dwell_capacity_available
            & candidate_state_valid
        )
        new_state = jax.tree_util.tree_map(
            lambda proposed, current: jnp.where(update_applied, proposed, current),
            candidate_state,
            state,
        )
        protection_used = update_applied & full_bank_eviction_requested
        target_adjusted = protection_used & (
            protected_lru_slot != ordinary_lru_slot
        )
        return ContextInferencePrioritizedUpdateResult(
            state=new_state,
            context_onehot=jax.nn.one_hot(
                new_state.active_context,
                k,
                dtype=jnp.float32,
            ),
            eviction_protection=safe_protection,
            pre_step_words=state.step_words,
            post_step_words=new_state.step_words,
            source_state_valid=source_state_valid,
            input_valid=input_valid,
            eviction_protection_input_valid=protection_input_valid,
            candidate_state_valid=candidate_state_valid,
            lifetime_capacity_available=lifetime_capacity_available,
            allocation_requested=update_applied & did_allocate,
            full_bank_eviction_requested=protection_used,
            ordinary_lru_slot=ordinary_lru_slot,
            protected_lru_slot=protected_lru_slot,
            selected_eviction_slot=jnp.where(
                protection_used,
                protected_lru_slot,
                jnp.asarray(-1, dtype=jnp.int32),
            ),
            eviction_protection_used=protection_used,
            eviction_target_adjusted=target_adjusted,
            update_applied=update_applied,
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def update(
        self,
        state: ContextInferenceState,
        observation: Array,
        action: Array,
        reward: Array,
    ) -> tuple[ContextInferenceState, Float[Array, " max_contexts"]]:
        """Preserve the historical tuple API around :meth:`update_result`."""

        result = self.update_result(state, observation, action, reward)
        return result.state, result.context_onehot


def context_inference_exact_clock_delta_nbytes(max_contexts: int) -> int:
    """Return bytes added by exact lifetime, dwell, and recency authorities."""

    if isinstance(max_contexts, bool) or not isinstance(max_contexts, int) or max_contexts < 2:
        raise ValueError("max_contexts must be an integer >= 2")
    return 8 * (max_contexts + 2)


def context_inference_clock_nbytes(max_contexts: int) -> int:
    """Return compatibility telemetry plus all exact authority bytes."""

    if isinstance(max_contexts, bool) or not isinstance(max_contexts, int) or max_contexts < 2:
        raise ValueError("max_contexts must be an integer >= 2")
    return 12 * (max_contexts + 2)


def measure_context_inference_state_nbytes(state: ContextInferenceState) -> int:
    """Measure persistent JAX-array bytes in one context-inference state."""

    return sum(
        int(leaf.size) * int(leaf.dtype.itemsize)
        for leaf in jax.tree.leaves(state)
        if isinstance(leaf, Array)
    )


def migrate_legacy_context_inference_state(
    legacy_state: Any,
    *,
    config: ContextInferenceConfig,
) -> ContextInferenceState:
    """Migrate only an exact, unsaturated pre-v2 counter history.

    Signed saturation cannot distinguish one genuine terminal int32 value
    from arbitrarily many later events, so no saturated clock or timestamp is
    inferred.  Such a lifecycle must restart under a fresh namespace.
    """

    if isinstance(legacy_state, Mapping):
        fields = dict(legacy_state)
    elif dataclasses.is_dataclass(legacy_state) and not isinstance(legacy_state, type):
        fields = {
            field.name: getattr(legacy_state, field.name)
            for field in dataclasses.fields(legacy_state)
        }
    else:
        raise TypeError("legacy context-inference state must be a mapping or dataclass")
    current_names = {
        field.name
        for field in dataclasses.fields(cast(Any, ContextInferenceState))
    }
    exact_names = {"step_words", "dwell_words", "last_active_words"}
    legacy_names = current_names - exact_names
    if set(fields) != legacy_names:
        missing = sorted(legacy_names - set(fields))
        extra = sorted(set(fields) - legacy_names)
        raise ValueError(
            "legacy context-inference field manifest is not exact; "
            f"missing={missing}, extra={extra}"
        )

    def scalar_counter(name: str) -> int:
        value = jnp.asarray(fields[name])
        if value.shape != () or value.dtype != jnp.dtype(jnp.int32):
            raise TypeError(f"legacy context-inference {name} must be scalar int32")
        count = int(value)
        if count < 0:
            raise ValueError(f"negative legacy context-inference {name} indicates wrap")
        if count >= _INT32_MAX:
            raise ValueError(f"saturated legacy context-inference {name} is ambiguous")
        return count

    step = scalar_counter("step_count")
    dwell = scalar_counter("dwell")
    if dwell > step:
        raise ValueError("legacy context-inference dwell exceeds step_count")
    in_use = jnp.asarray(fields["in_use"])
    active = jnp.asarray(fields["active_context"])
    last = jnp.asarray(fields["last_active_step"])
    if in_use.shape != (config.max_contexts,) or in_use.dtype != jnp.dtype(jnp.bool_):
        raise TypeError("legacy context-inference in_use has invalid shape or dtype")
    if active.shape != () or active.dtype != jnp.dtype(jnp.int32):
        raise TypeError("legacy context-inference active_context must be scalar int32")
    active_id = int(active)
    if not 0 <= active_id < config.max_contexts or not bool(in_use[active_id]):
        raise ValueError("legacy context-inference active context is not allocated")
    if last.shape != (config.max_contexts,) or last.dtype != jnp.dtype(jnp.int32):
        raise TypeError("legacy context-inference last_active_step has invalid shape or dtype")
    last_host = [int(value) for value in last]
    for slot, (timestamp, used) in enumerate(zip(last_host, in_use.tolist(), strict=True)):
        if not used:
            if timestamp != -1:
                raise ValueError("unused legacy context slot must have last_active_step -1")
            continue
        if timestamp < 0:
            raise ValueError("negative used-slot recency indicates legacy wrap")
        if timestamp >= _INT32_MAX:
            raise ValueError("saturated legacy context recency is ambiguous")
        if timestamp > step:
            raise ValueError("legacy context recency exceeds step_count")
    expected_active_timestamp = max(step - 1, 0)
    if last_host[active_id] != expected_active_timestamp:
        raise ValueError("legacy active-context recency is not aligned with step_count")

    last_words = jnp.zeros((config.max_contexts, 2), dtype=jnp.uint32)
    for slot, used in enumerate(in_use.tolist()):
        if used:
            last_words = last_words.at[slot, 1].set(
                jnp.asarray(last_host[slot], dtype=jnp.uint32)
            )
    fields["last_active_words"] = last_words
    fields["dwell_words"] = jnp.asarray((0, dwell), dtype=jnp.uint32)
    fields["step_words"] = jnp.asarray((0, step), dtype=jnp.uint32)
    migrated = ContextInferenceState(**fields)
    module = ContextInference(config)
    module._require_state_contract(migrated)
    if not bool(jax.device_get(module.state_is_valid(migrated))):
        raise ValueError("legacy context-inference state violates the v2 state contract")
    return migrated


def save_context_inference_checkpoint(
    module: ContextInference,
    state: ContextInferenceState,
    path: str | Path,
) -> None:
    """Persist one structurally and dynamically authenticated v2 state."""

    module._require_state_contract(state)
    if not bool(jax.device_get(module.state_is_valid(state))):
        raise ValueError("context-inference checkpoint state is invalid")
    if measure_context_inference_state_nbytes(state) != module.resource_budget.state_nbytes:
        raise ValueError("context-inference state violates its resource contract")
    save_checkpoint(
        state,
        path,
        metadata={
            "schema": CONTEXT_INFERENCE_CHECKPOINT_SCHEMA,
            "module_config": module.to_config(),
            "memory_accounting": module.resource_budget.to_dict(),
        },
    )


def load_context_inference_checkpoint(
    path: str | Path,
) -> tuple[ContextInference, ContextInferenceState]:
    """Restore only an authenticated exact-clock v2 checkpoint."""

    metadata = load_checkpoint_metadata(path)
    expected_fields = {"schema", "module_config", "memory_accounting"}
    if set(metadata) != expected_fields:
        raise ValueError("context-inference checkpoint metadata fields are invalid")
    schema = metadata.get("schema")
    if schema == _LEGACY_CONTEXT_INFERENCE_CHECKPOINT_SCHEMA:
        raise ValueError(
            "legacy context-inference checkpoint v1 lacks exact clocks; migrate "
            "its state with migrate_legacy_context_inference_state and resave it"
        )
    if schema != CONTEXT_INFERENCE_CHECKPOINT_SCHEMA:
        raise ValueError("context-inference checkpoint schema is unsupported")
    module_config = metadata.get("module_config")
    if not isinstance(module_config, Mapping):
        raise ValueError("context-inference checkpoint module_config is invalid")
    module = ContextInference.from_config(module_config)
    template = module.init()
    restored, restored_metadata = load_checkpoint(template, path)
    if restored_metadata != metadata:
        raise ValueError("context-inference checkpoint metadata changed between reads")
    state = cast(ContextInferenceState, restored)
    module._require_state_contract(state)
    if not bool(jax.device_get(module.state_is_valid(state))):
        raise ValueError("restored context-inference state is invalid")
    expected_resources = module.resource_budget.to_dict()
    if metadata.get("memory_accounting") != expected_resources:
        raise ValueError("context-inference checkpoint resource contract does not match")
    if measure_context_inference_state_nbytes(state) != module.resource_budget.state_nbytes:
        raise ValueError("restored context-inference state size does not match")
    return module, state
