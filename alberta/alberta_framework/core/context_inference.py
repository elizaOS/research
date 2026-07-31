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

import functools

import chex
import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Bool, Float, Int

__all__ = [
    "ContextInference",
    "ContextInferenceConfig",
    "ContextInferenceState",
]


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
        if self.n_actions < 1:
            raise ValueError("n_actions must be positive")
        if self.observation_dim < 1:
            raise ValueError("observation_dim must be positive")
        if self.max_contexts < 2:
            raise ValueError("max_contexts must be at least 2")
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
        if self.min_dwell < 0:
            raise ValueError("min_dwell must be non-negative")


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
            (``-1`` for never); drives least-recently-used eviction.
        dwell: Steps since the last context switch.
        step_count: Total updates applied.
    """

    reward_weights: Float[Array, "max_contexts n_actions observation_dim"]
    error_ema: Float[Array, " max_contexts"]
    in_use: Bool[Array, " max_contexts"]
    active_context: Int[Array, ""]
    last_active_step: Int[Array, " max_contexts"]
    dwell: Int[Array, ""]
    step_count: Int[Array, ""]


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
        )

    def context_onehot(self, state: ContextInferenceState) -> Float[Array, " max_contexts"]:
        """One-hot channels of the currently inferred context."""
        return jax.nn.one_hot(state.active_context, self._config.max_contexts, dtype=jnp.float32)

    def num_contexts_in_use(self, state: ContextInferenceState) -> Int[Array, ""]:
        """Number of allocated context slots."""
        return jnp.sum(state.in_use).astype(jnp.int32)

    @functools.partial(jax.jit, static_argnums=(0,))
    def update(
        self,
        state: ContextInferenceState,
        observation: Array,
        action: Array,
        reward: Array,
    ) -> tuple[ContextInferenceState, Float[Array, " max_contexts"]]:
        """Infer the context of one transition and refine the active model.

        Args:
            state: Current state (never mutated).
            observation: Observation in which ``action`` was taken (the
                pre-transition percept the reward is conditioned on).
            action: Discrete action taken.
            reward: Reward observed for taking ``action`` in ``observation``.

        Returns:
            Tuple of (new state, one-hot of the inferred context, shape
            ``(max_contexts,)``).
        """
        cfg = self._config
        k = cfg.max_contexts
        obs = jnp.asarray(observation, dtype=jnp.float32)
        action_index = jnp.squeeze(jnp.asarray(action, dtype=jnp.int32))
        reward_s = jnp.squeeze(jnp.asarray(reward, dtype=jnp.float32))

        # 1. Evidence: every slot predicts the observed reward; in-use slots
        #    track a recent-|error| EMA, unused slots sit at the fresh-model
        #    prior so reuse and allocation compete on one scale.
        predictions = state.reward_weights[:, action_index, :] @ obs
        errors = jnp.abs(reward_s - predictions)
        decay = jnp.float32(cfg.error_decay)
        error_ema = decay * state.error_ema + (1.0 - decay) * errors
        error_ema = jnp.where(state.in_use, error_ema, jnp.float32(cfg.novelty_prior_error))

        # 2. Change-point: leave a misfitting active slot for the best
        #    alternative — a stored regime if one explains recent rewards,
        #    else a fresh slot (evicting the least-recently-active slot when
        #    all are in use).
        active = state.active_context
        slot_ids = jnp.arange(k)
        active_mask = slot_ids == active
        stored_scores = jnp.where(active_mask | ~state.in_use, jnp.inf, error_ema)
        best_stored = jnp.argmin(stored_scores).astype(jnp.int32)
        best_stored_error = stored_scores[best_stored]
        allocate = best_stored_error > cfg.novelty_prior_error
        free_exists = jnp.any(~state.in_use)
        first_free = jnp.argmin(state.in_use).astype(jnp.int32)
        lru_scores = jnp.where(
            active_mask | ~state.in_use,
            jnp.asarray(2_147_483_647, dtype=jnp.int32),
            state.last_active_step,
        )
        lru_slot = jnp.argmin(lru_scores).astype(jnp.int32)
        fresh_slot = jnp.where(free_exists, first_free, lru_slot)
        target = jnp.where(allocate, fresh_slot, best_stored).astype(jnp.int32)
        target_error = jnp.minimum(best_stored_error, jnp.float32(cfg.novelty_prior_error))
        do_switch = (
            (error_ema[active] > cfg.switch_threshold)
            & (state.dwell >= cfg.min_dwell)
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
        prediction = reward_weights[new_active, action_index, :] @ obs
        model_error = reward_s - prediction
        gate = (jnp.abs(model_error) <= cfg.update_error_gate).astype(jnp.float32)
        norm = jnp.maximum(obs @ obs, 1e-8)
        reward_weights = reward_weights.at[new_active, action_index, :].add(
            jnp.float32(cfg.model_step_size) * gate * model_error * obs / norm
        )

        new_state = ContextInferenceState(
            reward_weights=reward_weights,
            error_ema=error_ema,
            in_use=in_use,
            active_context=new_active,
            last_active_step=state.last_active_step.at[new_active].set(state.step_count),
            dwell=jnp.where(do_switch, jnp.int32(0), state.dwell + 1),
            step_count=state.step_count + 1,
        )
        return new_state, jax.nn.one_hot(new_active, k, dtype=jnp.float32)
