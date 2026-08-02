"""Hidden-partner life with genuine recurrent world-state feedback.

This stream preserves the eight-channel, simultaneous-action A/B/C/D partner
problem while adding an evaluator-hidden Markov world sign ``z``.  Two noisy
ordinary cues report the current sign, and the contextual joint outcome is

``y[t + 1] = z[t] * focal_sign[t] * partner_sign[t] * outcome_noise[t]``.

Consequently reward remains exactly ``(1 + y) / 2``, as required by the
integrated hidden-partner transition contract, but is no longer determined by
the joint action alone.  After scoring the current action, the world sign may
flip and the next cues are sampled.  The partner, world, cue, ordinary-signal,
and schedule RNG streams are independent and action-independent.

The previous contextual outcome and previous partner action, together with
the learner's remembered previous focal action, provide a noisy measurement
of the preceding world sign.  Current noisy cues update the propagated Markov
posterior.  No single transition reveals ``z`` exactly, so optimal inference
uses a genuine multi-step filter without exposing a task id, boundary, or
oracle channel.
"""

# mypy: disable-error-code="call-arg"

from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Mapping
from numbers import Real
from typing import Any

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import Array
from jaxtyping import Bool, Float, Int, PRNGKeyArray

from alberta_framework.streams.hidden_partner_mapping import (
    DEFAULT_BASE_SEGMENT_LENGTHS,
    DEFAULT_JITTER_RADIUS,
    DEFAULT_PARTNER_FLIP_PROBABILITY,
    DEFAULT_REGIME_SCHEDULE,
    NEGATIVE_ACTION,
    POSITIVE_ACTION,
    REGIME_A,
    REGIME_B,
    REGIME_C,
    REGIME_D,
)

HIDDEN_PARTNER_WORLD_FEEDBACK_CONTRACT_VERSION = "hidden-partner-world-feedback-v1"
# All default flip rates lie strictly inside (0, 0.5): every noisy channel is
# individually informative about the hidden sign ``z``, yet none reveals it.
# The world flip rate 0.03 gives ``z`` an expected persistence of ~33 steps,
# slow relative to the per-step cue noise, so integrating evidence across
# steps improves on reading the current cues alone.  The two cues carry
# different reliabilities (0.25 vs 0.35 flip rate), so a correct filter must
# weight them unequally rather than averaging them.
DEFAULT_WORLD_FLIP_PROBABILITY = 0.03
DEFAULT_CUE_FLIP_PROBABILITIES = (0.25, 0.35)
DEFAULT_OUTCOME_FLIP_PROBABILITY = 0.15

REGIME_NAMES = ("A", "B", "C", "D")
N_ACTIONS = 2

X_INDEX = 0
PREVIOUS_OUTCOME_INDEX = 1
PREVIOUS_PARTNER_ACTION_INDEX = 2
HAS_PARTNER_HISTORY_INDEX = 3
U_INDEX = 4
V_INDEX = 5
CUE_1_INDEX = 6
CUE_2_INDEX = 7
OBSERVATION_FIELDS = (
    "x",
    "previous_contextual_outcome",
    "previous_partner_action",
    "has_partner_history",
    "u",
    "v",
    "world_cue_1",
    "world_cue_2",
)
OBSERVATION_DIM = len(OBSERVATION_FIELDS)

_N_SEGMENTS = len(DEFAULT_REGIME_SCHEDULE)
_N_CURRENT_SIGNALS = 3  # x, u, v
_N_CUES = 2
_INT32_MAX = 2**31 - 1

_SCHEDULE_RNG_TAG = 0
_SIGNAL_RNG_TAG = 1
_PARTNER_RNG_TAG = 2
_WORLD_RNG_TAG = 3
_CUE_RNG_TAG = 4
_OUTCOME_RNG_TAG = 5


def _probability(value: object, *, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise ValueError(f"{name} must be finite and lie in [0, 1]")
    return float(value)


@dataclasses.dataclass(frozen=True)
class HiddenPartnerWorldFeedbackConfig:
    """Strict static construction for the world-feedback v1 stream."""

    contract_version: str = HIDDEN_PARTNER_WORLD_FEEDBACK_CONTRACT_VERSION
    regime_schedule: tuple[int, ...] = DEFAULT_REGIME_SCHEDULE
    base_segment_lengths: tuple[int, ...] = DEFAULT_BASE_SEGMENT_LENGTHS
    jitter_radius: int = DEFAULT_JITTER_RADIUS
    partner_flip_probability: float = DEFAULT_PARTNER_FLIP_PROBABILITY
    world_flip_probability: float = DEFAULT_WORLD_FLIP_PROBABILITY
    cue_flip_probabilities: tuple[float, float] = DEFAULT_CUE_FLIP_PROBABILITIES
    outcome_flip_probability: float = DEFAULT_OUTCOME_FLIP_PROBABILITY

    def __post_init__(self) -> None:
        if self.contract_version != HIDDEN_PARTNER_WORLD_FEEDBACK_CONTRACT_VERSION:
            raise ValueError(
                f"contract_version must be {HIDDEN_PARTNER_WORLD_FEEDBACK_CONTRACT_VERSION!r}"
            )
        if not isinstance(self.regime_schedule, tuple):
            raise ValueError("regime_schedule must be a tuple")
        if any(type(regime) is not int for regime in self.regime_schedule):
            raise ValueError("regime_schedule must contain non-boolean integers")
        if self.regime_schedule != DEFAULT_REGIME_SCHEDULE:
            raise ValueError("regime_schedule must be the exact A->B->A->D->A->C->A->B->C schedule")
        if not isinstance(self.base_segment_lengths, tuple):
            raise ValueError("base_segment_lengths must be a tuple")
        if len(self.base_segment_lengths) != _N_SEGMENTS:
            raise ValueError(f"base_segment_lengths must contain exactly {_N_SEGMENTS} values")
        if any(type(length) is not int for length in self.base_segment_lengths):
            raise ValueError("base_segment_lengths must contain non-boolean integers")
        if type(self.jitter_radius) is not int or self.jitter_radius < 0:
            raise ValueError("jitter_radius must be a non-negative integer")
        if any(length <= self.jitter_radius for length in self.base_segment_lengths):
            raise ValueError(
                "every base segment length must exceed jitter_radius so all lengths are positive"
            )
        maximum_cycle_length = sum(self.base_segment_lengths) + (_N_SEGMENTS * self.jitter_radius)
        if maximum_cycle_length > _INT32_MAX:
            raise ValueError("maximum jittered cycle length must fit in int32")
        partner_probability = _probability(
            self.partner_flip_probability,
            name="partner_flip_probability",
        )
        world_probability = _probability(
            self.world_flip_probability,
            name="world_flip_probability",
        )
        if not isinstance(self.cue_flip_probabilities, tuple):
            raise ValueError("cue_flip_probabilities must be a tuple")
        if len(self.cue_flip_probabilities) != _N_CUES:
            raise ValueError(f"cue_flip_probabilities must contain exactly {_N_CUES} values")
        cue_probabilities = tuple(
            _probability(probability, name=f"cue_flip_probabilities[{index}]")
            for index, probability in enumerate(self.cue_flip_probabilities)
        )
        outcome_probability = _probability(
            self.outcome_flip_probability,
            name="outcome_flip_probability",
        )
        object.__setattr__(self, "partner_flip_probability", partner_probability)
        object.__setattr__(self, "world_flip_probability", world_probability)
        object.__setattr__(self, "cue_flip_probabilities", cue_probabilities)
        object.__setattr__(self, "outcome_flip_probability", outcome_probability)

    def to_config(self) -> dict[str, Any]:
        return {
            "type": type(self).__name__,
            "contract_version": self.contract_version,
            "regime_schedule": list(self.regime_schedule),
            "base_segment_lengths": list(self.base_segment_lengths),
            "jitter_radius": self.jitter_radius,
            "partner_flip_probability": float(self.partner_flip_probability),
            "world_flip_probability": float(self.world_flip_probability),
            "cue_flip_probabilities": [
                float(probability) for probability in self.cue_flip_probabilities
            ],
            "outcome_flip_probability": float(self.outcome_flip_probability),
        }

    def canonical_json(self) -> str:
        return json.dumps(
            self.to_config(),
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
    ) -> HiddenPartnerWorldFeedbackConfig:
        payload = dict(config)
        expected = {
            "type",
            "contract_version",
            "regime_schedule",
            "base_segment_lengths",
            "jitter_radius",
            "partner_flip_probability",
            "world_flip_probability",
            "cue_flip_probabilities",
            "outcome_flip_probability",
        }
        if set(payload) != expected:
            raise ValueError("config fields do not match the serialized schema")
        if payload.pop("type") != cls.__name__:
            raise ValueError("unexpected config type")
        schedule = payload.pop("regime_schedule")
        lengths = payload.pop("base_segment_lengths")
        cue_probabilities = payload.pop("cue_flip_probabilities")
        if (
            not isinstance(schedule, (list, tuple))
            or not isinstance(lengths, (list, tuple))
            or not isinstance(cue_probabilities, (list, tuple))
        ):
            raise ValueError("schedule and segment lengths must serialize as sequences")
        return cls(
            regime_schedule=tuple(schedule),
            base_segment_lengths=tuple(lengths),
            cue_flip_probabilities=tuple(cue_probabilities),
            **payload,
        )


@dataclasses.dataclass(frozen=True)
class HiddenPartnerWorldFeedbackResourceBudget:
    """Exact logical persistent-state accounting.

    ``state_nbytes`` counts only leaves of
    :class:`HiddenPartnerWorldFeedbackState`: float32/int32 scalars, one
    boolean, and five JAX PRNG keys counted as ten logical uint32 scalars.
    It excludes Python/config objects, transition outputs, compiler buffers,
    and device alignment.
    """

    observation_float32_scalars: int
    persistent_float32_scalars: int
    persistent_int32_scalars: int
    persistent_bool_scalars: int
    rng_uint32_scalars: int
    persistent_state_scalars: int
    state_nbytes: int
    trainable_scalars: int
    replay_capacity: int

    def to_dict(self) -> dict[str, int]:
        """Return a JSON-compatible accounting record."""
        return dataclasses.asdict(self)


@chex.dataclass(frozen=True)
class HiddenPartnerWorldFeedbackState:
    """Fixed-shape causal state for one uninterrupted environment life."""

    signal_key: PRNGKeyArray
    partner_key: PRNGKeyArray
    world_key: PRNGKeyArray
    cue_key: PRNGKeyArray
    outcome_key: PRNGKeyArray
    segment_lengths: Int[Array, " 9"]
    segment_ends: Int[Array, " 9"]
    current_signals: Float[Array, " 3"]
    current_cues: Float[Array, " 2"]
    world_sign: Float[Array, ""]
    previous_outcome: Float[Array, ""]
    previous_partner_action: Int[Array, ""]
    has_partner_history: Bool[Array, ""]
    step_count: Int[Array, ""]


@chex.dataclass(frozen=True)
class HiddenPartnerWorldFeedbackOracle:
    """Evaluator-only hidden world/schedule diagnostics for one step.

    Includes the hidden world sign before and after the step, per-channel
    noise-flip indicators, and the full-information optimal focal action
    (the action a policy knowing ``z`` and the partner's intent would take,
    with its expected-reward margin and tie flag).
    """

    step_count: Int[Array, ""]
    cycle_index: Int[Array, ""]
    cycle_step: Int[Array, ""]
    cycle_length: Int[Array, ""]
    segment_index: Int[Array, ""]
    segment_step: Int[Array, ""]
    segment_length: Int[Array, ""]
    regime_id: Int[Array, ""]
    next_cycle_index: Int[Array, ""]
    next_segment_index: Int[Array, ""]
    next_regime_id: Int[Array, ""]
    schedule_switched: Bool[Array, ""]
    partner_intended_action: Int[Array, ""]
    partner_intended_sign: Float[Array, ""]
    partner_flipped: Bool[Array, ""]
    focal_action_sign: Float[Array, ""]
    partner_action_sign: Float[Array, ""]
    world_sign: Float[Array, ""]
    world_cue_flipped: Bool[Array, " 2"]
    world_flipped: Bool[Array, ""]
    next_world_sign: Float[Array, ""]
    next_world_cue_flipped: Bool[Array, " 2"]
    outcome_flipped: Bool[Array, ""]
    noiseless_contextual_outcome: Float[Array, ""]
    contextual_outcome: Float[Array, ""]
    counterfactual_rewards: Float[Array, " 2"]
    full_information_optimal_focal_action: Int[Array, ""]
    full_information_action_margin: Float[Array, ""]
    full_information_action_tied: Bool[Array, ""]


@chex.dataclass(frozen=True)
class HiddenPartnerWorldFeedbackTransition:
    """One continuing simultaneous-action transition."""

    observation: Float[Array, " 8"]
    focal_action: Int[Array, ""]
    partner_action: Int[Array, ""]
    reward: Float[Array, ""]
    outcome: Float[Array, ""]
    next_observation: Float[Array, " 8"]
    terminated: Bool[Array, ""]
    discount: Float[Array, ""]
    oracle: HiddenPartnerWorldFeedbackOracle


@chex.dataclass(frozen=True)
class _SchedulePosition:
    """Internal JAX record for a position in the repeating hidden schedule."""

    cycle_index: Int[Array, ""]
    cycle_step: Int[Array, ""]
    cycle_length: Int[Array, ""]
    segment_index: Int[Array, ""]
    segment_step: Int[Array, ""]
    segment_length: Int[Array, ""]
    regime_id: Int[Array, ""]


class HiddenPartnerWorldFeedbackWorld:
    """Pure JAX world-feedback stream with a scripted stochastic partner."""

    def __init__(
        self,
        config: HiddenPartnerWorldFeedbackConfig | None = None,
    ) -> None:
        self._config = HiddenPartnerWorldFeedbackConfig() if config is None else config
        if not isinstance(self._config, HiddenPartnerWorldFeedbackConfig):
            raise TypeError("config must be a HiddenPartnerWorldFeedbackConfig")
        self._base_segment_lengths = jnp.asarray(
            self._config.base_segment_lengths,
            dtype=jnp.int32,
        )
        self._regime_schedule = jnp.asarray(
            self._config.regime_schedule,
            dtype=jnp.int32,
        )

    @property
    def config(self) -> HiddenPartnerWorldFeedbackConfig:
        return self._config

    @property
    def observation_dim(self) -> int:
        return OBSERVATION_DIM

    @property
    def feature_dim(self) -> int:
        return OBSERVATION_DIM

    @property
    def n_actions(self) -> int:
        return N_ACTIONS

    @property
    def n_segments(self) -> int:
        return _N_SEGMENTS

    @property
    def resource_budget(self) -> HiddenPartnerWorldFeedbackResourceBudget:
        float_scalars = _N_CURRENT_SIGNALS + _N_CUES + 2
        int_scalars = 2 * _N_SEGMENTS + 2
        bool_scalars = 1
        rng_scalars = 10
        state_scalars = float_scalars + int_scalars + bool_scalars + rng_scalars
        return HiddenPartnerWorldFeedbackResourceBudget(
            observation_float32_scalars=OBSERVATION_DIM,
            persistent_float32_scalars=float_scalars,
            persistent_int32_scalars=int_scalars,
            persistent_bool_scalars=bool_scalars,
            rng_uint32_scalars=rng_scalars,
            persistent_state_scalars=state_scalars,
            state_nbytes=4 * (float_scalars + int_scalars + rng_scalars) + bool_scalars,
            trainable_scalars=0,
            replay_capacity=0,
        )

    def to_config(self) -> dict[str, Any]:
        return {"type": type(self).__name__, "config": self._config.to_config()}

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
    ) -> HiddenPartnerWorldFeedbackWorld:
        payload = dict(config)
        if set(payload) != {"type", "config"}:
            raise ValueError("world config must contain exactly 'type' and 'config'")
        if payload["type"] != cls.__name__:
            raise ValueError("unexpected world type")
        inner = payload["config"]
        if not isinstance(inner, Mapping):
            raise ValueError("world config must contain a config mapping")
        return cls(HiddenPartnerWorldFeedbackConfig.from_config(inner))

    def init(self, key: Array) -> HiddenPartnerWorldFeedbackState:
        """Initialize a deterministic life with independent named RNG streams."""
        schedule_key = jr.fold_in(key, _SCHEDULE_RNG_TAG)
        signal_root = jr.fold_in(key, _SIGNAL_RNG_TAG)
        partner_key = jr.fold_in(key, _PARTNER_RNG_TAG)
        world_root = jr.fold_in(key, _WORLD_RNG_TAG)
        cue_root = jr.fold_in(key, _CUE_RNG_TAG)
        outcome_key = jr.fold_in(key, _OUTCOME_RNG_TAG)

        initial_signal_key, signal_key = jr.split(signal_root)
        initial_world_key, world_key = jr.split(world_root)
        initial_cue_key, cue_key = jr.split(cue_root)
        world_sign = self._sample_sign(initial_world_key)
        jitters = jr.randint(
            schedule_key,
            shape=(_N_SEGMENTS,),
            minval=-self._config.jitter_radius,
            maxval=self._config.jitter_radius + 1,
            dtype=jnp.int32,
        )
        segment_lengths = self._base_segment_lengths + jitters
        return HiddenPartnerWorldFeedbackState(
            signal_key=signal_key,
            partner_key=partner_key,
            world_key=world_key,
            cue_key=cue_key,
            outcome_key=outcome_key,
            segment_lengths=segment_lengths,
            segment_ends=jnp.cumsum(segment_lengths, dtype=jnp.int32),
            current_signals=self._sample_signals(initial_signal_key),
            current_cues=self._sample_cues(initial_cue_key, world_sign),
            world_sign=world_sign,
            previous_outcome=jnp.asarray(0.0, dtype=jnp.float32),
            previous_partner_action=jnp.asarray(0, dtype=jnp.int32),
            has_partner_history=jnp.asarray(False, dtype=jnp.bool_),
            step_count=jnp.asarray(0, dtype=jnp.int32),
        )

    def observe(self, state: HiddenPartnerWorldFeedbackState) -> Array:
        """Return exactly the eight ordinary, task-oracle-free channels."""
        x, u, v = state.current_signals
        cue_1, cue_2 = state.current_cues
        return jnp.stack(
            (
                x,
                state.previous_outcome,
                state.previous_partner_action.astype(jnp.float32),
                state.has_partner_history.astype(jnp.float32),
                u,
                v,
                cue_1,
                cue_2,
            )
        ).astype(jnp.float32)

    def step(
        self,
        state: HiddenPartnerWorldFeedbackState,
        focal_action: Array,
    ) -> tuple[HiddenPartnerWorldFeedbackTransition, HiddenPartnerWorldFeedbackState]:
        """Apply one scalar focal action id under exact simultaneous timing.

        Valid dynamic action ids are ``0`` and ``1``.  Shape and integer
        dtype are static and checked before tracing; the traced value cannot
        raise under ``jit``, so an out-of-range id instead *poisons* the
        transition: ``focal_sign`` becomes NaN (hence NaN outcome and
        reward) and the world state does not advance.  A policy bug is
        therefore loudly visible to finiteness checks rather than being
        silently coerced onto a legal action.
        """
        action = jnp.asarray(focal_action)
        if action.shape != ():
            raise ValueError(f"focal_action must be scalar, got shape {action.shape}")
        if not jnp.issubdtype(action.dtype, jnp.integer):
            raise TypeError("focal_action must have an integer dtype")
        action_id = action.astype(jnp.int32)

        observation = self.observe(state)
        schedule = self._schedule_position(state, state.step_count)
        intended_sign = self._partner_intended_sign(state, schedule.regime_id)

        next_partner_key, partner_flip_key = jr.split(state.partner_key)
        partner_flipped = jr.bernoulli(
            partner_flip_key,
            p=jnp.asarray(self._config.partner_flip_probability, dtype=jnp.float32),
        )
        partner_sign = jnp.where(partner_flipped, -intended_sign, intended_sign).astype(jnp.float32)
        partner_action = ((partner_sign + 1.0) / 2.0).astype(jnp.int32)

        action_valid = (action_id == NEGATIVE_ACTION) | (action_id == POSITIVE_ACTION)
        focal_sign = 2.0 * action_id.astype(jnp.float32) - 1.0
        focal_sign = jnp.where(action_valid, focal_sign, jnp.float32(jnp.nan))
        noiseless_outcome = (state.world_sign * focal_sign * partner_sign).astype(jnp.float32)
        next_outcome_key, outcome_flip_key = jr.split(state.outcome_key)
        outcome_flipped = jr.bernoulli(
            outcome_flip_key,
            p=jnp.asarray(self._config.outcome_flip_probability, dtype=jnp.float32),
        )
        outcome = jnp.where(
            outcome_flipped,
            -noiseless_outcome,
            noiseless_outcome,
        ).astype(jnp.float32)
        reward = ((1.0 + outcome) / 2.0).astype(jnp.float32)

        next_world_key, world_flip_key = jr.split(state.world_key)
        world_flipped = jr.bernoulli(
            world_flip_key,
            p=jnp.asarray(self._config.world_flip_probability, dtype=jnp.float32),
        )
        next_world_sign = jnp.where(
            world_flipped,
            -state.world_sign,
            state.world_sign,
        ).astype(jnp.float32)
        next_cue_key, cue_sample_key = jr.split(state.cue_key)
        next_cues = self._sample_cues(cue_sample_key, next_world_sign)
        next_signal_key, signal_sample_key = jr.split(state.signal_key)
        next_step = jnp.minimum(
            state.step_count,
            jnp.asarray(_INT32_MAX - 1, dtype=jnp.int32),
        ) + jnp.asarray(1, dtype=jnp.int32)
        proposed_next_state = HiddenPartnerWorldFeedbackState(
            signal_key=next_signal_key,
            partner_key=next_partner_key,
            world_key=next_world_key,
            cue_key=next_cue_key,
            outcome_key=next_outcome_key,
            segment_lengths=state.segment_lengths,
            segment_ends=state.segment_ends,
            current_signals=self._sample_signals(signal_sample_key),
            current_cues=next_cues,
            world_sign=next_world_sign,
            previous_outcome=outcome,
            previous_partner_action=partner_sign.astype(jnp.int32),
            has_partner_history=jnp.asarray(True, dtype=jnp.bool_),
            step_count=next_step,
        )
        next_state = jax.lax.cond(
            action_valid,
            lambda _: proposed_next_state,
            lambda _: state,
            operand=None,
        )
        next_schedule = self._schedule_position(next_state, next_state.step_count)

        focal_signs = jnp.asarray((-1.0, 1.0), dtype=jnp.float32)
        outcome_noise_sign = jnp.where(outcome_flipped, -1.0, 1.0).astype(jnp.float32)
        counterfactual_rewards = (
            1.0 + state.world_sign * focal_signs * partner_sign * outcome_noise_sign
        ) / 2.0
        intended_action = ((intended_sign + 1.0) / 2.0).astype(jnp.int32)
        full_information_coefficient = (
            (1.0 - 2.0 * self._config.partner_flip_probability)
            * (1.0 - 2.0 * self._config.outcome_flip_probability)
            * state.world_sign
            * intended_sign
        ).astype(jnp.float32)
        optimal_sign = jnp.where(full_information_coefficient >= 0.0, 1.0, -1.0)
        optimal_action = ((optimal_sign + 1.0) / 2.0).astype(jnp.int32)
        oracle = HiddenPartnerWorldFeedbackOracle(
            step_count=state.step_count,
            cycle_index=schedule.cycle_index,
            cycle_step=schedule.cycle_step,
            cycle_length=schedule.cycle_length,
            segment_index=schedule.segment_index,
            segment_step=schedule.segment_step,
            segment_length=schedule.segment_length,
            regime_id=schedule.regime_id,
            next_cycle_index=next_schedule.cycle_index,
            next_segment_index=next_schedule.segment_index,
            next_regime_id=next_schedule.regime_id,
            schedule_switched=(
                (schedule.cycle_index != next_schedule.cycle_index)
                | (schedule.segment_index != next_schedule.segment_index)
            ),
            partner_intended_action=intended_action,
            partner_intended_sign=intended_sign,
            partner_flipped=partner_flipped,
            focal_action_sign=focal_sign,
            partner_action_sign=partner_sign,
            world_sign=state.world_sign,
            world_cue_flipped=state.current_cues != state.world_sign,
            world_flipped=world_flipped,
            next_world_sign=next_world_sign,
            next_world_cue_flipped=next_cues != next_world_sign,
            outcome_flipped=outcome_flipped,
            noiseless_contextual_outcome=noiseless_outcome,
            contextual_outcome=outcome,
            counterfactual_rewards=counterfactual_rewards.astype(jnp.float32),
            full_information_optimal_focal_action=optimal_action,
            full_information_action_margin=jnp.abs(full_information_coefficient),
            full_information_action_tied=full_information_coefficient == 0.0,
        )
        transition = HiddenPartnerWorldFeedbackTransition(
            observation=observation,
            focal_action=action_id,
            partner_action=partner_action,
            reward=reward,
            outcome=outcome,
            next_observation=self.observe(next_state),
            terminated=jnp.asarray(False, dtype=jnp.bool_),
            discount=jnp.asarray(1.0, dtype=jnp.float32),
            oracle=oracle,
        )
        return transition, next_state

    @staticmethod
    def _sample_sign(key: Array) -> Array:
        return jnp.where(jr.bernoulli(key), 1.0, -1.0).astype(jnp.float32)

    @staticmethod
    def _sample_signals(key: Array) -> Array:
        positive = jr.bernoulli(key, p=0.5, shape=(_N_CURRENT_SIGNALS,))
        return jnp.where(positive, 1.0, -1.0).astype(jnp.float32)

    def _sample_cues(self, key: Array, world_sign: Array) -> Array:
        flipped = jr.bernoulli(
            key,
            p=jnp.asarray(self._config.cue_flip_probabilities, dtype=jnp.float32),
            shape=(_N_CUES,),
        )
        return jnp.where(flipped, -world_sign, world_sign).astype(jnp.float32)

    def _partner_intended_sign(
        self,
        state: HiddenPartnerWorldFeedbackState,
        regime_id: Array,
    ) -> Array:
        x, u, v = state.current_signals
        targets = jnp.stack(
            (
                x,
                -x,
                x * state.previous_partner_action.astype(jnp.float32),
                u * v,
            )
        )
        return targets[regime_id].astype(jnp.float32)

    def _schedule_position(
        self,
        state: HiddenPartnerWorldFeedbackState,
        step_count: Array,
    ) -> _SchedulePosition:
        cycle_length = state.segment_ends[-1]
        cycle_index = jnp.floor_divide(step_count, cycle_length).astype(jnp.int32)
        cycle_step = jnp.mod(step_count, cycle_length).astype(jnp.int32)
        segment_index = jnp.sum(cycle_step >= state.segment_ends).astype(jnp.int32)
        prior_index = jnp.maximum(segment_index - 1, 0)
        segment_start = jnp.where(
            segment_index == 0,
            jnp.asarray(0, dtype=jnp.int32),
            state.segment_ends[prior_index],
        )
        return _SchedulePosition(
            cycle_index=cycle_index,
            cycle_step=cycle_step,
            cycle_length=cycle_length,
            segment_index=segment_index,
            segment_step=(cycle_step - segment_start).astype(jnp.int32),
            segment_length=state.segment_lengths[segment_index],
            regime_id=self._regime_schedule[segment_index],
        )


__all__ = [
    "CUE_1_INDEX",
    "CUE_2_INDEX",
    "DEFAULT_CUE_FLIP_PROBABILITIES",
    "DEFAULT_OUTCOME_FLIP_PROBABILITY",
    "DEFAULT_WORLD_FLIP_PROBABILITY",
    "HAS_PARTNER_HISTORY_INDEX",
    "HIDDEN_PARTNER_WORLD_FEEDBACK_CONTRACT_VERSION",
    "HiddenPartnerWorldFeedbackConfig",
    "HiddenPartnerWorldFeedbackOracle",
    "HiddenPartnerWorldFeedbackResourceBudget",
    "HiddenPartnerWorldFeedbackState",
    "HiddenPartnerWorldFeedbackTransition",
    "HiddenPartnerWorldFeedbackWorld",
    "N_ACTIONS",
    "OBSERVATION_DIM",
    "OBSERVATION_FIELDS",
    "PREVIOUS_OUTCOME_INDEX",
    "PREVIOUS_PARTNER_ACTION_INDEX",
    "REGIME_A",
    "REGIME_B",
    "REGIME_C",
    "REGIME_D",
    "REGIME_NAMES",
    "U_INDEX",
    "V_INDEX",
    "X_INDEX",
]
