"""Learner-visible Bayes reference for the noisy hidden-partner world sign.

This evaluator consumes only information available after an executed
transition: current noisy cues and the corrected feedback
``d[t+1] = outcome[t+1] * focal_sign[t] * partner_sign[t]``.  It never consumes
the evaluator-hidden world sign, partner regime, segment boundary, or noise
bits.  The resulting filter is an oracle baseline and must not feed the
learning agent.
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
import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import Bool, Float, Int

from alberta_framework.streams.hidden_partner_world_feedback import (
    DEFAULT_CUE_FLIP_PROBABILITIES,
    DEFAULT_OUTCOME_FLIP_PROBABILITY,
    DEFAULT_WORLD_FLIP_PROBABILITY,
    HiddenPartnerWorldFeedbackConfig,
)

HIDDEN_PARTNER_WORLD_FILTER_CONTRACT_VERSION = "hidden-partner-world-filter-v1"
_N_CUES = 2
_N_ACTIONS = 2
_INT32_MAX = 2**31 - 1


def _probability(
    value: object,
    *,
    name: str,
    upper_inclusive: bool,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be a finite non-boolean real number")
    probability = float(value)
    upper_valid = probability <= 0.5 if upper_inclusive else probability < 0.5
    lower_valid = probability >= 0.0 if upper_inclusive else probability > 0.0
    if not lower_valid or not upper_valid:
        bracket = "[0, 0.5]" if upper_inclusive else "(0, 0.5)"
        raise ValueError(f"{name} must lie in {bracket}")
    if not upper_inclusive:
        probability32 = np.float32(probability)
        reliability32 = np.float32(1.0) - np.float32(2.0) * probability32
        if not np.float32(0.0) < probability32 < np.float32(0.5):
            raise ValueError(f"{name} must remain strictly inside (0, 0.5) in float32")
        if not np.float32(0.0) < reliability32 < np.float32(1.0):
            raise ValueError(f"{name} reliability must remain strictly inside (0, 1) in float32")
    return probability


def _require_array(
    value: Array,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: Any,
) -> Array:
    array = jnp.asarray(value)
    expected_dtype = jnp.dtype(dtype)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if array.dtype != expected_dtype:
        raise TypeError(f"{name} must have dtype {expected_dtype}, got {array.dtype}")
    return array


def _signs_valid(values: Array) -> Array:
    return jnp.all(jnp.isfinite(values) & ((values == -1.0) | (values == 1.0)))


def _observe_binary_symmetric_channel(
    prior_mean: Array,
    observation: Array,
    error_probability: float,
) -> Array:
    reliability = jnp.asarray(1.0 - 2.0 * error_probability, dtype=jnp.float32)
    denominator = 1.0 + observation * reliability * prior_mean
    return ((prior_mean + observation * reliability) / denominator).astype(jnp.float32)


@dataclasses.dataclass(frozen=True)
class HiddenPartnerWorldFilterConfig:
    """Strict static probabilities for the learner-visible reference filter."""

    contract_version: str = HIDDEN_PARTNER_WORLD_FILTER_CONTRACT_VERSION
    world_flip_probability: float = DEFAULT_WORLD_FLIP_PROBABILITY
    cue_flip_probabilities: tuple[float, float] = DEFAULT_CUE_FLIP_PROBABILITIES
    outcome_flip_probability: float = DEFAULT_OUTCOME_FLIP_PROBABILITY

    def __post_init__(self) -> None:
        if self.contract_version != HIDDEN_PARTNER_WORLD_FILTER_CONTRACT_VERSION:
            raise ValueError(
                f"contract_version must be {HIDDEN_PARTNER_WORLD_FILTER_CONTRACT_VERSION!r}"
            )
        world_probability = _probability(
            self.world_flip_probability,
            name="world_flip_probability",
            upper_inclusive=True,
        )
        if not isinstance(self.cue_flip_probabilities, tuple):
            raise ValueError("cue_flip_probabilities must be a tuple")
        if len(self.cue_flip_probabilities) != _N_CUES:
            raise ValueError(f"cue_flip_probabilities must contain exactly {_N_CUES} values")
        cue_probabilities = tuple(
            _probability(
                value,
                name=f"cue_flip_probabilities[{index}]",
                upper_inclusive=False,
            )
            for index, value in enumerate(self.cue_flip_probabilities)
        )
        outcome_probability = _probability(
            self.outcome_flip_probability,
            name="outcome_flip_probability",
            upper_inclusive=False,
        )
        object.__setattr__(self, "world_flip_probability", world_probability)
        object.__setattr__(self, "cue_flip_probabilities", cue_probabilities)
        object.__setattr__(self, "outcome_flip_probability", outcome_probability)

    def to_config(self) -> dict[str, Any]:
        return {
            "type": type(self).__name__,
            "contract_version": self.contract_version,
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
    def from_config(cls, config: Mapping[str, Any]) -> HiddenPartnerWorldFilterConfig:
        payload = dict(config)
        expected = {
            "type",
            "contract_version",
            "world_flip_probability",
            "cue_flip_probabilities",
            "outcome_flip_probability",
        }
        if set(payload) != expected:
            raise ValueError("filter config fields do not match the serialized schema")
        if payload.pop("type") != cls.__name__:
            raise ValueError("unexpected filter config type")
        cue_probabilities = payload.pop("cue_flip_probabilities")
        if not isinstance(cue_probabilities, (list, tuple)):
            raise ValueError("cue_flip_probabilities must serialize as a sequence")
        return cls(
            cue_flip_probabilities=tuple(cue_probabilities),
            **payload,
        )

    @classmethod
    def from_world_config(
        cls,
        world_config: HiddenPartnerWorldFeedbackConfig,
    ) -> HiddenPartnerWorldFilterConfig:
        """Construct only when the stream lies inside the supported filter domain."""
        if not isinstance(world_config, HiddenPartnerWorldFeedbackConfig):
            raise TypeError("world_config must be a HiddenPartnerWorldFeedbackConfig")
        return cls(
            world_flip_probability=world_config.world_flip_probability,
            cue_flip_probabilities=world_config.cue_flip_probabilities,
            outcome_flip_probability=world_config.outcome_flip_probability,
        )


@chex.dataclass(frozen=True)
class HiddenPartnerWorldFilterState:
    """Posterior state; one malformed datum latches ``valid=False`` fail-stop."""

    posterior_mean: Float[Array, ""]
    step_count: Int[Array, ""]
    valid: Bool[Array, ""]


@chex.dataclass(frozen=True)
class HiddenPartnerWorldFilterUpdate:
    state: HiddenPartnerWorldFilterState
    posterior_after_outcome: Float[Array, ""]
    propagated_prior: Float[Array, ""]
    valid: Bool[Array, ""]


@chex.dataclass(frozen=True)
class HiddenPartnerWorldRewardCells:
    rewards: Float[Array, "2 2"]
    valid: Bool[Array, ""]


@chex.dataclass(frozen=True)
class HiddenPartnerWorldDecision:
    expected_rewards: Float[Array, " 2"]
    greedy_action: Int[Array, ""]
    optimal_value: Float[Array, ""]
    action_regrets: Float[Array, " 2"]
    action_margin: Float[Array, ""]
    tied: Bool[Array, ""]
    valid: Bool[Array, ""]


class HiddenPartnerWorldBayesFilter:
    """Exact two-state recursive filter and ex-ante reward oracle."""

    def __init__(self, config: HiddenPartnerWorldFilterConfig | None = None) -> None:
        self._config = HiddenPartnerWorldFilterConfig() if config is None else config
        if not isinstance(self._config, HiddenPartnerWorldFilterConfig):
            raise TypeError("config must be a HiddenPartnerWorldFilterConfig")

    @property
    def config(self) -> HiddenPartnerWorldFilterConfig:
        return self._config

    def initialize(self, current_cues: Array) -> HiddenPartnerWorldFilterState:
        cues = _require_array(
            current_cues,
            name="current_cues",
            shape=(_N_CUES,),
            dtype=jnp.float32,
        )
        valid = _signs_valid(cues)
        safe_cues = jnp.where(jnp.isfinite(cues), cues, 1.0)
        posterior = jnp.asarray(0.0, dtype=jnp.float32)
        for index, error_probability in enumerate(self._config.cue_flip_probabilities):
            posterior = _observe_binary_symmetric_channel(
                posterior,
                safe_cues[index],
                error_probability,
            )
        numeric_valid = jnp.isfinite(posterior) & (jnp.abs(posterior) <= 1.0)
        valid = valid & numeric_valid
        return HiddenPartnerWorldFilterState(
            posterior_mean=jnp.where(valid, posterior, 0.0),
            step_count=jnp.asarray(0, dtype=jnp.int32),
            valid=valid,
        )

    def advance(
        self,
        state: HiddenPartnerWorldFilterState,
        corrected_outcome: Array,
        next_cues: Array,
    ) -> HiddenPartnerWorldFilterUpdate:
        """Assimilate one corrected outcome, propagate, then assimilate next cues.

        Dynamic invalidity preserves the posterior and counter but permanently
        latches the returned state invalid.  Evaluators must reject that life;
        they must not attempt to recover or use its oracle metrics.
        """
        if not isinstance(state, HiddenPartnerWorldFilterState):
            raise TypeError("state must be a HiddenPartnerWorldFilterState")
        state_mean = _require_array(
            state.posterior_mean,
            name="state.posterior_mean",
            shape=(),
            dtype=jnp.float32,
        )
        state_count = _require_array(
            state.step_count,
            name="state.step_count",
            shape=(),
            dtype=jnp.int32,
        )
        state_valid = _require_array(
            state.valid,
            name="state.valid",
            shape=(),
            dtype=jnp.bool_,
        )
        outcome = _require_array(
            corrected_outcome,
            name="corrected_outcome",
            shape=(),
            dtype=jnp.float32,
        )
        cues = _require_array(
            next_cues,
            name="next_cues",
            shape=(_N_CUES,),
            dtype=jnp.float32,
        )
        input_valid = (
            state_valid
            & jnp.isfinite(state_mean)
            & (jnp.abs(state_mean) <= 1.0)
            & (state_count >= 0)
            & jnp.isfinite(outcome)
            & ((outcome == -1.0) | (outcome == 1.0))
            & _signs_valid(cues)
        )
        safe_outcome = jnp.where(jnp.isfinite(outcome), outcome, 1.0)
        safe_cues = jnp.where(jnp.isfinite(cues), cues, 1.0)
        posterior_after_outcome = _observe_binary_symmetric_channel(
            state_mean,
            safe_outcome,
            self._config.outcome_flip_probability,
        )
        propagated_prior = (
            jnp.asarray(
                1.0 - 2.0 * self._config.world_flip_probability,
                dtype=jnp.float32,
            )
            * posterior_after_outcome
        ).astype(jnp.float32)
        posterior = propagated_prior
        for index, error_probability in enumerate(self._config.cue_flip_probabilities):
            posterior = _observe_binary_symmetric_channel(
                posterior,
                safe_cues[index],
                error_probability,
            )
        numeric_valid = (
            jnp.isfinite(posterior_after_outcome)
            & jnp.isfinite(propagated_prior)
            & jnp.isfinite(posterior)
            & (jnp.abs(posterior_after_outcome) <= 1.0)
            & (jnp.abs(propagated_prior) <= 1.0)
            & (jnp.abs(posterior) <= 1.0)
        )
        valid = input_valid & numeric_valid
        maximum = jnp.asarray(_INT32_MAX, dtype=jnp.int32)
        next_count = jnp.minimum(state_count, maximum - 1) + 1
        next_state = HiddenPartnerWorldFilterState(
            posterior_mean=jnp.where(valid, posterior, state_mean),
            step_count=jnp.where(valid, next_count, state_count),
            valid=valid,
        )
        return HiddenPartnerWorldFilterUpdate(
            state=next_state,
            posterior_after_outcome=jnp.where(valid, posterior_after_outcome, 0.0),
            propagated_prior=jnp.where(valid, propagated_prior, 0.0),
            valid=valid,
        )

    def expected_reward_cells(self, posterior_mean: Array) -> HiddenPartnerWorldRewardCells:
        mean = _require_array(
            posterior_mean,
            name="posterior_mean",
            shape=(),
            dtype=jnp.float32,
        )
        valid = jnp.isfinite(mean) & (jnp.abs(mean) <= 1.0)
        safe_mean = jnp.where(valid, mean, 0.0)
        signs = jnp.asarray((-1.0, 1.0), dtype=jnp.float32)
        reliability = jnp.asarray(
            1.0 - 2.0 * self._config.outcome_flip_probability,
            dtype=jnp.float32,
        )
        rewards = 0.5 * (1.0 + reliability * safe_mean * signs[:, None] * signs[None, :])
        return HiddenPartnerWorldRewardCells(
            rewards=jnp.where(valid, rewards, jnp.full_like(rewards, 0.5)),
            valid=valid,
        )

    def marginalize_partner(
        self,
        reward_cells: HiddenPartnerWorldRewardCells,
        partner_probabilities: Array,
    ) -> HiddenPartnerWorldDecision:
        if not isinstance(reward_cells, HiddenPartnerWorldRewardCells):
            raise TypeError("reward_cells must be a HiddenPartnerWorldRewardCells")
        cell_rewards = _require_array(
            reward_cells.rewards,
            name="reward_cells.rewards",
            shape=(_N_ACTIONS, _N_ACTIONS),
            dtype=jnp.float32,
        )
        cells_valid_flag = _require_array(
            reward_cells.valid,
            name="reward_cells.valid",
            shape=(),
            dtype=jnp.bool_,
        )
        probabilities = _require_array(
            partner_probabilities,
            name="partner_probabilities",
            shape=(_N_ACTIONS,),
            dtype=jnp.float32,
        )
        probabilities_valid = (
            jnp.all(jnp.isfinite(probabilities))
            & jnp.all((probabilities >= 0.0) & (probabilities <= 1.0))
            & jnp.isclose(jnp.sum(probabilities), 1.0, atol=1e-6, rtol=0.0)
        )
        rewards_valid = (
            cells_valid_flag
            & jnp.all(jnp.isfinite(cell_rewards))
            & jnp.all((cell_rewards >= 0.0) & (cell_rewards <= 1.0))
        )
        valid = probabilities_valid & rewards_valid
        safe_probabilities_pre = jnp.where(
            probabilities_valid,
            probabilities,
            jnp.asarray((0.5, 0.5), dtype=jnp.float32),
        )
        safe_probabilities = safe_probabilities_pre / jnp.sum(safe_probabilities_pre)
        safe_cells = jnp.where(
            rewards_valid,
            cell_rewards,
            jnp.full((_N_ACTIONS, _N_ACTIONS), 0.5, dtype=jnp.float32),
        )
        expected = safe_cells @ safe_probabilities
        optimal_value = jnp.max(expected)
        regrets = optimal_value - expected
        decision_values_valid = (
            jnp.all(jnp.isfinite(expected))
            & jnp.all((expected >= 0.0) & (expected <= 1.0))
            & jnp.isfinite(optimal_value)
            & (optimal_value >= 0.0)
            & (optimal_value <= 1.0)
        )
        valid = valid & decision_values_valid
        action_margin = jnp.abs(expected[1] - expected[0])
        neutral = jnp.full((_N_ACTIONS,), 0.5, dtype=jnp.float32)
        return HiddenPartnerWorldDecision(
            expected_rewards=jnp.where(valid, expected, neutral),
            greedy_action=jnp.where(
                valid,
                jnp.argmax(expected).astype(jnp.int32),
                jnp.asarray(0, dtype=jnp.int32),
            ),
            optimal_value=jnp.where(valid, optimal_value, 0.5),
            action_regrets=jnp.where(valid, regrets, jnp.zeros_like(regrets)),
            action_margin=jnp.where(valid, action_margin, 0.0),
            tied=valid & (action_margin <= jnp.asarray(1e-7, dtype=jnp.float32)),
            valid=valid,
        )


__all__ = [
    "HIDDEN_PARTNER_WORLD_FILTER_CONTRACT_VERSION",
    "HiddenPartnerWorldBayesFilter",
    "HiddenPartnerWorldDecision",
    "HiddenPartnerWorldFilterConfig",
    "HiddenPartnerWorldFilterState",
    "HiddenPartnerWorldFilterUpdate",
    "HiddenPartnerWorldRewardCells",
]
