# mypy: disable-error-code="attr-defined,call-arg,name-defined"
"""Conventional option-return and option-duration prediction.

This module implements the narrow mechanism called for in Alberta Plan Step 5:
learn an option's conventional cumulative reward *and* its expected remaining
primitive-step duration.  Each supplied option owns two linear GVF-equivalent
TD(0) heads:

.. math::

    V^r_o(s_t) &= \\mathbb{E}[R_{t+1} + \\gamma_{t+1} V^r_o(s_{t+1})], \\
    V^\tau_o(s_t) &= \\mathbb{E}[1 + \\gamma_{t+1} V^\tau_o(s_{t+1})].

The caller supplies the effective per-transition continuation discount
``gamma`` explicitly: zero on option termination, one while an undiscounted
option continues, or a value in ``[0, 1]`` for a discounted question.  The
reward target deliberately does **not** subtract an average-reward baseline.
That conventional prediction is separate from the differential learners in
``average_reward.py``.

The implementation is linear, online, and fixed-memory.  It assumes option
identities and features are supplied; it does not discover options, learn their
policies, or establish Step 5 completion on its own.
"""

from __future__ import annotations

import dataclasses
import functools
import math
import time
from typing import Any

import chex
import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float, Int

REWARD_HEAD = 0
DURATION_HEAD = 1
N_HEADS = 2


@dataclasses.dataclass(frozen=True)
class OptionValueDurationConfig:
    """Hyperparameters for the two conventional TD(0) heads.

    Args:
        reward_step_size: Step-size for the cumulative-reward head.
        duration_step_size: Step-size for the remaining-duration head.
        duration_floor: Positive denominator floor used only when reporting
            reward-per-step scores.  It does not alter either TD target.
    """

    reward_step_size: float = 0.1
    duration_step_size: float = 0.1
    duration_floor: float = 1e-6

    def __post_init__(self) -> None:
        """Validate scalar hyperparameters."""
        if not math.isfinite(self.reward_step_size) or self.reward_step_size < 0.0:
            raise ValueError("reward_step_size must be finite and non-negative")
        if not math.isfinite(self.duration_step_size) or self.duration_step_size < 0.0:
            raise ValueError("duration_step_size must be finite and non-negative")
        if not math.isfinite(self.duration_floor) or self.duration_floor <= 0.0:
            raise ValueError("duration_floor must be finite and positive")

    def to_config(self) -> dict[str, Any]:
        """Return a JSON-serializable configuration."""
        return {
            "type": "OptionValueDurationConfig",
            "reward_step_size": self.reward_step_size,
            "duration_step_size": self.duration_step_size,
            "duration_floor": self.duration_floor,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> OptionValueDurationConfig:
        """Reconstruct a config from :meth:`to_config` output."""
        payload = dict(config)
        payload.pop("type", None)
        return cls(**payload)


@chex.dataclass(frozen=True)
class OptionValueDurationState:
    """Fixed-memory state for all supplied options.

    ``weights[o, 0]`` is option ``o``'s reward head and ``weights[o, 1]`` is
    its duration head.  There are exactly
    ``2 * n_options * feature_dim`` trainable scalar parameters.
    """

    weights: Float[Array, "n_options 2 feature_dim"]
    option_update_counts: Int[Array, " n_options"]
    step_count: Int[Array, ""]
    birth_timestamp: float = 0.0
    uptime_s: float = 0.0


@chex.dataclass(frozen=True)
class OptionValueDurationPrediction:
    """Predictions for every option at one feature vector.

    ``reward_rates`` is the descriptive ratio ``reward_values / durations``.
    It is a valid direct decision score in renewal settings where the options
    return to the same decision condition, as in the accompanying diagnostic.
    It is not a general replacement for a semi-Markov control backup when
    options lead to different downstream states.
    """

    reward_values: Float[Array, " n_options"]
    durations: Float[Array, " n_options"]
    reward_rates: Float[Array, " n_options"]


@chex.dataclass(frozen=True)
class OptionValueDurationUpdateResult:
    """Diagnostics from one selected-option TD(0) update.

    The two-vector fields are ordered ``[reward, duration]``.
    """

    state: OptionValueDurationState
    predictions: Float[Array, " 2"]
    next_predictions: Float[Array, " 2"]
    td_targets: Float[Array, " 2"]
    td_errors: Float[Array, " 2"]
    continuation_discount: Float[Array, ""]


@chex.dataclass(frozen=True)
class OptionValueDurationArrayResult:
    """Result of scanning the learner over primitive option transitions."""

    state: OptionValueDurationState
    predictions: Float[Array, "num_steps 2"]
    next_predictions: Float[Array, "num_steps 2"]
    td_targets: Float[Array, "num_steps 2"]
    td_errors: Float[Array, "num_steps 2"]
    continuation_discounts: Float[Array, " num_steps"]


class OptionValueDurationLearner:
    """Two conventional linear TD(0) heads for each supplied option.

    This learner is intentionally independent of differential option/control
    values.  To use it, call :meth:`update` for every primitive transition
    generated while an option runs, passing that option's index and an explicit
    continuation discount.  A terminating transition must have discount zero.
    """

    def __init__(
        self,
        n_options: int,
        config: OptionValueDurationConfig | None = None,
    ):
        """Create a fixed-capacity option predictor."""
        if n_options < 1:
            raise ValueError("n_options must be positive")
        self._n_options = n_options
        self._config = config or OptionValueDurationConfig()

    @property
    def n_options(self) -> int:
        """Number of supplied option identities."""
        return self._n_options

    @property
    def config(self) -> OptionValueDurationConfig:
        """Learner configuration."""
        return self._config

    def to_config(self) -> dict[str, Any]:
        """Return a JSON-serializable learner configuration."""
        return {
            "type": "OptionValueDurationLearner",
            "n_options": self._n_options,
            "config": self._config.to_config(),
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> OptionValueDurationLearner:
        """Reconstruct a learner from :meth:`to_config` output."""
        payload = dict(config)
        payload.pop("type", None)
        return cls(
            n_options=int(payload["n_options"]),
            config=OptionValueDurationConfig.from_config(payload["config"]),
        )

    def trainable_parameter_count(self, feature_dim: int) -> int:
        """Return the exact history-independent trainable parameter count."""
        if feature_dim < 1:
            raise ValueError("feature_dim must be positive")
        return self._n_options * N_HEADS * feature_dim

    def init(self, feature_dim: int) -> OptionValueDurationState:
        """Initialize all heads to zero."""
        if feature_dim < 1:
            raise ValueError("feature_dim must be positive")
        return OptionValueDurationState(
            weights=jnp.zeros(
                (self._n_options, N_HEADS, feature_dim),
                dtype=jnp.float32,
            ),
            option_update_counts=jnp.zeros((self._n_options,), dtype=jnp.int32),
            step_count=jnp.array(0, dtype=jnp.int32),
            birth_timestamp=time.time(),
            uptime_s=0.0,
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def predict_heads(
        self,
        state: OptionValueDurationState,
        observation: Array,
    ) -> Float[Array, "n_options 2"]:
        """Return raw ``[reward, duration]`` head predictions for all options."""
        observation = jnp.asarray(observation, dtype=jnp.float32)
        return jnp.einsum("ohf,f->oh", state.weights, observation)

    @functools.partial(jax.jit, static_argnums=(0,))
    def predict(
        self,
        state: OptionValueDurationState,
        observation: Array,
    ) -> OptionValueDurationPrediction:
        """Return conventional values, durations, and reward-per-step scores."""
        heads = self.predict_heads(state, observation)
        reward_values = heads[:, REWARD_HEAD]
        durations = heads[:, DURATION_HEAD]
        safe_durations = jnp.maximum(
            durations,
            jnp.asarray(self._config.duration_floor, dtype=jnp.float32),
        )
        return OptionValueDurationPrediction(
            reward_values=reward_values,
            durations=durations,
            reward_rates=reward_values / safe_durations,
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def update(
        self,
        state: OptionValueDurationState,
        observation: Array,
        option_index: Array,
        reward: Array,
        next_observation: Array,
        continuation_discount: Array,
    ) -> OptionValueDurationUpdateResult:
        """Apply one primitive transition to one option's two heads.

        The targets are exactly
        ``[reward, 1] + continuation_discount * next_predictions``.  No
        average-reward estimate is present or subtracted.  The caller is
        responsible for supplying a scalar discount in ``[0, 1]`` and for
        setting it to zero when the option terminates.
        """
        observation = jnp.asarray(observation, dtype=jnp.float32)
        next_observation = jnp.asarray(next_observation, dtype=jnp.float32)
        option_index = jnp.asarray(option_index, dtype=jnp.int32)
        reward = jnp.squeeze(jnp.asarray(reward, dtype=jnp.float32))
        continuation_discount = jnp.squeeze(jnp.asarray(continuation_discount, dtype=jnp.float32))

        option_weights = state.weights[option_index]
        predictions = option_weights @ observation
        next_predictions = option_weights @ next_observation
        cumulants = jnp.stack(
            (reward, jnp.array(1.0, dtype=jnp.float32)),
        )
        td_targets = cumulants + continuation_discount * next_predictions
        td_errors = td_targets - predictions
        step_sizes = jnp.array(
            [
                self._config.reward_step_size,
                self._config.duration_step_size,
            ],
            dtype=jnp.float32,
        )
        updated_option_weights = (
            option_weights + step_sizes[:, None] * td_errors[:, None] * observation[None, :]
        )
        new_state = state.replace(
            weights=state.weights.at[option_index].set(updated_option_weights),
            option_update_counts=state.option_update_counts.at[option_index].add(1),
            step_count=state.step_count + 1,
        )
        return OptionValueDurationUpdateResult(
            state=new_state,
            predictions=predictions,
            next_predictions=next_predictions,
            td_targets=td_targets,
            td_errors=td_errors,
            continuation_discount=continuation_discount,
        )


def run_option_value_duration_from_arrays(
    learner: OptionValueDurationLearner,
    state: OptionValueDurationState,
    observations: Float[Array, "num_steps feature_dim"],
    option_indices: Int[Array, " num_steps"],
    rewards: Float[Array, " num_steps"],
    next_observations: Float[Array, "num_steps feature_dim"],
    continuation_discounts: Float[Array, " num_steps"],
) -> OptionValueDurationArrayResult:
    """Scan online TD(0) updates over primitive option transitions."""
    start = time.time()

    def _scan_fn(
        carry: OptionValueDurationState,
        inputs: tuple[Array, Array, Array, Array, Array],
    ) -> tuple[OptionValueDurationState, tuple[Array, Array, Array, Array, Array]]:
        observation, option_index, reward, next_observation, discount = inputs
        result = learner.update(
            carry,
            observation,
            option_index,
            reward,
            next_observation,
            discount,
        )
        return result.state, (
            result.predictions,
            result.next_predictions,
            result.td_targets,
            result.td_errors,
            result.continuation_discount,
        )

    (
        final_state,
        (
            predictions,
            next_predictions,
            td_targets,
            td_errors,
            discounts,
        ),
    ) = jax.lax.scan(
        _scan_fn,
        state,
        (
            observations,
            option_indices,
            rewards,
            next_observations,
            continuation_discounts,
        ),
    )
    final_state = final_state.replace(uptime_s=final_state.uptime_s + (time.time() - start))
    return OptionValueDurationArrayResult(
        state=final_state,
        predictions=predictions,
        next_predictions=next_predictions,
        td_targets=td_targets,
        td_errors=td_errors,
        continuation_discounts=discounts,
    )
