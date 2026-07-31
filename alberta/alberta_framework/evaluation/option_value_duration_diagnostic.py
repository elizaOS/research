"""Deterministic development diagnostic for option value plus duration.

This is a mechanism-level (L1), development-only diagnostic.  It uses supplied
one-hot features and supplied deterministic options; it has no random seeds,
held-out split, preregistered threshold, or completion claim.

At a recurring decision state, the slow option returns 6 reward in 10 primitive
steps and the fast option returns 4 reward in 2 primitive steps.  Conventional
return alone therefore ranks the slow option first even though its continuing
reward rate is lower.  The two-head predictor learns both quantities online,
and return divided by predicted duration ranks the fast option first.

Both options return to the same recurring decision state, so this renewal-rate
comparison is exact here.  The ratio is not claimed to solve general
semi-Markov control when options have different successor-state values.
"""

from __future__ import annotations

import dataclasses
import math

import jax.numpy as jnp
from jax import Array

from alberta_framework.core.option_value_duration import (
    OptionValueDurationConfig,
    OptionValueDurationLearner,
    run_option_value_duration_from_arrays,
)

SLOW_OPTION = 0
FAST_OPTION = 1


@dataclasses.dataclass(frozen=True)
class OptionValueDurationDiagnosticConfig:
    """Configuration for the deterministic two-option diagnostic."""

    slow_return: float = 6.0
    slow_duration: int = 10
    fast_return: float = 4.0
    fast_duration: int = 2
    executions_per_option: int = 12

    def __post_init__(self) -> None:
        """Reject configurations that do not preserve the diagnostic."""
        if self.slow_duration < 1 or self.fast_duration < 1:
            raise ValueError("option durations must be positive")
        if not math.isfinite(self.slow_return) or not math.isfinite(self.fast_return):
            raise ValueError("option returns must be finite")
        if self.slow_return <= self.fast_return:
            raise ValueError("slow_return must exceed fast_return")
        if self.slow_return / self.slow_duration >= self.fast_return / self.fast_duration:
            raise ValueError("the fast option must have the larger true reward rate")
        if self.executions_per_option < max(self.slow_duration, self.fast_duration):
            raise ValueError("executions_per_option must cover the longest TD(0) backup chain")


@dataclasses.dataclass(frozen=True)
class OptionValueDurationDiagnosticResult:
    """Learned rankings and exact deterministic reference quantities."""

    transition_count: int
    learned_reward_values: tuple[float, float]
    learned_durations: tuple[float, float]
    learned_reward_rates: tuple[float, float]
    true_reward_rates: tuple[float, float]
    option_update_counts: tuple[int, int]
    value_only_choice: int
    reward_rate_choice: int
    optimal_reward_rate_choice: int
    mechanism_passed: bool
    evidence_level: str = "L1 deterministic development diagnostic"


def _build_continuing_transitions(
    config: OptionValueDurationDiagnosticConfig,
) -> tuple[Array, Array, Array, Array, Array]:
    """Alternate complete supplied-option executions without episodic resets."""
    feature_dim = max(config.slow_duration, config.fast_duration)
    basis = jnp.eye(feature_dim, dtype=jnp.float32)
    observations: list[Array] = []
    option_indices: list[int] = []
    rewards: list[float] = []
    next_observations: list[Array] = []
    discounts: list[float] = []

    option_specs = (
        (SLOW_OPTION, config.slow_duration, config.slow_return),
        (FAST_OPTION, config.fast_duration, config.fast_return),
    )
    for _ in range(config.executions_per_option):
        for option_index, duration, terminal_return in option_specs:
            for option_step in range(duration):
                terminated = option_step == duration - 1
                observations.append(basis[option_step])
                option_indices.append(option_index)
                rewards.append(terminal_return if terminated else 0.0)
                next_observations.append(basis[0] if terminated else basis[option_step + 1])
                discounts.append(0.0 if terminated else 1.0)

    return (
        jnp.stack(observations),
        jnp.asarray(option_indices, dtype=jnp.int32),
        jnp.asarray(rewards, dtype=jnp.float32),
        jnp.stack(next_observations),
        jnp.asarray(discounts, dtype=jnp.float32),
    )


def run_option_value_duration_diagnostic(
    config: OptionValueDurationDiagnosticConfig | None = None,
) -> OptionValueDurationDiagnosticResult:
    """Run the online continuing semi-Markov mechanism diagnostic.

    Unit step-sizes make each deterministic one-hot TD backup exact.  Repeating
    each option at least as many times as its duration propagates its terminal
    return and duration all the way back to the recurring decision feature.
    """
    cfg = config or OptionValueDurationDiagnosticConfig()
    (
        observations,
        option_indices,
        rewards,
        next_observations,
        discounts,
    ) = _build_continuing_transitions(cfg)
    learner = OptionValueDurationLearner(
        n_options=2,
        config=OptionValueDurationConfig(
            reward_step_size=1.0,
            duration_step_size=1.0,
        ),
    )
    state = learner.init(int(observations.shape[1]))
    result = run_option_value_duration_from_arrays(
        learner,
        state,
        observations,
        option_indices,
        rewards,
        next_observations,
        discounts,
    )
    decision_feature = jnp.eye(
        int(observations.shape[1]),
        dtype=jnp.float32,
    )[0]
    predictions = learner.predict(result.state, decision_feature)

    learned_reward_values = (
        float(predictions.reward_values[SLOW_OPTION]),
        float(predictions.reward_values[FAST_OPTION]),
    )
    learned_durations = (
        float(predictions.durations[SLOW_OPTION]),
        float(predictions.durations[FAST_OPTION]),
    )
    learned_reward_rates = (
        float(predictions.reward_rates[SLOW_OPTION]),
        float(predictions.reward_rates[FAST_OPTION]),
    )
    true_reward_rates = (
        cfg.slow_return / cfg.slow_duration,
        cfg.fast_return / cfg.fast_duration,
    )
    value_only_choice = int(jnp.argmax(predictions.reward_values))
    reward_rate_choice = int(jnp.argmax(predictions.reward_rates))
    optimal_choice = (
        FAST_OPTION
        if true_reward_rates[FAST_OPTION] > true_reward_rates[SLOW_OPTION]
        else SLOW_OPTION
    )
    option_update_counts = (
        int(result.state.option_update_counts[SLOW_OPTION]),
        int(result.state.option_update_counts[FAST_OPTION]),
    )
    values_exact = bool(
        jnp.allclose(
            predictions.reward_values,
            jnp.array([cfg.slow_return, cfg.fast_return], dtype=jnp.float32),
        )
    )
    durations_exact = bool(
        jnp.allclose(
            predictions.durations,
            jnp.array([cfg.slow_duration, cfg.fast_duration], dtype=jnp.float32),
        )
    )
    mechanism_passed = (
        value_only_choice == SLOW_OPTION
        and reward_rate_choice == optimal_choice == FAST_OPTION
        and values_exact
        and durations_exact
    )
    return OptionValueDurationDiagnosticResult(
        transition_count=int(observations.shape[0]),
        learned_reward_values=learned_reward_values,
        learned_durations=learned_durations,
        learned_reward_rates=learned_reward_rates,
        true_reward_rates=true_reward_rates,
        option_update_counts=option_update_counts,
        value_only_choice=value_only_choice,
        reward_rate_choice=reward_rate_choice,
        optimal_reward_rate_choice=optimal_choice,
        mechanism_passed=mechanism_passed,
    )
