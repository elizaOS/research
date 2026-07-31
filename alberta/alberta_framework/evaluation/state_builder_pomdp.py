"""Deterministic DEVELOPMENT-only diagnostic for causal state construction.

This is a deliberately small write/hold POMDP, intended for implementation
debugging and hyperparameter development rather than scientific acceptance.
Each continuing trial presents a binary cue once, followed by same-channel
distractors, and finally a query.  A separate cue marker says when the state
should be written; no reset is exposed between trials.

Three representation arms share the same online normalized-LMS probe:

* current observation only;
* a fixed multi-timescale trace bank; and
* the learnable gated recurrent builder with online sensitivity credit.

Only query labels in the training prefix update either the probe or recurrent
parameters.  The suffix is scored with both frozen.  Seeds are explicitly
development seeds and this module performs no confidence intervals,
preregistration, held-out evaluation, control learning, or Alberta Plan
completion test.

The arms deliberately are not resource matched: this diagnostic asks whether
the plumbing can learn at all, not whether its larger learned representation
beats a fixed baseline under an equal budget. Builder and downstream probe
storage are therefore reported separately and together.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array
from numpy.typing import NDArray

from alberta_framework.core.state_builder import (
    FixedTraceStateBuilder,
    FixedTraceStateBuilderConfig,
    IdentityStateBuilder,
    IdentityStateBuilderConfig,
    OnlineGatedStateBuilder,
    OnlineGatedStateBuilderConfig,
    StateBuilder,
)

DEVELOPMENT_SEEDS = (0, 1, 2, 3)


@dataclass(frozen=True)
class StateBuilderPomdpConfig:
    """Configuration frozen only for the local development diagnostic."""

    development_seeds: tuple[int, ...] = DEVELOPMENT_SEEDS
    training_trials: int = 1_200
    scoring_trials: int = 300
    trial_length: int = 9
    probe_step_size: float = 0.35
    probe_epsilon: float = 1.0
    trace_decay_rates: tuple[float, ...] = (0.5, 0.9, 0.99)
    recurrent_hidden_dim: int = 8
    recurrent_step_size: float = 0.03
    recurrent_gradient_clip: float = 5.0
    recurrent_initial_gate_bias: float = -2.5
    recurrent_initialization_scale: float = 0.2

    def __post_init__(self) -> None:
        if not self.development_seeds:
            raise ValueError("development_seeds must not be empty")
        if any(seed < 0 for seed in self.development_seeds):
            raise ValueError("development_seeds must be non-negative")
        if self.training_trials < 1 or self.scoring_trials < 1:
            raise ValueError("training_trials and scoring_trials must be positive")
        if self.trial_length < 3:
            raise ValueError("trial_length must include cue, distractor, and query")
        if not np.isfinite(self.probe_step_size) or self.probe_step_size <= 0.0:
            raise ValueError("probe_step_size must be finite and positive")
        if not np.isfinite(self.probe_epsilon) or self.probe_epsilon <= 0.0:
            raise ValueError("probe_epsilon must be finite and positive")
        if any(
            not np.isfinite(decay) or decay < 0.0 or decay >= 1.0
            for decay in self.trace_decay_rates
        ):
            raise ValueError("trace_decay_rates must contain finite values in [0, 1)")
        if self.recurrent_hidden_dim < 1:
            raise ValueError("recurrent_hidden_dim must be positive")
        if not np.isfinite(self.recurrent_step_size) or self.recurrent_step_size <= 0.0:
            raise ValueError("recurrent_step_size must be finite and positive")
        if not np.isfinite(self.recurrent_gradient_clip) or self.recurrent_gradient_clip <= 0.0:
            raise ValueError("recurrent_gradient_clip must be finite and positive")
        if not np.isfinite(self.recurrent_initial_gate_bias):
            raise ValueError("recurrent_initial_gate_bias must be finite")
        if (
            not np.isfinite(self.recurrent_initialization_scale)
            or self.recurrent_initialization_scale <= 0.0
        ):
            raise ValueError("recurrent_initialization_scale must be finite and positive")

    def to_config(self) -> dict[str, Any]:
        """Return a JSON-compatible diagnostic configuration."""
        payload = asdict(self)
        payload["development_seeds"] = list(self.development_seeds)
        payload["trace_decay_rates"] = list(self.trace_decay_rates)
        payload["evidence_scope"] = "development_only"
        return payload


@dataclass(frozen=True)
class StateBuilderPomdpArmResult:
    """Frozen-suffix metrics for one state-construction arm."""

    name: str
    mean_accuracy: float
    min_seed_accuracy: float
    mean_squared_error: float
    per_seed_accuracy: tuple[float, ...]
    per_seed_mse: tuple[float, ...]
    builder_output_scalars: int
    builder_trainable_scalars: int
    builder_persistent_state_bytes: int
    probe_trainable_scalars: int
    total_persistent_state_bytes: int
    mean_parameter_change_norm: float

    def to_config(self) -> dict[str, Any]:
        """Return a JSON-compatible result."""
        payload = asdict(self)
        payload["per_seed_accuracy"] = list(self.per_seed_accuracy)
        payload["per_seed_mse"] = list(self.per_seed_mse)
        return payload


@dataclass(frozen=True)
class StateBuilderPomdpResult:
    """Complete comparison, explicitly scoped to development."""

    config: StateBuilderPomdpConfig
    observation_only: StateBuilderPomdpArmResult
    fixed_trace: StateBuilderPomdpArmResult
    learned_gated: StateBuilderPomdpArmResult

    def to_config(self) -> dict[str, Any]:
        """Return a JSON-compatible result payload."""
        return {
            "schema": "alberta.state_builder_pomdp.development.v1",
            "evidence_scope": "development_only",
            "accepted_scientific_evidence": False,
            "config": self.config.to_config(),
            "arms": {
                "observation_only": self.observation_only.to_config(),
                "fixed_trace": self.fixed_trace.to_config(),
                "learned_gated": self.learned_gated.to_config(),
            },
            "limitations": [
                "development seeds only; no held-out evaluation",
                "supervised binary memory probe, not control or average reward",
                "one hand-designed cue marker and one fixed delay",
                "no non-stationarity, retention recurrence, or feature recycling",
                "arms have unequal representation and persistent-state budgets",
                "environment and learned-initialization seeds are paired one-to-one",
            ],
        }


@dataclass(frozen=True)
class _DiagnosticArrays:
    observations: Array
    targets: Array
    training_weights: Array
    scoring_weights: Array


def _make_development_sequence(
    config: StateBuilderPomdpConfig,
    seed: int,
) -> _DiagnosticArrays:
    """Construct one deterministic continuing cue/distractor sequence."""
    rng = np.random.default_rng(seed)
    total_trials = config.training_trials + config.scoring_trials
    total_steps = total_trials * config.trial_length
    cues = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), size=total_trials)
    distractors = rng.choice(
        np.asarray([-1.0, 1.0], dtype=np.float32),
        size=total_steps,
    )

    observations: NDArray[np.float32] = np.zeros(
        (total_steps, 3),
        dtype=np.float32,
    )
    observations[:, 0] = distractors
    targets: NDArray[np.float32] = np.zeros((total_steps,), dtype=np.float32)
    training_weights: NDArray[np.float32] = np.zeros(
        (total_steps,),
        dtype=np.float32,
    )
    scoring_weights: NDArray[np.float32] = np.zeros(
        (total_steps,),
        dtype=np.float32,
    )

    for trial, cue in enumerate(cues):
        start = trial * config.trial_length
        query = start + config.trial_length - 1
        observations[start, 0] = cue
        observations[start, 1] = 1.0
        observations[query, 2] = 1.0
        targets[query] = cue
        if trial < config.training_trials:
            training_weights[query] = 1.0
        else:
            scoring_weights[query] = 1.0

    return _DiagnosticArrays(
        observations=jnp.asarray(observations),
        targets=jnp.asarray(targets),
        training_weights=jnp.asarray(training_weights),
        scoring_weights=jnp.asarray(scoring_weights),
    )


def _run_arm(
    builder: StateBuilder[Any],
    arrays: _DiagnosticArrays,
    *,
    builder_key: Array,
    probe_step_size: float,
    probe_epsilon: float,
) -> tuple[float, float, float]:
    """Run one builder/probe pair and return accuracy, MSE, parameter drift."""
    initial_builder_state = builder.init(builder_key)
    initial_parameters = getattr(initial_builder_state, "parameters", None)
    weights = jnp.zeros((builder.feature_dim(),), dtype=jnp.float32)
    bias = jnp.asarray(0.0, dtype=jnp.float32)

    def step(
        carry: tuple[Any, Array, Array],
        inputs: tuple[Array, Array, Array, Array],
    ) -> tuple[tuple[Any, Array, Array], tuple[Array, Array, Array]]:
        builder_state, probe_weights, probe_bias = carry
        observation, target, training_weight, scoring_weight = inputs
        builder_state, features = builder.update(
            builder_state,
            observation,
            -1,
            0.0,
            1.0,
        )

        prediction = probe_weights @ features + probe_bias
        error = prediction - target
        representation_gradient = training_weight * error * probe_weights
        builder_state = jax.lax.cond(
            training_weight > 0.0,
            lambda current: builder.learn(
                current,
                representation_gradient,
            )[0],
            lambda current: current,
            builder_state,
        )

        denominator = jnp.asarray(probe_epsilon, dtype=jnp.float32) + features @ features
        normalized_step = (
            jnp.asarray(probe_step_size, dtype=jnp.float32) * training_weight / denominator
        )
        next_weights = probe_weights - normalized_step * error * features
        next_bias = probe_bias - normalized_step * error

        predicted_class = jnp.where(prediction >= 0.0, 1.0, -1.0)
        correct = scoring_weight * (predicted_class == target)
        squared_error = scoring_weight * error * error
        return (
            (builder_state, next_weights, next_bias),
            (correct, squared_error, scoring_weight),
        )

    (final_builder_state, _, _), (correct, squared_errors, scoring_weights) = jax.lax.scan(
        step,
        (initial_builder_state, weights, bias),
        (
            arrays.observations,
            arrays.targets,
            arrays.training_weights,
            arrays.scoring_weights,
        ),
    )
    score_count = jnp.sum(scoring_weights)
    accuracy = jnp.sum(correct) / score_count
    mse = jnp.sum(squared_errors) / score_count
    final_parameters = getattr(final_builder_state, "parameters", None)
    parameter_change = (
        jnp.asarray(0.0, dtype=jnp.float32)
        if initial_parameters is None or final_parameters is None
        else jnp.linalg.norm(final_parameters - initial_parameters)
    )
    result = (float(accuracy), float(mse), float(parameter_change))
    if not all(np.isfinite(value) for value in result):
        raise FloatingPointError("state-builder diagnostic produced non-finite metrics")
    return result


def _summarize_arm(
    name: str,
    builder: StateBuilder[Any],
    config: StateBuilderPomdpConfig,
    seed_results: list[tuple[float, float, float]],
) -> StateBuilderPomdpArmResult:
    accuracies = tuple(result[0] for result in seed_results)
    mses = tuple(result[1] for result in seed_results)
    parameter_changes = tuple(result[2] for result in seed_results)
    if not all(
        np.isfinite(value) for values in (accuracies, mses, parameter_changes) for value in values
    ):
        raise FloatingPointError(f"{name} produced non-finite per-seed metrics")
    budget = builder.resource_budget()
    probe_scalars = builder.feature_dim() + 1
    return StateBuilderPomdpArmResult(
        name=name,
        mean_accuracy=float(np.mean(accuracies)),
        min_seed_accuracy=float(np.min(accuracies)),
        mean_squared_error=float(np.mean(mses)),
        per_seed_accuracy=accuracies,
        per_seed_mse=mses,
        builder_output_scalars=budget.output_scalars,
        builder_trainable_scalars=budget.trainable_scalars,
        builder_persistent_state_bytes=budget.state_bytes,
        probe_trainable_scalars=probe_scalars,
        total_persistent_state_bytes=budget.state_bytes + 4 * probe_scalars,
        mean_parameter_change_norm=float(np.mean(parameter_changes)),
    )


def run_development_state_builder_pomdp(
    config: StateBuilderPomdpConfig | None = None,
) -> StateBuilderPomdpResult:
    """Run the three-arm deterministic diagnostic on development seeds only."""
    cfg = config or StateBuilderPomdpConfig()
    builders: tuple[tuple[str, StateBuilder[Any]], ...] = (
        (
            "observation_only",
            IdentityStateBuilder(IdentityStateBuilderConfig(observation_dim=3)),
        ),
        (
            "fixed_trace",
            FixedTraceStateBuilder(
                FixedTraceStateBuilderConfig(
                    observation_dim=3,
                    observation_decay_rates=cfg.trace_decay_rates,
                    action_decay_rates=(),
                    outcome_decay_rates=(),
                )
            ),
        ),
        (
            "learned_gated",
            OnlineGatedStateBuilder(
                OnlineGatedStateBuilderConfig(
                    observation_dim=3,
                    hidden_dim=cfg.recurrent_hidden_dim,
                    step_size=cfg.recurrent_step_size,
                    gradient_clip=cfg.recurrent_gradient_clip,
                    initial_gate_bias=cfg.recurrent_initial_gate_bias,
                    initialization_scale=cfg.recurrent_initialization_scale,
                )
            ),
        ),
    )

    summaries: dict[str, StateBuilderPomdpArmResult] = {}
    for builder_index, (name, builder) in enumerate(builders):
        results = []
        for seed in cfg.development_seeds:
            arrays = _make_development_sequence(cfg, seed)
            results.append(
                _run_arm(
                    builder,
                    arrays,
                    builder_key=jr.fold_in(jr.key(seed), builder_index),
                    probe_step_size=cfg.probe_step_size,
                    probe_epsilon=cfg.probe_epsilon,
                )
            )
        summaries[name] = _summarize_arm(name, builder, cfg, results)

    return StateBuilderPomdpResult(
        config=cfg,
        observation_only=summaries["observation_only"],
        fixed_trace=summaries["fixed_trace"],
        learned_gated=summaries["learned_gated"],
    )


__all__ = [
    "DEVELOPMENT_SEEDS",
    "StateBuilderPomdpArmResult",
    "StateBuilderPomdpConfig",
    "StateBuilderPomdpResult",
    "run_development_state_builder_pomdp",
]
