# mypy: disable-error-code="call-arg"
"""Development-only A-B-A diagnostic for joint partner/world prediction.

The changing partner sees the same public discrete signal as the learner. In
mode A it emits the signal; in mode B it emits the signal plus one modulo the
action count; mode A then returns. The learner never receives the mode,
segment, boundary, phase index, or absolute step. It may remember only the
preceding ordinary signal and observed partner action.

Two otherwise identical bounded partner-aware models are compared:

* ``observable_history`` conditions the partner prediction on that one-step
  interaction history;
* ``stationary`` ignores history while retaining the same allocated table,
  joint-outcome model, optimizer, exploration stream, and update budget.

This is a one-step-history context ablation. It is not a comparison against a
world-only or no-partner model: both arms learn a partner distribution and
marginalize their joint-outcome tables through it.

The outcome is a coordination indicator plus the joint-action mean, and reward
is +1 for matching the partner and -1 otherwise. Both are learned only from
executed interactions. Partner actions are predicted before observation;
world outcomes are predicted from the selected own action combined with the
predicted distribution and, separately, with the subsequently observed
partner action before reward/outcome learning.

This seed-0 lane is for mechanism development and regression testing only. It
uses no held-out seeds, confidence interval, immutable artifact, or promoted
threshold, and it does not replicate LeWorldModel or establish continual
learning, autonomous feature discovery, or Alberta Plan completion.
"""

from __future__ import annotations

import dataclasses
from typing import Any, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import numpy.typing as npt
from jax import Array
from jaxtyping import Float, Int

from alberta_framework.core.partner_world_model import (
    BoundedPartnerWorldModel,
    BoundedPartnerWorldModelConfig,
    BoundedPartnerWorldState,
    PartnerContextMode,
    PartnerWorldResourceBudget,
)


@dataclasses.dataclass(frozen=True)
class ChangingPartnerDiagnosticConfig:
    """Frozen shape and development calibration for the A-B-A stream.

    The default ``behavior_step_size`` (0.05), ``world_step_size`` (0.25),
    and ``exploration_probability`` (0.10) match the corresponding defaults
    of :class:`~alberta_framework.core.integrated_hidden_partner.IntegratedHiddenPartnerConfig`,
    so this diagnostic exercises the substrate at its shipped rates.
    ``early_window`` averages the first steps of each phase — adaptation
    speed immediately after an unsignaled switch — while ``tail_window``
    averages the last steps, i.e. settled end-of-phase performance.
    """

    seed: int = 0
    phase_length: int = 512
    phase_offsets: tuple[int, ...] = (0, 1, 0)
    n_actions: int = 2
    behavior_step_size: float = 0.05
    world_step_size: float = 0.25
    exploration_probability: float = 0.10
    early_window: int = 64
    tail_window: int = 128

    def __post_init__(self) -> None:
        """Validate that the requested stream is a genuine A-B-A diagnostic."""
        if self.phase_length <= 0:
            raise ValueError("phase_length must be positive")
        if len(self.phase_offsets) != 3:
            raise ValueError("phase_offsets must contain exactly A, B, A")
        if self.phase_offsets[0] != self.phase_offsets[2]:
            raise ValueError("the first and third phase offsets must match")
        if self.phase_offsets[0] == self.phase_offsets[1]:
            raise ValueError("the B offset must differ from A")
        if self.n_actions < 2:
            raise ValueError("n_actions must be at least 2")
        if any(not 0 <= offset < self.n_actions for offset in self.phase_offsets):
            raise ValueError("every phase offset must lie in [0, n_actions)")
        if not 0.0 < self.behavior_step_size <= 1.0:
            raise ValueError("behavior_step_size must lie in (0, 1]")
        if not 0.0 < self.world_step_size <= 1.0:
            raise ValueError("world_step_size must lie in (0, 1]")
        if not 0.0 <= self.exploration_probability <= 1.0:
            raise ValueError("exploration_probability must lie in [0, 1]")
        if not 0 < self.early_window <= self.phase_length:
            raise ValueError("early_window must lie in [1, phase_length]")
        if not 0 < self.tail_window <= self.phase_length:
            raise ValueError("tail_window must lie in [1, phase_length]")

    @property
    def num_steps(self) -> int:
        """Total continuing transitions (there are no resets between phases)."""
        return self.phase_length * len(self.phase_offsets)

    def to_config(self) -> dict[str, Any]:
        """Return a JSON-compatible record."""
        payload = dataclasses.asdict(self)
        payload["phase_offsets"] = list(self.phase_offsets)
        payload["development_only"] = True
        return payload


@chex.dataclass(frozen=True)
class PartnerWorldArmTrace:
    """Prequential trace for one context-mode arm."""

    public_signals: Int[Array, " num_steps"]
    partner_actions: Int[Array, " num_steps"]
    predicted_partner_actions: Int[Array, " num_steps"]
    partner_action_probabilities: Float[Array, " num_steps"]
    partner_log_losses: Float[Array, " num_steps"]
    partner_correct: Float[Array, " num_steps"]
    greedy_actions: Int[Array, " num_steps"]
    executed_actions: Int[Array, " num_steps"]
    greedy_decision_correct: Float[Array, " num_steps"]
    rewards: Float[Array, " num_steps"]
    observed_reward_squared_errors: Float[Array, " num_steps"]
    observed_outcome_mses: Float[Array, " num_steps"]
    marginal_reward_squared_errors: Float[Array, " num_steps"]
    marginal_outcome_mses: Float[Array, " num_steps"]
    context_indices: Int[Array, " num_steps"]
    final_state: BoundedPartnerWorldState


@dataclasses.dataclass(frozen=True)
class ChangingPartnerDiagnosticResult:
    """Matched A-B-A traces and exact static resource budgets."""

    config: ChangingPartnerDiagnosticConfig
    history_aware: PartnerWorldArmTrace
    stationary: PartnerWorldArmTrace
    history_aware_budget: PartnerWorldResourceBudget
    stationary_budget: PartnerWorldResourceBudget


def _model(
    config: ChangingPartnerDiagnosticConfig,
    context_mode: PartnerContextMode,
) -> BoundedPartnerWorldModel:
    return BoundedPartnerWorldModel(
        BoundedPartnerWorldModelConfig(
            n_public_signals=config.n_actions,
            n_actions=config.n_actions,
            outcome_dim=2,
            partner_context_mode=context_mode,
            behavior_step_size=config.behavior_step_size,
            world_step_size=config.world_step_size,
            reward_bound=1.0,
            outcome_bound=1.0,
        )
    )


def _run_arm(
    model: BoundedPartnerWorldModel,
    public_signals: Array,
    partner_actions: Array,
    explore: Array,
    random_actions: Array,
) -> PartnerWorldArmTrace:
    """Run one uninterrupted online arm with common random numbers."""

    def step(
        state: BoundedPartnerWorldState,
        inputs: tuple[Array, Array, Array, Array],
    ) -> tuple[BoundedPartnerWorldState, tuple[Array, ...]]:
        public_signal, partner_action, do_explore, random_action = inputs
        decision = model.decide(state, public_signal)
        own_action = jnp.where(
            do_explore,
            random_action,
            decision.greedy_action,
        ).astype(jnp.int32)
        coordinated = (own_action == partner_action).astype(jnp.float32)
        reward = 2.0 * coordinated - 1.0
        joint_action_mean = (
            own_action.astype(jnp.float32) + partner_action.astype(jnp.float32)
        ) / (2.0 * (model.config.n_actions - 1))
        outcome = jnp.stack((coordinated, joint_action_mean))
        update = model.update(
            state,
            public_signal,
            own_action,
            partner_action,
            reward,
            outcome,
        )
        return update.state, (
            update.decision.partner.predicted_action,
            update.partner_action_probability,
            update.partner_log_loss,
            update.partner_correct,
            update.decision.greedy_action,
            own_action,
            (update.decision.greedy_action == partner_action).astype(jnp.float32),
            reward,
            update.observed_reward_error**2,
            jnp.mean(update.observed_outcome_error**2),
            update.marginal_reward_error**2,
            jnp.mean(update.marginal_outcome_error**2),
            update.decision.partner.context_index,
        )

    final_state, outputs = jax.lax.scan(
        step,
        model.init(),
        (public_signals, partner_actions, explore, random_actions),
    )
    (
        predicted_partner_actions,
        partner_action_probabilities,
        partner_log_losses,
        partner_correct,
        greedy_actions,
        executed_actions,
        greedy_decision_correct,
        rewards,
        observed_reward_squared_errors,
        observed_outcome_mses,
        marginal_reward_squared_errors,
        marginal_outcome_mses,
        context_indices,
    ) = outputs
    return PartnerWorldArmTrace(
        public_signals=public_signals,
        partner_actions=partner_actions,
        predicted_partner_actions=predicted_partner_actions,
        partner_action_probabilities=partner_action_probabilities,
        partner_log_losses=partner_log_losses,
        partner_correct=partner_correct,
        greedy_actions=greedy_actions,
        executed_actions=executed_actions,
        greedy_decision_correct=greedy_decision_correct,
        rewards=rewards,
        observed_reward_squared_errors=observed_reward_squared_errors,
        observed_outcome_mses=observed_outcome_mses,
        marginal_reward_squared_errors=marginal_reward_squared_errors,
        marginal_outcome_mses=marginal_outcome_mses,
        context_indices=context_indices,
        final_state=final_state,
    )


def run_changing_partner_diagnostic(
    config: ChangingPartnerDiagnosticConfig | None = None,
) -> ChangingPartnerDiagnosticResult:
    """Run the deterministic-seed development comparison.

    The evaluator uses phase offsets only to generate the external partner
    action stream and slice metrics. Neither model sees those offsets or phase
    boundaries.
    """
    cfg = config or ChangingPartnerDiagnosticConfig()
    signal_key, explore_key, random_action_key = jr.split(jr.key(cfg.seed), 3)
    public_signals = jr.randint(
        signal_key,
        (cfg.num_steps,),
        minval=0,
        maxval=cfg.n_actions,
        dtype=jnp.int32,
    )
    offsets = jnp.repeat(
        jnp.asarray(cfg.phase_offsets, dtype=jnp.int32),
        cfg.phase_length,
    )
    partner_actions = jnp.mod(public_signals + offsets, cfg.n_actions).astype(jnp.int32)
    explore = jr.bernoulli(
        explore_key,
        cfg.exploration_probability,
        (cfg.num_steps,),
    )
    random_actions = jr.randint(
        random_action_key,
        (cfg.num_steps,),
        minval=0,
        maxval=cfg.n_actions,
        dtype=jnp.int32,
    )

    aware_model = _model(cfg, "observable_history")
    stationary_model = _model(cfg, "stationary")
    aware = _run_arm(
        aware_model,
        public_signals,
        partner_actions,
        explore,
        random_actions,
    )
    stationary = _run_arm(
        stationary_model,
        public_signals,
        partner_actions,
        explore,
        random_actions,
    )
    return ChangingPartnerDiagnosticResult(
        config=cfg,
        history_aware=aware,
        stationary=stationary,
        history_aware_budget=aware_model.resource_budget,
        stationary_budget=stationary_model.resource_budget,
    )


def _phase_means(values: np.ndarray, config: ChangingPartnerDiagnosticConfig) -> list[float]:
    phases = values.reshape((len(config.phase_offsets), config.phase_length))
    return [float(value) for value in phases.mean(axis=1)]


def _phase_window_means(
    values: np.ndarray,
    config: ChangingPartnerDiagnosticConfig,
    *,
    first: bool,
) -> list[float]:
    phases = values.reshape((len(config.phase_offsets), config.phase_length))
    width = config.early_window if first else config.tail_window
    window = phases[:, :width] if first else phases[:, -width:]
    return [float(value) for value in window.mean(axis=1)]


def _arm_summary(
    trace: PartnerWorldArmTrace,
    config: ChangingPartnerDiagnosticConfig,
) -> dict[str, Any]:
    partner_correct = np.asarray(jax.device_get(trace.partner_correct), dtype=np.float64)
    decision_correct = np.asarray(
        jax.device_get(trace.greedy_decision_correct),
        dtype=np.float64,
    )
    rewards = np.asarray(jax.device_get(trace.rewards), dtype=np.float64)
    partner_log_losses = np.asarray(
        jax.device_get(trace.partner_log_losses),
        dtype=np.float64,
    )
    observed_reward_mse = np.asarray(
        jax.device_get(trace.observed_reward_squared_errors),
        dtype=np.float64,
    )
    observed_outcome_mse = np.asarray(
        jax.device_get(trace.observed_outcome_mses),
        dtype=np.float64,
    )
    marginal_reward_mse = np.asarray(
        jax.device_get(trace.marginal_reward_squared_errors),
        dtype=np.float64,
    )
    marginal_outcome_mse = np.asarray(
        jax.device_get(trace.marginal_outcome_mses),
        dtype=np.float64,
    )
    own_actions = np.asarray(
        jax.device_get(trace.executed_actions),
        dtype=np.int64,
    )
    partner_actions = np.asarray(
        jax.device_get(trace.partner_actions),
        dtype=np.int64,
    )
    visits: npt.NDArray[np.int64] = np.zeros(
        (config.n_actions, config.n_actions),
        dtype=np.int64,
    )
    np.add.at(visits, (own_actions, partner_actions), 1)

    action_ids: npt.NDArray[np.float64] = np.arange(
        config.n_actions,
        dtype=np.float64,
    )
    grids = cast(
        tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
        np.meshgrid(
            action_ids,
            action_ids,
            indexing="ij",
        ),
    )
    own_grid: npt.NDArray[np.float64] = grids[0]
    partner_grid: npt.NDArray[np.float64] = grids[1]
    coordinated = (own_grid == partner_grid).astype(np.float64)
    target_rewards = 2.0 * coordinated - 1.0
    target_outcomes = np.stack(
        (
            coordinated,
            (own_grid + partner_grid) / (2.0 * (config.n_actions - 1)),
        ),
        axis=-1,
    )
    final_rewards = np.asarray(
        jax.device_get(trace.final_state.reward_predictions),
        dtype=np.float64,
    )
    final_outcomes = np.asarray(
        jax.device_get(trace.final_state.outcome_predictions),
        dtype=np.float64,
    )
    return {
        "overall": {
            "partner_accuracy": float(partner_correct.mean()),
            "partner_log_loss": float(partner_log_losses.mean()),
            "greedy_decision_fidelity": float(decision_correct.mean()),
            "executed_mean_reward": float(rewards.mean()),
            "observed_joint_reward_mse": float(observed_reward_mse.mean()),
            "observed_joint_outcome_mse": float(observed_outcome_mse.mean()),
            "partner_marginal_reward_mse": float(marginal_reward_mse.mean()),
            "partner_marginal_outcome_mse": float(marginal_outcome_mse.mean()),
            "final_full_table_reward_mse": float(np.mean((final_rewards - target_rewards) ** 2)),
            "final_full_table_outcome_mse": float(np.mean((final_outcomes - target_outcomes) ** 2)),
        },
        "joint_action_visit_counts": visits.tolist(),
        "phase_partner_accuracy": _phase_means(partner_correct, config),
        "phase_greedy_decision_fidelity": _phase_means(decision_correct, config),
        "phase_executed_mean_reward": _phase_means(rewards, config),
        "early_partner_accuracy": _phase_window_means(
            partner_correct,
            config,
            first=True,
        ),
        "early_greedy_decision_fidelity": _phase_window_means(
            decision_correct,
            config,
            first=True,
        ),
        "tail_partner_accuracy": _phase_window_means(
            partner_correct,
            config,
            first=False,
        ),
        "tail_greedy_decision_fidelity": _phase_window_means(
            decision_correct,
            config,
            first=False,
        ),
        "phase_partner_log_loss": _phase_means(partner_log_losses, config),
        "phase_observed_joint_reward_mse": _phase_means(
            observed_reward_mse,
            config,
        ),
        "phase_observed_joint_outcome_mse": _phase_means(
            observed_outcome_mse,
            config,
        ),
        "phase_partner_marginal_reward_mse": _phase_means(
            marginal_reward_mse,
            config,
        ),
        "phase_partner_marginal_outcome_mse": _phase_means(
            marginal_outcome_mse,
            config,
        ),
    }


def summarize_changing_partner_diagnostic(
    result: ChangingPartnerDiagnosticResult,
) -> dict[str, Any]:
    """Return exact JSON-compatible development metrics and comparisons."""
    aware = _arm_summary(result.history_aware, result.config)
    stationary = _arm_summary(result.stationary, result.config)
    aware_overall = aware["overall"]
    stationary_overall = stationary["overall"]
    aware_early = aware["early_greedy_decision_fidelity"]
    stationary_early = stationary["early_greedy_decision_fidelity"]
    aware_tail = aware["tail_greedy_decision_fidelity"]
    return {
        "claim_scope": "development_only",
        "held_out": False,
        "config": result.config.to_config(),
        "history_aware_budget": result.history_aware_budget.to_dict(),
        "stationary_budget": result.stationary_budget.to_dict(),
        "history_aware": aware,
        "stationary": stationary,
        "comparison_scope": (
            "one-step observable-history context versus current-signal-only "
            "stationary context; both arms contain partner and world models"
        ),
        "comparisons": {
            "history_context_partner_accuracy_gain_over_stationary": (
                aware_overall["partner_accuracy"] - stationary_overall["partner_accuracy"]
            ),
            "history_context_decision_fidelity_gain_over_stationary": (
                aware_overall["greedy_decision_fidelity"]
                - stationary_overall["greedy_decision_fidelity"]
            ),
            "b_phase_early_history_context_fidelity_gain": (aware_early[1] - stationary_early[1]),
            "return_a_early_history_context_fidelity_gain": (aware_early[2] - stationary_early[2]),
            "history_context_return_a_early_minus_initial_a_tail_fidelity": (
                aware_early[2] - aware_tail[0]
            ),
        },
    }
