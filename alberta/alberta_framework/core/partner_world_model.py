# mypy: disable-error-code="call-arg"
"""Bounded online prediction of another agent and joint-action outcomes.

This module supplies a deliberately small contract that is absent from the
standalone behavior and world-model surfaces:

1. predict a partner action from information available *before* that action;
2. combine that distribution with joint-action reward/outcome predictions;
3. after the transition, update both models without replay or task labels.

The partner context can use one step of ordinary interaction history
(``public_signal_{t-1}, partner_action_{t-1}``) or deliberately ignore that
history to represent a stationary-partner baseline.  Both modes allocate the
same fixed table, so their storage and logical per-transition update budgets
match exactly.  A phase, task, boundary, or absolute-time identifier is never
an input.

This is a tabular development component, not a learned visual encoder, a
replication of LeWorldModel (an action-conditioned joint-embedding predictive
world model; arXiv:2603.19312), autonomous state construction, or evidence
that an Alberta Plan step is complete.  Its useful role is to make the causal
ordering and resource accounting of a minimal "predict world and agent"
experiment explicit.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from typing import Any, Literal

import chex
import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from jax import Array
from jaxtyping import Float, Int

PartnerContextMode = Literal["observable_history", "stationary"]

_CHECKPOINT_SCHEMA = "alberta.bounded_partner_world_model.v1"


@dataclasses.dataclass(frozen=True)
class BoundedPartnerWorldModelConfig:
    """Static configuration for :class:`BoundedPartnerWorldModel`.

    Args:
        n_public_signals: Cardinality of the ordinary public observation.
        n_actions: Discrete action count for both the learner and partner.
        outcome_dim: Number of bounded world-outcome scalars predicted in
            addition to reward.
        partner_context_mode: ``"observable_history"`` selects a partner row
            using the current public signal and the preceding public
            signal/observed partner action. ``"stationary"`` uses only the
            current signal. Both modes allocate all rows.
        behavior_step_size: EMA step size for the selected partner-policy row.
        world_step_size: EMA step size for the observed joint-action reward and
            outcome cell.
        reward_bound: Absolute clipping bound for reward targets.
        outcome_bound: Absolute clipping bound for outcome targets.
        min_probability: Numerical floor used only when reporting log loss.
    """

    n_public_signals: int = 2
    n_actions: int = 2
    outcome_dim: int = 2
    partner_context_mode: PartnerContextMode = "observable_history"
    behavior_step_size: float = 0.25
    world_step_size: float = 0.25
    reward_bound: float = 1.0
    outcome_bound: float = 1.0
    min_probability: float = 1e-6

    def __post_init__(self) -> None:
        """Reject configurations that cannot preserve the stated bounds."""
        if self.n_public_signals <= 0:
            raise ValueError("n_public_signals must be positive")
        if self.n_actions < 2:
            raise ValueError("n_actions must be at least 2")
        if self.outcome_dim <= 0:
            raise ValueError("outcome_dim must be positive")
        if self.partner_context_mode not in ("observable_history", "stationary"):
            raise ValueError("partner_context_mode must be observable_history or stationary")
        if not 0.0 < self.behavior_step_size <= 1.0:
            raise ValueError("behavior_step_size must lie in (0, 1]")
        if not 0.0 < self.world_step_size <= 1.0:
            raise ValueError("world_step_size must lie in (0, 1]")
        if not math.isfinite(self.reward_bound) or self.reward_bound <= 0.0:
            raise ValueError("reward_bound must be finite and positive")
        if not math.isfinite(self.outcome_bound) or self.outcome_bound <= 0.0:
            raise ValueError("outcome_bound must be finite and positive")
        if not 0.0 < self.min_probability < 1.0:
            raise ValueError("min_probability must lie in (0, 1)")

    @property
    def history_buckets(self) -> int:
        """Unknown history plus every preceding signal/action pair."""
        return 1 + self.n_public_signals * self.n_actions

    @property
    def partner_context_count(self) -> int:
        """Allocated partner-policy rows in either context mode."""
        return self.n_public_signals * self.history_buckets

    def to_config(self) -> dict[str, Any]:
        """Return a JSON-compatible configuration."""
        payload = dataclasses.asdict(self)
        payload["type"] = type(self).__name__
        return payload

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
    ) -> BoundedPartnerWorldModelConfig:
        """Strictly reconstruct from :meth:`to_config` output."""
        payload = dict(config)
        expected = {field.name for field in dataclasses.fields(cls)} | {"type"}
        if set(payload) != expected:
            raise ValueError("config fields do not match the serialized schema")
        model_type = payload.pop("type")
        if model_type != cls.__name__:
            raise ValueError(f"unexpected config type: {model_type!r}")
        return cls(**payload)


@dataclasses.dataclass(frozen=True)
class PartnerWorldResourceBudget:
    """Exact static storage and logical update surface.

    ``learned_float32_scalars_touched_per_update`` counts model-state entries
    whose values are logically replaced on each transition. It is not a FLOP,
    memory-bandwidth, or compiled-kernel accounting claim. ``state_nbytes``
    covers only array leaves in :class:`BoundedPartnerWorldState`; it excludes
    Python/config objects, evaluator traces, temporaries, and compiled code.
    """

    partner_context_rows: int
    allocated_float32_scalars: int
    allocated_int32_scalars: int
    state_nbytes: int
    learned_float32_scalars_touched_per_update: int
    administrative_int32_scalars_touched_per_update: int
    replay_capacity: int

    def to_dict(self) -> dict[str, int]:
        """Return a JSON-compatible budget record."""
        return dataclasses.asdict(self)


@chex.dataclass(frozen=True)
class BoundedPartnerWorldState:
    """Fixed-shape partner-policy and joint-outcome state."""

    partner_probabilities: Float[Array, "partner_contexts n_actions"]
    reward_predictions: Float[Array, "n_actions n_actions"]
    outcome_predictions: Float[Array, "n_actions n_actions outcome_dim"]
    previous_public_signal: Int[Array, ""]
    previous_partner_action: Int[Array, ""]
    has_history: Int[Array, ""]
    step_count: Int[Array, ""]


@chex.dataclass(frozen=True)
class PartnerPrediction:
    """Partner prediction made before observing the current partner action."""

    probabilities: Float[Array, " n_actions"]
    predicted_action: Int[Array, ""]
    context_index: Int[Array, ""]


@chex.dataclass(frozen=True)
class JointWorldPrediction:
    """Reward and world-outcome prediction for an observed joint action."""

    reward: Float[Array, ""]
    outcome: Float[Array, " outcome_dim"]


@chex.dataclass(frozen=True)
class PartnerAwareDecision:
    """Expected joint outcomes after marginalizing over the partner model."""

    partner: PartnerPrediction
    expected_rewards: Float[Array, " n_actions"]
    expected_outcomes: Float[Array, "n_actions outcome_dim"]
    greedy_action: Int[Array, ""]


@chex.dataclass(frozen=True)
class PartnerWorldUpdateResult:
    """Pre-update predictions, errors, and the updated bounded state."""

    state: BoundedPartnerWorldState
    decision: PartnerAwareDecision
    observed_joint_prediction: JointWorldPrediction
    selected_expected_reward: Float[Array, ""]
    selected_expected_outcome: Float[Array, " outcome_dim"]
    partner_action_probability: Float[Array, ""]
    partner_log_loss: Float[Array, ""]
    partner_correct: Float[Array, ""]
    observed_reward_error: Float[Array, ""]
    observed_outcome_error: Float[Array, " outcome_dim"]
    marginal_reward_error: Float[Array, ""]
    marginal_outcome_error: Float[Array, " outcome_dim"]


class BoundedPartnerWorldModel:
    """Fixed-memory online model of a partner policy and joint outcomes.

    Dynamic signal/action ids must lie in their configured ranges. The update
    surface is traceable inside :func:`jax.lax.scan`, so it intentionally does
    not perform Python-side value checks on traced scalar ids.
    """

    def __init__(self, config: BoundedPartnerWorldModelConfig):
        self._config = config

    @property
    def config(self) -> BoundedPartnerWorldModelConfig:
        """Static model configuration."""
        return self._config

    @property
    def resource_budget(self) -> PartnerWorldResourceBudget:
        """Exact state size and logical update surface."""
        cfg = self._config
        float_scalars = (
            cfg.partner_context_count * cfg.n_actions
            + cfg.n_actions * cfg.n_actions
            + cfg.n_actions * cfg.n_actions * cfg.outcome_dim
        )
        int_scalars = 4
        return PartnerWorldResourceBudget(
            partner_context_rows=cfg.partner_context_count,
            allocated_float32_scalars=float_scalars,
            allocated_int32_scalars=int_scalars,
            state_nbytes=4 * (float_scalars + int_scalars),
            learned_float32_scalars_touched_per_update=(cfg.n_actions + 1 + cfg.outcome_dim),
            administrative_int32_scalars_touched_per_update=4,
            replay_capacity=0,
        )

    def to_config(self) -> dict[str, Any]:
        """Serialize the model and configuration."""
        return {
            "type": type(self).__name__,
            "config": self._config.to_config(),
        }

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
    ) -> BoundedPartnerWorldModel:
        """Reconstruct from :meth:`to_config` output."""
        payload = dict(config)
        model_type = payload.pop("type", cls.__name__)
        if model_type != cls.__name__:
            raise ValueError(f"unexpected model type: {model_type!r}")
        if set(payload) != {"config"}:
            raise ValueError("model config must contain exactly one 'config' mapping")
        nested = payload.get("config")
        if not isinstance(nested, Mapping):
            raise ValueError("model config must contain a mapping under 'config'")
        return cls(BoundedPartnerWorldModelConfig.from_config(nested))

    def init(self) -> BoundedPartnerWorldState:
        """Initialize uniform behavior beliefs and zero world predictions."""
        cfg = self._config
        return BoundedPartnerWorldState(
            partner_probabilities=jnp.full(
                (cfg.partner_context_count, cfg.n_actions),
                1.0 / cfg.n_actions,
                dtype=jnp.float32,
            ),
            reward_predictions=jnp.zeros(
                (cfg.n_actions, cfg.n_actions),
                dtype=jnp.float32,
            ),
            outcome_predictions=jnp.zeros(
                (cfg.n_actions, cfg.n_actions, cfg.outcome_dim),
                dtype=jnp.float32,
            ),
            previous_public_signal=jnp.array(0, dtype=jnp.int32),
            previous_partner_action=jnp.array(0, dtype=jnp.int32),
            has_history=jnp.array(0, dtype=jnp.int32),
            step_count=jnp.array(0, dtype=jnp.int32),
        )

    def _context_index(
        self,
        state: BoundedPartnerWorldState,
        public_signal: Array,
    ) -> Array:
        signal = jnp.asarray(public_signal, dtype=jnp.int32)
        if self._config.partner_context_mode == "stationary":
            history_bucket = jnp.array(0, dtype=jnp.int32)
        else:
            observed_history = (
                1
                + state.previous_public_signal * self._config.n_actions
                + state.previous_partner_action
            )
            history_bucket = jnp.where(
                state.has_history > 0,
                observed_history,
                jnp.array(0, dtype=jnp.int32),
            )
        return signal * self._config.history_buckets + history_bucket

    def predict_partner(
        self,
        state: BoundedPartnerWorldState,
        public_signal: Array,
    ) -> PartnerPrediction:
        """Predict the current partner action before it is observed."""
        context_index = self._context_index(state, public_signal)
        probabilities = state.partner_probabilities[context_index]
        return PartnerPrediction(
            probabilities=probabilities,
            predicted_action=jnp.argmax(probabilities).astype(jnp.int32),
            context_index=context_index,
        )

    def predict_world(
        self,
        state: BoundedPartnerWorldState,
        own_action: Array,
        partner_action: Array,
    ) -> JointWorldPrediction:
        """Predict outcomes conditioned on both observed joint-action values."""
        own_id = jnp.asarray(own_action, dtype=jnp.int32)
        partner_id = jnp.asarray(partner_action, dtype=jnp.int32)
        return JointWorldPrediction(
            reward=state.reward_predictions[own_id, partner_id],
            outcome=state.outcome_predictions[own_id, partner_id],
        )

    def decide(
        self,
        state: BoundedPartnerWorldState,
        public_signal: Array,
    ) -> PartnerAwareDecision:
        """Rank own actions using a partner prediction made at decision time.

        This factorization assumes that the simultaneous partner action is
        conditionally independent of the candidate own action and that the
        joint-outcome table is sufficient for the public state. Reactive
        partners require an own-action-conditioned behavior model; richer
        worlds require state-conditioned outcome predictions.
        """
        partner = self.predict_partner(state, public_signal)
        expected_rewards = state.reward_predictions @ partner.probabilities
        expected_outcomes = jnp.einsum(
            "apo,p->ao",
            state.outcome_predictions,
            partner.probabilities,
        )
        return PartnerAwareDecision(
            partner=partner,
            expected_rewards=expected_rewards,
            expected_outcomes=expected_outcomes,
            greedy_action=jnp.argmax(expected_rewards).astype(jnp.int32),
        )

    def update(
        self,
        state: BoundedPartnerWorldState,
        public_signal: Array,
        own_action: Array,
        partner_action: Array,
        reward: Array,
        outcome: Array,
    ) -> PartnerWorldUpdateResult:
        """Predict first, then learn from one observed interaction.

        The current ``partner_action``, ``reward``, and ``outcome`` affect only
        the state returned by this method. All predictions and reported errors
        are computed from ``state`` before those targets are incorporated.
        """
        cfg = self._config
        signal = jnp.asarray(public_signal, dtype=jnp.int32)
        own_id = jnp.asarray(own_action, dtype=jnp.int32)
        partner_id = jnp.asarray(partner_action, dtype=jnp.int32)
        reward_target = jnp.clip(
            jnp.asarray(reward, dtype=jnp.float32),
            -cfg.reward_bound,
            cfg.reward_bound,
        )
        outcome_target = jnp.clip(
            jnp.asarray(outcome, dtype=jnp.float32).reshape((cfg.outcome_dim,)),
            -cfg.outcome_bound,
            cfg.outcome_bound,
        )

        decision = self.decide(state, signal)
        joint_prediction = self.predict_world(state, own_id, partner_id)
        selected_expected_reward = decision.expected_rewards[own_id]
        selected_expected_outcome = decision.expected_outcomes[own_id]
        action_probability = decision.partner.probabilities[partner_id]
        partner_log_loss = -jnp.log(
            jnp.maximum(
                action_probability,
                jnp.asarray(cfg.min_probability, dtype=jnp.float32),
            )
        )

        behavior_target = jax.nn.one_hot(
            partner_id,
            cfg.n_actions,
            dtype=jnp.float32,
        )
        old_behavior_row = state.partner_probabilities[decision.partner.context_index]
        next_behavior_row = old_behavior_row + cfg.behavior_step_size * (
            behavior_target - old_behavior_row
        )
        next_partner_probabilities = state.partner_probabilities.at[
            decision.partner.context_index
        ].set(next_behavior_row)

        old_reward = state.reward_predictions[own_id, partner_id]
        next_reward = old_reward + cfg.world_step_size * (reward_target - old_reward)
        next_reward_predictions = state.reward_predictions.at[
            own_id,
            partner_id,
        ].set(next_reward)

        old_outcome = state.outcome_predictions[own_id, partner_id]
        next_outcome = old_outcome + cfg.world_step_size * (outcome_target - old_outcome)
        next_outcome_predictions = state.outcome_predictions.at[
            own_id,
            partner_id,
        ].set(next_outcome)

        max_step = jnp.asarray(2_147_483_647, dtype=jnp.int32)
        next_step = jnp.minimum(state.step_count, max_step - 1) + 1
        next_state = BoundedPartnerWorldState(
            partner_probabilities=next_partner_probabilities,
            reward_predictions=next_reward_predictions,
            outcome_predictions=next_outcome_predictions,
            previous_public_signal=signal,
            previous_partner_action=partner_id,
            has_history=jnp.array(1, dtype=jnp.int32),
            step_count=next_step,
        )
        return PartnerWorldUpdateResult(
            state=next_state,
            decision=decision,
            observed_joint_prediction=joint_prediction,
            selected_expected_reward=selected_expected_reward,
            selected_expected_outcome=selected_expected_outcome,
            partner_action_probability=action_probability,
            partner_log_loss=partner_log_loss,
            partner_correct=(decision.partner.predicted_action == partner_id).astype(jnp.float32),
            observed_reward_error=joint_prediction.reward - reward_target,
            observed_outcome_error=joint_prediction.outcome - outcome_target,
            marginal_reward_error=selected_expected_reward - reward_target,
            marginal_outcome_error=selected_expected_outcome - outcome_target,
        )

    def checkpoint_payload(
        self,
        state: BoundedPartnerWorldState,
    ) -> dict[str, Any]:
        """Serialize config and state into a versioned JSON-compatible payload."""
        return {
            "schema": _CHECKPOINT_SCHEMA,
            "model": self.to_config(),
            "state": {
                "partner_probabilities": np.asarray(
                    jax.device_get(state.partner_probabilities),
                    dtype=np.float32,
                ).tolist(),
                "reward_predictions": np.asarray(
                    jax.device_get(state.reward_predictions),
                    dtype=np.float32,
                ).tolist(),
                "outcome_predictions": np.asarray(
                    jax.device_get(state.outcome_predictions),
                    dtype=np.float32,
                ).tolist(),
                "previous_public_signal": int(state.previous_public_signal),
                "previous_partner_action": int(state.previous_partner_action),
                "has_history": int(state.has_history),
                "step_count": int(state.step_count),
            },
        }

    @classmethod
    def from_checkpoint_payload(
        cls,
        checkpoint: Mapping[str, Any],
    ) -> tuple[BoundedPartnerWorldModel, BoundedPartnerWorldState]:
        """Strictly reconstruct a model and state from :meth:`checkpoint_payload`."""
        if set(checkpoint) != {"schema", "model", "state"}:
            raise ValueError("checkpoint fields do not match the v1 schema")
        if checkpoint.get("schema") != _CHECKPOINT_SCHEMA:
            raise ValueError("unexpected partner-world checkpoint schema")
        model_payload = checkpoint.get("model")
        state_payload = checkpoint.get("state")
        if not isinstance(model_payload, Mapping) or not isinstance(
            state_payload,
            Mapping,
        ):
            raise ValueError("checkpoint must contain model and state mappings")
        model = cls.from_config(model_payload)
        cfg = model.config
        required_state_keys = {
            "partner_probabilities",
            "reward_predictions",
            "outcome_predictions",
            "previous_public_signal",
            "previous_partner_action",
            "has_history",
            "step_count",
        }
        if set(state_payload) != required_state_keys:
            raise ValueError("checkpoint state fields do not match the v1 schema")

        partner_probabilities: npt.NDArray[np.float32] = np.asarray(
            state_payload["partner_probabilities"],
            dtype=np.float32,
        )
        reward_predictions: npt.NDArray[np.float32] = np.asarray(
            state_payload["reward_predictions"],
            dtype=np.float32,
        )
        outcome_predictions: npt.NDArray[np.float32] = np.asarray(
            state_payload["outcome_predictions"],
            dtype=np.float32,
        )
        expected_shapes = (
            (cfg.partner_context_count, cfg.n_actions),
            (cfg.n_actions, cfg.n_actions),
            (cfg.n_actions, cfg.n_actions, cfg.outcome_dim),
        )
        actual_shapes = (
            partner_probabilities.shape,
            reward_predictions.shape,
            outcome_predictions.shape,
        )
        if actual_shapes != expected_shapes:
            raise ValueError(
                f"checkpoint state shapes {actual_shapes!r} do not match {expected_shapes!r}"
            )
        if not all(
            np.all(np.isfinite(array))
            for array in (
                partner_probabilities,
                reward_predictions,
                outcome_predictions,
            )
        ):
            raise ValueError("checkpoint arrays must be finite")
        if np.any(partner_probabilities < 0.0) or not np.allclose(
            partner_probabilities.sum(axis=1),
            1.0,
            atol=1e-6,
            rtol=0.0,
        ):
            raise ValueError("checkpoint partner rows must be probability simplexes")
        if np.any(np.abs(reward_predictions) > cfg.reward_bound + 1e-6):
            raise ValueError("checkpoint rewards exceed reward_bound")
        if np.any(np.abs(outcome_predictions) > cfg.outcome_bound + 1e-6):
            raise ValueError("checkpoint outcomes exceed outcome_bound")

        def strict_int(field: str) -> int:
            value = state_payload[field]
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"checkpoint {field} must be an integer")
            return int(value)

        previous_signal = strict_int("previous_public_signal")
        previous_action = strict_int("previous_partner_action")
        has_history = strict_int("has_history")
        step_count = strict_int("step_count")
        if not 0 <= previous_signal < cfg.n_public_signals:
            raise ValueError("checkpoint previous_public_signal is out of range")
        if not 0 <= previous_action < cfg.n_actions:
            raise ValueError("checkpoint previous_partner_action is out of range")
        if has_history not in (0, 1):
            raise ValueError("checkpoint has_history must be 0 or 1")
        if not 0 <= step_count <= np.iinfo(np.int32).max:
            raise ValueError("checkpoint step_count is out of int32 range")
        if (step_count == 0) != (has_history == 0):
            raise ValueError("checkpoint history flag is inconsistent with step_count")
        if step_count == 0 and (previous_signal != 0 or previous_action != 0):
            raise ValueError("checkpoint initial history fields must be zero")

        state = BoundedPartnerWorldState(
            partner_probabilities=jnp.asarray(
                partner_probabilities,
                dtype=jnp.float32,
            ),
            reward_predictions=jnp.asarray(
                reward_predictions,
                dtype=jnp.float32,
            ),
            outcome_predictions=jnp.asarray(
                outcome_predictions,
                dtype=jnp.float32,
            ),
            previous_public_signal=jnp.asarray(
                previous_signal,
                dtype=jnp.int32,
            ),
            previous_partner_action=jnp.asarray(
                previous_action,
                dtype=jnp.int32,
            ),
            has_history=jnp.asarray(has_history, dtype=jnp.int32),
            step_count=jnp.asarray(step_count, dtype=jnp.int32),
        )
        return model, state
