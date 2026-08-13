# mypy: disable-error-code="call-arg,name-defined"
"""Bounded online joint-action outcome model with external partner beliefs.

This module is the smallest causal world-model surface needed by a
simultaneous two-agent continual-control experiment:

1. another component predicts the partner's current action before it occurs;
2. this model exposes predictions for every candidate joint action;
3. the caller marginalizes those cells under the pre-action partner belief;
4. only after the real transition is scored does the executed cell update.

The partner policy is intentionally *not* modeled here. Keeping behavior and
world prediction separate prevents a changing partner from being conflated
with changing environment physics, and lets learned-state and stationary
partner baselines share the same joint-outcome table.

The reference implementation is a fixed-size float32 EMA table. It is a
development mechanism for causal ordering and resource accounting, not a
general function-approximation or continual-world-model result.
"""

from __future__ import annotations

import dataclasses
import functools
import math
from collections.abc import Mapping
from typing import Any

import chex
import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Bool, Float, Int, UInt

JOINT_OUTCOME_STATE_SCHEMA = "alberta.bounded-joint-outcome-state.v2"
JOINT_OUTCOME_LIFETIME_COUNTER_NBYTES = 12
JOINT_OUTCOME_LIFETIME_COUNTER_DELTA_NBYTES = 8
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
        raise ValueError("joint-outcome lifetime words must have shape (2,)")
    if getattr(words, "dtype", None) != jnp.dtype(jnp.uint32):
        raise TypeError("joint-outcome lifetime words must have dtype uint32")
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
    """Validate exact identity against saturating compatibility telemetry."""

    if getattr(words, "shape", None) != (2,):
        raise ValueError("joint-outcome lifetime words must have shape (2,)")
    if getattr(words, "dtype", None) != jnp.dtype(jnp.uint32):
        raise TypeError("joint-outcome lifetime words must have dtype uint32")
    if getattr(telemetry, "shape", None) != ():
        raise ValueError("joint-outcome step_count must be scalar")
    if getattr(telemetry, "dtype", None) != jnp.dtype(jnp.int32):
        raise TypeError("joint-outcome step_count must have dtype int32")
    maximum_i32 = jnp.asarray(_INT32_MAX, dtype=jnp.int32)
    below_saturation = (words[0] == jnp.asarray(0, dtype=jnp.uint32)) & (
        words[1] < jnp.asarray(_INT32_MAX, dtype=jnp.uint32)
    )
    return (telemetry >= 0) & jnp.where(
        below_saturation,
        telemetry == words[1].astype(jnp.int32),
        telemetry == maximum_i32,
    )


__all__ = [
    "JOINT_OUTCOME_LIFETIME_COUNTER_DELTA_NBYTES",
    "JOINT_OUTCOME_LIFETIME_COUNTER_NBYTES",
    "JOINT_OUTCOME_STATE_SCHEMA",
    "BoundedJointOutcomeConfig",
    "BoundedJointOutcomeModel",
    "BoundedJointOutcomeState",
    "JointOutcomePrediction",
    "JointOutcomeResourceBudget",
    "JointOutcomeUpdateResult",
    "PartnerMarginalDecision",
    "joint_outcome_lifetime_counter_nbytes",
    "measure_bounded_joint_outcome_state_nbytes",
    "migrate_legacy_bounded_joint_outcome_state",
]


@dataclasses.dataclass(frozen=True)
class BoundedJointOutcomeConfig:
    """Configuration for :class:`BoundedJointOutcomeModel`."""

    n_actions: int = 2
    outcome_dim: int = 1
    step_size: float = 0.25
    reward_bound: float = 1.0
    outcome_bound: float = 1.0
    probability_tolerance: float = 1e-5

    def __post_init__(self) -> None:
        """Reject ambiguous dimensions and non-finite numeric controls."""
        if (
            isinstance(self.n_actions, bool)
            or not isinstance(self.n_actions, int)
            or self.n_actions < 2
        ):
            raise ValueError("n_actions must be an integer of at least 2")
        if (
            isinstance(self.outcome_dim, bool)
            or not isinstance(self.outcome_dim, int)
            or self.outcome_dim < 1
        ):
            raise ValueError("outcome_dim must be a positive integer")
        if not math.isfinite(self.step_size) or not 0.0 < self.step_size <= 1.0:
            raise ValueError("step_size must be finite and lie in (0, 1]")
        if not math.isfinite(self.reward_bound) or self.reward_bound <= 0.0:
            raise ValueError("reward_bound must be finite and positive")
        if not math.isfinite(self.outcome_bound) or self.outcome_bound <= 0.0:
            raise ValueError("outcome_bound must be finite and positive")
        if not math.isfinite(self.probability_tolerance) or self.probability_tolerance <= 0.0:
            raise ValueError("probability_tolerance must be finite and positive")

    def to_config(self) -> dict[str, Any]:
        """Return a strict JSON-compatible configuration."""
        return {
            "type": type(self).__name__,
            **dataclasses.asdict(self),
        }

    @classmethod
    def from_config(
        cls,
        payload: Mapping[str, Any],
    ) -> BoundedJointOutcomeConfig:
        """Strictly reconstruct :meth:`to_config` output."""
        values = dict(payload)
        expected = {field.name for field in dataclasses.fields(cls)} | {"type"}
        if set(values) != expected:
            raise ValueError("joint-outcome config fields do not match the schema")
        model_type = values.pop("type")
        if model_type != cls.__name__:
            raise ValueError(f"unexpected config type: {model_type!r}")
        return cls(**values)


@dataclasses.dataclass(frozen=True)
class JointOutcomeResourceBudget:
    """Exact persistent array accounting for the fixed table."""

    joint_cells: int
    allocated_float32_scalars: int
    allocated_int32_scalars: int
    allocated_uint32_scalars: int
    state_nbytes: int
    learned_float32_scalars_touched_per_update: int
    administrative_int32_scalars_touched_per_update: int
    administrative_uint32_scalars_touched_per_update: int
    planner_cell_evaluations_per_decision: int
    replay_capacity: int

    def to_dict(self) -> dict[str, int]:
        """Return a JSON-compatible resource record."""
        return dataclasses.asdict(self)


@chex.dataclass(frozen=True)
class BoundedJointOutcomeState:
    """Fixed-shape reward/outcome tables and monotonic visit counters."""

    reward_predictions: Float[Array, "n_actions n_actions"]
    outcome_predictions: Float[Array, "n_actions n_actions outcome_dim"]
    visit_counts: Int[Array, "n_actions n_actions"]
    step_count: Int[Array, ""]
    step_words: UInt[Array, " 2"]


@chex.dataclass(frozen=True)
class JointOutcomePrediction:
    """Pre-update prediction for one candidate joint action."""

    reward: Float[Array, ""]
    outcome: Float[Array, " outcome_dim"]
    visit_count: Int[Array, ""]


@chex.dataclass(frozen=True)
class PartnerMarginalDecision:
    """All own-action outcomes marginalized under an external partner belief."""

    partner_probabilities: Float[Array, " n_actions"]
    partner_probabilities_valid: Bool[Array, ""]
    probability_violation: Float[Array, ""]
    expected_rewards: Float[Array, " n_actions"]
    expected_outcomes: Float[Array, "n_actions outcome_dim"]
    greedy_action: Int[Array, ""]
    cell_evaluations: Int[Array, ""]


@chex.dataclass(frozen=True)
class JointOutcomeUpdateResult:
    """Prequential prediction, validity diagnostics, and updated state."""

    state: BoundedJointOutcomeState
    prediction: JointOutcomePrediction
    reward_error: Float[Array, ""]
    outcome_error: Float[Array, " outcome_dim"]
    target_valid: Bool[Array, ""]
    visit_count_after: Int[Array, ""]
    pre_step_words: UInt[Array, " 2"]
    post_step_words: UInt[Array, " 2"]
    lifetime_counter_valid: Bool[Array, ""]
    lifetime_capacity_available: Bool[Array, ""]
    update_applied: Bool[Array, ""]


class BoundedJointOutcomeModel:
    """Online table for reward and bounded outcome of each joint action."""

    def __init__(self, config: BoundedJointOutcomeConfig):
        self._config = config

    @property
    def config(self) -> BoundedJointOutcomeConfig:
        """Return the immutable configuration."""
        return self._config

    @property
    def resource_budget(self) -> JointOutcomeResourceBudget:
        """Return exact static state and logical update accounting."""
        cells = self._config.n_actions**2
        float_scalars = cells * (1 + self._config.outcome_dim)
        int_scalars = cells + 1
        uint_scalars = 2
        return JointOutcomeResourceBudget(
            joint_cells=cells,
            allocated_float32_scalars=float_scalars,
            allocated_int32_scalars=int_scalars,
            allocated_uint32_scalars=uint_scalars,
            state_nbytes=4 * (float_scalars + int_scalars + uint_scalars),
            learned_float32_scalars_touched_per_update=1 + self._config.outcome_dim,
            administrative_int32_scalars_touched_per_update=2,
            administrative_uint32_scalars_touched_per_update=2,
            planner_cell_evaluations_per_decision=cells,
            replay_capacity=0,
        )

    def to_config(self) -> dict[str, Any]:
        """Serialize the model without learned state."""
        return {
            "type": type(self).__name__,
            "state_schema": JOINT_OUTCOME_STATE_SCHEMA,
            "config": self._config.to_config(),
        }

    @classmethod
    def from_config(
        cls,
        payload: Mapping[str, Any],
    ) -> BoundedJointOutcomeModel:
        """Strictly reconstruct :meth:`to_config` output."""
        values = dict(payload)
        if set(values) != {"type", "state_schema", "config"}:
            raise ValueError(
                "model config must contain exactly type, state_schema, and config"
            )
        if values["type"] != cls.__name__:
            raise ValueError(f"unexpected model type: {values['type']!r}")
        if values["state_schema"] != JOINT_OUTCOME_STATE_SCHEMA:
            raise ValueError("joint-outcome state schema is unsupported")
        nested = values["config"]
        if not isinstance(nested, Mapping):
            raise ValueError("model config must contain a config mapping")
        return cls(BoundedJointOutcomeConfig.from_config(nested))

    def init(self) -> BoundedJointOutcomeState:
        """Initialize zero predictions and zero visits."""
        n = self._config.n_actions
        return BoundedJointOutcomeState(
            reward_predictions=jnp.zeros((n, n), dtype=jnp.float32),
            outcome_predictions=jnp.zeros(
                (n, n, self._config.outcome_dim),
                dtype=jnp.float32,
            ),
            visit_counts=jnp.zeros((n, n), dtype=jnp.int32),
            step_count=jnp.asarray(0, dtype=jnp.int32),
            step_words=jnp.zeros((2,), dtype=jnp.uint32),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def predict_joint(
        self,
        state: BoundedJointOutcomeState,
        own_action: Array,
        partner_action: Array,
    ) -> JointOutcomePrediction:
        """Return the current cell before any update."""
        own = jnp.asarray(own_action, dtype=jnp.int32)
        partner = jnp.asarray(partner_action, dtype=jnp.int32)
        return JointOutcomePrediction(
            reward=state.reward_predictions[own, partner],
            outcome=state.outcome_predictions[own, partner],
            visit_count=state.visit_counts[own, partner],
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def marginalize(
        self,
        state: BoundedJointOutcomeState,
        partner_probabilities: Array,
    ) -> PartnerMarginalDecision:
        """Marginalize all joint cells under a pre-action partner belief.

        Invalid probabilities are surfaced explicitly. A clipped, normalized
        distribution is used for the numeric decision so small floating-point
        simplex drift is harmless; callers must nevertheless fail promoted
        evidence when ``partner_probabilities_valid`` is false.
        """
        cfg = self._config
        raw = jnp.asarray(partner_probabilities, dtype=jnp.float32).reshape((cfg.n_actions,))
        finite = jnp.all(jnp.isfinite(raw))
        lower_violation = jnp.max(jnp.maximum(-raw, 0.0))
        upper_violation = jnp.max(jnp.maximum(raw - 1.0, 0.0))
        sum_violation = jnp.abs(jnp.sum(raw) - 1.0)
        violation = jnp.maximum(
            jnp.maximum(lower_violation, upper_violation),
            sum_violation,
        )
        valid = finite & (violation <= cfg.probability_tolerance)
        clipped = jnp.clip(raw, 0.0, 1.0)
        normalizer = jnp.sum(clipped)
        normalized = jnp.where(
            normalizer > 0.0,
            clipped / normalizer,
            jnp.full_like(clipped, 1.0 / cfg.n_actions),
        )
        expected_rewards = state.reward_predictions @ normalized
        expected_outcomes = jnp.einsum(
            "abk,b->ak",
            state.outcome_predictions,
            normalized,
        )
        return PartnerMarginalDecision(
            partner_probabilities=normalized,
            partner_probabilities_valid=valid,
            probability_violation=violation,
            expected_rewards=expected_rewards,
            expected_outcomes=expected_outcomes,
            greedy_action=jnp.argmax(expected_rewards).astype(jnp.int32),
            cell_evaluations=jnp.asarray(cfg.n_actions**2, dtype=jnp.int32),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def update(
        self,
        state: BoundedJointOutcomeState,
        own_action: Array,
        partner_action: Array,
        reward: Array,
        outcome: Array,
    ) -> JointOutcomeUpdateResult:
        """Score then update exactly the executed joint-action cell."""
        cfg = self._config
        if getattr(state.visit_counts, "shape", None) != (
            cfg.n_actions,
            cfg.n_actions,
        ):
            raise ValueError("joint-outcome visit_counts shape does not match config")
        if getattr(state.visit_counts, "dtype", None) != jnp.dtype(jnp.int32):
            raise TypeError("joint-outcome visit_counts must have dtype int32")
        proposed_step_words, lifetime_capacity_available = (
            _checked_lifetime_words_increment(state.step_words)
        )
        lifetime_counter_valid = _lifetime_counter_valid(
            state.step_words,
            state.step_count,
        ) & jnp.all(state.visit_counts >= 0)
        update_available = lifetime_counter_valid & lifetime_capacity_available
        own = jnp.asarray(own_action, dtype=jnp.int32)
        partner = jnp.asarray(partner_action, dtype=jnp.int32)
        reward_target = jnp.asarray(reward, dtype=jnp.float32)
        outcome_target = jnp.asarray(outcome, dtype=jnp.float32).reshape((cfg.outcome_dim,))
        target_valid = (
            jnp.isfinite(reward_target)
            & (jnp.abs(reward_target) <= cfg.reward_bound)
            & jnp.all(jnp.isfinite(outcome_target))
            & jnp.all(jnp.abs(outcome_target) <= cfg.outcome_bound)
        )
        bounded_reward = jnp.nan_to_num(
            jnp.clip(reward_target, -cfg.reward_bound, cfg.reward_bound),
            nan=0.0,
            posinf=cfg.reward_bound,
            neginf=-cfg.reward_bound,
        )
        bounded_outcome = jnp.nan_to_num(
            jnp.clip(outcome_target, -cfg.outcome_bound, cfg.outcome_bound),
            nan=0.0,
            posinf=cfg.outcome_bound,
            neginf=-cfg.outcome_bound,
        )
        prediction = self.predict_joint(state, own, partner)
        reward_error = bounded_reward - prediction.reward
        outcome_error = bounded_outcome - prediction.outcome
        alpha = jnp.asarray(cfg.step_size, dtype=jnp.float32)

        reward_predictions = state.reward_predictions.at[own, partner].add(alpha * reward_error)
        outcome_predictions = state.outcome_predictions.at[own, partner].add(alpha * outcome_error)
        visit_counts = state.visit_counts.at[own, partner].set(
            _saturating_int32_increment(state.visit_counts[own, partner])
        )
        proposed_state = BoundedJointOutcomeState(
            reward_predictions=reward_predictions,
            outcome_predictions=outcome_predictions,
            visit_counts=visit_counts,
            step_count=_saturating_int32_increment(state.step_count),
            step_words=proposed_step_words,
        )
        next_state = jax.lax.cond(
            update_available,
            lambda _: proposed_state,
            lambda _: state,
            operand=None,
        )
        return JointOutcomeUpdateResult(
            state=next_state,
            prediction=prediction,
            reward_error=reward_error,
            outcome_error=outcome_error,
            target_valid=target_valid,
            visit_count_after=next_state.visit_counts[own, partner],
            pre_step_words=state.step_words,
            post_step_words=next_state.step_words,
            lifetime_counter_valid=lifetime_counter_valid,
            lifetime_capacity_available=lifetime_capacity_available,
            update_applied=update_available,
        )


def joint_outcome_lifetime_counter_nbytes() -> int:
    """Return bytes occupied by compatibility telemetry plus exact identity."""

    return JOINT_OUTCOME_LIFETIME_COUNTER_NBYTES


def measure_bounded_joint_outcome_state_nbytes(
    state: BoundedJointOutcomeState,
) -> int:
    """Measure persistent JAX-array bytes in one bounded table state."""

    return sum(
        int(leaf.size) * int(leaf.dtype.itemsize)
        for leaf in jax.tree.leaves(state)
        if isinstance(leaf, Array)
    )


def migrate_legacy_bounded_joint_outcome_state(
    legacy_state: Any,
) -> BoundedJointOutcomeState:
    """Migrate an exact pre-v2 table whose int32 clock is unambiguous."""

    if isinstance(legacy_state, Mapping):
        fields = dict(legacy_state)
    elif dataclasses.is_dataclass(legacy_state) and not isinstance(legacy_state, type):
        fields = {
            field.name: getattr(legacy_state, field.name)
            for field in dataclasses.fields(legacy_state)
        }
    else:
        raise TypeError("legacy joint-outcome state must be a mapping or dataclass")
    current_names = {
        field.name
        for field in dataclasses.fields(BoundedJointOutcomeState)  # type: ignore[arg-type]
    }
    legacy_names = current_names - {"step_words"}
    supplied_names = set(fields)
    if supplied_names != legacy_names:
        missing = sorted(legacy_names - supplied_names)
        extra = sorted(supplied_names - legacy_names)
        raise ValueError(
            "legacy joint-outcome field manifest is not exact; "
            f"missing={missing}, extra={extra}"
        )
    step_count = jnp.asarray(fields["step_count"])
    if step_count.shape != () or step_count.dtype != jnp.dtype(jnp.int32):
        raise TypeError("legacy joint-outcome step_count must be scalar int32")
    step = int(step_count)
    if step < 0:
        raise ValueError("negative legacy joint-outcome step_count indicates wrap")
    if step >= _INT32_MAX:
        raise ValueError("saturated legacy joint-outcome step_count is ambiguous")
    visit_counts = jnp.asarray(fields["visit_counts"])
    if visit_counts.dtype != jnp.dtype(jnp.int32):
        raise TypeError("legacy joint-outcome visit_counts must have dtype int32")
    if bool(jnp.any(visit_counts < 0)):
        raise ValueError("negative legacy joint-outcome visit count indicates wrap")
    visit_total = sum(int(value) for value in jax.device_get(visit_counts).reshape(-1))
    if visit_total != step:
        raise ValueError("legacy joint-outcome global and per-cell counters are not aligned")
    fields["step_words"] = jnp.asarray((0, step), dtype=jnp.uint32)
    return BoundedJointOutcomeState(**fields)
