# mypy: disable-error-code="attr-defined,call-arg,no-untyped-call,type-var"
"""Bounded nonlinear discrete differential actor-critic (isolated L0 core).

This module closes one mechanism gap in WP6 without replacing any promoted
surface.  The actor and critic own separate one-hidden-layer ``tanh`` networks,
eligibility traces, momentum buffers, plasticity policies, and bounded online
utility EMAs.  The continuing TD error is

``delta_t = reward_t - average_reward_t + V(s_(t+1)) - V(s_t)``.

Two explicitly different actor objectives are available:

``ordinary_behavior``
    The caller-supplied behavior policy must equal the configured
    epsilon-uniform mixture of the current target policy.  The actor trace uses
    the exact chain-rule behavior score
    ``((1-epsilon) * pi(a|s) / b(a|s)) * grad(log pi(a|s))``.  The critic and
    reward-rate learner are ordinary on-behavior differential learners.

``clipped_target_importance``
    The caller may supply any finite, normalized, full-support categorical
    behavior policy.  ``min(pi(a|s) / b(a|s), importance_clip)`` multiplies the
    actor trace, critic trace, and reward-rate update.  This is a clipped
    per-decision *action* correction.  It does not correct state-visitation or
    initial-state distribution mismatch and is not claimed to be unbiased.

Every action is bound to the full target and caller-owned behavior policy,
their log probabilities, a fixed 32-byte owner digest, policy revisions, and
an exact uint64 identity.  A pure proposal phase exposes the learned candidate
and next target without drawing.  Commit recomputes that proposal from the
live state and record, validates the next owner and exact successor revision,
then consumes exactly one categorical draw for the successor action.  Any
invalid source, receipt mismatch, exhausted clock, non-finite candidate, or
configured numeric-bound violation returns the original state byte-for-byte.
Float32 target-policy underflow to zero is rejected because action-importance
semantics require full target and behavior support.

Paper-defined delight is not used for likelihood correction in this module.
Passing tests establish only L0 mechanism contracts.  Outcome status is
``not_assessed``: there is no convergence, control-quality, retention, safety,
resource-peak, benchmark, or state-distribution-correction claim.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from numbers import Real
from pathlib import Path
from typing import Any, Literal, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array
from jaxtyping import Bool, Float, Int, UInt

from alberta_framework.core.checkpoints import load_checkpoint as _load_checkpoint
from alberta_framework.core.checkpoints import (
    load_checkpoint_metadata as _load_checkpoint_metadata,
)
from alberta_framework.core.checkpoints import save_checkpoint as _save_checkpoint

NONLINEAR_AVERAGE_REWARD_ACTOR_CRITIC_CONFIG_SCHEMA = (
    "alberta.nonlinear-average-reward-actor-critic-config.v1"
)
NONLINEAR_AVERAGE_REWARD_ACTOR_CRITIC_STATE_SCHEMA = (
    "alberta.nonlinear-average-reward-actor-critic-state.v1"
)
NONLINEAR_AVERAGE_REWARD_ACTOR_CRITIC_CHECKPOINT_SCHEMA = (
    "alberta.nonlinear-average-reward-actor-critic-checkpoint.v1"
)
NONLINEAR_AVERAGE_REWARD_ACTOR_CRITIC_RESOURCE_SCHEMA = (
    "alberta.nonlinear-average-reward-actor-critic-resource.v1"
)
NONLINEAR_AVERAGE_REWARD_ACTOR_CRITIC_EVIDENCE_LEVEL = "L0"
NONLINEAR_AVERAGE_REWARD_ACTOR_CRITIC_OUTCOME_STATUS = "not_assessed"
NONLINEAR_AVERAGE_REWARD_ACTOR_CRITIC_LIFETIME_SEMANTICS = "exact-uint64-fail-stop"
NONLINEAR_AVERAGE_REWARD_ACTOR_CRITIC_MAX_DECISIONS = 2**64 - 1
NONLINEAR_AVERAGE_REWARD_ACTOR_CRITIC_MAX_UPDATES = 2**64 - 1
NONLINEAR_AVERAGE_REWARD_ACTOR_CRITIC_OWNER_DIGEST_WORDS = 8

NonlinearAverageRewardObjectiveMode = Literal[
    "ordinary_behavior",
    "clipped_target_importance",
]
NonlinearAverageRewardPlasticityPolicy = Literal["plastic", "frozen"]

_UINT32_MAX = 2**32 - 1
_INT32_MAX = 2**31 - 1


def _exact_manifest(
    payload: Mapping[str, Any],
    expected: set[str],
    *,
    label: str,
) -> dict[str, Any]:
    if type(payload) is not dict:
        raise TypeError(f"{label} must be an exact dict")
    fields = dict(payload)
    supplied = set(fields)
    if supplied != expected:
        missing = sorted(expected - supplied)
        extra = sorted(supplied - expected)
        raise ValueError(f"{label} field manifest is not exact; missing={missing}, extra={extra}")
    return fields


def _exact_int(
    value: Any,
    *,
    label: str,
    minimum: int,
    maximum: int = _INT32_MAX,
) -> int:
    if type(value) is not int:
        raise TypeError(f"{label} must be an exact integer")
    if value < minimum:
        raise ValueError(f"{label} must be >= {minimum}")
    if value > maximum:
        raise ValueError(f"{label} must be <= {maximum}")
    return value


def _finite_real(
    value: Any,
    *,
    label: str,
    minimum: float,
    maximum: float | None = None,
    strict_minimum: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{label} must be a real number")
    scalar = float(value)
    if not math.isfinite(scalar):
        raise ValueError(f"{label} must be finite")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        narrowed = float(np.float32(scalar))
    if not math.isfinite(narrowed):
        raise ValueError(f"{label} must remain finite in float32")
    if scalar != 0.0 and narrowed == 0.0:
        raise ValueError(f"{label} must not underflow to zero in float32")
    below = scalar <= minimum if strict_minimum else scalar < minimum
    if below or (maximum is not None and scalar > maximum):
        comparator = ">" if strict_minimum else ">="
        upper = "" if maximum is None else f" and <= {maximum}"
        raise ValueError(f"{label} must be {comparator} {minimum}{upper}")
    return scalar


def _plasticity(value: Any, *, label: str) -> NonlinearAverageRewardPlasticityPolicy:
    if type(value) is not str:
        raise TypeError(f"{label} must be an exact string")
    if value not in ("plastic", "frozen"):
        raise ValueError(f"{label} must be 'plastic' or 'frozen'")
    return cast(NonlinearAverageRewardPlasticityPolicy, value)


def _objective(value: Any) -> NonlinearAverageRewardObjectiveMode:
    if type(value) is not str:
        raise TypeError("objective_mode must be an exact string")
    if value not in ("ordinary_behavior", "clipped_target_importance"):
        raise ValueError(
            "objective_mode must be 'ordinary_behavior' or 'clipped_target_importance'"
        )
    return cast(NonlinearAverageRewardObjectiveMode, value)


def _require_array(
    value: Any,
    *,
    label: str,
    shape: tuple[int, ...],
    dtype: jnp.dtype,
) -> Array:
    if getattr(value, "shape", None) != shape:
        raise ValueError(f"{label} must have shape {shape}")
    if getattr(value, "dtype", None) != dtype:
        raise TypeError(f"{label} must have dtype {dtype}")
    return jnp.asarray(value)


def _require_float32_scalar(value: Any, *, label: str) -> Array:
    return _require_array(value, label=label, shape=(), dtype=jnp.dtype(jnp.float32))


def _require_float32_vector(value: Any, size: int, *, label: str) -> Array:
    return _require_array(
        value,
        label=label,
        shape=(size,),
        dtype=jnp.dtype(jnp.float32),
    )


def _require_int32_scalar(value: Any, *, label: str) -> Array:
    return _require_array(value, label=label, shape=(), dtype=jnp.dtype(jnp.int32))


def _require_bool_scalar(value: Any, *, label: str) -> Array:
    return _require_array(value, label=label, shape=(), dtype=jnp.dtype(jnp.bool_))


def _require_words(value: Any, *, label: str) -> Array:
    return _require_array(value, label=label, shape=(2,), dtype=jnp.dtype(jnp.uint32))


def _require_threefry_key(value: Any, *, label: str) -> None:
    try:
        data = jr.key_data(value)
        implementation = str(jr.key_impl(value))
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{label} must be one typed Threefry JAX key") from exc
    if (
        getattr(value, "shape", None) != ()
        or data.shape != (2,)
        or data.dtype != jnp.dtype(jnp.uint32)
        or implementation != "threefry2x32"
    ):
        raise TypeError(f"{label} must be one typed Threefry JAX key")


def _increment_words(words: Array) -> tuple[UInt[Array, " 2"], Bool[Array, ""]]:
    maximum = jnp.asarray(_UINT32_MAX, dtype=jnp.uint32)
    carry = words[1] == maximum
    capacity = ~(carry & (words[0] == maximum))
    low = words[1] + jnp.asarray(1, dtype=jnp.uint32)
    high = words[0] + carry.astype(jnp.uint32)
    return jnp.stack((high, low)).astype(jnp.uint32), capacity


def _owner_digest(value: Any, *, label: str) -> tuple[int, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{label} must be an exact tuple")
    if len(value) != NONLINEAR_AVERAGE_REWARD_ACTOR_CRITIC_OWNER_DIGEST_WORDS:
        raise ValueError(
            f"{label} must contain exactly "
            f"{NONLINEAR_AVERAGE_REWARD_ACTOR_CRITIC_OWNER_DIGEST_WORDS} uint32 words"
        )
    parsed: list[int] = []
    for index, word in enumerate(value):
        if type(word) is not int:
            raise TypeError(f"{label}[{index}] must be an exact integer")
        if word < 0 or word > _UINT32_MAX:
            raise ValueError(f"{label}[{index}] must be a uint32 word")
        parsed.append(word)
    return tuple(parsed)


def _bits_equal(left: Array, right: Array) -> Bool[Array, ""]:
    left_bits = jax.lax.bitcast_convert_type(left, jnp.uint32)
    right_bits = jax.lax.bitcast_convert_type(right, jnp.uint32)
    return jnp.all(left_bits == right_bits)


def _trees_exact(left: Any, right: Any) -> Bool[Array, ""]:
    """Return bit equality without allowing shape/dtype broadcasting."""

    left_leaves, left_structure = jax.tree.flatten(left)
    right_leaves, right_structure = jax.tree.flatten(right)
    if cast(Any, left_structure) != cast(Any, right_structure) or len(left_leaves) != len(
        right_leaves
    ):
        return jnp.asarray(False, dtype=jnp.bool_)
    verdict = jnp.asarray(True, dtype=jnp.bool_)
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        if (
            getattr(left_leaf, "shape", None) != getattr(right_leaf, "shape", None)
            or getattr(left_leaf, "dtype", None) != getattr(right_leaf, "dtype", None)
        ):
            return jnp.asarray(False, dtype=jnp.bool_)
        if str(getattr(left_leaf, "dtype", "")).startswith("key<"):
            verdict = verdict & jnp.all(
                jr.key_data(left_leaf) == jr.key_data(right_leaf)
            )
        elif getattr(left_leaf, "dtype", None) == jnp.dtype(jnp.float32):
            verdict = verdict & _bits_equal(left_leaf, right_leaf)
        else:
            verdict = verdict & jnp.all(left_leaf == right_leaf)
    return verdict


def _array_nbytes(value: Array) -> int:
    return int(value.size) * int(value.dtype.itemsize)


def _tree_select(condition: Array, yes: Any, no: Any) -> Any:
    return jax.tree.map(lambda x, y: jnp.where(condition, x, y), yes, no)


@dataclasses.dataclass(frozen=True)
class NonlinearAverageRewardActorCriticConfig:
    """Static architecture, objective, optimizer, bounds, and ownership contract."""

    n_actions: int
    behavior_owner_digest: tuple[int, ...]
    hidden_size: int = 64
    objective_mode: NonlinearAverageRewardObjectiveMode = "ordinary_behavior"
    ordinary_behavior_epsilon: float = 0.1
    actor_head_step_size: float = 0.001
    actor_trunk_step_size: float = 0.001
    critic_head_step_size: float = 0.01
    critic_trunk_step_size: float = 0.01
    average_reward_step_size: float = 0.01
    actor_trace_decay: float = 0.9
    critic_trace_decay: float = 0.9
    momentum: float = 0.0
    importance_clip: float = 1.0
    policy_temperature: float = 1.0
    initialization_scale: float = 0.1
    utility_decay: float = 0.99
    max_component_utility: float = 1_000.0
    max_trace_magnitude: float = 1_000.0
    max_update_component_magnitude: float = 1.0
    max_parameter_magnitude: float = 1_000.0
    actor_head_plasticity: NonlinearAverageRewardPlasticityPolicy = "plastic"
    actor_trunk_plasticity: NonlinearAverageRewardPlasticityPolicy = "plastic"
    critic_head_plasticity: NonlinearAverageRewardPlasticityPolicy = "plastic"
    critic_trunk_plasticity: NonlinearAverageRewardPlasticityPolicy = "plastic"

    def __post_init__(self) -> None:
        _exact_int(self.n_actions, label="n_actions", minimum=2)
        _exact_int(self.hidden_size, label="hidden_size", minimum=1)
        _owner_digest(self.behavior_owner_digest, label="behavior_owner_digest")
        _objective(self.objective_mode)
        for name in (
            "ordinary_behavior_epsilon",
            "actor_trace_decay",
            "critic_trace_decay",
            "utility_decay",
        ):
            _finite_real(getattr(self, name), label=name, minimum=0.0, maximum=1.0)
        for name in (
            "actor_head_step_size",
            "actor_trunk_step_size",
            "critic_head_step_size",
            "critic_trunk_step_size",
            "average_reward_step_size",
        ):
            _finite_real(getattr(self, name), label=name, minimum=0.0)
        _finite_real(self.momentum, label="momentum", minimum=0.0, maximum=1.0)
        if self.momentum == 1.0:
            raise ValueError("momentum must be < 1.0")
        for name in (
            "importance_clip",
            "policy_temperature",
            "initialization_scale",
            "max_component_utility",
            "max_trace_magnitude",
            "max_update_component_magnitude",
            "max_parameter_magnitude",
        ):
            _finite_real(
                getattr(self, name),
                label=name,
                minimum=0.0,
                strict_minimum=True,
            )
        for name in (
            "actor_head_plasticity",
            "actor_trunk_plasticity",
            "critic_head_plasticity",
            "critic_trunk_plasticity",
        ):
            _plasticity(getattr(self, name), label=name)

    def to_config(self) -> dict[str, Any]:
        payload = dataclasses.asdict(self)
        payload["behavior_owner_digest"] = list(self.behavior_owner_digest)
        return {
            "type": "NonlinearAverageRewardActorCritic",
            "schema": NONLINEAR_AVERAGE_REWARD_ACTOR_CRITIC_CONFIG_SCHEMA,
            "evidence_level": NONLINEAR_AVERAGE_REWARD_ACTOR_CRITIC_EVIDENCE_LEVEL,
            "outcome_status": NONLINEAR_AVERAGE_REWARD_ACTOR_CRITIC_OUTCOME_STATUS,
            **payload,
        }

    @classmethod
    def from_config(
        cls,
        payload: dict[str, Any],
    ) -> NonlinearAverageRewardActorCriticConfig:
        expected = {field.name for field in dataclasses.fields(cls)} | {
            "type",
            "schema",
            "evidence_level",
            "outcome_status",
        }
        fields = _exact_manifest(payload, expected, label="nonlinear average-reward config")
        if fields.pop("type") != "NonlinearAverageRewardActorCritic":
            raise ValueError("nonlinear average-reward config type is unsupported")
        if fields.pop("schema") != NONLINEAR_AVERAGE_REWARD_ACTOR_CRITIC_CONFIG_SCHEMA:
            raise ValueError("nonlinear average-reward config schema is unsupported")
        if fields.pop("evidence_level") != NONLINEAR_AVERAGE_REWARD_ACTOR_CRITIC_EVIDENCE_LEVEL:
            raise ValueError("evidence level must remain L0")
        if fields.pop("outcome_status") != NONLINEAR_AVERAGE_REWARD_ACTOR_CRITIC_OUTCOME_STATUS:
            raise ValueError("outcome status must remain not_assessed")
        serialized_digest = fields["behavior_owner_digest"]
        if type(serialized_digest) is not list:
            raise TypeError("serialized behavior_owner_digest must be an exact list")
        fields["behavior_owner_digest"] = tuple(serialized_digest)
        return cls(**fields)


@chex.dataclass(frozen=True)
class DiscreteBehaviorPolicyReceipt:
    """Caller-owned full-support categorical policy and revision receipt."""

    probabilities: Float[Array, " action"]
    log_probabilities: Float[Array, " action"]
    behavior_owner_digest: UInt[Array, " 8"]
    revision_words: UInt[Array, " 2"]


@chex.dataclass(frozen=True)
class NonlinearAverageRewardActionRecord:
    """Exact cached identity and decision-time policy semantics."""

    observation: Float[Array, " feature"]
    action: Int[Array, ""]
    target_policy: Float[Array, " action"]
    target_log_policy: Float[Array, " action"]
    behavior_policy: Float[Array, " action"]
    behavior_log_policy: Float[Array, " action"]
    behavior_owner_digest: UInt[Array, " 8"]
    target_log_probability: Float[Array, ""]
    behavior_log_probability: Float[Array, ""]
    target_revision_words: UInt[Array, " 2"]
    behavior_revision_words: UInt[Array, " 2"]
    action_identity_words: UInt[Array, " 2"]


@chex.dataclass(frozen=True)
class NonlinearAverageRewardActorCriticState:
    """Immutable learner, pending decision, RNG, utility, and exact-clock state."""

    actor_trunk_w: Float[Array, "hidden feature"]
    actor_trunk_b: Float[Array, " hidden"]
    actor_head_w: Float[Array, "action hidden"]
    actor_head_b: Float[Array, " action"]
    critic_trunk_w: Float[Array, "hidden feature"]
    critic_trunk_b: Float[Array, " hidden"]
    critic_head_w: Float[Array, " hidden"]
    critic_head_b: Float[Array, ""]

    actor_trunk_trace_w: Float[Array, "hidden feature"]
    actor_trunk_trace_b: Float[Array, " hidden"]
    actor_head_trace_w: Float[Array, "action hidden"]
    actor_head_trace_b: Float[Array, " action"]
    critic_trunk_trace_w: Float[Array, "hidden feature"]
    critic_trunk_trace_b: Float[Array, " hidden"]
    critic_head_trace_w: Float[Array, " hidden"]
    critic_head_trace_b: Float[Array, ""]

    actor_trunk_velocity_w: Float[Array, "hidden feature"]
    actor_trunk_velocity_b: Float[Array, " hidden"]
    actor_head_velocity_w: Float[Array, "action hidden"]
    actor_head_velocity_b: Float[Array, " action"]
    critic_trunk_velocity_w: Float[Array, "hidden feature"]
    critic_trunk_velocity_b: Float[Array, " hidden"]
    critic_head_velocity_w: Float[Array, " hidden"]
    critic_head_velocity_b: Float[Array, ""]

    average_reward: Float[Array, ""]
    actor_head_utility: Float[Array, ""]
    actor_trunk_utility: Float[Array, ""]
    critic_head_utility: Float[Array, ""]
    critic_trunk_utility: Float[Array, ""]

    pending_observation: Float[Array, " feature"]
    pending_action: Int[Array, ""]
    pending_target_policy: Float[Array, " action"]
    pending_target_log_policy: Float[Array, " action"]
    pending_behavior_policy: Float[Array, " action"]
    pending_behavior_log_policy: Float[Array, " action"]
    pending_behavior_owner_digest: UInt[Array, " 8"]
    pending_target_log_probability: Float[Array, ""]
    pending_behavior_log_probability: Float[Array, ""]
    pending_target_revision_words: UInt[Array, " 2"]
    pending_behavior_revision_words: UInt[Array, " 2"]
    pending_action_identity_words: UInt[Array, " 2"]
    pending_valid: Bool[Array, ""]

    rng_key: Array
    decision_words: UInt[Array, " 2"]
    update_words: UInt[Array, " 2"]
    target_revision_words: UInt[Array, " 2"]


@chex.dataclass(frozen=True)
class NonlinearAverageRewardStartResult:
    """Atomic first-decision result."""

    state: NonlinearAverageRewardActorCriticState
    record: NonlinearAverageRewardActionRecord
    action: Int[Array, ""]
    target_policy: Float[Array, " action"]
    behavior_policy: Float[Array, " action"]
    state_valid: Bool[Array, ""]
    behavior_receipt_valid: Bool[Array, ""]
    behavior_owner_valid: Bool[Array, ""]
    target_support_valid: Bool[Array, ""]
    objective_policy_valid: Bool[Array, ""]
    lifetime_capacity_available: Bool[Array, ""]
    start_applied: Bool[Array, ""]


@chex.dataclass(frozen=True)
class NonlinearAverageRewardUpdateProposal:
    """Pure source-bound learning proposal; it owns no successor RNG draw."""

    candidate_state: NonlinearAverageRewardActorCriticState
    source_record: NonlinearAverageRewardActionRecord
    reward: Float[Array, ""]
    next_observation: Float[Array, " feature"]
    next_target_policy: Float[Array, " action"]
    next_target_log_policy: Float[Array, " action"]
    expected_behavior_revision_words: UInt[Array, " 2"]
    value: Float[Array, ""]
    next_value: Float[Array, ""]
    td_error: Float[Array, ""]
    pre_average_reward: Float[Array, ""]
    proposed_average_reward: Float[Array, ""]
    raw_importance_ratio: Float[Array, ""]
    clipped_importance_ratio: Float[Array, ""]
    actor_score_multiplier: Float[Array, ""]
    critic_trace_multiplier: Float[Array, ""]
    reward_rate_multiplier: Float[Array, ""]
    ratio_truncation: Float[Array, ""]
    pre_update_words: UInt[Array, " 2"]
    post_update_words: UInt[Array, " 2"]
    pre_decision_words: UInt[Array, " 2"]
    post_decision_words: UInt[Array, " 2"]
    pre_target_revision_words: UInt[Array, " 2"]
    post_target_revision_words: UInt[Array, " 2"]
    state_valid: Bool[Array, ""]
    record_identity_valid: Bool[Array, ""]
    target_policy_valid: Bool[Array, ""]
    transition_valid: Bool[Array, ""]
    numerical_source_valid: Bool[Array, ""]
    ratio_valid: Bool[Array, ""]
    behavior_revision_capacity_available: Bool[Array, ""]
    lifetime_capacity_available: Bool[Array, ""]
    candidate_numeric_valid: Bool[Array, ""]
    next_target_support_valid: Bool[Array, ""]
    proposal_valid: Bool[Array, ""]


@chex.dataclass(frozen=True)
class NonlinearAverageRewardUpdateResult:
    """One atomic differential update followed by one successor draw."""

    state: NonlinearAverageRewardActorCriticState
    successor_record: NonlinearAverageRewardActionRecord
    successor_action: Int[Array, ""]
    value: Float[Array, ""]
    next_value: Float[Array, ""]
    td_error: Float[Array, ""]
    pre_average_reward: Float[Array, ""]
    post_average_reward: Float[Array, ""]
    raw_importance_ratio: Float[Array, ""]
    clipped_importance_ratio: Float[Array, ""]
    actor_score_multiplier: Float[Array, ""]
    critic_trace_multiplier: Float[Array, ""]
    reward_rate_multiplier: Float[Array, ""]
    ratio_truncation: Float[Array, ""]
    pre_update_words: UInt[Array, " 2"]
    post_update_words: UInt[Array, " 2"]
    pre_decision_words: UInt[Array, " 2"]
    post_decision_words: UInt[Array, " 2"]
    pre_target_revision_words: UInt[Array, " 2"]
    post_target_revision_words: UInt[Array, " 2"]
    state_valid: Bool[Array, ""]
    proposal_valid: Bool[Array, ""]
    proposal_identity_valid: Bool[Array, ""]
    record_identity_valid: Bool[Array, ""]
    target_policy_valid: Bool[Array, ""]
    transition_valid: Bool[Array, ""]
    ratio_valid: Bool[Array, ""]
    next_behavior_receipt_valid: Bool[Array, ""]
    next_behavior_owner_valid: Bool[Array, ""]
    next_behavior_revision_valid: Bool[Array, ""]
    behavior_revision_capacity_available: Bool[Array, ""]
    next_objective_policy_valid: Bool[Array, ""]
    lifetime_capacity_available: Bool[Array, ""]
    candidate_state_valid: Bool[Array, ""]
    successor_sampled: Bool[Array, ""]
    update_applied: Bool[Array, ""]


@dataclasses.dataclass(frozen=True)
class NonlinearAverageRewardResourceBudget:
    """Exact persistent-state array byte partition."""

    schema: str
    persistent_bytes_scope: str
    parameter_nbytes: int
    trace_nbytes: int
    optimizer_nbytes: int
    utility_nbytes: int
    pending_cache_nbytes: int
    clock_nbytes: int
    rng_nbytes: int
    total_state_nbytes: int


@chex.dataclass(frozen=True)
class NonlinearAverageRewardScanResult:
    """Fixed-shape start plus transition scan result."""

    state: NonlinearAverageRewardActorCriticState
    actions: Int[Array, " decisions"]
    td_errors: Float[Array, " steps"]
    average_rewards: Float[Array, " steps"]
    raw_importance_ratios: Float[Array, " steps"]
    clipped_importance_ratios: Float[Array, " steps"]
    update_applied: Bool[Array, " steps"]
    start_applied: Bool[Array, ""]


def _actor_forward(
    state: NonlinearAverageRewardActorCriticState,
    observation: Array,
    temperature: float,
) -> tuple[Array, Array, Array]:
    hidden = jnp.tanh(state.actor_trunk_w @ observation + state.actor_trunk_b)
    logits = state.actor_head_w @ hidden + state.actor_head_b
    log_policy = jax.nn.log_softmax(
        logits / jnp.asarray(temperature, dtype=jnp.float32)
    )
    return hidden, jnp.exp(log_policy), log_policy


def _critic_forward(
    state: NonlinearAverageRewardActorCriticState,
    observation: Array,
) -> tuple[Array, Array]:
    hidden = jnp.tanh(state.critic_trunk_w @ observation + state.critic_trunk_b)
    return hidden, jnp.dot(state.critic_head_w, hidden) + state.critic_head_b


def _empty_record(
    state: NonlinearAverageRewardActorCriticState,
) -> NonlinearAverageRewardActionRecord:
    return NonlinearAverageRewardActionRecord(
        observation=jnp.zeros_like(state.pending_observation),
        action=jnp.asarray(-1, dtype=jnp.int32),
        target_policy=jnp.zeros_like(state.pending_target_policy),
        target_log_policy=jnp.zeros_like(state.pending_target_log_policy),
        behavior_policy=jnp.zeros_like(state.pending_behavior_policy),
        behavior_log_policy=jnp.zeros_like(state.pending_behavior_log_policy),
        behavior_owner_digest=jnp.zeros_like(state.pending_behavior_owner_digest),
        target_log_probability=jnp.asarray(0.0, dtype=jnp.float32),
        behavior_log_probability=jnp.asarray(0.0, dtype=jnp.float32),
        target_revision_words=jnp.zeros((2,), dtype=jnp.uint32),
        behavior_revision_words=jnp.zeros((2,), dtype=jnp.uint32),
        action_identity_words=jnp.zeros((2,), dtype=jnp.uint32),
    )


def _record_from_state(
    state: NonlinearAverageRewardActorCriticState,
) -> NonlinearAverageRewardActionRecord:
    return NonlinearAverageRewardActionRecord(
        observation=state.pending_observation,
        action=state.pending_action,
        target_policy=state.pending_target_policy,
        target_log_policy=state.pending_target_log_policy,
        behavior_policy=state.pending_behavior_policy,
        behavior_log_policy=state.pending_behavior_log_policy,
        behavior_owner_digest=state.pending_behavior_owner_digest,
        target_log_probability=state.pending_target_log_probability,
        behavior_log_probability=state.pending_behavior_log_probability,
        target_revision_words=state.pending_target_revision_words,
        behavior_revision_words=state.pending_behavior_revision_words,
        action_identity_words=state.pending_action_identity_words,
    )


def _state_float_arrays(state: NonlinearAverageRewardActorCriticState) -> tuple[Array, ...]:
    leaves = jax.tree.leaves(state)
    return tuple(leaf for leaf in leaves if getattr(leaf, "dtype", None) == jnp.float32)


def measure_nonlinear_average_reward_actor_critic_state_nbytes(
    state: NonlinearAverageRewardActorCriticState,
) -> int:
    """Return exact bytes occupied by every persistent array leaf."""

    return sum(_array_nbytes(leaf) for leaf in jax.tree.leaves(state))


class NonlinearAverageRewardActorCritic:
    """Separate nonlinear actor/critic with exact continuing transactions."""

    def __init__(self, config: NonlinearAverageRewardActorCriticConfig) -> None:
        if type(config) is not NonlinearAverageRewardActorCriticConfig:
            raise TypeError("config must be an exact NonlinearAverageRewardActorCriticConfig")
        self._config = config

    @property
    def config(self) -> NonlinearAverageRewardActorCriticConfig:
        return self._config

    def to_config(self) -> dict[str, Any]:
        return self._config.to_config()

    @classmethod
    def from_config(cls, payload: dict[str, Any]) -> NonlinearAverageRewardActorCritic:
        return cls(NonlinearAverageRewardActorCriticConfig.from_config(payload))

    def init(self, feature_dim: int, key: Array) -> NonlinearAverageRewardActorCriticState:
        """Initialize fixed-shape state from one typed Threefry key."""

        feature_dim = _exact_int(feature_dim, label="feature_dim", minimum=1)
        _require_threefry_key(key, label="key")
        keys = jr.split(key, 5)
        scale = jnp.asarray(self._config.initialization_scale, dtype=jnp.float32)
        hidden = self._config.hidden_size
        actions = self._config.n_actions
        actor_trunk_w = scale * jr.normal(keys[1], (hidden, feature_dim), dtype=jnp.float32)
        actor_head_w = scale * jr.normal(keys[2], (actions, hidden), dtype=jnp.float32)
        critic_trunk_w = scale * jr.normal(keys[3], (hidden, feature_dim), dtype=jnp.float32)
        critic_head_w = scale * jr.normal(keys[4], (hidden,), dtype=jnp.float32)
        zero_hidden_feature = jnp.zeros((hidden, feature_dim), dtype=jnp.float32)
        zero_hidden = jnp.zeros((hidden,), dtype=jnp.float32)
        zero_action_hidden = jnp.zeros((actions, hidden), dtype=jnp.float32)
        zero_action = jnp.zeros((actions,), dtype=jnp.float32)
        zero_feature = jnp.zeros((feature_dim,), dtype=jnp.float32)
        zero_scalar = jnp.asarray(0.0, dtype=jnp.float32)
        zero_words = jnp.zeros((2,), dtype=jnp.uint32)
        zero_owner_digest = jnp.zeros(
            (NONLINEAR_AVERAGE_REWARD_ACTOR_CRITIC_OWNER_DIGEST_WORDS,),
            dtype=jnp.uint32,
        )
        return NonlinearAverageRewardActorCriticState(
            actor_trunk_w=actor_trunk_w,
            actor_trunk_b=zero_hidden,
            actor_head_w=actor_head_w,
            actor_head_b=zero_action,
            critic_trunk_w=critic_trunk_w,
            critic_trunk_b=zero_hidden,
            critic_head_w=critic_head_w,
            critic_head_b=zero_scalar,
            actor_trunk_trace_w=zero_hidden_feature,
            actor_trunk_trace_b=zero_hidden,
            actor_head_trace_w=zero_action_hidden,
            actor_head_trace_b=zero_action,
            critic_trunk_trace_w=zero_hidden_feature,
            critic_trunk_trace_b=zero_hidden,
            critic_head_trace_w=zero_hidden,
            critic_head_trace_b=zero_scalar,
            actor_trunk_velocity_w=zero_hidden_feature,
            actor_trunk_velocity_b=zero_hidden,
            actor_head_velocity_w=zero_action_hidden,
            actor_head_velocity_b=zero_action,
            critic_trunk_velocity_w=zero_hidden_feature,
            critic_trunk_velocity_b=zero_hidden,
            critic_head_velocity_w=zero_hidden,
            critic_head_velocity_b=zero_scalar,
            average_reward=zero_scalar,
            actor_head_utility=zero_scalar,
            actor_trunk_utility=zero_scalar,
            critic_head_utility=zero_scalar,
            critic_trunk_utility=zero_scalar,
            pending_observation=zero_feature,
            pending_action=jnp.asarray(-1, dtype=jnp.int32),
            pending_target_policy=zero_action,
            pending_target_log_policy=zero_action,
            pending_behavior_policy=zero_action,
            pending_behavior_log_policy=zero_action,
            pending_behavior_owner_digest=zero_owner_digest,
            pending_target_log_probability=zero_scalar,
            pending_behavior_log_probability=zero_scalar,
            pending_target_revision_words=zero_words,
            pending_behavior_revision_words=zero_words,
            pending_action_identity_words=zero_words,
            pending_valid=jnp.asarray(False, dtype=jnp.bool_),
            rng_key=keys[0],
            decision_words=zero_words,
            update_words=zero_words,
            target_revision_words=zero_words,
        )

    def _require_state_contract(self, state: NonlinearAverageRewardActorCriticState) -> int:
        if type(state) is not NonlinearAverageRewardActorCriticState:
            raise TypeError("state must be an exact NonlinearAverageRewardActorCriticState")
        if getattr(state.actor_trunk_w, "ndim", None) != 2:
            raise ValueError("actor_trunk_w must have rank 2")
        feature_dim = state.actor_trunk_w.shape[1]
        hidden = self._config.hidden_size
        actions = self._config.n_actions
        if feature_dim < 1:
            raise ValueError("state feature dimension must be positive")
        float_contracts = {
            "actor_trunk_w": ((hidden, feature_dim), state.actor_trunk_w),
            "actor_trunk_b": ((hidden,), state.actor_trunk_b),
            "actor_head_w": ((actions, hidden), state.actor_head_w),
            "actor_head_b": ((actions,), state.actor_head_b),
            "critic_trunk_w": ((hidden, feature_dim), state.critic_trunk_w),
            "critic_trunk_b": ((hidden,), state.critic_trunk_b),
            "critic_head_w": ((hidden,), state.critic_head_w),
            "critic_head_b": ((), state.critic_head_b),
            "actor_trunk_trace_w": ((hidden, feature_dim), state.actor_trunk_trace_w),
            "actor_trunk_trace_b": ((hidden,), state.actor_trunk_trace_b),
            "actor_head_trace_w": ((actions, hidden), state.actor_head_trace_w),
            "actor_head_trace_b": ((actions,), state.actor_head_trace_b),
            "critic_trunk_trace_w": ((hidden, feature_dim), state.critic_trunk_trace_w),
            "critic_trunk_trace_b": ((hidden,), state.critic_trunk_trace_b),
            "critic_head_trace_w": ((hidden,), state.critic_head_trace_w),
            "critic_head_trace_b": ((), state.critic_head_trace_b),
            "actor_trunk_velocity_w": ((hidden, feature_dim), state.actor_trunk_velocity_w),
            "actor_trunk_velocity_b": ((hidden,), state.actor_trunk_velocity_b),
            "actor_head_velocity_w": ((actions, hidden), state.actor_head_velocity_w),
            "actor_head_velocity_b": ((actions,), state.actor_head_velocity_b),
            "critic_trunk_velocity_w": ((hidden, feature_dim), state.critic_trunk_velocity_w),
            "critic_trunk_velocity_b": ((hidden,), state.critic_trunk_velocity_b),
            "critic_head_velocity_w": ((hidden,), state.critic_head_velocity_w),
            "critic_head_velocity_b": ((), state.critic_head_velocity_b),
            "average_reward": ((), state.average_reward),
            "actor_head_utility": ((), state.actor_head_utility),
            "actor_trunk_utility": ((), state.actor_trunk_utility),
            "critic_head_utility": ((), state.critic_head_utility),
            "critic_trunk_utility": ((), state.critic_trunk_utility),
            "pending_observation": ((feature_dim,), state.pending_observation),
            "pending_target_policy": ((actions,), state.pending_target_policy),
            "pending_target_log_policy": ((actions,), state.pending_target_log_policy),
            "pending_behavior_policy": ((actions,), state.pending_behavior_policy),
            "pending_behavior_log_policy": ((actions,), state.pending_behavior_log_policy),
            "pending_behavior_owner_digest": (
                (NONLINEAR_AVERAGE_REWARD_ACTOR_CRITIC_OWNER_DIGEST_WORDS,),
                state.pending_behavior_owner_digest,
            ),
            "pending_target_log_probability": ((), state.pending_target_log_probability),
            "pending_behavior_log_probability": ((), state.pending_behavior_log_probability),
        }
        owner_shape = (NONLINEAR_AVERAGE_REWARD_ACTOR_CRITIC_OWNER_DIGEST_WORDS,)
        owner_shape_and_value = float_contracts.pop("pending_behavior_owner_digest")
        for label, (shape, value) in float_contracts.items():
            _require_array(value, label=label, shape=shape, dtype=jnp.dtype(jnp.float32))
        _require_array(
            owner_shape_and_value[1],
            label="pending_behavior_owner_digest",
            shape=owner_shape,
            dtype=jnp.dtype(jnp.uint32),
        )
        _require_int32_scalar(state.pending_action, label="pending_action")
        _require_bool_scalar(state.pending_valid, label="pending_valid")
        for label in (
            "pending_target_revision_words",
            "pending_behavior_revision_words",
            "pending_action_identity_words",
            "decision_words",
            "update_words",
            "target_revision_words",
        ):
            _require_words(getattr(state, label), label=label)
        _require_threefry_key(state.rng_key, label="state.rng_key")
        return feature_dim

    def _require_behavior_receipt_contract(
        self,
        receipt: DiscreteBehaviorPolicyReceipt,
    ) -> None:
        if type(receipt) is not DiscreteBehaviorPolicyReceipt:
            raise TypeError("behavior receipt must be an exact DiscreteBehaviorPolicyReceipt")
        _require_float32_vector(
            receipt.probabilities,
            self._config.n_actions,
            label="behavior.probabilities",
        )
        _require_float32_vector(
            receipt.log_probabilities,
            self._config.n_actions,
            label="behavior.log_probabilities",
        )
        _require_array(
            receipt.behavior_owner_digest,
            label="behavior.behavior_owner_digest",
            shape=(NONLINEAR_AVERAGE_REWARD_ACTOR_CRITIC_OWNER_DIGEST_WORDS,),
            dtype=jnp.dtype(jnp.uint32),
        )
        _require_words(receipt.revision_words, label="behavior.revision_words")

    def _behavior_receipt_valid(
        self,
        receipt: DiscreteBehaviorPolicyReceipt,
    ) -> Bool[Array, ""]:
        probabilities = receipt.probabilities
        logs = receipt.log_probabilities
        return (
            jnp.all(jnp.isfinite(probabilities))
            & jnp.all(probabilities > 0.0)
            & jnp.isclose(
                jnp.sum(probabilities),
                jnp.asarray(1.0, dtype=jnp.float32),
                rtol=1e-6,
                atol=1e-6,
            )
            & jnp.all(jnp.isfinite(logs))
            & jnp.all(logs <= 0.0)
            & _bits_equal(logs, jnp.log(probabilities))
        )

    def _behavior_owner_valid(
        self,
        receipt: DiscreteBehaviorPolicyReceipt,
    ) -> Bool[Array, ""]:
        expected = jnp.asarray(self._config.behavior_owner_digest, dtype=jnp.uint32)
        return jnp.all(receipt.behavior_owner_digest == expected)

    def _ordinary_behavior_policy(self, target_policy: Array) -> Array:
        epsilon = jnp.asarray(self._config.ordinary_behavior_epsilon, dtype=jnp.float32)
        uniform = jnp.asarray(1.0 / self._config.n_actions, dtype=jnp.float32)
        return (jnp.asarray(1.0, dtype=jnp.float32) - epsilon) * target_policy + epsilon * uniform

    def _objective_policy_valid(
        self,
        target_policy: Array,
        behavior_policy: Array,
    ) -> Bool[Array, ""]:
        if self._config.objective_mode == "ordinary_behavior":
            expected = self._ordinary_behavior_policy(target_policy)
            return jnp.allclose(expected, behavior_policy, rtol=1e-6, atol=1e-7)
        return jnp.asarray(True, dtype=jnp.bool_)

    def _dynamic_state_valid(
        self,
        state: NonlinearAverageRewardActorCriticState,
    ) -> Bool[Array, ""]:
        finite = jnp.asarray(True, dtype=jnp.bool_)
        for value in _state_float_arrays(state):
            finite = finite & jnp.all(jnp.isfinite(value))
        utility_bound = jnp.asarray(self._config.max_component_utility, dtype=jnp.float32)
        utilities = jnp.stack(
            (
                state.actor_head_utility,
                state.actor_trunk_utility,
                state.critic_head_utility,
                state.critic_trunk_utility,
            )
        )
        utility_valid = jnp.all((utilities >= 0.0) & (utilities <= utility_bound))
        parameter_bound = jnp.asarray(
            self._config.max_parameter_magnitude,
            dtype=jnp.float32,
        )
        bounded_parameters = (
            state.actor_trunk_w,
            state.actor_trunk_b,
            state.actor_head_w,
            state.actor_head_b,
            state.critic_trunk_w,
            state.critic_trunk_b,
            state.critic_head_w,
            state.critic_head_b,
            state.average_reward,
        )
        parameter_valid = jnp.asarray(True, dtype=jnp.bool_)
        for parameter in bounded_parameters:
            parameter_valid = parameter_valid & jnp.all(jnp.abs(parameter) <= parameter_bound)
        trace_bound = jnp.asarray(self._config.max_trace_magnitude, dtype=jnp.float32)
        bounded_traces = (
            state.actor_trunk_trace_w,
            state.actor_trunk_trace_b,
            state.actor_head_trace_w,
            state.actor_head_trace_b,
            state.critic_trunk_trace_w,
            state.critic_trunk_trace_b,
            state.critic_head_trace_w,
            state.critic_head_trace_b,
        )
        trace_valid = jnp.asarray(True, dtype=jnp.bool_)
        for trace in bounded_traces:
            trace_valid = trace_valid & jnp.all(jnp.abs(trace) <= trace_bound)
        velocity_bound = jnp.asarray(
            self._config.max_update_component_magnitude,
            dtype=jnp.float32,
        )
        bounded_velocities = (
            state.actor_trunk_velocity_w,
            state.actor_trunk_velocity_b,
            state.actor_head_velocity_w,
            state.actor_head_velocity_b,
            state.critic_trunk_velocity_w,
            state.critic_trunk_velocity_b,
            state.critic_head_velocity_w,
            state.critic_head_velocity_b,
        )
        velocity_valid = jnp.asarray(True, dtype=jnp.bool_)
        for velocity in bounded_velocities:
            velocity_valid = velocity_valid & jnp.all(jnp.abs(velocity) <= velocity_bound)
        action_valid = (state.pending_action >= 0) & (
            state.pending_action < self._config.n_actions
        )
        safe_pending_action = jnp.clip(
            state.pending_action,
            0,
            self._config.n_actions - 1,
        )
        pending_valid = (
            action_valid
            & jnp.all(state.pending_target_policy > 0.0)
            & jnp.all(state.pending_behavior_policy > 0.0)
            & jnp.isclose(jnp.sum(state.pending_target_policy), 1.0, rtol=1e-6, atol=1e-6)
            & jnp.isclose(jnp.sum(state.pending_behavior_policy), 1.0, rtol=1e-6, atol=1e-6)
            & jnp.allclose(
                state.pending_target_log_policy,
                jnp.log(state.pending_target_policy),
                rtol=1e-6,
                atol=1e-7,
            )
            & _bits_equal(
                state.pending_behavior_log_policy,
                jnp.log(state.pending_behavior_policy),
            )
            & _bits_equal(
                state.pending_target_log_probability,
                state.pending_target_log_policy[safe_pending_action],
            )
            & _bits_equal(
                state.pending_behavior_log_probability,
                state.pending_behavior_log_policy[safe_pending_action],
            )
            & jnp.all(
                state.pending_behavior_owner_digest
                == jnp.asarray(self._config.behavior_owner_digest, dtype=jnp.uint32)
            )
            & jnp.all(state.pending_target_revision_words == state.target_revision_words)
            & jnp.all(state.pending_action_identity_words == state.decision_words)
            & self._objective_policy_valid(
                state.pending_target_policy,
                state.pending_behavior_policy,
            )
        )
        pending_empty = (
            (state.pending_action == -1)
            & jnp.all(state.pending_observation == 0.0)
            & jnp.all(state.pending_target_policy == 0.0)
            & jnp.all(state.pending_target_log_policy == 0.0)
            & jnp.all(state.pending_behavior_policy == 0.0)
            & jnp.all(state.pending_behavior_log_policy == 0.0)
            & jnp.all(state.pending_behavior_owner_digest == 0)
            & (state.pending_target_log_probability == 0.0)
            & (state.pending_behavior_log_probability == 0.0)
            & jnp.all(state.pending_target_revision_words == 0)
            & jnp.all(state.pending_behavior_revision_words == 0)
            & jnp.all(state.pending_action_identity_words == 0)
        )
        zero_words = jnp.zeros((2,), dtype=jnp.uint32)
        proposed_decision, update_capacity = _increment_words(state.update_words)
        actor_changes = (
            self._config.actor_head_plasticity == "plastic"
            or self._config.actor_trunk_plasticity == "plastic"
        )
        target_clock_valid = (
            jnp.all(state.target_revision_words == state.update_words)
            if actor_changes
            else jnp.all(state.target_revision_words == zero_words)
        )
        empty_clock_valid = (
            jnp.all(state.decision_words == zero_words)
            & jnp.all(state.update_words == zero_words)
            & jnp.all(state.target_revision_words == zero_words)
        )
        armed_clock_valid = (
            update_capacity
            & jnp.all(state.decision_words == proposed_decision)
            & target_clock_valid
        )
        clock_valid = jnp.where(state.pending_valid, armed_clock_valid, empty_clock_valid)
        return (
            finite
            & utility_valid
            & parameter_valid
            & trace_valid
            & velocity_valid
            & clock_valid
            & jnp.where(state.pending_valid, pending_valid, pending_empty)
        )

    def state_valid(
        self,
        state: NonlinearAverageRewardActorCriticState,
    ) -> Bool[Array, ""]:
        self._require_state_contract(state)
        return self._dynamic_state_valid(state)

    def target_policy(
        self,
        state: NonlinearAverageRewardActorCriticState,
        observation: Array,
    ) -> Float[Array, " action"]:
        feature_dim = self._require_state_contract(state)
        observation = _require_float32_vector(observation, feature_dim, label="observation")
        return _actor_forward(state, observation, self._config.policy_temperature)[1]

    def ordinary_behavior_receipt(
        self,
        state: NonlinearAverageRewardActorCriticState,
        observation: Array,
        revision_words: Array,
    ) -> DiscreteBehaviorPolicyReceipt:
        """Construct the exact epsilon-mixture receipt for ordinary mode.

        This helper is for an already authoritative state, including the first
        decision.  For a plastic successor use :meth:`propose_update` followed
        by :meth:`ordinary_successor_behavior_receipt`; commit validates that
        exact post-update mixture before drawing.
        """

        target = self.target_policy(state, observation)
        revision_words = _require_words(revision_words, label="revision_words")
        behavior = self._ordinary_behavior_policy(target)
        return DiscreteBehaviorPolicyReceipt(
            probabilities=behavior,
            log_probabilities=jnp.log(behavior),
            behavior_owner_digest=jnp.asarray(
                self._config.behavior_owner_digest,
                dtype=jnp.uint32,
            ),
            revision_words=revision_words,
        )

    def _bind_sample(
        self,
        state: NonlinearAverageRewardActorCriticState,
        observation: Array,
        target_policy: Array,
        target_log_policy: Array,
        behavior: DiscreteBehaviorPolicyReceipt,
        decision_words: Array,
        next_rng_key: Array,
        sample_key: Array,
    ) -> tuple[NonlinearAverageRewardActorCriticState, NonlinearAverageRewardActionRecord]:
        action = jr.categorical(sample_key, behavior.log_probabilities).astype(jnp.int32)
        sampled = dataclasses.replace(
            state,
            pending_observation=observation,
            pending_action=action,
            pending_target_policy=target_policy,
            pending_target_log_policy=target_log_policy,
            pending_behavior_policy=behavior.probabilities,
            pending_behavior_log_policy=behavior.log_probabilities,
            pending_behavior_owner_digest=behavior.behavior_owner_digest,
            pending_target_log_probability=target_log_policy[action],
            pending_behavior_log_probability=behavior.log_probabilities[action],
            pending_target_revision_words=state.target_revision_words,
            pending_behavior_revision_words=behavior.revision_words,
            pending_action_identity_words=decision_words,
            pending_valid=jnp.asarray(True, dtype=jnp.bool_),
            rng_key=next_rng_key,
            decision_words=decision_words,
        )
        return sampled, _record_from_state(sampled)

    def start(
        self,
        state: NonlinearAverageRewardActorCriticState,
        observation: Array,
        behavior: DiscreteBehaviorPolicyReceipt,
    ) -> NonlinearAverageRewardStartResult:
        """Validate and atomically cache the first sampled continuing action."""

        feature_dim = self._require_state_contract(state)
        self._require_behavior_receipt_contract(behavior)
        observation = _require_float32_vector(observation, feature_dim, label="observation")
        state_valid = self._dynamic_state_valid(state)
        target_hidden, target_policy, target_log_policy = _actor_forward(
            state,
            observation,
            self._config.policy_temperature,
        )
        del target_hidden
        target_support_valid = (
            jnp.all(jnp.isfinite(target_policy))
            & jnp.all(jnp.isfinite(target_log_policy))
            & jnp.all(target_policy > 0.0)
        )
        behavior_valid = self._behavior_receipt_valid(behavior)
        behavior_owner_valid = self._behavior_owner_valid(behavior)
        objective_valid = self._objective_policy_valid(target_policy, behavior.probabilities)
        source_valid = (
            jnp.all(jnp.isfinite(observation))
            & target_support_valid
            & behavior_valid
            & behavior_owner_valid
            & objective_valid
            & (~state.pending_valid)
        )
        next_decision_words, capacity = _increment_words(state.decision_words)
        uniform = jnp.full(
            (self._config.n_actions,),
            1.0 / self._config.n_actions,
            dtype=jnp.float32,
        )
        safe_target_policy = jnp.where(target_support_valid, target_policy, uniform)
        safe_target_log_policy = jnp.where(
            target_support_valid,
            target_log_policy,
            jnp.log(uniform),
        )
        next_key, sample_key = jr.split(state.rng_key)
        start_ready = state_valid & source_valid & capacity

        def bind(_: None) -> tuple[
            NonlinearAverageRewardActorCriticState,
            NonlinearAverageRewardActionRecord,
        ]:
            return self._bind_sample(
                state,
                observation,
                safe_target_policy,
                safe_target_log_policy,
                behavior,
                next_decision_words,
                next_key,
                sample_key,
            )

        def reject(_: None) -> tuple[
            NonlinearAverageRewardActorCriticState,
            NonlinearAverageRewardActionRecord,
        ]:
            return state, _empty_record(state)

        sampled, record = jax.lax.cond(start_ready, bind, reject, operand=None)
        candidate_valid = start_ready & self._dynamic_state_valid(sampled)
        applied = start_ready & candidate_valid
        next_state = _tree_select(applied, sampled, state)
        diagnostic_record = _tree_select(applied, record, _empty_record(state))
        return NonlinearAverageRewardStartResult(
            state=next_state,
            record=diagnostic_record,
            action=jnp.where(applied, record.action, jnp.asarray(-1, dtype=jnp.int32)),
            target_policy=safe_target_policy,
            behavior_policy=behavior.probabilities,
            state_valid=state_valid,
            behavior_receipt_valid=behavior_valid,
            behavior_owner_valid=behavior_owner_valid,
            target_support_valid=target_support_valid,
            objective_policy_valid=objective_valid,
            lifetime_capacity_available=capacity,
            start_applied=applied,
        )

    def _require_record_contract(
        self,
        record: NonlinearAverageRewardActionRecord,
        feature_dim: int,
    ) -> None:
        if type(record) is not NonlinearAverageRewardActionRecord:
            raise TypeError("record must be an exact NonlinearAverageRewardActionRecord")
        _require_float32_vector(record.observation, feature_dim, label="record.observation")
        _require_int32_scalar(record.action, label="record.action")
        for label in (
            "target_policy",
            "target_log_policy",
            "behavior_policy",
            "behavior_log_policy",
        ):
            _require_float32_vector(
                getattr(record, label),
                self._config.n_actions,
                label=f"record.{label}",
            )
        _require_float32_scalar(
            record.target_log_probability,
            label="record.target_log_probability",
        )
        _require_float32_scalar(
            record.behavior_log_probability,
            label="record.behavior_log_probability",
        )
        _require_array(
            record.behavior_owner_digest,
            label="record.behavior_owner_digest",
            shape=(NONLINEAR_AVERAGE_REWARD_ACTOR_CRITIC_OWNER_DIGEST_WORDS,),
            dtype=jnp.dtype(jnp.uint32),
        )
        _require_words(record.target_revision_words, label="record.target_revision_words")
        _require_words(record.behavior_revision_words, label="record.behavior_revision_words")
        _require_words(record.action_identity_words, label="record.action_identity_words")

    def _record_identity_valid(
        self,
        state: NonlinearAverageRewardActorCriticState,
        record: NonlinearAverageRewardActionRecord,
    ) -> Bool[Array, ""]:
        return (
            state.pending_valid
            & _bits_equal(record.observation, state.pending_observation)
            & (record.action == state.pending_action)
            & _bits_equal(record.target_policy, state.pending_target_policy)
            & _bits_equal(record.target_log_policy, state.pending_target_log_policy)
            & _bits_equal(record.behavior_policy, state.pending_behavior_policy)
            & _bits_equal(record.behavior_log_policy, state.pending_behavior_log_policy)
            & jnp.all(
                record.behavior_owner_digest == state.pending_behavior_owner_digest
            )
            & _bits_equal(
                record.target_log_probability,
                state.pending_target_log_probability,
            )
            & _bits_equal(
                record.behavior_log_probability,
                state.pending_behavior_log_probability,
            )
            & jnp.all(record.target_revision_words == state.pending_target_revision_words)
            & jnp.all(record.behavior_revision_words == state.pending_behavior_revision_words)
            & jnp.all(record.action_identity_words == state.pending_action_identity_words)
        )

    @staticmethod
    def _bounded_ema(old: Array, signal: Array, decay: Array, bound: Array) -> Array:
        clipped_signal = jnp.clip(signal, 0.0, bound)
        return jnp.clip(decay * old + (1.0 - decay) * clipped_signal, 0.0, bound)

    def propose_update(
        self,
        state: NonlinearAverageRewardActorCriticState,
        record: NonlinearAverageRewardActionRecord,
        reward: Array,
        next_observation: Array,
    ) -> NonlinearAverageRewardUpdateProposal:
        """Propose a source-bound learning commit without consuming RNG."""

        feature_dim = self._require_state_contract(state)
        self._require_record_contract(record, feature_dim)
        reward = _require_float32_scalar(reward, label="reward")
        next_observation = _require_float32_vector(
            next_observation,
            feature_dim,
            label="next_observation",
        )
        state_valid = self._dynamic_state_valid(state)
        record_valid = self._record_identity_valid(state, record)
        transition_valid = jnp.isfinite(reward) & jnp.all(jnp.isfinite(next_observation))

        actor_hidden, target_policy, target_log_policy = _actor_forward(
            state,
            state.pending_observation,
            self._config.policy_temperature,
        )
        target_support_valid = (
            jnp.all(jnp.isfinite(target_policy))
            & jnp.all(jnp.isfinite(target_log_policy))
            & jnp.all(target_policy > 0.0)
        )
        target_policy_valid = (
            target_support_valid
            & _bits_equal(target_policy, state.pending_target_policy)
            & _bits_equal(target_log_policy, state.pending_target_log_policy)
            & jnp.all(state.pending_target_revision_words == state.target_revision_words)
        )
        critic_hidden, value = _critic_forward(state, state.pending_observation)
        next_critic_hidden, next_value = _critic_forward(state, next_observation)
        td_error = reward - state.average_reward + next_value - value
        forward_valid = (
            jnp.all(jnp.isfinite(actor_hidden))
            & target_support_valid
            & jnp.all(jnp.isfinite(critic_hidden))
            & jnp.all(jnp.isfinite(next_critic_hidden))
            & jnp.isfinite(value)
            & jnp.isfinite(next_value)
            & jnp.isfinite(td_error)
        )

        safe_action = jnp.clip(state.pending_action, 0, self._config.n_actions - 1)
        log_ratio = (
            state.pending_target_log_probability - state.pending_behavior_log_probability
        )
        max_log = jnp.log(jnp.asarray(jnp.finfo(jnp.float32).max, dtype=jnp.float32))
        raw_ratio_candidate = jnp.exp(log_ratio)
        ratio_valid = (
            jnp.isfinite(log_ratio)
            & (log_ratio <= max_log)
            & jnp.isfinite(raw_ratio_candidate)
            & (raw_ratio_candidate > 0.0)
        )
        raw_ratio = jnp.where(ratio_valid, raw_ratio_candidate, jnp.float32(0.0))
        clipped_ratio = jnp.minimum(
            raw_ratio,
            jnp.asarray(self._config.importance_clip, dtype=jnp.float32),
        )
        if self._config.objective_mode == "ordinary_behavior":
            epsilon = jnp.asarray(
                self._config.ordinary_behavior_epsilon,
                dtype=jnp.float32,
            )
            actor_multiplier = (
                (jnp.float32(1.0) - epsilon)
                * state.pending_target_policy[safe_action]
                / state.pending_behavior_policy[safe_action]
            )
            critic_multiplier = jnp.asarray(1.0, dtype=jnp.float32)
            reward_rate_multiplier = jnp.asarray(1.0, dtype=jnp.float32)
        else:
            actor_multiplier = clipped_ratio
            critic_multiplier = clipped_ratio
            reward_rate_multiplier = clipped_ratio
        multipliers_valid = (
            jnp.isfinite(actor_multiplier)
            & jnp.isfinite(critic_multiplier)
            & jnp.isfinite(reward_rate_multiplier)
        )

        action_mask = jax.nn.one_hot(
            safe_action,
            self._config.n_actions,
            dtype=jnp.float32,
        )
        actor_logit_score = (action_mask - target_policy) / jnp.asarray(
            self._config.policy_temperature,
            dtype=jnp.float32,
        )
        actor_head_grad_w = jnp.outer(actor_logit_score, actor_hidden)
        actor_head_grad_b = actor_logit_score
        actor_hidden_score = state.actor_head_w.T @ actor_logit_score
        actor_pre_score = actor_hidden_score * (1.0 - jnp.square(actor_hidden))
        actor_trunk_grad_w = jnp.outer(actor_pre_score, state.pending_observation)
        actor_trunk_grad_b = actor_pre_score
        critic_head_grad_w = critic_hidden
        critic_head_grad_b = jnp.asarray(1.0, dtype=jnp.float32)
        critic_pre_score = state.critic_head_w * (1.0 - jnp.square(critic_hidden))
        critic_trunk_grad_w = jnp.outer(critic_pre_score, state.pending_observation)
        critic_trunk_grad_b = critic_pre_score
        gradients = (
            actor_head_grad_w,
            actor_head_grad_b,
            actor_trunk_grad_w,
            actor_trunk_grad_b,
            critic_head_grad_w,
            critic_head_grad_b,
            critic_trunk_grad_w,
            critic_trunk_grad_b,
        )
        gradients_valid = jnp.asarray(True, dtype=jnp.bool_)
        for gradient in gradients:
            gradients_valid = gradients_valid & jnp.all(jnp.isfinite(gradient))

        actor_decay = jnp.asarray(self._config.actor_trace_decay, dtype=jnp.float32)
        critic_decay = jnp.asarray(self._config.critic_trace_decay, dtype=jnp.float32)
        if self._config.objective_mode == "ordinary_behavior":
            next_actor_head_trace_w = (
                actor_decay * state.actor_head_trace_w
                + actor_multiplier * actor_head_grad_w
            )
            next_actor_head_trace_b = (
                actor_decay * state.actor_head_trace_b
                + actor_multiplier * actor_head_grad_b
            )
            next_actor_trunk_trace_w = (
                actor_decay * state.actor_trunk_trace_w
                + actor_multiplier * actor_trunk_grad_w
            )
            next_actor_trunk_trace_b = (
                actor_decay * state.actor_trunk_trace_b
                + actor_multiplier * actor_trunk_grad_b
            )
            next_critic_head_trace_w = (
                critic_decay * state.critic_head_trace_w + critic_head_grad_w
            )
            next_critic_head_trace_b = (
                critic_decay * state.critic_head_trace_b + critic_head_grad_b
            )
            next_critic_trunk_trace_w = (
                critic_decay * state.critic_trunk_trace_w + critic_trunk_grad_w
            )
            next_critic_trunk_trace_b = (
                critic_decay * state.critic_trunk_trace_b + critic_trunk_grad_b
            )
        else:
            next_actor_head_trace_w = actor_multiplier * (
                actor_decay * state.actor_head_trace_w + actor_head_grad_w
            )
            next_actor_head_trace_b = actor_multiplier * (
                actor_decay * state.actor_head_trace_b + actor_head_grad_b
            )
            next_actor_trunk_trace_w = actor_multiplier * (
                actor_decay * state.actor_trunk_trace_w + actor_trunk_grad_w
            )
            next_actor_trunk_trace_b = actor_multiplier * (
                actor_decay * state.actor_trunk_trace_b + actor_trunk_grad_b
            )
            next_critic_head_trace_w = critic_multiplier * (
                critic_decay * state.critic_head_trace_w + critic_head_grad_w
            )
            next_critic_head_trace_b = critic_multiplier * (
                critic_decay * state.critic_head_trace_b + critic_head_grad_b
            )
            next_critic_trunk_trace_w = critic_multiplier * (
                critic_decay * state.critic_trunk_trace_w + critic_trunk_grad_w
            )
            next_critic_trunk_trace_b = critic_multiplier * (
                critic_decay * state.critic_trunk_trace_b + critic_trunk_grad_b
            )
        trace_candidates = (
            next_actor_head_trace_w,
            next_actor_head_trace_b,
            next_actor_trunk_trace_w,
            next_actor_trunk_trace_b,
            next_critic_head_trace_w,
            next_critic_head_trace_b,
            next_critic_trunk_trace_w,
            next_critic_trunk_trace_b,
        )
        trace_bound = jnp.asarray(self._config.max_trace_magnitude, dtype=jnp.float32)
        traces_valid = jnp.asarray(True, dtype=jnp.bool_)
        for trace in trace_candidates:
            traces_valid = traces_valid & jnp.all(jnp.isfinite(trace)) & jnp.all(
                jnp.abs(trace) <= trace_bound
            )

        actor_head_plastic = self._config.actor_head_plasticity == "plastic"
        actor_trunk_plastic = self._config.actor_trunk_plasticity == "plastic"
        critic_head_plastic = self._config.critic_head_plasticity == "plastic"
        critic_trunk_plastic = self._config.critic_trunk_plasticity == "plastic"
        momentum = jnp.asarray(self._config.momentum, dtype=jnp.float32)
        update_bound = jnp.asarray(
            self._config.max_update_component_magnitude,
            dtype=jnp.float32,
        )

        def velocity(old: Array, step_size: float, trace: Array) -> tuple[Array, Array]:
            raw = momentum * old + jnp.asarray(step_size, dtype=jnp.float32) * td_error * trace
            return raw, jnp.clip(raw, -update_bound, update_bound)

        raw_actor_head_velocity_w, actor_head_velocity_w = velocity(
            state.actor_head_velocity_w,
            self._config.actor_head_step_size,
            next_actor_head_trace_w,
        )
        raw_actor_head_velocity_b, actor_head_velocity_b = velocity(
            state.actor_head_velocity_b,
            self._config.actor_head_step_size,
            next_actor_head_trace_b,
        )
        raw_actor_trunk_velocity_w, actor_trunk_velocity_w = velocity(
            state.actor_trunk_velocity_w,
            self._config.actor_trunk_step_size,
            next_actor_trunk_trace_w,
        )
        raw_actor_trunk_velocity_b, actor_trunk_velocity_b = velocity(
            state.actor_trunk_velocity_b,
            self._config.actor_trunk_step_size,
            next_actor_trunk_trace_b,
        )
        raw_critic_head_velocity_w, critic_head_velocity_w = velocity(
            state.critic_head_velocity_w,
            self._config.critic_head_step_size,
            next_critic_head_trace_w,
        )
        raw_critic_head_velocity_b, critic_head_velocity_b = velocity(
            state.critic_head_velocity_b,
            self._config.critic_head_step_size,
            next_critic_head_trace_b,
        )
        raw_critic_trunk_velocity_w, critic_trunk_velocity_w = velocity(
            state.critic_trunk_velocity_w,
            self._config.critic_trunk_step_size,
            next_critic_trunk_trace_w,
        )
        raw_critic_trunk_velocity_b, critic_trunk_velocity_b = velocity(
            state.critic_trunk_velocity_b,
            self._config.critic_trunk_step_size,
            next_critic_trunk_trace_b,
        )
        raw_velocities = (
            raw_actor_head_velocity_w,
            raw_actor_head_velocity_b,
            raw_actor_trunk_velocity_w,
            raw_actor_trunk_velocity_b,
            raw_critic_head_velocity_w,
            raw_critic_head_velocity_b,
            raw_critic_trunk_velocity_w,
            raw_critic_trunk_velocity_b,
        )
        raw_velocities_valid = jnp.asarray(True, dtype=jnp.bool_)
        for raw_velocity in raw_velocities:
            raw_velocities_valid = raw_velocities_valid & jnp.all(
                jnp.isfinite(raw_velocity)
            )

        decay = jnp.asarray(self._config.utility_decay, dtype=jnp.float32)
        utility_bound = jnp.asarray(self._config.max_component_utility, dtype=jnp.float32)

        def utility_signal(*traces: Array) -> Array:
            total = sum(jnp.sum(jnp.abs(td_error * item)) for item in traces)
            count = sum(item.size for item in traces)
            return total / jnp.asarray(count, dtype=jnp.float32)

        raw_actor_head_utility = utility_signal(
            next_actor_head_trace_w,
            next_actor_head_trace_b,
        )
        raw_actor_trunk_utility = utility_signal(
            next_actor_trunk_trace_w,
            next_actor_trunk_trace_b,
        )
        raw_critic_head_utility = utility_signal(
            next_critic_head_trace_w,
            next_critic_head_trace_b,
        )
        raw_critic_trunk_utility = utility_signal(
            next_critic_trunk_trace_w,
            next_critic_trunk_trace_b,
        )
        raw_utilities = jnp.stack(
            (
                raw_actor_head_utility,
                raw_actor_trunk_utility,
                raw_critic_head_utility,
                raw_critic_trunk_utility,
            )
        )
        raw_utilities_valid = jnp.all(jnp.isfinite(raw_utilities))
        actor_head_utility = self._bounded_ema(
            state.actor_head_utility,
            raw_actor_head_utility,
            decay,
            utility_bound,
        )
        actor_trunk_utility = self._bounded_ema(
            state.actor_trunk_utility,
            raw_actor_trunk_utility,
            decay,
            utility_bound,
        )
        critic_head_utility = self._bounded_ema(
            state.critic_head_utility,
            raw_critic_head_utility,
            decay,
            utility_bound,
        )
        critic_trunk_utility = self._bounded_ema(
            state.critic_trunk_utility,
            raw_critic_trunk_utility,
            decay,
            utility_bound,
        )
        raw_reward_step = (
            jnp.asarray(self._config.average_reward_step_size, dtype=jnp.float32)
            * reward_rate_multiplier
            * td_error
        )
        raw_reward_step_valid = jnp.isfinite(raw_reward_step)
        reward_step = jnp.clip(raw_reward_step, -update_bound, update_bound)

        proposed_update_words, update_capacity = _increment_words(state.update_words)
        actor_changes = actor_head_plastic or actor_trunk_plastic
        if actor_changes:
            proposed_target_revision, target_capacity = _increment_words(
                state.target_revision_words
            )
        else:
            proposed_target_revision = state.target_revision_words
            target_capacity = jnp.asarray(True, dtype=jnp.bool_)
        proposed_decision_words, decision_capacity = _increment_words(state.decision_words)
        expected_behavior_revision, behavior_revision_capacity = _increment_words(
            state.pending_behavior_revision_words
        )

        learned = dataclasses.replace(
            state,
            actor_trunk_w=(
                state.actor_trunk_w + actor_trunk_velocity_w
                if actor_trunk_plastic
                else state.actor_trunk_w
            ),
            actor_trunk_b=(
                state.actor_trunk_b + actor_trunk_velocity_b
                if actor_trunk_plastic
                else state.actor_trunk_b
            ),
            actor_head_w=(
                state.actor_head_w + actor_head_velocity_w
                if actor_head_plastic
                else state.actor_head_w
            ),
            actor_head_b=(
                state.actor_head_b + actor_head_velocity_b
                if actor_head_plastic
                else state.actor_head_b
            ),
            critic_trunk_w=(
                state.critic_trunk_w + critic_trunk_velocity_w
                if critic_trunk_plastic
                else state.critic_trunk_w
            ),
            critic_trunk_b=(
                state.critic_trunk_b + critic_trunk_velocity_b
                if critic_trunk_plastic
                else state.critic_trunk_b
            ),
            critic_head_w=(
                state.critic_head_w + critic_head_velocity_w
                if critic_head_plastic
                else state.critic_head_w
            ),
            critic_head_b=(
                state.critic_head_b + critic_head_velocity_b
                if critic_head_plastic
                else state.critic_head_b
            ),
            actor_trunk_trace_w=(
                next_actor_trunk_trace_w
                if actor_trunk_plastic
                else state.actor_trunk_trace_w
            ),
            actor_trunk_trace_b=(
                next_actor_trunk_trace_b
                if actor_trunk_plastic
                else state.actor_trunk_trace_b
            ),
            actor_head_trace_w=(
                next_actor_head_trace_w
                if actor_head_plastic
                else state.actor_head_trace_w
            ),
            actor_head_trace_b=(
                next_actor_head_trace_b
                if actor_head_plastic
                else state.actor_head_trace_b
            ),
            critic_trunk_trace_w=(
                next_critic_trunk_trace_w
                if critic_trunk_plastic
                else state.critic_trunk_trace_w
            ),
            critic_trunk_trace_b=(
                next_critic_trunk_trace_b
                if critic_trunk_plastic
                else state.critic_trunk_trace_b
            ),
            critic_head_trace_w=(
                next_critic_head_trace_w
                if critic_head_plastic
                else state.critic_head_trace_w
            ),
            critic_head_trace_b=(
                next_critic_head_trace_b
                if critic_head_plastic
                else state.critic_head_trace_b
            ),
            actor_trunk_velocity_w=(
                actor_trunk_velocity_w
                if actor_trunk_plastic
                else state.actor_trunk_velocity_w
            ),
            actor_trunk_velocity_b=(
                actor_trunk_velocity_b
                if actor_trunk_plastic
                else state.actor_trunk_velocity_b
            ),
            actor_head_velocity_w=(
                actor_head_velocity_w
                if actor_head_plastic
                else state.actor_head_velocity_w
            ),
            actor_head_velocity_b=(
                actor_head_velocity_b
                if actor_head_plastic
                else state.actor_head_velocity_b
            ),
            critic_trunk_velocity_w=(
                critic_trunk_velocity_w
                if critic_trunk_plastic
                else state.critic_trunk_velocity_w
            ),
            critic_trunk_velocity_b=(
                critic_trunk_velocity_b
                if critic_trunk_plastic
                else state.critic_trunk_velocity_b
            ),
            critic_head_velocity_w=(
                critic_head_velocity_w
                if critic_head_plastic
                else state.critic_head_velocity_w
            ),
            critic_head_velocity_b=(
                critic_head_velocity_b
                if critic_head_plastic
                else state.critic_head_velocity_b
            ),
            average_reward=state.average_reward + reward_step,
            actor_head_utility=actor_head_utility,
            actor_trunk_utility=actor_trunk_utility,
            critic_head_utility=critic_head_utility,
            critic_trunk_utility=critic_trunk_utility,
            target_revision_words=proposed_target_revision,
            update_words=proposed_update_words,
        )
        parameter_bound = jnp.asarray(
            self._config.max_parameter_magnitude,
            dtype=jnp.float32,
        )
        parameter_candidates = (
            learned.actor_trunk_w,
            learned.actor_trunk_b,
            learned.actor_head_w,
            learned.actor_head_b,
            learned.critic_trunk_w,
            learned.critic_trunk_b,
            learned.critic_head_w,
            learned.critic_head_b,
            learned.average_reward,
        )
        parameters_valid = jnp.asarray(True, dtype=jnp.bool_)
        for parameter in parameter_candidates:
            parameters_valid = parameters_valid & jnp.all(jnp.isfinite(parameter)) & jnp.all(
                jnp.abs(parameter) <= parameter_bound
            )
        candidate_numeric_valid = (
            parameters_valid
            & jnp.all(jnp.isfinite(raw_utilities))
            & jnp.all(jnp.isfinite(jnp.stack(
                (
                    actor_head_utility,
                    actor_trunk_utility,
                    critic_head_utility,
                    critic_trunk_utility,
                )
            )))
        )
        next_actor_hidden, next_target_policy, next_target_log_policy = _actor_forward(
            learned,
            next_observation,
            self._config.policy_temperature,
        )
        next_target_support_valid = (
            jnp.all(jnp.isfinite(next_actor_hidden))
            & jnp.all(jnp.isfinite(next_target_policy))
            & jnp.all(jnp.isfinite(next_target_log_policy))
            & jnp.all(next_target_policy > 0.0)
            & jnp.isclose(jnp.sum(next_target_policy), 1.0, rtol=1e-6, atol=1e-6)
        )
        numerical_source_valid = (
            forward_valid
            & multipliers_valid
            & gradients_valid
            & traces_valid
            & raw_velocities_valid
            & raw_utilities_valid
            & raw_reward_step_valid
        )
        lifetime_capacity = update_capacity & target_capacity & decision_capacity
        proposal_valid = (
            state_valid
            & record_valid
            & target_policy_valid
            & transition_valid
            & ratio_valid
            & numerical_source_valid
            & candidate_numeric_valid
            & next_target_support_valid
            & lifetime_capacity
            & behavior_revision_capacity
        )
        safe_candidate = _tree_select(proposal_valid, learned, state)
        uniform = jnp.full(
            (self._config.n_actions,),
            1.0 / self._config.n_actions,
            dtype=jnp.float32,
        )
        safe_next_target = jnp.where(proposal_valid, next_target_policy, uniform)
        safe_next_target_log = jnp.where(
            proposal_valid,
            next_target_log_policy,
            jnp.log(uniform),
        )
        diagnostics_valid = (
            state_valid
            & record_valid
            & target_policy_valid
            & transition_valid
            & ratio_valid
            & numerical_source_valid
            & candidate_numeric_valid
            & next_target_support_valid
        )

        def diagnostic(value: Array) -> Array:
            return jnp.where(diagnostics_valid, value, jnp.zeros_like(value))

        return NonlinearAverageRewardUpdateProposal(
            candidate_state=safe_candidate,
            source_record=record,
            reward=reward,
            next_observation=next_observation,
            next_target_policy=safe_next_target,
            next_target_log_policy=safe_next_target_log,
            expected_behavior_revision_words=expected_behavior_revision,
            value=diagnostic(value),
            next_value=diagnostic(next_value),
            td_error=diagnostic(td_error),
            pre_average_reward=state.average_reward,
            proposed_average_reward=diagnostic(learned.average_reward),
            raw_importance_ratio=diagnostic(raw_ratio),
            clipped_importance_ratio=diagnostic(clipped_ratio),
            actor_score_multiplier=diagnostic(actor_multiplier),
            critic_trace_multiplier=diagnostic(critic_multiplier),
            reward_rate_multiplier=diagnostic(reward_rate_multiplier),
            ratio_truncation=diagnostic(raw_ratio - clipped_ratio),
            pre_update_words=state.update_words,
            post_update_words=proposed_update_words,
            pre_decision_words=state.decision_words,
            post_decision_words=proposed_decision_words,
            pre_target_revision_words=state.target_revision_words,
            post_target_revision_words=proposed_target_revision,
            state_valid=state_valid,
            record_identity_valid=record_valid,
            target_policy_valid=target_policy_valid,
            transition_valid=transition_valid,
            numerical_source_valid=numerical_source_valid,
            ratio_valid=ratio_valid,
            behavior_revision_capacity_available=behavior_revision_capacity,
            lifetime_capacity_available=lifetime_capacity,
            candidate_numeric_valid=candidate_numeric_valid,
            next_target_support_valid=next_target_support_valid,
            proposal_valid=proposal_valid,
        )

    def ordinary_successor_behavior_receipt(
        self,
        proposal: NonlinearAverageRewardUpdateProposal,
    ) -> DiscreteBehaviorPolicyReceipt:
        """Build the exact post-proposal epsilon-mixture receipt without RNG."""

        if self._config.objective_mode != "ordinary_behavior":
            raise ValueError("ordinary successor receipts require ordinary_behavior mode")
        if type(proposal) is not NonlinearAverageRewardUpdateProposal:
            raise TypeError("proposal must be an exact NonlinearAverageRewardUpdateProposal")
        behavior = self._ordinary_behavior_policy(proposal.next_target_policy)
        return DiscreteBehaviorPolicyReceipt(
            probabilities=behavior,
            log_probabilities=jnp.log(behavior),
            behavior_owner_digest=jnp.asarray(
                self._config.behavior_owner_digest,
                dtype=jnp.uint32,
            ),
            revision_words=proposal.expected_behavior_revision_words,
        )

    def commit_update(
        self,
        state: NonlinearAverageRewardActorCriticState,
        record: NonlinearAverageRewardActionRecord,
        reward: Array,
        next_observation: Array,
        proposal: NonlinearAverageRewardUpdateProposal,
        next_behavior: DiscreteBehaviorPolicyReceipt,
    ) -> NonlinearAverageRewardUpdateResult:
        """Validate a pure proposal and atomically commit one successor draw."""

        feature_dim = self._require_state_contract(state)
        self._require_record_contract(record, feature_dim)
        self._require_behavior_receipt_contract(next_behavior)
        if type(proposal) is not NonlinearAverageRewardUpdateProposal:
            raise TypeError("proposal must be an exact NonlinearAverageRewardUpdateProposal")
        reward = _require_float32_scalar(reward, label="reward")
        next_observation = _require_float32_vector(
            next_observation,
            feature_dim,
            label="next_observation",
        )
        expected = self.propose_update(state, record, reward, next_observation)
        proposal_identity_valid = _trees_exact(proposal, expected)
        next_behavior_valid = self._behavior_receipt_valid(next_behavior)
        next_owner_valid = self._behavior_owner_valid(next_behavior)
        next_revision_valid = expected.behavior_revision_capacity_available & jnp.all(
            next_behavior.revision_words == expected.expected_behavior_revision_words
        )
        next_objective_valid = self._objective_policy_valid(
            expected.next_target_policy,
            next_behavior.probabilities,
        )
        receipt_ready = (
            next_behavior_valid
            & next_owner_valid
            & next_revision_valid
            & next_objective_valid
        )
        uniform = jnp.full(
            (self._config.n_actions,),
            1.0 / self._config.n_actions,
            dtype=jnp.float32,
        )
        fallback_receipt = DiscreteBehaviorPolicyReceipt(
            probabilities=uniform,
            log_probabilities=jnp.log(uniform),
            behavior_owner_digest=jnp.asarray(
                self._config.behavior_owner_digest,
                dtype=jnp.uint32,
            ),
            revision_words=expected.expected_behavior_revision_words,
        )
        safe_behavior = _tree_select(receipt_ready, next_behavior, fallback_receipt)
        commit_ready = expected.proposal_valid & proposal_identity_valid & receipt_ready
        next_key, sample_key = jr.split(expected.candidate_state.rng_key)

        def bind(_: None) -> tuple[
            NonlinearAverageRewardActorCriticState,
            NonlinearAverageRewardActionRecord,
        ]:
            return self._bind_sample(
                expected.candidate_state,
                next_observation,
                expected.next_target_policy,
                expected.next_target_log_policy,
                safe_behavior,
                expected.post_decision_words,
                next_key,
                sample_key,
            )

        def reject(_: None) -> tuple[
            NonlinearAverageRewardActorCriticState,
            NonlinearAverageRewardActionRecord,
        ]:
            return state, _record_from_state(state)

        successor, successor_record = jax.lax.cond(
            commit_ready,
            bind,
            reject,
            operand=None,
        )
        candidate_state_valid = commit_ready & self._dynamic_state_valid(successor)
        applied = commit_ready & candidate_state_valid
        next_state = _tree_select(applied, successor, state)
        diagnostic_successor = _tree_select(
            applied,
            successor_record,
            _record_from_state(state),
        )

        def diagnostic(value: Array) -> Array:
            return jnp.where(applied, value, jnp.zeros_like(value))

        return NonlinearAverageRewardUpdateResult(
            state=next_state,
            successor_record=diagnostic_successor,
            successor_action=jnp.where(
                applied,
                successor_record.action,
                state.pending_action,
            ),
            value=diagnostic(expected.value),
            next_value=diagnostic(expected.next_value),
            td_error=diagnostic(expected.td_error),
            pre_average_reward=state.average_reward,
            post_average_reward=next_state.average_reward,
            raw_importance_ratio=diagnostic(expected.raw_importance_ratio),
            clipped_importance_ratio=diagnostic(expected.clipped_importance_ratio),
            actor_score_multiplier=diagnostic(expected.actor_score_multiplier),
            critic_trace_multiplier=diagnostic(expected.critic_trace_multiplier),
            reward_rate_multiplier=diagnostic(expected.reward_rate_multiplier),
            ratio_truncation=diagnostic(expected.ratio_truncation),
            pre_update_words=state.update_words,
            post_update_words=next_state.update_words,
            pre_decision_words=state.decision_words,
            post_decision_words=next_state.decision_words,
            pre_target_revision_words=state.target_revision_words,
            post_target_revision_words=next_state.target_revision_words,
            state_valid=expected.state_valid,
            proposal_valid=expected.proposal_valid,
            proposal_identity_valid=proposal_identity_valid,
            record_identity_valid=expected.record_identity_valid,
            target_policy_valid=expected.target_policy_valid,
            transition_valid=expected.transition_valid,
            ratio_valid=expected.ratio_valid,
            next_behavior_receipt_valid=next_behavior_valid,
            next_behavior_owner_valid=next_owner_valid,
            next_behavior_revision_valid=next_revision_valid,
            behavior_revision_capacity_available=(
                expected.behavior_revision_capacity_available
            ),
            next_objective_policy_valid=next_objective_valid,
            lifetime_capacity_available=expected.lifetime_capacity_available,
            candidate_state_valid=candidate_state_valid,
            successor_sampled=applied,
            update_applied=applied,
        )

    def update(
        self,
        state: NonlinearAverageRewardActorCriticState,
        record: NonlinearAverageRewardActionRecord,
        reward: Array,
        next_observation: Array,
        next_behavior: DiscreteBehaviorPolicyReceipt,
    ) -> NonlinearAverageRewardUpdateResult:
        """Convenience proposal/commit transaction with exactly one draw."""

        proposal = self.propose_update(state, record, reward, next_observation)
        return self.commit_update(
            state,
            record,
            reward,
            next_observation,
            proposal,
            next_behavior,
        )

    def resource_budget(
        self,
        state: NonlinearAverageRewardActorCriticState,
    ) -> NonlinearAverageRewardResourceBudget:
        """Partition every persistent array byte exactly once."""

        self._require_state_contract(state)
        parameters = (
            state.actor_trunk_w,
            state.actor_trunk_b,
            state.actor_head_w,
            state.actor_head_b,
            state.critic_trunk_w,
            state.critic_trunk_b,
            state.critic_head_w,
            state.critic_head_b,
            state.average_reward,
        )
        traces = (
            state.actor_trunk_trace_w,
            state.actor_trunk_trace_b,
            state.actor_head_trace_w,
            state.actor_head_trace_b,
            state.critic_trunk_trace_w,
            state.critic_trunk_trace_b,
            state.critic_head_trace_w,
            state.critic_head_trace_b,
        )
        optimizers = (
            state.actor_trunk_velocity_w,
            state.actor_trunk_velocity_b,
            state.actor_head_velocity_w,
            state.actor_head_velocity_b,
            state.critic_trunk_velocity_w,
            state.critic_trunk_velocity_b,
            state.critic_head_velocity_w,
            state.critic_head_velocity_b,
        )
        utilities = (
            state.actor_head_utility,
            state.actor_trunk_utility,
            state.critic_head_utility,
            state.critic_trunk_utility,
        )
        pending = (
            state.pending_observation,
            state.pending_action,
            state.pending_target_policy,
            state.pending_target_log_policy,
            state.pending_behavior_policy,
            state.pending_behavior_log_policy,
            state.pending_behavior_owner_digest,
            state.pending_target_log_probability,
            state.pending_behavior_log_probability,
            state.pending_target_revision_words,
            state.pending_behavior_revision_words,
            state.pending_action_identity_words,
            state.pending_valid,
        )
        clocks = (
            state.decision_words,
            state.update_words,
            state.target_revision_words,
        )

        def total(values: tuple[Array, ...]) -> int:
            return sum(_array_nbytes(value) for value in values)

        parameter_nbytes = total(parameters)
        trace_nbytes = total(traces)
        optimizer_nbytes = total(optimizers)
        utility_nbytes = total(utilities)
        pending_cache_nbytes = total(pending)
        clock_nbytes = total(clocks)
        rng_nbytes = _array_nbytes(state.rng_key)
        total_state_nbytes = (
            parameter_nbytes
            + trace_nbytes
            + optimizer_nbytes
            + utility_nbytes
            + pending_cache_nbytes
            + clock_nbytes
            + rng_nbytes
        )
        measured = measure_nonlinear_average_reward_actor_critic_state_nbytes(state)
        if total_state_nbytes != measured:
            raise AssertionError("nonlinear average-reward resource partition is incomplete")
        return NonlinearAverageRewardResourceBudget(
            schema=NONLINEAR_AVERAGE_REWARD_ACTOR_CRITIC_RESOURCE_SCHEMA,
            persistent_bytes_scope=(
                "all-persistent-state-array-leaves; excludes-host-object-overhead,"
                "temporaries,compiler-and-xla-workspaces; not-a-measured-device-peak"
            ),
            parameter_nbytes=parameter_nbytes,
            trace_nbytes=trace_nbytes,
            optimizer_nbytes=optimizer_nbytes,
            utility_nbytes=utility_nbytes,
            pending_cache_nbytes=pending_cache_nbytes,
            clock_nbytes=clock_nbytes,
            rng_nbytes=rng_nbytes,
            total_state_nbytes=total_state_nbytes,
        )

    def save_checkpoint(
        self,
        state: NonlinearAverageRewardActorCriticState,
        path: str | Path,
    ) -> None:
        """Save an exact valid state with construction and resource metadata."""

        feature_dim = self._require_state_contract(state)
        if not bool(jax.device_get(self.state_valid(state))):
            raise ValueError("cannot checkpoint invalid nonlinear average-reward state")
        _save_checkpoint(
            state,
            path,
            metadata={
                "schema": NONLINEAR_AVERAGE_REWARD_ACTOR_CRITIC_CHECKPOINT_SCHEMA,
                "state_schema": NONLINEAR_AVERAGE_REWARD_ACTOR_CRITIC_STATE_SCHEMA,
                "construction": self.to_config(),
                "feature_dim": feature_dim,
                "state_nbytes": measure_nonlinear_average_reward_actor_critic_state_nbytes(
                    state
                ),
            },
        )

    def checkpoint_metadata(self, path: str | Path) -> dict[str, Any]:
        """Load and validate the exact checkpoint metadata manifest."""

        metadata = _load_checkpoint_metadata(path)
        expected = {"schema", "state_schema", "construction", "feature_dim", "state_nbytes"}
        fields = _exact_manifest(
            metadata,
            expected,
            label="nonlinear average-reward checkpoint",
        )
        if fields["schema"] != NONLINEAR_AVERAGE_REWARD_ACTOR_CRITIC_CHECKPOINT_SCHEMA:
            raise ValueError("nonlinear average-reward checkpoint schema is unsupported")
        if fields["state_schema"] != NONLINEAR_AVERAGE_REWARD_ACTOR_CRITIC_STATE_SCHEMA:
            raise ValueError("nonlinear average-reward state schema is unsupported")
        if fields["construction"] != self.to_config():
            raise ValueError("checkpoint construction is incompatible")
        _exact_int(fields["feature_dim"], label="checkpoint feature_dim", minimum=1)
        _exact_int(fields["state_nbytes"], label="checkpoint state_nbytes", minimum=1)
        return fields

    def load_checkpoint(
        self,
        state_template: NonlinearAverageRewardActorCriticState,
        path: str | Path,
    ) -> NonlinearAverageRewardActorCriticState:
        """Restore only a construction-compatible, resource-exact valid state."""

        feature_dim = self._require_state_contract(state_template)
        metadata = self.checkpoint_metadata(path)
        if metadata["feature_dim"] != feature_dim:
            raise ValueError("checkpoint feature dimension does not match template")
        loaded_raw, restored_metadata = _load_checkpoint(state_template, path)
        loaded = cast(NonlinearAverageRewardActorCriticState, loaded_raw)
        if restored_metadata != metadata:
            raise ValueError("checkpoint metadata changed between validation and restore")
        self._require_state_contract(loaded)
        if not bool(jax.device_get(self.state_valid(loaded))):
            raise ValueError("restored nonlinear average-reward state is invalid")
        measured = measure_nonlinear_average_reward_actor_critic_state_nbytes(loaded)
        if measured != metadata["state_nbytes"]:
            raise ValueError("restored nonlinear average-reward resource size is invalid")
        return loaded


def run_nonlinear_average_reward_actor_critic_from_arrays(
    agent: NonlinearAverageRewardActorCritic,
    state: NonlinearAverageRewardActorCriticState,
    observations: Array,
    behavior_probabilities: Array,
    behavior_log_probabilities: Array,
    behavior_owner_digests: Array,
    behavior_revisions: Array,
    rewards: Array,
) -> NonlinearAverageRewardScanResult:
    """Run one start and fixed-shape continuing transactions with ``lax.scan``.

    ``observations`` and behavior receipts have ``steps + 1`` rows.  Each
    transition consumes one reward and always samples the next row's action.
    """

    if type(agent) is not NonlinearAverageRewardActorCritic:
        raise TypeError("agent must be an exact NonlinearAverageRewardActorCritic")
    feature_dim = agent._require_state_contract(state)
    if getattr(observations, "ndim", None) != 2:
        raise ValueError("observations must have rank 2")
    decisions = observations.shape[0]
    if decisions < 1:
        raise ValueError("observations must contain at least one decision")
    steps = decisions - 1
    contracts = (
        (observations, (decisions, feature_dim), jnp.dtype(jnp.float32), "observations"),
        (
            behavior_probabilities,
            (decisions, agent.config.n_actions),
            jnp.dtype(jnp.float32),
            "behavior_probabilities",
        ),
        (
            behavior_log_probabilities,
            (decisions, agent.config.n_actions),
            jnp.dtype(jnp.float32),
            "behavior_log_probabilities",
        ),
        (
            behavior_owner_digests,
            (decisions, NONLINEAR_AVERAGE_REWARD_ACTOR_CRITIC_OWNER_DIGEST_WORDS),
            jnp.dtype(jnp.uint32),
            "behavior_owner_digests",
        ),
        (
            behavior_revisions,
            (decisions, 2),
            jnp.dtype(jnp.uint32),
            "behavior_revisions",
        ),
        (rewards, (steps,), jnp.dtype(jnp.float32), "rewards"),
    )
    for value, shape, dtype, label in contracts:
        _require_array(value, label=label, shape=shape, dtype=dtype)

    initial_receipt = DiscreteBehaviorPolicyReceipt(
        probabilities=behavior_probabilities[0],
        log_probabilities=behavior_log_probabilities[0],
        behavior_owner_digest=behavior_owner_digests[0],
        revision_words=behavior_revisions[0],
    )
    start = agent.start(state, observations[0], initial_receipt)

    def body(
        carry: NonlinearAverageRewardActorCriticState,
        sources: tuple[Array, Array, Array, Array, Array, Array],
    ) -> tuple[NonlinearAverageRewardActorCriticState, tuple[Array, ...]]:
        observation, probabilities, logs, owner, revision, reward = sources
        receipt = DiscreteBehaviorPolicyReceipt(
            probabilities=probabilities,
            log_probabilities=logs,
            behavior_owner_digest=owner,
            revision_words=revision,
        )
        result = agent.update(
            carry,
            _record_from_state(carry),
            reward,
            observation,
            receipt,
        )
        return result.state, (
            result.successor_action,
            result.td_error,
            result.post_average_reward,
            result.raw_importance_ratio,
            result.clipped_importance_ratio,
            result.update_applied,
        )

    final_state, outputs = jax.lax.scan(
        body,
        start.state,
        (
            observations[1:],
            behavior_probabilities[1:],
            behavior_log_probabilities[1:],
            behavior_owner_digests[1:],
            behavior_revisions[1:],
            rewards,
        ),
    )
    actions, td_errors, average_rewards, raw_ratios, clipped_ratios, updates = outputs
    all_actions = jnp.concatenate((jnp.reshape(start.action, (1,)), actions), axis=0)
    return NonlinearAverageRewardScanResult(
        state=final_state,
        actions=all_actions,
        td_errors=td_errors,
        average_rewards=average_rewards,
        raw_importance_ratios=raw_ratios,
        clipped_importance_ratios=clipped_ratios,
        update_applied=updates,
        start_applied=start.start_applied,
    )


__all__ = [
    "NONLINEAR_AVERAGE_REWARD_ACTOR_CRITIC_CHECKPOINT_SCHEMA",
    "NONLINEAR_AVERAGE_REWARD_ACTOR_CRITIC_CONFIG_SCHEMA",
    "NONLINEAR_AVERAGE_REWARD_ACTOR_CRITIC_EVIDENCE_LEVEL",
    "NONLINEAR_AVERAGE_REWARD_ACTOR_CRITIC_LIFETIME_SEMANTICS",
    "NONLINEAR_AVERAGE_REWARD_ACTOR_CRITIC_MAX_DECISIONS",
    "NONLINEAR_AVERAGE_REWARD_ACTOR_CRITIC_MAX_UPDATES",
    "NONLINEAR_AVERAGE_REWARD_ACTOR_CRITIC_OWNER_DIGEST_WORDS",
    "NONLINEAR_AVERAGE_REWARD_ACTOR_CRITIC_OUTCOME_STATUS",
    "NONLINEAR_AVERAGE_REWARD_ACTOR_CRITIC_RESOURCE_SCHEMA",
    "NONLINEAR_AVERAGE_REWARD_ACTOR_CRITIC_STATE_SCHEMA",
    "DiscreteBehaviorPolicyReceipt",
    "NonlinearAverageRewardActionRecord",
    "NonlinearAverageRewardActorCritic",
    "NonlinearAverageRewardActorCriticConfig",
    "NonlinearAverageRewardActorCriticState",
    "NonlinearAverageRewardResourceBudget",
    "NonlinearAverageRewardScanResult",
    "NonlinearAverageRewardStartResult",
    "NonlinearAverageRewardUpdateResult",
    "NonlinearAverageRewardUpdateProposal",
    "NonlinearAverageRewardObjectiveMode",
    "NonlinearAverageRewardPlasticityPolicy",
    "measure_nonlinear_average_reward_actor_critic_state_nbytes",
    "run_nonlinear_average_reward_actor_critic_from_arrays",
]
