"""Bounded nonlinear discrete actor-critic with explicit off-policy correction.

This is an isolated WP6 mechanism, not a modification of the existing Horde
actor-critic family.  A one-hidden-layer nonlinear trunk is shared by separate
softmax-actor and scalar-critic heads.  Executed actions are first bound to an
immutable receipt containing the target-policy log probability, behavior-policy
log probability, both policy revisions, and an exact uint64 action identity.
Learning consumes that exact receipt or fails closed without mutating state.

For one accepted transition the implementation uses clipped per-decision
importance sampling

``rho_t = exp(log pi_target(a_t | s_t) - log mu(a_t | s_t))``

``c_t = min(rho_t, importance_clip)``

and advances actor and critic traces as

``e_t = c_t * (discount_t * lambda * e_(t-1) + gradient_t)``.

The pre-update TD error then drives the actor head, critic head, and the two
corresponding shared-trunk traces.  The order is deliberately fixed:

1. validate the complete cached decision and transition;
2. evaluate the pre-update policy, values, TD error, and score gradients;
3. advance clipped importance-weighted traces;
4. advance the independently owned actor, critic, and trunk momentum states;
5. update parameters from those new optimizer states;
6. atomically commit, clear the pending receipt, and advance exact clocks.

Ratio clipping is disclosed by both the raw ratio and ``raw - clipped``
diagnostic.  It is a bias/variance choice, not an unbiased objective, and this
per-decision action ratio does not by itself correct an initial-state or
state-visitation-distribution mismatch.  This module never treats paper-defined
delight or Kondo selection as an off-policy correction.
The implementation and tests establish L0 mechanism contracts only: they make
no convergence, benchmark, retention, safety, or evidence claim.
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
from jax import Array
from jaxtyping import Bool, Float, Int, UInt

from alberta_framework.core.checkpoints import (
    load_checkpoint as _load_checkpoint,
)
from alberta_framework.core.checkpoints import (
    load_checkpoint_metadata as _load_checkpoint_metadata,
)
from alberta_framework.core.checkpoints import (
    save_checkpoint as _save_checkpoint,
)

NONLINEAR_OFF_POLICY_ACTOR_CRITIC_CONFIG_SCHEMA = (
    "alberta.nonlinear-off-policy-actor-critic-config.v1"
)
NONLINEAR_OFF_POLICY_ACTOR_CRITIC_STATE_SCHEMA = (
    "alberta.nonlinear-off-policy-actor-critic-state.v1"
)
NONLINEAR_OFF_POLICY_ACTOR_CRITIC_CHECKPOINT_SCHEMA = (
    "alberta.nonlinear-off-policy-actor-critic-checkpoint.v1"
)
NONLINEAR_OFF_POLICY_ACTOR_CRITIC_RESOURCE_SCHEMA = (
    "alberta.nonlinear-off-policy-actor-critic-resource.v1"
)
NONLINEAR_OFF_POLICY_ACTOR_CRITIC_EVIDENCE_LEVEL = "L0"
NONLINEAR_OFF_POLICY_ACTOR_CRITIC_OUTCOME_STATUS = "not_assessed"
NONLINEAR_OFF_POLICY_ACTOR_CRITIC_LIFETIME_SEMANTICS = "exact-uint64-fail-stop"
NONLINEAR_OFF_POLICY_ACTOR_CRITIC_MAX_DECISIONS = 2**64 - 1
NONLINEAR_OFF_POLICY_ACTOR_CRITIC_MAX_UPDATES = 2**64 - 1

PlasticityPolicy = Literal["plastic", "frozen"]

_UINT32_MAX = 2**32 - 1


def _exact_manifest(
    payload: Mapping[str, Any],
    expected: set[str],
    *,
    label: str,
) -> dict[str, Any]:
    """Copy a host mapping after rejecting missing and unknown fields."""

    if type(payload) is not dict:
        raise TypeError(f"{label} must be an exact dict")
    fields = dict(payload)
    supplied = set(fields)
    if supplied != expected:
        missing = sorted(expected - supplied)
        extra = sorted(supplied - expected)
        raise ValueError(f"{label} field manifest is not exact; missing={missing}, extra={extra}")
    return fields


def _exact_int(value: Any, *, label: str, minimum: int) -> int:
    if type(value) is not int:
        raise TypeError(f"{label} must be an exact integer")
    if value < minimum:
        raise ValueError(f"{label} must be >= {minimum}")
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
    below = scalar <= minimum if strict_minimum else scalar < minimum
    if below or (maximum is not None and scalar > maximum):
        comparator = ">" if strict_minimum else ">="
        upper = "" if maximum is None else f" and <= {maximum}"
        raise ValueError(f"{label} must be {comparator} {minimum}{upper}")
    return scalar


def _plasticity(value: Any, *, label: str) -> PlasticityPolicy:
    if type(value) is not str:
        raise TypeError(f"{label} must be an exact string")
    if value not in ("plastic", "frozen"):
        raise ValueError(f"{label} must be 'plastic' or 'frozen'")
    return cast(PlasticityPolicy, value)


def _require_feature_dim(value: Any) -> int:
    return _exact_int(value, label="feature_dim", minimum=1)


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


def _require_float32_vector(value: Any, size: int, *, label: str) -> Array:
    return _require_array(
        value,
        label=label,
        shape=(size,),
        dtype=jnp.dtype(jnp.float32),
    )


def _require_float32_scalar(value: Any, *, label: str) -> Array:
    return _require_array(
        value,
        label=label,
        shape=(),
        dtype=jnp.dtype(jnp.float32),
    )


def _require_int32_scalar(value: Any, *, label: str) -> Array:
    return _require_array(
        value,
        label=label,
        shape=(),
        dtype=jnp.dtype(jnp.int32),
    )


def _require_words(value: Any, *, label: str) -> Array:
    return _require_array(
        value,
        label=label,
        shape=(2,),
        dtype=jnp.dtype(jnp.uint32),
    )


def _require_bool_scalar(value: Any, *, label: str) -> Array:
    return _require_array(
        value,
        label=label,
        shape=(),
        dtype=jnp.dtype(jnp.bool_),
    )


def _require_threefry_key(value: Any, *, label: str) -> None:
    try:
        key_data = jr.key_data(value)
        implementation = str(jr.key_impl(value))
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{label} must be one typed Threefry JAX key") from exc
    if (
        getattr(value, "shape", None) != ()
        or key_data.shape != (2,)
        or key_data.dtype != jnp.dtype(jnp.uint32)
        or implementation != "threefry2x32"
    ):
        raise TypeError(f"{label} must be one typed Threefry JAX key")


def _increment_words(words: Array) -> tuple[UInt[Array, " 2"], Bool[Array, ""]]:
    """Return a big-endian uint64 successor and fail-stop capacity verdict."""

    maximum = jnp.asarray(_UINT32_MAX, dtype=jnp.uint32)
    carry = words[1] == maximum
    capacity = ~(carry & (words[0] == maximum))
    next_low = words[1] + jnp.asarray(1, dtype=jnp.uint32)
    next_high = words[0] + carry.astype(jnp.uint32)
    proposed = jnp.stack((next_high, next_low)).astype(jnp.uint32)
    return proposed, capacity


def _float_bits_equal(left: Array, right: Array) -> Bool[Array, ""]:
    left_bits = jax.lax.bitcast_convert_type(left, jnp.uint32)
    right_bits = jax.lax.bitcast_convert_type(right, jnp.uint32)
    return jnp.all(left_bits == right_bits)


def _array_nbytes(value: Array) -> int:
    return int(value.size) * int(value.dtype.itemsize)


@dataclasses.dataclass(frozen=True)
class NonlinearOffPolicyActorCriticConfig:
    """Static architecture, correction, optimizer, and plasticity policy."""

    n_actions: int
    hidden_size: int = 64
    actor_step_size: float = 0.001
    critic_step_size: float = 0.01
    trunk_actor_step_size: float = 0.001
    trunk_critic_step_size: float = 0.01
    actor_trace_decay: float = 0.9
    critic_trace_decay: float = 0.9
    momentum: float = 0.0
    importance_clip: float = 1.0
    initialization_scale: float = 0.1
    actor_plasticity: PlasticityPolicy = "plastic"
    critic_plasticity: PlasticityPolicy = "plastic"
    trunk_plasticity: PlasticityPolicy = "plastic"

    def __post_init__(self) -> None:
        _exact_int(self.n_actions, label="n_actions", minimum=2)
        _exact_int(self.hidden_size, label="hidden_size", minimum=1)
        for name in (
            "actor_step_size",
            "critic_step_size",
            "trunk_actor_step_size",
            "trunk_critic_step_size",
        ):
            _finite_real(getattr(self, name), label=name, minimum=0.0)
        for name in ("actor_trace_decay", "critic_trace_decay"):
            _finite_real(getattr(self, name), label=name, minimum=0.0, maximum=1.0)
        _finite_real(self.momentum, label="momentum", minimum=0.0, maximum=1.0)
        if self.momentum == 1.0:
            raise ValueError("momentum must be < 1.0")
        _finite_real(
            self.importance_clip,
            label="importance_clip",
            minimum=0.0,
            strict_minimum=True,
        )
        _finite_real(
            self.initialization_scale,
            label="initialization_scale",
            minimum=0.0,
            strict_minimum=True,
        )
        _plasticity(self.actor_plasticity, label="actor_plasticity")
        _plasticity(self.critic_plasticity, label="critic_plasticity")
        _plasticity(self.trunk_plasticity, label="trunk_plasticity")

    def to_config(self) -> dict[str, Any]:
        payload = dataclasses.asdict(self)
        return {
            "type": "NonlinearOffPolicyActorCritic",
            "schema": NONLINEAR_OFF_POLICY_ACTOR_CRITIC_CONFIG_SCHEMA,
            "evidence_level": NONLINEAR_OFF_POLICY_ACTOR_CRITIC_EVIDENCE_LEVEL,
            "outcome_status": NONLINEAR_OFF_POLICY_ACTOR_CRITIC_OUTCOME_STATUS,
            **payload,
        }

    @classmethod
    def from_config(cls, payload: dict[str, Any]) -> NonlinearOffPolicyActorCriticConfig:
        expected = {field.name for field in dataclasses.fields(cls)} | {
            "type",
            "schema",
            "evidence_level",
            "outcome_status",
        }
        fields = _exact_manifest(payload, expected, label="off-policy actor-critic config")
        if fields.pop("type") != "NonlinearOffPolicyActorCritic":
            raise ValueError("off-policy actor-critic config type is unsupported")
        if fields.pop("schema") != NONLINEAR_OFF_POLICY_ACTOR_CRITIC_CONFIG_SCHEMA:
            raise ValueError("off-policy actor-critic config schema is unsupported")
        if fields.pop("evidence_level") != NONLINEAR_OFF_POLICY_ACTOR_CRITIC_EVIDENCE_LEVEL:
            raise ValueError("off-policy actor-critic evidence level must remain L0")
        if fields.pop("outcome_status") != NONLINEAR_OFF_POLICY_ACTOR_CRITIC_OUTCOME_STATUS:
            raise ValueError("off-policy actor-critic outcome status must remain not_assessed")
        return cls(**fields)


@chex.dataclass(frozen=True)
class NonlinearOffPolicyActorCriticState:
    """Pure bounded learner state with separate component ownership."""

    trunk_w: Float[Array, "hidden feature"]
    trunk_b: Float[Array, " hidden"]
    actor_w: Float[Array, "action hidden"]
    actor_b: Float[Array, " action"]
    critic_w: Float[Array, " hidden"]
    critic_b: Float[Array, ""]

    actor_head_trace_w: Float[Array, "action hidden"]
    actor_head_trace_b: Float[Array, " action"]
    critic_head_trace_w: Float[Array, " hidden"]
    critic_head_trace_b: Float[Array, ""]
    actor_trunk_trace_w: Float[Array, "hidden feature"]
    actor_trunk_trace_b: Float[Array, " hidden"]
    critic_trunk_trace_w: Float[Array, "hidden feature"]
    critic_trunk_trace_b: Float[Array, " hidden"]

    trunk_velocity_w: Float[Array, "hidden feature"]
    trunk_velocity_b: Float[Array, " hidden"]
    actor_head_velocity_w: Float[Array, "action hidden"]
    actor_head_velocity_b: Float[Array, " action"]
    critic_head_velocity_w: Float[Array, " hidden"]
    critic_head_velocity_b: Float[Array, ""]

    pending_observation: Float[Array, " feature"]
    pending_action: Int[Array, ""]
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
class OffPolicyActionRecord:
    """Transient exact receipt for the action that actually executed."""

    observation: Float[Array, " feature"]
    action: Int[Array, ""]
    target_log_probability: Float[Array, ""]
    behavior_log_probability: Float[Array, ""]
    target_revision_words: UInt[Array, " 2"]
    behavior_revision_words: UInt[Array, " 2"]
    action_identity_words: UInt[Array, " 2"]


@chex.dataclass(frozen=True)
class OffPolicyActionCacheResult:
    """Result of atomically binding one externally executed or sampled action."""

    state: NonlinearOffPolicyActorCriticState
    record: OffPolicyActionRecord
    target_policy: Float[Array, " action"]
    pre_decision_words: UInt[Array, " 2"]
    post_decision_words: UInt[Array, " 2"]
    state_valid: Bool[Array, ""]
    source_valid: Bool[Array, ""]
    behavior_support_valid: Bool[Array, ""]
    cache_available: Bool[Array, ""]
    lifetime_capacity_available: Bool[Array, ""]
    cache_applied: Bool[Array, ""]


@chex.dataclass(frozen=True)
class NonlinearOffPolicyActorCriticUpdateResult:
    """One corrected update plus explicit ratio and transaction diagnostics."""

    state: NonlinearOffPolicyActorCriticState
    value: Float[Array, ""]
    next_value: Float[Array, ""]
    td_error: Float[Array, ""]
    importance_ratio: Float[Array, ""]
    clipped_importance_ratio: Float[Array, ""]
    ratio_truncation: Float[Array, ""]
    ratio_was_clipped: Bool[Array, ""]
    pre_update_words: UInt[Array, " 2"]
    post_update_words: UInt[Array, " 2"]
    pre_target_revision_words: UInt[Array, " 2"]
    post_target_revision_words: UInt[Array, " 2"]
    state_valid: Bool[Array, ""]
    source_valid: Bool[Array, ""]
    behavior_support_valid: Bool[Array, ""]
    record_identity_valid: Bool[Array, ""]
    target_revision_valid: Bool[Array, ""]
    target_log_probability_valid: Bool[Array, ""]
    ratio_valid: Bool[Array, ""]
    lifetime_capacity_available: Bool[Array, ""]
    candidate_state_valid: Bool[Array, ""]
    update_applied: Bool[Array, ""]


@dataclasses.dataclass(frozen=True)
class NonlinearOffPolicyActorCriticResourceBudget:
    """Exact persistent-array accounting for one concrete state."""

    schema: str
    persistent_bytes_scope: str
    parameter_nbytes: int
    trace_nbytes: int
    optimizer_nbytes: int
    pending_cache_nbytes: int
    clock_nbytes: int
    rng_nbytes: int
    total_state_nbytes: int


@chex.dataclass(frozen=True)
class NonlinearOffPolicyActorCriticScanResult:
    """JAX scan result over explicit behavior-policy transitions."""

    state: NonlinearOffPolicyActorCriticState
    td_errors: Float[Array, " steps"]
    importance_ratios: Float[Array, " steps"]
    clipped_importance_ratios: Float[Array, " steps"]
    ratio_truncations: Float[Array, " steps"]
    cache_applied: Bool[Array, " steps"]
    update_applied: Bool[Array, " steps"]


def _forward(
    state: NonlinearOffPolicyActorCriticState,
    observation: Array,
) -> tuple[Array, Array, Array, Array, Array]:
    hidden = jnp.tanh(state.trunk_w @ observation + state.trunk_b)
    logits = state.actor_w @ hidden + state.actor_b
    log_policy = jax.nn.log_softmax(logits)
    policy = jnp.exp(log_policy)
    value = jnp.dot(state.critic_w, hidden) + state.critic_b
    return hidden, logits, policy, log_policy, value


def _empty_pending(
    state: NonlinearOffPolicyActorCriticState,
) -> NonlinearOffPolicyActorCriticState:
    return dataclasses.replace(  # type: ignore[type-var]
        state,
        pending_observation=jnp.zeros_like(state.pending_observation),
        pending_action=jnp.asarray(-1, dtype=jnp.int32),
        pending_target_log_probability=jnp.asarray(0.0, dtype=jnp.float32),
        pending_behavior_log_probability=jnp.asarray(0.0, dtype=jnp.float32),
        pending_target_revision_words=jnp.zeros((2,), dtype=jnp.uint32),
        pending_behavior_revision_words=jnp.zeros((2,), dtype=jnp.uint32),
        pending_action_identity_words=jnp.zeros((2,), dtype=jnp.uint32),
        pending_valid=jnp.asarray(False, dtype=jnp.bool_),
    )


def _record_from_state(state: NonlinearOffPolicyActorCriticState) -> OffPolicyActionRecord:
    return OffPolicyActionRecord(  # type: ignore[call-arg]
        observation=state.pending_observation,
        action=state.pending_action,
        target_log_probability=state.pending_target_log_probability,
        behavior_log_probability=state.pending_behavior_log_probability,
        target_revision_words=state.pending_target_revision_words,
        behavior_revision_words=state.pending_behavior_revision_words,
        action_identity_words=state.pending_action_identity_words,
    )


def _state_float_arrays(state: NonlinearOffPolicyActorCriticState) -> tuple[Array, ...]:
    return (
        state.trunk_w,
        state.trunk_b,
        state.actor_w,
        state.actor_b,
        state.critic_w,
        state.critic_b,
        state.actor_head_trace_w,
        state.actor_head_trace_b,
        state.critic_head_trace_w,
        state.critic_head_trace_b,
        state.actor_trunk_trace_w,
        state.actor_trunk_trace_b,
        state.critic_trunk_trace_w,
        state.critic_trunk_trace_b,
        state.trunk_velocity_w,
        state.trunk_velocity_b,
        state.actor_head_velocity_w,
        state.actor_head_velocity_b,
        state.critic_head_velocity_w,
        state.critic_head_velocity_b,
        state.pending_observation,
        state.pending_target_log_probability,
        state.pending_behavior_log_probability,
    )


def _dynamic_state_valid(
    state: NonlinearOffPolicyActorCriticState,
    *,
    n_actions: int,
) -> Bool[Array, ""]:
    finite = jnp.asarray(True, dtype=jnp.bool_)
    for array in _state_float_arrays(state):
        finite = finite & jnp.all(jnp.isfinite(array))
    pending_valid = (
        (state.pending_action >= 0)
        & (state.pending_action < n_actions)
        & (state.pending_target_log_probability <= 0.0)
        & (state.pending_behavior_log_probability <= 0.0)
        & jnp.all(state.pending_target_revision_words == state.target_revision_words)
        & jnp.all(state.pending_action_identity_words == state.decision_words)
    )
    pending_empty = (
        (state.pending_action == -1)
        & jnp.all(state.pending_observation == 0.0)
        & (state.pending_target_log_probability == 0.0)
        & (state.pending_behavior_log_probability == 0.0)
        & jnp.all(state.pending_target_revision_words == 0)
        & jnp.all(state.pending_behavior_revision_words == 0)
        & jnp.all(state.pending_action_identity_words == 0)
    )
    return finite & jnp.where(state.pending_valid, pending_valid, pending_empty)


def measure_nonlinear_off_policy_actor_critic_state_nbytes(
    state: NonlinearOffPolicyActorCriticState,
) -> int:
    """Return exact bytes occupied by all persistent JAX array leaves."""

    return sum(_array_nbytes(leaf) for leaf in jax.tree.leaves(state))


class NonlinearOffPolicyActorCritic:
    """Nonlinear shared-trunk actor-critic with causal action receipts."""

    def __init__(self, config: NonlinearOffPolicyActorCriticConfig) -> None:
        if type(config) is not NonlinearOffPolicyActorCriticConfig:
            raise TypeError("config must be an exact NonlinearOffPolicyActorCriticConfig")
        self._config = config

    @property
    def config(self) -> NonlinearOffPolicyActorCriticConfig:
        return self._config

    def to_config(self) -> dict[str, Any]:
        return self._config.to_config()

    @classmethod
    def from_config(cls, payload: dict[str, Any]) -> NonlinearOffPolicyActorCritic:
        return cls(NonlinearOffPolicyActorCriticConfig.from_config(payload))

    def init(self, feature_dim: int, key: Array) -> NonlinearOffPolicyActorCriticState:
        """Initialize bounded state from one scalar typed Threefry key."""

        feature_dim = _require_feature_dim(feature_dim)
        _require_threefry_key(key, label="key")
        keys = jr.split(key, 4)
        scale = jnp.asarray(self._config.initialization_scale, dtype=jnp.float32)
        trunk_w = scale * jr.normal(
            keys[1], (self._config.hidden_size, feature_dim), dtype=jnp.float32
        )
        actor_w = scale * jr.normal(
            keys[2], (self._config.n_actions, self._config.hidden_size), dtype=jnp.float32
        )
        critic_w = scale * jr.normal(
            keys[3], (self._config.hidden_size,), dtype=jnp.float32
        )
        zero_trunk_w = jnp.zeros_like(trunk_w)
        zero_trunk_b = jnp.zeros((self._config.hidden_size,), dtype=jnp.float32)
        zero_actor_w = jnp.zeros_like(actor_w)
        zero_actor_b = jnp.zeros((self._config.n_actions,), dtype=jnp.float32)
        zero_critic_w = jnp.zeros_like(critic_w)
        zero_scalar = jnp.asarray(0.0, dtype=jnp.float32)
        zero_words = jnp.zeros((2,), dtype=jnp.uint32)
        return NonlinearOffPolicyActorCriticState(  # type: ignore[call-arg]
            trunk_w=trunk_w,
            trunk_b=zero_trunk_b,
            actor_w=actor_w,
            actor_b=zero_actor_b,
            critic_w=critic_w,
            critic_b=zero_scalar,
            actor_head_trace_w=zero_actor_w,
            actor_head_trace_b=zero_actor_b,
            critic_head_trace_w=zero_critic_w,
            critic_head_trace_b=zero_scalar,
            actor_trunk_trace_w=zero_trunk_w,
            actor_trunk_trace_b=zero_trunk_b,
            critic_trunk_trace_w=zero_trunk_w,
            critic_trunk_trace_b=zero_trunk_b,
            trunk_velocity_w=zero_trunk_w,
            trunk_velocity_b=zero_trunk_b,
            actor_head_velocity_w=zero_actor_w,
            actor_head_velocity_b=zero_actor_b,
            critic_head_velocity_w=zero_critic_w,
            critic_head_velocity_b=zero_scalar,
            pending_observation=jnp.zeros((feature_dim,), dtype=jnp.float32),
            pending_action=jnp.asarray(-1, dtype=jnp.int32),
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

    def _require_state_contract(self, state: NonlinearOffPolicyActorCriticState) -> int:
        if type(state) is not NonlinearOffPolicyActorCriticState:
            raise TypeError("state must be an exact NonlinearOffPolicyActorCriticState")
        if getattr(state.trunk_w, "ndim", None) != 2:
            raise ValueError("trunk_w must have rank 2")
        feature_dim = state.trunk_w.shape[1]
        hidden = self._config.hidden_size
        actions = self._config.n_actions
        if feature_dim < 1:
            raise ValueError("state feature dimension must be positive")
        float_contracts = {
            "trunk_w": ((hidden, feature_dim), state.trunk_w),
            "trunk_b": ((hidden,), state.trunk_b),
            "actor_w": ((actions, hidden), state.actor_w),
            "actor_b": ((actions,), state.actor_b),
            "critic_w": ((hidden,), state.critic_w),
            "critic_b": ((), state.critic_b),
            "actor_head_trace_w": ((actions, hidden), state.actor_head_trace_w),
            "actor_head_trace_b": ((actions,), state.actor_head_trace_b),
            "critic_head_trace_w": ((hidden,), state.critic_head_trace_w),
            "critic_head_trace_b": ((), state.critic_head_trace_b),
            "actor_trunk_trace_w": ((hidden, feature_dim), state.actor_trunk_trace_w),
            "actor_trunk_trace_b": ((hidden,), state.actor_trunk_trace_b),
            "critic_trunk_trace_w": ((hidden, feature_dim), state.critic_trunk_trace_w),
            "critic_trunk_trace_b": ((hidden,), state.critic_trunk_trace_b),
            "trunk_velocity_w": ((hidden, feature_dim), state.trunk_velocity_w),
            "trunk_velocity_b": ((hidden,), state.trunk_velocity_b),
            "actor_head_velocity_w": ((actions, hidden), state.actor_head_velocity_w),
            "actor_head_velocity_b": ((actions,), state.actor_head_velocity_b),
            "critic_head_velocity_w": ((hidden,), state.critic_head_velocity_w),
            "critic_head_velocity_b": ((), state.critic_head_velocity_b),
            "pending_observation": ((feature_dim,), state.pending_observation),
            "pending_target_log_probability": ((), state.pending_target_log_probability),
            "pending_behavior_log_probability": ((), state.pending_behavior_log_probability),
        }
        for label, (shape, value) in float_contracts.items():
            _require_array(
                value,
                label=label,
                shape=shape,
                dtype=jnp.dtype(jnp.float32),
            )
        _require_int32_scalar(state.pending_action, label="pending_action")
        for label in (
            "pending_target_revision_words",
            "pending_behavior_revision_words",
            "pending_action_identity_words",
            "decision_words",
            "update_words",
            "target_revision_words",
        ):
            _require_words(getattr(state, label), label=label)
        _require_bool_scalar(state.pending_valid, label="pending_valid")
        _require_threefry_key(state.rng_key, label="state.rng_key")
        return feature_dim

    def state_valid(self, state: NonlinearOffPolicyActorCriticState) -> Bool[Array, ""]:
        self._require_state_contract(state)
        return _dynamic_state_valid(state, n_actions=self._config.n_actions)

    def target_policy(
        self,
        state: NonlinearOffPolicyActorCriticState,
        observation: Array,
    ) -> Float[Array, " action"]:
        feature_dim = self._require_state_contract(state)
        observation = _require_float32_vector(observation, feature_dim, label="observation")
        return _forward(state, observation)[2]

    def _cache_executed_action(
        self,
        state: NonlinearOffPolicyActorCriticState,
        observation: Array,
        action: Array,
        behavior_log_probability: Array,
        behavior_revision_words: Array,
        *,
        replacement_rng_key: Array,
        additional_source_valid: Array,
    ) -> OffPolicyActionCacheResult:
        feature_dim = self._require_state_contract(state)
        observation = _require_float32_vector(observation, feature_dim, label="observation")
        action = _require_int32_scalar(action, label="action")
        behavior_log_probability = _require_float32_scalar(
            behavior_log_probability,
            label="behavior_log_probability",
        )
        behavior_revision_words = _require_words(
            behavior_revision_words,
            label="behavior_revision_words",
        )
        _require_threefry_key(replacement_rng_key, label="replacement_rng_key")
        additional_source_valid = _require_bool_scalar(
            additional_source_valid,
            label="additional_source_valid",
        )

        state_valid = _dynamic_state_valid(state, n_actions=self._config.n_actions)
        action_valid = (action >= 0) & (action < self._config.n_actions)
        safe_action = jnp.clip(action, 0, self._config.n_actions - 1)
        _, _, target_policy, target_log_policy, _ = _forward(state, observation)
        target_log_probability = target_log_policy[safe_action]
        behavior_support_valid = (
            jnp.isfinite(behavior_log_probability)
            & (behavior_log_probability <= 0.0)
            & additional_source_valid
        )
        source_valid = (
            jnp.all(jnp.isfinite(observation))
            & action_valid
            & behavior_support_valid
            & jnp.all(jnp.isfinite(target_policy))
            & jnp.isfinite(target_log_probability)
        )
        cache_available = ~state.pending_valid
        proposed_decision_words, lifetime_capacity = _increment_words(state.decision_words)
        cache_applied = (
            state_valid & source_valid & cache_available & lifetime_capacity
        )
        candidate = dataclasses.replace(  # type: ignore[type-var]
            state,
            pending_observation=observation,
            pending_action=action,
            pending_target_log_probability=target_log_probability,
            pending_behavior_log_probability=behavior_log_probability,
            pending_target_revision_words=state.target_revision_words,
            pending_behavior_revision_words=behavior_revision_words,
            pending_action_identity_words=proposed_decision_words,
            pending_valid=jnp.asarray(True, dtype=jnp.bool_),
            rng_key=replacement_rng_key,
            decision_words=proposed_decision_words,
        )
        next_state = jax.lax.cond(cache_applied, lambda _: candidate, lambda _: state, operand=None)
        record = _record_from_state(candidate)
        return OffPolicyActionCacheResult(  # type: ignore[call-arg]
            state=next_state,
            record=record,
            target_policy=target_policy,
            pre_decision_words=state.decision_words,
            post_decision_words=next_state.decision_words,
            state_valid=state_valid,
            source_valid=source_valid,
            behavior_support_valid=behavior_support_valid,
            cache_available=cache_available,
            lifetime_capacity_available=lifetime_capacity,
            cache_applied=cache_applied,
        )

    def cache_executed_action(
        self,
        state: NonlinearOffPolicyActorCriticState,
        observation: Array,
        action: Array,
        behavior_log_probability: Array,
        behavior_revision_words: Array,
    ) -> OffPolicyActionCacheResult:
        """Bind an externally executed action and its exact behavior semantics."""

        return self._cache_executed_action(
            state,
            observation,
            action,
            behavior_log_probability,
            behavior_revision_words,
            replacement_rng_key=state.rng_key,
            additional_source_valid=jnp.asarray(True, dtype=jnp.bool_),
        )

    def sample_behavior_action(
        self,
        state: NonlinearOffPolicyActorCriticState,
        observation: Array,
        behavior_probabilities: Array,
        behavior_revision_words: Array,
    ) -> OffPolicyActionCacheResult:
        """Sample, then bind, one action from an explicit behavior policy."""

        feature_dim = self._require_state_contract(state)
        observation = _require_float32_vector(observation, feature_dim, label="observation")
        behavior_probabilities = _require_float32_vector(
            behavior_probabilities,
            self._config.n_actions,
            label="behavior_probabilities",
        )
        behavior_revision_words = _require_words(
            behavior_revision_words,
            label="behavior_revision_words",
        )
        finite_nonnegative = jnp.all(
            jnp.isfinite(behavior_probabilities) & (behavior_probabilities >= 0.0)
        )
        normalized = jnp.isclose(
            jnp.sum(behavior_probabilities),
            jnp.asarray(1.0, dtype=jnp.float32),
            rtol=1e-6,
            atol=1e-6,
        )
        policy_valid = finite_nonnegative & normalized & jnp.any(behavior_probabilities > 0.0)
        safe_probabilities = jnp.where(
            policy_valid,
            behavior_probabilities,
            jnp.full_like(behavior_probabilities, 1.0 / self._config.n_actions),
        )
        next_key, sample_key = jr.split(state.rng_key)
        action = jr.categorical(sample_key, jnp.log(safe_probabilities)).astype(jnp.int32)
        behavior_log_probability = jnp.log(safe_probabilities[action])
        return self._cache_executed_action(
            state,
            observation,
            action,
            behavior_log_probability,
            behavior_revision_words,
            replacement_rng_key=next_key,
            additional_source_valid=policy_valid,
        )

    def _require_record_contract(self, record: OffPolicyActionRecord, feature_dim: int) -> None:
        if type(record) is not OffPolicyActionRecord:
            raise TypeError("record must be an exact OffPolicyActionRecord")
        _require_float32_vector(record.observation, feature_dim, label="record.observation")
        _require_int32_scalar(record.action, label="record.action")
        _require_float32_scalar(
            record.target_log_probability,
            label="record.target_log_probability",
        )
        _require_float32_scalar(
            record.behavior_log_probability,
            label="record.behavior_log_probability",
        )
        _require_words(record.target_revision_words, label="record.target_revision_words")
        _require_words(record.behavior_revision_words, label="record.behavior_revision_words")
        _require_words(record.action_identity_words, label="record.action_identity_words")

    def update(
        self,
        state: NonlinearOffPolicyActorCriticState,
        record: OffPolicyActionRecord,
        reward: Array,
        discount: Array,
        next_observation: Array,
    ) -> NonlinearOffPolicyActorCriticUpdateResult:
        """Consume one exact cached action and atomically apply a corrected update."""

        feature_dim = self._require_state_contract(state)
        self._require_record_contract(record, feature_dim)
        reward = _require_float32_scalar(reward, label="reward")
        discount = _require_float32_scalar(discount, label="discount")
        next_observation = _require_float32_vector(
            next_observation,
            feature_dim,
            label="next_observation",
        )

        state_valid = _dynamic_state_valid(state, n_actions=self._config.n_actions)
        record_identity_valid = (
            state.pending_valid
            & _float_bits_equal(record.observation, state.pending_observation)
            & (record.action == state.pending_action)
            & _float_bits_equal(
                record.target_log_probability,
                state.pending_target_log_probability,
            )
            & _float_bits_equal(
                record.behavior_log_probability,
                state.pending_behavior_log_probability,
            )
            & jnp.all(record.target_revision_words == state.pending_target_revision_words)
            & jnp.all(record.behavior_revision_words == state.pending_behavior_revision_words)
            & jnp.all(record.action_identity_words == state.pending_action_identity_words)
        )
        target_revision_valid = (
            jnp.all(state.pending_target_revision_words == state.target_revision_words)
            & jnp.all(record.target_revision_words == state.target_revision_words)
        )
        behavior_support_valid = (
            jnp.isfinite(state.pending_behavior_log_probability)
            & (state.pending_behavior_log_probability <= 0.0)
        )
        transition_valid = (
            jnp.isfinite(reward)
            & jnp.isfinite(discount)
            & (discount >= 0.0)
            & (discount <= 1.0)
            & jnp.all(jnp.isfinite(next_observation))
        )

        hidden, _, policy, target_log_policy, value = _forward(
            state, state.pending_observation
        )
        next_hidden, _, _, _, next_value = _forward(state, next_observation)
        del next_hidden
        safe_action = jnp.clip(state.pending_action, 0, self._config.n_actions - 1)
        current_target_log_probability = target_log_policy[safe_action]
        target_log_probability_valid = _float_bits_equal(
            current_target_log_probability,
            state.pending_target_log_probability,
        )
        log_ratio = (
            state.pending_target_log_probability - state.pending_behavior_log_probability
        )
        max_log_ratio = jnp.log(
            jnp.asarray(jnp.finfo(jnp.float32).max, dtype=jnp.float32)  # type: ignore[no-untyped-call]
        )
        ratio_domain_valid = jnp.isfinite(log_ratio) & (log_ratio <= max_log_ratio)
        raw_ratio_candidate = jnp.exp(log_ratio)
        ratio_valid = (
            ratio_domain_valid
            & jnp.isfinite(raw_ratio_candidate)
            & (raw_ratio_candidate > 0.0)
        )
        raw_ratio = jnp.where(ratio_valid, raw_ratio_candidate, jnp.asarray(0.0, jnp.float32))
        clipped_ratio = jnp.minimum(
            raw_ratio,
            jnp.asarray(self._config.importance_clip, dtype=jnp.float32),
        )
        ratio_truncation = raw_ratio - clipped_ratio

        proposed_update_words, update_capacity = _increment_words(state.update_words)
        target_changes = (
            self._config.actor_plasticity == "plastic"
            or self._config.trunk_plasticity == "plastic"
        )
        if target_changes:
            proposed_target_revision_words, target_capacity = _increment_words(
                state.target_revision_words
            )
        else:
            proposed_target_revision_words = state.target_revision_words
            target_capacity = jnp.asarray(True, dtype=jnp.bool_)
        lifetime_capacity = update_capacity & target_capacity
        source_valid = (
            transition_valid
            & behavior_support_valid
            & record_identity_valid
            & target_revision_valid
            & target_log_probability_valid
            & ratio_valid
        )

        td_error = reward + discount * next_value - value
        action_indicator = jax.nn.one_hot(
            safe_action,
            self._config.n_actions,
            dtype=jnp.float32,
        )
        logit_score = action_indicator - policy
        actor_head_grad_w = jnp.outer(logit_score, hidden)
        actor_head_grad_b = logit_score
        actor_hidden_score = state.actor_w.T @ logit_score
        actor_pre_activation_score = actor_hidden_score * (1.0 - jnp.square(hidden))
        actor_trunk_grad_w = jnp.outer(actor_pre_activation_score, state.pending_observation)
        actor_trunk_grad_b = actor_pre_activation_score
        critic_head_grad_w = hidden
        critic_head_grad_b = jnp.asarray(1.0, dtype=jnp.float32)
        critic_pre_activation_score = state.critic_w * (1.0 - jnp.square(hidden))
        critic_trunk_grad_w = jnp.outer(critic_pre_activation_score, state.pending_observation)
        critic_trunk_grad_b = critic_pre_activation_score

        actor_trace_factor = discount * jnp.asarray(
            self._config.actor_trace_decay,
            dtype=jnp.float32,
        )
        critic_trace_factor = discount * jnp.asarray(
            self._config.critic_trace_decay,
            dtype=jnp.float32,
        )
        candidate_actor_head_trace_w = clipped_ratio * (
            actor_trace_factor * state.actor_head_trace_w + actor_head_grad_w
        )
        candidate_actor_head_trace_b = clipped_ratio * (
            actor_trace_factor * state.actor_head_trace_b + actor_head_grad_b
        )
        candidate_actor_trunk_trace_w = clipped_ratio * (
            actor_trace_factor * state.actor_trunk_trace_w + actor_trunk_grad_w
        )
        candidate_actor_trunk_trace_b = clipped_ratio * (
            actor_trace_factor * state.actor_trunk_trace_b + actor_trunk_grad_b
        )
        candidate_critic_head_trace_w = clipped_ratio * (
            critic_trace_factor * state.critic_head_trace_w + critic_head_grad_w
        )
        candidate_critic_head_trace_b = clipped_ratio * (
            critic_trace_factor * state.critic_head_trace_b + critic_head_grad_b
        )
        candidate_critic_trunk_trace_w = clipped_ratio * (
            critic_trace_factor * state.critic_trunk_trace_w + critic_trunk_grad_w
        )
        candidate_critic_trunk_trace_b = clipped_ratio * (
            critic_trace_factor * state.critic_trunk_trace_b + critic_trunk_grad_b
        )

        actor_plastic = self._config.actor_plasticity == "plastic"
        critic_plastic = self._config.critic_plasticity == "plastic"
        trunk_plastic = self._config.trunk_plasticity == "plastic"
        actor_head_trace_w = (
            candidate_actor_head_trace_w if actor_plastic else state.actor_head_trace_w
        )
        actor_head_trace_b = (
            candidate_actor_head_trace_b if actor_plastic else state.actor_head_trace_b
        )
        critic_head_trace_w = (
            candidate_critic_head_trace_w if critic_plastic else state.critic_head_trace_w
        )
        critic_head_trace_b = (
            candidate_critic_head_trace_b if critic_plastic else state.critic_head_trace_b
        )
        actor_trunk_trace_w = (
            candidate_actor_trunk_trace_w
            if actor_plastic and trunk_plastic
            else state.actor_trunk_trace_w
        )
        actor_trunk_trace_b = (
            candidate_actor_trunk_trace_b
            if actor_plastic and trunk_plastic
            else state.actor_trunk_trace_b
        )
        critic_trunk_trace_w = (
            candidate_critic_trunk_trace_w
            if critic_plastic and trunk_plastic
            else state.critic_trunk_trace_w
        )
        critic_trunk_trace_b = (
            candidate_critic_trunk_trace_b
            if critic_plastic and trunk_plastic
            else state.critic_trunk_trace_b
        )

        momentum = jnp.asarray(self._config.momentum, dtype=jnp.float32)
        actor_velocity_w_candidate = (
            momentum * state.actor_head_velocity_w
            + jnp.asarray(self._config.actor_step_size, dtype=jnp.float32)
            * td_error
            * actor_head_trace_w
        )
        actor_velocity_b_candidate = (
            momentum * state.actor_head_velocity_b
            + jnp.asarray(self._config.actor_step_size, dtype=jnp.float32)
            * td_error
            * actor_head_trace_b
        )
        critic_velocity_w_candidate = (
            momentum * state.critic_head_velocity_w
            + jnp.asarray(self._config.critic_step_size, dtype=jnp.float32)
            * td_error
            * critic_head_trace_w
        )
        critic_velocity_b_candidate = (
            momentum * state.critic_head_velocity_b
            + jnp.asarray(self._config.critic_step_size, dtype=jnp.float32)
            * td_error
            * critic_head_trace_b
        )
        trunk_signal_w = td_error * (
            jnp.asarray(self._config.trunk_actor_step_size, dtype=jnp.float32)
            * actor_trunk_trace_w
            + jnp.asarray(self._config.trunk_critic_step_size, dtype=jnp.float32)
            * critic_trunk_trace_w
        )
        trunk_signal_b = td_error * (
            jnp.asarray(self._config.trunk_actor_step_size, dtype=jnp.float32)
            * actor_trunk_trace_b
            + jnp.asarray(self._config.trunk_critic_step_size, dtype=jnp.float32)
            * critic_trunk_trace_b
        )
        trunk_velocity_w_candidate = momentum * state.trunk_velocity_w + trunk_signal_w
        trunk_velocity_b_candidate = momentum * state.trunk_velocity_b + trunk_signal_b

        actor_velocity_w = (
            actor_velocity_w_candidate if actor_plastic else state.actor_head_velocity_w
        )
        actor_velocity_b = (
            actor_velocity_b_candidate if actor_plastic else state.actor_head_velocity_b
        )
        critic_velocity_w = (
            critic_velocity_w_candidate if critic_plastic else state.critic_head_velocity_w
        )
        critic_velocity_b = (
            critic_velocity_b_candidate if critic_plastic else state.critic_head_velocity_b
        )
        trunk_velocity_w = (
            trunk_velocity_w_candidate if trunk_plastic else state.trunk_velocity_w
        )
        trunk_velocity_b = (
            trunk_velocity_b_candidate if trunk_plastic else state.trunk_velocity_b
        )

        candidate = dataclasses.replace(  # type: ignore[type-var]
            state,
            trunk_w=state.trunk_w + trunk_velocity_w if trunk_plastic else state.trunk_w,
            trunk_b=state.trunk_b + trunk_velocity_b if trunk_plastic else state.trunk_b,
            actor_w=state.actor_w + actor_velocity_w if actor_plastic else state.actor_w,
            actor_b=state.actor_b + actor_velocity_b if actor_plastic else state.actor_b,
            critic_w=(
                state.critic_w + critic_velocity_w if critic_plastic else state.critic_w
            ),
            critic_b=(
                state.critic_b + critic_velocity_b if critic_plastic else state.critic_b
            ),
            actor_head_trace_w=actor_head_trace_w,
            actor_head_trace_b=actor_head_trace_b,
            critic_head_trace_w=critic_head_trace_w,
            critic_head_trace_b=critic_head_trace_b,
            actor_trunk_trace_w=actor_trunk_trace_w,
            actor_trunk_trace_b=actor_trunk_trace_b,
            critic_trunk_trace_w=critic_trunk_trace_w,
            critic_trunk_trace_b=critic_trunk_trace_b,
            trunk_velocity_w=trunk_velocity_w,
            trunk_velocity_b=trunk_velocity_b,
            actor_head_velocity_w=actor_velocity_w,
            actor_head_velocity_b=actor_velocity_b,
            critic_head_velocity_w=critic_velocity_w,
            critic_head_velocity_b=critic_velocity_b,
            update_words=proposed_update_words,
            target_revision_words=proposed_target_revision_words,
        )
        candidate = _empty_pending(candidate)
        candidate_state_valid = _dynamic_state_valid(
            candidate,
            n_actions=self._config.n_actions,
        )
        update_applied = (
            state_valid
            & source_valid
            & lifetime_capacity
            & candidate_state_valid
        )
        next_state = jax.lax.cond(
            update_applied,
            lambda _: candidate,
            lambda _: state,
            operand=None,
        )
        diagnostics_valid = state_valid & transition_valid & state.pending_valid
        diagnostic_value = jnp.where(diagnostics_valid, value, jnp.float32(0.0))
        diagnostic_next_value = jnp.where(
            diagnostics_valid,
            next_value,
            jnp.float32(0.0),
        )
        diagnostic_td_error = jnp.where(
            diagnostics_valid,
            td_error,
            jnp.float32(0.0),
        )
        return NonlinearOffPolicyActorCriticUpdateResult(  # type: ignore[call-arg]
            state=next_state,
            value=diagnostic_value,
            next_value=diagnostic_next_value,
            td_error=diagnostic_td_error,
            importance_ratio=raw_ratio,
            clipped_importance_ratio=clipped_ratio,
            ratio_truncation=ratio_truncation,
            ratio_was_clipped=ratio_truncation > 0.0,
            pre_update_words=state.update_words,
            post_update_words=next_state.update_words,
            pre_target_revision_words=state.target_revision_words,
            post_target_revision_words=next_state.target_revision_words,
            state_valid=state_valid,
            source_valid=source_valid,
            behavior_support_valid=behavior_support_valid,
            record_identity_valid=record_identity_valid,
            target_revision_valid=target_revision_valid,
            target_log_probability_valid=target_log_probability_valid,
            ratio_valid=ratio_valid,
            lifetime_capacity_available=lifetime_capacity,
            candidate_state_valid=candidate_state_valid,
            update_applied=update_applied,
        )

    def resource_budget(
        self,
        state: NonlinearOffPolicyActorCriticState,
    ) -> NonlinearOffPolicyActorCriticResourceBudget:
        """Partition every persistent byte exactly once by component role."""

        self._require_state_contract(state)
        parameters = (
            state.trunk_w,
            state.trunk_b,
            state.actor_w,
            state.actor_b,
            state.critic_w,
            state.critic_b,
        )
        traces = (
            state.actor_head_trace_w,
            state.actor_head_trace_b,
            state.critic_head_trace_w,
            state.critic_head_trace_b,
            state.actor_trunk_trace_w,
            state.actor_trunk_trace_b,
            state.critic_trunk_trace_w,
            state.critic_trunk_trace_b,
        )
        optimizers = (
            state.trunk_velocity_w,
            state.trunk_velocity_b,
            state.actor_head_velocity_w,
            state.actor_head_velocity_b,
            state.critic_head_velocity_w,
            state.critic_head_velocity_b,
        )
        pending = (
            state.pending_observation,
            state.pending_action,
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
        parameter_nbytes = sum(_array_nbytes(value) for value in parameters)
        trace_nbytes = sum(_array_nbytes(value) for value in traces)
        optimizer_nbytes = sum(_array_nbytes(value) for value in optimizers)
        pending_cache_nbytes = sum(_array_nbytes(value) for value in pending)
        clock_nbytes = sum(_array_nbytes(value) for value in clocks)
        rng_nbytes = _array_nbytes(state.rng_key)
        total = (
            parameter_nbytes
            + trace_nbytes
            + optimizer_nbytes
            + pending_cache_nbytes
            + clock_nbytes
            + rng_nbytes
        )
        measured = measure_nonlinear_off_policy_actor_critic_state_nbytes(state)
        if total != measured:
            raise AssertionError("off-policy actor-critic resource partition is incomplete")
        return NonlinearOffPolicyActorCriticResourceBudget(
            schema=NONLINEAR_OFF_POLICY_ACTOR_CRITIC_RESOURCE_SCHEMA,
            persistent_bytes_scope=(
                "all-persistent-state-array-leaves; excludes-host-object-overhead,"
                "temporaries,compiler-and-xla-workspaces; not-a-measured-device-peak"
            ),
            parameter_nbytes=parameter_nbytes,
            trace_nbytes=trace_nbytes,
            optimizer_nbytes=optimizer_nbytes,
            pending_cache_nbytes=pending_cache_nbytes,
            clock_nbytes=clock_nbytes,
            rng_nbytes=rng_nbytes,
            total_state_nbytes=total,
        )

    def save_checkpoint(
        self,
        state: NonlinearOffPolicyActorCriticState,
        path: str | Path,
    ) -> None:
        """Save state with strict construction, schema, and byte metadata."""

        feature_dim = self._require_state_contract(state)
        if not bool(jax.device_get(self.state_valid(state))):
            raise ValueError("cannot checkpoint invalid off-policy actor-critic state")
        _save_checkpoint(
            state,
            path,
            metadata={
                "schema": NONLINEAR_OFF_POLICY_ACTOR_CRITIC_CHECKPOINT_SCHEMA,
                "state_schema": NONLINEAR_OFF_POLICY_ACTOR_CRITIC_STATE_SCHEMA,
                "construction": self.to_config(),
                "feature_dim": feature_dim,
                "state_nbytes": measure_nonlinear_off_policy_actor_critic_state_nbytes(state),
            },
        )

    def checkpoint_metadata(self, path: str | Path) -> dict[str, Any]:
        """Load and validate the exact checkpoint metadata manifest."""

        metadata = _load_checkpoint_metadata(path)
        expected = {"schema", "state_schema", "construction", "feature_dim", "state_nbytes"}
        fields = _exact_manifest(metadata, expected, label="off-policy actor-critic checkpoint")
        if fields["schema"] != NONLINEAR_OFF_POLICY_ACTOR_CRITIC_CHECKPOINT_SCHEMA:
            raise ValueError("off-policy actor-critic checkpoint schema is unsupported")
        if fields["state_schema"] != NONLINEAR_OFF_POLICY_ACTOR_CRITIC_STATE_SCHEMA:
            raise ValueError("off-policy actor-critic state schema is unsupported")
        if fields["construction"] != self.to_config():
            raise ValueError("off-policy actor-critic checkpoint construction is incompatible")
        _exact_int(fields["feature_dim"], label="checkpoint feature_dim", minimum=1)
        _exact_int(fields["state_nbytes"], label="checkpoint state_nbytes", minimum=1)
        return fields

    def load_checkpoint(
        self,
        state_template: NonlinearOffPolicyActorCriticState,
        path: str | Path,
    ) -> NonlinearOffPolicyActorCriticState:
        """Restore only an exact, construction-compatible, valid state."""

        template_feature_dim = self._require_state_contract(state_template)
        metadata = self.checkpoint_metadata(path)
        if metadata["feature_dim"] != template_feature_dim:
            raise ValueError("checkpoint feature dimension does not match the template")
        loaded_raw, restored_metadata = _load_checkpoint(state_template, path)
        loaded = cast(NonlinearOffPolicyActorCriticState, loaded_raw)
        if restored_metadata != metadata:
            raise ValueError("checkpoint metadata changed between validation and restore")
        self._require_state_contract(loaded)
        if not bool(jax.device_get(self.state_valid(loaded))):
            raise ValueError("restored off-policy actor-critic state is invalid")
        if (
            measure_nonlinear_off_policy_actor_critic_state_nbytes(loaded)
            != metadata["state_nbytes"]
        ):
            raise ValueError("restored off-policy actor-critic resource size is invalid")
        return loaded


def run_nonlinear_off_policy_actor_critic_from_arrays(
    agent: NonlinearOffPolicyActorCritic,
    state: NonlinearOffPolicyActorCriticState,
    observations: Array,
    actions: Array,
    behavior_log_probabilities: Array,
    behavior_revisions: Array,
    rewards: Array,
    discounts: Array,
    next_observations: Array,
) -> NonlinearOffPolicyActorCriticScanResult:
    """Run exact cache-then-update transactions with ``jax.lax.scan``."""

    if type(agent) is not NonlinearOffPolicyActorCritic:
        raise TypeError("agent must be an exact NonlinearOffPolicyActorCritic")
    feature_dim = agent._require_state_contract(state)
    if getattr(observations, "ndim", None) != 2:
        raise ValueError("observations must have rank 2")
    steps = observations.shape[0]
    contracts = (
        (observations, (steps, feature_dim), jnp.dtype(jnp.float32), "observations"),
        (actions, (steps,), jnp.dtype(jnp.int32), "actions"),
        (
            behavior_log_probabilities,
            (steps,),
            jnp.dtype(jnp.float32),
            "behavior_log_probabilities",
        ),
        (
            behavior_revisions,
            (steps, 2),
            jnp.dtype(jnp.uint32),
            "behavior_revisions",
        ),
        (rewards, (steps,), jnp.dtype(jnp.float32), "rewards"),
        (discounts, (steps,), jnp.dtype(jnp.float32), "discounts"),
        (
            next_observations,
            (steps, feature_dim),
            jnp.dtype(jnp.float32),
            "next_observations",
        ),
    )
    for value, shape, dtype, label in contracts:
        _require_array(value, label=label, shape=shape, dtype=dtype)

    def body(
        carry: NonlinearOffPolicyActorCriticState,
        sources: tuple[Array, Array, Array, Array, Array, Array, Array],
    ) -> tuple[NonlinearOffPolicyActorCriticState, tuple[Array, ...]]:
        observation, action, behavior_logp, behavior_revision, reward, discount, next_obs = (
            sources
        )
        cache = agent.cache_executed_action(
            carry,
            observation,
            action,
            behavior_logp,
            behavior_revision,
        )
        update = agent.update(
            cache.state,
            cache.record,
            reward,
            discount,
            next_obs,
        )
        return update.state, (
            update.td_error,
            update.importance_ratio,
            update.clipped_importance_ratio,
            update.ratio_truncation,
            cache.cache_applied,
            update.update_applied,
        )

    final_state, outputs = jax.lax.scan(
        body,
        state,
        (
            observations,
            actions,
            behavior_log_probabilities,
            behavior_revisions,
            rewards,
            discounts,
            next_observations,
        ),
    )
    td_errors, ratios, clipped, truncations, cache_applied, update_applied = outputs
    return NonlinearOffPolicyActorCriticScanResult(  # type: ignore[call-arg]
        state=final_state,
        td_errors=td_errors,
        importance_ratios=ratios,
        clipped_importance_ratios=clipped,
        ratio_truncations=truncations,
        cache_applied=cache_applied,
        update_applied=update_applied,
    )


__all__ = [
    "NONLINEAR_OFF_POLICY_ACTOR_CRITIC_CHECKPOINT_SCHEMA",
    "NONLINEAR_OFF_POLICY_ACTOR_CRITIC_CONFIG_SCHEMA",
    "NONLINEAR_OFF_POLICY_ACTOR_CRITIC_EVIDENCE_LEVEL",
    "NONLINEAR_OFF_POLICY_ACTOR_CRITIC_LIFETIME_SEMANTICS",
    "NONLINEAR_OFF_POLICY_ACTOR_CRITIC_MAX_DECISIONS",
    "NONLINEAR_OFF_POLICY_ACTOR_CRITIC_MAX_UPDATES",
    "NONLINEAR_OFF_POLICY_ACTOR_CRITIC_OUTCOME_STATUS",
    "NONLINEAR_OFF_POLICY_ACTOR_CRITIC_RESOURCE_SCHEMA",
    "NONLINEAR_OFF_POLICY_ACTOR_CRITIC_STATE_SCHEMA",
    "NonlinearOffPolicyActorCritic",
    "NonlinearOffPolicyActorCriticConfig",
    "NonlinearOffPolicyActorCriticResourceBudget",
    "NonlinearOffPolicyActorCriticScanResult",
    "NonlinearOffPolicyActorCriticState",
    "NonlinearOffPolicyActorCriticUpdateResult",
    "OffPolicyActionCacheResult",
    "OffPolicyActionRecord",
    "PlasticityPolicy",
    "measure_nonlinear_off_policy_actor_critic_state_nbytes",
    "run_nonlinear_off_policy_actor_critic_from_arrays",
]
