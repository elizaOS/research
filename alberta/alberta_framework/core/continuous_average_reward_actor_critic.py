# mypy: disable-error-code="attr-defined,call-arg,no-untyped-call"
"""Bounded continuous average-reward actor-critic (isolated L0 core).

This module is deliberately separate from :mod:`actor_critic`'s discounted,
clipped-Gaussian continuous preview.  It implements a continuing differential
actor-critic with a reward-rate baseline and a bounded, tanh-squashed diagonal
Gaussian policy.  The action transform is bijective in real arithmetic; raw
Gaussian samples are never clipped.

The behavior policy has the target mean and a configurable multiplicative
standard-deviation scale.  Every cached decision records both transformed log
densities (including the affine-tanh Jacobian) and their exact finite-precision
ratio.  That likelihood-ratio correction does **not** correct behavior-policy
state-distribution mismatch.

The first implementation intentionally uses distinct fixed-step LMS optimizer
states for every actor, critic, and reward-rate parameter group.  It is an L0
mechanism contract, not efficacy evidence and not an off-policy convergence
claim.
"""

from __future__ import annotations

import dataclasses
import functools
import math
import numbers
from collections.abc import Mapping, Sequence
from typing import Any, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array
from jaxtyping import Bool, Float, Int

from alberta_framework.core.optimizers import LMS
from alberta_framework.core.types import LMSState

_CONFIG_SCHEMA = "continuous_average_reward_actor_critic.config.v1"
_CHECKPOINT_SCHEMA = "continuous_average_reward_actor_critic.checkpoint.v1"
_INT32_MAX = 2**31 - 1
_LOG_TWO_PI = math.log(2.0 * math.pi)
# XLA reassociation reached eight float32 ULPs across a 500-seed CPU stress
# probe.  The explicit arithmetic barriers below reduce common-path drift to
# zero, while this narrow symmetric bound remains the fail-closed backend
# allowance for the two diagnostic transformed densities.  Policy-defining
# fields (latent draw, transform, policy parameters, and latent likelihood
# ratio) remain exact-bit owners.
TRANSFORMED_LOG_DENSITY_MAX_ULPS = 8
_FLOAT32_SAFE_LOG_STD_MIN = float(
    np.nextafter(
        np.float32(math.log(float(np.finfo(np.float32).tiny))),
        np.float32(np.inf),
    )
)
_FLOAT32_SAFE_LOG_STD_MAX = float(
    np.nextafter(
        np.float32(math.log(float(np.finfo(np.float32).max))),
        np.float32(-np.inf),
    )
)


def _saturating_int32_increment(value: Array) -> Array:
    counter = jnp.asarray(value, dtype=jnp.int32)
    maximum = jnp.asarray(_INT32_MAX, dtype=jnp.int32)
    return jnp.where(counter >= maximum, maximum, jnp.maximum(counter, 0) + 1)


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be a positive integer")
    parsed = int(value)
    if parsed < 1 or parsed > _INT32_MAX:
        raise ValueError(f"{name} must be in [1, {_INT32_MAX}]")
    return parsed


def _finite_float(
    value: object,
    *,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a finite real number")
    parsed = float(value)
    if not math.isfinite(parsed) or abs(parsed) > float(np.finfo(np.float32).max):
        raise ValueError(f"{name} must be finite in float32")
    if minimum is not None and parsed < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and parsed > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return parsed


def _normalise_bound(value: object, *, action_dim: int, name: str) -> tuple[float, ...]:
    if isinstance(value, bool):
        raise ValueError(f"{name} must contain finite real numbers")
    if isinstance(value, numbers.Real):
        item = float(np.float32(_finite_float(value, name=name)))
        return (item,) * action_dim
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a real scalar or length-action_dim sequence")
    if len(value) != action_dim:
        raise ValueError(f"{name} must have length action_dim")
    return tuple(
        float(np.float32(_finite_float(item, name=f"{name}[{index}]")))
        for index, item in enumerate(value)
    )


@dataclasses.dataclass(frozen=True)
class ContinuousAverageRewardActorCriticConfig:
    """Static contract for the bounded differential actor-critic.

    ``behavior_std_scale`` broadens only the Gaussian used to draw the cached
    pre-tanh decision.  The actor uses the exact target/behavior action-density
    ratio.  This is not a state-distribution correction.
    """

    action_dim: int
    action_low: float | tuple[float, ...] = -1.0
    action_high: float | tuple[float, ...] = 1.0
    actor_step_size: float = 0.001
    critic_step_size: float = 0.05
    average_reward_step_size: float = 0.01
    actor_trace_lambda: float = 0.0
    critic_trace_lambda: float = 0.0
    target_log_std_init: float = -0.5
    target_log_std_min: float = -5.0
    target_log_std_max: float = 1.0
    behavior_std_scale: float = 1.0
    max_updates: int = _INT32_MAX

    def __post_init__(self) -> None:
        """Normalise action bounds and reject unsafe finite-precision controls."""
        action_dim = _positive_int(self.action_dim, name="action_dim")
        max_updates = _positive_int(self.max_updates, name="max_updates")
        low = _normalise_bound(self.action_low, action_dim=action_dim, name="action_low")
        high = _normalise_bound(self.action_high, action_dim=action_dim, name="action_high")
        if any(left >= right for left, right in zip(low, high, strict=True)):
            raise ValueError("every action_low entry must be below action_high")
        low_array = np.asarray(low, dtype=np.float32)
        high_array = np.asarray(high, dtype=np.float32)
        with np.errstate(over="ignore", invalid="ignore"):
            midpoint = np.float32(0.5) * (high_array + low_array)
            half_range = np.float32(0.5) * (high_array - low_array)
        if not np.all(np.isfinite(midpoint)):
            raise ValueError("action-bound float32 midpoint must be finite")
        if not np.all(np.isfinite(half_range)) or not np.all(half_range > 0.0):
            raise ValueError("action-bound float32 half-range must be finite and positive")
        actor_step_size = _finite_float(self.actor_step_size, name="actor_step_size", minimum=0.0)
        critic_step_size = _finite_float(
            self.critic_step_size, name="critic_step_size", minimum=0.0
        )
        average_reward_step_size = _finite_float(
            self.average_reward_step_size,
            name="average_reward_step_size",
            minimum=0.0,
        )
        actor_trace_lambda = _finite_float(
            self.actor_trace_lambda,
            name="actor_trace_lambda",
            minimum=0.0,
            maximum=1.0,
        )
        critic_trace_lambda = _finite_float(
            self.critic_trace_lambda,
            name="critic_trace_lambda",
            minimum=0.0,
            maximum=1.0,
        )
        log_std_init = _finite_float(self.target_log_std_init, name="target_log_std_init")
        log_std_min = _finite_float(self.target_log_std_min, name="target_log_std_min")
        log_std_max = _finite_float(self.target_log_std_max, name="target_log_std_max")
        if log_std_min > log_std_init or log_std_init > log_std_max:
            raise ValueError(
                "target_log_std_min <= target_log_std_init <= target_log_std_max is required"
            )
        if log_std_min < _FLOAT32_SAFE_LOG_STD_MIN:
            raise ValueError("target_log_std_min must exponentiate positively in JAX float32")
        if log_std_max > _FLOAT32_SAFE_LOG_STD_MAX:
            raise ValueError("target_log_std_max must exponentiate finitely in JAX float32")
        behavior_scale = _finite_float(
            self.behavior_std_scale, name="behavior_std_scale", minimum=1.0
        )
        target_std_max = np.exp(np.float32(log_std_max), dtype=np.float32)
        with np.errstate(over="ignore", invalid="ignore"):
            behavior_std_max = np.float32(target_std_max * np.float32(behavior_scale))
        if not np.isfinite(behavior_std_max) or behavior_std_max <= 0.0:
            raise ValueError("maximum behavior standard deviation must be finite in float32")
        # Because the behavior variance is never narrower than the target, the
        # largest likelihood ratio occurs at the common mean and is scale**d.
        if action_dim * math.log(behavior_scale) > _FLOAT32_SAFE_LOG_STD_MAX:
            raise ValueError("behavior_std_scale ** action_dim must be finite in float32")
        object.__setattr__(self, "action_dim", action_dim)
        object.__setattr__(self, "action_low", low)
        object.__setattr__(self, "action_high", high)
        object.__setattr__(self, "actor_step_size", actor_step_size)
        object.__setattr__(self, "critic_step_size", critic_step_size)
        object.__setattr__(self, "average_reward_step_size", average_reward_step_size)
        object.__setattr__(self, "actor_trace_lambda", actor_trace_lambda)
        object.__setattr__(self, "critic_trace_lambda", critic_trace_lambda)
        object.__setattr__(self, "target_log_std_init", log_std_init)
        object.__setattr__(self, "target_log_std_min", log_std_min)
        object.__setattr__(self, "target_log_std_max", log_std_max)
        object.__setattr__(self, "behavior_std_scale", behavior_scale)
        object.__setattr__(self, "max_updates", max_updates)

    def to_config(self) -> dict[str, Any]:
        """Return a strict, JSON-compatible, versioned configuration."""
        return {
            "schema_version": _CONFIG_SCHEMA,
            "action_dim": self.action_dim,
            "action_low": list(cast(tuple[float, ...], self.action_low)),
            "action_high": list(cast(tuple[float, ...], self.action_high)),
            "actor_step_size": self.actor_step_size,
            "critic_step_size": self.critic_step_size,
            "average_reward_step_size": self.average_reward_step_size,
            "actor_trace_lambda": self.actor_trace_lambda,
            "critic_trace_lambda": self.critic_trace_lambda,
            "target_log_std_init": self.target_log_std_init,
            "target_log_std_min": self.target_log_std_min,
            "target_log_std_max": self.target_log_std_max,
            "behavior_std_scale": self.behavior_std_scale,
            "max_updates": self.max_updates,
        }

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> ContinuousAverageRewardActorCriticConfig:
        """Reconstruct only the exact supported configuration schema."""
        expected = {
            "schema_version",
            "action_dim",
            "action_low",
            "action_high",
            "actor_step_size",
            "critic_step_size",
            "average_reward_step_size",
            "actor_trace_lambda",
            "critic_trace_lambda",
            "target_log_std_init",
            "target_log_std_min",
            "target_log_std_max",
            "behavior_std_scale",
            "max_updates",
        }
        if set(config) != expected:
            raise ValueError(f"config fields must be exactly {sorted(expected)}")
        if config["schema_version"] != _CONFIG_SCHEMA:
            raise ValueError(f"schema_version must be {_CONFIG_SCHEMA!r}")
        values = dict(config)
        values.pop("schema_version")
        return cls(**values)


@chex.dataclass(frozen=True)
class ContinuousAverageRewardActorParameters:
    """Linear target-policy parameters."""

    mean_weights: Float[Array, "action_dim feature_dim"]
    mean_bias: Float[Array, " action_dim"]
    log_std: Float[Array, " action_dim"]


@chex.dataclass(frozen=True)
class ContinuousAverageRewardCriticParameters:
    """Linear differential-value parameters."""

    weights: Float[Array, " feature_dim"]
    bias: Float[Array, ""]


@chex.dataclass(frozen=True)
class ContinuousAverageRewardActorTrace:
    """Accumulating target-score eligibility trace owned by cached decisions."""

    mean_weights: Float[Array, "action_dim feature_dim"]
    mean_bias: Float[Array, " action_dim"]
    log_std: Float[Array, " action_dim"]


@chex.dataclass(frozen=True)
class ContinuousAverageRewardCriticTrace:
    """Accumulating differential-value eligibility trace."""

    weights: Float[Array, " feature_dim"]
    bias: Float[Array, ""]


@chex.dataclass(frozen=True)
class ContinuousAverageRewardActorOptimizerState:
    """Separate fixed-step LMS state for each actor parameter group."""

    mean_weights: LMSState
    mean_bias: LMSState
    log_std: LMSState


@chex.dataclass(frozen=True)
class ContinuousAverageRewardCriticOptimizerState:
    """Separate fixed-step LMS state for each critic parameter group."""

    weights: LMSState
    bias: LMSState


@chex.dataclass(frozen=True)
class SquashedGaussianPolicySample:
    """Exact decision-time target/behavior contract for one action.

    The pre-tanh action is the behavior-Gaussian draw.  Both log densities
    are densities of the transformed action and include the same affine-tanh
    Jacobian.  ``target_behavior_ratio`` is the exact finite-precision
    action-likelihood correction used by the actor.
    """

    observation: Float[Array, " feature_dim"]
    pre_tanh_action: Float[Array, " action_dim"]
    action: Float[Array, " action_dim"]
    target_mean: Float[Array, " action_dim"]
    target_std: Float[Array, " action_dim"]
    behavior_std: Float[Array, " action_dim"]
    target_log_density: Float[Array, ""]
    behavior_log_density: Float[Array, ""]
    target_behavior_ratio: Float[Array, ""]
    valid: Bool[Array, ""]


@chex.dataclass(frozen=True)
class ContinuousAverageRewardActorCriticState:
    """Immutable continuing control state."""

    actor_params: ContinuousAverageRewardActorParameters
    critic_params: ContinuousAverageRewardCriticParameters
    actor_trace: ContinuousAverageRewardActorTrace
    critic_trace: ContinuousAverageRewardCriticTrace
    actor_optimizer_state: ContinuousAverageRewardActorOptimizerState
    critic_optimizer_state: ContinuousAverageRewardCriticOptimizerState
    average_reward_optimizer_state: LMSState
    average_reward: Float[Array, ""]
    last_sample: SquashedGaussianPolicySample
    rng_key: Array
    decision_count: Int[Array, ""]
    update_count: Int[Array, ""]


@chex.dataclass(frozen=True)
class ContinuousAverageRewardActorCriticStartResult:
    """Result of atomically caching the first continuing decision."""

    state: ContinuousAverageRewardActorCriticState
    sample: SquashedGaussianPolicySample
    action: Float[Array, " action_dim"]
    accepted: Bool[Array, ""]


@chex.dataclass(frozen=True)
class ContinuousAverageRewardActorCriticDiagnostics:
    """Fail-closed validity and correction diagnostics for one transition."""

    state_valid: Bool[Array, ""]
    input_valid: Bool[Array, ""]
    cached_decision_valid: Bool[Array, ""]
    capacity_available: Bool[Array, ""]
    candidate_finite: Bool[Array, ""]
    target_behavior_ratio: Float[Array, ""]


@chex.dataclass(frozen=True)
class ContinuousAverageRewardActorCriticUpdateResult:
    """Atomic differential update and its post-commit successor decision."""

    state: ContinuousAverageRewardActorCriticState
    sample: SquashedGaussianPolicySample
    action: Float[Array, " action_dim"]
    value: Float[Array, ""]
    next_value: Float[Array, ""]
    td_error: Float[Array, ""]
    average_reward: Float[Array, ""]
    accepted: Bool[Array, ""]
    diagnostics: ContinuousAverageRewardActorCriticDiagnostics


@dataclasses.dataclass(frozen=True)
class ContinuousAverageRewardActorCriticResourceBudget:
    """Exact fixed-state resource declaration for the isolated L0 core."""

    feature_dim: int
    action_dim: int
    trainable_float32_scalars: int
    state_nbytes: int
    max_updates: int
    optimizer_kind: str = "LMS"
    evidence_level: str = "L0"
    off_policy_state_distribution_correction: bool = False
    replay_capacity: int = 0

    def to_dict(self) -> dict[str, int | str | bool]:
        """Return a JSON-compatible resource declaration."""
        return dataclasses.asdict(self)


def transformed_diagonal_gaussian_log_density(
    pre_tanh_action: Array,
    mean: Array,
    std: Array,
    action_low: Array,
    action_high: Array,
) -> Float[Array, ""]:
    """Return the affine-tanh transformed diagonal-Gaussian log density.

    The stable identity
    ``log(1 - tanh(z)^2) = 2(log(2) - |z| - softplus(-2|z|))``
    avoids taking a logarithm of a rounded ``tanh`` result.  Callers retain the
    exact pre-tanh draw, so no inverse transform or boundary clipping is used.
    """

    latent = jax.lax.optimization_barrier(jnp.asarray(pre_tanh_action, dtype=jnp.float32))
    location = jax.lax.optimization_barrier(jnp.asarray(mean, dtype=jnp.float32))
    scale = jax.lax.optimization_barrier(jnp.asarray(std, dtype=jnp.float32))
    low = jax.lax.optimization_barrier(jnp.asarray(action_low, dtype=jnp.float32))
    high = jax.lax.optimization_barrier(jnp.asarray(action_high, dtype=jnp.float32))
    normal_log_density = jax.lax.optimization_barrier(
        _diagonal_gaussian_log_density(latent, location, scale)
    )
    absolute_latent = jax.lax.optimization_barrier(jnp.abs(latent))
    log_two = jax.lax.optimization_barrier(jnp.log(jnp.asarray(2.0, dtype=jnp.float32)))
    softplus_input = jax.lax.optimization_barrier(
        -jnp.asarray(2.0, dtype=jnp.float32) * absolute_latent
    )
    softplus = jax.lax.optimization_barrier(jax.nn.softplus(softplus_input))
    derivative_inner = jax.lax.optimization_barrier(
        jax.lax.optimization_barrier(log_two - absolute_latent) - softplus
    )
    log_tanh_derivative = jax.lax.optimization_barrier(
        jnp.asarray(2.0, dtype=jnp.float32) * derivative_inner
    )
    affine_scale = jax.lax.optimization_barrier(
        jnp.asarray(0.5, dtype=jnp.float32) * jax.lax.optimization_barrier(high - low)
    )
    log_affine_scale = jax.lax.optimization_barrier(jnp.log(affine_scale))
    log_abs_jacobian = jax.lax.optimization_barrier(
        jnp.sum(jax.lax.optimization_barrier(log_affine_scale + log_tanh_derivative))
    )
    return cast(
        Array,
        jax.lax.optimization_barrier(
            jnp.asarray(normal_log_density - log_abs_jacobian, dtype=jnp.float32)
        ),
    )


def _diagonal_gaussian_log_density(
    sample: Array,
    mean: Array,
    std: Array,
) -> Float[Array, ""]:
    """Return a diagonal-Gaussian latent log density without action Jacobian."""
    latent = jax.lax.optimization_barrier(jnp.asarray(sample, dtype=jnp.float32))
    location = jax.lax.optimization_barrier(jnp.asarray(mean, dtype=jnp.float32))
    scale = jax.lax.optimization_barrier(jnp.asarray(std, dtype=jnp.float32))
    difference = jax.lax.optimization_barrier(latent - location)
    normalized = jax.lax.optimization_barrier(difference / scale)
    square = jax.lax.optimization_barrier(jnp.square(normalized))
    quadratic = jax.lax.optimization_barrier(-jnp.asarray(0.5, dtype=jnp.float32) * square)
    log_scale = jax.lax.optimization_barrier(jnp.log(scale))
    normalizer = jnp.asarray(0.5 * _LOG_TWO_PI, dtype=jnp.float32)
    terms = jax.lax.optimization_barrier(
        jax.lax.optimization_barrier(quadratic - log_scale) - normalizer
    )
    return cast(
        Array,
        jax.lax.optimization_barrier(jnp.asarray(jnp.sum(terms), dtype=jnp.float32)),
    )


def diagonal_gaussian_target_behavior_ratio(
    sample: Array,
    mean: Array,
    target_std: Array,
    behavior_std: Array,
) -> Float[Array, ""]:
    """Return a reproducible latent target/behavior density ratio.

    Optimization barriers make the float32 operation order an explicit cache
    contract across the fused sampling path and later validation paths.  The
    shared affine-tanh Jacobian cancels analytically before exponentiation.
    """
    latent = jax.lax.optimization_barrier(jnp.asarray(sample, dtype=jnp.float32))
    location = jax.lax.optimization_barrier(jnp.asarray(mean, dtype=jnp.float32))
    target_scale = jax.lax.optimization_barrier(jnp.asarray(target_std, dtype=jnp.float32))
    behavior_scale = jax.lax.optimization_barrier(jnp.asarray(behavior_std, dtype=jnp.float32))

    def condition(carry: tuple[Array, Array]) -> Array:
        iteration, _ = carry
        return iteration < jnp.asarray(1, dtype=jnp.int32)

    def compute(carry: tuple[Array, Array]) -> tuple[Array, Array]:
        iteration, _ = carry
        difference = jax.lax.optimization_barrier(latent - location)
        target_normalized = jax.lax.optimization_barrier(difference / target_scale)
        target_square = jax.lax.optimization_barrier(jnp.square(target_normalized))
        scale_ratio = jax.lax.optimization_barrier(behavior_scale / target_scale)
        log_scale_ratio = jax.lax.optimization_barrier(jnp.log(scale_ratio))
        inverse_scale_ratio = jax.lax.optimization_barrier(
            jnp.asarray(1.0, dtype=jnp.float32) / scale_ratio
        )
        inverse_scale_ratio_square = jax.lax.optimization_barrier(jnp.square(inverse_scale_ratio))
        quadratic_scale = jax.lax.optimization_barrier(
            jnp.asarray(1.0, dtype=jnp.float32) - inverse_scale_ratio_square
        )
        quadratic_penalty = jax.lax.optimization_barrier(
            jnp.asarray(0.5, dtype=jnp.float32)
            * jax.lax.optimization_barrier(target_square * quadratic_scale)
        )
        log_ratio_terms = jax.lax.optimization_barrier(log_scale_ratio - quadratic_penalty)
        log_ratio = jax.lax.optimization_barrier(jnp.sum(log_ratio_terms))
        ratio = jax.lax.optimization_barrier(jnp.exp(log_ratio))
        return iteration + jnp.asarray(1, dtype=jnp.int32), ratio

    _, ratio = jax.lax.while_loop(
        condition,
        compute,
        (
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(1.0, dtype=jnp.float32),
        ),
    )
    return ratio


def _squash_action(pre_tanh_action: Array, action_low: Array, action_high: Array) -> Array:
    midpoint = jax.lax.optimization_barrier(0.5 * (action_high + action_low))
    half_range = jax.lax.optimization_barrier(0.5 * (action_high - action_low))
    # In real arithmetic tanh has an open range.  Float32 tanh may round a
    # sufficiently large finite latent to +/-1, so the stored finite-precision
    # action contract is closed-bounded.  Do not post-adjust it: the cached
    # action remains exactly the declared affine-tanh transformation.
    squashed = jax.lax.optimization_barrier(jnp.tanh(pre_tanh_action))
    scaled = jax.lax.optimization_barrier(half_range * squashed)
    return jnp.asarray(midpoint + scaled, dtype=jnp.float32)


def _all_finite(tree: Any) -> Array:
    finite = jnp.asarray(True)
    for leaf in jax.tree_util.tree_leaves(tree):
        array = jnp.asarray(leaf)
        if jnp.issubdtype(array.dtype, jnp.inexact):
            finite = jnp.logical_and(finite, jnp.all(jnp.isfinite(array)))
    return finite


def _float_bits_equal(left: Array, right: Array) -> Array:
    lhs = jnp.asarray(left, dtype=jnp.float32)
    rhs = jnp.asarray(right, dtype=jnp.float32)
    return jnp.all(
        jax.lax.bitcast_convert_type(lhs, jnp.uint32)
        == jax.lax.bitcast_convert_type(rhs, jnp.uint32)
    )


def float32_ulp_distance(left: Array, right: Array) -> Array:
    """Return symmetric elementwise distance in the ordered float32 bit space."""
    lhs = jnp.asarray(left, dtype=jnp.float32)
    rhs = jnp.asarray(right, dtype=jnp.float32)
    sign = jnp.asarray(0x80000000, dtype=jnp.uint32)

    def ordered(value: Array) -> Array:
        bits = jax.lax.bitcast_convert_type(value, jnp.uint32)
        return jnp.where((bits & sign) != 0, jnp.bitwise_not(bits), bits | sign)

    lhs_ordered = ordered(lhs)
    rhs_ordered = ordered(rhs)
    return jnp.maximum(lhs_ordered, rhs_ordered) - jnp.minimum(lhs_ordered, rhs_ordered)


class ContinuousAverageRewardActorCriticAgent:
    """Linear differential actor-critic with a bounded continuous policy.

    The critic learns the behavior-policy differential value with
    ``delta = reward - average_reward + V(next) - V(previous)``.  Actor and
    critic traces are separate.  The current actor score comes exclusively
    from ``state.last_sample``; the successor is drawn only after every
    candidate parameter, trace, optimizer, reward-rate, and counter update is
    finite.  Rejected transitions leave state and RNG bit-exactly unchanged.
    """

    def __init__(self, config: ContinuousAverageRewardActorCriticConfig):
        self._config = config
        self._actor_optimizer = LMS(step_size=config.actor_step_size)
        self._critic_optimizer = LMS(step_size=config.critic_step_size)
        self._average_reward_optimizer = LMS(step_size=config.average_reward_step_size)
        self._action_low = jnp.asarray(config.action_low, dtype=jnp.float32)
        self._action_high = jnp.asarray(config.action_high, dtype=jnp.float32)

    @property
    def config(self) -> ContinuousAverageRewardActorCriticConfig:
        """Static mechanism configuration."""
        return self._config

    def to_config(self) -> dict[str, Any]:
        """Return a strict, versioned, JSON-compatible agent configuration."""
        return {
            "type": "ContinuousAverageRewardActorCriticAgent",
            "config": self._config.to_config(),
        }

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> ContinuousAverageRewardActorCriticAgent:
        """Reconstruct only the exact supported agent schema."""
        if set(config) != {"type", "config"}:
            raise ValueError("agent config fields must be exactly ['config', 'type']")
        if config["type"] != "ContinuousAverageRewardActorCriticAgent":
            raise ValueError("unsupported agent type")
        nested = config["config"]
        if not isinstance(nested, Mapping):
            raise ValueError("config must be a mapping")
        return cls(ContinuousAverageRewardActorCriticConfig.from_config(nested))

    def init(self, feature_dim: int, key: Array) -> ContinuousAverageRewardActorCriticState:
        """Initialize separate parameters, traces, optimizer states, and sentinel cache."""
        feature_dim = _positive_int(feature_dim, name="feature_dim")
        if key.shape != () or not jax.dtypes.issubdtype(key.dtype, jax.dtypes.prng_key):
            raise ValueError("key must be a typed scalar JAX PRNG key")
        action_dim = self._config.action_dim
        actor_params = ContinuousAverageRewardActorParameters(
            mean_weights=jnp.zeros((action_dim, feature_dim), dtype=jnp.float32),
            mean_bias=jnp.zeros((action_dim,), dtype=jnp.float32),
            log_std=jnp.full((action_dim,), self._config.target_log_std_init, dtype=jnp.float32),
        )
        critic_params = ContinuousAverageRewardCriticParameters(
            weights=jnp.zeros((feature_dim,), dtype=jnp.float32),
            bias=jnp.asarray(0.0, dtype=jnp.float32),
        )
        actor_trace = ContinuousAverageRewardActorTrace(
            mean_weights=jnp.zeros_like(actor_params.mean_weights),
            mean_bias=jnp.zeros_like(actor_params.mean_bias),
            log_std=jnp.zeros_like(actor_params.log_std),
        )
        critic_trace = ContinuousAverageRewardCriticTrace(
            weights=jnp.zeros_like(critic_params.weights),
            bias=jnp.asarray(0.0, dtype=jnp.float32),
        )
        actor_optimizer_state = ContinuousAverageRewardActorOptimizerState(
            mean_weights=self._actor_optimizer.init_for_shape(actor_params.mean_weights.shape),
            mean_bias=self._actor_optimizer.init_for_shape(actor_params.mean_bias.shape),
            log_std=self._actor_optimizer.init_for_shape(actor_params.log_std.shape),
        )
        critic_optimizer_state = ContinuousAverageRewardCriticOptimizerState(
            weights=self._critic_optimizer.init_for_shape(critic_params.weights.shape),
            bias=self._critic_optimizer.init_for_shape(critic_params.bias.shape),
        )
        midpoint = 0.5 * (self._action_low + self._action_high)
        sentinel = SquashedGaussianPolicySample(
            observation=jnp.zeros((feature_dim,), dtype=jnp.float32),
            pre_tanh_action=jnp.zeros((action_dim,), dtype=jnp.float32),
            action=midpoint,
            target_mean=jnp.zeros((action_dim,), dtype=jnp.float32),
            target_std=jnp.ones((action_dim,), dtype=jnp.float32),
            behavior_std=jnp.ones((action_dim,), dtype=jnp.float32),
            target_log_density=jnp.asarray(0.0, dtype=jnp.float32),
            behavior_log_density=jnp.asarray(0.0, dtype=jnp.float32),
            target_behavior_ratio=jnp.asarray(1.0, dtype=jnp.float32),
            valid=jnp.asarray(False),
        )
        return ContinuousAverageRewardActorCriticState(
            actor_params=actor_params,
            critic_params=critic_params,
            actor_trace=actor_trace,
            critic_trace=critic_trace,
            actor_optimizer_state=actor_optimizer_state,
            critic_optimizer_state=critic_optimizer_state,
            average_reward_optimizer_state=self._average_reward_optimizer.init_for_shape(()),
            average_reward=jnp.asarray(0.0, dtype=jnp.float32),
            last_sample=sentinel,
            rng_key=key,
            decision_count=jnp.asarray(0, dtype=jnp.int32),
            update_count=jnp.asarray(0, dtype=jnp.int32),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def target_policy_params(
        self,
        state: ContinuousAverageRewardActorCriticState,
        observation: Array,
    ) -> tuple[Float[Array, " action_dim"], Float[Array, " action_dim"]]:
        """Return target pre-tanh Gaussian mean and diagonal standard deviation."""
        obs = jnp.asarray(observation, dtype=jnp.float32)
        mean = state.actor_params.mean_weights @ obs + state.actor_params.mean_bias
        return mean, jnp.exp(state.actor_params.log_std)

    @functools.partial(jax.jit, static_argnums=(0,))
    def behavior_policy_params(
        self,
        state: ContinuousAverageRewardActorCriticState,
        observation: Array,
    ) -> tuple[
        Float[Array, " action_dim"],
        Float[Array, " action_dim"],
        Float[Array, " action_dim"],
    ]:
        """Return the common mean, target std, and explicitly broader behavior std."""
        mean, target_std = self.target_policy_params(state, observation)
        behavior_std = target_std * jnp.asarray(self._config.behavior_std_scale, dtype=jnp.float32)
        return mean, target_std, behavior_std

    @functools.partial(jax.jit, static_argnums=(0,))
    def value(
        self,
        state: ContinuousAverageRewardActorCriticState,
        observation: Array,
    ) -> Float[Array, ""]:
        """Return the linear differential-value estimate."""
        obs = jnp.asarray(observation, dtype=jnp.float32)
        return jnp.dot(state.critic_params.weights, obs) + state.critic_params.bias

    @functools.partial(jax.jit, static_argnums=(0,))
    def squash_pre_tanh_action(
        self,
        pre_tanh_action: Array,
    ) -> Float[Array, " action_dim"]:
        """Apply the exact finite-precision affine-tanh action transform."""
        return _squash_action(pre_tanh_action, self._action_low, self._action_high)

    def _sample_with_key(
        self,
        state: ContinuousAverageRewardActorCriticState,
        observation: Array,
        key: Array,
    ) -> tuple[SquashedGaussianPolicySample, Array]:
        """Use one typed normal draw and return its successor key."""
        next_key, draw_key = jr.split(key)
        obs = jnp.asarray(observation, dtype=jnp.float32)
        mean, target_std, behavior_std = self.behavior_policy_params(state, obs)
        mean = jax.lax.optimization_barrier(mean)
        target_std = jax.lax.optimization_barrier(target_std)
        behavior_std = jax.lax.optimization_barrier(behavior_std)
        standard_normal = jr.normal(draw_key, shape=(self._config.action_dim,), dtype=jnp.float32)
        # This barrier makes the cached float32 latent the exact owner of the
        # action, densities, and score.  Without it XLA may fuse the affine
        # Gaussian draw into downstream expressions and retain a different
        # intermediate than the stored ``pre_tanh_action`` leaf.
        scaled_draw = jax.lax.optimization_barrier(behavior_std * standard_normal)
        pre_tanh_action = jax.lax.optimization_barrier(mean + scaled_draw)
        action = self.squash_pre_tanh_action(pre_tanh_action)
        target_log_density = transformed_diagonal_gaussian_log_density(
            pre_tanh_action,
            mean,
            target_std,
            self._action_low,
            self._action_high,
        )
        behavior_log_density = transformed_diagonal_gaussian_log_density(
            pre_tanh_action,
            mean,
            behavior_std,
            self._action_low,
            self._action_high,
        )
        # The affine-tanh Jacobian is common and cancels analytically.  Derive
        # the ratio before adding that potentially huge term so saturated
        # actions retain the exact finite-precision likelihood correction.
        ratio = diagonal_gaussian_target_behavior_ratio(
            pre_tanh_action, mean, target_std, behavior_std
        )
        return SquashedGaussianPolicySample(
            observation=obs,
            pre_tanh_action=pre_tanh_action,
            action=action,
            target_mean=mean,
            target_std=target_std,
            behavior_std=behavior_std,
            target_log_density=target_log_density,
            behavior_log_density=behavior_log_density,
            target_behavior_ratio=ratio,
            valid=jnp.asarray(True),
        ), next_key

    @functools.partial(jax.jit, static_argnums=(0,))
    def sample_policy(
        self,
        state: ContinuousAverageRewardActorCriticState,
        observation: Array,
    ) -> tuple[SquashedGaussianPolicySample, Array]:
        """Sample one behavior action and log exact target/behavior densities."""
        return self._sample_with_key(state, observation, state.rng_key)

    @functools.partial(jax.jit, static_argnums=(0,))
    def target_policy_score(
        self,
        state: ContinuousAverageRewardActorCriticState,
        sample: SquashedGaussianPolicySample,
    ) -> ContinuousAverageRewardActorTrace:
        """Return the target log-density score for the fixed cached decision."""
        mean, target_std = self.target_policy_params(state, sample.observation)
        normalised = (sample.pre_tanh_action - mean) / target_std
        mean_bias = normalised / target_std
        return ContinuousAverageRewardActorTrace(
            mean_weights=mean_bias[:, None] * sample.observation[None, :],
            mean_bias=mean_bias,
            log_std=jnp.square(normalised) - 1.0,
        )

    def _cached_decision_valid(
        self,
        state: ContinuousAverageRewardActorCriticState,
    ) -> Array:
        sample = state.last_sample
        mean, target_std, behavior_std = self.behavior_policy_params(state, sample.observation)
        action = _squash_action(sample.pre_tanh_action, self._action_low, self._action_high)
        target_log_density = transformed_diagonal_gaussian_log_density(
            sample.pre_tanh_action,
            mean,
            target_std,
            self._action_low,
            self._action_high,
        )
        behavior_log_density = transformed_diagonal_gaussian_log_density(
            sample.pre_tanh_action,
            mean,
            behavior_std,
            self._action_low,
            self._action_high,
        )
        ratio = diagonal_gaussian_target_behavior_ratio(
            sample.pre_tanh_action, mean, target_std, behavior_std
        )
        fields_match = (
            _float_bits_equal(sample.target_mean, mean)
            & _float_bits_equal(sample.target_std, target_std)
            & _float_bits_equal(sample.behavior_std, behavior_std)
            & _float_bits_equal(sample.action, action)
            & jnp.all(
                float32_ulp_distance(sample.target_log_density, target_log_density)
                <= TRANSFORMED_LOG_DENSITY_MAX_ULPS
            )
            & jnp.all(
                float32_ulp_distance(sample.behavior_log_density, behavior_log_density)
                <= TRANSFORMED_LOG_DENSITY_MAX_ULPS
            )
            & _float_bits_equal(sample.target_behavior_ratio, ratio)
        )
        bounds_valid = jnp.all(sample.action >= self._action_low) & jnp.all(
            sample.action <= self._action_high
        )
        return (
            sample.valid
            & _all_finite(sample)
            & fields_match
            & bounds_valid
            & jnp.all(sample.target_std > 0.0)
            & jnp.all(sample.behavior_std >= sample.target_std)
            & (sample.target_behavior_ratio >= 0.0)
        )

    def _static_shapes_valid(
        self,
        state: ContinuousAverageRewardActorCriticState,
    ) -> bool:
        """Check all static state shapes before tracing arithmetic branches."""
        action_dim = self._config.action_dim
        mean_weights_shape = state.actor_params.mean_weights.shape
        if len(mean_weights_shape) != 2 or mean_weights_shape[0] != action_dim:
            return False
        feature_dim = mean_weights_shape[1]
        return (
            feature_dim > 0
            and state.actor_params.mean_bias.shape == (action_dim,)
            and state.actor_params.log_std.shape == (action_dim,)
            and state.critic_params.weights.shape == (feature_dim,)
            and state.critic_params.bias.shape == ()
            and state.actor_trace.mean_weights.shape == (action_dim, feature_dim)
            and state.actor_trace.mean_bias.shape == (action_dim,)
            and state.actor_trace.log_std.shape == (action_dim,)
            and state.critic_trace.weights.shape == (feature_dim,)
            and state.critic_trace.bias.shape == ()
            and state.actor_optimizer_state.mean_weights.step_size.shape == ()
            and state.actor_optimizer_state.mean_bias.step_size.shape == ()
            and state.actor_optimizer_state.log_std.step_size.shape == ()
            and state.critic_optimizer_state.weights.step_size.shape == ()
            and state.critic_optimizer_state.bias.step_size.shape == ()
            and state.average_reward_optimizer_state.step_size.shape == ()
            and state.average_reward.shape == ()
            and state.last_sample.observation.shape == (feature_dim,)
            and state.last_sample.pre_tanh_action.shape == (action_dim,)
            and state.last_sample.action.shape == (action_dim,)
            and state.last_sample.target_mean.shape == (action_dim,)
            and state.last_sample.target_std.shape == (action_dim,)
            and state.last_sample.behavior_std.shape == (action_dim,)
            and state.last_sample.target_log_density.shape == ()
            and state.last_sample.behavior_log_density.shape == ()
            and state.last_sample.target_behavior_ratio.shape == ()
            and state.last_sample.valid.shape == ()
            and state.decision_count.shape == ()
            and state.update_count.shape == ()
        )

    def _storage_valid(self, state: ContinuousAverageRewardActorCriticState) -> Array:
        expected_actor_step = jnp.asarray(self._config.actor_step_size, dtype=jnp.float32)
        expected_critic_step = jnp.asarray(self._config.critic_step_size, dtype=jnp.float32)
        expected_average_step = jnp.asarray(
            self._config.average_reward_step_size, dtype=jnp.float32
        )
        optimizer_valid = (
            _float_bits_equal(
                state.actor_optimizer_state.mean_weights.step_size, expected_actor_step
            )
            & _float_bits_equal(
                state.actor_optimizer_state.mean_bias.step_size, expected_actor_step
            )
            & _float_bits_equal(state.actor_optimizer_state.log_std.step_size, expected_actor_step)
            & _float_bits_equal(
                state.critic_optimizer_state.weights.step_size, expected_critic_step
            )
            & _float_bits_equal(state.critic_optimizer_state.bias.step_size, expected_critic_step)
            & _float_bits_equal(
                state.average_reward_optimizer_state.step_size, expected_average_step
            )
        )
        log_std_valid = jnp.all(
            state.actor_params.log_std >= self._config.target_log_std_min
        ) & jnp.all(state.actor_params.log_std <= self._config.target_log_std_max)
        key_data = jr.key_data(state.rng_key)
        key_valid = jnp.asarray(
            state.rng_key.shape == ()
            and jax.dtypes.issubdtype(state.rng_key.dtype, jax.dtypes.prng_key)
        )
        counters_valid = (
            (state.decision_count >= 0)
            & (state.update_count >= 0)
            & (state.update_count <= self._config.max_updates)
        )
        return (
            _all_finite(
                (
                    state.actor_params,
                    state.critic_params,
                    state.actor_trace,
                    state.critic_trace,
                    state.actor_optimizer_state,
                    state.critic_optimizer_state,
                    state.average_reward_optimizer_state,
                    state.average_reward,
                    state.last_sample,
                )
            )
            & jnp.all(jnp.isfinite(key_data))
            & key_valid
            & optimizer_valid
            & log_std_valid
            & counters_valid
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def start(
        self,
        state: ContinuousAverageRewardActorCriticState,
        observation: Array,
    ) -> ContinuousAverageRewardActorCriticStartResult:
        """Atomically draw and cache a continuing decision."""
        obs = jnp.asarray(observation, dtype=jnp.float32)
        static_shapes_valid = self._static_shapes_valid(state)
        feature_dim = (
            state.actor_params.mean_weights.shape[1]
            if len(state.actor_params.mean_weights.shape) == 2
            else -1
        )
        if not static_shapes_valid or obs.shape != (feature_dim,):
            return ContinuousAverageRewardActorCriticStartResult(
                state=state,
                sample=state.last_sample,
                action=state.last_sample.action,
                accepted=jnp.asarray(False),
            )
        valid = self._storage_valid(state) & jnp.all(jnp.isfinite(obs))

        def accept(_: None) -> ContinuousAverageRewardActorCriticStartResult:
            sample, key = self._sample_with_key(state, obs, state.rng_key)
            candidate = state.replace(
                last_sample=sample,
                rng_key=key,
                decision_count=_saturating_int32_increment(state.decision_count),
            )
            candidate_valid = self._storage_valid(candidate) & self._cached_decision_valid(
                candidate
            )

            def commit(_: None) -> ContinuousAverageRewardActorCriticStartResult:
                return ContinuousAverageRewardActorCriticStartResult(
                    state=candidate,
                    sample=sample,
                    action=sample.action,
                    accepted=jnp.asarray(True),
                )

            def roll_back(_: None) -> ContinuousAverageRewardActorCriticStartResult:
                return ContinuousAverageRewardActorCriticStartResult(
                    state=state,
                    sample=state.last_sample,
                    action=state.last_sample.action,
                    accepted=jnp.asarray(False),
                )

            return cast(
                ContinuousAverageRewardActorCriticStartResult,
                jax.lax.cond(candidate_valid, commit, roll_back, operand=None),
            )

        def reject(_: None) -> ContinuousAverageRewardActorCriticStartResult:
            return ContinuousAverageRewardActorCriticStartResult(
                state=state,
                sample=state.last_sample,
                action=state.last_sample.action,
                accepted=jnp.asarray(False),
            )

        return cast(
            ContinuousAverageRewardActorCriticStartResult,
            jax.lax.cond(valid, accept, reject, operand=None),
        )

    def _rollback_update(
        self,
        state: ContinuousAverageRewardActorCriticState,
        *,
        state_valid: Array,
        input_valid: Array,
        cached_decision_valid: Array,
        capacity_available: Array,
        candidate_finite: Array,
        value: Array | None = None,
        next_value: Array | None = None,
        td_error: Array | None = None,
    ) -> ContinuousAverageRewardActorCriticUpdateResult:
        zero = jnp.asarray(0.0, dtype=jnp.float32)
        ratio = jnp.where(
            state.last_sample.valid,
            state.last_sample.target_behavior_ratio,
            zero,
        )
        return ContinuousAverageRewardActorCriticUpdateResult(
            state=state,
            sample=state.last_sample,
            action=state.last_sample.action,
            value=zero if value is None else jnp.nan_to_num(value),
            next_value=zero if next_value is None else jnp.nan_to_num(next_value),
            td_error=zero if td_error is None else jnp.nan_to_num(td_error),
            average_reward=state.average_reward,
            accepted=jnp.asarray(False),
            diagnostics=ContinuousAverageRewardActorCriticDiagnostics(
                state_valid=state_valid,
                input_valid=input_valid,
                cached_decision_valid=cached_decision_valid,
                capacity_available=capacity_available,
                candidate_finite=candidate_finite,
                target_behavior_ratio=ratio,
            ),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def update(
        self,
        state: ContinuousAverageRewardActorCriticState,
        reward: Array,
        next_observation: Array,
    ) -> ContinuousAverageRewardActorCriticUpdateResult:
        """Apply one atomic continuing differential actor-critic transition."""
        reward_value = jnp.asarray(reward, dtype=jnp.float32)
        next_obs = jnp.asarray(next_observation, dtype=jnp.float32)
        static_shapes_valid = self._static_shapes_valid(state)
        feature_dim = (
            state.actor_params.mean_weights.shape[1]
            if len(state.actor_params.mean_weights.shape) == 2
            else -1
        )
        if not static_shapes_valid:
            return self._rollback_update(
                state,
                state_valid=jnp.asarray(False),
                input_valid=jnp.asarray(False),
                cached_decision_valid=jnp.asarray(False),
                capacity_available=jnp.asarray(False),
                candidate_finite=jnp.asarray(False),
            )
        if reward_value.shape != () or next_obs.shape != (feature_dim,):
            state_valid_for_shape_rejection = self._storage_valid(state)
            cached_valid_for_shape_rejection = self._cached_decision_valid(state)
            return self._rollback_update(
                state,
                state_valid=state_valid_for_shape_rejection,
                input_valid=jnp.asarray(False),
                cached_decision_valid=cached_valid_for_shape_rejection,
                capacity_available=state.update_count < self._config.max_updates,
                candidate_finite=jnp.asarray(False),
            )
        state_valid = self._storage_valid(state)
        input_valid = jnp.all(jnp.isfinite(reward_value)) & jnp.all(jnp.isfinite(next_obs))
        cached_valid = self._cached_decision_valid(state)
        capacity_available = state.update_count < self._config.max_updates
        base_valid = state_valid & input_valid & cached_valid & capacity_available

        def reject_base(_: None) -> ContinuousAverageRewardActorCriticUpdateResult:
            return self._rollback_update(
                state,
                state_valid=state_valid,
                input_valid=input_valid,
                cached_decision_valid=cached_valid,
                capacity_available=capacity_available,
                candidate_finite=jnp.asarray(False),
            )

        def form_candidate(_: None) -> ContinuousAverageRewardActorCriticUpdateResult:
            sample = state.last_sample
            old_value = self.value(state, sample.observation)
            successor_value = self.value(state, next_obs)
            td_error = reward_value - state.average_reward + successor_value - old_value
            score = self.target_policy_score(state, sample)
            correction = sample.target_behavior_ratio
            actor_trace = ContinuousAverageRewardActorTrace(
                mean_weights=correction
                * (
                    self._config.actor_trace_lambda * state.actor_trace.mean_weights
                    + score.mean_weights
                ),
                mean_bias=correction
                * (self._config.actor_trace_lambda * state.actor_trace.mean_bias + score.mean_bias),
                log_std=correction
                * (self._config.actor_trace_lambda * state.actor_trace.log_std + score.log_std),
            )
            critic_trace = ContinuousAverageRewardCriticTrace(
                weights=(
                    self._config.critic_trace_lambda * state.critic_trace.weights
                    + sample.observation
                ),
                bias=self._config.critic_trace_lambda * state.critic_trace.bias + 1.0,
            )
            raw_mean_weights_step, actor_mean_weights_optimizer = (
                self._actor_optimizer.update_from_gradient(
                    state.actor_optimizer_state.mean_weights,
                    actor_trace.mean_weights,
                    error=td_error,
                )
            )
            raw_mean_bias_step, actor_mean_bias_optimizer = (
                self._actor_optimizer.update_from_gradient(
                    state.actor_optimizer_state.mean_bias,
                    actor_trace.mean_bias,
                    error=td_error,
                )
            )
            raw_log_std_step, actor_log_std_optimizer = self._actor_optimizer.update_from_gradient(
                state.actor_optimizer_state.log_std,
                actor_trace.log_std,
                error=td_error,
            )
            raw_critic_weights_step, critic_weights_optimizer = (
                self._critic_optimizer.update_from_gradient(
                    state.critic_optimizer_state.weights,
                    critic_trace.weights,
                    error=td_error,
                )
            )
            raw_critic_bias_step, critic_bias_optimizer = (
                self._critic_optimizer.update_from_gradient(
                    state.critic_optimizer_state.bias,
                    critic_trace.bias,
                    error=td_error,
                )
            )
            raw_average_reward_step, average_reward_optimizer = (
                self._average_reward_optimizer.update_from_gradient(
                    state.average_reward_optimizer_state,
                    jnp.asarray(1.0, dtype=jnp.float32),
                    error=td_error,
                )
            )
            mean_weights_step = td_error * raw_mean_weights_step
            mean_bias_step = td_error * raw_mean_bias_step
            log_std_step = td_error * raw_log_std_step
            critic_weights_step = td_error * raw_critic_weights_step
            critic_bias_step = td_error * raw_critic_bias_step
            average_reward_step = td_error * raw_average_reward_step
            raw_mean_weights = state.actor_params.mean_weights + mean_weights_step
            raw_mean_bias = state.actor_params.mean_bias + mean_bias_step
            raw_log_std = state.actor_params.log_std + log_std_step
            raw_critic_weights = state.critic_params.weights + critic_weights_step
            raw_critic_bias = state.critic_params.bias + critic_bias_step
            raw_average_reward = state.average_reward + average_reward_step
            candidate_finite = _all_finite(
                (
                    td_error,
                    actor_trace,
                    critic_trace,
                    mean_weights_step,
                    mean_bias_step,
                    log_std_step,
                    critic_weights_step,
                    critic_bias_step,
                    average_reward_step,
                    raw_mean_weights,
                    raw_mean_bias,
                    raw_log_std,
                    raw_critic_weights,
                    raw_critic_bias,
                    raw_average_reward,
                    actor_mean_weights_optimizer,
                    actor_mean_bias_optimizer,
                    actor_log_std_optimizer,
                    critic_weights_optimizer,
                    critic_bias_optimizer,
                    average_reward_optimizer,
                )
            )

            def reject_candidate(_: None) -> ContinuousAverageRewardActorCriticUpdateResult:
                return self._rollback_update(
                    state,
                    state_valid=state_valid,
                    input_valid=input_valid,
                    cached_decision_valid=cached_valid,
                    capacity_available=capacity_available,
                    candidate_finite=jnp.asarray(False),
                    value=old_value,
                    next_value=successor_value,
                    td_error=td_error,
                )

            def commit_candidate(_: None) -> ContinuousAverageRewardActorCriticUpdateResult:
                actor_params = ContinuousAverageRewardActorParameters(
                    mean_weights=raw_mean_weights,
                    mean_bias=raw_mean_bias,
                    log_std=jnp.clip(
                        raw_log_std,
                        self._config.target_log_std_min,
                        self._config.target_log_std_max,
                    ),
                )
                critic_params = ContinuousAverageRewardCriticParameters(
                    weights=raw_critic_weights,
                    bias=raw_critic_bias,
                )
                actor_optimizer_state = ContinuousAverageRewardActorOptimizerState(
                    mean_weights=actor_mean_weights_optimizer,
                    mean_bias=actor_mean_bias_optimizer,
                    log_std=actor_log_std_optimizer,
                )
                critic_optimizer_state = ContinuousAverageRewardCriticOptimizerState(
                    weights=critic_weights_optimizer,
                    bias=critic_bias_optimizer,
                )
                committed = state.replace(
                    actor_params=actor_params,
                    critic_params=critic_params,
                    actor_trace=actor_trace,
                    critic_trace=critic_trace,
                    actor_optimizer_state=actor_optimizer_state,
                    critic_optimizer_state=critic_optimizer_state,
                    average_reward_optimizer_state=average_reward_optimizer,
                    average_reward=raw_average_reward,
                    update_count=_saturating_int32_increment(state.update_count),
                )
                # This is the only successor draw.  It uses the original stored
                # RNG key but the fully committed actor parameters.
                next_sample, key = self._sample_with_key(committed, next_obs, committed.rng_key)
                completed = committed.replace(
                    last_sample=next_sample,
                    rng_key=key,
                    decision_count=_saturating_int32_increment(state.decision_count),
                )
                successor_valid = (
                    self._storage_valid(completed)
                    & self._cached_decision_valid(completed)
                    & _all_finite(next_sample)
                    & jnp.all(next_sample.action >= self._action_low)
                    & jnp.all(next_sample.action <= self._action_high)
                    & jnp.all(next_sample.target_std > 0.0)
                    & jnp.all(next_sample.behavior_std >= next_sample.target_std)
                    & (next_sample.target_behavior_ratio >= 0.0)
                )

                def finish(_: None) -> ContinuousAverageRewardActorCriticUpdateResult:
                    return ContinuousAverageRewardActorCriticUpdateResult(
                        state=completed,
                        sample=next_sample,
                        action=next_sample.action,
                        value=old_value,
                        next_value=successor_value,
                        td_error=td_error,
                        average_reward=raw_average_reward,
                        accepted=jnp.asarray(True),
                        diagnostics=ContinuousAverageRewardActorCriticDiagnostics(
                            state_valid=state_valid,
                            input_valid=input_valid,
                            cached_decision_valid=cached_valid,
                            capacity_available=capacity_available,
                            candidate_finite=jnp.asarray(True),
                            target_behavior_ratio=sample.target_behavior_ratio,
                        ),
                    )

                def reject_successor(_: None) -> ContinuousAverageRewardActorCriticUpdateResult:
                    return self._rollback_update(
                        state,
                        state_valid=state_valid,
                        input_valid=input_valid,
                        cached_decision_valid=cached_valid,
                        capacity_available=capacity_available,
                        candidate_finite=jnp.asarray(False),
                        value=old_value,
                        next_value=successor_value,
                        td_error=td_error,
                    )

                return cast(
                    ContinuousAverageRewardActorCriticUpdateResult,
                    jax.lax.cond(successor_valid, finish, reject_successor, operand=None),
                )

            return cast(
                ContinuousAverageRewardActorCriticUpdateResult,
                jax.lax.cond(candidate_finite, commit_candidate, reject_candidate, operand=None),
            )

        return cast(
            ContinuousAverageRewardActorCriticUpdateResult,
            jax.lax.cond(base_valid, form_candidate, reject_base, operand=None),
        )

    def resource_budget(self, feature_dim: int) -> ContinuousAverageRewardActorCriticResourceBudget:
        """Return exact persistent-state bytes and declared mechanism scope."""
        feature_dim = _positive_int(feature_dim, name="feature_dim")
        state = self.init(feature_dim, jr.key(0))
        state_nbytes = 0
        for leaf in jax.tree_util.tree_leaves(state):
            if jax.dtypes.issubdtype(leaf.dtype, jax.dtypes.prng_key):
                state_nbytes += int(np.asarray(jr.key_data(leaf)).nbytes)
            else:
                state_nbytes += int(np.asarray(leaf).nbytes)
        action_dim = self._config.action_dim
        trainable = action_dim * feature_dim + action_dim + action_dim + feature_dim + 1 + 1
        return ContinuousAverageRewardActorCriticResourceBudget(
            feature_dim=feature_dim,
            action_dim=action_dim,
            trainable_float32_scalars=trainable,
            state_nbytes=state_nbytes,
            max_updates=self._config.max_updates,
        )

    def checkpoint_payload(self, state: ContinuousAverageRewardActorCriticState) -> dict[str, Any]:
        """Serialize configuration and state to a strict JSON-compatible payload."""
        if not bool(np.asarray(self._storage_valid(state))):
            raise ValueError("state must be finite and structurally valid")
        if bool(np.asarray(state.last_sample.valid)) and not bool(
            np.asarray(self._cached_decision_valid(state))
        ):
            raise ValueError("cached decision is invalid")

        def array(value: Array) -> Any:
            return np.asarray(value).tolist()

        return {
            "schema_version": _CHECKPOINT_SCHEMA,
            "agent": self.to_config(),
            "state": {
                "actor_params": {
                    "mean_weights": array(state.actor_params.mean_weights),
                    "mean_bias": array(state.actor_params.mean_bias),
                    "log_std": array(state.actor_params.log_std),
                },
                "critic_params": {
                    "weights": array(state.critic_params.weights),
                    "bias": array(state.critic_params.bias),
                },
                "actor_trace": {
                    "mean_weights": array(state.actor_trace.mean_weights),
                    "mean_bias": array(state.actor_trace.mean_bias),
                    "log_std": array(state.actor_trace.log_std),
                },
                "critic_trace": {
                    "weights": array(state.critic_trace.weights),
                    "bias": array(state.critic_trace.bias),
                },
                "actor_optimizer_state": {
                    "mean_weights": array(state.actor_optimizer_state.mean_weights.step_size),
                    "mean_bias": array(state.actor_optimizer_state.mean_bias.step_size),
                    "log_std": array(state.actor_optimizer_state.log_std.step_size),
                },
                "critic_optimizer_state": {
                    "weights": array(state.critic_optimizer_state.weights.step_size),
                    "bias": array(state.critic_optimizer_state.bias.step_size),
                },
                "average_reward_optimizer_state": array(
                    state.average_reward_optimizer_state.step_size
                ),
                "average_reward": array(state.average_reward),
                "last_sample": {
                    "observation": array(state.last_sample.observation),
                    "pre_tanh_action": array(state.last_sample.pre_tanh_action),
                    "action": array(state.last_sample.action),
                    "target_mean": array(state.last_sample.target_mean),
                    "target_std": array(state.last_sample.target_std),
                    "behavior_std": array(state.last_sample.behavior_std),
                    "target_log_density": array(state.last_sample.target_log_density),
                    "behavior_log_density": array(state.last_sample.behavior_log_density),
                    "target_behavior_ratio": array(state.last_sample.target_behavior_ratio),
                    "valid": bool(np.asarray(state.last_sample.valid)),
                },
                "rng_key_data": array(jr.key_data(state.rng_key)),
                "decision_count": int(np.asarray(state.decision_count)),
                "update_count": int(np.asarray(state.update_count)),
            },
        }

    @classmethod
    def from_checkpoint_payload(
        cls, payload: Mapping[str, Any]
    ) -> tuple[
        ContinuousAverageRewardActorCriticAgent,
        ContinuousAverageRewardActorCriticState,
    ]:
        """Restore a strict JSON checkpoint and reject inconsistent caches."""
        if set(payload) != {"schema_version", "agent", "state"}:
            raise ValueError(
                "checkpoint fields must be exactly ['agent', 'schema_version', 'state']"
            )
        if payload["schema_version"] != _CHECKPOINT_SCHEMA:
            raise ValueError(f"schema_version must be {_CHECKPOINT_SCHEMA!r}")
        agent_payload = payload["agent"]
        state_payload = payload["state"]
        if not isinstance(agent_payload, Mapping) or not isinstance(state_payload, Mapping):
            raise ValueError("agent and state must be mappings")
        agent = cls.from_config(agent_payload)
        expected_state_fields = {
            "actor_params",
            "critic_params",
            "actor_trace",
            "critic_trace",
            "actor_optimizer_state",
            "critic_optimizer_state",
            "average_reward_optimizer_state",
            "average_reward",
            "last_sample",
            "rng_key_data",
            "decision_count",
            "update_count",
        }
        if set(state_payload) != expected_state_fields:
            raise ValueError(f"state fields must be exactly {sorted(expected_state_fields)}")

        def mapping(value: object, *, name: str, expected: set[str]) -> Mapping[str, Any]:
            if not isinstance(value, Mapping) or set(value) != expected:
                raise ValueError(f"{name} fields must be exactly {sorted(expected)}")
            return cast(Mapping[str, Any], value)

        actor_params_payload = mapping(
            state_payload["actor_params"],
            name="actor_params",
            expected={"mean_weights", "mean_bias", "log_std"},
        )
        mean_weights_raw = actor_params_payload["mean_weights"]
        if not isinstance(mean_weights_raw, Sequence) or isinstance(mean_weights_raw, (str, bytes)):
            raise ValueError("mean_weights must be a rank-two JSON array")
        if len(mean_weights_raw) != agent.config.action_dim or not mean_weights_raw:
            raise ValueError("mean_weights action dimension is invalid")
        first_row = mean_weights_raw[0]
        if not isinstance(first_row, Sequence) or isinstance(first_row, (str, bytes)):
            raise ValueError("mean_weights must be a rank-two JSON array")
        feature_dim = _positive_int(len(first_row), name="feature_dim")

        def float_array(value: object, *, shape: tuple[int, ...], name: str) -> Array:
            raw = np.asarray(value, dtype=object)
            if raw.shape != shape:
                raise ValueError(f"{name} must have shape {shape}")
            for item in raw.flat:
                if isinstance(item, bool) or not isinstance(item, numbers.Real):
                    raise ValueError(f"{name} must contain JSON real numbers")
                if not math.isfinite(float(item)):
                    raise ValueError(f"{name} must be finite")
            converted = np.asarray(value, dtype=np.float32)
            if not np.all(np.isfinite(converted)):
                raise ValueError(f"{name} must be finite in float32")
            return jnp.asarray(converted, dtype=jnp.float32)

        def integer(value: object, *, name: str) -> Array:
            if isinstance(value, bool) or not isinstance(value, numbers.Integral):
                raise ValueError(f"{name} must be an integer")
            parsed = int(value)
            if parsed < 0 or parsed > _INT32_MAX:
                raise ValueError(f"{name} must be in int32 non-negative range")
            return jnp.asarray(parsed, dtype=jnp.int32)

        action_dim = agent.config.action_dim
        critic_params_payload = mapping(
            state_payload["critic_params"],
            name="critic_params",
            expected={"weights", "bias"},
        )
        actor_trace_payload = mapping(
            state_payload["actor_trace"],
            name="actor_trace",
            expected={"mean_weights", "mean_bias", "log_std"},
        )
        critic_trace_payload = mapping(
            state_payload["critic_trace"],
            name="critic_trace",
            expected={"weights", "bias"},
        )
        actor_optimizer_payload = mapping(
            state_payload["actor_optimizer_state"],
            name="actor_optimizer_state",
            expected={"mean_weights", "mean_bias", "log_std"},
        )
        critic_optimizer_payload = mapping(
            state_payload["critic_optimizer_state"],
            name="critic_optimizer_state",
            expected={"weights", "bias"},
        )
        sample_payload = mapping(
            state_payload["last_sample"],
            name="last_sample",
            expected={
                "observation",
                "pre_tanh_action",
                "action",
                "target_mean",
                "target_std",
                "behavior_std",
                "target_log_density",
                "behavior_log_density",
                "target_behavior_ratio",
                "valid",
            },
        )
        valid_raw = sample_payload["valid"]
        if not isinstance(valid_raw, bool):
            raise ValueError("last_sample.valid must be a JSON boolean")
        key_raw = np.asarray(state_payload["rng_key_data"], dtype=object)
        if key_raw.shape != (2,):
            raise ValueError("rng_key_data must have shape (2,)")
        key_values: list[int] = []
        for item in key_raw.flat:
            if isinstance(item, bool) or not isinstance(item, numbers.Integral):
                raise ValueError("rng_key_data must contain uint32 integers")
            parsed = int(item)
            if parsed < 0 or parsed > int(np.iinfo(np.uint32).max):
                raise ValueError("rng_key_data must contain uint32 integers")
            key_values.append(parsed)
        state = ContinuousAverageRewardActorCriticState(
            actor_params=ContinuousAverageRewardActorParameters(
                mean_weights=float_array(
                    actor_params_payload["mean_weights"],
                    shape=(action_dim, feature_dim),
                    name="actor_params.mean_weights",
                ),
                mean_bias=float_array(
                    actor_params_payload["mean_bias"],
                    shape=(action_dim,),
                    name="actor_params.mean_bias",
                ),
                log_std=float_array(
                    actor_params_payload["log_std"],
                    shape=(action_dim,),
                    name="actor_params.log_std",
                ),
            ),
            critic_params=ContinuousAverageRewardCriticParameters(
                weights=float_array(
                    critic_params_payload["weights"],
                    shape=(feature_dim,),
                    name="critic_params.weights",
                ),
                bias=float_array(
                    critic_params_payload["bias"], shape=(), name="critic_params.bias"
                ),
            ),
            actor_trace=ContinuousAverageRewardActorTrace(
                mean_weights=float_array(
                    actor_trace_payload["mean_weights"],
                    shape=(action_dim, feature_dim),
                    name="actor_trace.mean_weights",
                ),
                mean_bias=float_array(
                    actor_trace_payload["mean_bias"],
                    shape=(action_dim,),
                    name="actor_trace.mean_bias",
                ),
                log_std=float_array(
                    actor_trace_payload["log_std"],
                    shape=(action_dim,),
                    name="actor_trace.log_std",
                ),
            ),
            critic_trace=ContinuousAverageRewardCriticTrace(
                weights=float_array(
                    critic_trace_payload["weights"],
                    shape=(feature_dim,),
                    name="critic_trace.weights",
                ),
                bias=float_array(critic_trace_payload["bias"], shape=(), name="critic_trace.bias"),
            ),
            actor_optimizer_state=ContinuousAverageRewardActorOptimizerState(
                mean_weights=LMSState(
                    step_size=float_array(
                        actor_optimizer_payload["mean_weights"],
                        shape=(),
                        name="actor_optimizer_state.mean_weights",
                    )
                ),
                mean_bias=LMSState(
                    step_size=float_array(
                        actor_optimizer_payload["mean_bias"],
                        shape=(),
                        name="actor_optimizer_state.mean_bias",
                    )
                ),
                log_std=LMSState(
                    step_size=float_array(
                        actor_optimizer_payload["log_std"],
                        shape=(),
                        name="actor_optimizer_state.log_std",
                    )
                ),
            ),
            critic_optimizer_state=ContinuousAverageRewardCriticOptimizerState(
                weights=LMSState(
                    step_size=float_array(
                        critic_optimizer_payload["weights"],
                        shape=(),
                        name="critic_optimizer_state.weights",
                    )
                ),
                bias=LMSState(
                    step_size=float_array(
                        critic_optimizer_payload["bias"],
                        shape=(),
                        name="critic_optimizer_state.bias",
                    )
                ),
            ),
            average_reward_optimizer_state=LMSState(
                step_size=float_array(
                    state_payload["average_reward_optimizer_state"],
                    shape=(),
                    name="average_reward_optimizer_state",
                )
            ),
            average_reward=float_array(
                state_payload["average_reward"], shape=(), name="average_reward"
            ),
            last_sample=SquashedGaussianPolicySample(
                observation=float_array(
                    sample_payload["observation"],
                    shape=(feature_dim,),
                    name="last_sample.observation",
                ),
                pre_tanh_action=float_array(
                    sample_payload["pre_tanh_action"],
                    shape=(action_dim,),
                    name="last_sample.pre_tanh_action",
                ),
                action=float_array(
                    sample_payload["action"], shape=(action_dim,), name="last_sample.action"
                ),
                target_mean=float_array(
                    sample_payload["target_mean"],
                    shape=(action_dim,),
                    name="last_sample.target_mean",
                ),
                target_std=float_array(
                    sample_payload["target_std"],
                    shape=(action_dim,),
                    name="last_sample.target_std",
                ),
                behavior_std=float_array(
                    sample_payload["behavior_std"],
                    shape=(action_dim,),
                    name="last_sample.behavior_std",
                ),
                target_log_density=float_array(
                    sample_payload["target_log_density"],
                    shape=(),
                    name="last_sample.target_log_density",
                ),
                behavior_log_density=float_array(
                    sample_payload["behavior_log_density"],
                    shape=(),
                    name="last_sample.behavior_log_density",
                ),
                target_behavior_ratio=float_array(
                    sample_payload["target_behavior_ratio"],
                    shape=(),
                    name="last_sample.target_behavior_ratio",
                ),
                valid=jnp.asarray(valid_raw),
            ),
            rng_key=jr.wrap_key_data(jnp.asarray(key_values, dtype=jnp.uint32)),
            decision_count=integer(state_payload["decision_count"], name="decision_count"),
            update_count=integer(state_payload["update_count"], name="update_count"),
        )
        if not bool(np.asarray(agent._storage_valid(state))):
            raise ValueError("checkpoint state is not finite or structurally valid")
        if bool(np.asarray(state.last_sample.valid)) and not bool(
            np.asarray(agent._cached_decision_valid(state))
        ):
            raise ValueError("checkpoint cached decision is invalid")
        return agent, state


__all__ = [
    "TRANSFORMED_LOG_DENSITY_MAX_ULPS",
    "ContinuousAverageRewardActorCriticAgent",
    "ContinuousAverageRewardActorCriticConfig",
    "ContinuousAverageRewardActorCriticDiagnostics",
    "ContinuousAverageRewardActorCriticResourceBudget",
    "ContinuousAverageRewardActorCriticStartResult",
    "ContinuousAverageRewardActorCriticState",
    "ContinuousAverageRewardActorCriticUpdateResult",
    "ContinuousAverageRewardActorOptimizerState",
    "ContinuousAverageRewardActorParameters",
    "ContinuousAverageRewardActorTrace",
    "ContinuousAverageRewardCriticOptimizerState",
    "ContinuousAverageRewardCriticParameters",
    "ContinuousAverageRewardCriticTrace",
    "SquashedGaussianPolicySample",
    "diagonal_gaussian_target_behavior_ratio",
    "float32_ulp_distance",
    "transformed_diagonal_gaussian_log_density",
]
